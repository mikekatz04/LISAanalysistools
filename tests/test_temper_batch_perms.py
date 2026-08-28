"""``GB_TEMPER_BATCH_PERMS``: batched tempering walker permutations.

``_tempering_swap_grid`` needs one independent walker permutation per
(band, temperature) row of the swap grid. It built them in a python list
comprehension -- ``ntemps * num_bands_tempered`` separate
``cp.random.permutation`` calls per unit (24 * 1230 = 29,520 on the v6
production grid) -- and then kept only every ``units``-th band, throwing
away 8 of every 9. Across a 9-pass move that is ~265,680 draws to use
~29,520, and the cost is kernel LAUNCHES, not sorting.

The batched path draws only the KEPT rows, as one ``(n_rows, nwalkers)``
uniform matrix plus one ``argsort``.

THE GATE IS DISTRIBUTIONAL, NOT BIT-IDENTITY. An argsort of iid
continuous uniforms is a uniformly-distributed random permutation, so the
swap grid has exactly the same law -- but it consumes a different RNG
stream, so realized permutations differ call for call. This file proves
the distribution claim by sampling rather than asserting it:

* every row is a genuine permutation of ``arange(nwalkers)``;
* over many draws all ``nwalkers!`` orderings appear about equally often
  (chi-square against the uniform law over permutations);
* the per-position marginal is uniform over walkers;
* the multi-GPU group form permutes WITHIN each device block, so a swap
  pair's parent walkers still share a device;
* and, as a sensitivity control, a deliberately BIASED generator is
  rejected by the same chi-square -- so a pass means something.

These run on the bare method against a duck-typed stub: no global-fit
fixture, no gbgpu, no likelihood engine.
"""

from __future__ import annotations

import itertools
import os
import unittest
from contextlib import contextmanager

import numpy as np


@contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Stub:
    """Minimal duck type carrying only what the method reads."""

    def __init__(self, nwalkers, groups=None):
        self.nwalkers = nwalkers
        self._tempering_walker_groups = groups


def _batched(nwalkers, n_rows, groups=None):
    from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

    fn = GBSpecialStretchMove._batched_walker_permutations
    return np.asarray(fn(_Stub(nwalkers, groups), n_rows))


def _perm_counts(rows, nwalkers):
    """Frequency of each of the ``nwalkers!`` orderings."""
    index = {p: i for i, p in enumerate(itertools.permutations(range(nwalkers)))}
    counts = np.zeros(len(index), dtype=int)
    for r in rows:
        counts[index[tuple(int(x) for x in r)]] += 1
    return counts


def _chi2(counts):
    expected = counts.sum() / counts.shape[0]
    return float(((counts - expected) ** 2 / expected).sum())


class BatchedPermutationLawTest(unittest.TestCase):
    """The distribution claim, proven by sampling."""

    NWALKERS = 4          # 24 orderings
    NDRAWS = 12000        # 500 expected per ordering

    @classmethod
    def setUpClass(cls):
        np.random.seed(20260828)
        cls.rows = _batched(cls.NWALKERS, cls.NDRAWS)

    def test_every_row_is_a_permutation(self):
        self.assertEqual(self.rows.shape, (self.NDRAWS, self.NWALKERS))
        srt = np.sort(self.rows, axis=1)
        np.testing.assert_array_equal(
            srt,
            np.tile(np.arange(self.NWALKERS), (self.NDRAWS, 1)),
            err_msg="a row was not a permutation of arange(nwalkers)",
        )

    def test_all_orderings_are_equally_likely(self):
        counts = _perm_counts(self.rows, self.NWALKERS)
        self.assertEqual(counts.shape[0], 24)
        self.assertTrue(
            (counts > 0).all(), "some ordering was never produced"
        )
        # 23 dof; the 0.999 quantile is ~49.7. A uniform generator clears
        # this with room to spare; the control test below shows a biased
        # one does not.
        self.assertLess(
            _chi2(counts), 60.0,
            "batched permutations are not uniform over the 24 orderings",
        )

    def test_position_marginals_are_uniform(self):
        for pos in range(self.NWALKERS):
            counts = np.bincount(self.rows[:, pos], minlength=self.NWALKERS)
            self.assertLess(
                _chi2(counts), 20.0,
                f"walker marginal at grid position {pos} is not uniform",
            )

    def test_the_chi_square_can_actually_fail(self):
        """Sensitivity control -- a pass above must not be vacuous.

        A generator biased toward the identity ordering has to be
        REJECTED by the very same statistic, otherwise the uniformity
        test proves nothing.
        """
        rng = np.random.default_rng(7)
        biased = []
        for _ in range(self.NDRAWS):
            if rng.random() < 0.25:
                biased.append(np.arange(self.NWALKERS))     # identity bias
            else:
                biased.append(rng.permutation(self.NWALKERS))
        counts = _perm_counts(np.asarray(biased), self.NWALKERS)
        self.assertGreater(
            _chi2(counts), 60.0,
            "the chi-square threshold cannot detect a 25% identity bias -- "
            "the uniformity assertions above would be vacuous",
        )

    def test_rows_are_independent_of_each_other(self):
        """Consecutive rows must not be correlated (one draw per row)."""
        eq = (self.rows[:-1] == self.rows[1:]).all(axis=1).mean()
        # P(two independent perms of 4 coincide) = 1/24 ~ 0.0417
        self.assertLess(
            eq, 0.08, "consecutive grid rows repeat far too often"
        )


class BatchedPermutationGroupTest(unittest.TestCase):
    """Multi-GPU: permute within each device block, never across."""

    def test_walkers_stay_inside_their_device_block(self):
        groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        np.random.seed(11)
        rows = _batched(6, 400, groups=groups)
        self.assertEqual(rows.shape, (400, 6))
        for g in groups:
            block = rows[:, g]
            self.assertTrue(
                np.isin(block, g).all(),
                "a device block position held a walker from another device "
                "-- a swap pair would straddle two GPUs",
            )
            srt = np.sort(block, axis=1)
            np.testing.assert_array_equal(
                srt, np.tile(np.sort(g), (400, 1)),
                err_msg="a device block was not permuted within itself",
            )

    def test_group_blocks_are_still_uniform(self):
        groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        np.random.seed(3)
        rows = _batched(6, 6000, groups=groups)
        for g in groups:
            counts = _perm_counts(
                np.searchsorted(np.sort(g), rows[:, g]), 3
            )
            self.assertLess(
                _chi2(counts), 25.0,
                "orderings within a device block are not uniform",
            )


class SwapGridWiringTest(unittest.TestCase):
    """The knob is actually consulted, and the grid keeps its shape.

    ``_tempering_swap_grid`` is exercised against stubs for everything it
    touches, so this needs no fixture. What must hold with the knob ON:
    the returned arrays have the legacy shapes, the kept-band selection is
    unchanged, and each (band, temp) column of a row set is a permutation.
    """

    class _Sorter:
        @staticmethod
        def get_special_band_index(temp, walker, band):
            return (temp * 8 + walker) * int(1e6) + band

    def _grid(self, batch, start=0, units=3, num_bands=13, ntemps=4,
              nwalkers=8):
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialStretchMove,
        )

        class _M(_Stub):
            pass

        m = _M(nwalkers)
        m.ntemps = ntemps
        m.num_bands = num_bands
        m._batched_walker_permutations = (
            GBSpecialStretchMove._batched_walker_permutations.__get__(m)
        )
        m._permute_walkers_for_swaps = (
            GBSpecialStretchMove._permute_walkers_for_swaps.__get__(m)
        )
        fn = GBSpecialStretchMove._tempering_swap_grid
        with _env(GB_TEMPER_BATCH_PERMS="1" if batch else "0"):
            np.random.seed(5)
            return fn(m, self._Sorter(), start, units=units)

    def test_shapes_and_band_selection_match_the_legacy_path(self):
        for start in (0, 1, 2):
            on = self._grid(True, start=start)
            off = self._grid(False, start=start)
            self.assertEqual(on[4], off[4], "num_bands_unit changed")
            for k in range(4):
                self.assertEqual(
                    np.asarray(on[k]).shape, np.asarray(off[k]).shape,
                    f"grid array {k} changed shape at start={start}",
                )
            # bands and temps are deterministic -> must be IDENTICAL
            np.testing.assert_array_equal(
                np.asarray(on[0]), np.asarray(off[0]),
                err_msg="the band index grid must not change",
            )
            np.testing.assert_array_equal(
                np.asarray(on[1]), np.asarray(off[1]),
                err_msg="the temperature index grid must not change",
            )

    def test_each_band_temperature_column_is_a_permutation(self):
        band_index, temp_index, walkers, special, nbu = self._grid(True)
        walkers = np.asarray(walkers)          # (bands, nwalkers, ntemps)
        nwalkers = walkers.shape[1]
        for b in range(walkers.shape[0]):
            for t in range(walkers.shape[2]):
                np.testing.assert_array_equal(
                    np.sort(walkers[b, :, t]), np.arange(nwalkers),
                    err_msg=f"(band {b}, temp {t}) is not a permutation",
                )

    def test_special_index_is_consistent_with_the_grid(self):
        band_index, temp_index, walkers, special, nbu = self._grid(True)
        expect = (np.asarray(temp_index) * 8 + np.asarray(walkers)) * int(
            1e6
        ) + np.asarray(band_index)
        np.testing.assert_array_equal(np.asarray(special), expect)


if __name__ == "__main__":
    unittest.main()
