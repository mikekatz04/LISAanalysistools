"""``_cap_swap_census`` on empty selections — the r2 probe crash.

WHAT HAPPENED. The first v8-parity probe (A band, 20.07–20.76 mHz, ONE
catalogue source) died in its opening propose:

    ValueError: zero-size array to reduction operation CUPY_CUB_MAX which
    has no identity

at ``lo = xp.bincount(bflat[alive & ~upper], ...)``. CuPy's ``bincount``
sizes its output via ``max(x)`` and RAISES on a zero-size array, where
NumPy's returns zeros — so the bug is unreachable on CPU and invisible to
every numpy-backed test. The empty selection is ordinary in a sparse
band: one source means "every alive leaf in its band's upper half-cell"
(or none alive at all) happens constantly. It is also V8-BLOCKING, not
probe-specific: v8 arms the cap-temper gate (cap_divisor 1 + stagger) and
starts gb_search with an EMPTY branch, so production would crash on the
sibling ``flat[alive]`` bincount in its first GB propose.

The guard pattern already exists in-house — ``_cap_cell_counts`` guards
the identical hazard with a documented int32 zeros path ("the zero-leaf
search start hits this on GPU"). ``_cap_swap_census`` was added later and
did not replicate it. This suite pins the replication.

HOW IT TESTS. A fake ``xp`` that is numpy everywhere except ``bincount``,
which reproduces CuPy's empty-input ValueError exactly. The real
``_cap_swap_census`` runs against a stub move + sorter with the REAL flat
-index helpers bound, across every empty-selection case a sparse band can
produce. Numpy-parity is asserted on the non-empty case so the guard
cannot change healthy-path values.
"""
import types
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

NTEMPS, NWALKERS, NCELLS, NBANDS = 2, 3, 5, 4


class _CupyLikeXP:
    """numpy, except ``bincount`` crashes on empty input like CuPy."""

    def __getattr__(self, k):
        return getattr(np, k)

    @staticmethod
    def bincount(x, weights=None, minlength=0):
        if x.size == 0:
            raise ValueError(
                "zero-size array to reduction operation CUPY_CUB_MAX "
                "which has no identity")
        return np.bincount(x, weights=weights, minlength=minlength)


def _move(cells_out):
    s = types.SimpleNamespace(
        ntemps=NTEMPS, nwalkers=NWALKERS,
        num_cap_cells=NCELLS, num_bands=NBANDS,
        _cap_leaf_cap=np.full(NCELLS, 4, dtype=np.int32),
        _sorter_cap_cells=lambda bs: cells_out,
    )
    s._cap_flat_index = GBSpecialStretchMove._cap_flat_index.__get__(s)
    s._band_flat_index = GBSpecialStretchMove._band_flat_index.__get__(s)
    return s


def _sorter(n, alive, band, temp=None, walker=None):
    return types.SimpleNamespace(
        band_inds=np.asarray(band, dtype=np.int64),
        temp_inds=(np.zeros(n, dtype=np.int64) if temp is None
                   else np.asarray(temp, dtype=np.int64)),
        walker_inds=(np.zeros(n, dtype=np.int64) if walker is None
                     else np.asarray(walker, dtype=np.int64)),
        inds=np.asarray(alive, dtype=bool),
    )


CENSUS = GBSpecialStretchMove._cap_swap_census


def _run(alive, upper_mask, band=None):
    """Run the census through the CuPy-like xp.

    ``upper_mask`` chooses, per row, whether ``_sorter_cap_cells`` returns
    the band cell (lower half) or band+1 (upper half) — the K=1 geometry
    the census documents.
    """
    n = len(alive)
    band = np.zeros(n, dtype=np.int64) if band is None else np.asarray(band)
    bs = _sorter(n, alive, band)
    cells = band + np.asarray(upper_mask, dtype=np.int64)
    # get_array_module(band_sorter.band_inds) returns numpy for numpy
    # inputs, so route the census through the CuPy-like namespace by
    # wrapping band_inds in a subclass get_array_module maps to our fake.
    # Simpler and equivalent: monkeypatch the module resolver for the call.
    import lisatools.globalfit.moves.gbspecialstretch as G
    orig = G.get_array_module
    G.get_array_module = lambda a: _CupyLikeXP()
    try:
        return CENSUS(_move(cells), bs)
    finally:
        G.get_array_module = orig


class EmptySelectionTest(unittest.TestCase):
    """Each case is an ordinary sparse-band state, not a pathology."""

    def test_no_alive_sources_at_all(self):
        """The v8 production case: gb_search opens on an EMPTY branch."""
        counts, lo, hi, cap = _run(alive=[False] * 4,
                                   upper_mask=[0, 0, 1, 1])
        for a, n in ((counts, NTEMPS * NWALKERS * NCELLS),
                     (lo, NTEMPS * NWALKERS * NBANDS),
                     (hi, NTEMPS * NWALKERS * NBANDS)):
            self.assertEqual(int(np.asarray(a).shape[0]), n)
            self.assertEqual(float(np.asarray(a).sum()), 0.0)

    def test_every_alive_source_in_its_upper_half(self):
        """THE probe crash: one source, upper cell -> lo selection empty."""
        counts, lo, hi, _ = _run(alive=[True, True], upper_mask=[1, 1],
                                 band=[1, 2])
        self.assertEqual(float(np.asarray(lo).sum()), 0.0)
        self.assertEqual(float(np.asarray(hi).sum()), 2.0)
        self.assertEqual(float(np.asarray(counts).sum()), 2.0)

    def test_every_alive_source_in_its_lower_half(self):
        counts, lo, hi, _ = _run(alive=[True, True], upper_mask=[0, 0],
                                 band=[1, 2])
        self.assertEqual(float(np.asarray(hi).sum()), 0.0)
        self.assertEqual(float(np.asarray(lo).sum()), 2.0)

    def test_mixed_case_matches_plain_numpy(self):
        """The guard must not change healthy-path values."""
        alive = [True, True, False, True]
        upper = [0, 1, 0, 0]
        band = [0, 1, 2, 3]
        counts, lo, hi, cap = _run(alive, upper, band)
        # independent reference with plain numpy
        cells = np.asarray(band) + np.asarray(upper)
        t = np.zeros(4, dtype=np.int64)
        w = np.zeros(4, dtype=np.int64)
        flat = (t * NWALKERS + w) * NCELLS + cells
        bflat = (t * NWALKERS + w) * NBANDS + np.asarray(band)
        al = np.asarray(alive)
        up = cells > np.asarray(band)
        np.testing.assert_array_equal(
            np.asarray(counts),
            np.bincount(flat[al], minlength=NTEMPS * NWALKERS * NCELLS))
        np.testing.assert_array_equal(
            np.asarray(lo),
            np.bincount(bflat[al & ~up], minlength=NTEMPS * NWALKERS * NBANDS))
        np.testing.assert_array_equal(
            np.asarray(hi),
            np.bincount(bflat[al & up], minlength=NTEMPS * NWALKERS * NBANDS))
        np.testing.assert_array_equal(np.asarray(cap),
                                      np.full(NCELLS, 4, dtype=np.int32))

    def test_empty_outputs_are_integer_typed(self):
        """int zeros, not float: these feed integer occupancy arithmetic.

        (_cap_cell_counts uses int32 for its empty path for a documented
        CuPy scatter-add reason; anything integer-kind is acceptable here,
        float is not.)
        """
        counts, lo, hi, _ = _run(alive=[False] * 3, upper_mask=[0] * 3)
        for a in (counts, lo, hi):
            self.assertEqual(np.asarray(a).dtype.kind, "i", np.asarray(a).dtype)


if __name__ == "__main__":
    unittest.main()
