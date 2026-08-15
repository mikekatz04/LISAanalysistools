"""Orthogonality premise + bilinearity bookkeeping checks (default off).

Physics ruling (user, verified premise): FD inner product ~0 implies WDM
inner product ~0 even within one wavelet layer; sources with
``|df| * Tobs >> 1`` have ``<h_i|h_j> ~ 0``, so their likelihood deltas
add by bilinearity and their evaluations may run concurrently. These
tests cover the two monitors built on that premise:

* ``[GB_ORTHO_LL]`` (``GB_ORTHO_LL_CHECK=1``): per concurrent unit, the
  sum of per-buffer lnL deltas vs the realized parent-residual delta --
  exactly additive fakes pass, an injected interference term flags.
* ``[GB_ORTHO]`` (``GB_ORTHO_CHECK=1``): boundary-pair selection +
  normalized overlaps through the engine's swap-ll surface -- distant
  tones give tiny overlaps, same-bin tones flag. The engine here is a
  test stub computing REAL tone inner products (production uses the
  installed ``BandLikelihoodEngine.get_swap_ll`` kernels).
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _ortho_boundary_pairs,
    _ortho_ll_summary,
)


class OrthoLLSummaryTest(unittest.TestCase):
    def test_exactly_additive_passes(self):
        # bilinearity holds: per-cell credited deltas sum to the realized
        # residual delta on every walker.
        direct = np.array([0.3, -1.2, 4.5, 0.0])
        credited = direct.copy()
        out = _ortho_ll_summary(direct, credited, tol=1e-6)
        self.assertFalse(out["flagged"])
        self.assertEqual(out["max_abs"], 0.0)
        self.assertEqual(out["mean_abs"], 0.0)

    def test_small_discrepancy_within_tol_passes(self):
        out = _ortho_ll_summary([1.0, 2.0], [1.0 + 1e-4, 2.0], tol=0.05)
        self.assertFalse(out["flagged"])
        self.assertAlmostEqual(out["max_abs"], 1e-4)

    def test_injected_overlap_flags_worst_walker(self):
        # a cross term <h_i|h_j> leaking into walker 2's realized delta
        direct = np.array([0.5, 0.5, 0.5 + 0.2, 0.5])
        credited = np.full(4, 0.5)
        out = _ortho_ll_summary(direct, credited, tol=0.05)
        self.assertTrue(out["flagged"])
        self.assertEqual(out["worst_walker"], 2)
        self.assertAlmostEqual(out["max_abs"], 0.2)
        self.assertAlmostEqual(out["mean_abs"], 0.05)


class BoundaryPairSelectionTest(unittest.TestCase):
    def test_same_unit_cross_band_closest_first(self):
        # walker 0, unit = bands % 2 == 0: sources in bands 0/2/2/4.
        f0 = np.array([1.00, 1.99, 2.01, 2.02, 3.5, 7.0]) * 1e-3
        band = np.array([0, 0, 2, 2, 2, 4])
        walker = np.zeros(6, dtype=int)
        elig = np.ones(6, dtype=bool)
        i_idx, j_idx = _ortho_boundary_pairs(
            f0, walker, band, elig, units=2, remainder=0, max_pairs=8
        )
        # cross-band consecutive pairs: (1,2) df=2e-5 and (4,5) df=3.5e-3;
        # closest first.
        np.testing.assert_array_equal(i_idx, [1, 4])
        np.testing.assert_array_equal(j_idx, [2, 5])

    def test_other_unit_and_ineligible_rows_excluded(self):
        f0 = np.array([1.0, 1.001, 1.002, 1.003]) * 1e-3
        band = np.array([0, 1, 2, 2])  # band 1 is unit-1 under stride 2
        walker = np.zeros(4, dtype=int)
        elig = np.array([True, True, True, False])  # row 3 e.g. hot chain
        i_idx, j_idx = _ortho_boundary_pairs(
            f0, walker, band, elig, units=2, remainder=0
        )
        np.testing.assert_array_equal(i_idx, [0])
        np.testing.assert_array_equal(j_idx, [2])

    def test_pairs_never_cross_walkers(self):
        f0 = np.array([1.0, 1.0001, 1.0, 1.0001]) * 1e-3
        band = np.array([0, 2, 0, 2])
        walker = np.array([0, 1, 1, 0])  # closest-f pairs are cross-walker
        elig = np.ones(4, dtype=bool)
        i_idx, j_idx = _ortho_boundary_pairs(
            f0, walker, band, elig, units=2, remainder=0
        )
        for a, b in zip(i_idx, j_idx):
            self.assertEqual(walker[a], walker[b])

    def test_max_pairs_cap_and_empty(self):
        f0 = np.linspace(1.0, 2.0, 12) * 1e-3
        band = np.arange(12) // 2 * 2  # bands 0,0,2,2,4,4,... all unit 0
        walker = np.zeros(12, dtype=int)
        elig = np.ones(12, dtype=bool)
        i_idx, _ = _ortho_boundary_pairs(
            f0, walker, band, elig, units=2, remainder=0, max_pairs=3
        )
        self.assertEqual(len(i_idx), 3)
        # single-band unit -> no cross-band pair
        i_idx, j_idx = _ortho_boundary_pairs(
            f0, walker, np.zeros(12, int), elig, units=2, remainder=0
        )
        self.assertEqual(len(i_idx), 0)
        self.assertEqual(len(j_idx), 0)


class _ToneSwapEngine:
    """Test stub scoring REAL monochromatic-tone inner products.

    ``get_swap_ll`` mirrors the ``BandLikelihoodEngine`` call convention
    the move uses and returns ``hh_add / hh_remove / hh_cross`` computed
    from sampled cosine tones over an observation of length ``T``:
    ``<h_i|h_j> ~ 0`` for ``|df| * T >> 1`` and ``= <h|h>`` at ``df = 0``
    -- the physics premise itself, in miniature.
    """

    def __init__(self, T=1e5, n=4096):
        self.t = np.linspace(0.0, T, n, endpoint=False)
        self.calls = []

    def _tone(self, f0):
        return np.cos(2 * np.pi * f0 * self.t)

    def get_swap_ll(self, holder, params_remove_phys, params_add_phys, *,
                    data_index, noise_index, N_vals, phase_maximize,
                    waveform_kwargs):
        self.calls.append(dict(data_index=np.asarray(data_index)))
        f_add = np.asarray(params_add_phys)[:, 1]
        f_rem = np.asarray(params_remove_phys)[:, 1]
        hh_add = np.array([self._tone(f) @ self._tone(f) for f in f_add])
        hh_rem = np.array([self._tone(f) @ self._tone(f) for f in f_rem])
        hh_x = np.array([
            self._tone(fa) @ self._tone(fr) for fa, fr in zip(f_add, f_rem)
        ])
        return SimpleNamespace(
            hh_add=hh_add, hh_remove=hh_rem, hh_cross=hh_x
        )


def _premise_move(engine):
    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.name = "test"
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"
    m._likelihood_engine = engine
    m.waveform_kwargs = {}
    return m


def _sorter(f0, band, walker=None, temp=None, alive=None):
    n = len(f0)
    coords_in = np.zeros((n, 9))
    coords_in[:, 1] = f0
    return SimpleNamespace(
        inds=np.ones(n, bool) if alive is None else alive,
        temp_inds=np.zeros(n, int) if temp is None else temp,
        walker_inds=np.zeros(n, int) if walker is None else walker,
        band_inds=np.asarray(band),
        coords_in=coords_in,
        N_vals=np.full(n, 128),
    )


class OrthoPremiseCheckTest(unittest.TestCase):
    def setUp(self):
        os.environ["GB_ORTHO_CHECK"] = "1"
        self.addCleanup(os.environ.pop, "GB_ORTHO_CHECK", None)
        self.model = SimpleNamespace(analysis_container_arr=object())
        self.logger_name = "lisatools.globalfit.moves.gbspecialstretch"

    def test_off_by_default(self):
        os.environ.pop("GB_ORTHO_CHECK")
        engine = _ToneSwapEngine()
        m = _premise_move(engine)
        m._run_ortho_premise_check(
            self.model, _sorter([1e-3, 2e-3], [0, 2]), 2, 0
        )
        self.assertEqual(engine.calls, [])
        os.environ["GB_ORTHO_CHECK"] = "1"  # restore for addCleanup pop

    def test_distant_tones_tiny_overlap_no_warning(self):
        # |df| * T = 1e-4 * 1e5 = 10 >> 1 -> overlap ~ sinc scale ~ 1e-2
        # at worst; use well-separated tones so it is << tol.
        engine = _ToneSwapEngine(T=1e5)
        m = _premise_move(engine)
        sorter = _sorter([1.0e-3, 1.5e-3], [0, 2])
        with self.assertLogs(self.logger_name, level="INFO") as cm:
            m._run_ortho_premise_check(self.model, sorter, 2, 0)
        text = "\n".join(cm.output)
        self.assertIn("[GB_ORTHO test]", text)
        self.assertIn("boundary pairs", text)
        self.assertNotIn("exceeds", text)
        self.assertEqual(len(engine.calls), 1)

    def test_same_bin_tones_flagged(self):
        engine = _ToneSwapEngine()
        m = _premise_move(engine)
        sorter = _sorter([1.0e-3, 1.0e-3 + 1e-9], [0, 2])  # ~identical f
        with self.assertLogs(self.logger_name, level="WARNING") as cm:
            m._run_ortho_premise_check(self.model, sorter, 2, 0)
        self.assertIn("exceeds", "\n".join(cm.output))

    def test_hot_chain_and_other_unit_ignored(self):
        engine = _ToneSwapEngine()
        m = _premise_move(engine)
        # same-bin pair, but one row is hot (temp 1): must be excluded ->
        # no cross-band cold pair -> engine never called.
        sorter = _sorter(
            [1.0e-3, 1.0e-3], [0, 2], temp=np.array([0, 1])
        )
        with self.assertLogs(self.logger_name, level="INFO"):
            m._run_ortho_premise_check(self.model, sorter, 2, 0)
        self.assertEqual(engine.calls, [])

    def test_data_index_is_pair_walker(self):
        engine = _ToneSwapEngine()
        m = _premise_move(engine)
        sorter = _sorter(
            [1.0e-3, 1.1e-3], [0, 2], walker=np.array([3, 3])
        )
        m._run_ortho_premise_check(self.model, sorter, 2, 0)
        np.testing.assert_array_equal(engine.calls[0]["data_index"], [3])

    def test_internal_failure_never_raises(self):
        class _Boom:
            def get_swap_ll(self, *a, **k):
                raise RuntimeError("kernel unavailable")

        m = _premise_move(_Boom())
        sorter = _sorter([1.0e-3, 1.1e-3], [0, 2])
        with self.assertLogs(self.logger_name, level="WARNING") as cm:
            m._run_ortho_premise_check(self.model, sorter, 2, 0)
        self.assertIn("skipped", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main()
