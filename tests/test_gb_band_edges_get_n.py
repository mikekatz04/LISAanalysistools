"""Tests for the get_N-based GB band-edge builder (free-floating widths).

Covers :func:`lisatools.globalfit.stock.erebor.gb.get_n_based_band_edges`
under the 2026-08-15 user rulings: band width = the MINIMUM
``2 * get_N(f_max_band) / Tobs`` (self-consistent in the band's own upper
edge), maximal packing, NO snapping to WDM layer boundaries (scheduling
is independent of the pixelization), optional target-count coarsening by
merging, support-based separation guard -- plus the guarantee that
``GB_BAND_EDGES_MODE=uniform`` (the default) reproduces today's
per-layer construction bit-for-bit.
"""

import unittest
import warnings

import numpy as np
from gbgpu.utils.utility import get_N

from lisatools.globalfit.stock.erebor.gb import get_n_based_band_edges
from lisatools.utils.constants import YRSID_SI

# A 3-month-like configuration (matches the live run's scale).
TOBS = YRSID_SI / 4.0
LAYER_DF = 1.393e-4
F_LO = 5.5e-4
F_HI = 2.2e-2
DF = 1.0 / TOBS


def uniform_edges(start, end, layer_df, div=1):
    """Today's per-layer construction (mirror of the init_band_structure else-branch)."""
    k_lo = int(np.ceil(start / layer_df)) * div
    k_hi = int(np.floor(end / layer_df)) * div
    return np.asarray([k * layer_df / div for k in range(k_lo, k_hi + 1)])


def band_floor(f_hi):
    """A band's own minimum width 2*get_N(f_max_band)/Tobs (Hz)."""
    return 2.0 * float(get_N(1e-30, f_hi, TOBS, oversample=4).item()) * DF


class GetNBandEdgesTest(unittest.TestCase):
    def test_monotonic_free_floating_outer_edges_verbatim(self):
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        self.assertTrue(np.all(np.diff(edges) > 0))
        # NO snapping (user ruling): outer edges are the inputs verbatim
        # -- F_LO/F_HI are deliberately not layer-aligned.
        self.assertEqual(edges[0], F_LO)
        self.assertEqual(edges[-1], F_HI)
        self.assertNotAlmostEqual((F_LO / LAYER_DF) % 1.0, 0.0)

    def test_interior_widths_at_their_own_2getn_floor(self):
        # THE width rule + maximal packing: every interior band's width
        # equals its own minimum 2*get_N(f_max_band)*df to within one
        # fixed-point tolerance -> the band count is maximal for the rule.
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        widths = np.diff(edges)
        for i in range(len(widths) - 1):  # final band absorbs the remainder
            self.assertLess(
                abs(widths[i] - band_floor(edges[i + 1])), 1e-3 * DF,
                f"band {i} widened beyond its 2*get_N floor",
            )

    def test_width_fixed_point_self_consistency(self):
        # w solves w = 2*get_N(f_lo + w)*df exactly: evaluating the floor
        # at the produced upper edge reproduces the width (the monotone
        # fixed-point iteration converged, not merely approximated).
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        for lo, hi in zip(edges[:-2], edges[1:-1]):
            self.assertAlmostEqual(
                (hi - lo) / DF, band_floor(hi) / DF, places=6
            )

    def test_widths_track_get_n(self):
        # get_N is non-decreasing in f -> widths non-decreasing, and
        # strictly wider at the top than at the bottom at this scale.
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        widths = np.diff(edges)
        self.assertGreater(widths[-2], widths[0])
        self.assertTrue(np.all(np.diff(widths[:-1]) >= -1e-15))

    def test_no_layer_snapping_anywhere(self):
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        # interior edges land wherever the walk puts them: generically
        # OFF the layer grid (assert the vast majority are unaligned).
        frac = (edges / LAYER_DF) % 1.0
        off_grid = np.minimum(frac, 1.0 - frac) > 1e-6
        self.assertGreater(np.mean(off_grid), 0.9)

    def test_target_count_coarsens_by_merging(self):
        base = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        n = len(base) - 1
        target = max(2, n // 2)
        merged = get_n_based_band_edges(
            F_LO, F_HI, TOBS, LAYER_DF, target_count=target
        )
        self.assertEqual(len(merged) - 1, target)
        # merging only: merged edges are a subset of the maximal packing
        self.assertTrue(np.all(np.isin(merged, base)))
        # a target above the maximal packing saturates (no-op)
        np.testing.assert_array_equal(
            get_n_based_band_edges(
                F_LO, F_HI, TOBS, LAYER_DF, target_count=10 * n
            ),
            base,
        )

    def test_deprecated_knobs_warn_and_do_not_change_edges(self):
        base = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = get_n_based_band_edges(
                F_LO, F_HI, TOBS, LAYER_DF,
                subband_divisor=2, min_band_layers=0.5,
            )
        cats = [str(r.message) for r in rec
                if issubclass(r.category, DeprecationWarning)]
        self.assertTrue(any("subband_divisor" in m for m in cats))
        self.assertTrue(any("min_band_layers" in m for m in cats))
        np.testing.assert_array_equal(got, base)


class UniformModeBitIdentityTest(unittest.TestCase):
    """GB_BAND_EDGES_MODE=uniform must reproduce today's edges bit-for-bit."""

    def test_wdm_uniform_edges_unchanged(self):
        # exercise the actual init_band_structure else-branch expression
        div = 1
        layer_df = LAYER_DF
        k_lo = int(np.ceil(F_LO / layer_df)) * div
        k_hi = int(np.floor(F_HI / layer_df)) * div
        expected = np.asarray(
            [k * layer_df / div for k in range(k_lo, k_hi + 1)]
        )
        got = uniform_edges(F_LO, F_HI, layer_df, div)
        np.testing.assert_array_equal(got, expected)

    def test_settings_default_mode_is_uniform(self):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        self.assertEqual(
            GBSettings.__dataclass_fields__["band_edges_mode"].default_factory(),
            "uniform",
        )


class ExplicitSlabContractTest(unittest.TestCase):
    """band_slab_Nf refuses explicit slabs below the coverage floors.

    Two floors (both computed from the band grid): the WDM-window floor
    ``max_span + 2*m_band_half_width`` (a smaller slab silently
    annihilates edge sources), and the FD-SUPPORT floor
    ``max(span_b + 2*ceil(get_N(f_hi_b)/Tobs/layer_df))`` (2026-08-15
    "strong checks" ruling: the slab must cover
    ``[f_lo - get_N(f_hi)*df, f_hi + get_N(f_hi)*df]``;
    ``GB_SLAB_SUPPORT_CHECK=0`` downgrades that one to a warning).
    """

    def _stub(self, slab_layers, edges, layer_df):
        from types import SimpleNamespace

        from lisatools.domains import WDMSettings

        wdm = WDMSettings(Nf=256, Nt=64, dt=5.0, force_backend="cpu")
        return SimpleNamespace(
            _wdm_band_slab_layers=slab_layers,
            _wdm_slab_guard_layers=1,
            _basis_settings=wdm,
            df=layer_df,
            xp=np,
            band_edges=np.asarray(edges),
        )

    @staticmethod
    def _support_floor(edges, ldf, Tobs):
        edges = np.asarray(edges, dtype=float)
        lo = np.floor(edges[:-1] / ldf + 1e-6).astype(int)
        hi = np.ceil(edges[1:] / ldf - 1e-6).astype(int)
        spans = np.maximum(1, hi - lo)
        sup = np.array([
            float(get_N(1e-30, f, Tobs, oversample=4).item()) / Tobs
            for f in edges[1:]
        ])
        return int(np.max(spans + 2 * np.ceil(sup / ldf - 1e-6).astype(int)))

    def test_contract(self):
        import os

        from lisatools.globalfit.moves.gbbands import SubBandBuffer

        ldf = 0.000390625  # the stub WDMSettings layer_df (Nf=256, dt=5)
        Tobs = self._stub(0, [0.0, ldf], ldf)._basis_settings.Tobs
        # variable-width grid: widest band 4 layers
        edges = ldf * np.array([10, 11, 12, 16, 20], dtype=float)
        window_floor = 4 + 2 * 1
        support_floor = self._support_floor(edges, ldf, Tobs)
        self.assertGreaterEqual(support_floor, window_floor)
        prop = SubBandBuffer.band_slab_Nf.fget
        # at/above the support floor: passes
        self.assertEqual(prop(self._stub(support_floor, edges, ldf)),
                         support_floor)
        # below the WDM-window floor: the original "too small" raise
        with self.assertRaisesRegex(ValueError, "too small"):
            prop(self._stub(window_floor - 1, edges, ldf))
        if support_floor > window_floor:
            # between the floors: the FD-support raise ...
            with self.assertRaisesRegex(ValueError, "does not cover"):
                prop(self._stub(window_floor, edges, ldf))
            # ... which GB_SLAB_SUPPORT_CHECK=0 downgrades to a warning
            os.environ["GB_SLAB_SUPPORT_CHECK"] = "0"
            try:
                self.assertEqual(
                    prop(self._stub(window_floor, edges, ldf)), window_floor
                )
            finally:
                os.environ.pop("GB_SLAB_SUPPORT_CHECK", None)

    def test_auto_sizing_covers_support(self):
        from lisatools.globalfit.moves.gbbands import SubBandBuffer

        ldf = 0.000390625
        edges = ldf * np.array([10, 11, 12, 16, 20], dtype=float)
        stub = self._stub(0, edges, ldf)  # 0 = AUTO
        Tobs = stub._basis_settings.Tobs
        auto = SubBandBuffer.band_slab_Nf.fget(stub)
        # auto slab >= the FD-support floor (guard adds headroom)
        self.assertGreaterEqual(auto, self._support_floor(edges, ldf, Tobs))


class FStatCacheBandGridGuardTest(unittest.TestCase):
    """check_cached_band_grid refuses grids fitted on different band edges."""

    def _epoch(self, tmp, band_edges=None, with_key=True):
        import os

        from lisatools.sampling.fstat_gridfit import GRID_BASENAME

        path = os.path.join(
            tmp, GRID_BASENAME.replace(".npz", "_peaks_stacked.npz")
        )
        payload = dict(peak_f0_mHz=np.array([1.0]), band_idx=np.array([1]))
        if with_key:
            payload["band_edges"] = np.asarray(band_edges)
        np.savez(path, **payload)
        return tmp

    def test_guard(self):
        import tempfile

        from lisatools.sampling.fstat_gridfit import check_cached_band_grid

        edges = uniform_edges(F_LO, F_HI, LAYER_DF)
        with tempfile.TemporaryDirectory() as tmp:
            # empty dir: nothing stale
            check_cached_band_grid(tmp, edges)
            # matching edges: passes
            self._epoch(tmp, band_edges=edges)
            check_cached_band_grid(tmp, edges)
            # different band count: refused loudly
            other = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
            self.assertNotEqual(len(other), len(edges))
            with self.assertRaisesRegex(ValueError, "band"):
                check_cached_band_grid(tmp, other)
        with tempfile.TemporaryDirectory() as tmp:
            # a cache with no band metadata is unverifiable: refused
            self._epoch(tmp, with_key=False)
            with self.assertRaisesRegex(ValueError, "band_edges"):
                check_cached_band_grid(tmp, edges)


if __name__ == "__main__":
    unittest.main()
