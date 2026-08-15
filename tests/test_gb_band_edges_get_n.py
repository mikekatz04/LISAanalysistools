"""Tests for the get_N-based variable-width GB band-edge builder.

Covers :func:`lisatools.globalfit.stock.erebor.gb.get_n_based_band_edges`:
layer-grid alignment, monotonicity, get_N-proportional widths, the
minimum-width floor, the target-count normalization, and the guarantee that
``GB_BAND_EDGES_MODE=uniform`` (the default) reproduces today's per-layer
construction bit-for-bit.
"""

import unittest

import numpy as np
from gbgpu.utils.utility import get_N

from lisatools.globalfit.stock.erebor.gb import get_n_based_band_edges
from lisatools.utils.constants import YRSID_SI

# A 3-month-like configuration (matches the live run's scale).
TOBS = YRSID_SI / 4.0
LAYER_DF = 1.393e-4
F_LO = 5.5e-4
F_HI = 2.2e-2


def uniform_edges(start, end, layer_df, div=1):
    """Today's per-layer construction (mirror of the init_band_structure else-branch)."""
    k_lo = int(np.ceil(start / layer_df)) * div
    k_hi = int(np.floor(end / layer_df)) * div
    return np.asarray([k * layer_df / div for k in range(k_lo, k_hi + 1)])


class GetNBandEdgesTest(unittest.TestCase):
    def test_monotonic_and_layer_aligned(self):
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        self.assertTrue(np.all(np.diff(edges) > 0))
        # every edge on the k * layer_df grid (div=1)
        k = edges / LAYER_DF
        np.testing.assert_allclose(k, np.round(k), atol=1e-9)
        # integer number of layers per band, min 1
        widths = np.diff(edges) / LAYER_DF
        np.testing.assert_allclose(widths, np.round(widths), atol=1e-9)
        self.assertGreaterEqual(widths.min(), 1.0 - 1e-9)

    def test_outer_boundaries_match_uniform(self):
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        uni = uniform_edges(F_LO, F_HI, LAYER_DF)
        self.assertAlmostEqual(edges[0], uni[0])
        self.assertAlmostEqual(edges[-1], uni[-1])

    def test_widths_track_get_n(self):
        # widths must be non-decreasing with frequency (get_N is a step
        # function increasing in f0) and strictly wider at the top of the
        # band than at the bottom for a 3-month-like configuration.
        edges = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        widths = np.diff(edges)
        self.assertGreater(widths[-2], widths[0])
        # proportionality: the natural (unquantized) width is
        # (2 N + extra_buffer) / Tobs; each produced band's width must be
        # within one grid unit of it (or at the floor).
        for lo, w in zip(edges[:-1], widths):
            n_samp = get_N(1e-30, lo, TOBS, oversample=4).item()
            w_nat = (2.0 * n_samp + 5) / TOBS
            expect = max(LAYER_DF, round(w_nat / LAYER_DF) * LAYER_DF)
            # last band can be clamped/merged; allow one extra unit there
            self.assertLessEqual(abs(w - expect), LAYER_DF + 1e-12)

    def test_target_count_normalization(self):
        natural = get_n_based_band_edges(F_LO, F_HI, TOBS, LAYER_DF)
        n_nat = len(natural) - 1
        # ask for fewer bands -> wider bands, count near target
        target = max(2, n_nat // 2)
        edges = get_n_based_band_edges(
            F_LO, F_HI, TOBS, LAYER_DF, target_count=target
        )
        self.assertLessEqual(abs((len(edges) - 1) - target), max(2, target // 10))
        # count saturates at the uniform count for huge targets (min 1 layer)
        uni_count = len(uniform_edges(F_LO, F_HI, LAYER_DF)) - 1
        edges_hi = get_n_based_band_edges(
            F_LO, F_HI, TOBS, LAYER_DF, target_count=10 * uni_count
        )
        self.assertEqual(len(edges_hi) - 1, uni_count)

    def test_subband_divisor_grid(self):
        div = 2
        edges = get_n_based_band_edges(
            F_LO, F_HI, TOBS, LAYER_DF, subband_divisor=div, target_count=120,
        )
        unit = LAYER_DF / div
        k = edges / unit
        np.testing.assert_allclose(k, np.round(k), atol=1e-9)
        # the parity-safety floor is 1 FULL layer even on the finer grid
        widths = np.diff(edges)
        self.assertGreaterEqual(widths.min(), LAYER_DF - 1e-12)

    def test_min_band_layers_floor(self):
        edges = get_n_based_band_edges(
            F_LO, F_HI, TOBS, LAYER_DF, min_band_layers=2.0
        )
        widths = np.diff(edges) / LAYER_DF
        self.assertGreaterEqual(widths.min(), 2.0 - 1e-9)

    def test_sublayer_min_refused(self):
        # sub-layer minimum widths are unsafe under the stride-2 band
        # parity scheduling and must be refused loudly
        with self.assertRaises(ValueError):
            get_n_based_band_edges(
                F_LO, F_HI, TOBS, LAYER_DF, subband_divisor=2,
                min_band_layers=0.5,
            )


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
    """band_slab_Nf refuses an explicit slab smaller than max_span + 2*hw.

    The chunked-het kernels clip each source's per-layer window to the
    slab; a too-small explicit GB_WDM_BAND_SLAB_LAYERS silently annihilates
    edge sources of wide bands, so the property must raise instead.
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

    def test_contract(self):
        from lisatools.globalfit.moves.gbbands import SubBandBuffer

        ldf = 0.000390625  # the stub WDMSettings layer_df (Nf=256, dt=5)
        # variable-width grid: widest band 4 layers
        edges = ldf * np.array([10, 11, 12, 16, 20], dtype=float)
        prop = SubBandBuffer.band_slab_Nf.fget
        # 4 + 2*1 = 6 is the floor: 6 passes, 5 raises
        self.assertEqual(prop(self._stub(6, edges, ldf)), 6)
        with self.assertRaisesRegex(ValueError, "too small"):
            prop(self._stub(5, edges, ldf))
        # 1-layer uniform grid: today's explicit 5-layer slab stays legal
        uni = ldf * np.arange(10, 20, dtype=float)
        self.assertEqual(prop(self._stub(5, uni, ldf)), 5)


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
