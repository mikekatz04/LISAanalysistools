"""Stride-k band-unit scheduling + sub-layer band guards.

The concurrency machinery's correctness premise (user physics ruling,
verified): FD inner product ~0 implies WDM inner product ~0 even within
one wavelet layer, so the constraint on concurrently-scheduled GB
sub-bands is ORTHOGONALITY (frequency separation), not wavelet-pixel
support. These tests pin down:

* the stride resolution knob (``GB_BAND_UNIT_STRIDE``; default 2 =
  legacy parity, bit-identical),
* the tempering unit selection (grid slice <-> open-class remainder
  mapping, including the legacy stride-2 formula),
* the separation guard rule ``(stride - 1) * min_band_width_layers >= 1``
  (=> stride >= div + 1 for 1/div-layer minimum bands),
* the get_N edge builder's sub-layer enablement, and
* that slab geometry stays WHOLE-LAYER under sub-layer bands
  (scheduling-only division; user architecture ruling).
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbbands import (
    SubBandBuffer,
    check_band_stride_separation,
)
from lisatools.globalfit.moves.gbspecialstretch import (
    _resolve_band_unit_stride,
    _tempering_open_remainder,
)


class StrideResolutionTest(unittest.TestCase):
    def test_default_is_two(self):
        self.assertEqual(_resolve_band_unit_stride("gb", 2), 2)

    def test_ctor_value_kept_without_env(self):
        self.assertEqual(_resolve_band_unit_stride("gb", 3), 3)

    def test_env_wins_and_is_branch_prefixed(self):
        os.environ["GB_BAND_UNIT_STRIDE"] = "4"
        self.addCleanup(os.environ.pop, "GB_BAND_UNIT_STRIDE", None)
        self.assertEqual(_resolve_band_unit_stride("gb", 2), 4)
        # a different branch does not read the gb knob
        self.assertEqual(_resolve_band_unit_stride("vgb", 2), 2)
        os.environ["VGB_BAND_UNIT_STRIDE"] = "3"
        self.addCleanup(os.environ.pop, "VGB_BAND_UNIT_STRIDE", None)
        self.assertEqual(_resolve_band_unit_stride("vgb", 2), 3)

    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            _resolve_band_unit_stride("gb", 0)
        os.environ["GB_BAND_UNIT_STRIDE"] = "0"
        self.addCleanup(os.environ.pop, "GB_BAND_UNIT_STRIDE", None)
        with self.assertRaises(ValueError):
            _resolve_band_unit_stride("gb", 2)


class TemperingUnitSelectionTest(unittest.TestCase):
    """The tempering grid slices interior bands ``arange(1, nb-1)
    [start::units]``; the residual open/close must expose exactly the
    ``band % units == _tempering_open_remainder(start, units)`` class."""

    def _selected(self, nb, start, units):
        return np.arange(1, nb - 1)[start::units]

    def test_grid_slice_matches_open_class(self):
        for nb in (4, 5, 9, 12, 31):
            for units in (2, 3, 4, 5):
                for start in range(units):
                    sel = self._selected(nb, start, units)
                    rem = _tempering_open_remainder(start, units)
                    self.assertTrue(
                        np.all(sel % units == rem),
                        f"nb={nb} units={units} start={start}",
                    )

    def test_stride2_reproduces_legacy_bool_remainder(self):
        # Legacy code: ``bool_remainder = 1 if start == 0 else 0``.
        self.assertEqual(_tempering_open_remainder(0, 2), 1)
        self.assertEqual(_tempering_open_remainder(1, 2), 0)

    def test_units_partition_interior_bands(self):
        for nb in (5, 9, 12):
            for units in (2, 3, 4):
                parts = [self._selected(nb, s, units) for s in range(units)]
                allb = np.sort(np.concatenate(parts))
                np.testing.assert_array_equal(allb, np.arange(1, nb - 1))
                # disjoint
                self.assertEqual(len(allb), len(np.unique(allb)))

    def test_same_unit_bands_k_apart(self):
        # stride k => consecutive same-unit bands differ by exactly k,
        # i.e. k - 1 closed bands between any two concurrent bands.
        for units in (2, 3, 4):
            sel = self._selected(40, 0, units)
            self.assertTrue(np.all(np.diff(sel) == units))


class StrideSeparationGuardTest(unittest.TestCase):
    """Rule: (stride - 1) * min_band_width_layers >= 1 full WDM layer."""

    ldf = 1e-3

    def _edges(self, width_layers, n=10):
        return np.arange(n) * width_layers * self.ldf

    def test_whole_layer_stride2_passes(self):
        w = check_band_stride_separation(self._edges(1.0), self.ldf, 2)
        self.assertAlmostEqual(w, 1.0)

    def test_stride1_whole_layer_refused(self):
        # units=1 means every band concurrent: zero separation.
        with self.assertRaises(ValueError):
            check_band_stride_separation(self._edges(1.0), self.ldf, 1)

    def test_half_layer_needs_stride3(self):
        with self.assertRaises(ValueError) as cm:
            check_band_stride_separation(self._edges(0.5), self.ldf, 2)
        self.assertIn("GB_BAND_UNIT_STRIDE >= 3", str(cm.exception))
        self.assertAlmostEqual(
            check_band_stride_separation(self._edges(0.5), self.ldf, 3), 0.5
        )

    def test_quarter_layer_needs_stride5(self):
        # div=4 -> min width 1/4 layer -> stride >= div + 1 = 5.
        for bad in (2, 3, 4):
            with self.assertRaises(ValueError):
                check_band_stride_separation(self._edges(0.25), self.ldf, bad)
        check_band_stride_separation(self._edges(0.25), self.ldf, 5)

    def test_mixed_widths_use_minimum(self):
        # variable-width grid: guard binds on the NARROWEST band.
        edges = np.array([0.0, 0.5, 1.5, 3.5, 7.5]) * self.ldf
        with self.assertRaises(ValueError):
            check_band_stride_separation(edges, self.ldf, 2)
        check_band_stride_separation(edges, self.ldf, 3)

    def test_single_band_always_passes(self):
        # < 2 bands: nothing is ever concurrent.
        out = check_band_stride_separation(
            np.array([0.0, 0.5]) * self.ldf, self.ldf, 2
        )
        self.assertEqual(out, float("inf"))

    def test_float_noise_on_whole_layer_grid_tolerated(self):
        edges = (np.arange(8) * self.ldf) * (1.0 + 1e-13)
        check_band_stride_separation(edges, self.ldf, 2)  # must not raise


class GetNBuilderSubLayerTest(unittest.TestCase):
    Tobs = 3.15576e7 / 4.0  # ~3 months
    ldf = 1.0 / (2 * 15.0 * 512)  # a WDM-layer-like width (~6.5e-5 Hz)

    def _build(self, **kw):
        from lisatools.globalfit.stock.erebor.gb import get_n_based_band_edges

        args = dict(
            start_freq=1e-3,
            end_freq=3e-3,
            Tobs=self.Tobs,
            layer_df=self.ldf,
        )
        args.update(kw)
        return get_n_based_band_edges(**args)

    def test_default_min_layers_independent_of_stride(self):
        # whole-layer minimum: the stride argument must not change the
        # edges (bit-identity of the default construction).
        e2 = self._build(min_band_layers=1.0, unit_stride=2)
        e5 = self._build(min_band_layers=1.0, unit_stride=5)
        np.testing.assert_array_equal(e2, e5)
        # widths are whole numbers of layers
        widths = np.diff(e2) / self.ldf
        np.testing.assert_allclose(widths, np.round(widths), atol=1e-6)
        self.assertGreaterEqual(widths.min(), 1.0 - 1e-9)

    def test_sub_layer_requires_stride_bump(self):
        with self.assertRaises(ValueError) as cm:
            self._build(min_band_layers=0.5, subband_divisor=2, unit_stride=2)
        self.assertIn("GB_BAND_UNIT_STRIDE >= 3", str(cm.exception))
        edges = self._build(
            min_band_layers=0.5, subband_divisor=2, unit_stride=3
        )
        # edges land on the half-layer grid; min width >= 0.5 layer
        k = edges / (self.ldf / 2)
        np.testing.assert_allclose(k, np.round(k), atol=1e-6)
        widths = np.diff(edges) / self.ldf
        self.assertGreaterEqual(widths.min(), 0.5 - 1e-9)
        # the built grid passes its own separation guard at stride 3
        check_band_stride_separation(edges, self.ldf, 3)

    def test_quarter_layer_rule_div_plus_one(self):
        with self.assertRaises(ValueError):
            self._build(min_band_layers=0.25, subband_divisor=4, unit_stride=4)
        edges = self._build(
            min_band_layers=0.25, subband_divisor=4, unit_stride=5
        )
        widths = np.diff(edges) / self.ldf
        self.assertGreaterEqual(widths.min(), 0.25 - 1e-9)

    def test_nonpositive_min_layers_refused(self):
        with self.assertRaises(ValueError):
            self._build(min_band_layers=0.0, unit_stride=8)

    def test_outer_edges_full_layer_aligned(self):
        edges = self._build(
            min_band_layers=0.5, subband_divisor=2, unit_stride=3
        )
        for e in (edges[0], edges[-1]):
            self.assertLess(abs(e / self.ldf - round(e / self.ldf)), 1e-6)


class SlabWholeLayerTest(unittest.TestCase):
    """USER ARCHITECTURE RULING: sub-layer bands are a SCHEDULING-ONLY
    division -- the WDM buffer slabs keep whole layers and never shrink
    below the band's layer +- the likelihood-window spread."""

    ldf = 1e-3

    def test_recommended_slab_is_whole_layers_and_floored(self):
        # half-layer bands: span floors to 1 layer; slab = 1 + 2*(2+1) = 7
        edges = np.arange(9) * 0.5 * self.ldf
        out = SubBandBuffer.recommend_band_slab_layers(
            edges, self.ldf, guard=1, xp=np
        )
        self.assertIsInstance(out, int)
        self.assertEqual(out, 7)

    def test_slab_origins_whole_layer_and_cover_band(self):
        # duck-typed stub (the property fget dispatches through the class,
        # a pattern the production code documents for exactly this use).
        edges = np.arange(9) * 0.5 * self.ldf + 10 * self.ldf
        slab_Nf = SubBandBuffer.recommend_band_slab_layers(
            edges, self.ldf, guard=1, xp=np
        )
        stub = SimpleNamespace(
            band_slab_Nf=slab_Nf,
            df=self.ldf,
            xp=np,
            band_edges=edges,
            unique_band_combos=np.stack(
                [np.zeros(8, int), np.zeros(8, int), np.arange(8)], axis=1
            ),
            _basis_settings=SimpleNamespace(ind_min_f=0, ind_max_f=400),
            _n_slots_alloc=8,
        )
        origins = SubBandBuffer._compute_slab_min_f(stub)
        self.assertTrue(np.issubdtype(origins.dtype, np.integer))
        hw = 1  # m_band_half_width: source window spreads +-1 layer
        for slot in range(8):
            b = stub.unique_band_combos[slot, 2]
            lo_layer = int(edges[b] / self.ldf)
            hi_layer = int(edges[b + 1] / self.ldf)
            self.assertLessEqual(int(origins[slot]), lo_layer - hw)
            self.assertGreaterEqual(
                int(origins[slot]) + slab_Nf, hi_layer + hw + 1
            )


if __name__ == "__main__":
    unittest.main()
