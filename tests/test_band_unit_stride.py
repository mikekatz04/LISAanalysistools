"""Stride-k band-unit scheduling + support-based separation guards.

The concurrency machinery's correctness premise (user physics ruling,
verified): FD inner product ~0 implies WDM inner product ~0 even within
one wavelet layer, so the constraint on concurrently-scheduled GB
sub-bands is ORTHOGONALITY (frequency separation), not wavelet-pixel
support. The 2026-08-15 width ruling makes the separation guard
SUPPORT-based: band edges are free-floating frequencies, band widths in
get_n mode are the minimum ``2 * get_N(f_max_band) / Tobs``, and the
same-unit gap is compared against the sum of edge-source half-supports
``get_N(f_edge)/Tobs`` times ``GB_ORTHO_SEP_FACTOR``. These tests pin:

* the stride resolution knob (``GB_BAND_UNIT_STRIDE``; default 2 =
  legacy parity, bit-identical),
* the tempering unit selection (grid slice <-> open-class remainder
  mapping, including the legacy stride-2 formula),
* the support-based separation rule (stride 2 always passes for
  get_n-built grids at factor 1.0; stride 3 = a full band of clearance),
* the get_N width rule (free-floating, maximal packing, self-consistent
  fixed point), and
* that STORAGE slab geometry stays WHOLE-LAYER, floor/ceiled from each
  band's actual (unaligned) frequency range, straddlers included.
"""

import os
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
from gbgpu.utils.utility import get_N

from lisatools.globalfit.moves.gbbands import (
    SubBandBuffer,
    band_support_halfwidths,
    check_band_support_separation,
)
from lisatools.globalfit.moves.gbspecialstretch import (
    _resolve_band_unit_stride,
    _tempering_open_remainder,
)
from lisatools.utils.constants import YRSID_SI

TOBS = YRSID_SI / 4.0  # 3-month-like
LDF = 1.393e-4


def _get_n_edges(start=5.5e-4, end=6e-3, **kw):
    from lisatools.globalfit.stock.erebor.gb import get_n_based_band_edges

    return get_n_based_band_edges(start, end, TOBS, LDF, **kw)


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


class SupportSeparationGuardTest(unittest.TestCase):
    """Rule: gap(same-unit pair) >= GB_ORTHO_SEP_FACTOR * sum of the
    edge-source half-supports get_N(f_edge)/Tobs."""

    def test_get_n_grid_passes_stride2_at_factor_one(self):
        # The 2*get_N width rule guarantees stride 2 at factor 1.0
        # (gap = middle band = 2*s(f_hi_mid) >= s(f_hi_low) + s(f_hi_mid)).
        edges = _get_n_edges()
        out = check_band_support_separation(edges, TOBS, 2)
        self.assertTrue(out["passes"])
        self.assertEqual(out["min_safe_stride"], 2)

    def test_factor_two_needs_stride3_full_band_clearance(self):
        edges = _get_n_edges()
        out = check_band_support_separation(
            edges, TOBS, 2, sep_factor=2.0, enforce=False
        )
        self.assertFalse(out["passes"])
        self.assertEqual(out["min_safe_stride"], 3)
        out3 = check_band_support_separation(edges, TOBS, 3, sep_factor=2.0)
        self.assertTrue(out3["passes"])

    def test_env_factor_resolves(self):
        edges = _get_n_edges()  # built at the default factor 1.0
        os.environ["GB_ORTHO_SEP_FACTOR"] = "2.0"
        self.addCleanup(os.environ.pop, "GB_ORTHO_SEP_FACTOR", None)
        out = check_band_support_separation(edges, TOBS, 2, enforce=False)
        self.assertEqual(out["sep_factor"], 2.0)
        self.assertFalse(out["passes"])

    def test_narrow_grid_fails_and_min_stride_matches_manual(self):
        # Bands far narrower than the source support: gap << overhangs.
        f0 = 4e-3
        s = float(get_N(1e-30, f0, TOBS, oversample=4).item()) / TOBS
        delta = s / 2.0  # band width half the edge half-support
        edges = f0 + np.arange(8) * delta
        with self.assertRaisesRegex(ValueError, "unsafe"):
            check_band_support_separation(edges, TOBS, 2, context="test")
        out = check_band_support_separation(edges, TOBS, 2, enforce=False)
        self.assertFalse(out["passes"])
        # manual: gap = (k-1)*delta, need = 2*s (get_N flat here) ->
        # min k = 1 + ceil(2*s/delta) = 1 + 4 = 5
        self.assertEqual(out["min_safe_stride"], 5)
        self.assertTrue(
            check_band_support_separation(edges, TOBS, 5)["passes"]
        )

    def test_uniform_legacy_grid_reports_not_raises_in_diagnostic(self):
        # 1-layer uniform bands: the conservative FD-support envelope
        # fails at stride 2 at 3 months (support ~0.93 layers per side);
        # enforce=False is the diagnostic mode the legacy paths use.
        uni = np.arange(4, 159) * LDF
        out = check_band_support_separation(uni, TOBS, 2, enforce=False)
        self.assertFalse(out["passes"])
        self.assertEqual(out["min_safe_stride"], 3)

    def test_single_band_always_passes(self):
        out = check_band_support_separation(
            np.array([1e-3, 2e-3]), TOBS, 2
        )
        self.assertTrue(out["passes"])
        self.assertIsNone(out["min_safe_stride"])

    def test_support_halfwidths_are_get_n_over_tobs(self):
        edges = np.array([1e-3, 2e-3, 4e-3])
        s = band_support_halfwidths(edges, TOBS)
        expect = [
            float(get_N(1e-30, f, TOBS, oversample=4).item()) / TOBS
            for f in edges[1:]
        ]
        np.testing.assert_allclose(s, expect, rtol=0, atol=0)


class GetNWidthRuleBuilderTest(unittest.TestCase):
    """Free-floating maximal packing at the 2*get_N floor."""

    def test_outer_edges_verbatim_no_snapping(self):
        # deliberately NOT layer-aligned start/end
        start, end = 5.51e-4, 6.01e-3
        edges = _get_n_edges(start, end)
        self.assertEqual(edges[0], start)
        self.assertEqual(edges[-1], end)

    def test_interior_widths_at_their_own_floor(self):
        # THE width rule: every interior band's width equals its own
        # minimum 2*get_N(f_max_band)/Tobs (self-consistent fixed point)
        # to float precision -> the band count is maximal for the rule.
        edges = _get_n_edges()
        w = np.diff(edges)
        df = 1.0 / TOBS
        for i in range(len(w) - 1):  # last band absorbs the remainder
            floor_i = 2.0 * get_N(
                1e-30, edges[i + 1], TOBS, oversample=4
            ).item() * df
            self.assertLess(abs(w[i] - floor_i), 1e-3 * df,
                            f"band {i} width off its 2*get_N floor")

    def test_last_band_absorbs_remainder_only(self):
        edges = _get_n_edges()
        w = np.diff(edges)
        df = 1.0 / TOBS
        # last band: clamped to end_freq; never below its floor by more
        # than the merge rule allows (merged bands exceed the floor).
        floor_last = 2.0 * get_N(
            1e-30, edges[-1], TOBS, oversample=4
        ).item() * df
        self.assertGreaterEqual(w[-1], min(floor_last, w[-2]) * 0.999)

    def test_stride_guard_enforced_at_build(self):
        os.environ["GB_ORTHO_SEP_FACTOR"] = "2.0"
        self.addCleanup(os.environ.pop, "GB_ORTHO_SEP_FACTOR", None)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            _get_n_edges(unit_stride=2)
        edges = _get_n_edges(unit_stride=3)  # full band of clearance
        self.assertGreater(len(edges), 3)

    def test_deprecated_knobs_warned_and_ignored(self):
        base = _get_n_edges()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            div = _get_n_edges(subband_divisor=2)
            self.assertTrue(any(
                issubclass(r.category, DeprecationWarning)
                and "subband_divisor" in str(r.message) for r in rec))
        np.testing.assert_array_equal(div, base)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ml = _get_n_edges(min_band_layers=0.5)
            self.assertTrue(any(
                issubclass(r.category, DeprecationWarning)
                and "min_band_layers" in str(r.message) for r in rec))
        np.testing.assert_array_equal(ml, base)

    def test_target_count_merges_adjacent_floors(self):
        base = _get_n_edges()
        n = len(base) - 1
        target = max(2, n // 3)
        merged = _get_n_edges(target_count=target)
        self.assertEqual(len(merged) - 1, target)
        # merged edges are a subset of the maximal-packing edges
        self.assertTrue(np.all(np.isin(merged, base)))
        # outer edges preserved; every merged band still >= its own floor
        self.assertEqual(merged[0], base[0])
        self.assertEqual(merged[-1], base[-1])
        df = 1.0 / TOBS
        w = np.diff(merged)
        for i in range(len(w) - 1):
            floor_i = 2.0 * get_N(
                1e-30, merged[i + 1], TOBS, oversample=4
            ).item() * df
            self.assertGreaterEqual(w[i], floor_i * (1 - 1e-12))
        # target above the maximal packing is a no-op
        np.testing.assert_array_equal(_get_n_edges(target_count=10 * n), base)


class SlabWholeLayerTest(unittest.TestCase):
    """STORAGE slabs stay layer-derived under free-floating edges: whole-
    layer origins/spans floor/ceiled from each band's actual frequency
    range (straddlers count both layers), support-aware margins."""

    ldf = 1e-3

    def test_recommend_floors_ceils_touched_layers(self):
        # straddling band [10.7, 11.2]*ldf touches layers 10 and 11 ->
        # span 2; within-layer band [12.2, 12.6]*ldf -> span 1.
        edges = np.array([10.7, 11.2, 12.2]) * self.ldf
        out = SubBandBuffer.recommend_band_slab_layers(
            edges, self.ldf, guard=1, xp=np
        )
        self.assertIsInstance(out, int)
        self.assertEqual(out, 2 + 2 * (2 + 1))  # straddler span 2 + margins

    def test_recommend_aligned_uniform_unchanged(self):
        # exactly aligned 1-layer bands: legacy value span 1 + 2*(2+1)=7,
        # bit-identical (epsilon guards float noise).
        edges = np.arange(10, 20) * self.ldf * (1.0 + 1e-13)
        out = SubBandBuffer.recommend_band_slab_layers(
            edges, self.ldf, guard=1, xp=np
        )
        self.assertEqual(out, 7)

    def test_recommend_support_aware_margins(self):
        # When Tobs is given the margin is max(leakage, support layers):
        # pick ldf small enough that the FD support spans > 2 layers.
        edges = np.array([1e-3, 1.2e-3])
        s = band_support_halfwidths(edges, TOBS)[0]
        ldf = s / 3.0  # support = 3 layers > leakage 2
        span = int(np.ceil(edges[1] / ldf - 1e-6)) - int(
            np.floor(edges[0] / ldf + 1e-6))
        out = SubBandBuffer.recommend_band_slab_layers(
            edges, ldf, guard=1, xp=np, Tobs=TOBS
        )
        self.assertEqual(out, span + 2 * (3 + 1))
        # without Tobs: legacy leakage-only margins
        out_legacy = SubBandBuffer.recommend_band_slab_layers(
            edges, ldf, guard=1, xp=np
        )
        self.assertEqual(out_legacy, span + 2 * (2 + 1))

    def test_slab_origins_whole_layer_and_cover_straddlers(self):
        # unaligned half-layer-wide bands, several straddling boundaries
        edges = (np.arange(9) * 0.5 + 10.3) * self.ldf
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
            lo_touched = int(np.floor(edges[b] / self.ldf + 1e-6))
            hi_touched = int(np.ceil(edges[b + 1] / self.ldf - 1e-6)) - 1
            self.assertLessEqual(int(origins[slot]), lo_touched - hw)
            self.assertGreaterEqual(
                int(origins[slot]) + slab_Nf, hi_touched + hw + 1
            )

    def test_aligned_center_formula_matches_legacy(self):
        # aligned 1-layer bands: center must stay the legacy
        # (lo + hi) // 2 = lo, i.e. origin = lo - slab//2.
        edges = np.arange(10, 19).astype(float) * self.ldf
        stub = SimpleNamespace(
            band_slab_Nf=7,
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
        for slot in range(8):
            b = stub.unique_band_combos[slot, 2]
            self.assertEqual(int(origins[slot]), (10 + b) - 7 // 2)


if __name__ == "__main__":
    unittest.main()
