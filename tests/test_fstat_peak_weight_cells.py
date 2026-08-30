"""The F-stat birth draw must be stratified by SUB-BAND, not by cap cell.

USER RULING 2026-08-29: "The sub-bands should define that grid since those
are the effective rj limits as well."

That is the physical argument. RJ births are confined to sub-bands (the
draw is global, then candidates are divided up and assigned to sub-bands by
drawn f0), and scheduling is serial-within-band -- one RJ source per
sub-band per round. So the sub-band is the unit a birth actually competes
in. Cap cells are a *cap bookkeeping* overlay, and under the staggered grid
they deliberately straddle sub-band seams; stratifying the proposal on them
mismatches the RJ limits.

TWO DEFECTS this pins:

1. ``peak_weight_cells_env`` defaulted to ``GB_CAP_DIVISOR``, tying the
   draw grid to the cap grid.
2. ``peak_box_weights`` built its cells with
   ``np.linspace(be[0], be[-1], ncell + 1)`` -- a UNIFORM grid over the
   full range, ignoring ``band_edges`` entirely. That coincides with the
   sub-bands only when the band grid happens to be uniform, and matches the
   (staggered) cap cells never -- despite the docstring claiming "the same
   construction as the leaf-cap grid".

The mixture stays exact either way: these are flat per-box weights and
``StackedFStatProposal4D`` drives both ``rvs`` and ``logpdf`` from them, so
changing which cell a box belongs to changes the weights, not the
construction.
"""
import os
import unittest
from unittest import mock

import numpy as np

from lisatools.sampling.fstat_gridfit import (
    peak_box_weights, peak_weight_cells_env,
)


UNIFORM = np.linspace(1e-3, 2e-3, 11)                        # 10 equal bands
RAGGED = np.array([1.0, 1.3, 1.45, 2.0, 2.2, 3.0]) * 1e-3    # 5 uneven bands


def _mid(be, b, frac=0.5):
    """A frequency inside sub-band ``b``, in mHz (the arg unit)."""
    return (be[b] + frac * (be[b + 1] - be[b])) * 1e3


class CellsFollowBandEdgesTest(unittest.TestCase):
    """Stratification cells must be derived from ``band_edges``."""

    def test_ragged_grid_strata_are_the_sub_bands(self):
        """THE BUG: with linspace cells, two peaks in the SAME wide
        sub-band land in different strata (and peaks in different narrow
        sub-bands share one), so per-stratum mass is not per-sub-band."""
        be = RAGGED
        # band 3 is [2.0, 2.2) mHz -- narrow; band 2 is [1.45, 2.0) -- wide
        f0 = np.array([_mid(be, 2, 0.1), _mid(be, 2, 0.9), _mid(be, 3, 0.5)])
        F = np.array([1.0, 1.0, 1.0])
        w = peak_box_weights(F, peak_f0_mHz=f0, band_edges=be, alpha=1.0,
                             cells=1)
        self.assertIsNotNone(w)
        w = np.asarray(w, dtype=float)
        # two occupied sub-bands -> half the mass each; the wide band's two
        # peaks split its half
        np.testing.assert_allclose(w[0] + w[1], w[2], rtol=1e-12)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)

    def test_k1_gives_every_occupied_sub_band_equal_mass(self):
        be = UNIFORM
        f0 = np.array([_mid(be, 0), _mid(be, 5, 0.2), _mid(be, 5, 0.8)])
        F = np.array([100.0, 1.0, 1.0])   # band 0 far louder
        w = np.asarray(peak_box_weights(F, peak_f0_mHz=f0, band_edges=be,
                                        alpha=1.0, cells=1), dtype=float)
        np.testing.assert_allclose(w[0], 0.5, rtol=1e-12)
        np.testing.assert_allclose(w[1] + w[2], 0.5, rtol=1e-12)

    def test_within_a_sub_band_F_alpha_preference_survives(self):
        be = UNIFORM
        f0 = np.array([_mid(be, 5, 0.2), _mid(be, 5, 0.8)])
        F = np.array([3.0, 1.0])
        w = np.asarray(peak_box_weights(F, peak_f0_mHz=f0, band_edges=be,
                                        alpha=1.0, cells=1), dtype=float)
        np.testing.assert_allclose(w[0] / w[1], 3.0, rtol=1e-12)

    def test_subdivision_uses_each_sub_bands_own_width(self):
        """K>1 splits each sub-band by ITS width, not a global linspace."""
        be = RAGGED
        # lower and upper half of the WIDE band 2 = different strata at K=2
        f0 = np.array([_mid(be, 2, 0.25), _mid(be, 2, 0.75)])
        F = np.array([9.0, 1.0])
        w = np.asarray(peak_box_weights(F, peak_f0_mHz=f0, band_edges=be,
                                        alpha=1.0, cells=2), dtype=float)
        np.testing.assert_allclose(w, [0.5, 0.5], rtol=1e-12)

    def test_weights_are_normalised_and_non_negative(self):
        rng = np.random.default_rng(11)
        for be in (UNIFORM, RAGGED):
            for cells in (1, 2, 4):
                f0 = rng.uniform(be[0], be[-1], size=200) * 1e3
                F = rng.uniform(0.1, 50.0, size=200)
                w = np.asarray(peak_box_weights(F, peak_f0_mHz=f0,
                                                band_edges=be, alpha=1.0,
                                                cells=cells), dtype=float)
                self.assertTrue(np.all(w >= 0))
                np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-9)

    def test_no_peak_lands_outside_the_band_range(self):
        """Clipping must not silently dump edge peaks into stratum 0."""
        be = UNIFORM
        f0 = np.array([_mid(be, 0, 0.001), _mid(be, 9, 0.999)])
        w = np.asarray(peak_box_weights(np.array([1.0, 1.0]), peak_f0_mHz=f0,
                                        band_edges=be, alpha=1.0, cells=1),
                       dtype=float)
        np.testing.assert_allclose(w, [0.5, 0.5], rtol=1e-12)


class GlobalFallbackTest(unittest.TestCase):
    """The historical global mixture stays reachable."""

    def test_cells_zero_is_the_global_mixture(self):
        be = UNIFORM
        f0 = np.array([_mid(be, 0), _mid(be, 5)])
        F = np.array([3.0, 1.0])
        w = peak_box_weights(F, peak_f0_mHz=f0, band_edges=be, alpha=1.0,
                             cells=0)
        np.testing.assert_allclose(np.asarray(w, dtype=float), F, rtol=1e-12)

    def test_equal_returns_none(self):
        self.assertIsNone(peak_box_weights(np.array([1.0, 2.0]), equal=True))

    def test_missing_band_edges_falls_back_not_crashes(self):
        w = peak_box_weights(np.array([2.0, 1.0]), peak_f0_mHz=None,
                             band_edges=None, alpha=1.0, cells=1)
        np.testing.assert_allclose(np.asarray(w, dtype=float), [2.0, 1.0])


class CellsEnvTest(unittest.TestCase):
    """The draw grid must not track GB_CAP_DIVISOR."""

    def test_default_is_per_sub_band_not_the_cap_divisor(self):
        with mock.patch.dict(os.environ, {"GB_CAP_DIVISOR": "8"}, clear=False):
            os.environ.pop("FSTAT_PEAK_WEIGHT_CELLS", None)
            self.assertEqual(peak_weight_cells_env(), 1)

    def test_explicit_override_still_honoured(self):
        with mock.patch.dict(os.environ,
                             {"FSTAT_PEAK_WEIGHT_CELLS": "4"}, clear=False):
            self.assertEqual(peak_weight_cells_env(), 4)

    def test_zero_selects_the_global_mixture(self):
        with mock.patch.dict(os.environ,
                             {"FSTAT_PEAK_WEIGHT_CELLS": "0"}, clear=False):
            self.assertEqual(peak_weight_cells_env(), 0)

    def test_garbage_does_not_crash(self):
        with mock.patch.dict(os.environ,
                             {"FSTAT_PEAK_WEIGHT_CELLS": "xyz"}, clear=False):
            self.assertEqual(peak_weight_cells_env(), 1)


if __name__ == "__main__":
    unittest.main()
