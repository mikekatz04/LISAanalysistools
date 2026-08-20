"""Staggered leaf-cap cell grid (GB_CAP_STAGGER, user design 2026-08-20).

The v5 grid: interior cap edges shifted half a cell so no cap edge
coincides with a sub-band edge. These tests pin:

* the staggered ``make_cap_edges`` construction (edge/cell counts match
  the nested grid exactly; no shared band/cap edges; half + 1.5 boundary
  cells; ``K == 1`` ignores the flag),
* exact agreement of the move's arithmetic cell lookup
  (``_cap_cell_index`` and its numpy twin ``_np_cap_cells``) with
  ``searchsorted`` over the stored edges, in BOTH modes, on uniform AND
  non-uniform band grids,
* the staggered band-saturation rule (a band is birth-saturated only when
  the straddling boundary cell it does not own is also full).

All through the installed classes -- no reimplementation of the grid.
"""
import unittest

import numpy as np

from lisatools.globalfit.state import make_cap_edges, cap_divisor_from_edges
from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


def _shim(band_edges, k, stagger):
    """A bare cap-machinery carrier: the real class, no heavy __init__."""
    mv = object.__new__(GBSpecialBase)
    be = np.asarray(band_edges, dtype=float)
    mv.cap_divisor = int(k)
    mv.cap_stagger = bool(stagger) and k > 1
    mv.num_bands = len(be) - 1
    mv.num_cap_cells = mv.num_bands * k
    mv.cap_edges = make_cap_edges(be, k, stagger=mv.cap_stagger)
    mv._cap_band_lo = be[:-1]
    mv._cap_band_step = (be[1:] - be[:-1]) / k
    return mv


UNIFORM = np.linspace(1e-3, 2e-3, 11)              # 10 equal bands
RAGGED = np.array([1.0, 1.3, 1.45, 2.0, 2.2, 3.0]) * 1e-3  # 5 uneven bands


class MakeCapEdgesStaggerTest(unittest.TestCase):
    def test_counts_match_nested(self):
        for be in (UNIFORM, RAGGED):
            for k in (2, 4, 8, 32):
                nested = make_cap_edges(be, k)
                stag = make_cap_edges(be, k, stagger=True)
                self.assertEqual(len(stag), len(nested))
                self.assertEqual(cap_divisor_from_edges(be, stag), k)

    def test_no_shared_edges(self):
        for be in (UNIFORM, RAGGED):
            stag = make_cap_edges(be, 8, stagger=True)
            interior = stag[1:-1]
            for e in be:
                self.assertGreater(
                    np.abs(interior - e).min(), 1e-9 * abs(e),
                    msg=f"band edge {e} coincides with a staggered cap edge",
                )

    def test_boundary_cell_widths(self):
        be = UNIFORM
        k = 8
        step = (be[1] - be[0]) / k
        stag = make_cap_edges(be, k, stagger=True)
        w = np.diff(stag)
        self.assertAlmostEqual(w[0], step / 2, delta=1e-12 * step)
        self.assertAlmostEqual(w[-1], 1.5 * step, delta=1e-12 * step)
        np.testing.assert_allclose(w[1:-1], step, rtol=1e-9)
        self.assertAlmostEqual(stag[0], be[0])
        self.assertAlmostEqual(stag[-1], be[-1])

    def test_k1_ignores_stagger(self):
        np.testing.assert_array_equal(
            make_cap_edges(UNIFORM, 1, stagger=True),
            make_cap_edges(UNIFORM, 1),
        )


class CellAssignmentTest(unittest.TestCase):
    def _check(self, be, k, stagger):
        mv = _shim(be, k, stagger)
        rng = np.random.default_rng(7)
        f = rng.uniform(be[0], be[-1] * (1 - 1e-12), size=20000)
        # include exact band edges and near-edge frequencies
        f = np.concatenate([f, be[:-1], be[:-1] + 1e-15, be[1:] - 1e-12])
        band = np.clip(
            np.searchsorted(be, f, side="right") - 1, 0, mv.num_bands - 1
        )
        got = mv._cap_cell_index(band.astype(np.int64), f)
        want = np.clip(
            np.searchsorted(mv.cap_edges, f, side="right") - 1,
            0, mv.num_cap_cells - 1,
        )
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(mv._np_cap_cells(f, band, be), want)

    def test_nested_uniform(self):
        self._check(UNIFORM, 8, stagger=False)

    def test_staggered_uniform(self):
        self._check(UNIFORM, 8, stagger=True)

    def test_staggered_ragged(self):
        self._check(RAGGED, 4, stagger=True)

    def test_staggered_straddle_ownership(self):
        """A source just below a band seam lands in the NEXT band's cell 0."""
        be, k = UNIFORM, 8
        mv = _shim(be, k, stagger=True)
        step = (be[1] - be[0]) / k
        f = np.array([be[1] - 0.25 * step])   # top half-cell of band 0
        band = np.array([0], dtype=np.int64)
        self.assertEqual(int(mv._cap_cell_index(band, f)[0]), k)  # cell 1*K


class BandSaturationTest(unittest.TestCase):
    def _counts(self, mv, full_cells):
        counts = np.zeros(mv.num_cap_cells, dtype=np.int64)
        counts[list(full_cells)] = 1
        cap = np.ones(mv.num_cap_cells, dtype=np.int64)
        return counts, cap

    def test_staggered_needs_boundary_cell(self):
        be, k = UNIFORM, 4
        mv = _shim(be, k, stagger=True)
        owned = range(0, k)  # band 0's owned cells
        counts, cap = self._counts(mv, owned)
        sat = mv._band_saturated_flat(counts, cap).reshape(-1, mv.num_bands)
        self.assertFalse(bool(sat[0, 0]))  # boundary cell k has room
        counts, cap = self._counts(mv, list(owned) + [k])
        sat = mv._band_saturated_flat(counts, cap).reshape(-1, mv.num_bands)
        self.assertTrue(bool(sat[0, 0]))

    def test_staggered_last_band_owned_only(self):
        be, k = UNIFORM, 4
        mv = _shim(be, k, stagger=True)
        last = range((mv.num_bands - 1) * k, mv.num_bands * k)
        counts, cap = self._counts(mv, last)
        sat = mv._band_saturated_flat(counts, cap).reshape(-1, mv.num_bands)
        self.assertTrue(bool(sat[0, -1]))

    def test_nested_unchanged(self):
        be, k = UNIFORM, 4
        mv = _shim(be, k, stagger=False)
        counts, cap = self._counts(mv, range(0, k))
        sat = mv._band_saturated_flat(counts, cap).reshape(-1, mv.num_bands)
        self.assertTrue(bool(sat[0, 0]))
        self.assertFalse(bool(sat[0, 1]))


if __name__ == "__main__":
    unittest.main()
