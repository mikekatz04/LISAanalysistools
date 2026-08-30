"""Staggered leaf-cap cell grid (GB_CAP_STAGGER, user design 2026-08-20).

The v5 grid: interior cap edges shifted half a cell so no cap edge
coincides with a sub-band edge. These tests pin:

* the staggered ``make_cap_edges`` construction (edge/cell counts match
  the nested grid exactly; no shared band/cap edges; half + 1.5 boundary
  cells; ``K == 1`` + stagger = the midpoint-to-midpoint grid),
* exact agreement of the move's arithmetic cell lookup
  (``_cap_cell_index`` and its numpy twin ``_np_cap_cells``) with
  ``searchsorted`` over the stored edges, in BOTH modes, on uniform AND
  non-uniform band grids,
* the staggered band-saturation rule (a band is birth-saturated only when
  the straddling boundary cell it does not own is also full).

All through the installed classes -- no reimplementation of the grid.
"""
import types
import unittest

import numpy as np

from lisatools.globalfit.state import (
    make_cap_edges, cap_divisor_from_edges, ensure_cap_cell_fields,
)
from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


def _shim(band_edges, k, stagger):
    """A bare cap-machinery carrier: the real class, no heavy __init__."""
    mv = object.__new__(GBSpecialBase)
    be = np.asarray(band_edges, dtype=float)
    mv.cap_divisor = int(k)
    # Mirrors production (gbspecialstretch.py: ``self.cap_stagger =
    # bool(cap_stagger) and self.cap_divisor > 1``) EXCEPT that K=1 stagger
    # is honoured here, so the K=1 lookup tests below pin the target
    # behaviour rather than today's short-circuit. See _K1_LOOKUP_TODO.
    mv.cap_stagger = bool(stagger)
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

    def test_k1_unstaggered_is_the_band_grid(self):
        np.testing.assert_array_equal(
            make_cap_edges(UNIFORM, 1), UNIFORM,
        )

    def test_k1_staggered_runs_midpoint_to_midpoint(self):
        """K=1 + stagger: one cell per sub-band, seam in the MIDDLE.

        User design 2026-08-29: "The cap cell should go from the midpoint
        of 1 sub-band to the midpoint of the next. There should be
        approximately the same number of cap cells and sub-bands (with
        some slight adjustment on the edges)."

        This is what makes a cap cell straddle exactly one band seam while
        keeping cell count == band count, so a band's own cell IS its
        destination cell and the dead-row birth gate (which falls through
        to the ``cap_divisor == 1`` own-cell branch of
        ``_cap_at_cap_mask``) tests the cell a birth actually lands in.
        """
        for be in (UNIFORM, RAGGED):
            stag = make_cap_edges(be, 1, stagger=True)
            mids = 0.5 * (be[:-1] + be[1:])
            # one cell per sub-band, so one more edge than there are bands
            self.assertEqual(len(stag), len(be))
            self.assertEqual(len(stag) - 1, len(be) - 1)
            # ends pinned to the band grid, interior edges are the midpoints
            self.assertAlmostEqual(stag[0], be[0])
            self.assertAlmostEqual(stag[-1], be[-1])
            np.testing.assert_allclose(stag[1:-1], mids[:-1], rtol=1e-12)
            # strictly ascending, and no cap edge sits on a band seam
            self.assertTrue(np.all(np.diff(stag) > 0))
            for e in be[1:-1]:
                self.assertGreater(np.abs(stag[1:-1] - e).min(), 1e-9 * abs(e))

    def test_k1_staggered_boundary_cell_widths(self):
        be = UNIFORM
        w_band = be[1] - be[0]
        w = np.diff(make_cap_edges(be, 1, stagger=True))
        self.assertAlmostEqual(w[0], w_band / 2, delta=1e-12 * w_band)
        self.assertAlmostEqual(w[-1], 1.5 * w_band, delta=1e-12 * w_band)
        np.testing.assert_allclose(w[1:-1], w_band, rtol=1e-9)

    def test_k1_staggered_cell_straddles_exactly_one_seam(self):
        """Interior cell i spans the top half of band i and the bottom half
        of band i+1 -- so the two sides of a seam share ONE cap cell."""
        be = UNIFORM
        stag = make_cap_edges(be, 1, stagger=True)
        for i in range(1, len(stag) - 2):
            seam = be[i]
            self.assertLess(stag[i], seam)
            self.assertGreater(stag[i + 1], seam)


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

    # K=1 STAGGER LOOKUP (landed 2026-08-29). ``make_cap_edges`` builds
    # the midpoint-to-midpoint grid and every ``cap_divisor == 1``
    # short-circuit was requalified to ``_cap_is_band_grid``
    # (``divisor == 1 AND NOT stagger``), so the band index is no longer
    # mistaken for a cell index under this grid. The CUDA twin
    # (gf_routing_kernels.cu ``gf_cap_cell_index``) carries the same
    # predicate and MUST be rebuilt with it -- python and kernel disagreeing
    # on cell membership is silent, not loud.
    def test_k1_staggered_uniform(self):
        """K=1 stagger must agree with searchsorted on the stored edges.

        The membership formula ``b*K + floor((f-lo_b)/step_b + 1/2)``
        reduces at K=1 to "band b if f is in b's LOWER half, band b+1 if
        in its UPPER half" -- which is exactly the midpoint-to-midpoint
        cell containing f. Guards the clip at the 1.5-width last cell.
        """
        self._check(UNIFORM, 1, stagger=True)

    def test_k1_staggered_ragged(self):
        self._check(RAGGED, 1, stagger=True)

    def test_k1_unstaggered_still_the_band_grid(self):
        self._check(UNIFORM, 1, stagger=False)

    def test_k1_staggered_straddle_ownership(self):
        """Just below a seam -> the NEXT band's (only) cell; just above ->
        the same cell. That shared cell IS the anti-bimodality mechanism."""
        be = UNIFORM
        mv = _shim(be, 1, stagger=True)
        w = be[1] - be[0]
        lo = np.array([be[1] - 0.25 * w])      # top half of band 0
        hi = np.array([be[1] + 0.25 * w])      # bottom half of band 1
        c_lo = int(mv._cap_cell_index(np.array([0], dtype=np.int64), lo)[0])
        c_hi = int(mv._cap_cell_index(np.array([1], dtype=np.int64), hi)[0])
        self.assertEqual(c_lo, 1)
        self.assertEqual(c_hi, 1)
        self.assertEqual(c_lo, c_hi)

    def test_staggered_straddle_ownership(self):
        """A source just below a band seam lands in the NEXT band's cell 0."""
        be, k = UNIFORM, 8
        mv = _shim(be, k, stagger=True)
        step = (be[1] - be[0]) / k
        f = np.array([be[1] - 0.25 * step])   # top half-cell of band 0
        band = np.array([0], dtype=np.int64)
        self.assertEqual(int(mv._cap_cell_index(band, f)[0]), k)  # cell 1*K


def _gate_shim(band_edges, k, stagger, ntemps=1, nwalkers=1):
    """``_shim`` plus the (temp, walker) axes the flat indexers need."""
    mv = _shim(band_edges, k, stagger)
    mv.ntemps = int(ntemps)
    mv.nwalkers = int(nwalkers)
    mv.cap_overlap_frac = 0.0
    return mv


class DeadRowGateTest(unittest.TestCase):
    """A BIRTH must be gated on the cell it LANDS IN, not on band saturation.

    THE PRODUCTION FAILURE (3-month v7, rows 5 and 6, 2026-08-29): cap cell
    2284 straddles the 1141/1142 seam at 20.381944 mHz and held BOTH
    flagship modes; ``cap_cell_leaf_cap`` was 1.0 for all 2464 cells; and
    yet 4 of 24 cold walkers held TWO leaves in that cap-1 cell, while no
    sub-band ever held more than one.

    Because a dead (birth-candidate) row was gated on
    ``_band_saturated_flat`` -- "is EVERY cap cell of my band full" -- and
    a band's ownership of cells ``b*K .. b*K+K-1`` is index arithmetic, not
    geometry. Band 1142 owns the full straddling cell 2284 AND the empty
    2285, so the band reads unsaturated and the birth is waved into a cell
    that is already at capacity.

    The fix: gate a dead row on its OWN destination cell, exactly as an
    alive row is gated. Strictly more accurate -- it forbids precisely the
    births that are impossible -- and it is what makes a straddling cap
    cell actually able to force two sides of a seam to compete.
    """

    def _full_cell_scenario(self, k):
        """Band 1's straddling cell full, its interior cell empty.

        Returns the pieces ``_cap_at_cap_mask`` takes, for two rows:
        row 0 = a DEAD birth candidate whose f0 lands in the FULL
        straddling cell; row 1 = a DEAD candidate landing in the EMPTY
        interior cell (the control -- it must stay proposable).
        """
        be = UNIFORM
        mv = _gate_shim(be, k, stagger=True)
        w = be[1] - be[0]
        straddling = 1 * k          # band 1's cell 0 straddles the 0/1 seam
        interior = 1 * k + k // 2   # comfortably inside band 1

        # f0s that actually land in those two cells
        f_full = be[1] + 0.10 * (w / k)      # just above the 0/1 seam
        f_free = be[1] + 0.55 * w            # mid-band 1
        freqs = np.array([f_full, f_free])
        bands = np.array([1, 1], dtype=np.int64)
        cells = mv._cap_cell_index(bands, freqs)
        self.assertEqual(int(cells[0]), straddling)
        self.assertEqual(int(cells[1]), interior)

        counts = np.zeros(mv.ntemps * mv.nwalkers * mv.num_cap_cells,
                          dtype=np.int64)
        cap = np.ones(mv.num_cap_cells, dtype=np.int64)
        flat = mv._cap_flat_index(
            np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), cells)
        counts[int(flat[0])] = 1          # the straddling cell is FULL

        sorter = types.SimpleNamespace(
            temp_inds=np.zeros(2, dtype=np.int64),
            walker_inds=np.zeros(2, dtype=np.int64),
            band_inds=bands,
            inds=np.array([False, False]),   # both are DEAD birth candidates
        )
        return mv, sorter, counts, cap, flat, cells

    def test_birth_into_a_full_straddling_cell_is_blocked(self):
        for k in (2, 4, 8):
            mv, sorter, counts, cap, flat, cells = self._full_cell_scenario(k)
            mask = mv._cap_at_cap_mask(sorter, counts, cap, flat, cells)
            self.assertTrue(
                bool(mask[0]),
                msg=f"K={k}: birth into a FULL straddling cell was allowed "
                    f"because the band still had an empty owned cell",
            )

    def test_birth_into_a_free_cell_of_the_same_band_still_allowed(self):
        """The control: gating on the destination cell must not over-block."""
        for k in (2, 4, 8):
            mv, sorter, counts, cap, flat, cells = self._full_cell_scenario(k)
            mask = mv._cap_at_cap_mask(sorter, counts, cap, flat, cells)
            self.assertFalse(bool(mask[1]), msg=f"K={k}")

    def test_alive_row_gating_is_unchanged(self):
        """Alive rows were ALREADY gated on their own cell -- keep it."""
        for k in (2, 4):
            mv, sorter, counts, cap, flat, cells = self._full_cell_scenario(k)
            sorter.inds = np.array([True, True])
            mask = mv._cap_at_cap_mask(sorter, counts, cap, flat, cells)
            self.assertTrue(bool(mask[0]))
            self.assertFalse(bool(mask[1]))

    def test_dead_and_alive_rows_agree_on_the_same_cell(self):
        """The whole point: one rule for both, so a cap means a cap."""
        for k in (2, 4, 8):
            mv, sorter, counts, cap, flat, cells = self._full_cell_scenario(k)
            sorter.inds = np.array([False, False])
            dead = mv._cap_at_cap_mask(sorter, counts, cap, flat, cells)
            sorter.inds = np.array([True, True])
            alive = mv._cap_at_cap_mask(sorter, counts, cap, flat, cells)
            np.testing.assert_array_equal(dead, alive)


class K1StaggerDriftTest(unittest.TestCase):
    """Does the K=1 midpoint-to-midpoint grid survive in-model DRIFT?

    Two facts collide here:

    * a source's band label is FROZEN at initial buffer fill within
      ``propose`` and is NOT re-homed when its f0 drifts across a seam
      (architectural invariant -- the label keys the residual bookkeeping);
    * an in-model repeat may move f0 by up to N/4 bins past the band edge,
      and N/2 on RJ-provenance buffers whose ``frequency_lims`` are
      pre-widened -- which at a 3-month band width of ~135 FD bins and
      N=512 is comfortably MORE THAN A WHOLE BAND.

    So the cap census routinely computes a cell from a STALE band label and
    a drifted f0. If the answer depended on the label, the drift gate would
    police the wrong cell.
    """

    def _cells(self, mv, f, band):
        return int(mv._cap_cell_index(
            np.array([band], dtype=np.int64), np.array([f]))[0])

    def test_stale_band_label_gives_the_same_cell_on_a_uniform_grid(self):
        """The whole safety argument: on a UNIFORM grid the handed band
        index cancels out of ``b + floor((f-lo_b)/w + 1/2)``, so a frozen
        label and the true label agree for ANY drift."""
        be = UNIFORM
        mv = _gate_shim(be, 1, stagger=True)
        w = be[1] - be[0]
        rng = np.random.default_rng(3)
        for _ in range(400):
            f = rng.uniform(be[0], be[-1] * (1 - 1e-12))
            true_band = int(np.clip(
                np.searchsorted(be, f, side="right") - 1, 0, mv.num_bands - 1))
            truth = int(np.clip(
                np.searchsorted(mv.cap_edges, f, side="right") - 1,
                0, mv.num_cap_cells - 1))
            # every plausible stale label: up to +/- 2 whole bands away
            for off in (-2, -1, 0, 1, 2):
                stale = int(np.clip(true_band + off, 0, mv.num_bands - 1))
                self.assertEqual(
                    self._cells(mv, f, stale), truth,
                    msg=f"f={f!r} true_band={true_band} stale={stale}")

    def test_drift_of_more_than_a_full_band_still_censuses_correctly(self):
        """N/2 bins on an RJ buffer can exceed a band width -- check the
        census follows the source rather than its frozen label."""
        be = UNIFORM
        mv = _gate_shim(be, 1, stagger=True)
        w = be[1] - be[0]
        start_band = 4
        for drift in (-1.9, -1.1, -0.6, -0.2, 0.2, 0.6, 1.1, 1.9):
            f = be[start_band] + 0.5 * w + drift * w
            truth = int(np.clip(
                np.searchsorted(mv.cap_edges, f, side="right") - 1,
                0, mv.num_cap_cells - 1))
            self.assertEqual(self._cells(mv, f, start_band), truth,
                             msg=f"drift={drift} bands")

    def test_ragged_grid_is_the_documented_limitation(self):
        """On a NON-uniform grid the extrapolation uses the stale band's
        OWN width, so a stale label CAN mis-census. Production runs a
        uniform grid (1232 bands, equal 1.7361e-5 Hz); this pins the
        caveat so a future get_n free-frequency grid trips it loudly."""
        be = RAGGED
        mv = _gate_shim(be, 1, stagger=True)
        disagreements = 0
        for b in range(mv.num_bands):
            for frac in (0.1, 0.5, 0.9):
                f = be[b] + frac * (be[b + 1] - be[b])
                truth = int(np.clip(
                    np.searchsorted(mv.cap_edges, f, side="right") - 1,
                    0, mv.num_cap_cells - 1))
                for stale in range(mv.num_bands):
                    if self._cells(mv, f, stale) != truth:
                        disagreements += 1
        self.assertGreater(
            disagreements, 0,
            "if this ever hits zero the ragged caveat is gone and this "
            "test should be replaced by the uniform guarantee")

    def test_production_band_grid_is_uniform(self):
        """The safety argument above holds because the run grid is uniform."""
        be = np.linspace(0.555555556e-3, 21.944444444e-3, 1233)
        d = np.diff(be)
        # uniform to floating-point rounding (linspace spreads the widths by
        # ~3e-18 absolute = 2e-13 relative); the cell-index extrapolation
        # needs equal widths, not bit-equal ones
        np.testing.assert_allclose(d, d[0], rtol=1e-12)

    def test_drift_gate_arms_at_k1_stagger_without_the_edge_leak_knob(self):
        """``GB_CAP_DRIFT_GATE_EDGE_LEAK`` was a workaround for the FALSE
        premise that cells==bands means f0 cannot change cell. Under the
        staggered grid the premise is correctly false by construction, so
        the gate must arm on its own -- the knob becomes moot."""
        mv = _gate_shim(UNIFORM, 1, stagger=True)
        # The short-circuit reads
        #     _cap_is_band_grid and overlap <= 0 and not edge_leak
        # so a False first term keeps the gate live whatever the knob says.
        self.assertFalse(
            mv._cap_is_band_grid,
            "K=1 + stagger must NOT be treated as the band grid, or the "
            "drift gate short-circuits off and nothing polices seam "
            "crossings",
        )

    def test_unstaggered_k1_still_short_circuits(self):
        """The historical regime is untouched: cells ARE bands there."""
        mv = _gate_shim(UNIFORM, 1, stagger=False)
        self.assertTrue(mv._cap_is_band_grid)


class EnsureCapCellFieldsTest(unittest.TestCase):
    """``ensure_cap_cell_fields`` must not use ``num_cells == num_bands``
    as a proxy for "the cap grid is the band grid".

    At K=1 + stagger the counts are EQUAL (1232 cells over 1232 sub-bands)
    while membership is shifted half a sub-band, so the cap-cell arrays are
    genuinely needed. The count proxy skipped allocating them, and
    ``_cap_state_arrays`` -- which correctly takes the cell branch under
    ``_cap_is_band_grid`` -- then read ``bi["cap_cell_leaf_cap"]`` on a
    dict that had none. KeyError at move construction, on the first
    propose of the relaunch.
    """

    def _bi(self, nbands=10, nwalkers=4):
        return {"num_bands": nbands, "nwalkers": nwalkers}

    def test_allocates_when_counts_match_but_grid_is_staggered(self):
        bi = self._bi()
        ensure_cap_cell_fields(bi, 10, staggered=True)
        for k in ("cap_cell_leaf_cap", "cap_cell_iters", "cap_cell_best_ll"):
            self.assertIn(k, bi, msg=f"{k} missing at K=1 + stagger")
            self.assertEqual(np.shape(bi[k]), (10,))
        self.assertEqual(np.shape(bi["cap_cell_cold_ll"]), (4, 10))

    def test_sentinels_are_the_documented_ones(self):
        bi = self._bi()
        ensure_cap_cell_fields(bi, 10, staggered=True)
        np.testing.assert_array_equal(bi["cap_cell_leaf_cap"],
                                      np.full(10, -1))
        np.testing.assert_array_equal(bi["cap_cell_iters"], np.zeros(10))
        self.assertTrue(np.all(np.isneginf(bi["cap_cell_best_ll"])))

    def test_band_grid_still_short_circuits(self):
        """divisor 1 WITHOUT stagger keeps reading the band_* arrays."""
        bi = self._bi()
        ensure_cap_cell_fields(bi, 10, staggered=False)
        self.assertNotIn("cap_cell_leaf_cap", bi)

    def test_default_preserves_the_historical_behaviour(self):
        bi = self._bi()
        ensure_cap_cell_fields(bi, 10)
        self.assertNotIn("cap_cell_leaf_cap", bi)

    def test_more_cells_than_bands_always_allocates(self):
        for staggered in (False, True):
            bi = self._bi()
            ensure_cap_cell_fields(bi, 40, staggered=staggered)
            self.assertEqual(np.shape(bi["cap_cell_leaf_cap"]), (40,))


class CudaTwinPredicateTest(unittest.TestCase):
    """The CUDA cell lookup must short-circuit on the SAME predicate.

    ``gf_routing_kernels.cu::gf_cap_cell_index`` mirrors
    ``_cap_cell_index``. If one of them treats the band index as the cell
    index and the other does not, nothing raises -- the two sides simply
    census a source into different cells, and the caps enforce against a
    census that disagrees with the router. There is no numeric parity test
    (the kernel needs a built backend), so this pins the source text, the
    same technique the band-unit scan-order suite uses on the drift gate.
    """

    def _cu(self):
        import pathlib
        import lisatools
        p = (pathlib.Path(lisatools.__file__).parent
             / "cutils" / "gf_routing_kernels.cu")
        self.assertTrue(p.is_file(), f"missing {p}")
        return p.read_text()

    def test_kernel_short_circuit_requires_not_stagger(self):
        src = self._cu()
        self.assertIn("cap_divisor == 1 && !cap_stagger", src)

    def test_kernel_has_no_bare_divisor1_short_circuit(self):
        """A bare ``cap_divisor == 1`` return would silently re-break K=1."""
        src = self._cu()
        self.assertNotIn("if (cap_divisor == 1) {", src)

    def test_python_predicate_is_the_named_property(self):
        """Python must route through ``_cap_is_band_grid``, not the divisor."""
        import inspect
        src = inspect.getsource(GBSpecialBase._cap_cell_index)
        self.assertIn("_cap_is_band_grid", src)
        self.assertNotIn("cap_divisor == 1", src)

    def test_property_is_divisor_and_stagger(self):
        for divisor, stagger, expected in (
            (1, False, True),    # the historical band-grid regime
            (1, True, False),    # midpoint-to-midpoint: cells != bands
            (2, True, False),
            (8, False, False),
        ):
            mv = _shim(UNIFORM, divisor, stagger)
            self.assertIs(mv._cap_is_band_grid, expected,
                          msg=f"divisor={divisor} stagger={stagger}")


class CapBudgetTransitionTest(unittest.TestCase):
    """The scheduler's finish budget must move on the SAME rule as the gate.

    The pick pool gates a dead row on the cell its birth lands in, so the
    budget transitions have to be own-cell transitions. If they stayed
    band-level the scheduler would hold rows it will never hand out (band
    unsaturated, destination cell full) and fail to release rows it should
    (cell freed inside a band that was never fully saturated).
    """

    def _t(self, pre, post, cap, alive):
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        flat = np.arange(len(pre))
        return GBSpecialBase._cap_budget_transitions(
            pre, post, flat, np.asarray(cap, dtype=np.int64),
            np.asarray(alive, dtype=bool),
        )

    def test_death_from_at_cap_frees_the_cell(self):
        freed, capped = self._t([2], [1], [2], [True])
        self.assertTrue(bool(freed[0]))
        self.assertFalse(bool(capped[0]))

    def test_death_from_below_cap_frees_nothing(self):
        freed, capped = self._t([1], [0], [2], [True])
        self.assertFalse(bool(freed[0]))

    def test_birth_that_fills_the_cell_caps_it(self):
        freed, capped = self._t([1], [2], [2], [False])
        self.assertTrue(bool(capped[0]))
        self.assertFalse(bool(freed[0]))

    def test_birth_leaving_headroom_does_not_cap(self):
        freed, capped = self._t([0], [1], [2], [False])
        self.assertFalse(bool(capped[0]))

    def test_reduces_to_the_historical_divisor1_expressions(self):
        """Exhaustive equivalence with the pre-2026-08-29 divisor-1 form."""
        for cap in (1, 2, 3, 5):
            for pre in range(0, cap + 2):
                for alive in (True, False):
                    post = pre - 1 if alive else pre + 1
                    freed, capped = self._t([pre], [post], [cap], [alive])
                    old_freed = alive and (pre == cap)
                    # the old birth form lacked the `pre < cap` term; it is
                    # implied because the gate never offers a full cell
                    old_capped = (not alive) and (pre + 1 >= cap)
                    self.assertEqual(bool(freed[0]), old_freed,
                                     msg=f"cap={cap} pre={pre} alive={alive}")
                    if pre < cap:
                        self.assertEqual(
                            bool(capped[0]), old_capped,
                            msg=f"cap={cap} pre={pre} alive={alive}")

    def test_a_cap1_straddling_cell_caps_on_the_first_birth(self):
        """The v7 case: cap 1, empty cell, one birth -> capped immediately,
        so the SECOND side of the seam finds no headroom."""
        freed, capped = self._t([0], [1], [1], [False])
        self.assertTrue(bool(capped[0]))


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


class CapDriftGateTest(unittest.TestCase):
    """Setup/veto logic of the in-model CAP DRIFT GATE (2026-08-20)."""

    def _gate_shim(self, k=4, stagger=False, ntemps=2, nwalkers=3):
        import types
        be = UNIFORM
        mv = _shim(be, k, stagger)
        mv.cap_drift_gate = True
        mv._f0_col = 1
        mv.ntemps, mv.nwalkers = ntemps, nwalkers
        nb = mv.num_bands
        # 4 alive sources for walker 0, temp 0: three in cell 0, one in cell 5
        f_cell0 = be[0] + 1e-7
        f_cell5 = be[0] + (5.25 if not stagger else 5.75) * (be[1] - be[0]) / k
        n = 4
        sorter = types.SimpleNamespace(
            band_inds=np.array([0, 0, 0, 1]),
            temp_inds=np.zeros(n, dtype=int),
            walker_inds=np.zeros(n, dtype=int),
            freqs=np.array([f_cell0, f_cell0, f_cell0, f_cell5]),
            inds=np.ones(n, dtype=bool),
        )
        caps = np.full(mv.num_cap_cells, 1.0)
        mv._cap_leaf_cap = caps
        return mv, sorter, caps

    def test_setup_off_switch(self):
        mv, sorter, caps = self._gate_shim()
        mv.cap_drift_gate = False
        self.assertIsNone(mv._cap_drift_gate_setup(sorter))
        mv.cap_drift_gate = True
        mv._cap_leaf_cap = None
        self.assertIsNone(mv._cap_drift_gate_setup(sorter))
        mv._cap_leaf_cap = np.full(len(caps), -1.0)  # all disarmed
        self.assertIsNone(mv._cap_drift_gate_setup(sorter))

    def test_setup_census(self):
        mv, sorter, _ = self._gate_shim()
        counts, cap_dev = mv._cap_drift_gate_setup(sorter)
        occ = counts.reshape(mv.ntemps, mv.nwalkers, mv.num_cap_cells)
        self.assertEqual(int(occ[0, 0, 0]), 3)
        self.assertEqual(int(occ[0, 0, 5]), 1)
        self.assertEqual(int(counts.sum()), 4)
        self.assertEqual(len(cap_dev), mv.num_cap_cells)

    def test_veto_semantics(self):
        mv, sorter, _ = self._gate_shim()
        counts, cap = mv._cap_drift_gate_setup(sorter)
        t = np.zeros(3, dtype=int); w = np.zeros(3, dtype=int)
        cell_c = np.array([5, 0, 0])   # current cells
        cell_n = np.array([0, 0, 6])   # -> into full cell 0; stay; into empty 6
        cross = cell_n != cell_c
        flat_n = mv._cap_flat_index(t, w, cell_n)
        veto = cross & (cap[cell_n] >= 0) & (counts[flat_n] >= cap[cell_n])
        # into over-full cell 0: vetoed; within-cell: never; into empty: allowed
        self.assertTrue(bool(veto[0]))
        self.assertFalse(bool(veto[1]))
        self.assertFalse(bool(veto[2]))
        # DRAIN property: a source leaving over-full cell 0 for empty cell 2
        veto_out = (np.array([True]) & (cap[[2]] >= 0)
                    & (counts[mv._cap_flat_index(t[:1], w[:1], np.array([2]))]
                       >= cap[[2]]))
        self.assertFalse(bool(veto_out[0]))

    def test_scatter_add_duplicates(self):
        counts = np.zeros(8, dtype=np.int64)
        GBSpecialBase._cap_gate_scatter_add(
            counts, np.array([3, 3, 5]), np.array([1, 1, -1], dtype=np.int64))
        self.assertEqual(int(counts[3]), 2)
        self.assertEqual(int(counts[5]), -1)


# ----------------------------------------------------------------------
# THE SEAM-STRADDLING CELL: why v7/v8 run (divisor=2, stagger=1).
# ----------------------------------------------------------------------
#: v7 band grid: GB_SUBBAND_DIVISOR=8 -> uniform sub-bands of 135 FD bins.
V7_BAND_BINS = 135
#: 3-month observation -> the FD bin width the band grid is expressed in.
V7_DF = 1.0 / (90.0 * 86400.0)
#: the flagship bimodal source's band pair (measured, 2026-08-29).
V7_SEAM_BAND = 1142


def _v7_band_edges(num_bands=1400):
    """A uniform band grid with v7's 135-FD-bin sub-bands."""
    return 1e-3 + np.arange(num_bands + 1) * (V7_BAND_BINS * V7_DF)


class SeamStraddlingCellTest(unittest.TestCase):
    """The 3-month v7 bimodality mechanism, pinned so it cannot be lost.

    MEASURED FINDING (2026-08-29, v7 row 5, flagship 20.380377 mHz). The
    flagship's leaves split across the band 1141/1142 seam -- which sits
    at +12.19 FD bins from the flagship -- with an empty gap between the
    two populations. Cap-cell membership of those actual leaves::

        v7    (divisor=1, stagger=0):
           cell 1141  [-122.8, +12.2] bins   below-seam 24   above-seam  0
           cell 1142  [ +12.2, +147.2] bins  below-seam  0   above-seam 22

        probe (divisor=2, stagger=1):
           cell 2284  [ -21.6,  +45.9] bins  below-seam 24   above-seam 22

    Under the ALIGNED grid each band carries an INDEPENDENT cap, so the
    two modes never compete and eight leaves can sit across the seam
    forever. Under the STAGGERED grid both modes fall in ONE cell and
    compete for one cap, which is what gives the RJ death move direct
    pressure to kill the weaker side. The dedicated high-f probe, which
    ran (2, 1), solved this on the same band with a bit-identical BAND
    grid -- so the cap grid, not the band grid, is the discriminator.

    This compounds with the block-Gibbs scan: bands 1141/1142 are residue
    classes 7 and 8 mod GB_BAND_UNIT_STRIDE=9 and are therefore never
    co-open, so no single move can even see both modes at once. The
    straddling cell is what reaches across that.

    DO NOT "FIX" THE ALIGNMENT BACK. Aligning cap edges to band edges is
    exactly the configuration that cannot break this degeneracy.
    """

    def test_two_modes_across_a_seam_share_one_cell_only_when_staggered(self):
        """THE MECHANISM: same cell at (2, 1), different cells at (1, 0)."""
        be = _v7_band_edges()
        seam = be[V7_SEAM_BAND]
        # one leaf either side of the band seam, well inside their bands
        f_below = seam - 5.0 * V7_DF
        f_above = seam + 5.0 * V7_DF
        freqs = np.array([f_below, f_above])
        bands = np.array([V7_SEAM_BAND - 1, V7_SEAM_BAND])

        staggered = _shim(be, 2, True)._cap_cell_index(bands, freqs)
        self.assertEqual(
            int(staggered[0]), int(staggered[1]),
            msg="(divisor=2, stagger=1) MUST map two leaves either side of a "
                "band seam into ONE cap cell -- that shared cap is the only "
                "thing that makes the two modes compete.",
        )
        # and it is the straddling cell, index b*K of the UPPER band
        self.assertEqual(int(staggered[0]), V7_SEAM_BAND * 2)

        aligned = _shim(be, 1, False)._cap_cell_index(bands, freqs)
        self.assertNotEqual(
            int(aligned[0]), int(aligned[1]),
            msg="(divisor=1, stagger=0) files them under independent caps -- "
                "the configuration that produced the persistent bimodality.",
        )
        self.assertEqual([int(c) for c in aligned], list(bands))

    def test_straddling_cell_reaches_half_a_cell_either_side_of_the_seam(self):
        """The measured [-21.6, +45.9]-bin span, from the stored edges."""
        be = _v7_band_edges()
        mv = _shim(be, 2, True)
        seam = be[V7_SEAM_BAND]
        cell = V7_SEAM_BAND * 2
        lo, hi = mv.cap_edges[cell], mv.cap_edges[cell + 1]
        half_cell_bins = V7_BAND_BINS / 2.0 / 2.0          # band/K/2 = 33.75
        self.assertAlmostEqual((seam - lo) / V7_DF, half_cell_bins, places=6)
        self.assertAlmostEqual((hi - seam) / V7_DF, half_cell_bins, places=6)
        # the flagship sits 12.19 bins BELOW the seam; reproduce the
        # measured span relative to it
        flagship = seam - 12.19 * V7_DF
        self.assertAlmostEqual((lo - flagship) / V7_DF, -21.56, places=2)
        self.assertAlmostEqual((hi - flagship) / V7_DF, +45.94, places=2)

    def test_only_the_staggered_grid_has_no_cap_edge_on_a_band_seam(self):
        """Aligned grids put a cap edge on EVERY seam; staggered on none."""
        be = _v7_band_edges(num_bands=40)
        for k, stagger, want_shared in ((1, False, True), (2, False, True),
                                        (2, True, False)):
            ce = _shim(be, k, stagger).cap_edges
            shared = np.abs(be[:, None] - ce[None, :]).min(axis=1) <= 1e-15
            # interior seams only (the two grid ends always coincide)
            got_shared = bool(shared[1:-1].any())
            self.assertEqual(
                got_shared, want_shared,
                msg=f"(divisor={k}, stagger={stagger}): interior band seams "
                    f"sharing a cap edge = {got_shared}, expected "
                    f"{want_shared}.",
            )

    def test_staggered_lookup_ignores_the_handed_band_on_a_uniform_grid(self):
        """At (2, 1) on a UNIFORM grid the handed band index is irrelevant.

        The staggered branch of ``_cap_cell_index`` applies NO per-band
        clip (only the global cell range), so ``b*K + floor((f - lo_b)/
        step_b + 1/2)`` is algebraically the SAME number for every band
        index handed in when the bands are equal width: the ``b*K`` terms
        cancel. An out-of-band destination frequency therefore resolves to
        its TRUE cell instead of being folded back into the source band's
        boundary cell -- the nested grid's behaviour, pinned in the next
        test. So GB_CAP_DEST_BAND cannot change the answer here (it still
        matters on a ragged ``get_n`` grid, where the steps differ).

        EXACT CAP EDGES ARE THE ONE EXCEPTION (measured 2026-08-29): the
        cancellation is exact in real arithmetic but not in floating
        point, and a frequency sitting exactly ON a cap edge makes
        ``(f - lo_b)/step_b + 1/2`` an integer, so the two band
        references can round the tie to different sides and disagree by
        one cell. Measure-zero, but it is a reason to keep
        GB_CAP_DEST_BAND=1 even on a uniform grid: resolving the band
        from f0 first reproduces ``searchsorted(cap_edges)`` on the ties
        too, which the source-attributed lookup does not.
        """
        be = _v7_band_edges(num_bands=40)
        mv = _shim(be, 2, True)
        mv.band_edges = be
        seam_band = 20
        ce = mv.cap_edges
        n_ties = 0
        for d_bins in np.linspace(-0.9 * V7_BAND_BINS, 0.9 * V7_BAND_BINS, 37):
            f = np.array([be[seam_band] + d_bins * V7_DF])
            want = int(np.searchsorted(ce, f[0], side="right") - 1)
            on_edge = np.abs(ce - f[0]).min() / V7_DF < 1e-6
            # GB_CAP_DEST_BAND=1 is exact everywhere, ties included
            self.assertEqual(
                int(mv._cap_cell_index(np.array([seam_band - 1]), f,
                                       resolve_band=True)[0]),
                want, msg=f"resolve_band d={d_bins}",
            )
            if on_edge:
                n_ties += 1
                continue
            for handed in (seam_band - 1, seam_band):
                self.assertEqual(
                    int(mv._cap_cell_index(np.array([handed]), f)[0]), want,
                    msg=f"d={d_bins} handed={handed}",
                )
        self.assertGreater(n_ties, 0, "sampling never hit a cap edge")

    def test_nested_grid_folds_an_out_of_band_frequency_back(self):
        """Contrast: the NESTED branch DOES clip ``sub`` into ``[0, K-1]``.

        Pins that the clip the stagger branch avoids is real, so nobody
        reads the nested docstring and assumes it applies at (2, 1).
        """
        be = _v7_band_edges(num_bands=40)
        mv = _shim(be, 2, False)
        seam_band = 20
        f_above = np.array([be[seam_band] + 10.0 * V7_DF])   # in band 20
        folded = int(mv._cap_cell_index(np.array([seam_band - 1]), f_above)[0])
        # charged to band 19's TOP cell, not to where it actually sits
        self.assertEqual(folded, (seam_band - 1) * 2 + 1)
        self.assertNotEqual(
            folded,
            int(np.searchsorted(mv.cap_edges, f_above[0], side="right") - 1),
        )


if __name__ == "__main__":
    unittest.main()
