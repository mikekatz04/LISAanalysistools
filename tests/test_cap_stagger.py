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
