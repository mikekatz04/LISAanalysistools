# -*- coding: utf-8 -*-
"""RJ/in-model sub-band semantics (user contract 2026-08-29).

The contract, in the user's words:

    "I want the caps to perfectly align with the sub-bands. no overlap.
    No N/4 outside perfectly. The N/4 is just the limits the in-model
    moves can take outside of their band. The cap and RJ apply within
    the sub-band limits only, not N/4 outside. We want to have RJ follow
    the cap exactly in its sub-band. We want in-model to allow movement
    across the band edge up to N/4 outside and we only want to allow
    this if the current cap count in the neighboring sub-band is
    < cap + 2."

    "RJ should be within sub-band. in-model can go across."

Four requirements, and where each stands:

1. cap cells == sub-bands exactly -- ALREADY TRUE (GB_CAP_DIVISOR=1,
   GB_CAP_OVERLAP_FRAC=0, GB_CAP_STAGGER=0).
2. RJ births confined to the sub-band -- ALREADY TRUE BY CONSTRUCTION,
   and NO code change was made for it. The draw is global; candidates are
   then divided up and assigned to sub-bands by their own drawn f0
   (``band_inds = searchsorted(band_edges, freqs)``, gbbands), which
   applies to dead rows too. So a birth is inside the band it is assigned
   to by construction, and the RJ support gate is a tautology for it.
   ``f0`` is never rewritten afterwards on the birth path.
3. in-model MAY cross the edge, up to N/4 outside -- NOT enforced before
   this change: the window was ~1080x too wide (see
   ``UnitCollisionCharacterisationTest``). Fixed by
   ``GB_BAND_WINDOW_STRICT`` (default "0"; v7 arms it).
4. a cross-edge in-model move is allowed only while the DESTINATION
   sub-band's occupancy is under cap + headroom -- the destination was
   never actually consulted at divisor 1 (the cell lookup returned the
   SOURCE band and never read f0), so the veto was a tautology. Fixed by
   ``GB_CAP_DEST_BAND`` (default "1"; user: "always from the candidate
   f0"). ``<= cap + 2`` already matched the user's ruling and is
   unchanged.

Band ASSIGNMENT is frozen for the whole propose and that is deliberate
and residual-critical -- see the gate-site comment in gbspecialstretch.
Nothing here relabels a source.
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbbands import rj_band_window
from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

# ---------------------------------------------------------------- geometry
# The v7 3-month production grid, to scale:
#   TOBS_TARGET=7776000, WDM layer_df = 1/(2*Nf*dt) = 1/7200 Hz,
#   GB_SUBBAND_DIVISOR=8  ->  sub-band width = layer_df/8.
TOBS = 7776000.0
LAYER_DF = 1.0 / 7200.0          # SubBandBuffer.df on the WDM path
DF_MOVE = 1.0 / TOBS             # GBSpecialBase.df on the WDM path
SUBBAND_W = LAYER_DF / 8.0       # GB_SUBBAND_DIVISOR=8
N_FLAGSHIP = 512                 # get_N(1e-30, 10 mHz, TOBS, oversample=2)

# 6 sub-bands starting at 10 mHz, on the real k*layer_df/8 grid.
K0 = int(round(10e-3 / SUBBAND_W))
BAND_EDGES = np.array([(K0 + i) * SUBBAND_W for i in range(7)])
NUM_BANDS = len(BAND_EDGES) - 1
BAND_N_VALS = np.full(NUM_BANDS, N_FLAGSHIP, dtype=int)


class _Knob:
    """Set/restore GB_BAND_WINDOW_STRICT around a test."""

    def setUp(self):
        self._old = os.environ.pop("GB_BAND_WINDOW_STRICT", None)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_BAND_WINDOW_STRICT", None)
        else:
            os.environ["GB_BAND_WINDOW_STRICT"] = self._old


def _move(cap_divisor=1, ntemps=1, nwalkers=1, band_edges=BAND_EDGES):
    """A GB move with ONLY the cap-grid attributes filled in."""
    from lisatools.globalfit.state import make_cap_edges

    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.name = "test"
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"
    m.band_edges = np.asarray(band_edges, dtype=float)
    m.num_bands = len(band_edges) - 1
    m.ntemps = ntemps
    m.nwalkers = nwalkers
    m.cap_divisor = max(1, int(cap_divisor))
    m.cap_stagger = False
    m.cap_overlap_frac = 0.0
    m.num_cap_cells = m.num_bands * m.cap_divisor
    m.cap_edges = np.asarray(make_cap_edges(m.band_edges, m.cap_divisor))
    m._cap_band_lo = m.band_edges[:-1]
    m._cap_band_step = (
        (m.band_edges[1:] - m.band_edges[:-1]) / m.cap_divisor
    )
    m._cap_leaf_cap = None
    m._band_leaf_cap = None
    m._f0_col = 1
    return m


# =====================================================================
# REQUIREMENT 2 -- RJ strictly inside the sub-band
# =====================================================================
class RJStrictSubBandWindowTest(_Knob, unittest.TestCase):
    """``frequency_lims`` on an RJ-provenance buffer IS the RJ support gate
    (gbspecialstretch ``curr_logp[(~alive) & out_of_band] = -inf``), so the
    window this returns is exactly what confines an RJ birth."""

    def test_armed_window_is_exactly_the_sub_band(self):
        os.environ["GB_BAND_WINDOW_STRICT"] = "1"
        band_inds = np.arange(NUM_BANDS)
        lo, hi = rj_band_window(
            BAND_EDGES, BAND_N_VALS, band_inds, LAYER_DF, is_rj=True)
        np.testing.assert_array_equal(lo, BAND_EDGES[:-1])
        np.testing.assert_array_equal(hi, BAND_EDGES[1:])

    def test_armed_rejects_a_birth_just_outside_the_edge(self):
        """An RJ birth one FD bin past the band edge must be out of support."""
        os.environ["GB_BAND_WINDOW_STRICT"] = "1"
        b = np.array([2])
        lo, hi = rj_band_window(
            BAND_EDGES, BAND_N_VALS, b, LAYER_DF, is_rj=True)
        just_below = BAND_EDGES[2] - DF_MOVE
        just_above = BAND_EDGES[3] + DF_MOVE
        self.assertTrue(just_below < lo[0], "birth below the edge must be out")
        self.assertTrue(just_above > hi[0], "birth above the edge must be out")
        # and a birth INSIDE stays in support
        inside = 0.5 * (BAND_EDGES[2] + BAND_EDGES[3])
        self.assertFalse(inside < lo[0] or inside > hi[0])

    def test_armed_leaves_the_non_rj_buffer_untouched(self):
        """In-model-provenance buffers were never widened; still aren't."""
        os.environ["GB_BAND_WINDOW_STRICT"] = "1"
        b = np.arange(NUM_BANDS)
        lo_a, hi_a = rj_band_window(
            BAND_EDGES, BAND_N_VALS, b, LAYER_DF, is_rj=False)
        os.environ.pop("GB_BAND_WINDOW_STRICT")
        lo_b, hi_b = rj_band_window(
            BAND_EDGES, BAND_N_VALS, b, LAYER_DF, is_rj=False)
        np.testing.assert_array_equal(lo_a, lo_b)
        np.testing.assert_array_equal(hi_a, hi_b)
        np.testing.assert_array_equal(lo_a, BAND_EDGES[:-1])


class KnobOffIsBitIdenticalTest(_Knob, unittest.TestCase):
    """Knob off == the historical expression, to the bit."""

    def test_off_reproduces_the_legacy_widening(self):
        b = np.arange(NUM_BANDS)
        lo, hi = rj_band_window(
            BAND_EDGES, BAND_N_VALS, b, LAYER_DF, is_rj=True)
        legacy_lo = BAND_EDGES[:-1] - BAND_N_VALS * LAYER_DF / 4
        legacy_hi = BAND_EDGES[1:] + BAND_N_VALS * LAYER_DF / 4
        np.testing.assert_array_equal(lo, legacy_lo)
        np.testing.assert_array_equal(hi, legacy_hi)


class UnitCollisionCharacterisationTest(_Knob, unittest.TestCase):
    """VERDICT B, pinned as a characterisation test.

    The legacy widening is computed in the BUFFER's df (``layer_df`` on
    WDM) while the consuming gates divide by the MOVE's df (``1/Tobs``).
    The overstatement factor is exactly ``layer_df * Tobs == Nt/2``.
    """

    def test_legacy_widening_is_Nt_over_2_times_too_wide(self):
        b = np.array([0])
        lo, hi = rj_band_window(
            BAND_EDGES, BAND_N_VALS, b, LAYER_DF, is_rj=True)
        widen_hz = float(BAND_EDGES[0] - lo[0])
        # what the consuming gate believes it got, in MOVE bins
        widen_bins = widen_hz / DF_MOVE
        intended_bins = N_FLAGSHIP / 4
        self.assertAlmostEqual(
            widen_bins / intended_bins, LAYER_DF * TOBS, places=6)
        self.assertAlmostEqual(LAYER_DF * TOBS, 1080.0, places=6)
        # N=512: intended 128 bins (0.95 sub-bands), actual 138,240 bins
        # = 1024 sub-bands -- i.e. 0.0178 Hz, essentially the whole
        # 3-21 mHz analysis band, so the window is effectively unbounded.
        self.assertAlmostEqual(widen_bins, 138240.0, places=3)
        self.assertAlmostEqual(widen_hz / SUBBAND_W, 1024.0, places=3)

    def test_armed_removes_the_unit_mixing_entirely(self):
        os.environ["GB_BAND_WINDOW_STRICT"] = "1"
        b = np.array([0])
        lo, hi = rj_band_window(
            BAND_EDGES, BAND_N_VALS, b, LAYER_DF, is_rj=True)
        # no term computed in buffer-df units survives
        self.assertEqual(float(lo[0]), float(BAND_EDGES[0]))
        self.assertEqual(float(hi[0]), float(BAND_EDGES[1]))


# =====================================================================
# REQUIREMENTS 3 + 4 -- in-model may cross, gated on the DESTINATION
# =====================================================================
class DestinationBandResolutionTest(_Knob, unittest.TestCase):
    """VERDICT C: at cap_divisor == 1 ``_cap_cell_index`` returns the
    PASSED band index and never reads f0, so a cross-edge proposal was
    scored against its SOURCE band. ``resolve_band=True`` makes the
    lookup answer 'which band does this frequency land in'."""

    def test_divisor1_default_still_returns_the_source_band(self):
        m = _move(1)
        src = np.array([2])
        f_in_band_4 = 0.5 * (BAND_EDGES[4] + BAND_EDGES[5])
        out = m._cap_cell_index(src, np.array([f_in_band_4]))
        self.assertEqual(int(out[0]), 2)  # source-attributed (legacy)

    def test_resolve_band_returns_the_destination_band(self):
        m = _move(1)
        src = np.array([2])
        f_in_band_4 = 0.5 * (BAND_EDGES[4] + BAND_EDGES[5])
        out = m._cap_cell_index(
            src, np.array([f_in_band_4]), resolve_band=True)
        self.assertEqual(int(out[0]), 4)  # DESTINATION

    def test_resolve_band_clips_outside_the_grid(self):
        m = _move(1)
        src = np.array([2, 2])
        f = np.array([BAND_EDGES[0] - 1.0, BAND_EDGES[-1] + 1.0])
        out = m._cap_cell_index(src, f, resolve_band=True)
        np.testing.assert_array_equal(out, [0, NUM_BANDS - 1])

    def test_resolve_band_agrees_with_the_sorter_for_in_band_rows(self):
        """A row that has NOT moved must resolve to its own band, so the
        census and the gate cannot disagree about a stationary source."""
        m = _move(1)
        f = np.array([0.5 * (BAND_EDGES[i] + BAND_EDGES[i + 1])
                      for i in range(NUM_BANDS)])
        src = np.arange(NUM_BANDS)
        out = m._cap_cell_index(src, f, resolve_band=True)
        np.testing.assert_array_equal(out, src)

    def test_dest_band_knob_defaults_on(self):
        """User ruling: the destination 'should always be from the
        candidate f0'. Ships ON; GB_CAP_DEST_BAND=0 is the escape hatch."""
        from lisatools.globalfit.moves.gbspecialstretch import _cap_dest_band

        old = os.environ.pop("GB_CAP_DEST_BAND", None)
        try:
            self.assertTrue(_cap_dest_band())
            os.environ["GB_CAP_DEST_BAND"] = "0"
            self.assertFalse(_cap_dest_band())
        finally:
            os.environ.pop("GB_CAP_DEST_BAND", None)
            if old is not None:
                os.environ["GB_CAP_DEST_BAND"] = old

    def test_members_forwards_resolve_band(self):
        m = _move(1)
        src = np.array([2])
        f = np.array([0.5 * (BAND_EDGES[4] + BAND_EDGES[5])])
        p, nb, hn = m._cap_cell_members(src, f, resolve_band=True)
        self.assertEqual(int(p[0]), 4)
        self.assertIsNone(nb)


class CrossEdgeHeadroomTest(_Knob, unittest.TestCase):
    """Requirement 4 on the resolved destination: allowed while the
    DESTINATION sub-band holds < cap + headroom, vetoed at/over."""

    def setUp(self):
        _Knob.setUp(self)
        self._old_h = os.environ.pop("GB_CAP_INMODEL_HEADROOM", None)

    def tearDown(self):
        if self._old_h is not None:
            os.environ["GB_CAP_INMODEL_HEADROOM"] = self._old_h
        else:
            os.environ.pop("GB_CAP_INMODEL_HEADROOM", None)
        _Knob.tearDown(self)

    def _veto(self, dest_count, cap_val=1, src_band=2, dest_band=3):
        m = _move(1)
        cap = np.full(m.num_cap_cells, cap_val, dtype=int)
        counts = np.zeros(m.ntemps * m.nwalkers * m.num_cap_cells,
                          dtype=np.int32)
        counts[dest_band] = dest_count
        t = np.zeros(1, dtype=np.int64)
        w = np.zeros(1, dtype=np.int64)
        f_src = np.array([0.5 * (BAND_EDGES[src_band]
                                 + BAND_EDGES[src_band + 1])])
        f_dst = np.array([0.5 * (BAND_EDGES[dest_band]
                                 + BAND_EDGES[dest_band + 1])])
        b = np.array([src_band])
        cur = m._cap_cell_members(b, f_src, resolve_band=True)
        new = m._cap_cell_members(b, f_dst, resolve_band=True)
        return bool(m._cap_new_entry_veto(counts, cap, t, w, cur, new)[0])

    def test_destination_under_headroom_is_allowed(self):
        self.assertFalse(self._veto(dest_count=1))   # at cap
        self.assertFalse(self._veto(dest_count=2))   # cap + 1

    def test_destination_at_cap_plus_headroom_is_vetoed(self):
        self.assertTrue(self._veto(dest_count=3))    # cap + 2

    def test_the_destination_not_the_source_is_consulted(self):
        """Fill the SOURCE band to bursting and leave the destination
        empty: the move must still be allowed."""
        m = _move(1)
        cap = np.full(m.num_cap_cells, 1, dtype=int)
        counts = np.zeros(m.ntemps * m.nwalkers * m.num_cap_cells,
                          dtype=np.int32)
        counts[2] = 99            # SOURCE band jammed
        counts[3] = 0             # destination empty
        t = np.zeros(1, dtype=np.int64)
        w = np.zeros(1, dtype=np.int64)
        b = np.array([2])
        f_src = np.array([0.5 * (BAND_EDGES[2] + BAND_EDGES[3])])
        f_dst = np.array([0.5 * (BAND_EDGES[3] + BAND_EDGES[4])])
        cur = m._cap_cell_members(b, f_src, resolve_band=True)
        new = m._cap_cell_members(b, f_dst, resolve_band=True)
        self.assertFalse(bool(
            m._cap_new_entry_veto(counts, cap, t, w, cur, new)[0]))

    def test_a_within_band_move_never_vetoes(self):
        """Both endpoints in the same band -> not a foreign cell."""
        m = _move(1)
        cap = np.full(m.num_cap_cells, 1, dtype=int)
        counts = np.zeros(m.ntemps * m.nwalkers * m.num_cap_cells,
                          dtype=np.int32)
        counts[2] = 99
        t = np.zeros(1, dtype=np.int64)
        w = np.zeros(1, dtype=np.int64)
        b = np.array([2])
        f_a = np.array([BAND_EDGES[2] + 0.25 * SUBBAND_W])
        f_b = np.array([BAND_EDGES[2] + 0.75 * SUBBAND_W])
        cur = m._cap_cell_members(b, f_a, resolve_band=True)
        new = m._cap_cell_members(b, f_b, resolve_band=True)
        self.assertFalse(bool(
            m._cap_new_entry_veto(counts, cap, t, w, cur, new)[0]))

    def test_legacy_source_attribution_cannot_veto_a_cross_edge_move(self):
        """Documents the bug: WITHOUT resolve_band the destination and
        source cells are equal by construction, so the veto is a
        tautology that can never fire at divisor 1."""
        m = _move(1)
        cap = np.full(m.num_cap_cells, 1, dtype=int)
        counts = np.zeros(m.ntemps * m.nwalkers * m.num_cap_cells,
                          dtype=np.int32)
        counts[3] = 999           # destination band massively over cap
        t = np.zeros(1, dtype=np.int64)
        w = np.zeros(1, dtype=np.int64)
        b = np.array([2])
        f_src = np.array([0.5 * (BAND_EDGES[2] + BAND_EDGES[3])])
        f_dst = np.array([0.5 * (BAND_EDGES[3] + BAND_EDGES[4])])
        cur = m._cap_cell_members(b, f_src)      # legacy
        new = m._cap_cell_members(b, f_dst)      # legacy
        self.assertFalse(bool(
            m._cap_new_entry_veto(counts, cap, t, w, cur, new)[0]))


# =====================================================================
# REGRESSION -- e79dbd7c deleted locals the accept path still reads
# =====================================================================
class NoDeadNamesTest(unittest.TestCase):
    """``_run_in_model_repeats`` must not read a local it never stores.

    e79dbd7c replaced the inline cap-drift veto with
    ``_cap_new_entry_veto`` and deleted the block that defined
    ``_dg_cell_c`` / ``_dg_cross`` / ``_dg_flat_n`` (and the overlap
    branch's ``_c_p`` / ``_n_p`` ...), but left the ACCEPT-side occupancy
    scatter referencing them. Any in-model block with the drift gate
    armed -- exactly the v7 configuration (GB_CAP_DRIFT_GATE=1 +
    GB_CAP_DRIFT_GATE_EDGE_LEAK=1) -- raises UnboundLocalError there.

    A bytecode check catches the whole bug class for pennies: a name that
    is LOADed as a local but never STOREd in the same code object is an
    UnboundLocalError the moment control reaches it.
    """

    @staticmethod
    def _dead_names(func):
        """Names the function READS but that exist nowhere.

        A name assigned nowhere in a function is compiled as a GLOBAL
        load, so the failure mode is ``NameError`` (not
        ``UnboundLocalError``) the moment control reaches the line. Any
        LOAD_GLOBAL that resolves neither in the defining module nor in
        builtins is therefore a guaranteed crash.
        """
        import builtins
        import dis
        import sys

        mod = sys.modules[func.__module__]

        def scan(code):
            bad = set()
            for ins in dis.get_instructions(code):
                if ins.opname == "LOAD_GLOBAL":
                    n = ins.argval
                    if not hasattr(mod, n) and not hasattr(builtins, n):
                        bad.add(n)
            for c in code.co_consts:
                if hasattr(c, "co_code"):
                    bad |= scan(c)
            return bad

        return scan(func.__code__)

    def test_in_model_repeats_has_no_dead_names(self):
        bad = self._dead_names(GBSpecialStretchMove._run_in_model_repeats)
        self.assertEqual(
            bad, set(),
            f"_run_in_model_repeats reads names that exist nowhere: "
            f"{sorted(bad)} -- NameError when reached. At 66203dad this "
            f"was {{_c_hn, _c_nb, _c_p, _dg_cell_c, _dg_cross, "
            f"_dg_flat_n, _n_hn, _n_nb, _n_p}}: e79dbd7c deleted the "
            f"veto block that defined them and left the accept-side "
            f"occupancy scatter reading them.")

    def test_replace_step_has_no_dead_names(self):
        bad = self._dead_names(GBSpecialStretchMove._run_replace_step)
        self.assertEqual(bad, set(),
                         f"_run_replace_step: {sorted(bad)}")

    def test_rj_step_has_no_dead_names(self):
        bad = self._dead_names(GBSpecialStretchMove._run_rj_step)
        self.assertEqual(bad, set(), f"_run_rj_step: {sorted(bad)}")


if __name__ == "__main__":
    unittest.main()
