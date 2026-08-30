"""Tempering must not swap a cap cell over its cap.

THE HOLE (user diagnosis 2026-08-30). Tempering exchanges a whole
``(temp, walker, band)`` cell between rungs -- "every source of both cells
trades its temperature" -- and no cap is read anywhere in ~700 lines of
``run_tempering`` / ``_vertical_swap_sweep`` / ``_tempering_swap_grid`` /
``_permute_walkers_for_swaps``.

That was SAFE while cap cells were ALIGNED with sub-bands: a band swap
moved a cell's entire contents, so occupancy transferred exactly.
STAGGERING split each band across two cells and removed the invariant, so
"a swap down from two neighbouring sub-bands" can load two sources into
one straddling cell with nothing to stop it. This is the third route into
the cold chain, alongside RJ births (gated, measured rejecting 859,590 in
one propose) and in-model drift (gated; closing it changed nothing).

SCOPE (user correction): this is a SEARCH-ONLY constraint, not a
correctness fix. The RJ ``-inf`` cap gate is a proposal-level veto on
birth rows, not a prior; tempering's ratio is pure likelihood, so the cap
was never in the swap's target. Vetoing here ADDS a constraint -- fine
under the search policy, and it must stay out of PE. PE runs every cap
disarmed at ``-1``, which is why the disarmed case is tested hardest: a
naive ``post <= cap`` reads ``post <= -1`` and would reject EVERY swap in
PE.
"""
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import tempering_swap_cap_ok


class SwapCapPredicateTest(unittest.TestCase):
    """Shapes are (npair, ncells_affected); at divisor 1 a band touches the
    two cells sharing its edges, so ncells_affected == 2."""

    def _ok(self, occ_a, occ_b, fb_a, fb_b, cap):
        return tempering_swap_cap_ok(
            np.array(occ_a), np.array(occ_b),
            np.array(fb_a), np.array(fb_b), np.array(cap))

    def test_swap_of_equal_contributions_is_always_fine(self):
        """The aligned-grid case: contents transfer exactly, occupancy
        unchanged on both sides."""
        got = self._ok([[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]])
        self.assertTrue(bool(got[0]))

    def test_the_seam_double_is_rejected(self):
        """THE CASE. Cell 0 holds 1 from the NEIGHBOUR band (not swapped)
        on side A; the hot side's band contributes 1 into that same cell.
        Post-swap side A would hold 2 against a cap of 1."""
        got = self._ok(
            occ_a=[[1, 0]],      # side A: 1 in the straddling cell...
            fb_a=[[0, 0]],       # ...none of it from the swapped band
            occ_b=[[1, 0]],      # side B holds 1 there...
            fb_b=[[1, 0]],       # ...and it IS from the swapped band
            cap=[[1, 1]])
        self.assertFalse(bool(got[0]))

    def test_swap_that_empties_and_fills_within_cap_is_allowed(self):
        got = self._ok(
            occ_a=[[1, 0]], fb_a=[[1, 0]],
            occ_b=[[0, 1]], fb_b=[[0, 1]],
            cap=[[1, 1]])
        self.assertTrue(bool(got[0]))

    def test_both_sides_are_checked_not_just_the_cold_one(self):
        """Search caps bind at EVERY temperature, so an overflow on the
        hot side must veto too."""
        got = self._ok(
            occ_a=[[0, 0]], fb_a=[[0, 0]],
            occ_b=[[1, 0]], fb_b=[[0, 0]],   # B keeps its 1 and gains 0
            cap=[[1, 1]])
        self.assertTrue(bool(got[0]))
        # now make side B overflow: it keeps 1 non-swapped and receives 1
        got = self._ok(
            occ_a=[[1, 0]], fb_a=[[1, 0]],
            occ_b=[[1, 0]], fb_b=[[0, 0]],
            cap=[[1, 1]])
        self.assertFalse(bool(got[0]))

    def test_a_violation_in_either_affected_cell_vetoes(self):
        for bad in (0, 1):
            occ_a = [[0, 0]]; fb_a = [[0, 0]]
            occ_b = [[0, 0]]; fb_b = [[0, 0]]
            occ_b[0][bad] = 2; fb_b[0][bad] = 2
            got = self._ok(occ_a, occ_b, fb_a, fb_b, [[1, 1]])
            self.assertFalse(bool(got[0]), msg=f"cell {bad}")

    def test_higher_caps_admit_more(self):
        args = dict(occ_a=[[1, 0]], fb_a=[[0, 0]],
                    occ_b=[[1, 0]], fb_b=[[1, 0]])
        self.assertFalse(bool(self._ok(cap=[[1, 1]], **args)[0]))
        self.assertTrue(bool(self._ok(cap=[[2, 2]], **args)[0]))

    def test_per_cell_caps_are_independent(self):
        got = self._ok(
            occ_a=[[1, 1]], fb_a=[[0, 0]],
            occ_b=[[1, 1]], fb_b=[[1, 1]],
            cap=[[2, 1]])          # cell 0 has room, cell 1 does not
        self.assertFalse(bool(got[0]))

    def test_vectorises_over_pairs(self):
        got = self._ok(
            occ_a=[[1, 0], [0, 0], [1, 0]],
            fb_a=[[0, 0], [0, 0], [1, 0]],
            occ_b=[[1, 0], [0, 0], [0, 1]],
            fb_b=[[1, 0], [0, 0], [0, 1]],
            cap=[[1, 1], [1, 1], [1, 1]])
        np.testing.assert_array_equal(got, [False, True, True])


class DisarmedCapTest(unittest.TestCase):
    """PE runs every cap disarmed at -1. The predicate must be vacuous
    there -- a naive ``post <= cap`` would reject every swap in PE."""

    def test_disarmed_caps_admit_everything(self):
        got = tempering_swap_cap_ok(
            np.array([[5, 3]]), np.array([[7, 2]]),
            np.array([[1, 0]]), np.array([[4, 2]]),
            np.array([[-1, -1]]))
        self.assertTrue(bool(got[0]))

    def test_disarmed_is_per_cell(self):
        """One armed cell, one disarmed: only the armed one constrains."""
        got = tempering_swap_cap_ok(
            np.array([[1, 9]]), np.array([[1, 9]]),
            np.array([[0, 0]]), np.array([[1, 9]]),
            np.array([[1, -1]]))
        self.assertFalse(bool(got[0]))      # cell 0 would reach 2 vs cap 1
        got = tempering_swap_cap_ok(
            np.array([[0, 9]]), np.array([[0, 9]]),
            np.array([[0, 0]]), np.array([[0, 9]]),
            np.array([[1, -1]]))
        self.assertTrue(bool(got[0]))       # cell 1 unbounded, cell 0 fine

    def test_all_disarmed_is_identical_to_no_gate(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            a = rng.integers(0, 6, size=(8, 2))
            b = rng.integers(0, 6, size=(8, 2))
            fa = rng.integers(0, 3, size=(8, 2))
            fb = rng.integers(0, 3, size=(8, 2))
            got = tempering_swap_cap_ok(a, b, fa, fb, np.full((8, 2), -1))
            self.assertTrue(bool(np.all(got)))



class IncrementalCensusTest(unittest.TestCase):
    """``_cap_swap_apply`` must leave the census exactly where a full
    rebuild would.

    This is the OOM fix's correctness condition. ``_cap_swap_census`` walks
    ~5.5 M sorter rows and allocates ~8 arrays that long; rebuilding it
    inside the per-repeat vertical sweep drove GPU0 78.5 -> 95.3 GB and
    GPU1 55.2 -> 91.4 GB, both to the ceiling, and killed the run. It is
    now built once per block and carried forward by an O(npair) update --
    so the update has to be exact, or the gate silently drifts off the
    truth it is supposed to enforce.

    Modelled directly on the array layout: ``counts`` is flat over
    (temp, walker, cell), ``lo``/``hi`` flat over (temp, walker, band).
    """

    NT, NW, NB = 2, 2, 4          # -> NCELLS == NB at divisor 1

    def _flat_cell(self, t, w, c):
        return (t * self.NW + w) * self.NB + c

    def _rebuild(self, occ):
        """Full census from a per-(t,w,band) -> (lower, upper) table."""
        counts = np.zeros(self.NT * self.NW * self.NB, dtype=np.int64)
        lo = np.zeros(self.NT * self.NW * self.NB, dtype=np.int64)
        hi = np.zeros_like(lo)
        for (t, w, b), (nlo, nhi) in occ.items():
            f = (t * self.NW + w) * self.NB + b
            lo[f], hi[f] = nlo, nhi
            counts[self._flat_cell(t, w, b)] += nlo
            counts[self._flat_cell(t, w, min(b + 1, self.NB - 1))] += nhi
        return counts, lo, hi

    def test_update_matches_a_full_rebuild(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        mv = object.__new__(GBSpecialBase)
        mv.ntemps, mv.nwalkers = self.NT, self.NW
        mv.num_bands = mv.num_cap_cells = self.NB

        occ = {(0, 0, 1): (1, 0), (1, 0, 1): (0, 1),
               (0, 1, 2): (1, 1), (1, 1, 2): (0, 0)}
        counts, lo, hi = self._rebuild(occ)
        cap = np.full(self.NB, 5, dtype=np.int64)

        # swap band 1 between (t0,w0) and (t1,w0); band 2 between the w1 pair
        t_a = np.array([0, 0]); w_a = np.array([0, 1])
        t_b = np.array([1, 1]); w_b = np.array([0, 1])
        bands = np.array([1, 2])
        acc = np.array([True, True])

        mv._cap_swap_apply((counts, lo, hi, cap), t_a, w_a, t_b, w_b,
                           bands, acc)

        swapped = dict(occ)
        for (ta, wa, tb, wb, b) in ((0, 0, 1, 0, 1), (0, 1, 1, 1, 2)):
            swapped[(ta, wa, b)], swapped[(tb, wb, b)] = (
                occ[(tb, wb, b)], occ[(ta, wa, b)])
        exp_counts, exp_lo, exp_hi = self._rebuild(swapped)
        np.testing.assert_array_equal(counts, exp_counts)
        np.testing.assert_array_equal(lo, exp_lo)
        np.testing.assert_array_equal(hi, exp_hi)

    def test_rejected_pairs_leave_the_census_untouched(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        mv = object.__new__(GBSpecialBase)
        mv.ntemps, mv.nwalkers = self.NT, self.NW
        mv.num_bands = mv.num_cap_cells = self.NB
        occ = {(0, 0, 1): (1, 0), (1, 0, 1): (0, 1)}
        counts, lo, hi = self._rebuild(occ)
        before = (counts.copy(), lo.copy(), hi.copy())
        mv._cap_swap_apply(
            (counts, lo, hi, np.full(self.NB, 5, dtype=np.int64)),
            np.array([0]), np.array([0]), np.array([1]), np.array([0]),
            np.array([1]), np.array([False]))
        for got, exp in zip((counts, lo, hi), before):
            np.testing.assert_array_equal(got, exp)

class SharedCensusThreeWritersTest(unittest.TestCase):
    """One census, three writers -- and the drift writer is the one that
    was missing.

    THE SNEAK THIS CLOSES. The drift gate and the swap gate kept SEPARATE
    occupancy arrays. A source drifts across its band's midpoint into cell
    c (drift gate allows it -- c was empty -- and updates ITS census); a
    vertical swap for the NEIGHBOURING band then reads the SWAP census,
    which still says c is empty, and accepts. Cell c ends holding two, one
    from each adjacent band. That is the shape of all 53 cross-seam
    doubles that survived the tempering gate (down from 528).

    counts is now one shared array; ``lo``/``hi`` -- the per-band split the
    swap gate reads to decide what a swap moves -- are carried across drift
    by _cap_lohi_transition, which is what these pin.
    """

    NT, NW, NB = 2, 2, 4

    def _mv(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        mv = object.__new__(GBSpecialBase)
        mv.ntemps, mv.nwalkers = self.NT, self.NW
        mv.num_bands = mv.num_cap_cells = self.NB
        return mv

    def _bflat(self, t, w, b):
        return (t * self.NW + w) * self.NB + b

    def test_drift_across_the_midpoint_moves_lower_to_upper(self):
        mv = self._mv()
        lo = np.zeros(self.NT * self.NW * self.NB, dtype=np.int64)
        hi = np.zeros_like(lo)
        f = self._bflat(0, 0, 1)
        lo[f] = 1                                  # one source, lower half
        mv._cap_lohi_transition(
            lo, hi,
            np.array([0]), np.array([0]), np.array([1]),
            cur_cell=np.array([1]),                # cell b   (lower)
            new_cell=np.array([2]),                # cell b+1 (upper)
            accept=np.array([True]))
        self.assertEqual(int(lo[f]), 0)
        self.assertEqual(int(hi[f]), 1)

    def test_drift_back_is_the_exact_inverse(self):
        mv = self._mv()
        lo = np.zeros(self.NT * self.NW * self.NB, dtype=np.int64)
        hi = np.zeros_like(lo)
        f = self._bflat(1, 1, 2)
        hi[f] = 1
        mv._cap_lohi_transition(
            lo, hi, np.array([1]), np.array([1]), np.array([2]),
            np.array([3]), np.array([2]), np.array([True]))
        self.assertEqual((int(lo[f]), int(hi[f])), (1, 0))

    def test_non_crossing_and_rejected_moves_change_nothing(self):
        mv = self._mv()
        lo = np.zeros(self.NT * self.NW * self.NB, dtype=np.int64)
        hi = np.zeros_like(lo)
        lo[self._bflat(0, 0, 1)] = 1
        before = (lo.copy(), hi.copy())
        # same cell -> no transition
        mv._cap_lohi_transition(lo, hi, np.array([0]), np.array([0]),
                                np.array([1]), np.array([1]), np.array([1]),
                                np.array([True]))
        # crossing but REJECTED
        mv._cap_lohi_transition(lo, hi, np.array([0]), np.array([0]),
                                np.array([1]), np.array([1]), np.array([2]),
                                np.array([False]))
        for got, exp in zip((lo, hi), before):
            np.testing.assert_array_equal(got, exp)

    def test_the_sneak_is_vetoed_once_the_census_is_shared(self):
        """End to end: drift fills cell c, then the neighbour's swap is
        offered. With lo/hi carried across the drift, the gate sees the
        occupancy and refuses."""
        mv = self._mv()
        counts = np.zeros(self.NT * self.NW * self.NB, dtype=np.int64)
        lo = np.zeros(self.NT * self.NW * self.NB, dtype=np.int64)
        hi = np.zeros_like(lo)
        cap = np.ones(self.NB, dtype=np.int64)

        # cold (t0,w0): band 1 holds one source in its LOWER half -> cell 1
        f1 = self._bflat(0, 0, 1)
        lo[f1] = 1
        counts[(0 * self.NW + 0) * self.NB + 1] = 1
        # hot (t1,w0): band 2 holds one source in its LOWER half -> cell 2
        f2 = self._bflat(1, 0, 2)
        lo[f2] = 1
        counts[(1 * self.NW + 0) * self.NB + 2] = 1

        # DRIFT: the cold band-1 source crosses its midpoint into cell 2.
        mv._cap_lohi_transition(lo, hi, np.array([0]), np.array([0]),
                                np.array([1]), np.array([1]), np.array([2]),
                                np.array([True]))
        counts[(0 * self.NW + 0) * self.NB + 1] -= 1
        counts[(0 * self.NW + 0) * self.NB + 2] += 1   # shared counts array

        # Now offer the band-2 swap between cold and hot. Post-swap the
        # cold side would hold its drifted source AND the hot band-2 one.
        ok = mv._swap_cap_ok(
            (counts, lo, hi, cap),
            np.array([0]), np.array([0]),      # cold side
            np.array([1]), np.array([0]),      # hot side
            np.array([2]))
        self.assertFalse(bool(ok[0]),
                         "the swap that produced the surviving cross-seam "
                         "doubles must now be vetoed")


if __name__ == "__main__":
    unittest.main()
