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


if __name__ == "__main__":
    unittest.main()
