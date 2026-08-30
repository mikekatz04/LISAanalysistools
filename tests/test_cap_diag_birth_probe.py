"""GB_CAP_DIAG birth probe: does a birth land in an already-at-cap cell?

The probe exists because reading the code has been wrong twice about the
seam doubles (528 cap cells holding one leaf either side of a sub-band
seam, at 5.1x the chance rate at duplication separation, against a cap of
1). Every static path audits clean and closing the in-model route changed
nothing, so this counts the event directly.

The counter is only worth having if IT is right, hence these. The
semantics under test:

* ``into_at_cap`` counts births whose destination cell ALREADY held
  ``>= cap`` in the census the gate scored against -- the gate sets
  ``curr_logp = -inf`` for exactly those, so a non-zero count means the
  enforcement was bypassed, not out-voted;
* ``same_flat_repeat`` counts births beyond the first landing in one
  ``(temp, walker, cell)`` inside a single scored batch -- supposed to be
  impossible under serial-within-band plus the residue stride.
"""
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    cap_diag_birth_violations,
)


class BirthViolationCountTest(unittest.TestCase):
    def test_no_births_is_all_zero(self):
        self.assertEqual(
            cap_diag_birth_violations(
                np.zeros(10, dtype=np.int64), np.ones(5, dtype=np.int64),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)),
            (0, 0, 0))

    def test_clean_births_into_empty_cells(self):
        counts = np.zeros(10, dtype=np.int64)
        cap = np.ones(10, dtype=np.int64)
        flat = np.array([1, 3, 7]); cells = np.array([1, 3, 7])
        self.assertEqual(cap_diag_birth_violations(counts, cap, flat, cells),
                         (3, 0, 0))

    def test_birth_into_an_at_cap_cell_is_caught(self):
        """The event the whole probe exists to detect."""
        counts = np.zeros(10, dtype=np.int64); counts[4] = 1   # cell 4 full
        cap = np.ones(10, dtype=np.int64)
        flat = np.array([2, 4]); cells = np.array([2, 4])
        n, into, rep = cap_diag_birth_violations(counts, cap, flat, cells)
        self.assertEqual((n, into, rep), (2, 1, 0))

    def test_over_cap_cell_also_counts(self):
        counts = np.zeros(10, dtype=np.int64); counts[4] = 3
        cap = np.ones(10, dtype=np.int64)
        n, into, _ = cap_diag_birth_violations(
            counts, cap, np.array([4]), np.array([4]))
        self.assertEqual((n, into), (1, 1))

    def test_below_cap_is_not_a_violation(self):
        """cap 2 with one occupant has headroom -- not a violation."""
        counts = np.zeros(10, dtype=np.int64); counts[4] = 1
        cap = np.full(10, 2, dtype=np.int64)
        n, into, _ = cap_diag_birth_violations(
            counts, cap, np.array([4]), np.array([4]))
        self.assertEqual((n, into), (1, 0))

    def test_per_cell_caps_are_honoured_independently(self):
        counts = np.array([1, 1, 1, 1], dtype=np.int64)
        cap = np.array([1, 2, 1, 5], dtype=np.int64)
        n, into, _ = cap_diag_birth_violations(
            counts, cap, np.arange(4), np.arange(4))
        self.assertEqual((n, into), (4, 2))      # cells 0 and 2 only

    def test_same_flat_repeats_counted_beyond_the_first(self):
        counts = np.zeros(10, dtype=np.int64)
        cap = np.ones(10, dtype=np.int64)
        flat = np.array([5, 5, 5, 2]); cells = np.array([5, 5, 5, 2])
        n, into, rep = cap_diag_birth_violations(counts, cap, flat, cells)
        self.assertEqual((n, rep), (4, 2))       # 3 into cell 5 -> 2 extra

    def test_repeats_and_violations_are_independent_counters(self):
        counts = np.zeros(10, dtype=np.int64); counts[5] = 1
        cap = np.ones(10, dtype=np.int64)
        flat = np.array([5, 5]); cells = np.array([5, 5])
        n, into, rep = cap_diag_birth_violations(counts, cap, flat, cells)
        self.assertEqual((n, into, rep), (2, 2, 1))

    def test_flat_and_cell_indices_are_used_for_their_own_lookups(self):
        """flat indexes the (temp, walker, cell) census; cells index the
        per-cell cap. Swapping them must change the answer -- this pins
        that the probe is not accidentally symmetric."""
        ncells = 4
        counts = np.zeros(3 * ncells, dtype=np.int64)
        counts[2 * ncells + 1] = 1               # walker 2, cell 1 is full
        cap = np.array([1, 1, 1, 1], dtype=np.int64)
        flat = np.array([2 * ncells + 1]); cells = np.array([1])
        self.assertEqual(
            cap_diag_birth_violations(counts, cap, flat, cells), (1, 1, 0))
        # same cell index, a DIFFERENT walker -> that cell is empty there
        flat2 = np.array([0 * ncells + 1])
        self.assertEqual(
            cap_diag_birth_violations(counts, cap, flat2, cells), (1, 0, 0))


class KnobTest(unittest.TestCase):
    def test_diag_is_off_by_default(self):
        import os
        from unittest import mock
        from lisatools.globalfit.moves.gbspecialstretch import _cap_diag_on
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_CAP_DIAG", None)
            self.assertFalse(_cap_diag_on())
        with mock.patch.dict(os.environ, {"GB_CAP_DIAG": "1"}, clear=False):
            self.assertTrue(_cap_diag_on())
        with mock.patch.dict(os.environ, {"GB_CAP_DIAG": "0"}, clear=False):
            self.assertFalse(_cap_diag_on())


if __name__ == "__main__":
    unittest.main()
