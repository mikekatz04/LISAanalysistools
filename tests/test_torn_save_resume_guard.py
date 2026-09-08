"""A torn save must be caught at RESUME, not at the first propose.

The main backend and each sub-backend are not flushed atomically. A job killed
between them -- a spot preemption, ~1.7/day on the 3-month partition -- leaves a
row whose main half is written and whose sub-backend half is still zeros. The
row DECOMPRESSES FINE, so the readability validator passed it, the backup
promotion never fired, and the run resumed into it and MPI-aborted minutes later
at ``GFState.check_cold_row``:

    [vgb] cold-chain inds mismatch between the main state and its sub-state
    (550 differing leaf slots)

(observed: 3mo v8 10-walker, job 464, stored iteration 1). These pin the
detection at resume time, where the existing backup-promotion path can recover.
"""

import os
import tempfile
import unittest

import h5py
import numpy as np

from lisatools.globalfit.hdfbackend import _validate_resume_readable


def _store(path, *, n_it=3, ntemps_sub=8, nwalkers=10, nleaves=55, tear=False,
           empty_branch=False):
    """Minimal store with one 'vgb' branch; ``tear`` zeroes the sub half."""
    with h5py.File(path, "w") as f:
        root = f.create_group("global_fit")
        root.attrs["iteration"] = n_it
        row = n_it - 1
        main = np.zeros((n_it, 1, 1, nwalkers, nleaves), dtype=bool)
        sub = np.zeros((n_it, ntemps_sub, nwalkers, nleaves), dtype=bool)
        if not empty_branch:
            main[:, 0, 0, :, : nleaves // 2] = True
            sub[:, :, :, : nleaves // 2] = True
        if tear:
            sub[row] = False          # sub-backend half never flushed
        root.create_group("inds").create_dataset("vgb", data=main)
        sg = root.create_group("sub_backend").create_group("vgb")
        sg.create_dataset("inds", data=sub)
        sg.create_dataset("chain", data=np.zeros(
            (n_it, ntemps_sub, nwalkers, nleaves, 5)))


class TornSaveIsCaughtAtResume(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.p = os.path.join(self.d.name, "s.h5")

    def test_healthy_store_passes(self):
        _store(self.p)
        _validate_resume_readable(self.p)          # must not raise

    def test_torn_last_row_raises(self):
        _store(self.p, tear=True)
        with self.assertRaises(ValueError) as cm:
            _validate_resume_readable(self.p)
        msg = str(cm.exception)
        self.assertIn("INCOMPLETE", msg)
        self.assertIn("vgb", msg)
        self.assertIn("torn save", msg)

    def test_reports_the_number_of_differing_slots(self):
        """The count is the diagnostic -- job 464 reported 550 = 55 x 10."""
        _store(self.p, tear=True, nwalkers=10, nleaves=55)
        with self.assertRaises(ValueError) as cm:
            _validate_resume_readable(self.p)
        # half the leaves were alive, so half the slots differ
        self.assertIn(str(10 * (55 // 2)), str(cm.exception))

    def test_legitimately_empty_branch_is_not_a_false_positive(self):
        """A branch with zero leaves has all-False inds on BOTH sides.

        This is the case that rules out a naive 'is the row all zeros' check:
        gb genuinely holds no leaves in the first iterations of a fresh run.
        """
        _store(self.p, empty_branch=True)
        _validate_resume_readable(self.p)          # must not raise

    def test_earlier_rows_are_not_inspected(self):
        """Only the row a resume would load matters; history may be anything."""
        _store(self.p, n_it=4)
        with h5py.File(self.p, "r+") as f:
            f["global_fit"]["sub_backend"]["vgb"]["inds"][0] = False
        _validate_resume_readable(self.p)          # must not raise

    def test_missing_sub_backend_is_tolerated(self):
        """A psd-only / noise-only store has no such group; do not crash."""
        with h5py.File(self.p, "w") as f:
            root = f.create_group("global_fit")
            root.attrs["iteration"] = 2
            root.create_group("inds").create_dataset(
                "psd", data=np.ones((2, 1, 1, 4, 1), dtype=bool))
        _validate_resume_readable(self.p)          # must not raise


if __name__ == "__main__":
    unittest.main()
