"""The F-stat grid fit must appear in the [GB_TIMING] tracked total.

Snapshot-11 forensics (2026-08-28): two of 475 [GB_TIMING] lines showed a
huge untracked block -- rj_fstat_search total=1887.4s tracked=1186.8s
untracked=700.5s (and 3011.2/651.4 the day before), against ~0.3s
untracked on every other line. Root cause: the once-per-epoch F-stat grid
fit (``_run_fstat_fit``, logged "grid fit epoch 1 done in 689.0s") runs
OUTSIDE the ``run_proposal`` span, and its wall was never recorded into
the propose timer -- so a known, deliberate, already-measured cost read as
an unexplained 37% accounting hole and cost a forensic dive to re-derive.

The fit's wall is now accumulated as the top-level stage
``fstat_grid_fit``. It must be TRACKED (it is real propose time outside
every other top-level span); nested stages must stay untracked so the
accounting cannot double-count.
"""

import unittest

from lisatools.globalfit.moves.gbspecialstretch import _ProposeTimer


class FstatGridFitTrackedTest(unittest.TestCase):
    def test_grid_fit_counts_as_tracked(self):
        tm = _ProposeTimer()
        tm.add("run_proposal", 984.670)
        tm.add("run_tempering", 197.126)
        tm.add("fstat_grid_fit", 689.0)
        line = tm.report(total=1887.367)
        self.assertIn("tracked=1870.796s", line)
        self.assertIn("untracked=16.571s", line)

    def test_grid_fit_appears_in_the_stage_list(self):
        tm = _ProposeTimer()
        tm.add("fstat_grid_fit", 689.0)
        self.assertIn("fstat_grid_fit=689.000s", tm.report(total=700.0))

    def test_nested_stages_stay_untracked(self):
        # rj_fstat_centers lives INSIDE run_proposal; counting it would
        # double-count the same seconds.
        tm = _ProposeTimer()
        tm.add("run_proposal", 984.670)
        tm.add("rj_fstat_centers", 783.167)
        line = tm.report(total=984.670)
        self.assertIn("tracked=984.670s", line)
        self.assertIn("untracked=0.000s", line)

    def test_no_grid_fit_is_unchanged(self):
        # A propose with no refit this epoch reports exactly as before.
        tm = _ProposeTimer()
        tm.add("run_proposal", 1000.0)
        line = tm.report(total=1000.5)
        self.assertIn("tracked=1000.000s", line)
        self.assertIn("untracked=0.500s", line)


if __name__ == "__main__":
    unittest.main()
