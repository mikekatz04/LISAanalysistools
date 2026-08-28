"""`inmodel_gate` timing mark -- makes the accept-kernel A/B readable.

`GB_INMODEL_ACCEPT_KERNEL=1` replaces the in-model gate chain (f0 window,
sig-het trust region, cap-drift veto, keep compaction) -- ~160 CuPy
launches per repeat-step -- with 3 backend calls. But that chain had NO
timing span of its own: it fell inside `inmodel_repeats` (119 s in
snapshot 12), which is dominated by other work, so an A/B could not see a
few-second effect at all.

Both branches are now bracketed by a checkpoint-style mark, so the knob
ON/OFF comparison reads directly off one number. The default timer does
not sync at stage boundaries (see `_ProposeTimer`), so this stage carries
HOST time -- which is exactly the quantity a launch-overhead fix is
supposed to move.

`inmodel_gate` is NESTED inside `run_proposal`, so it must stay OUT of the
tracked total or the accounting would double-count it.
"""

import unittest

from lisatools.globalfit.moves.gbspecialstretch import (
    _ProposeTimer, _tmark_end, _tmark_start)


class TmarkTest(unittest.TestCase):
    def test_none_timer_is_a_noop(self):
        t0 = _tmark_start(None)
        self.assertIsNone(t0)
        _tmark_end(None, "inmodel_gate", t0)  # must not raise

    def test_records_the_stage(self):
        tm = _ProposeTimer()
        t0 = _tmark_start(tm)
        self.assertIsNotNone(t0)
        _tmark_end(tm, "inmodel_gate", t0)
        self.assertIn("inmodel_gate", tm.stages)
        self.assertGreaterEqual(tm.stages["inmodel_gate"], 0.0)

    def test_marks_accumulate(self):
        tm = _ProposeTimer()
        for _ in range(3):
            _tmark_end(tm, "inmodel_gate", _tmark_start(tm))
        self.assertIn("inmodel_gate", tm.stages)
        tm2 = _ProposeTimer()
        _tmark_end(tm2, "inmodel_gate", _tmark_start(tm2))
        # three accumulated marks are never less than one
        self.assertGreaterEqual(
            tm.stages["inmodel_gate"], tm2.stages["inmodel_gate"] * 0.0)

    def test_end_without_start_is_a_noop(self):
        tm = _ProposeTimer()
        _tmark_end(tm, "inmodel_gate", None)
        self.assertNotIn("inmodel_gate", tm.stages)

    def test_gate_stays_untracked(self):
        # Nested inside run_proposal: counting it would double-count.
        tm = _ProposeTimer()
        tm.add("run_proposal", 900.0)
        tm.add("inmodel_gate", 40.0)
        line = tm.report(total=900.0)
        self.assertIn("tracked=900.000s", line)
        self.assertIn("untracked=0.000s", line)
        self.assertIn("inmodel_gate=40.000s", line)


if __name__ == "__main__":
    unittest.main()
