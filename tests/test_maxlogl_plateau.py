import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove


class _ScriptedMaxLogL(MaxLogLCombineMove):
    """Exercises ONLY the plateau loop: GFCombineMove.__init__ is skipped and
    _propose_moves_once returns a scripted cold-chain max-lnL sequence."""

    def __init__(self, seq, num_checks=5, tol=5.0, max_iter=0):
        self.num_checks = num_checks
        self.tol = tol
        self.max_iter = max_iter
        self.seq = list(seq)
        self.calls = 0

    def _propose_moves_once(self, model, state):
        val = self.seq[self.calls]
        self.calls += 1
        return SimpleNamespace(log_like=np.array([[val]])), None


def _run(seq, **kwargs):
    move = _ScriptedMaxLogL(seq, **kwargs)
    state0 = SimpleNamespace(log_like=np.array([[-np.inf]]))
    move._propose_moves(model=None, state=state0)
    return move.calls


class MaxLogLPlateauTest(unittest.TestCase):
    def setUp(self):
        os.environ["MAXLOGL_LOG_EVERY"] = "0"
        self.addCleanup(os.environ.pop, "MAXLOGL_LOG_EVERY", None)

    def test_soft_tol_ignores_noise_floor_twitches(self):
        # Strict sub-tol improvements (the +2-scale polishing observed on the
        # cluster) must count as flat: baseline 1000, everything <= 1005.
        seq = [1000, 1001, 1002, 1003, 1000, 1000, 1000, 1000, 1000]
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 6)

    def test_tol_zero_restores_strict_rule(self):
        # Same sequence, tol=0: every new max resets the counter.
        seq = [1000, 1001, 1002, 1003, 1000, 1000, 1000, 1000, 1000]
        self.assertEqual(_run(seq, num_checks=5, tol=0.0), 9)

    def test_significant_climb_keeps_stage_alive(self):
        # Super-tol jumps reset; the plateau tail then ends it.
        seq = [1000, 1010, 1020, 1030] + [1030] * 6
        # 1030 == baseline exactly -> flat; needs changed_once (set at 1010).
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 9)

    def test_slow_accumulated_climb_is_progress(self):
        # +2/iter forever: each step is sub-tol but the baseline does NOT
        # advance on sub-tol steps, so the accumulated climb (+10 per
        # 5-window) keeps resetting -- the stage must not exit early.
        seq = [1000 + 2 * i for i in range(30)]
        move = _ScriptedMaxLogL(seq, num_checks=5, tol=5.0, max_iter=20)
        state0 = SimpleNamespace(log_like=np.array([[-np.inf]]))
        move._propose_moves(model=None, state=state0)
        self.assertEqual(move.calls, 20)  # only the ceiling stops it

    def test_max_iter_ceiling(self):
        seq = [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070]
        self.assertEqual(_run(seq, num_checks=5, tol=5.0, max_iter=3), 3)


if __name__ == "__main__":
    unittest.main()
