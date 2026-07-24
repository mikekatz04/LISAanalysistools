"""GF_MOVE_TIMING instrumentation: per-move [GF_TIMING] lines from
GFCombineMove.propose, armed purely by env at call time.

Uses ``erebor.blank`` (zero data, hidden idle branch/move) so the whole test
runs in seconds on CPU with no external data, while exercising the exact
Stage -> GFCombineMove -> propose path every stock fit uses.
"""

import contextlib
import copy
import io
import os
import pickle
import re
import unittest

from lisatools.globalfit.stock import erebor

_LINE = re.compile(
    r"^\[GF_TIMING\] stage=(?P<stage>\S+) move=(?P<move>\S+) it=(?P<it>\d+) "
    r"wall_s=[\d.]+ rss_mb=\d+"
)


class MoveTimingTest(unittest.TestCase):
    def _run_blank(self, iterations=2):
        fit = erebor.blank()
        # repo contract: pre-build fits deepcopy/pickle regardless of env
        fit = pickle.loads(pickle.dumps(copy.deepcopy(fit)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _model, _state in fit.sample(iterations=iterations):
                pass
        return buf.getvalue()

    def test_timing_lines_per_move_per_iteration(self):
        os.environ["GF_MOVE_TIMING"] = "1"
        try:
            out = self._run_blank(iterations=2)
        finally:
            os.environ.pop("GF_MOVE_TIMING", None)

        lines = [m.groupdict() for m in
                 (_LINE.match(ln) for ln in out.splitlines()) if m]
        self.assertTrue(lines, f"no [GF_TIMING] lines in output:\n{out[-2000:]}")

        by_iter = {}
        for d in lines:
            by_iter.setdefault(int(d["it"]), []).append(d["move"])
        self.assertGreaterEqual(len(by_iter), 2, "expected >=2 timed iterations")
        for it, moves in by_iter.items():
            self.assertEqual(
                moves.count("__total__"), 1,
                f"iteration {it}: expected exactly one __total__ line, got {moves}",
            )
            self.assertGreaterEqual(
                len([m for m in moves if m != "__total__"]), 1,
                f"iteration {it}: expected at least one per-move line",
            )
        # stage tag flowed through from Stage.setup
        self.assertTrue(all(d["stage"] != "?" for d in lines),
                        "stage name not tagged on GFCombineMove")

    def test_off_path_emits_nothing(self):
        os.environ.pop("GF_MOVE_TIMING", None)
        out = self._run_blank(iterations=1)
        self.assertNotIn("[GF_TIMING]", out)


if __name__ == "__main__":
    unittest.main()
