"""``init_logger(shared_console=False)``: a per-branch Setup's logger init must
not downgrade the shared ``lisatools`` console handler (which the run-level
owners configure with the run's ``verbose``).

Before 2026-10-13 every branch Setup re-ran ``init_logger`` with the default
``console=False`` and so swapped the shared handler to a WARNING-only one
part-way through ``fit.build()``: every build-time INFO line from
``lisatools.*`` after the first branch Setup vanished from the job log (the
9mo job 758 showed no MBH inner-move report although the stack was built).
"""

import logging
import unittest

from lisatools.globalfit.loginfo import _CONSOLE_HANDLER_NAME, init_logger


def _managed(logger):
    return [h for h in logger.handlers if h.get_name() == _CONSOLE_HANDLER_NAME]


class SharedConsoleTest(unittest.TestCase):
    def setUp(self):
        self.lt = logging.getLogger("lisatools")
        self._saved = (list(self.lt.handlers), self.lt.level, self.lt.propagate)
        for h in _managed(self.lt):
            self.lt.removeHandler(h)

    def tearDown(self):
        for h in _managed(self.lt):
            self.lt.removeHandler(h)
        handlers, level, propagate = self._saved
        for h in handlers:
            if h not in self.lt.handlers:
                self.lt.addHandler(h)
        self.lt.setLevel(level)
        self.lt.propagate = propagate
        for name in ("_T_run", "_T_branch", "_T_branch2"):
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                lg.removeHandler(h)

    def test_branch_setup_keeps_the_verbose_shared_console(self):
        init_logger(name="_T_run", level=logging.INFO, console=True)
        (h,) = _managed(self.lt)
        self.assertEqual(h.level, logging.INFO)
        # a branch Setup: its own named logger only
        branch = init_logger(name="_T_branch", level=logging.DEBUG, shared_console=False)
        self.assertEqual(_managed(self.lt), [h])
        self.assertEqual(h.level, logging.INFO)
        self.assertTrue(_managed(branch))
        # the shared console still receives lisatools.* INFO
        with self.assertLogs("lisatools.globalfit.stock.erebor.source_runtime", level="INFO"):
            logging.getLogger("lisatools.globalfit.stock.erebor.source_runtime").info("visible")

    def test_run_level_call_still_flips_the_shared_console(self):
        init_logger(name="_T_run", level=logging.INFO, console=True)
        (h,) = _managed(self.lt)
        self.assertEqual(h.level, logging.INFO)
        init_logger(name="_T_branch2", level=logging.DEBUG)  # default shared_console=True
        (h2,) = _managed(self.lt)
        self.assertEqual(h2.level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
