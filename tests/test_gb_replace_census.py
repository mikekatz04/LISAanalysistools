"""rj_replace acceptance census (instrumentation, user request 2026-08-27).

The [GB_ACCEPT rj-split] census covers only the birth/death path;
rj_replace's own swap acceptance was invisible in production (the probe
verdicts -- cold ~0.2% -- could not be checked at full band). The move
now accumulates a per-propose ``_replace_split`` census inside
``_run_replace_step`` and prints/resets it in the propose-end
diagnostics block:

    [GB_ACCEPT replace-split <name>] proposals N (cold n): accepted
    A = r (cold a/n = rc) | gated: snr S nonfinite B | cold-accepted
    dll mean m max M

These tests drive the two helpers directly (device-agnostic numpy
inputs; the wiring in _run_replace_step is a thin call).
"""

import logging
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _Stub:
    name = "rj_replace"


class ReplaceCensusTest(unittest.TestCase):
    def setUp(self):
        self.stub = _Stub()

    def _add(self, t_i, accept, dll, n_snr, n_bad):
        GBSpecialBase._replace_census_add(
            self.stub, np.asarray(t_i), np.asarray(accept, dtype=bool),
            np.asarray(dll, dtype=float), int(n_snr), int(n_bad))

    def test_accumulates_across_calls(self):
        self._add([0, 0, 2], [True, False, True], [5.0, -1.0, 9.0], 1, 0)
        self._add([0, 1], [True, True], [3.0, 7.0], 0, 2)
        sp = self.stub._replace_split
        self.assertEqual(sp["proposals"], 5)
        self.assertEqual(sp["proposals_cold"], 3)
        self.assertEqual(sp["acc"], 4)
        self.assertEqual(sp["acc_cold"], 2)
        self.assertEqual(sp["snr"], 1)
        self.assertEqual(sp["nonfinite"], 2)
        self.assertEqual(sp["dll_cold_sum"], 8.0)   # 5.0 + 3.0
        self.assertEqual(sp["dll_cold_max"], 5.0)

    def test_report_emits_and_resets(self):
        self._add([0, 0], [True, False], [4.0, 0.0], 0, 0)
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch",
                level=logging.INFO) as cm:
            GBSpecialBase._replace_census_report(self.stub)
        joined = "\n".join(cm.output)
        self.assertIn("replace-split rj_replace", joined)
        self.assertIn("proposals 2 (cold 2)", joined)
        self.assertIn("accepted 1", joined)
        self.assertIsNone(self.stub._replace_split)

    def test_report_noop_when_empty(self):
        # no _replace_split attribute at all -> no log, no error
        GBSpecialBase._replace_census_report(self.stub)
        with self.assertRaises(AssertionError):
            with self.assertLogs(
                    "lisatools.globalfit.moves.gbspecialstretch",
                    level=logging.INFO):
                GBSpecialBase._replace_census_report(self.stub)


if __name__ == "__main__":
    unittest.main()
