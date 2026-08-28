"""Table-vs-per-row F-stat center audit (GB_FSTAT_CTR_AUDIT).

The per-row center solve is the single biggest cost in the search move
(~725-743 s/propose, ~63% of it, measured snapshot 12). The epoch center
TABLE is the cheap alternative, but it was retired for candidate quality
in the 2026-08-26 per-row ruling and nobody has ever measured HOW far
apart the two answers actually are.

This audit measures exactly that: on a subsample of the rows the unit
precompute already solved per-row, it ALSO reads the table and logs the
delta distribution per output. Diagnostic only -- it never feeds a
proposal, so it cannot affect sampling or detailed balance.

Only phi0/iota/psi/A_max/F come from the table; ``_fstat_ctr_table_lookup``
re-derives ``(ln_center, sigma)`` per row with the row's OWN (f0, Mc), so
the Mc**(5/3) scaling is already exact and the mismatch we are measuring
lives in the maximized extrinsics and in A_max/F.

Angle metrics are CIRCULAR with the right period (phi0 2*pi, psi pi), and
inclination is compared as cos(iota) -- the actual sampled column.
"""

import os
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _Stub:
    name = "rj_fstat_search"
    xp = np


class CircularDiffTest(unittest.TestCase):
    def test_wrap_is_shortest_arc_2pi(self):
        a = np.array([0.1, 0.0, np.pi])
        b = np.array([2 * np.pi - 0.1, 0.0, 0.0])
        d = GBSpecialBase._circ_absdiff(a, b, 2 * np.pi, np)
        np.testing.assert_allclose(d, [0.2, 0.0, np.pi], atol=1e-12)

    def test_period_pi_for_psi(self):
        # psi is degenerate mod pi: 0.05 and pi-0.05 are 0.1 apart.
        a = np.array([0.05])
        b = np.array([np.pi - 0.05])
        d = GBSpecialBase._circ_absdiff(a, b, np.pi, np)
        np.testing.assert_allclose(d, [0.1], atol=1e-12)

    def test_never_exceeds_half_period(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(-10, 10, 500)
        b = rng.uniform(-10, 10, 500)
        d = GBSpecialBase._circ_absdiff(a, b, 2 * np.pi, np)
        self.assertLessEqual(d.max(), np.pi + 1e-12)
        self.assertGreaterEqual(d.min(), 0.0)


class AuditRowsKnobTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("GB_FSTAT_CTR_AUDIT")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GB_FSTAT_CTR_AUDIT", None)
        else:
            os.environ["GB_FSTAT_CTR_AUDIT"] = self._saved

    def test_default_is_off(self):
        os.environ.pop("GB_FSTAT_CTR_AUDIT", None)
        self.assertEqual(GBSpecialBase._fstat_ctr_audit_rows(_Stub()), 0)

    def test_on_gives_default_sample(self):
        os.environ["GB_FSTAT_CTR_AUDIT"] = "1"
        self.assertEqual(GBSpecialBase._fstat_ctr_audit_rows(_Stub()), 4096)

    def test_explicit_row_count(self):
        os.environ["GB_FSTAT_CTR_AUDIT"] = "512"
        self.assertEqual(GBSpecialBase._fstat_ctr_audit_rows(_Stub()), 512)

    def test_zero_disables(self):
        os.environ["GB_FSTAT_CTR_AUDIT"] = "0"
        self.assertEqual(GBSpecialBase._fstat_ctr_audit_rows(_Stub()), 0)


class SummaryTest(unittest.TestCase):
    def test_median_p90_max(self):
        d = np.arange(1, 101, dtype=float)  # 1..100
        med, p90, mx = GBSpecialBase._absdiff_summary(d, np)
        self.assertAlmostEqual(med, 50.5, places=6)
        self.assertAlmostEqual(mx, 100.0, places=6)
        self.assertGreater(p90, med)
        self.assertLessEqual(p90, mx)

    def test_empty_is_nan_not_crash(self):
        med, p90, mx = GBSpecialBase._absdiff_summary(
            np.zeros(0, dtype=float), np)
        self.assertTrue(np.isnan(med) and np.isnan(p90) and np.isnan(mx))


if __name__ == "__main__":
    unittest.main()
