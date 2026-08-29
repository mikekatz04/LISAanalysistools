"""Pair-borrow: joint two-band cold-chain substitution gated on TRUE lnL.

USER DESIGN 2026-08-29. A source sitting near a sub-band edge splits into
two leaves in ADJACENT bands (the 20.380377 mHz flagship: 30 leaves in
band 1141, 22 in band 1142, corr -0.747, total power conserved). Those two
bands are in DIFFERENT residue classes (1141 % 9 == 7, 1142 % 9 == 8), so
they are NEVER open in the same scheduling unit and the joint update that
would drop one leaf while boosting the other is structurally inexpressible
-- a block-Gibbs trap, not a mixing failure.

The move: for a flagged (walker, band-pair), take a DONOR walker that
already holds the clean single-leaf configuration over that pair, and
substitute the donor's cold-chain pair content into the recipient. The
proposal is generated however we like (a donor copy here; a "pretend"
rework later) -- what makes it safe is that ACCEPTANCE IS PRICED ON THE
TRUE LIKELIHOOD in the recipient's OWN residual context, so the cold chain
can never end up worse than it started.

Because the gate scores BOTH bands jointly it is a genuine two-band move
even though every sampler stage that produced the content ran per-band.

SEARCH ONLY. The donor is chosen by cold logL, which is state-dependent
and carries no detailed-balance cost only because the gate is on true
improvement rather than on a pretended density. PE must never run it:
:meth:`_pair_borrow_enabled` hard-refuses on the ``replace_pe_stage``
install stamp and outside ``GB_MODE=search``, regardless of the knob.
"""

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _Stub:
    """Minimal stand-in carrying only what the pair-borrow helpers read."""

    name = "rj_fstat_search"

    def __init__(self, replace_pe_stage=False):
        self.replace_pe_stage = replace_pe_stage


def _enabled(stub):
    return GBSpecialBase._pair_borrow_enabled(stub)


class TestPairBorrowKnob(unittest.TestCase):
    """The knob defaults OFF in code and is PROVABLY inert in PE."""

    def test_default_off(self):
        with mock.patch.dict(os.environ, {"GB_MODE": "search"}, clear=False):
            os.environ.pop("GB_PAIR_BORROW", None)
            self.assertFalse(_enabled(_Stub()))

    def test_armed_in_search(self):
        with mock.patch.dict(
            os.environ, {"GB_PAIR_BORROW": "1", "GB_MODE": "search"}
        ):
            self.assertTrue(_enabled(_Stub()))

    def test_explicit_off_values(self):
        for val in ("0", "off", "false", ""):
            with mock.patch.dict(
                os.environ, {"GB_PAIR_BORROW": val, "GB_MODE": "search"}
            ):
                self.assertFalse(_enabled(_Stub()), msg=val)

    def test_pe_stamp_refuses_even_when_armed(self):
        # The install-site stamp wins over an explicit =1. This is the
        # "provably inert in PE" requirement: no env setting can arm it
        # for a PE-stamped move.
        with mock.patch.dict(
            os.environ, {"GB_PAIR_BORROW": "1", "GB_MODE": "search"}
        ):
            self.assertFalse(_enabled(_Stub(replace_pe_stage=True)))

    def test_non_search_mode_refuses_even_when_armed(self):
        for mode in ("pe", "PE", "", "mixed"):
            with mock.patch.dict(
                os.environ, {"GB_PAIR_BORROW": "1", "GB_MODE": mode}
            ):
                self.assertFalse(_enabled(_Stub()), msg=mode)


class TestPairBorrowCandidates(unittest.TestCase):
    """Recipient / donor / pair selection is pure index logic."""

    @staticmethod
    def _cands(occ, cold_ll, side=1, max_pairs=8):
        return GBSpecialBase._pair_borrow_candidates(
            np.asarray(occ), np.asarray(cold_ll, dtype=float),
            side=side, max_pairs=max_pairs,
        )

    def test_flagship_shape(self):
        # 10 walkers over 4 bands; the pair under test is (1, 2).
        # Walkers 0-6 are SPLIT (1 leaf each side = 2 over the pair),
        # walkers 7-9 are CLEAN single-leaf donors.
        occ = np.zeros((10, 4), dtype=int)
        occ[:7, 1] = 1
        occ[:7, 2] = 1
        occ[7:, 1] = 1
        cold_ll = np.arange(10, dtype=float)      # walker 9 is the best
        out = self._cands(occ, cold_ll, side=1, max_pairs=64)

        self.assertEqual(len(out), 7, msg=f"{out=}")
        recips = sorted(w for w, _, _ in out)
        self.assertEqual(recips, list(range(7)))
        # Every candidate names the SAME low band and the best-logL donor.
        self.assertTrue(all(b == 1 for _, b, _ in out), msg=f"{out=}")
        self.assertTrue(all(d == 9 for _, _, d in out), msg=f"{out=}")

    def test_no_donor_yields_nothing(self):
        # Every walker is split -> no clean single-leaf donor exists.
        occ = np.zeros((6, 3), dtype=int)
        occ[:, 0] = 1
        occ[:, 1] = 1
        self.assertEqual(self._cands(occ, np.zeros(6)), [])

    def test_recipient_needs_two_over_the_pair(self):
        # A walker holding exactly one leaf over the pair is a DONOR, and
        # is never emitted as a recipient of its own content.
        occ = np.zeros((4, 3), dtype=int)
        occ[0, 0] = 2          # split (both leaves in one band still counts)
        occ[1:, 0] = 1         # clean donors
        out = self._cands(occ, np.arange(4, dtype=float))
        self.assertEqual([w for w, _, _ in out], [0])
        self.assertNotIn(0, [d for _, _, d in out])

    def test_max_pairs_truncates(self):
        occ = np.zeros((10, 3), dtype=int)
        occ[:8, 0] = 2
        occ[8:, 0] = 1
        out = self._cands(occ, np.arange(10, dtype=float), max_pairs=3)
        self.assertEqual(len(out), 3)

    def test_left_side_normalises_the_band_pair(self):
        # side=-1 must report the SAME (low, low+1) pair as side=+1 does.
        occ = np.zeros((4, 3), dtype=int)
        occ[0, 1] = 1
        occ[0, 2] = 1
        occ[1:, 2] = 1
        out = self._cands(occ, np.arange(4, dtype=float), side=-1)
        self.assertEqual(len(out), 1, msg=f"{out=}")
        self.assertEqual(out[0][1], 1, msg="low band of the (1,2) pair")

    def test_deterministic_order(self):
        occ = np.zeros((8, 4), dtype=int)
        occ[:5, 1] = 2
        occ[5:, 1] = 1
        ll = np.arange(8, dtype=float)
        self.assertEqual(self._cands(occ, ll), self._cands(occ, ll))


class TestPairBorrowGate(unittest.TestCase):
    """The cold chain can NEVER end up worse: strict improvement only."""

    def test_strict_improvement_only(self):
        acc = GBSpecialBase._pair_borrow_accept
        self.assertTrue(acc(1e-6))
        self.assertFalse(acc(0.0))
        self.assertFalse(acc(-1.0))
        self.assertFalse(acc(np.nan))

    def test_eps_floor(self):
        acc = GBSpecialBase._pair_borrow_accept
        self.assertFalse(acc(5.0, eps=10.0))
        self.assertTrue(acc(50.0, eps=10.0))


class TestPairBorrowLog(unittest.TestCase):
    """Mandatory observability: one greppable line per propose."""

    def test_line_names_donors_pairs_and_true_dll(self):
        line = GBSpecialBase._pair_borrow_log_line(
            "rj_fstat_search", propose=12, side=1, attempted=6,
            subs=[(19, 3, 1141, 1834.0), (7, 3, 880, 412.0)],
        )
        self.assertIn("[GB_PAIRBORROW", line)
        self.assertIn("rj_fstat_search", line)
        self.assertIn("propose=12", line)
        self.assertIn("attempted=6", line)
        self.assertIn("substituted=2", line)
        self.assertIn("w19<-w3", line)          # recipient <- donor
        self.assertIn("(1141,1142)", line)      # the band pair
        self.assertIn("+1.834e+03", line)       # the TRUE dll
        self.assertIn("(880,881)", line)

    def test_empty_run_still_reports(self):
        line = GBSpecialBase._pair_borrow_log_line(
            "rj_fstat_search", propose=3, side=-1, attempted=0, subs=[],
        )
        self.assertIn("attempted=0", line)
        self.assertIn("substituted=0", line)


if __name__ == "__main__":
    unittest.main()
