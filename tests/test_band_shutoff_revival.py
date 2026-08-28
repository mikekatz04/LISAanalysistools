"""Band-shutoff revival: reset the barren-band streaks and shut-off set.

USER RULING 2026-08-28. The high-f barren-band shutoff was PERMANENT for
the process ("revival semantics deliberately not implemented -- an OFF
band whose source later dies stays OFF"). Two things make that wrong over
a long run:

* a NEW F-stat epoch fit brings both a new proposal grid AND an updated
  noise/foreground profile, so a band that was unreachable may now be
  reachable;
* even with no refit, the noise model keeps evolving, so a long enough
  stretch should re-open the question on its own.

So the shut-off set and the occupancy streaks are cleared on either
trigger. Enforcement scope is unchanged and deliberately ALL-TEMPERATURE
(`_shut_ok = inds | ~shut[band_inds]` is indexed by BAND, so a shut-off
band's dead rows leave the subset at every temperature, while alive rows
stay proposable everywhere so the band drains rather than freezing).
"""

import os
import re
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _Stub:
    name = "rj_fstat_search"
    num_bands = 4
    _band_leaf_cap = None

    def __init__(self):
        # band 3 is the only one above the 10 mHz floor
        self.band_edges = np.array([1e-3, 5e-3, 9e-3, 11e-3, 13e-3])
        self._band_occ_streak = np.array([0, 0, 0, 7], dtype=np.int64)
        self._band_occ_last = np.array([0, 0, 0, 0], dtype=np.int64)
        self._rj_band_shutoff = np.array([False, False, False, True])

    _band_shutoff_revive = GBSpecialBase._band_shutoff_revive
    _band_shutoff_reset_iters = GBSpecialBase._band_shutoff_reset_iters


class ResetItersKnobTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("GB_RJ_BAND_SHUTOFF_RESET_ITERS")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GB_RJ_BAND_SHUTOFF_RESET_ITERS", None)
        else:
            os.environ["GB_RJ_BAND_SHUTOFF_RESET_ITERS"] = self._saved

    def test_default_is_100(self):
        os.environ.pop("GB_RJ_BAND_SHUTOFF_RESET_ITERS", None)
        self.assertEqual(GBSpecialBase._band_shutoff_reset_iters(_Stub()), 100)

    def test_explicit_value(self):
        os.environ["GB_RJ_BAND_SHUTOFF_RESET_ITERS"] = "37"
        self.assertEqual(GBSpecialBase._band_shutoff_reset_iters(_Stub()), 37)

    def test_zero_disables_the_iteration_trigger(self):
        os.environ["GB_RJ_BAND_SHUTOFF_RESET_ITERS"] = "0"
        self.assertEqual(GBSpecialBase._band_shutoff_reset_iters(_Stub()), 0)


class ReviveTest(unittest.TestCase):
    def test_revive_clears_shutoff_and_streaks(self):
        s = _Stub()
        n = s._band_shutoff_revive("epoch 2 fit")
        self.assertEqual(n, 1)                       # one band was off
        self.assertFalse(s._rj_band_shutoff.any())   # all revived
        self.assertTrue((s._band_occ_streak == 0).all())
        # the last-occupancy memory must also reset, or the very next
        # update counts an "unchanged" streak against a stale value.
        self.assertTrue((s._band_occ_last == -1).all())

    def test_revive_is_a_noop_when_nothing_is_off(self):
        s = _Stub()
        s._rj_band_shutoff[:] = False
        s._band_occ_streak[:] = 3
        n = s._band_shutoff_revive("epoch 2 fit")
        self.assertEqual(n, 0)
        # streaks still clear -- the evidence they were built from is stale
        self.assertTrue((s._band_occ_streak == 0).all())

    def test_revive_before_any_state_exists_does_not_raise(self):
        class _Bare:
            name = "rj_fstat_search"
            _band_shutoff_revive = GBSpecialBase._band_shutoff_revive
        self.assertEqual(_Bare()._band_shutoff_revive("epoch 1 fit"), 0)


LOGGER_NAME = "lisatools.globalfit.moves.gbspecialstretch"

# The regex the monitor's cap-plot overlay uses to find shut-off bands.
# A revival line must NOT match it (gf_monitor_gen.py).
_MONITOR_SHUTOFF_RE = re.compile(r"\[GB_BAND_SHUTOFF[^\]]*\] band (\d+)")


class _MoveStub:
    """Enough of a move to drive ``_update_band_shutoff`` end to end.

    Deliberately separate from ``_Stub`` above so the original spec's
    fixture stays untouched. Four bands, only band 3 above the 10 mHz
    shutoff floor, no cap machinery (so a second source is always
    "allowed" and occupancy 0 or 1 both qualify).
    """

    name = "rj_fstat_search"
    num_bands = 4
    _band_leaf_cap = None

    def __init__(self, epoch=0):
        self.band_edges = np.array([1e-3, 5e-3, 9e-3, 11e-3, 13e-3])
        self._fstat_epoch = epoch

    _band_shutoff_revive = GBSpecialBase._band_shutoff_revive
    _band_shutoff_reset_iters = GBSpecialBase._band_shutoff_reset_iters
    _band_shutoff_iters = GBSpecialBase._band_shutoff_iters
    _band_shutoff_epoch_sync = GBSpecialBase._band_shutoff_epoch_sync
    _update_band_shutoff = GBSpecialBase._update_band_shutoff


def _env(**kw):
    """Pin the shutoff knobs so a stray shell export cannot skew a test."""
    base = {
        "GB_RJ_BAND_SHUTOFF_FMIN_MHZ": "10.0",
        "GB_RJ_BAND_SHUTOFF_ITERS": "5",
        "GB_RJ_BAND_SHUTOFF_RESET_ITERS": "0",  # off unless a test asks
    }
    base.update(kw)
    return mock.patch.dict(os.environ, base)


BARREN = np.zeros(4, dtype=np.int64)


def _drive(move, n, occ=BARREN):
    for _ in range(n):
        move._update_band_shutoff(occ)


class ShutoffStillFiresTest(unittest.TestCase):
    """Baseline: the valve itself is unchanged by the revival work."""

    def test_barren_high_band_shuts_off_after_five(self):
        with _env():
            m = _MoveStub()
            _drive(m, 4)
            self.assertFalse(m._rj_band_shutoff.any())  # not yet
            _drive(m, 1)
            self.assertTrue(m._rj_band_shutoff[3])
            # low-f bands are never eligible, however barren
            self.assertFalse(m._rj_band_shutoff[:3].any())


class EpochTriggerTest(unittest.TestCase):
    def test_new_epoch_revives(self):
        with _env():
            m = _MoveStub(epoch=0)
            _drive(m, 5)
            self.assertTrue(m._rj_band_shutoff[3])
            m._fstat_epoch = 1
            _drive(m, 1)
            self.assertFalse(m._rj_band_shutoff.any())
            self.assertEqual(m._band_shutoff_epoch, 1)

    def test_same_epoch_does_not_refire(self):
        """The guard is "epoch changed", not "a fit ran"."""
        with _env():
            m = _MoveStub(epoch=0)
            _drive(m, 5)
            self.assertTrue(m._rj_band_shutoff[3])
            # many more iterations at the SAME epoch: the band stays off,
            # i.e. no revival kept quietly re-opening it.
            _drive(m, 25)
            self.assertTrue(m._rj_band_shutoff[3])

    def test_epoch_sync_is_idempotent_across_call_sites(self):
        """_install, setup-reuse and the per-iteration poll all call it.

        The first call at a new epoch revives; every later call at that
        same epoch is a no-op, so the three sites cannot compound.
        """
        with _env():
            m = _MoveStub(epoch=0)
            _drive(m, 5)
            m._fstat_epoch = 1
            self.assertEqual(m._band_shutoff_epoch_sync(), 1)  # revived one
            self.assertEqual(m._band_shutoff_epoch_sync(), 0)  # no-op
            self.assertEqual(m._band_shutoff_epoch_sync(), 0)  # still no-op

    def test_first_observation_adopts_silently(self):
        """No spurious revival line at process start."""
        with _env():
            m = _MoveStub(epoch=0)
            with self.assertNoLogs(LOGGER_NAME, level="INFO"):
                m._band_shutoff_epoch_sync()
            self.assertEqual(m._band_shutoff_epoch, 0)


class IterTriggerTest(unittest.TestCase):
    def test_reset_iters_revives_without_an_epoch_change(self):
        with _env(GB_RJ_BAND_SHUTOFF_RESET_ITERS="10"):
            m = _MoveStub(epoch=0)
            _drive(m, 5)
            self.assertTrue(m._rj_band_shutoff[3])
            _drive(m, 4)                      # 9 iterations elapsed
            self.assertTrue(m._rj_band_shutoff[3])
            _drive(m, 1)                      # the 10th fires the revival
            self.assertFalse(m._rj_band_shutoff.any())
            self.assertEqual(m._band_shutoff_since_revive, 0)
            self.assertEqual(m._fstat_epoch, 0)  # no refit involved

    def test_zero_disables_the_iteration_trigger(self):
        with _env(GB_RJ_BAND_SHUTOFF_RESET_ITERS="0"):
            m = _MoveStub(epoch=0)
            _drive(m, 200)
            self.assertTrue(m._rj_band_shutoff[3])


class RearmTest(unittest.TestCase):
    def test_a_revived_band_shuts_off_again_when_it_stays_barren(self):
        """Revival re-arms the valve; it does not disable it."""
        with _env():
            m = _MoveStub(epoch=0)
            _drive(m, 5)
            self.assertTrue(m._rj_band_shutoff[3])
            m._fstat_epoch = 1
            _drive(m, 1)                       # revive; streak restarts at 1
            self.assertFalse(m._rj_band_shutoff.any())
            self.assertEqual(int(m._band_occ_streak[3]), 1)
            _drive(m, 3)                       # streak 4 -- still on
            self.assertFalse(m._rj_band_shutoff.any())
            _drive(m, 1)                       # a full fresh AFTER window
            self.assertTrue(m._rj_band_shutoff[3])

    def test_stale_last_occupancy_cannot_shorten_the_second_window(self):
        """``_band_occ_last`` must reset to -1, not stay at the old count.

        If it kept the pre-revival value, the first post-revival update
        would score "unchanged" and the band would re-shut one iteration
        early off evidence gathered under the old epoch.
        """
        with _env():
            m = _MoveStub(epoch=0)
            _drive(m, 5)
            m._fstat_epoch = 1
            m._update_band_shutoff(BARREN)
            # streak 1 (a fresh start), never 2
            self.assertEqual(int(m._band_occ_streak[3]), 1)


class RevivalLogContractTest(unittest.TestCase):
    def test_revival_line_is_not_parsed_as_a_shutoff(self):
        with _env():
            m = _MoveStub(epoch=0)
            _drive(m, 5)
            m._fstat_epoch = 1
            with self.assertLogs(LOGGER_NAME, level="INFO") as cm:
                m._update_band_shutoff(BARREN)
        text = "\n".join(cm.output)
        self.assertIn("[GB_BAND_REVIVE rj_fstat_search]", text)
        self.assertEqual(_MONITOR_SHUTOFF_RE.findall(text), [])

    def test_shutoff_line_still_matches_the_monitor_regex(self):
        with _env():
            m = _MoveStub(epoch=0)
            _drive(m, 4)
            with self.assertLogs(LOGGER_NAME, level="INFO") as cm:
                m._update_band_shutoff(BARREN)
        self.assertEqual(
            _MONITOR_SHUTOFF_RE.findall("\n".join(cm.output)), ["3"])


if __name__ == "__main__":
    unittest.main()


class KillSwitchTest(unittest.TestCase):
    """``GB_RJ_BAND_SHUTOFF_ITERS <= 0`` disables the valve entirely.

    The shutoff is otherwise live with no way to turn it off from the
    environment, and its enforcement is a FULL RJ FREEZE -- so an operator
    needs a kill-switch that also RELEASES anything already shut off,
    not merely one that stops new shutoffs.
    """

    def test_zero_never_shuts_off(self):
        with _env(GB_RJ_BAND_SHUTOFF_ITERS="0"):
            m = _MoveStub()
            _drive(m, 50)
            self.assertFalse(m._rj_band_shutoff.any())

    def test_minus_one_never_shuts_off(self):
        with _env(GB_RJ_BAND_SHUTOFF_ITERS="-1"):
            m = _MoveStub()
            _drive(m, 50)
            self.assertFalse(m._rj_band_shutoff.any())

    def test_disabling_mid_run_releases_shut_off_bands(self):
        """Flipping it off must not strand a frozen band."""
        with _env(GB_RJ_BAND_SHUTOFF_ITERS="5"):
            m = _MoveStub()
            _drive(m, 5)
            self.assertTrue(m._rj_band_shutoff[3])
        with _env(GB_RJ_BAND_SHUTOFF_ITERS="0"):
            _drive(m, 1)
            self.assertFalse(m._rj_band_shutoff.any())
            self.assertTrue((m._band_occ_streak == 0).all())

    def test_default_is_five(self):
        """Unset means 5, not disabled."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("GB_RJ_BAND_SHUTOFF_ITERS", "GB_RJ_BAND_SHUTOFF_AFTER")}
        with mock.patch.dict(os.environ, env, clear=True):
            os.environ["GB_RJ_BAND_SHUTOFF_FMIN_MHZ"] = "10.0"
            os.environ["GB_RJ_BAND_SHUTOFF_RESET_ITERS"] = "0"
            m = _MoveStub()
            _drive(m, 4)
            self.assertFalse(m._rj_band_shutoff.any())
            _drive(m, 1)
            self.assertTrue(m._rj_band_shutoff[3])


class LegacyKnobNameTest(unittest.TestCase):
    """The old name still works, so an existing runbook is not downgraded.

    An unrecognised env var is silently ignored, so a hard rename would
    quietly fall back to the default instead of failing -- which is why the
    legacy name is honoured rather than dropped.
    """

    def _clean(self, **kw):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GB_RJ_BAND_SHUTOFF_ITERS",
                            "GB_RJ_BAND_SHUTOFF_AFTER")}
        env["GB_RJ_BAND_SHUTOFF_FMIN_MHZ"] = "10.0"
        env["GB_RJ_BAND_SHUTOFF_RESET_ITERS"] = "0"
        env.update(kw)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_legacy_name_is_honoured(self):
        with self._clean(GB_RJ_BAND_SHUTOFF_AFTER="2"):
            m = _MoveStub()
            _drive(m, 1)
            self.assertFalse(m._rj_band_shutoff.any())
            _drive(m, 1)
            self.assertTrue(m._rj_band_shutoff[3])

    def test_canonical_name_wins_over_legacy(self):
        with self._clean(GB_RJ_BAND_SHUTOFF_ITERS="5",
                         GB_RJ_BAND_SHUTOFF_AFTER="1"):
            m = _MoveStub()
            _drive(m, 4)
            self.assertFalse(m._rj_band_shutoff.any())
            _drive(m, 1)
            self.assertTrue(m._rj_band_shutoff[3])

    def test_garbage_value_falls_back_to_five(self):
        with self._clean(GB_RJ_BAND_SHUTOFF_ITERS="not-an-int"):
            m = _MoveStub()
            _drive(m, 4)
            self.assertFalse(m._rj_band_shutoff.any())
            _drive(m, 1)
            self.assertTrue(m._rj_band_shutoff[3])
