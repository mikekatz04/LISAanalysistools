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


# ===========================================================================
# THE CALL SITE — production defect 2026-08-28 (57 iterations, zero fires).
#
# Every test above drives ``_update_band_shutoff`` DIRECTLY, so the valve's
# LOGIC was covered 24 ways while its WIRING was not covered at all. In
# production the once-per-iteration tick sat at the tail of ``run_proposal``
# and read ``new_state`` — a local of ``propose``, never a name in
# ``run_proposal``'s scope (its parameter is ``state``). Every propose raised
# ``NameError: name 'new_state' is not defined`` straight into the enclosing
# ``except Exception: pass`` diagnostics guard, so:
#
#   * ``_update_band_shutoff`` never ran once, hence
#   * ``_band_occ_streak`` / ``_rj_band_shutoff`` were never even allocated,
#   * the enforcement site's ``getattr(self, "_rj_band_shutoff", None)``
#     stayed None so nothing was ever frozen, and
#   * ``_band_shutoff_revive`` no-oped WITHOUT a log line (it returns 0
#     silently when no state exists), so the two F-stat epoch changes in the
#     run produced zero ``[GB_BAND_REVIVE]`` lines too.
#
# 640 of the 1232 GB bands sat at zero cold occupancy for all 57 iterations
# against a 5-iteration clock. The defect was invisible because the only
# evidence it could ever have produced was a log line it could not reach.
# ===========================================================================

import builtins
import dis
import types

from lisatools.globalfit.moves import gbspecialstretch as _gbmod


def _codes(code):
    """``code`` and every code object nested inside it."""
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _codes(const)


def _unresolvable_globals(func):
    """Global names ``func`` loads that resolve nowhere at runtime.

    A LOAD_GLOBAL for a name that is neither a module attribute nor a
    builtin is a guaranteed ``NameError`` the moment that line executes.
    Inside a bare ``except Exception`` guard that is a silent no-op, which
    is exactly how the band-shutoff valve stayed dead for a whole run.
    """
    known = set(vars(_gbmod)) | set(dir(builtins))
    bad = set()
    for code in _codes(func.__code__):
        for ins in dis.get_instructions(code):
            if ins.opname == "LOAD_GLOBAL" and ins.argval not in known:
                bad.add(ins.argval)
    return sorted(bad)


def _tick_carriers():
    """Methods of ``GBSpecialBase`` that drive the per-iteration tick."""
    out = []
    for name, fn in vars(GBSpecialBase).items():
        if not isinstance(fn, types.FunctionType):
            continue
        names = set()
        for code in _codes(fn.__code__):
            names.update(code.co_names)
        if "_update_band_shutoff" in names or "_band_occupancy_cold_max" in names:
            out.append((name, fn))
    return out


class ValveWiringTest(unittest.TestCase):
    """The valve must be REACHABLE, not merely correct."""

    def test_the_tick_call_site_has_no_undefined_names(self):
        carriers = _tick_carriers()
        # the tick must live somewhere other than its own definition
        self.assertTrue(
            [n for n, _ in carriers if n != "_update_band_shutoff"],
            "nothing calls _update_band_shutoff — the valve is unreachable",
        )
        for name, fn in carriers:
            with self.subTest(method=name):
                self.assertEqual(
                    _unresolvable_globals(fn), [],
                    f"GBSpecialBase.{name} loads a global that resolves "
                    "nowhere; it will raise NameError into the diagnostics "
                    "guard and the shutoff tick will never run",
                )

    def test_run_proposal_and_propose_have_no_undefined_names(self):
        for name in ("run_proposal", "propose"):
            with self.subTest(method=name):
                self.assertEqual(
                    _unresolvable_globals(getattr(GBSpecialBase, name)), [],
                    f"GBSpecialBase.{name} references an undefined global",
                )


# ---------------------------------------------------------------------------
# The OTHER half of the dead path. ``_band_occupancy_cold_max`` is what feeds
# the valve, and because the tick never ran it had never executed once in
# production either. It carries two easy-to-get-wrong conversions — coords
# f0 is in mHz while ``band_edges`` is in Hz, and the cold chain is row 0 of
# the module sub-state's FULL ladder, not of the main state — so pin both
# before the tick goes live.
# ---------------------------------------------------------------------------

from lisatools.globalfit.moves.globalfitmove import GlobalFitMove


class _Branch:
    def __init__(self, inds, coords):
        self.inds, self.coords = inds, coords


class _State:
    sub_states = None

    def __init__(self, branch):
        self.branches = {"gb": branch}


class _OccStub:
    """Minimal move for ``_band_occupancy_cold_max``: 4 bands, 3 walkers."""

    name = "rj_fstat_search"
    branch_name = "gb"
    num_bands = 4
    nwalkers = 3

    def __init__(self):
        self.band_edges = np.array([1e-3, 5e-3, 9e-3, 11e-3, 13e-3])

    _work_branch = GlobalFitMove._work_branch
    _band_occupancy_cold_max = GBSpecialBase._band_occupancy_cold_max


def _mk_state(f0_mhz_per_walker, alive_per_walker, ntemps=3):
    """(ntemps, nwalkers, nleaves) branch; f0 lives in coords column 1."""
    nw = len(f0_mhz_per_walker)
    nl = max(len(r) for r in f0_mhz_per_walker)
    coords = np.zeros((ntemps, nw, nl, 8))
    inds = np.zeros((ntemps, nw, nl), dtype=bool)
    for w, (f0s, alive) in enumerate(zip(f0_mhz_per_walker, alive_per_walker)):
        for l, (f0, a) in enumerate(zip(f0s, alive)):
            coords[0, w, l, 1] = f0
            inds[0, w, l] = a
    # hot rows are deliberately STUFFED: the valve is a cold-chain rule, so
    # a hot-chain junk leaf must not keep a barren band alive.
    coords[1:, :, :, 1] = 12.0
    inds[1:] = True
    return _State(_Branch(inds, coords))


class ColdOccupancyTest(unittest.TestCase):
    def test_f0_is_read_in_mhz_against_hz_band_edges(self):
        # 12 mHz -> band 3 (11-13 mHz); 6 mHz -> band 1 (5-9 mHz)
        st = _mk_state([[12.0, 6.0]], [[True, True]])
        st.branches["gb"].inds = st.branches["gb"].inds[:, :1]
        st.branches["gb"].coords = st.branches["gb"].coords[:, :1]
        m = _OccStub()
        m.nwalkers = 1
        occ = m._band_occupancy_cold_max(st)
        self.assertEqual(list(occ), [0, 1, 0, 1])

    def test_max_over_walkers_not_sum(self):
        # every walker holds one source in band 3 -> max is 1, not 3
        st = _mk_state([[12.0], [12.5], [11.5]], [[True], [True], [True]])
        occ = _OccStub()._band_occupancy_cold_max(st)
        self.assertEqual(list(occ), [0, 0, 0, 1])

    def test_dead_leaves_do_not_count(self):
        st = _mk_state([[12.0], [12.5], [11.5]],
                       [[False], [False], [False]])
        occ = _OccStub()._band_occupancy_cold_max(st)
        self.assertEqual(list(occ), [0, 0, 0, 0])

    def test_hot_chain_occupancy_is_ignored(self):
        """The 640 barren production bands are barren IN THE COLD CHAIN."""
        st = _mk_state([[6.0], [6.0], [6.0]], [[True], [True], [True]])
        occ = _OccStub()._band_occupancy_cold_max(st)
        self.assertEqual(occ[3], 0)  # hot rows are all at 12 mHz

    def test_leaves_outside_the_grid_are_dropped(self):
        st = _mk_state([[0.5, 99.0, 12.0]], [[True, True, True]])
        st.branches["gb"].inds = st.branches["gb"].inds[:, :1]
        st.branches["gb"].coords = st.branches["gb"].coords[:, :1]
        m = _OccStub()
        m.nwalkers = 1
        occ = m._band_occupancy_cold_max(st)
        self.assertEqual(list(occ), [0, 0, 0, 1])


# ---------------------------------------------------------------------------
# OBSERVABILITY. The valve logged only when it FIRED, so a valve that never
# ran and a valve with nothing to do produced byte-identical evidence: none.
# The status line has to be emitted on every tick, including the boring ones.
# ---------------------------------------------------------------------------

_STATUS_RE = re.compile(r"\[GB_BAND_SHUTOFF status ([a-z_0-9]+)\]")


class StatusLineTest(unittest.TestCase):
    def test_status_is_emitted_even_when_nothing_fires(self):
        with _env():
            m = _MoveStub()
            with self.assertLogs(LOGGER_NAME, level="INFO") as cm:
                m._update_band_shutoff(BARREN)          # tick 1 of 5
        text = "\n".join(cm.output)
        self.assertEqual(_STATUS_RE.findall(text), ["rj_fstat_search"])
        self.assertFalse(m._rj_band_shutoff.any())      # nothing fired
        self.assertIn("clock 5 iters", text)
        self.assertIn("floor 10.000 mHz", text)
        self.assertIn("1/4 bands eligible", text)
        self.assertIn("1 qualifying now (cold occ 0)", text)
        self.assertIn("0 armed (streak >= clock)", text)

    def test_status_line_is_not_parsed_as_a_shutoff(self):
        """The monitor's cap-plot overlay must not paint a band from it."""
        with _env():
            m = _MoveStub()
            with self.assertLogs(LOGGER_NAME, level="INFO") as cm:
                _drive(m, 4)
        self.assertEqual(
            _MONITOR_SHUTOFF_RE.findall("\n".join(cm.output)), [])

    def test_status_reports_the_resolved_clock_not_the_default(self):
        with _env(GB_RJ_BAND_SHUTOFF_ITERS="9"):
            m = _MoveStub()
            with self.assertLogs(LOGGER_NAME, level="INFO") as cm:
                m._update_band_shutoff(BARREN)
        self.assertIn("clock 9 iters", "\n".join(cm.output))

    def test_status_counts_armed_and_off_when_the_valve_fires(self):
        with _env():
            m = _MoveStub()
            _drive(m, 4)
            with self.assertLogs(LOGGER_NAME, level="INFO") as cm:
                m._update_band_shutoff(BARREN)          # the 5th -> fires
        text = "\n".join(cm.output)
        self.assertIn("1 armed (streak >= clock)", text)
        self.assertIn("1 off (+1 this tick)", text)
        # and the real fire line is still there for the monitor
        self.assertEqual(_MONITOR_SHUTOFF_RE.findall(text), ["3"])

    def test_status_tracks_streak_max_and_median(self):
        with _env():
            m = _MoveStub()
            with self.assertLogs(LOGGER_NAME, level="INFO") as cm:
                _drive(m, 3)
        # one eligible band (band 3), streak 3 after three barren ticks
        self.assertIn("streak max 3 median 3", "\n".join(cm.output))
