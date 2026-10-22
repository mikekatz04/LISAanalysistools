"""The per-stage F-stat refit cadence, and the 2026-10-19 gb_search_2 defaults.

Mike 2026-10-19: "Make the fstat refit every 2 cycles during gb search 2."
``GB_FSTAT_REFIT_EVERY`` is one value for every search stage, so a stage that
wants its own cadence carries it in its profile:

* ``GB_SEARCH_{N}_FSTAT_REFIT_EVERY`` -> ``run_combined_staged.search_stage_profiles``
  writes ``fstat_refit_every`` on that stage's profile only;
* ``recipe.apply_fstat_refit_every`` (from ``SearchStageProfileStep._apply_profile``,
  BEFORE the peak-floor block) writes it on the stage's F-stat grid moves and
  RESTORES the built cadence on a stage without the key (the move objects are
  shared across stages); a move built with cadence <= 0 (fit once) is never
  touched;
* the head's ``_fstat_fit_decision`` reads the attribute (the ranks install
  the epoch the head ships).

Units are the refit clock's: global-fit iterations, one per stored row, three
per search cycle under ``GB_SEARCH_LEGS=1`` -- so 4 refits every second cycle.

Also pinned here, because the layout suite lists these knobs by name only:
the launchers' gb_search_2 defaults 30 repeats / 1-pass group window (Mike
2026-10-19, "fewer, fatter passes") and the F-stat cadence 4, with gb_search_1
left on ``GB_FSTAT_REFIT_EVERY=2`` (every cycle).
"""
import contextlib
import glob
import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "fstat_proposal"))


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


REFIT_ENV = ("GB_SEARCH_1_FSTAT_REFIT_EVERY", "GB_SEARCH_2_FSTAT_REFIT_EVERY",
             "GB_SEARCH_3_FSTAT_REFIT_EVERY")


def _gb(name, rj, **kw):
    base = dict(name=name, branch_name="gb", opt_snr_rej_samp_limit=8.0,
                phase_maximize=False, _snr_lim_table=None, is_rj_prop=rj,
                rj_removal_only=False, rj_replace=False,
                num_repeat_proposals=25, inmodel_group_iters=3,
                inmodel_converge_iters=250)
    base.update(kw)
    return SimpleNamespace(**base)


def _tree():
    """The gb_search cycle; only rj_fstat_search is an F-stat grid move."""
    moves = [_gb("rj_warm_search", True), _gb("in_model", False),
             _gb("rj_fstat_search", True, fstat_refit_every=2),
             _gb("in_model_fstat", False),
             _gb("rj_prior_removal", True), _gb("in_model_removal", False)]
    return {m.name: m for m in moves}, [SimpleNamespace(moves=moves)]


class _FakeBackend:
    def __init__(self, iteration):
        self.iteration = iteration

    def get_nleaves(self, branch_names=None, temp_index=0):
        return {branch_names[0]: np.arange(self.iteration)[:, None]}

    def stage_start_iteration(self, name):
        return None


def _step(profile, tree, stage_name="gb_search_x"):
    from lisatools.globalfit.recipe import SearchStageProfileStep

    st = SearchStageProfileStep(moves=tree, convergence_iter=2, plateau_branch="gb",
                                profile=profile, stage_name=stage_name)
    sampler = SimpleNamespace(backend=_FakeBackend(10), moves=tree, periodic=None,
                              temperature_control=None, weights=None)
    st.setup_run(10, None, sampler)
    st.note_recipe_step(1)
    return st


def _clear_peak_floor_override():
    from lisatools.sampling.fstat_proposal import set_peak_min_F_override

    set_peak_min_F_override(None)


class ApplyFstatRefitTest(unittest.TestCase):

    def tearDown(self):
        _clear_peak_floor_override()

    def test_the_stage_value_reaches_the_grid_move_only(self):
        by, tree = _tree()
        _step(dict(opt_snr=5.0, fstat_refit_every=4), tree, "gb_search_2")
        self.assertEqual(by["rj_fstat_search"].fstat_refit_every, 4)
        for n in ("rj_warm_search", "in_model", "rj_prior_removal", "in_model_removal"):
            self.assertFalse(hasattr(by[n], "fstat_refit_every"), n)   # not invented

    def test_a_stage_without_the_key_restores_the_built_cadence(self):
        """gb_search_1 keeps GB_FSTAT_REFIT_EVERY whichever stage ran first."""
        by, tree = _tree()
        _step(dict(fstat_refit_every=4), tree, "gb_search_2")
        self.assertEqual(by["rj_fstat_search"].fstat_refit_every, 4)
        _step(dict(opt_snr=8.0), tree, "gb_search_1")
        self.assertEqual(by["rj_fstat_search"].fstat_refit_every, 2)
        self.assertEqual(by["rj_fstat_search"]._fstat_refit_every_built, 2)

    def test_a_fit_once_move_is_never_armed(self):
        """Cadence <= 0 means fit once (the PE galaxy grid) or never fit; a
        stage profile must not turn it into a refitting move."""
        from lisatools.globalfit.recipe import apply_fstat_refit_every

        once = SimpleNamespace(name="rj_fstat_pe", fstat_refit_every=0)
        self.assertEqual(apply_fstat_refit_every([once], 4, "gb_search_2"), {})
        self.assertEqual(once.fstat_refit_every, 0)

    def test_the_entry_line_names_the_change(self):
        from lisatools.globalfit.recipe import apply_fstat_refit_every

        by, _ = _tree()
        with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
            changed = apply_fstat_refit_every(list(by.values()), 4, "gb_search_2")
        self.assertEqual(changed, {"rj_fstat_search.fstat_refit_every": (2, 4)})
        line = "\n".join(cm.output)
        self.assertIn("[V9-STAGE gb_search_2] F-stat refit cadence set on entry", line)
        self.assertIn("rj_fstat_search.fstat_refit_every 2 -> 4", line)
        self.assertEqual(apply_fstat_refit_every(list(by.values()), 4), {})   # no change, no line

    def test_bad_values_are_refused(self):
        from lisatools.globalfit.recipe import apply_fstat_refit_every

        by, _ = _tree()
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                apply_fstat_refit_every(list(by.values()), bad)

    def test_the_cadence_is_set_before_a_forced_refit_is_armed(self):
        """A forced refit armed at the same entry (the peak floor moved) judges
        staleness against the stage's cadence, so the cadence must already be
        on the move when force_fstat_refit runs."""
        import lisatools.globalfit.recipe as R

        by, tree = _tree()
        seen = []

        def _spy(moves, serial, reason):
            seen.append(by["rj_fstat_search"].fstat_refit_every)
            return 1

        _clear_peak_floor_override()
        with mock.patch.object(R, "force_fstat_refit", _spy):
            _step(dict(peak_min_snr=6.25, fstat_refit_every=4), tree, "gb_search_2")
        self.assertEqual(seen, [4])

    def test_an_unknown_profile_key_is_still_refused(self):
        from lisatools.globalfit.recipe import SearchStageProfileStep

        _, tree = _tree()
        with self.assertRaises(ValueError):
            SearchStageProfileStep(moves=tree, convergence_iter=2, plateau_branch="gb",
                                   profile=dict(fstat_refit_evry=4), stage_name="gb_search_2")


class DecisionFollowsTheAttributeTest(unittest.TestCase):
    """The head's refit decision reads ``fstat_refit_every`` per propose."""

    def _move(self, every, clock, last=100, epoch=7):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialRJFStatGridMove

        m = GBSpecialRJFStatGridMove.__new__(GBSpecialRJFStatGridMove)
        m.rj_proposal_distribution = object()      # an epoch is installed
        m._fstat_zero_peak_epoch = None
        m.fstat_refit_every = every
        m.fstat_refit_only_forced = False
        m._fstat_epoch = epoch
        m._fstat_last_fit_hit = last
        m._fstat_clock = lambda: clock
        return m

    def test_cadence_4_skips_one_cycle_and_refits_on_the_second(self):
        # three rows per cycle under legs: ages 3 then 6 at the F-stat propose
        self.assertEqual(self._move(4, 103)._fstat_fit_decision(), ("skip", 7))
        self.assertEqual(self._move(4, 106)._fstat_fit_decision(), ("fit", 8))
        # one or two ticks lost across a relaunch still refit on the second cycle
        self.assertEqual(self._move(4, 105)._fstat_fit_decision(), ("fit", 8))
        self.assertEqual(self._move(4, 104)._fstat_fit_decision(), ("fit", 8))

    def test_cadence_2_refits_every_cycle(self):
        self.assertEqual(self._move(2, 103)._fstat_fit_decision(), ("fit", 8))


class ProfileTableTest(unittest.TestCase):

    def test_unset_knobs_leave_every_profile_without_the_key(self):
        import run_combined_staged as R

        with env(**{k: None for k in REFIT_ENV}):
            rows = {n: p for n, p, _ in R.search_stage_profiles()}
        for n, p in rows.items():
            self.assertNotIn("fstat_refit_every", p, n)

    def test_the_stage_2_knob_writes_stage_2_only(self):
        import run_combined_staged as R

        with env(**{k: None for k in REFIT_ENV}), env(GB_SEARCH_2_FSTAT_REFIT_EVERY="4"):
            rows = {n: p for n, p, _ in R.search_stage_profiles()}
        self.assertEqual(rows["gb_search_2"]["fstat_refit_every"], 4)
        for n, p in rows.items():
            if n != "gb_search_2":
                self.assertNotIn("fstat_refit_every", p, n)

    def test_bad_values_are_refused(self):
        import run_combined_staged as R

        for bad in ("0", "-1", "two"):
            with env(**{k: None for k in REFIT_ENV}), env(GB_SEARCH_2_FSTAT_REFIT_EVERY=bad):
                with self.assertRaises(ValueError, msg=bad):
                    R.search_stage_profiles()


def _launchers():
    paths = [os.path.join(ROOT, "scripts", "flagship", "launcher.sh.in")]
    paths += sorted(glob.glob(os.path.join(ROOT, "scripts", "flagship", "submit_gf_*mo.sh")))
    paths += [os.path.join(ROOT, "scripts", "fstat_proposal", f) for f in (
        "submit_gf_3mo_v9_2gpu.sh", "submit_gf_6mo_v9_4gpu.sh",
        "submit_gf_9mo_v9.sh", "submit_gf_1yr_v9.sh")]
    return [p for p in paths if os.path.exists(p)]


def _default(text, knob):
    m = re.search(rf"^export {knob}=\$\{{{knob}:-([^}}]*)\}}", text, re.M)
    return m.group(1) if m else None


class LauncherDefaultsTest(unittest.TestCase):
    """The 2026-10-19 gb_search_2 defaults in every launcher and its source."""

    WANT = {"GB_SEARCH_2_NUM_REPEAT_PROPOSALS": "30",
            "GB_SEARCH_2_INMODEL_GROUP_ITERS": "1",
            "GB_SEARCH_2_INMODEL_CONVERGE_ITERS": "100",
            "GB_SEARCH_2_FSTAT_REFIT_EVERY": "4"}

    def test_every_launcher_carries_the_search_2_defaults(self):
        paths = _launchers()
        self.assertGreaterEqual(len(paths), 10)
        for p in paths:
            text = open(p).read()
            for knob, want in self.WANT.items():
                self.assertEqual(_default(text, knob), want, f"{os.path.basename(p)} {knob}")

    def test_search_1_keeps_the_every_cycle_refit(self):
        for p in _launchers():
            text = open(p).read()
            m = re.search(r"^export GB_FSTAT_REFIT_EVERY=(\S+)", text, re.M)
            self.assertIsNotNone(m, os.path.basename(p))
            self.assertEqual(m.group(1), "2", os.path.basename(p))
            self.assertIsNone(_default(text, "GB_SEARCH_1_FSTAT_REFIT_EVERY"), os.path.basename(p))


if __name__ == "__main__":
    unittest.main()
