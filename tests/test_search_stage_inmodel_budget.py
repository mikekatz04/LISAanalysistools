"""The per-stage in-model budget (Mike 2026-10-16).

"Gb search 2 should spend less time in the in-model mixing but maybe with
some quick checker to make sure the logL is not seriously improving during
those steps anymore"; gb_search_1 stays as is. The checkers are the existing
group rule (pure in-model moves) and per-row polish rule (RJ moves); the
stage profile key ``inmodel_budget`` shortens their windows and the repeats
per pass in ONE stage:

* ``GB_SEARCH_{N}_NUM_REPEAT_PROPOSALS`` / ``_INMODEL_GROUP_ITERS`` /
  ``_INMODEL_CONVERGE_ITERS`` -> ``run_combined_staged.search_stage_profiles``
  writes ``inmodel_budget`` on that stage's profile only;
* ``recipe.apply_inmodel_budget`` (from ``SearchStageProfileStep._apply_profile``)
  routes the first two to the pure in-model moves and the third to the RJ
  moves, and RESTORES the built values on a stage without the key (the move
  objects are shared across stages);
* the moves read the attributes per block / per propose, and the two armed
  lines re-log when the window changes; the attributes ride to the compute
  ranks in ``GB_RANK_LIVE_ATTRS``.
"""
import contextlib
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "fstat_proposal"))


def _make_move(n):
    """The convergence suite's bare-move builder, from either import spelling."""
    try:
        from tests.test_inmodel_converge import _make_move as mk
    except ImportError:
        sys.path.insert(0, HERE)
        from test_inmodel_converge import _make_move as mk
    return mk(n)


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


BUDGET_ENV = ("GB_SEARCH_1_NUM_REPEAT_PROPOSALS", "GB_SEARCH_1_INMODEL_GROUP_ITERS",
              "GB_SEARCH_1_INMODEL_CONVERGE_ITERS", "GB_SEARCH_2_NUM_REPEAT_PROPOSALS",
              "GB_SEARCH_2_INMODEL_GROUP_ITERS", "GB_SEARCH_2_INMODEL_CONVERGE_ITERS",
              "GB_SEARCH_3_NUM_REPEAT_PROPOSALS", "GB_SEARCH_3_INMODEL_GROUP_ITERS",
              "GB_SEARCH_3_INMODEL_CONVERGE_ITERS")


def _gb(name, rj, **kw):
    """A GB move as the profile sees it: the built budget attributes included."""
    base = dict(name=name, branch_name="gb", opt_snr_rej_samp_limit=8.0,
                phase_maximize=False, _snr_lim_table=None, is_rj_prop=rj,
                rj_removal_only=False, rj_replace=False,
                num_repeat_proposals=25, inmodel_group_iters=3,
                inmodel_converge_iters=250)
    base.update(kw)
    return SimpleNamespace(**base)


def _tree():
    moves = [_gb("rj_warm_search", True), _gb("in_model", False),
             _gb("rj_fstat_search", True), _gb("in_model_fstat", False),
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


class ApplyInmodelBudgetTest(unittest.TestCase):

    def tearDown(self):
        _clear_peak_floor_override()

    def test_repeats_and_group_window_reach_the_in_model_moves_only(self):
        by, tree = _tree()
        _step(dict(opt_snr=5.0, inmodel_budget=dict(num_repeat_proposals=15,
                                                    inmodel_group_iters=2)), tree)
        for n in ("in_model", "in_model_fstat", "in_model_removal"):
            self.assertEqual(by[n].num_repeat_proposals, 15, n)
            self.assertEqual(by[n].inmodel_group_iters, 2, n)
            self.assertEqual(by[n].inmodel_converge_iters, 250, n)     # not theirs
        for n in ("rj_warm_search", "rj_fstat_search", "rj_prior_removal"):
            self.assertEqual(by[n].num_repeat_proposals, 25, n)        # untouched
            self.assertEqual(by[n].inmodel_group_iters, 3, n)
            self.assertEqual(by[n].inmodel_converge_iters, 250, n)

    def test_the_polish_window_reaches_the_rj_moves_only(self):
        by, tree = _tree()
        _step(dict(inmodel_budget=dict(inmodel_converge_iters=100)), tree)
        for n in ("rj_warm_search", "rj_fstat_search", "rj_prior_removal"):
            self.assertEqual(by[n].inmodel_converge_iters, 100, n)
        for n in ("in_model", "in_model_fstat", "in_model_removal"):
            self.assertEqual(by[n].inmodel_converge_iters, 250, n)

    def test_a_stage_without_the_key_restores_the_built_values(self):
        """gb_search_1 after (or before) a gb_search_2 budget runs as built:
        the move objects are shared, so the restore is what keeps stage 1
        untouched on a resume order that visits stage 2 first."""
        by, tree = _tree()
        _step(dict(inmodel_budget=dict(num_repeat_proposals=15, inmodel_group_iters=2,
                                       inmodel_converge_iters=100)), tree, "gb_search_2")
        self.assertEqual(by["in_model"].num_repeat_proposals, 15)
        self.assertEqual(by["rj_warm_search"].inmodel_converge_iters, 100)
        _step(dict(opt_snr=8.0), tree, "gb_search_1")                  # no budget key
        self.assertEqual(by["in_model"].num_repeat_proposals, 25)
        self.assertEqual(by["in_model"].inmodel_group_iters, 3)
        self.assertEqual(by["rj_warm_search"].inmodel_converge_iters, 250)
        # the built values were stashed on the moves at the first application
        self.assertEqual(by["in_model"]._inmodel_budget_built,
                         dict(num_repeat_proposals=25, inmodel_group_iters=3,
                              inmodel_converge_iters=250))

    def test_the_entry_line_names_every_change(self):
        from lisatools.globalfit.recipe import apply_inmodel_budget

        by, tree = _tree()
        with self.assertLogs("lisatools.globalfit.recipe", level="INFO") as cm:
            changed = apply_inmodel_budget(list(by.values()),
                                           dict(num_repeat_proposals=15), "gb_search_2")
        self.assertEqual(changed, {"in_model.num_repeat_proposals": (25, 15),
                                   "in_model_fstat.num_repeat_proposals": (25, 15),
                                   "in_model_removal.num_repeat_proposals": (25, 15)})
        line = "\n".join(cm.output)
        self.assertIn("[V9-STAGE gb_search_2] in-model budget set on entry", line)
        self.assertIn("in_model.num_repeat_proposals 25 -> 15", line)
        # nothing to change = nothing logged
        self.assertEqual(apply_inmodel_budget(list(by.values()),
                                              dict(num_repeat_proposals=15)), {})

    def test_bad_keys_and_values_are_refused(self):
        from lisatools.globalfit.recipe import apply_inmodel_budget

        by, _ = _tree()
        with self.assertRaises(ValueError):
            apply_inmodel_budget(list(by.values()), dict(inmodel_post_gate_newborn=25))
        with self.assertRaises(ValueError):
            apply_inmodel_budget(list(by.values()), dict(num_repeat_proposals=0))

    def test_a_move_without_the_attributes_is_left_alone(self):
        """The legacy stand-ins the other profile tests use carry none of the
        budget attributes; the profile must not invent them."""
        from lisatools.globalfit.recipe import apply_inmodel_budget

        bare = SimpleNamespace(name="rj_fstat_search", is_rj_prop=True)
        self.assertEqual(apply_inmodel_budget([bare], dict(inmodel_converge_iters=100)), {})
        self.assertFalse(hasattr(bare, "inmodel_converge_iters"))


class ProfileTableTest(unittest.TestCase):

    def test_unset_knobs_leave_every_profile_without_the_key(self):
        import run_combined_staged as R

        with env(**{k: None for k in BUDGET_ENV}):
            rows = {n: p for n, p, _ in R.search_stage_profiles()}
        for n, p in rows.items():
            self.assertNotIn("inmodel_budget", p, n)

    def test_stage_2_knobs_write_stage_2_only(self):
        import run_combined_staged as R

        with env(**{k: None for k in BUDGET_ENV}), env(
                GB_SEARCH_2_NUM_REPEAT_PROPOSALS="15", GB_SEARCH_2_INMODEL_GROUP_ITERS="2",
                GB_SEARCH_2_INMODEL_CONVERGE_ITERS="100"):
            rows = {n: p for n, p, _ in R.search_stage_profiles()}
        self.assertEqual(rows["gb_search_2"]["inmodel_budget"],
                         dict(num_repeat_proposals=15, inmodel_group_iters=2,
                              inmodel_converge_iters=100))
        self.assertNotIn("inmodel_budget", rows["gb_search_1"])
        self.assertNotIn("inmodel_budget", rows["gb_search_3"])

    def test_a_partial_budget_carries_only_the_knobs_set(self):
        import run_combined_staged as R

        with env(**{k: None for k in BUDGET_ENV}), env(GB_SEARCH_2_INMODEL_CONVERGE_ITERS="100"):
            self.assertEqual(R.search_stage_inmodel_budget("gb_search_2"),
                             dict(inmodel_converge_iters=100))
            self.assertIsNone(R.search_stage_inmodel_budget("gb_search_1"))
        self.assertIsNone(R.search_stage_inmodel_budget("gb_search_seed"))

    def test_bad_values_are_refused(self):
        import run_combined_staged as R

        for bad in ("0", "ten", "-3"):
            with env(GB_SEARCH_2_NUM_REPEAT_PROPOSALS=bad):
                with self.assertRaises(ValueError):
                    R.search_stage_inmodel_budget("gb_search_2")


class MoveReadsTheLiveWindowTest(unittest.TestCase):
    """The RJ polish builds its state from ``inmodel_converge_iters`` per
    block, the pure in-model move its group state from ``inmodel_group_iters``
    per propose, and both armed lines re-log on a change."""

    def _rj_move(self):
        move = _make_move(100)
        move.name = "rj_fstat_search"
        move.inmodel_converge = "on"
        move.inmodel_converge_classes = frozenset({"newborn"})
        move.inmodel_converge_iters = 250
        move.inmodel_converge_dll = 4.0
        move.inmodel_converge_max = 20000
        move.inmodel_converge_stop_frac = 0.5
        move.inmodel_converge_gate_frac = 0.5
        move.inmodel_converge_refill = True
        move.ntemps = 24
        move.inmodel_repeats_newborn = 100
        move.inmodel_repeats_survivor = 50
        return move

    def test_the_polish_window_follows_the_attribute(self):
        move = self._rj_move()
        with self.assertLogs("lisatools.globalfit.moves.gbspecialstretch", "INFO") as cm:
            self.assertEqual(move._converge_state_for("newborn").window, 250)
        self.assertTrue(any("armed" in l and "window 250" in l for l in cm.output))
        # the same window again: no second armed line
        with self.assertNoLogs("lisatools.globalfit.moves.gbspecialstretch", "INFO"):
            move._converge_state_for("newborn")
        move.inmodel_converge_iters = 100                   # the stage profile
        with self.assertLogs("lisatools.globalfit.moves.gbspecialstretch", "INFO") as cm:
            st = move._converge_state_for("newborn")
        self.assertEqual(st.window, 100)
        self.assertTrue(any("armed" in l and "window 100" in l for l in cm.output))

    def test_the_group_window_and_repeats_follow_the_attributes(self):
        mv = _make_move(25)
        mv.name = "in_model"
        mv.is_rj_prop = False
        mv.inmodel_group = True
        mv.inmodel_group_iters = 3
        mv.inmodel_group_dll = 4.0
        mv.inmodel_group_max_passes = 20
        mv.inmodel_group_scale = "flat"
        mv.num_repeat_proposals = 25
        with self.assertLogs("lisatools.globalfit.moves.gbspecialstretch", "INFO") as cm:
            self.assertEqual(mv._group_state_or_none().window, 3)
        self.assertTrue(any("window 3 pass(es)" in l and "25 repeats" in l for l in cm.output))
        with self.assertNoLogs("lisatools.globalfit.moves.gbspecialstretch", "INFO"):
            mv._group_state_or_none()
        mv.inmodel_group_iters, mv.num_repeat_proposals = 2, 15        # the stage profile
        with self.assertLogs("lisatools.globalfit.moves.gbspecialstretch", "INFO") as cm:
            self.assertEqual(mv._group_state_or_none().window, 2)
        self.assertTrue(any("window 2 pass(es)" in l and "15 repeats" in l for l in cm.output))

    def test_the_budget_attributes_ride_to_the_ranks(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            GB_RANK_LIVE_ATTRS, gb_rank_live_attrs)

        for k in ("num_repeat_proposals", "inmodel_group_iters", "inmodel_converge_iters"):
            self.assertIn(k, GB_RANK_LIVE_ATTRS)
        mv = SimpleNamespace(name="in_model", num_repeat_proposals=15,
                             inmodel_group_iters=2, inmodel_converge_iters=np.int64(100))
        got = gb_rank_live_attrs(mv)
        self.assertEqual(got, {"num_repeat_proposals": 15, "inmodel_group_iters": 2,
                               "inmodel_converge_iters": 100})
        self.assertIsInstance(got["inmodel_converge_iters"], int)


if __name__ == "__main__":
    unittest.main()
