"""Head-set move knobs reach the compute ranks' bodies per propose.

A recipe step applies its stage profile (``opt_snr``, ``phase_maximize``,
``prior_births``) and its PE declarations (in-model repeats, RJ flip
fraction) to the HEAD's move objects: ``note_recipe_step`` runs on the
recipe's announce path, which only the head executes. The compute ranks hold
their own move objects built from the env at construction. 6mo jobs 675, 685
and 695 (4 walkers on 4 compute ranks) show ONE ``[V9-STAGE gb_search_3]
entering`` line against FOUR construction lines, and the
``[GB_ACCEPT rj-split rj_prior_removal]`` lines of job 695 carry births on
exactly one block and ``births 0`` on the other three: stage 3's prior births,
its opt_snr 8 -> 5 and its phase_maximize True -> False ran on one walker in
four. The fix ships the knobs with the propose and installs them around the
rank body:

* GB moves: ``GB_RANK_LIVE_ATTRS`` in ``clock_vals`` of every
  ``gb_run_proposal`` (``_enter_rank_block`` / ``_exit_rank_block``);
* every other fanned-out move: ``WalkerFanoutMixin.fanout_live_attrs`` under
  ``extra["_live_attrs"]`` (``gf_serve``).
"""

import inspect
import unittest
from types import SimpleNamespace

import numpy as np


class GbRankLiveAttrsTest(unittest.TestCase):

    def test_payload_carries_the_stage_and_pe_knobs_the_move_has(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            GB_RANK_LIVE_ATTRS, gb_rank_live_attrs)

        self.assertEqual(set(GB_RANK_LIVE_ATTRS),
                         {"opt_snr_rej_samp_limit", "phase_maximize", "rj_removal_only",
                          "rj_flip_fraction", "num_repeat_proposals",
                          # PE per-class budgets (user ruling 2026-10-04)
                          "inmodel_repeats_newborn", "inmodel_repeats_survivor",
                          # stage profile inmodel_budget (2026-10-16)
                          "inmodel_group_iters", "inmodel_converge_iters"})
        mv = SimpleNamespace(name="rj_prior_removal", opt_snr_rej_samp_limit=5.0,
                             phase_maximize=False, rj_removal_only=False,
                             rj_flip_fraction=np.float64(0.1), num_repeat_proposals=25,
                             nwalkers=4)
        got = gb_rank_live_attrs(mv)
        self.assertEqual(got, {"opt_snr_rej_samp_limit": 5.0, "phase_maximize": False,
                               "rj_removal_only": False, "rj_flip_fraction": 0.1,
                               "num_repeat_proposals": 25})
        self.assertIsInstance(got["rj_flip_fraction"], float)       # host scalar, picklable
        # a move without some knob ships what it has
        self.assertEqual(gb_rank_live_attrs(SimpleNamespace(phase_maximize=True)),
                         {"phase_maximize": True})
        self.assertEqual(gb_rank_live_attrs(SimpleNamespace()), {})

    def test_install_then_exit_restores_the_ranks_own_values(self):
        from lisatools.globalfit.moves import gbspecialstretch as M

        mv = M.GBSpecialStretchMove.__new__(M.GBSpecialStretchMove)
        # the rank's construction-time values (job 695: the search defaults)
        mv.opt_snr_rej_samp_limit = 8.0
        mv.phase_maximize = True
        mv.rj_removal_only = True
        mv.rj_flip_fraction = 1.0
        mv.num_repeat_proposals = 25
        shipped = {"opt_snr_rej_samp_limit": 5.0, "phase_maximize": False,
                   "rj_removal_only": False, "rj_flip_fraction": 0.1,
                   "num_repeat_proposals": 30, "foreign_knob": 7}
        saved = mv._install_rank_live_attrs(shipped)
        self.assertEqual(mv.opt_snr_rej_samp_limit, 5.0)
        self.assertFalse(mv.phase_maximize)
        self.assertFalse(mv.rj_removal_only)                           # prior BIRTHS on the rank
        self.assertEqual(mv.rj_flip_fraction, 0.1)
        self.assertEqual(mv.num_repeat_proposals, 30)
        self.assertFalse(hasattr(mv, "foreign_knob"))                  # never invents an attribute
        self.assertEqual(saved, {"opt_snr_rej_samp_limit": 8.0, "phase_maximize": True,
                                 "rj_removal_only": True, "rj_flip_fraction": 1.0,
                                 "num_repeat_proposals": 25})
        mv._restore_rank_live_attrs(saved)
        self.assertEqual((mv.opt_snr_rej_samp_limit, mv.phase_maximize, mv.rj_removal_only,
                          mv.rj_flip_fraction, mv.num_repeat_proposals),
                         (8.0, True, True, 1.0, 25))
        # nothing shipped (an older head) -> nothing changes, nothing to restore
        self.assertEqual(mv._install_rank_live_attrs(None), {})
        mv._restore_rank_live_attrs({})

    def test_the_block_protocol_is_wired_on_both_sides(self):
        """The helpers above only matter if the head SHIPS and the rank INSTALLS."""
        from lisatools.globalfit.moves import gbspecialstretch as M

        head = inspect.getsource(M.GBSpecialStretchMove._propose_orchestrated)
        self.assertIn('"live_attrs": gb_rank_live_attrs(self)', head)
        enter = inspect.getsource(M.GBSpecialStretchMove._enter_rank_block)
        self.assertIn('self._install_rank_live_attrs(cv.get("live_attrs"))', enter)
        exit_ = inspect.getsource(M.GBSpecialStretchMove._exit_rank_block)
        self.assertIn('self._restore_rank_live_attrs(saved.get("_live_attrs"))', exit_)


class _FannedMove:
    """A WalkerFanoutMixin move with the generic propose protocol and no fan-out bound."""

    def __init__(self):
        self.num_repeats = 10          # the rank's construction-time value
        self.seen = []
        self.extra_seen = None
        self.boom = False

    def propose_local(self, model, state):
        self.seen.append(self.num_repeats)
        if self.boom:
            raise RuntimeError("body failed")
        return state, np.zeros((1, 1), dtype=bool)

    def fanout_apply_extra(self, extra):
        self.extra_seen = dict(extra)


class GenericLiveAttrsTest(unittest.TestCase):

    def _move(self):
        from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

        # the stub FIRST so its propose_local / fanout_apply_extra win; the
        # mixin supplies gf_serve and the live-knob helpers
        cls = type("_M", (_FannedMove, WalkerFanoutMixin), {"fanout_branches": ()})
        return cls()

    def _payload(self, live):
        extra = {"betas": np.array([1.0, 0.5])}
        if live is not None:
            extra["_live_attrs"] = live
        return {"state": SimpleNamespace(sub_states={}), "extra": extra}

    def test_head_ships_num_repeats_by_default(self):
        from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

        self.assertEqual(WalkerFanoutMixin.fanout_live_attrs, ("num_repeats",))
        m = self._move()
        m.num_repeats = 25                                  # apply_inmodel_repeats on the head
        self.assertEqual(m.fanout_live_payload(), {"num_repeats": 25})

    def test_rank_body_runs_under_the_shipped_value_and_keeps_its_own_after(self):
        from lisatools.globalfit.moves import walkerfanout as WF

        m = self._move()
        reply = m.gf_serve(WF.PROPOSE_OP, self._payload({"num_repeats": 25, "foreign": 1}),
                           {"seq": 1}, None)
        self.assertEqual(m.seen, [25])                      # the body saw the head's value
        self.assertEqual(m.num_repeats, 10)                 # restored for the next command
        self.assertFalse(hasattr(m, "foreign"))
        # the subclass hook never sees the reserved key
        self.assertEqual(set(m.extra_seen), {"betas"})
        self.assertEqual(reply["accepted"].shape, (1, 1))

    def test_without_the_key_the_body_runs_on_its_own_value(self):
        from lisatools.globalfit.moves import walkerfanout as WF

        m = self._move()
        m.gf_serve(WF.PROPOSE_OP, self._payload(None), {"seq": 1}, None)
        self.assertEqual(m.seen, [10])
        self.assertEqual(m.num_repeats, 10)

    def test_a_failing_body_still_restores(self):
        from lisatools.globalfit.moves import walkerfanout as WF

        m = self._move()
        m.boom = True
        with self.assertRaises(RuntimeError):
            m.gf_serve(WF.PROPOSE_OP, self._payload({"num_repeats": 25}), {"seq": 1}, None)
        self.assertEqual(m.num_repeats, 10)

    def test_fanout_propose_puts_the_live_knobs_in_the_payload(self):
        """The head side of the generic protocol: ``extra`` carries ``_live_attrs``."""
        from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

        src = inspect.getsource(WalkerFanoutMixin.fanout_propose)
        self.assertIn('extra["_live_attrs"] = _live', src)
        self.assertIn("self.fanout_live_payload()", src)


if __name__ == "__main__":
    unittest.main()
