"""WalkerFanoutMixin: direct call with one compute rank; slice/run/merge with several."""

import os
import unittest
from unittest import mock

import numpy as np
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin, pooled_ladder_step
from lisatools.globalfit.state import GFState
from tests.test_gf_substate_roundtrip import NTEMPS, NWALKERS, make_state


class _Curr:
    def __init__(self, fanout, rank):
        self.fanout = fanout
        self.rank = rank


class _StubMove(WalkerFanoutMixin):
    """A move whose body is trivial but leaves fingerprints the merge must preserve."""

    gf_move_name = "stub"
    fanout_branches = ["mbh"]
    fanout_assigns_counters = True

    def __init__(self, tag, tc=None, assigns_counters=True):
        self.tag = float(tag)
        self.tc = tc if tc is not None else TemperatureControl(2, 2, ntemps=NTEMPS, permute=False)
        self.fanout_assigns_counters = assigns_counters
        self.eigen_store_path = "store.h5"
        self.applied = []
        self.merged = None
        self.seen_models = []

    def fanout_temperature_controls(self):
        return [self.tc]

    def fanout_payload_extra(self):
        return {"tick": 7}

    def fanout_apply_extra(self, extra):
        self.applied.append(extra["tick"])

    def fanout_reply_extra(self, part):
        return {"nw": int(part.branches["mbh"].coords.shape[1])}

    def fanout_merge_extra(self, replies, new_state):
        self.merged = replies

    def propose_local(self, model, state):
        self.seen_models.append(model)
        new = GFState(state, copy=True)
        new.branches["mbh"].coords[...] += 1.0
        nw = new.log_like.shape[1]
        new.log_like[...] = np.arange(nw)[None, :] + self.tag  # tag = which rank scored it
        sub = new.sub_states["mbh"]
        sub.in_model_accepted[...] = 1
        # accepted pattern is rank-dependent so a mis-ordered merge is visible
        accepted = np.full(new.log_like.shape, self.tag == 0.0, dtype=bool)
        return new, accepted


class MixinSingleTest(unittest.TestCase):
    def test_no_fanout_is_a_direct_call(self):
        move = _StubMove(0.0)
        state = make_state(np.random.default_rng(1))
        ref = GFState(state, copy=True)
        new, acc = move.propose("model", state)
        np.testing.assert_array_equal(new.branches["mbh"].coords, ref.branches["mbh"].coords + 1)
        self.assertEqual(acc.shape, (NTEMPS, NWALKERS))
        self.assertEqual(move.applied, [])  # no payload round trip
        self.assertFalse(move.fanout_active)

    def test_install_single_leaves_ladders_and_sidecar_alone(self):
        layout = FakeWorld(1).run(lambda r, c: build_layout(c, NWALKERS, [0], legacy=False))[0]
        fo = WalkerFanout(None, layout, 0)
        move = _StubMove(0.0)
        move.install_walker_fanout(_Curr(fo, 0))
        self.assertTrue(move.tc.adaptive)
        self.assertEqual(move.eigen_store_path, "store.h5")
        self.assertFalse(move.fanout_active)


class MixinFakeWorldTest(unittest.TestCase):
    SEED_COUNTER = 5  # pre-existing delta count on the head's incoming state

    def _run_two_ranks(self, assigns_counters):
        world = FakeWorld(3, nodes=[0, 0, 0])
        moves = {}

        def fn(rank, comm):
            layout = build_layout(comm, NWALKERS, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            role = layout.role_of(rank)
            if role == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move = _StubMove(10.0 * rank, assigns_counters=assigns_counters)
            move.install_walker_fanout(_Curr(fo, rank))
            moves[rank] = move
            if role == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                # DELIBERATELY STALE: run_global_fit binds fanout.model once
                # at setup and never refreshes it, so the head body must use
                # the model ``propose`` was called with, not this one
                fo.model = "stale-fanout-model"
                state = make_state(np.random.default_rng(1))
                state.sub_states["mbh"].in_model_accepted[...] = self.SEED_COUNTER
                ref = GFState(state, copy=True)
                try:
                    new, acc = move.propose("head-model", state)
                finally:
                    fo.stop()
                return ref, new, acc
            return ComputeService(
                fcomm, layout, rank, registry={("pe", "stub"): move}, model="worker-model"
            ).serve()

        out = world.run(fn)
        return out, moves

    def test_two_compute_ranks_slice_run_merge(self):
        out, moves = self._run_two_ranks(assigns_counters=True)
        ref, new, acc = out[0]
        self.assertEqual(out[1], 1)  # one propose served
        w0, w1 = 0, NWALKERS // 2
        np.testing.assert_array_equal(new.branches["mbh"].coords, ref.branches["mbh"].coords + 1)
        # the head scored walkers [0, N/2) (tag 0) and the worker [N/2, N) (tag 10)
        head_cols = np.arange(w1 - w0) + 0.0
        worker_cols = np.arange(NWALKERS - w1) + 10.0
        np.testing.assert_array_equal(new.log_like[0], np.concatenate([head_cols, worker_cols]))
        # accepted: head block True, worker block False -> a mis-ordered merge shows
        self.assertEqual(acc.shape, (NTEMPS, NWALKERS))
        self.assertTrue(acc[:, w0:w1].all())
        self.assertFalse(acc[:, w1:].any())
        # the body ASSIGNS counters: the head copy (5) is zeroed, then 1 + 1 summed
        self.assertTrue(np.all(new.sub_states["mbh"].in_model_accepted == 2))
        # the input state is never mutated by the head's merge
        self.assertTrue(np.all(ref.sub_states["mbh"].in_model_accepted == self.SEED_COUNTER))
        # untouched sub-states are left alone
        for name, sub in new.sub_states.items():
            if name != "mbh" and sub is not None:
                np.testing.assert_array_equal(sub.coords, ref.sub_states[name].coords)
        # clock round trip on both ranks; extras merged per rank; command clock kept
        self.assertEqual(moves[0].applied, [7])
        self.assertEqual(moves[1].applied, [7])
        self.assertEqual(moves[0].merged, {0: {"nw": w1 - w0}, 1: {"nw": NWALKERS - w1}})
        self.assertEqual(moves[1].gf_clock["stage"], "pe")
        self.assertEqual(moves[1].gf_clock["move"], "stub")
        # ranks never adapt; the configured value is remembered on the control
        for rank in (0, 1):
            self.assertFalse(moves[rank].tc.adaptive)
            self.assertTrue(moves[rank].tc.gf_configured_adaptive)
        self.assertEqual(moves[0].eigen_store_path, "store.h5")  # head keeps the sidecar
        self.assertFalse(getattr(moves[0], "eigen_store_readonly", False))
        # the worker keeps the PATH but read-only (2026-09-17): it adopts the
        # head's walker-independent tables and never writes the file
        self.assertEqual(moves[1].eigen_store_path, "store.h5")
        self.assertTrue(moves[1].eigen_store_readonly)
        # the head body ran against propose()'s model, NOT fanout.model
        self.assertEqual(moves[0].seen_models, ["head-model"])
        # the worker keeps its ComputeService model (its own live one)
        self.assertEqual(moves[1].seen_models, ["worker-model"])

    def test_counters_accumulate_when_the_body_adds(self):
        out, _moves = self._run_two_ranks(assigns_counters=False)
        _ref, new, _acc = out[0]
        # no zeroing: the incoming 5 plus the two rank slices' 1 each
        self.assertTrue(np.all(new.sub_states["mbh"].in_model_accepted == self.SEED_COUNTER + 2))

    def test_gf_serve_rejects_other_ops(self):
        move = _StubMove(0.0)
        with self.assertRaises(ValueError):
            move.gf_serve("score", {}, {}, None)

    def test_gf_serve_refuses_an_initialized_unlisted_sub_state(self):
        # the body is handed a slice whose unlisted sub-states are None; a body
        # that INITIALIZES one of them (here: every branch, via a full state)
        # violates the fanout_branches contract and must not be silently dropped
        move = _StubMove(0.0)
        full = make_state(np.random.default_rng(2))  # every sub-state initialized
        with self.assertRaisesRegex(RuntimeError, "not in fanout_branches"):
            move.gf_serve("propose", {"state": full, "extra": {"tick": 1}}, {}, None)

    def test_gf_serve_nulls_bare_unlisted_sub_states(self):
        from lisatools.globalfit.communication.walkerslice import slice_state

        move = _StubMove(0.0)
        part = slice_state(make_state(np.random.default_rng(2)), 0, 2, sub_states=["mbh"])
        reply = move.gf_serve("propose", {"state": part, "extra": {"tick": 1}}, {}, None)
        for name, sub in reply["state"].sub_states.items():
            if name == "mbh":
                self.assertTrue(sub.tempered_initialized)
            else:
                self.assertIsNone(sub)


class _ProposeOverridingMove(_StubMove):
    """The trap: a subclass whose preamble lives in ``propose`` runs on the head only."""

    def propose(self, model, state):
        return self.propose_local(model, state)


class ProposeOverrideGuardTest(unittest.TestCase):
    def _fanout(self, compute_ranks):
        # one compute rank => a 1-rank world (a size-2 world would demote rank 1
        # to the saver and warn); several => + the dedicated saver rank
        size = 1 if len(compute_ranks) == 1 else len(compute_ranks) + 1
        layout = FakeWorld(size).run(
            lambda r, c: build_layout(c, NWALKERS, list(compute_ranks), legacy=False)
        )[0]
        return WalkerFanout(None, layout, 0)

    def test_multi_rank_install_refuses_a_propose_override(self):
        move = _ProposeOverridingMove(0.0)
        with self.assertRaisesRegex(TypeError, "propose_local"):
            move.install_walker_fanout(_Curr(self._fanout([0, 1]), 0))
        # refused before anything is touched: the ladder is left configured
        self.assertTrue(move.tc.adaptive)
        self.assertFalse(hasattr(move.tc, "gf_configured_adaptive"))

    def test_single_rank_install_allows_it(self):
        move = _ProposeOverridingMove(0.0)
        move.install_walker_fanout(_Curr(self._fanout([0]), 0))
        self.assertFalse(move.fanout_active)

    def test_the_ported_families_do_not_override_propose(self):
        from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
        from lisatools.globalfit.moves.mbhspecialmove import MBHSpecialMove
        from lisatools.globalfit.moves.psdmove import MultiGPUPSDMove, PSDMove

        for cls in (ResidualAddOneRemoveOneMove, MBHSpecialMove, PSDMove, MultiGPUPSDMove):
            self.assertIs(cls.propose, WalkerFanoutMixin.propose, cls.__name__)


class _ACSModel:
    """A model whose ACA reports a walker-row count."""

    def __init__(self, entries):
        self.analysis_container_arr = type("A", (), {"acs_total_entries": entries})()


class HeadACAWidthGuardTest(unittest.TestCase):
    """The mixin's ACA-width rule: the head's ACA is its BLOCK, not the ensemble.

    The raise branch was untested (re-review NEW-3) -- the other mixin tests
    pass a string model, so ``entries is None`` short-circuits the guard.
    """

    def _move(self):
        layout = FakeWorld(3).run(
            lambda r, c: build_layout(c, NWALKERS, [0, 1], legacy=False))[0]
        move = _StubMove(0.0)
        move.install_walker_fanout(_Curr(WalkerFanout(None, layout, 0), 0))
        self.assertTrue(move.fanout_active)
        return move, layout

    def test_an_ensemble_width_aca_raises_before_any_fan_out(self):
        move, layout = self._move()
        w0, w1 = layout.block_of(0)
        state = make_state(np.random.default_rng(1))
        with self.assertRaises(RuntimeError) as ctx:
            move.propose(_ACSModel(NWALKERS), state)  # block is NWALKERS // 2
        msg = str(ctx.exception)
        self.assertIn("_StubMove", msg)
        self.assertIn(f"{NWALKERS} walker rows", msg)
        self.assertIn(f"[{w0}, {w1}) ({w1 - w0} walkers)", msg)
        # raised before the (None) fan-out comm was ever touched
        self.assertEqual(move.seen_models, [])


class SharedControlInstallTest(unittest.TestCase):
    def test_second_install_keeps_the_configured_adaptive(self):
        # the PSD search and PE moves share ONE TemperatureControl: the second
        # install must remember the CONFIGURED value, not the already-False one
        layout = FakeWorld(3).run(lambda r, c: build_layout(c, NWALKERS, [0, 1], legacy=False))[0]
        fo = WalkerFanout(None, layout, 0)
        tc = TemperatureControl(2, 2, ntemps=NTEMPS, permute=False)
        self.assertTrue(tc.adaptive)
        first, second = _StubMove(0.0, tc=tc), _StubMove(0.0, tc=tc)
        first.install_walker_fanout(_Curr(fo, 0))
        second.install_walker_fanout(_Curr(fo, 0))
        self.assertFalse(tc.adaptive)
        self.assertTrue(tc.gf_configured_adaptive)


class PooledLadderStepTest(unittest.TestCase):
    def test_matches_one_eryn_adjustment_and_advances_time(self):
        tc = TemperatureControl(2, 4, ntemps=4, permute=False)
        tc.gf_configured_adaptive = True
        betas0 = np.array(tc.betas, copy=True)
        acc = np.array([3.0, 1.0, 2.0])
        prop = np.array([8.0, 8.0, 8.0])
        expect = betas0 + tc._get_ladder_adjustment(0, betas0.copy(), acc / prop)
        out = pooled_ladder_step(tc, betas0, acc, prop)
        np.testing.assert_allclose(out, expect)
        self.assertEqual(tc.time, 1)
        # a sequential per-rank adaptation (two half-size steps) differs from the pooled one
        seq = betas0 + tc._get_ladder_adjustment(0, betas0.copy(), np.array([2.0, 0.0, 1.0]) / 4)
        seq = seq + tc._get_ladder_adjustment(1, seq.copy(), np.array([1.0, 1.0, 1.0]) / 4)
        self.assertFalse(np.allclose(out, seq))

    def test_not_adaptive_or_single_rung_is_identity(self):
        tc = TemperatureControl(2, 4, ntemps=4, permute=False)
        tc.gf_configured_adaptive = False
        betas0 = np.array(tc.betas, copy=True)
        np.testing.assert_array_equal(
            pooled_ladder_step(tc, betas0, np.ones(3), np.ones(3)), betas0
        )
        self.assertEqual(tc.time, 1)
        one = TemperatureControl(2, 4, ntemps=1, permute=False)
        one.gf_configured_adaptive = True
        np.testing.assert_array_equal(
            pooled_ladder_step(one, np.array(one.betas), np.zeros(0), np.zeros(0)), one.betas
        )


class _ReplicaStub(WalkerFanoutMixin):
    gf_move_name = "stub"
    fanout_branches = ["mbh"]
    branch_name = "mbh"

    def __init__(self):
        self.calls = []
        self.tc = TemperatureControl(2, 1, ntemps=3, permute=False)
        self.tc.adaptive = True

    def fanout_temperature_controls(self):
        return [self.tc]

    def propose_local(self, model, state):
        self.calls.append((model, state))
        return state, np.zeros((3, 1), dtype=bool)

    def serve_echo(self, payload, clock, model):
        return {"echo": payload, "stage": clock.get("stage")}


class ReplicaModeMixinTest(unittest.TestCase):
    def _world(self, head_fn, env=None):
        world = FakeWorld(3, nodes=[0, 0, 0])
        moves = {}

        def fn(rank, comm):
            layout = build_layout(comm, 1, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move = _ReplicaStub()
            with mock.patch.dict(os.environ, env or {}):
                move.install_walker_fanout(_Curr(fo, rank))
            moves[rank] = move
            if layout.role_of(rank) == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                try:
                    return head_fn(move, fo)
                finally:
                    fo.stop()
            return ComputeService(fcomm, layout, rank, registry={("pe", "stub"): move}).serve()

        return world.run(fn), moves

    def test_head_runs_the_body_on_the_full_state_and_ranks_serve_nothing(self):
        sentinel = object()

        def head(move, fo):
            new, acc = move.propose("head-model", sentinel)
            return new is sentinel, acc.shape, move.calls[0][0]

        out, moves = self._world(head)
        self.assertEqual(out[0], (True, (3, 1), "head-model"))
        self.assertEqual(out[1], 0)  # the worker served no propose
        for r in (0, 1):
            self.assertIsNotNone(moves[r].row_fanout)
            self.assertTrue(moves[r].rows_active())
            self.assertTrue(moves[r].tc.adaptive)  # single-process semantics: ladders adapt in the body

    def test_knob_off_keeps_rows_local(self):
        out, moves = self._world(lambda move, fo: move.rows_active(),
                                 env={"MBH_LIKELIHOOD_FANOUT": "0"})
        self.assertFalse(out[0])
        self.assertFalse(moves[0].likelihood_fanout)

    def test_gf_serve_dispatches_serve_methods(self):
        move = _ReplicaStub()
        out = move.gf_serve("echo", {"a": 1}, {"stage": "pe"}, None)
        self.assertEqual(out, {"echo": {"a": 1}, "stage": "pe"})
        self.assertEqual(move.gf_clock, {"stage": "pe"})
        with self.assertRaises(ValueError):
            move.gf_serve("nope", {}, {}, None)
        with self.assertRaises(ValueError):
            move.gf_serve("_private", {}, {}, None)


if __name__ == "__main__":
    unittest.main()


class _BlockStub(_StubMove):
    """A NON-flat family (PSD/galfor's shape) that implements the replica sync."""

    fanout_flat_body = False

    def __init__(self, tag, **kw):
        super().__init__(tag, **kw)
        self.synced = []

    def fanout_replica_sync_payload(self, new_state, w0, w1):
        return {"w": (int(w0), int(w1)),
                "ll": np.array(new_state.log_like[0, w0:w1], copy=True),
                "coords": np.array(new_state.branches["mbh"].coords[0, w0:w1], copy=True)}

    def fanout_apply_replica_sync(self, payload):
        self.synced.append(payload)


class _NoSyncBlockStub(_StubMove):
    """A non-flat family WITHOUT the sync hooks: refused under replicas."""

    fanout_flat_body = False


#: ``NGPUS=8 GF_GPU_ROUTING=1`` on 4 walkers: the run world (monitor split off)
#: over 4 nodes, ``-ppn 1`` round robin -> 8 compute ranks, 4 blocks x R = 2
NODES_4x2 = [0, 1, 2, 3, 0, 1, 2, 3, 0]


class ReplicaBlockPathTest(unittest.TestCase):
    """The BLOCK path under replicas over several blocks (2026-10-15).

    PSD/galfor are not flat-capable, so at 4 walkers x R = 2 only each
    block's LEAD proposes. Before 2026-10-15 the family still got a
    ``RowFanout`` there (``replica_mode`` alone decided it), so its body issued
    a second fan-out command from INSIDE the propose command -- head-only on a
    worker lead, nested on the head -- and nothing ever told the passive
    replicas that their block's noise model had moved. Now the body runs
    block-local and the head publishes the merged result in a SECOND command.
    """

    def _run(self, cls=_BlockStub):
        world = FakeWorld(9, nodes=NODES_4x2)
        moves = {}

        def fn(rank, comm):
            layout = build_layout(comm, NWALKERS, [0, 1], legacy=False, gpu_routing=True)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move = cls(10.0 * rank)
            move.install_walker_fanout(_Curr(fo, rank))
            moves[rank] = move
            if layout.role_of(rank) == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                state = make_state(np.random.default_rng(1))
                ref = GFState(state, copy=True)
                try:
                    new, acc = move.propose("head-model", state)
                finally:
                    fo.stop()
                return ref, new, acc, layout
            return ComputeService(
                fcomm, layout, rank, registry={("pe", "stub"): move}, model="worker-model"
            ).serve()

        return world.run(fn), moves

    def test_only_the_leads_propose_and_the_passives_get_their_block(self):
        out, moves = self._run()
        ref, new, acc, layout = out[0]
        self.assertEqual((layout.n_blocks, layout.ranks_per_block), (4, 2))
        leads, passive = layout.block_leads, (1, 3, 5, 7)
        self.assertEqual(leads, (0, 2, 4, 6))
        for r in leads:
            self.assertEqual(len(moves[r].seen_models), 1, r)       # proposed once
            self.assertEqual(moves[r].synced, [], r)                 # nothing to apply
        for r in passive:
            self.assertEqual(moves[r].seen_models, [], r)            # never proposed
            self.assertEqual(len(moves[r].synced), 1, r)             # one publish
            got = moves[r].synced[0]
            b = r // 2
            self.assertEqual(got["w"], (b, b + 1))
            # the payload is the MERGED state's block: the lead's lnL tag
            np.testing.assert_allclose(got["ll"], [10.0 * (r - 1)])
            np.testing.assert_allclose(got["coords"], ref.branches["mbh"].coords[0, b:b + 1] + 1)
        # each walker's lnL is its lead's, in walker order
        np.testing.assert_allclose(new.log_like[0], [0.0, 20.0, 40.0, 60.0])
        # passive ranks answered TWO commands: the passive propose + the sync
        self.assertEqual(out[1], 2)
        self.assertEqual(out[2], 2)                                  # a lead: propose + no-op sync
        # the merge extras come from one representative per block
        self.assertEqual(sorted(moves[0].merged), [0, 2, 4, 6])

    def test_the_block_rules_apply_no_row_scatter_and_ladders_pooled(self):
        _out, moves = self._run()
        for r, m in moves.items():
            self.assertIsNone(m.row_fanout, r)
            self.assertFalse(m.rows_active(), r)
            self.assertFalse(m.tc.adaptive, r)            # the head pools ONE step
            self.assertTrue(m.tc.gf_configured_adaptive, r)
        self.assertFalse(getattr(moves[0], "eigen_store_readonly", False))
        self.assertTrue(moves[1].eigen_store_readonly)

    def test_a_family_without_the_sync_is_refused_at_install(self):
        layout = FakeWorld(9, nodes=NODES_4x2).run(
            lambda r, c: build_layout(c, NWALKERS, [0, 1], legacy=False,
                                      gpu_routing=True))[0]
        fo = WalkerFanout(None, layout, 0)
        with self.assertRaisesRegex(TypeError, "replica sync"):
            _NoSyncBlockStub(0.0).install_walker_fanout(_Curr(fo, 0))
        # a flat family at the same layout is untouched: rows scatter as before
        flat = _StubMove(0.0)
        flat.install_walker_fanout(_Curr(fo, 0))
        self.assertIsNotNone(flat.row_fanout)

    def test_likelihood_fanout_off_is_refused_with_several_blocks(self):
        """``{BRANCH}_LIKELIHOOD_FANOUT=0`` means "the head scores every row
        itself", which needs the head to HOLD every walker. With 4 blocks it
        holds one, so the knob is kept ON with a warning instead of indexing
        rows that are not there."""
        import logging

        layout = FakeWorld(9, nodes=NODES_4x2).run(
            lambda r, c: build_layout(c, NWALKERS, [0, 1], legacy=False,
                                      gpu_routing=True))[0]
        fo = WalkerFanout(None, layout, 0)
        move = _StubMove(0.0)
        move.branch_name = "mbh"
        with mock.patch.dict(os.environ, {"MBH_LIKELIHOOD_FANOUT": "0"}), \
                self.assertLogs("lisatools.globalfit.moves.walkerfanout",
                                level=logging.WARNING) as cm:
            move.install_walker_fanout(_Curr(fo, 0))
        self.assertTrue(move.likelihood_fanout)
        self.assertTrue(move.rows_active())
        self.assertIn("MBH_LIKELIHOOD_FANOUT=0 ignored", "\n".join(cm.output))

    def test_the_psd_family_implements_the_sync(self):
        from lisatools.globalfit.moves.psdmove import PSDMove
        for name in ("fanout_replica_sync_payload", "fanout_apply_replica_sync"):
            self.assertIsNot(getattr(PSDMove, name), getattr(WalkerFanoutMixin, name), name)


class FlatBodyOptOutTest(unittest.TestCase):
    """``fanout_flat_body`` keeps a family on the BLOCK path under replicas.

    Once ``replica_mode`` became ``R > 1``, ``propose`` routed EVERY family
    to the full-ensemble body. That is right for addremove, whose every
    evaluation funnels through one scoring seam, and wrong for PSD/galfor,
    which read their walker count and device map off the LOCAL ACA. Those
    families opt out and keep the block path at any R -- with only the block
    LEAD proposing, because the group-mates hold the same walkers and R
    bodies would draw R different proposals for one block.
    """

    def test_the_mixin_defaults_to_flat_capable(self):
        self.assertTrue(WalkerFanoutMixin.fanout_flat_body)

    def test_psd_opts_out(self):
        from lisatools.globalfit.moves.psdmove import PSDMove
        self.assertFalse(PSDMove.fanout_flat_body)

    def test_addremove_stays_flat_capable(self):
        from lisatools.globalfit.moves.addremovemove import (
            ResidualAddOneRemoveOneMove,
        )
        self.assertTrue(ResidualAddOneRemoveOneMove.fanout_flat_body)

    def test_a_flat_family_runs_the_full_body_under_replicas(self):
        move = _ReplicaStub()
        move.fanout = mock.Mock()
        move.fanout.single = False
        move.fanout.layout = mock.Mock(replica_mode=True, n_blocks=4)
        move.fanout_propose = mock.Mock(
            side_effect=AssertionError("must not take the block path"))
        state = object()
        move.propose("model", state)
        self.assertEqual(len(move.calls), 1)          # propose_local ran
        self.assertIs(move.calls[0][1], state)        # with the FULL state

    def test_a_non_flat_family_takes_the_block_path_with_SEVERAL_blocks(self):
        move = _ReplicaStub()
        move.fanout_flat_body = False
        move.fanout = mock.Mock()
        move.fanout.single = False
        move.fanout.layout = mock.Mock(replica_mode=True, n_blocks=2)
        move.fanout_propose = mock.Mock(return_value=("merged", "acc"))
        out = move.propose("model", "state")
        self.assertEqual(out, ("merged", "acc"))
        self.assertEqual(move.calls, [])              # the body did NOT run here
        move.fanout_propose.assert_called_once()

    def test_a_non_flat_family_STILL_goes_flat_at_one_block(self):
        """REGRESSION (2026-09-23): one-walker replica mode must stay flat.

        ``fanout_flat_body`` is about a state WIDER than the rank's ACA. At
        ``n_blocks == 1`` the block IS the ensemble, so there is no gap and
        even an ACA-sizing body is correct -- and that case is exactly
        one-walker replica mode, which has always used the flat path.
        Forcing it onto the block path put ``propose_local`` inside a
        ``propose`` command, whose ``_replay_noise_begin`` then issued a
        second fan-out command from within the first: a NESTED fan-out,
        surfacing as ``fan-out sequence mismatch``. Caught by the
        multi-rank noise smoke, not by a mock.
        """
        move = _ReplicaStub()
        move.fanout_flat_body = False
        move.fanout = mock.Mock()
        move.fanout.single = False
        move.fanout.layout = mock.Mock(replica_mode=True, n_blocks=1)
        move.fanout_propose = mock.Mock(
            side_effect=AssertionError("one block must NOT take the block path"))
        state = object()
        move.propose("model", state)
        self.assertEqual(len(move.calls), 1)
        self.assertIs(move.calls[0][1], state)

    def test_flat_body_ok_truth_table(self):
        move = _ReplicaStub()
        for flat, nb, expect in ((True, 1, True), (True, 4, True),
                                 (False, 1, True), (False, 4, False)):
            with self.subTest(fanout_flat_body=flat, n_blocks=nb):
                move.fanout_flat_body = flat
                self.assertEqual(
                    move._flat_body_ok(mock.Mock(n_blocks=nb)), expect)

    def test_a_passive_reply_is_shaped_like_a_real_one_and_changes_nothing(self):
        move = _ReplicaStub()

        class _Part:
            branches_coords = {"mbh": np.zeros((3, 2, 1, 4))}
            sub_states = {"mbh": object(), "other": object()}

        part = _Part()
        rep = move._passive_propose_reply({"state": part})
        self.assertEqual(set(rep), {"state", "accepted", "extra"})
        self.assertIs(rep["state"], part)             # the slice, unchanged
        self.assertEqual(rep["accepted"].shape, (3, 2))
        self.assertFalse(rep["accepted"].any())       # nothing accepted
        self.assertEqual(rep["extra"], {})
        # sub-states outside fanout_branches are nulled, as gf_serve does
        self.assertIsNone(part.sub_states["other"])
        self.assertIsNotNone(part.sub_states["mbh"])

    def test_the_passive_reply_survives_an_empty_branch_set(self):
        move = _ReplicaStub()

        class _Part:
            branches_coords = {}
            sub_states = {}

        rep = move._passive_propose_reply({"state": _Part()})
        self.assertEqual(rep["accepted"].shape, (1, 1))
        self.assertFalse(rep["accepted"].any())
