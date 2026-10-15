"""EMRIDirectLikeMove under the add/remove move's fan-out: REAL proposes on FakeComm ranks.

The direct EMRI move keeps ALL of :class:`ResidualAddOneRemoveOneMove`'s choreography;
its only per-replica state is the exposed-residual offset (``acs.likelihood()`` at
setup), armed by the expose/setup replays. Two full proposes (expose -> setup ->
eigen-table refresh -> in-model repeats -> fast-vs-production cross-check ->
inner-product record -> fold) of the REAL move, with the MBH toy problem of
``tests/test_mbh_batched_fanout.py`` (tiny WDM grid, two leaves, per-walker DIFFERENT
noise and PSD) and a direct adapter that renders the containers' own toy templates,
under :class:`FakeWorld` ranks, compared with a single-process run:

* ONE-WALKER REPLICAS (``nwalkers=1`` on 2 compute ranks + a saver): rows scatter over
  the replicas, the expose/setup/fold replay on every replica;
* WALKER BLOCKS (``nwalkers=4`` on 4 ranks = 1 walker/rank, the 6-month v9 layout).
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.domains import WDMSignal
from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import LIKELIHOOD_OP, ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.engine import GlobalFitInfo
from lisatools.globalfit.state import EMRIState, GFState
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.utility import asnumpy

try:
    from tests import test_mbh_batched_fanout as H
    from tests.test_mbh_batched_move import NT, _slow_gen
except ImportError:  # pragma: no cover - run from tests/
    import test_mbh_batched_fanout as H
    from test_mbh_batched_move import NT, _slow_gen

MOVE_NAME = "emri_pe"
_CLEARED_ENV = tuple(k for k in H._CLEARED_ENV if not k.startswith("MBH_")) + (
    "EMRI_PERMUTE_EVERY", "EMRI_EIGEN_REFRESH", "EMRI_EIGEN_EPS_REL", "EMRI_EIGEN_SCOPE",
    "EMRI_LIKELIHOOD_FANOUT", "EMRI_RECORD_DH", "EMRI_CHECK_LL_TOL", "EMRI_SWAP_DEBUG",
    "EMRI_DEBUG",
)


class _Direct:
    """Direct adapter rendering the containers' own toy templates (thread-safe)."""

    def __init__(self, wdm):
        self.domain_settings = wdm

    def templates(self, rows, **kwargs):
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        return np.stack([np.asarray(_slow_gen(*r).arr) for r in rows]), np.ones(len(rows), bool)


def _aca(walkers):
    wdm = H._wdm()
    acs_list = []
    for w in walkers:
        model = ("scirdv1", "mrdv1")[w % 2]
        ac = AnalysisContainer(WDMSignal(H._residual(w), wdm), XYZ2SensitivityMatrix(wdm, model=model))
        ac.signal_gen = {"emri": _slow_gen}
        acs_list.append(ac)
    return AnalysisContainerArray(acs_list)


def _state(walkers):
    tempered = H._tempered_coords(walkers)
    nw = len(walkers)
    state = GFState(
        {"emri": tempered[:1].copy()},
        inds={"emri": np.ones((1, nw, H.NLEAVES), dtype=bool)},
        log_like=np.zeros((1, nw)),
        log_prior=np.zeros((1, nw)),
        betas=np.ones(1),
        sub_state_bases={"emri": EMRIState},
    )
    sub = state.sub_states["emri"]
    sub.betas_all = np.tile(H.BETAS, (H.NLEAVES, 1))
    sub.initialize_tempered(H.NTEMPS, nw, H.NLEAVES, H.NDIM, coords=tempered)
    return state


def _make_move(acs, nwalkers, *, scope, permute_every, adaptive, refresh=1):
    from eryn.moves import EigenAxisMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from lisatools.globalfit.moves import EMRIDirectLikeMove

    priors = {"emri": ProbDistContainer({i: uniform_dist(lo, hi) for i, (lo, hi) in enumerate(H.PRIOR_BOUNDS)})}
    move = EMRIDirectLikeMove(
        "emri", (H.NTEMPS, nwalkers, H.NLEAVES, H.NDIM), None, {}, {}, acs, H.NUM_REPEATS,
        H._Identity(), priors, [(EigenAxisMove(mode="axis"), 1.0)],
        betas_all=np.tile(H.BETAS, (H.NLEAVES, 1)), permute_every=permute_every,
        direct_gen=_Direct(acs.acs.flatten()[0].data.settings), batch_max_size=H.BATCH,
        name=MOVE_NAME, eigen_refresh_every=refresh, eigen_table_scope=scope,
    )
    move.gf_move_name = MOVE_NAME
    move.fanout_branches = ["emri"]
    for tc in move.temperature_controls:
        tc.adaptive = bool(adaptive)
    spy = move._spy = dict(score_rows=0, score_calls=[], check_rows=0, setups=[], unarmed=[])
    local, chunk, check, setup = (move.compute_like_local, move._score_chunk,
                                  move._check_like_local, move.setup_likelihood_here)

    def compute_like_local(coords_in, data_index):
        if move._exposed_offset is None:
            spy["unarmed"].append(int(move._current_leaf))
        spy["score_rows"] += int(np.atleast_2d(np.asarray(coords_in)).shape[0])
        return local(coords_in, data_index)

    def score_chunk(adapter, coords, idx, leaf):
        spy["score_calls"].append((int(leaf), np.array(idx, copy=True)))
        return chunk(adapter, coords, idx, leaf)

    def check_like_local(coords_in, data_index):
        spy["check_rows"] += int(np.atleast_2d(np.asarray(coords_in)).shape[0])
        return check(coords_in, data_index)

    def setup_likelihood_here(coords):
        setup(coords)
        spy["setups"].append((int(move._current_leaf), np.array(move._exposed_offset, copy=True)))

    move.compute_like_local = compute_like_local
    move._score_chunk = score_chunk
    move._check_like_local = check_like_local
    move.setup_likelihood_here = setup_likelihood_here
    return move


def _snapshot(move, acs, w0):
    return dict(
        w0=int(w0), fallbacks=int(move.n_batch_fallbacks), spy=move._spy,
        residual=[np.array(asnumpy(ac.data.arr), copy=True) for ac in acs.acs.flatten()],
        tables=dict(getattr(move, "_eigen_tables", {}) or {}),
    )


class _Case(unittest.TestCase):
    N_PROPOSES = 2

    def setUp(self):
        H._wdm()
        for w in range(4):
            H._residual(w)
        env = {k: v for k, v in os.environ.items() if k not in _CLEARED_ENV}
        env.update(EMRI_CHECK_LL_EVERY="1", EMRI_CHECK_LL="1")
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        import logging

        self.cap = H._WarnCapture()
        lat = logging.getLogger("lisatools")
        lat.addHandler(self.cap)
        self.addCleanup(lat.removeHandler, self.cap)
        if lat.getEffectiveLevel() > logging.WARNING:
            old = lat.level
            lat.setLevel(logging.WARNING)
            self.addCleanup(lat.setLevel, old)

    def _single(self, nwalkers, *, rng, **kw):
        walkers = list(range(nwalkers))
        acs = _aca(walkers)
        move = _make_move(acs, nwalkers, **kw)
        model = GlobalFitInfo(acs, map, rng())
        if isinstance(model.random, np.random.RandomState):
            np.random.seed(H.SEED)
        states, accepted = H._run_proposes(move, model, _state(walkers), self.N_PROPOSES)
        return dict(states=states, accepted=accepted, snap=_snapshot(move, acs, 0))

    def _world(self, n_compute, nwalkers, *, rng, gpu_routing=False, **kw):
        from lisatools.globalfit.moves import EMRIDirectLikeMove
        from lisatools.globalfit.recipe import _body_nwalkers

        world = FakeWorld(n_compute + 1, timeout=600.0)

        def fn(rank, comm):
            layout = build_layout(comm, nwalkers, [], legacy=False,
                                  gpu_routing=gpu_routing)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return dict(role="saver", layout=layout)
            w0, w1 = layout.block_of(rank)
            acs = _aca(range(w0, w1))
            width = _body_nwalkers(SimpleNamespace(rank_layout=layout), acs, EMRIDirectLikeMove)
            move = _make_move(acs, width, **kw)
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move.install_walker_fanout(SimpleNamespace(fanout=fo, rank=rank))
            model = GlobalFitInfo(acs, map, rng())
            fcomm.barrier()
            if layout.role_of(rank) == RankRole.HEAD:
                fo.enter_stage(H.STAGE, "pe")
                try:
                    if isinstance(model.random, np.random.RandomState):
                        np.random.seed(H.SEED)
                    states, accepted = H._run_proposes(move, model, _state(range(nwalkers)), self.N_PROPOSES)
                finally:
                    fo.stop()
                return dict(role="head", layout=layout, states=states, accepted=accepted,
                            snap=_snapshot(move, acs, w0))
            served = ComputeService(
                fcomm, layout, rank, registry={(H.STAGE, MOVE_NAME): move}, model=model,
                builtins={LIKELIHOOD_OP: lambda payload, clock, m: np.asarray(
                    asnumpy(m.analysis_container_arr.likelihood(complex=False)))},
            ).serve()
            return dict(role="compute", layout=layout, served=served, snap=_snapshot(move, acs, w0))

        return world.run(fn)

    def _assert_match(self, out, ref):
        head = out[0]
        for k, (s, r) in enumerate(zip(head["states"], ref["states"])):
            with self.subTest(propose=k):
                ss, rs = s.sub_states["emri"], r.sub_states["emri"]
                np.testing.assert_array_equal(ss.in_model_accepted, rs.in_model_accepted)
                np.testing.assert_array_equal(ss.swaps_accepted, rs.swaps_accepted)
                np.testing.assert_array_equal(head["accepted"][k], ref["accepted"][k])
                np.testing.assert_allclose(ss.coords, rs.coords, rtol=1e-12, atol=0)
                np.testing.assert_allclose(s.log_like, r.log_like, rtol=1e-9, atol=0)
                np.testing.assert_allclose(ss.log_like, rs.log_like, rtol=1e-9, atol=0)
                self.assertTrue(np.all(np.isfinite(rs.d_h)) and np.all(np.isfinite(rs.h_h)))
                np.testing.assert_allclose(ss.d_h, rs.d_h, rtol=1e-9, atol=0)
        last = ref["states"][-1].sub_states["emri"]
        self.assertGreater(int(last.in_model_accepted.sum()), 0)
        self.assertLess(int(last.in_model_accepted.sum()), int(last.in_model_proposed.sum()))
        ref_res = ref["snap"]["residual"]
        computes = {r: v for r, v in out.items() if v["role"] != "saver"}
        for rank, res in computes.items():
            snap = res["snap"]
            with self.subTest(rank=rank):
                self.assertEqual(snap["fallbacks"], 0)
                self.assertEqual(snap["spy"]["unarmed"], [])
                self.assertGreater(snap["spy"]["score_rows"], 0)
                for j, arr in enumerate(snap["residual"]):
                    want = ref_res[snap["w0"] + j]
                    np.testing.assert_allclose(arr, want, rtol=0, atol=1e-12 * np.abs(want).max())
        self.assertEqual(sum(v["snap"]["spy"]["score_rows"] for v in computes.values()),
                         ref["snap"]["spy"]["score_rows"])
        bad = [line for line in self.cap.lines if any(b in line for b in H._BAD_WARNINGS)]
        self.assertEqual(bad, [], "\n".join(self.cap.lines))
        return computes


class ReplicaRowFanoutTest(_Case):
    def test_two_replicas(self):
        kw = dict(scope="walker_max", permute_every=1, adaptive=True,
                  rng=lambda: np.random.RandomState(H.SEED))
        ref = self._single(1, **kw)
        out = self._world(2, 1, **kw)
        self.assertTrue(out[0]["layout"].replica_mode)
        computes = self._assert_match(out, ref)
        # rows really left the head, and every replica armed the head's offsets in order
        self.assertGreater(out[1]["snap"]["spy"]["score_rows"], 0)
        ref_setups = ref["snap"]["spy"]["setups"]
        self.assertEqual(len(ref_setups), self.N_PROPOSES * H.NLEAVES)
        for rank, res in computes.items():
            setups = res["snap"]["spy"]["setups"]
            self.assertEqual([s[0] for s in setups], [s[0] for s in ref_setups])
            for (_, off), (_, off_r) in zip(setups, ref_setups):
                np.testing.assert_allclose(off, off_r, rtol=1e-14, atol=0)
        # the cross-check ran every visit, fanned out too
        self.assertEqual(sum(v["snap"]["spy"]["check_rows"] for v in computes.values()),
                         self.N_PROPOSES * H.NLEAVES * H.NTEMPS)


class RoutedRowFanoutTest(_Case):
    """``NGPUS=8 GF_GPU_ROUTING=1`` on the 4-walker store: 4 blocks x R = 2.

    The head runs the 4-walker EMRI direct body; every scoring row is routed
    to the two ranks holding its walker (block-local row 0), every cold-chain
    replay is cut to the receiving rank's block, and the merged state and
    every rank's residual equal the single-process run's."""

    def test_four_blocks_two_replicas_each(self):
        kw = dict(scope="walker_max", permute_every=1, adaptive=True,
                  rng=lambda: np.random.RandomState(H.SEED))
        ref = self._single(4, **kw)
        out = self._world(8, 4, gpu_routing=True, **kw)
        layout = out[0]["layout"]
        self.assertEqual((layout.n_compute, layout.n_blocks,
                          layout.ranks_per_block, layout.block), (8, 4, 2, 1))
        computes = self._assert_match(out, ref)
        self.assertEqual(sorted(computes), list(range(8)))
        for rank, res in computes.items():
            snap = res["snap"]
            with self.subTest(rank=rank):
                local = set(np.concatenate([i for _, i in snap["spy"]["score_calls"]]).tolist())
                self.assertEqual(local, {0})
                self.assertEqual(snap["w0"], rank // 2)


class WalkerBlockFanoutTest(_Case):
    def test_four_ranks_one_walker_each(self):
        kw = dict(scope="per_walker", adaptive=False, refresh=10, rng=H._RowLocalRandom,
                  permute_every=0)
        with H._row_local_numpy():
            ref = self._single(4, **kw)
            out = self._world(4, 4, **kw)
        layout = out[0]["layout"]
        self.assertFalse(layout.replica_mode)
        self.assertEqual((layout.n_blocks, layout.block), (4, 1))
        computes = self._assert_match(out, ref)
        for rank, res in computes.items():
            snap = res["snap"]
            w0, w1 = layout.block_of(rank)
            with self.subTest(rank=rank):
                local = set(np.concatenate([i for _, i in snap["spy"]["score_calls"]]).tolist())
                self.assertEqual(local, {0})
                self.assertEqual(snap["spy"]["check_rows"], self.N_PROPOSES * H.NLEAVES * H.NTEMPS)
                for (leaf, off), (leaf_r, off_r) in zip(snap["spy"]["setups"], ref["snap"]["spy"]["setups"]):
                    self.assertEqual(leaf, leaf_r)
                    np.testing.assert_allclose(off, off_r[w0:w1], rtol=1e-14, atol=0)


if __name__ == "__main__":
    unittest.main()
