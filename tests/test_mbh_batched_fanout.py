"""MBHBatchedLikeMove under the add/remove move's fan-out: REAL proposes on FakeComm ranks.

The batched MBH move (the default ``mbh_pe`` since 2026-09-30) keeps ALL of
:class:`ResidualAddOneRemoveOneMove`'s choreography and adds per-replica
state the fan-out has to carry: the exposed-residual offset
(``acs.likelihood()`` at setup), the per-leaf window (set on expose/setup
from the cold-chain coords), the run-constant caches and the fallback
counter. These tests run two full proposes (expose -> setup -> eigen-table
refresh -> in-model repeats -> fast-vs-slow cross-check -> inner-product
record -> fold) of the REAL move with the CPU toy generator + windowed
adapter of ``tests/test_mbh_batched_move.py`` (tiny WDM grid, two leaves,
per-walker DIFFERENT noise and PSD), under :class:`FakeWorld` ranks, and
compare every observable with a single-process run of the same move:

* ONE-WALKER REPLICA mode (``nwalkers=1`` on 2 and 4 compute ranks + a
  saver): the head runs the single-process body, ``compute_like`` /
  ``compute_check_like`` scatter rows over the replicas (``ll_rows`` /
  ``ll_rows_check``) and the cold-chain expose/setup/fold replays on every
  replica (``ar_replay``). Production eigen scope (``walker_max``) and real
  seeded RNGs: the head consumes exactly the single-process stream.
* WALKER-BLOCK mode (``nwalkers=4`` on 2 ranks = 2 walkers/rank, and on 4
  ranks = 1 walker/rank, the 6-month v9 launcher's layout): every rank runs
  the body on its block against its own ACA and the head merges. Ranks hold
  independent RNG streams by design (and pool the ladder step), so the only
  meaningful single-process equality is under ROW-LOCAL draws: a
  call-counter RNG that hands every row of a call the same variate,
  identity leaf order / swap permutations, a fixed ladder, ``per_walker``
  eigen tables (each row's table is built at its own point against its own
  walker) and the walker-permuting swap off.

Each assertion is load-bearing; ``MUTATIONS`` in the report lists the
source edits each one was checked against.
"""

from __future__ import annotations

import logging
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.domains import WDMSignal
from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import (
    LIKELIHOOD_OP,
    ComputeService,
    WalkerFanout,
)
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.engine import GlobalFitInfo
from lisatools.globalfit.state import GFState, MBHState
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.utility import asnumpy

try:
    from tests.test_mbh_batched_move import (
        BASE_ROW, LAYER, NT, WINDOW_WIDE, _FastGen, _slow_gen, _toy_wdm,
    )
except ImportError:  # pragma: no cover - run from tests/
    from test_mbh_batched_move import (
        BASE_ROW, LAYER, NT, WINDOW_WIDE, _FastGen, _slow_gen, _toy_wdm,
    )

NTEMPS = 2                      # MBH_NTEMPS of the 6-month v9 launcher
NLEAVES = 2
NDIM = 11
NUM_REPEATS = 2                 # MBH_NUM_PROP_REPEATS of the launcher
BATCH = 3                       # small enough that the eigen sweep chunks
BETAS = np.array([1.0, 0.5])
SEED = 20261001
#: leaf merger times in WDM layers: two separate windows on the 128-layer grid
LEAF_LAYERS = (40.0, 88.0)
#: in-layer offsets of every walker's cold merger. All inside [0.40, 0.55]
#: of ONE layer, so the window of ANY subset of walkers (a rank's block)
#: lands on the same layers as the ensemble window -- see
#: WalkerBlockWindowTest for what happens when they do not.
COLD_FRAC = np.array([0.45, 0.48, 0.51, 0.54])
PRIOR_BOUNDS = [
    (5e5, 2e6), (2e5, 1e6), (-1.0, 1.0), (-1.0, 1.0), (1e3, 4e3),
    (0.0, 2 * np.pi), (0.0, np.pi), (0.0, np.pi), (0.0, 2 * np.pi),
    (-np.pi / 2, np.pi / 2), (30 * LAYER, 100 * LAYER),
]
MOVE_NAME = "mbh_pe"
STAGE = "full_pe"
#: env the move reads: cleared so a developer shell cannot change the test
_CLEARED_ENV = (
    "MBH_PERMUTE_EVERY", "ADDREMOVE_PERMUTE_EVERY", "MBH_EIGEN_REFRESH",
    "MBH_EIGEN_EPS_REL", "MBH_EIGEN_SCOPE", "MBH_LIKELIHOOD_FANOUT",
    "MBH_BATCHED_FILL", "MBH_RECORD_DH", "MBH_CHECK_LL_TOL", "MBH_SWAP_DEBUG",
    "MBH_DEBUG", "SOBBH_SINGLE_CALL", "ADDREMOVE_CHECK_LL", "ADDREMOVE_ROUNDTRIP",
    "GF_SUBSTATE_CHECK", "EIGEN_TABLES_PERSIST", "GF_GPU_ROUTING",
)
#: warnings that mean the fan-out (or the move) is broken
_BAD_WARNINGS = (
    "disagree", "EXPOSE INVARIANT", "WITHIN-WALKER", "fallback", "failed",
    "batch refused", "Very low log likelihood", "Suspicious likelihood",
)


# ---------------------------------------------------------------------------
# toy problem: deterministic per walker, so every rank rebuilds its walkers
# ---------------------------------------------------------------------------

class _Identity:
    """Sampling basis == waveform basis (the toy needs no transform)."""

    n_leaf_fills = None

    def both_transforms(self, x, **kwargs):
        return np.array(x, dtype=np.float64, copy=True)


def _truth(leaf):
    row = np.array(BASE_ROW, dtype=float)
    row[5] = 0.3 + 0.7 * leaf
    row[10] = (LEAF_LAYERS[leaf] + 0.47) * LAYER
    return row


def _cold(w, leaf):
    row = _truth(leaf)
    row[4] *= 1.0 + 0.03 * (w - 1.5)
    row[5] += 0.05 * w
    row[10] = (LEAF_LAYERS[leaf] + COLD_FRAC[w]) * LAYER
    return row


def _hot(w, leaf):
    row = _cold(w, leaf)
    row[4] *= 1.06
    row[5] += 0.2
    return row


def _tempered_coords(walkers):
    out = np.empty((NTEMPS, len(walkers), NLEAVES, NDIM))
    for j, w in enumerate(walkers):
        for leaf in range(NLEAVES):
            out[0, j, leaf] = _cold(w, leaf)
            out[1, j, leaf] = _hot(w, leaf)
    return out


_WDM = {}


def _wdm():
    """ONE settings object per process (read-only), warmed before threads."""
    if "wdm" not in _WDM:
        _WDM["wdm"] = _toy_wdm(0.0)
    _slow_gen.wdm = _WDM["wdm"]
    _slow_gen.t0_stock = 0.0
    _slow_gen.data_t0 = 0.0
    return _WDM["wdm"]


_DATA = {}


def _residual(w):
    """Walker ``w``'s residual: its own noise + every leaf's truth minus its
    cold-chain source (the engine rebuild, with the stock generator)."""
    if w not in _DATA:
        wdm = _wdm()
        rng = np.random.default_rng(1000 + w)
        arr = 1e-23 * rng.normal(size=(3, int(wdm.Nf_active), NT))
        for leaf in range(NLEAVES):
            arr = arr + np.asarray(_slow_gen(*_truth(leaf)).arr)
            arr = arr - np.asarray(_slow_gen(*_cold(w, leaf)).arr)
        _DATA[w] = arr
    return np.array(_DATA[w], copy=True)


def _aca(walkers):
    wdm = _wdm()
    acs_list = []
    for w in walkers:
        model = ("scirdv1", "mrdv1")[w % 2]        # per-walker DIFFERENT PSD
        ac = AnalysisContainer(WDMSignal(_residual(w), wdm), XYZ2SensitivityMatrix(wdm, model=model))
        ac.signal_gen = {"mbh": _slow_gen}
        acs_list.append(ac)
    return AnalysisContainerArray(acs_list)


def _state(walkers):
    """Engine state (cold chain only) + the MBH sub-state's tempered ladder."""
    tempered = _tempered_coords(walkers)
    nw = len(walkers)
    state = GFState(
        {"mbh": tempered[:1].copy()},
        inds={"mbh": np.ones((1, nw, NLEAVES), dtype=bool)},
        log_like=np.zeros((1, nw)),
        log_prior=np.zeros((1, nw)),
        betas=np.ones(1),
        sub_state_bases={"mbh": MBHState},
    )
    sub = state.sub_states["mbh"]
    sub.betas_all = np.tile(BETAS, (NLEAVES, 1))
    sub.initialize_tempered(NTEMPS, nw, NLEAVES, NDIM, coords=tempered)
    return state


# ---------------------------------------------------------------------------
# RNGs
# ---------------------------------------------------------------------------

class _RowLocalRandom:
    """``model.random`` stand-in whose every call hands EVERY ROW the same draw.

    The draw cycles with the CALL count only, and every rank's body makes
    the same calls in the same order (same leaves, repeats and inner move),
    so a row's proposal and accept draw do not depend on which rank -- or
    which block width -- scores it."""

    Z = (0.9, -0.6, 1.3, -1.1, 0.4, -0.25)
    U = (0.5, 0.2, 0.8, 0.35)

    def __init__(self):
        self.k = 0

    def _next(self, seq):
        self.k += 1
        return seq[self.k % len(seq)]

    def choice(self, a, p=None):
        return np.asarray(a)[0]

    def rand(self, *shape):
        return np.full(shape, self._next(self.U))

    def randn(self, *shape):
        return np.full(shape, self._next(self.Z))

    def randint(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low
        return np.full(size, low + self.k % (high - low), dtype=int)


def _row_local_numpy():
    """Identity leaf order / swap pairing and a fixed swap draw (np.random is
    shared by every FakeWorld thread, so it must not be a stream here)."""

    def permutation(x):
        return np.arange(x) if np.ndim(x) == 0 else np.array(x, copy=True)

    def uniform(low=0.0, high=1.0, size=None):
        return np.full(size, 0.5)

    return mock.patch.multiple(np.random, permutation=permutation, uniform=uniform)


# ---------------------------------------------------------------------------
# the move + per-rank spies
# ---------------------------------------------------------------------------

def _make_move(acs, nwalkers, *, scope, permute_every, adaptive, refresh=1):
    from eryn.moves import EigenAxisMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from lisatools.globalfit.moves import MBHBatchedLikeMove
    from lisatools.sources.batching import MBHWindowedWDMSignalGen

    wdm = acs.acs.flatten()[0].data.settings
    adapter = MBHWindowedWDMSignalGen(_FastGen(0.0), wdm, nchannels=3, tukey_alpha=0.0)
    priors = {"mbh": ProbDistContainer({i: uniform_dist(lo, hi) for i, (lo, hi) in enumerate(PRIOR_BOUNDS)})}
    move = MBHBatchedLikeMove(
        "mbh", (NTEMPS, nwalkers, NLEAVES, NDIM), None, {}, {}, acs, NUM_REPEATS,
        _Identity(), priors, [(EigenAxisMove(mode="axis"), 1.0)],
        betas_all=np.tile(BETAS, (NLEAVES, 1)), permute_every=permute_every,
        batched_gen=adapter, batch_max_size=BATCH, name=MOVE_NAME,
        eigen_refresh_every=refresh, eigen_table_scope=scope, **WINDOW_WIDE,
    )
    move.gf_move_name = MOVE_NAME
    move.fanout_branches = ["mbh"]
    for tc in move.temperature_controls:
        tc.adaptive = bool(adaptive)
    _install_spies(move)
    return move


def _install_spies(move):
    """Instance-level wrappers: who scored which rows, with what armed state."""
    spy = move._spy = dict(score_rows=0, score_calls=[], check_rows=0, setups=[], unarmed=[])
    local, chunk, check, setup = (
        move.compute_like_local, move._score_chunk, move._check_like_local, move.setup_likelihood_here,
    )

    def compute_like_local(coords_in, data_index):
        leaf = int(move._current_leaf)
        if move._exposed_offset is None or leaf not in move._leaf_windows:
            spy["unarmed"].append(leaf)
        spy["score_rows"] += int(np.atleast_2d(np.asarray(coords_in)).shape[0])
        return local(coords_in, data_index)

    def score_chunk(adapter, coords, idx, leaf):
        spy["score_calls"].append((int(leaf), np.array(idx, copy=True)))
        return chunk(adapter, coords, idx, leaf)

    def check_like_local(coords_in, data_index):
        # the slow cross-check scorer: reached directly single-process and
        # through ``_check_rows_local`` / ``serve_ll_rows_check`` on replicas
        spy["check_rows"] += int(np.atleast_2d(np.asarray(coords_in)).shape[0])
        return check(coords_in, data_index)

    def setup_likelihood_here(coords):
        setup(coords)
        leaf = int(move._current_leaf)
        g = move._leaf_windows[leaf]
        spy["setups"].append((leaf, np.array(move._exposed_offset, copy=True),
                              (g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"])))

    move.compute_like_local = compute_like_local
    move._score_chunk = score_chunk
    move._check_like_local = check_like_local
    move.setup_likelihood_here = setup_likelihood_here


def _window_keys(move):
    return {
        leaf: (g["n_start"], g["Nt_keep"], g["n_pad_lo"], g["n_pad_hi"])
        for leaf, g in move._leaf_windows.items()
    }


def _snapshot(move, acs, w0):
    return dict(
        move=move,
        w0=int(w0),
        residual=[np.array(asnumpy(ac.data.arr), copy=True) for ac in acs.acs.flatten()],
        fallbacks=int(move.n_batch_fallbacks),
        windows=_window_keys(move),
        offset=None if move._exposed_offset is None else np.array(move._exposed_offset, copy=True),
        spy=move._spy,
        tables=dict(getattr(move, "_eigen_tables", {}) or {}),
    )


def _run_proposes(move, model, state, n):
    states, accepted = [], []
    for _ in range(n):
        state, acc = move.propose(model, state)
        states.append(state)
        accepted.append(np.asarray(acc))
    return states, accepted


class _WarnCapture(logging.Handler):
    """Every WARNING+ record under ``lisatools`` (all FakeWorld threads)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.lines = []
        self._lock_ = threading.Lock()

    def emit(self, record):
        with self._lock_:
            self.lines.append(f"{record.name}: {record.getMessage()}")


class _FanoutCase(unittest.TestCase):
    N_PROPOSES = 2

    def setUp(self):
        _wdm()
        # warm the shared settings and every walker's data before any thread
        for w in range(4):
            _residual(w)
        env = {k: v for k, v in os.environ.items() if k not in _CLEARED_ENV}
        env.update(MBH_CHECK_LL_EVERY="1", MBH_CHECK_LL="1")
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cap = _WarnCapture()
        lat = logging.getLogger("lisatools")
        lat.addHandler(self.cap)
        self.addCleanup(lat.removeHandler, self.cap)
        self._old_level = lat.level
        if lat.getEffectiveLevel() > logging.WARNING:
            lat.setLevel(logging.WARNING)
            self.addCleanup(lat.setLevel, self._old_level)

    # -- runs ---------------------------------------------------------------
    def _single(self, nwalkers, *, rng, **move_kw):
        walkers = list(range(nwalkers))
        acs = _aca(walkers)
        move = _make_move(acs, nwalkers, **move_kw)
        model = GlobalFitInfo(acs, map, rng())
        if isinstance(model.random, np.random.RandomState):
            np.random.seed(SEED)
        states, accepted = _run_proposes(move, model, _state(walkers), self.N_PROPOSES)
        return dict(states=states, accepted=accepted, snap=_snapshot(move, acs, 0))

    def _world(self, n_compute, nwalkers, *, rng, gpu_routing=False, **move_kw):
        from lisatools.globalfit.moves import MBHBatchedLikeMove
        from lisatools.globalfit.recipe import _body_nwalkers

        world = FakeWorld(n_compute + 1, timeout=600.0)   # + the dedicated saver

        def fn(rank, comm):
            layout = build_layout(comm, nwalkers, [], legacy=False, gpu_routing=gpu_routing)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return dict(role="saver", layout=layout)
            w0, w1 = layout.block_of(rank)
            acs = _aca(range(w0, w1))
            # the move's width by the SAME rule SingleSourcePEBuilder applies
            width = _body_nwalkers(SimpleNamespace(rank_layout=layout), acs, MBHBatchedLikeMove)
            move = _make_move(acs, width, **move_kw)
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move.install_walker_fanout(SimpleNamespace(fanout=fo, rank=rank))
            model = GlobalFitInfo(acs, map, rng())
            fcomm.barrier()        # every rank built before the head draws
            if layout.role_of(rank) == RankRole.HEAD:
                fo.enter_stage(STAGE, "pe")
                try:
                    if isinstance(model.random, np.random.RandomState):
                        np.random.seed(SEED)
                    states, accepted = _run_proposes(move, model, _state(range(nwalkers)), self.N_PROPOSES)
                finally:
                    fo.stop()
                return dict(role="head", layout=layout, states=states, accepted=accepted,
                            snap=_snapshot(move, acs, w0))
            served = ComputeService(
                fcomm, layout, rank, registry={(STAGE, MOVE_NAME): move}, model=model,
                builtins={LIKELIHOOD_OP: lambda payload, clock, m: np.asarray(
                    asnumpy(m.analysis_container_arr.likelihood(complex=False)))},
            ).serve()
            return dict(role="compute", layout=layout, served=served, snap=_snapshot(move, acs, w0))

        return world.run(fn)

    # -- shared assertions ---------------------------------------------------
    def _assert_no_bad_warnings(self):
        bad = [line for line in self.cap.lines if any(b in line for b in _BAD_WARNINGS)]
        self.assertEqual(bad, [], "\n".join(self.cap.lines))

    def _assert_states_match(self, got, ref):
        """Every propose's merged state equals the single-process one."""
        self.assertEqual(len(got["states"]), len(ref["states"]))
        for k, (s, r) in enumerate(zip(got["states"], ref["states"])):
            with self.subTest(propose=k):
                ss, rs = s.sub_states["mbh"], r.sub_states["mbh"]
                # identical accept/reject decisions, identical swap decisions
                np.testing.assert_array_equal(ss.in_model_accepted, rs.in_model_accepted)
                np.testing.assert_array_equal(ss.in_model_proposed, rs.in_model_proposed)
                np.testing.assert_array_equal(ss.swaps_accepted, rs.swaps_accepted)
                np.testing.assert_array_equal(got["accepted"][k], ref["accepted"][k])
                # identical accepted coords (every rung, both leaves) + main cold row
                np.testing.assert_allclose(ss.coords, rs.coords, rtol=1e-12, atol=0)
                np.testing.assert_allclose(
                    s.branches["mbh"].coords, r.branches["mbh"].coords, rtol=1e-12, atol=0)
                # per-walker lnL: engine cold lnL and the per-leaf ladder lnL
                np.testing.assert_allclose(s.log_like, r.log_like, rtol=1e-9, atol=0)
                np.testing.assert_allclose(ss.log_like, rs.log_like, rtol=1e-9, atol=0)
                # the inner-product record (cold rung, per walker x leaf)
                self.assertTrue(np.all(np.isfinite(rs.d_h)) and np.all(np.isfinite(rs.h_h)))
                np.testing.assert_allclose(ss.d_h, rs.d_h, rtol=1e-9, atol=0)
                np.testing.assert_allclose(ss.h_h, rs.h_h, rtol=1e-9, atol=0)
        last = ref["states"][-1].sub_states["mbh"]
        # the comparison is not vacuous: moves were both accepted and rejected
        self.assertGreater(int(last.in_model_accepted.sum()), 0)
        self.assertLess(int(last.in_model_accepted.sum()), int(last.in_model_proposed.sum()))

    def _assert_residuals_match(self, out, ref):
        """Every rank's residual rows equal the single-process rows after fold."""
        ref_res = ref["snap"]["residual"]
        for rank, res in out.items():
            if res["role"] == "saver":
                continue
            snap = res["snap"]
            for j, arr in enumerate(snap["residual"]):
                want = ref_res[snap["w0"] + j]
                np.testing.assert_allclose(arr, want, rtol=0, atol=1e-12 * np.abs(want).max(),
                                           err_msg=f"rank {rank} walker {snap['w0'] + j}")

    def _assert_rank_health(self, out, ref):
        """No fallbacks anywhere; every rank armed before scoring; windows match."""
        for rank, res in out.items():
            if res["role"] == "saver":
                continue
            snap = res["snap"]
            with self.subTest(rank=rank):
                self.assertEqual(snap["fallbacks"], 0)
                self.assertEqual(snap["spy"]["unarmed"], [])
                self.assertEqual(snap["windows"], ref["snap"]["windows"])
                # the batched scorer RAN here (not some fallback / other rank)
                self.assertGreater(snap["spy"]["score_rows"], 0)
                self.assertGreater(len(snap["spy"]["score_calls"]), 0)


# ---------------------------------------------------------------------------
# (a) FLAT body on the head: one-walker replicas, and replicas over blocks
# ---------------------------------------------------------------------------

class _FlatCase(_FanoutCase):
    """The head runs the single-process body; rows and replays fan out.

    Real seeded RNGs and the production ``walker_max`` eigen scope: only
    the head draws, so it consumes exactly the single-process stream."""

    KW = dict(scope="walker_max", permute_every=1, adaptive=True,
              rng=lambda: np.random.RandomState(SEED))

    def _check(self, n_compute, nwalkers, gpu_routing=False):
        ref = self._single(nwalkers, **self.KW)
        out = self._world(n_compute, nwalkers, gpu_routing=gpu_routing, **self.KW)
        head = out[0]
        layout = head["layout"]
        self.assertTrue(layout.replica_mode)

        self._assert_states_match(head, ref)
        self._assert_residuals_match(out, ref)
        self._assert_rank_health(out, ref)
        self._assert_no_bad_warnings()

        # the fan-out neither drops nor duplicates rows: summed over the
        # ranks, exactly the single-process row counts were scored and
        # cross-checked
        computes = {r: v for r, v in out.items() if v["role"] != "saver"}
        self.assertEqual(sum(v["snap"]["spy"]["score_rows"] for v in computes.values()),
                         ref["snap"]["spy"]["score_rows"])
        self.assertEqual(sum(v["snap"]["spy"]["check_rows"] for v in computes.values()),
                         ref["snap"]["spy"]["check_rows"])
        # the cross-check really ran on every visit (MBH_CHECK_LL_EVERY=1)
        # and its rows really left the head (``serve_ll_rows_check``)
        self.assertEqual(ref["snap"]["spy"]["check_rows"],
                         self.N_PROPOSES * NLEAVES * NTEMPS * nwalkers)
        self.assertGreater(out[1]["snap"]["spy"]["check_rows"], 0)

        # every rank armed, at every leaf visit and in the head's order, the
        # window and the exposed-residual offset of ITS walkers -- the
        # single-process values (the expose/setup replays)
        ref_setups = ref["snap"]["spy"]["setups"]
        self.assertEqual(len(ref_setups), self.N_PROPOSES * NLEAVES)
        for rank, res in computes.items():
            w0, w1 = layout.block_of(rank)
            setups = res["snap"]["spy"]["setups"]
            with self.subTest(rank=rank):
                self.assertEqual([(s[0], s[2]) for s in setups], [(s[0], s[2]) for s in ref_setups])
                for (_, off, _), (_, off_r, _) in zip(setups, ref_setups):
                    np.testing.assert_allclose(off, off_r[w0:w1], rtol=1e-14, atol=0)

        # walker_max eigen tables: built on the head from rows scored on the
        # ranks holding the max-lnL walker, equal to the single-process tables
        self.assertEqual(set(head["snap"]["tables"]), set(range(NLEAVES)))
        for leaf, (axes, sigmas) in ref["snap"]["tables"].items():
            got_axes, got_sigmas = head["snap"]["tables"][leaf]
            np.testing.assert_allclose(got_sigmas, sigmas, rtol=1e-7, atol=0)
            np.testing.assert_allclose(np.abs(got_axes), np.abs(axes), rtol=0, atol=1e-7)
        return out, ref


class ReplicaRowFanoutTest(_FlatCase):
    """nwalkers=1 on n compute ranks: rows split contiguously, replays reach all."""

    def test_two_replicas(self):
        out, _ref = self._check(2, 1)
        self.assertEqual((out[0]["layout"].n_blocks, out[0]["layout"].ranks_per_block), (1, 2))
        # the eigen sweep put rows on BOTH ranks
        self.assertGreater(out[1]["snap"]["spy"]["score_rows"], 100)
        self.assertGreater(out[1]["served"], 0)

    def test_four_replicas(self):
        out, _ref = self._check(4, 1)
        self.assertEqual((out[0]["layout"].n_blocks, out[0]["layout"].ranks_per_block), (1, 4))


class RoutedRowFanoutTest(_FlatCase):
    """``GF_GPU_ROUTING`` both axes: 2 walkers on 4 ranks = 2 blocks x R=2.

    The head runs the 2-walker body; each row is routed to the two ranks
    holding its walker with ``data_index`` rewritten to the block-local row;
    every replay is cut to the receiving rank's block. Before 2026-10-01 the
    first expose crashed here (all N rows applied to a 1-row ACA) and the
    move was built 1 walker wide against the 2-walker state."""

    def test_two_blocks_two_replicas_each(self):
        out, _ref = self._check(4, 2, gpu_routing=True)
        layout = out[0]["layout"]
        self.assertEqual((layout.n_blocks, layout.ranks_per_block, layout.block), (2, 2, 1))
        for rank, res in out.items():
            if res["role"] == "saver":
                continue
            snap = res["snap"]
            # a 1-row ACA: every row a rank scored came in rewritten to row 0
            idx = np.concatenate([i for _, i in snap["spy"]["score_calls"]])
            self.assertEqual(set(idx.tolist()), {0}, rank)
            self.assertEqual(len(snap["residual"]), 1)
            # the move runs the 2-walker body on every rank's build
            self.assertEqual(snap["move"].nwalkers, 2)

    def test_four_blocks_two_replicas_each(self):
        """``NGPUS=8 GF_GPU_ROUTING=1`` on the 4-walker store: 4 blocks x R = 2.

        The 9mo production shape on 8 GPUs. The head runs the 4-walker body;
        every row is routed to the two ranks holding its walker (block-local
        row 0 on each), so BOTH replicas of EVERY block score rows -- the
        add/remove family's within-walker split -- and the merged state, every
        rank's residual and the eigen tables equal the single-process run's
        (``_check``)."""
        out, _ref = self._check(8, 4, gpu_routing=True)
        layout = out[0]["layout"]
        self.assertEqual((layout.n_compute, layout.n_blocks,
                          layout.ranks_per_block, layout.block), (8, 4, 2, 1))
        for b in range(4):
            for rank in layout.ranks_in_block(b):
                snap = out[rank]["snap"]
                self.assertEqual(snap["w0"], b)
                self.assertEqual(len(snap["residual"]), 1)
                idx = np.concatenate([i for _, i in snap["spy"]["score_calls"]])
                self.assertEqual(set(idx.tolist()), {0}, rank)
                self.assertGreater(snap["spy"]["score_rows"], 0, rank)
                self.assertEqual(snap["move"].nwalkers, 4)


# ---------------------------------------------------------------------------
# (b) walker-block fan-out: every rank runs the body on its own block
# ---------------------------------------------------------------------------

class WalkerBlockFanoutTest(_FanoutCase):
    """nwalkers=4 on 2 ranks (2/rank) and on 4 ranks (1/rank, the v9 layout)."""

    NWALKERS = 4
    KW = dict(scope="per_walker", adaptive=False, refresh=10, rng=_RowLocalRandom)

    def _check(self, n_compute, permute_every):
        kw = dict(self.KW, permute_every=permute_every)
        with _row_local_numpy():
            ref = self._single(self.NWALKERS, **kw)
            out = self._world(n_compute, self.NWALKERS, **kw)
        head = out[0]
        layout = head["layout"]
        self.assertFalse(layout.replica_mode)
        self.assertEqual((layout.n_blocks, layout.block), (n_compute, self.NWALKERS // n_compute))

        self._assert_states_match(head, ref)
        self._assert_residuals_match(out, ref)
        self._assert_rank_health(out, ref)
        self._assert_no_bad_warnings()

        computes = {r: v for r, v in out.items() if v["role"] != "saver"}
        for rank, res in computes.items():
            snap = res["snap"]
            w0, w1 = layout.block_of(rank)
            with self.subTest(rank=rank):
                # this rank scored ONLY its own walkers (local rows), and all of them
                local = set(np.concatenate([idx for _, idx in snap["spy"]["score_calls"]]).tolist())
                self.assertEqual(local, set(range(w1 - w0)))
                # it cross-checked its own rows: (proposes x leaves x ntemps x block)
                self.assertEqual(snap["spy"]["check_rows"], self.N_PROPOSES * NLEAVES * NTEMPS * (w1 - w0))
                # its per-(temp, walker) eigen tables are the single-process
                # tables of ITS walkers
                for leaf, (axes, sigmas) in ref["snap"]["tables"].items():
                    got_axes, got_sigmas = snap["tables"][leaf]
                    np.testing.assert_allclose(got_sigmas, sigmas[:, w0:w1], rtol=1e-7, atol=0)
                    np.testing.assert_allclose(np.abs(got_axes), np.abs(axes[:, w0:w1]), rtol=0, atol=1e-7)
                # its exposed offsets are the single-process offsets of its walkers
                for (leaf, off, key), (leaf_r, off_r, key_r) in zip(
                        snap["spy"]["setups"], ref["snap"]["spy"]["setups"]):
                    self.assertEqual((leaf, key), (leaf_r, key_r))
                    np.testing.assert_allclose(off, off_r[w0:w1], rtol=1e-14, atol=0)
        self.assertEqual(sum(v["snap"]["spy"]["score_rows"] for v in computes.values()),
                         ref["snap"]["spy"]["score_rows"])
        return out, ref

    def test_two_ranks_two_walkers_each(self):
        # walker-permuting swap ON: the fancy-swap re-scoring
        # (log_like_for_fancy_swaping -> compute_like) runs on every rank
        self._check(2, permute_every=1)

    def test_four_ranks_one_walker_each(self):
        # the v9 launcher's layout; one walker per rank never fancy-swaps
        self._check(4, permute_every=0)


class WalkerBlockWindowTest(unittest.TestCase):
    """DOCUMENTS (does not endorse) the walker-block window: a rank places a
    leaf's window at ITS block's median cold-chain merger, not the
    ensemble's. With mergers straddling a layer boundary a block's window
    sits one layer off the single-process window; every rank stays
    self-consistent (expose, setup and fold of a visit share its window), so
    each walker is scored on a valid windowed likelihood, just not on the
    single-process run's window. The fan-out tests above keep every merger
    inside one layer so their single-process comparisons are like for like."""

    def test_block_window_follows_the_block_median(self):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        _wdm()
        # mergers straddle the start of layer 41: block medians 40.825 and
        # 41.275 layers, ensemble median 41.05 (none near a boundary)
        frac = np.array([0.80, 0.85, 1.25, 1.30])
        cold = np.stack([_cold(w, 0) for w in range(4)])
        cold[:, 10] = (LEAF_LAYERS[0] + frac) * LAYER
        n_start = {}
        for name, (w0, w1) in {"ensemble": (0, 4), "block0": (0, 2), "block1": (2, 4)}.items():
            acs = _aca(range(w0, w1))
            move = _make_move(acs, w1 - w0, scope="walker_max", permute_every=0, adaptive=False)
            move._current_leaf = 0
            move.remove_cold_chain_sources(cold[w0:w1])
            n_start[name] = move._leaf_windows[0]["n_start"]
            want = mbh_window_layers(_wdm(), float(np.median(cold[w0:w1, 10])), **WINDOW_WIDE)
            self.assertEqual(n_start[name], want["n_start"], name)
        self.assertEqual(n_start["block0"], n_start["ensemble"] - 1)
        self.assertEqual(n_start["block1"], n_start["ensemble"])


class BuilderBodyWidthTest(unittest.TestCase):
    """``SingleSourcePEBuilder`` sizes the move for the state its BODY runs on.

    Only the ``GF_GPU_ROUTING`` both-axes layout differs from the local ACA
    width: the head's flat body runs the whole ensemble there."""

    @staticmethod
    def _layout(nwalkers, n_compute, gpu_routing):
        return FakeWorld(n_compute + 1).run(
            lambda r, c: build_layout(c, nwalkers, [], legacy=False, gpu_routing=gpu_routing)
        )[0]

    def _built(self, layout, n_local, flat=True):
        from lisatools.globalfit.recipe import MBHMoveBuilder

        seen = {}

        class _StubMove:
            fanout_flat_body = flat

            def __init__(self, branch, coords_shape, *args, **kwargs):
                seen["shape"] = coords_shape

            def install_walker_fanout(self, curr):
                seen["installed"] = True

        class _Builder(MBHMoveBuilder):
            move_class = _StubMove
            use_dcga = False

        info = SimpleNamespace(
            betas=BETAS.copy(), ntemps=NTEMPS, ndim=NDIM, nleaves_max=NLEAVES,
            waveform_kwargs={}, num_prop_repeats=1, transform=None, inner_moves=[],
            eigen_refresh_every=None, eigen_table_scope=None, info_matrix_gen=None,
        )
        curr = SimpleNamespace(
            source_info={"mbh": info}, rank_layout=layout,
            general_info=SimpleNamespace(ntemps=1, main_file_path=None),
        )
        state = SimpleNamespace(sub_states={"mbh": SimpleNamespace()})
        _, (move,) = _Builder(wave_gen=None).build(
            None, curr, SimpleNamespace(acs_total_entries=n_local), None, state)
        self.assertTrue(seen["installed"])
        self.assertEqual(move.accepted.shape, (NTEMPS, seen["shape"][1]))
        return seen["shape"]

    def test_routed_flat_body_is_ensemble_wide(self):
        layout = self._layout(2, 4, gpu_routing=True)          # 2 blocks x R=2
        self.assertEqual(self._built(layout, 1), (NTEMPS, 2, NLEAVES, NDIM))
        # a family that keeps the block path keeps its block width
        self.assertEqual(self._built(layout, 1, flat=False)[1], 1)

    def test_every_other_layout_keeps_the_local_width(self):
        self.assertEqual(self._built(None, 4)[1], 4)                               # single
        self.assertEqual(self._built(self._layout(4, 2, False), 2)[1], 2)          # walker blocks
        self.assertEqual(self._built(self._layout(4, 4, False), 1)[1], 1)          # v9: 1/rank
        self.assertEqual(self._built(self._layout(1, 2, False), 1)[1], 1)          # one walker


if __name__ == "__main__":
    unittest.main()
