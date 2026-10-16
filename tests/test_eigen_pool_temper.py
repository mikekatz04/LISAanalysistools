"""Pooled step-scale adaptation + per-rung tempered eigen steps (ruling 2026-10-15).

Why (9mo, EMRI leaf 3, SNR 50, plunge in the window): walker 0's hot rungs
(T >= 2.1) spanned the whole degenerate ridge (p0 12.3309..12.3372, M and a
sliding along it) while walkers 1 and 3 had ladders whose EVERY rung stayed
within 1e-4. Two reasons, both fixed here behind code-default-OFF knobs:

* every rung proposed with the same cold-sized sigma (the shared source
  table is untempered) -> ``{BRANCH}_EIGEN_TEMPER_SIGMAS``: rung ``k`` steps
  with ``sigma / sqrt(beta_k)`` (capped at the table's own cap);
* the per-rank adaptation can only learn 1e-4 from a stuck walker ->
  ``{BRANCH}_EIGEN_ADAPT_POOL``: under the walker-block fan-out the head
  ships the FULL ensemble's rung coordinates (all walkers x all rungs) in
  every propose payload, each rank buffers that union, and its adapted
  sigma is the robust width of the union on its OWN table axes.

Pinned here (all CPU):

* (i) pooling on a real FakeWorld fan-out (stub body): a stuck rank adapts
  to the union's width with the knob on, its own width with it off;
* (ii) tempering: rung k's proposal std = sigma_k / sqrt(beta_k), the cold
  rung unchanged, off = unchanged, the sidecar keeps the COLD sigma;
* (iii) lives in ``tests/test_eigen_axis_adapt.py::PooledRidgeTest`` (the
  synthetic 20x ridge, 4 walkers x 8 rungs, 3 walkers stuck at one end:
  traversal / IAT of the stuck walkers, pooling + tempering on vs off; the
  numbers are printed with ``EIGEN_POOL_HARNESS_VERBOSE=1``);
* (iv) the 4 walkers x 2 replicas routed layout: the addremove body runs
  FLAT on the head there, so the head's own buffer IS the union -- no
  payload is needed and none is shipped; the routed run equals the
  single-process run with both knobs on;
* (v) with the knobs off the single-process proposal stream (and the 4-rank
  block fan-out) is BIT-IDENTICAL to HEAD aaace7ab: ``tests/data/
  eigen_pool_temper_golden.npz`` was written by :func:`write_golden` run
  against a ``git archive aaace7ab`` export of src/ + tests/ (the editable
  finder removed from ``sys.meta_path``, the export first on ``sys.path``).
"""

from __future__ import annotations

import copy
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.moves import EigenAxisMove

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.moves import eigen_refresh, eigen_table_persist
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.state import EMRIState, GFState

try:
    from tests import test_eigen_axis_adapt as A
    from tests import test_emri_direct_fanout as E
except ImportError:  # pragma: no cover - run from tests/
    import test_eigen_axis_adapt as A
    import test_emri_direct_fanout as E

H = E.H
BR = "emri"
LOGGER = "lisatools.globalfit.moves.addremovemove"
_ALL_KNOBS = A._KNOBS + tuple(
    f"{b}_{k}" for b in ("EMRI", "MBH", "SOBBH")
    for k in ("EIGEN_ADAPT_POOL", "EIGEN_TEMPER_SIGMAS"))

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "eigen_pool_temper_golden.npz")
N_GOLDEN = 3
_KNOB_ENV = tuple(f"EMRI_{k}" for k in (
    "EIGEN_AXIS_ADAPT", "EIGEN_ADAPT_VISITS", "EIGEN_ADAPT_MIN_VISITS",
    "EIGEN_ADAPT_POOL", "EIGEN_TEMPER_SIGMAS", "EIGEN_ADAPT_MAX_FACTOR", "COLD_TO_HOT_EVERY",
    "LADDER_REANCHOR_NATS", "EIGEN_INFO", "EIGEN_JUMP_MIX"))

#: (single process?, nwalkers, env, move kwargs). HEAD-compatible on purpose:
#: write_golden runs this table against the pre-change code.
GOLDEN_SCENARIOS = {
    "single_off": (True, 4, {}, dict(scope="walker_max", permute_every=1, adaptive=True)),
    "single_adapt": (True, 4, {"EMRI_EIGEN_AXIS_ADAPT": "1", "EMRI_EIGEN_ADAPT_MIN_VISITS": "1"},
                     dict(scope="walker_max", permute_every=1, adaptive=True)),
    "single_per_walker": (True, 4, {}, dict(scope="per_walker", permute_every=0, adaptive=False)),
    "block4_off": (False, 4, {}, dict(scope="per_walker", permute_every=0, adaptive=False,
                                      refresh=10)),
    "block4_adapt": (False, 4, {"EMRI_EIGEN_AXIS_ADAPT": "1", "EMRI_EIGEN_ADAPT_MIN_VISITS": "1"},
                     dict(scope="walker_max", permute_every=0, adaptive=False)),
}


def _golden_env(extra):
    env = {k: v for k, v in os.environ.items()
           if k not in E._CLEARED_ENV and k not in _KNOB_ENV}
    env.update(EMRI_CHECK_LL_EVERY="1", EMRI_CHECK_LL="1", EIGEN_TABLES_PERSIST="0")
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


def golden_stream(name, n_proposes=N_GOLDEN):
    """The proposal stream + per-propose end states of one scenario, as arrays.

    Single process: the REAL ``EMRIDirectLikeMove`` (the toy WDM fixture of
    test_emri_direct_fanout) through ``move.propose`` -- the call
    ``fit.sample()`` makes per iteration -- with every inner-move proposal
    recorded. Block: the 4-rank walker-block fan-out (FakeWorld), the merged
    end states.
    """
    single, nw, env, kw = GOLDEN_SCENARIOS[name]
    out = {}
    with _golden_env(env):
        H._wdm()
        for w in range(4):
            H._residual(w)
        if single:
            from lisatools.globalfit.engine import GlobalFitInfo

            walkers = list(range(nw))
            acs = E._aca(walkers)
            move = E._make_move(acs, nw, **kw)
            qs = []
            inner = move.moves[0]
            orig = inner.get_proposal

            def rec(*a, **k):
                q, f = orig(*a, **k)
                qs.append(np.array(q["emri"], dtype=float, copy=True))
                return q, f

            inner.get_proposal = rec
            model = GlobalFitInfo(acs, map, np.random.RandomState(H.SEED))
            np.random.seed(H.SEED)
            states, accepted = H._run_proposes(move, model, E._state(walkers), n_proposes)
            out["q"] = np.stack(qs)
            for leaf, (axes, sig) in sorted((getattr(move, "_eigen_tables", None) or {}).items()):
                out[f"axes_{leaf}"] = np.asarray(axes, dtype=float)
                out[f"sigmas_{leaf}"] = np.asarray(sig, dtype=float)
        else:
            case = SimpleNamespace(N_PROPOSES=n_proposes)
            with H._row_local_numpy():
                res = E._Case._world(case, 4, nw, rng=H._RowLocalRandom, **kw)
            states, accepted = res[0]["states"], res[0]["accepted"]
        for k, s in enumerate(states):
            sub = s.sub_states["emri"]
            out[f"coords_{k}"] = np.asarray(sub.coords, dtype=float)
            out[f"log_like_{k}"] = np.asarray(sub.log_like, dtype=float)
            out[f"in_model_accepted_{k}"] = np.asarray(sub.in_model_accepted)
            out[f"swaps_accepted_{k}"] = np.asarray(sub.swaps_accepted)
            out[f"betas_all_{k}"] = np.asarray(sub.betas_all, dtype=float)
            out[f"accepted_{k}"] = np.asarray(accepted[k])
    return out


def write_golden(path=GOLDEN):
    """Regenerate the golden file (run against the PRE-change code only)."""
    flat = {}
    for name in GOLDEN_SCENARIOS:
        for key, arr in golden_stream(name).items():
            flat[f"{name}/{key}"] = arr
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **flat)
    return path


class GoldenStreamTest(unittest.TestCase):
    """(v) knobs off (and pool/temper off with the adaptation on): bit-identical to HEAD."""

    def test_the_proposal_stream_is_bit_identical_to_head(self):
        ref = np.load(GOLDEN)
        names = sorted({k.split("/", 1)[0] for k in ref.files})
        self.assertEqual(names, sorted(GOLDEN_SCENARIOS))
        for name in names:
            got = golden_stream(name)
            want = {k.split("/", 1)[1]: ref[k] for k in ref.files if k.startswith(name + "/")}
            with self.subTest(scenario=name):
                self.assertEqual(sorted(got), sorted(want))
                for key in want:
                    np.testing.assert_array_equal(got[key], want[key], err_msg=f"{name}/{key}")
                if name.startswith("single"):
                    self.assertGreater(len(got["q"]), 0)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _env(persist=False, **extra):
    env = {k: v for k, v in os.environ.items() if k not in _ALL_KNOBS}
    if not persist:
        env["EIGEN_TABLES_PERSIST"] = "0"
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


def _stub(table, *, ndim, ntemps, nwalkers, pool=None, temper=None, **kw):
    """test_eigen_axis_adapt's data-free add/remove move plus the two new knobs."""
    move = A._stub(table, ndim=ndim, ntemps=ntemps, nwalkers=nwalkers, **kw)
    move.eigen_adapt_pool = pool
    move.eigen_temper_sigmas = temper
    return move


def _expected_sigmas(axes, floor, xs, betas, box, max_factor=100.0):
    """The adapted sigma from rung samples ``xs`` ``[(nt, nw, nd), ...]`` (no periodics),
    with the default safety cap ``{BRANCH}_EIGEN_ADAPT_MAX_FACTOR`` = 100."""
    from lisatools.globalfit.moves.addremovemove import factor_capped_sigmas

    nt = xs[0].shape[0]
    rungs = [(np.concatenate([x[k] for x in xs]),
              np.concatenate([np.full(x.shape[1], betas[k]) for x in xs])) for k in range(nt)]
    emp = eigen_refresh.empirical_axis_widths(axes, eigen_refresh.cold_unit_deviations(rungs))
    widths = np.full(axes.shape[-1], 2.0 * box)
    sig = eigen_refresh.adapt_axis_sigmas(axes, floor, emp, widths)
    return factor_capped_sigmas(floor, sig, max_factor), emp


# ---------------------------------------------------------------------------
# (i) pooling across ranks: FakeWorld, the real fan-out, a stub body
# ---------------------------------------------------------------------------

POOL_NT, POOL_ND, POOL_NL, POOL_BOX = 4, 5, 1, 200.0
POOL_BETAS = np.array([1.0, 0.6, 0.35, 0.2])
POOL_R = A._rot(POOL_ND, 61)
POOL_FISHER = np.full(POOL_ND, 0.1)
#: walker 0 STUCK at one end of the ridge (every rung within 1e-4), walker 1
#: SPANS it (20x the table sigma along axis 4)
POOL_SPREAD = {0: np.full(POOL_ND, 1e-4), 1: A.LONG}
POOL_MU = {0: -40.0 * POOL_R[:, 4], 1: np.zeros(POOL_ND)}


def _pool_draw(walker, n):
    """Walker ``walker``'s rung ensemble after its ``n``-th body (n = 0: the start)."""
    rng = np.random.default_rng(1000 * walker + n)
    z = rng.normal(size=(POOL_NT, POOL_ND)) * POOL_SPREAD[walker]
    return POOL_MU[walker] + (z @ POOL_R.T) / np.sqrt(POOL_BETAS)[:, None]


def _pool_state(nw=2):
    tempered = np.stack([_pool_draw(w, 0) for w in range(nw)], axis=1)[:, :, None, :]
    state = GFState(
        {BR: tempered[:1].copy()},
        inds={BR: np.ones((1, nw, POOL_NL), dtype=bool)},
        log_like=np.zeros((1, nw)), log_prior=np.zeros((1, nw)), betas=np.ones(1),
        sub_state_bases={BR: EMRIState},
    )
    sub = state.sub_states[BR]
    sub.betas_all = np.tile(POOL_BETAS, (POOL_NL, 1))
    sub.initialize_tempered(POOL_NT, nw, POOL_NL, POOL_ND, coords=tempered)
    return state


class _PoolBody(ResidualAddOneRemoveOneMove):
    """The REAL add/remove fan-out hooks around a synthetic per-block body.

    Per leaf visit: the real table refresh (adaptation, pool, tempering),
    then each local walker's rungs are redrawn from its own fixed law
    (:func:`_pool_draw`), then the real end-of-visit buffer note.
    """

    gf_move_name = "emri_pe"

    def propose_local(self, model, state):
        new = GFState(state, copy=True)
        sub = new.sub_states[BR]
        work = sub.branch
        self._n_body = getattr(self, "_n_body", 0) + 1
        for leaf in range(POOL_NL):
            betas = np.array(sub.betas_all[leaf][: self.ntemps])
            self.refresh_inner_move_tables(leaf, work, betas=betas)
            for j in range(work.coords.shape[1]):
                work.coords[: self.ntemps, j, leaf] = _pool_draw(self._w0 + j, self._n_body)
            self._note_eigen_adapt_sample(leaf, work, betas)
        self._sync_cold_row(new)
        return new, np.zeros((1, work.coords.shape[1]), dtype=bool)


def _pool_body(w0, nw, pool):
    base = _stub((POOL_R, POOL_FISHER), ndim=POOL_ND, ntemps=POOL_NT, nwalkers=nw,
                 pool=pool, min_visits=3, box=POOL_BOX)
    move = _PoolBody.__new__(_PoolBody)
    move.__dict__.update(base.__dict__)
    move._w0 = int(w0)
    move.fanout_branches = [BR]
    move.temperature_controls = []
    move.num_repeats = 1
    return move


def _pool_snap(move):
    return dict(
        table=tuple(np.array(a, copy=True) for a in move._eigen_tables[0]),
        pool=[(np.array(x), np.array(b)) for x, b in
              (getattr(move, "_eigen_pool_buffer", None) or {}).get(0, [])],
        own=[(np.array(x), np.array(b)) for x, b in
             (getattr(move, "_eigen_adapt_buffer", None) or {}).get(0, [])],
    )


def _pool_world(pool, n_proposes=4):
    """2 compute ranks (one walker each) + a saver; the head records its inputs."""
    world = FakeWorld(3, timeout=120.0)

    def fn(rank, comm):
        layout = build_layout(comm, 2, [], legacy=False, gpu_routing=False)
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return dict(role="saver")
        w0, w1 = layout.block_of(rank)
        move = _pool_body(w0, w1 - w0, pool)
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        move.install_walker_fanout(SimpleNamespace(fanout=fo, rank=rank))
        fcomm.barrier()
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("full_pe", "pe")
            inputs, state = [], _pool_state()
            try:
                for _ in range(n_proposes):
                    inputs.append(GFState(state, copy=True))
                    state, _acc = move.propose("model", state)
            finally:
                fo.stop()
            return dict(role="head", w0=w0, inputs=inputs, final=state, snap=_pool_snap(move))
        ComputeService(fcomm, layout, rank, registry={("full_pe", "emri_pe"): move},
                       model="model").serve()
        return dict(role="compute", w0=w0, snap=_pool_snap(move))

    return world.run(fn)


class PoolAcrossRanksTest(unittest.TestCase):
    """(i) a STUCK walker's rank adapts to the UNION's width with the knob on."""

    def test_the_stuck_rank_adapts_to_the_union_width(self):
        with _env():
            on = _pool_world(True)
            off = _pool_world(False)
        computes = {r: v for r, v in on.items() if v["role"] != "saver"}
        self.assertEqual(len(computes), 2)
        inputs = on[0]["inputs"]
        union = [np.asarray(s.sub_states[BR].coords[:POOL_NT, :, 0]) for s in inputs]
        want, emp = _expected_sigmas(POOL_R, POOL_FISHER, union, POOL_BETAS, POOL_BOX)
        stuck = next(v for v in computes.values() if v["w0"] == 0)
        for rank, res in computes.items():
            with self.subTest(rank=rank, walker=res["w0"]):
                ring = res["snap"]["pool"]
                # every rank buffered the head's FULL ensemble, every propose
                self.assertEqual(len(ring), len(inputs))
                for (x, b), u in zip(ring, union):
                    self.assertEqual(x.shape, (POOL_NT, 2, POOL_ND))
                    np.testing.assert_array_equal(x, u)
                    np.testing.assert_array_equal(b, POOL_BETAS)
                # ... and adapted from it on its own axes
                axes, sig = res["snap"]["table"]
                np.testing.assert_array_equal(axes, POOL_R)
                np.testing.assert_allclose(sig, want, rtol=1e-12, atol=0)
        # the stuck walker learned the ridge: the union width, not its 1e-4 --
        # ~200x the table sigma, so at the default safety cap (100x)
        self.assertGreater(emp[4] / POOL_FISHER[4], 150.0)
        np.testing.assert_allclose(stuck["snap"]["table"][1][4], 100.0 * POOL_FISHER[4], rtol=1e-12)
        # paired control, knob OFF: the stuck rank adapts from its own walker
        # (the last refresh saw its first 3 end-of-visit ensembles; the 4th
        # is noted after it) -> the information-matrix floor
        stuck_off = next(v for v in off.values() if v.get("w0") == 0 and v["role"] != "saver")
        self.assertEqual(stuck_off["snap"]["pool"], [])
        own = [x for x, _ in stuck_off["snap"]["own"]]
        self.assertEqual(len(own), 4)
        for k, x in enumerate(own, start=1):
            self.assertEqual(x.shape, (POOL_NT, 1, POOL_ND))
            np.testing.assert_array_equal(x[:, 0], _pool_draw(0, k))
        want_off, _ = _expected_sigmas(POOL_R, POOL_FISHER, own[:3], POOL_BETAS, POOL_BOX)
        np.testing.assert_allclose(stuck_off["snap"]["table"][1], want_off, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(stuck_off["snap"]["table"][1], POOL_FISHER)
        # the merged end state is the same either way (the body ignores the table)
        np.testing.assert_array_equal(on[0]["final"].sub_states[BR].coords,
                                      off[0]["final"].sub_states[BR].coords)


class PoolSeamTest(unittest.TestCase):
    """The head-side note, the payload, the rank-side buffer and the fallbacks."""

    def _head(self, pool=True, adapt=True, nw=4):
        move = _stub((POOL_R, POOL_FISHER), ndim=POOL_ND, ntemps=POOL_NT, nwalkers=nw,
                     pool=pool, adapt=adapt, min_visits=2, box=POOL_BOX)
        move.fanout = SimpleNamespace(single=False, layout=SimpleNamespace(replica_mode=False))
        return move

    def _state(self, nw=4, nl=2, dead=()):
        rng = np.random.default_rng(80)
        coords = rng.normal(size=(POOL_NT + 2, nw, nl, POOL_ND))   # a longer stored ladder
        inds = np.ones(coords.shape[:3], dtype=bool)
        for leaf in dead:
            inds[:, :, leaf] = False
        betas_all = np.tile(np.r_[POOL_BETAS, 0.1, 0.05], (nl, 1))
        sub = SimpleNamespace(coords=coords, inds=inds, betas_all=betas_all,
                              tempered_initialized=True)
        return SimpleNamespace(sub_states={BR: sub})

    def test_the_head_ships_every_walker_and_rung_of_each_alive_leaf(self):
        state = self._state(dead=(1,))
        with _env():
            head = self._head()
            head.fanout_prepare_state(None, state)
            with self.assertLogs(LOGGER, level="DEBUG") as cm:
                extra = head.fanout_payload_extra()
        self.assertEqual(sorted(extra), ["dbg_step", "eigen_pool", "fancy_swap_clock"])
        self.assertEqual(sorted(extra["eigen_pool"]), [0])           # leaf 1 is dead
        x, b = extra["eigen_pool"][0]
        sub = state.sub_states[BR]
        np.testing.assert_array_equal(x, sub.coords[:POOL_NT, :, 0])  # the sampled rungs only
        np.testing.assert_array_equal(b, POOL_BETAS)
        self.assertIn("[EMRI_EIGEN] propose payload: pooled adaptation sample, 1 leaves x "
                      "4 rungs x 4 walkers", "\n".join(cm.output))
        # shipped once: the next payload carries nothing until the next note
        self.assertNotIn("eigen_pool", head.fanout_payload_extra())

    def test_off_single_and_flat_layouts_ship_nothing(self):
        state = self._state()
        cases = {
            "pool off": dict(pool=False),
            "adapt off": dict(adapt=False),
            "pool None, env unset": dict(pool=None),
        }
        with _env():
            for name, kw in cases.items():
                with self.subTest(name):
                    head = self._head(**kw)
                    head.fanout_prepare_state(None, state)
                    self.assertEqual(sorted(head.fanout_payload_extra()),
                                     ["dbg_step", "fancy_swap_clock"])
            single = self._head()
            single.fanout = None                                    # single process
            single.fanout_prepare_state(None, state)
            self.assertIsNone(single._eigen_pool_outgoing)
            flat = self._head()                                     # GF_GPU_ROUTING replicas
            flat.fanout = SimpleNamespace(single=False, layout=SimpleNamespace(
                replica_mode=True, n_blocks=4))
            flat.fanout_prepare_state(None, state)
            self.assertIsNone(flat._eigen_pool_outgoing)
        with _env(EMRI_EIGEN_ADAPT_POOL="1", EMRI_EIGEN_AXIS_ADAPT="1"):
            head = self._head(pool=None, adapt=None)               # env-resolved
            head.fanout_prepare_state(None, state)
            self.assertIn("eigen_pool", head.fanout_payload_extra())

    def test_a_rank_without_a_shipped_sample_runs_todays_adaptation(self):
        """Fallback: the knob on, no payload ever received -> the own buffer."""
        X = A._tempered_draws(np.random.default_rng(81), POOL_R, A.LONG, POOL_BETAS, 12, 1)
        tabs = {}
        with _env():
            for pool in (False, True):
                move = _stub((POOL_R, POOL_FISHER), ndim=POOL_ND, ntemps=POOL_NT, nwalkers=1,
                             pool=pool, min_visits=10, box=POOL_BOX)
                move._fanout_body = True
                A._feed(move, X, POOL_BETAS)
                move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
                tabs[pool] = move._eigen_tables[0][1]
                self.assertFalse(getattr(move, "_eigen_pool_buffer", None))
        np.testing.assert_array_equal(tabs[True], tabs[False])
        self.assertGreater(tabs[True][4], 1.0)

    def test_a_short_pool_falls_back_to_the_own_buffer(self):
        X = A._tempered_draws(np.random.default_rng(82), POOL_R, A.LONG, POOL_BETAS, 12, 1)
        with _env():
            ref = _stub((POOL_R, POOL_FISHER), ndim=POOL_ND, ntemps=POOL_NT, nwalkers=1,
                        pool=False, min_visits=10, box=POOL_BOX)
            move = _stub((POOL_R, POOL_FISHER), ndim=POOL_ND, ntemps=POOL_NT, nwalkers=1,
                         pool=True, min_visits=10, box=POOL_BOX)
            move._fanout_body = True
            for m in (ref, move):
                A._feed(m, X, POOL_BETAS)
            for x in X[:9]:                                     # 9 < the 10-visit minimum
                move.fanout_apply_extra({"eigen_pool": {0: (np.repeat(x, 3, axis=1), POOL_BETAS)}})
            self.assertEqual(len(move._eigen_pool_buffer[0]), 9)
            for m in (ref, move):
                m.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        np.testing.assert_array_equal(move._eigen_tables[0][1], ref._eigen_tables[0][1])

    def test_the_log_line_says_pooled_over_W_walkers(self):
        R = POOL_R
        fisher = (R, np.array([0.1, 0.1, 0.1, 0.1, 0.25]))
        X1 = A._tempered_draws(np.random.default_rng(83), R, A.LONG, A.BETAS8, 50, 1)
        X4 = A._tempered_draws(np.random.default_rng(84), R, A.LONG, A.BETAS8, 50, 4)
        pat = r"\[EMRI_EIGEN\] leaf 0: step scale adapted on 1 of 5 axes, .*\(50 visits, 8 rungs%s\)"
        with _env():
            # a block rank fed by the head: pooled over the shipped 4 walkers
            rank = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, pool=True)
            rank._fanout_body = True
            A._feed(rank, X1, A.BETAS8)
            for x in X4:
                rank.fanout_apply_extra({"eigen_pool": {0: (x, A.BETAS8)}})
            with self.assertLogs(LOGGER, level="INFO") as cm:
                rank.refresh_inner_move_tables(0, SimpleNamespace(coords=X1[-1][:, :, None, :]))
            self.assertRegex("\n".join(cm.output), pat % ", pooled over 4 walkers")
            # single process with the knob on: the own buffer IS the union
            single = _stub(fisher, ndim=5, ntemps=8, nwalkers=4, pool=True)
            A._feed(single, X4, A.BETAS8)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                single.refresh_inner_move_tables(0, SimpleNamespace(coords=X4[-1][:, :, None, :]))
            self.assertRegex("\n".join(cm.output), pat % ", pooled over 4 walkers")
            # knob off: today's line, word for word
            plain = _stub(fisher, ndim=5, ntemps=8, nwalkers=4, pool=False)
            A._feed(plain, X4, A.BETAS8)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                plain.refresh_inner_move_tables(0, SimpleNamespace(coords=X4[-1][:, :, None, :]))
            self.assertRegex("\n".join(cm.output), pat % "")
            self.assertNotIn("pooled", "\n".join(cm.output))
        np.testing.assert_array_equal(single._eigen_tables[0][1], plain._eigen_tables[0][1])


# ---------------------------------------------------------------------------
# (ii) the per-rung tempered step
# ---------------------------------------------------------------------------

T_BETAS = np.array([1.0, 0.5, 0.2, 0.05])
T_SIG = np.array([0.1, 0.2, 0.3, 0.4, 2.0])


class TemperTest(unittest.TestCase):
    def _move(self, temper=True, nw=3, table=None, **kw):
        R = A._rot(5, 71)
        return _stub(table or (R, T_SIG), ndim=5, ntemps=4, nwalkers=nw, adapt=False,
                     temper=temper, **kw), R

    def test_rung_k_steps_with_sigma_over_sqrt_beta_and_the_cold_rung_is_unchanged(self):
        rng = np.random.default_rng(90)
        coords = rng.normal(size=(4, 3, 1, 5))
        with _env():
            move, R = self._move()
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=coords), betas=T_BETAS)
        axes5, sig4 = move.moves[0]._tables[BR]
        self.assertEqual((axes5.shape, sig4.shape), ((4, 3, 1, 5, 5), (4, 3, 1, 5)))
        np.testing.assert_array_equal(axes5, np.broadcast_to(R, axes5.shape))
        np.testing.assert_array_equal(sig4[0], np.broadcast_to(T_SIG, (3, 1, 5)))
        for k, b in enumerate(T_BETAS):
            np.testing.assert_allclose(sig4[k], np.broadcast_to(T_SIG / np.sqrt(b), (3, 1, 5)),
                                       rtol=1e-15, atol=0)
        # the STORED table stays the cold one
        np.testing.assert_array_equal(move._eigen_tables[0][1], T_SIG)
        # the same random stream: rung 0 draws are bit-identical to the plain
        # (untempered) move, rung k's are exactly 1/sqrt(beta_k) wider
        plain = EigenAxisMove()
        plain.set_axes(BR, R, T_SIG)
        q_on = move.moves[0].get_proposal({BR: coords}, np.random.RandomState(5))[0][BR]
        q_off = plain.get_proposal({BR: coords}, np.random.RandomState(5))[0][BR]
        np.testing.assert_array_equal(q_on[0], q_off[0])
        for k in range(1, 4):
            np.testing.assert_allclose(q_on[k] - coords[k], (q_off[k] - coords[k]) / np.sqrt(T_BETAS[k]),
                                       rtol=1e-9, atol=1e-14)

    def test_the_proposal_std_per_rung_and_axis(self):
        """Statistically: std of the step along axis j on rung k = sigma_j / sqrt(beta_k)."""
        nw = 2500
        coords = np.zeros((4, nw, 1, 5))
        with _env():
            move, R = self._move(nw=nw)
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=coords), betas=T_BETAS)
            q = move.moves[0].get_proposal({BR: coords}, np.random.default_rng(91))[0][BR]
        c = np.einsum("ij,kwj->kwi", np.linalg.inv(R), q[:, :, 0, :])     # axis-basis steps
        for k, b in enumerate(T_BETAS):
            for j in range(5):
                rows = np.abs(c[k, :, j]) > 1e-12
                with self.subTest(rung=k, axis=j):
                    self.assertGreater(rows.sum(), 350)
                    got = np.sqrt(np.mean(c[k, rows, j] ** 2))
                    self.assertAlmostEqual(got / (T_SIG[j] / np.sqrt(b)), 1.0, delta=0.12)

    def test_off_installs_the_shared_cold_table(self):
        coords = np.random.default_rng(92).normal(size=(4, 3, 1, 5))
        for kw, env in ((dict(temper=False), {}), (dict(temper=None), {}),
                        (dict(temper=None), {"EMRI_EIGEN_TEMPER_SIGMAS": "0"})):
            with self.subTest(**kw, env=env), _env(**env):
                move, R = self._move(**kw)
                with self.assertNoLogs(LOGGER, level="INFO"):
                    move.refresh_inner_move_tables(0, SimpleNamespace(coords=coords), betas=T_BETAS)
                axes, sig = move.moves[0]._tables[BR]
                self.assertEqual(axes.shape, (5, 5))
                np.testing.assert_array_equal(sig, T_SIG)
                self.assertFalse(getattr(move, "_eigen_rung_scales", {}))
                # a split install is still the no-op it was
                move._install_eigen_split_table(move.moves[0], 0, np.ones((4, 3), dtype=bool))
                self.assertEqual(move.moves[0]._tables[BR][0].shape, (5, 5))
        with _env(EMRI_EIGEN_TEMPER_SIGMAS="1"):
            move, _ = self._move(temper=None)
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=coords), betas=T_BETAS)
            self.assertEqual(move.moves[0]._tables[BR][0].ndim, 5)

    def test_the_cap_and_the_floor_hold(self):
        """A railed axis stays at the cap at every rung; a hot step is capped."""
        nd = 3
        R = np.eye(nd)
        box = 1.0                                          # cap 2 per axis
        sig = np.array([2.0, 1.0, 0.01])                   # axis 0 railed; axis 1 hits the cap
        move = _stub((R, sig), ndim=nd, ntemps=4, nwalkers=2, adapt=False, temper=True, box=box)
        with _env():
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((4, 2, 1, nd))),
                                           betas=T_BETAS)
        sig4 = move.moves[0]._tables[BR][1][:, 0, 0]
        np.testing.assert_array_equal(sig4[:, 0], 2.0)
        np.testing.assert_array_equal(sig4[:, 1], np.minimum(1.0 / np.sqrt(T_BETAS), 2.0))
        np.testing.assert_allclose(sig4[:, 2], 0.01 / np.sqrt(T_BETAS), rtol=1e-15)

    def test_a_per_walker_stash_is_tempered_row_by_row(self):
        nt, nw, nd = 4, 4, 5
        rng = np.random.default_rng(93)
        axes = np.stack([np.stack([A._rot(nd, 100 + 10 * t + w) for w in range(nw)]) for t in range(nt)])
        sig = rng.uniform(0.05, 0.5, size=(nt, nw, nd))
        move = _stub((axes, sig), ndim=nd, ntemps=nt, nwalkers=nw, adapt=False, temper=True)
        with _env():
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((nt, nw, 1, nd))),
                                           betas=T_BETAS)
            mask = np.zeros((nt, nw), dtype=bool)
            mask[:, [0, 2]] = True                         # a red/blue half
            move._install_eigen_split_table(move.moves[0], 0, mask)
        ax5, sg4 = move.moves[0]._tables[BR]
        self.assertEqual(sg4.shape, (nt, 2, 1, nd))
        for k in range(nt):
            for j, w in enumerate((0, 2)):
                np.testing.assert_array_equal(ax5[k, j, 0], axes[k, w])
                np.testing.assert_allclose(sg4[k, j, 0], sig[k, w] / np.sqrt(T_BETAS[k]), rtol=1e-15)
        np.testing.assert_array_equal(move._eigen_tables[0][1], sig)     # stored: cold

    def test_the_sidecar_keeps_the_cold_sigma(self):
        R = A._rot(5, 94)
        X = A._tempered_draws(np.random.default_rng(95), R, A.LONG, A.BETAS8, 50, 1)
        with tempfile.TemporaryDirectory() as tmp, _env(persist=True):
            store = os.path.join(tmp, "gf_prod_9mo_testing.h5")
            for adapt in (False, True):
                with self.subTest(adapt=adapt):
                    move = _stub((R, np.full(5, 0.1)), ndim=5, ntemps=8, nwalkers=1, adapt=adapt,
                                 temper=True, store=store)
                    A._feed(move, X, A.BETAS8)
                    move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]),
                                                   betas=A.BETAS8)
                    entry = eigen_table_persist.load_sidecar(
                        eigen_table_persist.sidecar_path(store))["entries"][f"{BR}:0"]
                    cold = move._eigen_tables[0][1]
                    np.testing.assert_array_equal(entry["sigmas"], cold)
                    hot = move.moves[0]._tables[BR][1][-1, 0, 0]
                    np.testing.assert_allclose(hot, np.minimum(cold * np.sqrt(6.0), hot), rtol=1e-12)
                    self.assertGreater(np.min(hot / cold), 2.0)
                    if adapt:
                        np.testing.assert_array_equal(entry["sigmas_fisher"], np.full(5, 0.1))
                        self.assertGreater(cold[4], 1.0)

    def test_the_log_says_tempered_once_per_refresh(self):
        R = A._rot(5, 96)
        coords = np.zeros((4, 1, 1, 5))
        with _env():
            move = _stub((R, T_SIG), ndim=5, ntemps=4, nwalkers=1, adapt=False, temper=True,
                         refresh=2)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                for _ in range(4):                       # refreshes at visits 1 and 3
                    move.refresh_inner_move_tables(0, SimpleNamespace(coords=coords), betas=T_BETAS)
        lines = [l for l in cm.output if "tempered 1/sqrt(beta)" in l]
        self.assertEqual(len(lines), 2, cm.output)
        self.assertIn("[EMRI_EIGEN] leaf 0: step tempered 1/sqrt(beta) over 4 rungs, x1.00..x4.47",
                      lines[0])
        # with the adaptation line firing, the note rides in it (still once)
        fisher = (R, np.array([0.1, 0.1, 0.1, 0.1, 0.25]))
        X = A._tempered_draws(np.random.default_rng(97), R, A.LONG, A.BETAS8, 50, 1)
        with _env():
            move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, temper=True)
            A._feed(move, X, A.BETAS8)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]),
                                               betas=A.BETAS8)
        self.assertEqual(len(cm.output), 1, cm.output)
        self.assertRegex(cm.output[0], r"step scale adapted on 1 of 5 axes, .*\(50 visits, 8 rungs, "
                                       r"tempered 1/sqrt\(beta\) over 8 rungs, x1\.00\.\.x2\.45\)")

    def test_the_ladder_defaults_to_the_leafs_temperature_control(self):
        with _env():
            move, _ = self._move()
            move.temperature_controls = [SimpleNamespace(betas=T_BETAS)]
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((4, 3, 1, 5))))
            np.testing.assert_allclose(move.moves[0]._tables[BR][1][3, 0, 0], T_SIG / np.sqrt(0.05))
            # no ladder at all: warn once, the cold table
            bare, _ = self._move()
            with self.assertLogs(LOGGER, level="WARNING") as cm:
                bare.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((4, 3, 1, 5))))
            self.assertIn("no ladder to temper", "\n".join(cm.output))
            self.assertEqual(bare.moves[0]._tables[BR][0].shape, (5, 5))


class RealProposeTemperTest(E._Case):
    """The real ``EMRIDirectLikeMove`` single-process: every inner draw is rung-tempered."""

    def test_every_draw_sees_the_tempered_rows(self):
        seen = []
        real = E._make_move

        def make(acs, nw, **kw):
            move = real(acs, nw, **kw)
            inner = move.moves[0]
            orig = inner.get_proposal

            def spy(*a, **k):
                axes, sig = inner._tables[BR]
                cold = move._eigen_tables[int(move._current_leaf)][1]
                seen.append((np.array(axes), np.array(sig), np.array(cold)))
                return orig(*a, **k)

            inner.get_proposal = spy
            return move

        with _env(EMRI_EIGEN_TEMPER_SIGMAS="1"), \
                mock.patch.object(E, "_make_move", make):
            out = self._single(2, scope="walker_max", permute_every=1, adaptive=True,
                               rng=lambda: np.random.RandomState(H.SEED))
        self.assertEqual(len(seen), self.N_PROPOSES * H.NLEAVES * H.NUM_REPEATS)
        for axes, sig, cold in seen:
            self.assertEqual(sig.shape, (H.NTEMPS, 2, 1, H.NDIM))
            np.testing.assert_array_equal(sig[0], np.broadcast_to(cold, sig[0].shape))
            # rung 1 (beta 0.5): sqrt(2) wider wherever the cap allows
            ratio = sig[1, 0, 0] / cold
            self.assertTrue(np.all(ratio <= np.sqrt(2.0) * (1 + 1e-12)))
            self.assertTrue(np.all(ratio >= 1.0))
            self.assertGreater(np.max(ratio), 1.4)
        final = out["states"][-1].sub_states["emri"]
        self.assertGreater(int(final.in_model_accepted.sum()), 0)
        bad = [line for line in self.cap.lines if any(b in line for b in H._BAD_WARNINGS)]
        self.assertEqual(bad, [], "\n".join(self.cap.lines))


# ---------------------------------------------------------------------------
# (iv) the routed 4 walkers x 2 replicas layout, and the real 4-rank block fan-out
# ---------------------------------------------------------------------------

_REAL_SNAPSHOT = E._snapshot


def _snap_eigen(move, acs, w0):
    out = _REAL_SNAPSHOT(move, acs, w0)
    out["own"] = {k: [np.array(x) for x, _ in v]
                  for k, v in (getattr(move, "_eigen_adapt_buffer", None) or {}).items()}
    out["pool"] = {k: [(np.array(x), np.array(b)) for x, b in v]
                   for k, v in (getattr(move, "_eigen_pool_buffer", None) or {}).items()}
    out["fisher"] = {k: np.array(v) for k, v in (getattr(move, "_eigen_fisher_sigmas", None) or {}).items()}
    out["installed"] = tuple(np.array(a) for a in move.moves[0]._tables.get(BR, ())) \
        if getattr(move.moves[0], "_tables", None) else None
    out["periodic"] = move._eigen_periodic_columns()
    out["box"] = eigen_refresh.prior_box_widths(move.priors[BR], move.ndim)
    out["max_factor"] = move._eigen_adapt_max_factor()
    return out


class RoutedReplicaPoolTest(E._Case):
    """(iv) ``NGPUS=8 GF_GPU_ROUTING=1``: 4 walker blocks x R = 2 replicas.

    The add/remove family is flat-capable, so under the routing the HEAD runs
    the full 4-walker body and the replicas only score rows: the head's own
    buffer already holds every walker (the pool needs no payload, and none is
    built), the replicas never build or hold a table, and the routed run with
    BOTH knobs on equals the single-process run with both knobs on."""

    ENV = {"EMRI_EIGEN_AXIS_ADAPT": "1", "EMRI_EIGEN_ADAPT_MIN_VISITS": "1",
           "EMRI_EIGEN_ADAPT_POOL": "1", "EMRI_EIGEN_TEMPER_SIGMAS": "1",
           "EIGEN_TABLES_PERSIST": "0"}

    def test_four_blocks_two_replicas_each_with_pool_and_temper_on(self):
        kw = dict(scope="walker_max", permute_every=1, adaptive=True,
                  rng=lambda: np.random.RandomState(H.SEED))
        with _env(**self.ENV), mock.patch.object(E, "_snapshot", _snap_eigen):
            ref = self._single(4, **kw)
            out = self._world(8, 4, gpu_routing=True, **kw)
        layout = out[0]["layout"]
        self.assertEqual((layout.n_compute, layout.n_blocks, layout.ranks_per_block), (8, 4, 2))
        computes = self._assert_match(out, ref)          # bit-for-bit the single-process chain
        head = out[0]["snap"]
        self.assertEqual(head["pool"], {})                # nothing shipped, nothing needed
        for leaf in range(H.NLEAVES):
            self.assertEqual(len(head["own"][leaf]), self.N_PROPOSES)
            self.assertEqual(head["own"][leaf][-1].shape, (H.NTEMPS, 4, H.NDIM))   # the union
            np.testing.assert_array_equal(head["tables"][leaf][1], ref["snap"]["tables"][leaf][1])
            self.assertIn(leaf, head["fisher"])
        self.assertEqual(head["installed"][1].shape[:2], (H.NTEMPS, 4))   # tempered rows
        for rank, res in computes.items():
            if rank == 0:
                continue
            with self.subTest(rank=rank):
                self.assertEqual(res["snap"]["tables"], {})      # replicas hold no table
                self.assertEqual(res["snap"]["pool"], {})
                self.assertEqual(res["snap"]["own"], {})


class BlockPoolRealTest(E._Case):
    """The real EMRI move on the 4-rank walker-block fan-out (one walker per rank)."""

    ENV = {"EMRI_EIGEN_AXIS_ADAPT": "1", "EMRI_EIGEN_ADAPT_MIN_VISITS": "1",
           "EMRI_EIGEN_ADAPT_POOL": "1", "EIGEN_TABLES_PERSIST": "0"}

    def test_every_rank_adapts_from_the_union_on_its_own_axes(self):
        kw = dict(scope="walker_max", adaptive=False, rng=H._RowLocalRandom, permute_every=0)
        with _env(**self.ENV), mock.patch.object(E, "_snapshot", _snap_eigen), \
                H._row_local_numpy():
            out = self._world(4, 4, **kw)
        inputs = [E._state(range(4))] + list(out[0]["states"][:-1])
        computes = {r: v for r, v in out.items() if v["role"] != "saver"}
        self.assertEqual(len(computes), 4)
        for rank, res in computes.items():
            snap = res["snap"]
            with self.subTest(rank=rank, walker=snap["w0"]):
                for leaf in range(H.NLEAVES):
                    ring = snap["pool"][leaf]
                    self.assertEqual(len(ring), self.N_PROPOSES)
                    union = []
                    for (x, b), s in zip(ring, inputs):
                        sub = s.sub_states["emri"]
                        np.testing.assert_array_equal(x, sub.coords[:, :, leaf])
                        np.testing.assert_array_equal(b, sub.betas_all[leaf])
                        union.append(x)
                    self.assertEqual(snap["own"][leaf][-1].shape[1], 1)   # its own buffer: 1 walker
                    axes, sig = snap["tables"][leaf]
                    inds, per = snap["periodic"]
                    rungs = [(np.concatenate([x[k] for x in union]),
                              np.concatenate([np.full(x.shape[1], H.BETAS[k]) for x in union]))
                             for k in range(H.NTEMPS)]
                    emp = eigen_refresh.empirical_axis_widths(
                        axes, eigen_refresh.cold_unit_deviations(rungs, inds, per))
                    from lisatools.globalfit.moves.addremovemove import factor_capped_sigmas

                    self.assertEqual(snap["max_factor"], 100.0)           # the default cap
                    floor = snap["fisher"][leaf]
                    want = factor_capped_sigmas(
                        floor, eigen_refresh.adapt_axis_sigmas(axes, floor, emp, snap["box"]),
                        snap["max_factor"])
                    np.testing.assert_allclose(sig, want, rtol=1e-12, atol=0)
        bad = [line for line in self.cap.lines if any(b in line for b in H._BAD_WARNINGS)]
        self.assertEqual(bad, [], "\n".join(self.cap.lines))


# ---------------------------------------------------------------------------
# knobs: move, settings fields, the stock stamping
# ---------------------------------------------------------------------------

class KnobTest(unittest.TestCase):
    KNOBS = (("eigen_adapt_pool", "EIGEN_ADAPT_POOL", "_eigen_adapt_pool"),
             ("eigen_temper_sigmas", "EIGEN_TEMPER_SIGMAS", "_eigen_temper_sigmas"))

    def test_the_move_resolves_kwarg_then_env_then_off(self):
        for branch in ("emri", "mbh", "sobbh"):
            B = branch.upper()
            mv = ResidualAddOneRemoveOneMove(
                branch, (2, 2, 1, 4), lambda *a, **k: None, {}, {}, None, 2, None, None,
                [EigenAxisMove()])
            for attr, knob, getter in self.KNOBS:
                with self.subTest(branch=branch, knob=knob):
                    self.assertIsNone(getattr(mv, attr))
                    with _env():
                        self.assertIs(getattr(mv, getter)(), False)
                    for raw, want in (("1", True), ("on", True), ("0", False), ("no", False)):
                        with _env(**{f"{B}_{knob}": raw}):
                            self.assertIs(getattr(mv, getter)(), want)
                    with _env(**{f"{B}_{knob}": "1"}):
                        setattr(mv, attr, False)                     # the attribute wins
                        self.assertIs(getattr(mv, getter)(), False)
                        setattr(mv, attr, None)
        mv = ResidualAddOneRemoveOneMove(
            "emri", (2, 2, 1, 4), lambda *a, **k: None, {}, {}, None, 2, None, None,
            [EigenAxisMove()], eigen_adapt_pool=True, eigen_temper_sigmas=True)
        with _env():
            self.assertEqual((mv._eigen_adapt_pool(), mv._eigen_temper_sigmas()), (True, True))

    def test_the_settings_fields_default_off_and_read_the_env(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        classes = {"emri": (EMRISettings, sr.SourceEMRISettings),
                   "mbh": (MBHSettings, sr.SourceMBHSettings),
                   "sobbh": (SOBBHSettings, sr.SourceSOBBHSettings)}
        for branch, clss in classes.items():
            for cls in clss:
                for attr, knob, _ in self.KNOBS:
                    with self.subTest(cls=cls.__name__, attr=attr):
                        with _env():
                            self.assertIs(getattr(cls(), attr), False)
                        with _env(**{f"{branch.upper()}_{knob}": "1"}):
                            s = cls()
                        self.assertIs(getattr(s, attr), True)
                        self.assertIs(getattr(pickle.loads(pickle.dumps(copy.deepcopy(s))), attr), True)

    def test_the_stock_build_stamps_the_fields_onto_the_move(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        def built(name):
            return SimpleNamespace(eigen_adapt_pool=None, eigen_temper_sigmas=None, tag=name)

        info = {
            "mbh": SimpleNamespace(eigen_adapt_pool=True, eigen_temper_sigmas=False),
            "emri": SimpleNamespace(eigen_adapt_pool=False, eigen_temper_sigmas=True),
            "sobbh": SimpleNamespace(eigen_adapt_pool=None, eigen_temper_sigmas=None),
        }
        curr = SimpleNamespace(source_info=info)
        with mock.patch.object(sr, "build_mbh_move_runtime", lambda *a: built("mbh")), \
                mock.patch.object(sr, "build_emri_move_runtime", lambda *a: built("emri")), \
                mock.patch.object(sr, "build_sobbh_move_runtime", lambda *a: built("sobbh")):
            moves = sr.build_source_moves(curr, None, None, None, {})
        self.assertEqual(sorted(moves), ["emri_pe", "mbh_pe", "sobbh_pe"])
        self.assertEqual((moves["mbh_pe"].eigen_adapt_pool, moves["mbh_pe"].eigen_temper_sigmas),
                         (True, False))
        self.assertEqual((moves["emri_pe"].eigen_adapt_pool, moves["emri_pe"].eigen_temper_sigmas),
                         (False, True))
        self.assertEqual((moves["sobbh_pe"].eigen_adapt_pool, moves["sobbh_pe"].eigen_temper_sigmas),
                         (None, None))
        # an attribute the builder already set wins; a move without it is untouched
        mv = SimpleNamespace(eigen_adapt_pool=False, eigen_temper_sigmas=None)
        sr.stamp_source_move_settings(mv, SimpleNamespace(eigen_adapt_pool=True, eigen_temper_sigmas=True))
        self.assertEqual((mv.eigen_adapt_pool, mv.eigen_temper_sigmas), (False, True))
        other = SimpleNamespace()
        sr.stamp_source_move_settings(other, SimpleNamespace(eigen_adapt_pool=True))
        self.assertFalse(hasattr(other, "eigen_adapt_pool"))


if __name__ == "__main__":
    unittest.main()
