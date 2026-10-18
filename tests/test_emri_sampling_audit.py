"""The EMRI sampling audit adjustments (user ruling 2026-10-18; docs/emri-sampling-audit.md).

Why (9mo job 773, rows 840-1007): the loud EMRIs' posteriors are a COMB of
discrete islands along their intrinsic degeneracy (EMRI 3: p0 12.3307 /
12.3310 / 12.3329 / 12.3346 / 12.3371, within ~10 nats of each other), each
walker's ladder holds one or two of them, the step-scale adaptation widened
the narrow intrinsic axes x14-100 from the between-island offsets (the
walkers' own spread along them is 0.04-0.2 of the widened step), and nothing
moves a walker from one island to the next. Three adjustments, code default
OFF (the launchers turn them on for EMRI):

* B1 ``{BRANCH}_EIGEN_AXIS_ACC_LOG``: per-axis proposals / acceptances of the
  eigen move (cold rung vs hot rungs), logged once per table refresh;
* B2 ``{BRANCH}_EIGEN_AXIS_ACC_TUNE``: an axis whose cold acceptance falls
  below ``_ACC_MIN`` has its step divided by ``_ACC_FACTOR`` (floor = the
  Fisher sigma), one above ``_ACC_MAX`` multiplied (ceiling = the adapted
  width / the box cap); frozen inside a visit;
* B3 ``{BRANCH}_DE_JUMP_MIX``: the cross-walker difference jump
  ``x + gamma (x_a - x_b)`` with partners from the OTHER walker group's
  start-of-propose states, the groups alternating by global iteration.

Pinned here (CPU): the jump's symmetry, formula, wrap and bookkeeping; the
parity schedule and the partner-pool wiring (payload, apply, the one-group
rule), single process and the 4-rank block fan-out of the REAL EMRI move
(row-local random stream: the fan-out equals the single-process run); the
tallies (recorded and derived axis picks, jump-mix rows kept apart, the log
lines); the controller (shrink, widen, floor, ceiling, carry-over, the
sidecar round trip without double application); the knobs and the stock
plumbing; telemetry on = the same proposal stream; and the CURVED-RIDGE
harness with a secondary maximum, 4 walkers x 8 rungs (numbers printed with
``EMRI_AUDIT_HARNESS_VERBOSE=1``).
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.moves import EigenAxisMove
from eryn.utils import PeriodicContainer

from lisatools.globalfit.moves import eigen_refresh, eigen_table_persist
from lisatools.globalfit.moves.addremovemove import (
    EIGEN_AXIS_ACC_MIN_PROPOSALS,
    ResidualAddOneRemoveOneMove,
    describe_inner_moves,
)
from lisatools.sampling.moves.eigenjumpmix import EigenAxisJumpMixMove
from lisatools.sampling.moves.walkerdiffjump import (
    DE_GAMMA_SCALE,
    KIND_MODE,
    KIND_SCALED,
    WalkerDifferenceJumpMove,
)

try:
    from tests import test_eigen_axis_adapt as A
except ImportError:  # pragma: no cover - run from tests/
    import test_eigen_axis_adapt as A

LOGGER = "lisatools.globalfit.moves.addremovemove"
BR = "emri"
NEW_KNOBS = tuple(f"{b}_{k}" for b in ("EMRI", "MBH", "SOBBH") for k in (
    "EIGEN_AXIS_ACC_LOG", "EIGEN_AXIS_ACC_TUNE", "EIGEN_AXIS_ACC_MIN", "EIGEN_AXIS_ACC_MAX",
    "EIGEN_AXIS_ACC_FACTOR", "DE_JUMP_MIX", "DE_JUMP_MODE_FRAC"))


def _env(persist=False, **extra):
    env = {k: v for k, v in os.environ.items() if k not in A._KNOBS + NEW_KNOBS}
    if not persist:
        env["EIGEN_TABLES_PERSIST"] = "0"
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


def _stub(table, *, ndim, ntemps, nwalkers, stack=None, tune=None, log=None, **kw):
    """test_eigen_axis_adapt's data-free move with an explicit inner-move stack."""
    move = A._stub(table, ndim=ndim, ntemps=ntemps, nwalkers=nwalkers, **kw)
    if stack is not None:
        move.moves = [m for m, _ in stack]
        w = np.array([w for _, w in stack], dtype=float)
        move.move_weights = w / w.sum()
    move.eigen_axis_acc_tune = tune
    move.eigen_axis_acc_log = log
    return move


# ---------------------------------------------------------------------------
# B3: the jump itself
# ---------------------------------------------------------------------------

class _Zero(EigenAxisMove):
    """A fallback that proposes the current point (factor 0.25 to see it kept)."""

    def get_proposal(self, branches_coords, random, branches_inds=None, **kw):
        q = {k: np.array(v, copy=True) for k, v in branches_coords.items()}
        nt, nw = next(iter(branches_coords.values())).shape[:2]
        return q, np.full((nt, nw), 0.25)


class JumpMoveTest(unittest.TestCase):
    def _coords(self, nt=3, nw=2, nd=4, seed=0):
        return np.random.default_rng(seed).normal(size=(nt, nw, 1, nd))

    def test_the_jump_is_x_plus_gamma_times_a_partner_difference(self):
        P = np.array([[0.0, 0, 0, 0], [10.0, 0, 0, 0], [0.0, 5, 0, 0]])
        move = WalkerDifferenceJumpMove(_Zero(), mode_frac=1.0)
        x = self._coords()
        active = np.array([[True, False]] * 3)
        move.set_partners(BR, active, P)
        q, f = move.get_proposal({BR: x}, np.random.default_rng(1))
        d = q[BR] - x
        # inactive rows: the fallback (the current point) and its factor
        np.testing.assert_array_equal(d[:, 1], 0.0)
        np.testing.assert_array_equal(f[:, 1], 0.25)
        np.testing.assert_array_equal(f[:, 0], 0.0)
        # active rows: gamma = 1 x (P_i - P_j), i != j
        diffs = {tuple(P[i] - P[j]) for i in range(3) for j in range(3) if i != j}
        for t in range(3):
            self.assertIn(tuple(np.round(d[t, 0, 0], 12)), {tuple(np.round(v, 12)) for v in diffs})
        np.testing.assert_array_equal(move.last_kind[BR].reshape(3, 2)[:, 0], KIND_MODE)
        np.testing.assert_array_equal(move.last_kind[BR].reshape(3, 2)[:, 1], 0)
        np.testing.assert_array_equal(move.last_pick[BR].reshape(3, 2)[:, 0], -1)

    def test_pairs_are_uniform_ordered_and_distinct_so_the_proposal_is_symmetric(self):
        P = np.arange(4, dtype=float)[:, None] * np.array([[1.0, 0.0]]) + np.array([[0.0, 1.0]])
        move = WalkerDifferenceJumpMove(_Zero(), mode_frac=1.0)
        n = 60000
        x = np.zeros((1, n, 1, 2))
        move.set_partners(BR, np.ones((1, n), dtype=bool), P)
        q, _ = move.get_proposal({BR: x}, np.random.default_rng(7))
        d = np.round(q[BR][0, :, 0, 0]).astype(int)        # P_i - P_j in column 0
        self.assertFalse(np.any(q[BR][0, :, 0, 1] != 0.0))   # identical column 1 cancels
        self.assertNotIn(0, set(d.tolist()))                 # never a state with itself
        vals, counts = np.unique(d, return_counts=True)
        # d = i - j over ordered pairs: count(k) = (4 - |k|) of 12; symmetric in k
        want = {k: (4 - abs(k)) / 12.0 for k in range(-3, 4) if k}
        for v, c in zip(vals, counts):
            self.assertAlmostEqual(c / n, want[int(v)], delta=0.01)
            self.assertAlmostEqual(c, counts[list(vals).index(-v)], delta=0.05 * c)

    def test_the_scaled_gamma_is_2_38_over_sqrt_2d(self):
        P = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        move = WalkerDifferenceJumpMove(_Zero(), mode_frac=0.0, gamma_sigma=0.1)
        n = 40000
        move.set_partners(BR, np.ones((1, n), dtype=bool), P)
        q, _ = move.get_proposal({BR: np.zeros((1, n, 1, 3))}, np.random.default_rng(3))
        g = np.abs(q[BR][0, :, 0, 0])
        g0 = DE_GAMMA_SCALE / np.sqrt(2 * 3)
        self.assertAlmostEqual(np.mean(g), g0, delta=0.01 * g0)
        self.assertAlmostEqual(np.std(g), 0.1 * g0, delta=0.01 * g0)
        np.testing.assert_array_equal(move.last_kind[BR], KIND_SCALED)

    def test_periodic_differences_wrap_and_the_proposal_lands_in_the_period(self):
        per = PeriodicContainer({BR: {1: 2 * np.pi}})
        P = np.array([[0.0, 0.1], [0.0, 2 * np.pi - 0.1]])
        # the scaled jump (gamma = 2.38 / sqrt(4) exactly): only there does the
        # wrap of the DIFFERENCE matter (a gamma = 1 jump lands on the same
        # point mod 2 pi either way)
        move = WalkerDifferenceJumpMove(_Zero(), mode_frac=0.0, gamma_sigma=0.0, periodic=per)
        x = np.zeros((1, 400, 1, 2))
        x[..., 1] = 0.05
        move.set_partners(BR, np.ones((1, 400), dtype=bool), P)
        q, _ = move.get_proposal({BR: x}, np.random.default_rng(5))
        new = q[BR][0, :, 0, 1]
        step = DE_GAMMA_SCALE / 2.0 * 0.2
        # the wrapped difference is +-0.2, never +-(2 pi - 0.2)
        self.assertTrue(np.all(np.isclose(new, 0.05 + step) | np.isclose(new, 2 * np.pi + 0.05 - step)),
                        new[:8])
        self.assertTrue(np.any(np.isclose(new, 0.05 + step)) and np.any(np.isclose(new, 2 * np.pi + 0.05 - step)))
        self.assertTrue(np.all((new >= 0) & (new < 2 * np.pi)))
        # the fallback received the periodic spec too
        self.assertIs(move.fallback.periodic, per)

    def test_no_spec_means_the_fallback_everywhere_and_one_draw_call(self):
        calls = []

        class Rng:
            def randint(self, lo, hi=None, size=None):
                calls.append("randint")
                return np.zeros(size, dtype=int)

            def rand(self, *s):
                calls.append("rand")
                return np.zeros(s)

            def randn(self, *s):
                calls.append("randn")
                return np.zeros(s)

        move = WalkerDifferenceJumpMove(_Zero())
        x = self._coords()
        q, f = move.get_proposal({BR: x}, Rng())
        np.testing.assert_array_equal(q[BR], x)
        np.testing.assert_array_equal(f, 0.25)
        self.assertEqual(calls, ["randint", "randint", "rand", "randn"])
        np.testing.assert_array_equal(move.last_kind[BR], 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            WalkerDifferenceJumpMove(None)
        with self.assertRaises(ValueError):
            WalkerDifferenceJumpMove(_Zero(), mode_frac=1.5)
        move = WalkerDifferenceJumpMove(_Zero())
        with self.assertRaises(ValueError):
            move.set_partners(BR, np.ones((1, 1), bool), np.zeros((1, 3)))
        move.set_partners(BR, np.ones((2, 2), bool), np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            move.get_proposal({BR: self._coords(nt=3)}, np.random.default_rng(0))

    def test_the_stack_builder_and_its_log_label(self):
        from lisatools.globalfit.stock.erebor.common import make_default_inner_moves

        (m, w), = make_default_inner_moves("eigen", jump_mix=0.1, de_jump_mix=0.0)
        self.assertIsInstance(m, EigenAxisJumpMixMove)
        stack = make_default_inner_moves("eigen", jump_mix=0.1, de_jump_mix=0.2, de_jump_mode_frac=0.4)
        (eig, w0), (de, w1) = stack
        self.assertIsInstance(de, WalkerDifferenceJumpMove)
        self.assertIs(de.fallback, eig)
        self.assertEqual((w0, w1, de.mode_frac), (0.8, 0.2, 0.4))
        self.assertEqual(
            describe_inner_moves([eig, de], [w0, w1]),
            "EigenAxisJumpMixMove 0.800 (jump mix 0.10 x 3.0) | "
            "WalkerDifferenceJumpMove 0.200 (cross-walker, mode frac 0.40)")
        with self.assertRaises(ValueError):
            make_default_inner_moves("eigen", de_jump_mix=1.0)


# ---------------------------------------------------------------------------
# B3: the schedule and the partner pool
# ---------------------------------------------------------------------------

def _ensemble_state(nt=3, nw=4, nl=2, nd=3, seed=0, dead=()):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(nt, nw, nl, nd))
    inds = np.ones((nt, nw, nl), dtype=bool)
    for leaf in dead:
        inds[:, :, leaf] = False
    sub = SimpleNamespace(coords=coords, inds=inds, betas_all=np.tile(np.linspace(1, 0.3, nt), (nl, 1)),
                          tempered_initialized=True)
    return SimpleNamespace(sub_states={BR: sub})


def _de_move(nw, nt=3, nd=3, block_head=False, pool=False, adapt=False):
    eig = EigenAxisMove()
    de = WalkerDifferenceJumpMove(eig)
    move = _stub((np.eye(nd), np.ones(nd)), ndim=nd, ntemps=nt, nwalkers=nw,
                 stack=[(eig, 0.8), (de, 0.2)], adapt=adapt)
    move.eigen_adapt_pool = pool
    move.gf_stage_kind = "pe"
    if block_head:
        move.fanout = SimpleNamespace(single=False, layout=SimpleNamespace(replica_mode=False))
    return move, de


class ScheduleTest(unittest.TestCase):
    def test_the_parity_alternates_per_distinct_iteration(self):
        move, _ = _de_move(4)
        got = []
        for it in (0, 0, 1, 2, 2, 5, 9):
            move.gf_iteration = it
            got.append(move._de_next_parity())
        self.assertEqual(got, [1, 1, 0, 1, 1, 0, 1])
        # an even source cadence (iterations 0, 4, 8, ...) still alternates
        move2, _ = _de_move(4)
        seq = []
        for it in (0, 4, 8, 12):
            move2.gf_iteration = it
            seq.append(move2._de_next_parity())
        self.assertEqual(seq, [1, 0, 1, 0])
        # unstamped: every propose is a new one
        move3, _ = _de_move(4)
        self.assertEqual([move3._de_next_parity() for _ in range(4)], [1, 0, 1, 0])

    def test_single_process_active_groups_and_partners_alternate(self):
        with _env():
            move, de = _de_move(4)
            state = _ensemble_state()
            seen = []
            for it in range(4):
                move.gf_iteration = it
                move.fanout_prepare_state(None, state)
                active, P = move._de_leaf_spec(0)
                seen.append(tuple(np.flatnonzero(active)))
                partner_w = [w for w in range(4) if not active[w]]
                want = np.unique(state.sub_states[BR].coords[:, partner_w, 0].reshape(-1, 3), axis=0)
                np.testing.assert_array_equal(P, want)
                # an active walker's own (and its group's) states are never partners
                own = state.sub_states[BR].coords[:, active, 0].reshape(-1, 3)
                self.assertFalse(any(np.any(np.all(P == r, axis=1)) for r in own))
                # the payload is untouched in single process
                self.assertNotIn("de_pool", move.fanout_payload_extra())
        self.assertEqual(seen, [(0, 2), (1, 3), (0, 2), (1, 3)])

    def test_the_block_head_ships_the_pool_and_the_parity_and_a_rank_maps_it(self):
        with _env():
            head, _ = _de_move(4, block_head=True)
            state = _ensemble_state(dead=(1,))
            head.gf_iteration = 7
            head.fanout_prepare_state(None, state)
            extra = head.fanout_payload_extra()
            self.assertEqual(sorted(extra["de_pool"]), [0])
            self.assertEqual(extra["de_parity"], 1)
            np.testing.assert_array_equal(extra["de_pool"][0], state.sub_states[BR].coords[:, :, 0])
            self.assertNotIn("de_pool", head.fanout_payload_extra())   # shipped once
            for w0, want_active in ((0, True), (1, False), (2, True), (3, False)):
                rank, de = _de_move(1)
                rank._fanout_body = True
                rank.fanout = SimpleNamespace(rank=w0, layout=SimpleNamespace(block_of=lambda r: (r, r + 1)))
                rank.fanout_apply_extra(extra)
                spec = rank._de_leaf_spec(0)
                if not want_active:
                    self.assertIsNone(spec)
                    rank._install_de_split(de, 0, np.ones((3, 1), dtype=bool))
                    self.assertIsNone(de.partner_spec(BR))
                    continue
                active, P = spec
                self.assertTrue(active[0])
                want = np.unique(state.sub_states[BR].coords[:, [1, 3], 0].reshape(-1, 3), axis=0)
                np.testing.assert_array_equal(P, want)
                rank._install_de_split(de, 0, np.ones((3, 1), dtype=bool))
                act, PP = de.partner_spec(BR)
                np.testing.assert_array_equal(act, np.ones((3, 1), bool))
                np.testing.assert_array_equal(PP, want)

    def test_the_pooled_adaptation_sample_and_the_pool_share_arrays(self):
        with _env():
            head, _ = _de_move(4, block_head=True, pool=True, adapt=True)
            state = _ensemble_state()
            head.fanout_prepare_state(None, state)
            extra = head.fanout_payload_extra()
            for leaf in extra["de_pool"]:
                self.assertIs(extra["de_pool"][leaf], extra["eigen_pool"][leaf][0])
            # pickle memoizes the shared arrays: the jump's pool costs ~nothing extra
            both = len(pickle.dumps(extra))
            alone = len(pickle.dumps({k: v for k, v in extra.items() if k != "de_pool"}))
            self.assertLess(both - alone, 200)

    def test_off_is_silent_and_ships_nothing(self):
        with _env():
            move = _stub((np.eye(3), np.ones(3)), ndim=3, ntemps=3, nwalkers=4)
            move.fanout = SimpleNamespace(single=False, layout=SimpleNamespace(replica_mode=False))
            move.fanout_prepare_state(None, _ensemble_state())
            extra = move.fanout_payload_extra()
            self.assertNotIn("de_pool", extra)
            self.assertNotIn("de_parity", extra)
            self.assertIsNone(move._de_pool)

    def test_one_walker_has_no_partners(self):
        with _env():
            move, de = _de_move(1)
            move.fanout_prepare_state(None, _ensemble_state(nw=1))
            self.assertIsNone(move._de_leaf_spec(0))


# ---------------------------------------------------------------------------
# B3: exactness on a bimodal target, with paired negative controls
# ---------------------------------------------------------------------------

BI_W, BI_MU, BI_S = 0.7, 4.0, 0.5         # right-mode weight, mode centres +-4, width 0.5


def _bimodal_lnl(c):
    x, y = c[..., 0, 0], c[..., 0, 1]
    return np.logaddexp(np.log(1 - BI_W) - 0.5 * ((x + BI_MU) / BI_S) ** 2,
                        np.log(BI_W) - 0.5 * ((x - BI_MU) / BI_S) ** 2) - 0.5 * y ** 2


def _bimodal_run(control=None, betas=(1.0,), proposes=1500, repeats=10, seed=0):
    """4 walkers, the eigen move (sigma 0.5: it never crosses the 32-nat barrier)
    + the cross-walker jump 50/50, through the single-process seams. Returns the
    right-mode fraction of the cold rung (burn-in 10 %) and the cold switches.

    ``control``: ``"nosplit"`` -- every walker jumps with partners from EVERY
    walker's start-of-propose state (its own group, itself included);
    ``"live"`` -- the same from the CURRENT states. Both break the one-group
    rule the exactness argument needs."""
    betas = np.asarray(betas, dtype=float)
    nt, nw = betas.size, 4
    eig = EigenAxisMove(mode="axis")
    de = WalkerDifferenceJumpMove(eig, mode_frac=0.5)
    mv = _stub((np.eye(2), np.array([BI_S, 1.0])), ndim=2, ntemps=nt, nwalkers=nw,
               stack=[(eig, 0.5), (de, 0.5)], adapt=False, refresh=10 ** 9)
    mv.gf_stage_kind = "pe"
    rng = np.random.default_rng(seed)
    work = SimpleNamespace(coords=np.zeros((nt, nw, 1, 2)))
    work.coords[:, :2, 0, 0], work.coords[:, 2:, 0, 0] = BI_MU, -BI_MU
    state = SimpleNamespace(sub_states={BR: SimpleNamespace(
        coords=work.coords, inds=np.ones((nt, nw, 1), bool), betas_all=betas[None, :],
        tempered_initialized=True)})
    if control == "nosplit":
        mv._de_leaf_spec = lambda leaf: (np.ones(nw, bool), np.unique(mv._de_pool[0].reshape(-1, 2), axis=0))
    elif control == "live":
        mv._de_leaf_spec = lambda leaf: (np.ones(nw, bool), np.unique(work.coords[:, :, 0].reshape(-1, 2), axis=0))
    full = np.ones((nt, nw), bool)
    ll = _bimodal_lnl(work.coords)
    xs = []
    for _ in range(proposes):
        mv.fanout_prepare_state(None, state)
        mv.refresh_inner_move_tables(0, work, betas=betas)
        for _ in range(repeats):
            m = mv.moves[int(rng.choice(2, p=mv.move_weights))]
            mv._install_eigen_split_table(m, 0, full)
            mv._install_de_split(m, 0, full)
            q, _ = m.get_proposal({BR: work.coords}, rng)
            new = q[BR]
            lln = _bimodal_lnl(new)
            keep = betas[:, None] * (lln - ll) > np.log(rng.random((nt, nw)))
            work.coords[keep] = new[keep]
            ll = np.where(keep, lln, ll)
            for w in range(nw):
                for k in range(nt - 1, 0, -1):
                    if np.log(rng.random()) < (betas[k - 1] - betas[k]) * (ll[k, w] - ll[k - 1, w]):
                        work.coords[[k - 1, k], w] = work.coords[[k, k - 1], w]
                        ll[[k - 1, k], w] = ll[[k, k - 1], w]
            xs.append(work.coords[0, :, 0, 0].copy())
    xs = np.array(xs)
    right = xs[len(xs) // 10:] > 0
    return float(right.mean()), int(np.sum(np.diff(xs > 0, axis=0)))


class JumpExactnessTest(unittest.TestCase):
    """The jump with the one-group rule leaves the product target invariant.

    With ONE rung the only way between the modes is the jump, and the jump needs
    a partner pair that straddles them: a configuration with every walker in one
    mode can then never be entered (the last walker of the other mode sees only
    same-mode partners). The chain samples the exact product target RESTRICTED
    to the reachable configurations, whose right-mode fraction for weight 0.7
    and 4 walkers is (2.8 - 4 x 0.7^4) / (1 - 0.7^4 - 0.3^4) / 4 = 0.612 -- a
    sharp test of exactness. With a ladder (production: 8 rungs) the hot rungs
    straddle the modes and the cold fraction is the target's 0.7. Breaking the
    one-group rule collapses the ensemble onto one mode.
    """

    WANT_RESTRICTED = (4 * BI_W - 4 * BI_W ** 4) / (1 - BI_W ** 4 - (1 - BI_W) ** 4) / 4

    def test_one_rung_samples_the_restricted_product_target(self):
        with _env():
            frac, sw = _bimodal_run(seed=0)
        self.assertAlmostEqual(self.WANT_RESTRICTED, 0.6117, places=3)
        self.assertAlmostEqual(frac, self.WANT_RESTRICTED, delta=0.04)
        self.assertGreater(sw, 500)                   # the jump really mixes the modes

    def test_with_a_ladder_the_cold_mode_weight_is_the_target(self):
        with _env():
            frac, sw = _bimodal_run(betas=(1.0, 0.5, 0.25, 0.12), proposes=1200, seed=1)
        self.assertAlmostEqual(frac, BI_W, delta=0.03)
        self.assertGreater(sw, 2000)

    def test_breaking_the_one_group_rule_collapses_the_ensemble(self):
        with _env():
            for control in ("nosplit", "live"):
                with self.subTest(control=control):
                    frac, _ = _bimodal_run(control=control, proposes=600, seed=0)
                    self.assertGreater(abs(frac - self.WANT_RESTRICTED), 0.25, frac)


# ---------------------------------------------------------------------------
# B1: the tallies
# ---------------------------------------------------------------------------

def _visit_loop(move, nt=3, nw=2, nd=4, visits=3, repeats=20, seed=0, lnl=None, refresh=None,
                betas=None, start=None):
    """The move's own per-visit order on a Gaussian: refresh, per repeat (install,
    draw, tempered MH, tally), end-of-visit note. Returns the coords."""
    rng = np.random.default_rng(seed)
    betas = np.linspace(1.0, 0.4, nt) if betas is None else betas
    work = SimpleNamespace(coords=(np.zeros((nt, nw, 1, nd)) if start is None else start.copy()))
    lnl = lnl or (lambda c: -0.5 * np.sum(c[..., 0, :] ** 2, axis=-1))
    full = np.ones((nt, nw), dtype=bool)
    tinds = np.repeat(np.arange(nt)[:, None], nw, axis=1)[full]
    ll = lnl(work.coords)
    for _ in range(visits):
        move.refresh_inner_move_tables(0, work, betas=betas)
        for _ in range(repeats):
            m = move.moves[int(rng.choice(len(move.moves), p=move.move_weights))] if len(move.moves) > 1 else move.moves[0]
            move._install_eigen_split_table(m, 0, full)
            move._install_de_split(m, 0, full)
            old = work.coords[:, :, 0, :].copy()
            q, _ = m.get_proposal({BR: work.coords}, rng)
            new = q[BR]
            ll_new = lnl(new)
            keep = betas[:, None] * (ll_new - ll) > np.log(rng.random((nt, nw)))
            move._note_inner_outcome(m, 0, tinds, keep, old, new[:, :, 0, :], full)
            work.coords[keep] = new[keep]
            ll = np.where(keep, ll_new, ll)
        move._note_eigen_adapt_sample(0, work, betas)
    return work


class TallyTest(unittest.TestCase):
    ND = 4

    def _move(self, inner, **kw):
        return _stub((np.eye(self.ND), np.full(self.ND, 0.5)), ndim=self.ND, ntemps=3, nwalkers=2,
                     stack=[(inner, 1.0)], log=True, adapt=False, refresh=1000, **kw)

    def test_recorded_and_derived_picks_give_the_same_counts(self):
        with _env():
            rec = self._move(EigenAxisJumpMixMove(mode="axis", jump_mix=1e-12, jump_mix_factor=1.0))
            _visit_loop(rec, visits=2, repeats=30)
            der = self._move(EigenAxisMove(mode="axis"))
            _visit_loop(der, visits=2, repeats=30)
        a, b = rec._eigen_axis_tally[0], der._eigen_axis_tally[0]
        # same stream (the mix draws one extra uniform per row, after the step,
        # so compare totals and shape, not row for row)
        self.assertEqual(a.shape, (2, 2, self.ND))
        self.assertEqual(a[:, 0].sum(), 2 * 30 * 3 * 2)
        self.assertEqual(b[:, 0].sum(), 2 * 30 * 3 * 2)
        self.assertEqual(a[0, 0].sum(), 2 * 30 * 2)            # the cold rung: nw rows / repeat
        np.testing.assert_array_less(a[:, 1], a[:, 0] + 1e-9)
        # derived picks == the recorded ones on the same proposals, on a table
        # whose axes are unit but NOT orthogonal (as the production tables are:
        # orthonormal in the prior-box-whitened basis only)
        m = EigenAxisJumpMixMove(mode="axis", jump_mix=1e-12, jump_mix_factor=1.0)
        Aw = np.diag([1.0, 30.0, 0.2, 5.0]) @ A._rot(self.ND, 3)
        Aw /= np.linalg.norm(Aw, axis=0, keepdims=True)
        self.assertGreater(np.abs(Aw.T @ Aw - np.eye(self.ND)).max(), 0.1)
        mv = _stub((Aw, np.full(self.ND, 0.5)), ndim=self.ND, ntemps=3, nwalkers=2,
                   stack=[(m, 1.0)], log=True, adapt=False, refresh=1000)
        mv.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((3, 2, 1, self.ND))))
        x = np.random.default_rng(2).normal(size=(3, 2, 1, self.ND))
        q, _ = m.get_proposal({BR: x}, np.random.default_rng(4))
        derived = mv._derive_axis_picks(0, x[:, :, 0], q[BR][:, :, 0], np.ones((3, 2), bool))
        np.testing.assert_array_equal(derived, m.last_pick[BR])

    def test_a_derived_pick_unwraps_a_periodic_crossing(self):
        """A step along axis 1 that carries the periodic column across 2 pi:
        the raw displacement is off by 2 pi and would be read as axis 0 on
        these non-orthogonal axes; wrapped, it is axis 1."""
        axes = np.array([[0.6, 0.8], [0.8, 0.6]])
        mv = _stub((axes, np.full(2, 0.2)), ndim=2, ntemps=1, nwalkers=1,
                   stack=[(EigenAxisMove(mode="axis"), 1.0)], log=True, adapt=False,
                   periodic={BR: {1: 2 * np.pi}})
        mv.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((1, 1, 1, 2))))
        old = np.array([[[0.5, 2 * np.pi - 0.05]]])
        new = old + 0.2 * axes[:, 1]
        new[..., 1] = np.mod(new[..., 1], 2 * np.pi)
        got = mv._derive_axis_picks(0, old, new, np.ones((1, 1), bool))
        np.testing.assert_array_equal(got, [1])

    def test_jump_mix_rows_are_kept_out_of_the_axis_counts(self):
        with _env():
            mv = self._move(EigenAxisJumpMixMove(mode="axis", jump_mix=0.5, jump_mix_factor=3.0))
            _visit_loop(mv, visits=2, repeats=40)
        t, j = mv._eigen_axis_tally[0], mv._eigen_jump_tally[0]
        total = 2 * 40 * 3 * 2
        self.assertEqual(t[:, 0].sum() + j[:, 0].sum(), total)
        self.assertGreater(j[:, 0].sum(), 0.35 * total)
        self.assertLess(j[:, 0].sum(), 0.65 * total)

    def test_a_too_wide_axis_shows_low_acceptance(self):
        sig = np.array([0.5, 0.5, 0.5, 30.0])                  # axis 3 = 30 x the width 1
        with _env():
            mv = _stub((np.eye(self.ND), sig), ndim=self.ND, ntemps=3, nwalkers=2,
                       stack=[(EigenAxisMove(mode="axis"), 1.0)], log=True, adapt=False, refresh=1000)
            _visit_loop(mv, visits=4, repeats=40)
        t = mv._eigen_axis_tally[0]
        acc = t[0, 1] / t[0, 0]
        self.assertLess(acc[3], 0.1)
        self.assertTrue(np.all(acc[:3] > 0.5), acc)

    def test_the_refresh_logs_the_window_and_resets_it(self):
        with _env():
            mv = _stub((np.eye(self.ND), np.full(self.ND, 0.5)), ndim=self.ND, ntemps=3, nwalkers=2,
                       stack=[(EigenAxisJumpMixMove(mode="axis", jump_mix=0.2, jump_mix_factor=3.0), 1.0)],
                       log=True, adapt=False, refresh=2)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                _visit_loop(mv, visits=3, repeats=10)
        lines = [l for l in cm.output if "axis acceptance" in l]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertRegex(lines[0], r"\[EMRI_EIGEN\] leaf 0 \(walkers 0-1\): axis acceptance over the "
                                   r"last 2 visits, cold (\d\.\d\d ){3}\d\.\d\d \(n \d+-\d+ per axis\), "
                                   r"hot (\d\.\d\d ){3}\d\.\d\d; jump-mix draws cold \d+/\d+ hot \d+/\d+")
        # the visit after the refresh started a fresh window
        self.assertLessEqual(mv._eigen_axis_tally[0][:, 0].sum(), 10 * 3 * 2)

    def test_off_tallies_nothing_and_logs_nothing(self):
        with _env():
            mv = _stub((np.eye(self.ND), np.full(self.ND, 0.5)), ndim=self.ND, ntemps=3, nwalkers=2,
                       stack=[(EigenAxisMove(mode="axis"), 1.0)], adapt=False, refresh=2)
            with self.assertNoLogs(LOGGER, level="INFO"):
                _visit_loop(mv, visits=3, repeats=10)
        self.assertIsNone(getattr(mv, "_eigen_axis_tally", None))

    def test_the_adaptation_line_names_the_widened_axes_acceptance(self):
        with _env():
            mv = _stub((np.eye(self.ND), np.full(self.ND, 0.1)), ndim=self.ND, ntemps=3, nwalkers=2,
                       stack=[(EigenAxisMove(mode="axis"), 1.0)], log=True, adapt=True,
                       min_visits=1, refresh=2)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                _visit_loop(mv, visits=5, repeats=10)
        adapt = [l for l in cm.output if "step scale adapted" in l]
        self.assertTrue(adapt, cm.output)
        self.assertRegex(adapt[-1], r"cold acceptance of the widened axes( \d:(\d\.\d\d|-))+$")


# ---------------------------------------------------------------------------
# B2: the controller
# ---------------------------------------------------------------------------

class ControllerTest(unittest.TestCase):
    ND = 4

    def _move(self, sig, adapt=False, refresh=2, **kw):
        return _stub((np.eye(self.ND), np.asarray(sig, dtype=float)), ndim=self.ND, ntemps=3,
                     nwalkers=2, stack=[(EigenAxisMove(mode="axis"), 1.0)], tune=True,
                     adapt=adapt, refresh=refresh, **kw)

    def test_widened_but_rejected_axis_is_shrunk_back_and_the_others_stay(self):
        """The 9mo situation: the ring buffer carries a between-island spread
        on axis 3 (x30 the local width), so the adaptation widens it x30 and
        it is never accepted; the controller brings it back into the band."""
        with _env():
            mv = self._move(np.full(self.ND, 0.5), adapt=True, min_visits=1, visits=1000)
            X = np.random.default_rng(11).normal(size=(200, 3, 2, self.ND))
            X[..., 3] *= 60.0                       # adapted = the x100 factor cap, 50
            A._feed(mv, X, np.ones(3))
            with self.assertLogs(LOGGER, level="INFO") as cm:
                work = _visit_loop(mv, visits=12, repeats=40, betas=np.ones(3))
        m = mv._eigen_axis_tune[0]
        self.assertLessEqual(m[3], 0.25 + 1e-12, m)
        np.testing.assert_array_equal(m[:3], 1.0)
        floor = mv._eigen_fisher_sigmas[0]
        tuned = mv._eigen_tables[0][1]
        self.assertTrue(np.all(tuned >= floor - 1e-12))
        self.assertGreater(tuned[3], 2.0)            # 50 / 2^k, still above the true width 1
        lines = [l for l in cm.output if "acceptance-tuned step" in l]
        self.assertTrue(lines, cm.output)
        self.assertRegex(lines[0], r"\(walkers 0-1\): acceptance-tuned step: shrunk axes 3 "
                                   r"\(cold acc 0\.\d\d-0\.\d\d\), widened none")
        self.assertTrue(np.all(np.isfinite(work.coords)))

    def test_the_rule_floor_ceiling_and_carry_over(self):
        mv = self._move(np.full(self.ND, 1.0))
        mv.eigen_axis_acc_min, mv.eigen_axis_acc_max, mv.eigen_axis_acc_factor = 0.1, 0.5, 2.0
        n = EIGEN_AXIS_ACC_MIN_PROPOSALS
        t = np.zeros((2, 2, self.ND))
        t[0, 0] = [n, n, n, n - 1]                 # axis 3 below the minimum count
        t[0, 1] = [0, 0.3 * n, 0.9 * n, 0]         # 0.0, 0.3, 0.9, (0)
        mv._eigen_fisher_sigmas = {0: np.full(self.ND, 0.3)}
        with _env():
            axes, s = mv._eigen_axis_tune_table(0, (np.eye(self.ND), np.full(self.ND, 1.0)),
                                                True, {"axis": t})
        # adapted: ceiling = the adapted 1.0, floor 0.3 -> axis 0 halved, axis 2 cannot widen
        np.testing.assert_allclose(s, [0.5, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(mv._eigen_axis_tune[0], [0.5, 1.0, 1.0, 1.0])
        # axis 3 carried its 7 proposals into the next window
        np.testing.assert_allclose(mv._eigen_axis_ctrl[0][0], [0, 0, 0, n - 1])
        # twice more on axis 0: the floor 0.3 holds (multiplier clipped at 0.3)
        for _ in range(2):
            with _env():
                axes, s = mv._eigen_axis_tune_table(0, (np.eye(self.ND), np.full(self.ND, 1.0)),
                                                    True, {"axis": t})
        self.assertAlmostEqual(s[0], 0.3)
        self.assertAlmostEqual(mv._eigen_axis_tune[0][0], 0.3)
        # adaptation OFF: the ceiling is the box cap, so a high-acceptance axis widens
        mv2 = self._move(np.full(self.ND, 1.0))
        with _env():
            _, s2 = mv2._eigen_axis_tune_table(0, (np.eye(self.ND), np.full(self.ND, 1.0)),
                                               False, {"axis": t})
        np.testing.assert_allclose(s2, [1.0, 1.0, 2.0, 1.0])     # floor = the table here

    def test_off_returns_the_table_untouched_and_writes_no_key(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = os.path.join(tmp.name, "run.h5")
        with _env(persist=True):
            mv = self._move(np.full(self.ND, 0.5), store=store)
            mv.eigen_axis_acc_tune = False
            _visit_loop(mv, visits=3, repeats=5)
            np.testing.assert_array_equal(mv._eigen_tables[0][1], np.full(self.ND, 0.5))
            entry = eigen_table_persist.load_sidecar(eigen_table_persist.sidecar_path(store))["entries"]["emri:0"]
        self.assertNotIn("axis_tune", entry)

    def test_the_sidecar_round_trip_restores_the_multiplier_without_applying_it_twice(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = os.path.join(tmp.name, "run.h5")
        axes, fisher = np.eye(self.ND), np.full(self.ND, 0.1)
        with _env(persist=True):
            mv = self._move(fisher, adapt=True, store=store, min_visits=1)
            mv._eigen_axis_tune = {0: np.array([1.0, 0.5, 0.25, 1.0])}
            mv._eigen_tables = {0: (axes, np.array([1.0, 0.5, 0.25, 1.0]))}
            mv._eigen_visit_count = {0: 7}
            mv._eigen_fisher_sigmas = {0: fisher}
            mv._persist_eigen_table(0)
            entry = eigen_table_persist.load_sidecar(eigen_table_persist.sidecar_path(store))["entries"]["emri:0"]
            np.testing.assert_allclose(entry["axis_tune"], [1.0, 0.5, 0.25, 1.0])
            # a fresh process adopts table + multiplier; the carried scale is the
            # PRE-tune width (1.0 everywhere)
            new = self._move(fisher, adapt=True, store=store, min_visits=50)
            new._eigen_tables, new._eigen_visit_count = {}, {}
            self.assertTrue(new._adopt_persisted_eigen_table(0))
            np.testing.assert_allclose(new._eigen_axis_tune[0], [1.0, 0.5, 0.25, 1.0])
            np.testing.assert_allclose(new._eigen_carried[0][1], 1.0)
            # the next refresh (buffer short): carried 1.0 x multiplier = the same tuned step
            new._eigen_visit_count[0] = 10
            new.eigen_refresh_every = 10
            new.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((3, 2, 1, self.ND))))
            np.testing.assert_allclose(new._eigen_tables[0][1], [1.0, 0.5, 0.25, 1.0])

    def test_inconsistent_knobs_fall_back_to_the_defaults(self):
        mv = self._move(np.ones(self.ND))
        mv.eigen_axis_acc_min, mv.eigen_axis_acc_max = 0.6, 0.2
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(mv._eigen_axis_acc_bounds(), (0.1, 0.5, 2.0))


# ---------------------------------------------------------------------------
# the harness: a CURVED ridge with a secondary maximum, 4 walkers x 8 rungs
# ---------------------------------------------------------------------------

#: the comb in the table's axis coordinates u (x = R u): island A at u = 0, island
#: B at (u0, u4) = (H, D); the islands are joined by a curved bridge u0 = h(u4)
#: (h' = 0 at both ends, so the islands are translated copies, as the EMRI
#: islands are) whose lnL floor sits ``DEPTH`` nats below the peaks (7.5 nats
#: at the T_max 6 rung); the table is the information matrix at island A:
#: the right axes, sigma 1 everywhere (the island is 1 wide on axis 0, 2 on axis 4)
COMB_D, COMB_H, COMB_L, COMB_DEPTH = 40.0, 30.0, 2.0, 45.0


def _comb_u_lnl(u):
    t = u[..., 4]
    tc = np.clip(t, 0.0, COMB_D)
    h = 0.5 * COMB_H * (1.0 - np.cos(np.pi * tc / COMB_D))
    g = np.logaddexp(-0.5 * (t / COMB_L) ** 2, -0.5 * ((t - COMB_D) / COMB_L) ** 2)
    g = np.where((t > 0.0) & (t < COMB_D), np.maximum(g, -COMB_DEPTH), g)
    return -0.5 * ((u[..., 0] - h) ** 2 + np.sum(u[..., 1:4] ** 2, axis=-1)) + g


def _comb_ridge_run(*, adapt=True, pool=True, temper=True, jump_mix=0.1, tune=False, de=0.0,
                    seed=0, visits=200, repeats=25, refresh=5):
    """The 9mo EMRI situation on a synthetic comb, through the move's real hooks.

    4 walkers x 8 rungs (T_max 6, ``A.BETAS8``), each walker its own one-walker
    "rank" under a head move that takes the pooled adaptation sample and the
    jump's partner pool and builds the payload the way
    ``WalkerFanoutMixin.propose`` does (``fanout_prepare_state`` ->
    ``fanout_payload_extra`` -> ``fanout_apply_extra``; the head's
    ``gf_iteration`` is the visit number). Walkers 0 and 1 start with every
    rung in island B, walkers 2 and 3 in island A. Per visit and walker: the
    table refresh (adaptation, controller, logs), then ``repeats`` x (the
    stack draw, the eigen and jump split installs, the proposal, the tempered
    MH accept, the tallies, adjacent-rung swaps top down), then the
    end-of-visit buffer note -- ``propose_local``'s own order.

    Returns a dict: per walker the cold island label per repeat, the number of
    cold island switches, the fraction of cold repeats in island B, the first
    visit with a switch; each rank's last cold table; the summed cold tallies.
    """
    nd, nt, nw = 5, 8, 4
    R = A._rot(nd, 21)
    betas = A.BETAS8
    box = 200.0
    rng = np.random.default_rng(seed)

    def mk(nwalk):
        eig = EigenAxisJumpMixMove(mode="axis", jump_mix=jump_mix, jump_mix_factor=3.0)
        stack = [(eig, 1.0 - de)] + ([(WalkerDifferenceJumpMove(eig, mode_frac=0.5), de)] if de else [])
        m = _stub((R, np.ones(nd)), ndim=nd, ntemps=nt, nwalkers=nwalk, adapt=adapt,
                  refresh=refresh, box=box, stack=stack, tune=tune, log=True)
        m.eigen_adapt_pool = pool
        m.eigen_temper_sigmas = temper
        m.gf_stage_kind = "pe"
        return m

    head = mk(nw)
    head.fanout = SimpleNamespace(single=False, layout=SimpleNamespace(replica_mode=False))
    ranks = []
    for w in range(nw):
        m = mk(1)
        m._fanout_body = True
        m.fanout = SimpleNamespace(single=False, rank=w, layout=SimpleNamespace(
            replica_mode=False, block_of=lambda r: (r, r + 1)))
        ranks.append(m)

    u = rng.normal(size=(nt, nw, nd)) / np.sqrt(betas)[:, None, None]
    u[:, :, 4] *= COMB_L
    u[:, :2, 0] += COMB_H
    u[:, :2, 4] += COMB_D
    sub = SimpleNamespace(coords=np.einsum("ij,twj->twi", R, u)[:, :, None, :].copy(),
                          inds=np.ones((nt, nw, 1), dtype=bool), betas_all=betas[None, :].copy(),
                          tempered_initialized=True)
    state = SimpleNamespace(sub_states={BR: sub})

    def lnl(x):
        return _comb_u_lnl(x @ R)

    full = np.ones((nt, 1), dtype=bool)
    tinds = np.arange(nt)
    island = np.empty((visits * repeats, nw), dtype=int)
    for v in range(visits):
        head.gf_iteration = v
        head.fanout_prepare_state(None, state)
        extra = head.fanout_payload_extra()
        for w, m in enumerate(ranks):
            m.fanout_apply_extra(extra)
            work = SimpleNamespace(coords=np.array(sub.coords[:, w:w + 1], copy=True))
            m.refresh_inner_move_tables(0, work, betas=betas)
            ll = lnl(work.coords[:, :, 0, :])
            for r in range(repeats):
                mh = m.moves[int(rng.choice(len(m.moves), p=m.move_weights))]
                m._install_eigen_split_table(mh, 0, full)
                m._install_de_split(mh, 0, full)
                old = work.coords[:, :, 0, :].copy()
                q, _ = mh.get_proposal({BR: work.coords}, rng)
                new = q[BR]
                inbox = np.all(np.abs(new[..., 0, :]) < box, axis=-1)
                ll_new = np.where(inbox, lnl(new[:, :, 0, :]), -np.inf)
                keep = betas[:, None] * (ll_new - ll) > np.log(rng.random((nt, 1)))
                m._note_inner_outcome(mh, 0, tinds, keep, old, new[:, :, 0, :], full)
                work.coords[keep] = new[keep]
                ll = np.where(keep, ll_new, ll)
                for k in range(nt - 1, 0, -1):
                    if np.log(rng.random()) < (betas[k - 1] - betas[k]) * (ll[k, 0] - ll[k - 1, 0]):
                        work.coords[[k - 1, k]] = work.coords[[k, k - 1]]
                        ll[[k - 1, k]] = ll[[k, k - 1]]
                island[v * repeats + r, w] = int((work.coords[0, 0, 0, :] @ R)[4] > 0.5 * COMB_D)
            m._note_eigen_adapt_sample(0, work, betas)
            sub.coords[:, w:w + 1] = work.coords
    switches = np.sum(np.diff(island, axis=0) != 0, axis=0)
    per_visit = island[repeats - 1::repeats]
    switches_visit = np.sum(np.diff(per_visit, axis=0) != 0, axis=0)
    first = np.array([
        (int(np.flatnonzero(np.diff(island[:, w]) != 0)[0]) // repeats
         if np.any(np.diff(island[:, w]) != 0) else np.inf) for w in range(nw)])
    tally = sum((getattr(m, "_eigen_axis_tally", {}) or {}).get(0, np.zeros((2, 2, nd))) for m in ranks)
    return dict(island=island, switches=switches, switches_visit=switches_visit,
                frac_b=island.mean(axis=0), first=first,
                tables=[m._eigen_tables[0] for m in ranks],
                tune=[(getattr(m, "_eigen_axis_tune", None) or {}).get(0) for m in ranks],
                tally=tally)


class CombRidgeHarnessTest(unittest.TestCase):
    """The curved ridge with a secondary maximum: today's setup vs + B2 vs + B2 + B3.

    The numbers are printed with ``EMRI_AUDIT_HARNESS_VERBOSE=1``.
    """

    VISITS = 200
    CONFIGS = {
        "today (adapt+pool+temper+jump mix)": dict(),
        "+ B2 (acceptance controller)": dict(tune=True),
        "+ B2 + B3 (cross-walker jump 0.2)": dict(tune=True, de=0.2),
        "+ B3 alone (no controller)": dict(de=0.2),
    }

    def test_the_cross_walker_jump_switches_islands_the_others_do_not(self):
        res = {}
        with _env():
            for name, kw in self.CONFIGS.items():
                res[name] = _comb_ridge_run(seed=5, visits=self.VISITS, **kw)
        if os.environ.get("EMRI_AUDIT_HARNESS_VERBOSE"):
            for name, r in res.items():
                sig = np.array([t[1] for t in r["tables"]])
                t = r["tally"]
                with np.errstate(invalid="ignore", divide="ignore"):
                    acc = t[0, 1] / t[0, 0]
                print(f"{name:36s} cold island switches per walker: per repeat {r['switches']}, "
                      f"visit to visit {r['switches_visit']} (of {self.VISITS - 1}); "
                      f"fraction in B {np.round(r['frac_b'], 2)} (target 0.5); first switch at "
                      f"visit {r['first']}; cold sigma axes 0/4 per rank "
                      f"{np.round(sig[:, [0, 4]], 1).tolist()} (island widths 1 / 2); "
                      f"cold axis acceptance {np.round(acc, 2)}")
        today = res["today (adapt+pool+temper+jump mix)"]
        b2 = res["+ B2 (acceptance controller)"]
        b3 = res["+ B2 + B3 (cross-walker jump 0.2)"]
        # today: the widened axes 0 / 4 (between-island offsets) are dead
        sig_today = np.array([t[1] for t in today["tables"]])
        self.assertTrue(np.all(sig_today[:, 0] > 5.0), sig_today)
        # B2 brings them back near the islands' own widths (1 and 2)
        sig_b2 = np.array([t[1] for t in b2["tables"]])
        self.assertTrue(np.all(sig_b2[:, 0] < 0.5 * sig_today[:, 0]), (sig_b2, sig_today))
        # B3: every walker switches islands, many times, and visits both islands
        self.assertTrue(np.all(b3["switches"] >= 4), b3["switches"])
        self.assertTrue(np.all(np.isfinite(b3["first"])))
        self.assertGreater(b3["switches"].sum(), 5 * max(1, today["switches"].sum()))
        self.assertGreater(b3["switches"].sum(), 5 * max(1, b2["switches"].sum()))
        self.assertTrue(np.all((b3["frac_b"] > 0.1) & (b3["frac_b"] < 0.9)), b3["frac_b"])


# ---------------------------------------------------------------------------
# the REAL EMRI move: single process, the 4-rank walker-block fan-out, golden
# ---------------------------------------------------------------------------

try:
    from tests import test_eigen_pool_temper as P
    from tests import test_emri_direct_fanout as E
except ImportError:  # pragma: no cover - run from tests/
    import test_eigen_pool_temper as P
    import test_emri_direct_fanout as E


def _with_stack(builder, jump_first=True, mix=0.0):
    """``E._make_move`` with the cross-walker jump in the stack (FIRST, so the
    row-local stream's ``choice`` -- always index 0 -- draws it every repeat)."""

    def make(acs, nwalkers, **kw):
        move = builder(acs, nwalkers, **kw)
        eig = (EigenAxisJumpMixMove(mode="axis", jump_mix=mix, jump_mix_factor=3.0)
               if mix else move.moves[0])
        de = WalkerDifferenceJumpMove(eig, mode_frac=0.5)
        rows = move._spy.setdefault("de_rows", [])
        orig = de.get_proposal

        def spy(*a, **k):
            out = orig(*a, **k)
            rows.append(int(np.count_nonzero(de.last_kind[BR])))
            return out

        de.get_proposal = spy
        move.moves = [de, eig] if jump_first else [eig, de]
        move.move_weights = np.array([0.5, 0.5])
        return move

    return make


class RealMoveTest(E._Case):
    N_PROPOSES = 3

    def test_the_block_fanout_equals_single_process_with_the_jump_in_the_stack(self):
        """The partners and the parity reach every rank exactly as the
        single-process body sees them: with a row-local stream the 4-rank
        fan-out reproduces the single-process states row for row."""
        kw = dict(scope="per_walker", adaptive=False, refresh=10, rng=E.H._RowLocalRandom,
                  permute_every=0)
        with mock.patch.object(E, "_make_move", _with_stack(E._make_move)), \
                mock.patch.dict(os.environ, {"EMRI_EIGEN_AXIS_ACC_LOG": "1"}):
            with E.H._row_local_numpy():
                ref = self._single(4, **kw)
                out = self._world(4, 4, **kw)
        computes = self._assert_match(out, ref)
        # the jump fired: 2 active walkers x 2 rungs per leaf visit single-process,
        # and on exactly the ranks of the active group per propose
        self.assertEqual(sum(ref["snap"]["spy"]["de_rows"]), self.N_PROPOSES * E.H.NLEAVES
                         * E.H.NUM_REPEATS * 2 * E.H.NTEMPS)
        per_rank = {r: sum(v["snap"]["spy"]["de_rows"]) for r, v in computes.items()}
        self.assertEqual(sum(per_rank.values()), sum(ref["snap"]["spy"]["de_rows"]))
        # parity 1, 0, 1 over the three proposes: walkers 0/2 jump twice, 1/3 once
        unit = E.H.NLEAVES * E.H.NUM_REPEATS * E.H.NTEMPS
        self.assertEqual(per_rank, {0: 2 * unit, 1: unit, 2: 2 * unit, 3: unit})

    def test_single_process_with_every_new_knob_on_runs_and_logs(self):
        kw = dict(scope="walker_max", adaptive=True, refresh=1, permute_every=1,
                  rng=lambda: np.random.RandomState(E.H.SEED))
        env = {"EMRI_EIGEN_AXIS_ACC_TUNE": "1", "EMRI_EIGEN_AXIS_ADAPT": "1",
               "EMRI_EIGEN_ADAPT_MIN_VISITS": "1", "EMRI_EIGEN_TEMPER_SIGMAS": "1"}
        with mock.patch.object(E, "_make_move", _with_stack(E._make_move, jump_first=False, mix=0.1)), \
                mock.patch.dict(os.environ, env), self.assertLogs(LOGGER, level="INFO") as cm:
            res = self._single(4, **kw)
        last = res["states"][-1].sub_states["emri"]
        self.assertTrue(np.all(np.isfinite(last.coords)))
        self.assertGreater(int(last.in_model_accepted.sum()), 0)
        text = "\n".join(cm.output)
        self.assertRegex(text, r"\[EMRI_EIGEN\] leaf \d \(walkers 0-3\): axis acceptance over the last 1 visits")
        self.assertRegex(text, r"\[EMRI_DE\] leaf \d \(walkers 0-3\): cross-walker jumps over the last 1 "
                               r"visits \(active on 1 of 1\): mode cold \d+/\d+")
        self.assertIn("step scale adapted", text)


class TelemetryGoldenTest(unittest.TestCase):
    """B1 is telemetry only: with ``EMRI_EIGEN_AXIS_ACC_LOG=1`` the real EMRI move's
    single-process proposal stream is still bit-identical to HEAD aaace7ab's golden."""

    def test_the_log_on_stream_is_the_golden_off_stream(self):
        ref = np.load(P.GOLDEN)
        scen = dict(P.GOLDEN_SCENARIOS)
        single, nw, env, kw = scen["single_off"]
        scen["single_log"] = (single, nw, dict(env, EMRI_EIGEN_AXIS_ACC_LOG="1"), kw)
        with mock.patch.object(P, "GOLDEN_SCENARIOS", scen), _env():
            got = P.golden_stream("single_log")
        want = {k.split("/", 1)[1]: ref[k] for k in ref.files if k.startswith("single_off/")}
        self.assertEqual(sorted(got), sorted(want))
        for key in want:
            np.testing.assert_array_equal(got[key], want[key], err_msg=key)


# ---------------------------------------------------------------------------
# knobs and the stock plumbing
# ---------------------------------------------------------------------------

class KnobTest(unittest.TestCase):
    def test_move_knobs_resolve_kwarg_then_env_then_default(self):
        mv = _stub((np.eye(2), np.ones(2)), ndim=2, ntemps=2, nwalkers=1)
        with _env():
            self.assertFalse(mv._eigen_axis_acc_log())
            self.assertFalse(mv._eigen_axis_acc_tune())
            self.assertEqual(mv._eigen_axis_acc_bounds(), (0.1, 0.5, 2.0))
        with _env(EMRI_EIGEN_AXIS_ACC_LOG="1", EMRI_EIGEN_AXIS_ACC_TUNE="yes",
                  EMRI_EIGEN_AXIS_ACC_MIN="0.05", EMRI_EIGEN_AXIS_ACC_MAX="0.6",
                  EMRI_EIGEN_AXIS_ACC_FACTOR="3"):
            self.assertTrue(mv._eigen_axis_acc_log())
            self.assertTrue(mv._eigen_axis_acc_tune())
            self.assertEqual(mv._eigen_axis_acc_bounds(), (0.05, 0.6, 3.0))
            mv.eigen_axis_acc_tune = False
            mv.eigen_axis_acc_factor = 1.5
            self.assertFalse(mv._eigen_axis_acc_tune())
            self.assertEqual(mv._eigen_axis_acc_bounds()[2], 1.5)

    def test_settings_fields_default_off_and_read_the_env(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        for cls, b in ((EMRISettings, "EMRI"), (MBHSettings, "MBH"), (SOBBHSettings, "SOBBH")):
            with self.subTest(branch=b), _env():
                s = cls()
                self.assertEqual((s.eigen_axis_acc_log, s.eigen_axis_acc_tune, s.de_jump_mix),
                                 (False, False, 0.0))
                self.assertEqual((s.eigen_axis_acc_min, s.eigen_axis_acc_max,
                                  s.eigen_axis_acc_factor, s.de_jump_mode_frac), (0.1, 0.5, 2.0, 0.5))
            with self.subTest(branch=b, env=True), _env(**{
                    f"{b}_EIGEN_AXIS_ACC_TUNE": "1", f"{b}_DE_JUMP_MIX": "0.2",
                    f"{b}_DE_JUMP_MODE_FRAC": "0.3"}):
                s = cls()
                self.assertTrue(s.eigen_axis_acc_tune)
                self.assertEqual((s.de_jump_mix, s.de_jump_mode_frac), (0.2, 0.3))
                from lisatools.globalfit.stock.erebor.common import inner_move_kwargs

                kw = inner_move_kwargs(s)
                self.assertEqual((kw["de_jump_mix"], kw["de_jump_mode_frac"]), (0.2, 0.3))

    def test_the_stock_build_stamps_the_fields(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SOURCE_MOVE_SETTINGS_ATTRS,
            stamp_source_move_settings,
        )

        for attr in ("eigen_axis_acc_log", "eigen_axis_acc_tune", "eigen_axis_acc_min",
                     "eigen_axis_acc_max", "eigen_axis_acc_factor"):
            self.assertIn(attr, SOURCE_MOVE_SETTINGS_ATTRS)
        mv = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
        for attr in SOURCE_MOVE_SETTINGS_ATTRS:
            setattr(mv, attr, None)
        mv.eigen_axis_acc_min = 0.2
        info = SimpleNamespace(eigen_axis_acc_tune=True, eigen_axis_acc_log=False,
                               eigen_axis_acc_min=0.05, eigen_axis_acc_max=0.6,
                               eigen_axis_acc_factor=3.0)
        stamp_source_move_settings(mv, info)
        self.assertEqual((mv.eigen_axis_acc_tune, mv.eigen_axis_acc_log, mv.eigen_axis_acc_min,
                          mv.eigen_axis_acc_max, mv.eigen_axis_acc_factor),
                         (True, False, 0.2, 0.6, 3.0))


if __name__ == "__main__":
    unittest.main()
