"""The add/remove move's lost-rung guard (user ruling 2026-10-13).

9mo v9, 2026-10-13: the loudest EMRI (leaf 3, SNR 49) had rungs 2-7 ~1,000
nats below its cold rung on every walker since the first source_search block;
the hot rungs had lost the source for good. Ruling: at the start of a leaf's
block, a rung more than THRESH (``{BRANCH}_LADDER_REANCHOR_NATS``, default
200; 0 = off) nats below its OWN cold rung -- beyond the rung's expected
tempering drop ``ndim/2 (1/beta_k - 1)`` -- is re-anchored to the cold state:
the coords AND the per-rung bookkeeping the temperature swap carries (inds,
lnL, log prior; the per-walker eigen table row under that scope).

These tests drive ``_reanchor_lost_rungs`` on a light fake (no waveform
build) and check what the end-of-visit telemetry then writes to the
sub-state; ``tests/test_addremove_ladder_guard_fanout.py`` runs it inside
REAL proposes, single process and under the walker-block fan-out.
"""

from __future__ import annotations

import logging
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.moves.tempering import make_ladder

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.state import EMRIState

NT, NW, NLEAVES, NDIM = 4, 3, 5, 12
LEAF = 3
LOGGER = "lisatools.globalfit.moves.addremovemove"
GEOMETRIC = 1.0 / 1.2 ** np.arange(NT)   # expected drop <= 6 * (1.2**3 - 1) = 4.4


def _move(**attrs):
    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "emri"
    move.ntemps = NT
    move.nwalkers = NW
    move.ndim = NDIM
    move.ladder_reanchor_nats = None
    move._fanout_body = False
    move._fanout_swap_tally = {}
    for k, v in attrs.items():
        setattr(move, k, v)
    return move


def _work(seed=1):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        coords=rng.normal(size=(NT, NW, NLEAVES, NDIM)),
        inds=np.ones((NT, NW, NLEAVES), dtype=bool),
    )


def _lnl(cold=(-100.0, -200.0, -300.0), drops=None):
    """``(NT, NW)`` lnL: rung k = cold - drops[k, w] (default 3 nats)."""
    cold = np.asarray(cold, dtype=float)
    d = np.full((NT, NW), 3.0) if drops is None else np.asarray(drops, dtype=float)
    d[0] = 0.0
    return cold[None, :] - d


def _logp():
    return -np.arange(NT * NW, dtype=float).reshape(NT, NW) / 7.0


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k != "EMRI_LADDER_REANCHOR_NATS"}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class ReanchorTest(unittest.TestCase):
    def _run(self, move, drops, betas=GEOMETRIC, work=None, logp=None, cold=None):
        work = _work() if work is None else work
        ll = _lnl(drops=drops) if cold is None else _lnl(cold=cold, drops=drops)
        lp = _logp() if logp is None else logp
        before = (work.coords.copy(), work.inds.copy(), ll.copy(), lp.copy())
        out_ll, out_lp, rec = move._reanchor_lost_rungs(work, LEAF, ll, lp, betas)
        return work, before, out_ll, out_lp, rec

    def test_a_rung_1000_below_is_reanchored_with_its_bookkeeping(self):
        drops = np.full((NT, NW), 3.0)
        drops[2, 1] = 1000.0          # walker 1, rung 2: lost
        drops[1, 0] = 50.0            # walker 0, rung 1: low but within reach
        with _clean_env(), self.assertLogs(LOGGER, level="INFO") as cm:
            work, (c0, i0, l0, p0), ll, lp, rec = self._run(_move(), drops)
        self.assertEqual(rec, [(1, 2, 1000.0)])
        # the lost rung IS the cold state now: coords, inds, lnL, log prior
        np.testing.assert_array_equal(work.coords[2, 1, LEAF], c0[0, 1, LEAF])
        self.assertTrue(work.inds[2, 1, LEAF])
        self.assertEqual(ll[2, 1], l0[0, 1])
        self.assertEqual(lp[2, 1], p0[0, 1])
        # everything else untouched: the 50-nat rung, other walkers, other
        # leaves, the cold rung
        mask = np.ones((NT, NW), dtype=bool)
        mask[2, 1] = False
        np.testing.assert_array_equal(ll[mask], l0[mask])
        np.testing.assert_array_equal(lp[mask], p0[mask])
        np.testing.assert_array_equal(work.coords[1, 0, LEAF], c0[1, 0, LEAF])
        other = np.delete(np.arange(NLEAVES), LEAF)
        np.testing.assert_array_equal(work.coords[:, :, other], c0[:, :, other])
        np.testing.assert_array_equal(work.coords[0], c0[0])
        line = "\n".join(cm.output)
        self.assertIn("[EMRI_LADDER] leaf 3: re-anchored 1 rung(s) to the cold state", line)
        self.assertIn("walker 1 rung 2,", line)
        self.assertIn("1000-1000 nats below cold", line)
        self.assertIn("threshold 200", line)

    def test_the_log_names_walkers_rung_ranges_and_deficits(self):
        drops = np.full((NT, NW), 3.0)
        drops[1:, 0] = [954.0, 1010.0, 1125.0]   # walker 0, rungs 1-3
        drops[3, 2] = 400.0                       # walker 2, rung 3
        with _clean_env(), self.assertLogs(LOGGER, level="INFO") as cm:
            *_, rec = self._run(_move(), drops)
        self.assertEqual(len(rec), 4)
        line = "\n".join(cm.output)
        self.assertIn("re-anchored 4 rung(s)", line)
        self.assertIn("walker 0 rungs 1-3; walker 2 rung 3", line)
        self.assertIn("400-1125 nats below cold", line)

    def test_a_rung_50_below_is_untouched(self):
        drops = np.full((NT, NW), 50.0)
        with _clean_env(), self.assertNoLogs(LOGGER, level="INFO"):
            work, (c0, i0, l0, p0), ll, lp, rec = self._run(_move(), drops)
        self.assertEqual(rec, [])
        np.testing.assert_array_equal(work.coords, c0)
        np.testing.assert_array_equal(ll, l0)

    def test_threshold_zero_disables(self):
        drops = np.full((NT, NW), 5000.0)
        for move, env in ((_move(ladder_reanchor_nats=0.0), {}),
                          (_move(), {"EMRI_LADDER_REANCHOR_NATS": "0"})):
            with self.subTest(env=env), _clean_env(**env):
                work, (c0, *_), ll, lp, rec = self._run(move, drops)
                self.assertEqual(rec, [])
                np.testing.assert_array_equal(work.coords, c0)

    def test_the_threshold_knob(self):
        drops = np.full((NT, NW), 3.0)
        drops[1, 2] = 120.0
        with _clean_env(EMRI_LADDER_REANCHOR_NATS="100"):
            *_, rec = self._run(_move(), drops)
        self.assertEqual([r[:2] for r in rec], [(2, 1)])
        with _clean_env(EMRI_LADDER_REANCHOR_NATS="100"):
            *_, rec = self._run(_move(ladder_reanchor_nats=150.0), drops)   # kwarg wins
        self.assertEqual(rec, [])

    def test_the_expected_tempering_drop_is_part_of_the_threshold(self):
        """make_ladder's top rung (T ~ 120 at 8 rungs, ndim 12) is EXPECTED far down."""
        nt = 8
        betas = make_ladder(NDIM, ntemps=nt)
        drops = np.full((nt, NW), 3.0)
        drops[0] = 0.0
        drops[7, 0] = 600.0     # top rung: threshold 200 + 6 (T - 1) ~ 918 -> legitimate
        drops[7, 1] = 1100.0    # top rung, far beyond -> lost
        drops[2, 2] = 250.0     # rung 2 (T ~ 3.9): 200 + 17.6 -> lost
        ll = np.array([-100.0, -200.0, -300.0])[None, :] - drops
        work = SimpleNamespace(coords=np.zeros((nt, NW, NLEAVES, NDIM)),
                               inds=np.ones((nt, NW, NLEAVES), dtype=bool))
        with _clean_env():
            *_, rec = _move(ntemps=nt)._reanchor_lost_rungs(
                work, LEAF, ll.copy(), np.zeros((nt, NW)), betas)
        self.assertEqual(sorted(r[:2] for r in rec), [(1, 7), (2, 2)])
        # the same deficits on the retired 1/1.2^k ladder (drop <= 15.5): all three
        with _clean_env():
            *_, rec = _move(ntemps=nt)._reanchor_lost_rungs(
                work, LEAF, ll.copy(), np.zeros((nt, NW)), 1 / 1.2 ** np.arange(nt))
        self.assertEqual(sorted(r[:2] for r in rec), [(0, 7), (1, 7), (2, 2)])

    def test_an_infinite_temperature_rung_never_fires(self):
        betas = np.array([1.0, 0.5, 0.25, 0.0])
        drops = np.full((NT, NW), 1e6)
        with _clean_env():
            *_, rec = self._run(_move(), drops, betas=betas)
        self.assertNotIn(3, {k for _, k, _ in rec})
        self.assertEqual(len(rec), 2 * NW)

    def test_an_invalid_cold_rung_anchors_nothing_a_nan_hot_rung_is_reanchored(self):
        drops = np.full((NT, NW), 3.0)
        ll = _lnl(drops=drops)
        ll[0, 0] = -1e300            # walker 0: cold at the out-of-domain sentinel
        ll[2, 1] = np.nan            # walker 1, rung 2: broken
        work = _work()
        with _clean_env():
            _, _, rec = _move()._reanchor_lost_rungs(work, LEAF, ll, _logp(), GEOMETRIC)
        self.assertEqual([r[:2] for r in rec], [(1, 2)])

    def test_per_walker_eigen_table_rows_follow(self):
        rng = np.random.default_rng(5)
        axes = rng.normal(size=(NT, NW, NDIM, NDIM))
        sigmas = rng.uniform(size=(NT, NW, NDIM))
        shared = (rng.normal(size=(NDIM, NDIM)), rng.uniform(size=NDIM))
        a0, s0 = axes.copy(), sigmas.copy()
        move = _move(_eigen_tables={LEAF: (axes, sigmas), 0: shared})
        drops = np.full((NT, NW), 3.0)
        drops[3, 2] = 900.0
        with _clean_env():
            self._run(move, drops)
        np.testing.assert_array_equal(axes[3, 2], a0[0, 2])
        np.testing.assert_array_equal(sigmas[3, 2], s0[0, 2])
        np.testing.assert_array_equal(axes[1:3], a0[1:3])
        self.assertIs(move._eigen_tables[0], shared)

    def test_walker_indices_are_global_under_the_block_fanout(self):
        fanout = SimpleNamespace(rank=2, layout=SimpleNamespace(block_of=lambda r: (2 * r, 2 * r + NW)))
        move = _move(_fanout_body=True, fanout=fanout, row_fanout=None)
        drops = np.full((NT, NW), 3.0)
        drops[1, 0] = 1000.0
        with _clean_env(), self.assertLogs(LOGGER, level="INFO") as cm:
            *_, rec = self._run(move, drops)
        self.assertEqual(rec, [(4, 1, 1000.0)])
        self.assertIn("walker 4 rung 1,", "\n".join(cm.output))
        self.assertEqual(move._ladder_reanchor_log, {LEAF: [(4, 1, 1000.0)]})
        # the flat (replica) body runs the whole ensemble: no offset
        move = _move(_fanout_body=True, fanout=fanout, row_fanout=object())
        with _clean_env():
            *_, rec = self._run(move, drops)
        self.assertEqual(rec, [(0, 1, 1000.0)])

    def test_nothing_lost_returns_the_inputs_untouched(self):
        ll, lp = _lnl(), _logp()
        with _clean_env():
            out_ll, out_lp, rec = _move()._reanchor_lost_rungs(_work(), LEAF, ll, lp, GEOMETRIC)
        self.assertIs(out_ll, ll)
        self.assertIs(out_lp, lp)
        self.assertEqual(rec, [])


class TelemetryAfterReanchorTest(unittest.TestCase):
    """The sub-state's per-leaf lnL / log prior record the re-anchored rung."""

    def test_the_stored_bookkeeping_equals_the_cold_rung(self):
        sub = EMRIState(None, betas_all=np.tile(GEOMETRIC, (NLEAVES, 1)))
        sub.initialize_tempered(NT, NW, NLEAVES, NDIM)
        work = sub.branch                     # the view the move samples
        work.coords[:] = np.random.default_rng(2).normal(size=work.coords.shape)
        move = _move()
        drops = np.full((NT, NW), 3.0)
        drops[2, 0] = 1000.0
        ll, lp = _lnl(drops=drops), _logp()
        with _clean_env():
            ll, lp, rec = move._reanchor_lost_rungs(work, LEAF, ll, lp, GEOMETRIC)
        self.assertEqual(len(rec), 1)
        state = SimpleNamespace(sub_states={"emri": sub})
        move._record_leaf_telemetry(state, LEAF, np.zeros(NT, int), np.zeros(NT, int), ll, lp)
        self.assertEqual(sub.log_like[LEAF, 2, 0], sub.log_like[LEAF, 0, 0])
        self.assertEqual(sub.log_prior[LEAF, 2, 0], sub.log_prior[LEAF, 0, 0])
        np.testing.assert_array_equal(sub.coords[2, 0, LEAF], sub.coords[0, 0, LEAF])
        self.assertEqual(sub.log_like[LEAF, 1, 0], ll[1, 0])


if __name__ == "__main__":
    unittest.main()
