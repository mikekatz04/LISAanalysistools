"""Eigen step-scale adaptation from the chain (user ruling 2026-10-14).

Why: on the 9mo v9 run (job 763, rows 453-561, T_max 6 EMRI ladder) every
EMRI sampled a long degenerate intrinsic ridge; the eigen table (the Gram /
ll information matrix at ONE point) had the ridge DIRECTION as its widest
intrinsic axis but a step 1.4-56x smaller than the chain's own cold spread
along it, so the eryn eigen move needed ratio^2 iterations to cross the
ridge. ``{BRANCH}_EIGEN_AXIS_ADAPT`` keeps the axes and widens each axis's
step to the chain's robust width along it.

Pinned here (all CPU, synthetic; nothing builds a waveform):

* the empirical width: a tempered rung ensemble with a long axis 20x the
  "Fisher" sigma recovers that width within 20 % after K visits; the
  ``sqrt(beta)`` cold-unit mapping is exact; deviations are taken from each
  RUNG'S OWN centre (a displaced hot rung does not inflate the width);
  periodic columns are wrapped first; the Fisher floor and the table cap
  hold; the width is robust to outliers; the projection is the axis-basis
  coordinate ``A^-1 dev`` (the table's axes are not orthogonal);
* END-TO-END MIXING: the eryn ``EigenAxisMove`` fed by the move's refresh
  hook on a 5-D Gaussian whose long axis is 20x the table sigma -- the
  autocorrelation time along that axis drops by >= 5x with the adaptation,
  and a chain started at the mode bootstraps its own scale;
* persistence: the sidecar carries the adapted AND the Fisher sigmas; the
  head and a read-only compute rank adopt the adapted scale; a resume with
  the switch off runs the Fisher table; the carried scale survives a refresh
  before the ring buffer refills;
* the off switch: tables bit-identical, no buffer, no log line, no new
  sidecar key;
* the settings fields / env knobs and the builder plumbing; the ring buffer
  is fed by REAL proposes (single process and the walker-block fan-out).
"""

from __future__ import annotations

import copy
import os
import pickle
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.moves import EigenAxisMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import PeriodicContainer

from lisatools.globalfit.moves import eigen_refresh, eigen_table_persist
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

LOGGER = "lisatools.globalfit.moves.addremovemove"
BR = "emri"
_KNOBS = tuple(f"{b}_{k}" for b in ("EMRI", "MBH", "SOBBH")
               for k in ("EIGEN_AXIS_ADAPT", "EIGEN_ADAPT_VISITS",
                         "EIGEN_ADAPT_MIN_VISITS", "EIGEN_REFRESH")) + (
    "EIGEN_TABLES_PERSIST",)


def _clean_env(persist=False, **extra):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    if not persist:
        env["EIGEN_TABLES_PERSIST"] = "0"
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


def _rot(nd, seed):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(nd, nd)))[0]


def _stub(table, *, ndim, ntemps, nwalkers, adapt=True, visits=50, min_visits=10,
          refresh=1, box=200.0, store=None, periodic=None):
    """The add/remove move without data (``__new__``), as the eigen suites build it.

    ``table`` = the "information-matrix" ``(axes, sigmas)`` the custom builder
    hands back at every refresh (copies, so the installed arrays can be
    compared with it).
    """
    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = BR
    move.ndim, move.ntemps, move.nwalkers = ndim, ntemps, nwalkers
    move.moves = [EigenAxisMove()]
    move.priors = {BR: ProbDistContainer({i: uniform_dist(-box, box) for i in range(ndim)})}
    move.eigen_refresh_every = refresh
    move.eigen_eps_rel = None
    move.eigen_table_scope = None
    move.eigen_axis_adapt = adapt
    move.eigen_adapt_visits = visits
    move.eigen_adapt_min_visits = min_visits
    move.eigen_store_path = store
    move._to_phys = lambda x: x
    axes, sig = table
    move.eigen_table_builder = lambda mv, leaf, widths: (np.array(axes, copy=True),
                                                         np.array(sig, copy=True))
    if periodic is not None:
        move._periodic = PeriodicContainer(periodic)
    return move


def _feed(move, X, betas, leaf=0):
    """Push visits ``X (V, nt, nw, ndim)`` through the move's end-of-visit hook."""
    for x in X:
        move._note_eigen_adapt_sample(leaf, SimpleNamespace(coords=x[:, :, None, :]), betas)


def _tempered_draws(rng, R, s_cold, betas, V, nw, mu=None):
    """``(V, nt, nw, ndim)`` exact draws of N(mu_k, R diag(s^2) R^T / beta_k)."""
    nd = R.shape[0]
    nt = len(betas)
    z = rng.normal(size=(V, nt, nw, nd)) * s_cold
    x = np.einsum("ij,vtwj->vtwi", R, z) / np.sqrt(np.asarray(betas))[None, :, None, None]
    if mu is not None:
        x = x + np.asarray(mu)[None, :, None, :] if np.ndim(mu) == 2 else x + mu
    return x


def _installed(move):
    return move.moves[0]._tables[BR]


BETAS8 = 1.0 / np.geomspace(1.0, 6.0, 8)          # a T_max 6 ladder, 8 rungs
LONG = np.array([0.1, 0.1, 0.1, 0.1, 2.0])        # cold widths: axis 4 = 20x the table


# ---------------------------------------------------------------------------
# (a) the empirical width
# ---------------------------------------------------------------------------

class EmpiricalWidthTest(unittest.TestCase):
    def test_a_long_axis_20x_the_fisher_sigma_is_recovered_after_K_visits(self):
        R = _rot(5, 1)
        fisher = (R, np.full(5, 0.1))
        rng = np.random.default_rng(2)
        X = _tempered_draws(rng, R, LONG, BETAS8, 80, 1, mu=np.full(5, 3.0))
        with _clean_env():
            move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1)
            _feed(move, X, BETAS8)
            self.assertEqual(len(move._eigen_adapt_buffer[0]), 50)   # the last K only
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        axes, sig = _installed(move)
        np.testing.assert_array_equal(axes, R)                       # directions kept
        np.testing.assert_allclose(sig, LONG, rtol=0.2)
        self.assertGreater(sig[4] / 0.1, 16.0)
        np.testing.assert_array_equal(move._eigen_fisher_sigmas[0], fisher[1])

    def test_the_sqrt_beta_mapping_is_exact(self):
        """The SAME unit draws at every rung, scaled by 1/sqrt(beta_k): the
        cold-unit deviations of every rung coincide to rounding."""
        rng = np.random.default_rng(3)
        z = rng.normal(size=(200, 4)) * [1.0, 0.3, 5.0, 0.01]
        betas = np.array([1.0, 0.37, 0.081, 0.0123])
        mu = np.array([1.0, -2.0, 0.5, 7.0])
        rungs = [(mu + z / np.sqrt(b), np.full(len(z), b)) for b in betas]
        dev = eigen_refresh.cold_unit_deviations(rungs)
        self.assertEqual(len(dev), len(betas))
        for k in range(1, len(betas)):
            np.testing.assert_allclose(dev[k], dev[0], rtol=1e-12, atol=1e-12)
        # and on a tempered Gaussian ensemble the pooled width is the COLD one
        R = _rot(4, 4)
        s = np.array([1.0, 0.3, 5.0, 0.01])
        X = _tempered_draws(rng, R, s, betas, 1500, 1)[:, :, 0, :]
        rungs = [(X[:, k], np.full(len(X), b)) for k, b in enumerate(betas)]
        w = eigen_refresh.empirical_axis_widths(R, eigen_refresh.cold_unit_deviations(rungs))
        np.testing.assert_allclose(w, s, rtol=0.05)

    def test_deviations_are_taken_from_each_rungs_own_centre(self):
        """A hot rung sitting elsewhere on the ridge does not widen the cold step."""
        R = _rot(5, 5)
        rng = np.random.default_rng(6)
        mu = np.zeros((8, 5))
        mu[:, :] = (np.arange(8)[:, None] * 40.0) * R[:, 4][None, :]   # rung k displaced 40k along the long axis
        X = _tempered_draws(rng, R, LONG, BETAS8, 50, 1, mu=mu)[:, :, 0, :]
        rungs = [(X[:, k], np.full(len(X), b)) for k, b in enumerate(BETAS8)]
        w = eigen_refresh.empirical_axis_widths(R, eigen_refresh.cold_unit_deviations(rungs))
        np.testing.assert_allclose(w, LONG, rtol=0.2)
        # the alternative (every rung from the COLD centre) would size the step
        # to the between-rung displacement: shown here, not used by the code
        cold_c = np.median(X[:, 0], axis=0)
        alt = np.concatenate([(X[:, k] - cold_c) * np.sqrt(b) for k, b in enumerate(BETAS8)])
        w_alt = eigen_refresh.empirical_axis_widths(R, alt)
        self.assertGreater(w_alt[4] / LONG[4], 5.0)

    def test_periodic_columns_are_wrapped_before_projecting(self):
        """A phase straddling 0 / 2 pi: wrapped, the widths are right; the
        paired control without the periodic spec inflates every axis it feeds."""
        nd = 3
        R = _rot(nd, 7)
        s = np.array([0.05, 0.1, 0.2])
        rng = np.random.default_rng(8)
        X = _tempered_draws(rng, R, s, BETAS8[:4], 60, 2)
        X[..., 1] = np.mod(X[..., 1], 2 * np.pi)            # phase column on [0, 2 pi), centred at 0
        self.assertTrue(np.any(X[..., 1] > 6.0) and np.any(X[..., 1] < 0.3))
        table = (R, np.full(nd, 1e-3))
        with _clean_env():
            for per, ok in (({BR: {1: 2 * np.pi}}, True), (None, False)):
                with self.subTest(periodic=per is not None):
                    move = _stub(table, ndim=nd, ntemps=4, nwalkers=2, periodic=per,
                                 box=10.0)
                    _feed(move, X, BETAS8[:4])
                    move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
                    _, sig = _installed(move)
                    if ok:
                        np.testing.assert_allclose(sig, s, rtol=0.2)
                    else:
                        fed = np.abs(R[1]) > 0.25                # the axes the phase feeds most
                        self.assertTrue(np.any(fed))
                        self.assertTrue(np.all(sig[fed] > 3 * s[fed]), sig / s)

    def test_the_periodic_spec_is_read_off_an_inner_move_too(self):
        move = _stub((np.eye(3), np.ones(3)), ndim=3, ntemps=2, nwalkers=1)
        self.assertEqual(move._eigen_periodic_columns()[0].size, 0)
        move.moves[0].periodic = PeriodicContainer({BR: {2: np.pi, 0: 2 * np.pi}})
        inds, per = move._eigen_periodic_columns()
        self.assertEqual(sorted(zip(inds.tolist(), per.tolist())), [(0, 2 * np.pi), (2, np.pi)])

    def test_the_fisher_floor_holds(self):
        """A chain NARROWER than the table on some axes never shrinks them."""
        R = _rot(5, 9)
        fisher = (R, np.array([0.5, 0.5, 0.5, 0.5, 0.1]))
        X = _tempered_draws(np.random.default_rng(10), R, LONG, BETAS8, 50, 1)
        with _clean_env():
            move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1)
            _feed(move, X, BETAS8)
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        _, sig = _installed(move)
        np.testing.assert_array_equal(sig[:4], fisher[1][:4])
        np.testing.assert_allclose(sig[4], 2.0, rtol=0.2)

    def test_the_cap_holds_and_a_railed_axis_stays_capped(self):
        nd = 3
        R = np.eye(nd)
        box = 1.0                                          # widths 2 -> cap 2 per axis
        cap = eigen_refresh.axis_caps(R, np.full(nd, 2 * box))
        np.testing.assert_allclose(cap, 2.0)
        fisher = (R, np.array([2.0, 0.01, 0.01]))          # axis 0 railed at the cap
        s = np.array([0.001, 50.0, 0.02])                  # axis 1 far wider than the box
        X = _tempered_draws(np.random.default_rng(11), R, s, [1.0, 0.5], 50, 2)
        with _clean_env():
            move = _stub(fisher, ndim=nd, ntemps=2, nwalkers=2, box=box)
            _feed(move, X, np.array([1.0, 0.5]))
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        _, sig = _installed(move)
        self.assertEqual(sig[0], 2.0)                       # railed: stays at the cap
        self.assertEqual(sig[1], cap[1])                    # wider than the box: capped
        np.testing.assert_allclose(sig[2], 0.02, rtol=0.2)

    def test_the_width_is_robust_to_transient_outliers(self):
        R = _rot(5, 12)
        rng = np.random.default_rng(13)
        X = _tempered_draws(rng, R, LONG, BETAS8, 50, 1)[:, :, 0, :]
        out = rng.random(X.shape[:2]) < 0.05               # 5 % of the samples thrown far away
        X[out] += 1e3 * rng.normal(size=(int(out.sum()), 5))
        rungs = [(X[:, k], np.full(len(X), b)) for k, b in enumerate(BETAS8)]
        dev = eigen_refresh.cold_unit_deviations(rungs)
        w = eigen_refresh.empirical_axis_widths(R, dev)
        np.testing.assert_allclose(w, LONG, rtol=0.25)
        # the plain std of the same samples is ruined
        self.assertGreater(np.min(np.std(np.concatenate(dev) @ R, axis=0) / LONG), 10.0)

    def test_the_projection_is_the_axis_basis_coordinate(self):
        """Production axes are unit vectors un-whitened by very different prior
        widths, hence NOT orthogonal: c = A^-1 dev recovers the generating
        widths, the dot products A^T dev do not."""
        rng = np.random.default_rng(14)
        widths = np.array([1e-3, 1.0, 50.0, 2 * np.pi])
        M = rng.normal(size=(4, 4))
        info = (M @ M.T + 0.1 * np.eye(4)) / np.outer(widths, widths)
        axes, _ = eigen_refresh._table_from_info(info, widths)
        self.assertGreater(np.max(np.abs(axes.T @ axes - np.eye(4))), 0.1)
        s = np.array([0.2, 1.5, 0.03, 0.7])
        c = rng.normal(size=(20000, 4)) * s
        dev = c @ axes.T                                   # x = sum_k c_k a_k
        np.testing.assert_allclose(eigen_refresh.empirical_axis_widths(axes, dev), s, rtol=0.03)
        dots = eigen_refresh.MAD_TO_SIGMA * np.median(np.abs(dev @ axes), axis=0)
        self.assertGreater(np.max(np.abs(dots / s - 1.0)), 0.5)

    def test_below_min_visits_the_table_is_the_fisher_one(self):
        R = _rot(5, 15)
        fisher = (R, np.full(5, 0.1))
        X = _tempered_draws(np.random.default_rng(16), R, LONG, BETAS8, 10, 1)
        with _clean_env():
            move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, min_visits=10)
            _feed(move, X[:9], BETAS8)
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[0][:, :, None, :]))
            np.testing.assert_array_equal(_installed(move)[1], fisher[1])
            _feed(move, X[9:], BETAS8)
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[0][:, :, None, :]))
        self.assertGreater(_installed(move)[1][4], 1.0)

    def test_a_per_walker_stash_is_adapted_table_by_table(self):
        """The ``per_walker`` scope's (ntemps, nwalkers) tables: each adapted on
        its own axes, the floor and the stash shape kept."""
        nt, nw, nd = 2, 2, 5
        R = _rot(nd, 17)
        axes = np.broadcast_to(R, (nt, nw, nd, nd)).copy()
        floor = np.full((nt, nw, nd), 0.1)
        floor[1, 1, 0] = 0.5
        X = _tempered_draws(np.random.default_rng(18), R, LONG, [1.0, 0.5], 100, nw)
        with _clean_env():
            move = _stub((axes, floor), ndim=nd, ntemps=nt, nwalkers=nw, visits=100)
            _feed(move, X, np.array([1.0, 0.5]))
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        a, sig = move._eigen_tables[0]
        self.assertEqual(sig.shape, (nt, nw, nd))
        np.testing.assert_array_equal(a, axes)
        np.testing.assert_allclose(sig[..., 4], 2.0, rtol=0.2)
        self.assertEqual(sig[1, 1, 0], 0.5)
        np.testing.assert_array_equal(move._eigen_fisher_sigmas[0], floor)

    def test_the_lost_rung_guard_carries_the_floor_row_with_the_table_row(self):
        """A re-anchored rung takes the cold row's per-walker table AND its floor."""
        nt, nw, nd = 3, 2, 4
        rng = np.random.default_rng(19)
        axes = rng.normal(size=(nt, nw, nd, nd))
        sig = rng.uniform(1, 2, size=(nt, nw, nd))
        floor = rng.uniform(0, 1, size=(nt, nw, nd))
        f0 = floor.copy()
        move = _stub((np.eye(nd), np.ones(nd)), ndim=nd, ntemps=nt, nwalkers=nw)
        move.ladder_reanchor_nats = 10.0
        move._fanout_body = False
        move._eigen_tables = {0: (axes, sig)}
        move._eigen_fisher_sigmas = {0: floor}
        work = SimpleNamespace(coords=rng.normal(size=(nt, nw, 1, nd)),
                               inds=np.ones((nt, nw, 1), dtype=bool))
        ll = np.array([[0.0, 0.0], [-1.0, -1e4], [-2.0, -2.0]])
        with _clean_env(), self.assertLogs(LOGGER, level="INFO"):
            *_, rec = move._reanchor_lost_rungs(work, 0, ll, np.zeros((nt, nw)), np.array([1.0, 0.8, 0.6]))
        self.assertEqual([r[:2] for r in rec], [(1, 1)])
        np.testing.assert_array_equal(floor[1, 1], f0[0, 1])
        np.testing.assert_array_equal(sig[1, 1], sig[0, 1])
        np.testing.assert_array_equal(floor[2], f0[2])

    def test_the_knobs_K_and_min_visits(self):
        with _clean_env(EMRI_EIGEN_ADAPT_VISITS="7", EMRI_EIGEN_ADAPT_MIN_VISITS="3"):
            move = _stub((np.eye(2), np.ones(2)), ndim=2, ntemps=2, nwalkers=1,
                         visits=None, min_visits=None)
            self.assertEqual((move._eigen_adapt_visits(), move._eigen_adapt_min_visits()), (7, 3))
            _feed(move, np.zeros((12, 2, 1, 2)), np.array([1.0, 0.5]))
            self.assertEqual(len(move._eigen_adapt_buffer[0]), 7)
            move.eigen_adapt_visits = 4                      # an attribute set later wins
            _feed(move, np.zeros((1, 2, 1, 2)), np.array([1.0, 0.5]))
            self.assertEqual(len(move._eigen_adapt_buffer[0]), 4)
        with _clean_env():
            move.eigen_adapt_visits = move.eigen_adapt_min_visits = None
            self.assertEqual((move._eigen_adapt_visits(), move._eigen_adapt_min_visits()), (50, 10))


# ---------------------------------------------------------------------------
# (b) end-to-end mixing
# ---------------------------------------------------------------------------

def _tau(x):
    """Integrated autocorrelation time (FFT acf, Sokal window c = 5)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    win = np.arange(n) < 5.0 * taus
    m = int(np.argmin(win)) if np.any(~win) else n - 1
    return float(taus[m])


def _mixing_run(adapt, *, start, visits_burn, visits_meas, seed, repeats=25, refresh=5):
    """The eryn eigen move on a tempered 5-D Gaussian, tables from the move's hook.

    Target: N(0, R diag(LONG2^2) R^T) with LONG2 = [1,1,1,1,20]; the
    "information-matrix" table is (R, ones): the right directions, the long
    axis 20x too narrow. 4 rungs x 2 walkers, ``repeats`` MH steps per leaf
    visit, the table refreshed every ``refresh`` visits, the end-of-visit
    ensemble pushed into the ring buffer after each visit -- the move's own
    order. Returns (cold long-axis coordinate per repeat (n, nw), last table).
    """
    nd, nt, nw = 5, 4, 2
    R = _rot(nd, 21)
    s_true = np.array([1.0, 1.0, 1.0, 1.0, 20.0])
    betas = np.array([1.0, 0.5, 0.25, 0.125])
    box = 200.0
    rng = np.random.default_rng(seed)
    move = _stub((R, np.ones(nd)), ndim=nd, ntemps=nt, nwalkers=nw, adapt=adapt,
                 refresh=refresh, box=box)
    inner = move.moves[0]
    if start == "stationary":
        x = _tempered_draws(rng, R, s_true, betas, 1, nw)[0]
    else:                                                  # the mode, Fisher-sized jitter
        x = 0.1 * rng.normal(size=(nt, nw, nd))
    work = SimpleNamespace(coords=x[:, :, None, :].copy())

    def lnl(c):
        u = np.einsum("ij,...i->...j", R, c[..., 0, :]) / s_true
        return -0.5 * np.sum(u * u, axis=-1)

    ll = lnl(work.coords)
    trace = []
    for v in range(visits_burn + visits_meas):
        move.refresh_inner_move_tables(0, work)
        for _ in range(repeats):
            q, _ = inner.get_proposal({BR: work.coords}, rng)
            new = q[BR]
            inbox = np.all(np.abs(new[..., 0, :]) < box, axis=-1)
            ll_new = np.where(inbox, lnl(new), -np.inf)
            acc = betas[:, None] * (ll_new - ll) > np.log(rng.random((nt, nw)))
            work.coords[acc] = new[acc]
            ll = np.where(acc, ll_new, ll)
            if v >= visits_burn:
                trace.append(work.coords[0, :, 0, :] @ R[:, 4])
        move._note_eigen_adapt_sample(0, work, betas)
    return np.array(trace), move._eigen_tables[0]


class MixingTest(unittest.TestCase):
    def test_the_long_axis_autocorrelation_time_drops_by_5x(self):
        with _clean_env():
            tr_off, tab_off = _mixing_run(False, start="stationary", visits_burn=50,
                                          visits_meas=400, seed=31)
            tr_on, tab_on = _mixing_run(True, start="stationary", visits_burn=50,
                                        visits_meas=400, seed=31)
        tau_off = np.mean([_tau(tr_off[:, w]) for w in range(tr_off.shape[1])])
        tau_on = np.mean([_tau(tr_on[:, w]) for w in range(tr_on.shape[1])])
        # the off chain is far shorter than ~50 tau, so tau_off is a LOWER
        # bound: the true ratio is larger than the one asserted
        self.assertGreaterEqual(tau_off / tau_on, 5.0, (tau_off, tau_on))
        np.testing.assert_array_equal(tab_off[1], np.ones(5))
        np.testing.assert_allclose(tab_on[1][4], 20.0, rtol=0.25)
        np.testing.assert_allclose(tab_on[1][:4], 1.0, rtol=0.25)
        # the adapted chain really explores the long axis at its cold width
        self.assertGreater(np.std(tr_on), 10.0)

    def test_a_chain_started_at_the_mode_bootstraps_its_own_scale(self):
        with _clean_env():
            _, tab = _mixing_run(True, start="mode", visits_burn=250, visits_meas=0, seed=32)
        np.testing.assert_allclose(tab[1][4], 20.0, rtol=0.3)


# ---------------------------------------------------------------------------
# (c) persistence
# ---------------------------------------------------------------------------

class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = os.path.join(self._tmp.name, "gf_prod_9mo_testing.h5")
        self.sidecar = eigen_table_persist.sidecar_path(self.store)

    def _guards(self, **kw):
        g = dict(scope="walker_max", ndim=5, ntemps=8, nwalkers=1)
        g.update(kw)
        return g

    def test_entry_round_trip_of_adapted_and_fisher_sigmas(self):
        rng = np.random.default_rng(40)
        for shape in ((5,), (2, 3, 5)):
            with self.subTest(shape=shape):
                axes = rng.normal(size=shape[:-1] + (5, 5))
                sig, fis = rng.uniform(1, 2, size=shape), rng.uniform(0, 1, size=shape)
                scope = "walker_max" if len(shape) == 1 else "per_walker"
                g = self._guards(scope=scope, ntemps=2, nwalkers=3)
                eigen_table_persist.save_entry(self.sidecar, BR, 0, eigen_table_persist.make_entry(
                    axes, sig, 6, sigmas_fisher=fis, **g))
                a, s, v, f = eigen_table_persist.load_entry(self.sidecar, BR, 0, return_fisher=True, **g)
                np.testing.assert_array_equal(a, axes)
                np.testing.assert_array_equal(s, sig)
                np.testing.assert_array_equal(f, fis)
                self.assertEqual(v, 6)
                # the default call keeps its 3-tuple
                self.assertEqual(len(eigen_table_persist.load_entry(self.sidecar, BR, 0, **g)), 3)

    def test_an_entry_without_the_key_reads_as_fisher_none(self):
        g = self._guards()
        entry = eigen_table_persist.make_entry(np.eye(5), np.ones(5), 1, **g)
        self.assertNotIn("sigmas_fisher", entry)
        eigen_table_persist.save_entry(self.sidecar, BR, 0, entry)
        *_, f = eigen_table_persist.load_entry(self.sidecar, BR, 0, return_fisher=True, **g)
        self.assertIsNone(f)

    def test_a_misshapen_fisher_entry_is_discarded(self):
        g = self._guards()
        entry = eigen_table_persist.make_entry(np.eye(5), np.ones(5), 1, sigmas_fisher=np.ones(4), **g)
        eigen_table_persist.save_entry(self.sidecar, BR, 0, entry)
        with self.assertLogs(eigen_refresh.logger, level="WARNING") as cm:
            self.assertIsNone(eigen_table_persist.load_entry(self.sidecar, BR, 0, **g))
        self.assertIn("sigmas_fisher shape", "\n".join(cm.output))

    def _adapted_run(self):
        """A head move that adapted leaf 0 and wrote the sidecar."""
        R = _rot(5, 41)
        fisher = (R, np.full(5, 0.1))
        X = _tempered_draws(np.random.default_rng(42), R, LONG, BETAS8, 50, 1)
        move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, store=self.store, refresh=100)
        _feed(move, X, BETAS8)
        move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        return move, fisher

    def test_the_sidecar_carries_both_and_the_head_adopts_the_adapted_scale(self):
        with _clean_env(persist=True):
            first, fisher = self._adapted_run()
            adapted = first._eigen_tables[0][1].copy()
            self.assertGreater(adapted[4], 1.0)
            entry = eigen_table_persist.load_sidecar(self.sidecar)["entries"][f"{BR}:0"]
            np.testing.assert_array_equal(entry["sigmas"], adapted)
            np.testing.assert_array_equal(entry["sigmas_fisher"], fisher[1])
            # the resumed process: a fresh move, empty buffer
            again = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, store=self.store, refresh=100)
            with mock.patch.object(again, "eigen_table_builder") as build:
                again.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((8, 1, 1, 5))))
            build.assert_not_called()
        np.testing.assert_array_equal(_installed(again)[1], adapted)
        np.testing.assert_array_equal(again._eigen_fisher_sigmas[0], fisher[1])
        self.assertEqual(again._eigen_visit_count[0], 2)

    def test_a_read_only_compute_rank_adopts_the_adapted_scale(self):
        with _clean_env(persist=True):
            first, fisher = self._adapted_run()
            rank = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, store=self.store, refresh=100)
            rank.eigen_store_readonly = True
            rank._eigen_tables, rank._eigen_visit_count = {}, {}
            self.assertTrue(rank._adopt_persisted_eigen_table(0))
        np.testing.assert_array_equal(rank._eigen_tables[0][1], first._eigen_tables[0][1])
        np.testing.assert_array_equal(rank._eigen_fisher_sigmas[0], fisher[1])

    def test_a_resume_with_the_switch_off_runs_the_fisher_table(self):
        with _clean_env(persist=True):
            first, fisher = self._adapted_run()
            off = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, store=self.store, refresh=100,
                        adapt=False)
            off.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((8, 1, 1, 5))))
        np.testing.assert_array_equal(_installed(off)[1], fisher[1])
        self.assertFalse(getattr(off, "_eigen_fisher_sigmas", None))

    def test_the_carried_scale_survives_a_refresh_before_the_buffer_refills(self):
        with _clean_env(persist=True):
            first, fisher = self._adapted_run()
            adapted = first._eigen_tables[0][1].copy()
            # the stored counter lands ON the cadence: the resumed process
            # rebuilds at its first visit, with an EMPTY buffer
            entry = eigen_table_persist.load_sidecar(self.sidecar)["entries"][f"{BR}:0"]
            entry["visits"] = 100
            eigen_table_persist.save_entry(self.sidecar, BR, 0, entry)
            again = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, store=self.store, refresh=100)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                again.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((8, 1, 1, 5))))
        np.testing.assert_allclose(_installed(again)[1], adapted, rtol=1e-10)
        self.assertIn("carried from the persisted table, buffer 0 of 10 visits", "\n".join(cm.output))

    def test_carried_widths_are_exact_on_unchanged_axes_and_map_onto_new_ones(self):
        R = _rot(4, 43)
        s = np.array([0.3, 2.0, 0.01, 5.0])
        np.testing.assert_allclose(eigen_refresh.carried_axis_widths(R, R, s), s, rtol=1e-10)
        A2 = _rot(4, 44)
        C = R @ np.diag(s ** 2) @ R.T
        want = np.sqrt(np.diag(np.linalg.inv(A2) @ C @ np.linalg.inv(A2).T))
        np.testing.assert_allclose(eigen_refresh.carried_axis_widths(A2, R, s), want, rtol=1e-10)


# ---------------------------------------------------------------------------
# (d) the off switch
# ---------------------------------------------------------------------------

class OffSwitchTest(unittest.TestCase):
    def test_off_is_bit_identical_silent_and_bufferless(self):
        R = _rot(5, 50)
        fisher = (R, np.full(5, 0.1))
        X = _tempered_draws(np.random.default_rng(51), R, LONG, BETAS8, 30, 1)
        for kw, env in ((dict(adapt=False), {}), (dict(adapt=None), {}),
                        (dict(adapt=None), {"EMRI_EIGEN_AXIS_ADAPT": "0"})):
            with self.subTest(**kw, env=env), _clean_env(**env):
                move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1, **kw)
                with self.assertNoLogs(LOGGER, level="INFO"):
                    for x in X:
                        move.refresh_inner_move_tables(0, SimpleNamespace(coords=x[:, :, None, :]))
                        _feed(move, x[None], BETAS8)
                axes, sig = _installed(move)
                np.testing.assert_array_equal(axes, R)
                np.testing.assert_array_equal(sig, fisher[1])
                self.assertFalse(hasattr(move, "_eigen_adapt_buffer"))
                self.assertFalse(getattr(move, "_eigen_fisher_sigmas", None))

    def test_off_writes_the_pre_adaptation_sidecar_entry(self):
        with tempfile.TemporaryDirectory() as tmp, _clean_env(persist=True):
            store = os.path.join(tmp, "s.h5")
            move = _stub((np.eye(3), np.ones(3)), ndim=3, ntemps=2, nwalkers=1, adapt=False,
                         store=store)
            move.refresh_inner_move_tables(0, SimpleNamespace(coords=np.zeros((2, 1, 1, 3))))
            entry = eigen_table_persist.load_sidecar(
                eigen_table_persist.sidecar_path(store))["entries"][f"{BR}:0"]
        self.assertNotIn("sigmas_fisher", entry)

    def test_the_log_line_names_the_axis_ratio_and_the_buffer(self):
        """The table's widest axis (largest sigma / cap) is the ridge."""
        R = _rot(5, 52)
        fisher = (R, np.array([0.1, 0.1, 0.1, 0.1, 0.25]))
        X = _tempered_draws(np.random.default_rng(53), R, LONG, BETAS8, 50, 1)
        with _clean_env():
            move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1)
            _feed(move, X, BETAS8)
            with self.assertLogs(LOGGER, level="INFO") as cm:
                move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        line = "\n".join(cm.output)
        m = re.search(r"\[EMRI_EIGEN\] leaf 0: step scale adapted on 1 of 5 axes, max x(\d+\.\d) "
                      r"\(axis 4, ridge\), median x(\d+\.\d) \(50 visits, 8 rungs\)", line)
        self.assertIsNotNone(m, line)
        self.assertGreater(float(m.group(1)), 6.0)
        self.assertEqual(m.group(1), m.group(2))          # one adapted axis: max == median

    def test_on_but_nothing_moved_by_20_percent_is_silent(self):
        R = _rot(5, 54)
        fisher = (R, np.full(5, 0.1))
        X = _tempered_draws(np.random.default_rng(55), R, np.full(5, 0.1), BETAS8, 50, 1)
        with _clean_env():
            move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1)
            _feed(move, X, BETAS8)
            with self.assertNoLogs(LOGGER, level="INFO"):
                move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        self.assertLess(np.max(_installed(move)[1] / 0.1), 1.2)

    def test_a_failing_adaptation_falls_back_to_the_fisher_table(self):
        R = _rot(5, 56)
        fisher = (R, np.full(5, 0.1))
        X = _tempered_draws(np.random.default_rng(57), R, LONG, BETAS8, 20, 1)
        with _clean_env():
            move = _stub(fisher, ndim=5, ntemps=8, nwalkers=1)
            _feed(move, X, BETAS8)
            with mock.patch.object(eigen_refresh, "empirical_axis_widths",
                                   side_effect=RuntimeError("boom")), \
                    self.assertLogs(LOGGER, level="WARNING") as cm:
                move.refresh_inner_move_tables(0, SimpleNamespace(coords=X[-1][:, :, None, :]))
        np.testing.assert_array_equal(_installed(move)[1], fisher[1])
        self.assertIn("step-scale adaptation failed", "\n".join(cm.output))


# ---------------------------------------------------------------------------
# (e) settings fields, env knobs, builder plumbing
# ---------------------------------------------------------------------------

class KnobTest(unittest.TestCase):
    def _settings(self, branch):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        return {"emri": (EMRISettings, sr.SourceEMRISettings),
                "mbh": (MBHSettings, sr.SourceMBHSettings),
                "sobbh": (SOBBHSettings, sr.SourceSOBBHSettings)}[branch]

    def test_the_settings_field_default_and_env(self):
        for branch in ("emri", "mbh", "sobbh"):
            for cls in self._settings(branch):
                with self.subTest(cls=cls.__name__):
                    with _clean_env():
                        self.assertIs(cls().eigen_axis_adapt, False)
                    with _clean_env(**{f"{branch.upper()}_EIGEN_AXIS_ADAPT": "1"}):
                        s = cls()
                    self.assertIs(s.eigen_axis_adapt, True)
                    self.assertIs(pickle.loads(pickle.dumps(copy.deepcopy(s))).eigen_axis_adapt, True)

    def test_the_move_resolves_kwarg_then_env_then_off(self):
        for branch in ("emri", "mbh", "sobbh"):
            B = branch.upper()
            mv = ResidualAddOneRemoveOneMove(
                branch, (2, 2, 1, 4), lambda *a, **k: None, {}, {}, None, 2, None, None,
                [EigenAxisMove()])
            with self.subTest(branch=branch):
                with _clean_env():
                    self.assertIs(mv._eigen_axis_adapt(), False)
                for raw, want in (("1", True), ("on", True), ("0", False), ("no", False)):
                    with _clean_env(**{f"{B}_EIGEN_AXIS_ADAPT": raw}):
                        self.assertIs(mv._eigen_axis_adapt(), want)
                with _clean_env(**{f"{B}_EIGEN_AXIS_ADAPT": "1", f"{B}_EIGEN_ADAPT_VISITS": "30",
                                   f"{B}_EIGEN_ADAPT_MIN_VISITS": "5"}):
                    self.assertEqual((mv._eigen_adapt_visits(), mv._eigen_adapt_min_visits()), (30, 5))
                    mv.eigen_axis_adapt = False                       # the attribute wins
                    self.assertIs(mv._eigen_axis_adapt(), False)
                    mv.eigen_axis_adapt = None
        mv = ResidualAddOneRemoveOneMove(
            "emri", (2, 2, 1, 4), lambda *a, **k: None, {}, {}, None, 2, None, None,
            [EigenAxisMove()], eigen_axis_adapt=True, eigen_adapt_visits=7, eigen_adapt_min_visits=3)
        with _clean_env():
            self.assertEqual((mv._eigen_axis_adapt(), mv._eigen_adapt_visits(),
                              mv._eigen_adapt_min_visits()), (True, 7, 3))

    def test_the_builder_hands_the_settings_field_to_the_move(self):
        try:
            from tests import test_source_default_ladders as L
        except ImportError:  # pragma: no cover - run from tests/
            import test_source_default_ladders as L
        from eryn.moves.tempering import make_ladder

        cfg = make_ladder(L.NDIM, ntemps=4)
        for val in (True, False, None):
            with self.subTest(val=val):
                info = L._info(cfg)
                if val is not None:
                    info.eigen_axis_adapt = val
                move, _ = L._build(info, L._sub(4))
                self.assertIs(move.kwargs["eigen_axis_adapt"], val)


# ---------------------------------------------------------------------------
# the ring buffer is fed by REAL proposes
# ---------------------------------------------------------------------------

try:
    from tests import test_emri_direct_fanout as E
except ImportError:  # pragma: no cover - run from tests/
    import test_emri_direct_fanout as E


def _snap_with_buffer(move, acs, w0):
    out = _REAL_SNAPSHOT(move, acs, w0)
    out["buffer"] = {k: list(v) for k, v in (getattr(move, "_eigen_adapt_buffer", None) or {}).items()}
    out["fisher"] = dict(getattr(move, "_eigen_fisher_sigmas", None) or {})
    return out


_REAL_SNAPSHOT = E._snapshot


class RealProposeTest(E._Case):
    """``EMIRDirectLikeMove`` on the tiny WDM fixture of test_emri_direct_fanout."""

    KW = dict(scope="per_walker", adaptive=False, refresh=1, rng=E.H._RowLocalRandom,
              permute_every=0)

    def _env(self, on):
        env = {"EMRI_EIGEN_AXIS_ADAPT": "1" if on else "0", "EMRI_EIGEN_ADAPT_MIN_VISITS": "1",
               "EIGEN_TABLES_PERSIST": "0"}
        return mock.patch.dict(os.environ, env)

    def test_each_visit_pushes_the_end_of_visit_rung_ensemble(self):
        nw = 2
        with self._env(True), mock.patch.object(E, "_snapshot", _snap_with_buffer), \
                E.H._row_local_numpy():
            out = self._single(nw, **self.KW)
        final = out["states"][-1].sub_states["emri"]
        for leaf in range(E.H.NLEAVES):
            ring = out["snap"]["buffer"][leaf]
            self.assertEqual(len(ring), self.N_PROPOSES)               # one per visit
            x, b = ring[-1]
            self.assertEqual(x.shape, (E.H.NTEMPS, nw, E.H.NDIM))
            np.testing.assert_array_equal(x, final.coords[:, :, leaf])
            np.testing.assert_array_equal(b, final.betas_all[leaf])
            # the second visit's refresh adapted from the first visit's sample
            self.assertIn(leaf, out["snap"]["fisher"])
        # ... without falling back (no "[EMRI_EIGEN] ... failed / not recorded")
        self.assertEqual([l for l in self.cap.lines if "EIGEN]" in l], [])
        # paired control: off -> no buffer at all
        with self._env(False), mock.patch.object(E, "_snapshot", _snap_with_buffer), \
                E.H._row_local_numpy():
            off = self._single(nw, **self.KW)
        self.assertEqual(off["snap"]["buffer"], {})
        self.assertEqual(off["snap"]["fisher"], {})

    def test_under_the_walker_block_fanout_each_rank_buffers_its_own_walker(self):
        nw = 4
        with self._env(True), mock.patch.object(E, "_snapshot", _snap_with_buffer), \
                E.H._row_local_numpy():
            out = self._world(4, nw, **self.KW)
        final = out[0]["states"][-1].sub_states["emri"]
        computes = {r: v for r, v in out.items() if v["role"] != "saver"}
        self.assertEqual(len(computes), 4)
        for rank, res in computes.items():
            w0 = res["snap"]["w0"]
            with self.subTest(rank=rank, walker=w0):
                for leaf in range(E.H.NLEAVES):
                    ring = res["snap"]["buffer"][leaf]
                    self.assertEqual(len(ring), self.N_PROPOSES)
                    x, _ = ring[-1]
                    self.assertEqual(x.shape, (E.H.NTEMPS, 1, E.H.NDIM))  # its own block only
                    np.testing.assert_array_equal(x[:, 0], final.coords[:, w0, leaf])
                    self.assertIn(leaf, res["snap"]["fisher"])        # each rank adapts its own
        self.assertEqual([l for l in self.cap.lines if "EIGEN]" in l], [])


if __name__ == "__main__":
    unittest.main()
