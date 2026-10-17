"""GB parallel residual search (rj_para_search_pe): the cheap pieces.

User ruling 2026-10-16 (docs/gb-para-search.md). Everything here runs without
a GB comp: the knobs and their env names, the clustering and band grouping,
the band box prior, the per-group convergence tracker (also through a real
``ParaEnsembleSampler`` on a toy likelihood, where the batch must SHRINK as
groups converge), the head's merge across walkers and band edges, the epoch
npz loading through ``WarmStartComponents.from_npz`` unchanged, the holder's
poll / directive / lag rules, the build clock, the global-RNG scope, the
recipe composition, the PE draw-one share, and the move's wiring on a
skeleton (stage gate, first-entry build, one-draw lag, rank-side apply,
replica-layout refusal, the op dispatch). CPU only, seconds.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.warmstart import para_search as ps


def _env(**kw):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kw.items()}, clear=False)


def _clean_env():
    return mock.patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                        if not k.startswith("GB_PARA_SEARCH")
                                        and k != "GB_PE_RJ_PARA_FRACTION"},
                           clear=True)


TOBS = 1.0e6
DF = 1.0 / TOBS


def _transform():
    from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

    return make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                       use_distance=True, mc_lims=(0.001, 1.0))


def _blob(f0_hz, n=400, seed=0, mc=0.3, dist=2.0):
    """``(n, 9)`` sampling rows around one source (a stand-in posterior)."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, 9))
    x[:, 0] = dist * np.exp(rng.normal(0, 0.05, n))
    x[:, 1] = f0_hz * 1e3 + rng.normal(0, 0.05, n) * DF * 1e3
    x[:, 2] = mc * np.exp(rng.normal(0, 0.02, n))
    x[:, 3] = (1.0 + rng.normal(0, 0.1, n)) % (2 * np.pi)
    x[:, 4] = np.clip(0.3 + rng.normal(0, 0.05, n), -0.99, 0.99)
    x[:, 5] = (0.7 + rng.normal(0, 0.05, n)) % np.pi
    x[:, 6] = (2.0 + rng.normal(0, 0.05, n)) % (2 * np.pi)
    x[:, 7] = np.clip(0.2 + rng.normal(0, 0.05, n), -0.99, 0.99)
    x[:, 8] = rng.normal(0, 0.05, n)
    return x


def _finds(walker, srcs):
    """A walker record with ``srcs`` = [(f0_hz, band, opt, edge, seed)]."""
    rec = ps.empty_finds(walker)
    fl = []
    for f0, band, opt, edge, seed in srcs:
        s = _blob(f0, seed=seed)
        fl.append(dict(band=band, f0_hz=f0, fdot=1e-17, F1=0.5 * opt ** 2,
                       lnl2=0.5 * opt ** 2, opt_snr=opt, det_snr=opt,
                       opt_med=opt, det_med=opt, edge=edge, samples=s,
                       best=s[0]))
    ps._pack_finds(rec, fl)
    rec["bands"] = np.arange(4)
    rec["steps"] = np.array([100, 120, 140, 160])
    rec["converged"] = np.ones(4, dtype=bool)
    rec["band_finds"] = np.bincount([f[1] for f in srcs], minlength=4)
    return rec


# ---------------------------------------------------------------------------
class KnobTest(unittest.TestCase):
    def test_env_names_are_the_capitalised_attribute_names(self):
        for field, env, _cast in ps.ParaSearchConfig._ENV:
            self.assertEqual(env, "GB_PARA_SEARCH_" + field.upper())

    def test_defaults_match_the_ruling(self):
        with _clean_env():
            c = ps.ParaSearchConfig.from_env()
        self.assertEqual((c.every_pe, c.ntemps, c.nwalkers, c.converge_window,
                          c.min_steps, c.max_steps, c.stage2_steps),
                         (100, 8, 16, 50, 100, 600, 100))
        self.assertEqual((c.snr_min, c.det_frac, c.converge_tol, c.split_bins),
                         (5.0, 0.7, 0.5, 4.0))
        self.assertEqual((c.scorer, c.init), ("auto", "comb"))

    def test_gbsettings_fields_resolve_and_feed_the_config(self):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        with _clean_env():
            off = GBSettings()
        self.assertFalse(off.para_search)
        with _clean_env(), _env(GB_PARA_SEARCH=1, GB_PARA_SEARCH_NTEMPS=6,
                                GB_PARA_SEARCH_NWALKERS=12,
                                GB_PARA_SEARCH_SNR_MIN=7.0,
                                GB_PARA_SEARCH_SPLIT_BINS=3.5,
                                GB_PARA_SEARCH_EVERY_PE=150,
                                GB_PARA_SEARCH_CONVERGE_TOL=0.25):
            on = GBSettings()
        self.assertTrue(on.para_search)
        c = ps.ParaSearchConfig.from_settings(on)
        self.assertEqual((c.ntemps, c.nwalkers, c.snr_min, c.split_bins,
                          c.every_pe, c.converge_tol), (6, 12, 7.0, 3.5, 150, 0.25))
        # the settings tree must survive deepcopy + pickle (sprint rule)
        pickle.loads(pickle.dumps(copy.deepcopy(on)))
        pickle.loads(pickle.dumps(copy.deepcopy(c)))

    def test_explicit_kwarg_beats_env(self):
        with _env(GB_PARA_SEARCH_NTEMPS=6):
            c = ps.ParaSearchConfig.from_env(ntemps=3)
        self.assertEqual(c.ntemps, 3)

    def test_validation(self):
        for bad in (dict(nwalkers=7), dict(nwalkers=2), dict(min_steps=10, max_steps=5),
                    dict(scorer="fd"), dict(init="grid"), dict(det_frac=1.5),
                    dict(stage2_steps=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ps.ParaSearchConfig(**bad)

    def test_detection_gates(self):
        c = ps.ParaSearchConfig(snr_min=5.0, det_frac=0.7)
        self.assertTrue(ps.passes_gate_a(12.5, c))            # sqrt(2 F) = 5
        self.assertFalse(ps.passes_gate_a(12.4, c))
        self.assertTrue(ps.passes_gate_b(5.0, 3.5, c))
        self.assertFalse(ps.passes_gate_b(4.99, 4.99, c))     # opt below the floor
        self.assertFalse(ps.passes_gate_b(6.0, 4.1, c))       # det < 0.7 opt
        c7 = ps.ParaSearchConfig(snr_min=7.0)
        self.assertFalse(ps.passes_gate_a(18.0, c7))          # SNR 6: out at 7

    def test_stage_gate_is_pe_only(self):
        self.assertTrue(ps.para_search_stage_ok("pe", "full_pe"))
        self.assertFalse(ps.para_search_stage_ok("gb_search", "gb_search_2"))
        self.assertFalse(ps.para_search_stage_ok("pe", "replica_pe"))
        self.assertFalse(ps.para_search_stage_ok(None, None))


# ---------------------------------------------------------------------------
class ClusterAndGroupTest(unittest.TestCase):
    def test_two_sources_eight_bins_apart_split_at_four(self):
        rng = np.random.default_rng(0)
        f = np.concatenate([rng.normal(0, 0.5, 300), rng.normal(8, 0.5, 30)])
        s = np.concatenate([24 - (f[:300]) ** 2, 18 - (f[300:] - 8) ** 2])
        lab, cen = ps.greedy_peaks(f, s, 4.0, min_score=12.5)
        self.assertEqual(len(cen), 2)
        self.assertAlmostEqual(f[cen[0]], 0.0, delta=1.0)
        self.assertAlmostEqual(f[cen[1]], 8.0, delta=1.0)
        self.assertTrue(np.all(lab[:300] == 0))

    def test_close_sources_merge_and_the_gate_stops_the_scan(self):
        f = np.array([0.0, 0.5, 2.0, 2.5, 10.0])
        s = np.array([30.0, 29.0, 20.0, 19.0, 5.0])
        lab, cen = ps.greedy_peaks(f, s, 4.0, min_score=12.5)
        self.assertEqual(cen, [0])                 # 2 bins away = same cluster
        self.assertEqual(lab[-1], -1)              # below the gate: unassigned
        lab, cen = ps.greedy_peaks(f, s, 1.0, min_score=12.5)
        self.assertEqual(cen, [0, 2])

    def test_max_peaks(self):
        f = np.arange(10) * 10.0
        s = 100.0 - np.arange(10)
        _, cen = ps.greedy_peaks(f, s, 1.0, max_peaks=3)
        self.assertEqual(len(cen), 3)

    def test_band_groups(self):
        edges = np.arange(21) * 100 * DF
        # chunked: one group per contiguous run
        g = ps.band_groups(edges, [1, 2, 3, 7, 8], spacing_hz=0.0)
        self.assertEqual([list(x) for x in g], [[1, 2, 3], [7, 8]])
        # sig-het: (hi - lo) / spacing + 3 references must fit one block
        g = ps.band_groups(edges, range(1, 19), spacing_hz=10 * DF, ref_block=33)
        for grp in g:
            span = edges[grp[-1] + 1] - edges[grp[0]]
            self.assertLessEqual(span / (10 * DF) + 3, 33)
        self.assertEqual(sum(len(x) for x in g), 18)
        self.assertEqual(max(len(x) for x in g), 3)
        g = ps.band_groups(edges, range(1, 19), spacing_hz=10 * DF, group_bands=5)
        self.assertEqual([len(x) for x in g], [5, 5, 5, 3])


class BandBoxTest(unittest.TestCase):
    def test_round_trip_and_density(self):
        box = ps.BandBoxPrior([1.0, 5.0], [2.0, 9.0], [0.1, 0.1], [1.1, 0.6])
        rng = np.random.default_rng(1)
        u = rng.uniform(size=(2, 3, 4, 9))
        x = u.copy()
        box.transform_from_prior_basis(x, None)
        self.assertTrue(np.all((x[0, ..., 1] >= 1) & (x[0, ..., 1] <= 2)))
        self.assertTrue(np.all((x[1, ..., 1] >= 5) & (x[1, ..., 1] <= 9)))
        y = x.copy()
        box.transform_to_prior_basis(y, None)
        np.testing.assert_allclose(y, u, rtol=0, atol=1e-12)
        lp = np.zeros((2, 3, 4))
        box.adjust_logp(lp, None)
        np.testing.assert_allclose(lp[0], -np.log(1.0) - np.log(1.0))
        np.testing.assert_allclose(lp[1], -np.log(4.0) - np.log(0.5))
        # the running-group index subset addresses the right boxes
        lp = np.zeros((1, 3, 4))
        box.adjust_logp(lp, np.array([1]))
        np.testing.assert_allclose(lp[0], -np.log(4.0) - np.log(0.5))


# ---------------------------------------------------------------------------
class _FakeBranch:
    def __init__(self, coords):
        self.coords = coords


class _FakeState:
    def __init__(self, ll, coords):
        self.log_like = ll
        self.branches = {"gb": _FakeBranch(coords)}
        self.groups_running = np.ones(ll.shape[0], dtype=bool)


class TrackerTest(unittest.TestCase):
    def test_per_group_stop_rules(self):
        ng, nt, nw = 3, 2, 4
        tr = ps._Tracker(ng, nt, nw, 9, window=5, tol=0.5, min_steps=10,
                         max_steps=40, keep=5, n_rungs=2)
        st = _FakeState(np.zeros((ng, nt, nw)), np.zeros((ng, nt, nw, 9)))
        tr.seed(st)
        stop_at = {}
        for i in range(40):
            # g0 improves by 1 nat/step for 20 steps then plateaus; g1 is flat
            # from the start; g2 improves forever (hits max_steps)
            st.log_like[0] = min(i + 1, 20)
            st.log_like[2] = i + 1.0
            st.branches["gb"].coords[:] = i
            done = tr(i, st, None)
            for g in range(ng):
                if not st.groups_running[g] and g not in stop_at:
                    stop_at[g] = i + 1
            if done:
                break
        self.assertEqual(stop_at[1], 10)                  # min_steps
        self.assertEqual(stop_at[0], 25)                  # plateau at 20 + window 5
        self.assertEqual(stop_at[2], 40)                  # max_steps
        np.testing.assert_array_equal(tr.converged, [True, True, False])
        np.testing.assert_array_equal(tr.steps, [25, 10, 40])
        # a stopped group's ring is frozen at its last kept steps
        x, _ = tr.pool(1)
        self.assertEqual(sorted(set(x[:, 0].tolist())), [5.0, 6.0, 7.0, 8.0, 9.0])

    def test_the_sampler_batch_shrinks_as_groups_converge(self):
        """Real ParaEnsembleSampler + BandBoxPrior + the tracker: a flat group
        stops at min_steps while a narrow-peak group keeps running, and from
        then on the likelihood is handed fewer rows per call."""
        from eryn.paraensemble import ParaEnsembleSampler
        from eryn.prior import ProbDistContainer, UniformDistribution
        from eryn.state import ParaState

        rows = []

        def lnl(x):
            rows.append(x.shape[0])
            return np.where(x[:, 1] < 5.0, -0.5 * ((x[:, 1] - 0.5) / 1e-3) ** 2, 0.0)

        prior = ProbDistContainer({i: UniformDistribution(0.0, 1.0) for i in range(3)})
        box = ps.BandBoxPrior([0.0, 10.0], [1.0, 11.0], [0.0, 0.0], [1.0, 1.0])
        tr = ps._Tracker(2, 2, 8, 3, window=5, tol=0.5, min_steps=8,
                         max_steps=60, keep=5, n_rungs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            s = ParaEnsembleSampler(3, 8, 2, lnl, {"gb": prior},
                                    tempering_kwargs=dict(ntemps=2, Tmax=np.inf),
                                    prior_transform_fn=box, name="gb",
                                    stopping_fn=tr, stopping_iterations=1)
        with ps.seeded_global_rng(3):
            start = prior.rvs(size=(2, 2, 8))
            box.transform_from_prior_basis(start, None)
            state = ParaState({"gb": start}, groups_running=np.ones(2, dtype=bool))
            state.log_prior = s.compute_log_prior(state.branches_coords,
                                                  groups_running=np.arange(2))
            state.log_like = s.compute_log_like(state.branches_coords,
                                                groups_running=np.arange(2),
                                                logp=state.log_prior)
            tr.seed(state)
            rows.clear()
            s.run_mcmc(state, 60, burn=0, store=False)
        self.assertEqual(tr.steps[1], 8)                  # the flat group
        self.assertGreater(tr.steps[0], 12)
        # each call scores the in-prior proposals of 2 rungs x 4 walkers of
        # every RUNNING group: up to 16 rows while both run, at most 8 once
        # the flat group is off
        self.assertGreater(max(rows[:2 * 8]), 8)
        self.assertLessEqual(max(rows[2 * 8:]), 8)
        self.assertGreater(len(rows), 2 * 12)


# ---------------------------------------------------------------------------
class MergeTest(unittest.TestCase):
    def test_duplicates_across_walkers_and_band_edges(self):
        a = 3.0e-3
        w0 = _finds(0, [(a, 1, 9.0, False, 1), (a + 30 * DF, 2, 6.0, False, 2)])
        # walker 1: the same source 0.5 bin off; an edge-pinned copy of it in
        # the neighbouring band; and a source of its own 10 bins away
        w1 = _finds(1, [(a + 0.5 * DF, 1, 8.0, False, 3),
                        (a + 1.0 * DF, 2, 5.5, True, 4),
                        (a + 10 * DF, 1, 7.0, False, 5)])
        src = ps.merge_finds([w0, w1], tobs=TOBS, radius_bins=4.0, seed=0)
        self.assertEqual(len(src), 3)
        s0 = src[0]
        self.assertEqual(s0["walkers"], [0, 1])
        self.assertEqual(s0["n_finds"], 2)                 # the pinned copy dropped
        self.assertEqual(s0["n_pinned_dropped"], 1)
        self.assertEqual(s0["samples"].shape[0], 2 * 400)  # equal weight per walker
        self.assertAlmostEqual(s0["opt_snr"], 9.0)
        self.assertEqual(src[1]["walkers"], [1])
        self.assertEqual(src[2]["walkers"], [0])

    def test_same_f0_far_fdot_stays_two_sources(self):
        w0 = _finds(0, [(3e-3, 1, 9.0, False, 1)])
        w1 = _finds(1, [(3e-3, 1, 9.0, False, 2)])
        w1["fdot"][:] = 1e-17 + 100.0 / TOBS ** 2
        self.assertEqual(len(ps.merge_finds([w0, w1], tobs=TOBS, radius_bins=4.0)), 2)

    def test_no_finds(self):
        self.assertEqual(ps.merge_finds([ps.empty_finds(0)], tobs=TOBS, radius_bins=4.0), [])


class EpochTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="para_epoch_")
        self.addCleanup(shutil.rmtree, self.d, True)

    def _write(self, k, walkers):
        src = ps.merge_finds(walkers, tobs=TOBS, radius_bins=4.0, seed=k)
        from lisatools.globalfit.warmstart.self_refit import epoch_dir

        return ps.write_epoch(epoch_dir(self.d, k), src, walkers, tobs=TOBS,
                              transform_fn=_transform(),
                              manifest_extra=dict(epoch=k, built_at_clock=10 * k))

    def test_npz_loads_through_warmstart_components_unchanged(self):
        from lisatools.globalfit.warmstart.proposal import WarmStartComponents
        from lisatools.globalfit.warmstart.self_refit import latest_complete_epoch

        w0 = _finds(0, [(3e-3, 1, 9.0, False, 1), (3e-3 + 30 * DF, 2, 6.0, False, 2)])
        w1 = _finds(1, [(3e-3 + 0.5 * DF, 1, 8.0, False, 3)])
        man = self._write(0, [w0, w1])
        self.assertEqual(man["n_sources"], 2)
        self.assertEqual(man["finds_per_walker"], {"0": 2, "1": 1})
        found = latest_complete_epoch(self.d)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], 0)
        box = ([0.001, 2.9, 0.001, 0, -1, 0, 0, -1, -1.0],
               [40.0, 3.1, 1.0, 2 * np.pi, 1, np.pi, 2 * np.pi, 1, 1.0])
        c = WarmStartComponents.from_npz(found[1]["npz_path"], new_tobs=TOBS,
                                         floor_box=box, floor_eps=0.05,
                                         circ_images=3, seed=4)
        self.assertEqual(c.basis, "observable")
        x = np.asarray(c.rvs(size=400))
        self.assertEqual(x.shape, (400, 9))
        lp = np.asarray(c.logpdf(x))
        self.assertTrue(np.all(np.isfinite(lp)))
        # draws land on the two sources (the floor takes ~5 %)
        near = (np.abs(x[:, 1] - 3.0) < 3 * DF * 1e3) | (np.abs(x[:, 1] - (3.0 + 30 * DF * 1e3)) < 3 * DF * 1e3)
        self.assertGreater(near.mean(), 0.85)
        # the raw finds ride along
        recs = ps.load_finds_npz(os.path.join(found[1]["npz_path"][:-len(ps.EPOCH_NPZ)],
                                              ps.FINDS_NPZ))
        self.assertEqual([int(r["walker"]) for r in recs], [0, 1])

    def test_zero_sources_is_recorded_but_not_complete(self):
        from lisatools.globalfit.warmstart.self_refit import latest_complete_epoch

        man = self._write(0, [ps.empty_finds(0)])
        self.assertIsNone(man)
        self.assertIsNone(latest_complete_epoch(self.d))
        with open(os.path.join(self.d, "epoch_0000", "DONE.json")) as fh:
            self.assertEqual(json.load(fh)["n_sources"], 0)


class HolderTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="para_holder_")
        self.addCleanup(shutil.rmtree, self.d, True)
        from lisatools.globalfit.warmstart.self_refit import epoch_dir

        for k, f in ((0, 3e-3), (1, 3.0002e-3)):
            w = _finds(0, [(f, 1, 9.0, False, k)])
            src = ps.merge_finds([w], tobs=TOBS, radius_bins=4.0)
            ps.write_epoch(epoch_dir(self.d, k), src, [w], tobs=TOBS,
                           transform_fn=_transform(),
                           manifest_extra=dict(epoch=k, built_at_clock=k))

    def _holder(self, seed=11):
        return ps.ParaSearchHolder(epoch_root=self.d, load_kwargs=dict(new_tobs=TOBS),
                                   build_seed=seed, tag=0x3A14)

    def test_poll_adopts_the_newest_and_the_directive_is_authoritative(self):
        h = self._holder()
        with self.assertRaises(RuntimeError):
            h.rvs(size=1)                                  # nothing adopted yet
        h.poll()
        self.assertEqual(h.directive(), {"epoch": 1})
        r = self._holder()
        r.apply_directive({"epoch": 0})                    # the head may lag
        self.assertEqual(r.epoch, 0)
        self.assertAlmostEqual(float(np.median(np.asarray(r.rvs(size=200))[:, 1])),
                               3.0, delta=1e-4)
        r.apply_directive({"epoch": None})
        self.assertIsNone(r.container)
        with self.assertRaises(RuntimeError):
            r.apply_directive({"epoch": 7})                # cannot load: refuse

    def test_seeds_are_per_epoch_and_reproducible(self):
        a, b = self._holder(), self._holder()
        a.apply_directive({"epoch": 1})
        b.apply_directive({"epoch": 1})
        np.testing.assert_array_equal(np.asarray(a.rvs(size=5)), np.asarray(b.rvs(size=5)))
        self.assertNotEqual(a.epoch_seed(0), a.epoch_seed(1))
        c = pickle.loads(pickle.dumps(copy.deepcopy(a)))
        self.assertEqual(c.epoch, 1)


class ClockAndRngTest(unittest.TestCase):
    def test_clock_counts_iteration_differences_and_survives_a_restart(self):
        d = tempfile.mkdtemp(prefix="para_clock_")
        self.addCleanup(shutil.rmtree, d, True)
        c = ps.ParaSearchClock(d)
        self.assertEqual(c.tick(0), 0)          # first tick of a process adds 0
        self.assertEqual(c.tick(5), 5)
        self.assertEqual(c.tick(5), 5)          # same iteration: no advance
        self.assertEqual(c.tick(12), 12)
        c2 = ps.ParaSearchClock(d)              # a restart: gf_iteration restarts at 0
        self.assertEqual(c2.tick(0), 12)
        self.assertEqual(c2.tick(3), 15)

    def test_global_rng_is_seeded_and_restored(self):
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        with ps.seeded_global_rng(9):
            a = np.random.uniform(size=3)
        np.testing.assert_array_equal(np.random.get_state()[1], before)
        with ps.seeded_global_rng(9):
            b = np.random.uniform(size=3)
        np.testing.assert_array_equal(a, b)


class ComposeTest(unittest.TestCase):
    def _recipe(self):
        from lisatools.globalfit.moves.globalfitmove import Move
        from lisatools.globalfit.recipe import Recipe, Stage

        return Recipe([
            Stage(name="gb_search_2", kind="gb_search",
                  moves=[Move("rj_fstat_search", branch="gb"),
                         Move("rj_prior_removal", branch="gb")]),
            Stage(name="full_pe", kind="pe",
                  moves=[Move("rj_warm_pe", branch="gb"), Move("rj_fstat_pe", branch="gb"),
                         Move("rj_prior_pe", branch="gb"), Move("gb_ridge_gibbs", branch="gb")]),
        ])

    def test_added_after_rj_prior_pe_in_pe_stages_only_and_idempotent(self):
        r = self._recipe()
        self.assertEqual(ps.compose_para_search(r), ["full_pe"])
        names = [m.name for m in r.stages[1].moves]
        self.assertEqual(names, ["rj_warm_pe", "rj_fstat_pe", "rj_prior_pe",
                                 ps.PARA_SEARCH_MOVE, "gb_ridge_gibbs"])
        self.assertNotIn(ps.PARA_SEARCH_MOVE, [m.name for m in r.stages[0].moves])
        self.assertEqual(ps.compose_para_search(r), [])
        self.assertIn(ps.PARA_SEARCH_MOVE, r.stock_names())

    def test_no_anchor_adds_nothing(self):
        from lisatools.globalfit.moves.globalfitmove import Move
        from lisatools.globalfit.recipe import Recipe, Stage

        r = Recipe([Stage(name="gb_pe", kind="pe", moves=[Move("rj_fstat_pe", branch="gb")])])
        with self.assertLogs("lisatools.globalfit.warmstart.para_search", "WARNING"):
            self.assertEqual(ps.compose_para_search(r), [])
        self.assertNotIn(ps.PARA_SEARCH_MOVE, r.move_names())


# ---------------------------------------------------------------------------
class _StubMove:
    temperature_control = None
    periodic = None

    def __init__(self, name, log):
        self.gf_move_name = name
        self.log = log
        self.accepted = np.zeros((2, 4))

    def propose(self, model, state):
        self.log.append(self.gf_move_name)
        return state, np.ones((2, 4))


class _StubState:
    sub_states = None


PE7 = ("rj_warm_pe", "rj_fstat_pe", "rj_fstat_pe_residual", "rj_prior_pe",
       ps.PARA_SEARCH_MOVE, "gb_ridge_gibbs")
RJ5 = PE7[:5]


def _combine(names, log):
    from lisatools.globalfit.moves.globalfitmove import GFCombineMove

    moves = [_StubMove(n, log) for n in names]
    comb = GFCombineMove(moves=moves, share_temperature_control=False,
                         weighted_cycle=True)
    comb.gf_stage_kind = "pe"
    comb.gf_stage_name = "full_pe"
    return comb


def _picks(names, n, seed, **env):
    log = []
    comb = _combine(names, log)
    model = SimpleNamespace(random=np.random.RandomState(seed))
    out = []
    with _env(GB_PE_RJ_DRAW_ONE=1, **env):
        for _ in range(n):
            log.clear()
            comb.propose(model, _StubState())
            out.append(tuple(log))
    return out


class DrawOneTest(unittest.TestCase):
    def test_para_joins_the_draw_last_at_its_fraction(self):
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        self.assertEqual(GFCombineMove._PE_RJ_DRAW_ONE_MEMBERS[-1], ps.PARA_SEARCH_MOVE)
        out = _picks(PE7, 4000, 5, GB_PE_RJ_FSTAT_FRACTION=0.4,
                     GB_PE_RJ_WARM_FRACTION=0.1, GB_PE_RJ_PARA_FRACTION=0.2)
        rj = [[x for x in o if x in RJ5] for o in out]
        self.assertTrue(all(len(r) == 1 for r in rj))
        p = np.mean([r[0] == ps.PARA_SEARCH_MOVE for r in rj])
        pr = np.mean([r[0] == "rj_prior_pe" for r in rj])
        self.assertTrue(0.17 < p < 0.23, p)                 # 4000 draws: 5 sigma
        self.assertTrue(0.27 < pr < 0.33, pr)               # 1 - 0.4 - 0.1 - 0.2

    def test_zero_fraction_keeps_it_outside_the_draw_and_the_draws_unchanged(self):
        with_para = _picks(PE7, 300, 9, GB_PE_RJ_FSTAT_FRACTION=0.45,
                           GB_PE_RJ_WARM_FRACTION=0.1)
        without = _picks(tuple(n for n in PE7 if n != ps.PARA_SEARCH_MOVE), 300, 9,
                         GB_PE_RJ_FSTAT_FRACTION=0.45, GB_PE_RJ_WARM_FRACTION=0.1)
        # at 0 the move runs every iteration in its slot; the drawn RJ move
        # sequence is EXACTLY the one of a stage without it
        self.assertTrue(all(ps.PARA_SEARCH_MOVE in o for o in with_para))
        strip = [tuple(x for x in o if x != ps.PARA_SEARCH_MOVE) for o in with_para]
        self.assertEqual(strip, without)

    def test_fractions_over_one_raise(self):
        log = []
        comb = _combine(PE7, log)
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.6,
                  GB_PE_RJ_WARM_FRACTION=0.1, GB_PE_RJ_PARA_FRACTION=0.35):
            with self.assertRaises(ValueError):
                comb.propose(SimpleNamespace(random=np.random.RandomState(1)), _StubState())


# ---------------------------------------------------------------------------
def _skeleton(root, *, finds_fn=None, kind="pe", stage="full_pe", every=2):
    """A GBParaSearchRJMove without __init__: just what setup() / the build /
    the rank apply read. The search body is replaced by ``finds_fn``."""
    from lisatools.globalfit.moves import gbspecialstretch as gbs

    class _Skel(gbs.GBParaSearchRJMove):
        backend = SimpleNamespace(uses_cupy=False, uses_cuda=False)

        def _gb_serve_para_search(self, payload, clock, model):
            self.served.append(payload)
            return {"finds": [finds_fn(w) for w in range(payload["w0"], payload["w1"])],
                    "errors": [], "rank": None}

    m = object.__new__(_Skel)
    m.name = m.gf_move_name = ps.PARA_SEARCH_MOVE
    m.branch_name = "gb"
    m.fanout = None
    m.gf_rank = None
    m.nwalkers = 2
    m.gf_stage_kind = kind
    m.gf_stage_name = stage
    m.gf_iteration = 0
    m.transform_fn = _transform()
    m._basis_settings = SimpleNamespace(Tobs=TOBS)
    m.served = []
    m._para_search_cfg = ps.ParaSearchConfig(every_pe=every)
    m._para_search_root = root
    m._para_holder = ps.ParaSearchHolder(epoch_root=root,
                                         load_kwargs=dict(new_tobs=TOBS),
                                         build_seed=5, tag=0x3A14,
                                         name=ps.PARA_SEARCH_MOVE)
    m._para_build_seed = 5
    m._para_clock = None
    m._para_last_attempt = None
    m._para_search_directive = None
    m._para_stage_warned = False
    m._para_epochs_built = 0
    m.rj_proposal_distribution = None
    return m


class MoveWiringTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="para_move_")
        self.addCleanup(shutil.rmtree, self.d, True)
        self.root = os.path.join(self.d, "warmstart", "parasearch")

    def _finds(self, w):
        return _finds(w, [(3e-3 + 0.3 * w * DF, 1, 9.0, False, w),
                          (3e-3 + 30 * DF, 2, 6.0, False, 10 + w)])

    def test_first_entry_builds_and_the_build_propose_draws_nothing(self):
        m = _skeleton(self.root, finds_fn=self._finds)
        m.setup(None, None)
        self.assertEqual(len(m.served), 1)                 # built at the first entry
        self.assertEqual(m.served[0]["w0"], 0)
        self.assertEqual(m.served[0]["w1"], 2)             # single process: every walker
        self.assertIsNone(m.rj_proposal_distribution)      # the one-draw lag
        self.assertIsNone(m._para_search_directive)
        with open(os.path.join(self.root, "epoch_0000", "DONE.json")) as fh:
            man = json.load(fh)
        self.assertEqual(man["n_sources"], 2)              # walker duplicates merged
        # the next draw adopts it; no rebuild before the cadence
        m.gf_iteration = 1
        m.setup(None, None)
        self.assertEqual(len(m.served), 1)
        self.assertEqual(m._para_search_directive, {"epoch": 0})
        self.assertIs(m.rj_proposal_distribution["gb"], m._para_holder)
        # cadence 2: the build at clock 2 ships epoch 0 (lag), the next adopts 1
        m.gf_iteration = 2
        m.setup(None, None)
        self.assertEqual(len(m.served), 2)
        self.assertEqual(m._para_search_directive, {"epoch": 0})
        m.gf_iteration = 3
        m.setup(None, None)
        self.assertEqual(m._para_search_directive, {"epoch": 1})

    def test_a_restart_keeps_the_cadence(self):
        m = _skeleton(self.root, finds_fn=self._finds, every=3)
        m.setup(None, None)
        m.gf_iteration = 2
        m.setup(None, None)
        # restart: a fresh move, gf_iteration from 0 again; the journal says 2
        m2 = _skeleton(self.root, finds_fn=self._finds, every=3)
        m2.setup(None, None)
        self.assertEqual(len(m2.served), 0)                # 2 < 3: not due
        self.assertEqual(m2._para_search_directive, {"epoch": 0})
        m2.gf_iteration = 1
        m2.setup(None, None)                               # clock 3: due
        self.assertEqual(len(m2.served), 1)

    def test_search_stages_propose_nothing_and_never_build(self):
        m = _skeleton(self.root, finds_fn=self._finds, kind="gb_search",
                      stage="gb_search_2")
        m.setup(None, None)
        self.assertEqual(m.served, [])
        self.assertIsNone(m.rj_proposal_distribution)
        self.assertFalse(os.path.isdir(self.root))

    def test_a_failed_build_never_raises_and_retries_after_the_cadence(self):
        def boom(w):
            raise RuntimeError("no residual")

        m = _skeleton(self.root, finds_fn=boom, every=2)
        with self.assertLogs("lisatools.globalfit.moves.gbspecialstretch", "ERROR"):
            m.setup(None, None)
        self.assertIsNone(m.rj_proposal_distribution)
        m.gf_iteration = 1
        m.setup(None, None)
        self.assertEqual(len(m.served), 1)                 # not retried yet

    def test_rank_apply_holds_exactly_the_heads_epoch(self):
        m = _skeleton(self.root, finds_fn=self._finds)
        m.setup(None, None)
        r = _skeleton(self.root, finds_fn=self._finds)
        r._para_search_apply({"epoch": 0})
        self.assertEqual(r._para_holder.epoch, 0)
        self.assertIs(r.rj_proposal_distribution["gb"], r._para_holder)
        r._para_search_apply(None)
        self.assertIsNone(r.rj_proposal_distribution)

    def test_the_replica_layout_is_refused(self):
        m = _skeleton(self.root)
        m._needs_even_walker_block = lambda: False
        curr = SimpleNamespace(
            fanout=SimpleNamespace(single=False, is_head=True,
                                   layout=SimpleNamespace(replica_mode=True)),
            rank=0)
        with self.assertRaises(NotImplementedError) as ctx:
            m.install_walker_fanout(curr)
        self.assertIn("GF_GPU_ROUTING", str(ctx.exception))
        curr.fanout.layout.replica_mode = False
        m.install_walker_fanout(curr)                      # the block layout is fine

    def test_only_the_para_move_serves_the_op(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        other = object.__new__(gbs.GBSpecialRJPriorMove)
        other.name = "rj_prior_pe"
        with self.assertRaises(ValueError):
            other.gf_serve(gbs.GB_PARA_SEARCH_OP, {}, {}, None)
        self.assertNotIn(gbs.GB_PARA_SEARCH_OP, gbs.GB_OPS)

    def test_the_propose_payload_and_the_rank_body_carry_the_directive(self):
        """The two seams the orchestrator closes over (closures, so this pins
        the source; the gated integration test runs them)."""
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        src = inspect.getsource(gbs.GBSpecialBase._propose_orchestrated)
        self.assertIn('"para_search": getattr(self, "_para_search_directive", None)', src)
        body = inspect.getsource(gbs.GBSpecialBase._gb_serve_run_proposal)
        self.assertLess(body.index("_para_search_apply"), body.index("rj_prop = ("))


if __name__ == "__main__":
    unittest.main()
