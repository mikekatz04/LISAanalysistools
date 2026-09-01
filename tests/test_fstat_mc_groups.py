"""Banded Mc stacks for the F-stat stage-B grid (user ruling 2026-08-26).

At full band the auto Mc density (``fstat_n_mc``) is driven by the MAX
peak f0, so one rectangular stack over-resolves every low-frequency box
by up to ~30x. The fix: group the (f0-sorted) peak boxes by a doubling
ladder of Mc-node requirements, sweep/build one rectangular stack per
group with that group's own max RAW requirement, and expose the groups
through ``GroupedStackedFStatProposal`` -- an EXACT reweighting of the
single-stack mixture (group mass = summed box weights). Single-group
cases (every probe; pinned FSTAT_N_MC; FSTAT_MC_GROUPING=0) must remain
bit-identical to the historical single-stack path, including the legacy
npz cache format.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from lisatools.sampling.fstat_proposal import (
    GroupedStackedFStatProposal,
    StackedFStatProposal4D,
    fstat_n_mc,
    iter_stacked_components,
    stacked_from_cache,
)
from lisatools.sampling.fstat_gridfit import (
    enumerate_center_nodes,
    mc_ladder_levels,
    run_stacked_stage_b,
)

_ENV_KEYS = (
    "FSTAT_N_MC", "FSTAT_N_PER_AXIS", "FSTAT_N_ALPHA", "FSTAT_N_SINDELTA",
    "FSTAT_PEAK_HALF_MHZ", "FSTAT_PEAKS_TO_FIT", "FSTAT_MC_GROUPING",
    "FSTAT_PEAK_WEIGHTING", "FSTAT_PEAK_WEIGHT_ALPHA", "FSTAT_GRID_MEM_MB",
)


# These tests were written against the Mc-basis grid, which is what
# FSTAT_FDOT_AXIS=0 selects -- and that path must keep working, since it is
# the documented escape from the fdot axis and the way an in-flight run
# resumes across the change. Pin it for the whole module rather than
# letting the new default silently retarget them; the fdot basis has its
# own end-to-end coverage in test_fstat_fdot_birth.
_FDOT_AXIS_SAVED = None


def setUpModule():
    global _FDOT_AXIS_SAVED
    _FDOT_AXIS_SAVED = os.environ.get("FSTAT_FDOT_AXIS")
    os.environ["FSTAT_FDOT_AXIS"] = "0"


def tearDownModule():
    if _FDOT_AXIS_SAVED is None:
        os.environ.pop("FSTAT_FDOT_AXIS", None)
    else:
        os.environ["FSTAT_FDOT_AXIS"] = _FDOT_AXIS_SAVED


class _EnvMixin:
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ["FSTAT_N_ALPHA"] = "2"
        os.environ["FSTAT_N_SINDELTA"] = "2"
        os.environ["FSTAT_PEAK_HALF_MHZ"] = "0.0005"

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class McLadderTest(unittest.TestCase):
    def test_ladder_quantizes_and_covers(self):
        req = np.array([3, 3, 4, 7, 20, 90])
        lv = mc_ladder_levels(req)
        self.assertTrue(np.all(lv >= req))
        self.assertEqual(list(lv), [3, 3, 6, 12, 24, 96])

    def test_running_max_keeps_groups_contiguous(self):
        lv = mc_ladder_levels(np.array([3, 7, 4]))
        self.assertEqual(list(lv), [3, 12, 12])

    def test_uniform_requirement_is_one_group(self):
        lv = mc_ladder_levels(np.array([71, 71, 71]))
        self.assertEqual(len(set(lv.tolist())), 1)


def _mk_stack(f0_los, mc_n, weights=None, seed=0):
    """Tiny deterministic stack: n_f0=3, len(f0_los) boxes."""
    K = len(f0_los)
    rng = np.random.default_rng(seed)
    grids = rng.normal(size=(K, 3, mc_n, 2, 2))
    f0_los = np.asarray(f0_los, dtype=float)
    f0_dxs = np.full(K, 0.01)
    mc_ax = np.linspace(0.1, 1.0, mc_n)
    al_ax = np.array([0.0, 2 * np.pi])
    sd_ax = np.array([-1.0, 1.0])
    stack = StackedFStatProposal4D(
        grids, f0_los, f0_dxs, mc_ax, al_ax, sd_ax, weights=weights)
    return stack, dict(logp_grids=grids, f0_los=f0_los, f0_dxs=f0_dxs,
                       mc_ax=mc_ax, alpha_ax=al_ax, sin_delta_ax=sd_ax)


class GroupedProposalTest(unittest.TestCase):
    def _make(self):
        w = np.array([1.0, 2.0, 3.0])
        s0, d0 = _mk_stack([1.0, 1.1], 3, weights=w[:2], seed=1)
        s1, d1 = _mk_stack([2.0], 5, weights=w[2:], seed=2)
        return GroupedStackedFStatProposal([s0, s1], box_weights=w,
                                           seed=42), s0, s1, w

    def test_group_masses_from_box_weights(self):
        g, s0, s1, w = self._make()
        np.testing.assert_allclose(g.weights, [0.5, 0.5])
        self.assertEqual(g.K, 3)
        self.assertEqual(list(g.group_sizes), [2, 1])

    def test_logpdf_is_group_mixture(self):
        g, s0, s1, _ = self._make()
        pts = np.array([
            [1.005, 0.5, 1.0, 0.0],    # inside group 0 only
            [2.015, 0.5, 1.0, 0.0],    # inside group 1 only
            [5.0, 0.5, 1.0, 0.0],      # outside every box
        ])
        got = np.asarray(g.logpdf(pts))
        parts = np.stack([
            np.log(0.5) + np.asarray(s0.logpdf(pts)),
            np.log(0.5) + np.asarray(s1.logpdf(pts)),
        ])
        expect = np.logaddexp.reduce(parts, axis=0)
        self.assertTrue(np.isfinite(got[0]) and np.isfinite(got[1]))
        self.assertEqual(got[2], -np.inf)
        m = np.isfinite(expect)
        np.testing.assert_allclose(got[m], expect[m], rtol=1e-12)

    def test_exact_vs_single_stack_when_axes_match(self):
        """Same n_Mc in both groups -> grouped == single stack, exactly."""
        w = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(7)
        grids = rng.normal(size=(3, 3, 4, 2, 2))
        f0_los = np.array([1.0, 1.1, 2.0])
        f0_dxs = np.full(3, 0.01)
        mc_ax = np.linspace(0.1, 1.0, 4)
        al_ax = np.array([0.0, 2 * np.pi])
        sd_ax = np.array([-1.0, 1.0])
        single = StackedFStatProposal4D(
            grids, f0_los, f0_dxs, mc_ax, al_ax, sd_ax, weights=w)
        s0 = StackedFStatProposal4D(
            grids[:2], f0_los[:2], f0_dxs[:2], mc_ax, al_ax, sd_ax,
            weights=w[:2])
        s1 = StackedFStatProposal4D(
            grids[2:], f0_los[2:], f0_dxs[2:], mc_ax, al_ax, sd_ax,
            weights=w[2:])
        g = GroupedStackedFStatProposal([s0, s1], box_weights=w)
        rng2 = np.random.default_rng(3)
        pts = np.column_stack([
            rng2.uniform(0.99, 2.03, 64),
            rng2.uniform(0.1, 1.0, 64),
            rng2.uniform(0.0, 2 * np.pi, 64),
            rng2.uniform(-1.0, 1.0, 64),
        ])
        a = np.asarray(single.logpdf(pts))
        b = np.asarray(g.logpdf(pts))
        finite = np.isfinite(a) | np.isfinite(b)
        self.assertTrue(finite.any())
        np.testing.assert_allclose(
            np.where(np.isfinite(a), a, -1e300)[finite],
            np.where(np.isfinite(b), b, -1e300)[finite], rtol=1e-10)

    def test_rvs_and_draw_counts(self):
        g, *_ = self._make()
        x = np.asarray(g.rvs(size=200))
        self.assertEqual(x.shape, (200, 4))
        self.assertTrue(np.all(np.isfinite(x)))
        in_g0 = (x[:, 0] >= 1.0) & (x[:, 0] <= 1.1 + 0.02)
        in_g1 = (x[:, 0] >= 2.0) & (x[:, 0] <= 2.0 + 0.02)
        self.assertTrue(np.all(in_g0 | in_g1))
        c = g.pop_draw_counts()
        self.assertEqual(len(c), 3)
        self.assertEqual(int(c.sum()), 200)
        self.assertEqual(int(c[:2].sum()), int(in_g0.sum()))

    def test_rvs_per_box_concatenates(self):
        g, *_ = self._make()
        s = np.asarray(g.rvs_per_box(5))
        self.assertEqual(s.shape, (3, 5, 4))

    def test_iter_and_census_surfaces(self):
        g, s0, s1, _ = self._make()
        comps = list(iter_stacked_components(g))
        self.assertEqual({id(c) for c in comps}, {id(s0), id(s1)})
        self.assertTrue(hasattr(g, "pop_draw_counts"))
        np.testing.assert_allclose(g.f0_los, [1.0, 1.1, 2.0])
        self.assertEqual(len(g.f0_dxs), 3)


class FromCacheDispatchTest(unittest.TestCase):
    def test_legacy_cache_gives_plain_stack(self):
        _, d = _mk_stack([1.0, 1.1], 3)
        out = stacked_from_cache(d)
        self.assertIsInstance(out, StackedFStatProposal4D)

    def test_grouped_cache_roundtrip(self):
        w = np.array([1.0, 2.0, 3.0])
        s0, d0 = _mk_stack([1.0, 1.1], 3, weights=w[:2], seed=1)
        s1, d1 = _mk_stack([2.0], 5, weights=w[2:], seed=2)
        direct = GroupedStackedFStatProposal([s0, s1], box_weights=w)
        cache = dict(
            group_sizes=np.array([2, 1]),
            f0_los=np.concatenate([d0["f0_los"], d1["f0_los"]]),
            f0_dxs=np.concatenate([d0["f0_dxs"], d1["f0_dxs"]]),
            alpha_ax=d0["alpha_ax"], sin_delta_ax=d0["sin_delta_ax"],
            logp_grids_g0=d0["logp_grids"], mc_ax_g0=d0["mc_ax"],
            logp_grids_g1=d1["logp_grids"], mc_ax_g1=d1["mc_ax"],
        )
        out = stacked_from_cache(cache, weights=w)
        self.assertIsInstance(out, GroupedStackedFStatProposal)
        np.testing.assert_allclose(out.weights, direct.weights)
        pts = np.array([[1.005, 0.5, 1.0, 0.0], [2.015, 0.5, 1.0, 0.0]])
        np.testing.assert_allclose(np.asarray(out.logpdf(pts)),
                                   np.asarray(direct.logpdf(pts)), rtol=1e-12)


def _stub_call_fstat(pr):
    n = len(pr)
    return np.ones(n), np.ones(n)


def _flat_fstat(N, M):
    return N * 0 + 1.0


class StageBGroupingTest(_EnvMixin, unittest.TestCase):
    TOBS = 7776000.0
    MC_LIMS = [0.05, 1.0]
    BAND_EDGES = np.array([1.5e-3, 2.5e-3, 2.05e-2])

    def _peaks(self, mixed=True):
        rows = [[2.0, 10.0, 0.0, 0]]
        if mixed:
            rows += [[20.0, 50.0, 0.0, 1], [20.2, 30.0, 0.0, 1]]
        return np.asarray(rows, dtype=float)

    def _run(self, peaks, cache_path=None):
        with mock.patch(
                "lisatools.sampling.fstat_proposal.compute_fstat",
                side_effect=_flat_fstat):
            return run_stacked_stage_b(
                _stub_call_fstat, peaks, xp=np, Tobs=self.TOBS,
                band_edges_hz=self.BAND_EDGES, mc_lims=self.MC_LIMS,
                cache_path=cache_path)

    def test_mixed_band_builds_groups_with_raw_group_max(self):
        stacked = self._run(self._peaks())
        self.assertIsInstance(stacked, GroupedStackedFStatProposal)
        self.assertEqual(len(stacked.components), 2)
        self.assertEqual(list(stacked.group_sizes), [1, 2])
        lo, hi = self.MC_LIMS
        n_low = fstat_n_mc(2.0, lo, hi, self.TOBS)
        n_high = max(fstat_n_mc(20.0, lo, hi, self.TOBS),
                     fstat_n_mc(20.2, lo, hi, self.TOBS))
        self.assertEqual(len(stacked.components[0]._axes3[0]), n_low)
        self.assertEqual(len(stacked.components[1]._axes3[0]), n_high)

    def test_grouped_cache_written_and_reloadable(self):
        with tempfile.TemporaryDirectory() as td:
            cp = os.path.join(td, "fstat_grid.npz")
            live = self._run(self._peaks(), cache_path=cp)
            sp = cp.replace(".npz", "_peaks_stacked.npz")
            self.assertTrue(os.path.exists(sp))
            d = np.load(sp, allow_pickle=False)
            self.assertIn("group_sizes", d.files)
            self.assertIn("logp_grids_g0", d.files)
            self.assertNotIn("logp_grids", d.files)
            for key in ("peak_f0_mHz", "peak_F", "band_idx", "band_edges"):
                self.assertIn(key, d.files)
            # round-trip: reload with the SAME weighting the live fit used
            # (mirrors run_fstat_grid_fit's stacked-cache branch) -> the
            # reloaded proposal must match the live one exactly.
            from lisatools.sampling.fstat_gridfit import (
                peak_box_weights,
                peak_weight_alpha_env,
                peak_weight_cells_env,
            )
            w = peak_box_weights(
                np.clip(np.asarray(d["peak_F"], dtype=float), 0.0, None),
                peak_f0_mHz=d["peak_f0_mHz"], band_edges=d["band_edges"],
                alpha=peak_weight_alpha_env(None),
                cells=peak_weight_cells_env(), equal=False)
            reloaded = stacked_from_cache(d, weights=w)
            self.assertIsInstance(reloaded, GroupedStackedFStatProposal)
            pts = np.array([[2.0, 0.5, 1.0, 0.0], [20.0, 0.5, 1.0, 0.0]])
            np.testing.assert_allclose(
                np.asarray(reloaded.logpdf(pts)),
                np.asarray(live.logpdf(pts)), rtol=1e-10)
            # center-table enumeration reads the grouped cache
            nodes = enumerate_center_nodes(td, mc_lims=self.MC_LIMS)
            n_f0 = live.components[0]._node_shape[0]
            self.assertEqual(int(nodes["n_peak_nodes"]), 3 * n_f0)

    def test_single_band_keeps_legacy_stack_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            cp = os.path.join(td, "fstat_grid.npz")
            stacked = self._run(self._peaks(mixed=False), cache_path=cp)
            self.assertIsInstance(stacked, StackedFStatProposal4D)
            d = np.load(cp.replace(".npz", "_peaks_stacked.npz"),
                        allow_pickle=False)
            self.assertIn("logp_grids", d.files)
            self.assertNotIn("group_sizes", d.files)

    def test_grouping_off_restores_max_f0_single_stack(self):
        os.environ["FSTAT_MC_GROUPING"] = "0"
        stacked = self._run(self._peaks())
        self.assertIsInstance(stacked, StackedFStatProposal4D)
        lo, hi = self.MC_LIMS
        self.assertEqual(len(stacked._axes3[0]),
                         fstat_n_mc(20.2, lo, hi, self.TOBS))

    def test_pinned_n_mc_collapses_to_single_group(self):
        os.environ["FSTAT_N_MC"] = "4"
        stacked = self._run(self._peaks())
        self.assertIsInstance(stacked, StackedFStatProposal4D)
        self.assertEqual(len(stacked._axes3[0]), 4)


if __name__ == "__main__":
    unittest.main()
