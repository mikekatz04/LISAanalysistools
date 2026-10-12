"""Step 1 of the F-stat grid memory plan (user ruling 2026-10-12): a
``StackedFStatProposal4D`` keeps its node grids after construction only
when a replace move will read them (the trilinear in-cell mode).

On the 9mo PE galaxy grid the node stack was 24.2 GB of the ~43 GB resident
per card, read by nothing in the v9 recipe (no replace move). The RJ birth
machinery draws and scores in the uniform mode from the cell log-weights
alone, so dropping the nodes is bit-identical there. What is tested:

* the resolver (recipe default, env override, restore);
* ``keep_nodes=False``: no node grids retained, draws and densities
  bit-identical to ``keep_nodes=True`` in the uniform mode, a trilinear
  request raises instead of silently sampling something else;
* ``keep_nodes=None`` follows the resolver through ``from_cache`` and both
  ``stacked_from_cache`` layouts (flat and grouped) -- the mutation guard:
  with the resolver at False and no explicit flag the nodes are gone;
* the recipe sets the resolver from "a replace move is composed".
"""

import inspect
import os
import unittest
from unittest import mock

import numpy as np

from lisatools.sampling import fstat_proposal as FP
from lisatools.sampling.fstat_proposal import (
    StackedFStatProposal4D,
    set_stacked_keep_nodes,
    stacked_from_cache,
    stacked_in_cell_mode,
    stacked_keep_nodes,
)


def _grid(seed=0, K=5, shape=(6, 3, 4, 4)):
    rng = np.random.default_rng(seed)
    logp = 3.0 * rng.normal(size=(K,) + shape)
    f0_los = np.linspace(2.0, 2.01, K)
    f0_dxs = np.full(K, 1.0e-4)
    mc_ax = np.linspace(0.1, 1.0, shape[1])
    alpha_ax = np.linspace(0.0, 2.0 * np.pi, shape[2])
    sin_delta_ax = np.linspace(-1.0, 1.0, shape[3])
    return dict(logp_grids=logp, f0_los=f0_los, f0_dxs=f0_dxs, mc_ax=mc_ax,
                alpha_ax=alpha_ax, sin_delta_ax=sin_delta_ax)


def _nodes_kept(stacked):
    return [ch.get("log_node") is not None for ch in stacked._chunks]


class _Restore(unittest.TestCase):
    def setUp(self):
        prev = FP._STACKED_KEEP_NODES
        self.addCleanup(set_stacked_keep_nodes, prev)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GB_FSTAT_KEEP_NODES", None)


class ResolverTest(_Restore):
    def test_default_keeps_the_nodes(self):
        set_stacked_keep_nodes(True)
        self.assertTrue(stacked_keep_nodes())

    def test_the_recipe_value_is_honoured_and_the_previous_one_returned(self):
        set_stacked_keep_nodes(True)
        self.assertTrue(set_stacked_keep_nodes(False))
        self.assertFalse(stacked_keep_nodes())
        self.assertFalse(set_stacked_keep_nodes(True))

    def test_env_overrides_either_way(self):
        set_stacked_keep_nodes(False)
        os.environ["GB_FSTAT_KEEP_NODES"] = "1"
        self.assertTrue(stacked_keep_nodes())
        set_stacked_keep_nodes(True)
        os.environ["GB_FSTAT_KEEP_NODES"] = "0"
        self.assertFalse(stacked_keep_nodes())
        os.environ["GB_FSTAT_KEEP_NODES"] = "maybe"       # not a switch: ignored
        self.assertTrue(stacked_keep_nodes())


class DropIsExactTest(_Restore):
    def setUp(self):
        super().setUp()
        self.g = _grid()

    def _pair(self, **kw):
        a = StackedFStatProposal4D(**self.g, seed=11, keep_nodes=True, **kw)
        b = StackedFStatProposal4D(**self.g, seed=11, keep_nodes=False, **kw)
        return a, b

    def test_no_node_grids_are_retained(self):
        a, b = self._pair()
        self.assertTrue(all(_nodes_kept(a)))
        self.assertFalse(any(_nodes_kept(b)))
        self.assertTrue(a.keep_nodes)
        self.assertFalse(b.keep_nodes)

    def test_draws_and_densities_are_bit_identical_in_the_uniform_mode(self):
        for kw in ({}, {"mem_budget_mb": 1e-3}):      # one chunk, and K-chunked
            a, b = self._pair(**kw)
            xa, xb = np.asarray(a.rvs(size=400)), np.asarray(b.rvs(size=400))
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(np.asarray(a.logpdf(xa)),
                                          np.asarray(b.logpdf(xa)))
            self.assertTrue(np.isfinite(np.asarray(b.logpdf(xa))).all())
            np.testing.assert_array_equal(a._log_norm, b._log_norm)
            for ca, cb in zip(a._chunks, b._chunks):
                np.testing.assert_array_equal(ca["log_wcell"], cb["log_wcell"])
                np.testing.assert_array_equal(ca["cdf"], cb["cdf"])

    def test_trilinear_raises_instead_of_sampling_something_else(self):
        a, b = self._pair()
        with stacked_in_cell_mode(a, "trilinear"):
            np.asarray(a.rvs(size=8))                 # the kept nodes serve it
        with stacked_in_cell_mode(b, "trilinear"):
            with self.assertRaises(RuntimeError):
                b.rvs(size=8)


class NoneFollowsTheResolverTest(_Restore):
    """``keep_nodes=None`` -- what every production path passes -- takes the
    resolver's answer at construction time, through every loader."""

    def setUp(self):
        super().setUp()
        self.g = _grid()

    def test_constructor(self):
        set_stacked_keep_nodes(True)
        self.assertTrue(all(_nodes_kept(StackedFStatProposal4D(**self.g))))
        set_stacked_keep_nodes(False)
        self.assertFalse(any(_nodes_kept(StackedFStatProposal4D(**self.g))))

    def test_from_cache_flat(self):
        set_stacked_keep_nodes(False)
        s = StackedFStatProposal4D.from_cache(self.g)
        self.assertFalse(any(_nodes_kept(s)))
        self.assertTrue(all(_nodes_kept(
            StackedFStatProposal4D.from_cache(self.g, keep_nodes=True))))

    def test_stacked_from_cache_flat_and_grouped(self):
        set_stacked_keep_nodes(False)
        # the loader refuses a cache whose axis-2 basis disagrees with
        # FSTAT_FDOT_AXIS; stamp the synthetic caches with the basis in force
        basis = "fdot" if FP.fdot_axis_on() else "Mc"
        flat = stacked_from_cache({**self.g, "grid_basis": np.array(basis)})
        self.assertFalse(any(_nodes_kept(flat)))
        g2 = _grid(seed=1, K=3)
        grouped = {
            "grid_basis": np.array(basis),
            "group_sizes": np.array([5, 3]),
            "f0_los": np.concatenate([self.g["f0_los"], g2["f0_los"]]),
            "f0_dxs": np.concatenate([self.g["f0_dxs"], g2["f0_dxs"]]),
            "logp_grids_g0": self.g["logp_grids"], "mc_ax_g0": self.g["mc_ax"],
            "logp_grids_g1": g2["logp_grids"], "mc_ax_g1": g2["mc_ax"],
            "alpha_ax": self.g["alpha_ax"], "sin_delta_ax": self.g["sin_delta_ax"],
        }
        comps = list(FP.iter_stacked_components(stacked_from_cache(grouped)))
        self.assertEqual(len(comps), 2)
        for c in comps:
            self.assertFalse(any(_nodes_kept(c)))
        # an explicit True wins over the resolver (the replace path)
        for c in FP.iter_stacked_components(stacked_from_cache(grouped, keep_nodes=True)):
            self.assertTrue(all(_nodes_kept(c)))

    def test_mutation_guard_the_env_switch_restores_the_old_footprint(self):
        set_stacked_keep_nodes(False)
        os.environ["GB_FSTAT_KEEP_NODES"] = "1"
        self.assertTrue(all(_nodes_kept(StackedFStatProposal4D(**self.g))))


class RecipeWiringTest(unittest.TestCase):
    def test_build_gb_moves_sets_the_resolver_from_the_replace_moves(self):
        from lisatools.globalfit import recipe as R

        src = inspect.getsource(R.build_gb_moves)
        i = src.index("set_stacked_keep_nodes(_keep_nodes)")
        self.assertIn("_keep_nodes = (gb_replace_move is not None) or "
                      "(gb_pe_replace_move is not None)", src[i - 400:i])
        # decided before the PE move list is assembled, i.e. before any
        # container can be built in this process
        self.assertLess(i, src.index("gb_pe_moves = [gb_pe_prior_move, gb_pe_residual_move]"))


if __name__ == "__main__":
    unittest.main()
