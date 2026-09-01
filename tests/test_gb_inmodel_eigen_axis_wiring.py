"""Wiring for the per-eigenaxis in-model proposal (GB_INMODEL_EIGEN_AXIS).

The primitives are covered by ``test_gb_inmodel_eigen_axis``. This file
covers the parts that decide whether they are ever REACHED and whether the
production path is left alone -- the gate, the prior-width lookup and the
draw branch -- which is where a change like this actually goes wrong.

The methods are exercised against a stub ``self`` rather than a built move:
constructing a real ``GBSpecialStretchMove`` needs data, a backend and GPUs,
and none of that is what is under test here.
"""
import os
import types
import unittest
from unittest import mock

import numpy as np

from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    gb_prior_box_scales,
)

DIST, F0, MC, R = 0, 1, 2, 8
NDIM = 9


def _stub(**over):
    """Minimal stand-in carrying only what the methods under test read."""
    s = types.SimpleNamespace(
        xp=np,
        name="rj_fstat_search",
        branch_name="gb",
        _dist_col=DIST, _mc_col=MC, _fdot_astro_col=R, _f0_col=F0,
        _eigen_axis_min_dim=NDIM,
        _eigen_axis_widths_cache=None,
        _last_im_kind=None,
        _last_axis_pick=None,
        jump_factor=1.2,
        stretch_probability=0.0,
        time=0,
        _proposal_param_scales=np.ones(NDIM),
        gpu_priors={},
    )
    for k, v in over.items():
        setattr(s, k, v)
    # in_model_proposal calls these; bind the REAL methods so the draw
    # branch is gated by production logic, not by a test double. The
    # observable gate comes FIRST in in_model_proposal, so every test in
    # this file runs with GB_INMODEL_PROPOSAL=legacy -- these cover the
    # legacy path, and _stub carries no transform_fn / df to build a map
    # with anyway.
    s._eigen_axis_ready = lambda: GBSpecialStretchMove._eigen_axis_ready(s)
    s._observable_basis_ready = (
        lambda: GBSpecialStretchMove._observable_basis_ready(s))
    s._observable_map = lambda: GBSpecialStretchMove._observable_map(s)
    return s


READY = GBSpecialStretchMove._eigen_axis_ready
WIDTHS = GBSpecialStretchMove._eigen_axis_widths
PROPOSE = GBSpecialStretchMove.in_model_proposal


class GateTest(unittest.TestCase):
    """The gate must be OFF by default and refuse incomplete bases."""

    def test_off_by_default_even_with_a_complete_basis(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_EIGEN_AXIS", None)
            self.assertFalse(READY(_stub()))

    def test_on_when_armed_and_basis_complete(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_EIGEN_AXIS": "1",
                                          "GB_INMODEL_PROPOSAL": "legacy"}):
            self.assertTrue(READY(_stub()))

    def test_refuses_a_basis_without_the_fiber_columns(self):
        """VGB / 8-column bases have no dist / Mc / r -- must stay joint."""
        with mock.patch.dict(os.environ, {"GB_INMODEL_EIGEN_AXIS": "1",
                                          "GB_INMODEL_PROPOSAL": "legacy"}):
            for missing in ("_dist_col", "_mc_col", "_fdot_astro_col",
                            "_f0_col"):
                self.assertFalse(READY(_stub(**{missing: None})),
                                 f"{missing}=None must disable the path")

    def test_missing_attribute_entirely_does_not_raise(self):
        """A move built before this change has no _dist_col at all."""
        s = _stub()
        del s._dist_col
        with mock.patch.dict(os.environ, {"GB_INMODEL_EIGEN_AXIS": "1",
                                          "GB_INMODEL_PROPOSAL": "legacy"}):
            self.assertFalse(READY(s))


class PriorWidthTest(unittest.TestCase):
    def test_reads_the_real_eryn_uniform_bounds(self):
        """Regression: eryn exposes minimum/maximum, NOT min_val/max_val.

        Reading the wrong attribute silently yields unit widths, which
        removes the prior bound with no runtime signal at all.
        """
        pri = ProbDistContainer({
            0: uniform_dist(0.1, 30.0),
            1: uniform_dist(3.0, 22.0),
            2: uniform_dist(1e-3, 1.0),
        })
        w = WIDTHS(_stub(gpu_priors={"gb": pri}), NDIM)
        self.assertAlmostEqual(float(w[0]), 29.9, places=6)
        self.assertAlmostEqual(float(w[1]), 19.0, places=6)
        self.assertAlmostEqual(float(w[2]), 0.999, places=6)

    def test_unbounded_columns_fall_back_to_one(self):
        pri = ProbDistContainer({0: uniform_dist(0.0, 4.0)})
        w = WIDTHS(_stub(gpu_priors={"gb": pri}), NDIM)
        self.assertAlmostEqual(float(w[0]), 4.0, places=6)
        self.assertTrue(np.all(np.asarray(w)[1:] == 1.0))

    def test_a_broken_prior_warns_and_does_not_raise(self):
        class Boom:
            @property
            def priors_in(self):
                raise RuntimeError("no")
        w = WIDTHS(_stub(gpu_priors={"gb": Boom()}), NDIM)
        np.testing.assert_allclose(np.asarray(w), np.ones(NDIM))

    def test_result_is_cached_per_ndim(self):
        pri = ProbDistContainer({0: uniform_dist(0.0, 4.0)})
        s = _stub(gpu_priors={"gb": pri})
        first = WIDTHS(s, NDIM)
        s.gpu_priors = {}                      # would give ones if re-read
        np.testing.assert_allclose(np.asarray(WIDTHS(s, NDIM)),
                                   np.asarray(first))


class DrawBranchTest(unittest.TestCase):
    """The draw must use ONE axis when armed and all of them when not."""

    @staticmethod
    def _chol():
        """Column k = sigma_k * a_k, with distinguishable columns."""
        c = np.zeros((6, NDIM, NDIM))
        for i in range(6):
            c[i] = np.eye(NDIM) * (np.arange(NDIM) + 1.0)
        return c

    def _run(self, armed):
        coords = np.zeros((6, NDIM))
        bs = types.SimpleNamespace(friend_start_inds=None)
        env = {"GB_INMODEL_EIGEN_AXIS": "1" if armed else "0",
               "GB_INMODEL_PROPOSAL": "legacy"}
        with mock.patch.dict(os.environ, env):
            s = _stub()
            new, factors = PROPOSE(s, coords, self._chol(), bs,
                                   np.arange(6), None)
        return s, new, factors

    def test_armed_moves_exactly_one_column_per_source(self):
        s, new, _ = self._run(armed=True)
        for i in range(6):
            nz = np.flatnonzero(np.abs(new[i]) > 0)
            self.assertLessEqual(len(nz), 1,
                                 "a per-axis step must touch one coordinate")
        self.assertEqual(s._last_im_kind, "eigen_axis")
        self.assertIsNotNone(s._last_axis_pick)

    def test_unarmed_moves_every_coordinate(self):
        """The production joint draw must be untouched by this change."""
        s, new, _ = self._run(armed=False)
        moved = [int((np.abs(new[i]) > 0).sum()) for i in range(6)]
        self.assertTrue(all(m == NDIM for m in moved), moved)
        self.assertEqual(s._last_im_kind, "infomat")

    def test_factors_are_zero_in_both_branches(self):
        """Symmetric proposal => no MH correction, armed or not."""
        for armed in (True, False):
            _, _, f = self._run(armed=armed)
            np.testing.assert_allclose(f, 0.0)

    def test_armed_step_respects_the_picked_column_scale(self):
        """Step magnitude must come from the picked column, not another."""
        s, new, _ = self._run(armed=True)
        pick = np.asarray(s._last_axis_pick)
        for i in range(6):
            nz = np.flatnonzero(np.abs(new[i]) > 0)
            if len(nz):
                self.assertEqual(int(nz[0]), int(pick[i]))

    def test_narrow_basis_falls_back_to_the_joint_draw(self):
        """chol narrower than the 9-column basis must not take the path."""
        coords = np.zeros((3, 4))
        chol = np.repeat(np.eye(4)[None, ...], 3, axis=0)
        bs = types.SimpleNamespace(friend_start_inds=None)
        with mock.patch.dict(os.environ, {"GB_INMODEL_EIGEN_AXIS": "1",
                                          "GB_INMODEL_PROPOSAL": "legacy"}):
            s = _stub(_proposal_param_scales=np.ones(4))
            new, _ = PROPOSE(s, coords, chol, bs, np.arange(3), None)
        self.assertEqual(s._last_im_kind, "infomat")
        self.assertTrue(all(int((np.abs(new[i]) > 0).sum()) == 4
                            for i in range(3)))


class EndToEndContractTest(unittest.TestCase):
    """The two halves are coupled only through the array shape.

    ``_compute_proposal_cholesky`` returns ``axes * sigmas`` and
    ``in_model_proposal`` reads one column of it. Nothing type-checks that
    contract, so pin it against the REAL flagship Fisher.
    """

    def setUp(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            axis_prior_bounds, eigen_axis_set, gb_fiber_tangent,
            gb_lnfdot_gradient)
        d = os.path.join(os.path.dirname(__file__), "data",
                         "flagship_fisher.npz")
        if not os.path.exists(d):
            self.skipTest("flagship Fisher fixture not present")
        z = np.load(d)
        self.F, self.y = z["F"], z["y"]
        t = gb_fiber_tangent(self.y, DIST, MC, R)
        self.axes, self.sig = eigen_axis_set(
            self.F[None, ...], t, self.y, F0, MC, R, DIST, 7.776e6,
            sigma_max=np.inf)
        w = np.array([30.0, 19.0, 1.0, 2 * np.pi, 2.0, np.pi,
                      2 * np.pi, 2.0, 4.0])
        self.sig = np.minimum(self.sig, axis_prior_bounds(self.axes, w))
        self.grad = gb_lnfdot_gradient(self.y, F0, MC, R)[0]

    def test_step_magnitude_matches_the_picked_axis_sigma(self):
        chol = self.axes * self.sig[:, None, :]
        coords = np.zeros((1, NDIM))
        bs = types.SimpleNamespace(friend_start_inds=None)
        with mock.patch.dict(os.environ, {"GB_INMODEL_EIGEN_AXIS": "1",
                                          "GB_INMODEL_PROPOSAL": "legacy"}):
            s = _stub()
            for _ in range(60):
                new, _ = PROPOSE(s, coords.copy(), chol, bs, np.arange(1),
                                 None)
                k = int(np.asarray(s._last_axis_pick)[0])
                step = new[0] / s.jump_factor
                # the step must be parallel to axis k, with |step| <= a few
                # sigma_k (it is sigma_k times one standard normal)
                a = self.axes[0, :, k]
                proj = float(step @ a)
                np.testing.assert_allclose(step, proj * a, atol=1e-9,
                                           rtol=1e-6)
                self.assertLess(abs(proj), 6.0 * self.sig[0, k] + 1e-12)

    def test_the_ridge_axis_is_reachable_and_moves_fdot(self):
        """Picking the last column must actually change ln(fdot).

        The installed ridge is the ANALYTIC shear ridge, so what is
        asserted is that it is REACHABLE from the draw branch and that it
        carries real fdot motion -- not that it wins a score computed
        under the same ``F`` whose f0 block is the thing being routed
        around. The geometry itself is pinned in
        ``test_gb_inmodel_eigen_axis``.
        """
        k = self.axes.shape[-1] - 1
        ridge_motion = abs(float(self.grad @ self.axes[0, :, k])
                           * self.sig[0, k])
        self.assertGreater(ridge_motion, 0.0)
        self.assertTrue(np.isfinite(ridge_motion))


if __name__ == "__main__":
    unittest.main()


class AxisAcceptanceReportTest(unittest.TestCase):
    """Per-axis acceptance logging -- the diagnostic a pooled rate hides."""

    def _stub_with_acc(self):
        s = _stub()
        s._axis_acc_store = None
        s._axis_acc = GBSpecialStretchMove._axis_acc.fget(s)
        return s

    def test_accumulator_starts_at_zero_and_is_sized_by_min_dim(self):
        s = self._stub_with_acc()
        prop, good = s._axis_acc
        self.assertEqual(prop.shape, (NDIM,))
        self.assertEqual(float(prop.sum()), 0.0)
        self.assertEqual(float(good.sum()), 0.0)

    def test_report_is_a_noop_before_any_proposal(self):
        s = _stub(); s._axis_acc_store = None
        GBSpecialStretchMove._report_axis_acceptance(s)   # must not raise

    def test_report_logs_each_axis_and_resets(self):
        s = _stub()
        prop = np.zeros(NDIM); good = np.zeros(NDIM)
        prop[0], good[0] = 100.0, 20.0
        prop[NDIM - 1], good[NDIM - 1] = 50.0, 35.0      # the ridge axis
        s._axis_acc_store = [prop, good]
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "INFO") as cm:
            GBSpecialStretchMove._report_axis_acceptance(s)
        msg = "\n".join(cm.output)
        self.assertIn("a0 20/100 (0.2000)", msg)
        self.assertIn("a8(ridge) 35/50 (0.7000)", msg)
        self.assertIsNone(s._axis_acc_store, "must reset after reporting")
