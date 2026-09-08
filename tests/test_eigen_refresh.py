"""Tests for the per-leaf eigen-table refresh infrastructure.

``lisatools.globalfit.moves.eigen_refresh`` turns a likelihood (or a
waveform + inner product) into the ``(axes, sigmas)`` tables the Eryn
:class:`~eryn.moves.EigenAxisMove` consumes: information matrix in the
SAMPLING basis -> prior-box whitening -> ``eigen_axis_set`` -> map back ->
prior cap, with an identity/0.01-width fallback that must never crash the
sampler. The refresh hook on ``ResidualAddOneRemoveOneMove`` feeds those
tables to eigen inner moves on a per-leaf cadence.
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

from eryn.moves import EigenAxisMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.globalfit.moves import eigen_refresh
from lisatools.globalfit.moves.eigen_refresh import (
    eigen_table_from_ll,
    eigen_table_from_waveform,
    prior_box_widths,
)


class PriorBoxWidthsTest(unittest.TestCase):
    def test_reads_the_real_eryn_uniform_bounds(self):
        pri = ProbDistContainer({
            0: uniform_dist(0.1, 30.0),
            1: uniform_dist(3.0, 22.0),
            2: uniform_dist(1e-3, 1.0),
        })
        w = prior_box_widths(pri, 5)
        np.testing.assert_allclose(w[:3], [29.9, 19.0, 0.999], rtol=1e-9)
        # columns without a distribution fall back to unit width
        np.testing.assert_allclose(w[3:], 1.0)

    def test_a_broken_prior_returns_ones_and_does_not_raise(self):
        class Boom:
            @property
            def priors_in(self):
                raise RuntimeError("no")

        w = prior_box_widths(Boom(), 4)
        np.testing.assert_allclose(w, np.ones(4))


def _gaussian_call_ll(cov):
    inv = np.linalg.inv(cov)

    def call_ll(x):
        x = np.atleast_2d(x)
        return -0.5 * np.einsum("ni,ij,nj->n", x, inv, x)

    return call_ll


class EigenTableFromLLTest(unittest.TestCase):
    def test_recovers_analytic_gaussian_curvature(self):
        rng = np.random.default_rng(41)
        ndim = 4
        a = rng.standard_normal((ndim, ndim))
        cov = a @ a.T + 0.5 * np.eye(ndim)
        info_true = np.linalg.inv(cov)
        widths = np.full(ndim, 50.0)  # wide box: the prior cap must not bind

        axes, sigmas = eigen_table_from_ll(
            _gaussian_call_ll(cov), np.zeros(ndim), widths
        )

        self.assertEqual(axes.shape, (ndim, ndim))
        self.assertEqual(sigmas.shape, (ndim,))
        # every axis width is the true curvature width along that axis
        for k in range(ndim):
            ak = axes[:, k]
            np.testing.assert_allclose(np.linalg.norm(ak), 1.0, rtol=1e-10)
            np.testing.assert_allclose(
                sigmas[k], 1.0 / np.sqrt(ak @ info_true @ ak), rtol=1e-4
            )
        # the axes are info-orthogonal (A^T F A diagonal): the defining
        # property of the whitened eigenbasis, invariant under the
        # whiten/unwhiten round trip
        quad = axes.T @ info_true @ axes
        off = quad - np.diag(np.diag(quad))
        self.assertLess(
            np.abs(off).max(), 1e-3 * np.abs(np.diag(quad)).max()
        )

    def test_ill_scaled_problem_stays_finite_and_full_rank(self):
        # column scales spanning 12 decades: without prior-box whitening the
        # relative eigen floor would decide the axes by unit choice
        scales = np.array([1e-6, 1.0, 1e2, 1e-3])
        ndim = scales.size
        rng = np.random.default_rng(43)
        a = rng.standard_normal((ndim, ndim))
        base_cov = a @ a.T + 0.5 * np.eye(ndim)
        cov = base_cov * np.outer(scales, scales)
        widths = 20.0 * scales

        axes, sigmas = eigen_table_from_ll(
            _gaussian_call_ll(cov), np.zeros(ndim), widths
        )
        self.assertTrue(np.all(np.isfinite(axes)))
        self.assertTrue(np.all(np.isfinite(sigmas)))
        self.assertTrue(np.all(sigmas > 0))
        # full rank IN THE WHITENED METRIC (the Euclidean det of a
        # correctly whitened basis is legitimately tiny when scales differ
        # wildly): un-whitening the columns must recover an orthonormal set
        b = axes / widths[:, None]
        b = b / np.linalg.norm(b, axis=0)[None, :]
        self.assertGreater(np.abs(np.linalg.det(b)), 0.9)

    def test_flat_likelihood_is_prior_capped(self):
        ndim = 3
        widths = np.array([2.0, 8.0, 0.5])

        axes, sigmas = eigen_table_from_ll(
            lambda x: np.zeros(np.atleast_2d(x).shape[0]),
            np.zeros(ndim),
            widths,
        )
        self.assertTrue(np.all(np.isfinite(sigmas)))
        # a flat direction can never step past the prior box
        from eryn.moves.eigenaxis import axis_prior_bounds

        bounds = axis_prior_bounds(axes[None], widths)[0]
        self.assertTrue(np.all(sigmas <= bounds * (1 + 1e-12)))

    def test_sigma_max_frac_tightens_the_cap(self):
        ndim = 3
        widths = np.ones(ndim)
        flat = lambda x: np.zeros(np.atleast_2d(x).shape[0])  # noqa: E731
        _, sig_full = eigen_table_from_ll(flat, np.zeros(ndim), widths)
        _, sig_tight = eigen_table_from_ll(
            flat, np.zeros(ndim), widths, sigma_max_frac=0.1
        )
        self.assertTrue(np.all(sig_tight <= 0.1 * sig_full * (1 + 1e-12)))

    def test_nonfinite_likelihood_degrades_to_finite_capped_table(self):
        # information_matrix_from_ll owns this failure mode: non-finite
        # entries are replaced by identity rows (warned on ITS logger), so
        # the pipeline must come back finite and prior-capped, not raise
        ndim = 3
        widths = np.array([1.0, 2.0, 3.0])

        def bad_ll(x):
            return np.full(np.atleast_2d(x).shape[0], np.nan)

        with self.assertLogs("lisatools.info_matrix_ll", level="WARNING"):
            axes, sigmas = eigen_table_from_ll(
                bad_ll, np.zeros(ndim), widths
            )
        self.assertTrue(np.all(np.isfinite(axes)))
        self.assertTrue(np.all(np.isfinite(sigmas)))
        self.assertTrue(np.all(sigmas <= widths + 1e-12))

    def test_fallback_on_raising_likelihood(self):
        ndim = 2
        widths = np.ones(ndim)

        def raising_ll(x):
            raise RuntimeError("engine unavailable")

        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            axes, sigmas = eigen_table_from_ll(
                raising_ll, np.zeros(ndim), widths
            )
        np.testing.assert_array_equal(axes, np.eye(ndim))
        np.testing.assert_allclose(sigmas, 1e-2 * widths)


class EigenTablesBatchTest(unittest.TestCase):
    """eigen_tables_from_ll_batch: one table per point, one batched call."""

    def test_per_point_curvatures_recovered(self):
        from lisatools.globalfit.moves.eigen_refresh import (
            eigen_tables_from_ll_batch,
        )

        rng = np.random.default_rng(59)
        ndim, n = 3, 4
        a = rng.standard_normal((ndim, ndim))
        base_cov = a @ a.T + 0.5 * np.eye(ndim)
        # point k lives on a Gaussian with covariance scaled by (k+1)^2
        scales = (np.arange(n) + 1.0) ** 2
        invs = np.stack([np.linalg.inv(base_cov * s) for s in scales])
        centers = rng.standard_normal((n, ndim))

        def call_ll(x):
            x = np.atleast_2d(x)
            reps = x.shape[0] // n
            assert x.shape[0] == reps * n, "rows must be whole n-blocks"
            point = np.tile(np.arange(n), reps)
            diff = x - centers[point]
            return -0.5 * np.einsum(
                "ni,nij,nj->n", diff, invs[point], diff
            )

        widths = np.full(ndim, 50.0)
        axes, sigmas = eigen_tables_from_ll_batch(
            call_ll, centers, widths
        )
        self.assertEqual(axes.shape, (n, ndim, ndim))
        self.assertEqual(sigmas.shape, (n, ndim))
        for k in range(n):
            for j in range(ndim):
                ak = axes[k, :, j]
                np.testing.assert_allclose(
                    sigmas[k, j],
                    1.0 / np.sqrt(ak @ invs[k] @ ak),
                    rtol=1e-4,
                )

    def test_fallback_on_raising_likelihood_is_batched(self):
        from lisatools.globalfit.moves.eigen_refresh import (
            eigen_tables_from_ll_batch,
        )

        widths = np.array([1.0, 2.0])

        def bad(x):
            raise RuntimeError("no engine")

        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            axes, sigmas = eigen_tables_from_ll_batch(
                bad, np.zeros((5, 2)), widths
            )
        self.assertEqual(axes.shape, (5, 2, 2))
        for k in range(5):
            np.testing.assert_array_equal(axes[k], np.eye(2))
            np.testing.assert_allclose(sigmas[k], 1e-2 * widths)


class EigenTableFromWaveformTest(unittest.TestCase):
    """The waveform route shares the post-processing pipeline; the Gram
    information matrix itself is delegated verbatim to
    ``lisatools.diagnostic.info_matrix`` (tested there)."""

    def test_delegates_and_post_processes_like_the_ll_route(self):
        ndim = 3
        rng = np.random.default_rng(47)
        a = rng.standard_normal((ndim, ndim))
        info_true = a @ a.T + 0.5 * np.eye(ndim)
        widths = np.full(ndim, 50.0)
        seen = {}

        def fake_info_matrix(eps, waveform_model, params, **kwargs):
            seen["eps"] = np.asarray(eps)
            seen["params"] = np.asarray(params)
            seen["kwargs"] = kwargs
            return info_true

        with mock.patch.object(
            eigen_refresh, "info_matrix", side_effect=fake_info_matrix
        ):
            axes, sigmas = eigen_table_from_waveform(
                lambda *args, **kw: None,
                np.arange(ndim, dtype=float),
                widths,
                eps_rel=1e-5,
                parameter_transforms="TRANSFORMS",
                inner_product_kwargs={"psd": None},
                waveform_kwargs={"foo": 1},
            )

        # eps is the per-parameter vector eps_rel * widths
        np.testing.assert_allclose(seen["eps"], 1e-5 * widths)
        self.assertEqual(
            seen["kwargs"]["parameter_transforms"], "TRANSFORMS"
        )
        self.assertEqual(
            seen["kwargs"]["inner_product_kwargs"], {"psd": None}
        )
        self.assertEqual(seen["kwargs"]["waveform_kwargs"], {"foo": 1})

        for k in range(ndim):
            ak = axes[:, k]
            np.testing.assert_allclose(
                sigmas[k], 1.0 / np.sqrt(ak @ info_true @ ak), rtol=1e-8
            )

    def test_fallback_on_raising_info_matrix(self):
        widths = np.ones(2)
        with mock.patch.object(
            eigen_refresh, "info_matrix",
            side_effect=RuntimeError("waveform failed"),
        ):
            with self.assertLogs(eigen_refresh.logger, level="WARNING"):
                axes, sigmas = eigen_table_from_waveform(
                    lambda *a, **k: None, np.zeros(2), widths
                )
        np.testing.assert_array_equal(axes, np.eye(2))
        np.testing.assert_allclose(sigmas, 1e-2 * widths)


def _make_stub_move(inner_moves, ndim=3, refresh=3, scope=None,
                    ntemps=2, nwalkers=4):
    from lisatools.globalfit.moves.addremovemove import (
        ResidualAddOneRemoveOneMove,
    )

    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "sobbh"
    move.ndim = ndim
    move.ntemps = ntemps
    move.nwalkers = nwalkers
    move.moves = inner_moves
    move.priors = {
        "sobbh": ProbDistContainer(
            {i: uniform_dist(-5.0, 5.0) for i in range(ndim)}
        )
    }
    move.eigen_refresh_every = refresh
    move.eigen_eps_rel = None
    move.eigen_table_scope = scope
    move._to_phys = lambda x: x
    cov = np.eye(ndim)
    inv = np.linalg.inv(cov)
    move.compute_like = lambda x, data_index=None: -0.5 * np.einsum(
        "ni,ij,nj->n", np.atleast_2d(x), inv, np.atleast_2d(x)
    )
    return move


class RefreshHookTest(unittest.TestCase):
    """The per-leaf cadence + table delivery on ResidualAddOneRemoveOneMove."""

    def _stub_move(self, inner_moves, ndim=3, refresh=3):
        return _make_stub_move(inner_moves, ndim=ndim, refresh=refresh)

    def _work(self, ndim=3, nl=2):
        return types.SimpleNamespace(
            coords=np.random.default_rng(51).standard_normal((2, 4, nl, ndim))
        )

    def test_cadence_and_table_delivery(self):
        inner = EigenAxisMove()
        move = self._stub_move([inner], refresh=3)
        work = self._work()

        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll",
            wraps=eigen_refresh.eigen_table_from_ll,
        ) as spy:
            for _ in range(5):
                move.refresh_inner_move_tables(0, work)

        # visits 0..4 at cadence 3 -> compute on visit 0 and visit 3
        self.assertEqual(spy.call_count, 2)
        # the table reached the inner move under the branch name
        self.assertIn("sobbh", inner._tables)
        axes, sigmas = inner._tables["sobbh"]
        self.assertEqual(axes.shape, (move.ndim, move.ndim))
        self.assertTrue(np.all(np.isfinite(sigmas)))

    def test_leaves_have_independent_cadence_counters(self):
        inner = EigenAxisMove()
        move = self._stub_move([inner], refresh=10)
        work = self._work()
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll",
            wraps=eigen_refresh.eigen_table_from_ll,
        ) as spy:
            move.refresh_inner_move_tables(0, work)
            move.refresh_inner_move_tables(1, work)
            move.refresh_inner_move_tables(0, work)
        # first visit per leaf computes; the revisit inside the cadence does
        # not
        self.assertEqual(spy.call_count, 2)

    def test_noop_without_eigen_inner_moves(self):
        move = self._stub_move([StretchMove()])
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll"
        ) as spy:
            move.refresh_inner_move_tables(0, self._work())
        spy.assert_not_called()

    def test_env_knobs_resolve_with_branch_prefix(self):
        inner = EigenAxisMove()
        move = self._stub_move([inner], refresh=None)
        with mock.patch.dict(
            os.environ,
            {"SOBBH_EIGEN_REFRESH": "7", "SOBBH_EIGEN_EPS_REL": "1e-3"},
        ):
            self.assertEqual(move._eigen_refresh_cadence(), 7)
            self.assertEqual(move._eigen_eps_rel(), 1e-3)

    def test_table_failure_falls_back_and_does_not_raise(self):
        inner = EigenAxisMove()
        move = self._stub_move([inner])
        move.compute_like = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            move.refresh_inner_move_tables(0, self._work())
        axes, sigmas = inner._tables["sobbh"]
        np.testing.assert_array_equal(axes, np.eye(move.ndim))

    def test_call_ll_routes_through_to_phys(self):
        # per-leaf-fill branches (EMRI) resolve their fills inside
        # _to_phys, keyed by the move's _current_leaf — the table build
        # must score TRANSFORMED rows, never raw sampling rows
        inner = EigenAxisMove()
        move = self._stub_move([inner])
        marker = 1234.5

        def to_phys(x):
            out = np.array(np.atleast_2d(x), copy=True)
            out[:, 0] = marker
            return out

        seen = {"ok": True}
        real_like = move.compute_like

        def checking_like(x, data_index=None):
            if not np.all(np.atleast_2d(x)[:, 0] == marker):
                seen["ok"] = False
            return real_like(x, data_index=data_index)

        move._to_phys = to_phys
        move.compute_like = checking_like
        move.refresh_inner_move_tables(0, self._work())
        self.assertTrue(seen["ok"])

    def test_custom_builder_overrides_the_ll_route(self):
        inner = EigenAxisMove()
        move = self._stub_move([inner])
        axes_sent = np.eye(move.ndim)[:, ::-1].copy()
        sig_sent = np.full(move.ndim, 0.125)
        calls = []

        def builder(mv, leaf, widths):
            calls.append((mv, leaf, widths.copy()))
            return axes_sent, sig_sent

        move.eigen_table_builder = builder
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll"
        ) as ll_spy:
            move.refresh_inner_move_tables(3, self._work())
        ll_spy.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], move)
        self.assertEqual(calls[0][1], 3)
        axes, sigmas = inner._tables["sobbh"]
        np.testing.assert_array_equal(axes, axes_sent)
        np.testing.assert_array_equal(sigmas, sig_sent)


class RefreshScopeTest(unittest.TestCase):
    """Table scope: per_walker (VGB/SOBBH-style, one table per (temp,
    walker)) vs walker_max (MBH/EMRI, one table at the max-lnL cold
    walker)."""

    def _work(self, ndim=3, nl=2):
        return types.SimpleNamespace(
            coords=np.random.default_rng(53).standard_normal((2, 4, nl, ndim))
        )

    def test_scope_env_resolution(self):
        move = _make_stub_move([EigenAxisMove()], scope=None)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOBBH_EIGEN_SCOPE", None)
            self.assertEqual(move._eigen_scope(), "walker_max")
        with mock.patch.dict(
            os.environ, {"SOBBH_EIGEN_SCOPE": "per_walker"}
        ):
            self.assertEqual(move._eigen_scope(), "per_walker")
        with mock.patch.dict(os.environ, {"SOBBH_EIGEN_SCOPE": "junk"}):
            with self.assertLogs(eigen_refresh.logger, level="WARNING"):
                self.assertEqual(move._eigen_scope(), "walker_max")

    def test_walker_max_builds_at_the_max_ll_walker(self):
        inner = EigenAxisMove()
        move = _make_stub_move([inner], scope="walker_max")
        work = self._work()
        best = 1
        seen_idx = []

        real_like = move.compute_like

        def like(x, data_index=None):
            x = np.atleast_2d(x)
            seen_idx.append(np.asarray(data_index))
            if x.shape[0] == move.nwalkers:  # the selection call
                out = np.zeros(move.nwalkers)
                out[best] = 5.0
                return out
            return real_like(x, data_index=data_index)

        move.compute_like = like
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll",
            wraps=eigen_refresh.eigen_table_from_ll,
        ) as spy:
            move.refresh_inner_move_tables(0, work)
        x0_sent = np.asarray(spy.call_args[0][1])
        np.testing.assert_allclose(x0_sent, work.coords[0, best, 0])
        # the corner sweep scores against the chosen walker's data
        build_idx = seen_idx[-1]
        self.assertTrue(np.all(build_idx == best))
        # and the table reached the inner move as a SHARED (ndim, ndim)
        self.assertEqual(
            inner._tables["sobbh"][0].shape, (move.ndim, move.ndim)
        )

    def test_per_walker_stashes_full_tables_no_early_install(self):
        inner = EigenAxisMove()
        move = _make_stub_move([inner], scope="per_walker")
        work = self._work()
        nt, nw, ndim = move.ntemps, move.nwalkers, move.ndim
        n = nt * nw
        axes_fake = np.broadcast_to(np.eye(ndim), (n, ndim, ndim)).copy()
        sig_fake = np.tile(
            (np.arange(n) + 1.0)[:, None], (1, ndim)
        )  # point id + 1 encoded in sigma

        with mock.patch.object(
            eigen_refresh, "eigen_tables_from_ll_batch",
            return_value=(axes_fake, sig_fake),
        ) as spy:
            move.refresh_inner_move_tables(0, work)
        # points sent = the leaf's (temp, walker) grid, t-major
        pts = np.asarray(spy.call_args[0][1])
        np.testing.assert_allclose(
            pts, work.coords[:nt, :, 0].reshape(n, ndim)
        )
        # stashed full-shape, NOT installed yet (the seam slices per split)
        axes, sig = move._eigen_tables[0]
        self.assertEqual(axes.shape, (nt, nw, ndim, ndim))
        self.assertEqual(sig.shape, (nt, nw, ndim))
        self.assertNotIn("sobbh", inner._tables)

    def test_per_walker_build_tiles_data_index_by_point(self):
        inner = EigenAxisMove()
        move = _make_stub_move([inner], scope="per_walker")
        work = self._work()
        nt, nw = move.ntemps, move.nwalkers
        pattern = np.tile(np.arange(nw), nt)
        seen = []

        real_like = move.compute_like

        def like(x, data_index=None):
            seen.append(np.asarray(data_index))
            return real_like(x, data_index=data_index)

        move.compute_like = like
        move.refresh_inner_move_tables(0, work)
        for idx in seen:
            self.assertEqual(idx.size % (nt * nw), 0)
            reps = idx.size // (nt * nw)
            np.testing.assert_array_equal(idx, np.tile(pattern, reps))

    def test_seam_slices_full_tables_by_the_split_mask(self):
        inner = EigenAxisMove()
        move = _make_stub_move([inner], scope="per_walker")
        nt, nw, ndim = move.ntemps, move.nwalkers, move.ndim
        n = nt * nw
        axes_full = np.broadcast_to(
            np.eye(ndim), (nt, nw, ndim, ndim)
        ).copy()
        sig_full = (
            np.arange(n, dtype=float).reshape(nt, nw)[:, :, None]
            + 1.0
        ) * np.ones(ndim)
        move._eigen_tables = {0: (axes_full, sig_full)}

        inds = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        move._install_eigen_split_table(inner, 0, inds == 0)
        ax, sg = inner._tables["sobbh"]
        self.assertEqual(ax.shape, (nt, 2, 1, ndim, ndim))
        self.assertEqual(sg.shape, (nt, 2, 1, ndim))
        # split 0 walkers: t0 -> w0, w2 (points 1, 3); t1 -> w1, w3
        # (points 6, 8) in the id+1 encoding
        np.testing.assert_allclose(
            sg[..., 0].reshape(nt, 2), [[1.0, 3.0], [6.0, 8.0]]
        )

    def test_seam_install_is_noop_for_shared_tables(self):
        inner = EigenAxisMove()
        move = _make_stub_move([inner], scope="walker_max")
        move._eigen_tables = {0: (np.eye(3), np.ones(3))}
        inds = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        move._install_eigen_split_table(inner, 0, inds == 0)
        self.assertNotIn("sobbh", inner._tables)

    def test_seam_install_ignores_non_eigen_moves(self):
        move = _make_stub_move([StretchMove()], scope="per_walker")
        move._eigen_tables = {0: (np.zeros((2, 4, 3, 3)), np.zeros((2, 4, 3)))}
        inds = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        # must simply not raise / not touch the stretch move
        move._install_eigen_split_table(move.moves[0], 0, inds == 0)


if __name__ == "__main__":
    unittest.main()
