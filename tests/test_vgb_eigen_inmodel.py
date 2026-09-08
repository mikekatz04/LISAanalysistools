"""VGB eigen in-model proposal: generic (no-fiber) axes on GB infrastructure.

The VGB move inherits the whole GB info-matrix machinery and historically
ran pure stretch. These tests pin the eigen enablement:

* ``GBSpecialBase._eigen_axes_from_info`` — the axes/width table builder
  factored out of ``_compute_proposal_cholesky``: the GB branch (fiber +
  analytic ridge) is bit-pinned elsewhere; here the GENERIC branch (no
  fiber columns, opt-in via ``eigen_axis_generic_ok``) must produce plain
  whitened-eigen axes with prior-bounded widths, and stay None when not
  opted in.
* ``_vgb_inmodel_defaults`` — the ``VGB_INMODEL_PROPOSAL`` env knob
  (default ``eigen``: info-matrix proposal armed, stretch escape via env).
* The VGB draw branch — one-axis symmetric steps off the (axes * sigma)
  table, graceful fallback to stretch when no table could be built.
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialBase,
    VGBSpecialStretchMove,
    _vgb_inmodel_defaults,
)

NDIM = 5


def _spd(rng, n, ndim):
    a = rng.standard_normal((n, ndim, ndim))
    return a @ np.swapaxes(a, -1, -2) + 0.5 * np.eye(ndim)


def _stub(**over):
    s = types.SimpleNamespace(
        xp=np,
        name="vgb",
        branch_name="vgb",
        use_gpu=False,
        _eigen_axis_min_dim=9,
        _last_axis_sigmas=None,
        _last_axis_pick=None,
        _last_im_kind=None,
        jump_factor=1.0,
        use_info_mat_proposal=True,
        stretch_probability=0.0,
        eigen_axis_generic_ok=True,
        _proposal_param_scales=np.ones(NDIM),
    )
    s._eigen_axis_ready = lambda: False
    widths = np.array([2.0, 1.0, 4.0, 0.5, 3.0])
    s._eigen_axis_widths = lambda ndim: widths[:ndim]
    for k, v in over.items():
        setattr(s, k, v)
    return s


class GenericAxesFromInfoTest(unittest.TestCase):
    def test_generic_branch_builds_prior_bounded_eigen_table(self):
        from eryn.moves.eigenaxis import axis_prior_bounds

        rng = np.random.default_rng(83)
        info_y = _spd(rng, 4, NDIM)
        coords = rng.standard_normal((4, NDIM))
        s = _stub()

        out = GBSpecialBase._eigen_axes_from_info(s, info_y, coords, NDIM)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (4, NDIM, NDIM))

        evals, evecs = np.linalg.eigh(info_y)
        widths = s._eigen_axis_widths(NDIM)
        sig = np.minimum(
            1.0 / np.sqrt(evals), axis_prior_bounds(evecs, widths)
        )
        np.testing.assert_allclose(
            np.abs(out), np.abs(evecs * sig[:, None, :]), atol=1e-12
        )
        np.testing.assert_allclose(s._last_axis_sigmas, sig, atol=1e-12)

    def test_not_opted_in_returns_none(self):
        rng = np.random.default_rng(89)
        s = _stub(eigen_axis_generic_ok=False)
        out = GBSpecialBase._eigen_axes_from_info(
            s, _spd(rng, 2, NDIM), rng.standard_normal((2, NDIM)), NDIM
        )
        self.assertIsNone(out)

    def test_vgb_class_opts_in(self):
        self.assertTrue(VGBSpecialStretchMove.eigen_axis_generic_ok)
        self.assertTrue(VGBSpecialStretchMove.infomat_per_block)
        # plain GB moves keep the historical narrow-basis fallback-to-joint
        self.assertFalse(getattr(GBSpecialBase, "eigen_axis_generic_ok"))


class VGBInmodelDefaultsTest(unittest.TestCase):
    def test_default_is_the_legacy_stretch(self):
        # LIVE-CAMPAIGN GUARD: 3mo_v8 / 1yr_v8 submit scripts predate the
        # knob, so the unset-env default MUST be the pre-eigen pure-stretch
        # config — eigen is an explicit opt-in until its arming gate
        # (VGB sig-het accuracy / chunked cost probe) clears
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VGB_INMODEL_PROPOSAL", None)
            kw = {}
            _vgb_inmodel_defaults(kw)
        self.assertFalse(kw["use_info_mat_proposal"])
        self.assertEqual(kw["stretch_probability"], 1.0)

    def test_env_arms_eigen(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "eigen"}
        ):
            kw = {}
            _vgb_inmodel_defaults(kw)
        self.assertTrue(kw["use_info_mat_proposal"])
        self.assertEqual(kw["stretch_probability"], 0.0)

    def test_explicit_kwargs_win(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "eigen"}
        ):
            kw = {"use_info_mat_proposal": False,
                  "stretch_probability": 0.25}
            _vgb_inmodel_defaults(kw)
        self.assertFalse(kw["use_info_mat_proposal"])
        self.assertEqual(kw["stretch_probability"], 0.25)

    def test_unknown_value_warns_and_stays_stretch(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "banana"}
        ):
            kw = {}
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch",
                level="WARNING",
            ):
                _vgb_inmodel_defaults(kw)
        self.assertFalse(kw["use_info_mat_proposal"])


class VGBEigenDrawTest(unittest.TestCase):
    def _chol(self, n):
        # identity axes with distinct per-column sigmas -> steps are
        # unambiguous single columns
        sig = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return np.broadcast_to(
            np.eye(NDIM) * sig[None, :], (n, NDIM, NDIM)
        ).copy(), sig

    def test_use_eigen_draw_decision(self):
        s = _stub()
        chol = np.zeros((3, NDIM, NDIM))
        self.assertTrue(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s, chol)
        )
        self.assertFalse(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s, None)
        )
        s2 = _stub(use_info_mat_proposal=False)
        self.assertFalse(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s2, chol)
        )
        s3 = _stub(stretch_probability=1.0)
        self.assertFalse(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s3, chol)
        )

    def test_eigen_draw_moves_one_column_and_is_symmetric(self):
        n = 64
        rng = np.random.default_rng(97)
        coords = rng.standard_normal((n, NDIM))
        chol, sig = self._chol(n)
        s = _stub(jump_factor=2.0,
                  _proposal_param_scales=np.full(NDIM, 0.5))

        np.random.seed(11)
        new, factors = VGBSpecialStretchMove._vgb_eigen_axis_draw(
            s, coords, chol
        )
        np.testing.assert_array_equal(factors, 0.0)
        self.assertEqual(s._last_im_kind, "eigen_axis")
        dy = new - coords
        moved = np.abs(dy) > 0
        # exactly one column moves per source (identity axes)
        np.testing.assert_array_equal(moved.sum(axis=-1), 1)
        # magnitude carries jump_factor * sigma_k * param_scale
        for i in range(n):
            k = int(np.argmax(moved[i]))
            self.assertEqual(k, int(s._last_axis_pick[i]))
            self.assertLessEqual(
                np.abs(dy[i, k]), 2.0 * sig[k] * 0.5 * 6.0
            )  # 6-sigma sanity bound on the normal draw


class VGBCholGracefulTest(unittest.TestCase):
    def test_proposal_cholesky_failure_degrades_to_none(self):
        s = _stub()

        def boom(*args, **kwargs):
            raise RuntimeError("engine cannot serve the vgb basis")

        with mock.patch.object(
            GBSpecialBase, "_proposal_cholesky", side_effect=boom
        ):
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch",
                level="WARNING",
            ):
                out = VGBSpecialStretchMove._proposal_cholesky(
                    s, None, None, np.arange(3)
                )
        self.assertIsNone(out)
        # warn ONCE per process-lifetime of the move, not per block
        with mock.patch.object(
            GBSpecialBase, "_proposal_cholesky", side_effect=boom
        ):
            out2 = VGBSpecialStretchMove._proposal_cholesky(
                s, None, None, np.arange(3)
            )
        self.assertIsNone(out2)


class PerBlockTableGateTest(unittest.TestCase):
    def test_instance_flag_skips_the_cold_chain_table(self):
        calls = []
        s = types.SimpleNamespace(
            _prop_timer=None,
            _tables_indexed=False,
            share_proposal_tables=False,
            _build_friend_table=False,
            stretch_probability=0.0,
            use_info_mat_proposal=True,
            infomat_per_block=True,
            name="vgb",
            _infomat_freqs_sorted="stale",
            _infomat_chol_sorted="stale",
        )
        s._refresh_infomat_table = lambda *a: calls.append("refresh")
        bs = types.SimpleNamespace(
            build_infomat_index=lambda *a: calls.append("index"),
            index_friends=lambda *a: calls.append("friends"),
        )
        assert "GB_INFOMAT_PER_BLOCK" not in os.environ
        GBSpecialBase._ensure_proposal_tables(s, None, bs)
        self.assertEqual(calls, [])
        self.assertIsNone(s._infomat_freqs_sorted)
        self.assertIsNone(s._infomat_chol_sorted)


if __name__ == "__main__":
    unittest.main()
