"""Parity between gbspecialstretch's eigen primitives and eryn.moves.eigenaxis.

The generic eigen-axis machinery (prior-box scales, direction projection,
axis prior bounds, the eigen-axis set builder, the one-axis draw) lives in
``eryn.moves.eigenaxis``; ``gbspecialstretch`` re-exports it and keeps only
the GB-specific injections (fiber tangent, analytic shear ridge) plus an
old-signature ``eigen_axis_set`` shim. These tests pin two things:

1. the re-exports really ARE the eryn objects (``is`` identity — a drifted
   copy would silently fork the numerics), and
2. the shim is BIT-IDENTICAL to the pre-extraction implementation, frozen
   here as an inline replica of the old body (same ops, same order).
"""

import unittest

import numpy as np

import eryn.moves.eigenaxis as eeig
from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.gbspecialstretch import (
    axis_prior_bounds,
    draw_axis_step,
    eigen_axis_set,
    gb_fiber_tangent,
    gb_prior_box_scales,
    gb_shear_ridge_axis,
    project_out_direction,
)

DIST, F0, MC, R = 0, 1, 2, 8
NDIM = 9


def _gb_like_coords(rng, n):
    """Rows shaped like the 9-column GB sampling basis, plausible values."""
    coords = rng.standard_normal((n, NDIM)) * 0.1
    coords[:, DIST] = rng.uniform(1.0, 20.0, n)       # kpc
    coords[:, F0] = rng.uniform(3.0, 25.0, n)         # mHz
    coords[:, MC] = rng.uniform(0.2, 1.0, n)          # Msun
    coords[:, R] = rng.uniform(-0.5, 0.5, n)
    return coords


def _spd(rng, n, ndim):
    a = rng.standard_normal((n, ndim, ndim))
    return a @ np.swapaxes(a, -1, -2) + 0.5 * np.eye(ndim)


def _old_eigen_axis_set(info, t_fiber, coords, f0_col, mc_col, r_col,
                        dist_col, tobs, sigma_max=1.0):
    """The pre-extraction body, frozen verbatim (numpy), as the reference."""
    Fp = project_out_direction(info, t_fiber)
    evals, evecs = np.linalg.eigh(Fp)
    ov = np.abs(np.einsum("ni,nij->nj", t_fiber, evecs))
    order = np.argsort(ov, axis=-1)
    axes = np.take_along_axis(evecs, order[:, None, :], axis=-1)
    ridge = gb_shear_ridge_axis(coords, f0_col, mc_col, r_col, dist_col,
                                tobs)
    ridge = ridge - t_fiber * (t_fiber * ridge).sum(axis=-1, keepdims=True)
    rn = np.sqrt((ridge * ridge).sum(axis=-1, keepdims=True))
    ridge = ridge / np.where(rn > 0, rn, np.ones_like(rn))
    axes[:, :, -1] = ridge
    quad = np.einsum("nik,nij,njk->nk", axes, info, axes)
    sigmas = 1.0 / np.sqrt(np.maximum(quad, 1e-300))
    return axes, np.minimum(sigmas, float(sigma_max))


class ReExportIdentityTest(unittest.TestCase):
    """The shared primitives must BE the eryn objects, not copies."""

    def test_primitives_are_eryn_objects(self):
        self.assertIs(project_out_direction, eeig.project_out_direction)
        self.assertIs(axis_prior_bounds, eeig.axis_prior_bounds)
        self.assertIs(draw_axis_step, eeig.draw_axis_step)
        self.assertIs(gb_prior_box_scales, eeig.prior_box_scales)

    def test_gb_specific_injections_stay_in_lat(self):
        for name in ("gb_fiber_tangent", "gb_shear_ridge_axis",
                     "gb_lnfdot_gradient", "gb_ridge_axis"):
            self.assertTrue(hasattr(gbs, name), name)
            self.assertFalse(hasattr(eeig, name),
                             f"{name} must not leak into eryn")


class ShimBitIdentityTest(unittest.TestCase):
    """The old-signature shim reproduces the pre-extraction output EXACTLY."""

    def test_gb_eigen_axis_set_shim_bit_identical(self):
        rng = np.random.default_rng(97)
        n = 16
        tobs = 7864320.0  # ~3 months
        coords = _gb_like_coords(rng, n)
        info = _spd(rng, n, NDIM)
        t_fiber = gb_fiber_tangent(coords, DIST, MC, R)

        for sigma_max in (1.0, np.inf):
            axes_new, sig_new = eigen_axis_set(
                info, t_fiber, coords, F0, MC, R, DIST, tobs,
                sigma_max=sigma_max,
            )
            axes_old, sig_old = _old_eigen_axis_set(
                info, t_fiber, coords, F0, MC, R, DIST, tobs,
                sigma_max=sigma_max,
            )
            self.assertTrue(np.array_equal(axes_new, axes_old),
                            "axes drifted from the pre-extraction output")
            self.assertTrue(np.array_equal(sig_new, sig_old),
                            "sigmas drifted from the pre-extraction output")


if __name__ == "__main__":
    unittest.main()
