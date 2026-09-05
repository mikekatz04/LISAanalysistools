"""Unit tests for the sky-Doppler degeneracy jump (GB in-model mode-hop).

The jump (user design 2026-09-04) shifts the sky and returns the (f0, fdot)
that preserve the *observed*, Doppler-modulated frequency track over the
window -- landing a walker on the sky-Doppler alias (the measured band-1100
second mode). These tests pin the math on synthetic orbits; the physics was
verified separately against the real ESA orbit + the #2 numbers.
"""

import unittest

import numpy as np

from lisatools.sampling.gb_observable_basis import (
    sky_doppler_alias_jump,
    sky_doppler_alias_jump_batch,
)


def _nhat(alpha, sd):
    sd = np.clip(sd, -1, 1)
    cd = np.sqrt(1 - sd * sd)
    return np.array([cd * np.cos(alpha), cd * np.sin(alpha), sd])


def _f_obs(f0, fdot, alpha, sd, v_over_c, t):
    """Observed Doppler-modulated instantaneous frequency over the window."""
    return f0 + fdot * t + f0 * (v_over_c @ _nhat(alpha, sd))


class SkyDopplerJumpTest(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0.0, 7.632e6, 300)
        self.f0 = 19.668e-3
        self.fdot = 1.0e-13
        self.alpha, self.sd = 4.6, -0.55

    def test_zero_sky_step_is_identity(self):
        f0n, fdn, an, sdn = sky_doppler_alias_jump(
            self.f0, self.fdot, self.alpha, self.sd,
            sky_step=(0.0, 0.0),
            v_over_c=np.zeros((self.t.size, 3)), times=self.t)
        self.assertAlmostEqual(f0n, self.f0)
        self.assertAlmostEqual(fdn, self.fdot)
        self.assertAlmostEqual(an, self.alpha)
        self.assertAlmostEqual(sdn, self.sd)

    def test_linear_velocity_preserves_observed_track_exactly(self):
        # v(t) = v0 + v1*t  ->  Doppler is linear in t  ->  the jump's
        # (Δf0, Δfdot) cancel it EXACTLY, so the observed track is unchanged.
        rng = np.random.default_rng(0)
        v0 = rng.normal(size=3) * 1e-4
        v1 = rng.normal(size=3) * 1e-4 / self.t[-1]
        v_over_c = v0[None, :] + v1[None, :] * self.t[:, None]
        step = (0.3, 0.15)
        f0n, fdn, an, sdn = sky_doppler_alias_jump(
            self.f0, self.fdot, self.alpha, self.sd,
            sky_step=step, v_over_c=v_over_c, times=self.t)
        before = _f_obs(self.f0, self.fdot, self.alpha, self.sd, v_over_c, self.t)
        after = _f_obs(f0n, fdn, an, sdn, v_over_c, self.t)
        # residual is only the (here zero) nonlinear part -> ~machine precision
        np.testing.assert_allclose(after, before, atol=1e-6 * self.f0)

    def test_curved_velocity_cancels_offset_and_slope_only(self):
        # v(t) with a quadratic term -> Doppler has curvature the linear jump
        # cannot cancel; the RESIDUAL must have ~zero offset and slope but a
        # nonzero curvature (the imperfect degeneracy = the ~9-nat gap).
        v0 = np.array([1e-4, -2e-4, 0.5e-4])
        v1 = np.array([1e-4, 1e-4, -1e-4]) / self.t[-1]
        v2 = np.array([2e-4, -1e-4, 1e-4]) / self.t[-1] ** 2
        v_over_c = (v0[None] + v1[None] * self.t[:, None]
                    + v2[None] * (self.t ** 2)[:, None])
        f0n, fdn, an, sdn = sky_doppler_alias_jump(
            self.f0, self.fdot, self.alpha, self.sd,
            sky_step=(0.25, 0.2), v_over_c=v_over_c, times=self.t)
        resid = (_f_obs(f0n, fdn, an, sdn, v_over_c, self.t)
                 - _f_obs(self.f0, self.fdot, self.alpha, self.sd, v_over_c, self.t))
        # fit residual to [1, t]: offset + slope must be ~0, curvature not
        A = np.vstack([np.ones_like(self.t), self.t]).T
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        lin = A @ coef
        self.assertLess(np.abs(lin).max(), 1e-6 * self.f0)   # offset+slope killed
        self.assertGreater(np.abs(resid - lin).max(), 1e-12)  # curvature remains

    def test_reversible_to_first_order(self):
        # Jump out then back (negated sky step from the new point). Sky returns
        # EXACTLY (it's just +da then -da); f0/fdot return to FIRST ORDER --
        # n̂(alpha, sin_delta) is nonlinear, so a finite step leaves a
        # second-order round-trip residual (~step^2). That residual is the
        # proposal asymmetry the exact-DB caller's Jacobian absorbs; a search
        # caller ignores it. Here the 0.3-rad step gives ~5e-6 relative.
        v0 = np.array([1e-4, -2e-4, 0.5e-4])
        v1 = np.array([1e-4, 1e-4, -1e-4]) / self.t[-1]
        v_over_c = v0[None] + v1[None] * self.t[:, None]
        step = (0.3, 0.15)
        f0a, fda, aa, sda = sky_doppler_alias_jump(
            self.f0, self.fdot, self.alpha, self.sd,
            sky_step=step, v_over_c=v_over_c, times=self.t)
        f0b, fdb, ab, sdb = sky_doppler_alias_jump(
            f0a, fda, aa, sda,
            sky_step=(-step[0], -step[1]), v_over_c=v_over_c, times=self.t)
        self.assertAlmostEqual(ab, self.alpha, places=10)
        self.assertAlmostEqual(sdb, self.sd, places=10)
        np.testing.assert_allclose(f0b, self.f0, rtol=1e-4, atol=1e-9)
        np.testing.assert_allclose(fdb, self.fdot, rtol=1e-4, atol=1e-20)


class SkyDopplerJumpBatchTest(unittest.TestCase):
    def test_batch_matches_scalar_rowwise(self):
        t = np.linspace(0.0, 7.632e6, 250)
        rng = np.random.default_rng(3)
        v_over_c = (rng.normal(size=3) * 1e-4)[None] + \
            (rng.normal(size=3) * 1e-4 / t[-1])[None] * t[:, None] + \
            (rng.normal(size=3) * 1e-4 / t[-1] ** 2)[None] * (t ** 2)[:, None]
        n = 7
        f0 = 1e-3 * rng.uniform(3, 21, n)
        fd = rng.normal(size=n) * 1e-13
        al = rng.uniform(0, 2 * np.pi, n)
        sd = rng.uniform(-0.9, 0.9, n)
        step = np.stack([rng.normal(size=n) * 0.3,
                         rng.normal(size=n) * 0.2], axis=-1)
        f0b, fdb, ab, sdb = sky_doppler_alias_jump_batch(
            f0, fd, al, sd, sky_step=step, v_over_c=v_over_c, times=t)
        for i in range(n):
            s = sky_doppler_alias_jump(
                f0[i], fd[i], al[i], sd[i],
                sky_step=(step[i, 0], step[i, 1]), v_over_c=v_over_c, times=t)
            self.assertAlmostEqual(f0b[i], s[0], places=12)
            self.assertAlmostEqual(fdb[i], s[1], places=22)
            self.assertAlmostEqual(ab[i], s[2], places=12)
            self.assertAlmostEqual(sdb[i], s[3], places=12)


if __name__ == "__main__":
    unittest.main()
