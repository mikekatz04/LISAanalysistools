"""PE-mode extrinsic draw for the GB RJ F-stat proposals (2026-08-25).

User design ruling: the SEARCH RJ stages keep the historical convention —
(phi0, cos_iota, psi) PINNED at the F-stat maximizers and charged as
uniform constants — bit-identically; the PE stages (rj_fstat_pe /
rj_prior_pe with the F-stat distance-birth path active) DRAW each
extrinsic from a genuine distribution centered on its maximizer and
charge the real forward AND reverse densities in the RJ factors:

* phi0     ~ von Mises (period 2 pi), center phi0_max;
* psi      ~ von Mises on the DOUBLED angle (period pi), center psi_max;
* cos_iota ~ truncated Gaussian on [-1, 1], center cos(iota_max);
* each component eps-floor-mixed with the uniform law on its domain;
* the (phi0 + pi, psi + pi/2) F-stat identity handled by summing the
  (phi0, psi) density over both representatives.

Covered here (light fakes, no built move/ACA/backend):

* joint density normalization (von Mises wrap + psi period-pi Jacobian +
  cos-iota truncation all at once), including a boundary-hugging
  cos-iota center and eps in {0, 0.05, 0.3};
* seam continuity (the circular wraps) and the identity invariance on
  both the evaluated point and the center;
* draw/density consistency through circular and truncated-normal
  moments of large ``pe_extrinsic_rvs`` samples;
* the bounded (~3 log eps) reverse bill for far-off-center rows;
* the move-side gating: ``_pe_or_pin_extrinsics`` pins with correction
  exactly 0.0 when the knob is off (search / knob-off bit-identity),
  and in PE mode the birth correction + the death correction of the
  same row about the same centers cancel exactly (detailed balance);
* knob defaults: constructor default False, ``GBSettings.
  pe_extrinsic_draw`` env default True, eps/geom env defaults.
"""

import dataclasses
import inspect
import os
import unittest

import numpy as np

from lisatools.sampling.fstat_proposal import (
    pe_extrinsic_logpdf,
    pe_extrinsic_rvs,
    pe_extrinsic_sigma,
)
from tests.test_gb_cap_cell_grid import _move


class _EnvPatch:
    """Set/unset env vars for one test, restoring on exit."""

    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _logpdf_pt(phi0, ci, psi, phi0_c, iota_c, psi_c, ln_snr, **kw):
    """Scalar-argument wrapper around the per-row vector density."""
    return float(pe_extrinsic_logpdf(
        np.array([phi0]), np.array([ci]), np.array([psi]),
        np.array([phi0_c]), np.array([iota_c]), np.array([psi_c]),
        np.array([ln_snr]), **kw)[0])


class NormalizationTest(unittest.TestCase):
    """The density integrates to 1 over [0, 2pi) x [-1, 1] x [0, pi)."""

    def _mass(self, phi0_c, iota_c, psi_c, snr, eps,
              n_phi=200, n_ci=120, n_psi=120):
        phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
        ci = -1.0 + (np.arange(n_ci) + 0.5) * (2.0 / n_ci)
        psi = (np.arange(n_psi) + 0.5) * (np.pi / n_psi)
        P, C, S = np.meshgrid(phi, ci, psi, indexing="ij")
        n = P.size
        lp = pe_extrinsic_logpdf(
            P.ravel(), C.ravel(), S.ravel(),
            np.full(n, phi0_c), np.full(n, iota_c), np.full(n, psi_c),
            np.full(n, np.log(snr)), eps=eps)
        vol = (2.0 * np.pi / n_phi) * (2.0 / n_ci) * (np.pi / n_psi)
        return float(np.sum(np.exp(lp)) * vol)

    def test_generic_center(self):
        for eps in (0.0, 0.05, 0.3):
            m = self._mass(1.3, 1.1, 0.7, snr=5.0, eps=eps)
            self.assertAlmostEqual(m, 1.0, delta=5e-3,
                                   msg=f"eps={eps}: mass={m}")

    def test_cos_iota_center_at_the_boundary(self):
        # iota_c near 0 -> cos center ~ +1: the truncation Z = Phi(b) -
        # Phi(a) ~ 1/2 carries half the Gaussian mass; without the Z
        # normalization the integral would come out ~0.5 in the
        # concentrated part.
        m = self._mass(0.4, 0.02, 2.9, snr=5.0, eps=0.05, n_ci=400)
        self.assertAlmostEqual(m, 1.0, delta=5e-3)

    def test_wrap_seams_are_continuous(self):
        # circular wrap: the density is periodic across both seams
        # (phi0: 0 <-> 2 pi; psi: 0 <-> pi through the doubled angle).
        d = 1e-12
        for ctr in ((0.2, 1.0, 0.1), (6.1, 2.0, 3.0)):
            lo = _logpdf_pt(0.0 + d, 0.3, 0.0 + d, *ctr, np.log(20.0))
            hi = _logpdf_pt(2.0 * np.pi - d, 0.3, np.pi - d, *ctr,
                            np.log(20.0))
            self.assertAlmostEqual(lo, hi, places=6)

    def test_weak_f_floor_is_broad_not_degenerate(self):
        # ln_snr = 0 (the max(2F, 1) clip) -> sigma = geom = 2 rad: the
        # proposal must be within a factor ~e of flat across the circle.
        n = 64
        rng = np.random.default_rng(3)
        lp = pe_extrinsic_logpdf(
            rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n),
            rng.uniform(0, np.pi, n),
            np.full(n, 1.0), np.full(n, 1.2), np.full(n, 2.0),
            np.zeros(n), eps=0.05)
        self.assertLess(float(lp.max() - lp.min()), 3.0)
        self.assertTrue(np.all(np.isfinite(lp)))


class IdentityInvarianceTest(unittest.TestCase):
    """(phi0 + pi, psi + pi/2) leaves the density invariant, applied to
    either the evaluated point or the center."""

    def _rand(self, rng, n):
        return (
            rng.uniform(0, 2 * np.pi, n),      # phi0
            rng.uniform(-1, 1, n),             # cos_iota
            rng.uniform(0, np.pi, n),          # psi
            rng.uniform(0, 2 * np.pi, n),      # phi0_c
            rng.uniform(0.05, np.pi - 0.05, n),  # iota_c
            rng.uniform(0, np.pi, n),          # psi_c
            np.log(rng.uniform(1.0, 300.0, n)),  # ln_snr
        )

    def test_point_identity(self):
        rng = np.random.default_rng(11)
        p, c, s, pc, ic, sc, ls = self._rand(rng, 256)
        base = pe_extrinsic_logpdf(p, c, s, pc, ic, sc, ls, eps=0.05)
        shifted = pe_extrinsic_logpdf(
            (p + np.pi) % (2 * np.pi), c, (s + np.pi / 2) % np.pi,
            pc, ic, sc, ls, eps=0.05)
        np.testing.assert_allclose(shifted, base, rtol=0, atol=1e-9)

    def test_center_identity(self):
        rng = np.random.default_rng(13)
        p, c, s, pc, ic, sc, ls = self._rand(rng, 256)
        base = pe_extrinsic_logpdf(p, c, s, pc, ic, sc, ls, eps=0.05)
        shifted = pe_extrinsic_logpdf(
            p, c, s, (pc + np.pi) % (2 * np.pi), ic,
            (sc + np.pi / 2) % np.pi, ls, eps=0.05)
        np.testing.assert_allclose(shifted, base, rtol=0, atol=1e-9)

    def test_partial_shift_is_not_invariant(self):
        # the identity is the JOINT shift; shifting phi0 alone must move
        # the density (guards against an accidentally phi0-flat density).
        # Evaluate ON the concentrated peak — random far points sit on the
        # (deliberately shift-invariant) eps floor and cannot discriminate.
        rng = np.random.default_rng(17)
        n = 32
        _, _, _, pc, ic, sc, _ = self._rand(rng, n)
        ls = np.full(n, np.log(50.0))
        ci_c = np.cos(ic % np.pi)
        base = pe_extrinsic_logpdf(pc, ci_c, sc, pc, ic, sc, ls, eps=0.05)
        shifted = pe_extrinsic_logpdf(
            (pc + np.pi) % (2 * np.pi), ci_c, sc, pc, ic, sc, ls, eps=0.05)
        self.assertTrue(np.all(base - shifted > 1.0))


class EpsFloorTest(unittest.TestCase):
    """The eps floor bounds the worst-case (reverse) log density."""

    def test_bounded_reverse_bill(self):
        # a maximally off-center point at snr 1000 (sigma ~ 2e-3 rad):
        # concentrated parts are ~exp(-kappa) ~ 0, so the density falls
        # onto the floors: >= log(eps^3 / (2 pi * 2 * pi)) exactly-ish.
        eps = 0.05
        # phi0 off by pi/2 breaks BOTH identity branches; psi off by pi/4
        # likewise; cos_iota at the far end.
        lp = _logpdf_pt(np.pi / 2, -0.999, np.pi / 4 + 1.2,
                        0.0, 0.05, 1.2, np.log(1000.0), eps=eps)
        floor = 3.0 * np.log(eps) - np.log(2 * np.pi) - np.log(2.0) - np.log(np.pi)
        self.assertGreaterEqual(lp, floor - 1e-6)
        self.assertLess(lp, floor + 0.2)   # and it IS the floor, not more

    def test_eps_zero_far_point_is_astronomically_disfavored(self):
        lp = _logpdf_pt(np.pi / 2, -0.999, np.pi / 4 + 1.2,
                        0.0, 0.05, 1.2, np.log(1000.0), eps=0.0)
        self.assertLess(lp, -1e4)


class DrawDensityConsistencyTest(unittest.TestCase):
    """``pe_extrinsic_rvs`` samples the density ``pe_extrinsic_logpdf``
    evaluates: circular / truncated-normal moments match analytics."""

    N = 200_000

    def _draws(self, snr, eps, seed=101, geom=2.0):
        rng = np.random.default_rng(seed)
        n = self.N
        ctr = dict(phi0_c=np.full(n, 2.2), iota_c=np.full(n, 0.9),
                   psi_c=np.full(n, 0.6), ln_snr=np.full(n, np.log(snr)))
        p, c, s = pe_extrinsic_rvs(
            ctr["phi0_c"], ctr["iota_c"], ctr["psi_c"], ctr["ln_snr"],
            eps=eps, geom=geom, rand=lambda m: rng.random(m))
        return p, c, s, ctr

    def test_domains(self):
        p, c, s, _ = self._draws(snr=20.0, eps=0.05)
        self.assertTrue(np.all((p >= 0) & (p < 2 * np.pi)))
        self.assertTrue(np.all((s >= 0) & (s < np.pi)))
        self.assertTrue(np.all((c >= -1) & (c <= 1)))

    def test_circular_moments(self):
        from scipy.special import iv

        snr, eps, geom = 20.0, 0.05, 2.0
        sigma = float(pe_extrinsic_sigma(np.array([np.log(snr)]), geom)[0])
        kappa = 1.0 / sigma**2
        p, c, s, ctr = self._draws(snr=snr, eps=eps)
        tol = 5.0 / np.sqrt(self.N)
        # E[cos(2 (phi0 - c))]: invariant under the +pi identity branch
        # and zero-mean under the uniform floor.
        want = (1 - eps) * iv(2, kappa) / iv(0, kappa)
        got = float(np.mean(np.cos(2 * (p - ctr["phi0_c"]))))
        self.assertAlmostEqual(got, want, delta=tol)
        # E[cos(1 * (phi0 - c))]: the identity branch KILLS the first
        # harmonic (the two branches contribute +/-I1/I0) -> 0.
        got1 = float(np.mean(np.cos(p - ctr["phi0_c"])))
        self.assertAlmostEqual(got1, 0.0, delta=tol)
        # E[cos(4 (psi - c))] through the doubled angle: cos(2 x) moment
        # of the vM on x = 2 psi, branch-invariant.
        want4 = (1 - eps) * iv(2, kappa) / iv(0, kappa)
        got4 = float(np.mean(np.cos(4 * (s - ctr["psi_c"]))))
        self.assertAlmostEqual(got4, want4, delta=tol)
        # E[cos(2 (psi - c))]: killed by the pi/2 psi branch shift -> 0.
        got2 = float(np.mean(np.cos(2 * (s - ctr["psi_c"]))))
        self.assertAlmostEqual(got2, 0.0, delta=tol)

    def test_cos_iota_moments(self):
        from scipy.stats import truncnorm

        snr, eps, geom = 20.0, 0.05, 2.0
        sigma = float(pe_extrinsic_sigma(np.array([np.log(snr)]), geom)[0])
        p, c, s, ctr = self._draws(snr=snr, eps=eps, seed=202)
        ci_c = float(np.cos(ctr["iota_c"][0]))
        a, b = (-1 - ci_c) / sigma, (1 - ci_c) / sigma
        tn_mean = float(truncnorm.mean(a, b, loc=ci_c, scale=sigma))
        want = (1 - eps) * tn_mean + eps * 0.0
        self.assertAlmostEqual(float(np.mean(c)), want,
                               delta=5.0 / np.sqrt(self.N))
        # the eps fraction actually lands far out: P(|c - ci_c| > 5 sigma)
        far = float(np.mean(np.abs(c - ci_c) > 5 * sigma))
        lo = max(-1.0, ci_c - 5 * sigma)
        hi = min(1.0, ci_c + 5 * sigma)
        p_far_uniform = 1.0 - (hi - lo) / 2.0
        self.assertAlmostEqual(far, eps * p_far_uniform, delta=6e-3)

    def test_weak_f_row_draws_are_broad(self):
        # ln_snr = 0 (weak-F clip): sigma = 2 rad -> near-uniform draws,
        # every domain corner reachable.
        p, c, s, _ = self._draws(snr=1.0, eps=0.05, seed=303)
        self.assertGreater(float(np.std(p)), 1.2)
        self.assertGreater(float(np.std(c)), 0.4)


class MoveHelperGatingTest(unittest.TestCase):
    """The move-side wiring: pin + 0.0 correction when the knob is off
    (search / knob-off bit-identity), draw + exact birth/death factor
    cancellation when on."""

    def _shim(self, active):
        m = _move(4)
        if active is not None:
            m.pe_extrinsic_draw = active
        return m

    @staticmethod
    def _centers(rng, n):
        return (rng.uniform(0, 2 * np.pi, n),
                rng.uniform(0.05, np.pi - 0.05, n),
                rng.uniform(0, np.pi, n),
                np.log(rng.uniform(5.0, 200.0, n)))

    def test_active_gate(self):
        self.assertFalse(self._shim(None)._pe_extr_active())   # attr absent
        self.assertFalse(self._shim(False)._pe_extr_active())
        self.assertTrue(self._shim(True)._pe_extr_active())

    def test_constructor_default_is_off(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        sig = inspect.signature(GBSpecialBase.__init__)
        self.assertIs(sig.parameters["pe_extrinsic_draw"].default, False)

    def test_knob_off_pins_with_zero_correction(self):
        m = self._shim(False)
        rng = np.random.default_rng(7)
        n = 32
        p0, io, ps, ls = self._centers(rng, n)
        params = rng.random((n, 8))
        rows = np.arange(n)
        corr = m._pe_or_pin_extrinsics(params, rows, p0, io, ps, ls)
        self.assertEqual(corr, 0.0)   # EXACTLY the pre-flag factor term
        np.testing.assert_array_equal(params[:, 3], p0 % (2 * np.pi))
        np.testing.assert_array_equal(params[:, 4], np.cos(io % np.pi))
        np.testing.assert_array_equal(params[:, 5], ps % np.pi)
        # death side likewise contributes exactly 0.0
        self.assertEqual(
            m._pe_death_extr_corr(params, rows, p0, io, ps, ls), 0.0)

    def test_pe_mode_birth_death_cancellation(self):
        # birth correction -(log g + logV) followed by the death
        # correction +(log g + logV) of the SAME row about the SAME
        # centers must cancel exactly: the detailed-balance identity the
        # factor accounting relies on.
        m = self._shim(True)
        rng = np.random.default_rng(19)
        n = 64
        p0, io, ps, ls = self._centers(rng, n)
        params = np.zeros((n, 8))
        rows = np.arange(n)
        with _EnvPatch(GB_PE_EXTRINSIC_FLOOR_EPS=None,
                       GB_PE_EXTRINSIC_SIGMA_GEOM=None):
            corr_b = m._pe_or_pin_extrinsics(params, rows, p0, io, ps, ls)
            corr_d = m._pe_death_extr_corr(params, rows, p0, io, ps, ls)
        corr_b = np.asarray(corr_b)
        self.assertTrue(np.all(np.isfinite(corr_b)))
        np.testing.assert_allclose(corr_b + np.asarray(corr_d), 0.0,
                                   atol=1e-12)
        # and the drawn extrinsics landed in the sampling domains
        self.assertTrue(np.all((params[:, 3] >= 0)
                               & (params[:, 3] < 2 * np.pi)))
        self.assertTrue(np.all((params[:, 4] >= -1) & (params[:, 4] <= 1)))
        self.assertTrue(np.all((params[:, 5] >= 0) & (params[:, 5] < np.pi)))

    def test_pe_mode_draws_concentrate_on_the_centers(self):
        m = self._shim(True)
        n = 512
        rng = np.random.default_rng(23)
        p0 = np.full(n, 1.0)
        io = np.full(n, 1.1)
        ps = np.full(n, 2.0)
        ls = np.full(n, np.log(100.0))
        params = np.zeros((n, 8))
        _ = m._pe_or_pin_extrinsics(params, np.arange(n), p0, io, ps, ls)
        # identity-folded distances: most draws within a few sigma
        dphi = np.abs((params[:, 3] - 1.0 + np.pi) % np.pi - np.pi / 2)
        dphi = np.pi / 2 - dphi   # distance to {c, c+pi} on the circle
        frac_close = float(np.mean(dphi < 5 * (2.0 / 100.0)))
        self.assertGreater(frac_close, 0.85)   # ~1 - eps

    def test_env_knob_defaults(self):
        m = self._shim(True)
        with _EnvPatch(GB_PE_EXTRINSIC_FLOOR_EPS=None,
                       GB_PE_EXTRINSIC_SIGMA_GEOM=None):
            self.assertAlmostEqual(m._pe_extr_floor_eps(), 0.05)
            self.assertAlmostEqual(m._pe_extr_sigma_geom(), 2.0)
        with _EnvPatch(GB_PE_EXTRINSIC_FLOOR_EPS="0.1",
                       GB_PE_EXTRINSIC_SIGMA_GEOM="1.0"):
            self.assertAlmostEqual(m._pe_extr_floor_eps(), 0.1)
            self.assertAlmostEqual(m._pe_extr_sigma_geom(), 1.0)

    def test_log_extr_uniform_vol_matches_the_container(self):
        # phi0 U[0, 2 pi) x cos_iota U[-1, 1] x psi U[0, pi) — the
        # make_gb_rj_birth_container / global-prior extrinsic block whose
        # constant the PE factors replace.
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        self.assertAlmostEqual(
            GBSpecialBase._LOG_EXTR_UNIFORM_VOL,
            np.log(2 * np.pi) + np.log(2.0) + np.log(np.pi), places=14)


class SettingsKnobTest(unittest.TestCase):
    """GBSettings.pe_extrinsic_draw: env-seeded, default ON."""

    def _field(self):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        for f in dataclasses.fields(GBSettings):
            if f.name == "pe_extrinsic_draw":
                return f
        raise AssertionError("pe_extrinsic_draw field missing on GBSettings")

    def test_default_on(self):
        with _EnvPatch(GB_PE_EXTRINSIC_DRAW=None):
            self.assertIs(self._field().default_factory(), True)

    def test_env_off(self):
        with _EnvPatch(GB_PE_EXTRINSIC_DRAW="0"):
            self.assertIs(self._field().default_factory(), False)
        with _EnvPatch(GB_PE_EXTRINSIC_DRAW="1"):
            self.assertIs(self._field().default_factory(), True)


if __name__ == "__main__":
    unittest.main()
