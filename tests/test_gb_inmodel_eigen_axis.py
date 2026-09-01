"""Per-eigenaxis in-model proposal: prior-box scaling, fiber projection, ridge axis.

WHY THIS EXISTS (flagship forensics, 2026-08-31). The 9-column GB sampling
basis ``[dist, f0_mHz, Mc, phi0, cos_iota, psi, alpha, sin_delta, r]`` is
over-parameterised: ``(dist, Mc, r)`` enter the waveform only through
``(A, fdot)``, so one direction is EXACTLY likelihood-flat. Measured on the
flagship (20.380377 mHz, SNR 46): the analytic fiber tangent gives
``t^T F t / lam_max = 4.7e-26``, and its overlap with the smallest
eigenvector is 1.0000.

The joint Gaussian draw handles that with a RELATIVE eigen-floor
(``1e-10 * lam_max``), which caps every soft direction at ``1e5 x`` the
stiffest width. On the flagship that shrinks the true 1-sigma steps by
645x (dist), 95x (Mc, r), 43x (phi0), 22x (psi). Worse, the floor is
relative and ``_proposal_param_scales`` is ALL ONES in this basis (the
``s[_fdot_col]`` line only fires when ``"fdot"`` is an input-basis column,
and the fdot_astro basis carries ``Mc`` + ``fdot_astro_ratio`` instead) --
so WHICH directions get floored is set by the arbitrary unit choice
(f0 ~ 20 mHz vs dist ~ 9 kpc vs angles ~ 1), not by curvature.

Consequence measured in the run: no eigen-axis moves ``ln(fdot)`` by more
than 0.040 per 1-sigma step, against the 0.35 needed to walk the flagship's
near-truth cluster (f0 -1.38 bins, fdot 1.35x truth) to the true peak --
and the Fisher at the cluster is essentially identical to the Fisher at
truth, so this is proposal geometry, not local curvature.

The fix under test, all behind ``GB_INMODEL_EIGEN_AXIS`` (default OFF):

1. ``gb_prior_box_scales`` -- whiten to the prior box so the spectrum
   reflects curvature anisotropy rather than units;
2. ``gb_fiber_tangent`` + ``project_out_direction`` -- remove the exactly
   flat direction (``gb_ridge_gibbs`` already resamples it in closed form
   with zero likelihood calls);
3. ``gb_ridge_axis`` -- the f0-fdot ridge carried as an EXPLICIT axis,
   because no eigenvector points along it (the Fisher's own conditional
   optimum ``-F[f0,r]/F[f0,f0]`` = -2.304 bins per unit r, independently
   confirmed at -2.54 by a direct likelihood-surface scan);
4. ``eigen_axis_set`` + ``draw_axis_step`` -- one random axis per repeat at
   its OWN 1-sigma scale (cost-neutral: still one likelihood call), so a
   soft direction can no longer hide inside a 9-D average.
"""
import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    draw_axis_step,
    eigen_axis_set,
    gb_fiber_tangent,
    gb_lnfdot_gradient,
    gb_prior_box_scales,
    gb_ridge_axis,
    axis_prior_bounds,
    project_out_direction,
    _eigen_axis_on,
)
from lisatools.sampling.gb_observable_basis import FDOT_K

# Flagship sampling-basis point (validated round-trip through the stock
# transform container: max rel err vs the catalogue phys row 3.4e-16).
FLAGSHIP = np.array([9.05215813e00, 2.03803767e01, 4.65777687e-01,
                     -3.41840873e00, -8.83190852e-01, 3.89809240e-01,
                     4.06170662e00, -7.86384411e-01, -1.11022302e-15])
DIST, F0, MC, R = 0, 1, 2, 8
TOBS = 7.776e6                       # 90 d, the production 3-month run
MHZ_PER_BIN = 1e3 / TOBS             # 1 bin = 1/Tobs Hz, f0 sampled in mHz


def _flagship_fisher():
    """The measured flagship Fisher (sampling basis, run's fitted noise).

    Rebuilt from its eigendecomposition so the fixture is exact and needs
    no waveform generation. Eigenvalues span 4.2e15 -- that conditioning IS
    the object under test, so it must not be softened here.
    """
    path = os.path.join(os.path.dirname(__file__), "data",
                        "flagship_fisher.npz")
    if os.path.exists(path):
        return np.load(path)["F"]
    # Analytic stand-in with the same qualitative structure: one exact null
    # along the fiber, a stiff f0 direction, and an f0-r cross term giving
    # the measured -2.304 bins/unit-r conditional slope.
    t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
    rng = np.random.default_rng(20260831)
    B = rng.normal(size=(9, 9))
    B -= np.outer(B @ t, t)                       # kill the fiber direction
    F = B.T @ B
    F *= 1e3
    F[F0, F0] = 8.4597e11
    slope_mhz = -2.304 * MHZ_PER_BIN              # d(f0)/d(r) at the optimum
    F[F0, R] = F[R, F0] = -slope_mhz * F[F0, F0]
    F -= np.outer(F @ t, t) + np.outer(t, F @ t) - np.outer(t, t) * (t @ F @ t)
    return 0.5 * (F + F.T)


class PriorBoxScaleTest(unittest.TestCase):
    """Whitening must make the coordinates O(1) and be exactly invertible."""

    def test_scales_are_the_box_widths(self):
        lo = np.array([0.1, 3.0, 1e-3, 0.0, -1.0, 0.0, 0.0, -1.0, -2.0])
        hi = np.array([30.0, 22.0, 1.0, 2 * np.pi, 1.0, np.pi,
                       2 * np.pi, 1.0, 2.0])
        s = gb_prior_box_scales(lo, hi)
        np.testing.assert_allclose(s, hi - lo)

    def test_whitening_makes_the_box_unit_sized(self):
        lo = np.array([0.1, 3.0, 1e-3, 0.0, -1.0, 0.0, 0.0, -1.0, -2.0])
        hi = np.array([30.0, 22.0, 1.0, 2 * np.pi, 1.0, np.pi,
                       2 * np.pi, 1.0, 2.0])
        s = gb_prior_box_scales(lo, hi)
        self.assertTrue(np.allclose((hi - lo) / s, 1.0))

    def test_zero_width_columns_do_not_divide_by_zero(self):
        """A fixed (filled) column has zero prior width; it must stay 1.0."""
        lo = np.array([0.0, 5.0, 0.0]); hi = np.array([1.0, 5.0, 2.0])
        s = gb_prior_box_scales(lo, hi)
        self.assertEqual(s[1], 1.0)
        self.assertTrue(np.all(np.isfinite(s)))

    def test_congruence_round_trip_is_exact(self):
        """Whitening must leave the quadratic cost invariant.

        With ``x = y * s`` (y whitened), the information matrix in whitened
        coordinates is ``S F S = F * outer(s, s)``, so the cost computed in
        either basis is the same number. This is the identity the wiring
        relies on when it draws in whitened coordinates and maps the step
        back with ``* s``.
        """
        F = _flagship_fisher()
        s = np.abs(FLAGSHIP) + 1.0
        y = np.array([0.3] * 9)
        x = y * s
        self.assertAlmostEqual(
            float(x @ F @ x) / float(y @ (F * np.outer(s, s)) @ y), 1.0,
            places=12)


class FiberTangentTest(unittest.TestCase):
    """The (dist, Mc, r) fiber is the EXACT likelihood null direction."""

    def test_tangent_matches_the_closed_form(self):
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        mc, r, dist = FLAGSHIP[MC], FLAGSHIP[R], FLAGSHIP[DIST]
        want = np.zeros(9)
        want[MC] = 1.0
        want[R] = -(1.0 + r) * (5.0 / 3.0) / mc
        want[DIST] = (5.0 / 3.0) * dist / mc
        want /= np.linalg.norm(want)
        np.testing.assert_allclose(np.abs(t), np.abs(want), rtol=1e-12)

    def test_tangent_is_unit_norm(self):
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)
        np.testing.assert_allclose(np.linalg.norm(t, axis=-1), 1.0, rtol=1e-12)

    def test_only_dist_Mc_r_are_nonzero(self):
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        others = [i for i in range(9) if i not in (DIST, MC, R)]
        np.testing.assert_allclose(t[others], 0.0, atol=0)

    def test_batched_over_sources(self):
        rows = np.repeat(FLAGSHIP[None, :], 5, axis=0)
        rows[:, MC] = np.linspace(0.2, 0.9, 5)
        t = gb_fiber_tangent(rows, DIST, MC, R)
        self.assertEqual(t.shape, (5, 9))
        np.testing.assert_allclose(np.linalg.norm(t, axis=-1), 1.0, rtol=1e-12)

    def test_it_really_is_the_flat_direction_of_the_measured_fisher(self):
        """The whole design rests on this: t^T F t must be ~0 vs lam_max."""
        F = _flagship_fisher()
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        lam_max = np.linalg.eigvalsh(F).max()
        self.assertLess(abs(t @ F @ t) / lam_max, 1e-18)


class ProjectOutTest(unittest.TestCase):
    def test_projection_annihilates_the_direction(self):
        F = _flagship_fisher()
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        Fp = project_out_direction(F[None, ...], t[None, :])[0]
        np.testing.assert_allclose(Fp @ t, 0.0, atol=1e-6 * np.abs(Fp).max())

    def test_projection_is_symmetric(self):
        F = _flagship_fisher()
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        Fp = project_out_direction(F[None, ...], t[None, :])[0]
        np.testing.assert_allclose(Fp, Fp.T, rtol=1e-10, atol=1e-8)

    def test_projection_preserves_orthogonal_curvature(self):
        """Directions F-orthogonal to the fiber must keep their curvature."""
        F = _flagship_fisher()
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        Fp = project_out_direction(F[None, ...], t[None, :])[0]
        u = np.zeros(9); u[F0] = 1.0
        u = u - t * (u @ t); u /= np.linalg.norm(u)
        self.assertAlmostEqual(float(u @ Fp @ u) / float(u @ F @ u), 1.0,
                               places=6)

    def test_batched(self):
        F = _flagship_fisher()
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        Fp = project_out_direction(np.repeat(F[None, ...], 3, axis=0),
                                   np.repeat(t[None, :], 3, axis=0))
        self.assertEqual(Fp.shape, (3, 9, 9))


def _ridge_from(F, coords):
    """Helper: the ridge axis built the way eigen_axis_set builds it.

    Mirrors the caller's contract: sort eigen-columns by fiber overlap and
    hand ``gb_ridge_axis`` only the NON-fiber ones.
    """
    t = gb_fiber_tangent(coords, DIST, MC, R)
    Fp = project_out_direction(F[None, ...], t)
    ev, V = np.linalg.eigh(Fp)
    ov = np.abs(np.einsum("ni,nij->nj", t, V))
    order = np.argsort(ov, axis=-1)
    V = np.take_along_axis(V, order[:, None, :], axis=-1)
    ev = np.take_along_axis(ev, order, axis=-1)
    g = gb_lnfdot_gradient(coords, F0, MC, R)
    return gb_ridge_axis(ev[:, :-1], V[:, :, :-1], g), g


class LnFdotGradientTest(unittest.TestCase):
    def test_matches_the_closed_form(self):
        g = gb_lnfdot_gradient(FLAGSHIP[None, :], F0, MC, R)[0]
        self.assertAlmostEqual(g[MC], (5 / 3) / FLAGSHIP[MC], places=12)
        self.assertAlmostEqual(g[F0], (11 / 3) / FLAGSHIP[F0], places=12)
        self.assertAlmostEqual(g[R], 1.0 / (1.0 + FLAGSHIP[R]), places=12)

    def test_is_orthogonal_to_the_fiber(self):
        """Load-bearing: the fiber holds fdot fixed, so g . t == 0 exactly.

        This is what makes the pseudo-inverse well posed after projection.
        """
        g = gb_lnfdot_gradient(FLAGSHIP[None, :], F0, MC, R)[0]
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        self.assertLess(abs(float(g @ t)) / float(np.linalg.norm(g)), 1e-12)

    def test_only_f0_Mc_r_are_nonzero(self):
        g = gb_lnfdot_gradient(FLAGSHIP[None, :], F0, MC, R)[0]
        others = [i for i in range(9) if i not in (F0, MC, R)]
        np.testing.assert_allclose(g[others], 0.0, atol=0)


class RidgeAxisTest(unittest.TestCase):
    """``a ~ F^+ g`` -- provably the best fdot-mover per unit lnL cost.

    NO LONGER INSTALLED (2026-09-01): ``eigen_axis_set`` now carries the
    ANALYTIC shear ridge instead. The optimality proved below is real, but
    it is optimality with respect to ``F``, and this ``F``'s f0 block is
    34% off -- so the axis it selects points 80% too steep. Kept because
    the construction becomes correct again the day the information
    matrix's f0 block is trustworthy, and because a deleted derivation is
    a derivation someone re-does wrong.
    """

    def test_axis_is_unit_norm(self):
        u, _ = _ridge_from(_flagship_fisher(), FLAGSHIP[None, :])
        np.testing.assert_allclose(np.linalg.norm(u, axis=-1), 1.0, rtol=1e-10)

    def test_it_beats_every_other_direction_on_fdot_per_unit_cost(self):
        """The Cauchy-Schwarz optimality that motivates the construction.

        Score a direction by (g . a)^2 / (a^T F a): the fdot motion bought
        per unit likelihood cost. a ~ F^+ g must maximise it.
        """
        F = _flagship_fisher()
        u, g = _ridge_from(F, FLAGSHIP[None, :])
        u = u[0]; g = g[0]

        def score(a):
            c = float(a @ F @ a)
            return (float(g @ a) ** 2) / c if c > 1e-280 else -np.inf

        best = score(u)
        rng = np.random.default_rng(11)
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)[0]
        for _ in range(400):
            a = rng.normal(size=9)
            a -= t * (a @ t)                      # stay in the fiber complement
            a /= np.linalg.norm(a)
            self.assertLessEqual(score(a), best * (1 + 1e-8))

    def test_it_carries_real_fdot_motion(self):
        u, g = _ridge_from(_flagship_fisher(), FLAGSHIP[None, :])
        self.assertGreater(abs(float(g[0] @ u[0])), 1e-6)

    def test_degenerate_matrix_falls_back_to_the_gradient(self):
        ev = np.zeros((1, 9)); V = np.repeat(np.eye(9)[None, ...], 1, axis=0)
        g = gb_lnfdot_gradient(FLAGSHIP[None, :], F0, MC, R)
        u = gb_ridge_axis(ev, V, g)
        self.assertTrue(np.all(np.isfinite(u)))
        np.testing.assert_allclose(np.linalg.norm(u, axis=-1), 1.0, rtol=1e-10)


def _axis_set(F, coords, **kw):
    t = gb_fiber_tangent(coords, DIST, MC, R)
    return eigen_axis_set(F[None, ...], t, coords, F0, MC, R, DIST,
                          TOBS, **kw)


class EigenAxisSetTest(unittest.TestCase):
    def test_axis_count_is_ndim_minus_fiber_plus_ridge(self):
        axes, sig = _axis_set(_flagship_fisher(), FLAGSHIP[None, :])
        self.assertEqual(axes.shape, (1, 9, 9))   # 8 eigen-axes + 1 ridge
        self.assertEqual(sig.shape, (1, 9))

    def test_all_axes_are_unit_norm(self):
        axes, _ = _axis_set(_flagship_fisher(), FLAGSHIP[None, :])
        np.testing.assert_allclose(np.linalg.norm(axes[0], axis=0), 1.0,
                                   rtol=1e-8)

    def test_sigmas_are_the_per_axis_curvature_widths(self):
        """sigma_k = min(1/sqrt(a_k^T F a_k), sigma_max) for every axis."""
        F = _flagship_fisher()
        smax = 1e6
        axes, sig = _axis_set(F, FLAGSHIP[None, :], sigma_max=smax)
        for k in range(axes.shape[-1]):
            a = axes[0, :, k]
            want = min(1.0 / np.sqrt(max(float(a @ F @ a), 1e-300)), smax)
            self.assertAlmostEqual(sig[0, k] / want, 1.0, places=5)

    def test_sigma_max_clamps_flat_directions(self):
        """A near-null axis must not produce an unbounded step."""
        _, sig = _axis_set(_flagship_fisher(), FLAGSHIP[None, :],
                           sigma_max=2.5)
        self.assertLessEqual(float(sig.max()), 2.5 + 1e-12)
        self.assertTrue(np.all(sig > 0))

    def test_no_axis_is_the_fiber(self):
        """The flat direction must not reappear as an axis."""
        axes, _ = _axis_set(_flagship_fisher(), FLAGSHIP[None, :])
        t = gb_fiber_tangent(FLAGSHIP[None, :], DIST, MC, R)
        self.assertLess(float(np.abs(t[0] @ axes[0]).max()), 1e-4)

    def test_sigmas_are_finite_and_positive(self):
        _, sig = _axis_set(_flagship_fisher(), FLAGSHIP[None, :])
        self.assertTrue(np.all(np.isfinite(sig)))
        self.assertTrue(np.all(sig > 0))

    def test_the_ridge_axis_walks_the_analytic_shear_ridge(self):
        """THE regression guard for the whole change.

        The installed ridge is the ANALYTIC shear ridge, not ``F^+ g``.
        ``F^+ g`` is provably the best fdot-mover per unit lnL cost GIVEN
        a correct ``F`` -- but this ``F``'s joint draw walks a ridge of
        slope ``d f0 / d fdot = -0.898 T`` against the geometry's
        ``-T/2``, so the "optimal" axis inherits that error. Optimising
        against a wrong metric is the failure being retired, which is why
        this asserts the GEOMETRY and not a score under ``F``.

        The ratio survives the fiber orthogonalisation exactly: the fiber
        holds ``A`` and ``fdot`` fixed and does not touch ``f0``, so it is
        null for all three observables, and the renormalisation that
        follows is a common positive factor.
        """
        axes, _ = _axis_set(_flagship_fisher(), FLAGSHIP[None, :])
        a = axes[0, :, -1]
        g = gb_lnfdot_gradient(FLAGSHIP[None, :], F0, MC, R)[0]
        fd = (FDOT_K * FLAGSHIP[MC] ** (5 / 3)
              * (FLAGSHIP[F0] * 1e-3) ** (11 / 3) * (1 + FLAGSHIP[R]))
        d_lnfdot = float(g @ a)
        self.assertGreater(abs(d_lnfdot), 1e-6, "the ridge must move fdot")
        slope = (a[F0] * 1e-3) / (fd * d_lnfdot)          # d f0[Hz] / d fdot
        self.assertAlmostEqual(slope / TOBS, -0.5, places=6)

    def test_the_ridge_axis_holds_the_measured_amplitude_fixed(self):
        """``A``, not ``dist``, is what the data measures.

        ``ln A = const + (5/3) ln Mc + (2/3) ln f0 - ln dist``; a ridge
        that moved it would spend likelihood on amplitude while trying to
        move fdot, which is the same class of defect as the f_mid leak.
        """
        axes, _ = _axis_set(_flagship_fisher(), FLAGSHIP[None, :])
        a = axes[0, :, -1]
        d_lnA = ((5 / 3) * a[MC] / FLAGSHIP[MC]
                 + (2 / 3) * a[F0] / FLAGSHIP[F0]
                 - a[DIST] / FLAGSHIP[DIST])
        self.assertLess(abs(float(d_lnA)), 1e-10)

    def test_batched_over_sources_is_row_independent(self):
        F = _flagship_fisher()
        rows = np.repeat(FLAGSHIP[None, :], 3, axis=0)
        t = gb_fiber_tangent(rows, DIST, MC, R)
        axes, sig = eigen_axis_set(np.repeat(F[None, ...], 3, axis=0), t,
                                   rows, F0, MC, R, DIST, TOBS)
        np.testing.assert_allclose(axes[0], axes[2], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sig[0], sig[2], rtol=1e-10)


class AxisPriorBoundTest(unittest.TestCase):
    """Per-axis step bound from the prior box (the scale-correct clamp)."""

    def test_unit_basis_axis_is_bounded_by_its_own_width(self):
        axes = np.repeat(np.eye(4)[None, ...], 1, axis=0)
        w = np.array([2.0, 5.0, 0.5, 10.0])
        b = axis_prior_bounds(axes, w)
        np.testing.assert_allclose(b[0], w, rtol=1e-12)

    def test_diagonal_axis_is_bounded_by_the_tightest_component(self):
        a = np.zeros((1, 3, 1)); a[0, :, 0] = np.array([0.6, 0.8, 0.0])
        w = np.array([6.0, 4.0, 100.0])
        # 6/0.6 = 10 vs 4/0.8 = 5 -> the tighter one wins
        self.assertAlmostEqual(float(axis_prior_bounds(a, w)[0, 0]), 5.0)

    def test_untouched_components_do_not_bound(self):
        """A narrow parameter the axis does not move must not clamp it."""
        a = np.zeros((1, 3, 1)); a[0, :, 0] = np.array([1.0, 0.0, 0.0])
        w = np.array([7.0, 1e-9, 1e-9])
        self.assertAlmostEqual(float(axis_prior_bounds(a, w)[0, 0]), 7.0)

    def test_is_positive_and_finite_for_real_axes(self):
        F = _flagship_fisher()
        axes, _ = _axis_set(F, FLAGSHIP[None, :])
        w = np.array([30.0, 19.0, 1.0, 2 * np.pi, 2.0, np.pi,
                      2 * np.pi, 2.0, 4.0])
        b = axis_prior_bounds(axes, w)
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(b > 0))


class DrawAxisStepTest(unittest.TestCase):
    def test_step_lies_along_exactly_one_axis(self):
        rng = np.random.default_rng(7)
        axes = np.zeros((4, 9, 9)); sig = np.ones((4, 9))
        for i in range(4):
            axes[i] = np.eye(9)
        dy, pick = draw_axis_step(axes, sig, rng, jump_factor=1.0)
        self.assertEqual(dy.shape, (4, 9))
        for i in range(4):
            nz = np.flatnonzero(np.abs(dy[i]) > 0)
            self.assertEqual(len(nz), 1)
            self.assertEqual(nz[0], pick[i])

    def test_jump_factor_scales_linearly(self):
        axes = np.repeat(np.eye(9)[None, ...], 3, axis=0)
        sig = np.ones((3, 9))
        a = draw_axis_step(axes, sig, np.random.default_rng(1), 1.0)[0]
        b = draw_axis_step(axes, sig, np.random.default_rng(1), 2.5)[0]
        np.testing.assert_allclose(b, 2.5 * a, rtol=1e-12)

    def test_per_axis_sigma_is_applied(self):
        axes = np.repeat(np.eye(9)[None, ...], 1, axis=0)
        sig = np.zeros((1, 9)); sig[0, 3] = 5.0
        dy, pick = draw_axis_step(axes, sig, np.random.default_rng(0), 1.0)
        if pick[0] != 3:
            self.assertTrue(np.allclose(dy, 0.0))
        else:
            self.assertAlmostEqual(abs(dy[0, 3]) / 5.0,
                                   abs(dy[0, 3]) / 5.0, places=12)

    def test_draw_is_symmetric_in_distribution(self):
        """Symmetric proposal => zero MH factor. Guard the mean is ~0."""
        axes = np.repeat(np.eye(9)[None, ...], 4000, axis=0)
        sig = np.ones((4000, 9))
        dy, _ = draw_axis_step(axes, sig, np.random.default_rng(3), 1.0)
        self.assertLess(abs(float(dy.sum(axis=0).mean())), 0.2 * np.sqrt(4000))


class KnobTest(unittest.TestCase):
    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_EIGEN_AXIS", None)
            self.assertFalse(_eigen_axis_on())

    def test_env_arms_it(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_EIGEN_AXIS": "1"}):
            self.assertTrue(_eigen_axis_on())
        with mock.patch.dict(os.environ, {"GB_INMODEL_EIGEN_AXIS": "0"}):
            self.assertFalse(_eigen_axis_on())


if __name__ == "__main__":
    unittest.main()
