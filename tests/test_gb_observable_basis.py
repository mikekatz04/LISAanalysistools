"""Observable-basis bijection for the GB in-model proposal.

WHY (flagship forensics, 2026-08-31/09-01). The 9-column GB sampling basis
``[dist, f0(mHz), Mc, phi0, cos_iota, psi, alpha, sin_delta, r]`` hides two
defects from any proposal built on it:

1. ``(dist, Mc, r) -> (A, fdot)`` is 3->2, so one direction is EXACTLY
   likelihood-flat (measured ``t^T F t / lam_max = 4.7e-26``);
2. ``f0`` is anchored at the START of the data while the data constrains the
   frequency at the MIDDLE -- ``f0 = f_mid - fdot*T/2`` is a SHEAR, and a shear
   turns an uncorrelated pair into a correlated one. Slope
   ``-(T/2)*fdot_gr*T``, i.e. ``prop fdot*T^2``: 0.04 bins at 6.8 mHz, 3.1 bins
   at 20.4 mHz (verified across the band, measured/analytic 0.95-1.06).

Consequence measured in production: the proposal moves ``Mc`` and ``r`` in
combinations that CANCEL in ``fdot_total`` -- one eigen-axis moves ``r`` by
0.61 and ``ln(fdot)`` by 0.0062 -- so an isolated SNR-46 source sat at
``f0 -1.38 bins, fdot 1.35x truth`` for 19 iterations with a smooth monotone
climb to truth available and untaken.

This module is the fix's foundation: a bijection to observables
``z = [lnA, f_mid, fdot, phi0, cos_iota, psi, alpha, sin_delta, Mc]`` where
``fdot`` is a coordinate (so nothing can cancel in it) and ``f_mid``
decorrelates the shear. The sampling basis is UNCHANGED -- conversion is
internal to the proposal, which is what keeps ``f0`` anchored at ``t_ref`` as
the mission's observation span grows.
"""
import pickle
import unittest

import numpy as np

from lisatools.sampling.gb_observable_basis import (
    FDOT_K,
    GBObservableFiberBasis,
    f0_from_f_mid,
    f_mid_from_f0,
    fdot_coherence_width,
    fdot_gr,
    fdot_shear_hz,
    gb_observable_step_scales,
)

TOBS = 7.776e6
BIN_HZ = 1.0 / TOBS
DIST, F0, MC, PHI0, CI, PSI, AL, SD, R = range(9)
IN_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
            "sin_delta", "fdot_astro_ratio"]

# Flagship, sampling basis (round-trip validated against the catalogue row to
# 3.4e-16 through the stock transform container). f0 in mHz.
FLAGSHIP = np.array([9.05215813e00, 2.03803767e01, 4.65777687e-01,
                     -3.41840873e00, -8.83190852e-01, 3.89809240e-01,
                     4.06170662e00, -7.86384411e-01, -1.11022302e-15])


class _FakeTC:
    """Minimal stand-in exposing only ``input_basis``.

    The real container needs data/backends to build; the basis object only
    ever reads ``input_basis``, so this keeps phase-1 tests GPU-free and fast.
    """

    def __init__(self, basis=None):
        self.input_basis = list(basis if basis is not None else IN_BASIS)


def _basis(**kw):
    kw.setdefault("Tobs", TOBS)
    return GBObservableFiberBasis(_FakeTC(), **kw)


def _rows(n=64, seed=3):
    rng = np.random.default_rng(seed)
    y = np.repeat(FLAGSHIP[None, :], n, axis=0)
    y[:, DIST] = rng.uniform(0.2, 25.0, n)
    y[:, F0] = rng.uniform(3.0, 21.0, n)
    y[:, MC] = rng.uniform(0.02, 0.95, n)
    y[:, R] = rng.uniform(-0.8, 0.8, n)
    for c in (PHI0, AL):
        y[:, c] = rng.uniform(0.0, 2 * np.pi, n)
    y[:, PSI] = rng.uniform(0.0, np.pi, n)
    y[:, CI] = rng.uniform(-1.0, 1.0, n)
    y[:, SD] = rng.uniform(-1.0, 1.0, n)
    return y


class ScalarPhysicsTest(unittest.TestCase):
    """The scalar half, which the F-stat grid change also consumes."""

    def test_fdot_gr_matches_gbgpu(self):
        """Load-bearing: the F-stat grid rows are built with get_fdot.

        If these ever disagree the alignment silently degrades toward the
        unaligned case and looks like 'alignment did not help'.
        """
        from gbgpu.utils.utility import get_fdot
        f0 = np.array([3e-3, 7e-3, 1.2e-2, 2.038e-2])
        mc = np.array([0.05, 0.3, 0.6, 0.4657777])
        np.testing.assert_allclose(fdot_gr(f0, mc), get_fdot(f=f0, Mc=mc),
                                   rtol=1e-14, atol=0.0)

    def test_FDOT_K_reproduces_fdot_gr(self):
        f0, mc = 2.038e-2, 0.4657777
        self.assertAlmostEqual(
            fdot_gr(f0, mc) / (FDOT_K * mc ** (5 / 3) * f0 ** (11 / 3)), 1.0,
            places=12)

    def test_shear_pair_is_inverse(self):
        f0 = np.array([3e-3, 2.038e-2]); fd = np.array([1e-17, 1.02e-13])
        fm = f_mid_from_f0(f0, fd, TOBS)
        np.testing.assert_allclose(f0_from_f_mid(fm, fd, TOBS), f0,
                                   rtol=0, atol=1e-20)

    def test_shear_is_the_measured_slope(self):
        """-(T/2)*fdot*T in bins; measured -3.085 at the flagship, analytic -3.097."""
        fd = 1.02453309e-13
        d_bins = (f_mid_from_f0(2.038e-2, fd, TOBS) - 2.038e-2) / BIN_HZ
        self.assertAlmostEqual(d_bins, 3.0975, delta=0.01)

    def test_shear_scales_as_fdot_T_squared(self):
        """Negligible at low f, large at high f -- the whole reason this is
        a high-frequency problem."""
        lo = abs(fdot_shear_hz(0.3, 6.8e-3, TOBS)) / BIN_HZ
        hi = abs(fdot_shear_hz(0.3, 2.04e-2, TOBS)) / BIN_HZ
        self.assertLess(lo, 0.1)
        self.assertGreater(hi, 1.0)

    def test_coherence_width_aligned_is_coarser(self):
        """eta/(pi T^2) vs eta*sqrt(720)/(2 pi T^2) -- 13.4x at equal eta."""
        un = fdot_coherence_width(TOBS, aligned=False, eta=1.0)
        al = fdot_coherence_width(TOBS, aligned=True, eta=1.0)
        self.assertAlmostEqual(al / un, np.sqrt(720) / 2.0, places=9)
        self.assertAlmostEqual(al / un, 13.416, delta=0.01)

    def test_unaligned_width_is_byte_identical_to_the_live_rule(self):
        """fstat_n_mc must reproduce its pinned values with the knob off."""
        self.assertEqual(fdot_coherence_width(TOBS, aligned=False, eta=1.0),
                         1.0 / (np.pi * TOBS ** 2))


class BijectionTest(unittest.TestCase):
    def test_round_trip(self):
        b = _basis()
        y = _rows(256)
        np.testing.assert_allclose(b.from_internal(b.to_internal(y)), y,
                                   rtol=1e-12, atol=1e-15)

    def test_round_trip_pathological_rows(self):
        """r -> -1 (fdot -> 0), Mc at both box edges, dist at the floor."""
        b = _basis()
        y = np.repeat(FLAGSHIP[None, :], 5, axis=0)
        y[0, R] = -1.0 + 1e-12          # fdot ~ 0
        y[1, MC] = 1e-3                 # mc_lims low edge
        y[2, MC] = 1.0                  # mc_lims high edge
        y[3, DIST] = 1e-3               # 1 pc
        y[4, R] = -2.0                  # NEGATIVE fdot (21% of high-f leaves)
        back = b.from_internal(b.to_internal(y))
        np.testing.assert_allclose(back, y, rtol=1e-11, atol=1e-14)

    def test_internal_columns_are_the_documented_order(self):
        b = _basis()
        z = b.to_internal(FLAGSHIP[None, :])[0]
        self.assertEqual(b.INTERNAL_BASIS[2], "fdot")
        self.assertEqual(b.INTERNAL_BASIS[8], "Mc")
        self.assertAlmostEqual(z[8], FLAGSHIP[MC], places=15)
        # fdot column really is fdot_total, including the (1+r) factor
        want = fdot_gr(FLAGSHIP[F0] * 1e-3, FLAGSHIP[MC]) * (1 + FLAGSHIP[R])
        self.assertAlmostEqual(z[2] / want, 1.0, places=12)

    def test_extrinsic_columns_pass_through_untouched(self):
        b = _basis()
        y = _rows(32)
        z = b.to_internal(y)
        for zc, yc in ((3, PHI0), (4, CI), (5, PSI), (6, AL), (7, SD)):
            np.testing.assert_allclose(z[:, zc], y[:, yc], rtol=0, atol=0)


class ShearTest(unittest.TestCase):
    def test_f_mid_step_moves_f0_and_holds_the_observables(self):
        """A step in f_mid alone moves f0 by exactly that and leaves the
        OBSERVABLES (A, fdot) and the fiber coordinate Mc untouched.

        ``dist`` deliberately DOES move: ``A propto f0**(2/3) / dist``, so
        holding the observable amplitude fixed while f0 shifts requires the
        derived distance to follow. Asserting dist constant here would be
        asserting that A changes -- the opposite of what the basis is for.
        """
        b = _basis()
        y = _rows(16)
        z = b.to_internal(y)
        z2 = z.copy(); z2[:, 1] += 5.0 * BIN_HZ
        y2 = b.from_internal(z2)
        z2_back = b.to_internal(y2)

        np.testing.assert_allclose((y2[:, F0] - y[:, F0]) * 1e-3 / BIN_HZ, 5.0,
                                   rtol=1e-9)
        np.testing.assert_allclose(z2_back[:, 0], z[:, 0], rtol=1e-12)  # lnA
        np.testing.assert_allclose(z2_back[:, 2], z[:, 2], rtol=1e-12)  # fdot
        np.testing.assert_allclose(y2[:, MC], y[:, MC], rtol=1e-12)     # fiber

    def test_dist_follows_f0_as_the_two_thirds_power(self):
        """The derived-distance response, pinned explicitly.

        This is the one place a sign or exponent slip in the A <-> dist
        inverse would hide: it leaves the round trip exact (both directions
        use the same helper) while making every amplitude-direction step
        wrong. ``A propto f0**(2/3)/dist`` at fixed (A, Mc) => dist propto
        f0**(2/3).
        """
        b = _basis()
        y = _rows(16)
        z = b.to_internal(y)
        z2 = z.copy(); z2[:, 1] += 200.0 * BIN_HZ     # big enough to measure
        y2 = b.from_internal(z2)
        ratio = y2[:, DIST] / y[:, DIST]
        want = (y2[:, F0] / y[:, F0]) ** (2.0 / 3.0)
        np.testing.assert_allclose(ratio, want, rtol=1e-10)

    def test_fiber_step_leaves_A_f0_fdot_exactly_invariant(self):
        """The strongest single statement: moving the fiber coordinate Mc at
        fixed observables cannot change the waveform."""
        b = _basis()
        y = _rows(32)
        z = b.to_internal(y)
        z2 = z.copy(); z2[:, 8] *= 1.35          # move Mc along the fiber
        y2 = b.from_internal(z2)
        z2_back = b.to_internal(y2)
        for col, name in ((0, "lnA"), (1, "f_mid"), (2, "fdot")):
            np.testing.assert_allclose(z2_back[:, col], z[:, col], rtol=1e-10,
                                       err_msg=f"{name} moved along the fiber")


class JacobianTest(unittest.TestCase):
    """|dy/dz| = dist / fdot_gr(f0, Mc). Verified analytically, numerically,
    and by prior-invariance simulation before implementation."""

    def _numeric_logdet(self, b, y_row):
        z = b.to_internal(y_row[None, :])[0]
        eps = np.array([1e-7, 1e-13, 1e-20, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-8])
        J = np.zeros((9, 9))
        for j in range(9):
            zp = z.copy(); zm = z.copy()
            zp[j] += eps[j]; zm[j] -= eps[j]
            J[:, j] = ((b.from_internal(zp[None, :])[0]
                        - b.from_internal(zm[None, :])[0]) / (2 * eps[j]))
        return np.log(abs(np.linalg.det(J)))

    def test_matches_numerical_determinant_up_to_a_constant(self):
        b = _basis()
        y = _rows(12, seed=9)
        num = np.array([self._numeric_logdet(b, r) for r in y])
        ana = b.log_jacobian(y)
        off = num - ana
        self.assertLess(float(np.ptp(off)), 1e-5,
                        f"offset not constant: spread {np.ptp(off):.2e}")

    def test_is_the_analytic_closed_form(self):
        b = _basis()
        y = _rows(32)
        want = np.log(y[:, DIST]) - np.log(fdot_gr(y[:, F0] * 1e-3, y[:, MC]))
        np.testing.assert_allclose(b.log_jacobian(y) - want,
                                   b.log_jacobian(y)[0] - want[0], atol=1e-12)

    def test_shear_independent(self):
        """Verified numerically for cT in {0, T/2, 0.41T, T, -3T}: |J| identical
        to 8 s.f. A wrong T/2 must cost efficiency only, never correctness."""
        y = _rows(16)
        ref = _basis(shear=0.5).log_jacobian(y)
        for s in (0.0, 0.41, 1.0, -3.0):
            np.testing.assert_allclose(_basis(shear=s).log_jacobian(y), ref,
                                       rtol=1e-13)

    def test_factors_is_new_minus_old(self):
        """Sign confirmed by prior-invariance simulation: flipped collapses
        the dist marginal to KS p = 2.4e-38."""
        b = _basis()
        y0 = _rows(32, seed=1)
        y1 = _rows(32, seed=2)
        np.testing.assert_allclose(b.factors(y0, y1),
                                   b.log_jacobian(y1) - b.log_jacobian(y0),
                                   rtol=1e-12)

    def test_factors_of_a_null_move_is_zero(self):
        b = _basis()
        y = _rows(16)
        np.testing.assert_allclose(b.factors(y, y), 0.0, atol=0.0)

    def test_non_finite_rows_are_finite_and_negative(self):
        """factors is computed BEFORE the prior gate, so ln of a non-positive
        dist/Mc is reachable. nan must never reach the accept test -- NumPy and
        CUDA comparison semantics need not agree on it."""
        b = _basis()
        y0 = _rows(4); y1 = _rows(4, seed=7)
        y1[0, DIST] = -1.0
        y1[1, MC] = 0.0
        f = b.factors(y0, y1)
        self.assertTrue(np.all(np.isfinite(f)), f)
        self.assertLess(f[0], -1e200)
        self.assertLess(f[1], -1e200)


class ColumnResolutionTest(unittest.TestCase):
    """Pattern copied from ridge_fiber.py:73-88 -- raise loudly, never guess."""

    def test_missing_column_raises_with_the_basis_echoed(self):
        for drop in ("dist", "Mc", "fdot_astro_ratio", "f0"):
            bad = [c for c in IN_BASIS if c != drop]
            with self.assertRaises(ValueError) as cm:
                GBObservableFiberBasis(_FakeTC(bad), Tobs=TOBS)
            self.assertIn(drop, str(cm.exception))

    def test_indices_come_from_input_basis_not_position(self):
        """A reordered basis must resolve correctly, not by hard-coded slot."""
        perm = ["phi0", "dist", "cos_iota", "f0", "psi", "Mc", "alpha",
                "sin_delta", "fdot_astro_ratio"]
        b = GBObservableFiberBasis(_FakeTC(perm), Tobs=TOBS)
        self.assertEqual(b.dist_index, 1)
        self.assertEqual(b.f0_index, 3)
        self.assertEqual(b.mc_index, 5)
        self.assertEqual(b.ratio_index, 8)

    def test_tobs_is_required(self):
        with self.assertRaises(TypeError):
            GBObservableFiberBasis(_FakeTC())

    def test_pickles(self):
        b = _basis()
        y = _rows(8)
        b2 = pickle.loads(pickle.dumps(b))
        np.testing.assert_allclose(b2.to_internal(y), b.to_internal(y),
                                   rtol=0, atol=0)


class StepScaleTest(unittest.TestCase):
    def test_signature_cannot_see_coords(self):
        """API-level enforcement of 'state-dependence goes in the coordinate
        change, never the step size'. If it cannot see the point it cannot
        depend on it -- and a state-dependent scale silently breaks symmetry
        while leaving the acceptance rate looking healthy."""
        import inspect
        params = set(inspect.signature(gb_observable_step_scales).parameters)
        self.assertNotIn("coords", params)
        self.assertNotIn("y", params)

    def test_scales_go_as_one_over_snr(self):
        ex = np.ones(5)
        a = gb_observable_step_scales(np.array([20.0]), TOBS,
                                      extrinsic_scales=ex, mc_step=0.05)
        b = gb_observable_step_scales(np.array([40.0]), TOBS,
                                      extrinsic_scales=ex, mc_step=0.05)
        for c in (0, 1, 2):
            self.assertAlmostEqual(float(a[0, c] / b[0, c]), 2.0, places=9)

    def test_finite_and_positive_including_zero_snr(self):
        ex = np.ones(5)
        s = gb_observable_step_scales(np.array([0.0, 1e-9, 1e4]), TOBS,
                                      extrinsic_scales=ex, mc_step=0.05)
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertTrue(np.all(s > 0))

    def test_mc_column_is_prior_set(self):
        ex = np.ones(5)
        s = gb_observable_step_scales(np.array([46.0]), TOBS,
                                      extrinsic_scales=ex, mc_step=0.07)
        self.assertAlmostEqual(float(s[0, 8]), 0.07, places=12)


if __name__ == "__main__":
    unittest.main()
