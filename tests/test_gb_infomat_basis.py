"""The GB in-model proposal information matrix, in the SAMPLING basis.

The proposal draws Gaussian jumps in the sampled coordinates

    y = [dist, f0(mHz), Mc, phi0, cos_i, psi, alpha, sin_d, r]

but the likelihood engines only ever return curvature in the PHYSICAL ones

    x = [A, f0(Hz), fdot, fddot, phi0, iota, psi, lam, beta].

The bridge is the congruence ``Gamma_y = J^T Gamma_x J`` with the full
Jacobian ``J[a, i] = dx[test_inds[a]] / dy_i``. That map is what these tests
pin down, because two things went wrong in it and both silently destroyed the
in-model move rather than raising:

1. ``J`` was kept DIAGONAL. The transform is not separable -- ``test_inds``
   pairs Mc with fdot and fdot_astro_ratio with the dead fddot slot, while the
   real map is ``A = A(f0, Mc, dist)`` and ``fdot = fdot_gr(f0, Mc)*(1+r)``.
   So Mc was scored on ``d(fdot)/d(Mc)`` alone and ``r`` was handed exactly
   zero curvature (its target column is identically zero).
2. The fdot CONDITIONING scale ``s = 1e-16`` -- which exists for a basis that
   samples fdot directly -- was resolved by matching ``("fdot", "Mc")``, so on
   the chirp-mass basis it landed on Mc, whose natural scale is O(0.1-1). That
   pushed the Mc eigenvalue under the relative eigen-floor, and the Mc
   proposal width came out as ``1e-16 / sqrt(1e-10 * lambda_max)``: a number
   set by the floor rather than by any curvature.

Ground truth throughout is the thing that needs no Jacobian at all -- a direct
central second difference of ``lnL(x(y))`` with respect to ``y``.

``AnalyticGroundTruthTest`` does this against an exact quadratic likelihood
(so the physical matrix is known in closed form and only the basis map is
under test); ``RealWaveformGroundTruthTest`` does it against the real FD GB
likelihood with the source injected into the data, where the residual term of
the observed information vanishes and the second difference IS the Fisher.
"""

from __future__ import annotations

import unittest

import numpy as np


def _have_gbgpu() -> bool:
    try:
        import gbgpu  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


# sampling / physical bases and the reorder the stock 9-column container uses
IB = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha", "sin_delta",
      "fdot_astro_ratio"]
TEST_INDS = np.array([0, 1, 2, 4, 5, 6, 7, 8, 3])
NDIM = 9
FDOT_ASTRO_COL = 8


def _stretch_move(use_distance):
    """A real ``GBSpecialStretchMove`` on the CPU flow fixture."""
    from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

    from .test_gbspecial_flow import build_fixture

    fx = build_fixture(use_distance=use_distance,
                       use_fdot_astro=use_distance)
    move = GBSpecialStretchMove(
        *fx["move_args"], is_rj_prop=False, name="infomat_basis",
        stretch_probability=0.0, **fx["move_kwargs"],
    )
    move.temperature_control = fx["temperature_control"]
    move.time = 0
    return move, fx


def _scales(move, ndim):
    """The conditioning vector ``_compute_proposal_cholesky`` builds."""
    s = np.ones(ndim)
    if move._fdot_col is not None:
        s[move._fdot_col] = move._fdot_scale
    return s


def _proposal_widths(info_y, s, floor_rel=1e-10):
    """Per-coordinate proposal sigma, exactly as the move forms it.

    ``chol = V / sqrt(max(|lambda|, floor))``, ``dy = chol @ randn``, and the
    step applied to the coordinates is ``dy * s`` -- so the marginal width of
    coordinate ``i`` is the row norm of ``chol * s[:, None]``.
    """
    ev, V = np.linalg.eigh(info_y)
    floor = floor_rel * max(float(np.abs(ev).max()), 1e-300)
    ev = np.maximum(np.abs(ev), floor)
    chol = V / np.sqrt(ev)[None, :]
    return np.sqrt(((chol * s[:, None]) ** 2).sum(axis=1)), chol


def _second_difference_infomat(ll_of_y, y0, eps_y):
    """``-d^2 lnL / dy_i dy_j`` -- the sampling-basis matrix, no Jacobian."""
    n = len(y0)
    G = np.zeros((n, n))
    l0 = ll_of_y(y0)
    for i in range(n):
        e = np.zeros(n); e[i] = eps_y[i]
        G[i, i] = -(ll_of_y(y0 + e) - 2.0 * l0 + ll_of_y(y0 - e)) / eps_y[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = eps_y[i]
            ej = np.zeros(n); ej[j] = eps_y[j]
            v = (ll_of_y(y0 + ei + ej) - ll_of_y(y0 + ei - ej)
                 - ll_of_y(y0 - ei + ej) + ll_of_y(y0 - ei - ej))
            G[i, j] = G[j, i] = -v / (4.0 * eps_y[i] * eps_y[j])
    return G


# ---------------------------------------------------------------------------


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class ConditioningColumnTest(unittest.TestCase):
    """``_fdot_scale`` may only condition a literally-sampled fdot."""

    def test_chirp_mass_basis_has_no_conditioning_column(self):
        move, _ = _stretch_move(use_distance=True)
        self.assertEqual(list(move.transform_fn.input_basis), IB)
        self.assertIsNone(
            move._fdot_col,
            "the distance/chirp-mass basis samples Mc, not fdot: applying the "
            "1e-16 fdot conditioning scale to it drives the Mc eigenvalue "
            "under the eigen-floor and freezes the Mc proposal",
        )
        self.assertEqual(move._fdot_astro_col, FDOT_ASTRO_COL)
        np.testing.assert_array_equal(_scales(move, NDIM), np.ones(NDIM))

    def test_legacy_basis_still_conditions_fdot(self):
        move, _ = _stretch_move(use_distance=False)
        ib = list(move.transform_fn.input_basis)
        self.assertIn("fdot", ib)
        self.assertEqual(move._fdot_col, ib.index("fdot"))
        self.assertIsNone(move._fdot_astro_col)
        s = _scales(move, len(ib))
        self.assertEqual(s[move._fdot_col], move._fdot_scale)


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class AnalyticGroundTruthTest(unittest.TestCase):
    """Basis map against an exact quadratic, where the physical matrix is known.

    The likelihood is ``-0.5 (x - x0)^T M (x - x0)`` with realistic GB
    curvature scales and a non-separable ``M``; ``fddot`` is dead exactly as
    it is in the waveform. Second differences are then EXACT, so any error is
    attributable to the basis map alone.
    """

    SNR = 20.0
    Y0 = np.array([5.0, 6.30, 0.30, 1.2, 0.35, 0.9, 2.1, -0.25, 0.02])

    @classmethod
    def setUpClass(cls):
        cls.move, _ = _stretch_move(use_distance=True)
        cls.tc = cls.move.transform_fn
        cls.x0 = cls._phys(cls.Y0)[0]
        cls.M = cls._curvature(cls.x0, cls.SNR)

    @classmethod
    def _phys(cls, y):
        return cls.move.transform_fn.both_transforms(
            np.atleast_2d(np.asarray(y, float)).copy(), xp=np)

    @staticmethod
    def _curvature(x0, snr, seed=0, mix=0.35):
        """Realistic GB Fisher scales, rotated so ``M`` is not separable."""
        from lisatools.utils.constants import YRSID_SI

        tobs = 0.25 * YRSID_SI
        sig = np.array([
            x0[0] / snr,                        # A
            1.0 / (2 * np.pi * snr * tobs),     # f0   [Hz]
            1.0 / (np.pi * snr * tobs ** 2),    # fdot [Hz/s]
            np.inf,                             # fddot -- dead in the waveform
            1.0 / snr, 1.0 / snr, 1.0 / snr,    # phi0, iota, psi
            3.0 / snr, 3.0 / snr,               # lam, beta
        ])
        live = np.array([i for i in range(9) if i != 3])
        d = 1.0 / sig[live]
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((8, 8)) * mix
        q, _ = np.linalg.qr(np.eye(8) + a - a.T)
        m8 = (d[:, None] * q) @ (d[:, None] * q).T
        M = np.zeros((9, 9))
        M[np.ix_(live, live)] = m8
        return M

    def _ll_phys(self, rows):
        d = np.atleast_2d(np.asarray(rows, float)) - self.x0[None, :]
        return -0.5 * np.einsum("ni,ij,nj->n", d, self.M, d)

    def _ground_truth(self, eps_scale=1.0):
        base = np.array([1e-3, 1e-6, 1e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
        return _second_difference_infomat(
            lambda y: float(self._ll_phys(self._phys(y))[0]),
            self.Y0, base * eps_scale)

    def _info_phys(self):
        """The production physical kernel; exact on a quadratic."""
        from lisatools.info_matrix_ll import information_matrix_from_ll

        eps = np.array([1e-25, 2e-14, 1e-21, 1e-28, 1e-6, 1e-6, 1e-6, 1e-6,
                        1e-6])
        return information_matrix_from_ll(
            self._ll_phys, self.x0[None, :], xp=np, param_eps=eps,
            inds=TEST_INDS.tolist(), psd_project=False)[0]

    # -- the tests ---------------------------------------------------------

    def test_ground_truth_is_step_size_stable(self):
        g_half, g_one, g_two = (self._ground_truth(f) for f in (0.5, 1.0, 2.0))
        for other, tag in ((g_half, "x0.5"), (g_two, "x2")):
            rel = np.abs(np.diag(other) - np.diag(g_one)) / np.abs(
                np.diag(g_one))
            self.assertLess(float(np.nanmax(rel)), 1e-4, tag)

    def test_physical_kernel_is_exact_on_a_quadratic(self):
        got = self._info_phys()
        want = self.M[np.ix_(TEST_INDS, TEST_INDS)]
        scale = np.sqrt(np.abs(np.outer(np.diag(want), np.diag(want))))
        rel = np.abs(got - want) / np.maximum(scale, 1e-300)
        self.assertLess(float(np.nanmax(rel)), 1e-3)

    def test_full_congruence_reproduces_the_sampling_basis_matrix(self):
        """The headline: the fixed map == a direct second difference in y."""
        g_true = self._ground_truth()
        info_phys = self._info_phys()
        s = _scales(self.move, NDIM)
        j = self.move._infomat_jacobian(
            np.atleast_2d(self.Y0).copy(), TEST_INDS, s)[0]
        info_y = j.T @ info_phys @ j

        scale = np.sqrt(np.abs(np.outer(np.diag(g_true), np.diag(g_true))))
        rel = np.abs(info_y - g_true) / np.maximum(scale, 1e-300)
        self.assertLess(
            float(np.nanmax(rel)), 1e-3,
            "full-congruence sampling-basis information matrix disagrees with "
            "a direct second difference of lnL(x(y))")

        # every column, including the two the diagonal map got wrong
        ratio = np.diag(info_y) / np.diag(g_true)
        for i in range(NDIM):
            self.assertAlmostEqual(
                ratio[i], 1.0, delta=2e-3,
                msg=f"column {IB[i]} off by {ratio[i]:.4f}x")

    def test_diagonal_jacobian_is_the_regression_it_replaced(self):
        """Pin the two failures so neither can come back unnoticed."""
        g_true = self._ground_truth()
        info_phys = self._info_phys()
        s_bad = np.ones(NDIM)
        s_bad[IB.index("Mc")] = self.move._fdot_scale   # the old ("fdot","Mc")
        j_full = self.move._infomat_jacobian(
            np.atleast_2d(self.Y0).copy(), TEST_INDS, s_bad)[0]
        j_diag = np.diag(np.diag(j_full))
        info_old = j_diag.T @ info_phys @ j_diag

        w_old, chol_old = _proposal_widths(info_old, s_bad)
        w_true = np.sqrt(np.maximum(
            np.diag(np.linalg.pinv(g_true, rcond=1e-12)), 0.0))

        mc = IB.index("Mc")
        self.assertLess(
            w_old[mc] / w_true[mc], 1e-10,
            "the old path is supposed to have frozen Mc; if this no longer "
            "reproduces, the regression pin is stale")
        # ... and the old code had to zero the ratio row because the diagonal
        # map gave it no curvature at all.
        self.assertAlmostEqual(float(info_old[FDOT_ASTRO_COL,
                                              FDOT_ASTRO_COL]), 0.0, places=12)

        s_new = _scales(self.move, NDIM)
        j_new = self.move._infomat_jacobian(
            np.atleast_2d(self.Y0).copy(), TEST_INDS, s_new)[0]
        w_new, _ = _proposal_widths(j_new.T @ info_phys @ j_new, s_new)
        self.assertGreater(w_new[mc] / w_true[mc], 0.5)
        self.assertLess(w_new[mc] / w_true[mc], 2.0)
        self.assertGreater(
            float(j_new.T @ info_phys @ j_new)
            if np.isscalar(j_new) else
            float((j_new.T @ info_phys @ j_new)[FDOT_ASTRO_COL,
                                                FDOT_ASTRO_COL]),
            0.0,
            "fdot_astro_ratio must carry real curvature under the full map")


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class RealWaveformGroundTruthTest(unittest.TestCase):
    """Same check against the real FD GB likelihood, source injected.

    With ``d = h(x0)`` exactly, the residual term of the observed information
    vanishes at ``x0``, so a second difference of ``lnL(x(y))`` in ``y`` IS
    the sampling-basis Fisher -- and it is computed with no Jacobian anywhere.
    """

    @classmethod
    def setUpClass(cls):
        cls.move, cls.fx = _stretch_move(use_distance=True)
        cls.comp = cls.move.gb_fd_comp
        cls.acs = cls.fx["acs"]

        sorter_coords = cls.fx["state"].branches["gb"].coords[0, 0]
        alive = cls.fx["state"].branches["gb"].inds[0, 0]
        cls.y0 = np.asarray(sorter_coords[alive][0], float)
        cls.x0 = cls._phys(cls.y0)[0]

        # Inject h(x0) into every walker's (zero) data row so lnL peaks
        # exactly at x0. Filling THROUGH the holder is what gives each row
        # the window start the likelihood reads it back at -- a raw ndarray
        # target defaults to start 0 and lands disjoint from the data window,
        # which shows up as <d|h> == 0 rather than as an error.
        n_rows = int(cls.acs.data_shaped[0].shape[0])
        cls.comp.fill_global(
            np.tile(cls.x0, (n_rows, 1)), cls.acs,
            data_index=np.arange(n_rows, dtype=np.int32),
        )
        # ``comp.d_d`` is left at its default: it is an additive CONSTANT in
        # get_ll_fd, so it cancels identically out of every second difference.

    @classmethod
    def _phys(cls, y):
        return cls.move.transform_fn.both_transforms(
            np.atleast_2d(np.asarray(y, float)).copy(), xp=np)

    def _ll_of_y(self, y):
        p = self._phys(y)
        return float(self.comp.get_ll_fd(
            p, self.acs,
            data_index=np.zeros(1, dtype=np.int32),
            noise_index=np.zeros(1, dtype=np.int32))[0])

    def test_full_congruence_matches_the_real_likelihood_curvature(self):
        # the injection must actually be an injection: lnL has to PEAK at y0,
        # which is what makes the second difference the Fisher matrix.
        l0 = self._ll_of_y(self.y0)
        for i in range(NDIM):
            d = np.zeros(NDIM)
            d[i] = 1e-2 * max(abs(self.y0[i]), 1e-2)
            self.assertGreater(l0, self._ll_of_y(self.y0 + d), IB[i])
            self.assertGreater(l0, self._ll_of_y(self.y0 - d), IB[i])

        eps = np.array([
            max(abs(self.y0[0]), 1.0) * 3e-3,   # dist  [kpc]
            2e-7,                               # f0    [mHz]
            3e-4,                               # Mc    [Msol]
            3e-3, 3e-3, 3e-3, 3e-3, 3e-3,       # phase / angles
            3e-2,                               # r
        ])
        g_true = _second_difference_infomat(self._ll_of_y, self.y0, eps)

        info_phys = self.comp.information_matrix(
            self.x0[None, :], self.acs, inds=TEST_INDS.tolist(),
            noise_index=np.zeros(1, dtype=np.int32))[0]
        s = _scales(self.move, NDIM)
        j = self.move._infomat_jacobian(
            np.atleast_2d(self.y0).copy(), TEST_INDS, s)[0]
        info_y = np.asarray(j.T @ info_phys @ j)

        ratio = np.diag(info_y) / np.diag(g_true)
        for i in range(NDIM):
            self.assertGreater(ratio[i], 0.90, f"{IB[i]}: {ratio[i]:.4f}")
            self.assertLess(ratio[i], 1.10, f"{IB[i]}: {ratio[i]:.4f}")


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class ProposalCholeskyEndToEndTest(unittest.TestCase):
    """``_compute_proposal_cholesky`` through the real engine and band sorter."""

    def _chol(self):
        from lisatools.globalfit.moves.gbbands import BandSorter

        move, fx = _stretch_move(use_distance=True)
        move.nwalkers, move.ntemps = 4, 2
        sorter = BandSorter(
            fx["state"].branches["gb"], move.band_edges, move.band_N_vals,
            force_backend="cpu", transform_fn=fx["transform"],
            max_data_store_size=512, gb=fx["gb"],
            gb_fd_comp=move.gb_fd_comp,
            waveform_kwargs=fx["waveform_kwargs"],
        )
        ids = np.where(np.asarray(sorter.inds))[0][:4]
        chol = move._compute_proposal_cholesky(fx["model"], sorter, ids)
        return move, sorter, ids, np.asarray(chol)

    def test_factors_are_finite_and_the_ratio_row_is_live(self):
        move, sorter, ids, chol = self._chol()
        self.assertEqual(chol.shape, (len(ids), NDIM, NDIM))
        self.assertTrue(np.all(np.isfinite(chol)))
        # the fdot_astro_ratio freeze is retired: the full congruence gives
        # that column real curvature, so its proposal row must be non-zero
        row = np.abs(chol[:, FDOT_ASTRO_COL, :]).max(axis=1)
        self.assertTrue(
            np.all(row > 0.0),
            "fdot_astro_ratio proposal row is identically zero -- the freeze "
            "removed with the diagonal Jacobian has come back")

    def test_chirp_mass_step_is_physical_not_floor_set(self):
        move, sorter, ids, chol = self._chol()
        s = _scales(move, NDIM)
        widths = np.sqrt(((chol * s[None, :, None]) ** 2).sum(axis=2))
        mc = widths[:, IB.index("Mc")]
        # under the old path this came out ~1e-17 (1e-16 / sqrt(1e-10 lmax)).
        # a real chirp-mass step is O(1e-4 .. 1e-1) Msol on a fixture source.
        self.assertTrue(
            np.all(mc > 1e-8),
            f"Mc proposal widths {mc} are floor-set, not curvature-set")
        self.assertTrue(np.all(mc < 10.0), f"Mc widths implausibly large: {mc}")


if __name__ == "__main__":
    unittest.main()
