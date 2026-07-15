"""Tests for the grid + inverse-CDF F-stat proposal
(:mod:`lisatools.sampling.fstat_proposal`).

Hermetic: the F-stat evaluator is a mock whose ``(N, M)`` encode a known
4-D Gaussian, so every property (peak location, sampling moments,
normalization, rvs/logpdf consistency) is checked against analytics without
any WDM data.
"""

import copy
import pickle
import unittest

import numpy as np

from lisatools.sampling.fstat_proposal import (
    FStatProposal4D,
    GridSpec,
    compute_fstat,
)

MU = np.array([20.38, 0.466, 4.06, -0.786])
SIGMA = np.array([0.10, 0.04, 0.10, 0.0225])
F_PEAK = 50.0


class MockGaussianFstat:
    """Mock evaluator: F is a 4-D Gaussian in the sampling basis."""

    def get_fstat_ll_wdm(self, params, wdm_holder):
        from gbgpu.utils.utility import get_chirp_mass_from_f_fdot

        f0_mHz = params[:, 1] * 1e3
        Mc = np.asarray(get_chirp_mass_from_f_fdot(params[:, 1], params[:, 2]))
        theta = np.stack(
            [f0_mHz, Mc, params[:, 7], np.sin(params[:, 8])], axis=-1
        )
        F = -0.5 * np.sum(((theta - MU) / SIGMA) ** 2, axis=-1) + F_PEAK
        F = np.clip(F, 1e-6, None)

        n = params.shape[0]
        N = np.zeros((n, 4))
        N[:, 0] = np.sqrt(2.0 * F)
        M_upper = np.zeros((n, 10))
        M_upper[:, [0, 4, 7, 9]] = 1.0  # identity, row-major upper triangle
        return N, M_upper


def _make_prop(n_per_axis=32, beta=1.0, seed=7):
    grid = GridSpec(
        f0_range=(MU[0] - 1.0, MU[0] + 1.0),
        Mc_range=(MU[1] - 0.2, MU[1] + 0.2),
        alpha_range=(MU[2] - 0.5, MU[2] + 0.5),
        sin_delta_range=(MU[3] - 0.15, MU[3] + 0.15),
        n_f0=n_per_axis, n_Mc=n_per_axis,
        n_alpha=n_per_axis, n_sin_delta=n_per_axis,
    )
    return FStatProposal4D(
        MockGaussianFstat(), None, grid, beta=beta, seed=seed
    )


class ComputeFstatTest(unittest.TestCase):
    def test_identity_gram(self):
        rng = np.random.default_rng(3)
        N = rng.normal(size=(64, 4))
        M = np.zeros((64, 10))
        M[:, [0, 4, 7, 9]] = 1.0
        F = compute_fstat(N, M)
        expected = 0.5 * np.sum(N**2, axis=-1)
        self.assertTrue(np.allclose(F, expected, rtol=1e-9))

    def test_general_gram(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(16, 4, 4))
        M4 = A @ np.swapaxes(A, 1, 2) + 4 * np.eye(4)  # SPD
        N = rng.normal(size=(16, 4))
        rows = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
        cols = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)
        M_upper = np.stack([M4[:, i, j] for i, j in zip(rows, cols)], axis=1)
        F = compute_fstat(N, M_upper)
        expected = 0.5 * np.einsum(
            "bi,bij,bj->b", N, np.linalg.inv(M4), N
        )
        self.assertTrue(np.allclose(F, expected, rtol=1e-6))

    def test_singular_gram_returns_finite_or_neginf(self):
        N = np.ones((2, 4))
        M_upper = np.zeros((2, 10))  # fully singular
        F = compute_fstat(N, M_upper)
        self.assertEqual(F.shape, (2,))
        self.assertFalse(np.isnan(F).any())


class ProposalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prop = _make_prop()

    def test_grid_argmax_at_source(self):
        g = np.asarray(self.prop._logp_grid)
        idx = np.unravel_index(np.argmax(g), g.shape)
        peak = np.array([self.prop._axes[j][idx[j]] for j in range(4)])
        # within one cell of the injected mean
        self.assertTrue(np.all(np.abs(peak - MU) <= self.prop._dx + 1e-12))

    def test_rvs_moments(self):
        s = np.asarray(self.prop.rvs(size=(200_000,)))
        self.assertEqual(s.shape, (200_000, 4))
        mean = s.mean(axis=0)
        std = s.std(axis=0)
        # trapezoid cells remove the half-cell bias: mean to ~0.1 sigma
        self.assertTrue(np.all(np.abs(mean - MU) < 0.1 * SIGMA))
        self.assertTrue(np.all(np.abs(std / SIGMA - 1) < 0.1))

    def test_logpdf_normalized(self):
        # MC integral of exp(logpdf) over the box must be ~1.
        rng = np.random.default_rng(11)
        lo, hi = self.prop._lo, self.prop._hi
        pts = rng.uniform(lo, hi, size=(200_000, 4))
        vol = float(np.prod(hi - lo))
        integ = float(np.mean(np.exp(self.prop.logpdf(pts)))) * vol
        self.assertLess(abs(integ - 1.0), 0.05)

    def test_rvs_logpdf_consistent(self):
        # E_p[-logpdf] via rvs draws == entropy from the cell weights.
        s = self.prop.rvs(size=(100_000,))
        lp = np.asarray(self.prop.logpdf(s))
        self.assertTrue(np.all(np.isfinite(lp)))
        w = np.exp(self.prop._log_wcell - self.prop._log_norm)
        cell_vol = float(np.prod(self.prop._dx))
        p_cell = w * cell_vol  # cell probabilities
        entropy = -np.sum(p_cell * np.log(np.clip(w, 1e-300, None)))
        self.assertLess(abs(-lp.mean() - entropy) / abs(entropy), 0.02)

    def test_outside_box_neg_inf(self):
        x = np.array([[MU[0] + 100.0, MU[1], MU[2], MU[3]]])
        self.assertEqual(float(self.prop.logpdf(x)[0]), -np.inf)

    def test_beta_tempering_broadens(self):
        prop_cold = _make_prop(beta=0.25, seed=8)
        s_hot = np.asarray(self.prop.rvs(size=(50_000,)))
        s_cold = np.asarray(prop_cold.rvs(size=(50_000,)))
        # beta = 0.25 doubles every Gaussian width
        self.assertTrue(np.all(s_cold.std(axis=0) > 1.5 * s_hot.std(axis=0)))

    def test_eryn_shapes(self):
        s = self.prop.rvs(size=(5, 3))
        self.assertEqual(np.asarray(s).shape, (5, 3, 4))
        lp = self.prop.logpdf(np.asarray(s).reshape(-1, 4))
        self.assertEqual(np.asarray(lp).shape, (15,))

    def test_from_grid_matches(self):
        clone = FStatProposal4D.from_grid(
            self.prop._axes, self.prop._logp_grid, seed=3
        )
        x = np.array([MU, MU + 0.5 * SIGMA])
        self.assertTrue(np.allclose(
            np.asarray(clone.logpdf(x)), np.asarray(self.prop.logpdf(x))
        ))
        s = np.asarray(clone.rvs(size=(100_000,)))
        self.assertTrue(np.all(np.abs(s.mean(axis=0) - MU) < 0.1 * SIGMA))

    def test_pickle_keeps_grid(self):
        clone = pickle.loads(pickle.dumps(copy.deepcopy(self.prop)))
        self.assertIsNone(clone.gb_wdm_comp)
        x = np.array([MU])
        self.assertAlmostEqual(
            float(np.asarray(clone.logpdf(x))[0]),
            float(np.asarray(self.prop.logpdf(x))[0]),
        )
        self.assertEqual(np.asarray(clone.rvs(size=(10,))).shape, (10, 4))


if __name__ == "__main__":
    unittest.main()
