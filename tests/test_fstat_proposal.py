"""Tests for the grid + inverse-CDF F-stat proposal
(:mod:`lisatools.sampling.fstat_proposal`).

Hermetic: the F-stat evaluator is a mock whose ``(N, M)`` encode a known
4-D Gaussian, so every property (peak location, sampling moments,
normalization, rvs/logpdf consistency) is checked against analytics without
any WDM data.
"""

import copy
import pickle
import os
import unittest

import numpy as np

from lisatools.sampling.fstat_proposal import (
    CombIntrinsicProposal,
    FStatProposal4D,
    GridSpec,
    MixtureProposal,
    UniformFloorMixture,
    compute_fstat,
    make_gb_rj_birth_container,
)

MU = np.array([20.38, 0.466, 4.06, -0.786])
SIGMA = np.array([0.10, 0.04, 0.10, 0.0225])
F_PEAK = 50.0


# These tests were written against the Mc-basis grid, which is what
# FSTAT_FDOT_AXIS=0 selects -- and that path must keep working, since it is
# the documented escape from the fdot axis and the way an in-flight run
# resumes across the change. Pin it for the whole module rather than
# letting the new default silently retarget them; the fdot basis has its
# own end-to-end coverage in test_fstat_fdot_birth.
_FDOT_AXIS_SAVED = None


def setUpModule():
    global _FDOT_AXIS_SAVED
    _FDOT_AXIS_SAVED = os.environ.get("FSTAT_FDOT_AXIS")
    os.environ["FSTAT_FDOT_AXIS"] = "0"


def tearDownModule():
    if _FDOT_AXIS_SAVED is None:
        os.environ.pop("FSTAT_FDOT_AXIS", None)
    else:
        os.environ["FSTAT_FDOT_AXIS"] = _FDOT_AXIS_SAVED


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


FLOOR_LO = np.array([MU[0] - 1.0, 0.001, 0.0, -1.0])
FLOOR_HI = np.array([MU[0] + 1.0, 1.0, 2 * np.pi, 1.0])


class FloorMixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _make_prop(n_per_axis=16, seed=2)
        cls.mix = UniformFloorMixture(cls.base, FLOOR_LO, FLOOR_HI,
                                      eps=0.1, seed=4)

    def test_logpdf_finite_across_floor_box(self):
        # points far outside the base grid but inside the floor box
        rng = np.random.default_rng(9)
        pts = rng.uniform(FLOOR_LO, FLOOR_HI, size=(2000, 4))
        lp = self.mix.logpdf(pts)
        self.assertTrue(np.all(np.isfinite(lp)))
        # outside the floor box AND the base box -> -inf
        far = np.array([[MU[0] + 50.0, 0.5, 1.0, 0.0]])
        self.assertEqual(float(self.mix.logpdf(far)[0]), -np.inf)

    def test_normalized(self):
        rng = np.random.default_rng(10)
        pts = rng.uniform(FLOOR_LO, FLOOR_HI, size=(400_000, 4))
        vol = float(np.prod(FLOOR_HI - FLOOR_LO))
        integ = float(np.mean(np.exp(self.mix.logpdf(pts)))) * vol
        self.assertLess(abs(integ - 1.0), 0.05)

    def test_rvs_mix(self):
        s = np.asarray(self.mix.rvs(size=(50_000,)))
        self.assertEqual(s.shape, (50_000, 4))
        # ~90% of draws follow the narrow base near MU; ~10% floor-spread
        near = np.all(np.abs(s - MU) < 8 * SIGMA, axis=1).mean()
        self.assertGreater(near, 0.8)
        self.assertLess(near, 0.99)


class BirthContainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = _make_prop(n_per_axis=16, seed=3)
        mix = UniformFloorMixture(base, FLOOR_LO, FLOOR_HI, eps=0.1, seed=5)
        cls.A_lims = [1e-24, 1e-21]
        cls.dist = make_gb_rj_birth_container(mix, cls.A_lims)

    def test_rvs_shape_and_columns(self):
        s = np.asarray(self.dist.rvs(size=1000))
        self.assertEqual(s.shape, (1000, 8))
        lnA, f0, Mc, phi0, ci, psi, alpha, sd = s.T
        self.assertTrue(np.all((lnA >= np.log(self.A_lims[0]))
                               & (lnA <= np.log(self.A_lims[1]))))
        self.assertTrue(np.all((f0 >= FLOOR_LO[0]) & (f0 <= FLOOR_HI[0])))
        self.assertTrue(np.all((Mc >= 0.001) & (Mc <= 1.0)))
        self.assertTrue(np.all((phi0 >= 0) & (phi0 <= 2 * np.pi)))
        self.assertTrue(np.all((ci >= -1) & (ci <= 1)))
        self.assertTrue(np.all((psi >= 0) & (psi <= np.pi)))
        self.assertTrue(np.all((alpha >= 0) & (alpha <= 2 * np.pi)))
        self.assertTrue(np.all((sd >= -1) & (sd <= 1)))

    def test_logpdf_shape_and_consistency(self):
        s = np.asarray(self.dist.rvs(size=500))
        lp = np.asarray(self.dist.logpdf(s))
        self.assertEqual(lp.shape, (500,))
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_pickles(self):
        clone = pickle.loads(pickle.dumps(copy.deepcopy(self.dist)))
        s = np.asarray(clone.rvs(size=16))
        self.assertEqual(s.shape, (16, 8))


class DistanceRatioBirthContainerTest(unittest.TestCase):
    """9-column distance + fdot_astro_ratio birth container.

    This is the container the stock recipe now feeds directly into the RJ
    birth path (the earlier ``_rj_birth_prop = gpu_priors`` override that
    bypassed it has been removed). It must round-trip rvs -> logpdf, keep
    every column inside its prior support, and survive deepcopy/pickle, so a
    regression here surfaces before it reaches a GPU run. The array-module
    agnosticism the override worried about (cupy coords through ``logpdf``)
    can only be exercised on a device; the components dispatch via
    ``get_array_module`` on their inputs, so the CPU round-trip pins the
    layout and leaves the device path to the GPU smoke.
    """

    @classmethod
    def setUpClass(cls):
        base = _make_prop(n_per_axis=16, seed=3)
        mix = UniformFloorMixture(base, FLOOR_LO, FLOOR_HI, eps=0.1, seed=5)
        cls.A_lims = [1e-24, 1e-21]
        cls.dist_lims = [0.001, 40.0]
        cls.M = 5.0
        cls.dist = make_gb_rj_birth_container(
            mix, cls.A_lims, fdot_astro_ratio_max=cls.M,
            dist_lims=cls.dist_lims,
        )

    def test_rvs_shape_and_columns(self):
        s = np.asarray(self.dist.rvs(size=1000))
        self.assertEqual(s.shape, (1000, 9))
        dist, f0, Mc, phi0, ci, psi, alpha, sd, ratio = s.T
        # slot 0 is luminosity distance (kpc), NOT lnA
        self.assertTrue(np.all((dist >= self.dist_lims[0])
                               & (dist <= self.dist_lims[1])))
        self.assertTrue(np.all((f0 >= FLOOR_LO[0]) & (f0 <= FLOOR_HI[0])))
        self.assertTrue(np.all((Mc >= 0.001) & (Mc <= 1.0)))
        self.assertTrue(np.all((phi0 >= 0) & (phi0 <= 2 * np.pi)))
        self.assertTrue(np.all((ci >= -1) & (ci <= 1)))
        self.assertTrue(np.all((psi >= 0) & (psi <= np.pi)))
        self.assertTrue(np.all((alpha >= 0) & (alpha <= 2 * np.pi)))
        self.assertTrue(np.all((sd >= -1) & (sd <= 1)))
        self.assertTrue(np.all((ratio >= -self.M) & (ratio <= self.M)))

    def test_logpdf_shape_and_consistency(self):
        s = np.asarray(self.dist.rvs(size=500))
        lp = np.asarray(self.dist.logpdf(s))
        self.assertEqual(lp.shape, (500,))
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_pickles(self):
        clone = pickle.loads(pickle.dumps(copy.deepcopy(self.dist)))
        s = np.asarray(clone.rvs(size=16))
        self.assertEqual(s.shape, (16, 9))


class CombProposalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # synthetic comb: two triangular peaks, F ratio 4:1
        f0 = np.linspace(7.5, 7.6, 2001)
        F = np.zeros_like(f0)
        F += 400 * np.exp(-0.5 * ((f0 - 7.52) / 3e-4) ** 2)
        F += 100 * np.exp(-0.5 * ((f0 - 7.58) / 3e-4) ** 2)
        cls.comb = CombIntrinsicProposal(
            f0, F, mc_lims=(0.001, 1.0), seed=6
        )

    def test_rvs_mass_ratio_linear_in_F(self):
        s = np.asarray(self.comb.rvs(size=(100_000,)))
        n1 = np.sum(np.abs(s[:, 0] - 7.52) < 2e-3)
        n2 = np.sum(np.abs(s[:, 0] - 7.58) < 2e-3)
        self.assertAlmostEqual(n1 / n2, 4.0, delta=0.4)

    def test_logpdf_normalized(self):
        rng = np.random.default_rng(12)
        pts = np.column_stack([
            rng.uniform(7.5, 7.6, 300_000),
            rng.uniform(0.001, 1.0, 300_000),
            rng.uniform(0, 2 * np.pi, 300_000),
            rng.uniform(-1, 1, 300_000),
        ])
        vol = 0.1 * (1.0 - 0.001) * 2 * np.pi * 2.0
        integ = float(np.mean(np.exp(self.comb.logpdf(pts)))) * vol
        self.assertLess(abs(integ - 1.0), 0.05)

    def test_outside_neg_inf(self):
        self.assertEqual(
            float(self.comb.logpdf(np.array([[7.7, 0.5, 1.0, 0.0]]))[0]),
            -np.inf,
        )


class MixtureProposalTest(unittest.TestCase):
    def test_weighted_mixture(self):
        base = _make_prop(n_per_axis=12, seed=8)
        f0 = np.linspace(MU[0] - 1, MU[0] + 1, 501)
        F = np.full_like(f0, 10.0)
        comb = CombIntrinsicProposal(f0, F, mc_lims=(0.001, 1.0), seed=7)
        mix = MixtureProposal([base, comb], weights=[0.7, 0.3], seed=9)
        s = np.asarray(mix.rvs(size=(20_000,)))
        self.assertEqual(s.shape, (20_000, 4))
        lp = np.asarray(mix.logpdf(s))
        self.assertTrue(np.all(np.isfinite(lp)))
        # exact: mixture logpdf == logaddexp of weighted component logpdfs
        expected = np.logaddexp(
            np.log(0.7) + np.asarray(base.logpdf(s)),
            np.log(0.3) + np.asarray(comb.logpdf(s)),
        )
        self.assertTrue(np.allclose(lp, expected))
        # draws split between components ~ per the weights: comb-only region
        # (outside the base's narrow sin_delta box) holds ~the comb share
        outside_base = np.abs(s[:, 3] - MU[3]) > 0.15
        self.assertGreater(outside_base.mean(), 0.15)
        self.assertLess(outside_base.mean(), 0.45)


if __name__ == "__main__":
    unittest.main()


class AutoMcDensityTest(unittest.TestCase):
    """AUTO Mc node density (user rule 2026-08-26): one node per
    FSTAT_MC_ETA fdot-coherence widths (1/(pi*Tobs^2)) across the box's
    GR-fdot span at the box's f0 — the same criterion that justified 3
    nodes at 7.5 mHz (band75) and demands tens at the 20.4 mHz flagship
    (span scales as f0^{11/3})."""

    T3MO = 7776000.0
    T1YR = 31536000.0

    def setUp(self):
        self._saved = {}
        for k in ("FSTAT_N_MC", "FSTAT_N_PER_AXIS", "FSTAT_MC_ETA"):
            self._saved[k] = os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_flagship_band_needs_tens_of_nodes(self):
        from lisatools.sampling.fstat_proposal import fstat_n_mc
        n = fstat_n_mc(20.380377, 0.01, 1.0, self.T3MO)
        self.assertGreaterEqual(n, 40)
        self.assertLessEqual(n, 96)

    def test_low_frequency_band_stays_coarse(self):
        # the band75 regime: GR-fdot span ~2 coherence widths -> ~3 nodes
        from lisatools.sampling.fstat_proposal import fstat_n_mc
        n = fstat_n_mc(7.5803, 0.01, 1.0, self.T3MO)
        self.assertLessEqual(n, 6)
        self.assertGreaterEqual(n, 3)

    def test_one_year_clamps_at_the_ceiling(self):
        from lisatools.sampling.fstat_proposal import fstat_n_mc
        self.assertEqual(fstat_n_mc(20.380377, 0.01, 1.0, self.T1YR), 96)

    def test_explicit_env_wins(self):
        from lisatools.sampling.fstat_proposal import fstat_n_mc
        os.environ["FSTAT_N_MC"] = "24"
        self.assertEqual(fstat_n_mc(20.380377, 0.01, 1.0, self.T3MO), 24)

    def test_eta_loosens_the_grid(self):
        from lisatools.sampling.fstat_proposal import fstat_n_mc
        n1 = fstat_n_mc(20.380377, 0.01, 1.0, self.T3MO)
        os.environ["FSTAT_MC_ETA"] = "2.0"
        n2 = fstat_n_mc(20.380377, 0.01, 1.0, self.T3MO)
        self.assertLess(n2, n1)
        self.assertAlmostEqual(n2, (n1 + 1) / 2, delta=2)
