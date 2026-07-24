"""Tests for the astrophysical (f0, Mc) GMM prior
(:mod:`lisatools.sampling.f0_mchirp_prior`).

The mixture parameters come from ``scripts/fstat_proposal/heatmap_GMMs.ipynb``;
these tests check the *wrapping*: the mHz-axis Jacobian, box truncation with
exact renormalization, eryn conventions, and the GBSetup integration.
"""

import copy
import pickle
import unittest

import numpy as np

from lisatools.sampling.f0_mchirp_prior import (
    F0McGMMSampling,
    HEATMAP_GMM_PARAMS,
)


def _notebook_density(f_log10Hz, Mc):
    """Independent reimplementation of the notebook's combined_fit_func."""
    from scipy.stats import multivariate_normal

    pt = np.column_stack([np.ravel(f_log10Hz), np.ravel(Mc)])
    out = np.zeros(len(pt))
    for w, mu, cov in zip(HEATMAP_GMM_PARAMS["weights"],
                          HEATMAP_GMM_PARAMS["means"],
                          HEATMAP_GMM_PARAMS["covs"]):
        out += w * multivariate_normal.pdf(pt, mean=mu, cov=cov)
    return out


def _box_integral(dist, n=600):
    """Quadrature of exp(logpdf) over the truncation box (log-f0 grid)."""
    x_lo = np.log10(dist.f0_lims_mHz[0])
    x_hi = np.log10(dist.f0_lims_mHz[1])
    xs = np.linspace(x_lo, x_hi, n)
    ms = np.linspace(dist.mc_lims[0], dist.mc_lims[1], n)
    xx, mm = np.meshgrid(xs, ms, indexing="ij")
    f0 = 10.0 ** xx
    p = np.exp(dist.logpdf(
        np.column_stack([f0.ravel(), mm.ravel()])
    )).reshape(xx.shape)
    # df0 = f0 ln10 dx
    integrand = p * f0 * np.log(10.0)
    return float(np.trapezoid(np.trapezoid(integrand, ms, axis=1), xs, axis=0))


class FullBoxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dist = F0McGMMSampling.from_heatmap(seed=1)

    def test_rvs_shapes_and_bounds(self):
        s = np.asarray(self.dist.rvs(size=(5000,)))
        self.assertEqual(s.shape, (5000, 2))
        self.assertTrue(np.all((s[:, 0] >= 0.1) & (s[:, 0] <= 100.0)))
        self.assertTrue(np.all((s[:, 1] >= 0.1) & (s[:, 1] <= 1.4)))
        s2 = self.dist.rvs(size=(3, 7))
        self.assertEqual(np.asarray(s2).shape, (3, 7, 2))

    def test_normalized(self):
        self.assertLess(abs(_box_integral(self.dist) - 1.0), 2e-3)

    def test_matches_notebook_density_shape(self):
        # interior points: exp(logpdf) * f0 * ln10 * Z == notebook density
        rng = np.random.default_rng(3)
        x = rng.uniform(-3.8, -1.2, 400)
        mc = rng.uniform(0.15, 1.3, 400)
        f0_mHz = 10.0 ** x * 1e3
        lp = self.dist.logpdf(np.column_stack([f0_mHz, mc]))
        ours = np.exp(lp) * f0_mHz * np.log(10.0) * self.dist._box_mass
        ref = _notebook_density(x, mc)
        self.assertTrue(np.allclose(ours, ref, rtol=1e-8))

    def test_marginal_sanity(self):
        s = np.asarray(self.dist.rvs(size=(50_000,)))
        logf = np.log10(s[:, 0] * 1e-3)
        # heatmap mass concentrates at low frequency, Mc ~ 0.5-0.6
        self.assertLess(np.median(logf), -2.8)
        self.assertGreater(np.median(logf), -3.6)
        self.assertLess(abs(np.median(s[:, 1]) - 0.57), 0.1)

    def test_outside_neg_inf(self):
        bad = np.array([
            [0.05, 0.5],    # f0 below box
            [200.0, 0.5],   # f0 above box
            [5.0, 0.05],    # Mc below box
            [5.0, 2.0],     # Mc above box
        ])
        lp = self.dist.logpdf(bad)
        self.assertTrue(np.all(np.isneginf(lp)))

    def test_pickle_deepcopy(self):
        clone = pickle.loads(pickle.dumps(copy.deepcopy(self.dist)))
        x = np.array([[3.0, 0.5]])
        self.assertAlmostEqual(float(clone.logpdf(x)[0]),
                               float(self.dist.logpdf(x)[0]))
        self.assertEqual(np.asarray(clone.rvs(size=8)).shape, (8, 2))

    def test_eryn_toggle_attrs(self):
        self.assertTrue(hasattr(self.dist, "use_cupy"))
        self.assertTrue(hasattr(self.dist, "return_gpu"))


class NarrowBandTest(unittest.TestCase):
    """A 3-WDM-layer band at 20.4 mHz: tiny box mass, rejection must cope."""

    @classmethod
    def setUpClass(cls):
        cls.dist = F0McGMMSampling.from_heatmap(
            f0_lims_mHz=(20.278, 20.417), mc_lims=(0.001, 1.0), seed=2
        )

    def test_normalized(self):
        self.assertLess(abs(_box_integral(self.dist) - 1.0), 2e-3)

    def test_rvs_in_box(self):
        s = np.asarray(self.dist.rvs(size=(2000,)))
        self.assertTrue(np.all((s[:, 0] >= 20.278) & (s[:, 0] <= 20.417)))
        self.assertTrue(np.all((s[:, 1] >= 0.001) & (s[:, 1] <= 1.0)))

    def test_mc_extrapolation_finite(self):
        # below the heatmap's Mc support but inside the run box: finite tails
        lp = self.dist.logpdf(np.array([[20.35, 0.05], [20.35, 0.5]]))
        self.assertTrue(np.all(np.isfinite(lp)))
        self.assertLess(lp[0], lp[1])  # tail density below the bulk


class GBSetupIntegrationTest(unittest.TestCase):
    """The stock GB prior container builds and behaves with the flag on."""

    @classmethod
    def setUpClass(cls):
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.gb import GBSetup

        fit = erebor.gb_no_fg(nwalkers=2, ntemps=1)
        gs = fit.gb
        gs.use_chirp_mass = True
        gs.use_astrophysical_f0_mc_prior = True
        gs.f0_lims = [6e-3, 2.5e-2]
        gs.Tobs = 365.25 * 24 * 3600.0
        gs.dt = 5.0
        cls.setup = GBSetup(gs)
        cls.setup.init_sampling_info()

    def test_prior_container_builds_and_draws(self):
        # astro prior on => 9-column DISTANCE basis: dist @0, f0 @1, Mc @2,
        # alpha @6, sin_delta @7, fdot_astro_ratio @8 (remapped by
        # reset_key_order from the ("dist","alpha","sin_delta") joint).
        self.assertTrue(self.setup.use_fdot_astro)
        self.assertTrue(self.setup.use_distance)
        self.assertEqual(self.setup.transform.input_basis[0], "dist")
        pri = self.setup.priors["gb"]
        s = np.asarray(pri.rvs(size=500))
        self.assertEqual(s.shape, (500, 9))
        # dist @0 in dist_lims (kpc)
        dlo, dhi = self.setup.dist_lims
        self.assertTrue(np.all((s[:, 0] >= dlo) & (s[:, 0] <= dhi)))
        # column layout: f0 [mHz] at 1, Mc at 2 -- bounds are the band
        # structure's resolved f0_lims, not the pre-init settings values
        f0_lims_mHz = np.asarray(self.setup.f0_lims) * 1e3
        self.assertTrue(np.all((s[:, 1] >= f0_lims_mHz[0])
                               & (s[:, 1] <= f0_lims_mHz[-1])))
        self.assertTrue(np.all((s[:, 2] >= 0.001) & (s[:, 2] <= 1.0)))
        # sky at cols 6/7 (from the remapped joint)
        self.assertTrue(np.all((s[:, 6] >= 0.0) & (s[:, 6] <= 2 * np.pi + 1e-9)))
        self.assertTrue(np.all(np.abs(s[:, 7]) <= 1.0 + 1e-9))
        # fdot_astro_ratio @8 ~ U[-M, M]
        M = self.setup.fdot_astro_ratio_max
        self.assertTrue(np.all(np.abs(s[:, 8]) <= M))
        lp = np.asarray(pri.logpdf(s))
        self.assertEqual(lp.shape, (500,))
        self.assertTrue(np.all(np.isfinite(lp)))
        # the derived physical amplitude is positive and finite
        phys = self.setup.transform.both_transforms(s[:32])
        self.assertTrue(np.all(phys[:, 0] > 0) and np.all(np.isfinite(phys[:, 0])))

    def test_astro_prior_reshapes_f0(self):
        # the GMM strongly prefers the low-f0 end of [6, 25] mHz vs uniform
        pri = self.setup.priors["gb"]
        s = np.asarray(pri.rvs(size=20_000))
        frac_low = np.mean(s[:, 1] < 10.0)  # uniform would give ~0.21
        self.assertGreater(frac_low, 0.5)

    def test_prior_pickles(self):
        clone = pickle.loads(pickle.dumps(copy.deepcopy(
            self.setup.priors["gb"]
        )))
        self.assertEqual(np.asarray(clone.rvs(size=16)).shape, (16, 9))

    def test_astro_prior_off_is_eight_col(self):
        # astro prior off => 8-column chirp-mass basis (no ratio column)
        from lisatools.globalfit.stock import erebor
        from lisatools.globalfit.stock.erebor.gb import GBSetup

        fit = erebor.gb_no_fg(nwalkers=2, ntemps=1)
        gs = fit.gb
        gs.use_chirp_mass = True
        gs.use_astrophysical_f0_mc_prior = False
        gs.f0_lims = [6e-3, 2.5e-2]
        gs.Tobs = 365.25 * 24 * 3600.0
        gs.dt = 5.0
        setup = GBSetup(gs)
        setup.init_sampling_info()
        self.assertFalse(setup.use_fdot_astro)
        s = np.asarray(setup.priors["gb"].rvs(size=200))
        self.assertEqual(s.shape, (200, 8))


if __name__ == "__main__":
    unittest.main()
