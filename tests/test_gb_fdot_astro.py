"""Tests for the GB fdot_astro ratio sampling basis.

The 9-column GB basis appends ``fdot_astro_ratio`` (``r = fdot_astro /
fdot_gr ~ U[-M, M]``); physical ``fdot = fdot_gr(f0, Mc) * (1 + r)`` with
``fddot`` exactly 0. Covers the transform factory (forward exactness,
mirror-convention inverse incl. fdot<0, pickle), the knob-off identity, the
prior/seeding wiring, and the 9-column RJ-birth / GMM containers.
"""

import copy
import pickle
import os
import unittest

import numpy as np

from gbgpu.utils.utility import get_fdot
from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container


MC_LIMS = (0.001, 1.0)


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


def _sampled_row(logA=None, f0_mHz=7.5, Mc=0.3, phi0=1.0, cos_i=0.4,
                 psi=1.0, alpha=2.0, sin_d=0.1, ratio=0.0):
    logA = np.log(1e-22) if logA is None else logA
    return np.array([[logA, f0_mHz, Mc, phi0, cos_i, psi, alpha, sin_d, ratio]])


class TransformFactoryTest(unittest.TestCase):
    def setUp(self):
        self.tc = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, mc_lims=MC_LIMS
        )

    def test_basis_shapes(self):
        self.assertEqual(self.tc.ndim, 9)
        self.assertEqual(self.tc.input_basis[-1], "fdot_astro_ratio")
        self.assertEqual(self.tc.input_basis[2], "Mc")

    def test_forward_fdot_and_fddot(self):
        f0_mHz, Mc, r = 7.5, 0.3, -2.0
        phys = self.tc.both_transforms(_sampled_row(f0_mHz=f0_mHz, Mc=Mc, ratio=r))
        fdot_gr = get_fdot(f=f0_mHz * 1e-3, Mc=Mc)
        # physical fdot = fdot_gr * (1 + r); here r=-2 => negative total fdot
        self.assertAlmostEqual(phys[0, 2], fdot_gr * (1.0 + r), delta=abs(fdot_gr) * 1e-12)
        self.assertLess(phys[0, 2], 0.0)
        # fddot slot exactly 0 (transform-zeroed, not filled)
        self.assertEqual(phys[0, 3], 0.0)
        # f0 in Hz
        self.assertAlmostEqual(phys[0, 1], f0_mHz * 1e-3, places=15)

    def test_inverse_mirror_convention(self):
        # fdot>0 in-box seeds at r=0; fdot<0 in-box seeds at r=-2
        for f0_mHz, Mc, r_expected in [(7.5, 0.3, 0.0), (7.5, 0.3, -2.0)]:
            phys = self.tc.both_transforms(_sampled_row(f0_mHz=f0_mHz, Mc=Mc, ratio=r_expected))
            back = self.tc.both_inverse_transforms(phys)
            self.assertTrue(MC_LIMS[0] <= back[0, 2] <= MC_LIMS[1])
            # round-trip reproduces physical fdot exactly (either sign)
            phys2 = self.tc.both_transforms(back)
            self.assertAlmostEqual(
                phys2[0, 2], phys[0, 2], delta=abs(phys[0, 2]) * 1e-10
            )

    def test_inverse_no_nan_for_fdot_negative(self):
        # a batch spanning fdot<0, fdot=0-ish, fdot>0 must invert without NaN
        rows = np.vstack([
            _sampled_row(Mc=0.3, ratio=-3.0),   # strongly negative fdot
            _sampled_row(Mc=0.3, ratio=-1.0),   # fdot ~ 0
            _sampled_row(Mc=0.3, ratio=0.0),    # fdot > 0
        ])
        phys = self.tc.both_transforms(rows)
        back = self.tc.both_inverse_transforms(phys)
        self.assertFalse(np.any(np.isnan(back)))
        self.assertTrue(np.all((back[:, 2] >= MC_LIMS[0]) & (back[:, 2] <= MC_LIMS[1])))

    def test_pickle_deepcopy(self):
        rows = _sampled_row(ratio=-2.0)
        phys = self.tc.both_transforms(rows)
        clone = pickle.loads(pickle.dumps(copy.deepcopy(self.tc)))
        self.assertTrue(np.array_equal(clone.both_transforms(rows), phys))

    def test_requires_chirp_mass(self):
        with self.assertRaises(ValueError):
            make_gb_transform_container(use_chirp_mass=False, use_fdot_astro=True)


class KnobOffIdentityTest(unittest.TestCase):
    """use_fdot_astro=False leaves today's containers bytewise-unchanged."""

    def test_chirp_mass_8col_unchanged(self):
        tc = make_gb_transform_container(use_chirp_mass=True)
        self.assertEqual(tc.ndim, 8)
        self.assertEqual(list(tc.input_basis)[-1], "sin_delta")
        self.assertNotIn("fdot_astro_ratio", tc.input_basis)

    def test_legacy_fdot_8col_unchanged(self):
        tc = make_gb_transform_container(use_chirp_mass=False)
        self.assertEqual(tc.ndim, 8)
        self.assertEqual(list(tc.input_basis)[2], "fdot")


class SeedingHelperTest(unittest.TestCase):
    """gb_fdot_rows_to_run_basis: catalogue fdot rows -> run sampling basis."""

    def setUp(self):
        # FDOT-basis rows (slot 2 = physical fdot): a fdot>0 and a fdot<0 source
        self.rows = np.array([
            [np.log(1e-22), 7.5803, 3.0e-16, 1.0, 0.5, 1.0, 2.0, 0.1],
            [np.log(5e-23), 7.56749, -9.9e-16, 0.3, -0.2, 0.5, 4.9, -0.06],
        ])

    def test_ratio_basis_represents_fdot_negative(self):
        from lisatools.globalfit.recipe import gb_fdot_rows_to_run_basis

        out = gb_fdot_rows_to_run_basis(
            self.rows, use_chirp_mass=True, use_fdot_astro=True, m_chirp_lims=MC_LIMS
        )
        self.assertEqual(out.shape, (2, 9))
        # seed through the run transform -> physical fdot equals the catalogue fdot
        tc = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, mc_lims=MC_LIMS
        )
        phys = tc.both_transforms(out)
        np.testing.assert_allclose(phys[:, 2], self.rows[:, 2], rtol=1e-9)
        self.assertLess(phys[1, 2], 0.0)   # fdot<0 source represented exactly
        # input array not mutated
        self.assertEqual(self.rows[1, 2], -9.9e-16)

    def test_chirp_mass_8col_floor_clamps(self):
        from lisatools.globalfit.recipe import gb_fdot_rows_to_run_basis

        out = gb_fdot_rows_to_run_basis(
            self.rows, use_chirp_mass=True, use_fdot_astro=False, m_chirp_lims=MC_LIMS
        )
        self.assertEqual(out.shape, (2, 8))
        self.assertEqual(out[1, 2], MC_LIMS[0])  # fdot<0 -> Mc floor

    def test_legacy_fdot_unchanged(self):
        from lisatools.globalfit.recipe import gb_fdot_rows_to_run_basis

        out = gb_fdot_rows_to_run_basis(
            self.rows, use_chirp_mass=False, use_fdot_astro=False, m_chirp_lims=MC_LIMS
        )
        np.testing.assert_array_equal(out, self.rows)


class SettingsWiringTest(unittest.TestCase):
    def test_use_fdot_astro_property(self):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        self.assertTrue(GBSettings(
            use_chirp_mass=True, use_astrophysical_f0_mc_prior=True
        ).use_fdot_astro)
        self.assertFalse(GBSettings(
            use_chirp_mass=True, use_astrophysical_f0_mc_prior=False
        ).use_fdot_astro)
        self.assertFalse(GBSettings(
            use_chirp_mass=False, use_astrophysical_f0_mc_prior=True
        ).use_fdot_astro)


class RJContainerTest(unittest.TestCase):
    """9-column RJ-birth / GMM containers draw a valid ratio column."""

    def test_gmm_container_9col(self):
        from lisatools.sampling.gmm import fit_gb_gmm_rj_container

        rng = np.random.default_rng(0)
        samples = rng.normal(size=(2, 300, 6)) * np.array(
            [0.5, 0.01, 0.05, 0.3, 0.3, 0.3]
        ) + np.array([-50, 7.5, 0.3, 0.0, 3.0, 0.0])
        c = fit_gb_gmm_rj_container(
            samples, use_chirp_mass=True, use_cupy=False, gpu=None,
            fdot_astro_ratio_max=5.0,
        )
        r = np.asarray(c.rvs(size=200))
        self.assertEqual(r.shape, (200, 9))
        self.assertTrue(np.all(np.abs(r[:, 8]) <= 5.0))
        lp = np.asarray(c.logpdf(r))
        self.assertEqual(lp.shape, (200,))
        self.assertTrue(np.all(np.isfinite(lp)))

    def test_birth_container_9col(self):
        from lisatools.sampling.fstat_proposal import make_gb_rj_birth_container

        rng = np.random.default_rng(1)

        class Intr4:
            ndim = 4
            use_cupy = False
            return_gpu = False

            def rvs(self, size, **kw):
                shape = (size,) if isinstance(size, int) else tuple(size)
                n = int(np.prod(shape))
                out = np.column_stack([
                    rng.uniform(7.4, 7.6, n), rng.uniform(0.1, 0.9, n),
                    rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n),
                ])
                return out.reshape((*shape, 4))

            def logpdf(self, x, **kw):
                return np.zeros(x.shape[0])

        c = make_gb_rj_birth_container(
            Intr4(), [7e-26, 1e-19], use_cupy=False, fdot_astro_ratio_max=5.0
        )
        b = np.asarray(c.rvs(size=50))
        self.assertEqual(b.shape, (50, 9))
        self.assertTrue(np.all(np.abs(b[:, 8]) <= 5.0))


DIST_LIMS = (0.001, 40.0)


def _dist_row(dist=10.0, f0_mHz=7.5, Mc=0.3, phi0=1.0, cos_i=0.4,
              psi=1.0, alpha=2.0, sin_d=0.1, ratio=0.0):
    return np.array([[dist, f0_mHz, Mc, phi0, cos_i, psi, alpha, sin_d, ratio]])


class DistanceBasisTest(unittest.TestCase):
    """The 9-column distance basis: slot 0 = dist(kpc), A derived."""

    def setUp(self):
        self.tc = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, use_distance=True,
            mc_lims=MC_LIMS,
        )

    def test_basis(self):
        self.assertEqual(self.tc.ndim, 9)
        self.assertEqual(self.tc.input_basis[0], "dist")
        self.assertEqual(self.tc.input_basis[-1], "fdot_astro_ratio")

    def test_forward_amplitude_and_fdot(self):
        from lisatools.globalfit.stock.erebor.transforms import gb_amp_from_dist

        f0_mHz, Mc, dist, r = 7.5, 0.3, 10.0, -2.0
        phys = self.tc.both_transforms(
            _dist_row(dist=dist, f0_mHz=f0_mHz, Mc=Mc, ratio=r))
        A_exp = gb_amp_from_dist(f0_mHz * 1e-3, Mc, dist)
        self.assertAlmostEqual(phys[0, 0], A_exp, delta=abs(A_exp) * 1e-12)
        self.assertAlmostEqual(
            phys[0, 2], get_fdot(f=f0_mHz * 1e-3, Mc=Mc) * (1.0 + r),
            delta=abs(get_fdot(f=f0_mHz * 1e-3, Mc=Mc)) * 1e-12)
        self.assertLess(phys[0, 2], 0.0)       # fdot<0 via r=-2
        self.assertEqual(phys[0, 3], 0.0)      # fddot
        # A propto 1/d: doubling distance halves the amplitude
        phys2 = self.tc.both_transforms(_dist_row(dist=20.0, Mc=Mc, f0_mHz=f0_mHz))
        phys1 = self.tc.both_transforms(_dist_row(dist=10.0, Mc=Mc, f0_mHz=f0_mHz))
        self.assertAlmostEqual(phys2[0, 0] * 2.0, phys1[0, 0],
                               delta=abs(phys1[0, 0]) * 1e-12)

    def test_inverse_round_trip_incl_fdot_negative(self):
        rows = np.vstack([
            _dist_row(dist=5.0, Mc=0.3, ratio=-3.0),   # fdot<0
            _dist_row(dist=15.0, Mc=0.2, ratio=0.0),   # fdot>0
        ])
        phys = self.tc.both_transforms(rows)
        back = self.tc.both_inverse_transforms(phys)
        self.assertFalse(np.any(np.isnan(back)))
        self.assertTrue(np.all(back[:, 0] > 0.0))  # positive distances
        # PHYSICAL A and fdot are preserved exactly for either sign (the
        # invariant the likelihood sees); the (dist, Mc) SPLIT need not be
        # recovered when fdot<0 (mirror Mc != truth Mc -> dist compensates).
        phys2 = self.tc.both_transforms(back)
        np.testing.assert_allclose(phys2[:, [0, 1, 2, 3]], phys[:, [0, 1, 2, 3]],
                                   rtol=1e-9)
        # fdot>0 in-box row DOES recover (dist, Mc) exactly
        self.assertAlmostEqual(back[1, 0], rows[1, 0], delta=rows[1, 0] * 1e-9)
        self.assertAlmostEqual(back[1, 2], rows[1, 2], delta=rows[1, 2] * 1e-9)

    def test_pickle(self):
        rows = _dist_row(ratio=-2.0)
        phys = self.tc.both_transforms(rows)
        clone = pickle.loads(pickle.dumps(copy.deepcopy(self.tc)))
        self.assertTrue(np.array_equal(clone.both_transforms(rows), phys))

    def test_requires_fdot_astro(self):
        with self.assertRaises(ValueError):
            make_gb_transform_container(use_chirp_mass=True, use_distance=True)

    def test_seeding_reproduces_catalogue_amplitude(self):
        from lisatools.globalfit.recipe import gb_fdot_rows_to_run_basis

        # FDOT-basis rows: [lnA, f0_mHz, fdot, phi0, cos_i, psi, alpha, sin_d]
        rows = np.array([
            [np.log(1e-22), 7.5803, 3.0e-16, 1.0, 0.5, 1.0, 2.0, 0.1],
            [np.log(5e-23), 7.56749, -9.9e-16, 0.3, -0.2, 0.5, 4.9, -0.06],
        ])
        out = gb_fdot_rows_to_run_basis(
            rows, use_chirp_mass=True, use_fdot_astro=True, use_distance=True,
            m_chirp_lims=MC_LIMS)
        self.assertEqual(out.shape, (2, 9))
        self.assertTrue(np.all(out[:, 0] > 0.0))   # positive distances
        phys = self.tc.both_transforms(out)
        # forward reproduces the catalogue amplitude (and fdot, incl fdot<0)
        np.testing.assert_allclose(phys[:, 0], np.exp(rows[:, 0]), rtol=1e-9)
        np.testing.assert_allclose(phys[:, 2], rows[:, 2], rtol=1e-9)

    def test_birth_container_distance(self):
        from lisatools.sampling.fstat_proposal import make_gb_rj_birth_container

        rng = np.random.default_rng(2)

        class Intr4:
            ndim = 4
            use_cupy = False
            return_gpu = False

            def rvs(self, size, **kw):
                shape = (size,) if isinstance(size, int) else tuple(size)
                n = int(np.prod(shape))
                out = np.column_stack([
                    rng.uniform(7.4, 7.6, n), rng.uniform(0.1, 0.9, n),
                    rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n)])
                return out.reshape((*shape, 4))

            def logpdf(self, x, **kw):
                return np.zeros(x.shape[0])

        c = make_gb_rj_birth_container(
            Intr4(), [7e-26, 1e-19], use_cupy=False, fdot_astro_ratio_max=5.0,
            dist_lims=list(DIST_LIMS))
        b = np.asarray(c.rvs(size=50))
        self.assertEqual(b.shape, (50, 9))
        self.assertTrue(np.all((b[:, 0] >= DIST_LIMS[0]) & (b[:, 0] <= DIST_LIMS[1])))


if __name__ == "__main__":
    unittest.main()
