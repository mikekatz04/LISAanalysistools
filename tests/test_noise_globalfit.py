"""Tests for the ``noise_only`` / ``noise_sgwb`` stock global-fit variants.

The registration/config checks are cheap. ``test_noise_only_recovers_injection``
is a CPU integration test: it builds the synthetic-noise ``noise_only`` fit and
verifies that the likelihood at the injected noise parameters beats a
prior-draw start — i.e. the data really are drawn from the injected composite
covariance and the composite PSD+foreground sensitivity reads into the analysis
container array framework like any other.
"""

import os
import unittest

import numpy as np

from lisatools.globalfit.stock import erebor


class NoiseVariantRegistrationTest(unittest.TestCase):
    def test_registered(self):
        names = [name for name, _ in erebor.get_stock_options()]
        self.assertIn("noise_only", names)
        self.assertIn("noise_sgwb", names)

    def test_module_defaults(self):
        self.assertIsInstance(erebor.noise_only, erebor.NoiseOnlyGlobalFit)
        self.assertIsInstance(erebor.noise_sgwb, erebor.NoiseSGWBGlobalFit)

    def test_default_branches(self):
        self.assertEqual(set(erebor.noise_only.default_branches()), {"psd", "galfor"})
        self.assertEqual(
            set(erebor.noise_sgwb.default_branches()), {"psd", "galfor", "sgwb"}
        )


class NoiseOnlyIntegrationTest(unittest.TestCase):
    def test_noise_only_recovers_injection(self):
        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        from mpi4py import MPI
        from eryn.state import BranchSupplemental

        from lisatools.globalfit.run import GlobalFit

        # The recovery premise requires data actually drawn from the injected
        # covariance — pin synthetic mode explicitly. Under the default
        # data_mode="mojito" (when the local mojito cache exists) the data are
        # the real NOISE brick and psd_injection is only the brick-FITTED
        # 2-parameter approximation, which a prior draw can legitimately beat.
        fit = erebor.noise_only(nwalkers=4, ntemps=2, data_mode="synthetic")
        curr = fit.build()
        psd_inj = list(curr.general_info.psd_injection)
        galfor_inj = list(curr.general_info.galfor_injection)

        gf = GlobalFit(curr, MPI.COMM_WORLD)
        priors = {}
        for name in curr.branch_names:
            priors.update(curr.source_info[name].priors)
        state = gf.load_info(priors)
        nt, nw = gf.ntemps, gf.nwalkers
        state.supplemental = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(nw), (nt, 1))},
            base_shape=(nt, nw),
            copy=True,
        )
        acs = gf.setup_acs(state)

        backend = curr.general_info.sensitivity_backend
        inj_sens = backend("injection", psd_inj, galfor_params=galfor_inj)
        original = acs[0].sens_mat
        acs[0].sens_mat = inj_sens
        acs.reset_linear_psd_arr()

        # goal #2: the composite PSD+foreground matrix populates the array
        # framework's contiguous inverse-PSD buffer, finite everywhere.
        self.assertIsNotNone(acs.linear_psd_arr)
        self.assertTrue(np.all(np.isfinite(np.asarray(acs.linear_psd_arr))))

        logl_injection = float(np.asarray(acs.likelihood())[0])
        acs[0].sens_mat = original
        acs.reset_linear_psd_arr()
        logl_start = np.asarray(acs.likelihood())

        self.assertTrue(np.isfinite(logl_injection))
        # goal #1: data drawn from the injection => injection is the best fit.
        self.assertGreater(logl_injection, float(np.max(logl_start)))


class PSDLogSamplingTest(unittest.TestCase):
    """``psd.log_sampling``: the ``ln(Soms_d), ln(Sa_a)`` sampling basis."""

    def test_branch_prep(self):
        """Prior, transform, and injection all land in the ln basis."""
        from lisatools.globalfit.stock.erebor.noise import (
            PSD_PRIOR_RANGE,
            prepare_psd_branch,
        )
        from lisatools.globalfit.stock.erebor.variants.noise import NoisePSDSettings

        linear_injection = [15e-12, 3e-15]
        psd = prepare_psd_branch(
            NoisePSDSettings(log_sampling=True), linear_injection
        )

        np.testing.assert_allclose(np.exp(psd.injection), linear_injection, rtol=1e-14)
        # the ln prior covers exactly the linear prior's physical support
        draws = psd.priors["psd"].rvs(size=(500,))
        levels = psd.transform.both_transforms(draws.copy())
        for i, (lo, hi) in enumerate(PSD_PRIOR_RANGE):
            self.assertGreaterEqual(levels[:, i].min(), lo)
            self.assertLessEqual(levels[:, i].max(), hi)
        # ...and the injection is inside it (the fit can reach the truth)
        self.assertTrue(np.isfinite(psd.priors["psd"].logpdf(psd.injection[None, :])[0]))

        # the sampled basis is what the chain stores, so it must round-trip
        np.testing.assert_allclose(
            psd.transform.both_inverse_transforms(levels), draws, rtol=1e-14
        )

    def test_linear_default_unchanged(self):
        """Without the knob the branches stay linear and transform-free."""
        from lisatools.globalfit.stock.erebor.noise import (
            GalForSettings,
            prepare_galfor_branch,
            prepare_psd_branch,
        )
        from lisatools.globalfit.stock.erebor.variants.noise import NoisePSDSettings

        psd = prepare_psd_branch(NoisePSDSettings(), [15e-12, 3e-15])
        self.assertIsNone(psd.transform)
        np.testing.assert_allclose(psd.injection, [15e-12, 3e-15], rtol=1e-14)
        self.assertIsNone(prepare_galfor_branch(GalForSettings()).transform)


class GalForLogSamplingTest(unittest.TestCase):
    """``galfor.log_sampling``: log10 for amp/fk/f_1/f_2, alpha left linear."""

    def test_branch_prep(self):
        from lisatools.globalfit.stock.erebor.noise import (
            GALFOR_BASIS,
            GALFOR_LOG_PARAMS,
            GALFOR_PRIOR_RANGE,
            GalForSettings,
            prepare_galfor_branch,
        )

        galfor = prepare_galfor_branch(GalForSettings(log_sampling=True))
        draws = galfor.priors["galfor"].rvs(size=(500,))
        physical = galfor.transform.both_transforms(draws.copy())

        for i, name in enumerate(GALFOR_BASIS):
            lo, hi = GALFOR_PRIOR_RANGE[i]
            # every parameter lands inside its physical prior range...
            self.assertGreaterEqual(physical[:, i].min(), lo)
            self.assertLessEqual(physical[:, i].max(), hi)
            if name in GALFOR_LOG_PARAMS:
                # ...via log10 for the four scale parameters (NOT ln -- the
                # psd branch is the one in ln)
                np.testing.assert_allclose(
                    physical[:, i], 10.0 ** draws[:, i], rtol=1e-14
                )
            else:
                # ...and untouched for alpha, which is sampled linearly
                np.testing.assert_array_equal(physical[:, i], draws[:, i])

        # atol as well as rtol: 10**x then log10 loses ~1 ulp more than
        # exp/log does, and a draw landing near f_1 = 1 has log10 ~ 0, where a
        # pure relative tolerance is meaningless. Absolute error stays < 1e-16.
        np.testing.assert_allclose(
            galfor.transform.both_inverse_transforms(physical),
            draws,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_move_transforms_galfor_on_the_container_route(self):
        """``PSDMove`` must transform galfor before the sensitivity backend.

        The backend only knows about the psd transform, so the non-kernel
        (WDM) route would otherwise hand the foreground model ``ln amp`` as an
        amplitude.
        """
        from lisatools.globalfit.moves.psdmove import PSDMove
        from lisatools.globalfit.stock.erebor.noise import (
            GalForSettings,
            prepare_galfor_branch,
        )

        galfor = prepare_galfor_branch(GalForSettings(log_sampling=True))
        seen = {}

        class _RecordingBackend:
            def __call__(self, name, psd_params, galfor_params=None, **kwargs):
                seen["psd"] = psd_params
                seen["galfor"] = galfor_params
                return "sens"

        move = PSDMove.__new__(PSDMove)  # no __init__: this is a unit check
        move.sensitivity_backend = _RecordingBackend()
        move.psd_transform_fn = None
        move.galfor_transform_fn = galfor.transform
        move.sgwb_transform_fn = None

        galfor_row = np.array([-43.486, -2.679, 1.183, -2.941, -3.471])  # log10
        psd_row = np.array([15e-12, 3e-15])
        self.assertEqual(move._build_sensitivity_for_walker(0, psd_row, galfor_row), "sens")

        np.testing.assert_allclose(seen["psd"], psd_row, rtol=1e-14)
        expected = galfor_row.copy()
        expected[[0, 1, 3, 4]] = 10.0 ** expected[[0, 1, 3, 4]]
        np.testing.assert_allclose(seen["galfor"], expected, rtol=1e-14)


class LogSamplingEngineTest(unittest.TestCase):
    """Both log-sampled noise branches, end to end through the engine."""

    def test_engine_applies_the_transform_exactly_once(self):
        """A log-basis state builds the SAME sensitivity as the linear params.

        Covers both log-sampled noise branches through ``run.py::setup_acs``,
        which transforms the coords itself; the sensitivity backend would
        transform the psd row again if it were also handed ``transform_fn``.
        Double-exponentiating ``ln S ~ -25`` gives ``exp(exp(-25)) ~ 1``, so
        the likelihoods separate hugely — this pins them together.
        """
        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        from mpi4py import MPI
        from eryn.state import BranchSupplemental

        from lisatools.globalfit.run import GlobalFit

        fit = erebor.noise_only(nwalkers=4, ntemps=2, data_mode="synthetic")
        fit.psd.log_sampling = True
        fit.galfor.log_sampling = True
        curr = fit.build()
        psd_inj = list(curr.general_info.psd_injection)  # LINEAR, as always
        galfor_inj = list(curr.general_info.galfor_injection)
        self.assertIsNotNone(curr.source_info["psd"].transform)
        self.assertIsNotNone(curr.source_info["galfor"].transform)

        gf = GlobalFit(curr, MPI.COMM_WORLD)
        priors = {}
        for name in curr.branch_names:
            priors.update(curr.source_info[name].priors)
        state = gf.load_info(priors)
        nt, nw = gf.ntemps, gf.nwalkers
        state.supplemental = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(nw), (nt, 1))},
            base_shape=(nt, nw),
            copy=True,
        )
        # pin every walker at the injection, in each branch's sampling basis
        galfor_log = np.asarray(galfor_inj, dtype=float)
        # galfor is log10, psd is ln -- the two branches differ (2026-08)
        galfor_log[[0, 1, 3, 4]] = np.log10(galfor_log[[0, 1, 3, 4]])  # alpha stays linear
        state.branches_coords["psd"][:] = np.log(psd_inj)
        state.branches_coords["galfor"][:] = galfor_log

        acs = gf.setup_acs(state)
        logl_from_state = float(np.asarray(acs.likelihood())[0])

        backend = curr.general_info.sensitivity_backend
        acs[0].sens_mat = backend("injection", psd_inj, galfor_params=galfor_inj)
        acs.reset_linear_psd_arr()
        logl_direct = float(np.asarray(acs.likelihood())[0])

        self.assertTrue(np.isfinite(logl_from_state))
        self.assertAlmostEqual(
            logl_from_state, logl_direct, delta=1e-10 * abs(logl_direct)
        )


if __name__ == "__main__":
    unittest.main()
