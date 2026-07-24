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


if __name__ == "__main__":
    unittest.main()
