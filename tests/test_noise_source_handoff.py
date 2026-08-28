"""Coarse-sidecar lifecycle: noise (search_approx) -> source-move handoff.

Integration test for plan-2 T6 Step 3, on the lightest honest fixture: the
noise_only_lite build (synthetic data, CPU) plus one SIMPLE-API source
branch whose FunctionMove stands in for a source move. The added branch
makes the run all-source-shaped, so ``run.py`` builds the coarse sidecar
runtime (fine backend canonical, per-walker statistics) and the
``GFCombineMove`` guard runs ``ensure_fine_noise_covariance_current``
before every source sub-move.

What this proves end-to-end, per iteration: the sidecar engages; the PSD
move scores on the coarse surrogate (search_approx), refreshes the
per-walker statistics from the CURRENT residuals, and publishes the FULL
fine covariance + packed buffer before returning; the source move then
observes only fine state (the guard would raise otherwise, and the in-move
assertions check it again explicitly).

The corruption NEGATIVES live at the unit level
(``FineHandoffPreconditionTest`` / the combine-dispatch tests): an in-loop
corruption cannot trip the guard here because the next PSD publication
heals it — which is itself the designed behavior.

NOTE the two-layer gate this fixture documents: the noise-only VARIANT
refuses coarse modes (its validator — noise-only keeps the historical CPU
backend-replacement path), while the RUN-level gate keys on the branch
set. The test therefore sets ``coarse_gpu_mode`` on the built
``general_info`` after the variant validation has run, exactly as an
all-source variant does through its knobs.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from eryn.prior import ProbDistContainer, uniform_dist


def _pin_threads():
    for var in (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


_SOURCE_MOVE_LOG = []


def _source_move_stand_in(model, state):
    """A 'source move': observes the fine state and does nothing else."""
    acs = model.analysis_container_arr
    record = {
        "n_walkers": len(acs),
        "bases": [acs[w].sens_mat.basis_settings for w in range(len(acs))],
        "shapes": [tuple(acs[w].sens_mat.data_shape) for w in range(len(acs))],
    }
    _SOURCE_MOVE_LOG.append(record)
    return state, np.zeros(np.asarray(state.log_like).shape, dtype=bool)


class CoarseSidecarLifecycleTest(unittest.TestCase):
    """search_approx noise -> source handoff on the real machinery (slow-ish)."""

    def test_noise_to_source_fine_handoff(self):
        _pin_threads()
        _SOURCE_MOVE_LOG.clear()
        np.random.seed(20260828)
        store = tempfile.mkdtemp(prefix="coarse_lifecycle_")
        self.addCleanup(shutil.rmtree, store, True)

        from lisatools.globalfit.stock import erebor

        fit = erebor.noise_only_lite(
            data_mode="synthetic",
            coarse_Q=8,
            file_store_dir=store,
            base_file_name="lifecycle",
        )
        fit.add_branch(
            "srcdummy",
            ndim=1,
            nleaves_max=1,
            nleaves_min=1,
            priors={"srcdummy": ProbDistContainer({0: uniform_dist(-1.0, 1.0)})},
            moves=[_source_move_stand_in],
        )
        curr = fit.build()
        # Run-level opt-in (see module docstring): the branch-keyed gate in
        # run.py builds the sidecar because 'srcdummy' is a source branch.
        curr.general_info.coarse_gpu_mode = "search_approx"

        from lisatools.globalfit.run import GlobalFit

        gf = GlobalFit(curr)
        fine_settings = curr.general_info.domain_settings
        iterations = 0
        for model, state in gf.sample(iterations=2, store=False, progress=False):
            iterations += 1
            runtime = getattr(curr.general_info, "coarse_wdm_runtime", None)
            self.assertIsNotNone(runtime, "sidecar runtime was not built")
            self.assertEqual(runtime.mode, "search_approx")
            self.assertTrue(
                runtime._P_store, "per-walker statistics never refreshed"
            )
            for w in range(len(gf.acs)):
                sens_mat = gf.acs[w].sens_mat
                self.assertEqual(sens_mat.basis_settings, fine_settings)
                self.assertEqual(
                    tuple(sens_mat.data_shape)[-2:],
                    tuple(fine_settings.basis_shape_active),
                )
            self.assertTrue(
                np.all(np.isfinite(np.asarray(state.log_like)[0])),
                "cold fine log-likes must be finite after publication",
            )

        self.assertEqual(iterations, 2)
        self.assertGreater(
            len(_SOURCE_MOVE_LOG), 0, "the source stand-in move never ran"
        )
        for record in _SOURCE_MOVE_LOG:
            for basis, shape in zip(record["bases"], record["shapes"]):
                self.assertEqual(basis, fine_settings)
                self.assertEqual(
                    shape[-2:], tuple(fine_settings.basis_shape_active)
                )


if __name__ == "__main__":
    unittest.main()
