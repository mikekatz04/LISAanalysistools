"""Integration smoke: erebor.blank + branch-info append + fn moves + fit.sample().

This builds a (small, synthetic) fit, so it is gated behind RUN_GF_SMOKE=1::

    RUN_GF_SMOKE=1 python -m unittest tests.test_globalfit_sample

It exercises the full simple-entrance chain end to end: blank variant ->
add_branch(branch info + plain fn move) -> generator run mode -> in-loop
mutation -> mid-loop add_move -> HDF persistence + resumed-state consistency.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from lisatools.globalfit.stock import erebor

RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")


def counting_move(model, state):
    counting_move.calls += 1
    return state, None


counting_move.calls = 0


def late_move(model, state):
    late_move.calls += 1
    return state, None


late_move.calls = 0


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the build+sample smoke")
class BlankSampleSmokeTest(unittest.TestCase):
    def setUp(self):
        from eryn.prior import uniform_dist

        counting_move.calls = 0
        late_move.calls = 0
        self.tmpdir = tempfile.mkdtemp(prefix="gf_blank_smoke_")
        self.fit = erebor.blank(
            nwalkers=4,
            ntemps=2,
            file_store_dir=self.tmpdir,
            make_diagnostic_plots=False,
        )
        self.fit.add_branch(
            "line",
            ndim=2,
            priors={0: uniform_dist(0.0, 1.0), 1: uniform_dist(0.0, 1.0)},
            moves=[counting_move],
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sample_generator_end_to_end(self):
        n_iters = 4
        lls = []
        for i, (model, state) in enumerate(self.fit.sample(iterations=n_iters)):
            # the residual the user is meant to see/adjust
            self.assertTrue(hasattr(model, "analysis_container_arr"))
            lls.append(state.log_like.copy())
            if i == 1:
                # (b) in-loop residual mutation flows into the next iteration's
                # bookkeeping via the auto log-like resync
                model.analysis_container_arr.likelihood(complex=False)
                # (c) mid-loop add starts firing on the NEXT iteration
                self.fit.add_move(late_move, branch="line")
        # zero-noise default: the data are all zeros, so the (source-only)
        # null log-like is exactly 0 and stays there (the only move returns
        # the state unchanged)
        self.assertTrue(np.allclose(lls[0], 0.0))
        # (a) the branch-info move fired every iteration
        self.assertGreaterEqual(counting_move.calls, n_iters)
        # (c) the live-added move fired on the remaining iterations
        self.assertGreaterEqual(late_move.calls, n_iters - 2)
        self.assertEqual(len(lls), n_iters)
        # runner detached again
        self.assertIsNone(self.fit._runner)

        # (e) HDF persistence: the run saved its steps + recipe bookkeeping
        from lisatools.globalfit.hdfbackend import GFHDFBackend

        reader = GFHDFBackend(self.fit.general_info.main_file_path)
        self.assertGreaterEqual(reader.iteration, 1)
        # (f) valid last sample after generator close
        last = reader.get_last_sample()
        self.assertIn("line", last.branches)

    def test_bare_canvas_runs(self):
        # nothing added at all: the hidden idle branch/move keeps the
        # generator ticking, and store=False leaves storage to the user
        bare = erebor.blank(
            nwalkers=4, ntemps=2,
            file_store_dir=self.tmpdir + "/bare/",
            make_diagnostic_plots=False,
        )
        my_records = []
        for model, state in bare.sample(iterations=2, store=False, progress=False):
            my_records.append(state.log_like[0].copy())
        self.assertEqual(len(my_records), 2)
        self.assertIn("idle", bare.branches)

    def test_post_build_branch_append(self):
        from eryn.prior import uniform_dist

        # (d) a post-build append lands in the built products
        self.fit.build()
        self.fit.add_branch(
            "extra",
            ndim=1,
            priors={0: uniform_dist(0.0, 1.0)},
            moves=[late_move],
        )
        self.assertIn("extra", self.fit.current_info.source_info)
        self.assertIn("extra", self.fit.engine_info.branch_names)
        for model, state in self.fit.sample(iterations=2):
            self.assertIn("extra", state.branches)
        self.assertGreaterEqual(late_move.calls, 2)


if __name__ == "__main__":
    unittest.main()
