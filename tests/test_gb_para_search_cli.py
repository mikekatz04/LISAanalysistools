"""The offline CLI end to end on a synthetic store (CPU, ~1 min).

``python -m lisatools.globalfit.warmstart.para_search --store <h5> --walker w
--bands a:b --out <dir>`` on a KB-sized store + residual snapshot written by
``tests/_para_search_world.write_synthetic_store``: the world WITH Gaussian
WDM noise (the CLI estimates the noise covariance from the residual itself),
two walkers (walker 1's residual already has the SNR-15 source subtracted).
Pinned: the store / snapshot / domain-settings plumbing, the empirical noise,
the report and finds files, the one-walker epoch npz, and the physics on the
loud sources (SNR 9 and 15 found on walker 0; the SNR-15 one NOT on walker 1).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _para_search_world import build_world, write_synthetic_store  # noqa: E402

from lisatools.globalfit.warmstart import para_search as ps  # noqa: E402

ENV = dict(GB_PARA_SEARCH_NTEMPS="4", GB_PARA_SEARCH_NWALKERS="16",
           GB_PARA_SEARCH_MIN_STEPS="30", GB_PARA_SEARCH_CONVERGE_WINDOW="20",
           GB_PARA_SEARCH_MAX_STEPS="120", GB_PARA_SEARCH_STAGE2_STEPS="20",
           GB_PARA_SEARCH_STAGE2_NTEMPS="2")


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="para_cli_")
        cls.w = w = build_world(noise=True, sighet=False)
        loud = [t for t in w["truth"] if t["snr"] == 15.0][0]
        res = np.stack([w["data"], w["data"] - loud["template"]])
        cls.store = write_synthetic_store(cls.d, w, res, leaves=[[], [loud["x"]]])
        cls.loud = loud

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, True)

    def _run(self, walker, extra=()):
        out = os.path.join(self.d, f"out_w{walker}")
        argv = ["--store", self.store, "--walker", str(walker), "--bands", "4:7",
                "--out", out, "--orbit-dt", "1000", "--t-ref", str(float(self.w["wdm"].t0)),
                "--nt-sub", "64", "--n-pad", "8", "--n-sparse", "128", "--n-cp-sig", "0",
                "--n-cp-orbit", "0", "--sighet-v4-knots", "64", "--sighet-v3-nodes", "32",
                "--dist-max", "1e6", *extra]
        with mock.patch.dict(os.environ, ENV, clear=False):
            return ps.main(argv), out

    def test_walker0_finds_the_loud_sources_and_writes_its_products(self):
        f, out = self._run(0, ["--merge-epoch"])
        df = self.w["df"]
        for snr in (9.0, 15.0):
            t = [t for t in self.w["truth"] if t["snr"] == snr][0]
            m = np.abs(f["f0_hz"] - t["f0_hz"]) / df <= 1.0
            self.assertEqual(int(m.sum()), 1, f"SNR {snr}")
            self.assertGreater(float(f["opt_snr"][m][0]), 0.6 * snr)
        self.assertEqual(list(f["bands"]), [4, 5, 6])
        txt = open(os.path.join(out, "report_w0.txt")).read()
        self.assertIn("find  band", txt)
        self.assertIn("already held by the walker's last cold row: 0", txt)
        recs = ps.load_finds_npz(os.path.join(out, "finds_w0.npz"))
        self.assertEqual(int(recs[0]["n_finds"]), int(f["n_finds"]))
        from lisatools.globalfit.warmstart.proposal import WarmStartComponents

        c = WarmStartComponents.from_npz(os.path.join(out, "epoch", ps.EPOCH_NPZ))
        self.assertGreaterEqual(c.n_components, int(f["n_finds"]))

    def test_walker1_has_the_loud_source_subtracted(self):
        f, out = self._run(1)
        m = np.abs(f["f0_hz"] - self.loud["f0_hz"]) / self.w["df"] <= 2.0
        self.assertEqual(int(m.sum()), 0)
        t = [t for t in self.w["truth"] if t["snr"] == 9.0][0]
        self.assertEqual(int((np.abs(f["f0_hz"] - t["f0_hz"]) / self.w["df"] <= 1.0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
