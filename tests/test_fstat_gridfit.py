"""Unit tests for the F-stat grid fit library + the in-move fit decision.

CPU-only, no sampler and no kernel: ``call_fstat`` is a fake returning
analytic ``(N, M_upper)`` rows, and the decision state machine is exercised
on a bare object rather than a built move.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from lisatools.sampling import fstat_gridfit as G


def _fake_call_fstat(counter=None, raise_after=None):
    """``params -> (N (n,4), M_upper (n,10))`` with an f0-dependent bump.

    ``M_upper`` is the upper triangle of the identity, so
    ``compute_fstat`` reduces to ``0.5 * sum(N**2)`` and F is a clean
    analytic function of f0 -- enough to exercise peak selection.
    """
    state = {"rows": 0}

    def call(params):
        p = np.asarray(params.get() if hasattr(params, "get") else params)
        n = p.shape[0]
        if raise_after is not None and state["rows"] + n > raise_after:
            raise RuntimeError("simulated death mid-sweep")
        state["rows"] += n
        if counter is not None:
            counter["calls"] += 1
            counter["rows"] += n
        f0_mHz = p[:, 1] * 1e3
        # three well-separated Gaussian bumps
        amp = np.zeros(n)
        for c, a in ((6.30, 40.0), (6.60, 25.0), (6.90, 60.0)):
            amp += a * np.exp(-0.5 * ((f0_mHz - c) / 2e-3) ** 2)
        N = np.zeros((n, 4))
        N[:, 0] = np.sqrt(2.0 * np.maximum(amp, 0.0))
        M = np.zeros((n, 10))
        M[:, 0] = M[:, 4] = M[:, 7] = M[:, 9] = 1.0  # identity upper-triangle
        return N, M

    return call


BAND_EDGES = np.linspace(6.2e-3, 7.0e-3, 6)   # 5 sub-bands, 3 interior
F0_LIMS = (float(BAND_EDGES[1]), float(BAND_EDGES[-2]))


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.environ["FSTAT_CKPT_SECS"] = "0"      # save every chunk
        os.environ["FSTAT_BATCH"] = "64"

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)
        os.environ.pop("FSTAT_CKPT_SECS", None)
        os.environ.pop("FSTAT_BATCH", None)

    def _params(self, n=256):
        p = np.zeros((n, 9))
        p[:, 0] = 1e-22
        p[:, 1] = np.linspace(6.25e-3, 6.95e-3, n)
        return p

    def test_sweep_resume_is_bit_identical(self):
        """A death mid-sweep resumes at the cursor and matches an
        uninterrupted run exactly."""
        p = self._params()
        ckpt = os.path.join(self.d, "sweep")

        ref = G.chunked_fstat_sweep(_fake_call_fstat(), p, xp=np,
                                    label=":ref")

        with self.assertRaises(RuntimeError):
            G.chunked_fstat_sweep(_fake_call_fstat(raise_after=100), p, xp=np,
                                  label=":r", ckpt=ckpt)
        self.assertTrue(os.path.exists(ckpt + ".progress.npz"))

        c = {"calls": 0, "rows": 0}
        got = G.chunked_fstat_sweep(_fake_call_fstat(counter=c), p, xp=np,
                                    label=":r", ckpt=ckpt)
        np.testing.assert_array_equal(ref, got)
        self.assertLess(c["rows"], p.shape[0],
                        "resume recomputed every row")

    def test_changed_fingerprint_restarts_clean(self):
        p = self._params()
        ckpt = os.path.join(self.d, "sweep")
        with self.assertRaises(RuntimeError):
            G.chunked_fstat_sweep(_fake_call_fstat(raise_after=100), p, xp=np,
                                  label=":r", ckpt=ckpt,
                                  fingerprint_extra="|epoch=0")
        c = {"calls": 0, "rows": 0}
        G.chunked_fstat_sweep(_fake_call_fstat(counter=c), p, xp=np,
                              label=":r", ckpt=ckpt,
                              fingerprint_extra="|epoch=1")
        self.assertEqual(c["rows"], p.shape[0],
                         "a new epoch must not resume the old epoch's rows")


class PeakSelectionTest(unittest.TestCase):
    def test_floor_interior_and_cap(self):
        f0 = np.linspace(6.2, 7.0, 2001)          # mHz
        F = np.zeros_like(f0)
        for c, a in ((6.30, 40.0), (6.60, 25.0), (6.90, 60.0)):
            F += a * np.exp(-0.5 * ((f0 - c) / 2e-3) ** 2)
        spacing = float(f0[1] - f0[0])

        os.environ["FSTAT_PEAK_MIN_F"] = "10"
        os.environ["FSTAT_PEAKS_PER_BAND"] = "5"
        try:
            peaks = G.select_comb_peaks(f0, F, BAND_EDGES, spacing, np)
        finally:
            os.environ.pop("FSTAT_PEAK_MIN_F", None)
            os.environ.pop("FSTAT_PEAKS_PER_BAND", None)

        self.assertGreater(len(peaks), 0)
        self.assertTrue(np.all(np.diff(peaks[:, 1]) <= 0),
                        "peaks must come back sorted by F descending")
        nb = len(BAND_EDGES) - 1
        bands = peaks[:, 3].astype(int)
        self.assertTrue(np.all(bands >= 1) and np.all(bands <= nb - 2),
                        "only interior sub-bands may be kept")
        counts = np.bincount(bands, minlength=nb)
        self.assertTrue(np.all(counts <= 5), "per-band cap not enforced")

    def test_floor_rejects_everything(self):
        f0 = np.linspace(6.2, 7.0, 501)
        F = np.full_like(f0, 1.0)
        os.environ["FSTAT_PEAK_MIN_F"] = "1e6"
        try:
            peaks = G.select_comb_peaks(f0, F, BAND_EDGES, float(f0[1] - f0[0]),
                                        np)
        finally:
            os.environ.pop("FSTAT_PEAK_MIN_F", None)
        self.assertEqual(len(peaks), 0)


class GridFitEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = {
            "FSTAT_BATCH": "512", "FSTAT_CKPT_SECS": "0",
            "FSTAT_F0_SPACING_MHZ": "0.01", "FSTAT_COMB_NSKY": "2",
            "FSTAT_PEAK_MIN_F": "10", "FSTAT_PEAKS_PER_BAND": "2",
            "FSTAT_N_MC": "2", "FSTAT_N_ALPHA": "2", "FSTAT_N_SINDELTA": "2",
            "FSTAT_PEAK_HALF_MHZ": "0.02",
        }
        self._old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_fit_then_rerun_short_circuits(self):
        c1 = {"calls": 0, "rows": 0}
        stacked, n_peaks = G.run_fstat_grid_fit(
            _fake_call_fstat(counter=c1), xp=np, Tobs=7.776e6,
            band_edges_hz=BAND_EDGES, f0_lims_hz=F0_LIMS,
            mc_lims=[0.01, 1.0], cache_dir=self.d, fingerprint_extra="|epoch=0")
        self.assertIsNotNone(stacked)
        self.assertGreater(n_peaks, 0)
        self.assertGreater(c1["rows"], 0)
        base = os.path.join(self.d, G.GRID_BASENAME)
        self.assertTrue(os.path.exists(base.replace(".npz", "_comb.npz")))
        self.assertTrue(os.path.exists(
            base.replace(".npz", "_peaks_stacked.npz")))

        c2 = {"calls": 0, "rows": 0}
        stacked2, n2 = G.run_fstat_grid_fit(
            _fake_call_fstat(counter=c2), xp=np, Tobs=7.776e6,
            band_edges_hz=BAND_EDGES, f0_lims_hz=F0_LIMS,
            mc_lims=[0.01, 1.0], cache_dir=self.d, fingerprint_extra="|epoch=0")
        self.assertIsNotNone(stacked2)
        self.assertEqual(n2, n_peaks)
        self.assertEqual(c2["calls"], 0,
                         "a complete fit must not re-enter the kernel")

    def test_birth_container_logpdf_is_finite_across_the_box(self):
        G.run_fstat_grid_fit(
            _fake_call_fstat(), xp=np, Tobs=7.776e6, band_edges_hz=BAND_EDGES,
            f0_lims_hz=F0_LIMS, mc_lims=[0.01, 1.0], cache_dir=self.d)
        cont = G.build_gb_birth_distribution(
            cache_dir=self.d, mc_lims=[0.01, 1.0], A_lims=[1e-24, 1e-21],
            use_cupy=False, floor_eps=0.1, comb_weight=0.0)
        self.assertIsNotNone(cont)
        draws = cont.rvs(size=(512,))
        lp = np.asarray(cont.logpdf(draws))
        self.assertTrue(np.all(np.isfinite(lp)),
                        "UniformFloorMixture must keep logpdf finite on its "
                        "own draws (BandSorter evaluates it for factors)")


class FitDecisionTest(unittest.TestCase):
    """The setup() state machine, exercised without building a move."""

    class _Stub:
        _fstat_root_dir = None
        rj_proposal_distribution = None
        fstat_refit_every = 0
        num_proposals = 0
        _fstat_epoch = None
        _fstat_last_fit_hit = -1

        # bind the real implementations
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as _M,
        )
        _fstat_fit_decision = _M._fstat_fit_decision
        _latest_epoch = _M._latest_epoch
        _epoch_dir = _M._epoch_dir
        _epoch_complete = staticmethod(_M._epoch_complete)

        @property
        def _fstat_root(self):
            return self._fstat_root_dir

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.s = self._Stub()
        self.s._fstat_root_dir = self.d

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_first_ever_fit(self):
        self.assertEqual(self.s._fstat_fit_decision(), ("fit", 0))

    def test_never_refit_by_default(self):
        self.s.rj_proposal_distribution = {"gb": object()}
        self.s._fstat_epoch = 0
        self.s.num_proposals = 500
        self.assertEqual(self.s._fstat_fit_decision(), ("skip", 0))

    def test_cadence_refit(self):
        self.s.rj_proposal_distribution = {"gb": object()}
        self.s._fstat_epoch = 0
        self.s.fstat_refit_every = 10
        self.s._fstat_last_fit_hit = 0
        self.s.num_proposals = 9
        self.assertEqual(self.s._fstat_fit_decision(), ("skip", 0))
        self.s.num_proposals = 10
        self.assertEqual(self.s._fstat_fit_decision(), ("fit", 1))

    def test_load_complete_epoch(self):
        d0 = os.path.join(self.d, "epoch_0000")
        os.makedirs(d0)
        with open(os.path.join(d0, "DONE.json"), "w") as f:
            json.dump({"epoch": 0, "n_peaks": 0}, f)
        self.assertEqual(self.s._fstat_fit_decision(), ("load", 0))

    def test_resume_incomplete_epoch(self):
        os.makedirs(os.path.join(self.d, "epoch_0000"))  # no DONE.json
        self.assertEqual(self.s._fstat_fit_decision(), ("fit", 0))

    def test_zero_peak_epoch_is_complete(self):
        """A fit that legitimately found no peaks must not refit forever."""
        d0 = os.path.join(self.d, "epoch_0000")
        os.makedirs(d0)
        with open(os.path.join(d0, "DONE.json"), "w") as f:
            json.dump({"epoch": 0, "n_peaks": 0}, f)
        self.assertTrue(self._Stub._epoch_complete(d0))


if __name__ == "__main__":
    unittest.main()
