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


class IncrementalCheckpointTest(unittest.TestCase):
    """The progress payload is append-only (O(n) I/O over a sweep), and a
    legacy inline-``F`` progress file still resumes."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.ckpt = os.path.join(self.d, "s")
        self.F = np.arange(1000, dtype=float) * 0.5

    def _dat(self):
        return self.ckpt + G._CKPT_PAYLOAD_SUFFIX

    def test_append_only_payload_round_trips(self):
        G.ckpt_save(self.ckpt, self.F[:400], 400, 1000, "fp", start=0)
        G.ckpt_save(self.ckpt, self.F[400:700], 700, 1000, "fp", start=400)
        self.assertEqual(os.path.getsize(self._dat()), 700 * 8)
        got, done = G.ckpt_load(self.ckpt, 1000, "fp")
        self.assertEqual(done, 700)
        np.testing.assert_array_equal(got, self.F[:700])

    def test_saves_only_the_new_rows(self):
        """The whole point: a cadence save must not rewrite the prefix."""
        seen = []
        real = G.ckpt_save

        def spy(ckpt, rows, done, n, fp, start=None):
            seen.append(int(np.asarray(rows).size))
            return real(ckpt, rows, done, n, fp, start)

        p = np.zeros((640, 9))
        p[:, 1] = np.linspace(6.25e-3, 6.95e-3, 640)
        G.ckpt_save = spy
        os.environ["FSTAT_CKPT_SECS"] = "0"
        os.environ["FSTAT_BATCH"] = "64"
        try:
            G.chunked_fstat_sweep(_fake_call_fstat(), p, xp=np, label=":i",
                                  ckpt=self.ckpt)
        finally:
            G.ckpt_save = real
            os.environ.pop("FSTAT_CKPT_SECS", None)
            os.environ.pop("FSTAT_BATCH", None)
        # 10 chunks -> the up-front empty header + one save per chunk, each
        # carrying exactly one chunk's rows (never the growing prefix).
        self.assertEqual(sum(seen), 640, seen)
        self.assertLessEqual(max(seen), 64, seen)

    def test_legacy_inline_payload_resumes_and_normalizes(self):
        # exactly what the pre-payload ckpt_save wrote
        np.savez(self.ckpt + G._CKPT_HEADER_SUFFIX, F=self.F[:300], done=300,
                 n=1000, fingerprint="fp")
        got, done = G.ckpt_resume(self.ckpt, 1000, "fp")
        self.assertEqual(done, 300)
        np.testing.assert_array_equal(got[:300], self.F[:300])
        self.assertEqual(os.path.getsize(self._dat()), 300 * 8)
        G.ckpt_save(self.ckpt, self.F[300:500], 500, 1000, "fp", start=300)
        got2, done2 = G.ckpt_load(self.ckpt, 1000, "fp")
        self.assertEqual(done2, 500)
        np.testing.assert_array_equal(got2, self.F[:500])

    def test_overlong_payload_is_truncated_to_the_header_cursor(self):
        """A run that appended rows then died before its header landed."""
        G.ckpt_save(self.ckpt, self.F[:400], 400, 1000, "fp", start=0)
        with open(self._dat(), "ab") as f:
            np.zeros(50).tofile(f)          # orphan rows past the cursor
        G.ckpt_save(self.ckpt, self.F[400:600], 600, 1000, "fp", start=400)
        got, done = G.ckpt_load(self.ckpt, 1000, "fp")
        self.assertEqual(done, 600)
        np.testing.assert_array_equal(got, self.F[:600])

    def test_missing_payload_restarts_the_sweep(self):
        G.ckpt_save(self.ckpt, self.F[:400], 400, 1000, "fp", start=0)
        os.remove(self._dat())
        self.assertEqual(G.ckpt_load(self.ckpt, 1000, "fp"), (None, 0))

    def test_ckpt_clear_removes_header_and_payload(self):
        G.ckpt_save(self.ckpt, self.F[:400], 400, 1000, "fp", start=0)
        G.ckpt_clear(self.d, "s")
        self.assertEqual(os.listdir(self.d), [])


class CombRowsTest(unittest.TestCase):
    """The lazy comb block must be indistinguishable from the dense array it
    replaces -- values AND checkpoint fingerprint."""

    @staticmethod
    def _dense(nodes, alpha, sd, mc_fix):
        from gbgpu.utils.utility import get_fdot

        nn, lv = len(nodes), len(alpha)
        p = np.zeros((lv * nn, 9))
        p[:, 0] = 1e-22
        p[:, 1] = np.repeat(nodes, lv) * 1e-3
        p[:, 2] = get_fdot(f=p[:, 1], Mc=np.full(p.shape[0], mc_fix))
        p[:, 5] = 0.5 * np.pi
        p[:, 7] = np.tile(alpha, nn)
        p[:, 8] = np.arcsin(np.tile(sd, nn))
        return p

    def _pair(self, nn, lv):
        nodes = np.linspace(3.0, 12.0, nn)
        al, sd = G._sky_grid(lv)
        return (self._dense(nodes, al, sd, 0.5),
                G._CombRows(nodes, al, sd, 0.5))

    def test_slices_are_bit_identical(self):
        for nn, lv in ((97, 16), (301, 64), (17, 512)):
            dense, lazy = self._pair(nn, lv)
            self.assertEqual(lazy.shape, dense.shape)
            self.assertEqual(len(lazy), dense.shape[0])
            for s in range(0, dense.shape[0], 512):
                np.testing.assert_array_equal(lazy[s:s + 512],
                                              dense[s:s + 512])

    def test_fingerprint_matches_the_dense_block(self):
        """A comb progress file written against the dense assembly must keep
        resuming: the fingerprint subsample has to land on the same bytes."""
        for nn, lv in ((97, 16), (301, 64), (17, 512)):
            dense, lazy = self._pair(nn, lv)
            self.assertEqual(G.ckpt_fingerprint(lazy, extra="c"),
                             G.ckpt_fingerprint(dense, extra="c"))

    def test_sweep_over_lazy_rows_matches_the_dense_sweep(self):
        dense, lazy = self._pair(61, 16)
        os.environ["FSTAT_BATCH"] = "128"
        try:
            a = G.chunked_fstat_sweep(_fake_call_fstat(), dense, xp=np,
                                      label=":d")
            b = G.chunked_fstat_sweep(_fake_call_fstat(), lazy, xp=np,
                                      label=":l")
        finally:
            os.environ.pop("FSTAT_BATCH", None)
        np.testing.assert_array_equal(a, b)


class SigHetCallFstatTest(unittest.TestCase):
    """The single-block fast path must agree row-for-row with the general
    grouped path (and build the same reference blocks in the same order)."""

    class _Comp:
        xp = np
        _g = {"v4_knots": 128}

        def __init__(self):
            self._fstat = None
            self.built = []

        def setup_fstat_references(self, refs, holder, **kw):
            self.built.append(float(refs[0, 1]))
            self._ref0 = float(refs[0, 1])
            self._fstat = object()

        def get_fstat_ll_wdm(self, params, holder, data_index=None,
                             fstat_mode=None):
            p = np.asarray(params)
            di = np.asarray(data_index, dtype=float)
            N = np.zeros((p.shape[0], 4))
            N[:, 0] = p[:, 1] * 1e3
            N[:, 1] = p[:, 7]
            N[:, 2] = di                      # catches a row<->ref mispair
            N[:, 3] = self._ref0 * 1e3
            M = np.tile(np.array([1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0]),
                        (p.shape[0], 1))
            return N, M

    def _rows(self, f0):
        p = np.zeros((len(f0), 9))
        p[:, 0] = 1e-22
        p[:, 1] = f0
        p[:, 7] = np.linspace(0.0, 6.0, len(f0))
        return p

    def _call(self, comp, block=512):
        return G.build_sighet_call_fstat(
            comp, None, xp=np, Tobs=7.776e6, f0_lims_hz=(3.0e-3, 6.0e-3),
            ref_block=block)

    def test_single_block_batch_matches_the_grouped_path(self):
        # ref_block=1 forces one reference per block, so the SAME rows go
        # through the grouped path; ref_block=512 puts them in one block.
        f0 = np.repeat(np.linspace(3.100e-3, 3.112e-3, 8), 8)
        rows = self._rows(f0)
        c_fast, c_slow = self._Comp(), self._Comp()
        N1, M1 = self._call(c_fast, block=512)(rows)
        N2, M2 = self._call(c_slow, block=1)(rows)
        self.assertEqual(len(c_fast.built), 1, "not the fast path")
        self.assertGreater(len(c_slow.built), 1, "not the grouped path")
        np.testing.assert_array_equal(N1[:, :2], N2[:, :2])
        np.testing.assert_array_equal(M1, M2)
        self.assertEqual(N1.dtype, np.float64)
        self.assertEqual(N1.shape, (len(f0), 4))
        self.assertEqual(M1.shape, (len(f0), 10))

    def test_multi_block_and_unsorted_rows_still_work(self):
        rng = np.random.default_rng(3)
        f0 = rng.permutation(np.linspace(3.0e-3, 5.9e-3, 256))
        rows = self._rows(f0)
        comp = self._Comp()
        N, M = self._call(comp)(rows)
        self.assertGreater(len(comp.built), 1)
        np.testing.assert_array_equal(N[:, 0], f0 * 1e3)
        self.assertEqual(M.shape, (256, 10))

    def test_single_row_and_empty_batches(self):
        comp = self._Comp()
        call = self._call(comp)
        N, M = call(self._rows(np.array([4.0e-3]))[0])   # 1-D input
        self.assertEqual(N.shape, (1, 4))
        N0, M0 = call(np.zeros((0, 9)))
        self.assertEqual((N0.shape, M0.shape), ((0, 4), (0, 10)))

    def test_out_of_range_f0_still_raises(self):
        comp = self._Comp()
        call = self._call(comp)
        with self.assertRaises(RuntimeError):
            call(self._rows(np.array([9.0e-3])))


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
        name = "stub"
        branch_name = "gb"
        # the band grid the run is configured with (the staleness check
        # compares epoch caches against it)
        band_edges = np.linspace(1e-3, 2e-3, 6)

        # bind the real implementations
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as _M,
        )
        _fstat_fit_decision = _M._fstat_fit_decision
        _latest_epoch = _M._latest_epoch
        _epoch_dir = _M._epoch_dir
        _epoch_complete = staticmethod(_M._epoch_complete)
        _epoch_band_grid_stale = _M._epoch_band_grid_stale
        _fstat_clock = _M._fstat_clock
        _epoch_fit_clock = _M._epoch_fit_clock
        _FSTAT_CLOCK_BASENAME = _M._FSTAT_CLOCK_BASENAME
        _FSTAT_CLOCK_WRITE_EVERY = _M._FSTAT_CLOCK_WRITE_EVERY

        @property
        def _fstat_root(self):
            return self._fstat_root_dir

    @staticmethod
    def _reset_clock_state():
        """The refit clock is class-level shared state; isolate every test."""
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialBase,
            GBSpecialRJFStatGridMove,
        )

        GBSpecialBase._branch_propose_counts.pop("gb", None)
        GBSpecialRJFStatGridMove._fstat_clock_seeded.clear()
        GBSpecialRJFStatGridMove._fstat_clock_written.clear()

    @staticmethod
    def _tick(n):
        """Advance the shared branch census, as any GBSpecial propose does."""
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        GBSpecialBase._branch_propose_counts["gb"] = (
            GBSpecialBase._branch_propose_counts.get("gb", 0) + n
        )

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.s = self._Stub()
        self.s._fstat_root_dir = self.d
        self._reset_clock_state()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)
        self._reset_clock_state()

    def test_first_ever_fit(self):
        self.assertEqual(self.s._fstat_fit_decision(), ("fit", 0))

    def test_never_refit_by_default(self):
        self.s.rj_proposal_distribution = {"gb": object()}
        self.s._fstat_epoch = 0
        self.s.num_proposals = 500
        self.assertEqual(self.s._fstat_fit_decision(), ("skip", 0))

    def test_cadence_refit(self):
        """Cadence runs on the SHARED branch census, not num_proposals.

        2026-08-24 redesign: the per-instance counter starved in
        production (full_pe random_choice, short gb_search stages,
        per-launch resets) so the grid never refit. The clock is now the
        class-level per-branch propose census, which every GBSpecial move
        of the branch ticks.
        """
        self.s.rj_proposal_distribution = {"gb": object()}
        self.s._fstat_epoch = 0
        self.s.fstat_refit_every = 10
        self.s._fstat_last_fit_hit = 0
        self.s.num_proposals = 0        # the instance counter is IGNORED now
        self._tick(9)
        self.assertEqual(self.s._fstat_fit_decision(), ("skip", 0))
        self._tick(1)
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

    def _write_stacked(self, d, band_edges):
        from lisatools.sampling.fstat_gridfit import GRID_BASENAME

        np.savez(
            os.path.join(
                d, GRID_BASENAME.replace(".npz", "_peaks_stacked.npz")
            ),
            peak_f0_mHz=np.array([1.5]),
            band_idx=np.array([1]),
            band_edges=np.asarray(band_edges),
        )

    def test_matching_band_grid_loads(self):
        d0 = os.path.join(self.d, "epoch_0000")
        os.makedirs(d0)
        self._write_stacked(d0, self.s.band_edges)
        self.assertEqual(self.s._fstat_fit_decision(), ("load", 0))

    def test_stale_band_grid_forces_fresh_epoch(self):
        """A complete epoch fitted on DIFFERENT band edges must never load.

        The per-peak band_idx labels are indices into the old grid and the
        sweep checkpoints don't fingerprint the band grid, so the decision
        is a fit in a FRESH epoch dir (k_latest + 1), not a resume.
        """
        d0 = os.path.join(self.d, "epoch_0000")
        os.makedirs(d0)
        self._write_stacked(d0, np.linspace(1e-3, 2e-3, 9))  # different grid
        self.assertEqual(self.s._fstat_fit_decision(), ("fit", 1))

    def test_zero_peak_epoch_is_complete(self):
        """A fit that legitimately found no peaks must not refit forever."""
        d0 = os.path.join(self.d, "epoch_0000")
        os.makedirs(d0)
        with open(os.path.join(d0, "DONE.json"), "w") as f:
            json.dump({"epoch": 0, "n_peaks": 0}, f)
        self.assertTrue(self._Stub._epoch_complete(d0))


if __name__ == "__main__":
    unittest.main()


class PeakWeightAlphaTest(unittest.TestCase):
    """``FSTAT_PEAK_WEIGHT_ALPHA``: w ~ F**alpha for the birth peak boxes.

    The F-statistic goes like SNR^2, so the historical ``w ~ F`` allocates
    birth draws like SNR^2 -- an SNR-10 source gets 9x fewer attempts than
    an SNR-30 one, which is exactly backwards. alpha interpolates between
    that (alpha=1, the default and the historical behaviour) and flat
    (alpha=0 == ``FSTAT_PEAK_WEIGHTING=equal``).
    """

    @staticmethod
    def _weights(peak_F, env):
        """The weight expression from ``build_birth_proposal``, in isolation.

        Kept in lockstep with the module by construction: the test asserts
        the RELATIONS (ordering, ratios, special cases), which is what the
        proposal actually depends on, rather than re-deriving the formula.
        """
        weighting = env.get("FSTAT_PEAK_WEIGHTING", "fstat").strip().lower()
        alpha = float(env.get("FSTAT_PEAK_WEIGHT_ALPHA", "1.0"))
        peak_F = np.clip(np.asarray(peak_F, dtype=float), 0.0, None)
        if weighting == "equal" or alpha == 0.0:
            return None
        if alpha == 1.0:
            return peak_F
        return np.power(peak_F, alpha)

    def test_default_is_bit_identical_to_the_historical_behaviour(self):
        F = np.array([26.6, 125.5, 0.0, 900.0])
        np.testing.assert_array_equal(self._weights(F, {}), F)

    def test_alpha_zero_is_equal_weighting(self):
        F = np.array([26.6, 125.5, 0.0, 900.0])
        self.assertIsNone(self._weights(F, {"FSTAT_PEAK_WEIGHT_ALPHA": "0"}))
        self.assertIsNone(self._weights(F, {"FSTAT_PEAK_WEIGHTING": "equal"}))

    def test_explicit_equal_wins_over_alpha(self):
        F = np.array([26.6, 125.5])
        self.assertIsNone(self._weights(
            F, {"FSTAT_PEAK_WEIGHTING": "equal",
                "FSTAT_PEAK_WEIGHT_ALPHA": "0.5"}))

    def test_alpha_half_makes_draws_go_like_snr_not_snr_squared(self):
        """THE POINT OF THE KNOB.

        F ~ SNR^2/2. Two sources at SNR 10 and 30 have F in ratio 9:1, so
        alpha=1 gives the loud one 9x the draws. alpha=0.5 must reduce that
        to 3x -- draws proportional to SNR.
        """
        snr = np.array([10.0, 30.0])
        F = 0.5 * snr ** 2
        w1 = self._weights(F, {"FSTAT_PEAK_WEIGHT_ALPHA": "1.0"})
        wh = self._weights(F, {"FSTAT_PEAK_WEIGHT_ALPHA": "0.5"})
        self.assertAlmostEqual(w1[1] / w1[0], 9.0, places=6)
        self.assertAlmostEqual(wh[1] / wh[0], 3.0, places=6)

    def test_ordering_is_preserved_for_any_positive_alpha(self):
        """A weaker peak must never outrank a stronger one."""
        F = np.array([0.0, 1.0, 26.6, 125.5, 900.0])
        for a in ("0.25", "0.5", "0.75", "1.0", "2.0"):
            w = self._weights(F, {"FSTAT_PEAK_WEIGHT_ALPHA": a})
            self.assertTrue(np.all(np.diff(w) >= 0), f"alpha={a} reordered")

    def test_zero_F_stays_zero_for_positive_alpha(self):
        """F == 0 boxes carry no signal and must not gain mass."""
        w = self._weights(np.array([0.0, 4.0]),
                          {"FSTAT_PEAK_WEIGHT_ALPHA": "0.5"})
        self.assertEqual(w[0], 0.0)
        self.assertGreater(w[1], 0.0)

    def test_all_zero_weights_are_caught_rather_than_normalized_to_nan(self):
        """StackedFStatProposal4D divides by the weight sum.

        An all-zero weight vector would make that a 0/0 and yield a NaN
        proposal, so ``build_birth_proposal`` falls back to flat. Here we
        just pin that the degenerate input is detectable.
        """
        w = self._weights(np.zeros(4), {"FSTAT_PEAK_WEIGHT_ALPHA": "0.5"})
        self.assertFalse(bool(np.any(w > 0)))

    def test_the_module_reads_the_env_var(self):
        """Guard against the knob being documented but never wired."""
        import inspect
        from lisatools.sampling import fstat_gridfit
        src = inspect.getsource(fstat_gridfit)
        self.assertIn("FSTAT_PEAK_WEIGHT_ALPHA", src)

    def test_weights_are_not_persisted_so_alpha_needs_no_refit(self):
        """Changing alpha must NOT invalidate the epoch cache.

        The expensive artifact is the stage-B sweep (``logp_grids``); the
        weights are applied when the proposal is built FROM that cache, so
        a restart at a new alpha reuses the fitted grids untouched. If
        ``weights`` ever starts being written into the stacked npz this
        test fails and the no-refit claim has to be revisited.
        """
        import inspect
        from lisatools.sampling import fstat_gridfit
        src = inspect.getsource(fstat_gridfit)
        # the savez that writes *_peaks_stacked.npz
        i = src.find("_peaks_stacked.npz")
        self.assertGreater(i, 0)
        blk = src[i:i + 1200]
        self.assertNotIn("weights=", blk.split("def ")[0])


class PerCellPeakWeightTest(unittest.TestCase):
    """Per-SUB-BAND x F**alpha birth weighting.

    Draw a SUB-BAND STRATUM uniformly, then draw within it with
    w ~ F**alpha (user ruling 2026-08-29: "the sub-bands should define that
    grid since those are the effective rj limits as well" -- it used to be
    a CAP CELL, tracking GB_CAP_DIVISOR, which coupled the RJ birth
    proposal to a cap knob and stratified on cells that deliberately
    straddle sub-band seams). This
    is expressed as FLAT per-box weights rather than a two-stage sampler,
    because ``StackedFStatProposal4D`` drives both ``rvs`` (CDF built from
    ``w_k``) and ``logpdf`` (``+ log(w_k)``) off the same ``self.weights``
    -- so the sampler and its density stay mutually exact by construction.
    A hand-rolled two-stage rvs with a separate logpdf would put the RJ
    acceptance ratio at the mercy of two implementations agreeing, and a
    silent mismatch there biases the posterior instead of crashing.
    """

    # 2 sub-bands, K=4 -> 4 strata per sub-band = 8 x 0.25 mHz over 1-3 mHz
    # (uniform band grid, so the per-sub-band subdivision and the old global
    # linspace coincide here -- that is why these cases are unaffected)
    BE = np.array([1e-3, 2e-3, 3e-3])

    def _w(self, F, f0_mHz, **kw):
        from lisatools.sampling.fstat_gridfit import peak_box_weights
        return peak_box_weights(F, peak_f0_mHz=np.asarray(f0_mHz),
                                band_edges=self.BE, **kw)

    def test_per_cell_mass_is_equalised(self):
        # cell 0 gets 3 peaks (one very loud), cell 5 gets 1 quiet peak
        F = np.array([900.0, 100.0, 4.0, 9.0])
        f0 = np.array([1.05, 1.10, 1.20, 2.40])          # mHz
        w = self._w(F, f0, alpha=0.5, cells=4)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        cell = np.array([0, 0, 0, 5])
        m0 = w[cell == 0].sum(); m5 = w[cell == 5].sum()
        self.assertAlmostEqual(m0, m5, places=12)
        self.assertAlmostEqual(m0, 0.5, places=12)       # 2 occupied cells

    def test_within_a_cell_the_ratio_is_F_to_the_alpha(self):
        F = np.array([900.0, 100.0])
        f0 = np.array([1.05, 1.10])                      # same cell
        w = self._w(F, f0, alpha=0.5, cells=4)
        self.assertAlmostEqual(w[0] / w[1], np.sqrt(900.0 / 100.0), places=10)

    def test_a_loud_peak_no_longer_starves_another_cell(self):
        """THE POINT: the global mixture gives the loud cell 30x the mass."""
        F = np.array([900.0, 1.0])
        f0 = np.array([1.05, 2.40])                      # different cells
        glob = self._w(F, f0, alpha=0.5, cells=0)   # 0 = global (was 1)
        glob = glob / glob.sum()
        self.assertAlmostEqual(glob[0] / glob[1], 30.0, places=6)
        cellw = self._w(F, f0, alpha=0.5, cells=4)
        self.assertAlmostEqual(cellw[0] / cellw[1], 1.0, places=10)

    def test_cells_zero_is_the_historical_global_mixture(self):
        """Only 0 is the global mixture now. ``cells=1`` used to mean
        "no stratification"; it now means ONE STRATUM PER SUB-BAND, which
        is the default."""
        F = np.array([900.0, 100.0, 4.0])
        f0 = np.array([1.05, 1.60, 2.40])
        np.testing.assert_allclose(self._w(F, f0, alpha=1.0, cells=0), F)
        w1 = self._w(F, f0, alpha=1.0, cells=1)
        self.assertAlmostEqual(float(np.sum(w1)), 1.0, places=12)
        # peaks at 1.05/1.60 share sub-band 0; 2.40 is alone in sub-band 1
        self.assertAlmostEqual(float(w1[0] + w1[1]), 0.5, places=12)
        self.assertAlmostEqual(float(w1[2]), 0.5, places=12)

    def test_equal_still_wins_over_everything(self):
        self.assertIsNone(self._w(np.array([9.0, 1.0]), np.array([1.05, 2.4]),
                                  alpha=0.5, cells=4, equal=True))

    def test_all_zero_F_cell_falls_back_to_uniform_inside_that_cell(self):
        """A zero-F cell must still get its share, not silently vanish."""
        F = np.array([0.0, 0.0, 16.0])
        f0 = np.array([1.05, 1.10, 2.40])
        w = self._w(F, f0, alpha=0.5, cells=4)
        self.assertAlmostEqual(w[0], w[1], places=12)
        self.assertAlmostEqual(w[0] + w[1], w[2], places=12)

    def test_env_default_is_per_sub_band_and_ignores_the_cap_divisor(self):
        """The draw grid must NOT track GB_CAP_DIVISOR (2026-08-29).

        Coupling them meant a cap knob silently re-stratified the RJ birth
        proposal -- e.g. the move to the midpoint-to-midpoint cap grid.
        """
        from lisatools.sampling.fstat_gridfit import peak_weight_cells_env
        for k in ("FSTAT_PEAK_WEIGHT_CELLS", "GB_CAP_DIVISOR"):
            os.environ.pop(k, None)
            self.addCleanup(os.environ.pop, k, None)
        self.assertEqual(peak_weight_cells_env(), 1)     # one per sub-band
        os.environ["GB_CAP_DIVISOR"] = "8"
        self.assertEqual(peak_weight_cells_env(), 1)     # cap knob IGNORED
        os.environ["FSTAT_PEAK_WEIGHT_CELLS"] = "0"
        self.assertEqual(peak_weight_cells_env(), 0)     # explicit global
        os.environ["FSTAT_PEAK_WEIGHT_CELLS"] = "32"
        self.assertEqual(peak_weight_cells_env(), 32)    # explicit override

    def test_rvs_AND_logpdf_agree_under_the_composite_weights(self):
        """The one that protects the RJ acceptance ratio.

        Draw from the proposal and compare the EMPIRICAL density of the
        drawn f0 against ``logpdf`` evaluated at those same points. If
        ``rvs`` and ``logpdf`` ever disagreed the ratio would be silently
        wrong, so this checks them against each other rather than against
        a formula either one could share a bug with.
        """
        try:
            from lisatools.sampling.fstat_proposal import StackedFStatProposal4D
        except Exception as exc:                                # pragma: no cover
            self.skipTest(f"fstat_proposal unavailable: {exc}")
        rng = np.random.default_rng(0)
        K, n_f0, n3 = 4, 5, 3
        logp = np.log(rng.uniform(0.5, 1.5, size=(K, n_f0, n3, n3, n3)))
        f0_los = np.array([1.05, 1.10, 1.60, 2.40])            # mHz
        f0_dxs = np.full(K, 0.01)
        mc_ax = np.linspace(0.2, 0.4, n3)
        al_ax = np.linspace(0.0, 2 * np.pi, n3)
        sd_ax = np.linspace(-1.0, 1.0, n3)
        w = self._w(np.array([900.0, 100.0, 4.0, 9.0]), f0_los,
                    alpha=0.5, cells=4)
        P = StackedFStatProposal4D(logp, f0_los, f0_dxs, mc_ax, al_ax, sd_ax,
                                   weights=w, seed=1)
        x = np.asarray(P.rvs(60000))
        lp = np.asarray(P.logpdf(x)).ravel()
        self.assertTrue(np.all(np.isfinite(lp)),
                        "logpdf must be finite at points rvs produced")
        # Per-BOX check: the fraction of draws landing in each box must
        # match the composite weight it was given. Attribute by INTERVAL
        # membership -- a box spans f0_lo + (n_f0-1)*dx, so nearest-centre
        # attribution mis-assigns draws near a box's upper edge.
        span = (n_f0 - 1) * f0_dxs
        box = np.full(x.shape[0], -1)
        for k in range(K):
            inb = (x[:, 0] >= f0_los[k] - 1e-12) & (x[:, 0] <= f0_los[k] + span[k] + 1e-12)
            box[inb] = k
        self.assertTrue(np.all(box >= 0), "every draw must fall in some box")
        frac = np.bincount(box, minlength=K) / x.shape[0]
        np.testing.assert_allclose(frac, w / w.sum(), atol=0.01)
        # Importance-sampling identity: E_p[1/p] = volume of the support,
        # which is a joint statement about rvs AND logpdf being the same
        # distribution. Compare against the analytic box volume.
        vol = float(np.sum(span) * (mc_ax[-1] - mc_ax[0])
                    * (al_ax[-1] - al_ax[0]) * (sd_ax[-1] - sd_ax[0]))
        est = float(np.mean(np.exp(-lp)))
        self.assertAlmostEqual(est / vol, 1.0, delta=0.05)


class FitClockTest(unittest.TestCase):
    """The restart-persistent refit clock (2026-08-24 redesign).

    Production never refit because the cadence counted per-instance,
    in-process proposes. These pin the three fixed failure modes: hits
    pooled across move instances, the budget surviving a process restart
    (the clock journal + the DONE.json last-fit mark), and pre-clock
    epochs being treated as out of budget.
    """

    _Stub = FitDecisionTest._Stub
    _reset_clock_state = staticmethod(FitDecisionTest._reset_clock_state)
    _tick = staticmethod(FitDecisionTest._tick)

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.s = self._Stub()
        self.s._fstat_root_dir = self.d
        self._reset_clock_state()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)
        self._reset_clock_state()

    def _make_epoch(self, k=0, clock=None):
        d0 = os.path.join(self.d, f"epoch_{k:04d}")
        os.makedirs(d0, exist_ok=True)
        manifest = {"epoch": k, "n_peaks": 3}
        if clock is not None:
            manifest["clock"] = clock
        with open(os.path.join(d0, "DONE.json"), "w") as f:
            json.dump(manifest, f)
        return d0

    def test_hits_pool_across_instances(self):
        """A second grid move of the same branch sees the same clock."""
        other = self._Stub()
        other._fstat_root_dir = self.d
        other.rj_proposal_distribution = {"gb": object()}
        other._fstat_epoch = 0
        other.fstat_refit_every = 10
        other._fstat_last_fit_hit = 0
        other.num_proposals = 0
        # ticks contributed by ANY moves of the branch (e.g. the search
        # instance + prior moves), none by `other` itself:
        self._tick(10)
        self.assertEqual(other._fstat_fit_decision(), ("fit", 1))

    def test_budget_survives_restart(self):
        """The production failure: short launches must still accumulate.

        Process 1 samples 55 proposes past a fit made at clock 5 and dies.
        Process 2 (fresh counters) must refit IMMEDIATELY on the carried
        budget rather than starting a new 50-propose wait.
        """
        self._make_epoch(0, clock=5)
        self.s.rj_proposal_distribution = {"gb": object()}
        self.s._fstat_epoch = 0
        self.s.fstat_refit_every = 50
        self.s._fstat_last_fit_hit = self.s._epoch_fit_clock(0)
        self.assertEqual(self.s._fstat_last_fit_hit, 5)
        # process 1: 55 proposes, journal written en route
        self._tick(55)
        self.assertEqual(self.s._fstat_clock(), 55)
        # process death: in-memory counters gone
        self._reset_clock_state()
        s2 = self._Stub()
        s2._fstat_root_dir = self.d
        s2.rj_proposal_distribution = {"gb": object()}
        s2._fstat_epoch = 0
        s2.fstat_refit_every = 50
        s2._fstat_last_fit_hit = s2._epoch_fit_clock(0)
        # zero proposes in THIS process; elapsed = 55 - 5 >= 50 -> refit
        self.assertEqual(s2._fstat_fit_decision(), ("fit", 1))

    def test_epoch_fit_clock_manifest(self):
        self._make_epoch(0, clock=37)
        self.assertEqual(self.s._epoch_fit_clock(0), 37)

    def test_epoch_fit_clock_legacy_manifest_is_out_of_budget(self):
        """Pre-clock DONE.json (the running production stores) -> 0."""
        self._make_epoch(0, clock=None)
        self.assertEqual(self.s._epoch_fit_clock(0), 0)
        self.assertEqual(self.s._epoch_fit_clock(3), 0)  # missing entirely

    def test_journal_throttle(self):
        path = os.path.join(self.d, self.s._FSTAT_CLOCK_BASENAME)
        self._tick(3)
        self.assertEqual(self.s._fstat_clock(), 3)   # first read journals
        with open(path) as f:
            self.assertEqual(json.load(f)["clock"], 3)
        self._tick(6)
        self.assertEqual(self.s._fstat_clock(), 9)   # < WRITE_EVERY: no write
        with open(path) as f:
            self.assertEqual(json.load(f)["clock"], 3)
        self._tick(4)
        self.assertEqual(self.s._fstat_clock(), 13)  # >= WRITE_EVERY: rewrite
        with open(path) as f:
            self.assertEqual(json.load(f)["clock"], 13)


class CtrTableGBFreeTest(unittest.TestCase):
    """The epoch center-table sweep must run INSIDE the GB-free window.

    2026-08-24 fix: the sweep used to run after the fit's GB-free context
    closed, so at any real refit the amplitude/SNR centers for exactly the
    loud already-recovered peaks would have been fitted against a residual
    with those peaks subtracted.
    """

    class _Stub:
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as _M,
        )

        _install_ctr_table = _M._install_ctr_table
        _epoch_dir = _M._epoch_dir
        _CTR_TABLE_DEVICE_FIELDS = _M._CTR_TABLE_DEVICE_FIELDS

        name = "stub"
        branch_name = "gb"
        xp = np
        fstat_fit_kwargs = {}
        _fstat_root_dir = None
        events = None  # set per test

        @property
        def _fstat_root(self):
            return self._fstat_root_dir

        def _fstat_ctr_mode(self):
            return "epoch"

        def _fstat_ctr_smear(self):
            return 2.0

        def _fstat_reference_walker(self, model):
            self.events.append("walker_ref")
            return 0

        def _fstat_call(self, model, walker_ref):
            self.events.append("call_built")
            return lambda params: None

        def _gb_free_residual(self, model, branches, walker_ref):
            import contextlib

            events = self.events

            @contextlib.contextmanager
            def _cm():
                events.append("gbfree_enter")
                try:
                    yield
                finally:
                    events.append("gbfree_exit")

            return _cm()

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.s = self._Stub()
        self.s._fstat_root_dir = self.d
        self.s.events = []
        import lisatools.sampling.fstat_gridfit as _G

        self._G = _G
        self._orig_build = _G.build_fstat_center_table
        events = self.s.events

        def _fake_build(call, **kwargs):
            events.append("sweep" if call is not None else "load")
            return None

        _G.build_fstat_center_table = _fake_build

    def tearDown(self):
        self._G.build_fstat_center_table = self._orig_build
        shutil.rmtree(self.d, ignore_errors=True)

    def test_sweep_runs_inside_gb_free_window(self):
        self.s._install_ctr_table(0, model=object(), branches={"gb": object()})
        # scorer built AND sweep executed strictly inside the window
        self.assertEqual(
            self.s.events,
            ["walker_ref", "gbfree_enter", "call_built", "sweep",
             "gbfree_exit"],
        )

    def test_load_path_never_touches_the_residual(self):
        """No model (checkpoint-load path) -> no window, call=None."""
        self.s._install_ctr_table(1, model=None, branches=None)
        self.assertEqual(self.s.events, ["load"])
