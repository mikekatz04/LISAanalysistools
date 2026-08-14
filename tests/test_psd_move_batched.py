"""PSDMove multi-GPU build discipline + walker-batched scoring.

Tier 1 (device-local builds): every per-walker sensitivity build enters the
walker's owning ``device_context``; non-primary shards build against a
per-device domain-settings replica (the ``source_runtime`` idiom), and the
walker-independent instrument basis cache is keyed by device. CPU /
single-shard behavior is byte-identical to the historical path.

Tier 2 (concurrent shard groups): the batch's builds dispatch through the
ACA's ``_run_per_split`` — serial on CPU/single-split, threaded per split
under ``run_threaded``.

Tier 3 (walker-batched build + likelihood): the ``PSD_BATCH`` fast path
collapses the per-walker Python loop into one batched covariance build +
one batched likelihood per shard; parity with the per-walker route is
asserted at <= 1e-12 and via identical seeded accept/reject decisions.

All tests are CPU-only, using the fake multi-shard ACA in
``tests/_multishard.py`` (device-context entries are recorded, not
executed).
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:
    from _multishard import FakeMultiShardACA


class _RecordingBackend:
    """Duck-typed sensitivity backend recording the device of each build.

    ``basis_settings`` is a real domain-settings object so the move's
    per-device replica resolution runs against the genuine
    ``source_runtime`` helper.
    """

    def __init__(self, acs, settings):
        self.acs = acs
        self.basis_settings = settings
        self.use_splines = False
        self.calls = []

    def __call__(self, name, psd_params, galfor_params=None, sgwb_params=None,
                 basis_settings=None, **kwargs):
        self.calls.append(
            dict(
                name=name,
                device=self.acs.xp.current_device,
                basis_settings=basis_settings,
                psd_params=None if psd_params is None else np.asarray(psd_params).copy(),
            )
        )
        return dict(name=name, device=self.acs.xp.current_device)


def _make_move(PSDMove, acs, backend, **kwargs):
    return PSDMove(acs, {}, sensitivity_backend=backend, name="psd test",
                   **kwargs)


class DeviceLocalBuildTest(unittest.TestCase):
    """Tier 1: builds enter the owning device; replicas fan out per device."""

    def setUp(self):
        try:
            from lisatools.domains import WDMSettings
            from lisatools.globalfit.moves.psdmove import PSDMove
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"psdmove not available: {exc}")
        self.PSDMove = PSDMove
        # blocked layout: walkers 0,1 -> shard/device 0; walkers 2,3 -> 1
        self.acs = FakeMultiShardACA((3, 4), 4, 2, layout="blocked")
        self.settings = WDMSettings(8, 8, 5.0)

    def test_build_enters_owning_device_context(self):
        backend = _RecordingBackend(self.acs, self.settings)
        move = _make_move(self.PSDMove, self.acs, backend)
        walker_inds = np.arange(4)
        psd = np.tile([1.0e-11, 3.0e-15], (4, 1))
        move._build_batch(np.arange(4), walker_inds, psd, None, None)

        self.assertEqual(len(backend.calls), 4)
        by_name = {c["name"]: c for c in backend.calls}
        for w in range(4):
            self.assertEqual(
                by_name[f"walker_{w}"]["device"], int(self.acs.gpu_map[w]),
                f"walker {w} build ran on the wrong device",
            )
            # the built matrix was installed on the walker's AC
            self.assertIsNotNone(self.acs[w].sens_mat)

    def test_non_primary_device_gets_settings_replica(self):
        backend = _RecordingBackend(self.acs, self.settings)
        move = _make_move(self.PSDMove, self.acs, backend)
        move._build_batch(np.arange(4), np.arange(4),
                          np.tile([1.0e-11, 3.0e-15], (4, 1)), None, None)
        by_name = {c["name"]: c for c in backend.calls}
        # primary device (0): shared settings — no override forwarded
        for w in (0, 1):
            self.assertIsNone(by_name[f"walker_{w}"]["basis_settings"])
        # non-primary device (1): a distinct but value-equal replica
        replicas = [by_name[f"walker_{w}"]["basis_settings"] for w in (2, 3)]
        for rep in replicas:
            self.assertIsNotNone(rep)
            self.assertIsNot(rep, self.settings)
            self.assertEqual(rep, self.settings)
        # the replica is cached: both shard-1 walkers see the SAME object
        self.assertIs(replicas[0], replicas[1])

    def test_replica_resolution_primary_and_cpu(self):
        backend = _RecordingBackend(self.acs, self.settings)
        move = _make_move(self.PSDMove, self.acs, backend)
        # primary device and CPU (None) reuse the shared settings object
        self.assertIs(move._backend_settings_for_device(0), self.settings)
        self.assertIs(move._backend_settings_for_device(None), self.settings)
        # CPU ACA (gpus is None): device always resolves to None
        acs_cpu = FakeMultiShardACA((3, 4), 4, 1)
        acs_cpu.gpus = None
        move_cpu = _make_move(self.PSDMove, acs_cpu, backend)
        self.assertIsNone(move_cpu._walker_device(0))
        self.assertIs(move_cpu._backend_settings_for_device(None), self.settings)

    def test_warm_once_per_device(self):
        backend = _RecordingBackend(self.acs, self.settings)
        move = _make_move(self.PSDMove, self.acs, backend)
        self.assertEqual(move._build_warmed, set())
        move._build_batch(np.arange(4), np.arange(4),
                          np.tile([1.0e-11, 3.0e-15], (4, 1)), None, None)
        self.assertEqual(move._build_warmed, {0, 1})
        # second batch: no re-warm bookkeeping change, builds still run
        n_before = len(backend.calls)
        move._build_batch(np.arange(4), np.arange(4),
                          np.tile([1.0e-11, 3.0e-15], (4, 1)), None, None)
        self.assertEqual(len(backend.calls), n_before + 4)
        self.assertEqual(move._build_warmed, {0, 1})


class PerSplitDispatchTest(unittest.TestCase):
    """Tier 2: batch builds dispatch through ``acs._run_per_split``."""

    def setUp(self):
        try:
            from lisatools.domains import WDMSettings
            from lisatools.globalfit.moves.psdmove import PSDMove
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"psdmove not available: {exc}")
        self.PSDMove = PSDMove
        self.settings = WDMSettings(8, 8, 5.0)

    def _spy_run_per_split(self, acs, record):
        orig = acs._run_per_split

        def spy(worker, split_to_rows, run_threaded=None):
            record.append(
                dict(splits=sorted(split_to_rows), run_threaded=run_threaded)
            )
            return orig(worker, split_to_rows, run_threaded=run_threaded)

        acs._run_per_split = spy

    def test_build_batch_dispatches_per_split(self):
        acs = FakeMultiShardACA((3, 4), 4, 2, layout="blocked")
        backend = _RecordingBackend(acs, self.settings)
        move = _make_move(self.PSDMove, acs, backend)
        record = []
        self._spy_run_per_split(acs, record)

        move._build_batch(np.arange(4), np.arange(4),
                          np.tile([1.0e-11, 3.0e-15], (4, 1)), None, None)
        # one dispatch; the two warm rows (one per device) ran serially
        # before it, leaving one row per split
        self.assertEqual(len(record), 1)
        self.assertEqual(record[0]["splits"], [0, 1])
        # ACA default decides threading unless the move forces it
        self.assertIsNone(record[0]["run_threaded"])
        self.assertEqual(len(backend.calls), 4)
        by_name = {c["name"]: c for c in backend.calls}
        for w in range(4):
            self.assertEqual(by_name[f"walker_{w}"]["device"],
                             int(acs.gpu_map[w]))

    def test_threaded_dispatch_keeps_device_routing(self):
        acs = FakeMultiShardACA((3, 4), 4, 2, layout="blocked",
                                run_threaded=True)
        backend = _RecordingBackend(acs, self.settings)
        move = _make_move(self.PSDMove, acs, backend)
        # warm both devices first so the second batch is fully threaded
        move._build_batch(np.arange(4), np.arange(4),
                          np.tile([1.0e-11, 3.0e-15], (4, 1)), None, None)
        backend.calls.clear()
        move._build_batch(np.arange(4), np.arange(4),
                          np.tile([1.0e-11, 3.0e-15], (4, 1)), None, None)
        self.assertEqual(len(backend.calls), 4)
        by_name = {c["name"]: c for c in backend.calls}
        for w in range(4):
            self.assertEqual(by_name[f"walker_{w}"]["device"],
                             int(acs.gpu_map[w]))

    def test_move_run_threaded_flag_forces_threaded(self):
        acs = FakeMultiShardACA((3, 4), 4, 2, layout="blocked")
        backend = _RecordingBackend(acs, self.settings)
        move = _make_move(self.PSDMove, acs, backend, run_threaded=True)
        record = []
        self._spy_run_per_split(acs, record)
        move._build_batch(np.arange(4), np.arange(4),
                          np.tile([1.0e-11, 3.0e-15], (4, 1)), None, None)
        self.assertEqual(len(record), 1)
        self.assertIs(record[0]["run_threaded"], True)

    def test_run_rows_per_split_warms_and_runs_each_row_once(self):
        acs = FakeMultiShardACA((3, 4), 4, 2, layout="blocked")
        backend = _RecordingBackend(acs, self.settings)
        move = _make_move(self.PSDMove, acs, backend)
        ran = []
        move._run_rows_per_split(lambda r: ran.append(int(r)),
                                 np.arange(4), np.arange(4))
        self.assertEqual(sorted(ran), [0, 1, 2, 3])
        self.assertEqual(move._build_warmed, {0, 1})


class InstrumentBasisCacheDeviceKeyTest(unittest.TestCase):
    """Tier 1: ``InstrumentNoise._bases`` keys its cache by current device."""

    def setUp(self):
        try:
            import lisatools.sensitivity as sens_mod
            from lisatools import detector as lisa_models
            from lisatools.domains import FDSettings
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools not available: {exc}")
        self.sens_mod = sens_mod
        self.settings = FDSettings(N=32, df=1e-4)
        self.model = lisa_models.LISAModel(
            (1.5e-11) ** 2, (3.0e-15) ** 2, lisa_models.DefaultOrbits(), "test"
        )

    def test_cache_fans_out_per_device(self):
        sens_mod = self.sens_mod
        cache = {}
        comp = sens_mod.InstrumentNoise(model=self.model, basis_cache=cache)

        orig = sens_mod.current_device
        fake_device = [None]
        sens_mod.current_device = lambda xp: fake_device[0]
        try:
            b0 = comp._bases(self.settings)
            self.assertEqual(len(cache), 1)
            # same device -> cache hit (identical arrays returned)
            b0_again = comp._bases(self.settings)
            self.assertIs(b0[0], b0_again[0])
            self.assertEqual(len(cache), 1)
            # different device -> its own entry, numerically identical
            fake_device[0] = 1
            b1 = comp._bases(self.settings)
            self.assertEqual(len(cache), 2)
            self.assertIsNot(b0[0], b1[0])
            np.testing.assert_array_equal(np.asarray(b0[0]), np.asarray(b1[0]))
            np.testing.assert_array_equal(np.asarray(b0[1]), np.asarray(b1[1]))
        finally:
            sens_mod.current_device = orig


class BatchedLikelihoodParityTest(unittest.TestCase):
    """Tier 3: the batched likelihood twin matches the per-walker producer."""

    def _parity(self, settings, complex_data=False):
        from lisatools.diagnostic import (
            batched_residual_full_source_and_noise_likelihoods,
            residual_full_source_and_noise_likelihood,
        )
        from lisatools.sensitivity import SensitivityMatrixBase, _mat3x3_det_inv

        basis = tuple(settings.basis_shape_active)
        nw, nch = 5, 3
        rng = np.random.default_rng(3)
        # random SPD covariance stacks (elementwise over the basis grid)
        A = rng.normal(size=(nw, nch, nch) + basis)
        C = np.einsum("wik...,wjk...->wij...", A, A)
        for i in range(nch):
            C[:, i, i] += 3.0
        if complex_data:
            res = rng.normal(size=(nw, nch) + basis) + 1j * rng.normal(
                size=(nw, nch) + basis
            )
        else:
            res = rng.normal(size=(nw, nch) + basis)

        per_walker = []
        for w in range(nw):
            sm = SensitivityMatrixBase(settings)
            sm.sens_mat = C[w].copy()
            sig = settings.associated_class(res[w].copy(), settings)
            per_walker.append(
                residual_full_source_and_noise_likelihood(sig, sm)
            )
        per_walker = np.asarray(per_walker, dtype=float)

        Cb = np.moveaxis(C, 0, 2)  # (nch, nch, nw, *basis)
        detC, invC = _mat3x3_det_inv(Cb, np)
        bad = ~np.isfinite(invC)
        if bad.any():
            invC = np.where(bad, 0.0, invC)
        det_bad = ~np.isfinite(detC)
        if det_bad.any():
            detC = np.where(det_bad, 1.0, detC)
        batched = np.asarray(
            batched_residual_full_source_and_noise_likelihoods(
                res, Cb, invC, detC, settings
            ),
            dtype=float,
        )
        self.assertTrue(np.all(np.isfinite(batched)))
        np.testing.assert_allclose(batched, per_walker, rtol=1e-12)

    def test_wdm_parity(self):
        try:
            from lisatools.domains import WDMSettings
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools not available: {exc}")
        self._parity(WDMSettings(8, 8, 5.0))

    def test_fd_parity(self):
        try:
            from lisatools.domains import FDSettings
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools not available: {exc}")
        self._parity(FDSettings(N=64, df=1e-4), complex_data=True)


class BatchedMoveStockParityTest(unittest.TestCase):
    """Tier 3, end to end: PSD_BATCH route vs the per-walker route on the
    real synthetic noise_only fit (real ACA, real composite backend, real
    WDM grid) — scores equal to <= 1e-12 and identical seeded accept/reject
    decisions."""

    @classmethod
    def setUpClass(cls):
        import os
        import tempfile

        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        cls._file_store = tempfile.mkdtemp(prefix="psdbatch_test_")
        cls._old_store = os.environ.get("FILE_STORE_DIR")
        os.environ["FILE_STORE_DIR"] = cls._file_store

    @classmethod
    def tearDownClass(cls):
        import os
        import shutil

        if cls._old_store is None:
            os.environ.pop("FILE_STORE_DIR", None)
        else:
            os.environ["FILE_STORE_DIR"] = cls._old_store
        shutil.rmtree(cls._file_store, ignore_errors=True)

    def _fixture(self, variant="noise_only", sampled=("psd", "galfor")):
        try:
            from mpi4py import MPI
            from eryn.state import BranchSupplemental

            from lisatools.globalfit.recipe import build_noise_moves
            from lisatools.globalfit.run import GlobalFit
            from lisatools.globalfit.stock import erebor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"stock global-fit stack not available: {exc}")

        np.random.seed(1234)  # load_info start draws -> reproducible fixture
        fit = getattr(erebor, variant)(
            nwalkers=4, ntemps=2, data_mode="synthetic"
        )
        curr = fit.build()
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

        def make_move(psd_batch=True):
            _, pe_move = build_noise_moves(
                curr.engine_info, curr, acs, priors,
                sampled_branches=list(sampled), num_repeats=1,
            )
            pe_move.psd_batch = bool(psd_batch)
            return pe_move

        return curr, state, acs, make_move, tuple(sampled)

    def _log_like_parity(self, variant, sampled):
        curr, state, acs, make_move, sampled = self._fixture(
            variant=variant, sampled=sampled
        )
        from eryn.state import BranchSupplemental

        move_b = make_move(psd_batch=True)
        move_p = make_move(psd_batch=False)

        coords = {
            k: np.array(move_b._work_branch(state, k).coords)
            for k in sampled
        }
        nt_mod, nw_mod = coords["psd"].shape[:2]
        supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(nw_mod), (nt_mod, 1))},
            base_shape=(nt_mod, nw_mod),
            copy=True,
        )

        self.assertTrue(
            move_b._batched_route_ready(),
            f"batched route unexpectedly OFF: {move_b._batch_route_reason}",
        )
        self.assertFalse(move_p._batched_route_ready())

        logl_b, _ = move_b.compute_log_like(coords, supps=supps)
        logl_p, _ = move_p.compute_log_like(coords, supps=supps)

        finite = np.isfinite(logl_p) & (logl_p > -1e299)
        self.assertTrue(finite.any())
        np.testing.assert_allclose(logl_b, logl_p, rtol=1e-12)

    def test_compute_log_like_parity(self):
        self._log_like_parity("noise_only", ("psd", "galfor"))

    def test_compute_log_like_parity_sgwb(self):
        # covers the batched SGWB magnitude branch (kernel fast path is
        # gated off whenever the model carries an sgwb branch)
        self._log_like_parity("noise_sgwb", ("psd", "galfor", "sgwb"))

    def test_seeded_propose_decisions_match(self):
        from copy import deepcopy
        from types import SimpleNamespace

        results = []
        for psd_batch in (True, False):
            curr, state, acs, make_move, _ = self._fixture()
            move = make_move(psd_batch=psd_batch)
            model = SimpleNamespace(
                analysis_container_arr=acs,
                map_fn=map,
                random=np.random.RandomState(42),
            )
            np.random.seed(7)  # temperature swaps / permutations
            new_state, accepted = move.propose(model, deepcopy(state))
            results.append(
                dict(
                    accepted=np.asarray(accepted).copy(),
                    psd=np.array(new_state.branches_coords["psd"]),
                    galfor=np.array(new_state.branches_coords["galfor"]),
                    logl=np.array(new_state.log_like),
                )
            )

        b, p = results
        np.testing.assert_array_equal(b["accepted"], p["accepted"])
        np.testing.assert_allclose(b["psd"], p["psd"], rtol=0, atol=0)
        np.testing.assert_allclose(b["galfor"], p["galfor"], rtol=0, atol=0)
        np.testing.assert_allclose(b["logl"], p["logl"], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
