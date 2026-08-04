"""Structural tests for the GB/VGB multi-shard engine router.

Exercises ``lisatools.globalfit.moves.gbbands._RoutedBandEngine`` +
``_ShardHolderView`` against a NumPy :class:`FakeMultiShardACA` and a stub
single-shard engine: row partitioning, intra-shard index mapping, per-shard
device-context entry, output scatter order, ``min_freq_inds`` pointer
identity across in-place parent updates, per-slot kwarg slicing, the
per-slot ``slab_min_f`` shard slice, the per-device comp / engine replicas
(including the sig-het in-model reference isolation), and the single-shard
passthrough.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._multishard import FakeDeviceComp, FakeMultiShardACA
except ImportError:  # direct invocation from inside tests/
    from _multishard import FakeDeviceComp, FakeMultiShardACA


class _StubEngine:
    """Single-shard engine stand-in that records every call.

    Outputs encode (shard device, intra row) so the router's scatter order
    is verifiable: ``ll = 1000 * device + intra``.
    """

    def __init__(self):
        self.calls = []

    def _record(self, kind, holder, **info):
        self.calls.append(dict(kind=kind, holder=holder, **info))

    def fill_template(self, holder, params_phys, params_index, N_vals, *,
                      factor, waveform_kwargs, **kwargs):
        assert len(holder.linear_data_arr) == 1, "engine must see one shard"
        # Real engines read len(holder) as num_data/num_noise -- exercise the
        # __len__ dunder so a missing one fails HERE, not on the cluster.
        assert len(holder) == holder.acs_total_entries
        self._record(
            "fill", holder,
            intra=np.asarray(params_index).copy(),
            params=np.asarray(params_phys).copy(),
            N_vals=None if N_vals is None else np.asarray(N_vals).copy(),
            factor=factor, kwargs=dict(kwargs),
            device=holder.xp.current_device
            if hasattr(holder.xp, "current_device") else None,
        )

    def get_ll(self, holder, params_phys, *, data_index, noise_index,
               N_vals, phase_maximize=False, waveform_kwargs, **kwargs):
        assert len(holder.linear_data_arr) == 1
        assert len(holder) == holder.acs_total_entries
        intra = np.asarray(data_index)
        dev = holder.gpus[0] if holder.gpus is not None else 0
        self._record("get_ll", holder, intra=intra.copy(),
                     params=np.asarray(params_phys).copy())
        n = len(intra)
        self.d_h_out = 1000.0 * dev + intra.astype(float)
        self.h_h_out = 2000.0 * dev + intra.astype(float)
        self.phase_angle = None
        self.kept_out = np.ones(n, dtype=bool)
        return 1000.0 * dev + intra.astype(float)

    def get_swap_ll(self, holder, params_remove_phys, params_add_phys, *,
                    data_index, noise_index, N_vals, phase_maximize=False,
                    waveform_kwargs, **kwargs):
        from gbgpu.gb_likelihood import SwapLLResult

        assert len(holder.linear_data_arr) == 1
        assert len(holder) == holder.acs_total_entries
        intra = np.asarray(data_index).astype(float)
        dev = holder.gpus[0] if holder.gpus is not None else 0
        n = len(intra)
        return SwapLLResult(
            ll_diff=1000.0 * dev + intra,
            d_h_add=intra + 0.1,
            d_h_remove=intra + 0.2,
            hh_add=intra + 0.3,
            hh_remove=intra + 0.4,
            hh_cross=None,
            opt_snr_add=intra + 0.5,
            phase_angle=None,
            kept=np.ones(n, dtype=bool),
        )

    def setup_in_model(self, holder, params_phys, data_index, N_vals=None):
        self._record("setup_in_model", holder,
                     intra=np.asarray(data_index).copy())
        return getattr(self, "return_truthy", False)

    def clear_in_model(self):
        self._record("clear_in_model", None)


class _StubComp:
    """Raw-comp stand-in exposing ``information_matrix`` (the proposal-Cholesky
    entry point that bypasses the wrapped engine). Encodes (device, intra
    noise row) into the Fisher stack so routing + reassembly is verifiable.
    """

    def __init__(self, ndim=3):
        self.ndim = ndim
        self.calls = []

    def information_matrix(self, params, holder, *, inds, noise_index,
                           **swap_kwargs):
        assert len(holder.linear_data_arr) == 1, "comp must see one shard"
        assert len(holder) == holder.acs_total_entries
        intra = np.asarray(noise_index)
        dev = holder.gpus[0] if holder.gpus is not None else 0
        self.calls.append(dict(holder=holder, intra=intra.copy(),
                               nparams=np.asarray(params).shape[0]))
        n = len(intra)
        nd = len(inds) if inds is not None else self.ndim
        out = np.zeros((n, nd, nd), dtype=float)
        # diagonal encodes 1000*device + intra so scatter order is checkable
        diag = 1000.0 * dev + intra.astype(float)
        for r in range(n):
            np.fill_diagonal(out[r], diag[r])
        return out

    def get_fstat_ll_wdm(self, params, wdm_holder, data_index=None,
                         noise_index=None, **kwargs):
        """F-stat raw-comp stand-in: (N (n,4), M (n,10)) encode
        1000*device + intra (+ column offset) so routing + reassembly
        order is checkable, mirroring ``information_matrix``."""
        assert len(wdm_holder.linear_data_arr) == 1, "comp must see one shard"
        assert len(wdm_holder) == wdm_holder.acs_total_entries
        intra = np.asarray(data_index)
        dev = wdm_holder.gpus[0] if wdm_holder.gpus is not None else 0
        self.calls.append(dict(kind="fstat", holder=wdm_holder,
                               intra=intra.copy(),
                               nparams=np.asarray(params).shape[0],
                               kwargs=dict(kwargs)))
        base = 1000.0 * dev + intra.astype(float)
        N = base[:, None] + np.arange(4)[None, :]
        M = base[:, None] + np.arange(10)[None, :]
        return N, M


class _StubSigHetEngine:
    """Engine stand-in reproducing the sig-het in-model reference contract.

    Mirrors ``GBSignalHetComputations.setup_in_model``: a flat
    ``_slot_to_ref`` map over INTRA-shard slot ids, a coefficient stash keyed
    by reference row, and a single ``_in_model`` flag whose presence turns the
    next call from a fresh build into a mid-block PATCH of the existing
    references. That flat state is exactly what two shards collide over when
    they share one comp -- their intra-shard slot ids both start at zero -- so
    reproducing it here is what makes the cross-shard corruption regression
    testable on CPU.
    """

    def __init__(self, device=None):
        self.device = device
        self._in_model = None
        self._slot_to_ref = None
        self.stash = None
        self.n_builds = 0
        self.n_patches = 0

    def setup_in_model(self, holder, params_phys, data_index, N_vals=None):
        slots = np.asarray(data_index, dtype=int)
        # First parameter column stands in for the whole coefficient block.
        vals = np.asarray(params_phys, dtype=float)[:, 0].copy()
        if self._in_model is not None:
            if int(slots.max()) >= len(self._slot_to_ref):
                raise RuntimeError(
                    "sig-het in-model patch hit a slot outside the block's "
                    "reference set.")
            ref = self._slot_to_ref[slots]
            if np.any(ref < 0):
                raise RuntimeError(
                    "sig-het in-model patch hit a slot with no reference.")
            self.stash[ref] = vals
            self.n_patches += 1
            return True
        slot_map = np.full(int(slots.max()) + 1, -1, dtype=int)
        slot_map[slots] = np.arange(len(slots))
        self._slot_to_ref = slot_map
        self.stash = vals
        self._in_model = True
        self.n_builds += 1
        return True

    def clear_in_model(self):
        self._in_model = None
        self._slot_to_ref = None


class ShardRouterTest(unittest.TestCase):
    PER_BAND = (3, 8)
    NUM_ACS = 7
    NUM_SHARDS = 3

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbbands import (
                _RoutedBandEngine, _ShardHolderView)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbbands router not available: {exc}")
        self.RoutedEngine = _RoutedBandEngine
        self.ShardView = _ShardHolderView
        self.holder = FakeMultiShardACA(
            self.PER_BAND, self.NUM_ACS, self.NUM_SHARDS,
            layout="striped", with_min_freq_inds=True,
        )
        self.engine = _StubEngine()
        self.router = _RoutedBandEngine(self.engine)
        # rows deliberately shuffled and cross-shard
        self.data_index = np.array([4, 0, 5, 2, 6, 1])
        self.params = np.arange(len(self.data_index) * 9, dtype=float
                                ).reshape(len(self.data_index), 9)

    def _expected_intra(self, rows):
        intra = np.empty(self.NUM_ACS, dtype=int)
        for rr in self.holder.gpu_splits:
            intra[rr] = np.arange(len(rr))
        return intra[rows]

    def test_get_ll_partition_and_scatter(self):
        ll = self.router.get_ll(
            self.holder, self.params,
            data_index=self.data_index, noise_index=self.data_index,
            N_vals=None, waveform_kwargs={},
        )
        dev = self.holder.gpu_map[self.data_index].astype(float)
        intra = self._expected_intra(self.data_index).astype(float)
        np.testing.assert_array_equal(np.asarray(ll), 1000.0 * dev + intra)
        # stashed outputs assembled in the same global order
        np.testing.assert_array_equal(
            np.asarray(self.router.d_h_out), 1000.0 * dev + intra)
        np.testing.assert_array_equal(
            np.asarray(self.router.h_h_out), 2000.0 * dev + intra)
        self.assertTrue(np.all(np.asarray(self.router.kept_out)))
        self.assertIsNone(self.router.phase_angle)
        # every engine call saw a single-shard view of the right shard
        seen_devices = sorted(
            c["holder"].gpus[0] for c in self.engine.calls
            if c["kind"] == "get_ll"
        )
        self.assertEqual(
            seen_devices,
            sorted(set(self.holder.gpu_map[self.data_index].tolist())),
        )

    def test_get_ll_params_rows_match_partition(self):
        self.router.get_ll(
            self.holder, self.params,
            data_index=self.data_index, noise_index=self.data_index,
            N_vals=None, waveform_kwargs={},
        )
        for call in self.engine.calls:
            if call["kind"] != "get_ll":
                continue
            dev = call["holder"].gpus[0]
            pos = np.where(self.holder.gpu_map[self.data_index] == dev)[0]
            np.testing.assert_array_equal(call["params"], self.params[pos])
            np.testing.assert_array_equal(
                call["intra"], self._expected_intra(self.data_index[pos]))

    def test_get_swap_ll_reassembly(self):
        try:
            import gbgpu.gb_likelihood  # noqa: F401
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbgpu not available: {exc}")
        res = self.router.get_swap_ll(
            self.holder, self.params, self.params,
            data_index=self.data_index, noise_index=self.data_index,
            N_vals=None, waveform_kwargs={},
        )
        dev = self.holder.gpu_map[self.data_index].astype(float)
        intra = self._expected_intra(self.data_index).astype(float)
        np.testing.assert_array_equal(
            np.asarray(res.ll_diff), 1000.0 * dev + intra)
        np.testing.assert_array_equal(
            np.asarray(res.opt_snr_add), intra + 0.5)
        self.assertIsNone(res.hh_cross)   # None on every shard -> None
        self.assertIsNone(res.phase_angle)
        self.assertTrue(np.all(np.asarray(res.kept)))

    def test_fill_template_routes_and_slices_slab(self):
        slab = np.arange(self.NUM_ACS, dtype=float) * 1.5
        N_vals = np.full(len(self.data_index), 128)
        self.router.fill_template(
            self.holder, self.params, self.data_index, N_vals,
            factor=-1, waveform_kwargs={}, slab_min_f=slab,
        )
        fills = [c for c in self.engine.calls if c["kind"] == "fill"]
        self.assertEqual(
            len(fills), len(set(self.holder.gpu_map[self.data_index]))
        )
        for call in fills:
            dev = call["holder"].gpus[0]
            rows = self.holder.gpu_splits[dev]
            # per-slot kwarg sliced to the shard's rows (intra alignment)
            np.testing.assert_array_equal(call["kwargs"]["slab_min_f"], slab[rows])
            self.assertEqual(call["factor"], -1)
            pos = np.where(self.holder.gpu_map[self.data_index] == dev)[0]
            np.testing.assert_array_equal(call["N_vals"], N_vals[pos])

    def test_min_freq_inds_pointer_identity_across_refresh(self):
        self.router.get_ll(
            self.holder, self.params,
            data_index=self.data_index, noise_index=self.data_index,
            N_vals=None, waveform_kwargs={},
        )
        views = self.holder._shard_holder_views
        stores = [v.min_freq_inds for v in views]
        # parent updates its starts IN PLACE (cell swap) ...
        self.holder.min_freq_inds[...] = self.holder.min_freq_inds + 7
        self.router.get_ll(
            self.holder, self.params,
            data_index=self.data_index, noise_index=self.data_index,
            N_vals=None, waveform_kwargs={},
        )
        for v, store in zip(self.holder._shard_holder_views, stores):
            # ... and every view refreshed the SAME array object in place
            self.assertIs(v.min_freq_inds, store)
            np.testing.assert_array_equal(
                v.min_freq_inds,
                self.holder.min_freq_inds[v.rows],
            )

    def test_shard_view_protocol(self):
        view = self.ShardView(self.holder, 1)
        self.assertEqual(len(view.linear_data_arr), 1)
        self.assertIs(view.linear_data_arr[0], self.holder.linear_data_arr[1])
        self.assertEqual(len(view.linear_psd_arr), 1)
        self.assertEqual(len(view.data_shaped), 1)
        self.assertEqual(view.acs_total_entries,
                         len(self.holder.gpu_splits[1]))
        # ``len(view)`` is what the chunked-het / gb_likelihood engines read
        # as ``num_data``/``num_noise``. It MUST be an explicit dunder --
        # len() resolves __len__ on the type, bypassing __getattr__
        # delegation -- and must equal the shard's cell count.
        self.assertEqual(len(view), len(self.holder.gpu_splits[1]))
        self.assertEqual(len(view), view.acs_total_entries)
        self.assertEqual(view.gpus, [1])
        np.testing.assert_array_equal(
            view.start_freq_ind,
            self.holder.start_freq_ind[self.holder.gpu_splits[1]],
        )
        # public long tail delegates to the parent
        self.assertEqual(view.per_band_shape, self.holder.per_band_shape)

    def test_single_shard_passthrough(self):
        single = FakeMultiShardACA(self.PER_BAND, 4, 1, layout="striped")
        engine = _StubEngine()
        router = self.RoutedEngine(engine)
        idx = np.array([2, 0, 3])
        ll = router.get_ll(
            single, self.params[:3],
            data_index=idx, noise_index=idx,
            N_vals=None, waveform_kwargs={},
        )
        # passthrough: engine saw the holder itself, indices untranslated
        self.assertIs(engine.calls[0]["holder"], single)
        np.testing.assert_array_equal(engine.calls[0]["intra"], idx)
        np.testing.assert_array_equal(np.asarray(ll), idx.astype(float))
        # router mirrors engine outputs after passthrough
        np.testing.assert_array_equal(
            np.asarray(router.d_h_out), idx.astype(float))

    def test_information_matrix_routes_and_reassembles(self):
        comp = _StubComp(ndim=3)
        idx = np.array([5, 0, 3, 2, 1])  # global walker/noise rows
        params = np.arange(len(idx) * 3, dtype=float).reshape(len(idx), 3)
        info = self.RoutedEngine.route_information_matrix(
            comp, self.holder, params, inds=[0, 1, 2], noise_index=idx)
        info = np.asarray(info)
        self.assertEqual(info.shape, (len(idx), 3, 3))
        # each source scored once, on its owning device, at the right intra row
        split_map = np.asarray(self.holder.split_map)
        intra_lut = np.empty(self.holder.acs_total_entries, dtype=int)
        for rows in self.holder.gpu_splits:
            intra_lut[np.asarray(rows)] = np.arange(len(rows))
        for row, g in enumerate(idx):
            dev = int(self.holder.gpu_map[g])
            expected = 1000.0 * dev + intra_lut[g]
            np.testing.assert_allclose(np.diag(info[row]), expected)

    def test_information_matrix_single_shard_passthrough(self):
        comp = _StubComp(ndim=3)
        single = FakeMultiShardACA(self.PER_BAND, 4, 1, layout="striped")
        idx = np.array([2, 0, 3])
        params = np.zeros((3, 3))
        info = np.asarray(self.RoutedEngine.route_information_matrix(
            comp, single, params, inds=[0, 1, 2], noise_index=idx))
        self.assertIs(comp.calls[0]["holder"], single)  # no view wrapping
        for row, g in enumerate(idx):
            np.testing.assert_allclose(np.diag(info[row]), float(g))

    def test_fstat_routes_and_reassembles(self):
        comp = _StubComp()
        N, M = self.RoutedEngine.route_fstat_ll(
            comp, "get_fstat_ll_wdm", self.holder, self.params,
            data_index=self.data_index, noise_index=self.data_index,
            convert_to_ra_dec=False)
        dev = np.asarray(self.holder.gpu_map)[self.data_index].astype(float)
        intra = self._expected_intra(self.data_index).astype(float)
        base = 1000.0 * dev + intra
        np.testing.assert_array_equal(
            np.asarray(N), base[:, None] + np.arange(4)[None, :])
        np.testing.assert_array_equal(
            np.asarray(M), base[:, None] + np.arange(10)[None, :])
        # every comp call saw a single-shard view on its owning device,
        # with the extra kwarg passed through
        fcalls = [c for c in comp.calls if c.get("kind") == "fstat"]
        for call in fcalls:
            self.assertEqual(len(call["holder"].linear_data_arr), 1)
            self.assertIs(call["kwargs"]["convert_to_ra_dec"], False)
        seen = sorted(c["holder"].gpus[0] for c in fcalls)
        self.assertEqual(
            seen,
            sorted(set(
                np.asarray(self.holder.gpu_map)[self.data_index].tolist())),
        )

    def test_fstat_single_shard_passthrough(self):
        comp = _StubComp()
        single = FakeMultiShardACA(self.PER_BAND, 4, 1, layout="striped")
        di = np.array([2, 0, 1])
        params = self.params[:3]
        self.RoutedEngine.route_fstat_ll(
            comp, "get_fstat_ll_wdm", single, params,
            data_index=di, noise_index=di)
        # passthrough: comp saw the ORIGINAL holder and ORIGINAL indices
        self.assertEqual(len(comp.calls), 1)
        self.assertIs(comp.calls[0]["holder"], single)
        np.testing.assert_array_equal(comp.calls[0]["intra"], di)

    def test_fstat_multi_shard_requires_data_index(self):
        comp = _StubComp()
        with self.assertRaises(ValueError):
            self.RoutedEngine.route_fstat_ll(
                comp, "get_fstat_ll_wdm", self.holder, self.params,
                data_index=None)

    def test_fstat_rejects_slab_holders(self):
        comp = _StubComp()
        self.holder.slab_min_f = np.zeros(self.NUM_ACS)
        try:
            with self.assertRaises(NotImplementedError):
                self.RoutedEngine.route_fstat_ll(
                    comp, "get_fstat_ll_wdm", self.holder, self.params,
                    data_index=self.data_index)
        finally:
            del self.holder.slab_min_f

    def test_sig_het_in_model_rejected_multi_shard(self):
        self.engine.return_truthy = True
        with self.assertRaises(NotImplementedError):
            self.router.setup_in_model(
                self.holder, self.params, self.data_index)

    def test_noop_in_model_routes_multi_shard(self):
        self.engine.return_truthy = False
        ret = self.router.setup_in_model(
            self.holder, self.params, self.data_index)
        self.assertIsNone(ret)
        n_setup = sum(
            1 for c in self.engine.calls if c["kind"] == "setup_in_model")
        self.assertEqual(
            n_setup, len(set(self.holder.gpu_map[self.data_index]))
        )

    # ---------------- per-slot slab_min_f shard slice ----------------

    def test_shard_view_slices_slab_min_f(self):
        """``slab_min_f`` is per BUFFER SLOT, so the view must hand each
        shard its OWN rows -- ``__getattr__`` delegation would return the
        parent's global-slot array against intra-shard indices."""
        self.holder.slab_min_f = np.arange(
            10, 10 + self.NUM_ACS, dtype=np.int32)
        try:
            for s in range(self.NUM_SHARDS):
                view = self.ShardView(self.holder, s)
                np.testing.assert_array_equal(
                    np.asarray(view.slab_min_f),
                    self.holder.slab_min_f[view.rows],
                )
                # band_slab_Nf is a scalar extent -> shard-invariant, and
                # keeps delegating to the parent.
                self.holder.band_slab_Nf = 13
                self.assertEqual(view.band_slab_Nf, 13)
                del self.holder.band_slab_Nf
        finally:
            del self.holder.slab_min_f

    def test_shard_view_slab_min_f_survives_cell_swap(self):
        """A cell swap rewrites the parent's per-slot origins in place; the
        views must re-slice on ``refresh_row_metadata`` (which every routed
        call runs) rather than keep the block's first values."""
        self.holder.slab_min_f = np.arange(
            10, 10 + self.NUM_ACS, dtype=np.int32)
        try:
            views = self.RoutedEngine._shard_views(self.holder)
            first = [np.asarray(v.slab_min_f).copy() for v in views]
            # cell swap: the parent recomputes its per-slot origins
            self.holder.slab_min_f = self.holder.slab_min_f + 100
            self.RoutedEngine._shard_views(self.holder)
            for v, before in zip(views, first):
                np.testing.assert_array_equal(
                    np.asarray(v.slab_min_f),
                    self.holder.slab_min_f[v.rows],
                )
                self.assertFalse(np.array_equal(np.asarray(v.slab_min_f),
                                                before))
        finally:
            del self.holder.slab_min_f

    def test_shard_view_slab_min_f_none_without_parent_metadata(self):
        """The parent residual ACA carries no slab metadata -> None, not a
        delegated AttributeError."""
        view = self.ShardView(self.holder, 0)
        self.assertIsNone(view.slab_min_f)

    # ---------------- per-device comp replicas ----------------

    def _fake_comp(self):
        return FakeDeviceComp(self.holder.xp, tag="wdm")

    def test_fstat_dispatches_to_device_local_comp(self):
        """Each shard's F-stat call must reach the comp replica whose OWN
        build device matches the shard's device (the comp output stamps its
        build device, so a foreign-device comp is visible in the result)."""
        comp = self._fake_comp()
        N, _M = self.RoutedEngine.route_fstat_ll(
            comp, "get_fstat_ll_wdm", self.holder, self.params,
            data_index=self.data_index, noise_index=self.data_index)
        dev = np.asarray(self.holder.gpu_map)[self.data_index].astype(float)
        # column 0 carries the build device of the comp that produced the row
        np.testing.assert_array_equal(np.asarray(N)[:, 0], dev)
        # ... and every shard's comp reports its own device, not shard 0's
        seen = {c["holder"].gpus[0]: c["build_device"] for c in comp.calls}
        for shard_dev, build_dev in seen.items():
            self.assertEqual(int(build_dev), int(shard_dev))

    def test_information_matrix_dispatches_to_device_local_comp(self):
        comp = self._fake_comp()
        idx = np.array([5, 0, 3, 2, 1])
        params = np.zeros((len(idx), 3))
        info = np.asarray(self.RoutedEngine.route_information_matrix(
            comp, self.holder, params, inds=[0, 1, 2], noise_index=idx))
        for row, g in enumerate(idx):
            np.testing.assert_allclose(
                np.diag(info[row]), float(self.holder.gpu_map[g]))

    def test_comp_replica_reused_across_calls(self):
        """Replicas are allocate-once and module-cached: a second routed call
        must not rebuild them (memory-lifecycle rule)."""
        comp = self._fake_comp()
        for _ in range(2):
            self.RoutedEngine.route_fstat_ll(
                comp, "get_fstat_ll_wdm", self.holder, self.params,
                data_index=self.data_index, noise_index=self.data_index)
        from lisatools.globalfit.stock.erebor.source_runtime import (
            _DEVICE_GB_COMP_REPLICAS)
        mine = [k for k in _DEVICE_GB_COMP_REPLICAS if k[0] == id(comp)]
        # one replica per NON-prototype device seen (shard 0 reuses ``comp``)
        self.assertEqual(
            sorted(dev for _cid, dev in mine),
            sorted(set(self.holder.gpu_map[self.data_index].tolist()) - {0}),
        )

    def test_comp_device_assert_fires_on_foreign_buffers(self):
        """The permanent guard: a comp that cannot be replicated (no recorded
        ctor args) must fail loudly on a foreign shard instead of silently
        depending on peer access."""
        comp = self._fake_comp()

        class _Unreplicable(FakeDeviceComp):
            @property
            def args(self):
                raise AttributeError("args")

        comp.__class__ = _Unreplicable
        view = self.ShardView(self.holder, 1)
        with self.assertRaises(RuntimeError) as ctx:
            self.RoutedEngine._comp_for(comp, self.holder, view)
        self.assertIn("device", str(ctx.exception))

    def test_single_shard_never_builds_replicas(self):
        """No-regression on the fast path: one shard -> the SAME engine
        object, no comp replica, nothing allocated."""
        single = FakeMultiShardACA(self.PER_BAND, 4, 1, layout="striped")
        engine = _StubEngine()
        router = self.RoutedEngine(
            engine, engine_factory=lambda dev, primary: _StubEngine())
        idx = np.array([2, 0, 3])
        router.get_ll(single, self.params[:3], data_index=idx,
                      noise_index=idx, N_vals=None, waveform_kwargs={})
        self.assertIs(engine.calls[0]["holder"], single)
        self.assertEqual(router.device_engines, {})
        self.assertIs(router.wrapped_engine, engine)

    def test_primary_shard_reuses_the_prototype_engine(self):
        router = self.RoutedEngine(
            self.engine, engine_factory=lambda dev, primary: _StubEngine())
        views = self.RoutedEngine._shard_views(self.holder)
        # holder.gpus[0] == 0 and the stub engine records no build device,
        # so shard 0 is the prototype's shard.
        self.assertIs(router._engine_for(self.holder, views[0]), self.engine)
        self.assertIsNot(router._engine_for(self.holder, views[1]),
                         self.engine)

    # ---------------- sig-het in-model, per shard ----------------

    def _sighet_router(self):
        """Router whose factory hands every non-primary device its own
        sig-het engine (the per-device comp replica, in miniature)."""
        made = {}

        def _factory(device, primary):
            made[int(device)] = _StubSigHetEngine(device=int(device))
            return made[int(device)]

        primary = _StubSigHetEngine(device=0)
        return self.RoutedEngine(primary, engine_factory=_factory), primary, made

    def test_sig_het_shards_build_independent_references(self):
        """The L1 regression test: two shards whose intra-shard slot ids
        BOTH start at zero must each take the fresh-build branch, and
        neither's stash may move when the other builds."""
        holder = FakeMultiShardACA(self.PER_BAND, 6, 2, layout="blocked")
        router, primary, made = self._sighet_router()
        data_index = np.arange(6)              # rows 0-2 shard 0, 3-5 shard 1
        params = np.arange(6 * 9, dtype=float).reshape(6, 9)
        router.setup_in_model(holder, params, data_index)

        self.assertEqual(sorted(made), [1])    # only device 1 needed a replica
        replica = made[1]
        # BOTH built fresh -- no shard took the mid-block patch branch
        self.assertEqual((primary.n_builds, primary.n_patches), (1, 0))
        self.assertEqual((replica.n_builds, replica.n_patches), (1, 0))
        # ... against their OWN rows, through their OWN slot map
        np.testing.assert_array_equal(primary.stash, params[0:3, 0])
        np.testing.assert_array_equal(replica.stash, params[3:6, 0])
        # ... and the maps are per shard: intra slot 0 exists on both
        self.assertEqual(int(primary._slot_to_ref[0]), 0)
        self.assertEqual(int(replica._slot_to_ref[0]), 0)
        # shard 1's build did not touch shard 0's references
        before = primary.stash.copy()
        router.setup_in_model(holder, params[3:6], data_index[3:6])
        np.testing.assert_array_equal(primary.stash, before)
        self.assertEqual(replica.n_patches, 1)

    def test_sig_het_clear_in_model_fans_out_to_replicas(self):
        """L4: a replica that keeps ``_in_model`` set makes the NEXT block's
        first setup silently take the patch branch."""
        holder = FakeMultiShardACA(self.PER_BAND, 6, 2, layout="blocked")
        router, primary, made = self._sighet_router()
        params = np.arange(6 * 9, dtype=float).reshape(6, 9)
        router.setup_in_model(holder, params, np.arange(6))
        self.assertIsNotNone(primary._in_model)
        self.assertIsNotNone(made[1]._in_model)
        router.clear_in_model()
        self.assertIsNone(primary._in_model)
        self.assertIsNone(made[1]._in_model)
        for eng in (primary, made[1]):
            self.assertIsNone(eng._slot_to_ref)

    def test_sig_het_without_factory_still_refuses_multi_shard(self):
        """Un-wired routers must keep failing loudly: one shared engine
        cannot hold two shards' references."""
        holder = FakeMultiShardACA(self.PER_BAND, 6, 2, layout="blocked")
        router = self.RoutedEngine(_StubSigHetEngine(device=0))
        params = np.arange(6 * 9, dtype=float).reshape(6, 9)
        with self.assertRaises(NotImplementedError):
            router.setup_in_model(holder, params, np.arange(6))


def _have_gbgpu_comps() -> bool:
    try:
        from gbgpu.gbcomps import GBFDComputations, GBWDMComputations  # noqa
        from gbgpu.gbsignalhetcomputations import (  # noqa
            GBSignalHetComputations)
    except (ImportError, ModuleNotFoundError):
        return False
    return True


@unittest.skipUnless(
    _have_gbgpu_comps(),
    "requires gbgpu.gbcomps / gbgpu.gbsignalhetcomputations",
)
class GBCompReplicaContractTest(unittest.TestCase):
    """The per-device replica rebuild contract, on REAL comps.

    ``_device_local_gb_comp`` is a no-op on CPU (there is no second device to
    replicate onto), so the shard-router tests above can only prove the
    dispatch. This class proves the other half -- that the recorded
    reconstruction arguments actually rebuild an equivalent comp -- which is
    otherwise only discoverable on a multi-GPU node. A missing/renamed
    constructor argument would be a TypeError on the cluster and nowhere else.

    Small grid on purpose; the cost is dominated by the orbit configuration.
    """

    @classmethod
    def setUpClass(cls):
        from lisatools.domains import WDMSettings

        cls.wdm_settings = WDMSettings(Nf=32, Nt=64, dt=15.0,
                                       force_backend="cpu")

    def _wdm_comp(self):
        from gbgpu.gbcomps import GBWDMComputations

        return GBWDMComputations(
            self.wdm_settings, t_ref=0.0, Nt_sub=16, n_pad=2, N_sparse=64,
            tdi_config="1st generation", force_backend="cpu")

    def test_wdm_comp_rebuilds_from_recorded_args(self):
        comp = self._wdm_comp()
        replica = type(comp)(*comp.args, **comp.kwargs)
        self.assertIsNot(replica, comp)
        np.testing.assert_allclose(np.asarray(replica.wdm_window),
                                   np.asarray(comp.wdm_window))
        np.testing.assert_allclose(np.asarray(replica.chunk_t_starts),
                                   np.asarray(comp.chunk_t_starts))
        np.testing.assert_array_equal(np.asarray(replica.chunk_keep_lo),
                                      np.asarray(comp.chunk_keep_lo))
        self.assertEqual(replica.n_chunks, comp.n_chunks)
        self.assertEqual(replica.resolved_tukey_alpha,
                         comp.resolved_tukey_alpha)
        self.assertEqual(replica.t_obs_start, comp.t_obs_start)
        # CPU records no device; on CUDA this is the device the buffers
        # above were allocated on and is what the router keys replicas by.
        self.assertIsNone(comp._build_device)

    def test_fd_comp_rebuilds_from_recorded_args(self):
        from gbgpu.gbcomps import GBFDComputations
        from lisatools.domains import FDSettings

        comp = GBFDComputations(
            FDSettings(N=1024, df=1e-6), t_ref=0.0, N_sparse=64,
            tdi_config="1st generation", force_backend="cpu")
        replica = type(comp)(*comp.args, **comp.kwargs)
        self.assertIsNot(replica, comp)
        for attr in ("ind_min", "ind_max", "N_sparse", "nchannels", "df",
                     "t_ref", "t_start", "tukey_alpha", "edge_frac"):
            self.assertEqual(getattr(replica, attr), getattr(comp, attr))
        self.assertIsNone(comp._build_device)

    def test_sighet_wrapper_rebuilds_from_its_knob_dict(self):
        """The sig-het wrapper is built through ``for_band_engine``, never
        ``__class__(*args)``, so its replica knobs are recovered from the
        instance's own ``_g`` grid/knob dict -- the mapping the replica
        helper uses must reproduce ``_g`` EXACTLY (both the snapped
        ``nt_layer`` and the resolved ``n_cp_build`` are idempotent)."""
        from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

        from lisatools.globalfit.stock.erebor.source_runtime import (
            _SIGHET_REPLICA_KNOBS)

        comp = self._wdm_comp()
        sig = GBSignalHetComputations.for_band_engine(
            comp, nt_layer=8, n_sparse_fd=128)
        knobs = {
            name: sig._g[gkey]
            for name, gkey in _SIGHET_REPLICA_KNOBS.items()
            if gkey in sig._g
        }
        # every for_band_engine knob must be recoverable
        import inspect

        params = inspect.signature(
            GBSignalHetComputations.for_band_engine).parameters
        expected = {n for n, p in params.items()
                    if p.kind is inspect.Parameter.KEYWORD_ONLY}
        self.assertEqual(set(knobs), expected)

        replica = GBSignalHetComputations.for_band_engine(
            type(comp)(*comp.args, **comp.kwargs), **knobs)
        self.assertEqual(replica._g, sig._g)
        np.testing.assert_allclose(np.asarray(replica.window_full),
                                   np.asarray(sig.window_full))
        np.testing.assert_array_equal(np.asarray(replica.n_sparse_local),
                                      np.asarray(sig.n_sparse_local))
        self.assertIsNone(replica._in_model)


if __name__ == "__main__":
    unittest.main()
