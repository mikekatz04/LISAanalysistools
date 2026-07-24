"""Structural tests for the GB/VGB multi-shard engine router.

Exercises ``lisatools.globalfit.moves.gbbands._RoutedBandEngine`` +
``_ShardHolderView`` against a NumPy :class:`FakeMultiShardACA` and a stub
single-shard engine: row partitioning, intra-shard index mapping, per-shard
device-context entry, output scatter order, ``min_freq_inds`` pointer
identity across in-place parent updates, per-slot kwarg slicing, and the
single-shard passthrough.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:  # direct invocation from inside tests/
    from _multishard import FakeMultiShardACA


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


if __name__ == "__main__":
    unittest.main()
