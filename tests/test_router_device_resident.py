"""Focused equality tests for the 2026-08 router/fill perf rework.

Covers, on the numpy path (device_context no-ops, so the device-resident
code is exercised verbatim):

* ``GB_ROUTER_DEVICE_RESIDENT`` 1-vs-0 bit-identity for the routed legs
  (get_ll / fill_template / route_information_matrix);
* the BandView ``_fancy_dispatch`` fast row-wise path vs the legacy
  broadcast kernel (get/set, WDM-style 4-tuples, fallback patterns);
* ``BandView.accumulate`` vs the gather/add/scatter ``+=`` it replaces;
* the static shard-lookup / bound-count / WDM ``min_freq_inds`` caches and
  their invalidation hooks.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:  # direct invocation from inside tests/
    from _multishard import FakeMultiShardACA


class _MiniEngine:
    """Minimal single-shard engine stub for knob-equality tests.

    Outputs are deterministic functions of (device, intra row, params sum)
    so any staging/assembly divergence between the two router paths shows
    up as a value mismatch.
    """

    def __init__(self):
        self.calls = []

    def get_ll(self, holder, params_phys, *, data_index, noise_index,
               N_vals, phase_maximize=False, waveform_kwargs, **kwargs):
        intra = np.asarray(data_index).astype(float)
        dev = holder.gpus[0] if holder.gpus is not None else 0
        p = np.asarray(params_phys)
        self.calls.append(dict(device=dev, intra=intra.copy(), params=p.copy(),
                               N_vals=None if N_vals is None
                               else np.asarray(N_vals).copy()))
        n = len(intra)
        self.d_h_out = 1000.0 * dev + intra + p.sum(axis=1)
        self.h_h_out = 2000.0 * dev + intra
        self.phase_angle = 0.25 * intra
        self.kept_out = (intra % 2 == 0)
        return 100.0 * dev + intra + 1e-3 * p.sum(axis=1)

    def fill_template(self, holder, params_phys, params_index, N_vals, *,
                      factor, waveform_kwargs, **kwargs):
        self.calls.append(dict(
            kind="fill",
            device=holder.gpus[0] if holder.gpus is not None else 0,
            intra=np.asarray(params_index).copy(),
            params=np.asarray(params_phys).copy(),
            N_vals=None if N_vals is None else np.asarray(N_vals).copy(),
            factor=factor,
            kwargs={k: np.asarray(v) for k, v in kwargs.items()},
        ))


class _MiniComp:
    """information_matrix stub (device, intra)-stamped for reassembly checks."""

    def information_matrix(self, params, holder, *, inds, noise_index,
                           **swap_kwargs):
        intra = np.asarray(noise_index)
        dev = holder.gpus[0] if holder.gpus is not None else 0
        out = np.zeros((len(intra), 3, 3))
        for r in range(len(intra)):
            np.fill_diagonal(out[r], 10.0 * dev + float(intra[r]))
        return out


class DeviceResidentKnobTest(unittest.TestCase):
    """GB_ROUTER_DEVICE_RESIDENT=1 (default) vs =0 must be bit-identical."""

    PER_BAND = (2, 6)
    NUM_ACS = 7
    NUM_SHARDS = 3

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbbands router not available: {exc}")
        self.RoutedEngine = _RoutedBandEngine
        self.data_index = np.array([4, 0, 5, 2, 6, 1])
        self.params = np.arange(len(self.data_index) * 4, dtype=float
                                ).reshape(len(self.data_index), 4)
        self.N_vals = np.full(len(self.data_index), 64)

    def _holder(self):
        return FakeMultiShardACA(self.PER_BAND, self.NUM_ACS,
                                 self.NUM_SHARDS, layout="striped")

    def _get_ll(self, knob):
        with mock.patch.dict(os.environ,
                             {"GB_ROUTER_DEVICE_RESIDENT": knob}):
            router = self.RoutedEngine(_MiniEngine())
            ll = router.get_ll(
                self._holder(), self.params,
                data_index=self.data_index, noise_index=self.data_index,
                N_vals=self.N_vals, waveform_kwargs={},
            )
            return (np.asarray(ll), np.asarray(router.d_h_out),
                    np.asarray(router.h_h_out),
                    np.asarray(router.phase_angle),
                    np.asarray(router.kept_out))

    def test_get_ll_bit_identical_across_knob(self):
        out_dev = self._get_ll("1")
        out_host = self._get_ll("0")
        for a, b in zip(out_dev, out_host):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.dtype, b.dtype)

    def test_get_ll_default_is_device_resident(self):
        # No env var set -> knob defaults to "1"; results must equal the
        # explicit host-staging path.
        env = dict(os.environ)
        env.pop("GB_ROUTER_DEVICE_RESIDENT", None)
        with mock.patch.dict(os.environ, env, clear=True):
            router = self.RoutedEngine(_MiniEngine())
            ll = np.asarray(router.get_ll(
                self._holder(), self.params,
                data_index=self.data_index, noise_index=self.data_index,
                N_vals=self.N_vals, waveform_kwargs={},
            ))
        np.testing.assert_array_equal(ll, self._get_ll("0")[0])

    def test_engine_inputs_identical_across_knob(self):
        seen = {}
        for knob in ("1", "0"):
            with mock.patch.dict(os.environ,
                                 {"GB_ROUTER_DEVICE_RESIDENT": knob}):
                engine = _MiniEngine()
                self.RoutedEngine(engine).get_ll(
                    self._holder(), self.params,
                    data_index=self.data_index, noise_index=self.data_index,
                    N_vals=self.N_vals, waveform_kwargs={},
                )
                seen[knob] = sorted(engine.calls, key=lambda c: c["device"])
        for c1, c0 in zip(seen["1"], seen["0"]):
            np.testing.assert_array_equal(c1["params"], c0["params"])
            np.testing.assert_array_equal(c1["intra"], c0["intra"])
            np.testing.assert_array_equal(c1["N_vals"], c0["N_vals"])

    def test_fill_template_bit_identical_across_knob(self):
        slab = np.arange(self.NUM_ACS, dtype=np.int32) * 3
        seen = {}
        for knob in ("1", "0"):
            with mock.patch.dict(os.environ,
                                 {"GB_ROUTER_DEVICE_RESIDENT": knob}):
                engine = _MiniEngine()
                self.RoutedEngine(engine).fill_template(
                    self._holder(), self.params, self.data_index,
                    self.N_vals, factor=-1, waveform_kwargs={},
                    slab_min_f=slab,
                )
                seen[knob] = sorted(engine.calls, key=lambda c: c["device"])
        self.assertEqual(len(seen["1"]), len(seen["0"]))
        for c1, c0 in zip(seen["1"], seen["0"]):
            np.testing.assert_array_equal(c1["params"], c0["params"])
            np.testing.assert_array_equal(c1["intra"], c0["intra"])
            np.testing.assert_array_equal(c1["kwargs"]["slab_min_f"],
                                          c0["kwargs"]["slab_min_f"])

    def test_information_matrix_bit_identical_across_knob(self):
        outs = {}
        for knob in ("1", "0"):
            with mock.patch.dict(os.environ,
                                 {"GB_ROUTER_DEVICE_RESIDENT": knob}):
                outs[knob] = np.asarray(
                    self.RoutedEngine.route_information_matrix(
                        _MiniComp(), self._holder(), self.params,
                        inds=None, noise_index=self.data_index,
                    ))
        np.testing.assert_array_equal(outs["1"], outs["0"])
        # sanity: (device, intra)-stamped diagonal proves per-shard routing
        holder = self._holder()
        dev = holder.gpu_map[self.data_index]
        intra = np.array([np.where(holder.gpu_splits[s] == b)[0][0]
                          for s, b in zip(dev, self.data_index)])
        np.testing.assert_array_equal(outs["1"][:, 0, 0],
                                      10.0 * dev + intra.astype(float))

    def test_partition_lookup_cached_on_holder(self):
        holder = self._holder()
        router = self.RoutedEngine(_MiniEngine())
        parts1 = router._partition(holder, self.data_index)
        self.assertTrue(hasattr(holder, "_shard_lookup_cache"))
        cache_obj = holder._shard_lookup_cache
        parts2 = router._partition(holder, self.data_index)
        self.assertIs(holder._shard_lookup_cache, cache_obj)  # reused
        for (p1, i1, _), (p2, i2, _) in zip(parts1, parts2):
            np.testing.assert_array_equal(p1, p2)
            np.testing.assert_array_equal(i1, i2)


class _SlotSpaceComp:
    """information_matrix stub proving slot-space coherence.

    Encodes each row's output from its GLOBAL buffer slot -- recovered from
    the shard view's ``rows`` (intra -> global) when routed, or directly on
    a single-shard passthrough -- so any wrong-space/wrong-shard routing
    changes the assembled numbers.
    """

    def __init__(self):
        self.calls = []

    def information_matrix(self, params, holder, *, inds, noise_index,
                           data_index=None, **kw):
        di = None if data_index is None else np.asarray(data_index)
        self.calls.append(dict(
            holder=holder,
            data_index=None if di is None else di.copy(),
            noise_index=np.asarray(noise_index).copy(),
        ))
        n = np.asarray(params).shape[0]
        out = np.zeros((n, 2, 2))
        rows = getattr(holder, "rows", None)
        if di is None:
            # chunked leg marker: negative stamp so tests can tell the legs
            # apart while keeping deterministic values
            key = -1.0 - np.asarray(noise_index).astype(float)
        elif rows is not None:
            key = np.asarray(rows)[di].astype(float)  # intra -> global
        else:
            key = di.astype(float)                    # global already
        for r in range(n):
            np.fill_diagonal(out[r], key[r] + 1.0)
        return out


class SlotShardInfomatRouteTest(unittest.TestCase):
    """route_information_matrix slot-space routing (sig-het fast leg).

    The buffer holder stripes slots across shards (band parity round-robin)
    while the parent ACA blocks walkers contiguously -- different axes. The
    slot leg must partition by the BUFFER and hand each shard INTRA rows.
    """

    PER_BAND = (2, 4)
    NUM_SLOTS = 8
    NUM_WALKERS = 6
    NUM_SHARDS = 2

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbbands router not available: {exc}")
        self.RoutedEngine = _RoutedBandEngine
        # slot space: striped (parity round-robin, like band_gpu_assignment)
        self.buffer_holder = FakeMultiShardACA(
            self.PER_BAND, self.NUM_SLOTS, self.NUM_SHARDS, layout="striped")
        # walker space: contiguous blocks (the parent-ACA layout)
        self.parent = FakeMultiShardACA(
            self.PER_BAND, self.NUM_WALKERS, self.NUM_SHARDS,
            layout="blocked")
        self.slots = np.array([5, 0, 3, 6, 2])       # global buffer slots
        self.walkers = np.array([0, 4, 2, 5, 1])     # per-source walkers
        self.params = np.arange(len(self.slots) * 3, dtype=float
                                ).reshape(len(self.slots), 3)

    def test_slot_leg_partitions_by_buffer_and_remaps_intra(self):
        comp = _SlotSpaceComp()
        out = np.asarray(self.RoutedEngine.route_information_matrix(
            comp, self.parent, self.params,
            inds=None, noise_index=self.walkers,
            data_index=self.slots, slot_holder=self.buffer_holder,
        ))
        # every shard call got INTRA rows in the BUFFER's slot space
        for call in comp.calls:
            view = call["holder"]
            rows = np.asarray(view.rows)
            intra = call["data_index"]
            self.assertIsNotNone(intra)
            # intra ids valid for the shard and mapping back to the
            # requested global slots on that shard
            self.assertTrue(np.all(intra < len(rows)))
            back = rows[intra]
            self.assertTrue(np.all(np.isin(back, self.slots)))
            # noise_index rides the same intra rows (per-slot invC rows)
            np.testing.assert_array_equal(call["noise_index"], intra)
        # assembled output == the single-shard reference (global-slot keyed)
        single = FakeMultiShardACA(self.PER_BAND, self.NUM_SLOTS, 1,
                                   layout="striped")
        ref = np.asarray(self.RoutedEngine.route_information_matrix(
            _SlotSpaceComp(), single, self.params,
            inds=None, noise_index=self.slots, data_index=self.slots,
        ))
        np.testing.assert_array_equal(out, ref)
        np.testing.assert_array_equal(out[:, 0, 0],
                                      self.slots.astype(float) + 1.0)

    def test_slot_leg_bit_identical_across_device_resident_knob(self):
        outs = {}
        for knob in ("1", "0"):
            with mock.patch.dict(os.environ,
                                 {"GB_ROUTER_DEVICE_RESIDENT": knob}):
                outs[knob] = np.asarray(
                    self.RoutedEngine.route_information_matrix(
                        _SlotSpaceComp(), self.parent, self.params,
                        inds=None, noise_index=self.walkers,
                        data_index=self.slots,
                        slot_holder=self.buffer_holder,
                    ))
        np.testing.assert_array_equal(outs["1"], outs["0"])

    def test_multi_shard_without_slot_holder_gates_to_chunked(self):
        comp = _SlotSpaceComp()
        with self.assertLogs("lisatools.globalfit.moves.gbbands",
                             level="WARNING") as cm:
            out = np.asarray(self.RoutedEngine.route_information_matrix(
                comp, self.parent, self.params,
                inds=None, noise_index=self.walkers,
                data_index=self.slots,   # global slots, no slot_holder
            ))
        self.assertTrue(any("dropping data_index" in m for m in cm.output))
        # every comp call took the chunked leg (no data_index forwarded)
        self.assertTrue(comp.calls)
        for call in comp.calls:
            self.assertIsNone(call["data_index"])
        # and the output is the walker-routed chunked stamp, reassembled in
        # source order (negative marker keyed by intra walker rows)
        self.assertTrue(np.all(out[:, 0, 0] <= 0.0))

    def test_single_shard_passthrough_keeps_global_slots(self):
        single = FakeMultiShardACA(self.PER_BAND, self.NUM_SLOTS, 1,
                                   layout="striped")
        comp = _SlotSpaceComp()
        self.RoutedEngine.route_information_matrix(
            comp, single, self.params,
            inds=None, noise_index=self.slots, data_index=self.slots,
            slot_holder=single,
        )
        self.assertEqual(len(comp.calls), 1)
        np.testing.assert_array_equal(comp.calls[0]["data_index"],
                                      self.slots)
        self.assertIs(comp.calls[0]["holder"], single)


class FancyDispatchFastPathTest(unittest.TestCase):
    """Fast row-wise ``_fancy_dispatch`` vs the legacy broadcast kernel."""

    PER_BAND = (2, 5, 3)  # (nchannels, Nf, Nt) -- WDM-style slabs
    NUM_ACS = 6
    NUM_SHARDS = 2

    def setUp(self):
        try:
            from lisatools.analysiscontainer import BandView
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools BandView not available: {exc}")
        self.BandView = BandView
        self.aca = FakeMultiShardACA(self.PER_BAND, self.NUM_ACS,
                                     self.NUM_SHARDS, layout="striped")
        self.view = BandView(self.aca, kind="data")
        self.ref = self.aca.reference_rows()

    def _wdm_tuple(self, bands, layer_lo, Nf_use=2):
        nc, _, Nt = self.PER_BAND
        inds1 = bands[:, None, None, None]
        inds2 = np.arange(nc)[None, :, None, None]
        inds3 = (layer_lo[:, None, None, None]
                 + np.arange(Nf_use)[None, None, :, None])
        inds4 = np.arange(Nt)[None, None, None, :]
        return (inds1, inds2, inds3, inds4)

    def test_wdm_style_get_matches_reference_and_legacy(self):
        bands = np.array([0, 3, 5, 2])
        layer_lo = np.array([0, 1, 2, 3])
        idx = self._wdm_tuple(bands, layer_lo)
        out = self.view[idx]
        np.testing.assert_array_equal(np.asarray(out), self.ref[idx])
        legacy = self.view._fancy_broadcast(
            idx[0], idx[1:], None, mode="get", val=None)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(legacy))

    def test_fast_path_is_taken_for_leading_axis_band(self):
        bands = np.array([1, 4])
        idx = self._wdm_tuple(bands, np.array([0, 2]))
        with mock.patch.object(
                self.BandView, "_fancy_broadcast",
                side_effect=AssertionError("legacy path taken")):
            out = self.view[idx]
        np.testing.assert_array_equal(np.asarray(out), self.ref[idx])

    def test_trailing_axis_band_falls_back_and_matches(self):
        # band varying along the LAST axis: outside the fast path's shape
        # contract -> must fall back to the legacy kernel, same numbers.
        bands = np.array([0, 3])[None, :]           # (1, 2): band on axis 1
        chans = np.array([0, 1])[:, None]           # (2, 1)
        freqs = np.zeros((1, 1), dtype=int)
        times = np.ones((1, 1), dtype=int)
        idx = (bands, chans, freqs, times)
        out = self.view[idx]
        np.testing.assert_array_equal(np.asarray(out), self.ref[idx])

    def test_set_per_row_payload_matches_reference(self):
        bands = np.array([2, 5, 1])
        idx = self._wdm_tuple(bands, np.array([1, 0, 2]))
        payload = (np.arange(np.prod(np.broadcast_shapes(
            *[a.shape for a in idx]))).reshape(
                np.broadcast_shapes(*[a.shape for a in idx]))
            .astype(complex))
        self.view[idx] = payload
        self.ref[idx] = payload
        np.testing.assert_array_equal(
            np.asarray(self.view.gather()), self.ref)

    def test_set_scalar_matches_reference(self):
        bands = np.array([0, 4])
        idx = self._wdm_tuple(bands, np.array([3, 3]))
        self.view[idx] = -7.5
        self.ref[idx] = -7.5
        np.testing.assert_array_equal(
            np.asarray(self.view.gather()), self.ref)

    def test_set_broadcast_payload_matches_reference(self):
        # payload with leading axis 1 (broadcast across rows)
        bands = np.array([1, 3, 5])
        idx = self._wdm_tuple(bands, np.array([0, 0, 1]))
        tgt = np.broadcast_shapes(*[a.shape for a in idx])
        payload = (np.arange(np.prod(tgt[1:])).reshape((1,) + tgt[1:])
                   .astype(complex))
        self.view[idx] = payload
        self.ref[idx] = payload
        np.testing.assert_array_equal(
            np.asarray(self.view.gather()), self.ref)

    def test_repeated_bands_get(self):
        # same band twice with different layer windows (RJ repeat pattern)
        bands = np.array([2, 2, 3])
        idx = self._wdm_tuple(bands, np.array([0, 2, 1]))
        np.testing.assert_array_equal(
            np.asarray(self.view[idx]), self.ref[idx])


class AccumulateTest(unittest.TestCase):
    """BandView.accumulate == the gather/add/scatter ``+=`` it replaces."""

    PER_BAND = (2, 4)
    NUM_ACS = 6
    NUM_SHARDS = 2

    def setUp(self):
        try:
            from lisatools.analysiscontainer import BandView
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools BandView not available: {exc}")
        self.BandView = BandView

    def _fresh(self):
        aca = FakeMultiShardACA(self.PER_BAND, self.NUM_ACS,
                                self.NUM_SHARDS, layout="striped")
        return aca, self.BandView(aca, kind="data")

    def test_accumulate_matches_iadd(self):
        idx = np.array([0, 3, 5, 2])
        delta = (np.arange(len(idx) * np.prod(self.PER_BAND))
                 .reshape((len(idx),) + self.PER_BAND).astype(complex))
        aca_a, view_a = self._fresh()
        aca_b, view_b = self._fresh()
        view_a.accumulate(idx, delta)
        view_b[idx] += delta          # legacy gather/add/scatter
        np.testing.assert_array_equal(
            np.asarray(view_a.gather()), np.asarray(view_b.gather()))

    def test_accumulate_scalar(self):
        idx = np.array([1, 4])
        aca_a, view_a = self._fresh()
        aca_b, view_b = self._fresh()
        view_a.accumulate(idx, 2.5)
        view_b[idx] += 2.5
        np.testing.assert_array_equal(
            np.asarray(view_a.gather()), np.asarray(view_b.gather()))

    def test_accumulate_writes_in_place_on_shards(self):
        aca, view = self._fresh()
        before = [buf for buf in aca.linear_data_arr]
        view.accumulate(np.array([0, 1]), 1.0)
        for b_old, b_new in zip(before, aca.linear_data_arr):
            self.assertIs(b_old, b_new)  # no shard buffer was rebound


class LookupCacheTest(unittest.TestCase):
    """shard_lookup_maps / bound-count / WDM min_freq_inds caches."""

    def setUp(self):
        try:
            from lisatools.analysiscontainer import (BandView,
                                                     shard_lookup_maps)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools not available: {exc}")
        self.BandView = BandView
        self.shard_lookup_maps = shard_lookup_maps

    def test_maps_correct_and_cached(self):
        aca = FakeMultiShardACA((2, 3), 7, 3, layout="striped")
        sm, g2i = self.shard_lookup_maps(aca)
        np.testing.assert_array_equal(sm, aca.split_map)
        for s, rows in enumerate(aca.gpu_splits):
            np.testing.assert_array_equal(g2i[rows], np.arange(len(rows)))
        sm2, g2i2 = self.shard_lookup_maps(aca)
        self.assertIs(sm, sm2)
        self.assertIs(g2i, g2i2)

    def test_bound_counts_cached_per_n_bands(self):
        aca = FakeMultiShardACA((2, 3), 8, 2, layout="blocked")
        view = self.BandView(aca, kind="data", n_bands=5)
        shards = view._shards
        # blocked split of 8 over 2: rows [0..3], [4..7]; bound ids < 5
        self.assertEqual([s.shape[0] for s in shards], [4, 1])
        self.assertIn(5, aca._bandview_bound_counts)
        # a different bound count gets its own key
        view3 = self.BandView(aca, kind="data", n_bands=3)
        self.assertEqual([s.shape[0] for s in view3._shards], [3, 0])
        self.assertIn(3, aca._bandview_bound_counts)

    def test_wdm_min_freq_inds_cached_and_invalidated(self):
        try:
            from lisatools.domains import WDMSettings
            from lisatools.globalfit.moves.gbbands import SubBandBuffer
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools deps not available: {exc}")
        try:
            wdm = WDMSettings(Nf=8, Nt=8, dt=10.0)
        except Exception as exc:  # noqa: BLE001 -- environment-dependent
            self.skipTest(f"WDMSettings construction failed: {exc}")

        class _Stub:
            pass

        stub = _Stub()
        stub._basis_settings = wdm
        stub._n_slots_alloc = 5
        stub.xp = np
        first = SubBandBuffer.min_freq_inds.fget(stub)
        self.assertEqual(first.shape[0], 5)
        self.assertTrue(np.all(first == int(wdm.ind_min_f)))
        second = SubBandBuffer.min_freq_inds.fget(stub)
        self.assertIs(first, second)  # cached, no fresh per-call alloc
        # rebind/resize hook drops the cache -> fresh array next read
        SubBandBuffer._invalidate_slab_metadata_cache(stub)
        third = SubBandBuffer.min_freq_inds.fget(stub)
        self.assertIsNot(first, third)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(third))
        # allocation-size change (resize flows) also refreshes
        stub._n_slots_alloc = 7
        fourth = SubBandBuffer.min_freq_inds.fget(stub)
        self.assertEqual(fourth.shape[0], 7)


if __name__ == "__main__":
    unittest.main()
