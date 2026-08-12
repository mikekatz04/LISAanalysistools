"""Structural tests for the GB/VGB multi-shard engine router.

Exercises ``lisatools.globalfit.moves.gbbands._RoutedBandEngine`` +
``_ShardHolderView`` against a NumPy :class:`FakeMultiShardACA` and a stub
single-shard engine: row partitioning, intra-shard index mapping, per-shard
device-context entry, output scatter order, ``min_freq_inds`` pointer
identity across in-place parent updates, per-slot kwarg slicing, the
per-slot ``slab_min_f`` shard slice, the per-device comp / engine replicas
(including the sig-het in-model reference isolation), the sig-het F-stat
scorer route (``route_sighet_fstat``), and the single-shard passthrough.
"""

from __future__ import annotations

import os
import threading
import unittest
from unittest import mock

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
        # Concurrency hook (RouterThreadedDispatchTest): with a 2-party
        # barrier installed, both shards must be inside get_ll at once or
        # the wait times out -> BrokenBarrierError -> test failure.
        b = getattr(self, "barrier", None)
        if b is not None:
            b.wait()
        self._record("get_ll", holder, intra=intra.copy(),
                     params=np.asarray(params_phys).copy(),
                     thread=threading.get_ident(),
                     device=holder.xp.current_device
                     if hasattr(holder.xp, "current_device") else None)
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


class _FakeSigHetFStatComp:
    """Sig-het wrapper stand-in carrying the F-stat scorer contract.

    Reproduces the three things ``route_sighet_fstat`` depends on:

    * the STATEFUL reference stash -- ``setup_fstat_references`` folds the
      holder row it is given into ``_fstat`` ON the comp, and
      ``get_fstat_ll_wdm`` refuses to score without it -- so a route that
      set up on one comp object and scored through another fails loudly;
    * the replica contract (``chunked`` delegate + ``for_band_engine`` +
      the ``_g`` knob dict) so ``_device_local_gb_comp`` can rebuild it;
    * outputs derived ONLY from the stashed residual row and the candidate
      params -- FakeMultiShardACA seeds global row ``b`` with ``b + 1``, so
      routing to the wrong shard/row changes the numbers, while identically
      seeded 1-shard and n-shard holders give IDENTICAL results. The device
      each call ran under is recorded in ``calls`` (never in the output).
    """

    def __init__(self, chunked, v4_knots=128):
        self.chunked = chunked
        self.xp = chunked.xp
        self._g = dict(v4_knots=int(v4_knots))
        self._fstat = None
        self.calls = []

    @classmethod
    def for_band_engine(cls, chunked_comp, *, v4_knots=128, **_knobs):
        return cls(chunked_comp, v4_knots=v4_knots)

    def setup_fstat_references(self, params_ref_phys, wdm_holder,
                               data_index=0, noise_index=None,
                               assert_max_df0=None):
        assert len(wdm_holder.linear_data_arr) == 1, "comp must see one shard"
        assert len(wdm_holder) == wdm_holder.acs_total_entries
        refs = np.asarray(params_ref_phys, dtype=float)
        row_val = float(np.real(np.asarray(
            wdm_holder.linear_data_arr[0]).reshape(len(wdm_holder), -1)
        )[int(data_index), 0])
        self._fstat = dict(refs=refs.copy(), data_val=row_val)
        self.calls.append(dict(
            kind="setup_fstat", holder=wdm_holder,
            data_index=int(data_index),
            noise_index=None if noise_index is None else int(noise_index),
            device=self.xp.current_device,
            build_device=self.chunked._build_device,
            n_refs=len(refs)))
        return len(refs)

    def clear_fstat_references(self):
        self._fstat = None

    def get_fstat_ll_wdm(self, params, wdm_holder=None, data_index=None,
                         noise_index=None, fstat_mode=None, **kwargs):
        if self._fstat is None:
            raise RuntimeError("no reference stash; call "
                               "setup_fstat_references first.")
        # Concurrency hook (SigHetFStatMultiDevTest): a 2-party barrier
        # only passes if both device lanes score simultaneously.
        b = getattr(self, "barrier", None)
        if b is not None:
            b.wait()
        x = np.atleast_2d(np.asarray(params, dtype=float))
        di = (np.zeros(x.shape[0], dtype=int) if data_index is None
              else np.asarray(data_index, dtype=int))
        self.calls.append(dict(
            kind="fstat", n=int(x.shape[0]), data_index=di.copy(),
            device=self.xp.current_device,
            build_device=self.chunked._build_device))
        base = (self._fstat["data_val"] + 1e3 * x[:, 1]
                + self._fstat["refs"][di, 1])
        return (base[:, None] + np.arange(4)[None, :],
                base[:, None] + np.arange(10)[None, :])


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

        # ... and every shard's comp reports its own device, not shard 0's.
        # NOTE: a non-primary shard runs on a REPLICA, which is a separate
        # object with its own call log -- so ``comp.calls`` holds the primary
        # shard's call and nothing else. Checking only it would be nearly
        # vacuous; walk the replica cache for the rest.
        from lisatools.globalfit.stock.erebor.source_runtime import (
            _DEVICE_GB_COMP_REPLICAS,
        )
        primary = self.RoutedEngine._primary_device(comp, self.holder)
        by_device = {int(primary): comp}
        for (proto_id, cdev), (_proto, replica) in (
                _DEVICE_GB_COMP_REPLICAS.items()):
            if proto_id == id(comp):
                by_device[int(cdev)] = replica

        ran = set()
        for cdev, c in by_device.items():
            for call in c.calls:
                self.assertEqual(int(call["build_device"]), int(cdev))
                self.assertEqual(int(call["holder"].gpus[0]), int(cdev))
                ran.add(int(cdev))
        # every shard that owned a row actually ran, on its own comp
        self.assertEqual(
            ran, {int(d) for d in
                  np.asarray(self.holder.gpu_map)[self.data_index]})

    def test_fstat_all_rows_on_one_non_primary_shard(self):
        """The PRODUCTION F-stat call shape, and the one that actually fails
        without replicas.

        ``_fstat_NM`` scores every candidate against a single reference
        walker -- ``di = xp.full(n, walker_ref)`` -- so unlike the spread
        ``data_index`` above, exactly ONE shard is ever non-empty. When
        ``_fstat_reference_walker``'s argmax lands on a non-primary shard
        (roughly half of all proposals, since sharding splits walkers
        contiguously) that shard launches against comp buffers belonging to
        another device. A spread-index test cannot catch a partition bug that
        only shows up when a single shard is populated, so pin the real shape.
        """
        comp = self._fake_comp()
        gpu_map = np.asarray(self.holder.gpu_map)
        primary = self.RoutedEngine._primary_device(comp, self.holder)
        self.assertIsNotNone(
            primary, "fixture must emulate devices or this test is vacuous")

        foreign = [int(r) for r in range(self.NUM_ACS)
                   if int(gpu_map[r]) != int(primary)]
        self.assertTrue(foreign, "need a row owned by a non-primary shard")
        walker_ref = foreign[0]
        ref_dev = int(gpu_map[walker_ref])

        n = 6
        di = np.full(n, walker_ref, dtype=int)
        N, _M = self.RoutedEngine.route_fstat_ll(
            comp, "get_fstat_ll_wdm", self.holder, np.zeros((n, 3)),
            data_index=di, noise_index=di)

        # every row scored by a comp resident on the REFERENCE walker's
        # device, not on the prototype's
        np.testing.assert_array_equal(
            np.asarray(N)[:, 0], np.full(n, float(ref_dev)))
        self.assertNotEqual(ref_dev, int(primary))

        # The prototype itself must NOT have run: the work went to a replica,
        # which is a different object with its own call log. (This is also why
        # asserting over ``comp.calls`` alone is weak -- for a non-primary
        # shard it is empty by construction.)
        self.assertEqual(comp.calls, [])

        from lisatools.globalfit.stock.erebor.source_runtime import (
            _DEVICE_GB_COMP_REPLICAS,
        )
        replica = _DEVICE_GB_COMP_REPLICAS[(id(comp), ref_dev)][1]
        self.assertIsNot(replica, comp)
        # exactly one shard ran -- an empty shard must be skipped, not
        # launched with a zero-row batch
        self.assertEqual(len(replica.calls), 1)
        self.assertEqual(int(replica.calls[0]["build_device"]), ref_dev)
        self.assertEqual(int(replica.calls[0]["holder"].gpus[0]), ref_dev)

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


class SigHetFStatRouteTest(unittest.TestCase):
    """``route_sighet_fstat``: the stateful sig-het F-stat scorer route,
    PINNED single-device form.

    Unlike ``route_fstat_ll`` (stateless, one routed call per batch) the
    sig-het scorer keeps its reference stash ON the comp, so the pinned
    route keeps the WHOLE scorer -- reference builds and every score -- on
    the reference walker's shard on ONE stable device-local comp replica.
    Multi-GPU holders now default to the all-device fan-out
    (``SigHetFStatMultiDevTest``), so this class runs under the
    ``FSTAT_SIGHET_MULTIDEV=0`` kill-switch -- it IS the kill-switch
    coverage: these assertions passing under the knob proves the old
    pinned path is restored verbatim. Structural half on the CPU fakes;
    the numerical half (real CPU comp, bit-identical across holders) is
    ``SigHetFStatRouteRealCompTest``.
    """

    PER_BAND = (3, 8)
    NUM_ACS = 6
    # Tobs chosen so sighet_fstat_ref_margin_hz(128, TOBS) == 1.18e-6 --
    # the measured-anchor scale, keeping the default ref spacing realistic.
    TOBS = 7.776e6
    F0_LO = 1e-3

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbbands router not available: {exc}")
        from lisatools.sampling.fstat_gridfit import (
            sighet_fstat_ref_margin_hz)

        self.RoutedEngine = _RoutedBandEngine
        self.margin = sighet_fstat_ref_margin_hz(128, self.TOBS)
        # blocked layout: rows 0-2 on shard 0, rows 3-5 on shard 1
        self.holder = FakeMultiShardACA(
            self.PER_BAND, self.NUM_ACS, 2, layout="blocked")
        self.comp = _FakeSigHetFStatComp(
            FakeDeviceComp(self.holder.xp, tag="sighet-chunked"))

    def _route(self, comp, holder, walker_ref, **kw):
        # spacing pinned so a stray FSTAT_SIGHET_REF_SPACING_HZ in the test
        # environment cannot move the bucket layout the assertions rely on;
        # the kill-switch pinned so this class keeps testing the
        # single-device form (see the class docstring).
        with mock.patch.dict(os.environ, {"FSTAT_SIGHET_MULTIDEV": "0"}):
            return self.RoutedEngine.route_sighet_fstat(
                comp, holder, xp=holder.xp, Tobs=self.TOBS,
                f0_lims_hz=(self.F0_LO, self.F0_LO + 10.0 * self.margin),
                data_index=walker_ref, noise_index=walker_ref,
                spacing_hz=self.margin, **kw)

    def _candidates(self, fracs):
        """9-col rows at ``F0_LO + frac * margin`` (all inside the comb)."""
        p = np.zeros((len(fracs), 9))
        p[:, 0] = 1e-22
        p[:, 1] = self.F0_LO + np.asarray(fracs) * self.margin
        return p

    def _replica(self, comp, device):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            _DEVICE_GB_COMP_REPLICAS)

        return _DEVICE_GB_COMP_REPLICAS[(id(comp), int(device))][1]

    def test_routes_to_reference_shard_with_intra_index(self):
        """Walker 4 lives on shard 1 (blocked): setup + score must land on a
        device-1 replica, against the shard view, with the INTRA-shard row
        (4 -> 1), inside device 1's context."""
        call = self._route(self.comp, self.holder, walker_ref=4)
        self.holder.xp.device_log.clear()
        call(self._candidates([0.3, 2.6, 5.1]))

        # the prototype never ran; the work went to the device-1 replica
        self.assertEqual(self.comp.calls, [])
        replica = self._replica(self.comp, 1)
        self.assertIsNot(replica, self.comp)
        setup_calls = [c for c in replica.calls if c["kind"] == "setup_fstat"]
        score_calls = [c for c in replica.calls if c["kind"] == "fstat"]
        self.assertEqual(len(setup_calls), 1)
        self.assertEqual(len(score_calls), 1)
        setup = setup_calls[0]
        # intra-shard translation: global row 4 -> intra row 1 on shard 1
        self.assertEqual(setup["data_index"], 1)
        self.assertEqual(setup["noise_index"], 1)
        # ... against the shard VIEW (single-shard protocol, device 1)
        self.assertEqual(len(setup["holder"].linear_data_arr), 1)
        self.assertEqual(setup["holder"].gpus, [1])
        # ... inside device 1's context, on a device-1 replica
        for c in setup_calls + score_calls:
            self.assertEqual(int(c["device"]), 1)
            self.assertEqual(int(c["build_device"]), 1)
        self.assertEqual(set(self.holder.xp.device_log), {1})

    def test_results_identical_to_single_shard_passthrough(self):
        """Identically seeded 1-shard and 2-shard holders must give
        IDENTICAL (N, M) -- the route changes where the scorer runs, never
        what it computes."""
        fracs = [0.3, 1.4, 2.6, 5.1]
        single = FakeMultiShardACA(
            self.PER_BAND, self.NUM_ACS, 1, layout="blocked")
        comp_single = _FakeSigHetFStatComp(
            FakeDeviceComp(single.xp, tag="sighet-chunked"))
        call_s = self._route(comp_single, single, walker_ref=4)
        N_s, M_s = call_s(self._candidates(fracs))
        # passthrough: the comp saw the ORIGINAL holder and GLOBAL index
        setup = comp_single.calls[0]
        self.assertIs(setup["holder"], single)
        self.assertEqual(setup["data_index"], 4)

        call_m = self._route(self.comp, self.holder, walker_ref=4)
        N_m, M_m = call_m(self._candidates(fracs))
        np.testing.assert_array_equal(np.asarray(N_m), np.asarray(N_s))
        np.testing.assert_array_equal(np.asarray(M_m), np.asarray(M_s))

    def test_replica_stable_across_setup_and_score(self):
        """The stash lives ON the comp, so setup and every later score must
        resolve the SAME module-cached replica -- and a scorer rebuild must
        reuse it too (allocate-once rule)."""
        from lisatools.globalfit.stock.erebor.source_runtime import (
            _DEVICE_GB_COMP_REPLICAS)

        call = self._route(self.comp, self.holder, walker_ref=4)
        call(self._candidates([0.3]))
        call(self._candidates([1.4, 2.6]))
        replica = self._replica(self.comp, 1)
        # stash on the replica (where the scores read it), not the prototype
        self.assertIsNone(self.comp._fstat)
        self.assertIsNotNone(replica._fstat)
        # same block -> ONE reference build, both score batches through it
        kinds = [c["kind"] for c in replica.calls]
        self.assertEqual(kinds, ["setup_fstat", "fstat", "fstat"])
        # a second scorer build resolves the SAME replica object
        call2 = self._route(self.comp, self.holder, walker_ref=4)
        call2(self._candidates([0.3]))
        self.assertIs(self._replica(self.comp, 1), replica)
        mine = [k for k in _DEVICE_GB_COMP_REPLICAS if k[0] == id(self.comp)]
        self.assertEqual(len(mine), 1)

    def test_primary_shard_reuses_prototype_comp(self):
        """A reference walker on the prototype's own shard runs on the
        prototype itself -- no replica, nothing allocated."""
        from lisatools.globalfit.stock.erebor.source_runtime import (
            _DEVICE_GB_COMP_REPLICAS)

        call = self._route(self.comp, self.holder, walker_ref=1)
        call(self._candidates([0.3]))
        self.assertEqual([c["kind"] for c in self.comp.calls],
                         ["setup_fstat", "fstat"])
        self.assertEqual(self.comp.calls[0]["data_index"], 1)  # intra == global here
        self.assertEqual(
            [k for k in _DEVICE_GB_COMP_REPLICAS if k[0] == id(self.comp)],
            [])

    def test_multi_shard_requires_data_index(self):
        with self.assertRaises(ValueError):
            self.RoutedEngine.route_sighet_fstat(
                self.comp, self.holder, xp=self.holder.xp, Tobs=self.TOBS,
                f0_lims_hz=(self.F0_LO, self.F0_LO + 10.0 * self.margin),
                data_index=None)

    def test_rejects_slab_holders(self):
        self.holder.slab_min_f = np.zeros(self.NUM_ACS)
        try:
            with self.assertRaises(NotImplementedError):
                self._route(self.comp, self.holder, walker_ref=4)
        finally:
            del self.holder.slab_min_f


class SigHetFStatMultiDevTest(unittest.TestCase):
    """``route_sighet_fstat``'s default all-device fan-out.

    With >= 2 run devices every device gets its own lane -- a comp replica
    plus a private one-row holder carrying a copy of the reference walker's
    residual/inverse-PSD rows -- and each ``call_fstat`` batch is row-split
    into near-equal contiguous chunks scored concurrently. Structural +
    fake-numerical here; the real-comp bit-identity legs live in
    ``SigHetFStatRouteRealCompTest``.
    """

    PER_BAND = (3, 8)
    NUM_ACS = 6
    TOBS = 7.776e6
    F0_LO = 1e-3

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbbands router not available: {exc}")
        from lisatools.sampling.fstat_gridfit import (
            sighet_fstat_ref_margin_hz)

        self.RoutedEngine = _RoutedBandEngine
        self.margin = sighet_fstat_ref_margin_hz(128, self.TOBS)
        # blocked layout: rows 0-2 on shard/device 0, rows 3-5 on 1
        self.holder = FakeMultiShardACA(
            self.PER_BAND, self.NUM_ACS, 2, layout="blocked")
        self.comp = _FakeSigHetFStatComp(
            FakeDeviceComp(self.holder.xp, tag="sighet-chunked"))

    def _route(self, comp, holder, walker_ref, multidev="1"):
        with mock.patch.dict(os.environ,
                             {"FSTAT_SIGHET_MULTIDEV": multidev}):
            return self.RoutedEngine.route_sighet_fstat(
                comp, holder, xp=holder.xp, Tobs=self.TOBS,
                f0_lims_hz=(self.F0_LO, self.F0_LO + 10.0 * self.margin),
                data_index=walker_ref, noise_index=walker_ref,
                spacing_hz=self.margin)

    def _candidates(self, fracs):
        p = np.zeros((len(fracs), 9))
        p[:, 0] = 1e-22
        p[:, 1] = self.F0_LO + np.asarray(fracs) * self.margin
        return p

    def _replica(self, comp, device):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            _DEVICE_GB_COMP_REPLICAS)

        return _DEVICE_GB_COMP_REPLICAS[(id(comp), int(device))][1]

    def test_rows_split_near_equally_across_device_lanes(self):
        """5 rows over 2 lanes -> contiguous 2 + 3; lane 0 is the prototype
        comp (its own device), lane 1 the device-1 replica; every lane runs
        inside its own device context."""
        call = self._route(self.comp, self.holder, walker_ref=4)
        self.holder.xp.device_log.clear()
        call(self._candidates([0.3, 0.9, 1.4, 2.6, 5.1]))

        proto = [c for c in self.comp.calls if c["kind"] == "fstat"]
        replica = self._replica(self.comp, 1)
        rep = [c for c in replica.calls if c["kind"] == "fstat"]
        self.assertEqual([c["n"] for c in proto], [2])
        self.assertEqual([c["n"] for c in rep], [3])
        for c in proto:
            self.assertEqual(int(c["device"]), 0)
        for c in rep:
            self.assertEqual(int(c["device"]), 1)
        self.assertEqual(set(self.holder.xp.device_log), {0, 1})

    def test_lanes_have_private_one_row_replicas(self):
        """Each lane builds its OWN reference stash against its OWN one-row
        holder (the reference walker's copied row, scored as data_index 0)
        -- no lane touches another device's arrays or the live shard
        buffer."""
        call = self._route(self.comp, self.holder, walker_ref=4)
        call(self._candidates([0.3, 2.6]))
        replica = self._replica(self.comp, 1)
        proto_setup = [c for c in self.comp.calls
                       if c["kind"] == "setup_fstat"]
        rep_setup = [c for c in replica.calls if c["kind"] == "setup_fstat"]
        self.assertEqual(len(proto_setup), 1)
        self.assertEqual(len(rep_setup), 1)
        # per-lane stash, one per comp replica
        self.assertIsNotNone(self.comp._fstat)
        self.assertIsNotNone(replica._fstat)
        h0, h1 = proto_setup[0]["holder"], rep_setup[0]["holder"]
        self.assertIsNot(h0, h1)
        for h, dev, setup in ((h0, 0, proto_setup[0]),
                              (h1, 1, rep_setup[0])):
            self.assertEqual(setup["data_index"], 0)
            self.assertEqual(h.acs_total_entries, 1)
            self.assertEqual(len(h), 1)
            self.assertEqual(h.device, dev)
            self.assertEqual(h.gpus, [dev])
            # the one row is walker 4's (global seed b + 1 = 5), copied --
            # never the shard's live multi-row buffer
            row = np.asarray(h.linear_data_arr[0]).ravel()
            self.assertEqual(float(np.real(row[0])), 5.0)
            self.assertIsNot(h.linear_data_arr[0],
                             self.holder.linear_data_arr[1])
        # both lanes folded the SAME reference residual value
        self.assertEqual(self.comp._fstat["data_val"],
                         replica._fstat["data_val"])

    def test_multidev_bit_identical_to_pinned_and_single(self):
        """Determinism contract: each row is scored by exactly one lane
        with the single-device arithmetic and the merge is a permutation --
        so single-shard, pinned 2-shard and fan-out 2-shard all agree
        bit-for-bit."""
        fracs = [0.3, 0.9, 1.4, 2.6, 5.1]
        outs = {}
        for label, shards, multidev in (("single", 1, "1"),
                                        ("pinned", 2, "0"),
                                        ("multidev", 2, "1")):
            holder = FakeMultiShardACA(
                self.PER_BAND, self.NUM_ACS, shards, layout="blocked")
            comp = _FakeSigHetFStatComp(
                FakeDeviceComp(holder.xp, tag="sighet-chunked"))
            call = self._route(comp, holder, walker_ref=4,
                               multidev=multidev)
            outs[label] = tuple(np.asarray(a)
                                for a in call(self._candidates(fracs)))
        for label in ("pinned", "multidev"):
            np.testing.assert_array_equal(outs[label][0], outs["single"][0])
            np.testing.assert_array_equal(outs[label][1], outs["single"][1])

    def test_lanes_run_concurrently(self):
        """Both lanes must be inside the scoring kernel at once: a 2-party
        barrier passes only under concurrent dispatch (serial would time
        out and raise BrokenBarrierError)."""
        call = self._route(self.comp, self.holder, walker_ref=4)
        call(self._candidates([0.3, 2.6]))     # materialize both lanes
        replica = self._replica(self.comp, 1)
        barrier = threading.Barrier(2, timeout=10)
        self.comp.barrier = replica.barrier = barrier
        try:
            N, _M = call(self._candidates([0.3, 2.6]))
        finally:
            self.comp.barrier = replica.barrier = None
        self.assertEqual(np.asarray(N).shape, (2, 4))

    def test_kill_switch_restores_single_device_pin(self):
        """FSTAT_SIGHET_MULTIDEV=0: all rows on the reference walker's
        shard replica, scored against the SHARD VIEW with the intra-shard
        row -- the prototype never runs."""
        call = self._route(self.comp, self.holder, walker_ref=4,
                           multidev="0")
        call(self._candidates([0.3, 2.6, 5.1]))
        self.assertEqual(self.comp.calls, [])
        replica = self._replica(self.comp, 1)
        self.assertEqual(
            [c["n"] for c in replica.calls if c["kind"] == "fstat"], [3])
        setup = [c for c in replica.calls
                 if c["kind"] == "setup_fstat"][0]
        self.assertEqual(setup["holder"].acs_total_entries, 3)
        self.assertEqual(setup["data_index"], 1)

    def test_single_shard_passthrough_verbatim_no_pool(self):
        """CPU / single-GPU holders keep EXACTLY the current code path: the
        raw build_sighet_call_fstat closure against the original holder and
        global index, and no thread pool is ever spun up."""
        single = FakeMultiShardACA(
            self.PER_BAND, self.NUM_ACS, 1, layout="blocked")
        comp = _FakeSigHetFStatComp(
            FakeDeviceComp(single.xp, tag="sighet-chunked"))
        call = self._route(comp, single, walker_ref=4)
        self.assertIn("build_sighet_call_fstat", call.__qualname__)
        call(self._candidates([0.3, 1.4]))
        setup = comp.calls[0]
        self.assertIs(setup["holder"], single)
        self.assertEqual(setup["data_index"], 4)
        self.assertIsNone(single._thread_pool)


class RouterThreadedDispatchTest(unittest.TestCase):
    """``GB_ROUTER_THREADED``: opt-in concurrent per-shard dispatch.

    Default OFF -- serial launch-sync-launch is the de-facto safety net
    while the 2-GPU incremental-ll-drift investigation is open. With the
    knob on and multiple shards populated, per-shard work runs on host
    threads, each inside its own device context, writing pre-allocated
    result slots -- so knob-on results are bit-identical to serial.
    """

    PER_BAND = (3, 8)
    NUM_ACS = 6

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbbands router not available: {exc}")
        self.RoutedEngine = _RoutedBandEngine
        self.holder = FakeMultiShardACA(
            self.PER_BAND, self.NUM_ACS, 2, layout="blocked")
        self.data_index = np.array([4, 0, 5, 2, 1])
        self.params = np.arange(len(self.data_index) * 9, dtype=float
                                ).reshape(len(self.data_index), 9)

    def _factory_router(self):
        """Router whose shards resolve DISTINCT engines (the multi-GPU
        production shape -- and the precondition for threaded dispatch)."""
        engines = {}

        def _factory(device, primary):
            engines[int(device)] = _StubEngine()
            return engines[int(device)]

        primary = _StubEngine()
        return self.RoutedEngine(primary, engine_factory=_factory), \
            primary, engines

    def _get_ll(self, router, threaded):
        with mock.patch.dict(os.environ,
                             {"GB_ROUTER_THREADED": threaded}):
            return np.asarray(router.get_ll(
                self.holder, self.params,
                data_index=self.data_index, noise_index=self.data_index,
                N_vals=None, waveform_kwargs={}))

    def test_knob_on_runs_shards_concurrently_in_own_contexts(self):
        router, primary, engines = self._factory_router()
        self._get_ll(router, "0")              # materialize the replica
        primary.calls.clear()
        engines[1].calls.clear()
        barrier = threading.Barrier(2, timeout=10)
        primary.barrier = engines[1].barrier = barrier
        try:
            self._get_ll(router, "1")
        finally:
            primary.barrier = engines[1].barrier = None
        pc = [c for c in primary.calls if c["kind"] == "get_ll"]
        rc = [c for c in engines[1].calls if c["kind"] == "get_ll"]
        self.assertEqual((len(pc), len(rc)), (1, 1))
        # distinct worker threads (guaranteed by the barrier: one thread
        # cannot be inside both calls at once), each in its own context
        self.assertNotEqual(pc[0]["thread"], rc[0]["thread"])
        self.assertEqual(int(pc[0]["device"]), 0)
        self.assertEqual(int(rc[0]["device"]), 1)

    def test_knob_on_results_bit_identical_to_serial(self):
        r_off, _, _ = self._factory_router()
        ll_off = self._get_ll(r_off, "0")
        d_off = np.asarray(r_off.d_h_out)
        h_off = np.asarray(r_off.h_h_out)
        r_on, _, _ = self._factory_router()
        ll_on = self._get_ll(r_on, "1")
        np.testing.assert_array_equal(ll_on, ll_off)
        np.testing.assert_array_equal(np.asarray(r_on.d_h_out), d_off)
        np.testing.assert_array_equal(np.asarray(r_on.h_h_out), h_off)

    def test_knob_on_classmethod_routes_bit_identical(self):
        outs = {}
        for threaded in ("0", "1"):
            comp = FakeDeviceComp(self.holder.xp, tag="wdm")
            with mock.patch.dict(os.environ,
                                 {"GB_ROUTER_THREADED": threaded}):
                N, M = self.RoutedEngine.route_fstat_ll(
                    comp, "get_fstat_ll_wdm", self.holder, self.params,
                    data_index=self.data_index,
                    noise_index=self.data_index)
                info = self.RoutedEngine.route_information_matrix(
                    comp, self.holder, self.params[:, :3], inds=[0, 1, 2],
                    noise_index=self.data_index)
            outs[threaded] = tuple(np.asarray(a) for a in (N, M, info))
        for a, b in zip(outs["0"], outs["1"]):
            np.testing.assert_array_equal(b, a)

    def test_knob_off_creates_no_thread_pool(self):
        router, _, _ = self._factory_router()
        self._get_ll(router, "0")
        self.assertIsNone(self.holder._thread_pool)

    def test_shared_engine_forces_serial_despite_knob(self):
        """Shards resolving to ONE engine (no engine_factory) would race
        its output attrs -- duplicate state ids must force serial, without
        ever touching the thread pool."""
        engine = _StubEngine()
        router = self.RoutedEngine(engine)
        ll = self._get_ll(router, "1")
        self.assertIsNone(self.holder._thread_pool)
        dev = self.holder.gpu_map[self.data_index].astype(float)
        intra = np.empty(self.NUM_ACS, dtype=int)
        for rr in self.holder.gpu_splits:
            intra[rr] = np.arange(len(rr))
        np.testing.assert_array_equal(
            ll, 1000.0 * dev + intra[self.data_index].astype(float))


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


@unittest.skipUnless(
    _have_gbgpu_comps(),
    "requires gbgpu.gbcomps / gbgpu.gbsignalhetcomputations",
)
class SigHetFStatRouteRealCompTest(unittest.TestCase):
    """Numerical half of the sig-het F-stat route proof, on the REAL comp.

    The structural tests above prove WHERE the scorer runs; this proves the
    route changes nothing about WHAT it computes: setup refs + score a
    batch through ``route_sighet_fstat`` on a fake 2-shard holder vs a
    1-shard holder seeded identically (per GLOBAL row id), expecting
    bit-identical (N, M). The 2-shard run's reference walker sits on the
    NON-primary shard on purpose, so it exercises the full replica rebuild
    (``for_band_engine`` around a rebuilt chunked delegate) -- the path a
    cluster run takes about half the time.

    Small grid on purpose (the GBCompReplicaContractTest scale); the cost
    is dominated by the orbit configuration of the comp + its replica.
    """

    NUM_ACS = 4
    WALKER_REF = 3          # blocked 4/2 -> shard 1, intra 1 (non-primary)

    @classmethod
    def setUpClass(cls):
        from gbgpu.gbcomps import GBWDMComputations
        from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
        from lisatools.domains import WDMSettings

        chunked = GBWDMComputations(
            WDMSettings(Nf=32, Nt=64, dt=15.0, force_backend="cpu"),
            t_ref=0.0, Nt_sub=16, n_pad=2, N_sparse=64,
            tdi_config="1st generation", force_backend="cpu")
        cls.sig = GBSignalHetComputations.for_band_engine(
            chunked, nt_layer=8, n_sparse_fd=128, v3_n_nodes=64,
            v4_knots=128, v4_band=16)

    def _seeded_holder(self, num_shards):
        """Holder whose per-row slabs depend only on the GLOBAL row id, so
        1-shard and 2-shard instances hold byte-identical rows."""
        g = self.sig._g
        NfA, NtA = int(g["Nf_active"]), int(g["Nt_active"])
        holder = FakeMultiShardACA(
            (3, NfA * NtA), self.NUM_ACS, num_shards, layout="blocked")
        rng = np.random.default_rng(11)
        data = rng.normal(size=(self.NUM_ACS, 3, NfA, NtA)) * 1e-22
        invc = np.zeros((self.NUM_ACS, 3, 3, NfA, NtA))
        for c in range(3):
            invc[:, c, c] = 1e44
        # overwrite the fake's constant-seeded buffers with the layouts the
        # real comp consumes: (rows, 3, NfA, NtA) data and (rows, 3, 3,
        # NfA, NtA) XYZ cross-channel invC, flat per shard.
        holder.linear_data_arr = [
            np.ascontiguousarray(data[rows]).ravel()
            for rows in holder.gpu_splits]
        holder.linear_psd_arr = [
            np.ascontiguousarray(invc[rows]).ravel()
            for rows in holder.gpu_splits]
        return holder

    def _candidates(self, f0_lo, margin, n=6):
        rng = np.random.default_rng(5)
        cands = np.zeros((n, 9))
        cands[:, 0] = 1e-22
        cands[:, 1] = f0_lo + rng.uniform(0.0, 3.0, n) * margin
        cands[:, 2] = 1e-17
        cands[:, 5] = rng.uniform(0, np.pi, n)
        cands[:, 6] = rng.uniform(0, np.pi, n)
        cands[:, 7] = rng.uniform(0, 2 * np.pi, n)
        cands[:, 8] = np.arcsin(rng.uniform(-1.0, 1.0, n))
        return cands

    def test_two_shard_route_bit_identical_to_single_shard(self):
        """Three legs on the REAL comp: single-shard passthrough, pinned
        2-shard (FSTAT_SIGHET_MULTIDEV=0) and the default all-device
        fan-out -- all must agree bit-for-bit (the fan-out's determinism
        contract: one lane per row, single-device arithmetic, permutation
        merge)."""
        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        from lisatools.sampling.fstat_gridfit import (
            sighet_fstat_ref_margin_hz)

        g = self.sig._g
        Tobs = float(g["Tobs"])
        margin = sighet_fstat_ref_margin_hz(int(g["v4_knots"]), Tobs)
        f0_lo = (g["ind_min_f"] + g["Nf_active"] // 2 + 0.3) * g["layer_df"]
        cands = self._candidates(f0_lo, margin)

        results = {}
        for label, shards, multidev in (("single", 1, "1"),
                                        ("pinned", 2, "0"),
                                        ("multidev", 2, "1")):
            self.sig.clear_fstat_references()
            with mock.patch.dict(os.environ,
                                 {"FSTAT_SIGHET_MULTIDEV": multidev}):
                call = _RoutedBandEngine.route_sighet_fstat(
                    self.sig, self._seeded_holder(shards), xp=np, Tobs=Tobs,
                    f0_lims_hz=(f0_lo, f0_lo + 3.0 * margin),
                    data_index=self.WALKER_REF, noise_index=self.WALKER_REF,
                    spacing_hz=margin)
            results[label] = tuple(np.asarray(a) for a in call(cands))

        N1, M1 = results["single"]
        self.assertTrue(np.all(np.isfinite(N1)) and np.all(np.isfinite(M1)))
        for label in ("pinned", "multidev"):
            np.testing.assert_array_equal(results[label][0], N1)
            np.testing.assert_array_equal(results[label][1], M1)

    def test_router_threaded_chunked_fstat_bit_identical(self):
        """GB_ROUTER_THREADED on the REAL chunked comp: the routed
        stateless F-stat (spread across both shards) must be bit-identical
        knob-on vs knob-off."""
        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        from lisatools.sampling.fstat_gridfit import (
            sighet_fstat_ref_margin_hz)

        g = self.sig._g
        margin = sighet_fstat_ref_margin_hz(int(g["v4_knots"]),
                                            float(g["Tobs"]))
        f0_lo = (g["ind_min_f"] + g["Nf_active"] // 2 + 0.3) * g["layer_df"]
        cands = self._candidates(f0_lo, margin)
        holder = self._seeded_holder(2)
        di = np.array([0, 3, 1, 2, 3, 0])  # rows on BOTH shards

        outs = {}
        for threaded in ("0", "1"):
            with mock.patch.dict(os.environ,
                                 {"GB_ROUTER_THREADED": threaded}):
                N, M = _RoutedBandEngine.route_fstat_ll(
                    self.sig.chunked, "get_fstat_ll_wdm", holder, cands,
                    data_index=di, noise_index=di)
            outs[threaded] = (np.asarray(N).copy(), np.asarray(M).copy())
        self.assertTrue(np.all(np.isfinite(outs["0"][0])))
        np.testing.assert_array_equal(outs["1"][0], outs["0"][0])
        np.testing.assert_array_equal(outs["1"][1], outs["0"][1])


if __name__ == "__main__":
    unittest.main()
