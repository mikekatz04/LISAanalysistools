"""No cross-device reads in the WDM transform or the SOBBH Gram build.

The 6mo relaunch (2026-10-09) died with an illegal memory access that
``CUDA_LAUNCH_BLOCKING=1`` made disappear: a race. The log carried cupy's
"The device where the array resides (0) is different from the current device
(1)" warnings from two places: ``FDSignal.wdmtransform`` reading the WDM
settings' window and fold map (cached on GPU 0) for a GPU-1 signal, and the
SOBBH Gram eigen table multiplying templates on the comp's device by the
walker container's ``invC`` on the walker's device. Peer reads are unordered
against the owning device's stream.

There is no second GPU here, so a fake two-device ``xp`` (numpy arrays
tagged with a device id; ``asnumpy`` counts host hops) checks the helpers,
and the WDM transform's wiring is checked by patching them.

The 9mo job 751 (2026-10-09) logged the same warning at three more sites,
covered at the end of this file: the warm-start proposal's table cache
(keyed by a literal, so GPU 0 owned it), the GB cap-cell tables (built once
on GPU 0, indexed with GPU-1 band indices) and, again, the SOBBH Gram (its
guard was handed the move's eryn ``xp``, which is numpy on GPU runs).
"""
from __future__ import annotations

import copy
import os
import pickle
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.sensitivity import SensitivityMatrixBase
from lisatools.utils import device as D


class _Arr(np.ndarray):
    """A numpy array that says which (fake) GPU it lives on."""

    def __array_finalize__(self, obj):
        self._dev = getattr(obj, "_dev", 0)

    @property
    def device(self):
        return SimpleNamespace(id=self._dev)


def _on(a, dev):
    out = np.array(a).view(_Arr)
    out._dev = int(dev)
    return out


class _TwoGPU:
    """Just the cupy surface ``lisatools.utils.device`` touches."""

    def __init__(self):
        self.current = 0
        self.hops = 0
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda: self.current),
            Device=self._device)

    @contextmanager
    def _device(self, i):
        prev, self.current = self.current, int(i)
        try:
            yield
        finally:
            self.current = prev

    def asnumpy(self, a):
        self.hops += 1
        return np.array(a).view(np.ndarray)

    def asarray(self, a):
        return _on(a, self.current)


class ToCurrentDeviceCachedTest(unittest.TestCase):
    def setUp(self):
        self.xp = _TwoGPU()
        self.owner = SimpleNamespace()
        self.src = (_on(np.arange(4.0), 0), _on(np.ones(3), 0))

    def test_a_foreign_array_is_copied_once_per_device(self):
        self.xp.current = 1
        a, b = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        self.assertEqual((a.device.id, b.device.id), (1, 1))
        np.testing.assert_array_equal(a, self.src[0])
        self.assertEqual(self.xp.hops, 2)
        again = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        self.assertIs(again[0], a)
        self.assertEqual(self.xp.hops, 2)          # cached: no new host hop

    def test_a_rebuilt_source_is_copied_again(self):
        self.xp.current = 1
        a, _ = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        rebuilt = (_on(np.arange(4.0) + 1, 0), self.src[1])
        a2, _ = D.to_current_device_cached(self.xp, self.owner, "t", rebuilt)
        self.assertIsNot(a2, a)
        np.testing.assert_array_equal(a2, rebuilt[0])

    def test_home_device_reads_are_the_source_objects(self):
        out = D.to_current_device_cached(self.xp, self.owner, "t", self.src)
        self.assertIs(out[0], self.src[0])
        self.assertEqual(self.xp.hops, 0)

    def test_numpy_is_a_strict_noop(self):
        src = (np.arange(3.0),)
        out = D.to_current_device_cached(np, self.owner, "t", src)
        self.assertIs(out[0], src[0])
        self.assertFalse(hasattr(self.owner, "_device_copy_cache"))


def _sens(xp_dev, dirty=False):
    """A real ``SensitivityMatrixBase`` (no basis needed) on fake GPU ``xp_dev``."""
    sm = object.__new__(SensitivityMatrixBase)
    sm._sens_mat = _on(np.full((2, 2, 5), 2.0), xp_dev)
    sm._invC = _on(np.full((2, 2, 5), 0.5), xp_dev)
    sm._detC = _on(np.full(5, 4.0), xp_dev)
    sm.do_inv_det = True
    sm._inv_det_dirty = dirty
    sm.tag = "shared"
    return sm


class SensitivityToCurrentDeviceTest(unittest.TestCase):
    def setUp(self):
        self.xp = _TwoGPU()

    def test_foreign_psd_becomes_a_same_class_clone_on_this_device(self):
        sm = _sens(1)
        out = D.sensitivity_to_current_device(self.xp, sm)
        # inner_product type-checks its psd: a proxy would be rebuilt as a
        # brand-new SensitivityMatrix around the proxy
        self.assertIs(type(out), SensitivityMatrixBase)
        self.assertEqual((out.invC.device.id, out.detC.device.id), (0, 0))
        np.testing.assert_array_equal(out.invC, sm.invC)
        self.assertIs(out.sens_mat, sm.sens_mat)        # shared, shape only
        self.assertEqual(out.tag, "shared")
        self.assertEqual(sm.invC.device.id, 1)          # the original is untouched

    def test_a_dirty_psd_is_inverted_on_its_own_device(self):
        sm = _sens(1, dirty=True)
        seen = []

        def setup():
            seen.append(self.xp.current)
            sm._invC = _on(np.full((2, 2, 5), 0.25), 1)
            sm._detC = _on(np.full(5, 16.0), 1)
            sm._inv_det_dirty = False

        sm._setup_det_and_inv = setup
        out = D.sensitivity_to_current_device(self.xp, sm)
        self.assertEqual(seen, [1])
        np.testing.assert_array_equal(out.invC, 0.25)
        self.assertFalse(out._inv_det_dirty)

    def test_the_decision_follows_invC_not_sens_mat(self):
        # 6mo job 748: the walker matrix held sens_mat on the home device
        # (== current) while invC / detC lived on the walker's GPU; keying on
        # sens_mat let that through and diagnostic.py read invC across devices
        sm = _sens(1)
        sm._sens_mat = _on(np.asarray(sm._sens_mat), 0)
        out = D.sensitivity_to_current_device(self.xp, sm)
        self.assertIsNot(out, sm)
        self.assertEqual((out.invC.device.id, out.detC.device.id), (0, 0))
        # and the reverse: everything inner_product reads is already local
        sm2 = _sens(0)
        sm2._sens_mat = _on(np.asarray(sm2._sens_mat), 1)
        self.assertIs(D.sensitivity_to_current_device(self.xp, sm2), sm2)

    def test_same_device_and_numpy_return_the_object_itself(self):
        sm = _sens(0)
        self.assertIs(D.sensitivity_to_current_device(self.xp, sm), sm)
        self.assertIs(D.sensitivity_to_current_device(np, sm), sm)
        self.assertEqual(self.xp.hops, 0)


class WDMTransformWiringTest(unittest.TestCase):
    """``wdmtransform`` runs in the signal's device context and reads the
    settings' window / fold map through the per-device cache."""

    def _signal(self):
        from lisatools.domains import FDSettings, FDSignal, WDMSettings

        wdm = WDMSettings(Nf=128, Nt=64, dt=5.0, min_freq=3e-4, max_freq=8e-3,
                          force_backend="cpu")
        f = np.fft.rfftfreq(wdm.N, wdm.data_dt)
        fd = FDSettings(f.shape[0], float(f[1] - f[0]), force_backend="cpu")
        rng = np.random.default_rng(3)
        arr = rng.normal(size=(3, fd.N)) + 1j * rng.normal(size=(3, fd.N))
        return FDSignal(arr, fd), wdm

    def test_settings_arrays_go_through_the_per_device_cache(self):
        sig, wdm = self._signal()
        ref = sig.wdmtransform(settings=wdm)
        tags = []
        real = D.to_current_device_cached

        def spy(xp, owner, tag, arrays):
            self.assertIs(owner, wdm)
            tags.append(tag)
            return real(xp, owner, tag, arrays)

        with mock.patch.object(D, "to_current_device_cached", side_effect=spy), \
                mock.patch.object(D, "device_context",
                                  side_effect=D.device_context) as ctx:
            out = sig.wdmtransform(settings=wdm)
        self.assertEqual(sorted(tags), ["fold_shift_map", "window"])
        ctx.assert_called_once()
        np.testing.assert_array_equal(np.asarray(out.arr), np.asarray(ref.arr))

    def test_the_transform_uses_what_the_cache_returns(self):
        # a cache that hands back a doubled window must change the output:
        # the fold map / window are not read from the settings behind its back
        sig, wdm = self._signal()
        ref = np.asarray(sig.wdmtransform(settings=wdm).arr)

        def doubled(xp, owner, tag, arrays):
            arrays = tuple(arrays)
            return (2.0 * arrays[0],) if tag == "window" else arrays

        with mock.patch.object(D, "to_current_device_cached", side_effect=doubled):
            out = np.asarray(sig.wdmtransform(settings=wdm).arr)
        np.testing.assert_allclose(out, 2.0 * ref, rtol=1e-12, atol=1e-30)

    def test_copies_and_pickles_drop_the_device_copies(self):
        _, wdm = self._signal()
        wdm._device_copy_cache = {("window", 1): ((), ())}
        for clone in (copy.deepcopy(wdm), pickle.loads(pickle.dumps(wdm))):
            self.assertFalse(hasattr(clone, "_device_copy_cache"))


# ---------------------------------------------------------------------------
# 9mo job 751: warm-start tables, GB cap-cell tables, the SOBBH Gram again
# ---------------------------------------------------------------------------


def _check_current(xp, *arrays):
    """Fail like a GPU without peer access: cupy only warns, then reads
    across the link (the 10-08 race)."""
    for a in arrays:
        if isinstance(a, _Arr) and a._dev != xp.current:
            raise AssertionError(
                f"cross-device read: array on device {a._dev}, "
                f"current device {xp.current}")


class _StrictArr(_Arr):
    """An ``_Arr`` whose indexing and ufuncs refuse any operand that is not
    on ``xp.current`` (``xp`` = the fake it was made by)."""

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._xp = getattr(obj, "_xp", None)

    def __getitem__(self, idx):
        _check_current(self._xp, self, *(idx if isinstance(idx, tuple) else (idx,)))
        return super().__getitem__(idx)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _check_current(self._xp, *inputs)
        plain = [a.view(np.ndarray) if isinstance(a, np.ndarray) else a for a in inputs]
        res = getattr(ufunc, method)(*plain, **kwargs)
        return self._xp.wrap(res) if isinstance(res, np.ndarray) else res


class _StrictTwoGPU(_TwoGPU):
    """``_TwoGPU`` running numpy's functions the way cupy does: on the
    CURRENT device, every device argument checked. ``asarray`` hands a
    device array back as is, wherever it lives (cupy moves nothing)."""

    def wrap(self, a, dev=None):
        out = np.array(a).view(_StrictArr)
        out._dev = self.current if dev is None else int(dev)
        out._xp = self
        return out

    def asarray(self, a):
        return a if isinstance(a, _Arr) else self.wrap(a)

    def __getattr__(self, name):
        fn = getattr(np, name)
        if not callable(fn) or isinstance(fn, type):
            return fn

        def call(*args, **kwargs):
            _check_current(self, *args)
            plain = [a.view(np.ndarray) if isinstance(a, np.ndarray) else a
                     for a in args]
            res = fn(*plain, **kwargs)
            return self.wrap(res) if isinstance(res, np.ndarray) else res

        return call


class WarmStartTablesPerDeviceTest(unittest.TestCase):
    """``WarmStartComponents._t`` keeps one table copy PER DEVICE (job 751:
    keyed by the literal "dev", the first caller's GPU 0 owned the tables
    and GPU-1 proposals read them across devices)."""

    def _components(self):
        from lisatools.globalfit.warmstart.proposal import WarmStartComponents

        means = np.array([
            [8.0, 2.0, 0.60, 3.00, 0.30, 1.20, 4.00, 0.20, 0.05],
            [5.0, 5.0, 0.70, 0.05, 0.00, 0.50, 5.50, 0.60, 0.20],
        ])
        sig = np.array([0.5, 2.0e-5, 0.01, 0.30, 0.05, 0.15, 0.05, 0.05, 0.02])
        covs = np.stack([np.diag(sig ** 2)] * 2)
        return WarmStartComponents(means, covs, np.array([0.6, 0.9]),
                                   new_tobs=7776000.0)

    def test_each_device_gets_its_own_copy_from_the_masters(self):
        ws = self._components()
        xp = _TwoGPU()
        t0 = ws._t(xp)
        xp.current = 1
        t1 = ws._t(xp)
        self.assertEqual({a.device.id for a in t0.values()}, {0})
        self.assertEqual({a.device.id for a in t1.values()}, {1})
        for k, master in ws._tables.items():
            np.testing.assert_array_equal(np.asarray(t1[k]), master)
        self.assertEqual(xp.hops, 0)          # uploaded from the host masters
        # cached per device, each served on its own device
        xp.current = 0
        self.assertIs(ws._t(xp), t0)
        xp.current = 1
        self.assertIs(ws._t(xp), t1)
        self.assertEqual(sorted(ws._dev_cache), [0, 1])
        self.assertIs(ws._t(np), ws._tables)  # numpy: the masters themselves

    def test_copies_and_pickles_drop_the_device_tables(self):
        ws = self._components()
        xp = _TwoGPU()
        ws._t(xp)
        xp.current = 1
        ws._t(xp)
        for clone in (copy.deepcopy(ws), pickle.loads(pickle.dumps(ws))):
            self.assertEqual(clone._dev_cache, {})
            self.assertIs(clone._t(np), clone._tables)


class CapTablesPerDeviceTest(unittest.TestCase):
    """The GB cap-cell lookups read their tables on the CURRENT device (job
    751: gbspecialstretch ``_cap_cell_index`` gathered GPU-0 tables with
    GPU-1 band indices)."""

    BE = np.array([5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
    K = 4
    F0 = np.array([5.1e-3, 6.26e-3, 7.49e-3, 8.99e-3, 6.0e-3, 7.76e-3])

    def _shim(self, on, stagger=False, overlap=0.0):
        """The real class with only the cap attributes, built the ctor's way
        (``on`` places each table, e.g. on the home GPU 0)."""
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        from lisatools.globalfit.state import make_cap_edge_extensions, make_cap_edges

        be, k = self.BE, self.K
        mv = object.__new__(GBSpecialBase)
        mv.cap_divisor = k
        mv.cap_stagger = stagger
        mv.num_bands = len(be) - 1
        mv.num_cap_cells = mv.num_bands * k
        ce = make_cap_edges(be, k, stagger=stagger)
        mv.band_edges = on(be)
        mv.cap_edges = on(ce)
        mv._cap_band_lo = on(be[:-1])
        mv._cap_band_step = on((be[1:] - be[:-1]) / k)
        mv.cap_overlap_frac = overlap
        if overlap > 0:
            mv._cap_edge_ext = on(make_cap_edge_extensions(be, ce, k, overlap))
        return mv

    def _bands(self, f0):
        return np.searchsorted(self.BE, f0, side="right") - 1

    def _on_device(self, mv, xp, f0, fn, **kw):
        """``fn`` (a bound cap lookup) on the current device's band / f0 rows."""
        from lisatools.globalfit.moves import gbspecialstretch as G

        b, f = xp.wrap(self._bands(f0)), xp.wrap(f0)
        with mock.patch.object(
                G, "get_array_module",
                side_effect=lambda a: xp if isinstance(a, _Arr) else np):
            return fn(b, f, **kw)

    def test_cell_index_on_a_non_home_device(self):
        for stagger in (False, True):
            xp = _StrictTwoGPU()
            mv = self._shim(lambda a: xp.wrap(a, 0), stagger=stagger)
            ref = self._shim(np.asarray, stagger=stagger)._np_cap_cells(
                self.F0, self._bands(self.F0), self.BE)
            xp.current = 1
            out = self._on_device(mv, xp, self.F0, mv._cap_cell_index)
            np.testing.assert_array_equal(np.asarray(out), ref)
            self.assertEqual(out._dev, 1)
            hops = xp.hops
            self._on_device(mv, xp, self.F0, mv._cap_cell_index)
            self.assertEqual(xp.hops, hops)       # cached for device 1

    def test_destination_band_resolution_reads_local_band_edges(self):
        xp = _StrictTwoGPU()
        mv = self._shim(lambda a: xp.wrap(a, 0), stagger=True)
        ref = self._shim(np.asarray, stagger=True)._np_cap_cells(
            self.F0, self._bands(self.F0), self.BE)
        xp.current = 1
        # filed in band 0, resolved from f0
        b, f = xp.wrap(np.zeros(self.F0.size, dtype=int)), xp.wrap(self.F0)
        from lisatools.globalfit.moves import gbspecialstretch as G

        with mock.patch.object(
                G, "get_array_module",
                side_effect=lambda a: xp if isinstance(a, _Arr) else np):
            out = mv._cap_cell_index(b, f, resolve_band=True)
        np.testing.assert_array_equal(np.asarray(out), ref)

    def test_overlap_members_on_a_non_home_device(self):
        xp = _StrictTwoGPU()
        mv = self._shim(lambda a: xp.wrap(a, 0), overlap=0.25)
        host = self._shim(np.asarray, overlap=0.25)
        bands = self._bands(self.F0)
        ref = host._np_cap_members(self.F0, bands, self.BE)
        xp.current = 1
        out = self._on_device(mv, xp, self.F0, mv._cap_cell_members)
        for got, want in zip(out, ref):
            np.testing.assert_array_equal(np.asarray(got), want)

    def test_home_device_reads_the_masters(self):
        xp = _StrictTwoGPU()
        mv = self._shim(lambda a: xp.wrap(a, 0))
        self._on_device(mv, xp, self.F0, mv._cap_cell_index)
        self.assertIs(mv._cap_table(xp, "_cap_band_lo"), mv._cap_band_lo)
        self.assertEqual(xp.hops, 0)

    def test_copies_and_pickles_drop_the_device_tables(self):
        mv = self._shim(np.asarray)
        mv._device_copy_cache = {("_cap_band_lo", 1): ((), ())}
        for clone in (copy.deepcopy(mv), pickle.loads(pickle.dumps(mv))):
            self.assertFalse(hasattr(clone, "_device_copy_cache"))
            np.testing.assert_array_equal(clone._cap_band_lo, mv._cap_band_lo)


class GramPsdOnTemplatesDeviceTest(unittest.TestCase):
    """The Gram eigen table scores its templates against the walker PSD on
    the TEMPLATES' device (job 751: diagnostic.py:248/294 on two ranks; the
    10-08 guard was handed the move's eryn ``xp``, numpy on GPU runs)."""

    def test_the_walker_psd_reaches_inner_product_on_the_templates_device(self):
        from eryn.priors import ProbDistContainer, UniformDistribution

        import lisatools.diagnostic as diag
        from lisatools.globalfit.moves import addremovemove as arm
        from lisatools.globalfit.moves.sobbhspecialmove import SOBBHChunkedLikeMove

        xp = _TwoGPU()                 # current device 0: the comp's
        nd = 2
        move = object.__new__(SOBBHChunkedLikeMove)
        # the stock source-move builders pass no use_gpu: eryn's default
        move._use_gpu = False
        self.assertIs(move.xp, np)
        move.branch_name = "sobbh"
        move.priors = {"sobbh": ProbDistContainer(
            {i: UniformDistribution(-1.0, 1.0) for i in range(nd)})}
        move.eigen_gram_target = 0.0   # one template pass
        move.eigen_gram_eps_rel = 1e-3
        move.waveform_like_kwargs = {}
        psd = _sens(1)                 # the walker's PSD, on its shard GPU
        ac = SimpleNamespace(_slice_to_template=lambda h: (None, None, psd))
        # a single-shard holder with no GPU list: the move's shard context
        # (SOBBHSingleShardDeviceTest) is then the null context, so the
        # fake templates' device 0 stays "current" for this test
        move.acs = SimpleNamespace(acs=np.array([ac], dtype=object),
                                   gpus=None, linear_data_arr=[object()])
        move._to_phys = lambda X: X
        rng = np.random.default_rng(1)
        move._gram_templates = lambda X, w: (_on(rng.normal(size=(2 * nd + 1, 2, 5)), 0),
                                             "box")
        seen = []

        def ip(a, b, psd=None, **kw):
            seen.append(psd.invC.device.id)
            return 1.0

        with mock.patch.object(arm, "get_array_module",
                               side_effect=lambda a: xp if isinstance(a, _Arr) else np), \
                mock.patch.object(arm, "WDMSignal",
                                  side_effect=lambda arr, box: SimpleNamespace(
                                      arr=arr, settings=box)), \
                mock.patch.object(diag, "inner_product", side_effect=ip):
            info = move._gram_info(np.zeros(nd), 0, np.full(nd, 2.0))
        self.assertEqual(info.shape, (nd, nd))
        self.assertEqual(len(seen), nd * (nd + 1) // 2)
        self.assertEqual(set(seen), {0})
        self.assertEqual(psd.invC.device.id, 1)     # the walker's matrix is untouched


class _TwoGPUSync(_TwoGPU):
    """``_TwoGPU`` plus the ``deviceSynchronize`` the scorer's timing calls."""

    def __init__(self):
        super().__init__()
        self.cuda.runtime.deviceSynchronize = lambda: None


class _RecordingComp:
    """A chunked/lookup comp that records the CURRENT device at each call."""

    def __init__(self, xp, n):
        self.xp = xp
        self.calls = []
        self.d_h_out = np.ones(n)
        self.h_h_out = np.ones(n)

    def get_ll_wdm(self, params, holder, **kw):
        self.calls.append(("get_ll_wdm", self.xp.current))
        return np.zeros(np.shape(params)[0])

    def fill_global_wdm(self, params, holder, **kw):
        self.calls.append(("fill_global_wdm", self.xp.current))


class SOBBHSingleShardDeviceTest(unittest.TestCase):
    """Every SOBBH pass enters the device that owns the walker's shard, as
    the MBH/EMRI moves do (job 751, 2026-10-10: a rank whose walker lived
    on GPU 1 ran its SOBBH Gram on GPU 0 -- a second lookup comp there and
    the GPU-1 residual/PSD read across the link). The multi-shard branches
    already entered each view's device; the single-shard scorer, the
    single-shard expose/fold-back and the Gram context did not."""

    def _move(self, gpus):
        from lisatools.globalfit.moves.sobbhspecialmove import SOBBHChunkedLikeMove

        xp = _TwoGPUSync()           # process default: device 0
        n = 3
        move = object.__new__(SOBBHChunkedLikeMove)
        move.acs = SimpleNamespace(xp=xp, gpus=gpus, linear_data_arr=[object()])
        move.comp = _RecordingComp(xp, n)
        move.m_band_half_width = 3
        move.to_chunked_basis = lambda c: np.asarray(c, dtype=float)
        move._f_band_lo, move._f_band_hi = 0.0, 1.0
        move._shard_spans = lambda *a, **k: None
        move._ll_spans_add = lambda *a, **k: None
        return move, xp, n

    def test_gram_context_enters_the_walker_device(self):
        move, xp, _ = self._move(gpus=[1])
        with move._gram_context(0):
            self.assertEqual(xp.current, 1)
        self.assertEqual(xp.current, 0)
        move_cpu, xp_cpu, _ = self._move(gpus=None)   # no GPUs: a null context
        with move_cpu._gram_context(0):
            self.assertEqual(xp_cpu.current, 0)

    def test_single_shard_scoring_runs_on_the_owning_device(self):
        move, xp, n = self._move(gpus=[1])
        params = np.full((n, 11), 0.5)
        ll, d_h, h_h = move._kernel_ll(params, np.arange(n, dtype=np.int32))
        self.assertEqual(move.comp.calls, [("get_ll_wdm", 1)])
        self.assertEqual(xp.current, 0)              # context released
        self.assertEqual(ll.shape, (n,))

    def test_single_shard_expose_runs_on_the_owning_device(self):
        move, xp, n = self._move(gpus=[1])
        coords = np.full((n, 11), 0.5)               # f_low (col 5) inside the band
        with mock.patch.dict(os.environ, {"SOBBH_CHUNKED_FILL": "1"}):
            move._apply_cold_chain_sources(coords, +1)
            move._apply_cold_chain_sources(coords, -1)
        self.assertEqual(move.comp.calls,
                         [("fill_global_wdm", 1), ("fill_global_wdm", 1)])
        self.assertEqual(xp.current, 0)


# ---------------------------------------------------------------------------
# 9mo job 754: the SOBBH lookup comp's PN step runs through JAX
# ---------------------------------------------------------------------------


class _StrictTwoGPUSync(_StrictTwoGPU):
    """``_StrictTwoGPU`` + ``deviceSynchronize`` + JAX's default device
    (``jax_default``: GPU 0 unless a ``jax_device_context`` moved it)."""

    def __init__(self):
        super().__init__()
        self.cuda.runtime.deviceSynchronize = lambda: None
        self.jax_default = 0


def _tof_class():
    from lisatools.sources.sobbh import wdm_direct as wd

    class _TOF(wd.SOBBHBatchedTOF):
        """The real response build on the fake two-device ``xp``."""

        fake_xp = None

        @property
        def xp(self):
            return self.fake_xp

    return _TOF


class SOBBHLookupJaxDeviceTest(unittest.TestCase):
    """The lookup comp's PN step leaves the walker's device current (9mo job
    754, 2026-10-11).

    ``sobbh_amp_phase_batch`` evaluates the PN series through ``jnp``
    (sources/sobbh/waveform.py). On a GPU JAX those ops run on JAX's DEFAULT
    device, and XLA leaves that device's CUDA context current on the calling
    thread when it returns (its outermost ``ScopedActivateContext`` does not
    restore). On a rank whose walker lives on GPU 1 the caller's cupy
    ``Device(1)`` context was therefore GPU 0 from the PN step on: the
    response, the lookup kernel (against the GPU-1 residual / table) and,
    after the fill returned, the Gram's half-differences
    (addremovemove.py:746) and inner products (diagnostic.py:288) ran on GPU
    0, and the Gram's second pass built a second lookup comp there.

    The fake ``xp`` plays XLA: ``sobbh_amp_phase_batch`` (patched) makes
    ``jax_default`` current, and a patched ``jax_device_context`` moves
    ``jax_default`` the way ``jax.default_device`` does. Everything else --
    the move's ``_gram_info`` / ``_gram_templates`` / ``_gram_context``, the
    router, the comp's ``fill_global_wdm``, ``SOBBHDirectWDM.response`` and
    ``SOBBHBatchedTOF.build`` -- is the production code; strict arrays refuse
    any operand off the current device, as a GPU without peer access would.
    """

    ND = 11                      # the SOBBH sampling basis
    NCH, NFA, NTA = 2, 3, 4      # active-band slab

    def setUp(self):
        from lisatools.sources.sobbh import wdm_direct as wd

        self.wd = wd
        self.xp = xp = _StrictTwoGPUSync()
        self.built, self.launches, self.gens, self.pn_jax = [], [], [], []
        test = self

        def fake_pn(p, node_t, reference_time):
            # XLA: the jnp ops run on JAX's default device, which stays current
            test.pn_jax.append(xp.jax_default)
            xp.current = xp.jax_default
            n = p.shape[0]
            amp = np.repeat((1.0 + p @ np.arange(1.0, p.shape[1] + 1.0))[:, None],
                            node_t.size, axis=1)
            return amp, np.zeros_like(amp), np.full(n, np.inf)

        class FakeGen:
            def __init__(self, t, amp, phase, *args, t_input=None, **kwargs):
                test.gens.append((xp.current, {a._dev for a in (t, amp, phase, t_input)}))
                self.amp = amp

            def __call__(self, *angles, return_spline=False):
                return SimpleNamespace(coef=self.amp)

        def kernel_fill(out, flat, di, fac):
            # the fused kernel writes through a raw pointer: record WHERE it
            # launched, then accumulate fac * h into each row's slab
            test.launches.append((xp.current, flat._dev))
            coef = out.coef.view(np.ndarray)[:, 0]
            slabs = flat.view(np.ndarray).reshape(-1, self.NCH, self.NFA, self.NTA)
            pattern = np.arange(1.0, self.NCH * self.NFA * self.NTA + 1.0).reshape(
                self.NCH, self.NFA, self.NTA)
            for k, slot in enumerate(np.asarray(di)):
                slabs[slot] += fac[k] * (coef[k] * pattern + coef[k] ** 2 * pattern[::-1])
            return {"dropped_pixels": 0}

        def build(dev):
            test.built.append(dev)
            tof = _tof_class()(None, None, 0.0, n_grid=16, buffer_time=600.0,
                               eval_dt=1.0e4, force_backend="cpu")
            tof.fake_xp = xp
            tof._orbit_span_cache = (-1.0e12, 1.0e12)
            direct = object.__new__(wd.SOBBHDirectWDM)
            direct.ev = SimpleNamespace(xp=xp)
            direct.tof = tof
            direct.t_pixels = np.array([0.0, 1.0e5])
            direct.num_m_layers = 2
            direct.last_spans = {}
            direct.kernel_fill = kernel_fill
            comp = object.__new__(wd.SOBBHLookupComputations)
            comp.direct = direct
            comp._build_device = dev
            comp._band_noted = False
            comp.wdm_settings = SimpleNamespace(Nf_active=self.NFA, Nt_active=self.NTA)
            comp.nchannels = self.NCH
            comp.uses_kernel = True
            comp.row_batch = 32
            comp.last_call_spans = None
            return comp

        for p in (
            mock.patch.object(wd, "sobbh_amp_phase_batch", side_effect=fake_pn),
            mock.patch("lisatools.response.tdionfly.TDTDIonTheFly", FakeGen),
            mock.patch.object(wd.SOBBHLookupRouter, "xp", new=property(lambda s: xp)),
        ):
            p.start()
            self.addCleanup(p.stop)
        # the router is built after the run pinned the walker's device
        with xp.cuda.Device(1):
            self.router = wd.SOBBHLookupRouter(build, "cpu")
        self.assertEqual(self.built, [1])

    def _jax_follows_cupy(self):
        """``jax_device_context`` as JAX does it: the default device moves."""
        xp = self.xp

        @contextmanager
        def ctx(device):
            prev = xp.jax_default
            if device is not None:
                xp.jax_default = int(device)
            try:
                yield
            finally:
                xp.jax_default = prev

        return mock.patch.object(self.wd, "jax_device_context", side_effect=ctx,
                                 create=True)

    def _jax_cannot_see_the_device(self):
        """``jax_device_context`` is a null context (JAX on CPU, or fewer JAX
        devices than cupy sees): the PN step still runs on JAX's GPU 0."""
        from contextlib import nullcontext

        return mock.patch.object(self.wd, "jax_device_context",
                                 side_effect=lambda device: nullcontext(), create=True)

    def _move(self):
        from eryn.priors import ProbDistContainer, UniformDistribution

        from lisatools.globalfit.moves.sobbhspecialmove import SOBBHChunkedLikeMove

        xp, nd = self.xp, self.ND
        move = object.__new__(SOBBHChunkedLikeMove)
        move.branch_name = "sobbh"
        move.priors = {"sobbh": ProbDistContainer(
            {i: UniformDistribution(0.0, 1.0) for i in range(nd)})}
        move.eigen_gram_target = 1e-3          # the production two-pass default
        move.eigen_gram_eps_rel = 1e-3
        move.waveform_like_kwargs = {}
        move._to_phys = lambda X: X
        move.comp = self.router
        move.m_band_half_width = 3
        move._f_band_lo, move._f_band_hi = 0.0, 1.0
        # the walker's PSD, on its shard GPU
        psd = object.__new__(SensitivityMatrixBase)
        psd._sens_mat = xp.wrap(np.full((2, 2, 5), 2.0), 1)
        psd._invC = xp.wrap(np.full((2, 2, 5), 0.5), 1)
        psd._detC = xp.wrap(np.full(5, 4.0), 1)
        psd.do_inv_det, psd._inv_det_dirty = True, False
        self.psd = psd
        data = SimpleNamespace(
            arr=xp.wrap(np.zeros((self.NCH, self.NFA, self.NTA)), 1), settings="box")
        ac = SimpleNamespace(data=data, _slice_to_template=lambda h: (None, None, psd))
        # production holder: one walker, one shard, gpus=[1], data on GPU 1
        move.acs = SimpleNamespace(
            acs=np.array([ac], dtype=object), xp=xp, gpus=[1],
            linear_data_arr=[xp.wrap(np.zeros(self.NCH * self.NFA * self.NTA), 1)])
        return move

    def _run_gram(self):
        import lisatools.diagnostic as diag
        from lisatools.globalfit.moves import addremovemove as arm

        xp, move, seen = self.xp, self._move(), []

        def ip(a, b, psd=None, **kw):
            seen.append((xp.current, a.arr._dev, b.arr._dev, psd.invC._dev))
            _check_current(xp, a.arr, b.arr, psd.invC)
            return float(np.sum(a.arr.view(np.ndarray) * b.arr.view(np.ndarray)))

        with mock.patch.object(arm, "get_array_module",
                               side_effect=lambda a: xp if isinstance(a, _Arr) else np), \
                mock.patch.object(arm, "WDMSignal",
                                  side_effect=lambda arr, box: SimpleNamespace(
                                      arr=arr, settings=box)), \
                mock.patch.object(diag, "inner_product", side_effect=ip):
            info = move._gram_info(np.full(self.ND, 0.5), 0, np.ones(self.ND))
        return info, seen

    def _assert_all_on_the_walker_device(self, info, seen):
        self.assertEqual(info.shape, (self.ND, self.ND))
        self.assertTrue(np.all(np.isfinite(info)))
        self.assertEqual(self.built, [1])                    # no second comp
        self.assertEqual(len(self.launches), 2)              # two template passes
        self.assertEqual(set(self.launches), {(1, 1)})       # fill on the buffer's GPU
        self.assertEqual({g[0] for g in self.gens}, {1})     # response built on GPU 1
        self.assertEqual(set().union(*(g[1] for g in self.gens)), {1})
        self.assertEqual({s[0] for s in seen}, {1})          # inner products on GPU 1
        self.assertEqual(set().union(*(set(s[1:]) for s in seen)), {1})
        self.assertEqual(self.xp.current, 0)                 # context released

    def test_gram_runs_on_the_walker_device(self):
        with self._jax_follows_cupy():
            info, seen = self._run_gram()
        self._assert_all_on_the_walker_device(info, seen)
        self.assertEqual(set(self.pn_jax), {1})              # JAX pinned with cupy

    def test_gram_survives_a_jax_that_cannot_follow(self):
        with self._jax_cannot_see_the_device():
            info, seen = self._run_gram()
        self._assert_all_on_the_walker_device(info, seen)
        self.assertEqual(set(self.pn_jax), {0})              # XLA left GPU 0 current ...
        self.assertEqual(self.xp.jax_default, 0)             # ... and the build re-entered 1

    def test_in_model_fill_launches_on_the_walker_device(self):
        # the expose / fold-back ran the same response inside the shard
        # context: on GPU-1 ranks the kernel launched on GPU 0 against the
        # GPU-1 residual (silent: raw pointers, no cupy warning)
        move = self._move()
        coords = np.full((1, self.ND), 0.5)
        with self._jax_follows_cupy(), \
                mock.patch.dict(os.environ, {"SOBBH_CHUNKED_FILL": "1"}):
            move._apply_cold_chain_sources(coords, +1)
            move._apply_cold_chain_sources(coords, -1)
        self.assertEqual(self.launches, [(1, 1), (1, 1)])
        self.assertEqual({g[0] for g in self.gens}, {1})
        self.assertEqual(self.built, [1])
        self.assertEqual(self.xp.current, 0)


if __name__ == "__main__":
    unittest.main()
