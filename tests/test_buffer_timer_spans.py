"""Buffer-work sub-mark timer spans (2026-08-14).

CPU-only unit tests for the ``timer=`` instrumentation on
``BandSorter.get_buffer``: the phases accumulate under exactly the names

* ``bufbuild_alloc``    -- fresh SubBandBuffer construction
* ``buffill_resid_psd`` -- fill_buffer_residual_and_psd_from_acs
* ``buffill_inject``    -- source injection into the band buffers
* ``buffill_template``  -- template-twin reset + template injection

and ``timer=None`` (the default) is behavior-identical to today. Also
checks ``_cached_get_buffer`` forwards ``self._prop_timer`` on both its
build and rebind paths.

Full ``SubBandBuffer`` construction needs the GB computation objects +
likelihood engine, so — following ``tests/test_buffer_fixed_capacity.py``
— the buffer here is built via ``__new__`` with the engine-backed methods
stubbed (the spans wrap those calls; what is timed is the dispatch), and
the fresh-build branch runs against a patched module-level
``SubBandBuffer`` name.
"""

from __future__ import annotations

import os
import types
import unittest

import numpy as np


def _make_stub_buffer(k=4, use_template_arr=False):
    """A minimal SubBandBuffer (``__new__``) with recorded stub methods."""
    from lisatools.globalfit.moves.gbbands import SubBandBuffer

    buf = SubBandBuffer.__new__(SubBandBuffer)
    buf.use_template_arr = use_template_arr
    buf.num_bands_now = int(k)
    buf.calls = []
    buf.update_special_indices = (
        lambda *a, **kw: buf.calls.append("update_specials")
    )
    buf.fill_buffer_residual_and_psd_from_acs = (
        lambda *a, **kw: buf.calls.append("fill_resid")
    )
    buf.get_index = lambda x: np.arange(len(x))
    buf.reset_template_buffers = (
        lambda *a, **kw: buf.calls.append("reset_tmpl")
    )
    buf.add_sources_to_template_buffer = (
        lambda *a, **kw: buf.calls.append("add_tmpl")
    )
    buf.add_sources_to_band_buffer = (
        lambda *a, **kw: buf.calls.append("add_band")
    )
    return buf


def _make_stub_sorter(nwalkers=3, num_bands=8):
    """A stub sorter carrying the REAL ``BandSorter.get_buffer``.

    Zero sources (empty injection maps), so the engine-backed methods are
    reached with empty argument arrays — the code under test is
    get_buffer's phase structure, not the engine.
    """
    from lisatools.globalfit.moves.gbbands import (
        BandSorter,
        unpack_special_index,
    )

    class _StubSorter:
        def __init__(self):
            self.rj_prop = object()
            self.nwalkers = nwalkers
            self.num_bands = num_bands
            self.band_N_vals = np.full(num_bands, 128, dtype=np.int64)
            self.main_band_sorter = types.SimpleNamespace(
                special_band_inds=np.zeros(0, dtype=np.int64),
                inds=np.zeros(0, dtype=bool),
                coords=np.zeros((0, 9)),
                band_inds=np.zeros(0, dtype=np.int64),
                leaf_inds=np.zeros(0, dtype=np.int64),
                # BandSorter.get_buffer guards against a pending deferred
                # relabel through main_band_sorter (gbbands.py, the
                # GB_CELL_LABEL_DEFERRED work). A real sorter with nothing
                # deferred is a no-op; the stub says so explicitly.
                _assert_cell_labels_flushed=lambda where: None,
            )

        def get_separate_inds_from_special_index(self, specials):
            return unpack_special_index(specials, self.nwalkers)

        get_buffer = BandSorter.get_buffer

    return _StubSorter()


def _pack(sorter, cell_ids):
    from lisatools.globalfit.moves.gbbands import pack_special_index

    cell_ids = np.asarray(cell_ids)
    assert len(cell_ids) <= sorter.nwalkers * sorter.num_bands
    return pack_special_index(
        np.zeros_like(cell_ids),
        cell_ids % sorter.nwalkers,
        cell_ids % sorter.num_bands,
        sorter.nwalkers,
    )


class GetBufferSpanTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbspecialstretch import (
                _ProposeTimer,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")
        self._ProposeTimer = _ProposeTimer
        self.sorter = _make_stub_sorter()
        self.specials = _pack(self.sorter, np.arange(4))

    def _rebind(self, buf, timer):
        return self.sorter.get_buffer(
            object(),
            self.specials,
            inds_fill=np.arange(4),
            buffer_obj=buf,
            timer=timer,
        )

    def test_rebind_band_buffer_spans(self):
        buf = _make_stub_buffer(use_template_arr=False)
        tm = self._ProposeTimer()
        self._rebind(buf, tm)
        self.assertIn("buffill_resid_psd", tm.stages)
        self.assertIn("buffill_inject", tm.stages)
        self.assertNotIn("buffill_template", tm.stages)
        self.assertNotIn("bufbuild_alloc", tm.stages)  # no fresh alloc
        self.assertEqual(buf.calls[-2:], ["fill_resid", "add_band"])

    def test_rebind_template_twin_spans(self):
        buf = _make_stub_buffer(use_template_arr=True)
        tm = self._ProposeTimer()
        self._rebind(buf, tm)
        self.assertIn("buffill_resid_psd", tm.stages)
        self.assertIn("buffill_template", tm.stages)
        self.assertNotIn("buffill_inject", tm.stages)
        self.assertEqual(
            buf.calls[-3:], ["fill_resid", "reset_tmpl", "add_tmpl"]
        )

    def test_timer_none_is_default_and_identical(self):
        buf_timed = _make_stub_buffer(use_template_arr=True)
        tm = self._ProposeTimer()
        self._rebind(buf_timed, tm)

        buf_plain = _make_stub_buffer(use_template_arr=True)
        # No timer kwarg at all -- the pre-instrumentation call shape.
        self.sorter.get_buffer(
            object(),
            self.specials,
            inds_fill=np.arange(4),
            buffer_obj=buf_plain,
        )
        self.assertEqual(buf_plain.calls, buf_timed.calls)

    def test_fresh_build_alloc_span(self):
        # Patch the module-level SubBandBuffer name so the REAL fresh-build
        # branch of get_buffer runs without the GB engine stack.
        import lisatools.globalfit.moves.gbbands as gbbands
        from lisatools.domains import FDSettings

        class _FakeBuilt:
            def __init__(self, *args, **kwargs):
                self.use_template_arr = kwargs.get("use_template_arr", False)
                self.num_bands_now = int(args[7])
                self.calls = []

            def fill_buffer_residual_and_psd_from_acs(self, *a, **kw):
                self.calls.append("fill_resid")

            def get_index(self, x):
                return np.arange(len(x))

            def reset_template_buffers(self, *a, **kw):
                self.calls.append("reset_tmpl")

            def add_sources_to_template_buffer(self, *a, **kw):
                self.calls.append("add_tmpl")

            def add_sources_to_band_buffer(self, *a, **kw):
                self.calls.append("add_band")

        sorter = self.sorter
        sorter.gb = None
        sorter.band_edges = np.linspace(1e-3, 2e-3, 9)
        sorter.max_data_store_size = 16
        sorter.transform_fn = None
        sorter.waveform_kwargs = {}
        sorter.gb_wdm_comp = None
        sorter.gb_fd_comp = None
        sorter.force_backend = "cpu"
        sorter.wdm_band_slab_layers = None
        sorter.wdm_slab_guard_layers = 1
        acs = types.SimpleNamespace(
            nchannels=3, settings=FDSettings(N=16, df=1e-4)
        )

        tm = self._ProposeTimer()
        orig = gbbands.SubBandBuffer
        gbbands.SubBandBuffer = _FakeBuilt
        try:
            buf = sorter.get_buffer(acs, self.specials, timer=tm)
        finally:
            gbbands.SubBandBuffer = orig

        self.assertIn("bufbuild_alloc", tm.stages)
        self.assertIn("buffill_resid_psd", tm.stages)
        self.assertIn("buffill_inject", tm.stages)
        self.assertEqual(buf.calls, ["fill_resid", "add_band"])


class CachedGetBufferTimerForwardingTest(unittest.TestCase):
    """``_cached_get_buffer`` passes ``self._prop_timer`` on both paths."""

    class _RecordingSorter:
        def __init__(self):
            self.rj_prop = object()
            self.calls = []

        def get_buffer(
            self,
            acs,
            specials,
            inds_fill=None,
            buffer_obj=None,
            allow_resize=False,
            timer=None,
            **kwargs,
        ):
            self.calls.append(
                dict(
                    kind="build" if buffer_obj is None else "rebind",
                    timer=timer,
                    kwargs=dict(kwargs),
                )
            )
            if buffer_obj is None:
                buffer_obj = types.SimpleNamespace(
                    num_bands_now=int(len(specials)),
                    alloc_capacity=kwargs.get("alloc_capacity"),
                )
            elif allow_resize and buffer_obj.alloc_capacity is not None:
                buffer_obj.num_bands_now = int(len(specials))
            return buffer_obj

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbspecialstretch import (
                GBSpecialBase,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")

        class _FakeMove:
            _cached_get_buffer = GBSpecialBase._cached_get_buffer
            xp = np
            ntemps = 2
            nwalkers = 3
            band_edges = np.linspace(1e-3, 2e-3, 9)  # 8 bands
            num_band_preload_total = 16
            backend = types.SimpleNamespace(uses_cupy=False)

        self.mv = _FakeMove()
        self.sorter = self._RecordingSorter()
        self._env_old = os.environ.get("GB_BUFFER_FIXED_CAPACITY")
        os.environ["GB_BUFFER_FIXED_CAPACITY"] = "1"

    def tearDown(self):
        if self._env_old is None:
            os.environ.pop("GB_BUFFER_FIXED_CAPACITY", None)
        else:
            os.environ["GB_BUFFER_FIXED_CAPACITY"] = self._env_old

    def test_timer_forwarded_on_build_and_rebind(self):
        marker = object()
        self.mv._prop_timer = marker
        self.mv._cached_get_buffer(self.sorter, None, np.arange(10))
        self.mv._cached_get_buffer(self.sorter, None, np.arange(4))
        kinds = [c["kind"] for c in self.sorter.calls]
        self.assertEqual(kinds, ["build", "rebind"])
        for call in self.sorter.calls:
            self.assertIs(call["timer"], marker)
            # timer is plumbing, never a construction kwarg / signature key.
            self.assertNotIn("timer", call["kwargs"])

    def test_no_timer_attribute_passes_none(self):
        self.mv._cached_get_buffer(self.sorter, None, np.arange(6))
        self.assertIsNone(self.sorter.calls[0]["timer"])


if __name__ == "__main__":
    unittest.main()
