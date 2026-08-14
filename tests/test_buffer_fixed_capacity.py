"""Fixed-capacity SubBandBuffer allocation (user ruling 2026-08-14).

CPU-only unit tests for the fixed-capacity GB RJ staging buffer:

* ``resize_to`` view semantics — the per-slot ``band_buffer`` /
  ``psd_buffer`` accessors are FRONT VIEWS sharing memory with the
  capacity allocation, stay C-contiguous, and track ``num_bands_now``
  across shrink/grow resizes without touching data;
* the no-stale-specials property — after ``resize_to`` the special-index
  map is invalid (``get_index`` hard-errors, partial
  ``update_special_indices`` fills are rejected) until the mandatory FULL
  rebind, after which ``get_index`` can only ever return bound slots;
* the FD per-slot window-start store — capacity-sized, front-written by
  the specials setter;
* the ``GB_BUFFER_FIXED_CAPACITY`` env gate
  (``gbspecialstretch._buffer_fixed_capacity_active``);
* the bounded :class:`BandView` (``n_bands`` front limit).

Full ``SubBandBuffer`` construction needs the GB computation objects +
likelihood engine, so — per the test plan — the buffer here is built via
``__new__`` with exactly the attributes the tested methods consume, plus a
REAL (CPU) ``AnalysisContainerArray`` allocation underneath so the view
semantics are exercised against live memory, mirroring
``SubBandBuffer._build_band_ac_list``.
"""

from __future__ import annotations

import os
import unittest

import numpy as np


def _make_capacity_buffer(
    capacity: int,
    num_bound: int,
    nwalkers: int = 3,
    n_bands: int = 8,
    nchannels: int = 3,
    store_len: int = 16,
    df: float = 1e-4,
):
    """A minimally-constructed fixed-capacity SubBandBuffer (CPU, FD basis).

    Returns ``(buf, specials)`` where ``specials`` are the ``num_bound``
    packed (temp, walker, band) cell indices currently bound.
    """
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.domains import FDSettings
    from lisatools.globalfit.moves.gbbands import SubBandBuffer, pack_special_index
    from lisatools.sensitivity import SensitivityMatrixBase
    from lisatools.utils.parallelbase import LISAToolsParallelModule

    settings = FDSettings(N=store_len, df=df, force_backend="cpu")

    buf = SubBandBuffer.__new__(SubBandBuffer)
    buf.force_backend = "cpu"
    LISAToolsParallelModule.__init__(buf, force_backend="cpu")

    # --- knobs the tested methods consume -----------------------------
    buf.nwalkers = nwalkers
    buf.nchannels = nchannels
    buf.tdi_channel_setup = "AET"
    buf.is_rj = False
    buf.edge_buffer = 2
    buf._df = df
    # Band edges start well above 0 so every window start index stays >= 0.
    buf.band_edges = df * (100.0 + 50.0 * np.arange(n_bands + 1))
    buf.num_bands = n_bands
    buf._basis_settings = settings
    buf._fd_store_length_value = store_len
    buf._parent_ind_min = None
    buf._parent_stored_len = None
    buf.use_template_arr = False
    buf.alloc_capacity = int(capacity)
    buf.num_bands_now = int(num_bound)
    buf._specials_placeholder = False
    # FD per-slot window-start store: capacity-sized (as the real ctor
    # allocates it under fixed capacity), front-written by the setter.
    buf._min_freq_inds_store = np.zeros(capacity, dtype=np.int32)

    # --- real per-slot allocation at capacity (mirrors
    # _build_band_ac_list; SubBandBuffer IS the ACA) --------------------
    ac_list = []
    for _ in range(capacity):
        res_data = np.zeros((nchannels, store_len), dtype=complex)
        data_domain = settings.associated_class(res_data, settings)
        sm = SensitivityMatrixBase(settings, skip_inv_det=True)
        sm.sens_mat = np.broadcast_to(
            np.zeros((), dtype=float), (nchannels, store_len)
        )
        sm.invC = np.zeros((nchannels, store_len), dtype=float)
        sm.channel_shape = (nchannels,)
        ac_list.append(AnalysisContainer(data_domain, sm))
    AnalysisContainerArray.__init__(buf, ac_list, gpus=None, complex_psd=False)

    # --- bind the first num_bound cells --------------------------------
    specials = _pack_cells(buf, np.arange(num_bound))
    buf.special_indices_unique = specials
    buf.psd_shape = (buf.num_bands_now, nchannels, store_len)
    return buf, specials


def _pack_cells(buf, cell_ids):
    """Distinct packed specials for ``cell_ids`` (temp 0, cycling walker/band)."""
    from lisatools.globalfit.moves.gbbands import pack_special_index

    cell_ids = np.asarray(cell_ids)
    temps = np.zeros_like(cell_ids)
    walkers = cell_ids % buf.nwalkers
    bands = cell_ids % buf.num_bands
    # (walker, band) pairs must be unique per special; with nwalkers=3 and
    # num_bands=8 the pair (i % 3, i % 8) is unique for i < 24.
    assert len(cell_ids) <= buf.nwalkers * buf.num_bands
    return pack_special_index(temps, walkers, bands, buf.nwalkers)


class ResizeViewSemanticsTest(unittest.TestCase):
    def setUp(self):
        try:
            self.buf, self.specials = _make_capacity_buffer(
                capacity=8, num_bound=6
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")

    def test_front_views_share_memory_with_capacity_alloc(self):
        buf = self.buf
        # Bound-count leading axis...
        self.assertEqual(buf.band_buffer.shape[0], 6)
        self.assertEqual(buf.psd_buffer.shape[0], 6)
        # ...over a capacity-sized allocation.
        full = buf.data_shaped[0]
        self.assertEqual(full.shape[0], 8)
        # Writes through the front view land in the capacity allocation.
        buf.band_buffer[2] = 7.0 + 0.0j
        np.testing.assert_array_equal(
            np.asarray(full[2]).real, np.full(full[2].shape, 7.0)
        )
        self.assertTrue(np.shares_memory(buf.band_buffer, full))
        self.assertTrue(
            np.shares_memory(buf.psd_buffer, buf.psd_shaped[0])
        )

    def test_views_stay_contiguous(self):
        self.assertTrue(self.buf.band_buffer.flags["C_CONTIGUOUS"])
        self.assertTrue(self.buf.psd_buffer.flags["C_CONTIGUOUS"])

    def test_resize_shrink_and_grow_preserve_data(self):
        buf = self.buf
        buf.band_buffer[1] = 3.0 + 0.0j
        buf.psd_buffer[1] = 5.0

        buf.resize_to(4)
        self.assertEqual(int(buf.num_bands_now), 4)
        self.assertEqual(buf.band_buffer.shape[0], 4)
        self.assertEqual(buf.psd_buffer.shape[0], 4)
        self.assertEqual(buf.psd_shape[0], 4)
        # resize_to never touches data.
        np.testing.assert_array_equal(
            np.asarray(buf.band_buffer[1]).real,
            np.full(buf.band_buffer[1].shape, 3.0),
        )
        np.testing.assert_array_equal(
            np.asarray(buf.psd_buffer[1]), np.full(buf.psd_buffer[1].shape, 5.0)
        )

        buf.resize_to(8)  # grow to full capacity
        self.assertEqual(int(buf.num_bands_now), 8)
        self.assertEqual(buf.band_buffer.shape[0], 8)
        np.testing.assert_array_equal(
            np.asarray(buf.band_buffer[1]).real,
            np.full(buf.band_buffer[1].shape, 3.0),
        )
        self.assertTrue(buf.band_buffer.flags["C_CONTIGUOUS"])

    def test_resize_bounds_and_noncapacity_guard(self):
        buf = self.buf
        with self.assertRaises(AssertionError):
            buf.resize_to(0)
        with self.assertRaises(AssertionError):
            buf.resize_to(9)  # > alloc_capacity
        buf.alloc_capacity = None
        with self.assertRaises(AssertionError):
            buf.resize_to(4)

    def test_min_freq_inds_store_is_capacity_sized_and_front_written(self):
        buf = self.buf
        # The FD binding shape-checks min_freq_inds against the ACA's
        # allocated row count: capacity-length, front entries live.
        self.assertEqual(buf.min_freq_inds.shape[0], 8)
        np.testing.assert_array_equal(
            np.asarray(buf._min_freq_inds_store[:6]),
            np.asarray(buf.start_freq_inds),
        )
        # After a shrink + full rebind, only the first k entries refresh.
        buf.resize_to(4)
        new_specials = _pack_cells(buf, np.arange(6, 10))
        buf.update_special_indices(new_specials, inds_fill=np.arange(4))
        np.testing.assert_array_equal(
            np.asarray(buf._min_freq_inds_store[:4]),
            np.asarray(buf.start_freq_inds),
        )
        self.assertEqual(buf._min_freq_inds_store.shape[0], 8)


class NoStaleSpecialsTest(unittest.TestCase):
    def setUp(self):
        try:
            self.buf, self.specials = _make_capacity_buffer(
                capacity=8, num_bound=6
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")

    def test_get_index_maps_bound_cells_before_resize(self):
        now = np.asarray(self.buf.get_index(self.specials))
        np.testing.assert_array_equal(now, np.arange(6))

    def test_get_index_hard_errors_in_placeholder_window(self):
        self.buf.resize_to(4)
        with self.assertRaises(AssertionError):
            self.buf.get_index(self.specials[:2])

    def test_partial_update_rejected_in_placeholder_window(self):
        self.buf.resize_to(4)
        with self.assertRaises(AssertionError):
            self.buf.update_special_indices(
                self.specials[:2], inds_fill=np.arange(2)
            )

    def test_full_rebind_leaves_no_stale_specials(self):
        buf = self.buf
        old_specials = self.specials.copy()
        buf.resize_to(4)
        new_specials = _pack_cells(buf, np.arange(6, 10))
        buf.update_special_indices(new_specials, inds_fill=np.arange(4))

        # Map arrays are fully rebuilt at length k from the new specials.
        self.assertEqual(buf.special_indices_unique.shape[0], 4)
        self.assertEqual(buf.special_indices_unique_sort.shape[0], 4)
        np.testing.assert_array_equal(
            np.sort(np.asarray(buf.special_indices_unique)),
            np.sort(np.asarray(new_specials)),
        )
        self.assertEqual(buf.unique_band_combos.shape[0], 4)

        # New cells map to their slots...
        now = np.asarray(buf.get_index(new_specials))
        np.testing.assert_array_equal(np.sort(now), np.arange(4))
        # ...and NO probe (including every previous binding's special) can
        # ever reach a slot >= k: the searchsorted tables hold only the 4
        # live entries.
        probe = np.asarray(buf.get_index(old_specials))
        self.assertTrue(np.all(probe < 4))

    def test_grow_rebind_after_shrink(self):
        buf = self.buf
        buf.resize_to(3)
        buf.update_special_indices(
            _pack_cells(buf, np.arange(3)), inds_fill=np.arange(3)
        )
        buf.resize_to(8)
        specials8 = _pack_cells(buf, np.arange(8, 16))
        buf.update_special_indices(specials8, inds_fill=np.arange(8))
        self.assertEqual(buf.special_indices_unique.shape[0], 8)
        now = np.asarray(buf.get_index(specials8))
        np.testing.assert_array_equal(np.sort(now), np.arange(8))


class EnvGateTest(unittest.TestCase):
    class _RJSorter:
        rj_prop = object()

    class _InModelSorter:
        rj_prop = None

    def _call(self, sorter, kwargs):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _buffer_fixed_capacity_active,
        )

        return _buffer_fixed_capacity_active(sorter, kwargs)

    def _with_env(self, value):
        old = os.environ.get("GB_BUFFER_FIXED_CAPACITY")
        if value is None:
            os.environ.pop("GB_BUFFER_FIXED_CAPACITY", None)
        else:
            os.environ["GB_BUFFER_FIXED_CAPACITY"] = value
        return old

    def _restore_env(self, old):
        if old is None:
            os.environ.pop("GB_BUFFER_FIXED_CAPACITY", None)
        else:
            os.environ["GB_BUFFER_FIXED_CAPACITY"] = old

    def test_default_on_for_rj_proposal_buffers(self):
        old = self._with_env(None)
        try:
            self.assertTrue(self._call(self._RJSorter(), {}))
        finally:
            self._restore_env(old)

    def test_env_zero_restores_legacy_behavior(self):
        old = self._with_env("0")
        try:
            self.assertFalse(self._call(self._RJSorter(), {}))
        finally:
            self._restore_env(old)

    def test_non_rj_sorters_never_fixed_capacity(self):
        old = self._with_env("1")
        try:
            self.assertFalse(self._call(self._InModelSorter(), {}))
            self.assertFalse(self._call(object(), {}))  # no rj_prop attr
        finally:
            self._restore_env(old)

    def test_template_twin_buffers_excluded(self):
        old = self._with_env("1")
        try:
            self.assertFalse(
                self._call(self._RJSorter(), {"use_template_arr": True})
            )
            self.assertTrue(
                self._call(self._RJSorter(), {"use_template_arr": False})
            )
        finally:
            self._restore_env(old)


class BoundedBandViewTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.analysiscontainer import (
                AnalysisContainer,
                AnalysisContainerArray,
            )
            from lisatools.domains import FDSettings
            from lisatools.sensitivity import SensitivityMatrixBase
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")

        settings = FDSettings(N=16, df=1e-4, force_backend="cpu")
        acs = []
        for i in range(5):
            arr = np.full((2, 16), float(i + 1), dtype=complex)
            data = settings.associated_class(arr, settings)
            sm = SensitivityMatrixBase(settings, skip_inv_det=True)
            sm.sens_mat = np.broadcast_to(np.zeros((), dtype=float), (2, 16))
            sm.invC = np.zeros((2, 16), dtype=float)
            sm.channel_shape = (2,)
            acs.append(AnalysisContainer(data, sm))
        self.aca = AnalysisContainerArray(acs, gpus=None)

    def test_default_view_is_unbounded(self):
        view = self.aca.data_shaped_view()
        self.assertEqual(view.shape[0], 5)
        self.assertEqual(len(view), 5)

    def test_bounded_view_exposes_front_slots_only(self):
        view = self.aca.data_shaped_view(n_bands=3)
        self.assertEqual(view.shape[0], 3)
        self.assertEqual(len(view), 3)
        self.assertEqual(view._shards[0].shape[0], 3)
        gathered = view.gather()
        self.assertEqual(gathered.shape[0], 3)
        np.testing.assert_array_equal(
            gathered.real[:, 0, 0], np.array([1.0, 2.0, 3.0])
        )

    def test_bounded_view_writes_reach_backing_buffer(self):
        view = self.aca.data_shaped_view(n_bands=3)
        view[np.array([1])] = 9.0 + 0.0j
        np.testing.assert_array_equal(
            np.asarray(self.aca.data_shaped[0][1]).real,
            np.full((2, 16), 9.0),
        )
        # Slots beyond the bound are untouched.
        np.testing.assert_array_equal(
            np.asarray(self.aca.data_shaped[0][3]).real,
            np.full((2, 16), 4.0),
        )

    def test_bounded_view_rejects_bad_limits(self):
        from lisatools.analysiscontainer import BandView

        with self.assertRaises(ValueError):
            BandView(self.aca, kind="data", n_bands=0)
        with self.assertRaises(ValueError):
            BandView(self.aca, kind="data", n_bands=6)


if __name__ == "__main__":
    unittest.main()
