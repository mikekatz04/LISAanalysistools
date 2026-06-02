"""Parity tests for :class:`BandView`.

The :class:`BandView` is the multi-shard router behind the GB Buffer's
``band_buffer`` / ``psd_buffer`` / ``template_buffer`` per-band accessors.
It must behave like a ``(num_bands, *per_band_shape)`` ndarray for the
operations the Buffer uses:

* ``view[band_inds] = scalar``
* ``view[band_inds] = ndarray`` (per-band slabs)
* ``view[band_inds]`` -> gathered ndarray
* ``view[i] = view[j]`` (per-band copy, used by the template swap path)
* ``view[band_inds] += ndarray`` (read-modify-write)
* ``view.copy()`` / ``view.gather()``
* ``view.shape`` / ``view.dtype``

CPU-only here; the multi-GPU shard routing uses the same code path
(``np.unique(shard_ids)`` + per-shard slicing) -- the only difference is
the ``cp.cuda.Device(gpu)`` context entry, which doesn't change the
numerical result. GPU parity is verified by the end-to-end GB special
move regression on a multi-GPU host.
"""

from __future__ import annotations

import unittest

import numpy as np


def _make_aca_and_reference(num_acs: int):
    """Build a CPU ACA with ``num_acs`` ACs and a reference ndarray.

    Each AC's residual is a unique constant so per-band read/write
    operations are visually distinguishable.
    """
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.domains import FDSettings, FDSignal
    from lisatools.sensitivity import AET2SensitivityMatrix
    from lisatools import detector as lisa

    settings = FDSettings(
        N=128, df=1e-4, min_freq=1e-4, max_freq=1e-2,
        force_backend="cpu",
    )
    sens_mat_template = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)

    nchannels = sens_mat_template.shape[0]
    data_length = settings.N_active

    acs = []
    expected_data = np.zeros(
        (num_acs, nchannels, data_length), dtype=complex
    )
    for i in range(num_acs):
        arr = np.full((nchannels, data_length), float(i + 1), dtype=complex)
        data = FDSignal(arr, settings)
        # Each AC gets its own sens_mat instance so the per-shard
        # ``invC`` storage doesn't alias across ACs.
        sm = AET2SensitivityMatrix(settings, model=lisa.sangria_v2)
        acs.append(AnalysisContainer(data, sm))
        expected_data[i] = arr

    aca = AnalysisContainerArray(acs, gpus=None)
    return aca, expected_data


class BandViewBasicsTest(unittest.TestCase):
    def setUp(self):
        try:
            self.aca, self.expected = _make_aca_and_reference(num_acs=4)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")
        # The data_shaped_view should behave like the (num_acs, *) ndarray
        # we synthesised in expected.
        self.view = self.aca.data_shaped_view()

    def test_shape_and_dtype(self):
        self.assertEqual(self.view.shape, self.expected.shape)
        self.assertEqual(self.view.dtype, self.expected.dtype)
        self.assertEqual(self.view.ndim, self.expected.ndim)
        self.assertEqual(len(self.view), self.expected.shape[0])

    def test_gather_full_buffer_matches_reference(self):
        gathered = self.view.gather()
        np.testing.assert_array_equal(gathered, self.expected)

    def test_copy_alias_of_gather(self):
        np.testing.assert_array_equal(self.view.copy(), self.view.gather())

    def test_scalar_index_read_matches(self):
        for i in range(self.expected.shape[0]):
            np.testing.assert_array_equal(self.view[i], self.expected[i])

    def test_array_index_read_matches(self):
        idx = np.array([3, 0, 2])
        np.testing.assert_array_equal(self.view[idx], self.expected[idx])

    def test_array_index_write_scalar(self):
        idx = np.array([1, 2])
        self.view[idx] = 0.0
        self.expected[idx] = 0.0
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_array_index_write_per_row(self):
        idx = np.array([0, 3])
        payload = np.full(
            (len(idx),) + tuple(self.expected.shape[1:]), 9.5, dtype=complex
        )
        self.view[idx] = payload
        self.expected[idx] = payload
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_in_place_iadd_round_trips(self):
        """``view[idx] += X`` must round-trip through __getitem__/__setitem__."""
        idx = np.array([0, 1])
        delta = np.full(
            (len(idx),) + tuple(self.expected.shape[1:]), 0.25, dtype=complex
        )
        self.view[idx] += delta
        self.expected[idx] += delta
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_swap_via_get_then_set(self):
        """Mirror the template_buffer swap pattern at gbspecialstretch.py:2987-2992."""
        i1 = np.array([0, 2])
        i2 = np.array([1, 3])
        # Reference: in-place swap on the expected ndarray.
        tmp_ref = self.expected[i1].copy()
        self.expected[i1] = self.expected[i2]
        self.expected[i2] = tmp_ref
        # Same operations on the view.
        tmp = self.view[i1].copy()
        self.view[i1] = self.view[i2]
        self.view[i2] = tmp
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_boolean_mask_read(self):
        mask = np.array([True, False, True, False])
        np.testing.assert_array_equal(self.view[mask], self.expected[mask])

    def test_boolean_mask_write(self):
        mask = np.array([False, True, True, False])
        payload = np.full(
            (int(mask.sum()),) + tuple(self.expected.shape[1:]),
            -1.5, dtype=complex,
        )
        self.view[mask] = payload
        self.expected[mask] = payload
        np.testing.assert_array_equal(self.view.gather(), self.expected)


class TupleFancyIndexTest(unittest.TestCase):
    """Tuple-fancy ``view[band_idx, intra1, intra2, ...]`` routing.

    Verifies the BandView's ``_fancy_get`` / ``_fancy_set`` paths match
    NumPy's tuple-fancy indexing on a reference ndarray of the same
    shape. The test mirrors the access pattern in
    ``Buffer.fill_buffer_residual_and_psd_from_acs``: a 3-tuple of
    broadcasted index arrays picking ``(band, channel, freq)`` slabs.
    """

    NUM_ACS = 5

    def setUp(self):
        try:
            self.aca, _ = _make_aca_and_reference(num_acs=self.NUM_ACS)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools test deps not installed: {exc}")
        # Synthesise a (num_acs, nchannels, data_length) reference array
        # by gathering the view; that way both view and reference start
        # in the same state.
        self.view = self.aca.data_shaped_view()
        self.ref = np.asarray(self.view.gather())

    def _make_tuple_index(self, band_inds_fill: np.ndarray, start_inds: np.ndarray):
        """Reproduce the (inds1, inds2, inds3) tuple from gbspecialstretch's
        ``_get_fill_buffer_ind_map`` for the FD non-XYZ path.

        Shapes:
        - inds1: (len(band_inds_fill), 1, 1)
        - inds2: (1, nchannels, 1)
        - inds3: (len(band_inds_fill), 1, sub_len)
        """
        nchannels = self.ref.shape[1]
        sub_len = 4  # arbitrary, must be <= data_length
        inds1 = band_inds_fill[:, None, None]
        inds2 = np.arange(nchannels)[None, :, None]
        inds3 = start_inds[:, None, None] + np.arange(sub_len)[None, None, :]
        return (inds1, inds2, inds3), sub_len

    def test_tuple_fancy_get_matches_reference(self):
        band_inds_fill = np.array([0, 2, 4])
        start_inds = np.array([1, 0, 3])
        idx_tuple, sub_len = self._make_tuple_index(band_inds_fill, start_inds)
        out_view = self.view[idx_tuple]
        out_ref = self.ref[idx_tuple]
        self.assertEqual(out_view.shape, out_ref.shape)
        np.testing.assert_array_equal(out_view, out_ref)

    def test_tuple_fancy_set_scalar(self):
        band_inds_fill = np.array([1, 3])
        start_inds = np.array([2, 5])
        idx_tuple, _ = self._make_tuple_index(band_inds_fill, start_inds)
        self.view[idx_tuple] = 0.0
        self.ref[idx_tuple] = 0.0
        np.testing.assert_array_equal(self.view.gather(), self.ref)

    def test_tuple_fancy_set_broadcastable_payload(self):
        band_inds_fill = np.array([0, 4])
        start_inds = np.array([0, 0])
        idx_tuple, sub_len = self._make_tuple_index(band_inds_fill, start_inds)
        nchannels = self.ref.shape[1]
        payload = np.full((2, nchannels, sub_len), -3.25, dtype=complex)
        self.view[idx_tuple] = payload
        self.ref[idx_tuple] = payload
        np.testing.assert_array_equal(self.view.gather(), self.ref)

    def test_tuple_fancy_inplace_iadd_matches_reference(self):
        """Mirror the ``buffer[inds_fill] += outer_view[inds_get_data]`` path."""
        band_inds_fill = np.array([0, 1, 4])
        start_inds = np.array([0, 2, 1])
        idx_tuple, sub_len = self._make_tuple_index(band_inds_fill, start_inds)
        nchannels = self.ref.shape[1]
        delta = np.full((3, nchannels, sub_len), 0.5, dtype=complex)
        # On the view: tuple-fancy in-place add
        self.view[idx_tuple] += delta
        self.ref[idx_tuple] += delta
        np.testing.assert_array_equal(self.view.gather(), self.ref)


if __name__ == "__main__":
    unittest.main()
