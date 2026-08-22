"""Multi-shard routing structural test for :class:`BandView`.

CPU runs always live on a single shard; we cannot construct a true
multi-GPU ACA on a CPU-only host because ``AnalysisContainerArray``
enters ``cp.cuda.Device`` contexts when ``gpus`` is set. Instead, we
exercise the BandView routing logic directly against a synthetic ACA
stand-in whose ``gpu_map`` / ``gpu_splits`` / ``split_map`` /
``data_shaped`` / ``psd_shaped`` / ``xp`` / ``gpus`` look multi-shard
but whose backing arrays are NumPy.

This is enough to verify the per-band index → shard routing
(``_resolve_array`` and friends) end to end. The actual
``cp.cuda.Device(gpu)`` context entry is just a no-op pass-through on
CPU; the numerical result is identical to the multi-shard path on a
real GPU.
"""

from __future__ import annotations

import unittest

import numpy as np


class _FakeMultiShardACA:
    """Minimal duck-typed ACA for BandView routing tests.

    Exposes only the attributes :class:`BandView` reads:
    ``acs_total_entries``, ``gpu_map``, ``gpu_splits``, ``split_map``,
    ``gpus``, ``xp``, ``data_shaped``, ``psd_shaped``.

    Backing arrays are NumPy and live in two synthetic per-shard lists,
    each shaped ``(num_acs_on_shard, *per_band_shape)``.
    """

    class _FakeXp:
        class _FakeCuda:
            class _FakeRuntime:
                @staticmethod
                def getDevice():
                    return 0

                @staticmethod
                def setDevice(_gpu):
                    return None

            runtime = _FakeRuntime()

            @staticmethod
            def Device(_gpu):
                class _Ctx:
                    def __enter__(self_inner):
                        return None

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                return _Ctx()

        cuda = _FakeCuda()

        @staticmethod
        def asarray(x):
            return np.asarray(x)

        @staticmethod
        def zeros(*args, **kwargs):
            return np.zeros(*args, **kwargs)

    xp = _FakeXp()

    def __init__(self, per_band_shape: tuple, num_acs: int, num_shards: int):
        self.acs_total_entries = num_acs
        self.gpus = list(range(num_shards))
        self.gpu_map = np.array([b % num_shards for b in range(num_acs)], dtype=int)
        self.gpu_splits = [
            np.where(self.gpu_map == s)[0] for s in self.gpus
        ]
        self.split_map = np.zeros(num_acs, dtype=int)
        for s_i, split in enumerate(self.gpu_splits):
            self.split_map[split] = s_i
        # Per-shard data_shaped: (n_acs_on_shard, *per_band_shape).
        self._data_shards = [
            np.zeros((len(self.gpu_splits[s]),) + per_band_shape, dtype=complex)
            for s in range(num_shards)
        ]
        # Seed each band with a unique constant so we can verify routing.
        for s_i, split in enumerate(self.gpu_splits):
            for intra, ac_i in enumerate(split):
                self._data_shards[s_i][intra] = float(int(ac_i) + 1)

    @property
    def data_shaped(self):
        return self._data_shards

    @property
    def psd_shaped(self):
        # We only need data_shaped for these tests; psd_shaped points at
        # the same buffers so the kind="psd" code path is exercised too.
        return self._data_shards


def _expected_from_fake(aca):
    """Reconstruct the (num_acs, *per_band_shape) reference."""
    per_band = aca._data_shards[0].shape[1:]
    out = np.zeros((aca.acs_total_entries,) + per_band, dtype=complex)
    for s_i, split in enumerate(aca.gpu_splits):
        for intra, ac_i in enumerate(split):
            out[int(ac_i)] = aca._data_shards[s_i][intra]
    return out


class MultiShardRoutingTest(unittest.TestCase):
    PER_BAND = (3, 8)  # nchannels, data_length
    NUM_ACS = 6
    NUM_SHARDS = 2

    def setUp(self):
        try:
            from lisatools.analysiscontainer import BandView
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools BandView not available: {exc}")
        self.BandView = BandView
        self.aca = _FakeMultiShardACA(self.PER_BAND, self.NUM_ACS, self.NUM_SHARDS)
        self.view = BandView(self.aca, kind="data")
        self.expected = _expected_from_fake(self.aca)

    def test_striped_layout(self):
        for b in range(self.NUM_ACS):
            self.assertEqual(int(self.aca.gpu_map[b]), b % self.NUM_SHARDS)

    def test_gather_matches_reference(self):
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_scalar_index_read_each_band(self):
        for b in range(self.NUM_ACS):
            np.testing.assert_array_equal(self.view[b], self.expected[b])

    def test_array_index_read_cross_shard(self):
        idx = np.array([0, 1, 4, 3])  # mixes shard 0 and shard 1
        np.testing.assert_array_equal(self.view[idx], self.expected[idx])

    def test_array_index_write_scalar_cross_shard(self):
        idx = np.array([1, 2, 5])
        self.view[idx] = 0.0
        self.expected[idx] = 0.0
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_array_index_write_per_row_cross_shard(self):
        idx = np.array([0, 3, 5])
        payload = np.full((len(idx),) + self.PER_BAND, 7.5, dtype=complex)
        self.view[idx] = payload
        self.expected[idx] = payload
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_swap_across_shards(self):
        i1 = np.array([0, 2])  # shard 0
        i2 = np.array([1, 3])  # shard 1
        ref_tmp = self.expected[i1].copy()
        self.expected[i1] = self.expected[i2]
        self.expected[i2] = ref_tmp
        tmp = self.view[i1].copy()
        self.view[i1] = self.view[i2]
        self.view[i2] = tmp
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_inplace_iadd_across_shards(self):
        idx = np.array([0, 1, 4])
        delta = np.full((len(idx),) + self.PER_BAND, 1.25, dtype=complex)
        self.view[idx] += delta
        self.expected[idx] += delta
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_tuple_fancy_get_across_shards(self):
        """Tuple-fancy ``view[band_inds, chan_inds, freq_inds]`` routes per band."""
        nchannels, data_len = self.PER_BAND
        band_inds = np.array([0, 3, 4])  # shards: 0, 1, 0
        start_inds = np.array([0, 1, 2])
        inds1 = band_inds[:, None, None]
        inds2 = np.arange(nchannels)[None, :, None]
        sub_len = 3
        inds3 = start_inds[:, None, None] + np.arange(sub_len)[None, None, :]
        out_view = self.view[(inds1, inds2, inds3)]
        out_ref = self.expected[inds1, inds2, inds3]
        self.assertEqual(out_view.shape, out_ref.shape)
        np.testing.assert_array_equal(out_view, out_ref)

    def test_tuple_fancy_set_across_shards(self):
        nchannels, data_len = self.PER_BAND
        band_inds = np.array([1, 2, 5])  # shards: 1, 0, 1
        start_inds = np.array([0, 2, 1])
        sub_len = 3
        inds1 = band_inds[:, None, None]
        inds2 = np.arange(nchannels)[None, :, None]
        inds3 = start_inds[:, None, None] + np.arange(sub_len)[None, None, :]
        payload = np.full((3, nchannels, sub_len), 11.0, dtype=complex)
        self.view[(inds1, inds2, inds3)] = payload
        self.expected[inds1, inds2, inds3] = payload
        np.testing.assert_array_equal(self.view.gather(), self.expected)

    def test_tuple_fancy_iadd_across_shards(self):
        """Mirror the fill_buffer_residual_and_psd_from_acs in-place += pattern."""
        nchannels, data_len = self.PER_BAND
        band_inds = np.array([0, 4, 5])
        start_inds = np.array([1, 0, 2])
        sub_len = 3
        inds1 = band_inds[:, None, None]
        inds2 = np.arange(nchannels)[None, :, None]
        inds3 = start_inds[:, None, None] + np.arange(sub_len)[None, None, :]
        delta = np.full((3, nchannels, sub_len), 0.5, dtype=complex)
        self.view[(inds1, inds2, inds3)] += delta
        self.expected[inds1, inds2, inds3] += delta
        np.testing.assert_array_equal(self.view.gather(), self.expected)


class MultiShardSlotSubsetTest(unittest.TestCase):
    """``SubBandBuffer._likelihood_slots`` routing across shards.

    The tempering empty-cell skip (``GB_TEMPER_SKIP_EMPTY``) scores a SLOT
    SUBSET instead of the whole buffer, which means the per-slot reduction
    has to find each requested slot on its owning shard and return the
    values in the REQUESTED order. That routing is what is checked here.

    The reduction passed in is a marker, not a likelihood: the fake ACA
    seeds band ``b`` with the constant ``b + 1``, so a reduction that reads
    one element per row reports which band each reduced row came from.
    The assertion is therefore purely about slot -> shard -> row bookkeeping
    (``gpu_splits`` / ``split_map``), with no numerics duplicated from the
    likelihood path.
    """

    PER_BAND = (3, 8)
    NUM_ACS = 6
    NUM_SHARDS = 2

    def setUp(self):
        try:
            from lisatools.analysiscontainer import BandView
            from lisatools.globalfit.moves.gbbands import SubBandBuffer
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools BandView not available: {exc}")
        self.SubBandBuffer = SubBandBuffer
        self.aca = _FakeMultiShardACA(self.PER_BAND, self.NUM_ACS, self.NUM_SHARDS)
        # CPU host: no device contexts (the shard routing itself is
        # unchanged by that -- see this module's docstring).
        self.aca.gpus = None
        self.view = BandView(self.aca, kind="data")

    @staticmethod
    def _marker(num, psd):
        k = num.shape[0]
        return num.reshape(k, -1)[:, 0].real

    def _slots(self, slots, chunk):
        from types import SimpleNamespace

        return np.asarray(
            self.SubBandBuffer._likelihood_slots(
                SimpleNamespace(xp=np),
                np.asarray(slots, dtype=int),
                self.view, self.view, None, self._marker, chunk,
            )
        )

    def test_subset_rows_and_order(self):
        for slots in ([0], [5], [1, 3, 5], [5, 0, 3], [0, 1, 2, 3, 4, 5],
                      [4, 4, 1]):
            for chunk in (1, 2, 64):
                got = self._slots(slots, chunk)
                np.testing.assert_array_equal(
                    got, np.asarray(slots, dtype=float) + 1.0,
                    err_msg=f"slots={slots} chunk={chunk}",
                )

    def test_empty_subset(self):
        self.assertEqual(self._slots([], 4).shape, (0,))

    def test_subset_matches_full_pass(self):
        """A subset carries the same per-row values a full pass would."""
        full = self._slots(list(range(self.NUM_ACS)), 64)
        sub = [4, 0, 3]
        np.testing.assert_array_equal(self._slots(sub, 2), full[sub])


if __name__ == "__main__":
    unittest.main()
