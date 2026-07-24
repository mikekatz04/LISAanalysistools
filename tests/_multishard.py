"""Shared CPU stand-ins for multi-shard (multi-GPU) structural tests.

CPU hosts cannot build a true multi-GPU ``AnalysisContainerArray`` (it
enters ``cp.cuda.Device`` contexts when ``gpus`` is set), so multi-shard
logic is exercised against duck-typed fakes whose ``gpu_map`` /
``gpu_splits`` / ``split_map`` / linear buffers look multi-shard but whose
backing arrays are NumPy. Device-context entries are recorded (not
executed), so tests can assert WHICH device each call ran under.

Promoted from ``test_band_view_multi_shard.py`` and extended with the flat
``linear_data_arr`` / ``linear_psd_arr`` protocol the gbgpu band engines
and the LAT shard router consume.
"""

from __future__ import annotations

import numpy as np


class RecordingXp:
    """NumPy-backed fake ``xp`` with a cupy-like ``cuda`` namespace.

    ``cuda.Device(i)`` context entries push/pop a current-device stack and
    append to :attr:`device_log`, so a test can assert the device each
    operation was routed to.
    """

    __name__ = "numpy"

    def __init__(self):
        self.device_log = []
        self._current = [0]

        outer = self

        class _Runtime:
            @staticmethod
            def getDevice():
                return outer._current[-1]

            @staticmethod
            def setDevice(gpu):
                outer._current[-1] = int(gpu)

            @staticmethod
            def deviceSynchronize():
                return None

        class _Cuda:
            runtime = _Runtime()

            class device:
                @staticmethod
                def Device(gpu):
                    return outer._device_ctx(gpu)

            @staticmethod
            def Device(gpu):
                return outer._device_ctx(gpu)

        self.cuda = _Cuda()

    def _device_ctx(self, gpu):
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                outer._current.append(int(gpu))
                outer.device_log.append(int(gpu))
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                outer._current.pop()
                return False

        return _Ctx()

    @property
    def current_device(self):
        return self._current[-1]

    # numpy passthrough for everything else (asarray, zeros, where, ...)
    def __getattr__(self, name):
        return getattr(np, name)


class FakeMultiShardACA:
    """Duck-typed multi-shard ACA over NumPy arrays.

    Exposes the attributes consumed by :class:`~lisatools.analysiscontainer.BandView`,
    the LAT shard router (``_ShardHolderView`` / ``_RoutedBandEngine``), and
    the ACA gather/scatter helpers: ``acs_total_entries``, ``gpus``,
    ``gpu_map``, ``gpu_splits``, ``split_map``, ``xp`` (a
    :class:`RecordingXp`), ``linear_data_arr``, ``linear_psd_arr``,
    ``data_shaped``, ``psd_shaped``, and (optionally) ``min_freq_inds``.

    Args:
        per_band_shape: per-row shape, e.g. ``(nchannels, data_length)``.
        num_acs: total rows (cells / walkers).
        num_shards: shard count; ``gpus = list(range(num_shards))``.
        layout: ``"striped"`` (row ``b`` on shard ``b % num_shards``) or
            ``"blocked"`` (contiguous ``np.array_split`` blocks, possibly
            uneven — the main-ACA walker layout).
        with_min_freq_inds: attach a per-row int32 ``min_freq_inds`` store.
    """

    def __init__(self, per_band_shape: tuple, num_acs: int, num_shards: int,
                 layout: str = "striped", with_min_freq_inds: bool = False,
                 dtype=complex):
        self.xp = RecordingXp()
        self.acs_total_entries = int(num_acs)
        self.gpus = list(range(num_shards))
        if layout == "striped":
            self.gpu_map = np.array(
                [b % num_shards for b in range(num_acs)], dtype=int
            )
            self.gpu_splits = [
                np.where(self.gpu_map == s)[0] for s in range(num_shards)
            ]
        elif layout == "blocked":
            self.gpu_splits = [
                np.asarray(chunk, dtype=int)
                for chunk in np.array_split(np.arange(num_acs), num_shards)
            ]
            self.gpu_map = np.empty(num_acs, dtype=int)
            for s, rows in enumerate(self.gpu_splits):
                self.gpu_map[rows] = s
        else:
            raise ValueError(f"unknown layout {layout!r}")
        self.split_map = np.zeros(num_acs, dtype=int)
        for s_i, split in enumerate(self.gpu_splits):
            self.split_map[split] = s_i

        self.per_band_shape = tuple(per_band_shape)
        per_row = int(np.prod(per_band_shape))
        # Flat per-shard buffers (row-major in intra-shard order), seeded so
        # row ``b`` holds the constant ``b + 1`` — routing is verifiable.
        self.linear_data_arr = []
        self.linear_psd_arr = []
        for rows in self.gpu_splits:
            buf = np.zeros(len(rows) * per_row, dtype=dtype)
            psd = np.zeros(len(rows) * per_row, dtype=float)
            for intra, ac_i in enumerate(rows):
                buf[intra * per_row:(intra + 1) * per_row] = float(ac_i + 1)
                psd[intra * per_row:(intra + 1) * per_row] = 1.0
            self.linear_data_arr.append(buf)
            self.linear_psd_arr.append(psd)

        self.min_freq_inds = (
            np.arange(100, 100 + num_acs, dtype=np.int32)
            if with_min_freq_inds else None
        )
        self.start_freq_ind = np.arange(num_acs, dtype=np.int32) * 10

    @property
    def data_shaped(self):
        return [
            buf.reshape((len(rows),) + self.per_band_shape)
            for buf, rows in zip(self.linear_data_arr, self.gpu_splits)
        ]

    @property
    def psd_shaped(self):
        return [
            buf.reshape((len(rows),) + self.per_band_shape)
            for buf, rows in zip(self.linear_psd_arr, self.gpu_splits)
        ]

    def reference_rows(self):
        """(num_acs, *per_band_shape) reference in global row order."""
        out = np.zeros(
            (self.acs_total_entries,) + self.per_band_shape,
            dtype=self.linear_data_arr[0].dtype,
        )
        for s_i, split in enumerate(self.gpu_splits):
            shaped = self.data_shaped[s_i]
            for intra, ac_i in enumerate(split):
                out[int(ac_i)] = shaped[intra]
        return out
