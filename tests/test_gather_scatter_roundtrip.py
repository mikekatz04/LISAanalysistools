"""Gather -> mutate -> scatter roundtrip for the ACA flat-buffer API.

Regression for the WDM init-subtraction sites (recipe.py neighbor/GB/VGB
subtraction): on multi-shard ACAs ``gather_linear_data_arr`` returns a
COPY, so writes must be pushed back with ``scatter_linear_data_arr`` or
they silently vanish. The real :class:`AnalysisContainerArray` methods are
exercised unbound against the NumPy :class:`FakeMultiShardACA` (the device
contexts they enter are recorded no-ops on CPU).
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:
    from _multishard import FakeMultiShardACA


class GatherScatterRoundtripTest(unittest.TestCase):
    PER_BAND = (3, 8)
    NUM_ACS = 7
    NUM_SHARDS = 3

    def setUp(self):
        try:
            from lisatools.analysiscontainer import AnalysisContainerArray
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"lisatools ACA not available: {exc}")
        self.ACA = AnalysisContainerArray
        self.aca = FakeMultiShardACA(
            self.PER_BAND, self.NUM_ACS, self.NUM_SHARDS, layout="blocked"
        )

    def _gather(self):
        return self.ACA._gather_per_gpu_to_single(
            self.aca, self.aca.linear_data_arr
        )

    def _scatter(self, flat):
        self.ACA._scatter_single_to_per_gpu(
            self.aca, flat, self.aca.linear_data_arr
        )

    def test_gather_is_copy_on_multi_shard(self):
        flat = self._gather()
        for buf in self.aca.linear_data_arr:
            self.assertIsNot(flat, buf)
        # mutating the gathered array does NOT reach the shards ...
        before = [buf.copy() for buf in self.aca.linear_data_arr]
        flat[:] = -1.0
        for buf, ref in zip(self.aca.linear_data_arr, before):
            np.testing.assert_array_equal(buf, ref)

    def test_roundtrip_lands_in_shards(self):
        per_row = int(np.prod(self.PER_BAND))
        flat = self._gather()
        # global AC order: row b holds constant b+1
        for b in range(self.NUM_ACS):
            np.testing.assert_array_equal(
                flat[b * per_row:(b + 1) * per_row],
                np.full(per_row, float(b + 1), dtype=complex),
            )
        # subtract a per-row template (the fill_global_wdm pattern) ...
        flat = flat - 0.5
        self._scatter(flat)
        # ... and the write lands in every shard, in intra-shard order
        ref = self.aca.reference_rows()
        for b in range(self.NUM_ACS):
            np.testing.assert_array_equal(
                ref[b], np.full(self.PER_BAND, float(b + 1) - 0.5,
                                dtype=complex),
            )

    def test_scatter_preserves_buffer_identity(self):
        """Shard buffers are written IN PLACE, never reallocated
        (memory-lifecycle rule) — views into them stay valid."""
        ids_before = [id(buf) for buf in self.aca.linear_data_arr]
        flat = self._gather()
        flat[:] = 3.25
        self._scatter(flat)
        self.assertEqual(
            ids_before, [id(buf) for buf in self.aca.linear_data_arr]
        )
        for buf in self.aca.linear_data_arr:
            np.testing.assert_array_equal(
                buf, np.full(buf.shape, 3.25, dtype=complex))

    def test_single_shard_fast_path_no_copy(self):
        single = FakeMultiShardACA(self.PER_BAND, 4, 1, layout="blocked")
        flat = self.ACA._gather_per_gpu_to_single(
            single, single.linear_data_arr
        )
        # single-shard gather returns the buffer itself ...
        self.assertIs(flat, single.linear_data_arr[0])
        flat[:] = 9.0
        # ... and scattering the same object is a no-op that keeps values
        self.ACA._scatter_single_to_per_gpu(
            single, flat, single.linear_data_arr
        )
        np.testing.assert_array_equal(
            single.linear_data_arr[0], np.full(flat.shape, 9.0,
                                               dtype=complex))

    def test_scatter_enters_owning_device_contexts(self):
        flat = self._gather()
        self.aca.xp.device_log.clear()
        self._scatter(flat)
        # one context entry per non-empty shard, each with the owning id
        self.assertEqual(
            sorted(set(self.aca.xp.device_log)),
            [s for s, rows in enumerate(self.aca.gpu_splits) if len(rows)],
        )


if __name__ == "__main__":
    unittest.main()
