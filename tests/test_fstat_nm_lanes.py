"""All-device fan-out for the per-row F-stat (N, M) scorer.

Production telemetry (v7 snapshot 2, 2026-08-27): `route_fstat_ll`
partitions by walker and every row carries the ONE reference walker, so
the whole per-row center chain ran on that walker's device while the
other GPU idled (43% of gb_search samples GPU1-only). The fix mirrors
the validated `_sighet_fstat_multidevice` pattern: replicate the
reference walker's residual + inverse-PSD rows onto every device once
(`_FStatRefRowHolder`), then split each batch into contiguous per-device
lanes. `GB_FSTAT_NM_MULTIDEV=check` shadow-scores against the pinned
single-device path and raises on the first divergence.

CPU tests exercise routing/merge/check semantics against the
tests/_multishard.py fakes; real-GPU parity is the on-cluster `=check`
gate.
"""

import unittest

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:
    from _multishard import FakeMultiShardACA


class _DevComp:
    """Device-encoding stand-in: N/M rows encode 1000*device + data_index
    so WHERE each row ran is checkable (routing test; deliberately NOT
    device-independent)."""

    def __init__(self):
        self.calls = []

    def get_fstat_ll_wdm(self, params, wdm_holder, data_index=None,
                         noise_index=None, **kwargs):
        assert len(wdm_holder.linear_data_arr) == 1
        assert len(wdm_holder) == wdm_holder.acs_total_entries
        intra = np.asarray(data_index)
        dev = wdm_holder.gpus[0] if wdm_holder.gpus is not None else 0
        self.calls.append(dict(holder=wdm_holder, intra=intra.copy(),
                               n=np.asarray(params).shape[0],
                               kwargs=dict(kwargs)))
        base = 1000.0 * dev + intra.astype(float)
        return (base[:, None] + np.arange(4)[None, :],
                base[:, None] + np.arange(10)[None, :])


class _UniformComp:
    """Device-INDEPENDENT stand-in (like a real comp replica): output is a
    pure function of params, so the fan-out must bit-match the pinned
    path."""

    def get_fstat_ll_wdm(self, params, wdm_holder, data_index=None,
                         noise_index=None, **kwargs):
        p = np.atleast_2d(np.asarray(params, dtype=float))
        return (p[:, :1] + np.arange(4)[None, :],
                p[:, :1] + np.arange(10)[None, :])


class FStatNMLanesTest(unittest.TestCase):
    N_WALKERS = 6
    N_SHARDS = 2
    WALKER_REF = 4  # blocked layout -> shard 1, intra 1, row constant 5.0

    def setUp(self):
        try:
            from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"gbbands router not available: {exc}")
        self.RoutedEngine = _RoutedBandEngine
        self.holder = FakeMultiShardACA(
            (3, 8), self.N_WALKERS, self.N_SHARDS, layout="blocked")
        self.params = np.arange(10 * 9, dtype=float).reshape(10, 9)

    def _make(self, comp, **kw):
        return self.RoutedEngine.make_fstat_nm_lanes(
            comp, "get_fstat_ll_wdm", self.holder, self.WALKER_REF, **kw)

    def test_single_shard_returns_none(self):
        single = FakeMultiShardACA((3, 8), 4, 1, layout="blocked")
        comp = _DevComp()
        self.assertIsNone(self.RoutedEngine.make_fstat_nm_lanes(
            comp, "get_fstat_ll_wdm", single, 0))

    def test_rows_split_contiguously_across_devices(self):
        comp = _DevComp()
        call = self._make(comp, convert_to_ra_dec=False)
        self.assertIsNotNone(call)
        N, M = call(self.params)
        N, M = np.asarray(N), np.asarray(M)
        # contiguous near-equal split: rows [0:5) on device 0 (base 0),
        # rows [5:10) on device 1 (base 1000); every row scored with
        # data_index=0 against a one-slab ref holder.
        expect_base = np.array([0.0] * 5 + [1000.0] * 5)
        np.testing.assert_array_equal(
            N, expect_base[:, None] + np.arange(4)[None, :])
        np.testing.assert_array_equal(
            M, expect_base[:, None] + np.arange(10)[None, :])
        devs = sorted(c["holder"].gpus[0] for c in comp.calls)
        self.assertEqual(devs, [0, 1])
        for c in comp.calls:
            np.testing.assert_array_equal(c["intra"],
                                          np.zeros(c["n"], dtype=int))
            self.assertIs(c["kwargs"]["convert_to_ra_dec"], False)

    def test_ref_rows_snapshot_from_owning_walker(self):
        # the fake seeds walker b's row with the constant b+1; walker 4 -> 5.0
        comp = _DevComp()
        call = self._make(comp)
        call(self.params[:2])
        for c in comp.calls:
            row = np.asarray(c["holder"].linear_data_arr[0])
            self.assertTrue(np.all(row == 5.0),
                            "ref holder must carry walker 4's residual row")

    def test_check_mode_passes_for_device_independent_comp(self):
        comp = _UniformComp()
        call = self._make(comp, check=True)
        N, M = call(self.params)
        p = self.params
        np.testing.assert_array_equal(
            np.asarray(N), p[:, :1] + np.arange(4)[None, :])
        np.testing.assert_array_equal(
            np.asarray(M), p[:, :1] + np.arange(10)[None, :])

    def test_check_mode_raises_on_divergence(self):
        comp = _DevComp()  # device-dependent -> lanes != pinned shadow
        call = self._make(comp, check=True)
        with self.assertRaises(RuntimeError):
            call(self.params)


if __name__ == "__main__":
    unittest.main()
