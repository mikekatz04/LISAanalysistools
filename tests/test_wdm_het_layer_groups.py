"""Focused parity test for :func:`lisatools.wdm_het.compute_layer_groups`.

The 2026-08 perf change moved the group-boundary scan onto HOST numpy (one
D2H copy of two small int arrays instead of O(num_bin) implicit device
syncs). The algorithm is unchanged — this test pins that by comparing the
production function bit-for-bit against an independent pure-Python
reference reimplementation on cases with multiple ``data_index`` groups,
adjacent/duplicate carriers, and group splits at the ``group_band_layers``
threshold.
"""
import unittest

import numpy as np

from lisatools.wdm_het import compute_layer_groups


def _reference_layer_groups(params_array, layer_df, f0_param_index=1,
                            group_band_layers=5, margin_layers=0,
                            data_index_all=None):
    """Independent pure-Python reimplementation of the grouping spec.

    Sort by (data_index primary, carrier layer secondary) with a stable
    lexsort, then greedily cut groups: a group extends while the data index
    matches and the carrier layer stays within ``group_band_layers`` of the
    group's first carrier. The iterated m-band is a ``group_band_layers``
    window centred on the group's carriers (``half = group_band_layers//2``
    below the low carrier, ``group_band_layers - half`` above the high one)
    padded by ``margin_layers``.
    """
    params_array = np.asarray(params_array)
    num_bin = params_array.shape[0]
    f0 = params_array[:, int(f0_param_index)]
    m_floor = np.floor(f0 / float(layer_df)).astype(np.int32)
    if data_index_all is None:
        data_index_all = np.zeros(num_bin, dtype=np.int32)
    else:
        data_index_all = np.asarray(data_index_all, dtype=np.int32)

    order = np.lexsort((m_floor, data_index_all))
    half = int(group_band_layers) // 2

    groups = []
    i = 0
    while i < num_bin:
        m0 = int(m_floor[order[i]])
        d0 = int(data_index_all[order[i]])
        j = i
        while (j < num_bin
               and int(data_index_all[order[j]]) == d0
               and int(m_floor[order[j]]) - m0 < group_band_layers):
            j += 1
        groups.append(dict(
            start=i, end=j,
            m_lo=m0 - half - int(margin_layers),
            m_hi=(int(m_floor[order[j - 1]]) + (group_band_layers - half)
                  + int(margin_layers)),
            data_index=d0,
        ))
        i = j

    return dict(
        binary_perm=np.asarray(order, dtype=np.int32),
        group_starts=np.asarray([g["start"] for g in groups], dtype=np.int32),
        group_ends=np.asarray([g["end"] for g in groups], dtype=np.int32),
        group_m_lo=np.asarray([g["m_lo"] for g in groups], dtype=np.int32),
        group_m_hi=np.asarray([g["m_hi"] for g in groups], dtype=np.int32),
        group_data_index=np.asarray([g["data_index"] for g in groups],
                                    dtype=np.int32),
        n_groups=len(groups),
    )


def _make_params(m_layers, layer_df):
    """9-col GB-style params with f0 (col 1) placed mid-layer."""
    m_layers = np.asarray(m_layers, dtype=float)
    params = np.zeros((m_layers.shape[0], 9))
    params[:, 1] = (m_layers + 0.5) * layer_df
    return params


class LayerGroupHostScanParityTest(unittest.TestCase):
    LAYER_DF = 1.0 / 64.0

    def _check_case(self, m_layers, data_idx, group_band_layers=5,
                    margin_layers=0):
        params = _make_params(m_layers, self.LAYER_DF)
        data_idx = (None if data_idx is None
                    else np.asarray(data_idx, dtype=np.int32))
        got = compute_layer_groups(
            params, layer_df=self.LAYER_DF, f0_param_index=1,
            group_band_layers=group_band_layers,
            margin_layers=margin_layers,
            data_index_all=data_idx,
            noise_index_all=None if data_idx is None else data_idx.copy(),
        )
        ref = _reference_layer_groups(
            params, layer_df=self.LAYER_DF, f0_param_index=1,
            group_band_layers=group_band_layers,
            margin_layers=margin_layers,
            data_index_all=data_idx,
        )
        self.assertEqual(int(got["n_groups"]), int(ref["n_groups"]))
        for key in ("binary_perm", "group_starts", "group_ends",
                    "group_m_lo", "group_m_hi", "group_data_index"):
            got_arr = np.asarray(got[key])
            self.assertEqual(got_arr.dtype, np.int32, msg=key)
            np.testing.assert_array_equal(got_arr, ref[key], err_msg=key)

    def test_multi_data_groups_adjacent_carriers(self):
        # Three data groups, interleaved input order; adjacent carriers
        # (consecutive layers) and exact duplicates within groups.
        m = [10, 11, 30, 10, 12, 10, 11, 31, 33, 12, 12, 55]
        d = [0, 0, 1, 2, 0, 0, 2, 1, 1, 2, 0, 1]
        self._check_case(m, d)

    def test_group_split_at_band_threshold(self):
        # Carriers spread past group_band_layers force multiple groups
        # inside one data index; boundary at exactly m0 + band_layers.
        m = [10, 11, 12, 13, 14, 15, 16, 20, 21]
        d = [0] * len(m)
        self._check_case(m, d, group_band_layers=5)

    def test_margin_and_even_band(self):
        # Even group_band_layers (asymmetric half split) + nonzero margin.
        m = [7, 8, 9, 40, 41, 7, 90]
        d = [1, 1, 0, 0, 1, 0, 1]
        self._check_case(m, d, group_band_layers=4, margin_layers=2)

    def test_default_data_index(self):
        self._check_case([5, 6, 5, 9, 100], None)

    def test_single_binary(self):
        self._check_case([42], [3])


if __name__ == "__main__":
    unittest.main()
