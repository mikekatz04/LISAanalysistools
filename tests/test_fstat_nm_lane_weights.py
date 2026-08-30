"""Weighted lane split for the multi-device F-stat (N, M) scorer.

``make_fstat_nm_lanes`` splits each ``GB_FSTAT_CTR_BATCH`` batch into
near-equal CONTIGUOUS row lanes, one per run device, and joins on the
slowest. The two devices are not symmetric in production: on the
2026-08-29 3-month v7 restart GPU0 ran at 39.1% mean utilisation with a
90,698 MiB peak while GPU1 ran at 72.1% with a 70,020 MiB peak, yet the
split was exactly 50/50 (``bounds = (n * arange(L+1)) // L``), so every
one of the ~1,409 joins per propose waited on the busier card.

``GB_FSTAT_NM_LANE_WEIGHTS`` makes the split proportional to integer
per-lane weights. INTEGER cumulative arithmetic, not float shares, so
the all-equal default reproduces ``(n * arange(L+1)) // L`` EXACTLY --
the lane boundaries are what decide which device scores which row, and
the default must stay bit-identical to the pinned path.
"""
import unittest

import numpy as np

from lisatools.globalfit.moves.gbbands import fstat_nm_lane_bounds


def _legacy(n, lanes):
    """The formula this replaces, verbatim from make_fstat_nm_lanes."""
    return (n * np.arange(lanes + 1)) // lanes


class DefaultIsBitIdenticalTest(unittest.TestCase):
    """No weights (or all-equal weights) must reproduce the old split."""

    def test_matches_legacy_formula_for_every_small_n(self):
        for lanes in (1, 2, 3, 4, 8):
            for n in range(0, 200):
                np.testing.assert_array_equal(
                    fstat_nm_lane_bounds(n, lanes, None),
                    _legacy(n, lanes),
                    err_msg=f"n={n} lanes={lanes}",
                )

    def test_matches_legacy_formula_at_production_batch_sizes(self):
        for n in (4096, 8192, 16384, 65536, 5753053, 4083, 4095, 4097):
            for lanes in (2, 4):
                np.testing.assert_array_equal(
                    fstat_nm_lane_bounds(n, lanes, None), _legacy(n, lanes)
                )

    def test_explicit_equal_weights_match_legacy(self):
        for lanes in (2, 3, 4):
            for n in (0, 1, 7, 4096, 5753053):
                np.testing.assert_array_equal(
                    fstat_nm_lane_bounds(n, lanes, [1] * lanes),
                    _legacy(n, lanes),
                )


class WeightedSplitTest(unittest.TestCase):
    def test_weights_shift_rows_toward_the_heavier_lane(self):
        n = 1000
        b = fstat_nm_lane_bounds(n, 2, [3, 1])
        self.assertEqual(list(b), [0, 750, 1000])

    def test_weights_are_proportional_and_cover_every_row(self):
        for n in (0, 1, 2, 999, 4096, 5753053):
            for w in ([3, 1], [2, 1], [7, 5], [1, 1, 2], [5, 3, 1, 1]):
                b = fstat_nm_lane_bounds(n, len(w), w)
                self.assertEqual(len(b), len(w) + 1)
                self.assertEqual(int(b[0]), 0)
                self.assertEqual(int(b[-1]), n)
                # contiguous, non-overlapping, ascending, complete
                self.assertTrue(np.all(np.diff(b) >= 0))
                self.assertEqual(int(np.diff(b).sum()), n)

    def test_lane_share_tracks_the_weight_at_scale(self):
        n, w = 5_753_053, [3, 1]
        b = fstat_nm_lane_bounds(n, 2, w)
        got = np.diff(b) / n
        np.testing.assert_allclose(got, [0.75, 0.25], atol=1e-6)

    def test_a_zero_weight_lane_gets_no_rows(self):
        b = fstat_nm_lane_bounds(1000, 2, [1, 0])
        self.assertEqual(list(b), [0, 1000, 1000])
        self.assertEqual(int(np.diff(b)[1]), 0)


class WeightParsingTest(unittest.TestCase):
    """``GB_FSTAT_NM_LANE_WEIGHTS`` is operator input -- never crash a run."""

    def _b(self, spec, n=1000, lanes=2):
        return list(fstat_nm_lane_bounds(n, lanes, spec))

    def test_none_and_blank_fall_back_to_equal(self):
        for spec in (None, "", "   "):
            self.assertEqual(self._b(spec), list(_legacy(1000, 2)))

    def test_comma_string_is_accepted(self):
        self.assertEqual(self._b("3,1"), [0, 750, 1000])
        self.assertEqual(self._b(" 3 , 1 "), [0, 750, 1000])

    def test_wrong_length_falls_back_to_equal(self):
        for spec in ("1,2,3", "5"):
            self.assertEqual(self._b(spec), list(_legacy(1000, 2)))

    def test_garbage_falls_back_to_equal(self):
        for spec in ("a,b", "1,x", "-1,2", "0,0", "1.5,2"):
            self.assertEqual(self._b(spec), list(_legacy(1000, 2)))


if __name__ == "__main__":
    unittest.main()
