"""The per-walker RJ valve's cross-walker lag rule (Mike 2026-10-17).

"let's make it neighbor aware" -- across walkers: a (walker, band) pair that
trails another walker by >= LAG_LEAVES cold leaves in that band is BEHIND.
A behind pair needs CONV_ITER + LAG_EXTRA flat judges to shut instead of
CONV_ITER, and on the first judge of a process (a relaunch) shut behind
pairs reopen with a fresh streak. Pinned here on ``_ColdBandLnL.judge``
directly, with the band_info fixture the persistence suite uses.

Why: on the 1yr flagship job 780 the valve shut 46 of walker 0's 58 pairs
at 1.5-2.5 mHz while the other walkers kept adding sources there; a shut
pair runs no RJ, so walker 0 fell 3,750 nats and 185 leaves behind.
"""
import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as g

NW, NB = 4, 6


def _band_info():
    import lisatools.globalfit.state as st
    bi = {"num_bands": NB, "nwalkers": NW}
    st.ensure_search_shutoff_window(bi, NB, NW)
    bi["band_rj_shutoff_w"] = np.zeros((NW, NB), dtype=bool)
    return bi


def _view(bi):
    """A view PRIMED by one judge: the first judge of a fresh window is an
    'improvement' from -inf (max -inf -> 0) that resets every streak, so
    the flat judges counted by the tests start after it."""
    v = g.cold_band_lnl_from_band_info(bi, (NW, NB))
    v.peak[:] = 0.0
    v.judge(4.0, 3, np.ones((NW, NB), dtype=np.int64), lag_leaves=0)
    assert int(v.streak.max()) == 0 and np.all(v.max == 0.0)
    return v


def _flat_judges(v, n, occ, **kw):
    """``n`` judges with no improvement (the peak never beats the max)."""
    out = None
    for _ in range(n):
        v.peak[:] = 0.0                       # the same gain every cycle
        out = v.judge(4.0, 3, occ, **kw)
    return out


class LagRuleTest(unittest.TestCase):

    def setUp(self):
        self.occ = np.ones((NW, NB), dtype=np.int64) * 5
        self.occ[0, 1] = 3                     # walker 0 trails by 2 in band 1
        self.occ[2, 4] = 4                     # walker 2 trails by 1 in band 4 (not behind)

    def test_old_behaviour_with_lag_disabled(self):
        bi = _band_info(); v = _view(bi)
        conv = _flat_judges(v, 3, self.occ, lag_leaves=0)
        self.assertTrue(conv.all())            # every occupied pair shuts at streak 3

    def test_a_behind_pair_needs_the_extra_judges(self):
        bi = _band_info(); v = _view(bi)
        conv = _flat_judges(v, 3, self.occ, lag_leaves=2, lag_extra=6)
        want = np.ones((NW, NB), dtype=bool); want[0, 1] = False
        np.testing.assert_array_equal(conv, want)
        self.assertEqual(v.lag_held, 1)        # the one pair held past the plain streak
        self.assertEqual(v.lag_reopened, 0)
        bi["band_rj_shutoff_w"][conv] = True   # the caller shuts the converged ones
        # 5 more flat judges: still held (streak 8 < 3 + 6)
        conv = _flat_judges(v, 5, self.occ, lag_leaves=2, lag_extra=6)
        self.assertFalse(conv[0, 1])
        self.assertFalse(bi["band_rj_shutoff_w"][0, 1])
        # the 9th flat judge shuts it: the patience is finite
        conv = _flat_judges(v, 1, self.occ, lag_leaves=2, lag_extra=6)
        self.assertTrue(conv[0, 1])
        self.assertEqual(v.lag_held, 0)

    def test_one_leaf_of_disagreement_is_not_behind(self):
        bi = _band_info(); v = _view(bi)
        conv = _flat_judges(v, 3, self.occ, lag_leaves=2, lag_extra=6)
        self.assertTrue(conv[2, 4])            # trails by 1 only: shuts normally

    def test_an_improvement_resets_the_behind_pairs_streak_like_any_other(self):
        bi = _band_info(); v = _view(bi)
        _flat_judges(v, 2, self.occ, lag_leaves=2, lag_extra=6)
        v.peak[:] = 0.0; v.peak[0, 1] = 50.0   # walker 0 catches up in band 1
        conv = v.judge(4.0, 3, self.occ, lag_leaves=2, lag_extra=6)
        self.assertEqual(int(v.streak[0, 1]), 0)
        self.assertFalse(conv[0, 1])

    def test_the_first_judge_of_a_process_reopens_shut_behind_pairs(self):
        bi = _band_info(); v = _view(bi)
        shut = bi["band_rj_shutoff_w"]
        shut[:] = True                         # everything shut, as at a relaunch
        v.streak[:] = 7
        conv = _flat_judges(v, 1, self.occ, lag_leaves=2, lag_extra=6, reopen_lagging=True)
        self.assertFalse(shut[0, 1])           # reopened: behind by 2
        self.assertEqual(int(v.streak[0, 1]), 0)
        self.assertTrue(shut[2, 4])            # behind by 1: stays shut
        self.assertTrue(shut[1, 1] and shut[3, 3])
        self.assertEqual(v.lag_reopened, 1)
        self.assertFalse(conv.any())           # a reopened pair does not re-shut the same judge
        # a later judge (not the first) never reopens
        shut[0, 1] = True
        _flat_judges(v, 1, self.occ, lag_leaves=2, lag_extra=6, reopen_lagging=False)
        self.assertTrue(shut[0, 1])
        self.assertEqual(v.lag_reopened, 0)

    def test_an_empty_pair_is_not_reopened_under_require_occ(self):
        bi = _band_info(); v = _view(bi)
        occ = self.occ.copy(); occ[0, 1] = 0   # walker 0 holds nothing there
        shut = bi["band_rj_shutoff_w"]; shut[:] = True
        _flat_judges(v, 1, occ, lag_leaves=2, lag_extra=6, reopen_lagging=True)
        self.assertTrue(shut[0, 1])
        _flat_judges(v, 1, occ, lag_leaves=2, lag_extra=6, reopen_lagging=True, require_occ=False)
        self.assertFalse(shut[0, 1])

    def test_the_knobs(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("GB_SEARCH_BAND_SHUTOFF_LAG_LEAVES", "GB_SEARCH_BAND_SHUTOFF_LAG_EXTRA_JUDGES")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(g.GBSpecialBase._shutoff_lag_leaves(), 2)
            self.assertEqual(g.GBSpecialBase._shutoff_lag_extra_judges(), 6)
        with mock.patch.dict(os.environ, {"GB_SEARCH_BAND_SHUTOFF_LAG_LEAVES": "0",
                                          "GB_SEARCH_BAND_SHUTOFF_LAG_EXTRA_JUDGES": "9"}):
            self.assertEqual(g.GBSpecialBase._shutoff_lag_leaves(), 0)
            self.assertEqual(g.GBSpecialBase._shutoff_lag_extra_judges(), 9)
        with mock.patch.dict(os.environ, {"GB_SEARCH_BAND_SHUTOFF_LAG_LEAVES": "-1"}):
            with self.assertRaises(ValueError):
                g.GBSpecialBase._shutoff_lag_leaves()


if __name__ == "__main__":
    unittest.main()
