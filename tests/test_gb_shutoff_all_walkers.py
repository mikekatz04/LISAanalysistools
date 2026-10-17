"""The per-walker RJ valve's ALL-WALKER rule (Mike 2026-10-17).

"I just want it to be the old gate based on logl ... BUT, I don't want it to
shut off unless all 4 walkers have reached shutoff together. ... If one
walker reaches convergence criterion but the others are not, then all
walkers should be open including the one that might pass the convergence
criterion. If at a future time the 'converged' walker's statistic increases
enough that it is no longer converged, that is okay. It can change its
status back and forth ... That sub-band only gets shut off when all other
walkers are converged together at the same instant."

Pinned on ``_ColdBandLnL.judge`` directly with the band_info fixture the
persistence suite uses. Why: on the 1yr flagship job 780 the per-walker
valve shut 46 of walker 0's 58 pairs at 1.5-2.5 mHz while the other walkers
kept adding sources there; a shut pair runs no RJ, so walker 0 fell 3,750
nats and 185 leaves behind.
"""
import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as g

NW, NB = 4, 5


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
    v.judge(4.0, 3, np.ones((NW, NB), dtype=np.int64), all_walkers=False)
    assert int(v.streak.max()) == 0 and np.all(v.max == 0.0)
    return v


def _flat(v, n, occ, **kw):
    """``n`` judges with no improvement anywhere (the peak never beats the max)."""
    out = None
    for _ in range(n):
        v.peak[:] = 0.0
        out = v.judge(4.0, 3, occ, **kw)
    return out


OCC = np.ones((NW, NB), dtype=np.int64) * 3


class AllWalkerRuleTest(unittest.TestCase):

    def test_old_per_walker_behaviour_when_off(self):
        bi = _band_info(); v = _view(bi)
        _flat(v, 2, OCC, all_walkers=False)
        v.peak[:] = 0.0; v.peak[1, 2] = 50.0            # walker 1 improves in band 2
        conv = v.judge(4.0, 3, OCC, all_walkers=False)
        want = np.ones((NW, NB), dtype=bool); want[1, 2] = False
        np.testing.assert_array_equal(conv, want)        # the others shut alone, as before

    def test_a_band_shuts_only_when_every_walker_is_converged_together(self):
        bi = _band_info(); v = _view(bi); shut = bi["band_rj_shutoff_w"]
        _flat(v, 2, OCC)
        v.peak[:] = 0.0; v.peak[1, 2] = 50.0            # walker 1 improves in band 2 at the 3rd judge
        conv = v.judge(4.0, 3, OCC)
        self.assertTrue(conv[:, [0, 1, 3, 4]].all())     # those bands: all four converged -> all shut
        self.assertFalse(conv[:, 2].any())               # band 2: three converged, one not -> all OPEN
        self.assertEqual(v.held_open, 3)                 # the three converged walkers held open
        shut[conv] = True
        # two more flat judges: walker 1's streak reaches 3 -> band 2 shuts on all four at once
        conv = _flat(v, 2, OCC)
        self.assertFalse(conv[:, 2].any())               # streak 2 after the first
        shut[conv] = True
        conv = _flat(v, 1, OCC)
        self.assertTrue(conv[:, 2].all())
        shut[conv] = True
        self.assertTrue(shut.all())

    def test_a_shut_band_reopens_on_every_walker_when_one_stops_being_converged(self):
        bi = _band_info(); v = _view(bi); shut = bi["band_rj_shutoff_w"]
        shut[conv := _flat(v, 3, OCC)] = True
        self.assertTrue(shut.all())
        v.peak[:] = 0.0; v.peak[3, 4] = 30.0            # walker 3's band 4 gain beats max + tol
        conv = v.judge(4.0, 3, OCC)
        self.assertFalse(conv.any())
        self.assertFalse(shut[:, 4].any())               # band 4 reopened on all four walkers
        self.assertTrue(shut[:, [0, 1, 2, 3]].all())     # the other bands stay shut
        self.assertEqual(v.reopened, 4)
        self.assertEqual(int(v.streak[3, 4]), 0)
        # back and forth: three flat judges and it shuts again on all four
        conv = _flat(v, 3, OCC)
        self.assertTrue(conv[:, 4].all())

    def test_a_walker_holding_nothing_does_not_vote_and_is_not_shut(self):
        bi = _band_info(); v = _view(bi); shut = bi["band_rj_shutoff_w"]
        occ = OCC.copy(); occ[0, 1] = 0                  # walker 0 holds nothing in band 1
        conv = _flat(v, 3, occ)
        self.assertTrue(conv[1:, 1].all())               # the three occupied walkers shut together
        self.assertFalse(conv[0, 1])                     # the empty pair stays open for births
        self.assertEqual(v.held_open, 0)
        occ2 = occ.copy(); occ2[:, 3] = 0                # a band nobody holds never shuts
        conv = _flat(v, 1, occ2)
        self.assertFalse(conv[:, 3].any())

    def test_require_occ_off_makes_empty_pairs_vote_by_their_streak(self):
        bi = _band_info(); v = _view(bi)
        occ = OCC.copy(); occ[0, 1] = 0
        conv = _flat(v, 3, occ, require_occ=False)
        self.assertTrue(conv[:, 1].all())                # the empty pair converged (flat) and shuts too

    def test_the_knob(self):
        env = {k: v for k, v in os.environ.items() if k != "GB_SEARCH_BAND_SHUTOFF_ALL_WALKERS"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(g.GBSpecialBase._shutoff_all_walkers())
        with mock.patch.dict(os.environ, {"GB_SEARCH_BAND_SHUTOFF_ALL_WALKERS": "0"}):
            self.assertFalse(g.GBSpecialBase._shutoff_all_walkers())


if __name__ == "__main__":
    unittest.main()
