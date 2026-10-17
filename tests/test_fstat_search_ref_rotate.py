"""The search F-stat refit cycles through the cold walkers (Mike 2026-10-16).

"let's rotate the fits during search. Not random, but cycle through ...
during gb search 1 and 2". The search moves' reference-walker rule
(``GBSpecialBase._fstat_fit_ref_rule`` / ``_fstat_fit_ref_from``) is
``rotate`` by default: epoch k fits walker ``pool[k % len(pool)]`` over the
cold walkers with a finite lnL, in walker order, so every residual gets a
grid every ``nwalkers`` epochs and a resumed store re-derives the walker from
the epoch number. ``GB_FSTAT_SEARCH_REF=min`` restores the 2026-09-21 min-lnL
rule; the PE rule (``GB_FSTAT_PE_REF``, random by default) is untouched, and
so is the per-propose distance-birth centre (no epoch -> argmin).
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

ENV = ("GB_FSTAT_SEARCH_REF", "GB_FSTAT_PE_REF", "GB_FSTAT_PE_REF_SEED")


def _bare(search):
    """A bare move (no __init__): the rule reads the two mode flags only."""
    mv = GBSpecialBase.__new__(GBSpecialBase)
    mv.name = "rj_fstat_search" if search else "rj_fstat_pe"
    mv.fstat_search_residual = search
    mv.fstat_pe_residual = False
    return mv


def _search():
    return _bare(True)


def _pe():
    return _bare(False)


def _rule(mv):
    return mv._fstat_fit_ref_rule()


def _ref(mv, lls, epoch=None):
    return mv._fstat_fit_ref_from(lls, epoch)


class SearchRotationTest(unittest.TestCase):

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for k in ENV:
            os.environ.pop(k, None)

    def tearDown(self):
        self._env.stop()

    def test_search_default_is_rotate_and_pe_default_is_random(self):
        self.assertEqual(_rule(_search()), "rotate")
        self.assertEqual(_rule(_pe()), "random")

    def test_epochs_cycle_through_the_walkers_in_order(self):
        """Four walkers, the 1yr flagship's layout: epochs 12..19 fit
        3, 0, 1, 2, 3, 0, 1, 2 -- every walker twice, none skipped."""
        lls = np.array([-10.0, -30.0, -20.0, -40.0])     # walker 1 would be the min
        got = [_ref(_search(), lls, k) for k in range(12, 20)]
        self.assertEqual(got, [0, 1, 2, 3, 0, 1, 2, 3])
        # independent of the likelihood values
        self.assertEqual([_ref(_search(), -lls, k) for k in range(12, 20)], got)

    def test_the_min_lnl_walker_is_no_longer_the_pick(self):
        lls = np.array([-10.0, -30.0, -20.0, -40.0])     # argmin = 3
        picks = {_ref(_search(), lls, k) for k in range(4)}
        self.assertEqual(picks, {0, 1, 2, 3})

    def test_non_finite_walkers_are_skipped_without_breaking_the_cycle(self):
        lls = np.array([-10.0, np.nan, -20.0, -np.inf, -5.0])    # pool = 0, 2, 4
        self.assertEqual([_ref(_search(), lls, k) for k in range(6)], [0, 2, 4, 0, 2, 4])
        allnan = np.full(3, np.nan)                               # nothing finite: all walkers
        self.assertEqual([_ref(_search(), allnan, k) for k in range(3)], [0, 1, 2])

    def test_min_knob_restores_the_min_lnl_rule(self):
        lls = np.array([-10.0, -30.0, -20.0, -40.0])
        with mock.patch.dict(os.environ, {"GB_FSTAT_SEARCH_REF": "min"}):
            self.assertEqual(_rule(_search()), "min")
            self.assertEqual([_ref(_search(), lls, k) for k in range(4)], [3, 3, 3, 3])

    def test_random_is_refused_for_search(self):
        with mock.patch.dict(os.environ, {"GB_FSTAT_SEARCH_REF": "random"}):
            with self.assertRaisesRegex(ValueError, "rotate"):
                _rule(_search())

    def test_the_per_propose_centre_keeps_the_argmin(self):
        """No epoch = the F-stat distance-birth centre: argmin, in search and PE."""
        lls = np.array([-10.0, -30.0, -20.0, -40.0])
        self.assertEqual(_ref(_search(), lls), 3)
        self.assertEqual(_ref(_pe(), lls), 3)

    def test_the_pe_rule_is_not_touched_by_the_search_knob(self):
        lls = np.array([-5.0, -2.0, -9.0, -1.0])
        with mock.patch.dict(os.environ, {"GB_FSTAT_SEARCH_REF": "min"}):
            self.assertEqual(_rule(_pe()), "random")
            want = [int(np.random.default_rng([0, k]).integers(4)) for k in range(12)]
            self.assertEqual([_ref(_pe(), lls, k) for k in range(12)], want)
        # and rotate is accepted on the PE knob for symmetry (not the ruled default)
        with mock.patch.dict(os.environ, {"GB_FSTAT_PE_REF": "rotate"}):
            self.assertEqual([_ref(_pe(), lls, k) for k in range(5)], [0, 1, 2, 3, 0])

    def test_the_plural_selector_still_reproduces_the_no_epoch_scalar(self):
        """``_fstat_fit_refs_from(lls, 1)`` is the W=1 bit-identity hinge for the
        multi-walker plan; it ranks by lnL and the no-epoch scalar is argmin."""
        lls = np.array([-10.0, -30.0, -20.0, -40.0])
        mv = _search()
        self.assertEqual(mv._fstat_fit_refs_from(lls, 1)[0], _ref(mv, lls))


if __name__ == "__main__":
    unittest.main()
