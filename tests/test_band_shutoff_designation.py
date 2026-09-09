"""The high-f band-shutoff valve must not be gated on an unrelated knob.

Until 2026-09-08 ``_band_shutoff_enabled`` required ``rj_fstat_dist_birth``
as its "am I the designated move?" test. That flag follows
``GB_RJ_FSTAT_DIST_BIRTH`` -- a knob about how amplitude/distance are drawn
AT BIRTH -- and **every v8 production script exports it as 0**. With it
falsy the method returned False on every move, so ``_update_band_shutoff``
was never ticked, the occupancy streaks never accumulated, and nothing was
ever logged: the tick's guard only logs when the tick RAISES, and it never
got that far. The valve read as "criterion unreachable" when it had in fact
been switched off at the gate.

It was not unreachable. Measured on the 3-month v8 10-walker run (job 465)
at stored iteration 80: **646 of 688 eligible bands held zero cold leaves in
every walker**, and all 688 had all-walker-zero streaks of >= 20 against a
threshold of 5. Only 42 of the cold walker's 669 leaves sat above 10 mHz.

The designation is now ``leaf_cap_update``, which is True in exactly one
place (the base ``gb_move_kwargs`` in recipe.py, carried by
``rj_fstat_search``) and explicitly False on every other move -- precisely
the "exactly ONE enabled move per stage" the valve documents.
"""

import os
import unittest
from unittest import mock

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _Stub:
    """Duck-typed stand-in exposing only what the gate reads."""

    def __init__(self, name="rj_fstat_search", *, leaf_cap_update=True,
                 is_rj_prop=True, rj_removal_only=False, rj_replace=False,
                 rj_fstat_dist_birth=False):
        self.name = name
        self.leaf_cap_update = leaf_cap_update
        self.is_rj_prop = is_rj_prop
        self.rj_removal_only = rj_removal_only
        self.rj_replace = rj_replace
        # False mirrors production (GB_RJ_FSTAT_DIST_BIRTH=0 everywhere).
        self.rj_fstat_dist_birth = rj_fstat_dist_birth

    _band_shutoff_enabled = GBSpecialBase._band_shutoff_enabled


class BandShutoffDesignation(unittest.TestCase):
    def setUp(self):
        # Default scope, not whatever the ambient environment carries.
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("GB_RJ_BAND_SHUTOFF_SCOPE", None)

    # ---- the regression -------------------------------------------------
    def test_enabled_with_dist_birth_off(self):
        """THE BUG: production runs with GB_RJ_FSTAT_DIST_BIRTH=0."""
        self.assertTrue(_Stub(rj_fstat_dist_birth=False)
                        ._band_shutoff_enabled())

    def test_dist_birth_does_not_change_the_answer(self):
        """The valve must be independent of that knob, either way."""
        on = _Stub(rj_fstat_dist_birth=True)._band_shutoff_enabled()
        off = _Stub(rj_fstat_dist_birth=False)._band_shutoff_enabled()
        self.assertEqual(on, off)
        self.assertTrue(on)

    # ---- the designation still holds ------------------------------------
    def test_non_designated_move_is_disabled(self):
        """Exactly one move per stage ticks; the rest must not."""
        self.assertFalse(_Stub(leaf_cap_update=False)
                         ._band_shutoff_enabled())

    def test_removal_only_and_replace_are_disabled(self):
        self.assertFalse(_Stub(rj_removal_only=True)._band_shutoff_enabled())
        self.assertFalse(_Stub(rj_replace=True)._band_shutoff_enabled())

    def test_non_rj_move_is_disabled(self):
        self.assertFalse(_Stub(is_rj_prop=False)._band_shutoff_enabled())

    # ---- scope ----------------------------------------------------------
    def test_scope_off_disables(self):
        with mock.patch.dict(os.environ,
                             {"GB_RJ_BAND_SHUTOFF_SCOPE": "off"}):
            self.assertFalse(_Stub()._band_shutoff_enabled())

    def test_default_scope_is_search_named_only(self):
        self.assertTrue(_Stub("rj_fstat_search")._band_shutoff_enabled())
        self.assertFalse(_Stub("rj_fstat_pe")._band_shutoff_enabled())

    def test_scope_all_admits_a_pe_named_designated_move(self):
        with mock.patch.dict(os.environ,
                             {"GB_RJ_BAND_SHUTOFF_SCOPE": "all"}):
            self.assertTrue(_Stub("rj_fstat_pe")._band_shutoff_enabled())


class DesignationIsUniqueInTheRecipe(unittest.TestCase):
    """``leaf_cap_update=True`` must stay a single-site designation.

    The gate is only as good as that uniqueness; a second True would let
    two moves tick the streaks and double-count iterations, which is the
    failure the docstring warns about.
    """

    def test_exactly_one_true_site(self):
        import pathlib

        import lisatools.globalfit.recipe as recipe

        src = pathlib.Path(recipe.__file__).read_text()
        hits = [ln for ln in src.splitlines()
                if "leaf_cap_update" in ln and "True" in ln]
        self.assertEqual(len(hits), 1, f"expected 1 designation, got {hits}")


if __name__ == "__main__":
    unittest.main()
