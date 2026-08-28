"""rj_replace phase-maximized scoring knob (user directive 2026-08-27).

The replace move scores the NEW side phase-maximized and applies the
maximizing rotation to the accepted parameters (rotation-on-accept), so
the credited maximized delta IS attainable at the written phi0 -- the
resolution of the 2026-08-24 "phase max not attainable" drift flaw that
forced exact-only scoring. ``GB_REPLACE_PHASE_MAX=0`` restores the exact
concrete-parameter scoring bit-identically (the get_replace_ll
``phase_maximize=False`` path plus no rotation).

The numeric attainability property (rotated params re-score to the
maximized delta) is pinned by GBGPU tests/test_phase_max_fused.py at the
engine level and by ``_debug_verify_replace_step`` stage 2b in
production (its docstring was written for exactly this mode); here we
pin the knob semantics the move-level wiring keys off.
"""

import os
import unittest

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _Stub:
    name = "rj_replace"


class _SearchStub:
    """The production SEARCH install: plain name + the search stage stamp."""

    name = "rj_replace"
    replace_search_stage = True


class _PEStub:
    """The PE install: the PE stage stamp, no search stamp."""

    name = "rj_replace_pe"
    replace_pe_stage = True


class ReplacePhaseMaxFlagTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("GB_REPLACE_PHASE_MAX")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GB_REPLACE_PHASE_MAX", None)
        else:
            os.environ["GB_REPLACE_PHASE_MAX"] = self._saved

    def test_default_is_on(self):
        os.environ.pop("GB_REPLACE_PHASE_MAX", None)
        self.assertTrue(GBSpecialBase._replace_phase_max(_Stub()))

    def test_zero_disables(self):
        os.environ["GB_REPLACE_PHASE_MAX"] = "0"
        self.assertFalse(GBSpecialBase._replace_phase_max(_Stub()))

    def test_one_enables(self):
        os.environ["GB_REPLACE_PHASE_MAX"] = "1"
        self.assertTrue(GBSpecialBase._replace_phase_max(_Stub()))


class ReplacePhaseMaxStageSplitTest(unittest.TestCase):
    """GENERAL RULE (user, 2026-08-28): "no maximizing over parameters
    during PE". Phase-max scoring with rotation-on-accept IS a
    maximize-and-keep over phi0, so the PE-stage install must resolve it
    OFF by default. Search keeps it (maximization is the usual search
    default) but stays independently switchable via the env knob.
    """

    def setUp(self):
        self._saved = os.environ.get("GB_REPLACE_PHASE_MAX")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GB_REPLACE_PHASE_MAX", None)
        else:
            os.environ["GB_REPLACE_PHASE_MAX"] = self._saved

    def test_auto_is_off_for_the_pe_stamp(self):
        os.environ.pop("GB_REPLACE_PHASE_MAX", None)
        self.assertFalse(GBSpecialBase._replace_phase_max(_PEStub()))

    def test_auto_is_on_for_search_and_unstamped(self):
        # BIT-IDENTITY: every pre-existing caller keeps resolving True.
        os.environ.pop("GB_REPLACE_PHASE_MAX", None)
        self.assertTrue(GBSpecialBase._replace_phase_max(_SearchStub()))
        self.assertTrue(GBSpecialBase._replace_phase_max(_Stub()))

    def test_auto_is_the_explicit_spelling_of_the_default(self):
        os.environ["GB_REPLACE_PHASE_MAX"] = "auto"
        self.assertTrue(GBSpecialBase._replace_phase_max(_SearchStub()))
        self.assertFalse(GBSpecialBase._replace_phase_max(_PEStub()))

    def test_env_forces_both_directions_over_the_stamp(self):
        # "Generally we will use maximization during search, but that is
        # not a requirement": =0 must still switch SEARCH off, and =1
        # must still switch a PE-stamped move on.
        os.environ["GB_REPLACE_PHASE_MAX"] = "0"
        self.assertFalse(GBSpecialBase._replace_phase_max(_SearchStub()))
        self.assertFalse(GBSpecialBase._replace_phase_max(_PEStub()))
        os.environ["GB_REPLACE_PHASE_MAX"] = "1"
        self.assertTrue(GBSpecialBase._replace_phase_max(_SearchStub()))
        self.assertTrue(GBSpecialBase._replace_phase_max(_PEStub()))


class ReplacePhaseMaxScoringInterlockTest(unittest.TestCase):
    """``_replace_phase_max_scoring`` — the mode ``_run_replace_step``
    actually scores with.

    Rotation-on-accept OVERWRITES the candidate's phi0. When the PE
    extrinsic draw is active that phi0 was drawn from a proposal whose
    density is charged in the RJ factors, so overwriting it would break
    detailed balance (exactly why the PE birth path sets
    ``_pin_mode = not self._pe_extr_active()``). The interlock forces the
    scoring mode off in that combination, whatever the knob says.
    """

    def setUp(self):
        self._saved = os.environ.get("GB_REPLACE_PHASE_MAX")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GB_REPLACE_PHASE_MAX", None)
        else:
            os.environ["GB_REPLACE_PHASE_MAX"] = self._saved

    @staticmethod
    def _move_stub(**stamps):
        # A real (bare) move: the interlock calls back into
        # _replace_phase_max, so the plain duck-stubs above do not bind.
        from tests.test_gb_cap_cell_grid import _move

        m = _move(4)
        m.name = "rj_replace"
        for k, v in stamps.items():
            setattr(m, k, v)
        return m

    def test_search_scoring_follows_the_knob(self):
        m = self._move_stub(replace_search_stage=True)
        os.environ.pop("GB_REPLACE_PHASE_MAX", None)
        self.assertTrue(m._replace_phase_max_scoring(False))
        os.environ["GB_REPLACE_PHASE_MAX"] = "0"
        self.assertFalse(m._replace_phase_max_scoring(False))

    def test_drawn_extrinsics_veto_the_rotation(self):
        os.environ["GB_REPLACE_PHASE_MAX"] = "1"   # forced ON
        self.assertFalse(
            self._move_stub(replace_pe_stage=True,
                            name="rj_replace_pe")._replace_phase_max_scoring(
                                True))
        self.assertFalse(
            self._move_stub(
                replace_search_stage=True)._replace_phase_max_scoring(True))

    def test_knob_off_stays_off_with_drawn_extrinsics(self):
        os.environ["GB_REPLACE_PHASE_MAX"] = "0"
        m = self._move_stub(replace_pe_stage=True, name="rj_replace_pe")
        self.assertFalse(m._replace_phase_max_scoring(True))


if __name__ == "__main__":
    unittest.main()
