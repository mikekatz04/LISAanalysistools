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


if __name__ == "__main__":
    unittest.main()
