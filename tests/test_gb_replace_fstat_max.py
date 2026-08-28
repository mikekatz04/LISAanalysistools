"""rj_replace SEARCH-mode maximize-and-pretend-uniform knob.

USER RULING 2026-08-28: search proposals "maximize parameters and pretend
they all have uniform distributions" -- no detailed-balance bookkeeping.
For the REPLACE move that means the candidate IS the JKS maximizer at its
drawn intrinsics: slot 0 pinned AT the per-row F-stat center (no lognormal
draw, no floor mix; phi0/iota/psi were already pinned) and the proposal
factors dropped entirely from the acceptance. ``GB_REPLACE_FSTAT_MAX``:
``auto`` (default) arms it only for moves with "search" in the name (the
established band-shutoff / per-row-centers scoping idiom) that carry the
F-stat birth container; ``1``/``0`` force either way. PE replace keeps
the exact-DB draw path bit-identically.

The full-flow numeric behavior rides on ``_debug_verify_replace_step``
(GB_DEBUG) in production, matching the GB_REPLACE_PHASE_MAX precedent;
here we pin the knob semantics and the two helpers the wiring keys off.
"""

import os
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _SearchStub:
    name = "gb_search_rj_replace"
    rj_fstat_dist_birth = True


class _PEStub:
    name = "rj_refit"
    rj_fstat_dist_birth = True


class _NoContainerStub:
    name = "gb_search_rj_replace"
    rj_fstat_dist_birth = False


class _DistTF:
    input_basis = ["dist", "f0_ms", "fdot_ratio", "phi0", "cosi", "psi",
                   "lam", "sinbeta"]


class _AmpTF:
    input_basis = ["lnA", "f0_ms", "fdot_ratio", "phi0", "cosi", "psi",
                   "lam", "sinbeta"]


class ReplaceFstatMaxFlagTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("GB_REPLACE_FSTAT_MAX")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GB_REPLACE_FSTAT_MAX", None)
        else:
            os.environ["GB_REPLACE_FSTAT_MAX"] = self._saved

    def test_auto_on_for_search_move(self):
        os.environ.pop("GB_REPLACE_FSTAT_MAX", None)
        self.assertTrue(GBSpecialBase._replace_fstat_max(_SearchStub()))

    def test_auto_off_for_pe_move(self):
        os.environ.pop("GB_REPLACE_FSTAT_MAX", None)
        self.assertFalse(GBSpecialBase._replace_fstat_max(_PEStub()))

    def test_auto_off_without_fstat_container(self):
        os.environ.pop("GB_REPLACE_FSTAT_MAX", None)
        self.assertFalse(GBSpecialBase._replace_fstat_max(_NoContainerStub()))

    def test_auto_on_for_stage_stamped_rj_replace(self):
        # The production move is named plain "rj_replace"; the recipe's
        # search-only install site stamps ``replace_search_stage`` so auto
        # arms it there without relying on the name.
        os.environ.pop("GB_REPLACE_FSTAT_MAX", None)
        stub = _PEStub()
        stub.name = "rj_replace"
        stub.replace_search_stage = True
        self.assertTrue(GBSpecialBase._replace_fstat_max(stub))

    def test_auto_off_for_unstamped_rj_replace(self):
        # No stamp + no "search" in the name (a future PE install) stays
        # on the exact-DB path.
        os.environ.pop("GB_REPLACE_FSTAT_MAX", None)
        stub = _PEStub()
        stub.name = "rj_replace"
        self.assertFalse(GBSpecialBase._replace_fstat_max(stub))

    def test_force_on_overrides_pe_name(self):
        os.environ["GB_REPLACE_FSTAT_MAX"] = "1"
        self.assertTrue(GBSpecialBase._replace_fstat_max(_PEStub()))

    def test_force_on_still_needs_container(self):
        # No centers -> nothing to pin; force cannot conjure the machinery.
        os.environ["GB_REPLACE_FSTAT_MAX"] = "1"
        self.assertFalse(GBSpecialBase._replace_fstat_max(_NoContainerStub()))

    def test_force_off(self):
        os.environ["GB_REPLACE_FSTAT_MAX"] = "0"
        self.assertFalse(GBSpecialBase._replace_fstat_max(_SearchStub()))


class ReplaceSlot0PinTest(unittest.TestCase):
    """Slot-0 pin at the center, respecting the sampling basis."""

    def test_distance_basis_pins_exp_center(self):
        stub = _SearchStub()
        stub.transform_fn = _DistTF()
        ln_center = np.array([-1.5, 0.0, 2.25])
        out = GBSpecialBase._replace_slot0_pin(stub, ln_center, np)
        np.testing.assert_array_equal(out, np.exp(ln_center))

    def test_amplitude_basis_pins_raw_center(self):
        stub = _SearchStub()
        stub.transform_fn = _AmpTF()
        ln_center = np.array([-51.2, -49.7])
        out = GBSpecialBase._replace_slot0_pin(stub, ln_center, np)
        np.testing.assert_array_equal(out, ln_center)


class ReplaceFactorsFinalTest(unittest.TestCase):
    """Pretend-uniform: the whole factors vector is zeroed, in place."""

    def test_fmax_zeroes_every_row(self):
        f = np.array([0.3, -2.0, 5.0, -1e300])
        out = GBSpecialBase._replace_factors_final(_SearchStub(), f, True)
        self.assertIs(out, f)
        np.testing.assert_array_equal(out, np.zeros_like(f))

    def test_off_is_identity(self):
        f = np.array([0.3, -2.0, 5.0])
        ref = f.copy()
        out = GBSpecialBase._replace_factors_final(_SearchStub(), f, False)
        self.assertIs(out, f)
        np.testing.assert_array_equal(out, ref)


if __name__ == "__main__":
    unittest.main()
