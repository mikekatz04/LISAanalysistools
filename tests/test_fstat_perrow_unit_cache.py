"""Per-row F-stat centers THROUGH the unit-open cache (2026-08-27).

Job 349's [GB_TIMING] measured rj_fstat_centers at 726 s/row: the
2026-08-26 per-row ruling (exact JKS maximizers for search births/deaths,
bypassing the f0-node epoch table) ALSO bypassed the job-195 unit-open
center cache as collateral -- so every pick round recomputes centers whose
inputs are fixed at unit open (birth coords are pre-drawn at sorter build;
alive coords cannot change before their single in-model block at unit
end; the F-stat ignores the extrinsic columns an accepted birth
overwrites). Re-routing per-row mode through the unit cache keeps the
exact per-row values (same _fstat_dist_centers + _dist_center_and_width
path, batched once per unit, with the blessed 1.5x snapshot smear and the
lookup's inline miss fallback) and collapses ~40 rounds of recomputation
per unit to one.

GB_FSTAT_PERROW_UNIT_CACHE=0 restores the per-round direct computation
bit-for-bit. These tests pin the gating matrix; the numeric path is the
already-shared _fstat_ctr_compute (its cache-vs-fallback identity is a
documented invariant) and the cluster [FSTAT_CTR] census lines are the
production visibility.
"""

import os
import unittest

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


class _Stub:
    def __init__(self, name="rj_fstat_search", fstat_birth=True,
                 rj_replace=False, table=None):
        self.name = name
        self.rj_fstat_dist_birth = fstat_birth
        self.rj_replace = rj_replace
        self._table = table

    def _fstat_ctr_table_active(self):
        return self._table

    # bind the real resolution helpers through the class
    _rj_birth_perrow = GBSpecialBase._rj_birth_perrow
    _perrow_unit_cache = GBSpecialBase._perrow_unit_cache
    _fstat_ctr_hoist_wanted = GBSpecialBase._fstat_ctr_hoist_wanted
    _resolve_rj_ctr = GBSpecialBase._resolve_rj_ctr
    _unit_cache_smear = GBSpecialBase._unit_cache_smear
    _fstat_ctr_smear = GBSpecialBase._fstat_ctr_smear
    _fstat_ctr_mode = staticmethod(GBSpecialBase._fstat_ctr_mode)


_ENVS = ("GB_RJ_BIRTH_CTR_MODE", "GB_FSTAT_PERROW_UNIT_CACHE",
         "GB_RJ_FSTAT_CTR_HOIST", "GB_FSTAT_CTR_SMEAR",
         "GB_FSTAT_CTR_MODE")


class _EnvCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENVS}
        for k in _ENVS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class PerrowUnitCacheKnobTest(_EnvCase):
    def test_default_on(self):
        self.assertTrue(_Stub()._perrow_unit_cache())

    def test_zero_disables(self):
        os.environ["GB_FSTAT_PERROW_UNIT_CACHE"] = "0"
        self.assertFalse(_Stub()._perrow_unit_cache())


class HoistWantedTest(_EnvCase):
    """The unit-open precompute must now ALSO run when a live epoch table
    will be bypassed by per-row mode (previously: table active -> no
    hoist -> per-row fell through to per-round direct computation)."""

    def test_table_active_perrow_default_hoists(self):
        s = _Stub(table={"fake": 1})
        self.assertTrue(s._fstat_ctr_hoist_wanted())

    def test_table_active_perrow_knob_off_no_hoist(self):
        os.environ["GB_FSTAT_PERROW_UNIT_CACHE"] = "0"
        s = _Stub(table={"fake": 1})
        self.assertFalse(s._fstat_ctr_hoist_wanted())

    def test_table_active_not_perrow_no_hoist(self):
        # PE-mode move name -> auto per-row is OFF -> table serves lookups
        s = _Stub(name="gb_pe", table={"fake": 1})
        self.assertFalse(s._fstat_ctr_hoist_wanted())

    def test_no_table_hoists_as_before(self):
        self.assertTrue(_Stub(table=None)._fstat_ctr_hoist_wanted())

    def test_replace_move_never_hoists(self):
        s = _Stub(rj_replace=True, table=None)
        self.assertFalse(s._fstat_ctr_hoist_wanted())

    def test_hoist_env_off_wins(self):
        os.environ["GB_RJ_FSTAT_CTR_HOIST"] = "0"
        self.assertFalse(_Stub(table=None)._fstat_ctr_hoist_wanted())

    def test_no_fstat_birth_no_hoist(self):
        self.assertFalse(_Stub(fstat_birth=False)._fstat_ctr_hoist_wanted())


class ResolveRjCtrTest(_EnvCase):
    """The per-step (table, unit_cache) source resolution."""

    def test_perrow_default_keeps_cache_drops_table(self):
        s = _Stub()
        tbl, ctr = s._resolve_rj_ctr({"fake": 1}, {"cache": 1})
        self.assertIsNone(tbl)
        self.assertEqual(ctr, {"cache": 1})

    def test_perrow_knob_off_drops_both(self):
        os.environ["GB_FSTAT_PERROW_UNIT_CACHE"] = "0"
        s = _Stub()
        tbl, ctr = s._resolve_rj_ctr({"fake": 1}, {"cache": 1})
        self.assertIsNone(tbl)
        self.assertIsNone(ctr)

    def test_not_perrow_passthrough(self):
        s = _Stub(name="gb_pe")
        tbl, ctr = s._resolve_rj_ctr({"fake": 1}, {"cache": 1})
        self.assertEqual(tbl, {"fake": 1})
        self.assertEqual(ctr, {"cache": 1})

    def test_forced_table_mode_passthrough(self):
        os.environ["GB_RJ_BIRTH_CTR_MODE"] = "table"
        s = _Stub()
        tbl, ctr = s._resolve_rj_ctr({"fake": 1}, {"cache": 1})
        self.assertEqual(tbl, {"fake": 1})
        self.assertEqual(ctr, {"cache": 1})

    def test_no_table_passthrough(self):
        # table already absent (unit mode): nothing to bypass
        s = _Stub()
        tbl, ctr = s._resolve_rj_ctr(None, {"cache": 1})
        self.assertIsNone(tbl)
        self.assertEqual(ctr, {"cache": 1})


class UnitCacheSmearTest(_EnvCase):
    """The unit cache's smear tracks the MACHINERY's staleness (mid-unit
    drift -> 1.5), not the mode env: under the production
    GB_FSTAT_CTR_MODE=epoch pin the generic resolver hands out the 2.0
    epoch smear, which would over-widen the per-row-through-unit-cache
    proposals."""

    def test_unit_cache_smear_default_1p5_even_under_epoch_mode(self):
        os.environ["GB_FSTAT_CTR_MODE"] = "epoch"
        s = _Stub()
        self.assertEqual(s._unit_cache_smear(), 1.5)
        # ... while the generic resolver still says 2.0 for the table
        self.assertEqual(s._fstat_ctr_smear(), 2.0)

    def test_env_override_wins_everywhere(self):
        os.environ["GB_FSTAT_CTR_SMEAR"] = "1.7"
        s = _Stub()
        self.assertEqual(s._unit_cache_smear(), 1.7)
        self.assertEqual(s._fstat_ctr_smear(), 1.7)


if __name__ == "__main__":
    unittest.main()
