import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _resolve_rj_flip_fraction,
)


def _move(fraction):
    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.rj_flip_fraction = fraction
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"  # self.xp resolves through the backend
    return m


def _picked(n, num_sources):
    ids = np.random.choice(num_sources, size=n, replace=False)
    return {
        "ids": ids,
        "specials": ids * 7,
        "slot_index": np.arange(n, dtype=np.int32),
        "temp_inds": np.zeros(n, dtype=int),
        "walker_inds": np.arange(n) % 4,
        "band_inds": np.arange(n) % 11,
        "N_vals": np.full(n, 128),
    }


def _sorter(num_sources, inds=None):
    # The 2026-08-14 deaths-only gate reads ``band_sorter.inds``: births are
    # thinned EARLY (unit-open subset exclusion), so the in-step filter only
    # gates alive rows. All-alive is the death-gating worst case the legacy
    # assertions exercise.
    if inds is None:
        inds = np.ones(num_sources, dtype=bool)
    return SimpleNamespace(num_sources=num_sources, inds=inds)


class RJFlipFractionTest(unittest.TestCase):
    def test_fraction_one_is_passthrough(self):
        m = _move(1.0)
        sorter = _sorter(100)
        picked = _picked(30, 100)
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertIs(out, picked)
        self.assertFalse(hasattr(sorter, "_rj_flip_allowed"))

    def test_subset_size_and_without_replacement(self):
        m = _move(0.3)
        sorter = _sorter(100)
        m._apply_rj_flip_fraction(sorter, _picked(30, 100))
        allowed = sorter._rj_flip_allowed
        self.assertEqual(allowed.dtype, np.bool_)
        self.assertEqual(int(allowed.sum()), 30)  # round(0.3 * 100), no repeats

    def test_mask_drawn_once_per_sorter(self):
        m = _move(0.5)
        sorter = _sorter(200)
        m._apply_rj_flip_fraction(sorter, _picked(10, 200))
        mask_first = sorter._rj_flip_allowed
        m._apply_rj_flip_fraction(sorter, _picked(10, 200))
        self.assertIs(sorter._rj_flip_allowed, mask_first)
        # a NEW sorter (new proposal) gets its own draw
        sorter2 = _sorter(200)
        m._apply_rj_flip_fraction(sorter2, _picked(10, 200))
        self.assertIsNot(sorter2._rj_flip_allowed, mask_first)

    def test_filter_consistent_across_keys(self):
        np.random.seed(3)
        m = _move(0.4)
        sorter = _sorter(50)
        picked = _picked(50, 50)  # every source picked -> keep == allowed
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertIsNotNone(out)
        allowed = sorter._rj_flip_allowed
        self.assertTrue(np.all(allowed[out["ids"]]))
        n_kept = len(out["ids"])
        for key, value in out.items():
            self.assertEqual(len(value), n_kept, key)
        # rows outside the subset never appear
        dropped = picked["ids"][~allowed[picked["ids"]]]
        self.assertFalse(np.isin(out["ids"], dropped).any())

    def test_empty_batch_returns_none(self):
        m = _move(0.02)  # keeps max(1, round(0.02*100)) = 2 of 100
        sorter = _sorter(100)
        m._apply_rj_flip_fraction(sorter, _picked(5, 100))
        allowed = sorter._rj_flip_allowed
        blocked = np.where(~allowed)[0][:5]
        picked = _picked(5, 100)
        picked["ids"] = blocked
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertIsNone(out)

    def test_minimum_one_slot(self):
        m = _move(0.001)
        sorter = _sorter(10)  # round(0.001*10) = 0 -> 1
        m._apply_rj_flip_fraction(sorter, _picked(10, 10))
        self.assertEqual(int(sorter._rj_flip_allowed.sum()), 1)

    def test_original_picked_untouched_for_in_model(self):
        # _run_in_model_repeats receives the CALLER's picked dict and masks
        # to alive itself -- the flip filter must return a new dict and
        # never mutate the input, so every inds=True slot still gets its
        # in-model updates even when it skipped the RJ flip.
        np.random.seed(11)
        m = _move(0.2)
        sorter = _sorter(60)
        picked = _picked(40, 60)
        originals = {k: v.copy() for k, v in picked.items()}
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertIsNot(out, picked)
        for key, value in picked.items():
            np.testing.assert_array_equal(value, originals[key])
        self.assertLess(len(out["ids"]), len(picked["ids"]))


class DeathsOnlyGateTest(unittest.TestCase):
    def test_births_pass_unconditionally(self):
        # Births are thinned EARLY (unit-open subset exclusion): every dead
        # row reaching a pick is already in the flip subset, so the in-step
        # gate must not re-thin it (that would square the fraction).
        np.random.seed(7)
        m = _move(0.2)
        inds = np.zeros(50, dtype=bool)  # all rows dead = all picks births
        sorter = _sorter(50, inds=inds)
        picked = _picked(50, 50)
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertEqual(len(out["ids"]), 50)

    def test_deaths_gated_births_kept_mixed(self):
        np.random.seed(13)
        m = _move(0.2)
        inds = np.zeros(100, dtype=bool)
        inds[:40] = True  # rows 0-39 alive (deaths), 40-99 dead (births)
        sorter = _sorter(100, inds=inds)
        picked = _picked(100, 100)
        out = m._apply_rj_flip_fraction(sorter, picked)
        allowed = sorter._rj_flip_allowed
        out_alive = out["ids"][inds[out["ids"]]]
        out_dead = out["ids"][~inds[out["ids"]]]
        # every dead pick survives; alive picks survive iff in the subset
        self.assertEqual(len(out_dead), 60)
        self.assertTrue(np.all(allowed[out_alive]))
        dropped = picked["ids"][inds[picked["ids"]] & ~allowed[picked["ids"]]]
        self.assertFalse(np.isin(out["ids"], dropped).any())


class SchedulerFrozenAdvanceTest(unittest.TestCase):
    def _scheduler(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler
        # 4 cells x 2 sources each, 2 slots staged
        specials = np.repeat(np.array([10, 20, 30, 40]), 2)
        return BandScheduler(specials, 2, xp=np)

    def test_frozen_cell_never_retired(self):
        sched = self._scheduler()
        staged = np.asarray(sched.slot_specials).copy()
        # finish both staged cells (one pick per cell per round, as in
        # production -- fancy-index += dedupes repeated specials in one call)
        sched.record_picks(staged)
        sched.record_picks(staged)
        frozen = staged[:1]
        inds_fill, new_specials = sched.advance(frozen_specials=frozen)
        # only the unfrozen finished slot retires/refills
        self.assertEqual(len(inds_fill), 1)
        self.assertTrue(np.all(np.asarray(sched.slot_specials)[inds_fill] != frozen[0]))
        self.assertIn(int(frozen[0]), np.asarray(sched.slot_specials).tolist())
        # after the flush the frozen cell retires normally
        inds_fill2, _ = sched.advance()
        self.assertEqual(len(inds_fill2), 1)

    def test_advance_without_frozen_matches_legacy(self):
        sched = self._scheduler()
        staged = np.asarray(sched.slot_specials).copy()
        sched.record_picks(staged)
        sched.record_picks(staged)
        inds_fill, new_specials = sched.advance()
        self.assertEqual(len(inds_fill), 2)


class FstatCenterCacheLookupTest(unittest.TestCase):
    def test_lookup_maps_main_ids_to_cache_rows(self):
        m = _move(1.0)
        m.name = "test"
        ids = np.array([3, 7, 11, 40, 41, 90])  # ascending, as get_subset_inds
        m._fstat_ctr = {"ids": ids, "ln_center": np.arange(6.0)}
        pos = m._fstat_ctr_lookup(np.array([11, 3, 90]))
        np.testing.assert_array_equal(pos, [2, 0, 5])
        np.testing.assert_array_equal(
            m._fstat_ctr["ln_center"][pos], [2.0, 0.0, 5.0]
        )

    def test_lookup_raises_on_foreign_id(self):
        m = _move(1.0)
        m.name = "test"
        m._fstat_ctr = {"ids": np.array([3, 7, 11])}
        with self.assertRaises(RuntimeError):
            m._fstat_ctr_lookup(np.array([7, 8]))  # 8 not in the unit


class ResolveRJFlipFractionTest(unittest.TestCase):
    def test_kwarg_wins_over_env(self):
        import os
        os.environ["GB_RJ_FLIP_FRACTION"] = "0.7"
        self.addCleanup(os.environ.pop, "GB_RJ_FLIP_FRACTION", None)
        self.assertEqual(_resolve_rj_flip_fraction("gb", 0.4), 0.4)
        self.assertEqual(_resolve_rj_flip_fraction("gb", None), 0.7)

    def test_default_is_one(self):
        self.assertEqual(_resolve_rj_flip_fraction("gb", None), 1.0)

    def test_mode_default_chain(self):
        import os
        # builder default honored when no env / kwarg (the PE-cycle 0.1)
        self.assertEqual(_resolve_rj_flip_fraction("gb", None, 0.1), 0.1)
        self.assertEqual(_resolve_rj_flip_fraction("gb", None, 1.0), 1.0)
        # env beats the builder default
        os.environ["GB_RJ_FLIP_FRACTION"] = "0.5"
        self.addCleanup(os.environ.pop, "GB_RJ_FLIP_FRACTION", None)
        self.assertEqual(_resolve_rj_flip_fraction("gb", None, 0.1), 0.5)
        # explicit kwarg beats both
        self.assertEqual(_resolve_rj_flip_fraction("gb", 0.9, 0.1), 0.9)

    def test_bounds(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                _resolve_rj_flip_fraction("gb", bad)

    def test_vgb_has_no_rj_knob_surface(self):
        import os
        # env is IGNORED for the fixed-leaf vgb branch...
        os.environ["VGB_RJ_FLIP_FRACTION"] = "0.3"
        self.addCleanup(os.environ.pop, "VGB_RJ_FLIP_FRACTION", None)
        self.assertEqual(_resolve_rj_flip_fraction("vgb", None), 1.0)
        # ...and an explicit kwarg is rejected, not silently dropped.
        with self.assertRaises(ValueError):
            _resolve_rj_flip_fraction("vgb", 0.5)


if __name__ == "__main__":
    unittest.main()
