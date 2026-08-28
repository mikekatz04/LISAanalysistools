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


class SchedulerAddCountsTest(unittest.TestCase):
    def test_free_then_recap_budget_exact(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler
        specials = np.repeat(np.array([10, 20]), 2)  # 2 cells x 2 countable
        sched = BandScheduler(specials, 2, xp=np)
        # cell 10 frees: +3 staged birth rows join its budget
        sched.add_counts(np.array([10]), np.array([3]))
        pos = sched._cells_of(np.array([10]))[0]
        self.assertEqual(int(sched.cell_counts[pos]), 5)
        # re-cap: 2 of them were picked meanwhile -> only 1 leaves
        sched.add_counts(np.array([10]), np.array([-1]))
        self.assertEqual(int(sched.cell_counts[pos]), 4)
        # other cell untouched
        pos20 = sched._cells_of(np.array([20]))[0]
        self.assertEqual(int(sched.cell_counts[pos20]), 2)


class BandShutoffTest(unittest.TestCase):
    # OCCUPANCY-based rules (user key change 2026-08-15): see
    # _update_band_shutoff. Bands: edges 5/8/11/14/17 mHz -> bands 0-1
    # below the 10 mHz floor, bands 2-3 eligible.
    def _move(self, name="rj_fstat_search", cap=None):
        m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
        m.name = name
        m.num_bands = 4
        m.is_rj_prop = True
        m.rj_removal_only = False
        m.rj_replace = False
        m.rj_fstat_dist_birth = True
        m.band_edges = np.array([5e-3, 8e-3, 11e-3, 14e-3, 17e-3])
        m._band_leaf_cap = cap
        return m

    def test_scope_gating(self):
        import os
        m = self._move("rj_fstat_search")
        self.assertTrue(m._band_shutoff_enabled())
        self.assertFalse(self._move("rj_fstat_pe")._band_shutoff_enabled())
        os.environ["GB_RJ_BAND_SHUTOFF_SCOPE"] = "all"
        self.addCleanup(os.environ.pop, "GB_RJ_BAND_SHUTOFF_SCOPE", None)
        self.assertTrue(self._move("rj_fstat_pe")._band_shutoff_enabled())
        os.environ["GB_RJ_BAND_SHUTOFF_SCOPE"] = "off"
        self.assertFalse(m._band_shutoff_enabled())

    def test_zero_occupancy_shutoff_after_5(self):
        m = self._move(cap=np.array([2, 2, 2, 2]))
        for _ in range(4):
            m._update_band_shutoff(np.array([0, 0, 0, 0]))
        self.assertFalse(m._rj_band_shutoff.any())
        m._update_band_shutoff(np.array([0, 0, 0, 0]))
        # low bands never; high bands off at the 5th zero iteration
        np.testing.assert_array_equal(
            m._rj_band_shutoff, [False, False, True, True])

    def test_one_source_never_shuts_off(self):
        """ZERO sources is the only qualifying state (user ruling
        2026-08-28). A band holding a source is never silenced, whatever
        its cap allowed -- previously occupancy 1 qualified when the cap
        was above 1, which meant a band was shut off the moment it caught
        its FIRST source. Under the RJ freeze that also trapped that
        source until revival."""
        for cap in (np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2]), None):
            m = self._move(cap=cap)
            for _ in range(10):
                m._update_band_shutoff(np.array([1, 1, 1, 1]))
            self.assertFalse(
                m._rj_band_shutoff.any(),
                f"a band holding one source was shut off (cap={cap})")

    def test_occupancy_change_resets_streak(self):
        m = self._move(cap=np.array([2, 2, 2, 2]))
        for _ in range(4):
            m._update_band_shutoff(np.array([0, 0, 0, 0]))
        # band 2 gains its first source -> its zero-streak resets and,
        # since only ZERO qualifies, it can never re-arm while occupied
        for _ in range(4):
            m._update_band_shutoff(np.array([0, 0, 1, 0]))
        self.assertFalse(bool(m._rj_band_shutoff[2]))
        self.assertTrue(bool(m._rj_band_shutoff[3]))  # stayed 0 -> off
        for _ in range(20):
            m._update_band_shutoff(np.array([0, 0, 1, 0]))
        self.assertFalse(bool(m._rj_band_shutoff[2]))

    def test_two_or_more_never_counts(self):
        m = self._move(cap=np.array([3, 3, 3, 3]))
        for _ in range(10):
            m._update_band_shutoff(np.array([2, 2, 2, 2]))
        self.assertFalse(m._rj_band_shutoff.any())

    def test_cap_is_irrelevant_now_that_only_zero_qualifies(self):
        """The cap no longer enters the shutoff rule at all."""
        for cap in (None, np.array([1, 1, 1, 1]), np.array([5, 5, 5, 5])):
            m = self._move(cap=cap)
            for _ in range(5):
                m._update_band_shutoff(np.array([0, 0, 0, 0]))
            np.testing.assert_array_equal(
                m._rj_band_shutoff, [False, False, True, True],
                err_msg=f"zero-source shutoff changed with cap={cap}")


class SchedulerAddCountsTest(unittest.TestCase):
    def test_free_then_recap_budget_exact(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler
        specials = np.repeat(np.array([10, 20]), 2)  # 2 cells x 2 countable
        sched = BandScheduler(specials, 2, xp=np)
        # cell 10 frees: +3 staged birth rows join its budget
        sched.add_counts(np.array([10]), np.array([3]))
        pos = sched._cells_of(np.array([10]))[0]
        self.assertEqual(int(sched.cell_counts[pos]), 5)
        # re-cap: 2 of them were picked meanwhile -> only 1 leaves
        sched.add_counts(np.array([10]), np.array([-1]))
        self.assertEqual(int(sched.cell_counts[pos]), 4)
        # other cell untouched
        pos20 = sched._cells_of(np.array([20]))[0]
        self.assertEqual(int(sched.cell_counts[pos20]), 2)


class TemperCadenceTest(unittest.TestCase):
    def _move(self, branch, n):
        m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
        m.branch_name = branch
        m.temper_every_proposes = n
        return m

    def setUp(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        self._base = GBSpecialBase
        self._base._branch_propose_counts = {}
        self._base._branch_last_temper = {}

    def _tick(self, branch):
        self._base._branch_propose_counts[branch] = (
            self._base._branch_propose_counts.get(branch, 0) + 1)

    def test_every_n_total_proposes_shared_across_moves(self):
        # two swap-enabled moves + one non-swap move all tick the census;
        # tempering fires once per 3 TOTAL proposes on whichever
        # swap-enabled move crosses the budget first.
        a, b = self._move("gb", 3), self._move("gb", 3)
        fires = []
        for i in range(9):
            self._tick("gb")
            mover = (a, b)[i % 2]
            if mover._temper_cadence_fire():
                fires.append(i + 1)
        # first eligible propose fires (fresh process), then every 3rd tick
        self.assertEqual(fires, [1, 4, 7])

    def test_n_one_is_legacy_passthrough(self):
        m = self._move("gb", 1)
        for _ in range(5):
            self._tick("gb")
            self.assertTrue(m._temper_cadence_fire())

    def test_branches_independent(self):
        g, v = self._move("gb", 3), self._move("vgb", 1)
        self._tick("gb"); self._tick("vgb")
        self.assertTrue(g._temper_cadence_fire())
        self.assertTrue(v._temper_cadence_fire())
        self._tick("gb"); self._tick("vgb")
        self.assertFalse(g._temper_cadence_fire())  # gb budget not reached
        self.assertTrue(v._temper_cadence_fire())   # vgb cadence 1


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


class ModeDefaultPolicyTest(unittest.TestCase):
    """The per-mode flip-fraction policy (user ruling 2026-08-28).

    Search flips every eligible slot; PE settles to a random subset. These
    were bare literals duplicated at two sites in ``recipe.py``, which is
    how they drift apart -- hoisted to named constants so the policy is
    pinned in one place and visible here.
    """

    def test_search_default_is_every_slot(self):
        from lisatools.globalfit.recipe import _SEARCH_RJ_FLIP_DEFAULT
        self.assertEqual(_SEARCH_RJ_FLIP_DEFAULT, 1.0)

    def test_pe_default_is_thirty_percent(self):
        from lisatools.globalfit.recipe import _PE_RJ_FLIP_DEFAULT
        self.assertEqual(_PE_RJ_FLIP_DEFAULT, 0.3)

    def test_both_are_legal_fractions(self):
        """Whatever the policy, it has to survive the resolver's bounds."""
        from lisatools.globalfit.recipe import (
            _PE_RJ_FLIP_DEFAULT, _SEARCH_RJ_FLIP_DEFAULT,
        )
        for v in (_SEARCH_RJ_FLIP_DEFAULT, _PE_RJ_FLIP_DEFAULT):
            self.assertEqual(_resolve_rj_flip_fraction("gb", None, v), v)


if __name__ == "__main__":
    unittest.main()
