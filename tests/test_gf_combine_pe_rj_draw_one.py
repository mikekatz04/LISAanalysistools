"""GFCombineMove PE-only exclusive RJ draw (user ruling 2026-09-10).

In a stage of kind ``"pe"`` that wraps BOTH ``rj_fstat_pe`` and
``rj_prior_pe``, ``GB_PE_RJ_DRAW_ONE=1`` makes exactly ONE of the two run
per iteration, drawn from the sampler RNG with
P(rj_fstat_pe) = ``GB_PE_RJ_FSTAT_FRACTION`` (default 0.8); every other
wrapped move runs once per iteration in its fixed order. Search / rj
stages, PE stages without the pair, and runs with the knob unset are
untouched (they keep the stage's configured weighted-cycle / sequential
mode).

Stub moves, no sampler, no GPU, no data (same style as
``test_gf_combine_weighted.py``).
"""

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.globalfitmove import GFCombineMove


class _StubMove:
    temperature_control = None
    periodic = None

    def __init__(self, name, log):
        self.gf_move_name = name
        self.log = log
        self.accepted = np.zeros((2, 4))

    def propose(self, model, state):
        self.log.append(self.gf_move_name)
        return state, np.ones((2, 4))


class _StubModel:
    def __init__(self, seed=1234):
        self.random = np.random.RandomState(seed)


class _StubState:
    sub_states = None


PE_NAMES = ("rj_warm_pe", "rj_fstat_pe", "rj_prior_pe", "gb_ridge_gibbs")
RJ = ("rj_fstat_pe", "rj_prior_pe")


def _make(names, log, kind="pe", **kwargs):
    moves = [_StubMove(n, log) for n in names]
    kwargs.setdefault("share_temperature_control", False)
    comb = GFCombineMove(moves=moves, **kwargs)
    comb.gf_stage_kind = kind
    comb.gf_stage_name = "full_pe" if kind == "pe" else "gb_search"
    return moves, comb


def _env(**kw):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kw.items()}, clear=False)


class PeRjDrawOneTest(unittest.TestCase):
    def test_exactly_one_rj_per_iteration_others_in_order(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        model = _StubModel(seed=3)
        with _env(GB_PE_RJ_DRAW_ONE=1):
            for _ in range(200):
                log.clear()
                comb.propose(model, _StubState())
                rj = [n for n in log if n in RJ]
                self.assertEqual(len(rj), 1, log)
                # the drawn RJ move sits where the pair sat: after warm,
                # before ridge; everything else exactly once, in order
                self.assertEqual(log, ["rj_warm_pe", rj[0], "gb_ridge_gibbs"])

    def test_fraction_is_honoured_and_reproducible(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        picks = []
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.8):
            model = _StubModel(seed=11)
            for _ in range(2000):
                log.clear()
                comb.propose(model, _StubState())
                picks.append([n for n in log if n in RJ][0])
            frac = np.mean([p == "rj_fstat_pe" for p in picks])
            # 2000 Bernoulli(0.8) draws: sigma = 0.009 -> a 5-sigma band
            self.assertGreater(frac, 0.755)
            self.assertLess(frac, 0.845)
            # same seed -> identical sequence (the draw comes from model.random)
            log2 = []
            _, comb2 = _make(PE_NAMES, log2, weighted_cycle=True)
            model2 = _StubModel(seed=11)
            picks2 = []
            for _ in range(2000):
                log2.clear()
                comb2.propose(model2, _StubState())
                picks2.append([n for n in log2 if n in RJ][0])
            self.assertEqual(picks, picks2)

    def test_fraction_endpoints(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        model = _StubModel()
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=1.0):
            for _ in range(20):
                log.clear()
                comb.propose(model, _StubState())
                self.assertIn("rj_fstat_pe", log)
                self.assertNotIn("rj_prior_pe", log)
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.0):
            for _ in range(20):
                log.clear()
                comb.propose(model, _StubState())
                self.assertIn("rj_prior_pe", log)
                self.assertNotIn("rj_fstat_pe", log)

    def test_bad_fraction_raises(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=1.5):
            with self.assertRaises(ValueError):
                comb.propose(_StubModel(), _StubState())

    def test_accepted_sums_over_executed_moves(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=1):
            _, accepted = comb.propose(_StubModel(), _StubState())
        # warm + one RJ + ridge = 3 executed moves, each accepting ones
        np.testing.assert_array_equal(accepted, 3.0 * np.ones((2, 4)))

    # ---- the mode must NOT leak ------------------------------------------------
    def test_search_stage_untouched(self):
        log = []
        names = ("rj_fstat_search", "rj_prior_removal", "rj_fstat_pe", "rj_prior_pe")
        _, comb = _make(names, log, kind="search")
        with _env(GB_PE_RJ_DRAW_ONE=1):
            comb.propose(_StubModel(), _StubState())
        self.assertEqual(log, list(names))  # sequential, all of them

    def test_knob_unset_keeps_weighted_cycle(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=0):
            model = _StubModel(seed=99)
            comb.propose(model, _StubState())
        mirror = np.random.RandomState(99)
        expected = list(mirror.choice(4, size=4, replace=True, p=np.full(4, 0.25)))
        self.assertEqual(log, [PE_NAMES[k] for k in expected])

    # ---- rj_warm_pe in the draw (user ruling 2026-10-02: "put the warm proposal
    # in with the other two rj moves. Draw them at 0.45 prior, 0.45 fstat,
    # 0.1 warm start") -------------------------------------------------------

    def test_warm_joins_the_draw_at_the_configured_weights(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        trio = ("rj_warm_pe", "rj_fstat_pe", "rj_prior_pe")
        picks = []
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.45,
                  GB_PE_RJ_WARM_FRACTION=0.1):
            model = _StubModel(seed=21)
            for _ in range(4000):
                log.clear()
                comb.propose(model, _StubState())
                rj = [n for n in log if n in trio]
                self.assertEqual(len(rj), 1, log)               # exactly ONE of the three
                self.assertEqual(log, [rj[0], "gb_ridge_gibbs"])  # in the first member's slot
                picks.append(rj[0])
        f = np.mean([p == "rj_fstat_pe" for p in picks])
        pr = np.mean([p == "rj_prior_pe" for p in picks])
        w = np.mean([p == "rj_warm_pe" for p in picks])
        # 4000 draws: sigma ~0.008 for the 0.45s, ~0.005 for the 0.1 -> 5-sigma bands
        self.assertTrue(0.41 < f < 0.49, f)
        self.assertTrue(0.41 < pr < 0.49, pr)
        self.assertTrue(0.075 < w < 0.125, w)

    def test_warm_fraction_zero_keeps_warm_outside_the_draw(self):
        """The pre-2026-10-02 behaviour: rj_warm_pe runs every iteration and
        the draw is between fstat and prior only."""
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        model = _StubModel(seed=4)
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.5, GB_PE_RJ_WARM_FRACTION=0):
            for _ in range(50):
                log.clear()
                comb.propose(model, _StubState())
                self.assertEqual(log[0], "rj_warm_pe")
                self.assertEqual(len([n for n in log if n in RJ]), 1)

    def test_warm_fraction_on_a_stage_without_warm_is_redistributed(self):
        log = []
        _, comb = _make(("rj_fstat_pe", "rj_prior_pe", "gb_ridge_gibbs"), log, weighted_cycle=True)
        picks = []
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.45, GB_PE_RJ_WARM_FRACTION=0.1):
            model = _StubModel(seed=8)
            for _ in range(2000):
                log.clear()
                comb.propose(model, _StubState())
                picks.append([n for n in log if n in RJ][0])
        f = np.mean([p == "rj_fstat_pe" for p in picks])
        self.assertTrue(0.45 < f < 0.55, f)                    # 0.45 / (0.45 + 0.45)

    def test_fractions_summing_over_one_raise(self):
        log = []
        _, comb = _make(PE_NAMES, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.95, GB_PE_RJ_WARM_FRACTION=0.1):
            with self.assertRaises(ValueError):
                comb.propose(_StubModel(), _StubState())

    # ---- two F-stat moves (user ruling 2026-10-12: "two fstat moves that will
    # split their share of the total moves") ------------------------------------

    FOUR = ("rj_warm_pe", "rj_fstat_pe", "rj_fstat_pe_residual", "rj_prior_pe",
            "gb_ridge_gibbs")
    RJ4 = ("rj_warm_pe", "rj_fstat_pe", "rj_fstat_pe_residual", "rj_prior_pe")

    def _picks(self, names, n, seed, **env):
        log = []
        _, comb = _make(names, log, weighted_cycle=True)
        picks = []
        with _env(GB_PE_RJ_DRAW_ONE=1, **env):
            model = _StubModel(seed=seed)
            for _ in range(n):
                log.clear()
                comb.propose(model, _StubState())
                rj = [x for x in log if x in self.RJ4]
                self.assertEqual(len(rj), 1, log)                 # exactly ONE RJ move
                self.assertEqual(log, [rj[0], "gb_ridge_gibbs"])  # in the first member's slot
                picks.append(rj[0])
        return picks

    def test_config_splits_the_fstat_share_and_sums_to_one(self):
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.45,
                  GB_PE_RJ_WARM_FRACTION=0.1, GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=0.5):
            on, p = GFCombineMove._pe_rj_draw_one_config()
        self.assertTrue(on)
        self.assertAlmostEqual(p["rj_fstat_pe"], 0.225)
        self.assertAlmostEqual(p["rj_fstat_pe_residual"], 0.225)
        self.assertAlmostEqual(p["rj_prior_pe"], 0.45)
        self.assertAlmostEqual(p["rj_warm_pe"], 0.1)
        self.assertAlmostEqual(sum(p.values()), 1.0)
        # the split is honoured, the total F-stat share and prior / warm are not
        # touched by it
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.45,
                  GB_PE_RJ_WARM_FRACTION=0.1, GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=0.2):
            _, q = GFCombineMove._pe_rj_draw_one_config()
        self.assertAlmostEqual(q["rj_fstat_pe"], 0.36)
        self.assertAlmostEqual(q["rj_fstat_pe_residual"], 0.09)
        self.assertAlmostEqual(q["rj_prior_pe"], 0.45)
        self.assertAlmostEqual(q["rj_warm_pe"], 0.1)
        self.assertAlmostEqual(sum(q.values()), 1.0)

    def test_the_default_split_is_one_half(self):
        with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.4):
            os.environ.pop("GB_PE_RJ_FSTAT_RESIDUAL_SPLIT", None)
            _, p = GFCombineMove._pe_rj_draw_one_config()
        self.assertAlmostEqual(p["rj_fstat_pe"], 0.2)
        self.assertAlmostEqual(p["rj_fstat_pe_residual"], 0.2)

    def test_both_fstat_moves_drawn_at_the_split(self):
        picks = self._picks(self.FOUR, 6000, 31, GB_PE_RJ_FSTAT_FRACTION=0.45,
                            GB_PE_RJ_WARM_FRACTION=0.1,
                            GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=0.5)
        fr = {n: np.mean([p == n for p in picks]) for n in self.RJ4}
        # 6000 draws: sigma ~0.0054 at 0.225, ~0.0064 at 0.45, ~0.0039 at 0.1
        self.assertTrue(0.198 < fr["rj_fstat_pe"] < 0.252, fr)
        self.assertTrue(0.198 < fr["rj_fstat_pe_residual"] < 0.252, fr)
        self.assertTrue(0.418 < fr["rj_prior_pe"] < 0.482, fr)
        self.assertTrue(0.08 < fr["rj_warm_pe"] < 0.12, fr)

    def test_split_endpoints_degrade_to_one_fstat_move(self):
        """0 / 1: the zero-share move is never drawn and so NEVER RUNS (it must
        not fall back to running every iteration the way a zero-mass warm move
        does), and the other takes the whole F-stat share."""
        for split, gone, kept in ((0.0, "rj_fstat_pe_residual", "rj_fstat_pe"),
                                  (1.0, "rj_fstat_pe", "rj_fstat_pe_residual")):
            picks = self._picks(self.FOUR, 3000, 7, GB_PE_RJ_FSTAT_FRACTION=0.45,
                                GB_PE_RJ_WARM_FRACTION=0.1,
                                GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=split)
            self.assertNotIn(gone, picks, split)
            f = np.mean([p == kept for p in picks])
            self.assertTrue(0.42 < f < 0.48, (split, f))

    def test_a_stage_with_only_the_galaxy_move_draws_as_before(self):
        """A pre-2026-10-12 composition (no residual move): rj_fstat_pe takes the
        WHOLE F-stat share -- the draws are the old three-member draws exactly,
        whatever the split says."""
        names = ("rj_warm_pe", "rj_fstat_pe", "rj_prior_pe", "gb_ridge_gibbs")
        picks = self._picks(names, 2000, 21, GB_PE_RJ_FSTAT_FRACTION=0.45,
                            GB_PE_RJ_WARM_FRACTION=0.1,
                            GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=0.5)
        # the old algorithm, by hand: members (fstat, prior, warm) at
        # (0.45, 0.45, 0.1), one uniform per iteration from the same RNG
        mirror = np.random.RandomState(21)
        cdf = np.cumsum([0.45, 0.45, 0.1])
        old = [("rj_fstat_pe", "rj_prior_pe", "rj_warm_pe")[
            min(int(np.searchsorted(cdf, mirror.uniform(), side="right")), 2)]
            for _ in range(2000)]
        # not assertEqual on the lists: unittest's diff of two long differing
        # lists runs for many minutes; name the first mismatch instead
        bad = next((i for i, (a, b) in enumerate(zip(picks, old)) if a != b), None)
        self.assertIsNone(bad, None if bad is None else
                          f"draw {bad}: {picks[bad]} != the old draw {old[bad]}")
        self.assertEqual(len(picks), len(old))

    def test_a_stage_with_only_the_residual_move_still_draws(self):
        names = ("rj_fstat_pe_residual", "rj_prior_pe", "gb_ridge_gibbs")
        picks = self._picks(names, 3000, 5, GB_PE_RJ_FSTAT_FRACTION=0.45,
                            GB_PE_RJ_WARM_FRACTION=0.0,
                            GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=0.5)
        f = np.mean([p == "rj_fstat_pe_residual" for p in picks])
        self.assertTrue(0.42 < f < 0.48, f)          # the whole 0.45, not 0.225

    def test_bad_split_raises(self):
        log = []
        _, comb = _make(self.FOUR, log, weighted_cycle=True)
        for bad in ("1.5", "-0.1", "nan"):
            with _env(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=bad):
                with self.assertRaises(ValueError, msg=bad):
                    comb.propose(_StubModel(), _StubState())

    def test_pe_stage_without_the_pair_falls_through(self):
        log = []
        names = ("rj_warm_pe", "rj_fstat_pe", "gb_ridge_gibbs")
        _, comb = _make(names, log, weighted_cycle=True)
        with _env(GB_PE_RJ_DRAW_ONE=1):
            model = _StubModel(seed=5)
            comb.propose(model, _StubState())
        mirror = np.random.RandomState(5)
        expected = list(mirror.choice(3, size=3, replace=True, p=np.full(3, 1 / 3)))
        self.assertEqual(log, [names[k] for k in expected])


if __name__ == "__main__":
    unittest.main()
