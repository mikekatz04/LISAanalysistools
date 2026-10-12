"""The recipe installs the fan-out on every addremove/PSD move it builds."""

import unittest

from lisatools.globalfit import recipe as recipe_mod
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.run import _fanout_unready_moves


class FanoutInstallTest(unittest.TestCase):
    def test_ported_families_are_not_unready(self):
        a = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
        a.gf_move_name = "mbh_pe"
        p = PSDMove.__new__(PSDMove)
        p.gf_move_name = "psd_pe"
        unready, head_only = _fanout_unready_moves([a, p])
        self.assertEqual((unready, head_only), ([], []))

    def test_builders_call_install(self):
        # the builders are exercised end to end by the gated smokes; here we pin
        # the call sites so a refactor cannot silently drop them.
        # 3 (pre-Plan-4) + 14 (Plan 4, Task 5, after the ridge fix) = 17,
        # + 1 for rj_fstat_pe_residual (the second PE F-stat move, user
        # ruling 2026-10-12) = 18:
        #   - addremove (SingleSourcePEBuilder.build, one call site shared by
        #     MBH/EMRI/SOBBH): 1
        #   - noise (build_psd_moves: search_move, pe_move): 2
        #   - GB (build_gb_moves: 14 GBSpecial* moves): 14
        #   - VGB (build_vgb_moves: the vgb move): 1
        src = open(recipe_mod.__file__).read()
        # both ridge-gibbs moves (gb_ridge_gibbs and its vgb twin) are plain
        # eryn GFRidgeGibbsMove objects -- head-only, so neither gets an
        # install_walker_fanout and the count stays 18.
        self.assertEqual(src.count("install_walker_fanout(curr)"), 18)
        self.assertIn("gb_pe_residual_move.install_walker_fanout(curr)", src)


class TemperSeedBaseStampTest(unittest.TestCase):
    """The recipe stamps the temper-RNG seed base on every GB/VGB move.

    ``GBSpecialBase._make_temper_rng`` falls back to this integer whenever
    ``_rank_rng_seed`` is unset -- a single-rank run through
    ``_propose_legacy`` AND the orchestrator at one compute rank -- so an
    unstamped move puts the two bodies back on different streams (NEW-E2).
    """

    def test_both_builders_stamp(self):
        src = open(recipe_mod.__file__).read()
        # one call at the end of build_gb_moves, one at the end of
        # build_vgb_moves (the definition itself is not a call)
        self.assertEqual(src.count("_stamp_temper_seed_base("), 3)

    def test_the_stamp_is_derived_deterministically_from_the_run_seed(self):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        class _Move:
            gf_temper_seed_base = None

        moves = [_Move(), _Move()]
        base = recipe_mod._stamp_temper_seed_base(moves, 4242)
        self.assertIsInstance(base, int)
        self.assertEqual([m.gf_temper_seed_base for m in moves], [base, base])
        # deterministic, and NOT the run seed itself (domain separation keeps
        # it away from the per-rank seeds derive_rank_seed spawns off 4242)
        self.assertEqual(recipe_mod._stamp_temper_seed_base([], 4242), base)
        self.assertNotEqual(base, 4242)
        self.assertNotEqual(recipe_mod._stamp_temper_seed_base([], 4243), base)
        # the real attribute name, on the real class
        self.assertIsNone(GBSpecialBase.gf_temper_seed_base)

    def test_no_run_seed_leaves_every_stream_on_entropy(self):
        class _Move:
            gf_temper_seed_base = 7

        move = _Move()
        self.assertIsNone(recipe_mod._stamp_temper_seed_base([move], None))
        self.assertIsNone(move.gf_temper_seed_base)

    def test_a_move_without_the_attribute_is_left_alone(self):
        # the ridge-gibbs fiber move is a plain eryn move with no Generator
        class _Plain:
            pass

        plain = _Plain()
        recipe_mod._stamp_temper_seed_base([plain], 4242)
        self.assertFalse(hasattr(plain, "gf_temper_seed_base"))


if __name__ == "__main__":
    unittest.main()
