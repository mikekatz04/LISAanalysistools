"""Tests for the stock global-fit API (lisatools.globalfit.stock).

Everything here runs without GPU / mojito / Sangria data: it exercises the
cheap configured-but-unbuilt layer (registry, knobs, recipe/move editing,
branch composition, pickle/deepcopy safety, env-default resolution) — the
heavy build paths are covered by the parity/smoke harnesses.
"""

import copy
import os
import shutil
import tempfile
import warnings
import pickle
import unittest

import numpy as np

from lisatools.globalfit import FunctionMove, Move, MoveBuildContext, Recipe, Stage
from lisatools.globalfit.stock import erebor


class _EnvGuard:
    """Context manager: set env vars, restore on exit."""

    def __init__(self, **env):
        self.env = env
        self._old = {}

    def __enter__(self):
        for k, v in self.env.items():
            self._old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, old in self._old.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


class StockRegistryTest(unittest.TestCase):
    def test_options(self):
        names = [name for name, _ in erebor.get_stock_options()]
        self.assertIn("gb_no_fg", names)
        self.assertIn("all_sources", names)
        self.assertIn("full_year_combined", names)
        for _, description in erebor.get_stock_options():
            self.assertTrue(description)

    def test_options_alphabetical(self):
        names = [name for name, _ in erebor.get_stock_options()]
        self.assertEqual(names, sorted(names))
        self.assertEqual(list(erebor.__stock_globalfit_options__), sorted(names))

    def test_unknown_option(self):
        with self.assertRaises(ValueError) as ctx:
            erebor.get_stock("nope")
        self.assertIn("gb_no_fg", str(ctx.exception))

    def test_module_attributes_are_default_instances(self):
        self.assertIsInstance(erebor.gb_no_fg, erebor.GBNoForegroundGlobalFit)
        self.assertFalse(erebor.gb_no_fg.built)

    def test_calling_module_default_clones(self):
        fresh = erebor.gb_no_fg(nwalkers=17)
        self.assertEqual(fresh.nwalkers, 17)
        self.assertNotEqual(erebor.gb_no_fg.nwalkers, 17)
        fresh.gb.center_freq = 9e-3
        self.assertNotEqual(erebor.gb_no_fg.gb.center_freq, 9e-3)

    def test_unknown_kwarg_raises(self):
        with self.assertRaises(TypeError):
            erebor.get_stock("gb_no_fg", not_a_knob=3)


class GlobalFitSetupTest(unittest.TestCase):
    def test_currentinfo_alias(self):
        from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFitSetup

        self.assertIs(CurrentInfoGlobalFit, GlobalFitSetup)
        self.assertTrue(issubclass(erebor.GBNoForegroundGlobalFit, GlobalFitSetup))

    def test_describe_full_lists_all_fields(self):
        fit = erebor.get_stock("gb_no_fg")
        headline = fit.describe()
        full = fit.describe(full=True)
        # ``random_seed`` is a general-only field (never a headline knob), so
        # it appears only in the full illustration.
        self.assertNotIn("random_seed", headline)
        self.assertIn("random_seed", full)
        self.assertGreater(len(full.splitlines()), len(headline.splitlines()))

    def test_summarize_run_is_a_globalfitsetup_method(self):
        # The stock gallery (LATW 02) calls curr.summarize_run(...) on a built
        # fit instead of defining a local helper; keep that contract.
        from lisatools.globalfit.run import GlobalFitSetup

        self.assertTrue(callable(getattr(GlobalFitSetup, "summarize_run", None)))
        self.assertTrue(callable(getattr(erebor.get_stock("gb_no_fg"), "summarize_run", None)))

    def test_describe_reports_built_products_only_when_built(self):
        # Unbuilt: no built-product annotations (the built path is exercised by
        # the gallery notebook, which builds for real).
        self.assertNotIn("[built:", erebor.get_stock("gb_no_fg").describe())

    def test_all_settings_structure(self):
        fit = erebor.get_stock("gb_no_fg")
        s = fit.all_settings()
        self.assertTrue(
            {"option_name", "headline", "general", "branches", "recipe", "setup_function"}
            <= set(s)
        )
        self.assertIn("num_iterations", s["general"])
        self.assertIn("gb", s["branches"])
        self.assertIn("num_repeat_proposals", s["branches"]["gb"])


class KnobTest(unittest.TestCase):
    def test_headline_knobs_delegate_to_general(self):
        fit = erebor.get_stock("gb_no_fg", nwalkers=8, base_file_name="knob_test")
        self.assertEqual(fit.general.nwalkers, 8)
        self.assertEqual(fit.general.base_file_name, "knob_test")
        fit.ntemps = 5
        self.assertEqual(fit.general.ntemps, 5)

    def test_env_default_resolution_order(self):
        with _EnvGuard(NWALKERS="11"):
            self.assertEqual(erebor.get_stock("gb_no_fg").nwalkers, 11)
            # explicit kwarg beats the env var
            self.assertEqual(erebor.get_stock("gb_no_fg", nwalkers=2).nwalkers, 2)
        self.assertEqual(erebor.get_stock("gb_no_fg").nwalkers, 4)  # hard default

    def test_clone_honors_debug_preset(self):
        with _EnvGuard(GB_DEBUG=None, TOBS_TARGET=None, NWALKERS=None, NTEMPS=None,
                       CHUNKED_NT_SUB=None, NUM_ITERATIONS=None):
            clone = erebor.gb_no_fg(debug=True)
            self.assertTrue(clone.debug)
            self.assertEqual(clone.tobs_target, 3 * 86400.0)
            self.assertEqual(clone.nwalkers, 3)
            self.assertEqual(clone.gb.nt_sub, 64)

    def test_gb_debug_preset(self):
        with _EnvGuard(GB_DEBUG="1", TOBS_TARGET=None, NWALKERS=None, NTEMPS=None,
                       CHUNKED_NT_SUB=None, CHUNKED_N_PAD=None, CHUNKED_N_SPARSE=None,
                       CHUNKED_N_CP_SIG=None, CHUNKED_N_CP_ORBIT=None, NUM_ITERATIONS=None):
            fit = erebor.get_stock("gb_no_fg")
            self.assertTrue(fit.debug)
            self.assertEqual(fit.tobs_target, 3 * 86400.0)
            self.assertEqual(fit.nwalkers, 3)

    def test_compute_backend_env_knobs(self):
        with _EnvGuard(GPU_BACKEND="cuda12x", GPUS="2,3", USE_GPU="0"):
            fit = erebor.get_stock("gb_no_fg")
            self.assertEqual(fit.general.gpu_backend, "cuda12x")
            self.assertEqual(fit.general.gpus, [2, 3])
            self.assertIs(fit.general.use_gpu, False)
            # USE_GPU=0 forces the CPU path regardless of the flavor knob
            gs = fit.make_general_settings()
            self.assertIsNone(gs.gpus)

    def test_construction_is_cheap(self):
        # No directories created, and a nonexistent data path is fine
        # (nothing touches the filesystem until build()).
        fit = erebor.get_stock(
            "gb_no_fg",
            mojito_data_path="/nonexistent/nowhere/",
            file_store_dir="./_stock_test_should_not_exist/",
        )
        self.assertFalse(fit.built)
        self.assertFalse(os.path.exists("./_stock_test_should_not_exist/"))

    def test_prebuild_guard_message(self):
        fit = erebor.get_stock("gb_no_fg")
        with self.assertRaises(AttributeError) as ctx:
            fit.branch_names
        self.assertIn("build()", str(ctx.exception))


class EnvNamingTest(unittest.TestCase):
    """An env knob is the capitalized attribute name; legacy spellings alias to it."""

    def test_canonical_name_matches_attribute(self):
        with _EnvGuard(NUM_ITERATIONS="7", DATA_MODE="synthetic"):
            fit = erebor.get_stock("gb_no_fg")
            self.assertEqual(fit.general.num_iterations, 7)      # NUM_ITERATIONS
            self.assertEqual(fit.general.data_mode, "synthetic")  # DATA_MODE

    def test_canonical_wins_over_legacy_alias(self):
        with _EnvGuard(NUM_ITERATIONS="7", GF_NUM_ITER="99"):
            self.assertEqual(erebor.get_stock("gb_no_fg").general.num_iterations, 7)

    def test_legacy_alias_still_honored_and_warns(self):
        # A hard rename would SILENTLY ignore the old name (env vars are not
        # validated), so the legacy spelling must keep working — loudly.
        with _EnvGuard(NUM_ITERATIONS=None, GF_NUM_ITER="42"):
            with self.assertWarns(DeprecationWarning):
                fit = erebor.get_stock("gb_no_fg")
            self.assertEqual(fit.general.num_iterations, 42)

    def test_legacy_data_processor_alias(self):
        with _EnvGuard(DATA_MODE=None, DATA_PROCESSOR="synthetic"):
            with self.assertWarns(DeprecationWarning):
                fit = erebor.get_stock("gb_no_fg")
            self.assertEqual(fit.general.data_mode, "synthetic")

    def test_alias_table_keys_are_upper_case(self):
        from lisatools.globalfit.stock.base import ENV_ALIASES

        for canonical, legacy in ENV_ALIASES.items():
            self.assertEqual(canonical, canonical.upper())
            for old in legacy:
                self.assertEqual(old, old.upper())
                self.assertNotEqual(old, canonical)


class PickleTest(unittest.TestCase):
    def _roundtrip(self, fit):
        clone = pickle.loads(pickle.dumps(copy.deepcopy(fit)))
        self.assertEqual(clone.nwalkers, fit.nwalkers)
        self.assertEqual(clone.recipe.move_names(), fit.recipe.move_names())
        self.assertEqual(list(clone.branches), list(fit.branches))
        self.assertIs(clone.setup_function, fit.setup_function)
        return clone

    def test_all_variants_roundtrip(self):
        for name in ("gb_no_fg", "all_sources", "full_year_combined"):
            with self.subTest(variant=name):
                self._roundtrip(erebor.get_stock(name))

    def test_roundtrip_after_mutation(self):
        fit = erebor.get_stock("gb_no_fg", nwalkers=13)
        fit.gb.a_lims = [1e-25, 1e-20]
        fit.add_move("rj_fstat_mcmc", branch="gb", stage="gb_pe")
        clone = self._roundtrip(fit)
        self.assertEqual(clone.gb.a_lims, [1e-25, 1e-20])


def _fn_move(model, state):
    """Named module-level move so fits carrying it stay picklable."""
    return state, None


class RecipeTest(unittest.TestCase):
    def _recipe(self):
        return Recipe(
            [
                Stage("s1", kind="search", moves=[Move("a"), Move("b")]),
                Stage("s2", kind="pe", moves=[Move("c")]),
            ]
        )

    def test_pop_unknown_lists_options(self):
        r = self._recipe()
        with self.assertRaises(KeyError) as ctx:
            r.pop_move("zzz")
        for name in ("a", "b", "c"):
            self.assertIn(name, str(ctx.exception))

    def test_add_placements(self):
        r = self._recipe()
        r.add_move(Move("d"), before="b")
        self.assertEqual([m.name for m in r._stage("s1").moves], ["a", "d", "b"])
        r.add_move(Move("e"), stage="s2", index=0)
        self.assertEqual([m.name for m in r._stage("s2").moves], ["e", "c"])
        r.add_move(Move("f"), after="c")
        self.assertEqual([m.name for m in r._stage("s2").moves], ["e", "c", "f"])

    def test_duplicate_name_rejected(self):
        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(Move("a"), stage="s2")

    def test_ambiguous_stage_required(self):
        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(Move("g"))  # two stages, no placement

    def test_conflicting_placements_rejected(self):
        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(Move("g"), before="a", index=0)

    def test_stage_composition(self):
        r = self._recipe()
        r.add_stage(Stage("s0", kind="pe", moves=[Move("z")]), before="s1")
        self.assertEqual([s.name for s in r.stages], ["s0", "s1", "s2"])
        popped = r.pop_stage("s1")
        self.assertEqual(popped.name, "s1")
        self.assertEqual([s.name for s in r.stages], ["s0", "s2"])

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            Stage("bad", kind="wiggle")

    def test_runtime_move_needs_name(self):
        class _Proposer:
            def propose(self, model, state):
                return state, None

        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(_Proposer(), stage="s2")  # has .propose, no name
        r.add_move(_Proposer(), stage="s2", name="prop")
        self.assertIn("prop", r.move_names())

    def test_non_move_rejected(self):
        r = self._recipe()
        with self.assertRaises(TypeError):
            r.add_move(object(), stage="s2")

    def test_coercion(self):
        r = self._recipe()
        # str -> stock-name Move
        mv = r.add_move("rj_prior", branch="gb", stage="s2")
        self.assertIsInstance(mv, Move)
        self.assertTrue(mv.is_stock)
        self.assertEqual(mv.branch, "gb")
        # plain fn -> FunctionMove
        fm = r.add_move(_fn_move, stage="s2")
        self.assertIsInstance(fm, FunctionMove)
        self.assertFalse(fm.is_stock)
        self.assertEqual(fm.name, "_fn_move")
        self.assertIn("stock", r.list_moves())

    def test_add_move_auto_creates_main_stage(self):
        r = Recipe()
        mv = r.add_move(_fn_move)
        self.assertEqual([s.name for s in r.stages], ["main"])
        self.assertEqual(r.stages[0].kind, "pe")
        self.assertEqual(r.move_names(), [mv.name])

    def test_stock_names_filter(self):
        r = self._recipe()
        r.add_move(_fn_move, stage="s2")
        self.assertEqual(r.stock_names(), ["a", "b", "c"])

    def test_recipe_pickles_without_runtime(self):
        r = self._recipe()
        r.stock_moves = {"a": object()}  # unpicklable runtime product
        r.recipe.append({"name": "s1", "adjust": object(), "status": False})
        clone = pickle.loads(pickle.dumps(copy.deepcopy(r)))
        self.assertEqual(clone.move_names(), r.move_names())
        self.assertEqual(clone.recipe, [])
        self.assertEqual(clone.stock_moves, {})


class _StubCurr:
    """Minimal curr stub for Recipe/Stage.setup tests."""

    class _EI:
        branch_names = ["line"]

    engine_info = _EI()
    _info_branches = ()


def _stub_ctx(**overrides):
    kwargs = dict(
        recipe=None, engine_info=None, curr=_StubCurr(), acs=None,
        priors={}, state=None, stock_moves={}, ntemps=2, nwalkers=4,
    )
    kwargs.update(overrides)
    return MoveBuildContext(**kwargs)


class MoveSetupTest(unittest.TestCase):
    def test_stock_lookup_missing_lists_options(self):
        with self.assertRaises(ValueError) as ctx:
            Move("nope").materialize(_stub_ctx(stock_moves={"psd_pe": object()}))
        self.assertIn("psd_pe", str(ctx.exception))
        self.assertIn("setup(ctx)", str(ctx.exception))

    def test_stock_lookup_resolves(self):
        runtime = object()
        mv = Move("psd_pe")
        self.assertIs(mv.materialize(_stub_ctx(stock_moves={"psd_pe": runtime})), runtime)
        self.assertIs(mv.runtime, runtime)

    def test_subclass_setup_none_means_self(self):
        fm = FunctionMove(_fn_move)
        self.assertIs(fm.materialize(_stub_ctx()), fm)
        self.assertEqual(fm.accepted.shape, (2, 4))

    def test_stage_setup_materializes(self):
        from lisatools.globalfit.moves import GFCombineMove
        from lisatools.globalfit.recipe import PERecipeStep

        st = Stage("main", kind="pe", moves=[FunctionMove(_fn_move, branch="line")])
        step = st.setup(_stub_ctx())
        self.assertIsInstance(step, PERecipeStep)
        combined = step.moves[0]
        self.assertIsInstance(combined, GFCombineMove)
        self.assertEqual(len(combined.moves), 1)
        # CombineMove's ``accepted`` setter propagates onto the sub-moves (the
        # combine-level array itself is initialized later by the sampler).
        self.assertEqual(combined.moves[0].accepted.shape, (2, 4))

    def test_stage_setup_rejects_unknown_branch(self):
        st = Stage("main", moves=[FunctionMove(_fn_move, branch="nope")])
        with self.assertRaises(ValueError):
            st.setup(_stub_ctx())

    def test_recipe_setup_requires_move_per_info_branch(self):
        class _Curr(_StubCurr):
            _info_branches = ("line",)

        r = Recipe([Stage("main", moves=[FunctionMove(_fn_move)])])  # no branch tag
        with self.assertRaises(ValueError) as ctx:
            r.setup(_stub_ctx(curr=_Curr()))
        self.assertIn("line", str(ctx.exception))

    def test_recipe_setup_registers_steps(self):
        r = Recipe([Stage("main", moves=[FunctionMove(_fn_move, branch="line")])])
        r.setup(_stub_ctx())
        self.assertEqual(len(r.recipe), 1)
        self.assertIsNotNone(r.get_step("main"))
        with self.assertRaises(KeyError):
            r.get_step("zzz")


class BranchInfoTest(unittest.TestCase):
    def _fit(self):
        return erebor.get_stock("blank", nwalkers=4, ntemps=2)

    def test_blank_registered_zero_branches(self):
        fit = self._fit()
        self.assertEqual(list(fit.branches), [])
        self.assertEqual([s.name for s in fit.recipe.stages], ["main"])
        self.assertEqual(fit.recipe.move_names(), [])

    def test_add_branch_info_path(self):
        from eryn.prior import uniform_dist

        fit = self._fit()
        fit.add_branch(
            "line", ndim=2,
            priors={0: uniform_dist(0.0, 1.0), 1: uniform_dist(0.0, 1.0)},
            moves=[_fn_move],
        )
        self.assertIn("line", fit.branches)
        self.assertEqual(fit.line.ndim, 2)
        self.assertEqual(fit.line.nleaves_max, 1)
        self.assertEqual(fit.line.nleaves_min, 1)  # fixed-leaf default
        self.assertIn("line", fit.line.priors)
        self.assertEqual(fit._info_branches, ["line"])
        mv = fit.recipe.get_move("_fn_move")
        self.assertIsInstance(mv, FunctionMove)
        self.assertEqual(mv.branch, "line")

    def test_add_branch_info_requires_ndim_and_priors(self):
        fit = self._fit()
        with self.assertRaises(TypeError):
            fit.add_branch("line", ndim=2)  # no priors

    def test_zero_noise_default_adjustable(self):
        from lisatools.globalfit.stock.erebor.injections import (
            SyntheticNoiseProcessingStep,
            ZeroDataProcessingStep,
        )

        gs = self._fit().make_general_settings()
        self.assertIs(gs.data_processor_class, ZeroDataProcessingStep)
        gs_noise = erebor.get_stock(
            "blank", nwalkers=4, ntemps=2, include_noise=True
        ).make_general_settings()
        self.assertIs(gs_noise.data_processor_class, SyntheticNoiseProcessingStep)

    def test_info_fit_pickles(self):
        from eryn.prior import uniform_dist

        fit = self._fit()
        fit.add_branch(
            "line", ndim=1, priors={0: uniform_dist(0.0, 1.0)}, moves=[_fn_move]
        )
        clone = pickle.loads(pickle.dumps(copy.deepcopy(fit)))
        self.assertEqual(list(clone.branches), ["line"])
        self.assertEqual(clone.recipe.move_names(), ["_fn_move"])
        self.assertEqual(clone._info_branches, ["line"])


class VerboseKnobTest(unittest.TestCase):
    def test_default_quiet_everywhere(self):
        for name in ("blank", "gb_no_fg", "all_sources", "full_year_combined", "noise_only"):
            with self.subTest(variant=name):
                fit = erebor.get_stock(name)
                self.assertFalse(fit.general.verbose)
                self.assertFalse(fit.verbose)  # headline alias
                # stock recipes no longer hardcode the combine progress bar on
                for st in fit.recipe.stages:
                    self.assertNotIn("verbose", st.combine_kwargs)

    def test_headline_knob_and_env(self):
        fit = erebor.get_stock("blank", verbose=True)
        self.assertTrue(fit.general.verbose)
        fit.verbose = False  # headline alias writes through to general
        self.assertFalse(fit.general.verbose)
        with _EnvGuard(VERBOSE="1"):
            self.assertTrue(erebor.get_stock("blank").general.verbose)
        with _EnvGuard(VERBOSE="0"):
            self.assertFalse(erebor.get_stock("blank", ).general.verbose)


class BranchCompositionTest(unittest.TestCase):
    def test_add_remove_swap(self):
        fit = erebor.get_stock("all_sources")
        self.assertEqual(
            list(fit.branches),
            ["gb", "vgb", "psd", "galfor", "mbh", "emri", "sobbh"],
        )
        removed = fit.remove_branch("galfor")
        self.assertNotIn("galfor", fit.branches)
        fit.add_branch("galfor", removed)
        self.assertIn("galfor", fit.branches)
        with self.assertRaises(KeyError):
            fit.remove_branch("zzz")
        with self.assertRaises(TypeError):
            fit.add_branch("bad", object())
        with self.assertRaises(ValueError):
            fit.add_branch("recipe", removed)  # clashes with an attribute

    def test_fit_level_move_aliases(self):
        fit = erebor.get_stock("all_sources")
        popped = fit.pop_move("sobbh_pe")
        self.assertEqual(popped.name, "sobbh_pe")
        self.assertNotIn("sobbh_pe", fit.recipe.move_names())
        fit.add_move(popped, stage="full_pe", after="emri_pe")
        self.assertIn("sobbh_pe", fit.recipe.move_names())


class WaveformPathDefaultsTest(unittest.TestCase):
    def test_full_year_defaults(self):
        with _EnvGuard(MBHB_IDS="0", EMRI_IDS="1", SOBHB_IDS="2", USE_TDIONFLY=None):
            fit = erebor.get_stock("full_year_combined")
            self.assertEqual(list(fit.branches), ["mbh", "emri", "sobbh"])
            self.assertFalse(fit.mbh.use_tdionfly)   # MBH: legacy
            self.assertTrue(fit.sobbh.use_tdionfly)  # SOBBH: TDI-on-the-fly

    def test_use_tdionfly_env_flips_both(self):
        with _EnvGuard(MBHB_IDS="0", EMRI_IDS="1", SOBHB_IDS="2", USE_TDIONFLY="0"):
            fit = erebor.get_stock("full_year_combined")
            self.assertFalse(fit.mbh.use_tdionfly)
            self.assertFalse(fit.sobbh.use_tdionfly)
        with _EnvGuard(MBHB_IDS="0", EMRI_IDS="1", SOBHB_IDS="2", USE_TDIONFLY="1"):
            fit = erebor.get_stock("full_year_combined")
            self.assertTrue(fit.mbh.use_tdionfly)
            self.assertTrue(fit.sobbh.use_tdionfly)

    def test_zero_leaf_branches_dropped(self):
        with _EnvGuard(MBHB_IDS="", EMRI_IDS="1", SOBHB_IDS="", USE_TDIONFLY=None):
            fit = erebor.get_stock("full_year_combined")
            self.assertEqual(list(fit.branches), ["emri"])
            self.assertEqual(fit.recipe.move_names(), ["emri_pe"])


class DataProcessorSwapTest(unittest.TestCase):
    """data_mode swaps the whole data pipeline in one assignment."""

    def test_gb_no_fg_modes(self):
        from lisatools.globalfit.preprocessing import L1ProcessingStep
        from lisatools.globalfit.stock.erebor.injections import (
            SyntheticGBProcessingStep,
        )

        gs = erebor.get_stock("gb_no_fg").make_general_settings()
        self.assertIs(gs.data_processor_class, L1ProcessingStep)

        fit = erebor.get_stock("gb_no_fg")
        fit.general.data_mode = "synthetic"
        gs = fit.make_general_settings()
        self.assertIs(gs.data_processor_class, SyntheticGBProcessingStep)
        self.assertEqual(gs.processor_init_kwargs["injection_params"].shape[1], 9)
        # synthetic streams are exactly Nf*Nt samples -> no preprocess trims
        self.assertIsNone(gs.preprocess_kwargs["trim_kwargs"])

        with self.assertRaises(ValueError):
            erebor.get_stock("gb_no_fg", data_mode="nope").make_general_settings()

    def test_gb_no_fg_env_switch(self):
        from lisatools.globalfit.stock.erebor.injections import (
            SyntheticGBProcessingStep,
        )

        with _EnvGuard(DATA_MODE="synthetic"):
            gs = erebor.get_stock("gb_no_fg").make_general_settings()
        self.assertIs(gs.data_processor_class, SyntheticGBProcessingStep)

    def test_gb_no_fg_explicit_class_wins(self):
        from lisatools.globalfit.stock.erebor.injections import (
            SyntheticGBProcessingStep,
        )

        fit = erebor.get_stock("gb_no_fg")  # data_mode stays "mojito"
        fit.general.data_processor_class = SyntheticGBProcessingStep
        fit.general.processor_init_kwargs = dict(
            Tobs=fit.wdm_grid[3], dt=fit.general.dt, t_start=0.0
        )
        gs = fit.make_general_settings()
        self.assertIs(gs.data_processor_class, SyntheticGBProcessingStep)

    def test_all_sources_modes(self):
        from lisatools.globalfit.preprocessing import SangriaProcessingStep
        from lisatools.globalfit.stock.erebor.injections import (
            L1ProcessingStepWithSyntheticNoise,
            SyntheticCombinedProcessingStep,
        )

        gs = erebor.get_stock("all_sources").make_general_settings()
        self.assertIs(gs.data_processor_class, L1ProcessingStepWithSyntheticNoise)
        self.assertEqual(
            sorted(gs.processor_init_kwargs["source_ids"]),
            ["EMRI", "GB", "MBHB", "SOBHB", "VGB"],
        )

        gs = erebor.get_stock("all_sources", data_mode="sangria").make_general_settings()
        self.assertIs(gs.data_processor_class, SangriaProcessingStep)

        fit = erebor.get_stock("all_sources", data_mode="synthetic")
        fit.remove_branch("sobbh")
        fit.pop_move("sobbh_pe")
        gs = fit.make_general_settings()
        self.assertIs(gs.data_processor_class, SyntheticCombinedProcessingStep)
        specs = dict(
            (cls.__name__, kwargs)
            for cls, kwargs in gs.processor_init_kwargs["processor_specs"]
        )
        # GB via the gb_no_fg-style generator; MBH/EMRI/SOBBH (+ noise) via
        # full_year's SyntheticDataProcessor so the data injection matches
        # the branch templates. The removed sobbh branch injects nothing.
        self.assertIn("SyntheticGBProcessingStep", specs)
        self.assertIn("SyntheticDataProcessor", specs)
        src = specs["SyntheticDataProcessor"]
        self.assertEqual(
            np.atleast_2d(src["sobbh_injection_params_full_basis"]).shape[0], 0
        )
        self.assertEqual(
            np.atleast_2d(src["emri_injection_params_full_basis"]).shape[0], 1
        )
        self.assertEqual(
            np.atleast_2d(src["mbh_injection_params_sampling_basis"]).shape[0], 1
        )

    def test_full_year_modes(self):
        from lisatools.globalfit.stock.erebor.injections import (
            L1ProcessingStepWithSyntheticNoise,
            SyntheticDataProcessor,
        )

        with _EnvGuard(EMRI_IDS="1", MBHB_IDS=None, SOBHB_IDS=None):
            gs = erebor.get_stock("full_year_combined").make_general_settings()
            self.assertIs(gs.data_processor_class, L1ProcessingStepWithSyntheticNoise)
            gs = erebor.get_stock(
                "full_year_combined", data_mode="synthetic"
            ).make_general_settings()
            self.assertIs(gs.data_processor_class, SyntheticDataProcessor)


class LiteVariantTest(unittest.TestCase):
    LITE_NAMES = (
        "gb_no_fg_lite",
        "all_sources_lite",
        "full_year_combined_lite",
        "noise_only_lite",
        "noise_sgwb_lite",
    )

    def test_lite_twins_registered(self):
        names = [name for name, _ in erebor.get_stock_options()]
        for lite in self.LITE_NAMES:
            self.assertIn(lite, names)
            self.assertIn(lite[: -len("_lite")], names)

    def test_lite_preset_applied(self):
        fit = erebor.all_sources_lite()
        self.assertEqual(fit.general.num_iterations, 3)
        self.assertEqual(fit.general.nwalkers, 4)
        self.assertEqual(fit.general.ntemps, 2)
        self.assertIs(fit.general.use_gpu, False)
        self.assertEqual(fit.general.nt, 180)
        self.assertEqual(fit.gb.num_repeat_proposals, 2)

    def test_lite_kwarg_matches_twin(self):
        via_kwarg = erebor.all_sources(lite=True)
        via_twin = erebor.all_sources_lite()
        for path in ("num_iterations", "nwalkers", "ntemps", "nf", "nt", "use_gpu"):
            self.assertEqual(
                getattr(via_kwarg.general, path), getattr(via_twin.general, path)
            )
        self.assertEqual(
            via_kwarg.gb.num_repeat_proposals, via_twin.gb.num_repeat_proposals
        )

    def test_explicit_kwarg_beats_lite(self):
        fit = erebor.all_sources_lite(nwalkers=12)
        self.assertEqual(fit.general.nwalkers, 12)
        self.assertEqual(fit.general.num_iterations, 3)

    def test_lite_stays_cheap_and_picklable(self):
        for lite in self.LITE_NAMES:
            fit = erebor.get_stock(lite)
            self.assertFalse(fit.built)
            clone = pickle.loads(pickle.dumps(copy.deepcopy(fit)))
            self.assertEqual(
                clone.general.num_iterations, fit.general.num_iterations
            )

    def test_lite_to_heavy_is_knob_changes(self):
        fit = erebor.gb_no_fg_lite()
        fit.general.tobs_target = 90 * 86400.0
        fit.gb.num_repeat_proposals = 100
        heavy = erebor.gb_no_fg()
        self.assertEqual(fit.general.tobs_target, heavy.general.tobs_target)
        self.assertEqual(
            fit.gb.num_repeat_proposals, heavy.gb.num_repeat_proposals
        )

    # -- env vars overrule the lite preset (precedence: kwarg > env > lite) ---
    def test_env_overrules_lite(self):
        with _EnvGuard(NWALKERS="16", NUM_ITERATIONS="9", NTEMPS=None, USE_GPU=None):
            fit = erebor.get_stock("gb_no_fg_lite")
            self.assertEqual(fit.general.nwalkers, 16)       # env beats lite 4
            self.assertEqual(fit.general.num_iterations, 9)  # env beats lite 3
            self.assertEqual(fit.general.ntemps, 2)          # no env -> lite 2

    def test_use_gpu_env_overrules_lite(self):
        # USE_GPU is folded into the general env>lite mechanism (no longer a
        # bespoke special case): the env var overrules the lite CPU default.
        with _EnvGuard(USE_GPU="1", NWALKERS=None, NTEMPS=None, NUM_ITERATIONS=None):
            self.assertIs(erebor.get_stock("gb_no_fg_lite").general.use_gpu, True)
        with _EnvGuard(USE_GPU=None):
            self.assertIs(erebor.get_stock("gb_no_fg_lite").general.use_gpu, False)

    def test_kwarg_beats_env_beats_lite(self):
        with _EnvGuard(NWALKERS="16"):
            fit = erebor.get_stock("gb_no_fg_lite", nwalkers=8)
            self.assertEqual(fit.general.nwalkers, 8)  # explicit kwarg wins

    def test_gb_branch_env_overrules_lite(self):
        with _EnvGuard(GB_NUM_REPEAT_PROPOSALS="50"):
            fit = erebor.get_stock("gb_no_fg_lite")
            self.assertEqual(fit.gb.num_repeat_proposals, 50)  # env beats lite 2


class SyntheticFallbackTest(unittest.TestCase):
    def test_missing_mojito_falls_back_to_synthetic_prior(self):
        fit = erebor.all_sources_lite(mojito_data_path="/definitely/not/there/")
        fit.resolve_data_source()
        self.assertEqual(fit.general.data_mode, "synthetic")
        self.assertEqual(fit.general.synthetic_injections, "prior")
        self.assertIsNotNone(fit.general.gb_injection_params)
        gb_rows = np.atleast_2d(fit.general.gb_injection_params)
        self.assertEqual(gb_rows.shape[1], 9)
        lo, hi = fit._gb_injection_band()
        self.assertTrue(((gb_rows[:, 1] >= lo) & (gb_rows[:, 1] <= hi)).all())

    def test_fallback_processor_is_synthetic(self):
        fit = erebor.all_sources_lite(mojito_data_path="/definitely/not/there/")
        gs = fit.make_general_settings()
        from lisatools.globalfit.stock.erebor.injections import (
            SyntheticCombinedProcessingStep,
        )

        self.assertIs(gs.data_processor_class, SyntheticCombinedProcessingStep)

    def test_explicit_synthetic_keeps_stock_tables(self):
        fit = erebor.all_sources_lite(data_mode="synthetic")
        fit.resolve_data_source()
        self.assertEqual(fit.general.synthetic_injections, "stock")
        self.assertIsNone(fit.general.gb_injection_params)

    def test_user_gb_table_wins_over_prior_mode(self):
        table = np.array([[1e-22, 3e-3, 1e-16, 0.0, 0.0, 0.5, 0.4, 1.0, 0.2]])
        fit = erebor.all_sources_lite(
            mojito_data_path="/definitely/not/there/", gb_injection_params=table
        )
        fit.resolve_data_source()
        self.assertTrue(np.array_equal(fit.general.gb_injection_params, table))

    def test_existing_mojito_path_untouched(self):
        with _EnvGuard(MOJITO_DATA_PATH=None):
            fit = erebor.all_sources_lite(mojito_data_path=os.getcwd())
            fit.resolve_data_source()
            self.assertEqual(fit.general.data_mode, "mojito")


class PriorInjectionDrawTest(unittest.TestCase):
    def test_reproducible_and_seed_sensitive(self):
        from lisatools.globalfit.stock.erebor.injections import (
            make_emri_injections,
            make_gb_injections,
            make_mbh_injections,
            make_sobbh_injections,
        )

        for maker, args in (
            (make_emri_injections, (3,)),
            (make_sobbh_injections, (3,)),
            (make_mbh_injections, (3, 1.0e6)),
        ):
            a = maker(*args, mode="prior", seed=5)
            self.assertTrue(np.array_equal(a, maker(*args, mode="prior", seed=5)))
            self.assertFalse(np.array_equal(a, maker(*args, mode="prior", seed=6)))
        g = make_gb_injections(3, mode="prior", seed=5)
        self.assertTrue(np.array_equal(g, make_gb_injections(3, mode="prior", seed=5)))

    def test_stock_mode_matches_legacy(self):
        from lisatools.globalfit.stock.erebor.injections import (
            GB_INJECTION_PARAMS,
            make_emri_injections,
            make_gb_injections,
            make_mbh_injections,
            make_sobbh_injections,
        )

        # stock mode ignores the seed and reproduces the historical tables
        self.assertTrue(
            np.array_equal(
                make_emri_injections(2), make_emri_injections(2, seed=99)
            )
        )
        self.assertTrue(
            np.array_equal(
                make_sobbh_injections(2), make_sobbh_injections(2, seed=99)
            )
        )
        self.assertTrue(
            np.array_equal(
                make_mbh_injections(2, 1e6), make_mbh_injections(2, 1e6, seed=99)
            )
        )
        self.assertTrue(
            np.array_equal(make_gb_injections(2), GB_INJECTION_PARAMS)
        )

    def test_prior_draws_respect_interior_ranges(self):
        from lisatools.globalfit.stock.erebor.injections import (
            make_emri_injections,
            make_mbh_injections,
            make_sobbh_injections,
        )

        e = make_emri_injections(8, mode="prior", seed=3)
        self.assertTrue(((e[:, 3] >= 9.0) & (e[:, 3] <= 13.0)).all())  # p0
        self.assertTrue(((e[:, 4] >= 0.1) & (e[:, 4] <= 0.5)).all())  # e0
        s = make_sobbh_injections(8, mode="prior", seed=3)
        self.assertTrue((s[:, 1] <= s[:, 0]).all())  # m2 <= m1
        self.assertTrue(((s[:, 6] >= 6e-3) & (s[:, 6] <= 1.8e-2)).all())  # f_low
        m = make_mbh_injections(8, 1.0e6, mode="prior", seed=3)
        self.assertTrue(((m[:, 10] >= 0.25e6) & (m[:, 10] <= 0.85e6)).all())

    def test_branch_prep_and_processor_agree(self):
        # The same (mode, seed) flows to the data processor and the branch
        # prep — spot-check via the module-level helpers they both call.
        from lisatools.globalfit.stock.erebor.injections import make_mbh_injections
        from lisatools.globalfit.stock.erebor.source_runtime import (
            synthetic_injection_mode,
        )

        fit = erebor.all_sources_lite(
            data_mode="synthetic", synthetic_injections="prior",
            synthetic_injection_seed=77,
        )
        fit.resolve_data_source()
        mode, seed = synthetic_injection_mode(fit.general)
        self.assertEqual((mode, seed), ("prior", 77))
        a = make_mbh_injections(1, 1e6, mode=mode, seed=seed)
        b = make_mbh_injections(1, 1e6, mode="prior", seed=77)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()


class GFHDFBackendNameTest(unittest.TestCase):
    """The run's HDF group name is a knob, and everything must agree on it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def test_default_is_global_fit(self):
        from lisatools.globalfit.hdfbackend import GFHDFBackend

        self.assertEqual(GFHDFBackend(self._path("a.h5")).name, "global_fit")

    def test_explicit_name_honored(self):
        from lisatools.globalfit.hdfbackend import GFHDFBackend

        self.assertEqual(GFHDFBackend(self._path("b.h5"), name="zzz").name, "zzz")

    def test_legacy_mcmc_file_is_adopted(self):
        """A file written before the rename stores its run under "mcmc".

        It must be detected and honored -- otherwise the run reads as
        uninitialized and gets sampled over the top of real data.
        """
        import h5py

        from lisatools.globalfit.hdfbackend import GFHDFBackend

        p = self._path("legacy.h5")
        with h5py.File(p, "w") as f:
            f.create_group("mcmc")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend = GFHDFBackend(p)
        self.assertEqual(backend.name, "mcmc")
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_new_style_file_not_hijacked(self):
        import h5py

        from lisatools.globalfit.hdfbackend import GFHDFBackend

        p = self._path("new.h5")
        with h5py.File(p, "w") as f:
            f.create_group("global_fit")
        self.assertEqual(GFHDFBackend(p).name, "global_fit")

    def test_sub_backends_share_the_parent_name(self):
        """Sub-backends write into f[name]["sub_backend"] of the PARENT's file.

        They subclass eryn's HDFBackend directly, so left to their own default
        they look for "mcmc" while the parent writes "global_fit" -- reset()
        then dies with KeyError: object 'mcmc' doesn't exist.
        """
        from lisatools.globalfit.hdfbackend import (
            GFHDFBackend,
            MBHHDFBackend,
            SOBBHHDFBackend,
        )

        backend = GFHDFBackend(
            self._path("subs.h5"),
            sub_backend={"mbh": MBHHDFBackend, "sobbh": SOBBHHDFBackend},
        )
        for name, sub in backend.sub_backend.items():
            self.assertEqual(sub.name, backend.name, f"{name} sub-backend name drifted")
