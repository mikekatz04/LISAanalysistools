"""Tests for the stock global-fit API (lisatools.globalfit.stock).

Everything here runs without GPU / mojito / Sangria data: it exercises the
cheap configured-but-unbuilt layer (registry, knobs, recipe/move editing,
branch composition, pickle/deepcopy safety, env-default resolution) — the
heavy build paths are covered by the parity/smoke harnesses.
"""

import copy
import os
import pickle
import unittest

import numpy as np

from lisatools.globalfit.stock import MoveSpec, RecipeSpec, StageSpec
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
                       CHUNKED_NT_SUB=None, GF_NUM_ITER=None):
            clone = erebor.gb_no_fg(debug=True)
            self.assertTrue(clone.debug)
            self.assertEqual(clone.tobs_target, 3 * 86400.0)
            self.assertEqual(clone.nwalkers, 3)
            self.assertEqual(clone.gb.nt_sub, 64)

    def test_gb_debug_preset(self):
        with _EnvGuard(GB_DEBUG="1", TOBS_TARGET=None, NWALKERS=None, NTEMPS=None,
                       CHUNKED_NT_SUB=None, CHUNKED_N_PAD=None, CHUNKED_N_SPARSE=None,
                       CHUNKED_N_CP_SIG=None, CHUNKED_N_CP_ORBIT=None, GF_NUM_ITER=None):
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
        fit.recipe.add_move(MoveSpec("rj_fstat_mcmc", branch="gb"), stage="gb_pe")
        clone = self._roundtrip(fit)
        self.assertEqual(clone.gb.a_lims, [1e-25, 1e-20])


class RecipeSpecTest(unittest.TestCase):
    def _recipe(self):
        return RecipeSpec(
            [
                StageSpec("s1", kind="search", moves=[MoveSpec("a"), MoveSpec("b")]),
                StageSpec("s2", kind="pe", moves=[MoveSpec("c")]),
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
        r.add_move(MoveSpec("d"), before="b")
        self.assertEqual([m.name for m in r._stage("s1").moves], ["a", "d", "b"])
        r.add_move(MoveSpec("e"), stage="s2", index=0)
        self.assertEqual([m.name for m in r._stage("s2").moves], ["e", "c"])
        r.add_move(MoveSpec("f"), after="c")
        self.assertEqual([m.name for m in r._stage("s2").moves], ["e", "c", "f"])

    def test_duplicate_name_rejected(self):
        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(MoveSpec("a"), stage="s2")

    def test_ambiguous_stage_required(self):
        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(MoveSpec("g"))  # two stages, no placement

    def test_conflicting_placements_rejected(self):
        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(MoveSpec("g"), before="a", index=0)

    def test_stage_composition(self):
        r = self._recipe()
        r.add_stage(StageSpec("s0", kind="pe", moves=[MoveSpec("z")]), before="s1")
        self.assertEqual([s.name for s in r.stages], ["s0", "s1", "s2"])
        popped = r.pop_stage("s1")
        self.assertEqual(popped.name, "s1")
        self.assertEqual([s.name for s in r.stages], ["s0", "s2"])

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            StageSpec("bad", kind="wiggle")

    def test_instance_move_needs_name(self):
        r = self._recipe()
        with self.assertRaises(ValueError):
            r.add_move(object(), stage="s2")


class BranchCompositionTest(unittest.TestCase):
    def test_add_remove_swap(self):
        fit = erebor.get_stock("all_sources")
        self.assertEqual(
            list(fit.branches), ["gb", "psd", "galfor", "mbh", "emri", "sobbh"]
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

        with _EnvGuard(DATA_PROCESSOR="synthetic"):
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
            ["EMRI", "GB", "MBHB", "SOBHB"],
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


if __name__ == "__main__":
    unittest.main()
