"""The non-GB sources start at, and can be pushed back to, the catalogue injection.

Mike 2026-10-22: "We want those [VGB, MBHB, SOBHB, EMRI] to always start at the
catalog injected values rather than the last run's values. Make that update and
make a script that pushes the ongoing runs (when stopped) back to the injection
values."

Pinned here:

* ``lisatools.globalfit.source_reset``: an armed branch's leaves go to the EXACT
  injection on every walker and every rung of the main state and of the
  branch sub-state, inds on; unarmed branches and GB / noise are untouched;
  refusals for a missing injection, a shape mismatch and an absent branch; a
  fresh start is reported, never reset;
* a ROUND TRIP through a real store: write rows, resume the last one the way a
  run does, reset, save, and read the injection back from ``chain/<b>`` and
  ``sub_backend/<b>/chain`` at every rung;
* the four settings fields ``reset_to_injection_on_resume`` read their
  ``{BRANCH}_RESET_TO_INJECTION_ON_RESUME`` knob;
* ``GlobalFit.load_info`` calls the reset, gated on a loaded state;
* every launcher defaults ``SOURCE_WARM_START_STORE`` to empty and bakes in no
  one-launch reset knob;
* ``scripts/flagship/relaunch_sources_at_injection.sh`` arms exactly the
  chosen knobs and passes the rest of the environment through.
"""
import dataclasses
import glob
import inspect
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
WRAPPER = os.path.join(ROOT, "scripts", "flagship", "relaunch_sources_at_injection.sh")

NW, NT = 4, 3
NUM_BANDS = 3
BAND_EDGES = np.linspace(1e-3, 4e-3, NUM_BANDS + 1)
SHAPES = {"vgb": (5, 6), "mbh": (4, 11), "emri": (3, 12), "sobbh": (2, 11),
          "gb": (6, 8), "psd": (1, 2)}


def _bases():
    from lisatools.globalfit.state import (
        EMRIState, GBState, MBHState, ModuleSubState, SOBBHState)

    return {"vgb": GBState, "mbh": MBHState, "emri": EMRIState, "sobbh": SOBBHState,
            "gb": GBState, "psd": ModuleSubState}


def make_state(rng):
    """A resumed-looking state: distinct values everywhere, full ladders."""
    from lisatools.globalfit.state import GFState

    coords = {b: rng.standard_normal((NT, NW, nl, nd)) for b, (nl, nd) in SHAPES.items()}
    inds = {b: np.ones((NT, NW, nl), dtype=bool) for b, (nl, _) in SHAPES.items()}
    inds["gb"][:, :, -1] = False
    state = GFState(
        {b: c[:1].copy() for b, c in coords.items()},
        inds={b: i[:1].copy() for b, i in inds.items()},
        log_like=rng.standard_normal((1, NW)),
        log_prior=rng.standard_normal((1, NW)),
        betas=np.ones(1),
        random_state=np.random.get_state(),
        sub_state_bases=_bases(),
    )
    for b in ("gb", "vgb"):
        state.sub_states[b].initialize_band_information(
            NW, NT, BAND_EDGES, np.tile(np.linspace(1.0, 0.3, NT), (NUM_BANDS, 1)))
    for b, sub in state.sub_states.items():
        nl, nd = SHAPES[b]
        if b in ("mbh", "emri", "sobbh"):
            sub.betas_all = np.tile(1.0 / 1.2 ** np.arange(NT), (nl, 1))
        sub.initialize_tempered(NT, NW, nl, nd, coords=coords[b], inds=inds[b])
    return state


def make_info(rng, armed, *, flat=()):
    """source_info: an injection per source branch, the reset armed on ``armed``."""
    info = {}
    for b in ("vgb", "mbh", "emri", "sobbh"):
        nl, nd = SHAPES[b]
        inj = rng.standard_normal(nd) if b in flat else rng.standard_normal((nl, nd))
        info[b] = SimpleNamespace(injection=inj, reset_to_injection_on_resume=b in armed)
    info["gb"] = SimpleNamespace(injection=None)
    return info


def _snapshot(state):
    out = {}
    for b in SHAPES:
        out[("main", b)] = (np.array(state.branches[b].coords, copy=True),
                            np.array(state.branches[b].inds, copy=True))
        sub = state.sub_states[b]
        out[("sub", b)] = (np.array(sub.coords, copy=True), np.array(sub.inds, copy=True))
    return out


class ResetTest(unittest.TestCase):

    def test_armed_branches_go_to_the_exact_injection_everywhere(self):
        from lisatools.globalfit.source_reset import reset_sources_to_injection

        rng = np.random.default_rng(1)
        state = make_state(rng)
        armed = ("vgb", "mbh", "emri")
        info = make_info(rng, armed)
        before = _snapshot(state)
        out = reset_sources_to_injection(state, info)
        self.assertEqual(sorted(out), sorted(armed))
        for b in armed:
            inj = np.asarray(info[b].injection)
            main, sub = state.branches[b], state.sub_states[b]
            np.testing.assert_array_equal(main.coords, np.broadcast_to(inj, main.coords.shape))
            np.testing.assert_array_equal(sub.coords, np.broadcast_to(inj, sub.coords.shape))
            self.assertEqual(sub.coords.shape[0], NT)            # every rung of the ladder
            self.assertTrue(main.inds.all() and sub.inds.all())
            # the cold rows of main and sub agree (the moves' consistency check)
            np.testing.assert_array_equal(main.coords[0], sub.coords[0])
            # the reported move = the cold chain's largest |change| per leaf
            want = np.abs(before[("main", b)][0][0] - inj[None]).max(axis=(0, 2))
            np.testing.assert_allclose(out[b]["max_abs_move_per_leaf"], want)
            self.assertEqual((out[b]["ntemps_sub"], out[b]["nwalkers"]), (NT, NW))
        for b in ("sobbh", "gb", "psd"):                    # not armed / not a source
            for layer in ("main", "sub"):
                np.testing.assert_array_equal(
                    before[(layer, b)][0], (state.branches[b] if layer == "main"
                                            else state.sub_states[b]).coords)

    def test_one_row_is_broadcast_only_to_a_single_leaf_branch(self):
        """A (1, ndim) or (ndim,) injection over several stored leaves would put
        the same source into every leaf: refused; fine for one leaf."""
        from lisatools.globalfit.source_reset import injection_rows

        nd = SHAPES["mbh"][1]
        row = np.arange(1.0, nd + 1.0)
        for inj in (row, row[None, :]):
            with self.subTest(shape=inj.shape):
                got = injection_rows(SimpleNamespace(injection=inj), "mbh", 1, nd)
                np.testing.assert_array_equal(got, row[None, :])
                with self.assertRaisesRegex(ValueError, "different id list"):
                    injection_rows(SimpleNamespace(injection=inj), "mbh", 4, nd)

    def test_the_per_leaf_snr_records_are_cleared(self):
        from lisatools.globalfit.source_reset import reset_sources_to_injection

        rng = np.random.default_rng(12)
        state = make_state(rng)
        for b in SHAPES:
            state.sub_states[b].d_h[:] = 3.0
            state.sub_states[b].h_h[:] = 9.0
        reset_sources_to_injection(state, make_info(rng, ("mbh", "vgb")))
        for b in ("mbh", "vgb"):
            self.assertTrue(np.isnan(state.sub_states[b].d_h).all(), b)
            self.assertTrue(np.isnan(state.sub_states[b].h_h).all(), b)
        for b in ("emri", "sobbh", "gb"):
            self.assertTrue((state.sub_states[b].h_h == 9.0).all(), b)

    def test_refusals(self):
        from lisatools.globalfit.source_reset import reset_sources_to_injection

        rng = np.random.default_rng(3)
        info = make_info(rng, ("emri",))
        info["emri"].injection = None
        with self.assertRaisesRegex(ValueError, "no catalogue injection"):
            reset_sources_to_injection(make_state(rng), info)
        info = make_info(rng, ("emri",))
        info["emri"].injection = rng.standard_normal((SHAPES["emri"][0] + 1, SHAPES["emri"][1]))
        with self.assertRaisesRegex(ValueError, "different id list"):
            reset_sources_to_injection(make_state(rng), info)
        info = make_info(rng, ("sobbh",))
        state = make_state(rng)
        del state.branches["sobbh"]
        with self.assertRaisesRegex(ValueError, "carries no 'sobbh' branch"):
            reset_sources_to_injection(state, info)

    def test_nothing_armed_is_a_no_op(self):
        from lisatools.globalfit.source_reset import reset_sources_to_injection

        rng = np.random.default_rng(4)
        state = make_state(rng)
        before = _snapshot(state)
        self.assertEqual(reset_sources_to_injection(state, make_info(rng, ())), {})
        for k, (c, i) in before.items():
            obj = state.branches[k[1]] if k[0] == "main" else state.sub_states[k[1]]
            np.testing.assert_array_equal(c, obj.coords)

    def test_the_log_line(self):
        from lisatools.globalfit.source_reset import reset_sources_to_injection

        rng = np.random.default_rng(5)
        log = logging.getLogger("test_source_reset")
        with self.assertLogs(log, level="INFO") as cm:
            reset_sources_to_injection(make_state(rng), make_info(rng, ("mbh",)), log)
        line = "\n".join(cm.output)
        self.assertIn("[SOURCE-RESET] mbh: 4 leaves x 4 walkers set to the catalogue injection", line)
        self.assertIn("all 3 rungs of the sub-state ladder", line)
        self.assertIn("MBH_RESET_TO_INJECTION_ON_RESUME=1", line)

    def test_a_fresh_start_is_reported_never_reset(self):
        from lisatools.globalfit.source_reset import maybe_reset_on_resume

        rng = np.random.default_rng(6)
        state = make_state(rng)
        before = _snapshot(state)
        log = logging.getLogger("test_source_reset_fresh")
        with self.assertLogs(log, level="INFO") as cm:
            out = maybe_reset_on_resume(state, make_info(rng, ("mbh", "vgb")),
                                        fresh_start=True, logger=log)
        self.assertEqual(out, {})
        self.assertIn("fresh start", "\n".join(cm.output))
        for k, (c, _) in before.items():
            obj = state.branches[k[1]] if k[0] == "main" else state.sub_states[k[1]]
            np.testing.assert_array_equal(c, obj.coords)
        # and a loaded state IS reset
        out = maybe_reset_on_resume(state, make_info(rng, ("mbh",)), fresh_start=False)
        self.assertEqual(list(out), ["mbh"])


class OneLaunchTest(unittest.TestCase):
    """The marker next to the store makes the knob requeue-safe."""

    def _go(self, store, job, it, armed=("mbh",), seed=0, restart=None, ckpt=None):
        from lisatools.globalfit.source_reset import maybe_reset_on_resume

        rng = np.random.default_rng(100 + seed)
        state = make_state(rng)
        info = make_info(rng, armed)
        out = maybe_reset_on_resume(state, info, fresh_start=False, store_path=store,
                                    stored_iteration=it, job_id=job, restart_count=restart,
                                    ckpt_meta=ckpt, environ={})
        applied = all(np.array_equal(state.sub_states[b].coords,
                                     np.broadcast_to(info[b].injection, state.sub_states[b].coords.shape))
                      for b in armed)
        self.assertEqual(bool(out), applied)
        return applied

    def _marker(self, store):
        import json

        from lisatools.globalfit.source_reset import reset_marker_path

        with open(reset_marker_path(store)) as fp:
            return json.load(fp)

    def test_hdf_path(self):
        from lisatools.globalfit.source_reset import reset_marker_path

        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "run_testing.h5")
            self.assertTrue(self._go(store, "777", 10))
            self.assertEqual(reset_marker_path(store), os.path.join(tmp, "run_testing_source_reset.json"))
            m = self._marker(store)
            self.assertEqual((m["job_id"], m["stored_iteration"], m["branches"], m["restart_count"]),
                             ("777", 10, ["mbh"], None))
            self.assertIsInstance(m["unix_time"], float)
            # the same job requeued AFTER it saved rows: skip
            self.assertFalse(self._go(store, "777", 13, seed=1, restart="1"))
            # the same job requeued before its first save: apply again
            self.assertTrue(self._go(store, "777", 10, seed=2, restart="1"))
            # a NEW job (a deliberate second relaunch): apply
            self.assertTrue(self._go(store, "778", 13, seed=3))
            # outside SLURM there is no job identity: every launch applies
            self.assertTrue(self._go(store, None, 20, seed=4))

    def test_the_same_allocation_launched_again_applies(self):
        """Equal restart counts = not a requeue (an interactive allocation)."""
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "run_testing.h5")
            self.assertTrue(self._go(store, "500", 10, restart="0"))
            self.assertTrue(self._go(store, "500", 14, seed=1, restart="0"))
            self.assertFalse(self._go(store, "500", 18, seed=2, restart="1"))   # a requeue

    def test_mid_iteration_checkpoint_path(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "run_testing.h5")
            self.assertTrue(self._go(store, "600", 10))
            t_reset = self._marker(store)["unix_time"]
            # requeued before the first save, state from a checkpoint written AFTER the reset
            self.assertFalse(self._go(store, "600", 10, seed=1, restart="1",
                                      ckpt=dict(stored_iteration=10, unix_time=t_reset + 60.0)))
            # a checkpoint of a LATER row (the saver lost a row): skip
            self.assertFalse(self._go(store, "600", 10, seed=2, restart="1",
                                      ckpt=dict(stored_iteration=11, unix_time=t_reset - 60.0)))
            # a checkpoint older than the reset, same row: apply
            self.assertTrue(self._go(store, "600", 10, seed=3, restart="1",
                                     ckpt=dict(stored_iteration=10, unix_time=t_reset - 60.0)))
            self.assertLess(time.time() - self._marker(store)["unix_time"], 300)

    def test_a_malformed_marker_means_no_marker(self):
        from lisatools.globalfit.source_reset import reset_marker_path

        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "run_testing.h5")
            for body in ('{"job_id": "1", "stored_iteration": null}',
                         '{"job_id": "1", "stored_iteration": "x", "unix_time": "y"}',
                         "[1, 2]", "not json"):
                with open(reset_marker_path(store), "w") as fp:
                    fp.write(body)
                with self.subTest(body=body):
                    self.assertTrue(self._go(store, "1", 50, restart="3"))

    def test_the_eigen_tables_of_the_reset_branches_are_dropped(self):
        import pickle

        from lisatools.globalfit.moves import eigen_table_persist as E

        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "run_testing.h5")
            side = E.sidecar_path(store)
            entries = {"mbh:0": {"axes": 1}, "mbh:3": {"axes": 2},
                       "emri:1": {"axes": 3}, "gb:0": {"axes": 4}}
            with open(side, "wb") as fp:
                pickle.dump({"format": E.FORMAT_VERSION, "entries": dict(entries)}, fp)
            self._go(store, "900", 5, armed=("mbh",))
            left = E.load_sidecar(side)                     # the module still reads it
            self.assertEqual(left["format"], E.FORMAT_VERSION)
            self.assertEqual(sorted(left["entries"]), ["emri:1", "gb:0"])
            backup = side + ".pre_source_reset_900"
            self.assertEqual(sorted(E.load_sidecar(backup)["entries"]), sorted(entries))
            # a re-application in the same job never overwrites the true backup
            payload = E.load_sidecar(side)
            payload["entries"]["mbh:7"] = {"axes": 9}     # a post-reset table
            with open(side, "wb") as fp:
                pickle.dump(payload, fp)
            self._go(store, "900", 5, armed=("mbh",), seed=1, restart="1")
            self.assertEqual(sorted(E.load_sidecar(backup)["entries"]), sorted(entries))
            self.assertEqual(sorted(E.load_sidecar(side)["entries"]), ["emri:1", "gb:0"])

    def test_a_knob_for_an_unconfigured_branch_warns(self):
        from lisatools.globalfit.source_reset import maybe_reset_on_resume

        rng = np.random.default_rng(14)
        info = make_info(rng, ())
        del info["vgb"]
        log = logging.getLogger("test_source_reset_unconf")
        with self.assertLogs(log, level="WARNING") as cm:
            out = maybe_reset_on_resume(make_state(rng), info, fresh_start=False, logger=log,
                                        environ={"VGB_RESET_TO_INJECTION_ON_RESUME": "1"})
        self.assertEqual(out, {})
        self.assertIn("configures no 'vgb' branch", "\n".join(cm.output))


class StoreRoundTripTest(unittest.TestCase):
    """Write a real store, resume its last row, reset, save, read back."""

    def test_the_reset_state_is_what_the_store_saves(self):
        import h5py

        sys.path.insert(0, HERE)
        import test_copy_store_leaf as H
        from lisatools.globalfit.source_reset import reset_sources_to_injection

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.h5")
            H.make_store(path)
            backend = H.make_backend(path)
            state = backend.get_last_sample()          # what load_info resumes from
            rng = np.random.default_rng(7)
            info = {}
            for b in ("mbh", "emri", "sobbh"):
                nl, nd = H.BRANCH_SHAPES[b]
                info[b] = SimpleNamespace(injection=rng.standard_normal((nl, nd)),
                                          reset_to_injection_on_resume=b != "sobbh")
            reset_sources_to_injection(state, info)
            backend.save_step(state, np.ones((1, H.NW)))
            with h5py.File(path, "r") as f:
                g = f["global_fit"]
                row = int(g.attrs["iteration"]) - 1
                self.assertEqual(row, H.SAVED)                     # a NEW row was written
                for b in ("mbh", "emri"):
                    inj = info[b].injection
                    np.testing.assert_array_equal(
                        g[f"chain/{b}"][row, 0], np.broadcast_to(inj, g[f"chain/{b}"][row, 0].shape))
                    sub = g[f"sub_backend/{b}/chain"][row]       # (ntemps, nwalkers, nleaves, ndim)
                    self.assertEqual(sub.shape[0], H.NT)
                    np.testing.assert_array_equal(sub, np.broadcast_to(inj, sub.shape))
                    self.assertTrue(g[f"sub_backend/{b}/inds"][row].all())
                # the unarmed branch and GB carry the resumed row forward unchanged
                np.testing.assert_array_equal(g["chain/sobbh"][row], g["chain/sobbh"][row - 1])
                np.testing.assert_array_equal(g["sub_backend/sobbh/chain"][row],
                                              g["sub_backend/sobbh/chain"][row - 1])
                np.testing.assert_array_equal(g["chain/gb"][row], g["chain/gb"][row - 1])
                np.testing.assert_array_equal(g["inds/gb"][row], g["inds/gb"][row - 1])


class SettingsFieldTest(unittest.TestCase):

    def test_each_branch_reads_its_knob(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings
        from lisatools.globalfit.stock.erebor.vgb import VGBSettings

        for cls, knob in ((MBHSettings, "MBH_RESET_TO_INJECTION_ON_RESUME"),
                          (SOBBHSettings, "SOBBH_RESET_TO_INJECTION_ON_RESUME"),
                          (EMRISettings, "EMRI_RESET_TO_INJECTION_ON_RESUME"),
                          (VGBSettings, "VGB_RESET_TO_INJECTION_ON_RESUME")):
            field = {f.name: f for f in dataclasses.fields(cls)}["reset_to_injection_on_resume"]
            env = {k: v for k, v in os.environ.items() if k != knob}
            with self.subTest(cls=cls.__name__):
                with mock.patch.dict(os.environ, env, clear=True):
                    self.assertIs(field.default_factory(), False)
                for raw, want in (("1", True), ("0", False), ("true", True)):
                    with mock.patch.dict(os.environ, {knob: raw}):
                        self.assertIs(field.default_factory(), want, raw)

    def test_the_module_names_the_same_knobs(self):
        from lisatools.globalfit.source_reset import RESET_BRANCHES, reset_knob

        self.assertEqual(RESET_BRANCHES, ("vgb", "mbh", "emri", "sobbh"))
        self.assertEqual(reset_knob("sobbh"), "SOBBH_RESET_TO_INJECTION_ON_RESUME")


class LoadInfoWiringTest(unittest.TestCase):

    def test_load_info_resets_a_loaded_state_only(self):
        from lisatools.globalfit.run import GlobalFit

        src = inspect.getsource(GlobalFit.load_info)
        i_flag = src.find("_fresh_start = state is None")
        i_fresh = src.find("if state is None:", i_flag)
        i_call = src.find("maybe_reset_on_resume(")
        i_ret = src.rfind("return state")
        self.assertGreater(i_flag, 0, "the fresh-start flag is gone")
        self.assertGreater(i_fresh, i_flag, "the flag must be taken before the fresh-start branch")
        self.assertTrue(i_fresh < i_call < i_ret, "the reset must run last, before the return")
        call = src[i_call:i_ret]
        self.assertIn("fresh_start=_fresh_start", call)
        self.assertIn("store_path=backend_path", call)
        self.assertIn("stored_iteration=_stored_it", call)
        self.assertIn('os.environ.get("SLURM_JOB_ID")', call)
        self.assertIn('os.environ.get("SLURM_RESTART_COUNT")', call)
        self.assertIn("ckpt_meta=(_ckpt[1] if _ckpt is not None else None)", call)


def _launchers():
    paths = [os.path.join(ROOT, "scripts", "flagship", "launcher.sh.in")]
    paths += sorted(glob.glob(os.path.join(ROOT, "scripts", "flagship", "submit_gf_*mo.sh")))
    paths += [os.path.join(ROOT, "scripts", "fstat_proposal", f) for f in (
        "submit_gf_3mo_v9_2gpu.sh", "submit_gf_6mo_v9_4gpu.sh",
        "submit_gf_9mo_v9.sh", "submit_gf_1yr_v9.sh")]
    return [p for p in paths if os.path.exists(p)]


class LauncherDefaultsTest(unittest.TestCase):

    def test_no_launcher_arms_the_parent_source_warm_start(self):
        paths = _launchers()
        self.assertGreaterEqual(len(paths), 10)
        for p in paths:
            text = open(p).read()
            with self.subTest(launcher=os.path.basename(p)):
                lines = re.findall(r"^export SOURCE_WARM_START_STORE=.*$", text, re.M)
                for line in lines:
                    self.assertIn(line, ("export SOURCE_WARM_START_STORE=${SOURCE_WARM_START_STORE-}",
                                         "export SOURCE_WARM_START_STORE="), line)
                self.assertNotIn("SOURCE_WARM_START_STORE-${GF_SEED_STORE}", text)

    def test_no_launcher_bakes_in_a_one_launch_reset(self):
        for p in _launchers():
            with self.subTest(launcher=os.path.basename(p)):
                self.assertIsNone(re.search(r"^export [A-Z]+_RESET_TO_INJECTION_ON_RESUME=",
                                            open(p).read(), re.M))


class WrapperScriptTest(unittest.TestCase):

    def _fake_launcher(self, tmp):
        path = os.path.join(tmp, "fake_launcher.sh")
        with open(path, "w") as f:
            f.write("#!/usr/bin/env bash\n"
                    "env | grep -E '_RESET_TO_INJECTION_ON_RESUME=|^PASS_THROUGH=' | sort\n")
        os.chmod(path, 0o755)
        return path

    def _run(self, *args, env=None):
        full = {k: v for k, v in os.environ.items() if "_RESET_TO_INJECTION_ON_RESUME" not in k}
        full.update(env or {})
        return subprocess.run([WRAPPER, *args], capture_output=True, text=True, env=full)

    def test_it_is_executable(self):
        self.assertTrue(os.access(WRAPPER, os.X_OK))

    def test_all_four_by_default_and_the_environment_passes_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(self._fake_launcher(tmp), env={"PASS_THROUGH": "yes"})
        self.assertEqual(r.returncode, 0, r.stderr)
        got = set(l for l in r.stdout.splitlines() if "=" in l and not l.startswith("["))
        self.assertEqual(got, {"EMRI_RESET_TO_INJECTION_ON_RESUME=1",
                               "MBH_RESET_TO_INJECTION_ON_RESUME=1",
                               "SOBBH_RESET_TO_INJECTION_ON_RESUME=1",
                               "VGB_RESET_TO_INJECTION_ON_RESUME=1",
                               "PASS_THROUGH=yes"})

    def test_a_chosen_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(self._fake_launcher(tmp), "mbh,emri")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = set(l for l in r.stdout.splitlines() if l.endswith("=1") and not l.startswith("["))
        self.assertEqual(got, {"EMRI_RESET_TO_INJECTION_ON_RESUME=1",
                               "MBH_RESET_TO_INJECTION_ON_RESUME=1"})

    def test_refusals_launch_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = self._fake_launcher(tmp)
            for args in ((fake, "mbh,gb"), (fake, ","), (os.path.join(tmp, "missing.sh"),), ()):
                r = self._run(*args)
                with self.subTest(args=args):
                    self.assertEqual(r.returncode, 2)
                    self.assertNotIn("launching", r.stdout)
                    self.assertNotIn("_RESET_TO_INJECTION_ON_RESUME=1", r.stdout.replace(
                        "one-launch reset armed", ""))

    def test_a_bare_launcher_name_runs_from_the_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._fake_launcher(tmp)
            full = {k: v for k, v in os.environ.items() if "_RESET_TO_INJECTION_ON_RESUME" not in k}
            r = subprocess.run([WRAPPER, "fake_launcher.sh", "sobbh"], capture_output=True,
                               text=True, env=full, cwd=tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("launching ./fake_launcher.sh", r.stdout)
        self.assertIn("SOBBH_RESET_TO_INJECTION_ON_RESUME=1", r.stdout)

if __name__ == "__main__":
    unittest.main()
