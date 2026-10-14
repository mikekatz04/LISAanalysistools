"""GB warm-start SELF refit (user rulings 2026-10-13; ``warmstart/self_refit.py``).

Mike, 2026-10-13: "Starting in gb search 2 (and in full pe). Let's rebuild our
warm start proposal with the last 50 samples (x4 walkers)." "We should keep
the original warm start in the fit." "... sample from the previous full warm
start 25% of the time and the updated refit warm start 75% of the time."
"during gb search 2, the warm start refit should occur every 5 iterations."
"full pe adaptivity: just refit every 250 iterations."

What is pinned here (CPU only, KB-sized fixtures, the real pipeline stubbed
where it would need waveforms):

* the five ``GBSettings`` fields and their env knobs;
* the cadence / clock function (stage-relative, both cadences, resume,
  a rewound store) and the stage read off the store's recipe group;
* the epoch directory protocol (incomplete epochs ignored, the newest
  complete one wins, a crashed build's index never reused);
* the two-container alternation (exactly 3 of 4 at 0.75, all-original before
  the first epoch, one container for the births AND every density of a
  propose), its per-epoch per-rank seeds and the per-container tallies;
* adoption on every rank from the head's directive, and the three wire
  points in the GB move (``propose`` begins once, the payload ships the
  directive, the rank handler applies it before the sorter is built);
* the saver trigger: the save loop calls the hook after a saved row (with or
  without the monitor rank), the hook reads the store through the saver's
  own ``open("r")`` and leaves no handle behind, builds epochs at the
  cadence, survives a failing build, and its watchdog reports a held save
  and stops on dropped states;
* the leaf-row copy reads back through the pipeline's own readers, and the
  pipeline is called exactly as ``build.py`` calls it.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import logging
import os
import pickle
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

from lisatools.globalfit import hdfbackend as hb
from lisatools.globalfit import recipe
from lisatools.globalfit.warmstart import self_refit as sr
from lisatools.globalfit.warmstart.proposal import (
    CIRCULAR_COLS,
    COLUMN_NAMES,
    WarmStartComponents,
)

TOBS = 7776000.0
NDIM = 9


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _components(shift=0.0):
    means = np.array([
        [8.0, 2.0000 + shift, 0.60, 3.00, 0.30, 1.20, 4.00, 0.20, 0.05],
        [12.0, 2.0008 + shift, 0.50, 2.00, -0.40, 0.80, 1.00, -0.50, -0.10],
        [5.0, 5.0000 + shift, 0.70, 0.05, 0.00, 0.50, 5.50, 0.60, 0.20],
    ])
    sig = np.array([0.5, 2.0e-5, 0.01, 0.30, 0.05, 0.15, 0.05, 0.05, 0.02])
    covs = np.stack([np.diag(sig ** 2) for _ in range(len(means))])
    return means, covs, np.array([0.6, 0.3, 0.9])


def _write_components(path, shift=0.0):
    """The fit_from_store writer schema (sampling basis), exactly."""
    means, covs, p = _components(shift)
    n = len(means)
    meta = dict(store="synthetic", tobs=TOBS, df_mhz=1e3 / TOBS, last_k=None,
                column_names=COLUMN_NAMES, f0_units="mHz",
                circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()})
    np.savez_compressed(
        path, means=means, covs=covs, p=p, mult=np.ones(n),
        n_members=np.full(n, 100, dtype=np.int64),
        island_id=np.arange(n, dtype=np.int64),
        f0_window_edges=np.array([[1.99, 2.01], [4.99, 5.01]]),
        meta=json.dumps(meta))


_LO = [0.001, 0.5, 0.001, 0.0, -1.0, 0.0, 0.0, -1.0, -5.0]
_HI = [40.0, 30.0, 1.0, 2 * np.pi, 1.0, np.pi, 2 * np.pi, 1.0, 5.0]
LOAD_KW = dict(new_tobs=TOBS, use_cupy=False, floor_box=(_LO, _HI),
               floor_eps=0.05, p_floor=0.0)


def _write_epoch(root, k, *, shift=0.01, complete=True, npz=True, **extra):
    d = sr.epoch_dir(root, k)
    os.makedirs(d, exist_ok=True)
    if npz:
        _write_components(os.path.join(d, sr.REFEREED_NPZ), shift=shift)
    if complete:
        man = dict(epoch=k, built_at_row=extra.pop("built_at_row", 10 * k),
                   rows=[0, 49], n_samples=200, n_components=3,
                   npz=sr.REFEREED_NPZ, stage="gb_search_2", **extra)
        with open(os.path.join(d, sr.MANIFEST), "w") as fh:
            json.dump(man, fh)
    return d


class _TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="warmself_")
        self.addCleanup(shutil.rmtree, self.tmp, True)


def _mixer(root, *, build_seed=101, tag=recipe._WARM_SEED_TAG_SEARCH,
           fraction=0.75, name="rj_warm_search", orig_path=None):
    if orig_path is None:
        orig_path = os.path.join(root, "..", "orig.npz")
        if not os.path.exists(orig_path):
            _write_components(orig_path)
    orig = WarmStartComponents.from_npz(orig_path, seed=7, **LOAD_KW)
    return sr.WarmSelfMixer(orig, epoch_root=root, fraction=fraction,
                            load_kwargs=LOAD_KW, build_seed=build_seed,
                            tag=tag, name=name)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
class SettingsKnobTest(unittest.TestCase):
    FIELDS = {
        "warm_self": ("GB_WARM_SELF", False, "1", True),
        "warm_self_every_search": ("GB_WARM_SELF_EVERY_SEARCH", 5, "7", 7),
        "warm_self_every_pe": ("GB_WARM_SELF_EVERY_PE", 250, "100", 100),
        "warm_self_last_k": ("GB_WARM_SELF_LAST_K", 50, "30", 30),
        "warm_self_fraction": ("GB_WARM_SELF_FRACTION", 0.75, "0.5", 0.5),
    }

    def _fields(self):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        return {f.name: f for f in dataclasses.fields(GBSettings)}

    def test_defaults_and_env_knobs(self):
        fields = self._fields()
        for name, (env, default, raw, parsed) in self.FIELDS.items():
            self.assertIn(name, fields)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(env, None)
                self.assertEqual(fields[name].default_factory(), default, name)
                os.environ[env] = raw
                self.assertEqual(fields[name].default_factory(), parsed, name)

    def test_off_by_default_means_no_mixer(self):
        """``GB_WARM_SELF`` unset -> the warm move holds its ORIGINAL container."""
        cont = object()
        out = recipe._warm_self_wrap(
            cont, gb_info=SimpleNamespace(warm_self=False),
            general_info=SimpleNamespace(main_file_path="/x/s.h5"),
            tag=1, name="rj_warm_search", load_kwargs={})
        self.assertIs(out, cont)


class WrapTest(_TmpDir):
    def test_on_wraps_with_the_run_root_and_the_original(self):
        orig_path = os.path.join(self.tmp, "orig.npz")
        _write_components(orig_path)
        orig = WarmStartComponents.from_npz(orig_path, seed=1, **LOAD_KW)
        store = os.path.join(self.tmp, "run", "gf_testing.h5")
        gb_info = SimpleNamespace(warm_self=True, warm_self_fraction=0.75,
                                  build_seed=99)
        out = recipe._warm_self_wrap(
            orig, gb_info=gb_info,
            general_info=SimpleNamespace(main_file_path=store),
            tag=recipe._WARM_SEED_TAG_PE, name="rj_warm_pe",
            load_kwargs=dict(LOAD_KW, circ_images=3))
        self.assertTrue(out.is_warm_self_mixer)
        self.assertIs(out.original, orig)
        self.assertEqual(out.epoch_root,
                         os.path.join(self.tmp, "run", "warmstart", "self"))
        self.assertEqual(out.epoch_root, sr.self_epoch_root(store))
        self.assertEqual(out.load_kwargs["circ_images"], 3)
        self.assertEqual(out.tag, recipe._WARM_SEED_TAG_PE)
        self.assertEqual(str(out.fraction), "3/4")

    def test_both_twins_and_the_saver_are_wired(self):
        """``build_gb_moves`` / ``run_global_fit`` are too heavy to construct
        here; their wiring lines are pinned by source instead."""
        import re

        from lisatools.globalfit import run as gf_run

        src = inspect.getsource(recipe.build_gb_moves)
        for tag, name in (("_WARM_SEED_TAG_SEARCH", "rj_warm_search"),
                          ("_WARM_SEED_TAG_PE", "rj_warm_pe")):
            self.assertRegex(
                src, r'rj_proposal_distribution=\{"gb": _warm_self_wrap\(\s*'
                     r'_warm_(pe_)?container,[^)]*tag=' + tag
                     + r',\s*name="' + re.escape(name) + '"')
        self.assertIn("seed=_warm_start_seed(gb_info, _WARM_SEED_TAG_SEARCH),"
                      "\n            **_warm_load_kwargs,", src)
        self.assertIn("**_warm_pe_load_kwargs,", src)
        run_src = inspect.getsource(gf_run.GlobalFit.run_global_fit)
        self.assertIn("warm_self_hook=make_warm_self_hook(", run_src)


# ---------------------------------------------------------------------------
# cadence + clock
# ---------------------------------------------------------------------------
class CadenceClockTest(unittest.TestCase):
    def test_stage_kinds(self):
        self.assertEqual(sr.warm_self_stage_kind("gb_search_2"), "search")
        self.assertEqual(sr.warm_self_stage_kind("gb_search_3"), "search")
        self.assertEqual(sr.warm_self_stage_kind("full_pe"), "pe")
        for s in ("gb_search_1", "gb_search_seed", "source_search",
                  "noise_search", "replica_pe", None, ""):
            self.assertIsNone(sr.warm_self_stage_kind(s), s)

    def test_search_cadence_is_stage_relative(self):
        for rows, due in ((10, False), (14, False), (15, True), (16, True)):
            got, cad, clock = sr.warm_self_due(rows, "gb_search_2", 10, None,
                                               5, 250)
            self.assertEqual((got, cad, clock), (due, 5, 10), rows)

    def test_pe_cadence(self):
        self.assertFalse(sr.warm_self_due(389, "full_pe", 140, None, 5, 250)[0])
        self.assertTrue(sr.warm_self_due(390, "full_pe", 140, None, 5, 250)[0])

    def test_latest_epoch_seeds_the_clock_resume(self):
        """A resume at row 104 after an epoch built at 100 does NOT rebuild."""
        self.assertEqual(sr.warm_self_due(104, "gb_search_2", 10, 100, 5, 250),
                         (False, 5, 100))
        self.assertEqual(sr.warm_self_due(105, "gb_search_2", 10, 100, 5, 250),
                         (True, 5, 100))

    def test_an_earlier_stages_epoch_does_not_shorten_the_next_stage(self):
        # built at 120 in gb_search_2; full_pe started at 140
        self.assertEqual(sr.warm_self_due(141, "full_pe", 140, 120, 5, 250),
                         (False, 250, 140))
        self.assertTrue(sr.warm_self_due(390, "full_pe", 140, 120, 5, 250)[0])

    def test_a_rewound_store_ignores_an_epoch_built_past_it(self):
        self.assertEqual(sr.warm_self_due(30, "gb_search_2", 10, 80, 5, 250),
                         (True, 5, 10))

    def test_other_stages_and_disabled_cadences_never_build(self):
        self.assertFalse(sr.warm_self_due(999, "gb_search_1", 0, None, 5, 250)[0])
        self.assertFalse(sr.warm_self_due(999, "gb_search_2", 0, None, 0, 250)[0])
        self.assertFalse(sr.warm_self_due(999, "full_pe", 0, None, 5, -1)[0])


class StoreStageTest(_TmpDir):
    def _group(self, steps):
        path = tempfile.mktemp(suffix=".h5", dir=self.tmp)
        f = h5py.File(path, "w")
        self.addCleanup(f.close)
        g = f.create_group("global_fit")
        rg = g.create_group("recipe")
        for name, attrs in steps.items():
            s = rg.create_group(name)
            for k, v in attrs.items():
                s.attrs[k] = v
        return g

    def test_active_stage_and_its_start_stamp(self):
        g = self._group({
            "gb_search_1": {"order num": 1, "status": True,
                            "completed_iteration": 55},
            "gb_search_2": {"order num": 2, "status": False,
                            "start_iteration": 55},
            "full_pe": {"order num": 3, "status": False}})
        self.assertEqual(sr.store_stage(g), ("gb_search_2", 55))

    def test_falls_back_to_the_previous_completion(self):
        g = self._group({
            "gb_search_2": {"order num": 2, "status": True,
                            "completed_iteration": 140},
            "gb_search_1": {"order num": 1, "status": True,
                            "completed_iteration": 55},
            "full_pe": {"order num": 3, "status": False}})
        self.assertEqual(sr.store_stage(g), ("full_pe", 140))

    def test_first_stage_and_finished_recipe(self):
        g = self._group({"gb_search_seed": {"order num": 1, "status": False}})
        self.assertEqual(sr.store_stage(g), ("gb_search_seed", 0))
        g2 = self._group({"a": {"order num": 1, "status": True}})
        self.assertEqual(sr.store_stage(g2), (None, 0))


# ---------------------------------------------------------------------------
# the epoch directory protocol
# ---------------------------------------------------------------------------
class EpochProtocolTest(_TmpDir):
    def test_incomplete_epochs_are_ignored_and_the_latest_complete_wins(self):
        root = os.path.join(self.tmp, "self")
        _write_epoch(root, 0)
        _write_epoch(root, 1, complete=False)                # still building
        _write_epoch(root, 2, npz=False)                     # npz missing
        d3 = _write_epoch(root, 3, complete=False)
        with open(os.path.join(d3, sr.MANIFEST), "w") as fh:
            fh.write("{not json")                            # torn manifest
        k, man = sr.latest_complete_epoch(root)
        self.assertEqual(k, 0)
        self.assertTrue(man["npz_path"].endswith(sr.REFEREED_NPZ))
        # the next build skips EVERY existing dir, complete or not (a crashed
        # build's leftovers are never mixed into a new epoch)
        self.assertEqual(sr.next_epoch_index(root), 4)
        _write_epoch(root, 4)
        self.assertEqual(sr.latest_complete_epoch(root)[0], 4)
        self.assertIsNone(sr.latest_complete_epoch(root, newer_than=4))
        self.assertEqual(sr.latest_complete_epoch(root, newer_than=3)[0], 4)
        # a crashed build's directory index is never reused
        self.assertEqual(sr.next_epoch_index(root), 5)

    def test_a_manifest_for_another_epoch_is_not_complete(self):
        root = os.path.join(self.tmp, "self")
        d = _write_epoch(root, 2)
        with open(os.path.join(d, sr.MANIFEST), "w") as fh:
            json.dump({"epoch": 1, "npz": sr.REFEREED_NPZ}, fh)
        self.assertIsNone(sr.read_manifest(root, 2))

    def test_empty_or_missing_root(self):
        self.assertIsNone(sr.latest_complete_epoch(os.path.join(self.tmp, "x")))
        self.assertEqual(sr.next_epoch_index(os.path.join(self.tmp, "x")), 0)


# ---------------------------------------------------------------------------
# the two-container alternation
# ---------------------------------------------------------------------------
class AlternationTest(_TmpDir):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "run", "warmstart", "self")
        os.makedirs(self.root)

    def test_all_original_before_the_first_epoch(self):
        m = _mixer(self.root)
        x = np.array(_components()[0])
        for _ in range(8):
            d = m.begin_propose(rank=0)
            self.assertEqual(d, {"epoch": None, "choice": "orig"})
            self.assertIs(m.active, m.original)
            np.testing.assert_array_equal(m.logpdf(x), m.original.logpdf(x))
        self.assertEqual(m.n_self_window, 0)

    def test_exactly_three_of_four_at_075(self):
        m = _mixer(self.root)
        _write_epoch(self.root, 0)
        choices = [m.begin_propose(rank=0)["choice"] for _ in range(400)]
        self.assertEqual(m.epoch, 0)
        self.assertEqual(choices[:8], ["orig", "self", "self", "self"] * 2)
        self.assertEqual(choices.count("self"), 300)
        for i in range(0, 400, 4):              # every window of 4: exactly 3
            self.assertEqual(choices[i:i + 4].count("self"), 3)

    def test_other_fractions(self):
        for frac, want in ((0.0, 0), (1.0, 40), (0.5, 20), (0.25, 10)):
            m = _mixer(self.root, fraction=frac)
            _write_epoch(self.root, 0)
            got = [m.begin_propose()["choice"] for _ in range(40)]
            self.assertEqual(got.count("self"), want, frac)
        with self.assertRaises(ValueError):
            _mixer(self.root, fraction=1.5)

    def test_one_container_serves_births_and_every_density_of_a_propose(self):
        m = _mixer(self.root)
        _write_epoch(self.root, 0, shift=0.5)       # a clearly different law
        for _ in range(12):
            c = m.begin_propose()["choice"]
            held = m.self_container if c == "self" else m.original
            other = m.original if c == "self" else m.self_container
            self.assertIs(m.active, held)
            x = m.rvs(size=64)                       # the forward draws ...
            lp_fwd = m.logpdf(x)
            np.testing.assert_array_equal(lp_fwd, held.logpdf(x))
            # ... and a later (reverse / death-side) density in the SAME
            # propose come from the same container
            np.testing.assert_array_equal(m.logpdf(x), lp_fwd)
            self.assertFalse(np.allclose(other.logpdf(x), lp_fwd))

    def test_pickle_and_deepcopy(self):
        m = _mixer(self.root)
        _write_epoch(self.root, 0)
        m.begin_propose()
        m.begin_propose()
        m2 = pickle.loads(pickle.dumps(copy.deepcopy(m)))
        self.assertEqual((m2.epoch, m2.choice, m2.n_self_window),
                         (m.epoch, m.choice, m.n_self_window))
        x = np.array(_components()[0])
        np.testing.assert_allclose(m2.logpdf(x), m.logpdf(x))

    def test_census_tallies_per_container(self):
        m = _mixer(self.root)
        sp = dict(births=10, viable=8, birth_acc=2, viable_cold=4,
                  birth_acc_cold=1, deaths=3, death_acc=1)
        m.begin_propose()
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            m.note_census(sp, move_name="rj_warm_search", rank=2)
        self.assertIn("[GB_WARM_SELF rj_warm_search] r2 this propose used ORIG",
                      "\n".join(cm.output))
        _write_epoch(self.root, 0)
        for _ in range(4):
            m.begin_propose()
            m.note_census(sp)
        self.assertEqual(m.tallies["orig"]["proposes"], 2)
        self.assertEqual(m.tallies["self"]["proposes"], 3)
        self.assertEqual(m.tallies["self"]["birth_acc_cold"], 3)
        self.assertEqual(m.tallies["orig"]["viable"], 16)


class SeedTest(_TmpDir):
    def test_epoch_seed_is_the_recipe_formula_per_rank_and_per_epoch(self):
        root = os.path.join(self.tmp, "self")
        os.makedirs(root)
        tag = recipe._WARM_SEED_TAG_SEARCH
        m = _mixer(root, build_seed=12345, tag=tag)
        for k in range(4):
            self.assertEqual(
                m.epoch_seed(k),
                recipe._warm_start_seed(SimpleNamespace(build_seed=12345),
                                        tag ^ (k + 1)))
        self.assertEqual(len({m.epoch_seed(k) for k in range(6)}), 6)
        m2 = _mixer(root, build_seed=54321, tag=tag)
        self.assertNotEqual(m.epoch_seed(0), m2.epoch_seed(0))
        self.assertIsNone(_mixer(root, build_seed=None).epoch_seed(0))

    def test_adoption_uses_the_epoch_seed(self):
        root = os.path.join(self.tmp, "self")
        os.makedirs(root)
        _write_epoch(root, 3)
        m = _mixer(root, build_seed=77)
        with mock.patch.object(WarmStartComponents, "from_npz",
                               wraps=WarmStartComponents.from_npz) as fz:
            m.poll()
        self.assertEqual(fz.call_args.kwargs["seed"], m.epoch_seed(3))
        for k, v in LOAD_KW.items():
            self.assertEqual(fz.call_args.kwargs[k], v)


class AdoptionTest(_TmpDir):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "run", "warmstart", "self")
        os.makedirs(self.root)

    def test_every_rank_adopts_the_heads_epoch_and_choice(self):
        head = _mixer(self.root, build_seed=1)
        ranks = [_mixer(self.root, build_seed=s) for s in (2, 3, 4)]
        _write_epoch(self.root, 0)
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            for _ in range(5):
                d = head.begin_propose(rank=0)
                for r, m in enumerate(ranks, start=1):
                    m.apply_directive(d, rank=r)
                    self.assertEqual((m.epoch, m.choice),
                                     (head.epoch, head.choice))
        text = "\n".join(cm.output)
        for r in range(4):
            self.assertIn(f"[GB_WARM_SELF] rank {r} adopted epoch 0 "
                          "(rows 0..49, 200 samples, 3 components)", text)
        # each rank adopted ONCE, not per propose
        self.assertEqual(text.count("adopted epoch 0"), 4)
        # a newer complete epoch: the head polls, the ranks follow
        _write_epoch(self.root, 1, shift=0.02)
        d = head.begin_propose()
        self.assertEqual(d["epoch"], 1)
        for m in ranks:
            m.apply_directive(d)
            self.assertEqual(m.epoch, 1)

    def test_an_incomplete_epoch_is_not_adopted(self):
        m = _mixer(self.root)
        _write_epoch(self.root, 0, complete=False)
        self.assertEqual(m.begin_propose()["epoch"], None)
        self.assertIsNone(m.self_container)

    def test_a_rank_that_cannot_load_keeps_the_original(self):
        r = _mixer(self.root)
        with self.assertLogs(sr.logger, level=logging.WARNING):
            r.apply_directive({"epoch": 5, "choice": "self"}, rank=3)
        self.assertEqual(r.choice, "orig")
        self.assertIsNone(r.self_container)
        r.apply_directive(None)
        self.assertEqual(r.choice, "orig")

    def test_a_corrupt_npz_keeps_the_held_epoch(self):
        m = _mixer(self.root)
        _write_epoch(self.root, 0)
        m.begin_propose()
        d = _write_epoch(self.root, 1)
        with open(os.path.join(d, sr.REFEREED_NPZ), "wb") as fh:
            fh.write(b"garbage")
        with self.assertLogs(sr.logger, level=logging.WARNING):
            m.begin_propose()
        self.assertEqual(m.epoch, 0)


class MoveWiringTest(_TmpDir):
    """The three call sites in the GB move (bare ``__new__`` moves)."""

    @classmethod
    def setUpClass(cls):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        cls.gbs = gbs

    def _move(self, mixer, rank=None):
        m = self.gbs.GBSpecialRJPriorMove.__new__(self.gbs.GBSpecialRJPriorMove)
        m.name = "rj_warm_search"
        m.branch_name = "gb"
        m.gf_rank = rank
        m.rj_proposal_distribution = {"gb": mixer}
        return m

    def test_propose_begins_once_and_the_ranks_apply_it(self):
        root = os.path.join(self.tmp, "self")
        os.makedirs(root)
        _write_epoch(root, 0)
        head = self._move(_mixer(root, build_seed=1), rank=0)
        ranks = [self._move(_mixer(root, build_seed=s), rank=r)
                 for r, s in ((1, 2), (2, 3))]
        head._replace_passes = lambda: ("a", "b")     # even with 2 passes
        head._propose_replace_passes = lambda model, state, passes: "done"
        for _ in range(4):
            self.assertEqual(self.gbs.GBSpecialBase.propose(head, None, None),
                             "done")
            for r in ranks:
                r._warm_self_apply(head._warm_self_directive)
                self.assertEqual(r._warm_self_mixer().choice,
                                 head._warm_self_mixer().choice)
        self.assertEqual(head._warm_self_mixer().n_self_window, 4)

    def test_no_mixer_is_a_no_op(self):
        m = self._move(object())
        m._warm_self_begin()
        self.assertIsNone(m._warm_self_directive)
        m._warm_self_apply({"epoch": 1, "choice": "self"})
        m._warm_self_note({"births": 1})
        m.rj_proposal_distribution = None
        m._warm_self_begin()
        self.assertIsNone(m._warm_self_directive)

    def test_payload_ships_the_directive_and_the_rank_applies_it_first(self):
        src = inspect.getsource(self.gbs.GBSpecialBase._propose_orchestrated)
        self.assertIn('"warm_self": getattr(self, "_warm_self_directive", None)',
                      src)
        srv = inspect.getsource(self.gbs.GBSpecialBase._gb_serve_run_proposal)
        i_apply = srv.index('self._warm_self_apply((payload or {}).get("warm_self"))')
        i_res = srv.index("self.rj_proposal_distribution[self.branch_name]")
        self.assertLess(i_apply, i_res)
        census = inspect.getsource(self.gbs.GBSpecialBase)
        self.assertIn("self._warm_self_note(sp)", census)


# ---------------------------------------------------------------------------
# the store read + the saver trigger
# ---------------------------------------------------------------------------
class _Reader:
    """The saver backend's surface the hook uses: ``filename``, ``name``,
    ``open(mode)`` returning an h5py File (eryn's ``HDFBackend.open``)."""

    name = "global_fit"

    def __init__(self, filename):
        self.filename = filename
        self.modes = []

    def open(self, mode="r"):
        self.modes.append(mode)
        return h5py.File(self.filename, mode)


class _Comm:
    def __init__(self, probes=()):
        self._probes = list(probes)

    def iprobe(self, source=None):
        return self._probes.pop(0) if self._probes else False


def _make_store(path, *, alloc=40, nw=4, nleaves=12, seed=3):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = 0
        ch = rng.uniform(0.1, 1.0, size=(alloc, 1, 1, nw, nleaves, NDIM))
        ch[..., 1] = rng.uniform(1.0, 10.0, size=ch.shape[:-1])   # f0 mHz
        ind = rng.random((alloc, 1, 1, nw, nleaves)) < 0.6
        ind[:, :, :, :, 0] = True
        g.create_group("chain").create_dataset("gb", data=ch)
        g.create_group("inds").create_dataset("gb", data=ind)
        g.create_dataset("log_like", data=-1e3 - rng.random((alloc, 1, 1, nw)))
        rg = g.create_group("recipe")
        for name, attrs in (
                ("gb_search_1", {"order num": 1, "status": True,
                                 "completed_iteration": 10}),
                ("gb_search_2", {"order num": 2, "status": False,
                                 "start_iteration": 10}),
                ("full_pe", {"order num": 3, "status": False})):
            s = rg.create_group(name)
            for k, v in attrs.items():
                s.attrs[k] = v


def _set_rows(path, rows):
    with h5py.File(path, "a") as f:
        f["global_fit"].attrs["iteration"] = int(rows)


class _StubBuilder:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, leaf, d, *, tobs, last_k, device, runner=None):
        with h5py.File(leaf, "r") as f:
            g = f["global_fit"]
            rows = int(g.attrs["iteration"])
            n = int(np.asarray(g["inds"]["gb"]).sum())
        self.calls.append(dict(leaf=leaf, d=d, tobs=tobs, last_k=last_k,
                               device=device, rows=rows, n=n))
        if self.fail:
            raise RuntimeError("synthetic build failure")
        _write_components(os.path.join(d, sr.REFEREED_NPZ))
        return dict(npz=sr.REFEREED_NPZ, n_clusters=3, n_components=3,
                    walls=dict(fit=0.1, referee=0.2, apply=0.01))


class CopyLeafRowsTest(_TmpDir):
    def test_packed_rows_read_back_through_the_pipeline_readers(self):
        from lisatools.globalfit.warmstart.fit_from_store import load_leaf_table
        from lisatools.globalfit.warmstart.match_referee import load_leaf_rows

        src = os.path.join(self.tmp, "s.h5")
        _make_store(src)
        out = os.path.join(self.tmp, "leaf.h5")
        with h5py.File(src, "r") as f:
            prov = sr.copy_leaf_rows(f["global_fit"], out, last_k=6, rows=20)
            ch = f["global_fit/chain/gb"][14:20, 0, 0]
            ind = f["global_fit/inds/gb"][14:20, 0, 0]
        self.assertEqual(prov["rows"], [14, 19])
        self.assertEqual(prov["n_samples"], 24)
        want = np.concatenate([ch[r][w][ind[r][w]] for r in range(6)
                               for w in range(4)])
        X, sid, info = load_leaf_table(out, 6)
        self.assertEqual(info["n_samples"], 24)
        np.testing.assert_array_equal(X, want)
        np.testing.assert_array_equal(sid // 4, np.repeat(
            np.arange(6), [ind[r].sum() for r in range(6)]))
        X2, _ = load_leaf_rows(out, 6)
        np.testing.assert_array_equal(X2, want)
        self.assertEqual(prov["n_leaf_rows"], len(want))

    def test_short_store_takes_what_is_there(self):
        src = os.path.join(self.tmp, "s.h5")
        _make_store(src)
        with h5py.File(src, "r") as f:
            prov = sr.copy_leaf_rows(f["global_fit"],
                                     os.path.join(self.tmp, "l.h5"), 50, 7)
        self.assertEqual(prov["rows"], [0, 6])


class HookTest(_TmpDir):
    def setUp(self):
        super().setUp()
        self.run = os.path.join(self.tmp, "run")
        os.makedirs(self.run)
        self.store = os.path.join(self.run, "gf_testing.h5")
        _make_store(self.store)
        self.root = sr.self_epoch_root(self.store)
        self.reader = _Reader(self.store)
        self.builder = _StubBuilder()
        self.hook = self._hook()

    def _hook(self, **kw):
        return sr.WarmSelfHook(epoch_root=self.root, tobs=TOBS, last_k=6,
                               every_search=5, every_pe=250,
                               builder=kw.pop("builder", self.builder), **kw)

    def _save(self, rows, hook=None, comm=None):
        _set_rows(self.store, rows)
        (hook or self.hook)(self.reader, comm or _Comm(), 0, rows)

    def assert_no_handle_left(self):
        with h5py.File(self.store, "a") as f:     # the saver's next write
            f["global_fit"].attrs["probe"] = 1

    def test_builds_at_the_cadence_and_only_in_the_right_stages(self):
        for rows in range(10, 15):
            self._save(rows)
        self.assertEqual(self.builder.calls, [])
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            self._save(15)
        self.assertEqual(len(self.builder.calls), 1)
        self.assertIn("[GB_WARM_SELF] epoch 0 built from rows 9..14 (24 "
                      "samples, 3 components)", "\n".join(cm.output))
        k, man = sr.latest_complete_epoch(self.root)
        self.assertEqual(k, 0)
        self.assertEqual(man["built_at_row"], 15)
        self.assertEqual(man["rows"], [9, 14])
        self.assertEqual(man["stage"], "gb_search_2")
        self.assertEqual((man["n_samples"], man["n_components"]), (24, 3))
        for w in ("read", "fit", "referee", "apply", "total"):
            self.assertIn(w, man["walls"])
        call = self.builder.calls[0]
        self.assertEqual((call["rows"], call["last_k"], call["tobs"]),
                         (6, 6, TOBS))
        self.assertFalse(os.path.exists(call["leaf"]))   # the copy is removed
        for rows in range(16, 20):
            self._save(rows)
        self.assertEqual(len(self.builder.calls), 1)
        self._save(20)
        self.assertEqual(len(self.builder.calls), 2)
        self.assertEqual(sr.latest_complete_epoch(self.root)[0], 1)
        # the saver opened its store READ-ONLY only, and left it closed
        self.assertEqual(set(self.reader.modes), {"r"})
        self.assert_no_handle_left()

    def test_resume_never_rebuilds_early(self):
        self._save(15)
        self.assertEqual(len(self.builder.calls), 1)
        fresh = self._hook()                      # a new process, same disk
        for rows in (16, 17, 18, 19):
            self._save(rows, hook=fresh)
        self.assertEqual(len(self.builder.calls), 1)
        self._save(20, hook=fresh)
        self.assertEqual(len(self.builder.calls), 2)

    def test_full_pe_cadence_read_off_the_store(self):
        with h5py.File(self.store, "a") as f:
            rg = f["global_fit/recipe"]
            rg["gb_search_2"].attrs["status"] = True
            rg["gb_search_2"].attrs["completed_iteration"] = 12
            rg["full_pe"].attrs["start_iteration"] = 12
        self._save(30)
        self.assertEqual(self.builder.calls, [])
        hook = self._hook()
        hook.every_pe = 20
        self._save(31, hook=hook)
        self.assertEqual(self.builder.calls, [])
        self._save(32, hook=hook)
        self.assertEqual(len(self.builder.calls), 1)
        self.assertEqual(sr.latest_complete_epoch(self.root)[1]["stage"],
                         "full_pe")

    def test_a_queued_save_wins(self):
        _set_rows(self.store, 15)
        self.hook(self.reader, _Comm(probes=[True]), 0, 15)
        self.assertEqual(self.builder.calls, [])
        self._save(16)
        self.assertEqual(len(self.builder.calls), 1)

    def test_a_failing_build_is_ignored_and_retried_after_a_cadence(self):
        bad = _StubBuilder(fail=True)
        hook = self._hook(builder=bad)
        with self.assertLogs(sr.logger, level=logging.WARNING) as cm:
            self._save(15, hook=hook)
        self.assertIn("epoch 0 build FAILED", "\n".join(cm.output))
        self.assertIsNone(sr.latest_complete_epoch(self.root))
        self.assertTrue(os.path.isdir(sr.epoch_dir(self.root, 0)))
        self.assert_no_handle_left()
        for rows in (16, 17, 18, 19):
            self._save(rows, hook=hook)
        self.assertEqual(len(bad.calls), 1)
        hook._builder = self.builder
        self._save(20, hook=hook)
        self.assertEqual(sr.latest_complete_epoch(self.root)[0], 1)

    def test_the_hook_never_raises(self):
        self.reader.open = mock.Mock(side_effect=OSError("boom"))
        with self.assertLogs(sr.logger, level=logging.WARNING):
            self.hook(self.reader, _Comm(), 0, 15)

    def test_watchdog_reports_a_held_save(self):
        _set_rows(self.store, 15)
        with self.assertLogs(sr.logger, level=logging.WARNING) as cm:
            self.hook(self.reader, _Comm(probes=[False, True]), 0, 15)
        self.assertIn("HOLDING THE", "\n".join(cm.output))
        self.assertEqual(self.hook.watchdog.delayed, 1)

    def test_watchdog_stops_the_builder_on_dropped_states(self):
        """States a build cost are dropped while the saver drains the backlog
        it left -- in the NEXT batch -- so the verdict comes at the next call."""
        _set_rows(self.store, 15)
        self.hook(self.reader, _Comm(), 0, 15, dropped_so_far=3)
        self.assertEqual(len(self.builder.calls), 1)
        self.assertIsNone(self.hook.watchdog.disabled_reason)
        _set_rows(self.store, 16)
        with self.assertLogs(sr.logger, level=logging.WARNING) as cm:
            self.hook(self.reader, _Comm(), 0, 16, dropped_so_far=5)
        self.assertIn("STOPPING THE SELF REFIT", "\n".join(cm.output))
        self.assertIsNotNone(self.hook.watchdog.disabled_reason)
        _set_rows(self.store, 25)
        self.hook(self.reader, _Comm(), 0, 25, dropped_so_far=5)
        self.assertEqual(len(self.builder.calls), 1)

    def test_drops_before_a_build_are_not_blamed_on_it(self):
        _set_rows(self.store, 12)
        self.hook(self.reader, _Comm(), 0, 12, dropped_so_far=4)  # no build
        for rows in (15, 16, 20):
            _set_rows(self.store, rows)
            self.hook(self.reader, _Comm(), 0, rows, dropped_so_far=4)
        self.assertIsNone(self.hook.watchdog.disabled_reason)
        self.assertEqual(len(self.builder.calls), 2)

    def test_a_failed_build_that_held_a_save_is_reported(self):
        hook = self._hook(builder=_StubBuilder(fail=True))
        _set_rows(self.store, 15)
        with self.assertLogs(sr.logger, level=logging.WARNING) as cm:
            hook(self.reader, _Comm(probes=[False, True]), 0, 15)
        text = "\n".join(cm.output)
        self.assertIn("build FAILED", text)
        self.assertIn("HOLDING THE", text)


class SaverLoopTest(unittest.TestCase):
    class _LoopComm:
        def __init__(self, payloads):
            self._payloads = list(payloads)

        def recv(self, source=None):
            return self._payloads.pop(0)

        def iprobe(self, source=None):
            return False

    class _Reader:
        filename = "/run/dir/gf_prod.h5"

        def save_step_main(self, *a, **k):
            pass

    class _Hook:
        def __init__(self):
            self.saves = 0
            self.calls = []

        def note_save(self):
            self.saves += 1

        def __call__(self, gb_reader, comm, main_rank, i, *, dropped_so_far=0):
            self.calls.append(i)

    def _run(self, payloads, hook, monitor_rank=False):
        from lisatools.globalfit.monitor import rank as mr

        with mock.patch.object(hb, "_atomic_backup_copy"), \
                mock.patch.object(mr, "active", return_value=monitor_rank), \
                mock.patch.object(mr, "notify_saved") as ns:
            hb.save_to_backend_asynchronously_and_plot(
                self._Reader(), self._LoopComm(payloads), main_rank=0,
                plot_container=None, warm_self_hook=hook)
        return ns

    def _saves(self, n):
        return [{"save_args": (), "save_kwargs": {}}] * n

    def test_called_after_every_saved_batch(self):
        h = self._Hook()
        self._run(self._saves(1) + self._saves(1) + [{"finish_run": True}], h)
        self.assertEqual(h.saves, 2)
        self.assertEqual(h.calls, [1, 2])

    def test_called_with_the_monitor_rank_too(self):
        h = self._Hook()
        ns = self._run(self._saves(1) + [{"finish_run": True}], h,
                       monitor_rank=True)
        self.assertEqual(h.calls, [1])
        ns.assert_called_once()

    def test_no_hook_is_the_old_loop(self):
        self._run(self._saves(2) + [{"finish_run": True}], None)


class MakeHookTest(unittest.TestCase):
    def _curr(self, **gb):
        g = dict(warm_self=True, warm_start_components="/p/ws.npz",
                 warm_self_last_k=50, warm_self_every_search=5,
                 warm_self_every_pe=250)
        g.update(gb)
        return SimpleNamespace(
            source_info={"gb": SimpleNamespace(**g)},
            general_info=SimpleNamespace(main_file_path="/run/x/s.h5",
                                         Tobs=2.3e7, gpus=[1]))

    def test_off_and_on(self):
        self.assertIsNone(sr.make_warm_self_hook(self._curr(warm_self=False)))
        with mock.patch.dict(os.environ, {"GB_WARM_START_COMPONENTS": ""}), \
                self.assertLogs(sr.logger, level=logging.WARNING):
            self.assertIsNone(sr.make_warm_self_hook(
                self._curr(warm_start_components="")))
        h = sr.make_warm_self_hook(self._curr())
        self.assertEqual((h.epoch_root, h.tobs, h.last_k, h.every_search,
                          h.every_pe, h.device),
                         ("/run/x/warmstart/self", 2.3e7, 50, 5, 250, 1))
        with self.assertRaises(ValueError):
            sr.make_warm_self_hook(self._curr(warm_self_last_k=0))


class PipelineCallTest(_TmpDir):
    def test_the_three_steps_exactly_as_build_py_runs_them(self):
        calls = []

        def runner(cmd):
            calls.append(list(cmd))
            a = cmd[1:]
            if cmd[0] == "fit_from_store":
                open(a[a.index("--out") + 1], "wb").close()
            elif cmd[0] == "match_referee":
                npz = a[a.index("--npz") + 1]
                open(os.path.splitext(npz)[0] + "_referee.npz", "wb").close()
            else:
                _write_components(a[a.index("--out") + 1])

        res = sr.build_self_epoch("/r/leaf.h5", self.tmp, tobs=TOBS,
                                  last_k=50, device=None, runner=runner)
        fit = os.path.join(self.tmp, sr.FIT_NPZ)
        ref = os.path.splitext(fit)[0] + "_referee.npz"
        self.assertEqual(calls, [
            ["fit_from_store", "--store", "/r/leaf.h5", "--last-k", "50",
             "--tobs", repr(TOBS), "--out", fit, "--gmm-gpu", "cpu"],
            ["match_referee", "--npz", fit, "--store", "/r/leaf.h5"],
            ["referee_apply", "--fit", fit, "--referee", ref, "--out",
             os.path.join(self.tmp, sr.REFEREED_NPZ)],
        ])
        self.assertEqual((res["n_clusters"], res["n_components"]), (3, 3))

    def test_the_default_runner_is_build_pys(self):
        from lisatools.globalfit.warmstart import build

        with mock.patch.object(build, "_default_runner",
                               side_effect=RuntimeError("stop")) as dr:
            with self.assertRaises(RuntimeError):
                sr.build_self_epoch("/r/leaf.h5", self.tmp, tobs=TOBS,
                                    last_k=5)
        dr.assert_called_once()


if __name__ == "__main__":
    unittest.main()
