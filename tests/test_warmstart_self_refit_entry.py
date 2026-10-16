"""GB warm-start SELF refit built ON THE HEAD at the warm move's entry, with a
three-way split (Mike, 2026-10-14/15; ``warmstart/self_refit.py``).

"Let's just build everything on the head at entry time for the refit ... Make
that the default." "... split the refits three ways: old run refit that never
changes, new refit with all sources, and new refit removal of sources that are
already in all walkers. ... each 1/3 of the time ... 1 refit ... per update."
"search split should be 100% refit on sources that do not exist in all four
(with high confidence)." "Use the same refitting cadence as we do now." "Set
the default refitting duration for full pe ... to be 25 iterations."

Pinned here (CPU, KB-sized fixtures, the fit/referee pipeline stubbed where it
would need waveforms):

* the settings and their env knobs (head entry ON, PE every 25, the splits,
  the consensus threshold) and that the saver hook is off at entry;
* the consensus filter: a cluster is dropped only when EVERY walker holds it in
  >= threshold of its rows with the same leaf count in >= threshold of samples;
* the head's buffer: what it writes equals the saver's store copy, it keeps
  the last ``last_k`` rows, and tracks the stage of the rows it saw;
* the head builder: due only in numbered search stages / full_pe at the
  cadence, one epoch per build carrying BOTH npz files, never raises;
* the mixer: original / full / filtered by the split's deterministic schedule,
  the build happens at the head's propose entry and is adopted in that same
  propose with ``adopt_lag=False`` (the default one-draw lag is pinned in
  ``test_warmstart_self_refit_lag.py``), compute ranks never build.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import pickle
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

from lisatools.globalfit import recipe
from lisatools.globalfit.warmstart import self_refit as sr
from lisatools.globalfit.warmstart.proposal import (
    CIRCULAR_COLS,
    COLUMN_NAMES,
    WarmStartComponents,
)
from tests.test_warmstart_self_refit import (
    LOAD_KW,
    TOBS,
    _make_store,
    _StubBuilder,
    _TmpDir,
    _write_components,
    _write_epoch,
)

NW = 4
F0S = (2.0, 3.0, 4.0, 5.0)                 # mHz: clusters A, B, C, D
SIG = np.array([0.5, 2.0e-5, 0.01, 0.30, 0.05, 0.15, 0.05, 0.05, 0.02])
BASE = np.array([8.0, 0.0, 0.60, 3.00, 0.30, 1.20, 4.00, 0.20, 0.05])


def _mean(f0):
    m = BASE.copy()
    m[1] = f0
    return m


def _write_four(path):
    """Legacy (sampling-basis) components npz: one cluster per F0S entry."""
    means = np.stack([_mean(f) for f in F0S])
    covs = np.stack([np.diag(SIG ** 2) for _ in F0S])
    n = len(F0S)
    meta = dict(store="synthetic", tobs=TOBS, df_mhz=1e3 / TOBS, last_k=None,
                column_names=COLUMN_NAMES, f0_units="mHz",
                circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()})
    np.savez_compressed(
        path, means=means, covs=covs, p=np.ones(n), mult=np.ones(n),
        n_members=np.full(n, 24, dtype=np.int64),
        island_id=np.arange(n, dtype=np.int64),
        f0_window_edges=np.array([[f - 0.01, f + 0.01] for f in F0S]),
        meta=json.dumps(meta))


def _write_leaf_store(path, rows=6, seed=5):
    """The copy_leaf_rows layout. A: every walker, every row, one leaf.
    B: walkers 0-2 only. C: every walker, but walker 3 holds TWO leaves.
    D: every walker, but walker 1 misses one row (5 of 6)."""
    rng = np.random.default_rng(seed)
    per = [[[] for _ in range(NW)] for _ in range(rows)]
    for r in range(rows):
        for w in range(NW):
            per[r][w].append(_mean(F0S[0]))
            if w < 3:
                per[r][w].append(_mean(F0S[1]))
            per[r][w].append(_mean(F0S[2]))
            if w == 3:
                second = _mean(F0S[2])
                second[1] += 0.5 * SIG[1]
                per[r][w].append(second)
            if not (w == 1 and r == 2):
                per[r][w].append(_mean(F0S[3]))
    M = max(len(x) for row in per for x in row)
    ch = np.zeros((rows, 1, 1, NW, M, 9))
    ind = np.zeros((rows, 1, 1, NW, M), bool)
    for r in range(rows):
        for w in range(NW):
            x = np.asarray(per[r][w]) + 0.05 * SIG * rng.standard_normal(
                (len(per[r][w]), 9))
            ch[r, 0, 0, w, :len(x)] = x
            ind[r, 0, 0, w, :len(x)] = True
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = rows
        g.create_group("chain").create_dataset("gb", data=ch)
        g.create_group("inds").create_dataset("gb", data=ind)
        g.create_dataset("log_like", data=-np.ones((rows, 1, 1, NW)))


class SettingsTest(unittest.TestCase):
    FIELDS = {
        "warm_self_at_entry": ("GB_WARM_SELF_AT_ENTRY", True, "0", False),
        "warm_self_every_pe": ("GB_WARM_SELF_EVERY_PE", 25, "40", 40),
        "warm_self_split_search": ("GB_WARM_SELF_SPLIT_SEARCH", "0,0,1", "1,0,1", "1,0,1"),
        "warm_self_split_pe": ("GB_WARM_SELF_SPLIT_PE", "1,1,1", "2,1,1", "2,1,1"),
        "warm_self_consensus": ("GB_WARM_SELF_CONSENSUS", 0.95, "0.9", 0.9),
    }

    def test_defaults_and_env_knobs(self):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        fields = {f.name: f for f in dataclasses.fields(GBSettings)}
        for name, (env, default, raw, parsed) in self.FIELDS.items():
            self.assertIn(name, fields)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(env, None)
                self.assertEqual(fields[name].default_factory(), default, name)
                os.environ[env] = raw
                self.assertEqual(fields[name].default_factory(), parsed, name)
        self.assertNotIn("warm_self_fraction", fields)
        self.assertNotIn("warm_self_fraction_search", fields)

    def test_split_parsing(self):
        self.assertEqual(sr.parse_split("1,1,1"), (1.0, 1.0, 1.0))
        self.assertEqual(sr.parse_split("0, 0, 1"), (0.0, 0.0, 1.0))
        for bad in ("1,1", "0,0,0", "1,-1,1", "a,b,c"):
            with self.assertRaises(ValueError):
                sr.parse_split(bad)

    def test_the_saver_hook_is_off_when_the_head_builds(self):
        def curr(at_entry):
            gb = SimpleNamespace(warm_self=True, warm_self_at_entry=at_entry,
                                 warm_start_components="/x/w.npz",
                                 warm_self_last_k=50, warm_self_every_search=1,
                                 warm_self_every_pe=25)
            gi = SimpleNamespace(main_file_path="/x/run/s.h5", Tobs=TOBS,
                                 gpus=[0])
            return SimpleNamespace(source_info={"gb": gb}, general_info=gi)
        self.assertIsNone(sr.make_warm_self_hook(curr(True)))
        self.assertIsInstance(sr.make_warm_self_hook(curr(False)),
                              sr.WarmSelfHook)


class ConsensusFilterTest(_TmpDir):
    def setUp(self):
        super().setUp()
        self.npz = os.path.join(self.tmp, "ref.npz")
        self.leaf = os.path.join(self.tmp, "leaf.h5")
        _write_four(self.npz)
        _write_leaf_store(self.leaf)

    def test_only_the_cluster_every_walker_holds_once_is_dropped(self):
        out = os.path.join(self.tmp, "filt.npz")
        res = sr.consensus_filter(self.npz, self.leaf, out, threshold=0.95)
        np.testing.assert_array_equal(res["consensus"],
                                      [True, False, False, False])
        self.assertEqual((res["n_clusters"], res["n_kept"]), (4, 3))
        with np.load(out) as z:
            np.testing.assert_allclose(z["means"][:, 1], F0S[1:])
            self.assertEqual(len(z["p"]), 3)
            np.testing.assert_array_equal(z["island_id"], [1, 2, 3])
        cont = WarmStartComponents.from_npz(out, seed=1, **LOAD_KW)
        self.assertEqual(cont.n_components, 3)

    def test_a_looser_threshold_drops_the_cluster_one_row_short(self):
        res = sr.consensus_filter(self.npz, self.leaf,
                                  os.path.join(self.tmp, "f.npz"),
                                  threshold=0.8)
        # D: walker 1 holds it in 5 of 6 rows (0.83) -> consensus at 0.8;
        # C: 6 of 24 samples hold two leaves (0.75) -> still kept
        np.testing.assert_array_equal(res["consensus"],
                                      [True, False, False, True])


class BufferTest(_TmpDir):
    def test_written_rows_equal_the_savers_store_copy(self):
        store = os.path.join(self.tmp, "s.h5")
        _make_store(store)
        with h5py.File(store, "a") as f:
            f["global_fit"].attrs["iteration"] = 20
        buf = sr.WarmSelfBuffer(last_k=6)
        with h5py.File(store, "r") as f:
            buf.fill_from_store(f["global_fit"])
            sr.copy_leaf_rows(f["global_fit"], os.path.join(self.tmp, "a.h5"),
                              6, 20)
        prov = buf.write(os.path.join(self.tmp, "b.h5"))
        self.assertEqual((buf.n_stored, buf.stage_name, buf.stage_start),
                         (20, "gb_search_2", 10))
        self.assertEqual(prov["rows"], [14, 19])
        with h5py.File(os.path.join(self.tmp, "a.h5")) as a, \
                h5py.File(os.path.join(self.tmp, "b.h5")) as b:
            for k in ("chain/gb", "inds/gb", "log_like"):
                np.testing.assert_array_equal(a["global_fit"][k][()],
                                              b["global_fit"][k][()], k)
            self.assertEqual(a["global_fit"].attrs["iteration"],
                             b["global_fit"].attrs["iteration"])

    def test_appended_states_keep_the_last_k_cold_rows(self):
        buf = sr.WarmSelfBuffer(last_k=3)
        rng = np.random.default_rng(1)
        for r in range(5):
            coords = rng.uniform(1, 2, size=(2, NW, 7, 9))   # 2 temps: cold = 0
            inds = np.zeros((2, NW, 7), bool)
            inds[0, :, : r + 1] = True
            inds[1] = True                                    # hot: ignored
            st = SimpleNamespace(
                branches={"gb": SimpleNamespace(coords=coords, inds=inds)},
                log_like=-np.ones((2, NW)) * r)
            buf.append_state(st)
        self.assertEqual(buf.n_stored, 5)
        prov = buf.write(os.path.join(self.tmp, "c.h5"))
        self.assertEqual(prov["rows"], [2, 4])
        self.assertEqual(prov["n_leaf_rows"], NW * (3 + 4 + 5))
        with h5py.File(os.path.join(self.tmp, "c.h5")) as f:
            np.testing.assert_array_equal(f["global_fit/log_like"][:, 0, 0, 0],
                                          [-2.0, -3.0, -4.0])


class HeadBuilderTest(_TmpDir):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "run", "warmstart", "self")
        self.buf = sr.WarmSelfBuffer(last_k=6)
        self.calls = []

    def _filter(self, refereed, leaf, out, *, threshold):
        self.calls.append(threshold)
        _write_components(out, shift=0.3)
        return dict(n_clusters=3, n_kept=2, n_components_kept=2,
                    consensus=np.array([True, False, False]))

    def _builder(self, **kw):
        return sr.WarmSelfHeadBuilder(
            epoch_root=self.root, buffer=self.buf, tobs=TOBS, last_k=6,
            every_search=1, every_pe=25, device=None, consensus=0.95,
            builder=kw.pop("builder", _StubBuilder()), filter_fn=self._filter,
            **kw)

    def _rows(self, n):
        for _ in range(n):
            coords = np.ones((1, NW, 3, 9))
            inds = np.ones((1, NW, 3), bool)
            self.buf.append_state(SimpleNamespace(
                branches={"gb": SimpleNamespace(coords=coords, inds=inds)},
                log_like=-np.ones((1, NW))))

    def test_search_builds_at_a_new_stages_first_entry_then_per_new_row(self):
        hb_ = self._builder()
        self._rows(4)
        self.assertFalse(hb_.maybe_build("gb_search_seed"))
        # first entry of gb_search_1: no refit of this stage yet -> build now
        # from the rows already stored (the seed / source_search population)
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            self.assertTrue(hb_.maybe_build("gb_search_1"))
        self.assertFalse(hb_.maybe_build("gb_search_1"))   # no new rows
        self._rows(3)                                # one search cycle
        self.assertTrue(hb_.maybe_build("gb_search_1"))
        text = "\n".join(cm.output)
        self.assertIn("ON THE HEAD", text)
        self.assertIn("kept 2 of 3 clusters", text)
        k, man = sr.latest_complete_epoch(self.root)
        self.assertEqual((k, man["built_at_row"], man["built_on"]),
                         (1, 7, "head"))
        self.assertEqual(man["row_count"], "head states")
        self.assertEqual(sr.read_manifest(self.root, 0)["built_at_row"], 4)
        self.assertTrue(os.path.exists(
            os.path.join(sr.epoch_dir(self.root, 1), man["npz_filtered"])))
        self.assertEqual(self.calls, [0.95, 0.95])
        self.assertFalse(os.path.exists(
            os.path.join(sr.epoch_dir(self.root, 1), sr.LEAF_ROWS_H5)))

    def test_pe_builds_at_its_first_entry_then_every_25_rows(self):
        hb_ = self._builder()
        self._rows(1)
        self.assertTrue(hb_.maybe_build("full_pe"))    # PE first seen at row 1
        self._rows(24)
        self.assertFalse(hb_.maybe_build("full_pe"))   # 24 rows later
        self._rows(1)
        self.assertTrue(hb_.maybe_build("full_pe"))    # 25 rows later (row 26)
        self._rows(24)
        # a new process on the same disk: the store fill supplies the stage
        self.buf.stage_name, self.buf.stage_start = "full_pe", 1
        fresh = self._builder()
        self.assertFalse(fresh.maybe_build("full_pe"))  # 24 rows since the build
        self._rows(1)
        self.assertTrue(fresh.maybe_build("full_pe"))

    def test_a_resume_whose_last_build_is_past_the_stored_rows_builds_at_once(self):
        """The head counts every state it hands to the saver; the saver drops
        states when behind, so a resumed store can hold FEWER rows than the
        latest epoch's built_at_row. That epoch must not silence the clock."""
        hb_ = self._builder()
        self._rows(1)
        self.assertTrue(hb_.maybe_build("full_pe"))
        man_path = os.path.join(sr.epoch_dir(self.root, 0), sr.MANIFEST)
        with open(man_path) as fh:
            man = json.load(fh)
        man["built_at_row"] = 60                   # head count past the store
        with open(man_path, "w") as fh:
            json.dump(man, fh)
        self._rows(29)                             # the store holds 30 rows
        self.buf.stage_name, self.buf.stage_start = "full_pe", 1
        fresh = self._builder()
        self.assertTrue(fresh.maybe_build("full_pe"))

    def test_a_failing_build_returns_false_and_never_raises(self):
        hb_ = self._builder(builder=_StubBuilder(fail=True))
        self._rows(3)
        with self.assertLogs(sr.logger, level=logging.WARNING):
            self.assertFalse(hb_.maybe_build("gb_search_2"))
        self.assertIsNone(sr.latest_complete_epoch(self.root))


class SplitMixerTest(_TmpDir):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "run", "warmstart", "self")
        os.makedirs(self.root)
        orig_path = os.path.join(self.tmp, "orig.npz")
        _write_components(orig_path)
        self.orig_path = orig_path

    def _mixer(self, split, build_seed=1):
        orig = WarmStartComponents.from_npz(self.orig_path, seed=7, **LOAD_KW)
        return sr.WarmSelfMixer(orig, epoch_root=self.root, split=split,
                                load_kwargs=LOAD_KW, build_seed=build_seed,
                                tag=recipe._WARM_SEED_TAG_PE, name="rj_warm_pe")

    def _epoch(self, k, filtered=True):
        d = _write_epoch(self.root, k)
        if filtered:
            _write_components(os.path.join(d, sr.FILTERED_NPZ), shift=0.4)
            with open(os.path.join(d, sr.MANIFEST)) as fh:
                man = json.load(fh)
            man["npz_filtered"] = sr.FILTERED_NPZ
            with open(os.path.join(d, sr.MANIFEST), "w") as fh:
                json.dump(man, fh)

    def test_original_only_before_the_first_epoch(self):
        m = self._mixer((1, 1, 1))
        self.assertEqual([m.begin_propose()["choice"] for _ in range(6)],
                         ["orig"] * 6)

    def test_pe_draws_each_container_with_probability_one_third(self):
        """Mike, 2026-10-15: "just draw each of these with 1/3 probability
        each" -- a random draw per propose, not a fixed rotation."""
        m = self._mixer((1, 1, 1))
        self._epoch(0)
        got = [m.begin_propose()["choice"] for _ in range(3000)]
        for c in ("orig", "full", "filtered"):
            # binomial(3000, 1/3): sd 25.8; 5 sd
            self.assertLess(abs(got.count(c) - 1000), 130, c)
        # not a rotation: some propose repeats the previous container
        self.assertTrue(any(a == b for a, b in zip(got, got[1:])))

    def test_the_draws_are_seeded_per_build_seed(self):
        self._epoch(0)
        a, b, c = (self._mixer((1, 1, 1), build_seed=s) for s in (5, 5, 6))
        seq = [[m.begin_propose()["choice"] for _ in range(40)] for m in (a, b, c)]
        self.assertEqual(seq[0], seq[1])
        self.assertNotEqual(seq[0], seq[2])

    def test_search_uses_only_the_filtered_refit(self):
        m = self._mixer((0, 0, 1))
        self._epoch(0)
        self.assertEqual([m.begin_propose()["choice"] for _ in range(9)],
                         ["filtered"] * 9)

    def test_each_choice_serves_its_own_container_for_every_density(self):
        m = self._mixer((1, 1, 1))
        self._epoch(0)
        x = np.array([_mean(2.0)])
        seen = {}
        for _ in range(200):
            c = m.begin_propose()["choice"]
            seen[c] = m.active
            np.testing.assert_array_equal(m.logpdf(x), m.active.logpdf(x))
            if len(seen) == 3:
                break
        self.assertIs(seen["orig"], m.original)
        self.assertIs(seen["full"], m.self_full)
        self.assertIs(seen["filtered"], m.self_filtered)
        self.assertEqual(len({id(v) for v in seen.values()}), 3)

    def test_an_epoch_without_a_filtered_npz_serves_full_for_filtered(self):
        m = self._mixer((0, 0, 1))
        self._epoch(0, filtered=False)
        m.begin_propose()
        self.assertEqual(m.choice, "filtered")
        self.assertIs(m.active, m.self_full)

    def test_ranks_adopt_the_heads_epoch_and_choice(self):
        head = self._mixer((1, 1, 1), build_seed=1)
        ranks = [self._mixer((1, 1, 1), build_seed=s) for s in (2, 3)]
        self._epoch(0)
        for _ in range(6):
            d = head.begin_propose(rank=0)
            for r, m in enumerate(ranks, start=1):
                m.apply_directive(d, rank=r)
                self.assertEqual((m.epoch, m.choice), (head.epoch, head.choice))
                self.assertIsNotNone(m.self_filtered)

    def test_pickle_and_deepcopy(self):
        m = self._mixer((1, 1, 1))
        self._epoch(0)
        m.begin_propose()
        m.begin_propose()
        m2 = pickle.loads(pickle.dumps(copy.deepcopy(m)))
        self.assertEqual((m2.epoch, m2.choice), (m.epoch, m.choice))
        self.assertEqual(m2.begin_propose()["choice"], m.begin_propose()["choice"])

    def test_the_head_builds_at_entry_and_adopts_it_in_the_same_propose(self):
        buf = sr.WarmSelfBuffer(last_k=6)
        built = []

        class _Builder:
            epoch_root = self.root

            def maybe_build(inner, stage):
                built.append(stage)
                self._epoch(len(built) - 1)
                return True

        sr.register_head_builder(_Builder())
        self.addCleanup(sr.unregister_head_builder, self.root)
        m = self._mixer((0, 0, 1))
        # the pre-2026-10-16 behaviour; the default one-draw adoption lag is
        # pinned in tests/test_warmstart_self_refit_lag.py
        m.adopt_lag = False
        d = m.begin_propose(rank=0, stage="gb_search_1")
        self.assertEqual(built, ["gb_search_1"])
        self.assertEqual((d["epoch"], d["choice"]), (0, "filtered"))
        del buf

    def test_only_the_head_rank_builds(self):
        built = []
        sr.register_head_builder(SimpleNamespace(
            epoch_root=self.root, maybe_build=lambda stage: built.append(stage)))
        self.addCleanup(sr.unregister_head_builder, self.root)
        m = self._mixer((0, 0, 1))
        m.begin_propose(rank=2, stage="gb_search_1")
        self.assertEqual(built, [])
        m.begin_propose(rank=0, stage="gb_search_1")
        m.begin_propose(rank=None, stage="gb_search_1")
        self.assertEqual(built, ["gb_search_1"] * 2)

    def test_the_seed_and_other_non_refit_stages_draw_only_the_original(self):
        """Mike, 2026-10-15: "making sure gb search seed is fixed to the old
        fit" -- even with a self epoch on disk (a reused run directory), a
        stage that never refits draws every propose from the original."""
        m = self._mixer((0, 0, 1))
        self._epoch(0)
        for stage in ("gb_search_seed", "source_search", "noise_search"):
            d = m.begin_propose(rank=0, stage=stage)
            self.assertEqual(d["choice"], "orig", stage)
            self.assertIs(m.active, m.original)
        self.assertEqual(m.begin_propose(rank=0, stage="gb_search_1")["choice"],
                         "filtered")

    def test_without_a_registered_builder_nothing_is_built(self):
        self.assertIsNone(sr.head_builder_for(self.root))
        m = self._mixer((0, 0, 1))
        self.assertEqual(m.begin_propose(rank=1, stage="gb_search_1"),
                         {"epoch": None, "choice": "orig"})


class SubsetWriterTest(unittest.TestCase):
    """The filtered npz keeps the production (observable / GMM) schema."""

    def _gmm_data(self):
        ncomp = np.array([2, 1, 3])
        n = int(ncomp.sum())
        return dict(
            p=np.array([1.0, 0.5, 0.25]), mult=np.ones(3),
            n_members=np.array([30, 20, 10]), island_id=np.array([0, 1, 1]),
            blend=np.array([False, False, True]),
            f0_window_edges=np.array([[1.0, 1.1], [2.0, 2.1]]),
            gmm_ncomp=ncomp, gmm_weights=np.arange(n, dtype=float),
            gmm_means=np.arange(n * 2.0).reshape(n, 2),
            meta=np.asarray(json.dumps({"basis": "observable"})))

    def test_masks_per_cluster_and_per_component_arrays(self):
        out = sr._subset_clusters(self._gmm_data(), np.array([False, True, True]),
                                  np.array([2, 1, 3]))
        np.testing.assert_array_equal(out["gmm_ncomp"], [1, 3])
        np.testing.assert_array_equal(out["gmm_weights"], [2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(out["gmm_means"][:, 0], [4.0, 6.0, 8.0, 10.0])
        np.testing.assert_array_equal(out["p"], [0.5, 0.25])
        np.testing.assert_array_equal(out["blend"], [False, True])
        np.testing.assert_array_equal(out["f0_window_edges"],
                                      [[1.0, 1.1], [2.0, 2.1]])

    def test_an_unknown_layout_is_refused(self):
        d = self._gmm_data()
        d["odd"] = np.zeros(5)
        with self.assertRaises(ValueError):
            sr._subset_clusters(d, np.array([True, True, True]),
                                np.array([2, 1, 3]))


class HeadWiringTest(_TmpDir):
    """The head's save, its set-up and the move's entry feed the builder."""

    def test_the_heads_save_appends_the_state_it_saves(self):
        from lisatools.globalfit import hdfbackend as hb

        buf = sr.WarmSelfBuffer(last_k=4)
        fake = SimpleNamespace(
            _has_saver_rank=lambda: True, save_plot_rank=4,
            comm=SimpleNamespace(send=lambda *a, **k: None),
            filename=os.path.join(self.tmp, "none.h5"), warm_self_buffer=buf)
        st = SimpleNamespace(
            branches={"gb": SimpleNamespace(coords=np.ones((1, NW, 3, 9)),
                                            inds=np.ones((1, NW, 3), bool))},
            log_like=-np.ones((1, NW)))
        for _ in range(3):
            hb.GFHDFBackend.save_step(fake, st, np.zeros((1, NW)))
        self.assertEqual((buf.n_stored, len(buf)), (3, 3))

    def _curr(self, store, **gb_kw):
        gb = SimpleNamespace(warm_self=True, warm_self_at_entry=True,
                             warm_start_components="/x/w.npz",
                             warm_self_last_k=6, warm_self_every_search=1,
                             warm_self_every_pe=25, warm_self_consensus=0.9)
        for k, v in gb_kw.items():
            setattr(gb, k, v)
        gi = SimpleNamespace(main_file_path=store, Tobs=TOBS, gpus=[1])
        return SimpleNamespace(source_info={"gb": gb}, general_info=gi)

    def test_the_head_builder_fills_from_the_store_and_registers(self):
        run = os.path.join(self.tmp, "run")
        os.makedirs(run)
        store = os.path.join(run, "gf_testing.h5")
        _make_store(store)
        with h5py.File(store, "a") as f:
            f["global_fit"].attrs["iteration"] = 20
        backend = SimpleNamespace(name="global_fit",
                                  open=lambda mode="r": h5py.File(store, mode))
        b = sr.make_head_builder(self._curr(store), backend)
        self.addCleanup(sr.unregister_head_builder, sr.self_epoch_root(store))
        self.assertIs(sr.head_builder_for(sr.self_epoch_root(store)), b)
        self.assertIs(backend.warm_self_buffer, b.buffer)
        self.assertEqual((b.buffer.n_stored, len(b.buffer)), (20, 6))
        self.assertEqual((b.buffer.stage_name, b.buffer.stage_start),
                         ("gb_search_2", 10))
        self.assertEqual((b.device, b.consensus, b.every_pe), (1, 0.9, 25))
        for off in (dict(warm_self_at_entry=False), dict(warm_self=False),
                    dict(warm_start_components="")):
            with mock.patch.dict(os.environ, {"GB_WARM_START_COMPONENTS": ""}):
                self.assertIsNone(sr.make_head_builder(self._curr(store, **off),
                                                       backend), off)

    def test_the_fill_reads_through_the_real_store_backend(self):
        """The resume fill as prepare_main runs it: a real GFHDFBackend on
        the store, its own open("r"), and the store writable right after."""
        from lisatools.globalfit import hdfbackend as hb

        run = os.path.join(self.tmp, "run")
        os.makedirs(run)
        store = os.path.join(run, "gf_testing.h5")
        _make_store(store)
        with h5py.File(store, "a") as f:
            f["global_fit"].attrs["iteration"] = 20
        backend = hb.GFHDFBackend(store)
        with self.assertNoLogs(sr.logger, level=logging.WARNING):
            b = sr.make_head_builder(self._curr(store), backend)
        self.addCleanup(sr.unregister_head_builder, sr.self_epoch_root(store))
        self.assertEqual((b.buffer.n_stored, len(b.buffer)), (20, 6))
        self.assertIs(backend.warm_self_buffer, b.buffer)
        with h5py.File(store, "a") as f:            # no handle left behind
            f["global_fit"].attrs["probe"] = 1

    def test_the_stage_combine_hands_the_live_stage_down_to_the_builder(self):
        """A move shared by two stages carries a STALE static stamp; the
        stage combine re-stamps it before every propose, and that live name
        is what reaches the head builder (rj_warm_pe under full_pe)."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        root = os.path.join(self.tmp, "self")
        os.makedirs(root)
        seen = []
        sr.register_head_builder(SimpleNamespace(
            epoch_root=root, maybe_build=lambda stage: seen.append(stage)))
        self.addCleanup(sr.unregister_head_builder, root)
        orig_path = os.path.join(self.tmp, "orig.npz")
        _write_components(orig_path)
        mixer = sr.WarmSelfMixer(
            WarmStartComponents.from_npz(orig_path, seed=1, **LOAD_KW),
            epoch_root=root, split=(1, 1, 1), load_kwargs=LOAD_KW)
        m = gbs.GBSpecialRJPriorMove.__new__(gbs.GBSpecialRJPriorMove)
        m.name, m.branch_name, m.gf_rank = "rj_warm_pe", "gb", 0
        m.rj_proposal_distribution = {"gb": mixer}
        m.gf_stage_name = "gb_search_2"                # the static, stale stamp
        combine = GFCombineMove.__new__(GFCombineMove)
        combine.gf_stage_name, combine.gf_stage_kind = "full_pe", "pe"
        combine.moves = []
        combine._gf_precondition(m, SimpleNamespace())
        m._warm_self_begin()
        self.assertEqual(seen, ["full_pe"])

    def test_the_move_hands_its_live_stage_to_the_head_builder(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        root = os.path.join(self.tmp, "self")
        os.makedirs(root)
        seen = []
        sr.register_head_builder(SimpleNamespace(
            epoch_root=root, maybe_build=lambda stage: seen.append(stage)))
        self.addCleanup(sr.unregister_head_builder, root)
        orig_path = os.path.join(self.tmp, "orig.npz")
        _write_components(orig_path)
        mixer = sr.WarmSelfMixer(
            WarmStartComponents.from_npz(orig_path, seed=1, **LOAD_KW),
            epoch_root=root, split=(0, 0, 1), load_kwargs=LOAD_KW)
        m = gbs.GBSpecialRJPriorMove.__new__(gbs.GBSpecialRJPriorMove)
        m.name, m.branch_name, m.gf_rank = "rj_warm_search", "gb", 0
        m.rj_proposal_distribution = {"gb": mixer}
        m.gf_stage_name = "gb_search_2"
        m._warm_self_begin()
        self.assertEqual(seen, ["gb_search_2"])

    def test_prepare_main_arms_the_head_builder(self):
        import inspect

        from lisatools.globalfit import run as gf_run

        src = inspect.getsource(gf_run.GlobalFit.prepare_main)
        i_arm = src.index("make_head_builder(self.curr, backend)")
        i_engine = src.index("GlobalFitEngine(")
        self.assertLess(i_arm, i_engine)


class WrapSplitTest(_TmpDir):
    def test_each_twin_takes_its_split(self):
        orig_path = os.path.join(self.tmp, "orig.npz")
        _write_components(orig_path)
        orig = WarmStartComponents.from_npz(orig_path, seed=1, **LOAD_KW)
        store = os.path.join(self.tmp, "run", "gf_testing.h5")
        gb_info = SimpleNamespace(warm_self=True, warm_self_split_search="0,0,1",
                                  warm_self_split_pe="1,1,1", build_seed=99)
        kw = dict(gb_info=gb_info,
                  general_info=SimpleNamespace(main_file_path=store),
                  load_kwargs=dict(LOAD_KW))
        search = recipe._warm_self_wrap(orig, tag=recipe._WARM_SEED_TAG_SEARCH,
                                        name="rj_warm_search", **kw)
        pe = recipe._warm_self_wrap(orig, tag=recipe._WARM_SEED_TAG_PE,
                                    name="rj_warm_pe", **kw)
        self.assertEqual(search.split, (0.0, 0.0, 1.0))
        self.assertEqual(pe.split, (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
