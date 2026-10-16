"""The ONE-DRAW ADOPTION LAG of the head's entry-time GB warm-start self refit
(``warmstart/self_refit.py``; ``GB_WARM_SELF_ADOPT_LAG``, default on).

RULING (Mike, 2026-10-16): "Let's do this, but carefully, let's refit on the
current data holding our previous within-run refit information, use the
previous information on the current proposal and then switch it out." On the
9mo job 767 every iteration whose warm move drew from an epoch the head had
just built at that same entry lost 15-20 GB leaves and 100-527 nats per walker
(a fresh fit sits exactly on the current cold leaves, so the RJ death ratio's
reverse-birth term is large for the weakest leaves and they die together),
while draws on epochs >= 13 rows old were the job's best iterations.

Pinned here (CPU, KB-sized fixtures, the fit / referee / filter stubbed):

* a build at the warm entry with a previous epoch present: the directive and
  the active container are the PREVIOUS epoch's; the build line says so;
* the next draw (no build) adopts the fresh epoch;
* the run's first build: the original warm start this propose, then the fresh
  epoch at the next draw;
* consecutive build draws (search cadence 1): each uses the one before, also
  across the search -> full_pe twin hand-over;
* a compute rank adopts exactly the shipped epoch, never a newer one on disk;
* ``adopt_lag=False`` restores immediate adoption;
* the held epoch (and the epoch in use) survive the prune;
* the GBSettings default / env resolution and the plumbing into the mixer and
  the head builder.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import pickle
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit import recipe
from lisatools.globalfit.warmstart import self_refit as sr
from lisatools.globalfit.warmstart.proposal import WarmStartComponents
from tests.test_warmstart_self_refit import (
    LOAD_KW,
    TOBS,
    _components,
    _StubBuilder,
    _TmpDir,
    _write_components,
    _write_epoch,
)

NW = 4
#: the PRE-EXISTING epoch's law (written by _write_epoch), clearly unlike the
#: stub builder's (shift 0) and the stub filter's (shift 0.3)
OLD_SHIFT = 0.5


class _LagCase(_TmpDir):
    """A real head builder (fit / referee / filter stubbed) on a tmp epoch root."""

    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "run", "warmstart", "self")
        os.makedirs(self.root)
        self.buf = sr.WarmSelfBuffer(last_k=6)
        self.orig_path = os.path.join(self.tmp, "orig.npz")
        _write_components(self.orig_path, shift=-0.5)
        self.x = np.array(_components(OLD_SHIFT)[0])

    # -- fixtures ---------------------------------------------------------------
    def _filter(self, refereed, leaf, out, *, threshold):
        _write_components(out, shift=0.3)
        return dict(n_clusters=3, n_kept=2, n_components_kept=2,
                    consensus=np.array([True, False, False]))

    def _builder(self, **kw):
        b = sr.WarmSelfHeadBuilder(
            epoch_root=self.root, buffer=self.buf, tobs=TOBS, last_k=6,
            every_search=1, every_pe=25, device=None, consensus=0.95,
            builder=_StubBuilder(), filter_fn=self._filter, **kw)
        sr.register_head_builder(b)
        self.addCleanup(sr.unregister_head_builder, self.root)
        return b

    def _mixer(self, split=(0, 1, 0), build_seed=1, name="rj_warm_search", **kw):
        orig = WarmStartComponents.from_npz(self.orig_path, seed=7, **LOAD_KW)
        return sr.WarmSelfMixer(orig, epoch_root=self.root, split=split,
                                load_kwargs=LOAD_KW, build_seed=build_seed,
                                tag=recipe._WARM_SEED_TAG_SEARCH, name=name,
                                **kw)

    def _rows(self, n):
        for _ in range(n):
            self.buf.append_state(SimpleNamespace(
                branches={"gb": SimpleNamespace(coords=np.ones((1, NW, 3, 9)),
                                                inds=np.ones((1, NW, 3), bool))},
                log_like=-np.ones((1, NW))))

    def _law(self, k, filtered=False):
        """Epoch ``k``'s container as a rank would load it (densities only)."""
        man = sr.read_manifest(self.root, k)
        self.assertIsNotNone(man, k)
        return WarmStartComponents.from_npz(
            man["npz_filtered_path" if filtered else "npz_path"], seed=1,
            **LOAD_KW)

    def assertSameLaw(self, cont, law):
        np.testing.assert_array_equal(cont.logpdf(self.x), law.logpdf(self.x))

    def assertNotSameLaw(self, cont, law):
        self.assertFalse(np.allclose(cont.logpdf(self.x), law.logpdf(self.x)))


class BuildProposeTest(_LagCase):
    def test_a_build_propose_draws_from_the_previous_epoch(self):
        """(i) epoch 0 on disk; this entry builds epoch 1 and draws from 0."""
        _write_epoch(self.root, 0, shift=OLD_SHIFT, built_at_row=0)
        self._builder()
        self._rows(3)
        m = self._mixer()                         # the constructor default lag
        self.assertTrue(m.adopt_lag)
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            d = m.begin_propose(rank=0, stage="gb_search_2")
            m.note_census({}, move_name="rj_warm_search", rank=0)
        self.assertIsNotNone(sr.read_manifest(self.root, 1))   # 1 was built
        self.assertEqual(d, {"epoch": 0, "choice": "full"})
        self.assertEqual((m.epoch, m.held_epoch), (0, 1))
        self.assertIs(m.active, m.self_full)
        self.assertSameLaw(m.active, self._law(0))
        self.assertNotSameLaw(m.active, self._law(1))
        text = "\n".join(cm.output)
        self.assertIn("epoch 1 built ON THE HEAD at the gb_search_2 warm entry",
                      text)
        self.assertIn("every rank waited for it; held for the next warm draw, "
                      "this propose uses epoch 0.", text)
        self.assertIn("this propose used FULL (epoch 0)", text)
        self.assertNotIn("adopted epoch 1", text)

    def test_the_next_draw_adopts_the_held_epoch(self):
        """(ii) the next begin_propose (no build due) switches to epoch 1."""
        _write_epoch(self.root, 0, shift=OLD_SHIFT, built_at_row=0)
        self._builder()
        self._rows(3)
        m = self._mixer()
        m.begin_propose(rank=0, stage="gb_search_2")
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            d = m.begin_propose(rank=0, stage="gb_search_2")   # no new row
        self.assertEqual(sr.latest_complete_epoch(self.root)[0], 1)  # no build
        self.assertEqual(d, {"epoch": 1, "choice": "full"})
        self.assertEqual((m.epoch, m.held_epoch), (1, None))
        self.assertSameLaw(m.active, self._law(1))
        self.assertIn("rank 0 adopted epoch 1", "\n".join(cm.output))

    def test_the_runs_first_build_draws_the_original_then_the_fresh_epoch(self):
        """(iii) no previous epoch: ORIG this propose, epoch 0 next draw."""
        self._builder()
        self._rows(3)
        m = self._mixer(split=(0, 0, 1))
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            d = m.begin_propose(rank=0, stage="gb_search_1")
        self.assertIsNotNone(sr.read_manifest(self.root, 0))
        self.assertEqual(d, {"epoch": None, "choice": "orig"})
        self.assertEqual(m.held_epoch, 0)
        self.assertIs(m.active, m.original)
        self.assertEqual(m.n_self_window, 0)
        self.assertIn("held for the next warm draw, this propose uses the "
                      "original warm start.", "\n".join(cm.output))
        d = m.begin_propose(rank=0, stage="gb_search_1")
        self.assertEqual(d, {"epoch": 0, "choice": "filtered"})
        self.assertIs(m.active, m.self_filtered)
        self.assertSameLaw(m.active, self._law(0, filtered=True))

    def test_consecutive_build_draws_each_use_the_one_before(self):
        """(iv) search cadence 1, one stored row per cycle: every draw builds."""
        self._builder()
        m = self._mixer(split=(0, 0, 1))
        used, held = [], []
        for _ in range(5):
            self._rows(1)
            used.append(m.begin_propose(rank=0, stage="gb_search_2")["epoch"])
            held.append(m.held_epoch)
        self.assertEqual(used, [None, 0, 1, 2, 3])
        self.assertEqual(held, [0, 1, 2, 3, 4])
        self.assertEqual(sr.latest_complete_epoch(self.root)[0], 4)

    def test_the_pe_twins_first_entry_uses_the_last_search_refit(self):
        """The next warm draw may be the OTHER twin's in another stage: the PE
        twin's first full_pe entry builds again and uses the search's last."""
        self._builder()
        search = self._mixer(split=(0, 0, 1))
        pe = self._mixer(split=(0, 1, 0), name="rj_warm_pe")
        self._rows(2)
        search.begin_propose(rank=0, stage="gb_search_2")        # builds 0
        self._rows(1)
        self.assertEqual(search.begin_propose(rank=0, stage="gb_search_2"),
                         {"epoch": 0, "choice": "filtered"})      # builds 1
        self._rows(1)
        d = pe.begin_propose(rank=0, stage="full_pe")             # builds 2
        self.assertEqual(d, {"epoch": 1, "choice": "full"})
        self.assertEqual(pe.held_epoch, 2)
        self.assertEqual(sr.read_manifest(self.root, 2)["stage"], "full_pe")
        d = pe.begin_propose(rank=0, stage="full_pe")             # 0 rows later
        self.assertEqual(d, {"epoch": 2, "choice": "full"})

    def test_an_epoch_already_on_disk_is_adopted_at_once(self):
        """Only an epoch built INSIDE the call is held: a resume's (or a saver-
        built) epoch is adopted at the first draw, build or no build."""
        _write_epoch(self.root, 3, shift=OLD_SHIFT, built_at_row=0)
        m = self._mixer()
        self.assertEqual(m.begin_propose(rank=0, stage="gb_search_2"),
                         {"epoch": 3, "choice": "full"})
        self.assertIsNone(m.held_epoch)
        self.assertSameLaw(m.active, self._law(3))


class ComputeRankTest(_LagCase):
    def test_a_rank_adopts_exactly_the_shipped_epoch(self):
        """(v) the head ships epoch 0 while epoch 1 already sits complete on
        disk: every rank loads 0, without ever scanning the epoch root."""
        _write_epoch(self.root, 0, shift=OLD_SHIFT, built_at_row=0)
        self._builder()
        self._rows(3)
        head = self._mixer(build_seed=1)
        ranks = [self._mixer(build_seed=s) for s in (2, 3, 4)]
        d = head.begin_propose(rank=0, stage="gb_search_2")
        self.assertIsNotNone(sr.latest_complete_epoch(self.root, newer_than=0))
        with mock.patch.object(sr, "latest_complete_epoch",
                               side_effect=AssertionError("a rank polled")), \
                mock.patch.object(sr, "list_epochs",
                                  side_effect=AssertionError("a rank listed")):
            for r, m in enumerate(ranks, start=1):
                m.apply_directive(d, rank=r)
                self.assertEqual((m.epoch, m.choice), (0, "full"))
                self.assertSameLaw(m.active, self._law(0))
        # the next draw: the head adopts 1 and the ranks follow
        d = head.begin_propose(rank=0, stage="gb_search_2")
        self.assertEqual(d["epoch"], 1)
        for r, m in enumerate(ranks, start=1):
            with self.assertLogs(sr.logger, level=logging.INFO) as cm:
                m.apply_directive(d, rank=r)
            self.assertEqual(m.epoch, 1)
            self.assertIn(f"rank {r} adopted epoch 1", "\n".join(cm.output))

    def test_the_directive_is_authoritative_both_ways(self):
        _write_epoch(self.root, 0, shift=OLD_SHIFT)
        _write_epoch(self.root, 1)
        r = self._mixer(build_seed=2)
        r.apply_directive({"epoch": 1, "choice": "full"}, rank=1)
        self.assertEqual(r.epoch, 1)
        # told an OLDER epoch than it holds: it follows the head
        r.apply_directive({"epoch": 0, "choice": "full"}, rank=1)
        self.assertEqual(r.epoch, 0)
        self.assertSameLaw(r.active, self._law(0))
        # no epoch on the directive (the head holds no self refit): the
        # original, whatever the rank holds
        r.apply_directive({"epoch": None, "choice": "full"}, rank=1)
        self.assertEqual(r.choice, "orig")
        self.assertIs(r.active, r.original)

    def test_the_move_hooks_carry_the_lagged_epoch(self):
        """Through the GB move's own hooks: ``_warm_self_begin`` on the head
        (live stage), ``_warm_self_apply`` on a rank, the census line."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        _write_epoch(self.root, 0, shift=OLD_SHIFT, built_at_row=0)
        self._builder()
        self._rows(3)

        def move(mixer, rank):
            mv = gbs.GBSpecialRJPriorMove.__new__(gbs.GBSpecialRJPriorMove)
            mv.name, mv.branch_name, mv.gf_rank = "rj_warm_search", "gb", rank
            mv.rj_proposal_distribution = {"gb": mixer}
            mv.gf_stage_name = "gb_search_2"
            return mv

        head = move(self._mixer(build_seed=1), 0)
        rank = move(self._mixer(build_seed=2), 1)
        head._warm_self_begin()
        self.assertEqual(head._warm_self_directive,
                         {"epoch": 0, "choice": "full"})
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            rank._warm_self_apply(head._warm_self_directive)
            rank._warm_self_note({})
        text = "\n".join(cm.output)
        self.assertIn("rank 1 adopted epoch 0", text)
        self.assertIn("[GB_WARM_SELF rj_warm_search] r1 this propose used FULL "
                      "(epoch 0)", text)


class NoLagTest(_LagCase):
    def test_adopt_lag_false_adopts_the_fresh_epoch_at_once(self):
        """(vi) the pre-2026-10-16 behaviour, on request."""
        _write_epoch(self.root, 0, shift=OLD_SHIFT, built_at_row=0)
        self._builder(adopt_lag=False)
        self._rows(3)
        m = self._mixer(adopt_lag=False)
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            d = m.begin_propose(rank=0, stage="gb_search_2")
        self.assertEqual(d, {"epoch": 1, "choice": "full"})
        self.assertEqual((m.epoch, m.held_epoch), (1, None))
        self.assertSameLaw(m.active, self._law(1))
        text = "\n".join(cm.output)
        self.assertIn("epoch 1 built ON THE HEAD", text)
        self.assertNotIn("held for the next warm draw", text)
        # the first build of a run (a fresh root): the new epoch at once, not
        # the original
        self.root = os.path.join(self.tmp, "run2", "warmstart", "self")
        os.makedirs(self.root)
        self._builder(adopt_lag=False)
        self._rows(1)
        m = self._mixer(adopt_lag=False)
        self.assertEqual(m.begin_propose(rank=0, stage="gb_search_1"),
                         {"epoch": 0, "choice": "full"})


class PruneTest(_LagCase):
    def _complete(self, k):
        return sr.read_manifest(self.root, k) is not None

    def test_the_held_and_the_used_epochs_survive_the_prune(self):
        """(vii) every build prunes; the held epoch (the newest) and the one
        this propose ships (the newest but one, loaded by the ranks AFTER the
        prune) stay whole even at keep_last=1; older ones are stripped."""
        self._builder(keep_last=1, keep_every=0)
        head = self._mixer(split=(0, 0, 1), build_seed=1)
        late = self._mixer(split=(0, 0, 1), build_seed=9)  # loads only now
        for i in range(6):
            self._rows(1)
            d = head.begin_propose(rank=0, stage="gb_search_2")
            k, j = head.held_epoch, d["epoch"]
            self.assertEqual(k, i)
            self.assertTrue(self._complete(k), (i, "held"))
            if j is not None:
                self.assertTrue(self._complete(j), (i, "used"))
                with self.assertNoLogs(sr.logger, level=logging.WARNING):
                    late.apply_directive(d, rank=3)
                self.assertEqual(late.epoch, j)
            for old in range(0, i - 1):
                self.assertFalse(self._complete(old), (i, old))
                self.assertEqual(os.listdir(sr.epoch_dir(self.root, old)),
                                 [sr.MANIFEST])
        # the next draw adopts the held one, still whole
        self.assertEqual(head.begin_propose(rank=0, stage="gb_search_2")["epoch"],
                         5)

    def test_a_default_prune_after_a_held_build_keeps_the_held_epoch(self):
        for k in range(4):
            _write_epoch(self.root, k, built_at_row=0)
        self._builder()
        self._rows(3)
        m = self._mixer()
        self.assertEqual(m.begin_propose(rank=0, stage="gb_search_2")["epoch"], 3)
        self.assertEqual(m.held_epoch, 4)
        sr.prune_epochs(self.root)                   # the defaults (3 / 10)
        self.assertTrue(self._complete(4))
        self.assertTrue(self._complete(3))
        self.assertEqual(m.begin_propose(rank=0, stage="gb_search_2")["epoch"], 4)


class KnobTest(_TmpDir):
    def test_gbsettings_default_and_env(self):
        """(viii) code default ON; GB_WARM_SELF_ADOPT_LAG=0 restores immediate
        adoption."""
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        f = {x.name: x for x in dataclasses.fields(GBSettings)}
        self.assertIn("warm_self_adopt_lag", f)
        fac = f["warm_self_adopt_lag"].default_factory
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_WARM_SELF_ADOPT_LAG", None)
            self.assertIs(fac(), True)
            for raw, want in (("0", False), ("1", True), ("false", False)):
                os.environ["GB_WARM_SELF_ADOPT_LAG"] = raw
                self.assertIs(fac(), want, raw)

    def test_the_wrap_and_the_head_builder_take_the_knob(self):
        orig_path = os.path.join(self.tmp, "orig.npz")
        _write_components(orig_path)
        orig = WarmStartComponents.from_npz(orig_path, seed=1, **LOAD_KW)
        store = os.path.join(self.tmp, "run", "gf_testing.h5")
        kw = dict(general_info=SimpleNamespace(main_file_path=store),
                  load_kwargs=dict(LOAD_KW), tag=recipe._WARM_SEED_TAG_PE,
                  name="rj_warm_pe")
        for attrs, want in (({}, True), ({"warm_self_adopt_lag": False}, False),
                            ({"warm_self_adopt_lag": True}, True)):
            gb_info = SimpleNamespace(warm_self=True, build_seed=5, **attrs)
            m = recipe._warm_self_wrap(orig, gb_info=gb_info, **kw)
            self.assertIs(m.adopt_lag, want, attrs)
            m2 = pickle.loads(pickle.dumps(copy.deepcopy(m)))
            self.assertEqual((m2.adopt_lag, m2.held_epoch), (want, None))
        for attrs, want in (({}, True), ({"warm_self_adopt_lag": False}, False)):
            gb = SimpleNamespace(warm_self=True, warm_self_at_entry=True,
                                 warm_start_components="/x/w.npz",
                                 warm_self_last_k=6, warm_self_every_search=1,
                                 warm_self_every_pe=25, **attrs)
            curr = SimpleNamespace(
                source_info={"gb": gb},
                general_info=SimpleNamespace(main_file_path=store, Tobs=TOBS,
                                             gpus=[]))
            b = sr.make_head_builder(curr, None)
            self.addCleanup(sr.unregister_head_builder, sr.self_epoch_root(store))
            self.assertIs(b.adopt_lag, want, attrs)


if __name__ == "__main__":
    unittest.main()
