"""Pruning of old GB warm-start SELF refit epochs (warmstart/self_refit.py).

Ruling (Mike, 2026-10-14): "remove like all except the last few ... or keep
every 10 refits ... so we can see the difference over time." Each epoch keeps
~40-70 MB of npz files and nothing deleted them, so a refit every row would
grow a run directory by ~10 GB a day. Kept in full: the newest 3 complete
epochs and every 10th; every other older epoch is stripped to its DONE.json
(the build history stays readable, the epoch stops being adoptable).
"""
import json
import logging
import os
import unittest

from lisatools.globalfit.warmstart import self_refit as sr
from tests.test_warmstart_self_refit import (
    TOBS,
    _Comm,
    _make_store,
    _Reader,
    _set_rows,
    _StubBuilder,
    _mixer,
    _TmpDir,
    _write_epoch,
)

EXTRA = (sr.FIT_NPZ, "self_fit_referee.npz")


def _files(root, k):
    return sorted(os.listdir(sr.epoch_dir(root, k)))


class PruneEpochsTest(_TmpDir):

    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "self")
        for k in range(25):                        # complete 0..24, 5 crashed
            d = _write_epoch(self.root, k, complete=(k != 5))
            for name in EXTRA:
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(b"x" * 1000)
        _write_epoch(self.root, 25, complete=False)  # a build in progress

    def test_keeps_the_last_three_and_every_tenth_in_full(self):
        full = sorted([sr.MANIFEST, sr.REFEREED_NPZ, *EXTRA])
        stripped = [k for k in range(25) if k not in (0, 10, 20, 22, 23, 24)]
        want_freed = sum(
            os.path.getsize(os.path.join(sr.epoch_dir(self.root, k), n))
            for k in stripped for n in _files(self.root, k) if n != sr.MANIFEST)
        n, freed = sr.prune_epochs(self.root, keep_last=3, keep_every=10)
        self.assertEqual((n, freed), (len(stripped), want_freed))
        for k in (0, 10, 20, 22, 23, 24):
            self.assertEqual(_files(self.root, k), full, k)
        for k in stripped:
            want = [] if k == 5 else [sr.MANIFEST]
            self.assertEqual(_files(self.root, k), want, k)
        # the in-progress build newer than the newest complete is untouched
        self.assertEqual(_files(self.root, 25), [sr.REFEREED_NPZ])

    def test_history_stays_readable_and_only_kept_epochs_are_adoptable(self):
        sr.prune_epochs(self.root, keep_last=3, keep_every=10)
        self.assertEqual(sr.latest_complete_epoch(self.root)[0], 24)
        self.assertIsNone(sr.read_manifest(self.root, 7))
        self.assertIsNotNone(sr.read_manifest(self.root, 10))
        with open(os.path.join(sr.epoch_dir(self.root, 7), sr.MANIFEST)) as fh:
            self.assertEqual(json.load(fh)["built_at_row"], 70)
        self.assertEqual(sr.next_epoch_index(self.root), 26)

    def test_a_second_pass_frees_nothing(self):
        sr.prune_epochs(self.root, keep_last=3, keep_every=10)
        self.assertEqual(sr.prune_epochs(self.root, keep_last=3, keep_every=10),
                         (0, 0))


class PrunedEpochAdoptionTest(_TmpDir):
    """A compute rank told to adopt an epoch that has since been pruned keeps
    what it holds (or the original) with a warning, and never raises."""

    def test_a_rank_told_to_adopt_a_pruned_epoch_keeps_what_it_holds(self):
        root = os.path.join(self.tmp, "run", "warmstart", "self")
        os.makedirs(root)
        for k in range(5):
            _write_epoch(root, k)
        held = _mixer(root)
        held.apply_directive({"epoch": 0, "choice": "self"}, rank=2)
        empty = _mixer(root)
        sr.prune_epochs(root, keep_last=3, keep_every=10)   # strips epoch 1
        self.assertIsNone(sr.read_manifest(root, 1))
        for m, want in ((held, (0, "full")), (empty, (None, "orig"))):
            with self.assertLogs(sr.logger, level=logging.WARNING):
                m.apply_directive({"epoch": 1, "choice": "self"}, rank=3)
            self.assertEqual((m.epoch, m.choice), want)


class HookPruneTest(_TmpDir):

    def test_the_saver_prunes_after_each_successful_build(self):
        run = os.path.join(self.tmp, "run")
        os.makedirs(run)
        store = os.path.join(run, "gf_testing.h5")
        _make_store(store)
        root = sr.self_epoch_root(store)
        hook = sr.WarmSelfHook(epoch_root=root, tobs=TOBS, last_k=6,
                               every_search=1, every_pe=250,
                               builder=_StubBuilder())
        reader = _Reader(store)
        with self.assertLogs(sr.logger, level=logging.INFO) as cm:
            for rows in range(11, 26):             # 15 builds: epochs 0..14
                _set_rows(store, rows)
                hook(reader, _Comm(), 0, rows)
        self.assertEqual(sr.list_epochs(root), list(range(15)))
        for k in range(15):
            want = ([sr.MANIFEST, sr.REFEREED_NPZ] if k in (0, 10, 12, 13, 14)
                    else [sr.MANIFEST])
            self.assertEqual(_files(root, k), want, k)
        self.assertTrue(any("[GB_WARM_SELF] pruned" in ln for ln in cm.output))


if __name__ == "__main__":
    unittest.main()
