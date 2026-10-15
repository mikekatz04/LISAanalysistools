"""Merge-by-owned-source for one-walker GB replicas, and the ``gb_sync`` command.

Skeleton-level, like ``tests/test_gb_rank_session.py``: no GB kernel, no real
``BandSorter``, no GPU. What is under test is (a) the head's merge of the
finish replies BY PHYSICAL SOURCE -- leaf slots are not stable across the
ranks, because ``_write_back_state`` renumbers leaves densely in frequency
order, so one rank's birth shifts its stale copies of the other ranks'
sources -- (b) the head's ``gb_sync`` bookkeeping, and (c) the fourth fan-out
command itself, which rebuilds every replica's residual from the MERGED branch.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.gbspecialstretch import merge_owned_sources

NT, NL, ND = 2, 6, 2
#: the sort column ``_write_back_state`` ranks live sources by
F0 = 1


def _reply(coords, inds, bands, d_h=None, h_h=None):
    """A `gb_finish` reply over ONE walker: (NT, 1, NL[, ND]) arrays."""
    return {
        "block_coords": np.asarray(coords, float),
        "block_inds": np.asarray(inds, bool),
        "block_band_inds": np.asarray(bands, np.int64),
        "d_h": np.zeros((1, NL)) if d_h is None else np.asarray(d_h, float),
        "h_h": np.zeros((1, NL)) if h_h is None else np.asarray(h_h, float),
    }


def _blank(fill=-99.0):
    return (np.full((NT, 1, NL, ND), fill), np.zeros((NT, 1, NL), bool),
            np.full((NT, 1, NL), -1, np.int64))


def _lay(rows, f0s, bands):
    """Dense (rows, …) branch arrays from a list of (f0, band) sources."""
    c, i, b = _blank()
    for slot, (f0, band) in enumerate(zip(f0s, bands)):
        for t in range(NT):
            c[t, 0, slot] = [float(band), float(f0)]
            i[t, 0, slot] = True
            b[t, 0, slot] = int(band)
    return c, i, b


class MergeOwnedSourcesTest(unittest.TestCase):
    """Two replicas, interleaved band ranges, a birth on one and a death on
    the other -- the SLOT-shifting case a slot-keyed merge got wrong."""

    def setUp(self):
        # the head's pre-propose cold/hot branch: four sources, one per band,
        # at leaves 0..3 in frequency order
        self.work_c, self.work_i, _ = _lay(4, [1.5, 3.5, 5.5, 7.5], [0, 1, 2, 3])
        self.d_h = np.full((1, NL), np.nan)
        self.h_h = np.full((1, NL), np.nan)
        # RANK 0 owns bands [0, 2). It accepted a BIRTH in band 1 (f0 = 3.0),
        # so its dense repack shifts EVERY higher leaf by one -- its copies of
        # rank 1's band-2/3 sources now sit at leaves 3 and 4, not 2 and 3.
        c0, i0, b0 = _lay(5, [1.5, 3.0, 3.5, 5.5, 7.5], [0, 1, 1, 2, 3])
        # per-leaf cold d_h, tagged so the carry-along is checkable
        dh0 = np.full((1, NL), np.nan); dh0[0, :5] = [10.0, 11.0, 12.0, 13.0, 14.0]
        # RANK 1 owns bands [2, 4). It accepted a DEATH in band 3, so its
        # repack holds three sources -- and its band-0/1 rows are the STALE
        # pre-propose pair (no birth), at leaves 0 and 1.
        c1, i1, b1 = _lay(3, [1.5, 3.5, 5.5], [0, 1, 2])
        dh1 = np.full((1, NL), np.nan); dh1[0, :3] = [20.0, 21.0, 22.0]
        self.replies = {
            0: _reply(c0, i0, b0, d_h=dh0, h_h=dh0 * 2),
            1: _reply(c1, i1, b1, d_h=dh1, h_h=dh1 * 2),
        }
        self.ranges = {0: (0, 2), 1: (2, 4)}

    def _merge(self, **kw):
        merge_owned_sources(self.work_c, self.work_i, self.replies, (0, 1),
                            self.ranges, F0, **kw)

    def test_union_of_owned_sources_sorted_and_dense(self):
        self._merge(d_h=self.d_h, h_h=self.h_h)
        for t in range(NT):
            # rank 0's bands 0/1 (1.5, 3.0, 3.5) + rank 1's band 2 (5.5);
            # rank 1's band-3 source DIED and rank 0's stale copy of it is
            # ignored, rank 0's stale copies of 5.5/7.5 likewise
            np.testing.assert_allclose(
                self.work_c[t, 0, :4, F0], [1.5, 3.0, 3.5, 5.5])
            np.testing.assert_array_equal(
                self.work_i[t, 0], [1, 1, 1, 1, 0, 0])
            # bands ride in column 0 of this fixture: ownership is respected
            np.testing.assert_allclose(
                self.work_c[t, 0, :4, 0], [0.0, 1.0, 1.0, 2.0])

    def test_cold_inner_products_follow_their_sources(self):
        self._merge(d_h=self.d_h, h_h=self.h_h)
        # rank 0's leaves 0,1,2 (10, 11, 12) then rank 1's leaf 2 (22)
        np.testing.assert_allclose(self.d_h[0, :4], [10.0, 11.0, 12.0, 22.0])
        np.testing.assert_allclose(self.h_h[0, :4], [20.0, 22.0, 24.0, 44.0])
        self.assertTrue(np.all(np.isnan(self.d_h[0, 4:])))

    def test_dead_slot_coords_are_left_alone(self):
        before = np.array(self.work_c[0, 0, 5], copy=True)
        self._merge()
        np.testing.assert_array_equal(self.work_c[0, 0, 5], before)

    def test_a_neutral_reply_contributes_nothing(self):
        self.replies[1] = {"block_coords": None, "block_inds": None,
                           "block_band_inds": None}
        self._merge()
        for t in range(NT):
            np.testing.assert_allclose(
                self.work_c[t, 0, :3, F0], [1.5, 3.0, 3.5])
            np.testing.assert_array_equal(self.work_i[t, 0], [1, 1, 1, 0, 0, 0])

    def test_all_neutral_leaves_the_branch_untouched(self):
        before_c = np.array(self.work_c, copy=True)
        before_i = np.array(self.work_i, copy=True)
        for r in (0, 1):
            self.replies[r] = {"block_coords": None, "block_inds": None,
                               "block_band_inds": None}
        self._merge()
        np.testing.assert_array_equal(self.work_c, before_c)
        np.testing.assert_array_equal(self.work_i, before_i)

    def test_overflowing_the_branch_raises(self):
        work_c = np.zeros((1, 1, 2, ND)); work_i = np.zeros((1, 1, 2), bool)
        c, i, b = np.zeros((1, 1, 2, ND)), np.ones((1, 1, 2), bool), np.zeros((1, 1, 2), np.int64)
        c2, i2, b2 = np.zeros((1, 1, 2, ND)), np.ones((1, 1, 2), bool), np.full((1, 1, 2), 2, np.int64)
        with self.assertRaisesRegex(RuntimeError, "2 leaves"):
            merge_owned_sources(
                work_c, work_i,
                {0: _reply(c, i, b), 1: _reply(c2, i2, b2)},
                (0, 1), {0: (0, 2), 1: (2, 4)}, F0)

    def test_out_of_grid_band_label_raises(self):
        # An alive source whose FROZEN band label falls outside the union of
        # the ranges belongs to no rank, so the merge would drop it silently.
        # ``-1`` is the sentinel a rank ships for a slot it could not label.
        self.replies[0]["block_band_inds"][0, 0, 0] = -1
        with self.assertRaisesRegex(
                RuntimeError,
                r"rank 0 holds 1 alive source\(s\) with a band label outside "
                r"\[0, 4\) at rung 0"):
            self._merge()

    def test_band_label_at_the_top_of_the_grid_raises(self):
        self.replies[1]["block_band_inds"][1, 0, 2] = 4   # == num_bands
        with self.assertRaisesRegex(
                RuntimeError,
                r"rank 1 holds 1 alive source\(s\) with a band label outside "
                r"\[0, 4\) at rung 1"):
            self._merge()

    def test_an_out_of_grid_DEAD_slot_is_fine(self):
        # dead slots carry -1 by construction; only ALIVE ones are claimed
        self.replies[0]["block_band_inds"][:, 0, 5] = -1
        self.replies[0]["block_inds"][:, 0, 5] = False
        self._merge()
        np.testing.assert_allclose(
            self.work_c[0, 0, :4, F0], [1.5, 3.0, 3.5, 5.5])

    def test_mismatched_block_shapes_raise(self):
        self.replies[1]["block_inds"] = np.zeros((NT, 1, NL + 1), bool)
        with self.assertRaisesRegex(RuntimeError, "rank 1 shipped"):
            self._merge()

    def test_preserve_leaf_identity_keeps_every_source_at_its_own_slot(self):
        # VGB and friends: leaf i IS a physical source (per-leaf transform
        # fills), and ``_write_back_state`` never moves it -- so the owned
        # slots are written back in place and the dense re-sort is skipped.
        c0, i0, b0 = _blank()
        c1, i1, b1 = _blank()
        for t in range(NT):
            c0[t, 0, 4] = [0.0, 9.0]; i0[t, 0, 4] = True; b0[t, 0, 4] = 1
            c1[t, 0, 1] = [0.0, 1.0]; i1[t, 0, 1] = True; b1[t, 0, 1] = 3
        dh0 = np.full((1, NL), np.nan); dh0[0, 4] = 4.0
        dh1 = np.full((1, NL), np.nan); dh1[0, 1] = 5.0
        replies = {0: _reply(c0, i0, b0, d_h=dh0, h_h=dh0),
                   1: _reply(c1, i1, b1, d_h=dh1, h_h=dh1)}
        merge_owned_sources(self.work_c, self.work_i, replies, (0, 1),
                            self.ranges, F0, d_h=self.d_h, h_h=self.h_h,
                            preserve_leaf_identity=True)
        np.testing.assert_array_equal(self.work_i[0, 0], [0, 1, 0, 0, 1, 0])
        self.assertEqual(self.work_c[0, 0, 4, F0], 9.0)   # rank 0's own slot
        self.assertEqual(self.work_c[0, 0, 1, F0], 1.0)   # rank 1's own slot
        np.testing.assert_allclose(self.d_h[0, [1, 4]], [5.0, 4.0])


# ----------------------------------------------------------------------
# the head's replica bookkeeping
# ----------------------------------------------------------------------
class _Layout:
    """Just enough ``WalkerBlockLayout`` for the two replica helpers."""

    head_rank = 0
    compute_ranks = (0, 1)

    @staticmethod
    def replica_index(rank):
        return int(rank)


def _head_move(**attrs):
    m = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
    m.name = "gb_test"
    m._f0_col = F0
    m.preserve_leaf_identity = False
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class ReplicaHeadBookkeepingTest(unittest.TestCase):
    def test_merge_finish_maps_ranks_to_their_band_ranges(self):
        work_c, work_i, _ = _lay(2, [1.5, 5.5], [0, 2])
        c0, i0, b0 = _lay(2, [1.0, 5.5], [0, 2])      # rank 0 moved its band-0 source
        c1, i1, b1 = _lay(2, [1.5, 5.0], [0, 2])      # rank 1 moved its band-2 source
        replies = {0: _reply(c0, i0, b0), 1: _reply(c1, i1, b1)}
        work = SimpleNamespace(coords=work_c, inds=work_i)
        sub = SimpleNamespace(d_h=np.full((1, NL), np.nan),
                              h_h=np.full((1, NL), np.nan))
        _head_move()._replica_merge_finish(
            work, sub, replies, _Layout(), [(0, 2), (2, 4)])
        # each rank's OWN band survived; neither stale copy did
        np.testing.assert_allclose(work_c[0, 0, :2, F0], [1.0, 5.0])
        np.testing.assert_array_equal(work_i[0, 0, :2], [True, True])

    def test_apply_sync_takes_the_heads_reply(self):
        log_like_final = np.zeros(1)
        band_counts = np.zeros((NT, 1, 3), dtype=int)
        replies = {
            # rank 1 agrees to ~1e-13 (the atomicAdd-fill level of spec
            # decision 3) but ships its own band_counts: the HEAD's reply is
            # the one the state takes, exactly.
            0: {"log_like_final": np.array([-5.0]),
                "band_counts": np.full((NT, 1, 3), 2, dtype=int),
                "residual_hash": "aaaa"},
            1: {"log_like_final": np.array([-5.0 * (1 + 1e-13)]),
                "band_counts": np.full((NT, 1, 3), 9, dtype=int),
                "residual_hash": "aaaa"},
        }
        move = _head_move()
        with mock.patch.object(gbs.logger, "warning") as warn:
            hashes = move._replica_apply_sync(
                replies, _Layout(), log_like_final, band_counts)
        self.assertEqual(float(log_like_final[0]), -5.0)
        np.testing.assert_array_equal(band_counts, np.full((NT, 1, 3), 2))
        self.assertEqual(hashes, {0: "aaaa", 1: "aaaa"})
        warn.assert_not_called()

    def test_apply_sync_warns_when_the_log_likes_disagree(self):
        # NEGATIVE CONTROL for the tolerance guard: the replicas agree only to
        # ~1e-12 on GPU, so the alarm is on ``log_like_final``, not on the
        # bit-exact residual hash (which agrees here).
        replies = {
            0: {"log_like_final": np.array([-5.0]),
                "band_counts": np.zeros((NT, 1, 3), dtype=int),
                "residual_hash": "aaaa"},
            1: {"log_like_final": np.array([-5.001]),
                "band_counts": np.zeros((NT, 1, 3), dtype=int),
                "residual_hash": "aaaa"},
        }
        move = _head_move()
        with self.assertLogs(gbs.logger, level="WARNING") as cm:
            move._replica_apply_sync(
                replies, _Layout(), np.zeros(1), np.zeros((NT, 1, 3), dtype=int))
        text = "\n".join(cm.output)
        self.assertIn("log_like_final disagrees", text)
        self.assertIn("[GB_REPLICA", text)
        self.assertIn("r1", text)

    def test_apply_sync_tolerates_a_hash_mismatch_when_the_log_likes_agree(self):
        # Hashes differ, lnL agree: INFO only, no WARNING. The text the
        # digest parser greps for is preserved.
        replies = {
            0: {"log_like_final": np.array([-5.0]),
                "band_counts": np.zeros((NT, 1, 3), dtype=int),
                "residual_hash": "aaaa"},
            1: {"log_like_final": np.array([-5.0 - 1e-12]),
                "band_counts": np.zeros((NT, 1, 3), dtype=int),
                "residual_hash": "bbbb"},
        }
        move = _head_move()
        with self.assertLogs(gbs.logger, level="INFO") as cm:
            move._replica_apply_sync(
                replies, _Layout(), np.zeros(1), np.zeros((NT, 1, 3), dtype=int))
        self.assertEqual(
            [r.levelname for r in cm.records if r.levelname == "WARNING"], [])
        text = "\n".join(cm.output)
        self.assertIn("residual hashes disagree after sync", text)
        self.assertIn("r0:aaaa", text)
        self.assertIn("r1:bbbb", text)


# ----------------------------------------------------------------------
# the fourth command
# ----------------------------------------------------------------------
class _FakeACS:
    gpus = None


class _FakeModel:
    def __init__(self):
        self.analysis_container_arr = _FakeACS()


class _FakeSorter:
    """Records what it was built from; answers ``get_band_info`` only."""

    built = []

    def __init__(self, branch, band_edges, band_N_vals, **kwargs):
        self.branch, self.kwargs = branch, kwargs
        type(self).built.append(self)

    def get_band_info(self):
        return {"band_counts": np.full((2, 1, 3), 7, dtype=int)}


def _branch(coords, inds):
    return SimpleNamespace(coords=np.asarray(coords, float), inds=np.asarray(inds, bool))


def _sync_move():
    """A skeleton GB move carrying only what ``_gb_serve_sync`` reads."""
    m = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
    m.branch_name = "gb"
    m.name = "gb_test"
    m._backend_name = "lisatools_cpu"
    m.gf_rank = 1
    m.fanout = None
    m._gb_session = None                     # gb_sync runs AFTER the teardown
    m.mempool = gbs._NoOpMempool()
    m._configure_domain = lambda acs: None
    m._bind_parent_acs = lambda acs: None
    # the BandSorter kwargs the finish-time rebuild passes
    m.band_edges = np.linspace(1e-3, 4e-3, 4)
    m.band_N_vals = np.full(3, 128)
    m.force_backend = "cpu"
    m.parameter_transforms = None
    m.max_data_store_size = 6000
    m.gb = None
    m.gb_wdm_comp = None
    m.gb_fd_comp = None
    m.wdm_band_slab_layers = None
    m.wdm_slab_guard_layers = 1
    m.psd_shared_mirror = False
    m.psd_mirror_parity_proposes = 0
    m.psd_mirror_parity_rows = 64
    m.waveform_kwargs = {}
    # the propose-start non-GB snapshot survives the finish teardown
    m.reset_non_gb_linear_data_arr = object()
    m.check_ll_inject = lambda model, sorter: np.array([-12.5])
    return m


def _sync_payload(coords, inds):
    part = SimpleNamespace(
        sub_states={"gb": SimpleNamespace(
            tempered_initialized=True, branch=_branch(coords, inds))}
    )
    return {
        "ntemps": 2, "nwalkers": 1, "clock_vals": {}, "tables": {},
        "rank_seed": 7, "band_range": (0, 2), "replica": (1, 2),
        "merged_state": part,
    }


class SyncCommandTest(unittest.TestCase):
    def setUp(self):
        _FakeSorter.built = []
        for target, value in (("pin_main_device", lambda xp, gpus: None),
                              ("BandSorter", _FakeSorter),
                              ("residual_hash", lambda acs: "cafef00d")):
            p = mock.patch.object(gbs, target, value)
            p.start()
            self.addCleanup(p.stop)

    def test_gb_ops_carries_the_fourth_command(self):
        # the session's four commands, in order; the F-stat epoch fit's three
        # (gb_fstat_ref_row / gb_fstat_stage_b / gb_fstat_release,
        # parallel-fit branch) follow
        self.assertEqual(
            gbs.GB_OPS[:4],
            ("gb_run_proposal", "gb_run_tempering", "gb_finish", "gb_sync"),
        )
        self.assertIn("gb_sync", gbs.GB_OPS)

    def test_sync_rebuilds_from_the_merged_branch_with_no_open_session(self):
        move = _sync_move()
        coords = np.arange(2 * 1 * 4 * 3, dtype=float).reshape(2, 1, 4, 3)
        inds = np.ones((2, 1, 4), dtype=bool)
        rep = move.gf_serve("gb_sync", _sync_payload(coords, inds),
                            {"seq": 4, "call_index": 1}, _FakeModel())
        self.assertEqual(set(rep), {"log_like_final", "band_counts", "residual_hash"})
        np.testing.assert_array_equal(rep["log_like_final"], [-12.5])
        np.testing.assert_array_equal(rep["band_counts"], np.full((2, 1, 3), 7))
        self.assertEqual(rep["residual_hash"], "cafef00d")
        # the sorter was built over the arrays the head shipped, not a session's
        self.assertEqual(len(_FakeSorter.built), 1)
        np.testing.assert_array_equal(_FakeSorter.built[0].branch.coords, coords)
        self.assertIsNone(_FakeSorter.built[0].kwargs.get("rj_prop"))

    def test_sync_restores_the_rank_block_attributes(self):
        move = _sync_move()
        move.nwalkers, move.ntemps = 4, 9
        move._owned_band_range = None
        coords = np.zeros((2, 1, 4, 3)); inds = np.zeros((2, 1, 4), bool)
        move.gf_serve("gb_sync", _sync_payload(coords, inds),
                      {"seq": 4, "call_index": 1}, _FakeModel())
        self.assertEqual((move.nwalkers, move.ntemps), (4, 9))
        self.assertIsNone(move._owned_band_range)

    def test_sync_refuses_a_missing_non_gb_snapshot(self):
        move = _sync_move()
        move.reset_non_gb_linear_data_arr = None
        coords = np.zeros((2, 1, 4, 3)); inds = np.zeros((2, 1, 4), bool)
        with self.assertRaisesRegex(RuntimeError, "non-GB residual snapshot"):
            move.gf_serve("gb_sync", _sync_payload(coords, inds),
                          {"seq": 4, "call_index": 1}, _FakeModel())

    def test_a_neutral_block_answers_without_a_rebuild(self):
        # Several walker blocks under replicas (GF_GPU_ROUTING): a block the
        # head marked neutral ran nothing, took no non-GB snapshot and must
        # answer gb_sync without one -- the live blocks' sync goes on.
        move = _sync_move()
        move.reset_non_gb_linear_data_arr = None
        coords = np.zeros((2, 1, 4, 3)); inds = np.zeros((2, 1, 4), bool)
        payload = dict(_sync_payload(coords, inds), neutral=True)
        rep = move.gf_serve("gb_sync", payload, {"seq": 4, "call_index": 1}, _FakeModel())
        self.assertTrue(rep["neutral"])
        self.assertIsNone(rep["log_like_final"])
        self.assertEqual(_FakeSorter.built, [])        # nothing rebuilt

    def test_unknown_op_still_raises(self):
        with self.assertRaises(ValueError):
            _sync_move().gf_serve("nope", {}, {"seq": 1, "call_index": 1}, _FakeModel())


class MergeWalkerAxisTest(unittest.TestCase):
    """The merge spans the whole walker BLOCK, not walker 0 (2026-09-23).

    Under ``n_compute = n_blocks x R`` a replicated block can be several
    walkers wide, so every ``[t, 0]`` in the merge became ``[t, w]``. At a
    one-walker block the loop runs once and the result is unchanged, which
    the rest of this module already pins.
    """

    NW = 2

    def _reply_w(self, coords, inds, bands):
        return {
            "block_coords": np.asarray(coords, float),
            "block_inds": np.asarray(inds, bool),
            "block_band_inds": np.asarray(bands, np.int64),
            "d_h": np.zeros((self.NW, NL)),
            "h_h": np.zeros((self.NW, NL)),
        }

    def _blank_w(self, fill=-99.0):
        return (np.full((NT, self.NW, NL, ND), fill),
                np.zeros((NT, self.NW, NL), bool))

    def test_each_walker_is_merged_independently(self):
        # rank 0 owns bands [0,2), rank 1 owns [2,4). Each walker carries a
        # DIFFERENT source per rank, so a merge that only ever looked at
        # walker 0 would leave walker 1 empty.
        co0 = np.full((NT, self.NW, NL, ND), -99.0)
        in0 = np.zeros((NT, self.NW, NL), bool)
        bd0 = np.full((NT, self.NW, NL), -1, np.int64)
        co1 = co0.copy(); in1 = in0.copy(); bd1 = bd0.copy()
        for t in range(NT):
            for w in range(self.NW):
                # rank 0's owned source, f0 = 10 + w
                co0[t, w, 0, F0] = 10.0 + w; in0[t, w, 0] = True; bd0[t, w, 0] = 1
                # rank 1's owned source, f0 = 20 + w
                co1[t, w, 3, F0] = 20.0 + w; in1[t, w, 3] = True; bd1[t, w, 3] = 2

        wc, wi = self._blank_w()
        merge_owned_sources(
            wc, wi,
            {0: self._reply_w(co0, in0, bd0), 1: self._reply_w(co1, in1, bd1)},
            [0, 1], {0: (0, 2), 1: (2, 4)}, F0,
        )
        for t in range(NT):
            for w in range(self.NW):
                self.assertEqual(int(wi[t, w].sum()), 2,
                                 msg=f"rung {t}, walker {w}")
                np.testing.assert_allclose(
                    wc[t, w, :2, F0], [10.0 + w, 20.0 + w],
                    err_msg=f"rung {t}, walker {w} sorted by f0")

    def test_shape_gate_names_the_block_width(self):
        wc, wi = self._blank_w()
        # a reply shaped for ONE walker against a two-walker block
        bad = _reply(np.zeros((NT, 1, NL, ND)), np.zeros((NT, 1, NL), bool),
                     np.full((NT, 1, NL), -1, np.int64))
        with self.assertRaisesRegex(RuntimeError, "walker block"):
            merge_owned_sources(wc, wi, {0: bad}, [0], {0: (0, 4)}, F0)

    def test_overflow_message_names_the_walker(self):
        co = np.full((NT, self.NW, NL, ND), -99.0)
        ind = np.zeros((NT, self.NW, NL), bool)
        bd = np.full((NT, self.NW, NL), -1, np.int64)
        ind[:, :, :] = True; bd[:, :, :] = 1
        co[:, :, :, F0] = np.arange(NL)
        # one rank owning every slot of a 6-leaf branch is fine; two ranks
        # each owning all 6 overflow it
        with self.assertRaisesRegex(RuntimeError, "walker"):
            merge_owned_sources(
                *self._blank_w(),
                {0: self._reply_w(co, ind, bd), 1: self._reply_w(co, ind, bd)},
                [0, 1], {0: (0, 2), 1: (0, 2)}, F0,
            )

    def test_preserve_leaf_identity_spans_walkers(self):
        co = np.full((NT, self.NW, NL, ND), -99.0)
        ind = np.zeros((NT, self.NW, NL), bool)
        bd = np.full((NT, self.NW, NL), -1, np.int64)
        for t in range(NT):
            for w in range(self.NW):
                co[t, w, 4, F0] = 30.0 + w; ind[t, w, 4] = True; bd[t, w, 4] = 1
        wc, wi = self._blank_w()
        merge_owned_sources(wc, wi, {0: self._reply_w(co, ind, bd)}, [0],
                            {0: (0, 4)}, F0, preserve_leaf_identity=True)
        for t in range(NT):
            for w in range(self.NW):
                # written back AT ITS OWN SLOT, not densely repacked
                self.assertTrue(wi[t, w, 4])
                self.assertEqual(int(wi[t, w].sum()), 1)
                self.assertAlmostEqual(wc[t, w, 4, F0], 30.0 + w)
