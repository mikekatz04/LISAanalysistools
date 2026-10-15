"""GB under the ``GF_GPU_ROUTING`` layout at the 9mo production shape: 4 walkers x R = 2.

``NGPUS=8 GF_GPU_ROUTING=1`` on the 4-walker store builds 8 compute ranks,
ONE device each, as 4 walker blocks of one walker x R = 2 replicas. Inside a
block the two replicas split the GB band grid by range; the head merges each
block's replies BY OWNED SOURCE and ships every replica the merged block
back (``gb_sync``). Before 2026-10-15 that head bookkeeping treated all
compute ranks as ONE group -- right only for one-walker replica mode -- and
at this shape it shape-failed in ``merge_owned_sources`` (an N-walker branch
against one-walker replies), broadcast the head's one walker over all four
in the sync, kept a single block's rows of the per-walker valve statistic,
and skipped the sync for EVERY block when one block was neutral.

Skeleton-level, exactly like ``tests/test_gb_orchestrator_merge.py`` (whose
skeleton move and model this module reuses): the REAL head
``_propose_orchestrated`` drives FakeWorld ranks whose ``gf_serve`` is a
deterministic stub. The stub plays a physical scenario:

* every (rung, walker) starts with one source in each of bands 0-3
  (``NLEAVES = 5``: a birth needs a dead slot to land in);
* in the propose, walker ``w`` LOSES its band-``w`` source and BIRTHS one in
  band ``(w + 3) % 6`` (so every walker's change is different);
* a replica reports the post-propose sources of the bands it OWNS (the
  payload's ``band_range``) and its STALE pre-propose copies of the rest,
  densely repacked in frequency order -- what ``_write_back_state`` does;
* a walker-block (R = 1) rank reports the whole post-propose walker.

So the R = 1 world (4 ranks, the 4-GPU production layout) and the R = 2 world
(8 ranks) describe THE SAME PROPOSE, and the head must end both with the same
state -- the parity claim the laptop can make (the GB kernels themselves
cannot run here; their end-to-end parity is cluster gate G4).
"""

import hashlib
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.moves import gbspecialstretch as gbs

from tests.test_gb_orchestrator_merge import (
    MOVE_NAME,
    NDIM,
    NLEAVES,
    _FakeModel,
    make_move,
)
from tests.test_gf_substate_roundtrip import (
    BAND_EDGES,
    NTEMPS,
    NUM_BANDS,
    NWALKERS,
    make_state,
)

#: the head's merge key: ``_f0_col`` unset on the skeleton -> column 1
F0 = 1
#: ``-ppn 1`` round robin of the run world over 4 nodes (the monitor rank is
#: split off before the layout): 8 compute ranks + the saver on node 0
NODES_R2 = [0, 1, 2, 3, 0, 1, 2, 3, 0]
#: the 4-GPU production world: 4 compute ranks + saver over 2 nodes
NODES_R1 = [0, 1, 0, 1, 0]
SYNC_KEYS = frozenset({"log_like_final", "band_counts", "residual_hash"})


# ----------------------------------------------------------------------
# the scenario
# ----------------------------------------------------------------------
def _src(t, w, band, born=False):
    """One source row: band tag, f0 [mHz] inside the band, (rung, walker) tag."""
    row = np.zeros(NDIM)
    row[0] = float(band)
    # BAND_EDGES are 1..7 mHz in 6 bands of 1 mHz: band b is (1 + b, 2 + b)
    row[F0] = 1.5 + band + (0.25 if born else 0.0) + 0.01 * w
    row[2] = 100.0 * t + w
    row[3] = 1.0 if born else 0.0
    return row


def _pre(t, w):
    return [(b, _src(t, w, b)) for b in range(4)]


def _post(t, w):
    """Walker ``w`` loses its band-``w`` source and births one in band (w+3)%6."""
    out = [(b, _src(t, w, b)) for b in range(4) if b != w]
    nb = (w + 3) % NUM_BANDS
    out.append((nb, _src(t, w, nb, born=True)))
    return sorted(out, key=lambda br: br[1][F0])


def _uid(row):
    return int(round(row[F0] * 100)) + 1000 * int(row[2] % 100) + 10000 * int(row[2] // 100)


def _band_of(f0_mhz):
    b = int(np.searchsorted(BAND_EDGES, f0_mhz / 1e3, side="right")) - 1
    return min(max(b, 0), NUM_BANDS - 1)


def _seed_state(state):
    """Every (rung, walker) holds exactly ``_pre``; IDs and cold d_h tagged."""
    sub = state.sub_states["gb"]
    sub.branch.coords[:] = -7.0
    sub.branch.inds[:] = False
    sub.source_uid[:] = -1
    sub.d_h[:] = np.nan
    sub.h_h[:] = np.nan
    for t in range(NTEMPS):
        for w in range(NWALKERS):
            for slot, (_b, row) in enumerate(_pre(t, w)):
                sub.branch.coords[t, w, slot] = row
                sub.branch.inds[t, w, slot] = True
                sub.source_uid[t, w, slot] = _uid(row)
                if t == 0:
                    sub.d_h[w, slot] = 10.0 * row[F0]
                    sub.h_h[w, slot] = 20.0 * row[F0]
    sub.sync_cold_row(state, "gb")
    return state


def _owned(band, rng):
    return rng is None or (rng[0] <= band < rng[1])


# ----------------------------------------------------------------------
# the replica-aware rank stub
# ----------------------------------------------------------------------
def _stub(move, op, payload, clock, model):
    rank, (w0, w1) = move._test_rank, move._test_block
    B, ntemps, nb = int(payload["nwalkers"]), int(payload["ntemps"]), NUM_BANDS
    assert B == w1 - w0
    rng = payload.get("band_range")
    rep_id = payload.get("replica")
    move.served.append(op)
    move.payloads.append(payload)
    if op == "gb_run_proposal":
        move.token = (clock["seq"], clock["call_index"])
        move.neutral = bool(payload["neutral"])
        move.band_range, move.replica_id = rng, rep_id
    elif op != "gb_sync":
        assert payload["session"] == move.token
    owned = np.array([_owned(b, rng) for b in range(nb)], dtype=int)

    if op == "gb_run_proposal":
        if move.neutral:
            return {
                "log_like_cold": np.zeros(B),
                "prop_counts": np.zeros((2, ntemps, nb), dtype=int),
                "acc_counts": np.zeros((2, ntemps, nb), dtype=int),
                "cold_counts": np.zeros((2, 2, B, nb), dtype=int),
                "alive_per_temp": [0] * ntemps, "n_alive": 0,
                "start_diffs": np.zeros(B), "drift": 0.0, "rj_at_cap": None,
                "timing": None,
            }
        # BAND-DISJOINT counters: a replica counts only its own bands
        pc = np.broadcast_to(owned, (2, ntemps, nb)).astype(int)
        return {
            "log_like_cold": 100.0 + w0 + np.arange(B, dtype=float),
            "prop_counts": 2 * pc, "acc_counts": pc,
            "cold_counts": np.stack([np.broadcast_to(2 * owned, (2, B, nb)),
                                     np.broadcast_to(owned, (2, B, nb))]).astype(int),
            # the WHOLE block's alive count (stale copies included) on EVERY
            # replica -- the head must take it from one per block
            "alive_per_temp": [4 * B] * ntemps, "n_alive": 4 * B * ntemps,
            "start_diffs": np.zeros(B), "drift": 0.0, "rj_at_cap": 0,
            "timing": None,
        }

    if op == "gb_run_tempering":
        sw = np.broadcast_to(owned[:, None], (nb, ntemps - 1)).astype(int)
        if move.neutral:
            sw = np.zeros_like(sw)
        return {
            "log_like_cold": 200.0 + w0 + np.arange(B, dtype=float),
            "band_swaps_accepted": sw, "band_swaps_proposed": 2 * sw,
            "ll_change_sum_temp_cold": np.zeros(B), "drift": 0.0,
            "census": None, "timing": None,
        }

    if op == "gb_finish":
        tag = float(w0) + np.arange(B, dtype=float)
        rows = np.repeat(tag[:, None], nb, axis=1)
        cap = None
        if payload["want_cap_stats"]:
            cap = {"band_lls": rows, "band_gain": 1000.0 + rows, "lls": rows,
                   "dof": 7.0, "band_dof": np.full(nb, 5.0), "is_cells": False}
        if move.neutral:
            return {
                "block_coords": None, "block_inds": None,
                "block_band_inds": None, "block_source_uid": None,
                "d_h": np.full((B, NLEAVES), np.nan),
                "h_h": np.full((B, NLEAVES), np.nan),
                "band_counts": np.zeros((ntemps, B, nb), dtype=int),
                "log_like_final": np.zeros(B), "cap_stats": cap,
                "fstat_ctr_fallback_rows": 0,
                "band_dof": None if cap is None else np.full(nb, 5.0),
                "rj_split": None, "replace_census": None, "timing": None,
            }
        coords = np.full((ntemps, B, NLEAVES, NDIM), -1.0 - rank)
        inds = np.zeros((ntemps, B, NLEAVES), dtype=bool)
        bands = np.full((ntemps, B, NLEAVES), -1, dtype=np.int64)
        uid = np.full((ntemps, B, NLEAVES), -1, dtype=np.int64)
        d_h = np.full((B, NLEAVES), np.nan)
        h_h = np.full((B, NLEAVES), np.nan)
        counts = np.zeros((ntemps, B, nb), dtype=int)
        for t in range(ntemps):
            for j, w in enumerate(range(w0, w1)):
                if rng is None:
                    view = _post(t, w)
                else:
                    # owned bands: what THIS replica proposed; the rest: its
                    # stale pre-propose copies, renumbered with them
                    view = ([br for br in _post(t, w) if _owned(br[0], rng)]
                            + [br for br in _pre(t, w) if not _owned(br[0], rng)])
                    view.sort(key=lambda br: br[1][F0])
                for slot, (b, row) in enumerate(view):
                    coords[t, j, slot] = row
                    inds[t, j, slot] = True
                    bands[t, j, slot] = b
                    uid[t, j, slot] = _uid(row)
                    counts[t, j, b] += 1
                    if t == 0:
                        d_h[j, slot] = 10.0 * row[F0]
                        h_h[j, slot] = 20.0 * row[F0]
        replica = rng is not None
        return {
            "block_coords": coords, "block_inds": inds,
            "block_band_inds": bands if replica else None,
            "block_source_uid": uid, "d_h": d_h, "h_h": h_h,
            # replica mode defers both to gb_sync, exactly like the real body
            "band_counts": None if replica else counts,
            "log_like_final": None if replica else 500.0 + tag,
            "cap_stats": cap, "fstat_ctr_fallback_rows": 0,
            "band_dof": np.full(nb, 5.0), "rj_split": None,
            "replace_census": None, "timing": None,
        }

    if op == "gb_sync":
        if payload.get("neutral"):
            return {"neutral": True, "log_like_final": None,
                    "band_counts": None, "residual_hash": None}
        part = payload["merged_state"]
        br = part.sub_states["gb"].branch
        move.synced = (np.array(br.coords, copy=True), np.array(br.inds, copy=True))
        counts = np.zeros((ntemps, B, nb), dtype=int)
        for t, j, leaf in zip(*np.nonzero(br.inds)):
            counts[t, j, _band_of(br.coords[t, j, leaf, F0])] += 1
        alive = np.ascontiguousarray(br.coords[br.inds])
        return {
            "log_like_final": 500.0 + w0 + np.arange(B, dtype=float),
            "band_counts": counts,
            "residual_hash": hashlib.sha1(alive.tobytes()).hexdigest()[:16],
        }
    raise AssertionError(f"unexpected op {op!r}")


def run_world(state, *, size, nodes, gpu_routing, use_prior_removal=False):
    """One orchestrated propose over a FakeWorld; ``(head result, moves, layout)``."""
    world = FakeWorld(size, nodes=nodes)
    moves, layouts = {}, {}

    def fn(rank, comm):
        layout = build_layout(comm, NWALKERS, [0, 1], legacy=False,
                              gpu_routing=gpu_routing)
        layouts[rank] = layout
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        move = make_move(rank, layout.block_of(rank),
                         use_prior_removal=use_prior_removal)
        move.gf_serve = types.MethodType(_stub, move)
        move.band_range = move.replica_id = move.synced = None
        moves[rank] = move
        model = _FakeModel(seed=0)
        if layout.role_of(rank) == RankRole.HEAD:
            move.fanout = WalkerFanout(fcomm, layout, rank, model=_FakeModel(99))
            move.fanout.enter_stage("gb", "gb_kind")
            try:
                return move.propose(model, state)
            finally:
                move.fanout.stop()
        ComputeService(fcomm, layout, rank, registry={MOVE_NAME: move},
                       model=model).serve()
        return "served"

    out = world.run(fn)
    return out[0], moves, layouts[0]


class ReplicaBlocksGBTest(unittest.TestCase):
    """4 walkers x R = 2: the band split, the per-block merge, the sync."""

    def setUp(self):
        if gbs.cp is not np:  # pragma: no cover - GPU box
            self.skipTest("the ladder helper runs on the host array module")
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore)
        p = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        p.start()
        self.addCleanup(p.stop)

    def _restore(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    def _r2(self, **kw):
        state = _seed_state(make_state(np.random.default_rng(7)))
        return run_world(state, size=9, nodes=NODES_R2, gpu_routing=True, **kw)

    def _r1(self, **kw):
        state = _seed_state(make_state(np.random.default_rng(7)))
        return run_world(state, size=5, nodes=NODES_R1, gpu_routing=False, **kw)

    # -- the layout and the band split -------------------------------
    def test_the_layout_is_four_blocks_of_two_replicas(self):
        _res, moves, lay = self._r2()
        self.assertEqual((lay.n_compute, lay.n_blocks, lay.ranks_per_block),
                         (8, 4, 2))
        self.assertEqual(sorted(moves), list(range(8)))
        for r in range(8):
            self.assertEqual(lay.block_of(r), (r // 2, r // 2 + 1))

    def test_each_block_splits_the_whole_band_grid_between_its_replicas(self):
        _res, moves, lay = self._r2()
        ranges = None
        for b in range(4):
            r0, r1 = lay.ranks_in_block(b)
            g0, g1 = moves[r0].band_range, moves[r1].band_range
            self.assertEqual((moves[r0].replica_id, moves[r1].replica_id),
                             ((0, 2), (1, 2)))
            # disjoint, contiguous, covering [0, NUM_BANDS)
            self.assertEqual(g0[0], 0)
            self.assertEqual(g0[1], g1[0])
            self.assertEqual(g1[1], NUM_BANDS)
            self.assertGreater(g0[1], 0)
            self.assertLess(g0[1], NUM_BANDS)
            # ONE static split, the same in every block
            ranges = ranges or (g0, g1)
            self.assertEqual((g0, g1), ranges)

    def test_every_rank_answers_four_commands_including_gb_sync(self):
        _res, moves, _lay = self._r2()
        for r, m in moves.items():
            self.assertEqual(
                m.served,
                ["gb_run_proposal", "gb_run_tempering", "gb_finish", "gb_sync"],
                r)

    # -- births and deaths land in the right walker ------------------
    def test_births_and_deaths_land_in_the_right_walker(self):
        (new_state, _acc), moves, _lay = self._r2()
        work = moves[0]._work_branch(new_state)
        for t in range(NTEMPS):
            for w in range(NWALKERS):
                want = np.array([row for _b, row in _post(t, w)])
                n = len(want)
                np.testing.assert_array_equal(
                    work.inds[t, w], [True] * n + [False] * (NLEAVES - n),
                    err_msg=f"rung {t} walker {w}")
                np.testing.assert_allclose(
                    work.coords[t, w, :n], want, err_msg=f"rung {t} walker {w}")
                # every surviving source is THIS walker's (the rung/walker tag)
                np.testing.assert_allclose(work.coords[t, w, :n, 2], 100.0 * t + w)
                # the death and the birth, explicitly
                bands = set(work.coords[t, w, :n, 0].astype(int).tolist())
                self.assertNotIn(w, bands - {(w + 3) % NUM_BANDS})
                born = work.coords[t, w, :n, 3] == 1.0
                self.assertEqual(int(born.sum()), 1)
                self.assertEqual(int(work.coords[t, w, :n][born][0, 0]),
                                 (w + 3) % NUM_BANDS)

    def test_ids_and_cold_inner_products_follow_their_sources(self):
        (new_state, _acc), _moves, _lay = self._r2()
        sub = new_state.sub_states["gb"]
        for w in range(NWALKERS):
            post = _post(0, w)
            n = len(post)
            np.testing.assert_array_equal(
                sub.source_uid[0, w, :n], [_uid(row) for _b, row in post])
            np.testing.assert_allclose(
                sub.d_h[w, :n], [10.0 * row[F0] for _b, row in post])
            self.assertTrue(np.all(np.isnan(sub.d_h[w, n:])))

    def test_gb_sync_ships_each_block_ITS_merged_walker(self):
        (_new, _acc), moves, lay = self._r2()
        for b in range(4):
            got = [moves[r].synced for r in lay.ranks_in_block(b)]
            for coords, inds in got:
                self.assertEqual(coords.shape[1], 1)
                want = np.array([row for _b, row in _post(0, b)])
                np.testing.assert_allclose(coords[0, 0, :len(want)], want)
                self.assertTrue(inds[0, 0, :len(want)].all())
            # both replicas of the block rebuilt from the SAME slice
            np.testing.assert_array_equal(got[0][0], got[1][0])

    def test_log_like_and_band_counts_come_from_each_blocks_lead(self):
        (new_state, _acc), _moves, _lay = self._r2()
        np.testing.assert_allclose(new_state.log_like[0], 500.0 + np.arange(NWALKERS))
        bi = new_state.sub_states["gb"].band_info
        for t in range(NTEMPS):
            for w in range(NWALKERS):
                want = np.zeros(NUM_BANDS, dtype=int)
                for b, _row in _post(t, w):
                    want[b] += 1
                np.testing.assert_array_equal(bi["band_num_binaries"][t, w], want)

    def test_the_valve_statistic_is_one_row_per_walker_from_the_leads(self):
        (_new, _acc), moves, _lay = self._r2()
        stats = moves[0].cap_calls[0]
        self.assertEqual(stats["lls"].shape, (NWALKERS, NUM_BANDS))
        np.testing.assert_allclose(stats["lls"][:, 0], np.arange(NWALKERS))
        np.testing.assert_allclose(stats["band_gain"][:, 0],
                                   1000.0 + np.arange(NWALKERS))

    def test_no_replica_disagreement_is_logged(self):
        with self.assertLogs(gbs.logger, level="INFO") as cm:
            self._r2()
        text = "\n".join(cm.output)
        self.assertNotIn("log_like_final disagrees", text)
        self.assertNotIn("residual hashes disagree", text)

    def test_the_split_and_the_merge_announce_themselves(self):
        """The two lines a cluster log is read for (G4 / G6): the band-range
        assignment -- source-weighted, with the load per range -- and the
        per-propose sync over every block."""
        with self.assertLogs(gbs.logger, level="INFO") as cm:
            self._r2()
        text = "\n".join(cm.output)
        # 4 walkers x one source in each of bands 0-3: weights 4,4,4,4,0,0
        self.assertIn(
            "[GB_REPLICA gb_test] band split over R=2 replica(s) per walker block "
            "(source-weighted): replica 0 -> bands [0, 2), replica 1 -> bands "
            "[2, 6); alive cold sources per range 8 / 8", text)
        self.assertIn("[GB_REPLICA gb_test] synced 4 walker block(s) x 2 replica(s)",
                      text)

    # -- parity with the R = 1 layout (the 4-GPU production one) ------
    def test_the_same_propose_on_R1_and_R2_gives_the_same_state(self):
        (s2, a2), m2, _l2 = self._r2()
        (s1, a1), m1, l1 = self._r1()
        self.assertEqual((l1.n_compute, l1.n_blocks, l1.ranks_per_block), (4, 4, 1))
        w1, w2 = m1[0]._work_branch(s1), m2[0]._work_branch(s2)
        np.testing.assert_array_equal(w2.inds, w1.inds)
        np.testing.assert_allclose(w2.coords[w2.inds], w1.coords[w1.inds])
        np.testing.assert_allclose(s2.log_like, s1.log_like)
        b1, b2 = s1.sub_states["gb"], s2.sub_states["gb"]
        for key in ("band_num_binaries", "band_temps", "band_num_proposed",
                    "band_num_accepted", "band_num_proposed_rj",
                    "band_num_accepted_rj", "band_swaps_proposed",
                    "band_swaps_accepted"):
            np.testing.assert_allclose(b2.band_info[key], b1.band_info[key],
                                       err_msg=key)
        np.testing.assert_array_equal(b2.source_uid[b2.branch.inds],
                                      b1.source_uid[b1.branch.inds])
        np.testing.assert_allclose(b2.d_h[w2.inds[0]], b1.d_h[w1.inds[0]])
        np.testing.assert_array_equal(a2, a1)
        st1, st2 = m1[0].cap_calls[0], m2[0].cap_calls[0]
        np.testing.assert_allclose(st2["lls"], st1["lls"])

    # -- a neutral block among live ones -----------------------------
    def test_a_neutral_block_skips_only_itself_in_the_sync(self):
        state = _seed_state(make_state(np.random.default_rng(7)))
        sub = state.sub_states["gb"]
        sub.branch.inds[:, 1] = False          # walker 1 (block 1) has nothing
        sub.sync_cold_row(state, "gb")
        before = np.array(state.log_like[0], copy=True)
        (new_state, _acc), moves, lay = run_world(
            state, size=9, nodes=NODES_R2, gpu_routing=True,
            use_prior_removal=True)
        for r in lay.ranks_in_block(1):
            self.assertTrue(moves[r].payloads[0]["neutral"])
            self.assertTrue(moves[r].payloads[-1]["neutral"])
            self.assertIsNone(moves[r].synced)
        # the gb_sync still ran for the live blocks...
        for b in (0, 2, 3):
            for r in lay.ranks_in_block(b):
                self.assertEqual(moves[r].served[-1], "gb_sync")
                self.assertIsNotNone(moves[r].synced)
        # ...so they carry their rebuilt lnL; the neutral walker keeps its own
        np.testing.assert_allclose(new_state.log_like[0, [0, 2, 3]],
                                   500.0 + np.array([0, 2, 3]))
        self.assertAlmostEqual(float(new_state.log_like[0, 1]), float(before[1]))
        work = moves[0]._work_branch(new_state)
        self.assertEqual(int(work.inds[:, 1].sum()), 0)


class ReplicaHeadHelpersTest(unittest.TestCase):
    """``_replica_apply_sync`` compares each replica with ITS block's lead."""

    def test_blocks_are_compared_separately(self):
        nodes = NODES_R2
        lay = FakeWorld(9, nodes=nodes).run(
            lambda r, c: build_layout(c, NWALKERS, [0, 1], legacy=False,
                                      gpu_routing=True))[0]
        replies = {
            r: {"log_like_final": np.array([10.0 * (r // 2)]),
                "band_counts": np.full((NTEMPS, 1, NUM_BANDS), r // 2),
                "residual_hash": f"h{r // 2}"}
            for r in range(8)
        }
        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        ll = np.zeros(NWALKERS)
        counts = np.zeros((NTEMPS, NWALKERS, NUM_BANDS), dtype=int)
        with self.assertLogs(gbs.logger, level="INFO") as cm:
            hashes = move._replica_apply_sync(replies, lay, ll, counts)
        text = "\n".join(cm.output)
        self.assertNotIn("disagree", text)
        self.assertIn("synced 4 walker block(s) x 2 replica(s): max "
                      "|log_like_final - block lead| = 0.000e+00; residual-hash "
                      "mismatch in 0 block(s)", text)
        np.testing.assert_allclose(ll, [0.0, 10.0, 20.0, 30.0])
        for w in range(NWALKERS):
            self.assertTrue((counts[:, w] == w).all())
        self.assertEqual(hashes, {r: f"h{r // 2}" for r in range(8)})
        # a replica that drifts from ITS lead is named
        replies[5]["log_like_final"] = np.array([20.5])
        with self.assertLogs(gbs.logger, level="WARNING") as cm:
            move._replica_apply_sync(replies, lay, ll, counts)
        self.assertIn("block 2 lead r4", "\n".join(cm.output))
        self.assertIn("r5:", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main()
