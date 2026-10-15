"""Head-directed commands over the fake communicator: round trip, errors, stop, isolation."""

import time
import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import (
    LIKELIHOOD_OP,
    RESIDUAL_HASH_OP,
    ComputeService,
    RemoteWorkerError,
    WalkerFanout,
    concat_blocks,
    fanout_digest_line,
    replicas_agree,
    residual_hash,
    _array_bytes,
    _sha1_16,
)
from lisatools.globalfit.communication.ranks import RankRole, build_layout


class _Acs:
    """Stands in for an AnalysisContainerArray: one likelihood per local walker."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def likelihood(self, complex=False):
        return self.values


class _StubMove:
    gf_move_name = "stub"

    def __init__(self):
        self.calls = []
        self.gf_stage_kind = None

    def gf_serve(self, op, payload, clock, model):
        self.calls.append((op, self.gf_stage_kind, clock["call_index"]))
        if op == "boom":
            raise ValueError("kaboom")
        if op == "unpicklable":
            return {"fn": lambda: None}  # pickle.dumps rejects this, like FakeComm.send
        return {"rank_sum": float(np.sum(payload["x"])), "model": model}


class _DigestState:
    log_like = np.zeros((1, 1))
    branches_coords = {"mbh": np.zeros((1, 1, 1, 2))}
    branches_inds = {"mbh": np.ones((1, 1, 1), dtype=bool)}


def _run_world(size, nodes, nwalkers, head_fn, saver_fn=None, ranks_per_gpu=1):
    world = FakeWorld(size, nodes=nodes)
    stubs = {}

    def fn(rank, comm):
        # gpu_routing pinned ON: some callers below ask for shapes only the
        # unified rule resolves. It is opt-in at the launcher
        # (ranks.GPU_ROUTING_ENV) and agrees with the legacy rule on every
        # shape the legacy rule accepts, so pinning it changes nothing for
        # the walker-block and one-walker cases this helper also serves.
        layout = build_layout(
            comm, nwalkers, [0, 1], legacy=False, ranks_per_gpu=ranks_per_gpu,
            gpu_routing=True,
        )
        fcomm = layout.make_fanout_comm(comm)
        role = layout.role_of(rank)
        if role == RankRole.SAVER:
            return saver_fn(rank, comm) if saver_fn else "saver-idle"
        model = f"model-{rank}"
        if role == RankRole.HEAD:
            fo = WalkerFanout(fcomm, layout, rank, model=model)
            fo.enter_stage("pe", "pe_kind")
            try:
                return head_fn(fo, layout)
            finally:
                fo.stop()
        stubs[rank] = _StubMove()
        service = ComputeService(fcomm, layout, rank, registry={"stub": stubs[rank]}, model=model)
        return service.serve()

    return world.run(fn), stubs


class FanoutFakeCommTest(unittest.TestCase):
    def test_round_trip_in_walker_order_and_stage_stamp(self):
        def head(fo, layout):
            def payload(rank, w0, w1):
                return {"x": np.arange(w0, w1)}

            def body(p, model):
                return {"rank_sum": float(np.sum(p["x"])), "model": model}

            out = fo.run(
                "score", move="stub", per_rank_payload=payload, local_body=body, merge=lambda r: r
            )
            out2 = fo.run(
                "score", move="stub", per_rank_payload=payload, local_body=body, merge=lambda r: r
            )
            return (
                {r: rep["rank_sum"] for r, rep in out.items()},
                out[0]["model"],
                out[1]["model"],
                out2 == out,
            )

        (out, stubs) = _run_world(3, [0, 0, 0], 4, head)
        sums, m0, m1, same = out[0]
        self.assertEqual(sums, {0: 1.0, 1: 5.0})
        self.assertEqual((m0, m1), ("model-0", "model-1"))
        self.assertTrue(same)
        self.assertEqual(out[1], 2)  # the worker served two commands before stop
        self.assertEqual(out[2], "saver-idle")
        self.assertEqual(stubs[1].calls, [("score", "pe_kind", 1), ("score", "pe_kind", 2)])

    def test_worker_error_surfaces_on_the_head_with_traceback(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "boom",
                    move="stub",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            err = cm.exception
            return err.rank, err.op, err.move, "kaboom" in err.remote_traceback

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], (1, "boom", "stub", True))
        self.assertEqual(out[1], 1)

    def test_unpicklable_result_becomes_a_senderror_reply_not_a_hang(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "unpicklable",
                    move="stub",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            return cm.exception.rank, cm.exception.error["type"]

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], (1, "SendError"))
        self.assertEqual(out[1], 1)  # the worker still served the command and stopped cleanly

    def test_unknown_move_is_an_error_reply_not_a_hang(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "score",
                    move="nope",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            return cm.exception.error["type"]

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], "KeyError")

    def test_registry_resolves_stage_and_name_before_the_bare_name(self):
        # run.py's serve registry is keyed (stage, name): one move name used in two
        # stages addresses two different objects; a bare-name entry is the fallback.
        world = FakeWorld(2, nodes=[0, 0])
        stubs = {}

        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if rank == layout.head_rank:
                fo = WalkerFanout(fcomm, layout, rank, model="m")
                kw = dict(
                    per_rank_payload=lambda r, w0, w1: {"x": np.ones(1)},
                    local_body=lambda p, m: {"rank_sum": 1.0},
                    merge=lambda r: r,
                )
                try:
                    fo.enter_stage("noise_search", "search")
                    fo.run("score", move="stub", **kw)
                    fo.enter_stage("full_pe", "pe")
                    fo.run("score", move="stub", **kw)
                    fo.run("score", move="other", **kw)
                finally:
                    fo.stop()
                return "head"
            stubs.update(early=_StubMove(), late=_StubMove(), bare=_StubMove())
            registry = {
                ("noise_search", "stub"): stubs["early"],
                ("full_pe", "stub"): stubs["late"],
                "other": stubs["bare"],
            }
            return ComputeService(fcomm, layout, rank, registry=registry, model="m").serve()

        out = world.run(fn)
        self.assertEqual(out[1], 3)
        self.assertEqual([c[1] for c in stubs["early"].calls], ["search"])
        self.assertEqual([c[1] for c in stubs["late"].calls], ["pe"])
        self.assertEqual([c[1] for c in stubs["bare"].calls], ["pe"])

    def test_one_shared_object_registered_under_two_stage_keys_serves_both(self):
        # The production case run.py's step-list registry exists for: a stock
        # runtime move is ONE object listed by two stages (``psd_pe`` in
        # noise_search AND full_pe), so both (stage, name) keys map to it and
        # the head's commands in EITHER stage must be served.
        world = FakeWorld(2, nodes=[0, 0])
        stubs = {}

        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if rank == layout.head_rank:
                fo = WalkerFanout(fcomm, layout, rank, model="m")
                kw = dict(
                    per_rank_payload=lambda r, w0, w1: {"x": np.ones(1)},
                    local_body=lambda p, m: {"rank_sum": 1.0},
                    merge=lambda r: r,
                )
                try:
                    fo.enter_stage("noise_search", "search")
                    fo.run("score", move="stub", **kw)
                    fo.enter_stage("full_pe", "pe")
                    fo.run("score", move="stub", **kw)
                finally:
                    fo.stop()
                return "head"
            stubs["shared"] = _StubMove()
            registry = {
                ("noise_search", "stub"): stubs["shared"],
                ("full_pe", "stub"): stubs["shared"],
                "stub": stubs["shared"],
            }
            return ComputeService(fcomm, layout, rank, registry=registry, model="m").serve()

        out = world.run(fn)
        self.assertEqual(out[1], 2)
        self.assertEqual(len(stubs["shared"].calls), 2)
        self.assertEqual([c[1] for c in stubs["shared"].calls], ["search", "pe"])

    def test_gather_likelihood_concatenates_blocks_in_walker_order(self):
        # Head block (0, 2) scored from its own ACA; worker block (2, 4) answered by
        # the LIKELIHOOD_OP builtin run.py installs on every ComputeService.
        world = FakeWorld(3, nodes=[0, 0, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            role = layout.role_of(rank)
            if role == RankRole.SAVER:
                return "saver-idle"
            if role == RankRole.HEAD:
                fo = WalkerFanout(fcomm, layout, rank, model=None)
                try:
                    return fo.gather_likelihood(_Acs([10.0, 11.0]))
                finally:
                    fo.stop()
            builtins = {
                LIKELIHOOD_OP: lambda payload, clock, model: np.asarray(
                    model.likelihood(complex=False)
                )
            }
            service = ComputeService(
                fcomm, layout, rank, registry={}, model=_Acs([12.0, 13.0]), builtins=builtins
            )
            return service.serve()

        out = world.run(fn)
        np.testing.assert_array_equal(out[0], [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(out[1], 1)

    def test_gather_likelihood_single_rank_is_a_direct_call(self):
        layout = FakeWorld(1).run(lambda r, c: build_layout(c, 4, [0], legacy=False))[0]
        fo = WalkerFanout(None, layout, 0, model=None)
        np.testing.assert_array_equal(
            fo.gather_likelihood(_Acs([1.0, 2.0, 3.0, 4.0])), [1.0, 2.0, 3.0, 4.0]
        )

    def test_saver_never_sees_fanout_traffic(self):
        def saver(rank, comm):
            time.sleep(0.2)
            return comm.iprobe(source=0)

        def head(fo, layout):
            return fo.run(
                "score",
                move="stub",
                per_rank_payload=lambda r, w0, w1: {"x": np.ones(2)},
                local_body=lambda p, m: {"rank_sum": 2.0},
                merge=lambda r: sorted(r),
            )

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head, saver_fn=saver)
        self.assertEqual(out[0], [0, 1])
        self.assertFalse(out[2])

    def test_ping_checks_layout_digests(self):
        (out, _stubs) = _run_world(3, [0, 0, 0], 4, lambda fo, layout: fo.ping())
        self.assertEqual(set(out[0].values()), {out[0][0]})
        self.assertEqual(sorted(out[0]), [0, 1])

    def test_allgather_walker_vector_is_a_setup_phase_collective(self):
        world = FakeWorld(3)

        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return None
            fo = WalkerFanout(fcomm, layout, rank)
            w0, w1 = layout.block_of(rank)
            return fo.allgather_walker_vector(np.arange(w0, w1) * 10.0)

        out = world.run(fn)
        np.testing.assert_array_equal(out[0], [0.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_equal(out[1], out[0])

    def test_concat_blocks_follows_compute_rank_order(self):
        layout = FakeWorld(5, nodes=[0, 1, 0, 1, 0]).run(
            lambda r, c: build_layout(c, 8, [0, 1], legacy=False)
        )[0]
        results = {
            3: np.array([6, 7]),
            0: np.array([0, 1]),
            2: np.array([4, 5]),
            1: np.array([2, 3]),
        }
        np.testing.assert_array_equal(concat_blocks(results, layout), np.arange(8))

    def test_run_on_a_worker_raises(self):
        layout = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[1]
        fo = WalkerFanout(None, layout, 1)
        with self.assertRaises(RuntimeError):
            fo.run(
                "x",
                per_rank_payload=lambda r, w0, w1: None,
                local_body=lambda p, m: None,
                merge=lambda r: r,
            )

    def test_head_body_failure_drains_and_next_run_succeeds(self):
        def head(fo, layout):
            def payload(rank, w0, w1):
                return {"x": np.arange(w0, w1)}

            calls = {"n": 0}

            def body(p, model):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ValueError("head boom")
                return {"rank_sum": float(np.sum(p["x"])), "model": model}

            with self.assertRaises(ValueError) as cm:
                fo.run(
                    "score",
                    move="stub",
                    per_rank_payload=payload,
                    local_body=body,
                    merge=lambda r: r,
                )
            self.assertNotIsInstance(cm.exception, RemoteWorkerError)
            self.assertIn("head boom", str(cm.exception))

            # a second run(), with a normal body, must succeed cleanly: nothing
            # left over in the channel from the failed first run()
            return fo.run(
                "score",
                move="stub",
                per_rank_payload=payload,
                local_body=body,
                merge=lambda r: r,
            )

        (out, stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(
            out[0],
            {
                0: {"rank_sum": 1.0, "model": "model-0"},
                1: {"rank_sum": 5.0, "model": "model-1"},
            },
        )
        # worker served the (successfully-processed) first command AND the second
        self.assertEqual(out[1], 2)

    def test_two_workers_both_fail_first_in_rank_order_and_channel_drains(self):
        def head(fo, layout):
            def payload(rank, w0, w1):
                return {"x": np.arange(w0, w1)}

            def body(p, model):
                return {"rank_sum": float(np.sum(p["x"])), "model": model}

            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "boom",
                    move="stub",
                    per_rank_payload=payload,
                    local_body=body,
                    merge=lambda r: r,
                )
            self.assertEqual(cm.exception.rank, 1)  # first failure in worker order

            out2 = fo.run(
                "score",
                move="stub",
                per_rank_payload=payload,
                local_body=body,
                merge=lambda r: r,
            )
            return sorted(out2)

        (out, stubs) = _run_world(5, [0, 0, 0, 0, 0], 8, head, ranks_per_gpu=2)
        self.assertEqual(out[0], [0, 1, 2, 3])
        for r in (1, 2, 3):
            self.assertEqual(out[r], 2)  # boom command + the following success
        self.assertEqual(out[4], "saver-idle")

    def test_stop_is_idempotent(self):
        def head(fo, layout):
            fo.stop()
            fo.stop()
            return "done"

        (out, stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], "done")
        self.assertEqual(out[1], 0)  # worker served nothing but the (single) stop

    def test_wait_s_recorded(self):
        def head(fo, layout):
            fo.run(
                "score",
                move="stub",
                per_rank_payload=lambda r, w0, w1: {"x": np.arange(w0, w1)},
                local_body=lambda p, m: {"rank_sum": float(np.sum(p["x"]))},
                merge=lambda r: r,
            )
            return fo.last_wait_s

        (out, stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertIsInstance(out[0], float)
        self.assertGreaterEqual(out[0], 0.0)

    def test_init_rejects_wrong_comm_size(self):
        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            if rank == 0:
                with self.assertRaises(ValueError):
                    WalkerFanout(comm, layout, 0)  # WORLD comm (size 3) vs n_compute 2
                return "raised"
            return "ok"

        out = FakeWorld(3).run(fn)
        self.assertEqual(out[0], "raised")


class ReplicaGathersTest(unittest.TestCase):
    def _replica_world(self, head_fn):
        world = FakeWorld(3, nodes=[0, 0, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 1, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            if layout.role_of(rank) == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                try:
                    return head_fn(fo, layout)
                finally:
                    fo.stop()
            service = ComputeService(
                fcomm, layout, rank, registry={}, model=None,
                builtins={RESIDUAL_HASH_OP: lambda p, c, m: f"h{rank}"},
            )
            return service.serve()

        return world.run(fn)

    def test_concat_blocks_returns_the_head_block_in_replica_mode(self):
        out = self._replica_world(lambda fo, layout: concat_blocks({0: [1.5], 1: [9.9]}, layout))
        np.testing.assert_array_equal(out[0], [1.5])

    def test_concat_blocks_reduces_over_block_leads_when_both_axes_exist(self):
        """2 blocks x 2 replicas: one representative per block, in walker order.

        The old code had exactly two branches -- concatenate EVERY compute
        rank, or (replica mode) take the head alone. With both axes live,
        the first counts each walker R times and the second drops every
        block but the head's; only a reduction over block LEADS is right.
        """
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 2, [0, 1], legacy=False,
                                  gpu_routing=True)  # 2 blocks x R=2
            return concat_blocks({0: [1.0], 1: [1.0], 2: [2.0], 3: [2.0]}, layout)

        out = world.run(fn)
        np.testing.assert_array_equal(out[0], [1.0, 2.0])

    def test_allgather_walker_vector_concatenates_blocks_not_replicas(self):
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 2, [0, 1], legacy=False,
                                  gpu_routing=True)  # 2 blocks x R=2
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            w0, _w1 = layout.block_of(rank)
            # every replica of a block holds the SAME walker rows
            return fo.allgather_walker_vector(np.array([100.0 + w0]))

        out = world.run(fn)
        for r in range(4):
            np.testing.assert_array_equal(out[r], [100.0, 101.0])

    def test_group_and_reps_communicators(self):
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 2, [0, 1], legacy=False,
                                  gpu_routing=True)  # 2 blocks x R=2
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                # The saver is NOT in the fan-out comm, and both splits are
                # taken ON the fan-out comm -- so it must not call them. The
                # "every rank calls it the same number of times" rule is
                # scoped to the fan-out comm's members, i.e. compute ranks.
                return "saver"
            gcomm = layout.make_group_comm(fcomm)
            rcomm = layout.make_reps_comm(fcomm)
            return (
                int(gcomm.Get_size()), int(gcomm.Get_rank()),
                (int(rcomm.Get_size()), int(rcomm.Get_rank()))
                if layout.is_block_lead(rank) else None,
            )

        out = world.run(fn)
        # each group holds R == 2 ranks, lead first
        self.assertEqual(out[0][:2], (2, 0))
        self.assertEqual(out[1][:2], (2, 1))
        self.assertEqual(out[2][:2], (2, 0))
        self.assertEqual(out[3][:2], (2, 1))
        # the representatives comm holds one rank per block, in block order
        self.assertEqual(out[0][2], (2, 0))
        self.assertEqual(out[2][2], (2, 1))
        self.assertIsNone(out[1][2])
        self.assertIsNone(out[3][2])

    def test_allgather_walker_vector_returns_the_head_vector_everywhere(self):
        world = FakeWorld(3, nodes=[0, 0, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 1, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            return fo.allgather_walker_vector(np.array([10.0 + rank]))

        out = world.run(fn)
        np.testing.assert_array_equal(out[0], [10.0])
        np.testing.assert_array_equal(out[1], [10.0])

    def test_gather_residual_hashes_and_digest_line(self):
        acs = _Acs([1.0])
        out = self._replica_world(lambda fo, layout: fo.gather_residual_hashes(acs))
        hashes = out[0]
        self.assertEqual(set(hashes), {0, 1})
        self.assertEqual(hashes[1], "h1")
        self.assertEqual(hashes[0], residual_hash(acs))
        line = fanout_digest_line(3, _DigestState(), residual_hashes={0: "aa", 1: "aa"})
        self.assertIn("residual=r0:aa,r1:aa replicas_agree=True", line)
        line = fanout_digest_line(3, _DigestState(), residual_hashes={0: "aa", 1: "bb"})
        self.assertIn("replicas_agree=False", line)

    def test_replicas_agree_is_per_block_at_four_walkers_two_replicas(self):
        """``[FANOUT_DIGEST] replicas_agree`` under 4 walker blocks x R = 2.

        Different blocks hold different walkers, so their residual hashes
        differ BY CONSTRUCTION; before 2026-10-15 the whole map was compared
        and every digest line of a both-axes run read ``replicas_agree=False``
        -- the G4 parity gate's pass line could never pass."""
        layout = FakeWorld(9, nodes=[0, 1, 2, 3, 0, 1, 2, 3, 0]).run(
            lambda r, c: build_layout(c, 4, [0, 1], legacy=False, gpu_routing=True))[0]
        self.assertEqual((layout.n_blocks, layout.ranks_per_block), (4, 2))
        per_block = {r: f"b{r // 2}" for r in range(8)}
        line = fanout_digest_line(3, _DigestState(), residual_hashes=per_block,
                                  layout=layout)
        self.assertIn("replicas_agree=True", line)
        drift = {**per_block, 5: "zz"}            # block 2's second replica drifted
        line = fanout_digest_line(3, _DigestState(), residual_hashes=drift,
                                  layout=layout)
        self.assertIn("replicas_agree=False", line)
        # no layout: the whole-map rule of one-walker replica mode, unchanged
        self.assertFalse(replicas_agree(per_block))
        self.assertTrue(replicas_agree({0: "aa", 1: "aa"}))

    def test_residual_hash_covers_data_and_psd_buffers(self):
        """F2: residual_hash must catch a noise-model-only drift too."""
        class _AcsWithBuffers:
            def __init__(self, data, psd):
                self.linear_data_arr = [data]
                self.linear_psd_arr = [psd]

        data = np.arange(4.0)
        psd = np.ones(4)
        acs = _AcsWithBuffers(data, psd)

        expected = _sha1_16(_array_bytes(data) + _array_bytes(psd))
        self.assertEqual(residual_hash(acs), expected)

        acs_diff_data = _AcsWithBuffers(np.arange(4.0) + 1.0, psd)
        self.assertNotEqual(residual_hash(acs), residual_hash(acs_diff_data))

        acs_diff_psd = _AcsWithBuffers(data, psd * 2.0)
        self.assertNotEqual(residual_hash(acs), residual_hash(acs_diff_psd))


if __name__ == "__main__":
    unittest.main()
