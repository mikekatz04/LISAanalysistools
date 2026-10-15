"""Rank roles + walker-block layout resolved identically on every rank (FakeComm)."""

import os
import unittest
from unittest import mock

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import (
    RankRole,
    build_layout,
    derive_rank_seed,
    prepare_rank,
    rank_build_seed,
    rank_tag,
    resolve_roles,
    select_rank_device,
)


def _layouts(world, nwalkers, pool, **kwargs):
    # gpu_routing defaults ON here and OFF in build_layout: the unified rule
    # is what this module is ABOUT, while the shipped default is opt-in (see
    # GpuRoutingOptInTest, which pins the gate itself). Pinning it here also
    # keeps these tests independent of a GF_GPU_ROUTING left in the shell.
    # It cannot mask a regression on the legacy shapes, because ON and OFF
    # agree on every shape the legacy rule accepts -- which is itself an
    # assertion below.
    kwargs.setdefault("gpu_routing", True)
    return world.run(lambda r, c: build_layout(c, nwalkers, pool, legacy=False, **kwargs))


class ResolveRolesTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(resolve_roles(1), (0, 0, (0,)))
        self.assertEqual(resolve_roles(2), (0, 0, (0, 1)))
        self.assertEqual(resolve_roles(3), (0, 2, (0, 1)))
        self.assertEqual(resolve_roles(5), (0, 4, (0, 1, 2, 3)))
        with self.assertRaises(ValueError):
            resolve_roles(0)


class BuildLayoutTest(unittest.TestCase):
    def test_two_nodes_one_rank_per_gpu(self):
        # cyclic placement: ranks 0,2,4 on node0, ranks 1,3 on node1; rank 4 = saver
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])
        outs = _layouts(world, 8, [0, 1])
        lay = outs[0]
        for r in range(5):
            self.assertEqual(outs[r].describe(), lay.describe())
            self.assertEqual(outs[r].digest(), lay.digest())
        self.assertEqual(lay.compute_ranks, (0, 1, 2, 3))
        self.assertEqual(lay.saver_rank, 4)
        self.assertEqual(lay.worker_ranks, (1, 2, 3))
        self.assertEqual(lay.block, 2)
        self.assertFalse(lay.is_single())
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(lay.placements[2].devices, (1,))
        self.assertEqual(lay.placements[1].devices, (0,))
        self.assertEqual(lay.placements[3].devices, (1,))
        blocks = [lay.block_of(r) for r in lay.compute_ranks]
        self.assertEqual(blocks, [(0, 2), (2, 4), (4, 6), (6, 8)])
        self.assertEqual(lay.role_of(4), RankRole.SAVER)
        self.assertEqual(lay.block_of(4), (0, 0))
        self.assertEqual(lay.local_gpus(3), [1])
        self.assertEqual(lay.fanout_rank(3), 3)
        self.assertEqual(lay.placements[4].local_index, 2)

    def test_ranks_per_gpu_share_one_device(self):
        world = FakeWorld(3)  # head + one compute rank + saver, ONE GPU
        lay = _layouts(world, 6, [0], ranks_per_gpu=2)[0]
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(lay.placements[1].devices, (0,))
        self.assertEqual((lay.placements[0].device_slot, lay.placements[1].device_slot), (0, 1))
        self.assertEqual(lay.ranks_on_device("node0", 0), (0, 1))
        self.assertEqual(lay.block, 3)

    def test_gpus_per_rank_owns_two_devices(self):
        world = FakeWorld(2)  # saver aliased to the head: compute = (0, 1)
        lay = _layouts(world, 4, [0, 1, 2, 3], gpus_per_rank=2)[0]
        self.assertEqual(lay.placements[0].devices, (0, 1))
        self.assertEqual(lay.placements[1].devices, (2, 3))
        self.assertEqual(lay.local_gpus(1), [2, 3])

    def test_errors(self):
        world = FakeWorld(3)
        # NOTE (2026-09-23): ``nwalkers % n_compute != 0`` is NO LONGER an
        # error -- the unified factorization resolves every pair (7 walkers
        # on 2 ranks = one block of 7, replicated). It warns instead; see
        # ReplicaModeTest.test_non_divisible_now_resolves_instead_of_raising.
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0])  # two compute ranks on one GPU without ranks_per_gpu
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0, 1], gpus_per_rank=2, ranks_per_gpu=2)
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0, 1], gpus_per_rank=0)

    def test_single_rank_cpu(self):
        lay = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertTrue(lay.is_single())
        self.assertIsNone(lay.local_gpus(0))
        self.assertEqual(lay.block_of(0), (0, 4))
        self.assertEqual(lay.role_of(0), RankRole.HEAD)
        self.assertEqual(lay.saver_rank, 0)

    def test_legacy_layout_keeps_todays_roles(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 5, [0, 1], legacy=True))[0]
        self.assertTrue(lay.legacy)
        self.assertEqual(lay.compute_ranks, (0,))
        self.assertEqual(lay.block, 5)
        self.assertEqual(lay.placements[0].devices, (0, 1))
        self.assertEqual(lay.role_of(1), RankRole.SPARE)
        self.assertEqual(lay.role_of(2), RankRole.SAVER)

    def test_legacy_ranks_on_device_excludes_saver_and_spare(self):
        # head owns both devices in legacy mode; the spare (rank 1, device 1)
        # and the saver (rank 2, device 0) must not show up as co-occupants.
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 5, [0, 1], legacy=True))[0]
        self.assertEqual(lay.ranks_on_device("node0", 0), (0,))
        self.assertEqual(lay.ranks_on_device("node0", 1), (0,))

    def test_size2_single_gpu_demotes_rank1_to_saver_with_warning(self):
        import warnings

        world = FakeWorld(2)

        def fn(rank, comm):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lay = build_layout(comm, 4, [0], legacy=False)
            return lay, [str(w.message) for w in caught]

        out = world.run(fn)
        lay, messages = out[0]
        self.assertEqual(lay.compute_ranks, (0,))
        self.assertEqual(lay.saver_rank, 1)
        self.assertTrue(lay.is_single())
        self.assertEqual(lay.role_of(1), RankRole.SAVER)
        self.assertEqual(lay.block_of(0), (0, 4))
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(len(lay.notes), 1)
        self.assertIn("RANKS_PER_GPU=2", lay.notes[0])
        self.assertIn("-n 1", lay.notes[0])
        self.assertTrue(any("dedicated saver" in m for m in messages))
        self.assertIn("dedicated saver", lay.describe())
        self.assertEqual(out[1][0].describe(), lay.describe())

    def test_size2_with_enough_gpus_keeps_two_compute_ranks(self):
        lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.notes, ())

    def test_size2_ranks_per_gpu_2_on_one_gpu_keeps_two_compute_ranks(self):
        lay = FakeWorld(2).run(
            lambda r, c: build_layout(c, 4, [0], legacy=False, ranks_per_gpu=2)
        )[0]
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.notes, ())

    def test_size3_single_gpu_still_hard_errors(self):
        with self.assertRaises(RuntimeError):
            _layouts(FakeWorld(3), 4, [0])


class SeedTest(unittest.TestCase):
    def test_distinct_and_deterministic(self):
        lay = FakeWorld(5, nodes=[0, 1, 0, 1, 0]).run(
            lambda r, c: build_layout(c, 8, [0, 1], legacy=False)
        )[0]
        seeds = [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(seeds, [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks])
        self.assertNotEqual(seeds, [derive_rank_seed(103210, lay, r) for r in lay.compute_ranks])


class _SeedFit:
    """The three attributes ``rank_build_seed`` reads off a StockGlobalFit."""

    def __init__(self, random_seed, layout=None, rank=None):
        self.general = type("G", (), {"random_seed": random_seed})()
        if layout is not None:
            self.rank_layout = layout
        if rank is not None:
            self.rank = rank


class RankBuildSeedTest(unittest.TestCase):
    """Build-time seeds for prior/proposal objects: per rank, never shared."""

    def _layout(self):
        return FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]

    def test_no_run_seed_stays_on_entropy(self):
        self.assertIsNone(rank_build_seed(_SeedFit(None)))
        self.assertIsNone(rank_build_seed(_SeedFit(None, self._layout(), 1)))

    def test_no_layout_is_the_domain_separated_run_seed(self):
        # single process / fit.sample(): prepare_rank never ran. Still NOT the
        # bare run seed -- that integer is what _seed_rank_streams feeds
        # np.random.seed() (domain tag 0xB01D).
        got = rank_build_seed(_SeedFit(103209))
        self.assertEqual(got, rank_build_seed(_SeedFit(103209)))  # deterministic
        self.assertNotEqual(got, 103209)

    def test_each_compute_rank_gets_its_own_seed(self):
        lay = self._layout()
        seeds = [rank_build_seed(_SeedFit(103209, lay, r)) for r in lay.compute_ranks]
        self.assertEqual(len(set(seeds)), len(lay.compute_ranks))
        self.assertNotIn(103209, seeds)  # never the bare run seed
        # ... and never the GLOBAL-stream sub-seed of the same rank either:
        # the two domains must not coincide by construction, not by luck
        for rank in lay.compute_ranks:
            self.assertNotEqual(
                rank_build_seed(_SeedFit(103209, lay, rank)),
                derive_rank_seed(103209, lay, rank),
            )

    def test_the_same_rank_twice_is_the_same_seed(self):
        lay = self._layout()
        self.assertEqual(
            rank_build_seed(_SeedFit(103209, lay, 1)), rank_build_seed(_SeedFit(103209, lay, 1))
        )
        self.assertNotEqual(
            rank_build_seed(_SeedFit(103209, lay, 1)), rank_build_seed(_SeedFit(103210, lay, 1))
        )

    def test_a_non_compute_rank_falls_back_to_the_head(self):
        lay = self._layout()  # rank 2 is the saver: no walker block
        self.assertEqual(
            rank_build_seed(_SeedFit(103209, lay, lay.saver_rank)),
            rank_build_seed(_SeedFit(103209, lay, lay.head_rank)),
        )

    def test_a_layout_without_a_stamped_rank_reads_as_the_head(self):
        lay = self._layout()
        self.assertEqual(
            rank_build_seed(_SeedFit(103209, lay)),
            rank_build_seed(_SeedFit(103209, lay, lay.head_rank)),
        )


class SelectRankDeviceTest(unittest.TestCase):
    def _layout(self):
        return FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]

    def test_visible_mode_narrows_env_and_renumbers(self):
        env = {}
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_visible_mode_maps_through_a_prenarrowed_env(self):
        env = {"CUDA_VISIBLE_DEVICES": "2,3"}
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")

    def test_no_runtime_probe_still_counts_as_visible(self):
        env = {}
        gpus, mode = select_rank_device(
            self._layout(),
            0,
            environ=env,
            device_count_fn=lambda: None,
            set_device_fn=lambda d: None,
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_fallback_setdevice_when_the_runtime_is_already_up(self):
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        pinned = []
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 2, set_device_fn=pinned.append
        )
        self.assertEqual((gpus, mode), ([1], "setdevice"))
        self.assertEqual(pinned, [1])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")

    def test_cpu_and_legacy(self):
        cpu = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertEqual(select_rank_device(cpu, 0, environ={}), (None, "cpu"))
        legacy = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=True))[0]
        env = {}
        self.assertEqual(select_rank_device(legacy, 0, environ=env), ([0, 1], "legacy"))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_pool_index_outside_visible_set_raises(self):
        env = {"CUDA_VISIBLE_DEVICES": "5"}
        with self.assertRaises(ValueError) as cm:
            select_rank_device(
                self._layout(),
                1,
                environ=env,
                device_count_fn=lambda: 1,
                set_device_fn=lambda d: None,
            )
        self.assertIn("gpu-bind", str(cm.exception))

    def test_empty_visible_env_raises_actionable_error(self):
        env = {"CUDA_VISIBLE_DEVICES": ""}
        with self.assertRaises(ValueError) as cm:
            select_rank_device(
                self._layout(),
                1,  # rank 1 has devices in this layout
                environ=env,
                device_count_fn=lambda: 1,
                set_device_fn=lambda d: None,
            )
        msg = str(cm.exception)
        self.assertIn("CUDA_VISIBLE_DEVICES", msg)
        self.assertIn("gpu-bind", msg)


class _General:
    def __init__(self):
        self.nwalkers = 4
        self.gpus = [0, 1]
        self.gpus_per_rank = 1
        self.ranks_per_gpu = 1


class _Fit:
    main_rank = 0
    built = False

    def __init__(self):
        self.general = _General()


class PrepareRankTest(unittest.TestCase):
    def test_sets_local_gpus_layout_and_mode_on_every_rank(self):
        def fn(rank, comm):
            fit = _Fit()
            env = {}
            lay = prepare_rank(
                fit, comm, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
            )
            again = prepare_rank(fit, comm, environ=env)  # idempotent: no second layout
            return (
                fit.general.gpus,
                fit.rank_device_mode,
                env.get("CUDA_VISIBLE_DEVICES"),
                lay.block_of(rank),
                fit.rank_layout is lay and again is lay,
                rank_tag(lay, rank),
            )

        out = FakeWorld(3).run(fn)
        self.assertEqual(out[0], ([0], "visible", "0", (0, 2), True, "r0/head"))
        self.assertEqual(out[1], ([0], "visible", "1", (2, 4), True, "r1/c1"))
        self.assertEqual(out[2][:3], ([0], "visible", "0"))
        self.assertEqual(out[2][5], "r2/saver")

    def test_stamps_the_rank_for_build_time_helpers(self):
        # GlobalFit.__init__ sets fit.rank too, but that is AFTER the build;
        # rank_build_seed runs inside it
        def fn(rank, comm):
            fit = _Fit()
            prepare_rank(fit, comm, environ={}, device_count_fn=lambda: 1)
            return fit.rank

        out = FakeWorld(3).run(fn)
        self.assertEqual((out[0], out[1], out[2]), (0, 1, 2))

    def test_refuses_to_run_after_build_with_several_ranks(self):
        def fn(rank, comm):
            fit = _Fit()
            fit.built = True
            return prepare_rank(fit, comm, environ={}, device_count_fn=lambda: 1)

        with self.assertRaises(RuntimeError):
            FakeWorld(2).run(fn)

    def test_cpu_fit_keeps_gpus_none(self):
        def fn(rank, comm):
            fit = _Fit()
            fit.general.gpus = None
            prepare_rank(fit, comm, environ={})
            return fit.general.gpus, fit.rank_device_mode

        self.assertEqual(FakeWorld(2).run(fn)[1], (None, "cpu"))


class AutoGpusPerRankTest(unittest.TestCase):
    """gpus_per_rank=None: a lone compute rank on a node owns the whole pool (today's -n 1)."""

    def test_single_rank_owns_the_whole_pool(self):
        lay = FakeWorld(1).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.local_gpus(0), [0, 1])
        self.assertEqual(lay.n_compute, 1)

    def test_single_rank_explicit_one_narrows_to_the_first_device(self):
        lay = FakeWorld(1).run(
            lambda r, c: build_layout(c, 4, [0, 1], legacy=False, gpus_per_rank=1)
        )[0]
        self.assertEqual(lay.local_gpus(0), [0])

    def test_two_compute_ranks_on_a_two_gpu_node_get_one_device_each(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual((lay.local_gpus(0), lay.local_gpus(1)), ([0], [1]))

    def test_size2_on_one_gpu_still_demotes_rank1_and_head_owns_the_pool(self):
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0], legacy=False))[0]
        self.assertEqual(lay.role_of(1), RankRole.SAVER)
        self.assertEqual(lay.local_gpus(0), [0])

    def test_head_plus_saver_on_two_gpus_head_owns_both(self):
        # -n 2 on a 2-GPU pool: rank 1 is a COMPUTE rank (two compute ranks), not a saver
        lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.n_compute, 2)
        self.assertEqual((lay.local_gpus(0), lay.local_gpus(1)), ([0], [1]))

    def test_prepare_rank_passes_none_through(self):
        # the fixture's general block leaves gpus_per_rank unset -> AUTO
        fit = _Fit()
        fit.general.gpus_per_rank = None
        lay = FakeWorld(1).run(
            lambda r, c: prepare_rank(fit, c, environ={}, device_count_fn=lambda: 2)
        )[0]
        self.assertEqual(lay.local_gpus(0), [0, 1])
        self.assertEqual(fit.general.gpus, [0, 1])


class LayoutDryRunTest(unittest.TestCase):
    def test_prints_and_returns_true_only_when_armed(self):
        from lisatools.globalfit.communication.ranks import layout_dry_run

        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        lines = []
        armed = {"GF_LAYOUT_DRY_RUN": "1"}
        self.assertTrue(layout_dry_run(lay, None, environ=armed, out=lines.append))
        self.assertTrue(any("head" in ln for ln in lines))
        self.assertFalse(layout_dry_run(lay, None, environ={}, out=lines.append))


class ReplicaModeTest(unittest.TestCase):
    def test_one_walker_on_two_compute_ranks_is_replica_mode(self):
        world = FakeWorld(3, nodes=[0, 0, 0])
        outs = _layouts(world, 1, [0, 1])
        lay = outs[0]
        self.assertTrue(lay.replica_mode)
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.n_replicas, 2)
        self.assertEqual(lay.block, 1)
        for r in lay.compute_ranks:
            self.assertEqual(lay.block_of(r), (0, 1))
        self.assertEqual([lay.replica_index(r) for r in lay.compute_ranks], [0, 1])
        self.assertIn("REPLICAS", lay.describe())
        for r in range(3):
            self.assertEqual(outs[r].digest(), lay.digest())

    def test_one_walker_across_two_nodes_round_robin(self):
        # campaign layout (c): `GPUS=0 mpiexec -n 3 -ppn 1` -- head + saver on
        # node0, the second replica alone on node1 with that node's one-device
        # pool (the runbook's GPUS=0 pin keeps AUTO gpus_per_rank at 1 there)
        world = FakeWorld(3, nodes=[0, 1, 0])
        outs = _layouts(world, 1, [0])
        lay = outs[0]
        self.assertTrue(lay.replica_mode)
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.saver_rank, 2)
        self.assertEqual(lay.n_replicas, 2)
        self.assertEqual([lay.block_of(r) for r in lay.compute_ranks], [(0, 1), (0, 1)])
        self.assertEqual([lay.replica_index(r) for r in lay.compute_ranks], [0, 1])
        self.assertEqual((lay.placements[0].node, lay.placements[1].node), ("node0", "node1"))
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(lay.placements[1].devices, (0,))
        self.assertEqual(lay.placements[1].local_index, 0)  # first process on its node
        self.assertEqual(lay.ranks_on_device("node0", 0), (0,))
        self.assertEqual(lay.ranks_on_device("node1", 0), (1,))
        self.assertIn("REPLICAS", lay.describe())
        for r in range(3):
            self.assertEqual(outs[r].describe(), lay.describe())
            self.assertEqual(outs[r].digest(), lay.digest())

    def test_one_walker_four_replicas_on_two_nodes(self):
        # the NGPUS=4 dispatch: 2 nodes x 2 GPUs, cyclic placement, rank 4 = saver
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])
        outs = _layouts(world, 1, [0, 1])
        lay = outs[0]
        self.assertTrue(lay.replica_mode)
        self.assertEqual(lay.n_replicas, 4)
        self.assertEqual(lay.compute_ranks, (0, 1, 2, 3))
        self.assertEqual(lay.saver_rank, 4)
        self.assertEqual([lay.block_of(r) for r in lay.compute_ranks], [(0, 1)] * 4)
        self.assertEqual([lay.replica_index(r) for r in lay.compute_ranks], [0, 1, 2, 3])
        self.assertEqual([lay.placements[r].devices for r in range(4)], [(0,), (0,), (1,), (1,)])
        self.assertEqual([lay.placements[r].node for r in range(5)],
                         ["node0", "node1", "node0", "node1", "node0"])
        self.assertEqual(lay.block_of(4), (0, 0))
        for r in range(5):
            self.assertEqual(outs[r].digest(), lay.digest())

    def test_more_walkers_is_unchanged(self):
        lay = _layouts(FakeWorld(3), 4, [0, 1])[0]
        self.assertFalse(lay.replica_mode)
        self.assertEqual(lay.n_replicas, 1)
        self.assertEqual(lay.block_of(0), (0, 2))
        self.assertEqual(lay.block_of(1), (2, 4))
        self.assertEqual(lay.replica_index(1), 0)
        self.assertNotIn("REPLICAS", lay.describe())

    def test_one_walker_one_compute_rank_is_single(self):
        lay = _layouts(FakeWorld(1), 1, [0])[0]
        self.assertFalse(lay.replica_mode)
        self.assertTrue(lay.is_single())
        self.assertEqual(lay.block_of(0), (0, 1))

    def test_escape_hatch_refuses_replica_mode(self):
        with mock.patch.dict(os.environ, {"GF_ONE_WALKER_REPLICAS": "0"}):
            with self.assertRaises(RuntimeError):  # FakeWorld re-raises rank failures
                _layouts(FakeWorld(3), 1, [0, 1])

    def test_truthy_string_still_selects_replica_mode(self):
        """F7: parsed like the likelihood-fanout knob, not a bare '== "1"' check."""
        with mock.patch.dict(os.environ, {"GF_ONE_WALKER_REPLICAS": "true"}):
            lay = _layouts(FakeWorld(3, nodes=[0, 0, 0]), 1, [0, 1])[0]
            self.assertTrue(lay.replica_mode)

    def test_falsy_string_refuses_replica_mode(self):
        with mock.patch.dict(os.environ, {"GF_ONE_WALKER_REPLICAS": "false"}):
            with self.assertRaises(RuntimeError):  # FakeWorld re-raises rank failures
                _layouts(FakeWorld(3), 1, [0, 1])

    def test_non_divisible_now_resolves_instead_of_raising(self):
        """BEHAVIOUR CHANGE 2026-09-23 (unified factorization).

        ``nwalkers % n_compute`` used to be a hard error. Under
        ``n_compute = n_blocks x R`` with ``n_blocks = gcd(...)`` every pair
        resolves: 3 walkers on 2 ranks is one block of 3, replicated twice.
        It is legal but rarely what was meant, so it must WARN.
        """
        lay = _layouts(FakeWorld(3), 3, [0, 1])[0]
        self.assertEqual((lay.n_blocks, lay.ranks_per_block, lay.block), (1, 2, 3))
        self.assertTrue(any("one block" in n or "gcd" in n for n in lay.notes))


class UnifiedFactorizationTest(unittest.TestCase):
    """``n_compute = n_blocks x R`` for ANY (nwalkers, n_compute) pair.

    Blocks are the cheap axis (GB is sublinear in block width, and the GB
    buffer saturates above ~2.5 walkers); replicas cost a full ACA and
    inverse-PSD per rank. AUTO therefore MAXIMIZES blocks:
    ``n_blocks = gcd(nwalkers, n_compute)``, ``R = n_compute // n_blocks``.
    """

    def _lay(self, size, nwalkers, pool=(0, 1), nodes=None, **kw):
        return _layouts(FakeWorld(size, nodes=nodes), nwalkers, list(pool), **kw)[0]

    def test_auto_factorization_table(self):
        # (size, nwalkers, pool) -> (n_blocks, R, block)
        cases = [
            ((3, 4, [0, 1]), (2, 1, 2)),        # 2 compute ranks, 4 walkers: today
            ((5, 8, [0, 1]), (4, 1, 2)),        # 4 compute ranks, 8 walkers: today
            # one-walker replica mode: today. Pool [0] -- the 2-node x gpu:1
            # shape the replica gates run. With [0, 1] and nodes [0, 1, 0]
            # each node's LONE compute rank takes both devices (AUTO) for its
            # one walker: the job-764 empty-shard trap, refused since
            # 2026-10-15 (DevicesBeyondWalkersTest).
            ((3, 1, [0]), (1, 2, 1)),
            ((5, 1, [0, 1]), (1, 4, 1)),        # one walker, 4 replicas: today
            ((5, 2, [0, 1]), (2, 2, 1)),        # NEW: GPUs > walkers, 2 blocks x 2
            ((5, 4, [0, 1]), (4, 1, 1)),        # 4 walkers on 4 ranks: today
        ]
        for (size, nw, pool), expect in cases:
            with self.subTest(size=size, nwalkers=nw):
                lay = self._lay(size, nw, pool, nodes=[0, 1, 0, 1, 0][:size])
                self.assertEqual(
                    (lay.n_blocks, lay.ranks_per_block, lay.block), expect)
                self.assertEqual(lay.n_blocks * lay.ranks_per_block, lay.n_compute)
                self.assertEqual(lay.n_blocks * lay.block, lay.nwalkers)

    def test_gpus_greater_than_walkers_pairs_blocks_and_replicas(self):
        # 4 compute ranks, 2 walkers -> 2 blocks of 1 walker, 2 replicas each.
        lay = self._lay(5, 2, [0, 1], nodes=[0, 1, 0, 1, 0])
        self.assertEqual(lay.compute_ranks, (0, 1, 2, 3))
        self.assertEqual([lay.block_of(r) for r in lay.compute_ranks],
                         [(0, 1), (0, 1), (1, 2), (1, 2)])
        self.assertEqual([lay.block_index(r) for r in lay.compute_ranks], [0, 0, 1, 1])
        self.assertEqual([lay.replica_index(r) for r in lay.compute_ranks], [0, 1, 0, 1])
        self.assertEqual(lay.ranks_in_block(0), (0, 1))
        self.assertEqual(lay.ranks_in_block(1), (2, 3))
        self.assertEqual(lay.block_leads, (0, 2))
        self.assertTrue(lay.is_block_lead(0) and lay.is_block_lead(2))
        self.assertFalse(lay.is_block_lead(1) or lay.is_block_lead(3))
        self.assertTrue(lay.replica_mode)
        self.assertEqual(lay.n_replicas, 2)

    def test_owner_of_returns_the_block_lead_and_owners_of_the_group(self):
        lay = self._lay(5, 2, [0, 1], nodes=[0, 1, 0, 1, 0])
        self.assertEqual(lay.owner_of(0), (0, 0))   # walker 0 -> block 0's lead
        self.assertEqual(lay.owner_of(1), (2, 0))   # walker 1 -> block 1's lead
        self.assertEqual(lay.owners_of(1), ((2, 3), 0))
        with self.assertRaises(ValueError):
            lay.owner_of(2)

    # ---- endpoint identity: both of today's regimes must be reproduced ----
    def test_endpoint_r_equals_one_matches_todays_block_layout(self):
        lay = self._lay(5, 8, [0, 1], nodes=[0, 1, 0, 1, 0])
        self.assertEqual(lay.ranks_per_block, 1)
        self.assertFalse(lay.replica_mode)
        self.assertEqual(lay.n_replicas, 1)
        for i, r in enumerate(lay.compute_ranks):
            self.assertEqual(lay.block_of(r), (i * 2, (i + 1) * 2))
            self.assertEqual(lay.replica_index(r), 0)
            self.assertEqual(lay.block_index(r), i)
            self.assertEqual(lay.owner_of(i * 2), (r, 0))
            self.assertTrue(lay.is_block_lead(r))

    def test_endpoint_one_walker_matches_todays_replica_mode(self):
        lay = self._lay(5, 1, [0, 1], nodes=[0, 1, 0, 1, 0])
        self.assertEqual((lay.n_blocks, lay.ranks_per_block), (1, 4))
        self.assertTrue(lay.replica_mode)
        self.assertEqual(lay.n_replicas, 4)
        for i, r in enumerate(lay.compute_ranks):
            self.assertEqual(lay.block_of(r), (0, 1))
            self.assertEqual(lay.replica_index(r), i)   # == fanout_rank, as before
            self.assertEqual(lay.block_index(r), 0)
        self.assertIn("REPLICAS", lay.describe())

    def test_single_compute_rank_is_untouched(self):
        # the lite / laptop path: one process owns every walker, no replicas
        lay = _layouts(FakeWorld(1), 10, [])[0]
        self.assertTrue(lay.is_single())
        self.assertEqual((lay.n_blocks, lay.ranks_per_block, lay.block), (1, 1, 10))
        self.assertFalse(lay.replica_mode)
        self.assertEqual(lay.block_of(0), (0, 10))

    # ---- explicit R ----
    def test_explicit_ranks_per_block_overrides_auto(self):
        lay = self._lay(5, 4, [0, 1], nodes=[0, 1, 0, 1, 0], ranks_per_block=4)
        self.assertEqual((lay.n_blocks, lay.ranks_per_block, lay.block), (1, 4, 4))
        self.assertFalse(lay.ranks_per_block_auto)

    def test_explicit_r_must_divide_the_compute_rank_count(self):
        with self.assertRaises(RuntimeError):
            self._lay(5, 4, [0, 1], nodes=[0, 1, 0, 1, 0], ranks_per_block=3)

    def test_explicit_r_must_leave_divisible_blocks(self):
        # 4 compute ranks, R=2 -> 2 blocks, but 3 walkers do not split in 2
        with self.assertRaises(RuntimeError):
            self._lay(5, 3, [0, 1], nodes=[0, 1, 0, 1, 0], ranks_per_block=2)

    def test_escape_hatch_refuses_any_replication(self):
        with mock.patch.dict(os.environ, {"GF_ONE_WALKER_REPLICAS": "0"}):
            with self.assertRaises(RuntimeError):
                self._lay(5, 2, [0, 1], nodes=[0, 1, 0, 1, 0])

    def test_describe_and_digest_agree_on_every_rank(self):
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])
        outs = _layouts(world, 2, [0, 1])
        for r in range(5):
            self.assertEqual(outs[r].describe(), outs[0].describe())
            self.assertEqual(outs[r].digest(), outs[0].digest())
        self.assertIn("n_blocks=2", outs[0].describe())
        self.assertIn("ranks_per_block=", outs[0].describe())


if __name__ == "__main__":
    unittest.main()


class FactorizeLayoutPublicApiTest(unittest.TestCase):
    """``factorize_layout`` is the ONE rule; nothing should mirror it.

    ``scripts/walker_scaleup/scale_up.py`` (branch ``walker-scaleup-24``)
    re-implements ``build_layout``'s arithmetic in its own ``n_compute`` /
    ``block_width`` helpers, with a docstring saying it "mirrors" them. A
    mirror keeps answering the old question after the rule changes -- which
    it now has -- so the planner would report layouts the engine will not
    build. Exporting the rule makes re-anchoring it an import.
    """

    def test_it_is_importable_from_the_package(self):
        from lisatools.globalfit.communication import factorize_layout as f
        self.assertEqual(f(4, 4), (4, 1, 1))

    def test_the_documented_table(self):
        from lisatools.globalfit.communication import factorize_layout as f
        self.assertEqual(f(4, 4), (4, 1, 1))     # today's production run
        self.assertEqual(f(4, 16), (4, 4, 1))    # GPUs > walkers
        self.assertEqual(f(24, 8), (8, 1, 3))    # PE blocks
        self.assertEqual(f(24, 32), (8, 4, 3))   # both axes
        self.assertEqual(f(1, 4), (1, 4, 1))     # one-walker replicas
        self.assertEqual(f(10, 4), (2, 2, 5))
        self.assertEqual(f(32, 16), (16, 1, 2))

    def test_the_invariants_hold_across_a_sweep(self):
        from lisatools.globalfit.communication import factorize_layout as f
        for nw in range(1, 33):
            for nc in range(1, 33):
                nb, r, blk = f(nw, nc)
                self.assertEqual(nb * r, nc, f"({nw}, {nc})")
                self.assertEqual(nb * blk, nw, f"({nw}, {nc})")

    def test_it_agrees_with_build_layout(self):
        # the rule and the layout that uses it must never disagree
        for nw, size in ((4, 3), (8, 5), (1, 5), (2, 5)):
            lay = _layouts(FakeWorld(size, nodes=[0, 1, 0, 1, 0][:size]),
                           nw, [0, 1])[0]
            from lisatools.globalfit.communication import factorize_layout as f
            self.assertEqual(
                f(nw, lay.n_compute),
                (lay.n_blocks, lay.ranks_per_block, lay.block),
                msg=f"nwalkers={nw} size={size}")

    def test_it_is_pure(self):
        # no env read, no warning, no communicator -- a planner must be able
        # to call it without side effects
        import warnings
        from lisatools.globalfit.communication import factorize_layout as f
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(f(3, 2), (1, 2, 3))   # the coprime case that WARNS in build_layout

    def test_it_raises_the_same_refusals(self):
        from lisatools.globalfit.communication import factorize_layout as f
        with self.assertRaises(ValueError):
            f(4, 16, ranks_per_block=3)    # does not divide n_compute
        with self.assertRaises(ValueError):
            f(3, 16, ranks_per_block=8)    # 2 blocks do not divide 3 walkers


class DevicesBeyondWalkersTest(unittest.TestCase):
    """A rank may never drive more GPUs than it holds walkers (job 764).

    ``NGPUS=8 GPUS_PER_RANK=2`` on the 4-walker 9mo store: 4 compute ranks
    with devices ``[0, 1]`` and ONE walker each. ``AnalysisContainerArray``
    shards a rank's walker rows over its devices, so the second device got
    an EMPTY shard and the build died two minutes in at the first GB gather
    ("Per-AC element size mismatch across shards: [3415857, 0]"). The layout
    now refuses it on every rank, before any device is touched, naming both
    ways out.
    """

    #: 9mo job 764: 4 nodes x gpu:2, -ppn 1 round robin, 4 compute + saver
    JOB_764_NODES = [0, 1, 2, 3, 0]

    def test_job_764_shape_is_refused_with_both_routes(self):
        with self.assertRaises(RuntimeError) as cm:
            _layouts(FakeWorld(5, nodes=self.JOB_764_NODES), 4, [0, 1],
                     gpus_per_rank=2, gpu_routing=False)
        msg = str(cm.exception)
        self.assertIn("ValueError", msg)
        self.assertIn("would drive 2 GPUs [0, 1] but holds only 1 walker(s)", msg)
        self.assertIn("GF_GPU_ROUTING=1", msg)                 # route 1
        self.assertIn("unset GPUS_PER_RANK", msg)
        self.assertIn("NWALKERS a multiple of 8", msg)          # route 2: 4 ranks x 2
        self.assertIn("Per-AC element size mismatch", msg)      # the crash it prevents

    def test_auto_gpus_per_rank_hits_it_too(self):
        # a lone compute rank per node takes the whole [0, 1] pool (AUTO) --
        # with ONE walker that is the same empty shard
        with self.assertRaises(RuntimeError) as cm:
            _layouts(FakeWorld(3, nodes=[0, 1, 0]), 1, [0, 1])
        self.assertIn("would drive 2 GPUs", str(cm.exception))
        with self.assertRaises(RuntimeError):
            FakeWorld(1).run(lambda r, c: build_layout(c, 1, [0, 1], legacy=False))

    def test_enough_walkers_per_rank_still_resolves(self):
        # the same allocation with 8 walkers: 2 per rank on 2 GPUs -- legal
        lay = _layouts(FakeWorld(5, nodes=self.JOB_764_NODES), 8, [0, 1],
                       gpus_per_rank=2, gpu_routing=False)[0]
        self.assertEqual(lay.placements[0].devices, (0, 1))
        self.assertEqual(lay.block_of(0), (0, 2))
        # one GPU per rank never trips it, whatever the walker count
        lay = _layouts(FakeWorld(3), 4, [0, 1], gpus_per_rank=1)[0]
        self.assertEqual(lay.placements[0].devices, (0,))

    def test_setup_acs_refuses_it_too_before_building_anything(self):
        """The build-time backstop (a path that never resolved a layout):
        ``GlobalFit.setup_acs`` raises the same message BEFORE it reads the
        state or builds a single walker container."""
        from types import SimpleNamespace

        from lisatools.globalfit.run import GlobalFit

        fake = SimpleNamespace(
            curr=SimpleNamespace(general_info=SimpleNamespace(gpus=[0, 1])),
            nwalkers=4, rank=3, layout=None)
        with self.assertRaises(ValueError) as cm:
            # ``state=None``: the refusal must come before anything touches it
            GlobalFit.setup_acs(fake, None, walker_block=(3, 4))
        self.assertIn("rank 3 would drive 2 GPUs [0, 1] but holds only 1 walker(s)",
                      str(cm.exception))
        self.assertIn("GF_GPU_ROUTING=1", str(cm.exception))

    def test_the_routing_production_shape(self):
        """``NGPUS=8 GF_GPU_ROUTING=1`` on the 4-walker store: what the run builds.

        10 tasks over 4 nodes, ``-ppn 1`` round robin; the driver splits the
        monitor rank (world rank 9) off BEFORE the layout, so the run world
        is ranks 0..8 on nodes [0, 1, 2, 3, 0, 1, 2, 3, 0]: 8 compute ranks,
        ONE device each, 4 walker blocks x R = 2, the head = replica 0 of
        block 0, the saver = rank 8 (node 0, where the head is)."""
        nodes = [0, 1, 2, 3, 0, 1, 2, 3, 0]
        outs = _layouts(FakeWorld(9, nodes=nodes), 4, [0, 1], gpu_routing=True)
        lay = outs[0]
        for r in range(9):
            self.assertEqual(outs[r].digest(), lay.digest())
        self.assertEqual((lay.n_compute, lay.n_blocks, lay.ranks_per_block,
                          lay.block), (8, 4, 2, 1))
        self.assertEqual(lay.head_rank, 0)
        self.assertEqual(lay.saver_rank, 8)
        self.assertEqual(lay.role_of(8), RankRole.SAVER)
        self.assertTrue(lay.gpu_routing and lay.replica_mode)
        for r in range(8):
            self.assertEqual(len(lay.placements[r].devices), 1, r)
            self.assertEqual(lay.block_of(r), (r // 2, r // 2 + 1), r)
            self.assertEqual(lay.replica_index(r), r % 2, r)
        self.assertEqual(lay.block_leads, (0, 2, 4, 6))
        self.assertEqual(lay.block_index(0), 0)
        self.assertTrue(lay.is_block_lead(0))
        # two compute ranks per 2-GPU node, on different devices
        for node in range(4):
            ranks = [r for r in range(8) if nodes[r] == node]
            self.assertEqual(sorted(lay.placements[r].devices[0] for r in ranks),
                             [0, 1], node)
        self.assertIn("gpu_routing=on", lay.describe())
        self.assertIn("REPLICAS", lay.describe())
        # the same shape with the knob OFF is the legacy refusal, naming it
        with self.assertRaises(RuntimeError) as cm:
            _layouts(FakeWorld(9, nodes=nodes), 4, [0, 1], gpu_routing=False)
        self.assertIn("GF_GPU_ROUTING=1", str(cm.exception))
