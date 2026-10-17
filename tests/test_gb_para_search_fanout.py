"""The parallel-search BUILD over two compute ranks (FakeWorld threads).

The real head path -- ``GBParaSearchRJMove._para_build`` -> ``_fanout_cmd``
-> ``WalkerFanout.run`` -> every rank's ``ComputeService`` -> the real
``gf_serve`` dispatch -> the real ``_gb_serve_para_search`` rank body -- with
only the per-walker SEARCH replaced (``_para_builder`` returns a stub whose
``run`` hands back a canned finds record; there is no residual on a thread).
Pinned: each rank searches exactly its own walker block, with the head's
per-walker seed and the head's knobs; a rank that fails reports instead of
raising; the head merges the same source found on both ranks into ONE
component (the union of the two walkers' samples) and keeps each rank's own
find; the manifest records both walkers; and the next draw of the move
adopts the epoch, the directive the ranks then load.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.warmstart import para_search as ps

try:
    from tests.test_gb_para_search_units import DF, TOBS, _finds, _transform
except ImportError:  # pragma: no cover - run from tests/
    from test_gb_para_search_units import DF, TOBS, _finds, _transform


SHARED_F0 = 3.0e-3


def _canned(w):
    """Walker w: the shared source (0.4 w bins off) and one of its own."""
    return _finds(w, [(SHARED_F0 + 0.4 * w * DF, 1, 9.0, False, w),
                      (SHARED_F0 + (20 + 30 * w) * DF, 2, 6.0, False, 10 + w)])


def _move(root, rank, fail=False):
    from lisatools.globalfit.moves import gbspecialstretch as gbs

    class _StubBuilder:
        def __init__(self, move):
            self.move = move
            self.tag = ""

        def run(self, walker=None, seed=None):
            self.move.searched.append((int(walker), int(seed)))
            if fail:
                raise RuntimeError("no residual on this rank")
            return _canned(int(walker))

    class _Skel(gbs.GBParaSearchRJMove):
        backend = SimpleNamespace(uses_cupy=False, uses_cuda=False)
        mempool = SimpleNamespace(free_all_blocks=lambda: None)

        def _bind_rank_acs(self, model):
            return None

        def _para_builder(self, model, row, cfg):
            self.cfgs.append(cfg)
            return _StubBuilder(self)

    m = object.__new__(_Skel)
    m.name = m.gf_move_name = ps.PARA_SEARCH_MOVE
    m.branch_name = "gb"
    m.fanout = None
    m.gf_rank = rank
    m.nwalkers = 2
    m.ntemps = 2
    m.gf_stage_kind, m.gf_stage_name, m.gf_iteration = "pe", "full_pe", 0
    m.transform_fn = _transform()
    m._basis_settings = SimpleNamespace(Tobs=TOBS)
    m.searched, m.cfgs = [], []
    m._para_search_cfg = ps.ParaSearchConfig(every_pe=5, ntemps=6, nwalkers=10)
    m._para_search_root = root
    m._para_holder = ps.ParaSearchHolder(epoch_root=root, load_kwargs=dict(new_tobs=TOBS),
                                         build_seed=5, tag=0x3A14,
                                         name=ps.PARA_SEARCH_MOVE)
    m._para_build_seed = 5
    m._para_clock = None
    m._para_last_attempt = None
    m._para_search_directive = None
    m._para_stage_warned = False
    m._para_epochs_built = 0
    m.rj_proposal_distribution = None
    m._prop_timer = None
    return m


def _world(root, *, worker_fails=False):
    world = FakeWorld(3, nodes=[0, 0, 0])
    moves = {}

    def fn(rank, comm):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            layout = build_layout(comm, 2, [0, 1], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        role = layout.role_of(rank)
        if role == RankRole.SAVER:
            return None
        move = _move(root, rank, fail=(worker_fails and role != RankRole.HEAD))
        moves[rank] = move
        if role == RankRole.HEAD:
            fo = WalkerFanout(fcomm, layout, rank, model="m0")
            fo.enter_stage("full_pe", "pe")
            move.fanout = fo
            try:
                move.setup("m0", None)                 # first entry: builds epoch 0
                d1 = move._para_search_directive
                move.gf_iteration = 1
                move.setup("m0", None)                 # next draw: adopts it
                return dict(d1=d1, d2=move._para_search_directive,
                            rj=move.rj_proposal_distribution is not None)
            finally:
                fo.stop()
        service = ComputeService(fcomm, layout, rank,
                                 registry={ps.PARA_SEARCH_MOVE: move}, model=f"m{rank}")
        return service.serve()

    return world.run(fn), moves


class FanoutBuildTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="para_fanout_")
        self.addCleanup(shutil.rmtree, self.d, True)
        self.root = os.path.join(self.d, "warmstart", "parasearch")

    def _manifest(self, k=0):
        with open(os.path.join(self.root, f"epoch_{k:04d}", "DONE.json")) as fh:
            return json.load(fh)

    def test_two_ranks_search_their_own_walker_and_the_head_merges(self):
        out, moves = _world(self.root)
        head = out[0]
        self.assertIsNone(head["d1"])                  # the build propose: the lag
        self.assertEqual(head["d2"], {"epoch": 0})
        self.assertTrue(head["rj"])
        self.assertEqual(out[1], 1)                    # the worker served one command
        # each rank searched exactly its block, with the head's knobs + seeds
        self.assertEqual([w for w, _ in moves[0].searched], [0])
        self.assertEqual([w for w, _ in moves[1].searched], [1])
        seeds = {moves[0].searched[0][1], moves[1].searched[0][1]}
        self.assertEqual(len(seeds), 2)
        self.assertEqual((moves[1].cfgs[0].ntemps, moves[1].cfgs[0].nwalkers), (6, 10))
        man = self._manifest()
        self.assertEqual(man["walkers"], [0, 1])
        self.assertEqual(man["finds_per_walker"], {"0": 2, "1": 2})
        self.assertEqual(man["n_finds"], 4)
        self.assertEqual(man["n_sources"], 3)          # the shared source merged once
        self.assertEqual(man["n_compute"], 2)
        self.assertEqual(man["errors"], [])
        with np.load(os.path.join(self.root, "epoch_0000", ps.EPOCH_NPZ)) as d:
            nw = d["src_n_walkers"]
            f0 = d["src_f0_hz"]
            n_mem = d["n_members"]
        i = int(np.argmin(np.abs(f0 - SHARED_F0)))
        self.assertEqual(int(nw[i]), 2)
        self.assertEqual(int(n_mem[i]), 800)           # the union, 400 per walker
        self.assertEqual(sorted(nw.tolist()), [1, 1, 2])

    def test_a_failing_rank_reports_and_the_head_builds_from_the_rest(self):
        out, moves = _world(self.root, worker_fails=True)
        self.assertEqual(out[0]["d2"], {"epoch": 0})
        man = self._manifest()
        self.assertEqual(man["walkers"], [0])
        self.assertEqual(len(man["errors"]), 1)
        self.assertIn("walker 1", man["errors"][0])
        self.assertEqual(man["n_sources"], 2)


if __name__ == "__main__":
    unittest.main()
