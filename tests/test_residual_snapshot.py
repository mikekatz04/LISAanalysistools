"""``GF_RESIDUAL_SNAPSHOT_EVERY``: the run writes its own residuals (data
minus EVERY fitted signal) beside the store, so the monitor's data /
template / residual panels stop subtracting GB + VGB only
(Mike 2026-10-07: an MBH merger at 111 d showed as "residual" on the page
while the run had fitted it).

Since 2026-10-11 EVERY cold walker's residual is written (``residuals`` +
``walkers``), not only the max-lnL one: the page's min-lnL walker was
otherwise rebuilt as data minus its GB + VGB only, MBH / EMRI / SOBBH left
in. ``residual`` / ``walker`` / the rest keep their old meaning so an old
reader still works."""
import inspect
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit import residual_snapshot as rs

SHAPE = (3, 4, 6)
#: every key a pre-2026-10-11 sidecar carried, with the meaning it keeps
LEGACY_KEYS = {"data", "residual", "f_edges", "t_edges", "iteration",
               "store_row", "walker", "lnl", "nf", "nt", "dt", "written_at"}


class _Settings:
    Nf, Nt, data_dt = 16, 12, 2.5
    f_arr_edges = np.arange(5) * 0.1
    t_arr_edges = np.arange(7) * 10.0


class _AC:
    def __init__(self, arr):
        self.data = SimpleNamespace(arr=arr, settings=_Settings())


class _ACA:
    """Stand-in AnalysisContainerArray: one residual per walker."""

    def __init__(self, arrs, lls):
        self.acs = [_AC(a) for a in arrs]
        self.lls = np.asarray(lls, float)

    def __getitem__(self, i):
        return self.acs[i]

    def likelihood(self, complex=False):
        return self.lls


def _arrs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=SHAPE) + 10 * k for k in range(n)]


class _Layout:
    """4 walkers on 2 compute ranks (head = rank 0, worker = rank 1).

    ``block_leads`` is deliberately WORKER FIRST: the merge must place each
    block by its own ``block_of`` range, not by the order ranks reply in.
    """
    head_rank = 0
    worker_ranks = (1,)
    block_leads = (1, 0)

    def block_of(self, rank):
        return (0, 2) if rank == 0 else (2, 4)

    def owner_of(self, w):
        return (0, w) if w < 2 else (1, w - 2)


class _Fanout:
    """Runs the worker side through the real builtin, like ComputeService."""

    def __init__(self, head_acs, worker_acs, lls):
        self.layout = _Layout()
        self.head_acs, self.worker_acs, self.lls = head_acs, worker_acs, lls
        self.payloads = {}

    def gather_likelihood(self, acs):
        return np.asarray(self.lls, float)

    def run(self, op, *, move, per_rank_payload, local_body, merge):
        assert op == rs.RESIDUAL_SNAPSHOT_OP
        self.payloads = {0: per_rank_payload(0, 0, 2), 1: per_rank_payload(1, 2, 4)}
        worker_model = SimpleNamespace(analysis_container_arr=self.worker_acs)
        return merge({1: rs.walker_residual(worker_model, self.payloads[1]),
                      0: local_body(self.payloads[0], None)})


class TakeSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "run_testing.h5")
        self.data = np.full(SHAPE, 7.0)
        self.holder = SimpleNamespace(data_res_arr=SimpleNamespace(arr=self.data))

    def test_single_process_writes_the_max_lnl_walkers_residual(self):
        arrs = _arrs(4)
        acs = _ACA(arrs, [3.0, 9.0, 1.0, np.nan])
        path = rs.take_snapshot(acs=acs, data_holder=self.holder,
                                store_path=self.store, iteration=41,
                                store_iteration0=707)
        self.assertEqual(path, os.path.join(self.tmp, "run_testing_residual_snapshot.npz"))
        with np.load(path) as z:
            self.assertEqual(int(z["walker"]), 1)
            self.assertEqual(int(z["iteration"]), 41)
            # the store row, not the job's own count (which restarts at 0
            # on every resume: a page read "iteration 161" at row ~870)
            self.assertEqual(int(z["store_row"]), 748)
            np.testing.assert_allclose(z["residual"], arrs[1].astype(np.float32))
            np.testing.assert_array_equal(z["data"], self.data.astype(np.float32))
            self.assertEqual(z["residual"].dtype, np.float32)
            np.testing.assert_array_equal(z["t_edges"], _Settings.t_arr_edges)
        self.assertFalse(any(f.endswith(".tmp.npz") for f in os.listdir(self.tmp)))

    def test_single_process_writes_every_cold_walker(self):
        # (b) no fan-out: the head's ACA holds every walker.
        arrs = _arrs(4, seed=3)
        acs = _ACA(arrs, [3.0, 9.0, 1.0, np.nan])
        path = rs.take_snapshot(acs=acs, data_holder=self.holder,
                                store_path=self.store, iteration=2)
        with np.load(path) as z:
            self.assertTrue(LEGACY_KEYS <= set(z.files))
            self.assertEqual(z["residuals"].shape, (4,) + SHAPE)
            self.assertEqual(z["residuals"].dtype, np.float32)
            np.testing.assert_array_equal(z["walkers"], np.arange(4))
            for k in range(4):
                np.testing.assert_array_equal(z["residuals"][k],
                                              arrs[k].astype(np.float32))
            # the legacy key is the max-lnL walker's row of the stack
            np.testing.assert_array_equal(z["residual"], z["residuals"][int(z["walker"])])
            np.testing.assert_array_equal(z["lnl"], [3.0, 9.0, 1.0, np.nan])

    def test_fanout_every_block_lead_returns_its_walkers_in_order(self):
        # (a) fan-out: both blocks answer, the stack comes back in GLOBAL
        # walker order, and the legacy keys mean what they always meant.
        arrs = _arrs(4, seed=1)
        head, worker = _ACA(arrs[:2], [0, 0]), _ACA(arrs[2:], [0, 0])
        fan = _Fanout(head, worker, lls=[1.0, 2.0, 5.0, 3.0])   # max = walker 2
        path = rs.take_snapshot(acs=head, data_holder=self.holder,
                                store_path=self.store, iteration=5, fanout=fan)
        self.assertEqual(fan.payloads[0], {"rows": [0, 1]})
        self.assertEqual(fan.payloads[1], {"rows": [0, 1]})
        with np.load(path) as z:
            self.assertTrue(LEGACY_KEYS <= set(z.files))
            self.assertEqual(int(z["walker"]), 2)
            np.testing.assert_allclose(z["residual"], arrs[2].astype(np.float32))
            np.testing.assert_array_equal(z["walkers"], np.arange(4))
            for k in range(4):
                np.testing.assert_array_equal(z["residuals"][k],
                                              arrs[k].astype(np.float32))

    def test_a_block_that_does_not_answer_refuses(self):
        arrs = _arrs(4, seed=2)
        head, worker = _ACA(arrs[:2], [0, 0]), _ACA(arrs[2:], [0, 0])
        fan = _Fanout(head, worker, lls=[1.0, 2.0, 5.0, 3.0])
        with mock.patch.object(rs, "walker_residual", return_value=None):
            with self.assertRaises(RuntimeError):
                rs.take_snapshot(acs=head, data_holder=self.holder,
                                 store_path=self.store, iteration=5, fanout=fan)

    def test_a_shape_mismatch_refuses(self):
        acs = _ACA(_arrs(1), [0.0])
        bad = SimpleNamespace(data_res_arr=SimpleNamespace(arr=np.zeros((3, 4, 5))))
        with self.assertRaises(RuntimeError):
            rs.take_snapshot(acs=acs, data_holder=bad, store_path=self.store,
                             iteration=0)


class SidecarRoundTripTest(unittest.TestCase):
    """(c) What the run writes is what the monitor page reads back, and a
    sidecar written before 2026-10-11 (no ``residuals``) still loads."""

    @classmethod
    def setUpClass(cls):
        from lisatools.globalfit.monitor._short import _generator_functions

        cls.ns = _generator_functions({"np": np}, names=("load_residual_snapshot",))

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "run_testing.h5")
        self.holder = SimpleNamespace(
            data_res_arr=SimpleNamespace(arr=np.full(SHAPE, 7.0)))

    def test_new_sidecar_round_trips_and_keeps_only_the_asked_walkers(self):
        arrs = _arrs(4, seed=4)
        acs = _ACA(arrs, [3.0, 9.0, 1.0, 2.0])     # max lnL = 1, min = 2
        path = rs.take_snapshot(acs=acs, data_holder=self.holder,
                                store_path=self.store, iteration=8,
                                store_iteration0=81)
        snap = self.ns["load_residual_snapshot"](path, keep_walkers=(1, 2))
        self.assertEqual(int(snap["walker"]), 1)
        self.assertEqual(snap["row"], 89)
        self.assertEqual(snap["walkers"], [1, 2])
        self.assertEqual(snap["residuals"].shape, (2,) + SHAPE)
        np.testing.assert_array_equal(snap["residuals"][1], arrs[2].astype(np.float32))
        np.testing.assert_array_equal(snap["residual"], arrs[1].astype(np.float32))
        full = self.ns["load_residual_snapshot"](path)
        self.assertEqual(full["walkers"], [0, 1, 2, 3])

    def test_an_old_single_walker_sidecar_still_loads(self):
        path = os.path.join(self.tmp, "old_testing_residual_snapshot.npz")
        res = np.ones(SHAPE, np.float32)
        np.savez(path, data=np.zeros(SHAPE, np.float32), residual=res,
                 f_edges=np.arange(5) * 0.1, t_edges=np.arange(7) * 10.0,
                 iteration=32, store_row=89, walker=1, lnl=np.zeros(4),
                 nf=16, nt=12, dt=2.5, written_at=0.0)
        snap = self.ns["load_residual_snapshot"](path, keep_walkers=(1, 0))
        self.assertNotIn("residuals", snap)
        self.assertNotIn("walkers", snap)
        self.assertEqual((int(snap["walker"]), snap["row"]), (1, 89))
        np.testing.assert_array_equal(snap["residual"], res)


def _block_world(size, nodes, nwalkers):
    """Real ``WalkerFanout`` + ``ComputeService`` over the fake communicator,
    with the builtin wired exactly as ``run.py`` installs it. Returns the
    head's sidecar contents, the walkers' residuals and every rank's payload."""
    from lisatools.globalfit.communication.fakecomm import FakeWorld
    from lisatools.globalfit.communication.fanout import (
        LIKELIHOOD_OP, ComputeService, WalkerFanout)
    from lisatools.globalfit.communication.ranks import RankRole, build_layout

    arrs = _arrs(nwalkers, seed=11)
    lls = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0][:nwalkers])
    tmp = tempfile.mkdtemp()
    seen = {}

    def fn(rank, comm):
        layout = build_layout(comm, nwalkers, [0, 1], legacy=False, gpu_routing=True)
        fcomm = layout.make_fanout_comm(comm)
        role = layout.role_of(rank)
        if role == RankRole.SAVER:
            return "saver"
        w0, w1 = layout.block_of(rank)
        aca = _ACA(arrs[w0:w1], lls[w0:w1])
        if role == RankRole.HEAD:
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            holder = SimpleNamespace(data_res_arr=SimpleNamespace(arr=np.zeros(SHAPE)))
            try:
                path = rs.take_snapshot(acs=aca, data_holder=holder,
                                        store_path=os.path.join(tmp, "w_testing.h5"),
                                        iteration=2, fanout=fo)
            finally:
                fo.stop()
            with np.load(path) as z:
                return {k: z[k] for k in z.files}, layout

        def _snap(payload, clock, model):
            seen[rank] = payload
            return rs.walker_residual(model, payload)

        model = SimpleNamespace(analysis_container_arr=aca)
        builtins = {
            LIKELIHOOD_OP: lambda payload, clock, model: np.asarray(
                model.analysis_container_arr.likelihood(complex=False)),
            # the same lambda shape run.py installs
            rs.RESIDUAL_SNAPSHOT_OP: _snap,
        }
        return ComputeService(fcomm, layout, rank, registry={}, model=model,
                              builtins=builtins).serve()

    out = FakeWorld(size, nodes=nodes).run(fn)
    return out[0], arrs, lls, seen


class FanoutFakeCommTest(unittest.TestCase):
    """(a) through the REAL fan-out: blocks assembled in global walker order."""

    def test_three_blocks_of_two_walkers(self):
        (z, layout), arrs, lls, seen = _block_world(4, [0, 1, 0, 1], 6)
        self.assertEqual((layout.n_blocks, layout.ranks_per_block), (3, 1))
        self.assertEqual(int(z["walker"]), int(np.argmax(lls)))
        np.testing.assert_array_equal(z["lnl"], lls)
        np.testing.assert_array_equal(z["walkers"], np.arange(6))
        for k in range(6):
            np.testing.assert_array_equal(z["residuals"][k], arrs[k].astype(np.float32))
        np.testing.assert_array_equal(z["residual"], arrs[int(np.argmax(lls))].astype(np.float32))
        self.assertEqual(seen, {1: {"rows": [0, 1]}, 2: {"rows": [0, 1]}})

    def test_replicas_of_a_block_are_not_asked(self):
        # 4 compute ranks, 2 walkers -> 2 blocks of 1 walker, R = 2: ranks
        # (0, 1) hold walker 0 and (2, 3) walker 1. Only the leads 0 and 2
        # ship arrays; 1 and 3 get no payload and return nothing.
        (z, layout), arrs, lls, seen = _block_world(5, [0, 1, 0, 1, 0], 2)
        self.assertEqual((layout.n_blocks, layout.ranks_per_block), (2, 2))
        self.assertEqual(seen, {1: None, 2: {"rows": [0]}, 3: None})
        np.testing.assert_array_equal(z["walkers"], [0, 1])
        for k in range(2):
            np.testing.assert_array_equal(z["residuals"][k], arrs[k].astype(np.float32))


class CadenceTest(unittest.TestCase):
    def _kw(self, tmp):
        return dict(acs=_ACA(_arrs(2), [0.0, 1.0]),
                    data_holder=SimpleNamespace(
                        data_res_arr=SimpleNamespace(arr=np.zeros(SHAPE))),
                    store_path=os.path.join(tmp, "s.h5"))

    def test_every_third_iteration_by_default_and_zero_turns_it_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if k != rs.SNAPSHOT_EVERY_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                got = [rs.maybe_snapshot(iteration=i, **self._kw(tmp)) is not None
                       for i in range(7)]
            self.assertEqual(got, [False, False, True, False, False, True, False])
            with mock.patch.dict(os.environ, {rs.SNAPSHOT_EVERY_ENV: "0"}):
                self.assertIsNone(rs.maybe_snapshot(iteration=2, **self._kw(tmp)))

    def test_a_failure_never_reaches_the_run(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {rs.SNAPSHOT_EVERY_ENV: "1"}), \
                mock.patch.object(rs, "take_snapshot", side_effect=RuntimeError("x")):
            with self.assertLogs(rs.logger, level="WARNING"):
                self.assertIsNone(rs.maybe_snapshot(iteration=0, **self._kw(tmp)))


class WiringTest(unittest.TestCase):
    def test_recipe_hook_and_pickling(self):
        from lisatools.globalfit.recipe import Recipe

        src = inspect.getsource(Recipe.__call__)
        self.assertIn("maybe_snapshot(", src)
        r = Recipe()
        r._init_runtime()
        r.snapshot_context = dict(acs=object(), data_holder=None, store_path="x")
        self.assertNotIn("snapshot_context", pickle.loads(pickle.dumps(r)).__dict__)

    def test_compute_ranks_serve_the_builtin_and_the_head_gets_a_context(self):
        from lisatools.globalfit import run

        src = inspect.getsource(run)
        self.assertIn("RESIDUAL_SNAPSHOT_OP: lambda payload, clock, model: walker_residual(", src)
        self.assertIn("self.recipe.snapshot_context = dict(", src)
        self.assertIn('store_iteration0=int(getattr(self, "_resume_store_iteration", 0))', src)
        self.assertIn("self._resume_store_iteration = int(backend.iteration)", src)

    def test_short_tar_ships_the_snapshot(self):
        from lisatools.globalfit.monitor import snapshot as snap

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "run_testing_residual_snapshot.npz")
            np.savez(p, a=np.zeros(1))
            self.assertIn(p, snap._short_members(tmp, None, 1))


if __name__ == "__main__":
    unittest.main()
