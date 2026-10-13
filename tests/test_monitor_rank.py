"""The DEDICATED MONITOR RANK (GF_MONITOR_RANK=1, 2026-10-04).

User ruling: "adjust this so they are separated and you can have a separate
rank run the html". On the saver the page + tar build held the head's next
(blocking) save ~400 s every third iteration on job 717. Pinned here:

* the split: off by default, the HIGHEST world rank leaves, the run keeps
  world order, too few ranks refuses;
* the saver only NOTIFIES (non-blocking) and never builds when a monitor
  rank exists; without one, the old in-place hook still runs;
* the monitor loop: GF_MONITOR_ITER cadence robust to coalesced notices,
  the run dir from the notice, a failing build never escapes, exit on the
  closing barrier;
* the driver and the launchers wire it (source pins), and the allocation
  check disarms it when the job has no spare task;
* a REAL ``mpiexec -n 4`` run of the real saver loop: the head's saves never
  wait on a slow build.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from lisatools.globalfit import hdfbackend as hb
from lisatools.globalfit import monitor as mon
from lisatools.globalfit.monitor import rank as mr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Req:
    def __init__(self, done=True):
        self.done = done
        self.waited = False

    def Test(self):
        return self.done

    def Wait(self):
        self.waited = True
        self.done = True


class _World:
    """Scripted world communicator for one rank."""

    def __init__(self, rank=0, size=6, inbox=(), barrier_after=None):
        self.rank, self.size = rank, size
        self.split_calls = []
        self.sent = []
        self.inbox = list(inbox)          # notices this rank will receive
        self.barrier_after = barrier_after  # Test() calls before it completes
        self._tests = 0
        self.ibarriers = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Split(self, color, key):
        self.split_calls.append((color, key))
        return ("RUN_COMM", color)

    def isend(self, obj, dest, tag):
        self.sent.append((obj, dest, tag))
        return _Req(done=False)

    def iprobe(self, source=None, tag=None):
        return bool(self.inbox) and self.inbox[0] is not None

    def recv(self, source=None, tag=None):
        return self.inbox.pop(0)

    def Ibarrier(self):
        self.ibarriers += 1
        world = self

        class _B:
            def Test(self_inner):
                world._tests += 1
                if world.inbox and world.inbox[0] is None:
                    world.inbox.pop(0)   # a None marks "poll boundary"
                    return False
                return (world.barrier_after is not None
                        and world._tests > world.barrier_after)

            def Wait(self_inner):
                return None
        return _B()


class SplitTest(unittest.TestCase):
    def tearDown(self):
        mr._reset_state()

    def test_off_by_default_changes_nothing(self):
        w = _World(rank=0, size=6)
        run, m = mr.split_monitor_rank(w, environ={})
        self.assertIs(run, w)
        self.assertIsNone(m)
        self.assertEqual(w.split_calls, [])
        self.assertFalse(mr.active())

    def test_the_highest_rank_leaves_and_the_rest_keep_world_order(self):
        env = {"GF_MONITOR_RANK": "1"}
        w = _World(rank=2, size=6)
        run, m = mr.split_monitor_rank(w, environ=env)
        self.assertEqual(m, 5)
        self.assertEqual(w.split_calls, [(0, 2)])     # color 0, key = rank
        self.assertEqual(run, ("RUN_COMM", 0))
        self.assertTrue(mr.active())
        mr._reset_state()
        wm = _World(rank=5, size=6)
        run_m, m2 = mr.split_monitor_rank(wm, environ=env)
        self.assertIsNone(run_m)
        self.assertEqual(m2, 5)
        self.assertEqual(wm.split_calls, [(1, 5)])
        self.assertFalse(mr.active())                  # the monitor never notifies

    def test_too_few_ranks_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            mr.split_monitor_rank(_World(rank=0, size=3),
                                  environ={"GF_MONITOR_RANK": "1"})

    def test_a_single_process_is_left_alone(self):
        w = _World(rank=0, size=1)
        self.assertEqual(mr.split_monitor_rank(w, environ={"GF_MONITOR_RANK": "1"}),
                         (w, None))


class NotifyFinishTest(unittest.TestCase):
    def tearDown(self):
        mr._reset_state()

    def _arm(self, w):
        mr.split_monitor_rank(w, environ={"GF_MONITOR_RANK": "1"})

    def test_notice_is_nonblocking_and_carries_the_store(self):
        w = _World(rank=4, size=6)
        self._arm(w)
        mr.notify_saved("/run/dir/gf.h5", 7, dropped=1)
        self.assertEqual(w.sent, [({"saved": 7, "store": "/run/dir/gf.h5",
                                    "dropped": 1}, 5, mr.MONITOR_TAG)])

    def test_completed_sends_are_released(self):
        w = _World(rank=4, size=6)
        self._arm(w)
        mr.notify_saved("/s.h5", 1)
        mr._STATE["reqs"][0].done = True
        mr.notify_saved("/s.h5", 2)
        self.assertEqual(len(mr._STATE["reqs"]), 1)

    def test_a_failing_send_never_reaches_the_saver(self):
        w = _World(rank=4, size=6)
        self._arm(w)
        w.isend = mock.Mock(side_effect=RuntimeError("boom"))
        mr.notify_saved("/s.h5", 3)           # no raise

    def test_finish_waits_its_sends_then_joins_the_barrier(self):
        w = _World(rank=4, size=6)
        self._arm(w)
        mr.notify_saved("/s.h5", 1)
        req = mr._STATE["reqs"][0]
        mr.finish_run_ranks()
        self.assertTrue(req.waited)
        self.assertEqual(w.ibarriers, 1)
        self.assertFalse(mr.active())

    def test_finish_is_a_noop_without_a_monitor_rank(self):
        mr.finish_run_ranks()


class CadenceTest(unittest.TestCase):
    def test_due_counts_saves_and_survives_coalescing(self):
        self.assertFalse(mr._due(9, 0, 10))
        self.assertTrue(mr._due(10, 0, 10))
        self.assertTrue(mr._due(12, 0, 10))       # 10 was coalesced away
        self.assertFalse(mr._due(19, 12, 10))
        self.assertTrue(mr._due(20, 12, 10))
        self.assertTrue(mr._due(1, 0, 0))         # 0 / garbage -> every save
        self.assertFalse(mr._due(0, 0, 1))


class MonitorLoopTest(unittest.TestCase):
    ENV = {"GF_MONITOR_AFTER_SAVE": "1", "GF_MONITOR_ITER": "2"}

    def _notice(self, i, store="/run/dir/gf.h5"):
        return {"saved": i, "store": store, "dropped": 0}

    def test_builds_on_the_cadence_from_the_notice_run_dir(self):
        inbox = [self._notice(1), None, self._notice(2), None,
                 self._notice(3), self._notice(4), None]
        w = _World(rank=5, size=6, inbox=inbox, barrier_after=3)
        seen = []
        n = mr.run_monitor_rank(w, build=lambda d: seen.append(d) or 1.0,
                                environ=self.ENV, sleep=lambda s: None,
                                hide_gpus=False)
        self.assertEqual(seen, ["/run/dir", "/run/dir"])   # after saves 2 and 4
        self.assertEqual(n, 2)

    def test_after_save_off_builds_nothing_but_still_exits(self):
        w = _World(rank=5, size=6, inbox=[self._notice(2), None],
                   barrier_after=1)
        build = mock.Mock()
        mr.run_monitor_rank(w, build=build, environ={"GF_MONITOR_ITER": "1"},
                            sleep=lambda s: None, hide_gpus=False)
        build.assert_not_called()

    def test_a_failing_build_never_escapes(self):
        w = _World(rank=5, size=6, inbox=[self._notice(2), None],
                   barrier_after=2)
        build = mock.Mock(side_effect=RuntimeError("page broke"))
        with self.assertLogs(mr.logger, level="WARNING") as cm:
            n = mr.run_monitor_rank(w, build=build, environ=self.ENV,
                                    sleep=lambda s: None, hide_gpus=False)
        self.assertEqual(n, 0)
        self.assertTrue(any("page broke" in m for m in cm.output))

    def test_the_default_builder_is_the_shared_one(self):
        import inspect
        src = inspect.getsource(mr.run_monitor_rank)
        self.assertIn("from .hooks import build_products as build", src)


def _payloads(n=1):
    return ([{"save_args": (), "save_kwargs": {}}] * n) + [{"finish_run": True}]


class _Comm:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    def recv(self, source=None):
        return self._payloads.pop(0)

    def iprobe(self, source=None):
        return False


class _Reader:
    filename = "/run/dir/gf_prod.h5"

    def __init__(self):
        self.calls = 0

    def save_step_main(self, *a, **k):
        self.calls += 1


class SaverLoopTest(unittest.TestCase):
    """The REAL saver loop with a monitor rank configured."""

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"GF_MONITOR_AFTER_SAVE": "1",
                                                "GF_MONITOR_ITER": "1",
                                                "GF_MONITOR_SNAPSHOT": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(mr._reset_state)

    def _run(self, n):
        reader = _Reader()
        with mock.patch.object(hb, "_atomic_backup_copy"), \
                mock.patch.object(mon, "build_monitor") as bm:
            hb.save_to_backend_asynchronously_and_plot(
                reader, _Comm(_payloads(n)), main_rank=0, plot_container=None)
        return reader, bm

    def test_with_a_monitor_rank_the_saver_only_notifies(self):
        w = _World(rank=4, size=6)
        mr.split_monitor_rank(w, environ={"GF_MONITOR_RANK": "1"})
        reader, bm = self._run(3)
        self.assertEqual(reader.calls, 3)
        bm.assert_not_called()                              # nothing built here
        self.assertEqual([s[0]["saved"] for s in w.sent], [1, 2, 3])
        self.assertTrue(all(s[1] == 5 for s in w.sent))

    def test_without_one_the_in_place_hook_still_runs(self):
        reader, bm = self._run(1)
        bm.assert_called_once()


class WiringTest(unittest.TestCase):
    def _read(self, rel):
        with open(os.path.join(ROOT, rel)) as fh:
            return fh.read()

    def test_the_driver_splits_first_and_runs_on_the_run_comm(self):
        src = self._read("scripts/fstat_proposal/run_combined_staged.py")
        i_split = src.index("_monitor_rank.split_monitor_rank(MPI.COMM_WORLD)")
        self.assertLess(i_split, src.index("    fit = build_fit()"))
        self.assertIn("_monitor_rank.run_monitor_rank(MPI.COMM_WORLD)", src)
        self.assertIn("layout = prepare_rank(fit, _run_comm)", src)
        self.assertIn("if layout_dry_run(layout, _run_comm):", src)
        self.assertIn("fit.run(comm=_run_comm)", src)
        self.assertNotIn("prepare_rank(fit, MPI.COMM_WORLD)", src)
        self.assertIn("_monitor_rank.finish_run_ranks()", src)

    def test_both_v9_launchers_add_the_task_and_check_the_allocation(self):
        for rel in ("scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh",
                    "scripts/fstat_proposal/submit_gf_1yr_v9.sh",
                    "scripts/fstat_proposal/submit_gf_9mo_v9.sh",
                    # the flagship series (2026-10-11) inherits the block from the 9mo
                    *(f"scripts/flagship/submit_gf_{r}.sh"
                      for r in ("3mo", "6mo", "9mo", "12mo", "18mo", "24mo"))):
            s = self._read(rel)
            self.assertIn("export GF_MONITOR_RANK=${GF_MONITOR_RANK:-1}", s)
            self.assertIn("NTASKS=$(( NTASKS + 1 ))", s)
            self.assertIn('GF_MONITOR_RANK="${GF_MONITOR_RANK}"', s)   # --export
            self.assertIn('[ "${SLURM_NTASKS}" -ne $(( _RUN_TASKS + 1 )) ]', s)
            self.assertIn("export GF_MONITOR_RANK=0", s)
            self.assertIn("export GF_MONITOR_ITER=${GF_MONITOR_ITER:-10}", s)
            self.assertNotIn("export GF_MONITOR_ITER=3", s)

    def _allocation_check(self, ntasks, monitor="1"):
        s = self._read("scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh")
        start = s.index("# DEDICATED MONITOR RANK, checked against the GRANTED")
        end = s.index("\nfi\n", s.index("DISARMING", start)) + 4
        block = s[start:end]
        script = ("set -eu\nGF_LEGACY_RANK_LAYOUT=0\nN_COMPUTE_EFF=4\n"
                  + block + '\necho "RESULT=${GF_MONITOR_RANK}"\n')
        env = {"PATH": os.environ.get("PATH", ""), "SLURM_NTASKS": str(ntasks),
               "GF_MONITOR_RANK": monitor}
        out = subprocess.run(["bash", "-c", script], env=env, check=True,
                             capture_output=True, text=True).stdout
        return out.strip().splitlines()[-1]

    def test_the_allocation_check_keeps_it_with_the_extra_task(self):
        self.assertEqual(self._allocation_check(6), "RESULT=1")

    def test_the_allocation_check_disarms_it_without_one(self):
        self.assertEqual(self._allocation_check(5), "RESULT=0")

    def test_off_stays_off(self):
        self.assertEqual(self._allocation_check(6, monitor="0"), "RESULT=0")


_MPI_SCRIPT = textwrap.dedent('''
    import json, os, sys, time
    os.environ["GF_MONITOR_RANK"] = "1"
    os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
    os.environ["GF_MONITOR_ITER"] = "2"
    from mpi4py import MPI
    from lisatools.globalfit.monitor import rank as mr
    from lisatools.globalfit import hdfbackend as hb
    out_dir = sys.argv[1]
    world = MPI.COMM_WORLD
    run, mon = mr.split_monitor_rank(world)
    if run is None:
        def build(run_dir):
            with open(os.path.join(out_dir, "builds.txt"), "a") as fh:
                fh.write(run_dir + "\\n")
            time.sleep(1.5)            # a SLOW page: the saver must not wait
            return 1.5
        mr.run_monitor_rank(world, build=build, poll_s=0.05, hide_gpus=False)
        sys.exit(0)
    me, size = run.Get_rank(), run.Get_size()
    saver = size - 1
    if me == saver and len(sys.argv) > 2 and sys.argv[2] == "neg":
        # NEGATIVE CONTROL: the pre-2026-10-04 placement -- the same slow
        # build, but IN PLACE on the saver (no notices).
        from lisatools.globalfit.monitor import hooks
        def slow(run_dir, watchdog=None):
            time.sleep(1.5)
            return 1.5
        hooks.build_products = slow
    if me == saver:
        class R:
            filename = os.path.join(out_dir, "gf_prod.h5")
            def save_step_main(self, *a, **k):
                pass
        hb._atomic_backup_copy = lambda *a, **k: None
        _neg = len(sys.argv) > 2 and sys.argv[2] == "neg"
        _active = mr.active
        if _neg:
            mr.active = lambda: False     # the saver builds in place
        hb.save_to_backend_asynchronously_and_plot(R(), run, main_rank=0)
        mr.active = _active               # still joins the closing barrier
    elif me == 0:
        waits = []
        for _ in range(6):
            t = time.perf_counter()
            # 8 MB like a real pickled state: a large message is a
            # RENDEZVOUS send, which is what blocks on a busy saver (a
            # tiny one goes out eagerly and never waits, build or not).
            run.send({"save_args": (), "save_kwargs": {},
                      "blob": bytearray(8_000_000)}, dest=saver)
            waits.append(time.perf_counter() - t)
            time.sleep(0.4)
        run.send({"finish_run": True}, dest=saver)
        with open(os.path.join(out_dir, "waits.json"), "w") as fh:
            json.dump(waits, fh)
    mr.finish_run_ranks()
''')


@unittest.skipUnless(shutil.which("mpiexec"), "mpiexec not available")
class RealMpiTest(unittest.TestCase):
    """mpiexec -n 4: head 0, compute 1, saver 2, monitor 3 (world)."""

    def _run(self, *extra):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "mpi_monitor_rank.py")
        with open(path, "w") as fh:
            fh.write(_MPI_SCRIPT)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get(
            "PYTHONPATH", "")
        r = subprocess.run(
            ["mpiexec", "--oversubscribe", "-n", "4", sys.executable, path, d,
             *extra],
            env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        import json
        with open(os.path.join(d, "waits.json")) as fh:
            return d, json.load(fh)

    def test_NEGATIVE_CONTROL_the_in_place_build_does_hold_saves(self):
        """Same slow build on the SAVER: the head's send waits on it. If
        this ever passes under 0.5 s the timing test above proves nothing."""
        _d, waits = self._run("neg")
        self.assertGreater(max(waits), 0.5, waits)

    def test_saves_never_wait_on_a_slow_build_and_the_job_ends(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "mpi_monitor_rank.py")
        with open(path, "w") as fh:
            fh.write(_MPI_SCRIPT)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get(
            "PYTHONPATH", "")
        st = time.perf_counter()
        r = subprocess.run(
            ["mpiexec", "--oversubscribe", "-n", "4", sys.executable, path, d],
            env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        import json
        with open(os.path.join(d, "waits.json")) as fh:
            waits = json.load(fh)
        # A 1.5 s build ran while the head sent saves 0.4 s apart: with the
        # build on the saver, at least one send would wait ~1 s.
        self.assertLess(max(waits), 0.5, waits)
        with open(os.path.join(d, "builds.txt")) as fh:
            builds = fh.read().split()
        self.assertGreaterEqual(len(builds), 1)
        self.assertTrue(all(b == d for b in builds), builds)
        self.assertLess(time.perf_counter() - st, 170)


if __name__ == "__main__":
    unittest.main()
