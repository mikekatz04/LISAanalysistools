"""Tests for lisatools.globalfit.memwatch on a fake /proc tree (runs on any OS)."""

import os
import signal
import tempfile
import unittest

from lisatools.globalfit import memwatch as mw

SIG = getattr(signal, "SIGWINCH", 28)


def _write(root, rel, text, binary=False):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if binary else "w") as fh:
        fh.write(text)


class FakeProc:
    """A /proc tree: meminfo + processes {pid: (ppid, rss_kb, uid, cmdline)}."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def meminfo(self, total_gb, avail_gb, shmem_gb=3.0, sunreclaim_gb=1.5):
        kb = 1024 ** 2
        _write(self.root, "meminfo",
               f"MemTotal: {int(total_gb * kb)} kB\nMemFree: 1 kB\n"
               f"MemAvailable: {int(avail_gb * kb)} kB\nCached: {int(20 * kb)} kB\n"
               f"AnonPages: {int(150 * kb)} kB\nShmem: {int(shmem_gb * kb)} kB\n"
               f"SUnreclaim: {int(sunreclaim_gb * kb)} kB\nHugePages_Total: 0\n")

    def proc(self, pid, ppid, rss_gb, uid, cmd):
        _write(self.root, f"{pid}/status",
               f"Name:\tpython\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nPPid:\t{ppid}\n"
               f"VmRSS:\t{int(rss_gb * 1024 ** 2)} kB\n")
        _write(self.root, f"{pid}/cmdline", ("\x00".join(cmd) + "\x00").encode(), binary=True)

    def cleanup(self):
        self.tmp.cleanup()


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class MemWatchTest(unittest.TestCase):
    def setUp(self):
        self.fp = FakeProc()
        self.reg = tempfile.TemporaryDirectory()
        self.uid = os.getuid()
        self.me = os.getpid()
        self.fp.meminfo(640, 400)
        self.fp.proc(self.me, 1, 70, self.uid, ["python", "run_combined_staged.py"])
        # two trajectory-pool workers of this process, one grandchild
        self.fp.proc(90001, self.me, 2.0, self.uid, ["python", "-c", "spawn worker"])
        self.fp.proc(90002, self.me, 3.0, self.uid, ["python", "-c", "spawn worker"])
        self.fp.proc(90003, 90002, 1.0, self.uid, ["python", "-c", "grandchild"])
        # another rank on the node and a foreign user's process
        self.fp.proc(80000, 1, 71, self.uid, ["python", "run_combined_staged.py"])
        self.fp.proc(70000, 1, 500, self.uid + 1, ["foreign"])
        self.clock = Clock()
        self.lines = []
        self.kills = []

    def tearDown(self):
        self.fp.cleanup()
        self.reg.cleanup()

    def _watch(self, tag="w1", pid=None, **kw):
        kw.setdefault("node_reporter", False)
        kw.setdefault("registry_dir", self.reg.name)
        w = mw.MemWatch(tag, root=self.fp.root, clock=self.clock, wall=self.clock,
                        log=self.lines.append, kill=lambda p, s: self.kills.append((p, s)),
                        dump_signal=SIG, **kw)
        if pid is not None:
            w.pid = pid
        return w

    def test_meminfo_and_absent_proc(self):
        tot, av = mw.read_meminfo(self.fp.root)
        self.assertAlmostEqual(tot / mw.GB, 640, places=3)
        self.assertAlmostEqual(av / mw.GB, 400, places=3)
        f = mw.read_meminfo_fields(self.fp.root)
        self.assertAlmostEqual(f["Shmem"] / mw.GB, 3.0, places=3)
        self.assertEqual(f["HugePages_Total"], 0.0)              # a unitless field is kept as is
        self.assertIsNone(mw.read_meminfo(os.path.join(self.fp.root, "nope")))

    def test_first_line_reports_own_children_and_node(self):
        w = self._watch()
        out = w.sample_once()
        self.assertEqual(len(out), 1)
        line = out[0]
        self.assertIn(f"[GF_MEM] w1 pid {self.me}", line)
        self.assertIn("rss 70.0 GB", line)
        self.assertIn("children 6.0 GB (3)", line)          # 2 + 3 + the grandchild 1
        self.assertIn("node avail 400.0 of 640.0 GB", line)
        self.assertFalse(w.pressure)

    def test_quiet_until_a_threshold_then_a_line(self):
        w = self._watch(delta_gb=2.0, heartbeat_s=600.0)
        w.sample_once()
        self.clock.t = 30
        self.assertEqual(w.sample_once(), [])                 # nothing moved
        self.fp.proc(90002, self.me, 6.0, self.uid, ["python", "-c", "spawn worker"])
        self.clock.t = 60
        out = w.sample_once()                                 # children +3 GB > 2
        self.assertEqual(len(out), 1)
        self.assertIn("children 9.0 GB", out[0])
        self.clock.t = 90
        self.fp.meminfo(640, 395)                             # 5 GB node move > 2 x 2 GB
        self.assertEqual(len(w.sample_once()), 1)
        self.clock.t = 120
        self.fp.meminfo(640, 392)                             # 3 GB < 4 GB: quiet
        self.assertEqual(w.sample_once(), [])
        self.clock.t = 120 + 601
        self.assertEqual(len(w.sample_once()), 1)             # heartbeat

    def test_own_rss_growth_triggers(self):
        w = self._watch()
        w.sample_once()
        self.fp.proc(self.me, 1, 73, self.uid, ["python", "run_combined_staged.py"])
        self.clock.t = 30
        out = w.sample_once()
        self.assertEqual(len(out), 1)
        self.assertIn("rss 73.0 GB", out[0])

    def test_pressure_flag_switches_the_cadence(self):
        w = self._watch()
        w.sample_once()
        self.assertFalse(w.pressure)
        self.fp.meminfo(640, 150)                             # < 30 %: pressure, not low
        self.clock.t = 30
        w.sample_once()
        self.assertTrue(w.pressure)
        self.assertEqual(self.kills, [])

    def test_node_reporter_table_names_registered_processes_and_node_totals(self):
        w = self._watch(node_reporter=True, top_s=600.0, top_n=4)
        w.register()
        rank3 = self._watch("w3", pid=80000)
        rank3.register()
        out = w.sample_once()
        top = [ln for ln in out if "largest processes" in ln]
        self.assertEqual(len(top), 1)
        self.assertNotIn("LOW MEMORY", top[0])
        self.assertIn("pid 80000 [w3] 71.0 GB run_combined_staged.py", top[0])
        self.assertIn(f"pid {self.me} [w1] 70.0 GB", top[0])
        self.assertIn("pid 90002 [child of w1] 3.0 GB", top[0])
        self.assertIn("shmem 3.0", top[0])
        self.assertIn("slab unreclaimable 1.5", top[0])
        self.assertNotIn("foreign", top[0])                  # other users excluded
        self.assertLess(top[0].index("pid 80000"), top[0].index(f"pid {self.me}"))
        self.clock.t = 120
        self.assertFalse(any("largest" in ln for ln in w.sample_once()))   # within top_s

    def test_low_memory_table_once_per_window_across_processes(self):
        a = self._watch("w1", low_s=15.0)
        b = self._watch("w3", pid=80000, low_s=15.0)
        self.fp.meminfo(640, 50, shmem_gb=400.0)              # < 15 %
        self.clock.t = 100
        la = [ln for ln in a.sample_once() if "largest" in ln]
        lb = [ln for ln in b.sample_once() if "largest" in ln]
        self.assertEqual((len(la), len(lb)), (1, 0))          # one table per node per window
        self.assertIn("LOW MEMORY node", la[0])
        self.assertIn("shmem 400.0", la[0])
        self.clock.t = 116                                    # next 15 s window
        self.assertEqual(len([ln for ln in b.sample_once() if "largest" in ln]), 1)
        # every process logs its own line at the low cadence
        self.clock.t = 132
        self.assertTrue(any(f"pid {self.me} on" in ln for ln in a.sample_once()))

    def test_low_memory_signals_the_largest_registered_process_once(self):
        a = self._watch("w1", dump_gap_s=120.0)
        a.register()
        b = self._watch("w3", pid=80000, dump_gap_s=120.0)
        b.register()
        self.fp.proc(self.me, 1, 400, self.uid, ["python", "run_combined_staged.py"])
        self.fp.meminfo(640, 50)
        self.clock.t = 1000
        out = a.sample_once()
        b.sample_once()
        self.assertEqual(self.kills, [(self.me, SIG)])        # the biggest registered one, once
        self.assertTrue(any("asking pid" in ln and "[w1]" in ln for ln in out))
        self.clock.t = 1050                                   # same 120 s bucket: no repeat
        a.sample_once()
        self.assertEqual(len(self.kills), 1)
        self.clock.t = 1200                                   # next bucket: again
        b.sample_once()
        self.assertEqual(len(self.kills), 2)

    def test_low_memory_never_signals_an_unregistered_process(self):
        # the largest process is a pool worker (unregistered): its registered parent gets it
        a = self._watch("w1")
        a.register()
        self.fp.proc(90001, self.me, 300.0, self.uid, ["python", "-c", "spawn worker"])
        self.fp.meminfo(640, 50)
        self.clock.t = 500
        a.sample_once()
        self.assertEqual(self.kills, [(self.me, SIG)])
        # nothing registered at all: no signal
        self.kills.clear()
        empty = tempfile.TemporaryDirectory()
        try:
            c = self._watch("w7", registry_dir=empty.name)
            self.clock.t = 900
            c.sample_once()
            self.assertEqual(self.kills, [])
        finally:
            empty.cleanup()

    def test_start_is_off_without_proc_and_with_the_knob(self):
        mw._STARTED = None
        self.assertIsNone(mw.start("w0", root=os.path.join(self.fp.root, "nope")))
        old = os.environ.get("GF_MEMWATCH")
        os.environ["GF_MEMWATCH"] = "0"
        try:
            self.assertIsNone(mw.start("w0", root=self.fp.root))
        finally:
            if old is None:
                os.environ.pop("GF_MEMWATCH")
            else:
                os.environ["GF_MEMWATCH"] = old
            mw._STARTED = None

    def test_start_runs_a_daemon_thread_once_and_registers(self):
        mw._STARTED = None
        old = os.environ.get("GF_MEMWATCH_DUMP")
        os.environ["GF_MEMWATCH_DUMP"] = "0"                  # no faulthandler in the test runner
        try:
            w = mw.start("w0", root=self.fp.root, registry_dir=self.reg.name)
            self.assertIsNotNone(w)
            self.assertTrue(w.daemon)
            self.assertIsNone(w.dump_signal)
            self.assertIs(mw.start("w9", root=self.fp.root), w)     # idempotent
            self.assertIn(f"pid_{os.getpid()}", os.listdir(self.reg.name))
            w.stop()
            w.join(timeout=5)
            self.assertFalse(w.is_alive())
        finally:
            if old is None:
                os.environ.pop("GF_MEMWATCH_DUMP", None)
            else:
                os.environ["GF_MEMWATCH_DUMP"] = old
            mw._STARTED = None


if __name__ == "__main__":
    unittest.main()
