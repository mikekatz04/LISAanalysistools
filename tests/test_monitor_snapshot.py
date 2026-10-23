"""Snapshot tar production from the installed package.

``lisatools.globalfit.monitor.snapshot`` is ``make_snapshots.sh``
expressed in Python so the saver rank can build a snapshot without a
shell, a ``cd`` to the repo root, or a source checkout.

THE SHELL SCRIPT IS DELIBERATELY STILL THERE (user instruction
2026-09-26: "try not to touch any other code"), so two implementations
of one recipe exist. ``SnapshotMatchesTheShellRecipeTest`` compares the
parts that can silently diverge -- the kept log-line families and the
exclusion list -- so the pair cannot drift without a test saying so.
"""

from __future__ import annotations

import os
import re
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lisatools.globalfit.monitor import snapshot as snap

REPO = Path(__file__).resolve().parents[1]
SHELL = REPO / "scripts" / "fstat_proposal" / "make_snapshots.sh"


def _touch(p, size=16):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(b"x" * size)
    return p


class LiveStoreTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_it_ignores_backup_corrupt_and_extract(self):
        import time
        for name in ("gf_prod_testing.h5",
                     "gf_prod_testing_running_backup_copy.h5",
                     "gf_prod_testing_CORRUPT.h5",
                     "gf_prod_testing_extract.h5"):
            _touch(os.path.join(self.d, name))
            time.sleep(0.01)
        self.assertEqual(os.path.basename(snap._live_store(self.d)),
                         "gf_prod_testing.h5")

    def test_no_store_is_None_not_an_exception(self):
        self.assertIsNone(snap._live_store(self.d))


class MembersTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_run")
        for rel in (
            "gf_prod_testing.h5",                  # full store: OUT
            "gf_prod_testing_extract.h5",          # reduced store: IN
            "gf_prod_testing.h5.bak",              # backup copy: OUT
            "gf_prod_testing_midit_checkpoint.pkl",  # OUT
            "old_snapshot.tar.gz",                 # OUT
            "gf_artifacts/globalfit_run.log",      # IN
            "gf_artifacts/diagnostics/plot.png",   # OUT
            "dissect/dump.npz",                    # OUT
            "gb_fstat_fit/DONE.json",              # IN
            "gb_fstat_fit/peaks_stacked.npz",      # OUT (payload)
            "fstat_grid_parts/part0.npz",          # OUT
            "run_settings.log",                    # IN
        ):
            _touch(os.path.join(self.run, rel))

    def _names(self, include_fstat=False):
        got = snap._members(self.run, include_fstat, None)
        return sorted(os.path.relpath(p, self.run) for p in got)

    def test_the_exclusions_hold(self):
        self.assertEqual(self._names(), [
            "gb_fstat_fit/DONE.json",
            "gf_artifacts/globalfit_run.log",
            "gf_prod_testing_extract.h5",
            "run_settings.log",
        ])

    def test_the_full_store_is_out_and_the_extract_is_in(self):
        names = self._names()
        self.assertIn("gf_prod_testing_extract.h5", names)
        self.assertNotIn("gf_prod_testing.h5", names)

    def test_INCLUDE_FSTAT_keeps_the_payloads(self):
        self.assertIn("gb_fstat_fit/peaks_stacked.npz",
                      self._names(include_fstat=True))

    def test_the_grid_parts_stay_out_even_with_INCLUDE_FSTAT(self):
        """The one family that is excluded unconditionally -- it is the
        heaviest thing in the directory."""
        self.assertNotIn("fstat_grid_parts/part0.npz",
                         self._names(include_fstat=True))

    def test_an_explicitly_skipped_raw_log_is_dropped(self):
        raw = os.path.join(self.run, "gf_artifacts", "globalfit_run.log")
        got = snap._members(self.run, False, raw)
        self.assertNotIn(raw, got)


class BuildSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_run")
        _touch(os.path.join(self.run, "gf_prod_testing.h5"))
        _touch(os.path.join(self.run, "run_settings.log"))

    def _fake_extract(self, src, dst, keep, cold_keep=None):
        _touch(dst, 32)

    def test_it_writes_a_tar_containing_the_extract(self):
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=self._fake_extract):
            out = snap.build_snapshot(self.run)
        self.assertIsNotNone(out)
        self.assertTrue(out.endswith("gf_run_snapshot.tar.gz"))
        with tarfile.open(out) as tf:
            names = tf.getnames()
        self.assertIn("gf_run/gf_prod_testing_extract.h5", names)
        self.assertIn("gf_run/run_settings.log", names)
        self.assertNotIn("gf_run/gf_prod_testing.h5", names)

    def test_it_publishes_atomically(self):
        """A consumer polling for the tar must never see a partial one."""
        seen = {}
        real = os.replace

        def spy(a, b):
            seen["from"], seen["to"] = a, b
            return real(a, b)

        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=self._fake_extract), \
                mock.patch.object(snap.os, "replace", side_effect=spy):
            out = snap.build_snapshot(self.run)
        self.assertEqual(seen["from"], out + ".tmp")
        self.assertEqual(seen["to"], out)

    def test_a_failed_extract_returns_None_and_does_not_raise(self):
        """This runs on the run's only writer. A failed diagnostic must
        never become a failed run."""
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=RuntimeError("boom")):
            self.assertIsNone(snap.build_snapshot(self.run))

    def test_no_live_store_returns_None_and_does_not_raise(self):
        empty = os.path.join(self.d, "empty")
        os.makedirs(empty)
        self.assertIsNone(snap.build_snapshot(empty))

    def test_no_tmp_file_is_left_behind_on_failure(self):
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=RuntimeError("boom")):
            snap.build_snapshot(self.run)
        self.assertFalse(os.path.exists(self.run + "_snapshot.tar.gz.tmp"))


class LogFilterTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.run = os.path.join(self.d, "gf_run")
        self.log = os.path.join(self.run, "gf_artifacts", "globalfit_run.log")

    def _write(self, mb):
        body = (b"[SAVE] keep me\nnoise line drop me\nWARNING keep me too\n")
        _touch(self.log, 1)
        with open(self.log, "wb") as fh:
            fh.write(body)
            fh.write(b"filler drop\n" * (mb * 1048576 // 12))

    def test_below_the_cap_the_raw_log_ships_whole(self):
        self._write(0)
        self.assertIsNone(snap._filter_run_log(self.run, log_cap_mb=200))

    def test_above_the_cap_it_writes_filtered_plus_tail_and_excludes_raw(self):
        self._write(3)
        skip = snap._filter_run_log(self.run, log_cap_mb=1)
        self.assertEqual(skip, self.log)
        base = self.log[:-4]
        self.assertTrue(os.path.exists(base + "_filtered.log"))
        self.assertTrue(os.path.exists(base + "_tail.log"))
        kept = open(base + "_filtered.log", "rb").read()
        self.assertIn(b"[SAVE] keep me", kept)
        self.assertIn(b"WARNING keep me too", kept)
        self.assertNotIn(b"filler drop", kept)

    def test_a_filtering_failure_ships_the_log_whole_rather_than_nothing(self):
        self._write(3)
        with mock.patch("builtins.open", side_effect=OSError("nope")):
            self.assertIsNone(snap._filter_run_log(self.run, log_cap_mb=1))


class SnapshotMatchesTheShellRecipeTest(unittest.TestCase):
    """Anti-drift between the Python port and make_snapshots.sh."""

    @classmethod
    def setUpClass(cls):
        cls.sh = SHELL.read_text() if SHELL.exists() else ""

    def test_the_shell_script_is_still_there_to_compare_against(self):
        """If it is ever deleted, delete this class too -- do not let it
        pass vacuously."""
        self.assertTrue(self.sh, f"{SHELL} missing")

    def test_the_kept_log_line_families_match(self):
        """The grep -aE alternation and LOG_KEEP_PATTERN must select the
        same line families, or a filtered snapshot silently loses a
        diagnostic the page reads."""
        m = re.search(r'grep -aE "([^"]+)"', self.sh, re.S)
        self.assertIsNotNone(m, "the grep -aE alternation moved")
        shell_alts = {a for a in m.group(1).replace("\\\n", "").split("|") if a}
        py_alts = {a for a in snap.LOG_KEEP_PATTERN.split("|") if a}
        self.assertEqual(shell_alts, py_alts)

    def test_the_heavy_exclusions_match(self):
        for frag in ("fstat_grid_parts", "dissect", "_midit_checkpoint.pkl"):
            self.assertIn(frag, self.sh)
            self.assertTrue(
                any(frag in x for x in
                    snap._EXCLUDE_PATH_PARTS + snap._EXCLUDE_SUFFIXES),
                f"{frag} excluded by the shell but not by the port")

    def test_the_default_keep_windows_match(self):
        """The RESOLVED default, not the signature literal.

        ``keep``/``cold_keep`` are ``None`` sentinels since 2026-09-28
        (short mode resolves them to 1, and the previous
        ``if keep == 5`` override silently demoted an EXPLICIT keep=5).
        The property this test exists for -- the Python default equals
        the shell recipe's -- is unchanged, so check it by CALLING
        rather than by reading the declaration.
        """
        import tempfile as _tf
        from unittest import mock as _mock
        self.assertIn("KEEP=${KEEP:-5}", self.sh)
        self.assertIn("COLD_KEEP=${COLD_KEEP:-12}", self.sh)
        seen = {}

        def _fake(src, dst, keep, cold_keep=None):
            seen["k"] = (keep, cold_keep)
            _touch(dst, 32)

        d = _tf.mkdtemp()
        run = os.path.join(d, "gf_run")
        _touch(os.path.join(run, "gf_prod_testing.h5"), 64)
        with _mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=_fake):
            snap.build_snapshot(run)
        self.assertEqual(seen["k"], (5, 12))

    def test_the_log_cap_matches(self):
        self.assertIn("LOG_CAP_MB=${LOG_CAP_MB:-200}", self.sh)
        import inspect
        self.assertEqual(
            inspect.signature(snap.build_snapshot)
            .parameters["log_cap_mb"].default, 200)


if __name__ == "__main__":
    unittest.main()


class ShortSnapshotTest(unittest.TestCase):
    """``_short.tar.gz`` -- an ALLOWLIST of logs + latest state.

    USER RULING 2026-09-28, after the audit measured the first short
    tar at 129 MB of which ~95% was filtered dead-job and duplicate
    logs, with LOG_KEEP_PATTERN dropping every gate family
    (GB_GATE, V9-STAGE, GB_IMCONV, GB_STAGE, GF_TIMING). The recipe is
    now explicit about what goes IN rather than filtering what to leave
    out.
    """

    #: Fixture "one saved iteration" in bytes. Small ON PURPOSE: the
    #: real 35 MB constant made each setUp write ~125 MB, and with no
    #: cleanup this class alone leaked ~28 GiB of /tmp over one
    #: afternoon of mutation runs and filled the disk. build_snapshot
    #: takes ``young_bytes`` so the threshold can be tested without
    #: writing the real thing.
    YOUNG = 4096

    def setUp(self):
        import shutil as _shutil
        self.d = tempfile.mkdtemp()
        self.addCleanup(_shutil.rmtree, self.d, ignore_errors=True)
        self.run = os.path.join(self.d, "gf_run")
        # three jobs; only the newest should ship
        _touch(os.path.join(self.run, "slurm_stdout_640.log"), 8192)
        _touch(os.path.join(self.run, "slurm_stdout_650.log"), 8192)
        _touch(os.path.join(self.run, "slurm_stdout_662.log"), 8192)
        for i, n in ((640, 1), (650, 2), (662, 3)):
            os.utime(os.path.join(self.run, f"slurm_stdout_{i}.log"),
                     (10 ** 9 + n, 10 ** 9 + n))
        _touch(os.path.join(self.run, "gf_prod_testing.h5"), 4096)
        _touch(os.path.join(self.run, "gf_monitor.html"), 2048)
        _touch(os.path.join(self.run, "run_settings.log"), 128)
        _touch(os.path.join(self.run, "gb_setup.log"), 256)
        _touch(os.path.join(self.run, "gf_prod_eigen_tables.pkl"), 512)
        _touch(os.path.join(self.run, "gpu_util_662.csv"), 1024)
        _touch(os.path.join(self.run, "gpu_util_650.csv"), 1024)
        _touch(os.path.join(self.run, "gpu_procs_662.csv"), 1024)
        _touch(os.path.join(self.run, "gb_fstat_fit/shared_search/"
                                      "epoch_0011/DONE.json"), 64)
        _touch(os.path.join(self.run, "gb_truth_3to21.npz"), 4096)
        _touch(os.path.join(self.run, "warmstart_v7.npz"), 4096)
        _touch(os.path.join(self.run, "gf_artifacts/globalfit_run.log"), 4096)
        _touch(os.path.join(self.run, "gf_artifacts/globalfit_run.rank1.log"),
               4096)

    def _fake_extract(self, src, dst, keep, cold_keep=None):
        self.seen_keep = (keep, cold_keep)
        _touch(dst, 32)

    def _build(self, **kw):
        kw.setdefault("young_bytes", self.YOUNG)
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=self._fake_extract):
            return snap.build_snapshot(self.run, **kw)

    def _names(self, **kw):
        with tarfile.open(self._build(**kw)) as tf:
            return set(n.replace("gf_run/", "", 1) for n in tf.getnames())

    # -- identity -------------------------------------------------------
    def test_it_is_a_SEPARATE_file_from_the_full_snapshot(self):
        self.assertTrue(self._build(short=True).endswith("gf_run_short.tar.gz"))

    def test_the_default_name_is_unchanged_without_the_flag(self):
        self.assertTrue(self._build().endswith("gf_run_snapshot.tar.gz"))

    # -- (a) the newest stdout, RAW -------------------------------------
    def test_only_the_NEWEST_stdout_ships_and_it_is_RAW(self):
        n = self._names(short=True)
        self.assertIn("slurm_stdout_662.log", n)
        self.assertNotIn("slurm_stdout_650.log", n)
        self.assertNotIn("slurm_stdout_640.log", n)
        self.assertFalse([x for x in n if x.endswith(("_filtered.log",
                                                      "_tail.log"))],
                         "short mode must produce no reduced logs at all")

    def test_a_YOUNG_newest_job_brings_the_previous_one_along(self):
        """So the tar always spans at least one saved iteration."""
        with open(os.path.join(self.run, "slurm_stdout_662.log"), "wb") as fh:
            fh.write(b"\0" * 512)                 # < YOUNG
        os.utime(os.path.join(self.run, "slurm_stdout_662.log"),
                 (10 ** 9 + 3, 10 ** 9 + 3))
        n = self._names(short=True)
        self.assertIn("slurm_stdout_662.log", n)
        self.assertIn("slurm_stdout_650.log", n)
        self.assertNotIn("slurm_stdout_640.log", n, "only ONE extra")

    # -- (b)..(e) the rest of the allowlist -----------------------------
    def test_it_keeps_ONE_iteration_of_state(self):
        self._build(short=True)
        self.assertEqual(self.seen_keep, (1, 1))

    def test_an_explicit_keep_still_overrides_the_preset(self):
        self._build(short=True, keep=5)
        self.assertEqual(self.seen_keep[0], 5)

    def test_the_allowlist_members_are_all_present(self):
        n = self._names(short=True)
        for want in ("gf_prod_testing_extract.h5", "gf_monitor.html",
                     "run_settings.log", "gb_setup.log",
                     "gf_prod_eigen_tables.pkl", "gpu_util_662.csv",
                     "gb_fstat_fit/shared_search/epoch_0011/DONE.json"):
            self.assertIn(want, n, want)

    # -- the drops ------------------------------------------------------
    def test_the_dropped_families_are_all_absent(self):
        n = self._names(short=True)
        for gone in ("gb_truth_3to21.npz", "warmstart_v7.npz",
                     "gpu_procs_662.csv", "gpu_util_650.csv",
                     "gf_artifacts/globalfit_run.log",
                     "gf_artifacts/globalfit_run.rank1.log",
                     "gf_prod_testing.h5"):
            self.assertNotIn(gone, n, gone)

    def test_the_result_is_actually_small(self):
        out = self._build(short=True)
        self.assertLess(os.path.getsize(out), 2 * 1048576,
                        "a 'short' tar that is not short is not short")

    def test_it_never_raises(self):
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=RuntimeError("boom")):
            self.assertIsNone(snap.build_snapshot(self.run, short=True))

    # -- the full build must not inherit short mode's leftovers ---------
    def test_a_FULL_build_after_a_SHORT_one_ships_no_reduced_logs(self):
        self._build(short=True)
        n = self._names()
        self.assertFalse([x for x in n if x.endswith(("_filtered.log",
                                                      "_tail.log"))])
        self.assertIn("slurm_stdout_662.log", n, "the full tar keeps them all")
        self.assertIn("slurm_stdout_640.log", n)

    def test_a_STALE_reduced_log_is_dropped_but_the_run_log_PAIR_is_kept(self):
        """⚠ The full build DELIBERATELY ships the run-log pair when the
        run log is over the cap (SnapshotMatchesTheShellRecipeTest pins
        it). Only products this call did NOT write are stale."""
        _touch(os.path.join(self.run, "slurm_stdout_650_filtered.log"), 512)
        _touch(os.path.join(self.run, "slurm_stdout_650_tail.log"), 512)
        big = os.path.join(self.run, "gf_artifacts/globalfit_run.log")
        # _filter_run_log tests `getsize // 1048576 > log_cap_mb`, so the
        # fixture needs to round to at least 1 MB; log_cap_mb=0 then puts
        # it over the cap without writing the real 200 MB.
        with open(big, "wb") as fh:
            fh.write(b"WARNING x\n" * 110_000)        # ~1.1 MB
        n = self._names(log_cap_mb=0)
        self.assertIn("gf_artifacts/globalfit_run_filtered.log", n)
        self.assertIn("gf_artifacts/globalfit_run_tail.log", n)
        self.assertNotIn("gf_artifacts/globalfit_run.log", n)
        self.assertNotIn("slurm_stdout_650_filtered.log", n)
        self.assertNotIn("slurm_stdout_650_tail.log", n)


class OutputDataSnapshotTest(unittest.TestCase):
    """``_output_data.tar.gz`` -- the store extract and the residual
    snapshot, nothing else (user request 2026-10-20: "data only, no logs").

    The keep window must reach the snapshot's ``store_row``: the snapshot is
    written every few rows, so a one-row extract (the short tar) can miss
    the row the residual belongs to.
    """

    def setUp(self):
        import shutil as _shutil

        import h5py
        import numpy as np

        self.d = tempfile.mkdtemp()
        self.addCleanup(_shutil.rmtree, self.d, ignore_errors=True)
        self.run = os.path.join(self.d, "gf_run")
        os.makedirs(self.run)
        self.store = os.path.join(self.run, "gf_prod_testing.h5")
        # 16 allocated rows, 10 stored; row r's two walkers have lnL -r-0.1, -r-0.2
        self.log_like = -(np.arange(16)[:, None] + np.array([0.1, 0.2])).reshape(16, 1, 1, 2)
        with h5py.File(self.store, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = 10
            g.create_dataset("log_like", data=self.log_like)
        self.sidecar = os.path.join(self.run, "gf_prod_testing_residual_snapshot.npz")
        self._set(store_row=7)
        # no real waiting: a test that waits sets its own budget and a fake clock
        _p = mock.patch.object(snap, "_OUTPUT_DATA_WAIT_S", 0.0)
        _p.start()
        self.addCleanup(_p.stop)
        _touch(os.path.join(self.run, "wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5"), 64)
        for rel in ("slurm_stdout_662.log", "run_settings.log", "gb_setup.log",
                    "gf_prod_eigen_tables.pkl", "gb_truth_3to21.npz",
                    "gf_artifacts/globalfit_run.log", "gpu_util_662.csv",
                    "gb_fstat_fit/shared_search/epoch_0011/DONE.json"):
            _touch(os.path.join(self.run, rel), 256)

    def _fake_extract(self, src, dst, keep, cold_keep=None, **filters):
        self.seen_keep = (keep, cold_keep)
        self.seen_filters = filters
        _touch(dst, 32)

    def _build(self, **kw):
        with mock.patch(
                "lisatools.globalfit.monitor._store_extract.extract",
                side_effect=self._fake_extract):
            return snap.build_snapshot(self.run, **kw)

    def _names(self, **kw):
        with tarfile.open(self._build(output_data=True, **kw)) as tf:
            return sorted(n.replace("gf_run/", "", 1) for n in tf.getnames())

    def test_it_ships_the_extract_the_residual_snapshot_and_the_lookup_table_only(self):
        self.assertEqual(self._names(), [
            "gf_prod_testing_extract.h5",
            "gf_prod_testing_residual_snapshot.npz",
            "wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5",
        ])

    def test_it_is_its_own_file(self):
        self.assertTrue(self._build(output_data=True)
                        .endswith("gf_run_output_data.tar.gz"))

    def test_the_cold_chains_keep_every_row_and_the_tempered_window_reaches_the_snapshot_row(self):
        """10 rows stored, snapshot at row 7: every row of the cold chains (the
        posterior samples), the tempered arrays from row 7 to row 9, and the
        extract pinned to those 10 rows."""
        self._build(output_data=True)
        self.assertEqual(self.seen_keep, (3, 10))
        self.assertEqual(self.seen_filters["iteration"], 10)

    def test_an_explicit_keep_still_wins(self):
        self._build(output_data=True, keep=2, cold_keep=3)
        self.assertEqual(self.seen_keep, (2, 3))

    def test_without_a_snapshot_it_ships_the_extract_alone(self):
        os.remove(self.sidecar)
        self.assertEqual(self._names(), ["gf_prod_testing_extract.h5",
                                         "wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5"])
        self.assertEqual(self.seen_keep, (1, 10))

    def test_short_and_output_data_together_build_nothing(self):
        self.assertIsNone(self._build(output_data=True, short=True))

    def _set(self, iteration=None, store_row=None, lnl_row=None):
        """Move the store to ``iteration`` rows and/or write a snapshot of
        ``store_row`` carrying the lnL of ``lnl_row`` (default: its own row)."""
        import h5py
        import numpy as np

        if iteration is not None:
            with h5py.File(self.store, "a") as f:
                f["global_fit"].attrs["iteration"] = iteration
        if store_row is not None:
            r = store_row if lnl_row is None else lnl_row
            np.savez(self.sidecar, store_row=store_row, lnl=self.log_like[r, 0, 0])

    def _shipped_row(self, out):
        import io

        import numpy as np

        with tarfile.open(out) as tf:
            return int(np.load(io.BytesIO(tf.extractfile(
                "gf_run/gf_prod_testing_residual_snapshot.npz").read()))["store_row"])

    def test_the_snapshot_shipped_is_the_one_taken_before_the_extract(self):
        """The run overwrites the snapshot while a long extract reads (the
        9-month tar of 2026-10-22 shipped a 2016-row extract with a snapshot
        from row 2021): the tar carries the copy the extract is pinned to."""
        def extract_while_the_run_moves_on(src, dst, keep, cold_keep=None, **filters):
            self._fake_extract(src, dst, keep, cold_keep, **filters)
            self._set(iteration=13, store_row=12)

        with mock.patch("lisatools.globalfit.monitor._store_extract.extract",
                        side_effect=extract_while_the_run_moves_on):
            out = snap.build_snapshot(self.run, output_data=True)
        self.assertEqual(self._shipped_row(out), 7)
        self.assertEqual(self.seen_filters["iteration"], 10)
        self.assertEqual(sorted(os.listdir(self.d)), ["gf_run", "gf_run_output_data.tar.gz"])

    def _waiting(self, on_sleep, budget=600.0):
        clock = iter(range(0, 100000, 30))
        return mock.patch.multiple(
            "lisatools.globalfit.monitor.snapshot.time",
            sleep=mock.Mock(side_effect=on_sleep),
            monotonic=mock.Mock(side_effect=lambda: next(clock))), \
            mock.patch.object(snap, "_OUTPUT_DATA_WAIT_S", budget)

    def test_it_waits_for_the_saver_to_store_the_snapshot_row(self):
        self._set(store_row=12)
        a, b = self._waiting(lambda s: self._set(iteration=13))
        with a, b, mock.patch("builtins.print"):
            self._build(output_data=True)
        self.assertEqual(self.seen_filters["iteration"], 13)
        self.assertEqual(self.seen_keep, (1, 13))

    def test_a_stale_snapshot_is_not_shipped_while_the_run_writes_a_new_one(self):
        """The row it names is stored but holds other lnL (a snapshot left from
        an earlier trajectory): wait for the run's next snapshot."""
        self._set(store_row=8, lnl_row=14)
        a, b = self._waiting(lambda s: self._set(store_row=9))
        with a, b, mock.patch("builtins.print"):
            out = self._build(output_data=True)
        self.assertEqual(self._shipped_row(out), 9)
        self.assertEqual(self.seen_keep, (1, 10))

    def test_a_lagging_saver_is_awaited_not_chased(self):
        """Row 12 not stored yet and the head writes row 15 meanwhile: the
        copy of row 12 ships once the saver stores it."""
        self._set(store_row=12)
        steps = iter([lambda: self._set(store_row=15), lambda: self._set(iteration=13)])
        a, b = self._waiting(lambda s: next(steps, lambda: None)())
        with a, b, mock.patch("builtins.print"):
            out = self._build(output_data=True)
        self.assertEqual(self._shipped_row(out), 12)
        self.assertEqual(self.seen_filters["iteration"], 13)

    def test_after_the_wait_it_ships_what_it_has_and_says_why(self):
        self._set(store_row=8, lnl_row=4)
        with mock.patch.object(snap, "_OUTPUT_DATA_WAIT_S", 0.0), \
                self.assertLogs(snap.logger, "WARNING") as logs:
            self.assertIsNotNone(self._build(output_data=True))
        self.assertIn("does not match the stored row 8 (it matches the stored row 4)",
                      logs.output[0])

    def test_the_wait_says_so_and_reports_every_minute(self):
        """A wait is printed, not logged at INFO (which the CLI does not show),
        with the store's last-write age so a stopped run is recognisable."""
        self._set(store_row=12)
        a, b = self._waiting(None, budget=200.0)
        with a, b, mock.patch("builtins.print") as out:
            self._build(output_data=True)
        said = [c.args[0] for c in out.call_args_list if "waiting for" in str(c.args)]
        self.assertIn("its row 12 is not stored yet; the store holds 10 rows", said[0])
        self.assertIn("last written", said[0])
        self.assertGreater(len(said), 1)

    def test_it_gives_up_after_the_wait_and_still_builds(self):
        self._set(store_row=12)
        with mock.patch.object(snap, "_OUTPUT_DATA_WAIT_S", 0.0):
            self.assertIsNotNone(self._build(output_data=True))
        self.assertEqual(self.seen_filters["iteration"], 10)

    def test_the_extract_is_written_at_maximal_compression(self):
        self._build(output_data=True)
        self.assertEqual((self.seen_filters["gzip_level"], self.seen_filters["shuffle"]), (9, True))


class ExtractFiltersTest(unittest.TestCase):
    """``extract(gzip_level=, shuffle=)``: the output-data filters on a real
    store; the default stays gzip level 4 without shuffle."""

    def setUp(self):
        import shutil as _shutil

        import h5py
        import numpy as np

        self.d = tempfile.mkdtemp()
        self.addCleanup(_shutil.rmtree, self.d, ignore_errors=True)
        self.src = os.path.join(self.d, "s_testing.h5")
        self.chain = np.random.default_rng(1).normal(size=(6, 1, 1, 2, 3, 4))
        with h5py.File(self.src, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = 6
            g.create_dataset("chain/mbh", data=self.chain)
            g.create_dataset("saved_after", data=["in_model"] * 6,
                             dtype=h5py.string_dtype())

    def _extract(self, **kw):
        import h5py

        from lisatools.globalfit.monitor._store_extract import extract

        dst = os.path.join(self.d, "s_testing_extract.h5")
        with mock.patch("builtins.print"):
            extract(self.src, dst, 1, **kw)
        return h5py.File(dst, "r")

    def test_level_9_with_shuffle_on_numbers_never_on_strings(self):
        import numpy as np

        with self._extract(gzip_level=9, shuffle=True) as f:
            d = f["global_fit/chain/mbh"]
            self.assertEqual((d.compression, d.compression_opts, d.shuffle), ("gzip", 9, True))
            np.testing.assert_array_equal(d[...], self.chain)
            self.assertFalse(f["global_fit/saved_after"].shuffle)

    def test_iteration_pins_the_rows_and_is_recorded(self):
        with self._extract(iteration=4, cold_keep=4) as f:
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 4)

    def test_a_long_window_reports_its_progress(self):
        import h5py
        import numpy as np

        with h5py.File(self.src, "a") as f:
            f["global_fit"].attrs["iteration"] = 250
            f["global_fit"].create_dataset("chain/gb", data=np.ones((250, 1, 1, 2, 3, 4)))
        from lisatools.globalfit.monitor import _store_extract as se

        with mock.patch.object(se, "LISTED_CAP", 0), mock.patch("builtins.print") as out:
            se.extract(self.src, os.path.join(self.d, "x.h5"), 1, cold_keep=250)
        lines = [str(c.args[0]) for c in out.call_args_list]
        self.assertEqual([m.split(" (")[0] for m in lines if "..." in m],
                         ["  ... global_fit/chain/gb row 100/250",
                          "  ... global_fit/chain/gb row 200/250"])

    def test_the_default_is_unchanged(self):
        with self._extract() as f:
            d = f["global_fit/chain/mbh"]
            self.assertEqual((d.compression_opts, d.shuffle), (4, False))


class OutputDataCLITest(unittest.TestCase):
    """``python -m lisatools.globalfit.monitor RUN_DIR --output-data``."""

    def _main(self, *argv):
        from lisatools.globalfit.monitor import __main__ as cli

        with mock.patch.object(cli, "build_snapshot", return_value="x.tar.gz") as bs, \
                mock.patch.object(cli, "build_monitor") as bm, \
                mock.patch.object(cli, "build_short_monitor") as bsm, \
                mock.patch.object(cli, "check_truth", return_value=(True, "ok")), \
                mock.patch.object(cli, "describe_run", return_value=""), \
                mock.patch.object(cli, "parse_add_info", return_value=[]):
            rc = cli.main(list(argv))
        return rc, bs, bm, bsm

    def test_it_builds_the_data_tar_and_no_page(self):
        rc, bs, bm, bsm = self._main("RUN", "--output-data")
        self.assertEqual(rc, 0)
        bm.assert_not_called()
        bsm.assert_not_called()
        self.assertTrue(bs.call_args.kwargs["output_data"])
        self.assertFalse(bs.call_args.kwargs["short"])

    def test_verbose_shows_the_monitor_info_log(self):
        import logging

        lg = logging.getLogger("lisatools.globalfit.monitor")
        self.addCleanup(lg.setLevel, lg.level)
        with mock.patch("logging.basicConfig"):
            self._main("RUN", "--output-data", "-v")
        self.assertEqual(lg.level, logging.INFO)

    def test_it_refuses_the_other_tar_and_page_flags(self):
        for other in ("--short", "--short-page", "--snapshot"):
            with self.assertRaises(SystemExit):
                self._main("RUN", "--output-data", other)
