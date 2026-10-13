"""Other runs overlaid on the full monitor page (``monitor/_add_info.py``).

User request 2026-10-12: "make a change to the monitor call to add a second run
to the same html only in specific places where it is reasonable visually to
compare between runs, so the base of everything stays the specific one you want
(9 mo in this case), but I want to plot the phase maximized overlap plots, a
comparison of the final psd and foreground measurements (final for the previous
run and current for the ongoing run) ... it should be able to add a number of
runs. It should be something like --add-info /path/to/gf_prod_6mo_v9_4gpu."

Pinned here:
  * the knob: the repeatable CLI flag and the GF_MONITOR_ADD_INFO list, order
    kept, duplicates (and the base run) dropped, the flag winning, ``none``
    silencing; the CLI and from_tar hand the list to the build, and the in-run
    route (no list) reaches the generator through the inherited environment;
  * reading an added run: its last stored row only (torn-row step-back), the
    max-lnL walker's leaves, the last noise rows, its own Tobs; a missing dir /
    store is skipped with a MISSING line;
  * matching an added run against ITS OWN truth set through the real GBGPU
    path; no truth set / a Tobs mismatch / the CWD's truth set is skipped;
  * the generator's wiring: an overlay call before exactly the eight listed
    figures and no other, every one behind ``if ADDED``, the caption sentence on
    exactly those panels;
  * the generator's own figure blocks, executed on synthetic arrays: the added
    runs' legend entries land on exactly those figures, two runs get two
    distinct entries, a skipped run draws nothing, and an empty list renders
    the figure byte-identical to the block without the overlay lines.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import tempfile
import textwrap
import unittest
from unittest import mock

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

import lisatools.globalfit.monitor as mon  # noqa: E402
from lisatools.globalfit.monitor import _add_info as add  # noqa: E402
from lisatools.globalfit.monitor import _short as sh  # noqa: E402
from tests.test_monitor_short_page import NIT, NW, _make_run  # noqa: E402

with open(mon.generator_path(), encoding="utf-8") as _fh:
    SRC = _fh.read()

#: the figures an added run is drawn on, and nothing else
OVERLAY_KEYS = {"f3_match", "f6_snr", "f5_counts", "f11_psd", "f11_fg", "psd_curves",
                "psd_hist", "gal_hist"}
BASE = "9mo_v9"
COLORS = dict(CYAN="#4FD8EB", AMBER="#F5A623", GREEN="#58C48A", RED="#E5484D",
              VIOLET="#9B7BFF", FG="#B8C6D4", DIM="#67788A", BG="#0A0E14",
              PANEL="#10161F", LINE="#223041")


def legend_labels(ax):
    leg = ax.get_legend()
    return [] if leg is None else [t.get_text() for t in leg.get_texts()]


def block(start, end, src=SRC):
    """The generator's source from the line holding ``start`` through ``end``,
    dedented (``start`` must occur once in the file)."""
    k = src.index(start)
    assert src.count(start) == 1, start
    i = src.rindex("\n", 0, k) + 1
    j = src.index(end, k) + len(end)
    return textwrap.dedent(src[i:j])


def strip_overlays(code):
    """``code`` without its ``if ADDED ...`` overlay statements (the block as
    it was before the overlays existed)."""
    out, skip_indent = [], None
    for line in code.split("\n"):
        ind = len(line) - len(line.lstrip())
        if skip_indent is not None:
            if line.strip() and ind <= skip_indent:
                skip_indent = None
            else:
                continue
        if re.match(r"\s*if ADDED\b", line):
            skip_indent = ind
            continue
        out.append(line)
    return "\n".join(out)


def synth_added(k, label, *, nw=4, rec=True, mm_n=40, ndet=60, row=500, skip=None,
                seed=None):
    """An added-run record as ``load_added_runs`` + ``compute_recoveries`` leave it."""
    rng = np.random.default_rng(100 + k if seed is None else seed)
    A = dict(dir=f"/runs/{label}", label=label, style=add.style(k), skip=skip,
             rec=None, rec_skip=None, notes=[])
    if skip is not None:
        return A
    A.update(store_name=f"gf_prod_{label}_testing.h5", row=row, nit=row + 1, wb=0,
             nwalk=nw, noise_row=row, tobs=180 * 86400.0, galfor_log=True,
             psd=np.stack([np.column_stack([1.5e-11 * (1 + 2e-3 * rng.standard_normal(nw)),
                                            3e-15 * (1 + 5e-3 * rng.standard_normal(nw))])
                           for _ in range(3)]),
             gal_phys=np.tile(np.array([10 ** -43.8, 10 ** -2.6, 5.0, 10 ** -2.0, 10 ** -2.85]),
                              (3, nw, 1)) * (1 + 1e-2 * rng.standard_normal((3, nw, 5))))
    if rec:
        snr = rng.uniform(7, 60, ndet)
        ti = np.sort(rng.choice(ndet, mm_n, replace=False))
        mm = np.clip(rng.uniform(0.5, 1.05, mm_n), 0, 1)
        A["rec"] = dict(ndet=ndet, band=(0.8e-3, 21.94e-3), walker=0, row=row, thresh=0.8,
                        mm=mm, ti=ti, ti_ov=ti[mm >= 0.8], det_f0=np.geomspace(1e-3, 2e-2, ndet),
                        det_snr=snr, n_match=int((mm >= 0.8).sum()), n_proxy=mm_n,
                        truth="gb_truth_3to21.npz", truth_orbits="mojito_l1")
    else:
        A["rec_skip"] = "no gb_truth_*.npz in its run directory"
    return A


def figure_ns(added, **extra):
    """A namespace for the generator's figure blocks, with ``fig_b64`` caught."""
    figs = {}
    ns = sh._generator_functions({"np": np}, names=("_survival", "_wilson", "_match_pairs"))
    ns.update(COLORS)
    ns.update(plt=plt, os=os, h5py=h5py, ADDED=added, _add=add, ADD_BASE_LABEL=BASE,
              MISSING=[], SHOW_MATCH_STATS=True, _USETEX=False,
              fig_b64=lambda fig, key, dpi=None: figs.__setitem__(key, fig))
    ns.update(extra)
    return ns, figs


def run_block(code, ns):
    exec(compile(code, "generator-block", "exec"), ns)       # noqa: S102


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------
# the knob
# --------------------------------------------------------------------------
class ParseAddInfoTest(unittest.TestCase):

    def test_the_repeatable_flag_keeps_its_order(self):
        self.assertEqual(add.parse_add_info(["/r/b", "/r/a", "/r/c"], environ={}),
                         ["/r/b", "/r/a", "/r/c"])

    def test_the_env_list_keeps_its_order_and_drops_duplicates(self):
        env = {add.ENV: os.pathsep.join(["/r/b", "/r/a/", "/r/b", " /r/c ", ""])}
        self.assertEqual(add.parse_add_info(environ=env), ["/r/b", "/r/a", "/r/c"])

    def test_the_flag_wins_over_the_env(self):
        env = {add.ENV: "/r/env"}
        self.assertEqual(add.parse_add_info(["/r/cli"], environ=env), ["/r/cli"])
        self.assertEqual(add.parse_add_info([], environ=env), ["/r/env"])
        self.assertEqual(add.parse_add_info(None, environ=env), ["/r/env"])

    def test_a_flag_value_may_itself_be_a_list(self):
        self.assertEqual(add.parse_add_info([f"/r/a{os.pathsep}/r/b", "/r/a"], environ={}),
                         ["/r/a", "/r/b"])

    def test_the_base_run_never_overlays_itself(self):
        self.assertEqual(add.parse_add_info(["/r/base/", "/r/x"], environ={},
                                            base_run_dir="/r/base"), ["/r/x"])

    def test_none_silences_and_unset_is_empty(self):
        self.assertEqual(add.parse_add_info(environ={}), [])
        for off in ("none", "NONE", "0", "off"):
            self.assertEqual(add.parse_add_info(environ={add.ENV: off}), [], off)
        self.assertEqual(add.parse_add_info(["none"], environ={add.ENV: "/r/env"}), [])
        # the explicit empty list a build is handed travels as "none"
        self.assertEqual(add.add_info_env_value([]), "none")
        self.assertEqual(add.parse_add_info(environ={add.ENV: add.add_info_env_value([])}), [])
        self.assertEqual(add.parse_add_info(environ={add.ENV: add.add_info_env_value(
            ["/r/a", "/r/b"])}), ["/r/a", "/r/b"])

    def test_two_spellings_of_one_directory_count_once(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "run"))
        os.symlink(os.path.join(d, "run"), os.path.join(d, "link"))
        got = add.parse_add_info([os.path.join(d, "run"), os.path.join(d, "link"),
                                  os.path.join(d, ".", "run")], environ={})
        self.assertEqual(got, [os.path.join(d, "run")])

    def test_the_label_is_the_pages_own_run_label(self):
        self.assertEqual(add.run_label("/x/gf_prod_9mo_v9"), "9mo_v9")
        self.assertEqual(add.run_label("/x/gf_prod_1yr_v9/"), "1yr_v9")
        from lisatools.globalfit.monitor._identity import arm_tag, run_identity

        self.assertEqual(add.run_label("/x/gf_prod_6mo_v9_4gpu"),
                         arm_tag(run_identity("gf_prod_6mo_v9_4gpu")[1]))


class CliTest(unittest.TestCase):

    def setUp(self):
        import lisatools.globalfit.monitor.__main__ as m

        self.m = m
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.rundir = os.path.join(self.d, "gf_prod_9mo_v9")
        os.makedirs(self.rundir)

    def _main(self, argv, env=None):
        with mock.patch.object(self.m, "build_monitor") as bm, \
                mock.patch.dict(os.environ, env or {}), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            if env is None:
                os.environ.pop(add.ENV, None)
            self.assertEqual(self.m.main(argv), 0)
        return bm, out.getvalue()

    def test_the_flag_is_repeatable_and_reaches_the_build_in_order(self):
        bm, out = self._main([self.rundir, "--add-info", "/r/6mo", "--add-info", "/r/3mo",
                              "--add-info", "/r/6mo"])
        self.assertEqual(bm.call_args.kwargs["add_info"], ["/r/6mo", "/r/3mo"])
        self.assertIn("added runs", out)

    def test_without_the_flag_the_build_reads_the_environment(self):
        bm, out = self._main([self.rundir], env={add.ENV: "/r/6mo"})
        self.assertIsNone(bm.call_args.kwargs["add_info"])
        self.assertIn(f"(from {add.ENV})", out)
        bm, _ = self._main([self.rundir])
        self.assertIsNone(bm.call_args.kwargs["add_info"])

    def test_none_on_the_line_drops_an_exported_list(self):
        bm, _ = self._main([self.rundir, "--add-info", "none"], env={add.ENV: "/r/6mo"})
        self.assertEqual(bm.call_args.kwargs["add_info"], [])

    def test_from_tar_passes_it_through(self):
        from lisatools.globalfit.monitor import from_tar as ft

        _make_run(self.d, "gf_prod_tar")
        tar = os.path.join(self.d, "snap.tar.gz")
        open(tar, "w").close()
        with mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft.os.path, "getsize", return_value=1), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = ft.main([tar, "--run-dir", os.path.join(self.d, "gf_prod_tar"),
                          "--add-info", "/r/6mo"])
        self.assertEqual(rc, 0)
        self.assertEqual(bm.call_args.kwargs["add_info"], ["/r/6mo"])


class InRunEnvironmentTest(unittest.TestCase):
    """The monitor rank builds through ``hooks.build_products`` with no list, so
    the page must read the launcher's export from the environment; an explicit
    list replaces it for one build only."""

    FAKE = ("import os, sys\n"
            "with open(sys.argv[2], 'w') as fh:\n"
            "    fh.write(os.environ.get('GF_MONITOR_ADD_INFO', 'UNSET'))\n")

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.gen = os.path.join(self.d, "fake_generator.py")
        with open(self.gen, "w") as fh:
            fh.write(self.FAKE)
        self.run_dir = os.path.join(self.d, "gf_prod_9mo_v9")
        os.makedirs(self.run_dir)
        self.out = os.path.join(self.d, "page.html")

    def _seen(self):
        with open(self.out) as fh:
            return fh.read()

    def test_the_in_run_build_sees_the_export_in_process_and_in_a_child(self):
        from lisatools.globalfit.monitor import hooks

        for sub in ("0", "1"):
            env = {add.ENV: "/r/6mo:/r/3mo", "GF_MONITOR_SNAPSHOT": "0", "GF_MONITOR_PAGE": "1",
                   "GF_MONITOR_OUT": self.out, "GF_MONITOR_SUBPROCESS": sub}
            with self.subTest(subprocess=sub), \
                    mock.patch.object(mon, "generator_path", return_value=self.gen), \
                    mock.patch.dict(os.environ, env):
                os.environ.pop("GF_MONITOR_PAGE_SHORT", None)
                self.assertIsNotNone(hooks.build_products(self.run_dir))
                self.assertEqual(self._seen(), "/r/6mo:/r/3mo")

    def test_an_explicit_list_wins_for_one_build_and_the_env_is_restored(self):
        with mock.patch.object(mon, "generator_path", return_value=self.gen), \
                mock.patch.dict(os.environ, {add.ENV: "/r/env"}):
            mon.build_monitor(self.run_dir, self.out, add_info=["/r/a", "/r/b"])
            self.assertEqual(self._seen(), os.pathsep.join(["/r/a", "/r/b"]))
            self.assertEqual(os.environ[add.ENV], "/r/env")
            mon.build_monitor(self.run_dir, self.out, add_info=[])
            self.assertEqual(self._seen(), "none")
            mon.build_monitor(self.run_dir, self.out, add_info=["/r/c"], in_process=False)
            self.assertEqual(self._seen(), "/r/c")
            self.assertEqual(os.environ[add.ENV], "/r/env")


# --------------------------------------------------------------------------
# reading and matching an added run (synthetic stores, the real GBGPU path)
# --------------------------------------------------------------------------
class _Dir:
    def setUp(self):
        self._cwd = os.getcwd()
        self.d = tempfile.mkdtemp()
        os.chdir(self.d)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.d, ignore_errors=True)


def _truth_equal_to_model(run, name):
    """A truth set whose detectable sources ARE walker 0's last-row leaves."""
    from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

    store = os.path.join(run, f"{name}_testing.h5")
    with h5py.File(store, "r") as f:
        al = f["global_fit/inds/gb"][NIT - 1, 0, 0, 0]
        rec = f["global_fit/chain/gb"][NIT - 1, 0, 0, 0][al]
    tc = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                     use_distance=True, mc_lims=(0.001, 1.0))
    phys = tc.both_transforms(np.asarray(rec, float).copy())
    snr = np.linspace(8.0, 40.0, len(phys))
    np.savez(os.path.join(run, mon.TRUTH_NAME), f0=phys[:, 1], amp=phys[:, 0], snr=snr,
             det=np.ones(len(phys), bool), phys=phys, band=np.array([0.8e-3, 21.94e-3]),
             tobs=1440 * 4320 * 2.5, orbits="mojito_l1")
    return int(al.sum())


class LoadAddedRunTest(_Dir, unittest.TestCase):

    def test_the_last_row_the_max_lnl_walker_and_the_last_noise_rows(self):
        run = _make_run(self.d, "gf_prod_6mo_v9_4gpu", log=False, csv=False)
        missing, logs = [], []
        (A,) = add.load_added_runs([run], base_label=BASE, missing=missing, log=logs.append)
        self.assertIsNone(A["skip"])
        self.assertEqual(A["label"], add.run_label(run))
        self.assertEqual((A["row"], A["noise_row"], A["wb"]), (NIT - 1, NIT - 1, 0))
        self.assertEqual(A["rec9"].shape, (5 + 2 * (NIT - 1), 9))   # the fixture's alive count
        self.assertEqual(A["psd"].shape, (3, NW, 2))
        with h5py.File(os.path.join(run, "gf_prod_6mo_v9_4gpu_testing.h5"), "r") as f:
            want = f["global_fit/sub_backend/psd/chain"][NIT - 3:NIT, 0, :, 0, :]
            gal = f["global_fit/sub_backend/galfor/chain"][NIT - 1, 0, :, 0, :]
            al = f["global_fit/inds/gb"][NIT - 1, 0, 0, 0]
            leaves = f["global_fit/chain/gb"][NIT - 1, 0, 0, 0][al]
        np.testing.assert_array_equal(A["rec9"], leaves)       # walker 0's last row, exactly
        np.testing.assert_array_equal(A["psd"], want)
        # stored log10 (the store's identity says so) -> physical
        self.assertTrue(A["galfor_log"])
        np.testing.assert_allclose(A["gal_phys"][-1][:, [0, 1, 3, 4]], 10 ** gal[:, [0, 1, 3, 4]])
        self.assertAlmostEqual(A["tobs"], 1440 * 4320 * 2.5)
        self.assertEqual(missing, [])
        self.assertTrue(any(l.startswith(f"[add-info] {A['label']}: row {NIT - 1}") for l in logs))

    def test_a_newer_lookup_table_in_the_run_dir_is_not_the_store(self):
        run = _make_run(self.d, "gf_prod_6mo_v9_4gpu", log=False, csv=False)
        with h5py.File(os.path.join(run, "wdm_lookup_emri_cx_NF1440_DT2.5.h5"), "w") as f:
            f.create_dataset("x", data=np.zeros(3))
        (A,) = add.load_added_runs([run], base_label=BASE, missing=[], log=None)
        self.assertIsNone(A["skip"])
        self.assertEqual(A["store_name"], "gf_prod_6mo_v9_4gpu_testing.h5")

    def test_a_torn_last_row_steps_back(self):
        run = _make_run(self.d, "gf_prod_6mo_v9_4gpu", log=False, csv=False)
        with h5py.File(os.path.join(run, "gf_prod_6mo_v9_4gpu_testing.h5"), "a") as f:
            f["global_fit/inds/gb"][NIT - 1] = False
        (A,) = add.load_added_runs([run], base_label=BASE, missing=[], log=None)
        self.assertEqual(A["row"], NIT - 2)
        self.assertEqual(A["rec9"].shape[0], 5 + 2 * (NIT - 2))
        self.assertTrue(any("torn" in n for n in A["notes"]))

    def test_a_missing_dir_and_a_storeless_dir_are_skipped_with_a_missing_line(self):
        empty = os.path.join(self.d, "gf_prod_3mo_v9")
        os.makedirs(empty)
        missing, logs = [], []
        runs = add.load_added_runs([os.path.join(self.d, "nope_6mo_v9"), empty],
                                   base_label=BASE, missing=missing, log=logs.append)
        self.assertEqual(runs[0]["skip"], "no such run directory")
        self.assertIn("no store", runs[1]["skip"])
        self.assertEqual(len(missing), 2)
        self.assertEqual(sum(l.startswith("[add-info] MISSING") for l in logs), 2)
        # the requested position keeps its style: a skipped run repaints nobody
        self.assertEqual([A["style"]["color"] for A in runs],
                         [add.STYLES[0]["color"], add.STYLES[1]["color"]])

    def test_colliding_labels_fall_back_to_the_directory_name(self):
        a = _make_run(os.path.join(self.d), "gf_prod_9mo_v9", log=False, csv=False)
        os.makedirs(os.path.join(self.d, "other"))
        b = _make_run(os.path.join(self.d, "other"), "gf_prod_9mo_v9", log=False, csv=False)
        runs = add.load_added_runs([a, b], base_label=BASE, missing=[], log=None)
        # both read as "9mo_v9", the base run's own label: directory name, then numbered
        self.assertEqual([A["label"] for A in runs], ["gf_prod_9mo_v9", "gf_prod_9mo_v9 (2)"])


class ComputeRecoveryTest(_Dir, unittest.TestCase):
    """The real waveform path: a truth set equal to the model matches at 1."""

    def _load(self, run, **kw):
        runs = add.load_added_runs([run], base_label=BASE, missing=[], log=None)
        missing = []
        with mock.patch.object(sh, "_l1_orbits_from", return_value=(None, "test: none")):
            add.compute_recoveries(runs, thresh=0.8, missing=missing, log=None, **kw)
        return runs[0], missing

    def test_its_own_truth_set_matches_everything_at_overlap_one(self):
        run = _make_run(self.d, "gf_prod_6mo_v9_4gpu", log=False, csv=False)
        n = _truth_equal_to_model(run, "gf_prod_6mo_v9_4gpu")
        A, missing = self._load(run)
        R = A["rec"]
        self.assertIsNone(A["rec_skip"])
        self.assertEqual(missing, [])
        self.assertEqual((R["ndet"], R["n_match"], int(np.size(R["mm"]))), (n, n, n))
        np.testing.assert_allclose(R["mm"], 1.0, atol=1e-9)
        np.testing.assert_array_equal(np.sort(R["ti_ov"]), np.arange(n))
        np.testing.assert_allclose(R["det_snr"], np.linspace(8.0, 40.0, n))
        self.assertEqual((R["truth"], R["walker"], R["row"]), (mon.TRUTH_NAME, 0, NIT - 1))
        self.assertNotIn("rec9", A)                    # freed once matched

    def test_no_truth_set_and_the_cwd_truth_set_is_not_used(self):
        run = _make_run(self.d, "gf_prod_6mo_v9_4gpu", log=False, csv=False)
        _truth_equal_to_model(run, "gf_prod_6mo_v9_4gpu")
        shutil.move(os.path.join(run, mon.TRUTH_NAME), os.path.join(self.d, mon.TRUTH_NAME))
        A, missing = self._load(run)
        self.assertIsNone(A["rec"])
        self.assertIn("no gb_truth_*.npz", A["rec_skip"])
        self.assertEqual(len(missing), 1)
        self.assertIn("has no recovery overlay", missing[0])

    def test_a_truth_set_for_another_tobs_is_refused(self):
        run = _make_run(self.d, "gf_prod_6mo_v9_4gpu", log=False, csv=False)
        _truth_equal_to_model(run, "gf_prod_6mo_v9_4gpu")
        p = os.path.join(run, mon.TRUTH_NAME)
        z = dict(np.load(p))
        z["tobs"] = np.float64(7776000.0)
        np.savez(p, **z)
        A, _ = self._load(run)
        self.assertIsNone(A["rec"])
        self.assertIn("is for Tobs 90 d", A["rec_skip"])


# --------------------------------------------------------------------------
# the generator's wiring
# --------------------------------------------------------------------------
class GeneratorWiringTest(unittest.TestCase):

    def test_an_overlay_call_precedes_exactly_the_listed_figures(self):
        keys = [(m.start(), m.group(1)) for m in re.finditer(r'fig_b64\(fig, "(\w+)"', SRC)]
        calls = [m.start() for m in re.finditer(r"_add\.overlay_\w+\(", SRC)]
        self.assertEqual(len(calls), len(OVERLAY_KEYS))
        hit = set()
        for c in calls:
            nxt = next(k for pos, k in keys if pos > c)
            prev = max((pos for pos, _ in keys if pos < c), default=-1)
            self.assertIn(nxt, OVERLAY_KEYS)
            self.assertNotIn(nxt, hit, f"two overlay calls before {nxt}")
            hit.add(nxt)
            # each call sits in this figure's own block (after the previous figure)
            self.assertLess(prev, c)
        self.assertEqual(hit, OVERLAY_KEYS)

    def test_every_overlay_is_behind_if_added(self):
        for m in re.finditer(r"_add\.overlay_\w+\(", SRC):
            line = SRC[SRC.rindex("\n", 0, m.start()) + 1:SRC.index("\n", m.start())]
            above = SRC[:m.start()].rstrip().split("\n")
            guarded = line.lstrip().startswith("if ADDED") or any(
                re.match(r"\s*if ADDED\b", l) for l in above[-4:])
            self.assertTrue(guarded, line)

    def test_the_two_phases_are_wired(self):
        p1 = SRC.index("ADDED = _add.load_added_runs(_add.parse_add_info(base_run_dir=RUN_DIR)")
        self.assertLess(p1, SRC.index('fig_b64(fig, "ll")'))
        p2 = SRC.index("_add.compute_recoveries(")
        self.assertLess(SRC.index("ARM_COL[_tag] = _ARM_CYCLE"), p2)
        self.assertLess(p2, SRC.index('fig_b64(fig, "f2_progress")'))
        self.assertIn("thresh=MATCH_MM_THRESH", SRC[p2:p2 + 400])

    def test_the_caption_sentence_is_on_exactly_those_panels(self):
        for cap in ("cap_f3", "cap_f5", "cap_f6"):
            self.assertIn(f"{cap} += ADD_CAP_REC", SRC)
        self.assertEqual(len(re.findall(r"\+= ADD_CAP_REC", SRC)), 3)
        tpl = SRC[SRC.index('html = f"""<title>'):]
        for key in ("f11_psd", "f11_fg", "psd_curves"):
            panel = tpl[tpl.index(f'{{img("{key}"'):]
            panel = panel[:panel.index("</div></div>")]
            self.assertIn("{ADD_CAP_NOISE}", panel, key)
        for key in ("psd_hist", "gal_hist"):
            self.assertIn(f'{{img("{key}")}}{{ADD_CAP_NOISE_DIV}}</div>', tpl, key)
        self.assertEqual(tpl.count("{ADD_CAP_NOISE}"), 3)
        self.assertEqual(tpl.count("{ADD_CAP_NOISE_DIV}"), 2)


# --------------------------------------------------------------------------
# the generator's own figure blocks, on synthetic arrays
# --------------------------------------------------------------------------
F3 = block("# ---- F3: match CDF + survival COUNT", 'fig_b64(fig, "f3_match")')
F6 = block("# ---- F6: completeness vs SNR, with Wilson", 'fig_b64(fig, "f6_snr")')
F5 = block("# ---- F5: source counts vs frequency", 'fig_b64(fig, "f5_counts")')
F11_PSD = block("_nsh = min(3, SUB_NIT)", 'fig_b64(fig, "f11_psd")')
PSD_HIST = block("fig, ax = plt.subplots(1, 2, figsize=(11, 2.9))", 'fig_b64(fig, "psd_hist")')
GAL_HIST = block("fig, ax = plt.subplots(1, 5, figsize=(14, 2.5))", 'fig_b64(fig, "gal_hist")')
FG_TRY = block("try:\n    # XYZ EVERYWHERE", 'MISSING.append(f"foreground curve render failed: {e!r}")')
SENS_TRY = block("try:\n    from lisatools.sensitivity import get_sensitivity, LISASens",
                 'MISSING.append(f"LISASens curve render failed: {e!r}")')


def recovery_ns(added, ndet=80, seed=3):
    rng = np.random.default_rng(seed)
    mm = np.clip(rng.uniform(0.4, 1.05, 70), 0, 1)
    ti = np.sort(rng.choice(ndet, 70, replace=False))
    arm = dict(mm=mm, ti=ti[mm >= 0.8], n_match=np.arange(30), n_band=np.arange(30) + 5,
               it0=np.int64(4), rec_f0=np.geomspace(1e-3, 2e-2, 70), ndet=ndet)
    found = np.zeros(ndet, bool)
    found[ti] = True
    return figure_ns(added, ARMS={BASE: arm}, ARM_COL={BASE: "#FF7BAC"}, NDET=ndet,
                     T_SNR=rng.uniform(7, 60, ndet), T_F0=np.geomspace(1e-3, 2e-2, ndet),
                     FOUND=found, FLO=0.8e-3, FHI=21.94e-3, MOJITO_CAT_DIR="/nonexistent",
                     SCI={}, REC9=np.zeros((0, 9)))


def noise_ns(added, nit=6, nw=4, seed=4):
    rng = np.random.default_rng(seed)
    psd = np.stack([np.column_stack([1.5e-11 * (1 + 3e-3 * rng.standard_normal(nw)),
                                     3e-15 * (1 + 6e-3 * rng.standard_normal(nw))])
                    for _ in range(nit)])
    gal_log = np.tile(np.array([-43.8, -2.6, 5.0, -2.0, -2.85]), (nit, nw, 1)) \
        + 0.01 * rng.standard_normal((nit, nw, 5))
    gal_phys = gal_log.copy()
    gal_phys[..., [0, 1, 3, 4]] = 10 ** gal_phys[..., [0, 1, 3, 4]]
    return figure_ns(added, psd_cold=psd, gal_cold=gal_log, gal_cold_phys=gal_phys,
                     SUB_NIT=nit, NIT=nit, nwalk=nw, SOMS_INJ=1.5e-11, SA_INJ=3e-15,
                     GALFOR_LOG=True, SCI_TOBS=180 * 86400.0, PSD_ZOOM_N=5,
                     GAL_NAMES=["log10 amp", "log10 fk", "alpha", "log10 f_1", "log10 f_2"])


ADDED_TWO = [synth_added(0, "6mo"), synth_added(1, "3mo_v9")]


class RenderedOverlayTest(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_the_overlap_figure_gets_one_entry_per_run(self):
        ns, figs = recovery_ns(ADDED_TWO)
        run_block(F3, ns)
        ax0, ax1 = figs["f3_match"].axes
        n_base = int(np.size(ns["ARMS"][BASE]["mm"]))
        self.assertEqual(legend_labels(ax0), [f"{BASE}  ({n_base} matched)", "6mo  (40 matched)",
                                              "3mo_v9  (40 matched)"])
        cols = [to_hex(l.get_color()) for l in ax0.get_lines()]
        styles = [l.get_linestyle() for l in ax0.get_lines()]
        self.assertEqual(cols[0], "#ff7bac")                         # the base keeps its style
        self.assertEqual(styles[0], "-")
        self.assertEqual(cols[1:], [s["color"].lower() for s in (add.style(0), add.style(1))])
        self.assertEqual(len(set(cols)), 3)
        self.assertEqual(len(set(styles)), 3)
        # the survival twin: the same two runs, same colour and style (plus the
        # base curve and its detectable-count reference line)
        surv = [(to_hex(l.get_color()), l.get_linestyle()) for l in ax1.get_lines()]
        for s in (add.style(0), add.style(1)):
            self.assertEqual(surv.count((s["color"].lower(), s["ls"])), 1)
        self.assertEqual(len(surv), 4)

    def test_the_snr_and_frequency_figures_get_one_entry_per_run(self):
        ns, figs = recovery_ns(ADDED_TWO)
        run_block(F6, ns)
        labels = legend_labels(figs["f6_snr"].axes[0])
        self.assertEqual(labels[1:], ["6mo, row 500", "3mo_v9, row 500"])
        self.assertTrue(labels[0].startswith(f"{BASE}, "))
        ns, figs = recovery_ns(ADDED_TWO)
        run_block(F5, ns)
        top, bottom = figs["f5_counts"].axes
        self.assertEqual(legend_labels(bottom), [BASE, "6mo, row 500", "3mo_v9, row 500"])
        self.assertFalse(any("6mo" in t for t in legend_labels(top)))   # the counts stay this run's

    def test_the_noise_histograms_get_one_entry_per_run(self):
        ns, figs = noise_ns(ADDED_TWO)
        run_block(F11_PSD, ns)
        for ax in figs["f11_psd"].axes:
            self.assertEqual(legend_labels(ax), [BASE, "injected", "6mo, last 3 rows to 500",
                                                 "3mo_v9, last 3 rows to 500"])
        run_block(PSD_HIST, ns)
        run_block(GAL_HIST, ns)
        # small panels: ONE figure legend under them, no panel covered
        for key in ("psd_hist", "gal_hist"):
            fig = figs[key]
            self.assertEqual(len(fig.legends), 1, key)
            self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                             [BASE, "6mo, last 3 rows to 500", "3mo_v9, last 3 rows to 500"])
            self.assertTrue(all(a.get_legend() is None for a in fig.axes), key)
        gal_axes = figs["gal_hist"].axes
        # the added foreground lands in THIS run's basis (log10 amp ~ -43.8)
        step = [p for p in gal_axes[0].patches if p.get_label() == "6mo, last 3 rows to 500"]
        self.assertTrue(step)
        xs = step[0].get_xy()[:, 0]
        self.assertTrue(-44.5 < xs.min() < xs.max() < -43.0, (xs.min(), xs.max()))

    def test_the_noise_curves_get_one_entry_per_run_and_the_other_curves_none(self):
        ns, figs = noise_ns(ADDED_TWO)
        run_block(FG_TRY, ns)
        run_block(SENS_TRY, ns)
        self.assertEqual(ns["MISSING"], [])
        want = ["6mo, row 500 (4 walkers)", "3mo_v9, row 500 (4 walkers)"]
        for key in ("f11_fg", "psd_curves"):
            labels = legend_labels(figs[key].axes[0])
            self.assertEqual(labels[-2:], want, key)
            n_added = [l for l in figs[key].axes[0].get_lines() if l.get_linestyle() == "--"
                       and to_hex(l.get_color()) == add.style(0)["color"].lower()]
            self.assertEqual(len(n_added), 4, key)                  # every cold walker
        # the evolution / zoom figures stay this run's
        for key in ("f11_fg_zoom", "psd_evolution", "psd_evolution_zoom"):
            self.assertIn(key, figs)
            labels = [l.get_label() for a in figs[key].axes for l in a.get_lines()]
            self.assertFalse(any("6mo" in l or "3mo_v9" in l for l in labels), key)

    def test_a_skipped_or_truthless_run_draws_nothing_and_says_why(self):
        added = [synth_added(0, "6mo", rec=False), synth_added(1, "1yr_v9",
                                                               skip="no such run directory")]
        ns, figs = recovery_ns(added)
        run_block(F3, ns)
        self.assertEqual(len(legend_labels(figs["f3_match"].axes[0])), 1)
        cap = add.caption_html(added, "recovery", base_label=BASE, base_row=99)
        self.assertIn("<em>Not overlaid:</em> <b>6mo</b> (no gb_truth_*.npz in its run "
                      "directory); <b>1yr_v9</b> (no such run directory).", cap)
        ns, figs = noise_ns(added)
        run_block(F11_PSD, ns)
        self.assertIn("6mo, last 3 rows to 500", legend_labels(figs["f11_psd"].axes[0]))
        self.assertNotIn("1yr_v9", " ".join(legend_labels(figs["f11_psd"].axes[0])))
        cap = add.caption_html(added, "noise", base_label=BASE, base_row=5)
        self.assertIn("<b>6mo</b> (dashed): its last written row 500 of "
                      "<code>gf_prod_6mo_testing.h5</code> for the curves (the histograms pool its "
                      "last 3 rows", cap)
        self.assertIn("<em>Not overlaid:</em> <b>1yr_v9</b> (no such run directory).", cap)

    def test_a_broken_record_costs_its_overlay_not_the_figure(self):
        add.load_added_runs([], base_label=BASE, log=None)  # a fresh build
        bad = synth_added(0, "6mo")
        bad["rec"]["mm"] = "not an array"                  # a drawing error, not a skip
        bad["psd"] = None
        ns, figs = recovery_ns([bad, synth_added(1, "3mo_v9")])
        with contextlib.redirect_stdout(io.StringIO()):
            run_block(F3, ns)                               # does not raise
        # the broken run is left out, the other added run still draws
        labels = legend_labels(figs["f3_match"].axes[0])
        self.assertEqual(labels[1:], ["3mo_v9  (40 matched)"])
        self.assertEqual(len(add.OVERLAY_ERRORS), 1)
        self.assertIn("added run 6mo: overlay_overlap failed", add.OVERLAY_ERRORS[0])
        self.assertIn("MISSING.extend(_add.OVERLAY_ERRORS)", SRC)   # listed on the page
        ns, figs = noise_ns([bad])
        run_block(F11_PSD, ns)                              # psd None: drawn as nothing
        self.assertEqual(legend_labels(figs["f11_psd"].axes[0]), ["injected"])
        add.load_added_runs([], base_label=BASE, log=None)  # a new build resets the list
        self.assertEqual(add.OVERLAY_ERRORS, [])

    def test_the_caption_names_the_rows_and_no_list_means_no_sentence(self):
        cap = add.caption_html(ADDED_TWO, "recovery", base_label=BASE, base_row=1922)
        self.assertIn(f"<b>{BASE}</b> (this run, as drawn, row 1922)", cap)
        self.assertIn("<b>6mo</b> (dashed): its stored row 500 (its last)", cap)
        self.assertIn("<b>3mo_v9</b> (dotted): its stored row 500 (its last)", cap)
        self.assertIn("own <code>gb_truth_3to21.npz</code> (60 detectable", cap)
        self.assertEqual(add.caption_html([], "recovery", base_label=BASE, base_row=1), "")
        self.assertEqual(add.caption_html([], "noise", base_label=BASE, base_row=1), "")

    def test_an_empty_list_renders_each_figure_exactly_as_before(self):
        for name, code, mk in (("f3_match", F3, recovery_ns), ("f6_snr", F6, recovery_ns),
                               ("f5_counts", F5, recovery_ns), ("f11_psd", F11_PSD, noise_ns),
                               ("psd_hist", PSD_HIST, noise_ns), ("gal_hist", GAL_HIST, noise_ns)):
            with self.subTest(figure=name):
                self.assertNotEqual(strip_overlays(code), code)
                ns, figs = mk([])
                run_block(code, ns)
                ns0, figs0 = mk([])
                run_block(strip_overlays(code), ns0)
                self.assertEqual(png(figs[name]), png(figs0[name]))


if __name__ == "__main__":
    unittest.main()
