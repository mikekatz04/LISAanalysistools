"""The SHORT monitor page (``lisatools.globalfit.monitor._short``), which is
also the full page's at-a-glance block.

User request 2026-10-03: "let's make a short version of the html as well.
That picks out the most important plots and topline info. phase maximized
overlap. current psd measurements. nleaves over time, likelihood over time."
User request 2026-10-11: the most important / succinct plots and info first
-- the short page -- and "stack the other sources closer to the top".

Pinned here:
  * it renders from a small synthetic run directory, with the topline fields
    and the section headings in order, as one self-contained, well-formed
    document;
  * the topline reads the head log the way gf_last_iteration.py does (legs and
    their walls, the RJ valve's active pairs, the empty-band valve, the F-stat
    epochs), and the two parsers give one answer on one log;
  * the other-sources table names a walker on another mode and colours a
    parameter pulled off its catalogue truth;
  * it DEGRADES rather than fails when the head log, the GPU CSV, the truth
    set or a catalogue is absent;
  * the overlap section, given a truth set equal to the model, scores every
    pair at overlap 1 -- the real GBGPU path, end to end;
  * the generator's own helpers are compiled out of its syntax tree, not run,
    and the overlap loop it mirrors is executed against the generator's
    inline block so the two cannot drift;
  * the CLI flag, the in-run knob and from_tar select it, and the full page
    keeps its name.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import textwrap
import unittest
from datetime import datetime
from html.parser import HTMLParser
from unittest import mock

import h5py
import numpy as np

import lisatools.globalfit.monitor as mon
from lisatools.globalfit.monitor import _short as sh

NIT, CAP, NW, NLEAF = 12, 16, 4, 40
NOW = datetime(2026, 10, 3, 11, 12)
GALFOR0 = np.array([-43.84, -2.6, 5.0, -2.0, -2.85])      # log10 basis, alpha linear
_HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(_HERE, os.pardir, "scripts", "diagnostics", "gf_last_iteration.py")

_T = "2026-10-03 {} - lisatools.globalfit.{} - INFO - {}\n"
HEAD_LOG = [
    ("10:00:00,100", "recipe", "[LEG stage_b] resume: rows 4..7 of this stage, last saved after "
                               "'in_model_removal' -> next move 'vgb_pe' (leg 1/2), 0 cycle(s) completed"),
    ("10:20:00,100", "communication.fanout", "[FANOUT] op=gb_run_proposal move=vgb_pe head_s=1.0 "
                                             "max_rank_s=1.0 wait_s=0.000"),
    ("10:29:59,000", "communication.fanout", "[FANOUT] op=gb_run_proposal move=in_model head_s=500.0 "
                                             "max_rank_s=540.0 wait_s=0.030"),
    ("10:30:00,100", "moves.globalfitmove", "[LEG stage_b] cycle 0 leg 1/2: ran ['vgb_pe', 'in_model'] "
                                            "-> row saved after 'in_model'"),
    ("10:30:00,200", "hdfbackend", "[SAVE] save_step 0.13 s (handoff to rank 4)"),
    ("10:40:00,000", "moves.gbspecialstretch", "[GB_GATE rj_fstat_search] occupied 40 | shut 10 (25.0%) "
                                               "| active 30 -- streak p50 1 p90 2 of 3; resets/pair mean "
                                               "0.10 max 1 | max-minus-now gain p50 0.00 p90 3.00 (tol "
                                               "20.00) | peak: 40 of 40 occupied cells"),
    ("10:41:00,000", "moves.gbspecialstretch", "rj_fstat_search: F-stat grid fit epoch 3 starting "
                                               "(walker_ref=0 GLOBAL on rank 0 row 0)"),
    ("10:45:00,000", "sampling.fstat_gridfit", "[stageA] comb + peak selection: 12 peaks in 3.5m (split "
                                               "by node range over the compute ranks); 5 of 20 bands "
                                               "skipped (100 of 400 f0 nodes dropped from the sweep)"),
    ("10:46:00,000", "moves.gbspecialstretch", "rj_fstat_search: F-stat grid fit epoch 3 done in 300.0s "
                                               "(12 peaks)"),
    ("10:47:00,000", "moves.gbspecialstretch", "[GB_BAND_SHUTOFF status rj_fstat_search] clock 2 iters, "
                                               "floor 10.000 mHz: 12/20 bands eligible, 5 qualifying now "
                                               "(cold occ 0), streak max 2 median 2, 5 armed (streak >= "
                                               "clock); 5 off (+0 this tick, -0 re-homed); 3/10 iters "
                                               "since revive; persist restored"),
    ("11:10:00,100", "moves.globalfitmove", "[LEG stage_b] cycle 0 leg 2/2: ran ['in_model_fstat', "
                                            "'in_model_removal'] -> row saved after 'in_model_removal'"),
    ("11:10:00,200", "hdfbackend", "[SAVE] save_step 0.13 s (handoff to rank 4)"),
    ("11:40:00,100", "moves.globalfitmove", "[LEG stage_b] cycle 1 leg 1/2: ran ['vgb_pe', 'in_model'] "
                                            "-> row saved after 'in_model'"),
    ("11:40:00,200", "hdfbackend", "[SAVE] save_step 0.13 s (handoff to rank 4)"),
    ("11:50:00,000", "moves.gbspecialstretch", "[GB_GATE rj_fstat_search] occupied 41 | shut 20 (48.8%) "
                                               "| active 21 -- streak p50 2 p90 3 of 3; resets/pair mean "
                                               "0.20 max 1 | max-minus-now gain p50 0.00 p90 1.00 (tol "
                                               "20.00) | peak: 41 of 41 occupied cells"),
    ("12:30:00,100", "moves.globalfitmove", "[LEG stage_b] cycle 1 leg 2/2: ran ['in_model_fstat', "
                                            "'in_model_removal'] -> row saved after 'in_model_removal'"),
    ("12:30:00,200", "hdfbackend", "[SAVE] save_step 0.13 s (handoff to rank 4)"),
    ("12:35:00,000", "moves.gbspecialstretch", "rj_fstat_search: F-stat grid fit epoch 4 starting "
                                               "(walker_ref=0 GLOBAL on rank 0 row 0)"),
    ("12:36:00,000", "sampling.fstat_gridfit", "[sweep:comb.nsky512] 50/100 evals (50.0%) | 10 evals/s "
                                               "| ETA 5s | elapsed 5s"),
]

# the MBHB catalogue rows the fixture's two MBH leaves are injected from
# (loader keys 1 and 0: positional rows of the ID-sorted catalogue)
MBH_CAT = dict(
    ID=[10, 11], PrimaryMassSSBFrame=[1.0e6, 3.0e5], SecondaryMassSSBFrame=[3.0e5, 1.0e5],
    PrimarySpinCompZ=[0.5, 0.2], SecondarySpinCompZ=[0.3, -0.1], LuminosityDistance=[2.0e4, 1.5e4],
    PhaseReferenceSourceFrame=[1.0, 2.0], InclinationAngle=[0.7, 1.1], RightAscension=[2.0, 4.0],
    Declination=[0.3, -0.5], PolarisationAngle=[0.4, 1.4], TimeCoalescenceSSBFrame=[1.0e7, 2.0e7])


def _dashboard():
    """``scripts/diagnostics/gf_last_iteration.py`` as a module (stdlib-only at import)."""
    spec = importlib.util.spec_from_file_location("gf_last_iteration", DASHBOARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mbh_truths():
    """The two leaves' sampling-basis truths, through the generator's own map:
    leaf i is loader key ``[1, 0][i]``, the positional row of the ID-sorted
    catalogue (IDs 10, 11 -> keys 0, 1)."""
    ns = sh._generator_functions({"np": np}, names=("_src_cat_to_sampling_mbh",))
    rows = [{k: np.asarray(v)[i] for k, v in MBH_CAT.items()} for i in range(2)]
    to = ns["_src_cat_to_sampling_mbh"]
    return to(rows[1]), to(rows[0])


def _make_run(root, name="gf_prod_test", *, log=True, csv=True, sources=False):
    """A run directory the short page can read: the store plus, optionally,
    the head log, the head node's GPU CSV, and an MBH branch with its
    run_settings.log source ids."""
    run = os.path.join(root, name)
    os.makedirs(run)
    rng = np.random.default_rng(7)
    ll = np.zeros((CAP, 1, 1, NW))
    for r in range(NIT):
        ll[r, 0, 0] = 1.0e8 + 1000.0 * r - (500.0 if r == 6 else 0.0) + rng.uniform(0, 5, NW)
    ll[NIT - 1, 0, 0, 0] += 50.0                           # walker 0 is the best at the end
    alive = np.zeros((CAP, 1, 1, NW, NLEAF), bool)
    for r in range(NIT):
        alive[r, 0, 0, :, :5 + 2 * r] = True
    chain = np.zeros((CAP, 1, 1, NW, NLEAF, 9))
    f0 = np.geomspace(1.2, 15.0, NLEAF) * (1.0 + 0.01 * np.arange(NLEAF) / NLEAF)
    base = np.column_stack([
        rng.uniform(0.5, 5.0, NLEAF), f0, rng.uniform(0.3, 0.8, NLEAF),
        rng.uniform(0, 2 * np.pi, NLEAF), rng.uniform(-0.9, 0.9, NLEAF),
        rng.uniform(0, np.pi, NLEAF), rng.uniform(0, 2 * np.pi, NLEAF),
        rng.uniform(-0.9, 0.9, NLEAF), rng.uniform(-2, 2, NLEAF)])
    chain[:NIT, 0, 0] = base[None, None]
    chain[~alive] = 0.0
    gal = np.zeros((CAP, 2, NW, 1, 5))
    psd = np.zeros((CAP, 2, NW, 1, 2))
    g_now = np.tile(GALFOR0, (NW, 1))
    p_now = np.column_stack([1.5e-11 * (1 + 1e-4 * rng.standard_normal(NW)),
                             3.0e-15 * (1 + 1e-3 * rng.standard_normal(NW))])
    for r in range(NIT):
        if r == 6:
            g_now = g_now + np.array([-0.05, -0.1, 0.0, 0.0, -0.15])
        elif r == 9:
            g_now = g_now + 0.02 * rng.standard_normal((NW, 5))
            p_now = p_now * (1 + 1e-4 * rng.standard_normal((NW, 2)))
        gal[r, :, :, 0] = g_now
        psd[r, :, :, 0] = p_now
    saved = [("in_model", "in_model_fstat", "in_model_removal")[r % 3] for r in range(NIT)]
    with h5py.File(os.path.join(run, f"{name}_testing.h5"), "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = NIT
        g.create_dataset("log_like", data=ll)
        g.create_dataset("inds/gb", data=alive)
        g.create_dataset("chain/gb", data=chain)
        for b, n in (("vgb", 3), ("psd", 1), ("galfor", 1)):
            g.create_dataset(f"inds/{b}", data=np.ones((CAP, 1, 1, NW, n), bool))
        vhh = np.full((CAP, NW, 3), np.nan)
        vhh[:NIT] = np.array([100.0, 81.0, 4.0])           # SNR 10, 9, 2: two above 7
        g.create_dataset("sub_backend/vgb/h_h", data=vhh)
        g.create_dataset("saved_after", data=np.array(saved, dtype="S32"))
        g.create_dataset("sub_backend/psd/chain", data=psd)
        g.create_dataset("sub_backend/galfor/chain", data=gal)
        g.create_group("noise_model_identity").attrs["galfor_log_sampling"] = True
        a = g.create_group("domain_settings").create_group("args").attrs
        a["0"], a["1"], a["2"] = 1440, 4320, 2.5
        rec = g.create_group("recipe")
        s1 = rec.create_group("stage_a").attrs
        s1["order num"], s1["status"], s1["completed_iteration"] = 1, True, 4
        s2 = rec.create_group("stage_b").attrs
        s2["order num"], s2["status"], s2["start_iteration"] = 2, False, 4
        s2["move_order"] = json.dumps(["vgb_pe", "in_model", "in_model_fstat", "in_model_removal"])
        s3 = rec.create_group("stage_c").attrs             # stale stamp from a reset
        s3["order num"], s3["status"], s3["start_iteration"] = 3, False, 10
        if sources:
            truths = _mbh_truths()
            mch = np.zeros((CAP, 1, 1, NW, 2, 11))
            sig = np.abs(np.asarray(truths)) * 1e-3 + 1e-3
            for leaf in range(2):
                mch[:NIT, 0, 0, :, leaf] = (truths[leaf][None, None]
                                            + sig[leaf] * rng.standard_normal((NIT, NW, 11)))
            mch[:NIT, 0, 0, :, 0, 0] += 6.0 * sig[0, 0]      # leaf 0's logM pulled ~ +6 sigma
            g.create_dataset("chain/mbh", data=mch)
            g.create_dataset("inds/mbh", data=np.ones((CAP, 1, 1, NW, 2), bool))
            hh = np.full((CAP, NW, 2), 400.0)               # SNR 20 for both leaves ...
            dh = np.full((CAP, NW, 2), 400.0)
            dh[:, 0, 1] = 100.0                             # ... but walker 0 holds leaf 1 at SNR 5
            g.create_dataset("sub_backend/mbh/h_h", data=hh)
            g.create_dataset("sub_backend/mbh/d_h", data=dh)
    art = os.path.join(run, f"{name}_artifacts")
    os.makedirs(art)
    if sources:
        with open(os.path.join(art, "run_settings.log"), "w") as fh:
            fh.write("processor_init_kwargs: {'source_ids': {'MBHB': [1, 0]}}\n")
    if log:
        with open(os.path.join(art, "globalfit_run.log"), "w") as fh:
            for hm, mod, msg in HEAD_LOG:
                fh.write(_T.format(hm, mod, msg))
    if csv:
        with open(os.path.join(run, "gpu_util_123.csv"), "w") as fh:
            for s in (20, 40, 60):
                for i, u in ((0, 55), (1, 80)):
                    fh.write(f"2026/10/03 11:09:{s % 60:02d}.000, {i}, NVIDIA H100 NVL, {u}, 0, "
                             f"30000, 95830, 120.5, 45\n")
    return run


def _make_mojito(root):
    """A mojito tree holding the fixture's MBHB catalogue (and an empty data/)."""
    moj = os.path.join(root, "mojito")
    os.makedirs(os.path.join(moj, "catalogues"))
    os.makedirs(os.path.join(moj, "data"))
    with h5py.File(os.path.join(moj, "catalogues",
                                "mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5"), "w") as f:
        b = f.create_group("Binaries")
        for k, v in MBH_CAT.items():
            b.create_dataset(k, data=np.asarray(v))
    return moj


class _WellFormed(HTMLParser):
    VOID = {"meta", "img", "br", "hr", "input", "link", "area", "base", "col",
            "embed", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append((tag, self.stack[-3:]))


class _Cwd:
    """Run from an empty directory: the truth lookup also searches the CWD."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.d = tempfile.mkdtemp()
        os.chdir(self.d)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.d, ignore_errors=True)


class ShortPageRenderTest(unittest.TestCase):
    """One render of the full fixture, many readings of it."""

    @classmethod
    def setUpClass(cls):
        # erebor's import chain restyles matplotlib (eryn); take that hit
        # BEFORE snapshotting, so the rcParams test measures this module only
        import matplotlib

        import lisatools.globalfit.stock.erebor.noise  # noqa: F401

        cls._cwd = os.getcwd()
        cls.d = tempfile.mkdtemp()
        os.chdir(cls.d)
        cls.rundir = _make_run(cls.d)
        cls.rc_before = dict(matplotlib.rcParams)
        cls.path = mon.build_short_monitor(cls.rundir, now=NOW)
        cls.rc_after = dict(matplotlib.rcParams)
        with open(cls.path, encoding="utf-8") as fh:
            cls.html = fh.read()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_it_lands_beside_the_run_under_its_own_name(self):
        self.assertEqual(self.path, os.path.abspath(self.rundir) + "_monitor_short.html")
        self.assertNotEqual(self.path, mon.default_out_path(self.rundir))
        self.assertFalse(os.path.exists(mon.default_out_path(self.rundir)),
                         "the short page must never write the full page's file")

    def test_the_sections_are_there_in_order(self):
        heads = ["<h2>Topline</h2>", "<h2>Log-likelihood</h2>", "<h2>GB Leaf Count</h2>",
                 "<h2>Other Sources</h2>", "<h2>Phase-Maximised Overlap</h2>",
                 "<h2>Noise: Current Measurement</h2>", "<h2>Notes</h2>"]
        pos = [self.html.index(h) for h in heads]
        self.assertEqual(pos, sorted(pos))

    def test_the_topline_carries_every_requested_field(self):
        top = self.html[self.html.index('<section id="topline">'):
                        self.html.index('<section id="lnl">')]
        for want in ("last saved row", f"<b>{NIT - 1}</b>", "saved after",
                     "<code>in_model_removal</code>",         # row 11: 11 % 3 = 2
                     "next leg <code>vgb_pe</code>",           # the cycle wraps
                     "<b>stage_b</b>", "step 2 of 3", "started row 4, 8 rows in",
                     "log written", "last save",
                     "last full cycle 80.0 min (cycle 1: 30.0 min + 50.0 min)",
                     "leg in flight", "leg 1/2 (vgb_pe .. in_model): 6.0 min in, 30.0 min last time",
                     "rj valve", "<b>21</b> active of 41 occupied", "48.8% shut",
                     "streak p50 2 p90 3 of 3", "tol 20 nats",
                     "last 2 judges:</span> 30 &gt; 21",
                     "empty bands", "5 of 12 eligible bands off", "next revive check in 7",
                     "f-stat", "epoch 4 <b>FITTING</b> for 60 s", "comb.nsky512 50% ETA 5s",
                     "<b>12</b> peaks", "fit 5.0 min", "stage A 3.5 min, 5/20 bands skipped",
                     "gpu (head node)", "GPU0 util 55%", "GPU1 util 80%",
                     "head node only", "29.3/93.6 GiB", "job peak 29.3",
                     f"cold lnL, row {NIT - 1}", "change vs row", "spread",
                     "cold leaves", f"gb min <b>{5 + 2 * (NIT - 1)}</b>",
                     "fixed-size branches", "vgb 3",
                     "verification binaries", "<b>2</b> of 3 above SNR 7",
                     "instrument noise"):
            self.assertIn(want, top, want)
        for gone in ("galfor ratchet", "GALFOR_RATCHET", "verdict"):
            self.assertNotIn(gone, top, gone)

    def test_the_lnl_and_leaf_panels_are_embedded_images(self):
        for alt in ("cold log-likelihood", "GB leaf count", "current sensitivity per walker",
                    "instrument-noise posteriors"):
            self.assertRegex(self.html, rf'<img src="data:image/png;base64,[A-Za-z0-9+/=]{{200,}}" '
                                        rf'alt="{alt}">', alt)

    def test_no_truth_set_degrades_the_overlap_section(self):
        sec = self.html[self.html.index('<section id="overlap">'):
                        self.html.index('<section id="noise-now">')]
        self.assertIn('class="missing"', sec)
        self.assertIn("No usable truth set", sec)
        self.assertIn("--build-truth", sec)

    def test_no_source_branch_degrades_the_sources_section(self):
        sec = self.html[self.html.index('<section id="sources">'):
                        self.html.index('<section id="overlap">')]
        self.assertIn('class="missing"', sec)
        self.assertIn("No MBHB / EMRI / SOBHB branch", sec)

    def test_it_is_one_well_formed_self_contained_document(self):
        self.assertTrue(self.html.startswith("<!doctype html>"))
        p = _WellFormed()
        p.feed(self.html)
        p.close()
        self.assertEqual(p.errors, [])
        self.assertEqual(p.stack, [])
        for bad in ('src="http', "href=\"http", "<script src", "<link "):
            self.assertNotIn(bad, self.html)

    def test_the_callers_matplotlib_style_is_left_alone(self):
        # it runs on the saver rank: rcParams are set inside rc_context only
        changed = {k for k in self.rc_before if k != "backend"   # resolved lazily
                   and str(self.rc_before[k]) != str(self.rc_after.get(k))}
        self.assertEqual(changed, set())

    def test_it_is_small(self):
        self.assertLess(os.path.getsize(self.path), 3 * 1024 * 1024)


class ShortPageSourcesTest(_Cwd, unittest.TestCase):
    """The other-sources table against a real (tiny) MBHB catalogue."""

    def test_a_walker_off_the_mode_is_named_and_a_pull_is_coloured(self):
        run = _make_run(self.d, log=False, csv=False, sources=True)
        moj = _make_mojito(self.d)
        html = sh.render_short_page(run, mojito=moj, now=NOW)
        sec = html[html.index('<section id="sources">'):html.index('<section id="overlap">')]
        self.assertIn("<th class='l'>MBHB</th>", sec)
        self.assertIn("<td class='l'>src 1</td><td><b>20.0</b></td>", sec)    # leaf 0 = key 1
        self.assertIn("<td class='l'>src 0</td><td><b>20.0</b></td>", sec)
        self.assertIn("w0 at SNR 5.0 (median 20.0)", sec)
        rows = sec.split("<tr>")
        leaf0 = next(r for r in rows if "src 1" in r)
        leaf1 = next(r for r in rows if "src 0" in r)
        # leaf 0's first key parameter (logM) is ~ +6 sigma off its truth: red
        self.assertRegex(leaf0, r"<td class='dim'>[^<]*</td><td class='p3'>\+[5-7]\.\d</td>")
        self.assertNotIn("p3", leaf1)
        self.assertNotIn("w0 at SNR", leaf0)

    def test_without_a_catalogue_the_table_keeps_the_snrs(self):
        run = _make_run(self.d, log=False, csv=False, sources=True)
        with mock.patch("lisatools.globalfit.monitor.resolve_mojito_path",
                        return_value=(None, "test: none")):
            html = sh.render_short_page(run, now=NOW)
        sec = html[html.index('<section id="sources">'):html.index('<section id="overlap">')]
        self.assertIn("<b>20.0</b>", sec)
        self.assertIn("no catalogue truth", sec)
        self.assertIn("SNRs only, no pulls", html)


class HeadLogParserTest(unittest.TestCase):
    """The topline's head-log reader IS gf_last_iteration.py's, mirrored."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.path = os.path.join(cls.d, "globalfit_run.log")
        with open(cls.path, "w") as fh:
            for hm, mod, msg in HEAD_LOG:
                fh.write(_T.format(hm, mod, msg))
        cls.lines, _ = sh._scan(cls.path, 0, os.path.getsize(cls.path))
        cls.out = sh.parse_head_log(cls.lines)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_the_two_parsers_give_one_answer(self):
        dash = _dashboard()
        lines_d, _ = dash._scan(self.path, 0, os.path.getsize(self.path))
        self.assertEqual(lines_d, self.lines)
        self.assertEqual(dash._parse_log(lines_d), self.out)
        self.assertEqual(dash.last_cycle(self.out["legs"], "stage_b"),
                         sh.last_cycle(self.out["legs"], "stage_b"))
        for name in ("_TAGS", "TAIL_BYTES", "TAIL_BYTES_MAX", "TREND"):
            self.assertEqual(getattr(dash, name), getattr(sh, name), name)

    def test_a_resume_breaks_the_leg_chain(self):
        legs = self.out["legs"]
        self.assertIsNone(legs[0]["wall"], "the leg after a resume line has no wall")
        self.assertEqual([r["wall"] for r in legs[1:]], [2400.0, 1800.0, 3000.0])
        self.assertEqual(sh.last_cycle(legs, "stage_b"), (1, 4800.0, [1800.0, 3000.0]))

    def test_move_walls_come_from_the_fanout_lines(self):
        self.assertEqual(self.out["legs"][0]["moves"], {})            # untimed leg
        dash = _dashboard()
        fan = [(datetime(2026, 1, 1, 0, 1), "rj_prior_removal"), (datetime(2026, 1, 1, 0, 3), "sobbh_pe"),
               (datetime(2026, 1, 1, 0, 4), "psd_pe"), (datetime(2026, 1, 1, 0, 5), "galfor_pe")]
        ran = ["rj_prior_removal", "gb_ridge_gibbs", "sobbh_pe", "noise_cycle_end_search"]
        want = {"rj_prior_removal": 60.0, "sobbh_pe": 120.0, "noise_cycle_end_search": 120.0}
        for mod in (sh, dash):
            self.assertEqual(mod._move_walls(fan, ran, datetime(2026, 1, 1)), want)

    def test_a_stage_end_restarts_the_stage_lists(self):
        extra = [_T.format("13:00:00,000", "recipe", "[V9-STAGE stage_b] STAGE COMPLETE on the "
                                                     "SHUTOFF rule alone: every occupied pair shut").rstrip("\n")]
        out = sh.parse_head_log(self.lines + extra)
        self.assertEqual(out["stage_done"]["stage"], "stage_b")
        self.assertEqual((out["gates"], out["epochs"], out["empty"]), ([], [], None))


class ShortPageDegradesTest(_Cwd, unittest.TestCase):

    def test_no_log_and_no_csv_still_renders(self):
        run = _make_run(self.d, log=False, csv=False)
        html = sh.render_short_page(run, now=NOW)
        self.assertIn("n/a (no head log or job stdout in the run dir)", html)
        self.assertIn("n/a (no gpu_util_*.csv in the run dir)", html)
        notes = html[html.index('<section id="notes">'):]
        self.assertIn("no head log", notes)
        self.assertIn("no gpu_util_*.csv", notes)
        self.assertIn('alt="cold log-likelihood"', html)

    def test_a_failed_build_leaves_the_previous_page_alone(self):
        out = os.path.join(self.d, "page.html")
        with open(out, "w") as fh:
            fh.write("previous")
        empty = os.path.join(self.d, "empty_run")
        os.makedirs(empty)
        self.assertIsNone(sh.build_short_monitor(empty, out, check=False))
        with open(out) as fh:
            self.assertEqual(fh.read(), "previous")
        self.assertFalse(os.path.exists(out + ".tmp"))

    def test_match_stats_knob_suppresses_the_overlap_section(self):
        run = _make_run(self.d, log=False, csv=False)
        with mock.patch.dict(os.environ, {"GF_MONITOR_MATCH_STATS": "0"}):
            html = sh.render_short_page(run, now=NOW)
        self.assertIn("SUPPRESSED (GF_MONITOR_MATCH_STATS=0)", html)


class ShortPageOverlapTest(_Cwd, unittest.TestCase):
    """The real waveform path: a truth set that IS the model scores 1."""

    def test_a_truth_set_equal_to_the_model_matches_everything_at_overlap_one(self):
        from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

        run = _make_run(self.d, log=False, csv=False)
        store = os.path.join(run, "gf_prod_test_testing.h5")
        with h5py.File(store, "r") as f:
            al = f["global_fit/inds/gb"][NIT - 1, 0, 0, 0]
            rec = f["global_fit/chain/gb"][NIT - 1, 0, 0, 0][al]
        tc = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                         use_distance=True, mc_lims=(0.001, 1.0))
        phys = tc.both_transforms(np.asarray(rec, float).copy())
        np.savez(os.path.join(run, mon.TRUTH_NAME), f0=phys[:, 1], amp=phys[:, 0],
                 snr=np.full(len(phys), 20.0), det=np.ones(len(phys), bool), phys=phys,
                 band=np.array([0.8e-3, 21.94e-3]), tobs=1440 * 4320 * 2.5,
                 orbits="mojito_l1")
        with mock.patch.object(sh, "_l1_orbits_from", return_value=(None, "test: none")):
            html = sh.render_short_page(run, now=NOW)
        sec = html[html.index('<section id="overlap">'):html.index('<section id="noise-now">')]
        n = int(al.sum())
        self.assertIn(f"<b>{n:,}</b><span>matched, overlap &ge; 0.80</span>", sec)
        self.assertIn("<b>100.0%</b><span>completeness", sec)
        self.assertIn("<b>1.000</b><span>median overlap</span>", sec)
        self.assertRegex(sec, r'<img src="data:image/png;base64,[^"]{200,}" '
                              r'alt="phase-maximised overlap">')
        self.assertIn("analytic DefaultOrbits", sec)


class PhaseMaxOverlapTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.nw = 32
        self.a = rng.standard_normal(self.nw) + 1j * rng.standard_normal(self.nw)
        self.e = rng.standard_normal(self.nw) + 1j * rng.standard_normal(self.nw)
        self.SA = rng.uniform(1.0, 2.0, 400)
        self.SE = rng.uniform(1.0, 2.0, 400)

    def _mm(self, a2, e2, s1, s2):
        return sh.phase_max_overlaps(self.a[None], self.e[None], np.array([s1]),
                                     a2[None], e2[None], np.array([s2]),
                                     self.SA, self.SE, self.nw)[0]

    def test_identical_waveforms_overlap_one(self):
        self.assertAlmostEqual(self._mm(self.a, self.e, 10, 10), 1.0, places=12)

    def test_an_overall_phase_and_amplitude_are_maximised_away(self):
        z = 3.0 * np.exp(1j * 0.7)
        self.assertAlmostEqual(self._mm(z * self.a, z * self.e, 10, 10), 1.0, places=12)

    def test_disjoint_windows_overlap_zero(self):
        self.assertAlmostEqual(self._mm(self.a, self.e, 10, 10 + self.nw), 0.0, places=12)

    def test_a_window_off_the_grid_scores_zero(self):
        self.assertEqual(self._mm(self.a, self.e, 390, 390), 0.0)

    def test_it_is_the_generators_own_loop(self):
        """Execute the generator's inline block on the same inputs."""
        with open(mon.generator_path(), encoding="utf-8") as fh:
            src = fh.read()
        head, tail = "        MM = np.zeros(MI.size)\n", "        MM = np.clip(MM, 0.0, 1.0)\n"
        block = textwrap.dedent(src[src.index(head):src.index(tail) + len(tail)])
        rng = np.random.default_rng(5)
        n, nw = 25, self.nw
        Ar, Er, At, Et = (rng.standard_normal((n, nw)) + 1j * rng.standard_normal((n, nw))
                          for _ in range(4))
        sr = rng.integers(0, 300, n)
        st = np.clip(sr + rng.integers(-40, 40, n), 0, None)
        st[0], sr[1] = 395, -3                                 # both skip paths
        ns = dict(np=np, MI=np.arange(n), _Ar=Ar, _Er=Er, _sr=sr, _At=At, _Et=Et,
                  _st=st, NW_=nw, SA_G=self.SA, SE_G=self.SE)
        exec(block, ns)                                        # noqa: S102
        got = sh.phase_max_overlaps(Ar, Er, sr, At, Et, st, self.SA, self.SE, nw)
        np.testing.assert_array_equal(got, ns["MM"])
        self.assertTrue(np.any(got > 0))


class GeneratorAdapterTest(unittest.TestCase):

    def test_the_named_helpers_are_compiled_and_nothing_else_runs(self):
        ns = sh._generator_functions({"np": np})
        for name in sh.GEN_HELPERS:
            self.assertTrue(callable(ns.get(name)), name)
        self.assertEqual(set(ns["_SRC_BRANCH_SPECS"]), {"mbh", "emri", "sobbh"})
        for script_global in ("RUN_DIR", "OUT", "IMGS", "f", "g", "html", "MISSING"):
            self.assertNotIn(script_global, ns)
        mi, ti, d = ns["_match_pairs"](np.array([1.0, 2.0]), np.array([2.05, 0.98]), 0.1)
        self.assertEqual(list(mi), [0, 1])
        self.assertEqual(list(ti), [1, 0])

    def test_a_missing_helper_is_an_error_not_a_silent_gap(self):
        with self.assertRaisesRegex(RuntimeError, "no longer defines"):
            sh._generator_functions({"np": np}, names=("no_such_helper",))

    def test_a_constant_that_runs_code_is_refused(self):
        # MATCH_MM_THRESH = float(os.environ.get(...)): a call, not plain data
        with self.assertRaisesRegex(RuntimeError, "no longer plain data"):
            sh._generator_functions({"np": np}, names=(), consts=("MATCH_MM_THRESH",))

    def test_stage_bounds_skip_stale_stamps_of_steps_not_reached(self):
        steps = [dict(name="a", order=1, status=True, start=None, completed=4),
                 dict(name="b", order=2, status=False, start=4, completed=9),
                 dict(name="c", order=3, status=False, start=10, completed=None)]
        self.assertEqual(sh.stage_bounds(steps), [(4, "a done -> b")])


class ShortPageCliTest(unittest.TestCase):

    def setUp(self):
        import lisatools.globalfit.monitor.__main__ as m

        self.m = m
        self.d = tempfile.mkdtemp()
        self.rundir = os.path.join(self.d, "gf_prod_run")
        os.makedirs(self.rundir)
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_the_flag_builds_the_short_page_INSTEAD_of_the_full_one(self):
        with mock.patch.object(self.m, "build_monitor") as bm, \
                mock.patch.object(self.m, "build_short_monitor") as bs:
            self.assertEqual(self.m.main(["--short-page", self.rundir]), 0)
        bm.assert_not_called()
        bs.assert_called_once()
        self.assertEqual(bs.call_args[0][1], os.path.abspath(self.rundir) + "_monitor_short.html")

    def test_without_the_flag_the_full_page_is_unchanged(self):
        with mock.patch.object(self.m, "build_monitor") as bm, \
                mock.patch.object(self.m, "build_short_monitor") as bs:
            self.m.main([self.rundir])
        bs.assert_not_called()
        self.assertEqual(bm.call_args[0][1], os.path.abspath(self.rundir) + "_monitor.html")

    def test_an_explicit_out_path_is_honoured(self):
        out = os.path.join(self.d, "s.html")
        with mock.patch.object(self.m, "build_short_monitor") as bs:
            self.m.main(["--short-page", self.rundir, out])
        self.assertEqual(bs.call_args[0][1], out)

    def test_it_composes_with_snapshot_but_not_snapshot_only(self):
        with mock.patch.object(self.m, "build_short_monitor") as bs, \
                mock.patch.object(self.m, "build_snapshot", return_value="/t.tar.gz") as bt:
            self.assertEqual(self.m.main(["--short-page", "--snapshot", self.rundir]), 0)
        bs.assert_called_once()
        bt.assert_called_once()
        with mock.patch.object(self.m, "build_short_monitor"), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as cm:
            self.m.main(["--short-page", "--snapshot-only", self.rundir])
        self.assertEqual(cm.exception.code, 2)

    def test_from_tar_passes_it_through(self):
        from lisatools.globalfit.monitor import from_tar as ft

        _make_run(self.d, "gf_prod_tar")
        tar = os.path.join(self.d, "snap.tar.gz")
        open(tar, "w").close()
        with mock.patch.object(mon, "build_short_monitor") as bs, \
                mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft.os.path, "getsize", return_value=1), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = ft.main([tar, "--short-page", "--run-dir", os.path.join(self.d, "gf_prod_tar")])
        self.assertEqual(rc, 0)
        bm.assert_not_called()
        self.assertTrue(bs.call_args[0][1].endswith("gf_prod_tar_monitor_short.html"))


class ShortPageHookTest(unittest.TestCase):
    """GF_MONITOR_PAGE_SHORT=1 swaps the page the saver rank builds."""

    class _Comm:
        def __init__(self, payloads):
            self._p = list(payloads)

        def recv(self, source=None):
            return self._p.pop(0)

        def iprobe(self, source=None):
            return False

    class _Reader:
        filename = "/run/dir/gf_prod.h5"

        def save_step_main(self, *a, **k):
            pass

    def _loop(self, **env):
        from lisatools.globalfit import hdfbackend as hb

        base = {"GF_MONITOR_AFTER_SAVE": "1", "GF_MONITOR_SNAPSHOT": "0"}
        with mock.patch.dict(os.environ, {**base, **env}), \
                mock.patch.object(hb, "_atomic_backup_copy"), \
                mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(mon, "build_short_monitor") as bs:
            for k in ("GF_MONITOR_PAGE", "GF_MONITOR_PAGE_SHORT", "GF_MONITOR_ITER"):
                if k not in env:
                    os.environ.pop(k, None)
            hb.save_to_backend_asynchronously_and_plot(
                self._Reader(), self._Comm([{"save_args": (), "save_kwargs": {}},
                                            {"finish_run": True}]),
                main_rank=0, plot_container=None)
        return bm, bs

    def test_the_knob_builds_the_short_page_instead(self):
        bm, bs = self._loop(GF_MONITOR_PAGE_SHORT="1")
        bm.assert_not_called()
        bs.assert_called_once()
        self.assertEqual(bs.call_args[0][0], "/run/dir")

    def test_the_full_page_stays_the_default(self):
        bm, bs = self._loop()
        bm.assert_called_once()
        bs.assert_not_called()

    def test_page_off_turns_the_short_page_off_too(self):
        bm, bs = self._loop(GF_MONITOR_PAGE_SHORT="1", GF_MONITOR_PAGE="0")
        bm.assert_not_called()
        bs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
