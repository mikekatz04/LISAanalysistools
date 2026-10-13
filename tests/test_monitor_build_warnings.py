"""Every warning the 9mo monitor build printed, fixed at its source.

2026-10-13, ``python -m lisatools.globalfit.monitor`` on the live 9mo store
printed, besides the page's own lines:

* ``_generator.py:3862: FutureWarning`` -- SciPy 1.17's ``anderson`` wants a
  ``method``;
* ``:4150`` / ``:4188: UserWarning: This figure includes Axes that are not
  compatible with tight_layout`` -- a gridspec carrying its own ``hspace`` is
  one tight_layout skips, so the call was a no-op;
* ``:5754`` / ``:5864: RuntimeWarning: Mean of empty slice`` -- eight VGBs
  below 0.42 mHz are never scored, so their ``<h|h>`` is NaN in every walker;
* ``:6737: UserWarning: Data has no positive values`` -- the ``[GB_TIMING]``
  regex skipped the fan-out lines whose tail after ``|`` is empty, kept the one
  with a counter-free tail, and drew a throughput of zero on a log axis;
* ``WARNING: skipped 4 unusable L1 brick(s)`` -- four truncated bricks in the
  cluster's mojito cache, reported once per caller and never on the page;
* 51 lines of ``snr: 20000/1008443`` ... -- one per batch.

And the page's ``missing=18`` came with no reasons in the log. The generator is
a script, so its helpers are compiled by name and its blocks exec'd on small
arrays; each test turns the warning it guards into an error.
"""
import ast
import base64
import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest
import warnings
from datetime import datetime
from unittest import mock

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lisatools.globalfit.monitor import build_truth as bt  # noqa: E402
from lisatools.globalfit.monitor import generator_path  # noqa: E402
from lisatools.globalfit.monitor._png import shrink_png  # noqa: E402
from lisatools.globalfit.monitor._short import _generator_functions  # noqa: E402


def _src():
    with open(generator_path(), encoding="utf-8") as fh:
        return fh.read()


def _between(src, start, end):
    return src[src.index(start):src.index(end)]


class VgbUnscoredTest(unittest.TestCase):
    """VGBs with no <h|h> in any walker: a caption note, not a RuntimeWarning."""

    def _hh(self):
        hh = np.full((5, 4, 6), 100.0)               # SNR 10 everywhere ...
        hh[:, :, 4:] = np.nan                       # ... but leaves 4, 5 never scored
        hh[:, 1, 2] = np.nan                        # one walker missing leaf 2: fine
        return hh

    def _snr_block(self, hh):
        ns = {"np": np, "vgb_hh": hh, "MISSING": [], "VGB_IDS": [f"V{k}" for k in range(6)],
              "VGB_F0": np.array([1.0, 2.0, 3.0, 4.0, 0.34, 0.31])}
        _generator_functions(ns, names=("finite_mean", "vgb_unscored_note"))
        blk = _between(_src(), "_nv = int(vgb_hh.shape[-1])", "order = np.argsort(snr)")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            exec(blk, ns)                            # noqa: S102
        return ns

    def test_the_snr_block_warns_nothing_and_says_why(self):
        ns = self._snr_block(self._hh())
        np.testing.assert_allclose(ns["snr"], [10, 10, 10, 10, 0, 0])
        self.assertEqual(ns["MISSING"], [])
        self.assertEqual(ns["VGB_UNSCORED"].tolist(), [4, 5])
        note = ns["VGB_HH_NOTE"]
        self.assertIn("2 of 6 carry no &lang;h|h&rang; in any cold walker at row 4 "
                      "or on any earlier stored row", note)
        self.assertIn("V4 (0.34 mHz), V5 (0.31 mHz)", note)
        self.assertIn("unscored, not measured faint", note)

    def test_all_scored_adds_no_note(self):
        ns = self._snr_block(np.full((3, 4, 6), 49.0))
        self.assertEqual(ns["VGB_HH_NOTE"], "")

    def test_the_per_iteration_snr_panel_warns_nothing(self):
        ns = _generator_functions({"np": np}, names=("finite_mean",))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            m = ns["finite_mean"](self._hh(), axis=1)
        self.assertEqual(m.shape, (5, 6))
        self.assertTrue(np.isnan(m[:, 4:]).all())
        np.testing.assert_allclose(m[:, :4], 100.0)

    def test_the_caption_and_the_trend_panel_use_them(self):
        src = _src()
        self.assertIn("not a fit failure.{VGB_HH_NOTE}</div>", src)
        self.assertIn("snr_it = np.sqrt(np.clip(finite_mean(vgb_hh[:VGB_NIT], axis=1)", src)


class AndersonTest(unittest.TestCase):

    def test_same_statistic_and_no_future_warning(self):
        from scipy.stats import anderson

        ns = _generator_functions({"np": np}, names=("ad_statistic",))
        v = np.random.default_rng(0).standard_normal(600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = float(anderson(v, dist="norm").statistic)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            self.assertEqual(ns["ad_statistic"](v), want)

    def test_the_residual_panel_calls_it(self):
        src = _src()
        self.assertIn("_A2 = ad_statistic(_v)", src)
        self.assertNotIn('_anderson(_v, dist="norm")', src)


class TightLayoutTest(unittest.TestCase):

    def test_matplotlib_skips_a_gridspec_with_its_own_spacing(self):
        # the rule the generator check below rests on
        fig, ax = plt.subplots(2, 1, gridspec_kw=dict(hspace=0.08))
        self.addCleanup(plt.close, fig)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig.tight_layout()
        self.assertTrue(any("not compatible with tight_layout" in str(x.message) for x in w))

    def test_no_figure_sets_gridspec_spacing_and_then_calls_tight_layout(self):
        lines = _src().splitlines()
        bad, spaced, i = [], None, 0
        while i < len(lines):
            ln = lines[i]
            if "plt.subplots(" in ln and not ln.lstrip().startswith("#"):
                stmt, depth, j = "", 0, i
                while True:
                    stmt += lines[j]
                    depth += lines[j].count("(") - lines[j].count(")")
                    if depth <= 0:
                        break
                    j += 1
                spaced = (i + 1) if re.search(r"gridspec_kw=.*[hw]space", stmt) else None
                i = j
            elif "tight_layout(" in ln and not ln.lstrip().startswith("#") and spaced:
                bad.append((spaced, i + 1))
            elif "fig_b64(" in ln:
                spaced = None
            i += 1
        self.assertEqual(bad, [], "figure (line) with gridspec spacing -> tight_layout (line)")


def _tim_re():
    tree = ast.parse(_src())
    for n in tree.body:
        if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "TIM_RE"):
            return eval(compile(ast.Expression(n.value), "TIM_RE", "eval"), {"re": re})
    raise AssertionError("TIM_RE is gone")


FANOUT_LOG = "\n".join([
    "2026-10-13 05:38:19,955 - lisatools.globalfit.moves.gbspecialstretch - INFO - "
    "[GB_TIMING rj_prior_pe] total=317.381s tracked=0.000s untracked=317.381s | ",
    "2026-10-13 05:38:36,396 - lisatools.globalfit.moves.gbspecialstretch - INFO - "
    "[GB_TIMING vgb_pe] total=16.402s tracked=0.000s untracked=16.402s | ",
    "2026-10-13 02:44:28,169 - lisatools.globalfit.moves.gbspecialstretch - INFO - "
    "[GB_TIMING rj_fstat_pe_residual] total=2380.209s tracked=2100.383s "
    "untracked=279.826s | fstat_grid_fit=2100.383s",
])


class TimingPanelTest(unittest.TestCase):

    def test_fanout_lines_with_an_empty_tail_are_read(self):
        rx = _tim_re()
        self.assertEqual(sum(1 for ln in FANOUT_LOG.splitlines() if rx.match(ln)), 3)
        m = rx.match("2026-10-13 00:00:00,1 - x - INFO - [GB_TIMING rj_x] total=5.0s "
                     "tracked=1s | picked_sources=40 inmodel_blocks=2")
        self.assertEqual(m.group(4), "picked_sources=40 inmodel_blocks=2")

    def test_no_counters_draws_no_log_axis_and_says_so(self):
        ns = {"np": np, "re": re, "plt": plt, "io": io, "base64": base64, "datetime": datetime,
              "log_text": FANOUT_LOG, "IMGS": {}, "_shrink_png": shrink_png,
              "GREEN": "#58C48A", "VIOLET": "#9B7BFF", "AMBER": "#F5A623",
              "CYAN": "#4FD8EB", "DIM": "#67788A"}
        _generator_functions(ns, names=("fig_b64",))
        blk = _between(_src(), "TIM_RE = re.compile(", "# ---- 8b. SEARCH EFFICIENCY")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            exec(blk, ns)                            # noqa: S102
        self.assertEqual(len(ns["recs"]), 3)
        self.assertIn("timing_moves", ns["IMGS"])
        self.assertIn("None of the 3", ns["TIMING_TXT"])
        self.assertIn("{TIMING_TXT}", _src())


class MissingLogTest(unittest.TestCase):

    def test_one_plain_line_per_note(self):
        ns = _generator_functions({"np": np}, names=("missing_log_lines",))
        out = ns["missing_log_lines"](
            ["chain/mbh: torn in the live store &mdash; <code>x</code>",
             "VGB SNR>7 panel: " + "y" * 900],
            ["overlap: none &lt;here&gt;"])
        self.assertEqual(out[0], "[monitor] MISSING: chain/mbh: torn in the live store — x")
        self.assertTrue(out[1].startswith("[monitor] MISSING: VGB SNR>7 panel: yyy"))
        self.assertLessEqual(len(out[1]), len("[monitor] MISSING: ") + 600)
        self.assertEqual(out[2], "[monitor] MISSING: at a glance: overlap: none <here>")

    def test_the_page_prints_them_after_its_summary_line(self):
        src = _src()
        i_sum = src.index('f"missing={len(MISSING)}")')
        i_log = src.index("for _ln in missing_log_lines(MISSING, GLANCE_NOTES):\n    print(_ln)")
        self.assertLess(i_sum, i_log)
        self.assertIn("GLANCE_NOTES = list(_G[\"notes\"])", src)


class _FakeGBW:
    """Just enough of GBGPU for opt_snr: unit A / E from bin 0."""

    def run_wave(self, *cols, N, T, dt, tdi2, tdi_channel_setup):
        n = len(cols[0])
        self.A = np.ones((n, N), complex)
        self.E = np.ones((n, N), complex)
        self.start_inds = np.zeros(n, int)


class SnrProgressTest(unittest.TestCase):

    def _run(self, nrows, batch, **kw):
        phys = np.ones((nrows, 9))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = bt.opt_snr(phys, np.ones(64), np.ones(64), _FakeGBW(), 1.0, 1.0, 8,
                             batch=batch, **kw)
        return out, [ln for ln in buf.getvalue().splitlines() if "snr:" in ln]

    def test_at_most_a_line_per_quarter_and_one_at_the_end(self):
        out, lines = self._run(1009, 20)                # 51 batches, as on the 9mo page
        np.testing.assert_allclose(out, 8.0)            # sqrt(4 df (8 + 8)), df = 1
        self.assertLessEqual(len(lines), 5)
        self.assertEqual(lines[-1].strip(), "snr: 1009/1009")

    def test_a_single_batch_prints_once(self):
        _, lines = self._run(48, 20000)
        self.assertEqual([ln.strip() for ln in lines], ["snr: 48/48"])

    def test_quiet_when_asked(self):
        _, lines = self._run(100, 10, progress=False)
        self.assertEqual(lines, [])

    def test_the_kappa_prefilter_is_quiet(self):
        with open(bt.__file__, encoding="utf-8") as fh:
            src = fh.read()
        call = src[src.index("kap = np.maximum(kap, opt_snr("):]
        self.assertIn("progress=False", call[:call.index("/ A0)")])


def _brick(path, good):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if good:
        with h5py.File(path, "w") as f:
            f.create_group("ltts")
            f.create_group("orbits")
    else:
        with open(path, "wb") as fh:                   # a truncated download
            fh.write(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)


class BrickWarningTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        d = os.path.join(self.root, "data")
        self.noise = os.path.join(d, "INSTRUMENT", "L1",
                                  "NOISE_731d_2.5s_L1_source0_0_20251206T220508Z.h5")
        self.sobhb = os.path.join(d, "SOBHB", "L1",
                                  "SOBHB_731d_2.5s_L1_source2_0_20251215T183326Z.h5")
        self.vgb = os.path.join(d, "VGB", "L1", "VGB_731d_2.5s_L1_source0_0_20251215T1919Z.h5")
        _brick(self.noise, False)
        _brick(self.sobhb, False)
        _brick(self.vgb, True)
        bt._REPORTED_BAD_BRICKS.clear()
        self.addCleanup(bt._REPORTED_BAD_BRICKS.clear)

    def _find(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"MOJITO_INFO_PATH": self.root}), \
                contextlib.redirect_stdout(buf):
            fp = bt.find_l1_brick()
        return fp, buf.getvalue().splitlines()

    def test_each_bad_brick_is_one_line_saying_what_the_page_loses(self):
        fp, lines = self._find()
        self.assertEqual(fp, self.vgb)
        self.assertIn("skipped 2 unusable mojito L1 brick(s)", lines[0])
        self.assertIn("the truth set loses nothing", lines[0])
        noise = next(ln for ln in lines if "NOISE_731d" in ln)
        self.assertIn("noise truth lines fall back to the analytic injection", noise)
        sob = next(ln for ln in lines if "SOBHB_731d" in ln)
        self.assertIn("left out of the page's data sum, but only on a run that did not "
                      "read COMBINED", sob)
        self.assertIn("re-fetch them", lines[-1])
        self.assertEqual(len(lines), 4)

    def test_a_second_caller_in_the_same_process_prints_nothing(self):
        self._find()
        fp, lines = self._find()
        self.assertEqual(fp, self.vgb)
        self.assertEqual(lines, [])

    def test_the_page_names_an_unreadable_brick_in_its_notes(self):
        ns = {"os": os, "_l1_usable": bt._l1_usable, "brick_loss_note": bt.brick_loss_note}
        _generator_functions(ns, names=("unusable_brick_note",))
        note = ns["unusable_brick_note"](self.noise, "noise truth lines")
        self.assertTrue(note.startswith("noise truth lines: mojito brick NOISE_731d"))
        self.assertIn("re-fetch it", note)
        self.assertIsNone(ns["unusable_brick_note"](self.vgb, "x"))
        self.assertIsNone(ns["unusable_brick_note"](self.vgb + ".absent", "x"))

    def test_the_data_sum_skips_a_truncated_brick_instead_of_failing(self):
        blk = _between(_src(), "    if not _files:\n        for _t in _types:",
                       "        if not _files:\n            raise FileNotFoundError")
        self.assertIn('_bad = unusable_brick_note(_fp_b, f"data/template/residual', blk)
        self.assertLess(blk.index("_bad = unusable_brick_note(_fp_b"),
                        blk.index("_files[f\"{_t}{_sid}\""))


if __name__ == "__main__":
    unittest.main()
