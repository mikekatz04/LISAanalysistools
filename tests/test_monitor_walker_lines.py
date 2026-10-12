"""The monitor's lnL and GB leaf-count figures draw every cold walker (2026-10-12).

User request: "Adjust the logL and leaf plots over iteration from min-max-mean
to all 4 walkers + mean". The at-a-glance block (``_short.fig_history``, at the
top of both pages) drew a min-max band over the walkers plus the mean; the full
page's own detail figures drew every walker in one colour plus the max (and
the median). All four now go through the generator's ``plot_walker_lines``:
one thin line per cold walker in ``walker_color(k)`` -- the same colour for a
walker on every panel, inset and page -- and the walker mean as one thick line.
The walker count is the array's, NaN rows are gaps.

The generator is a script, so its helpers are compiled by name (as
``tests/test_monitor_tf_pooling.py`` does) and its two inline figure blocks are
cut out of its source and executed on synthetic arrays (as
``tests/test_monitor_short_page.py`` does with the overlap loop).
"""
import ast
import io
import re
import unittest
import warnings
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

from lisatools.globalfit.monitor import _short as sh  # noqa: E402
from lisatools.globalfit.monitor import generator_path  # noqa: E402
from lisatools.globalfit.monitor._short import _generator_functions  # noqa: E402

HELPERS = ("walker_color", "plot_walker_lines", "mark_stages", "_zoom_its", "_mhz")
CONSTS = ("WALKER_COLORS", "WALKER_OTHER_COLOR", "WALKER_MEAN_COLOR")
RED = "#E5484D"
NIT, NLEAF = 120, 60
STAGES = [(70, "gb_search_1 done -> gb_search_2")]
DET_LO, DET_HI = 0.8e-3, 21.94e-3


def compile_ns(stage_bounds=STAGES):
    ns = {"np": np}
    _generator_functions(ns, names=HELPERS, consts=CONSTS)
    ns["STAGE_BOUNDS"] = list(stage_bounds)
    return ns


NS = compile_ns()
walker_color = NS["walker_color"]


def _src():
    with open(generator_path(), encoding="utf-8") as fh:
        return fh.read()


SRC = _src()


def func_src(src, name):
    """The source of top-level ``def name`` in ``src``."""
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise KeyError(name)


def ll_block(src=SRC):
    i = src.index("# ---- 1. likelihood ----")
    j = src.index('fig_b64(fig, "ll")')
    return src[i:j + len('fig_b64(fig, "ll")')]


def leaf_block(src=SRC):
    k = src.index("# 2026-10-12; was every walker in one green")
    i = src.rindex("fig, ax = plt.subplots(", 0, k)
    j = src.index('fig_b64(fig, "gb_leaves")', k)
    return src[i:j + len('fig_b64(fig, "gb_leaves")')]


# --------------------------------------------------------------------------
# synthetic runs
# --------------------------------------------------------------------------
def synth_ll(nw=4, nit=NIT, seed=0):
    """Cold lnL (rows, walkers): a rising curve, one walker stuck lower."""
    rng = np.random.default_rng(seed)
    base = -5e4 * np.exp(-np.arange(nit) / 25.0)[:, None]
    y = base + rng.normal(0.0, 3.0, size=(nit, nw)) + 4.0 * np.arange(nw)[None, :]
    y[:, 0] -= 40.0 * (np.arange(nit)[:, None] > 60).ravel()
    return y


def synth_leaves(nw=4, nit=NIT, nleaf=NLEAF, seed=1):
    """``inds`` (rows, walkers, leaves) growing over the run, and f0 [mHz]
    per (walker, leaf) with a few leaves below the analysed band."""
    rng = np.random.default_rng(seed)
    n_alive = np.clip((np.arange(nit)[:, None] * 0.45 + rng.integers(0, 4, (nit, nw))),
                      0, nleaf).astype(int)
    inds = np.arange(nleaf)[None, None, :] < n_alive[:, :, None]
    f0 = rng.uniform(1.0, 20.0, size=(nw, nleaf))
    f0[:, ::6] = 0.6                                     # outside 0.8-21.94 mHz
    return inds, f0


def short_figure(y, ns=None, target=None, title="cold log-likelihood"):
    """The at-a-glance history figure for ``y``, as a Figure (``_png`` is
    intercepted, so nothing is encoded)."""
    ns = compile_ns() if ns is None else ns
    got = []
    with mock.patch.object(sh, "_png", lambda fig, dpi=None: got.append(fig) or "png"):
        sh.fig_history(ns, y, "cold lnL - 0.0", title, target=target)
    return got[0]


def generator_figures(ll, inds, f0, *, det=True, unreadable_rows=(), ns=None):
    """Run the generator's own lnL and GB-leaf blocks on synthetic arrays and
    return ``{"ll": fig, "gb_leaves": fig}``."""
    figs = {}
    ns = compile_ns() if ns is None else ns   # NOT a copy: the helpers' globals are ns
    nit = ll.shape[0]

    def _row(name, sel):
        assert name == "chain/gb", name
        if sel[0] in unreadable_rows:
            return None
        return f0.copy()

    ns.update(
        plt=plt, re=re, inset_axes=inset_axes, it=np.arange(nit), ll=ll, nwalk=ll.shape[1],
        NIT=nit, gb_inds=inds, gb_counts=inds.sum(axis=-1), _row=_row,
        DET_F0=np.arange(55, dtype=float) if det else None, DET_LO=DET_LO, DET_HI=DET_HI,
        caps=np.full((nit, 12), 30.0), CAP_UNIT="cap cell", CAP_K=1, log_text="",
        CYAN="#4FD8EB", AMBER="#F5A623", GREEN="#58C48A", RED=RED, VIOLET="#9B7BFF",
        FG="#B8C6D4", DIM="#67788A",
        fig_b64=lambda fig, key, dpi=None: figs.__setitem__(key, fig))
    exec(compile(ll_block(), "generator:ll", "exec"), ns)          # noqa: S102
    exec(compile(leaf_block(), "generator:gb_leaves", "exec"), ns)  # noqa: S102
    return figs


def legend_labels(ax):
    leg = ax.get_legend()
    return [] if leg is None else [t.get_text() for t in leg.get_texts()]


def line_of(ax, ydata):
    """The one line on ``ax`` drawing exactly ``ydata`` (NaN == NaN)."""
    hits = [ln for ln in ax.get_lines()
            if np.asarray(ln.get_ydata(), float).shape == np.shape(ydata)
            and np.array_equal(np.asarray(ln.get_ydata(), float), ydata, equal_nan=True)]
    assert len(hits) == 1, f"{len(hits)} lines draw this series"
    return hits[0]


def walker_colors_on(ax, y):
    """Colour of each walker column of ``y`` as drawn on ``ax``."""
    return [to_hex(line_of(ax, np.asarray(y, float)[:, w]).get_color())
            for w in range(np.shape(y)[1])]


def insets(fig):
    """The zoom insets (axes that are not the two main panels or a colorbar)."""
    main = fig.axes[:2]
    return [a for a in fig.axes[2:] if a not in main and a.get_label() != "<colorbar>"]


def _close(*figs):
    for f in figs:
        plt.close(f)


# --------------------------------------------------------------------------
# (a) wiring: no min-max fill, one line per walker plus the mean
# --------------------------------------------------------------------------
class WiringTest(unittest.TestCase):
    def test_the_glance_history_figure_draws_walkers_not_a_band(self):
        with open(sh.__file__, encoding="utf-8") as fh:
            short = fh.read()
        body = func_src(short, "fig_history")
        self.assertIn('ns["plot_walker_lines"](a, x[sl], y[sl], label=a is ax[0])', body)
        self.assertNotIn("fill_between", body)
        self.assertNotIn("min-max over walkers", short)
        self.assertNotIn("_band_panel", short)
        self.assertIn("plot_walker_lines", sh.GEN_HELPERS)
        self.assertIn("walker_color", sh.GEN_HELPERS)
        for c in CONSTS:
            self.assertIn(c, sh.GEN_CONSTS)

    def test_the_full_pages_two_figures_draw_walkers_and_the_mean(self):
        ll, leaf = ll_block(), leaf_block()
        self.assertIn("plot_walker_lines(ax[0], it, ll)", ll)
        self.assertIn("plot_walker_lines(axins0, _it_zoom, _ll_zoom, label=False", ll)
        self.assertIn("plot_walker_lines(ax[0], it, _cnt)", leaf)
        self.assertIn("plot_walker_lines(ax[0], it, gb_counts)", leaf)
        self.assertIn("plot_walker_lines(axins_gb, _it_zoom, _gb_zoom, label=False", leaf)
        for blk in (ll, leaf):
            self.assertNotIn("fill_between", blk)
            # a max / min / median over walkers drawn as its own line
            self.assertIsNone(re.search(r"plot\([^,()]+, *\w+\.(max|min)\(axis=1\), *color=",
                                        blk))
            self.assertNotIn("np.median(", blk)
            self.assertNotIn("range(nwalk)", blk)
        # the right-hand spread panel is kept as it was
        self.assertIn("ax[1].plot(it, ll.max(axis=1) - ll.min(axis=1), color=VIOLET", ll)
        # the helper is defined before the script first calls it (it runs top-down)
        self.assertLess(SRC.index("def plot_walker_lines("),
                        SRC.index("plot_walker_lines(ax[0], it, ll)"))

    def test_the_walker_count_is_never_a_literal(self):
        body = func_src(SRC, "plot_walker_lines")
        self.assertIn("nw, nc = y.shape[1], len(WALKER_COLORS)", body)
        self.assertIsNone(re.search(r"range\(\s*4\s*\)|% *4\b", body))


class RenderedLabelsTest(unittest.TestCase):
    """The rendered Line2D labels: walker 0..N-1 once per figure, then the mean."""

    def test_glance_figure_labels_and_no_fill(self):
        y = synth_ll()
        fig = short_figure(y)
        ax0, ax1 = fig.axes[:2]
        self.assertEqual(legend_labels(ax0), [f"walker {w}" for w in range(4)] + ["mean"])
        self.assertIsNone(ax1.get_legend())
        for a in (ax0, ax1):
            self.assertEqual(len(a.collections), 0)           # no fill_between band
            labelled = [ln.get_label() for ln in a.get_lines()
                        if not ln.get_label().startswith("_")]
            self.assertEqual(labelled, legend_labels(ax0) if a is ax0 else [])
        # the mean is the plain walker mean, thicker and in the mean colour
        mean = line_of(ax0, y.mean(axis=1))
        walkers = [line_of(ax0, y[:, w]) for w in range(4)]
        self.assertEqual(to_hex(mean.get_color()), NS["WALKER_MEAN_COLOR"].lower())
        self.assertGreater(mean.get_linewidth(), 2 * max(ln.get_linewidth() for ln in walkers))
        # both panels draw every walker: the zoom panel the trailing window
        nz = NS["_zoom_its"](NIT)
        self.assertEqual(len(walker_colors_on(ax1, y[-nz:])), 4)
        line_of(ax1, y[-nz:].mean(axis=1))
        # stage marker + its label still drawn on the whole-run panel
        self.assertTrue(any(t.get_text() == STAGES[0][1] for t in ax0.texts))
        _close(fig)

    def test_glance_leaf_figure_keeps_its_target(self):
        inds, _ = synth_leaves()
        y = inds.sum(axis=-1)
        fig = short_figure(y, target=(55, "55 detectable (SNR > 7)"), title="GB leaf count")
        ax0 = fig.axes[0]
        self.assertEqual(legend_labels(ax0), [f"walker {w}" for w in range(4)] + ["mean"])
        self.assertTrue(any(to_hex(ln.get_color()) == RED.lower() and ln.get_linestyle() == ":"
                            for ln in ax0.get_lines()))
        self.assertTrue(any("detectable" in t.get_text() for t in ax0.texts))
        _close(fig)

    def test_full_page_figures_labels(self):
        ll = synth_ll()
        inds, f0 = synth_leaves()
        figs = generator_figures(ll, inds, f0)
        a_ll = figs["ll"].axes[0]
        self.assertEqual(legend_labels(a_ll), [f"walker {w}" for w in range(4)] + ["mean"])
        self.assertEqual(figs["ll"].axes[1].get_title(), "walker lnL spread (max - min)")
        a_gb = figs["gb_leaves"].axes[0]
        self.assertEqual(legend_labels(a_gb),
                         [f"walker {w}" for w in range(4)] + ["mean", "all f, mean only"])
        self.assertEqual(a_gb.get_legend().get_title().get_text(),
                         "0.8-21.94 mHz: walkers + mean")
        # per walker = the analysed-band count, all f = its mean only
        band = ((f0[None] >= 0.8) & (f0[None] <= 21.94) & inds).sum(axis=-1).astype(float)
        self.assertEqual(len(walker_colors_on(a_gb, band)), 4)
        line_of(a_gb, band.mean(axis=1))
        line_of(a_gb, inds.sum(axis=-1).mean(axis=1))
        # the detectable reference line and its label are still there
        self.assertTrue(any(to_hex(ln.get_color()) == RED.lower() for ln in a_gb.get_lines()))
        self.assertTrue(any("detectable (SNR>7)" in t.get_text() for t in a_gb.texts))
        # the insets: every walker and the mean of the trailing window, no legend
        nz = NS["_zoom_its"](NIT)
        (ins_ll,), (ins_gb,) = insets(figs["ll"])[:1], insets(figs["gb_leaves"])
        self.assertEqual(ins_ll.get_title(), f"last {nz} iter")
        self.assertEqual(ins_gb.get_title(), f"last {nz} iter")
        self.assertEqual(len(walker_colors_on(ins_ll, ll[-nz:])), 4)
        self.assertEqual(len(walker_colors_on(ins_gb, band[-nz:])), 4)
        line_of(ins_gb, band[-nz:].mean(axis=1))
        self.assertIsNone(ins_gb.get_legend())
        _close(*figs.values())

    def test_full_page_leaf_figure_without_a_truth_set(self):
        ll = synth_ll()
        inds, f0 = synth_leaves()
        figs = generator_figures(ll, inds, f0, det=False)
        a_gb = figs["gb_leaves"].axes[0]
        self.assertEqual(legend_labels(a_gb), [f"walker {w}" for w in range(4)] + ["mean"])
        self.assertEqual(a_gb.get_legend().get_title().get_text(), "all f: walkers + mean")
        counts = inds.sum(axis=-1).astype(float)
        self.assertEqual(len(walker_colors_on(a_gb, counts)), 4)
        self.assertFalse(any(to_hex(ln.get_color()) == RED.lower() for ln in a_gb.get_lines()))
        _close(*figs.values())


# --------------------------------------------------------------------------
# (b) one colour per walker index, the same on every figure / panel / inset
# --------------------------------------------------------------------------
class SameColourEverywhereTest(unittest.TestCase):
    def test_walker_k_has_one_colour_on_all_four_figures(self):
        ll = synth_ll()
        inds, f0 = synth_leaves()
        leaves = inds.sum(axis=-1)
        band = ((f0[None] >= 0.8) & (f0[None] <= 21.94) & inds).sum(axis=-1).astype(float)
        nz = NS["_zoom_its"](NIT)
        f_lnl = short_figure(ll)
        f_lv = short_figure(leaves, title="GB leaf count")
        gen = generator_figures(ll, inds, f0)
        seen = {
            "glance lnL, all rows": walker_colors_on(f_lnl.axes[0], ll),
            "glance lnL, last rows": walker_colors_on(f_lnl.axes[1], ll[-nz:]),
            "glance leaves, all rows": walker_colors_on(f_lv.axes[0], leaves),
            "glance leaves, last rows": walker_colors_on(f_lv.axes[1], leaves[-nz:]),
            "page lnL": walker_colors_on(gen["ll"].axes[0], ll),
            "page lnL inset": walker_colors_on(insets(gen["ll"])[0], ll[-nz:]),
            "page leaves": walker_colors_on(gen["gb_leaves"].axes[0], band),
            "page leaves inset": walker_colors_on(insets(gen["gb_leaves"])[0], band[-nz:]),
        }
        want = [to_hex(walker_color(w)) for w in range(4)]
        for where, cols in seen.items():
            with self.subTest(where=where):
                self.assertEqual(cols, want)
        _close(f_lnl, f_lv, *gen.values())

    def test_the_palette_is_distinct_and_never_the_target_red(self):
        cols = [to_hex(c) for c in NS["WALKER_COLORS"]]
        self.assertEqual(len(set(cols)), len(cols))
        self.assertGreaterEqual(len(cols), 4)                 # today's 4 walkers all hued
        self.assertNotIn(RED.lower(), cols)
        self.assertNotIn(to_hex(NS["WALKER_MEAN_COLOR"]), cols)
        self.assertNotIn(to_hex(NS["WALKER_OTHER_COLOR"]), cols)
        # the colour is a function of the walker INDEX, not of how many there are
        self.assertEqual([walker_color(w) for w in range(3)], list(NS["WALKER_COLORS"][:3]))


# --------------------------------------------------------------------------
# (c) the walker count is the array's
# --------------------------------------------------------------------------
class WalkerCountFromShapeTest(unittest.TestCase):
    def test_three_and_five_walkers(self):
        for nw in (3, 5):
            with self.subTest(nwalkers=nw):
                ll = synth_ll(nw=nw)
                inds, f0 = synth_leaves(nw=nw)
                f = short_figure(ll)
                gen = generator_figures(ll, inds, f0)
                want = [f"walker {w}" for w in range(nw)]
                self.assertEqual(legend_labels(f.axes[0]), want + ["mean"])
                self.assertEqual(legend_labels(gen["ll"].axes[0]), want + ["mean"])
                self.assertEqual(legend_labels(gen["gb_leaves"].axes[0])[:nw], want)
                cols = [to_hex(walker_color(w)) for w in range(nw)]
                self.assertEqual(walker_colors_on(f.axes[0], ll), cols)
                self.assertEqual(walker_colors_on(gen["ll"].axes[0], ll), cols)
                # the mean is over all nw walkers
                line_of(f.axes[0], ll.mean(axis=1))
                line_of(gen["ll"].axes[0], ll.mean(axis=1))
                _close(f, *gen.values())

    def test_walkers_past_the_palette_fold_into_one_labelled_muted_line(self):
        nc = len(NS["WALKER_COLORS"])
        y = synth_ll(nw=nc + 3)
        f = short_figure(y)
        ax0 = f.axes[0]
        self.assertEqual(legend_labels(ax0), [f"walker {w}" for w in range(nc)]
                         + [f"walkers {nc}-{nc + 2}", "mean"])
        cols = walker_colors_on(ax0, y)
        self.assertEqual(cols[nc:], [to_hex(NS["WALKER_OTHER_COLOR"])] * 3)
        self.assertEqual(len(set(cols[:nc])), nc)
        _close(f)


# --------------------------------------------------------------------------
# (d) NaN rows (not yet written / unreadable) are gaps, not errors
# --------------------------------------------------------------------------
class NaNRowsTest(unittest.TestCase):
    def test_nan_rows_render_as_gaps_without_warnings(self):
        y = synth_ll()
        y[-8:] = np.nan                       # rows not written yet
        y[40, 2] = np.nan                     # one walker unreadable on one row
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            f = short_figure(y)
            f.savefig(io.BytesIO(), format="png")
        mean = line_of(f.axes[0], y.mean(axis=1)).get_ydata()
        self.assertTrue(np.isnan(mean[-8:]).all())
        self.assertTrue(np.isnan(mean[40]))  # the plain mean: not averaged over 3 walkers
        self.assertTrue(np.isfinite(mean[:40]).all())
        _close(f)

    def test_an_all_nan_zoom_window_does_not_raise(self):
        y = synth_ll()
        y[-60:] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            f = short_figure(y)
            f.savefig(io.BytesIO(), format="png")          # the real draw
        _close(f)

    def test_full_page_figures_with_unreadable_rows(self):
        ll = synth_ll()
        inds, f0 = synth_leaves()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            gen = generator_figures(ll, inds, f0, unreadable_rows=(5, 6, NIT - 1))
            for g_ in gen.values():
                g_.savefig(io.BytesIO(), format="png")
        band_mean = line_of(gen["gb_leaves"].axes[0],
                            np.where(np.isin(np.arange(NIT), (5, 6, NIT - 1))[:, None], np.nan,
                                     ((f0[None] >= 0.8) & (f0[None] <= 21.94) & inds)
                                     .sum(axis=-1)).mean(axis=1)).get_ydata()
        self.assertTrue(np.isnan(band_mean[[5, 6, NIT - 1]]).all())
        _close(*gen.values())


# --------------------------------------------------------------------------
# negative control: the checks above SEE the old min-max band
# --------------------------------------------------------------------------
def _old_band(ax, x, y, *, label=True, **_):
    """The pre-2026-10-12 glance panel: min-max fill + min/max edges + mean."""
    y = np.asarray(y, float)
    ax.fill_between(x, y.min(axis=1), y.max(axis=1), color="#4FD8EB", alpha=0.22, lw=0,
                    label="min-max over walkers" if label else None)
    ax.plot(x, y.max(axis=1), color="#4FD8EB", lw=1.0)
    ax.plot(x, y.min(axis=1), color="#4FD8EB", lw=1.0)
    ax.plot(x, y.mean(axis=1), color="#B8C6D4", lw=1.5, label="mean" if label else None)


class NegativeControlTest(unittest.TestCase):
    def test_the_old_band_fails_the_label_fill_and_colour_checks(self):
        ns = compile_ns()
        ns["plot_walker_lines"] = _old_band
        y = synth_ll()
        f = short_figure(y, ns=ns)
        ax0 = f.axes[0]
        self.assertNotEqual(legend_labels(ax0), [f"walker {w}" for w in range(4)] + ["mean"])
        self.assertGreater(len(ax0.collections), 0)
        with self.assertRaises(AssertionError):
            walker_colors_on(ax0, y)
        _close(f)


if __name__ == "__main__":
    unittest.main()
