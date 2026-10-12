"""The monitor's time-frequency figure pools |w| to the PIXEL GRID (2026-10-11).

The "WDM domain - |w_mn| on the run's own grid" figure (data | all fitted
signals | residual, rows X/Y/Z) showed MBHB merger lines in the wrong panels
on the 9mo page: the 102 d merger (SNR ~1010) bright in the data column and
faint in the fitted-signals column, 109 d and 171 d the other way round. A
merger is ONE WDM time column (3600 s); the page max-pooled 5 columns per
drawn column (6361 -> 1272) into a ~347 px axes, and matplotlib's
nearest-neighbour resampling kept one drawn column in ~3.7 -- a different
subset per panel, because the panels differ in width (the colorbar column
is narrower). Raster aliasing, not data.

The figure now lays out first, pools every column's |w| with MAX to at most
one column per screen pixel (the same target for all three columns) and
draws with ``interpolation="nearest"``. The generator is a script, so its
helpers are compiled by name, as ``tests/test_monitor_walker_residual.py``
does.
"""
import re
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lisatools.globalfit.monitor import generator_path  # noqa: E402
from lisatools.globalfit.monitor._short import _generator_functions  # noqa: E402

HELPERS = ("tf_pool_block", "pool_tf_for_display", "tf_display_grid",
           "mbh_merger_plot_days", "draw_wdm_tf_panels")


def _compile(**overrides):
    ns = {"np": np, "plt": plt, "FG": "#B8C6D4", "DIM": "#67788A"}
    _generator_functions(ns, names=HELPERS, consts=("MBH_T_PLUNGE_COL",))
    ns.update(overrides)
    return ns


NS = _compile()
pool = NS["pool_tf_for_display"]
block = NS["tf_pool_block"]


def _ceil(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------
# (a) a one-column spike survives MAX pooling at every factor and phase
# --------------------------------------------------------------------------
class MaxPoolKeepsOneColumnSpikes(unittest.TestCase):
    NT, NF = 600, 20

    def _bg(self):
        return np.random.default_rng(3).uniform(0.0, 1.0, size=(3, self.NF, self.NT))

    def test_merger_column_survives_every_factor_and_phase(self):
        seen = set()
        for factor in range(1, 31):
            target = _ceil(self.NT, factor)
            b = block(self.NT, target)
            seen.add(b)
            for phase in range(b):
                col = 7 * b + phase
                a = self._bg()
                a[..., col] = 10.0                 # a merger: every layer, every channel
                out = pool(a, target)
                self.assertLessEqual(out.shape[-1], target)
                k = col // b
                with self.subTest(block=b, phase=phase):
                    np.testing.assert_array_equal(out[..., k], 10.0)
                    # and nowhere else: the spike did not smear into a neighbour
                    self.assertEqual(int((out == 10.0).sum()), 3 * self.NF)
        self.assertTrue({1, 2, 3, 5, 8, 13, 20, 30} <= seen)

    def test_single_pixel_survives_time_and_layer_pooling(self):
        for bt, bf in ((4, 1), (7, 3), (19, 2), (30, 5)):
            for phase_t in range(bt):
                for phase_f in range(bf):
                    t, f = 5 * bt + phase_t, 1 * bf + phase_f
                    if t >= self.NT or f >= self.NF:
                        continue
                    a = self._bg()
                    a[1, f, t] = 10.0
                    out = pool(a, _ceil(self.NT, bt), n_rows=_ceil(self.NF, bf))
                    with self.subTest(bt=bt, bf=bf, t=t, f=f):
                        self.assertEqual(out[1, f // block(self.NF, _ceil(self.NF, bf)),
                                             t // block(self.NT, _ceil(self.NT, bt))], 10.0)


# --------------------------------------------------------------------------
# (b) MEAN pooling dilutes the same spike by the block length
# --------------------------------------------------------------------------
class MeanPoolAttenuates(unittest.TestCase):
    def test_mean_dilutes_a_spike_by_the_factor(self):
        nt = 600
        for factor in (2, 3, 4, 5, 8, 10, 12):
            target = _ceil(nt, factor)
            b = block(nt, target)
            for phase in range(b):
                a = np.zeros((2, 4, nt))
                col = 3 * b + phase
                a[..., col] = 12.0
                with self.subTest(block=b, phase=phase):
                    np.testing.assert_allclose(pool(a, target, how="mean")[..., col // b],
                                               12.0 / b)
                    np.testing.assert_allclose(pool(a, target, how="max")[..., col // b],
                                               12.0)


# --------------------------------------------------------------------------
# (c) ragged last block, NaNs, no-op targets, input checks
# --------------------------------------------------------------------------
class RaggedNaNAndNoop(unittest.TestCase):
    def test_ragged_last_block_reduces_over_what_it_has(self):
        a = np.arange(10.0)[None, :]                   # block 4: [0-3] [4-7] [8-9]
        self.assertEqual(block(10, 3), 4)
        np.testing.assert_array_equal(pool(a, 3), [[3.0, 7.0, 9.0]])
        np.testing.assert_allclose(pool(a, 3, how="mean"), [[1.5, 5.5, 8.5]])
        s = np.zeros((1, 10))
        s[0, 9] = 5.0                                  # a spike in the ragged tail
        np.testing.assert_array_equal(pool(s, 3), [[0.0, 0.0, 5.0]])

    def test_nans_are_ignored_and_an_all_nan_block_is_nan(self):
        a = np.array([[np.nan, 1.0, 2.0, np.nan, np.nan, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")             # no all-NaN RuntimeWarning either
            mx = pool(a, 3)
            mn = pool(a, 3, how="mean")
        np.testing.assert_array_equal(mx[0, :2], [1.0, 2.0])
        np.testing.assert_array_equal(mn[0, :2], [1.0, 2.0])
        self.assertTrue(np.isnan(mx[0, 2]) and np.isnan(mn[0, 2]))

    def test_target_wider_than_the_input_is_a_noop(self):
        a = np.random.default_rng(0).uniform(size=(3, 5, 40)).astype(np.float32)
        for t in (40, 41, 10_000):
            out = pool(a, t, n_rows=5)
            self.assertEqual(out.shape, a.shape)
            np.testing.assert_array_equal(out, a)
        self.assertEqual(pool(a, 40).dtype, np.float32)  # no float64 copy for nothing
        self.assertEqual(block(40, 40), 1)
        self.assertEqual(block(40, 1000), 1)

    def test_integer_input_is_promoted_and_bad_arguments_raise(self):
        self.assertTrue(np.issubdtype(pool(np.arange(12).reshape(1, 12), 4).dtype,
                                      np.floating))
        with self.assertRaises(ValueError):
            pool(np.ones((2, 8)), 4, how="median")
        with self.assertRaises(ValueError):
            pool(np.ones(8), 4)
        with self.assertRaises(ValueError):
            block(8, 0)


# --------------------------------------------------------------------------
# (d) output size <= target; one target -> one pooling for all three panels
# --------------------------------------------------------------------------
class SizeAndSamePoolingForAllPanels(unittest.TestCase):
    def test_output_never_exceeds_the_target(self):
        for n in (1, 2, 7, 179, 600, 6361):
            for t in (1, 2, 3, 50, 179, 347, 1000, 7000):
                b = block(n, t)
                out = pool(np.ones((1, n, n if n < 700 else 9)), t, n_rows=t)
                with self.subTest(n=n, t=t):
                    self.assertLessEqual(_ceil(n, b), t)
                    self.assertEqual(out.shape[-2], _ceil(n, b))
                    self.assertLessEqual(out.shape[-1], t)

    def test_three_panels_pool_identically(self):
        rng = np.random.default_rng(7)
        data = rng.uniform(size=(3, 20, 6361))
        res = 0.3 * rng.uniform(size=data.shape)
        panels = {"data": data, "tmpl": data - res, "res": res}
        for k in panels:
            panels[k] = np.abs(panels[k])
            panels[k][..., 2457] = 50.0                # the same merger column in each
        out = {k: pool(v, 341) for k, v in panels.items()}
        shapes = {v.shape for v in out.values()}
        self.assertEqual(len(shapes), 1)
        cols = {k: set(np.nonzero(v[0, 0] == 50.0)[0]) for k, v in out.items()}
        self.assertEqual(cols["data"], cols["tmpl"])
        self.assertEqual(cols["data"], cols["res"])
        self.assertEqual(cols["data"], {2457 // block(6361, 341)})
        # the block map itself is content-free: pooling the column index
        # gives every panel the same last-member-per-block
        idx = np.broadcast_to(np.arange(6361.0), (1, 1, 6361))
        np.testing.assert_array_equal(pool(idx, 341)[0, 0],
                                      np.minimum(np.arange(1, 341 + 1) * block(6361, 341),
                                                 6361)[:_ceil(6361, block(6361, 341))] - 1)


# --------------------------------------------------------------------------
# the rendered figure: the spike is ON SCREEN in all nine panels
# --------------------------------------------------------------------------
NT_R, NF_R, DT, DF = 2400, 20, 3600.0, 1.0 / 7200.0
DPI = 88
BRIGHT = (180, 120)          # viridis at the top of the scale: R high, B low


def _arrays(col):
    a = np.zeros((3, NF_R, NT_R))
    a[:, 0, :] = 1.0          # a bright GB-like track: pins the row scale at 1
    a[..., col] = 1.0         # the merger column
    return a


def _render(ns, col):
    a = _arrays(col)
    src = [(k, (lambda a=a: a)) for k in ("data", "tmpl", "res")]
    fig, info = ns["draw_wdm_tf_panels"](
        src, layer_dt=DT, layer_df=DF, t_edge0=216000.0, f_edge0=0.0,
        nt=NT_R, nf=NF_R, col_titles=["data", "all fitted signals", "residual"],
        dpi=DPI)
    fig.set_dpi(DPI)
    fig.canvas.draw()
    return fig, info, np.asarray(fig.canvas.buffer_rgba())


def _spike_on_screen(fig, buf, a_, x_day):
    """Is there a bright pixel column within 2 px of ``x_day``, through the
    middle of the panel (clear of the bottom track and the edge ticks)?"""
    H = buf.shape[0]
    bb = a_.get_window_extent()
    xp = a_.transData.transform((x_day, 0.0))[0]
    rows = slice(int(H - (bb.y0 + 0.75 * bb.height)), int(H - (bb.y0 + 0.30 * bb.height)))
    for x in range(int(xp) - 2, int(xp) + 3):
        px = buf[rows, x, :3].astype(int)
        if np.mean((px[:, 0] > BRIGHT[0]) & (px[:, 2] < BRIGHT[1])) > 0.8:
            return True
    return False


class RenderedSpikeInEveryPanel(unittest.TestCase):
    def test_merger_is_on_screen_in_all_nine_panels_at_every_phase(self):
        fig, info, _ = _render(NS, 0)
        plt.close(fig)
        bt = info["bt"]
        self.assertGreater(bt, 1)              # the regime that aliased: > 1 column/pixel
        self.assertLessEqual(info["pooled_shape"][-1], info["n_cols"])
        for phase in sorted({0, 1, bt // 2, bt - 1}):
            col = 150 * bt + phase
            fig, info, buf = _render(NS, col)
            x_day = (col // bt + 0.5) * bt * DT / 86400.0
            panels = [a_ for a_ in fig.axes if a_.get_label() != "<colorbar>"]
            self.assertEqual(len(panels), 9)
            for i, a_ in enumerate(panels):
                with self.subTest(phase=phase, panel=i):
                    self.assertTrue(_spike_on_screen(fig, buf, a_, x_day))
            plt.close(fig)

    def test_negative_control_unpooled_full_resolution_aliases(self):
        """The same figure WITHOUT pooling (full-resolution image, ~7 columns
        per pixel) loses the merger in some panel at some phase -- the 9mo
        symptom. Proves the check above can see aliasing at all."""
        ns = _compile()
        ns["pool_tf_for_display"] = lambda m, *a, **k: m
        ns["tf_pool_block"] = lambda n, t: 1
        missing = 0
        for phase in range(8):
            col = 1200 + phase
            fig, _, buf = _render(ns, col)
            x_day = (col + 0.5) * DT / 86400.0
            for a_ in fig.axes:
                if a_.get_label() != "<colorbar>":
                    missing += not _spike_on_screen(fig, buf, a_, x_day)
            plt.close(fig)
        self.assertGreater(missing, 0)

    def test_display_grid_is_at_most_one_cell_per_saved_pixel(self):
        fig, info, _ = _render(NS, 10)
        for a_ in fig.axes:
            if a_.get_label() == "<colorbar>":
                continue
            bb = a_.get_window_extent()
            self.assertLessEqual(info["pooled_shape"][-1], bb.width)
            self.assertLessEqual(info["pooled_shape"][-2], bb.height)
        plt.close(fig)


# --------------------------------------------------------------------------
# MBHB merger ticks: t_plunge -> axis days
# --------------------------------------------------------------------------
class MergerTimes(unittest.TestCase):
    f = staticmethod(NS["mbh_merger_plot_days"])

    def test_the_9mo_row_lands_on_the_diagnosed_plot_times(self):
        # medians of chain/mbh[row, 0, 0, :, :, 10] on the 9mo store (days),
        # axis origin = (data_t0 - epoch = 850.5 s) + the 60 x 3600 s crop
        tp = np.array([173.272, 104.74, 263.767, 243.812, 111.387, 91.968]) * 86400.0
        row = np.zeros((4, 6, 11))
        row[..., 10] = tp
        got = self.f(row, np.ones((4, 6), bool), 850.5 + 216000.0)
        np.testing.assert_allclose(got, [89.46, 102.23, 108.88, 170.76, 241.30, 261.26],
                                   atol=0.01)

    def test_dead_walkers_and_dead_leaves_are_skipped(self):
        row = np.zeros((3, 3, 11))
        row[:, 0, 10] = [86400.0, 86400.0, 9e9]      # walker 2 dead, junk value
        row[:, 1, 10] = 2 * 86400.0
        row[:, 2, 10] = 5 * 86400.0                  # leaf 2: no live walker
        inds = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 0]], bool)
        self.assertEqual(self.f(row, inds, 0.0), [1.0, 2.0])

    def test_empty_or_foreign_shapes_give_no_ticks(self):
        self.assertEqual(self.f(np.zeros((4, 0, 11)), None, 0.0), [])
        self.assertEqual(self.f(np.zeros((4, 3, 5)), None, 0.0), [])


# --------------------------------------------------------------------------
# (e) wiring: the page's TF figure goes through the pooled path
# --------------------------------------------------------------------------
class Wiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(generator_path(), encoding="utf-8") as fh:
            cls.src = fh.read()
        i = cls.src.index("# ===================== WDM figure")
        j = cls.src.index("# ================= F1: the residual spectrum")
        cls.block = cls.src[i:j]
        k = cls.src.index("def draw_wdm_tf_panels(")
        cls.draw = cls.src[k:cls.src.index("\ndef ", k + 1)]

    def test_the_page_draws_the_tf_figure_through_the_pooled_panels(self):
        self.assertIn("fig, _tfi = draw_wdm_tf_panels(", self.block)
        self.assertNotIn(".imshow(", self.block)
        self.assertIsNone(re.search(r"DEC\)\.max\(axis=-1\)|\[\.\.\., *::", self.src))

    def test_the_grid_is_measured_at_the_dpi_the_png_is_saved_at(self):
        self.assertIn("mergers_d=_merge_d, merger_src=_merge_src, dpi=_WDM_DPI)", self.block)
        self.assertIn('fig_b64(fig, "dtr_wdm", dpi=_WDM_DPI)', self.block)

    def test_every_panel_is_max_pooled_to_one_shared_grid_and_drawn_nearest(self):
        self.assertIn("n_cols, n_rows = tf_display_grid(ax.ravel(), dpi)", self.draw)
        self.assertEqual(self.draw.count("pool_tf_for_display("), 1)
        self.assertIn('pool_tf_for_display(m, n_cols, n_rows, how="max")', self.draw)
        self.assertIn('interpolation="nearest"', self.draw)
        self.assertIn("np.nanpercentile(pooled[keys[0]][r], 99.5)", self.draw)

    def test_merger_ticks_come_from_the_mbh_cold_chain(self):
        self.assertIn('_row("chain/mbh", (NIT - 1, 0, 0))', self.block)
        self.assertIn("t_offset=(_t0_data - MOJITO_REFERENCE_TIME) + float(_te[0])",
                      self.block)


if __name__ == "__main__":
    unittest.main()
