"""The monitor's per-sub-band HISTORY figures (``gb_band_evo``, ``gb_band_movers``).

User request 2026-10-13: "examine making the PER-subband evolution of lnL and
counts of GBs clearer over iteration. The current plots we have at the per
sub-band level over iteration are not very clear." Section 5a-1 of the
generator draws, from the per-row read of 5a-0 (``BAND_HIST``), three band x
iteration rasters on one y axis (band lnL minus its final value, GB count minus
the final count, the per-walker RJ valve's shut share) and small multiples of
the sub-bands that moved most, per walker, with the valve-shut rows marked. On
a snapshot extract (leaf frequencies in the last row only) the lnL raster falls
back to the valve's gain ``band_cold_logl_w``, blanking the rows written in the
residual units of before da45aa98 and folding the sub-band pairs that share a
WDM layer window.

The generator is a script, so these tests compile its helpers by name and exec
the two blocks (5a-0's read + line panel, then 5a-1) against small synthetic
stores, the way ``test_monitor_band_ll.py`` does.
"""
import base64
import io
import os
import shutil
import tempfile
import unittest

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lisatools.globalfit.monitor import generator_path  # noqa: E402
from lisatools.globalfit.monitor._png import shrink_png  # noqa: E402
from lisatools.globalfit.monitor._short import _generator_functions  # noqa: E402

EDGES = np.arange(1.0, 10.0) * 1e-3                       # 8 bands of 1 mHz [Hz]
NW, NL = 4, 10
F0_MHZ = np.array([1.5, 1.6, 2.5, 3.5, 4.5, 4.6, 5.5, 6.5, 7.5, 8.5])
#                  band 0    1    2    3    3    4    5    6    7
BIRTH = np.array([0, 10, 0, 5, 0, 20, 15, 0, 30, 0])
DEATH_SLOT, DEATH_ROW = 9, 25                             # band 7 empties at row 25

HELPERS = ("gb_band_ll", "band_evo_rows", "gb_band_counts", "band_history",
           "valve_gain_rows", "valve_residual_rows", "band_rel_final", "band_pool",
           "band_rank_ticks",
           "band_movers", "shut_runs", "band_evo_cmaps")


def _src():
    with open(generator_path(), encoding="utf-8") as fh:
        return fh.read()


def _read_block():
    src = _src()
    return src[src.index("BAND_LL_ZOOM = 100"):src.index("# ---- 5a. CAP-CELL OCCUPANCY")]


def _evo_block():
    src = _src()
    return src[src.index("# ---- 5a-1. PER-SUB-BAND HISTORY"):
               src.index("# ---- 5b. PER-(WALKER, BAND) SEARCH STATE")]


def _alive(nit):
    i = np.arange(nit)[:, None, None]
    a = np.broadcast_to(BIRTH[None, None, :] <= i, (nit, NW, NL)).copy()
    a[DEATH_ROW:, :, DEATH_SLOT] = False
    a[:, 3, 5] = False                                    # walker 3 never gets slot 5
    return a


def _store(path, nit, extract=False, valve=True, grow_to=40):
    """A cold GB chain on 8 one-mHz sub-bands with births, a death, a growing leaf.

    Per live leaf ``d_h - h_h/2 = 10 + 2 s (+ 5 i for slot 4, to row grow_to) + 0.1 w``.
    ``extract``: leaf frequencies and d_h / h_h written in the LAST row only (a
    snapshot extract). ``valve``: the per-walker valve and its gain history.
    """
    alive = _alive(nit)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = nit
        ch = np.zeros((nit, 1, 1, NW, NL, 9))
        ch[..., 1] = F0_MHZ
        if extract:
            ch[:-1] = 0.0
        g["chain/gb"] = ch
        g["inds/gb"] = alive[:, None, None]
        s = g.create_group("sub_backend/gb")
        s["band_edges"] = EDGES
        i = np.arange(nit)[:, None, None]
        w = np.arange(NW)[None, :, None]
        sl = np.arange(NL)[None, None, :]
        gain = 10.0 + 2.0 * sl + 0.1 * w + np.where(sl == 4, 5.0 * np.minimum(i, grow_to), 0.0)
        hh = np.full((nit, NW, NL), 40.0)
        dh = hh / 2 + gain
        if extract:
            hh[:-1] = 0.0
            dh[:-1] = 0.0
        s["h_h"] = hh
        s["d_h"] = dh
        if valve:
            sw = np.zeros((nit, NW, 8), bool)
            sw[30:, 0:2, 1] = True                        # band 1, walkers 0 and 1
            sw[35:45, 2, 5] = True                        # band 5, walker 2, rows 35-44
            sw[41:, 1, 3] = True                          # band 3 (a mover), walker 1
            s["band_rj_shutoff_w"] = sw
            G = np.zeros((nit, NW, 8))
            G[:2] = -np.inf                               # the valve not armed yet
            G[2:5, :, 0:2] = -1400.0                      # residual units (pre-da45aa98)
            r = np.arange(5, nit)[:, None]
            G[5:, :, 0] = 500.0 + r + np.arange(NW)       # bands 0 / 1 share a window
            G[5:, :, 1] = 500.5 + r + np.arange(NW)
            G[5:, :, 4] = 300.0 + 2.0 * r                 # band 4 owns a window
            s["band_cold_logl_w"] = G                     # everything else: blind (0)


class HelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _generator_functions({"np": np}, names=HELPERS)

    def test_rows_are_every_row_then_each_blocks_last(self):
        rows, xe = self.ns["band_evo_rows"](40, 400)
        self.assertEqual(rows, list(range(40)))
        np.testing.assert_array_equal(xe, np.arange(41) - 0.5)
        rows, xe = self.ns["band_evo_rows"](1000, 400)       # blocks of 3
        self.assertEqual(rows[:3], [2, 5, 8])
        self.assertEqual(rows[-1], 999)                       # the newest row, always
        self.assertLessEqual(len(rows), 400)
        self.assertEqual((xe[0], xe[-1]), (-0.5, 999.5))
        self.assertEqual(len(xe), len(rows) + 1)

    def test_counts_bin_live_leaves_and_refuse_a_row_without_coordinates(self):
        f0 = np.array([[1.5e-3, 1.6e-3, 3.5e-3, 8.5e-3]])
        out = self.ns["gb_band_counts"](f0, np.array([[True, True, True, False]]), EDGES)
        np.testing.assert_array_equal(out[0], [2, 0, 1, 0, 0, 0, 0, 0])
        # an extract's row outside its keep window: alive leaves, zero frequencies
        self.assertIsNone(self.ns["gb_band_counts"](np.zeros((1, 4)),
                                                   np.ones((1, 4), bool), EDGES))

    def test_history_gaps_unwritten_and_unreadable_rows(self):
        alive = _alive(4)
        f0 = np.broadcast_to(F0_MHZ, (NW, NL))

        def read(name, sel):
            i = sel[0]
            if i == 1:
                return None                                   # unreadable row
            if name == "chain/gb":
                return f0
            v = np.full((NW, NL), 40.0)
            return np.zeros((NW, NL)) if i == 2 else v       # row 2: d_h unwritten

        h = self.ns["band_history"]([0, 1, 2, 3], read, alive, EDGES)
        self.assertTrue(np.isnan(h["ll"][1]).all() and np.isnan(h["n"][1]).all())
        self.assertTrue(np.isnan(h["ll"][2]).all())          # no lnL ...
        self.assertEqual(float(np.nansum(h["n"][2])), float(alive[2].sum()))   # ... counts
        self.assertEqual(h["ll"][0, 0, 0], 40.0 - 20.0)       # d_h - h_h/2, one leaf
        self.assertEqual(h["n"].dtype, np.float32)
        h2 = self.ns["band_history"]([0, 3], read, alive, EDGES, has_dh=False)
        self.assertTrue(np.isnan(h2["ll"]).all())
        self.assertFalse(h2["has_dh"])

    def test_gain_rows_exclude_residual_units_and_empty_rows(self):
        G = np.zeros((4, 2, 3))
        G[0] = -1400.0                                        # residual term
        G[1] = -np.inf                                        # never armed
        G[2, :, 0] = 5000.0                                   # gain
        out = self.ns["valve_gain_rows"](G)                   # row 3: all zero
        np.testing.assert_array_equal(out, [False, False, True, False])
        out = self.ns["valve_residual_rows"](G)
        np.testing.assert_array_equal(out, [True, False, False, False])

    def test_relative_to_each_walkers_own_final_value(self):
        v = np.full((4, 2, 2), np.nan)
        v[0] = [[1.0, np.nan], [10.0, 3.0]]
        v[1] = [[2.0, 4.0], [12.0, 3.0]]
        v[2] = [[5.0, 4.0], [11.0, np.nan]]                   # walker 1 band 1: empty
        rel, last = self.ns["band_rel_final"](v)              # row 3: no data at all
        self.assertEqual(last, 2)
        np.testing.assert_array_equal(rel[0], [[-4.0, -4.0], [-1.0, 3.0]])
        np.testing.assert_array_equal(rel[2], 0.0)
        self.assertTrue(np.isnan(rel[3]).all())
        rel, _ = self.ns["band_rel_final"](v, empty_as_zero=False)
        self.assertTrue(np.isnan(rel[0, 0, 1]))

    def test_pooling_keeps_a_one_column_transient_and_the_extent(self):
        a = np.zeros((2, 1000))
        a[0, 517] = 300.0
        a[1, 3] = np.nan
        e = np.arange(1001) - 0.5
        p, pe = self.ns["band_pool"](a, e, 100)
        self.assertEqual(p.shape, (2, 100))
        self.assertEqual(p[0, 51], 30.0)                      # diluted, never dropped
        self.assertEqual(p[1, 0], 0.0)                        # NaN ignored
        self.assertEqual((pe[0], pe[-1], len(pe)), (-0.5, 999.5, 101))
        p, pe = self.ns["band_pool"](np.ones((1, 7)), np.arange(8.0), 3)   # ragged
        self.assertEqual(p.shape, (1, 3))
        np.testing.assert_array_equal(pe, [0.0, 3.0, 6.0, 7.0])
        same, _ = self.ns["band_pool"](a, e, 5000)
        np.testing.assert_array_equal(np.nan_to_num(same), np.nan_to_num(a))
        allnan, _ = self.ns["band_pool"](np.full((1, 4), np.nan), np.arange(5.0), 2)
        self.assertTrue(np.isnan(allnan).all())

    def test_rank_ticks_fall_where_each_frequency_lies(self):
        fc = np.array([0.9, 1.1, 1.9, 2.1, 3.0, 9.0])
        pos, lab = self.ns["band_rank_ticks"](fc, marks=(0.5, 1, 2, 3, 20), min_gap=0.0)
        self.assertEqual(lab, ["1", "2", "3"])
        np.testing.assert_allclose(pos, [1.0, 3.0, 4.5])
        pos, lab = self.ns["band_rank_ticks"](fc, marks=(1, 2, 3), min_gap=0.4)
        self.assertEqual(lab, ["1", "3"])                     # 2 is too close to 1

    def test_movers_rank_by_range_then_count(self):
        ll = np.zeros((5, 2, 6))
        ll[2, 0, 0] = 100.0                                   # a birth and a death: net 0
        ll[:, 1, 1] = np.arange(5) * 10.0                     # +40
        ll[:, 0, 2] = 5.0                                     # flat: never picked
        ll[:, 0, 3] = np.arange(5) * 5.0                      # +20: third by lnL
        n = np.zeros((5, 2, 6))
        n[3:, 1, 4] = 1                                       # a count change, lnL flat
        mv = self.ns["band_movers"]
        self.assertEqual(mv(ll, n, k=3, k_count=1), [0, 1, 4])
        self.assertEqual(mv(ll, None, k=1, k_count=1), [0])
        self.assertEqual(mv(None, n, k=4, k_count=1), [4])
        self.assertEqual(mv(np.full((3, 2, 6), np.nan), None), [])

    def test_shut_runs(self):
        self.assertEqual(self.ns["shut_runs"]([0, 1, 1, 0, 1]), [(1, 3), (4, 5)])
        self.assertEqual(self.ns["shut_runs"]([0, 0]), [])

    def test_colormaps_are_neutral_at_zero_and_mark_no_data(self):
        div, seq = self.ns["band_evo_cmaps"]()
        self.assertEqual(matplotlib.colors.to_hex(div(0.5)), "#383835")
        self.assertEqual(matplotlib.colors.to_hex(div(np.nan)), "#0a0e14")
        self.assertEqual(matplotlib.colors.to_hex(seq(1.0)), "#f5a623")


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = os.path.join(self.tmp, "gf_testing.h5")

    def tearDown(self):
        plt.close("all")

    def _run(self, nit=50, max_rows=None, **kw):
        _store(self.store, nit, **kw)
        f = h5py.File(self.store, "r")
        self.addCleanup(f.close)
        ns = {"np": np, "plt": plt, "os": os, "h5py": h5py, "io": io,
              "base64": base64, "g": f["global_fit"], "sub": f["global_fit/sub_backend"],
              "NIT": nit, "RUN_DIR": self.tmp, "_BACKUP_G": None, "_TORN_ROWS": [],
              "MISSING": [], "IMGS": {}, "STAGE_BOUNDS": [(20, "gb_search_1")],
              "_shrink_png": shrink_png, "BG": "#0A0E14", "PANEL": "#10161F",
              "LINE": "#223041", "FG": "#B8C6D4", "DIM": "#67788A"}
        _generator_functions(ns, names=("_row", "_backup_group", "_opt", "mark_stages",
                                        "fig_b64", "walker_color", "_zoom_its"),
                             consts=("WALKER_COLORS", "WALKER_OTHER_COLOR",
                                     "WALKER_MEAN_COLOR"))
        ns["gb_inds"] = ns["_row"]("inds/gb", (slice(0, nit), 0, 0))
        ns["band_edges"] = f["global_fit/sub_backend/gb/band_edges"][:]
        read = _read_block()
        if max_rows is not None:
            read = read.replace("BAND_EVO_MAX_ROWS = 400", f"BAND_EVO_MAX_ROWS = {max_rows}")
        exec(read, ns)                                         # noqa: S102
        exec(_evo_block(), ns)                                 # noqa: S102
        return ns

    def test_full_store_draws_both_figures(self):
        ns = self._run()
        self.assertEqual(ns["MISSING"], [])
        for key in ("gb_band_evo", "gb_band_movers", "gb_band_ll"):
            self.assertTrue(base64.b64decode(ns["IMGS"][key]).startswith(b"\x89PNG"), key)
        self.assertEqual(ns["_be_src"], "dh")
        np.testing.assert_array_equal(ns["_be_bi"], np.arange(8))   # every band held a leaf
        cap = ns["GB_BAND_EVO_CAP"]
        self.assertIn("one GB sub-band that held a cold leaf in some walker in some row "
                      "read (8 of 8)", cap)
        # row 24 vs the final row 49: band 3 still growing (-80), band 6 not yet born
        # (-26), band 7 not yet dead (+28)
        self.assertIn("3 of the 8 sub-bands sit more than 20 nats (walker median) from "
                      "their final band lnL at row 24, 25 rows before the newest row with "
                      "data (49)", cap)
        self.assertIn("2 have a different walker-mean count than at row 24, and at row 49 "
                      "the walkers disagree by 2 or more leaves in 0", cap)
        self.assertIn("The newest row with a shut pair is 49: 3 (walker, band) pairs", cap)
        # occupied at row 49: bands 0-6; gain identically 0 in bands 2, 3, 5, 6, 7
        self.assertIn("Of the 7 sub-bands occupied at row 49, 4 have a valve gain of exactly "
                      "0 in every gain-unit row", cap)
        self.assertEqual(ns["_be_sel"], [0, 2, 3, 4, 6, 7])  # bands 1 and 5 never moved
        self.assertIn("The 6 sub-bands that moved most over the last 50 stored rows (0-49)",
                      ns["GB_BAND_MOVERS_CAP"])
        self.assertIn("4.500 mHz (n = 2/2/2/1)", ns["GB_BAND_MOVERS_CAP"])

    def test_the_movers_strip_marks_the_shut_rows(self):
        ns = self._run()
        strips = [a for a in ns["fig"].axes if a.get_title().startswith("4.500 mHz (band 3)")]
        self.assertEqual(len(strips), 1)
        segs = strips[0].collections[0].get_segments()
        self.assertEqual(len(segs), 1)                       # walker 1, rows 41-49
        np.testing.assert_array_equal(segs[0], [[40.5, 1.0], [49.5, 1.0]])

    def test_the_stage_rule_draws_over_the_raster(self):
        ns = self._run()
        ax = ns["_be_ax"][0]
        mesh = [c for c in ax.collections if isinstance(c, matplotlib.collections.QuadMesh)]
        rules = [ln for ln in ax.lines if list(ln.get_xdata()) == [20, 20]]
        self.assertEqual(len(mesh), 1)
        self.assertEqual(len(rules), 1)
        self.assertLess(mesh[0].get_zorder(), rules[0].get_zorder())

    def test_more_rows_than_pixels_are_pooled(self):
        ns = self._run(nit=700, max_rows=1000)               # every row read, 700 columns
        self.assertEqual(len(ns["_be_rows"]), 700)
        mesh = [c for c in ns["_be_ax"][0].collections
                if isinstance(c, matplotlib.collections.QuadMesh)][0]
        ncols = mesh.get_coordinates().shape[1] - 1
        self.assertLessEqual(ncols, ns["_be_px"])
        self.assertLess(ncols, 700)

    def test_a_long_store_reads_blocks(self):
        ns = self._run(nit=700, grow_to=650)                  # default 400: blocks of 2
        self.assertEqual(ns["_be_rows"][:2], [1, 3])
        self.assertIn("read as 350 blocks of 2 rows drawn at each block's last row",
                      ns["GB_BAND_EVO_CAP"])
        # the line panel reads its own rows out of the shared read: its last row is
        # row 699, where band 3 holds (10 + 8 + 5 * 650 + 0.2) + (20 + 0.2) on walker 2
        self.assertIn("the largest is 3,288 nats at 4.50 mHz", ns["GB_BAND_LL_CAP"])
        self.assertEqual(set(ns["BAND_HIST"]["rows"]),
                         set(ns["gb_band_ll_rows"](700, 100, 300)) | set(ns["_be_rows"]))

    def test_an_extract_falls_back_to_the_valve_gain(self):
        ns = self._run(extract=True)
        # the line panel's own note, nothing from the history figures
        self.assertEqual(len(ns["MISSING"]), 1)
        self.assertIn("GB log-likelihood per sub-band unavailable", ns["MISSING"][0])
        self.assertIn("gb_band_evo", ns["IMGS"])
        self.assertIn("gb_band_movers", ns["IMGS"])
        self.assertEqual(ns["_be_src"], "gain")
        np.testing.assert_array_equal(ns["_be_bi"], [0, 4])   # band 1 shares band 0's window
        # rows 0-4 hold the residual term (the pre-da45aa98 units): blank, not drawn
        mesh = [c for c in ns["_be_ax"][0].collections
                if isinstance(c, matplotlib.collections.QuadMesh)][0]
        mask = np.ma.getmaskarray(mesh.get_array()).reshape(2, 50)
        self.assertTrue(mask[:, :5].all())
        self.assertFalse(mask[:, 5:].any())
        cap = ns["GB_BAND_EVO_CAP"]
        self.assertIn("whose valve gain is non-zero in some row read (2 of 8)", cap)
        self.assertIn("3 of the rows read hold the residual term", cap)
        self.assertIn("(a ~1,400-nat unit step) and 2 carry no valve statistic at all", cap)
        self.assertIn("(1 rows folded)", cap)
        self.assertIn("Panel (b) is empty", cap)
        self.assertIn("holds in 1 of the 50 rows read", cap)
        self.assertIn("2 of the 2 sub-bands sit more than 20 nats (walker median) from "
                      "their final valve gain G at row 24, 25 rows before its newest change "
                      "(49)", cap)
        self.assertIn("Of the 7 sub-bands occupied at row 49, 4 have a valve gain", cap)
        self.assertEqual(ns["_be_sel"], [0, 4])
        self.assertIn("rows 5-49, the last 45 rows up to the valve gain's newest change",
                      ns["GB_BAND_MOVERS_CAP"])

    def test_a_store_without_the_valve(self):
        ns = self._run(valve=False)
        self.assertEqual(ns["MISSING"], [])
        cap = ns["GB_BAND_EVO_CAP"]
        self.assertIn("Panel (c) is empty: this store carries no per-walker RJ valve", cap)
        self.assertNotIn("valve gain of exactly 0", cap)
        self.assertIn("This store carries no per-walker RJ valve", ns["GB_BAND_MOVERS_CAP"])

    def test_an_extract_without_the_valve_is_a_missing_line(self):
        ns = self._run(extract=True, valve=False)
        self.assertNotIn("gb_band_evo", ns["IMGS"])
        self.assertTrue(any(m.startswith("per-sub-band history unavailable: ValueError: the "
                                         "band lnL is in 1 and the band count in 1 of the "
                                         "50 rows read") for m in ns["MISSING"]),
                        ns["MISSING"])

    def test_no_store_kind_warns(self):
        # 2026-10-13: every warning of a monitor build is fixed at its source
        # (test_monitor_build_warnings); empty bands, unwritten rows and all-NaN
        # columns are expected here and must not print one
        import warnings

        for k, kw in enumerate(({}, {"extract": True}, {"valve": False})):
            self.store = os.path.join(self.tmp, f"gf_{k}_testing.h5")   # one file each
            with self.subTest(**kw), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self._run(**kw)
                plt.close("all")
            bad = [f"{w.category.__name__}: {w.message}" for w in caught
                   if issubclass(w.category, (RuntimeWarning, UserWarning))]
            self.assertEqual(bad, [])

    def test_the_page_places_both_figures_above_the_line_panel(self):
        src = _src()
        tpl = src[src.index('html = f"""<title>'):]
        i_evo = tpl.index('{img("gb_band_evo", ')
        i_mov = tpl.index('{img("gb_band_movers", ')
        i_ll = tpl.index('{img("gb_band_ll", ')
        self.assertLess(i_evo, i_mov)
        self.assertLess(i_mov, i_ll)
        self.assertIn("{GB_BAND_EVO_CAP}", tpl)
        self.assertIn("{GB_BAND_MOVERS_CAP}", tpl)
        grp = tpl[tpl.index('<details class="group" id="grp-gbsearch">'):]
        self.assertLess(grp.index('{img("gb_band_evo", '), grp.index("</details>"))


if __name__ == "__main__":
    unittest.main()
