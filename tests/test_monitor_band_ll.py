"""The monitor's GB log-likelihood per sub-band panel (``gb_band_ll``).

User request 2026-10-09: "a plot ... that shows the logl per subband per
walker over time", all walkers drawn alike, sub-bands told apart by
frequency, plus the last 100 iterations. The statistic is the cap-cell
gate's source-attributed one: per (row, cold walker, GB band) the sum over
the band's live leaves of ``d_h - h_h/2``.

The generator is a script, so these tests compile its two helpers by name
and exec the panel's block against a small synthetic store, the same way
``tests/test_monitor_torn_rows.py`` does.
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
from lisatools.globalfit.monitor._short import _generator_functions  # noqa: E402
from lisatools.globalfit.monitor._png import shrink_png  # noqa: E402

EDGES = np.array([1e-3, 2e-3, 4e-3, 8e-3])            # 3 bands [Hz]
NW, NL = 2, 6


def _block():
    with open(generator_path(), encoding="utf-8") as fh:
        src = fh.read()
    return src[src.index("BAND_LL_ZOOM = 100"):
               src.index("# ---- 5a. CAP-CELL OCCUPANCY")]


def _store(path, nit, written_rows=None):
    """Leaves at 1.5 / 3 / 3.5 / 5 mHz; the d_h - h_h/2 gain grows with the row."""
    f0_mhz = np.array([1.5, 3.0, 3.5, 5.0, 6.0, 7.0])
    alive = np.array([[1, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0]], bool)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = nit
        ch = np.zeros((nit, 1, 1, NW, NL, 9))
        ch[..., 1] = f0_mhz
        g["chain/gb"] = ch
        g["inds/gb"] = np.broadcast_to(alive, (nit, 1, 1, NW, NL)).copy()
        s = g.create_group("sub_backend/gb")
        s["band_edges"] = EDGES
        hh = np.zeros((nit, NW, NL))
        dh = np.zeros((nit, NW, NL))
        rows = range(nit) if written_rows is None else written_rows
        for i in rows:
            hh[i] = 10.0 * (1 + i / nit)
            dh[i] = hh[i] + 2.0                 # gain per leaf = h_h/2 + 2
        s["h_h"] = hh
        s["d_h"] = dh


class HelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _generator_functions({"np": np},
                                      names=("gb_band_ll", "gb_band_ll_rows"))

    def test_the_band_sum_counts_live_leaves_only(self):
        d_h = np.array([[12.0, 22.0, 32.0, 99.0]])
        h_h = np.array([[4.0, 8.0, 12.0, 99.0]])
        f0 = np.array([[1.5e-3, 3.0e-3, 3.5e-3, 5.0e-3]])
        alive = np.array([[True, True, True, False]])
        out = self.ns["gb_band_ll"](d_h, h_h, f0, alive, EDGES)
        self.assertEqual(out[0, 0], 12.0 - 2.0)
        self.assertEqual(out[0, 1], (22.0 - 4.0) + (32.0 - 6.0))
        self.assertTrue(np.isnan(out[0, 2]))       # its only leaf is dead

    def test_a_non_finite_leaf_is_ignored(self):
        out = self.ns["gb_band_ll"](np.array([[np.nan, 5.0]]), np.array([[1.0, 2.0]]),
                                    np.array([[1.5e-3, 1.6e-3]]),
                                    np.array([[True, True]]), EDGES)
        self.assertEqual(out[0, 0], 4.0)

    def test_rows_are_the_whole_zoom_plus_a_strided_history(self):
        rows = self.ns["gb_band_ll_rows"](1000, zoom=100, max_rows=300)
        self.assertEqual(rows[-100:], list(range(900, 1000)))
        self.assertLessEqual(len(rows) - 100, 300)
        self.assertEqual(rows[0], 0)
        self.assertEqual(self.ns["gb_band_ll_rows"](40, zoom=100), list(range(40)))


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "gf_testing.h5")

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, nit, **kw):
        _store(self.store, nit, **kw)
        f = h5py.File(self.store, "r")
        self.addCleanup(f.close)
        ns = {"np": np, "plt": plt, "os": os, "h5py": h5py, "io": io,
              "base64": base64, "g": f["global_fit"], "sub": f["global_fit/sub_backend"],
              "NIT": nit, "RUN_DIR": self.tmp, "_BACKUP_G": None, "_TORN_ROWS": [],
              "MISSING": [], "IMGS": {}, "STAGE_BOUNDS": [],
              "_shrink_png": shrink_png}            # fig_b64's PNG shrinker (2026-10-11)
        _generator_functions(ns, names=("_row", "_backup_group", "_opt",
                                        "mark_stages", "fig_b64"))
        ns["gb_inds"] = ns["_row"]("inds/gb", (slice(0, nit), 0, 0))
        ns["band_edges"] = f["global_fit/sub_backend/gb/band_edges"][:]
        exec(_block(), ns)                                  # noqa: S102
        return ns

    def test_the_panel_renders_with_a_caption(self):
        ns = self._run(130)                 # 100-row zoom + a strided history
        self.assertEqual(ns["MISSING"], [])
        self.assertIn("gb_band_ll", ns["IMGS"])
        png = base64.b64decode(ns["IMGS"]["gb_band_ll"])
        self.assertTrue(png.startswith(b"\x89PNG"))
        cap = ns["GB_BAND_LL_CAP"]
        self.assertIn("3 sub-bands are occupied", cap)   # both walkers: 3 bands
        self.assertIn("last 100", cap)

    def test_one_written_row_is_a_missing_line_not_a_panel(self):
        # a short snapshot keeps d_h / h_h for the last row only
        ns = self._run(20, written_rows=[19])
        self.assertNotIn("gb_band_ll", ns["IMGS"])
        self.assertEqual(len(ns["MISSING"]), 1)
        self.assertIn("only 1 of the 20 rows", ns["MISSING"][0])

    def test_the_page_places_the_panel(self):
        with open(generator_path(), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('{img("gb_band_ll", "GB log-likelihood per sub-band")}', src)
        self.assertIn("{GB_BAND_LL_CAP}", src)


if __name__ == "__main__":
    unittest.main()
