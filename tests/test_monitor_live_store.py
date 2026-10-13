"""The monitor page against a LIVE store: the MBHB dropdown and a stale handle.

2026-10-13, 9mo production, a hand build on the live run directory
(``python -m lisatools.globalfit.monitor RUN_DIR --cold-keep 20 --add-info ...
--build-truth``, 973 s, dev 9ef0ebd8, job 759 saving underneath) lost:

* every MBHB panel. Notes: ``chain/mbh: torn in the live store, recovered from
  the run's backup copy, which holds 197 iteration(s) vs 199 in the main
  file`` and then, six times, ``mbh leaf k PNG render failed:
  IndexError('boolean index did not match indexed array along axis 0; size of
  axis is 197 but size of corresponding boolean axis is 199')``: the chain
  came from the lagging backup, ``inds/mbh`` from the live file;
* the at-a-glance "Other Sources" table ("No MBHB / EMRI / SOBHB branch in
  this store") and its overlap ("No GB coordinates to match at the last
  row"): every chain row it read was SKIPPED.

The common cause is a stale handle. HDF5 caches a file's metadata when it is
opened and a reader without SWMR never refreshes it; the run rewrote and
relocated chunks every save, so reads 10-15 minutes into the build hit freed
space. A second ``h5py.File`` on the same path in the same process SHARES that
cache, which is why the glance block, opening its own handle, failed too.

The generator is a script, so these tests exec its source-dropdown block and
compile its helpers by name against small synthetic stores, the way
``test_monitor_torn_rows.py`` and ``test_monitor_band_ll.py`` do.
"""
import base64
import io
import json
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

NW = 4
LEAVES = {"mbh": (6, 11), "emri": (1, 12), "sobbh": (1, 11)}   # the run's MBH count
CAP = 40


def _src():
    with open(generator_path(), encoding="utf-8") as fh:
        return fh.read()


def _dropdown_block():
    src = _src()
    return src[src.index('SRC_PANEL_JSON = {"mbh"'):
               src.index("# ============================ HTML")]


def _make_store(path, nit, seed=0, it_attr=None):
    """A store with the three fixed-dimension branches, ``nit`` rows written of
    ``CAP``, chunked 20 rows deep on the iteration axis like the live store."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = nit if it_attr is None else it_attr
        for b, (nl, nd) in LEAVES.items():
            ch = np.zeros((CAP, 1, 1, NW, nl, nd))
            ch[:nit] = 1.0 + 0.01 * rng.standard_normal((nit, 1, 1, NW, nl, nd))
            g.create_dataset(f"chain/{b}", data=ch, chunks=(20, 1, 1, 1, min(nl, 2), 3),
                             compression="gzip")
            ind = np.zeros((CAP, 1, 1, NW, nl), bool)
            ind[:nit] = True
            g.create_dataset(f"inds/{b}", data=ind, compression="gzip")
            hh = np.full((CAP, NW, nl), np.nan)
            hh[:nit] = 400.0
            g.create_dataset(f"sub_backend/{b}/h_h", data=hh)
            g.create_dataset(f"sub_backend/{b}/d_h", data=hh.copy())
        g.create_group("sub_backend/gb")


def _tear(path, name, coord):
    """Overwrite one compressed chunk in place, the way a racing writer does."""
    with h5py.File(path, "r") as f:
        info = f[f"global_fit/{name}"].id.get_chunk_info_by_coord(coord)
    with open(path, "r+b") as fh:
        fh.seek(info.byte_offset)
        fh.write(b"\xa5" * info.size)


def _ns(store, nit, run_dir):
    """The names the dropdown block reads, with the generator's own helpers."""
    f = h5py.File(store, "r")
    ns = {"np": np, "plt": plt, "os": os, "h5py": h5py, "io": io, "json": json,
          "base64": base64, "_shrink_png": shrink_png, "h5path": store, "f": f,
          "g": f["global_fit"], "sub": f["global_fit/sub_backend"], "rg": None,
          "_gb_inds_ds": None, "NIT": nit, "RUN_DIR": run_dir, "_BACKUP_G": None,
          "_TORN_ROWS": [], "MISSING": [], "MOJITO_CAT_DIR": run_dir,
          "_SRC_IDS": {}, "_SRC_FULL_PE_START": None,
          "CYAN": "#4FD8EB", "AMBER": "#F5A623", "GREEN": "#58C48A",
          "RED": "#E5484D", "VIOLET": "#9B7BFF", "FG": "#B8C6D4", "DIM": "#67788A"}
    _generator_functions(
        ns, names=("_safe", "_backup_group", "_row", "_close_store", "_reopen_store",
                   "_fresh_store", "_src_load_catalogue", "_src_cat_row", "_src_pool_window",
                   "_src_make_leaf_png", "_src_cat_to_sampling_mbh",
                   "_src_cat_to_sampling_emri", "_src_cat_to_sampling_sobbh"),
        consts=("_SRC_BRANCH_SPECS",))
    ns["_SRC_CAT_FNS"] = {b: ns[f"_src_cat_to_sampling_{b}"] for b in LEAVES}
    return ns


def _close(ns):
    for k in ("f", "_BACKUP_G"):
        h = ns.get(k)
        h = getattr(h, "file", h)
        try:
            if h is not None and h is not False and h.id.valid:
                h.close()
        except Exception:
            pass


class SourceDropdownTest(unittest.TestCase):
    """Six MBH leaves in, six MBHB panels out -- whatever the store did."""

    NIT = 30

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = os.path.join(self.tmp, "gf_prod_9mo_testing.h5")
        self.addCleanup(plt.close, "all")

    def _run(self, ns):
        self.addCleanup(_close, ns)
        exec(_dropdown_block(), ns)                         # noqa: S102
        return ns

    def _blob(self, ns, b):
        return json.loads(ns["SRC_PANEL_JSON"][b])

    def test_a_healthy_store_renders_every_mbh_leaf(self):
        _make_store(self.store, self.NIT)
        ns = self._run(_ns(self.store, self.NIT, self.tmp))
        blob = self._blob(ns, "mbh")
        self.assertEqual(len(blob["src"]), 6)
        self.assertTrue(all(base64.b64decode(s["png"]).startswith(b"\x89PNG")
                            for s in blob["src"]))
        self.assertEqual(ns["SRC_PANEL_META"]["mbh"]["n_fail"], 0)
        self.assertIn("6 of 6 leaves rendered", ns["SRC_PANEL_COUNT_TXT"]["mbh"])

    def test_a_chain_from_the_lagging_backup_still_renders_every_leaf(self):
        # The cluster case: chain/mbh torn in the live store, the backup copy
        # two saves behind (its iteration attr NIT - 2), inds/mbh read live.
        _make_store(self.store, self.NIT)
        backup = self.store[:-3] + "_running_backup_copy.h5"
        shutil.copyfile(self.store, backup)
        with h5py.File(backup, "r+") as fb:
            fb["global_fit"].attrs["iteration"] = self.NIT - 2
        _tear(self.store, "chain/mbh", (20, 0, 0, 0, 0, 0))
        ns = self._run(_ns(self.store, self.NIT, self.tmp))
        blob = self._blob(ns, "mbh")
        fails = [m for m in ns["MISSING"] if "PNG render failed" in m]
        self.assertEqual(fails, [])
        self.assertEqual(len(blob["src"]), 6)
        self.assertTrue(any("recovered from the run's backup copy" in m
                            for m in ns["MISSING"]))
        rows = [m for m in ns["MISSING"] if "different row counts" in m]
        self.assertEqual(len(rows), 1, ns["MISSING"])
        self.assertIn(f"end at row {self.NIT - 3}", rows[0])
        self.assertEqual(ns["SRC_PANEL_META"]["mbh"]["nrows"], self.NIT - 2)
        self.assertIn(f"Rows end at {self.NIT - 3}", ns["SRC_PANEL_COUNT_TXT"]["mbh"])
        # the other branches were read live and are untouched
        self.assertEqual(ns["SRC_PANEL_META"]["emri"]["nrows"], self.NIT)

    def test_the_backup_never_hands_back_rows_past_nit(self):
        # opened late in a build, the backup can be AHEAD of the page
        _make_store(self.store, self.NIT)
        backup = self.store[:-3] + "_running_backup_copy.h5"
        _make_store(backup, self.NIT + 3)
        _tear(self.store, "chain/mbh", (20, 0, 0, 0, 0, 0))
        ns = self._run(_ns(self.store, self.NIT, self.tmp))
        self.assertEqual(ns["SRC_PANEL_META"]["mbh"]["nrows"], self.NIT)
        self.assertEqual(len(self._blob(ns, "mbh")["src"]), 6)

    def test_safe_caps_a_backup_read_at_nit(self):
        # the cap itself, without the dropdown's row alignment behind it
        _make_store(self.store, self.NIT)
        _make_store(self.store[:-3] + "_running_backup_copy.h5", self.NIT + 3)
        _tear(self.store, "chain/mbh", (20, 0, 0, 0, 0, 0))
        ns = _ns(self.store, self.NIT, self.tmp)
        self.addCleanup(_close, ns)
        arr = ns["_safe"](ns["g"], "chain/mbh", label="chain/mbh")
        self.assertEqual(arr.shape[0], self.NIT)
        self.assertIn(f"holds {self.NIT} iteration(s) vs {self.NIT}", ns["MISSING"][0])

    def test_a_store_rewritten_under_an_open_handle_is_read_fresh(self):
        # Open the store (the build's first read), then let the "saver"
        # rewrite it with more rows: every chunk and object header moves. The
        # handle opened first is now stale; the block must re-open.
        _make_store(self.store, self.NIT)
        ns = _ns(self.store, self.NIT + 4, self.tmp)
        ns["g"]["chain/mbh"][: self.NIT]                    # metadata cached
        newer = os.path.join(self.tmp, "newer.h5")
        _make_store(newer, self.NIT + 4, seed=1)
        shutil.copyfile(newer, self.store)
        self._run(ns)
        self.assertEqual(len(self._blob(ns, "mbh")["src"]), 6, ns["MISSING"])
        self.assertEqual(ns["SRC_PANEL_META"]["mbh"]["nrows"], self.NIT + 4)
        with h5py.File(newer, "r") as fn:
            np.testing.assert_array_equal(ns["g"]["chain/mbh"][: self.NIT + 4],
                                          fn["global_fit/chain/mbh"][: self.NIT + 4])


class StaleHandleTest(unittest.TestCase):
    """The HDF5 behaviour the fix rests on, and the helpers that apply it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = os.path.join(self.tmp, "gf_testing.h5")
        self.newer = os.path.join(self.tmp, "newer.h5")
        _make_store(self.store, 10)
        _make_store(self.newer, 14, seed=1)
        with h5py.File(self.newer, "r") as fn:
            self.want = fn["global_fit/chain/emri"][:14]

    def _stale(self):
        f = h5py.File(self.store, "r")
        self.addCleanup(lambda: f.id.valid and f.close())
        f["global_fit/chain/mbh"][:10]
        shutil.copyfile(self.newer, self.store)
        return f

    def test_a_second_handle_in_the_same_process_shares_the_stale_cache(self):
        self._stale()
        # this is the glance block's situation: its own open, same path
        with self.assertRaises(Exception):
            with h5py.File(self.store, "r") as f2:
                np.testing.assert_array_equal(f2["global_fit/chain/emri"][:14], self.want)

    def test_reopen_store_rebinds_every_module_handle_to_a_fresh_file(self):
        f = self._stale()
        ns = {"h5py": h5py, "h5path": self.store, "f": f, "g": f["global_fit"],
              "sub": None, "rg": None, "_gb_inds_ds": None, "_BACKUP_G": None}
        _generator_functions(ns, names=("_close_store", "_reopen_store"))
        ns["_reopen_store"]()
        self.addCleanup(ns["f"].close)
        self.assertFalse(f.id.valid, "the stale handle must be CLOSED, not kept")
        np.testing.assert_array_equal(ns["g"]["chain/emri"][:14], self.want)
        self.assertEqual(ns["sub"].name, "/global_fit/sub_backend")

    def test_a_store_that_will_not_reopen_costs_a_note_not_the_page(self):
        f = h5py.File(self.store, "r")
        ns = {"h5py": h5py, "h5path": os.path.join(self.tmp, "gone.h5"), "f": f,
              "g": f["global_fit"], "sub": None, "rg": None, "_gb_inds_ds": None,
              "_BACKUP_G": None, "MISSING": []}
        _generator_functions(ns, names=("_close_store", "_reopen_store", "_fresh_store"))
        ns["_fresh_store"]("ratchet panels")
        self.assertEqual(len(ns["MISSING"]), 1)
        self.assertIn("ratchet panels: re-opening the store failed", ns["MISSING"][0])

    def test_close_store_also_drops_the_backup_handle(self):
        f = h5py.File(self.store, "r")
        b = h5py.File(self.newer, "r")
        ns = {"f": f, "_BACKUP_G": b["global_fit"]}
        _generator_functions(ns, names=("_close_store",))
        ns["_close_store"]()
        self.assertFalse(f.id.valid)
        self.assertFalse(b.id.valid)
        self.assertIsNone(ns["_BACKUP_G"])


class CallOrderTest(unittest.TestCase):
    """Where the generator re-opens, and that the glance block reads alone."""

    @classmethod
    def setUpClass(cls):
        cls.src = _src()

    def test_the_dropdowns_reopen_before_their_first_read(self):
        blk = _dropdown_block()
        self.assertLess(blk.index('_fresh_store("source dropdowns")'),
                        blk.index('_safe(g, f"chain/{_branch}"'))

    def test_the_late_sections_that_take_their_own_handles_reopen_first(self):
        # the 9mo page lost these to "filter returned failure" / "addr overflow"
        for where, first_read in (("ratchet panels", '_sa = g.get("saved_after")'),
                                  ("ratchet scorecard", '_chg_r = g["chain/gb"]'),
                                  ("RJ-per-band panel", '_pr = sub["gb/band_num_proposed_rj"]')):
            i = self.src.index(f'_fresh_store("{where}")')
            self.assertLess(i, self.src.index(first_read), where)
            self.assertLess(self.src.index(first_read) - i, 4000, where)

    def test_the_store_is_closed_before_the_glance_block_opens_its_own(self):
        i_close = self.src.index("_close_store()\ntry:\n    _G = build_glance(")
        self.assertLess(i_close, self.src.index("_G = build_glance("))

    def test_nothing_reads_the_store_after_the_glance_block(self):
        tail = self.src[self.src.index("_G = build_glance("):]
        for pat in ("_safe(", "_row(", "_opt(", 'g["', "g.get(", 'sub["', 'f["'):
            self.assertNotIn(pat, tail, pat)


if __name__ == "__main__":
    unittest.main()
