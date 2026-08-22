"""GB leaf-cap CELL grid (user design 2026-08-15).

Sub-band widths are set by what the likelihood engine can run concurrently;
the scale at which two GB sources get confused is the posterior width, which
is much narrower. So the LEAF CAPS -- and only the leaf caps -- move onto a
finer grid: every sub-band is split into ``cap_divisor`` equal cap cells and
the caps are enforced there. Scheduling / units / buffers / tempering / band
shutoff all stay on the band grid.

These tests use light fakes in the style of ``test_rj_flip_fraction.py``: the
cap machinery is pure index arithmetic on ``band_sorter``-shaped arrays, so
none of it needs a built move, an ACA, or a backend.
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _compact_index_ranges,
)
from lisatools.globalfit.state import (
    GBState,
    cap_divisor_from_edges,
    ensure_cap_cell_fields,
    ensure_leaf_cap_fields,
    make_cap_edges,
)

# 4 bands, 1 mHz wide, 5 -> 9 mHz (Hz)
BAND_EDGES = np.array([5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
NUM_BANDS = len(BAND_EDGES) - 1


def _move(cap_divisor, ntemps=1, nwalkers=1, num_bands=NUM_BANDS,
          band_edges=BAND_EDGES):
    """A GB move with ONLY the cap-grid attributes filled in."""
    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.name = "test"
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"
    m.num_bands = num_bands
    m.band_edges = np.asarray(band_edges, dtype=float)
    m.ntemps = ntemps
    m.nwalkers = nwalkers
    m.cap_divisor = max(1, int(cap_divisor))
    # the fake predates the staggered grid (642ee91f); the cell lookup reads
    # this unconditionally, so an unstaggered fake must set it explicitly
    m.cap_stagger = False
    m.num_cap_cells = num_bands * m.cap_divisor
    m.cap_edges = np.asarray(make_cap_edges(band_edges, m.cap_divisor))
    m._cap_band_lo = m.band_edges[:-1]
    m._cap_band_step = (
        (m.band_edges[1:] - m.band_edges[:-1]) / m.cap_divisor
    )
    m._cap_leaf_cap = None
    m._band_leaf_cap = None
    return m


def _sorter(f0_hz, alive, temp_inds=None, walker_inds=None,
            band_edges=BAND_EDGES):
    """A fake BandSorter carrying just what the cap machinery reads."""
    f0_hz = np.asarray(f0_hz, dtype=float)
    n = f0_hz.size
    band_inds = np.searchsorted(np.asarray(band_edges), f0_hz, side="right") - 1
    return SimpleNamespace(
        num_sources=n,
        freqs=f0_hz,
        band_inds=band_inds,
        inds=np.asarray(alive, dtype=bool),
        temp_inds=(np.zeros(n, dtype=np.int64) if temp_inds is None
                   else np.asarray(temp_inds, dtype=np.int64)),
        walker_inds=(np.zeros(n, dtype=np.int64) if walker_inds is None
                     else np.asarray(walker_inds, dtype=np.int64)),
    )


class CapEdgeConstructionTest(unittest.TestCase):
    def test_divisor_one_is_the_band_grid(self):
        ce = make_cap_edges(BAND_EDGES, 1)
        np.testing.assert_array_equal(ce, BAND_EDGES)
        self.assertIsNot(ce, BAND_EDGES)  # a copy, never the same object

    def test_containment_every_cell_inside_one_band(self):
        for k in (2, 4, 8):
            ce = make_cap_edges(BAND_EDGES, k)
            self.assertEqual(len(ce) - 1, NUM_BANDS * k)
            # every k-th cap edge IS a band edge (exactly)
            np.testing.assert_array_equal(ce[::k], BAND_EDGES)
            self.assertTrue(np.all(np.diff(ce) > 0))
            self.assertEqual(cap_divisor_from_edges(BAND_EDGES, ce), k)

    def test_non_uniform_bands(self):
        # the free-floating get_n grid is NOT uniform: each band still splits
        # into k equal pieces of ITS OWN width
        be = np.array([1e-3, 1.5e-3, 4.0e-3, 4.2e-3])
        ce = make_cap_edges(be, 4)
        np.testing.assert_allclose(ce[::4], be, rtol=0, atol=1e-18)
        widths = np.diff(ce).reshape(3, 4)
        for row, bw in zip(widths, np.diff(be)):
            np.testing.assert_allclose(row, bw / 4.0)

    def test_divisor_inference_rejects_non_refinement(self):
        with self.assertRaises(ValueError):
            cap_divisor_from_edges(BAND_EDGES, np.linspace(5e-3, 9e-3, 7))


class CapCellIndexTest(unittest.TestCase):
    def test_divisor_one_returns_band_inds_object(self):
        m = _move(1)
        s = _sorter([5.4e-3, 7.6e-3], [True, True])
        out = m._sorter_cap_cells(s)
        self.assertIs(out, s.band_inds)  # bit-identity: the same array

    def test_cell_assignment(self):
        m = _move(4)
        # band 0 = [5, 6) mHz, cells 0.25 mHz wide -> 0,1,2,3
        f0 = np.array([5.05e-3, 5.30e-3, 5.60e-3, 5.90e-3, 7.10e-3])
        s = _sorter(f0, np.ones(5, bool))
        np.testing.assert_array_equal(s.band_inds, [0, 0, 0, 0, 2])
        np.testing.assert_array_equal(
            m._sorter_cap_cells(s), [0, 1, 2, 3, 8]
        )

    def test_upper_edge_clips_into_last_cell(self):
        m = _move(4)
        # exactly ON a band's upper edge (band 3's top) -> last cell of band 3
        s = _sorter([9e-3 - 1e-18], [True])
        self.assertEqual(int(m._sorter_cap_cells(s)[0]), 15)


class OccupancyAndCapGateTest(unittest.TestCase):
    """Divisor 4: cells of one band do not block each other; one cell does."""

    def _census(self, m, s, cap):
        cap_inds = m._sorter_cap_cells(s)
        flat, counts = m._cap_cell_counts(s, cap_inds)
        return cap_inds, flat, counts, np.asarray(cap)

    def test_different_cells_of_one_band_do_not_block(self):
        m = _move(4)
        # two alive sources in band 0, cells 0 and 3; one dead row in band 0
        f0 = np.array([5.05e-3, 5.90e-3, 5.40e-3])
        s = _sorter(f0, [True, True, False])
        cap = np.ones(m.num_cap_cells, dtype=int)
        cap_inds, flat, counts, cap = self._census(m, s, cap)
        at_cap = m._cap_at_cap_mask(s, counts, cap, flat, cap_inds)
        # both alive rows sit in a full cell...
        self.assertTrue(bool(at_cap[0]) and bool(at_cap[1]))
        # ...but the dead row is NOT blocked: cells 1 and 2 are still free,
        # so a birth somewhere in band 0 is possible.
        self.assertFalse(bool(at_cap[2]))

    def test_band_saturated_blocks_the_dead_row(self):
        m = _move(4)
        # fill all four cells of band 0 at cap 1
        f0 = np.array([5.05e-3, 5.30e-3, 5.60e-3, 5.90e-3, 5.5e-3])
        s = _sorter(f0, [True, True, True, True, False])
        cap = np.ones(m.num_cap_cells, dtype=int)
        cap_inds, flat, counts, cap = self._census(m, s, cap)
        at_cap = m._cap_at_cap_mask(s, counts, cap, flat, cap_inds)
        self.assertTrue(bool(at_cap[4]))  # dead row blocked: band saturated

    def test_two_sources_in_one_cell_hit_the_cap(self):
        m = _move(4)
        f0 = np.array([5.05e-3, 5.10e-3])  # same cell 0
        s = _sorter(f0, [True, True])
        cap = np.full(m.num_cap_cells, 2, dtype=int)
        cap_inds, flat, counts, cap = self._census(m, s, cap)
        self.assertEqual(int(counts[flat[0]]), 2)
        at_cap = m._cap_at_cap_mask(s, counts, cap, flat, cap_inds)
        self.assertTrue(bool(at_cap.all()))  # cell 0 is at its cap of 2

    def test_divisor_one_matches_the_legacy_per_band_expression(self):
        """Bit-identity: the K=1 mask IS ``counts[flat] >= cap[band]``."""
        rng = np.random.default_rng(0)
        f0 = rng.uniform(5e-3, 9e-3, 60)
        alive = rng.random(60) < 0.6
        t = rng.integers(0, 3, 60)
        w = rng.integers(0, 2, 60)
        m = _move(1, ntemps=3, nwalkers=2)
        s = _sorter(f0, alive, t, w)
        cap = rng.integers(1, 4, m.num_cap_cells)
        cap_inds, flat, counts, cap = self._census(m, s, cap)
        legacy_flat = (
            (s.temp_inds * m.nwalkers + s.walker_inds) * m.num_bands
            + s.band_inds
        )
        legacy_counts = np.bincount(
            legacy_flat[s.inds], minlength=3 * 2 * NUM_BANDS
        )
        np.testing.assert_array_equal(flat, legacy_flat)
        np.testing.assert_array_equal(counts, legacy_counts)
        np.testing.assert_array_equal(
            m._cap_at_cap_mask(s, counts, cap, flat, cap_inds),
            legacy_counts[legacy_flat] >= cap[s.band_inds],
        )

    def test_band_saturation_flat_reduces_at_divisor_one(self):
        m1, m4 = _move(1), _move(4)
        counts = np.array([0, 1, 2, 1])
        cap = np.array([1, 1, 1, 1])
        np.testing.assert_array_equal(
            m1._band_saturated_flat(counts, cap), counts >= cap
        )
        # divisor 4, one (t,w): band 0 has cells [1,1,1,1] -> saturated;
        # band 1 has [1,0,1,1] -> not
        counts4 = np.array([1, 1, 1, 1, 1, 0, 1, 1] + [0] * 8)
        cap4 = np.ones(16, dtype=int)
        np.testing.assert_array_equal(
            m4._band_saturated_flat(counts4, cap4),
            [True, False, False, False],
        )


class CountableCentersMaskTest(unittest.TestCase):
    """The countable-centers precompute consumes ``_rj_at_cap_mask``."""

    def test_mask_follows_cells(self):
        m = _move(4)
        f0 = np.array([5.05e-3, 5.30e-3, 5.60e-3, 5.90e-3, 5.5e-3, 7.5e-3])
        s = _sorter(f0, [True, True, True, True, False, False])
        cap_inds = m._sorter_cap_cells(s)
        flat, counts = m._cap_cell_counts(s, cap_inds)
        cap = np.ones(m.num_cap_cells, dtype=int)
        m._rj_at_cap_mask = m._cap_at_cap_mask(s, counts, cap, flat, cap_inds)
        ids = np.arange(6)
        subset = SimpleNamespace(
            inds_main_band_sorter=ids, inds=s.inds,
        )
        countable = subset.inds | ~m._rj_at_cap_mask[subset.inds_main_band_sorter]
        # row 4 (dead, band 0 saturated) drops out; row 5 (dead, band 2
        # empty) stays countable
        np.testing.assert_array_equal(
            countable, [True, True, True, True, False, True]
        )


class CapStateArraysTest(unittest.TestCase):
    def test_divisor_one_uses_the_band_arrays(self):
        m = _move(1)
        bi = {"nwalkers": 2, "num_bands": NUM_BANDS}
        ensure_leaf_cap_fields(bi, NUM_BANDS)
        ensure_cap_cell_fields(bi, m.num_cap_cells)
        # nothing extra allocated -> pre-cap-grid stores resume untouched
        self.assertNotIn("cap_cell_leaf_cap", bi)
        cap, iters, best = m._cap_state_arrays(bi)
        self.assertIs(cap, bi["band_leaf_cap"])
        self.assertIs(iters, bi["band_cap_iters"])
        self.assertIs(best, bi["band_best_ll"])

    def test_divisor_four_allocates_cell_arrays(self):
        m = _move(4)
        bi = {"nwalkers": 2, "num_bands": NUM_BANDS}
        ensure_leaf_cap_fields(bi, NUM_BANDS)
        cap, iters, best = m._cap_state_arrays(bi)
        self.assertEqual(cap.shape, (NUM_BANDS * 4,))
        self.assertEqual(bi["cap_cell_cold_ll"].shape, (2, NUM_BANDS * 4))
        self.assertTrue(np.all(cap == -1))

    def test_band_mirror_is_the_max_over_cells(self):
        m = _move(4)
        bi = {"nwalkers": 2, "num_bands": NUM_BANDS}
        ensure_leaf_cap_fields(bi, NUM_BANDS)
        m._cap_state_arrays(bi)
        bi["cap_cell_leaf_cap"][:] = 1
        bi["cap_cell_leaf_cap"][5] = 3   # band 1, cell 1
        m._mirror_band_leaf_cap(bi)
        np.testing.assert_array_equal(bi["band_leaf_cap"], [1, 3, 1, 1])


class StorageRoundTripTest(unittest.TestCase):
    def test_cap_edges_is_static_and_cells_are_stored(self):
        st = GBState(None)
        st.initialize_band_information(
            2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)),
            cap_edges=make_cap_edges(BAND_EDGES, 4),
        )
        statics = st.static_arrays()
        self.assertIn("cap_edges", statics)
        self.assertEqual(len(statics["cap_edges"]), NUM_BANDS * 4 + 1)
        arrays = st.storage_arrays()
        # statics are NOT per-iteration arrays
        self.assertNotIn("cap_edges", arrays)
        self.assertNotIn("band_edges", arrays)
        for name in ("cap_cell_leaf_cap", "cap_cell_iters",
                     "cap_cell_best_ll", "cap_cell_cold_ll"):
            self.assertIn(name, arrays)
        self.assertEqual(st.storage_attrs()["num_cap_cells"], NUM_BANDS * 4)

    def test_from_stored_round_trip(self):
        st = GBState(None)
        st.initialize_band_information(
            2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)),
            cap_edges=make_cap_edges(BAND_EDGES, 4),
        )
        st.band_info["cap_cell_leaf_cap"][:] = 2
        arrays = {k: np.asarray(v)[None] for k, v in st.storage_arrays().items()}
        back = GBState.from_stored(arrays, statics=st.static_arrays(), attrs={})
        np.testing.assert_array_equal(
            np.asarray(back.band_info["cap_cell_leaf_cap"])[0],
            np.full(NUM_BANDS * 4, 2),
        )
        np.testing.assert_array_equal(
            back.band_info["cap_edges"], st.band_info["cap_edges"]
        )

    def test_backend_template_path_at_divisor_four(self):
        """reset_kwargs -> make_template -> storage_arrays -> from_stored."""
        st = GBState(None)
        st.initialize_band_information(
            2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)),
            cap_edges=make_cap_edges(BAND_EDGES, 4),
        )
        st.initialize_tempered(3, 2, 5, 9)
        rk = st.reset_kwargs
        self.assertIn("cap_edges", rk)
        self.assertEqual(rk["num_cap_cells"], NUM_BANDS * 4)
        tmpl = GBState.make_template(
            2, 3, **{k: v for k, v in rk.items()
                     if k not in ("nwalkers", "ntemps")}
        )
        arrays = {k: np.asarray(v)[None]
                  for k, v in tmpl.storage_arrays().items()}
        self.assertEqual(arrays["cap_cell_leaf_cap"].shape[1:],
                         (NUM_BANDS * 4,))
        self.assertEqual(arrays["cap_cell_cold_ll"].shape[1:],
                         (2, NUM_BANDS * 4))
        back = GBState.from_stored(
            arrays, statics=tmpl.static_arrays(), attrs=tmpl.storage_attrs()
        )
        # the resume path strips the step axis and validates the grid
        back.initialize_band_information(
            2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)),
            cap_edges=make_cap_edges(BAND_EDGES, 4),
        )
        self.assertEqual(back.band_info["num_cap_cells"], NUM_BANDS * 4)
        self.assertEqual(
            back.band_info["cap_cell_leaf_cap"].shape, (NUM_BANDS * 4,)
        )

    def test_from_stored_without_cap_edges_defaults_to_band_grid(self):
        """Stores written before the cap grid existed: divisor 1."""
        st = GBState(None)
        st.initialize_band_information(2, 3, BAND_EDGES,
                                       np.zeros((NUM_BANDS, 3)))
        arrays = {k: np.asarray(v)[None] for k, v in st.storage_arrays().items()}
        back = GBState.from_stored(
            arrays, statics={"band_edges": BAND_EDGES}, attrs={}
        )
        np.testing.assert_array_equal(back.band_info["cap_edges"], BAND_EDGES)


class ResumeGuardTest(unittest.TestCase):
    def _loaded_state(self, cap_edges):
        st = GBState(None)
        st.initialize_band_information(
            2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)), cap_edges=cap_edges
        )
        return st

    def test_matching_grid_resumes(self):
        st = self._loaded_state(make_cap_edges(BAND_EDGES, 4))
        st.initialize_band_information(
            2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)),
            cap_edges=make_cap_edges(BAND_EDGES, 4),
        )  # no raise

    def test_mismatched_divisor_refuses_and_points_at_the_script(self):
        st = self._loaded_state(make_cap_edges(BAND_EDGES, 4))
        with self.assertRaises(ValueError) as ctx:
            st.initialize_band_information(
                2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)),
                cap_edges=make_cap_edges(BAND_EDGES, 8),
            )
        msg = str(ctx.exception)
        self.assertIn("leaf-cap grid mismatch", msg)
        self.assertIn("migrate_gb_cap_grid.py", msg)

    def test_pre_cap_grid_store_refuses_against_a_divisor(self):
        st = self._loaded_state(None)  # -> cap grid == band grid
        with self.assertRaises(ValueError):
            st.initialize_band_information(
                2, 3, BAND_EDGES, np.zeros((NUM_BANDS, 3)),
                cap_edges=make_cap_edges(BAND_EDGES, 8),
            )


class MigrationTest(unittest.TestCase):
    """Inherit semantics of scripts/fstat_proposal/migrate_gb_cap_grid.py."""

    def test_split_to_cells_inherits(self):
        from lisatools.globalfit.moves import gbspecialstretch  # noqa: F401
        import importlib.util
        import pathlib

        here = pathlib.Path(gbspecialstretch.__file__).resolve()
        script = (here.parents[4] / "scripts" / "fstat_proposal"
                  / "migrate_gb_cap_grid.py")
        if not script.exists():  # installed-package layout: skip
            self.skipTest("migration script not present in this layout")
        spec = importlib.util.spec_from_file_location("_mig", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        band_caps = np.array([[1, 2, 3, 1]])          # (step, nb)
        cells = mod.split_to_cells(band_caps, 1, 4)
        self.assertEqual(cells.shape, (1, 16))
        np.testing.assert_array_equal(
            cells[0], [1] * 4 + [2] * 4 + [3] * 4 + [1] * 4
        )
        # never tightens: every child >= its parent's cap
        np.testing.assert_array_equal(
            cells.reshape(1, 4, 4).max(axis=2), band_caps
        )
        np.testing.assert_array_equal(
            cells.reshape(1, 4, 4).min(axis=2), band_caps
        )

    def test_cold_ll_axis(self):
        import importlib.util
        import pathlib
        from lisatools.globalfit.moves import gbspecialstretch
        script = (pathlib.Path(gbspecialstretch.__file__).resolve().parents[4]
                  / "scripts" / "fstat_proposal" / "migrate_gb_cap_grid.py")
        if not script.exists():
            self.skipTest("migration script not present in this layout")
        spec = importlib.util.spec_from_file_location("_mig2", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.CELL_AXIS["cap_cell_cold_ll"], 2)
        self.assertEqual(mod.CELL_AXIS["cap_cell_leaf_cap"], 1)


class LogFormatTest(unittest.TestCase):
    def test_runs_are_collapsed(self):
        self.assertEqual(_compact_index_ranges([]), "[]")
        self.assertEqual(_compact_index_ranges([4]), "[4]")
        self.assertEqual(
            _compact_index_ranges([0, 1, 2, 3, 17, 40, 41]),
            "[0-3, 17, 40-41]",
        )

    def test_long_lists_are_truncated_not_dumped(self):
        # 800 isolated indices would be an 800-number log line
        idx = np.arange(0, 1600, 2)
        out = _compact_index_ranges(idx)
        self.assertLess(len(out), 200)
        self.assertIn("more runs", out)

    def test_contiguous_block_stays_one_run(self):
        out = _compact_index_ranges(np.arange(1232))
        self.assertEqual(out, "[0-1231]")


class CapLLCheckTest(unittest.TestCase):
    """[GB_CAP_LL_CHECK] the cap windows must tile their band."""

    def _move_with_lls(self):
        m = _move(4, nwalkers=2)
        m.name = "t"
        return m

    def test_exact_partition_logs_no_warning(self):
        m = self._move_with_lls()
        cells = np.arange(2 * NUM_BANDS * 4, dtype=float).reshape(2, -1)
        band = cells.reshape(2, NUM_BANDS, 4).sum(axis=2)
        os.environ["GB_CAP_LL_CHECK"] = "1"
        self.addCleanup(os.environ.pop, "GB_CAP_LL_CHECK", None)
        with self.assertLogs(
            "lisatools.globalfit.moves.gbspecialstretch", level="INFO"
        ) as cm:
            m._cap_ll_check(cells, band)
        self.assertTrue(any("GB_CAP_LL_CHECK" in r for r in cm.output))
        self.assertFalse(any(r.startswith("WARNING") for r in cm.output))

    def test_injected_cross_term_is_flagged(self):
        m = self._move_with_lls()
        cells = np.arange(2 * NUM_BANDS * 4, dtype=float).reshape(2, -1)
        band = cells.reshape(2, NUM_BANDS, 4).sum(axis=2)
        band[1, 2] += 50.0  # a band the cells do NOT reproduce
        os.environ["GB_CAP_LL_CHECK"] = "1"
        self.addCleanup(os.environ.pop, "GB_CAP_LL_CHECK", None)
        with self.assertLogs(
            "lisatools.globalfit.moves.gbspecialstretch", level="INFO"
        ) as cm:
            m._cap_ll_check(cells, band)
        warns = [r for r in cm.output if r.startswith("WARNING")]
        self.assertEqual(len(warns), 1)
        self.assertIn("do not tile band 2", warns[0])

    def test_off_by_default(self):
        m = self._move_with_lls()
        os.environ.pop("GB_CAP_LL_CHECK", None)
        cells = np.zeros((2, NUM_BANDS * 4))
        band = np.full((2, NUM_BANDS), 99.0)  # would fail loudly if it ran
        m._cap_ll_check(cells, band)  # no logging, no raise


if __name__ == "__main__":
    unittest.main()


class VGBCapDivisorTest(unittest.TestCase):
    """VGB must stay on the plain band grid (fixed-dimensional: no leaf caps).

    Regression for a FRESH-run crash: ``VGBSettings`` subclasses
    ``GBSettings``, so it inherited ``cap_divisor`` (env ``GB_CAP_DIVISOR``,
    default 8). ``run.py::_branch_cap_edges`` read that 8 and built
    45*8 = 360 cap cells into the VGB state, while
    ``recipe.build_vgb_moves`` initializes on the 45-band grid -- the state
    guard then refused with "leaf-cap grid mismatch ... stores 360 ...
    config builds 45". The two call sites MUST agree.
    """

    def test_vgb_settings_pin_divisor_one_even_when_gb_env_is_set(self):
        import os
        from unittest import mock
        from lisatools.globalfit.stock.erebor.vgb import VGBSettings

        with mock.patch.dict(os.environ, {"GB_CAP_DIVISOR": "8"}):
            self.assertEqual(VGBSettings().cap_divisor, 1)

    def test_branch_cap_edges_agrees_with_the_band_grid_for_vgb(self):
        import os
        from types import SimpleNamespace
        from unittest import mock
        from lisatools.globalfit.run import _branch_cap_edges
        from lisatools.globalfit.stock.erebor.vgb import VGBSettings

        edges = np.linspace(1e-3, 5e-3, 46)  # 45 VGB bands
        with mock.patch.dict(os.environ, {"GB_CAP_DIVISOR": "8"}):
            info = SimpleNamespace(
                band_edges=edges, cap_divisor=VGBSettings().cap_divisor
            )
            cap_edges = _branch_cap_edges(info)
        # identical to the band grid -> what build_vgb_moves initializes with
        np.testing.assert_allclose(cap_edges, edges)
        self.assertEqual(len(cap_edges) - 1, len(edges) - 1)


class OverCapCellTest(unittest.TestCase):
    """USER RULING 2026-08-16: a cell OVER its cap behaves exactly like one AT it.

    In-model moves can walk a source's f0 across a cap-cell boundary
    mid-unit -- occupancy is censused at pick time and not re-checked when
    a proposal shifts frequency -- so a cell can legitimately end up holding
    MORE than its cap. (Measured in production: 6 such cell-walker instances
    out of 29,568, i.e. 0.02%.) The ruling is to treat that identically:
    the gate is ``count >= cap``, never ``== cap``.

    This is not cosmetic. Under an ``== cap`` test an over-full cell would
    read as NOT capped and would happily accept further births, so the one
    cell that had already over-filled is exactly the cell that would keep
    growing -- the opposite of what the cap is for. These tests pin the
    ``>=`` semantics at every gate.
    """

    def _census(self, m, s, cap):
        cap_inds = m._sorter_cap_cells(s)
        flat, counts = m._cap_cell_counts(s, cap_inds)
        return cap_inds, flat, counts, np.asarray(cap)

    def test_over_full_cell_still_blocks_like_a_full_one(self):
        m = _move(4)
        # THREE alive sources in ONE cell of band 0 at cap 1 -> count 3 > cap
        f0 = np.array([5.02e-3, 5.03e-3, 5.04e-3])
        s = _sorter(f0, [True, True, True])
        cap_inds, flat, counts, cap = self._census(
            m, s, np.ones(m.num_cap_cells, dtype=int))
        self.assertEqual(int(counts[flat[0]]), 3)          # genuinely over cap
        self.assertTrue(np.array_equal(cap_inds[:1], cap_inds[1:2]))
        at_cap = m._cap_at_cap_mask(s, counts, cap, flat, cap_inds)
        self.assertTrue(bool(at_cap.all()),
                        "an over-full cell must read as at-cap, not as free")

    def test_over_full_matches_exactly_full(self):
        """count == cap and count > cap must produce the SAME gate result."""
        m = _move(4)
        cap = np.ones(m.num_cap_cells, dtype=int)
        exactly = _sorter(np.array([5.02e-3]), [True])
        over = _sorter(np.array([5.02e-3, 5.03e-3, 5.04e-3]), [True] * 3)
        outs = []
        for s in (exactly, over):
            ci, fl, ct, c = self._census(m, s, cap)
            outs.append(bool(m._cap_at_cap_mask(s, ct, c, fl, ci)[0]))
        self.assertEqual(outs[0], outs[1], "== cap and > cap must agree")
        self.assertTrue(outs[0])

    def test_band_saturation_counts_an_over_full_cell_as_full(self):
        """A band whose cells are over-full is saturated, so dead rows block."""
        m = _move(4)
        nb, k = m.num_bands, m.cap_divisor
        cap = np.ones(nb * k, dtype=int)
        counts = np.zeros(nb * k, dtype=int)
        counts[: k] = np.array([1, 2, 5, 1])      # band 0: one at cap, others OVER
        sat = m._band_saturated_flat(counts, cap)
        self.assertTrue(bool(sat[0]),
                        "every cell at-or-over cap => band saturated")

    def test_one_free_cell_keeps_the_band_unsaturated_even_beside_an_over_full_one(self):
        m = _move(4)
        nb, k = m.num_bands, m.cap_divisor
        cap = np.ones(nb * k, dtype=int)
        counts = np.zeros(nb * k, dtype=int)
        counts[: k] = np.array([9, 9, 9, 0])      # three over-full, one EMPTY
        sat = m._band_saturated_flat(counts, cap)
        self.assertFalse(bool(sat[0]),
                         "a free cell must still admit a birth into the band")


class GhostIncrementGuardTest(unittest.TestCase):
    """The patience clock only runs for cells whose max ll has improved once.

    An EMPTY cap cell has no source to fit better, so its max ll never
    improves -- and under a bare counter it accrued patience every
    iteration and promoted itself on a fixed clock. Measured on the
    3-month production run that made 920 of 1,232 cells increment in
    LOCKSTEP at iteration 10, turning a progressive cap into a wall-clock
    ratchet: loosest exactly when the model is fullest.

    The guard mirrors ``changed_once`` in the PSD max-logL search
    (``psdmove.py::run_move_max_likelihood``), which likewise refuses to
    count a plateau iteration until the chain has moved at all.
    """

    def _drive(self, m, bi, ll_series, min_iters=3, guard=True):
        """Run ``_update_band_leaf_caps`` over a per-iteration ll series.

        ``ll_series[i]`` is the (nwalkers, ncell) cold-chain cell ll at
        iteration i. Everything the method reads other than the cell lls
        is stubbed, so this exercises the GATE and nothing else.
        """
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1" if guard else "0"
        self.addCleanup(os.environ.pop, "GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)
        m.leaf_cap_ll_improve = True
        m.leaf_cap_iter_only = False
        m.leaf_cap_require_occupancy = False
        m.leaf_cap_min_iters = min_iters
        m.leaf_cap_ndim = 8.0
        m.leaf_cap_ll_nsigma = 3.0
        m._band_dof = np.full(m.num_bands, 100.0)
        state = SimpleNamespace(
            sub_states={m.branch_name: SimpleNamespace(band_info=bi)})
        m._work_branch = lambda _s: SimpleNamespace(shape=(1, 1, 10000))
        m._band_residual_lls = lambda _aca: np.zeros(
            (bi["nwalkers"], m.num_bands))
        caps = []
        for lls in ll_series:
            arr = np.asarray(lls, dtype=float)
            m._cap_cell_lls = lambda *_a, _v=arr: (
                _v, np.full(m.num_cap_cells, 100.0))
            m._update_band_leaf_caps(SimpleNamespace(
                analysis_container_arr=None), state, None)
            caps.append(bi["cap_cell_leaf_cap"].copy())
        return caps

    def _setup(self, cap_divisor=4):
        m = _move(cap_divisor)
        m.branch_name = "gb"
        bi = {"nwalkers": 1, "num_bands": NUM_BANDS}
        ensure_leaf_cap_fields(bi, NUM_BANDS)
        m._cap_state_arrays(bi)
        bi["cap_cell_leaf_cap"][:] = 1
        bi["cap_cell_iters"][:] = 0
        bi["cap_cell_best_ll"][:] = -np.inf
        return m, bi

    def test_a_never_improving_cell_never_increments(self):
        """The ghost increment: a flat cell keeps its cap forever."""
        m, bi = self._setup()
        n = m.num_cap_cells
        # Ten iterations of a cell ll that only jitters -- far below the
        # D/2 = 4.0 improvement threshold.
        rng = np.random.default_rng(0)
        series = [(-100.0 + 1e-3 * rng.standard_normal(n))[None, :]
                  for _ in range(10)]
        caps = self._drive(m, bi, series)
        self.assertTrue(
            np.all(caps[-1] == 1),
            f"flat cells must stay at cap 1, got {np.unique(caps[-1])}")

    def test_an_improving_cell_still_ramps(self):
        """The guard must not disable the annealing it protects."""
        m, bi = self._setup()
        n = m.num_cap_cells
        series = [np.full((1, n), -100.0)]          # arm best (finite)
        series.append(np.full((1, n), -50.0))       # a real improvement
        series += [np.full((1, n), -50.0)] * 8      # then a genuine plateau
        caps = self._drive(m, bi, series, min_iters=3)
        self.assertGreater(
            int(caps[-1].max()), 1,
            "a cell that improved once must still ramp on its plateau")

    def test_only_the_improving_cell_ramps(self):
        """Cell 0 improves, the rest are flat: only cell 0 promotes."""
        m, bi = self._setup()
        n = m.num_cap_cells
        base = np.full((1, n), -100.0)
        series = [base.copy()]
        step = base.copy(); step[0, 0] = -50.0      # cell 0 alone improves
        series.append(step)
        series += [step.copy() for _ in range(8)]
        caps = self._drive(m, bi, series, min_iters=3)
        self.assertGreater(int(caps[-1][0]), 1)
        self.assertTrue(
            np.all(caps[-1][1:] == 1),
            "flat neighbours must not ride cell 0's clock")

    def test_first_observation_is_not_an_improvement(self):
        """best starts at -inf, so iteration 0 beats it trivially.

        Without the ``isfinite(best)`` guard every cell would latch
        'improved once' on its very first update and the guard would be a
        no-op -- exactly the bug the PSD idiom avoids with
        ``not np.isinf(max_logl)``.
        """
        m, bi = self._setup()
        n = m.num_cap_cells
        series = [np.full((1, n), -100.0)] * 10     # never changes at all
        self._drive(m, bi, series)
        self.assertFalse(
            bool(getattr(m, "_cap_ll_improved_once").any()),
            "a first observation against -inf is not evidence of improvement")

    def test_guard_is_OFF_by_default_so_the_ratchet_is_unchanged(self):
        """Default must reproduce the pre-guard behaviour exactly.

        User ruling 2026-08-16: hold at GB_CAP_DIVISOR=8 with no guard.
        The guard only makes sense shipped WITH a finer cell grid -- alone
        at K=8 it would re-impose the 24.5% structural exclusion.
        """
        m, bi = self._setup()
        n = m.num_cap_cells
        rng = np.random.default_rng(0)
        series = [(-100.0 + 1e-3 * rng.standard_normal(n))[None, :]
                  for _ in range(10)]
        caps = self._drive(m, bi, series, min_iters=3, guard=False)
        self.assertGreater(
            int(caps[-1].max()), 1,
            "with the guard OFF, flat cells must still ratchet as before")
