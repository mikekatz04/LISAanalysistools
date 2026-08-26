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
    make_cap_edge_extensions,
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
    """The patience clock only runs for ENGAGED, OCCUPIED cells.

    An EMPTY cap cell has no source to fit better, so its statistic never
    changes -- and under a bare counter it accrued patience every
    iteration and promoted itself on a fixed clock. Measured on the
    3-month production run that made 920 of 1,232 cells increment in
    LOCKSTEP at iteration 10, turning a progressive cap into a wall-clock
    ratchet: loosest exactly when the model is fullest.

    User spec 2026-08-26: engagement = any statistic change > 0.1, and a
    source already present at first sight counts (an OCCUPIED cell
    engages immediately); D/2 stays the HOLD test. Empty cells never
    engage, which is what this guard has always been for.
    """

    def _drive(self, m, bi, ll_series, min_iters=3, guard=True, occ=None):
        """Run ``_update_band_leaf_caps`` over a per-iteration ll series.

        ``ll_series[i]`` is the (nwalkers, ncell) cold-chain cell ll at
        iteration i. ``occ`` is the scripted cold occupancy per cell
        (default: all EMPTY). Everything else the method reads is
        stubbed, so this exercises the GATE and nothing else.
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
        _occ = (np.zeros((1, m.num_cap_cells), dtype=int) if occ is None
                else np.atleast_2d(np.asarray(occ, dtype=int)))
        m._cold_occupancy = lambda *_a: _occ
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

    def test_a_never_improving_empty_cell_never_increments(self):
        """The ghost increment: a flat EMPTY cell keeps its cap forever.

        (An OCCUPIED flat cell now legitimately ramps -- user spec
        2026-08-26, ``test_born_converged_cell_still_ramps``.)
        """
        m, bi = self._setup()
        n = m.num_cap_cells
        # Ten iterations of a cell ll that only jitters -- below the 0.1
        # engagement tolerance, with every cell empty.
        rng = np.random.default_rng(0)
        series = [(-100.0 + 1e-3 * rng.standard_normal(n))[None, :]
                  for _ in range(10)]
        caps = self._drive(m, bi, series)
        self.assertTrue(
            np.all(caps[-1] == 1),
            f"flat empty cells must stay at cap 1, got {np.unique(caps[-1])}")

    def test_an_improving_cell_still_ramps(self):
        """The guard must not disable the annealing it protects."""
        m, bi = self._setup()
        n = m.num_cap_cells
        series = [np.full((1, n), -100.0)]          # first sight (occupied)
        series.append(np.full((1, n), -50.0))       # a real improvement
        series += [np.full((1, n), -50.0)] * 8      # then a genuine plateau
        caps = self._drive(m, bi, series, min_iters=3,
                           occ=np.ones((1, n), dtype=int))
        self.assertGreater(
            int(caps[-1].max()), 1,
            "a cell that improved once must still ramp on its plateau")

    def test_only_the_occupied_cell_ramps(self):
        """Cell 0 is occupied and improving; its EMPTY flat neighbours
        must not ride its clock."""
        m, bi = self._setup()
        n = m.num_cap_cells
        base = np.full((1, n), -100.0)
        series = [base.copy()]
        step = base.copy(); step[0, 0] = -50.0      # cell 0 alone improves
        series.append(step)
        series += [step.copy() for _ in range(8)]
        occ = np.zeros((1, n), dtype=int); occ[0, 0] = 1
        caps = self._drive(m, bi, series, min_iters=3, occ=occ)
        self.assertGreater(int(caps[-1][0]), 1)
        self.assertTrue(
            np.all(caps[-1][1:] == 1),
            "empty flat neighbours must not ride cell 0's clock")

    def test_first_observation_of_an_empty_grid_never_engages(self):
        """best starts at -inf, so iteration 0 beats it trivially.

        The trivial beat must not engage an EMPTY grid's clocks (the
        pre-2026-08-26 guard used ``isfinite(best)`` for this; the
        engagement rule now gets it from occupancy: only a cell with a
        source at first sight — or a later >0.1 change — engages).
        """
        m, bi = self._setup()
        n = m.num_cap_cells
        series = [np.full((1, n), -100.0)] * 10     # never changes at all
        self._drive(m, bi, series)
        self.assertFalse(
            bool(getattr(m, "_cap_ll_improved_once").any()),
            "an empty grid's first observation must not engage any clock")

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


# ============================================================================
# OVERLAPPING CAP CELLS (user design 2026-08-23, GB_CAP_OVERLAP_FRAC).
# ============================================================================
# The edge grid never changes; each cell's enforcement SPAN widens so
# adjacent cells share a fraction p of the cell's own width w with each
# neighbour (p = 0.25 -> 1/4-overlap / 1/2-alone / 1/4-overlap; w = s/(1-p),
# x = (w-s)/2 = s/6). A leaf in an overlap zone is a member of BOTH covering
# cells: the census counts it into both, and a location is AT CAP when ANY
# covering cell is at its cap (AND-headroom for births / entries).


def _move_overlap(cap_divisor, overlap, stagger=False, ntemps=1, nwalkers=1,
                  band_edges=BAND_EDGES):
    """A fake GB move carrying the overlap-mode cap-grid attributes."""
    be = np.asarray(band_edges, dtype=float)
    m = _move(cap_divisor, ntemps=ntemps, nwalkers=nwalkers,
              num_bands=len(be) - 1, band_edges=be)
    m.cap_stagger = bool(stagger) and m.cap_divisor > 1
    m.cap_edges = np.asarray(
        make_cap_edges(be, m.cap_divisor, stagger=m.cap_stagger)
    )
    # 2026-08-26 user config: overlap is meaningful at divisor 1 too (cap
    # cells = sub-bands, spans widened across the seams), mirroring the
    # ctor rule.
    m.cap_overlap_frac = float(overlap)
    m._cap_edge_ext = None
    if m.cap_overlap_frac > 0.0:
        m._cap_edge_ext = make_cap_edge_extensions(
            be, m.cap_edges, m.cap_divisor, m.cap_overlap_frac
        )
    return m


class OverlapGeometryTest(unittest.TestCase):
    """The 1/4-1/2-1/4 layout numbers, through the installed constructor."""

    def test_production_v6_numbers_in_bins(self):
        # v6 fine grid analog with df = 1 "Hz" per bin: 135-bin sub-bands,
        # K = 4 -> stride s = 33.75 bins. p = 0.25 must give x = 5.625,
        # width 45, core 22.5, shared zone 11.25 -- the arm-log numbers.
        be = np.arange(0.0, 135.0 * 5 + 1, 135.0)
        ce = make_cap_edges(be, 4, stagger=True)
        x = make_cap_edge_extensions(be, ce, 4, 0.25)
        s = 33.75
        np.testing.assert_allclose(x[1:-1], s / 6.0, rtol=1e-12)
        self.assertAlmostEqual(float(x[1]), 5.625)
        w = s / (1 - 0.25)
        self.assertAlmostEqual(w, 45.0)                    # widened width
        self.assertAlmostEqual(s - 2 * float(x[1]), 22.5)  # exclusive core
        self.assertAlmostEqual(2 * float(x[1]), 11.25)     # shared zone
        # shared zone = w/4, core = w/2: the user's exact layout
        self.assertAlmostEqual(2 * float(x[1]), w / 4.0)
        self.assertAlmostEqual(s - 2 * float(x[1]), w / 2.0)

    def test_end_edges_never_extend(self):
        ce = make_cap_edges(BAND_EDGES, 4, stagger=True)
        x = make_cap_edge_extensions(BAND_EDGES, ce, 4, 0.25)
        self.assertEqual(float(x[0]), 0.0)
        self.assertEqual(float(x[-1]), 0.0)
        self.assertTrue(np.all(x[1:-1] > 0))

    def test_zero_overlap_is_all_zeros(self):
        ce = make_cap_edges(BAND_EDGES, 4)
        x = make_cap_edge_extensions(BAND_EDGES, ce, 4, 0.0)
        np.testing.assert_array_equal(x, np.zeros(len(ce)))

    def test_invalid_fractions_refuse(self):
        ce = make_cap_edges(BAND_EDGES, 4)
        for bad in (-0.1, 0.5, 0.75, 1.0):
            with self.assertRaises(ValueError):
                make_cap_edge_extensions(BAND_EDGES, ce, 4, bad)

    def test_non_uniform_bands_use_the_containing_band_stride(self):
        be = np.array([1e-3, 1.5e-3, 4.0e-3, 4.2e-3])
        ce = make_cap_edges(be, 4)
        x = make_cap_edge_extensions(be, ce, 4, 0.25)
        steps = (be[1:] - be[:-1]) / 4
        # an interior edge inside band 1 gets band 1's stride
        j = 6  # edge between cells 5 and 6, strictly inside band 1
        self.assertAlmostEqual(float(x[j]), steps[1] / 6.0)


class OverlapMembershipTest(unittest.TestCase):
    """core -> 1 cell, overlap zone -> 2 cells, exact boundary values.

    Runs on an exactly-representable binary grid (band width 1.0, stride
    0.25) so the "exactly ON the edge / exactly AT the core boundary"
    assertions test the SEMANTICS (strict vs non-strict comparisons), not
    float round-off; ``x`` is read from the move's own extension array for
    the same reason.
    """

    BE = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def setUp(self):
        self.m = _move_overlap(4, 0.25, band_edges=self.BE)
        self.s = (self.BE[1] - self.BE[0]) / 4  # cap stride = 0.25, exact
        self.x = float(self.m._cap_edge_ext[2])  # the move's own x

    def _members(self, f):
        f = np.atleast_1d(np.asarray(f, dtype=float))
        band = np.clip(
            np.searchsorted(self.BE, f, side="right") - 1,
            0, len(self.BE) - 2,
        ).astype(np.int64)
        return self.m._cap_cell_members(band, f)

    def test_core_point_is_single_membership(self):
        # dead center of cell 1
        f = self.BE[0] + 1.5 * self.s
        p, nb, hn = self._members(f)
        self.assertEqual(int(p[0]), 1)
        self.assertFalse(bool(hn[0]))

    def test_lower_overlap_zone_adds_the_lower_neighbour(self):
        # just above the cell 1/2 edge: covered by cells 2 (primary) and 1
        f = self.BE[0] + 2 * self.s + 0.5 * self.x
        p, nb, hn = self._members(f)
        self.assertEqual(int(p[0]), 2)
        self.assertTrue(bool(hn[0]))
        self.assertEqual(int(nb[0]), 1)

    def test_upper_overlap_zone_adds_the_upper_neighbour(self):
        f = self.BE[0] + 2 * self.s - 0.5 * self.x
        p, nb, hn = self._members(f)
        self.assertEqual(int(p[0]), 1)
        self.assertTrue(bool(hn[0]))
        self.assertEqual(int(nb[0]), 2)

    def test_boundary_values(self):
        e = self.BE[0] + 2 * self.s   # the original cell 1/2 edge
        # ON the shared edge: primary 2, also member of 1 (strict <)
        p, nb, hn = self._members(e)
        self.assertEqual(int(p[0]), 2)
        self.assertTrue(bool(hn[0]))
        self.assertEqual(int(nb[0]), 1)
        # EXACTLY at the core boundary e + x: core point, single membership
        p, nb, hn = self._members(e + self.x)
        self.assertEqual(int(p[0]), 2)
        self.assertFalse(bool(hn[0]))
        # EXACTLY at e - x (upper core boundary of cell 1): single
        p, nb, hn = self._members(e - self.x)
        self.assertEqual(int(p[0]), 1)
        self.assertFalse(bool(hn[0]))

    def test_global_ends_have_no_phantom_neighbours(self):
        eps = 1e-9 * self.s
        p, nb, hn = self._members(self.BE[0] + eps)
        self.assertEqual(int(p[0]), 0)
        self.assertFalse(bool(hn[0]))  # x[0] = 0: nothing below cell 0
        p, nb, hn = self._members(self.BE[-1] - eps)
        self.assertEqual(int(p[0]), self.m.num_cap_cells - 1)
        self.assertFalse(bool(hn[0]))  # x[-1] = 0: nothing above the last

    def test_membership_indices_always_in_range(self):
        rng = np.random.default_rng(3)
        f = rng.uniform(self.BE[0], self.BE[-1] * (1 - 1e-12), 5000)
        p, nb, hn = self._members(f)
        for arr in (p, nb):
            self.assertTrue(int(arr.min()) >= 0)
            self.assertTrue(int(arr.max()) <= self.m.num_cap_cells - 1)
        # p < 0.5: a neighbour is never 2+ cells away
        np.testing.assert_array_less(np.abs(nb - p), 2)

    def test_numpy_twin_agrees(self):
        rng = np.random.default_rng(4)
        f = rng.uniform(self.BE[0], self.BE[-1] * (1 - 1e-12), 5000)
        band = np.clip(
            np.searchsorted(self.BE, f, side="right") - 1, 0, NUM_BANDS - 1
        ).astype(np.int64)
        p_d, nb_d, hn_d = self.m._cap_cell_members(band, f)
        p_h, nb_h, hn_h = self.m._np_cap_members(f, band, self.BE)
        np.testing.assert_array_equal(p_d, p_h)
        np.testing.assert_array_equal(hn_d, hn_h)
        np.testing.assert_array_equal(nb_d[hn_d], nb_h[hn_h])

    def test_staggered_overlap_membership(self):
        m = _move_overlap(4, 0.25, stagger=True, band_edges=self.BE)
        s = (self.BE[1] - self.BE[0]) / 4
        x = s / 6.0
        # first interior staggered edge sits at lo + s/2 (cells 0|1)
        e = self.BE[0] + 0.5 * s
        f = np.array([e + 0.5 * x])
        band = np.zeros(1, dtype=np.int64)
        p, nb, hn = m._cap_cell_members(band, f)
        self.assertEqual(int(p[0]), 1)
        self.assertTrue(bool(hn[0]))
        self.assertEqual(int(nb[0]), 0)


class OverlapCensusTest(unittest.TestCase):
    """Multi-membership census + default-0 bit-identity."""

    def test_overlap_zone_leaf_counts_in_both_cells(self):
        m = _move_overlap(4, 0.25)
        s = (BAND_EDGES[1] - BAND_EDGES[0]) / 4
        x = s / 6.0
        # one alive leaf in the cells 1|2 overlap zone, one in cell 0's core
        f0 = np.array([BAND_EDGES[0] + 2 * s + 0.4 * x,
                       BAND_EDGES[0] + 0.5 * s])
        sorter = _sorter(f0, [True, True])
        _, counts = m._cap_cell_counts(sorter)
        self.assertEqual(int(counts[0]), 1)
        self.assertEqual(int(counts[1]), 1)  # zone leaf counts here...
        self.assertEqual(int(counts[2]), 1)  # ...and here
        self.assertEqual(int(counts.sum()), 3)  # 2 leaves, 3 memberships

    def test_default_zero_census_is_bit_identical_to_the_partition(self):
        rng = np.random.default_rng(11)
        n = 400
        f0 = rng.uniform(BAND_EDGES[0], BAND_EDGES[-1] * (1 - 1e-12), n)
        alive = rng.random(n) < 0.6
        t = rng.integers(0, 3, n)
        w = rng.integers(0, 2, n)
        m_part = _move(4, ntemps=3, nwalkers=2)
        m_zero = _move_overlap(4, 0.0, ntemps=3, nwalkers=2)
        s = _sorter(f0, alive, t, w)
        flat_a, counts_a = m_part._cap_cell_counts(s)
        flat_b, counts_b = m_zero._cap_cell_counts(s)
        np.testing.assert_array_equal(flat_a, flat_b)
        np.testing.assert_array_equal(counts_a, counts_b)
        cap = rng.integers(1, 3, m_part.num_cap_cells)
        mem_a = m_part._sorter_cap_members(s)
        mem_b = m_zero._sorter_cap_members(s)
        self.assertIsNone(mem_a[1])
        self.assertIsNone(mem_b[1])
        np.testing.assert_array_equal(
            m_part._cap_at_cap_mask(s, counts_a, cap, flat_a, mem_a[0]),
            m_zero._cap_at_cap_mask(s, counts_b, cap, flat_b, mem_b[0],
                                    mem_b[1], mem_b[2]),
        )

    def test_default_zero_staggered_census_matches_searchsorted_oracle(self):
        """Overlap 0 on the STAGGERED grid = the plain primary-cell census."""
        rng = np.random.default_rng(12)
        n = 400
        f0 = rng.uniform(BAND_EDGES[0], BAND_EDGES[-1] * (1 - 1e-12), n)
        alive = rng.random(n) < 0.6
        m = _move_overlap(4, 0.0, stagger=True)
        s = _sorter(f0, alive)
        _, counts = m._cap_cell_counts(s)
        oracle_cells = np.clip(
            np.searchsorted(m.cap_edges, f0, side="right") - 1,
            0, m.num_cap_cells - 1,
        )
        oracle = np.bincount(oracle_cells[np.asarray(alive, bool)],
                             minlength=m.num_cap_cells)
        np.testing.assert_array_equal(counts, oracle)


class OverlapANDHeadroomTest(unittest.TestCase):
    """AND-headroom: a birth needs headroom in EVERY covering cell."""

    def _setup(self):
        m = _move_overlap(4, 0.25)
        s = (BAND_EDGES[1] - BAND_EDGES[0]) / 4
        x = s / 6.0
        return m, s, x

    def _birth_blocked(self, m, counts, cap, f_draw):
        """The prior-gate expression on a drawn birth frequency."""
        f = np.atleast_1d(f_draw)
        band = np.clip(
            np.searchsorted(BAND_EDGES, f, side="right") - 1, 0, NUM_BANDS - 1
        ).astype(np.int64)
        cells, nb, hn = m._cap_cell_members(band, f)
        t = np.zeros(1, dtype=np.int64)
        w = np.zeros(1, dtype=np.int64)
        return bool(m._row_at_cap(counts, cap, t, w, cells, nb, hn)[0])

    def test_birth_blocked_when_the_neighbour_cell_is_at_cap(self):
        m, s, x = self._setup()
        cap = np.ones(m.num_cap_cells, dtype=int)
        # cell 1 full (core leaf); cell 2 empty
        occ = _sorter(np.array([BAND_EDGES[0] + 1.5 * s]), [True])
        _, counts = m._cap_cell_counts(occ)
        # draw in the 1|2 overlap zone, PRIMARY cell 2 (which has headroom):
        # blocked anyway -- covering cell 1 is at cap
        f_draw = BAND_EDGES[0] + 2 * s + 0.4 * x
        self.assertTrue(self._birth_blocked(m, counts, cap, f_draw))
        # draw in cell 2's CORE: allowed (only cell 2 covers it)
        f_core = BAND_EDGES[0] + 2.5 * s
        self.assertFalse(self._birth_blocked(m, counts, cap, f_core))

    def test_birth_blocked_when_the_primary_cell_is_at_cap(self):
        m, s, x = self._setup()
        cap = np.ones(m.num_cap_cells, dtype=int)
        occ = _sorter(np.array([BAND_EDGES[0] + 1.5 * s]), [True])  # cell 1
        _, counts = m._cap_cell_counts(occ)
        # draw in the 1|2 zone with PRIMARY 1 (below the edge): blocked
        f_draw = BAND_EDGES[0] + 2 * s - 0.4 * x
        self.assertTrue(self._birth_blocked(m, counts, cap, f_draw))

    def test_birth_allowed_when_both_covering_cells_have_headroom(self):
        m, s, x = self._setup()
        cap = np.ones(m.num_cap_cells, dtype=int)
        occ = _sorter(np.array([BAND_EDGES[0] + 0.5 * s]), [True])  # cell 0
        _, counts = m._cap_cell_counts(occ)
        # draw in the 2|3 overlap zone: both empty -> allowed
        f_draw = BAND_EDGES[0] + 3 * s + 0.4 * x
        self.assertFalse(self._birth_blocked(m, counts, cap, f_draw))

    def test_alive_row_at_cap_when_any_covering_cell_full(self):
        m, s, x = self._setup()
        cap = np.ones(m.num_cap_cells, dtype=int)
        # leaf A in cell 1's core; leaf B in the 1|2 overlap zone.
        # Multi-census: cell 1 holds 2 (over cap), cell 2 holds 1 (at cap).
        f0 = np.array([BAND_EDGES[0] + 1.5 * s,
                       BAND_EDGES[0] + 2 * s + 0.4 * x])
        sorter = _sorter(f0, [True, True])
        cells, nb, hn = m._sorter_cap_members(sorter)
        flat, counts = m._cap_cell_counts(sorter, cells, nb, hn)
        at_cap = m._cap_at_cap_mask(sorter, counts, cap, flat, cells, nb, hn)
        self.assertTrue(bool(at_cap[0]))
        self.assertTrue(bool(at_cap[1]))
        # raise cell 2's cap: leaf B is still at cap through covering cell 1
        cap2 = cap.copy()
        cap2[2] = 5
        at_cap2 = m._cap_at_cap_mask(sorter, counts, cap2, flat, cells,
                                     nb, hn)
        self.assertTrue(bool(at_cap2[1]),
                        "any-covering-cell semantics: cell 1 still caps B")


class OverlapStaggerInvariantTest(unittest.TestCase):
    """Band edges must remain strictly inside cell CORES under overlap.

    The staggered grid bisects a cell with every band edge; widening by
    x = s/6 (p = 0.25) leaves the band edge s/2 - x = s/3 from the nearest
    core boundary -- so no widened-span (or core) endpoint approaches a
    band seam and the cap/band seam decoupling survives the overlap.
    """

    def _check(self, band_edges, k, p):
        be = np.asarray(band_edges, dtype=float)
        m = _move_overlap(k, p, stagger=True, band_edges=be)
        ce = np.asarray(m.cap_edges)
        x = np.asarray(m._cap_edge_ext)
        for e in be[1:-1]:  # interior band edges
            j = int(np.searchsorted(ce, e, side="right") - 1)
            core_lo = ce[j] + x[j]
            core_hi = ce[j + 1] - x[j + 1]
            self.assertLess(core_lo, e,
                            msg=f"band edge {e} at/below core floor")
            self.assertGreater(core_hi, e,
                               msg=f"band edge {e} at/above core ceiling")
            # margin: the smaller adjacent stride bounds s/2 - x from below
            s_loc = min((be[1:] - be[:-1]) / k)
            margin = s_loc / 2 - s_loc * p / (2 * (1 - p))
            self.assertGreaterEqual(
                min(e - core_lo, core_hi - e), margin * (1 - 1e-9)
            )
            # and the band edge is therefore SINGLE membership
            band = np.clip(np.searchsorted(be, e, side="right") - 1,
                           0, len(be) - 2)
            _, _, hn = m._cap_cell_members(
                np.array([band], dtype=np.int64), np.array([e])
            )
            self.assertFalse(bool(hn[0]))

    def test_uniform_grid(self):
        self._check(BAND_EDGES, 4, 0.25)

    def test_v6_analog_grid(self):
        # 135-bin uniform sub-bands, K = 4: the production geometry
        self._check(np.arange(0.0, 135.0 * 12 + 1, 135.0), 4, 0.25)

    def test_margin_in_bins_matches_the_spec(self):
        # s = 33.75 bins, x = 5.625: a band edge sits 11.25 bins inside
        # the straddling cell's core on each side
        be = np.arange(0.0, 135.0 * 3 + 1, 135.0)
        m = _move_overlap(4, 0.25, stagger=True, band_edges=be)
        ce = np.asarray(m.cap_edges)
        x = np.asarray(m._cap_edge_ext)
        e = be[1]
        j = int(np.searchsorted(ce, e, side="right") - 1)
        self.assertAlmostEqual(e - (ce[j] + x[j]), 11.25)
        self.assertAlmostEqual((ce[j + 1] - x[j + 1]) - e, 11.25)


class SurvivorPoolAtCapTest(unittest.TestCase):
    """User ruling 2026-08-26 (REVERSES the 2026-08-13 exclusion): sources
    in at-cap cap cells still get their in-model moves — the survivor pool
    keeps them. GB_INMODEL_POOL_AT_CAP=0 restores the old exclusion."""

    def setUp(self):
        self._old = os.environ.pop("GB_INMODEL_POOL_AT_CAP", None)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_INMODEL_POOL_AT_CAP", None)
        else:
            os.environ["GB_INMODEL_POOL_AT_CAP"] = self._old

    def _picked(self):
        return {
            "ids": np.array([0, 1, 2]),
            "temp_inds": np.zeros(3, dtype=np.int64),
            "walker_inds": np.zeros(3, dtype=np.int64),
            "cap_inds": np.zeros(3, dtype=np.int64),
        }

    def test_default_keeps_at_cap_rows_in_the_pool(self):
        m = _move(2)
        m._live_cap_state = ("counts", "caps")
        m._row_at_cap = lambda *a, **k: np.array([True, True, False])
        alive = np.array([True, True, True])
        out = m._survivor_pool_mask(alive, self._picked())
        np.testing.assert_array_equal(out, alive)

    def test_knob_zero_restores_live_state_exclusion(self):
        os.environ["GB_INMODEL_POOL_AT_CAP"] = "0"
        m = _move(2)
        m._live_cap_state = ("counts", "caps")
        m._row_at_cap = lambda *a, **k: np.array([True, True, False])
        out = m._survivor_pool_mask(np.array([True, True, True]), self._picked())
        np.testing.assert_array_equal(out, [False, False, True])

    def test_knob_zero_restores_snapshot_exclusion(self):
        os.environ["GB_INMODEL_POOL_AT_CAP"] = "0"
        m = _move(2)
        m._live_cap_state = None
        m._rj_at_cap_mask = np.array([False, True, False, True])
        out = m._survivor_pool_mask(np.array([True, True, True]), self._picked())
        np.testing.assert_array_equal(out, [True, False, True])


def _gate_move(cap_divisor=2, min_iters=3, nwalkers=1):
    """A fake move with just what ``_update_band_leaf_caps`` reads."""
    m = _move(cap_divisor, nwalkers=nwalkers)
    m.branch_name = "gb"
    m.leaf_cap_ll_improve = True
    m.leaf_cap_iter_only = False
    m.leaf_cap_ll_nsigma = 3.0
    m.leaf_cap_require_occupancy = False
    m.leaf_cap_min_iters = min_iters
    m.leaf_cap_ndim = 8  # hold threshold D/2 = 4.0
    m._work_branch = lambda ns: SimpleNamespace(shape=(1, nwalkers, 10))
    m._band_residual_lls = lambda acs: np.zeros((nwalkers, NUM_BANDS))
    return m


def _gate_state(m):
    bi = {"num_bands": m.num_bands}
    ensure_leaf_cap_fields(bi, m.num_bands)
    ensure_cap_cell_fields(bi, m.num_cap_cells)
    bi["band_leaf_cap"][:] = 1
    bi["cap_cell_leaf_cap"][:] = 1
    bi["cap_cell_iters"][:] = 0
    bi["cap_cell_best_ll"][:] = -np.inf
    state = SimpleNamespace(sub_states={"gb": SimpleNamespace(band_info=bi)})
    return state, bi


def _gate_step(m, state, stats, occ=None):
    """One updater call with scripted cold cell stats (nwalkers, ncells)."""
    stats = np.atleast_2d(np.asarray(stats, dtype=float))
    if occ is None:
        occ = (stats != 0).astype(int)
    else:
        occ = np.atleast_2d(np.asarray(occ, dtype=int))
    m._cap_cell_lls = lambda model, ns, band_lls: (
        stats, np.zeros(m.num_cap_cells)
    )
    m._cold_occupancy = lambda bc, ns: occ
    m._update_band_leaf_caps(
        SimpleNamespace(analysis_container_arr=None), state, None
    )


class CapGateEngagementTest(unittest.TestCase):
    """User spec 2026-08-26: the clock ENGAGES on any cell-ll change > 0.1
    (a first source landing counts); D/2 is the HOLD test — a cell still
    improving by >= D/2 keeps its cap, a stagnant engaged cell increments
    after ``leaf_cap_min_iters``."""

    CELL = 5

    def setUp(self):
        self._old = os.environ.get("GB_LEAF_CAP_REQUIRE_IMPROVEMENT")
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)
        else:
            os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = self._old

    def _stats(self, value):
        s = np.zeros(NUM_BANDS * 2)
        s[self.CELL] = value
        return s

    def test_first_source_engages_and_stagnation_increments(self):
        m = _gate_move()
        state, bi = _gate_state(m)
        _gate_step(m, state, self._stats(0.0))      # baseline, empty
        _gate_step(m, state, self._stats(100.0))    # birth -> engaged
        for _ in range(2):
            _gate_step(m, state, self._stats(100.0))
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 1)
        _gate_step(m, state, self._stats(100.0))    # 3rd flat -> increment
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)
        self.assertEqual(int(bi["cap_cell_iters"][self.CELL]), 0)
        self.assertEqual(bi["cap_cell_best_ll"][self.CELL], -np.inf)

    def test_starved_statistic_still_increments(self):
        # The frozen highf-grid probe: the cell is BORN in the very first
        # update (never observed empty), then the statistic starves to 0
        # while the cell stays occupied. The cap must still ramp.
        m = _gate_move()
        state, bi = _gate_state(m)
        occ = np.ones(NUM_BANDS * 2)
        _gate_step(m, state, self._stats(1131.0), occ)
        _gate_step(m, state, self._stats(663.0), occ)
        _gate_step(m, state, self._stats(0.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 1)
        _gate_step(m, state, self._stats(0.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)

    def test_born_converged_cell_still_ramps(self):
        # Minimal frozen-probe repro: a source seats essentially converged
        # in the first update and the statistic never moves again. "A
        # source added counts" (user spec) -- occupied at first sight
        # engages; no D/2 improvement for min_iters -> increment.
        m = _gate_move()
        state, bi = _gate_state(m)
        occ = np.ones(NUM_BANDS * 2)
        for _ in range(3):
            _gate_step(m, state, self._stats(2000.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 1)
        _gate_step(m, state, self._stats(2000.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)
        self.assertEqual(bi["cap_cell_best_ll"][self.CELL], -np.inf)

    def test_d_over_2_improvement_holds_the_cap(self):
        m = _gate_move()
        state, bi = _gate_state(m)
        _gate_step(m, state, self._stats(0.0))
        for v in (100.0, 110.0, 120.0, 130.0, 140.0, 150.0):
            _gate_step(m, state, self._stats(v))
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 1)
        self.assertEqual(int(bi["cap_cell_iters"][self.CELL]), 0)

    def test_sub_threshold_creep_engages_but_does_not_hold(self):
        # +1/iter is a >0.1 change (engaged) but < D/2 (no hold).
        m = _gate_move()
        state, bi = _gate_state(m)
        _gate_step(m, state, self._stats(0.0))
        _gate_step(m, state, self._stats(100.0))
        for v in (101.0, 102.0):
            _gate_step(m, state, self._stats(v))
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 1)
        _gate_step(m, state, self._stats(103.0))
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)

    def test_empty_cell_never_engages(self):
        m = _gate_move()
        state, bi = _gate_state(m)
        for _ in range(6):
            _gate_step(m, state, np.zeros(NUM_BANDS * 2))
        np.testing.assert_array_equal(
            bi["cap_cell_leaf_cap"], np.ones(NUM_BANDS * 2)
        )
        np.testing.assert_array_equal(
            bi["cap_cell_iters"], np.zeros(NUM_BANDS * 2)
        )

    def test_emptied_cell_clock_freezes(self):
        # Occupant dies: the drop engages/keeps the flag, but patience only
        # accrues while the cell is OCCUPIED -- no ratchet on a corpse.
        m = _gate_move()
        state, bi = _gate_state(m)
        _gate_step(m, state, self._stats(0.0))
        _gate_step(m, state, self._stats(100.0))            # occupied
        for _ in range(6):                                   # death: empty
            _gate_step(m, state, self._stats(0.0),
                       occ=np.zeros(NUM_BANDS * 2))
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 1)


class ScatterLeafProductsPersistenceTest(unittest.TestCase):
    """At-cap d_h/h_h persistence (2026-08-26 root-cause fix): a cold leaf
    with no fresh in-model capture keeps its previous per-leaf values
    through the repack instead of being NaN-wiped."""

    def _move(self):
        m = _move(1)
        m.branch_name = "gb"
        return m

    def test_uncaptured_cold_leaf_keeps_value_through_repack(self):
        m = self._move()
        # A (row 0): no fresh capture, moves leaf 2 -> 0. B (row 1): fresh
        # capture 7.0, stays at leaf 1.
        m._sorter_dh = np.array([np.nan, 7.0])
        m._sorter_hh = np.array([np.nan, 49.0])
        d_h = np.full((1, 3), np.nan)
        h_h = np.full((1, 3), np.nan)
        d_h[0, 2], h_h[0, 2] = 5.0, 25.0
        d_h[0, 1], h_h[0, 1] = 6.0, 36.0
        sub = SimpleNamespace(d_h=d_h, h_h=h_h)
        ns = SimpleNamespace(sub_states={"gb": sub})
        alive = np.array([True, True])
        z = np.zeros(2, dtype=np.int64)
        inds_new = (z, z, np.array([0, 1]))
        inds_old = (z, z, np.array([2, 1]))
        m._scatter_leaf_products(ns, alive, inds_new, inds_old)
        self.assertEqual(d_h[0, 0], 5.0)
        self.assertEqual(h_h[0, 0], 25.0)
        self.assertEqual(d_h[0, 1], 7.0)
        self.assertEqual(h_h[0, 1], 49.0)
        self.assertTrue(np.isnan(d_h[0, 2]))

    def test_without_inds_old_uncaptured_is_wiped(self):
        m = self._move()
        m._sorter_dh = np.array([np.nan])
        m._sorter_hh = np.array([np.nan])
        d_h = np.full((1, 2), np.nan)
        h_h = np.full((1, 2), np.nan)
        d_h[0, 1], h_h[0, 1] = 5.0, 25.0
        sub = SimpleNamespace(d_h=d_h, h_h=h_h)
        ns = SimpleNamespace(sub_states={"gb": sub})
        z = np.zeros(1, dtype=np.int64)
        m._scatter_leaf_products(
            ns, np.array([True]), (z, z, np.array([0]))
        )
        self.assertTrue(np.isnan(d_h[0, 0]))
        self.assertTrue(np.isnan(d_h[0, 1]))

    def test_hot_rows_are_ignored(self):
        m = self._move()
        m._sorter_dh = np.array([np.nan, 3.0])
        m._sorter_hh = np.array([np.nan, 9.0])
        d_h = np.full((1, 2), np.nan)
        h_h = np.full((1, 2), np.nan)
        d_h[0, 0], h_h[0, 0] = 5.0, 25.0
        sub = SimpleNamespace(d_h=d_h, h_h=h_h)
        ns = SimpleNamespace(sub_states={"gb": sub})
        # row 0 cold uncaptured (persists); row 1 HOT (never scattered)
        t = np.array([0, 1], dtype=np.int64)
        w = np.zeros(2, dtype=np.int64)
        inds_new = (t, w, np.array([0, 1]))
        inds_old = (t, w, np.array([0, 1]))
        m._scatter_leaf_products(ns, np.array([True, True]), inds_new, inds_old)
        self.assertEqual(d_h[0, 0], 5.0)
        self.assertTrue(np.isnan(d_h[0, 1]))


class DeathCaptureHarvestTest(unittest.TestCase):
    """RJ death-side scoring harvested into the sorter capture (user-approved
    2026-08-26): every picked alive leaf is scored at its own params with
    phase_maximize=False each round — the same numbers the in-model capture
    stores, up to the exposed-residual convention d_h_cap = d_h_raw + h_h."""

    def test_harvest_applies_exposed_convention(self):
        m = _move(1)
        m.branch_name = "gb"
        m._sorter_dh = None
        m._sorter_hh = None
        m._harvest_death_capture(
            np.array([2, 4]), np.array([1.5, -0.5]),
            np.array([100.0, 64.0]), 6,
        )
        np.testing.assert_allclose(
            np.asarray(m._sorter_dh)[[2, 4]], [101.5, 63.5])
        np.testing.assert_allclose(
            np.asarray(m._sorter_hh)[[2, 4]], [100.0, 64.0])
        self.assertTrue(np.isnan(np.asarray(m._sorter_dh)[[0, 1, 3, 5]]).all())

    def test_harvest_preserves_other_captures(self):
        m = _move(1)
        m.branch_name = "gb"
        m._sorter_dh = np.full(4, np.nan)
        m._sorter_hh = np.full(4, np.nan)
        m._sorter_dh[1], m._sorter_hh[1] = 7.0, 49.0
        m._harvest_death_capture(
            np.array([2]), np.array([1.0]), np.array([9.0]), 4)
        self.assertEqual(m._sorter_dh[1], 7.0)
        np.testing.assert_allclose(m._sorter_dh[2], 10.0)
        np.testing.assert_allclose(m._sorter_hh[2], 9.0)


class Divisor1OverlapTest(unittest.TestCase):
    """User config 2026-08-26: cap cells LINED UP with the sub-bands
    (divisor 1) but with 1/4 overlap — spans widen across the band seams,
    multi-membership and the drift gate stay active."""

    def test_extensions_at_divisor_one(self):
        ce = make_cap_edges(BAND_EDGES, 1)
        x = make_cap_edge_extensions(BAND_EDGES, ce, 1, 0.25)
        width = BAND_EDGES[1] - BAND_EDGES[0]
        np.testing.assert_allclose(x[1:-1], width / 6.0, rtol=1e-12)
        self.assertEqual(float(x[0]), 0.0)
        self.assertEqual(float(x[-1]), 0.0)

    def test_members_cross_band_seam(self):
        m = _move_overlap(1, 0.25)
        width = BAND_EDGES[1] - BAND_EDGES[0]
        x = width / 6.0
        # one leaf just above the band-0/1 seam (inside cell 0's upper
        # overlap zone), one in band 1's core
        f0 = np.array([BAND_EDGES[1] + 0.5 * x, BAND_EDGES[1] + 0.5 * width])
        band = np.array([1, 1])
        p, nb, has = m._cap_cell_members(band, f0)
        np.testing.assert_array_equal(p, [1, 1])
        self.assertIsNotNone(nb)
        self.assertTrue(bool(has[0]))
        self.assertEqual(int(nb[0]), 0)
        self.assertFalse(bool(has[1]))
        # numpy twin agrees
        p2, nb2, has2 = m._np_cap_members(f0, band, BAND_EDGES)
        np.testing.assert_array_equal(p, p2)
        np.testing.assert_array_equal(has, has2)
        np.testing.assert_array_equal(nb, nb2)

    def test_drift_gate_setup_active(self):
        m = _move_overlap(1, 0.25)
        m.cap_drift_gate = True
        m._f0_col = 1
        m._cap_leaf_cap = np.ones(m.num_cap_cells, dtype=int)
        m._cap_cell_counts = lambda bs, *a, **k: (
            None, np.zeros(m.ntemps * m.nwalkers * m.num_cap_cells,
                           dtype=np.int32))
        out = m._cap_drift_gate_setup(SimpleNamespace())
        self.assertIsNotNone(out)

    def test_no_overlap_divisor_one_stays_off(self):
        m = _move_overlap(1, 0.0)
        p, nb, has = m._cap_cell_members(
            np.array([1]), np.array([BAND_EDGES[1] + 1e-6]))
        self.assertIsNone(nb)
        m.cap_drift_gate = True
        m._f0_col = 1
        m._cap_leaf_cap = np.ones(m.num_cap_cells, dtype=int)
        self.assertIsNone(m._cap_drift_gate_setup(SimpleNamespace()))


class EntryVetoHeadroomTest(unittest.TestCase):
    """User ruling 2026-08-26: in-model / replace f0 moves may enter a
    foreign cell up to GB_CAP_INMODEL_HEADROOM (default 2) OVER its cap;
    RJ birth gates stay strict (they do not use this veto)."""

    def setUp(self):
        self._old = os.environ.pop("GB_CAP_INMODEL_HEADROOM", None)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_CAP_INMODEL_HEADROOM", None)
        else:
            os.environ["GB_CAP_INMODEL_HEADROOM"] = self._old

    def _veto(self, dest_count, cap_val=1):
        m = _move(1)
        cap = np.full(m.num_cap_cells, cap_val, dtype=int)
        counts = np.zeros(m.ntemps * m.nwalkers * m.num_cap_cells,
                          dtype=np.int32)
        counts[1] = dest_count            # destination cell 1, (t0, w0)
        t = np.zeros(1, dtype=np.int64)
        w = np.zeros(1, dtype=np.int64)
        cur = (np.array([0]), None, None)  # moving from cell 0
        new = (np.array([1]), None, None)  # into cell 1
        return bool(m._cap_new_entry_veto(counts, cap, t, w, cur, new)[0])

    def test_default_headroom_two_admits_at_cap(self):
        self.assertFalse(self._veto(dest_count=1))   # at cap: allowed
        self.assertFalse(self._veto(dest_count=2))   # cap+1: allowed

    def test_default_headroom_two_blocks_at_cap_plus_two(self):
        self.assertTrue(self._veto(dest_count=3))    # cap+2 occupants: full

    def test_headroom_zero_restores_strict(self):
        os.environ["GB_CAP_INMODEL_HEADROOM"] = "0"
        self.assertTrue(self._veto(dest_count=1))


class CapGateOccupancyTest(unittest.TestCase):
    """User-approved fix 2: a cap only increments when its allowance is
    actually USED (some cold walker holds cap leaves in the cell), and
    never past GB_CAP_CELL_MAX."""

    CELL = 5

    def setUp(self):
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1"
        self._max = os.environ.pop("GB_CAP_CELL_MAX", None)

    def tearDown(self):
        os.environ.pop("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)
        if self._max is None:
            os.environ.pop("GB_CAP_CELL_MAX", None)
        else:
            os.environ["GB_CAP_CELL_MAX"] = self._max

    def _stats(self, value):
        s = np.zeros(NUM_BANDS * 2)
        s[self.CELL] = value
        return s

    def test_below_cap_stagnant_cell_holds(self):
        # cap 2, but every walker holds only 1 leaf: the allowance is not
        # used, so stagnation must NOT buy a third slot.
        m = _gate_move()
        state, bi = _gate_state(m)
        bi["cap_cell_leaf_cap"][self.CELL] = 2
        occ = np.zeros((1, NUM_BANDS * 2)); occ[0, self.CELL] = 1
        for _ in range(8):
            _gate_step(m, state, self._stats(2000.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)

    def test_at_cap_stagnant_cell_increments(self):
        m = _gate_move()
        state, bi = _gate_state(m)
        occ = np.zeros((1, NUM_BANDS * 2)); occ[0, self.CELL] = 1
        for _ in range(6):
            _gate_step(m, state, self._stats(2000.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)

    def test_cell_max_ceiling(self):
        os.environ["GB_CAP_CELL_MAX"] = "2"
        m = _gate_move()
        state, bi = _gate_state(m)
        bi["cap_cell_leaf_cap"][self.CELL] = 2
        occ = np.zeros((1, NUM_BANDS * 2)); occ[0, self.CELL] = 2
        for _ in range(8):
            _gate_step(m, state, self._stats(2000.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)

    def test_ramp_pending_published(self):
        m = _gate_move()
        state, bi = _gate_state(m)
        occ = np.zeros((1, NUM_BANDS * 2)); occ[0, self.CELL] = 1
        _gate_step(m, state, self._stats(2000.0), occ)   # engage, iters 0
        _gate_step(m, state, self._stats(2000.0), occ)   # iters 1
        self.assertEqual(int(m._cap_ramp_pending), 1)
        for _ in range(4):                                # ... increment
            _gate_step(m, state, self._stats(2000.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)
        # post-increment: occupancy (1) < new cap (2) -> nothing pending
        self.assertEqual(int(m._cap_ramp_pending), 0)


class Divisor1GateRoutingTest(unittest.TestCase):
    """At divisor 1 WITH overlap the gate must read the cap-cell statistic
    (source-attributed on sub-layer grids), not the band residual windows,
    and store what it read in band_cold_ll."""

    def setUp(self):
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1"

    def tearDown(self):
        os.environ.pop("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)

    def test_divisor1_overlap_routes_the_cell_statistic(self):
        m = _gate_move(cap_divisor=1)
        m.cap_overlap_frac = 0.25
        m._cap_edge_ext = make_cap_edge_extensions(
            BAND_EDGES, m.cap_edges, 1, 0.25)
        bi = {"num_bands": m.num_bands}
        ensure_leaf_cap_fields(bi, m.num_bands)
        bi["band_leaf_cap"][:] = 1
        bi["band_cold_ll"] = np.zeros((1, m.num_bands))
        state = SimpleNamespace(
            sub_states={"gb": SimpleNamespace(band_info=bi)})
        occ = np.zeros((1, m.num_bands)); occ[0, 2] = 1
        stats = None
        for v in (100.0, 110.0, 120.0, 130.0, 140.0, 150.0):
            stats = np.zeros((1, m.num_bands)); stats[0, 2] = v
            _gate_step(m, state, stats, occ)
        # improving cell: the D/2 hold must keep the cap at 1 (the band
        # residual stub is flat zeros, which would increment instead)
        np.testing.assert_array_equal(
            bi["band_leaf_cap"], np.ones(m.num_bands))
        # and the gate must have stored WHAT IT READ
        np.testing.assert_allclose(bi["band_cold_ll"], stats)


class StageCapQuiescenceTest(unittest.TestCase):
    """User-approved fix 1: the search-stage nleaves plateau must not
    declare convergence while a cap increment is pending (an engaged,
    occupied-at-cap cell mid patience window)."""

    def _step_obj(self):
        from lisatools.globalfit.recipe import RJRecipeStep
        s = RJRecipeStep.__new__(RJRecipeStep)
        s.convergence_fn = None
        s.convergence_iter = 2
        s.plateau_branch = "gb"
        s._stage_start_iter = 0
        return s

    def _sampler(self, moves):
        nl = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
        backend = SimpleNamespace(
            iteration=10,
            get_nleaves=lambda branch_names, temp_index: {"gb": nl},
        )
        return SimpleNamespace(backend=backend, moves=moves)

    def test_plateau_stops_when_ramp_quiet(self):
        s = self._step_obj()
        sampler = self._sampler([SimpleNamespace(_cap_ramp_pending=0)])
        self.assertTrue(s.stopping_function(0, None, sampler))

    def test_pending_ramp_vetoes_the_stop(self):
        s = self._step_obj()
        sampler = self._sampler([SimpleNamespace(_cap_ramp_pending=2)])
        self.assertFalse(s.stopping_function(0, None, sampler))

    def test_nested_moves_are_scanned(self):
        s = self._step_obj()
        inner = SimpleNamespace(_cap_ramp_pending=1)
        sampler = self._sampler([SimpleNamespace(moves=[inner])])
        self.assertFalse(s.stopping_function(0, None, sampler))


class RampPendingEngagedTest(unittest.TestCase):
    """Regression for the aligned-probe handoff bug (2026-08-26): a cell
    that keeps IMPROVING resets its clock, but it is the LEAST converged
    cell of all — pending must mean engaged & occupied-at-cap & below
    ceiling, regardless of the patience counter."""

    CELL = 5

    def setUp(self):
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1"

    def tearDown(self):
        os.environ.pop("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)

    def _stats(self, value):
        s = np.zeros(NUM_BANDS * 2)
        s[self.CELL] = value
        return s

    def test_improving_at_cap_cell_is_pending(self):
        m = _gate_move()
        state, bi = _gate_state(m)
        occ = np.zeros((1, NUM_BANDS * 2)); occ[0, self.CELL] = 1
        for v in (2000.0, 2010.0, 2020.0):   # > D/2 every step: clock resets
            _gate_step(m, state, self._stats(v), occ)
            self.assertEqual(int(m._cap_ramp_pending), 1)

    def test_pending_clears_after_increment(self):
        m = _gate_move()
        state, bi = _gate_state(m)
        occ = np.zeros((1, NUM_BANDS * 2)); occ[0, self.CELL] = 1
        for _ in range(6):
            _gate_step(m, state, self._stats(2000.0), occ)
        self.assertEqual(int(bi["cap_cell_leaf_cap"][self.CELL]), 2)
        self.assertEqual(int(m._cap_ramp_pending), 0)


class RjBirthCtrModeTest(unittest.TestCase):
    """User ruling 2026-08-26: SEARCH-cycle fstat births/deaths use PER-ROW
    F-stat centers at proposal time (the replace-move fix, completed);
    PE moves keep the epoch table. GB_RJ_BIRTH_CTR_MODE forces either."""

    def setUp(self):
        self._old = os.environ.pop("GB_RJ_BIRTH_CTR_MODE", None)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_RJ_BIRTH_CTR_MODE", None)
        else:
            os.environ["GB_RJ_BIRTH_CTR_MODE"] = self._old

    def _named(self, name):
        m = _move(1)
        m.name = name
        return m

    def test_auto_search_moves_go_perrow(self):
        self.assertTrue(self._named("rj_fstat_search")._rj_birth_perrow())

    def test_auto_pe_moves_keep_the_table(self):
        self.assertFalse(self._named("rj_fstat_pe")._rj_birth_perrow())

    def test_env_forces_perrow_everywhere(self):
        os.environ["GB_RJ_BIRTH_CTR_MODE"] = "perrow"
        self.assertTrue(self._named("rj_fstat_pe")._rj_birth_perrow())

    def test_env_forces_table_everywhere(self):
        os.environ["GB_RJ_BIRTH_CTR_MODE"] = "table"
        self.assertFalse(self._named("rj_fstat_search")._rj_birth_perrow())


class CtrGbFreeWindowTest(unittest.TestCase):
    """User ruling 2026-08-26: per-row F-stat centers must see the SAME
    GB-free residual the epoch fit sees (all reference-walker GBs restored)
    — otherwise centers collapse at spots the reference walker has found,
    choking the other walkers' births."""

    def setUp(self):
        for k in ("GB_RJ_CTR_GBFREE", "GB_RJ_BIRTH_CTR_MODE"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("GB_RJ_CTR_GBFREE", "GB_RJ_BIRTH_CTR_MODE"):
            os.environ.pop(k, None)

    def _move_with_recorder(self, name="rj_fstat_search", n_ref=3):
        m = _move(1)
        m.name = name
        m.calls = []
        m.remove_sources_from_residual = (
            lambda model, sorter, **sel: m.calls.append(("expose", dict(sel))))
        m.add_sources_to_residual = (
            lambda model, sorter, **sel: m.calls.append(("restore", dict(sel))))
        sorter = SimpleNamespace(
            get_subset_bool=lambda **sel: np.ones(n_ref, dtype=bool))
        return m, sorter

    def test_window_exposes_and_restores_for_search(self):
        m, sorter = self._move_with_recorder()
        with m._ctr_gbfree_window(None, sorter, 0):
            self.assertEqual(m.calls, [("expose", dict(
                temp=0, walker=0, apply_inds=True))])
        self.assertEqual(m.calls[-1][0], "restore")

    def test_restore_runs_on_exception(self):
        m, sorter = self._move_with_recorder()
        with self.assertRaises(RuntimeError):
            with m._ctr_gbfree_window(None, sorter, 0):
                raise RuntimeError("boom")
        self.assertEqual([c[0] for c in m.calls], ["expose", "restore"])

    def test_knob_zero_disables(self):
        os.environ["GB_RJ_CTR_GBFREE"] = "0"
        m, sorter = self._move_with_recorder()
        with m._ctr_gbfree_window(None, sorter, 0):
            pass
        self.assertEqual(m.calls, [])

    def test_empty_reference_walker_is_a_noop(self):
        m, sorter = self._move_with_recorder(n_ref=0)
        sorter.get_subset_bool = lambda **sel: np.zeros(0, dtype=bool)
        with m._ctr_gbfree_window(None, sorter, 0):
            pass
        self.assertEqual(m.calls, [])
