"""``rescale_store_walkers``: duplicate the walker axis, and ONLY that.

The script clones a run store into one with N x the walkers so a run can
continue at a larger walker block without discarding the fitted noise, the
GB leaves, the F-stat epoch cache or the warm start.

The defect it has to be safe against is misidentifying the walker axis.
``sub_backend/mbh`` routinely has ``nleaves_max == nwalkers`` (4 and 4 in
the 6-month production store), so ``mbh/chain`` at
``(nsteps, ntemps, nwalkers, nleaves, ndim)`` carries TWO axes of the same
length and ``mbh/in_model_accepted`` at ``(nsteps, nleaves, ntemps)``
carries one that is not a walker axis at all. Anything that picked the
axis by length would double the leaf axis and write a store that loads
cleanly and is wrong. So every dataset is matched against a full shape
signature built from its group's own attributes, and this file pins that.
"""

import os
import sys
import unittest

import h5py
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fstat_proposal"))

import rescale_store_walkers as rsw  # noqa: E402

NSTEPS, NS, NT, NW, IT = 6, 1, 1, 2, 4
GB_NT, GB_NL, GB_ND, GB_NB = 3, 5, 9, 4
# nleaves_max == nwalkers ON PURPOSE: this is the production ambiguity.
MBH_NT, MBH_NL, MBH_ND = 2, NW, 11


def _fill(ds, seed):
    rng = np.random.default_rng(seed)
    if ds.dtype.kind == "b":
        ds[...] = rng.integers(0, 2, ds.shape).astype(bool)
    elif ds.dtype.kind in "iu":
        ds[...] = rng.integers(0, 100, ds.shape)
    else:
        a = rng.normal(size=ds.shape)
        if a.size > 3:                 # NaN must survive the copy too
            a.flat[1] = np.nan
        ds[...] = a


def make_store(path, nw=NW):
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["nsamplers"] = NS
        g.attrs["ntemps"] = NT
        g.attrs["nwalkers"] = nw
        g.attrs["iteration"] = IT
        g.attrs["nbranches"] = 2
        nl = g.create_group("nleaves_max")
        nd = g.create_group("ndims")
        for br, (leaves, ndim) in (("gb", (GB_NL, GB_ND)),
                                   ("mbh", (MBH_NL, MBH_ND))):
            nl.attrs[br] = leaves
            nd.attrs[br] = ndim
        chain, inds = g.create_group("chain"), g.create_group("inds")
        for i, (br, (leaves, ndim)) in enumerate(
                (("gb", (GB_NL, GB_ND)), ("mbh", (MBH_NL, MBH_ND)))):
            _fill(chain.create_dataset(
                br, (NSTEPS, NS, NT, nw, leaves, ndim), dtype="f8",
                maxshape=(None, NS, NT, nw, leaves, ndim)), i)
            _fill(inds.create_dataset(
                br, (NSTEPS, NS, NT, nw, leaves), dtype="?",
                maxshape=(None, NS, NT, nw, leaves)), 10 + i)
        for name in ("log_like", "log_prior"):
            _fill(g.create_dataset(name, (NSTEPS, NS, NT, nw), dtype="f8",
                                   maxshape=(None, NS, NT, nw)), 20)
        _fill(g.create_dataset("betas", (NSTEPS, NS, NT), dtype="f8",
                               maxshape=(None, NS, NT)), 21)
        _fill(g.create_dataset("accepted", (NS, NT, nw), dtype="f8"), 22)
        _fill(g.create_dataset("samplers_running", (NSTEPS, NS), dtype="?",
                               maxshape=(None, NS)), 23)

        sb = g.create_group("sub_backend")
        gb = sb.create_group("gb")
        for k, v in (("ntemps", GB_NT), ("nwalkers", nw),
                     ("nleaves_max", GB_NL), ("ndim", GB_ND),
                     ("num_bands", GB_NB)):
            gb.attrs[k] = v
        _fill(gb.create_dataset("chain", (NSTEPS, GB_NT, nw, GB_NL, GB_ND),
                                dtype="f8"), 30)
        _fill(gb.create_dataset("inds", (NSTEPS, GB_NT, nw, GB_NL),
                                dtype="?"), 31)
        for name in ("d_h", "h_h"):
            _fill(gb.create_dataset(name, (NSTEPS, nw, GB_NL), dtype="f8"), 32)
        _fill(gb.create_dataset("band_cold_ll", (NSTEPS, nw, GB_NB),
                                dtype="f8"), 33)
        _fill(gb.create_dataset("band_num_binaries",
                                (NSTEPS, GB_NT, nw, GB_NB), dtype="f8"), 34)
        _fill(gb.create_dataset("band_temps", (NSTEPS, GB_NB, GB_NT),
                                dtype="f8"), 35)
        _fill(gb.create_dataset("band_leaf_cap", (NSTEPS, GB_NB),
                                dtype="f8"), 36)
        _fill(gb.create_dataset("band_swaps_accepted",
                                (NSTEPS, GB_NB, GB_NT - 1), dtype="f8"), 37)
        _fill(gb.create_dataset("band_edges", (GB_NB + 1,), dtype="f8"), 38)

        mbh = sb.create_group("mbh")
        for k, v in (("ntemps", MBH_NT), ("nwalkers", nw),
                     ("nleaves_max", MBH_NL), ("ndim", MBH_ND)):
            mbh.attrs[k] = v
        _fill(mbh.create_dataset(
            "chain", (NSTEPS, MBH_NT, nw, MBH_NL, MBH_ND), dtype="f8"), 40)
        _fill(mbh.create_dataset("inds", (NSTEPS, MBH_NT, nw, MBH_NL),
                                 dtype="?"), 41)
        for name in ("d_h", "h_h"):
            _fill(mbh.create_dataset(name, (NSTEPS, nw, MBH_NL),
                                     dtype="f8"), 42)
        for name in ("log_like", "log_prior"):
            _fill(mbh.create_dataset(name, (NSTEPS, MBH_NL, MBH_NT, nw),
                                     dtype="f8"), 43)
        _fill(mbh.create_dataset("betas_all", (NSTEPS, MBH_NL, MBH_NT),
                                 dtype="f8"), 44)
        # THE AMBIGUOUS ONE: (nsteps, nleaves, ntemps) where nleaves == nw
        _fill(mbh.create_dataset("in_model_accepted",
                                 (NSTEPS, MBH_NL, MBH_NT), dtype="i8"), 45)
        _fill(mbh.create_dataset("swaps_accepted",
                                 (NSTEPS, MBH_NL, MBH_NT - 1), dtype="i8"), 46)
    return path


def _eq(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        return np.array_equal(x, y, equal_nan=True)
    return np.array_equal(x, y)


class PlanTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.src = make_store(os.path.join(self.tmp, "s_testing.h5"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _plan(self):
        with h5py.File(self.src, "r") as f:
            return rsw.plan_rescale(f, "global_fit", 2)

    def test_no_refusals_on_a_well_formed_store(self):
        _, refusals, _, _ = self._plan()
        self.assertEqual(refusals, [])

    def test_walker_axes_are_the_expected_ones(self):
        entries, _, _, _ = self._plan()
        axes = {e.path: e.axis for e in entries}
        self.assertEqual(axes["log_like"], 3)
        self.assertEqual(axes["chain/gb"], 3)
        self.assertEqual(axes["accepted"], 2)
        self.assertEqual(axes["sub_backend/gb/chain"], 2)
        self.assertEqual(axes["sub_backend/gb/d_h"], 1)
        self.assertEqual(axes["sub_backend/gb/band_cold_ll"], 1)
        self.assertEqual(axes["sub_backend/gb/band_num_binaries"], 2)
        self.assertEqual(axes["sub_backend/mbh/log_like"], 3)

    def test_the_mbh_leaf_axis_is_NOT_taken_for_walkers(self):
        """nleaves_max == nwalkers: the production ambiguity."""
        entries, _, _, _ = self._plan()
        axes = {e.path: e.axis for e in entries}
        # chain: walkers at 2, leaves at 3 -- both length nw
        self.assertEqual(axes["sub_backend/mbh/chain"], 2)
        # per-leaf counters and ladders carry no walker axis at all
        for name in ("in_model_accepted", "swaps_accepted", "betas_all"):
            self.assertIsNone(axes[f"sub_backend/mbh/{name}"], name)

    def test_band_tables_are_walker_free(self):
        entries, _, _, _ = self._plan()
        axes = {e.path: e.axis for e in entries}
        for name in ("band_temps", "band_leaf_cap", "band_swaps_accepted",
                     "band_edges"):
            self.assertIsNone(axes[f"sub_backend/gb/{name}"], name)

    def test_per_walker_cap_tables_carry_a_walker_axis(self):
        """GB_LEAF_CAP_PER_WALKER (2026-09-22) gives the cap family one.

        The rest of the cap family is walker-free, which is exactly why a
        walker rescale can carry the whole GB search state forward. The
        ``_w`` twins are the exception and must be TILED like any other
        walker-axis table -- silently leaving them at the old width would
        gate the new walkers against a table that does not cover them.
        """
        with h5py.File(self.src, "a") as f:
            gb = f["global_fit/sub_backend/gb"]
            for name in ("cap_cell_leaf_cap_w", "cap_cell_iters_w",
                         "cap_cell_best_ll_w"):
                _fill(gb.create_dataset(
                    name, (NSTEPS, NW, GB_NB), dtype="f8"), 50)
            _fill(gb.create_dataset(
                "band_best_ll_w", (NSTEPS, NW, GB_NB), dtype="f8"), 51)
        entries, refusals, _, _ = self._plan()
        self.assertEqual(refusals, [])
        axes = {e.path: e.axis for e in entries}
        for name in ("cap_cell_leaf_cap_w", "cap_cell_iters_w",
                     "cap_cell_best_ll_w", "band_best_ll_w"):
            self.assertEqual(axes[f"sub_backend/gb/{name}"], 1, name)

    def test_an_unclassified_dataset_is_a_refusal_not_a_guess(self):
        with h5py.File(self.src, "a") as f:
            f["global_fit/sub_backend/gb"].create_dataset(
                "something_new", (NSTEPS, NW, GB_NB), dtype="f8")
        _, refusals, _, _ = self._plan()
        self.assertTrue(any("something_new" in r for r in refusals),
                        f"an unknown dataset with a walker-length axis must "
                        f"refuse; got {refusals}")

    def test_a_dataset_outside_the_enumerated_layout_is_a_refusal(self):
        """Completeness: anything not enumerated would be DROPPED silently."""
        with h5py.File(self.src, "a") as f:
            f["global_fit"].create_group("extra").create_dataset(
                "thing", (3,), dtype="f8")
        _, refusals, _, _ = self._plan()
        self.assertTrue(any("extra/thing" in r for r in refusals), refusals)

    def test_a_layout_change_refuses_rather_than_tiling_the_wrong_axis(self):
        with h5py.File(self.src, "a") as f:
            del f["global_fit/sub_backend/gb/d_h"]
            f["global_fit/sub_backend/gb"].create_dataset(
                "d_h", (NSTEPS, GB_NL, NW), dtype="f8")   # axes swapped
        _, refusals, _, _ = self._plan()
        self.assertTrue(any("d_h" in r for r in refusals), refusals)


class BuildTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.src = make_store(os.path.join(self.tmp, "s_testing.h5"))
        self.dst = os.path.join(self.tmp, "out.h5")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, mode="tile", factor=2):
        rsw.build_rescaled(self.src, self.dst, "global_fit", factor, mode)
        return factor

    def _check(self, factor, src_of):
        with h5py.File(self.src, "r") as a, h5py.File(self.dst, "r") as b:
            entries, _, _, it = rsw.plan_rescale(a, "global_fit", factor)
            n_scaled = 0
            for e in entries:
                p = f"global_fit/{e.path}"
                da, db = a[p], b[p]
                rows = it if e.rowwise else (da.shape[0] if da.shape else 0)
                if e.axis is None:
                    self.assertEqual(da.shape, db.shape, e.path)
                    sl = tuple([slice(0, rows)]
                               + [slice(None)] * (da.ndim - 1))
                    self.assertTrue(_eq(da[sl], db[sl]), e.path)
                    continue
                n_scaled += 1
                nw = da.shape[e.axis]
                self.assertEqual(db.shape[e.axis], nw * factor, e.path)
                for w in range(nw * factor):
                    sa = [slice(0, rows)] + [slice(None)] * (da.ndim - 1)
                    sb = list(sa)
                    sa[e.axis] = src_of(w, nw)
                    sb[e.axis] = w
                    self.assertTrue(_eq(da[tuple(sa)], db[tuple(sb)]),
                                    f"{e.path}: walker {w}")
            self.assertGreater(n_scaled, 8)

    def test_tile_maps_new_walker_w_to_source_w_mod_nwalkers(self):
        f = self._build("tile")
        self._check(f, lambda w, nw: w % nw)

    def test_repeat_maps_new_walker_w_to_source_w_over_factor(self):
        f = self._build("repeat")
        self._check(f, lambda w, nw: w // f)

    def test_factor_three_works(self):
        f = self._build("tile", 3)
        self._check(f, lambda w, nw: w % nw)

    def test_nwalkers_attrs_are_updated_everywhere(self):
        self._build()
        with h5py.File(self.dst, "r") as b:
            self.assertEqual(int(b["global_fit"].attrs["nwalkers"]), NW * 2)
            for br in b["global_fit/sub_backend"]:
                self.assertEqual(
                    int(b[f"global_fit/sub_backend/{br}"].attrs["nwalkers"]),
                    NW * 2, br)

    def test_iteration_and_group_attrs_survive(self):
        self._build()
        with h5py.File(self.dst, "r") as b:
            g = b["global_fit"]
            self.assertEqual(int(g.attrs["iteration"]), IT)
            self.assertEqual(int(g.attrs["ntemps"]), NT)
            self.assertEqual(int(g["nleaves_max"].attrs["mbh"]), MBH_NL)
            self.assertEqual(int(g["ndims"].attrs["gb"]), GB_ND)

    def test_the_step_axis_stays_growable(self):
        """The run resumes by calling grow() on these datasets."""
        self._build()
        with h5py.File(self.dst, "r") as b:
            self.assertIsNone(b["global_fit/chain/gb"].maxshape[0])

    def test_the_walker_axis_bound_is_scaled_with_its_extent(self):
        """A fixed maxshape on the walker axis rejects the doubled write."""
        self._build()
        with h5py.File(self.dst, "r") as b:
            d = b["global_fit/chain/gb"]
            self.assertEqual(d.maxshape[3], NW * 2)

    def test_rows_past_the_iteration_are_not_copied(self):
        """Only live rows are carried; the tail stays at the fill value."""
        self._build()
        with h5py.File(self.dst, "r") as b:
            tail = np.asarray(b["global_fit/log_like"][IT:])
            self.assertTrue(np.all(tail == 0.0),
                            "rows past `iteration` should be untouched")

    def test_a_refusal_prevents_any_write(self):
        with h5py.File(self.src, "a") as f:
            f["global_fit/sub_backend/gb"].create_dataset(
                "mystery", (NSTEPS, NW, GB_NB), dtype="f8")
        with self.assertRaises(RuntimeError):
            rsw.build_rescaled(self.src, self.dst, "global_fit", 2, "tile")


def _add_current_gb_tables(path, nw=NW):
    """The per-(walker, band) search tables and ``saved_after`` of a v9
    store (the 6mo production store refused on these until 2026-10-07)."""
    with h5py.File(path, "a") as f:
        g = f["global_fit"]
        gb = g["sub_backend/gb"]
        for i, name in enumerate(rsw.PER_WALKER_BAND_TABLES):
            _fill(gb.create_dataset(name, (NSTEPS, nw, GB_NB), dtype="f8"),
                  60 + i)
        _fill(gb.create_dataset("band_shutoff_w_step", (NSTEPS, 1),
                                dtype="i8"), 80)
        # the valve's window-rule stamp (2026-10-13): the first save on the
        # new code creates it on every v9 store
        _fill(gb.create_dataset("band_window_rule", (NSTEPS, 1), dtype="i8"), 82)
        _fill(gb.create_dataset("band_stage", (NSTEPS, GB_NB), dtype="i8"), 81)
        ds = g.create_dataset("saved_after", (IT,), maxshape=(None,),
                              dtype=h5py.string_dtype(encoding="utf-8"))
        ds[:] = [f"leg{i}" for i in range(IT)]


class CurrentLayoutTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.src = make_store(os.path.join(self.tmp, "s_testing.h5"))
        _add_current_gb_tables(self.src)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_v9_per_walker_tables_and_saved_after_are_classified(self):
        with h5py.File(self.src, "r") as f:
            entries, refusals, _, _ = rsw.plan_rescale(f, "global_fit", 2)
        self.assertEqual(refusals, [])
        axes = {e.path: e.axis for e in entries}
        for name in rsw.PER_WALKER_BAND_TABLES:
            self.assertEqual(axes[f"sub_backend/gb/{name}"], 1, name)
        for name in ("band_shutoff_w_step", "band_window_rule", "band_stage"):
            self.assertIsNone(axes[f"sub_backend/gb/{name}"], name)
        self.assertIsNone(axes["saved_after"])

    def test_the_valve_stamps_stay_walker_free_on_a_one_walker_store(self):
        """A ``(rows, 1)`` stamp carries an axis of length nwalkers when
        nwalkers is 1, so only its signature entry keeps it from being read
        as a walker table (or refused): the step stamp and, since
        2026-10-13, the window-rule stamp."""
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        src = make_store(os.path.join(tmp, "one_testing.h5"), nw=1)
        _add_current_gb_tables(src, nw=1)
        with h5py.File(src, "r") as f:
            entries, refusals, _, _ = rsw.plan_rescale(f, "global_fit", 2)
        axes = {e.path: e.axis for e in entries}
        for name in ("band_shutoff_w_step", "band_window_rule"):
            self.assertEqual([r for r in refusals if f"/{name}:" in r], [], name)
            self.assertIn(f"sub_backend/gb/{name}", axes, name)
            self.assertIsNone(axes[f"sub_backend/gb/{name}"], name)

    def test_the_table_list_matches_the_state_module(self):
        from lisatools.globalfit import state as S

        want = (set(S.SEARCH_STAGE_FIELDS) | set(S.SEARCH_SHUTOFF_WINDOW_FIELDS)
                | {S.SEARCH_SHUTOFF_FIELDS[0]})
        self.assertEqual(set(rsw.PER_WALKER_BAND_TABLES), want)


class LagBuildTest(unittest.TestCase):
    """PE scale-up from the chain (Mike 2026-10-07): new walker
    ``k = j * nw + i`` is source walker ``(i + j) % nw`` taken ``j`` saved
    rows earlier, for every row-wise walker-axis dataset."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.src = make_store(os.path.join(self.tmp, "s_testing.h5"))
        _add_current_gb_tables(self.src)
        self.dst = os.path.join(self.tmp, "out.h5")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _check(self, factor, **kw):
        rsw.build_rescaled(self.src, self.dst, "global_fit", factor, "lag", **kw)
        n_lagged = 0
        with h5py.File(self.src, "r") as a, h5py.File(self.dst, "r") as b:
            entries, _, _, it = rsw.plan_rescale(a, "global_fit", factor)
            for e in entries:
                p = f"global_fit/{e.path}"
                da, db = a[p], b[p]
                if e.axis is None:
                    if da.dtype.kind == "O":
                        self.assertEqual(list(da[:]), list(db[:]), e.path)
                    continue
                nw = da.shape[e.axis]
                for k in range(nw * factor):
                    j, i = divmod(k, nw)
                    for r in (range(it) if e.rowwise else [None]):
                        sa = [slice(None)] * da.ndim
                        sb = [slice(None)] * da.ndim
                        if r is not None:
                            sa[0], sb[0] = max(r - j, 0), r
                        sa[e.axis], sb[e.axis] = (i + j) % nw, k
                        self.assertTrue(
                            _eq(da[tuple(sa)], db[tuple(sb)]),
                            f"{e.path}: new walker {k}, row {r}")
                n_lagged += e.rowwise
        self.assertGreater(n_lagged, 15)

    def test_two_copies_take_the_last_two_rows(self):
        self._check(2)

    def test_three_copies_rotate_and_lag_by_one_more_each(self):
        self._check(3)

    def test_four_walkers_rotate_the_right_way(self):
        """nw = 4 (production): with 2 walkers a rotation by +k and -k
        coincide, so only nw >= 3 pins the direction."""
        os.remove(self.src)
        self.src = make_store(self.src, nw=4)
        _add_current_gb_tables(self.src, nw=4)
        self._check(3)

    def test_blocks_that_split_the_walker_axis(self):
        """A tiny buffer splits every block along the walker axis: the
        per-slot path, not the in-memory rotation."""
        os.remove(self.src)
        self.src = make_store(self.src, nw=4)
        _add_current_gb_tables(self.src, nw=4)
        self._check(2, cap_bytes=64)

    def test_too_few_rows_refuses(self):
        with self.assertRaises(RuntimeError):
            rsw.build_rescaled(self.src, self.dst, "global_fit", IT + 1, "lag")

    def test_cli_wants_a_multiple_of_four(self):
        import contextlib
        import io

        run = os.path.join(self.tmp, "run")
        os.makedirs(run)
        os.replace(self.src, os.path.join(run, "s_testing.h5"))
        out = os.path.join(self.tmp, "run8")
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(rsw.main([run, out, "--nwalkers", "6"]), 2)
            self.assertEqual(rsw.main([run, out, "--nwalkers", "8"]), 0)  # dry run
            # 2 -> 16 walkers needs 8 rows; the store has IT = 4
            self.assertEqual(rsw.main([run, out, "--nwalkers", "16"]), 3)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
