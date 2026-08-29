"""Tests for ``scripts/fstat_proposal/compact_gf_store.py``.

The synthetic store below reproduces the SHAPE of a real global-fit store,
not its size: a ``global_fit`` group with ``recipe/*`` status attrs, an
``inds/<branch>`` step-axis dataset to size the row count from, a
multi-dimensional step-axis chain whose step-axis CHUNK SPANS SEVERAL ROWS
(the whole reason the tool exists -- a partially live chunk cannot be
reclaimed by resize or h5repack), the ``band_edges`` / ``cap_edges``
statics that a naive ``shape[0] > iteration`` rule would destroy, and a
``sub_backend`` subgroup.

It also carries two adversarial datasets the real store's traps generalize
to: ``short_static`` (first axis larger than the iteration but smaller than
the row count -- the ``band_edges`` trap) and ``coincident_static`` (first
axis EQUAL to the row count but with a fixed ``maxshape``, so only the
resizability guard tells it apart from a real step dataset).

The real 570 MB store is never touched here; it is validated by hand once.
"""

import importlib.util
import itertools
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "fstat_proposal" / "compact_gf_store.py"

if not _SCRIPT.exists():        # e.g. installed without the scripts/ tree
    raise unittest.SkipTest(f"{_SCRIPT} is not present")

_spec = importlib.util.spec_from_file_location("compact_gf_store", _SCRIPT)
cgs = importlib.util.module_from_spec(_spec)
# registered BEFORE exec: @dataclasses.dataclass resolves annotations through
# sys.modules[cls.__module__] and dies on a module that is not there yet.
sys.modules["compact_gf_store"] = cgs
_spec.loader.exec_module(cgs)


NROWS = 40          # allocated step extent (the "2054")
ITER = 5            # live rows 0..4     (the rewind target)
NWALK = 4
NTEMP = 2
NLEAVES = 20
NDIM = 3
NBANDS = 16         # band_edges is (NBANDS + 1,) -- length 17, > ITER


def _make_store(path, iteration=ITER, nrows=NROWS):
    """Write a synthetic global-fit store shaped like the real one."""
    rng = np.random.default_rng(20260829)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = np.int64(iteration)
        g.attrs["nwalkers"] = np.int64(NWALK)
        g.attrs["ntemps"] = np.int64(1)
        g.attrs["nbranches"] = np.int64(1)
        g.attrs["gf_format_version"] = np.int64(2)
        g.attrs["has_recipe"] = np.True_
        g.attrs["has_blobs"] = np.False_
        # object array of str, exactly like the real branch_names
        g.attrs["branch_names"] = np.array(["gb"], dtype=object)
        # uint32 array, exactly like random_state_1
        g.attrs["random_state_0"] = "MT19937"
        g.attrs["random_state_1"] = rng.integers(
            0, 2**32, size=624, dtype=np.uint32)

        rec = g.create_group("recipe")
        for name, order, status in (("noise_search", 1, True),
                                    ("gb_search", 3, False)):
            r = rec.create_group(name)
            r.attrs["order num"] = np.int64(order)
            r.attrs["status"] = np.bool_(status)

        ds = g.create_group("domain_settings")
        ds.attrs["class_name"] = "WDMSettings"
        ds.create_group("kwargs").attrs["min_freq"] = np.float64(1e-4)

        inds = g.create_group("inds")
        d = inds.create_dataset(
            "gb", shape=(nrows, 1, 1, NWALK, NLEAVES), dtype=bool,
            maxshape=(None, 1, 1, NWALK, NLEAVES),
            chunks=(8, 1, 1, NWALK, NLEAVES), compression="gzip",
            compression_opts=4)
        d[:] = rng.random((nrows, 1, 1, NWALK, NLEAVES)) > 0.5

        chain = g.create_group("chain")
        d = chain.create_dataset(
            "gb", shape=(nrows, 1, 1, NWALK, NLEAVES, NDIM), dtype=np.float64,
            maxshape=(None, 1, 1, NWALK, NLEAVES, NDIM),
            chunks=(8, 1, 1, 2, 10, 1), compression="gzip",
            compression_opts=4)
        d[:] = rng.standard_normal((nrows, 1, 1, NWALK, NLEAVES, NDIM))

        d = g.create_dataset(
            "log_like", shape=(nrows, 1, 1, NWALK), dtype=np.float64,
            maxshape=(None, 1, 1, NWALK), chunks=(16, 1, 1, 2),
            compression="gzip", compression_opts=4)
        d[:] = rng.standard_normal((nrows, 1, 1, NWALK))

        # first axis 1, fixed maxshape -- never a step dataset
        d = g.create_dataset("accepted", shape=(1, 1, NWALK),
                             dtype=np.float64, chunks=(1, 1, NWALK))
        d[:] = rng.standard_normal((1, 1, NWALK))

        sb = g.create_group("sub_backend")
        gb = sb.create_group("gb")
        gb.attrs["ndim"] = np.int64(NDIM)
        gb.attrs["nleaves_max"] = np.int64(NLEAVES)
        gb.attrs["num_bands"] = np.int64(NBANDS)

        # THE POINT: step-axis chunk spans 8 rows, so the chunk covering
        # rows 0..7 is partially live at iteration 5.
        d = gb.create_dataset(
            "chain", shape=(nrows, NTEMP, NWALK, NLEAVES, NDIM),
            dtype=np.float64,
            maxshape=(None, NTEMP, NWALK, NLEAVES, NDIM),
            chunks=(8, 1, 2, 10, 1), compression="gzip", compression_opts=4)
        d[:] = rng.standard_normal(
            (nrows, NTEMP, NWALK, NLEAVES, NDIM))

        d = gb.create_dataset(
            "band_temps", shape=(nrows, NBANDS, NTEMP), dtype=np.float64,
            maxshape=(None, NBANDS, NTEMP), chunks=(8, NBANDS, NTEMP),
            compression="gzip", compression_opts=4)
        d[:] = rng.standard_normal((nrows, NBANDS, NTEMP))

        # GBState.static_names -- copied WHOLE, never truncated
        gb.create_dataset("band_edges", data=np.linspace(1e-4, 1e-2,
                                                         NBANDS + 1))
        gb.create_dataset("cap_edges", data=np.linspace(1e-4, 1e-2,
                                                       NBANDS + 1))
        # first axis == nrows but NOT resizable: only the maxshape guard
        # tells it apart from a step dataset
        gb.create_dataset("coincident_static",
                          data=rng.standard_normal(nrows))


def _sidecars(store):
    base, _ = os.path.splitext(store)
    return (base + "_running_backup_copy.h5",
            base + "_midit_checkpoint.pkl",
            os.path.join(os.path.dirname(store), "gb_fstat_fit"))


def _make_sidecars(store):
    backup, midit, fstat = _sidecars(store)
    shutil.copyfile(store, backup)
    with open(midit, "wb") as fh:
        fh.write(b"pickled-midit-state")
    os.makedirs(os.path.join(fstat, "shared", "epoch_0000"), exist_ok=True)
    with open(os.path.join(fstat, "shared", "clock.json"), "w") as fh:
        fh.write('{"clock": 66}')
    return backup, midit, fstat


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="compact_gf_store_")
        self.store = os.path.join(self.tmp, "gf_run_testing.h5")
        _make_store(self.store)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *argv):
        return cgs.main([self.store, *argv])


class BlockIteratorTest(unittest.TestCase):
    """The copier must never materialize more than the buffer cap."""

    def test_blocks_cover_rows_exactly_once(self):
        shape = (40, 3, 5)
        seen = np.zeros((7, 3, 5), dtype=int)
        for sl in cgs.iter_blocks(shape, 8, cap_bytes=1 << 20, n_rows=7):
            seen[sl] += 1
        self.assertTrue((seen == 1).all())

    def test_blocks_respect_cap_by_splitting_inside_a_row(self):
        # one row is 24 * 24 * 10000 * 9 * 8 B ~ 414 MB, like gb/chain
        shape = (2054, 24, 24, 10000, 9)
        cap = 64 * 1024**2
        blocks = list(cgs.iter_blocks(shape, 8, cap_bytes=cap, n_rows=2))
        self.assertGreater(len(blocks), 2, "a 414 MB row must be split")
        for sl in blocks:
            n = 1
            for s, dim in zip(sl, shape):
                n *= len(range(*s.indices(dim)))
            self.assertLessEqual(n * 8, cap)

    def test_zero_sized_dataset_yields_no_blocks(self):
        self.assertEqual(
            list(cgs.iter_blocks((1, 0), 8, cap_bytes=1 << 20, n_rows=1)), [])

    def test_chunk_aligned_blocks_touch_each_chunk_exactly_once(self):
        """The optimization's whole claim: one compress pass per chunk."""
        shape = (2054, 24, 24, 10000, 9)
        chunks = (32, 2, 2, 625, 1)          # the real gb/chain geometry
        cap = 64 * 1024**2
        touched = {}
        n = 0
        for sl in cgs.iter_blocks(shape, 8, cap, n_rows=5, chunks=chunks):
            size = 1
            for s, dim in zip(sl, shape):
                size *= len(range(*s.indices(dim)))
            self.assertLessEqual(size * 8, cap)
            n += 1
            grids = [range(s.start // c, (s.stop - 1) // c + 1)
                     for s, c in zip(sl, chunks)]
            for key in itertools.product(*grids):
                touched[key] = touched.get(key, 0) + 1
        self.assertGreater(n, 1)
        self.assertTrue(touched)
        self.assertEqual(set(touched.values()), {1},
                         "a chunk written by more than one block would be "
                         "decompressed and recompressed again")
        # rows 0..4 all live in step-chunk 0
        self.assertEqual({k[0] for k in touched}, {0})

    def test_falls_back_when_a_chunk_column_exceeds_the_cap(self):
        """Alignment is an optimization; coverage must survive without it."""
        shape, chunks = (40, 8, 100), (8, 4, 50)
        # one chunk-column of 5 live rows is 5*4*50*8 = 8000 B
        blocks = list(cgs.iter_blocks(shape, 8, 4000, n_rows=5,
                                      chunks=chunks))
        seen = np.zeros((5, 8, 100), dtype=int)
        for sl in blocks:
            seen[sl] += 1
        self.assertTrue((seen == 1).all())


class ProjectionTest(unittest.TestCase):
    """The dry run's size prediction, which is the number people act on."""

    def test_partly_kept_chunk_row_shrinks_by_occupancy(self):
        # one 32-row step-chunk row, all 32 rows written, keeping 5
        est, bound = cgs._project_new_bytes({0: 3200}, 32, keep_rows=5,
                                            written_rows=32)
        self.assertEqual(bound, 3200)
        self.assertEqual(est, 500)

    def test_chunk_rows_past_the_kept_range_are_dropped(self):
        est, bound = cgs._project_new_bytes({0: 1000, 1: 2000}, 32,
                                            keep_rows=5, written_rows=57)
        self.assertEqual(bound, 1000)          # chunk row 1 is not touched
        self.assertEqual(est, 156)             # 1000 * 5/32

    def test_fully_kept_chunk_row_keeps_everything(self):
        self.assertEqual(
            cgs._project_new_bytes({0: 1000}, 32, keep_rows=32,
                                   written_rows=32), (1000, 1000))

    def test_pure_fill_chunk_row_is_charged_in_full(self):
        """Nothing was written there, so there is no shrinkage to credit."""
        self.assertEqual(
            cgs._project_new_bytes({0: 40}, 32, keep_rows=5, written_rows=0),
            (40, 40))


class ClassificationTest(_StoreCase):
    """Step datasets vs. everything else."""

    def test_row_count_and_step_classification(self):
        with h5py.File(self.store, "r") as f:
            plan = cgs.plan_compaction(f, "global_fit", ITER)
        self.assertEqual(plan.n_rows, NROWS)
        step = {p.name for p in plan.datasets if p.is_step}
        whole = {p.name for p in plan.datasets if not p.is_step}
        self.assertEqual(step, {
            "global_fit/inds/gb", "global_fit/chain/gb",
            "global_fit/log_like", "global_fit/sub_backend/gb/chain",
            "global_fit/sub_backend/gb/band_temps"})
        self.assertEqual(whole, {
            "global_fit/accepted",
            "global_fit/sub_backend/gb/band_edges",
            "global_fit/sub_backend/gb/cap_edges",
            "global_fit/sub_backend/gb/coincident_static"})

    def test_statics_are_not_classified_by_length(self):
        """band_edges is (NBANDS+1,) = 17 > ITER = 5 -- the exact trap."""
        with h5py.File(self.store, "r") as f:
            plan = cgs.plan_compaction(f, "global_fit", ITER)
        by_name = {p.name: p for p in plan.datasets}
        be = by_name["global_fit/sub_backend/gb/band_edges"]
        self.assertGreater(be.old_rows, ITER)
        self.assertFalse(be.is_step)
        self.assertEqual(be.new_rows, NBANDS + 1)

    def test_coincident_length_static_saved_by_maxshape_guard(self):
        with h5py.File(self.store, "r") as f:
            plan = cgs.plan_compaction(f, "global_fit", ITER)
        by_name = {p.name: p for p in plan.datasets}
        cs = by_name["global_fit/sub_backend/gb/coincident_static"]
        self.assertEqual(cs.old_rows, NROWS)
        self.assertFalse(cs.is_step)
        self.assertEqual(cs.new_rows, NROWS)

    def test_rejects_out_of_range_iteration(self):
        with h5py.File(self.store, "r") as f:
            with self.assertRaises(ValueError):
                cgs.plan_compaction(f, "global_fit", NROWS + 1)
            with self.assertRaises(ValueError):
                cgs.plan_compaction(f, "global_fit", 0)


class BuildTest(_StoreCase):
    """The rebuilt file's structure and contents."""

    def setUp(self):
        super().setUp()
        self.out = os.path.join(self.tmp, "out.h5")
        cgs.build_compacted(self.store, self.out, "global_fit", ITER,
                            cap_bytes=1 << 16)

    def test_step_datasets_truncated_to_iteration(self):
        with h5py.File(self.out, "r") as f:
            for name in ("global_fit/inds/gb", "global_fit/chain/gb",
                         "global_fit/log_like",
                         "global_fit/sub_backend/gb/chain",
                         "global_fit/sub_backend/gb/band_temps"):
                self.assertEqual(f[name].shape[0], ITER, name)

    def test_statics_copied_whole(self):
        with h5py.File(self.store, "r") as a, h5py.File(self.out, "r") as b:
            for name in ("global_fit/sub_backend/gb/band_edges",
                         "global_fit/sub_backend/gb/cap_edges",
                         "global_fit/sub_backend/gb/coincident_static",
                         "global_fit/accepted"):
                self.assertEqual(a[name].shape, b[name].shape, name)
                np.testing.assert_array_equal(a[name][...], b[name][...])

    def test_kept_rows_are_bit_identical(self):
        with h5py.File(self.store, "r") as a, h5py.File(self.out, "r") as b:
            for name in ("global_fit/inds/gb", "global_fit/chain/gb",
                         "global_fit/log_like",
                         "global_fit/sub_backend/gb/chain",
                         "global_fit/sub_backend/gb/band_temps"):
                self.assertEqual(a[name][:ITER].tobytes(),
                                 b[name][:ITER].tobytes(), name)

    def test_maxshape_step_axis_still_none(self):
        with h5py.File(self.out, "r") as f:
            for name in ("global_fit/inds/gb", "global_fit/chain/gb",
                         "global_fit/log_like",
                         "global_fit/sub_backend/gb/chain",
                         "global_fit/sub_backend/gb/band_temps"):
                self.assertIsNone(f[name].maxshape[0], name)

    def test_backend_can_still_grow_the_step_axis(self):
        with h5py.File(self.out, "a") as f:
            d = f["global_fit/sub_backend/gb/chain"]
            d.resize(ITER + 10, axis=0)
            self.assertEqual(d.shape[0], ITER + 10)

    def test_chunks_dtypes_and_filters_preserved(self):
        with h5py.File(self.store, "r") as a, h5py.File(self.out, "r") as b:
            for name in ("global_fit/sub_backend/gb/chain",
                         "global_fit/chain/gb", "global_fit/inds/gb",
                         "global_fit/sub_backend/gb/band_edges"):
                self.assertEqual(a[name].chunks, b[name].chunks, name)
                self.assertEqual(a[name].dtype, b[name].dtype, name)
                self.assertEqual(a[name].compression,
                                 b[name].compression, name)
                self.assertEqual(a[name].compression_opts,
                                 b[name].compression_opts, name)
                self.assertEqual(
                    a[name].id.get_create_plist().get_nfilters(),
                    b[name].id.get_create_plist().get_nfilters(), name)

    def test_groups_and_names_match(self):
        def names(p):
            out = set()
            with h5py.File(p, "r") as f:
                f.visititems(lambda n, o: out.add(
                    (n, "D" if isinstance(o, h5py.Dataset) else "G")))
            return out
        self.assertEqual(names(self.store), names(self.out))

    def test_attrs_preserved(self):
        with h5py.File(self.store, "r") as a, h5py.File(self.out, "r") as b:
            for name in ("global_fit", "global_fit/recipe/gb_search",
                         "global_fit/recipe/noise_search",
                         "global_fit/domain_settings",
                         "global_fit/sub_backend/gb"):
                self.assertEqual(set(a[name].attrs), set(b[name].attrs), name)
                for k in a[name].attrs:
                    np.testing.assert_array_equal(
                        np.asarray(a[name].attrs[k]),
                        np.asarray(b[name].attrs[k]),
                        err_msg=f"{name}.{k}")
            self.assertEqual(int(b["global_fit/recipe/gb_search"]
                                 .attrs["order num"]), 3)
            self.assertFalse(bool(b["global_fit/recipe/gb_search"]
                                  .attrs["status"]))
            self.assertTrue(bool(b["global_fit/recipe/noise_search"]
                                 .attrs["status"]))

    def test_iteration_attr_matches_new_extent(self):
        with h5py.File(self.out, "r") as f:
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), ITER)

    def test_explicit_iteration_override_rewrites_the_attr(self):
        out2 = os.path.join(self.tmp, "out2.h5")
        cgs.build_compacted(self.store, out2, "global_fit", 3,
                            cap_bytes=1 << 16)
        with h5py.File(out2, "r") as f:
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 3)
            self.assertEqual(f["global_fit/sub_backend/gb/chain"].shape[0], 3)

    def test_new_file_is_smaller(self):
        self.assertLess(os.path.getsize(self.out),
                        os.path.getsize(self.store))

    def test_rebuild_never_exceeds_the_projected_bound(self):
        with h5py.File(self.store, "r") as f:
            plan = cgs.plan_compaction(f, "global_fit", ITER)
        actual = 0

        def add(_name, obj):
            nonlocal actual
            if isinstance(obj, h5py.Dataset):
                actual += obj.id.get_storage_size()

        with h5py.File(self.out, "r") as f:
            f.visititems(add)
        self.assertLessEqual(actual, plan.bound_new_bytes)


class VerifyTest(_StoreCase):
    """Verification must refuse a copy that is wrong in any way."""

    def setUp(self):
        super().setUp()
        self.out = os.path.join(self.tmp, "out.h5")
        cgs.build_compacted(self.store, self.out, "global_fit", ITER,
                            cap_bytes=1 << 16)

    def test_clean_copy_verifies(self):
        self.assertEqual(
            cgs.verify(self.store, self.out, "global_fit", ITER,
                       cap_bytes=1 << 16), [])

    def test_detects_corrupted_row(self):
        with h5py.File(self.out, "a") as f:
            f["global_fit/sub_backend/gb/chain"][0, 0, 0, 0, 0] += 1.0
        problems = cgs.verify(self.store, self.out, "global_fit", ITER,
                              cap_bytes=1 << 16)
        self.assertTrue(any("chain" in p for p in problems), problems)

    def test_detects_corrupted_middle_row(self):
        """5 live rows inside an 8-row step chunk: every row is compared."""
        with h5py.File(self.out, "a") as f:
            f["global_fit/chain/gb"][2, 0, 0, 0, 0, 0] += 7.0
        problems = cgs.verify(self.store, self.out, "global_fit", ITER,
                              cap_bytes=1 << 16)
        self.assertTrue(any("chain/gb" in p for p in problems), problems)

    def test_detects_corrupted_last_row(self):
        with h5py.File(self.out, "a") as f:
            f["global_fit/log_like"][ITER - 1, 0, 0, 0] = 12345.0
        problems = cgs.verify(self.store, self.out, "global_fit", ITER,
                              cap_bytes=1 << 16)
        self.assertTrue(any("log_like" in p for p in problems), problems)

    def test_detects_truncated_static(self):
        """The band_edges trap: a static shortened to the iteration."""
        bad = os.path.join(self.tmp, "bad.h5")
        shutil.copyfile(self.out, bad)
        with h5py.File(bad, "a") as f:
            data = f["global_fit/sub_backend/gb/band_edges"][:ITER]
            del f["global_fit/sub_backend/gb/band_edges"]
            f["global_fit/sub_backend/gb"].create_dataset(
                "band_edges", data=data)
        problems = cgs.verify(self.store, bad, "global_fit", ITER,
                              cap_bytes=1 << 16)
        self.assertTrue(any("band_edges" in p for p in problems), problems)

    def test_detects_missing_dataset(self):
        bad = os.path.join(self.tmp, "bad.h5")
        shutil.copyfile(self.out, bad)
        with h5py.File(bad, "a") as f:
            del f["global_fit/sub_backend/gb/cap_edges"]
        problems = cgs.verify(self.store, bad, "global_fit", ITER,
                              cap_bytes=1 << 16)
        self.assertTrue(any("cap_edges" in p for p in problems), problems)

    def test_detects_changed_attr(self):
        bad = os.path.join(self.tmp, "bad.h5")
        shutil.copyfile(self.out, bad)
        with h5py.File(bad, "a") as f:
            f["global_fit/recipe/noise_search"].attrs["status"] = np.False_
        problems = cgs.verify(self.store, bad, "global_fit", ITER,
                              cap_bytes=1 << 16)
        self.assertTrue(any("status" in p for p in problems), problems)

    def test_detects_frozen_step_axis(self):
        bad = os.path.join(self.tmp, "bad.h5")
        with h5py.File(self.out, "r") as a, h5py.File(bad, "w") as b:
            a.copy("global_fit", b, "global_fit")
            d = b["global_fit/log_like"]
            data = d[...]
            del b["global_fit/log_like"]
            b["global_fit"].create_dataset("log_like", data=data)  # fixed
        problems = cgs.verify(self.store, bad, "global_fit", ITER,
                              cap_bytes=1 << 16)
        self.assertTrue(any("maxshape" in p or "resiz" in p
                            for p in problems), problems)


class CliTest(_StoreCase):
    """Dry run, apply, swap, and the sidecar flags."""

    def _snapshot(self):
        return {p: (os.path.getsize(os.path.join(self.tmp, p)),
                    os.path.getmtime(os.path.join(self.tmp, p)))
                for p in sorted(os.listdir(self.tmp))}

    def test_dry_run_writes_nothing(self):
        before = self._snapshot()
        self.assertEqual(self._run(), 0)
        self.assertEqual(self._snapshot(), before)

    def test_dry_run_reports_each_dataset_and_a_total(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run()
        out = buf.getvalue()
        self.assertIn("sub_backend/gb/chain", out)
        self.assertIn("band_edges", out)
        self.assertIn("DRY RUN", out)
        self.assertRegex(out, r"40\s*->\s*5")       # old rows -> new rows
        self.assertRegex(out, r"(?i)total")

    def test_apply_swaps_and_keeps_the_original(self):
        old_size = os.path.getsize(self.store)
        self.assertEqual(self._run("--apply"), 0)
        kept = [p for p in os.listdir(self.tmp) if ".pre-compact-" in p]
        self.assertEqual(len(kept), 1, os.listdir(self.tmp))
        self.assertEqual(os.path.getsize(os.path.join(self.tmp, kept[0])),
                         old_size)
        self.assertLess(os.path.getsize(self.store), old_size)
        with h5py.File(self.store, "r") as f:
            self.assertEqual(f["global_fit/sub_backend/gb/chain"].shape[0],
                             ITER)
            self.assertEqual(
                f["global_fit/sub_backend/gb/band_edges"].shape[0],
                NBANDS + 1)

    def test_apply_leaves_no_temp_file(self):
        self._run("--apply")
        self.assertFalse([p for p in os.listdir(self.tmp)
                          if "compacting" in p], os.listdir(self.tmp))

    def test_sidecars_untouched_without_flags(self):
        backup, midit, fstat = _make_sidecars(self.store)
        self._run("--apply")
        self.assertTrue(os.path.exists(backup))
        self.assertTrue(os.path.exists(midit))
        self.assertTrue(os.path.isdir(fstat))

    def test_reset_backup_moves_rather_than_deletes(self):
        backup, _, _ = _make_sidecars(self.store)
        size = os.path.getsize(backup)
        self._run("--apply", "--reset-backup")
        self.assertFalse(os.path.exists(backup))
        moved = [p for p in os.listdir(self.tmp)
                 if p.startswith(os.path.basename(backup) + ".")]
        self.assertEqual(len(moved), 1, os.listdir(self.tmp))
        self.assertEqual(os.path.getsize(os.path.join(self.tmp, moved[0])),
                         size)

    def test_reset_midit_moves_rather_than_deletes(self):
        _, midit, _ = _make_sidecars(self.store)
        self._run("--apply", "--reset-midit")
        self.assertFalse(os.path.exists(midit))
        moved = [p for p in os.listdir(self.tmp)
                 if p.startswith(os.path.basename(midit) + ".")]
        self.assertEqual(len(moved), 1, os.listdir(self.tmp))
        with open(os.path.join(self.tmp, moved[0]), "rb") as fh:
            self.assertEqual(fh.read(), b"pickled-midit-state")

    def test_reset_fstat_moves_the_whole_cache_dir(self):
        _, _, fstat = _make_sidecars(self.store)
        self._run("--apply", "--reset-fstat")
        self.assertFalse(os.path.exists(fstat))
        moved = [p for p in os.listdir(self.tmp)
                 if p.startswith("gb_fstat_fit.")]
        self.assertEqual(len(moved), 1, os.listdir(self.tmp))
        clock = os.path.join(self.tmp, moved[0], "shared", "clock.json")
        self.assertTrue(os.path.exists(clock))
        with open(clock) as fh:
            self.assertIn("66", fh.read())

    def test_sidecar_flags_are_dry_run_safe(self):
        backup, midit, fstat = _make_sidecars(self.store)
        self._run("--reset-backup", "--reset-midit", "--reset-fstat")
        self.assertTrue(os.path.exists(backup))
        self.assertTrue(os.path.exists(midit))
        self.assertTrue(os.path.isdir(fstat))

    def test_explicit_iteration_flag(self):
        self.assertEqual(self._run("--iteration", "2", "--apply"), 0)
        with h5py.File(self.store, "r") as f:
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 2)
            self.assertEqual(f["global_fit/sub_backend/gb/chain"].shape[0], 2)

    def test_torn_chunk_aborts_and_leaves_the_store_untouched(self):
        """A store killed mid-write has chunks that only fail when READ.

        Hit for real on 2026-08-29: a bad snapshot copy whose every
        gb/chain chunk raised "filter returned failure during read". The
        tool must refuse, clean up its temp file, and not touch the
        original.
        """
        with h5py.File(self.store, "r") as f:
            info = f["global_fit/sub_backend/gb/chain"].id.get_chunk_info(0)
        with open(self.store, "r+b") as fh:      # invalid deflate stream
            fh.seek(info.byte_offset)
            fh.write(b"\xff" * info.size)
        before = os.path.getsize(self.store)
        self.assertNotEqual(self._run("--apply"), 0)
        self.assertEqual(os.path.getsize(self.store), before)
        self.assertFalse([p for p in os.listdir(self.tmp)
                          if "compacting" in p or ".pre-compact-" in p],
                         os.listdir(self.tmp))

    def test_out_of_range_iteration_is_an_error(self):
        self.assertNotEqual(self._run("--iteration", "999", "--apply"), 0)
        with h5py.File(self.store, "r") as f:
            self.assertEqual(f["global_fit/sub_backend/gb/chain"].shape[0],
                             NROWS)


class OpenFileGuardTest(unittest.TestCase):
    """The lsof parse that keeps us off a file another process holds."""

    def test_other_pids_are_reported(self):
        pids = cgs.pids_holding(
            "/some/store.h5", _runner=lambda p: f"{os.getpid()}\n424242\n")
        self.assertEqual(pids, [424242])

    def test_our_own_pid_is_ignored(self):
        pids = cgs.pids_holding(
            "/some/store.h5", _runner=lambda p: f"{os.getpid()}\n")
        self.assertEqual(pids, [])

    def test_missing_lsof_is_not_fatal(self):
        def boom(_):
            raise FileNotFoundError("lsof")
        self.assertEqual(cgs.pids_holding("/some/store.h5", _runner=boom), [])


if __name__ == "__main__":
    unittest.main()
