"""Copy one leaf between walkers in a stopped store (scripts/fstat_proposal/copy_store_leaf.py).

User request 2026-10-10 (9mo production): EMRI leaf 3 on walker 0 sat in the
wrong mode with its whole ladder below its cold rung; copy walker 2's state of
that leaf, on every rung, into walker 0 while the run is down, then resume.

The store is written by the REAL ``GFHDFBackend`` + per-branch sub-backends in
the production layout (cold-chain main group at ntemps 1 with the sampler
axis, per-leaf ladders on the sub-backends, rows allocated past the
``iteration`` attr, mbh with ``nleaves_max == nwalkers``), and the resume
contract is checked through the real ``get_last_sample`` / ``check_cold_row``
/ ``_validate_resume_readable``. The copied slices are restated HERE,
independently of the tool's layout table, so a tool that skips a dataset
fails the byte-identity test rather than agreeing with itself.
"""

import contextlib
import hashlib
import io
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from lisatools.globalfit.hdfbackend import (
    EMRIHDFBackend,
    GBHDFBackend,
    GFHDFBackend,
    MBHHDFBackend,
    ModuleSubBackend,
    SOBBHHDFBackend,
    _validate_resume_readable,
)
from lisatools.globalfit.state import (
    EMRIState,
    GBState,
    GFState,
    MBHState,
    ModuleSubState,
    SOBBHState,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "scripts", "fstat_proposal"))

import copy_store_leaf as C  # noqa: E402

NW = 4            # walkers
NT = 3            # rungs on every sub-backend ladder
SAVED = 3         # rows written (the iteration attr)
ALLOC = 5         # rows allocated: production grows ahead of the iteration attr
LAST = SAVED - 1  # the row a resume reads
NUM_BANDS = 3
BAND_EDGES = np.linspace(1e-3, 4e-3, NUM_BANDS + 1)

BRANCH_SHAPES = {
    # branch: (nleaves_max, ndim); mbh has nleaves_max == nwalkers on purpose
    "gb": (5, 8),
    "mbh": (4, 11),
    "emri": (3, 12),
    "sobbh": (2, 11),
    "psd": (1, 2),
}
SUB_BACKENDS = {"gb": GBHDFBackend, "mbh": MBHHDFBackend, "emri": EMRIHDFBackend,
                "sobbh": SOBBHHDFBackend, "psd": ModuleSubBackend}
SUB_STATE_BASES = {"gb": GBState, "mbh": MBHState, "emri": EMRIState,
                   "sobbh": SOBBHState, "psd": ModuleSubState}
PER_LEAF = ("mbh", "emri", "sobbh")
LEAF_COUNT = {"mbh": "num_mbhs", "emri": "num_emris", "sobbh": "num_sobbhs"}

#: sobbh leaf 1 is dead on every (rung, walker): the --all-leaves skip case
DEAD = {"sobbh": 1}


def make_state(rng):
    """One saved row: every branch's tempered ensemble, distinct values everywhere."""
    coords = {b: rng.standard_normal((NT, NW, nl, nd)) for b, (nl, nd) in BRANCH_SHAPES.items()}
    inds = {b: np.ones((NT, NW, nl), dtype=bool) for b, (nl, _) in BRANCH_SHAPES.items()}
    inds["gb"][:, :, -1] = False
    for b, leaf in DEAD.items():
        inds[b][:, :, leaf] = False
    state = GFState(
        {b: c[:1].copy() for b, c in coords.items()},
        inds={b: i[:1].copy() for b, i in inds.items()},
        log_like=rng.standard_normal((1, NW)),
        log_prior=rng.standard_normal((1, NW)),
        betas=np.ones(1),
        random_state=np.random.get_state(),
        sub_state_bases=SUB_STATE_BASES,
    )
    state.sub_states["gb"].initialize_band_information(
        NW, NT, BAND_EDGES, np.tile(np.linspace(1.0, 0.3, NT), (NUM_BANDS, 1)))
    for b, sub in state.sub_states.items():
        nl, nd = BRANCH_SHAPES[b]
        if b in PER_LEAF:
            sub.betas_all = np.tile(1.0 / 1.2 ** np.arange(NT), (nl, 1)) * (1 + 0.01 * rng.random())
        sub.initialize_tempered(NT, NW, nl, nd, coords=coords[b], inds=inds[b])
        if b != "gb":
            sub.log_like[:] = 1e6 + 100.0 * rng.standard_normal(sub.log_like.shape)
            sub.log_prior[:] = rng.standard_normal(sub.log_prior.shape)
            for name in ("in_model_accepted", "in_model_proposed", "rj_accepted",
                         "rj_proposed", "swaps_accepted", "swaps_proposed"):
                getattr(sub, name)[:] = rng.integers(0, 50, getattr(sub, name).shape)
        sub.d_h[:] = rng.standard_normal(sub.d_h.shape)
        sub.h_h[:] = rng.standard_normal(sub.h_h.shape)
    return state


def make_backend(path):
    return GFHDFBackend(path, sub_backend=dict(SUB_BACKENDS),
                        sub_state_bases=dict(SUB_STATE_BASES))


def make_store(path, seed=20261010):
    rng = np.random.default_rng(seed)
    backend = make_backend(path)
    sub_kw = {b: dict(nleaves_max=nl, ndim=nd, ntemps=NT) for b, (nl, nd) in BRANCH_SHAPES.items()}
    sub_kw["gb"].update(num_bands=NUM_BANDS, band_edges=BAND_EDGES)
    for b in PER_LEAF:
        sub_kw[b][LEAF_COUNT[b]] = BRANCH_SHAPES[b][0]
    backend.reset(
        NW,
        {b: nd for b, (_, nd) in BRANCH_SHAPES.items()},
        nleaves_max={b: nl for b, (nl, _) in BRANCH_SHAPES.items()},
        ntemps=1,
        branch_names=list(BRANCH_SHAPES),
        nbranches=len(BRANCH_SHAPES),
        rj=False,
        moves=None,
        sub_reset_kwargs=sub_kw,
    )
    backend.grow(ALLOC, None)
    for _ in range(SAVED):
        backend.save_step(make_state(rng), np.ones((1, NW)))


def copied_slices(branch, leaf, src, dst, row):
    """The test's OWN statement of every (dataset, dst index, src index) the
    copy must write -- deliberately not read from the tool."""
    a = slice(None)
    return [
        (f"chain/{branch}", (row, a, a, dst, leaf), (row, a, a, src, leaf)),
        (f"inds/{branch}", (row, a, a, dst, leaf), (row, a, a, src, leaf)),
        (f"sub_backend/{branch}/chain", (row, a, dst, leaf), (row, a, src, leaf)),
        (f"sub_backend/{branch}/inds", (row, a, dst, leaf), (row, a, src, leaf)),
        (f"sub_backend/{branch}/log_like", (row, leaf, a, dst), (row, leaf, a, src)),
        (f"sub_backend/{branch}/log_prior", (row, leaf, a, dst), (row, leaf, a, src)),
        (f"sub_backend/{branch}/d_h", (row, dst, leaf), (row, src, leaf)),
        (f"sub_backend/{branch}/h_h", (row, dst, leaf), (row, src, leaf)),
    ]


def snapshot(path):
    """``(datasets, attrs)`` of the whole ``global_fit`` tree, by path."""
    data, attrs = {}, {}
    with h5py.File(path, "r") as f:
        root = f["global_fit"]
        attrs[""] = dict(root.attrs)

        def visit(name, obj):
            attrs[name] = dict(obj.attrs)
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[()]
        root.visititems(visit)
    return data, attrs


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="copyleaf_")
        self.path = os.path.join(self.tmp, "run_testing.h5")
        make_store(self.path)
        # lsof is slow on macOS and sees nothing here; the holder refusal has its own test
        p = mock.patch.object(C, "pids_holding", return_value=[])
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_tool(self, *args, store=None, apply=False):
        argv = ["--store", store or self.path] + [str(a) for a in args]
        if apply:
            argv += ["--apply", "--live-window", "0"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = C.main(argv)
        return rc, out.getvalue()

    def assert_only_copied(self, before, after, copies):
        """Every dataset equals ``before`` except the listed slices, which hold
        the BEFORE source values; every attr is unchanged."""
        data0, attrs0 = before
        data1, attrs1 = after
        self.assertEqual(set(data0), set(data1))
        expected = {k: np.array(v, copy=True) for k, v in data0.items()}
        for name, dsel, ssel in copies:
            expected[name][dsel] = data0[name][ssel]
        for name in sorted(data0):
            self.assertTrue(_equal(data1[name], expected[name]),
                            f"{name} differs from the expected post-copy contents")
        self.assertEqual(set(attrs0), set(attrs1))
        for obj in attrs0:
            self.assertEqual(set(attrs0[obj]), set(attrs1[obj]), obj)
            for k in attrs0[obj]:
                self.assertTrue(_equal(attrs0[obj][k], attrs1[obj][k]), f"{obj} attr {k}")


class FixtureTest(_Base):
    def test_the_fixture_is_the_production_layout(self):
        with h5py.File(self.path, "r") as f:
            g = f["global_fit"]
            self.assertEqual(int(g.attrs["iteration"]), SAVED)
            self.assertEqual(int(g.attrs["ntemps"]), 1)
            self.assertEqual(g["chain/emri"].shape, (ALLOC, 1, 1, NW, 3, 12))
            s = g["sub_backend/emri"]
            self.assertEqual(s["chain"].shape, (ALLOC, NT, NW, 3, 12))
            self.assertEqual(s["log_like"].shape, (ALLOC, 3, NT, NW))
            self.assertEqual(g["sub_backend/mbh/log_like"].shape, (ALLOC, NW, NT, NW))
            # the source walker's leaf genuinely differs from the destination's
            self.assertFalse(np.allclose(s["chain"][LAST, :, 0, 1], s["chain"][LAST, :, 2, 1]))
        _validate_resume_readable(self.path)


class CopyTest(_Base):
    def test_dry_run_writes_nothing(self):
        h0 = md5(self.path)
        rc, out = self.run_tool("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                                "--dst-walker", 0)
        self.assertEqual(rc, 0, out)
        self.assertIn("DRY RUN", out)
        self.assertIn("--apply", out)
        self.assertEqual(md5(self.path), h0)
        self.assertFalse(os.path.exists(self.path + C.BACKUP_SUFFIX))

    def test_apply_copies_the_leaf_on_every_rung_and_nothing_else(self):
        h0 = md5(self.path)
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                                "--dst-walker", 0, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("VERIFY OK", out)
        self.assert_only_copied(before, snapshot(self.path),
                                copied_slices("emri", 1, 2, 0, LAST))
        # the backup is the untouched original
        self.assertEqual(md5(self.path + C.BACKUP_SUFFIX), h0)

    def test_mbh_with_nleaves_equal_to_nwalkers(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "mbh", "--leaf", 3, "--leaf", 0,
                                "--src-walker", 1, "--dst-walker", 3, apply=True)
        self.assertEqual(rc, 0, out)
        self.assert_only_copied(before, snapshot(self.path),
                                copied_slices("mbh", 3, 1, 3, LAST)
                                + copied_slices("mbh", 0, 1, 3, LAST))

    def test_all_leaves_skips_a_leaf_dead_on_both_walkers(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "sobbh", "--all-leaves", "--src-walker", 3,
                                "--dst-walker", 1, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("skipped (dead on both walkers): leaves [1]", out)
        self.assert_only_copied(before, snapshot(self.path),
                                copied_slices("sobbh", 0, 3, 1, LAST))

    def test_explicit_row_edits_that_row_only(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "emri", "--leaf", 2, "--src-walker", 0,
                                "--dst-walker", 1, "--row", 0, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("is NOT the last row", out)
        self.assert_only_copied(before, snapshot(self.path),
                                copied_slices("emri", 2, 0, 1, 0))

    def test_the_resume_starts_the_destination_from_the_source(self):
        before = make_backend(self.path).get_last_sample()
        rc, out = self.run_tool("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                                "--dst-walker", 0, apply=True)
        self.assertEqual(rc, 0, out)
        _validate_resume_readable(self.path)
        state = make_backend(self.path).get_last_sample()   # what run.py loads
        sub, sub0 = state.sub_states["emri"], before.sub_states["emri"]
        # destination = the source's PRE-copy state, on every rung
        np.testing.assert_array_equal(sub.coords[:, 0, 1], sub0.coords[:, 2, 1])
        np.testing.assert_array_equal(sub.inds[:, 0, 1], sub0.inds[:, 2, 1])
        np.testing.assert_array_equal(sub.log_like[1, :, 0], sub0.log_like[1, :, 2])
        np.testing.assert_array_equal(state.branches["emri"].coords[0, 0, 1],
                                      before.branches["emri"].coords[0, 2, 1])
        # everything else the resume loads is as before
        np.testing.assert_array_equal(sub.coords[:, 1:], sub0.coords[:, 1:])
        np.testing.assert_array_equal(sub.coords[:, 0, [0, 2]], sub0.coords[:, 0, [0, 2]])
        np.testing.assert_array_equal(sub.betas_all, sub0.betas_all)
        np.testing.assert_array_equal(state.log_like, before.log_like)
        # the first propose's cold-row check passes on every branch
        for name, s in state.sub_states.items():
            s.check_cold_row(state, name)

    def test_verify_reports_before_and_after(self):
        args = ("--branch", "emri", "--leaf", 1, "--src-walker", 2, "--dst-walker", 0)
        h0 = md5(self.path)
        rc, out = self.run_tool(*args, "--verify")
        self.assertEqual(rc, 1, out)
        self.assertIn("VERIFY FAILED", out)
        self.assertEqual(md5(self.path), h0)
        self.assertEqual(self.run_tool(*args, apply=True)[0], 0)
        rc, out = self.run_tool(*args, "--verify")
        self.assertEqual(rc, 0, out)
        self.assertIn("VERIFY OK", out)

    def test_verify_catches_a_copy_that_skips_the_sub_backend_chain(self):
        """The mutation the user named: drop sub_backend/<b>/chain from the copy.
        The tool's own verify (cold-row agreement) and the independent slice
        spec must both see it."""
        real = C.sub_layout

        def skip_chain(attrs):
            out = dict(real(attrs))
            trailing, _, laxis = out["chain"]
            out["chain"] = (trailing, None, laxis)       # 'walker-free': never copied
            return out

        before = snapshot(self.path)
        with mock.patch.object(C, "sub_layout", skip_chain):
            with h5py.File(self.path, "r") as f:
                plan = C.plan_copy(f, "emri", [1], 2, 0)
            C.apply_copy(self.path, plan)
            problems = C.verify_copy(self.path, plan)
        self.assertTrue(any("cold row disagrees" in p for p in problems), problems)
        with self.assertRaises(AssertionError):
            self.assert_only_copied(before, snapshot(self.path),
                                    copied_slices("emri", 1, 2, 0, LAST))


class SidecarTest(_Base):
    def _twin(self):
        twin = os.path.splitext(self.path)[0] + "_running_backup_copy.h5"
        shutil.copyfile(self.path, twin)
        return twin

    def test_the_twin_is_left_alone_by_default(self):
        twin = self._twin()
        h = md5(twin)
        rc, out = self.run_tool("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                                "--dst-walker", 0, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("Read ONLY if the store is unreadable", out)
        self.assertEqual(md5(twin), h)

    def test_also_backup_edits_the_twin_too(self):
        twin = self._twin()
        before = snapshot(twin)
        rc, out = self.run_tool("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                                "--dst-walker", 0, "--also-backup", apply=True)
        self.assertEqual(rc, 0, out)
        self.assert_only_copied(before, snapshot(twin), copied_slices("emri", 1, 2, 0, LAST))
        self.assertTrue(os.path.exists(twin + C.BACKUP_SUFFIX))
        self.assertTrue(os.path.exists(self.path + C.BACKUP_SUFFIX))

    def test_midit_checkpoint_refused_unless_reset(self):
        midit = os.path.splitext(self.path)[0] + "_midit_checkpoint.pkl"
        with open(midit, "wb") as fp:
            pickle.dump({"meta": {}, "state": None}, fp)
        h0 = md5(self.path)
        args = ("--branch", "emri", "--leaf", 1, "--src-walker", 2, "--dst-walker", 0)
        rc, out = self.run_tool(*args, apply=True)
        self.assertEqual(rc, 2, out)
        self.assertIn("would override the edited store", out)
        self.assertEqual(md5(self.path), h0)
        rc, out = self.run_tool(*args, "--reset-midit", apply=True)
        self.assertEqual(rc, 0, out)
        self.assertFalse(os.path.exists(midit))
        self.assertTrue([n for n in os.listdir(self.tmp) if ".pkl.stale-" in n])

    def test_eigen_sidecar_is_reported_not_edited(self):
        side = os.path.splitext(self.path)[0] + C.EIGEN_SUFFIX
        payload = {"format": 3, "entries": {
            "emri:1": {"axes": np.eye(12), "sigmas": np.ones(12), "scope": "walker_max"},
        }}
        with open(side, "wb") as fp:
            pickle.dump(payload, fp)
        h = md5(side)
        rc, out = self.run_tool("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                                "--dst-walker", 0, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("one walker-independent table", out)
        self.assertEqual(md5(side), h)


class RefusalTest(_Base):
    def _refused(self, *args, apply=True, needle=None):
        h0 = md5(self.path)
        rc, out = self.run_tool(*args, apply=apply)
        self.assertEqual(rc, 2, out)
        self.assertIn("REFUSING", out)
        if needle:
            self.assertIn(needle, out)
        self.assertEqual(md5(self.path), h0)
        self.assertFalse(os.path.exists(self.path + C.BACKUP_SUFFIX))

    def test_banded_and_noise_branches(self):
        base = ("--leaf", 0, "--src-walker", 2, "--dst-walker", 0)
        self._refused("--branch", "gb", *base, needle="banded branch")
        self._refused("--branch", "vgb", *base, needle="banded branch")
        self._refused("--branch", "psd", *base, needle="noise branch")

    def test_bad_walkers_leaves_rows(self):
        self._refused("--branch", "emri", "--leaf", 1, "--src-walker", 2, "--dst-walker", 2,
                      needle="both 2")
        self._refused("--branch", "emri", "--leaf", 3, "--src-walker", 2, "--dst-walker", 0,
                      needle="outside emri's 3 leaf slots")
        self._refused("--branch", "emri", "--leaf", 1, "--src-walker", NW, "--dst-walker", 0,
                      needle="outside the store's 4 walkers")
        self._refused("--branch", "emri", "--leaf", 1, "--src-walker", 2, "--dst-walker", 0,
                      "--row", SAVED, needle="not a stored row")
        self._refused("--branch", "emri", "--src-walker", 2, "--dst-walker", 0,
                      needle="--all-leaves")

    def test_a_dead_source_leaf(self):
        self._refused("--branch", "sobbh", "--leaf", DEAD["sobbh"], "--src-walker", 2,
                      "--dst-walker", 0, needle="not alive on SOURCE walker 2")

    def test_a_torn_source_row(self):
        with h5py.File(self.path, "r+") as f:
            f["global_fit/sub_backend/emri/chain"][LAST, 0, 2, 1, 0] += 1.0
        self._refused("--branch", "emri", "--leaf", 1, "--src-walker", 2, "--dst-walker", 0,
                      needle="torn save")

    def test_an_unknown_sub_backend_dataset(self):
        with h5py.File(self.path, "r+") as f:
            f["global_fit/sub_backend/emri"].create_dataset("mystery", data=np.zeros((ALLOC, NW)))
        self._refused("--branch", "emri", "--leaf", 1, "--src-walker", 2, "--dst-walker", 0,
                      needle="not in this tool's layout table")

    def test_an_existing_backup(self):
        with open(self.path + C.BACKUP_SUFFIX, "w") as fp:
            fp.write("earlier backup")
        h0 = md5(self.path)
        rc, out = self.run_tool("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                                "--dst-walker", 0, apply=True)
        self.assertEqual(rc, 2, out)
        self.assertIn("already exists", out)
        self.assertEqual(md5(self.path), h0)

    def test_a_recently_written_store_needs_force(self):
        os.utime(self.path, None)
        h0 = md5(self.path)
        argv = ["--store", self.path, "--branch", "emri", "--leaf", 1, "--src-walker", 2,
                "--dst-walker", 0, "--apply"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = C.main([str(a) for a in argv])
        self.assertEqual(rc, 2, out.getvalue())
        self.assertIn("probably still writing", out.getvalue())
        self.assertEqual(md5(self.path), h0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(C.main([str(a) for a in argv] + ["--force"]), 0)

    def test_a_file_held_open_elsewhere(self):
        with h5py.File(self.path, "r"):
            self._refused("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                          "--dst-walker", 0, needle="cannot open")
        with mock.patch.object(C, "pids_holding", return_value=[4242]):
            self._refused("--branch", "emri", "--leaf", 1, "--src-walker", 2,
                          "--dst-walker", 0, apply=False, needle="4242")


if __name__ == "__main__":
    unittest.main()
