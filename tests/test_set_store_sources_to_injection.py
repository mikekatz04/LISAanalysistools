"""Put a stopped store's non-GB sources back at the catalogue injection
(scripts/fstat_proposal/set_store_sources_to_injection.py).

Mike 2026-10-22: the non-GB sources (VGB, MBHB, SOBHB, EMRI) "always start at
the catalog injected values rather than the last run's values"; "put the
parameters back to the injection values in place in the file while the run is
stopped", "then I will relaunch as a separate thing"; "default to 1 walker and
1 leaf, but add options to the command line to adjust all leaves in 1 walker,
or all walkers in 1 leaf, or all walkers and all leaves."

The store is written by the REAL ``GFHDFBackend`` + sub-backends in the
production layout (cold-chain main group with the sampler axis, per-leaf
ladders for mbh / emri / sobbh, a banded vgb sub-backend, rows allocated past
the ``iteration`` attr). The written slices are restated HERE, independently
of the tool, so a tool that writes too little or too much fails the
whole-tree comparison. The catalogue converters are checked against the run's
own ``prepare_*_branch`` on the local mojito catalogue when it is present.
"""

import contextlib
import glob
import hashlib
import io
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "scripts", "fstat_proposal"))

import set_store_sources_to_injection as T  # noqa: E402

NW = 4            # walkers
NT = 3            # rungs on every sub-backend ladder
SAVED = 3         # rows written (the iteration attr)
ALLOC = 5         # rows allocated: production grows ahead of the iteration attr
LAST = SAVED - 1  # the row a resume reads, the only row the tool edits
NUM_BANDS = 3
BAND_EDGES = np.linspace(1e-3, 4e-3, NUM_BANDS + 1)

BRANCH_SHAPES = {
    # branch: (nleaves_max, ndim); mbh has nleaves_max == nwalkers on purpose
    "gb": (5, 8),
    "vgb": (5, 6),
    "mbh": (4, 11),
    "emri": (3, 12),
    "sobbh": (2, 11),
    "psd": (1, 2),
}
SUB_BACKENDS = {"gb": GBHDFBackend, "vgb": GBHDFBackend, "mbh": MBHHDFBackend,
                "emri": EMRIHDFBackend, "sobbh": SOBBHHDFBackend, "psd": ModuleSubBackend}
SUB_STATE_BASES = {"gb": GBState, "vgb": GBState, "mbh": MBHState, "emri": EMRIState,
                   "sobbh": SOBBHState, "psd": ModuleSubState}
PER_LEAF = ("mbh", "emri", "sobbh")
BANDED = ("gb", "vgb")
LEAF_COUNT = {"mbh": "num_mbhs", "emri": "num_emris", "sobbh": "num_sobbhs"}
SOURCES = ("vgb", "mbh", "emri", "sobbh")

#: the injection rows the patched catalogue hands the tool (fixed, distinct)
INJ = {b: np.random.default_rng(99 + i).standard_normal(BRANCH_SHAPES[b]) + 10.0 * (i + 1)
       for i, b in enumerate(SOURCES)}

RUN_SETTINGS_LINE = (
    "  processor_init_kwargs: {'L1_folder': '/shared/data/mojito_cache', 'source_ids': "
    "{'GB': [0], 'VGB': [0], 'MBHB': [0, 2, 3, 4], 'EMRI': [0, 1, 2], 'SOBHB': [3, 5]}, "
    "'source_types': ['COMBINED', 'GB', 'VGB', 'MBHB', 'EMRI', 'SOBHB'], 'orbits_kwargs': "
    "{'frame': 'icrs', 'cls': <class 'lisatools.detector.L1Orbits'>}, 'verbose': True}\n")


def make_state(rng):
    """One saved row: every branch's tempered ensemble, distinct values everywhere."""
    coords = {b: rng.standard_normal((NT, NW, nl, nd)) for b, (nl, nd) in BRANCH_SHAPES.items()}
    inds = {b: np.ones((NT, NW, nl), dtype=bool) for b, (nl, _) in BRANCH_SHAPES.items()}
    inds["gb"][:, :, -1] = False
    state = GFState(
        {b: c[:1].copy() for b, c in coords.items()},
        inds={b: i[:1].copy() for b, i in inds.items()},
        log_like=rng.standard_normal((1, NW)),
        log_prior=rng.standard_normal((1, NW)),
        betas=np.ones(1),
        random_state=np.random.get_state(),
        sub_state_bases=SUB_STATE_BASES,
    )
    for b in BANDED:
        state.sub_states[b].initialize_band_information(
            NW, NT, BAND_EDGES, np.tile(np.linspace(1.0, 0.3, NT), (NUM_BANDS, 1)))
    for b, sub in state.sub_states.items():
        nl, nd = BRANCH_SHAPES[b]
        if b in PER_LEAF:
            sub.betas_all = np.tile(1.0 / 1.2 ** np.arange(NT), (nl, 1)) * (1 + 0.01 * rng.random())
        sub.initialize_tempered(NT, NW, nl, nd, coords=coords[b], inds=inds[b])
        if b in PER_LEAF or b == "psd":
            sub.log_like[:] = 1e6 + 100.0 * rng.standard_normal(sub.log_like.shape)
            sub.log_prior[:] = rng.standard_normal(sub.log_prior.shape)
        if b in PER_LEAF:
            for name in ("in_model_accepted", "in_model_proposed", "rj_accepted",
                         "rj_proposed", "swaps_accepted", "swaps_proposed"):
                getattr(sub, name)[:] = rng.integers(0, 50, getattr(sub, name).shape)
        sub.d_h[:] = rng.standard_normal(sub.d_h.shape)
        sub.h_h[:] = rng.standard_normal(sub.h_h.shape)
    return state


def make_backend(path):
    return GFHDFBackend(path, sub_backend=dict(SUB_BACKENDS),
                        sub_state_bases=dict(SUB_STATE_BASES))


def make_store(path, seed=20261022):
    rng = np.random.default_rng(seed)
    backend = make_backend(path)
    sub_kw = {b: dict(nleaves_max=nl, ndim=nd, ntemps=NT) for b, (nl, nd) in BRANCH_SHAPES.items()}
    for b in BANDED:
        sub_kw[b].update(num_bands=NUM_BANDS, band_edges=BAND_EDGES)
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


def written_slices(branch, walkers, leaves):
    """The test's OWN statement of every (dataset, index, value) the reset
    must write -- deliberately not read from the tool."""
    a = slice(None)
    out = []
    for l in leaves:
        for w in walkers:
            out.append((f"chain/{branch}", (LAST, a, a, w, l), INJ[branch][l]))
            out.append((f"sub_backend/{branch}/chain", (LAST, a, w, l), INJ[branch][l]))
            out.append((f"sub_backend/{branch}/d_h", (LAST, w, l), np.nan))
            out.append((f"sub_backend/{branch}/h_h", (LAST, w, l), np.nan))
    return out


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


def _fake_rows(branch, cat, *, vgb_basis=None):
    nl, nd = INJ[branch].shape
    return (np.array(INJ[branch], copy=True), [f"id {j}" for j in range(nl)],
            [f"c{k}" for k in range(nd)])


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="srcinj_")
        self.path = os.path.join(self.tmp, "run_testing.h5")
        make_store(self.path)
        os.makedirs(os.path.join(self.tmp, "run_artifacts"))
        with open(os.path.join(self.tmp, "run_artifacts", "run_settings.log"), "w") as fp:
            fp.write("GLOBAL FIT RUN SETTINGS\n" + RUN_SETTINGS_LINE + "  dt: 2.5\n")
        # the store looks idle: written "an hour ago"
        old = time.time() - 3600.0
        os.utime(self.path, (old, old))
        self.calls = []

        def fake_catalogue(l1, cls, ids):
            self.calls.append((l1, cls, list(ids)))
            return {i: {} for i in ids}, f"{l1}/catalogues/{cls}.hdf5"

        for name, value in (("pids_holding", mock.Mock(return_value=[])),
                            ("load_catalogue", fake_catalogue),
                            ("injection_rows", _fake_rows)):
            p = mock.patch.object(T, name, value)
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_tool(self, *args, store=None, apply=False):
        argv = ["--store", store or self.path] + [str(a) for a in args]
        if apply:
            argv += ["--apply"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = T.main(argv)
        return rc, out.getvalue()

    def assert_only_written(self, before, after, writes):
        """Every dataset equals ``before`` except the listed slices, which hold
        the listed values; every attr is unchanged."""
        data0, attrs0 = before
        data1, attrs1 = after
        self.assertEqual(set(data0), set(data1))
        expected = {k: np.array(v, copy=True) for k, v in data0.items()}
        for name, sel, value in writes:
            expected[name][sel] = value
        for name in sorted(data0):
            self.assertTrue(_equal(data1[name], expected[name]),
                            f"{name} differs from the expected post-reset contents")
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
            self.assertEqual(g["chain/mbh"].shape, (ALLOC, 1, 1, NW, 4, 11))
            self.assertEqual(g["sub_backend/emri/log_like"].shape, (ALLOC, 3, NT, NW))
            vgb = g["sub_backend/vgb"]
            self.assertEqual(vgb["chain"].shape, (ALLOC, NT, NW, 5, 6))
            self.assertTrue(any(k.startswith("band_") for k in vgb))
            # the stored rows genuinely differ from the injection
            self.assertFalse(np.allclose(g["sub_backend/mbh/chain"][LAST, :, 1, 2], INJ["mbh"][2]))
        _validate_resume_readable(self.path)


class ResetTest(_Base):
    def test_dry_run_writes_nothing(self):
        h0 = md5(self.path)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 1, "--leaf", 2)
        self.assertEqual(rc, 0, out)
        self.assertIn("DRY RUN", out)
        self.assertIn("--apply", out)
        self.assertIn("leaf 2 (id 2)", out)
        self.assertEqual(md5(self.path), h0)
        self.assertFalse(os.path.exists(self.path + T.BACKUP_SUFFIX))

    def test_the_run_settings_give_the_catalogue_folder_and_ids(self):
        rc, out = self.run_tool("--branch", "all", "--all-walkers", "--all-leaves")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.calls, [("/shared/data/mojito_cache", "VGB", [0]),
                                      ("/shared/data/mojito_cache", "MBHB", [0, 2, 3, 4]),
                                      ("/shared/data/mojito_cache", "EMRI", [0, 1, 2]),
                                      ("/shared/data/mojito_cache", "SOBHB", [3, 5])])
        self.calls.clear()
        rc, out = self.run_tool("--branch", "sobbh", "--walker", 0, "--leaf", 1,
                                "--l1-folder", "/here", "--ids", "7,9")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.calls, [("/here", "SOBHB", [7, 9])])

    def test_one_walker_one_leaf(self):
        h0 = md5(self.path)
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 1, "--leaf", 2, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("VERIFY OK", out)
        self.assert_only_written(before, snapshot(self.path), written_slices("mbh", [1], [2]))
        self.assertEqual(md5(self.path + T.BACKUP_SUFFIX), h0)   # the untouched original

    def test_all_leaves_of_one_walker(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "emri", "--walker", 3, "--all-leaves", apply=True)
        self.assertEqual(rc, 0, out)
        self.assert_only_written(before, snapshot(self.path),
                                 written_slices("emri", [3], range(3)))

    def test_one_leaf_on_all_walkers(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "sobbh", "--all-walkers", "--leaf", 0, apply=True)
        self.assertEqual(rc, 0, out)
        self.assert_only_written(before, snapshot(self.path),
                                 written_slices("sobbh", range(NW), [0]))

    def test_repeated_walkers_and_leaves(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--walker", 3,
                                "--leaf", 3, "--leaf", 1, apply=True)
        self.assertEqual(rc, 0, out)
        self.assert_only_written(before, snapshot(self.path),
                                 written_slices("mbh", [0, 3], [3, 1]))

    def test_vgb_band_records_are_left_alone(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "vgb", "--all-walkers", "--all-leaves", apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("band_* records", out)
        self.assert_only_written(before, snapshot(self.path),
                                 written_slices("vgb", range(NW), range(5)))

    def test_every_branch_walker_and_leaf(self):
        before = snapshot(self.path)
        rc, out = self.run_tool("--branch", "all", "--all-walkers", "--all-leaves", apply=True)
        self.assertEqual(rc, 0, out)
        writes = []
        for b in SOURCES:
            writes += written_slices(b, range(NW), range(BRANCH_SHAPES[b][0]))
        self.assert_only_written(before, snapshot(self.path), writes)

    def test_the_resume_starts_the_pairs_at_the_injection(self):
        before = make_backend(self.path).get_last_sample()
        rc, out = self.run_tool("--branch", "emri", "--walker", 2, "--leaf", 1, apply=True)
        self.assertEqual(rc, 0, out)
        _validate_resume_readable(self.path)
        state = make_backend(self.path).get_last_sample()   # what run.py loads
        sub, sub0 = state.sub_states["emri"], before.sub_states["emri"]
        for k in range(NT):
            np.testing.assert_array_equal(sub.coords[k, 2, 1], INJ["emri"][1])
        np.testing.assert_array_equal(state.branches["emri"].coords[0, 2, 1], INJ["emri"][1])
        self.assertTrue(np.isnan(sub.d_h[2, 1]) and np.isnan(sub.h_h[2, 1]))
        # every other pair, the stored lnL records and the ladders load as before
        keep = np.ones((NW, 3), dtype=bool)
        keep[2, 1] = False
        np.testing.assert_array_equal(sub.coords[:, keep], sub0.coords[:, keep])
        np.testing.assert_array_equal(sub.log_like, sub0.log_like)
        np.testing.assert_array_equal(sub.betas_all, sub0.betas_all)
        np.testing.assert_array_equal(state.log_like, before.log_like)
        # the first propose's cold-row check passes on every branch
        for name, s in state.sub_states.items():
            s.check_cold_row(state, name)

    def test_verify_reports_before_and_after(self):
        args = ("--branch", "mbh", "--walker", 0, "--leaf", 3)
        h0 = md5(self.path)
        rc, out = self.run_tool(*args, "--verify")
        self.assertEqual(rc, 1, out)
        self.assertIn("VERIFY FAILED", out)
        self.assertEqual(md5(self.path), h0)
        self.assertEqual(self.run_tool(*args, apply=True)[0], 0)
        rc, out = self.run_tool(*args, "--verify")
        self.assertEqual(rc, 0, out)
        self.assertIn("VERIFY OK", out)

    def test_verify_catches_a_reset_that_skips_the_sub_backend_rungs(self):
        """Mutation: write only the main cold row. The verify (and the
        cold-row check it restates) must fail the run, not pass it."""
        def main_only(path, plan):
            with h5py.File(path, "r+") as f:
                root = f[plan.group]
                for bp in plan.branches:
                    for w, l in bp.pairs():
                        root["chain"][bp.branch][plan.row, :, :, w, l] = bp.inj[l]
            return []

        with mock.patch.object(T, "apply_reset", main_only):
            rc, out = self.run_tool("--branch", "mbh", "--walker", 1, "--leaf", 2, apply=True)
        self.assertEqual(rc, 1, out)
        self.assertIn("rungs [0, 1, 2] differ from the injection", out)
        self.assertIn("check_cold_row would abort", out)
        self.assertIn("FAILED", out)


class FailurePathTest(_Base):
    """A failed write or verify stops there: nothing after it is edited, and
    the message says how to get back."""

    def _setup(self):
        twin = T.sidecar_paths(self.path)["backup"]
        shutil.copy2(self.path, twin)
        eig = T.eigen_path(self.path)
        with open(eig, "wb") as fp:
            pickle.dump({"format": 3, "entries": {"mbh:1": {"scope": "walker_max"}}}, fp)
        midit = T.sidecar_paths(self.path)["midit"]
        with open(midit, "wb") as fp:
            fp.write(b"x")
        old = time.time() - 3600.0
        for p in (twin, midit):
            os.utime(p, (old, old))
        return twin, eig, midit

    def test_a_write_exception(self):
        twin, eig, midit = self._setup()
        h_twin, h_eig = md5(twin), md5(eig)

        def boom(path, plan):
            raise OSError("disk went away")

        with mock.patch.object(T, "apply_reset", boom):
            with self.assertRaises(OSError):
                self.run_tool("--branch", "mbh", "--all-walkers", "--leaf", 1,
                              "--also-backup", "--reset-midit", apply=True)
        self.assertEqual((md5(twin), md5(eig)), (h_twin, h_eig))
        self.assertTrue(os.path.exists(self.path + T.BACKUP_SUFFIX))

    def test_a_store_verify_failure_edits_nothing_after_it(self):
        twin, eig, midit = self._setup()
        h_twin, h_eig = md5(twin), md5(eig)
        with mock.patch.object(T, "verify_reset", return_value=["forced"]):
            rc, out = self.run_tool("--branch", "mbh", "--all-walkers", "--leaf", 1,
                                    "--also-backup", "--reset-midit", apply=True)
        self.assertEqual(rc, 1, out)
        self.assertIn("running backup twin was NOT edited", out)
        self.assertIn("move " + midit + ".stale-", out)        # the way back names the midit
        self.assertIn(self.path + T.BACKUP_SUFFIX, out)
        self.assertEqual((md5(twin), md5(eig)), (h_twin, h_eig))


class SidecarTest(_Base):
    def _eigen(self, keys=("mbh:0", "mbh:1", "mbh:2", "emri:0")):
        path = T.eigen_path(self.path)
        payload = {"format": 3, "entries": {k: {"scope": "walker_max", "axes": np.eye(2)}
                                            for k in keys}}
        with open(path, "wb") as fp:
            pickle.dump(payload, fp)
        return path

    def _entries(self, path):
        with open(path, "rb") as fp:
            return sorted(pickle.load(fp)["entries"])

    def test_a_leaf_reset_on_every_walker_loses_its_table(self):
        path = self._eigen()
        h0 = md5(path)
        rc, out = self.run_tool("--branch", "mbh", "--all-walkers", "--leaf", 1, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self._entries(path), ["emri:0", "mbh:0", "mbh:2"])
        self.assertEqual(md5(path + T.BACKUP_SUFFIX), h0)

    def test_a_partial_walker_reset_keeps_it(self):
        path = self._eigen()
        h0 = md5(path)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--walker", 1,
                                "--leaf", 1, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("entries kept", out)
        self.assertEqual(md5(path), h0)

    def test_keep_eigen_tables(self):
        path = self._eigen()
        h0 = md5(path)
        rc, out = self.run_tool("--branch", "mbh", "--all-walkers", "--all-leaves",
                                "--keep-eigen-tables", apply=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual(md5(path), h0)

    def test_a_dry_run_leaves_the_sidecar(self):
        path = self._eigen()
        h0 = md5(path)
        rc, out = self.run_tool("--branch", "all", "--all-walkers", "--all-leaves")
        self.assertEqual(rc, 0, out)
        self.assertIn("DROPPED at --apply", out)
        self.assertEqual(md5(path), h0)

    def test_midit_checkpoint_refused_unless_reset(self):
        midit = T.sidecar_paths(self.path)["midit"]
        with open(midit, "wb") as fp:
            fp.write(b"x")
        old = time.time() - 3600.0
        os.utime(midit, (old, old))
        h0 = md5(self.path)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--leaf", 0, apply=True)
        self.assertEqual(rc, 2, out)
        self.assertIn("--reset-midit", out)
        self.assertEqual(md5(self.path), h0)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--leaf", 0,
                                "--reset-midit", apply=True)
        self.assertEqual(rc, 0, out)
        self.assertFalse(os.path.exists(midit))
        self.assertEqual(len(glob.glob(midit + ".stale-*")), 1)

    def test_also_backup_edits_the_twin_too(self):
        twin = T.sidecar_paths(self.path)["backup"]
        shutil.copy2(self.path, twin)
        before = snapshot(twin)
        h_twin = md5(twin)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 1, "--leaf", 2)
        self.assertEqual(md5(twin), h_twin)                  # dry run
        rc, out = self.run_tool("--branch", "mbh", "--walker", 1, "--leaf", 2, apply=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual(md5(twin), h_twin)                  # not without --also-backup
        os.remove(self.path + T.BACKUP_SUFFIX)
        old = time.time() - 3600.0                           # the apply just wrote it
        os.utime(self.path, (old, old))
        rc, out = self.run_tool("--branch", "mbh", "--walker", 1, "--leaf", 2,
                                "--also-backup", apply=True)
        self.assertEqual(rc, 0, out)
        self.assert_only_written(before, snapshot(twin), written_slices("mbh", [1], [2]))


class RefusalTest(_Base):
    def _refused(self, *args, apply=True, needle=None, rc_want=2):
        h0 = md5(self.path)
        rc, out = self.run_tool(*args, apply=apply)
        self.assertEqual(rc, rc_want, out)
        self.assertIn("REFUSING", out)
        if needle:
            self.assertIn(needle, out)
        self.assertEqual(md5(self.path), h0)
        self.assertFalse(os.path.exists(self.path + T.BACKUP_SUFFIX))

    def test_branches_out_of_scope(self):
        for b in ("gb", "psd", "galfor", "nope"):
            with self.subTest(branch=b):
                self._refused("--branch", b, "--walker", 0, "--leaf", 0, needle="out of scope")

    def test_the_selection_must_be_explicit(self):
        self._refused("--branch", "mbh", "--leaf", 0, needle="--all-walkers")
        self._refused("--branch", "mbh", "--walker", 0, needle="--all-leaves")
        self._refused("--branch", "mbh", "--walker", 0, "--all-walkers", "--leaf", 0)
        self._refused("--branch", "mbh,emri", "--walker", 0, "--leaf", 0,
                      needle="several branches need --all-leaves")
        self._refused("--branch", "mbh,emri", "--walker", 0, "--all-leaves", "--ids", "1,2",
                      needle="ONE --branch")

    def test_bad_walkers_and_leaves(self):
        self._refused("--branch", "mbh", "--walker", NW, "--leaf", 0, needle="outside")
        self._refused("--branch", "mbh", "--walker", 0, "--leaf", 4, needle="outside")
        self._refused("--branch", "mbh", "--walker", 1, "--walker", 1, "--leaf", 0,
                      needle="repeats")

    def test_catalogue_rows_that_do_not_match_the_store(self):
        def short(branch, cat, *, vgb_basis=None):
            rows, labels, names = _fake_rows(branch, cat)
            return rows[:-1], labels[:-1], names

        with mock.patch.object(T, "injection_rows", short):
            self._refused("--branch", "mbh", "--walker", 0, "--leaf", 0, needle="--ids")

    def test_a_dead_leaf(self):
        with h5py.File(self.path, "r+") as f:
            f["global_fit/sub_backend/sobbh/inds"][LAST, 1, 2, 1] = False
        self._refused("--branch", "sobbh", "--walker", 2, "--leaf", 1, needle="not alive")

    def test_a_torn_cold_row(self):
        with h5py.File(self.path, "r+") as f:
            f["global_fit/sub_backend/emri/chain"][LAST, 0, 1, 0] += 1.0
        self._refused("--branch", "emri", "--walker", 1, "--leaf", 0, needle="torn save")

    def test_an_unknown_sub_backend_dataset(self):
        for b in ("mbh", "vgb"):
            with self.subTest(branch=b):
                with h5py.File(self.path, "r+") as f:
                    f[f"global_fit/sub_backend/{b}"].create_dataset("mystery", data=np.zeros(3))
                self._refused("--branch", b, "--walker", 0, "--leaf", 0, needle="mystery")

    def test_an_existing_backup(self):
        open(self.path + T.BACKUP_SUFFIX, "w").close()
        h0 = md5(self.path)
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--leaf", 0, apply=True)
        self.assertEqual(rc, 2, out)
        self.assertIn("already exists", out)
        self.assertEqual(md5(self.path), h0)

    def test_a_recently_written_store_needs_force(self):
        os.utime(self.path, None)
        self._refused("--branch", "mbh", "--walker", 0, "--leaf", 0, needle="--force")
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--leaf", 0, "--force",
                                apply=True)
        self.assertEqual(rc, 0, out)

    def test_a_file_held_open_elsewhere(self):
        with mock.patch.object(T, "pids_holding", return_value=[4242]):
            self._refused("--branch", "mbh", "--walker", 0, "--leaf", 0, needle="4242")

    def test_a_refused_apply_leaves_the_midit_checkpoint_in_place(self):
        midit = T.sidecar_paths(self.path)["midit"]
        with open(midit, "wb") as fp:
            fp.write(b"x")
        old = time.time() - 3600.0
        os.utime(midit, (old, old))
        open(self.path + T.BACKUP_SUFFIX, "w").close()       # refused: backup exists
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--leaf", 0,
                                "--reset-midit", apply=True)
        self.assertEqual(rc, 2, out)
        self.assertTrue(os.path.exists(midit))
        self.assertEqual(glob.glob(midit + ".stale-*"), [])

    def test_the_store_column_names_must_match(self):
        names = [f"c{k}" for k in range(BRANCH_SHAPES["mbh"][1])]
        with h5py.File(self.path, "r+") as f:
            f["global_fit/key_order"].attrs["mbh"] = np.array(names, dtype=h5py.string_dtype())
        rc, out = self.run_tool("--branch", "mbh", "--walker", 0, "--leaf", 0)
        self.assertEqual(rc, 0, out)                          # same names: accepted
        swapped = [names[1], names[0]] + names[2:]
        with h5py.File(self.path, "r+") as f:
            f["global_fit/key_order"].attrs["mbh"] = np.array(swapped, dtype=h5py.string_dtype())
        self._refused("--branch", "mbh", "--walker", 0, "--leaf", 0, needle="key_order")

    def test_ids_do_not_apply_to_vgb(self):
        self._refused("--branch", "vgb", "--walker", 0, "--leaf", 0, "--ids", "0",
                      needle="does not apply to vgb")

    def test_no_catalogue_folder(self):
        shutil.rmtree(os.path.join(self.tmp, "run_artifacts"))
        self._refused("--branch", "mbh", "--walker", 0, "--leaf", 0, needle="--run-settings")


class SettingsTest(unittest.TestCase):
    def test_the_processor_line_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "x_artifacts"))
            p = os.path.join(tmp, "x_artifacts", "run_settings.log")
            with open(p, "w") as fp:
                fp.write("header\n" + RUN_SETTINGS_LINE)
            self.assertEqual(T.find_run_settings(os.path.join(tmp, "x_testing.h5")), [p])
            got = T.read_run_settings(p)
        self.assertEqual(got["l1_folder"], "/shared/data/mojito_cache")
        self.assertEqual(got["source_ids"], {"GB": [0], "VGB": [0], "MBHB": [0, 2, 3, 4],
                                             "EMRI": [0, 1, 2], "SOBHB": [3, 5]})

    def test_column_names_compare_as_the_store_spells_them(self):
        """The store's key_order spells EMRI's cosqS / cosqK as qS / qK."""
        from lisatools.globalfit.stock.erebor.transforms import make_emri_transform_container

        ours = list(make_emri_transform_container([1.0, 0.0]).input_basis)
        store = ["logm1", "m2", "a", "p0", "e0", "dist", "qS", "phiS", "qK", "phiK",
                 "Phi_phi0", "Phi_r0"]
        self.assertEqual([T._column_key(n) for n in ours], [T._column_key(n) for n in store])
        self.assertNotEqual(T._column_key("phiS"), T._column_key("phiK"))
        # a cosine and its angle stay distinct (only EMRI's qS / qK are aliased)
        for cos, ang in (("cos_iota", "iota"), ("cosinc", "inc"), ("sin_delta", "delta")):
            self.assertNotEqual(T._column_key(cos), T._column_key(ang))

    def test_the_vgb_basis(self):
        self.assertEqual(T.vgb_basis_for(6), "chirp")
        self.assertEqual(T.vgb_basis_for(5, "legacy"), "legacy")
        with self.assertRaisesRegex(ValueError, "--vgb-basis"):
            T.vgb_basis_for(5)
        with self.assertRaises(ValueError):
            T.vgb_basis_for(6, "bogus")


def _catalogue_dir():
    d = os.environ.get("MOJITO_CATALOGUE_L1_FOLDER",
                       os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0"))
    return d if os.path.isdir(os.path.join(d, "catalogues")) else None


@unittest.skipIf(_catalogue_dir() is None, "no local mojito catalogue folder")
class ConverterParityTest(unittest.TestCase):
    """The tool's rows equal the run's own fresh-start injection rows."""

    IDS = {"MBHB": [0, 2, 3, 4, 5, 7, 9, 10, 12, 15, 16, 18], "EMRI": [0, 1, 2, 3, 4, 5, 6, 7],
           "SOBHB": [0, 1, 2, 3, 4, 5]}

    def _prepared(self, branch, cls, cat):
        from lisatools.globalfit.stock.erebor import source_runtime as SR
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        settings = {"mbh": MBHSettings, "emri": EMRISettings, "sobbh": SOBBHSettings}[branch]()
        settings.initialize_kwargs = {}          # skip the waveform setup
        # an observation window holding every merger (the MBH filter keeps all):
        # obs_start = data_t0 - mbh_waveform_t0 = -1e11 s, obs_end = 9e11 s
        gen = SimpleNamespace(force_backend="cpu", Tobs=1e12, dt=5.0, data_t0=0.0,
                              catalogue={cls: cat})
        gs = SimpleNamespace(n_injections={cls: len(cat)}, tdi_gen_str="2nd generation",
                             tdi_chan="XYZ", data_mode="mojito", max_freq=1.0,
                             mbh_waveform_t0=1e11)
        prep = {"mbh": SR.prepare_mbh_branch, "emri": SR.prepare_emri_branch,
                "sobbh": SR.prepare_sobbh_branch}[branch]
        try:
            prep(settings, gen, gs)
        except AttributeError:
            # a setup-only field (t_plunge_pad, chirp_fdot_max, ...) read AFTER
            # the injection rows are set; the rows are what this test compares
            pass
        self.assertIsNotNone(settings.injection, f"prepare_{branch}_branch set no injection")
        return np.asarray(settings.injection, dtype=float), tuple(settings.injection_ids)

    def test_mbh_emri_sobbh(self):
        l1 = _catalogue_dir()
        for branch, cls in (("mbh", "MBHB"), ("emri", "EMRI"), ("sobbh", "SOBHB")):
            with self.subTest(branch=branch):
                # ids in REVERSE: the leaf order must come from sorting, as the run's
                cat, _ = T.load_catalogue(l1, cls, self.IDS[cls][::-1])
                rows, labels, names = T.injection_rows(branch, cat)
                want, ids = self._prepared(branch, cls, cat)
                np.testing.assert_array_equal(rows, want)
                self.assertEqual(labels, [f"id {i}" for i in ids])
                self.assertEqual(len(names), rows.shape[1])

    def test_vgb_chirp_basis(self):
        l1 = _catalogue_dir()
        cat, path = T.load_catalogue(l1, "VGB", [0])
        rows, labels, names = T.injection_rows("vgb", cat, vgb_basis="chirp")
        self.assertEqual(names, ["dist", "phi0", "cos_iota", "psi", "Mc", "fdot_astro_ratio"])
        with h5py.File(path, "r") as f:
            b = f["Binaries"]
            np.testing.assert_allclose(rows[:, 0], np.asarray(b["LuminosityDistance"]) * 1e3)
            np.testing.assert_allclose(rows[:, 4], np.asarray(b["ChirpMassSSBFrame"]))
        self.assertEqual(len(labels), rows.shape[0])
        legacy, _, lnames = T.injection_rows("vgb", cat, vgb_basis="legacy")
        self.assertEqual(lnames, ["A", "fdot", "phi0", "cos_iota", "psi"])
        np.testing.assert_array_equal(legacy[:, 2], rows[:, 1])     # the same phi0


def _launchers():
    paths = [os.path.join(ROOT, "scripts", "flagship", "launcher.sh.in")]
    paths += sorted(glob.glob(os.path.join(ROOT, "scripts", "flagship", "submit_gf_*mo.sh")))
    paths += [os.path.join(ROOT, "scripts", "fstat_proposal", f) for f in (
        "submit_gf_3mo_v9_2gpu.sh", "submit_gf_6mo_v9_4gpu.sh",
        "submit_gf_9mo_v9.sh", "submit_gf_1yr_v9.sh")]
    return [p for p in paths if os.path.exists(p)]


def _read(path):
    with open(path) as fp:
        return fp.read()


class LauncherTest(unittest.TestCase):
    def test_no_launcher_arms_the_parent_source_warm_start(self):
        """New runs start the non-GB sources at the injection (2026-10-22)."""
        paths = _launchers()
        self.assertGreaterEqual(len(paths), 10)
        for p in paths:
            text = _read(p)
            with self.subTest(launcher=os.path.basename(p)):
                for line in re.findall(r"^export SOURCE_WARM_START_STORE=.*$", text, re.M):
                    self.assertIn(line, ("export SOURCE_WARM_START_STORE=${SOURCE_WARM_START_STORE-}",
                                         "export SOURCE_WARM_START_STORE="), line)
                self.assertNotIn("SOURCE_WARM_START_STORE-${GF_SEED_STORE}", text)

    def test_the_launchers_point_at_this_tool(self):
        tool = os.path.join(ROOT, "scripts", "fstat_proposal", "set_store_sources_to_injection.py")
        self.assertTrue(os.path.exists(tool))
        named = 0
        for p in _launchers():
            text = _read(p)
            with self.subTest(launcher=os.path.basename(p)):
                self.assertNotIn("relaunch_sources_at_injection", text)
                self.assertNotIn("RESET_TO_INJECTION_ON_RESUME", text)
                named += "scripts/fstat_proposal/set_store_sources_to_injection.py" in text
        self.assertGreaterEqual(named, 6)


if __name__ == "__main__":
    unittest.main()
