"""Put a STOPPED store's non-GB sources back at the catalogue injection, in place.

Mike 2026-10-22: the non-GB sources (VGB, MBHB, SOBHB, EMRI) should "always
start at the catalog injected values rather than the last run's values";
"put the parameters back to the injection values in place in the file while
the run is stopped", "then I will relaunch as a separate thing"; "default to 1
walker and 1 leaf, but add options to the command line to adjust all leaves in
1 walker, or all walkers in 1 leaf, or all walkers and all leaves."

This tool edits ONLY the store's last row (``iteration - 1``, the row a resume
reads) for the chosen (walker, leaf) pairs of the chosen branch, on EVERY rung
of that leaf's ladder, and nothing else. The relaunch is a separate step: the
run's usual launch line, unchanged.

Selection (dry run by default; nothing is written without ``--apply``)::

    --branch mbh --walker 0 --leaf 3          one walker, one leaf (the default form)
    --branch mbh --walker 0 --all-leaves      every leaf of one walker
    --branch mbh --all-walkers --leaf 3       one leaf on every walker
    --branch mbh --all-walkers --all-leaves   the whole branch
    --branch all --all-walkers --all-leaves   vgb, mbh, emri and sobbh at once

``--walker`` and ``--leaf`` repeat for a few at a time. A leaf is the store's
leaf SLOT; the dry run prints the catalogue id each slot carries. Several
branches in one call (a comma list, or ``all``) need ``--all-leaves``, since a
leaf slot means a different source on each branch.

Where the injection rows come from
----------------------------------
The same catalogue rows, in the same leaf order and sampling basis, that the
run's own setup builds for a fresh start
(``stock/erebor/source_runtime.py::prepare_{mbh,emri,sobbh}_branch``,
``stock/erebor/vgb.py::prepare_vgb_branch``): the mojito catalogue files under
``<L1_folder>/catalogues`` read through ``L1DataLoader``, the run's catalogue
ids, leaf ``j`` = the ``j``-th SORTED id (VGB: the catalogue's file order),
converted by the stock functions
(``mbh_catalogue_to_sampling_basis``; ``emri_catalogue_to_waveform_basis`` ->
``make_emri_transform_container`` inverse; ``sobbh_catalogue_to_waveform_basis``
-> ``make_sobbh_transform_container`` inverse; VGB through
``prepare_vgb_branch`` itself, in the basis the store's column count names:
6 = the chirp-mass basis the launchers export, ``VGB_CHIRP_MASS_BASIS=1``).
``L1_folder`` and the ids are read from the run's
``<run dir>/*_artifacts/run_settings.log`` (``processor_init_kwargs``);
``--l1-folder`` / ``--ids`` override them (the login node's catalogue copy,
or the ids the MBH merger-window filter kept when it dropped some: the run
logs ``MBH merger-window filter: dropping ...`` then). The rows must match the
stored ``(nleaves, ndim)`` exactly or the tool refuses.

What is written, at row ``iteration - 1``, for each chosen (walker ``w``, leaf ``l``)
--------------------------------------------------------------------------------------
::

    chain/<b>                [row, :, :, w, l, :]   = injection[l]   main cold row
    sub_backend/<b>/chain    [row, :, w, l, :]      = injection[l]   EVERY rung
    sub_backend/<b>/d_h      [row, w, l]            = NaN            "not recorded"
    sub_backend/<b>/h_h      [row, w, l]            = NaN

Exact injection values, no ``START_FACTOR`` scatter. The main cold row and the
sub-backend rung 0 are written from one array, so the first propose's
``check_cold_row`` holds. The leaf must already be alive (``inds`` true on
every rung) on every chosen walker: these are fixed-leaf branches, so a dead
leaf means the store is not what this tool expects, and it refuses.

What is NOT written, and why that is safe
-----------------------------------------
* The per-walker totals ``log_like`` / ``log_prior`` of the main group: the
  resume rebuilds the residual from the main cold coordinates
  (``setup_acs(rebuild_residuals=True)``), re-evaluates ``log_like`` from it
  and zeroes ``log_prior``.
* The per-leaf ``sub_backend/<b>/log_like`` / ``log_prior`` (mbh, emri,
  sobbh): telemetry; every leaf visit recomputes every rung's value from the
  coordinates (``prev_logl`` in ``ResidualAddOneRemoveOneMove.propose_local``).
  Until the branch's first visit they read the OLD position's values.
* Walker-free records: the per-leaf ladders ``betas_all`` and the per-(leaf,
  rung) counters.
* The VGB ``band_*`` records, per-walker ones included: a VGB leaf's
  frequency is fixed, so its band and the band occupancy
  (``band_num_binaries``) do not change, and VGB has no birth / death move
  that reads its per-band lnL records.
* The GB and noise branches, every other walker and leaf, every other row.
  Side effect on a search stage with the GB valve live (``gb_search_*``):
  moving these sources changes every walker's GB band residuals, and the
  judge compares each band against its ALL-TIME best cold lnL
  (``band_cold_logl_max_w``, never released) plus a tolerance. A band whose
  lnL rises above that best counts as improved and, under the all-walker
  rule, reopens on every walker. A band whose lnL DROPS (the usual case: the
  injection point scores below a walker's best) reads as converged, so it
  keeps, or reaches, its RJ shut. Where GB leaves had absorbed a lost
  source's power, putting the source back can leave those bands shut with
  the doubled power until the periodic band revive lets their deaths run.
  ``full_pe`` (valve released) is not affected.
* The store's ``key_order`` names are compared with the injection's columns
  (mbh, emri, sobbh; VGB records column numbers), so a same-width basis
  change refuses instead of writing values into the wrong columns.

Sidecars
--------
* ``*_midit_checkpoint.pkl``: a checkpoint at or after the last row WINS over
  the store at resume and would discard the edit; ``--apply`` refuses while
  one exists unless ``--reset-midit`` moves it aside.
* ``*_running_backup_copy.h5``: read only when the store is unreadable at
  resume and overwritten by the first save; ``--also-backup`` applies the same
  edit to its own last row.
* ``<store>_eigen_tables.pkl``: the production scope ``walker_max`` keeps one
  walker-independent proposal table per leaf, built at the walkers' old
  positions. A leaf reset on EVERY walker has its entry dropped (after a
  backup copy ``<sidecar>.srcinj.bak``) so the first visit builds it at the
  injection, as a fresh run does; ``--keep-eigen-tables`` keeps it. A leaf
  reset on some walkers keeps its table (still the other walkers' scale).

Usage (12mo flagship; dry run first)::

    python scripts/fstat_proposal/set_store_sources_to_injection.py \\
        --store /shared/data/global_fit_output/gf_prod_12mo_flagship/gf_prod_12mo_testing.h5 \\
        --branch all --all-walkers --all-leaves
    ... the same command with --apply

``--apply`` copies the store to ``<store>.srcinj.bak`` first (refusing if that
exists; ``--no-backup`` skips it), writes in place, re-reads and verifies
every written slice, and runs lisatools' resume-readability check.
``--verify`` alone is a read-only check that the chosen pairs already hold
the injection. Then relaunch with the run's usual launch line (the 12mo:
``STAGE_SKIP_NOISE=1 scripts/flagship/submit_gf_12mo.sh``).

If the open hangs on ``/shared`` (h5py blocks, it does not error, when it
cannot take the HDF5 file lock there), run with
``HDF5_USE_FILE_LOCKING=FALSE`` as the launchers do; the run being down is
then guarded by the ``lsof`` and modification-time checks alone.
"""

import argparse
import ast
import glob
import os
import pickle
import re
import shlex
import shutil
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy_store_leaf as CSL  # noqa: E402  (store layout tables + safety helpers)
from compact_gf_store import pids_holding, sidecar_paths  # noqa: E402

MB = CSL.MB

#: The branches this tool resets, in the order a multi-branch call walks them.
BRANCHES = ("vgb", "mbh", "emri", "sobbh")
#: Each branch's mojito catalogue class (``L1DataLoader.catalogues_map`` key).
CATALOGUE_CLASS = {"vgb": "VGB", "mbh": "MBHB", "emri": "EMRI", "sobbh": "SOBHB"}
#: VGB sub-backend datasets with a leaf axis; every other one is band-keyed.
VGB_LEAF_DATASETS = ("chain", "inds", "d_h", "h_h")
#: VGB sampling bases (``stock/erebor/vgb.py::vgb_sampled_basis``).
VGB_BASES = ("legacy", "distance", "chirp")

#: Appended to the store (and the twin) path for the pre-write copy.
BACKUP_SUFFIX = ".srcinj.bak"
EIGEN_SUFFIX = CSL.EIGEN_SUFFIX

#: A leaf whose median |relative change| exceeds this on EVERY chosen walker
#: is flagged in the dry run (a wandered source, or a wrong id list).
FAR_REL_CHANGE = 0.5


# --------------------------------------------------------------------------
# the run's catalogue settings
# --------------------------------------------------------------------------
def find_run_settings(store):
    """``<store dir>/*_artifacts/run_settings.log`` candidates, sorted."""
    d = os.path.dirname(os.path.abspath(store))
    return sorted(glob.glob(os.path.join(d, "*_artifacts", "run_settings.log")))


def _dict_after(line, key):
    """The ``{...}`` literal that follows ``'key': `` on ``line``, or ``None``."""
    tag = f"'{key}': "
    i = line.find(tag)
    if i < 0:
        return None
    i += len(tag)
    if i >= len(line) or line[i] != "{":
        return None
    depth = 0
    for j in range(i, len(line)):
        if line[j] == "{":
            depth += 1
        elif line[j] == "}":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(line[i:j + 1])
    return None


def _id_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(v)]


def read_run_settings(path):
    """``{"l1_folder", "source_ids"}`` from the run's ``processor_init_kwargs``."""
    with open(path) as fp:
        lines = [ln for ln in fp.read().splitlines() if "processor_init_kwargs:" in ln]
    if len(lines) != 1:
        raise ValueError(f"{path}: {len(lines)} 'processor_init_kwargs:' lines (expected 1)")
    line = lines[0]
    m = re.search(r"'L1_folder': '([^']*)'", line)
    try:
        ids = _dict_after(line, "source_ids")
    except (ValueError, SyntaxError):
        ids = None
    if m is None or not isinstance(ids, dict):
        raise ValueError(f"{path}: no L1_folder / source_ids in processor_init_kwargs")
    return {"l1_folder": m.group(1),
            "source_ids": {str(k).upper(): _id_list(v) for k, v in ids.items()}}


def parse_ids(text):
    try:
        out = [int(x) for x in str(text).replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise ValueError(f"--ids expects a comma list of integers, got {text!r}") from None
    if not out:
        raise ValueError("--ids is empty")
    return out


def load_catalogue(l1_folder, cls, ids):
    """``({id: params}, catalogue file)`` read the way ``L1DataLoader`` reads it."""
    from lisatools.globalfit.preprocessing import L1DataLoader

    loader = L1DataLoader(l1_folder, source_types=[cls], source_ids={cls: list(ids)},
                          verbose=False)
    path = os.path.join(loader.catalogues_folder, loader.catalogues_map[cls])
    if not os.path.exists(path):
        raise ValueError(
            f"catalogue file not found: {path}. Point --l1-folder at the folder holding "
            "catalogues/ (the run's L1_folder, or this machine's copy of it).")
    use = [0] if cls in ("VGB", "GB") else list(ids)   # the loader's own rule
    with h5py.File(path, "r") as f:
        grp = f["Binaries"]
        return {i: loader.load_single_binary(grp, i, cls) for i in use}, path


# --------------------------------------------------------------------------
# injection rows (the run's own construction)
# --------------------------------------------------------------------------
def vgb_basis_for(ndim, requested="auto"):
    """The VGB basis: ``requested``, or by the stored column count."""
    if requested != "auto":
        if requested not in VGB_BASES:
            raise ValueError(f"--vgb-basis {requested!r} is not one of {VGB_BASES} or 'auto'")
        return requested
    if int(ndim) == 6:
        return "chirp"
    raise ValueError(
        f"the store's vgb rows have {ndim} columns: the legacy (lnA, fdot) and the distance "
        "bases both have 5, so pass --vgb-basis legacy or --vgb-basis distance (the run's "
        "VGB_SAMPLE_DISTANCE knob).")


def _vgb_rows(cat, basis):
    """``prepare_vgb_branch`` on a light setup: its ``injection``, f0 [mHz], names."""
    from lisatools.globalfit.stock.erebor.vgb import (
        VGBSettings,
        prepare_vgb_branch,
        vgb_sampled_basis,
    )

    v = VGBSettings()
    v.injection = None
    v.sample_distance = basis in ("distance", "chirp")
    v.chirp_mass_basis = basis == "chirp"
    gs = SimpleNamespace(catalogue={"VGB": cat}, domain_settings=None, Tobs=31536000.0,
                         min_freq=0.0, max_freq=1.0, mojito_data_path=None,
                         tdi_chan="XYZ", force_backend="cpu", ntemps=1)
    prepare_vgb_branch(v, gs, data_mode="mojito")
    f0 = np.asarray(v.fixed_params, dtype=float)[:, 0]
    return np.asarray(v.injection, dtype=float), f0, list(vgb_sampled_basis(v))


def injection_rows(branch, cat, *, vgb_basis=None):
    """``(rows, leaf labels, column names)`` for ``branch`` from ``cat``.

    ``rows[j]`` is leaf ``j``: the ``j``-th sorted catalogue id (VGB: the
    catalogue's file order), in the run's sampling basis.
    """
    from lisatools.globalfit.stock.erebor import transforms as T

    if branch == "vgb":
        rows, f0, names = _vgb_rows(cat, vgb_basis)
        return rows, [f"f0 {x:.6f} mHz" for x in f0], names
    ids = sorted(cat)
    labels = [f"id {i}" for i in ids]
    if branch == "mbh":
        from lisatools.globalfit.recipe import mbh_catalogue_to_sampling_basis

        rows = np.stack([mbh_catalogue_to_sampling_basis(cat[i]) for i in ids], axis=0)
        names = list(T.make_mbh_transform_container().input_basis)
    elif branch == "emri":
        from lisatools.sources.emri import emri_catalogue_to_waveform_basis

        full = np.asarray([emri_catalogue_to_waveform_basis(cat[i]) for i in ids])
        tc = T.make_emri_transform_container(full[:, [5, 12]])
        rows = tc.both_inverse_transforms(full)
        names = list(tc.input_basis)
    elif branch == "sobbh":
        from lisatools.globalfit.stock.erebor.injections import sobbh_catalogue_to_waveform_basis

        full = np.asarray([sobbh_catalogue_to_waveform_basis(cat[i]) for i in ids])
        tc = T.make_sobbh_transform_container()
        rows = np.stack([tc.both_inverse_transforms(r) for r in full], axis=0)
        names = list(tc.input_basis)
    else:
        raise ValueError(f"{branch!r} is not one of {BRANCHES}")
    return np.asarray(rows, dtype=float), labels, names


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
class BranchPlan:
    """One branch's edit: the chosen pairs, the rows and the dataset layout."""

    def __init__(self, branch):
        self.branch = branch
        self.walkers, self.leaves = [], []
        self.inj = None              # (nleaves, ndim)
        self.labels, self.names = [], []
        self.catalogue_path = None
        self.nwalkers = self.nleaves = self.ndim = self.ntemps = 0
        self.main_waxis = {}         # family -> (walker axis, leaf axis)
        self.has_snr = []            # of ("d_h", "h_h") present on the sub-backend
        self.kind = "per_leaf"       # or "banded" (vgb)

    def pairs(self):
        return [(w, l) for l in self.leaves for w in self.walkers]

    def fully_reset_leaves(self):
        return list(self.leaves) if len(self.walkers) == self.nwalkers else []


class Plan:
    def __init__(self, store, group, row, iteration, n_alloc):
        self.store, self.group = store, group
        self.row, self.iteration, self.n_alloc = int(row), int(iteration), int(n_alloc)
        self.branches = []


def _check_store(f, group):
    if group not in f:
        raise ValueError(f"no {group!r} group -- not a global-fit store")
    root = f[group]
    ra = dict(root.attrs)
    if int(ra.get("gf_format_version", 0)) != 2:
        raise ValueError(
            f"gf_format_version={ra.get('gf_format_version')!r}: this tool knows the "
            "cold-chain layout (format 2) only; the resume refuses older stores anyway.")
    return root, ra


def _pick(values, n, label, all_flag):
    if all_flag:
        return list(range(n))
    req = [int(x) for x in values]
    if not req:
        raise ValueError(f"no {label} chosen")
    if len(set(req)) != len(req):
        raise ValueError(f"--{label} repeats a value: {req}")
    for x in req:
        if not 0 <= x < n:
            plural = {"walker": "walkers", "leaf": "leaf slots"}[label]
            raise ValueError(f"--{label} {x} is outside the store's {n} {plural} (0..{n - 1})")
    return req


#: Stored ``key_order`` spellings that name the same column differently:
#: EMRI stores ``qS`` / ``qK`` for the sampled ``cosqS`` / ``cosqK``.
COLUMN_ALIASES = {"qs": "cosqs", "qk": "cosqk"}


def _column_key(name):
    """A column name with case and underscores ignored and the EMRI aliases
    (:data:`COLUMN_ALIASES`) applied; a cosine and an angle stay distinct."""
    n = str(name).lower().replace("_", "")
    return COLUMN_ALIASES.get(n, n)


def stored_columns(root, branch):
    """The store's own column names for ``branch`` (``key_order``), or ``None``
    when it records none or only column numbers (VGB)."""
    if "key_order" not in root or branch not in root["key_order"].attrs:
        return None
    raw = list(np.asarray(root["key_order"].attrs[branch]).ravel())
    out = [x.decode() if isinstance(x, bytes) else x for x in raw]
    return out if all(isinstance(x, str) for x in out) else None


def plan_branch(root, ra, branch, inj, row, walkers, leaves, *, all_walkers=False,
                all_leaves=False, names=None):
    """Validate one branch of the request against the open store ``root``.

    ``names``: the injection rows' column names; checked against the store's
    ``key_order`` where it records names, so a same-width basis change
    cannot put values into the wrong columns.
    """
    if branch not in BRANCHES:
        raise ValueError(f"{branch!r} is not one of the non-GB source branches {BRANCHES}")
    for fam in ("chain", "inds"):
        if fam not in root or branch not in root[fam]:
            raise ValueError(f"{branch!r}: no {fam}/{branch} in the main group")
    if "sub_backend" not in root or branch not in root["sub_backend"]:
        raise ValueError(f"{branch!r}: no sub_backend/{branch} group in the store")
    if "blobs" in root and branch in root["blobs"]:
        raise ValueError(f"blobs/{branch} exists: it carries a walker axis this tool does "
                         "not edit; refusing rather than leaving it inconsistent.")
    grp = root["sub_backend"][branch]
    sa = dict(grp.attrs)
    for need in ("ntemps", "nwalkers", "nleaves_max", "ndim"):
        if need not in sa:
            raise ValueError(f"sub_backend/{branch}: group attr {need!r} missing")
    nw, nl, nd, nt = (int(ra["nwalkers"]), int(sa["nleaves_max"]), int(sa["ndim"]),
                      int(sa["ntemps"]))
    if int(sa["nwalkers"]) != nw:
        raise ValueError(f"sub_backend/{branch} nwalkers={int(sa['nwalkers'])} != main {nw}")

    bp = BranchPlan(branch)
    bp.nwalkers, bp.nleaves, bp.ndim, bp.ntemps = nw, nl, nd, nt
    bp.kind = "banded" if branch == "vgb" else "per_leaf"

    inj = np.asarray(inj, dtype=float)
    if inj.shape != (nl, nd):
        raise ValueError(
            f"{branch}: the catalogue gives {inj.shape[0] if inj.ndim else 0} rows x "
            f"{inj.shape[-1] if inj.ndim else 0} columns, the store holds {nl} leaves x {nd} "
            "columns. Wrong id list (the run's MBH merger-window filter may have dropped "
            f"ids: run --branch {branch} on its own with --ids naming the ids it kept) or "
            "wrong basis; refusing rather than put a source into the wrong leaf.")
    bp.inj = inj
    stored = stored_columns(root, branch)
    if names is not None and stored is not None:
        keys = [_column_key(n) for n in names]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{branch}: the injection column names {list(names)} are not "
                             "unique; the key_order check cannot place them.")
        if keys != [_column_key(n) for n in stored]:
            raise ValueError(
                f"{branch}: the injection columns {list(names)} do not match the store's "
                f"key_order {stored}; refusing rather than write values into the wrong "
                "columns.")

    # every sub-backend dataset is known: a leaf-axis record the edit skipped
    # would leave the store inconsistent
    names = [k for k in grp if isinstance(grp[k], h5py.Dataset)]
    if bp.kind == "per_leaf":
        table = CSL.sub_layout(sa)
        unknown = [n for n in names if n not in table]
        for name in names:
            if name in table and tuple(grp[name].shape[1:]) != tuple(table[name][0]):
                raise ValueError(f"sub_backend/{branch}/{name}: shape {grp[name].shape} does "
                                 f"not match the expected (steps,) + {table[name][0]}")
    else:
        unknown = [n for n in names if n not in VGB_LEAF_DATASETS
                   and not n.startswith("band_") and n != "cap_edges"]
        want = {"chain": (nt, nw, nl, nd), "inds": (nt, nw, nl),
                "d_h": (nw, nl), "h_h": (nw, nl)}
        for name, shape in want.items():
            if name in grp and tuple(grp[name].shape[1:]) != shape:
                raise ValueError(f"sub_backend/{branch}/{name}: shape {grp[name].shape} does "
                                 f"not match the expected (steps,) + {shape}")
    if unknown:
        raise ValueError(f"sub_backend/{branch}: dataset(s) {unknown} are not in this tool's "
                         "layout table; refusing to guess whether they carry a leaf entry.")
    for need in ("chain", "inds"):
        if need not in names:
            raise ValueError(f"sub_backend/{branch}/{need} is missing")
    for name in ("chain", "inds", "d_h", "h_h"):
        if name in grp and int(grp[name].shape[0]) <= row:
            raise ValueError(f"sub_backend/{branch}/{name}: {grp[name].shape[0]} rows, the "
                             f"last stored row is {row}")
    bp.has_snr = [n for n in ("d_h", "h_h") if n in grp]

    ml = CSL.main_layout(ra, nl, nd)
    for fam in ("chain", "inds"):
        ds = root[fam][branch]
        trailing, waxis, laxis = ml[fam]
        if tuple(ds.shape[1:]) != tuple(trailing):
            raise ValueError(f"{fam}/{branch}: shape {ds.shape} does not match the expected "
                             f"(steps,) + {trailing}")
        if int(ds.shape[0]) <= row:
            raise ValueError(f"{fam}/{branch}: {ds.shape[0]} rows, the last stored row is {row}")
        bp.main_waxis[fam] = (waxis, laxis)

    bp.walkers = _pick(walkers, nw, "walker", all_walkers)
    bp.leaves = _pick(leaves, nl, "leaf", all_leaves)

    # liveness and the cold-row agreement of every chosen pair
    m_inds, m_chain = root["inds"][branch], root["chain"][branch]
    s_inds, s_chain = grp["inds"], grp["chain"]
    for w, l in bp.pairs():
        wa, la = bp.main_waxis["inds"]
        mi = np.asarray(m_inds[CSL._sel(m_inds.ndim, row, wa, w, la, l)], dtype=bool)
        si = np.asarray(s_inds[row, :, w, l], dtype=bool)
        if not (mi.all() and si.all()):
            raise ValueError(
                f"{branch} leaf {l} is not alive on walker {w} at row {row} (main inds "
                f"{mi.astype(int).ravel().tolist()}, per-rung inds {si.astype(int).tolist()}). "
                "These are fixed-leaf branches; this store is not what the tool expects.")
        wa, la = bp.main_waxis["chain"]
        mc = np.asarray(m_chain[CSL._sel(m_chain.ndim, row, wa, w, la, l)]).reshape(-1, nd)
        sc0 = np.asarray(s_chain[row, 0, w, l])
        if not all(CSL._same(c, sc0) for c in mc):
            raise ValueError(
                f"{branch} leaf {l}: walker {w}'s main cold row and its sub-backend rung 0 "
                f"disagree at row {row} (a torn save). Resuming that row already fails "
                "check_cold_row; restore the running backup first.")
    return bp


def plan_store(f, requests, group="global_fit", store=None):
    """``requests``: ``[(branch, inj, walkers, leaves, all_walkers, all_leaves[,
    column names])]``."""
    root, ra = _check_store(f, group)
    iteration = int(ra.get("iteration", 0))
    if iteration <= 0:
        raise ValueError("the store holds no saved rows (iteration attr 0)")
    first = requests[0][0]
    if "inds" not in root or first not in root["inds"]:
        raise ValueError(f"{first!r}: no inds/{first} in the main group")
    n_alloc = int(root["inds"][first].shape[0])
    if iteration > n_alloc:
        raise ValueError(f"iteration attr {iteration} > allocated rows {n_alloc}: a damaged store")
    plan = Plan(store, group, iteration - 1, iteration, n_alloc)
    for req in requests:
        branch, inj, walkers, leaves, all_w, all_l = req[:6]
        names = req[6] if len(req) > 6 else None
        plan.branches.append(plan_branch(root, ra, branch, inj, plan.row, walkers, leaves,
                                         all_walkers=all_w, all_leaves=all_l, names=names))
    return plan


# --------------------------------------------------------------------------
# reading, writing, verifying
# --------------------------------------------------------------------------
def current_cold(f, plan, bp):
    """``{(w, l): sub-backend rung-0 coords}`` before the edit."""
    s_chain = f[plan.group]["sub_backend"][bp.branch]["chain"]
    return {(w, l): np.asarray(s_chain[plan.row, 0, w, l]) for w, l in bp.pairs()}


def apply_reset(path, plan):
    """Write every planned slice in place. Returns ``[(dataset, selection, elements)]``."""
    written = []
    with h5py.File(path, "r+") as f:
        root = f[plan.group]
        for bp in plan.branches:
            m_chain = root["chain"][bp.branch]
            sub = root["sub_backend"][bp.branch]
            wa, la = bp.main_waxis["chain"]
            for w, l in bp.pairs():
                sel = CSL._sel(m_chain.ndim, plan.row, wa, w, la, l)
                block = np.broadcast_to(bp.inj[l], m_chain[sel].shape)
                m_chain[sel] = block
                written.append((f"chain/{bp.branch}", CSL._fmt_sel(sel), int(block.size)))
                block = np.broadcast_to(bp.inj[l], (bp.ntemps, bp.ndim))
                sub["chain"][plan.row, :, w, l] = block
                written.append((f"sub_backend/{bp.branch}/chain",
                                f"[{plan.row}, :, {w}, {l}, :]", int(block.size)))
                for name in bp.has_snr:
                    sub[name][plan.row, w, l] = np.nan
                    written.append((f"sub_backend/{bp.branch}/{name}",
                                    f"[{plan.row}, {w}, {l}]", 1))
        f.flush()
    return written


def verify_reset(path, plan):
    """``[]`` when every planned pair holds the injection on every rung, its
    main cold row agrees with its rung 0, its leaf is alive and its d_h / h_h
    read NaN; else the list of problems."""
    problems = []
    with h5py.File(path, "r") as f:
        root = f[plan.group]
        for bp in plan.branches:
            m_chain, m_inds = root["chain"][bp.branch], root["inds"][bp.branch]
            sub = root["sub_backend"][bp.branch]
            for w, l in bp.pairs():
                want = np.asarray(bp.inj[l], dtype=sub["chain"].dtype)
                rungs = np.asarray(sub["chain"][plan.row, :, w, l])
                if not np.array_equal(rungs, np.broadcast_to(want, rungs.shape)):
                    bad = [k for k in range(rungs.shape[0]) if not np.array_equal(rungs[k], want)]
                    problems.append(f"sub_backend/{bp.branch}/chain[{plan.row}, :, {w}, {l}]: "
                                    f"rungs {bad} differ from the injection")
                wa, la = bp.main_waxis["chain"]
                mc = np.asarray(m_chain[CSL._sel(m_chain.ndim, plan.row, wa, w, la, l)])
                if not np.array_equal(mc.reshape(-1, bp.ndim),
                                      np.broadcast_to(np.asarray(bp.inj[l], dtype=mc.dtype),
                                                      (mc.size // bp.ndim, bp.ndim))):
                    problems.append(f"chain/{bp.branch} walker {w} leaf {l}: the main cold row "
                                    "differs from the injection")
                if not CSL._same(mc.reshape(-1, bp.ndim)[0], rungs[0]):
                    problems.append(f"{bp.branch} walker {w} leaf {l}: main cold row != "
                                    "sub-backend rung 0 (check_cold_row would abort)")
                wa, la = bp.main_waxis["inds"]
                mi = np.asarray(m_inds[CSL._sel(m_inds.ndim, plan.row, wa, w, la, l)], dtype=bool)
                si = np.asarray(sub["inds"][plan.row, :, w, l], dtype=bool)
                if not (mi.all() and si.all()):
                    problems.append(f"{bp.branch} walker {w} leaf {l}: not alive on every rung")
                for name in bp.has_snr:
                    if not np.isnan(float(sub[name][plan.row, w, l])):
                        problems.append(f"sub_backend/{bp.branch}/{name}[{plan.row}, {w}, {l}] "
                                        "is not NaN")
    return problems


# --------------------------------------------------------------------------
# eigen-table sidecar
# --------------------------------------------------------------------------
def eigen_path(store):
    return os.path.splitext(store)[0] + EIGEN_SUFFIX


def eigen_plan(store, plan, keep=False):
    """``(path, payload | None, drop keys, lines)`` for the report and the write."""
    path = eigen_path(store)
    if not os.path.exists(path):
        return path, None, [], [f"eigen   {os.path.basename(path)}: absent (nothing to do)"]
    try:
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        entries = payload["entries"]
    except Exception as exc:  # noqa: BLE001 -- the run treats it as empty too
        return path, None, [], [f"eigen   {os.path.basename(path)}: unreadable "
                                f"({type(exc).__name__}); the run ignores it and rebuilds"]
    drop, lines = [], [f"eigen   {os.path.basename(path)}:"]
    for bp in plan.branches:
        full = set(bp.fully_reset_leaves())
        gone, kept, absent = [], [], []
        for l in bp.leaves:
            key = f"{bp.branch}:{int(l)}"
            if key not in entries:
                absent.append(l)
            elif l in full and not keep:
                drop.append(key)
                gone.append(l)
            else:
                kept.append(l)
        if gone:
            lines.append(f"          {bp.branch} leaves {gone}: entries DROPPED at --apply (reset "
                         "on every walker; the first visit builds them at the injection)")
        if kept:
            why = ("--keep-eigen-tables" if keep and set(kept) <= full else
                   f"reset on walkers {bp.walkers} of {bp.nwalkers} only; a walker-independent "
                   "table")
            lines.append(f"          {bp.branch} leaves {kept}: entries kept ({why})")
        if absent:
            lines.append(f"          {bp.branch}: {len(absent)} chosen lea"
                         f"{'f has' if len(absent) == 1 else 'ves have'} no entry (built at "
                         "the first visit)")
    return path, payload, drop, lines


def drop_eigen_entries(path, payload, drop):
    """Back the sidecar up (never overwriting a backup), then rewrite it
    atomically without the ``drop`` keys. Returns the backup path."""
    bak = CSL._free_name(path + BACKUP_SUFFIX)
    shutil.copy2(path, bak)
    for k in drop:
        payload["entries"].pop(k, None)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return bak


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _rel(now, inj):
    return float(np.median(np.abs(now - inj) / (np.abs(inj) + 1e-30)))


def report(plan, before, label="store"):
    print(f"\n  == {label}: {plan.store}")
    print(f"     stored iteration {plan.iteration}, allocated rows {plan.n_alloc}; editing row "
          f"{plan.row} (the row the resume reads) only")
    for bp in plan.branches:
        cold = before[bp.branch]
        print(f"\n     -- {bp.branch}: {bp.nleaves} leaves x {bp.nwalkers} walkers stored, "
              f"{bp.ntemps} rungs per leaf; resetting walkers {bp.walkers} x leaves "
              f"{bp.leaves} on every rung")
        print(f"        catalogue {bp.catalogue_path}")
        if bp.kind == "per_leaf":
            print("        leaf slots (sorted catalogue ids): "
                  + ", ".join(f"{j}={bp.labels[j]}" for j in range(bp.nleaves)))
        else:
            print(f"        leaf slots: the catalogue's {bp.nleaves} sources in file order "
                  f"({bp.labels[0]} ... {bp.labels[-1]})")
        # a VGB leaf's identity is pinned by its per-leaf f0 / sky fills (the
        # same file order), and its (dist, Mc) wander along the amplitude
        # degeneracy, so the far-from-catalogue flag is for the others only
        flaggable = bp.kind == "per_leaf"
        far = []
        if len(bp.leaves) <= 3:
            for l in bp.leaves:
                print(f"\n        leaf {l} ({bp.labels[l]}), rung-0 coords now vs the injection")
                head = "".join(f"{'w' + str(w) + ' now':>16}" for w in bp.walkers)
                print(f"          {'column':<18}{'injection':>16}{head}")
                for k, name in enumerate(bp.names):
                    vals = "".join(f"{cold[(w, l)][k]:16.8g}" for w in bp.walkers)
                    print(f"          {name:<18}{bp.inj[l][k]:16.8g}{vals}")
                rels = [_rel(cold[(w, l)], bp.inj[l]) for w in bp.walkers]
                if flaggable and min(rels) > FAR_REL_CHANGE:
                    far.append(l)
        else:
            print(f"\n        {'leaf':>4}  {'source':<22} median |relative change| per walker "
                  f"{bp.walkers}")
            for l in bp.leaves:
                rels = [_rel(cold[(w, l)], bp.inj[l]) for w in bp.walkers]
                flag = "   <- CHECK" if flaggable and min(rels) > FAR_REL_CHANGE else ""
                if flag:
                    far.append(l)
                print(f"        {l:>4}  {bp.labels[l]:<22} "
                      + " ".join(f"{r:9.2e}" for r in rels) + flag)
        if not flaggable:
            print("        (VGB: the leaf slot is pinned to its source by the per-leaf f0 / sky "
                  "fills; large changes follow the dist-Mc amplitude degeneracy)")
        if far:
            print(f"        CHECK leaves {far}: far from their catalogue row on every chosen "
                  "walker (median |relative change| > "
                  f"{FAR_REL_CHANGE}). A prior-only source that wandered, or a wrong id "
                  "list: confirm before --apply.")
        snr = " / ".join(bp.has_snr) or "none"
        print(f"        written per pair: chain/{bp.branch} (main cold), "
              f"sub_backend/{bp.branch}/chain (all {bp.ntemps} rungs) = injection; "
              f"{snr} = NaN")
        if bp.kind == "per_leaf":
            print(f"        NOT written: sub_backend/{bp.branch}/log_like, log_prior (rescored "
                  "at the first visit), betas_all and the counters (walker-free)")
        else:
            print("        NOT written: the band_* records (a VGB leaf's band is fixed)")
    print("     NOT written: the main log_like / log_prior totals (the resume rebuilds the "
          "residual and re-evaluates them), every other branch, walker, leaf and row")


def report_sidecars(store, group, eig_lines):
    side = sidecar_paths(store)
    print("\n  sidecars:")
    twin = side["backup"]
    if os.path.exists(twin):
        it = CSL._twin_iteration(twin, group)
        print(f"    backup  {os.path.basename(twin)}: iteration {it}, modified "
              f"{CSL._age(twin):.0f} s ago. Read ONLY if the store is unreadable at resume; the "
              "saver overwrites it after the first save. --also-backup edits it too.")
    else:
        print(f"    backup  {os.path.basename(twin)}: absent")
    midit = side["midit"]
    if os.path.exists(midit):
        print(f"    midit   {os.path.basename(midit)}: PRESENT ({os.path.getsize(midit) / MB:.1f} "
              f"MB, modified {CSL._age(midit):.0f} s ago). It would WIN over the edited store at "
              "resume: --apply refuses unless --reset-midit moves it aside.")
    else:
        print(f"    midit   {os.path.basename(midit)}: absent (good)")
    for line in eig_lines:
        print(f"    {line}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _branches(text):
    if str(text).strip().lower() == "all":
        return list(BRANCHES)
    out = [b.strip().lower() for b in str(text).split(",") if b.strip()]
    bad = [b for b in out if b not in BRANCHES]
    if bad:
        raise ValueError(f"--branch {bad} not in {BRANCHES} (or 'all'); GB and the noise "
                         "branches are out of scope")
    if len(set(out)) != len(out) or not out:
        raise ValueError(f"--branch {text!r}: empty or repeated")
    return out


def build_requests(f, args, branches, settings):
    """Per branch: the injection rows (from the catalogue) and the selection."""
    root, _ = _check_store(f, args.group)
    reqs, meta = [], {}
    for b in branches:
        cls = CATALOGUE_CLASS[b]
        ids = parse_ids(args.ids) if args.ids else settings["source_ids"].get(cls, [])
        if not ids:
            raise ValueError(f"{b}: no {cls} ids in the run settings; pass --ids")
        if b not in root["chain"]:
            raise ValueError(f"{b!r}: no chain/{b} in the main group")
        ndim = int(root["chain"][b].shape[-1])
        basis = vgb_basis_for(ndim, args.vgb_basis) if b == "vgb" else None
        cat, path = load_catalogue(settings["l1_folder"], cls, ids)
        inj, labels, names = injection_rows(b, cat, vgb_basis=basis)
        reqs.append((b, inj, args.walker, args.leaf, args.all_walkers, args.all_leaves, names))
        meta[b] = (labels, names, path)
    return reqs, meta


def _plan_file(path, args, branches, settings):
    try:
        with h5py.File(path, "r") as f:
            reqs, meta = build_requests(f, args, branches, settings)
            plan = plan_store(f, reqs, group=args.group, store=path)
            before = {}
            for bp in plan.branches:
                bp.labels, bp.names, bp.catalogue_path = meta[bp.branch]
                before[bp.branch] = current_cold(f, plan, bp)
    except OSError as exc:
        raise ValueError(f"cannot open {path} ({exc}). If the run (or its monitor/saver) "
                         "still holds it, stop it first.") from None
    return plan, before


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="path to the run's *_testing.h5")
    ap.add_argument("--branch", required=True,
                    help="vgb, mbh, emri or sobbh; a comma list or 'all' (needs --all-leaves)")
    ap.add_argument("--walker", type=int, action="append", default=[],
                    help="walker to reset (repeatable)")
    ap.add_argument("--all-walkers", action="store_true", help="reset every walker")
    ap.add_argument("--leaf", type=int, action="append", default=[],
                    help="leaf slot to reset (repeatable)")
    ap.add_argument("--all-leaves", action="store_true", help="reset every leaf")
    ap.add_argument("--run-settings", default=None,
                    help="the run's run_settings.log (default: <store dir>/*_artifacts/)")
    ap.add_argument("--l1-folder", default=None,
                    help="folder holding catalogues/ (default: the run's L1_folder)")
    ap.add_argument("--ids", default=None,
                    help="catalogue ids of the ONE --branch, comma list (default: the run's)")
    ap.add_argument("--vgb-basis", default="auto", choices=("auto",) + VGB_BASES,
                    help="VGB sampling basis (auto: 6 stored columns = chirp)")
    ap.add_argument("--group", default="global_fit")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--verify", action="store_true",
                    help="read-only: report whether the chosen pairs already hold the injection")
    ap.add_argument("--no-backup", action="store_true",
                    help=f"skip the {BACKUP_SUFFIX} copy of the store before writing")
    ap.add_argument("--also-backup", action="store_true",
                    help="apply the same edit to *_running_backup_copy.h5 (its own last row)")
    ap.add_argument("--reset-midit", action="store_true",
                    help="move *_midit_checkpoint.pkl aside (it would override the edited store)")
    ap.add_argument("--keep-eigen-tables", action="store_true",
                    help="keep the eigen-table sidecar entries of leaves reset on every walker")
    ap.add_argument("--live-window", type=float, default=300.0,
                    help="--apply refuses a store or writer sidecar modified this recently (s)")
    ap.add_argument("--force", action="store_true",
                    help="override the --live-window refusal; the job MUST be down")
    args = ap.parse_args(argv)
    cmd = " ".join(shlex.quote(a) for a in (sys.argv[1:] if argv is None else argv))
    script = os.path.relpath(os.path.abspath(__file__))

    store = args.store
    if not os.path.exists(store):
        print(f"no such store: {store}")
        return 2
    try:
        branches = _branches(args.branch)
    except ValueError as exc:
        print(f"REFUSING: {exc}")
        return 2
    if bool(args.walker) == bool(args.all_walkers):
        print("REFUSING: give --walker W (repeatable) or --all-walkers, not both / neither")
        return 2
    if bool(args.leaf) == bool(args.all_leaves):
        print("REFUSING: give --leaf L (repeatable) or --all-leaves, not both / neither")
        return 2
    if len(branches) > 1 and not args.all_leaves:
        print("REFUSING: several branches need --all-leaves (a leaf slot is a different "
              "source on each branch)")
        return 2
    if args.ids and len(branches) > 1:
        print("REFUSING: --ids names the catalogue ids of ONE --branch")
        return 2
    if args.ids and branches == ["vgb"]:
        print("REFUSING: --ids does not apply to vgb (one catalogue file; the leaves are its "
              "rows in file order)")
        return 2

    if args.run_settings is None:
        cands = find_run_settings(store)
        ids_known = bool(args.ids) or branches == ["vgb"]    # vgb needs no ids
        if len(cands) != 1 and not (args.l1_folder and ids_known):
            print(f"REFUSING: {len(cands)} run_settings.log candidates next to the store "
                  f"({cands}); pass --run-settings, or --l1-folder plus (except for "
                  "--branch vgb) --ids")
            return 2
        args.run_settings = cands[0] if len(cands) == 1 else None
    settings = {"l1_folder": None, "source_ids": {}}
    if args.run_settings is not None:
        try:
            settings = read_run_settings(args.run_settings)
        except (OSError, ValueError) as exc:
            print(f"REFUSING: {exc}")
            return 2
    if args.l1_folder:
        settings["l1_folder"] = args.l1_folder
    if not settings["l1_folder"]:
        print("REFUSING: no L1 folder (pass --l1-folder)")
        return 2

    holders = pids_holding(store)
    if holders:
        print(f"REFUSING: {store} is held open by PID(s) {holders}; stop the run first.")
        return 2
    try:
        plan, before = _plan_file(store, args, branches, settings)
    except ValueError as exc:
        print(f"REFUSING: {exc}")
        return 2
    print(f"{store}\n  run settings {args.run_settings or '(none: --l1-folder + --ids)'}; "
          f"L1 folder {settings['l1_folder']}")

    if args.verify and not args.apply:
        problems = verify_reset(store, plan)
        for p in problems:
            print(f"    DIFFERS: {p}")
        print("  VERIFY OK: every chosen pair holds the injection on every rung"
              if not problems else f"  VERIFY FAILED ({len(problems)} problem(s))")
        return 0 if not problems else 1

    report(plan, before)
    eig_path, eig_payload, eig_drop, eig_lines = eigen_plan(store, plan,
                                                            keep=args.keep_eigen_tables)
    report_sidecars(store, args.group, eig_lines)

    twin = sidecar_paths(store)["backup"]
    twin_plan = None
    if args.also_backup:
        if not os.path.exists(twin):
            print(f"\nREFUSING: --also-backup but {twin} does not exist")
            return 2
        try:
            twin_plan, twin_before = _plan_file(twin, args, branches, settings)
        except ValueError as exc:
            print(f"\nREFUSING (running backup twin): {exc}")
            return 2
        report(twin_plan, twin_before, label="running backup twin")

    if not args.apply:
        print("\n  DRY RUN -- nothing written.")
        print(f"  to write: python {script} {cmd} --apply")
        recent = CSL.recent_writes(store, args.live_window)
        if recent:
            print("  WARNING: modified within --live-window, is the run down? "
                  + ", ".join(f"{os.path.basename(p)} ({a:.0f} s)" for p, a in recent))
        return 0

    # ---- apply gates -------------------------------------------------------
    recent = CSL.recent_writes(store, args.live_window)
    if recent and not args.force:
        print("\nREFUSING: modified within the last "
              f"{args.live_window:.0f} s, so the run is probably still writing: "
              + ", ".join(f"{p} ({a:.0f} s ago)" for p, a in recent)
              + ". Stop the job (or pass --force if you are certain it is down). Two writers "
              "on one HDF5 file corrupt it.")
        return 2
    midit = sidecar_paths(store)["midit"]
    if os.path.exists(midit) and not args.reset_midit:
        print(f"\nREFUSING: {midit} exists and would override the edited store at resume "
              "(midit_checkpoint.load_for_resume). Re-run with --reset-midit to move it aside.")
        return 2
    targets = [plan] + ([twin_plan] if twin_plan is not None else [])
    for p in targets:
        try:
            with h5py.File(p.store, "r+"):
                pass
        except OSError as exc:
            print(f"\nREFUSING: cannot open {p.store} for writing ({exc}). Something still "
                  "holds it (the run, its saver or the monitor); stop it first.")
            return 2
    if not args.no_backup:
        for p in targets:
            if os.path.exists(p.store + BACKUP_SUFFIX):
                print(f"\nREFUSING: {p.store + BACKUP_SUFFIX} already exists; move or delete it "
                      "first so an earlier backup is never overwritten (or pass --no-backup).")
                return 2
        need = sum(os.path.getsize(p.store) for p in targets)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(store))).free
        if free < need:
            print(f"\nREFUSING: {free / MB:.0f} MB free, the backup copies need "
                  f"{need / MB:.0f} MB (or pass --no-backup).")
            return 2

    ts = time.strftime("%Y%m%d-%H%M%S")
    status = 0
    if not args.no_backup:
        for p in targets:
            bak = p.store + BACKUP_SUFFIX
            print(f"\n  copying {os.path.getsize(p.store) / MB:.1f} MB -> {bak} ...")
            shutil.copy2(p.store, bak)
    moved = None
    if os.path.exists(midit):
        moved = CSL._free_name(f"{midit}.stale-{ts}")
        os.rename(midit, moved)
        print(f"\n  moved aside: {midit} -> {moved}")

    def restore(path):
        how = (f"restore {path + BACKUP_SUFFIX} over {path}" if not args.no_backup
               else f"{path} has no backup (--no-backup)")
        if moved:
            how += f", and move {moved} back to {midit}"
        return how

    for i, p in enumerate(targets):
        later = ("; the running backup twin was NOT edited"
                 if i == 0 and len(targets) > 1 else "")
        print(f"\n  writing {p.store}:")
        try:
            written = apply_reset(p.store, p)
        except Exception as exc:  # noqa: BLE001 -- reported with the way back, then re-raised
            print(f"    FAILED while writing ({type(exc).__name__}: {exc}) -- {restore(p.store)}"
                  f" before resuming{later}; the eigen sidecar is untouched.")
            raise
        per_ds = {}
        for path, _sel, n in written:
            k, e = per_ds.get(path, (0, 0))
            per_ds[path] = (k + 1, e + n)
        for path, (k, e) in per_ds.items():
            print(f"    {path:<34} {k:>5} slice(s) {e:>8} element(s)")
        problems = verify_reset(p.store, p)
        for prob in problems:
            print(f"    VERIFY: {prob}")
        ok, msg = CSL.resume_readable(p.store)
        print(f"    {msg}")
        if problems or not ok:
            status = 1
            print(f"    FAILED -- {restore(p.store)} before resuming{later}; the eigen "
                  "sidecar is untouched.")
            break
        else:
            print("    VERIFY OK: every chosen pair holds the injection on every rung at row "
                  f"{p.row}, its main cold row agrees with rung 0.")
    if status == 0 and eig_drop:
        bak = drop_eigen_entries(eig_path, eig_payload, eig_drop)
        print(f"\n  eigen sidecar: dropped {len(eig_drop)} entr"
              f"{'y' if len(eig_drop) == 1 else 'ies'} ({', '.join(eig_drop)}); backup {bak}")
    if status == 0:
        print("\n  DONE. Relaunch the run with its usual launch line.")
    return status


if __name__ == "__main__":
    sys.exit(main())
