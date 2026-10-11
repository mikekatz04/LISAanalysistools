"""Copy one source leaf's state from one walker to another in a STOPPED store.

Why this exists (9mo production, 2026-10-10/11): EMRI leaf 3 on walker 0 sat
at a boundary maximum of the wrong mode (cos(qK) at the prior edge, SNR 22
against 45 on walkers 1-3, 300-650 nats below) and did not move through
dozens of source blocks. Its own 8-rung ladder was 60-450 nats below its cold
rung, so no swap could lift it. This tool copies another walker's state of
that leaf, on EVERY rung, into the stuck walker while the run is down; the
next resume starts the destination walker from the source walker's position.

Scope: the per-leaf-ladder source branches ``mbh`` / ``emri`` / ``sobbh``
(fixed leaf count, one ladder per leaf, the same leaf slot on every walker).
REFUSED: ``gb`` / ``vgb`` (banded: per-(rung, walker, band) occupancy, per-
walker band valves and caps, leaf slots not aligned across walkers) and the
noise branches (one leaf on a flat ladder; graft_noise_state.py moves noise).

What a resume reads, and therefore what is copied
-------------------------------------------------
``run.py`` loads the state in this order:

1. ``promote_backup_if_store_unreadable``: the ``*_running_backup_copy.h5``
   twin replaces the store ONLY when the store fails
   ``_validate_resume_readable``.
2. ``midit_checkpoint.load_for_resume``: a ``*_midit_checkpoint.pkl`` written
   at or after the store's last row WINS over the store, so an edit to the h5
   would be silently ignored. ``--apply`` refuses while one exists unless
   ``--reset-midit`` moves it aside (the launchers default it off).
3. ``backend.get_last_sample()``: row ``iteration - 1`` of the main group
   (``chain/<b>``, ``inds/<b>``, ``log_like``, ``log_prior``, ``betas``) and
   of every ``sub_backend/<b>`` dataset (``PerLeafLadderState.from_stored``:
   ``chain``, ``inds``, ``log_like``, ``log_prior``, ``betas_all``, ``d_h``,
   ``h_h``, counters).

Then ``setup_acs(state, rebuild_residuals=True)`` subtracts every walker's
templates built from the MAIN cold-row coords, so the main ``chain/<b>`` row
is what moves the destination's residual; the first source propose checks
the main cold row against the sub-backend rung 0 (``check_cold_row``), so both
are copied together. Per (walker, leaf) the tool copies, at the chosen row:

    chain/<b>                 [row, :, :, w, leaf, :]   main cold coords
    inds/<b>                  [row, :, :, w, leaf]
    sub_backend/<b>/chain     [row, :, w, leaf, :]      every rung
    sub_backend/<b>/inds      [row, :, w, leaf]
    sub_backend/<b>/log_like  [row, leaf, :, w]         every rung
    sub_backend/<b>/log_prior [row, leaf, :, w]
    sub_backend/<b>/d_h       [row, w, leaf]
    sub_backend/<b>/h_h       [row, w, leaf]

What the resume RECOMPUTES rather than trusts (copied anyway, so the store
reads consistently):

* the main per-walker ``log_like`` is re-evaluated from the rebuilt residual
  (``state.log_like[:] = self._global_likelihood(acs)``) and ``log_prior`` is
  zeroed. Both are walker TOTALS over every branch, so they cannot be copied
  leaf-wise and are NOT touched; the dry run prints them.
* the per-leaf ``log_like`` / ``log_prior``: every leaf visit recomputes
  ``prev_logl`` / ``prev_logp`` from the sub-state coords of every rung
  against the exposed residual (``ResidualAddOneRemoveOneMove.propose_local``);
  the stored values are telemetry written at the end of each visit. NOTE the
  per-leaf ``log_like`` is the walker's FULL lnL with that leaf at that rung
  (on the 9mo store it sits within ~1e3 of the main 1.58e8), so after the copy
  the destination's entry reads the SOURCE walker's total until its next visit.
* ``d_h`` / ``h_h`` are reset to NaN at the start of every propose.

Walker-free, so never touched: the leaf's ladder ``betas_all[row, leaf]``
(shared by every walker) and the per-(leaf, rung) delta counters.

The eigen-table sidecar (``*_eigen_tables.pkl``) is reported, not edited: the
production scope ``walker_max`` keeps ONE walker-independent table per leaf;
a per-walker stash would keep the destination's old rows until its next
refresh (a stale proposal scale, still correct Metropolis-Hastings).

The running backup twin is NOT edited by default: it is read only when the
store is unreadable at resume, and the saver overwrites it with a copy of the
edited store after the first save of the resumed job. ``--also-backup``
applies the same edit to it (its own last row).

Usage (dry run first; nothing is written without ``--apply``)::

    python scripts/fstat_proposal/copy_store_leaf.py \\
        --store /shared/data/global_fit_output/gf_prod_9mo_v9/gf_prod_9mo_testing.h5 \\
        --branch emri --leaf 3 --src-walker 2 --dst-walker 0
    ... same command with --apply

``--apply`` first copies the store to ``gf_prod_9mo_testing.h5.copyleaf.bak``
(refusing if that exists; ``--no-backup`` skips it), writes in place, then
re-reads and verifies the destination against the source on every rung and
runs lisatools' resume-readability check. ``--verify`` alone is a read-only
check of a store already edited.
"""

import argparse
import os
import pickle
import shlex
import shutil
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compact_gf_store import pids_holding, sidecar_paths  # noqa: E402

MB = 1024.0 ** 2

#: Branches this tool copies: one ladder per leaf, the same slot on every walker.
PER_LEAF_BRANCHES = ("mbh", "emri", "sobbh")
#: Banded branches: band bookkeeping per (rung, walker, band); out of scope.
BANDED_REFUSAL = ("gb", "vgb")
#: Noise branches: one leaf on a flat ladder; graft_noise_state.py moves noise.
NOISE_REFUSAL = ("psd", "galfor", "sgwb")

#: Appended to the store (and the twin) path for the pre-write copy.
BACKUP_SUFFIX = ".copyleaf.bak"
#: lisatools.globalfit.moves.eigen_table_persist.SIDECAR_SUFFIX
EIGEN_SUFFIX = "_eigen_tables.pkl"

#: The per-(leaf, rung) delta counters of a per-leaf-ladder sub-backend.
COUNTER_NAMES = ("in_model_accepted", "in_model_proposed", "rj_accepted", "rj_proposed")
SWAP_COUNTER_NAMES = ("swaps_accepted", "swaps_proposed")


# --------------------------------------------------------------------------
# layout tables
# --------------------------------------------------------------------------
def sub_layout(attrs):
    """``{name: (trailing shape, walker axis | None, leaf axis)}`` for one
    ``sub_backend/<branch>`` group of a per-leaf-ladder branch.

    Axes index the FULL dataset (the step axis is 0); a ``None`` walker axis
    marks a walker-free dataset that is never touched. Written out, never
    inferred from lengths: for mbh ``nleaves_max == nwalkers`` is common, and
    ``log_like`` puts the walker LAST, ``(nleaves, ntemps, nwalkers)``.
    """
    nt, nw = int(attrs["ntemps"]), int(attrs["nwalkers"])
    nl, nd = int(attrs["nleaves_max"]), int(attrs["ndim"])
    out = {
        "chain": ((nt, nw, nl, nd), 2, 3),
        "inds": ((nt, nw, nl), 2, 3),
        "log_like": ((nl, nt, nw), 3, 1),
        "log_prior": ((nl, nt, nw), 3, 1),
        "d_h": ((nw, nl), 1, 2),
        "h_h": ((nw, nl), 1, 2),
        # the leaf's own ladder: shared by every walker
        "betas_all": ((nl, nt), None, 1),
    }
    for name in COUNTER_NAMES:
        out[name] = ((nl, nt), None, 1)
    for name in SWAP_COUNTER_NAMES:
        out[name] = ((nl, max(nt - 1, 0)), None, 1)
    return out


def main_layout(root_attrs, nleaves, ndim):
    """``{family: (trailing shape, walker axis, leaf axis)}`` for the main
    group's ``chain/<b>`` and ``inds/<b>``: ``(step, nsamplers, ntemps,
    nwalkers, nleaves[, ndim])``, or without the sampler axis on a store
    written before eryn grew one."""
    nt, nw = int(root_attrs["ntemps"]), int(root_attrs["nwalkers"])
    nl, nd = int(nleaves), int(ndim)
    if "nsamplers" in root_attrs:
        ns = int(root_attrs["nsamplers"])
        return {"chain": ((ns, nt, nw, nl, nd), 3, 4), "inds": ((ns, nt, nw, nl), 3, 4)}
    return {"chain": ((nt, nw, nl, nd), 2, 3), "inds": ((nt, nw, nl), 2, 3)}


def _sel(ndim, row, waxis, walker, laxis, leaf):
    idx = [slice(None)] * int(ndim)
    idx[0] = int(row)
    idx[waxis] = int(walker)
    idx[laxis] = int(leaf)
    return tuple(idx)


def _fmt_sel(sel):
    return "[" + ", ".join(":" if isinstance(s, slice) else str(s) for s in sel) + "]"


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
class Entry:
    """One dataset the copy writes: ``path`` relative to the root group."""

    __slots__ = ("path", "waxis", "laxis", "ndim")

    def __init__(self, path, waxis, laxis, ndim):
        self.path, self.waxis, self.laxis, self.ndim = path, waxis, laxis, ndim

    def src(self, plan, leaf):
        return _sel(self.ndim, plan.row, self.waxis, plan.src, self.laxis, leaf)

    def dst(self, plan, leaf):
        return _sel(self.ndim, plan.row, self.waxis, plan.dst, self.laxis, leaf)


class Plan:
    def __init__(self, store, group, branch, src, dst, row, iteration, n_alloc):
        self.store, self.group, self.branch = store, group, branch
        self.src, self.dst, self.row = int(src), int(dst), int(row)
        self.iteration, self.n_alloc = int(iteration), int(n_alloc)
        self.leaves = []           # leaves copied
        self.skipped = []          # (--all-leaves) dead on both walkers
        self.entries = []          # Entry, main group first
        self.walker_free = []      # sub-backend datasets left alone
        self.ntemps = 0
        self.nwalkers = 0


def _refuse_branch(branch):
    if branch in BANDED_REFUSAL:
        raise ValueError(
            f"{branch!r} is a banded branch: its sub-backend keeps per-(rung, walker, band) "
            "occupancy (band_num_binaries), per-walker band valves/caps and leaf slots that "
            "are NOT aligned across walkers. Copying a leaf slot between walkers is out of "
            "scope for this tool.")
    if branch in NOISE_REFUSAL:
        raise ValueError(
            f"{branch!r} is a noise branch (one leaf on a flat ladder), not a per-leaf "
            "source; moving a walker's noise state is graft_noise_state.py's job.")
    if branch not in PER_LEAF_BRANCHES:
        raise ValueError(
            f"{branch!r} is not one of the per-leaf-ladder source branches "
            f"{PER_LEAF_BRANCHES}.")


def plan_copy(f, branch, leaves, src, dst, row="last", group="global_fit",
              all_leaves=False, store=None):
    """Validate the request against the store and describe the copy.

    Raises ``ValueError`` with the refusal reason. ``row`` is ``"last"`` (the
    row a resume reads, ``iteration - 1``) or a stored row index.
    """
    _refuse_branch(branch)
    if group not in f:
        raise ValueError(f"no {group!r} group -- not a global-fit store")
    root = f[group]
    ra = dict(root.attrs)
    if int(ra.get("gf_format_version", 0)) != 2:
        raise ValueError(
            f"gf_format_version={ra.get('gf_format_version')!r}: this tool knows the "
            "cold-chain layout (format 2) only; the resume refuses older stores anyway.")
    for fam in ("chain", "inds"):
        if fam not in root or branch not in root[fam]:
            raise ValueError(f"{branch!r}: no {fam}/{branch} in the main group")
    if "sub_backend" not in root or branch not in root["sub_backend"]:
        raise ValueError(f"{branch!r}: no sub_backend/{branch} group in the store")
    if "blobs" in root and branch in root["blobs"]:
        raise ValueError(
            f"blobs/{branch} exists: it carries a walker axis this tool does not copy; "
            "refusing rather than leaving it inconsistent.")
    grp = root["sub_backend"][branch]
    sa = dict(grp.attrs)
    for need in ("ntemps", "nwalkers", "nleaves_max", "ndim"):
        if need not in sa:
            raise ValueError(f"sub_backend/{branch}: group attr {need!r} missing")
    nw, nl = int(ra["nwalkers"]), int(sa["nleaves_max"])
    if int(sa["nwalkers"]) != nw:
        raise ValueError(
            f"sub_backend/{branch} nwalkers={int(sa['nwalkers'])} != main nwalkers={nw}")
    if "betas_all" not in grp:
        raise ValueError(f"sub_backend/{branch}: no betas_all -- not a per-leaf-ladder branch")

    src, dst = int(src), int(dst)
    if src == dst:
        raise ValueError(f"--src-walker and --dst-walker are both {src}: nothing to copy")
    for label, w in (("--src-walker", src), ("--dst-walker", dst)):
        if not 0 <= w < nw:
            raise ValueError(f"{label} {w} is outside the store's {nw} walkers (0..{nw - 1})")

    iteration = int(ra.get("iteration", 0))
    n_alloc = int(root["inds"][branch].shape[0])
    if iteration <= 0:
        raise ValueError("the store holds no saved rows (iteration attr 0)")
    if iteration > n_alloc:
        raise ValueError(
            f"iteration attr {iteration} > allocated rows {n_alloc}: a damaged store")
    if row in (None, "last"):
        row = iteration - 1
    row = int(row)
    if not 0 <= row < iteration:
        raise ValueError(f"--row {row} is not a stored row (0..{iteration - 1})")

    if all_leaves:
        req = list(range(nl))
    else:
        req = [int(x) for x in leaves]
        if not req:
            raise ValueError("no leaf requested: pass --leaf N (repeatable) or --all-leaves")
        if len(set(req)) != len(req):
            raise ValueError(f"--leaf repeats a leaf: {req}")
    for leaf in req:
        if not 0 <= leaf < nl:
            raise ValueError(f"--leaf {leaf} is outside {branch}'s {nl} leaf slots (0..{nl - 1})")

    plan = Plan(store, group, branch, src, dst, row, iteration, n_alloc)
    plan.ntemps, plan.nwalkers = int(sa["ntemps"]), nw

    # main group
    ml = main_layout(ra, nl, sa["ndim"])
    if "nleaves_max" in root and branch in root["nleaves_max"].attrs:
        if int(root["nleaves_max"].attrs[branch]) != nl:
            raise ValueError(
                f"main nleaves_max[{branch}]={int(root['nleaves_max'].attrs[branch])} != "
                f"sub_backend nleaves_max={nl}")
    for fam in ("chain", "inds"):
        ds = root[fam][branch]
        trailing, waxis, laxis = ml[fam]
        if tuple(ds.shape[1:]) != tuple(trailing):
            raise ValueError(
                f"{fam}/{branch}: shape {ds.shape} does not match the expected "
                f"(steps,) + {trailing}; refusing.")
        if int(ds.shape[0]) != n_alloc:
            raise ValueError(f"{fam}/{branch}: {ds.shape[0]} rows, inds/{branch} has {n_alloc}")
        plan.entries.append(Entry(f"{fam}/{branch}", waxis, laxis, ds.ndim))

    # sub-backend group: every dataset must be known (completeness guard)
    table = sub_layout(sa)
    names = [k for k in grp if isinstance(grp[k], h5py.Dataset)]
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(
            f"sub_backend/{branch}: dataset(s) {unknown} are not in this tool's layout "
            "table; refusing to guess whether they carry a (walker, leaf) entry.")
    for need in ("chain", "inds"):
        if need not in names:
            raise ValueError(f"sub_backend/{branch}/{need} is missing")
    for name in sorted(names):
        ds = grp[name]
        trailing, waxis, laxis = table[name]
        if tuple(ds.shape[1:]) != tuple(trailing):
            raise ValueError(
                f"sub_backend/{branch}/{name}: shape {ds.shape} does not match the expected "
                f"(steps,) + {trailing}; refusing.")
        if int(ds.shape[0]) < iteration:
            raise ValueError(
                f"sub_backend/{branch}/{name}: {ds.shape[0]} rows < iteration {iteration}")
        if waxis is None:
            plan.walker_free.append(name)
        else:
            plan.entries.append(Entry(f"sub_backend/{branch}/{name}", waxis, laxis, ds.ndim))

    # liveness + the source's cold-row agreement, per leaf
    m_inds, m_chain = root["inds"][branch], root["chain"][branch]
    s_inds, s_chain = grp["inds"], grp["chain"]
    mi_w, mi_l = ml["inds"][1], ml["inds"][2]
    mc_w, mc_l = ml["chain"][1], ml["chain"][2]
    for leaf in req:
        ms = np.asarray(m_inds[_sel(m_inds.ndim, row, mi_w, src, mi_l, leaf)], dtype=bool)
        md = np.asarray(m_inds[_sel(m_inds.ndim, row, mi_w, dst, mi_l, leaf)], dtype=bool)
        ss = np.asarray(s_inds[_sel(s_inds.ndim, row, 2, src, 3, leaf)], dtype=bool)
        sd = np.asarray(s_inds[_sel(s_inds.ndim, row, 2, dst, 3, leaf)], dtype=bool)
        if all_leaves and not (ms.any() or md.any() or ss.any() or sd.any()):
            plan.skipped.append(leaf)
            continue
        if not (ms.all() and ss.all()):
            raise ValueError(
                f"{branch} leaf {leaf} is not alive on SOURCE walker {src} at row {row} "
                f"(main inds {ms.astype(int).ravel().tolist()}, per-rung inds "
                f"{ss.astype(int).tolist()}): nothing to copy.")
        if not (md.all() and sd.all()):
            raise ValueError(
                f"{branch} leaf {leaf} is alive on walker {src} but not on DESTINATION "
                f"walker {dst} at row {row} (main inds {md.astype(int).ravel().tolist()}, "
                f"per-rung inds {sd.astype(int).tolist()}). The per-leaf move requires a "
                "leaf alive on every (rung, walker); this store is already inconsistent.")
        mc = np.asarray(m_chain[_sel(m_chain.ndim, row, mc_w, src, mc_l, leaf)])
        sc0 = np.asarray(s_chain[row, 0, src, leaf])
        cold = mc.reshape(-1, mc.shape[-1])
        if not all(_same(c, sc0) for c in cold):
            raise ValueError(
                f"{branch} leaf {leaf}: the SOURCE walker {src}'s main cold row and its "
                f"sub-backend rung 0 disagree at row {row} (a torn save). Resuming that "
                "row already fails check_cold_row; restore the running backup first.")
        plan.leaves.append(leaf)
    if not plan.leaves:
        raise ValueError(
            f"no {branch} leaf alive on walker {src} at row {row} (all {nl} slots dead)")
    return plan


# --------------------------------------------------------------------------
# reading, writing, verifying
# --------------------------------------------------------------------------
def describe(f, plan):
    """Before-values for the report: ``{leaf: {...}}`` plus the main totals."""
    root = f[plan.group]
    grp = root["sub_backend"][plan.branch]
    row, s, d = plan.row, plan.src, plan.dst
    out = {}
    for leaf in plan.leaves:
        v = {}
        for tag, w in (("src", s), ("dst", d)):
            v[f"{tag}_cold"] = np.asarray(grp["chain"][row, 0, w, leaf])
            for key, name in (("ll", "log_like"), ("lp", "log_prior")):
                v[f"{tag}_{key}"] = (np.asarray(grp[name][row, leaf, :, w])
                                     if name in grp else None)
            v[f"{tag}_dh"] = float(grp["d_h"][row, w, leaf]) if "d_h" in grp else float("nan")
            v[f"{tag}_hh"] = float(grp["h_h"][row, w, leaf]) if "h_h" in grp else float("nan")
        v["betas"] = np.asarray(grp["betas_all"][row, leaf])
        out[leaf] = v
    totals = {}
    for name in ("log_like", "log_prior"):
        if name in root:
            arr = np.asarray(root[name][row]).reshape(-1, plan.nwalkers)[0]
            totals[name] = (float(arr[s]), float(arr[d]))
    return out, totals


def apply_copy(path, plan):
    """Write every planned (dataset, leaf) slice in place. Returns the rows
    of the written table ``[(dataset, dst slice, src slice, elements)]``."""
    written = []
    with h5py.File(path, "r+") as f:
        root = f[plan.group]
        for leaf in plan.leaves:
            for e in plan.entries:
                ds = root[e.path]
                data = np.asarray(ds[e.src(plan, leaf)])
                ds[e.dst(plan, leaf)] = data
                written.append((e.path, _fmt_sel(e.dst(plan, leaf)),
                                _fmt_sel(e.src(plan, leaf)), int(data.size)))
        f.flush()
    return written


def verify_copy(path, plan):
    """Re-read the store: ``[]`` when, for every planned leaf, the destination
    equals the source in every copied dataset on every rung AND the
    destination's main cold row agrees with its sub-backend rung 0 (the
    check the first propose of a resume runs); else the list of problems."""
    problems = []
    with h5py.File(path, "r") as f:
        root = f[plan.group]
        for leaf in plan.leaves:
            for e in plan.entries:
                ds = root[e.path]
                if not _same(ds[e.src(plan, leaf)], ds[e.dst(plan, leaf)]):
                    problems.append(
                        f"{e.path}{_fmt_sel(e.dst(plan, leaf))} != source "
                        f"{_fmt_sel(e.src(plan, leaf))}")
            sub = root["sub_backend"][plan.branch]
            ml = main_layout(dict(root.attrs), sub.attrs["nleaves_max"], sub.attrs["ndim"])
            for fam in ("chain", "inds"):
                mds = root[fam][plan.branch]
                _, waxis, laxis = ml[fam]
                main_cold = np.asarray(mds[_sel(mds.ndim, plan.row, waxis, plan.dst, laxis, leaf)])
                sub_cold = np.asarray(sub[fam][plan.row, 0, plan.dst, leaf])
                flat = main_cold.reshape((-1,) + sub_cold.shape)
                if not all(_same(m, sub_cold) for m in flat):
                    problems.append(
                        f"leaf {leaf}: destination walker {plan.dst}'s main {fam}/{plan.branch} "
                        f"cold row disagrees with sub_backend/{plan.branch}/{fam} rung 0 "
                        "(the resume's check_cold_row would abort)")
    return problems


def resume_readable(path):
    """lisatools' own resume-readability check, or a note when lisatools is absent."""
    try:
        from lisatools.globalfit.hdfbackend import _validate_resume_readable
    except Exception as exc:  # noqa: BLE001 -- a bare env still gets the h5 checks
        return True, f"lisatools resume check skipped ({type(exc).__name__}: {exc})"
    try:
        _validate_resume_readable(path)
    except Exception as exc:  # noqa: BLE001 -- reported, the caller decides
        return False, f"lisatools _validate_resume_readable FAILED: {type(exc).__name__}: {exc}"
    return True, "lisatools _validate_resume_readable: OK"


# --------------------------------------------------------------------------
# sidecars
# --------------------------------------------------------------------------
def _age(path):
    return time.time() - os.path.getmtime(path)


def recent_writes(store, window):
    """``[(path, age s)]`` of the store and its writer sidecars modified within
    ``window`` seconds: the evidence of a run still writing. ``lsof`` on a
    login node cannot see a writer on a compute node; mtimes can."""
    side = sidecar_paths(store)
    cands = [store, side["backup"], side["backup"] + ".tmp", side["midit"], side["midit"] + ".tmp"]
    out = []
    for p in cands:
        if os.path.exists(p):
            age = _age(p)
            if age < window:
                out.append((p, age))
    return out


def eigen_report(store, branch, leaves):
    """Lines describing the eigen-table sidecar entries of the copied leaves."""
    path = os.path.splitext(store)[0] + EIGEN_SUFFIX
    if not os.path.exists(path):
        return [f"eigen   {os.path.basename(path)}: absent (nothing to do)"]
    try:
        with open(path, "rb") as fp:
            payload = pickle.load(fp)
        entries = payload["entries"]
    except Exception as exc:  # noqa: BLE001 -- the run treats it as empty too
        return [f"eigen   {os.path.basename(path)}: unreadable ({type(exc).__name__}); "
                "the run ignores it and rebuilds"]
    lines = [f"eigen   {os.path.basename(path)} (reported, NOT edited):"]
    for leaf in leaves:
        e = entries.get(f"{branch}:{int(leaf)}")
        if not isinstance(e, dict):
            lines.append(f"          {branch}:{leaf}: no entry (built at the first visit)")
            continue
        axes = np.asarray(e.get("axes"))
        if axes.ndim >= 4:
            lines.append(
                f"          {branch}:{leaf}: scope {e.get('scope')!r}, a per-(rung, walker) "
                f"stash {axes.shape} of the head's walker block: the destination's rows keep "
                "its OLD position's axes until the next refresh (stale scale, correct MH)")
        else:
            lines.append(
                f"          {branch}:{leaf}: scope {e.get('scope')!r}, one walker-independent "
                f"table {axes.shape}: nothing walker-specific to copy")
    return lines


def _twin_iteration(path, group):
    try:
        with h5py.File(path, "r") as f:
            return int(f[group].attrs.get("iteration", -1))
    except Exception as exc:  # noqa: BLE001 -- reported only
        return f"unreadable ({type(exc).__name__})"


def _free_name(base):
    if not os.path.exists(base):
        return base
    n = 1
    while os.path.exists(f"{base}.{n}"):
        n += 1
    return f"{base}.{n}"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _vec(a):
    return np.array2string(np.asarray(a), precision=6, max_line_width=150, separator=" ")


def report(plan, before, totals, label="store"):
    print(f"\n  == {label}: {plan.store}")
    print(f"     stored iteration {plan.iteration}, allocated rows {plan.n_alloc}; "
          f"the resume reads row {plan.iteration - 1}")
    if plan.row != plan.iteration - 1:
        print(f"     NOTE: --row {plan.row} is NOT the last row. The resume reads row "
              f"{plan.iteration - 1} only, so this edit does not change where it starts.")
    print(f"     {plan.branch}: walker {plan.src} -> walker {plan.dst} at row {plan.row}, "
          f"leaves {plan.leaves} ({plan.ntemps} rungs each)")
    if plan.skipped:
        print(f"     skipped (dead on both walkers): leaves {plan.skipped}")
    for leaf in plan.leaves:
        v = before[leaf]
        print(f"\n     -- leaf {leaf}")
        print(f"        cold coords (rung 0)")
        print(f"          src w{plan.src}          : {_vec(v['src_cold'])}")
        print(f"          dst w{plan.dst} before   : {_vec(v['dst_cold'])}")
        print(f"          dst w{plan.dst} after    : = src w{plan.src}")
        if v["src_ll"] is not None:
            print(f"        per-leaf lnL sub_backend/{plan.branch}/log_like"
                  f"[{plan.row}, {leaf}, :, w] (the walker's FULL lnL at this leaf's visit)")
            print(f"          {'rung':>4} {'beta':>8} {'src w' + str(plan.src):>18} "
                  f"{'dst w' + str(plan.dst) + ' before':>20} {'dst after':>18} {'dst change':>12}")
            for k in range(len(v["src_ll"])):
                print(f"          {k:>4} {v['betas'][k]:8.4f} {v['src_ll'][k]:18.3f} "
                      f"{v['dst_ll'][k]:20.3f} {v['src_ll'][k]:18.3f} "
                      f"{v['src_ll'][k] - v['dst_ll'][k]:12.3f}")
        if v["src_lp"] is not None:
            print(f"        per-leaf log_prior rung 0: src {v['src_lp'][0]:.4f}, "
                  f"dst before {v['dst_lp'][0]:.4f}")
        print(f"        d_h / h_h: src {v['src_dh']:.4f} / {v['src_hh']:.4f}, "
              f"dst before {v['dst_dh']:.4f} / {v['dst_hh']:.4f}")
    leaf0 = plan.leaves[0]
    print(f"\n     datasets written per leaf (shown for leaf {leaf0}):")
    print(f"       {'dataset':<30} {'destination':<24} {'<- source':<24}")
    for e in plan.entries:
        print(f"       {e.path:<30} {_fmt_sel(e.dst(plan, leaf0)):<24} "
              f"{_fmt_sel(e.src(plan, leaf0)):<24}")
    print(f"     NOT copied, walker-free (shared by both walkers): "
          f"{', '.join(plan.walker_free)}")
    if totals:
        parts = [f"{k}[{plan.row}] src {a:.3f} / dst {b:.3f}" for k, (a, b) in totals.items()]
        print("     NOT copied, per-walker TOTALS over every branch (the resume recomputes "
              "log_like from the rebuilt residual and zeroes log_prior): " + "; ".join(parts))


def report_sidecars(store, group, branch, leaves):
    side = sidecar_paths(store)
    print("\n  sidecars:")
    twin = side["backup"]
    if os.path.exists(twin):
        it = _twin_iteration(twin, group)
        print(f"    backup  {os.path.basename(twin)}: iteration {it}, modified "
              f"{_age(twin):.0f} s ago. Read ONLY if the store is unreadable at resume; the "
              "saver overwrites it with the edited store after the first save. --also-backup "
              "edits it too (default off).")
    else:
        print(f"    backup  {os.path.basename(twin)}: absent")
    midit = side["midit"]
    if os.path.exists(midit):
        print(f"    midit   {os.path.basename(midit)}: PRESENT "
              f"({os.path.getsize(midit) / MB:.1f} MB, "
              f"modified {_age(midit):.0f} s ago). A checkpoint written at or after the store's "
              "last row WINS over the store at resume and would discard this edit: --apply "
              "refuses unless --reset-midit moves it aside.")
    else:
        print(f"    midit   {os.path.basename(midit)}: absent (good)")
    for line in eigen_report(store, branch, leaves):
        print(f"    {line}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_row(text):
    if text is None or str(text).strip().lower() == "last":
        return "last"
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"--row expects 'last' or a row index, got {text!r}") from None


def _plan_file(path, args, row):
    try:
        with h5py.File(path, "r") as f:
            plan = plan_copy(f, args.branch, args.leaf, args.src_walker, args.dst_walker,
                             row=row, group=args.group, all_leaves=args.all_leaves, store=path)
            before, totals = describe(f, plan)
    except OSError as exc:
        raise ValueError(
            f"cannot open {path} ({exc}). If the run (or its monitor/saver) still holds it, "
            "stop it first.") from None
    return plan, before, totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="path to the run's *_testing.h5")
    ap.add_argument("--branch", required=True, help="mbh, emri or sobbh")
    ap.add_argument("--leaf", type=int, action="append", default=[],
                    help="leaf slot to copy (repeatable)")
    ap.add_argument("--all-leaves", action="store_true",
                    help="copy every leaf alive on the source walker (dead-on-both are skipped)")
    ap.add_argument("--src-walker", type=int, required=True, help="walker to copy FROM")
    ap.add_argument("--dst-walker", type=int, required=True, help="walker to copy INTO")
    ap.add_argument("--row", default="last",
                    help="stored row to edit: 'last' (default, the row a resume reads) or N")
    ap.add_argument("--group", default="global_fit")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--verify", action="store_true",
                    help="read-only: report whether the destination already equals the source")
    ap.add_argument("--no-backup", action="store_true",
                    help=f"skip the {BACKUP_SUFFIX} copy before writing")
    ap.add_argument("--also-backup", action="store_true",
                    help="apply the same edit to *_running_backup_copy.h5 (its own last row)")
    ap.add_argument("--reset-midit", action="store_true",
                    help="move *_midit_checkpoint.pkl aside (it would override the edited store)")
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
    if bool(args.leaf) == bool(args.all_leaves):
        print("REFUSING: give --leaf N (repeatable) or --all-leaves, not both / neither")
        return 2
    holders = pids_holding(store)
    if holders:
        print(f"REFUSING: {store} is held open by PID(s) {holders}; stop the run first.")
        return 2
    try:
        row = _parse_row(args.row)
        plan, before, totals = _plan_file(store, args, row)
    except ValueError as exc:
        print(f"REFUSING: {exc}")
        return 2

    if args.verify and not args.apply:
        problems = verify_copy(store, plan)
        print(f"{store}\n  verify {args.branch} leaves {plan.leaves}, walker {plan.dst} vs "
              f"walker {plan.src}, row {plan.row}:")
        for p in problems:
            print(f"    DIFFERS: {p}")
        print("  VERIFY OK: the destination equals the source on every rung and its cold rows "
              "agree" if not problems else f"  VERIFY FAILED ({len(problems)} problem(s))")
        return 0 if not problems else 1

    report(plan, before, totals)
    report_sidecars(store, args.group, args.branch, plan.leaves)

    twin = sidecar_paths(store)["backup"]
    twin_plan = None
    if args.also_backup:
        if not os.path.exists(twin):
            print(f"\nREFUSING: --also-backup but {twin} does not exist")
            return 2
        try:
            twin_plan, tb, tt = _plan_file(twin, args, row)
        except ValueError as exc:
            print(f"\nREFUSING (running backup twin): {exc}")
            return 2
        report(twin_plan, tb, tt, label="running backup twin")

    if not args.apply:
        print("\n  DRY RUN -- nothing written.")
        print(f"  to write: python {script} {cmd} --apply")
        recent = recent_writes(store, args.live_window)
        if recent:
            print("  WARNING: modified within --live-window, is the run down? "
                  + ", ".join(f"{os.path.basename(p)} ({a:.0f} s)" for p, a in recent))
        return 0

    # ---- apply gates -------------------------------------------------------
    recent = recent_writes(store, args.live_window)
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
        need = sum(os.path.getsize(p.store) for p in targets)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(store))).free
        for p in targets:
            if os.path.exists(p.store + BACKUP_SUFFIX):
                print(f"\nREFUSING: {p.store + BACKUP_SUFFIX} already exists; move or delete it "
                      "first so an earlier backup is never overwritten (or pass --no-backup).")
                return 2
        if free < need:
            print(f"\nREFUSING: {free / MB:.0f} MB free, the backup copies need "
                  f"{need / MB:.0f} MB (or pass --no-backup).")
            return 2

    ts = time.strftime("%Y%m%d-%H%M%S")
    status = 0
    if os.path.exists(midit):
        dest = _free_name(f"{midit}.stale-{ts}")
        os.rename(midit, dest)
        print(f"\n  moved aside: {midit} -> {dest}")
    for p in targets:
        if not args.no_backup:
            bak = p.store + BACKUP_SUFFIX
            print(f"\n  copying {os.path.getsize(p.store) / MB:.1f} MB -> {bak} ...")
            shutil.copy2(p.store, bak)
        print(f"  writing {p.store}:")
        written = apply_copy(p.store, p)
        print(f"    {'dataset':<30} {'destination':<24} {'<- source':<24} {'elements':>8}")
        for path, dsel, ssel, n in written:
            print(f"    {path:<30} {dsel:<24} {ssel:<24} {n:>8}")
        problems = verify_copy(p.store, p)
        for prob in problems:
            print(f"    VERIFY: {prob}")
        ok, msg = resume_readable(p.store)
        print(f"    {msg}")
        if problems or not ok:
            status = 1
            print("    FAILED -- restore from the backup before resuming.")
        else:
            print(f"    VERIFY OK: walker {p.dst} now equals walker {p.src} for {p.branch} "
                  f"leaves {p.leaves} on every rung at row {p.row}.")
    if status == 0:
        print("\n  DONE. Resume the run with its usual launch line.")
    return status


if __name__ == "__main__":
    sys.exit(main())
