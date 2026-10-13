"""Re-rung a run store: change the number of temperatures of a branch IN PLACE.

Why this exists (user request 2026-10-02): the 6mo production store carries
MBHB and EMRI at 2 rungs each (``MBH_NTEMPS=2`` / ``EMRI_NTEMPS=2`` in the
launcher) and SOBHB at 8. A branch's rung count is frozen into the store --
every ``sub_backend/<branch>`` dataset with a temperature axis has that axis
at a FIXED maxshape, the group's ``ntemps`` attr names it, and on a resume
the STORED ladder wins over ``{BRANCH}_NTEMPS`` (``recipe.resume_ladder_wins``,
with a warning). So "give the MBHBs more temperatures" is a store migration,
not a knob: this script rewrites those datasets at the new rung count and
updates the attr; the next launch then builds the branch's moves at the new
count from the store's own ladder.

What it does, per selected branch
---------------------------------
* GROWING (``--set mbh=12`` on a 2-rung branch): every new rung's STATE
  (coords, inds, per-rung log_like / log_prior) is a copy of an existing
  rung -- ``--fill cycle`` (default: rung j <- rung j mod old, i.e. the old
  ladder repeated up the new one, "just duplicate the 2 temperatures up the
  ladder for now"), ``--fill hottest`` (rung j <- old hottest) or
  ``--fill coldest`` (rung j <- cold rung 0). lnL is a property of the state,
  so the copied log_like rows are exact. The per-rung proposal / swap
  COUNTERS of the new rungs start at zero. The LADDER (``betas_all`` per
  leaf, or the flat ``betas``) is rewritten on every stored row with the
  branch's own construction rule -- ``1 / 1.2**k`` for the per-leaf-ladder
  branches (mbh / emri / sobbh, stock/erebor/*.py, user ruling 2026-08-27),
  eryn's ``make_ladder(ndim * 10, Tmax=inf)`` for psd / galfor -- or with
  ``--ratio R`` / ``--betas b0,b1,...`` when given.
* SHRINKING (``--set sobbh=4`` on an 8-rung branch): the coldest ``new``
  rungs are kept and the hottest dropped -- states, counters and the stored
  (adapted) ladder alike, so the surviving rungs keep the betas they were
  actually sampling at (``--ladder fresh`` rebuilds the ladder instead).
* The number of walkers, leaves, dims and the cold rung 0 never change: the
  main (cold-chain) group is untouched, and the cold-row cross-check the
  resume performs (``_validate_resume_readable``) still holds.

What it refuses
---------------
* gb / vgb: their ladders are per BAND (``band_temps``, ``band_num_*`` per
  rung) and their moves bind those tables -- a different migration.
* a dataset in a selected group that matches no entry of the layout table
  below (a new array this script does not know): refused, never guessed.
* a store another process holds open (``lsof``), an unreadable store, a
  branch that is not in the store.

Usage (dry run first -- nothing is written without ``--apply``)::

    python scripts/fstat_proposal/rerung_store_branches.py \
        /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5 \
        --set mbh=12 --set emri=12
    ... same command with --apply [--reset-backup --reset-midit]

``--apply`` first copies the store to ``<base>.pre_rerung-<timestamp>.h5``
(``--no-backup`` skips it), rewrites in place, then re-opens the file and
verifies every selected group against the layout table at the new count and
runs ``lisatools``' own resume-readability check. Afterwards launch with
``MBH_NTEMPS=12 EMRI_NTEMPS=12`` (the stored ladder would win anyway; matching
knobs keep the resume warning quiet).

Sidecars: ``--reset-backup`` moves the saver's ``<base>_running_backup_copy.h5``
aside (it still has the OLD rung count; the saver rewrites it on its first
save, but ``promote_backup_if_store_unreadable`` could otherwise hand a
future resume the old layout). ``--reset-midit`` moves the mid-iteration
checkpoint pickle aside (its gate rejects a ladder mismatch by itself).
"""

import argparse
import os
import shutil
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compact_gf_store import pids_holding, sidecar_paths  # noqa: E402

MB = 1024.0 ** 2

#: Branches whose ladder is per band: a different migration.
BANDED_REFUSAL = ("gb", "vgb")


# --------------------------------------------------------------------------
# layout table
# --------------------------------------------------------------------------
def layout_for(attrs):
    """``{dataset: (family, trailing shape, T axis | None)}`` for one
    ``sub_backend/<branch>`` group, from its own attrs.

    ``family`` is ``state`` (copied along the temperature axis), ``counter``
    (kept rungs copied, new rungs zero), ``swaps`` (the (T-1)-long pair
    counters, same rule), ``ladder`` (rebuilt) or ``none`` (no temperature
    axis). Trailing shapes exclude the step axis. ``T axis`` counts the
    step axis as 0. Two ladder families share the space and are told apart
    by which ladder dataset the group carries, never by shape -- for mbh
    ``nleaves_max == nwalkers == 4``.
    """
    nt, nw = int(attrs["ntemps"]), int(attrs["nwalkers"])
    nl, nd = int(attrs["nleaves_max"]), int(attrs["ndim"])
    per_leaf = "betas_all" in attrs.get("_datasets", ())
    out = {
        "chain": ("state", (nt, nw, nl, nd), 1),
        "inds": ("state", (nt, nw, nl), 1),
        "d_h": ("none", (nw, nl), None),
        "h_h": ("none", (nw, nl), None),
    }
    if per_leaf:
        out.update({
            "log_like": ("state", (nl, nt, nw), 2),
            "log_prior": ("state", (nl, nt, nw), 2),
            "betas_all": ("ladder", (nl, nt), 2),
            "in_model_accepted": ("counter", (nl, nt), 2),
            "in_model_proposed": ("counter", (nl, nt), 2),
            "rj_accepted": ("counter", (nl, nt), 2),
            "rj_proposed": ("counter", (nl, nt), 2),
            "swaps_accepted": ("swaps", (nl, max(nt - 1, 0)), 2),
            "swaps_proposed": ("swaps", (nl, max(nt - 1, 0)), 2),
        })
    else:
        out.update({
            "log_like": ("state", (nt, nw), 1),
            "log_prior": ("state", (nt, nw), 1),
            "betas": ("ladder", (nt,), 1),
            "in_model_accepted": ("counter", (nt,), 1),
            "in_model_proposed": ("counter", (nt,), 1),
            "rj_accepted": ("counter", (nt,), 1),
            "rj_proposed": ("counter", (nt,), 1),
            "swaps_accepted": ("swaps", (max(nt - 1, 0),), 1),
            "swaps_proposed": ("swaps", (max(nt - 1, 0),), 1),
        })
    return out


def fill_map(new_nt, old_nt, mode):
    """Source rung for every new rung (growing) or the kept rungs (shrinking)."""
    new_nt, old_nt = int(new_nt), int(old_nt)
    if new_nt <= old_nt:
        return list(range(new_nt))
    if mode == "cycle":
        return [j % old_nt for j in range(new_nt)]
    if mode == "hottest":
        return [min(j, old_nt - 1) for j in range(new_nt)]
    if mode == "coldest":
        return [j if j < old_nt else 0 for j in range(new_nt)]
    raise ValueError(f"unknown --fill {mode!r} (cycle, hottest, coldest)")


def new_ladder(new_nt, per_leaf, ndim, ratio=None, betas=None):
    """The ladder a FRESH run would build at ``new_nt`` rungs for this family."""
    if betas is not None:
        b = np.asarray(betas, dtype=float).reshape(-1)
        if b.size != int(new_nt):
            raise ValueError(f"--betas has {b.size} values, --set asks for {new_nt} rungs")
        if b[0] != 1.0 or np.any(np.diff(b) >= 0) or np.any(b <= 0):
            raise ValueError("--betas must start at 1.0 and decrease strictly, all > 0")
        return b
    if ratio is not None:
        if not ratio > 1.0:
            raise ValueError(f"--ratio {ratio} must be > 1")
        return 1.0 / float(ratio) ** np.arange(int(new_nt))
    if per_leaf:
        # The stock/erebor/{mbh,emri,sobbh}.py default UNTIL 2026-10-13: pure
        # geometric 1/1.2^k. Since then a fresh run builds
        # make_ladder(ndim, ntemps) (common.default_source_ladder); this tool
        # keeps the old default (tests pin it) -- pass --betas for the new one,
        # or re-rung and relaunch with {BRANCH}_LADDER_RESET_ON_RESUME=1.
        return 1.0 / 1.2 ** np.arange(int(new_nt))
    from eryn.moves.tempering import make_ladder

    # stock/erebor/noise.py: make_ladder(ndim * 10, Tmax=inf, ntemps)
    return np.asarray(make_ladder(int(ndim) * 10, Tmax=np.inf, ntemps=int(new_nt)), dtype=float)


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
class DatasetPlan:
    def __init__(self, name, family, axis, old_shape, new_shape, old_bytes):
        self.name, self.family, self.axis = name, family, axis
        self.old_shape, self.new_shape, self.old_bytes = old_shape, new_shape, old_bytes


class BranchPlan:
    def __init__(self, branch, old_nt, new_nt, per_leaf, fill, ladder_mode, ladder):
        self.branch, self.old_nt, self.new_nt = branch, int(old_nt), int(new_nt)
        self.per_leaf, self.fill, self.ladder_mode = per_leaf, fill, ladder_mode
        self.ladder = ladder                     # None = keep the stored (shrink, 'keep')
        self.datasets = []
        self.src = fill_map(new_nt, old_nt, fill)

    @property
    def unchanged(self):
        return self.old_nt == self.new_nt


def _group_datasets(grp):
    return [k for k in grp if isinstance(grp[k], h5py.Dataset)]


def plan_branch(root, branch, new_nt, fill, ladder_mode, ratio, betas):
    """Validate one branch against the layout table and describe the rewrite."""
    if branch in BANDED_REFUSAL:
        raise ValueError(
            f"{branch!r}: its ladder is per band (band_temps, band_num_* per rung) "
            "and its moves bind those tables; re-rungging it is a different migration.")
    if "sub_backend" not in root or branch not in root["sub_backend"]:
        raise ValueError(f"{branch!r}: no sub_backend/{branch} group in the store")
    grp = root["sub_backend"][branch]
    names = _group_datasets(grp)
    attrs = dict(grp.attrs)
    for need in ("ntemps", "nwalkers", "nleaves_max", "ndim"):
        if need not in attrs:
            raise ValueError(f"{branch!r}: group attr {need!r} missing")
    attrs["_datasets"] = tuple(names)
    per_leaf = "betas_all" in names
    if not per_leaf and "betas" not in names:
        raise ValueError(f"{branch!r}: neither betas_all nor betas in the group -- not a tempered branch")
    table = layout_for(attrs)
    old_nt = int(attrs["ntemps"])
    new_nt = int(new_nt)
    if new_nt < 1:
        raise ValueError(f"{branch!r}: --set {new_nt} must be >= 1")
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(
            f"{branch!r}: dataset(s) {unknown} are not in this script's layout table; "
            "refusing to guess which axis is the temperature axis.")
    if new_nt < old_nt and ladder_mode == "keep":
        ladder = None
    else:
        ladder = new_ladder(new_nt, per_leaf, attrs["ndim"], ratio=ratio, betas=betas)
    bp = BranchPlan(branch, old_nt, new_nt, per_leaf, fill, ladder_mode, ladder)
    for name in names:
        ds = grp[name]
        family, trailing, axis = table[name]
        if tuple(ds.shape[1:]) != tuple(trailing):
            raise ValueError(
                f"{branch}/{name}: shape {ds.shape} does not match the expected "
                f"(steps,) + {trailing} for ntemps={old_nt}; refusing.")
        if family == "none":
            continue
        new_shape = list(ds.shape)
        new_shape[axis] = (max(new_nt - 1, 0) if family == "swaps" else new_nt)
        bp.datasets.append(DatasetPlan(name, family, axis, tuple(ds.shape), tuple(new_shape),
                                       int(ds.id.get_storage_size())))
    return bp


def plan(path, targets, group="global_fit", fill="cycle", ladder_mode="keep",
         ratio=None, betas=None):
    """One :class:`BranchPlan` per ``targets`` entry (``{branch: new_ntemps}``).

    ``ladder_mode`` matters only when SHRINKING: ``keep`` (default) truncates
    the stored ladder, ``fresh`` rebuilds it; growing always rebuilds.
    """
    with h5py.File(path, "r") as f:
        root = f[group]
        return [plan_branch(root, b, n, fill, ladder_mode, ratio, betas)
                for b, n in targets.items()]


# --------------------------------------------------------------------------
# the rewrite
# --------------------------------------------------------------------------
def _transform(arr, dplan, bplan):
    """The new array for one dataset (whole, all stored rows at once)."""
    ax = dplan.axis
    old_nt, new_nt, src = bplan.old_nt, bplan.new_nt, bplan.src
    if dplan.family == "state":
        return np.take(arr, src, axis=ax)
    if dplan.family == "counter":
        out_shape = list(arr.shape)
        out_shape[ax] = new_nt
        out = np.zeros(out_shape, dtype=arr.dtype)
        keep = min(old_nt, new_nt)
        sl_o = [slice(None)] * arr.ndim
        sl_o[ax] = slice(0, keep)
        out[tuple(sl_o)] = arr[tuple(sl_o)]
        return out
    if dplan.family == "swaps":
        n_new = max(new_nt - 1, 0)
        out_shape = list(arr.shape)
        out_shape[ax] = n_new
        out = np.zeros(out_shape, dtype=arr.dtype)
        keep = min(max(old_nt - 1, 0), n_new)
        if keep:
            sl_o = [slice(None)] * arr.ndim
            sl_o[ax] = slice(0, keep)
            out[tuple(sl_o)] = arr[tuple(sl_o)]
        return out
    if dplan.family == "ladder":
        if bplan.ladder is None:                 # shrink + keep: truncate the stored ladder
            sl = [slice(None)] * arr.ndim
            sl[ax] = slice(0, new_nt)
            return np.ascontiguousarray(arr[tuple(sl)])
        out_shape = list(arr.shape)
        out_shape[ax] = new_nt
        out = np.empty(out_shape, dtype=arr.dtype)
        shape = [1] * arr.ndim
        shape[ax] = new_nt
        out[...] = np.asarray(bplan.ladder, dtype=arr.dtype).reshape(shape)
        return out
    raise ValueError(dplan.family)


def _maxshape_like(src_ds, new_shape):
    """The backend's maxshape convention at ``new_shape``: the ITERATION axis 0
    is UNLIMITED (``None``), every other axis fixed at its extent. Any other
    ``None`` the source carries is kept too."""
    src_max = tuple(src_ds.maxshape) if src_ds.maxshape is not None else tuple(src_ds.shape)
    return tuple(None if (i == 0 or m is None) else int(s)
                 for i, (m, s) in enumerate(zip(src_max, new_shape)))


def _recreate(grp, name, src_ds, new_shape, data, maxshape=None):
    """Replace ``grp[name]`` with a dataset of ``new_shape`` carrying ``data``.

    A temperature axis has a FIXED maxshape equal to its extent, so the
    dataset cannot be resized: it is deleted and created again with the
    source's dtype, compression, shuffle and fill value. Chunks follow the
    source with the temperature axis's chunk clamped to the new extent
    (HDF5 refuses a chunk longer than a fixed dimension).

    ⚠ maxshape (2026-10-03 production failure): the iteration axis 0 is
    UNLIMITED in every backend dataset (``maxshape[0] is None``) -- that is
    what lets eryn ``grow`` the store each run. The first version of this
    tool wrote ``maxshape = new_shape`` (every axis fixed), so the re-rung
    6mo store refused its first ``grow`` ("dimension cannot exceed the
    existing maximal size (new: 2085 max: 2084)") and the relaunch aborted
    before writing anything. Axis 0 now stays ``None`` (``_maxshape_like``);
    ``repair_unlimited`` mends a store written by the old version.
    """
    chunks = None
    if src_ds.chunks is not None:
        chunks = tuple(min(int(c), int(s)) for c, s in zip(src_ds.chunks, new_shape))
    kwargs = dict(shape=tuple(int(s) for s in new_shape), dtype=src_ds.dtype,
                  compression=src_ds.compression, compression_opts=src_ds.compression_opts,
                  shuffle=src_ds.shuffle, fletcher32=src_ds.fletcher32,
                  fillvalue=src_ds.fillvalue,
                  maxshape=_maxshape_like(src_ds, new_shape) if maxshape is None else maxshape)
    if chunks is not None:
        kwargs["chunks"] = chunks
    attrs = dict(src_ds.attrs)
    del grp[name]
    ds = grp.create_dataset(name, **kwargs)
    for k, v in attrs.items():
        ds.attrs[k] = v
    ds[...] = data
    return ds


def apply_plan(path, plans, group="global_fit", log=print):
    """Rewrite every planned dataset in place and update the groups' ``ntemps``."""
    with h5py.File(path, "r+") as f:
        root = f[group]
        for bp in plans:
            if bp.unchanged:
                continue
            grp = root["sub_backend"][bp.branch]
            for dp in bp.datasets:
                src_ds = grp[dp.name]
                arr = src_ds[...]
                data = _transform(arr, dp, bp)
                assert tuple(data.shape) == tuple(dp.new_shape), (dp.name, data.shape, dp.new_shape)
                _recreate(grp, dp.name, src_ds, dp.new_shape, data)
                log(f"    {bp.branch}/{dp.name:<20} {dp.old_shape} -> {dp.new_shape}")
            grp.attrs["ntemps"] = np.int64(bp.new_nt)
        f.flush()


def fixed_axis0(path, branches, group="global_fit"):
    """``["branch/dataset", ...]`` whose ITERATION axis is NOT unlimited.

    Every backend dataset has ``maxshape[0] is None``; a finite axis 0 means
    the store cannot grow and the next run aborts at its first ``grow``.
    """
    out = []
    with h5py.File(path, "r") as f:
        root = f[group]
        for b in branches:
            grp = root["sub_backend"][b]
            for name in sorted(grp.keys()):
                ds = grp[name]
                if isinstance(ds, h5py.Dataset) and ds.maxshape[0] is not None:
                    out.append(f"{b}/{name}")
    return out


def repair_unlimited(path, branches, group="global_fit", log=print):
    """Re-create every dataset of ``branches`` whose iteration axis is finite
    with ``maxshape = (None, *shape[1:])``, data / attrs / chunks / filters
    kept. Returns the repaired names. Mends a store the 2026-10-02 version of
    this tool re-rung (it fixed axis 0 at the allocated row count)."""
    done = []
    with h5py.File(path, "r+") as f:
        root = f[group]
        for b in branches:
            grp = root["sub_backend"][b]
            for name in sorted(grp.keys()):
                ds = grp[name]
                if not isinstance(ds, h5py.Dataset) or ds.maxshape[0] is None:
                    continue
                shape = tuple(int(s) for s in ds.shape)
                _recreate(grp, name, ds, shape, ds[...], maxshape=(None,) + shape[1:])
                done.append(f"{b}/{name}")
                log(f"    {b}/{name:<20} iteration axis {shape[0]} (fixed) -> unlimited; "
                    f"shape {shape} kept")
        f.flush()
    return done


def verify(path, targets, group="global_fit"):
    """Re-plan at the new counts: every selected branch must read as unchanged,
    and every dataset of the selected branches must still be growable."""
    again = plan(path, targets, group=group)
    bad = [bp.branch for bp in again if not bp.unchanged]
    if bad:
        raise RuntimeError(f"after the rewrite these branches still do not read at the "
                           f"requested count: {bad}")
    stuck = fixed_axis0(path, sorted(targets), group=group)
    if stuck:
        raise RuntimeError(f"these datasets have a FINITE iteration axis (the store could "
                           f"not grow; the next run would abort at its first grow): {stuck}")
    try:
        from lisatools.globalfit.hdfbackend import _validate_resume_readable
    except Exception as exc:  # noqa: BLE001 -- lisatools absent in a bare env
        return f"lisatools resume check skipped ({type(exc).__name__}: {exc})"
    _validate_resume_readable(path)
    return "lisatools _validate_resume_readable: OK"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _free_name(base):
    if not os.path.exists(base):
        return base
    n = 1
    while os.path.exists(f"{base}.{n}"):
        n += 1
    return f"{base}.{n}"


def _parse_targets(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects BRANCH=N, got {it!r}")
        b, n = it.split("=", 1)
        out[b.strip()] = int(n)
    if not out:
        raise ValueError("nothing to do: pass at least one --set BRANCH=N")
    return out


def _report(path, plans):
    with h5py.File(path, "r") as f:
        it = int(f["global_fit"].attrs.get("iteration", 0)) if "global_fit" in f else 0
    print(f"{path}")
    print(f"  stored iteration: {it}   file on disk: {os.path.getsize(path) / MB:.1f} MB")
    for bp in plans:
        print(f"\n  == {bp.branch}: {bp.old_nt} -> {bp.new_nt} rung(s)"
              + ("   (UNCHANGED: nothing to do)" if bp.unchanged else ""))
        if bp.unchanged:
            continue
        if bp.new_nt > bp.old_nt:
            print(f"     fill = {bp.fill}: new rung j <- old rung {bp.src}")
        else:
            print(f"     shrink: keeping the coldest {bp.new_nt} rung(s) {bp.src}")
        if bp.ladder is None:
            print("     ladder: the stored (adapted) ladder, truncated")
        else:
            print(f"     ladder ({'per leaf, betas_all' if bp.per_leaf else 'flat, betas'}, "
                  f"every stored row): {np.array2string(np.asarray(bp.ladder), precision=4)}")
            print(f"     max T = {1.0 / float(np.min(bp.ladder)):.3g}")
        print(f"     {'dataset':<22} {'family':<8} {'old shape':<28} {'new shape':<28} {'old MB':>7}")
        for dp in bp.datasets:
            print(f"     {dp.name:<22} {dp.family:<8} {str(dp.old_shape):<28} "
                  f"{str(dp.new_shape):<28} {dp.old_bytes / MB:7.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="path to the run's *_testing.h5")
    ap.add_argument("--set", action="append", metavar="BRANCH=N", default=[],
                    help="new rung count for a branch (repeatable): --set mbh=12 --set emri=12 --set sobbh=4")
    ap.add_argument("--fill", choices=("cycle", "hottest", "coldest"), default="cycle",
                    help="which existing rung a NEW rung copies its state from (growing only)")
    ap.add_argument("--ladder", choices=("keep", "fresh"), default="keep",
                    help="shrinking: 'keep' (default) truncates the stored ladder, 'fresh' rebuilds it; "
                         "growing always rebuilds")
    ap.add_argument("--ratio", type=float, default=None,
                    help="geometric ladder 1/ratio^k instead of the branch's construction rule")
    ap.add_argument("--betas", default=None,
                    help="explicit comma-separated ladder (must have N values, start at 1, decrease)")
    ap.add_argument("--group", default="global_fit")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the <base>.pre_rerung-<ts>.h5 copy before rewriting")
    ap.add_argument("--reset-backup", action="store_true",
                    help="move <base>_running_backup_copy.h5 aside (it has the OLD rung count)")
    ap.add_argument("--reset-midit", action="store_true",
                    help="move <base>_midit_checkpoint.pkl aside")
    ap.add_argument("--repair-unlimited", metavar="BRANCHES", default=None,
                    help="comma-separated branches: re-create every dataset whose ITERATION "
                         "axis is finite with an unlimited one (a store re-rung by the "
                         "2026-10-02 version cannot grow); dry run without --apply")
    args = ap.parse_args(argv)

    if not os.path.exists(args.store):
        print(f"no such store: {args.store}")
        return 2
    holders = pids_holding(args.store)
    if holders:
        print(f"REFUSING: {args.store} is held open by PID(s) {holders}; stop the run first.")
        return 2

    if args.repair_unlimited:
        branches = [b.strip() for b in args.repair_unlimited.split(",") if b.strip()]
        try:
            stuck = fixed_axis0(args.store, branches, group=args.group)
        except KeyError as exc:
            print(f"REFUSING: no such branch group in the store: {exc}")
            return 2
        print(f"{args.store}")
        if not stuck:
            print(f"  every dataset of {branches} already has an unlimited iteration axis. "
                  "nothing to do.")
            return 0
        print(f"  {len(stuck)} dataset(s) with a FINITE iteration axis (the store cannot grow):")
        for s in stuck:
            print(f"     {s}")
        if not args.apply:
            print("\n  DRY RUN -- nothing written. Re-run with --apply to repair in place.")
            return 0
        ts = _timestamp()
        base, _ = os.path.splitext(args.store)
        if not args.no_backup:
            bak = _free_name(f"{base}.pre_repair-{ts}.h5")
            print(f"\n  copying the store to {bak} ...")
            shutil.copyfile(args.store, bak)
        print("  repairing:")
        repair_unlimited(args.store, branches, group=args.group)
        left = fixed_axis0(args.store, branches, group=args.group)
        if left:
            print(f"  FAILED: still finite: {left}")
            return 1
        try:
            from lisatools.globalfit.hdfbackend import _validate_resume_readable
            _validate_resume_readable(args.store)
            print("  lisatools _validate_resume_readable: OK")
        except ImportError:
            pass
        print(f"  DONE. file on disk: {os.path.getsize(args.store) / MB:.1f} MB")
        return 0

    targets = _parse_targets(args.set)
    betas = None if args.betas is None else [float(x) for x in args.betas.split(",")]
    ladder_mode = args.ladder
    try:
        plans = plan(args.store, targets, group=args.group, fill=args.fill,
                     ladder_mode=ladder_mode, ratio=args.ratio, betas=betas)
    except Exception as exc:  # noqa: BLE001 -- reported, not traced
        print(f"REFUSING: {exc}")
        return 2
    _report(args.store, plans)
    todo = [bp for bp in plans if not bp.unchanged]
    if not todo:
        print("\n  nothing to do.")
        return 0
    knobs = " ".join(f"{bp.branch.upper()}_NTEMPS={bp.new_nt}" for bp in todo)
    if not args.apply:
        print("\n  DRY RUN -- nothing written. Re-run with --apply to rewrite in place.")
        print(f"  afterwards launch with: {knobs}")
        return 0

    ts = _timestamp()
    base, _ = os.path.splitext(args.store)
    if not args.no_backup:
        bak = _free_name(f"{base}.pre_rerung-{ts}.h5")
        print(f"\n  copying the store to {bak} ...")
        shutil.copyfile(args.store, bak)
    print("  rewriting:")
    apply_plan(args.store, todo, group=args.group)
    side = sidecar_paths(args.store)
    for key, flag in (("backup", args.reset_backup), ("midit", args.reset_midit)):
        if flag and os.path.exists(side[key]):
            dest = _free_name(f"{side[key]}.stale-{ts}")
            os.rename(side[key], dest)
            print(f"  moved aside: {side[key]} -> {dest}")
    print("  verifying ...")
    msg = verify(args.store, targets, group=args.group)
    print(f"  {msg}")
    print(f"  DONE. file on disk: {os.path.getsize(args.store) / MB:.1f} MB")
    print(f"  launch with: {knobs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
