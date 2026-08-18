"""Graft a finished ``noise_search`` state from one run's store into another's.

Purpose (2026-08-18, v3 -> v4): v4 changes nothing on the noise side -- the
config diff against v3 is entirely GB/sig-het/F-stat knobs -- so refitting the
PSD and the galactic foreground from scratch would burn ~1.5 h reproducing a
result we already have. But v4 DOES need ``noise_vgb_search`` re-run, because
the VGB beta ladder moved to eryn's ``make_ladder`` and a resumed store's
stored ladder WINS over the configured one (``stock/erebor/vgb.py``, and the
reconciliation in ``state.py``). So the target is exactly: v3's noise state,
v4's ladders and grids, ``noise_vgb_search`` onward re-run.

WHY GRAFT FORWARD RATHER THAN REWIND v3
---------------------------------------
The other direction -- copy v3's store, rewind it, re-open the stage -- makes
the DESTINATION carry v3's shapes: v3's VGB ladder, v3's GB band/cap grids,
v3's rung counts. Every one of those then needs its own migration step, and
that is precisely how the three failed band-grid migrations went (see
``migrate_gb_band_edges.py`` and the probe plan). Here the destination store is
authored by v4's own setup code, so every shape, grid and ladder is correct BY
CONSTRUCTION and this script only moves *fitted numbers* into it.

PROCEDURE
---------
1. Submit v4 normally against a fresh ``STORE_DIR``. Let it reach
   ``noise_search`` and save at least one iteration (watch for the first
   ``[SAVE]``), then ``scancel``. That single iteration is throwaway -- it
   exists only so every dataset is allocated with >= 1 row.
2. Run this script (dry run first).
3. Resubmit v4. It restores from row 0 -- the grafted noise state -- finds
   ``noise_search`` already complete, and starts ``noise_vgb_search`` on the
   new ladder.

WHAT MOVES, AND WHY EACH PIECE
------------------------------
Copied from source row ``r`` into destination row 0:

* ``log_like`` / ``log_prior`` / ``betas`` / ``samplers_running`` -- the whole
  row. The stored likelihood is only valid if the destination's model state at
  row 0 matches the source's at row ``r``, which is what makes the next item
  mandatory rather than optional.
* ``chain/<b>`` and ``inds/<b>`` for **psd, galfor AND vgb**. VGB is grafted
  even though it never moved during ``noise_search``, and that is deliberate:
  if we kept v4's own freshly-drawn VGB start instead, the grafted ``log_like``
  would describe a different model than the one on disk. Grafting all three
  non-GB branches makes row 0 a coherent state with no init-matching
  assumption to defend.
* ``sub_backend/psd`` and ``sub_backend/galfor`` per-iteration arrays -- these
  branches' own adapted ladders and counters.

Deliberately NOT copied:

* ``sub_backend/vgb`` -- this holds ``band_temps``, i.e. THE LADDER, and
  keeping the destination's is the entire point of the exercise. The script
  prints the destination ladder at the end so you can see ``make_ladder`` took.
* anything ``gb`` -- ``chain/gb``/``inds/gb`` (empty at this boundary in both
  stores, and gated on below) and ``sub_backend/gb`` (v4's band grid, cap-cell
  grid and rung count).
* ``accepted`` / ``swaps_accepted`` / ``rj_accepted`` -- cumulative diagnostic
  counters, not sampler state. They restart at zero, which is what you want for
  a run whose acceptance statistics you are about to read.
* the on-disk F-stat epoch cache under ``<store>/gb_fstat_fit/`` -- it is keyed
  to the GB grid and is fit against the residual, which the re-run
  ``noise_vgb_search`` is about to change. v4 refits it.

FINDING THE BOUNDARY ROW
------------------------
``global_fit/recipe/<stage>`` records only ``status`` and ``order num`` -- no
iteration -- so the handover row is not written down anywhere and has to be
recovered from the data. It is recoverable exactly because ``noise_search``'s
move list is ``[psd_pe, galfor_pe]`` only: the VGB branch is frozen for the
whole stage and starts moving on the first ``noise_vgb_search`` iteration. So
the last row on which ``chain/vgb`` still equals row 0 is the last
``noise_search`` row. ``--src-row`` overrides if you would rather pick by hand.

USAGE::

    # inventory either store first if anything looks off
    python scripts/fstat_proposal/graft_noise_state.py --list V3/gf_prod_3mo_testing.h5

    # dry run: reports the boundary, every gate, and every dataset it would touch
    python scripts/fstat_proposal/graft_noise_state.py \\
        V3_STORE/gf_prod_3mo_testing.h5  V4_STORE/gf_prod_3mo_v4_testing.h5
    # then, if it all reads right
    python scripts/fstat_proposal/graft_noise_state.py \\
        V3_STORE/gf_prod_3mo_testing.h5  V4_STORE/gf_prod_3mo_v4_testing.h5 --apply

Stop both jobs first: h5py blocks rather than erroring when it cannot take the
lock.
"""

from __future__ import annotations

import argparse
import shutil
import sys

import h5py
import numpy as np

#: Branches whose coordinates make up "the noise state". VGB is here because
#: the grafted ``log_like`` has to describe the state actually on disk -- see
#: the module docstring.
DEFAULT_BRANCHES = ("psd", "galfor", "vgb")

#: Whole-row datasets in the main group (everything except chain/inds).
ROW_DATASETS = ("log_like", "log_prior", "betas", "samplers_running")


# ---------------------------------------------------------------- inventory --


def inventory(path, group="global_fit"):
    """Print what a store contains, without changing it."""
    with h5py.File(path, "r") as f:
        g = f[group]
        it = int(g.attrs["iteration"])
        print(f"{path}")
        print(f"  iteration = {it}   branches = {list(g['chain'])}")
        for k in ("ntemps", "nwalkers", "nsamplers"):
            if k in g.attrs:
                print(f"  {k} = {g.attrs[k]}")
        print("  main datasets:")
        for name in ROW_DATASETS:
            if name in g:
                print(f"    {name:20s} {g[name].shape}")
        for b in sorted(g["chain"]):
            live = "?"
            if it >= 1:
                live = int(np.asarray(g["inds"][b][it - 1]).sum())
            print(f"    chain/{b:14s} {g['chain'][b].shape}   "
                  f"live leaves @ row {it - 1}: {live}")
        if "recipe" in g:
            print("  recipe:")
            steps = sorted(((int(g["recipe"][k].attrs["order num"]), k)
                            for k in g["recipe"]), key=lambda t: t[0])
            for order, name in steps:
                print(f"    {order}. {name:20s} "
                      f"done={bool(g['recipe'][name].attrs['status'])}")
        if "sub_backend" in g:
            print("  sub_backend:")
            for b in sorted(g["sub_backend"]):
                sb = g["sub_backend"][b]
                bt = sb.get("band_temps")
                extra = ""
                if bt is not None:
                    extra = f"   band_temps {bt.shape}"
                    if it >= 1 and bt.shape[0] >= it:
                        lad = np.asarray(bt[it - 1])
                        lad = lad.reshape(-1, lad.shape[-1])[0]
                        extra += ("  ladder=" + np.array2string(
                            lad, precision=4, threshold=10))
                print(f"    {b:16s}{extra}")


# ------------------------------------------------------------- the boundary --


def find_boundary(g, branch="vgb"):
    """Last row of ``noise_search``, found as the last row on which BRANCH is
    still frozen at its row-0 value.

    Returns ``(src_row, first_moved_row)``, or ``(None, None)`` if the branch
    never moves (i.e. the source never got past ``noise_search``, in which case
    its final row IS the handover state and the caller should just use it).
    """
    d = g["chain"][branch]
    nit = int(g.attrs["iteration"])
    ref = np.asarray(d[0])
    for i in range(1, nit):
        if not np.array_equal(np.asarray(d[i]), ref):
            return i - 1, i
    return None, None


# ------------------------------------------------------------------- copying --


def _copyable(src_grp, dst_grp, name):
    """``(ok, reason)`` for copying one row of dataset ``name`` between groups."""
    if name not in src_grp:
        return False, "absent in source"
    if name not in dst_grp:
        return False, "absent in destination"
    s, d = src_grp[name], dst_grp[name]
    if s.shape[1:] != d.shape[1:]:
        return False, f"shape {s.shape[1:]} != {d.shape[1:]}"
    return True, ""


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="source store (v3) -- read only, never written")
    ap.add_argument("dst", nargs="?", help="destination store (fresh v4)")
    ap.add_argument("--group", default="global_fit")
    ap.add_argument("--list", action="store_true",
                    help="just print an inventory of SRC (and DST if given)")
    ap.add_argument("--src-row", type=int, default=None,
                    help="source row to graft; default is the auto-detected "
                         "last noise_search row")
    ap.add_argument("--branches", default=",".join(DEFAULT_BRANCHES),
                    help=f"comma list (default: {','.join(DEFAULT_BRANCHES)}). "
                         "Dropping vgb breaks the log_like coherence gate -- "
                         "read the docstring before you do it.")
    ap.add_argument("--completed", default="noise_search",
                    help="recipe step to mark done in DST (default: "
                         "noise_search); '' to leave the recipe alone")
    ap.add_argument("--freeze-branch", default="vgb",
                    help="branch used to detect the stage boundary")
    ap.add_argument("--allow-gb-leaves", action="store_true",
                    help="proceed even if a store has live GB leaves at the "
                         "graft row (it should have none at this boundary)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this only reports")
    args = ap.parse_args(argv)

    if args.list:
        inventory(args.src, args.group)
        if args.dst:
            print()
            inventory(args.dst, args.group)
        return 0
    if not args.dst:
        ap.error("DST is required unless --list is given")

    branches = [b.strip() for b in args.branches.split(",") if b.strip()]

    with h5py.File(args.src, "r") as fs, \
            h5py.File(args.dst, "a" if args.apply else "r") as fd:
        gs, gd = fs[args.group], fd[args.group]
        it_s, it_d = int(gs.attrs["iteration"]), int(gd.attrs["iteration"])
        print(f"SRC {args.src}\n    iteration = {it_s}")
        print(f"DST {args.dst}\n    iteration = {it_d}")

        # ---- structural gates -------------------------------------------
        if it_d < 1:
            ap.error(
                f"destination has iteration={it_d}: its datasets have no rows "
                "yet, so there is nowhere to graft into. Let the v4 job run "
                "far enough to save one iteration (first [SAVE] line), then "
                "scancel and re-run this.")
        for k in ("ntemps", "nwalkers", "nsamplers"):
            if k in gs.attrs and k in gd.attrs and gs.attrs[k] != gd.attrs[k]:
                ap.error(f"{k} differs: src {gs.attrs[k]} vs dst "
                         f"{gd.attrs[k]}. The stores are not compatible; "
                         "row shapes would not line up.")
        missing = [b for b in branches if b not in gs["chain"] or b not in gd["chain"]]
        if missing:
            ap.error(f"branch(es) {missing} not present in both stores "
                     f"(src has {list(gs['chain'])}, dst has {list(gd['chain'])})")

        # ---- resolve the source row --------------------------------------
        if args.src_row is not None:
            row = int(args.src_row)
            if not 0 <= row < it_s:
                ap.error(f"--src-row must be in 0..{it_s - 1}")
            print(f"\n  source row {row} (given explicitly)")
        else:
            row, moved = find_boundary(gs, args.freeze_branch)
            if row is None:
                row = it_s - 1
                print(f"\n  branch {args.freeze_branch!r} never moves in SRC -- "
                      f"it never left noise_search. Using its final row {row}.")
            else:
                print(f"\n  boundary: {args.freeze_branch!r} is frozen through "
                      f"row {row} and first moves at row {moved}\n"
                      f"  -> source row {row} is the last noise_search "
                      "iteration")

        # ---- GB-emptiness gate -------------------------------------------
        for tag, grp, r in (("SRC", gs, row), ("DST", gd, 0)):
            if "gb" in grp["inds"]:
                live = int(np.asarray(grp["inds"]["gb"][r]).sum())
                print(f"  {tag} live GB leaves at row {r}: {live}")
                if live and not args.allow_gb_leaves:
                    ap.error(
                        f"{tag} has {live} live GB leaves at row {r}. This "
                        "graft assumes GB is empty on both sides -- otherwise "
                        "the grafted log_like counts GB power that the "
                        "destination's (differently-gridded) GB branch does "
                        "not carry. --allow-gb-leaves to override.")

        # ---- plan ---------------------------------------------------------
        # Allocated row count on each side. Used below to tell per-iteration
        # sub-backend arrays from statics.
        alloc_s, alloc_d = gs["log_like"].shape[0], gd["log_like"].shape[0]
        plan = []          # (label, src_dataset, dst_dataset)
        skipped = []
        for name in ROW_DATASETS:
            ok, why = _copyable(gs, gd, name)
            (plan.append((name, gs[name], gd[name])) if ok
             else skipped.append((name, why)))
        for b in branches:
            for grp_name in ("chain", "inds"):
                ok, why = _copyable(gs[grp_name], gd[grp_name], b)
                lbl = f"{grp_name}/{b}"
                (plan.append((lbl, gs[grp_name][b], gd[grp_name][b])) if ok
                 else skipped.append((lbl, why)))

        # sub_backend: per-iteration arrays only, and never for vgb/gb.
        sub_branches = [b for b in branches if b not in ("vgb", "gb")]
        for b in sub_branches:
            if "sub_backend" not in gs or b not in gs["sub_backend"]:
                skipped.append((f"sub_backend/{b}", "absent in source"))
                continue
            if "sub_backend" not in gd or b not in gd["sub_backend"]:
                skipped.append((f"sub_backend/{b}", "absent in destination"))
                continue
            sbs, sbd = gs["sub_backend"][b], gd["sub_backend"][b]
            for name in sorted(sbs):
                if not isinstance(sbs[name], h5py.Dataset):
                    continue
                if name not in sbd:
                    skipped.append((f"sub_backend/{b}/{name}",
                                    "absent in destination"))
                    continue
                # STATICS MUST NOT BE COPIED, and "has a first axis" does not
                # distinguish them: a static like ``band_edges`` is a plain 1-D
                # array, so a naive row copy would write element `row` of it
                # over element 0 and silently corrupt the grid. The exact test
                # is the allocated LENGTH: GFHDFBackend.grow drives every
                # sub-backend's grow with the same ngrow off the same parent
                # ``iteration`` attr, so a per-iteration array -- and only a
                # per-iteration array -- has the same first-axis length as the
                # main group's own row-indexed datasets.
                if (sbs[name].shape[0] != alloc_s
                        or sbd[name].shape[0] != alloc_d):
                    skipped.append((
                        f"sub_backend/{b}/{name}",
                        f"static (rows {sbs[name].shape[0]}/{sbd[name].shape[0]}"
                        f" != allocated {alloc_s}/{alloc_d})"))
                    continue
                ok, why = _copyable(sbs, sbd, name)
                lbl = f"sub_backend/{b}/{name}"
                (plan.append((lbl, sbs[name], sbd[name])) if ok
                 else skipped.append((lbl, why)))

        print(f"\n  {len(plan)} dataset(s) to graft (src row {row} -> dst row 0):")
        for lbl, s, _d in plan:
            print(f"    {lbl:34s} {str(s.shape[1:]):28s}")
        if skipped:
            print(f"\n  {len(skipped)} skipped:")
            for lbl, why in skipped:
                print(f"    {lbl:34s} {why}")

        # NOT-copied, stated explicitly so it is a decision on the record
        # rather than an omission noticed later.
        print("\n  intentionally NOT touched in DST: sub_backend/vgb (the "
              "make_ladder ladder), anything gb, the cumulative accepted/"
              "swaps_accepted counters, and <store>/gb_fstat_fit/.")

        if args.completed:
            if "recipe" not in gd:
                ap.error(f"destination has no {args.group}/recipe group")
            if args.completed not in gd["recipe"]:
                ap.error(f"no recipe step {args.completed!r} in DST; it has "
                         f"{list(gd['recipe'])}")

        if not args.apply:
            print(f"\n  DRY RUN -- would graft the above, set DST "
                  f"iteration=1"
                  + (f", mark {args.completed!r} complete." if args.completed
                     else ".")
                  + "\n  re-run with --apply to write.")
            return 0

        # ---- write --------------------------------------------------------
        fd.flush()
        bak = f"{args.dst}.bak_graft"
        shutil.copy2(args.dst, bak)
        print(f"\n  backup: {bak}")

        for lbl, s, d in plan:
            d[0] = s[row]
        gd.attrs["iteration"] = 1
        if args.completed:
            gd["recipe"][args.completed].attrs["status"] = True
            # Everything at or after the grafted stage must be OPEN, or the
            # resume would skip straight past the stage we are re-running.
            order_of = {k: int(gd["recipe"][k].attrs["order num"])
                        for k in gd["recipe"]}
            after = order_of[args.completed]
            for k, o in order_of.items():
                if o > after:
                    gd["recipe"][k].attrs["status"] = False

        print(f"\n  WROTE: {len(plan)} dataset row(s); iteration = 1"
              + (f"; {args.completed!r} marked complete, later stages open."
                 if args.completed else "."))

        # Show the ladder that survived -- the whole reason for grafting
        # forward instead of rewinding.
        if "sub_backend" in gd and "vgb" in gd["sub_backend"]:
            bt = gd["sub_backend"]["vgb"].get("band_temps")
            if bt is not None and bt.shape[0] >= 1:
                lad = np.asarray(bt[0]).reshape(-1, bt.shape[-1])[0]
                print(f"  DST vgb ladder ({bt.shape[-1]} rungs, untouched): "
                      + np.array2string(lad, precision=5, threshold=12))
        print("  resubmit v4; it resumes from row 0 into noise_vgb_search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
