"""Re-open a completed recipe stage, and optionally rewind the chain to it.

Why this exists: a staged run records per-stage completion in the store as
``global_fit/recipe/<name>/status``, and :meth:`GFHDFBackend.add_recipe` reads
those flags back on resume (``hdfbackend.py``), while ``Recipe`` advances past
every step whose status is already True (``recipe.py``). So resubmitting a
finished-stage run does NOT re-run that stage -- it moves on to the next one.

Re-opening the stage alone changes only WHICH MOVES RUN. The chain stays where
it is, so a stage re-opened after it found sources restarts on top of those
sources. ``--rewind-to-empty gb`` (or an explicit ``--iteration``) is the other
half: it moves the backend's sample counter back to the last iteration at which
that branch had ZERO leaves, i.e. the state the previous stage handed over.

Why moving one integer is sufficient. There is exactly ONE sample counter,
``global_fit.attrs["iteration"]``:

* ``Backend.get_last_sample`` resumes from row ``iteration - 1``;
* ``HDFBackend.grow`` resizes every main dataset to ``iteration + ngrow``, so
  the discarded rows are truncated on the next run rather than left dangling;
* every ``ModuleSubBackend`` (psd / galfor / gb / vgb) reads that SAME parent
  attr in its own ``grow`` / ``save_step`` -- the sub-backends have no counter
  of their own.

So one edit rewinds the main chain and all four sub-states coherently. And
because ``GBState.static_names`` is only ``("band_edges", "cap_edges")``,
everything else the GB search ratchets -- ``band_leaf_cap``,
``cap_cell_leaf_cap``, ``cap_cell_iters``, ``band_temps``, the counters -- is a
per-iteration dataset and rewinds with it. The leaf caps go back to where they
were before the search, not forward from where it left them.

What this deliberately KEEPS: the fitted noise and VGB state at the handover
iteration, the band grid, the cap-cell grid, and the on-disk F-stat epoch cache
under ``gb_fstat_fit/`` (which is not in the h5 at all). That is the expensive
part of a run and there is no reason to pay for it twice.

Two things this does NOT do, both handled by ``compact_gf_store.py``:

* It does not shrink the FILE. Moving the counter leaves the discarded rows
  allocated, and neither ``resize`` nor ``h5repack`` reclaims them, because
  ``sub_backend/gb/chain`` chunks the STEP axis 32 rows at a time -- the chunk
  holding the live rows also holds discarded ones, so it is partially live and
  cannot be freed. ``compact_gf_store.py`` rebuilds the store with only the
  live rows written, which is the only way the space comes back.
* It does not touch the sidecars, and after ``--rewind-to-empty`` all three are
  stale: ``*_running_backup_copy.h5`` sits at a LATER iteration (a truncation
  restore would silently undo the rewind), ``*_midit_checkpoint.pkl`` holds
  pre-rewind mid-iteration state, and the ``gb_fstat_fit/`` cache kept above is
  right for a stage RE-OPEN but wrong here -- its grids were fitted against a
  live residual that the removal of every GB leaf has just invalidated. See
  ``compact_gf_store.py --reset-backup/--reset-midit/--reset-fstat``.

What it does NOT change: array SHAPES. In particular a store written with a
2-rung GB ladder has ``band_temps`` shaped ``(nrows, nbands, 2)`` at every row,
so rewinding cannot restore a 24-rung ladder -- resume derives the rung count
from the stored shape and logs
``build_gb_moves: using the STORED n-rung gb ladder over the configured m``.
Run ``fix_vgb_band_temps.py <store.h5> 24`` for that, separately.

Usage (dry run first -- nothing is written without ``--apply``)::

    # re-open the stage AND rewind to where gb had no leaves
    python scripts/fstat_proposal/reset_recipe_stage.py STORE/base_testing.h5 \
        gb_search --rewind-to-empty gb
    python scripts/fstat_proposal/reset_recipe_stage.py STORE/base_testing.h5 \
        gb_search --rewind-to-empty gb --apply

    # re-open only, keeping the current sources
    python scripts/fstat_proposal/reset_recipe_stage.py STORE/base_testing.h5 gb_search

Stop the job before running this: an open writer holds the file, and h5py
blocks rather than erroring when it cannot take the lock.
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np


def _leaf_counts(grp, branch, n_rows):
    """Live leaves per stored row for ``branch`` (row-wise: the array is big)."""
    d = grp["inds"][branch]
    return np.array([int(d[i].sum()) for i in range(n_rows)], dtype=np.int64)


def _first_populated(counts):
    nz = np.where(counts > 0)[0]
    return int(nz.min()) if nz.size else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="path to the run's *_testing.h5")
    ap.add_argument("stage", help="recipe step name to re-open (e.g. gb_search)")
    ap.add_argument("--group", default="global_fit",
                    help="top-level group name (default: global_fit)")
    ap.add_argument("--rewind-to-empty", metavar="BRANCH", default=None,
                    help="also rewind the sample counter to the last row on "
                         "which BRANCH had zero live leaves (e.g. gb)")
    ap.add_argument("--iteration", type=int, default=None,
                    help="also rewind the sample counter to this explicit "
                         "value (rows 0..N-1 are kept); mutually exclusive "
                         "with --rewind-to-empty")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this only reports")
    args = ap.parse_args(argv)

    if args.rewind_to_empty and args.iteration is not None:
        ap.error("--rewind-to-empty and --iteration are mutually exclusive")

    with h5py.File(args.store, "a" if args.apply else "r") as f:
        if args.group not in f or "recipe" not in f[args.group]:
            ap.error(f"{args.store} has no {args.group}/recipe group -- is "
                     "this a staged global-fit store?")
        g = f[args.group]
        grp = g["recipe"]
        steps = sorted(((int(grp[k].attrs["order num"]), k) for k in grp),
                       key=lambda t: t[0])
        names = [k for _, k in steps]
        if args.stage not in names:
            ap.error(f"no recipe step {args.stage!r}; store has {names}")

        it_now = int(g.attrs["iteration"])
        print(f"{args.store}\n  iteration = {it_now}  (rows 0..{it_now - 1})")
        print("  recipe:")
        target = names.index(args.stage)
        changed = []
        for i, (order, name) in enumerate(steps):
            was = bool(grp[name].attrs["status"])
            now = False if (i >= target and was) else was
            flag = " -> False  <-- RE-OPEN" if now != was else ""
            print(f"    {order}. {name:20s} done={was!s:5s}{flag}")
            if now != was:
                changed.append(name)

        # ---- resolve the rewind target ----------------------------------
        it_new = None
        if args.iteration is not None:
            it_new = int(args.iteration)
            if not 1 <= it_new <= it_now:
                ap.error(f"--iteration must be in 1..{it_now}, got {it_new}")
        elif args.rewind_to_empty:
            branch = args.rewind_to_empty
            if branch not in g["inds"]:
                ap.error(f"no branch {branch!r}; store has {list(g['inds'])}")
            counts = _leaf_counts(g, branch, it_now)
            first = _first_populated(counts)
            if first is None:
                print(f"\n  branch {branch!r} never had a leaf -- nothing to "
                      "rewind past.")
            elif first == 0:
                ap.error(f"branch {branch!r} already had leaves at row 0; "
                         "there is no empty handover point to rewind to. "
                         "Use --iteration explicitly if you know better.")
            else:
                it_new = first          # keep rows 0..first-1
                print(f"\n  branch {branch!r}: first populated row = {first} "
                      f"({int(counts[first])} leaves); last empty row = "
                      f"{first - 1}. Final row holds {int(counts[-1])}.")

        if it_new is not None and it_new != it_now:
            print(f"  iteration {it_now} -> {it_new}"
                  f"  (discards {it_now - it_new} stored rows; the next "
                  "grow() truncates them)")
        elif it_new is not None:
            print(f"  iteration already {it_now} -- no rewind needed")
            it_new = None

        if not changed and it_new is None:
            print("\n  nothing to do.")
            return 0

        if not args.apply:
            todo = []
            if changed:
                todo.append("re-open " + ", ".join(changed))
            if it_new is not None:
                todo.append(f"set iteration={it_new}")
            print(f"\n  DRY RUN -- would {'; '.join(todo)}.")
            print("  re-run with --apply to write.")
            return 0

        for name in changed:
            grp[name].attrs["status"] = False
        if it_new is not None:
            g.attrs["iteration"] = it_new

        print("\n  WROTE:")
        if changed:
            print(f"    re-opened {', '.join(changed)}")
        if it_new is not None:
            print(f"    iteration = {it_new}")
        print("  resubmit the job.")

        # A ladder mismatch survives any rewind (shapes are not per-row) and
        # is the one thing that silently invalidates a re-run, so say it here
        # rather than leave it to the run log.
        if "sub_backend" in g and "gb" in g["sub_backend"]:
            bt = g["sub_backend"]["gb"].get("band_temps")
            if bt is not None and bt.ndim == 3:
                print(f"  NOTE: stored GB ladder is {bt.shape[-1]}-rung "
                      f"({bt.shape[1]} bands). Resume takes the STORED rung "
                      "count over GB_NTEMPS. To change it, run "
                      "rerunge_gb_ladder.py AFTER this rewind (it requires "
                      "zero live leaves). fix_vgb_band_temps.py is hardcoded "
                      "to the vgb branch and will NOT touch gb.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
