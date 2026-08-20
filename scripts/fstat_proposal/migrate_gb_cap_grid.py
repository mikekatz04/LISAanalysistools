"""Migrate a global-fit HDF5 store onto the GB LEAF-CAP CELL grid.

The 2026-08-15 cap-grid work (``GBSettings.cap_divisor`` / ``GB_CAP_DIVISOR``)
moves the progressive per-band leaf caps onto a finer "cap cell" grid: every
GB sub-band is split into ``K`` equal pieces and the caps are enforced per
piece. Sub-band widths are set by what the likelihood engine can run
concurrently; the scale at which two GB sources actually get confused is the
posterior width, which is much narrower. NOTHING ELSE MOVES -- scheduling,
units, buffers, tempering and band shutoff all stay on the band grid.

A store written before this change (or under a different ``K``) carries a cap
grid that disagrees with the run config, and
``GBState.initialize_band_information`` refuses the resume loudly and points
here. This script rewrites, in place after a ``.bak`` copy:

* ``cap_edges``            (static, NEW)        -> ``make_cap_edges(band_edges, K)``
* ``cap_cell_leaf_cap``    (step, nc, NEW)      -> each band's stored
                                                   ``band_leaf_cap`` BROADCAST
                                                   to its K children
* ``cap_cell_iters``       (step, nc, NEW)      -> each band's stored
                                                   ``band_cap_iters`` broadcast
* ``cap_cell_best_ll``     (step, nc, NEW)      -> ``-inf``
* ``cap_cell_cold_ll``     (step, nw, nc, NEW)  -> ``-inf``
* group attr ``num_cap_cells``                  -> ``K * num_bands``

INHERIT SEMANTICS (the point of the script). Children inherit the parent
band's CURRENT cap and its min-iters counter, so a band that has already
earned its way to cap 3 hands each of its cells a cap of 3 -- the migration
never tightens an existing source's cell retroactively, and never resets a
band's progression. The lnL baselines (``*_best_ll``) DO reset to ``-inf``:
they are running maxima of a per-band statistic that no longer means the same
thing on a per-cell grid, and ``-inf`` simply re-anchors each cell at its
next iteration (the D/2 improvement gate then re-earns from there, and the
inherited ``iters`` keeps the plateau counter intact).

The per-band arrays (``band_leaf_cap`` / ``band_cap_iters`` / ``band_best_ll``
/ ``band_cold_ll``) are LEFT IN PLACE and keep being written by the run -- the
monitor and the diagnostic scripts read them, and ``band_leaf_cap`` is
maintained as the per-band MAX over its cells. Leaves (``chain`` / ``inds``)
are untouched: cap cells are derived from ``f0`` at propose time.

``K = 1`` is a no-op grid (cap cells == sub-bands); the script still writes
``cap_edges`` so an older store gains the dataset the resume guard reads.

STAGGERED GRID (``--stagger``, user design 2026-08-20 / GB_CAP_STAGGER, the
v5 grid): writes the half-cell-shifted edges instead -- no cap edge
coincides with a band edge; cell counts and every array shape are identical
to the nested grid. Inheritance still broadcasts from each cell's OWNING
band (cell ``c`` -> band ``c // K``), which is exact for the interior cells
and conservative for the one straddling cell per seam. INTENDED FOR A
REWOUND-TO-EMPTY GB BRANCH (the v5-from-rewound-v4 flow: rewind first with
``reset_recipe_stage.py --rewind-to-empty gb``, then run this): with zero
alive leaves there is no occupancy to reattribute and the migration is
exactly "replace the edge values, re-broadcast the rewound cap state". On a
store with live GB leaves it still runs, but sources within half a cell of
a band seam silently change cells -- do that knowingly or not at all.

Usage::

    python scripts/fstat_proposal/migrate_gb_cap_grid.py <file.h5> \
        --cap-divisor 8 [--stagger] [--branch gb] [--dry-run]

Nothing else in the file is touched.
"""

import argparse
import os
import shutil

import h5py
import numpy as np

from lisatools.globalfit.state import make_cap_edges

#: on-disk name -> (band-array it inherits from, fill when there is none)
#: The step axis is axis 0 in every stored dataset.
CAP_CELL_SPEC = {
    "cap_cell_leaf_cap": ("band_leaf_cap", -1),
    "cap_cell_iters": ("band_cap_iters", 0),
    "cap_cell_best_ll": (None, -np.inf),
    "cap_cell_cold_ll": (None, -np.inf),
}

#: cap-cell axis of each stored dataset (leading step axis included)
#:   cap_cell_leaf_cap / _iters / _best_ll  (step, nc)      -> 1
#:   cap_cell_cold_ll                       (step, nw, nc)  -> 2
CELL_AXIS = {
    "cap_cell_leaf_cap": 1,
    "cap_cell_iters": 1,
    "cap_cell_best_ll": 1,
    "cap_cell_cold_ll": 2,
}


def split_to_cells(band_arr, axis, k):
    """Broadcast a per-band array to its ``k`` children along ``axis``.

    Band ``b`` owns cap cells ``b*k ... b*k + k - 1`` (containment), so the
    child grid is just a repeat along the band axis. This is the INHERIT
    step: every child starts at exactly the parent's value.
    """
    return np.repeat(np.asarray(band_arr), k, axis=axis)


def migrate(path, branch="gb", cap_divisor=8, stagger=False, dry_run=False,
            backup=True):
    grp_path = f"global_fit/sub_backend/{branch}"
    k = max(1, int(cap_divisor))

    with h5py.File(path, "r") as f:
        if grp_path not in f:
            raise SystemExit(f"{path}: no {grp_path} group (nothing to migrate).")
        grp = f[grp_path]
        band_edges = np.asarray(grp["band_edges"][:], dtype=float)
        nb = len(band_edges) - 1
        nsteps = int(grp["band_leaf_cap"].shape[0])
        has_cap_edges = "cap_edges" in grp
        stored_cap_edges = (
            np.asarray(grp["cap_edges"][:], dtype=float) if has_cap_edges
            else None
        )
        band_data = {
            name: np.asarray(grp[name][:])
            for name in ("band_leaf_cap", "band_cap_iters")
            if name in grp
        }
        nw = int(grp["band_cold_ll"].shape[1]) if "band_cold_ll" in grp else None
        existing_cells = [n for n in CAP_CELL_SPEC if n in grp]
        # the LIVE row is the last WRITTEN iteration, not the last
        # allocated slot (backends pre-grow their datasets)
        it_attr = int(f["global_fit"].attrs.get("iteration", nsteps))
        live_row = max(0, min(nsteps - 1, it_attr - 1))

    cap_edges = make_cap_edges(band_edges, k, stagger=stagger)
    nc = len(cap_edges) - 1

    print(f"file:            {path}")
    print(f"branch group:    {grp_path}")
    print(f"sub-bands:       {nb}  (edges {band_edges[0]:.6e} .. "
          f"{band_edges[-1]:.6e} Hz)")
    if has_cap_edges:
        print(f"stored cap grid: {len(stored_cap_edges) - 1} cells "
              f"(divisor {(len(stored_cap_edges) - 1) / nb:g})")
    else:
        print("stored cap grid: NONE (pre-cap-grid store -> divisor 1)")
    print(f"new cap grid:    {nc} cells (divisor {k}"
          + (", STAGGERED half-cell against the band grid" if stagger else "")
          + ")")
    print(f"stored steps:    {nsteps}; walkers: {nw}")
    if existing_cells:
        print(f"existing cap-cell datasets (will be replaced): {existing_cells}")
    if "band_leaf_cap" in band_data:
        _lc = band_data["band_leaf_cap"]
        _last = _lc[live_row] if _lc.ndim > 1 else _lc
        vals, cnts = np.unique(_last, return_counts=True)
        print(f"band_leaf_cap @ live row {live_row} (iteration attr "
              f"{it_attr}): "
              + ", ".join(f"cap {int(v)}: {int(c)} bands"
                          for v, c in zip(vals, cnts)))
        print("  -> after migration each of those bands hands its cap to "
              f"all {k} of its cap cells (inherit, never tighten).")
    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    if backup:
        bak = path + ".bak"
        if os.path.exists(bak):
            raise SystemExit(
                f"{bak} already exists -- move it aside first (refusing to "
                "overwrite an existing backup)."
            )
        shutil.copy2(path, bak)
        print(f"\nbackup written: {bak}")

    with h5py.File(path, "a") as f:
        grp = f[grp_path]

        if "cap_edges" in grp:
            del grp["cap_edges"]
        grp.create_dataset("cap_edges", data=cap_edges, dtype=float)

        for name, (src, fill) in CAP_CELL_SPEC.items():
            axis = CELL_AXIS[name]
            if src is not None and src in grp:
                new = split_to_cells(np.asarray(grp[src][:]), axis, k)
            else:
                if name == "cap_cell_cold_ll":
                    if nw is None:
                        continue
                    shape = (nsteps, nw, nc)
                else:
                    shape = (nsteps, nc)
                new = np.full(shape, fill, dtype=float)
            if name in grp:
                del grp[name]
            grp.create_dataset(
                name, data=new, maxshape=(None,) + new.shape[1:],
                dtype=new.dtype,
            )
            print(f"  wrote {name} {new.shape} {new.dtype}")

        grp.attrs["num_cap_cells"] = nc
        print(f"  attr num_cap_cells = {nc}")

    print(
        "\nDone. Resume with GB_CAP_DIVISOR / GBSettings.cap_divisor = "
        f"{k}"
        + (" and GB_CAP_STAGGER=1 / GBSettings.cap_stagger" if stagger
           else "")
        + "; the resume guard compares the stored cap_edges against the "
        "config-built ones and refuses any mismatch."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="global-fit HDF5 store to migrate in place")
    ap.add_argument("--branch", default="gb",
                    help="branch sub-backend group name (default: gb)")
    ap.add_argument("--cap-divisor", type=int, default=8,
                    help="cap cells per sub-band K (default: 8)")
    ap.add_argument("--stagger", action="store_true",
                    help="write the half-cell STAGGERED grid "
                         "(GB_CAP_STAGGER / the v5 grid); intended for a "
                         "rewound-to-empty GB branch")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and exit")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .bak copy (not recommended)")
    args = ap.parse_args()
    migrate(
        args.path, branch=args.branch, cap_divisor=args.cap_divisor,
        stagger=args.stagger, dry_run=args.dry_run,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
