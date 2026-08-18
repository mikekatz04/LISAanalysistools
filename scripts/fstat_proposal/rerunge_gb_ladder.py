"""Change a branch's band-temperature RUNG COUNT in place, on an emptied store.

Why this exists: resume derives a branch's rung count from the STORED
``band_temps`` shape, not from ``GB_NTEMPS`` -- so a store baked at 2 rungs
stays 2-rung through every resubmit, and the knob is silently ignored. That
is how both confined probes ran a degenerate ``[1.0, 1e-4]`` ladder for days
while their scripts said 24.

``fix_vgb_band_temps.py`` does this for the ``vgb`` branch but is hardcoded to
it. This one is branch-agnostic and, more importantly, SHAPE-DRIVEN: it finds
the rung axis by inspecting each dataset rather than carrying a hand-written
list, because GB puts the rung axis in two different places --

    band_temps          (nit, nbands, nrungs)      -> LAST axis
    band_num_accepted   (nit, nbands, nrungs)      -> LAST axis
    band_swaps_accepted (nit, nbands, nrungs - 1)  -> LAST axis, PAIRS
    band_num_binaries   (nit, nrungs, nwalkers, nbands)  -> axis 1
    chain               (nit, nrungs, nwalkers, nleaves, ndim)  -> axis 1
    inds                (nit, nrungs, nwalkers, nleaves)        -> axis 1

-- and a list would go stale the moment a dataset is added.

**The store must be rewound to ZERO live leaves for this branch first**
(``reset_recipe_stage.py --rewind-to-empty``). That is not a safety ritual, it
is what makes the operation well defined: with leaves present, going from 2 to
24 rungs would require deciding which rung each existing source belongs to,
and there is no correct answer. With none, only SHAPES matter and the chain
content is irrelevant -- so the datasets are recreated empty and HDF5 fills
them lazily, which is also why a 24-rung GB chain does not cost 6 GB on disk.

Datasets are recreated at the CURRENT ``iteration`` row count rather than
their allocated length, since everything past the rewind point is about to be
truncated by the next ``grow()`` anyway.

The ladder written into ``band_temps`` is ``1/1.2**i`` with the last rung
clobbered to 1e-4 -- verified against the production 24-rung v3 store, and the
same formula ``fix_vgb_band_temps.py`` uses.

Usage (dry run first; nothing is written without ``--apply``)::

    python scripts/fstat_proposal/reset_recipe_stage.py STORE/base_testing.h5 \
        gb_search --rewind-to-empty gb --apply
    python scripts/fstat_proposal/rerunge_gb_ladder.py STORE/base_testing.h5 gb 24
    python scripts/fstat_proposal/rerunge_gb_ladder.py STORE/base_testing.h5 gb 24 --apply

Stop the job first: h5py blocks rather than erroring when it cannot take the
lock.
"""

from __future__ import annotations

import argparse
import shutil
import sys

import h5py
import numpy as np


def ladder_for(k: int) -> np.ndarray:
    """The production beta ladder: 1/1.2**i, coldest rung pinned to 1e-4."""
    out = 1.0 / 1.2 ** np.arange(k, dtype=float)
    if k > 1:
        out[-1] = 1e-4
    return out


def rung_axis(shape, nr: int):
    """Which axis holds the rungs, and whether it is a PAIRS axis.

    Returns ``(axis, is_pairs)`` or ``None``. Axis 0 is the iteration axis and
    is never a rung axis. The last axis is checked first because the band_*
    counters put rungs there; ``nr - 1`` identifies the swap-pair arrays.
    """
    if len(shape) < 2:
        return None
    if shape[-1] == nr:
        return len(shape) - 1, False
    if nr > 1 and shape[-1] == nr - 1:
        return len(shape) - 1, True
    if shape[1] == nr:
        return 1, False
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="path to the run's *_testing.h5")
    ap.add_argument("branch", help="branch name, e.g. gb")
    ap.add_argument("rungs", type=int, help="target rung count, e.g. 24")
    ap.add_argument("--group", default="global_fit")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this only reports")
    args = ap.parse_args(argv)
    k = int(args.rungs)
    if k < 1:
        ap.error("rungs must be >= 1")

    if args.apply:
        bak = f"{args.store}.bak_rerunge_{args.branch}"
        shutil.copy2(args.store, bak)
        print(f"backup: {bak}")

    with h5py.File(args.store, "a" if args.apply else "r") as f:
        g = f[args.group]
        sub = g.get("sub_backend", {})
        if args.branch not in sub:
            ap.error(f"no sub_backend/{args.branch}; store has "
                     f"{list(sub)}")
        br = sub[args.branch]
        if "band_temps" not in br:
            ap.error(f"branch {args.branch!r} has no band_temps -- it does "
                     "not carry a per-band ladder.")

        nit = int(g.attrs["iteration"])
        nr = int(br["band_temps"].shape[-1])
        nw = int(br.attrs.get("nwalkers", -1))
        nb = int(br.attrs.get("num_bands", -1))
        print(f"{args.store}\n  branch={args.branch} iteration={nit} "
              f"stored rungs={nr} -> target {k}  (nwalkers={nw} nbands={nb})")

        # The shape probe is only unambiguous while the rung count differs
        # from the other axis lengths it could be confused with.
        if nr in (nw, nb):
            ap.error(f"stored rung count {nr} collides with nwalkers={nw} / "
                     f"num_bands={nb}; the shape probe cannot identify the "
                     "rung axis unambiguously. Migrate by hand.")
        if nr == k:
            print("  already at the target rung count; nothing to do.")
            return 0

        # Well-definedness gate: see the module docstring.
        if f"{args.branch}" in g.get("inds", {}):
            live = int(np.asarray(g["inds"][args.branch][nit - 1]).sum())
            if live:
                ap.error(
                    f"branch {args.branch!r} still has {live} live leaves at "
                    f"the resume row ({nit - 1}). Rewind to zero first:\n"
                    f"  python scripts/fstat_proposal/reset_recipe_stage.py "
                    f"{args.store} gb_search --rewind-to-empty {args.branch} "
                    "--apply\n"
                    "With leaves present there is no correct rung to assign "
                    "each existing source to.")

        todo = []
        for name in sorted(br):
            d = br[name]
            if not isinstance(d, h5py.Dataset):
                continue
            hit = rung_axis(d.shape, nr)
            if hit is None:
                continue
            ax, pairs = hit
            new = list(d.shape)
            new[0] = nit                      # drop the to-be-truncated rows
            new[ax] = max(k - 1, 0) if pairs else k
            todo.append((name, tuple(d.shape), tuple(new), ax, pairs,
                         d.dtype, d.compression, d.compression_opts))

        print(f"  {len(todo)} rung-dimensioned dataset(s):")
        for name, old, new, ax, pairs, *_ in todo:
            print(f"    {name:24s} {str(old):30s} -> {str(new):30s} "
                  f"(axis {ax}{', pairs' if pairs else ''})")

        if not args.apply:
            print("\n  DRY RUN -- re-run with --apply to write.")
            return 0

        lad = ladder_for(k)
        for name, _old, new, ax, pairs, dt, comp, copts in todo:
            del br[name]
            ds = br.create_dataset(
                name, shape=new, dtype=dt,
                maxshape=(None,) + tuple(new[1:]),
                compression=comp, compression_opts=copts,
                chunks=True,
            )
            # Everything but the ladder itself is a counter or a coordinate
            # slot: leave it at the HDF5 fill value (0 / False). Unwritten
            # chunks cost nothing on disk, which is what keeps a 24-rung GB
            # chain from materialising as gigabytes.
            if name == "band_temps":
                shp = [1] * len(new)
                shp[ax] = k
                ds[...] = np.broadcast_to(lad.reshape(shp), new)
        br.attrs["ntemps"] = k
        print(f"\n  WROTE: {len(todo)} dataset(s) recreated at {k} rungs; "
              f"ntemps attr -> {k}")
        print(f"  ladder: {np.array2string(lad, precision=5, threshold=8)}")
        print("  resubmit; the run derives its rung count from this shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
