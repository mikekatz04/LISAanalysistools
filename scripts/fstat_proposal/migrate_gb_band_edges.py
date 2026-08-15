"""Migrate a global-fit HDF5 store's GB band grid to a new set of band edges.

The 2026-08-15 variable-width band work (``GB_BAND_EDGES_MODE=get_n``,
``GB_BAND_TARGET_COUNT``, ``GB_BAND_MIN_LAYERS``; see
``lisatools.globalfit.stock.erebor.gb.get_n_based_band_edges``) changes the
number of GB sub-bands. The stored per-band arrays in the branch sub-backend
(``<group>/sub_backend/gb``) are all sized ``(num_bands, ...)``, so resuming
across a band-edge change fails loudly (run.py band-grid guard) and points
here. This script rewrites, in place after a ``.bak`` copy:

* ``band_edges``      (static)            -> the NEW edges
* ``band_temps``      (step, nb, ntemps)  -> per-temperature-rung ladders
                                             INTERPOLATED in frequency from
                                             the old band centers onto the
                                             new band centers (every stored
                                             step, so the dataset stays
                                             shape-consistent)
* ``band_leaf_cap``   (step, nb)          -> reset to the ``-1`` sentinel:
                                             the progressive-cap machinery
                                             re-arms at ``leaf_cap_start``
                                             (GB_LEAF_CAP_START) on the first
                                             RJ proposal and caps RE-EARN
                                             their growth (there is no
                                             defensible per-band remap of an
                                             earned cap onto different bands)
* ``band_cap_iters``  (step, nb)          -> zeros (cap plateau counters)
* ``band_best_ll``    (step, nb)          -> ``-inf`` (cap plateau baseline)
* ``band_cold_ll``    (step, nw, nb)      -> ``-inf`` (recomputed in-move)
* ``band_swaps_proposed`` / ``band_swaps_accepted``     -> zeros (counters)
* ``band_num_proposed`` / ``band_num_accepted`` (+_rj)  -> zeros (counters)
* ``band_num_binaries`` (step, nt, nw, nb) -> zeros (recomputed from the
                                             leaves by the move's BandSorter
                                             on the next proposal)
* group attr ``num_bands``                -> the new band count

Leaves themselves (``chain`` / ``inds``) are untouched: sources are assigned
to bands by ``searchsorted(band_edges, f0)`` at propose time, which is fully
generic in the edges.

The in-move F-stat fit epoch caches (``<fit_dir>/shared/epoch_*``) key their
per-peak grids by band index and are INVALID under new edges. Nothing needs
migrating there: the loader (``check_cached_band_grid`` /
``_epoch_band_grid_stale``) detects the mismatch, refuses the stale grids
loudly, and forces a fresh-epoch refit. Pass ``--fstat-fit-dir`` to also
rename the stale ``shared`` directory aside for tidiness.

New edges come from one of:

* ``--edges-npy <file.npy>`` -- explicit edges (Hz, ascending), used verbatim;
* otherwise the get_N-based builder, with ``--tobs`` (seconds) required and
  the band knobs read from the same env vars the run uses
  (GB_BAND_EDGES_MODE / GB_BAND_TARGET_COUNT / GB_BAND_MIN_LAYERS /
  GB_SUBBAND_DIVISOR) or their CLI overrides. ``--layer-df`` defaults to the
  stored grid's median edge spacing (correct for today's one-band-per-layer
  stores; pass it explicitly for subdivided stores).

ALWAYS verify the printed new band count/edges match what the resumed run
logs ("The number of subbands is ...") -- the run.py resume guard compares
the migrated edges against the settings-derived ones and refuses any
mismatch.

Usage::

    GB_BAND_EDGES_MODE=get_n GB_BAND_TARGET_COUNT=0 \
    python scripts/fstat_proposal/migrate_gb_band_edges.py <file.h5> \
        --tobs 7889238.0 [--branch gb] [--fstat-fit-dir <run>/gb_fstat_fit]

Nothing else in the file is touched.
"""

import argparse
import os
import shutil

import h5py
import numpy as np

from lisatools.globalfit.stock.erebor.gb import get_n_based_band_edges

# per-band datasets: how each one migrates. "interp" = frequency
# interpolation (band_temps only); everything else resets to the named fill
# because it is either a counter, a derived quantity the move recomputes, or
# progressive-cap state that must re-earn (see module docstring).
RESET_FILL = {
    "band_swaps_proposed": 0,
    "band_swaps_accepted": 0,
    "band_num_proposed": 0,
    "band_num_accepted": 0,
    "band_num_proposed_rj": 0,
    "band_num_accepted_rj": 0,
    "band_num_binaries": 0,
    "band_leaf_cap": -1,       # sentinel: progressive caps re-arm + re-earn
    "band_cap_iters": 0,
    "band_best_ll": -np.inf,
    "band_cold_ll": -np.inf,
}

# which axis of each STORED dataset (leading step axis included) is the band
# axis -- explicit, never guessed from sizes (GBState.initialize_band_
# information shapes + the step axis prepended by the HDF backend):
#   band_temps            (step, nb, ntemps)        -> 1
#   band_swaps_*          (step, nb, ntemps-1)      -> 1
#   band_num_*(_rj)       (step, nb, ntemps)        -> 1
#   band_num_binaries     (step, ntemps, nw, nb)    -> 3
#   band_leaf_cap/_cap_iters/_best_ll (step, nb)    -> 1
#   band_cold_ll          (step, nw, nb)            -> 2
BAND_AXIS = {
    "band_temps": 1,
    "band_swaps_proposed": 1,
    "band_swaps_accepted": 1,
    "band_num_proposed": 1,
    "band_num_accepted": 1,
    "band_num_proposed_rj": 1,
    "band_num_accepted_rj": 1,
    "band_num_binaries": 3,
    "band_leaf_cap": 1,
    "band_cap_iters": 1,
    "band_best_ll": 1,
    "band_cold_ll": 2,
}


def band_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def remap_band_axis(old, axis, nb_new, fill=None, interp_from=None,
                    interp_to=None):
    """Return ``old`` with band axis ``axis`` resized to ``nb_new``.

    ``fill`` -> constant-filled array (counters / re-earned state).
    ``interp_from``/``interp_to`` -> np.interp along the band axis (per all
    other indices), used for ``band_temps``.
    """
    new_shape = list(old.shape)
    new_shape[axis] = nb_new
    if fill is not None:
        return np.full(new_shape, fill, dtype=old.dtype)
    moved = np.moveaxis(old, axis, -1)          # (..., nb_old)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty((flat.shape[0], nb_new), dtype=old.dtype)
    for i in range(flat.shape[0]):
        out[i] = np.interp(interp_to, interp_from, flat[i])
    out = out.reshape(moved.shape[:-1] + (nb_new,))
    return np.moveaxis(out, -1, axis)


def rewrite_dataset(grp, name, new):
    """Replace dataset ``grp[name]`` preserving compression/maxshape style."""
    dset = grp[name]
    old_shape = tuple(dset.shape)
    compression = dset.compression
    compression_opts = dset.compression_opts
    maxshape = dset.maxshape
    del grp[name]
    # growable datasets carry maxshape (None, ...); statics keep theirs fixed
    new_maxshape = None
    if maxshape is not None and maxshape[0] is None:
        new_maxshape = (None,) + new.shape[1:]
    grp.create_dataset(
        name,
        data=new,
        maxshape=new_maxshape,
        compression=compression,
        compression_opts=compression_opts,
    )
    print(f"  rewrote {grp.name}/{name}: {old_shape} -> {new.shape}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("h5_path", help="global-fit HDF5 file to migrate")
    parser.add_argument("--branch", default="gb",
                        help="banded branch to migrate (default: gb)")
    parser.add_argument("--edges-npy", default=None,
                        help=".npy file of explicit new band edges (Hz); "
                             "bypasses the get_N builder")
    parser.add_argument("--tobs", type=float, default=None,
                        help="observation time in seconds (required unless "
                             "--edges-npy)")
    parser.add_argument("--layer-df", type=float, default=None,
                        help="WDM layer_df in Hz (default: the stored "
                             "grid's median edge spacing)")
    parser.add_argument("--mode",
                        default=os.environ.get("GB_BAND_EDGES_MODE", "get_n"),
                        help="band-edge mode for the builder (get_n; "
                             "default from GB_BAND_EDGES_MODE)")
    parser.add_argument("--target-count", type=int,
                        default=int(os.environ.get("GB_BAND_TARGET_COUNT", "0")),
                        help="approximate band-count target (0 = natural "
                             "get_N widths; default from GB_BAND_TARGET_COUNT)")
    parser.add_argument("--min-layers", type=float,
                        default=float(os.environ.get("GB_BAND_MIN_LAYERS", "1.0")),
                        help="minimum band width in WDM layers (default "
                             "from GB_BAND_MIN_LAYERS)")
    parser.add_argument("--divisor", type=int,
                        default=int(os.environ.get("GB_SUBBAND_DIVISOR", "1")),
                        help="subband divisor (default from GB_SUBBAND_DIVISOR)")
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--extra-buffer", type=int, default=5)
    parser.add_argument("--fstat-fit-dir", default=None,
                        help="optional in-move F-stat fit dir; its 'shared' "
                             "epoch tree is renamed aside (the loader would "
                             "refuse the stale grids anyway)")
    args = parser.parse_args()

    with h5py.File(args.h5_path, "r") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        sub = f[group]["sub_backend"][args.branch]
        old_edges = np.asarray(sub["band_edges"][:], dtype=float)

    nb_old = len(old_edges) - 1
    print(f"stored {args.branch!r} grid: {nb_old} bands, "
          f"[{old_edges[0]:.6e}, {old_edges[-1]:.6e}] Hz")

    if args.edges_npy is not None:
        new_edges = np.asarray(np.load(args.edges_npy), dtype=float)
    else:
        if args.tobs is None:
            raise SystemExit("--tobs (seconds) is required without --edges-npy")
        if str(args.mode).lower() != "get_n":
            raise SystemExit(
                f"--mode {args.mode!r}: only 'get_n' is built here; for "
                "'uniform' the store already matches (or pass --edges-npy)."
            )
        layer_df = args.layer_df
        if layer_df is None:
            layer_df = float(np.median(np.diff(old_edges)))
            print(f"inferred layer_df = {layer_df:.6e} Hz from the stored "
                  f"grid (median edge spacing; override with --layer-df)")
        new_edges = get_n_based_band_edges(
            float(old_edges[0]),
            float(old_edges[-1]),
            float(args.tobs),
            float(layer_df),
            subband_divisor=args.divisor,
            oversample=args.oversample,
            extra_buffer=args.extra_buffer,
            target_count=args.target_count,
            min_band_layers=args.min_layers,
        )

    nb_new = len(new_edges) - 1
    if np.array_equal(new_edges, old_edges):
        raise SystemExit("new edges are identical to the stored edges; "
                         "nothing to migrate.")
    print(f"new grid: {nb_new} bands, "
          f"[{new_edges[0]:.6e}, {new_edges[-1]:.6e}] Hz")

    bak = args.h5_path + ".bak"
    if os.path.exists(bak):
        raise SystemExit(f"refusing to overwrite existing backup {bak!r}")
    shutil.copy2(args.h5_path, bak)
    print(f"backup: {bak}")

    ctr_old = band_centers(old_edges)
    ctr_new = band_centers(new_edges)

    with h5py.File(args.h5_path, "r+") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        sub = f[group]["sub_backend"][args.branch]

        # static band_edges dataset -> the new grid
        rewrite_dataset(sub, "band_edges", new_edges)

        # band_temps: interpolate each stored step's per-rung ladder in
        # frequency (step, nb, ntemps) -- band axis is 1
        bt = sub["band_temps"][:]
        assert bt.shape[1] == nb_old, (
            f"band_temps band axis {bt.shape} != stored num_bands {nb_old}"
        )
        rewrite_dataset(
            sub, "band_temps",
            remap_band_axis(bt, 1, nb_new, interp_from=ctr_old,
                            interp_to=ctr_new),
        )

        # counters / derived / re-earned state -> constant fills
        for name, fill in RESET_FILL.items():
            if name not in sub:
                print(f"  (skip {name}: not stored)")
                continue
            arr = sub[name][:]
            axis = BAND_AXIS[name]
            if arr.ndim <= axis or arr.shape[axis] != nb_old:
                raise SystemExit(
                    f"{name} has shape {arr.shape}: expected the band axis "
                    f"at {axis} with length {nb_old}; migrate manually."
                )
            rewrite_dataset(
                sub, name, remap_band_axis(arr, axis, nb_new, fill=fill)
            )

        sub.attrs["num_bands"] = nb_new
        print(f"  {sub.name} attrs['num_bands'] = {nb_new}")

    if args.fstat_fit_dir:
        shared = os.path.join(args.fstat_fit_dir, "shared")
        if os.path.isdir(shared):
            aside = f"{shared}_stale_{nb_old}bands"
            k = 0
            while os.path.exists(aside):
                k += 1
                aside = f"{shared}_stale_{nb_old}bands.{k}"
            os.rename(shared, aside)
            print(f"moved stale F-stat epochs aside: {shared} -> {aside}")
        else:
            print(f"(no {shared} directory; nothing to move aside)")

    print(
        f"done: {nb_old} -> {nb_new} bands. band_temps interpolated in "
        "frequency; swap/proposal counters and band_num_binaries zeroed; "
        "band_leaf_cap reset to -1 (progressive caps re-arm at "
        "GB_LEAF_CAP_START and RE-EARN); band_best_ll/band_cold_ll reset to "
        "-inf. Resume with the SAME band knobs used here "
        "(GB_BAND_EDGES_MODE/GB_BAND_TARGET_COUNT/GB_BAND_MIN_LAYERS/"
        "GB_SUBBAND_DIVISOR) -- the run.py guard verifies the stored edges "
        "against the settings-derived ones. The in-move F-stat fit will "
        "start a fresh epoch (stale band-indexed grids are refused loudly)."
    )


if __name__ == "__main__":
    main()
