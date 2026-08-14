"""Migrate a global-fit HDF5 store's VGB branch 5 -> 6 columns (chirp basis).

The 2026-08-14 ruling moves the VGB distance basis from
``[dist, phi0, cos_iota, psi, fdot_astro_ratio]`` (Mc fixed per leaf) to the
opt-in ``VGB_CHIRP_MASS_BASIS=1`` basis
``[dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio]`` (only f0/alpha/sin_delta
fixed). Resuming a 5-dim store with the flag on fails loudly (run.py ndim
guard) and points here. This script rewrites, in place after a ``.bak`` copy:

* ``<group>/chain/vgb``            (main cold-chain history)
* ``<group>/sub_backend/vgb/chain`` (the branch's tempered history)
* ``<group>/ndims``/``sub_backend/vgb`` ndim attrs (5 -> 6)

Old columns copy through unchanged into slots [0, 1, 2, 3, 5]. The new Mc
column (slot 4) is filled from the catalogue per leaf (same sorted-key order
as ``prepare_vgb_branch``) times a small MULTIPLICATIVE jitter per stored
element so walkers are not identical; the ratio column (slot 5) is
RE-INITIALIZED with the additive fresh-init rule (zero truth: multiplicative
jitter would collapse the dimension under the pure stretch move).

Usage::

    python scripts/fstat_proposal/migrate_vgb_chirp_basis.py <file.h5> \
        <catalogue_dir>   # e.g. .../mojito_cache/brickmarket/mojito_light_v1_0_0

Dead leaves (inds False) keep the NaN sentinel. Nothing else in the file is
touched.
"""

import argparse
import os
import shutil

import h5py
import numpy as np

from lisatools.globalfit.stock.erebor.vgb import (
    VGB_SAMPLED_BASIS_CHIRP,
    VGB_SAMPLED_BASIS_DIST,
    load_vgb_catalogue_file,
)

OLD_NDIM = len(VGB_SAMPLED_BASIS_DIST)  # 5
NEW_NDIM = len(VGB_SAMPLED_BASIS_CHIRP)  # 6
MC_COL = VGB_SAMPLED_BASIS_CHIRP.index("Mc")  # 4
RATIO_COL = VGB_SAMPLED_BASIS_CHIRP.index("fdot_astro_ratio")  # 5
# old -> new column map for the 5 carried-over columns
OLD_TO_NEW = [0, 1, 2, 3, RATIO_COL]

# Fresh-init conventions (mirror VGBSettings defaults + run.py seeding):
# ratio_0 = START_FACTOR * RATIO_INIT_WIDTH * FDOT_ASTRO_RATIO_MAX * randn.
START_FACTOR = float(os.environ.get("VGB_START_FACTOR", "1e-5"))
RATIO_INIT_WIDTH = float(os.environ.get("VGB_RATIO_INIT_WIDTH", "0.02"))
FDOT_ASTRO_RATIO_MAX = float(os.environ.get("GB_FDOT_ASTRO_RATIO_MAX", "5.0"))
MC_JITTER_REL = 1e-6  # multiplicative, START_FACTOR-style, per stored element


def leaf_chirp_masses(catalogue_dir: str) -> np.ndarray:
    """Per-leaf catalogue Mc, in prepare_vgb_branch's sorted-key leaf order."""
    # the loader stores (V)GB catalogue entries as whole arrays under one id;
    # prepare_vgb_branch concatenates catalogue[k][column] over sorted keys
    cat = load_vgb_catalogue_file(catalogue_dir)
    return np.concatenate([
        np.atleast_1d(np.asarray(cat[k]["ChirpMassSSBFrame"], dtype=float))
        for k in sorted(cat.keys())
    ])


def widen_chain(grp, name, mc_per_leaf, rng, inds=None):
    """Replace dataset ``grp[name]`` (..., nleaves, 5) with the 6-col version."""
    dset = grp[name]
    old = dset[:]
    assert old.shape[-1] == OLD_NDIM, (
        f"{dset.name}: last axis is {old.shape[-1]}, expected {OLD_NDIM} "
        "(already migrated?)"
    )
    nleaves = old.shape[-2]
    assert mc_per_leaf.shape[0] == nleaves, (
        f"catalogue has {mc_per_leaf.shape[0]} sources but {dset.name} "
        f"stores {nleaves} leaves"
    )
    new = np.empty(old.shape[:-1] + (NEW_NDIM,), dtype=old.dtype)
    for j_old, j_new in enumerate(OLD_TO_NEW):
        new[..., j_new] = old[..., j_old]
    # Mc from the catalogue per leaf, small multiplicative jitter so walkers
    # differ; ratio re-initialized with the additive fresh-init rule.
    new[..., MC_COL] = mc_per_leaf * (
        1.0 + MC_JITTER_REL * rng.standard_normal(old.shape[:-1])
    )
    new[..., RATIO_COL] = (
        START_FACTOR * RATIO_INIT_WIDTH * FDOT_ASTRO_RATIO_MAX
        * rng.standard_normal(old.shape[:-1])
    )
    if inds is not None:
        new[~np.asarray(inds, dtype=bool)] = np.nan  # dead-leaf sentinel
    compression = dset.compression
    compression_opts = dset.compression_opts
    del grp[name]
    grp.create_dataset(
        name,
        data=new,
        maxshape=(None,) + new.shape[1:],
        compression=compression,
        compression_opts=compression_opts,
    )
    print(f"  rewrote {grp.name}/{name}: {old.shape} -> {new.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5_path", help="global-fit HDF5 file to migrate")
    parser.add_argument(
        "catalogue_dir",
        help="mojito data dir holding catalogues/vgb_cat_mojito_lite_"
             "processed.hdf5 (e.g. .../brickmarket/mojito_light_v1_0_0)",
    )
    parser.add_argument("--seed", type=int, default=101, help="jitter RNG seed")
    args = parser.parse_args()

    mc_per_leaf = leaf_chirp_masses(args.catalogue_dir)
    print(f"catalogue: {mc_per_leaf.shape[0]} VGB leaves, "
          f"Mc in [{mc_per_leaf.min():.3f}, {mc_per_leaf.max():.3f}] Msol")

    bak = args.h5_path + ".bak"
    if os.path.exists(bak):
        raise SystemExit(f"refusing to overwrite existing backup {bak!r}")
    shutil.copy2(args.h5_path, bak)
    print(f"backup: {bak}")

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.h5_path, "r+") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        g = f[group]

        # main cold-chain history (dead-leaf mask from the stored inds)
        inds = g["inds"]["vgb"][:] if "vgb" in g.get("inds", {}) else None
        widen_chain(g["chain"], "vgb", mc_per_leaf, rng, inds=inds)
        g["ndims"].attrs["vgb"] = NEW_NDIM
        print(f"  {g.name}/ndims attrs['vgb'] = {NEW_NDIM}")
        if "key_order" in g and "vgb" in g["key_order"].attrs:
            old_ko = list(g["key_order"].attrs["vgb"])
            if len(old_ko) == OLD_NDIM:
                g["key_order"].attrs["vgb"] = np.arange(NEW_NDIM)
                print(f"  key_order['vgb']: {old_ko} -> {list(range(NEW_NDIM))}")

        # branch sub-backend (tempered history)
        sub = g["sub_backend"]["vgb"]
        sub_inds = sub["inds"][:] if "inds" in sub else None
        widen_chain(sub, "chain", mc_per_leaf, rng, inds=sub_inds)
        sub.attrs["ndim"] = NEW_NDIM
        print(f"  {sub.name} attrs['ndim'] = {NEW_NDIM}")

    print(
        "done. Old columns [dist, phi0, cos_iota, psi, ratio] -> new slots "
        "[0, 1, 2, 3, 5]; Mc (slot 4) = catalogue value * "
        f"(1 + {MC_JITTER_REL:g}*randn); ratio (slot 5) re-initialized "
        f"additively with width {START_FACTOR:g} * {RATIO_INIT_WIDTH:g} * "
        f"{FDOT_ASTRO_RATIO_MAX:g}. Resume with VGB_CHIRP_MASS_BASIS=1."
    )


if __name__ == "__main__":
    main()
