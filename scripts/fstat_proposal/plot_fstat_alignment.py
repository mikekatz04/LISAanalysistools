"""F-stat / search alignment diagnostic: recovered leaves vs catalogue truth.

Lightweight, offline plotter -- consumes SAVED files, no GPU stack rebuild:

  * the search HDF5 backend  (``*_testing.h5``)  -> recovered cold-chain leaves
  * the F-stat comb ``*_comb.npz``  (optional)   -> the F(f0) proposal curve
  * the mojito wdwd catalogue ``.hdf5``          -> the injected truth

The stock GB distance basis samples ``[dist, f0(mHz), Mc, phi0, cos_iota,
psi, alpha, sin_delta, fdot_astro_ratio]``, so recovered ``dist/f0/Mc/ratio``
are read straight from the chain -- only ``dist -> amplitude`` needs a
conversion (``gb_amp_from_dist``) to compare with the catalogue amplitude.

Shows, for the sampled band:
  A. F(f0) comb  + injected f0 (stems, height ~ SNR proxy) + recovered f0.
  B. amplitude vs f0: recovered A (from dist) over injected A  -- the key
     check that the F-stat A_max -> distance center lands on the true loudness.
  C. chirp mass Mc vs f0: recovered over injected Mc_eff.
  D. fdot_astro_ratio recovered (interacting DWDs sit near r ~ -2).

Usage (laptop or cluster; needs the three files locally + gbgpu for the
amplitude conversion)::

    SEARCH_H5=./out_fa_fstat_v2gb_no_fg_test_2_testing.h5 \
    FSTAT_COMB_NPZ=./fa_grids/<band>_comb.npz \
    MOJITO_DATA_PATH=/scratch-jpl/335-lisa/mlkatz/cd1L_data \
    python scripts/fstat_proposal/plot_fstat_alignment.py

Env knobs:
    SEARCH_H5        path to the search backend (required)
    FSTAT_COMB_NPZ   path to *_comb.npz (optional; skips panel A curve if absent)
    MOJITO_DATA_PATH mojito data root (for the catalogue); or CATALOGUE_H5 direct
    CATALOGUE_H5     explicit catalogue path (overrides MOJITO_DATA_PATH)
    F0_LO_MHZ/F0_HI_MHZ  band window for the catalogue (default: recovered
                     f0 range padded by 5%)
    ITER             chain iteration index to read (default -1, the last)
    OUT_PNG          output figure path (default ./fstat_alignment.png)
"""

import os

import numpy as np


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _recovered_leaves(h5_path, it):
    """Cold-chain (temp 0) alive leaves at iteration ``it`` -> (n_alive, ndim)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        chain = f["global_fit/chain/gb"]
        inds = f["global_fit/inds/gb"]
        # layout (niter, ntemps, nwalkers, nleaves_max, ndim)
        coords = np.asarray(chain[it][0])   # (nwalkers, nleaves_max, ndim)
        alive = np.asarray(inds[it][0]).astype(bool)  # (nwalkers, nleaves_max)
    return coords[alive]


def _catalogue(f0_lo_mHz, f0_hi_mHz):
    """Injected truth in the band -> dict(f0[mHz], amp, Mc_eff, alpha, sin_delta)."""
    import h5py

    from gbgpu.utils.utility import get_chirp_mass_from_f_fdot

    path = _env("CATALOGUE_H5") or os.path.join(
        os.environ["MOJITO_DATA_PATH"], "catalogues",
        "wdwd_cat_mojito_lite_processed.hdf5")
    with h5py.File(path, "r") as f:
        B = f["Binaries"]
        f0 = B["GW22FrequencySSBFrame"][:] * 1e3
        keep = (f0 >= f0_lo_mHz) & (f0 <= f0_hi_mHz)
        amp = B["Amplitude"][keep]
        fdot = B["GW22FrequencyDerivativeSourceFrame"][keep]
        ra = B["RightAscension"][keep]
        dec = B["Declination"][keep]
    f0 = f0[keep]
    with np.errstate(invalid="ignore"):
        mc_eff = np.where(fdot > 0,
                          get_chirp_mass_from_f_fdot(f0 * 1e-3, np.clip(fdot, 0, None)),
                          np.nan)
    return dict(f0=f0, amp=amp, Mc_eff=mc_eff, alpha=ra, sin_delta=np.sin(dec), fdot=fdot)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5_path = _env("SEARCH_H5")
    if h5_path is None:
        raise SystemExit("set SEARCH_H5=<path to the *_testing.h5 search backend>")
    it = int(_env("ITER", "-1"))

    leaves = _recovered_leaves(h5_path, it)
    if leaves.size == 0:
        raise SystemExit(f"no alive leaves at iter {it} in {h5_path}")
    # distance basis columns
    r_dist = leaves[:, 0]        # kpc
    r_f0 = leaves[:, 1]          # mHz
    r_Mc = leaves[:, 2]          # Msol
    r_ratio = leaves[:, 8] if leaves.shape[1] > 8 else np.full(len(leaves), np.nan)

    # recovered amplitude from (f0, Mc, dist)
    from lisatools.globalfit.stock.erebor.transforms import gb_amp_from_dist
    r_amp = np.asarray(gb_amp_from_dist(r_f0 * 1e-3, r_Mc, r_dist))

    lo = float(_env("F0_LO_MHZ", str(r_f0.min() * 0.95)))
    hi = float(_env("F0_HI_MHZ", str(r_f0.max() * 1.05)))
    try:
        cat = _catalogue(lo, hi)
    except Exception as e:
        print(f"[cat] catalogue unavailable: {e}", flush=True)
        cat = None

    comb = None
    comb_npz = _env("FSTAT_COMB_NPZ")
    if comb_npz and os.path.exists(comb_npz):
        d = np.load(comb_npz)
        comb = (d["f0_nodes_mHz"], d["F_max"])

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    axA, axB, axC, axD = axes

    # A. F(f0) comb + injected f0 stems + recovered f0
    if comb is not None:
        axA.plot(comb[0], comb[1], lw=0.8, color="0.4", label="F-stat comb F(f0)")
        axA.set_ylabel("F  (= SNR$^2$/2)")
    if cat is not None:
        snr_proxy = cat["amp"] / np.nanmedian(cat["amp"])
        axA.vlines(cat["f0"], 0, snr_proxy * (comb[1].max() if comb is not None else 1.0),
                   color="tab:green", alpha=0.5, lw=1.5,
                   label="injected GBs (h ~ amp)")
    for x in r_f0:
        axA.axvline(x, color="tab:red", alpha=0.35, lw=0.8)
    axA.plot([], [], color="tab:red", alpha=0.6, lw=0.8, label="recovered leaves")
    axA.legend(loc="upper right", fontsize=8)
    axA.set_title(f"F-stat / search alignment  ({os.path.basename(h5_path)}, iter {it}, "
                  f"{len(leaves)} alive leaves)")

    # B. amplitude vs f0
    if cat is not None:
        axB.scatter(cat["f0"], cat["amp"], s=60, marker="x", color="tab:green",
                    label="injected A")
    axB.scatter(r_f0, r_amp, s=18, color="tab:red", alpha=0.6, label="recovered A (from dist)")
    axB.set_yscale("log")
    axB.set_ylabel("amplitude")
    axB.legend(loc="upper right", fontsize=8)

    # C. Mc vs f0
    if cat is not None:
        axC.scatter(cat["f0"], cat["Mc_eff"], s=60, marker="x", color="tab:green",
                    label="injected Mc_eff (fdot>0)")
    axC.scatter(r_f0, r_Mc, s=18, color="tab:red", alpha=0.6, label="recovered Mc")
    axC.set_ylabel("chirp mass [Msol]")
    axC.legend(loc="upper right", fontsize=8)

    # D. fdot_astro_ratio recovered
    axD.scatter(r_f0, r_ratio, s=18, color="tab:red", alpha=0.6, label="recovered r")
    axD.axhline(0.0, color="0.6", lw=0.8)
    axD.axhline(-2.0, color="tab:blue", lw=0.8, ls="--", label="r=-2 (interacting DWD seed)")
    axD.set_ylabel("fdot_astro_ratio")
    axD.set_xlabel("f0 [mHz]")
    axD.legend(loc="upper right", fontsize=8)

    out = _env("OUT_PNG", "./fstat_alignment.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {out}", flush=True)
    # quick numeric summary
    if cat is not None:
        print(f"[summary] injected in band: {len(cat['f0'])}   recovered leaves: {len(leaves)}",
              flush=True)
        print(f"[summary] recovered f0 [mHz]: {np.sort(r_f0)}", flush=True)
        print(f"[summary] injected  f0 [mHz]: {np.sort(cat['f0'])}", flush=True)


if __name__ == "__main__":
    main()
