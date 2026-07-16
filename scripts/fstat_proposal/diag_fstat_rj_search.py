"""Diagnostics for a run_fstat_rj_search.py backend.

Reads the eryn HDF backend and renders the birth-then-refine story:
per-iteration alive-leaf counts (cold chain), logL trace, alive-leaf f0
traces against the catalogue comb, and a final-iteration parameter summary
vs the target truth.

Usage:
    python diag_fstat_rj_search.py <backend.h5> <target: highest|band75> [out.png]
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

TRUTH = {
    "highest": dict(ID=7725228, f0=20.380377, Mc_eff=0.4658,
                    alpha=4.0617, sin_delta=-0.786363),
    "band75": dict(ID=1229636, f0=7.580260, Mc_eff=0.3355,
                   alpha=4.9791, sin_delta=-0.063058),
}


def catalogue_comb(f0_lo_mHz, f0_hi_mHz):
    try:
        path = os.path.join(
            os.environ.get(
                "MOJITO_DATA_PATH",
                os.path.expanduser(
                    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
            ),
            "catalogues", "wdwd_cat_mojito_lite_processed.hdf5",
        )
        with h5py.File(path, "r") as f:
            f0 = f["Binaries"]["GW22FrequencySSBFrame"][:] * 1e3
            keep = (f0 >= f0_lo_mHz) & (f0 <= f0_hi_mHz)
            amp = f["Binaries"]["Amplitude"][keep]
        return f0[keep], amp
    except Exception as e:
        print(f"[cat] unavailable: {e}")
        return None, None


def main():
    fp = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "highest"
    out = sys.argv[3] if len(sys.argv) > 3 else fp.replace(".h5", "_diag.png")
    truth = TRUTH[target]

    with h5py.File(fp, "r") as f:
        chain = f["mcmc/chain/gb"][:]          # (it, 1, T, W, L, 8)
        inds = f["mcmc/inds/gb"][:].astype(bool)
        ll = f["mcmc/log_like"][:]             # (it, 1, T, W)
        band_edges = f["mcmc/sub_backend/gb/band_edges"][:]
        caps = f["mcmc/sub_backend/gb/band_leaf_cap"][:]

    nit, _, T, W, L, ndim = chain.shape
    print(f"iterations={nit}  ntemps={T}  nwalkers={W}  nleaves_max={L}")

    alive_cold = inds[:, 0, 0]                 # (it, W, L)
    n_alive = alive_cold.sum(axis=-1)          # (it, W)
    ll_cold = ll[:, 0, 0]                      # (it, W)
    ll0 = ll_cold[0].min()

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)

    ax = axes[0]
    ax.plot(np.arange(nit), n_alive.mean(axis=1), "o-", color="C0",
            label="mean alive leaves / walker (cold)")
    ax.plot(np.arange(nit), n_alive.max(axis=1), "s--", color="C1", ms=4,
            label="max")
    ax.plot(np.arange(nit), caps.sum(axis=1), ":", color="0.4",
            label="sum of per-band leaf caps")
    ax.set_xlabel("iteration")
    ax.set_ylabel("alive leaves")
    ax.legend(fontsize=8)
    ax.set_title(f"Zero-leaf F-stat-proposal search — {target} "
                 f"(GB {truth['ID']})")

    ax = axes[1]
    dl = ll_cold - ll0
    ax.plot(np.arange(nit), dl.mean(axis=1), "o-", color="C0",
            label="mean cold logL − start")
    ax.fill_between(np.arange(nit), dl.min(axis=1), dl.max(axis=1),
                    alpha=0.25, color="C0", label="min–max")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Δ logL (cold chain)")
    ax.legend(fontsize=8)

    ax = axes[2]
    cf0, camp = catalogue_comb(band_edges[0] * 1e3, band_edges[-1] * 1e3)
    if cf0 is not None:
        for i, fv in enumerate(cf0):
            ax.axhline(fv, color="0.6", lw=0.8,
                       alpha=float(min(1.0, 0.2 + 0.8 * camp[i] / camp.max())),
                       zorder=0, label="catalogue GBs" if i == 0 else None)
    ax.axhline(truth["f0"], color="r", ls="--", lw=1.2, label="target GB")
    for it in range(nit):
        f0s = chain[it, 0, 0][alive_cold[it]][:, 1]
        ax.plot(np.full(f0s.shape, it), f0s, ".", color="C0", ms=5, alpha=0.6,
                label="alive-leaf f0 (cold)" if it == 0 else None)
    ax.set_xlabel("iteration")
    ax.set_ylabel("f0 [mHz]")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot] wrote {out}")

    # final-iteration summary vs truth
    pars = chain[-1, 0, 0][alive_cold[-1]]
    if len(pars):
        names = ["lnA", "f0", "Mc", "phi0", "cos_i", "psi", "alpha", "sin_d"]
        med = np.median(pars, axis=0)
        lo, hi = np.percentile(pars, [5, 95], axis=0)
        print("\nfinal cold-chain alive-leaf parameters (median [5%, 95%]):")
        for j, nm in enumerate(names):
            print(f"  {nm:6s} {med[j]:10.5f}  [{lo[j]:10.5f}, {hi[j]:10.5f}]")
        print(f"\ntruth: f0={truth['f0']:.5f}  Mc_eff={truth['Mc_eff']}  "
              f"alpha={truth['alpha']}  sin_delta={truth['sin_delta']}")
        df0 = np.abs(pars[:, 1] - truth["f0"])
        print(f"|f0 - truth|: median {np.median(df0):.2e} mHz  "
              f"max {df0.max():.2e} mHz")


if __name__ == "__main__":
    main()
