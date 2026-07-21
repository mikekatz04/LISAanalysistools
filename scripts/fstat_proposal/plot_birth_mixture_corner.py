"""Corner plot of the assembled GB RJ-birth intrinsic distribution.

Rebuilds the birth mixture exactly as ``run_fstat_rj_search`` wires it —
local F-stat peak grid(s) + linear-in-F comb component + uniform floor —
from cached npz grids (no kernel evaluations), draws samples, and renders a
``corner`` plot of the 4 intrinsic sampling parameters with all in-band
catalogue sources overlaid.

Two figures: the ACTUAL birth mixture (peak grid at beta=1 -- delta-like at
the loudest source by design) and a tempered-landscape variant (peak grid at
beta=0.01) that makes the local structure readable.

Usage::

    python plot_birth_mixture_corner.py <comb.npz> <out_prefix> \
        <peak_grid1.npz> [peak_grid2.npz ...]

Environment: MOJITO_DATA_PATH (catalogue overlay), FSTAT_COMB_WEIGHT [0.45],
FSTAT_FLOOR_EPS [0.1], N_SAMPLES [200000].
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import corner as corner_pkg
import h5py

from lisatools.sampling.fstat_proposal import (
    CombIntrinsicProposal,
    FStatProposal4D,
    MixtureProposal,
    UniformFloorMixture,
)

LABELS = [r"f0 [mHz]", r"Mc [$M_\odot$]", r"$\alpha$ [rad]", r"$\sin\delta$"]
TRUTH_BAND75 = [7.580260, 0.3355, 4.9791, -0.063058]  # loudest in-band GB
MC_LIMS = (0.001, 1.0)


def band_catalogue(f0_lo_mHz, f0_hi_mHz):
    path = os.path.join(
        os.environ.get(
            "MOJITO_DATA_PATH",
            os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
        ),
        "catalogues", "wdwd_cat_mojito_lite_processed.hdf5",
    )
    from gbgpu.utils.utility import get_chirp_mass_from_f_fdot

    with h5py.File(path, "r") as f:
        B = f["Binaries"]
        f0 = B["GW22FrequencySSBFrame"][:] * 1e3
        keep = (f0 >= f0_lo_mHz) & (f0 <= f0_hi_mHz)
        fdot = B["GW22FrequencyDerivativeSourceFrame"][keep]
        ra = B["RightAscension"][keep]
        dec = B["Declination"][keep]
        amp = B["Amplitude"][keep]
    f0 = f0[keep]
    with np.errstate(invalid="ignore"):
        mc = np.where(fdot > 0,
                      get_chirp_mass_from_f_fdot(f0 * 1e-3,
                                                 np.clip(fdot, 0, None)),
                      np.nan)
    return np.column_stack([f0, mc, ra, np.sin(dec)]), amp


def build_mixture(peak_grid_paths, comb_path, peak_beta=1.0,
                  comb_weight=0.45, floor_eps=0.1):
    peaks = []
    for path in peak_grid_paths:
        d = np.load(path)
        peaks.append(FStatProposal4D.from_grid(
            (d["f0_ax"], d["Mc_ax"], d["alpha_ax"], d["sin_delta_ax"]),
            peak_beta * d["logp_grid"], beta=peak_beta,
        ))
    c = np.load(comb_path)
    comb = CombIntrinsicProposal(c["f0_nodes_mHz"], c["F_max"],
                                 mc_lims=MC_LIMS)
    peak_share = (1.0 - comb_weight - floor_eps) / len(peaks)
    base = MixtureProposal(peaks + [comb],
                           weights=[peak_share] * len(peaks) + [comb_weight])
    lo = [float(c["f0_nodes_mHz"][0]), MC_LIMS[0], 0.0, -1.0]
    hi = [float(c["f0_nodes_mHz"][-1]), MC_LIMS[1], 2.0 * np.pi, 1.0]
    return UniformFloorMixture(base, lo, hi, eps=floor_eps, seed=7), (lo, hi)


def render(mix, box, out_path, title, cat, cat_amp, n_samples):
    lo, hi = box
    s = np.asarray(mix.rvs(size=(n_samples,)))
    fig = corner_pkg.corner(
        s,
        labels=LABELS,
        bins=70,
        range=[(lo[j], hi[j]) for j in range(4)],
        truths=TRUTH_BAND75,
        truth_color="red",
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
        plot_contours=False,
        color="C0",
        hist_kwargs=dict(density=True),
    )
    axes = np.array(fig.axes).reshape(4, 4)
    for i in range(4):
        for j in range(i):
            keep = np.isfinite(cat[:, j]) & np.isfinite(cat[:, i])
            axes[i, j].scatter(cat[keep, j], cat[keep, i], s=40, marker="P",
                               c="w", edgecolors="k", linewidths=0.7,
                               zorder=6)
    fig.suptitle(title, y=1.0, fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)
    return s


def main():
    comb_path, out_prefix = sys.argv[1:3]
    peak_grid_paths = sys.argv[3:]
    if not peak_grid_paths:
        sys.exit("need at least one peak grid npz")
    comb_weight = float(os.environ.get("FSTAT_COMB_WEIGHT", 0.45))
    floor_eps = float(os.environ.get("FSTAT_FLOOR_EPS", 0.1))
    n_samples = int(os.environ.get("N_SAMPLES", 200_000))

    c = np.load(comb_path)
    cat, cat_amp = band_catalogue(float(c["f0_nodes_mHz"][0]),
                                  float(c["f0_nodes_mHz"][-1]))
    print(f"[cat] {len(cat)} catalogue sources in band; "
          f"{len(peak_grid_paths)} local peak grid(s)", flush=True)

    mix, box = build_mixture(peak_grid_paths, comb_path, peak_beta=1.0,
                             comb_weight=comb_weight, floor_eps=floor_eps)
    s = render(mix, box, f"{out_prefix}_corner_beta1.png",
               f"GB RJ-birth intrinsic mixture (actual, {len(peak_grid_paths)}"
               r" peak grid(s) at $\beta$=1) + comb(w$\propto$F) + floor",
               cat, cat_amp, n_samples)
    inb = np.abs(s[:, 0] - TRUTH_BAND75[0]) < 2.5e-3
    print(f"[diag] beta=1 mixture: {inb.mean() * 100:.1f}% of draws within "
          "the loudest-peak box", flush=True)

    mix_t, box = build_mixture(peak_grid_paths, comb_path, peak_beta=0.01,
                               comb_weight=comb_weight, floor_eps=floor_eps)
    render(mix_t, box, f"{out_prefix}_corner_tempered.png",
           "Same mixture, peak grids tempered to " r"$\beta$=0.01 "
           "(landscape view)", cat, cat_amp, n_samples)


if __name__ == "__main__":
    main()
