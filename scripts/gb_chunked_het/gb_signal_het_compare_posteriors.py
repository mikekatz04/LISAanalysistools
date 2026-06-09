#!/usr/bin/env python
"""Overlay two GB signal-het posterior chains on a single corner plot.

Usage::
    python gb_signal_het_compare_posteriors.py \\
        STRETCH=mcmc_stretch_1000.h5 \\
        NUTS=mcmc_nuts.h5 \\
        OUT=corner_compare.png

Reads each HDF backend produced by ``test_gb_signal_het_mcmc.py``, takes
the cold-temperature chain, drops the first 20% as burn-in, and overlays
both posteriors on a single corner plot with truth lines.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

from eryn.backends import HDFBackend


SAMPLED_BASIS = ["amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam",
                 "sinbeta"]


def load_cold_chain(path: str) -> np.ndarray:
    be = HDFBackend(path)
    samples = be.get_chain()["gb"]   # (nsteps, ntemps, nwalkers, 1, ndim)
    n_burn = max(1, int(0.2 * samples.shape[0]))
    flat = samples[n_burn:, 0].reshape(-1, len(SAMPLED_BASIS))
    return flat


def parse_args(argv):
    parsed = {"STRETCH": None, "NUTS": None, "OUT": "corner_compare.png",
              "INJ": None}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k in parsed:
                parsed[k] = v
    return parsed


def main():
    args = parse_args(sys.argv[1:])
    if not args["STRETCH"] or not args["NUTS"]:
        print(__doc__)
        return 2

    s_flat = load_cold_chain(args["STRETCH"])
    n_flat = load_cold_chain(args["NUTS"])
    print(f"[stretch] {args['STRETCH']}: {s_flat.shape[0]} samples", flush=True)
    print(f"[nuts]    {args['NUTS']}: {n_flat.shape[0]} samples", flush=True)

    # Recover truth from sample distribution overlap -- both runs use the same
    # injection, so use the mean of the stretch chain as a proxy if truth is
    # not passed in. Better: re-derive truth from the script's defaults.
    # The script uses fixed injection values:
    F0_MHZ = float(os.environ.get("F0_MHZ", "14.22"))
    truth = np.array([
        float(os.environ.get("AMP_INJ", "0.0")),     # 0 = look from chain
        F0_MHZ * 1e-3,
        1e-16,                                        # fdot
        1.4,                                          # phi0
        np.cos(np.pi / 3.0),                          # cosinc
        0.7,                                          # psi
        2.1,                                          # lam
        np.sin(0.5),                                  # sinbeta
    ], dtype=float)
    if truth[0] == 0.0:
        # Amp depends on SNR_TARGET; pull from the joined chain mean.
        truth[0] = np.median(np.concatenate([s_flat[:, 0], n_flat[:, 0]]))

    # Compute joint ranges so both posteriors share consistent axes.
    joint = np.concatenate([s_flat, n_flat], axis=0)
    ranges = []
    for k in range(joint.shape[1]):
        lo = np.percentile(joint[:, k], 0.5)
        hi = np.percentile(joint[:, k], 99.5)
        if hi - lo < 1e-30:
            lo, hi = lo - 1.0, lo + 1.0
        ranges.append((lo, hi))

    fig = corner.corner(
        s_flat,
        labels=SAMPLED_BASIS,
        truths=truth,
        truth_color="black",
        color="C0",
        range=ranges,
        hist_kwargs={"density": True},
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        label_kwargs={"fontsize": 9},
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.39, 0.86),
    )
    fig = corner.corner(
        n_flat,
        fig=fig,
        labels=SAMPLED_BASIS,
        color="C1",
        range=ranges,
        hist_kwargs={"density": True},
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        label_kwargs={"fontsize": 9},
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.39, 0.86),
    )

    # Legend.
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="C0", lw=3, label="stretch"),
            plt.Line2D([0], [0], color="C1", lw=3, label="NUTS"),
            plt.Line2D([0], [0], color="black", lw=1.5, label="truth"),
        ],
        loc="upper right",
        fontsize=11,
        frameon=True,
    )
    fig.suptitle("GB signal-het: stretch vs NUTS posteriors",
                 fontsize=12, y=1.01)
    fig.savefig(args["OUT"], bbox_inches="tight", dpi=120)
    print(f"[compare] saved -> {args['OUT']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
