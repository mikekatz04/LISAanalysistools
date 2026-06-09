#!/usr/bin/env python
"""Corner plot of the last 300 cold-chain steps from the in-progress stretch
MCMC, contours only (1, 2, 3 sigma), with injection truths overlaid."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

from eryn.backends import HDFBackend


SAMPLED_BASIS = ["amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam", "sinbeta"]

# Injection from test_gb_signal_het_mcmc.py defaults at F0_MHZ=14.22, SNR=50.
# (amp scales with SNR; for SNR=50 we previously computed ~6.05e-23.)
TRUTHS = np.array([
    6.054094e-23,    # amp
    14.22e-3,        # f0
    1.0e-16,         # fdot
    1.4,             # phi0
    np.cos(np.pi/3), # cosinc = 0.5
    0.7,             # psi
    2.1,             # lam
    np.sin(0.5),     # sinbeta ~ 0.479
], dtype=float)

# 2D Gaussian containment levels for 1, 2, 3 sigma.
SIGMA_LEVELS = (0.393, 0.865, 0.989)


def load_chain(path, retries=20):
    last_err = None
    for k in range(retries):
        try:
            be = HDFBackend(path)
            return be.get_chain()["gb"]   # (nsteps, ntemps, nwalkers, 1, ndim)
        except (OSError, BlockingIOError) as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"could not open {path} after {retries} retries: "
                       f"{last_err}")


def main():
    backend_path = os.environ.get("BACKEND_PATH", "mcmc_stretch_clip.h5")
    n_steps = int(os.environ.get("N_STEPS_TAIL", "300"))
    out_path = os.environ.get("OUT", "corner_last_300.png")

    samples = load_chain(backend_path)
    print(f"[load] {backend_path}: shape={samples.shape}", flush=True)
    n_total = samples.shape[0]
    if n_steps > n_total:
        n_steps = n_total
    tail = samples[-n_steps:, 0].reshape(-1, len(SAMPLED_BASIS))
    print(f"[tail] using last {n_steps} steps x {samples.shape[2]} walkers "
          f"= {tail.shape[0]} samples", flush=True)

    fig = corner.corner(
        tail,
        labels=SAMPLED_BASIS,
        truths=TRUTHS,
        truth_color="black",
        color="C0",
        levels=SIGMA_LEVELS,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        smooth=1.0,
        contour_kwargs={"linewidths": 1.4},
        truth_kwargs={"alpha": 0.7, "lw": 1.0},
        hist_kwargs={"density": True, "histtype": "step", "lw": 1.0},
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
    )
    fig.suptitle(
        f"GB signal-het stretch (max_r=5, full-angle prior)  "
        f"last {n_steps} steps, 1/2/3 sigma contours",
        fontsize=11, y=1.01,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[corner] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
