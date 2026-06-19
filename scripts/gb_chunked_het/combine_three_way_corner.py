#!/usr/bin/env python
"""Overlay the FD / chunked-het / signal-het GB MCMC posteriors from saved chains.

Reads the per-method HDF5 chains written by gb_mojito_mcmc_three_ways.py (which may
have been run in separate jobs / with different step counts) and makes one corner
overlay. Uses build_shared() for the injection truths + the source rank.

Run:  GB_RANK=0 OUT_DIR=/tmp/gbmcmc_long python combine_three_way_corner.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
from eryn.backends import HDFBackend
import gbgpu  # noqa: F401

from gb_mojito_mcmc_three_ways import build_shared

LABELS = ["logA", "f0 [mHz]", "fdot", "phi0", "cos_iota", "psi", "alpha", "sin_delta"]
METHODS = [("fd", "C0"), ("wdm_chunked", "C1"), ("wdm_sighet", "C2")]
BURN_FRAC = float(os.environ.get("BURN_FRAC", "0.3"))
BURN_STEPS = int(os.environ.get("BURN_STEPS", "0"))  # absolute burn-in; 0 -> use BURN_FRAC


def main():
    s = build_shared()
    rank, inj = s["rank"], s["inj_sampled"]
    out_dir = os.environ.get("OUT_DIR", "/tmp/gbmcmc_long")
    samples = {}
    for kind, _ in METHODS:
        h5 = os.path.join(out_dir, f"mcmc_mojito_rank{rank}_{kind}.h5")
        if not os.path.exists(h5):
            print(f"[skip] missing {h5}", flush=True); continue
        try:
            chain = HDFBackend(h5).get_chain()["gb"]      # (nsteps, ntemps, nwalkers, 1, 8)
        except Exception as e:
            print(f"[skip] {kind}: unreadable ({e})", flush=True); continue
        nsteps = chain.shape[0]
        nb = min(BURN_STEPS, nsteps - 1) if BURN_STEPS > 0 else max(1, int(BURN_FRAC * nsteps))
        samples[kind] = chain[nb:, 0].reshape(-1, 8)      # cold chain, post burn-in
        print(f"[{kind:>11}] {nsteps} steps x {chain.shape[2]} walkers, burn {nb} "
              f"-> {samples[kind].shape[0]} cold samples", flush=True)

    if not samples:
        print("no chains found", flush=True); return 1

    fig = None
    for kind, color in METHODS:
        if kind not in samples:
            continue
        fig = corner.corner(
            samples[kind], labels=LABELS, fig=fig, color=color,
            truths=inj, truth_color="black", levels=(0.393, 0.865, 0.989),
            plot_datapoints=False, plot_density=False, no_fill_contours=True,
            smooth=1.0, hist_kwargs={"density": True})
    fig.legend(handles=[plt.Line2D([0], [0], color=c, lw=2, label=k)
                        for k, c in METHODS if k in samples],
               loc="upper right", fontsize=12)
    fig.suptitle(f"mojito GB rank {rank} (f0={s['f0']*1e3:.3f} mHz): "
                 f"FD vs chunked-het vs signal-het posteriors", y=1.02)
    out = os.path.join(out_dir, f"corner_mojito_rank{rank}_three_ways.png")
    fig.savefig(out, bbox_inches="tight", dpi=140); plt.close(fig)
    print(f"\n[plot] {out}", flush=True)

    # per-parameter mean/std table
    print(f"\n  {'param':>10}{'inj':>14}" + "".join(f"{k+' mean':>15}" for k, _ in METHODS if k in samples), flush=True)
    for j, lab in enumerate(LABELS):
        row = f"  {lab:>10}{inj[j]:>+14.5e}"
        for k, _ in METHODS:
            if k in samples:
                row += f"{samples[k][:, j].mean():>+15.5e}"
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
