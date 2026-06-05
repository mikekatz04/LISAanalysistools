#!/usr/bin/env python
"""Post-hoc outlier analysis for the v2 polyphase mm sweep cache.

Loads ``v2_polyphase_mm_sweep.npz`` (saved with ``CACHE=1`` by
``gb_signal_het_wdm_v2_mm_sweep.py``), then plots per-draw mm scatter
vs Nt_layer with the top-N worst draws highlighted, and correlates the
outlier mm against source parameters (f0, amp, sin(beta), SNR, m_floor
relative position).

Run::
    python gb_signal_het_wdm_v2_mm_analysis.py
Env vars:
    NPZ           input NPZ path (default v2_polyphase_mm_sweep.npz)
    OUT_PNG       output plot path
    N_OUTLIERS    number of worst draws to highlight (default 5)
"""

from __future__ import annotations

import os
import sys

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    npz_path = os.environ.get("NPZ", "v2_polyphase_mm_sweep.npz")
    data = np.load(npz_path)
    nt_layer = data["nt_layer_list"]
    stride = data["stride"]
    n_sparse = data["n_sparse"]
    mm5 = data["mm5"]                     # (n_ntl, N_draws)
    mm2 = data["mm2"]
    params_ref = data["params_ref"]        # (N_draws, 9)
    params_cand = data["params_cand"]      # (N_draws, 9)
    snr = data["snr"]                      # (N_draws,)
    df0_frac = float(data["df0_frac"])
    Nf = int(data["Nf"]); Nt = int(data["Nt"]); dt = float(data["dt"])
    layer_df = float(data["layer_df"])
    Nf_active = int(data["Nf_active"]); Nt_active = int(data["Nt_active"])
    N_draws = mm5.shape[1]
    print(f"[load] {npz_path}", flush=True)
    print(f"   N_draws={N_draws}  Nt_layer count={len(nt_layer)}", flush=True)
    print(f"   df0_frac={df0_frac}  df0={df0_frac*layer_df*1e3:.3e} mHz",
          flush=True)

    # Summary stats per Nt_layer
    print("\n[summary] per-Nt_layer mm5/mm2 statistics:", flush=True)
    print(f"   {'Nt_layer':>8s} {'stride':>6s} {'mm5 med':>10s} {'mm5 90%':>10s} "
          f"{'mm5 99%':>10s} {'mm5 max':>10s}  {'mm2 med':>10s} {'mm2 99%':>10s} "
          f"{'mm2 max':>10s}", flush=True)
    for li, ntl in enumerate(nt_layer):
        ms = mm5[li]; m2 = mm2[li]
        print(f"   {ntl:8d} {stride[li]:6d} {np.median(ms):10.2e} "
              f"{np.percentile(ms, 90):10.2e} {np.percentile(ms, 99):10.2e} "
              f"{ms.max():10.2e}  {np.median(m2):10.2e} "
              f"{np.percentile(m2, 99):10.2e} {m2.max():10.2e}",
              flush=True)

    # Pick a reference Nt_layer for outlier inspection (median in the sweep).
    # We want the smallest Nt_layer that's still "useful" -- pick the one
    # with smallest median below threshold ~1e-6, fallback to middle.
    threshold_med = 1e-6
    valid_li = [li for li in range(len(nt_layer))
                if np.median(mm5[li]) < threshold_med]
    if valid_li:
        # smallest Nt_layer with median below threshold
        ref_li = min(valid_li, key=lambda x: nt_layer[x])
    else:
        ref_li = len(nt_layer) // 2
    print(f"\n[outliers] inspecting Nt_layer={nt_layer[ref_li]} "
          f"(stride={stride[ref_li]}, median mm5={np.median(mm5[ref_li]):.2e})",
          flush=True)

    N_OUTLIERS = int(os.environ.get("N_OUTLIERS", "8"))
    mm5_ref = mm5[ref_li]
    order = np.argsort(mm5_ref)[::-1]                # worst first
    worst = order[:N_OUTLIERS]
    print(f"   top {N_OUTLIERS} worst draws (by mm5):", flush=True)
    print(f"   {'idx':>4s} {'snr':>7s} {'mm5':>10s} {'mm2':>10s} "
          f"{'f0(mHz)':>10s} {'f0/layer_df':>11s} {'m_floor':>8s} "
          f"{'fdot':>11s} {'amp':>9s} {'sin(beta)':>9s}",
          flush=True)
    for j, di in enumerate(worst):
        p = params_cand[di]
        m_floor = int(p[1] / layer_df)
        print(f"   {di:4d} {snr[di]:7.1f} {mm5_ref[di]:10.2e} "
              f"{mm2[ref_li, di]:10.2e} {p[1]*1e3:10.5f} {p[1]/layer_df:11.3f} "
              f"{m_floor:8d} {p[2]:11.2e} {p[0]:9.2e} {np.sin(p[8]):9.3f}",
              flush=True)

    # Outlier characterization at the ref Nt_layer
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    f0_mHz = params_cand[:, 1] * 1e3
    f0_layers = params_cand[:, 1] / layer_df
    amp = params_cand[:, 0]
    fdot = params_cand[:, 2]
    sinb = np.sin(params_cand[:, 8])

    # (0,0) mm5 vs Nt_layer per-draw scatter
    ax = axs[0, 0]
    for di in range(N_draws):
        is_outlier = di in worst
        col = "C3" if is_outlier else "C0"
        alpha = 0.85 if is_outlier else 0.2
        lw = 1.0 if is_outlier else 0.4
        ax.plot(nt_layer, mm5[:, di], color=col, alpha=alpha, linewidth=lw)
    ax.plot(nt_layer, np.median(mm5, axis=1), "ko-", label="median", linewidth=1.5)
    ax.plot(nt_layer, np.percentile(mm5, 99, axis=1), "k^--", label="99%-tile",
            linewidth=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Nt_layer"); ax.set_ylabel("mm5")
    ax.set_title(f"mm5 vs Nt_layer  (red = top-{N_OUTLIERS} outliers)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="best", fontsize=9)

    # (0,1) mm2 vs Nt_layer
    ax = axs[0, 1]
    for di in range(N_draws):
        is_outlier = di in worst
        col = "C3" if is_outlier else "C0"
        alpha = 0.85 if is_outlier else 0.2
        lw = 1.0 if is_outlier else 0.4
        ax.plot(nt_layer, mm2[:, di], color=col, alpha=alpha, linewidth=lw)
    ax.plot(nt_layer, np.median(mm2, axis=1), "ko-", linewidth=1.5)
    ax.plot(nt_layer, np.percentile(mm2, 99, axis=1), "k^--", linewidth=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Nt_layer"); ax.set_ylabel("mm2")
    ax.set_title("mm2 vs Nt_layer")
    ax.grid(alpha=0.3, which="both")

    # (0,2) mm5 (at ref Nt_layer) vs f0
    ax = axs[0, 2]
    ax.scatter(f0_mHz, mm5_ref, c="C0", alpha=0.5, label="all draws")
    ax.scatter(f0_mHz[worst], mm5_ref[worst], c="C3", s=80,
               label=f"top-{N_OUTLIERS} outliers")
    ax.set_yscale("log")
    ax.set_xlabel("f0 (mHz)")
    ax.set_ylabel(f"mm5 at Nt_layer={nt_layer[ref_li]}")
    ax.set_title("mm5 vs f0")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    # (0,3) mm5 vs fractional position within layer
    ax = axs[0, 3]
    frac_in_layer = f0_layers - np.floor(f0_layers)
    ax.scatter(frac_in_layer, mm5_ref, c="C0", alpha=0.5)
    ax.scatter(frac_in_layer[worst], mm5_ref[worst], c="C3", s=80)
    ax.set_yscale("log")
    ax.set_xlabel("f0 fractional position within layer")
    ax.set_ylabel(f"mm5 at Nt_layer={nt_layer[ref_li]}")
    ax.set_title("mm5 vs fractional layer position\n(does it spike near 0 or 1?)")
    ax.grid(alpha=0.3)

    # (1,0) mm5 vs SNR
    ax = axs[1, 0]
    ax.scatter(snr, mm5_ref, c="C0", alpha=0.5)
    ax.scatter(snr[worst], mm5_ref[worst], c="C3", s=80)
    ax.set_yscale("log")
    ax.set_xlabel("SNR (full-band, lisatools)")
    ax.set_ylabel(f"mm5 at Nt_layer={nt_layer[ref_li]}")
    ax.set_title("mm5 vs SNR")
    ax.grid(alpha=0.3)

    # (1,1) mm5 vs amp
    ax = axs[1, 1]
    ax.scatter(amp, mm5_ref, c="C0", alpha=0.5)
    ax.scatter(amp[worst], mm5_ref[worst], c="C3", s=80)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("amp (sampled)")
    ax.set_ylabel(f"mm5 at Nt_layer={nt_layer[ref_li]}")
    ax.set_title("mm5 vs amp")
    ax.grid(alpha=0.3, which="both")

    # (1,2) mm5 vs fdot
    ax = axs[1, 2]
    ax.scatter(fdot, mm5_ref, c="C0", alpha=0.5)
    ax.scatter(fdot[worst], mm5_ref[worst], c="C3", s=80)
    ax.set_yscale("log")
    ax.set_xlabel("fdot")
    ax.set_ylabel(f"mm5 at Nt_layer={nt_layer[ref_li]}")
    ax.set_title("mm5 vs fdot")
    ax.grid(alpha=0.3)

    # (1,3) mm5 vs sin(beta)
    ax = axs[1, 3]
    ax.scatter(sinb, mm5_ref, c="C0", alpha=0.5)
    ax.scatter(sinb[worst], mm5_ref[worst], c="C3", s=80)
    ax.set_yscale("log")
    ax.set_xlabel("sin(beta)")
    ax.set_ylabel(f"mm5 at Nt_layer={nt_layer[ref_li]}")
    ax.set_title("mm5 vs sin(beta)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"v2 polyphase + carrier de-rotation  mm sweep  "
        f"(N={N_draws}, df0={df0_frac}*layer_df = {df0_frac*layer_df*1e6:.3f} uHz)",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = os.environ.get("OUT_PNG", "v2_mm_outliers.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {out_png}", flush=True)
    print("DONE.")


if __name__ == "__main__":
    sys.exit(main())
