#!/usr/bin/env python
"""Plot 5-layer mismatch vs every GB parameter, highlighting which one
actually drives the spread (1e-13 → 1e-2 observed in
``gb_prior_*_results.npz``).

Usage:  python gb_lookup_pattern_plots.py [npz_prefix]
        (default: 'gb_prior_pattern')
"""
import os
import sys
import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "gb_prior_pattern"
    npz_path = prefix + "_results.npz"
    if not os.path.exists(npz_path):
        sys.exit(f"missing {npz_path}")
    d = np.load(npz_path, allow_pickle=True)

    snrs = np.asarray(d["snr"], dtype=float)
    mm5 = np.real(np.asarray(d["mismatch_5"], dtype=float))
    params = np.asarray(d["params"], dtype=float)
    layer_df = (
        float(d["f0_hi_hz"]) - float(d["f0_lo_hz"])
    ) / max(1, len(snrs))  # rough — we recompute below from f0/layer_df
    # Better: recover layer_df from the script's setup
    # dt=10, Nf=1460 → layer_df = 1/(2*Nf*dt)
    layer_df = 1.0 / (2.0 * 1460.0 * 10.0)

    A = params[:, 0]
    f0 = params[:, 1]
    fdot = params[:, 2]
    phi0 = params[:, 4]
    inc = params[:, 5]
    psi = params[:, 6]
    lam = params[:, 7]
    beta = params[:, 8]

    m_continuous = f0 / layer_df
    m_layer = np.floor(m_continuous).astype(int)
    f0_frac = m_continuous - m_layer  # in [0, 1) — sub-layer position
    f0_distance_from_center = np.abs(f0_frac - 0.5)  # 0 = at center, 0.5 = at boundary

    # ---- WDM-domain-specific geometry --------------------------------
    # The script uses min_freq=0.0001, max_freq=35e-3 → active layer band.
    min_freq, max_freq = 0.0001, 35.0e-3
    ind_min_f = int(np.ceil(min_freq / layer_df))   # mirrors WDMSettings rule
    ind_max_f = int(max_freq / layer_df)
    dist_to_low_edge = (m_layer - ind_min_f).astype(float)    # layers above min_freq
    dist_to_high_edge = (ind_max_f - m_layer).astype(float)   # layers below max_freq
    dist_to_nearest_edge = np.minimum(dist_to_low_edge, dist_to_high_edge)

    # Lookup table sub-layer interpolation grid: norm_freq_single_layer =
    # arange(0, layer_df, EPS_FREQ*layer_df).  At lookup time the source's
    # f_norm = f - m_layer*layer_df gets linear-interpolated between adjacent
    # samples. "Distance from nearest sample" probes whether linear-interp
    # error correlates with mm_5.
    eps_freq = 0.001          # default EPS_FREQ in the script
    delta_norm = eps_freq * layer_df
    f_norm = f0 - m_layer * layer_df            # in [0, layer_df)
    sample_idx = np.floor(f_norm / delta_norm)
    f_norm_at_sample = sample_idx * delta_norm
    interp_offset = (f_norm - f_norm_at_sample) / delta_norm  # in [0, 1) — sub-cell
    interp_distance = np.abs(interp_offset - 0.5)             # 0 = at cell center, 0.5 = at cell edge
    # |fdot| relative to a notional fdot-step (table is fdot=0 only, so any
    # non-zero fdot is "outside" the table — measure in absolute units).
    fdot_over_layer_df_per_Tobs = np.abs(fdot) * 31557600.0 / layer_df   # rough Doppler scale

    fin = np.isfinite(mm5) & (mm5 > 0)
    print(f"[loaded] {len(mm5)} draws from {npz_path};  {fin.sum()} usable")
    print(f"[wdm geom] layer_df = {layer_df:.4e} Hz  "
          f"ind_min_f = {ind_min_f}  ind_max_f = {ind_max_f}  "
          f"|active| = {ind_max_f - ind_min_f + 1} layers")
    print(f"[wdm geom] table sub-layer grid spacing = {delta_norm:.4e} Hz  "
          f"({1.0/eps_freq:.0f} samples/layer)")

    # ---- Spearman correlations with log10(mm5) ------------------------
    log_mm5 = np.log10(mm5[fin])
    features = {
        "log10(A)":          np.log10(A[fin]),
        "log10(SNR)":        np.log10(np.maximum(snrs[fin], 1e-9)),
        "f0 [mHz]":          f0[fin] * 1e3,
        "fdot":              fdot[fin],
        "|fdot|":            np.abs(fdot[fin]),
        "phi0":              phi0[fin],
        "inc":               inc[fin],
        "|cos(inc)|":        np.abs(np.cos(inc[fin])),
        "sin(inc)":          np.sin(inc[fin]),
        "psi":               psi[fin],
        "lam":               lam[fin],
        "beta":              beta[fin],
        "|beta|":            np.abs(beta[fin]),
        # ---- WDM-domain geometry ----
        "m_layer":                m_layer[fin].astype(float),
        "m_layer % 2":            (m_layer[fin] % 2).astype(float),
        "f0_frac (within layer)": f0_frac[fin],
        "|f0_frac-0.5|":          f0_distance_from_center[fin],
        "layers above min_freq":  dist_to_low_edge[fin],
        "layers below max_freq":  dist_to_high_edge[fin],
        "log10(dist to band edge)": np.log10(np.maximum(dist_to_nearest_edge[fin], 0.5)),
        "interp_offset (sub-cell)": interp_offset[fin],
        "|interp_offset-0.5|":      interp_distance[fin],
    }

    print("\n[Spearman rank correlation with log10(mm_5)]")
    print(f"  {'feature':>18s}  {'rho':>7s}  {'pval':>8s}")
    rows = []
    for name, x in features.items():
        if np.unique(x).size < 2:
            continue
        rho, pval = stats.spearmanr(x, log_mm5)
        rows.append((name, rho, pval))
        marker = "  <<" if abs(rho) > 0.3 else ""
        print(f"  {name:>18s}  {rho:+.4f}  {pval:.2e}{marker}")

    # ---- per-parameter plots -----------------------------------------
    plt.rcParams["text.usetex"] = False
    panels = [
        ("log10(A)",       np.log10(A[fin]),        False),
        ("log10(SNR)",     np.log10(np.maximum(snrs[fin], 1e-9)), False),
        ("f0 [mHz]",       f0[fin] * 1e3,           False),
        ("|fdot|",         np.abs(fdot[fin]),       True),
        ("inc",            inc[fin],                False),
        ("|cos(inc)|",     np.abs(np.cos(inc[fin])), False),
        ("psi",            psi[fin],                False),
        ("lam",            lam[fin],                False),
        ("beta",           beta[fin],               False),
        # WDM-domain geometry panels
        ("m_layer",                m_layer[fin].astype(float),   False),
        ("f0_frac (within layer)", f0_frac[fin],                 False),
        ("|f0_frac-0.5|",          f0_distance_from_center[fin], False),
        ("layers above min_freq",  dist_to_low_edge[fin],        False),
        ("layers below max_freq",  dist_to_high_edge[fin],       False),
        ("interp_offset (sub-cell)", interp_offset[fin],         False),
        ("|interp_offset-0.5|",      interp_distance[fin],       False),
    ]
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows))
    axes = axes.flatten()
    for ax, (name, x, logx) in zip(axes, panels):
        ax.scatter(x, mm5[fin], s=12, alpha=0.65, color="steelblue")
        if logx:
            ax.set_xscale("symlog", linthresh=1e-20)
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("mismatch (5 layers)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    plt.tight_layout()
    out = prefix + "_mm_vs_params.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n[saved] {out}")

    # ---- focused: the top-3 correlators -----------------------------
    rows_sorted = sorted(rows, key=lambda r: -abs(r[1]))
    if rows_sorted:
        top = rows_sorted[:3]
        fig, axes = plt.subplots(1, len(top), figsize=(5.5 * len(top), 4.2))
        if len(top) == 1:
            axes = [axes]
        for ax, (name, rho, pval) in zip(axes, top):
            x = features[name]
            ax.scatter(x, mm5[fin], s=20, alpha=0.7, color="darkorange")
            ax.set_yscale("log")
            ax.set_xlabel(name)
            ax.set_ylabel("mismatch (5 layers)")
            ax.set_title(f"rho={rho:+.3f}  p={pval:.1e}")
            ax.grid(True, which="both", ls=":", alpha=0.4)
        plt.tight_layout()
        out2 = prefix + "_top_correlators.png"
        plt.savefig(out2, dpi=140)
        plt.close(fig)
        print(f"[saved] {out2}")

    # ---- compare best vs worst quartiles ----------------------------
    n_q = max(3, int(0.15 * fin.sum()))
    order = np.argsort(mm5[fin])
    best_idx = np.where(fin)[0][order[:n_q]]
    worst_idx = np.where(fin)[0][order[-n_q:]]

    def _stats(idx):
        return dict(
            A=A[idx], f0=f0[idx]*1e3, fdot=fdot[idx],
            inc=inc[idx], cos_inc=np.cos(inc[idx]), psi=psi[idx],
            lam=lam[idx], beta=beta[idx], snr=snrs[idx],
            f0_frac=f0_frac[idx], m_layer=m_layer[idx],
        )

    bst = _stats(best_idx)
    wst = _stats(worst_idx)
    print(f"\n[best {n_q} (mm_5 < {mm5[fin][order[n_q-1]]:.2e}) vs worst {n_q} (mm_5 > {mm5[fin][order[-n_q]]:.2e})]")
    print(f"  {'feature':>10s}  {'best median':>14s}  {'worst median':>14s}  {'best range':>22s}  {'worst range':>22s}")
    for k in ("A", "f0", "fdot", "inc", "cos_inc", "psi", "lam", "beta", "snr", "f0_frac", "m_layer"):
        bm = float(np.median(bst[k])); wm = float(np.median(wst[k]))
        br = f"[{np.min(bst[k]):+.2e},{np.max(bst[k]):+.2e}]"
        wr = f"[{np.min(wst[k]):+.2e},{np.max(wst[k]):+.2e}]"
        print(f"  {k:>10s}  {bm:+14.4e}  {wm:+14.4e}  {br:>22s}  {wr:>22s}")


if __name__ == "__main__":
    main()
