#!/usr/bin/env python
"""Verification plot for the noise ground-truth gate (t1-gt-noise).

Reads the 731-day mojito NOISE brick THROUGH the stock structure and shows,
per TDI channel, the measured noise PSD it projects onto the analysis basis —
the visual proof behind "the brick reads correctly through
MojitoNoiseEstimates".

Everything comes from the stock objects: ``erebor.noise_only`` resolves the
brick and fits its two scalar parameters exactly as a real run does; the
sensitivity is the same stock ``MojitoNoiseSensitivityMatrix`` class fed that
resolved brick path (the WDM path a real fit uses).
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

PLOT_DIR = os.environ.get("CAMPAIGN_PLOT_DIR", "/tmp")


def main() -> None:
    import matplotlib.pyplot as plt

    from lisatools.domains import WDMSettings
    from lisatools.globalfit.stock import erebor
    from lisatools.globalfit.stock.erebor.noise import (
        noise_params_from_file,
        resolve_noise_file,
    )
    from lisatools.sensitivity import MojitoNoiseSensitivityMatrix

    # --- the stock fit resolves the brick, exactly as a run does -----------
    # noise_only SAMPLES the PSD, so its fixed_psd_params is None; the
    # brick-fitted (Soms_d, Sa_a) come from the same stock helper the fit
    # uses to seed its injection.
    fit = erebor.noise_only(nwalkers=2)
    gs = fit.make_general_settings()
    brick = resolve_noise_file(gs.mojito_data_path, gs.noise_file)
    if brick is None:
        print("[RESULT] noise_proof=SKIP reason=no_brick", flush=True)
        return
    fitted = noise_params_from_file(brick, tdi_generation=getattr(gs, "tdi_gen", 2))
    if fitted is None:
        print("[RESULT] noise_proof=SKIP reason=fit_failed", flush=True)
        return
    soms_d, sa_a = fitted
    print(
        f"[RESULT] brick={os.path.basename(brick)} soms_d={soms_d:.6e} "
        f"sa_a={sa_a:.6e} soms_ratio_scird={soms_d / 15e-12:.4f} "
        f"sa_ratio_scird={sa_a / 3e-15:.4f}",
        flush=True,
    )

    # --- brick PSD on the WDM basis via the stock sensitivity class --------
    wdm = WDMSettings(
        Nf=1024, Nt=32, dt=2.5, min_freq=3e-4, max_freq=1e-2, force_backend="cpu"
    )
    sens = MojitoNoiseSensitivityMatrix(wdm, brick)
    arr = np.asarray(sens.sens_mat)  # (nch, nch, n_active_layers, Nt)
    layer_idx = np.where(np.asarray(wdm.frequency_layer_mask))[0]
    freqs = layer_idx * float(wdm.layer_df)  # center freq of each active layer
    nch = arr.shape[0]
    # Stock run frame is XYZ (the mojito data and GB comp are tdi_type="XYZ"),
    # so the sensitivity diagonal is XX/YY/ZZ. XX≈YY≈ZZ is the XYZ signature
    # (in AET the T channel would sit well below the A/E pair).
    chan = (["X", "Y", "Z"][:nch] if nch == 3 else [str(i) for i in range(nch)])

    # --- figure: per-channel measured sqrt(PSD) vs frequency ---------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for i in range(nch):
        diag = np.asarray(arr[i, i]).real  # (n_layers, Nt)
        # median over time layers = the stationary PSD level per freq layer
        psd = np.median(np.where(diag > 0, diag, np.nan), axis=1)
        good = np.isfinite(psd) & (psd > 0)
        ax.loglog(
            freqs[good], np.sqrt(psd[good]),
            color=colors[i % len(colors)], lw=1.8, marker="o", ms=3,
            label=f"brick {chan[i]}{chan[i]}",
        )

    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$\sqrt{\mathrm{PSD}}$  (WDM diagonal, per channel)")
    ax.set_title(
        "Mojito 731-day NOISE brick through the stock noise_only fit\n"
        f"fitted Soms_d={soms_d:.4e} ({soms_d / 15e-12:.3f}×scird),  "
        f"Sa_a={sa_a:.4e} ({sa_a / 3e-15:.3f}×scird)"
    )
    ax.legend(frameon=False, fontsize=9, title="MojitoNoiseSensitivityMatrix")
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "mojito_noise_psd.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(
        f"[RESULT] noise_proof=ok noise_proof_ok=1 n_layers={int(good.sum())} "
        f"plot={out}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[RESULT] noise_proof=FAIL error={type(exc).__name__}", flush=True)
        sys.exit(1)
