#!/usr/bin/env python
"""Show signal-het accuracy degradation vs distance from reference params.

Signal-het builds its likelihood around a single reference point
(``params_ref_all`` -- here, the injection). The bin-fold pipeline is
exact at the reference and gets less accurate as the candidate moves
away. The ``max_r`` clip catches catastrophic departures but a smooth
moderate-distance degradation is expected.

Chunked-het, by contrast, re-evaluates the per-chunk FD waveform from
scratch for every candidate -- no reference -- so it stays accurate
across the prior. We use it as ground truth here.

This script sweeps each of the 8 sampled-basis parameters one at a
time around the injection and plots ``|logL_signalhet - logL_chunkedhet|``
vs the offset along that axis. The resulting curves show where
signal-het is reliable and where its bin-fold approximation breaks down.

Run::
    BACKEND=cpu  python show_signalhet_accuracy_vs_distance.py
    BACKEND=cuda12x  python show_signalhet_accuracy_vs_distance.py

Env vars (most knobs forwarded to ``build_pack`` in
``compare_signalhet_vs_chunked_mcmc.py``):

  SEED           42       random seed
  F0_MHZ         14.22    carrier in mHz
  SNR_TARGET     50.0
  N_PER_AXIS     21       sweep points per axis
  SWEEP_MAX_F0_FRAC      0.3   |Df0| / layer_df max
  SWEEP_MAX_AMP_FRAC     0.5   |Damp| / amp max
  SWEEP_MAX_ANG          1.5   max offset on angular params (rad)
  OUT_PNG        signalhet_accuracy_vs_distance.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_signalhet_vs_chunked_mcmc import build_pack

from eryn.utils import TransformContainer


def main() -> None:
    SEED          = int(os.environ.get("SEED", "42"))
    F0_MHZ        = float(os.environ.get("F0_MHZ", "14.22"))
    SNR_TARGET    = float(os.environ.get("SNR_TARGET", "50.0"))
    N_PER_AXIS    = int(os.environ.get("N_PER_AXIS", "21"))
    SWEEP_MAX_F0_FRAC  = float(os.environ.get("SWEEP_MAX_F0_FRAC",  "0.3"))
    SWEEP_MAX_AMP_FRAC = float(os.environ.get("SWEEP_MAX_AMP_FRAC", "0.5"))
    SWEEP_MAX_ANG      = float(os.environ.get("SWEEP_MAX_ANG",      "1.5"))
    OUT_PNG       = os.environ.get("OUT_PNG", "signalhet_accuracy_vs_distance.png")
    BACKEND_NAME  = os.environ.get("BACKEND", "cpu")

    # Build the kernel pack with the SAME defaults the compare script
    # uses post b3f8f42 (N_cp_sig=0, N_cp_orbit=0, tukey_alpha=0.0). That
    # makes chunked-het the lisatools-direct-equivalent ground truth.
    pack = build_pack(backend_name=BACKEND_NAME, f0_mhz=F0_MHZ,
                      snr_target=SNR_TARGET, seed=SEED, verbose=True)

    logl_sh = pack.logl_signalhet
    logl_ch = pack.logl_chunkedhet
    layer_df = pack.layer_df

    amp_inj  = pack.amp_inj
    f0_inj   = pack.params_inj[1]
    fdot_inj = pack.params_inj[2]
    phi0_inj = pack.params_inj[4]
    inc_inj  = pack.params_inj[5]
    psi_inj  = pack.params_inj[6]
    lam_inj  = pack.params_inj[7]
    beta_inj = pack.params_inj[8]

    full_basis    = ["amp", "f0", "fdot0", "fddot0", "phi0", "inc", "psi", "lam", "beta"]
    sampled_basis = ["amp", "f0", "fdot0", "phi0", "cosinc", "psi", "lam", "sinbeta"]
    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms={"cosinc": np.arccos, "sinbeta": np.arcsin},
        fill_dict={"fddot0": 0.0},
        key_map={"cosinc": "inc", "sinbeta": "beta"},
    )

    inj = np.array([
        amp_inj, f0_inj, fdot_inj, phi0_inj,
        np.cos(inc_inj), psi_inj, lam_inj, np.sin(beta_inj),
    ], dtype=float)

    # Per-axis offset grids (centred on injection). Tuned so the
    # excursions land inside the priors used by the compare-script MCMC.
    half = (N_PER_AXIS - 1) // 2
    sym_lin = lambda lo, hi: np.concatenate([
        np.linspace(lo, 0.0, half + 1)[:-1], [0.0],
        np.linspace(0.0, hi, half + 1)[1:],
    ])
    axes = [
        ("amp",     amp_inj  * SWEEP_MAX_AMP_FRAC, "amp [SNR units]"),
        ("f0",      layer_df * SWEEP_MAX_F0_FRAC,  "Df0 / layer_df"),
        ("fdot0",   1e-15,                          "Dfdot [Hz/s]"),
        ("phi0",    SWEEP_MAX_ANG,                  "Dphi0 [rad]"),
        ("cosinc",  min(SWEEP_MAX_ANG, 1.0),        "D(cos inc)"),
        ("psi",     SWEEP_MAX_ANG,                  "Dpsi [rad]"),
        ("lam",     SWEEP_MAX_ANG,                  "Dlam [rad]"),
        ("sinbeta", min(SWEEP_MAX_ANG, 1.0),        "D(sin beta)"),
    ]

    # Pre-allocate output curves.
    sweep_offsets = {name: sym_lin(-w, +w) for name, w, _ in axes}
    sweep_diffs   = {name: np.full(N_PER_AXIS, np.nan) for name, _, _ in axes}
    sweep_sh      = {name: np.full(N_PER_AXIS, np.nan) for name, _, _ in axes}
    sweep_ch      = {name: np.full(N_PER_AXIS, np.nan) for name, _, _ in axes}

    for k, (name, _, _) in enumerate(axes):
        offs = sweep_offsets[name]
        # Build the (N_PER_AXIS, ndim) candidate matrix.
        x = np.tile(inj[None, :], (N_PER_AXIS, 1))
        x[:, k] = inj[k] + offs
        # Clip cosinc / sinbeta to [-1, 1] -- arccos / arcsin would NaN.
        if name in ("cosinc", "sinbeta"):
            x[:, k] = np.clip(x[:, k], -0.999999, 0.999999)
        ll_s = np.asarray(logl_sh(x, transform_fn=tc))
        ll_c = np.asarray(logl_ch(x, transform_fn=tc))
        sweep_sh[name]    = ll_s
        sweep_ch[name]    = ll_c
        sweep_diffs[name] = ll_s - ll_c
        # Also report the chunked-het ll value at the sweep endpoints +
        # at the reference so the user can see how deep the likelihood
        # drops as the parameter moves away (which is the regime where
        # signal-het accuracy degrades).
        ll_c_at_inj = ll_c[N_PER_AXIS // 2]
        print(f"[axis {k} = {name:7s}]  "
              f"|diff| median={np.nanmedian(np.abs(ll_s - ll_c)):.3e}  "
              f"max={np.nanmax(np.abs(ll_s - ll_c)):.3e}  "
              f"|  ll_ch @ref={ll_c_at_inj:+.3e}  "
              f"range=[{np.nanmin(ll_c):+.3e}, {np.nanmax(ll_c):+.3e}]",
              flush=True)

    # Make the plot: 2x4 grid, one panel per axis. Left y-axis is the
    # signal-het / chunked-het diff (log); right twin y-axis is the
    # absolute chunked-het ll value (linear) so the user can see where
    # the diff sits relative to how deep the likelihood drops along
    # that axis.
    fig, axarr = plt.subplots(2, 4, figsize=(17, 8), sharey=False)
    for ax, (name, _, xlabel) in zip(axarr.flat, axes):
        offs = sweep_offsets[name]
        diff = sweep_diffs[name]
        ll_c = sweep_ch[name]
        ll_s = sweep_sh[name]
        # X axis: normalize where it makes sense.
        if name == "amp":
            x_disp = offs / amp_inj
            xlabel = "Damp / amp_inj"
        elif name == "f0":
            x_disp = offs / layer_df
            xlabel = "Df0 / layer_df"
        else:
            x_disp = offs

        # Left axis: |diff| on log scale.
        line_diff, = ax.plot(x_disp, np.abs(diff), "o-",
                             color="C0", lw=1.0, ms=3, label="|diff|")
        ax.axvline(0.0, color="r", alpha=0.5, lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|logL_sh - logL_ch|", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

        # Right twin axis: chunked-het logL (linear scale).
        ax2 = ax.twinx()
        line_llc, = ax2.plot(x_disp, ll_c, "-", color="C1",
                             lw=0.9, alpha=0.85, label="logL_chunked")
        line_lls, = ax2.plot(x_disp, ll_s, "--", color="C3",
                             lw=0.8, alpha=0.6, label="logL_signal")
        ax2.set_ylabel("logL value", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        # Only the first panel gets the legend (keeps the figure clean).
        if name == "amp":
            ax.legend([line_diff, line_llc, line_lls],
                      ["|diff|", "logL_chunked", "logL_signal"],
                      loc="upper right", fontsize=8)
    fig.suptitle(
        f"signal-het accuracy vs distance from reference  "
        f"(SNR={SNR_TARGET:.0f}, f0={F0_MHZ}mHz)\n"
        f"chunked-het is ground truth (matches lisatools direct); "
        f"orange = absolute chunked-het logL, dashed red = signal-het logL, "
        f"blue = |diff|",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(OUT_PNG, dpi=110)
    print(f"\n[ok] wrote {OUT_PNG}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
