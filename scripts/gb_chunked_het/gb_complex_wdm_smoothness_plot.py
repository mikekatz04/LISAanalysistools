#!/usr/bin/env python
"""
Visualize the smoothness of the complex/quadrature WDM representation
of a single galactic binary, over both axes (m, n).

For h0 = h(x0):
  - Real WDM     w0[c, m, n]            (real-valued; carries Wilson (m+n) parity sign-alternation)
  - Complex WDM  c0[c, m, n] = w0_real + i * w0_quad  (smooth: parity rotated out)

The two representations carry the same information; the script
shows it side-by-side so the parity pattern in real WDM (jagged
sign flips across n) and the smooth amplitude+phase in complex WDM
(slowly drifting modulus, gracefully winding phase) are both
visible.

Outputs:
  gb_complex_wdm_smoothness_heatmap_chN.png    -- 2D heatmaps over (m, n)
  gb_complex_wdm_smoothness_slices_chN.png     -- 1D slices vs n at m_peak +/-1

Run under the ``deving`` conda env::
    conda activate deving
    python gb_complex_wdm_smoothness_plot.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly


def build_x0(layer_df):
    m_ref = int(3e-3 / layer_df)
    _m_offset = int(os.environ.get("M_OFFSET", "0"))
    _f_frac = float(os.environ.get("F_FRAC", "0.5"))
    return np.array([
        1.0e-22,
        (m_ref + _m_offset + _f_frac) * layer_df,
        float(os.environ.get("SOURCE_FDOT", "1e-17")),
        0.0,
        2.09802430298,
        0.23984234,
        1.234019814,
        4.09808143,
        float(os.environ.get("BETA", "0.04")),
    ], dtype=float)


def gen_real_td(gb_gen, params9, t_arr):
    amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = params9
    spline = gb_gen(
        np.array([amp]), np.array([f0]), np.array([fdot]),
        np.array([fddot]), np.array([phi0]), np.array([inc]),
        np.array([psi]), np.array([lam]), np.array([beta]),
        convert_to_ra_dec=False, return_spline=True,
    )
    return np.asarray(spline.eval_tdi(t_arr))[0]


def find_source_m_peak(w0_real):
    mag = np.abs(w0_real).sum(axis=(0, -1))   # sum over channels, time
    return int(np.argmax(mag))


def heatmap_panel(ax, arr, title, *, vmin=None, vmax=None, cmap="viridis",
                  extent=None):
    im = ax.imshow(arr, origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("n (time pixel, active band)")
    ax.set_ylabel("m (layer, active band)")


def make_heatmap_figure(channel, w0_real, c0_complex, m_peak, plot_dir):
    """6-panel heatmap: Re/Im/|.|/arg for real and complex WDM in source m-band."""
    Nf_act, Nt_active = w0_real.shape[1], w0_real.shape[2]
    m_lo = max(m_peak - 8, 0)
    m_hi = min(m_peak + 8 + 1, Nf_act)
    band_extent = [0, Nt_active, m_lo, m_hi]

    wr = w0_real[channel, m_lo:m_hi, :]               # (M_band, Nt_active) real
    cr = c0_complex[channel, m_lo:m_hi, :]            # (M_band, Nt_active) complex

    # symmetric range for signed plots
    wr_max = float(np.max(np.abs(wr)))
    cr_re_max = float(np.max(np.abs(cr.real)))
    cr_im_max = float(np.max(np.abs(cr.imag)))
    sign_max = max(wr_max, cr_re_max, cr_im_max, 1e-300)

    fig, axs = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)

    # row 0: REAL WDM
    heatmap_panel(axs[0, 0], wr,
                  f"REAL WDM  w0[c={channel}, m, n]  (signed)",
                  vmin=-sign_max, vmax=+sign_max, cmap="seismic",
                  extent=band_extent)
    heatmap_panel(axs[0, 1], np.abs(wr),
                  f"REAL WDM  |w0[c={channel}, m, n]|",
                  vmin=0, vmax=sign_max, cmap="viridis",
                  extent=band_extent)
    # arg of real signal is 0 / pi only; show sign(w0) instead for clarity
    heatmap_panel(axs[0, 2], np.sign(wr),
                  f"REAL WDM  sign(w0[c={channel}, m, n])",
                  vmin=-1, vmax=+1, cmap="seismic",
                  extent=band_extent)

    # row 1: COMPLEX WDM
    heatmap_panel(axs[1, 0], cr.real,
                  f"COMPLEX WDM  Re(c0[c={channel}, m, n])",
                  vmin=-sign_max, vmax=+sign_max, cmap="seismic",
                  extent=band_extent)
    heatmap_panel(axs[1, 1], np.abs(cr),
                  f"COMPLEX WDM  |c0[c={channel}, m, n]|",
                  vmin=0, vmax=sign_max * np.sqrt(2), cmap="viridis",
                  extent=band_extent)
    # phase masked where amplitude is tiny
    arg_cr = np.where(np.abs(cr) > 1e-3 * np.abs(cr).max(),
                      np.angle(cr), np.nan)
    heatmap_panel(axs[1, 2], arg_cr,
                  f"COMPLEX WDM  arg(c0[c={channel}, m, n])",
                  vmin=-np.pi, vmax=+np.pi, cmap="twilight",
                  extent=band_extent)

    for ax in axs.ravel():
        ax.axhline(m_peak + 0.5, color="black", ls="--", lw=0.5)

    fig.suptitle(
        f"Single-binary WDM smoothness  channel {channel}  "
        f"m_peak (active idx) = {m_peak}",
        fontsize=11,
    )
    out = os.path.join(
        plot_dir, f"gb_complex_wdm_smoothness_heatmap_ch{channel}.png"
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def make_slice_figure(channel, w0_real, c0_complex, m_peak, plot_dir):
    """Two-panel-per-m comparison: real WDM (signed) vs complex WDM (|.| and arg)."""
    Nf_act, Nt_active = w0_real.shape[1], w0_real.shape[2]
    m_list = [m for m in (m_peak - 1, m_peak, m_peak + 1)
              if 0 <= m < Nf_act]
    n_grid = np.arange(Nt_active)

    fig, axs = plt.subplots(len(m_list), 3, figsize=(15, 3.0 * len(m_list)),
                            constrained_layout=True, squeeze=False)
    for i, m in enumerate(m_list):
        wr = w0_real[channel, m]
        cr = c0_complex[channel, m]

        axs[i, 0].plot(n_grid, wr, lw=0.5, color="tab:blue")
        axs[i, 0].set_title(f"REAL WDM  w0[c={channel}, m={m}, n]  (signed)",
                            fontsize=10)
        axs[i, 0].axhline(0, color="black", lw=0.3)
        axs[i, 0].set_xlabel("n")
        axs[i, 0].set_ylabel("w0")
        axs[i, 0].grid(alpha=0.3)

        axs[i, 1].plot(n_grid, np.abs(wr), lw=0.5, color="tab:blue",
                       label="|REAL WDM|")
        axs[i, 1].plot(n_grid, np.abs(cr), lw=0.8, color="tab:red",
                       label="|COMPLEX WDM|")
        axs[i, 1].set_title(f"|w0| vs |c0|  m={m}", fontsize=10)
        axs[i, 1].set_xlabel("n")
        axs[i, 1].set_ylabel("|.|")
        axs[i, 1].grid(alpha=0.3)
        axs[i, 1].legend(fontsize=8)

        arg_cr = np.where(np.abs(cr) > 1e-3 * np.abs(cr).max(),
                          np.angle(cr), np.nan)
        axs[i, 2].plot(n_grid, arg_cr, lw=0.5, color="tab:green")
        axs[i, 2].set_title(f"arg(c0)  m={m}", fontsize=10)
        axs[i, 2].set_xlabel("n")
        axs[i, 2].set_ylabel("arg [rad]")
        axs[i, 2].set_ylim(-np.pi, np.pi)
        axs[i, 2].grid(alpha=0.3)

    fig.suptitle(
        f"Single-binary WDM smoothness  channel {channel}  "
        f"per-layer slices (m_peak={m_peak})",
        fontsize=11,
    )
    out = os.path.join(
        plot_dir, f"gb_complex_wdm_smoothness_slices_ch{channel}.png"
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    backend = "cpu"
    plot_dir = os.environ.get("PLOT_DIR", ".")
    os.makedirs(plot_dir, exist_ok=True)

    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start

    N_inj = 16384
    gb_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = np.linspace(t_arr[0], t_arr[-1], N_inj)
    gb_gen = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs,
    )

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = np.ones(Nobs)
    min_freq = 1e-4
    max_freq = 35.0e-3
    _EDGE_CUT = int(os.environ.get("EDGE_CUT", "20"))
    min_time = _EDGE_CUT * wavelet_duration
    max_time = (Nt - _EDGE_CUT) * wavelet_duration

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=True, force_backend=backend,
    )

    layer_df = wdm_set_real.layer_df
    print(f"[setup] Nf={Nf} Nt={Nt}  layer_df={layer_df:.3e} Hz  "
          f"layer_dt={wdm_set_real.layer_dt:.3e} s", flush=True)

    x0 = build_x0(layer_df)
    print(f"[x0] amp={x0[0]:.3e} f0={x0[1]*1e3:.5f} mHz  fdot={x0[2]:.2e}  "
          f"beta={x0[8]:.3f} lam={x0[7]:.3f}", flush=True)

    h_real_td = gen_real_td(gb_gen, x0, t_arr)
    print("[step] transforming to REAL WDM and COMPLEX WDM ...", flush=True)
    w0 = np.asarray(
        TDSignal(h_real_td, settings=td_set).transform(
            wdm_set_real, window=window
        ).arr
    )
    c0 = np.asarray(
        TDSignal(h_real_td, settings=td_set).transform(
            wdm_set_complex, window=window
        ).arr
    )
    print(f"  REAL WDM    shape={w0.shape}  dtype={w0.dtype}", flush=True)
    print(f"  COMPLEX WDM shape={c0.shape}  dtype={c0.dtype}", flush=True)

    sanity_re = float(np.max(np.abs(c0.real - w0)))
    print(f"[sanity] max |Re(c0) - w0_real|  = {sanity_re:.3e}", flush=True)

    m_peak = find_source_m_peak(w0)
    print(f"[step] source m-peak at active index {m_peak} "
          f"(global m = {m_peak + wdm_set_real.ind_min_f})", flush=True)

    for channel in (0, 1):
        print(f"[plot] channel {channel}:", flush=True)
        make_heatmap_figure(channel, w0, c0, m_peak, plot_dir)
        make_slice_figure(channel, w0, c0, m_peak, plot_dir)

    print("\nDONE.")


if __name__ == "__main__":
    sys.exit(main())
