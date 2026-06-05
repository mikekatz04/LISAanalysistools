#!/usr/bin/env python
"""Plot the heterodyne ratio r(t) = c1/c0 at one active m-layer.

Builds dense complex-WDM coefficients for a reference x0 and a candidate
``x_cand = x0 + Delta f0`` via lisatools, takes r = c1/c0 at the
m_floor layer, and plots:

  - Re(r), Im(r) vs n
  - |r|, arg(r) vs n
  - Sparse samples + linear-interp reconstruction at chosen Nt_layers

for multiple ``DF0_FRAC`` values (Delta f0 / layer_df) so you can see how
smooth r actually is and where the linear interp starts to fail.

Run::
    python gb_signal_het_wdm_v2_r_plot.py
Env vars:
    DF0_FRACS    comma-list of Delta_f0/layer_df values to plot
                 (default "0.001,0.005,0.01,0.05")
    NT_LAYERS    Nt_layer values to overlay sparse+interp (default "16,64,256")
    OUT_PNG      output PNG path (default ./v2_r_t_plot.png)
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


def main():
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nt * Nf * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(
        t_tdi, Tobs, t_start, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = np.ones(Nobs)
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=True, force_backend=backend,
    )
    layer_df = wdm_set_complex.layer_df
    ind_min_f = wdm_set_complex.ind_min_f
    ind_min_t = wdm_set_complex.ind_min_t
    Nt_active = wdm_set_complex.Nt_active
    print(f"[grid] layer_df={layer_df:.3e}Hz  layer_dt={Nf*dt:.3e}s  "
          f"ind_min_t={ind_min_t}  Nt_active={Nt_active}", flush=True)

    # Reference x0: mid-band, deterministic
    m_ref = int(3e-3 / layer_df)
    x0 = np.array([
        1e-22,                         # amp
        (m_ref + 0.5) * layer_df,      # f0
        1e-17,                         # fdot
        0.0,                           # fddot
        2.098,                         # phi0
        0.24,                          # inc (radians)
        1.234,                         # psi
        4.098,                         # lam
        0.04,                          # beta
    ], dtype=float)
    print(f"[x0] f0={x0[1]*1e3:.5f}mHz  m_floor={int(x0[1]/layer_df)}", flush=True)

    def real_td(p):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = p
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]

    print("[gen] TD + WDM at x0 ...", flush=True)
    td0 = real_td(x0)
    c0 = np.asarray(TDSignal(td0, settings=td_set).transform(
        wdm_set_complex, window=window).arr)            # (3, Nf_act, Nt_act) complex
    print(f"   c0 shape={c0.shape}  max|c0|={np.abs(c0).max():.3e}", flush=True)

    # Active m_local
    m_floor = int(x0[1] / layer_df)
    m_active = np.arange(m_floor - 2, m_floor + 3)
    m_local = m_active - ind_min_f
    print(f"[active m] global={m_active}  local={m_local}", flush=True)

    df_fracs_env = os.environ.get("DF0_FRACS", "0.001,0.005,0.01,0.05")
    df_fracs = [float(s) for s in df_fracs_env.split(",")]
    nt_layers_env = os.environ.get("NT_LAYERS", "16,64,256")
    nt_layers = [int(s) for s in nt_layers_env.split(",")]
    print(f"[scan] DF0_FRACS={df_fracs}  NT_LAYERS={nt_layers}", flush=True)

    n_dense = np.arange(Nt_active)
    # Mask: where c0 amplitude is non-negligible (avoid divide-by-noise)
    c0_active = c0[:, m_local, :]                       # (3, 5, Nt_act)
    # Plot for one channel, the m=m_floor row (m_local[2])
    CHAN = 0
    M_ROW_LOCAL = 2                                     # m_local[2] = m_floor
    c0_row = c0_active[CHAN, M_ROW_LOCAL]               # (Nt_act,)
    c0_mag = np.abs(c0_row)
    floor = 1e-12 * c0_mag.max()
    mask = c0_mag > floor
    print(f"   |c0[m_floor]| max={c0_mag.max():.3e}  mask coverage={mask.sum()}/{Nt_active}",
          flush=True)

    # absolute time at each WDM pixel (s, relative to t_start)
    t_n_dense = (ind_min_t + n_dense) * Nf * dt          # (Nt_active,)

    fig, axs = plt.subplots(
        len(df_fracs), 5, figsize=(22, 4.0 * len(df_fracs)), sharex=True,
    )
    if len(df_fracs) == 1:
        axs = axs[None, :]

    for di, dff in enumerate(df_fracs):
        delta = dff * layer_df
        x_cand = x0.copy()
        x_cand[1] = x0[1] + delta
        td_cand = real_td(x_cand)
        c1 = np.asarray(TDSignal(td_cand, settings=td_set).transform(
            wdm_set_complex, window=window).arr)
        c1_row = c1[CHAN, m_local[M_ROW_LOCAL]]         # (Nt_act,)
        r = np.where(mask, c1_row / np.where(mask, c0_row, 1.0), 0.0 + 0.0j)

        # Carrier de-rotation: predicted phase from known params delta.
        # dphi(t) = 2 pi Delta_f0 * t + pi Delta_fdot * t^2  (analytic GB phase
        # diff). With only df0 here, the fdot term is zero; we still wire it.
        delta_f0 = x_cand[1] - x0[1]
        delta_fdot = x_cand[2] - x0[2]
        phase_pred = (
            2.0 * np.pi * delta_f0 * t_n_dense
            + np.pi * delta_fdot * t_n_dense ** 2
        )
        r_demod = r * np.exp(-1j * phase_pred)

        # ax 0: Re(r), Im(r) (raw)
        ax = axs[di, 0]
        ax.plot(n_dense, r.real, "C0-", label="Re(r)", linewidth=0.8)
        ax.plot(n_dense, r.imag, "C1-", label="Im(r)", linewidth=0.8)
        ax.set_title(f"r(t) RAW components  (Df0/layer_df={dff})")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)

        # ax 1: Re(r_demod), Im(r_demod)
        ax = axs[di, 1]
        ax.plot(n_dense, r_demod.real, "C0-", label="Re(r_demod)", linewidth=0.8)
        ax.plot(n_dense, r_demod.imag, "C1-", label="Im(r_demod)", linewidth=0.8)
        ax.set_title("r DE-ROTATED  (carrier removed)")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(-1.2, 1.2)

        # ax 2: |r| (essentially unchanged by de-rotation since |exp(...)|=1)
        ax = axs[di, 2]
        ax.plot(n_dense[mask], np.abs(r[mask]), "C2-", linewidth=1)
        ax.set_title("|r(t)|")
        ax.set_ylim(0.5, 1.5)
        ax.grid(alpha=0.3)

        # ax 3: arg(r) RAW vs DE-ROTATED (unwrapped)
        ax = axs[di, 3]
        ax.plot(n_dense[mask], np.unwrap(np.angle(r[mask])), "C3-",
                label="arg(r) raw", linewidth=0.8)
        ax.plot(n_dense[mask], np.unwrap(np.angle(r_demod[mask])), "C4-",
                label="arg(r_demod)", linewidth=0.8)
        ax.set_title("unwrap arg  (rad)")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
        dphi_pred = 2.0 * np.pi * delta * (Nf * dt)
        ax.text(0.02, 0.05,
                f"theory: dphi/dn = {dphi_pred:.2e} rad/pixel",
                transform=ax.transAxes, fontsize=8, va="bottom")

        # ax 4: linear-interp error -- raw vs de-rotated, several stride values
        ax = axs[di, 4]
        print(f"\n[interp err]  Df0/layer_df={dff}", flush=True)
        print(f"   {'Nt_layer':>8s} {'stride':>6s} {'max|err_raw|':>13s} {'max|err_demod|':>15s}",
              flush=True)
        for ntl in nt_layers:
            if Nt % ntl != 0:
                continue
            stride = Nt // ntl
            N_sparse = Nt_active // stride
            n_sparse_local = stride // 2 + np.arange(N_sparse) * stride
            b_idx = np.searchsorted(n_sparse_local, n_dense, side="right") - 1
            b_idx = np.clip(b_idx, 0, N_sparse - 2)
            offsets = (n_dense - n_sparse_local[b_idx]) / float(stride)

            # Raw r interpolation
            r_sparse_raw = r[n_sparse_local]
            r_interp_raw = (r_sparse_raw[b_idx] * (1.0 - offsets)
                            + r_sparse_raw[b_idx + 1] * offsets)
            err_raw = np.abs(r_interp_raw - r)

            # De-rotated r interpolation (interpolate r_demod, then re-rotate)
            r_sparse_demod = r_demod[n_sparse_local]
            r_interp_demod = (r_sparse_demod[b_idx] * (1.0 - offsets)
                              + r_sparse_demod[b_idx + 1] * offsets)
            r_interp_rerotated = r_interp_demod * np.exp(1j * phase_pred)
            err_demod = np.abs(r_interp_rerotated - r)

            print(f"   {ntl:8d} {stride:6d} {err_raw.max():13.3e} {err_demod.max():15.3e}",
                  flush=True)
            ax.plot(n_dense, np.log10(np.maximum(err_raw, 1e-20)),
                    "--", label=f"raw Nt={ntl} max={err_raw.max():.1e}", linewidth=0.8)
            ax.plot(n_dense, np.log10(np.maximum(err_demod, 1e-20)),
                    "-", label=f"demod Nt={ntl} max={err_demod.max():.1e}", linewidth=0.8)
        ax.set_title("log10 |r_interp - r_true|\nsolid = de-rotated, dashed = raw")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=6, ncol=1)
        ax.set_ylim(-16, 1)

        for ax in axs[di]:
            ax.set_xlabel("n (active)")

    fig.suptitle(
        f"GB signal-het:  r(t) = c1(x0+Df0*layer_df)/c0(x0)  at m_floor={m_floor}, "
        f"channel {CHAN}.  De-rotation removes the analytic 2*pi*Df0*t carrier "
        f"phase.\n"
        f"x0: amp={x0[0]:.1e}, f0={x0[1]*1e3:.3f}mHz, fdot={x0[2]:.1e}, beta={x0[8]:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.environ.get("OUT_PNG", "v2_r_t_plot.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {out}", flush=True)
    print("DONE.")


if __name__ == "__main__":
    sys.exit(main())
