#!/usr/bin/env python
"""
Step 1 of signal-based WDM heterodyne: visualize whether r(t) = h1/h0
is smooth when h0 and h1 are the COMPLEX TD ENVELOPES that
``GBTDIonTheFly`` builds internally, vs the JAGGED ratio you'd get in
the standard real WDM pixel-wise (Wilson sign-alternation).

The complex TD envelope is

    h_c_complex(t) = tdi_amp_c(t) * exp(-1j * (tdi_phase_c(t)
                                                + phase_ref(t)))

per channel c. ``GBTDIonTheFly.__call__(..., return_spline=True)``
returns a ``TDTDIOutput`` whose ``eval_spline_vals(t)`` gives back
``(tdi_amp, tdi_phase, phase_ref)`` cubic splines evaluated at the
caller-chosen times. The actual real TDI signal is just
``Re(h_c_complex)`` -- that's what ``eval_tdi`` returns.

For parameter-close (x0, x1), the complex envelopes share the same
slow amplitude and only differ in a slowly-varying phase increment,
so r(t) = h1_complex / h0_complex is smooth. Sampled at WDM column-
centre times ``t_n = (n + 0.5) * layer_dt + t_start`` (or any sparse
sub-set of those), this r(t_n) is the "smooth ratio" the heterodyne
will multiply into the standard real-WDM coefficients.

This script:
  1. builds h0, h1 splines for several parameter perturbations,
  2. evaluates the complex TD envelope on a dense grid + on the WDM
     column-centre times,
  3. computes r(t),
  4. ALSO does the standard real WDM transform of h0, h1 and takes
     the pixel-wise ratio w1[m,n] / w0[m,n],
  5. saves heatmaps + 1D slices comparing r_TD_complex(t_n) (smooth)
     to w1[m_peak, n] / w0[m_peak, n] (jagged).

Run under the ``deving`` conda env:
    conda activate deving
    python gb_ratio_smoothness_test.py

Env vars:
  NF, NT        WDM grid (default 1460, 2560)
  M_OFFSET      shift source m_floor relative to ~3 mHz (default 0)
  F_FRAC        sub-layer offset of f0 (default 0.5 = mid-layer)
  EDGE_CUT      WDM time-edge trim in wavelets (default 20)
  PLOT_DIR      directory for output PNGs (default ".")
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

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly


# Parameter perturbations to scan. (label, param_index, delta_callable(x0))
# param order: amp, f0, fdot, fddot, phi0, inc, psi, lam, beta
PERTURBATIONS = [
    ("df0_p1e-6",   1, lambda x0: x0[1] * 1e-6),
    ("df0_p1e-4",   1, lambda x0: x0[1] * 1e-4),
    ("df0_p1e-2",   1, lambda x0: x0[1] * 1e-2),
    ("damp_p1e-4",  0, lambda x0: x0[0] * 1e-4),
    ("damp_p1e-2",  0, lambda x0: x0[0] * 1e-2),
    ("dfdot_p10",   2, lambda x0: max(abs(x0[2]), 1e-18) * 10.0),
    ("dphi0_p1e-3", 4, lambda x0: 1e-3),
    ("dbeta_p1e-3", 8, lambda x0: 1e-3),
    ("dlam_p1e-3",  7, lambda x0: 1e-3),
]


def build_x0(layer_df):
    """Reference GB params -- mid-band layer, mid-pixel f_frac."""
    m_ref = int(3e-3 / layer_df)
    _m_offset = int(os.environ.get("M_OFFSET", "0"))
    _f_frac = float(os.environ.get("F_FRAC", "0.5"))
    return np.array([
        1.0e-22,                                              # amp
        (m_ref + _m_offset + _f_frac) * layer_df,             # f0
        float(os.environ.get("SOURCE_FDOT", "1e-17")),        # fdot
        0.0,                                                   # fddot
        2.09802430298,                                         # phi0
        0.23984234,                                            # inc
        1.234019814,                                           # psi
        4.09808143,                                            # lam
        float(os.environ.get("BETA", "0.04")),                # beta
    ], dtype=float)


def build_spline(params9, gb_gen):
    """Call GBTDIonTheFly to get a (TDTDIOutput) spline holder."""
    amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = params9
    return gb_gen(
        np.array([amp]), np.array([f0]), np.array([fdot]),
        np.array([fddot]), np.array([phi0]), np.array([inc]),
        np.array([psi]), np.array([lam]), np.array([beta]),
        convert_to_ra_dec=False, return_spline=True,
    )


def complex_td_envelope(spline, times):
    """Evaluate h_c_complex(t) = tdi_amp * exp(-1j*(phase + phase_ref))
    at the given times. Returns shape (3, len(times)) complex.
    """
    tdi_amp_new, tdi_phase_new, phase_ref_new = spline.eval_spline_vals(times)
    # shapes: amp/phase (num_bin, 3, len(t)); phase_ref (num_bin, len(t))
    h_complex = tdi_amp_new[0] * np.exp(
        -1j * (tdi_phase_new[0] + phase_ref_new[0][None, :])
    )
    return h_complex  # (3, len(times)) complex


def td_to_real_wdm(td_real_signal, t_arr, td_set, window, wdm_set):
    """Run h_real through the standard TD -> real-WDM transform."""
    return TDSignal(td_real_signal, settings=td_set).transform(
        wdm_set, window=window,
    )


def masked_pixel_ratio(w1, w0, floor_frac=1e-3):
    """Pixel-wise w1/w0 with a |w0|-based mask. Returns NaN where masked."""
    w0_mag = np.abs(w0)
    floor = floor_frac * w0_mag.max(axis=(-2, -1), keepdims=True)
    mask = w0_mag > floor
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(mask, w1 / np.where(mask, w0, 1.0), np.nan)
    return r


def find_source_m_floor(w0_real):
    """Return the m-layer (active-band local index) with peak |w0|."""
    mag = np.abs(w0_real).sum(axis=(0, -1))   # sum over channels, time
    return int(np.argmax(mag))


def plot_panel(label, r_TD_dense, t_arr_dense,
               r_TD_at_n, t_n_centers,
               r_wdm_real, m_peak, plot_dir):
    """Save the comparison plots for one parameter perturbation.

    Panels:
      Fig A: 1D r_TD(t) over dense t and at WDM column centres.
             Two channels (0, 1); |r| and arg(r).
      Fig B: per-m heatmap of |r_wdm_real(m,n)| (REAL WDM pixel ratio,
             channel 0), with r_TD_at_n[0] overlaid as a line.
      Fig C: 1D slices: for a few m near m_peak, plot |w1/w0|(m,n) and
             arg(w1/w0)(m,n) vs n in real WDM, overlay r_TD_at_n.
    """
    nch_dense, Nt_dense = r_TD_dense.shape
    nch_n, Nt_n = r_TD_at_n.shape
    Nf_act = r_wdm_real.shape[1]

    # ---- Fig A: r(t) smoothness in TD complex envelope ----
    fig, axs = plt.subplots(2, 2, figsize=(13, 6), constrained_layout=True)
    for ch in (0, 1):
        axs[0, ch].plot(t_arr_dense, np.abs(r_TD_dense[ch]),
                        lw=0.6, color="tab:blue", label="dense |r(t)|")
        axs[0, ch].plot(t_n_centers, np.abs(r_TD_at_n[ch]),
                        "o", ms=2, color="tab:red", label="at WDM column centres")
        axs[0, ch].set_title(f"|r_TD(t)|  channel {ch}")
        axs[0, ch].set_xlabel("t (s)")
        axs[0, ch].set_ylabel("|r|")
        axs[0, ch].grid(alpha=0.3)
        axs[0, ch].legend(fontsize=8)

        axs[1, ch].plot(t_arr_dense, np.angle(r_TD_dense[ch]),
                        lw=0.6, color="tab:blue", label="dense")
        axs[1, ch].plot(t_n_centers, np.angle(r_TD_at_n[ch]),
                        "o", ms=2, color="tab:red", label="at WDM column centres")
        axs[1, ch].set_title(f"arg r_TD(t)  channel {ch}")
        axs[1, ch].set_xlabel("t (s)")
        axs[1, ch].set_ylabel("arg(r) [rad]")
        axs[1, ch].grid(alpha=0.3)
        axs[1, ch].legend(fontsize=8)
    fig.suptitle(f"r_TD_complex(t) smoothness  [{label}]")
    out = os.path.join(plot_dir, f"gb_ratio_smoothness_{label}_TDenv.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")

    # ---- Fig B: real-WDM pixel ratio heatmap (channel 0) ----
    m_lo = max(m_peak - 8, 0)
    m_hi = min(m_peak + 8 + 1, Nf_act)
    band_extent = [0, Nt_n, m_lo, m_hi]

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    cropped = r_wdm_real[0, m_lo:m_hi, :]
    im0 = axs[0].imshow(np.abs(cropped), origin="lower", aspect="auto",
                        extent=band_extent, vmin=0, vmax=3, cmap="viridis")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("REAL WDM  |w1/w0|")
    axs[0].set_xlabel("n (time pixel)")
    axs[0].set_ylabel("m (layer, active band index)")
    axs[0].axhline(m_peak + 0.5, color="white", ls="--", lw=0.5)

    arg_cropped = np.where(np.abs(cropped) > 1e-6, np.angle(cropped), np.nan)
    im1 = axs[1].imshow(arg_cropped, origin="lower", aspect="auto",
                        extent=band_extent, vmin=-np.pi, vmax=np.pi,
                        cmap="twilight")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title("REAL WDM  arg(w1/w0)")
    axs[1].set_xlabel("n (time pixel)")
    axs[1].axhline(m_peak + 0.5, color="white", ls="--", lw=0.5)

    fig.suptitle(f"Real-WDM pixel ratio (channel 0)  [{label}]  "
                 f"m_peak (active)={m_peak}")
    out = os.path.join(plot_dir, f"gb_ratio_smoothness_{label}_realWDM.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")

    # ---- Fig C: per-m slice comparison (REAL WDM vs TD-complex envelope) ----
    m_slices = [m_peak - 1, m_peak, m_peak + 1]
    m_slices = [m for m in m_slices if 0 <= m < Nf_act]

    fig, axs = plt.subplots(len(m_slices), 2,
                            figsize=(13, 3.0 * len(m_slices)),
                            constrained_layout=True, squeeze=False)
    n_grid = np.arange(Nt_n)
    for i, m in enumerate(m_slices):
        axs[i, 0].plot(n_grid, np.abs(r_wdm_real[0, m]),
                       lw=0.7, color="tab:blue",
                       label=f"|w1/w0|  REAL WDM m={m}")
        axs[i, 0].plot(n_grid, np.abs(r_TD_at_n[0]),
                       lw=1.2, color="tab:red",
                       label="|r_TD_complex| at WDM columns")
        axs[i, 0].set_title(f"channel 0  m={m}")
        axs[i, 0].set_xlabel("n (time pixel)")
        axs[i, 0].set_ylabel("|r|")
        axs[i, 0].set_ylim(0, 3)
        axs[i, 0].grid(alpha=0.3)
        axs[i, 0].legend(fontsize=7)

        arg_wdm = np.where(np.abs(r_wdm_real[0, m]) > 1e-6,
                           np.angle(r_wdm_real[0, m]), np.nan)
        axs[i, 1].plot(n_grid, arg_wdm,
                       lw=0.7, color="tab:blue",
                       label=f"arg(w1/w0) REAL WDM m={m}")
        axs[i, 1].plot(n_grid, np.angle(r_TD_at_n[0]),
                       lw=1.2, color="tab:red",
                       label="arg(r_TD_complex) at WDM columns")
        axs[i, 1].set_title(f"channel 0  m={m}")
        axs[i, 1].set_xlabel("n (time pixel)")
        axs[i, 1].set_ylabel("arg(r) [rad]")
        axs[i, 1].set_ylim(-np.pi, np.pi)
        axs[i, 1].grid(alpha=0.3)
        axs[i, 1].legend(fontsize=7)
    fig.suptitle(f"WDM pixel ratio (jagged) vs TD complex envelope (smooth)  "
                 f"[{label}]")
    out = os.path.join(plot_dir, f"gb_ratio_smoothness_{label}_slices.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    backend = "cpu"
    plot_dir = os.environ.get("PLOT_DIR", ".")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Setup mirrors gb_chunked_test_script.py 111-205 ------------------
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))

    wavelet_duration = Nf * dt        # = layer_dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_start = int(1 / 2 * YRSID_SI / dt) * dt
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

    layer_dt = wdm_set_real.layer_dt
    layer_df = wdm_set_real.layer_df
    Nf_active = wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1
    Nt_active = wdm_set_real.Nt_active

    print(f"[setup] Nf={Nf} Nt={Nt} dt={dt}  layer_dt={layer_dt:.3e}s "
          f"layer_df={layer_df:.3e}Hz", flush=True)
    print(f"[setup] Nf_active={Nf_active}  Nt_active={Nt_active}", flush=True)

    # WDM column-centre times (active-band time slice).
    # column n has centre at t = (n_global + 0.5) * layer_dt + t0,
    # where n_global = wdm_set_real.ind_min_t + n_local
    n_global_active = (
        wdm_set_real.ind_min_t + np.arange(Nt_active)
    )
    t_n_centers = (n_global_active + 0.5) * layer_dt + t_start

    # --- Reference x0 ---
    x0 = build_x0(layer_df)
    print(f"[x0] amp={x0[0]:.3e} f0={x0[1]*1e3:.5f} mHz  fdot={x0[2]:.2e}  "
          f"beta={x0[8]:.3f}  lam={x0[7]:.3f}", flush=True)

    # --- h0: spline + complex TD envelope dense and at WDM column centres + real WDM ---
    print("[step] generating h0(x0) ...", flush=True)
    spl0 = build_spline(x0, gb_gen)
    # dense complex TD: sample on a coarse dense grid (every layer_dt/8 say)
    # to keep plots reasonable.
    dense_step = max(1, int(round(layer_dt / 8.0 / dt)))
    t_arr_dense = t_arr[::dense_step]
    h0_TD_dense = complex_td_envelope(spl0, t_arr_dense)         # (3, Td)
    h0_TD_at_n = complex_td_envelope(spl0, t_n_centers)          # (3, Nt_active)
    # Real TD signal at full t_arr (for the WDM transform):
    h0_real_td = np.real(complex_td_envelope(spl0, t_arr))       # (3, Nobs)
    # h0 in real WDM:
    h0_wdm_real_sig = td_to_real_wdm(h0_real_td, t_arr, td_set, window, wdm_set_real)
    h0_wdm = np.asarray(h0_wdm_real_sig.arr)                     # (3, Nf_act, Nt_active)
    print(f"  h0_wdm shape={h0_wdm.shape} dtype={h0_wdm.dtype}", flush=True)

    # consistency: eval_tdi vs Re of our complex envelope should match
    sanity = float(np.max(np.abs(np.real(complex_td_envelope(spl0, t_arr[:1000]))
                                  - spl0.eval_tdi(t_arr[:1000])[0])))
    print(f"[sanity] max|Re(complex_td) - eval_tdi| on first 1000 pts = {sanity:.3e}",
          flush=True)

    m_peak = find_source_m_floor(h0_wdm)
    print(f"[step] source peak at m_active={m_peak}  "
          f"(global m={m_peak + wdm_set_real.ind_min_f})", flush=True)

    # --- Sweep perturbations ---
    summary = []
    for label, idx, delta_fn in PERTURBATIONS:
        x1 = x0.copy()
        delta = float(delta_fn(x0))
        x1[idx] = x0[idx] + delta
        print(f"\n[pert {label}] idx={idx} delta={delta:+.3e}", flush=True)

        spl1 = build_spline(x1, gb_gen)
        h1_TD_dense = complex_td_envelope(spl1, t_arr_dense)
        h1_TD_at_n = complex_td_envelope(spl1, t_n_centers)
        h1_real_td = np.real(complex_td_envelope(spl1, t_arr))
        h1_wdm = np.asarray(
            td_to_real_wdm(h1_real_td, t_arr, td_set, window, wdm_set_real).arr
        )

        # TD-complex ratios (smooth):
        floor_dense = 1e-12 * np.abs(h0_TD_dense).max(axis=-1, keepdims=True)
        r_TD_dense = np.where(
            np.abs(h0_TD_dense) > floor_dense,
            h1_TD_dense / np.where(np.abs(h0_TD_dense) > floor_dense,
                                   h0_TD_dense, 1.0),
            np.nan,
        )
        floor_at_n = 1e-12 * np.abs(h0_TD_at_n).max(axis=-1, keepdims=True)
        r_TD_at_n = np.where(
            np.abs(h0_TD_at_n) > floor_at_n,
            h1_TD_at_n / np.where(np.abs(h0_TD_at_n) > floor_at_n,
                                  h0_TD_at_n, 1.0),
            np.nan,
        )

        # Real-WDM pixel ratio (jagged):
        r_wdm_real = masked_pixel_ratio(h1_wdm, h0_wdm, floor_frac=1e-3)

        # Quantitative roughness: mean |Δ| between consecutive n entries,
        # at m_peak (REAL WDM) and along the TD-complex r_at_n (channel 0).
        def step_var(arr):
            finite = np.isfinite(arr)
            v = arr[finite]
            return float(np.nanmean(np.abs(np.diff(v)))) if v.size > 4 else float("nan")

        rough_wdm_real = step_var(np.abs(r_wdm_real[0, m_peak]))
        rough_TD = step_var(np.abs(r_TD_at_n[0]))
        print(f"  roughness of |r|: REAL WDM m_peak ~ {rough_wdm_real:.3e};  "
              f"TD complex envelope at WDM columns ~ {rough_TD:.3e}", flush=True)
        summary.append((label, idx, delta, rough_wdm_real, rough_TD))

        plot_panel(label,
                   r_TD_dense, t_arr_dense,
                   r_TD_at_n, t_n_centers,
                   r_wdm_real, m_peak, plot_dir)

    # --- Summary ---
    print("\n" + "=" * 86)
    print(f"{'label':<14} {'idx':>3} {'delta':>13} "
          f"{'rough(REAL WDM)':>18} {'rough(TD complex)':>20} {'ratio R/T':>10}")
    print("-" * 86)
    for label, idx, delta, rR, rT in summary:
        ratio = rR / max(rT, 1e-30)
        print(f"{label:<14} {idx:>3} {delta:>+13.3e} "
              f"{rR:>18.3e} {rT:>20.3e} {ratio:>10.2e}")
    print("=" * 86)
    print("\nDONE. Inspect *_TDenv.png, *_realWDM.png, *_slices.png in",
          plot_dir)


if __name__ == "__main__":
    sys.exit(main())
