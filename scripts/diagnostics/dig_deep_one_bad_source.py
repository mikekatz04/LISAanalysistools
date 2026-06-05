#!/usr/bin/env python
"""For one bad-mismatch source whose f0 sits within ~v/c of a layer
boundary, walk through the Python lookup pipeline and inspect:

  1. f(t_n) at every active WDM time pixel (numerical phase derivative of
     the TDI spline) -- shows where it crosses the boundary.
  2. m_source = int(f(t_n) / layer_df) at every pixel -- shows the
     discrete jump.
  3. For each n where the kernel/Python adds a contribution to (m_layer,
     n), record the WDM coefficient value, the truth (from direct
     wdmtransform of the same TD signal), and the discrepancy.
  4. Confirm the discrepancy is concentrated at time-pixels right around
     the f(t) layer crossing.

Output: per-pixel diagnostic table + heatmap of (truth - lookup) magnitude.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly
from lisatools.domains import (
    TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal, WDMLookupTable,
)


def main():
    backend = "cpu"
    dt = 10.0
    Nf = 1460
    Nt = 256 * 10
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_ref = t_start
    t_arr_full = np.arange(Nobs) * dt + t_start

    min_freq = 0.0001
    max_freq = 35.0e-3
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration
    wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq,
                          min_time=min_time, max_time=max_time)
    td_set = TDSettings(Nobs, dt, force_backend=backend)
    N_fd = len(np.fft.rfftfreq(Nobs, dt))
    df_fd = 1.0 / (Nobs * dt)
    FDSettings(N_fd, df_fd, min_freq=min_freq, max_freq=max_freq, force_backend=backend)
    window = signal.windows.tukey(Nobs, alpha=0.05)

    layer_df = wdm_set.layer_df
    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation")

    N_INJ = 16384
    t_tdi_inj = np.linspace(t_arr_full[0], t_arr_full[-1], N_INJ)
    gb_tdi_kwargs = dict(tdi_config=tdi_config, orbits=orbits,
                          tdi_chan="XYZ", force_backend=backend)
    gb_gen_inj = GBTDIonTheFly(t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1,
                                **gb_tdi_kwargs)

    store_path = "wdm_lookup_new_all_time_layers_1.h5"
    wdm_lookup_table = WDMLookupTable.from_file(store_path, force_backend=backend)
    m_ref = int(wdm_lookup_table.m_ref)
    print(f"layer_df = {layer_df:.6e} Hz, m_ref = {m_ref}, Nt = {Nt}")

    # Pick a bad source from the prior-draw run that's near a layer boundary.
    # We use params close to draw 139 (mm5=+0.0765, snr=546, f0=15.891319 mHz,
    # f_frac=0.0265).  Easier to control: synthesize a clean source at a
    # specific f0 just above a layer boundary so the diagnostic is crisp.
    m_center = 500
    f0 = (m_center + 0.005) * layer_df       # f_frac = 0.005 above boundary
    params = np.array([3.0e-22, f0, 0.0, 0.0, 1.0, 1.5, 0.5, 2.0, 0.3])
    print(f"\nSource params: f0 = {f0*1e3:.6f} mHz   f_frac = "
          f"{f0/layer_df - np.floor(f0/layer_df):.4f}   m_floor = {int(f0/layer_df)}")

    # --- inject TD signal and WDM-transform it (this is the TRUTH) ---
    spl = gb_gen_inj(*params.reshape(9, 1), convert_to_ra_dec=False, return_spline=True)
    td_inj = np.asarray(spl.eval_tdi(t_arr_full))[0]   # shape (3, N)
    wdm_truth_sig = TDSignal(td_inj, settings=td_set).transform(wdm_set, window=window)
    truth_arr = np.asarray(wdm_truth_sig.arr if hasattr(wdm_truth_sig, "arr") else wdm_truth_sig)
    if truth_arr.shape == (3, wdm_set.Nf_active, wdm_set.Nt_active):
        pass
    elif truth_arr.shape == (3, wdm_set.Nt_active, wdm_set.Nf_active):
        truth_arr = truth_arr.transpose(0, 2, 1)
    print(f"truth shape = {truth_arr.shape}  (3, Nf_active={wdm_set.Nf_active}, "
          f"Nt_active={wdm_set.Nt_active})")

    # --- Python-side per-pixel input arrays ---
    t_wdm_active = wdm_set.t_arr + t_ref  # times at each WDM pixel (Nt_active long)
    t_wdm_3 = np.tile(t_wdm_active, (1, 3, 1))   # (1, 3, Nt_active)
    f_deriv_tdi = spl.tdi_phase_spl(t_wdm_3, derivative=1)[0] / (2 * np.pi)
    f_deriv_ref = spl.phase_ref_spl(t_wdm_active[None, :], derivative=1)[0] / (2 * np.pi)
    f_per_pixel = f_deriv_ref + f_deriv_tdi   # shape (3, Nt_active)
    tdi_amp = spl.tdi_amp_spl(t_wdm_3)[0]
    tdi_phase = spl.tdi_phase_spl(t_wdm_3)[0]
    ref_phase = spl.phase_ref_spl(t_wdm_active[None, :])[0]
    phi_per_pixel = (tdi_phase + ref_phase) + np.pi / 2.0

    print(f"\nf(t_n) range per channel: "
          f"min={f_per_pixel.min(axis=-1) * 1e3}  max={f_per_pixel.max(axis=-1) * 1e3}")

    # m_source at each pixel
    m_src = (f_per_pixel / layer_df).astype(int)
    # Per-channel: where does m_source change?
    print(f"m_source unique per channel: "
          f"ch0={np.unique(m_src[0])}  ch1={np.unique(m_src[1])}  ch2={np.unique(m_src[2])}")

    # Find crossing time bins (where m_source(t_n) != m_source(t_{n-1}))
    crossings = []
    for c in range(3):
        diffs = np.diff(m_src[c])
        idx = np.where(diffs != 0)[0]
        crossings.append(idx)
        if len(idx) > 0:
            print(f"  ch{c}: {len(idx)} crossings at active_n = {idx[:20]}{'...' if len(idx)>20 else ''}")

    # --- Run the Python lookup table -----
    n_arr_full = np.tile(np.arange(Nt)[wdm_set.active_slice_t], (3, 1))  # (3, Nt_active)
    amp_t = tdi_amp.flatten().copy()
    phi_t = phi_per_pixel.flatten().copy()
    freq_t = f_per_pixel.flatten().copy()
    fdot_t = np.full_like(freq_t, 0.0)
    n_arr_in = n_arr_full.flatten().copy()
    num_m_layers = 4
    wdm_coeffs, m_layers = wdm_lookup_table.get_wdm_coeffs(
        amp_t, phi_t, freq_t, fdot_t, n_arr_in, num_m_layers=num_m_layers,
    )
    # Reshape to (channel, n_active, m_offset)
    wdm_coeffs = wdm_coeffs.reshape(3, -1, 2 * num_m_layers + 1)
    m_layers = m_layers.reshape(3, -1, 2 * num_m_layers + 1)

    # Build the lookup template grid (3, Nf_active, Nt_active)
    py_tpl = np.zeros((3, wdm_set.Nf_active, wdm_set.Nt_active))
    m_min = wdm_set.ind_min_f
    for c in range(3):
        for n_local in range(wdm_set.Nt_active):
            for off in range(2 * num_m_layers + 1):
                m_here = int(m_layers[c, n_local, off])
                if m_here < wdm_set.ind_min_f or m_here >= wdm_set.ind_max_f:
                    continue
                py_tpl[c, m_here - m_min, n_local] += wdm_coeffs[c, n_local, off]

    # --- Diff between truth and Python lookup ---
    diff = truth_arr - py_tpl
    abs_diff = np.abs(diff)
    abs_truth = np.abs(truth_arr)

    # 5-layer mismatch
    m_floor = int(f0 / layer_df)
    layer_lo = m_floor - 3 - m_min
    layer_hi = m_floor + 2 - m_min + 1
    diff_5 = truth_arr[:, layer_lo:layer_hi, :] - py_tpl[:, layer_lo:layer_hi, :]
    truth_5 = truth_arr[:, layer_lo:layer_hi, :]
    print(f"\n5-layer L2 truth norm: {np.linalg.norm(truth_5):.6e}")
    print(f"5-layer L2 diff  norm: {np.linalg.norm(diff_5):.6e}")
    print(f"5-layer relative diff: {np.linalg.norm(diff_5) / np.linalg.norm(truth_5):.6e}")

    # --- Per-time-pixel breakdown: sum |truth - lookup|^2 over the 5
    #     central layers, plotted alongside f(t_n) / layer_df. -----
    per_n_diff = np.linalg.norm(
        truth_arr[:, layer_lo:layer_hi, :] - py_tpl[:, layer_lo:layer_hi, :], axis=(0, 1)
    )   # one value per active n
    per_n_truth = np.linalg.norm(truth_arr[:, layer_lo:layer_hi, :], axis=(0, 1))

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    n_ax = np.arange(wdm_set.Nt_active)

    for c in range(3):
        ax[0].plot(n_ax, f_per_pixel[c] / layer_df, label=f"ch{c}")
    ax[0].axhline(m_center, color="k", ls="--", alpha=0.6, label=f"layer boundary m={m_center}")
    ax[0].set_ylabel(r"$f(t_n)/\Delta f_{\rm layer}$")
    ax[0].legend(loc="best")
    ax[0].grid(True, ls=":", alpha=0.4)

    for c in range(3):
        ax[1].plot(n_ax, m_src[c], label=f"ch{c}", alpha=0.7)
    ax[1].axhline(m_center, color="k", ls="--", alpha=0.6)
    ax[1].set_ylabel(r"$m_{\rm source} = \lfloor f/\Delta f \rfloor$")
    ax[1].grid(True, ls=":", alpha=0.4)

    ax[2].plot(n_ax, per_n_truth, color="k", label="|truth|", alpha=0.6)
    ax[2].plot(n_ax, per_n_diff, color="crimson", label="|truth - lookup|")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("active WDM time pixel n")
    ax[2].set_ylabel("L2 norm over 5-layer window")
    ax[2].legend(loc="best")
    ax[2].grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig("dig_deep_per_pixel.png", dpi=130)
    print("[saved] dig_deep_per_pixel.png")

    # Heatmap of |truth - lookup| in the 5-layer window
    fig2, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for c in range(3):
        im = axes[c].imshow(
            np.abs(diff[c, layer_lo:layer_hi, :]),
            aspect="auto", origin="lower",
            extent=[0, wdm_set.Nt_active, m_floor - 3, m_floor + 2 + 1],
            cmap="viridis",
        )
        axes[c].set_ylabel(f"ch{c}  layer m")
        plt.colorbar(im, ax=axes[c], label="|truth - lookup|")
    axes[-1].set_xlabel("active WDM time pixel n")
    fig2.suptitle(
        f"|truth - Python-lookup|  in 5-layer window  (f0={f0*1e3:.6f} mHz, "
        f"f_frac={f0/layer_df - np.floor(f0/layer_df):.4f})"
    )
    plt.tight_layout()
    plt.savefig("dig_deep_heatmap.png", dpi=130)
    print("[saved] dig_deep_heatmap.png")

    np.savez(
        "dig_deep_results.npz",
        f0=f0, m_floor=m_floor, layer_df=layer_df,
        f_per_pixel=f_per_pixel, m_src=m_src,
        truth=truth_arr, py_tpl=py_tpl,
        per_n_truth=per_n_truth, per_n_diff=per_n_diff,
    )
    print("[saved] dig_deep_results.npz")


if __name__ == "__main__":
    main()
