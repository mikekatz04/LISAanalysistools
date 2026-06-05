#!/usr/bin/env python
"""Per-pixel residual vs instantaneous fdot scatter (Python lookup, n_ref_only).

For a single GB source, builds the WDM injection via accurate TDI-on-the-fly,
generates the Python n_ref_only lookup template at fdot=0, and scatters
the per-(channel, m, n) residual |inj - tpl|^2 against the instantaneous
|fdot| measured from the cubic spline at that pixel's centre time.

No PSD weighting on the residual. Single source so the spline-fdot mapping
to pixels is unambiguous.

Env vars:
    LOOKUP_PATH  default wdm_lookup_n_ref_only_NF365_NT10240_TL256.h5
    NF, NT       default 365 / 10240
    F0_MHZ       default 16.5
    BETA         default 0.04
    AMP          default 5e-22 (high SNR)
    OUT          default per_pixel_mm_vs_fdot.png
"""
import os
import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.domains import (
    TDSettings, TDSignal, FDSettings, WDMSettings, WDMSignal, WDMLookupTable,
)


def main():
    backend = "cpu"
    xp = np

    LOOKUP_PATH = os.environ.get("LOOKUP_PATH", "wdm_lookup_n_ref_only_NF365_NT10240_TL256.h5")
    Nf = int(os.environ.get("NF", 365))
    Nt = int(os.environ.get("NT", 10240))
    dt = 10.0

    # F0_MHZ_LIST overrides the sweep — comma-separated list of f0 in mHz.
    # Default: 20 values uniformly covering one m-layer around 16.5 mHz so
    # f_frac samples [0, 1).
    beta_v = float(os.environ.get("BETA", 0.04))
    amp = float(os.environ.get("AMP", 5e-22))
    OUT = os.environ.get("OUT", "per_pixel_mm_vs_fdot.png")
    _f0_env = os.environ.get("F0_MHZ_LIST", "")
    if _f0_env:
        f0_mHz_list = [float(s) for s in _f0_env.split(",") if s.strip()]
    else:
        N_f0 = int(os.environ.get("N_F0", 20))
        # m_base picks the central layer; we sweep f_frac over [0,1) of it.
        # layer_df = 1 / (2*Nf*dt) = 1.37e-4 Hz at Nf=365 → m=120 ≈ 16.4 mHz.
        m_base = int(os.environ.get("M_BASE", 120))
        layer_df_local = 1.0 / (2 * Nf * dt)
        f0_mHz_list = [(m_base + ff) * layer_df_local * 1e3
                       for ff in np.linspace(0.0, 1.0, N_f0, endpoint=False)]
    print(f"[sweep] N_f0={len(f0_mHz_list)}  f0 range="
          f"[{min(f0_mHz_list):.4f}, {max(f0_mHz_list):.4f}] mHz", flush=True)

    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation")

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start
    N_inj = 16384

    td_set = TDSettings(t_arr.shape[-1], dt, force_backend=backend)
    min_freq = 1e-4
    max_freq = 35e-3
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration
    wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq,
                          min_time=min_time, max_time=max_time)

    wdm_lookup_table = WDMLookupTable.from_file(LOOKUP_PATH, force_backend=backend)
    _wdm_settings = WDMSettings(*wdm_lookup_table.args, **wdm_lookup_table.kwargs)
    if not _wdm_settings.eq_without_inds(wdm_set):
        raise ValueError("WDM settings mismatch between lookup and computed wdm_set")
    print(f"[load] {LOOKUP_PATH}  build_kind={wdm_lookup_table.build_kind}  "
          f"m_ref={wdm_lookup_table.m_ref}  n_ref={wdm_lookup_table.n_ref}", flush=True)

    # ---- Build injection generator once (re-used across all sources) ----
    gb_tdi_kwargs = dict(tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ", force_backend=backend)
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_inj)
    gb_gen_inj = GBTDIonTheFly(t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs)

    # Accumulators across the f0 sweep
    all_x_fdot = []
    all_x_ffrac = []
    all_y = []
    all_ch = []

    for _i_src, f0_mHz in enumerate(f0_mHz_list):
        f0 = f0_mHz * 1e-3
        params = np.array([[amp, f0, 0.0, 0.0, 1.234, 1.123, 0.789, 2.456, beta_v]])
        print(f"\n[src {_i_src+1}/{len(f0_mHz_list)}] f0={f0_mHz:.5f} mHz", flush=True)
        inj_spline = gb_gen_inj(*params.T, convert_to_ra_dec=False, return_spline=True)
        td_inj = np.asarray(inj_spline.eval_tdi(t_arr))[0]
        inj_wdm = TDSignal(td_inj, settings=td_set).transform(wdm_set, window=None)
        inj_arr = np.asarray(inj_wdm.arr if hasattr(inj_wdm, "arr") else inj_wdm)

        # Spline-derived f, fdot, amp, phi at every pixel centre time
        t_pix_active = wdm_set.t_arr + t_ref  # (Nt_active,)
        f_deriv_tdi    = inj_spline.tdi_phase_spl (np.tile(t_pix_active, (1, 3, 1)), derivative=1)[0] / (2 * np.pi)
        f_deriv_ref    = inj_spline.phase_ref_spl (t_pix_active[None, :],            derivative=1)[0] / (2 * np.pi)
        fdot_deriv_tdi = inj_spline.tdi_phase_spl (np.tile(t_pix_active, (1, 3, 1)), derivative=2)[0] / (2 * np.pi)
        fdot_deriv_ref = inj_spline.phase_ref_spl (t_pix_active[None, :],            derivative=2)[0] / (2 * np.pi)
        tdi_amp_pix    = inj_spline.tdi_amp_spl   (np.tile(t_pix_active, (1, 3, 1)))[0]
        tdi_phase_pix  = inj_spline.tdi_phase_spl (np.tile(t_pix_active, (1, 3, 1)))[0]
        ref_phase_pix  = inj_spline.phase_ref_spl (t_pix_active[None, :])[0]
        f_pix    = f_deriv_ref + f_deriv_tdi
        fdot_pix = fdot_deriv_ref + fdot_deriv_tdi

        phi_t = ((tdi_phase_pix + ref_phase_pix) + np.pi / 2.0).flatten().copy()
        amp_t = tdi_amp_pix.flatten().copy()
        freq_t = f_pix.flatten().copy()
        fdot_t = np.zeros_like(freq_t)
        n_arr = np.tile(xp.arange(wdm_set.Nt)[wdm_set.active_slice_t], (3, 1))
        n_arr_in = n_arr.flatten().copy()

        _wdm_coeffs, _m_layers = wdm_lookup_table.get_wdm_coeffs(
            amp_t, phi_t, freq_t, fdot_t, n_arr_in, num_m_layers=2,
        )
        wdm_coeffs = _wdm_coeffs.reshape(3, -1, _wdm_coeffs.shape[-1])
        m_layers   = _m_layers.reshape(3, -1, _wdm_coeffs.shape[-1])
        n_layers   = np.repeat(n_arr[:, :, None], m_layers.shape[-1], axis=-1)
        tpl_arr = np.zeros_like(inj_arr)
        keep = (m_layers >= wdm_set.ind_min_f) & (m_layers <= wdm_set.ind_max_f)
        ch_ind = np.repeat(np.arange(3)[:, None],
                           m_layers.shape[-1] * m_layers.shape[-2], axis=-1).reshape(m_layers.shape)
        m_min = wdm_set.ind_min_f
        n_min = wdm_set.ind_min_t
        tpl_arr[ch_ind[keep], m_layers[keep] - m_min, n_layers[keep] - n_min] = wdm_coeffs[keep]

        diff2 = (inj_arr - tpl_arr) ** 2
        m_source_per_pix = (f_pix / wdm_set.layer_df).astype(int)
        NM = 2
        diag_mask = np.zeros_like(diff2, dtype=bool)
        m_idx_grid = np.arange(wdm_set.Nf_active) + m_min
        for ch in range(3):
            for nl in range(wdm_set.Nt_active):
                m_src = m_source_per_pix[ch, nl]
                m_mask = (m_idx_grid >= m_src - NM) & (m_idx_grid <= m_src + NM)
                diag_mask[ch, m_mask, nl] = True

        target_shape = diff2.shape
        fdot_pix_grid = np.broadcast_to(np.abs(fdot_pix)[:, None, :], target_shape)
        f_pix_norm = f_pix / wdm_set.layer_df
        f_frac_pix = f_pix_norm - np.floor(f_pix_norm)
        f_frac_grid = np.broadcast_to(f_frac_pix[:, None, :], target_shape)
        all_x_fdot.append(fdot_pix_grid[diag_mask].copy())
        all_x_ffrac.append(f_frac_grid[diag_mask].copy())
        all_y.append(diff2[diag_mask].copy())
        all_ch.append(np.broadcast_to(np.arange(3)[:, None, None], target_shape)[diag_mask].copy())
        print(f"  kept={int(diag_mask.sum())}  diff2 max={diff2[diag_mask].max():.3e}  "
              f"median={np.median(diff2[diag_mask]):.3e}  |fdot| max={fdot_pix_grid[diag_mask].max():.3e}",
              flush=True)

    x_fdot  = np.concatenate(all_x_fdot)
    x_ffrac = np.concatenate(all_x_ffrac)
    y_pts   = np.concatenate(all_y)
    ch_pts  = np.concatenate(all_ch)
    print(f"\n[stack] total points: {len(y_pts)}", flush=True)

    # ---- Plot ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["C0", "C1", "C2"]
    labels = ["X", "Y", "Z"]
    for ch in range(3):
        sel = ch_pts == ch
        if sel.sum() == 0:
            continue
        axes[0].scatter(x_fdot[sel],  y_pts[sel], s=4, alpha=0.2,
                        color=colors[ch], label=f"ch {labels[ch]}")
        axes[1].scatter(x_ffrac[sel], y_pts[sel], s=4, alpha=0.2,
                        color=colors[ch], label=f"ch {labels[ch]}")
    for ax_, xl in zip(axes, [r"$|\dot f|$ from cubic spline  [Hz/s]",
                              r"$f_{\rm inst}/\Delta f_{\rm layer}$ mod 1"]):
        ax_.set_yscale("log")
        if "Hz/s" in xl:
            ax_.set_xscale("log")
        ax_.set_xlabel(xl)
        ax_.set_ylabel(r"per-pixel residual $|inj - tpl_{\rm py}|^2$  (un-weighted)")
        ax_.grid(True, which="both", ls=":", alpha=0.4)
        ax_.legend(loc="best")
    axes[1].axvline(0.5, color="k", ls=":", alpha=0.5)
    fig.suptitle(f"n_ref_only / fdot=0 lookup  ·  {len(f0_mHz_list)} sources spanning "
                 f"[{min(f0_mHz_list):.3f},{max(f0_mHz_list):.3f}] mHz, β={beta_v}, Nf={Nf}")
    plt.tight_layout()
    plt.savefig(OUT, dpi=130)
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
