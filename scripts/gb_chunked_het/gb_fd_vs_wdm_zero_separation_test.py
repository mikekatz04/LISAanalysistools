#!/usr/bin/env python
# coding: utf-8
"""
Compare two Galactic Binaries generated with GBTDIonTheFly.

Goal
----
Find the smallest frequency separation df such that the noise-weighted FD
inner product <h_A | h_B>_FD is zero, and check whether the WDM inner
product <h_A | h_B>_WDM is also zero at that df.

Setup
-----
* Tobs = 1 year, dt = 10 s
* Wavelet duration = 2 hours  ->  Nf = 720, layer_df = 1/(2*Nf*dt)
* No lookup tables — both sources are generated in TD (accurate, high
  N_sparse) and then transformed directly to FD and WDM.
* Tukey window on the TD data; its taper region covers ~EDGE_LAYERS_T
  wavelet layers on each time edge (default 30 each side).
* Active WDM/FD band is set ~EDGE_LAYERS_F layers in from DC and Nyquist
  so the edge frequency layers are cut from the inner product.
* lisatools handles the inner products.

Run
---
    conda activate deving
    python gb_fd_vs_wdm_zero_separation_test.py

Env overrides
-------------
    N_SPARSE        sparse-control-point count for GBTDIonTheFly (default 16384)
    MID_FREQ_MHZ    centre frequency of source A in mHz (default 5.0)
    EDGE_LAYERS_T   time-layer buffer at each end (default 30)
    EDGE_LAYERS_F   frequency-layer buffer at each end (default 30)
    N_DF            number of df samples in the coarse sweep (default 60)
    DF_MAX_OVER_TOBS_INV  upper end of sweep in units of 1/Tobs (default 5.0)
    DF_MIN_OVER_TOBS_INV  lower end of sweep in units of 1/Tobs (default 0.05)
"""

import os
import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, WDMSettings


def _as_float(x):
    """Extract a plain Python float from a numpy/cupy scalar or 1-elem array."""
    if hasattr(x, "get"):
        x = x.get()
    a = np.asarray(x).reshape(-1)
    assert a.size == 1, f"expected scalar, got shape {a.shape}"
    return float(a[0])


def main():
    backend = "cpu"

    # ------------------------------------------------------------------ shape
    dt = 10.0
    wavelet_duration = 2.0 * 3600.0  # 2 hours
    Nf = int(round(wavelet_duration / dt))
    assert abs(Nf * dt - wavelet_duration) < 1e-9, "wavelet_duration must divide by dt"
    Tobs_target = 1.0 * YRSID_SI
    Nt = int(np.floor(Tobs_target / wavelet_duration))
    if Nt % 2 == 1:  # WDM transform requires even Nt
        Nt -= 1
    Tobs = Nt * wavelet_duration
    N = Nf * Nt
    layer_df = 1.0 / (2.0 * Nf * dt)
    layer_dt = wavelet_duration  # = Nf * dt
    print(f"[setup] dt={dt}s  wavelet_duration={wavelet_duration}s ({wavelet_duration/3600:.2f} h)")
    print(f"[setup] Nf={Nf}  Nt={Nt}  N={N}  Tobs={Tobs/YRSID_SI:.6f} yr  ({Tobs:.3e} s)")
    print(f"[setup] WDM layer_df={layer_df:.6e} Hz   layer_dt={layer_dt:.2f} s")

    # ------------------------------------------------------------------ time grid
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(N) * dt + t_start
    t_ref = t_start
    N_sparse = int(os.environ.get("N_SPARSE", 16384))
    t_tdi_sparse = np.linspace(t_arr[0], t_arr[-1], N_sparse)
    print(f"[setup] N_sparse={N_sparse}  t_start={t_start:.3e}s  t_ref={t_ref:.3e}s")

    # ------------------------------------------------------------------ orbits + GB generator
    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig('2nd generation')
    gb_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )
    gb_gen = GBTDIonTheFly(t_tdi_sparse, Tobs, t_ref, 1.0 / dt, 1, **gb_kwargs)

    # ------------------------------------------------------------------ domain settings
    td_set = TDSettings(N, dt, force_backend=backend)

    # Time edges: trim EDGE_LAYERS_T wavelet layers from each end of the analysis.
    EDGE_LAYERS_T = int(os.environ.get("EDGE_LAYERS_T", 30))
    EDGE_LAYERS_F = int(os.environ.get("EDGE_LAYERS_F", 30))
    min_time = EDGE_LAYERS_T * wavelet_duration
    max_time = (Nt - EDGE_LAYERS_T) * wavelet_duration

    # Frequency edges: keep EDGE_LAYERS_F layers away from DC and from Nyquist.
    min_freq = EDGE_LAYERS_F * layer_df
    max_freq = (Nf - EDGE_LAYERS_F) * layer_df

    # Source A central frequency (must sit comfortably inside the active band).
    mid_freq = float(os.environ.get("MID_FREQ_MHZ", "5.0")) * 1.0e-3
    assert min_freq < mid_freq < max_freq, (
        f"mid_freq={mid_freq:.3e} must be inside active band [{min_freq:.3e}, {max_freq:.3e}]"
    )

    wdm_set = WDMSettings(
        Nf, Nt, dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend,
    )

    freqs_fd = np.fft.rfftfreq(N, dt)
    df_fd = freqs_fd[1] - freqs_fd[0]
    N_fd = len(freqs_fd)
    fd_set = FDSettings(N_fd, df_fd, min_freq=min_freq, max_freq=max_freq, force_backend=backend)

    print(f"[setup] EDGE_LAYERS_T={EDGE_LAYERS_T}   min_time={min_time:.3e} s   max_time={max_time:.3e} s")
    print(f"[setup] EDGE_LAYERS_F={EDGE_LAYERS_F}   active band [{min_freq*1e3:.4f}, {max_freq*1e3:.4f}] mHz")
    print(f"[setup] mid_freq (source A) = {mid_freq*1e3:.6f} mHz  (= layer {mid_freq/layer_df:.2f})")
    print(f"[setup] FD df = {df_fd:.6e} Hz   1/Tobs = {1.0/Tobs:.6e} Hz   layer_df/df_fd = {layer_df/df_fd:.2f}")

    # ------------------------------------------------------------------ Tukey window
    # Tukey alpha defined so the tapered region on each side covers EDGE_LAYERS_T wavelet layers.
    alpha = 2.0 * EDGE_LAYERS_T * Nf / N  # = 2 * EDGE_LAYERS_T / Nt
    window = sp_signal.windows.tukey(N, alpha=alpha)
    print(f"[setup] Tukey alpha={alpha:.5f}  taper={int(alpha*N/2)} samples per side "
          f"(= {EDGE_LAYERS_T} wavelet layers)")

    # ------------------------------------------------------------------ sensitivity
    sens_mat_fd = XYZ2SensitivityMatrix(fd_set, model="scirdv1")
    sens_mat_wdm = XYZ2SensitivityMatrix(wdm_set, model="scirdv1")

    # ------------------------------------------------------------------ base GB parameters
    amp_v   = 8.0e-22
    fdot_v  = 1.0e-17
    fddot_v = 0.0
    phi0_v  = 2.09802430298
    inc_v   = 0.23984234
    psi_v   = 1.234019814
    lam_v   = 4.09808143
    beta_v  = 0.04

    def gen_fd_wdm(f0):
        """Generate the GB at f0 (TD, accurate), then return (FDSignal, WDMSignal)."""
        params = np.array([[amp_v, f0, fdot_v, fddot_v, phi0_v, inc_v, psi_v, lam_v, beta_v]])
        out = gb_gen(*params.T, convert_to_ra_dec=False, return_spline=True)
        td_data = np.asarray(out.eval_tdi(t_arr))  # (1, 3, N) -> (3, N)
        if td_data.ndim == 3 and td_data.shape[0] == 1:
            td_data = td_data[0]
        td_sig = TDSignal(td_data, settings=td_set)
        fd_sig = td_sig.transform(fd_set, window=window)
        wdm_sig = td_sig.transform(wdm_set, window=window)
        return fd_sig, wdm_sig

    # ------------------------------------------------------------------ reference source A
    f0_A = mid_freq
    print(f"\n[A] f0_A = {f0_A:.10e} Hz")
    fd_A, wdm_A = gen_fd_wdm(f0_A)

    inj_fd = DataResidualArray(fd_A)
    inj_wdm = DataResidualArray(wdm_A)
    analysis_fd  = AnalysisContainer(inj_fd,  sens_mat_fd)
    analysis_wdm = AnalysisContainer(inj_wdm, sens_mat_wdm)

    ip_AA_fd  = _as_float(analysis_fd.inner_product())
    ip_AA_wdm = _as_float(analysis_wdm.inner_product())
    print(f"[A] <hA|hA>_FD  = {ip_AA_fd:+.6e}")
    print(f"[A] <hA|hA>_WDM = {ip_AA_wdm:+.6e}")
    print(f"[A] WDM/FD ratio (sanity)  = {ip_AA_wdm/ip_AA_fd:.6f}")

    # ------------------------------------------------------------------ df sweep
    df_min = float(os.environ.get("DF_MIN_OVER_TOBS_INV", "0.05")) / Tobs
    df_max = float(os.environ.get("DF_MAX_OVER_TOBS_INV", "10.0")) / Tobs
    n_pts = int(os.environ.get("N_DF", 100))
    df_grid = np.linspace(df_min, df_max, n_pts)
    print(f"\n[sweep] {n_pts} pts  df in [{df_min:.3e}, {df_max:.3e}] Hz  "
          f"(= [{df_min*Tobs:.3f}, {df_max*Tobs:.3f}] / Tobs  "
          f"= [{df_min/layer_df:.3e}, {df_max/layer_df:.3e}] layer_df)")

    overlaps_fd  = np.zeros(n_pts)
    overlaps_wdm = np.zeros(n_pts)
    ip_fd_vals   = np.zeros(n_pts)
    ip_wdm_vals  = np.zeros(n_pts)
    for j, df_j in enumerate(df_grid):
        f0_B = f0_A + df_j
        fd_B, wdm_B = gen_fd_wdm(f0_B)
        ip_fd  = _as_float(analysis_fd.template_inner_product(fd_B))
        ip_wdm = _as_float(analysis_wdm.template_inner_product(wdm_B))
        ip_fd_vals[j]  = ip_fd
        ip_wdm_vals[j] = ip_wdm
        overlaps_fd[j]  = ip_fd  / ip_AA_fd
        overlaps_wdm[j] = ip_wdm / ip_AA_wdm
        print(f"  [{j+1:3d}/{n_pts}]  df={df_j:.4e} Hz  ({df_j*Tobs:6.3f}/Tobs, "
              f"{df_j/layer_df:.3e} layer_df)  "
              f"FD_norm={overlaps_fd[j]:+.4e}  WDM_norm={overlaps_wdm[j]:+.4e}",
              flush=True)

    # ------------------------------------------------------------------ all FD/WDM sign changes
    def _sign_changes(arr):
        s = np.sign(arr)
        return np.where(s[:-1] * s[1:] < 0)[0]

    fd_sc  = _sign_changes(overlaps_fd)
    wdm_sc = _sign_changes(overlaps_wdm)
    print(f"\n[sign-changes] FD bracketed sign changes: {len(fd_sc)} at indices {list(fd_sc)}")
    print(f"[sign-changes] WDM bracketed sign changes: {len(wdm_sc)} at indices {list(wdm_sc)}")
    if len(fd_sc) == 0:
        print("[zero] no FD sign change detected in sweep — try widening DF_MAX_OVER_TOBS_INV.")
        return

    n_bisect = int(os.environ.get("N_BISECT", 30))
    fd_tol   = float(os.environ.get("FD_TOL", 1e-6))   # |FD_norm| target

    def bisect_zero(j_lo, ip_eval, label, ip_AA, others=()):
        """Bisect between df_grid[j_lo] and df_grid[j_lo+1] on the sign of ip_eval.

        ip_eval(fd_sig, wdm_sig) -> float, the IP we want to zero.
        Returns (df_zero, fd_at_zero, wdm_at_zero, history list of (df, ipvals dict)).
        """
        lo, hi = float(df_grid[j_lo]), float(df_grid[j_lo+1])
        # Use the already-computed sweep IP for the sign at lo.
        y_lo = ip_eval(*("sweep", j_lo))   # special call below
        sign_lo = float(np.sign(y_lo))
        history = []
        for it in range(n_bisect):
            mid = 0.5 * (lo + hi)
            fd_mid, wdm_mid = gen_fd_wdm(f0_A + mid)
            v_fd  = _as_float(analysis_fd.template_inner_product(fd_mid)) / ip_AA_fd
            v_wdm = _as_float(analysis_wdm.template_inner_product(wdm_mid)) / ip_AA_wdm
            v = v_fd if label == "FD" else v_wdm
            history.append((mid, v_fd, v_wdm))
            print(f"    [{label}-bisect {it+1:2d}/{n_bisect}]  df={mid:.6e}  "
                  f"FD_norm={v_fd:+.4e}  WDM_norm={v_wdm:+.4e}", flush=True)
            if np.sign(v) == sign_lo:
                lo = mid
            else:
                hi = mid
            if abs(v) < fd_tol and (hi - lo) < 1e-7 / Tobs:
                break
        df_zero = 0.5 * (lo + hi)
        fd_at, wdm_at = gen_fd_wdm(f0_A + df_zero)
        return df_zero, fd_at, wdm_at, history

    # The helper above accepts a special "sweep" call to look up cached value.
    def _ip_eval_FD(*args):
        if args and args[0] == "sweep":
            return overlaps_fd[args[1]]
        raise RuntimeError("unexpected ip_eval call")

    def _ip_eval_WDM(*args):
        if args and args[0] == "sweep":
            return overlaps_wdm[args[1]]
        raise RuntimeError("unexpected ip_eval call")

    # Bisect every FD sign change and record both FD and WDM IPs at the zero.
    fd_zeros = []   # list of dicts
    print("\n[refine FD] bisecting every FD sign change")
    for k, j0 in enumerate(fd_sc):
        print(f"  ---- FD zero {k+1}/{len(fd_sc)} bracket "
              f"[{df_grid[j0]:.4e}, {df_grid[j0+1]:.4e}] ----")
        df_z, fd_z, wdm_z, hist = bisect_zero(int(j0), _ip_eval_FD, "FD", ip_AA_fd)
        v_fd  = _as_float(analysis_fd.template_inner_product(fd_z)) / ip_AA_fd
        v_wdm = _as_float(analysis_wdm.template_inner_product(wdm_z)) / ip_AA_wdm
        fd_zeros.append(dict(
            df=df_z, fd_norm=v_fd, wdm_norm=v_wdm,
            fd_sig=fd_z, wdm_sig=wdm_z, hist=hist,
        ))
        print(f"  -> df_FDzero[{k+1}] = {df_z:.8e} Hz  ({df_z*Tobs:.5f}/Tobs, "
              f"{df_z/layer_df:.4e} layer_df)   FD_norm={v_fd:+.3e}   WDM_norm={v_wdm:+.3e}")

    # Bisect every WDM sign change to find where WDM IP itself is zero.
    wdm_zeros = []
    print("\n[refine WDM] bisecting every WDM sign change")
    for k, j0 in enumerate(wdm_sc):
        print(f"  ---- WDM zero {k+1}/{len(wdm_sc)} bracket "
              f"[{df_grid[j0]:.4e}, {df_grid[j0+1]:.4e}] ----")
        df_z, fd_z, wdm_z, hist = bisect_zero(int(j0), _ip_eval_WDM, "WDM", ip_AA_wdm)
        v_fd  = _as_float(analysis_fd.template_inner_product(fd_z)) / ip_AA_fd
        v_wdm = _as_float(analysis_wdm.template_inner_product(wdm_z)) / ip_AA_wdm
        wdm_zeros.append(dict(
            df=df_z, fd_norm=v_fd, wdm_norm=v_wdm,
            fd_sig=fd_z, wdm_sig=wdm_z, hist=hist,
        ))
        print(f"  -> df_WDMzero[{k+1}] = {df_z:.8e} Hz  ({df_z*Tobs:.5f}/Tobs, "
              f"{df_z/layer_df:.4e} layer_df)   FD_norm={v_fd:+.3e}   WDM_norm={v_wdm:+.3e}")

    # ------------------------------------------------------------------ overlap plot
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axes[0].axhline(0.0, color='k', lw=0.5)
    axes[0].plot(df_grid * Tobs, overlaps_fd,  '.-', label=r'FD: $\langle h_A|h_B\rangle/\langle h_A|h_A\rangle$')
    axes[0].plot(df_grid * Tobs, overlaps_wdm, '.-', label=r'WDM: $\langle h_A|h_B\rangle/\langle h_A|h_A\rangle$')
    for z in fd_zeros:
        axes[0].axvline(z["df"] * Tobs, color='r', ls='--', alpha=0.4)
    for z in wdm_zeros:
        axes[0].axvline(z["df"] * Tobs, color='b', ls=':', alpha=0.4)
    axes[0].set_ylabel('normalized inner product')
    axes[0].legend(loc='best')
    axes[0].set_title(
        f'GB FD vs WDM IP, f0_A={f0_A*1e3:.3f} mHz, '
        f'Nf={Nf}, Nt={Nt}, Tobs={Tobs/YRSID_SI:.3f} yr,\n'
        f'layer_df={layer_df:.3e} Hz, 1/Tobs={1.0/Tobs:.3e} Hz, '
        f'EDGE_T/F={EDGE_LAYERS_T}/{EDGE_LAYERS_F},  '
        f'#FD zeros={len(fd_zeros)}, #WDM zeros={len(wdm_zeros)}'
    )
    axes[1].semilogy(df_grid * Tobs, np.abs(overlaps_fd),  '.-', label='|FD|')
    axes[1].semilogy(df_grid * Tobs, np.abs(overlaps_wdm), '.-', label='|WDM|')
    for z in fd_zeros:
        axes[1].axvline(z["df"] * Tobs, color='r', ls='--', alpha=0.4)
    for z in wdm_zeros:
        axes[1].axvline(z["df"] * Tobs, color='b', ls=':', alpha=0.4)
    axes[1].set_xlabel(r'$\Delta f \cdot T_{\rm obs}$  (cycles per observation)')
    axes[1].set_ylabel('|normalized inner product|')
    axes[1].legend(loc='best')
    plt.tight_layout()
    out_png = "gb_fd_vs_wdm_zero_separation.png"
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\n[plot] saved {out_png}")

    # ------------------------------------------------------------------ FD waveform + WDM heatmap at first FD zero
    z1 = fd_zeros[0]
    df1, fd_B1, wdm_B1 = z1["df"], z1["fd_sig"], z1["wdm_sig"]
    f0_B1 = f0_A + df1

    # FD waveform plot: |X(f)| vs frequency, zoomed around f0.
    # NOTE: fd.arr is already trimmed to the active band [ind_min:ind_max+1].
    fd_freqs_full = np.fft.rfftfreq(N, dt)
    fd_freqs = fd_freqs_full[fd_set.ind_min:fd_set.ind_max + 1]
    fd_A_arr  = np.asarray(fd_A.arr)
    fd_B1_arr = np.asarray(fd_B1.arr)
    assert fd_A_arr.shape[-1] == fd_freqs.shape[0], (
        f"fd.arr trim mismatch {fd_A_arr.shape} vs {fd_freqs.shape}"
    )
    # window in freq around f0 (~ 10/Tobs)
    halfwin = 10.0 / Tobs
    fmask = (fd_freqs >= f0_A - halfwin) & (fd_freqs <= f0_A + halfwin)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for ch, name in enumerate(("X", "Y", "Z")):
        axes[ch].plot((fd_freqs[fmask] - f0_A) * Tobs, np.abs(fd_A_arr[ch, fmask]),
                      'C0', lw=1.0, label=f'|{name}_A(f)| at f0_A={f0_A*1e3:.4f} mHz')
        axes[ch].plot((fd_freqs[fmask] - f0_A) * Tobs, np.abs(fd_B1_arr[ch, fmask]),
                      'C1', lw=1.0, label=f'|{name}_B(f)| at f0_B=f0_A+{df1*Tobs:.3f}/Tobs')
        axes[ch].axvline(0.0,             color='C0', ls=':', alpha=0.5)
        axes[ch].axvline(df1 * Tobs,      color='C1', ls=':', alpha=0.5)
        axes[ch].set_ylabel(f'|{name}(f)|')
        axes[ch].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel(r'$(f - f_{0,A}) \cdot T_{\rm obs}$')
    axes[0].set_title(
        f'FD waveforms at FD-zero #1: df={df1:.4e} Hz ({df1*Tobs:.4f}/Tobs)\n'
        f'<hA|hB>_FD/<hA|hA>_FD = {z1["fd_norm"]:+.3e}    '
        f'<hA|hB>_WDM/<hA|hA>_WDM = {z1["wdm_norm"]:+.3e}'
    )
    plt.tight_layout()
    out_fd = "gb_fd_waveforms_at_FDzero1.png"
    plt.savefig(out_fd, dpi=130)
    plt.close(fig)
    print(f"[plot] saved {out_fd}")

    # WDM heatmap comparison
    wdm_A_arr  = np.asarray(wdm_A.arr)            # (3, Nf_act, Nt_act)
    wdm_B1_arr = np.asarray(wdm_B1.arr)
    # m-index in the active grid where source A lives
    m_A_abs = int(round(f0_A / layer_df))
    m_A_act = m_A_abs - wdm_set.ind_min_f
    m_lo = max(0, m_A_act - 12)
    m_hi = min(wdm_A_arr.shape[1], m_A_act + 13)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
    vmax_pair = max(np.abs(wdm_A_arr[0, m_lo:m_hi]).max(),
                    np.abs(wdm_B1_arr[0, m_lo:m_hi]).max())
    extent = [0, wdm_A_arr.shape[2], (m_lo + wdm_set.ind_min_f) * layer_df * 1e3,
              (m_hi + wdm_set.ind_min_f) * layer_df * 1e3]
    im0 = axes[0].imshow(wdm_A_arr[0, m_lo:m_hi], aspect='auto', origin='lower',
                         cmap='RdBu_r', vmin=-vmax_pair, vmax=vmax_pair, extent=extent)
    axes[0].set_title(f'h_A WDM (X), f0_A={f0_A*1e3:.4f} mHz, m_A={m_A_abs}')
    axes[0].set_ylabel('f (mHz)')
    plt.colorbar(im0, ax=axes[0], fraction=0.025)

    im1 = axes[1].imshow(wdm_B1_arr[0, m_lo:m_hi], aspect='auto', origin='lower',
                         cmap='RdBu_r', vmin=-vmax_pair, vmax=vmax_pair, extent=extent)
    axes[1].set_title(
        f'h_B WDM (X) at FD-zero #1, df={df1:.4e} Hz ({df1*Tobs:.4f}/Tobs, '
        f'{df1/layer_df:.3e} layer_df)'
    )
    axes[1].set_ylabel('f (mHz)')
    plt.colorbar(im1, ax=axes[1], fraction=0.025)

    diff = wdm_A_arr[0, m_lo:m_hi] - wdm_B1_arr[0, m_lo:m_hi]
    vmax_d = np.abs(diff).max()
    im2 = axes[2].imshow(diff, aspect='auto', origin='lower',
                         cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, extent=extent)
    axes[2].set_title(
        f'h_A − h_B   (sanity: nonzero residual expected, FD IP zero is orthogonality)\n'
        f'FD_norm={z1["fd_norm"]:+.3e}   WDM_norm={z1["wdm_norm"]:+.3e}'
    )
    axes[2].set_xlabel('time pixel n')
    axes[2].set_ylabel('f (mHz)')
    plt.colorbar(im2, ax=axes[2], fraction=0.025)
    plt.tight_layout()
    out_wdm = "gb_wdm_heatmaps_at_FDzero1.png"
    plt.savefig(out_wdm, dpi=130)
    plt.close(fig)
    print(f"[plot] saved {out_wdm}")

    # ------------------------------------------------------------------ summary table + npz
    print("\n=== FD zeros ============================================")
    print("  # | df [Hz]        | df*Tobs   | df/layer_df  | FD_norm     | WDM_norm")
    for k, z in enumerate(fd_zeros):
        print(f"  {k+1:2d} | {z['df']:.6e} | {z['df']*Tobs:8.5f} | {z['df']/layer_df:.4e} | "
              f"{z['fd_norm']:+.3e} | {z['wdm_norm']:+.3e}")

    print("\n=== WDM zeros ===========================================")
    print("  # | df [Hz]        | df*Tobs   | df/layer_df  | FD_norm     | WDM_norm")
    for k, z in enumerate(wdm_zeros):
        print(f"  {k+1:2d} | {z['df']:.6e} | {z['df']*Tobs:8.5f} | {z['df']/layer_df:.4e} | "
              f"{z['fd_norm']:+.3e} | {z['wdm_norm']:+.3e}")

    # Pair up FD zeros with nearest WDM zero
    if wdm_zeros:
        print("\n=== FD-zero <-> nearest WDM-zero offset =================")
        print("  k | df_FD          | df_WDM         | (df_WDM-df_FD)   | offset/(1/Tobs) | offset/layer_df")
        for k, z in enumerate(fd_zeros):
            j = int(np.argmin([abs(zw["df"] - z["df"]) for zw in wdm_zeros]))
            off = wdm_zeros[j]["df"] - z["df"]
            print(f"  {k+1:2d} | {z['df']:.6e} | {wdm_zeros[j]['df']:.6e} | {off:+.3e}     | "
                  f"{off*Tobs:+.5f}      | {off/layer_df:+.3e}")

    print("\n=== SETUP ===============================================")
    print(f"  Tobs                    = {Tobs:.6e} s   (1/Tobs = {1.0/Tobs:.6e} Hz)")
    print(f"  WDM layer_df            = {layer_df:.6e} Hz")
    print(f"  layer_df * Tobs         = {layer_df * Tobs:.2f}  "
          f"(=> {layer_df * Tobs:.0f} FD zeros per layer if zeros are at integer/Tobs)")
    print(f"  <hA|hA>_FD              = {ip_AA_fd:+.6e}")
    print(f"  <hA|hA>_WDM             = {ip_AA_wdm:+.6e}")

    out_npz = "gb_fd_vs_wdm_zero_separation.npz"
    np.savez(
        out_npz,
        df_grid=df_grid, overlaps_fd=overlaps_fd, overlaps_wdm=overlaps_wdm,
        ip_fd_vals=ip_fd_vals, ip_wdm_vals=ip_wdm_vals,
        ip_AA_fd=ip_AA_fd, ip_AA_wdm=ip_AA_wdm,
        Tobs=Tobs, layer_df=layer_df, layer_dt=layer_dt,
        Nf=Nf, Nt=Nt, dt=dt, mid_freq=mid_freq,
        EDGE_LAYERS_T=EDGE_LAYERS_T, EDGE_LAYERS_F=EDGE_LAYERS_F,
        alpha=alpha,
        fd_zero_df=np.array([z["df"] for z in fd_zeros]),
        fd_zero_fd_norm=np.array([z["fd_norm"] for z in fd_zeros]),
        fd_zero_wdm_norm=np.array([z["wdm_norm"] for z in fd_zeros]),
        wdm_zero_df=np.array([z["df"] for z in wdm_zeros]),
        wdm_zero_fd_norm=np.array([z["fd_norm"] for z in wdm_zeros]),
        wdm_zero_wdm_norm=np.array([z["wdm_norm"] for z in wdm_zeros]),
    )
    print(f"[npz] saved {out_npz}")


if __name__ == "__main__":
    main()
