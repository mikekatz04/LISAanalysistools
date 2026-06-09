#!/usr/bin/env python
"""Walk-through diagnostic for the v2 polyphase + de-rotated signal-het pipeline.

For each of the first ``N_DRAWS_PLOT`` prior draws (SEED matches the mm sweep),
produce a four-column figure row:

  col 0:  Dense real-WDM injection ``c0_real[m, n]`` zoomed to the active
          m-band (m_floor +/- 2, 5 layers).
  col 1:  v2 sparse complex-WDM ``c1_sparse[m, n_sparse]`` at the active
          m-band (real part heatmap on the sparse n grid).
  col 2:  Heterodyne ratio r(n) = c1_sparse / c0_sparse for the carrier
          layer (m=m_floor) -- Re(r), Im(r) vs n, plus the de-rotated
          version after subtracting the analytic 2*pi*Df0*t phase.
  col 3:  Numeric panel: log-likelihood at injection (should be ~0 when
          DF0_FRAC=0), mm5, mm2, plus parameter summary.

Run::
    python gb_signal_het_wdm_v2_walkthrough.py
Env vars:
    DF0_FRAC      Df0/layer_df (default 0 -- self-consistency: x_cand = x_inj)
    N_DRAWS_PLOT  number of prior draws to plot (default 5)
    NT_LAYER      polyphase iFFT length (default 64)
    SEED          prior RNG seed (default 54321, matches the sweep)
    OUT_PNG       output path
"""

from __future__ import annotations

import os
import sys

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
import lisatools_backend_cpu.pycppdetector as _lat_pd
import gbgpu_backend_cpu.cgbgpu as _be  # GBComputationGroupWrap lives here post-3L.7g

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen
from gb_signal_het_wdm_v2_mm_sweep import (
    build_gb_prior, build_v2_real_template, compute_mm,
)
from gb_signal_het_cpp_validate import python_bin_fold


def main():
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20

    DF0_FRAC = float(os.environ.get("DF0_FRAC", "0"))
    # Additional per-parameter perturbations for Df{i} walkthrough plots.
    DFDOT      = float(os.environ.get("DFDOT", "0"))       # absolute Hz/s
    DAMP_FRAC  = float(os.environ.get("DAMP_FRAC", "0"))   # relative
    DPHI0      = float(os.environ.get("DPHI0", "0"))       # rad
    DINC       = float(os.environ.get("DINC", "0"))        # rad (on inc)
    DPSI       = float(os.environ.get("DPSI", "0"))        # rad
    DLAM       = float(os.environ.get("DLAM", "0"))        # rad
    DBETA      = float(os.environ.get("DBETA", "0"))       # rad
    PERT_LABEL = os.environ.get("PERT_LABEL", "")
    N_DRAWS_PLOT = int(os.environ.get("N_DRAWS_PLOT", "5"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    SEED = int(os.environ.get("SEED", "54321"))

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

    def real_td_cb(p):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = p
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=True, force_backend=backend,
    )
    layer_df = wdm_set_real.layer_df
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)
    Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
    ind_min_f = int(wdm_set_real.ind_min_f)
    print(f"[grid] layer_df={layer_df:.3e}Hz  Nt_active={Nt_active}  "
          f"ind_min_t={ind_min_t}  ind_min_f={ind_min_f}", flush=True)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=Nt_layer, m_active_half_width=2,
    )
    n_sparse = sparse_gen.n_sparse_local
    stride = sparse_gen.stride

    # C++ handles for the get_ll comparison panel
    _cpp = _be.GBComputationGroupWrapCPU()
    _tdi_wrap = gb_gen.wave_gen
    print(f"[v2] Nt_layer={Nt_layer}  stride={stride}  "
          f"N_sparse_t={sparse_gen.N_sparse_t}", flush=True)

    # Prior matches the sweep: f0_hi=25mHz, buffer 7 layers.
    np.random.seed(SEED)
    f0_lo = (ind_min_f + 7) * layer_df
    f0_hi = 0.025
    prior, tc = build_gb_prior(
        A_lims=(1e-23, 1e-20), f0_lims_hz=(f0_lo, f0_hi),
        fdot_lims=(-1e-15, 1e-15), beta_lims=None,
    )

    SNR_MIN, SNR_MAX = 5.0, 1100.0
    MAX_REJECT = 500

    fig, axs = plt.subplots(N_DRAWS_PLOT, 4, figsize=(22, 4.6 * N_DRAWS_PLOT))
    if N_DRAWS_PLOT == 1:
        axs = axs[None, :]

    sens_mat = None
    drawi = 0
    while drawi < N_DRAWS_PLOT:
        chosen = None
        for _ in range(MAX_REJECT):
            x_samp = prior.rvs(size=1)
            params_inj = tc.both_transforms(x_samp.copy())[0]
            td_inj = real_td_cb(params_inj)
            wdm_inj_real_sig = TDSignal(td_inj, settings=td_set).transform(
                wdm_set_real, window=window,
            )
            inj_data_arr = DataResidualArray(wdm_inj_real_sig)
            if sens_mat is None:
                sens_mat = XYZ2SensitivityMatrix(
                    inj_data_arr.data_res_arr.settings, model="scirdv1",
                )
            analysis = AnalysisContainer(inj_data_arr, sens_mat)
            snr_i = float(analysis.snr())
            if SNR_MIN <= snr_i <= SNR_MAX:
                chosen = (params_inj, td_inj, wdm_inj_real_sig, snr_i, analysis)
                break
        if chosen is None:
            print(f"[warn] draw {drawi}: exhausted reject; keeping last "
                  f"(snr={snr_i:.2f})", flush=True)
            chosen = (params_inj, td_inj, wdm_inj_real_sig, snr_i, analysis)
        params_inj, td_inj, wdm_inj_real_sig, snr_i, analysis = chosen
        wdm_inj_arr = np.asarray(wdm_inj_real_sig.arr)            # (3, Nf_act, Nt_act)

        # x_cand = x_inj + per-parameter delta (env-driven Df{i} walkthrough).
        # Phys param order: 0 amp 1 f0 2 fdot 3 fddot 4 phi0 5 inc 6 psi 7 lam 8 beta.
        params_cand = params_inj.copy()
        params_cand[0] *= (1.0 + DAMP_FRAC)                  # relative amp
        params_cand[1] += DF0_FRAC * layer_df                # f0
        params_cand[2] += DFDOT                              # fdot
        params_cand[4] += DPHI0                              # phi0
        params_cand[5] += DINC                               # inc
        params_cand[6] += DPSI                               # psi
        params_cand[7] += DLAM                               # lam
        params_cand[8] += DBETA                              # beta
        td_cand = real_td_cb(params_cand)
        fd_rfft_cand = np.fft.rfft(td_cand * window, axis=-1)

        # Reference complex c0 at x_inj
        wdm_ref_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window,
            ).arr
        ).copy()
        c0_sparse = wdm_ref_complex[:, :, n_sparse].copy()

        # v2 sparse + reconstruction
        c1_sparse, m_local_active = sparse_gen.sparse_from_rfft(
            fd_rfft_cand, params_cand,
        )                                                          # (3, 5, N_sp)
        tpl_real = build_v2_real_template(
            sparse_gen, fd_rfft_cand, params_cand, params_inj,
            Nf_active, Nt_active, wdm_ref_complex, c0_sparse,
            dt, Nf, ind_min_t,
        )

        # mm5 / mm2 (template = v2, data = dense lisatools inj)
        f0_cand = float(params_cand[1])
        mm5 = compute_mm(wdm_inj_arr, tpl_real, wdm_set_real, f0_cand,
                          "mm5", backend)
        mm2 = compute_mm(wdm_inj_arr, tpl_real, wdm_set_real, f0_cand,
                          "mm2", backend)

        # logL = <d|tpl> - 0.5 * <tpl|tpl> - 0.5 * <d|d>
        tpl_sig = WDMSignal(tpl_real, wdm_set_real)
        d_h = float(analysis.template_inner_product(tpl_sig))
        h_h = float(analysis.template_snr(tpl_sig)[0]) ** 2
        d_d = float(np.real(analysis.inner_product()))
        logL = d_h - 0.5 * h_h - 0.5 * d_d

        # Full lisatools direct: WDM transform of td_cand (no heterodyne).
        # This is the ground-truth likelihood the v2 path approximates.
        tpl_direct_wdm = TDSignal(td_cand, settings=td_set).transform(
            wdm_set_real, window=window,
        )
        d_h_dir = float(analysis.template_inner_product(tpl_direct_wdm))
        h_h_dir = float(analysis.template_snr(tpl_direct_wdm)[0]) ** 2
        logL_dir = d_h_dir - 0.5 * h_h_dir - 0.5 * d_d

        # C++ get_ll direct (Stage 2b): same FD + polyphase + bin-fold
        # as the production MCMC likelihood. Build bin-fold A/B tables
        # from the dense complex WDM of the injection.
        sens_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
        invC_complex = np.asarray(sens_complex.invC).copy()
        A0, A1, B0, B1 = python_bin_fold(
            wdm_ref_complex, wdm_ref_complex, invC_complex,
            n_sparse, stride, Nt_active, tdi_type="XYZ",
        )
        c0_sparse_all = c0_sparse[None, ...].copy()
        d_h_cpp = np.zeros(1, dtype=np.float64)
        h_h_cpp = np.zeros(1, dtype=np.float64)
        _cpp.gb_signal_het_get_ll_in_kernel(
            _tdi_wrap,
            d_h_cpp, h_h_cpp,
            c0_sparse_all,
            A0[None, ...].copy(), A1[None, ...].copy(),
            B0[None, ...].copy(), B1[None, ...].copy(),
            sparse_gen.window_full.astype(np.float64).copy(),
            np.asarray(sparse_gen.n_sparse_local, dtype=np.int32),
            params_cand.astype(np.float64).reshape(1, 9).copy(),
            params_inj.astype(np.float64).reshape(1, 9).copy(),
            np.zeros(1, dtype=np.int32),
            1, 1, 9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            Nt_layer, sparse_gen.N_sparse_t, stride,
            ind_min_t, ind_min_f, 2,
            layer_df, dt, Tobs, t_start, 3, 0,
            1024,        # N_SPARSE_FD (matches v2 default)
            0.05, 5.0,   # tukey_alpha, max_r
        )
        d_h_cpp_v = float(d_h_cpp[0])
        h_h_cpp_v = float(h_h_cpp[0])
        logL_cpp = -0.5 * d_d + d_h_cpp_v - 0.5 * h_h_cpp_v

        print(f"[diag {drawi}]  <d|d>={d_d:+.4e}", flush=True)
        print(f"           v2_het:  <d|h>={d_h:+.4e}  <h|h>={h_h:+.4e}  "
              f"logL={logL:+.4e}", flush=True)
        print(f"          lisat dir: <d|h>={d_h_dir:+.4e}  <h|h>={h_h_dir:+.4e}  "
              f"logL={logL_dir:+.4e}", flush=True)
        print(f"          cpp get_ll:<d|h>={d_h_cpp_v:+.4e}  <h|h>={h_h_cpp_v:+.4e}  "
              f"logL={logL_cpp:+.4e}", flush=True)

        # --- plotting -----------------------------------------------------
        m_floor = int(params_inj[1] / layer_df)
        m_active = np.arange(m_floor - 2, m_floor + 3)
        m_local = m_active - ind_min_f
        # carrier layer index inside the 5-row v2 sparse output (middle row)
        CHAN = 0
        carrier_row_in_sparse = 2

        # col 0: dense real-WDM injection in active band
        ax = axs[drawi, 0]
        inj_band = wdm_inj_arr[CHAN, m_local, :]                  # (5, Nt_act)
        vmax = float(np.abs(inj_band).max())
        im = ax.imshow(
            inj_band, aspect="auto", origin="lower",
            extent=[0, Nt_active, m_floor - 2 - 0.5, m_floor + 2 + 0.5],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        )
        ax.set_xlabel("n (active time pixel)")
        ax.set_ylabel("global m")
        ax.set_title(
            f"draw {drawi}: dense real-WDM injection (chan 0)\n"
            f"f0={params_inj[1]*1e3:.4f}mHz  m_floor={m_floor}  snr={snr_i:.1f}",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # col 1: v2 sparse complex-WDM (real part) at active m on sparse n
        ax = axs[drawi, 1]
        sparse_real = c1_sparse[CHAN].real                        # (5, N_sp)
        smax = float(np.abs(sparse_real).max())
        im = ax.imshow(
            sparse_real, aspect="auto", origin="lower",
            extent=[n_sparse[0], n_sparse[-1],
                    m_active[0] - 0.5, m_active[-1] + 0.5],
            cmap="RdBu_r", vmin=-smax, vmax=smax,
        )
        ax.set_xlabel(f"n (sparse, stride={stride})")
        ax.set_ylabel("global m (active band)")
        ax.set_title(
            f"v2 polyphase: Re(c1_sparse)  "
            f"shape={sparse_real.shape}  Nt_layer={Nt_layer}",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # col 2: r(t) at carrier layer m_floor
        ax = axs[drawi, 2]
        c0_at_sparse_carrier = c0_sparse[CHAN, m_local[carrier_row_in_sparse]]
        c1_at_sparse_carrier = c1_sparse[CHAN, carrier_row_in_sparse]
        mag = np.abs(c0_at_sparse_carrier)
        floor = 1e-12 * mag.max()
        mask = mag > floor
        r = np.where(
            mask, c1_at_sparse_carrier / np.where(mask, c0_at_sparse_carrier, 1.0),
            0.0 + 0.0j,
        )
        # de-rotation
        Df0 = params_cand[1] - params_inj[1]
        Dfdot = params_cand[2] - params_inj[2]
        t_n_sparse = (ind_min_t + n_sparse) * Nf * dt
        phase_pred = (
            2.0 * np.pi * Df0 * t_n_sparse + np.pi * Dfdot * t_n_sparse ** 2
        )
        r_demod = r * np.exp(-1j * phase_pred)

        ax.plot(n_sparse, r.real, "C0o-", label="Re(r)", markersize=3, linewidth=0.8)
        ax.plot(n_sparse, r.imag, "C1o-", label="Im(r)", markersize=3, linewidth=0.8)
        ax.plot(n_sparse, r_demod.real, "C2.--", label="Re(r_demod)", markersize=2,
                linewidth=0.6, alpha=0.7)
        ax.plot(n_sparse, r_demod.imag, "C3.--", label="Im(r_demod)", markersize=2,
                linewidth=0.6, alpha=0.7)
        ax.set_xlabel(f"n_sparse")
        ax.set_ylabel("r")
        ax.set_title(
            f"r = c1/c0 at m_floor={m_floor} (chan 0)\n"
            f"Df0/layer_df = {DF0_FRAC}",
            fontsize=10,
        )
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)

        # col 3: numeric panel
        ax = axs[drawi, 3]
        ax.axis("off")
        amp_str = f"{params_inj[0]:.2e}"
        fdot_str = f"{params_inj[2]:.2e}"
        sinb_str = f"{np.sin(params_inj[8]):+.3f}"
        # logL residuals
        d_logL_v2  = logL    - logL_dir
        d_logL_cpp = logL_cpp - logL_dir
        text = (
            f"DRAW {drawi}\n\n"
            f"params_inj:\n"
            f"  amp     = {amp_str}\n"
            f"  f0      = {params_inj[1]*1e3:.5f} mHz  "
            f"(m_floor={m_floor}, frac={params_inj[1]/layer_df - m_floor:.3f})\n"
            f"  fdot    = {fdot_str}\n"
            f"  beta    = {params_inj[8]:+.3f}  sin(beta)={sinb_str}\n"
            f"  lam     = {params_inj[7]:+.3f}\n"
            f"  snr     = {snr_i:.1f}\n\n"
            f"heterodyne: Df0 = {DF0_FRAC} * layer_df = {Df0*1e6:.3e} uHz\n"
            f"Nt_layer = {Nt_layer}  N_sparse_t = {sparse_gen.N_sparse_t}\n\n"
            f"<d|d>           = {d_d:+.4e}\n"
            f"              {'v2 het(py)':>12s} {'cpp get_ll':>12s} {'lisat direct':>13s}\n"
            f"<d|tpl>    =  {d_h:>+12.4e} {d_h_cpp_v:>+12.4e} {d_h_dir:>+13.4e}\n"
            f"<tpl|tpl>  =  {h_h:>+12.4e} {h_h_cpp_v:>+12.4e} {h_h_dir:>+13.4e}\n"
            f"logL       =  {logL:>+12.4e} {logL_cpp:>+12.4e} {logL_dir:>+13.4e}\n"
            f"DlogL: v2-dir = {d_logL_v2:+.3e}  cpp-dir = {d_logL_cpp:+.3e}\n\n"
            f"mm5       = {mm5:+.3e}\n"
            f"mm2       = {mm2:+.3e}\n"
        )
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=9,
        )
        print(
            f"[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz  snr={snr_i:.1f}  "
            f"logL={logL:+.4e}  mm5={mm5:.3e}  mm2={mm2:.3e}",
            flush=True,
        )
        drawi += 1

    pert_parts = []
    if DF0_FRAC:    pert_parts.append(f"Df0={DF0_FRAC}*layer_df")
    if DFDOT:       pert_parts.append(f"Dfdot={DFDOT:.1e}")
    if DAMP_FRAC:   pert_parts.append(f"Damp/amp={DAMP_FRAC:.1e}")
    if DPHI0:       pert_parts.append(f"Dphi0={DPHI0:.1e}")
    if DINC:        pert_parts.append(f"Dinc={DINC:.1e}")
    if DPSI:        pert_parts.append(f"Dpsi={DPSI:.1e}")
    if DLAM:        pert_parts.append(f"Dlam={DLAM:.1e}")
    if DBETA:       pert_parts.append(f"Dbeta={DBETA:.1e}")
    if not pert_parts:
        pert_parts = ["x_cand = x_inj"]
    pert_str = (PERT_LABEL + "  -- " if PERT_LABEL else "") + ", ".join(pert_parts)
    fig.suptitle(
        f"v2 polyphase + carrier de-rotation walk-through  "
        f"(N_draws={N_DRAWS_PLOT}, SEED={SEED}, Nt_layer={Nt_layer})\n"
        f"{pert_str}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.environ.get("OUT_PNG", "v2_walkthrough.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {out}", flush=True)
    print("DONE.")


if __name__ == "__main__":
    sys.exit(main())
