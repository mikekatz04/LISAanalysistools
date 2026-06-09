#!/usr/bin/env python
"""Test whether amplitude/phase clipping of r = c1/c0 fixes the positive
logL blowup that the (Re, Im) form produces at angle excursions.

Setup:
  * Inject a GB at (f0, lam, beta, ...). Build the reference c0 = WDM(inj).
  * Sweep a candidate that perturbs ONE angle (lam) progressively further.
  * For each candidate, compute c1 = WDM(cand) directly via lisatools and
    form r_sparse = c1_sparse / c0_sparse at the sparse n positions.
  * Compute logL three ways using the SAME bin-fold A0/A1/B0/B1 tables:

      V0  : lisatools direct path (ground truth -- d_h - 0.5*h_h via the
            actual WDM inner product, no heterodyne assumption)
      V1  : current (Re, Im) bin-fold formula on complex r
      V2  : amp/phase formula -- clip |r| per channel-cell to MAX_R, then
            reform r_clipped = |r|_clipped * exp(i*arg(r)) and run the
            same bin-fold sum

  Expected if amp/phase fixes blowup:
    - small Dlam: V1 == V2 (both heterodyne-valid, r ~ 1)
    - large Dlam: V1 -> +1e8/+1e9 (positive blowup), V2 stays bounded
      (logL goes negative, matching the true direction of departure
      from injection)

Run::
    python test_amp_phase_vs_re_im_blowup.py
Env vars:
    DLAM_LIST   default "0.01,0.05,0.1,0.3,0.5,1.0,2.0,3.0"
    MAX_R_LIST  default "2.0,5.0,10.0,50.0"
    F0_MHZ      default 14.22
    SNR         default 50.0
    SEED        default 42
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.diagnostic import inner_product
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen
from gb_signal_het_cpp_validate import python_bin_fold


TUKEY_ALPHA = 0.05


def compute_logL_binfold(r_sparse_active, dr_sparse_active, A0_all, A1_all,
                          B0_all, B1_all, d_d_lt, m_active, ind_min_f,
                          data_idx=0, tdi_type="XYZ"):
    """Apply the current C++-equivalent bin-fold formula on complex r_sparse.

    r_sparse_active, dr_sparse_active: (nch, M, N_sparse_t) complex128
    A0/A1: (1, nch, Nf_active, N_sparse_t) complex128
    B0/B1 (XYZ): (1, nch, nch, Nf_active, N_sparse_t) complex128
    Returns: (logL, d_h_real, h_h_real)  -- d_h, h_h are the *0.5* prefactor
                                            outputs that the C++ produces.
    """
    nch, M, N_sparse_t = r_sparse_active.shape

    d_h_c = 0.0 + 0.0j
    for c in range(nch):
        for im in range(M):
            m_local = m_active[im] - ind_min_f
            d_h_c += np.sum(A0_all[data_idx, c, m_local] * r_sparse_active[c, im]
                            + A1_all[data_idx, c, m_local] * dr_sparse_active[c, im])

    h_h_c = 0.0 + 0.0j
    if tdi_type == "XYZ":
        for c in range(nch):
            for c2 in range(nch):
                for im in range(M):
                    m_local = m_active[im] - ind_min_f
                    r_c   = r_sparse_active[c, im]
                    r_c2  = r_sparse_active[c2, im]
                    dr_c  = dr_sparse_active[c, im]
                    dr_c2 = dr_sparse_active[c2, im]
                    b0    = B0_all[data_idx, c, c2, m_local]
                    b1    = B1_all[data_idx, c, c2, m_local]
                    r_outer = np.conj(r_c) * r_c2
                    cross   = np.conj(r_c) * dr_c2 + np.conj(dr_c) * r_c2
                    h_h_c += np.sum(b0 * r_outer + b1 * cross)
    d_h = 0.5 * d_h_c.real
    h_h = 0.5 * h_h_c.real
    logL = -0.5 * d_d_lt + d_h - 0.5 * h_h
    return logL, d_h, h_h


def main():
    DLAM_LIST = [float(x) for x in
                 os.environ.get("DLAM_LIST",
                                "0.01,0.05,0.1,0.3,0.5,1.0,2.0,3.0").split(",")]
    MAX_R_LIST = [float(x) for x in
                  os.environ.get("MAX_R_LIST", "2.0,5.0,10.0,50.0").split(",")]
    F0_MHZ = float(os.environ.get("F0_MHZ", "14.22"))
    SNR    = float(os.environ.get("SNR", "50.0"))
    SEED   = int(os.environ.get("SEED", "42"))
    np.random.seed(SEED)

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
    window = _tukey(Nobs, alpha=TUKEY_ALPHA).astype(float)
    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start, min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start, min_freq=1e-4, max_freq=35e-3,
        min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
        is_complex=True, force_backend=backend,
    )
    layer_df = wdm_set_real.layer_df
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)
    Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
    ind_min_f = int(wdm_set_real.ind_min_f)

    # ----- Injection (calibrate amp to SNR) ----- #
    f0_inj   = F0_MHZ * 1e-3
    fdot_inj = 1e-16
    inc_inj  = np.pi / 3.0
    psi_inj  = 0.7
    phi0_inj = 1.4
    lam_inj  = 2.1
    beta_inj = 0.5
    amp_probe = 1e-22
    pp = np.array([amp_probe, f0_inj, fdot_inj, 0.0, phi0_inj,
                    inc_inj, psi_inj, lam_inj, beta_inj])
    wdm_probe = TDSignal(real_td_cb(pp), settings=td_set).transform(
        wdm_set_real, window=window)
    sens_mat_real = XYZ2SensitivityMatrix(
        DataResidualArray(wdm_probe).data_res_arr.settings, model="scirdv1")
    snr_probe = float(AnalysisContainer(DataResidualArray(wdm_probe),
                                         sens_mat_real).snr())
    amp_inj = amp_probe * SNR / snr_probe
    params_inj = np.array([amp_inj, f0_inj, fdot_inj, 0.0, phi0_inj,
                            inc_inj, psi_inj, lam_inj, beta_inj])

    td_inj = real_td_cb(params_inj)
    wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(
        wdm_set_real, window=window)
    inj_data = DataResidualArray(wdm_inj_real)
    analysis = AnalysisContainer(inj_data, sens_mat_real)
    snr_i = float(analysis.snr())
    d_d_lt = float(np.real(analysis.inner_product()))
    print(f"[inj] amp={amp_inj:.3e} f0={f0_inj*1e3:.4f}mHz "
          f"lam={lam_inj:.3f} snr={snr_i:.1f} <d|d>={d_d_lt:.3e}",
          flush=True)

    # Reference c0 + bin-fold tables.
    wdm_inj_complex = np.asarray(
        TDSignal(td_inj, settings=td_set).transform(
            wdm_set_complex, window=window).arr)
    c0_dense_active = wdm_inj_complex
    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=64, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    c0_sparse_active = c0_dense_active[:, :, n_sparse_local]
    sens_mat_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
    invC_complex = np.asarray(sens_mat_complex.invC)
    A0, A1, B0, B1 = python_bin_fold(
        wdm_inj_complex, c0_dense_active, invC_complex,
        n_sparse_local, stride, Nt_active, tdi_type="XYZ",
    )
    A0_all = A0[None, ...]
    A1_all = A1[None, ...]
    B0_all = B0[None, ...]
    B1_all = B1[None, ...]

    # m_active for the injection (held fixed -- f0 doesn't move in this sweep).
    layer_df = wdm_set_real.layer_df
    m_floor_inj = int(np.floor(f0_inj / layer_df))
    m_active = np.array([m_floor_inj - 2, m_floor_inj - 1, m_floor_inj,
                          m_floor_inj + 1, m_floor_inj + 2])
    M = m_active.size

    # Pre-build the V3 regularized B0 tables. Two flavors:
    #  per-cell : B0_diag += eps * trace(M_{m,b})/3  -- proportional to local
    #             cell amplitude (does NOT help at outlier cells where c0~0).
    #  global   : B0_diag += eps * max|B0|_global * I  -- absolute lower
    #             bound on h_h that catches the null-space collapse at
    #             every cell.
    # The script runs both and reports them as V3p / V3g.
    EPS_REG_LIST = [float(x) for x in
                    os.environ.get("EPS_REG_LIST", "0.01,0.1,1.0").split(",")]
    trace3_per_cell = (B0[0, 0] + B0[1, 1] + B0[2, 2]).real / 3.0  # (Nf_a, N_sparse_t)
    B0_diag_max_global = float(
        np.max(np.abs(np.stack([B0[c, c] for c in range(3)]))))
    print(f"[reg] max |B0_diag|_global = {B0_diag_max_global:.3e}",
          flush=True)
    B0_reg_per_cell = {}
    B0_reg_global   = {}
    for eps_reg in EPS_REG_LIST:
        B0p = B0_all.copy()
        B0g = B0_all.copy()
        for c in range(3):
            B0p[0, c, c] = B0[c, c] + eps_reg * np.abs(trace3_per_cell)
            B0g[0, c, c] = B0[c, c] + eps_reg * B0_diag_max_global
        B0_reg_per_cell[eps_reg] = B0p
        B0_reg_global  [eps_reg] = B0g

    # ===== Sweep dlam, compute logL multiple ways ===== #
    print(f"\nLegend: V0=truth (lisatools direct), V1=(Re,Im) bin-fold,",
          flush=True)
    print(f"        V2[R]=clip |r|<R, V3p[e]=per-cell reg (proportional to local M),",
          flush=True)
    print(f"        V3g[e]=global-floor reg (eps * max|B0|_global * I)",
          flush=True)
    print(f"\n  {'candidate':<14s} {'snr':>6s} "
          f"{'V0_truth':>13s} {'V1_ReIm':>13s} "
          f"{'max|r|':>9s}  " +
          "  ".join(f"{'V2[R=' + f'{r:g}]':>13s}" for r in MAX_R_LIST) +
          "  " +
          "  ".join(f"{'V3p[' + f'{e:g}]':>13s}" for e in EPS_REG_LIST) +
          "  " +
          "  ".join(f"{'V3g[' + f'{e:g}]':>13s}" for e in EPS_REG_LIST),
          flush=True)

    # Build the candidate list. If TEST_BAD_POINT=1, append the exact
    # bad-MCMC point that produced logL = +3.7e9 in the prior run, so we
    # can see whether amp/phase or regularization rescue it.
    candidates = []
    for dlam in DLAM_LIST:
        p = params_inj.copy()
        p[7] = (params_inj[7] + dlam) % (2.0 * np.pi)
        candidates.append((f"dlam={dlam:.2f}", p))
    if os.environ.get("TEST_BAD_POINT", "1") == "1":
        # Peak walker from the wide-prior MCMC blowup
        p_bad = np.array([
            1.21033703e-22,  # amp
            1.41896878e-02,  # f0  (still within injection's WDM layer)
            -2.15144509e-16, # fdot (sign-flipped)
            0.0,             # fddot
            0.98072821e+00,  # phi0
            np.arccos(0.87986477e+00),  # inc from cosinc=0.880
            2.99977945e+00,  # psi
            7.22774541e-02,  # lam (very different from inj 2.1)
            np.arcsin(0.82528623e+00),  # beta from sinbeta=0.825
        ])
        candidates.append(("BAD_MCMC_PT", p_bad))

    for label, params_cand in candidates:
        dlam = params_cand[7] - params_inj[7]
        td_cand = real_td_cb(params_cand)

        # V0: ground truth via lisatools direct
        wdm_cand_real = TDSignal(td_cand, settings=td_set).transform(
            wdm_set_real, window=window)
        snr_cand = float(AnalysisContainer(DataResidualArray(wdm_cand_real),
                                            sens_mat_real).snr())
        d_h_dir = float(np.real(analysis.template_inner_product(
            wdm_cand_real, complex=True)))
        h_h_dir = float(np.real(inner_product(
            wdm_cand_real, wdm_cand_real, psd=sens_mat_real)))
        logL_direct = -0.5 * d_d_lt + d_h_dir - 0.5 * h_h_dir
        # Note: direct's "d_h" really is <d|h_cand>; the formula above uses
        # the calibration where logL = 0 at h_cand = data.

        # Compute c1_sparse using direct WDM transform of candidate (so
        # the test is purely about the (Re, Im) -> (A, phi) reformulation,
        # not about the polyphase reconstruction).
        wdm_cand_complex = np.asarray(
            TDSignal(td_cand, settings=td_set).transform(
                wdm_set_complex, window=window).arr)
        c1_dense_active = wdm_cand_complex
        c1_sparse_active = c1_dense_active[:, :, n_sparse_local]

        # r_sparse on the 5 active layers around m_floor_inj.
        FLOOR_EPS = 1e-12
        r_sparse = np.zeros((3, M, N_sparse_t), dtype=np.complex128)
        for c in range(3):
            for im in range(M):
                m_local = m_active[im] - ind_min_f
                c0v = c0_sparse_active[c, m_local]
                c1v = c1_sparse_active[c, m_local]
                max_mag = np.max(np.abs(c0v))
                floor_th = max(FLOOR_EPS * max_mag, 1e-300)
                safe = np.abs(c0v) > floor_th
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(safe, c1v / c0v, 0.0 + 0.0j)
                r_sparse[c, im] = r

        # dr/dn via centred FD
        Dn = float(stride)
        dr_sparse = np.zeros_like(r_sparse)
        dr_sparse[:, :, 1:-1] = (r_sparse[:, :, 2:] - r_sparse[:, :, :-2]) / (2 * Dn)
        dr_sparse[:, :,  0]    = (r_sparse[:, :,  1] - r_sparse[:, :,  0]) / Dn
        dr_sparse[:, :, -1]    = (r_sparse[:, :, -1] - r_sparse[:, :, -2]) / Dn

        max_abs_r = float(np.max(np.abs(r_sparse)))

        # V1: current (Re, Im) bin-fold
        logL_v1, d_h_v1, h_h_v1 = compute_logL_binfold(
            r_sparse, dr_sparse, A0_all, A1_all, B0_all, B1_all,
            d_d_lt, m_active, ind_min_f, data_idx=0, tdi_type="XYZ")

        # V2: clip |r| per channel-cell at each MAX_R then re-run bin-fold
        logL_v2_list = []
        for max_R in MAX_R_LIST:
            A_r = np.abs(r_sparse)
            phi_r = np.angle(r_sparse)
            A_clipped = np.minimum(A_r, max_R)
            scale = np.where(A_r > 0, A_clipped / np.maximum(A_r, 1e-300), 1.0)
            r_clipped  = A_clipped * np.exp(1j * phi_r)
            dr_clipped = dr_sparse * scale
            logL_v2, d_h_v2, h_h_v2 = compute_logL_binfold(
                r_clipped, dr_clipped, A0_all, A1_all, B0_all, B1_all,
                d_d_lt, m_active, ind_min_f, data_idx=0, tdi_type="XYZ")
            logL_v2_list.append(logL_v2)

        # V3p: per-cell regularization (proportional to local trace)
        logL_v3p_list = []
        for eps_reg in EPS_REG_LIST:
            B0_reg = B0_reg_per_cell[eps_reg]
            logL_v3p, _, _ = compute_logL_binfold(
                r_sparse, dr_sparse, A0_all, A1_all, B0_reg, B1_all,
                d_d_lt, m_active, ind_min_f, data_idx=0, tdi_type="XYZ")
            logL_v3p_list.append(logL_v3p)

        # V3g: global-floor regularization (absolute lower bound on h_h)
        logL_v3g_list = []
        for eps_reg in EPS_REG_LIST:
            B0_reg = B0_reg_global[eps_reg]
            logL_v3g, _, _ = compute_logL_binfold(
                r_sparse, dr_sparse, A0_all, A1_all, B0_reg, B1_all,
                d_d_lt, m_active, ind_min_f, data_idx=0, tdi_type="XYZ")
            logL_v3g_list.append(logL_v3g)

        print(f"  {label:<14s} {snr_cand:>6.1f} "
              f"{logL_direct:>+13.3e} {logL_v1:>+13.3e} "
              f"{max_abs_r:>9.2e}  " +
              "  ".join(f"{l:>+13.3e}" for l in logL_v2_list) +
              "  " +
              "  ".join(f"{l:>+13.3e}" for l in logL_v3p_list) +
              "  " +
              "  ".join(f"{l:>+13.3e}" for l in logL_v3g_list),
              flush=True)

    print("\n[done] amp/phase clip test complete.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
