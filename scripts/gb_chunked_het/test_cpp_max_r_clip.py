#!/usr/bin/env python
"""Verify the C++ max_r clip fixes the positive-logL blowup at the bad MCMC point.

Runs gb_signal_het_get_ll_in_kernel on:
  (A) the injection (logL should be ~0 regardless of max_r)
  (B) a 1e-3 layer_df Df0 perturbation (heterodyne-valid)
  (C) the exact +3.68e9 bad MCMC walker

For each, evaluates with max_r in {0, 5, 10}:
  max_r = 0  -> clipping disabled, preserves pre-fix behavior
  max_r > 0  -> caps |r| per channel-cell

Expected:
  - At injection: logL ~ 0 for all max_r (clip never binds)
  - At Df0 perturbation: same logL for max_r=0 and max_r > 0 (|r| small)
  - At BAD point: max_r=0 -> +3.68e9 (blowup); max_r > 0 -> bounded negative
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
import lisatools_backend_cpu.pycppdetector as _lat_pd
import gbgpu_backend_cpu.cgbgpu as _be  # GBComputationGroupWrap lives here post-3L.7g

from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen
from gb_signal_het_cpp_validate import python_bin_fold


def main():
    TUKEY_ALPHA = 0.05
    NT_LAYER = 64
    N_SPARSE_FD = 1024
    F0_MHZ = 14.22

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
    tdi_wrap = gb_gen.wave_gen

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

    # ----- Injection -----
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
    amp_inj = amp_probe * 50.0 / snr_probe

    params_inj = np.array([amp_inj, f0_inj, fdot_inj, 0.0, phi0_inj,
                            inc_inj, psi_inj, lam_inj, beta_inj])
    td_inj = real_td_cb(params_inj)
    wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(
        wdm_set_real, window=window)
    inj_data = DataResidualArray(wdm_inj_real)
    analysis = AnalysisContainer(inj_data, sens_mat_real)
    snr_i = float(analysis.snr())
    d_d_lt = float(np.real(analysis.inner_product()))
    print(f"[inj] amp={amp_inj:.3e} snr={snr_i:.1f} <d|d>={d_d_lt:.3e}",
          flush=True)

    # Bin-fold tables.
    wdm_inj_complex = np.asarray(
        TDSignal(td_inj, settings=td_set).transform(
            wdm_set_complex, window=window).arr)
    c0_dense_active = wdm_inj_complex
    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=NT_LAYER, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    window_full = sparse_gen.window_full.astype(np.float64).copy()
    c0_sparse_active = c0_dense_active[:, :, n_sparse_local]
    sens_mat_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
    invC_complex = np.asarray(sens_mat_complex.invC).copy()
    A0, A1, B0, B1 = python_bin_fold(
        wdm_inj_complex, c0_dense_active, invC_complex,
        n_sparse_local, stride, Nt_active, tdi_type="XYZ",
    )
    c0_sparse_all = c0_sparse_active[None, ...].copy()
    A0_all = A0[None, ...].copy()
    A1_all = A1[None, ...].copy()
    B0_all = B0[None, ...].copy()
    B1_all = B1[None, ...].copy()
    params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
    data_index_all = np.zeros(1, dtype=np.int32)
    cpp = _be.GBComputationGroupWrapCPU()

    def run(params, max_r):
        params_cand_all = params.astype(np.float64).reshape(1, 9).copy()
        d_h = np.zeros(1, dtype=np.float64)
        h_h = np.zeros(1, dtype=np.float64)
        cpp.gb_signal_het_get_ll_in_kernel(
            tdi_wrap, d_h, h_h,
            c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, params_ref_all, data_index_all,
            1, 1, 9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            NT_LAYER, N_sparse_t, stride, ind_min_t, ind_min_f, 2,
            layer_df, dt, Tobs, t_start, 3, 0,
            N_SPARSE_FD, TUKEY_ALPHA, float(max_r),
        )
        return -0.5 * d_d_lt + float(d_h[0]) - 0.5 * float(h_h[0])

    # Candidates
    params_perturb = params_inj.copy()
    params_perturb[1] = params_inj[1] + 1e-3 * layer_df
    params_bad = np.array([
        1.21033703e-22, 1.41896878e-02, -2.15144509e-16, 0.0,
        0.98072821e+00, np.arccos(0.87986477e+00),
        2.99977945e+00, 7.22774541e-02,
        np.arcsin(0.82528623e+00),
    ])

    print(f"\n  {'candidate':<18s} {'max_r':>7s} {'logL':>16s}",
          flush=True)
    for label, p in [("injection", params_inj),
                     ("Df0=1e-3*layer_df", params_perturb),
                     ("BAD_MCMC_PT", params_bad)]:
        for max_r in [0.0, 5.0, 10.0]:
            ll = run(p, max_r)
            print(f"  {label:<18s} {max_r:>7.1f} {ll:>+16.4e}", flush=True)
        print()

    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
