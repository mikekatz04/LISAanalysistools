#!/usr/bin/env python
"""Validate gb_signal_het_get_ll_grad_in_kernel against Python finite differences.

The C++ grad runs central differences in C++:
  grad[k] = (logL(params + eps_k * e_k) - logL(params - eps_k * e_k)) / (2 eps_k)

The Python reference does the same loop but calls
gb_signal_het_get_ll_in_kernel from Python for each (+eps, -eps) build.
Since both use the SAME C++ get_ll_in_kernel under the hood, the gradients
must agree at FP precision (any difference is loop-order rounding noise).

Run::
    python gb_signal_het_cpp_validate_grad.py
Env vars:
    N_DRAWS         default 2
    NT_LAYER        default 64
    N_SPARSE_FD     default 1024
    SEED            default 54321

Future extension (when GB signal-het JAX path lands): add a jax.grad
comparison column. Right now there's no JAX implementation of the v2
polyphase signal-het, only Python prototype + C++ stages 1/2a/2b +
fill_global + this grad.
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
from gb_signal_het_wdm_v2_mm_sweep import build_gb_prior
from gb_signal_het_cpp_validate import python_bin_fold


def call_get_ll(cpp, tdi_wrap, params_cand, params_ref,
                c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local, data_index_all,
                Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, layer_df, dt, Tobs, t_start,
                N_SPARSE_FD, tukey_alpha):
    """Thin wrapper that calls C++ get_ll_in_kernel for a single binary."""
    d_h = np.zeros(1, dtype=np.float64)
    h_h = np.zeros(1, dtype=np.float64)
    params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()
    cpp.gb_signal_het_get_ll_in_kernel(
        tdi_wrap,
        d_h, h_h,
        c0_sparse_all,
        A0_all, A1_all, B0_all, B1_all,
        window_full, n_sparse_local,
        params_cand_all, params_ref, data_index_all,
        1, 1,
        9, 1, 2,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        2,
        layer_df, dt,
        Tobs, t_start,
        3, 0, N_SPARSE_FD,
        tukey_alpha,
    )
    return float(d_h[0]), float(h_h[0])


def main():
    N_DRAWS = int(os.environ.get("N_DRAWS", "2"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    N_SPARSE_FD = int(os.environ.get("N_SPARSE_FD", "1024"))
    SEED = int(os.environ.get("SEED", "54321"))
    TUKEY_ALPHA = 0.05

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
    print(f"[grid] Nf={Nf} Nt={Nt} layer_df={layer_df:.3e} "
          f"Nt_active={Nt_active} Nf_active={Nf_active}", flush=True)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex,
        data_dt=dt, ind_min_t=ind_min_t, Nt_active=Nt_active,
        Nt_layer=Nt_layer, m_active_half_width=2,
    )
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    print(f"[v2] Nt_layer={Nt_layer} stride={stride} N_sparse_t={N_sparse_t} "
          f"N_sparse_fd={N_SPARSE_FD} tukey_alpha={TUKEY_ALPHA}", flush=True)

    np.random.seed(SEED)
    f0_lo = (ind_min_f + 7) * layer_df
    f0_hi = 0.025
    prior, tc = build_gb_prior(
        A_lims=(1e-23, 1e-20), f0_lims_hz=(f0_lo, f0_hi),
        fdot_lims=(-1e-15, 1e-15), beta_lims=None,
    )

    SNR_MIN, SNR_MAX = 5.0, 1100.0
    MAX_REJECT = 500
    cpp = _be.GBComputationGroupWrapCPU()
    sens_mat = None
    window_full = sparse_gen.window_full.astype(np.float64).copy()

    # Per-param eps: amp ~ relative, f0/fdot ~ small layer/Tobs-driven,
    # angles ~ 1e-4 rad. Order: amp, f0, fdot, fddot, phi0, inc, psi, lam, beta.
    PARAM_EPS = np.array([
        1e-3,        # amp (will be scaled by amp itself below)
        1e-3 * layer_df,  # f0   ~ 3.4e-8 Hz
        1e-18,       # fdot ~ slow
        0.0,         # fddot (frozen; usually unused)
        1e-3,        # phi0
        1e-3,        # inc
        1e-3,        # psi
        1e-3,        # lam
        1e-3,        # beta
    ], dtype=np.float64)
    PARAM_NAMES = ["amp", "f0", "fdot", "fddot", "phi0",
                   "inc", "psi", "lam", "beta"]

    drawi = 0
    while drawi < N_DRAWS:
        for _ in range(MAX_REJECT):
            x_samp = prior.rvs(size=1)
            params_inj = tc.both_transforms(x_samp.copy())[0]
            td_inj = real_td_cb(params_inj)
            wdm_inj_real = TDSignal(td_inj, settings=td_set).transform(
                wdm_set_real, window=window)
            inj_data_arr = DataResidualArray(wdm_inj_real)
            if sens_mat is None:
                sens_mat = XYZ2SensitivityMatrix(
                    inj_data_arr.data_res_arr.settings, model="scirdv1")
            analysis = AnalysisContainer(inj_data_arr, sens_mat)
            snr_i = float(analysis.snr())
            if SNR_MIN <= snr_i <= SNR_MAX:
                break

        wdm_inj_complex = np.asarray(
            TDSignal(td_inj, settings=td_set).transform(
                wdm_set_complex, window=window).arr)
        c0_dense_complex = wdm_inj_complex.copy()
        c0_sparse_complex = c0_dense_complex[:, :, n_sparse_local].copy()
        sens_complex = XYZ2SensitivityMatrix(wdm_set_complex, model="scirdv1")
        invC_complex = np.asarray(sens_complex.invC).copy()
        A0, A1, B0, B1 = python_bin_fold(
            wdm_inj_complex, c0_dense_complex, invC_complex,
            n_sparse_local, stride, Nt_active, tdi_type="XYZ",
        )

        c0_sparse_all = c0_sparse_complex[None, ...].copy()
        A0_all = A0[None, ...].copy()
        A1_all = A1[None, ...].copy()
        B0_all = B0[None, ...].copy()
        B1_all = B1[None, ...].copy()
        params_ref_all = params_inj.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)

        # Candidate slightly off injection so the gradient is non-trivial.
        params_cand = params_inj.copy()
        params_cand[1] = params_inj[1] + 1e-3 * layer_df   # 1e-3 layer Df0
        params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()

        # Per-parameter epsilons; amp is relative -> scale by amp_cand.
        eps_arr = PARAM_EPS.copy()
        eps_arr[0] = max(abs(params_cand[0]) * 1e-3, 1e-30)

        # ---- A) C++ grad ----
        grad_cpp = np.zeros((1, 9), dtype=np.float64)
        d_h_central_cpp = np.zeros(1, dtype=np.float64)
        h_h_central_cpp = np.zeros(1, dtype=np.float64)
        cpp.gb_signal_het_get_ll_grad_in_kernel(
            tdi_wrap,
            grad_cpp, d_h_central_cpp, h_h_central_cpp,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, params_ref_all, data_index_all,
            eps_arr,
            1, 1,
            9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            2,
            layer_df, dt,
            Tobs, t_start,
            3, 0, N_SPARSE_FD,
            TUKEY_ALPHA,
        )
        ll_central_cpp = float(d_h_central_cpp[0]) - 0.5 * float(h_h_central_cpp[0])

        # ---- B) Python finite-difference grad ----
        # Call C++ get_ll_in_kernel from Python at +eps, -eps, central.
        d_h_c, h_h_c = call_get_ll(
            cpp, tdi_wrap, params_cand, params_ref_all,
            c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local, data_index_all,
            Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f, layer_df, dt, Tobs, t_start,
            N_SPARSE_FD, TUKEY_ALPHA,
        )
        ll_central_py = d_h_c - 0.5 * h_h_c

        grad_py = np.zeros(9, dtype=np.float64)
        for k in range(9):
            if eps_arr[k] <= 0:
                continue
            params_plus = params_cand.copy()
            params_plus[k] = params_cand[k] + eps_arr[k]
            d_h_p, h_h_p = call_get_ll(
                cpp, tdi_wrap, params_plus, params_ref_all,
                c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local, data_index_all,
                Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, layer_df, dt, Tobs, t_start,
                N_SPARSE_FD, TUKEY_ALPHA,
            )
            ll_p = d_h_p - 0.5 * h_h_p

            params_minus = params_cand.copy()
            params_minus[k] = params_cand[k] - eps_arr[k]
            d_h_m, h_h_m = call_get_ll(
                cpp, tdi_wrap, params_minus, params_ref_all,
                c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
                window_full, n_sparse_local, data_index_all,
                Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, layer_df, dt, Tobs, t_start,
                N_SPARSE_FD, TUKEY_ALPHA,
            )
            ll_m = d_h_m - 0.5 * h_h_m

            grad_py[k] = (ll_p - ll_m) / (2.0 * eps_arr[k])

        # ---- Compare ----
        print(f"\n[draw {drawi}] f0={params_inj[1]*1e3:.4f}mHz snr={snr_i:.1f}",
              flush=True)
        print(f"  logL central (C++)={ll_central_cpp:+.6e} "
              f"(Py)={ll_central_py:+.6e} "
              f"diff={ll_central_cpp - ll_central_py:+.3e}", flush=True)
        print(f"  {'param':>6s} {'eps':>11s} "
              f"{'grad_cpp':>14s} {'grad_py':>14s} "
              f"{'abs_diff':>11s} {'reldiff':>11s}", flush=True)
        for k in range(9):
            denom = max(abs(grad_py[k]), abs(grad_cpp[0, k]), 1.0)
            rd = abs(grad_cpp[0, k] - grad_py[k]) / denom
            print(f"  {PARAM_NAMES[k]:>6s} {eps_arr[k]:>11.3e} "
                  f"{grad_cpp[0, k]:>+14.6e} {grad_py[k]:>+14.6e} "
                  f"{abs(grad_cpp[0, k] - grad_py[k]):>11.3e} {rd:>11.3e}",
                  flush=True)

        drawi += 1

    print("\n[done] grad C++ vs Python FD validation complete.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
