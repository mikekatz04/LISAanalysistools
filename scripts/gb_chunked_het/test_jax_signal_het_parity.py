#!/usr/bin/env python
"""Validate JAX signal-het kernel against C++ Stage 2a.

Compares ``gb_signal_het_get_ll_sparse_jax`` against the C++
``gb_signal_het_get_ll_sparse`` at the same inputs.

Steps:
  1. Build the standard injection + bin-fold tables.
  2. Build X_het from a candidate via Python (slicing the dense rfft).
  3. Run both JAX and C++ versions; compare d_h, h_h, logL.
  4. Repeat across Df0 perturbations + the BAD_MCMC point.

If everything is bit-exact (reldiff <= 1e-12), the JAX mirror correctly
reproduces the C++ forward path. That's the precondition for the next
step: validate jax.grad against C++ central-difference grad.

Run::
    python test_jax_signal_het_parity.py
Env vars:
    SEED        default 42
    F0_MHZ      default 14.22
    SNR         default 50
    MAX_R       default 5.0
"""

from __future__ import annotations

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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

from gbgpu.jax.wdm.signal_het_kernels import gb_signal_het_get_ll_sparse_jax


def main():
    SEED   = int(os.environ.get("SEED", "42"))
    F0_MHZ = float(os.environ.get("F0_MHZ", "14.22"))
    SNR    = float(os.environ.get("SNR", "50.0"))
    MAX_R  = float(os.environ.get("MAX_R", "5.0"))
    TUKEY_ALPHA = 0.05
    NT_LAYER    = 64
    N_SPARSE_FD = 1024

    np.random.seed(SEED)

    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nt * Nf * dt
    df_abs = 1.0 / Tobs

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

    # Injection
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
    analysis = AnalysisContainer(DataResidualArray(wdm_inj_real),
                                  sens_mat_real)
    d_d_lt = float(np.real(analysis.inner_product()))
    print(f"[inj] f0={f0_inj*1e3:.4f}mHz <d|d>={d_d_lt:.3e}", flush=True)

    # Bin-fold tables
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
    invC_complex = np.asarray(sens_mat_complex.invC)
    A0, A1, B0, B1 = python_bin_fold(
        wdm_inj_complex, c0_dense_active, invC_complex,
        n_sparse_local, stride, Nt_active, tdi_type="XYZ",
    )
    c0_sparse_all = c0_sparse_active[None, ...].copy()
    A0_all = A0[None, ...].copy()
    A1_all = A1[None, ...].copy()
    B0_all = B0[None, ...].copy()
    B1_all = B1[None, ...].copy()

    cpp = _be.GBComputationGroupWrapCPU()

    # Build X_het for a candidate by slicing the dense rfft (avoids
    # needing FD generation in JAX -- pure consumer test).
    def build_X_het(params_cand):
        td_cand = real_td_cb(params_cand)
        fd_rfft = np.fft.rfft(td_cand * window, axis=-1).astype(np.complex128)
        n_rfft = fd_rfft.shape[-1]
        f0_cand = float(params_cand[1])
        k_f0 = int(round(f0_cand / df_abs))
        half_NS = N_SPARSE_FD // 2
        X_het = np.zeros((1, 3, N_SPARSE_FD), dtype=np.complex128)
        for c in range(3):
            for i in range(N_SPARSE_FD):
                k_abs = k_f0 + (i - half_NS)
                if 0 <= k_abs < n_rfft:
                    X_het[0, c, i] = fd_rfft[c, k_abs]
        k_f0_all = np.array([k_f0], dtype=np.int32)
        return X_het, k_f0_all

    # Test points
    points = [
        ("inject",            params_inj.copy()),
        ("Df0=1e-3*layer_df", params_inj.copy()),
    ]
    points[1][1][1] += 1e-3 * layer_df

    print(f"\n  {'case':<22s} {'logL_cpp':>14s} {'logL_jax':>14s} "
          f"{'abs_diff':>11s} {'reldiff':>11s}", flush=True)
    for label, params_cand in points:
        X_het, k_f0_all_np = build_X_het(params_cand)
        params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)

        # ---- C++ ----
        d_h_cpp = np.zeros(1, dtype=np.float64)
        h_h_cpp = np.zeros(1, dtype=np.float64)
        cpp.gb_signal_het_get_ll_sparse(
            d_h_cpp, h_h_cpp,
            X_het, k_f0_all_np,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, params_inj.reshape(1, 9).copy(),
            data_index_all,
            1, 1, 9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            NT_LAYER, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            2,
            layer_df, dt,
            3, 0, N_SPARSE_FD, MAX_R,
        )
        logL_cpp = -0.5 * d_d_lt + float(d_h_cpp[0]) - 0.5 * float(h_h_cpp[0])

        # ---- JAX ----
        d_h_j, h_h_j = gb_signal_het_get_ll_sparse_jax(
            jnp.asarray(X_het),
            jnp.asarray(k_f0_all_np),
            jnp.asarray(c0_sparse_all),
            jnp.asarray(A0_all),
            jnp.asarray(A1_all),
            jnp.asarray(B0_all),
            jnp.asarray(B1_all),
            jnp.asarray(window_full),
            jnp.asarray(n_sparse_local),
            jnp.asarray(params_cand_all),
            jnp.asarray(data_index_all),
            nparams=9, f0_idx=1,
            Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_layer=NT_LAYER,
            N_sparse_t=N_sparse_t, stride=stride,
            ind_min_t=ind_min_t, ind_min_f=ind_min_f,
            m_active_half_width=2,
            layer_df=layer_df, dt=dt,
            nchannels=3, tdi_type=0,
            N_sparse_fd=N_SPARSE_FD,
            max_r=MAX_R,
        )
        logL_jax = -0.5 * d_d_lt + float(d_h_j[0]) - 0.5 * float(h_h_j[0])

        abs_diff = abs(logL_cpp - logL_jax)
        reldiff  = abs_diff / max(abs(logL_cpp), abs(logL_jax), 1.0)
        print(f"  {label:<22s} {logL_cpp:>+14.6e} {logL_jax:>+14.6e} "
              f"{abs_diff:>11.3e} {reldiff:>11.3e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
