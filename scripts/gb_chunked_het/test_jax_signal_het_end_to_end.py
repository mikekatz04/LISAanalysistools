#!/usr/bin/env python
"""Validate JAX end-to-end signal-het (params -> logL) against C++ Stage 2b.

This is the test that matters: with FULL JAX pipeline, jax.grad becomes
the analytic gradient — used as NUTS's grad_log_like_fn in ~1-2x the cost
of one forward call, vs 17x for the C++ central-difference grad.

Steps:
  1. Build injection + bin-fold tables.
  2. For each candidate, run:
       - C++ get_ll_in_kernel(max_r=5)
       - JAX get_ll_in_kernel_jax(max_r=5)
  3. Compare logL forward parity.
  4. Optional: compute jax.grad and compare to C++ central-difference grad.

If forward parity is FP-precision (~1e-12 reldiff or better), the JAX
mirror is the right shape to wire into NUTS.

Run::
    python test_jax_signal_het_end_to_end.py
Env vars:
    F0_MHZ   default 14.22
    SNR      default 50
    MAX_R    default 5.0
    DO_GRAD  default 1 -- also test jax.grad vs C++ central-diff
"""

from __future__ import annotations

import os
import sys
import time

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

from gbgpu.jax.wdm.signal_het_kernels import gb_signal_het_get_ll_in_kernel_jax
from gbgpu.jax.sources.ucb import JaxUCBSource
from lisatools.jax.orbits import OrbitsWrapJAX
from lisatools.jax.response.tdi_config import TDIConfigWrapJAX


def main():
    F0_MHZ = float(os.environ.get("F0_MHZ", "14.22"))
    SNR    = float(os.environ.get("SNR", "50.0"))
    MAX_R  = float(os.environ.get("MAX_R", "5.0"))
    DO_GRAD = os.environ.get("DO_GRAD", "1") == "1"
    TUKEY_ALPHA = 0.05
    NT_LAYER    = 64
    N_SPARSE_FD = 1024

    np.random.seed(42)
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nt * Nf * dt

    orbits_cpp = ESAOrbits(force_backend=backend)
    tdi_config_cpp = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(
        t_tdi, Tobs, t_start, 1.0 / dt, 1,
        tdi_config=tdi_config_cpp, orbits=orbits_cpp, tdi_chan="XYZ",
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

    # Injection (calibrate amp to SNR)
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
    A0_all = A0[None, ...]; A1_all = A1[None, ...]
    B0_all = B0[None, ...]; B1_all = B1[None, ...]

    cpp = _be.GBComputationGroupWrapCPU()

    # JAX source / orbits / tdi_config -- configure CPU orbits first then
    # splat the pycppdetector args into the JAX wrap.
    try:
        orbits_cpp.configure(t_arr=t_arr, dt=dt, linear_interp_setup=True)
    except TypeError:
        orbits_cpp.configure(t_arr=t_arr)
    source_jax = JaxUCBSource(t_ref=t_start)
    orbits_jax = OrbitsWrapJAX(*orbits_cpp.pycppdetector_args)
    tdi_config_jax = TDIConfigWrapJAX(*tdi_config_cpp.pytdiconfig_args)

    # Test points
    points = [
        ("inject",            params_inj.copy()),
        ("Df0=1e-3*layer_df", params_inj.copy()),
    ]
    points[1][1][1] += 1e-3 * layer_df

    print(f"[setup] SNR={SNR} <d|d>={d_d_lt:.3e} max_r={MAX_R}", flush=True)
    print(f"\n  {'case':<22s} {'logL_cpp':>14s} {'logL_jax':>14s} "
          f"{'abs_diff':>11s} {'reldiff':>11s}", flush=True)

    for label, params_cand in points:
        params_cand_all = params_cand.astype(np.float64).reshape(1, 9).copy()
        data_index_all = np.zeros(1, dtype=np.int32)

        # ---- C++ ----
        d_h_cpp = np.zeros(1, dtype=np.float64)
        h_h_cpp = np.zeros(1, dtype=np.float64)
        cpp.gb_signal_het_get_ll_in_kernel(
            tdi_wrap, d_h_cpp, h_h_cpp,
            c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
            window_full, n_sparse_local,
            params_cand_all, params_inj.reshape(1, 9).copy(), data_index_all,
            1, 1, 9, 1, 2,
            Nf, Nt, Nf_active, Nt_active,
            NT_LAYER, N_sparse_t, stride, ind_min_t, ind_min_f, 2,
            layer_df, dt, Tobs, t_start, 3, 0, N_SPARSE_FD,
            TUKEY_ALPHA, MAX_R,
        )
        logL_cpp = -0.5 * d_d_lt + float(d_h_cpp[0]) - 0.5 * float(h_h_cpp[0])

        # ---- JAX ----
        t0 = time.perf_counter()
        d_h_j, h_h_j = gb_signal_het_get_ll_in_kernel_jax(
            jnp.asarray(params_cand_all),
            jnp.asarray(c0_sparse_all),
            jnp.asarray(A0_all), jnp.asarray(A1_all),
            jnp.asarray(B0_all), jnp.asarray(B1_all),
            jnp.asarray(window_full),
            jnp.asarray(n_sparse_local),
            jnp.asarray(params_inj.reshape(1, 9)),
            jnp.asarray(data_index_all),
            source_jax, orbits_jax, tdi_config_jax,
            nparams=9, f0_idx=1,
            Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
            Nt_layer=NT_LAYER, N_sparse_t=N_sparse_t, stride=stride,
            ind_min_t=ind_min_t, ind_min_f=ind_min_f,
            m_active_half_width=2,
            layer_df=layer_df, dt=dt,
            T_obs=Tobs, t_start=t_start,
            nchannels=3, tdi_type=0,
            N_sparse_fd=N_SPARSE_FD,
            tukey_alpha=TUKEY_ALPHA,
            max_r=MAX_R,
        )
        d_h_j = float(d_h_j[0]); h_h_j = float(h_h_j[0])
        logL_jax = -0.5 * d_d_lt + d_h_j - 0.5 * h_h_j
        t1 = time.perf_counter()

        abs_diff = abs(logL_cpp - logL_jax)
        reldiff  = abs_diff / max(abs(logL_cpp), abs(logL_jax), 1.0)
        print(f"  {label:<22s} {logL_cpp:>+14.6e} {logL_jax:>+14.6e} "
              f"{abs_diff:>11.3e} {reldiff:>11.3e}  ({t1-t0:.2f}s)",
              flush=True)

    if not DO_GRAD:
        return 0

    # =================================================================
    # jax.grad vs C++ central-difference grad
    # =================================================================
    print("\n[grad] comparing jax.grad vs C++ central-difference",
          flush=True)

    def logL_jax_fn(params):
        params_b = params[None, :]   # (1, 9)
        d_h, h_h = gb_signal_het_get_ll_in_kernel_jax(
            params_b,
            jnp.asarray(c0_sparse_all),
            jnp.asarray(A0_all), jnp.asarray(A1_all),
            jnp.asarray(B0_all), jnp.asarray(B1_all),
            jnp.asarray(window_full),
            jnp.asarray(n_sparse_local),
            jnp.asarray(params_inj.reshape(1, 9)),
            jnp.asarray(np.zeros(1, dtype=np.int32)),
            source_jax, orbits_jax, tdi_config_jax,
            nparams=9, f0_idx=1,
            Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
            Nt_layer=NT_LAYER, N_sparse_t=N_sparse_t, stride=stride,
            ind_min_t=ind_min_t, ind_min_f=ind_min_f,
            m_active_half_width=2,
            layer_df=layer_df, dt=dt,
            T_obs=Tobs, t_start=t_start,
            nchannels=3, tdi_type=0,
            N_sparse_fd=N_SPARSE_FD,
            tukey_alpha=TUKEY_ALPHA,
            max_r=MAX_R,
        )
        return (-0.5 * d_d_lt + d_h[0] - 0.5 * h_h[0])

    grad_jax_fn = jax.jit(jax.grad(logL_jax_fn))

    # Test at injection
    params_test = points[1][1]   # the Df0 perturbation candidate
    t0 = time.perf_counter()
    g_jax = np.asarray(grad_jax_fn(jnp.asarray(params_test)))
    t1 = time.perf_counter()
    print(f"  jax.grad (first call w/ JIT): {t1-t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    g_jax = np.asarray(grad_jax_fn(jnp.asarray(params_test)))
    t1 = time.perf_counter()
    print(f"  jax.grad (warm call):         {t1-t0:.3f}s", flush=True)

    # C++ central-difference grad with same eps as MCMC test.
    PARAM_EPS = np.array([
        params_test[0] * 1e-3, layer_df * 1e-3, 1e-18, 0.0,
        1e-3, 1e-3, 1e-3, 1e-3, 1e-3,
    ], dtype=np.float64)
    grad_cpp = np.zeros((1, 9), dtype=np.float64)
    d_h_central = np.zeros(1, dtype=np.float64)
    h_h_central = np.zeros(1, dtype=np.float64)
    t0 = time.perf_counter()
    cpp.gb_signal_het_get_ll_grad_in_kernel(
        tdi_wrap, grad_cpp, d_h_central, h_h_central,
        c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
        window_full, n_sparse_local,
        params_test.astype(np.float64).reshape(1, 9).copy(),
        params_inj.reshape(1, 9).copy(),
        np.zeros(1, dtype=np.int32),
        PARAM_EPS,
        1, 1, 9, 1, 2,
        Nf, Nt, Nf_active, Nt_active,
        NT_LAYER, N_sparse_t, stride, ind_min_t, ind_min_f, 2,
        layer_df, dt, Tobs, t_start, 3, 0, N_SPARSE_FD,
        TUKEY_ALPHA, MAX_R,
    )
    t1 = time.perf_counter()
    print(f"  C++ central-diff grad:        {t1-t0:.3f}s", flush=True)

    PARAM_NAMES = ["amp", "f0", "fdot", "fddot", "phi0", "inc",
                    "psi", "lam", "beta"]
    print(f"\n  {'param':>7s} {'eps':>11s} {'grad_jax':>14s} "
          f"{'grad_cpp':>14s} {'reldiff':>11s}", flush=True)
    for k in range(9):
        denom = max(abs(g_jax[k]), abs(grad_cpp[0, k]), 1.0)
        rd = abs(g_jax[k] - grad_cpp[0, k]) / denom
        print(f"  {PARAM_NAMES[k]:>7s} {PARAM_EPS[k]:>11.3e} "
              f"{g_jax[k]:>+14.6e} {grad_cpp[0, k]:>+14.6e} "
              f"{rd:>11.3e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
