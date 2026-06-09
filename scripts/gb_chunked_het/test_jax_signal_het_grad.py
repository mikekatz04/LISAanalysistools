#!/usr/bin/env python
"""Timing + correctness for jax.grad on signal-het end-to-end.

Quick fast script:
  1. Build setup (~minute)
  2. Forward call 1 (JIT, slow)
  3. Forward call 2-3 (warm, should be fast)
  4. jax.grad WITHOUT jit on call 1, with jit on call 2 (compare)
  5. C++ central-diff for reference

If warm forward is < 1 sec and jit'd grad is < 5 sec, jax-NUTS is feasible.
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

# Register OrbitsWrapJAX as a JAX pytree so the large jnp arrays (n, ltt, x)
# flow through jax.jit as dynamic arguments instead of closure constants
# (otherwise XLA constant-folds every slice op, ~20s each, locking up
# compile). This is the same patch we should land upstream in LAT.
from jax.tree_util import register_pytree_node
def _orbits_flatten(o):
    children = (o.n, o.ltt, o.x)
    aux = (o.ltt_t0, o.ltt_dt, o.ltt_N, o.sc_t0, o.sc_dt, o.sc_N,
           o.armlength, o.links, o.sc_r, o.sc_e, o._link_to_ind)
    return children, aux
def _orbits_unflatten(aux, children):
    obj = OrbitsWrapJAX.__new__(OrbitsWrapJAX)
    obj.n, obj.ltt, obj.x = children
    (obj.ltt_t0, obj.ltt_dt, obj.ltt_N, obj.sc_t0, obj.sc_dt, obj.sc_N,
     obj.armlength, obj.links, obj.sc_r, obj.sc_e, obj._link_to_ind) = aux
    return obj
try:
    register_pytree_node(OrbitsWrapJAX, _orbits_flatten, _orbits_unflatten)
    print("[patch] registered OrbitsWrapJAX as pytree", flush=True)
except ValueError:
    pass  # already registered


def setup_test():
    """Returns the kwargs dict needed for the JAX/C++ kernels."""
    TUKEY_ALPHA = 0.05
    NT_LAYER    = 64
    N_SPARSE_FD = 1024
    F0_MHZ      = 14.22
    SNR         = 50.0
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
    try:
        orbits_cpp.configure(t_arr=t_arr, dt=dt, linear_interp_setup=True)
    except TypeError:
        orbits_cpp.configure(t_arr=t_arr)

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

    source_jax = JaxUCBSource(t_ref=t_start)
    orbits_jax = OrbitsWrapJAX(*orbits_cpp.pycppdetector_args)
    tdi_config_jax = TDIConfigWrapJAX(*tdi_config_cpp.pytdiconfig_args)

    return dict(
        params_inj=params_inj, d_d_lt=d_d_lt,
        c0_sparse_all=c0_sparse_active[None, ...].copy(),
        A0_all=A0[None, ...], A1_all=A1[None, ...],
        B0_all=B0[None, ...], B1_all=B1[None, ...],
        window_full=window_full, n_sparse_local=n_sparse_local,
        layer_df=layer_df, dt=dt, Tobs=Tobs, t_start=t_start,
        Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
        NT_LAYER=NT_LAYER, N_sparse_t=N_sparse_t, stride=stride,
        ind_min_t=ind_min_t, ind_min_f=ind_min_f,
        N_SPARSE_FD=N_SPARSE_FD, TUKEY_ALPHA=TUKEY_ALPHA,
        source_jax=source_jax, orbits_jax=orbits_jax,
        tdi_config_jax=tdi_config_jax, tdi_wrap=tdi_wrap,
    )


def main():
    MAX_R = float(os.environ.get("MAX_R", "5.0"))
    print(f"[setup] starting...", flush=True)
    t0 = time.perf_counter()
    ctx = setup_test()
    t1 = time.perf_counter()
    print(f"[setup] done in {t1-t0:.1f}s", flush=True)

    # JAX forward, taking orbits as an explicit arg so jax.jit pytree-
    # flattens the big jnp arrays into dynamic inputs (avoids the
    # constant-folding compile blowup we saw with closure).
    c0_jax  = jnp.asarray(ctx["c0_sparse_all"])
    A0_jax  = jnp.asarray(ctx["A0_all"])
    A1_jax  = jnp.asarray(ctx["A1_all"])
    B0_jax  = jnp.asarray(ctx["B0_all"])
    B1_jax  = jnp.asarray(ctx["B1_all"])
    win_jax = jnp.asarray(ctx["window_full"])
    nsp_jax = jnp.asarray(ctx["n_sparse_local"])
    pref_jax = jnp.asarray(ctx["params_inj"].reshape(1, 9))
    didx_jax = jnp.asarray(np.zeros(1, dtype=np.int32))

    def logL_jax(params, orbits):
        d_h, h_h = gb_signal_het_get_ll_in_kernel_jax(
            params[None, :],
            c0_jax, A0_jax, A1_jax, B0_jax, B1_jax,
            win_jax, nsp_jax, pref_jax, didx_jax,
            ctx["source_jax"], orbits, ctx["tdi_config_jax"],
            nparams=9, f0_idx=1,
            Nf=ctx["Nf"], Nt=ctx["Nt"],
            Nf_active=ctx["Nf_active"], Nt_active=ctx["Nt_active"],
            Nt_layer=ctx["NT_LAYER"], N_sparse_t=ctx["N_sparse_t"],
            stride=ctx["stride"],
            ind_min_t=ctx["ind_min_t"], ind_min_f=ctx["ind_min_f"],
            m_active_half_width=2,
            layer_df=ctx["layer_df"], dt=ctx["dt"],
            T_obs=ctx["Tobs"], t_start=ctx["t_start"],
            nchannels=3, tdi_type=0,
            N_sparse_fd=ctx["N_SPARSE_FD"],
            tukey_alpha=ctx["TUKEY_ALPHA"],
            max_r=MAX_R,
        )
        return -0.5 * ctx["d_d_lt"] + d_h[0] - 0.5 * h_h[0]

    # Forward call 1 (no jit)
    params_test = ctx["params_inj"].copy()
    params_test[1] += 1e-3 * ctx["layer_df"]
    params_test_jax = jnp.asarray(params_test)
    orbits_jax = ctx["orbits_jax"]
    t0 = time.perf_counter()
    ll1 = float(logL_jax(params_test_jax, orbits_jax))
    t1 = time.perf_counter()
    print(f"[forward] call 1 (uncompiled): {t1-t0:.2f}s  logL={ll1:+.6e}",
          flush=True)

    # ---- JIT compile test ----
    print(f"\n[jit] compiling jax.jit(logL_jax)...", flush=True)
    logL_jit = jax.jit(logL_jax)
    t0 = time.perf_counter()
    _ = float(logL_jit(params_test_jax, orbits_jax).block_until_ready())
    t1 = time.perf_counter()
    print(f"[jit] forward call 1 (compile): {t1-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    _ = float(logL_jit(params_test_jax, orbits_jax).block_until_ready())
    t1 = time.perf_counter()
    print(f"[jit] forward call 2 (warm):    {t1-t0:.4f}s", flush=True)

    # ---- jax.grad test ----
    print(f"\n[grad] compiling jax.jit(jax.grad(logL_jax))...", flush=True)
    grad_jit = jax.jit(jax.grad(logL_jax, argnums=0))
    t0 = time.perf_counter()
    g_jax = np.asarray(grad_jit(params_test_jax, orbits_jax)
                       .block_until_ready())
    t1 = time.perf_counter()
    print(f"[grad] call 1 (compile): {t1-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    _ = np.asarray(grad_jit(params_test_jax, orbits_jax).block_until_ready())
    t1 = time.perf_counter()
    print(f"[grad] call 2 (warm):    {t1-t0:.4f}s", flush=True)

    # Validate against C++ central-diff
    import lisatools_backend_cpu.pycppdetector as _lat_pd
import gbgpu_backend_cpu.cgbgpu as _be  # GBComputationGroupWrap lives here post-3L.7g
    cpp = _be.GBComputationGroupWrapCPU()
    PARAM_EPS = np.array([
        params_test[0] * 1e-3, ctx["layer_df"] * 1e-3, 1e-18, 0.0,
        1e-3, 1e-3, 1e-3, 1e-3, 1e-3,
    ], dtype=np.float64)
    grad_cpp = np.zeros((1, 9), dtype=np.float64)
    d_h_central = np.zeros(1, dtype=np.float64)
    h_h_central = np.zeros(1, dtype=np.float64)
    t0 = time.perf_counter()
    cpp.gb_signal_het_get_ll_grad_in_kernel(
        ctx["tdi_wrap"], grad_cpp, d_h_central, h_h_central,
        ctx["c0_sparse_all"], ctx["A0_all"], ctx["A1_all"],
        ctx["B0_all"], ctx["B1_all"],
        ctx["window_full"], ctx["n_sparse_local"],
        params_test.astype(np.float64).reshape(1, 9).copy(),
        ctx["params_inj"].reshape(1, 9).copy(),
        np.zeros(1, dtype=np.int32),
        PARAM_EPS,
        1, 1, 9, 1, 2,
        ctx["Nf"], ctx["Nt"], ctx["Nf_active"], ctx["Nt_active"],
        ctx["NT_LAYER"], ctx["N_sparse_t"], ctx["stride"],
        ctx["ind_min_t"], ctx["ind_min_f"], 2,
        ctx["layer_df"], ctx["dt"], ctx["Tobs"], ctx["t_start"],
        3, 0, ctx["N_SPARSE_FD"],
        ctx["TUKEY_ALPHA"], MAX_R,
    )
    t1 = time.perf_counter()
    print(f"[grad] C++ central-diff: {t1-t0:.2f}s", flush=True)

    NAMES = ["amp", "f0", "fdot", "fddot", "phi0", "inc", "psi", "lam", "beta"]
    print(f"\n  {'param':>7s} {'jax.grad':>14s} {'cpp_grad':>14s} {'reldiff':>11s}",
          flush=True)
    for k in range(9):
        denom = max(abs(g_jax[k]), abs(grad_cpp[0, k]), 1.0)
        rd = abs(g_jax[k] - grad_cpp[0, k]) / denom
        print(f"  {NAMES[k]:>7s} {g_jax[k]:>+14.6e} {grad_cpp[0, k]:>+14.6e} "
              f"{rd:>11.3e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
