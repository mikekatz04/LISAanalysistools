"""Compare C++ central-FD gradient vs JAX autograd gradient for the
chunked-heterodyne likelihood (both get_ll and swap_ll).

The Python `swap_ll_grad` / `get_ll_grad` methods on GBWDMHeterodyne wrap
the existing C++ kernels with central finite differences (theta_add side
only). JAX autograd via `jax.grad` differentiates analytically through
the JAX-functional kernel. The two should agree to FD-truncation error
(~ eps^2 * |d^3L/dtheta^3| / 6).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gb_wdm_het import GBWDMHeterodyne
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI


DT      = 10.0
NF      = 256
NT      = 256
NT_SUB  = 128
N_PAD   = 16
N_SPARSE = 128
NCHANNELS = 3


def _max_rel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = np.maximum(np.abs(a), np.abs(b))
    nz = denom > 0
    if not np.any(nz):
        return 0.0
    return float(np.max(np.abs(a - b)[nz] / denom[nz]))


def main():
    # ---- common C++ setup -------------------------------------------------
    t_obs_start = 0.5 * YRSID_SI
    Tobs = NF * NT * DT
    orbits = EqualArmlengthOrbits()
    t_arr = np.arange(0.0, Tobs + DT, DT) + t_obs_start
    try:
        orbits.configure(t_arr=t_arr, dt=DT, linear_interp_setup=True)
    except TypeError:
        orbits.configure(t_arr=t_arr)

    cpp = GBWDMHeterodyne(
        Nf=NF, Nt=NT, dt=DT, T_full=Tobs, t_ref_full=t_obs_start,
        Nt_sub=NT_SUB, n_pad=N_PAD, N_sparse=N_SPARSE,
        force_backend="cpu", tdi_gen="2nd generation",
        orbits=orbits, t_obs_start=t_obs_start,
        N_cp_sig=0, N_cp_orbit=0,
    )

    # GB params: [amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta]
    params_add = [np.array([1e-22, 5e-3, 0.0, 0.0, 1.0, 0.5, 0.3, 2.0, 0.4])]
    params_rem = [np.array([1.1e-22, 5.05e-3, 0.0, 0.0, 0.9, 0.6, 0.4, 2.1, 0.3])]
    num_bin = len(params_add)

    cpp._ensure_cpp_setup()
    rng = np.random.default_rng(0)
    data_d = rng.standard_normal((NCHANNELS, NF, NT)) * 1e-22
    invC   = np.full((NCHANNELS, NF, NT), 1e44, dtype=float)

    # FD steps. Picked small enough to avoid truncation error but well
    # above FP noise for double precision. For log10 of cubic eps in
    # central FD, 1e-7 of param scale is the sweet spot for doubles.
    param_eps = np.array([
        1e-29,    # A: ~1e-7 * 1e-22
        2e-10,    # f0: 4e-8 of 5e-3 Hz
        1e-22,    # fdot0
        1e-26,    # fddot0
        1e-6,     # phi0 rad
        1e-6,     # inc rad
        1e-6,     # psi rad
        1e-6,     # lam rad
        1e-6,     # beta rad
    ])

    # ---- C++ central-FD gradient ------------------------------------------
    print("[1/4] computing get_ll C++ central-FD grad ...")
    grad_cpp_get = cpp.get_ll_grad(data_d, invC, params_add, param_eps)
    print(f"  shape: {grad_cpp_get.shape}")
    print(f"  values: {grad_cpp_get[0]}")

    print("[2/4] computing swap_ll C++ central-FD grad ...")
    swap_grad_cpp = cpp.swap_ll_grad(
        data_d, invC, params_add, params_rem, param_eps,
    )
    for k, v in swap_grad_cpp.items():
        print(f"  {k}: shape {v.shape}")

    # ---- JAX autograd gradient -------------------------------------------
    print("[3/4] computing get_ll JAX autograd grad ...")
    from fastlisaresponse.jax.orbits import OrbitsWrapJAX
    from fastlisaresponse.jax.tdi_config import TDIConfigWrapJAX
    from fastlisaresponse.jax.sources.ucb import JaxUCBSource
    from fastlisaresponse.jax.wdm.heterodyne_kernels import (
        gb_wdm_het_get_ll_grad_jax,
        gb_wdm_het_swap_ll_grad_jax,
    )

    jax_orbits = OrbitsWrapJAX(*cpp._orbits_py.pycppdetector_args)
    jax_tdi    = TDIConfigWrapJAX(*cpp._tdi_cfg_py.pytdiconfig_args)
    jax_source = JaxUCBSource(t_ref=cpp.t_ref_full)

    params_add_j = jnp.asarray(np.stack(params_add))
    params_rem_j = jnp.asarray(np.stack(params_rem))
    chunk_t_starts_j   = jnp.asarray(cpp._cpp_chunk_t_starts)
    chunk_keep_lo_j    = jnp.asarray(cpp._cpp_chunk_keep_lo)
    chunk_keep_hi_j    = jnp.asarray(cpp._cpp_chunk_keep_hi)
    chunk_n_global_lo_j = jnp.asarray(cpp._cpp_chunk_n_global_offset)
    wdm_window_j       = jnp.asarray(cpp._cpp_wdm_window)
    data_d_j = jnp.asarray(data_d)
    invC_j   = jnp.asarray(invC)

    grad_jax_get = gb_wdm_het_get_ll_grad_jax(
        params_add_j, data_d_j, invC_j,
        chunk_t_starts_j, chunk_keep_lo_j, chunk_keep_hi_j, chunk_n_global_lo_j,
        jax_source, jax_orbits, jax_tdi, wdm_window_j,
        Nf=NF, Nt=NT, Nt_sub=NT_SUB, N_sparse=N_SPARSE,
        dt=DT, T_chunk=cpp.T_chunk,
    )
    grad_jax_get = np.asarray(grad_jax_get)
    print(f"  values: {grad_jax_get[0]}")

    print("[4/4] computing swap_ll JAX autograd grad ...")
    swap_grad_jax = gb_wdm_het_swap_ll_grad_jax(
        params_add_j, params_rem_j, data_d_j, invC_j,
        chunk_t_starts_j, chunk_keep_lo_j, chunk_keep_hi_j, chunk_n_global_lo_j,
        jax_source, jax_orbits, jax_tdi, wdm_window_j,
        Nf=NF, Nt=NT, Nt_sub=NT_SUB, N_sparse=N_SPARSE,
        dt=DT, T_chunk=cpp.T_chunk,
    )

    # ---- compare ----------------------------------------------------------
    print()
    print("==== get_ll gradient comparison (per-parameter reldiff) ====")
    param_names = ["A", "f0", "fdot0", "fddot0", "phi0", "inc", "psi", "lam", "beta"]
    for k, name in enumerate(param_names):
        a = grad_cpp_get[0, k]
        b = grad_jax_get[0, k]
        rd = abs(a - b) / max(abs(a), abs(b), 1e-300)
        print(f"  d/d{name:6s}: C++_FD={a:+.6e}  JAX_grad={b:+.6e}  reldiff={rd:.2e}")
    print(f"  --- max rel diff: {_max_rel(grad_cpp_get, grad_jax_get):.3e}")

    print()
    print("==== swap_ll gradient comparison (theta_add only) ====")
    for key in ("grad_dh_add", "grad_dh_rem", "grad_aa", "grad_rr", "grad_ar",
                "grad_L_add"):
        a = np.asarray(swap_grad_cpp[key])
        b = np.asarray(swap_grad_jax[key])
        print(f"  {key}:  max rel diff = {_max_rel(a, b):.3e}")


if __name__ == "__main__":
    main()
