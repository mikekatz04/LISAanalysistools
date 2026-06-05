"""Validate Python-side central-FD gradient of GBWDMComputations.get_ll_wdm
against JAX autograd through the JAX chunked-het kernel.

This exercises the production code path (GBWDMComputations + chunked-het
C++ kernel) and the matching jax kernel for autograd. The two should
agree to FD truncation error (~ eps^2 * |d^3L/dtheta^3| / 6).

Runs on CPU. Times out a few seconds end-to-end for num_bin=1.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.domains import WDMSettings
from gbgpu.gbcomps import GBWDMComputations


DT       = 10.0
NF       = 256
NT       = 256
NT_SUB   = 128
N_PAD    = 16
N_SPARSE = 128
NCH      = 3


def _max_rel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = np.maximum(np.abs(a), np.abs(b))
    nz = denom > 0
    if not np.any(nz):
        return 0.0
    return float(np.max(np.abs(a - b)[nz] / denom[nz]))


def make_holder(data_d, invC, nch, Nf, Nt):
    """Minimal AnalysisContainerArray-like shim: exposes
    .linear_data_arr[0] / .linear_psd_arr[0] / __len__ / nchannels."""
    class _H:
        def __init__(self):
            self.linear_data_arr = [data_d.reshape(-1).copy()]
            self.linear_psd_arr  = [invC.reshape(-1).copy()]
            self.nchannels = nch
            self.Nf = Nf
            self.Nt = Nt
        def __len__(self): return 1
    return _H()


def main():
    Tobs = NF * NT * DT
    t_obs_start = 0.5 * YRSID_SI

    orbits = EqualArmlengthOrbits()
    t_arr = np.arange(0.0, Tobs + DT, DT) + t_obs_start
    try:
        orbits.configure(t_arr=t_arr, dt=DT, linear_interp_setup=True)
    except TypeError:
        orbits.configure(t_arr=t_arr)

    wdm_set = WDMSettings(
        Nf=NF, Nt=NT, dt=DT, t0=t_obs_start,
        min_freq=1e-4, max_freq=3e-2,
        force_backend="cpu",
    )

    cpp = GBWDMComputations(
        wdm_set, t_ref=t_obs_start,
        Nt_sub=NT_SUB, n_pad=N_PAD, N_sparse=N_SPARSE,
        N_cp_sig=0, N_cp_orbit=0,
        orbits=orbits, tdi_config="2nd generation",
        force_backend="cpu", tdi_type="XYZ", d_d=0.0,
    )

    # GB params: [A, f0, fdot, fddot, phi0, inc, psi, lam, beta]
    params = np.array([[1e-22, 5e-3, 0.0, 0.0, 1.0, 0.5, 0.3, 2.0, 0.4]])
    num_bin, nparams = params.shape

    rng = np.random.default_rng(0)
    Nfa = wdm_set.Nf_active
    Nta = wdm_set.Nt_active
    data_d = rng.standard_normal((NCH, Nfa, Nta)) * 1e-22
    invC   = np.full((NCH, NCH, Nfa, Nta), 0.0, dtype=float)
    for c in range(NCH):
        invC[c, c] = 1e44   # diagonal-only synthetic invC for the test

    holder = make_holder(data_d, invC, NCH, Nfa, Nta)

    # Central FD step sizes per param (tuned for double-precision FP noise floor)
    # Central-FD step per param. For params with reference value 0 (fdot, fddot)
    # the eps must lift |dL| above FP noise (~ 1e-16 * |L|) without entering
    # the truncation-error regime (~ eps^2 * |d^3 L/d theta^3| / 6).
    eps = np.array([
        1e-29,   # A      (rel ~ 1e-7 of 1e-22)
        2e-10,   # f0     (rel ~ 4e-8 of 5e-3 Hz)
        1e-17,   # fdot   (abs; ref=0; |dL/dfdot| ~ 1e13 -> |dL| ~ 1e-4)
        1e-21,   # fddot  (abs; ref=0; |dL/dfddot| ~ 1e17 -> |dL| ~ 1e-4)
        1e-6,    # phi0
        1e-6,    # inc
        1e-6,    # psi
        1e-6,    # lam
        1e-6,    # beta
    ])

    def loss(p):
        cpp.get_ll_wdm(p, holder,
                       convert_to_ra_dec=False,
                       use_layer_groups=False)
        # GBWDMComputations.get_ll_wdm stashes raw inner products on
        # the instance (.d_h_out / .h_h_out). We define
        #    L = sum_i (d_h_i - 0.5 * h_h_i)
        # matching the JAX scalar_loss in gb_wdm_het_get_ll_grad_jax.
        return float(np.asarray(cpp.d_h_out)[0] - 0.5 * np.asarray(cpp.h_h_out)[0])

    # Reference central evaluation (L) for FD baseline
    L_c = loss(params)

    print("[cpp] computing central-FD gradient ...")
    grad_cpp = np.zeros((num_bin, nparams))
    for k in range(nparams):
        ek = eps[k]
        p_plus  = params.copy(); p_plus[0,  k] += ek
        p_minus = params.copy(); p_minus[0, k] -= ek
        L_plus  = loss(p_plus)
        L_minus = loss(p_minus)
        grad_cpp[0, k] = (L_plus - L_minus) / (2.0 * ek)
    print(f"  L_central={L_c:+.6e}")

    print("[jax] computing autograd gradient ...")
    from lisatools.jax.orbits import OrbitsWrapJAX
    from lisatools.jax.response.tdi_config import TDIConfigWrapJAX
    from gbgpu.jax.sources.ucb import JaxUCBSource
    from gbgpu.jax.wdm.heterodyne_kernels import (
        gb_wdm_het_get_ll_grad_jax,
    )

    jax_orbits = OrbitsWrapJAX(*orbits.pycppdetector_args)
    jax_tdi    = TDIConfigWrapJAX(*cpp.tdi_config.pytdiconfig_args)
    jax_source = JaxUCBSource(t_ref=t_obs_start)

    grad_jax = gb_wdm_het_get_ll_grad_jax(
        jnp.asarray(params),
        jnp.asarray(data_d), jnp.asarray(invC),
        jnp.asarray(cpp.chunk_t_starts),
        jnp.asarray(cpp.chunk_keep_lo), jnp.asarray(cpp.chunk_keep_hi),
        jnp.asarray(cpp.chunk_n_global_offset),
        jax_source, jax_orbits, jax_tdi,
        jnp.asarray(cpp.wdm_window),
        Nf=NF, Nt=NT, Nt_sub=NT_SUB, N_sparse=N_SPARSE,
        dt=DT, T_chunk=cpp.T_chunk,
        ind_min_f=wdm_set.ind_min_f, ind_min_t=wdm_set.ind_min_t,
        Nf_active=wdm_set.Nf_active, Nt_active=wdm_set.Nt_active,
    )
    grad_jax = np.asarray(grad_jax)

    print()
    print("==== get_ll_wdm gradient comparison: C++ central-FD  vs  JAX autograd ====")
    names = ["A", "f0", "fdot", "fddot", "phi0", "inc", "psi", "lam", "beta"]
    for k, n in enumerate(names):
        a = grad_cpp[0, k]; b = grad_jax[0, k]
        rd = abs(a - b) / max(abs(a), abs(b), 1e-300)
        print(f"  d/d{n:6s}  C++_FD={a:+.6e}  JAX={b:+.6e}  reldiff={rd:.3e}")
    print(f"  --- max rel diff: {_max_rel(grad_cpp, grad_jax):.3e}")


if __name__ == "__main__":
    main()
