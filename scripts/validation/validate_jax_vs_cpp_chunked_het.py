"""End-to-end JAX vs C++ parity check for the chunked-heterodyne pipeline.

Builds a single GBWDMHeterodyne instance to provide the chunk geometry,
WDM window, and OrbitsWrap args, then runs both the C++ wrappers
(``gb_wdm_het_fill_global`` / ``gb_wdm_het_get_ll`` /
``gb_wdm_het_swap_ll``) and the JAX kernels in
``fastlisaresponse.jax.wdm.heterodyne_kernels`` on the same inputs.
Reports max relative difference per pair.

Direct path only -- the JAX side does not yet have the signal/orbit
spline caches or layer-grouping; those are next on the JAX mirror
list. With those off, the C++ direct path matches Python to ~5e-16
(see validate_cpp_chunked_het.py) so any JAX delta is a JAX-side
issue.
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
    # ---- common C++ setup ----------------------------------------------
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
        backend="cpu", tdi_gen="2nd generation",
        orbits=orbits, t_obs_start=t_obs_start, use_cpp=True,
        N_cp_sig=0, N_cp_orbit=0,
    )

    # GB params: [amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta]
    params = [np.array([1e-22, 5e-3, 0.0, 0.0, 1.0, 0.5, 0.3, 2.0, 0.4])]
    num_bin = len(params)

    # ---- C++ side ------------------------------------------------------
    cpp._ensure_cpp_setup()
    rng = np.random.default_rng(0)
    data_d = rng.standard_normal((NCHANNELS, NF, NT)) * 1e-22
    invC   = np.full((NCHANNELS, NF, NT), 1e44, dtype=float)

    template_cpp = np.zeros((NCHANNELS, NF, NT), dtype=float)
    cpp.fill_global(template_cpp, params)
    d_h_cpp, h_h_cpp = cpp.get_ll(data_d, invC, params)
    dh_a, dh_r, aa, rr, ar = cpp.swap_ll(data_d, invC, params, params)

    # ---- JAX side ------------------------------------------------------
    # The JAX kernels accept the same Orbits and TDIConfig args as the
    # C++ side; we build the JAX wrappers from the same pycppdetector
    # tuples that the GBWDMHeterodyne stored on cpp._cpp_orbits etc.
    from fastlisaresponse.jax.orbits import OrbitsWrapJAX
    from fastlisaresponse.jax.tdi_config import TDIConfigWrapJAX
    from fastlisaresponse.jax.sources.ucb import JaxUCBSource
    from fastlisaresponse.jax.wdm.heterodyne_kernels import (
        gb_wdm_het_fill_global_jax,
        gb_wdm_het_get_ll_jax,
        gb_wdm_het_swap_ll_jax,
    )

    jax_orbits = OrbitsWrapJAX(*cpp._orbits_py.pycppdetector_args)
    jax_tdi    = TDIConfigWrapJAX(*cpp._tdi_cfg_py.pytdiconfig_args)
    jax_source = JaxUCBSource(t_ref=cpp.t_ref_full)

    params_batch = jnp.asarray(np.stack(params))               # (num_bin, 9)
    factors      = jnp.ones(num_bin)
    chunk_t_starts_j = jnp.asarray(cpp._cpp_chunk_t_starts)
    chunk_keep_lo_j  = jnp.asarray(cpp._cpp_chunk_keep_lo)
    chunk_keep_hi_j  = jnp.asarray(cpp._cpp_chunk_keep_hi)
    chunk_n_global_lo_j = jnp.asarray(cpp._cpp_chunk_n_global_offset)
    wdm_window_j    = jnp.asarray(cpp._cpp_wdm_window)

    template_jax = gb_wdm_het_fill_global_jax(
        params_batch, factors,
        chunk_t_starts_j, chunk_keep_lo_j, chunk_keep_hi_j, chunk_n_global_lo_j,
        jax_source, jax_orbits, jax_tdi,
        wdm_window_j,
        Nf=NF, Nt=NT, Nt_sub=NT_SUB, N_sparse=N_SPARSE,
        dt=DT, T_chunk=cpp.T_chunk,
    )
    template_jax = np.asarray(template_jax)

    d_h_j, h_h_j = gb_wdm_het_get_ll_jax(
        params_batch,
        jnp.asarray(data_d), jnp.asarray(invC),
        chunk_t_starts_j, chunk_keep_lo_j, chunk_keep_hi_j, chunk_n_global_lo_j,
        jax_source, jax_orbits, jax_tdi,
        wdm_window_j,
        Nf=NF, Nt=NT, Nt_sub=NT_SUB, N_sparse=N_SPARSE,
        dt=DT, T_chunk=cpp.T_chunk,
    )
    dh_a_j, dh_r_j, aa_j, rr_j, ar_j = gb_wdm_het_swap_ll_jax(
        params_batch, params_batch,
        jnp.asarray(data_d), jnp.asarray(invC),
        chunk_t_starts_j, chunk_keep_lo_j, chunk_keep_hi_j, chunk_n_global_lo_j,
        jax_source, jax_orbits, jax_tdi,
        wdm_window_j,
        Nf=NF, Nt=NT, Nt_sub=NT_SUB, N_sparse=N_SPARSE,
        dt=DT, T_chunk=cpp.T_chunk,
    )

    # ---- compare ------------------------------------------------------
    print("fill_global JAX vs C++:")
    print(f"  max |template|: C++ {np.abs(template_cpp).max():.3e}  "
          f"JAX {np.abs(template_jax).max():.3e}")
    print(f"  max rel diff:   {_max_rel(template_cpp, template_jax):.3e}")
    print()
    print("get_ll JAX vs C++:")
    print(f"  d_h: C++ {d_h_cpp[0]:.6e}  JAX {float(d_h_j[0]):.6e}  "
          f"reldiff {_max_rel(d_h_cpp, np.asarray(d_h_j)):.3e}")
    print(f"  h_h: C++ {h_h_cpp[0]:.6e}  JAX {float(h_h_j[0]):.6e}  "
          f"reldiff {_max_rel(h_h_cpp, np.asarray(h_h_j)):.3e}")
    print()
    print("swap_ll JAX vs C++ (degenerate add==rem):")
    for name, c, j in [("d_h_add", dh_a, dh_a_j), ("d_h_rem", dh_r, dh_r_j),
                        ("aa", aa, aa_j), ("rr", rr, rr_j), ("ar", ar, ar_j)]:
        print(f"  {name}: C++ {float(c[0]):.6e}  JAX {float(j[0]):.6e}  "
              f"reldiff {_max_rel(c, np.asarray(j)):.3e}")


if __name__ == "__main__":
    main()
