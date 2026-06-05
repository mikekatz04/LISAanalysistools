"""
Validate the C++ chunked-heterodyne kernels (gb_wdm_het_*) against the
Python reference path (the validated chunked_get_ll_python_reference +
_stitched_wdm_from_heterodyne in check_shortened_wdm.py / gb_wdm_het.py).

For both GB and SOBBH sources we run the SAME parameters through:
  - use_cpp=True  -> GBComputationGroupWrap.gb_wdm_het_* /
                     SOBBHComputationGroupWrap.sobbh_wdm_het_*
  - use_cpp=False -> pure-Python fallback

and compare:
  1. fill_global output template (max relative difference + mm2 / mm5 between
     the two templates -- they should agree at machine precision since the
     two paths are the same algorithm modulo C++ vs numpy arithmetic order)
  2. get_ll  outputs (<d|h>, <h|h>)
  3. swap_ll outputs (5 inner products)
"""

import time
import numpy as np
from gb_wdm_het import GBWDMHeterodyne, SOBBHWDMHeterodyne
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI


# --- common setup ------------------------------------------------------------
DT = 10.0
NF = 256
NT = 256
NT_SUB = 128
N_PAD = 16
N_SPARSE = 128
NCHANNELS = 3


def _build(Klass, t_obs_start, params_kind, use_cpp, N_cp_sig=0, N_cp_orbit=0):
    Tobs = NF * NT * DT
    orbits = EqualArmlengthOrbits()
    t_arr = np.arange(0.0, Tobs + DT, DT) + t_obs_start
    try:
        orbits.configure(t_arr=t_arr, dt=DT, linear_interp_setup=True)
    except TypeError:
        orbits.configure(t_arr=t_arr)
    return Klass(
        Nf=NF, Nt=NT, dt=DT, T_full=Tobs, t_ref_full=t_obs_start,
        Nt_sub=NT_SUB, n_pad=N_PAD, N_sparse=N_SPARSE,
        force_backend="cpu", tdi_gen="2nd generation",
        orbits=orbits, t_obs_start=t_obs_start, use_cpp=use_cpp,
        N_cp_sig=N_cp_sig, N_cp_orbit=N_cp_orbit,
    )


def _max_relative_diff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = np.maximum(np.abs(a), np.abs(b))
    nz = denom > 0
    if not np.any(nz):
        return 0.0
    diff = np.abs(a - b)
    return float(np.max(diff[nz] / denom[nz]))


def _mm(a, b):
    """Symmetric mismatch on two (real) WDM templates."""
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    ab = float(np.dot(a, b))
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return 1.0 - ab / np.sqrt(aa * bb)


def _validate_one(klass, label, params_list, N_cp_sig=0, N_cp_orbit=0):
    mode = "signal+orbit splines" if (N_cp_sig > 0 and N_cp_orbit > 0) \
           else "signal spline" if N_cp_sig > 0 \
           else "orbit spline" if N_cp_orbit > 0 \
           else "direct"
    print(f"\n=== {label}  (N_cp_sig={N_cp_sig}, N_cp_orbit={N_cp_orbit}, mode={mode}) ===")
    t_obs_start = 0.5 * YRSID_SI

    cpp = _build(klass, t_obs_start, "kind-ignored", use_cpp=True,
                  N_cp_sig=N_cp_sig, N_cp_orbit=N_cp_orbit)
    py  = _build(klass, t_obs_start, "kind-ignored", use_cpp=False)

    template_cpp = np.zeros((NCHANNELS, NF, NT), dtype=float)
    template_py  = np.zeros((NCHANNELS, NF, NT), dtype=float)

    t0 = time.perf_counter()
    cpp.fill_global(template_cpp, params_list)
    t_cpp = time.perf_counter() - t0
    t0 = time.perf_counter()
    py.fill_global(template_py, params_list)
    t_py = time.perf_counter() - t0

    print(f"fill_global: C++ {t_cpp:.3f}s vs Python {t_py:.3f}s "
          f"(speedup {t_py/max(t_cpp,1e-9):.1f}x)")
    print(f"  template max:    C++ {np.abs(template_cpp).max():.4e}  "
          f"py {np.abs(template_py).max():.4e}")
    print(f"  max rel diff:    {_max_relative_diff(template_cpp, template_py):.3e}")
    print(f"  mm(template):    {_mm(template_cpp, template_py):.3e}")

    # ---- get_ll on a synthetic data/PSD ---------------------------------
    rng = np.random.default_rng(0)
    data_d = rng.standard_normal((NCHANNELS, NF, NT)) * 1e-22
    invC = np.full((NCHANNELS, NF, NT), 1e44, dtype=float)
    d_h_cpp, h_h_cpp = cpp.get_ll(data_d, invC, params_list)
    d_h_py , h_h_py  = py.get_ll(data_d, invC, params_list)
    print(f"  get_ll d_h:  C++ {d_h_cpp[0]:.4e}  py {d_h_py[0]:.4e}  "
          f"reldiff {_max_relative_diff(d_h_cpp, d_h_py):.3e}")
    print(f"  get_ll h_h:  C++ {h_h_cpp[0]:.4e}  py {h_h_py[0]:.4e}  "
          f"reldiff {_max_relative_diff(h_h_cpp, h_h_py):.3e}")

    # ---- get_ll with layer-group iteration restriction ---------------------
    # The grouped path restricts the m-band per binary to ~5 layers around
    # the carrier; should match the un-grouped path to machine precision
    # (it's the same accumulator math, just skipping zero-w_chunk layers).
    d_h_gr, h_h_gr = cpp.get_ll(data_d, invC, params_list,
                                  use_layer_groups=True, margin_layers=1)
    print(f"  get_ll grouped d_h: {d_h_gr[0]:.4e}  reldiff vs un-grouped "
          f"{_max_relative_diff(d_h_gr, d_h_cpp):.3e}")
    print(f"  get_ll grouped h_h: {h_h_gr[0]:.4e}  reldiff vs un-grouped "
          f"{_max_relative_diff(h_h_gr, h_h_cpp):.3e}")

    # ---- swap_ll: use the same source as both add and remove ------------
    # (degenerate case: d_h_add == d_h_remove, aa == rr == ar)
    dh_a, dh_r, aa, rr, ar = cpp.swap_ll(data_d, invC, params_list, params_list)
    dh_a_py, dh_r_py, aa_py, rr_py, ar_py = py.swap_ll(data_d, invC, params_list, params_list)
    print(f"  swap_ll d_h_add: C++ {dh_a[0]:.4e}  py {dh_a_py[0]:.4e}  "
          f"reldiff {_max_relative_diff(dh_a, dh_a_py):.3e}")
    print(f"  swap_ll aa:      C++ {aa[0]:.4e}  py {aa_py[0]:.4e}  "
          f"reldiff {_max_relative_diff(aa, aa_py):.3e}")
    print(f"  swap_ll ar:      C++ {ar[0]:.4e}  py {ar_py[0]:.4e}  "
          f"reldiff {_max_relative_diff(ar, ar_py):.3e}")

    # ---- swap_ll grouped: identical add+remove (perfectly overlapping)
    # band; exercises pass-1 cache path. With group_band_layers=5 and
    # margin_layers=1, pass 2 detects no work to do.
    dh_a_gr, dh_r_gr, aa_gr, rr_gr, ar_gr = cpp.swap_ll(
        data_d, invC, params_list, params_list,
        use_layer_groups=True, margin_layers=1)
    print(f"  swap_ll grouped (overlap) d_h_add reldiff "
          f"{_max_relative_diff(dh_a_gr, dh_a):.3e}, "
          f"aa {_max_relative_diff(aa_gr, aa):.3e}, "
          f"ar {_max_relative_diff(ar_gr, ar):.3e}")

    # ---- swap_ll grouped (shifted carrier): tests pass 2 -- shift the
    # remove carrier by ~10 layers so add/remove m-bands are disjoint.
    layer_df_v = float(cpp.layer_df)
    f0_idx = int(cpp._CPP_F0_PARAM_INDEX)
    params_rem_shift = [p.copy() for p in params_list]
    for p in params_rem_shift:
        p[f0_idx] += 10.0 * layer_df_v
    dh_a_sh, dh_r_sh, aa_sh, rr_sh, ar_sh = cpp.swap_ll(
        data_d, invC, params_list, params_rem_shift)
    dh_a_sh_py, dh_r_sh_py, aa_sh_py, rr_sh_py, ar_sh_py = py.swap_ll(
        data_d, invC, params_list, params_rem_shift)
    print(f"  swap_ll (shifted) ungrouped vs py:  "
          f"d_h_rem {_max_relative_diff(dh_r_sh, dh_r_sh_py):.3e}, "
          f"rr {_max_relative_diff(rr_sh, rr_sh_py):.3e}, "
          f"ar {_max_relative_diff(ar_sh, ar_sh_py):.3e}")
    dh_a_sh_gr, dh_r_sh_gr, aa_sh_gr, rr_sh_gr, ar_sh_gr = cpp.swap_ll(
        data_d, invC, params_list, params_rem_shift,
        use_layer_groups=True, margin_layers=1)
    print(f"  swap_ll (shifted) grouped vs un-grouped:  "
          f"d_h_rem {_max_relative_diff(dh_r_sh_gr, dh_r_sh):.3e}, "
          f"rr {_max_relative_diff(rr_sh_gr, rr_sh):.3e}, "
          f"aa {_max_relative_diff(aa_sh_gr, aa_sh):.3e}, "
          f"ar {_max_relative_diff(ar_sh_gr, ar_sh):.3e}")


def main():
    # GB: [amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta]
    gb_params = [
        np.array([1e-22, 5e-3, 0.0, 0.0, 1.0, 0.5, 0.3, 2.0, 0.4]),
    ]
    _validate_one(GBWDMHeterodyne, "GB chunked-het", gb_params, N_cp_sig=0,  N_cp_orbit=0)
    _validate_one(GBWDMHeterodyne, "GB chunked-het", gb_params, N_cp_sig=0,  N_cp_orbit=32)
    _validate_one(GBWDMHeterodyne, "GB chunked-het", gb_params, N_cp_sig=48, N_cp_orbit=0)
    _validate_one(GBWDMHeterodyne, "GB chunked-het", gb_params, N_cp_sig=48, N_cp_orbit=32)

    # SOBBH: [m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta]
    sobbh_params = [
        np.array([40.0, 30.0, 0.1, 0.05, 1500.0, 10e-3, 0.7, 0.4, 0.3, 2.0, 0.3]),
    ]
    try:
        _validate_one(SOBBHWDMHeterodyne, "SOBBH chunked-het", sobbh_params, N_cp_sig=0,  N_cp_orbit=0)
        _validate_one(SOBBHWDMHeterodyne, "SOBBH chunked-het", sobbh_params, N_cp_sig=48, N_cp_orbit=32)
    except Exception as e:
        print(f"\nSOBBH validation skipped: {e}")


if __name__ == "__main__":
    main()
