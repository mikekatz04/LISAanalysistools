"""Stand-alone JAX-only timing for the chunked-het GB pipeline.

Same data/PSD/parameter shape as ``gb_chunked_test_script.py``'s MCMC
batch (default N = NTEMPS * NWALKERS = 200). The C++ timer in the MCMC
script reported ~213 s / call at N=200 -- this script benchmarks the
JAX path side by side so we can see whether the JAX kernel avoids the
per-source GPU/CPU looping the C++ CPU build incurs.

Quick env knobs:
    N        : batch size               (default 200)
    REPEATS  : timed repetitions        (default 3)
    NF, NT, NT_SUB, N_SPARSE, N_PAD : chunked-het geometry
                                        (default matches MCMC script)
    JAX_GRAD_CHUNK : leaf-axis chunking for autograd (default = full N)

Reports:
    JAX get_ll          @ N
    JAX get_ll_grad     @ N (chunked if JAX_GRAD_CHUNK set)
    C++ get_ll          @ N for direct comparison
    C++ get_ll_grad (FD) @ small N (cost = 2*nparams forward passes)
    (Also N=1 baselines for each.)
"""
from __future__ import annotations

import os
import time

import numpy as np

from gb_wdm_het import GBWDMHeterodyne
from lisatools.detector import ESAOrbits
from lisatools.utils.constants import YRSID_SI


N           = int(os.environ.get("N", "200"))
REPEATS     = int(os.environ.get("REPEATS", "3"))
NF          = int(os.environ.get("NF", "1460"))
NT          = int(os.environ.get("NT", "2560"))
NT_SUB      = int(os.environ.get("NT_SUB", "256"))
N_SPARSE    = int(os.environ.get("N_SPARSE", "256"))
N_PAD       = int(os.environ.get("N_PAD", "32"))
N_CP_SIG    = int(os.environ.get("N_CP_SIG", "0"))   # MCMC script default
N_CP_ORBIT  = int(os.environ.get("N_CP_ORBIT", "0"))
DT          = 10.0
NCH         = 3


def _print_block(title):
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    print(f"[setup] N={N} REPEATS={REPEATS} NF={NF} NT={NT} Nt_sub={NT_SUB} "
          f"N_sparse={N_SPARSE} N_pad={N_PAD} N_cp_sig={N_CP_SIG} N_cp_orbit={N_CP_ORBIT}",
          flush=True)

    Tobs = NF * NT * DT
    t_obs_start = int(0.5 * YRSID_SI / DT) * DT
    orbits = ESAOrbits()
    chunked = GBWDMHeterodyne(
        Nf=NF, Nt=NT, dt=DT, T_full=Tobs, t_ref_full=t_obs_start,
        Nt_sub=NT_SUB, n_pad=N_PAD, N_sparse=N_SPARSE,
        backend="cpu", tdi_gen="2nd generation",
        orbits=orbits, t_obs_start=float(t_obs_start),
        use_cpp=True, N_cp_sig=N_CP_SIG, N_cp_orbit=N_CP_ORBIT,
    )
    chunked._ensure_cpp_setup()

    # ----------------------------------------------------------------
    # Build a *physical* injection on the global Nf x Nt grid, matching
    # gb_chunked_test_script.py / gb_chunked_prior_draws.py:
    #
    #   1) fill the full-grid template via chunked.fill_global at the
    #      injection params (so data_d_full is zero outside the active
    #      m-layer band -- the only realistic regime where layer-groups
    #      give the same inner products as ungrouped),
    #   2) flat invC over the active layers, zero outside.
    #
    # Per-source GB params are clustered around the injection (matches
    # the in-band PE regime; layer-grouping is sensible because every
    # source lives in the same narrow band as the injection).
    # ----------------------------------------------------------------
    f0_mid = 3.0e-3  # mid-band, comfortably inside [layer_df, (Nf-1)*layer_df]
    base = np.array([8.0e-22, f0_mid, 1.0e-17, 0.0, 2.1, 0.24, 1.23, 4.1, 0.04])
    rng2 = np.random.default_rng(1)
    spread = np.array([1e-3, 1e-8, 1e-3, 0.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    params = np.tile(base, (N, 1)) * (1.0 + rng2.standard_normal((N, 9)) * spread)
    params[:, 3] = 0.0  # fddot = 0

    # Build injection template at base params on the full (NCH, Nf, Nt) grid.
    data_d_full = np.zeros((NCH, NF, NT), dtype=float)
    chunked.fill_global(data_d_full, [base])

    # invC is supported only inside the active m-layer band of the
    # injection -- that's the realistic shape of XYZ2SensitivityMatrix
    # built on a WDMSettings whose ind_min_f / ind_max_f bracket the
    # band. Outside the band, invC = 0 so noise there can't enter the
    # inner product (matching ``gb_chunked_test_script.py`` line ~389).
    invC_full = np.zeros_like(data_d_full)
    layer_df = chunked.layer_df
    m_floor = int(f0_mid / layer_df)
    m_lo    = max(0, m_floor - 16)
    m_hi    = min(NF, m_floor + 16)
    invC_full[:, m_lo:m_hi, :] = 1e44

    # ----------------------------------------------------------------
    # C++ get_ll (matches gb_chunked_test_script.py's logl_vec inner call)
    # ----------------------------------------------------------------
    USE_LAYER_GROUPS = os.environ.get("USE_LAYER_GROUPS", "0") == "1"
    GROUP_BAND_LAYERS = int(os.environ.get("GROUP_BAND_LAYERS", "5"))
    MARGIN_LAYERS    = int(os.environ.get("MARGIN_LAYERS", "0"))
    _print_block(f"C++ get_ll  USE_LAYER_GROUPS={USE_LAYER_GROUPS} "
                  f"GROUP_BAND_LAYERS={GROUP_BAND_LAYERS} "
                  f"MARGIN_LAYERS={MARGIN_LAYERS}")
    params_list = [params[i] for i in range(N)]
    _gl_kw = dict(
        use_layer_groups=USE_LAYER_GROUPS,
        group_band_layers=GROUP_BAND_LAYERS,
        margin_layers=MARGIN_LAYERS,
    )
    # warmup (first call also amortises cpp setup costs)
    _ = chunked.get_ll(data_d_full, invC_full, params_list[:1], **_gl_kw)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        dh, hh = chunked.get_ll(data_d_full, invC_full, params_list, **_gl_kw)
    t_cpp = (time.perf_counter() - t0) / REPEATS
    print(f"[cpp]  get_ll  N={N}  : {t_cpp*1e3:10.2f} ms / call  "
          f"({t_cpp/N*1e3:8.3f} ms / source)", flush=True)

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        dh1, hh1 = chunked.get_ll(data_d_full, invC_full, params_list[:1], **_gl_kw)
    t_cpp1 = (time.perf_counter() - t0) / REPEATS
    print(f"[cpp]  get_ll  N=1    : {t_cpp1*1e3:10.2f} ms / call",
          flush=True)

    # Correctness vs ungrouped (only if grouping is on)
    if USE_LAYER_GROUPS:
        dh_g, hh_g = chunked.get_ll(data_d_full, invC_full, params_list, **_gl_kw)
        dh_n, hh_n = chunked.get_ll(data_d_full, invC_full, params_list)
        reldiff_dh = float(np.max(np.abs(dh_g - dh_n) /
                                    np.maximum(np.abs(dh_n), 1e-300)))
        reldiff_hh = float(np.max(np.abs(hh_g - hh_n) /
                                    np.maximum(np.abs(hh_n), 1e-300)))
        print(f"[chk]  layer-groups vs no-groups: dh max reldiff = {reldiff_dh:.3e}  "
              f"hh max reldiff = {reldiff_hh:.3e}", flush=True)

    if os.environ.get("CPP_ONLY", "0") == "1":
        print("\n[setup] CPP_ONLY=1 -- skipping JAX sections", flush=True)
        return 0

    # ----------------------------------------------------------------
    # JAX get_ll (chunked-het JAX kernel)
    # ----------------------------------------------------------------
    _print_block("JAX get_ll (autograd-free forward)")
    chunked._ensure_jax_setup()
    # Warmup (triggers JIT compile).
    print(f"[jax]  warming up get_ll JIT at N={N}...", flush=True)
    t0 = time.perf_counter()
    _ = chunked.get_ll_jax(params[:1], data_d_full, invC_full)  # N=1 warmup
    _ = chunked.get_ll_jax(params,     data_d_full, invC_full)  # full N warmup
    print(f"[jax]  warmup wall time = {time.perf_counter()-t0:.2f} s",
          flush=True)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        dh_j, hh_j = chunked.get_ll_jax(params, data_d_full, invC_full)
    t_jax = (time.perf_counter() - t0) / REPEATS
    print(f"[jax]  get_ll  N={N}  : {t_jax*1e3:10.2f} ms / call  "
          f"({t_jax/N*1e3:8.3f} ms / source)  "
          f"speedup vs C++: {t_cpp/t_jax:6.2f}x", flush=True)

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        dh_j1, hh_j1 = chunked.get_ll_jax(params[:1], data_d_full, invC_full)
    t_jax1 = (time.perf_counter() - t0) / REPEATS
    print(f"[jax]  get_ll  N=1    : {t_jax1*1e3:10.2f} ms / call",
          flush=True)

    # Correctness cross-check at N=1.
    dh_c, hh_c = chunked.get_ll(data_d_full, invC_full, params_list[:1])
    reldiff = float(np.max(np.abs(np.asarray(dh_j1) - dh_c) /
                            np.maximum(np.abs(dh_c), 1e-300)))
    print(f"[chk]  dh JAX vs C++ N=1 max reldiff = {reldiff:.3e}",
          flush=True)

    # ----------------------------------------------------------------
    # JAX get_ll_grad (autograd). Chunk via env to bound memory.
    # ----------------------------------------------------------------
    _print_block("JAX get_ll_grad (autograd)")
    grad_chunk = int(os.environ.get("JAX_GRAD_CHUNK", "0"))
    print(f"[jax]  JAX_GRAD_CHUNK={grad_chunk}  "
          f"(0 = full batch, otherwise chunk leaves into groups)",
          flush=True)
    try:
        print(f"[jax]  warming up get_ll_grad JIT at N={N}...", flush=True)
        t0 = time.perf_counter()
        _ = chunked.get_ll_grad_jax(params, data_d_full, invC_full,
                                     chunk=grad_chunk)
        t_warm = time.perf_counter() - t0
        print(f"[jax]  warmup wall time = {t_warm:.2f} s", flush=True)
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            g = chunked.get_ll_grad_jax(params, data_d_full, invC_full,
                                         chunk=grad_chunk)
        t_g = (time.perf_counter() - t0) / REPEATS
        print(f"[jax]  grad   N={N}  : {t_g*1e3:10.2f} ms / call  "
              f"({t_g/N*1e3:8.3f} ms / source)  "
              f"grad/logl ratio (jax) = {t_g/t_jax:5.2f}x", flush=True)
    except Exception as e:
        print(f"[jax]  grad N={N} FAILED: {type(e).__name__}: {e}", flush=True)

    try:
        t0 = time.perf_counter()
        _ = chunked.get_ll_grad_jax(params[:1], data_d_full, invC_full)
        t_g1 = time.perf_counter() - t0
        print(f"[jax]  grad   N=1 (incl JIT) : {t_g1*1e3:10.2f} ms / call",
              flush=True)
    except Exception as e:
        print(f"[jax]  grad N=1 FAILED: {type(e).__name__}: {e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
