#!/usr/bin/env python
"""
Three-way WDM speed benchmark: JAX vs cupy-batched vs C++ CUDA on GPU.

Mirrors the WDM-grid setup from ``gb_chunked_test_script.py`` (Nf=1460,
Nt=2560, dt=10s, N_sparse=256, 1-yr obs) but configures all three
paths to the SAME single-chunk-whole-obs heterodyne algorithm
(``Nt_sub == Nt``, ``n_pad == 0`` -> ``n_chunks == 1``):

    A) C++ CUDA          -- GBWDMComputations(force_backend='cuda12x')
    B) JAX-GPU           -- GBWDMComputations(force_backend='jax')
    C) cupy batched      -- GBHeterodyneWDMGetLL on cuda12x

All three compute the same ``<d|h>`` and ``<h|h>`` numerically (modulo
floating-point reassociation noise), so we get a true apples-to-apples
speed comparison plus an algorithmic-correctness cross-check.

Run
---
::

    # Default sweep: num_bin in {10, 100, 1000, 10000}
    python gb_speed_benchmark.py

    # Override sweep, skip slow paths at large batch:
    python gb_speed_benchmark.py --num-bins 1000,10000,100000 \\
        --skip-jax-large --skip-cpp-large

Environment knobs (mirror gb_chunked_test_script.py for compatibility):
    NF / NT / NT_SUB / N_SPARSE / N_CP_SIG / N_CP_ORBIT

Output
------
For each ``num_bin``, prints per-method wall-clock, per-binary microseconds,
and the relative speedup of (C) vs (A) and (B). Verification block at
``num_bin=10`` checks the C++ and JAX outputs match each other before
timing (they share an algorithm) and reports the cupy-batched ``<d|h>``,
``<h|h>`` for sanity.

This script does *not* try to make method (C) match the chunked-het
likelihood numerically; they use different algorithms (single chunk vs
n_chunks chunks). The point is throughput at scale.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# environment / backend availability
# ---------------------------------------------------------------------------
def _have_cupy_gpu() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _have_jax_gpu() -> bool:
    try:
        import jax
        return any(d.platform != "cpu" for d in jax.devices())
    except Exception:
        return False


def _sync_all():
    """Force device sync across cupy + JAX."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    try:
        import jax
        # JAX blocks on host access; explicit block_until_ready of a tiny
        # dummy works for the last computation, but the cleanest GPU sync
        # is .block_until_ready() on a known array. We rely on the caller
        # calling .block_until_ready() on JAX results before timing stops.
    except Exception:
        pass


def _time_best(callable_, n_warmup: int = 2, n_iter: int = 5) -> float:
    """Best-of-N timing with warmup and pre/post sync."""
    for _ in range(n_warmup):
        out = callable_()
        # Force JAX to actually compute if applicable.
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, (tuple, list)):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
        _sync_all()
    best = float("inf")
    for _ in range(n_iter):
        _sync_all()
        t0 = time.perf_counter()
        out = callable_()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, (tuple, list)):
            for x in out:
                if hasattr(x, "block_until_ready"):
                    x.block_until_ready()
        _sync_all()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# WDM grid setup (matches gb_chunked_test_script.py defaults)
# ---------------------------------------------------------------------------
def build_problem(backend: str, num_bin: int, rng):
    """Returns a dict of (params, data, invC, wdm_set, t_ref, t_start, ...)
    for the given backend, sized for the gb_chunked_test_script grid."""
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.utils.constants import YRSID_SI
    from lisatools.domains import WDMSettings
    from fastlisaresponse.tdiconfig import TDIConfig

    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))
    Tobs = Nf * Nt * dt
    N_sparse = int(os.environ.get("N_SPARSE", 256))
    Nt_sub = int(os.environ.get("NT_SUB", 256))
    N_cp_sig = int(os.environ.get("N_CP_SIG", 0))
    N_cp_orbit = int(os.environ.get("N_CP_ORBIT", 0))
    min_freq = 0.0001
    max_freq = 35.0e-3
    nchannels = 3

    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_ref = t_start

    tdi_config = TDIConfig("2nd generation")
    orbits = EqualArmlengthOrbits(force_backend=backend)
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=20.0 * Nf * dt, max_time=(Nt - 20) * Nf * dt,
        force_backend=backend,
    )

    # Random GB parameters; carrier in a layer near 3 mHz (in-band).
    layer_df = float(wdm_set.layer_df)
    m_ref = int(3e-3 / layer_df)
    p = np.zeros((num_bin, 9))
    p[:, 0] = rng.uniform(5e-23, 2e-22, num_bin)
    # spread carrier over a handful of layers around m_ref
    p[:, 1] = (m_ref + rng.uniform(-5, 5, num_bin)) * layer_df
    p[:, 2] = 1e-17
    p[:, 3] = 0.0
    p[:, 4] = rng.uniform(0, 2 * np.pi, num_bin)
    p[:, 5] = rng.uniform(0, np.pi, num_bin)
    p[:, 6] = rng.uniform(0, np.pi, num_bin)
    p[:, 7] = rng.uniform(0, 2 * np.pi, num_bin)
    p[:, 8] = rng.uniform(-0.5, 0.5, num_bin)

    return dict(
        dt=dt, Nf=Nf, Nt=Nt, Tobs=Tobs, N_sparse=N_sparse, Nt_sub=Nt_sub,
        N_cp_sig=N_cp_sig, N_cp_orbit=N_cp_orbit,
        t_start=t_start, t_ref=t_ref, nchannels=nchannels,
        orbits=orbits, tdi_config=tdi_config, wdm_set=wdm_set,
        params=p, layer_df=layer_df,
    )


def make_holder(backend: str, prob: dict):
    """Build a _FullGridWDMHolder duck-type with random data+diag-invC."""
    if backend == "cpu":
        xp = np
    elif backend == "jax":
        import jax.numpy as jnp
        xp = jnp
    else:
        import cupy
        xp = cupy

    Nf = prob["Nf"]; Nt = prob["Nt"]; nch = prob["nchannels"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((nch, Nf, Nt)) * 1e-22
    invC_diag = rng.uniform(1e20, 1e22, (nch, Nf, Nt))

    class _Holder:
        def __init__(self, d, ic):
            self.linear_data_arr = [xp.asarray(d).ravel()]
            self.linear_psd_arr = [xp.asarray(ic).ravel()]
        def __len__(self):
            return 1

    return _Holder(data, invC_diag), data, invC_diag


# ---------------------------------------------------------------------------
# the three benchmarked paths
# ---------------------------------------------------------------------------
def bench_cpp_cuda(prob, holder, params, label="C++/CUDA"):
    """Path A: GBWDMComputations(force_backend='cuda12x').get_ll_wdm.

    Configured for SINGLE-CHUNK heterodyne (Nt_sub=Nt, n_pad=0 -> n_chunks=1)
    so the C++ kernel runs the same algorithm as method (C) cupy single-chunk
    and method (B) JAX single-chunk for apples-to-apples timing.
    """
    from fastlisaresponse.gbcomps import GBWDMComputations
    import cupy

    comp = GBWDMComputations(
        wdm_settings=prob["wdm_set"], t_ref=prob["t_ref"],
        Nt_sub=prob["Nt"], n_pad=0,                  # single chunk = whole obs
        N_sparse=prob["N_sparse"],
        N_cp_sig=prob["N_cp_sig"], N_cp_orbit=prob["N_cp_orbit"],
        orbits=prob["orbits"], tdi_config="2nd generation",
        force_backend="cuda12x", d_d=0.0, tdi_type="AET",
    )
    p_xp = cupy.asarray(params)
    fn = lambda: comp.get_ll_wdm(
        p_xp, holder, convert_to_ra_dec=False,
        use_layer_groups=False, grid_dim=0,
    )
    t = _time_best(fn)
    out = fn()
    return t, np.asarray(out)


def bench_jax(prob, holder_jax, params, label="JAX/GPU"):
    """Path B: GBWDMComputations(force_backend='jax').get_ll_wdm.

    SINGLE-CHUNK configuration (Nt_sub=Nt, n_pad=0) so JAX runs the same
    single-chunk-whole-obs heterodyne as method (C) cupy.
    """
    from fastlisaresponse.gbcomps import GBWDMComputations
    import jax.numpy as jnp

    comp = GBWDMComputations(
        wdm_settings=prob["wdm_set"], t_ref=prob["t_ref"],
        Nt_sub=prob["Nt"], n_pad=0,                  # single chunk
        N_sparse=prob["N_sparse"],
        N_cp_sig=prob["N_cp_sig"], N_cp_orbit=prob["N_cp_orbit"],
        orbits=prob["orbits"], tdi_config="2nd generation",
        force_backend="jax", d_d=0.0, tdi_type="AET",
    )
    p_xp = jnp.asarray(params)
    fn = lambda: comp.get_ll_wdm(
        p_xp, holder_jax, convert_to_ra_dec=False,
        use_layer_groups=False, grid_dim=0,
    )
    t = _time_best(fn)
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return t, np.asarray(out)


def bench_cupy_batched_wdm(prob, data_wdm_np, invC_diag_np, params,
                            label="cupy WDM"):
    """Path C-WDM: GBHeterodyneWDMGetLL on cuda12x backend."""
    from gb_heterodyne_wdm_batched import GBHeterodyneWDMGetLL
    import cupy

    Nf = prob["Nf"]; Nt = prob["Nt"]; nch = prob["nchannels"]
    # Diagonal invC -> XYZ Hermitian form for the batched class.
    invC_xyz = np.zeros((1, nch, nch, Nf, Nt))
    for c in range(nch):
        invC_xyz[0, c, c, :, :] = invC_diag_np[c]

    comp = GBHeterodyneWDMGetLL(
        T=prob["Tobs"], t_ref=prob["t_ref"], t_start=prob["t_start"],
        N_sparse=prob["N_sparse"], Nf=Nf, Nt_sub=Nt, dt=prob["dt"],
        wdm_window=prob["wdm_set"].window,
        data_wdm=data_wdm_np[None],
        invC_wdm=invC_xyz,
        orbits=prob["orbits"], tdi_config="2nd generation",
        force_backend="cuda12x", tdi_type="XYZ",
    )
    p_xp = cupy.asarray(params)
    fn = lambda: comp.get_ll_wdm(p_xp, convert_to_ra_dec=False)
    t = _time_best(fn)
    out = fn()
    return t, np.asarray(out)


def bench_cupy_batched_fd(prob, params, label="cupy FD"):
    """Path C-FD: GBHeterodyneFDGetLL on cuda12x backend (FD inner product)."""
    from gb_heterodyne_fd_batched import GBHeterodyneFDGetLL
    import cupy

    Nf = prob["Nf"]; Nt = prob["Nt"]; nch = prob["nchannels"]
    N_dense = int(round(prob["Tobs"] / prob["dt"]))
    n_rfft = N_dense // 2 + 1
    df = 1.0 / prob["Tobs"]
    rng = np.random.default_rng(0)
    data_fd = (rng.standard_normal((1, nch, n_rfft))
               + 1j * rng.standard_normal((1, nch, n_rfft)))
    invC = np.zeros((1, nch, nch, n_rfft))
    for c in range(nch):
        invC[0, c, c, :] = rng.uniform(1e20, 1e22, n_rfft)

    comp = GBHeterodyneFDGetLL(
        T=prob["Tobs"], t_ref=prob["t_ref"], t_start=prob["t_start"],
        N_sparse=prob["N_sparse"], df=df,
        data_fd=cupy.asarray(data_fd), invC=cupy.asarray(invC),
        orbits=prob["orbits"], tdi_config="2nd generation",
        force_backend="cuda12x", tdi_type="XYZ",
    )
    p_xp = cupy.asarray(params)
    fn = lambda: comp.get_ll_fd(p_xp, convert_to_ra_dec=False)
    t = _time_best(fn)
    out = fn()
    return t, np.asarray(out)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-bins", default="10,100,1000,10000")
    ap.add_argument("--skip-jax-large", action="store_true",
                    help="skip JAX path at num_bin > 1000")
    ap.add_argument("--skip-cpp-large", action="store_true")
    args = ap.parse_args()

    have_cupy = _have_cupy_gpu()
    have_jax = _have_jax_gpu()
    print("=" * 76)
    print(f"GPU availability: cupy={have_cupy}  jax={have_jax}")
    print("=" * 76)
    if not have_cupy:
        print("No CUDA GPU detected for cupy -- this script needs a GPU.")
        print("Run on the cluster after `module load cuda` etc.")
        return 1

    nbs = [int(x) for x in args.num_bins.split(",")]
    rng = np.random.default_rng(123)

    # Build holders for C++ and JAX (different backends -> different array
    # types). Same underlying numbers though.
    rng_data = np.random.default_rng(7)
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))
    nch = 3
    data_np = rng_data.standard_normal((nch, Nf, Nt)) * 1e-22
    invC_diag_np = rng_data.uniform(1e20, 1e22, (nch, Nf, Nt))

    for num_bin in nbs:
        prob = build_problem("cuda12x", num_bin, rng)
        # Place data on cupy and on jax.
        import cupy
        class _Holder:
            def __init__(self, d, ic):
                self.linear_data_arr = [d.ravel()]
                self.linear_psd_arr = [ic.ravel()]
            def __len__(self):
                return 1

        d_cp = cupy.asarray(data_np); ic_cp = cupy.asarray(invC_diag_np)
        holder_cp = _Holder(d_cp, ic_cp)

        if have_jax:
            import jax.numpy as jnp
            holder_jx = _Holder(jnp.asarray(data_np).ravel().reshape(d_cp.shape),
                                 jnp.asarray(invC_diag_np))
            # Need to wrap as ravel'd; reuse class with jnp arrays.
            class _HJ:
                def __init__(self, d, ic):
                    self.linear_data_arr = [d.ravel()]
                    self.linear_psd_arr = [ic.ravel()]
                def __len__(self):
                    return 1
            holder_jx = _HJ(jnp.asarray(data_np), jnp.asarray(invC_diag_np))
        else:
            holder_jx = None

        print(f"\n--- num_bin = {num_bin} ---------------------------")
        results = {}

        # ---- A: C++ CUDA chunked-het ----
        if not (args.skip_cpp_large and num_bin > 1000):
            try:
                t_a, out_a = bench_cpp_cuda(prob, holder_cp, prob["params"])
                results["C++/CUDA chunked-het"] = (t_a, out_a)
                print(f"  C++/CUDA chunked-het       {t_a*1e3:9.2f} ms  "
                      f"({t_a*1e6/num_bin:8.2f} us/bin)")
            except Exception as e:
                print(f"  C++/CUDA: FAIL ({type(e).__name__}: {e})")

        # ---- B: JAX chunked-het ----
        if have_jax and not (args.skip_jax_large and num_bin > 1000):
            try:
                t_b, out_b = bench_jax(prob, holder_jx, prob["params"])
                results["JAX/GPU chunked-het"] = (t_b, out_b)
                print(f"  JAX/GPU chunked-het        {t_b*1e3:9.2f} ms  "
                      f"({t_b*1e6/num_bin:8.2f} us/bin)")
                # Verify C++ vs JAX (same single-chunk algorithm; should match).
                if "C++/CUDA chunked-het" in results:
                    out_a = results["C++/CUDA chunked-het"][1]
                    rel = np.max(np.abs(out_b - out_a)
                                  / np.maximum(np.abs(out_a), 1e-300))
                    print(f"    JAX vs C++ ll reldiff   = {rel:.3e}  "
                          f"{'PASS' if rel < 1e-8 else 'FAIL'}")
            except Exception as e:
                print(f"  JAX/GPU: FAIL ({type(e).__name__}: {e})")

        # ---- C: cupy batched WDM single-chunk ----
        try:
            t_c, out_c = bench_cupy_batched_wdm(
                prob, data_np, invC_diag_np, prob["params"],
            )
            results["cupy batched WDM"] = (t_c, out_c)
            print(f"  cupy batched WDM (single)  {t_c*1e3:9.2f} ms  "
                  f"({t_c*1e6/num_bin:8.2f} us/bin)")
            # cupy vs C++ -- same single-chunk algorithm; should match.
            if "C++/CUDA chunked-het" in results:
                out_a = results["C++/CUDA chunked-het"][1]
                rel = np.max(np.abs(out_c - out_a)
                              / np.maximum(np.abs(out_a), 1e-300))
                print(f"    cupy vs C++ ll reldiff  = {rel:.3e}  "
                      f"{'PASS' if rel < 1e-8 else 'FAIL'}")
        except Exception as e:
            print(f"  cupy batched WDM: FAIL ({type(e).__name__}: {e})")

        # ---- Relative speedups ----
        if "C++/CUDA chunked-het" in results:
            t_ref = results["C++/CUDA chunked-het"][0]
            print(f"  ---- vs C++/CUDA ----")
            for name, (t, _) in results.items():
                if name == "C++/CUDA chunked-het":
                    continue
                ratio = t_ref / t
                print(f"    {name:30s}  {ratio:6.2f}x  "
                      f"({'faster' if ratio > 1 else 'slower'})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
