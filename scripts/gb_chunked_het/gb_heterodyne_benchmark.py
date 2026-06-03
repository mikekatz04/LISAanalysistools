#!/usr/bin/env python
"""
Verify + benchmark the cupy/numpy batched single-chunk heterodyne
implementations against C++ and JAX backends.

Backends compared
-----------------
* **C++ on CPU**   -- ``GBFDComputations(force_backend='cpu').get_ll_fd``
                       / ``GBWDMComputations(force_backend='cpu').get_ll_wdm``
* **C++ on CUDA**  -- same, ``force_backend='cuda12x'``  (GPU only)
* **JAX on GPU**   -- ``GBWDMComputations(force_backend='jax').get_ll_wdm``
                       (FD has no JAX backend)
* **Batched cupy** -- :class:`GBHeterodyneFDGetLL` / :class:`GBHeterodyneWDMGetLL`
                       in this directory, with numpy on CPU or cupy on GPU.

Locally (no GPU) the script verifies the batched-numpy version against
the C++-CPU reference. On the cluster it additionally times all
backends side-by-side.

Verification tolerance: ``reldiff <= 1e-10`` (well above floating-point
reassociation noise).

Run
---
::

    # Locally -- verifies CPU paths only:
    python gb_heterodyne_benchmark.py

    # On cluster with GPU:
    python gb_heterodyne_benchmark.py --backends cpu,cuda12x,jax \\
        --num-bins 100,1000,10000 --skip-cpu-large
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse.tdiconfig import TDIConfig


def _has_gpu_backend(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "jax":
        try:
            import jax
            # Heuristic: GPU device present?
            return any(d.platform != "cpu" for d in jax.devices())
        except Exception:
            return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _resolve_backends(requested: list[str]) -> list[str]:
    out = []
    for b in requested:
        if b == "cpu" or _has_gpu_backend(b):
            out.append(b)
        else:
            print(f"  ! backend '{b}' not available -- skipping")
    return out


def _time(callable_, n_warmup: int = 1, n_iter: int = 3) -> float:
    """Best-of-3 wall clock, after warmup, with cuda/jax sync if available."""
    sync = _sync()
    for _ in range(n_warmup):
        callable_(); sync()
    best = float("inf")
    for _ in range(n_iter):
        t0 = time.perf_counter()
        callable_(); sync()
        best = min(best, time.perf_counter() - t0)
    return best


def _sync():
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            cupy_sync = cupy.cuda.runtime.deviceSynchronize
        else:
            cupy_sync = lambda: None
    except Exception:
        cupy_sync = lambda: None
    try:
        import jax
        def jax_sync():
            for d in jax.devices():
                pass  # JAX is sync on host access; arrays are blocked on .item() etc.
    except Exception:
        jax_sync = lambda: None

    def both():
        cupy_sync()
        jax_sync()
    return both


# ---------------------------------------------------------------------------
# common setup
# ---------------------------------------------------------------------------
def _build_setup(backend: str, Nf: int, Nt: int, dt: float,
                 N_sparse: int, nchannels: int):
    """Common (orbits, tdi_config, wdm_settings, t_start, Tobs, df) for one backend."""
    Tobs = Nf * Nt * dt
    df = 1.0 / Tobs
    t_start = 0.0
    tdi_config = TDIConfig("2nd generation")
    orbits = EqualArmlengthOrbits(force_backend=backend)
    wdm_set = WDMSettings(
        Nf, Nt, dt, min_time=0.0, max_time=Tobs, force_backend=backend,
    )
    return tdi_config, orbits, wdm_set, t_start, Tobs, df


def _make_test_params(num_bin: int, rng):
    """Random GB params with f0 around 20 mHz."""
    p = np.zeros((num_bin, 9))
    p[:, 0] = rng.uniform(5e-23, 1e-22, num_bin)              # amp
    p[:, 1] = 20e-3 + rng.normal(0.0, 1e-4, num_bin)          # f0
    p[:, 2] = 1e-14                                            # fdot
    p[:, 3] = 0.0                                              # fddot
    p[:, 4] = rng.uniform(0, 2 * np.pi, num_bin)              # phi0
    p[:, 5] = rng.uniform(0, np.pi, num_bin)                  # inc
    p[:, 6] = rng.uniform(0, np.pi, num_bin)                  # psi
    p[:, 7] = rng.uniform(0, 2 * np.pi, num_bin)              # lam
    p[:, 8] = rng.uniform(-np.pi / 2, np.pi / 2, num_bin)     # beta
    return p


# ---------------------------------------------------------------------------
# FD path
# ---------------------------------------------------------------------------
def run_fd_backend(backend, params, num_bin, Nf, Nt, dt, N_sparse, nchannels,
                   rng):
    """Run GBFDComputations.get_ll_fd on the given backend; return (ll, d_h, h_h, dt_call)."""
    from fastlisaresponse.gbcomps import GBFDComputations

    tdi_config, orbits, _wdm, t_start, Tobs, df = _build_setup(
        backend, Nf, Nt, dt, N_sparse, nchannels,
    )
    N_dense = int(round(Tobs / dt))
    n_rfft = N_dense // 2 + 1

    # Backend-side arrays.
    if backend == "cpu":
        xp = np
    else:
        import cupy
        xp = cupy

    data_fd = (xp.asarray(rng.standard_normal((1, nchannels, n_rfft)))
               + 1j * xp.asarray(rng.standard_normal((1, nchannels, n_rfft))))
    invC = xp.zeros((1, nchannels, nchannels, n_rfft))
    for c in range(nchannels):
        invC[0, c, c, :] = xp.asarray(rng.uniform(0.5, 1.5, n_rfft) ** 2)
    invC[:, :, :, 0] = 0.0

    fd = GBFDComputations(
        T=Tobs, t_ref=t_start, t_start=t_start, N_sparse=N_sparse, df=df,
        data_fd=data_fd, invC=invC,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=backend, tdi_type="XYZ",
    )

    p_xp = xp.asarray(params)
    fn = lambda: fd.get_ll_fd(p_xp, convert_to_ra_dec=False)
    t_call = _time(fn, n_warmup=1, n_iter=3)
    ll = fn()
    return (np.asarray(ll), np.asarray(fd.d_h_out), np.asarray(fd.h_h_out),
            t_call, data_fd, invC)


def run_fd_batched(backend, params, num_bin, Nf, Nt, dt, N_sparse, nchannels,
                    data_fd, invC):
    """Run the batched GBHeterodyneFDGetLL on the given backend."""
    from gb_heterodyne_fd_batched import GBHeterodyneFDGetLL

    tdi_config, orbits, _wdm, t_start, Tobs, df = _build_setup(
        backend, Nf, Nt, dt, N_sparse, nchannels,
    )

    # data_fd / invC may already be on the right backend.
    batched = GBHeterodyneFDGetLL(
        T=Tobs, t_ref=t_start, t_start=t_start, N_sparse=N_sparse, df=df,
        data_fd=data_fd, invC=invC,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=backend, tdi_type="XYZ",
    )

    if backend == "cpu":
        xp = np
    else:
        import cupy
        xp = cupy
    p_xp = xp.asarray(params)
    fn = lambda: batched.get_ll_fd(p_xp, convert_to_ra_dec=False)
    t_call = _time(fn, n_warmup=1, n_iter=3)
    ll = fn()
    return (np.asarray(ll), np.asarray(batched.d_h_out),
            np.asarray(batched.h_h_out), t_call)


# ---------------------------------------------------------------------------
# WDM path
# ---------------------------------------------------------------------------
def run_wdm_backend(backend, params, num_bin, Nf, Nt, dt, N_sparse, nchannels,
                     rng):
    """Run GBWDMComputations.get_ll_wdm; use Nt_sub=Nt for single-chunk."""
    from fastlisaresponse.gbcomps import GBWDMComputations
    from lisatools.analysiscontainer import AnalysisContainerArray
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.domains import WDMSignal

    tdi_config, orbits, wdm_set, t_start, Tobs, df = _build_setup(
        backend, Nf, Nt, dt, N_sparse, nchannels,
    )

    if backend == "cpu":
        xp = np
    elif backend == "jax":
        import jax.numpy as jnp
        xp = jnp
    else:
        import cupy
        xp = cupy

    # Build random WDM data.
    data_wdm_arr = rng.standard_normal((1, nchannels, Nf, Nt))

    # Wrap as AnalysisContainerArray (matches GBWDMComputations contract).
    # Avoid building a real sens_mat -- use diagonal invC stand-in.
    wdm_sig = WDMSignal(data_wdm_arr[0], wdm_set)
    ac = AnalysisContainerArray(
        [wdm_sig],
        sens_mat_class=XYZ2SensitivityMatrix,
        sens_mat_kwargs=dict(model="scirdv1"),
    )

    wdm = GBWDMComputations(
        wdm_settings=wdm_set, t_ref=t_start,
        Nt_sub=Nt, n_pad=0,         # n_pad=0 + Nt_sub=Nt -> single chunk
        N_sparse=N_sparse,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=backend, tdi_type="XYZ",
    )

    p_in = xp.asarray(params) if backend != "cpu" else params
    fn = lambda: wdm.get_ll_wdm(p_in, ac, convert_to_ra_dec=False,
                                use_layer_groups=False)
    t_call = _time(fn, n_warmup=1, n_iter=3)
    ll = fn()
    return (np.asarray(ll), np.asarray(wdm.d_h_out), np.asarray(wdm.h_h_out),
            t_call, ac, wdm_set)


def run_wdm_batched(backend, params, num_bin, Nf, Nt, dt, N_sparse, nchannels,
                     ac, wdm_set):
    """Run the batched GBHeterodyneWDMGetLL."""
    from gb_heterodyne_wdm_batched import GBHeterodyneWDMGetLL

    tdi_config, orbits, _wdm, t_start, Tobs, df = _build_setup(
        backend, Nf, Nt, dt, N_sparse, nchannels,
    )

    if backend == "cpu":
        xp = np
    else:
        import cupy
        xp = cupy

    # Pull data + invC out of the AnalysisContainer (matches what
    # GBWDMComputations sends to its kernel).
    data_wdm = xp.asarray(ac.linear_data_arr[0]).reshape(
        1, nchannels, Nf, Nt,
    )
    invC_wdm = xp.asarray(ac.linear_psd_arr[0]).reshape(
        1, nchannels, nchannels, Nf, Nt,
    )
    wdm_window = xp.asarray(wdm_set.window)

    batched = GBHeterodyneWDMGetLL(
        T=Tobs, t_ref=t_start, t_start=t_start, N_sparse=N_sparse,
        Nf=Nf, Nt_sub=Nt, dt=dt, wdm_window=wdm_window,
        data_wdm=data_wdm, invC_wdm=invC_wdm,
        orbits=orbits, tdi_config=tdi_config,
        force_backend=backend, tdi_type="XYZ",
    )
    p_xp = xp.asarray(params)
    fn = lambda: batched.get_ll_wdm(p_xp, convert_to_ra_dec=False)
    t_call = _time(fn, n_warmup=1, n_iter=3)
    ll = fn()
    return (np.asarray(ll), np.asarray(batched.d_h_out),
            np.asarray(batched.h_h_out), t_call)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def _compare(label, ref, new, tol=1e-10):
    r = np.abs(new - ref) / np.maximum(np.abs(ref), 1e-300)
    rmax = float(r.max())
    print(f"    {label:>20s}   reldiff max = {rmax:.3e}   "
          f"{'PASS' if rmax < tol else 'FAIL'}")
    return rmax < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="cpu",
                    help="comma-separated: cpu, cuda12x, jax")
    ap.add_argument("--num-bins", default="2,10",
                    help="comma-separated num_bin values")
    ap.add_argument("--Nf", type=int, default=64)
    ap.add_argument("--Nt", type=int, default=64)
    ap.add_argument("--dt", type=float, default=15.0)
    ap.add_argument("--N-sparse", type=int, default=256)
    ap.add_argument("--nchannels", type=int, default=3)
    ap.add_argument("--skip-cpu-large", action="store_true",
                    help="skip CPU runs when num_bin > 1000 (CPU too slow)")
    args = ap.parse_args()

    backends = _resolve_backends([b.strip() for b in args.backends.split(",")])
    nbs = [int(x) for x in args.num_bins.split(",")]
    rng = np.random.default_rng(42)

    print("=" * 76)
    print(f"Backends: {backends}")
    print(f"num_bin sweep: {nbs}")
    print(f"WDM grid: Nf={args.Nf} Nt={args.Nt} dt={args.dt} -> "
          f"Tobs={args.Nf * args.Nt * args.dt:.1f}s "
          f"({args.Nf * args.Nt * args.dt / YRSID_SI:.3f} yr)")
    print(f"N_sparse={args.N_sparse} nchannels={args.nchannels}")
    print("=" * 76)

    for num_bin in nbs:
        if args.skip_cpu_large and num_bin > 1000 and "cpu" in backends:
            print(f"\n[num_bin={num_bin}]  skipping CPU (large batch)")
            this_backends = [b for b in backends if b != "cpu"]
        else:
            this_backends = backends

        params = _make_test_params(num_bin, rng)
        print(f"\n--- num_bin = {num_bin} -----------------------------------")

        # ------------- FD path -------------
        print("\n  FD path (single-chunk heterodyne, FD inner product):")
        ref_dh = ref_hh = None
        ref_dt = None
        ref_data_fd = ref_invC = None
        # Need a consistent (data_fd, invC) across backends; build once on CPU
        # and move to the GPU backends for fair compare.
        timings_fd = {}
        for backend in this_backends:
            try:
                if backend != "cpu" and ref_data_fd is not None:
                    # Move CPU reference data to this backend.
                    import cupy
                    df_b = cupy.asarray(ref_data_fd)
                    invC_b = cupy.asarray(ref_invC)
                    # Build a fresh GBFDComputations with these arrays.
                    from fastlisaresponse.gbcomps import GBFDComputations
                    tdi_config, orbits, _w, t_start, Tobs, df = _build_setup(
                        backend, args.Nf, args.Nt, args.dt,
                        args.N_sparse, args.nchannels,
                    )
                    fd = GBFDComputations(
                        T=Tobs, t_ref=t_start, t_start=t_start,
                        N_sparse=args.N_sparse, df=df,
                        data_fd=df_b, invC=invC_b,
                        orbits=orbits, tdi_config=tdi_config,
                        force_backend=backend, tdi_type="XYZ",
                    )
                    p_xp = cupy.asarray(params)
                    fn = lambda: fd.get_ll_fd(p_xp, convert_to_ra_dec=False)
                    t = _time(fn, 1, 3); ll = fn()
                    dh = np.asarray(fd.d_h_out); hh = np.asarray(fd.h_h_out)
                    data_fd_use, invC_use = df_b, invC_b
                else:
                    ll, dh, hh, t, data_fd_use, invC_use = run_fd_backend(
                        backend, params, num_bin, args.Nf, args.Nt,
                        args.dt, args.N_sparse, args.nchannels, rng,
                    )
                    if backend == "cpu":
                        ref_data_fd, ref_invC = (
                            np.asarray(data_fd_use), np.asarray(invC_use),
                        )

                timings_fd[f"FD-C++-{backend}"] = t
                if ref_dh is None:
                    ref_dh = dh; ref_hh = hh
                    print(f"    {'C++ '+backend:>20s}   t={t*1e3:8.2f} ms  "
                          f"(REFERENCE)")
                else:
                    ok_dh = _compare(f"C++ {backend} <d|h>", ref_dh, dh)
                    ok_hh = _compare(f"C++ {backend} <h|h>", ref_hh, hh)
                    print(f"    {'C++ '+backend:>20s}   t={t*1e3:8.2f} ms")

                # Batched cupy version on this backend.
                ll_b, dh_b, hh_b, t_b = run_fd_batched(
                    backend, params, num_bin, args.Nf, args.Nt,
                    args.dt, args.N_sparse, args.nchannels,
                    data_fd_use, invC_use,
                )
                timings_fd[f"FD-batched-{backend}"] = t_b
                ok_dh = _compare(f"batched {backend} <d|h>", ref_dh, dh_b)
                ok_hh = _compare(f"batched {backend} <h|h>", ref_hh, hh_b)
                print(f"    {'batched '+backend:>20s}   t={t_b*1e3:8.2f} ms"
                      f"  (vs C++ ref)")
            except Exception as e:
                print(f"    FD {backend}: FAIL with {type(e).__name__}: {e}")

        print("\n  Timing summary (FD):")
        for k, v in timings_fd.items():
            print(f"    {k:30s}  {v*1e3:10.2f} ms   "
                  f"({v*1e6/num_bin:8.2f} us/binary)")

        # ------------- WDM path -------------
        print("\n  WDM path (single-chunk Nt_sub=Nt, WDM inner product):")
        ref_dh = ref_hh = None
        timings_wdm = {}
        ac_ref = wdm_set_ref = None
        for backend in this_backends:
            try:
                if backend != "cpu" and ac_ref is not None:
                    # Move ac data to backend? Easier: re-run on this backend
                    # which builds its own ac.
                    ll, dh, hh, t, ac, wdm_set = run_wdm_backend(
                        backend, params, num_bin, args.Nf, args.Nt,
                        args.dt, args.N_sparse, args.nchannels, rng,
                    )
                else:
                    ll, dh, hh, t, ac, wdm_set = run_wdm_backend(
                        backend, params, num_bin, args.Nf, args.Nt,
                        args.dt, args.N_sparse, args.nchannels, rng,
                    )
                    if backend == "cpu":
                        ac_ref, wdm_set_ref = ac, wdm_set

                timings_wdm[f"WDM-C++-{backend}"] = t
                if ref_dh is None:
                    ref_dh = dh; ref_hh = hh
                    print(f"    {backend+' (C++/JAX)':>20s}   "
                          f"t={t*1e3:8.2f} ms  (REFERENCE)")
                else:
                    _compare(f"{backend} <d|h>", ref_dh, dh)
                    _compare(f"{backend} <h|h>", ref_hh, hh)
                    print(f"    {backend+' (C++/JAX)':>20s}   "
                          f"t={t*1e3:8.2f} ms")

                # Batched
                ll_b, dh_b, hh_b, t_b = run_wdm_batched(
                    backend, params, num_bin, args.Nf, args.Nt,
                    args.dt, args.N_sparse, args.nchannels,
                    ac, wdm_set,
                )
                timings_wdm[f"WDM-batched-{backend}"] = t_b
                _compare(f"batched {backend} <d|h>", ref_dh, dh_b)
                _compare(f"batched {backend} <h|h>", ref_hh, hh_b)
                print(f"    {'batched '+backend:>20s}   "
                      f"t={t_b*1e3:8.2f} ms")
            except Exception as e:
                print(f"    WDM {backend}: FAIL with {type(e).__name__}: {e}")

        print("\n  Timing summary (WDM):")
        for k, v in timings_wdm.items():
            print(f"    {k:30s}  {v*1e3:10.2f} ms   "
                  f"({v*1e6/num_bin:8.2f} us/binary)")


if __name__ == "__main__":
    main()
