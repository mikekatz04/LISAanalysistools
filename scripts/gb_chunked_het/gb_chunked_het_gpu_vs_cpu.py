#!/usr/bin/env python
"""GPU-vs-CPU parity check for the chunked-heterodyne WDM GB likelihood.

One small, self-contained example (the ``gb_chunked_swap_ll_validate.py``
scaffolding): a tiny WDM grid, two GB sources in one 5-layer band, and a
batch of perturbed candidates. ``GBWDMComputations`` is built twice — once
``force_backend="cpu"``, once on the GPU backend — and every likelihood
surface is compared element-wise:

  1. ``fill_global_wdm``   — the dense template coefficients themselves;
  2. ``get_ll_wdm``        — ``d_h`` / ``h_h`` over the candidate batch;
  3. ``get_swap_ll_wdm``   — the 5 swap terms (d_h_a, d_h_r, aa, rr, ar).

CPU and GPU run the same C++ kernel structure (sprint backend hierarchy),
so agreement should be at floating-point order-of-operations level; the
default tolerances are rtol=1e-10 (ll terms) / 1e-12 (templates, relative
to the max coefficient).

Run (cluster):
    GPU_BACKEND=cuda12x python gb_chunked_het_gpu_vs_cpu.py

Harness self-test (CPU vs CPU; all diffs must be exactly 0):
    GPU_BACKEND=cpu python gb_chunked_het_gpu_vs_cpu.py

Env knobs: GPU_BACKEND (default cuda12x), N_CAND (default 8), SEED,
N_CP_SIG / N_CP_ORBIT (cache sizes; both the direct 0/0 and the production
48/32 configurations run by default), RTOL_LL, RTOL_TEMPLATE.
"""
from __future__ import annotations

import os

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations


class _FullGridWDMHolder:
    """Duck-type for the ``wdm_holder`` argument: full-grid data + invC."""

    def __init__(self, xp, data_full, invC_diag_full):
        self.linear_data_arr = [xp.ascontiguousarray(xp.asarray(data_full)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(xp.asarray(invC_diag_full)).ravel()]

    def __len__(self):
        return 1


def _to_np(arr):
    return np.asarray(arr.get() if hasattr(arr, "get") else arr)


def build_comp(backend, wdm_kwargs, comp_kwargs):
    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(force_backend=backend, **wdm_kwargs)
    comp = GBWDMComputations(
        wdm_set, orbits=orbits, force_backend=backend, **comp_kwargs
    )
    return comp, wdm_set


def evaluate(comp, wdm_set, A, B, cands, use_layer_groups):
    """All chunked-het likelihood surfaces on this comp's backend (as numpy)."""
    xp = comp.backend.xp
    nch = 3
    Nf, Nt = wdm_set.Nf, wdm_set.Nt
    kw = dict(convert_to_ra_dec=False, use_layer_groups=use_layer_groups)

    # 1. dense template coefficients
    hA = xp.zeros((nch, Nf, Nt))
    comp.fill_global_wdm(xp.asarray(A[None, :]), hA, convert_to_ra_dec=False)
    out = {"template": _to_np(hA)}

    # data = template(A) on the active band; identity invC across channels
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    hA_act = xp.ascontiguousarray(hA[:, ilo:ihi, wdm_set.active_slice_t])
    nfa, nta = hA_act.shape[1], hA_act.shape[2]
    invC = xp.zeros((nch, nch, nfa, nta))
    for c in range(nch):
        invC[c, c] = 1.0
    holder = _FullGridWDMHolder(xp, hA_act, invC)

    # 2. get_ll over the candidate batch
    comp.get_ll_wdm(xp.asarray(cands), holder, **kw)
    out["d_h"] = _to_np(comp.d_h_out).copy()
    out["h_h"] = _to_np(comp.h_h_out).copy()

    # 3. swap terms, candidate batch vs (tiled) B
    B_batch = np.tile(B, (cands.shape[0], 1))
    swap = comp.get_swap_ll_wdm(
        xp.asarray(cands), xp.asarray(B_batch), holder, **kw
    )
    for name, val in zip(("swap_0", "swap_1", "d_h_a", "d_h_r", "aa", "rr", "ar"), swap):
        out[name] = _to_np(val).copy()
    return out


def main():
    gpu_backend = os.environ.get("GPU_BACKEND", "cuda12x")
    n_cand = int(os.environ.get("N_CAND", 8))
    seed = int(os.environ.get("SEED", 2026))
    rtol_ll = float(os.environ.get("RTOL_LL", 1e-10))
    rtol_template = float(os.environ.get("RTOL_TEMPLATE", 1e-12))

    # Tiny grid (matches gb_chunked_swap_ll_validate.py)
    dt = 10.0
    Nf, Nt = 256, 512
    t_start = int(0.5 * YRSID_SI / dt) * dt
    layer_df = 1.0 / (2.0 * Nf * dt)
    wdm_kwargs = dict(Nf=Nf, Nt=Nt, dt=dt, t0=t_start, min_freq=1e-4, max_freq=2e-2)

    # Two sources in one 5-layer band + a candidate cloud around A
    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    B = A.copy()
    B[1] += 0.6 * layer_df
    B[4:7] = [2.9, 1.1, 0.9]

    rng = np.random.default_rng(seed)
    cands = np.tile(A, (n_cand, 1))
    cands[:, 0] *= 1.0 + 0.2 * rng.standard_normal(n_cand)      # amplitude
    cands[:, 1] += 0.3 * layer_df * rng.standard_normal(n_cand)  # f0
    cands[:, 4] = rng.uniform(0.0, 2 * np.pi, n_cand)            # phi0
    cands[:, 5] = rng.uniform(0.1, np.pi - 0.1, n_cand)          # iota

    failures = []

    def compare(tag, name, cpu_val, gpu_val, rtol):
        cpu_val, gpu_val = np.asarray(cpu_val), np.asarray(gpu_val)
        scale = max(np.abs(cpu_val).max(), 1e-300)
        rel = np.abs(gpu_val - cpu_val).max() / scale
        ok = rel <= rtol
        print(f"  [{tag}] {name:10s} max_rel={rel:.3e} (rtol={rtol:.0e}) "
              f"[{'OK' if ok else 'FAIL'}]")
        if not ok:
            failures.append(f"{tag}/{name}")

    # Direct (no cache) and production-cache configurations
    for n_cp_sig, n_cp_orbit in (
        (int(os.environ.get("N_CP_SIG", 0)), int(os.environ.get("N_CP_ORBIT", 0))),
        (48, 32),
    ):
        comp_kwargs = dict(
            t_ref=t_start, Nt_sub=128, n_pad=16, N_sparse=256,
            N_cp_sig=n_cp_sig, N_cp_orbit=n_cp_orbit,
            tdi_config="2nd generation", d_d=0.0, tdi_type="XYZ",
        )
        for use_layer_groups in (True, False):
            tag = (f"cache={n_cp_sig}/{n_cp_orbit} "
                   f"{'grouped' if use_layer_groups else 'ungrouped'}")
            print(f"[{tag}] building cpu + {gpu_backend} comps ...")
            comp_cpu, wdm_cpu = build_comp("cpu", wdm_kwargs, comp_kwargs)
            comp_gpu, wdm_gpu = build_comp(gpu_backend, wdm_kwargs, comp_kwargs)

            res_cpu = evaluate(comp_cpu, wdm_cpu, A, B, cands, use_layer_groups)
            res_gpu = evaluate(comp_gpu, wdm_gpu, A, B, cands, use_layer_groups)

            compare(tag, "template", res_cpu["template"], res_gpu["template"],
                    rtol_template)
            for name in ("d_h", "h_h", "d_h_a", "d_h_r", "aa", "rr", "ar"):
                compare(tag, name, res_cpu[name], res_gpu[name], rtol_ll)

    if failures:
        print(f"\nFAILED: {len(failures)} comparison(s): {failures}")
        raise SystemExit(1)
    print(f"\nALL CPU-vs-{gpu_backend.upper()} CHUNKED-HET CHECKS PASSED")


if __name__ == "__main__":
    main()
