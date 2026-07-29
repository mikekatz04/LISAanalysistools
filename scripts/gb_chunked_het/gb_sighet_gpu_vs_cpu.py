#!/usr/bin/env python
"""GPU-vs-CPU parity check for the sig-het in-model GB likelihood.

Same scaffolding as ``gb_chunked_het_gpu_vs_cpu.py`` (tiny WDM grid, one
5-layer band), but exercising the SIGNAL-HET band-engine path that the GB
special move's in-model repeat proposals use:

  1. ``for_band_engine``      -- comp built around a chunked delegate on each
                                 backend (backend INHERITED from the delegate);
  2. ``setup_in_model``       -- the per-source reference cache
                                 (backend ``gb_signal_het_make_reference`` +
                                 lisatools ``bin_fold_real``): c0_sparse and
                                 the A0/A1/B0/B1/B0nc/B1nc stash compared
                                 element-wise;
  3. ``get_ll``               -- the fused ``gb_signal_het_get_ll_in_kernel``
                                 kernel over a candidate batch: d_h / h_h / ll;
  4. ``get_ll(phase_maximize=True)`` -- the two-quadrature path (|D|,
                                 phase_angle);
  5. mid-block PATCH          -- second ``setup_in_model`` on the same slot
                                 (the drift-refresh path) then re-score.

CPU and GPU share ``gb_signal_het_consume_one_source`` / the make_reference
fold conventions, so agreement is at floating-point order-of-operations
level; default tolerances rtol=1e-10 (ll), 1e-12 (coefficient stash,
relative to each array's max).

Run (cluster):
    GPU_BACKEND=cuda12x python gb_sighet_gpu_vs_cpu.py

Harness self-test (CPU vs CPU; diffs must be ~0):
    GPU_BACKEND=cpu python gb_sighet_gpu_vs_cpu.py

Env knobs: GPU_BACKEND (default cuda12x), N_CAND (default 8), SEED,
NT_LAYER (default 64), N_SPARSE_FD (default 1024), RTOL_LL, RTOL_COEF,
N_REF (default 1) -- number of heterodyne references / buffer slots. N_REF>1
exercises the BATCHED setup_in_model (one bin_fold_real call over the slot
axis) and the partial mid-block refresh; GB_SIGHET_FOLD_MAX_BYTES forces the
batched fold to split into multiple chunks.
"""
from __future__ import annotations

import os

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations


class _FullGridWDMHolder:
    """Duck-type for the buffer: active-band data + XYZ cross invC slabs."""

    def __init__(self, xp, data_act, invC_cross_act):
        self.linear_data_arr = [xp.ascontiguousarray(xp.asarray(data_act)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(xp.asarray(invC_cross_act)).ravel()]

    def __len__(self):
        return 1


def _to_np(arr):
    return np.asarray(arr.get() if hasattr(arr, "get") else arr)


def build_sighet(backend, wdm_kwargs, comp_kwargs, nt_layer, n_sparse_fd):
    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(force_backend=backend, **wdm_kwargs)
    chunked = GBWDMComputations(
        wdm_set, orbits=orbits, force_backend=backend, **comp_kwargs
    )
    comp = GBSignalHetComputations.for_band_engine(
        chunked, nt_layer=nt_layer, n_sparse_fd=n_sparse_fd
    )
    return comp, chunked, wdm_set


def evaluate(comp, chunked, wdm_set, A_all, cands, cand_slots, patch_slots):
    """Build the in-model scenario on this comp's backend; return numpy dict.

    ``A_all`` is ``(n_ref, 9)``: one heterodyne reference per buffer slot, laid
    out on the slot axis exactly as the band buffer does. ``cand_slots`` maps
    each candidate to the slot (hence reference) it scores against, and
    ``patch_slots`` is the SUBSET refreshed mid-block. With ``n_ref == 1`` this
    reduces to the original single-reference scenario.
    """
    xp = comp.xp
    nch = 3
    Nf, Nt = wdm_set.Nf, wdm_set.Nt
    n_ref = A_all.shape[0]
    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1

    # One residual slab PER SLOT (each reference sees its own data), stacked on
    # the slot axis; identity cross invC per slot.
    slabs = []
    for i in range(n_ref):
        h = xp.zeros((nch, Nf, Nt))
        chunked.fill_global_wdm(xp.asarray(A_all[i][None, :]), h,
                                convert_to_ra_dec=False)
        slabs.append(xp.ascontiguousarray(h[:, ilo:ihi, wdm_set.active_slice_t]))
    data_act = xp.stack(slabs)                       # (n_ref, nch, nfa, nta)
    nfa, nta = data_act.shape[2], data_act.shape[3]
    invC = xp.zeros((n_ref, nch, nch, nfa, nta))
    for c in range(nch):
        invC[:, c, c] = 1.0
    holder = _FullGridWDMHolder(xp, data_act, invC)

    out = {}
    slots = np.arange(n_ref)

    # 1) reference cache (n_ref references in ONE batched setup_in_model)
    assert comp.setup_in_model(holder, xp.asarray(A_all), slots) is True
    for name in ("c0_sparse_all", "A0_all", "A1_all", "B0_all", "B1_all",
                 "B0nc_all", "B1nc_all"):
        out[name] = _to_np(getattr(comp, name)).copy()

    # 2) fused get_ll over the candidate batch (candidates spread across slots)
    ll = comp.get_ll(xp.asarray(cands), data_index=cand_slots)
    out["ll"] = _to_np(ll).copy()
    out["d_h"] = _to_np(comp.last_d_h).copy()
    out["h_h"] = _to_np(comp.last_h_h).copy()

    # 3) phase-maximized path
    ll_pm = comp.get_ll(xp.asarray(cands), data_index=cand_slots,
                        phase_maximize=True)
    out["ll_pm"] = _to_np(ll_pm).copy()
    out["d_h_pm"] = _to_np(comp.last_d_h).copy()
    out["phase_angle"] = _to_np(comp.phase_angle).copy()

    # 4) mid-block patch (drift refresh) on a SUBSET of slots, then re-score.
    #    Exercises the ref_idx scatter back into the coefficient stash -- the
    #    partial-update path a real drift refresh takes.
    A_shift = A_all.copy()
    A_shift[patch_slots, 4] += 0.1
    assert comp.setup_in_model(holder, xp.asarray(A_shift[patch_slots]),
                               patch_slots) is True
    for name in ("c0_sparse_all", "A0_all", "B0nc_all"):
        out[f"{name}_patched"] = _to_np(getattr(comp, name)).copy()
    ll2 = comp.get_ll(xp.asarray(cands), data_index=cand_slots)
    out["ll_patched"] = _to_np(ll2).copy()

    comp.clear_in_model()
    return out


def main():
    gpu_backend = os.environ.get("GPU_BACKEND", "cuda12x")
    n_cand = int(os.environ.get("N_CAND", 8))
    seed = int(os.environ.get("SEED", 2026))
    nt_layer = int(os.environ.get("NT_LAYER", 64))
    n_ref = int(os.environ.get("N_REF", 1))
    n_sparse_fd = int(os.environ.get("N_SPARSE_FD", 1024))
    rtol_ll = float(os.environ.get("RTOL_LL", 1e-10))
    rtol_coef = float(os.environ.get("RTOL_COEF", 1e-12))

    dt = 10.0
    Nf, Nt = 256, 512
    t_start = int(0.5 * YRSID_SI / dt) * dt
    layer_df = 1.0 / (2.0 * Nf * dt)
    # production-style time-edge crop (sig-het bin-fold needs it)
    edge = 40
    wdm_kwargs = dict(Nf=Nf, Nt=Nt, dt=dt, t0=t_start,
                      min_freq=1e-4, max_freq=2e-2,
                      min_time=edge * Nf * dt,
                      max_time=(Nt - edge) * Nf * dt)

    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])

    rng = np.random.default_rng(seed)

    # n_ref references, one per buffer slot: spread in f0 (well-separated
    # layers) and given distinct sky/phase so no two references are alike.
    A_all = np.tile(A, (n_ref, 1))
    A_all[:, 1] = f0_A + np.arange(n_ref) * 7.0 * layer_df
    A_all[:, 4] = (A[4] + 0.31 * np.arange(n_ref)) % (2 * np.pi)
    A_all[:, 7] = (A[7] + 0.17 * np.arange(n_ref)) % (2 * np.pi)
    A_all[:, 0] = A[0] * (1.0 + 0.05 * np.arange(n_ref))

    # Candidates round-robin over the slots, each perturbed around ITS OWN
    # reference -- so a slot -> reference mix-up shows up as a large ll error.
    cand_slots = np.arange(n_cand) % n_ref
    cands = A_all[cand_slots].copy()
    cands[:, 0] *= 1.0 + 0.1 * rng.standard_normal(n_cand)
    cands[:, 1] += 0.05 * layer_df * rng.standard_normal(n_cand)
    cands[:, 4] = rng.uniform(0.0, 2 * np.pi, n_cand)

    # Refresh a strict subset mid-block (every other slot) so the partial
    # scatter back into the stash is exercised, not a full rebuild.
    patch_slots = np.arange(0, n_ref, 2)

    comp_kwargs = dict(
        t_ref=t_start, Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0,
        tdi_config="2nd generation", d_d=0.0, tdi_type="XYZ",
    )

    failures = []

    def compare(name, cpu_val, gpu_val, rtol):
        cpu_val, gpu_val = np.asarray(cpu_val), np.asarray(gpu_val)
        scale = max(np.abs(cpu_val).max(), 1e-300)
        rel = np.abs(gpu_val - cpu_val).max() / scale
        ok = rel <= rtol
        print(f"  {name:14s} max_rel={rel:.3e} (rtol={rtol:.0e}) "
              f"[{'OK' if ok else 'FAIL'}]")
        if not ok:
            failures.append(name)

    print(f"[sighet] building cpu + {gpu_backend} engine comps "
          f"(nt_layer={nt_layer}, n_sparse_fd={n_sparse_fd}, "
          f"n_ref={n_ref}, n_cand={n_cand}) ...")
    comp_cpu, chk_cpu, wdm_cpu = build_sighet(
        "cpu", wdm_kwargs, comp_kwargs, nt_layer, n_sparse_fd)
    comp_gpu, chk_gpu, wdm_gpu = build_sighet(
        gpu_backend, wdm_kwargs, comp_kwargs, nt_layer, n_sparse_fd)
    print(f"[sighet] backends: cpu={comp_cpu.backend.name} "
          f"gpu={comp_gpu.backend.name}")

    res_cpu = evaluate(comp_cpu, chk_cpu, wdm_cpu, A_all, cands,
                       cand_slots, patch_slots)
    res_gpu = evaluate(comp_gpu, chk_gpu, wdm_gpu, A_all, cands,
                       cand_slots, patch_slots)

    for name in ("c0_sparse_all", "A0_all", "A1_all", "B0_all", "B1_all",
                 "B0nc_all", "B1nc_all"):
        compare(name, res_cpu[name], res_gpu[name], rtol_coef)
    for name in ("c0_sparse_all_patched", "A0_all_patched",
                 "B0nc_all_patched"):
        compare(name, res_cpu[name], res_gpu[name], rtol_coef)
    for name in ("d_h", "h_h", "ll", "ll_pm", "d_h_pm", "phase_angle",
                 "ll_patched"):
        compare(name, res_cpu[name], res_gpu[name], rtol_ll)

    if failures:
        print(f"\nFAILED: {len(failures)} comparison(s): {failures}")
        raise SystemExit(1)
    print(f"\nALL CPU-vs-{gpu_backend.upper()} SIG-HET CHECKS PASSED")


if __name__ == "__main__":
    main()
