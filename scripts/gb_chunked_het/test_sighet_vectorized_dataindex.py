#!/usr/bin/env python
"""Test 1 (Phase 0 de-risk): vectorized per-num_bin sig-het reference + data_index.

Goal: prove the sig-het ``gb_signal_het_get_ll_in_kernel`` already supports
``num_data > 1`` reference sets selected per-candidate by a length-num_bin
``data_index`` array -- WITHOUT any kernel or class change. We build TWO
``GBSignalHetComputations`` instances (two distinct heterodyne references on the
SAME data/grid), STACK their cached reference arrays along axis 0
(``c0_sparse_all``/``A*``/``B*``/``params_ref_all`` -> leading dim 2), assign each
candidate a ``data_index in {0,1}``, and call the kernel ONCE. We then assert the
per-candidate ``d_h``/``h_h`` exactly match running each candidate through its own
single-reference instance (``comp.get_ll`` -> ``comp.last_d_h``/``last_h_h``).

This validates the vectorized-reference + data_index plumbing the global-fit
conditional-Gibbs scheme needs (one reference per conditioned sub-band source,
selected at evaluation by data_index), with the dense reference build still in
place (Test 2 swaps the c0 source to chunked-het).

Run:  GB_RANK=0 GB_NT=512 python test_sighet_vectorized_dataindex.py
"""
import os
import numpy as np

import gbgpu  # noqa: F401  (registers backends)
from gb_mojito_mcmc_three_ways import build_shared, REF
from gbsignalhetcomputations import GBSignalHetComputations


def build_ref(s, ref_params):
    """One single-reference GBSignalHetComputations on the shared data/grid."""
    return GBSignalHetComputations(
        s["data_td"], ref_params, Nf=s["Nf"], Nt=s["Nt"], dt=s["dt"],
        t0=s["data_t0"], t_ref=REF,
        orbits=s["orbits"], tdi_config="2nd generation",
        min_freq=s["lo_f"], max_freq=s["hi_f"], edge_cut=s["EC"],
        m_active_half_width=s["M_HALF"], nt_layer=64, tukey_alpha=s["TUK"],
        force_backend=s["backend"])


def kernel_call(comp, c0, A0, A1, B0, B1, B0nc, B1nc, params_ref, x, data_index, num_data):
    """Mirror GBSignalHetComputations.get_ll but with externally supplied (stacked)
    reference arrays + num_data + data_index. Returns (d_h, h_h)."""
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x, float)))
    N = x.shape[0]
    d_h = np.zeros(N); h_h = np.zeros(N)
    g = comp._g
    comp.cpp.gb_signal_het_get_ll_in_kernel(
        comp.tdi_wrap, d_h, h_h, c0, A0, A1, B0, B1, B0nc, B1nc,
        comp.window_full, comp.n_sparse_local, x, params_ref,
        np.asarray(data_index, dtype=np.int32),
        N, int(num_data), 9, 1, 2,
        g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
        g["nt_layer"], g["N_sparse_t"], g["stride"],
        g["ind_min_t"], g["ind_min_f"], g["m_half"],
        g["layer_df"], g["dt"], g["Tobs"], g["t0"],
        3, 0, g["n_sparse_fd"], g["tukey_alpha"], g["max_r"], 1)
    return d_h, h_h


def main():
    s = build_shared()
    layer_df = s["layer_df"]; p = np.asarray(s["p_inj"], float)
    rng = np.random.default_rng(0)

    # --- two distinct references on the SAME data/grid -----------------------
    pA = p.copy()
    pB = p.copy(); pB[1] = p[1] + 0.4 * layer_df          # ref B: f0 + 0.4 layer
    print(f"[refs] f0_A={pA[1]*1e3:.6f} mHz  f0_B={pB[1]*1e3:.6f} mHz "
          f"(Delta={ (pB[1]-pA[1])/layer_df:.3f} layers)", flush=True)
    compA = build_ref(s, pA)
    compB = build_ref(s, pB)

    # sanity: geometry + window identical across the two references (kernel
    # takes ONE window/n_sparse/geometry block; the references differ only in
    # c0/bin-fold/params_ref).
    assert compA._g == compB._g, "geometry mismatch between references"
    assert np.array_equal(compA.n_sparse_local, compB.n_sparse_local)
    assert np.allclose(compA.window_full, compB.window_full)
    assert abs(compA.d_d - compB.d_d) < 1e-6 * abs(compA.d_d), "d_d differs (shared data)"
    print(f"[refs] shapes c0={compA.c0_sparse_all.shape} A0={compA.A0_all.shape} "
          f"B0={compA.B0_all.shape} params_ref={compA.params_ref_all.shape}", flush=True)

    # --- stacked (num_data=2) reference arrays -------------------------------
    cat = lambda a, b: np.ascontiguousarray(np.concatenate([a, b], axis=0))
    c0 = cat(compA.c0_sparse_all, compB.c0_sparse_all)
    A0 = cat(compA.A0_all, compB.A0_all); A1 = cat(compA.A1_all, compB.A1_all)
    B0 = cat(compA.B0_all, compB.B0_all); B1 = cat(compA.B1_all, compB.B1_all)
    B0nc = cat(compA.B0nc_all, compB.B0nc_all); B1nc = cat(compA.B1nc_all, compB.B1nc_all)
    params_ref = cat(compA.params_ref_all, compB.params_ref_all)
    print(f"[stack] c0={c0.shape} params_ref={params_ref.shape}  num_data=2", flush=True)

    # --- candidate batch: K near ref A (data_index 0) + K near ref B (idx 1) --
    K = int(os.environ.get("K", "6"))
    def jitter(p0):
        out = np.tile(p0, (K, 1)).astype(float)
        out[:, 1] += rng.uniform(-0.1, 0.1, K) * layer_df     # f0 within band
        out[:, 4] += rng.uniform(-0.3, 0.3, K)                # phi0
        out[:, 5] += rng.uniform(-0.1, 0.1, K)                # inc
        return out
    candsA = jitter(pA); candsB = jitter(pB)
    cands = np.concatenate([candsA, candsB], axis=0)
    data_index = np.concatenate([np.zeros(K, np.int32), np.ones(K, np.int32)])

    # --- batched (one kernel call, num_data=2, data_index) -------------------
    dh_bat, hh_bat = kernel_call(compA, c0, A0, A1, B0, B1, B0nc, B1nc,
                                 params_ref, cands, data_index, num_data=2)

    # --- baseline: each candidate through its OWN single-reference instance --
    dh_base = np.zeros(2 * K); hh_base = np.zeros(2 * K)
    for comp, idx in ((compA, np.where(data_index == 0)[0]),
                      (compB, np.where(data_index == 1)[0])):
        comp.get_ll(cands[idx])                         # populates last_d_h/last_h_h
        dh_base[idx] = comp.last_d_h; hh_base[idx] = comp.last_h_h

    # --- compare -------------------------------------------------------------
    def reldiff(a, b):
        scale = np.maximum(np.abs(a), np.abs(b)); scale[scale == 0] = 1.0
        return np.abs(a - b) / scale
    rd_dh = reldiff(dh_bat, dh_base); rd_hh = reldiff(hh_bat, hh_base)
    print("\n  i  di      d_h(batched)      d_h(baseline)   reldiff      h_h reldiff", flush=True)
    for i in range(2 * K):
        print(f" {i:2d}  {data_index[i]}  {dh_bat[i]:+15.7e}  {dh_base[i]:+15.7e}  "
              f"{rd_dh[i]:.2e}     {rd_hh[i]:.2e}", flush=True)
    tol = 1e-12
    ok = (rd_dh.max() < tol) and (rd_hh.max() < tol)
    print(f"\n[result] max reldiff  d_h={rd_dh.max():.2e}  h_h={rd_hh.max():.2e}  tol={tol:.0e}", flush=True)
    print(f"[result] {'PASS' if ok else 'FAIL'}: vectorized num_data=2 + data_index "
          f"{'==' if ok else '!='} per-reference baseline", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
