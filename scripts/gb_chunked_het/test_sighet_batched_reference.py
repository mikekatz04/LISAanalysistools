#!/usr/bin/env python
"""Test 3 (Phase 0 de-risk): the BATCHED reference generator.

Validates GBSignalHetReferenceSet -- which builds num_data references' c0 + bin-fold
in sub-batches with a vectorized bin-fold and chunked-het c0 -- against num_data
independent single-reference GBSignalHetComputations instances:

  1. the stacked reference arrays (c0_sparse / A0 / B0 / ...) match the per-reference
     single builds exactly (the vectorized bin-fold == python_bin_fold_real);
  2. get_ll with a per-candidate data_index == each candidate through its own
     single-reference instance (the data_index eval, on a genuinely-vectorized build).

Run:  GB_RANK=0 GB_NT=512 NUM_REF=3 python test_sighet_batched_reference.py
"""
import os
import numpy as np

import gbgpu  # noqa: F401
from gb_mojito_mcmc_three_ways import build_shared, REF
from gbsignalhetcomputations import GBSignalHetComputations
from gb_sighet_batched_reference import GBSignalHetReferenceSet


def main():
    s = build_shared()
    ldf = s["layer_df"]; p = np.asarray(s["p_inj"], float)
    impl = os.environ.get("SIGHET_REF_IMPL", "chunked")
    num_ref = int(os.environ.get("NUM_REF", "3"))
    bsz = int(os.environ.get("REF_BATCH_SIZE", "2"))     # < num_ref to exercise sub-batching

    # num_ref distinct references spread across the band
    rng = np.random.default_rng(7)
    ref_params_all = np.tile(p, (num_ref, 1)).astype(float)
    ref_params_all[:, 1] += np.linspace(-0.6, 0.6, num_ref) * ldf       # f0 offsets

    comp_kw = dict(Nf=s["Nf"], Nt=s["Nt"], dt=s["dt"], t0=s["data_t0"], t_ref=REF,
                   orbits=s["orbits"], tdi_config="2nd generation", min_freq=s["lo_f"],
                   max_freq=s["hi_f"], edge_cut=s["EC"], m_active_half_width=s["M_HALF"],
                   nt_layer=64, tukey_alpha=s["TUK"], force_backend=s["backend"])

    print(f"[setup] num_ref={num_ref} ref_batch_size={bsz} reference_impl={impl}", flush=True)
    rset = GBSignalHetReferenceSet(s["data_td"], ref_params_all, ref_batch_size=bsz,
                                   reference_impl=impl, **comp_kw)
    singles = [GBSignalHetComputations(s["data_td"], ref_params_all[i],
                                       reference_impl=impl, **comp_kw)
               for i in range(num_ref)]

    # (1) stacked reference arrays == per-reference single builds
    def amax(a, b):
        sc = np.maximum(np.abs(a), np.abs(b)); sc[sc == 0] = 1.0
        return float((np.abs(a - b) / sc).max())
    print("\n[arrays] vectorized-build vs single-build per-reference max reldiff:", flush=True)
    arr_ok = True
    for i, c in enumerate(singles):
        d = {
            "c0": amax(rset.c0_sparse_all[i], c.c0_sparse_all[0]),
            "A0": amax(rset.A0_all[i], c.A0_all[0]), "A1": amax(rset.A1_all[i], c.A1_all[0]),
            "B0": amax(rset.B0_all[i], c.B0_all[0]), "B1": amax(rset.B1_all[i], c.B1_all[0]),
            "B0nc": amax(rset.B0nc_all[i], c.B0nc_all[0]),
        }
        worst = max(d.values()); arr_ok &= worst < 1e-12
        print(f"   ref {i}: " + "  ".join(f"{k}={v:.1e}" for k, v in d.items()), flush=True)

    # (2) get_ll with data_index == per-reference single get_ll
    K = int(os.environ.get("K", "5"))
    cands_list, di_list = [], []
    for i in range(num_ref):
        c = np.tile(ref_params_all[i], (K, 1)).astype(float)
        c[:, 1] += rng.uniform(-0.1, 0.1, K) * ldf
        c[:, 4] += rng.uniform(-0.3, 0.3, K)
        cands_list.append(c); di_list.append(np.full(K, i, np.int32))
    cands = np.concatenate(cands_list, 0); data_index = np.concatenate(di_list)

    rset.get_ll(cands, data_index)
    dh_bat, hh_bat = rset.last_d_h, rset.last_h_h
    dh_base = np.zeros(num_ref * K); hh_base = np.zeros(num_ref * K)
    for i, c in enumerate(singles):
        idx = np.where(data_index == i)[0]
        c.get_ll(cands[idx]); dh_base[idx] = c.last_d_h; hh_base[idx] = c.last_h_h
    rd_dh = amax(dh_bat, dh_base); rd_hh = amax(hh_bat, hh_base)
    ll_ok = (rd_dh < 1e-12) and (rd_hh < 1e-12)
    print(f"\n[get_ll] batched+data_index vs per-reference: d_h reldiff={rd_dh:.2e}  "
          f"h_h reldiff={rd_hh:.2e}", flush=True)

    ok = arr_ok and ll_ok
    print(f"\n[result] arrays {'PASS' if arr_ok else 'FAIL'}  get_ll {'PASS' if ll_ok else 'FAIL'}", flush=True)
    print(f"[result] {'PASS' if ok else 'FAIL'}: batched/sub-batched reference generation "
          f"({impl}) == per-reference build + data_index eval", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
