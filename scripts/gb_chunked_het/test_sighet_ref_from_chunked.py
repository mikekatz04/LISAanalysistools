#!/usr/bin/env python
"""Test 2 (Phase 0 de-risk): source the sig-het reference c0 from the polyphase /
chunked-het WDM path instead of the dense ``TDSignal.transform(wdm_complex)``.

The sig-het reference carrier ``c0`` is currently built by a DENSE complex-WDM
transform of the reference time-domain waveform (gbsignalhetcomputations.py:166-180,
the dominant ~2 s/source cost). The same complex sparse-WDM coefficients can be
produced by the polyphase per-active-m-layer iFFT (``GBSparseComplexWDMGen``) --
the SAME machinery the kernel already uses to build each candidate's ``c1`` -- which
is the fast/GPU-batchable path shared with chunked-het.

This test (route (a) in the plan) confirms the polyphase reproduces the dense c0 at
FP precision over the WHOLE active band, and that swapping it into the kernel leaves
``get_ll`` unchanged. The follow-on (route (a), step 2) is to source the rfft from the
C++ chunked-het FD-heterodyne front-end (``sparse_from_rfft``) rather than a dense TD
synth -- noted but not exercised here.

Run:  GB_RANK=0 GB_NT=512 python test_sighet_ref_from_chunked.py
"""
import os
import numpy as np

import gbgpu  # noqa: F401
from gb_mojito_mcmc_three_ways import build_shared, REF
from gbsignalhetcomputations import GBSignalHetComputations
from gb_signal_het_wdm_v2 import _compute_sparse_complex_wdm


def main():
    s = build_shared()
    layer_df = s["layer_df"]; p = np.asarray(s["p_inj"], float)
    rng = np.random.default_rng(1)

    comp = GBSignalHetComputations(
        s["data_td"], p, Nf=s["Nf"], Nt=s["Nt"], dt=s["dt"], t0=s["data_t0"], t_ref=REF,
        orbits=s["orbits"], tdi_config="2nd generation", min_freq=s["lo_f"], max_freq=s["hi_f"],
        edge_cut=s["EC"], m_active_half_width=s["M_HALF"], nt_layer=64,
        tukey_alpha=s["TUK"], force_backend=s["backend"])
    sg = comp._keep_alive["sparse_gen"]           # GBSparseComplexWDMGen (polyphase)

    # --- dense-derived c0 the comp currently uses (3, Nf_active, N_sparse_t) ---
    c0_dense = np.asarray(comp.c0_sparse_all[0])
    print(f"[c0] dense-derived c0_sparse shape={c0_dense.shape}  "
          f"|c0|max={np.abs(c0_dense).max():.3e}", flush=True)

    # --- polyphase c0 over the FULL active band (route a) --------------------
    # _compute_sparse_complex_wdm(fd_rfft, m_active_global, Nf, Nt, Nt_layer,
    #                             window_full, n_sparse_global, stride) * sqrt_dt
    # The dense c0 the comp uses is TDSignal.transform(..., window=tukey) -- the
    # Tukey is applied to the TD BEFORE the WDM transform (gbsignalhetcomputations
    # .py:167). The polyphase's sg.window_full is the per-layer WDM phitilde, NOT
    # the Tukey, so we must feed the rfft of the Tukey-windowed reference TD (else
    # the two disagree exactly in the taper region -> the time-edge pixels).
    tukey_win = np.asarray(comp._keep_alive["window"], float)
    td_ref = np.asarray(sg.real_td_callable(p))
    if td_ref.ndim == 1:
        td_ref = td_ref[None, :]
    fd_rfft = np.fft.rfft(td_ref * tukey_win[None, :], axis=-1)
    m_full = np.arange(sg.ind_min_f, sg.ind_min_f + sg.Nf_active_total)
    c0_poly = _compute_sparse_complex_wdm(
        fd_rfft, m_full, sg.Nf, sg.Nt, sg.Nt_layer,
        sg.window_full, sg.n_sparse_global, sg.stride) * sg.sqrt_dt
    c0_poly = np.ascontiguousarray(c0_poly)
    print(f"[c0] polyphase  c0_sparse shape={c0_poly.shape}  "
          f"|c0|max={np.abs(c0_poly).max():.3e}", flush=True)

    # --- per-pixel reldiff (polyphase identity) ------------------------------
    scale = max(float(np.abs(c0_dense).max()), 1e-300)
    pix = np.abs(c0_poly - c0_dense) / scale
    # The active-band edges can carry small Hermitian-wrap leakage; report both
    # the full-band and the interior (drop m_active_half_width layers each end).
    mh = s["M_HALF"]
    interior = pix[:, mh:pix.shape[1] - mh, :]
    print(f"\n[identity] c0 polyphase-vs-dense reldiff  full-band  max={pix.max():.2e}  "
          f"median={np.median(pix):.2e}", flush=True)
    print(f"[identity] c0 polyphase-vs-dense reldiff  interior   max={interior.max():.2e}  "
          f"median={np.median(interior):.2e}", flush=True)

    # --- get_ll parity: swap c0_sparse_all -> polyphase, compare -------------
    # candidates jittered within the reference floor-band (so m_active(cand) ==
    # m_active(ref); all needed c0 layers are present in the full-band polyphase).
    K = int(os.environ.get("K", "8"))
    cands = np.tile(p, (K, 1)).astype(float)
    cands[:, 1] += rng.uniform(-0.15, 0.15, K) * layer_df
    cands[:, 4] += rng.uniform(-0.4, 0.4, K)
    cands[:, 5] += rng.uniform(-0.1, 0.1, K)

    ll_dense = np.asarray(comp.get_ll(cands)).copy()
    dh_dense = comp.last_d_h.copy(); hh_dense = comp.last_h_h.copy()

    saved = comp.c0_sparse_all
    comp.c0_sparse_all = np.ascontiguousarray(c0_poly[None]).copy()
    ll_poly = np.asarray(comp.get_ll(cands)).copy()
    dh_poly = comp.last_d_h.copy(); hh_poly = comp.last_h_h.copy()
    comp.c0_sparse_all = saved

    def reld(a, b):
        sc = np.maximum(np.abs(a), np.abs(b)); sc[sc == 0] = 1.0
        return np.abs(a - b) / sc
    rd_dh = reld(dh_poly, dh_dense); rd_hh = reld(hh_poly, hh_dense)
    print(f"\n[get_ll] polyphase-c0 vs dense-c0  d_h reldiff max={rd_dh.max():.2e}  "
          f"h_h reldiff max={rd_hh.max():.2e}", flush=True)
    print(f"[get_ll] logL  dense[:3]={ll_dense[:3]}  poly[:3]={ll_poly[:3]}", flush=True)

    # --- verdict -------------------------------------------------------------
    id_tol = 1e-9          # polyphase identity is mathematically exact (->1e-12);
    ll_tol = 1e-7          # CLAUDE.md mm5 band tolerance for the consumed reference
    id_ok = interior.max() < id_tol
    ll_ok = (rd_dh.max() < ll_tol) and (rd_hh.max() < ll_tol)
    print(f"\n[result] c0 identity  {'PASS' if id_ok else 'FAIL'} "
          f"(interior max {interior.max():.2e} < {id_tol:.0e})", flush=True)
    print(f"[result] get_ll parity {'PASS' if ll_ok else 'FAIL'} "
          f"(max {max(rd_dh.max(), rd_hh.max()):.2e} < {ll_tol:.0e})", flush=True)
    print(f"[result] {'PASS' if (id_ok and ll_ok) else 'FAIL'}: chunked/polyphase c0 "
          f"reproduces the dense sig-het reference", flush=True)
    return 0 if (id_ok and ll_ok) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
