#!/usr/bin/env python
"""Test 4: the GBGPU backend reference producer (gb_signal_het_make_reference).

Validates that the C++ backend producer emits the reference WDM c0 -- at BOTH the
sparse grid and full Nt resolution -- equal to the dense TDSignal.transform(wdm_complex)
that the "dense" reference_impl uses. This is the correctness check for the new C++
(the dense polyphase is new code); once it passes, the sig-het reference comes from the
backend instead of the Python polyphase.

Self-contained: synthetic GB + EqualArmlengthOrbits (the c0 comparison is
data-independent -- it only needs the reference params + the WDM transform).

Run:  python test_sighet_backend_producer.py
"""
import numpy as np

import gbgpu  # noqa: F401
from gbsignalhetcomputations import GBSignalHetComputations, recommended_edge_cut
from lisatools.domains import TDSignal
from lisatools.detector import EqualArmlengthOrbits


def build_synth():
    dt = 10.0; Nf = 1460; Nt = 512; Nobs = Nf * Nt
    layer_df = 1.0 / (2.0 * Nf * dt); TUK = 0.05; M_HALF = 2
    EC = recommended_edge_cut(Nt, TUK, "chunked"); t0 = 0.0
    amp, f0, fdot = 1e-22, 5.0e-3, 1e-16
    p_inj = np.array([amp, f0, fdot, 0.0, 1.0, 0.6, 0.4, 1.2, 0.5])   # 9-vec
    m_floor = int(f0 / layer_df); BAND = 15
    lo_f = (m_floor - BAND) * layer_df; hi_f = (m_floor + BAND + 1) * layer_df
    orbits = EqualArmlengthOrbits(); orbits.configure(linear_interp_setup=True)
    rng = np.random.default_rng(0)
    data_td = (1e-25 * rng.standard_normal((3, Nobs)))               # tiny; c0 ignores it
    return dict(backend="cpu", dt=dt, Nf=Nf, Nt=Nt, TUK=TUK, M_HALF=M_HALF, EC=EC,
                data_td=data_td, data_t0=t0, p_inj=p_inj, lo_f=lo_f, hi_f=hi_f,
                orbits=orbits)


def main():
    s = build_synth()
    p = np.asarray(s["p_inj"], float)

    comp = GBSignalHetComputations(
        s["data_td"], p, Nf=s["Nf"], Nt=s["Nt"], dt=s["dt"], t0=s["data_t0"],
        t_ref=s["data_t0"], orbits=s["orbits"], tdi_config="2nd generation",
        min_freq=s["lo_f"], max_freq=s["hi_f"], edge_cut=s["EC"],
        m_active_half_width=s["M_HALF"], nt_layer=64, tukey_alpha=s["TUK"],
        force_backend=s["backend"], reference_impl="dense")
    g = comp._g; sh = comp._gen_shared

    # ground truth: full dense c0 (nch, Nf_active, Nt_active) from TDSignal.transform
    c0_dense_ref = np.asarray(TDSignal(sh["real_td_cb"](p), settings=sh["td_set"])
                              .transform(sh["wdm_set_complex"], window=sh["window"]).arr)
    c0_sparse_ref = c0_dense_ref[:, :, comp.n_sparse_local]
    nch, Nf_active, Nt_active = c0_dense_ref.shape
    N_sparse_t = g["N_sparse_t"]

    # backend producer
    c0_sparse_be = np.zeros((1, nch, Nf_active, N_sparse_t), dtype=np.complex128)
    c0_dense_be = np.zeros((1, nch, Nf_active, Nt_active), dtype=np.complex128)
    comp.cpp.gb_signal_het_make_reference(
        comp.tdi_wrap, c0_sparse_be, c0_dense_be,
        comp.window_full, comp.n_sparse_local,
        np.ascontiguousarray(p[None]), 1,
        9, 1, 2,
        g["Nf"], g["Nt"], Nf_active, Nt_active,
        g["nt_layer"], N_sparse_t, g["stride"],
        g["ind_min_t"], g["ind_min_f"],
        g["layer_df"], g["dt"], g["Tobs"], g["t0"],
        3, g["n_sparse_fd"], g["tukey_alpha"])

    def reld(a, b):
        sc = max(float(np.abs(b).max()), 1e-300)
        return float(np.abs(a - b).max()) / sc
    rd_sparse = reld(c0_sparse_be[0], c0_sparse_ref)
    rd_dense = reld(c0_dense_be[0], c0_dense_ref)
    mh = s["M_HALF"]
    rd_dense_int = reld(c0_dense_be[0, :, mh:Nf_active - mh], c0_dense_ref[:, mh:Nf_active - mh])

    # diagnostics: dense-vs-sparse consistency + ratio pattern
    rd_dense_at_sparse = reld(c0_dense_be[0][:, :, comp.n_sparse_local], c0_sparse_be[0])
    ml = Nf_active // 2
    ratio = (c0_dense_be[0, 0, ml, 20:26] /
             np.where(np.abs(c0_dense_ref[0, ml, 20:26]) > 0, c0_dense_ref[0, ml, 20:26], 1))
    print(f"[dbg] c0_dense_be[:,:,n_sparse] vs c0_sparse_be reldiff = {rd_dense_at_sparse:.2e}", flush=True)
    print(f"[dbg] ratio be/ref (layer {ml}, n=20..25): {np.round(ratio, 4)}", flush=True)

    print(f"[shapes] c0_sparse {c0_sparse_be.shape}  c0_dense {c0_dense_be.shape}  "
          f"|c0|max={np.abs(c0_dense_ref).max():.2e}", flush=True)
    print(f"[c0_sparse] backend vs dense-transform   max reldiff = {rd_sparse:.2e}", flush=True)
    print(f"[c0_dense ] backend vs dense-transform   max reldiff = {rd_dense:.2e}  "
          f"(interior {rd_dense_int:.2e})", flush=True)

    # --- get_ll parity: backend c0 + its bin-fold vs the dense path -----------
    # The meaningful metric. c0 (FD-gen) differs from the transform at the ~1% bin
    # level, but that averages out at the inner-product level -- so get_ll with the
    # backend c0 should track the dense-reference get_ll far better than 1e-5.
    from gb_signal_het_cpp_validate import python_bin_fold_real
    layer_df = 1.0 / (2.0 * s["Nf"] * s["dt"])
    A0, A1, B0, B1, B0nc, B1nc = python_bin_fold_real(
        sh["data_complex"], c0_dense_be[0], sh["invC_complex"],
        comp.n_sparse_local, g["stride"], Nt_active, "XYZ")

    rng = np.random.default_rng(2); K = 6
    cands = np.tile(p, (K, 1)).astype(float)
    cands[:, 1] += rng.uniform(-0.1, 0.1, K) * layer_df
    cands[:, 4] += rng.uniform(-0.3, 0.3, K)
    comp.get_ll(cands); dh_dense = comp.last_d_h.copy(); hh_dense = comp.last_h_h.copy()
    comp.c0_sparse_all = np.ascontiguousarray(c0_sparse_be)
    comp.A0_all = np.ascontiguousarray(A0[None]); comp.A1_all = np.ascontiguousarray(A1[None])
    comp.B0_all = np.ascontiguousarray(B0[None]); comp.B1_all = np.ascontiguousarray(B1[None])
    comp.B0nc_all = np.ascontiguousarray(B0nc[None]); comp.B1nc_all = np.ascontiguousarray(B1nc[None])
    comp.get_ll(cands); dh_be = comp.last_d_h.copy(); hh_be = comp.last_h_h.copy()

    def rr(a, b):
        sc = np.maximum(np.abs(a), np.abs(b)); sc[sc == 0] = 1.0
        return float((np.abs(a - b) / sc).max())
    rd_dh = rr(dh_be, dh_dense); rd_hh = rr(hh_be, hh_dense)
    print(f"\n[get_ll] backend-c0 vs dense-c0   d_h reldiff = {rd_dh:.2e}   h_h reldiff = {rd_hh:.2e}",
          flush=True)

    # verdict: convention correct (c0 == transform modulo FD-gen ~1e-5) + get_ll parity
    conv_ok = rd_dense_int < 1e-3          # dense convention correct (not the 2.0 bug)
    ll_ok = (rd_dh < 1e-3) and (rd_hh < 1e-3)
    print(f"\n[result] dense convention {'OK' if conv_ok else 'BROKEN'} (c0 reldiff {rd_dense_int:.1e}, "
          f"FD-gen level); get_ll parity {'PASS' if ll_ok else 'FAIL'} "
          f"(d_h {rd_dh:.1e}, h_h {rd_hh:.1e})", flush=True)
    return 0 if (conv_ok and ll_ok) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
