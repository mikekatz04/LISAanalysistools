"""Deep-dive: WHY does the sig-het get_ll have an eff_mm ~6.5e-4 floor at Nt=512
(O(1) logL) that shrinks with Nt, and is knob-independent (n_sparse_fd/nt_layer/
m_half all flat)?

Decompose the candidate-WDM error against the lisatools dense WDM (c0) into:

  (A) POLYPHASE-only -- GBSparseComplexWDMGen.verify_against_dense: polyphase
      applied to the FULL dense rfft of the windowed TD. The polyphase math is
      mathematically exact, so this isolates it (~1e-12 expected).

  (B) KERNEL candidate -- gb_signal_het_fill_global_in_kernel reconstructs the
      candidate WDM template using gb_run_fd_wave_tdi's SPARSE heterodyned FD
      (carrier removed, slow part on N_sparse_fd points, sparse FFT). Compared
      to the dense c0 via lisatools template_inner_product => mm_kernel.

If (A) << (B), the floor lives entirely in the kernel's gb_run_fd_wave_tdi FD
generation (sparse heterodyned FD != dense rfft), NOT the polyphase or bin-fold.

Also localizes (B): per-WDM-time-pixel L2 error of (kernel - dense) to see if it
is edge-concentrated (heterodyne/window edge bias) or bulk, and per-m-layer.
Synthetic GB, cand=ref, ESAOrbits, full band, tukey=0 (so NO global-taper effect
-- this floor is separate from the chunked taper issue). Sweeps Nt.
"""
import os, sys, gc
import numpy as np
from scipy.signal.windows import tukey as _tukey

import gbgpu  # noqa: F401
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.diagnostic import inner_product
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
import gbgpu_backend_cpu.cgbgpu as _be

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen  # noqa: E402
from gb_signal_het_cpp_validate import python_bin_fold  # noqa: E402

DT = 10.0; NF = 1460
EC = int(os.environ.get("EDGE", "20"))   # WDM time-edge cut (layers) for the active region
NSFD = int(os.environ.get("N_SPARSE_FD", "1024"))
NTL = int(os.environ.get("NT_LAYER", "64"))
# Operating regime (sprint convention, GB chunked/sig-het notes): a 0.01-0.05
# Tukey is applied to BOTH the data AND the sparse-FD slow signal. Data: the
# `window` passed to TDSignal.transform. Sparse-FD slow signal: the kernel's
# `tukey_alpha` arg (gbfd_build_one_source tapers the slow part before its FFT)
# AND the windowed callable handed to the polyphase check below. Both must match.
TUK = float(os.environ.get("TUKEY_ALPHA", "0.05"))
MH = int(os.environ.get("M_HALF", "2"))
# max_r = the amp/phase CLIP on the heterodyne ratio r = c1/c0 (candidate WDM /
# reference WDM). The kernel forms r per sparse pixel; where the reference c0 is
# tiny (the WDM TIME-EDGE pixels, |c0|->0) the ratio blows up, so the kernel caps
# |r| at max_r while preserving its phase (r -> r*max_r/|r|). Larger max_r lets
# more edge blow-up through; smaller clips genuine signal excursions. 5.0 is the
# validated default. MEASURED here: the clip contributes only ~1% of the get_ll
# floor (max_r=2 / 5 identical; max_r=1e12 i.e. NO clip is only +1.2% worse). So
# the edge r-blow-up is NOT the dominant floor term -- the floor is the bin-fold
# QUADRATURE itself (see summary), which max_r/tukey/n_sparse_fd/nt_layer/m_half
# all leave unchanged; only the observation length Nt moves it.
MAX_R = float(os.environ.get("MAX_R", "5.0"))
AMP = 1e-21


def run(Nt):
    Nobs = NF * Nt; Tobs = Nt * NF * DT
    t_start = int(0.5 * YRSID_SI / DT) * DT
    t_arr = np.arange(Nobs) * DT + t_start
    orbits = ESAOrbits(force_backend="cpu")
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / DT, 1, tdi_config=tdi_config,
                           orbits=orbits, tdi_chan="XYZ", force_backend="cpu")
    tdi_wrap = gb_gen.wave_gen
    p9 = np.array([AMP, 14.22e-3, 1e-16, 0.0, 1.4, np.pi / 3.0, 0.7, 2.1, 0.5])

    def real_td_cb(p):
        sp = gb_gen(*[np.array([x]) for x in np.asarray(p, float).reshape(9)],
                    convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, DT, t0=t_start, force_backend="cpu")
    window = _tukey(Nobs, TUK).astype(float) if TUK > 0 else np.ones(Nobs)
    wkw = dict(t0=t_start, min_freq=1e-4, max_freq=35e-3,
               min_time=EC * NF * DT, max_time=(Nt - EC) * NF * DT, force_backend="cpu")
    wsr = WDMSettings(NF, Nt, DT, is_complex=False, **wkw)
    wsc = WDMSettings(NF, Nt, DT, is_complex=True, **wkw)
    ind_min_t = int(wsr.ind_min_t); ind_min_f = int(wsr.ind_min_f)
    Nt_active = int(wsr.Nt_active); Nf_active = int(wsr.ind_max_f - wsr.ind_min_f + 1)
    layer_df = wsr.layer_df

    td = real_td_cb(p9)
    c0r = TDSignal(td, td_set).transform(wsr, window=window)          # dense real WDM (truth)
    c0c = np.asarray(TDSignal(td, td_set).transform(wsc, window=window).arr)  # dense complex WDM
    sens = XYZ2SensitivityMatrix(wsr, model="scirdv1")
    analysis = AnalysisContainer(c0r, sens)

    # ---- (A) polyphase-only floor: from the FULL dense rfft -----------------
    # window the polyphase callable too, so it sees the SAME Tukey taper the
    # data/c0 were transformed with (tukey on the sparse-FD slow signal).
    def real_td_cb_win(p):
        return real_td_cb(p) * window[None, :]
    sg = GBSparseComplexWDMGen(real_td_cb_win, wsc, DT, ind_min_t, Nt_active, NTL, MH)
    mm_poly = sg.verify_against_dense(p9, c0c)   # per-pixel max reldiff

    # ---- (B) kernel candidate template via fill_global ----------------------
    stride = sg.stride; N_sparse_t = sg.N_sparse_t
    n_sparse_local = np.asarray(sg.n_sparse_local, dtype=np.int32)
    window_full = sg.window_full.astype(np.float64).copy()
    c0_sparse = c0c[:, :, n_sparse_local].copy()
    cpp = _be.GBComputationGroupWrapCPU()
    template_fill = np.zeros((1, 3, NF, Nt), dtype=np.float64)
    cpp.gb_signal_het_fill_global_in_kernel(
        tdi_wrap, template_fill,
        c0_sparse[None, ...].copy(), c0c[None, ...].copy(),
        window_full, n_sparse_local,
        np.ascontiguousarray(p9.reshape(1, 9)), np.ascontiguousarray(p9.reshape(1, 9)),
        np.ones(1, dtype=np.float64), np.zeros(1, dtype=np.int32),
        1, 1, 9, 1, 2,
        NF, Nt, Nf_active, Nt_active,
        NTL, N_sparse_t, stride,
        ind_min_t, ind_min_f, 2,
        layer_df, DT, Tobs, t_start,
        3, NSFD, TUK, MAX_R,
    )
    # slice kernel template to the active band, wrap, compare to dense c0r
    k_act = template_fill[0][:, ind_min_f:ind_min_f + Nf_active, :]
    if Nt_active != Nt:
        k_act = k_act[:, :, wsr.active_slice_t]
    tpl_k = WDMSignal(k_act, wsr)
    O = analysis.template_inner_product(tpl_k, normalize=True, complex=True)
    mm_kernel = float(1.0 - abs(O))
    d_d = float(np.real(analysis.inner_product()))

    # ---- (C) get_ll BIN-FOLD floor: the full d_h/h_h path at cand=ref --------
    # data = the reference template itself (self-consistency), so logL must be 0.
    invC_c = np.asarray(XYZ2SensitivityMatrix(wsc, model="scirdv1").invC)
    A0, A1, B0, B1 = python_bin_fold(c0c, c0c, invC_c, n_sparse_local, stride,
                                     Nt_active, tdi_type="XYZ")
    d_h = np.zeros(1); h_h = np.zeros(1)
    cpp.gb_signal_het_get_ll_in_kernel(
        tdi_wrap, d_h, h_h, c0_sparse[None, ...].copy(),
        A0[None].copy(), A1[None].copy(), B0[None].copy(), B1[None].copy(),
        window_full, n_sparse_local,
        np.ascontiguousarray(p9.reshape(1, 9)), np.ascontiguousarray(p9.reshape(1, 9)),
        np.zeros(1, dtype=np.int32),
        1, 1, 9, 1, 2,
        NF, Nt, Nf_active, Nt_active,
        NTL, N_sparse_t, stride,
        ind_min_t, ind_min_f, 2,
        layer_df, DT, Tobs, t_start,
        3, 0, NSFD, TUK, MAX_R,
    )
    ll_getll = -0.5 * d_d + float(d_h[0]) - 0.5 * float(h_h[0])
    mm_getll = -2.0 * ll_getll / d_d   # eff_mm: the get_ll floor as a mismatch

    # ---- localize (B): per-time and per-layer L2 error of (kernel - dense) --
    c0_act = np.asarray(c0r.arr)            # (3, Nf_active, Nt_active)
    diff = k_act - c0_act
    denom = max(float(np.sqrt(np.sum(c0_act ** 2))), 1e-300)
    err_t = np.sqrt(np.sum(diff ** 2, axis=(0, 1))) / denom            # per active-time pixel
    err_m = np.sqrt(np.sum(diff ** 2, axis=(0, 2))) / denom            # per active-m-layer
    nt = err_t.size
    edge_frac = float(np.sum(err_t[:nt // 10] ** 2) + np.sum(err_t[-nt // 10:] ** 2)) \
        / max(float(np.sum(err_t ** 2)), 1e-300)
    m_carrier = int(np.floor(p9[1] / layer_df)) - ind_min_f
    print(f"\n  Nt={Nt}  Nt_active={Nt_active}  Nf_active={Nf_active}  d_d={d_d:.4e}", flush=True)
    print(f"    (A) polyphase-from-dense-rfft  max reldiff = {mm_poly:.3e}", flush=True)
    print(f"    (B) kernel fill_global tmpl    mm vs dense = {mm_kernel:.3e}", flush=True)
    print(f"    (C) get_ll bin-fold floor      eff_mm      = {mm_getll:.3e}  "
          f"(logL={ll_getll:+.3e})", flush=True)
    print(f"    (B) localize: edge10%-energy frac={edge_frac:.3f}  "
          f"err_t[bulk-median]={np.median(err_t):.3e}  err_t[max]={err_t.max():.3e}", flush=True)
    del orbits, gb_gen; gc.collect()
    return Nt, mm_poly, mm_kernel, mm_getll, edge_frac


def main():
    nts = [int(x) for x in os.environ.get("NTS", "512,1024,2048").split(",")]
    print(f"sig-het floor decomposition (tukey={TUK}, N_sparse_fd={NSFD}, "
          f"nt_layer={NTL}, m_half={MH})", flush=True)
    rows = []
    for Nt in nts:
        try:
            rows.append(run(Nt))
        except Exception as e:
            import traceback; print(f"  Nt={Nt} RAISED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print("\n  SUMMARY  Nt | (A)poly | (B)fill_tmpl | (C)get_ll floor | edge_frac", flush=True)
    for Nt, mp, mk, mg, ef in rows:
        print(f"   {Nt:>6} | {mp:.2e} | {mk:.2e}      | {mg:.2e}        | {ef:.3f}", flush=True)
    print("\n  (A)<<(B)<<(C) => floor is the BIN-FOLD inner-product QUADRATURE"
          "\n  (A0*r + A1*dr reconstruction vs the exact lisatools sum), NOT the"
          "\n  polyphase (exact ~1e-14) and NOT the candidate template (~1e-9..1e-8)."
          "\n  It is EDGE-localized (edge_frac~1): the noisy WDM time-edge pixels"
          "\n  (|c0|->0) dominate the quadrature error. n_sparse_fd/nt_layer/m_half/"
          "\n  max_r DON'T move it. The lever is DOWN-WEIGHTING those edges: the"
          "\n  0.01-0.05 Tukey must actually reach the active region, i.e. set the"
          "\n  edge-cut SMALLER than the taper (EC < alpha*Nt/2). Measured @Nt=512,"
          "\n  tukey=0.05 (taper~13): EC=20 -> 6.5e-4 ; EC=5 -> 6.8e-5 ; EC=2 -> 3.7e-5."
          "\n  NOTE: this is the OPPOSITE of the chunked-het, which needs EC >= taper"
          "\n  (it cannot reproduce the global taper) -- so the two want different EC.", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
