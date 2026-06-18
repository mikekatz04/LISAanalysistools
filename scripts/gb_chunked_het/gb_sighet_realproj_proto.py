"""Prototype: REAL-PROJECTION bin-fold for the sig-het likelihood.

Root cause (proven in gb_sighet_floor_deepdive.py): the current bin-fold computes
the COMPLEX WDM inner product 0.5*Re(Sum C*.invC.C), which differs from the REAL
WDM likelihood Sum Re(C).invC.Re(C) by ~6.5e-4 (the stray Im.Im term). Re-
projecting to real FIRST reproduces lisatools to 1e-15.

This prototypes the FAST (sparse bin-fold) version of the real projection:
with c1 = r*c0, u=Re(c0), w=Im(c0),
    Re(c1) = u*Re(r) - w*Im(r)
so <d|h>_real = Sum_pixels Re(Cd).invC.Re(c1)
             = Sum_bins [ A0re*Re(r) - A0im*Im(r) + A1re*Re(dr) - A1im*Im(dr) ]
with real per-bin coeffs A0re=Sum Dre.u, A0im=Sum Dre.w (Dre = Sum_c Re(Cd).invC),
and the n_off-weighted A1re/A1im. <h|h>_real expands Re(c1c).invC.Re(c1c2) into
four real coeff blocks (uu,uw,wu,ww) x {RrRr, RrIr, IrRr, IrIr}.

Compares, at cand=ref AND a small Df0:
  (1) dense REAL <d|h>/<h|h>           (truth)
  (2) complex bin-fold  (current)      -> ~6.5e-4 floor
  (3) real-proj bin-fold (this reform) -> target ~1e-12
"""
import os, sys
import numpy as np

import gbgpu  # noqa: F401
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.diagnostic import inner_product
from lisatools.utils.constants import YRSID_SI
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen  # noqa: E402
from gb_signal_het_cpp_validate import python_bin_fold_real  # noqa: E402

DT = 10.0; NF = 1460; EC = 20; NTL = 64; MH = 2; TUK = 0.05
NT = int(os.environ.get("NT", "512"))
DF0_FRAC = float(os.environ.get("DF0_FRAC", "0.0"))   # 0 => cand=ref


def bin_idx_of(N_sparse_t, stride, Nt_active):
    edges = np.arange(N_sparse_t + 1) * stride; edges[-1] = Nt_active
    return np.repeat(np.arange(N_sparse_t), np.diff(edges).astype(int))


def main():
    Nobs = NF * NT; Tobs = NT * NF * DT
    t0 = int(0.5 * YRSID_SI / DT) * DT
    t_arr = np.arange(Nobs) * DT + t0
    orbits = ESAOrbits(force_backend="cpu")
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gen = GBTDIonTheFly(t_tdi, Tobs, t0, 1.0 / DT, 1, tdi_config=tdi_config,
                        orbits=orbits, tdi_chan="XYZ", force_backend="cpu")
    p_inj = np.array([1e-21, 14.22e-3, 1e-16, 0.0, 1.4, np.pi / 3.0, 0.7, 2.1, 0.5])

    td_set = TDSettings(Nobs, DT, t0=t0, force_backend="cpu")
    from scipy.signal.windows import tukey as _tk
    window = _tk(Nobs, TUK).astype(float)

    def real_td(p):
        sp = gen(*[np.array([x]) for x in np.asarray(p, float).reshape(9)],
                 convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    def real_td_win(p):
        return real_td(p) * window[None, :]

    wkw = dict(t0=t0, min_freq=1e-4, max_freq=35e-3, min_time=EC * NF * DT,
               max_time=(NT - EC) * NF * DT, force_backend="cpu")
    wsr = WDMSettings(NF, NT, DT, is_complex=False, **wkw)
    wsc = WDMSettings(NF, NT, DT, is_complex=True, **wkw)
    ind_min_t = int(wsr.ind_min_t); ind_min_f = int(wsr.ind_min_f)
    Nt_active = int(wsr.Nt_active); Nf_active = int(wsr.ind_max_f - wsr.ind_min_f + 1)
    layer_df = wsr.layer_df

    # data = dense real WDM at injection (the lisatools "truth" reference)
    td_inj = real_td(p_inj)
    data_r = TDSignal(td_inj, td_set).transform(wsr, window=window)       # real WDM data
    data_c = np.asarray(TDSignal(td_inj, td_set).transform(wsc, window=window).arr)  # complex
    sens = XYZ2SensitivityMatrix(wsr, model="scirdv1")
    invC = np.asarray(XYZ2SensitivityMatrix(wsc, model="scirdv1").invC)   # (3,3,Nf_a,Nt_a)
    analysis = AnalysisContainer(data_r, sens)

    # candidate
    p_cand = p_inj.copy()
    p_cand[1] = p_inj[1] + DF0_FRAC * layer_df

    # sparse polyphase c1 at candidate (windowed callable so taper matches)
    sg = GBSparseComplexWDMGen(real_td_win, wsc, DT, ind_min_t, Nt_active, NTL, MH)
    stride = sg.stride; N_sparse_t = sg.N_sparse_t
    nsl = np.asarray(sg.n_sparse_local, dtype=int)
    c1_act, m_local = sg(p_cand)            # (3, M, N_sparse_t), m_local local idx
    M = c1_act.shape[1]

    # c0 (heterodyne reference = dense complex WDM at injection) on the active band
    c0_dense = data_c[:, m_local, :]                        # (3, M, Nt_active) complex
    c0_sparse = c0_dense[:, :, nsl]                         # (3, M, N_sparse_t)
    invC_act = invC[:, :, m_local, :]                       # (3,3,M,Nt_active)

    # r, dr at sparse grid (same safe-divide + central diff as the kernel)
    mag = np.abs(c0_sparse); thr = np.maximum(1e-12 * mag.max(axis=-1, keepdims=True), 1e-300)
    r = np.where(mag > thr, c1_act / np.where(mag > thr, c0_sparse, 1.0), 0.0)
    Dn = float(stride)
    dr = np.zeros_like(r)
    dr[..., 1:-1] = (r[..., 2:] - r[..., :-2]) / (2.0 * Dn)
    dr[..., 0] = (r[..., 1] - r[..., 0]) / Dn
    dr[..., -1] = (r[..., -1] - r[..., -2]) / Dn

    # ---- dense REAL truth: <d|c1_dense> / <c1_dense|c1_dense> over active band ----
    # reconstruct c1 on the dense grid (linear-interp r), Re(), real WDM inner prod
    bidx = bin_idx_of(N_sparse_t, stride, Nt_active)
    n_off = (np.arange(Nt_active) - nsl[bidx]).astype(float)
    r_dense = r[:, :, bidx] + dr[:, :, bidx] * n_off[None, None, :]       # (3,M,Nt_active)
    c1_dense = r_dense * c0_dense
    Re_c1 = np.real(c1_dense)
    Re_d = np.real(data_c[:, m_local, :])
    d_h_true = float(np.einsum("amn,abmn,bmn->", Re_d, invC_act, Re_c1))
    h_h_true = float(np.einsum("amn,abmn,bmn->", Re_c1, invC_act, Re_c1))

    # ---- (2) COMPLEX bin-fold (current form): 0.5 Re(Sum A0 r + ...) -----------
    Dc = np.einsum("amn,abmn->bmn", np.conj(data_c[:, m_local, :]), invC_act)  # data*.invC
    wA = Dc * c0_dense
    A0c = np.stack([wA[:, :, bidx == b].sum(-1) for b in range(N_sparse_t)], axis=-1)
    A1c = np.stack([(wA[:, :, bidx == b] * n_off[bidx == b]).sum(-1) for b in range(N_sparse_t)], axis=-1)
    d_h_cplx = 0.5 * np.real(np.sum(A0c * r + A1c * dr))

    # ---- (3) REAL-PROJECTION bin-fold (the reform) ----------------------------
    u = np.real(c0_dense); w = np.imag(c0_dense)            # (3,M,Nt_active)
    # Real-projected data weight: Sum_a Re(data_a) * Re(invC[a,b])  (real WDM uses
    # the real part of both the data and the sensitivity).
    Dre = np.einsum("amn,abmn->bmn", Re_d, np.real(invC_act))
    wA_re = Dre * u; wA_im = Dre * w
    A0re = np.stack([wA_re[:, :, bidx == b].sum(-1) for b in range(N_sparse_t)], axis=-1)
    A0im = np.stack([wA_im[:, :, bidx == b].sum(-1) for b in range(N_sparse_t)], axis=-1)
    A1re = np.stack([(wA_re[:, :, bidx == b] * n_off[bidx == b]).sum(-1) for b in range(N_sparse_t)], axis=-1)
    A1im = np.stack([(wA_im[:, :, bidx == b] * n_off[bidx == b]).sum(-1) for b in range(N_sparse_t)], axis=-1)
    d_h_real = float(np.sum(A0re * np.real(r) - A0im * np.imag(r)
                            + A1re * np.real(dr) - A1im * np.imag(dr)))

    # ---- <h|h> REAL-PROJECTION bin-fold -------------------------------------
    # Re(c1_c)(pixel) = sum_i kcoef[i,c,bin] * basis[i,c,pixel], with
    #   basis: b0=u, b1=-w, b2=u*n_off, b3=-w*n_off   (per pixel/channel/layer)
    #   kcoef: k0=Re(r), k1=Im(r), k2=Re(dr), k3=Im(dr) (per bin/channel/layer)
    # h_h = sum_{m,bin,c,c2,i,j} k[i,c]*G_ij[c,c2]*k[j,c2],
    #   G_ij[c,c2,m,bin] = sum_{pixels in bin} basis[i,c]*invC[c,c2]*basis[j,c2].
    basis = np.stack([u, -w, u * n_off[None, None, :], -w * n_off[None, None, :]], axis=0)  # (4,3,M,Nt_a)
    kcoef = np.stack([np.real(r), np.imag(r), np.real(dr), np.imag(dr)], axis=0)            # (4,3,M,Ns)
    h_h_real = 0.0
    for b in range(N_sparse_t):
        sel = (bidx == b)
        bb = basis[:, :, :, sel]                  # (4,3,M,nb)
        ic = invC_act[:, :, :, sel]               # (3,3,M,nb)
        # G[i,j,c,c2,m] = sum_n bb[i,c,m,n]*ic[c,c2,m,n]*bb[j,c2,m,n]
        G = np.einsum("icmn,cdmn,jdmn->ijcdm", bb, ic, bb)
        k = kcoef[:, :, :, b]                      # (4,3,M)
        h_h_real += float(np.real(np.einsum("icm,ijcdm,jdm->", k, G, k)))

    # ---- STAY-COMPLEX route: add the NON-CONJUGATED sum, choose real at the end
    # real = 0.5 Re(conj_sum + nonconj_sum). conj_sum is the CURRENT bin-fold.
    def binsum(x):  # sum x (..., Nt_active) into (..., N_sparse_t) bins
        return np.stack([x[..., bidx == b].sum(-1) for b in range(N_sparse_t)], axis=-1)
    # <d|h>: nonconj uses data (NO conj) . invC . c0
    Dnc = np.einsum("amn,abmn->bmn", data_c[:, m_local, :], invC_act)
    wAnc = Dnc * c0_dense
    A0nc = binsum(wAnc); A1nc = binsum(wAnc * n_off[None, None, :])
    dh_conj_raw = np.sum(A0c * r + A1c * dr)                  # == current complex raw
    dh_nonc_raw = np.sum(A0nc * r + A1nc * dr)
    d_h_realcplx = 0.5 * np.real(dh_conj_raw + dh_nonc_raw)
    # <h|h>: E[c,c2] = (conj or not) c0_c . invC . c0_c2 ; conj form pairs conj(r_c) r_c2
    E_conj = np.conj(c0_dense)[:, None] * invC_act * c0_dense[None, :]   # (3,3,M,Nt_a)
    E_nonc = c0_dense[:, None] * invC_act * c0_dense[None, :]
    B0c = binsum(E_conj); B1c = binsum(E_conj * n_off[None, None, None, :])
    B0n = binsum(E_nonc); B1n = binsum(E_nonc * n_off[None, None, None, :])
    rc, drc = r, dr
    hh_conj_raw = (np.einsum("abmt,amt,bmt->", B0c, np.conj(rc), rc)
                   + np.einsum("abmt,amt,bmt->", B1c, np.conj(rc), drc)
                   + np.einsum("abmt,amt,bmt->", B1c, np.conj(drc), rc))
    hh_nonc_raw = (np.einsum("abmt,amt,bmt->", B0n, rc, rc)
                   + np.einsum("abmt,amt,bmt->", B1n, rc, drc)
                   + np.einsum("abmt,amt,bmt->", B1n, drc, rc))
    h_h_realcplx = 0.5 * np.real(hh_conj_raw + hh_nonc_raw)

    # ---- (4) via python_bin_fold_real (the FUNCTION to be ported to C++) ----
    # kernel-style: d_h = 0.5 Re(Sum A0p r + A1p dr);
    #               h_h = 0.5 Re(Sum B0 conj(rc)rc2 + B0nc rc rc2 + B1/B1nc dr terms)
    A0p, A1p, B0f, B1f, B0n, B1n = python_bin_fold_real(
        data_c[:, m_local, :], c0_dense, invC_act, nsl, stride, Nt_active, "XYZ")
    d_h_fn = 0.5 * float(np.real(np.sum(A0p * r + A1p * dr)))
    hh_c = (np.einsum("abmt,amt,bmt->", B0f, np.conj(r), r)
            + np.einsum("abmt,amt,bmt->", B1f, np.conj(r), dr)
            + np.einsum("abmt,amt,bmt->", B1f, np.conj(dr), r))
    hh_n = (np.einsum("abmt,amt,bmt->", B0n, r, r)
            + np.einsum("abmt,amt,bmt->", B1n, r, dr)
            + np.einsum("abmt,amt,bmt->", B1n, dr, r))
    h_h_fn = 0.5 * float(np.real(hh_c + hh_n))

    dd = float(np.real(analysis.inner_product()))
    print(f"  NT={NT}  DF0_FRAC={DF0_FRAC}  M={M}  N_sparse_t={N_sparse_t}  stride={stride}", flush=True)
    print(f"  <d|h>  true(real WDM) = {d_h_true:.10e}", flush=True)
    print(f"         complex binfold= {d_h_cplx:.10e}   reldiff={abs(d_h_cplx-d_h_true)/abs(d_h_true):.3e}", flush=True)
    print(f"         REAL-PROJ bin  = {d_h_real:.10e}   reldiff={abs(d_h_real-d_h_true)/abs(d_h_true):.3e}", flush=True)
    print(f"  (eff_mm complex ~ {2*abs(d_h_cplx-d_h_true)/dd:.2e};  real-proj ~ {2*abs(d_h_real-d_h_true)/dd:.2e})", flush=True)
    print(f"  <h|h>  true(real WDM) = {h_h_true:.10e}", flush=True)
    print(f"         REAL-PROJ bin  = {h_h_real:.10e}   reldiff={abs(h_h_real-h_h_true)/abs(h_h_true):.3e}", flush=True)
    ll_real = -0.5 * dd + d_h_real - 0.5 * h_h_real
    print(f"  logL real-proj bin-fold = {ll_real:+.4e}  (eff_mm {abs(2*ll_real/dd):.2e})", flush=True)
    print(f"  --- STAY-COMPLEX (conj + nonconj, choose real at end) ---", flush=True)
    print(f"  <d|h> real via 0.5Re(conj+nonconj) = {d_h_realcplx:.10e}  reldiff={abs(d_h_realcplx-d_h_true)/abs(d_h_true):.3e}", flush=True)
    print(f"  <h|h> real via 0.5Re(conj+nonconj) = {h_h_realcplx:.10e}  reldiff={abs(h_h_realcplx-h_h_true)/abs(h_h_true):.3e}", flush=True)
    ll_rc = -0.5 * dd + d_h_realcplx - 0.5 * h_h_realcplx
    print(f"  logL stay-complex->real = {ll_rc:+.4e}  (eff_mm {abs(2*ll_rc/dd):.2e})", flush=True)
    print(f"  --- python_bin_fold_real (FUNCTION -> C++ port contract) ---", flush=True)
    print(f"  <d|h> fn = {d_h_fn:.10e}  reldiff={abs(d_h_fn-d_h_true)/abs(d_h_true):.3e}", flush=True)
    print(f"  <h|h> fn = {h_h_fn:.10e}  reldiff={abs(h_h_fn-h_h_true)/abs(h_h_true):.3e}", flush=True)
    ll_fn = -0.5 * dd + d_h_fn - 0.5 * h_h_fn
    print(f"  logL fn  = {ll_fn:+.4e}  (eff_mm {abs(2*ll_fn/dd):.2e})", flush=True)


if __name__ == "__main__":
    main()
