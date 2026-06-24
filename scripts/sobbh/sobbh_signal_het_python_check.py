#!/usr/bin/env python
"""Pure-PYTHON validation that the signal-het algorithm works for SOBBHs.

No new C++ needed: this drives the SOURCE-AGNOSTIC v2 polyphase + bin-fold
prototype (GBSparseComplexWDMGen + python_bin_fold_real) with the SOBBH
waveform (lisatools.response.tdionfly.SOBBHTDIonTheFly, already available via
the existing cbbhx). It confirms the SOBBH sig-het likelihood reproduces the
dense lisatools likelihood -- i.e. the maths/port is correct for SOBBHs at the
Python level, before/independent of the C++ kernel rebuild.

For a synthetic SOBBH injection and a few f_low perturbations it prints:
    logL_sighet_py   (polyphase + real-projection bin-fold, pure numpy)
    logL_dense       (lisatools AnalysisContainer.calculate_signal_likelihood)
    |diff|, mm5, mm2

PASS if sig-het tracks dense (|diff| at the injection < 1e-3, growing smoothly
off-peak -- the documented linear-r budget).

Run::  python sobbh_signal_het_python_check.py
Env:   SEED (4), NT_LAYER (64)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import SOBBHTDIonTheFly

import bbhx  # noqa: F401

_GB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gb_chunked_het")
if _GB_DIR not in sys.path:
    sys.path.insert(0, _GB_DIR)
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen          # noqa: E402
from gb_signal_het_cpp_validate import python_bin_fold_real     # noqa: E402


def main():
    SEED = int(os.environ.get("SEED", "4"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 1460, 2560
    Nobs = Nf * Nt
    EC = 20
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    Tobs = Nobs * dt

    orbits = ESAOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
    gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / dt, 1,
                           tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                           force_backend=backend)

    def real_td_cb(p):
        sp = gen(*np.asarray(p, float).reshape(11, 1), convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)
    wdm_kw = dict(t0=t_start, min_freq=1e-4, max_freq=35e-3,
                  min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt, force_backend=backend)
    wdm_real = WDMSettings(Nf, Nt, dt, is_complex=False, **wdm_kw)
    wdm_cplx = WDMSettings(Nf, Nt, dt, is_complex=True, **wdm_kw)
    layer_df = wdm_real.layer_df
    ind_min_f = int(wdm_real.ind_min_f)
    ind_min_t = int(wdm_real.ind_min_t)
    Nt_active = int(wdm_real.Nt_active)
    Nf_active = int(wdm_real.ind_max_f - wdm_real.ind_min_f + 1)

    # synthetic SOBBH injection
    f_low = float((ind_min_f + 80) * layer_df)
    ref = np.array([42.0, 38.0, 0.1, 0.2, 5.0e8, f_low, 1.1,
                    np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)])
    td_inj = real_td_cb(ref)
    assert np.all(np.isfinite(td_inj)), "SOBBH injection not finite"
    data_real = TDSignal(td_inj, settings=td_set).transform(wdm_real, window=window)
    sens = XYZ2SensitivityMatrix(data_real.settings, model="scirdv1")
    analysis = AnalysisContainer(DataResidualArray(data_real), sens)
    analysis.signal_gen = lambda *p: TDSignal(real_td_cb(np.asarray(p, float).reshape(11)),
                                              settings=td_set).transform(wdm_real, window=window)
    d_d = float(np.real(analysis.inner_product()))
    snr = float(analysis.snr())
    print(f"[inj] f_low={f_low*1e3:.4f}mHz SNR={snr:.1f} <d|d>={d_d:.3e}", flush=True)

    data_complex = np.asarray(TDSignal(td_inj, settings=td_set).transform(wdm_cplx, window=window).arr)
    c0_dense = data_complex.copy()                 # reference = injection
    invC_complex = np.asarray(XYZ2SensitivityMatrix(wdm_cplx, model="scirdv1").invC)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_cplx, data_dt=dt,
        ind_min_t=ind_min_t, Nt_active=Nt_active, Nt_layer=Nt_layer, m_active_half_width=2)
    stride = sparse_gen.stride
    N_sparse_t = sparse_gen.N_sparse_t
    n_sparse_local = np.asarray(sparse_gen.n_sparse_local, dtype=np.int32)
    c0_sparse = c0_dense[:, :, n_sparse_local]
    print(f"[v2] Nt_layer={Nt_layer} stride={stride} N_sparse_t={N_sparse_t}", flush=True)

    # real-projection bin-fold coefficients (source-agnostic)
    A0, A1, B0, B1, B0nc, B1nc = python_bin_fold_real(
        data_complex, c0_dense, invC_complex, n_sparse_local, stride, Nt_active, tdi_type="XYZ")

    def sighet_py_logL(p):
        """pure-numpy signal-het logL via polyphase + real-projection bin-fold."""
        td = real_td_cb(p)
        fd = np.fft.rfft(td * window, axis=-1).astype(np.complex128)
        g9 = np.zeros(9); g9[1] = p[5]
        c1_act, m_local = sparse_gen.sparse_from_rfft(fd, g9)
        c1_full = np.zeros((3, Nf_active, N_sparse_t), dtype=np.complex128)
        c1_full[:, m_local, :] = c1_act
        c0m = np.abs(c0_sparse)
        floor = np.maximum(1e-12 * c0m.max(axis=-1, keepdims=True), 1e-300)
        mask = c0m > floor
        r = np.where(mask, c1_full / np.where(mask, c0_sparse, 1.0), 0.0)
        Dn = float(stride)
        dr = np.zeros_like(r)
        if N_sparse_t >= 3:
            dr[..., 1:-1] = (r[..., 2:] - r[..., :-2]) / (2.0 * Dn)
            dr[..., 0] = (r[..., 1] - r[..., 0]) / Dn
            dr[..., -1] = (r[..., -1] - r[..., -2]) / Dn
        # <d|h>: 0.5*Re sum(A0*r + A1*dr)   (A0/A1 already real-repacked)
        d_h = 0.5 * float((A0 * r + A1 * dr).sum().real)
        # <h|h>: 0.5*Re sum(conj + nonconj blocks)  (real projection)
        rc = r.conj()[:, None]; rc2 = r[None, :]
        drc = dr.conj()[:, None]; dr2 = dr[None, :]
        rcn = r[:, None]  # non-conj
        hh = (B0 * (rc * rc2) + B1 * (rc * dr2 + drc * rc2)
              + B0nc * (rcn * rc2) + B1nc * (rcn * dr2 + dr[:, None] * rc2)).sum()
        h_h = 0.5 * float(hh.real)
        return -0.5 * d_d + d_h - 0.5 * h_h

    def narrowband_mm(p, lo, hi):
        td = real_td_cb(p)
        tpl = np.asarray(TDSignal(td, settings=td_set).transform(wdm_real, window=window).arr)
        ws = WDMSettings(Nf, Nt, dt, min_time=wdm_real.min_time, max_time=wdm_real.max_time,
                         min_freq=lo, max_freq=hi, force_backend=backend)
        i_lo = ws.ind_min_f - ind_min_f; i_hi = ws.ind_max_f - ind_min_f + 1
        d_arr = np.asarray(data_real.arr)
        ac = AnalysisContainer(DataResidualArray(WDMSignal(d_arr[:, i_lo:i_hi], ws)),
                               XYZ2SensitivityMatrix(ws, model="scirdv1"))
        return float(1.0 - ac.template_inner_product(
            DataResidualArray(WDMSignal(tpl[:, i_lo:i_hi], ws)), normalize=True))

    print(f"\n   {'perturbation':>16s} {'logL_sighet_py':>15s} {'logL_dense':>13s} "
          f"{'|diff|':>11s} {'mm5':>10s} {'mm2':>10s}", flush=True)
    PERTS = [
        ("zero",         None, 0.0),
        ("df0 +1e-2·lf", 5, 1e-2 * layer_df),
        ("df0 +1e-1·lf", 5, 1e-1 * layer_df),
        ("df0 +3e-1·lf", 5, 3e-1 * layer_df),
        ("dphi_c +0.05", 6, 0.05),
    ]
    diff0 = None
    for label, idx, delta in PERTS:
        p = ref.copy()
        if idx is not None:
            p[idx] = ref[idx] + delta
        ll_s = sighet_py_logL(p)
        ll_d = float(analysis.calculate_signal_likelihood(*p, source_only=True))
        m_floor = int(p[5] / layer_df)
        mm5 = narrowband_mm(p, p[5] - 3 * layer_df, p[5] + 2 * layer_df)
        mm2 = narrowband_mm(p, m_floor * layer_df, (m_floor + 2) * layer_df - 0.5 * layer_df)
        diff = abs(ll_s - ll_d)
        if idx is None:
            diff0 = diff
        print(f"   {label:>16s} {ll_s:+15.5e} {ll_d:+13.5e} {diff:11.3e} "
              f"{mm5:10.2e} {mm2:10.2e}", flush=True)

    print(f"\n[summary] sig-het(py) vs dense |diff| at injection = {diff0:.3e}", flush=True)
    ok = diff0 is not None and diff0 < 1e-3
    print("PASS" if ok else "CHECK", "(SOBBH signal-het reproduces dense logL in pure Python)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
