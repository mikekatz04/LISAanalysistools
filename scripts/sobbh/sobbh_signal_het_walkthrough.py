#!/usr/bin/env python
"""4-panel explainer for the SOBBH signal-heterodyne (v2 polyphase) pipeline.

SOBBH analogue of ``gb_chunked_het/gb_signal_het_wdm_v2_walkthrough.py``.
For each of a few candidate logL levels (reached by perturbing f_low away from
the injection) it draws one row of four panels:

  0. candidate dense real-WDM template on the active m-band
     (the thing a brute-force dense likelihood would compare).
  1. v2 sparse complex-WDM  Re(c1_sparse[m, n_sparse]) from the polyphase fold.
  2. heterodyne ratio  r(n) = c1_sparse / c0_sparse  (Re, Im) -- slowly varying
     in n when the candidate is near the reference (the whole point of the
     heterodyne factorisation).
  3. numbers: lisatools-dense logL, SOBBH C++ signal-het logL, their diff,
     and the narrowband mm5 / mm2.

The panels use the source-agnostic Python v2 prototype (GBSparseComplexWDMGen,
fed the SOBBH TD callable); the C++ signal-het logL comes from
SOBBHSignalHetComputations. Synthetic SOBBH injection by default.

Run::
    python sobbh_signal_het_walkthrough.py
Env: SEED (321), NT_LAYER (64), OUT_PNG (sobbh_signal_het_walkthrough.png)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

_THIS = os.path.dirname(os.path.abspath(__file__))
_GB_DIR = os.path.join(_THIS, "..", "gb_chunked_het")
for _d in (_THIS, _GB_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen          # noqa: E402
from sobbhsignalhetcomputations import SOBBHSignalHetComputations  # noqa: E402


def main():
    SEED = int(os.environ.get("SEED", "321"))
    Nt_layer = int(os.environ.get("NT_LAYER", "64"))
    OUT_PNG = os.environ.get("OUT_PNG", "sobbh_signal_het_walkthrough.png")

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
    sobbh_gen = SOBBHTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / dt, 1,
                                 tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                                 force_backend=backend)

    def real_td_cb(p):
        sp = sobbh_gen(*np.asarray(p, float).reshape(11, 1),
                       convert_to_ra_dec=False, return_spline=True)
        return np.asarray(sp.eval_tdi(t_arr))[0]

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = _tukey(Nobs, alpha=0.05).astype(float)
    wdm_kw = dict(t0=t_start, min_freq=1e-4, max_freq=35e-3,
                  min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt, force_backend=backend)
    wdm_set_real = WDMSettings(Nf, Nt, dt, is_complex=False, **wdm_kw)
    wdm_set_complex = WDMSettings(Nf, Nt, dt, is_complex=True, **wdm_kw)
    layer_df = wdm_set_real.layer_df
    ind_min_f = int(wdm_set_real.ind_min_f)
    ind_min_t = int(wdm_set_real.ind_min_t)
    Nt_active = int(wdm_set_real.Nt_active)

    rng = np.random.default_rng(SEED)
    # injection: a mid-band SOBBH
    f_low = float((ind_min_f + 80) * layer_df)
    ref = np.array([42.0, 38.0, 0.1, 0.2, 5.0e8, f_low, 1.1,
                    np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)])
    td_inj = real_td_cb(ref)
    data_real = TDSignal(td_inj, settings=td_set).transform(wdm_set_real, window=window)
    sens = XYZ2SensitivityMatrix(data_real.settings, model="scirdv1")
    analysis = AnalysisContainer(DataResidualArray(data_real), sens)
    analysis.signal_gen = lambda *p: TDSignal(real_td_cb(np.asarray(p, float).reshape(11)),
                                              settings=td_set).transform(wdm_set_real, window=window)
    d_d = float(np.real(analysis.inner_product()))
    snr = float(analysis.snr())
    print(f"[inj] f_low={f_low*1e3:.4f}mHz SNR={snr:.1f} <d|d>={d_d:.3e}", flush=True)

    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb, wdm_set_complex=wdm_set_complex, data_dt=dt,
        ind_min_t=ind_min_t, Nt_active=Nt_active, Nt_layer=Nt_layer, m_active_half_width=2)
    c0_dense = np.asarray(TDSignal(td_inj, settings=td_set).transform(
        wdm_set_complex, window=window).arr)
    c0_sparse = c0_dense[:, :, sparse_gen.n_sparse_local]

    sighet = SOBBHSignalHetComputations(
        td_inj, ref, Nf=Nf, Nt=Nt, dt=dt, t0=t_start, t_ref=t_start,
        orbits=orbits, tdi_config=tdi_config, min_freq=1e-4, max_freq=35e-3,
        nt_layer=Nt_layer, force_backend=backend)

    def narrowband_mm(tpl_active_arr, lo, hi):
        ws = WDMSettings(Nf, Nt, dt, min_time=wdm_set_real.min_time, max_time=wdm_set_real.max_time,
                         min_freq=lo, max_freq=hi, force_backend=backend)
        i_lo = ws.ind_min_f - ind_min_f; i_hi = ws.ind_max_f - ind_min_f + 1
        d_arr = np.asarray(data_real.arr)
        ac = AnalysisContainer(DataResidualArray(WDMSignal(d_arr[:, i_lo:i_hi], ws)),
                               XYZ2SensitivityMatrix(ws, model="scirdv1"))
        return float(1.0 - ac.template_inner_product(
            DataResidualArray(WDMSignal(tpl_active_arr[:, i_lo:i_hi], ws)), normalize=True))

    # find f0 perturbations that land near target logL levels
    targets = [0.0, -5.0, -50.0]
    rows = []
    for tgt in targets:
        # crude bisection on df0 (in layer_df units) to hit the dense logL target
        lo, hi = 0.0, 0.5
        best = 0.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            p = ref.copy(); p[5] = f_low + mid * layer_df
            ll = float(analysis.calculate_signal_likelihood(*p, source_only=True))
            if ll > tgt:
                lo = mid
            else:
                hi = mid
            best = mid
        rows.append((tgt, ref[5] + best * layer_df))

    M = 2 * 2 + 1
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 3.6 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for ri, (tgt, f0_cand) in enumerate(rows):
        p = ref.copy(); p[5] = f0_cand
        td_cand = real_td_cb(p)
        # dense real-WDM template on active band
        tpl_real = np.asarray(TDSignal(td_cand, settings=td_set).transform(
            wdm_set_real, window=window).arr)
        m_floor = int(f0_cand / layer_df)
        ms = np.clip(np.arange(m_floor - 2, m_floor + 3) - ind_min_f, 0, tpl_real.shape[1] - 1)

        # v2 sparse complex + ratio (channel 0)
        fd_rfft = np.fft.rfft(td_cand * window, axis=-1).astype(np.complex128)
        g9 = np.zeros(9); g9[1] = f0_cand
        c1_active, m_local = sparse_gen.sparse_from_rfft(fd_rfft, g9)
        c0_act = c0_sparse[:, m_local, :]
        denom = np.where(np.abs(c0_act) > 1e-12 * np.abs(c0_act).max(), c0_act, 1.0)
        r = np.where(np.abs(c0_act) > 1e-12 * np.abs(c0_act).max(), c1_active / denom, 0.0)

        # logLs + mm
        ll_dense = float(analysis.calculate_signal_likelihood(*p, source_only=True))
        ll_sighet = float(np.asarray(sighet.get_ll(p)).reshape(()))
        mm5 = narrowband_mm(tpl_real, f0_cand - 3 * layer_df, f0_cand + 2 * layer_df)
        mm2 = narrowband_mm(tpl_real, m_floor * layer_df, (m_floor + 2) * layer_df - 0.5 * layer_df)

        ax = axes[ri, 0]
        ax.imshow(tpl_real[0, ms, :], aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_title(f"[row {ri}] dense real-WDM (chan X)\n5 active m-layers")
        ax.set_ylabel(f"target logL={tgt:.0f}\nm-layer")

        ax = axes[ri, 1]
        ax.imshow(np.real(c1_active[0]), aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_title("v2 sparse Re(c1_sparse)")
        ax.set_xlabel("n_sparse")

        ax = axes[ri, 2]
        ax.plot(np.real(r[0, 2]), label="Re r (carrier m)")
        ax.plot(np.imag(r[0, 2]), label="Im r")
        ax.set_title("heterodyne ratio r(n), chan X")
        ax.set_xlabel("n_sparse"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[ri, 3]; ax.axis("off")
        txt = (f"f0_cand = {f0_cand*1e3:.5f} mHz\n"
               f"Δf0 = {(f0_cand - f_low)/layer_df:+.4f} layer_df\n\n"
               f"logL dense   = {ll_dense:+.4e}\n"
               f"logL sig-het = {ll_sighet:+.4e}\n"
               f"|diff|       = {abs(ll_sighet - ll_dense):.3e}\n\n"
               f"mm5 = {mm5:+.3e}\n"
               f"mm2 = {mm2:+.3e}")
        ax.text(0.02, 0.95, txt, va="top", ha="left", family="monospace", fontsize=11)
        print(f"[row {ri}] target={tgt:+.0f} f0={f0_cand*1e3:.5f}mHz "
              f"dense={ll_dense:+.3e} sighet={ll_sighet:+.3e} "
              f"|diff|={abs(ll_sighet-ll_dense):.2e} mm5={mm5:.2e} mm2={mm2:.2e}", flush=True)

    fig.suptitle("SOBBH signal-heterodyne (v2 polyphase) walkthrough", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT_PNG, dpi=110)
    print(f"\n[write] {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
