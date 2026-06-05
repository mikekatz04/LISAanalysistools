#!/usr/bin/env python
"""Accuracy + timing validation for ``GBSignalHetWDMGetLLv2``.

Mirrors ``gb_signal_het_wdm_test_script.py`` (v1) but adds:

  - Per-pixel reldiff check between v2's sparse complex-WDM coefs and the
    lisatools dense reference at the sparse positions (this is what the
    calibration auto-derives ``Nt_layer`` to satisfy).
  - Side-by-side perturbation sweep: lisatools direct vs v1 ``get_ll``
    vs v2 ``get_ll``.
  - Wall-time benchmark: ``get_ll`` for v1 vs v2 (~repetition, median).

Run::
    conda activate deving
    python gb_signal_het_wdm_v2_test.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

PROFILE = bool(int(os.environ.get("PROFILE", "0")))

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

from gb_signal_het_wdm import GBSignalHetWDMGetLL
from gb_signal_het_wdm_v2 import GBSparseComplexWDMGen, GBSignalHetWDMGetLLv2


PERTURBATIONS = [
    ("zero",        None, lambda x0: 0.0),
    ("df0_p1e-7",   1, lambda x0: x0[1] * 1e-7),
    ("df0_p1e-6",   1, lambda x0: x0[1] * 1e-6),
    ("df0_p1e-5",   1, lambda x0: x0[1] * 1e-5),
    ("damp_p1e-3",  0, lambda x0: x0[0] * 1e-3),
    ("dfdot_p1",    2, lambda x0: max(abs(x0[2]), 1e-18) * 1.0),
    ("dphi0_p1e-4", 4, lambda x0: 1e-4),
    ("dbeta_p1e-4", 8, lambda x0: 1e-4),
    ("dlam_p1e-4",  7, lambda x0: 1e-4),
]


def build_x0(layer_df):
    m_ref = int(3e-3 / layer_df)
    _m_offset = int(os.environ.get("M_OFFSET", "0"))
    _f_frac = float(os.environ.get("F_FRAC", "0.5"))
    return np.array([
        1.0e-22,
        (m_ref + _m_offset + _f_frac) * layer_df,
        float(os.environ.get("SOURCE_FDOT", "1e-17")),
        0.0,
        2.09802430298,
        0.23984234,
        1.234019814,
        4.09808143,
        float(os.environ.get("BETA", "0.04")),
    ], dtype=float)


def make_real_td_callable(gb_gen, t_arr):
    def _real_td(params9):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = params9
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]
    return _real_td


def make_complex_wdm_gen(real_td_cb, td_set, wdm_set_complex, window):
    def _gen(params9):
        td = real_td_cb(np.asarray(params9, dtype=float).reshape(9))
        return np.asarray(
            TDSignal(td, settings=td_set).transform(
                wdm_set_complex, window=window
            ).arr
        )
    return _gen


def make_real_wdm_gen(real_td_cb, td_set, wdm_set_real, window):
    def _gen(*params9):
        td = real_td_cb(np.asarray(params9, dtype=float).reshape(9))
        return TDSignal(td, settings=td_set).transform(
            wdm_set_real, window=window
        )
    return _gen


def relerr(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    backend = "cpu"

    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))
    wavelet_duration = Nf * dt
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start

    N_inj = 16384
    gb_tdi_kwargs = dict(
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = np.linspace(t_arr[0], t_arr[-1], N_inj)
    gb_gen = GBTDIonTheFly(
        t_tdi_inj, Nt * wavelet_duration, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs,
    )
    real_td_cb = make_real_td_callable(gb_gen, t_arr)

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = np.ones(Nobs)

    min_freq = 1e-4
    max_freq = 35.0e-3
    _EDGE_CUT = int(os.environ.get("EDGE_CUT", "20"))
    min_time = _EDGE_CUT * wavelet_duration
    max_time = (Nt - _EDGE_CUT) * wavelet_duration

    wdm_set_real = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=False, force_backend=backend,
    )
    wdm_set_complex = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=True, force_backend=backend,
    )

    Nf_active = wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1
    Nt_active = wdm_set_real.Nt_active
    ind_min_t = wdm_set_real.ind_min_t

    print(f"[setup] Nf={Nf} Nt={Nt} Nf_active={Nf_active} Nt_active={Nt_active} "
          f"ind_min_t={ind_min_t}", flush=True)
    print(f"[setup] layer_df={wdm_set_real.layer_df:.3e}Hz", flush=True)

    x0 = build_x0(wdm_set_real.layer_df)
    print(f"[x0] amp={x0[0]:.3e} f0={x0[1]*1e3:.5f}mHz fdot={x0[2]:.2e} "
          f"beta={x0[8]:.3f} lam={x0[7]:.3f}", flush=True)

    print("[step] injection TD ...", flush=True)
    d_real_td = real_td_cb(x0)
    data_real_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_real, window=window,
    )
    data_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_complex, window=window,
    )

    print("[step] h0 references ...", flush=True)
    h0_real_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_real, window=window,
    )
    h0_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_complex, window=window,
    )

    sens_real = XYZ2SensitivityMatrix(data_real_sig.settings, model="scirdv1")
    gb_wave_wrap = make_real_wdm_gen(real_td_cb, td_set, wdm_set_real, window)
    analysis = AnalysisContainer(data_real_sig, sens_real, signal_gen=gb_wave_wrap)

    sens_complex = XYZ2SensitivityMatrix(data_complex_sig.settings, model="scirdv1")
    invC_complex = np.asarray(sens_complex.invC)

    # --- v2 sparse generator ------------------------------------------
    nt_layer_default = max(16, int(Nt) // 16)
    # Round down to a divisor of Nt
    while Nt % nt_layer_default != 0 and nt_layer_default > 2:
        nt_layer_default -= 2
    Nt_layer_user = int(os.environ.get("V2_NT_LAYER", nt_layer_default))
    m_active_half_width = int(os.environ.get("V2_M_HALF_WIDTH", "2"))
    print(
        f"[step] building v2 sparse generator "
        f"(Nt_layer={Nt_layer_user}, m_active_half_width={m_active_half_width}) ...",
        flush=True,
    )
    sparse_gen = GBSparseComplexWDMGen(
        real_td_callable=real_td_cb,
        wdm_set_complex=wdm_set_complex,
        data_dt=dt,
        ind_min_t=int(ind_min_t),
        Nt_active=int(Nt_active),
        Nt_layer=Nt_layer_user,
        m_active_half_width=m_active_half_width,
    )
    print(
        f"  Nt_layer={sparse_gen.Nt_layer}  stride={sparse_gen.stride}  "
        f"N_sparse_t={sparse_gen.N_sparse_t}",
        flush=True,
    )
    # Polyphase is mathematically exact -- one-shot sanity check.
    reldiff_x0 = sparse_gen.verify_against_dense(x0, h0_complex_sig.arr)
    print(f"  per-pixel reldiff vs lisatools dense at x0: {reldiff_x0:.3e}",
          flush=True)

    # --- v1 hetorodyne (for direct cross-check) ------------------------
    print("[step] building v1 heterodyne ...", flush=True)
    N_sparse_t_v1 = int(sparse_gen.N_sparse_t)
    complex_wdm_gen_v1 = make_complex_wdm_gen(
        real_td_cb, td_set, wdm_set_complex, window,
    )
    het_v1 = GBSignalHetWDMGetLL(
        wdm_set_complex=wdm_set_complex,
        data_wdm_complex=data_complex_sig.arr,
        invC=invC_complex,
        c0_dense_complex=h0_complex_sig.arr,
        gb_complex_wdm_gen=complex_wdm_gen_v1,
        reference_params=x0,
        N_sparse_t=N_sparse_t_v1,
        tdi_type="XYZ",
        force_backend=backend,
    )
    print(
        f"  v1 ref_ll={het_v1.ref_ll:+.6e} ref_d_h0={het_v1.ref_d_h0:+.6e} "
        f"ref_h0_h0={het_v1.ref_h0_h0:+.6e}",
        flush=True,
    )

    # --- v2 heterodyne -------------------------------------------------
    print("[step] building v2 heterodyne ...", flush=True)
    het_v2 = GBSignalHetWDMGetLLv2(
        wdm_set_complex=wdm_set_complex,
        data_wdm_complex=data_complex_sig.arr,
        invC=invC_complex,
        c0_dense_complex=h0_complex_sig.arr,
        sparse_gen=sparse_gen,
        reference_params=x0,
        tdi_type="XYZ",
        force_backend=backend,
    )
    print(
        f"  v2 ref_ll={het_v2.ref_ll:+.6e} ref_d_h0={het_v2.ref_d_h0:+.6e} "
        f"ref_h0_h0={het_v2.ref_h0_h0:+.6e}",
        flush=True,
    )

    # --- Cross-check at x0 vs lisatools real WDM -----------------------
    dd_real = float(np.real(analysis.inner_product()))
    dh0_real = float(analysis.template_inner_product(h0_real_sig))
    h0h0_real = float(analysis.template_snr(h0_real_sig)[0]) ** 2
    ll_x0_real = float(
        analysis.calculate_signal_likelihood(*x0, source_only=True)
    )

    print("\n=== cross-check at x0 vs lisatools real WDM ===", flush=True)
    print(
        f"  <d|d>   real-WDM={dd_real:+.6e}  v2.ref_d_d={het_v2.ref_d_d:+.6e}  "
        f"reldiff={relerr(het_v2.ref_d_d, dd_real):.3e}",
        flush=True,
    )
    print(
        f"  <d|h0>  real-WDM={dh0_real:+.6e}  v2.ref_d_h0={het_v2.ref_d_h0:+.6e}  "
        f"reldiff={relerr(het_v2.ref_d_h0, dh0_real):.3e}",
        flush=True,
    )
    print(
        f"  <h0|h0> real-WDM={h0h0_real:+.6e}  v2.ref_h0_h0={het_v2.ref_h0_h0:+.6e}  "
        f"reldiff={relerr(het_v2.ref_h0_h0, h0h0_real):.3e}",
        flush=True,
    )

    ll_v1_x0 = het_v1.get_ll(x0)
    ll_v2_x0 = het_v2.get_ll(x0)
    print(
        f"  ll(x0) real-WDM={ll_x0_real:+.6e}  v1={ll_v1_x0:+.6e}  v2={ll_v2_x0:+.6e}  "
        f"reldiff(v2,real)={relerr(ll_v2_x0, ll_x0_real):.3e}  "
        f"reldiff(v2,v1)={relerr(ll_v2_x0, ll_v1_x0):.3e}",
        flush=True,
    )

    # --- Perturbation sweep --------------------------------------------
    print("\n=== perturbation sweep (lisatools direct | v1 | v2) ===", flush=True)
    for label, idx, delta_fn in PERTURBATIONS:
        x1 = x0.copy()
        delta = float(delta_fn(x0))
        if idx is not None:
            x1[idx] = x0[idx] + delta
        ll_direct = float(
            analysis.calculate_signal_likelihood(*x1, source_only=True)
        )
        ll_v1 = het_v1.get_ll(x1)
        ll_v2 = het_v2.get_ll(x1)
        print(
            f"  [{label:<14}] delta={delta:+.3e}  "
            f"direct={ll_direct:+.6e}  v1={ll_v1:+.6e}  v2={ll_v2:+.6e}  "
            f"|v2-v1|={abs(ll_v2 - ll_v1):.3e}  "
            f"|v2-direct|={abs(ll_v2 - ll_direct):.3e}",
            flush=True,
        )

    if PROFILE:
        print("\n=== component profiling (per call) ===", flush=True)
        x_test_p = x0.copy()
        x_test_p[1] = x0[1] + x0[1] * 1e-7
        n_pwarm = 2
        for _ in range(n_pwarm):
            real_td_cb(x_test_p); complex_wdm_gen_v1(x_test_p); sparse_gen(x_test_p)
        # TD synth
        t0 = time.perf_counter()
        td_p = np.asarray(real_td_cb(x_test_p))
        t1 = time.perf_counter()
        # rfft
        fd_p = np.fft.rfft(td_p, axis=-1)
        t2 = time.perf_counter()
        # full TDSignal.transform (v1 dense WDM transform alone)
        cwdm = np.asarray(TDSignal(td_p, settings=td_set).transform(
            wdm_set_complex, window=window).arr)
        t3 = time.perf_counter()
        # v2 polyphase WDM math only (rfft already done)
        sparse_p, _ = sparse_gen.sparse_from_rfft(fd_p, x_test_p)
        t4 = time.perf_counter()
        # v2 full (TD synth + rfft + polyphase)
        sparse_full, _ = sparse_gen(x_test_p)
        t5 = time.perf_counter()
        print(f"  real_td_cb (TD synth)         : {(t1-t0)*1e3:.2f} ms  [one-time setup-equivalent]", flush=True)
        print(f"  rfft(td)                      : {(t2-t1)*1e3:.2f} ms  [one-time setup-equivalent]", flush=True)
        print(f"  v1 TDSignal.transform (dense) : {(t3-t2)*1e3:.2f} ms  [PER CALL in v1]", flush=True)
        print(f"  v2 polyphase WDM math ONLY    : {(t4-t3)*1e3:.2f} ms  [PER CALL in v2 -- this is the win]", flush=True)
        print(f"  v2 full (TD + rfft + polyphase): {(t5-t4)*1e3:.2f} ms  [if TD synth is on hot path]", flush=True)

    # --- Wall-time benchmark -------------------------------------------
    n_warm = 2
    n_reps = int(os.environ.get("N_REPS", "8"))
    x_test = x0.copy()
    x_test[1] = x0[1] + x0[1] * 1e-7  # small df0 perturb to avoid same-x cache
    for _ in range(n_warm):
        het_v1.get_ll(x_test)
        het_v2.get_ll(x_test)

    times_v1 = []
    times_v2 = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = het_v1.get_ll(x_test)
        t1 = time.perf_counter()
        _ = het_v2.get_ll(x_test)
        t2 = time.perf_counter()
        times_v1.append(t1 - t0)
        times_v2.append(t2 - t1)
    times_v1 = np.array(times_v1) * 1e3   # ms
    times_v2 = np.array(times_v2) * 1e3
    print(
        f"\n=== timing (median, n={n_reps}) ===\n"
        f"  v1 get_ll : {np.median(times_v1):.2f} ms  (min {times_v1.min():.2f}, max {times_v1.max():.2f})\n"
        f"  v2 get_ll : {np.median(times_v2):.2f} ms  (min {times_v2.min():.2f}, max {times_v2.max():.2f})\n"
        f"  speedup   : {np.median(times_v1) / max(np.median(times_v2), 1e-9):.1f}x",
        flush=True,
    )

    print("\nDONE.")


if __name__ == "__main__":
    sys.exit(main())
