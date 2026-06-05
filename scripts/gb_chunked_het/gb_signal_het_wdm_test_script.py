#!/usr/bin/env python
"""
Accuracy validation for :class:`GBSignalHetWDMGetLL`.

For a single GB injection ``d``:
  1. Build the data in REAL WDM (standard lisatools path -- this is
     the ground truth) and in COMPLEX/QUADRATURE WDM
     (``is_complex=True`` -- this is what the heterodyne consumes).
  2. Build h0 in real and complex WDM via the same TD->WDM transforms,
     using GBTDIonTheFly to make h0(x0) on the dense TD grid then
     ``TDSignal(...).transform(wdm_set_{real,complex})``.
  3. Build the heterodyne instance ``het = GBSignalHetWDMGetLL(...)``.
  4. Cross-check at ``x0``:
       - ``het.ref_d_d``   vs ``analysis.inner_product()``       (real WDM <d|d>)
       - ``het.ref_d_h0``  vs ``analysis.template_inner_product(h0_real)``
       - ``het.ref_h0_h0`` vs ``analysis.template_snr(h0_real)**2``
       - ``het.get_ll(x0)``    vs ``analysis.calculate_signal_likelihood(x0, source_only=True)``
  5. Sweep perturbations ``x1 = x0 + Delta`` (small enough that r stays
     smooth): compare ``het.get_ll(x1)`` to the lisatools direct
     ``calculate_signal_likelihood(x1)`` and report ``|delta_ll|``,
     ``reldiff_d_h``, ``reldiff_h_h``.
  6. Plot ``het`` vs direct ll across the sweep + per-perturbation
     residual heatmaps if requested.

Run::
    conda activate deving
    python gb_signal_het_wdm_test_script.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

from gb_signal_het_wdm import GBSignalHetWDMGetLL


# Perturbations to scan. (label, idx, delta_callable(x0)). Mostly small --
# the validation goal is to see het.get_ll(x1) track the lisatools ll within
# the relative-binning approximation error.
PERTURBATIONS = [
    ("zero",        None, lambda x0: 0.0),               # x1 = x0 sanity
    ("df0_p1e-7",   1, lambda x0: x0[1] * 1e-7),
    ("df0_p1e-6",   1, lambda x0: x0[1] * 1e-6),
    ("df0_p1e-5",   1, lambda x0: x0[1] * 1e-5),
    ("df0_p1e-4",   1, lambda x0: x0[1] * 1e-4),
    ("damp_p1e-3",  0, lambda x0: x0[0] * 1e-3),
    ("dfdot_p1",    2, lambda x0: max(abs(x0[2]), 1e-18) * 1.0),
    ("dphi0_p1e-4", 4, lambda x0: 1e-4),
    ("dphi0_p1e-3", 4, lambda x0: 1e-3),
    ("dbeta_p1e-4", 8, lambda x0: 1e-4),
    ("dlam_p1e-4",  7, lambda x0: 1e-4),
]


def build_x0(layer_df):
    """Mid-band reference GB params, matching gb_chunked_test_script.py."""
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
    """Return ``(params_9,) -> (nch, len(t_arr)) real`` TDI signal."""
    def _real_td(params9):
        amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = params9
        spline = gb_gen(
            np.array([amp]), np.array([f0]), np.array([fdot]),
            np.array([fddot]), np.array([phi0]), np.array([inc]),
            np.array([psi]), np.array([lam]), np.array([beta]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]   # (3, len(t_arr)) real
    return _real_td


def make_complex_wdm_gen(real_td_cb, td_set, wdm_set_complex, window):
    """Return ``(params_9,) -> (nch, Nf_act, Nt_active) complex`` WDM template.

    Re-runs the full dense TD -> complex-WDM transform per call.
    """
    def _gen(params9):
        td = real_td_cb(np.asarray(params9, dtype=float).reshape(9))
        return np.asarray(
            TDSignal(td, settings=td_set).transform(
                wdm_set_complex, window=window
            ).arr
        )
    return _gen


def make_real_wdm_gen(real_td_cb, td_set, wdm_set_real, window):
    """Return ``(*params_9,) -> WDMSignal`` real-WDM template.

    Used as the ``signal_gen`` for lisatools' AnalysisContainer.
    """
    def _gen(*params9):
        td = real_td_cb(np.asarray(params9, dtype=float).reshape(9))
        return TDSignal(td, settings=td_set).transform(
            wdm_set_real, window=window
        )
    return _gen


def main():
    backend = "cpu"
    plot_dir = os.environ.get("PLOT_DIR", ".")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Setup (mirrors gb_chunked_test_script.py 111-205) ----------------
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf = int(os.environ.get("NF", 1460))
    Nt = int(os.environ.get("NT", 256 * 10))
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start

    N_inj = 16384
    gb_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = np.linspace(t_arr[0], t_arr[-1], N_inj)
    gb_gen = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs,
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

    layer_dt = wdm_set_real.layer_dt
    Nf_active = wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1
    Nt_active = wdm_set_real.Nt_active

    print(f"[setup] Nf={Nf} Nt={Nt} Nf_active={Nf_active} Nt_active={Nt_active}",
          flush=True)
    print(f"[setup] layer_dt={layer_dt:.3e}s  layer_df={wdm_set_real.layer_df:.3e}Hz",
          flush=True)

    # --- Reference x0 ------------------------------------------------------
    x0 = build_x0(wdm_set_real.layer_df)
    print(f"[x0] amp={x0[0]:.3e} f0={x0[1]*1e3:.5f}mHz fdot={x0[2]:.2e} "
          f"beta={x0[8]:.3f} lam={x0[7]:.3f}", flush=True)

    # --- Inject GB and build BOTH real and complex WDM data ----------------
    print("[step] generating injection TD signal ...", flush=True)
    d_real_td = real_td_cb(x0)
    data_real_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_real, window=window,
    )
    data_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_complex, window=window,
    )
    print(f"  data REAL    WDM shape={data_real_sig.arr.shape} "
          f"dtype={data_real_sig.arr.dtype}", flush=True)
    print(f"  data COMPLEX WDM shape={data_complex_sig.arr.shape} "
          f"dtype={data_complex_sig.arr.dtype}", flush=True)

    # --- Build h0 in BOTH WDM domains via the same TD pipeline ------------
    print("[step] generating h0 = h(x0) reference in REAL and COMPLEX WDM ...",
          flush=True)
    h0_real_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_real, window=window,
    )
    h0_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_complex, window=window,
    )

    # --- lisatools AnalysisContainer on REAL WDM -- the ground truth ------
    sens_real = XYZ2SensitivityMatrix(data_real_sig.settings, model="scirdv1")
    gb_wave_wrap = make_real_wdm_gen(real_td_cb, td_set, wdm_set_real, window)
    analysis = AnalysisContainer(data_real_sig, sens_real, signal_gen=gb_wave_wrap)

    # --- invC on COMPLEX WDM (cross-channel XYZ) --------------------------
    sens_complex = XYZ2SensitivityMatrix(data_complex_sig.settings, model="scirdv1")
    invC_complex = np.asarray(sens_complex.invC)        # (3, 3, Nf_act, Nt_active)
    print(f"  invC (complex WDM) shape={invC_complex.shape} "
          f"dtype={invC_complex.dtype}", flush=True)

    # --- Build heterodyne ------------------------------------------------
    N_sparse_t = int(os.environ.get("N_SPARSE_T", Nt_active // 16))
    print(f"[step] building heterodyne: N_sparse_t={N_sparse_t}  "
          f"(mean bin width ~ {Nt_active / N_sparse_t:.1f} dense pixels)",
          flush=True)
    complex_wdm_gen = make_complex_wdm_gen(
        real_td_cb, td_set, wdm_set_complex, window,
    )
    het = GBSignalHetWDMGetLL(
        wdm_set_complex=wdm_set_complex,
        data_wdm_complex=data_complex_sig.arr,
        invC=invC_complex,
        c0_dense_complex=h0_complex_sig.arr,
        gb_complex_wdm_gen=complex_wdm_gen,
        reference_params=x0,
        N_sparse_t=N_sparse_t,
        tdi_type="XYZ",
        force_backend=backend,
    )
    print(f"[step] heterodyne built. ref_d_d={het.ref_d_d:+.6e}  "
          f"ref_d_h0={het.ref_d_h0:+.6e}  ref_h0_h0={het.ref_h0_h0:+.6e}  "
          f"ref_ll={het.ref_ll:+.6e}", flush=True)

    # --- Cross-check at x0 vs lisatools real WDM --------------------------
    dd_real = float(np.real(analysis.inner_product()))
    dh0_real = float(analysis.template_inner_product(h0_real_sig))
    h0h0_real = float(analysis.template_snr(h0_real_sig)[0]) ** 2
    ll_x0_real = float(analysis.calculate_signal_likelihood(*x0, source_only=True))

    def relerr(a, b):
        return abs(a - b) / max(abs(b), 1e-300)

    print("\n=== cross-check at x0 (heterodyne vs lisatools real-WDM) ===",
          flush=True)
    print(f"  <d|d>  : real-WDM={dd_real:+.6e}  het.ref_d_d={het.ref_d_d:+.6e}  "
          f"reldiff={relerr(het.ref_d_d, dd_real):.3e}", flush=True)
    print(f"  <d|h0> : real-WDM={dh0_real:+.6e}  het.ref_d_h0={het.ref_d_h0:+.6e}  "
          f"reldiff={relerr(het.ref_d_h0, dh0_real):.3e}", flush=True)
    print(f"  <h0|h0>: real-WDM={h0h0_real:+.6e}  het.ref_h0_h0={het.ref_h0_h0:+.6e}  "
          f"reldiff={relerr(het.ref_h0_h0, h0h0_real):.3e}", flush=True)

    ll_het_x0 = het.get_ll(x0)
    print(f"  ll(x0) : real-WDM={ll_x0_real:+.6e}  het.get_ll(x0)={ll_het_x0:+.6e}  "
          f"reldiff={relerr(ll_het_x0, ll_x0_real):.3e}", flush=True)
    print(f"  -- het.d_h_out={het.d_h_out:+.6e}  het.h_h_out={het.h_h_out:+.6e}",
          flush=True)

    # --- Perturbation sweep ----------------------------------------------
    rows = []
    print("\n=== perturbation sweep ===", flush=True)
    for label, idx, delta_fn in PERTURBATIONS:
        x1 = x0.copy()
        delta = float(delta_fn(x0))
        if idx is not None:
            x1[idx] = x0[idx] + delta
        ll_direct = float(analysis.calculate_signal_likelihood(*x1, source_only=True))
        dh_direct = float(analysis.template_inner_product(gb_wave_wrap(*x1)))
        hh_direct = float(analysis.template_snr(gb_wave_wrap(*x1))[0]) ** 2

        ll_het = het.get_ll(x1)
        dh_het = het.d_h_out
        hh_het = het.h_h_out

        rows.append((label, idx, delta, ll_direct, ll_het,
                     dh_direct, dh_het, hh_direct, hh_het))
        print(f"  [{label:<14}] delta={delta:+.3e}  "
              f"ll_direct={ll_direct:+.6e}  ll_het={ll_het:+.6e}  "
              f"|delta_ll|={abs(ll_direct - ll_het):.3e}  "
              f"reldiff_dh={relerr(dh_het, dh_direct):.3e}  "
              f"reldiff_hh={relerr(hh_het, hh_direct):.3e}", flush=True)

    # --- Plot ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    labels = [r[0] for r in rows]
    ll_dir_arr = np.array([r[3] for r in rows])
    ll_het_arr = np.array([r[4] for r in rows])
    dh_dir_arr = np.array([r[5] for r in rows])
    dh_het_arr = np.array([r[6] for r in rows])
    hh_dir_arr = np.array([r[7] for r in rows])
    hh_het_arr = np.array([r[8] for r in rows])

    xs = np.arange(len(rows))
    axs[0].plot(xs, ll_dir_arr, "o-", label="direct (real WDM)")
    axs[0].plot(xs, ll_het_arr, "x--", label="heterodyne")
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axs[0].set_title("ll")
    axs[0].grid(alpha=0.3); axs[0].legend()

    axs[1].plot(xs, dh_dir_arr, "o-", label="direct")
    axs[1].plot(xs, dh_het_arr, "x--", label="heterodyne")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axs[1].set_title("<d|h>")
    axs[1].grid(alpha=0.3); axs[1].legend()

    axs[2].plot(xs, hh_dir_arr, "o-", label="direct")
    axs[2].plot(xs, hh_het_arr, "x--", label="heterodyne")
    axs[2].set_xticks(xs)
    axs[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axs[2].set_title("<h|h>")
    axs[2].grid(alpha=0.3); axs[2].legend()
    fig.suptitle(f"GBSignalHetWDMGetLL accuracy  N_sparse_t={N_sparse_t}")
    out = os.path.join(plot_dir, "gb_signal_het_wdm_accuracy.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {out}", flush=True)

    print("\nDONE.")


if __name__ == "__main__":
    sys.exit(main())
