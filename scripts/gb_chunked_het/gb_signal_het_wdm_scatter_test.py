#!/usr/bin/env python
"""
Random-sample accuracy scatter for GBSignalHetWDMGetLL.

Draws ``N_SAMPLES`` (default 80) random parameter perturbations
around the reference x0, with per-parameter sigmas tuned so that
the resulting samples land in the regime where the relative-binning
ansatz is expected to hold (small enough deltas for r to stay
smooth). For each sample we compute:

  - lisatools direct ll (real WDM, ``calculate_signal_likelihood``)
  - heterodyne   ll (``GBSignalHetWDMGetLL.get_ll``)

Then plot:

  - scatter (lisatools ll on x, heterodyne ll on y), with y=x diagonal
  - residual histogram (ll_het - ll_direct) and abs-residual log-scale
  - reldiff (|residual| / |ll_direct|) vs |ll_direct|

The objective is a sanity check that the heterodyne tracks lisatools
to a precision worth porting to C++/GPU. Output:

  gb_signal_het_wdm_scatter.png
  gb_signal_het_wdm_scatter.npz  (raw samples, parameter values, ll's)

Env vars:
  N_SAMPLES   number of random parameter draws (default 80)
  SIGMA_SCALE multiplicative scale for all sigmas (default 1.0;
              use 0.1 for tighter ball, 3.0 to probe convergence)
  N_SPARSE_T  bin count (default Nt_active // 16)
  SEED        rng seed (default 0)
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.detector import ESAOrbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly

from gb_signal_het_wdm import GBSignalHetWDMGetLL


# Per-parameter Fisher-1sigma scaled so each parameter at 1 sigma contributes
# ll ~ -0.5 on its own. Joint 9-parameter draw at sigma_scale=1 lands in
# ll ~ -1 to -20, i.e. the MCMC-relevant tail. Back-extracted from the 1D
# delta sweep in gb_signal_het_wdm_test_script.log: at single-param delta,
# ll ~ -0.5 * (delta / sigma)^2, so sigma_param = delta * sqrt(2 / -2 ll).
SIGMAS = {
    # idx :  (kind, sigma)   kind in {"abs", "rel"}; rel scales by |x0[idx]|
    0: ("rel", 4.0e-2),              # amp  (1 sigma in dA/A)
    1: ("rel", 4.0e-8),              # f0   (1 sigma in df0/f0; ~1.2e-10 Hz abs)
    2: ("abs", 1.1e-17),             # fdot (1 sigma absolute)
    3: ("abs", 0.0),                 # fddot fixed
    4: ("abs", 2.0e-2),              # phi0
    5: ("abs", 2.0e-2),              # inc
    6: ("abs", 2.0e-2),              # psi
    7: ("abs", 2.7e-3),              # lam
    8: ("abs", 1.4e-2),              # beta
}


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


def draw_delta(rng, x0, sigma_scale):
    """Gaussian draw of a (9,) parameter perturbation."""
    delta = np.zeros(9)
    for idx, (kind, sig) in SIGMAS.items():
        if sig == 0:
            continue
        if kind == "rel":
            delta[idx] = rng.normal(0.0, sig * sigma_scale * abs(x0[idx]))
        else:
            delta[idx] = rng.normal(0.0, sig * sigma_scale)
    return delta


def main():
    backend = "cpu"
    plot_dir = os.environ.get("PLOT_DIR", ".")
    os.makedirs(plot_dir, exist_ok=True)

    N_samples = int(os.environ.get("N_SAMPLES", "80"))
    sigma_scale = float(os.environ.get("SIGMA_SCALE", "1.0"))
    seed = int(os.environ.get("SEED", "0"))
    rng = np.random.default_rng(seed)

    # --- Setup -----------------------------------------------------------
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

    Nt_active = wdm_set_real.Nt_active

    # --- x0 + injection -------------------------------------------------
    x0 = build_x0(wdm_set_real.layer_df)
    d_real_td = real_td_cb(x0)
    data_real_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_real, window=window
    )
    data_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_complex, window=window
    )
    h0_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_complex, window=window
    )
    sens_real = XYZ2SensitivityMatrix(data_real_sig.settings, model="scirdv1")
    sens_complex = XYZ2SensitivityMatrix(data_complex_sig.settings, model="scirdv1")

    gb_wave_wrap = make_real_wdm_gen(real_td_cb, td_set, wdm_set_real, window)
    analysis = AnalysisContainer(data_real_sig, sens_real, signal_gen=gb_wave_wrap)

    complex_wdm_gen = make_complex_wdm_gen(real_td_cb, td_set, wdm_set_complex, window)
    N_sparse_t = int(os.environ.get("N_SPARSE_T", Nt_active // 16))

    print(f"[setup] Nf={Nf} Nt={Nt} Nf_active={data_real_sig.arr.shape[1]} "
          f"Nt_active={Nt_active}  N_sparse_t={N_sparse_t}", flush=True)
    print(f"[setup] N_samples={N_samples}  sigma_scale={sigma_scale}  seed={seed}",
          flush=True)

    het = GBSignalHetWDMGetLL(
        wdm_set_complex=wdm_set_complex,
        data_wdm_complex=data_complex_sig.arr,
        invC=np.asarray(sens_complex.invC),
        c0_dense_complex=h0_complex_sig.arr,
        gb_complex_wdm_gen=complex_wdm_gen,
        reference_params=x0,
        N_sparse_t=N_sparse_t,
        tdi_type="XYZ",
        force_backend=backend,
    )

    print(f"[setup] ref_ll = {het.ref_ll:+.6e}", flush=True)

    # --- Sample loop ----------------------------------------------------
    print(f"\n[run] sampling {N_samples} points ...", flush=True)
    params_log = np.zeros((N_samples, 9))
    ll_direct_log = np.zeros(N_samples)
    ll_het_log = np.zeros(N_samples)
    t0 = time.perf_counter()
    for i in range(N_samples):
        x1 = x0 + draw_delta(rng, x0, sigma_scale)
        ll_direct = float(analysis.calculate_signal_likelihood(*x1, source_only=True))
        ll_het = het.get_ll(x1)
        params_log[i] = x1
        ll_direct_log[i] = ll_direct
        ll_het_log[i] = ll_het
        if (i + 1) % 10 == 0 or i == 0:
            dt_so_far = time.perf_counter() - t0
            rate = (i + 1) / dt_so_far
            eta = (N_samples - i - 1) / rate
            print(f"  [{i+1:>3}/{N_samples}] direct={ll_direct:+.4e}  "
                  f"het={ll_het:+.4e}  |Δll|={abs(ll_het-ll_direct):.3e}  "
                  f"({rate:.2f} samp/s, ETA {eta:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"[run] done in {elapsed:.1f}s ({N_samples / elapsed:.2f} samp/s)",
          flush=True)

    resid = ll_het_log - ll_direct_log
    abs_resid = np.abs(resid)
    reldiff = abs_resid / np.maximum(np.abs(ll_direct_log), 1e-300)

    print("\n=== summary ===", flush=True)
    print(f"  |residual|: max={abs_resid.max():.3e}  median={np.median(abs_resid):.3e}  "
          f"mean={abs_resid.mean():.3e}  RMS={np.sqrt((resid**2).mean()):.3e}",
          flush=True)
    print(f"  reldiff  : max={reldiff.max():.3e}  median={np.median(reldiff):.3e}  "
          f"95th-pctile={np.percentile(reldiff, 95):.3e}",
          flush=True)
    print(f"  ll_direct range: [{ll_direct_log.min():+.3e}, {ll_direct_log.max():+.3e}]",
          flush=True)

    # --- save ---------------------------------------------------------
    out_npz = os.path.join(plot_dir, "gb_signal_het_wdm_scatter.npz")
    np.savez_compressed(
        out_npz,
        x0=x0, params=params_log,
        ll_direct=ll_direct_log, ll_het=ll_het_log,
        N_sparse_t=N_sparse_t, sigma_scale=sigma_scale,
        seed=seed, sigmas=np.array([SIGMAS[i][1] for i in range(9)]),
    )
    print(f"\n[save] {out_npz}", flush=True)

    # --- plots --------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # (1) scatter ll_direct vs ll_het with y=x line
    lo = min(ll_direct_log.min(), ll_het_log.min())
    hi = max(ll_direct_log.max(), ll_het_log.max())
    axs[0].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    axs[0].scatter(ll_direct_log, ll_het_log, s=12, alpha=0.7)
    axs[0].set_xlabel("ll (lisatools direct, real WDM)")
    axs[0].set_ylabel("ll (heterodyne, complex WDM)")
    axs[0].set_title(f"Joint accuracy  (N={N_samples}, "
                     f"sigma_scale={sigma_scale})")
    axs[0].grid(alpha=0.3)
    axs[0].legend(loc="upper left", fontsize=9)

    # (2) residual histogram (signed)
    axs[1].hist(resid, bins=40, color="tab:blue", alpha=0.8)
    axs[1].set_xlabel("ll_het - ll_direct")
    axs[1].set_ylabel("count")
    axs[1].set_title(
        f"residual: RMS={np.sqrt((resid**2).mean()):.2e}  "
        f"max|.|={abs_resid.max():.2e}"
    )
    axs[1].grid(alpha=0.3)
    axs[1].axvline(0, color="k", lw=0.5)

    # (3) reldiff vs |ll_direct| (log-log)
    axs[2].scatter(np.abs(ll_direct_log), reldiff, s=12, alpha=0.7)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("|ll_direct|")
    axs[2].set_ylabel("|ll_het - ll_direct| / |ll_direct|")
    axs[2].set_title("relative residual vs ll magnitude")
    axs[2].grid(alpha=0.3, which="both")

    fig.suptitle(
        f"GBSignalHetWDMGetLL  vs  lisatools real-WDM  "
        f"(N_sparse_t={N_sparse_t}, x_0 near 3 mHz GB)",
        fontsize=11,
    )
    out_png = os.path.join(plot_dir, "gb_signal_het_wdm_scatter.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)

    print("\nDONE.")


if __name__ == "__main__":
    sys.exit(main())
