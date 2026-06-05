#!/usr/bin/env python
"""
Speed + accuracy sweeps for GBSignalHetWDMGetLL.

Three blocks, all on the same x0 / data setup:

  (1) tdi_type parity: XYZ (cross-channel invC) vs AET (diagonal invC).
      Confirm the AET-mode het tracks lisatools' AET-mode direct ll to
      similar accuracy as XYZ.

  (2) N_sparse_t sweep at XYZ: N_sparse_t in {78, 157, 314, 628}. Confirm
      the linearization error scales like 1/N_sparse_t**2 -- doubling
      bins cuts the residual ~4x.

  (3) Per-call timing benchmark: time get_ll() in steady state for the
      v1 implementation (dense regen each call). This is the speed floor;
      v2 sparse-only generator removes the dense transform from the
      inner loop.

Outputs:
  gb_signal_het_wdm_speed_accuracy.png  -- summary plots
  gb_signal_het_wdm_speed_accuracy.npz  -- raw arrays
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

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import GBTDIonTheFly

from gb_signal_het_wdm import GBSignalHetWDMGetLL


# Fisher-1-sigma per-parameter sigmas. Same as gb_signal_het_wdm_scatter_test.py.
SIGMAS = {
    0: ("rel", 4.0e-2),     # amp
    1: ("rel", 4.0e-8),     # f0
    2: ("abs", 1.1e-17),    # fdot
    3: ("abs", 0.0),        # fddot
    4: ("abs", 2.0e-2),     # phi0
    5: ("abs", 2.0e-2),     # inc
    6: ("abs", 2.0e-2),     # psi
    7: ("abs", 2.7e-3),     # lam
    8: ("abs", 1.4e-2),     # beta
}


def build_x0(layer_df):
    m_ref = int(3e-3 / layer_df)
    return np.array([
        1.0e-22,
        (m_ref + 0.5) * layer_df,
        1e-17, 0.0,
        2.09802430298, 0.23984234, 1.234019814, 4.09808143,
        0.04,
    ], dtype=float)


def draw_delta(rng, x0):
    delta = np.zeros(9)
    for idx, (kind, sig) in SIGMAS.items():
        if sig == 0:
            continue
        if kind == "rel":
            delta[idx] = rng.normal(0.0, sig * abs(x0[idx]))
        else:
            delta[idx] = rng.normal(0.0, sig)
    return delta


def make_real_td_callable(gb_gen, t_arr):
    def _cb(p):
        spline = gb_gen(
            np.array([p[0]]), np.array([p[1]]), np.array([p[2]]),
            np.array([p[3]]), np.array([p[4]]), np.array([p[5]]),
            np.array([p[6]]), np.array([p[7]]), np.array([p[8]]),
            convert_to_ra_dec=False, return_spline=True,
        )
        return np.asarray(spline.eval_tdi(t_arr))[0]
    return _cb


def make_complex_wdm_gen(real_td_cb, td_set, wdm_set_c, window):
    def _g(p):
        td = real_td_cb(np.asarray(p, dtype=float).reshape(9))
        return np.asarray(
            TDSignal(td, settings=td_set).transform(wdm_set_c, window=window).arr
        )
    return _g


def make_real_wdm_gen(real_td_cb, td_set, wdm_set_r, window):
    def _g(*p):
        td = real_td_cb(np.asarray(p, dtype=float).reshape(9))
        return TDSignal(td, settings=td_set).transform(wdm_set_r, window=window)
    return _g


def build_diag_invC_from_xyz(sens_real):
    """Diagonal AET-style invC = 1 / diag(Sigma_XYZ).

    NOTE: this is the chunked-het convention -- we keep XYZ signals but
    use the per-channel diagonal of the XYZ Sigma matrix as the invC.
    The cross-channel terms are dropped. This is what
    ``GBSignalHetWDMGetLL(tdi_type="AET")`` consumes.
    """
    sm = np.asarray(sens_real.sens_mat)
    if sm.ndim == 4:
        diag = np.stack([sm[c, c] for c in range(sm.shape[0])], axis=0)
    else:
        diag = sm
    with np.errstate(divide="ignore", invalid="ignore"):
        invC_diag = 1.0 / np.where(
            np.isfinite(diag) & (diag > 0), diag, np.inf,
        )
    return np.where(np.isfinite(invC_diag), invC_diag, 0.0)


def run_block_scatter(
    label,
    n_samples,
    rng,
    x0,
    real_td_cb,
    td_set,
    wdm_set_r,
    window,
    wdm_set_c,
    sens_real,
    data_real_sig,
    data_complex_sig,
    h0_complex_sig,
    sens_complex,
    sens_complex_diag_invC,
    tdi_type,
    N_sparse_t,
    backend,
):
    """One run of N random samples for the given (tdi_type, N_sparse_t).

    For tdi_type='AET' we use the diagonal-of-XYZ invC convention --
    same data signals, channels treated as uncorrelated. This is the
    same approximation the chunked-het test uses.
    """
    complex_wdm_gen = make_complex_wdm_gen(
        real_td_cb, td_set, wdm_set_c, window,
    )
    if tdi_type == "XYZ":
        invC = np.asarray(sens_complex.invC)
    else:
        invC = sens_complex_diag_invC

    het = GBSignalHetWDMGetLL(
        wdm_set_complex=wdm_set_c,
        data_wdm_complex=data_complex_sig.arr,
        invC=invC,
        c0_dense_complex=h0_complex_sig.arr,
        gb_complex_wdm_gen=complex_wdm_gen,
        reference_params=x0,
        N_sparse_t=N_sparse_t,
        tdi_type=tdi_type,
        force_backend=backend,
    )

    # For the lisatools direct ll we always use the cross-channel XYZ
    # IP (the AET diagonal direct ll would need a different sens matrix
    # to be apples-to-apples; here we report the XYZ direct as the ground
    # truth and the het reports its own approximation against the same
    # data with diagonal invC -- so the AET het reldiff should NOT match
    # XYZ direct exactly, only the AET direct).
    # Build an AET-direct ll evaluator too, for fairness.
    gb_wave_wrap = make_real_wdm_gen(real_td_cb, td_set, wdm_set_r, window)
    analysis_xyz = AnalysisContainer(data_real_sig, sens_real, signal_gen=gb_wave_wrap)
    # AET-direct: same data, same signal generator, but build a diag-only sens.
    # XYZ2SensitivityMatrix.sens_mat with the off-diagonals zeroed.
    if tdi_type == "AET":
        sens_real_diag = XYZ2SensitivityMatrix(
            data_real_sig.settings, model="scirdv1",
        )
        sm = sens_real_diag.sens_mat
        # zero off-diagonals in place
        nch = sm.shape[0]
        sm_diag = np.zeros_like(sm)
        for c in range(nch):
            sm_diag[c, c] = sm[c, c]
        sens_real_diag.sens_mat = sm_diag
        # rebuild invC for the diag version (let SensitivityMatrix recompute)
        try:
            sens_real_diag._invC_dirty = True
        except Exception:
            pass
        analysis_direct = AnalysisContainer(
            data_real_sig, sens_real_diag, signal_gen=gb_wave_wrap,
        )
    else:
        analysis_direct = analysis_xyz

    print(f"  [{label}] sampling N={n_samples} ...", flush=True)
    ll_direct = np.zeros(n_samples)
    ll_het = np.zeros(n_samples)
    abs_resid = np.zeros(n_samples)
    t0 = time.perf_counter()
    for i in range(n_samples):
        x1 = x0 + draw_delta(rng, x0)
        ll_direct[i] = float(
            analysis_direct.calculate_signal_likelihood(*x1, source_only=True),
        )
        ll_het[i] = het.get_ll(x1)
        abs_resid[i] = abs(ll_het[i] - ll_direct[i])
    elapsed = time.perf_counter() - t0
    resid = ll_het - ll_direct
    print(
        f"    n_samples={n_samples}  elapsed={elapsed:.1f}s  "
        f"({n_samples/elapsed:.2f} samp/s)  |Δll|: "
        f"median={np.median(abs_resid):.3e}  max={abs_resid.max():.3e}  "
        f"RMS={np.sqrt((resid**2).mean()):.3e}",
        flush=True,
    )
    return dict(
        label=label,
        tdi_type=tdi_type,
        N_sparse_t=N_sparse_t,
        n_samples=n_samples,
        elapsed_s=elapsed,
        ll_direct=ll_direct,
        ll_het=ll_het,
        abs_resid=abs_resid,
        median_abs_resid=float(np.median(abs_resid)),
        max_abs_resid=float(abs_resid.max()),
        rms_resid=float(np.sqrt((resid ** 2).mean())),
        het_instance=het,
    )


def benchmark_get_ll(het, x0, rng, n_calls):
    """Steady-state per-call timing of het.get_ll. Warm up first."""
    # warm up
    for _ in range(3):
        het.get_ll(x0 + draw_delta(rng, x0))
    times = []
    for _ in range(n_calls):
        x1 = x0 + draw_delta(rng, x0)
        t0 = time.perf_counter()
        het.get_ll(x1)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def main():
    backend = "cpu"
    plot_dir = os.environ.get("PLOT_DIR", ".")
    os.makedirs(plot_dir, exist_ok=True)

    n_samp = int(os.environ.get("N_SAMPLES", "12"))
    rng = np.random.default_rng(int(os.environ.get("SEED", "2")))

    # --- common setup ---
    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0
    Nf, Nt = 1460, 256 * 10
    wavelet_duration = Nf * dt
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    t_start = int(0.5 * YRSID_SI / dt) * dt
    t_arr = np.arange(Nobs) * dt + t_start
    t_ref = t_start
    t_tdi_inj = np.linspace(t_arr[0], t_arr[-1], 16384)
    gb_gen = GBTDIonTheFly(
        t_tdi_inj, Tobs, t_ref, 1.0 / dt, 1,
        tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
        force_backend=backend,
    )
    real_td_cb = make_real_td_callable(gb_gen, t_arr)

    td_set = TDSettings(Nobs, dt, force_backend=backend)
    window = np.ones(Nobs)
    min_freq, max_freq = 1e-4, 35e-3
    EDGE_CUT = 20
    min_time = EDGE_CUT * wavelet_duration
    max_time = (Nt - EDGE_CUT) * wavelet_duration
    wdm_set_r = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=False, force_backend=backend,
    )
    wdm_set_c = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        is_complex=True, force_backend=backend,
    )
    Nt_active = wdm_set_r.Nt_active

    x0 = build_x0(wdm_set_r.layer_df)
    d_real_td = real_td_cb(x0)
    data_real_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_r, window=window,
    )
    data_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_c, window=window,
    )
    h0_complex_sig = TDSignal(d_real_td, settings=td_set).transform(
        wdm_set_c, window=window,
    )
    sens_real = XYZ2SensitivityMatrix(data_real_sig.settings, model="scirdv1")
    sens_complex = XYZ2SensitivityMatrix(data_complex_sig.settings, model="scirdv1")
    sens_complex_diag_invC = build_diag_invC_from_xyz(sens_complex)

    print(f"[setup] Nt_active={Nt_active}  Nf_active=", flush=True)
    print(f"[setup] n_samp per config = {n_samp}", flush=True)

    # --- block 1: tdi_type parity at N_sparse_t = 157 ---
    print("\n=== Block 1: XYZ vs AET parity at N_sparse_t=157 ===", flush=True)
    block1 = []
    for tdi in ("XYZ", "AET"):
        rng_local = np.random.default_rng(7)   # same seed both modes
        r = run_block_scatter(
            f"tdi={tdi}, N=157", n_samp, rng_local, x0, real_td_cb, td_set,
            wdm_set_r, window, wdm_set_c, sens_real,
            data_real_sig, data_complex_sig, h0_complex_sig,
            sens_complex, sens_complex_diag_invC,
            tdi_type=tdi, N_sparse_t=157, backend=backend,
        )
        block1.append(r)

    # --- block 2: N_sparse_t sweep in XYZ ---
    print("\n=== Block 2: N_sparse_t sweep (XYZ) ===", flush=True)
    block2 = []
    for Nb in (78, 157, 314, 628):
        rng_local = np.random.default_rng(7)
        r = run_block_scatter(
            f"tdi=XYZ, N={Nb}", n_samp, rng_local, x0, real_td_cb, td_set,
            wdm_set_r, window, wdm_set_c, sens_real,
            data_real_sig, data_complex_sig, h0_complex_sig,
            sens_complex, sens_complex_diag_invC,
            tdi_type="XYZ", N_sparse_t=Nb, backend=backend,
        )
        block2.append(r)

    # --- block 3: timing benchmark ---
    print("\n=== Block 3: per-call timing benchmark ===", flush=True)
    het_for_bench = block2[1]["het_instance"]  # XYZ, N_sparse_t=157
    rng_local = np.random.default_rng(11)
    times = benchmark_get_ll(het_for_bench, x0, rng_local, n_calls=15)
    print(
        f"  v1 get_ll() at XYZ, N_sparse_t=157, N=15 calls: "
        f"median={np.median(times)*1e3:.1f} ms  "
        f"mean={times.mean()*1e3:.1f} ms  "
        f"min={times.min()*1e3:.1f} ms  max={times.max()*1e3:.1f} ms",
        flush=True,
    )
    # Coarsely: how much is the dense WDM transform?
    # We time TDSignal(td).transform(wdm_set_c) by itself.
    n_bench_xform = 5
    t_xform = []
    for _ in range(n_bench_xform):
        x1 = x0 + draw_delta(rng_local, x0)
        td = real_td_cb(x1)
        t0 = time.perf_counter()
        _ = TDSignal(td, settings=td_set).transform(wdm_set_c, window=window).arr
        t_xform.append(time.perf_counter() - t0)
    t_xform = np.array(t_xform)
    print(
        f"  dense TD->complex-WDM transform (per call, N={n_bench_xform}): "
        f"median={np.median(t_xform)*1e3:.1f} ms  "
        f"min={t_xform.min()*1e3:.1f} ms",
        flush=True,
    )
    print(
        f"  -> dense transform is ~{100*np.median(t_xform)/np.median(times):.0f}% "
        f"of get_ll() runtime; remaining is r/dr + A/B reductions.",
        flush=True,
    )

    # --- save raw -----------------------------------------------------
    save = dict()
    for tag, rs in (("block1_tdi_parity", block1), ("block2_nsparse_sweep", block2)):
        for r in rs:
            key_prefix = f"{tag}::{r['label'].replace(' ', '').replace(',', '_').replace('=', '')}"
            save[f"{key_prefix}__ll_direct"] = r["ll_direct"]
            save[f"{key_prefix}__ll_het"] = r["ll_het"]
            save[f"{key_prefix}__abs_resid"] = r["abs_resid"]
            save[f"{key_prefix}__median_abs_resid"] = r["median_abs_resid"]
            save[f"{key_prefix}__max_abs_resid"] = r["max_abs_resid"]
            save[f"{key_prefix}__rms_resid"] = r["rms_resid"]
            save[f"{key_prefix}__elapsed_s"] = r["elapsed_s"]
    save["bench_get_ll_times_s"] = times
    save["bench_wdm_xform_times_s"] = t_xform
    out_npz = os.path.join(plot_dir, "gb_signal_het_wdm_speed_accuracy.npz")
    np.savez_compressed(out_npz, **save)
    print(f"\n[save] {out_npz}", flush=True)

    # --- plot --------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # (1) tdi parity: scatter ll_direct vs ll_het for XYZ and AET
    for r in block1:
        color = "tab:blue" if r["tdi_type"] == "XYZ" else "tab:orange"
        axs[0].scatter(r["ll_direct"], r["ll_het"], s=14, alpha=0.7,
                       color=color, label=r["label"])
    lo = min(r["ll_direct"].min() for r in block1)
    hi = max(r["ll_direct"].max() for r in block1)
    axs[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axs[0].set_xlabel("ll_direct")
    axs[0].set_ylabel("ll_het")
    axs[0].set_title("XYZ vs AET parity (N_sparse_t=157)")
    axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)

    # (2) N_sparse_t sweep: median / max abs_resid vs N_sparse_t
    Nbs = np.array([r["N_sparse_t"] for r in block2])
    meds = np.array([r["median_abs_resid"] for r in block2])
    maxs = np.array([r["max_abs_resid"] for r in block2])
    rmss = np.array([r["rms_resid"] for r in block2])
    axs[1].loglog(Nbs, meds, "o-", label="median |Δll|")
    axs[1].loglog(Nbs, maxs, "s-", label="max |Δll|")
    axs[1].loglog(Nbs, rmss, "^-", label="RMS Δll")
    # reference 1/N^2 slope
    ref = meds[0] * (Nbs[0] / Nbs) ** 2
    axs[1].loglog(Nbs, ref, "k--", lw=0.6, label=r"~$1/N_{sparse}^2$ (ref)")
    axs[1].set_xlabel("N_sparse_t")
    axs[1].set_ylabel("|Δll|")
    axs[1].set_title("Linearization error vs sparse bin count")
    axs[1].grid(alpha=0.3, which="both")
    axs[1].legend(fontsize=8)

    # (3) timing
    axs[2].hist(times * 1e3, bins=15, color="tab:blue", alpha=0.8,
                label=f"get_ll v1 (median {np.median(times)*1e3:.0f} ms)")
    axs[2].axvline(np.median(t_xform) * 1e3, color="tab:red", lw=1.5,
                    ls="--", label=f"dense TD→cWDM xform (median {np.median(t_xform)*1e3:.0f} ms)")
    axs[2].set_xlabel("ms per call")
    axs[2].set_ylabel("count")
    axs[2].set_title(f"v1 per-call timing (N_sparse_t=157)")
    axs[2].grid(alpha=0.3); axs[2].legend(fontsize=9)

    fig.suptitle("GBSignalHetWDMGetLL  speed + accuracy", fontsize=12)
    out_png = os.path.join(plot_dir, "gb_signal_het_wdm_speed_accuracy.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)

    # --- final summary table -----------------------------------------
    print("\n=== Summary ===", flush=True)
    print("--- Block 1: XYZ vs AET parity ---", flush=True)
    for r in block1:
        print(f"  {r['label']:<24}  median|Δll|={r['median_abs_resid']:.3e}  "
              f"max|Δll|={r['max_abs_resid']:.3e}  "
              f"RMS={r['rms_resid']:.3e}", flush=True)
    print("--- Block 2: N_sparse_t sweep (XYZ) ---", flush=True)
    for r in block2:
        print(f"  N_sparse_t={r['N_sparse_t']:>4}  "
              f"median|Δll|={r['median_abs_resid']:.3e}  "
              f"max|Δll|={r['max_abs_resid']:.3e}  "
              f"RMS={r['rms_resid']:.3e}", flush=True)
    print(f"--- Block 3: timing (v1, XYZ, N_sparse_t=157) ---", flush=True)
    print(f"  get_ll() median = {np.median(times)*1e3:.1f} ms;  "
          f"dense TD→cWDM = {np.median(t_xform)*1e3:.1f} ms  "
          f"({100*np.median(t_xform)/np.median(times):.0f}% of get_ll)",
          flush=True)
    print("\nDONE.")


if __name__ == "__main__":
    sys.exit(main())
