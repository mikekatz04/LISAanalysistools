#!/usr/bin/env python
"""GB sparse-spline density study.

For a grid of (f0, Tobs) and source draws, compute the noise-weighted mismatch
between

  * a reference injection generated with GBTDIonTheFly using a *dense* spline
    of N_ref >= 16384 time knots, and

  * a template generated with GBTDIonTheFly using the *same* parameters but a
    *sparse* spline of N_sparse knots.

Both signals are evaluated on the same dense time grid and the mismatch is
computed in the frequency domain with lisatools (FDSettings + XYZ2SensitivityMatrix,
the same inner-product machinery used elsewhere in this repo).

The metric reported is "points per year" = N_sparse / Tobs_yr, so that runs at
different Tobs can be compared on a common axis. This is what tells you the
minimum spline density needed to keep the waveform accurate.

Usage:
    python gb_spline_density_test.py                  # full sweep, saves npz + png
    N_REF=16384 python gb_spline_density_test.py      # override reference knot count
    BACKEND=cpu python gb_spline_density_test.py      # force cpu (default auto-detects)
"""

from __future__ import annotations

import os
import sys
import time
import json
import warnings
import argparse
import numpy as np
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# Sensitivity at f=0 produces inf/nan in 1/f terms; lisatools' inner_product
# handles the first bin via an `ind_start` skip. Silence the cosmetic spam.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="divide by zero encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered")

try:
    import cupy as cp  # noqa: F401
    _HAS_CUPY = True
except (ImportError, ModuleNotFoundError):
    _HAS_CUPY = False

from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings
from lisatools.diagnostic import inner_product as _inner_product

from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly


# ----------------------------- configuration ------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        default=os.environ.get("BACKEND", "cpu"),
        choices=["cpu", "gpu", "cuda11x", "cuda12x"],
    )
    p.add_argument("--n-ref", type=int, default=int(os.environ.get("N_REF", 16384)),
                   help="Reference (dense) spline knot count. >=16384.")
    p.add_argument("--dt", type=float, default=float(os.environ.get("DT", 15.0)),
                   help="Output cadence (s) for dense TD evaluation grid.")
    p.add_argument("--tdi-gen", default=os.environ.get("TDI_GEN", "2nd generation"))
    p.add_argument("--out-prefix", default=os.environ.get("OUT_PREFIX", "gb_spline_density"))
    p.add_argument("--tukey-alpha", type=float, default=0.05)
    p.add_argument("--no-verify-sources", action="store_true",
                   help="Skip the multi-source verification pass.")
    return p.parse_args()


# (f0 [Hz], description)
F0_VALUES = [
    0.5e-3,
    1.0e-3,
    3.0e-3,
    6.0e-3,
    10.0e-3,
    18.0e-3,
    25.0e-3,
]

# Observation times in years.
TOBS_YR_VALUES = [0.25, 0.5, 1.0, 2.0]

# Sparse spline knot counts to sweep against the dense reference.
N_SPARSE_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Mismatch thresholds we report the minimum (pts/yr) for.
MISMATCH_THRESHOLDS = [1e-2, 1e-3, 1e-4, 1e-5]


# fiducial source (matches existing repo scripts). f0 is overridden per run.
FIDUCIAL = dict(
    amp=8.0e-22,
    fdot=1e-14,
    fddot=0.0,
    phi0=2.09802430298,
    inc=0.23984234,
    psi=1.234019814,
    lam=4.09808143,
    beta=0.04,
)


# ----------------------------- core routines ------------------------------- #


def make_generator(t_tdi: np.ndarray, Tobs: float, t_ref: float, dt: float,
                   gb_tdi_kwargs: dict) -> GBTDIonTheFly:
    return GBTDIonTheFly(
        t_tdi, Tobs, t_ref, 1.0 / dt, 1, **gb_tdi_kwargs
    )


def generate_signal(gen: GBTDIonTheFly, params: dict, f0: float,
                    t_eval: np.ndarray, xp) -> np.ndarray:
    """Return TD signal sampled at t_eval, shape (3, N_eval)."""
    out = gen(
        np.array([params["amp"]]),
        np.array([f0]),
        np.array([params["fdot"]]),
        np.array([params["fddot"]]),
        np.array([params["phi0"]]),
        np.array([params["inc"]]),
        np.array([params["psi"]]),
        np.array([params["lam"]]),
        np.array([params["beta"]]),
        convert_to_ra_dec=False,
        return_spline=True,
    )
    sig = out.eval_tdi(t_eval)
    sig = xp.asarray(sig)
    if sig.ndim == 3:
        sig = sig[0]  # (3, N_eval)
    return sig


def mismatch_fd(sig_d: np.ndarray, sig_h: np.ndarray, td_set: TDSettings,
                window: np.ndarray, sens_kwargs: dict) -> tuple[float, float, float]:
    """Return (mismatch, opt_snr_d, opt_snr_h).

    mismatch = 1 - Re(<d|h>) / sqrt(<d|d> <h|h>) using XYZ2SensitivityMatrix.
    """
    fd_d = TDSignal(sig_d, settings=td_set).fft(window=window)
    fd_h = TDSignal(sig_h, settings=td_set).fft(window=window)

    # match arrays to one FDSettings object (they share df/N by construction)
    inj = DataResidualArray(fd_d)
    tmpl = DataResidualArray(fd_h)
    sens = XYZ2SensitivityMatrix(inj.data_res_arr.settings, **sens_kwargs)

    ac = AnalysisContainer(inj, sens)
    dd = float(ac.inner_product().real)
    hh = float(np.real(_inner_product(tmpl, tmpl, psd=sens)))
    dh = float(np.real(ac.template_inner_product(tmpl)))
    mismatch = 1.0 - dh / np.sqrt(dd * hh)
    return float(mismatch), float(np.sqrt(dd)), float(np.sqrt(hh))


def run_one(
    f0: float, Tobs_yr: float, params: dict, dt: float,
    n_ref: int, n_sparse_values: list[int],
    gb_tdi_kwargs: dict, tukey_alpha: float, sens_kwargs: dict,
    xp,
) -> dict:
    """Return dict with arrays of N_sparse, mismatch, snr_h, snr_d."""
    Tobs = Tobs_yr * YRSID_SI
    # Round N to multiple of 2 so the FFT is fast.
    N = int(np.floor(Tobs / dt))
    N -= N % 2
    Tobs = N * dt  # actual span used
    t_start = 0.0
    t_ref = t_start
    t_eval = xp.arange(N) * dt + t_start
    window = xp.asarray(sp_signal.windows.tukey(N, alpha=tukey_alpha))
    td_set = TDSettings(N, dt, force_backend=gb_tdi_kwargs.get("force_backend", "cpu"))

    # Reference (dense) injection
    t_tdi_ref = xp.linspace(t_eval[0], t_eval[-1], n_ref)
    gen_ref = make_generator(t_tdi_ref, Tobs, t_ref, dt, gb_tdi_kwargs)
    t0 = time.perf_counter()
    sig_ref = generate_signal(gen_ref, params, f0, t_eval, xp)
    t_ref_gen = time.perf_counter() - t0

    mismatches = []
    snrs_h = []
    snr_d = None
    times = []
    for n_sparse in n_sparse_values:
        t_tdi_sp = xp.linspace(t_eval[0], t_eval[-1], int(n_sparse))
        gen_sp = make_generator(t_tdi_sp, Tobs, t_ref, dt, gb_tdi_kwargs)
        t0 = time.perf_counter()
        sig_sp = generate_signal(gen_sp, params, f0, t_eval, xp)
        mm, snr_dh, snr_hh = mismatch_fd(sig_ref, sig_sp, td_set, window, sens_kwargs)
        dt_run = time.perf_counter() - t0
        snr_d = snr_dh  # same each loop
        mismatches.append(mm)
        snrs_h.append(snr_hh)
        times.append(dt_run)

    return dict(
        f0=f0,
        Tobs_yr=Tobs_yr,
        N=N,
        n_ref=n_ref,
        n_sparse=np.asarray(n_sparse_values, dtype=int),
        mismatch=np.asarray(mismatches),
        snr_h=np.asarray(snrs_h),
        snr_d=float(snr_d) if snr_d is not None else np.nan,
        t_gen_ref=t_ref_gen,
        t_per_sparse=np.asarray(times),
    )


# ----------------------------- analysis helpers ---------------------------- #


def min_n_per_yr(result: dict, threshold: float) -> float:
    """Return smallest pts/yr where mismatch <= threshold; nan if never reached."""
    n_sparse = result["n_sparse"]
    mm = result["mismatch"]
    Tobs_yr = result["Tobs_yr"]
    ok = np.where(mm <= threshold)[0]
    if len(ok) == 0:
        return float("nan")
    return float(n_sparse[ok[0]] / Tobs_yr)


def plot_mismatch_vs_n_per_yr(results: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    by_f0: dict[float, list[dict]] = {}
    for r in results:
        by_f0.setdefault(r["f0"], []).append(r)

    cmap = plt.get_cmap("viridis")
    f0_keys = sorted(by_f0.keys())
    for fi, f0 in enumerate(f0_keys):
        color = cmap(fi / max(1, len(f0_keys) - 1))
        for r in by_f0[f0]:
            n_per_yr = r["n_sparse"] / r["Tobs_yr"]
            label = f"f0={f0 * 1e3:.2f} mHz, Tobs={r['Tobs_yr']:.2f} yr"
            ax.loglog(n_per_yr, r["mismatch"], marker="o", color=color,
                      alpha=0.7, label=label)

    for thr in MISMATCH_THRESHOLDS:
        ax.axhline(thr, color="grey", lw=0.7, ls="--")
        ax.text(ax.get_xlim()[1], thr, f"  {thr:g}", color="grey",
                fontsize=8, va="center")

    ax.set_xlabel("Spline knots per year  (N_sparse / Tobs_yr)")
    ax.set_ylabel("Noise-weighted mismatch  (1 - <d|h>/sqrt(<d|d><h|h>))")
    ax.set_title("GB sparse-spline accuracy vs density (XYZ2 sensitivity)")
    ax.grid(True, which="both", alpha=0.3)
    # collapse the legend
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        ax.legend(loc="lower left", fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_summary(results: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        fh.write("# Mismatch vs N_sparse for GBTDIonTheFly (XYZ TDI, XYZ2 sensitivity)\n")
        fh.write("# Reference uses N_REF spline knots; sparse uses N_sparse knots, same params.\n\n")
        for r in results:
            fh.write(f"== f0={r['f0']*1e3:.3f} mHz, Tobs={r['Tobs_yr']:.3f} yr, "
                     f"N_ref={r['n_ref']}, dense_N={r['N']}, snr_ref={r['snr_d']:.3e}\n")
            fh.write(f"   {'N_sparse':>10s} {'pts/yr':>10s} {'mismatch':>14s} {'snr_h':>12s} {'t[s]':>8s}\n")
            for n, mm, sh, t in zip(r["n_sparse"], r["mismatch"], r["snr_h"], r["t_per_sparse"]):
                fh.write(f"   {int(n):10d} {n / r['Tobs_yr']:10.1f} {mm:14.4e} {sh:12.4e} {t:8.2f}\n")
            for thr in MISMATCH_THRESHOLDS:
                n_yr = min_n_per_yr(r, thr)
                fh.write(f"   min pts/yr for mismatch<={thr:g}: {n_yr}\n")
            fh.write("\n")


# ----------------------------- main entrypoint ----------------------------- #


def main() -> int:
    args = parse_args()
    backend = args.backend
    if backend != "cpu" and not _HAS_CUPY:
        print(f"[warn] backend={backend} requested but cupy missing; falling back to cpu")
        backend = "cpu"
    xp = np if backend == "cpu" else cp

    if args.n_ref < 16384:
        print(f"[warn] N_REF={args.n_ref} < 16384; clamping to 16384")
        args.n_ref = 16384

    orbits = EqualArmlengthOrbits(force_backend=backend)
    orbits.configure(linear_interp_setup=True)
    tdi_config = TDIConfig(args.tdi_gen, force_backend=backend)
    gb_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    sens_kwargs = dict(model="scirdv1")

    print(f"[setup] backend={backend} N_ref={args.n_ref} dt={args.dt}s tdi={args.tdi_gen}", flush=True)
    print(f"[setup] f0 grid (mHz): {[f * 1e3 for f in F0_VALUES]}")
    print(f"[setup] Tobs grid (yr): {TOBS_YR_VALUES}")
    print(f"[setup] N_sparse: {N_SPARSE_VALUES}\n", flush=True)

    # main sweep: fiducial source at each (f0, Tobs)
    results: list[dict] = []
    for f0 in F0_VALUES:
        for Tobs_yr in TOBS_YR_VALUES:
            t0 = time.perf_counter()
            r = run_one(
                f0=f0, Tobs_yr=Tobs_yr, params=FIDUCIAL, dt=args.dt,
                n_ref=args.n_ref, n_sparse_values=N_SPARSE_VALUES,
                gb_tdi_kwargs=gb_tdi_kwargs, tukey_alpha=args.tukey_alpha,
                sens_kwargs=sens_kwargs, xp=xp,
            )
            results.append(r)
            print(f"[run] f0={f0*1e3:6.3f} mHz Tobs={Tobs_yr:4.2f} yr "
                  f"snr_d={r['snr_d']:.3e} "
                  f"mm(min)={r['mismatch'].min():.2e} "
                  f"mm(max)={r['mismatch'].max():.2e} "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            for thr in (1e-3, 1e-5):
                print(f"        threshold {thr:g}: min pts/yr = {min_n_per_yr(r, thr)}")

    write_summary(results, args.out_prefix + "_summary.txt")
    plot_mismatch_vs_n_per_yr(results, args.out_prefix + "_mismatch.png")
    np.savez(args.out_prefix + "_main.npz",
             f0=np.array([r["f0"] for r in results]),
             Tobs_yr=np.array([r["Tobs_yr"] for r in results]),
             n_sparse=np.array(N_SPARSE_VALUES),
             mismatch=np.array([r["mismatch"] for r in results]),
             snr_h=np.array([r["snr_h"] for r in results]),
             snr_d=np.array([r["snr_d"] for r in results]),
             n_ref=args.n_ref, dt=args.dt)

    # verification pass: same (f0, Tobs) sampled with different sky/inc/psi
    if not args.no_verify_sources:
        print("\n[verify] checking parameter sensitivity at f0=3 mHz, Tobs=1 yr ...", flush=True)
        rng = np.random.default_rng(20260517)
        verify_results: list[dict] = []
        n_draws = 4
        f0_v = 3.0e-3
        Tobs_v = 1.0
        for k in range(n_draws):
            p = dict(FIDUCIAL)
            p["inc"] = float(np.arccos(rng.uniform(-1.0, 1.0)))
            p["psi"] = float(rng.uniform(0.0, np.pi))
            p["lam"] = float(rng.uniform(0.0, 2 * np.pi))
            p["beta"] = float(np.arcsin(rng.uniform(-1.0, 1.0)))
            p["phi0"] = float(rng.uniform(0.0, 2 * np.pi))
            p["fdot"] = float(rng.uniform(1e-17, 1e-13))
            t0 = time.perf_counter()
            r = run_one(
                f0=f0_v, Tobs_yr=Tobs_v, params=p, dt=args.dt,
                n_ref=args.n_ref, n_sparse_values=N_SPARSE_VALUES,
                gb_tdi_kwargs=gb_tdi_kwargs, tukey_alpha=args.tukey_alpha,
                sens_kwargs=sens_kwargs, xp=xp,
            )
            r["draw"] = k
            r["params"] = p
            verify_results.append(r)
            print(f"  draw {k}: snr={r['snr_d']:.2e} mm(min)={r['mismatch'].min():.2e} "
                  f"mm(max)={r['mismatch'].max():.2e} ({time.perf_counter()-t0:.1f}s)",
                  flush=True)

        # plot verification
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for r in verify_results:
            n_per_yr = r["n_sparse"] / r["Tobs_yr"]
            ax.loglog(n_per_yr, r["mismatch"], marker="o", alpha=0.7,
                      label=f"draw {r['draw']}")
        for thr in MISMATCH_THRESHOLDS:
            ax.axhline(thr, color="grey", lw=0.7, ls="--")
        ax.set_xlabel("Spline knots per year (N_sparse / Tobs_yr)")
        ax.set_ylabel("Noise-weighted mismatch")
        ax.set_title(f"Verification: f0={f0_v*1e3:.2f} mHz, Tobs={Tobs_v} yr, random sky/inc/psi/phi0/fdot")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(args.out_prefix + "_verify.png", dpi=130)
        plt.close(fig)
        np.savez(args.out_prefix + "_verify.npz",
                 mismatch=np.array([r["mismatch"] for r in verify_results]),
                 n_sparse=np.array(N_SPARSE_VALUES),
                 params=json.dumps([r["params"] for r in verify_results]),
                 snr_d=np.array([r["snr_d"] for r in verify_results]),
                 f0=f0_v, Tobs_yr=Tobs_v, n_ref=args.n_ref, dt=args.dt)

    # final summary table to stdout
    print("\n==== minimum pts/yr to reach mismatch threshold ====")
    header = "  f0[mHz]   Tobs[yr]  " + "  ".join(f"<={thr:g}" for thr in MISMATCH_THRESHOLDS)
    print(header)
    for r in results:
        row = f"  {r['f0']*1e3:7.3f}   {r['Tobs_yr']:8.3f}  "
        row += "  ".join(f"{min_n_per_yr(r, thr):10.1f}" for thr in MISMATCH_THRESHOLDS)
        print(row)

    print(f"\nSaved: {args.out_prefix}_summary.txt, "
          f"{args.out_prefix}_mismatch.png, {args.out_prefix}_main.npz")
    if not args.no_verify_sources:
        print(f"       {args.out_prefix}_verify.png, {args.out_prefix}_verify.npz")

    return 0


if __name__ == "__main__":
    sys.exit(main())
