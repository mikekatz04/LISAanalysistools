"""SNR vs WDM max_freq for the stored long MBH templates (rows 0/16/18).

Same machinery as scripts/emri/emri_maxfreq_snr_scan.py (imported): the run's
composite XYZ TDI-2 sensitivity on the natural rFFT grid, per-bin quadratic
form cross-validated against AnalysisContainer.inner_product, unmasked AND
null-masked (+/-3 layers) scans, layer tables, figures.
"""

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import glob
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "emri"))
from emri_maxfreq_snr_scan import (  # noqa: E402
    C_CAP, C_NULL, F_MAX, F_MIN, LAYER_DF, RUN_MAX_FREQ, SURFACE, INK, INK2,
    build_matrix, quad_form_density,
)

from lisatools.analysiscontainer import AnalysisContainer  # noqa: E402
from lisatools.domains import FDSettings, TDSettings, TDSignal  # noqa: E402
from lisatools.utils.constants import YRSID_SI  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("mbh_maxfreq_scan")

WF_DIR = os.path.join(HERE, "..", "..", "gf_output", "mbh_long_waveforms")
PLOT_DIR = os.path.join(WF_DIR, "plots")
NULLS = [0.029979, 0.059958, 0.089938, 0.119917]
SCAN_MAXF = np.array([0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050,
                      0.060, 0.080, 0.100, 0.120])
C_MBH = {0: "#2a78d6", 16: "#1baf7a", 18: "#4a3aa7"}


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(WF_DIR, "mbh_row*_id*.npz")))
    assert files, f"no stored MBH waveforms under {WF_DIR}"

    fd_settings = None
    sm = None
    results = {}
    for path in files:
        z = np.load(path)
        xyz, dt = z["xyz"], float(z["dt"])
        row = int(z["row"])
        Nt = xyz.shape[-1]
        if fd_settings is None:
            fd_settings = FDSettings(
                N=Nt // 2 + 1, df=1.0 / (Nt * dt), min_freq=F_MIN,
                max_freq=F_MAX, force_backend="cpu",
            )
            log.info("building sensitivity matrix ...")
            sm = build_matrix(fd_settings, Nt * dt)
        h = TDSignal(
            np.asarray(xyz), TDSettings(N=Nt, dt=dt, t0=float(z["t_start"]))
        ).transform(fd_settings)
        f = np.asarray(fd_settings.f_arr)
        df = fd_settings.df
        q, detC, good = quad_form_density(h.arr, sm, df)
        ac = AnalysisContainer(data=h, sens_mat=sm)
        ip = float(np.real(ac.inner_product()))
        log.info("row %d: quad total %.6e vs installed %.6e (ratio %.6f)",
                 row, q[good].sum(), ip, q[good].sum() / ip)
        results[row] = dict(
            f=f, q=q, good=good, xyz=xyz, dt=dt,
            t_plunge=float(z["t_plunge_ssb"]),
            snr_cat=float(z["estimated_snr_catalogue"]),
        )

    f = results[min(results)]["f"]
    mask_bad = np.zeros_like(f, dtype=bool)
    for fn in NULLS:
        mask_bad |= np.abs(f - fn) < 3 * LAYER_DF

    print("\n===== MBH SNR(max_freq): unmasked | null-masked (+/-3 layers) ====")
    hdr = "   ".join(f"row{r:<2d} unm | msk" for r in sorted(results))
    print(f"max_freq[mHz]   {hdr}")
    for mf in SCAN_MAXF:
        sel = f < mf
        cells = []
        for r in sorted(results):
            q, good = results[r]["q"], results[r]["good"]
            unm = np.sqrt(q[sel & good].sum())
            msk = np.sqrt(q[sel & good & ~mask_bad].sum())
            cells.append(f"{unm:9.2f} | {msk:9.2f}")
        print(f"{1e3 * mf:12.1f}   " + "   ".join(cells))
    for r in sorted(results):
        print(f"row {r}: catalogue EstimatedSNR = {results[r]['snr_cat']:.1f} "
              f"(2-yr full mission)")

    # per-null bookkeeping
    print("\n===== null neighborhoods (masked bins' content per source) =====")
    for r in sorted(results):
        q, good = results[r]["q"], results[r]["good"]
        tot = q[good & ~mask_bad].sum()
        for fn in NULLS[:2]:
            m = np.abs(f - fn) < 3 * LAYER_DF
            print(f"row {r}: null {1e3 * fn:6.3f} mHz  masked SNR^2 "
                  f"{q[m & good].sum():.4e}  vs clean total {tot:.4e}")

    # figure: density + cumulative (masked) for all three, stacked panels
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
    nlay = int(F_MAX / LAYER_DF)
    edges = np.arange(nlay + 1) * LAYER_DF
    cen = 0.5 * (edges[:-1] + edges[1:])
    for r in sorted(results):
        q, good = results[r]["q"], results[r]["good"]
        w = np.where(good & ~mask_bad, q, 0.0)
        lay = np.histogram(f, bins=edges, weights=w)[0]
        axes[0].semilogx(cen * 1e3, np.maximum(lay, 1e-12), color=C_MBH[r],
                         lw=1.2, label=f"MBH row {r}")
        axes[1].semilogx(f * 1e3, np.sqrt(np.cumsum(w)), color=C_MBH[r], lw=1.8,
                         label=f"MBH row {r}")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-6, None)
    axes[0].set_ylabel("SNR$^2$ per WDM layer (null-masked)")
    axes[0].set_title(
        "MBH band content vs the 25 mHz cap: the low-mass row 0 merges "
        "through the cap; rows 16/18 live far below it"
    )
    axes[1].set_ylabel("cumulative SNR(<f), null-masked")
    axes[1].set_xlabel("frequency  [mHz]")
    for a in axes:
        a.axvline(RUN_MAX_FREQ * 1e3, color=C_CAP, ls="--", lw=1.4)
        for fn in NULLS:
            a.axvline(fn * 1e3, color=C_NULL, ls=":", lw=1.2)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set_xlim(0.2, 150)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "mbh_fig1_density_cumulative.png"), dpi=150)
    plt.close(fig)

    # spectrogram: row 0 only (the through-the-cap case)
    if 0 in results:
        R = results[0]
        X = R["xyz"][0]
        fs = 1.0 / R["dt"]
        nper = 1 << 16  # ~1.9 days
        ff, tt, Sxx = spectrogram(X, fs=fs, nperseg=nper, noverlap=nper // 2)
        keep = ff < 0.12
        fig, ax = plt.subplots(figsize=(9, 5))
        pw = 10 * np.log10(np.maximum(Sxx[keep], 1e-70))
        vmax = np.percentile(pw, 99.9)
        im = ax.pcolormesh(tt / YRSID_SI, ff[keep] * 1e3, pw, cmap="magma",
                           vmin=vmax - 100, vmax=vmax, shading="auto")
        fig.colorbar(im, ax=ax, label="10 log10 |X(t,f)|^2  [dB]")
        ax.axhline(RUN_MAX_FREQ * 1e3, color="#7fd4ff", ls="--", lw=1.4)
        for fn in NULLS:
            ax.axhline(fn * 1e3, color="#ff9d9d", ls=":", lw=1.2)
        ax.axvline(R["t_plunge"] / YRSID_SI, color="w", lw=0.8, alpha=0.6)
        ax.set_xlabel("time  [yr]")
        ax.set_ylabel("frequency  [mHz]")
        ax.set_title("MBH row 0 (Mtot 5.7e5): chirp through the 25 mHz cap and "
                     "the 30 mHz null at merger")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "mbh_fig2_spectrogram_row0.png"), dpi=150)
        plt.close(fig)

    log.info("plots -> %s", PLOT_DIR)


if __name__ == "__main__":
    main()
