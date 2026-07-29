"""SNR / likelihood-bookkeeping vs WDM ``max_freq`` for the two run EMRIs.

Consumes the stored long TD templates from ``emri_long_waveform_gen.py`` and
the run's OWN sensitivity recipe (full_year_combined ``finalize_general``:
InstrumentNoise with the fitted LISAModel + FittedHyperbolicTangentGalactic-
Foreground, TDI-2 XYZ), evaluated on the natural rFFT grid through the
installed ``CompositeSensitivityMatrix``. The per-bin quadratic form is
cross-validated against ``AnalysisContainer.inner_product`` (the installed
likelihood primitive) before any derived quantity is trusted.

Outputs (gf_output/emri_long_waveforms/plots/):
  fig1_psd_nulls.png      XX PSD + per-bin log10 det C, nulls marked
  fig2_snr_density_rowN   whitened SNR^2 density + cumulative SNR per source
  fig3_snr_vs_maxfreq.png SNR(max_freq) scan, both sources
  fig4_null_zoom.png      28-32 mHz zoom: weights, det C, WDM layer grid
  fig5_spectrogram_rowN   TD spectrograms with 25 mHz + null lines
plus stdout tables: SNR(max_freq), null locations, per-layer fold near the
first null, masked-vs-unmasked bookkeeping.
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import glob
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.detector import DefaultOrbits, LISAModel
from lisatools.domains import FDSettings, TDSettings, TDSignal
from lisatools.sensitivity import (
    CompositeSensitivityMatrix,
    GalacticForeground,
    InstrumentNoise,
)
from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground
from lisatools.utils.constants import YRSID_SI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("emri_maxfreq_scan")

HERE = os.path.dirname(os.path.abspath(__file__))
WF_DIR = os.path.join(HERE, "..", "..", "gf_output", "emri_long_waveforms")
PLOT_DIR = os.path.join(WF_DIR, "plots")

# run-matched constants (full_year_combined defaults / run_settings.log)
NOISE_SOMS_D = 1.496182116469066e-11
NOISE_SA_A = 2.9824117392856982e-15
LAYER_DF = 1.25e-5  # WDM layer spacing of the run (general_setup.log)
RUN_MAX_FREQ = 0.025
F_MIN, F_MAX = 1e-4, 0.15
SCAN_MAXF = np.array([0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120])

# dataviz reference palette
C_ROW = {0: "#2a78d6", 1: "#eb6834"}  # categorical slots 1/2 (fixed by entity)
C_NULL = "#e34948"  # serious: the divergent-weight frequencies
C_CAP = "#52514e"  # secondary ink: the current 25 mHz cap
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "lines.linewidth": 1.8,
})


def build_matrix(fd_settings, Tobs):
    comps = [
        InstrumentNoise(
            tdi_generation=2,
            model=LISAModel(
                NOISE_SOMS_D ** 2, NOISE_SA_A ** 2, DefaultOrbits(),
                "full_year_fixed_noise",
            ),
            fill_nans=0.0,
        ),
        # stationary isotropic limit of the run's annually-modulated galfor
        GalacticForeground(
            foreground_params=(Tobs,),
            modulation=None,
            tdi_generation=2,
            stochastic_fn=FittedHyperbolicTangentGalacticForeground,
        ),
    ]
    return CompositeSensitivityMatrix(fd_settings, comps)


def quad_form_density(h_arr, sm, df):
    """Per-bin <d|d> integrand 4*df*Re[h^H C^-1 h] via the matrix's own inverse."""
    S = np.moveaxis(np.asarray(sm.sens_mat), -1, 0)  # (nfreq, 3, 3)
    detC = np.linalg.det(S)
    good = np.isfinite(detC) & (detC > 0)
    invC = np.zeros_like(S)
    invC[good] = np.linalg.inv(S[good])
    h = np.moveaxis(np.asarray(h_arr), -1, 0)  # (nfreq, 3)
    q = 4.0 * df * np.real(np.einsum("fi,fij,fj->f", h.conj(), invC, h))
    return q, detC, good


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(WF_DIR, "emri_row*_id*.npz")))
    assert files, f"no stored waveforms under {WF_DIR}; run emri_long_waveform_gen.py"

    results = {}
    fd_settings = None
    sm = None

    for path in files:
        z = np.load(path)
        xyz, dt = z["xyz"], float(z["dt"])
        row, cat_id = int(z["row"]), int(z["cat_id"])
        Nt = xyz.shape[-1]
        Tobs = Nt * dt

        if fd_settings is None:
            fd_settings = FDSettings(
                N=Nt // 2 + 1, df=1.0 / (Nt * dt),
                min_freq=F_MIN, max_freq=F_MAX, force_backend="cpu",
            )
            log.info("building sensitivity matrix on %d active bins ...",
                     fd_settings.basis_shape_active[0])
            sm = build_matrix(fd_settings, Tobs)

        log.info("row %d (ID %d): TD->FD transform ...", row, cat_id)
        h = TDSignal(xyz, TDSettings(N=Nt, dt=dt, t0=float(z["t_start"]))).transform(
            fd_settings
        )
        f = np.asarray(fd_settings.f_arr)
        df = fd_settings.df

        q, detC, good = quad_form_density(h.arr, sm, df)

        # cross-check the density against the installed inner product
        ac = AnalysisContainer(data=h, sens_mat=sm)
        ip_installed = float(np.real(ac.inner_product()))
        ratio = q[good].sum() / ip_installed if ip_installed else np.nan
        log.info(
            "row %d: quad-form total %.6e vs AnalysisContainer.inner_product %.6e "
            "(ratio %.6f)", row, q[good].sum(), ip_installed, ratio,
        )

        csum = np.cumsum(np.where(good, q, 0.0))
        snr_scan = np.array([np.sqrt(csum[np.searchsorted(f, mf) - 1]) for mf in SCAN_MAXF])

        results[row] = dict(
            f=f, q=q, detC=detC, good=good, csum=csum, snr_scan=snr_scan,
            cat_id=cat_id, xyz=xyz, dt=dt, Tobs=Tobs,
            t_plunge=float(z["t_plunge_ssb"]),
            snr_cat=float(z["estimated_snr_catalogue"]),
            ip_installed=ip_installed, ratio=ratio,
        )

    f = results[min(results)]["f"]
    detC = results[min(results)]["detC"]
    good = results[min(results)]["good"]
    SXX = np.asarray(sm.sens_mat)[0, 0]

    # ---- locate the sensitivity nulls numerically (det C minima, 20-130 mHz)
    nulls = []
    band = (f > 0.02) & (f < 0.13)
    logdet = np.where(detC > 0, np.log10(np.maximum(detC, 1e-300)), np.nan)
    fb, db = f[band], logdet[band]
    for k in range(2, len(fb) - 2):
        if db[k] == np.nanmin(db[max(0, k - 2000):k + 2000]) and db[k] < np.nanmedian(db) - 3:
            if not nulls or fb[k] - nulls[-1] > 5e-3:
                nulls.append(fb[k])
    log.info("detected sensitivity nulls (det C minima): %s mHz",
             [f"{1e3 * x:.3f}" for x in nulls])

    # ---- stdout tables -------------------------------------------------------
    print("\n===== SNR(max_freq) scan =====")
    hdr = "max_freq[mHz]  " + "  ".join(
        f"row{r}(ID{results[r]['cat_id']})" for r in sorted(results)
    )
    print(hdr)
    for i, mf in enumerate(SCAN_MAXF):
        vals = "  ".join(f"{results[r]['snr_scan'][i]:12.3f}" for r in sorted(results))
        print(f"{1e3 * mf:12.1f}  {vals}")
    for r in sorted(results):
        R = results[r]
        print(
            f"row {r} (ID {R['cat_id']}): catalogue EstimatedSNR={R['snr_cat']:.2f}, "
            f"full-band(<{F_MAX * 1e3:.0f} mHz) SNR={np.sqrt(R['csum'][-1]):.2f}, "
            f"installed-inner-product cross-check ratio={R['ratio']:.6f}"
        )

    # near-null bookkeeping: SNR^2 and log det C in +/- 3 layers around nulls
    print("\n===== null-neighborhood bookkeeping =====")
    for r in sorted(results):
        R = results[r]
        for fn in nulls:
            m = np.abs(f - fn) < 3 * LAYER_DF
            frac = R["q"][m & good].sum() / R["csum"][-1]
            print(
                f"row {r}: null {1e3 * fn:7.3f} mHz  SNR^2 in +/-3 layers: "
                f"{R['q'][m & good].sum():.4e} ({100 * frac:.4f}% of total)  "
                f"min detC {detC[m & good].min():.3e}  "
                f"sum ln detC {np.log(detC[m & good]).sum():.1f} over {m.sum()} bins"
            )

    # per-layer fold around the first null
    print("\n===== WDM layer fold around first null =====")
    fn = nulls[0]
    lay0 = int(fn / LAYER_DF)
    print("layer  f_lo[mHz]  f_hi[mHz]  folded<SXX>      min SXX       "
          "sum lndetC   SNR2(row0)   SNR2(row1)")
    for lay in range(lay0 - 5, lay0 + 6):
        sel = (f >= lay * LAYER_DF) & (f < (lay + 1) * LAYER_DF)
        if not sel.any():
            continue
        r0 = results.get(0)
        r1 = results.get(1)
        print(
            f"{lay:5d}  {1e3 * lay * LAYER_DF:9.4f}  {1e3 * (lay + 1) * LAYER_DF:9.4f}  "
            f"{SXX[sel].mean():.4e}  {SXX[sel].min():.4e}  "
            f"{np.log(np.maximum(detC[sel], 1e-300)).sum():11.1f}  "
            f"{(r0['q'][sel & good].sum() if r0 else np.nan):.4e}  "
            f"{(r1['q'][sel & good].sum() if r1 else np.nan):.4e}"
        )

    # ---- fig 1: PSD + detC with nulls ---------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    ax = axes[0]
    ax.loglog(f * 1e3, SXX, color="#4a3aa7", lw=1.6, label="run PSD  S_XX(f)")
    ax.set_ylabel("S_XX  [1/Hz]")
    ax.set_title(
        "The run's TDI-2 XYZ sensitivity has transfer-function nulls — "
        "the first sits just above the 25 mHz cap"
    )
    ax2 = axes[1]
    ax2.semilogx(f * 1e3, logdet, color=INK2, lw=1.2, label="log10 det C(f)")
    ax2.set_ylabel("log10 det C")
    ax2.set_xlabel("frequency  [mHz]")
    for a in axes:
        a.axvline(RUN_MAX_FREQ * 1e3, color=C_CAP, ls="--", lw=1.4)
        for fn_ in nulls:
            a.axvline(fn_ * 1e3, color=C_NULL, ls=":", lw=1.4)
    axes[0].text(RUN_MAX_FREQ * 1e3 * 1.02, np.nanmax(SXX) * 0.1, "25 mHz cap",
                 color=C_CAP, fontsize=9, rotation=90, va="top")
    for fn_ in nulls[:2]:
        axes[0].text(fn_ * 1e3 * 1.01, np.nanmax(SXX) * 0.1, f"null {1e3 * fn_:.1f}",
                     color=C_NULL, fontsize=9, rotation=90, va="top")
    ax.legend(frameon=False)
    ax2.legend(frameon=False)
    ax.set_xlim(0.2, 150)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig1_psd_nulls.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2 per source: density + cumulative (stacked, one axis each) ----
    for r in sorted(results):
        R = results[r]
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
        w = np.where(R["good"], R["q"], np.nan)
        # layer-averaged density for readability (full grid is 7M bins)
        nlay = int(F_MAX / LAYER_DF)
        lay_edges = np.arange(nlay + 1) * LAYER_DF
        lay_q = np.histogram(f, bins=lay_edges, weights=np.nan_to_num(w))[0]
        lay_c = 0.5 * (lay_edges[:-1] + lay_edges[1:])
        axes[0].semilogx(lay_c * 1e3, lay_q, color=C_ROW[r], lw=1.2)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("SNR$^2$ per WDM layer")
        axes[0].set_title(
            f"EMRI row {r} (catalogue ID {R['cat_id']}): where the SNR lives "
            "vs the 25 mHz cap and the nulls"
        )
        axes[1].semilogx(f * 1e3, np.sqrt(R["csum"]), color=C_ROW[r])
        axes[1].set_ylabel("cumulative SNR(<f)")
        axes[1].set_xlabel("frequency  [mHz]")
        for a in axes:
            a.axvline(RUN_MAX_FREQ * 1e3, color=C_CAP, ls="--", lw=1.4)
            for fn_ in nulls:
                a.axvline(fn_ * 1e3, color=C_NULL, ls=":", lw=1.2)
        snr25 = np.sqrt(R["csum"][np.searchsorted(f, RUN_MAX_FREQ) - 1])
        snrF = np.sqrt(R["csum"][-1])
        axes[1].text(
            0.02, 0.95,
            f"SNR(<25 mHz) = {snr25:.2f}\nSNR(<{F_MAX * 1e3:.0f} mHz) = {snrF:.2f}\n"
            f"catalogue EstimatedSNR = {R['snr_cat']:.1f}",
            transform=axes[1].transAxes, va="top", fontsize=9, color=INK,
        )
        axes[1].set_xlim(0.2, 150)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"fig2_snr_density_row{r}.png"), dpi=150)
        plt.close(fig)

    # ---- fig 3: SNR vs max_freq scan ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in sorted(results):
        R = results[r]
        ax.plot(SCAN_MAXF * 1e3, R["snr_scan"], "-o", color=C_ROW[r], ms=5,
                label=f"row {r} (ID {R['cat_id']})")
        ax.annotate(f"{R['snr_scan'][1]:.1f}", (25, R["snr_scan"][1]),
                    textcoords="offset points", xytext=(6, -12), fontsize=9,
                    color=C_ROW[r])
    ax.axvline(RUN_MAX_FREQ * 1e3, color=C_CAP, ls="--", lw=1.4)
    for fn_ in nulls:
        ax.axvline(fn_ * 1e3, color=C_NULL, ls=":", lw=1.2)
    ax.set_xlabel("WDM max_freq  [mHz]")
    ax.set_ylabel("SNR(< max_freq)")
    ax.set_title("Raising max_freq: what each EMRI gains past the 25 mHz cap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig3_snr_vs_maxfreq.png"), dpi=150)
    plt.close(fig)

    # ---- fig 4: null zoom with WDM layer grid -------------------------------
    fn = nulls[0]
    zoom = (f > fn - 20 * LAYER_DF) & (f < fn + 20 * LAYER_DF)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    axes[0].semilogy(f[zoom] * 1e3, SXX[zoom], color="#4a3aa7", lw=1.4,
                     label="S_XX(f)")
    axes[0].set_ylabel("S_XX  [1/Hz]")
    axes[0].set_title(
        f"First TDI-2 null at {1e3 * fn:.3f} mHz vs the WDM layer grid "
        f"(layer_df = {LAYER_DF * 1e6:.1f} µHz)"
    )
    axes[1].semilogy(f[zoom] * 1e3, np.maximum(detC[zoom], 1e-300), color=INK2,
                     lw=1.2, label="det C(f)")
    axes[1].set_ylabel("det C")
    axes[1].set_xlabel("frequency  [mHz]")
    lay_lo = int((fn - 20 * LAYER_DF) / LAYER_DF)
    for a in axes:
        for k in range(lay_lo, lay_lo + 42):
            a.axvline(k * LAYER_DF * 1e3, color=INK2, alpha=0.18, lw=0.6)
        a.axvline(fn * 1e3, color=C_NULL, ls=":", lw=1.5)
    for a in axes:
        a.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig4_null_zoom.png"), dpi=150)
    plt.close(fig)

    # ---- fig 5: spectrograms -------------------------------------------------
    for r in sorted(results):
        R = results[r]
        X = R["xyz"][0]
        fs = 1.0 / R["dt"]
        nper = 1 << 17  # ~3.8 days
        ff, tt, Sxx = spectrogram(X, fs=fs, nperseg=nper, noverlap=nper // 2)
        keep = ff < 0.12
        fig, ax = plt.subplots(figsize=(9, 5))
        pw = 10 * np.log10(np.maximum(Sxx[keep], 1e-60))
        vmax = np.percentile(pw, 99.5)
        im = ax.pcolormesh(tt / YRSID_SI, ff[keep] * 1e3, pw, cmap="magma",
                           vmin=vmax - 90, vmax=vmax, shading="auto")
        fig.colorbar(im, ax=ax, label="10 log10 |X(t,f)|^2  [dB]")
        ax.axhline(RUN_MAX_FREQ * 1e3, color="#7fd4ff", ls="--", lw=1.4)
        for fn_ in nulls:
            ax.axhline(fn_ * 1e3, color="#ff9d9d", ls=":", lw=1.2)
        ax.axvline(R["t_plunge"] / YRSID_SI, color="w", ls="-", lw=0.8, alpha=0.6)
        ax.text(R["t_plunge"] / YRSID_SI, 0.112 * 1e3, " plunge", color="w",
                fontsize=9)
        ax.set_xlabel("time  [yr]")
        ax.set_ylabel("frequency  [mHz]")
        ax.set_title(
            f"EMRI row {r} (ID {R['cat_id']}): harmonic tracks vs the 25 mHz "
            "cap (dashed) and the nulls (dotted)"
        )
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"fig5_spectrogram_row{r}.png"), dpi=150)
        plt.close(fig)

    log.info("plots -> %s", PLOT_DIR)


if __name__ == "__main__":
    main()
