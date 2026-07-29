"""Follow-up to emri_maxfreq_snr_scan: null-MASKED scan + data-side check.

(a) SNR(max_freq) with the null neighborhoods masked (+/-1, +/-3, +/-5 WDM
    layers around each detected null) -> the PHYSICAL gain curve, plus the
    noise log-det bookkeeping each mask removes.
(b) Row 1 (ID 2) data-side whitened density from its local isolated L1
    stream (the only EMRI L1 file on this laptop), on the same grid/PSD ->
    how much above-25 mHz power the DATA carries vs the run's template
    (mode_selection_threshold=1e-2). Overlay plot.
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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emri_maxfreq_snr_scan import (  # noqa: E402
    C_CAP, C_NULL, C_ROW, F_MAX, F_MIN, LAYER_DF, PLOT_DIR, RUN_MAX_FREQ,
    WF_DIR, build_matrix, quad_form_density,
)

from lisatools.domains import FDSettings, TDSettings, TDSignal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("emri_null_mask")

NULLS = [0.029979, 0.059958, 0.089938, 0.119917]
SCAN_MAXF = np.array([0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120])
L1_ROW1 = os.path.expanduser(
    "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/data/EMRI/L1/"
    "EMRI_731d_2.5s_L1_source1_0_20251203T225452097594Z.h5"
)


def load_l1_xyz(h5, Nt):
    """Read the TDI-2 XYZ stream (``tdis/X2,Y2,Z2``) from a mojito L1 file."""
    return np.stack([h5[f"tdis/{c}2"][:Nt] for c in "XYZ"]).astype(np.float64)


def main():
    files = sorted(glob.glob(os.path.join(WF_DIR, "emri_row*_id*.npz")))
    data = {int(np.load(p)["row"]): np.load(p) for p in files}

    z0 = data[min(data)]
    Nt = z0["xyz"].shape[-1]
    dt = float(z0["dt"])
    fd_settings = FDSettings(
        N=Nt // 2 + 1, df=1.0 / (Nt * dt), min_freq=F_MIN, max_freq=F_MAX,
        force_backend="cpu",
    )
    log.info("building sensitivity matrix ...")
    sm = build_matrix(fd_settings, Nt * dt)
    f = np.asarray(fd_settings.f_arr)
    df = fd_settings.df

    dens = {}
    for row, z in sorted(data.items()):
        h = TDSignal(
            np.asarray(z["xyz"]), TDSettings(N=Nt, dt=dt, t0=float(z["t_start"]))
        ).transform(fd_settings)
        q, detC, good = quad_form_density(h.arr, sm, df)
        dens[row] = (q, good, int(z["cat_id"]))

    detC = None  # detC identical per row; recompute masks purely from f

    # ---- (a) masked scans ----------------------------------------------------
    print("\n===== null-MASKED SNR(max_freq) =====")
    print("mask half-widths in WDM layers (12.5 uHz each) around each null")
    for wlay in (1, 3, 5):
        mask_bad = np.zeros_like(f, dtype=bool)
        for fn in NULLS:
            mask_bad |= np.abs(f - fn) < wlay * LAYER_DF
        hdr = "  ".join(f"row{r}(ID{dens[r][2]})" for r in sorted(dens))
        print(f"\n-- mask +/-{wlay} layers --   max_freq[mHz]  {hdr}")
        for mf in SCAN_MAXF:
            sel = f < mf
            vals = []
            for r in sorted(dens):
                q, good, _ = dens[r]
                vals.append(np.sqrt(q[sel & good & ~mask_bad].sum()))
            print(f"{1e3 * mf:36.1f}  " + "  ".join(f"{v:12.3f}" for v in vals))
        nbins = int(mask_bad.sum())
        print(f"   (mask removes {nbins} bins = {nbins * df * 1e3:.4f} mHz of band)")

    # ---- (b) row-1 data-side density ----------------------------------------
    if not os.path.exists(L1_ROW1):
        log.warning("row-1 L1 file not found; skipping data-side check")
        return

    with h5py.File(L1_ROW1, "r") as h5:
        arr = load_l1_xyz(h5, Nt)
    log.info("data stream %s from tdis/X2,Y2,Z2, first %d samples", arr.shape, Nt)

    hd = TDSignal(arr, TDSettings(N=Nt, dt=dt, t0=0.0)).transform(fd_settings)
    qd, detC2, goodd = quad_form_density(hd.arr, sm, df)

    q1, good1, _ = dens[1]
    # unit/convention check: band-integrated ratio in the clean band 2-20 mHz
    clean = (f > 2e-3) & (f < 0.02)
    ratio = qd[clean & goodd].sum() / q1[clean & good1].sum()
    print(f"\ndata/template <d|d> ratio over 2-20 mHz: {ratio:.6f}")

    mask_bad = np.zeros_like(f, dtype=bool)
    for fn in NULLS:
        mask_bad |= np.abs(f - fn) < 3 * LAYER_DF

    print("\n===== row 1: DATA vs TEMPLATE band content (null-masked, scaled) ====")
    print("band[mHz]        template SNR^2   data SNR^2/ratio   data/template")
    for lo, hi in [(2, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 50),
                   (50, 59), (61, 80), (80, 89)]:
        sel = (f > lo * 1e-3) & (f < hi * 1e-3) & ~mask_bad
        t2 = q1[sel & good1].sum()
        d2 = qd[sel & goodd].sum() / ratio
        print(f"{lo:5.0f}-{hi:5.0f}      {t2:14.4e}  {d2:16.4e}  {d2 / t2 if t2 > 0 else np.inf:12.3f}")

    # overlay plot (layer-averaged densities)
    nlay = int(F_MAX / LAYER_DF)
    edges = np.arange(nlay + 1) * LAYER_DF
    cen = 0.5 * (edges[:-1] + edges[1:])
    lay_t = np.histogram(f, bins=edges, weights=np.where(good1 & ~mask_bad, q1, 0))[0]
    lay_d = np.histogram(f, bins=edges, weights=np.where(goodd & ~mask_bad, qd / ratio, 0))[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(cen * 1e3, np.maximum(lay_d, 1e-12), color="#0b0b0b", lw=1.2,
                label="mojito DATA stream (all modes)")
    ax.semilogx(cen * 1e3, np.maximum(lay_t, 1e-12), color=C_ROW[1], lw=1.2,
                label="run template (mode threshold 1e-2)")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, None)
    ax.axvline(RUN_MAX_FREQ * 1e3, color=C_CAP, ls="--", lw=1.4)
    for fn in NULLS:
        ax.axvline(fn * 1e3, color=C_NULL, ls=":", lw=1.2)
    ax.set_xlabel("frequency  [mHz]")
    ax.set_ylabel("SNR$^2$ per WDM layer (null-masked)")
    ax.set_title(
        "EMRI row 1 (ID 2): the data carries harmonic power past 25 mHz that "
        "the run template must also represent"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig6_data_vs_template_row1.png"), dpi=150)
    plt.close(fig)
    log.info("saved fig6_data_vs_template_row1.png")


if __name__ == "__main__":
    main()
