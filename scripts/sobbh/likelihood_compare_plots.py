"""Plots for the legacy-vs-on-the-fly mismatch + logL comparison (MBH & SOBBH).

Consumes the pre-built TD TDI arrays (/tmp/{mbh,sobbh}_ll_arrays_*.npz) written
by the *_likelihood_compare.py builders, and recomputes mm + logL with ONLY the
lisatools likelihood machinery (no waveform/response generators in scope -- same
pure path as likelihood_pure.py), then draws:

  2x2 grid:  rows = {mismatch 1-|O|, |logL| = 1/2 <d-h|d-h>}
             cols = {MBHB id0, SOBHB src0}
  legacy pyResponse vs TDI-on-the-fly, log-y bars per band (scirdv1 + mrdv1).

-> /tmp/likelihood_compare.png
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

# ONLY lisatools likelihood machinery -- no generators.
from lisatools.domains import TDSettings, FDSettings, TDSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix

_FORBIDDEN = ["bbhx", "phentax", "lisatools.response.tdionfly",
              "lisatools.response.directresponse", "lisatools.sources.sobbh.waveform",
              "lisatools.sources.bbh.waveform"]
_leaked = [m for m in _FORBIDDEN if m in sys.modules]
assert not _leaked, f"generators leaked into the pure likelihood plot: {_leaked}"

SENS = ["scirdv1", "mrdv1"]
TUKEY_ALPHA = 0.1
SOURCES = [("MBHB id0", "/tmp/mbh_ll_arrays_id0.npz"),
           ("SOBHB src0", "/tmp/sobbh_ll_arrays_src0.npz")]
C_LEG, C_FLY = "#c0392b", "#2471a3"   # legacy red, on-the-fly blue


def compute(npz):
    z = np.load(npz, allow_pickle=True)
    D, A, B = np.asarray(z["D"]), np.asarray(z["A"]), np.asarray(z["B"])
    dt = float(z["dt"]); N = D.shape[1]
    bands = [(str(l), float(lo), float(hi)) for l, lo, hi in
             zip(z["band_label"], z["band_lo"], z["band_hi"])]
    win = tukey(N, TUKEY_ALPHA); td = TDSettings(N, dt, t0=0.0, force_backend="cpu")
    out = {}  # out[(sens, label)] = dict(mm_A, mm_B, ll_A, ll_B)
    for sens in SENS:
        for lbl, lo, hi in bands:
            fd = FDSettings(N=N // 2 + 1, df=1.0 / (N * dt), min_freq=lo, max_freq=hi, force_backend="cpu")
            d = TDSignal(D, td).transform(fd, window=win)
            ac = AnalysisContainer(d, XYZ2SensitivityMatrix(fd, model=sens))
            res = {}
            for key, T in [("A", A), ("B", B)]:
                t = TDSignal(T, td).transform(fd, window=win)
                O = ac.template_inner_product(t, normalize=True, complex=True)
                res[f"mm_{key}"] = 1 - abs(O)
                res[f"ll_{key}"] = abs(float(np.real(ac.template_likelihood(t))))
            out[(sens, lbl)] = res
    return bands, out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for col, (title, npz) in enumerate(SOURCES):
        if not os.path.exists(npz):
            axes[0, col].set_title(f"{title}\n(missing {npz})"); continue
        bands, out = compute(npz)
        labels = [b[0] for b in bands]; x = np.arange(len(labels)); w = 0.2

        # --- row 0: mismatch (scirdv1) ---
        ax = axes[0, col]
        mmA = [out[("scirdv1", l)]["mm_A"] for l in labels]
        mmB = [out[("scirdv1", l)]["mm_B"] for l in labels]
        b1 = ax.bar(x - 0.2, mmA, 0.38, color=C_LEG, label="legacy pyResponse")
        b2 = ax.bar(x + 0.2, mmB, 0.38, color=C_FLY, label="TDI-on-the-fly")
        ax.set_yscale("log"); ax.set_title(f"{title}: mismatch 1-|O|  (scirdv1)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("mismatch"); ax.legend(fontsize=8); ax.grid(True, which="both", axis="y", alpha=.25)
        for bars in (b1, b2):
            for r in bars:
                ax.annotate(f"{r.get_height():.1e}", (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=6.5, rotation=90)

        # --- row 1: |logL| (scirdv1 + mrdv1) ---
        ax = axes[1, col]
        groups = [("scirdv1", "A", C_LEG, 1.0, -1.5 * w, "legacy scirdv1"),
                  ("scirdv1", "B", C_FLY, 1.0, -0.5 * w, "on-fly scirdv1"),
                  ("mrdv1", "A", C_LEG, 0.5, 0.5 * w, "legacy mrdv1"),
                  ("mrdv1", "B", C_FLY, 0.5, 1.5 * w, "on-fly mrdv1")]
        for sens, key, color, alpha, off, lab in groups:
            vals = [out[(sens, l)][f"ll_{key}"] for l in labels]
            bars = ax.bar(x + off, vals, w, color=color, alpha=alpha, label=lab,
                          hatch=("" if sens == "scirdv1" else "//"))
            for r in bars:
                ax.annotate(f"{r.get_height():.1e}", (r.get_x() + r.get_width() / 2, r.get_height()),
                            ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_yscale("log"); ax.set_title(f"{title}: |logL| = 1/2 <d-h|d-h>  (lower = better)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("|logL|"); ax.legend(fontsize=7, ncol=2); ax.grid(True, which="both", axis="y", alpha=.25)

    fig.suptitle("Legacy pyResponse vs TDI-on-the-fly at injection params, vs mojito data "
                 "(pure lisatools likelihood)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_png = "/tmp/likelihood_compare.png"; fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"DONE.  plot -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
