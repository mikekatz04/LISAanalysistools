#!/usr/bin/env python
"""MBH null mismatch: full band vs a low-frequency cut (legacy phentax response).

Reads the baseline (full-band) MBH null-check log and a cut-band study log (no
recompute), pairs them per catalogue source, and plots the mismatch and SNR for
both bands ordered by total mass — the visual proof that the high-mass MBH
mismatches live below the cut frequency (the <5e-4 Hz content the Lagrange-
interp legacy response truncates; the documented fix is TDI-on-the-fly).

  python mbh_freq_cut_study.py --baseline <log> --cut-dir <dir> --fcut 5e-4
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
PLOT_DIR = os.environ.get("CAMPAIGN_PLOT_DIR", "/tmp")
_YR = 31558149.7635456

_ID = re.compile(r"(?<![\w])id=(-?\d+)")
_MM = re.compile(r"mismatch=([\d.eE+-]+)")
_SNR = re.compile(r"data_snr=([\d.eE+-]+)")


def _rows(text):
    out = {}
    for line in text.splitlines():
        if "[RESULT] branch=mbh" not in line:
            continue
        i, m, s = _ID.search(line), _MM.search(line), _SNR.search(line)
        if i and m:
            out[int(i.group(1))] = (float(m.group(1)),
                                    float(s.group(1)) if s else np.nan)
    return out


def _catalogue():
    import h5py

    base = os.environ.get(
        "MOJITO_DATA_PATH",
        os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
    )
    fs = glob.glob(os.path.join(base, "catalogues", "*mbh*"))
    if not fs:
        return {}
    with h5py.File(fs[0], "r") as f:
        b = f["Binaries"]
        ids = np.asarray(b["ID"]).astype(int)
        mt = np.asarray(b["TotalMassSSBFrame"])
        tc = np.asarray(b["TimeCoalescencePetersSSBFrame"])
    return {int(i): (float(m), float(t) / _YR) for i, m, t in zip(ids, mt, tc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--cut-dir", required=True)
    ap.add_argument("--fcut", default="5e-4")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    with open(args.baseline) as f:
        base = _rows(f.read())
    cut = {}
    for lg in glob.glob(os.path.join(args.cut_dir, "*.log")):
        with open(lg) as f:
            cut.update(_rows(f.read()))
    cat = _catalogue()

    ids = sorted(set(base) & set(cut), key=lambda i: cat.get(i, (0,))[0])
    if not ids:
        print("[RESULT] study_ok=0 reason=no_common_ids", flush=True)
        sys.exit(1)

    mass = np.array([cat.get(i, (np.nan,))[0] for i in ids]) / 1e6
    mm_b = np.array([base[i][0] for i in ids])
    mm_c = np.array([cut[i][0] for i in ids])
    snr_b = np.array([base[i][1] for i in ids])
    snr_c = np.array([cut[i][1] for i in ids])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(max(8, 1.2 * len(ids) + 3), 7.2),
        gridspec_kw={"height_ratios": [2, 1]}, sharex=True,
    )
    x = np.arange(len(ids))
    w = 0.4
    ax.bar(x - w / 2, mm_b, w, color="#d03b3b", label="full band (legacy)", zorder=3)
    ax.bar(x + w / 2, mm_c, w, color="#1baf7a",
           label=f"> {args.fcut} Hz cut", zorder=3)
    for xi, b, c in zip(x, mm_b, mm_c):
        ax.annotate(f"{b / c:.0f}×" if c > 0 and b / c >= 1.5 else "≈",
                    (xi, max(b, c)), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8, color="#0b0b0b")
    ax.set_yscale("log")
    ax.set_ylabel("null mismatch  (1 - overlap)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.15)
    ax.set_title(
        f"MBH null mismatch: full band vs >{args.fcut} Hz cut (legacy phentax)\n"
        "high-mass MBHs merge low-f; their mismatch lives in the <"
        f"{args.fcut} Hz band the legacy response truncates",
        fontsize=11,
    )

    ax2.bar(x - w / 2, snr_b, w, color="#d03b3b", zorder=3)
    ax2.bar(x + w / 2, snr_c, w, color="#1baf7a", zorder=3)
    ax2.set_ylabel("data SNR")
    ax2.grid(True, axis="y", alpha=0.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"ID {i}\n$M_{{tot}}$ {m:.1f}e6" for i, m in zip(ids, mass)], fontsize=8
    )

    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "mbh_freq_cut_study.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    worst_gain = float(np.max(mm_b / np.clip(mm_c, 1e-30, None)))
    print(f"[RESULT] study_ok=1 n={len(ids)} max_improvement={worst_gain:.0f}x "
          f"plot={out}", flush=True)


if __name__ == "__main__":
    main()
