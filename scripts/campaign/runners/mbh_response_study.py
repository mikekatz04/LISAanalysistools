#!/usr/bin/env python
"""MBH null mismatch across response models and frequency bands.

Generalizes ``mbh_freq_cut_study.py`` from a two-way (full vs cut) comparison
to an arbitrary set of labeled null-check datasets — e.g. the 2x2 of
{legacy, TDI-on-the-fly} x {full band, >5e-4 Hz cut}. Each dataset's
``[RESULT] branch=mbh ...`` lines (terse or full format) are parsed, matched to
the mojito catalogue by ID, ordered by total mass, and plotted as null mismatch
vs source (one line per configuration).

This is the chain-of-custody proof for the MBH branch: it shows *which response
recovers the <5e-4 Hz content the high-mass MBHs merge into*. The legacy
Lagrange-interp response truncates that band -> its full-band mismatch climbs
with mass; a >5e-4 Hz cut (which removes the truncated band from the inner
product) or the TDI-on-the-fly response (which builds the band correctly)
should both flatten it.

  python mbh_response_study.py \
      "legacy full=/tmp/mbh_4config/legacy_full.txt" \
      "legacy >5e-4=/tmp/mbh_4config/legacy_cut.txt" \
      "TOF full=/tmp/mbh_4config/tof_full.txt" \
      "TOF >5e-4=/tmp/mbh_4config/tof_cut.txt"

Each argument is ``LABEL=PATH``; PATH is a single file or a directory of
``*.log``. Datasets whose file is missing are skipped with a note, so the same
command works incrementally as configs arrive.
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
_YR = 31558149.7635456  # YRSID_SI

_ID = re.compile(r"(?<![\w])id=(-?\d+)")
_MM = re.compile(r"mismatch=([\d.eE+-]+)")
_SNR = re.compile(r"data_snr=([\d.eE+-]+)")
_RR = re.compile(r"(?<![\w])rr=([\d.eE+-]+)")
_DD = re.compile(r"(?<![\w])dd=([\d.eE+-]+)")
_HH = re.compile(r"(?<![\w])hh=([\d.eE+-]+)")


def _rows(path):
    """id -> dict(mm, snr, rr, dd, hh) from a file or dir of null-check logs."""
    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "*.log"))))
    out = {}
    for fn in files:
        with open(fn) as f:
            for line in f:
                if "[RESULT] branch=mbh" not in line:
                    continue
                i, m = _ID.search(line), _MM.search(line)
                if not (i and m):
                    continue

                def _g(rx):
                    hit = rx.search(line)
                    return float(hit.group(1)) if hit else np.nan

                out[int(i.group(1))] = {
                    "mm": float(m.group(1)), "snr": _g(_SNR),
                    "rr": _g(_RR), "dd": _g(_DD), "hh": _g(_HH),
                }
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


def _style(label):
    """Color by response (legacy=red, TOF=blue); dash+marker by band (cut)."""
    low = label.lower()
    tof = ("tof" in low) or ("fly" in low)
    cut = any(k in low for k in ("cut", "5e-4", "5e4", ">", "clip"))
    color = "#1f77b4" if tof else "#d62728"
    return color, ("--" if cut else "-"), ("o" if cut else "s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="LABEL=PATH specs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    data = {}  # label -> {id: row}
    for spec in args.datasets:
        if "=" not in spec:
            print(f"[WARN] skip malformed spec (need LABEL=PATH): {spec}",
                  flush=True)
            continue
        label, path = spec.split("=", 1)
        label, path = label.strip(), os.path.expanduser(path.strip())
        exists = os.path.isfile(path) or (
            os.path.isdir(path) and glob.glob(os.path.join(path, "*.log")))
        if not exists:
            print(f"[WARN] dataset '{label}' has no data at {path} -- skipping",
                  flush=True)
            continue
        rows = _rows(path)
        if rows:
            data[label] = rows
            print(f"[info] '{label}': {len(rows)} sources", flush=True)

    if not data:
        print("[RESULT] study_ok=0 reason=no_datasets", flush=True)
        sys.exit(1)

    cat = _catalogue()
    all_ids = sorted(set().union(*[set(r) for r in data.values()]),
                     key=lambda i: cat.get(i, (0.0,))[0])
    x = np.arange(len(all_ids))
    mass = np.array([cat.get(i, (np.nan,))[0] for i in all_ids]) / 1e6

    # ---- plot: mismatch vs source (mass-ordered), one line per config ----
    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(all_ids) + 3), 5.8))
    for label, rows in data.items():
        c, ls, mk = _style(label)
        xs = [xi for xi, i in zip(x, all_ids) if i in rows]
        ys = [rows[i]["mm"] for i in all_ids if i in rows]
        ax.plot(xs, ys, ls=ls, marker=mk, ms=5.5, lw=1.7, color=c, label=label,
                alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("null mismatch  (1 - overlap)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"ID {i}\n{m:.1f}e6" if np.isfinite(m) else f"ID {i}"
         for i, m in zip(all_ids, mass)], fontsize=7.5)
    ax.set_xlabel("MBH source  (ordered by total mass ->)")
    ax.grid(True, which="both", axis="y", alpha=0.15)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title(
        "MBH null mismatch @ injection vs mojito data: response x frequency band\n"
        "legacy full truncates real <5e-4 Hz merger power (high mass); TOF full "
        "injects spurious <5e-4 Hz power (ids 1,2,4,16,18); >5e-4 Hz = common ground",
        fontsize=10.5)
    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = args.out or os.path.join(PLOT_DIR, "mbh_response_study.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # ---- text table (mass-ordered) ----
    labels = list(data.keys())
    hdr = "  ".join(f"{l:>13.13}" for l in labels)
    print("\n  id   Mtot/1e6   tc/yr   " + hdr)
    for i, m in zip(all_ids, mass):
        tc = cat.get(i, (np.nan, np.nan))[1]
        cells = []
        for l in labels:
            v = data[l].get(i, {}).get("mm", np.nan)
            cells.append(f"{v:>13.3e}" if np.isfinite(v) else f"{'--':>13}")
        print(f"  {i:>3} {m:>9.2f} {tc:>7.2f}   " + "  ".join(cells))

    # ---- response effect: legacy-full vs each other config, high-mass ----
    base_key = next((l for l in labels
                     if "legacy" in l.lower() and "full" in l.lower()), None)
    if base_key:
        base = data[base_key]
        print(f"\n  improvement vs '{base_key}' (mm_base / mm_cfg), "
              "high-mass ids (Mtot>7e6):")
        hi = [i for i, m in zip(all_ids, mass) if m > 7.0]
        for l in labels:
            if l == base_key:
                continue
            facs = [base[i]["mm"] / data[l][i]["mm"]
                    for i in hi if i in base and i in data[l]
                    and data[l][i]["mm"] > 0]
            if facs:
                print(f"    {l:>15}: median {np.median(facs):8.1f}x   "
                      f"min {min(facs):7.1f}x   max {max(facs):8.1f}x  "
                      f"(n={len(facs)})")

    # ---- two disjoint full-band failure populations (both full configs) ----
    lf = next((l for l in labels
               if "legacy" in l.lower() and "full" in l.lower()), None)
    tf = next((l for l in labels if ("tof" in l.lower() or "fly" in l.lower())
               and "full" in l.lower()), None)
    if lf and tf:
        trunc, excess = [], []
        for i in all_ids:
            a = data[lf].get(i, {}).get("mm")
            b = data[tf].get(i, {}).get("mm")
            if not (a and b):
                continue
            if a / b >= 5.0:       # legacy worse -> real <5e-4 power TOF recovers
                trunc.append(i)
            elif b / a >= 5.0:     # TOF worse -> spurious <5e-4 power
                excess.append(i)
        print(f"\n  legacy-truncation population (TOF full fixes, "
              f"legacy/TOF>=5x): {trunc}")
        print(f"  TOF <5e-4 excess population (TOF full worse, "
              f"TOF/legacy>=5x): {excess}")
    # common ground: worst mismatch left in each >5e-4-cut config
    for l in labels:
        if not any(k in l.lower() for k in ("cut", "5e-4", ">", "clip")):
            continue
        vals = [v["mm"] for v in data[l].values() if v.get("mm")]
        if vals:
            print(f"  {l:>15} (>5e-4): worst mm {max(vals):.2e}  n={len(vals)}")

    print(f"\n[RESULT] study_ok=1 n_datasets={len(data)} "
          f"n_sources={len(all_ids)} plot={out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[RESULT] study_ok=0 error={type(exc).__name__}", flush=True)
        sys.exit(1)
