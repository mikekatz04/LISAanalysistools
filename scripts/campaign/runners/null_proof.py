#!/usr/bin/env python
"""Proof plot for a source ground-truth (null-check) gate.

Reads the gate's already-captured null-check log (no recompute) and shows the
RAW physics at the injection point: the residual power ``<r|r>`` per source and
the noiseless log-likelihood at injection ``logL = -0.5 <r|r>`` (the source
term only, no noise/PSD-determinant term) — the absolute residual the stock
template leaves against the mojito data, NOT normalized by ``<d|d>``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

PLOT_DIR = os.environ.get("CAMPAIGN_PLOT_DIR", "/tmp")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# raw <r|r> (not rr_over_dd), source-only logL, data SNR, catalogue id, mismatch.
_RR = re.compile(r"(?<![\w])rr=([\d.eE+-]+)")
_SLL = re.compile(r"source_logL=(-?[\d.eE+-]+)")
_SNR = re.compile(r"data_snr=([\d.eE+-]+)")
_ID = re.compile(r"(?<![\w])id=(-?\d+)")
_MM = re.compile(r"mismatch=([\d.eE+-]+)")
_TOBS = re.compile(r"tobs_d=([\d.eE+-]+)")

_YR = 31558149.7635456  # YRSID_SI

# per-class mojito catalogue file glob + (total-mass, coalescence-time) fields.
_CAT = {
    "mbh": ("*mbh*", "TotalMassSSBFrame", "TimeCoalescencePetersSSBFrame"),
}


def _catalogue_lookup(branch):
    """id -> (total_mass_Msun, merger_time_yr) from the mojito catalogue,
    for source classes where it applies (MBH). Empty dict otherwise."""
    spec = _CAT.get(branch)
    if spec is None:
        return {}
    import glob as _glob

    import h5py

    base = os.environ.get(
        "MOJITO_DATA_PATH",
        os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"),
    )
    files = _glob.glob(os.path.join(base, "catalogues", spec[0]))
    if not files:
        return {}
    with h5py.File(files[0], "r") as f:
        b = f["Binaries"]
        ids = np.asarray(b["ID"]).astype(int)
        mtot = np.asarray(b[spec[1]])
        tc = np.asarray(b[spec[2]])
    return {int(i): (float(m), float(t) / _YR) for i, m, t in zip(ids, mtot, tc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)  # mbh | emri | sobbh
    ap.add_argument("--gate", required=True)     # t1-gt-<branch>
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    log = os.path.join(REPO, "gf_output", "campaign", args.gate, "null-check.log")
    if not os.path.exists(log):
        print(f"[RESULT] null_proof=SKIP reason=no_log path={log}", flush=True)
        return
    with open(log) as f:
        text = f.read()

    rows = []
    for line in text.splitlines():
        if "[RESULT]" not in line or "rr=" not in line:
            continue
        rrm = _RR.search(line)
        if not rrm:
            continue
        rr = float(rrm.group(1))
        sllm, snrm = _SLL.search(line), _SNR.search(line)
        idm, mmm, tbm = _ID.search(line), _MM.search(line), _TOBS.search(line)
        rows.append({
            "rr": rr,
            "logL": float(sllm.group(1)) if sllm else -0.5 * rr,
            "snr": float(snrm.group(1)) if snrm else np.nan,
            "id": int(idm.group(1)) if idm else -1,
            "mm": float(mmm.group(1)) if mmm else np.nan,
            "tobs": float(tbm.group(1)) if tbm else np.nan,
        })
    if not rows:
        print("[RESULT] null_proof=SKIP reason=no_rows", flush=True)
        return

    cls = {"mbh": "MBH", "emri": "EMRI", "sobbh": "SOBBH"}.get(args.branch, "MBH")
    cat = _catalogue_lookup(args.branch)  # id -> (Mtot_Msun, tc_yr)
    rows.sort(key=lambda r: r["rr"], reverse=True)

    def _fmt(v):  # decimals when O(1)+, scientific when tiny — readable for both
        return f"{v:.2f}" if abs(v) >= 0.005 else f"{v:.1e}"

    n = len(rows)
    rr = np.array([r["rr"] for r in rows])
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * n + 2.5), 5.4))
    x = np.arange(n)
    ax.bar(x, rr, color="#2a78d6", zorder=3, width=0.68)
    # per-bar annotation: data SNR + mismatch + noiseless logL at injection
    for xi, r in zip(x, rows):
        snr = f"SNR {r['snr']:.0f}\n" if np.isfinite(r["snr"]) else ""
        mm = f"mm {r['mm']:.1e}\n" if np.isfinite(r["mm"]) else ""
        ax.annotate(f"{snr}{mm}logL {_fmt(r['logL'])}", (xi, r["rr"]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=7, color="#52514e")
    # multi-line x labels: catalogue ID + total mass / merger time (MBH)
    labels = []
    for r in rows:
        parts = [f"ID {r['id']}" if r["id"] >= 0 else "ID ?"]
        if r["id"] in cat:
            mtot, tc = cat[r["id"]]
            parts.append(f"$M_{{tot}}$ {mtot / 1e6:.2f}e6")
            parts.append(f"$t_c$ {tc:.2f} yr")
        labels.append("\n".join(parts))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel(r"raw $\langle r|r\rangle$ at injection")
    tobs = np.nanmedian([r["tobs"] for r in rows])
    win = f"  ·  Tobs {tobs:.0f} d ({tobs / 30.44:.1f} mo)" if np.isfinite(tobs) else ""
    ax.set_title(
        f"{cls} noiseless null at injection  (logL = -0.5 <r|r>){win}\n"
        f"worst <r|r> {rr.max():.2e}   logL {_fmt(-0.5 * rr.max())}   "
        f"{n} source(s)",
        fontsize=11,
    )
    ax.grid(True, which="both", axis="y", alpha=0.15)
    ax.margins(y=0.18)  # headroom for the annotations
    fig.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"{args.branch}_null_residual.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[RESULT] null_proof=ok null_proof_ok=1 worst_rr={rr.max():.6e} "
          f"worst_logL_noiseless={(-0.5 * rr.max()):.6e} "
          f"n_sources={n} plot={out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[RESULT] null_proof=FAIL error={type(exc).__name__}", flush=True)
        sys.exit(1)
