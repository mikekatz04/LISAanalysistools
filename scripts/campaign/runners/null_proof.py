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

# raw <r|r> (not rr_over_dd), the source-only logL, and the data SNR.
_RR = re.compile(r"(?<![\w])rr=([\d.eE+-]+)")
_SLL = re.compile(r"source_logL=(-?[\d.eE+-]+)")
_SNR = re.compile(r"data_snr=([\d.eE+-]+)")


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
        rrm, sllm, snrm = _RR.search(line), _SLL.search(line), _SNR.search(line)
        if rrm:
            rr = float(rrm.group(1))
            sll = float(sllm.group(1)) if sllm else -0.5 * rr
            snr = float(snrm.group(1)) if snrm else np.nan
            rows.append((snr, rr, sll))
    if not rows:
        print("[RESULT] null_proof=SKIP reason=no_rows", flush=True)
        return

    cls = {"mbh": "MBH", "emri": "EMRI", "sobbh": "SOBBH"}.get(args.branch, "MBH")
    snrs = np.array([r[0] for r in rows])
    rr = np.array([r[1] for r in rows])
    sll = np.array([r[2] for r in rows])
    order = np.argsort(rr)[::-1]
    rr, snrs, sll = rr[order], snrs[order], sll[order]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rr) + 3), 4.6))
    x = np.arange(len(rr))
    ax.bar(x, rr, color="#2a78d6", zorder=3)
    # annotate each bar with the noiseless logL at injection (= -0.5<r|r>)
    for xi, rv, lv in zip(x, rr, sll):
        ax.annotate(f"logL={lv:.2f}", (xi, rv), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color="#52514e")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"SNR {s:.0f}" if np.isfinite(s) else str(i)
                        for i, s in enumerate(snrs)], rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel(r"raw $\langle r|r\rangle$ at injection")
    ax.set_title(
        f"{cls}: residual power + noiseless logL at injection "
        f"(logL = -0.5<r|r>) — {len(rr)} source(s)\n"
        f"worst <r|r> = {rr.max():.3e}  (logL = {(-0.5 * rr.max()):.2f})"
    )
    ax.grid(True, which="both", axis="y", alpha=0.15)
    fig.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"{args.branch}_null_residual.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[RESULT] null_proof=ok null_proof_ok=1 worst_rr={rr.max():.6e} "
          f"worst_logL_noiseless={(-0.5 * rr.max()):.6e} "
          f"n_sources={len(rr)} plot={out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[RESULT] null_proof=FAIL error={type(exc).__name__}", flush=True)
        sys.exit(1)
