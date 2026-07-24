#!/usr/bin/env python
"""Proof plot for a source ground-truth (null-check) gate.

Reads the gate's already-captured null-check log (no recompute), extracts the
per-source ``rr/dd`` from the ``[RESULT] branch=... rr_over_dd=...`` lines, and
draws a bar chart against the 2x-baseline null threshold — the visual proof
that the stock template nulls the mojito data for every source of the class.
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

# 2x the 2026-07-11 mojito null baselines (mirror gates.NULL_BASELINE_2X).
THRESH = {"MBH": 3.0e-3, "SOBBH": 3.0e-6, "EMRI": 1.2e-3}
_RR = re.compile(r"rr_over_dd=([\d.eE+-]+)")
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
        if "[RESULT]" not in line or "rr_over_dd" not in line:
            continue
        m, s = _RR.search(line), _SNR.search(line)
        if m:
            rows.append((float(s.group(1)) if s else np.nan, float(m.group(1))))
    if not rows:
        print("[RESULT] null_proof=SKIP reason=no_rows", flush=True)
        return

    cls = {"mbh": "MBH", "emri": "EMRI", "sobbh": "SOBBH"}.get(args.branch, "MBH")
    thr = THRESH.get(cls, 1e-3)
    snrs = np.array([r[0] for r in rows])
    rr = np.array([r[1] for r in rows])
    order = np.argsort(rr)[::-1]
    rr, snrs = rr[order], snrs[order]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rr) + 3), 4.4))
    x = np.arange(len(rr))
    ax.bar(x, rr, color=np.where(rr <= thr, "#1baf7a", "#d03b3b"), zorder=3)
    ax.axhline(thr, ls="--", color="#0b0b0b", lw=1.2,
               label=f"null threshold (2× baseline = {thr:.1e})")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"SNR {s:.0f}" if np.isfinite(s) else str(i)
                        for i, s in enumerate(snrs)], rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel(r"$\langle r|r\rangle / \langle d|d\rangle$  (null residual)")
    ax.set_title(f"{cls} template nulls the mojito data — {len(rr)} source(s)\n"
                 f"worst {rr.max():.2e}  (threshold {thr:.1e})")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.15)
    fig.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"{args.branch}_null_residual.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    ok = int((rr <= thr).all())
    print(f"[RESULT] null_proof=ok null_proof_ok={ok} worst_rr_dd={rr.max():.3e} "
          f"n_sources={len(rr)} plot={out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[RESULT] null_proof=FAIL error={type(exc).__name__}", flush=True)
        sys.exit(1)
