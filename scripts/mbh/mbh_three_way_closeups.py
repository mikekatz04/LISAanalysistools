#!/usr/bin/env python
"""TD comparison of data vs legacy vs TDI-on-the-fly: full window + closeups.

Reads the TD arrays dumped by ``mbh_three_way_compare.py``
(``/tmp/mbh_three_way_td_id<N>.npz``) so any zoom is free -- no waveform or
response regeneration.  Run the three-way script first for the id you want.

Rows (top = the whole window, then progressively tighter):

  FULL WINDOW       every sample, envelope-binned so nothing is hidden
  window START      first EDGE_H hours
  merger +/-4 h     the burst in context
  merger +/-30 min  ringdown structure
  merger +/-5 min   the peak, where a step/kink at t_c would show
  window END        last EDGE_H hours

Columns:

  overlay           data / legacy / on-the-fly
  residual          legacy-data and on-fly-data, linear (sign visible)
  |residual|        same, LOG scale -- spans the 4+ decades the linear axis
                    flattens, and is where a low-level broadband floor or a
                    hard template edge actually shows up

The edge rows matter because the on-the-fly generator zero-fills the data grid
outside its own span with no taper (bbhx mbhtdionfly.py, the ``keep`` mask), and
the legacy path's phentax ``waveform_duration`` can be SHORTER than the
window's pre-merger span (stock default YRSID/12 ~ 30.4 d), truncating the
low-frequency inspiral.  Both show up as a template that dies while the data
keeps going.  Each row prints per-region RMS and the zero-fraction of each
template so a genuine edge is distinguishable from ordinary low signal.

Usage::

    MBHB_ID=19 python mbh_three_way_closeups.py
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MBHB_ID = int(os.environ.get("MBHB_ID", "19"))
EDGE_H = float(os.environ.get("EDGE_H", "2.0"))      # hours shown at each edge
CH = int(os.environ.get("CHAN", "0"))                # 0=X, 1=Y, 2=Z
NBIN = int(os.environ.get("NBIN", "3000"))           # envelope bins, full window
SRC = f"/tmp/mbh_three_way_td_id{MBHB_ID}.npz"


def envelope(x, y, nbin):
    """Max-|y| per bin -- decimation that cannot hide a spike."""
    n = y.size
    if n <= nbin:
        return x, y, y
    step = n // nbin
    m = (n // step) * step
    yb = y[:m].reshape(-1, step)
    xb = x[:m].reshape(-1, step)[:, 0]
    return xb, yb.max(axis=1), yb.min(axis=1)


def main():
    if not os.path.exists(SRC):
        raise SystemExit(
            f"{SRC} not found -- run:  MBHB_ID={MBHB_ID} python "
            f"scripts/mbh/mbh_three_way_compare.py")
    z = np.load(SRC)
    D, A, B = z["D"][CH], z["A"][CH], z["B"][CH]
    dt = float(z["dt"])
    n = D.size
    merger_idx = int(round((float(z["abs_merger"]) - float(z["window_t0"])) / dt))
    inc = float(z["inc"])
    edge_n = int(round(EDGE_H * 3600.0 / dt))

    print(f"id={MBHB_ID}  N={n}  dt={dt}s  window={n*dt/86400:.3f} d  "
          f"merger idx={merger_idx} ({100*merger_idx/n:.1f}% through)", flush=True)

    rows = [
        ("FULL WINDOW", slice(0, n), None, True),
        ("window START", slice(0, edge_n), None, False),
        ("merger +/-4 h", slice(max(0, merger_idx - int(4 * 3600 / dt)),
                                min(n, merger_idx + int(4 * 3600 / dt))),
         merger_idx, False),
        ("merger +/-30 min", slice(max(0, merger_idx - int(1800 / dt)),
                                   min(n, merger_idx + int(1800 / dt))),
         merger_idx, False),
        ("merger +/-5 min", slice(max(0, merger_idx - int(300 / dt)),
                                  min(n, merger_idx + int(300 / dt))),
         merger_idx, False),
        ("window END", slice(max(0, n - edge_n), n), None, False),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(19, 2.9 * len(rows)))
    for r, (name, sl, mark, is_full) in enumerate(rows):
        idx = np.arange(n)[sl]
        if mark is not None:
            x = (idx - mark) * dt / 3600.0
            xlab = "hours from merger"
        elif is_full:
            x = idx * dt / 86400.0
            xlab = "days into window"
        else:
            x = idx * dt / 3600.0
            xlab = "hours into window"

        d, a, b = D[sl], A[sl], B[sl]
        ra, rb = a - d, b - d

        ax = axes[r, 0]
        if is_full:
            for arr, lab, st in ((d, "mojito data", "-"), (a, "legacy", "--"),
                                 (b, "on-the-fly", ":")):
                xb, hi, lo = envelope(x, arr, NBIN)
                ax.fill_between(xb, lo, hi, alpha=0.55, label=lab)
        else:
            ax.plot(x, d, label="mojito data", lw=1.5)
            ax.plot(x, a, "--", label="legacy", lw=1.0)
            ax.plot(x, b, ":", label="on-the-fly", lw=1.2)
        if mark is not None:
            ax.axvline(0, color="k", ls=":", alpha=0.4)
        ax.set_title(f"{name} — X")
        ax.set_xlabel(xlab)
        if r == 0:
            ax.legend(fontsize=8)

        ax2 = axes[r, 1]
        if is_full:
            for arr, lab in ((ra, "legacy - data"), (rb, "on-fly - data")):
                xb, hi, lo = envelope(x, arr, NBIN)
                ax2.fill_between(xb, lo, hi, alpha=0.6, label=lab)
        else:
            ax2.plot(x, ra, "--", label="legacy - data", lw=1.0)
            ax2.plot(x, rb, ":", label="on-fly - data", lw=1.2, color="tab:green")
        if mark is not None:
            ax2.axvline(0, color="k", ls=":", alpha=0.4)
        ax2.set_title(f"{name} — residual (linear)")
        ax2.set_xlabel(xlab)
        if r == 0:
            ax2.legend(fontsize=8)

        # |residual|, log scale -- the requested view
        ax3 = axes[r, 2]
        floor = max(np.abs(d).max() * 1e-12, 1e-30)
        for arr, lab, col in ((np.abs(ra), "|legacy - data|", "tab:blue"),
                              (np.abs(rb), "|on-fly - data|", "tab:green")):
            if is_full:
                xb, hi, _ = envelope(x, arr, NBIN)
                ax3.semilogy(xb, np.maximum(hi, floor), lw=0.9, label=lab, color=col)
            else:
                ax3.semilogy(x, np.maximum(arr, floor), lw=0.9, label=lab, color=col)
        xb2, hd, _ = (envelope(x, np.abs(d), NBIN) if is_full
                      else (x, np.abs(d), None))
        ax3.semilogy(xb2, np.maximum(hd, floor), lw=0.8, color="k", alpha=0.35,
                     label="|data|")
        if mark is not None:
            ax3.axvline(0, color="k", ls=":", alpha=0.4)
        ax3.set_title(f"{name} — |residual| (log)")
        ax3.set_xlabel(xlab)
        if r == 0:
            ax3.legend(fontsize=8)

        za = float(np.mean(a == 0.0))
        zb = float(np.mean(b == 0.0))
        print(f"  {name:18s} rms(data)={np.sqrt(np.mean(d**2)):.4e}  "
              f"rms(legacy-d)={np.sqrt(np.mean(ra**2)):.4e}  "
              f"rms(onfly-d)={np.sqrt(np.mean(rb**2)):.4e}   "
              f"zero-frac legacy={za:.3f} onfly={zb:.3f}", flush=True)

    fig.suptitle(f"MBHB id={MBHB_ID}: data vs legacy vs TDI-on-the-fly "
                 f"(inc={inc:.3f}, window {n*dt/86400:.1f} d, dt={dt:g}s)", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    out = f"/tmp/mbh_closeups_id{MBHB_ID}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"\nDONE.  plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
