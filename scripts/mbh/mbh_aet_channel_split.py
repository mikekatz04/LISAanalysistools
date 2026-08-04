#!/usr/bin/env python
"""Per-channel A / E / T mismatch: which TDI channel carries the MBH residual?

XYZ mixes the response, so an XYZ mismatch cannot say WHERE a template fails.
A and E are the two "science" channels (they carry the GW signal); T is the
null/Sagnac-like channel, signal-suppressed at low frequency. If a template
matches the data in A and E but not in T, the discrepancy lives in the part of
the response that cancels in the science channels -- a very different fault
from one that shows up in A/E.

Reads the TD arrays dumped by ``mbh_three_way_compare.py``
(``/tmp/mbh_three_way_td_id<N>.npz``, XYZ), converts with the stock
``lisatools.utils.utility.AET``, and computes the mismatch per channel through
stock ``AnalysisContainer`` + single-channel ``SensitivityMatrix`` objects
(A2TDISens / E2TDISens / T2TDISens), plus the AE-only and full AET combinations.
No waveform or response regeneration -- this is pure post-processing.

Usage::

    MBHB_ID=19 python mbh_aet_channel_split.py
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

MBHB_ID = int(os.environ.get("MBHB_ID", "19"))
SENS_MODEL = os.environ.get("SENS_MODEL", "scirdv1")
TUKEY_ALPHA = float(os.environ.get("TUKEY_ALPHA", "0.05"))
F_MIN, F_MAX = 1e-4, 2.5e-2
SRC = os.environ.get("TD_NPZ", f"/tmp/mbh_three_way_td_id{MBHB_ID}.npz")

BANDS = [("full", F_MIN, F_MAX), (">5e-4", 5e-4, F_MAX),
         (">1mHz", 1e-3, F_MAX), (">2mHz", 2e-3, F_MAX)]


def main():
    from scipy.signal.windows import tukey
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import FDSettings, TDSettings, TDSignal
    from lisatools.sensitivity import (
        A2TDISens, E2TDISens, T2TDISens, SensitivityMatrix,
    )
    from lisatools.utils.utility import AET

    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found -- run mbh_three_way_compare.py first")
    z = np.load(SRC)
    Dx, Ax, Bx = z["D"], z["A"], z["B"]          # (3, N) in XYZ
    dt = float(z["dt"])
    n = Dx.shape[-1]

    # stock XYZ -> AET
    to_aet = lambda a: np.asarray(AET(a[0], a[1], a[2]))
    D, A, B = to_aet(Dx), to_aet(Ax), to_aet(Bx)

    win = tukey(n, TUKEY_ALPHA)
    td = TDSettings(N=n, dt=dt, t0=float(z["window_t0"]), force_backend="cpu")

    # data power per channel (unweighted) -- context for the mismatches
    f_all = np.fft.rfftfreq(n, dt)
    sel = (f_all >= F_MIN) & (f_all <= F_MAX)
    P = np.abs(np.fft.rfft(D * win[None, :], axis=-1)) ** 2
    tot = P[:, sel].sum()
    print(f"id={MBHB_ID}  N={n}  dt={dt}s  window={n*dt/86400:.2f} d", flush=True)
    print("  in-band data power share:  " + "   ".join(
        f"{c}={100*P[i, sel].sum()/tot:7.4f}%" for i, c in enumerate("AET")), flush=True)

    CHANS = [("A", [A2TDISens], [0]), ("E", [E2TDISens], [1]),
             ("T", [T2TDISens], [2]),
             ("AE", [A2TDISens, E2TDISens], [0, 1]),
             ("AET", [A2TDISens, E2TDISens, T2TDISens], [0, 1, 2])]

    for label, tmpl_arr in (("legacy", A), ("on-the-fly", B)):
        print(f"\n  === {label} vs mojito data — 1-Re(O) (no max) ===", flush=True)
        hdr = f"{'chan':>5} " + " ".join(f"{b[0]:>12}" for b in BANDS)
        print(hdr, flush=True)
        for cname, sens_list, idx in CHANS:
            row = f"{cname:>5} "
            for bname, lo, hi in BANDS:
                fd = FDSettings(N=n // 2 + 1, df=1.0 / (n * dt), min_freq=lo,
                                max_freq=hi, force_backend="cpu")
                dsig = TDSignal(np.ascontiguousarray(D[idx]), td).transform(fd, window=win)
                tsig = TDSignal(np.ascontiguousarray(tmpl_arr[idx]), td).transform(fd, window=win)
                ac = AnalysisContainer(dsig, SensitivityMatrix(fd, sens_list,
                                                              model=SENS_MODEL))
                O = ac.template_inner_product(tsig, normalize=True, complex=True)
                row += f" {1 - O.real:>12.4e}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
