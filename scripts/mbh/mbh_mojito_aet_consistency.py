#!/usr/bin/env python
"""Is mojito's stored A2/E2/T2 the same linear combination of its X2/Y2/Z2 that we use?

The MBH residual is dominated by the T channel (see mbh_aet_channel_split.py /
mbh_xyz_residual_structure.py). Before blaming the response, verify the basis itself:

  * we compare template-AET vs data-AET where BOTH are converted from XYZ with
    ``lisatools.utils.utility.AET``, so a convention error in OUR transform cancels;
  * but if MOJITO's stored A2/E2/T2 are not that same combination of its own
    X2/Y2/Z2 -- different sign, normalisation, channel order, or a genuinely
    different TDI-2 delay chain -- then mojito's XYZ and our XYZ mean subtly
    different things, and T (the near-null combination) is exactly where such a
    difference would be amplified while A/E stay clean.

This reads BOTH sets straight from the L1 file over the same analysis window and
compares stored-vs-derived per channel: rms ratio, correlation, and the rms of the
difference relative to the channel's own rms.

Usage::

    MBHB_ID=19 python mbh_mojito_aet_consistency.py
"""

from __future__ import annotations

import os

import numpy as np

MBHB_ID = int(os.environ.get("MBHB_ID", "19"))
PATH = os.environ.get(
    "MOJITO_DATA_PATH",
    "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/")
WIN_NPZ = f"/tmp/mbh_mojito_data_id{MBHB_ID}.npz"
DT = 10.0
DT_NATIVE = 2.5


def main():
    import h5py
    from lisatools.globalfit.preprocessing import find_file
    from lisatools.utils.utility import AET

    z = np.load(WIN_NPZ, allow_pickle=True)
    window_t0 = float(z["window_t0"])
    data_t0 = float(z["data_t0"])
    n_win = int(z["data_td"].shape[-1])
    deci = int(round(DT / DT_NATIVE))
    start_full = int(round((window_t0 - data_t0) / DT_NATIVE))
    span = n_win * deci
    print(f"id={MBHB_ID}  window_t0={window_t0:.3f}  data_t0={data_t0:.3f}  "
          f"start={start_full}  n_win={n_win}  deci={deci}", flush=True)

    fp = find_file(os.path.join(PATH, "data", "MBHB", "L1"), "MBHB", MBHB_ID)
    print(f"L1: {fp}", flush=True)

    with h5py.File(fp, "r") as f:
        g = f["tdis"]
        print(f"tdis keys: {sorted(g.keys())}", flush=True)
        lf = float(f.attrs["laser_frequency"])
        sl = slice(start_full, start_full + span, deci)
        get = lambda k: np.asarray(g[k][sl][:n_win], dtype=np.float64)
        X, Y, Z = get("X2"), get("Y2"), get("Z2")
        stored = {c: get(f"{c}2") for c in "AET"}

    derived = dict(zip("AET", AET(X, Y, Z)))

    print(f"\n  laser_frequency = {lf:.6e}", flush=True)
    print(f"  {'chan':>5} {'rms(stored)':>14} {'rms(derived)':>14} {'ratio':>10} "
          f"{'corr':>10} {'rms(diff)/rms':>14}", flush=True)
    for c in "AET":
        s, d = stored[c], derived[c]
        rs, rd = float(np.sqrt(np.mean(s ** 2))), float(np.sqrt(np.mean(d ** 2)))
        corr = float(np.corrcoef(s, d)[0, 1])
        rel = float(np.sqrt(np.mean((s - d) ** 2)) / max(rs, 1e-300))
        print(f"  {c:>5} {rs:>14.6e} {rd:>14.6e} {rd/max(rs,1e-300):>10.6f} "
              f"{corr:>10.6f} {rel:>14.6e}", flush=True)

    # also try the common alternative sign/normalisation conventions for T
    print("\n  alternative T conventions vs stored T2:", flush=True)
    Ts = stored["T"]
    for name, cand in (("(X+Y+Z)/sqrt3", (X + Y + Z) / np.sqrt(3.0)),
                       ("(X+Y+Z)/3", (X + Y + Z) / 3.0),
                       ("(X+Y+Z)", (X + Y + Z)),
                       ("-(X+Y+Z)/sqrt3", -(X + Y + Z) / np.sqrt(3.0))):
        rel = float(np.sqrt(np.mean((Ts - cand) ** 2)) / max(np.sqrt(np.mean(Ts ** 2)), 1e-300))
        corr = float(np.corrcoef(Ts, cand)[0, 1])
        print(f"    {name:>18}: corr={corr:+.6f}  rms(diff)/rms(T2)={rel:.6e}", flush=True)


if __name__ == "__main__":
    main()
