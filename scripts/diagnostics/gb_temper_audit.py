"""Tempering audit for a GB global-fit store: is the ladder doing its job?

Renders, from a single snapshot h5, the evidence for/against the tempering
channel resolving a displaced mode (the 2026-08-26 highf-grid question:
hot rungs assemble ridge+truth pairs, rung-0 swaps import them cold, the
mosaic's fdot then corrects):

  * per-iteration rung-pair-0 (cold <-> rung 1) swap accepts/proposals for
    the bands of interest -- the cold-import channel, from the stored
    ``band_swaps_*`` (bands, ntemps-1) counters (cumulative; diffed here);
  * per-rung occupancy of a NEAR-TRUTH window vs a RIDGE window over
    iterations -- where does the correct solution form first;
  * per-rung physical-fdot quality of the near-truth leaves (median, and
    the fraction within 3x of truth) -- do hot rungs find the RIGHT fdot
    while cold still carries the mosaic's wrong one;
  * new-cold-leaf appearance counts per iteration (proxy for swap imports
    when the log's cold RJ accepts are known to be ~0).

Usage (from the LAT root, deving python):
  python scripts/diagnostics/gb_temper_audit.py <store>/<name>_testing.h5 \
      [--truth-f0-mhz 20.380377] [--near -2 4] [--ridge 11 16] \
      [--rungs 8] [--every 1]

NOTE on reduced ``*_extract.h5`` stores: chain COORDS are real only in
the final keep-window (inds/band arrays are full) -- window-based counts
before the keep-window are then meaningless. Probe snapshots are full
stores and are fine.
"""

import argparse

import h5py
import numpy as np

MSUN_S = 4.925490947641267e-6  # GM_sun / c^3 [s]


def physical_fdot(f0_hz, mc, ratio):
    return (
        (96.0 / 5.0)
        * np.pi ** (8.0 / 3.0)
        * (MSUN_S * mc) ** (5.0 / 3.0)
        * f0_hz ** (11.0 / 3.0)
        * (1.0 + ratio)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5")
    ap.add_argument("--truth-f0-mhz", type=float, default=20.380377)
    ap.add_argument("--truth-fdot", type=float, default=1.0245e-13)
    ap.add_argument("--near", type=float, nargs=2, default=(-2.0, 4.0),
                    help="near-truth window in bins (1/Tobs units)")
    ap.add_argument("--ridge", type=float, nargs=2, default=(11.0, 16.0),
                    help="displaced-mode window in bins")
    ap.add_argument("--tobs", type=float, default=7776000.0,
                    help="Tobs [s] defining the bin width 1/Tobs")
    ap.add_argument("--rungs", type=int, default=8,
                    help="report the first N rungs individually")
    ap.add_argument("--every", type=int, default=1,
                    help="report every Nth stored iteration")
    args = ap.parse_args()

    truth_hz = args.truth_f0_mhz * 1e-3
    dbin = 1.0 / args.tobs

    with h5py.File(args.h5, "r") as f:
        gf = f["global_fit"]
        nit = int(gf.attrs["iteration"])
        gb = gf["sub_backend"]["gb"]
        band_edges = np.asarray(gb["band_edges"])
        if band_edges.ndim > 1:
            band_edges = band_edges.reshape(-1)
        # bands whose [lo, hi) intersects either window (widened 2 bins)
        windows_hz = [
            (truth_hz + args.near[0] * dbin, truth_hz + args.near[1] * dbin),
            (truth_hz + args.ridge[0] * dbin, truth_hz + args.ridge[1] * dbin),
        ]
        lo = min(w[0] for w in windows_hz) - 2 * dbin
        hi = max(w[1] for w in windows_hz) + 2 * dbin
        bands = [
            b for b in range(len(band_edges) - 1)
            if band_edges[b + 1] > lo and band_edges[b] < hi
        ]
        print(f"store: {args.h5}")
        print(f"stored iterations: {nit}; truth f0 = {truth_hz:.9f} Hz; "
              f"1 bin = {dbin:.4e} Hz")
        print(f"bands intersecting the windows: {bands}")

        swaps_a = gb["band_swaps_accepted"]   # (it, [1,] bands, ntemps-1)
        swaps_p = gb["band_swaps_proposed"]
        chain = gb["chain"]                   # (it, nt, nw, nleaves, 9)
        inds = gb["inds"]                     # (it, nt, nw, nleaves)
        ntemps = chain.shape[1]
        n_r = min(args.rungs, ntemps)

        def _sw(arr, it):
            a = np.asarray(arr[it])
            return a.reshape(-1, a.shape[-1])  # (bands, ntemps-1)

        prev_cold_alive = None
        prev_sw_a = None
        prev_sw_p = None
        hdr = (
            "it | pair0 acc/prop " + "+".join(f"b{b}" for b in bands)
            + " | new-cold | per-rung near (fdot-ok) / ridge counts, "
            f"rungs 0..{n_r - 1}"
        )
        print(hdr)
        print("-" * len(hdr))
        for it in range(0, nit, max(1, args.every)):
            sa = _sw(swaps_a, it)
            sp = _sw(swaps_p, it)
            if prev_sw_a is not None and np.all(sa >= prev_sw_a):
                da = sa - prev_sw_a
                dp = sp - prev_sw_p
            else:  # first row / counter reset: report the raw row
                da, dp = sa, sp
            prev_sw_a, prev_sw_p = sa, sp
            acc0 = int(sum(da[b, 0] for b in bands))
            prop0 = int(sum(dp[b, 0] for b in bands))

            c = np.asarray(chain[it])
            al = np.asarray(inds[it]).astype(bool)
            f0 = c[..., 1] * 1e-3
            off = (f0 - truth_hz) / dbin
            near = al & (off >= args.near[0]) & (off <= args.near[1])
            ridge = al & (off >= args.ridge[0]) & (off <= args.ridge[1])
            fd = physical_fdot(f0, c[..., 2], c[..., 8])
            fd_ok = (fd > args.truth_fdot / 3.0) & (fd < args.truth_fdot * 3.0)

            cold_alive = al[0]
            if prev_cold_alive is not None:
                new_cold = int((cold_alive & ~prev_cold_alive).sum())
            else:
                new_cold = int(cold_alive.sum())
            prev_cold_alive = cold_alive

            rungs = []
            for k in range(n_r):
                nn = int(near[k].sum())
                nok = int((near[k] & fd_ok[k]).sum())
                nr = int(ridge[k].sum())
                rungs.append(f"{nn}({nok})/{nr}")
            print(f"{it:3d} | {acc0:5d}/{prop0:5d} | {new_cold:3d} | "
                  + "  ".join(rungs))

        # closing summary: where does the CORRECT solution live?
        c = np.asarray(chain[nit - 1])
        al = np.asarray(inds[nit - 1]).astype(bool)
        f0 = c[..., 1] * 1e-3
        off = (f0 - truth_hz) / dbin
        near = al & (off >= args.near[0]) & (off <= args.near[1])
        fd = physical_fdot(f0, c[..., 2], c[..., 8])
        print("\nlast iteration, per-rung NEAR-TRUTH fdot quality "
              "(n, median fdot, frac within 3x):")
        for k in range(n_r):
            m = near[k]
            n = int(m.sum())
            if n == 0:
                print(f"  rung {k:2d}: 0")
                continue
            med = float(np.median(fd[k][m]))
            ok = float(
                ((fd[k][m] > args.truth_fdot / 3.0)
                 & (fd[k][m] < args.truth_fdot * 3.0)).mean()
            )
            print(f"  rung {k:2d}: n={n:3d}  med={med:.3e}  frac3x={ok:.2f}")


if __name__ == "__main__":
    main()
