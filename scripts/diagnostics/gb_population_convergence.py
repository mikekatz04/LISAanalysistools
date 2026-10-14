"""SNR-threshold GB population convergence across a run's snapshot extracts.

usage: python gb_population_convergence.py --truth <gb_truth npz> --tobs <s> RUN_DIR [RUN_DIR ...]

Each RUN_DIR is an extracted snapshot run dir holding one *_extract.h5 (its
last stored row carries chain/gb + sub_backend/gb/h_h, d_h). Per snapshot:
per-walker N(SNR_opt > thr), the cross-walker union, truth completeness per
injected-SNR bin (the monitor's rule: one-to-one greedy f0 match within +-2 FD
bins), the fraction of SNR>8 / >10 leaves matched to a detectable injection,
then the churn of the SNR>8 / >10 sets between consecutive snapshots (f0 match
within 1.5e-7 Hz: new / lost / promoted / demoted). The last RUN_DIR's extract
also gives the per-row leaf-count and lnL slopes over the final 100 rows.

Written for the 2026-10-13 "when to launch the 1yr" assessment on the 9mo
(rows 19 -> 349). Slot-based leaf ages are NOT a discovery measure (vertical
swaps re-slot existing leaves), so they are deliberately not computed here.
"""
import argparse
import os

import h5py
import numpy as np

np.set_printoptions(linewidth=200, precision=3, suppress=True)
THR = [5, 6, 7, 8, 10, 12, 15, 20]
BINS = [6, 7, 8, 10, 12, 15, 20, 1e9]
TOL_W = 1.5e-7  # Hz, walker/snapshot match


def match_pairs(rf, tf, tol):
    if rf.size == 0 or tf.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    o = np.argsort(tf); tfs = tf[o]
    lo = np.searchsorted(tfs, rf - tol); hi = np.searchsorted(tfs, rf + tol)
    ri, ti = [], []
    for i in range(rf.size):
        for j in o[lo[i]:hi[i]]:
            ri.append(i); ti.append(j)
    if not ri:
        return np.zeros(0, int), np.zeros(0, int)
    ri = np.asarray(ri); ti = np.asarray(ti)
    d = np.abs(rf[ri] - tf[ti])
    ur = np.zeros(rf.size, bool); ut = np.zeros(tf.size, bool)
    out_r, out_t = [], []
    for m in np.argsort(d):
        a, b = ri[m], ti[m]
        if ur[a] or ut[b]:
            continue
        ur[a] = ut[b] = True
        out_r.append(a); out_t.append(b)
    return np.asarray(out_r, int), np.asarray(out_t, int)


def load_snapshot(d):
    p = sorted((x for x in os.listdir(d) if x.endswith("_extract.h5")),
               key=lambda x: os.path.getmtime(os.path.join(d, x)))[-1]
    with h5py.File(os.path.join(d, p), "r") as f:
        g = f["global_fit"]
        it = int(g.attrs["iteration"]); row = it - 1
        nw = int(g.attrs["nwalkers"])
        c = g["chain/gb"][row, 0, 0]
        ind = g["inds/gb"][row, 0, 0]
        hh = g["sub_backend/gb/h_h"][row]
    assert np.any(c[0, :, 1] != 0), f"{d}: empty coord row {row}"
    W = []
    for w in range(nw):
        m = ind[w]
        f0 = c[w, m, 1] * 1e-3
        o = np.argsort(f0)
        W.append(dict(f0=f0[o], snr=np.sqrt(np.maximum(hh[w, m], 0))[o]))
    return row, W


def union(W):
    uf = W[0]["f0"].copy(); us = W[0]["snr"].copy()
    for w in range(1, len(W)):
        o = np.argsort(uf); uf = uf[o]; us = us[o]
        lo = np.searchsorted(uf, W[w]["f0"] - TOL_W); hi = np.searchsorted(uf, W[w]["f0"] + TOL_W)
        new = hi == lo
        for i in np.nonzero(~new)[0]:
            j = lo[i] + np.argmin(np.abs(uf[lo[i]:hi[i]] - W[w]["f0"][i]))
            us[j] = max(us[j], W[w]["snr"][i])
        uf = np.r_[uf, W[w]["f0"][new]]; us = np.r_[us, W[w]["snr"][new]]
    return uf, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tobs", type=float, required=True, help="observation time [s] (match tol = 2/Tobs)")
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    tol_t = 2.0 / a.tobs
    tr = np.load(a.truth, allow_pickle=True)
    det = tr["det"]; T_F0 = tr["f0"][det]; T_SNR = tr["snr"][det]
    print(f"truth: {tr['f0'].size} injections, det {det.sum()} (min det SNR {T_SNR.min():.2f}), "
          f"built at row {int(tr['iteration'])}, tobs {float(tr['tobs']):.0f}")
    print("truth det N(snr>thr):", {t: int((T_SNR > t).sum()) for t in THR})

    snaps = []
    for d in a.run_dirs:
        row, W = load_snapshot(d)
        snaps.append((d, row, W))
        nw = len(W)
        uf, us = union(W)
        print(f"\n== {d} row {row}")
        for thr in THR:
            per = [int((W[w]["snr"] > thr).sum()) for w in range(nw)]
            print(f"     thr {thr:>2}: {per}  mean {np.mean(per):7.1f} | union {(us > thr).sum()}")
        found_any = np.zeros(T_F0.size, bool)
        founds = []
        for w in range(nw):
            ri, ti = match_pairs(W[w]["f0"], T_F0, tol_t)
            found = np.zeros(T_F0.size, bool); found[ti] = True
            founds.append(found); found_any |= found
            ml = np.zeros(W[w]["f0"].size, bool); ml[ri] = True
            comp = [found[(T_SNR >= BINS[i]) & (T_SNR < BINS[i + 1])].mean() if ((T_SNR >= BINS[i]) & (T_SNR < BINS[i + 1])).any() else np.nan
                    for i in range(len(BINS) - 1)]
            print(f"   w{w} completeness by truth-SNR bin {np.round(comp, 3)} | leaves SNR>8 matched {ml[W[w]['snr'] > 8].mean():.3f}, SNR>10 {ml[W[w]['snr'] > 10].mean():.3f}")
        compu = [found_any[(T_SNR >= BINS[i]) & (T_SNR < BINS[i + 1])].mean() if ((T_SNR >= BINS[i]) & (T_SNR < BINS[i + 1])).any() else np.nan
                 for i in range(len(BINS) - 1)]
        print(f"   UNION completeness by bin      {np.round(compu, 3)}")
        for thr in (7, 8, 10, 12, 15):
            sel = T_SNR > thr
            pw = [f[sel].mean() for f in founds]
            print(f"   completeness(truth SNR > {thr:>2}): per walker {np.round(pw, 4)} mean {np.mean(pw):.4f} union {found_any[sel].mean():.4f} (n={sel.sum()})")

    print("\n== churn between consecutive snapshots (per walker, f0 match within 1.5e-7 Hz) ==")
    for (da, ra, Wa), (db, rb, Wb) in zip(snaps[:-1], snaps[1:]):
        print(f"\n row {ra} -> row {rb}: {rb - ra} rows")
        for thr in (8, 10):
            line = []
            for w in range(len(Wa)):
                ri, ti = match_pairs(Wb[w]["f0"], Wa[w]["f0"], TOL_W)
                mb = np.zeros(Wb[w]["f0"].size, bool); mb[ri] = True
                ma = np.zeros(Wa[w]["f0"].size, bool); ma[ti] = True
                hb = Wb[w]["snr"] > thr; ha = Wa[w]["snr"] > thr
                sa = np.full(Wb[w]["f0"].size, np.nan); sa[ri] = Wa[w]["snr"][ti]
                sb = np.full(Wa[w]["f0"].size, np.nan); sb[ti] = Wb[w]["snr"][ri]
                line.append(f"w{w}: N {ha.sum()}->{hb.sum()} new {(hb & ~mb).sum():3d} lost {(ha & ~ma).sum():3d} "
                            f"prom {(hb & mb & (sa <= thr)).sum():3d} dem {(ha & ma & (sb <= thr)).sum():3d}")
            print(f"   SNR>{thr}: " + " | ".join(line))

    d = snaps[-1][0]
    p = sorted((x for x in os.listdir(d) if x.endswith("_extract.h5")),
               key=lambda x: os.path.getmtime(os.path.join(d, x)))[-1]
    with h5py.File(os.path.join(d, p), "r") as f:
        g = f["global_fit"]; it = int(g.attrs["iteration"])
        nl = g["inds/gb"][:it, 0, 0].sum(axis=2); ll = g["log_like"][:it, 0, 0, :]
    x = np.arange(max(0, it - 100), it)
    print(f"\n== {d}: last-100-row slopes (rows {x[0]}-{x[-1]}) ==")
    for w in range(nl.shape[1]):
        print(f"  w{w}: leaves {np.polyfit(x, nl[x, w], 1)[0]:+.2f}/row, lnL {np.polyfit(x, ll[x, w], 1)[0]:+.1f} nats/row")


if __name__ == "__main__":
    main()
