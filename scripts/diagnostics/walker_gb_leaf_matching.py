"""Per-layer GB leaf lists for the layers where the walkers differ most, f0-only
matching, and the per-walker leaf-count history by frequency band.

usage: python low_walker3.py <run_dir>
"""
import os
import sys

import h5py
import numpy as np

np.set_printoptions(linewidth=220, precision=3, suppress=True)
run = sys.argv[1]
h5p = [x for x in os.listdir(run) if x.endswith("_extract.h5")][0]
f = h5py.File(os.path.join(run, h5p), "r")
g = f["global_fit"]
it = int(g.attrs["iteration"])
row = it - 1
nw = int(g.attrs["nwalkers"])
c = g["chain/gb"][row, 0, 0]
ind = g["inds/gb"][row, 0, 0]
hh = g["sub_backend/gb/h_h"][row]
dh = g["sub_backend/gb/d_h"][row]
LW = 1.0 / (2 * 1440 * 2.5) * 1e3  # WDM layer width, mHz
F0 = 0.2778  # first layer edge in the snapshot, mHz (f_edges[0])

leaves = []
for w in range(nw):
    m = ind[w]
    f0 = c[w, m, 1]
    o = np.argsort(f0)
    leaves.append(dict(f0=f0[o], amp=c[w, m, 0][o], fdot=c[w, m, 2][o], ci=c[w, m, 4][o],
                       lam=c[w, m, 6][o], sb=c[w, m, 7][o],
                       snr=np.sqrt(np.maximum(hh[w, m], 0))[o],
                       gain=(dh[w, m] - 0.5 * hh[w, m])[o]))

# layers to inspect: centre frequencies in mHz (from the residual decomposition)
layers = [3.54, 2.57, 2.71, 2.85, 3.82, 3.96, 1.74, 2.01, 1.88, 1.60, 4.10, 3.12, 4.38, 3.26, 2.29]
print("== per-layer GB content (leaves with SNR >= 7 listed; counts/sums over all leaves) ==")
for fcen in layers:
    lo, hi = fcen - LW / 2, fcen + LW / 2
    print(f"\n-- layer {fcen:.2f} mHz ({lo:.3f}-{hi:.3f}) --")
    for w in range(nw):
        L = leaves[w]
        m = (L["f0"] >= lo) & (L["f0"] < hi)
        n = m.sum()
        print(f"  w{w}: n={n:2d} sum SNR^2={np.sum(L['snr'][m]**2):8.0f} sum gain={np.sum(L['gain'][m]):8.0f} | "
              + " ".join(f"{a:.4f}:{s:.1f}" for a, s in zip(L["f0"][m], L["snr"][m]) if s >= 7))

# f0-only matching: a leaf on v is MISSING on w when w has no leaf within TOL of its f0
TOL = 1.5e-4
print("\n== f0-only matching (|df0| <= 1.5e-4 mHz): leaves of column walker with no leaf nearby on the row walker ==")
M = np.zeros((nw, nw), dtype=int)
MS = np.zeros((nw, nw))
for w in range(nw):
    for v in range(nw):
        if v == w:
            continue
        fw = leaves[w]["f0"]
        fv = leaves[v]["f0"]
        lo = np.searchsorted(fw, fv - TOL)
        hi = np.searchsorted(fw, fv + TOL)
        miss = hi == lo
        M[w, v] = miss.sum()
        MS[w, v] = (leaves[v]["snr"][miss] ** 2).sum() / 2
print("counts:\n", M)
print("sum SNR^2/2 of the missing leaves (nats, upper bound on what the row walker forgoes):\n", MS.round(0))

print("\n== sources present on ALL THREE other walkers (f0-only) but absent on walker w ==")
for w in range(nw):
    others = [v for v in range(nw) if v != w]
    base = others[0]
    fb = leaves[base]["f0"]
    sb_ = leaves[base]["snr"]
    ok = np.ones(len(fb), dtype=bool)
    for v in others[1:]:
        fv = leaves[v]["f0"]
        lo = np.searchsorted(fv, fb - TOL)
        hi = np.searchsorted(fv, fb + TOL)
        ok &= hi > lo
    fw = leaves[w]["f0"]
    lo = np.searchsorted(fw, fb - TOL)
    hi = np.searchsorted(fw, fb + TOL)
    miss = ok & (hi == lo)
    snr = sb_[miss]
    print(f"  w{w}: {miss.sum()} absent (sum SNR^2/2 = {np.sum(snr**2)/2:.0f} nats); SNR>=8: {(snr>=8).sum()}, >=12: {(snr>=12).sum()}; "
          f"by 1-mHz bin: " + " ".join(f"{k}:{v}" for k, v in sorted({int(x): int(np.sum(fb[miss].astype(int) == int(x))) for x in fb[miss]}.items())))
    big = np.flatnonzero(miss & (sb_ >= 8))
    print("       SNR>=8 absent (f0:SNR on walker %d): " % base + " ".join(f"{fb[k]:.4f}:{sb_[k]:.1f}" for k in big[:40]))

# per-walker leaf-count history by band (cold rung) every 8 rows
bnb = g["sub_backend/gb/band_num_binaries"]
bedges = g["sub_backend/gb/band_edges"][...] * 1e3
bc = 0.5 * (bedges[:-1] + bedges[1:])
bands = [(0.5, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 25)]
print("\n== cold leaf counts per walker by band, every 8 rows (w0 w1 w2 w3) ==")
hdr = "row  " + "  ".join(f"[{lo}-{hi})".ljust(19) for lo, hi in bands) + "  total"
print(hdr)
for r in list(range(0, it, 8)) + [it - 1]:
    cnt = bnb[r, 0]  # (nw, nbands)
    if not np.isfinite(cnt).all() or cnt.sum() == 0:
        continue
    cells = []
    for lo, hi in bands:
        m = (bc >= lo) & (bc < hi)
        cells.append(" ".join(f"{int(cnt[w][m].sum()):4d}" for w in range(nw)))
    print(f"{r:4d} " + "  ".join(cells) + "  " + " ".join(f"{int(cnt[w].sum()):4d}" for w in range(nw)))

# lnL gap history alongside (every 8 rows)
ll = g["log_like"][:it, 0, 0, :]
print("\n== lnL gap vs the row max, every 8 rows ==")
for r in list(range(0, it, 8)) + [it - 1]:
    print(f"{r:4d} " + " ".join(f"{v:8.0f}" for v in ll[r] - ll[r].max()))
f.close()
