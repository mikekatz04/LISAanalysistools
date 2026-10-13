"""Walker-vs-walker lnL decomposition under a COMMON noise, from the per-walker
residual snapshot (raw WDM residuals), plus GB leaf matching across walkers.

usage: python low_walker2.py <run_dir>
"""
import os
import sys

import h5py
import numpy as np

np.set_printoptions(linewidth=220, precision=4, suppress=True)
run = sys.argv[1]
npz = [x for x in os.listdir(run) if x.endswith("_residual_snapshot.npz")][0]
h5p = [x for x in os.listdir(run) if x.endswith("_extract.h5")][0]
d = np.load(os.path.join(run, npz))
R = d["residuals"]  # (nw, 3, nlayer, ncol) float32 raw residual
fe = d["f_edges"]
te = d["t_edges"]
lnl = d["lnl"]
nw = R.shape[0]
fc = 0.5 * (fe[:-1] + fe[1:]) * 1e3  # mHz
tc = 0.5 * (te[:-1] + te[1:]) / 86400.0  # days
print(f"store_row {int(d['store_row'])}  lnl {lnl}  gap vs w0 {lnl - lnl[0]}")

# ---- common-noise estimate per (channel, layer): mean power over walkers and time
P = [R[w].astype(np.float64) ** 2 for w in range(nw)]  # each (3, nl, nc)
S = sum(P) / nw
S = S.mean(axis=2)  # (3, nl)
# per-layer and per-time dlnL vs walker 0:  -1/2 sum (P_w - P_0)/S
Dl = np.zeros((nw, 3, len(fc)))
Dt = np.zeros((nw, len(tc)))
for w in range(nw):
    diff = (P[w] - P[0]) / S[:, :, None]
    Dl[w] = -0.5 * diff.sum(axis=2)
    Dt[w] = -0.5 * diff.sum(axis=(0, 1))
tot = Dl.sum(axis=(1, 2))
print("\n== residual-based dlnL vs walker 0 under a common noise (all channels) ==")
for w in range(nw):
    print(f"  w{w}: residual-based {tot[w]:9.1f}   per channel {Dl[w].sum(axis=1)}   actual lnL gap {lnl[w]-lnl[0]:9.1f}"
          f"   -> noise-model share {lnl[w]-lnl[0]-tot[w]:9.1f}")

print("\n== per-frequency (1 mHz bins) residual-based dlnL vs w0, summed over channels ==")
edges = np.arange(0.0, 26.0, 1.0)
Dfreq = Dl.sum(axis=1)  # (nw, nl)
hdr = "   bin(mHz) " + " ".join(f"      w{w}" for w in range(1, nw))
print(hdr)
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (fc >= lo) & (fc < hi)
    if not m.any():
        continue
    vals = [Dfreq[w][m].sum() for w in range(1, nw)]
    if max(abs(v) for v in vals) < 15:
        continue
    print(f"   {lo:4.0f}-{hi:<4.0f} " + " ".join(f"{v:8.1f}" for v in vals))

print("\n== worst / best single layers (0.139 mHz) for each walker vs w0 ==")
for w in range(1, nw):
    o = np.argsort(Dfreq[w])
    print(f"  w{w} worst: " + "  ".join(f"{fc[i]:.2f}mHz:{Dfreq[w][i]:.0f}" for i in o[:10]))
    print(f"  w{w} best : " + "  ".join(f"{fc[i]:.2f}mHz:{Dfreq[w][i]:.0f}" for i in o[::-1][:6]))

print("\n== time structure: residual-based dlnL vs w0 per 1-day bin (|value| >= 25 shown) ==")
dedges = np.arange(np.floor(tc[0]), np.ceil(tc[-1]) + 1, 1.0)
mbh_days = [92.0, 104.7, 111.4, 173.3, 243.8, 263.8]
for w in range(1, nw):
    out = []
    for lo, hi in zip(dedges[:-1], dedges[1:]):
        m = (tc >= lo) & (tc < hi)
        v = Dt[w][m].sum()
        if abs(v) >= 25:
            out.append(f"d{lo:.0f}:{v:.0f}")
    print(f"  w{w}: std of daily sums {np.array([Dt[w][(tc >= lo) & (tc < hi)].sum() for lo, hi in zip(dedges[:-1], dedges[1:])]).std():.1f}; "
          f"days with |sum|>=25: {' '.join(out) if out else 'none'}")
    # the +-3 d windows around the MBH mergers
    print("      around MBH mergers (+-3 d): " + "  ".join(
        f"{md:.1f}d:{Dt[w][(tc >= md-3) & (tc < md+3)].sum():.0f}" for md in mbh_days))
    # 30-day bins
    m30 = np.arange(0, 300, 30)
    print("      30-day bins: " + " ".join(
        f"{lo:.0f}-{lo+30:.0f}:{Dt[w][(tc >= lo) & (tc < lo+30)].sum():.0f}" for lo in m30 if ((tc >= lo) & (tc < lo+30)).any()))

# ---- store-side: GB leaves, per-band counts, matching
f = h5py.File(os.path.join(run, h5p), "r")
g = f["global_fit"]
it = int(g.attrs["iteration"])
row = it - 1
c = g["chain/gb"][row, 0, 0]  # (nw, nleaves, 9)
ind = g["inds/gb"][row, 0, 0]
hh = g["sub_backend/gb/h_h"][row]
dh = g["sub_backend/gb/d_h"][row]
bedges = g["sub_backend/gb/band_edges"][...] * 1e3  # mHz
bnb = g["sub_backend/gb/band_num_binaries"][row]  # (ntemps, nw, nbands)
sums = bnb.sum(axis=2)  # (ntemps, nw)
cold = ind.sum(axis=1)
tidx = int(np.argmin(np.abs(sums - cold[None, :]).sum(axis=1)))
print(f"\n== GB per-band counts: temp index {tidx} matches the cold leaf counts {cold} (sums {sums[tidx]}) ==")
cnt = bnb[tidx]  # (nw, nbands)
bc = 0.5 * (bedges[:-1] + bedges[1:])
for w in range(nw):
    others = np.delete(cnt, w, axis=0)
    diff = cnt[w] - np.median(others, axis=0)
    print(f"  w{w}: total {cnt[w].sum():.0f}; sum of (w - median others) {diff.sum():+.0f}; "
          f"bands with diff <= -2: {(diff <= -2).sum()}, >= +2: {(diff >= 2).sum()}")
    # by 1 mHz
    line = []
    for lo in range(0, 23):
        m = (bc >= lo) & (bc < lo + 1)
        if m.any() and abs(diff[m].sum()) >= 3:
            line.append(f"{lo}-{lo+1}:{diff[m].sum():+.0f}")
    print("      1-mHz bins with |sum diff| >= 3: " + (" ".join(line) if line else "none"))

# leaf matching across walkers (f0 = chain col 1 in mHz, amp = col 0)
leaves = []
for w in range(nw):
    m = ind[w]
    f0 = c[w, m, 1]
    amp = c[w, m, 0]
    snr = np.sqrt(np.maximum(hh[w, m], 0))
    o = np.argsort(f0)
    leaves.append((f0[o], amp[o], snr[o]))
TOL = 2.0e-4  # mHz
AR = 2.5


def unmatched(w, v):
    """Leaves of walker v with no partner on walker w (f0 within TOL, amp ratio within AR)."""
    fw, aw, _ = leaves[w]
    fv, av, sv = leaves[v]
    lo = np.searchsorted(fw, fv - TOL)
    hi = np.searchsorted(fw, fv + TOL)
    out = []
    for k in range(len(fv)):
        ok = False
        for j in range(lo[k], hi[k]):
            r = aw[j] / av[k]
            if 1.0 / AR <= r <= AR:
                ok = True
                break
        if not ok:
            out.append(k)
    return np.array(out, dtype=int)


print("\n== leaf matching: M[w, v] = leaves of v with NO partner on w ==")
M = np.zeros((nw, nw), dtype=int)
U = {}
for w in range(nw):
    for v in range(nw):
        if w == v:
            continue
        U[(w, v)] = unmatched(w, v)
        M[w, v] = len(U[(w, v)])
print(M)
for w in range(nw):
    # consensus-missing on w: leaves of some v unmatched on w, that ARE matched on at least one other walker u != w, v
    rows = []
    for v in range(nw):
        if v == w:
            continue
        fv, av, sv = leaves[v]
        idx = U[(w, v)]
        for k in idx:
            # matched on another walker u?
            found = 0
            for u in range(nw):
                if u in (w, v):
                    continue
                fu, au, _ = leaves[u]
                lo = np.searchsorted(fu, fv[k] - TOL)
                hi = np.searchsorted(fu, fv[k] + TOL)
                if any(1.0 / AR <= au[j] / av[k] <= AR for j in range(lo, hi)):
                    found += 1
            if found >= 1:
                rows.append((fv[k], sv[k], v))
    rows = np.array(rows) if rows else np.zeros((0, 3))
    # dedupe by f0 cluster
    if len(rows):
        rows = rows[np.argsort(rows[:, 0])]
        keep = [0]
        for i in range(1, len(rows)):
            if rows[i, 0] - rows[keep[-1], 0] > TOL:
                keep.append(i)
        rows = rows[keep]
    print(f"\n  w{w}: consensus leaves (present on >= 2 other walkers) MISSING on w{w}: {len(rows)}")
    if len(rows):
        snr = rows[:, 1]
        print(f"      SNR quantiles 10/50/90: {np.percentile(snr, [10, 50, 90])}; SNR>=10: {(snr >= 10).sum()}, >=15: {(snr >= 15).sum()}")
        hist = {}
        for f0_, s_, _ in rows:
            b = int(f0_)
            hist[b] = hist.get(b, 0) + 1
        print("      by 1-mHz bin: " + " ".join(f"{k}-{k+1}:{v}" for k, v in sorted(hist.items())))
        big = rows[snr >= 12]
        print("      SNR>=12 missing leaves (f0 mHz, SNR on the other walker): "
              + " ".join(f"{a:.4f}:{b:.1f}" for a, b, _ in big[:25]))
    # private leaves of w: unmatched on EVERY other walker
    fw, aw, sw = leaves[w]
    priv = np.ones(len(fw), dtype=bool)
    for v in range(nw):
        if v == w:
            continue
        um = set(U[(v, w)].tolist())
        priv &= np.array([k in um for k in range(len(fw))])
    print(f"  w{w}: PRIVATE leaves (no partner on any other walker): {priv.sum()}; SNR quantiles 10/50/90 "
          f"{np.percentile(sw[priv], [10, 50, 90]) if priv.any() else '-'}; SNR>=10: {(sw[priv] >= 10).sum()}")
    if priv.any():
        hist = {}
        for f0_ in fw[priv]:
            hist[int(f0_)] = hist.get(int(f0_), 0) + 1
        print("      by 1-mHz bin: " + " ".join(f"{k}-{k+1}:{v}" for k, v in sorted(hist.items())))
        big = np.flatnonzero(priv & (sw >= 12))
        print("      SNR>=12 private leaves (f0 mHz, SNR): " + " ".join(f"{fw[k]:.4f}:{sw[k]:.1f}" for k in big[:25]))

# ---- source frequency annotations
print("\n== source frequencies (for annotating the maps) ==")
MSUN_S = 4.925490947e-6
ce = g["chain/emri"][row, 0, 0]  # (nw, 8, 12)
for leaf in range(ce.shape[1]):
    x = ce[0, leaf]
    Mtot = np.exp(x[0])
    a, p0, e0 = x[2], x[3], x[4]
    forb = 1.0 / (2 * np.pi * MSUN_S * Mtot * (p0 ** 1.5 + a))
    print(f"  EMRI {leaf}: M {Mtot:.3g} mu {x[1]:.1f} a {a:.3f} p0 {p0:.2f} e0 {e0:.3f} -> f_orb {forb*1e3:.3f} mHz; "
          f"harmonics 2,3,4 f_orb = {2*forb*1e3:.2f} {3*forb*1e3:.2f} {4*forb*1e3:.2f} mHz (start of the data; rises)")
cs = g["chain/sobbh"][row, 0, 0]
print("  SOBBH start f (mHz) per leaf:", np.round(cs[0, :, 6] * 1e3, 2))

# ---- noise per walker over the last 60 rows (cold)
r0 = max(0, it - 60)
ps = g["sub_backend/psd/chain"][r0:it, 0, :, 0, :]  # (rows, nw, 2)
gf = g["sub_backend/galfor/chain"][r0:it, 0, :, 0, :]  # (rows, nw, 5)
print("\n== noise params per walker, mean +- std over the last 60 rows (cold rung) ==")
for w in range(nw):
    print(f"  w{w}: Sa {ps[:, w, 0].mean():.5e} +- {ps[:, w, 0].std():.1e}   Soms {ps[:, w, 1].mean():.5e} +- {ps[:, w, 1].std():.1e}"
          f"   galfor {gf[:, w].mean(axis=0)} +- {gf[:, w].std(axis=0)}")
f.close()
