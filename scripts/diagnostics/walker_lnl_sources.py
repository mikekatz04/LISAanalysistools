"""Per-walker cold-chain lnL decomposition from a short-tar store extract.

usage: python low_walker.py <extract.h5> [nrows_back]
"""
import sys

import h5py
import numpy as np

np.set_printoptions(linewidth=220, precision=5, suppress=False)

p = sys.argv[1]
nback = int(sys.argv[2]) if len(sys.argv) > 2 else 60
f = h5py.File(p, "r")
g = f["global_fit"]
it = int(g.attrs["iteration"])
nw = int(g.attrs.get("nwalkers", 4))
print(f"iteration {it}  nwalkers {nw}")
print("attrs:", {k: (v if np.ndim(v) == 0 else np.shape(v)) for k, v in g.attrs.items()})


def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name:60s} {obj.shape} {obj.dtype}")


print("\n== datasets ==")
g.visititems(walk)

ll = g["log_like"][:it, 0, 0, :]  # (it, nw)
r0 = max(0, it - nback)
print(f"\n== cold lnL per walker, rows {r0}..{it-1} (gap = walker - row max) ==")
for r in range(r0, it):
    row = ll[r]
    print(f"row {r:4d} " + " ".join(f"{v:15.1f}" for v in row)
          + "   gaps " + " ".join(f"{v - row.max():9.1f}" for v in row))
gaps = ll[r0:it] - ll[r0:it].max(axis=1, keepdims=True)
print("mean gap per walker over these rows:", gaps.mean(axis=0))
print("rank order (0 = best) per row, last 20:")
print(np.argsort(np.argsort(-ll[max(0, it - 20):it], axis=1), axis=1))

chain = g["chain"]
inds = g["inds"]
sub = g["sub_backend"] if "sub_backend" in g else None

for br in ("mbh", "emri", "sobbh", "vgb"):
    if br not in chain:
        continue
    c = chain[br]
    ind = inds[br]
    last = c[it - 1, 0, 0]  # (nw, nleaves, ndim)
    ilast = ind[it - 1, 0, 0]
    hh = None
    if sub is not None and br in sub and "h_h" in sub[br]:
        hh = sub[br]["h_h"]
    print(f"\n== {br}: chain {c.shape}; h_h {None if hh is None else hh.shape} ==")
    for leaf in range(last.shape[1]):
        if not ilast[:, leaf].any():
            continue
        line = f"  leaf {leaf}: inds {ilast[:, leaf].astype(int)}"
        if hh is not None:
            snr_now = np.sqrt(np.maximum(hh[it - 1, :, leaf], 0.0))
            hist = np.sqrt(np.maximum(hh[r0:it, :, leaf], 0.0))
            line += "  SNR now " + " ".join(f"{v:7.2f}" for v in snr_now)
            line += "  | SNR min..max over rows: " + " ".join(
                f"{hist[:, w].min():.1f}..{hist[:, w].max():.1f}" for w in range(nw))
        print(line)
        # parameter spread across walkers in units of the cross-walker std
        x = last[:, leaf, :]
        sd = x.std(axis=0)
        med = np.median(x, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (x - med) / sd
        for w in range(nw):
            flag = " <-- outlier" if np.nanmax(np.abs(z[w])) > 1.4 else ""
            print(f"     w{w}: " + " ".join(f"{v:12.6g}" for v in x[w]) + flag)

for br in ("psd", "galfor"):
    if br not in chain:
        continue
    last = chain[br][it - 1, 0, 0]
    print(f"\n== {br} per walker (last row) ==")
    for w in range(nw):
        print(f"  w{w}: " + " ".join(f"{v:12.6g}" for v in last[w, 0]))

if "gb" in chain:
    ind = inds["gb"][it - 1, 0, 0]
    print(f"\n== gb leaves per walker (last row): {ind.sum(axis=1)} ==")
    hist_leaves = inds["gb"][r0:it, 0, 0].sum(axis=2)
    print("  leaves over rows (first/last):", hist_leaves[0], hist_leaves[-1])

if sub is not None and "gb" in sub:
    sb = sub["gb"]
    print("\n== gb sub_backend datasets ==")
    for k in sb:
        print(f"  {k:40s} {sb[k].shape} {sb[k].dtype}")
    for key in ("band_cold_logl_w", "band_cold_logl_max_w", "band_cold_logl_peak_w"):
        if key not in sb:
            continue
        arr = sb[key]
        B = arr[it - 1] if arr.ndim == 3 else arr[...]
        B = np.asarray(B, dtype=float)  # (nw, nbands)
        finite = np.isfinite(B).all(axis=0)
        print(f"\n  {key}: shape {B.shape}, finite bands {finite.sum()}")
        tot = np.where(finite[None, :], B, 0.0).sum(axis=1)
        print("   sum over finite bands per walker:", tot, " gap vs max:", tot - tot.max())
        Bm = B[:, finite]
        gapb = Bm - Bm.max(axis=0, keepdims=True)  # (nw, nb)
        bidx = np.flatnonzero(finite)
        for w in range(nw):
            order = np.argsort(gapb[w])[:12]
            print(f"   w{w} total deficit {gapb[w].sum():9.1f}; worst 12 bands (band:gap): "
                  + " ".join(f"{bidx[j]}:{gapb[w, j]:.0f}" for j in order))
        break

# band-> frequency map if present
for key in ("band_edges", "band_f_lo", "band_f_hi", "band_centers"):
    if sub is not None and "gb" in sub and key in sub["gb"]:
        v = sub["gb"][key][...]
        print(f"\n  {key}: shape {v.shape} first {v.ravel()[:4]} last {v.ravel()[-4:]}")
f.close()
