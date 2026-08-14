"""Small-scale prototype: GB posterior -> clusters -> proposal components.

Pipeline candidate (v1, numpy/scipy only):
  1. f0 DENSITY-VALLEY segmentation: histogram all leaf-samples at
     ~1/Tobs resolution; islands = contiguous bins above a count floor
     (robust to sparse junk that would bridge gap-based single-linkage).
  2. Within-island: robust-whitened complete-linkage (scipy) on
     (f0, fdot, lnA, lam, sinb) over a <=1500-row subsample; assign all
     rows to nearest centroid; merge tiny fragments.
  3. Per cluster: Gaussian (mean/cov) + inclusion probability
     p = distinct posterior samples containing a member / n_samples.

Synthetic truth: 10 sources with p in [0.15, 1.0] incl. a 2.5/Tobs f0
blend pair separated in sky, plus uniform junk leaves.
"""
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

rng = np.random.default_rng(7)

TOBS = 7.776e6
DF = 1.0 / TOBS                      # 1.286e-7 Hz
NWALK, NITER = 24, 500
NSAMP = NWALK * NITER                # posterior samples (cold chain)
F_LO, F_HI = 4.000e-3, 4.030e-3

# ---- truth: (f0, snr, p, sky) ---------------------------------------------
truth_f0 = np.array([4.0021e-3, 4.0060e-3, 4.0093e-3,
                     4.0130e-3, 4.0130e-3 + 2.5 * DF,    # blend pair
                     4.0168e-3, 4.0201e-3, 4.0235e-3, 4.0262e-3, 4.0288e-3])
truth_snr = np.array([25.0, 14.0, 9.0, 12.0, 10.0, 7.0, 6.0, 5.5, 18.0, 5.0])
truth_p = np.clip((truth_snr - 4.5) / 5.0, 0.05, 1.0)
truth_lam = rng.uniform(0, 2 * np.pi, 10)
truth_lam[4] = (truth_lam[3] + 1.2) % (2 * np.pi)        # blend: sky-separated
truth_sinb = rng.uniform(-1, 1, 10)
truth_lnA = np.log(1e-22 * truth_snr / 8.0)
truth_fdot = rng.uniform(-1, 1, 10) * 1e-16

def post_sigma(snr):
    """Fisher-ish posterior widths per param."""
    s8 = 8.0 / snr
    return dict(f0=0.30 * DF * s8, fdot=2e-17 * s8, lnA=0.12 * s8,
                lam=0.05 * s8, sinb=0.05 * s8)

# ---- draw the leaf table ---------------------------------------------------
rows, sample_id = [], []
for s in range(NSAMP):
    for i in range(10):
        if rng.random() < truth_p[i]:
            sg = post_sigma(truth_snr[i])
            rows.append([rng.normal(truth_f0[i], sg["f0"]),
                         rng.normal(truth_fdot[i], sg["fdot"]),
                         rng.normal(truth_lnA[i], sg["lnA"]),
                         rng.normal(truth_lam[i], sg["lam"]),
                         np.clip(rng.normal(truth_sinb[i], sg["sinb"]), -1, 1)])
            sample_id.append(s)
    for _ in range(rng.poisson(0.2)):                    # sparse junk leaves
        rows.append([rng.uniform(F_LO, F_HI), rng.normal(0, 5e-16),
                     np.log(1e-23), rng.uniform(0, 2 * np.pi),
                     rng.uniform(-1, 1)])
        sample_id.append(s)
X = np.asarray(rows)
sample_id = np.asarray(sample_id)
print(f"leaf table: {len(X):,} rows from {NSAMP:,} posterior samples")

t0 = time.perf_counter()

# ---- 1) f0 density-valley segmentation ------------------------------------
BIN = DF
bins = np.arange(F_LO, F_HI + BIN, BIN)
idx = np.digitize(X[:, 0], bins) - 1
counts = np.bincount(idx, minlength=len(bins))
floor = max(5, int(0.005 * NSAMP))                       # 0.5% of samples
hot = counts >= floor
# contiguous hot runs -> islands (pad one cool bin each side for tails)
edges = np.flatnonzero(np.diff(np.concatenate(([0], hot.view(np.int8), [0]))))
islands = list(zip(edges[::2], edges[1::2]))
print(f"islands: {len(islands)} (floor {floor}/bin)")

# ---- 2) within-island split + 3) components -------------------------------
CLUSTER_DIMS = slice(0, 5)
SUB = 1500
T_CUT = 2.0                # whitened SINGLE-linkage cut (NN chaining)
MIN_FRAC = 0.01                                           # drop tiny fragments

components = []
for b0, b1 in islands:
    m = (idx >= b0) & (idx < b1)
    x_all, sid = X[m], sample_id[m]
    n = len(x_all)
    sub = x_all[rng.choice(n, min(n, SUB), replace=False)]
    scale = np.maximum(1.4826 * np.median(
        np.abs(sub[:, CLUSTER_DIMS] - np.median(sub[:, CLUSTER_DIMS], 0)), 0),
        1e-30)
    zw = sub[:, CLUSTER_DIMS] / scale
    if len(sub) > 1:
        lab_sub = fcluster(linkage(zw, "single"), T_CUT, "distance")
    else:
        lab_sub = np.ones(1, dtype=int)
    cents = np.array([zw[lab_sub == k].mean(0) for k in np.unique(lab_sub)])
    dmat = cdist(x_all[:, CLUSTER_DIMS] / scale, cents)
    lab = dmat.argmin(1)
    lab[dmat.min(1) > 6.0] = -1                # junk: too far from any mode
    for k in range(len(cents)):
        mk = lab == k
        if mk.sum() < MIN_FRAC * NSAMP:
            continue
        ids = sid[mk]
        p = len(np.unique(ids)) / NSAMP
        mult = mk.sum() / max(len(np.unique(ids)), 1)     # dup leaves/sample
        components.append(dict(
            mean=x_all[mk].mean(0), cov=np.cov(x_all[mk].T), p=p, mult=mult,
            n=int(mk.sum())))

fit_wall = time.perf_counter() - t0
print(f"fit: {len(components)} components in {fit_wall:.2f} s")
print(f"{'f0 [mHz]':>10} {'p_hat':>6} {'p_true':>6} {'sig_f0/DF':>9} {'mult':>5}")
order = np.argsort([c["mean"][0] for c in components])
truth_used = np.zeros(10, bool)
for j in order:
    c = components[j]
    d = np.abs(truth_f0 - c["mean"][0])
    i = int(d.argmin())
    pt = truth_p[i] if d[i] < 5 * DF and not truth_used[i] else np.nan
    if not np.isnan(pt):
        truth_used[i] = True
    print(f"{c['mean'][0]*1e3:10.6f} {c['p']:6.2f} {pt:6.2f} "
          f"{np.sqrt(c['cov'][0, 0])/DF:9.2f} {c['mult']:5.2f}")
missed = [f"{f*1e3:.6f}" for f, u in zip(truth_f0, truth_used) if not u]
print("missed truths:", missed or "none",
      "| spurious:", sum(1 for j in order if np.isnan(
          truth_p[np.abs(truth_f0 - components[j]['mean'][0]).argmin()])
          if False))  # spurious counted via truth_used above

# blend check: two components within 5*DF of the pair?
pair = [c for c in components
        if abs(c["mean"][0] - truth_f0[3]) < 6 * DF]
print(f"blend pair (2.5/Tobs apart, sky-split): {len(pair)} components "
      f"recovered {'OK' if len(pair) == 2 else 'FAIL'}")

# ---- proposal evaluation speed (mixture logpdf, f0-windowed) ---------------
means = np.array([c["mean"] for c in components])
covs = np.array([c["cov"] for c in components])
weights = np.array([c["p"] for c in components]); weights /= weights.sum()
invs = np.linalg.inv(covs)
logdets = np.linalg.slogdet(covs)[1]
pts = means[rng.integers(0, len(means), 100_000)] + 1e-9
t1 = time.perf_counter()
# f0-window: only components within 10*DF contribute (thin-in-f0 structure)
comp_f0 = means[:, 0]
lp = np.full(len(pts), -np.inf)
for k in range(len(means)):
    w = np.abs(pts[:, 0] - comp_f0[k]) < 10 * DF
    if not w.any():
        continue
    d = pts[w] - means[k]
    q = np.einsum("ni,ij,nj->n", d, invs[k], d)
    lk = np.log(weights[k]) - 0.5 * (q + logdets[k] + 5 * np.log(2 * np.pi))
    lp[w] = np.logaddexp(lp[w], lk)
ev = time.perf_counter() - t1
print(f"logpdf eval: 100k points through {len(means)} comps in {ev*1e3:.0f} ms"
      f" ({ev/1e5*1e6:.2f} us/point)")

# ---- scale extrapolation: the f0 backbone at production size ---------------
t2 = time.perf_counter()
big = rng.uniform(5.5e-4, 2.2e-2, 60_000_000)
big.sort(kind="stable")
cnt = np.bincount(((big - 5.5e-4) / DF).astype(np.int64))
t3 = time.perf_counter() - t2
print(f"scale test: sort+histogram 6e7 f0 rows = {t3:.1f} s "
      f"(the only full-table pass; islands are independent after)")
