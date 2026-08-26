"""Warm-start GB proposal fitter: previous-run store -> Gaussian components npz.

Workstream B (docs/6mo-run-prep.md) real-data builder implementing the
3-stage pipeline of docs/warm-start-gb-proposal.md on a finished run's
cold-chain leaf table (validated prototype: proto_warmstart_cluster.py):

  1. f0 DENSITY-VALLEY segmentation at 1/Tobs bins with a count floor
     (0.5% of posterior samples per bin); islands padded by one bin.
  2. Within-island split (SWAPPABLE strategy, ``split(island_rows) ->
     labels``): robust-MAD-whiten (f0, Mc, ln dist, alpha, sin_delta),
     SINGLE-linkage on a <=1500-row subsample cut at 2.0 whitened units,
     nearest-centroid assignment with junk radius 6 (label -1), plus the
     v1 satellite-fragment merge pass.
  3. Cluster -> component: Gaussian mean/cov over the full 9-col sampled
     basis with CIRCULAR handling for phi0/psi/alpha (v1) and covariance
     eigenvalue floors (v1); inclusion probability
     p = distinct posterior samples containing a member / n_samples,
     and leaf multiplicity mult = members / distinct samples.

Waveform-free by design: numpy/scipy/h5py only, no lisatools import.
CPU budget: run with OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OPENBLAS_NUM_THREADS=1. Reads the chain iteration-by-iteration (streamed).

Sampled basis (verified against stock/erebor/gb.py init_sampling_info and
the store's column ranges; the store keeps NO column names itself):
  0 dist [kpc], 1 f0 [mHz], 2 Mc [Msol], 3 phi0 [rad, 2pi], 4 cos_iota,
  5 psi [rad, pi], 6 alpha [rad, 2pi], 7 sin_delta, 8 fdot_astro_ratio.

Usage:
  python warmstart_fit_from_store.py --store <h5> [--last-k N]
      [--tobs 7776000] [--out components.npz]
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time

import h5py
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist

COLUMN_NAMES = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
                "sin_delta", "fdot_astro_ratio"]
# circular columns in the sampled basis: column -> period
CIRCULAR_COLS = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}
# cluster-feature space: (f0 [mHz], Mc, ln dist, alpha, sin_delta)
FEAT_NAMES = ["f0", "Mc", "ln_dist", "alpha", "sin_delta"]

# Stage-1 valley split (2026-08-24 fix): at final leaf density (~900
# leaves/walker) the confusion band is CONTINUOUSLY occupied above the
# global count floor, so floor-only segmentation returned ONE island for
# the whole band (and one useless component). Islands are now recursively
# split at genuine density valleys, and islands wider than
# MAX_ISLAND_BINS are force-split at their weakest interior bin — a
# multi-thousand-row blended island is unresolvable by the 5-D subsample
# linkage anyway.
VALLEY_FRAC = 0.35                 # interior min <= frac * smaller flanking peak
MAX_ISLAND_BINS = 64               # force-split wider islands (64 x 1/Tobs)
SUB = 1500                         # linkage subsample size
T_CUT = 2.0                        # single-linkage cut, whitened units
JUNK_RADIUS = 6.0                  # nearest-centroid junk exclusion
SAT_MERGE_CUT = 2.0                # centroid merge distance (any pair)
SAT_FRAC = 0.05                    # satellite: n < frac * n_big ...
SAT_RADIUS = 4.0                   # ... and centroid within this radius
MIN_FRAC = 0.01                    # min members as fraction of n_samples
CORR_EIG_FLOOR = 1e-4              # eigenvalue floor on the correlation mat


def rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024 ** 3  # macOS: bytes


# --------------------------------------------------------------------------
# stage 0: streamed cold-chain leaf-table extraction
# --------------------------------------------------------------------------
def load_leaf_table(store: str, last_k: int | None, max_iter: int | None = None):
    """Read the cold-chain GB leaf table over the last K stored iterations.

    Returns (X (n, 9) float64, sample_id (n,) int64, info dict).
    sample_id = stored_iteration_index * nwalkers + walker.
    Handles both the gf_format_version-2 layout
    [it, nsamplers, ntemps, nwalkers, nleaves, ndim] and the legacy
    [it, ntemps, nwalkers, nleaves, ndim] one (top group "mcmc").
    """
    with h5py.File(store, "r") as f:
        g = f["global_fit"] if "global_fit" in f else f["mcmc"]
        chain = g["chain"]["gb"]
        inds = g["inds"]["gb"]
        ll = g["log_like"]
        six_d = chain.ndim == 6  # extra nsamplers axis
        cold = (0, 0) if six_d else (0,)
        nwalkers = chain.shape[-3]
        ndim = chain.shape[-1]

        # written iterations: any walker with a nonzero cold-chain log_like
        llv = ll[(slice(None),) + cold + (slice(None),)]
        written = np.flatnonzero(np.any(llv != 0.0, axis=-1))
        if max_iter is not None:
            written = written[written <= max_iter]
        # TORN-TAIL GUARD (replaces the old hardcoded 471 clamp, which
        # silently truncated any store past that iteration): a crash
        # mid-save can leave log_like written while the coord slab is
        # still zeros (the v5 it=472 signature). Drop trailing written
        # iterations whose coords are entirely zero.
        while len(written) and not chain[(written[-1],) + cold][:8].any():
            print(f"  dropping torn tail iteration {written[-1]} "
                  "(log_like written, coord slab empty)")
            written = written[:-1]
        if last_k is not None:
            written = written[-last_k:]

        rows, sids = [], []
        zero_iters = 0
        for it in written:
            ind_it = inds[(it,) + cold]              # (nwalkers, nleaves)
            c_it = chain[(it,) + cold]               # (nwalkers, nleaves, ndim)
            w_idx, l_idx = np.nonzero(ind_it)
            r = c_it[w_idx, l_idx]
            # KEEP-WINDOW GUARD (2026-08-24): snapshot EXTRACT h5s carry
            # FULL inds but coords only for the extractor's keep window
            # (--keep default 3) -- alive-flagged rows outside it read as
            # ZEROS and poisoned a fit with a 1.08M-row f0=0 mega
            # component. A physical GB row can never have f0 == 0 (band
            # floor 0.5556 mHz), so drop zero-coord rows and count the
            # iterations they emptied.
            good = r[:, 1] > 0.0
            if not good.all():
                if not good.any():
                    zero_iters += 1
                    continue
                r, w_idx = r[good], w_idx[good]
            rows.append(r)
            sids.append(np.int64(it) * nwalkers + w_idx)
        if zero_iters:
            print(f"  WARNING: {zero_iters}/{len(written)} requested "
                  "iterations have EMPTY coord slabs (keep-window extract?)"
                  " -- effective window is only the remainder.")
            written = np.array([int(s[0]) // nwalkers for s in sids])
        X = np.concatenate(rows, axis=0)
        sample_id = np.concatenate(sids, axis=0)
        info = dict(
            iterations=[int(written.min()), int(written.max())],
            n_iterations=int(len(written)),
            nwalkers=int(nwalkers), ndim=int(ndim),
            n_samples=int(len(written) * nwalkers),
            leaves_per_walker=float(len(X) / (len(written) * nwalkers)),
            store_iteration_attr=int(g.attrs.get("iteration", -1)),
        )
    return X, sample_id, info


# --------------------------------------------------------------------------
# stage 1: f0 density-valley segmentation
# --------------------------------------------------------------------------
def segment_f0(f0_mhz: np.ndarray, df_mhz: float, n_samples: int):
    """Islands = contiguous 1/Tobs bins above the count floor, padded by 1.

    Returns (bin_index_per_row, island list [(b0, b1) half-open bins],
    f_edge0, floor).
    """
    f_lo = np.floor(f0_mhz.min() / df_mhz) * df_mhz
    idx = ((f0_mhz - f_lo) / df_mhz).astype(np.int64)
    counts = np.bincount(idx)
    floor = max(5, int(0.005 * n_samples))
    hot = counts >= floor
    edges = np.flatnonzero(np.diff(np.concatenate(
        ([0], hot.view(np.int8), [0]))))
    islands = [(max(b0 - 1, 0), min(b1 + 1, len(counts)))     # 1-bin pad
               for b0, b1 in zip(edges[::2], edges[1::2])]
    merged = []
    for b0, b1 in islands:                                    # merge touching
        if merged and b0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b1)
        else:
            merged.append([b0, b1])
    # VALLEY SPLIT (2026-08-24): a global floor is not a valley detector
    # once the confusion band is continuously occupied (final v5 density
    # returned ONE island for the whole band). Recursively split each hot
    # run at interior bins that are genuine density valleys (below
    # VALLEY_FRAC of the smaller flanking peak, or below the floor), and
    # FORCE-split anything still wider than MAX_ISLAND_BINS at its weakest
    # interior bin -- blended-confusion islands beyond that width cannot
    # be resolved by the 5-D subsample linkage downstream anyway.
    final = []
    stack = [tuple(m) for m in merged]
    while stack:
        b0, b1 = stack.pop()
        width = b1 - b0
        seg = counts[b0:b1]
        if width <= 3:
            final.append((b0, b1))
            continue
        interior = seg[1:-1]
        m = int(np.argmin(interior)) + 1          # weakest interior bin
        left_peak = int(seg[:m].max())
        right_peak = int(seg[m + 1:].max())
        is_valley = (seg[m] <= max(floor, VALLEY_FRAC
                                   * min(left_peak, right_peak))
                     and left_peak >= floor and right_peak >= floor)
        if is_valley or width > MAX_ISLAND_BINS:
            stack.append((b0, b0 + m + 1))        # valley bin rides left
            stack.append((b0 + m + 1, b1))
        else:
            final.append((b0, b1))
    final.sort()
    return idx, final, f_lo, floor


# --------------------------------------------------------------------------
# stage 2: within-island split (swappable: split(island_rows) -> labels)
# --------------------------------------------------------------------------
def make_cluster_features(x_all: np.ndarray) -> np.ndarray:
    """(n, 9) sampled rows -> (n, 5) cluster features, alpha rotated so the
    2pi wrap sits in the emptiest region of the island's alpha histogram."""
    alpha = x_all[:, 6]
    hist = np.bincount((alpha / (2 * np.pi) * 36).astype(int) % 36,
                       minlength=36)
    shift = (int(hist.argmin()) + 0.5) * (2 * np.pi / 36)
    alpha_rot = (alpha - shift) % (2 * np.pi)
    return np.column_stack([x_all[:, 1], x_all[:, 2],
                            np.log(np.maximum(x_all[:, 0], 1e-30)),
                            alpha_rot, x_all[:, 7]])


def _satellite_merge(labels, zw, stats):
    """v1 refinement: merge satellite fragments in whitened space.

    Any centroid pair closer than SAT_MERGE_CUT merges; a cluster smaller
    than SAT_FRAC of a bigger one merges into it within SAT_RADIUS."""
    for _ in range(5):
        ks = np.unique(labels[labels >= 0])
        if len(ks) < 2:
            break
        cents = np.array([zw[labels == k].mean(0) for k in ks])
        sizes = np.array([(labels == k).sum() for k in ks])
        d = cdist(cents, cents)
        parent = np.arange(len(ks))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        merged_any = False
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                close = d[i, j] < SAT_MERGE_CUT
                sat = (min(sizes[i], sizes[j])
                       < SAT_FRAC * max(sizes[i], sizes[j])
                       and d[i, j] < SAT_RADIUS)
                if close or sat:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
                        merged_any = True
                        stats["satellite_merges"] += 1
        if not merged_any:
            break
        remap = {k: ks[find(i)] for i, k in enumerate(ks)}
        labels = np.array([remap[k] if k >= 0 else -1 for k in labels])
    # relabel compactly
    ks = np.unique(labels[labels >= 0])
    lut = {k: i for i, k in enumerate(ks)}
    return np.array([lut[k] if k >= 0 else -1 for k in labels])


def split_single_linkage(island_rows: np.ndarray, rng: np.random.Generator,
                         df_mhz: float, stats: dict) -> np.ndarray:
    """Default swappable splitter: split(island_rows) -> labels (-1 = junk).

    island_rows: (n, 9) sampled-basis rows of ONE island.
    """
    feats = make_cluster_features(island_rows)
    n = len(feats)
    sub = feats[rng.choice(n, min(n, SUB), replace=False)]
    med = np.median(sub, axis=0)
    mad = 1.4826 * np.median(np.abs(sub - med), axis=0)
    # column-aware scale floors (a zero MAD must not shatter the island)
    scale_floor = np.array([0.05 * df_mhz, 1e-4, 1e-3, 1e-3, 1e-3])
    scale = np.maximum(mad, scale_floor)
    zw_sub = (sub - med) / scale
    if len(zw_sub) > 1:
        lab_sub = fcluster(linkage(pdist(zw_sub), "single"), T_CUT,
                           "distance")
    else:
        lab_sub = np.ones(1, dtype=int)
    cents = np.array([zw_sub[lab_sub == k].mean(0)
                      for k in np.unique(lab_sub)])
    zw_all = (feats - med) / scale
    dmat = cdist(zw_all, cents)
    labels = dmat.argmin(1)
    labels[dmat.min(1) > JUNK_RADIUS] = -1
    return _satellite_merge(labels, zw_all, stats)


# --------------------------------------------------------------------------
# stage 3: cluster -> Gaussian component (circular params + eigval floors)
# --------------------------------------------------------------------------
def circular_wrap(x: np.ndarray, period: float) -> np.ndarray:
    """Wrap samples of a circular parameter around their circular mean."""
    ang = x * (2 * np.pi / period)
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    m *= period / (2 * np.pi)
    return (x - m + period / 2.0) % period + m - period / 2.0


def fit_component(rows: np.ndarray, stats: dict):
    """(m, 9) member rows -> (mean, cov) with circular phi0/psi/alpha and
    eigenvalue-floored covariance (v1 refinements)."""
    x = rows.copy()
    for col, period in CIRCULAR_COLS.items():
        x[:, col] = circular_wrap(x[:, col], period)
    mean = x.mean(0)
    cov = np.atleast_2d(np.cov(x.T))
    # principal-range means for circular columns
    for col, period in CIRCULAR_COLS.items():
        mean[col] = mean[col] % period
    # eigenvalue floors: floor the diagonal, then the correlation spectrum
    # last-resort scales, well below any physical posterior width:
    # dist 1e-4 relative, f0 0.02/Tobs, Mc 1e-5 Msol, angles/unitless 1e-3
    diag_floor = np.array([1e-4 * max(abs(mean[0]), 1e-3), 0.0, 1e-5,
                           1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]) ** 2
    diag_floor[1] = (0.02 * stats["df_mhz"]) ** 2
    d2 = np.diag(cov).copy()
    n_diag = int((d2 < diag_floor).sum())
    d2 = np.maximum(d2, diag_floor)
    d = np.sqrt(d2)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    w, v = np.linalg.eigh(corr)
    n_eig = int((w < CORR_EIG_FLOOR).sum())
    if n_diag or n_eig:
        stats["cov_floor_triggers"] += 1
        stats["cov_floor_diag"] += n_diag
        stats["cov_floor_eig"] += n_eig
    w = np.maximum(w, CORR_EIG_FLOOR)
    corr = (v * w) @ v.T
    cov = corr * np.outer(d, d)
    return mean, cov


# --------------------------------------------------------------------------
def run(store: str, last_k: int | None, tobs: float, out: str,
        split_fn=split_single_linkage, seed: int = 7,
        max_iter: int | None = None):
    rng = np.random.default_rng(seed)
    df_mhz = 1.0 / tobs * 1e3          # 1/Tobs in mHz (stored f0 is mHz)
    walls = {}

    t0 = time.perf_counter()
    X, sample_id, info = load_leaf_table(store, last_k, max_iter=max_iter)
    walls["load"] = time.perf_counter() - t0
    n_samples = info["n_samples"]
    print(f"leaf table: {len(X):,} rows, {n_samples:,} posterior samples "
          f"(its {info['iterations'][0]}..{info['iterations'][1]}, "
          f"{info['nwalkers']} walkers, "
          f"{info['leaves_per_walker']:.1f} leaves/walker) "
          f"[{walls['load']:.1f} s, RSS {rss_gb():.2f} GB]")

    t0 = time.perf_counter()
    bin_idx, islands, f_lo, floor = segment_f0(X[:, 1], df_mhz, n_samples)
    walls["segment"] = time.perf_counter() - t0
    print(f"stage 1: {len(islands)} islands (floor {floor}/bin, "
          f"df {df_mhz:.6g} mHz) [{walls['segment']:.2f} s]")

    stats = dict(satellite_merges=0, cov_floor_triggers=0,
                 cov_floor_diag=0, cov_floor_eig=0, df_mhz=df_mhz,
                 junk_rows=0, orphan_rows=0, dropped_fragments=0,
                 dropped_fragment_rows=0)
    in_island = np.zeros(len(X), dtype=bool)

    means, covs, ps, mults, ns, isl_id = [], [], [], [], [], []
    t0 = time.perf_counter()
    t_split_total = 0.0
    for isl, (b0, b1) in enumerate(islands):
        m = (bin_idx >= b0) & (bin_idx < b1)
        in_island |= m
        x_all, sid = X[m], sample_id[m]
        ts = time.perf_counter()
        labels = split_fn(x_all, rng, df_mhz, stats)
        t_split_total += time.perf_counter() - ts
        stats["junk_rows"] += int((labels == -1).sum())
        for k in range(labels.max() + 1 if labels.size else 0):
            mk = labels == k
            nk = int(mk.sum())
            if nk < max(3, MIN_FRAC * n_samples):
                stats["dropped_fragments"] += 1
                stats["dropped_fragment_rows"] += nk
                continue
            ids = np.unique(sid[mk])
            mean, cov = fit_component(x_all[mk], stats)
            means.append(mean)
            covs.append(cov)
            ps.append(len(ids) / n_samples)
            mults.append(nk / len(ids))
            ns.append(nk)
            isl_id.append(isl)
    walls["split"] = t_split_total
    walls["components"] = time.perf_counter() - t0 - t_split_total
    stats["orphan_rows"] = int((~in_island).sum())

    means = np.array(means)
    covs = np.array(covs)
    ps = np.array(ps)
    mults = np.array(mults)
    ns = np.array(ns, dtype=np.int64)
    isl_id = np.array(isl_id, dtype=np.int64)
    f0_window_edges = np.array(
        [[f_lo + b0 * df_mhz, f_lo + b1 * df_mhz] for b0, b1 in islands])

    order = np.argsort(means[:, 1])
    means, covs, ps, mults, ns, isl_id = (
        means[order], covs[order], ps[order], mults[order], ns[order],
        isl_id[order])

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd="/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools",
        ).stdout.strip()
    except Exception:
        git_head = "unknown"

    meta = dict(
        store=store, tobs=tobs, df_mhz=df_mhz, last_k=last_k,
        column_names=COLUMN_NAMES, f0_units="mHz",
        circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()},
        sample_id_def="stored_iteration_index * nwalkers + walker",
        git_head=git_head, seed=seed,
        pipeline="density-valley + single-linkage + satellite-merge v1",
        knobs=dict(SUB=SUB, T_CUT=T_CUT, JUNK_RADIUS=JUNK_RADIUS,
                   SAT_MERGE_CUT=SAT_MERGE_CUT, SAT_FRAC=SAT_FRAC,
                   SAT_RADIUS=SAT_RADIUS, MIN_FRAC=MIN_FRAC,
                   CORR_EIG_FLOOR=CORR_EIG_FLOOR),
        **info, **{k: v for k, v in stats.items() if k != "df_mhz"},
        walls={k: round(v, 3) for k, v in walls.items()},
    )
    t0 = time.perf_counter()
    np.savez_compressed(
        out, means=means, covs=covs, p=ps, mult=mults, n_members=ns,
        island_id=isl_id, f0_window_edges=f0_window_edges,
        meta=json.dumps(meta))
    walls["write"] = time.perf_counter() - t0

    total_rows = len(X)
    print(f"stage 2: split {walls['split']:.1f} s | stage 3: components "
          f"{walls['components']:.1f} s | write {walls['write']:.2f} s")
    print(f"components: {len(means)} | junk rows "
          f"{stats['junk_rows']:,} ({stats['junk_rows']/total_rows:.2%}) | "
          f"orphan rows (outside islands) {stats['orphan_rows']:,} "
          f"({stats['orphan_rows']/total_rows:.2%}) | dropped fragments "
          f"{stats['dropped_fragments']} ({stats['dropped_fragment_rows']:,}"
          f" rows) | satellite merges {stats['satellite_merges']} | "
          f"cov floors {stats['cov_floor_triggers']} comps "
          f"(diag {stats['cov_floor_diag']}, eig {stats['cov_floor_eig']})")
    print(f"p: sum {ps.sum():.1f} | >0.9: {(ps > 0.9).sum()} | 0.5-0.9: "
          f"{((ps >= 0.5) & (ps <= 0.9)).sum()} | <0.5: {(ps < 0.5).sum()}")
    print(f"peak RSS {rss_gb():.2f} GB | wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="previous-run h5 store")
    ap.add_argument("--last-k", type=int, default=None,
                    help="last K stored iterations (default: all written; "
                         "torn trailing iterations auto-dropped)")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="ignore stored iterations beyond this index "
                         "(default: none — the torn-tail guard handles "
                         "crash-truncated stores)")
    ap.add_argument("--tobs", type=float, default=7776000.0,
                    help="observation time [s] (default 3 mo)")
    ap.add_argument("--out", default="warmstart_components.npz")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(args.store, args.last_k, args.tobs, args.out, seed=args.seed,
        max_iter=args.max_iter)


if __name__ == "__main__":
    main()
