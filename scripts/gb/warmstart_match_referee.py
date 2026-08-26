"""MATCH-STATISTIC REFEREE for the warm-start GB clustering.

Judges the components npz produced by ``warmstart_fit_from_store.py``
with actual waveform matches (existing machinery only -- stock GB
transform + gbgpu FD templates, per the overlap-validation pattern of
``gb_mojito_match.py``):

  1. WITHIN-GROUP COHERENCE: for each refereed component, real member
     rows (re-extracted from the store by nearest-centroid assignment
     in each component's own sigma-whitened space -- robust to the
     fitter script being edited in parallel) are matched against the
     member closest to the component mean ("anchor"). A good group has
     median member match ~1.
  2. MERGE TEST: for every same-island component pair with |delta f0|
     <= ``--pair-bins`` / Tobs, the centroid-vs-centroid match. High
     cross-match (> 0.9) with coherent groups = split artifact (should
     merge); low (< 0.5) = genuinely distinct.

Match metric: normalized phase-maximized overlap summed over the X/Y/Z
FD channels with FLAT weighting over the narrow band,
``|<a|b>| / sqrt(<a|a><b|b>)`` (fine for a same-band relative match --
the PSD is constant to <~1% over a <~0.2 uHz-wide template support).
Templates: ``GBGPU.run_wave`` (CPU backend) on the run's L1 orbits
(ICRS frame), t0 = data start, Tobs/dt from the run.

Laptop rules: run with
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
Peak RSS ~2.4 GB, transient, during the L1 orbit load (the ltt table
is materialized in full before trimming -- same peak the fit pays).

Usage:
  python warmstart_match_referee.py --npz <components.npz> --store <h5>
      [--members-per-comp 8] [--tail-n 200] [--p-core 0.5]
      [--pair-p-floor 0.2] [--pair-bins 4.0] [--flagship-f0 20.380]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

SCRATCH = ("/private/tmp/claude-501/-Users-mkatz-Research-lisa-sprint-2026/"
           "e6aff7a6-5694-4077-a87d-ab6a7e1eb360/scratchpad/warmstart")
MOJITO_GB_L1 = ("/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
                "data/GB/L1")
DATA_T0 = 97729089.327664  # mojito data start (gb_mojito_match.py REF)
CIRC = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}  # sampled-basis periods
ASSIGN_RADIUS = 6.0        # whitened-sigma junk cut for member assignment
N_CAP = 16384


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3


def circ_diff(x, m, period):
    return (x - m + period / 2.0) % period - period / 2.0


# ------------------------------------------------------------------
# member re-extraction: nearest centroid in per-component sigma space
# ------------------------------------------------------------------
def load_leaf_rows(store: str, last_k: int, last_safe: int = 471):
    """Cold-chain GB leaf rows over the last K written iterations.

    Mirrors warmstart_fit_from_store.load_leaf_table (READ-ONLY reuse of
    its conventions; not imported because that file is being edited in
    parallel). Returns (X (n, 9), sample_id (n,))."""
    import h5py

    with h5py.File(store, "r") as f:
        g = f["global_fit"] if "global_fit" in f else f["mcmc"]
        chain, inds, ll = g["chain"]["gb"], g["inds"]["gb"], g["log_like"]
        cold = (0, 0) if chain.ndim == 6 else (0,)
        nwalkers = chain.shape[-3]
        llv = ll[(slice(None),) + cold + (slice(None),)]
        written = np.flatnonzero(np.any(llv != 0.0, axis=-1))
        written = written[written <= last_safe][-last_k:]
        rows, sids = [], []
        for it in written:
            ind_it = inds[(it,) + cold]
            c_it = chain[(it,) + cold]
            w_idx, l_idx = np.nonzero(ind_it)
            rows.append(c_it[w_idx, l_idx])
            sids.append(np.int64(it) * nwalkers + w_idx)
    return np.concatenate(rows, axis=0), np.concatenate(sids, axis=0)


def assign_members(X, means, covs, island_id, f0_window_edges):
    """Nearest-centroid member assignment, per island, in each
    component's own sqrt(diag(cov))-whitened space (circular columns
    wrapped). Returns (assign (n,) int comp index or -1, dist (n,))."""
    n = len(X)
    assign = np.full(n, -1, dtype=np.int64)
    adist = np.full(n, np.inf)
    sig = np.sqrt(np.maximum(
        np.einsum("kii->ki", covs), 1e-30))          # (ncomp, 9)
    # island of each row via the window edges (windows are disjoint)
    lo, hi = f0_window_edges[:, 0], f0_window_edges[:, 1]
    win = np.searchsorted(lo, X[:, 1], side="right") - 1
    valid = (win >= 0) & (X[:, 1] < hi[np.clip(win, 0, len(hi) - 1)])
    for isl in np.unique(island_id):
        comp_sel = np.flatnonzero(island_id == isl)
        row_sel = np.flatnonzero(valid & (win == isl))
        if not len(row_sel):
            continue
        x = X[row_sel]                               # (m, 9)
        d2 = np.zeros((len(row_sel), len(comp_sel)))
        for jj, k in enumerate(comp_sel):
            diff = x - means[k]
            for col, period in CIRC.items():
                diff[:, col] = circ_diff(x[:, col], means[k][col], period)
            d2[:, jj] = np.sum((diff / sig[k]) ** 2, axis=1)
        best = d2.argmin(axis=1)
        bestd = np.sqrt(d2[np.arange(len(row_sel)), best])
        keep = bestd <= ASSIGN_RADIUS
        assign[row_sel[keep]] = comp_sel[best[keep]]
        adist[row_sel] = bestd
    return assign, adist


# ------------------------------------------------------------------
# waveforms + matches
# ------------------------------------------------------------------
def build_wave_gen(tobs: float):
    from lisatools.detector import L1Orbits
    from lisatools.globalfit.preprocessing import find_file
    from gbgpu.gbgpu import GBGPU

    orb = L1Orbits(find_file(MOJITO_GB_L1, "GB", 0),
                   force_backend="cpu", frame="icrs")
    pad = 1.0e5
    lt = np.asarray(orb.ltt_t)
    mk = (lt >= max(DATA_T0 - pad, float(orb.sc_t0))) & \
         (lt <= DATA_T0 + tobs + pad)
    orb.ltt = np.asarray(orb.ltt)[mk].copy()
    orb.ltt_t = lt[mk].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    orb.configure(linear_interp_setup=True)
    return GBGPU(force_backend="cpu", orbits=orb, t0=DATA_T0)


class Matcher:
    """Batch sampled-basis rows -> XYZ FD templates -> matches."""

    def __init__(self, gb, tobs, dt, oversample):
        from lisatools.globalfit.stock.erebor.transforms import (
            make_gb_transform_container,
        )

        self.gb, self.tobs, self.dt, self.oversample = gb, tobs, dt, oversample
        self.tc = make_gb_transform_container(
            use_chirp_mass=True, use_fdot_astro=True, use_distance=True)
        self.n_wf = 0

    def waves(self, rows_sampled: np.ndarray):
        """(m, 9) sampled rows -> (XYZf (m, 3, N) complex, kmin (m,))."""
        from gbgpu.utils.utility import get_N

        p = self.tc.both_transforms(np.atleast_2d(rows_sampled))
        N = int(min(N_CAP, np.max(get_N(
            p[:, 0], p[:, 1], self.tobs, oversample=self.oversample))))
        self.gb.run_wave(*[p[:, i] for i in range(9)],
                         N=N, T=self.tobs, dt=self.dt)
        self.n_wf += len(p)
        return np.asarray(self.gb.XYZf), np.asarray(self.gb.start_inds)

    @staticmethod
    def match(a, ka, b, kb):
        """Phase-maximized normalized XYZ overlap, flat weighting."""
        lo = max(ka, kb)
        hi = min(ka + a.shape[-1], kb + b.shape[-1])
        if hi <= lo:
            return 0.0
        num = np.sum(np.conj(a[:, lo - ka:hi - ka]) * b[:, lo - kb:hi - kb])
        den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
        return float(abs(num) / den) if den > 0 else 0.0


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", default=os.path.join(SCRATCH, "v5_last20.npz"))
    ap.add_argument("--store", default=os.path.join(
        SCRATCH, "gf_prod_3mo_v5", "gf_prod_3mo_testing.h5"))
    ap.add_argument("--tobs", type=float, default=None,
                    help="default: npz meta tobs")
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--members-per-comp", type=int, default=8)
    ap.add_argument("--tail-n", type=int, default=200,
                    help="random sample size of the p < p-core tail")
    ap.add_argument("--p-core", type=float, default=0.5)
    ap.add_argument("--pair-p-floor", type=float, default=0.2)
    ap.add_argument("--pair-bins", type=float, default=4.0)
    ap.add_argument("--flagship-f0", type=float, default=20.380,
                    help="mHz; island refereed in detail regardless of p")
    ap.add_argument("--max-wf", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None,
                    help="results npz (default <npz>_referee.npz)")
    args = ap.parse_args()
    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    z = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    means, covs, p, mult = z["means"], z["covs"], z["p"], z["mult"]
    n_members, island_id = z["n_members"], z["island_id"]
    edges = z["f0_window_edges"]
    tobs = args.tobs or float(meta["tobs"])
    df_mhz = float(meta["df_mhz"])
    ncomp = len(means)
    print(f"npz: {ncomp} comps | Tobs {tobs:.0f} s | df {df_mhz:.6g} mHz")

    # ---- member re-extraction ------------------------------------
    t0 = time.perf_counter()
    X, _sid = load_leaf_rows(args.store, int(meta["last_k"]))
    assign, _ad = assign_members(X, means, covs, island_id, edges)
    counts = np.bincount(assign[assign >= 0], minlength=ncomp)
    frac = counts / np.maximum(n_members, 1)
    print(f"leaf rows {len(X):,} | assigned {np.sum(assign >= 0):,} "
          f"({np.mean(assign >= 0):.1%}) | per-comp count vs npz "
          f"n_members: median ratio {np.median(frac):.2f} "
          f"[{time.perf_counter() - t0:.1f} s, RSS {rss_gb():.2f} GB]")

    # ---- referee sets --------------------------------------------
    core = np.flatnonzero(p > args.p_core)
    tail_all = np.flatnonzero((p <= args.p_core) & (counts >= 2))
    tail = rng.choice(tail_all, min(args.tail_n, len(tail_all)),
                      replace=False)
    # flagship island: highest-p comp within 6 bins of --flagship-f0
    near = np.flatnonzero(np.abs(means[:, 1] - args.flagship_f0)
                          < 6 * df_mhz)
    flag = int(near[np.argmax(p[near])]) if len(near) else -1
    flag_isl = island_id[flag] if flag >= 0 else -1
    referee = np.unique(np.concatenate(
        [core, tail, np.flatnonzero(island_id == flag_isl)]))
    referee = referee[counts[referee] >= 2]
    print(f"referee: {len(core)} core (p>{args.p_core}) + {len(tail)} tail "
          f"+ flagship island {flag_isl} (comp {flag}, "
          f"f0 {means[flag, 1]:.5f} mHz, p {p[flag]:.2f})" if flag >= 0 else
          f"referee: {len(core)} core + {len(tail)} tail (no flagship found)")

    # ---- same-island neighbor pairs ------------------------------
    pairs = []
    for isl in np.unique(island_id):
        sel = np.flatnonzero(island_id == isl)
        ok = (p[sel] > args.pair_p_floor) | (isl == flag_isl)
        sel = sel[ok]
        for a_i in range(len(sel)):
            for b_i in range(a_i + 1, len(sel)):
                i, j = sel[a_i], sel[b_i]
                if abs(means[i, 1] - means[j, 1]) <= args.pair_bins * df_mhz:
                    pairs.append((i, j))
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    print(f"pairs: {len(pairs)} same-island neighbors within "
          f"{args.pair_bins}/Tobs (p>{args.pair_p_floor} or flagship isl)")

    budget = len(referee) * (args.members_per_comp + 1) + 2 * len(pairs)
    if budget > args.max_wf:
        keep = (args.max_wf - 2 * len(pairs)) // (args.members_per_comp + 1)
        referee = rng.choice(referee, max(keep, 50), replace=False)
        print(f"BUDGET: subsampled referee comps to {len(referee)} "
              f"(est. {budget} > max {args.max_wf} waveforms)")

    # ---- waveform machinery --------------------------------------
    t0 = time.perf_counter()
    gb = build_wave_gen(tobs)
    mm = Matcher(gb, tobs, args.dt, args.oversample)
    print(f"wave gen ready [{time.perf_counter() - t0:.1f} s, "
          f"RSS {rss_gb():.2f} GB]")

    # ---- within-group coherence ----------------------------------
    t0 = time.perf_counter()
    K = args.members_per_comp
    med_match = np.full(ncomp, np.nan)
    min_match = np.full(ncomp, np.nan)
    med_ratio = np.full(ncomp, np.nan)   # measured / sinc-predicted
    n_sampled = np.zeros(ncomp, dtype=np.int64)
    sig = np.sqrt(np.maximum(np.einsum("kii->ki", covs), 1e-30))
    flag_detail = None
    for k in referee:
        ridx = np.flatnonzero(assign == k)
        # anchor: member closest to the component mean (whitened)
        diff = X[ridx] - means[k]
        for col, period in CIRC.items():
            diff[:, col] = circ_diff(X[ridx][:, col], means[k][col], period)
        d = np.sqrt(np.sum((diff / sig[k]) ** 2, axis=1))
        anchor = ridx[d.argmin()]
        others = ridx[ridx != anchor]
        n_take = min(K, len(others)) if k != flag else min(3 * K, len(others))
        take = rng.choice(others, n_take, replace=False)
        wf, kmin = mm.waves(X[np.concatenate(([anchor], take))])
        ms = np.array([mm.match(wf[0], kmin[0], wf[i], kmin[i])
                       for i in range(1, len(wf))])
        # sinc prediction: for a windowed sinusoid, an f0 offset of x
        # bins alone gives match ~ |sinc(x)| (np.sinc = sin(pi x)/(pi x)).
        # ratio ~1 -> the group's incoherence is fully explained by its
        # f0 spread (self-consistent posterior width / smooth blend);
        # ratio << 1 -> members also disagree in other params (junk).
        off_bins = (X[take, 1] - X[anchor, 1]) / df_mhz
        pred = np.maximum(np.abs(np.sinc(off_bins)), 1e-3)
        med_match[k] = np.median(ms)
        min_match[k] = ms.min()
        med_ratio[k] = np.median(np.minimum(ms / pred, 1.2))
        n_sampled[k] = len(ms)
        if k == flag:
            off = (X[take, 1] - means[k, 1]) / df_mhz  # 1/Tobs units
            # far-vs-near cross-check: match the two extreme-offset members
            lo_i, hi_i = int(off.argmin()), int(off.argmax())
            cross = mm.match(wf[1 + lo_i], kmin[1 + lo_i],
                             wf[1 + hi_i], kmin[1 + hi_i])
            flag_detail = dict(offsets=off, matches=ms, cross_extreme=cross,
                               off_lo=off[lo_i], off_hi=off[hi_i], pred=pred)
    print(f"within-group: {len(referee)} comps, {mm.n_wf} waveforms "
          f"[{time.perf_counter() - t0:.1f} s]")

    # ---- merge test ----------------------------------------------
    t0 = time.perf_counter()
    cross_match = np.zeros(len(pairs))
    for n_i, (i, j) in enumerate(pairs):
        wf, kmin = mm.waves(means[[i, j]])
        cross_match[n_i] = mm.match(wf[0], kmin[0], wf[1], kmin[1])
    print(f"merge test: {len(pairs)} pairs [{time.perf_counter() - t0:.1f} s]")

    # ---- report ---------------------------------------------------
    def bucket(vals):
        v = vals[np.isfinite(vals)]
        if not len(v):
            return "n/a"
        return (f"n={len(v)} med={np.median(v):.4f} | >0.99: "
                f"{np.mean(v > 0.99):.1%} | 0.9-0.99: "
                f"{np.mean((v > 0.9) & (v <= 0.99)):.1%} | <0.9: "
                f"{np.mean(v <= 0.9):.1%}")

    def consis(sel):
        v = med_ratio[sel]
        v = v[np.isfinite(v)]
        return f"{np.mean(v > 0.7):.1%}" if len(v) else "n/a"

    print("\n=== WITHIN-GROUP median member match (vs anchor) ===")
    print(f"  p>0.9      : {bucket(med_match[p > 0.9])} | "
          f"sinc-consistent: {consis(p > 0.9)}")
    print(f"  p 0.5-0.9  : {bucket(med_match[(p > 0.5) & (p <= 0.9)])} | "
          f"sinc-consistent: {consis((p > 0.5) & (p <= 0.9))}")
    print(f"  tail p<=0.5: {bucket(med_match[tail])} | "
          f"sinc-consistent: {consis(tail)}")
    print("  (sinc-consistent: median member match >= 0.7x the |sinc(df0)| "
          "prediction, i.e. the f0 spread explains the incoherence)")

    ref_ok = referee[np.isfinite(min_match[referee])]
    worst = ref_ok[np.argsort(med_ratio[ref_ok])][:10]
    print("\n  10 worst groups (by measured/sinc-predicted ratio):")
    print("   comp   f0[mHz]     p    mult  n_mem  sig_f0/df  med     min"
          "    ratio")
    for k in worst:
        print(f"   {k:5d} {means[k, 1]:10.5f} {p[k]:5.2f} {mult[k]:6.2f} "
              f"{n_members[k]:6d} {sig[k, 1] / df_mhz:8.2f} "
              f"{med_match[k]:7.4f} {min_match[k]:7.4f} {med_ratio[k]:7.3f}")

    print("\n=== MERGE TEST (centroid cross-matches) ===")
    hi_m = pairs[cross_match > 0.9]
    mid = pairs[(cross_match > 0.5) & (cross_match <= 0.9)]
    print(f"  >0.9 (merge candidates): {len(hi_m)} | 0.5-0.9: {len(mid)} | "
          f"<0.5 (distinct): {np.sum(cross_match <= 0.5)}")
    for i, j in hi_m:
        c = cross_match[np.all(pairs == [i, j], axis=1)][0]
        print(f"   MERGE {i}+{j}: f0 {means[i, 1]:.5f}/{means[j, 1]:.5f} "
              f"p {p[i]:.2f}/{p[j]:.2f} mult {mult[i]:.1f}/{mult[j]:.1f} "
              f"cross {c:.4f} coher {med_match[i]:.3f}/{med_match[j]:.3f}")
    for i, j in mid:
        c = cross_match[np.all(pairs == [i, j], axis=1)][0]
        print(f"   mid   {i}+{j}: f0 {means[i, 1]:.5f}/{means[j, 1]:.5f} "
              f"p {p[i]:.2f}/{p[j]:.2f} cross {c:.4f}")

    if flag >= 0:
        print(f"\n=== FLAGSHIP comp {flag} (f0 {means[flag, 1]:.5f} mHz, "
              f"p {p[flag]:.2f}, sig_f0 {sig[flag, 1] / df_mhz:.2f}/Tobs) ===")
        if flag_detail is not None:
            fd = flag_detail
            o, m = fd["offsets"], fd["matches"]
            print(f"  {len(m)} members vs anchor: med {np.median(m):.4f} "
                  f"min {m.min():.4f}")
            pr = fd["pred"]
            for lo, hi in [(-99, -1.5), (-1.5, -0.3), (-0.3, 0.3),
                           (0.3, 1.5), (1.5, 99)]:
                s = (o >= lo) & (o < hi)
                if s.any():
                    print(f"   off [{lo:+5.1f},{hi:+5.1f})/Tobs: n={s.sum()}"
                          f" med match {np.median(m[s]):.4f} "
                          f"min {m[s].min():.4f} "
                          f"(sinc pred {np.median(pr[s]):.4f})")
            print(f"  extreme-offset members ({fd['off_lo']:+.2f} vs "
                  f"{fd['off_hi']:+.2f} /Tobs) direct match: "
                  f"{fd['cross_extreme']:.4f}")
        fp = np.flatnonzero(np.any(pairs == flag, axis=1))
        for n_i in fp:
            i, j = pairs[n_i]
            o = j if i == flag else i
            print(f"  pair w/ comp {o} (p {p[o]:.2f}, "
                  f"df0 {(means[o, 1] - means[flag, 1]) / df_mhz:+.2f}/Tobs)"
                  f": cross-match {cross_match[n_i]:.4f}")

    out = args.out or os.path.splitext(args.npz)[0] + "_referee.npz"
    np.savez_compressed(
        out, referee=referee, med_match=med_match, min_match=min_match,
        med_ratio=med_ratio,
        n_sampled=n_sampled, pairs=pairs, cross_match=cross_match,
        assigned_counts=counts, flagship=flag,
        meta=json.dumps(dict(
            npz=args.npz, store=args.store, tobs=tobs, dt=args.dt,
            oversample=args.oversample, members_per_comp=K,
            tail_n=int(len(tail)), p_core=args.p_core,
            pair_p_floor=args.pair_p_floor, pair_bins=args.pair_bins,
            seed=args.seed, n_waveforms=mm.n_wf,
            weighting="flat-narrowband XYZ, phase-maximized")))
    print(f"\nDONE: {mm.n_wf} waveforms | wall "
          f"{time.perf_counter() - t_start:.1f} s | peak RSS "
          f"{rss_gb():.2f} GB | wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
