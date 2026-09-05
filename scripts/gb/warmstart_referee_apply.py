"""Stage 2.5 of the warm-start pipeline: apply the match referee's verdict.

Consumes the fitter npz (``warmstart_fit_from_store.py``) and the referee
npz (``warmstart_match_referee.py``) and writes the REFEREED components
npz that production arms via ``GB_WARM_START_COMPONENTS``
(docs/6mo-run-prep.md: "point GB_WARM_START_COMPONENTS at a refereed
final-store npz"). Thresholds are the 2026-08-24 referee-verdict rulings:

* AUTO-MERGE same-island pairs with centroid cross-match > ``merge_cut``
  (0.9) — split artifacts of one source. Moment-matched Gaussian merge
  with weights ~ p; circular columns (phi0/psi/alpha) merged via the
  minimal image around the highest-p member; merged
  ``p = min(1, sum p_i)`` (fragments of one source occupy different
  posterior samples), members summed, mult p-weighted.
* FLAG BLENDS: refereed comps with ``p > blend_p`` (0.5) and coherence
  ``med_ratio < blend_ratio`` (0.5) — mosaics absorbing >1 real source.
  KEPT by default (they still seed births near real power; the new run's
  RJ resolves them); ``--drop-blends`` removes them. The ``blend`` bool
  column rides in the output either way.
* Pairs in (0.5, merge_cut] and everything un-refereed pass through
  untouched.

The output keeps the exact fitter writer schema (plus ``blend``,
``med_match``, ``med_ratio`` diagnostic columns and a ``referee_apply``
meta block), so ``WarmStartComponents.from_npz`` loads it unchanged.

Usage:
  python warmstart_referee_apply.py --fit <components.npz>
      --referee <referee.npz> --out <refereed.npz> [--drop-blends]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# sampled-basis circular columns (lockstep with the fitter/proposal)
CIRCULAR_COLS = {3: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}

MERGE_CUT = 0.9
BLEND_P = 0.5
BLEND_RATIO = 0.5


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _merge_group(idx, means, covs, p, mult, n_members):
    """Moment-matched Gaussian merge of components ``idx`` (weights ~ p)."""
    w = p[idx] / p[idx].sum()
    base = idx[int(np.argmax(p[idx]))]
    mu_shift = means[idx].copy()
    # minimal image of every circular mean around the highest-p member's
    for c, period in CIRCULAR_COLS.items():
        d = mu_shift[:, c] - means[base, c]
        mu_shift[:, c] = means[base, c] + d - period * np.round(d / period)
    mu = np.einsum("k,kd->d", w, mu_shift)
    cov = np.zeros_like(covs[0])
    for k, i in enumerate(idx):
        d = mu_shift[k] - mu
        cov += w[k] * (covs[i] + np.outer(d, d))
    for c, period in CIRCULAR_COLS.items():
        mu[c] = mu[c] % period
    return (mu, cov, min(1.0, float(p[idx].sum())),
            float(np.einsum("k,k->", w, mult[idx])),
            int(n_members[idx].sum()), base)


def apply(fit_npz: str, referee_npz: str, out: str,
          merge_cut: float = MERGE_CUT, blend_p: float = BLEND_P,
          blend_ratio: float = BLEND_RATIO, drop_blends: bool = False) -> str:
    with np.load(fit_npz, allow_pickle=False) as d:
        means = np.array(d["means"])
        covs = np.array(d["covs"])
        p = np.array(d["p"])
        mult = np.array(d["mult"])
        n_members = np.array(d["n_members"])
        island_id = np.array(d["island_id"])
        f0_window_edges = np.array(d["f0_window_edges"])
        meta = json.loads(str(d["meta"]))
    with np.load(referee_npz, allow_pickle=False) as r:
        pairs = np.array(r["pairs"]).reshape(-1, 2)
        cross = np.array(r["cross_match"]).ravel()
        med_ratio = np.array(r["med_ratio"])
        med_match = np.array(r["med_match"])
    n = len(p)
    if med_ratio.shape != (n,):
        raise ValueError(
            f"referee med_ratio has {med_ratio.shape}, fit has {n} comps -- "
            "the referee npz was built from a DIFFERENT fit npz.")

    # --- auto-merges (union-find; transitive chains collapse together) ---
    parent = list(range(n))
    for (i, j), cm in zip(pairs, cross):
        if cm > merge_cut:
            ri, rj = _find(parent, int(i)), _find(parent, int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([_find(parent, i) for i in range(n)])

    keep_rows = []
    n_merged_groups = 0
    for root in np.unique(roots):
        idx = np.flatnonzero(roots == root)
        if len(idx) == 1:
            i = idx[0]
            keep_rows.append((means[i], covs[i], p[i], mult[i],
                              n_members[i], i))
        else:
            n_merged_groups += 1
            keep_rows.append(_merge_group(idx, means, covs, p, mult,
                                          n_members))

    means2 = np.array([r[0] for r in keep_rows])
    covs2 = np.array([r[1] for r in keep_rows])
    p2 = np.array([r[2] for r in keep_rows])
    mult2 = np.array([r[3] for r in keep_rows])
    nm2 = np.array([r[4] for r in keep_rows], dtype=np.int64)
    src = np.array([r[5] for r in keep_rows], dtype=np.int64)
    isl2 = island_id[src]
    ratio2 = med_ratio[src]
    match2 = med_match[src]

    # --- blend flag (post-merge; NaN med_ratio = un-refereed = not a blend)
    blend = (p2 > blend_p) & np.isfinite(ratio2) & (ratio2 < blend_ratio)
    n_flagged = int(blend.sum())
    n_dropped = 0
    if drop_blends and n_flagged:
        keep = ~blend
        n_dropped = n_flagged
        means2, covs2, p2, mult2, nm2, isl2, ratio2, match2, blend = (
            means2[keep], covs2[keep], p2[keep], mult2[keep], nm2[keep],
            isl2[keep], ratio2[keep], match2[keep], blend[keep])

    order = np.argsort(means2[:, 1])
    means2, covs2, p2, mult2, nm2, isl2, ratio2, match2, blend = (
        means2[order], covs2[order], p2[order], mult2[order], nm2[order],
        isl2[order], ratio2[order], match2[order], blend[order])

    meta["referee_apply"] = dict(
        fit_npz=fit_npz, referee_npz=referee_npz, merge_cut=merge_cut,
        blend_p=blend_p, blend_ratio=blend_ratio,
        n_in=n, n_out=len(p2), merged_groups=n_merged_groups,
        blends_flagged=n_flagged, blends_dropped=n_dropped,
        drop_blends=bool(drop_blends),
    )
    np.savez_compressed(
        out, means=means2, covs=covs2, p=p2, mult=mult2, n_members=nm2,
        island_id=isl2, f0_window_edges=f0_window_edges, blend=blend,
        med_match=match2, med_ratio=ratio2, meta=json.dumps(meta))
    print(f"refereed: {n} -> {len(p2)} comps | merged groups "
          f"{n_merged_groups} | blends flagged {n_flagged}"
          f"{f' (DROPPED {n_dropped})' if n_dropped else ' (kept)'} | "
          f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", required=True, help="fitter components npz")
    ap.add_argument("--referee", required=True, help="referee verdict npz")
    ap.add_argument("--out", required=True, help="refereed output npz")
    ap.add_argument("--merge-cut", type=float, default=MERGE_CUT)
    ap.add_argument("--blend-p", type=float, default=BLEND_P)
    ap.add_argument("--blend-ratio", type=float, default=BLEND_RATIO)
    ap.add_argument("--drop-blends", action="store_true")
    args = ap.parse_args()
    apply(args.fit, args.referee, args.out, merge_cut=args.merge_cut,
          blend_p=args.blend_p, blend_ratio=args.blend_ratio,
          drop_blends=args.drop_blends)


if __name__ == "__main__":
    main()
