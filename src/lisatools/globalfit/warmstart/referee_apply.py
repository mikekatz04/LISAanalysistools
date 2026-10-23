"""Stage 2.5 of the warm-start pipeline: apply the match referee's verdict.

Consumes the fitter npz (:mod:`.fit_from_store`) and the referee npz
(:mod:`.match_referee`) and writes the REFEREED components npz that
production arms via ``GB_WARM_START_COMPONENTS``
(docs/6mo-run-prep.md: "point GB_WARM_START_COMPONENTS at a refereed
final-store npz"). Run in-process by
:func:`.build.ensure_warm_start_components` (output to a temporary name,
then moved onto ``GB_WARM_START_COMPONENTS``) and by every self-refit
epoch (``self_refereed.npz``), always with the default thresholds.
Thresholds are the 2026-08-24 referee-verdict rulings:

* AUTO-MERGE same-island pairs with centroid cross-match > ``merge_cut``
  (0.9) — split artifacts of one source. Merges are union-find over
  CLUSTERS and only across clusters (the components of one cluster's
  mixture are never merged with each other). Observable (mixture) set:
  the merged clusters' mixtures are CONCATENATED, weights rescaled by
  their ``p`` (:func:`_apply_observable`). Legacy sampling set:
  moment-matched Gaussian merge with weights ~ p; circular columns
  (phi0/psi/alpha) merged via the minimal image around the highest-p
  member (:func:`_merge_group`). Either way merged ``p = min(1, sum p_i)``
  (fragments of one source occupy different posterior samples), members
  summed, mult p-weighted.
* FLAG BLENDS: clusters with ``p > blend_p`` (0.5) and ANY of coherence
  ``med_ratio < blend_ratio`` (0.5), ``mult > blend_mult`` (2.0) or raw
  ``med_match < blend_match`` (0.6) (a NaN, un-refereed diagnostic never
  fires) — mosaics absorbing >1 real source.
  KEPT by default (they still seed births near real power; the new run's
  RJ resolves them); ``--drop-blends`` removes them. The ``blend`` bool
  column rides in the output either way.
* Pairs in (0.5, merge_cut] and everything un-refereed pass through
  untouched.

The output keeps the exact fitter writer schema (plus ``blend``,
``med_match``, ``med_ratio`` diagnostic columns and a ``referee_apply``
meta block), so ``WarmStartComponents.from_npz`` loads it unchanged.

Example:
    The call the build makes (the referee npz is ``<fit stem>_referee.npz``):

    .. code-block:: bash

        python -m lisatools.globalfit.warmstart.referee_apply \\
            --fit gf_prod_6mo_v9_4gpu_refereed_fit.npz \\
            --referee gf_prod_6mo_v9_4gpu_refereed_fit_referee.npz \\
            --out gf_prod_6mo_v9_4gpu_refereed.npz

    Options: ``[--merge-cut 0.9] [--blend-p 0.5] [--blend-ratio 0.5]
    [--blend-mult 2.0] [--blend-match 0.6] [--drop-blends]``.
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
# WIDE-BLEND discriminators (ruling 2026-09-08): sources absorbed BINS
# apart carry a HIGH med_ratio -- the member f0 spread makes the sinc
# prediction tiny, excusing a raw match of ~0.35 -- so the ratio flag
# missed every measured production wide blend (e.g. the SNR-153 pair at
# 5.34778 mHz, mult 2.8, ratio 1.09). mult > BLEND_MULT or raw
# med_match < BLEND_MATCH catches them.
BLEND_MULT = 2.0
BLEND_MATCH = 0.6


def merge_candidate_pairs(gmm_ncomp, island_id, pairs=None):
    """Flat component-index pairs eligible for the auto-merge test.

    Same island, DIFFERENT cluster. Mixture siblings are exempt: a cluster's
    K components are a deliberate multi-modal description of ONE source, so
    they cross-match highly and a merge would collapse them straight back
    into the single Gaussian the observable-basis design exists to avoid.
    Genuine split artifacts land in different clusters and are still merged
    exactly as before.

    ``gmm_ncomp`` is the per-cluster component count from
    ``pack_gmm_components``; it partitions the flat arrays, so no separate
    cluster id is needed. A legacy set is all-ones and reproduces the old
    behaviour exactly.

    ``pairs`` filters an EXISTING pair list (what production does with the
    referee's own pairs, O(len(pairs))); omitted, every eligible pair is
    enumerated, which is O(n^2) and only appropriate for small sets.

    Args:
        gmm_ncomp (array-like): ``(n_clusters,)`` component counts.
        island_id (array-like): Island per cluster (``(n_clusters,)``) or per
            flat component.
        pairs (array-like, optional): ``(P, 2)`` flat component pairs to
            filter. Default ``None`` = enumerate.

    Returns:
        list[tuple[int, int]]: The eligible flat component pairs.
    """
    ncomp = np.asarray(gmm_ncomp, dtype=int)
    cluster_of = np.repeat(np.arange(len(ncomp)), ncomp)
    isl = np.asarray(island_id)
    isl_of = np.repeat(isl, ncomp) if len(isl) == len(ncomp) else isl

    def ok(i, j):
        return cluster_of[i] != cluster_of[j] and isl_of[i] == isl_of[j]

    if pairs is not None:
        return [(int(i), int(j)) for i, j in pairs if ok(int(i), int(j))]
    n = len(cluster_of)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if ok(i, j)]


def _find(parent, a):
    """Union-find root of ``a`` (path halving, updates ``parent`` in place)."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _merge_group(idx, means, covs, p, mult, n_members):
    """Moment-matched Gaussian merge of components ``idx`` (weights ~ p); the sampling-basis path.

    Circular means are taken at their minimal image around the highest-p
    member before averaging and wrapped back afterwards.

    Returns:
        tuple: ``(mean (9,), cov (9, 9), p = min(1, sum p), p-weighted mult,
        summed n_members, index of the highest-p member)``.
    """
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


def _apply_observable(gmm, roots, p, mult, n_members, island_id,
                      f0_window_edges, med_ratio, med_match, meta, out,
                      fit_npz, referee_npz, merge_cut, blend_p, blend_ratio,
                      blend_mult, blend_match, drop_blends):
    """Apply the verdict to a MIXTURE (observable-basis) component set.

    An across-cluster auto-merge CONCATENATES the two clusters' mixtures --
    weights rescaled by their inclusion probabilities -- instead of
    moment-matching them into one Gaussian.

    That is the same argument the design makes against a single Gaussian per
    cluster: two split artifacts are two lobes of one posterior, and a
    moment-matched merge puts most of the merged mass in the EMPTY space
    between them, which is a birth proposal aimed where there is nothing.
    Concatenating keeps the lobes and simply relabels them as one source,
    which is what the merge verdict actually asserts. Legacy sets keep the
    moment-matched merge untouched.

    ``roots`` is the per-cluster union-find root (clusters sharing a root
    merge); a merged cluster's box is the union of its members' boxes, its
    blend diagnostics (``med_ratio`` / ``med_match``) those of its
    highest-p member's FIRST component. Clusters are written in ascending
    mixture-mean ``f_mid`` order. Writes ``out`` and returns it.
    """
    ncomp = np.asarray(gmm["gmm_ncomp"], dtype=int)
    splits = np.cumsum(ncomp)[:-1]
    w_of = np.split(np.asarray(gmm["gmm_weights"], dtype=float), splits)
    parts = {k: np.split(np.asarray(gmm[f"gmm_{k}"]), splits, axis=0)
             for k in ("means", "covs", "invcovs", "dets")}
    mins = np.asarray(gmm["gmm_mins"], dtype=float)
    maxs = np.asarray(gmm["gmm_maxs"], dtype=float)

    rows, n_merged_groups = [], 0
    for root in np.unique(roots):
        idx = np.flatnonzero(roots == root)
        base = idx[int(np.argmax(p[idx]))]
        if len(idx) > 1:
            n_merged_groups += 1
        pw = p[idx] / p[idx].sum()
        rows.append(dict(
            weights=np.concatenate([w_of[i] * float(pw[k])
                                    for k, i in enumerate(idx)]),
            means=np.concatenate([parts["means"][i] for i in idx], axis=0),
            covs=np.concatenate([parts["covs"][i] for i in idx], axis=0),
            invcovs=np.concatenate([parts["invcovs"][i] for i in idx],
                                   axis=0),
            dets=np.concatenate([parts["dets"][i] for i in idx]),
            # the merged cluster's box is the union of its members'
            mins=mins[idx].min(axis=0), maxs=maxs[idx].max(axis=0),
            p=min(1.0, float(p[idx].sum())),
            mult=float(np.einsum("k,k->", pw, mult[idx])),
            n_members=int(n_members[idx].sum()), src=int(base),
        ))

    p2 = np.array([r["p"] for r in rows])
    mult2 = np.array([r["mult"] for r in rows])
    nm2 = np.array([r["n_members"] for r in rows], dtype=np.int64)
    src = np.array([r["src"] for r in rows], dtype=np.int64)
    isl2 = island_id[src]
    ratio2 = np.array([med_ratio[np.cumsum(ncomp)[i] - ncomp[i]]
                       for i in src])
    match2 = np.array([med_match[np.cumsum(ncomp)[i] - ncomp[i]]
                       for i in src])

    blend = (p2 > blend_p) & (
        (np.isfinite(ratio2) & (ratio2 < blend_ratio))
        | (mult2 > blend_mult)
        | (np.isfinite(match2) & (match2 < blend_match))
    )
    n_flagged = int(blend.sum())
    n_dropped = 0
    keep = np.ones(len(rows), dtype=bool)
    if drop_blends and n_flagged:
        keep = ~blend
        n_dropped = n_flagged

    # clusters stay in ascending column-1 (f_mid) order, by mixture mean
    fmid = np.array([float(r["weights"] @ r["means"][:, 1]) for r in rows])
    order = [i for i in np.argsort(fmid) if keep[i]]
    rows = [rows[i] for i in order]
    p2, mult2, nm2, isl2, ratio2, match2, blend = (
        p2[order], mult2[order], nm2[order], isl2[order], ratio2[order],
        match2[order], blend[order])

    packed = dict(
        gmm_ncomp=np.array([len(r["weights"]) for r in rows], dtype=int),
        gmm_weights=np.concatenate([r["weights"] for r in rows]),
        gmm_means=np.concatenate([r["means"] for r in rows], axis=0),
        gmm_covs=np.concatenate([r["covs"] for r in rows], axis=0),
        gmm_invcovs=np.concatenate([r["invcovs"] for r in rows], axis=0),
        gmm_dets=np.concatenate([r["dets"] for r in rows]),
        gmm_mins=np.vstack([r["mins"] for r in rows]),
        gmm_maxs=np.vstack([r["maxs"] for r in rows]),
    )
    meta["referee_apply"] = dict(
        fit_npz=fit_npz, referee_npz=referee_npz, merge_cut=merge_cut,
        blend_p=blend_p, blend_ratio=blend_ratio,
        n_in=len(ncomp), n_out=len(rows), merged_groups=n_merged_groups,
        blends_flagged=n_flagged, blends_dropped=n_dropped,
        drop_blends=bool(drop_blends),
        merge_scope="across-cluster only (mixture siblings exempt)",
    )
    np.savez_compressed(
        out, p=p2, mult=mult2, n_members=nm2, island_id=isl2,
        f0_window_edges=f0_window_edges, blend=blend, med_match=match2,
        med_ratio=ratio2, meta=json.dumps(meta), **packed)
    print(f"refereed [observable]: {len(ncomp)} -> {len(rows)} clusters "
          f"({int(packed['gmm_ncomp'].sum())} components) | merged groups "
          f"{n_merged_groups} | blends flagged {n_flagged}"
          f"{f' (DROPPED {n_dropped})' if n_dropped else ' (kept)'} | "
          f"wrote {out}")
    return out


def apply(fit_npz: str, referee_npz: str, out: str,
          merge_cut: float = MERGE_CUT, blend_p: float = BLEND_P,
          blend_ratio: float = BLEND_RATIO, blend_mult: float = BLEND_MULT,
          blend_match: float = BLEND_MATCH,
          drop_blends: bool = False) -> str:
    """Apply the referee verdict to a fitter npz and write the refereed npz.

    Auto-merges same-island, across-cluster pairs with ``cross_match >
    merge_cut`` (union-find over clusters), then flags blends (see the
    module docstring) and writes the fitter schema plus ``blend``,
    ``med_match``, ``med_ratio`` and ``meta["referee_apply"]``. The
    observable (mixture) format goes through :func:`_apply_observable`, the
    legacy sampling format through :func:`_merge_group`.

    Args:
        fit_npz: The :mod:`.fit_from_store` npz.
        referee_npz: Its :mod:`.match_referee` npz (``pairs``,
            ``cross_match``, ``med_ratio``, ``med_match``).
        out: Output npz path.
        merge_cut: Default ``MERGE_CUT`` (0.9).
        blend_p: Default ``BLEND_P`` (0.5).
        blend_ratio: Default ``BLEND_RATIO`` (0.5).
        blend_mult: Default ``BLEND_MULT`` (2.0).
        blend_match: Default ``BLEND_MATCH`` (0.6).
        drop_blends: Remove flagged blends. Default ``False`` (kept).

    Returns:
        str: ``out``.

    Raises:
        ValueError: When the referee's per-component arrays do not match the
            fit's component count (a referee npz of a different fit).
    """
    with np.load(fit_npz, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        basis = str(meta.get("basis", "sampling"))
        p = np.array(d["p"])
        mult = np.array(d["mult"])
        n_members = np.array(d["n_members"])
        island_id = np.array(d["island_id"])
        f0_window_edges = np.array(d["f0_window_edges"])
        if basis == "observable":
            gmm = {k: np.array(d[k]) for k in (
                "gmm_ncomp", "gmm_weights", "gmm_means", "gmm_covs",
                "gmm_invcovs", "gmm_dets", "gmm_mins", "gmm_maxs")}
            means = gmm["gmm_means"]
            covs = gmm["gmm_covs"]
        else:
            gmm = None
            means = np.array(d["means"])
            covs = np.array(d["covs"])
    with np.load(referee_npz, allow_pickle=False) as r:
        pairs = np.array(r["pairs"]).reshape(-1, 2)
        cross = np.array(r["cross_match"]).ravel()
        med_ratio = np.array(r["med_ratio"])
        med_match = np.array(r["med_match"])
    n = len(means)                       # FLAT component count
    n_clusters = len(p)
    ncomp = (gmm["gmm_ncomp"] if gmm is not None
             else np.ones(n_clusters, dtype=int))
    if med_ratio.shape != (n,):
        raise ValueError(
            f"referee med_ratio has {med_ratio.shape}, fit has {n} comps -- "
            "the referee npz was built from a DIFFERENT fit npz.")

    # --- auto-merges (union-find; transitive chains collapse together) ---
    # Restricted to ACROSS-cluster pairs: a cluster's mixture components
    # describe ONE source on purpose and must not be merged with each other.
    # For a legacy set ncomp is all ones, so every pair is across-cluster
    # and this is exactly the previous behaviour.
    allowed = set(merge_candidate_pairs(ncomp, island_id, pairs=pairs))
    cluster_of = np.repeat(np.arange(n_clusters), ncomp)
    parent = list(range(n_clusters))
    for (i, j), cm in zip(pairs, cross):
        if cm > merge_cut and (int(i), int(j)) in allowed:
            ci, cj = int(cluster_of[int(i)]), int(cluster_of[int(j)])
            ri, rj = _find(parent, ci), _find(parent, cj)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([_find(parent, i) for i in range(n_clusters)])

    if gmm is not None:
        return _apply_observable(
            gmm, roots, p, mult, n_members, island_id, f0_window_edges,
            med_ratio, med_match, meta, out, fit_npz, referee_npz,
            merge_cut, blend_p, blend_ratio, blend_mult, blend_match,
            drop_blends)

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

    # --- blend flag (post-merge; NaN referee columns = un-refereed = not
    # a blend). Three discriminators: low sinc-ratio (close blends), high
    # mult or low RAW match (wide blends -- see BLEND_MULT/BLEND_MATCH).
    blend = (p2 > blend_p) & (
        (np.isfinite(ratio2) & (ratio2 < blend_ratio))
        | (mult2 > blend_mult)
        | (np.isfinite(match2) & (match2 < blend_match))
    )
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


def main(argv=None):
    """CLI: parse ``argv`` and run :func:`apply` (description = the module docstring's first line).

    Args:
        argv (list of str, optional): Arguments; ``None`` = ``sys.argv[1:]``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", required=True, help="fitter components npz")
    ap.add_argument("--referee", required=True, help="referee verdict npz")
    ap.add_argument("--out", required=True, help="refereed output npz")
    ap.add_argument("--merge-cut", type=float, default=MERGE_CUT)
    ap.add_argument("--blend-p", type=float, default=BLEND_P)
    ap.add_argument("--blend-ratio", type=float, default=BLEND_RATIO)
    ap.add_argument("--blend-mult", type=float, default=BLEND_MULT)
    ap.add_argument("--blend-match", type=float, default=BLEND_MATCH)
    ap.add_argument("--drop-blends", action="store_true")
    args = ap.parse_args(argv)
    apply(args.fit, args.referee, args.out, merge_cut=args.merge_cut,
          blend_p=args.blend_p, blend_ratio=args.blend_ratio,
          blend_mult=args.blend_mult, blend_match=args.blend_match,
          drop_blends=args.drop_blends)


if __name__ == "__main__":
    main()
