# Warm-start GB proposal: clustering the previous run's posterior

Design note for the next-big-run warm-start proposal (2026-08-14).
Scope: GBs only — they are the RJ branch; MBH/SOBBH/EMRI/PSD/galfor/VGB
warm-start from direct samples of the previous posterior.

## Problem

Turn a finished run's cold-chain leaf samples (iteration x walker x
variable-length leaves, ~10^7–10^8 rows x 9 params) into a **birth/death
proposal distribution** for the next run (longer Tobs), used alongside —
and possibly ahead of — the F-stat proposal. Requirements (user):

* fit ONCE before a run, minutes not hours;
* evaluation as fast as any other proposal (RJ needs `logpdf` at
  arbitrary points for both directions of the move);
* no a-priori number of sources; borderline sources appear in a
  FRACTION p in (0, 1] of posterior samples, and p must survive into the
  proposal (it is the birth weight and the faint-tail accounting);
* the clustering method is a **swappable strategy**.

## Validated pipeline (prototype `proto_gb_cluster.py`, 2026-08-14)

Three stages; only stage 2 is "the clustering method".

1. **f0 density-valley segmentation** (the hierarchical frequency first
   look). Histogram every leaf-sample at ~1/Tobs bins; islands =
   contiguous runs above a count floor (0.5% of posterior samples/bin).
   One pass over the full table — 13 s at a simulated 6x10^7 rows —
   after which islands are independent (embarrassingly parallel).
   Density-valley, not gap-based: sparse junk leaves BRIDGE gaps but
   never beat a density floor.
2. **Within-island split**: robust-whiten (f0, fdot, lnA, lam, sin_beta)
   by island MAD; **single-linkage** on a <=1500-row subsample, cut at
   2.0 whitened units; assign all island rows to the nearest centroid
   with a radius cap (>6 -> junk, excluded). Single linkage because
   modes are dense and inter-source gaps are empty; **complete linkage
   provably shatters** (measured: 181 fragments from 10 sources, p
   diluted to ~0.03 — it bounds the farthest pair, which a 1500-point
   5-D Gaussian never satisfies).
3. **Cluster -> component**: Gaussian mean/cov per cluster (circular
   phi0/psi handled at the distribution stage) + **inclusion
   probability p = distinct posterior samples containing a member /
   total samples** + a leaf-multiplicity check (dup detector).

Prototype results (synthetic 3-mo-like posterior, 86k rows, 12k
samples, junk leaves): 10/10 truths recovered including p=0.10; a
2.5/Tobs blend pair separated only in sky resolved into exactly 2
components; p_hat tracked p_true (0.50/0.50, 0.20/0.20, 0.10/0.10).
Proposal evaluation as an f0-windowed Gaussian mixture: **0.65 us/point**
— the same cost class and plumbing shape as `StackedFStatProposal4D`.

v1 refinements before real data: satellite-fragment merge pass,
circular phi0/psi in the fitted component, covariance eigenvalue
floors.

## Direct-sampling methods (no clustering) — the full assessment

The family: use the stored samples THEMSELVES as the proposal.

**(a) Pure empirical resampling (draw a stored leaf row, propose it).**
Invalid for MH, and not fixably so: a proposal must supply `q(x)` as a
density. Atoms give q = sum of deltas. The killer is the DEATH side:
the reverse-move factor needs q evaluated at the parameters of the
source being removed — an arbitrary point that is almost surely not a
stored atom, so q = 0 and the acceptance ratio is undefined/infinite.
Any direct method must therefore smooth into a density first.

**(b) Global KDE over all leaf samples.** Valid density. The f0
thinness makes evaluation tractable exactly as suspected: sort kernels
by f0 once, and only kernels within ~10 sigma_f0 of the candidate
contribute (KD-tree/f0-window pruning). Honest cost accounting:

* *Speed is NOT the decisive argument.* A loud source contributes
  ~2x10^4 kernels to its window; at ~2x10^6 rj proposals/propose
  (flip 0.3) that is ~10^10 kernel ops — GPU-batchable in ~seconds.
  Slower than 3 Gaussians/window by ~10^3, but not prohibitive.
* *Memory:* 6x10^7 x 9 f64 ~ 4.3 GB resident (host, + device copy for
  GPU eval). Thinning to ~10^6 kernels is standard and cheap.
* *Bandwidth is the real technical risk.* 8–9D KDE bandwidth selection
  on data whose scales differ by orders of magnitude across f0
  neighborhoods (loud vs faint posterior widths differ by ~SNR)
  requires LOCAL whitening — global Scott/Silverman rules either
  oversmooth faint sources or undersmooth loud ones. Doing local
  whitening properly means partitioning by f0 neighborhood… which is
  stage 1 of the clustering pipeline by another name.
* *The decisive argument:* a global KDE has **no source identity and no
  inclusion probability**. You cannot seed leaves, weight births by
  detectability, count the faint tail, or dedup against the new run's
  catalogue. Those aren't optional for the 6-mo plan — p IS the
  warm-start's value over the fstat proposal.

**(c) Per-island KDE (hybrid: segment, don't split).** Keep stage 1,
skip stage 2, fit a KDE per island. Gets local whitening for free and
captures non-Gaussian shapes (curved amp–distance ridges, sky
multimodality) exactly. Still loses per-source p inside blended
islands, and island-level p is ill-defined once two sources share one
(the blend pair would report p ~ 2.0 "sources per sample"). Fine as a
*density* backend; insufficient as the *bookkeeping* layer.

**(d) k-means on samples.** Needs K (unknown), and its Euclidean
objective on unwhitened multi-scale data is the same trap as (b)'s
bandwidth. Within-island with a gap statistic it is just a worse
single-linkage. No advantage found.

**(e) Normalizing flow / neural density estimator.** Fast eval, exact
density, one-time fit. But the target is a ~2,500-spike comb along f0
spanning 4 decades of local scale — flows need per-island conditioning
to fit that, which again reintroduces stage 1; adds a training
dependency and a validation burden; and still no identity/p.

**Synthesis — where direct methods actually fit.** The pipeline's
stage 3 ("cluster -> distribution") is an independent swap point:
**"cluster, then KDE-per-cluster"** keeps identity + p + weights and
gains everything the direct family offers (non-Gaussian shapes) at
bounded cost (a cluster's own samples are its kernels, locally
whitened by construction). So the direct-sampling idea is not
rejected — it is demoted from *architecture* to *per-cluster density
backend*, selectable per cluster (Gaussian default; KDE or small GMM
for clusters whose samples fail a Gaussianity check).

## Recommendation

* Architecture: the 3-stage pipeline above. Interfaces:
  `segment(f0_sorted_rows) -> islands` (fixed),
  `split(island_rows) -> labels` (swappable: single-linkage default;
  DBSCAN / GMM-BIC / match-graph drop-ins),
  `densify(cluster_rows) -> Distribution` (swappable: Gaussian default;
  KDE / GMM upgrades).
* The match statistic (<a|a>, <a|b>, <b|b> batch pair kernel — swap-ll
  shaped, the comps already compute the cross terms) is deferred to:
  ambiguous-island refereeing, cross-run dedup (f0 proximity is NOT a
  duplicate test), and validation. Not needed for v1.
* Cross-Tobs use: components are physical-parameter densities;
  widths shrink with the new Tobs's Fisher scaling only through the
  refit/anneal — v1 proposes at previous-run widths (conservative,
  wider = safer for MH) with f0 windows re-checked against the new
  1/Tobs.
* Weights: birth mixture weight ∝ p (optionally x SNR ramp), plus the
  comb/floor components exactly as the fstat proposal mixes today.
