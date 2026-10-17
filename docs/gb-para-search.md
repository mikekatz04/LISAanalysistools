# GB parallel residual search (`rj_para_search_pe`)

_Status 2026-10-16: built and tested on the laptop CPU (synthetic WDM worlds, a
FakeWorld two-rank fan-out, and the real `gb_no_fg` debug fit in single-process
mode). GPU and cluster behaviour are UNVERIFIED (section 11). Code default OFF:
nothing here changes a run until `GB_PARA_SEARCH=1`._

## 1. The ruling

Mike, 2026-10-16 (verbatim): "The old parallel search proposal for GBs, I want
to effectively update that with the modern infrastructure. This should be a new
proposal we will add to the mix. In the old setup it searched the max logL
residual. I want it to run one set on each of the four cold-chain walkers and
then combine the distribution at the end. Here are the steps: 1) enter GB RJ
proposal. 2) build the paraensemble setup to sample 1 GB per sub-band per walker
(cold-chain only). You can operate directly on the residuals, no need to put
them in multiple buffers. 3) You run the parallel-tempered fixed-dimensional
MCMC on each cold (walker, band) until they all converge per sub-band in max
logL. Then determine if any source has been found at all above optimal SNR of 5
or 7 (probably 5), turn each sub-band's surviving source into a GMM, combine
them into a large GMM, and then we slot that in and sample from it. There could
be multiple sources per sub-band at the low snrs but try to examine that. take
your time, be careful, update this. If it works we will slot it in to carve out
the foreground faster."

Why (9mo job 773, rows 871-1008): `full_pe` adds ~0.4 resolved GBs per row per
walker, two thirds of them in 2.5-4 mHz; the RJ cold birth acceptance is ~1e-3
per proposal (prior 3e-4, F-stat grid 1e-4..2e-4, full self refit 4e-3,
filtered 1e-2). The prior and the galaxy-wide F-stat grid are blind to what a
given walker's residual still holds; this search proposes exactly the next
source each walker's residual contains. On the single-process `gb_no_fg` debug
fit (2 walkers, 3 days) the move's cold birth acceptance was 2/16 over its
first draws (one toy number, not a production estimate).

## 2. The old move and what was carried over

`GBSpecialRJSerialSearchMCMC` (gbspecialstretch.py) ran Eryn's
`ParaEnsembleSampler` with one group per band: stage 1 an F-statistic
likelihood (amplitude, phi0, iota, psi maximised analytically; Gibbs on f0, the
third column and the sky), an SNR gate, stage 2 a full-likelihood refine and
`fit_gb_gmm_rj_container` on the stage-2 samples. It scored through the legacy
GBGPU FD kernels (`para_log_like` refuses a WDM residual since 2026-08-12) on
ONE walker (the max-lnL one) and refuses the multi-rank fan-out. It is left
untouched.

Carried over: band-restricted priors (f0 in the band, the third column per
`_third_col_search_bounds`, the run's other priors), the two stages, the SNR
gate, a GMM per find. New: the WDM scorers on the walker's own residual, one
search per cold walker on the rank that holds it, per-band convergence on the
cold-rung running max, a comb that seeds stage 1, the cluster split for 2+
sources per band, the observable-basis GMM in the warm-start npz schema, the
cross-walker merge, and the epoch / manifest / one-draw-lag adoption protocol
of the warm-start self refit.

## 3. Design as built

### 3.1 Data flow

```
rj_para_search_pe propose (HEAD / single process; PE stages only)
  setup():  poll -> adopt the newest COMPLETE epoch (held for this propose)
            clock.tick(gf_iteration)                       (journaled clock)
            build due?  no epoch ever built (first entry), or
                        clock - built_at_clock >= GB_PARA_SEARCH_EVERY_PE
            BUILD: one gb_para_search command, EVERY compute rank at once
              rank r, each walker row of its ACA (production: one):
                 per band group (one ParaEnsembleSampler each):
                   comb    F on a 1-bin f0 comb x n_sky sky points per band
                   stage 1 F-stat PT-MCMC, ntemps x nwalkers chains per band,
                           Gibbs on (f0, Mc, alpha, sin_delta), ratio held 0,
                           cold + warm rungs started at the best comb cells;
                           a band stops when its cold-rung running max has not
                           improved by > TOL over WINDOW steps (MIN..MAX steps)
                   cluster greedy peak picking on f0 over the cold + warm rung
                           samples and the comb cells (radius SPLIT_BINS bins)
                   gate A  the cluster's best sqrt(2 F) >= SNR_MIN
                 stage 2   full likelihood, all 9 columns, STAGE2_NTEMPS x
                           nwalkers chains per (band, cluster), STAGE2_STEPS,
                           started at the cluster's members with the extrinsics
                           at the exact F-stat maximisers
                 gate B    best sample opt SNR >= SNR_MIN, det >= DET_FRAC x opt
                 de-dup    the walker's finds within SPLIT_BINS of each other
                 reply     samples + scalars (host arrays)
              head: merge across walkers and band edges -> one source each;
                    GMM per source (observable basis, min BIC); ONE npz in the
                    WarmStartComponents schema; DONE.json LAST; prune
            this propose draws from the epoch adopted BEFORE the build (the
            one-draw lag); with none held it proposes nothing
  gb_run_proposal payload "para_search": {"epoch": k} -> every rank loads k
  births + densities: ParaSearchHolder -> WarmStartComponents(epoch k)
```

### 3.2 Stage 1: comb-seeded F-stat PT per band

* Basis: the run's 9-column sampling basis (`dist, f0 [mHz], Mc, phi0,
  cos_iota, psi, alpha, sin_delta, fdot_astro_ratio`); the move refuses any
  other (`ParaSearchBuilder` raises, `build_gb_moves` checks `ndim == 9`).
* Prior: the run's GB prior with the f0 and Mc keys replaced by unit-cube
  uniforms (`band_restricted_priors_in`, the old `_band_restricted_priors_in`
  free-standing, `reset_key_order` onto the run's columns), mapped per group by
  `BandBoxPrior` (the old `PriorTransformFn` contract): f0 in the band, Mc in
  the run's chirp-mass box (`_para_mc_lims` = the old `_third_col_search_bounds`).
* Likelihood: `F = 0.5 N^T M^-1 N` (`fstat_proposal.compute_fstat`), the
  extrinsic-maximised lnL. Gibbs mask on columns 1, 2, 6, 7. The ratio column is
  HELD AT 0 (deviation from the old move, which held each walker at its own
  prior draw): the F-stat is flat along the (Mc, ratio) split, and the stretch
  move is only valid when every walker of an ensemble samples one target.
* Ladder: `tempering_kwargs=dict(ntemps, Tmax=inf)` (the old move's), Eryn's
  adaptive per-group ladder.
* The comb (deviation, `GB_PARA_SEARCH_INIT=comb`, default): per band, F on a
  1-bin f0 comb times `n_sky` sky points (a golden-ratio spread rotated per node;
  `n_sky = sky_nodes_required(f, Tobs)` clipped to [8, 64]) with Mc drawn from its
  box. Rung k < ntemps-1 starts from the best 4 x nwalkers cells drawn with
  weight `exp(beta_k (F - F_max))`; the hottest rung starts from the prior. Why:
  a 1-bin-wide F-stat peak in a sub-band of hundreds of bins (2 get_N / Tobs) is
  found by blind PT only slowly (the hot rung samples the prior a few walkers at
  a time); on the 7.6-day world the comb-seeded search finds every SNR >= 6
  source at the 30-step minimum. `GB_PARA_SEARCH_INIT=prior` restores the old
  prior start.
* Convergence (the ruling's "until they all converge per sub-band in max
  logL"): a `stopping_fn` (`_Tracker`) called every step tracks each group's
  cold-rung running max; a group stops when the max has not improved by more
  than `CONVERGE_TOL` over `CONVERGE_WINDOW` steps, after at least `MIN_STEPS`,
  or at `MAX_STEPS` (not converged). Stopping sets
  `state.groups_running[g] = False`; Eryn's next propose copies the mask, so the
  likelihood batch SHRINKS (pinned with a real sampler,
  `test_the_sampler_batch_shrinks_as_groups_converge`). The per-band step
  counts and converged flags are in every walker's finds record and the CLI
  report; the manifest keeps min / median / max per walker.
* Band groups: the sig-het scorer keeps ONE reference block resident
  (`FSTAT_SIGHET_REF_BLOCK`, 512 references at the margin spacing
  `sighet_fstat_ref_margin_hz(K, Tobs)`), so the sub-bands are searched in
  contiguous groups whose reference comb fits one block (`band_groups`; ~11
  sub-bands at 9mo, 107 groups). The comb and the PT of a group share one
  scorer (one reference build). `GB_PARA_SEARCH_GROUP_BANDS > 0` caps the group
  size instead; the chunked scorer runs every band in one group.

### 3.3 Multiple sources per band, gate A

The pool of a band is the last `CONVERGE_WINDOW` steps of the cold rung and the
next `CLUSTER_RUNGS - 1` warm rungs (default: cold + one warm) plus the band's
best comb cells, each with its own (beta-independent) F. `greedy_peaks` visits
the pool in descending F: an unassigned sample becomes a cluster centre and takes
every unassigned sample within `SPLIT_BINS` bins in f0. A cluster passes gate A
when its best F reaches `snr_min^2 / 2`; at most 4 clusters per band
(`MAX_FINDS_PER_BAND`). A greedy pick, not the old gap rule: two sources 8 bins
apart whose sidelobes leave less than a 4-bin gap between their sample clouds
still split.

What it can and cannot see (examined on the synthetic world): a band with an SNR
7 and an SNR 6 source 8 bins apart yields TWO finds (F1 26.4 and 20.1; the cross
term of the band-mate lifts both above SNR^2 / 2). The warm rung samples a
secondary source with weight `exp(-beta_1 dF)`: comparable SNRs are seen, a weak
source beside a loud one in the same band is not (dF of hundreds of nats). In
PE that case is the walker's own model: once the loud source is a leaf it is
subtracted from the residual and the next epoch sees the weak one. Within-walker
duplicates (a sidelobe, the same source seen from a neighbouring band's edge)
are dropped after stage 2: finds whose best f0 lie within `SPLIT_BINS` keep the
higher lnL.

Second pass (subtract the found source, rerun the band): NOT implemented. On the
evidence above it would recover only weak sources next to loud ones in the same
band within one epoch, which the walker's own leaves and the next epoch already
recover; it would double the stage-1 cost of every band with a find. Revisit
only if the cluster validation shows 2+ comparable sources per band being
missed (the CLI report's `band_peaks` / finds per band columns show it).

### 3.4 Stage 2, gate B, the F-stat maximiser mapping

One `ParaEnsembleSampler` over every gate-A find of the walker: `STAGE2_NTEMPS`
x nwalkers chains (all 9 columns sampled), f0 prior a `SPLIT_BINS`-bin window
around the cluster centre (NOT clipped to the band: an edge source is fitted
whole and the merge drops the edge copy), Mc in its box, the other columns the
run's prior. Start: up to 16 distinct members of the cluster (best F first) with
the four extrinsics at the EXACT chunked F-stat maximisers, a spread in the
(Mc, ratio) split that keeps fdot and A fixed (`Mc' = Mc (1/(1+r))^(3/5)`,
`d' = d (Mc'/Mc)^(5/3)`, r ~ U[-0.3, 0.3] x min(1, M)) and a small extrinsic
jitter; start rows outside the run's prior are replaced by in-prior rows of the
same find, and a find with none is dropped (`n_prior_dropped`). The GMM is
fitted to the cold rung's last `STAGE2_STEPS / 2` steps (`nwalkers` samples
each). Gate B on the best cold sample of the whole stage (re-scored):
`opt = sqrt(<h|h>) >= SNR_MIN` and `det = <r|h> / sqrt(<h|h>) >= DET_FRAC x
opt`; the medians over 32 re-scored samples are reported beside them. At an
exact amplitude maximum `det == opt`, so the det clause catches a best sample
the residual does not support (a mis-scored or edge-pinned row), not a weak
source; the opt clause is the detection gate.

THE MAXIMISER MAPPING (measured, chunked WDM comp, noise-free injections): the
template that reproduces lnL = F exactly has PHYSICAL `phi0 = -phi0_max`,
`iota = pi - iota_max`, `psi = psi_max`, i.e. in the sampling basis (the
transform negates phi0) `phi0 = phi0_max`, `cos_iota = -cos(iota_max)`,
`psi = psi_max`. `cos_iota = +cos(iota_max)` -- the production birth pin's
choice, `GBSpecialBase._pe_or_pin_extrinsics` -- is exact only edge-on:

| source | cos iota | F | lnL, `cos(pi - iota_max)` | lnL, `cos(iota_max)` (production pin) |
|---|---|---|---|---|
| SNR 4 | -0.001 | 7.98 | 7.98 | 7.98 |
| SNR 6 | -0.773 | 17.92 | 17.92 | -16.00 |
| SNR 9 | +0.021 | 40.71 | 40.71 | 40.44 |
| SNR 7 | +0.307 | 24.02 | 24.02 | 0.39 |
| SNR 6 | -0.532 | 17.00 | 17.00 | -10.09 |
| SNR 15 | -0.476 | 112.62 | 112.62 | -152.51 |

(`scratchpad exp_conv.py`, reproduced by
`test_stage2_sits_on_the_stage1_maximum`.) Production exposure: the F-stat
distance birth is off (`GB_RJ_FSTAT_DIST_BIRTH=0` in the launchers), so plain
births never use the maximisers; the consumers that do are the PE extrinsic
draw centres, the epoch centre tables and the replace moves' pins. NOT changed
here -- a separate ruling (section 11).

### 3.5 The head: merge, GMM, epoch

`merge_finds`: two finds are one source when their best f0 differ by at most
`SPLIT_BINS / Tobs` and their fdot by at most `SPLIT_BINS / Tobs^2` (single
linkage). Finds pinned at a band edge are dropped whenever an unpinned find of
the source exists. The source's samples are the UNION of its finds' samples,
each find subsampled to the same count (at most 800) so every walker that found
it weighs the same -- chosen over "the highest-SNR cluster" because the birth
law should cover the spread between the walkers' residuals (what proposing to
each walker what the others found needs), and the per-source GMM is a min-BIC
mixture, so a bimodal union is represented rather than averaged. Each source is
mapped to the OBSERVABLE basis (`fit_from_store.build_observable_map` /
`to_observable`), fitted by `fit_cluster_gmms` (2048 resampled rows, at most 6
components, 25 members per component; the head's GPU when it has one, else
CPU), packed with `pack_gmm_components` and written exactly as
`fit_from_store.run(basis="observable")` writes: per-cluster `p`, `mult`,
`n_members`, `island_id`, `f0_window_edges`, the `gmm_*` arrays and the meta
(`basis`, `column_names`, `f0_units`, `circular_cols`, `bounded_cols`,
`map_params`, `tobs`, `gmm`). `p = 1` per source: equal birth weight (an SNR
weighting is a later knob). Extra `src_*` arrays (f0, opt / det SNR, band,
walkers per source) ride along for diagnostics; `WarmStartComponents.from_npz`
ignores them.

### 3.6 The epoch protocol and the move

* Root: `<run_dir>/warmstart/parasearch/` (`para_search_epoch_root`). Each build
  is `epoch_NNNN/` (`self_refit.next_epoch_index`: a crashed build's directory
  is never reused) with `parasearch.npz`, `finds.npz` (every walker's raw finds)
  and `DONE.json` written LAST (tmp + fsync + rename). A complete epoch is one
  whose manifest names an npz that exists (`self_refit.read_manifest`); a
  zero-source build writes the manifest with `npz: null` (recorded, never
  adopted). `self_refit.prune_epochs` after each build (the newest 3 complete
  epochs and every 10th stay whole).
* Clock: `clock.json` under the root counts the PE iterations the move has
  seen (`ParaSearchClock`: differences of the stage combine's per-process
  `gf_iteration`, so a restart continues the cadence). The first build is at
  the move's first PE entry; a failed build retries one cadence later.
* Lag: `setup()` polls BEFORE building, so the build propose ships the epoch
  held before the build (none on the run's first build: that propose proposes
  nothing) and the next draw adopts the new one (the 2026-10-16 self-refit
  rationale). Compute ranks never poll: `_para_search_apply` loads exactly the
  head's epoch (an epoch-less directive empties the holder; an epoch a rank
  cannot load raises -- a propose must draw and price from the head's
  container).
* The move: `GBParaSearchRJMove(GBSpecialRJPriorMove)` built in
  `recipe.build_gb_moves` like `rj_warm_pe` (`use_prior_removal=False`,
  `pe_extrinsic_draw=False`, the PE overrides `warm_pe_move_overrides()`, the
  PE flip and repeat defaults), container loaded like the warm PE twin's (the
  run's prior box as uniform floor, `GB_WARM_START_FLOOR_EPS`,
  `circ_images = GB_WARM_START_CIRC_IMAGES`), seeds
  `warm_seed(build_seed, 0x3A14 ^ (epoch + 1))` (per rank and epoch).
  `setup_gb_moves` (gb_no_fg / all_sources) composes it right after
  `rj_prior_pe` in every PE stage (`compose_para_search`) when
  `gb.para_search` is on; the driver is untouched (so `--print-recipe` does not
  show it, like `rj_replace_pe`).
* Stage gate: PE stages only (kind `pe`, not `replica_pe`); anywhere else the
  move proposes nothing and never builds.
* Draw share: `GB_PE_RJ_PARA_FRACTION` (default 0), appended LAST to the PE
  draw-one members with `rj_prior_pe` taking the remainder. At 0 the member is
  outside the draw (the draws are bit-identical, golden-pinned against HEAD
  57df9f6e) and the move runs every PE iteration in its slot -- the
  `rj_warm_pe` rule. Set it > 0 to make it one of the drawn RJ moves.
* Replica layout: `GF_GPU_ROUTING=1` with several ranks per walker block is
  REFUSED at `install_walker_fanout` (NotImplementedError naming the knob): v1
  searches each walker on the one rank that holds its residual.
* Failures never take the run down: a walker's search that raises is reported
  in the rank's `errors` and the head builds from the other walkers; a build
  that fails as a whole logs `[GB_PARA_SEARCH] epoch k build FAILED`, keeps the
  held epoch and retries a cadence later.
* Reproducibility: Eryn's `ParaEnsembleSampler` and `ProbDistContainer` draw
  from the GLOBAL numpy / cupy streams; every search runs inside
  `seeded_global_rng(seed)`, which seeds them and restores the caller's state,
  so a search is reproducible from its per-walker seed
  (`SeedSequence([build_seed, epoch, walker])`, shipped by the head) and the
  run's own global streams are untouched. (`ParaEnsembleSampler` is marked
  deprecated in Eryn in favour of `EnsembleSampler(nsamplers=...)`; the ruling
  names it, and it is used with the warning silenced.)

## 4. Scorers and entry points

| use | scorer | standalone (`ResidualScorer`) | in the move (`_make_move_para_scorer`) |
|---|---|---|---|
| comb + stage 1, `GB_PARA_SEARCH_SCORER=auto` / `sighet` | shared-reference sig-het F-stat | `fstat_gridfit.build_sighet_call_fstat(sig_comp, holder, xp=, Tobs=, f0_lims_hz=(group lo, hi), data_index=row, noise_index=row)` | `_RoutedBandEngine.route_sighet_fstat` (pins to the row's shard) |
| comb + stage 1, `chunked` | exact per-candidate F-stat | `chunked.get_fstat_ll_wdm(P, holder, data_index=di, noise_index=di)` | `_RoutedBandEngine.route_fstat_ll(chunked, "get_fstat_ll_wdm", acs, P, ...)` |
| stage-2 seeds | Jaranowski-Krol inversion | `fstat_proposal.fstat_maximized_extrinsics(N, M)` on the exact F-stat | same |
| stage 2, gate B | full lnL `<r|h> - <h|h>/2` | `chunked.get_ll_wdm(...)` -> `d_h_out`, `h_h_out` (comps built with `d_d = 0`) | the parent engine `_likelihood_engine.get_ll(acs, P, ..., N_vals=None)` inside `exact_scoring()` |

The residual is the walker's row of `model.analysis_container_arr` as it stands
at the move's entry: data minus every fitted template, GBs included -- what the
walker has NOT absorbed. Nothing is copied into sub-band buffers; no residual is
mutated. `auto` takes the sig-het scorer when the GB comp is a
`GBSignalHetComputations` with `v4_knots > 0` (the production default,
`GB_SIGHET_INMODEL=1`, `SIGHET_V4_KNOTS=128`), else the chunked one.

## 5. Cost model

### 5.1 Measured per-template costs (laptop CPU, one thread, `nice`)

| grid | scorer | cost |
|---|---|---|
| 15 d (Nf 256 x Nt 512, 433 active pixels; N_sparse 256, Nt_sub 128) | chunked F-stat `get_fstat_ll_wdm` | 32-36 ms / row |
| same | chunked full lnL `get_ll_wdm` | 16.5-17 ms / row |
| same, comp N_sparse 128 / Nt_sub 64 | chunked F-stat / full lnL | 35 / 18.6 ms / row |
| same | sig-het F-stat (K = 64) | 2.2 ms / row + 17 ms per reference built |
| 7.6 d (Nt 256, 209 active pixels; N_sparse 128) | chunked full lnL | 8.3 ms / row |

Whole searches (CPU): the 15-day world (8 bands of 64 bins, 4 x 16 chains, comb
seeding, 30-step minimum, 5 finds x 2 x 16 chains x 20 stage-2 steps): stage 1
35 s (15,144 F-stat rows), stage 2 92 s (4,965 full-lnL rows). The 7.6-day world
(the test default): ~80 s for the whole search.

### 5.2 GPU anchors (production logs; the search itself is GPU-unverified)

* sig-het F-stat: the F-stat fit's comb sweeps run at 75-79k evals/s per GPU
  (9mo jobs 751-754, deep-dives section 5) = **13 us / row**.
* chunked full lnL per row on the GPU: not measured in isolation. The CPU ratio
  full / sig-het F-stat is ~8 (17 / 2.2 ms); carried over, **~0.1 ms / row**.
  Upper bound from the RJ births (40k births scored in ~55 s including the
  buffers and newborn repeats): 1.4 ms / row.
* in-model sig-het template (repeats): 740k in ~140 s = 0.19 ms (context only).

### 5.3 One epoch at 9mo, per compute rank (one walker), the defaults

Tobs 2.33e7 s; 1232 sub-bands over 0.55-22 mHz = ~500k frequency bins
(df = 4.3e-8 Hz); K = 128 sig-het margin 3.9e-7 Hz -> ~54k references -> ~107 band
groups of 512.

| piece | count | estimate |
|---|---|---|
| comb | 500k bins x n_sky (Doppler law clipped to [8, 64]; 64 above 3.4 mHz, ~59 averaged) = ~30M F-stat rows | ~6.5 min |
| stage-1 PT, typical | 1232 bands x 128 chains x ~125 steps (comb-seeded bands stop near the 100-step minimum) = ~20M rows | ~4.5 min |
| stage-1 per-step overhead | 107 groups x ~125 steps x 10-20 ms (stretch, priors, transforms, tempering on small arrays) | ~2-5 min |
| reference builds | ~54k references (17 ms each on the CPU; GPU unmeasured) | ~0.1-0.5 min if ~0.1-0.5 ms each |
| stage 2 | finds x 4 x 16 chains x 100 steps; 200 finds = 1.3M rows, the 512 cap = 3.3M rows | 2-6 min at 0.1 ms / row (30-77 min at the 1.4 ms upper bound) |
| head merge + GMM + npz | a few hundred sources, batched GMM | < 1 min |
| **total, typical** | | **~15-25 min** |
| worst case (every band to 600 steps) | +75M rows + 600-step overhead | ~+35 min |

Overhead: at `GB_PARA_SEARCH_EVERY_PE=100` and ~6-8 min per 9mo PE row (the 6mo
`full_pe` ran 5.5 min per iteration) an epoch every 10-13 h costs ~2-4 %
(typical) to ~8 % (worst case). Every rank searches in parallel; the head waits
for the slowest. Levers, in order: `GB_PARA_SEARCH_STAGE2_NTEMPS` and
`_MAX_FINDS` (stage 2), `_INIT_NSKY_MAX` (comb), `_NTEMPS` / `_NWALKERS` (stage
1, linear), `_MIN_STEPS` / `_CONVERGE_WINDOW`.

### 5.4 Noise false alarms at SNR 5

In real noise the stage-1 maximum of a band is also taken over noise. The F-stat
of noise is chi^2_4 / 2, so `P(F > 12.5) = e^-12.5 (1 + 12.5) = 5e-5` per
independent template. A 9mo sub-band at 3 mHz holds ~400 bins x ~50 sky patches
~ 2e4 templates: ~60 % of the bands carry a noise peak above SNR 5 (at SNR 7,
`P(F > 24.5) = 6e-10` per template: ~1e-5 per band). Expect up to ~one SNR-5
noise "find" per sub-band per walker (the same noise in every walker's residual,
so the cross-walker merge does not reject it); the PE RJ judges them like any
other birth (the run's own opt-SNR floor applies). They cost stage-2 time, which
`GB_PARA_SEARCH_MAX_FINDS=512` (the loudest stage-1 finds) bounds.
Recommendation: validate at the ruled 5.0, then compare 6-7 on the cluster
(the CLI report gives the SNR distribution of the finds).

## 6. Knobs

All `GBSettings` fields (stock/erebor/gb.py; env = the capitalised attribute name,
`GB_` prefix), resolved into `ParaSearchConfig` by `from_settings`.

| env knob (field) | default | meaning |
|---|---|---|
| `GB_PARA_SEARCH` (`para_search`) | 0 | master switch: build `rj_para_search_pe` and compose it after `rj_prior_pe` in the PE stages; 0 = no move object, bit-identical |
| `GB_PARA_SEARCH_EVERY_PE` (`para_search_every_pe`) | 100 | PE iterations between searches on the journaled clock; the first build is at the move's first PE entry; <= 0 = that build only |
| `GB_PARA_SEARCH_NTEMPS` (`para_search_ntemps`) | 8 | stage-1 rungs per sub-band (Tmax = inf) |
| `GB_PARA_SEARCH_NWALKERS` (`para_search_nwalkers`) | 16 | walkers per rung (even, >= 4) |
| `GB_PARA_SEARCH_STAGE2_NTEMPS` (`para_search_stage2_ntemps`) | 4 | stage-2 rungs (<= 0 = the stage-1 count) |
| `GB_PARA_SEARCH_SNR_MIN` (`para_search_snr_min`) | 5.0 | the detection floor of gates A and B ("5 or 7 (probably 5)") |
| `GB_PARA_SEARCH_DET_FRAC` (`para_search_det_frac`) | 0.7 | gate B: det SNR >= this x opt SNR |
| `GB_PARA_SEARCH_CONVERGE_TOL` (`para_search_converge_tol`) | 0.5 | nats the cold-rung running max may gain over the window and still count as converged |
| `GB_PARA_SEARCH_CONVERGE_WINDOW` (`para_search_converge_window`) | 50 | steps of that window; also the steps pooled for the cluster split |
| `GB_PARA_SEARCH_MIN_STEPS` / `_MAX_STEPS` (`para_search_min_steps` / `_max_steps`) | 100 / 600 | per-band step floor / ceiling |
| `GB_PARA_SEARCH_STAGE2_STEPS` (`para_search_stage2_steps`) | 100 | the full-likelihood stage; the GMM takes its last half |
| `GB_PARA_SEARCH_SPLIT_BINS` (`para_search_split_bins`) | 4.0 | cluster radius (1/Tobs bins), the stage-2 f0 window half-width, the de-dup and merge radius |
| `GB_PARA_SEARCH_CLUSTER_RUNGS` (`para_search_cluster_rungs`) | 2 | rungs pooled for the split (cold + warm) |
| `GB_PARA_SEARCH_SCORER` (`para_search_scorer`) | auto | `auto` / `sighet` / `chunked` stage-1 F-stat |
| `GB_PARA_SEARCH_GROUP_BANDS` (`para_search_group_bands`) | 0 | sub-bands per sampler run (0 = one sig-het reference block; all with chunked) |
| `GB_PARA_SEARCH_INIT` (`para_search_init`) | comb | `comb` seeds stage 1 from the comb, `prior` from the prior |
| `GB_PARA_SEARCH_INIT_NSKY_MIN` / `_MAX` (`para_search_init_nsky_min` / `_max`) | 8 / 64 | comb sky points per f0 node (the Doppler law clipped) |
| `GB_PARA_SEARCH_MAX_FINDS` (`para_search_max_finds`) | 512 | stage-2 finds per walker (the loudest; <= 0 = all) |
| `GB_PE_RJ_PARA_FRACTION` (read per call by `GFCombineMove`) | 0 | the move's PE draw-one share; 0 = outside the draw, every iteration |

Reused, not new: `GB_WARM_START_FLOOR_EPS` (0.05) and `GB_WARM_START_CIRC_IMAGES`
(3) for loading an epoch; `FSTAT_SIGHET_REF_BLOCK` (512) and
`FSTAT_SIGHET_REF_SPACING_HZ` for the band groups; `FSTAT_BATCH` (4096) as the
scorer batch. Module constants (`warmstart/para_search.py`): 4 finds per band,
16 stage-2 seeds per find, 32 re-scored samples, GMM 2048 / 6 / 25, 800 union
rows per find.

## 7. Log lines

| line | where | read it for |
|---|---|---|
| `build_gb_moves: rj_para_search_pe armed (epochs under ...; every N PE iterations; T x W chains per sub-band, SNR >= s; draw share GB_PE_RJ_PARA_FRACTION=...)` | recipe, every rank at build | the move exists and its knobs |
| `[GB_PARA_SEARCH] rj_para_search_pe composed after rj_prior_pe in stage(s) [...]` | `compose_para_search` | the recipe carries it |
| `[GB_PARA_SEARCH] rank r walker w: N bands searched, M converged in a/b/c steps (min/median/max; ...), F finds (>= SNR s; gate A passed .., capped .., outside the prior .., gate B failed .., .. duplicate(s) dropped), B band(s) with 2+ finds, wall X s (fstat stage .. s [sighet, n evals], full stage .. s [n evals]; GMM on the head)` | every rank, per walker per build | the per-walker search |
| `[GB_PARA_SEARCH] rank r walker w: stage 1 group g/G done (.. s so far)` | every tenth band group | progress of a long stage 1 |
| `[GB_PARA_SEARCH] epoch k built: W walkers, F finds -> C components after de-dup (S sources; E rank error(s)), npz ... in T s (ranks .. s, head merge + GMM .. s); adopted at the NEXT draw of rj_para_search_pe` | head | the build; the wall every rank waited |
| `[GB_PARA_SEARCH] epoch k build FAILED (...); keeping epoch j, next attempt in N PE iterations.` | head | a failed build (never fatal) |
| `[GB_PARA_SEARCH] rank r: search FAILED (...)` | the failing rank | a walker the epoch was built without |
| `[GB_PARA_SEARCH rj_para_search_pe] no complete epoch held yet: this propose draws nothing (clock c).` | head | the first build propose (the lag) |
| `[GB_PARA_SEARCH] rank r adopted epoch k into rj_para_search_pe: S sources (F finds over W walkers), C components.` | every rank | adoption |
| `[GB_PARA_SEARCH rj_para_search_pe] rN this propose used epoch k (C components) \| cumulative proposes .. births viable .. acc .. (..) cold ../.. (..) deaths acc ../..` | every rank, every propose | the move's yield; with `[GB_ACCEPT rj-split rj_para_search_pe]` |
| `[GB_PARA_SEARCH rj_para_search_pe] stage X (kind Y): PE stages only in v1 -- this move proposes nothing here.` | head, once | composed into a non-PE stage |
| `[GB_PARA_SEARCH] pruned n older epoch(s) (.. MB)` | head | pruning |

## 8. Files

`<run_dir>/warmstart/parasearch/clock.json` (`{"iterations": n}`) and
`epoch_NNNN/`: `parasearch.npz` (the container), `finds.npz` (every walker's
record: bands, steps, converged, band_finds, band_peaks, per-find scalars
`band f0_hz fdot F1 lnl2 opt_snr det_snr opt_med det_med edge`, `best`,
`offsets`, `samples`; `para_search.load_finds_npz`), `DONE.json` (`epoch`,
`npz`, `built_at_clock`, `built_at_iteration`, `stage`, `n_walkers`, `walkers`,
`finds_per_walker`, `bands_2plus_finds`, `steps_per_walker` [min, median, max],
`converged_frac`, `walls_per_walker`, `n_finds`, `n_sources`, `n_components`,
`opt_snr` quantiles, `finds_per_band`, `tobs`, `knobs`, `errors`, `rank_wall`,
`n_compute`, `walls` (gmm / write / head_total), `gmm_device`, `built_on`).

## 9. The CLI (cluster validation runbook)

`python -m lisatools.globalfit.warmstart.para_search --store <h5> --walker <w>
[--bands a:b] --out <dir> [--merge-epoch]` searches ONE walker of a store
snapshot offline and writes `<out>/report_w<w>.txt` (knobs, per-band steps /
convergence / finds, every find's band, f0, fdot, F1, opt / det SNR and their
medians, edge flag, and how many finds the walker's own last cold row already
holds -- should be ~0), `<out>/finds_w<w>.npz`, and with `--merge-epoch` a
one-walker `<out>/epoch/parasearch.npz` + `DONE.json`.

What it reads: `global_fit/domain_settings` (the WDM grid) and
`noise_model_identity.data_t0` (the WDM t0) from the store,
`sub_backend/gb/band_edges`, the walker's residual from
`<store>_residual_snapshot.npz` (`residuals[walkers.index(w)]`; written by the
run every `GF_RESIDUAL_SNAPSHOT_EVERY` rows, every cold walker since 1f1289cf;
`--residual` overrides the path) and, for the "already held" count, the last
cold row of `chain/gb` / `inds/gb`. NOT reconstructed: the run's noise model.
The CLI uses the residual's own EMPIRICAL XYZ covariance per (WDM layer, time
block) (`empirical_inverse_covariance`; `--noise-time-blocks`, default
Nt_active // 512, `--noise-layer-smooth 1`) -- the noise plus the unresolved
signal the walker still holds, which is what the run's noise + foreground fit
describes; a loud unfound source inflates its own layers slightly (its SNR
reads low). The GB comp is rebuilt from the stock GB settings (env-resolved:
source the launcher's exports first, or pass `--nt-sub --n-pad --n-sparse
--n-cp-sig --n-cp-orbit --t-ref --sighet-v4-knots --sighet-v3-nodes`), the
priors as uniform boxes from `--mc-min/--mc-max/--dist-min/--dist-max/
--ratio-max` (stock defaults). The search knobs are the `GB_PARA_SEARCH_*` env.

Runbook (one GPU, an interactive node, the job stopped or on a tar's extract):

```
cd <run_dir>      # or the extracted tar: the *_extract.h5 + *_residual_snapshot.npz
source <the run's launcher exports>   # the stock GB settings resolve from them
python -m lisatools.globalfit.warmstart.para_search \
    --store gf_prod_9mo_testing.h5 --walker 0 --bands 400:480 \
    --out para_cli_w0 --backend cuda12x --gpu 0 --merge-epoch
```

Start with 80 bands around 2.5-4 mHz (the PE yield's peak; band index from
`sub_backend/gb/band_edges`), check (1) the walls against section 5 (the log
prints the eval counts per stage), (2) the SNR distribution of the finds and how
many sit below 7 (section 5.4), (3) `already held` ~ 0, (4) the overlap of the
finds with the next rows' new leaves of that walker (a find the walker later
births is a hit), (5) the bands with 2+ finds. Then all bands, then the 3 other
walkers. Only then `GB_PARA_SEARCH=1` with `GB_PE_RJ_PARA_FRACTION` > 0.

## 10. Tests

All CPU, never the full suite:

| module | what | wall |
|---|---|---|
| `tests/test_gb_para_search_units.py` (36) | knobs and env names, the gates, clustering, band groups, the band box, the convergence tracker (also through a real `ParaEnsembleSampler`: the batch shrinks), merge, epoch npz through `WarmStartComponents.from_npz`, holder poll / directive / seeds, clock, RNG scope, recipe composition, the draw-one share, the move's wiring on a skeleton (stage gate, first-entry build, the lag, the cadence across a restart, failed builds, rank apply, replica refusal, op dispatch, the payload / rank-body seams) | ~15 s |
| `tests/test_gb_para_search_world.py` (7) | the search on the noise-free 7.6-day world: SNR >= 6 found, SNR 4 not, the two-source band split, every band converges and stops, the SNR gate, stage 2 at the stage-1 maximum (the maximiser mapping), sig-het vs exact F-stat, the npz draws on the finds | ~90 s |
| `tests/test_gb_para_search_cli.py` (2) | the CLI on a synthetic store with noise, two walkers | ~70 s |
| `tests/test_gb_para_search_fanout.py` (2) | the build over two FakeWorld ranks through the real `_fanout_cmd` / `gf_serve` / rank body; the cross-rank merge; a failing rank | ~10 s |
| `tests/test_gb_para_search_golden.py` (2) | the PE draw-one off paths bit-identical to HEAD 57df9f6e (`tests/data/para_search_draw_one_golden.npz`), with a negative control | ~3 s |
| `tests/test_gb_para_search_integration.py` (1, gated `RUN_GF_PARA_INTEGRATION=1`) | the real `gb_no_fg` debug fit, single process: first-entry build, the lag, adoption, births, the cadence rebuild | ~2 min, ~2 GB |

Mutation checks (each applied, a named test failed, the file restored byte
identical by md5): M1 gate A floor (`test_detection_gates`; the world test still
holds through gate B), M2 gate B det clause, M3 split radius
(`ClusterAndGroupTest`), M4 convergence tolerance and M5 the switch-off
(`TrackerTest`, both), M6 merge radius (`MergeTest`, fan-out), M7 edge-pinned
drop, M8 the one-draw lag (`MoveWiringTest`, fan-out), M9 the stage gate, M10
the payload key, M11 the rank apply, M12 the draw-one member rule (draw-one +
golden), M13 the prior remainder, M14 the cos(iota) sign of the maximiser
mapping (world `test_stage2_sits_on_the_stage1_maximum`), M15 the clock
journal, M16 the authoritative directive, M17 the replica refusal, M18 the npz
schema (`EpochTest`), M19 an env name, M20 the RNG restore, M21 the composition
anchor, M22 the rank error capture, M23 the rank's walker block, M24 the
per-walker seeds, M25 the cadence.

## 11. Unverified, limits, follow-ups

* **GPU**: nothing ran on a GPU. The cupy paths (sampler `gpu=`, the scorer
  arrays, `seeded_global_rng`'s cupy state, the GMM on the head's device) are
  written but untested; the section-5 costs are estimates from production
  anchors and CPU ratios.
* **Multi-rank on real ranks**: the fan-out is tested over FakeWorld threads
  with a stub search; the real search ran single-process only. The 4-rank
  cluster run (each rank its walker, the head waiting) is unverified, as is the
  wall the head and the saver tolerate while every rank searches.
* **`GF_GPU_ROUTING=1`** (several ranks per walker block): refused in v1.
* **Search reliability at 9mo**: the comb + PT found every SNR >= 6 source on
  7.6- and 15-day worlds; 9mo has a finer sky (n_sky up to 64 in the comb vs the
  F-stat fit's 512) and wider bands. The cluster CLI validation (section 9) is
  the test.
* **Noise false alarms at SNR 5** (section 5.4): expected; the cap bounds the
  cost.
* **The F-stat maximiser cos(iota) sign** (section 3.4): this module uses the
  measured mapping. `GBSpecialBase._pe_or_pin_extrinsics` (the PE extrinsic
  draw centres and the replace / distance-birth pins) writes
  `cos_iota = cos(iota_max)`, exact only edge-on on the chunked WDM path.
  Not changed here; it needs its own ruling and a check of every consumer.
* **The deprecation**: `ParaEnsembleSampler` warns that it will be removed
  in favour of `EnsembleSampler(nsamplers=...)`.
* Not built: the second pass per band (section 3.3), an SNR-weighted `p`.
* Pre-existing failures seen while running neighbouring suites (also at HEAD
  57df9f6e, not from this work): `tests.test_rj_flip_fraction`
  `test_zero_occupancy_shutoff_after_5` and
  `test_every_rj_constructor_carries_the_flip_default` (`rj_warm_search`,
  `rj_warm_pe`).

## 12. Launcher export block (for the coordinator; not in any launcher)

```
# ---- GB parallel residual search (rj_para_search_pe; ruling 2026-10-16,
# docs/gb-para-search.md). Code default OFF. Validate with the CLI first.
export GB_PARA_SEARCH=${GB_PARA_SEARCH:-0}
export GB_PE_RJ_PARA_FRACTION=${GB_PE_RJ_PARA_FRACTION:-0}   # > 0: one of the drawn PE RJ moves (prior takes the remainder)
export GB_PARA_SEARCH_EVERY_PE=${GB_PARA_SEARCH_EVERY_PE:-100}
export GB_PARA_SEARCH_NTEMPS=${GB_PARA_SEARCH_NTEMPS:-8}
export GB_PARA_SEARCH_NWALKERS=${GB_PARA_SEARCH_NWALKERS:-16}
export GB_PARA_SEARCH_STAGE2_NTEMPS=${GB_PARA_SEARCH_STAGE2_NTEMPS:-4}
export GB_PARA_SEARCH_SNR_MIN=${GB_PARA_SEARCH_SNR_MIN:-5.0}
export GB_PARA_SEARCH_DET_FRAC=${GB_PARA_SEARCH_DET_FRAC:-0.7}
export GB_PARA_SEARCH_CONVERGE_TOL=${GB_PARA_SEARCH_CONVERGE_TOL:-0.5}
export GB_PARA_SEARCH_CONVERGE_WINDOW=${GB_PARA_SEARCH_CONVERGE_WINDOW:-50}
export GB_PARA_SEARCH_MIN_STEPS=${GB_PARA_SEARCH_MIN_STEPS:-100}
export GB_PARA_SEARCH_MAX_STEPS=${GB_PARA_SEARCH_MAX_STEPS:-600}
export GB_PARA_SEARCH_STAGE2_STEPS=${GB_PARA_SEARCH_STAGE2_STEPS:-100}
export GB_PARA_SEARCH_SPLIT_BINS=${GB_PARA_SEARCH_SPLIT_BINS:-4.0}
export GB_PARA_SEARCH_MAX_FINDS=${GB_PARA_SEARCH_MAX_FINDS:-512}
# (GB_PARA_SEARCH_SCORER=auto, _INIT=comb, _INIT_NSKY_MIN/_MAX=8/64,
#  _CLUSTER_RUNGS=2, _GROUP_BANDS=0 stay at their code defaults)
```

Needs `GB_PE_RJ_DRAW_ONE=1` (the launchers set it) for the share to act, and
the 9-column GB basis (the default).
