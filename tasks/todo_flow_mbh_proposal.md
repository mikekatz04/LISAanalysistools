# Make the MBH flow move effective (goal: acceptance-gated num_repeats reduction)

Date: 2026-07-13. Basis: splice test on the live joint run
(`test_flow_joint_sources_stft`, backend step 182, cold chain), harness at
`~/.claude/jobs/f7a00a42/tmp/splice_test.py` (results `splice_results.npz`).

## Measured facts (splice test — decides everything below)

Offline rebuild of the run's residuals + exact move `compute_like` path,
validated against stored `log_like` (median offset 0.000). Δlogl of candidate
leaf params vs the walker's own, per candidate kind:

| kind                        | median Δlogl | frac > −3 nats |
|-----------------------------|--------------|----------------|
| MBH walker splice same-mode | 0.0          | 0.82           |
| MBH walker splice cross-mode| −0.0         | 0.72           |
| MBH lag-3/6/12 (staleness)  | ±0.5         | ~0.8           |
| MBH flow draws (v180 ckpt)  | **−5220**    | **0.001**      |
| EMRI walker splice          | 0.1          | 0.80           |
| EMRI flow draws             | −1.9         | 0.64           |

- **Ceiling for a perfect per-leaf independence proposal: ~60% acceptance,
  every MBH leaf.** No conditional-vs-marginal gap, no SNR wall, no staleness
  cost, and cross-sky-mode jumps are likelihood-free (modes quasi-degenerate).
- The MBH flow is purely **mis-fit**: draws lose 10³–10⁵ nats that true
  posterior samples don't. EMRI flow is already within 1.9 nats of its
  ceiling (→ its observed 21%).
- Per-leaf flow deficits: unimodal leaves (0: −909, 2: −4154) vs multimodal
  leaves (3: −10653, 4: −51962, 5: −41049).

## Root causes (verified in code)

1. `train_noise` jitter is added per-dim in the **whitened latent space**
   (`eryn/flows/torch/flows.py` ~1080, scale = 0.1 × per-dim latent std).
2. `WhiteningTransform` is **block-diagonal**: Cholesky whitening for
   non-periodic dims, marginal-std only for periodic dims
   (`eryn/flows/torch/transforms.py`).
3. Consequences: (a) multimodal leaves — per-condition whitening maps each
   sky-mode island to a tiny fraction of unit scale; 0.1 jitter inflates
   islands 10–100× in the training data itself; (b) unimodal leaves — the
   razor phase/timing correlations involving φ_ref, ψ_L, λ_L never get
   Cholesky'd, stay ≪0.1 thin in latent space, and get puffed the same way;
   (c) NSF bridges between mode islands (mixture fitting).

## Phase 0 — offline iteration loop ✅ (executed 2026-07-14)

- [x] Splice harness: rebuild residuals from backend + score candidates via
  the moves' own `compute_like` (`splice_test.py`; patched settings copy
  `splice_settings.py` with `gpus=[7]`, scratch `head_dir`).
- [x] Driver: `train_offline_flows.py` (buffers from backend cold chain,
  time-ordered per leaf; trains candidates on GPU 7) +
  `score_offline_flows.py` (scores every candidate + the live checkpoint in
  one residual pass; results `score_results.npz`).

## Phase 1 — config-only ✅ (measured 2026-07-14, backend step 241)

Implied acceptance = E[min(1, e^Δll)], ceiling ~0.60. Offline replica of the
live config reproduces the live checkpoint (harness faithful).

| MBH config                  | overall acc | per-leaf median Δll (L0..L5)              |
|-----------------------------|-------------|-------------------------------------------|
| live ckpt (noise 0.1)       | 0.000       | −1152 −853 −4166 −14347 −30913 −34080     |
| offline noise 0.1 (replica) | 0.002       | −820 −936 −3462 −13885 −25359 −45568      |
| noise 0.01                  | 0.056       | −13 −18 −62 −289 −504 −1165               |
| noise 0                     | 0.139       | −4.7 −7.7 −10.2 −116 −148 −303            |
| noise 0, epochs 250         | 0.101       | (no gain — rejected)                      |
| **noise 0, 2× buffer**      | **0.191**   | **−6.1 −3.4 −5.2 −30.8 −46.1 −59.8**      |

EMRI: live 0.325 → noise 0.01: 0.425 → **noise 0: 0.444** (ceiling ~0.60).

- [x] MBH `fit_kwargs.train_noise` 0.1 → **0.0** (applied to
  `mojito_input/emri_mbh_psd_settings.py`; temporal split + patience remain
  the overfit guards — watch the val trend for sawtooth on the next run).
- [x] EMRI `fit_kwargs.train_noise` 0.1 → **0.0** (0 beat 0.01 in both branches).
- [x] MBH `max_buffer_samples` 2000 → **6000** (~25-step window at thin=5;
  splice-tested staleness cost ~0). `epochs_per_round` stays 150 (250 tested,
  no gain — LR-annealing schedule interplay).
- [ ] Optional untested lever: `flow_buffer_thin` 5 → 10 for MBH (more
  decorrelated rows per step; offline w168 result suggests decorrelation is
  part of the 2×-buffer win). Try on the next run if acceptance plateaus
  below ~0.2.
- Takes effect on the NEXT launch (live run deepcopied its settings).

## Phase 2 — ordered implementation steps (settled 2026-07-14 discussion)

Post-Phase-1 gap: unimodal leaves at −3…−6 nats (≈0.25–0.3 acc); deficit
concentrated in multimodal leaves (3/4/5 at −31…−60 nats). Note: the "0.60
ceiling" is a q-uncorrected diagnostic — a perfect flow reaches 1.0 (the
q-ratio cancels the target ratio), so real headroom extends past 0.60.
Every step below is measured offline with the harness before any run.

**Step 1 — exact-MH scorer upgrade** ✅ (2026-07-14). Scorer now reports
lnpdiff = [ℓ(y)−logq(y)] − [ℓ(x)−logq(x)]. Reframed Phase 1: the q-correction
roughly halves MBH numbers (flow density still too narrow/offset) and slightly
helps EMRI live. MBH noise0_w168: 0.080 exact-MH (was 0.191 q-uncorr).

**Step 2 — EMRI buffer-size test** ✅ (2026-07-14). EMRI exact-MH: live 0.370,
noise0_w84 0.309, noise0_w168 0.437, noise0_w240 0.416 (no gain past 2×).
Applied: EMRI `max_buffer_samples` 2000 → 4000 in the settings.

**Step 3 — `WhiteningTransform(periodic_in_cholesky=True)`** ✅ implemented
(Eryn 5a0e577, review Approved; default-False path verified bit-identical;
129 tests green). Measured: EMRI 0.427 → **0.501**; MBH overall 0.077 → 0.100
with unimodal leaves up strongly (L1 0.146→0.315, L2 0.146→0.217) and
multimodal leaves DOWN (island rotation makes the mixture harder for the
NSF) — strengthens Step 5 as the multimodal fix. Applied:
`periodic_in_cholesky=True` for both branches in the settings (post-restart
buffers seed true modes, so the multimodal penalty mostly reflects the
current frozen-mode state).

**Step 4 — temperature-scaled base** ✅ implemented (Eryn c434068 +
LAT 345eec2 wiring hook; 25/25 new+move tests green). `base_scale = β^{−1/2}`
per temperature row from the move's per-leaf ladder; None default
bit-identical. Effect shows up as hot-row acceptance in the next live run
(offline harness scores β=1 only). Upgrade path if hot rows still lag:
harvest all temps, condition on β as continuous context.

**Step 5 — mode-mixture conditioning** ✅ (GATE MET: Step-3 measurement showed
pcov *hurts* multimodal leaves — L3/4/5 at 0.005–0.012 — while helping
unimodal ones; this is the fix). **Full implementation plan:**
`Eryn/docs/superpowers/plans/2026-07-14-mode-mixture-conditioning.md`
(7 tasks: LeafModeConditioning, estimate_modes GMM+BIC, ModeMixtureFlow
fit/snapshot + mixture sampling/density, executor test, settings wiring,
harness measurement gate ≥0.08 on L3/4/5). **Eryn commits e91d6ef, 4f96192 landed + reviewed;
LAT settings wiring complete (this commit)**. Design summary — mode structure
is estimated FROM THE BUFFER, per leaf, each training round, trainer-side —
never imposed (the 8-mode sky lattice was only a diagnostic frame;
joint-space islands are the real objects):
- Cluster in the whitened JOINT space (periodic dims embedded as cos/sin);
  GMM + BIC over K=1..K_max (or separation-threshold agglomerative). We are
  not estimating "the true number of modes" — split whatever the NSF cannot
  bridge. Under-splitting = bridging (the failure); over-splitting ≈ free.
  Bias toward splitting. K=1 reduces to today's setup.
- Context = leaf one-hot ⊕ mode one-hot (fixed K_max slots); condition key
  (leaf, mode) gives per-island whitening via shared=False — islands become
  unit-scale N(0,I) each.
- Proposal: m ~ empirical weights w (floored so no mode is unproposable),
  x ~ q(·|leaf, m); factors use the MIXTURE density
  logq(x) = logsumexp_m [log w_m + logq(x|leaf, m)] at BOTH points (batched
  K-fold forward pass). Mode label is a proposal-internal latent — MCMC
  still targets the full joint posterior across all modes; cross-mode jumps
  happen at rate w_m (splice: cross-mode Δℓ ≈ 0 → they accept). Clustering
  quality affects fit only, never MH exactness.
- Stability: seed EM from previous round's centers; match labels by nearest
  center; mixture state ships inside the atomic weights snapshot (same
  pattern as data_transform).

**Step 6 — capacity bump** (GATED on Step-1 metric still showing within-mode
gap after Steps 3/5): transforms 8→10, bins 8→12 for MBH.

**Step 7 — mixture measurement gate: RUN 2026-07-14, backend step 290 → FAIL
(deferred, not refuted).** Exact-MH acceptance, all candidates retrained on
the same step-290 snapshot (per-leaf numbers are noisy at the ~2x level for
the failing leaves — 192 draws/leaf — but the magnitudes are robust):

| MBH config (w168)  | overall | L0    | L1    | L2        | L3    | L4    | L5    |
|--------------------|---------|-------|-------|-----------|-------|-------|-------|
| live ckpt (noise .1)| 0.002  | 0.005 | 0.007 | 0.001     | 0.000 | 0.000 | 0.000 |
| noise0 (no pcov)   | 0.067   | 0.081 | 0.114 | 0.116     | 0.063 | 0.020 | 0.007 |
| noise0 + pcov      | 0.071   | 0.095 | 0.165 | 0.130     | 0.021 | 0.005 | 0.007 |
| **+ mixture (k8)** | **0.110**| 0.055| 0.251 | **0.325** | 0.006 | 0.007 | 0.015 |
| + mixture (k12)    | 0.137   | 0.084 | 0.220 | **0.490** | 0.022 | 0.005 | 0.002 |

EMRI (control): pcov 0.493 → mixture 0.589 (clustering returns K=1 on both
leaves ⇒ wrapper degrades gracefully as designed; the delta is
conditioning-shape/training noise, not a mixture effect).

**Gate verdict: leaves 3/4/5 never reach the required 0.08 in any config.**

Attribution (each measured, not argued):
- NOT the mixture density: `dlogq = logq(y) − logq(x)` is 0.5–8.7 nats on
  every leaf — the exact-mixture MH machinery works. The failure is entirely
  in the likelihood term: median `dll` = −73…−4821 on L3/4/5 (draws land
  off-posterior), vs −0.2…−3.7 on L0/1/2.
- NOT the ceiling: re-ran the splice test at step 290 — same-mode walker
  splices still accept at **0.57–0.68 on every leaf**, lags free, and
  **cross-mode splices also 0.51–0.60 per leaf** (the sky images really are
  degenerate). The headroom is there; the flow can't reach it.
- NOT staleness: window smear ≈1.0 (w168 all leaves; w290 all but L0).
- NOT rows-per-island: w290 gives L3/4/5 ~1740 rows/island on a *clean*
  (smear ≈1.0) window and they still fail (0.005/0.012/0.047). (w290 does
  destroy L0 — smear 5.77, its window reaches the burn-in migration — a
  correct detection of the known stale-data mechanism.)
- The one variable that tracks success is **sky-image occupancy**:
  L0/L1/L2 occupy 1/2/2 images → acceptance 0.06–0.49; L3/L4/L5 occupy
  6/4/7 images → ~0.01. The clustering under-resolves the many-image leaves
  (L3: K=4 vs 6 images, L5: K=4 vs 7) and its K is window-unstable there.
- **The plan's "over-splitting is nearly free" assumption is FALSE**:
  k12 fragmenting L3 further pushed its median deficit −668 → −4821 nats.
  Keep kmax conservative.

**CORRECTION (same day, before this doc was acted on): the multimodality is
REAL and PERSISTENT — an earlier draft of this section claimed it was the
frozen legacy of the init-frame bug and would vanish at restart. That was
wrong.** Measured: `SkyMove` (an explicit sky-mode-hop proposal, weight 3×0.05
in the MBH inner moves) accepts at **13–15%**, and walkers hop between the 8
TRUE degenerate lattice images **18–135 times per walker over 290 steps**
(per-leaf median: L0 32, L1 85, L2 18, L3 26, L4 104, L5 135). Combined with
cross-mode splice acceptance 0.51–0.60, this says the MBHB sky images are
genuinely quasi-degenerate and the chains actively explore them. So multi-image
occupancy is the physically correct posterior, it will still be there after the
restart, and **the many-image leaves are a standing unsolved problem, not a
transient.** (The init bug still mattered — it corrupted truths/init/subtraction
— but it is not what makes L3/4/5 multimodal.)

**Honest verdict on the mixture.** Its measured value is real but narrow and
leaf-dependent, and it does NOT do the job it was built for:
- 2-image leaves — big, reproducible win: L2 0.130 → 0.325/0.490 (across two
  independent scoring passes), L1 0.165 → 0.251.
- 1-image leaf — small consistent LOSS: L0 0.095 → 0.055 (likely context
  dilution: kmax=8 adds 8 mostly-unused one-hot slots to the conditioner).
- 4–7-image leaves (its target) — no effect: ~0.01 with or without.
Net MBH 0.071 → 0.110 (k8) / 0.137 (k12), i.e. the whole mixture contributes
~+0.04 overall, while the *config* work (train_noise 0 + buffers + pcov) is what
took MBH from 0.000 → 0.07. Decision: **keep it** (exactness-safe, best measured
config, no-ops at K=1 — EMRI control returned K=1 on both leaves), but do not
treat the multimodal leaves as solved, and do NOT tune kmax up.
Harness: `scripts/diagnostics/flow_proposal_harness/` (the scorer now
dispatches ModeMixtureFlow checkpoints via the `mixture_state` dataset).

**What the many-image leaves actually need** (open, for a future session): the
flow must fit 4–7 degenerate islands *and* the chain is hopping between them
every few steps, so each island's buffer is both thin and non-persistent. The
clustering under-resolves them (L5: K=4 vs 7 images) and its K is
window-unstable. Candidate directions, none measured: (a) build the mode
lattice ANALYTICALLY from the known sky-degeneracy transformations (the same
maps `SkyMove` already applies) instead of clustering — the images are exact
images of each other, so one fitted island + 7 deterministic maps could give an
8-component mixture with full statistics per component; (b) accept that
`SkyMove` already handles cross-image moves at 13–15% and restrict the flow to
within-image proposals by conditioning on the current image label; (c) treat
`num_repeats` reduction as an EMRI-only win for now.

## Phase 3 — acceptance-gated num_repeats (lisatools)

- [ ] Per-leaf cold-chain flow-acceptance counters on
  `ResidualAddOneRemoveOneMove` (numbers exist at the debug-log site,
  addremovemove.py ~646) + running-window property.
- [ ] Controller in the PE recipe step: per branch,
  `num_repeats_eff = clip(base · target_refresh / max(acc · w_flow · n_prop, ε), floor, base)`
  with a sustained-acceptance gate (e.g. acc > 0.15 over 20 steps); floor
  ~10 keeps the non-flow move mix alive. First run in report-only mode, then
  enable. EMRI already qualifies at 21% once stationary.
- [ ] (Superseded by Step 4 if it lands first: skipping hot-temp flow rows.)

## Phase 4 — restart & verify

- [ ] Restart the joint run with the (already committed) init-frame fix —
  walkers seed the true sky modes; plotted truths correct.
- [ ] Watch `mbh_flow/` val trend + per-leaf acceptance; target ≥20–30%
  within a day of sampling; then enable the Phase-3 controller.
- [ ] Keep the splice harness as the regression check whenever the flow
  config changes.

## Context / provenance

- Truth/init frame bug (icrs_to_ecliptic detour) fixed 2026-07-13 in
  `recipe_steps.py::mbh_catalogue_to_sampling_basis` — affects plots, init,
  and initial subtraction; not sample validity. Current run's sky-mode
  occupancies are init-seeded; cross-mode splices ≈ 0 suggest modes are
  genuinely near-degenerate, so occupancies may re-equilibrate once flow
  (cross-mode) proposals start accepting.
- MBHB SNRs 227–1367; EMRI 53/65 (catalogue).

---

## 2026-07-14 (late): the reparametrization test — and a SkyMove finding

Question asked: instead of *modelling* the sky multimodality (ModeMixtureFlow),
can a **change of parametrization** remove it? Idea: the degeneracy group G is
known analytically — `SkyMove` implements its two generators, both exact
isometries of the sampling basis (|G| = 8). So fold every buffer row to a
canonical image, train ONE flow on ALL the leaf's rows (8x the data of any
clustered component), and propose `u ~ q_u ; g ~ Uniform(G) ; x = g(u)` with the
G-symmetrised density `q_x(x) = (1/8) Σ_g q_u(g⁻¹x)` — exact, no clustering, no
kmax, no window instability, cross-image jumps for free.

Geometrically it looked perfect: folding to the image nearest a per-leaf
reference (the antimode-cut trick, needed because G shifts λ and ψ *together*,
so a naive fundamental domain tears any cloud straddling λ = kπ/2 — leaf 0 split
K 1→2) collapsed **5 of 6 leaves to K=1 with all 4032 rows**, including leaf 4
(5 islands × ~800 rows → 1 × 4032), and the folded cloud's marginals matched a
single image's to 0–20%.

**Measured: it FAILED — 0.016 overall (worse than everything).** The failure is
diagnostic: the surviving-draw fraction is **0.146 / 0.109 / 0.130 on L0/L1/L2 =
1/8**, survivors median dll −2.7, the rest −2447. Exactly the identity element
survives; all 7 mapped images are catastrophic.

**Root cause — `SkyMove`'s sky-mode maps are NOT symmetries of this likelihood.**
Direct test (`scripts/diagnostics/flow_proposal_harness/map_test.py`): apply each
group element to a walker's OWN converged params and score against that walker's
own residual. Identity = 0.0 exactly (test is sound); the other seven cost
**−32 to −105,000 nats**, every leaf:

| leaf | g1 | g2 | g3 | g4 | g5 | g6 | g7 |
|------|----|----|----|----|----|----|----|
| 0 | −81632 | −378 | −77794 | −2388 | −105715 | −2416 | −107595 |
| 1 | −12302 | −102 | −12527 | −575 | −8077 | −482 | −8384 |
| 2 | −2575 | −31648 | −18887 | −31165 | −20452 | −16941 | −21544 |
| 3 | −46 | −722 | −649 | −1988 | −1511 | −582 | −880 |
| 4 | −2461 | −105 | −2249 | −270 | −2856 | −124 | −3212 |
| 5 | −3212 | −277 | −3556 | −353 | −5468 | −33 | −3416 |

The images are nonetheless genuinely quasi-degenerate **as regions** —
cross-mode splices of *real converged samples* accept at 0.51–0.60. So the modes
exist and are interchangeable; you simply cannot reach them by applying these
maps. (Marginals of the mapped points match the target image to ~1σ, so the maps
get the envelope right and the razor-thin correlation structure wrong.)

**Implications, in order of value:**
1. **`SkyMove` is close to a no-op for mode hopping.** Its logged 13–15%
   acceptance is dominated by the k=0 identity of `long_transform` (25% of its
   proposals are the current point → auto-accepted). Real hops are rare, so the
   sky-mode occupancy in this run is largely **init-frozen** and the per-source
   sky posteriors are NOT properly explored — a science-quality issue
   independent of the flow work. (This restores the original "frozen modes"
   reading; the intermediate "multimodality is dynamic" correction was based on
   a lattice-labelling artifact.)
2. **A correct degeneracy transformation is now the highest-value target.** It
   would fix `SkyMove` *and* unlock the fold (one island, 8× data → the regime
   where the flow measures 0.25–0.49). Acceptance test is 5 minutes:
   `map_test.py` must return Δlogl ≈ 0 on all 8 images.
3. Until then: keep the mixture (it is the best measured config), and do not
   trust the MBH sky posteriors from runs using the current SkyMove.
