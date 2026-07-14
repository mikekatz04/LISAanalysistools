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

**Step 5 — mode-mixture conditioning** (GATED: only if leaves 3/4/5 still
lag after Step 3). Mode structure is estimated FROM THE BUFFER, per leaf,
each training round, trainer-side — never imposed (the 8-mode sky lattice
was only a diagnostic frame; joint-space islands are the real objects):
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
