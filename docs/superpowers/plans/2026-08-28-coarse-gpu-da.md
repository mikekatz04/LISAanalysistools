# Coarse-WDM GPU PSD/GALFOR (Plan 2: delayed acceptance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Executes `coarse-wdm-gpu-psd-galfor-plan.md` (sprint root; the spec) on the
> `v8-noise-merge` branch, after plan 1
> (`2026-08-27-v8-noise-merge.md`) closed. Checkboxes track progress.

**Goal:** PSD/GALFOR proposals scored against a coarse per-walker WDM
statistic — usable on GPU — while every source move still sees a complete,
current fine covariance, and production PE keeps one fine target via
two-stage (delayed) acceptance.

**Architecture:** Fine backend stays canonical. A runtime-only sidecar
(`CoarseWDMRuntime`) owns per-walker coarse statistics `P_w`, frozen `Qeff`,
and the coarse backend; PSD/GALFOR moves gain a coarse scoring callback and a
two-stage acceptance wrapped AROUND eryn's stretch machinery (stage 1 = eryn
verbatim with coarse log-likes injected into the working state; stage 2 = a
revert pass over stage-1 survivors using fine/coarse ratios — priors and
stretch factors cancel). `state.log_like` carries fine values everywhere
outside stage 1, so identity and fancy temperature swaps are fine-valued by
construction. All numerics are written xp-generically: the numpy path is the
CPU-parity test surface; cupy paths are cupy-gated tests + a cluster
checklist (no CUDA on this laptop).

**Spec:** `coarse-wdm-gpu-psd-galfor-plan.md`. Where it and the code
disagree, the interrogation table below is ground truth.

## Interrogation results (2026-08-27 ~23:15, tree `239e83ff`)

| Spec claim | Verdict |
|---|---|
| all-source gate `allowed={psd,galfor,sgwb}` + `update_from_residual` seam named in the error | ✓ run.py:1098-1106 |
| one shared `CoarseWDMStatistic` from unsubtracted data on every AC | ✓ run.py `_prepare_coarse_wdm_runtime` (:1087-...) threads one stat; `AnalysisContainer(coarse_stats=...)` routes at analysiscontainer.py:878 |
| GPU refusal + CPU-backend check + real-WDM-only + Composite-only | ✓ :1110-1135 |
| fiducial policy machinery (injection/initial + atomic sidecar + resume refusal + WS/Bartlett) | ✓ :1138-1215; Bartlett skips the fiducial build entirely |
| batch APIs needed by §5.4 | **partially exist**: `coarse_wdm_log_likelihood_batch(_frequency_terms)` (coarsewdm.py:227/:318) score a covariance batch against ONE shared stat; per-walker-P and device dispatch are the new work |
| component seam for frozen branches | ✓ `component_covariance` (sensitivity.py:6649), `covariance_from_params` (:6701); PSDMove `_prepare_fixed_component_covariances` already exists |
| knobs on `NoiseGeneralSettings` (COARSE_Q/USE_WS/FIDUCIAL) + validation | ✓ variants/noise.py:179-184, :316-324 |
| "implement DA at the acceptance layer, not the callback" | confirmed necessary AND cheap: `PSDMove.run_move` (:1861) = `super().propose` (eryn stretch, accepts via state.log_like) + identity swap every repeat + fancy swap every `permute_every` via `temperature_control.temperature_swaps(compute_log_like=...)`. Stage 1 can reuse eryn verbatim by swapping coarse values into `tmp_state.log_like` for the inner accept; stage 2 reverts survivors that fail the fine/coarse ratio. No eryn rewrite. |
| publication sequence at end of `propose()` | ✓ propose (:2010+) writes accepted coords, rebuilds each walker's sens_mat, repacks ACA — the seam Phase-3 hardening strengthens |
| coarse basis reuse for unequal-arm | ✓ run.py already streams fused unit bases into coarse cells (`fused_unequal` branch) — interacts with plan-1's Task-3 refactor only through `_folded_unit_column`, which kept its host contract |

Deltas from the spec:
- The spec's `COARSE_GPU_BATCH_BYTES`/`COARSE_GPU_DEBUG` knobs land as
  documented fields but plumb into the runtime lazily (no dead config).
- Spec §9 names `tests/test_psd_move_batched.py` for DA tests; DA gets its
  own `tests/test_psd_delayed_acceptance.py` instead (the batched file is
  already 650+ lines of unrelated fixtures).
- "GPUCoarseWDMRuntime" is spelled `CoarseWDMRuntime` (xp-generic; nothing
  GPU-specific in its contract).

## Global Constraints (inherited + plan-2 specific)

- All plan-1 constraints (one guarded compute job, 8 GB box, no push, no
  balloon test locally, deepcopy/pickle rules, wt_run.sh runner).
- The coarse likelihood is an MCMC surrogate, never evidence.
- Noise-only behavior at `COARSE_Q>1` stays bit-for-bit (golden C below).
- `delayed_acceptance` is the only coarse mode valid for production PE;
  `search_approx` only in explicitly search-labelled stages.
- Every noise move returns with fine covariance + packed ACA current; source
  moves get an assertion-backed precondition, never a silent rebuild.
- No array modules or device arrays on the settings tree; the runtime is
  runtime-only with explicit `__getstate__`.

---

### Task 1: Coarse goldens (spec Phase 0)

**Files:** `.wtenv/golden_fixed_state.py` (extend), `goldens_v8/`

- [ ] Step 1: extend golden mode `a` with env `GOLDEN_COARSE_Q` (sets
  `COARSE_Q`, CPU noise-only lite): record coarse log-like values/digests,
  the shared statistic `P`/`Qeff` digests, and the fine-published sens_mat
  digests after one seeded `PSDMove.propose`.
- [ ] Step 2: run at the current tip → `goldens_v8/c_base.json`; run twice,
  assert deterministic.
- [ ] Step 3: unit-fixture reference: a tiny direct
  `CoarseWDMStatistic.from_wdm_signal` + `update_from_residual` NPZ (small
  synthetic WDMSignal, seeded) → `goldens_v8/coarse_stat_ref.npz` — the
  bit-parity target for Task 2's batched builder.

### Task 2: `CoarseWDMRuntime` + per-walker batched statistics

**Files:** Create `src/lisatools/coarsewdm.py` additions (same module);
Test: `tests/test_coarse_wdm.py` (extend)

**Interfaces (produced):**
```python
@dataclasses.dataclass
class CoarseWDMRuntime:
    coarse_settings: CoarseWDMSettings      # holds fine_settings
    coarse_backend: CompositeSensitivityBackend
    qeff: np.ndarray | None                 # (Nf_active, Ncoarse) or None (Bartlett -> cell sizes)
    use_ws: bool
    fiducial_digest: str
    mode: str                               # "off" | "search_approx" | "delayed_acceptance"
    batch_bytes: int
    # runtime-only, per __getstate__-excluded dict:
    #   per-device qeff copies, per-walker P (nw, 3, 3, Nf_active, Ncoarse),
    #   residual epochs {walker: int}
    def refresh_P(self, acs, walkers=None) -> None
    def P_rows(self, walkers) -> array      # view on the owning device
    def coarse_log_like_batch(self, covariances, walker_inds) -> np.ndarray
```
`build_P_batch(residuals, settings, chunk_bytes)`: rectangular-prefix view
`(3, Nf_active, Ncoarse_full, Q)` + ragged tail, one einsum per chunk —
xp-generic; bit-parity vs `_coarse_sample_covariance` on numpy required
(same reduction order: mean over the cell axis last).

- [ ] Step 1: failing tests — (a) `build_P_batch` == per-walker
  `_coarse_sample_covariance` bitwise (divisible and ragged Q); (b) chunking
  invariance; (c) `coarse_log_like_batch` with per-walker P == looped
  `coarse_wdm_log_likelihood` per walker (uses a per-walker `stat` clone);
  (d) degenerate-cell masks match CPU rules; (e)
  `pickle.loads(pickle.dumps(copy.deepcopy(runtime)))` drops runtime arrays.
- [ ] Step 2: implement; run; goldens A/B/C unchanged (runtime is inert
  until wired).
- [ ] Step 3: commit `feat(coarsewdm): per-walker batched coarse statistics
  + CoarseWDMRuntime`.

### Task 3: dual-backend all-source runtime + config surface (spec §4.1, §7)

**Files:** `run.py` (gate), `variants/noise.py` → shared lift,
`variants/all_sources.py`, `stock/base.py` if the lift needs it;
Test: `tests/test_coarse_wdm.py` + `tests/test_unequal_arm_wiring.py` style

- [ ] Step 1: lift `coarse_Q/use_ws/fiducial` fields to the shared settings
  base both variants inherit; add `coarse_gpu_mode` (`COARSE_GPU_MODE`,
  default "off"), `coarse_gpu_batch_bytes` (`COARSE_GPU_BATCH_BYTES`,
  bounded default 256 MiB), `coarse_gpu_debug` (`COARSE_GPU_DEBUG`, 0).
  Validation matrix (spec §7): all-source + Q>1 + mode=off → error;
  `search_approx` requires a search-labelled stage (checked at recipe
  setup); real WDM + psd branch required; unsupported components fail at
  setup.
- [ ] Step 2: `run.py`: factor the reusable parts of
  `_prepare_coarse_wdm_runtime` (fiducial resolution, Qeff, coarse backend
  build) into a helper shared by (a) the UNCHANGED noise-only path and (b) a
  new all-source path that builds a `CoarseWDMRuntime` sidecar on
  `general_info` and leaves `ac.coarse_stats = None`. The noise-only
  behavior must stay bit-for-bit (golden C gate).
- [ ] Step 3: extend `noise_model_identity` (plan-1 machinery) with
  `coarse_mode/coarse_Q/coarse_use_ws/fiducial_digest` so a resume across a
  coarse-mode change is refused for free.
- [ ] Step 4: construction-level tests (validation matrix, identity fields,
  pickling); golden C bit-check; commit.

### Task 4: device-aware coarse candidate covariances + fixed caches (spec §5.3)

**Files:** `psdmove.py` (extend `_prepare_fixed_component_covariances` to a
coarse variant keyed by cold-coordinate digest + device), `sensitivity.py`
only if a cache key needs the device added;
Test: `tests/test_coarse_wdm.py`

- [ ] Step 1: failing tests — coarse candidate covariance for a PSD-only
  proposal (galfor frozen) == full coarse build at the same params (CPU);
  same for GALFOR-only; fixed-cache invalidation when the frozen branch's
  cold coords change (digest key).
- [ ] Step 2: implement via the existing `component_covariance`/
  `covariance_from_params(fixed_covariances=...)` seam on the coarse
  backend; device goes into every cache key.
- [ ] Step 3: commit.

### Task 5: PSDMove coarse callback + P_w lifecycle (spec §6.1, §4.3-4.4)

**Files:** `psdmove.py`; Test: `tests/test_coarse_wdm.py`

- [ ] Step 1: `compute_coarse_log_like(coords, ...)` — same row semantics as
  `compute_log_like` but scoring through
  `runtime.coarse_log_like_batch`; never touches `state.log_like`,
  `ac.sens_mat`, or the ACA buffer. `propose()` in a coarse mode refreshes
  `P_w` from every walker's residual at entry (correctness-first; epochs are
  spec Phase 5 and NOT built now).
- [ ] Step 2: tests — coarse callback vs direct batch scoring; refresh
  actually re-reads mutated residuals; fine callback untouched.
- [ ] Step 3: commit.

### Task 6: `search_approx` mode + fine-publication hardening (spec Phase 3, §6.3-6.4)

**Files:** `psdmove.py`, `recipe.py` (stage labelling read),
`globalfitmove.py` (precondition helper);
Test: new `tests/test_noise_source_handoff.py`

- [ ] Step 1: in `search_approx`, the inner loop (`run_move*`) swaps the
  Model's likelihood fn to the coarse callback and keeps coarse values in a
  move-local array; on exit `propose()` publishes the accepted cold fine
  state through the EXISTING sequence, then recomputes the fine cold
  log-like into `new_state.log_like[0]` and marks tempered rows as
  diagnostics (move-local; never `GFState.log_like`).
- [ ] Step 2: `ensure_fine_noise_covariance_current(acs, general_info)` in
  `globalfitmove.py`: asserts basis settings, fine shape, packed-buffer
  binding (`sens_mat.invC` is a view of `linear_psd_arr`), device ownership.
  Called at source-move entry (common base). Debug env
  `COARSE_GPU_DEBUG=1` re-derives one walker's fine covariance and compares.
- [ ] Step 3: lifecycle test: a small CPU all-source-shaped fixture
  (`gb_no_fg_lite`-style GB branch + psd/galfor, synthetic) runs
  noise(search_approx) → GB move; asserts the GB move observed current fine
  state (the assertion helper passes) and that disabling publication makes
  it fail (negative control).
- [ ] Step 4: commit.

### Task 7: delayed acceptance (spec Phase 4 — the production kernel)

**Files:** `psdmove.py`; Test: new `tests/test_psd_delayed_acceptance.py`

**Design (locked during interrogation):** per repeat, in
`run_move`'s stage-1 call:
1. cache `fine_ll` (module-ladder shaped) for the current ensemble
   (invariant maintained across repeats; initialized once per propose).
2. compute `coarse_ll_x` for the current ensemble (or reuse cached);
   set `working_state.log_like = coarse_ll_x`; run `super().propose`
   with `compute_log_like = compute_coarse_log_like` → eryn performs the
   EXACT stage-1 accept (prior + stretch factor + tempered coarse ratio).
3. rows accepted in stage 1: evaluate fine ll of the new coords;
   `log_alpha2 = beta * ((Lf_y - Lc_y) - (Lf_x - Lc_x))`; independent
   uniforms; REVERT stage-2 rejects (coords, prior, and both ll caches) to
   the pre-propose copy.
4. restore `state.log_like = fine_ll` (updated for survivors) before the
   temperature swaps — identity swaps use fine values; fancy swaps get the
   FINE callback. Acceptance tallies count only stage-2 survivors.
- [ ] Step 1: failing tests, all CPU with `Q>1` noise-only lite fixtures:
  (a) **forced stage-1 accept** (patch coarse ll to +const) ⇒ transition
  == plain fine MH with the same RNG stream (exact array equality of the
  accepted ensemble);
  (b) **Lc ≡ Lf** (patch coarse callback to the fine one) ⇒ stage 2
  accepts everything to float precision and the chain equals the ordinary
  fine move run with the same seeds;
  (c) beta ≠ 1 rung: hand-computed `log_alpha2` on a 2-walker fixture;
  (d) reverted walkers bit-equal their pre-propose state (incl. caches);
  (e) fancy-swap path receives fine values (spy on the callback).
- [ ] Step 2: implement (mode `delayed_acceptance`); publication at end of
  propose reuses the accepted cold fine candidate when cached (else
  rebuild); run tests + goldens (A/B/C unchanged — DA off by default).
- [ ] Step 3: statistical gate (one guarded run, memory-checked first):
  noise_only_lite, ~500 iterations, exact-fine vs DA — PSD/GALFOR cold
  marginals consistent (KS p > 0.01 per parameter) and swap-acceptance
  bookkeeping sane. Runtime bounded (~lite grid, CPU); run via wt_run.sh.
- [ ] Step 4: commit.

### Task 8: GPU code paths + cluster checklist (spec Phases 2/5 device side)

**Files:** `coarsewdm.py` (device contexts in refresh/score), cupy-gated
tests; plan-doc checklist update

- [ ] Step 1: device plumbing — `refresh_P` enters each walker's owning
  device context (walker→device via the ACA maps, mirroring
  `_score_walker_batch`); per-device Qeff copies; rows grouped by device in
  `coarse_log_like_batch`, chunked under `batch_bytes`.
- [ ] Step 2: cupy-gated tests (skip locally): device residency of P/Qeff,
  CPU-vs-GPU parity of statistic + likelihood at fixed inputs, two-device
  cache separation.
- [ ] Step 3: extend the cluster checklist (plan-1 readout doc): coarse
  CPU/GPU parity matrix, delta-log-like over proposal-scale moves, P_w
  refresh + score timings, stage-1 survival fraction, peak memory split
  (persistent vs transient), and the §10 performance-acceptance table —
  activation decision is measurement-gated on the cluster, per spec.
- [ ] Step 4: commit + final readout, memory update.

## Execution log (2026-08-28 ~00:45)

- T1 DONE: coarse (Q=8) goldens a/b recorded + deterministic
  (`goldens_v8/c_base_*.json`); bonus check: coarse ll == fine ll to 4 ulp on
  the stationary lite covariance, as the algebra predicts (WS dof of a
  time-constant cell = cell size exactly).
- T2 DONE (`61d96e72`): build_coarse_P_batch bitwise vs reference (einsum
  with leading walker axis IS bit-identical), per_row_P on the batch
  scorers, CoarseWDMRuntime with pickle hygiene. 6 new tests.
- T3 DONE (`6deed128`): knobs lifted to EreborGeneralSettings +
  coarse_gpu_mode/batch_bytes/debug; validate_coarse_settings matrix (6
  tests); run.py sidecar fork (fine backend canonical, ACs keep
  coarse_stats=None); coarse identity fields on the persisted noise-model
  identity. Noise-only path byte-preserved (coarse golden B).
- T4+T5 DONE (`23298e2d`): compute_coarse_log_like + coarse frozen-component
  cache + propose-entry refresh_P; recipe threads the runtime. 4
  unbound-stub plumbing tests.
- T6 MOSTLY DONE / T7 DONE (this commit): mode dispatch
  (_resolve_mode_like_fns; search_approx = coarse everywhere,
  delayed_acceptance = fine invariant + coarse stage-1 + fine swaps; 3
  tests), _propose_delayed_acceptance (stage-1 eryn verbatim with TC
  detached — eryn's propose runs temper_comps internally, re-applied after
  the fine restore so ALL swaps act on fine values; stage-2 revert pass;
  prior-mask trick spends fine evals on survivors only),
  ensure_fine_noise_covariance_current + tests. DA kernel verification:
  Lc==Lf bit-equivalence with the plain fine move (global np.random pinned —
  eryn shuffles splits with GLOBAL np.random, red_blue.py:124),
  independently recomputed stage-2 ratios, bit-exact reverts, beta scaling,
  and a distributional gate: with a deliberately WRONG surrogate the DA
  chain matches the analytic fine target (tau ~31-35 iters both samplers;
  KS on iteration-thinned samples; symmetric criteria on the fine
  reference). Found+fixed latent GFState copy crash for dict-constructed
  states (is_eryn_state_input snapshot).
- T6 COMPLETE + T8 CODE COMPLETE (2026-08-28 ~00:00, commits `642890de` +
  final):
  * Guard wiring: `ensure_fine_noise_covariance_current` derives all state
    from the runtime and runs in GFCombineMove before every SOURCE sub-move
    across all three dispatch paths (the plain fast path defers to eryn only
    when no sidecar is active); MaxLogLCombineMove inherits via
    `_propose_moves_once`. Corruption negatives are unit-level (an in-loop
    corruption cannot trip the guard in integration — the next PSD
    publication heals it, by design).
  * Lifecycle smoke `tests/test_noise_source_handoff.py` (9 s!): noise-lite
    + a SIMPLE-API dummy source branch (add_branch + FunctionMove) makes
    the run all-source-shaped; run-level `coarse_gpu_mode` opt-in after
    build (documents the branch-keyed run gate vs the variant validator);
    2 real engine iterations of search_approx noise -> source stand-in,
    asserting sidecar engagement, per-walker statistics, fine-only state at
    every boundary, finite fine cold lls. Caught two real integration bugs:
    the ACA's `gpu_map` is an int ARRAY (all-zeros on CPU; only meaningful
    with `gpus` set) — `_device_groups` now handles dict/array/None — and
    combine-loop `accepted` dtype mixing (test-side).
  * A gb-bearing all_sources_lite probe was ABANDONED after 25 min pegged
    in the CPU GB F-stat grid fit: the GB-consumer leg of the handoff is
    cluster work (checklist §5), as the spec anticipated.
  * T8: per-device statistic stores ({device: (walkers, P)}) built under
    owning-device contexts, per-device qeff/template copies, strict
    wrong-device refusal, device-grouped scoring in psdmove mirroring
    `_score_walker_batch`; multi-group logic fully CPU-tested via integer
    device keys; cupy-gated residency/parity tests skip locally.
  * Cluster checklist: `docs/superpowers/plans/gpu-cluster-checklist.md`
    (ordered gates 0-5 incl. the spec-§10 performance-acceptance table and
    the activation rule).

## Self-review

- Spec coverage: §3.1→T2, §3.2→T7, §3.3→T6, §4.1/4.2→T2/T3, §4.3/4.4→T5/T6
  (epochs deliberately deferred = spec Phase 5, recorded, not silently
  dropped), §5.1/5.2→T2+T8, §5.3→T4, §5.4→T2+T8, §6.1→T5, §6.2→T4/T5,
  §6.3/6.4→T6, §7→T3, §8 sequence preserved (Phase 0=T1 … Phase 4=T7;
  Phase 5 caching/fusion explicitly out of local scope), §9 file list
  honored with the two named deltas, §10 gates: numerical→T2/T4 tests,
  MCMC→T7 tests, handoff→T6 test, multi-GPU/persistence→T8 checklist +
  T3 identity, performance→cluster.
- All heavy runs go through wt_run.sh, one at a time; the only long local
  run is T7 Step 3 (bounded lite grid).
