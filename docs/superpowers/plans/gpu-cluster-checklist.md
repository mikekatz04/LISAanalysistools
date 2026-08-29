# v8-noise-merge — GPU / cluster validation checklist

Status as of **2026-08-29**, after probe jobs **369** (exact-fine) and **376**
(coarse). `[x]` items are verified ON THE CLUSTER with the evidence quoted;
`[ ]` items are genuinely still open. Everything CPU-checkable is done.

Snapshot artifacts: monitor page for job 369 at
https://claude.ai/code/artifact/5d9c394f-203b-48f5-b08b-7596f3b074f0

## 0. Build preflight

- [x] Pull + `install.sh`. NOTE the cluster install is **editable**
  (`PROBE-0a` reported lisatools resolving to `.../lisa-analysis-tools/src/`),
  so pure-Python changes go live on `git pull` alone. `install.sh` is needed
  only for the NATIVE commits, which come from the v7 side, not this work:
  `0f0fc73a` (fused GB accept kernels), `07634536` (`__CUDA_ARCH__` guards),
  `349f32a7` (cutils backend-methods signature). **No commit in the noise
  merge touches C/C++/CUDA.**
- [x] Cupy-gated unit tests un-skip on a GPU node: job 376 `probe0_unittests`
  ran **85 tests, OK** (369 ran 76). Two GPU-only bugs were found and fixed
  this way — CPU-intent fixtures resolving to CUDA (`e6f4a486`) and
  BLAS-dependent Hermiticity (`85f50208`).

## 1. Unequal-arm GPU unblock (plan-1 Task 3)

- [x] Per-device CPU-vs-GPU parity for all three `wdm_psd_method` values, on
  BOTH devices (job 369 `[SMOKE]`): `fold` max_rel 5.99e-16, `layer_constant`
  0.0, `layer_calibrated` 5.77e-16 — machine precision.
- [x] **Two-GPU cache smoke** — the one thing a single-GPU run can never
  catch: one settings object + one shared `basis_cache` entered under both
  devices gave `shared-cache entries=2 basis devices=[0, 1]`, confirming the
  `current_device` cache-key fix (`fused_unit_covariances_v2`).
- [x] `layer_calibrated` validity warning does NOT fire on the production
  band: the run's own drift is **6.320e-07** on all four evaluations (3
  orders under the 1e-4 tolerance), correction spanning [0.99996, 1.08653]
  over 3240/3240 entries. The two >tolerance warnings in the logs come from
  the probe's own unit-test block on an unrestricted toy grid.
- [x] One-time basis build is cheap at this grid (startup to first sampling
  ~2 min in job 369; `layer_calibrated` is the ~200x path by design).

## 2. Noise-block performance

- [x] Exact-fine baseline (job 369, 2xH100): `psd_pe` 8.995 s first call then
  **5.872 s** steady; `galfor_pe` **5.12 s**; `vgb_pe` 13.12 s; cycle 24.41 s;
  iteration wall 245–248 s. Host RSS flat 23.8/24.2 GB (no leak); GPU pool
  oscillates 11.5–18.1 GB.
- [x] Coarse leg (job 376): `psd_pe` **2.317 s (2.53x faster)**, `galfor_pe`
  **5.92 s (1.16x SLOWER)**, `vgb_pe` 13.161 s (unchanged — a clean control),
  cycle 21.40 s, noise block 8.24 vs 11.0 s = **~25% faster**.
  DO NOT quote the raw `[SAVE]` walls as a speedup: 376's are noise_SEARCH
  iterations, 369's are noise_VGB_search — different stages.
- [ ] **OPEN (Robbie): galfor's band-limited fast path.** `psd` wins because
  its unequal-arm bases are cached and parameter-independent; galfor's
  spectral SHAPE is sampled and only the TIME axis coarsens, while
  `_build_coarse_covariance_batch` loops per row where the exact path
  batches. `PSDMove._compute_galfor_subband_loglike` already implements the
  band-limited answer (cache an exact full-band baseline, rescore only the
  ~25 of 178 layers the foreground touches) but is gated behind
  `_coarse_batch_fast_path_available()`, i.e. the shared-statistic noise-only
  mode — dark in the all-source sidecar. Porting it + vectorising the
  candidate build should take the noise block to ~3x.

## 3. v8 activation preflight

- [x] The wiring resolves correctly (both probes): `[unequal-arm] link-delay
  table ... stride=200, 126233 epochs over [9.772994e+07, 1.608459e+08] s,
  anchored at data_t0=9.772994e+07, digest=f1f3f00ea5d9cf13`;
  `[galfor-modulation] ... 199 epochs covering [0, 6.31162e+07] s of the data
  frame`; `[noise-model-identity] {...}` persisted into the h5 (12 attrs incl.
  `coarse_fiducial_digest`).
- [ ] Negative controls, once each then revert: misspell `WDM_PSD_METHOD`;
  unset `GALFOR_MODULATION_PATH`; point `NOISE_FILE` at a brick without
  `/ltts`; try resuming a store with a different noise identity. Each MUST
  fail at startup rather than silently running equal-arm/stationary.
- [ ] First v8 production launch: fresh store `gf_prod_3mo_v8`.

## 4. v8 physics acceptance — THE REMAINING SCIENCE GATE

- [ ] `scripts/noise/whitening_test.py` on the first v8 snapshot:
  instrument-only ⟨w²/C⟩ ≈ 1.000x (the 2-yr fit went 0.983 → 1.0002 once
  `f_1` came in band).
- [ ] **GALFOR RAILING — the open question.** Both probes drove the foreground
  to prior edges while GB is still EMPTY (the galaxy is entirely unsubtracted,
  so galfor absorbs it): job 369 had `f_1` climbing to 1.4e-6 below its 1e-2
  ceiling; job 376 had `amp` pinned at its 1e-41 ceiling and `f_2` at its
  1e-5 floor, with `alpha` 1.6e-3 from its 5.0 ceiling. Not yet established as
  a fault — it may be the search doing its job against an unsubtracted galaxy.
  **Re-check once `gb_search` populates leaves.** If galfor is still railed
  after GB subtraction, the ceilings are binding and the 2-yr plateau lesson
  is repeating at a new location.
  The clean A/B is `NOISE_PROBE_EXACT=1 sbatch submit_gf_noise_multigpu_probe.sh`
  for the same iteration count — it isolates the surrogate as a suspect.
  (Also: job 376 walker 15 is a single-walker outlier across 4 of 5 galfor
  params — worth a glance.)
- [ ] v8-vs-v7 monitor comparison (GB config identical by construction; the
  noise marginals are the new content).

## 5. Coarse sidecar on GPU

Ran ahead of gates 3–4 and is largely green; delayed acceptance is EXACT, so
these gates measure efficiency, not correctness of the sampled chain.

- [x] Ran clean on GPU to stored iteration 5, zero errors, coarse engaged:
  `coarse WDM sidecar runtime (all-source, mode=auto): Q=8, Nt_active=2121 ->
  Ncoarse=266, weighting=WS, fiducial=injection, digest=592e72d9d160108d`.
- [x] **WS weighting works on GPU** — the `fused_unequal` branch flagged as
  never-executed. The old "use `COARSE_USE_WS=0` (Bartlett) until ported"
  advice is SUPERSEDED; the exact-fold coarse basis was unblocked in
  `059fc73f` (device path = cell-mean of the cached fine bases).
- [x] Both GPUs active under coarse; noise-window peaks 26,252 / 7,826 MiB,
  i.e. −2% / −4% vs exact-fine (coarse saves no memory — expected, the fine
  residuals stay resident).
- [ ] Read the **`[COARSE_AUDIT]`** lines (added `89ffdea8`, so they appear
  from the next launch): stage-2 acceptance and the `|dlogl|` spread ARE the
  surrogate-accuracy metric (0 / 100% = an exact surrogate), and stage-1
  survival is the efficiency number spec §10 asks for. Poor numbers cost
  efficiency, never correctness — lower `COARSE_Q` if so.
  `scripts/noise/coarse_q_scan.py` is the offline tool for choosing Q; it
  builds on `run_noise_only` and needs the mojito brick, so it is cluster-only.
- [ ] Source-move guard exercised at a real noise→source boundary — needs a
  run that reaches `gb_search` with sources present (jobs 369/376 had GB
  empty).

## Known deferred items (recorded, deliberate)

- Residual epochs (skip `P_w` refresh when no source move intervened) — spec
  Phase 5. Arithmetic says the refresh is ~ms against a ~6 s move, so this is
  a micro-optimization; the clean invalidation point is the `GFCombineMove`
  guard hook.
- Fused 3×3 coarse likelihood kernel / six-element statistic storage / CUDA
  graphs — only if profiling shows CuPy temporaries dominate.
- `search_approx` stage-labelling enforcement at recipe setup.
- Mid-iteration checkpoint size: 450.1 MB/write, ~92% dense GB capacity — see
  the TODO in `midit_checkpoint.py`.
