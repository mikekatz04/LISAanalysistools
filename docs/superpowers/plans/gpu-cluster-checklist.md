# v8-noise-merge — GPU / cluster validation checklist

Everything CPU-checkable on the `v8-noise-merge` branch is done and green;
this file is the ordered list of what MUST run on the cluster (2× H100,
cuda13x) before v8 production and before enabling any coarse mode there.
Run cheapest-first; each gate is cheap relative to the one after it.

## 0. Build preflight

- [ ] Pull the branch; `install.sh` per the sprint recipe (the merge is pure
  Python for LAT, but the deving stack rebuild policy applies; GBGPU/BBHx
  unaffected by this branch).
- [ ] `python -m unittest tests.test_unequal_arm_noise tests.test_coarse_wdm
  tests.test_psd_delayed_acceptance tests.test_unequal_arm_wiring` on a GPU
  node — the cupy-gated tests stop skipping there:
  `test_wdm_bases_on_gpu_match_cpu` (fold / layer_constant /
  layer_calibrated CPU-vs-CuPy covariance parity + device type) and
  `test_gpu_statistics_device_resident` (per-device coarse P residency +
  CPU parity).

## 1. Unequal-arm GPU unblock (plan-1 Task 3)

- [ ] Single-device method matrix beyond the unit grid: build WDM
  unequal-arm bases on CPU and CuPy at the 3-mo production grid
  (Nf 1440, MIN/MAX_FREQ per submit script) for each `wdm_psd_method`;
  compare shapes/devices/finiteness/Hermiticity/values (tolerance from
  measured arithmetic, not assumed 1e-14). Watch the ONE-TIME build wall
  time and peak memory (expected order: bases ~(2,3,3,Nf_active,Nt_active)
  float64; ~106 MiB/device at the 6-mo grid, smaller at 3-mo).
- [ ] Two-GPU cache smoke: `GPUS=0,1`, warm the shared settings path on both
  devices, assert each cached basis is device-resident (the
  `current_device` key fix, `fused_unit_covariances_v2`). A single-GPU run
  cannot catch this.
- [ ] `layer_calibrated` validity warning must NOT fire at
  MAX_FREQ=2.5e-2 (12.5% of Nyquist — safe regime).

## 2. Noise-block performance (plan-1 §2.2 re-bench)

- [ ] On a dev-representative GPU config, re-run the noise-block timing:
  dev's ~43 s → ~9 s/iteration win must survive the merged
  (doubly-optimized) psdmove. `PSD_DEBUG_CHECKS=1` restores the NaN guard
  for bisecting any numerical difference. Report the one-time basis
  build/upload cost separately.

## 3. v8 activation preflight (plan-1 Task 5)

- [ ] `sbatch scripts/fstat_proposal/submit_gf_3mo_v8.sh` dry pass: the
  `[V8-NOISE]` preflight must print delay-table OK (+ sample count) and the
  build must log `[unequal-arm] link-delay table ...` (stride/epochs/span/
  digest, anchored at data_t0), `[galfor-modulation] ... anchored at
  data_t0`, and `[noise-model-identity] {...}`.
- [ ] Negative controls (once each, then revert): misspell
  `WDM_PSD_METHOD`; unset `GALFOR_MODULATION_PATH`; point `NOISE_FILE` at a
  brick without `/ltts`; try resuming a v7 store. EVERY one must fail at
  startup — none may silently run equal-arm/stationary.
- [ ] First v8 launch: fresh store `gf_prod_3mo_v8`; watch the first
  iterations for the unequal-arm one-time build, then normal noise-block
  walls.

## 4. v8 physics acceptance (plan-1 §6.9)

- [ ] `scripts/noise/whitening_test.py` on the first v8 snapshot:
  instrument-only ⟨w²/C⟩ ≈ 1.000x; the galfor fit must land with fk/f_1/f_2
  IN BAND (the new prior ceilings) — the 2-yr fit went 0.983 → 1.0002.
- [ ] Standard v-series monitor comparison v8-vs-v7 (same GB config by
  construction; the noise marginals are the new content).

## 5. Coarse sidecar on GPU (plan-2 T8 — only after 1-4 are green)

All CPU logic (per-walker statistics bitwise, DA kernel exactness,
distributional gate, mode dispatch, source-move guard) is verified; the
cluster validates the DEVICE side and measures whether the feature pays.

- [ ] `COARSE_Q>1` + `COARSE_GPU_MODE=delayed_acceptance` on a small
  all-source GPU config: coarse CPU-vs-GPU parity at fixed inputs
  (statistic, candidate covariances, batch log-likes) and
  delta-log-likelihood over proposal-scale moves (absolute offsets are not
  the acceptance diagnostic).
- [ ] Two-GPU: walkers sharded 12/12; per-device P/qeff residency; the
  source-move guard passing at every noise→source boundary
  (`GF_MOVE_TIMING=1` exercises the instrumented loop; the guard runs in
  all dispatch paths).
- [ ] Unequal-arm + coarse WS qeff is CPU-gated at the coarse-basis
  precompute (deliberate); use `COARSE_USE_WS=0` (Bartlett) with
  unequal-arm+GPU+coarse, or equal-arm, until that is ported.
- [ ] Measure and report separately (spec §10): P_w refresh per noise
  block; coarse proposal score time; stage-1 survival fraction; fine
  stage-2 count/time; fine cold-publication time; packed-buffer repack;
  source-move time after handoff; peak memory (persistent vs transient).
  **Activation rule (spec): delayed acceptance ships only if it beats
  exact GPU scoring end-to-end after statistic refresh + mandatory fine
  publication; otherwise search_approx stays a staged-init tool.**

## Known deferred items (recorded, deliberate)

- Residual epochs (skip P_w refresh when no source move intervened) —
  spec Phase 5, after correctness soak.
- Fused 3×3 coarse likelihood kernel / six-element statistic storage /
  CUDA graphs — only if profiling shows CuPy temporaries dominate.
- search_approx stage-labelling enforcement at recipe setup.
- Exact-`fold` unequal-arm coarse WS qeff on GPU.
