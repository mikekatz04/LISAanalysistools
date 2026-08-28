# v8 Noise Merge (Plan 1: noise-dev → dev + cluster wiring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Merge `origin/noise-dev` into a dev-based integration branch, unblock the
unequal-arm noise model on GPU, and wire it (plus the foreground modulation file)
into a new `submit_gf_3mo_v8.sh`, so v8's main difference from v7 is the improved
noise setup.

**Architecture:** Textually clean merge (verified via `git merge-tree`) followed by
two surgical commits in `src/lisatools/sensitivity.py` (GPU host/device boundary +
device-aware cache key) and a settings-level wiring commit on the `all_sources`
stock variant. Local validation is CPU-only (no mojito data on this laptop; no CUDA
on macOS); GPU/cluster validation is an explicit deferred checklist.

**Tech Stack:** Python 3.12 (`deving` conda env), unittest, git worktree.

**Spec:** `noise-dev-merge-handoff.md` (sprint root, provided by user). Plan 2
(`coarse-wdm-gpu-psd-galfor-plan.md`) is a separate follow-on plan, started only
after this plan's tasks are complete and tested.

## Verified ground truth (interrogation results, 2026-08-27 night)

All line references were re-verified against the real branches on this machine.

| Handoff claim | Verdict |
|---|---|
| merge base `7ae4214d`, 7 commits on noise-dev, 3 overlap files | ✓ confirmed against the *current* tips |
| `origin/dev = 9704c4a8` | **stale** — dev moved to `86ed9353` tonight (3 GB-side commits: `1f15f28d`, `c4050096`, `86ed9353`); none touch noise files; merge-tree still conflict-free |
| `9d05cf5c` local-only, must be published | **already satisfied** — `origin/noise-dev` now points at `9d05cf5c` (pushed 2026-08-27 19:48 CDT) |
| merged `analysiscontainer.py` keeps `_same_device` + `psd_data_length` | ✓ verified in merged blob (tree `d04a187d`) |
| merged `psdmove.py` `_build_warmed` set semantics | ✓ set at init (:198), `.add` at :893/:1130/:1189, `in` checks consistent |
| merged `run.py` has coarse runtime + `iteration == 0` guard | ✓ (:989, :1258, :1365, :1752) |
| GPU gate in `_unit_bases` (nd :5010) + coarse gate (:5065) | ✓ both present |
| `np.asarray(settings.fold_sparse_psd(stacked))` CuPy→NumPy bug (nd :4797) | ✓ confirmed |
| UnequalArm `_bases` key missing `current_device` (nd :4743-4751) | ✓ confirmed; generic key has it (nd :4110) |
| dev `_direct_base_covariance` is xp-generic and works on GPU (dev :4545) | ✓ confirmed |
| stock seams: `PSDSetup.instrument_component_cls` (:115), `noise_sensitivity_init_kwargs` (:553), `resolve_galfor_modulation` t0-less (:441), `galfor_modulation_path` env field (all_sources :212), `finalize_general` (:560) | ✓ all confirmed on dev tip |
| `modulation_unequal.dat` only on noise-dev | ✓ |
| Extra (not in handoff): `noise_sensitivity_init_kwargs` on noise-dev promotes `wdm_psd_method` from `instrument_component_kwargs` to backend-wide policy | ✓ — wiring must set it once, it propagates |
| Extra: UnequalArm `_direct_base_covariance` calls `_unit_bases` directly (uncached); cached path is `_bases`, used only when `basis_cache` is not None | noted — GPU fix must live in `_unit_bases`/`_folded_unit_column`, covering both |

## Global Constraints

- Never commit or push proactively to `origin` — the merge/fix commits land on the
  local branch `v8-noise-merge` only; the user pushes. (Committing locally on the
  integration branch IS the deliverable and is required by the plan.)
- No new global-fit settings files: wiring is dataclass fields + env knobs on the
  installed stock variant (`UNEQUAL_ARM` = capitalized attribute name rule).
- Deepcopy/pickle safety: no array modules or device arrays on the settings tree.
- Laptop CPU ≤50%: every test/bench run pins `OMP_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1`.
- NEVER run `tests.test_gbspecial_flow` locally (10-26 GB balloon; cluster-only).
- Do not "fix" the coarse-likelihood CPU-only gates (deliberate; irrelevant to
  cluster). The unequal-arm GPU gate is the one to lift.
- The main working tree (`LISAanalysistools/`, dirty with the user's mid-iteration
  checkpoint work) is never touched. All work in
  `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-noise-merge` on branch
  `v8-noise-merge`.
- The user's own plan docs override this plan where they conflict.

## Workspace facts (established)

- Worktree: `LISAanalysistools-noise-merge`, branch `v8-noise-merge` from
  `origin/dev @ 86ed9353`.
- Test invocation (shadows the editable install of the main tree):

```sh
S=/private/tmp/claude-501/-Users-mkatz-Research-lisa-sprint-2026/ae20b515-5522-45f6-aa6c-cf236cb9d3e9/scratchpad
cd /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-noise-merge
LAT_WORKTREE_SRC=$PWD/src PYTHONPATH=$S/wtenv \
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /Users/mkatz/miniconda3/envs/deving/bin/python -m unittest <targets> -v
```

  (`$S/wtenv/sitecustomize.py` strips the scikit-build-core redirect finder;
  compiled `lisatools_backend_cpu` still loads from site-packages; the merge is
  pure Python so no rebuild is needed locally.)
- Baseline at `86ed9353` (recorded in `$S/baseline_tests_{1,2}.log`): 213 tests,
  **1 known pre-existing failure**: `test_stock_globalfit.LiteVariantTest.
  test_lite_kwarg_matches_twin` (`use_gpu` None vs False; passes in the user's
  main tree because an uncommitted edit fixes it). Expected to persist unchanged
  through every task; any OTHER failure is a regression.
- `tests.test_ltt_averaging` raises SkipTest at module level — never name it
  directly in `-m unittest` invocations (aborts the loader); it self-skips under
  discover.
- No mojito data locally (`MOJITO_DATA_PATH=/shared/data/mojito_cache` is
  cluster-only); synthetic (`DATA_MODE=synthetic`) is the local fixture path.
- Baseline for v8 submit script: `scripts/fstat_proposal/submit_gf_3mo_v7.sh`
  (1078 lines; `gf_prod_3mo`, TOBS_TARGET=7776000, MIN_FREQ=4e-4, MAX_FREQ=2.5e-2,
  GPUS=0,1, cuda13x, NWALKERS=24, PSD/GALFOR_NUM_PROP_REPEATS=10).

---

### Task 1: Fixed-state goldens at the dev tip (pre-merge)

**Files:**
- Create: `$S/golden_fixed_state.py` (scratchpad; not part of the repo)
- Output: `$S/goldens_dev_tip.json` (+ `.npz` arrays)

**Interfaces:**
- Produces: `run_golden(out_json)` writing, for a fixed-seed synthetic noise-only
  build at fixed physical PSD+galfor parameters: SHA-256 digests + float64 values
  of (a) per-walker `log_like`, (b) residual buffer bytes, (c) `sens_mat.C/invC/detC`
  bytes for walker 0, (d) `linear_psd_arr` bytes, all recorded after `setup_acs`
  and again after one seeded `PSDMove.propose`.

- [ ] **Step 1:** Inspect `erebor` stock options (`erebor.get_stock_options()`) and
  the `noise_only`/`noise` variant + `scripts/campaign/` lite-gate harness; choose
  the smallest synthetic noise-bearing configuration that exercises
  `PSDMove.propose` on CPU (prefer `noise_only` with a small `TOBS_TARGET`/lite
  preset; fall back to the campaign T-gate harness if one already does this).
- [ ] **Step 2:** Write `$S/golden_fixed_state.py`: build the config with
  `DATA_MODE=synthetic`, fixed `numpy` seed, fixed noise params injected into the
  state (not `priors.rvs`), call the fit's likelihood + one `propose` with seeded
  RNG; dump digests/values to JSON+NPZ.
- [ ] **Step 3:** Run it at the dev tip in the worktree; save
  `$S/goldens_dev_tip.json`. Record wall time (this doubles as the CPU noise-block
  timing reference for Task 6).
- [ ] **Step 4:** Sanity: run twice; assert bit-identical JSON (determinism gate —
  if not deterministic, fix seeds until it is; a non-reproducible golden is
  useless).

### Task 2: The merge

**Files:**
- Modify (via merge): 149+ files; semantically reviewed: `analysiscontainer.py`,
  `moves/psdmove.py`, `globalfit/run.py`, `sensitivity.py`, `domains.py`,
  `stock/erebor/noise.py`, `stock/erebor/variants/noise.py`, `coarsewdm.py`,
  `_unequal_arm_fused.py`, `scripts/noise/*`, `tests/test_coarse_wdm.py` (+3
  modified test files)

- [ ] **Step 1:** `git merge --no-ff origin/noise-dev` with a message recording
  both parent SHAs (`86ed9353` + `9d05cf5c`) and the handoff doc. Expect zero
  conflicts (merge-tree verified). On any conflict: STOP, re-review.
- [ ] **Step 2:** Verify the three overlap files match the pre-reviewed blobs:
  `diff src/lisatools/analysiscontainer.py $S/merged_analysiscontainer.py` (same
  for psdmove.py, run.py) — must be byte-identical.
- [ ] **Step 3:** Run the merged noise suite:
  `tests.test_sensitivity tests.test_unequal_arm_noise tests.test_noise_split_moves
  tests.test_psd_move_batched tests.test_coarse_wdm tests.test_noise_globalfit
  tests.test_psd_move_multi_shard tests.test_mojito_noise tests.test_stock_globalfit
  tests.test_aca_vectorized_dispatch tests.test_wdm_domain_cpp`.
  Expected: all green except the 1 known lite-twin failure.
- [ ] **Step 4:** Golden re-check: run `$S/golden_fixed_state.py` on the merged
  tree; every digest must equal `goldens_dev_tip.json` **bit-for-bit** (defaults:
  `coarse_Q=1`, equal arms, `wdm_psd_method="fold"`; the galfor prior change moves
  ceilings, and the golden pins parameters *inside* the new ranges so priors do
  not change the recorded values — verify the chosen fixed params satisfy the new
  ceilings BEFORE Task 1 records them: fk, f_1, f_2 ≤ 1e-2).
- [ ] **Step 5:** Broader regression sweep (background, discover-mode with
  exclusions): `python -m unittest discover tests -v` **minus** balloon/GPU-only
  tests — check first for an existing skip mechanism; otherwise run the explicit
  module list from Step 3 plus the GB/engine suites named in the baseline logs.
  Same 1 known failure allowed.
- [ ] **Step 6:** Deepcopy/pickle gate: `pickle.loads(pickle.dumps(copy.deepcopy(x)))`
  for a `WDMSettings` carrying fold caches, `GalForTimeModulation`
  (`modulation_unequal.dat`, explicit t0), a constructed `PSDMove`, and the
  pre-build `noise_only` + `all_sources` stock fits (construction-level). Reuse
  existing tests where they exist; add to `tests/test_coarse_wdm.py` /
  `tests/test_unequal_arm_noise.py` only if a gap is found.

### Task 3: GPU unblock for the fused unequal-arm bases

**Files:**
- Modify: `src/lisatools/sensitivity.py` (UnequalArmInstrumentNoise:
  `_bases` key, `_folded_unit_column`, `_unit_bases`)
- Test: `tests/test_unequal_arm_noise.py` (extend)

**Interfaces:**
- Produces: `_unit_bases(settings)` returns a `(B_oms, B_acc)` tuple **on
  `settings.xp`** for FD and WDM settings alike; `_folded_unit_column` always
  returns host `np.ndarray`; `_layer_calibration` cache stays host-resident;
  coarse gates (`_build_coarse_basis_data`) untouched.

- [ ] **Step 1:** Read `domains.fold_sparse_psd` / `sparse_psd_fold_data` (merged
  tree) to pin the exact input/output shape contract and confirm leading axes
  batch (the `(basis, ch, ch)` axes already fold jointly — verify a `(Ncol,
  basis, ch, ch, Nfold)` batch folds identically).
- [ ] **Step 2:** Write failing/pinning CPU tests first:
  (a) bit-identity of `_unit_bases` output vs the pre-change implementation for
  all three `wdm_psd_method` values on a small WDM grid with 1-D and time-resolved
  delay tables (capture "before" arrays by running the merged-tree implementation
  once and saving NPZ to `$S/`);
  (b) batched-fold == per-column-fold equality on CPU;
  (c) the `_bases` cache key contains the current device discriminator (assert
  key structure via a direct `_bases`-key inspection or a second-settings cache
  poke — mirror how the generic `InstrumentNoise._bases` key test works if one
  exists);
  (d) a cupy-marked test (`@unittest.skipUnless(has cupy)`) asserting
  `_unit_bases` returns device arrays and matches CPU within tolerance — skips
  locally, runs on cluster.
- [ ] **Step 3:** Implement, keeping the host math byte-identical:
  - `_folded_unit_column`: layer-center branch unchanged (already host-only).
    Fold branch: build `stacked` on host as now; if the settings backend is
    CuPy, upload once (`settings.xp.asarray`), fold, `asnumpy` the result;
    else call as today. Explicit host return in both cases.
  - `_unit_bases` WDM branch: delete the `uses_cupy` ValueError. For the
    time-resolved `fold` path on a CuPy backend, batch columns (chunked to
    bound memory: chunk size from a fixed byte budget ~256 MB) through one
    upload → one `fold_sparse_psd` call → one download per chunk instead of
    per-column round trips. After assembling host `bases`, return
    `tuple(settings.xp.asarray(bases[i]) for i in range(2))` exactly like the
    FD branch.
  - `_bases` override key: append `current_device(settings.xp)` (import already
    present at :54) and bump the tag to `"fused_unit_covariances_v2"`.
  - `_layer_calibration`: no change (host numpy in, host numpy out; verify the
    ratio consumer multiplies host arrays before the final upload).
- [ ] **Step 4:** Run the Task 3 tests + full noise suite + golden re-check
  (all CPU paths must be bit-identical; goldens unchanged).
- [ ] **Step 5:** Commit: `fix(sensitivity): unequal-arm fused bases on GPU
  backends (explicit host/device fold boundary + device-aware cache key)`.

### Task 4: Cluster wiring — knobs on `all_sources` + modulation t0

**Files:**
- Modify: `src/lisatools/globalfit/stock/erebor/variants/all_sources.py`
  (fields + `finalize_general` + validation), `src/lisatools/globalfit/stock/
  erebor/noise.py` (`resolve_galfor_modulation` t0 threading) — exact seams
  re-read at execution
- Test: `tests/test_stock_globalfit.py` or a new `tests/test_unequal_arm_wiring.py`

**Interfaces:**
- Produces: `AllSourcesGeneralSettings.unequal_arm` (env `UNEQUAL_ARM`, default
  False), `.unequal_arm_stride` (env `UNEQUAL_ARM_STRIDE`, default 200),
  `.wdm_psd_method` (env `WDM_PSD_METHOD`, default `"fold"`); modulation loading
  that carries `t0 = data_t0`.

- [ ] **Step 1:** Read the build ordering: `finalize_general` vs `GeneralSetup`
  data processing; find where `general_info.data_t0` becomes authoritative, and
  where `noise_sensitivity_init_kwargs`' output is consumed. Decide between
  (a) deferring `LinkDelayTable`/modulation resolution to a post-processing hook
  or (b) porting `_domain_t0` (run_noise_only.py:143, via `MojitoL1File`, never
  `_read_xyz`) + a post-build assertion `computed_t0 == general_info.data_t0`.
  Do NOT blindly copy the 200 h trim assumption (all_sources mojito path
  disables the engine trim).
- [ ] **Step 2:** Failing tests first (construction-level, cheap):
  (a) `UNEQUAL_ARM=1` + `DATA_MODE=synthetic` → clear setup error;
  (b) `UNEQUAL_ARM=1` + mojito mode wires `instrument_component_cls =
  UnequalArmInstrumentNoise` with `ltts` from the L1 `/ltts` group, and errors
  loudly when the file lacks `/ltts`;
  (c) `GALFOR_MODULATION_PATH` on a mission-clock table loads with `t0` set
  (no `t0=0` guard trip);
  (d) knobs round-trip pickle/deepcopy pre-build.
  Mock the L1 file with a tiny local HDF5 fixture (create in test setUp; do not
  ship data).
- [ ] **Step 3:** Implement fields, validation, and threading; reproduce the
  `run_noise_only.py:490-505` recipe inside the variant (`LinkDelayTable.
  from_l1_file(noise_file, stride=<knob>, data_t0=<resolved>)`,
  `coarse_cache_dir` under the run's out dir, `wdm_psd_method=<knob>`); route
  the modulation through branch-level `fit.galfor.modulation =
  GalForTimeModulation(path, t0=<resolved>)` or a t0-threaded
  `resolve_galfor_modulation` — pick whichever the Step 1 ordering supports
  and note the choice in the commit message. Build must log: instrument class,
  wdm method, delay-table span+digest, modulation path+digest+coverage vs the
  data span, and `data_t0`.
- [ ] **Step 4:** Persist noise-model identity (unequal_arm flag, method, delay
  digest, modulation digest, data epoch) next to the stored domain settings;
  resume refuses a mismatch. Read how domain-settings resume checks work in the
  merged `run.py`/`hdfbackend.py` first (the `9d05cf5c` iteration-0 guard is the
  adjacent seam) and extend that mechanism rather than inventing a new one.
- [ ] **Step 5:** Run wiring tests + stock suite (same 1 known failure) + a
  synthetic `all_sources_lite` construction smoke with all knobs default
  (must be behaviorally inert when `UNEQUAL_ARM` unset).
- [ ] **Step 6:** Commit: `feat(stock): unequal-arm instrument + tabulated galfor
  modulation knobs on all_sources`.

### Task 5: `submit_gf_3mo_v8.sh`

**Files:**
- Create: `scripts/fstat_proposal/submit_gf_3mo_v8.sh` (from v7)

- [ ] **Step 1:** Read `submit_gf_3mo_v7.sh` in full; carry every v7 setting
  forward unchanged except: new `STORE_DIR`/`BASE_FILE_NAME`
  (`gf_prod_3mo_v8`), plus
  `UNEQUAL_ARM=1`, `UNEQUAL_ARM_STRIDE=200`, `WDM_PSD_METHOD=layer_calibrated`,
  `GALFOR_MODULATION_PATH=$PWD/scripts/noise/modulation_unequal.dat`.
- [ ] **Step 2:** Preflight block: echo every resolved noise knob; `test -f` the
  modulation file and the L1 noise file `/ltts` presence (python one-liner, no
  angle-bracket placeholders); refuse an existing store whose stored noise
  identity differs (reuse Task 4's metadata check); comment block explaining
  v8 = v7 + unequal-arm noise, referencing the handoff.
- [ ] **Step 3:** `bash -n` the script; local dry parse of the python preflight.
- [ ] **Step 4:** Commit: `feat(scripts): submit_gf_3mo_v8 — v7 + unequal-arm
  noise + galfor modulation`.

### Task 6: Bench, deferred-GPU checklist, readout

- [ ] **Step 1:** CPU noise-block micro-bench on the merged tree: the Task 1
  golden config timed pre-merge vs post-merge vs post-Task-3 (same seeds);
  confirm no CPU-path regression from the doubly-optimized `psdmove.py`
  (`PSD_BATCH=0` vs `1` at identical coordinates first — equality, then time).
- [ ] **Step 2:** Write the cluster/GPU validation checklist into the readout
  (not runnable locally): single-device method matrix (fold / layer_constant /
  layer_calibrated CPU-vs-CuPy), two-GPU cache smoke (per-device basis
  residency, peak mem, first-build time), stock preflight negative tests
  (misspelled knob must fail startup), noise-block re-bench (~43→9 s/it win
  must survive), whitening test on first v8 output, `layer_calibrated`
  validity warning clean at MAX_FREQ=2.5e-2.
- [ ] **Step 3:** Final full local suite pass; write the session readout
  (branch state, commits, what is deferred to cluster); do NOT push.

## Execution log (updated 2026-08-27 ~23:00, post-crash session)

- Task 1 DONE: goldens recorded (deterministic, seed 20260827). The laptop
  crashed/rebooted mid-Task-4 and wiped `/private/tmp` (scratchpad); helper
  scripts recreated under `<worktree>/.wtenv/`, goldens regenerate under
  `<worktree>/goldens_v8/`. Runs are now serialized through
  `.wtenv/wt_run.sh` (load-gated, nice, CPU-sampled) + a session load watchdog.
- Task 2 DONE: merge commit `2002e19e` (parents 86ed9353 + 9d05cf5c), stub fix
  `0cbfced6`. Overlap files byte-match reviewed blobs. Golden A across the
  merge: no-galfor scoring stack BIT-IDENTICAL; full-config drift <=1 ulp,
  isolated to (a) the documented FP-equal galfor quadrature cache (e1ee43af)
  and (b) the synthetic-injection fixture path (production mojito runs never
  touch it). Amended gate accordingly. Suites: green except the pre-existing
  lite-twin failure. Pickle gate passed (incl. live PSDMove).
- Task 3 CODE DONE (uncommitted until golden B confirms): gate removed,
  host/device fold boundary, `_folded_unit_columns_batched` (chunked),
  `current_device` in the `_bases` key (tag v2), 34/34 tests incl. new
  bitwise pinning + cupy-skipped GPU tests.
- Task 4 CODE DONE: `GeneralSetup._resolve_deferred_noise_model` (engine.py)
  resolves `ltts_l1_file`/`ltts_stride` -> LinkDelayTable at data_t0 and
  `galfor_modulation_anchor="data_t0"`; coverage checks; noise-model identity
  built there, persisted by run.py at iteration 0, resume refused on
  mismatch (hdfbackend `write/read_noise_model_identity`). all_sources knobs:
  `unequal_arm`/`UNEQUAL_ARM`, `unequal_arm_stride`, `wdm_psd_method`,
  `galfor_modulation_t0` ("data" -> anchor). `resolve_galfor_modulation`
  threads t0. 11/11 wiring tests (`tests/test_unequal_arm_wiring.py`).
- Task 5 SCRIPT WRITTEN: `scripts/fstat_proposal/submit_gf_3mo_v8.sh` = v7 +
  V8 header + noise knobs + [V8-NOISE] preflight (modulation file, /ltts,
  stored-identity match). `bash -n` clean. Uncommitted.
- ALL LOCAL VALIDATION CLOSED (2026-08-27 ~23:05):
  * Tasks 3/4/5 committed (`84c26f1c`, `0c3b52c9`, `245ed6c1`).
  * Golden B bit-identical across Task 3, Task 4, AND the dev fold-in merge
    `239e83ff` (dev `fca76736`+`9fa32109` = midit checkpointing + deferred
    cell labels; env-gated, numerically inert by default).
  * Golden A cross-tree reproduced the pre-crash numbers exactly (reboot-
    deterministic); no-galfor scoring stack zero differing bits.
  * Final targeted batch on `239e83ff`: 305 tests, 1 failure = the
    ORDER-DEPENDENT `test_lite_kwarg_matches_twin` artifact (passes
    standalone everywhere — combined-run state leakage, pre-existing at dev
    tip, NOT merge-related; earlier "user's uncommitted work fixes it"
    attribution was wrong), 1 skip = the cupy GPU test.
  * FULL local test sweep ABANDONED as laptop work: the machine has 8 GB RAM;
    the sweep python was jetsam-killed at ~40% under swap pressure (and the
    original three-job concurrency crashed/rebooted the box). The sweep is
    cluster/CI work; the noise-critical surface is fully covered by the
    targeted batches. Two `[fstat-NM]/[sighet-fstat] MULTIDEV CHECK FAILED`
    logger lines in the partial sweep are tests exercising that path
    (their tests passed), not failures.
  * Ephemeral comparison worktrees removed; goldens kept in `goldens_v8/`.
- Task 6 disposition: CPU noise-block bench folded into the golden timings
  (no regression signal); the quantitative psdmove re-bench (~43→9 s/it) is
  on the cluster checklist with the rest of the GPU validation.

## Self-review notes

- Spec coverage: handoff §0 preflight (done in interrogation), §2 merge (Task 2),
  §3 GPU blocker (Task 3), §3.3 cache key (Task 3), §4 wiring incl. the four
  "details that will bite" (Task 4: t0 lifecycle, plain-arrays rule, modulation
  t0, dat-file policy = repo path + preflight), §4.3 fresh store + resume
  identity (Tasks 4-5), §6 validation gates 1-5, 7, 10 local / 6, 8, 9 deferred
  to cluster checklist (Task 6), §7 landmines each mapped to a constraint above.
  §5 (coarse-beyond-noise) is explicitly Plan 2.
- The golden must be recorded with fixed params valid under the NEW galfor
  ceilings (fk, f_1, f_2 ≤ 1e-2) so pre/post-merge comparison is meaningful —
  encoded in Task 1 Step 2 and Task 2 Step 4.
- Type consistency: knob names follow rule 0 (field name ↔ env name).
