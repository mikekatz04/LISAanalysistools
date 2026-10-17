# DFFR integration: tests and results

This document records the unit tests, regression tests and validation runs of each step of the
DFFR integration; the steps are described in [`plan.md`](plan.md). Each entry states what was
compared, the backend and node, the commits, the tolerance, the result, and the command that
reproduces it.

Conventions:

- Outputs (`.npz`, `.json`, logs) are kept outside git. The commands write to `$OUT`, which for
  the author is `~/projects/erebor/dffr_integration` on SURF Spider. They run from the root of
  `lisa-analysis-tools`, with the Python of the environment in `$PYTHON`.
- A relative difference is normalised by the largest absolute value of the array compared.
  "Bit-identical" means `numpy.array_equal`.
- CPU runs use one OpenMP thread and one BLAS thread, so their outputs do not depend on the
  thread count. On nodes without an NVIDIA driver they also set `JAX_PLATFORMS=cpu`: there, the
  JAX CUDA plugin crashes in its CUDA version check as soon as `AnalysisContainerArray` asks JAX
  for its CPU device.
- CPU captures and timings are compared on the same node type only. NumPy and OpenBLAS select
  different code paths on AMD Bergamo (AVX-512) than on AMD Rome, which can change the fixture
  inputs in the last bit.

## Step 0: baseline of the existing response

Code: LAT `feat-dffr` at 7c12a36d and GBGPU `feat-dffr` at ed22789, both equal to `l2d-dev`;
editable builds of 2026-10-09 (`lisaanalysistools 1.2.8.post1.dev949+g7c12a36d0`,
`gbgpu 1.2.4.post1.dev136+ged2278993`); NumPy 2.5.3, SciPy 1.18.1. Besides `scripts/dffr/` and
this folder, the trees hold two uncommitted changes that the fixtures do not reach. In LAT, eight
lines in `gbspecialmove.py` set the periodic-lift attributes of `GBSpecialBase`, which Eryn's
`StretchMove` needs since Eryn commit 091cf42. In GBGPU, `src/gbgpu/testing.py` holds the reference
Fisher that `tests/test_stft_gb.py` imports; GBGPU commit 805d15b uses it but did not add it. The
golden outputs were captured without either change and recompute bit-identical with both
(`$OUT/regression/step0/recheck_after_wip_cpu.log`).

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41556045, 8 CPUs, 2026-10-17.

### Fixtures

`scripts/dffr/stft_fixtures.py` builds `STFTGBComputations` on fixed grids, data, inverse
covariances and sources. Data are seeded complex Gaussian arrays of scale 1e-20; inverse
covariances are seeded, of scale 1e40, Hermitian with cross-channel terms for XYZ and one value
per channel for the diagonal layout. Every fixture places carriers next to the edges of its
frequency window, so that the side-bin clamps run.

| Fixture | Orbits | Grid and settings | Domain | Covers |
| --- | --- | --- | --- | --- |
| `rect` | equal-arm | 16 six-hour segments, 61 active bins, `n_side_bins=4`, TDI 2nd generation, no window, 24 sources | one cell, shim | the unwindowed Fresnel path |
| `tukey_mid` | equal-arm | as `rect`, Tukey 0.3, midpoint anchoring | one cell, shim | the seven-term windowed path |
| `astro` | equal-arm | as `rect`, `freq_from_tdi_phase=False` | one cell, shim | the astrophysical frequency track |
| `esa_tdi2_envelope` | `ESAOrbits`, one 50 s grid, times from the file start | production: 32 one-day segments, Tukey 0.11574, midpoint, `linear_envelope=True`, `n_side_bins=10`, XYZ, TDI 2nd generation, 24 sources | real `STFTComputationGroup`, two cells of 64 bins at about 3 and 15 mHz | the production path on single-grid orbits |
| `esa_tdi1_const` | as above | TDI 1st generation, `linear_envelope=False` | as above | the comp's defaults |
| `esa_aet_diag` | as above | AET domain with a diagonal inverse covariance | as above | the diagonal branch of the likelihood kernels |
| `l1_tdi2_envelope` | `L1Orbits` from the Mojito light file `GB_731d_2.5s_L1_source0_0_20251205T020733787241Z.h5` (5 699 508 369 bytes), ICRS, `MOJITO_AVERAGE_ARMLENGTH`, `t_ref = MOJITO_REFERENCE_TIME`: positions resampled from the file's 500 000 s to 500 s, light travel times every 2.5 s from 850.5 s after `t_ref` | production, as `esa_tdi2_envelope` | as above | the production orbits: two grids at absolute times |

The first three fixtures repeat the configurations of `scripts/validation/stft_column_policy_oracle.py`
with the same seeds. The production fixtures follow `mojito_input/galfor_psd_multigpu_stft_settings.py`:
the band from 0.1 to 29 mHz, a Tukey taper of 1e4 s per one-day segment, and data starting 10
days after the first sample that both orbit grids hold. Their two cells are store windows built as
`Buffer._build_per_band_basis_settings` (`gbbands.py`) builds them: the band's lower edge, 64
active bins, and per-cell window starts of 250 and 1287 bins passed as `start_freq_inds`. Every
third production source carries `fddot = 11/3 fdot^2 / f0`. The AET domain contracts the X, Y and
Z values that the TDI configuration produces with a diagonal inverse covariance; the source
values are not converted to A, E and T.

For each fixture `scripts/dffr/stft_golden_outputs.py` stores every output of the seven kernels:
`get_ll_stft` (log-likelihood, `(d|h)`, `(h|h)`), `get_swap_ll_stft` (seven terms; each partner
three sources back in the same cell), `get_fstat_ll_stft` (eight sources, real and complex N and
M), `fill_global_stft` (two template rows, factors 1 and -0.5), `get_ll_grad_stft` and
`get_swap_ll_grad_stft` (envelope off, plus envelope on for the production fixtures), and
`information_matrix` on four sources (defaults, `easy_central_difference=True`,
`astro_track=False`, the subset f0, fdot, lambda, beta, plus envelope on for the production
fixtures).

### Regression tests

All commands run inside a Slurm allocation on a `wn-la` node, each as
`srun --jobid=<job> --overlap -n 1 -c 8 env PYTHON=$PYTHON scripts/dffr/run_baseline.sh cpu $OUT <part>`;
the driver sets the thread counts and `JAX_PLATFORMS`.

**Golden outputs on the CPU** (part `golden`). Every output of the seven kernels on the six
single-grid fixtures, 144 arrays, captured twice in separate processes. The two captures are
bit-identical, 144 of 144, so the CPU outputs are reproducible and later CPU comparisons can ask
for bit identity. Baseline: `$OUT/regression/step0/golden_cpu.npz`; the repeat and its comparison
are `golden_cpu_repeat.npz` and `golden_cpu_repeat_compare.txt`.

**Cross-check with the column-policy oracle** (part `oracle`). The capture of
`scripts/validation/stft_column_policy_oracle.py` holds 54 arrays for `rect`, `tukey_mid` and
`astro`; all 54 are bit-identical to the same arrays of `golden_cpu.npz`. The golden capture thus
reproduces the oracle's fixtures exactly, and the oracle remains usable as the gate of step 3.
Files: `column_oracle_cpu.npz`, `column_oracle_vs_golden_cpu.txt`.

**Golden outputs of `l1_tdi2_envelope` on the CPU** (part `golden_l1`, with `MOJITO_L1` set to
the file above). The 26 arrays of the production fixture on `L1Orbits`, captured twice in
separate processes, are bit-identical, 26 of 26. Baseline:
`$OUT/regression/step0/golden_cpu_l1.npz`; the comparison is in `golden_cpu_l1_repeat_compare.txt`.

**Pending.** The golden outputs on the GPU, all seven fixtures, captured twice, from the same base
build and before the first C++ change of step 1. Every GPU node of SURF Spider has been down since
2026-10-16 14:56. The two GPU captures also measure whether the GPU outputs are reproducible:
`fill_global_stft` adds into the template rows with `atomicAdd` (`lat_stft_kernels.hh:815`), so
sources that share pixels can be summed in a different order from run to run.

### Benchmarks

**CPU** (part `benchmarks`), fixture `esa_tdi2_envelope`: 32 one-day segments, two windows of 64
bins, `n_side_bins=10`. Each point is one warm-up call and five timed calls; the whole benchmark
ran three times in separate processes. The table gives the median over the three runs and the
spread of the three run medians, (largest - smallest) / median. Files:
`$OUT/benchmarks/step0/cpu_esa_run{1,2,3}.json`.

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Per source at 64 [ms] | Largest spread |
| --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.32 | 49.20 | 448.30 | 7.00 | 0.70 % |
| `get_swap_ll_stft` | 10.70 | 185.98 | 1398.43 | 21.85 | 0.46 % |
| `get_fstat_ll_stft` | 80.16 | 751.66 | 6978.13 | 109.03 | 0.46 % |
| `fill_global_stft` | 5.18 | 47.89 | 434.59 | 6.79 | 0.51 % |
| `get_ll_grad_stft` | 84.95 | 769.95 | 6927.72 | 108.25 | 0.63 % |
| `get_swap_ll_grad_stft` | 342.42 | 5648.33 | 42610.08 | 665.78 | 0.68 % |
| `information_matrix` | 42.42 | 362.43 | 3069.36 | 47.96 | 1.22 % |

The three runs agree to within 1.22 %, which sets the run-to-run noise for the timing gate of step
3 on this node type.

**CPU on `L1Orbits`** (part `benchmarks_l1`): fixture `l1_tdi2_envelope`, otherwise as above.
Files: `$OUT/benchmarks/step0/cpu_l1_run{1,2,3}.json`.

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Per source at 64 [ms] | Largest spread |
| --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.27 | 49.29 | 444.94 | 6.95 | 0.41 % |
| `get_swap_ll_stft` | 10.59 | 186.56 | 1390.90 | 21.73 | 0.45 % |
| `get_fstat_ll_stft` | 79.52 | 750.06 | 6906.57 | 107.92 | 0.61 % |
| `fill_global_stft` | 5.14 | 48.09 | 433.88 | 6.78 | 0.58 % |
| `get_ll_grad_stft` | 83.73 | 767.83 | 6898.88 | 107.80 | 0.45 % |
| `get_swap_ll_grad_stft` | 337.69 | 5636.89 | 42243.15 | 660.05 | 0.49 % |
| `information_matrix` | 42.12 | 360.06 | 3050.72 | 47.67 | 0.16 % |

Every point lies within 1.4 % of the same point on the ESA orbits (ratios 0.986 to 1.004). The two
grids of `L1Orbits`, with light travel times every 2.5 s, cost the existing response nothing
measurable on the CPU.

**Column-policy oracle** (`stft_column_policy_oracle.py --bench`, part `oracle`): 144 sources, 64
six-hour segments, `n_side_bins=4`, median of seven calls. `rect`: `get_ll` 863.87 ms, swap
1943.28 ms, fill 841.04 ms. `tukey_mid`: `get_ll` 1194.85 ms, swap 3299.53 ms, fill 1172.76 ms.
File: `$OUT/benchmarks/step0/column_oracle_bench_cpu.txt`.

**Pending:** the GPU benchmarks of both orbit fixtures, a year of one-day segments with batch sizes
1 to 10 000, on one A100 MIG slice and on a full A100.

### Test suites

Both suites ran on the CPU node through the driver parts `gbgpu_tests` and `lat_tests`, with the
two uncommitted changes above in place. Neither run left files in the repositories
(`git_status_after_*_cpu.txt`). Failures that already exist at the base commits are recorded
here, not fixed. The GPU runs are pending with the GPU baseline.

**GBGPU** (`pytest tests -rA --continue-on-collection-errors`, 119 s): 40 tests collected, 39
passed, 1 failed, 1 module not importable. Log and JUnit file:
`$OUT/regression/step0/gbgpu_tests_cpu.{log,xml}`.

| Outcome | Tests | Cause |
| --- | --- | --- |
| passed (39) | `test_stft_gb` (23, the Fisher kernel against `gbgpu.testing` included), `test_intra_split_index` (6), `test_stft_gb_accuracy` (4), `test_stft_gb_crossdomain` (6) | |
| failed (1) | `test_thirdbody::test_ecc_third_body` | needs a GPU: the test picks the GPU backend whenever CuPy imports, and the node has no driver |
| not importable (1) | `test_gbgpu.py` | needs a GPU: calls `cupy.cuda.runtime.setDevice(0)` at import |

**LAT** (`python -m unittest discover -v`, 261 s): 269 tests, 205 passed, 47 skipped, 2 failures,
15 errors. Log: `$OUT/regression/step0/lat_tests_cpu.log`.

| Outcome | Tests | Cause |
| --- | --- | --- |
| error (12) | `test_detector` (`test_get_vel`, `test_orbits`), the module `test_orbits`, `test_sensitivity` (`test_get_sen`, `test_get_sen_stft` and seven `test_sensitivity_matrix_*`) | need a GPU: they ask for a CUDA backend, or call CuPy at import |
| error (2), failure (2) | `test_gbspecial_flow` and `test_gbspecial_flow_stft` (`test_in_model_propose_two_passes`), `test_sighet_trust_region::test_trust_knob_defaults_and_env_wiring`, `test_gb_likelihood_engine::test_build_stft_band_aca_smoke` | built on `gbspecialstretch.py`, a parallel family of the GB moves that the global fit does not use; it uses `gbspecialmove.py`. The errors: that family's `GBSpecialBase` never sets the periodic-lift attributes that Eryn's `StretchMove` needs since Eryn 091cf42. The failures: out of date, since `recipe.py` no longer names `GB_SIGHET_TRUST_DLNA` and `domain_group_kwargs` now carries `linear_envelope` |
| error (1) | `test_gb_likelihood_engine::test_engine_calls_bind_to_STFTGBComputations_signatures` | out of date: its stand-in container lacks `split_map_by_split`. It checks the STFT band engine that step 5 extends |

**First run.** Before the two uncommitted changes were in place, GBGPU gave 16 passed, 1 failed and
2 modules not importable, `test_stft_gb.py` among them; without `--continue-on-collection-errors`,
pytest stops at the collection errors and runs nothing. LAT gave 204 passed, 2 failures and 16
errors; the extra error was `test_stft_store_windows::test_information_matrix_kernel_matches_reference`,
which imports `gbgpu.testing`. Logs: `$OUT/regression/step0/*_first_run.*`.

| Skipped | Reason |
| --- | --- |
| 33 | no CUDA backend or too few CUDA devices |
| 9 | developer checks, run only with `STFT_MULTI_GPU_CHECK=1` |
| 3 | a Mojito cache at a fixed path of another developer's machine |
| 1 | needs a GPU host with a fully initialised band buffer |
| 1 | the helper `gf_dev/ltt_averaging_diagnostic` is not in this checkout |

**Scripts after formatting.** After `black` (100 columns) and a variable rename in
`stft_fixtures.py`, `stft_golden_outputs.py --compare golden_cpu.npz --backend cpu` recomputed all
144 arrays bit-identical to the baseline (`$OUT/regression/step0/rerun_cpu.log`).

### Unit tests

None: step 0 changes no code.

### Validation runs

**Standalone dffr against its time-domain truth on the ESA orbits.** This run sets the accuracy
that `README.md` quotes: the standalone package, before any integration, against a time-domain
TDI 2.0 truth over 30 one-day segments on the ESA trailing orbits that lisaanalysistools ships.

- Code: direct-fast-fresnel-response at 01530a5, compiled CPU path, in its own environment
  (`env -u VIRTUAL_ENV uv sync --frozen --extra compare`): lisaanalysistools 1.2.8, lisaorbits
  3.1.0, lisagwresponse 3.0.2, lisainstrument 2.3.0, pytdi 2.2.1.
- Truth: lisagwresponse on the strain at the exact retarded times, lisainstrument with every noise
  and clock deviation off in the TCB frame, and pytdi's X2, Y2 and Z2, sampled every 5 s, followed
  by the same windowed segment transform. Grid: 30 one-day segments from 5 days after the start of
  the orbit file, Tukey window 0.116, every active bin of a band of +-3e-4 Hz around the carrier.
- Metrics: the mismatch takes all channels, segments and bins together. The segment error is the
  largest absolute difference in one segment and channel, divided by the peak of that segment and
  channel; the table gives its median and largest value over the 30 segments and 3 channels.
- Node: `wn-la-01`, the eight cases side by side, one CPU each: about 2 to 5 minutes and 6.1 GB of
  memory per case.
- Command, inside an allocation (`srun --jobid=<job> --overlap -n 1 -c 8 ...`):
  `scripts/dffr/run_dffr_truth.sh <dffr repository> $OUT/validation/step0_dffr_truth_esa_30d 30 esa`.
  Files: a JSON file and a log per case, and `summary.json`. A first run with the same arguments
  (`step0_dffr_truth_esa_30d_first_run/`) gave identical results in all eight cases.

| Source | Sky | Mismatch | Median segment error | Largest segment error |
| --- | --- | --- | --- | --- |
| 3 mHz | 0 | 1.23e-9 | 4.65e-5 | 1.41e-4 |
| 3 mHz | 1 | 6.47e-10 | 3.17e-5 | 9.89e-5 |
| 20 mHz | 0 | 1.24e-9 | 4.06e-5 | 4.79e-4 |
| 20 mHz | 1 | 7.49e-10 | 3.23e-5 | 5.63e-5 |
| 20 mHz, fdot = 1e-13 Hz/s | 0 | 1.24e-9 | 4.05e-5 | 4.75e-4 |
| 20 mHz, fdot = 1e-13 Hz/s | 1 | 7.49e-10 | 3.25e-5 | 5.66e-5 |
| 40 mHz | 0 | 9.92e-10 | 3.68e-5 | 8.94e-5 |
| 40 mHz | 1 | 1.08e-9 | 4.53e-5 | 7.41e-5 |

Over all eight cases the mismatch lies between 6.5e-10 and 1.2e-9 and the median segment error
between 3.2e-5 and 4.7e-5. The largest segment errors, up to 4.8e-4, occur at 20 mHz in sky 0. Step
6 compares both responses of the integration against this truth.

## Step 1: orbits cache

Not started.

### Unit tests

### Regression tests

### Validation runs

## Step 2: Fresnel split

Not started.

### Unit tests

### Regression tests

### Validation runs

## Step 3: tile seam

Not started.

### Unit tests

### Regression tests

### Validation runs

## Step 4: DFFR kernels

Not started.

### Unit tests

### Regression tests

### Validation runs

## Step 5: wiring

Not started.

### Unit tests

### Regression tests

### Validation runs

## Step 6: validation

Not started.

### Unit tests

### Regression tests

### Validation runs

## Step 7: review of documentation and scripts

Not started.

### Unit tests

### Regression tests

### Validation runs
