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

**Pending.** The golden outputs on the GPU, all seven fixtures, captured twice, from the step 0
code. Every GPU node of SURF Spider has been down since 2026-10-16 14:56, and step 1 rebuilt the
shared environment, so these captures come from the frozen environment of the step 0 code that
`scripts/dffr/build_base_env.sh` builds. That environment reproduces both CPU baselines above bit
for bit (step 1, "Frozen environment of the step 0 code"). The two GPU captures also measure
whether the GPU outputs are reproducible: `fill_global_stft` adds into the template rows with
`atomicAdd` (`lat_stft_kernels.hh:815`), so sources that share pixels can be summed in a
different order from run to run. Command, on a GPU node, with `BASE=$OUT/base_env`:
`PYTHON=$BASE/.venv/bin/python LAT_ROOT=$BASE/lisa-analysis-tools MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_baseline.sh --device gpu $OUT environment golden golden_l1 benchmarks benchmarks_l1 tests`.

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

Code: LAT `feat-dffr` at 71b3f351 with the step 1 changes uncommitted (`Detector.{hpp,cu}`,
`binding_detector.{hpp,cxx}`, `lisatools_header_abi.hpp`, `lat_stft_kernels.hh`,
`cutils/__init__.py`, `domaincomputation.py`, `tests/test_orbit_kinematics.py`, `scripts/dffr/`);
GBGPU `feat-dffr` at a156dc3 and BBHx `l2d-dev` at b7b0ae5, each with one added line in
`cutils/__init__.py` (BBHx also with its earlier uncommitted patch). Rebuilt on 2026-10-17 into
the shared environment with `ONLY="lisa-analysis-tools GBGPU" ./install_erebor_spider.sh`, CPU
and CUDA backends.

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41559105, 8 CPUs, 2026-10-17. The
regressions run inside that allocation as
`srun --jobid=<job> --overlap -n 1 -c 1 env PYTHON=$PYTHON BASE_ENV=$OUT/base_env DFFR_REPOSITORY=<dffr clone> MOJITO_L1=<file> scripts/dffr/run_step1.sh --device cpu $OUT <part>`;
the driver sets the thread counts and `JAX_PLATFORMS` as in step 0. The GPU runs are pending;
they repeat the same parts with `--device gpu`.

How to read the numbers. A relative value is the largest absolute difference over an array
divided by the largest absolute value of the reference array, as everywhere in this document;
an absolute value carries the units of the quantity. A bound is the threshold a test asserts:
the test passes when the measured value stays below it. Where no threshold exists, the text
says what the value is compared with.

### Frozen environment of the step 0 code

The GPU baseline of step 0 must come from the code before step 1, but every GPU node was down
and step 1 rebuilds the shared environment. `scripts/dffr/build_base_env.sh` therefore builds a
second environment, `$OUT/base_env`, from frozen clones: LAT at 71b3f351 and GBGPU at a156dc3
(the step 0 code plus the step 0 documents and scripts, the gbspecialmove fix and
`gbgpu/testing.py`, none of which the fixtures reach), and
GPUBackendTools b07bd18, Eryn 26dd14a and BBHx b7b0ae5 with the uncommitted changes of the shared
clones. The third-party packages get the exact versions of the shared environment. BBHx is
included because `lisatools.globalfit.recipe`, which holds the Mojito constants of the
`L1Orbits` fixture, imports it.

Criterion: the environment must recompute the step 0 CPU golden outputs bit for bit. Result:
144 of 144 arrays of `golden_cpu.npz` and 26 of 26 of `golden_cpu_l1.npz` are bit-identical,
pass (`$OUT/regression/step0/base_env_recheck_cpu{,_l1}.log`). Commands:
`LAT_COMMIT=71b3f351 GBGPU_COMMIT=a156dc3 JAX_PLATFORMS=cpu scripts/dffr/build_base_env.sh $OUT/base_env`,
then, with `BASE=$OUT/base_env`,
`$BASE/.venv/bin/python $BASE/lisa-analysis-tools/scripts/dffr/stft_golden_outputs.py --compare $OUT/regression/step0/golden_cpu.npz --backend cpu --configs rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag`
(and the same with `--configs l1_tdi2_envelope --mojito-l1 <file>` against `golden_cpu_l1.npz`).

### Unit tests

`tests/test_orbit_kinematics.py`, run as `python -m unittest tests.test_orbit_kinematics -v`
(part `unit`): 13 tests, 7 pass on the CPU and the 6 CUDA cases are skipped without a GPU
(`unit_tests_cpu.log`; last run on 2026-10-18, Slurm job 41562942 on `wn-la-01`, after the
budgets of the circle were tightened to the values below). `scripts/dffr/kinematics_unit_numbers.py --device cpu` prints each
measured value next to the threshold its test asserts (`unit_numbers_cpu.txt`).

The synthetic orbits put positions every 500 s and light travel times every 2.5 s, starting
850.5 s later, at absolute times near 2e9 s, as in `L1Orbits`; the table covers ten one-day
segments. The rounding bound of a field is 8 roundings of the largest table entry, multiplied by
the sum of the stencil coefficients over 12 h (rates) or 12 h^2 (accelerations); 8 roundings
cover the arithmetic of each value and the fused multiply-adds either side may use.

| Test | Quantity | Threshold | Measured | Result |
| --- | --- | --- | --- | --- |
| Quadratic positions and light travel times | absolute error of the four derivative fields against the exact derivatives | rounding bound: velocity 2.4e-7 m/s, acceleration 4.3e-10 m/s^2, light-travel-time rate 1.1e-17, its acceleration 2.0e-20 1/s | 1.4e-8 m/s, 3.1e-11 m/s^2, 8.5e-19, 1.3e-21 1/s (6 % to 8 % of the bound) | pass |
| Annual circle | relative error of every field against its analytic value | position 2e-9, velocity 1e-8, acceleration 5e-8, light travel time 1e-15, its rate 1e-9, its acceleration 2e-5, link vector and its rate 5e-8 | position 1.2e-9, velocity 1.2e-9, acceleration 4.7e-9, light travel time 2.1e-16, rate 1.2e-10, acceleration 1.1e-6, link vector and rate 4.9e-9 | pass |
| NumPy mirror | absolute difference between the table and a NumPy copy of the interpolation, stencils and link vectors | rounding bound of each field | 6 of 11 fields bit-identical; the others at most 3 % of their bound (accelerations differ by 1.0e-11 m/s^2) | pass |
| ESA orbits | relative difference between the receiver velocities and the file velocities (`Orbits.get_vel`), 30 one-day segments from day 10 | 2e-6 | 2.6e-7 to 2.8e-7 | pass |
| Coverage | the earliest and latest grids that fit build; grids 1 s earlier or later raise `ValueError` naming the light-travel-time or the position grid | | as expected | pass |
| Arguments | wrong length raises `ValueError`; float32 or strided arrays raise `TypeError` | | as expected | pass |
| Group cache | `cpp_kinematics` builds once per orbits object, another object gets its own table, and the fields equal a direct build bit for bit | | as expected | pass |
| CUDA against CPU | absolute difference of every field | rounding bound | pending | pending |

The thresholds of the circle come from the errors of the method, with margins from 1.7
(positions, whose error is the deterministic interpolation sag) to about 20 over the measured
values. Linear interpolation of node values on the 500 s grid contributes (dt^2 / 8) Omega^2 = 1.2e-9
(Omega the annual angular frequency); rounding of positions of 1 AU over h^2 contributes about
7e-9 of the acceleration per rounding, and rounding of 8.3 s over h^2 about 3e-6 of the
light-travel-time acceleration. The truncation error of the five-point stencil, (Omega h)^4 / 90
= 3e-16, plays no part. The mirror is not compared bit for bit because the CPU backend is built
for x86-64-v3, where GCC fuses multiply-adds that NumPy rounds separately; the six fields that do
match bit for bit show that the two follow the same arithmetic. The ESA threshold is the estimate
(Omega Delta)^3 / 24 = 1.4e-6 for the derivative of a cubic spline through samples Delta = 1.87
days apart, which is how the base orbit classes interpolate the file.

### Regression tests

**Golden outputs against step 0** (parts `golden` and `golden_l1`). Two captures in separate
processes are bit-identical, 144 of 144 and 26 of 26: the step 1 outputs are reproducible
(criterion: bit identity; pass). Against step 0 the outputs are expected to change, because the
TDI-phase frequency estimator now reads a five-point acceleration instead of a three-point one.
The comparison files (`golden_cpu_vs_step0.txt`, `golden_cpu_l1_vs_step0.txt`) mark every array
that is not bit-identical as `FAIL`, since the tool asks for bit identity; here that label only
means "changed". The criteria were set before measuring: arrays that never read the acceleration
stay bit-identical, and the others change by about 1e-11, the size of the removed truncation
error of the Doppler chirp rate (1.3e-8) carried into the phase of a segment, with finite
differences enlarging it.

Largest relative change per fixture and kind of output (no threshold; compared with the
prediction):

| Fixture | Unchanged arrays | Inner products, F-statistic, templates | Gradients | Fisher, TDI track | Fisher, other estimators |
| --- | --- | --- | --- | --- | --- |
| `astro` | 22 of 22 | 0 | 0 | 0 | 0 |
| `rect` | 3 of 22 | 4.7e-11 | 1.8e-9 | 1.7e-8 | 0 |
| `tukey_mid` | 3 of 22 | 1.4e-14 | 2.2e-9 | 3.2e-9 | 0 |
| `esa_tdi2_envelope` | 4 of 26 | 4.4e-11 | 3.3e-10 | 4.5e-11 | 0 |
| `esa_tdi1_const` | 4 of 26 | 4.4e-11 | 6.7e-10 | 5.8e-11 | 0 |
| `esa_aet_diag` | 4 of 26 | 3.3e-11 | 1.9e-10 | 1.7e-11 | 0 |
| `l1_tdi2_envelope` | 4 of 26 | 5.9e-11 | 3.0e-10 | 2.3e-10 | 0 |

Both criteria hold. `astro` uses the astrophysical frequency track and never reads the
acceleration; so do the plain Fisher estimators, which are the unchanged arrays of every other
fixture. Inner products and templates change by at most 6e-11. Gradients and Fisher matrices are
finite differences of templates and change more, up to 2e-9 and 2e-8.

Within the Fisher matrix on the TDI track, the change of each entry divided by its own scale,
the square root of F_ii F_jj, is largest in the row and column of f̈ (fddot): up to 1.9e-5 on the
four-day equal-arm fixtures (`rect`, `tukey_mid`) and up to 7.9e-7 on the 32-day fixtures
(`l1_tdi2_envelope`; at most 2.9e-7 on the ESA orbits). All other entries change by at most
1.5e-8 (four days) and 3.6e-9 (32 days) of their scale. The f̈ derivative is a finite difference
of templates that f̈ barely changes over four days (its phase grows as T^3), so changes at the
level of rounding in the templates are enlarged, more on the short fixtures than on the long
ones. On the equal-arm fixtures the f0 and ḟ entries change by up to 1.5e-8, the size of the
removed truncation error. Each number is the largest over the four sources of an array:
`scripts/dffr/fisher_entry_changes.py $OUT/regression/step0/golden_cpu.npz
$OUT/regression/step1/golden_cpu.npz`, and the same for `golden_cpu_l1.npz` (files
`golden_cpu{,_l1}_fisher_vs_step0.txt`). Corrected on 2026-10-18: the first version of this
paragraph gave the values of the first source only (8e-6, 1e-7 and 4e-10) and left out the
`L1Orbits` fixture. Once the author has reviewed these changes, the
step 1 captures `$OUT/regression/step1/golden_cpu.npz` and `golden_cpu_l1.npz` become the new
baseline.

**Accuracy before and after** (part `accuracy`, CPU only, since the test runs on the CPU
backend). `GBGPU/tests/test_stft_gb_accuracy.py` prints mismatches, one minus the normalised
overlap with a brute-force STFT of a time-domain injection. In the frozen step 0 environment and
after step 1, every printed value is identical to its four digits: end-to-end mismatch 9.049e-3
at 10 side bins (3.950e-3 at 20), on-bin mismatch 1.409e-5. The tests pass in both
environments (their own thresholds, for example an on-bin mismatch below 1e-3). A change of
1e-11 cannot show at this level, which is the point of the check: no loss of accuracy.
Files: `accuracy_base_cpu.log`, `accuracy_cpu.log`.

**Test suites** (part `tests`). Criterion: the same outcomes as step 0. GBGPU: 39 passed,
1 failed and 1 module not importable, the same two GPU-only cases as in step 0. LAT: 282 tests,
the 269 of step 0 plus the 13 new ones, with the same 2 failures and 15 errors as in step 0 and
53 skipped (47 plus the 6 new CUDA cases). Comparing the outcome of every test name between the
two logs shows only the new tests. Pass. Files: `gbgpu_tests_cpu.log`, `lat_tests_cpu.log`.

**Exchange with standalone dffr on the ESA orbits** (part `dffr`). Both sides build the table
at the same 365 one-day segments, from 5 days after the start of the orbit file; the LAT side
counts times from that start and the dffr side adds the file's t0 to whole seconds, which is
exact. dffr (01530a5) runs in its own environment (`scripts/dffr/kinematics_dffr.py`) with two
builders: its C builder with the stencil of LAT (five points, 2000 s) on the same 50 s
lisatools table, and its lisaorbits builder with analytic derivatives of a quintic spline.
Files: `kinematics_lat_esa_cpu.npz`, `kinematics_dffr_esa.npz`, `kinematics_esa_compare_cpu.txt`.

| Field | C builder, absolute | C builder, relative | lisaorbits, absolute | lisaorbits, relative |
| --- | --- | --- | --- | --- |
| receiver position | 0 | 0 | 486 m | 3.2e-9 |
| receiver velocity | 0 | 0 | 8.3e-3 m/s | 2.8e-7 |
| receiver acceleration | 0 | 0 | 5.0e-7 m/s^2 | 8.4e-5 |
| sender position | 3.5e-3 m | 2.4e-14 | 486 m | 3.2e-9 |
| sender velocity | 1.9e-8 m/s | 6.4e-13 | 8.3e-3 m/s | 2.8e-7 |
| sender acceleration | 3.4e-11 m/s^2 | 5.6e-9 | 5.0e-7 m/s^2 | 8.4e-5 |
| light travel time | 0 | 0 | 2.0e-9 s | 2.4e-10 |
| its rate | 0 | 0 | 3.9e-14 | 1.2e-6 |
| its acceleration | 0 | 0 | 2.1e-18 1/s | 2.2e-4 |
| link vector | 1.4e-12 | 1.4e-12 | 1.4e-8 | 1.4e-8 |
| its rate | 7.6e-18 1/s | 4.3e-11 | 2.2e-13 1/s | 1.2e-6 |

No threshold is asserted; each column is compared with what it should show. The C builder checks
the port: receivers and light travel times are bit-identical. The senders differ because dffr
subtracts the file's t0 = 2.07e9 s after the light travel time, which rounds the emission time
to 2.4e-7 s; 30 km/s times half of that is the 3.5 mm seen, and the link vectors follow from it.
The lisaorbits builder checks the physics: the differences agree with the error of a cubic
spline through file samples 1.87 days apart, estimated as (5/384)(Omega Delta)^4 = 1.4e-8 for
positions, (Omega Delta)^3 / 24 = 1.4e-6 for velocities and (Omega Delta)^2 / 12 = 8.6e-5 for
accelerations; the measured values lie at or below these. They come from how LAT represents the
orbit, not from the stencil, whose own errors in the unit tests are 1.2e-9 for velocities and
4.7e-9 for accelerations, 200 to 20 000 times smaller. At 20 mHz they shift the phase by about
2e-7 rad, against a median DFFR error of 3e-5 per segment.

**The Mojito orbits** (part `l1`). A table on `L1Orbits` over 365 one-day segments, from 10 days
after the first sample both grids hold: the coverage check passes. The receiver positions equal
`Orbits.get_pos` exactly (absolute difference 0, as both read the same interpolation). The
receiver velocities differ from the file velocities by 8.1e-6 relative; no threshold, compared
with the estimate (Omega Delta)^3 / 24 = 4e-5 for a cubic spline through position samples
5.8 days apart. Files: `kinematics_l1_cpu.{npz,txt}`.

**Scripts after formatting.** After `black` and `isort` (100 columns),
`stft_golden_outputs.py --compare $OUT/regression/step1/golden_cpu.npz --device cpu` recomputed
all 144 arrays of the six single-grid fixtures bit-identical (`rerun_after_format_cpu.log`), and
the unit tests passed again.

**The `--without-swap` option** (added on 2026-10-18 for the CPU runs from step 2 on). Against
`$OUT/regression/step1/golden_cpu.npz`, with the option on `rect` and `esa_tdi2_envelope`: 28 of
28 remaining arrays bit-identical, and the left-out swap arrays do not count as failures
(`without_swap_flag_check_cpu.log`); without the option on `rect`: 22 of 22 arrays, swap
included, bit-identical (`with_swap_flag_check_cpu.log`). `stft_benchmarks.py --without-swap`
times the five other methods. Pass.

### Benchmarks

**CPU** (parts `benchmarks` and `benchmarks_l1`), run alone in the allocation after the other
parts, with the fixtures and settings of step 0: 32 one-day segments, two windows of 64 bins,
`n_side_bins=10`, one warm-up call and five timed calls per point, three runs in separate
processes. The five-point acceleration reads 15 positions per column instead of 9, so a small
cost was expected; no threshold applies in this step (the timing gate of the plan is in step 3).
Compared with step 0 by `scripts/dffr/compare_benchmarks.py --reference step0/cpu_esa_run*.json
--candidate step1/cpu_esa_run*.json`: the ratio is the median of the three run medians of step 1
over that of step 0, and the spread is (largest - smallest) / median of the three run medians.
Files: `$OUT/benchmarks/step1/cpu_{esa,l1}_run{1,2,3}.json`, `compare_cpu_{esa,l1}.txt`.

ESA fixture (`esa_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 0, three batches | Largest spread of the step 1 runs |
| --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.36 | 49.13 | 447.25 | 0.998 to 1.006 | 1.69 % |
| `get_swap_ll_stft` | 10.72 | 185.92 | 1400.05 | 1.000 to 1.002 | 0.98 % |
| `get_fstat_ll_stft` | 80.63 | 755.68 | 6966.48 | 0.998 to 1.006 | 2.94 % |
| `fill_global_stft` | 5.21 | 48.00 | 436.24 | 1.002 to 1.006 | 2.55 % |
| `get_ll_grad_stft` | 85.30 | 774.08 | 6963.64 | 1.004 to 1.005 | 1.24 % |
| `get_swap_ll_grad_stft` | 342.99 | 5624.63 | 42595.00 | 0.996 to 1.002 | 0.68 % |
| `information_matrix` | 42.50 | 362.01 | 3071.94 | 0.999 to 1.002 | 1.45 % |

Over all 21 points the ratio lies between 0.996 and 1.006, median 1.002, within the run-to-run
spread of either step (up to 1.22 % in step 0 and 2.94 % in step 1).

`L1Orbits` fixture (`l1_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 0, three batches | Largest spread of the step 1 runs |
| --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.32 | 49.36 | 447.15 | 1.001 to 1.010 | 1.74 % |
| `get_swap_ll_stft` | 10.63 | 188.31 | 1403.10 | 1.004 to 1.009 | 0.95 % |
| `get_fstat_ll_stft` | 80.23 | 754.95 | 6974.53 | 1.007 to 1.010 | 0.98 % |
| `fill_global_stft` | 5.18 | 48.05 | 435.54 | 0.999 to 1.009 | 0.51 % |
| `get_ll_grad_stft` | 84.12 | 771.66 | 6959.00 | 1.005 to 1.009 | 1.15 % |
| `get_swap_ll_grad_stft` | 340.99 | 5680.39 | 42559.82 | 1.007 to 1.010 | 0.42 % |
| `information_matrix` | 42.37 | 362.01 | 3067.98 | 1.005 to 1.006 | 0.50 % |

Here the ratio lies between 0.999 and 1.010, median 1.007, and 20 of the 21 points are slower
than in step 0, while the run-to-run spread is at most 0.61 % in step 0 and 1.74 % in step 1. A
cost of up to about 1 % is therefore likely real on this fixture; on the ESA fixture it stays
within the noise. Both fit the extra work: six more position lookups per column, next to the
three TDI evaluations of the column. The step 0 and step 1 runs of a fixture were made hours
apart on a shared node, which limits how finely such small ratios can be read.

### Validation runs

None in this step.

### Pending on the GPU

The shared environment holds the step 2 code since 2026-10-18, so these runs come from the frozen
environment of the step 1 code, `$OUT/step1_env`, which reproduces the step 1 CPU golden outputs
bit for bit (step 2 records how it was built and checked). On a GPU node, with `S1=$OUT/step1_env`:
`PYTHON=$S1/.venv/bin/python BASE_ENV=$OUT/base_env DFFR_REPOSITORY=<dffr clone> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" $S1/lisa-analysis-tools/scripts/dffr/run_step1.sh --device gpu $OUT environment unit golden golden_l1 dffr l1 tests benchmarks benchmarks_l1`,
after the GPU baseline of step 0, so that `golden_gpu*.npz` of step 0 exist for the comparison.

## Step 2: Fresnel split

Code: LAT `feat-dffr` at 2863df4e with the step 2 changes uncommitted (`cutils/lat_fresnel.hh`,
`domains.{hpp,cu}`, `binding_domains.hpp`, `binding_detector.cxx`, `tests/test_fresnel.py`,
`scripts/dffr/`); GBGPU `feat-dffr` at 62c8d4c, unchanged. GBGPU compiles LAT's `domains.cu`, so
its kernels use the new functions. Rebuilt on 2026-10-18 into the shared environment with
`ONLY="lisa-analysis-tools GBGPU" ./install_erebor_spider.sh`, CPU and CUDA backends.

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41562944, 16 CPUs, 2026-10-18. The
regressions ran inside that allocation as
`srun --jobid=<job> --overlap -n 1 -c 16 env PYTHON=$PYTHON BASE_ENV=$OUT/step1_env MOJITO_L1=<file> scripts/dffr/run_step2.sh --device cpu $OUT <part>`:
the parts `environment unit golden accuracy`, `golden_l1` and `tests` side by side, then
`benchmarks benchmarks_l1` alone, so that nothing else ran during the timings
(`$OUT/regression/step2/step2_cpu_chain.sh`); `fd_steps fd_steps_l1` ran afterwards. The GPU
runs are pending.

Numbers are read as in step 1. A value relative to an array's peak is the largest absolute
difference over the array divided by the largest absolute value of the reference array. A bound
is the threshold a test asserts. From this step on, the CPU captures and benchmarks leave out the
two swap kernels (`--without-swap`): they are built from the same column evaluation as
`get_ll_stft` and `get_ll_grad_stft`, `GBGPU/tests/test_stft_gb.py` keeps checking their own
parts on the CPU, and they took about 70 % of a CPU benchmark run. The GPU runs keep them.

### Frozen environment of the step 1 code

The GPU runs of step 1, and the step 1 GPU baseline that step 2 is compared with, must come from
the step 1 code, but this step rebuilds the shared environment and every GPU node is still down.
`scripts/dffr/build_base_env.sh` therefore built `$OUT/step1_env` from LAT 2863df4e and GBGPU
62c8d4c, the committed step 1, with GPUBackendTools b07bd18, Eryn 26dd14a and BBHx b7b0ae5 plus
the uncommitted changes of the shared clones (for BBHx, its earlier patch and the line that passes
`KinematicsTableWrap`), and the third-party versions of the shared environment.

Criterion: the environment must recompute the step 1 CPU golden outputs bit for bit. Result: 144
of 144 arrays of `golden_cpu.npz`, the swap kernels included, and 26 of 26 of `golden_cpu_l1.npz`
are bit-identical, pass (`$OUT/regression/step1/step1_env_recheck_cpu{,_l1}.log`). Commands:
`LAT_COMMIT=2863df4e GBGPU_COMMIT=62c8d4c JAX_PLATFORMS=cpu scripts/dffr/build_base_env.sh $OUT/step1_env`,
then, with `S1=$OUT/step1_env`,
`$S1/.venv/bin/python $S1/lisa-analysis-tools/scripts/dffr/stft_golden_outputs.py --compare $OUT/regression/step1/golden_cpu.npz --device cpu --configs rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag`
(and the same with `--configs l1_tdi2_envelope --mojito-l1 <file>` against `golden_cpu_l1.npz`).

### Unit tests

`tests/test_fresnel.py`, run as `python -m unittest tests.test_fresnel -v` (part `unit`): 9
tests, 4 pass on the CPU and the 5 CUDA cases are skipped without a GPU (`unit_tests_cpu.log`).
`scripts/dffr/fresnel_unit_numbers.py --device cpu` prints each measured value next to the bound
its test asserts (`unit_numbers_cpu.txt`).

The tests reach the functions through two diagnostic methods of `STFTFresnelWrap`:
`compute_fresnel_functions` (C, S, f, g and the derivatives of f and g, through the two
`STFTFresnel` members that delegate to the header) and `compute_phase_kernel_moments` (the
single-interval kernel and its moment, which call the header directly). The references are
`scipy.special.fresnel` for C and S; quadrature of the integral representations of f and g
(DLMF 7.7) for f, g and their derivatives, chosen because their integrands are smooth and
positive; and adaptive quadrature (`scipy.integrate.quad`, 1e-13 requested, own error estimates
at most 1.1e-14 of the normalisations) for the phase kernel. The errors of the rational fits and
the truncation of the asymptotic series are deterministic, so their bounds sit 1.15 to 2.1 times
above the measured values; the bounds on rounding sit 2 to 3 times above.

| Test | Quantity | Bound | Measured | Result |
| --- | --- | --- | --- | --- |
| C and S against scipy, 24 806 arguments with \|x\| <= 1000 | absolute error, per branch | series 1e-15; rational fits 2e-10 (C) and 1e-10 (S); asymptotic series 2e-13 | series 4.4e-16 and 3.3e-16; rational fits 1.6e-10 and 7.0e-11, at the seam; asymptotic series 1.0e-13 and 8.5e-14, at \|x\| = 1000, from the rounding of the phase pi x^2 / 2 | pass |
| Odd symmetry | C(-x) + C(x) and S(-x) + S(x) | 0 | 0 (bit for bit) | pass |
| f, g and their derivatives against quadrature, 882 arguments from the first double above 1.6 to 1e8 | relative error; for g' on the rational branch absolute, because g' falls to -7.5e-5 near \|x\| = 8 | rational fits: f 1e-9, g and f' 2.5e-9, g' 1e-9; asymptotic series: f 1e-14, g and f' 2.5e-13, g' 2e-12 | rational fits: 8.6e-10, 2.2e-9, 2.2e-9 and 8.4e-10, all at the seam; asymptotic series: 5.8e-15, 1.3e-13, 1.3e-13 and 9.6e-13, at \|x\| = 8 | pass |
| Phase kernel and moment, the 16 cases of dffr's `tests/test_fresnel.py` (chirp rates +-4e-13, 1e-20 and -1e-22 Hz/s; 0, 0.3, 7 and -18.5 bins from the carrier; one day) | error of the integral as a fraction of the segment length, and of its moment as a fraction of the half-length squared | 2e-10 and 5e-8, dffr's bounds | 7.2e-11 and 2.9e-8, both 0.3 bins from the carrier at \|alpha\| = 4e-13, where an endpoint falls next to the seam; at most 3.9e-14 and 9.5e-12 in the other 14 cases | pass |
| CUDA against CPU | absolute difference for C, S and g', relative for f, g and f'; the phase kernel as a fraction of dffr's normalisation | 8 roundings; 1e-13, provisional | pending | pending |

### Regression tests

**Golden outputs against step 1** (parts `golden` and `golden_l1`, without the swap kernels).
Two captures in separate processes are bit-identical, 84 of 84 and 15 of 15 arrays: the step 2
outputs are reproducible (criterion: bit identity; pass). The changes against step 1 are expected
at the level of rounding, since the series and the rational fits are the same and only the order
of the arithmetic changed. The plan set the gate at 1e-12 of each array's peak
(`golden_cpu{,_l1}_vs_step1.txt`).

Largest change relative to the array's peak, per fixture and kind of output (gate 1e-12):

| Fixture | Log-likelihood, (d\|h), (h\|h), templates | F-statistic | Gradients | Fisher matrices |
| --- | --- | --- | --- | --- |
| `rect` | 5.6e-15 | 1.0e-14 | 6.2e-10 | 9.6e-9 |
| `tukey_mid` | 4.7e-15 | 1.6e-14 | 1.1e-9 | 3.5e-9 |
| `astro` | 5.5e-15 | 3.8e-15 | 1.8e-9 | 1.8e-9 |
| `esa_tdi2_envelope` | 1.1e-14 | 7.0e-16 | 1.7e-10 | 4.9e-10 |
| `esa_tdi1_const` | 8.2e-15 | 6.1e-16 | 5.6e-11 | 4.7e-10 |
| `esa_aet_diag` | 5.4e-15 | 1.8e-15 | 2.3e-11 | 1.6e-10 |
| `l1_tdi2_envelope` | 5.5e-15 | 6.2e-15 | 6.0e-11 | 3.9e-10 |

Every output computed directly from templates passes, by a factor of 90 or more. The 36 gradient
and Fisher arrays do not. These are central differences of templates, which divide any change in
rounding by their steps. Measured against the peak of a whole array, the numbers also hide how
large the changes are per parameter, because the arrays mix parameters whose scales differ by
many orders of magnitude and the amplitude terms set the peak. Per gradient component, relative
to its largest value over the sources, the changes are up to 1.0e-5 in f0, 1.5e-4 in ḟ, 4.0e-4
in f̈ and 8.3e-6 in φ0, below 1e-8 in the angles and below 1e-12 in the amplitude (envelope
gradients included). Per Fisher entry, relative to sqrt(F_ii F_jj),
they are up to 1.1e-5 in the row and column of f̈ on the four-day fixtures and 2.7e-7 on the
32-day fixtures, and at most 1.1e-8 and 1.6e-9 elsewhere (`fisher_entry_changes.py`, files
`golden_cpu{,_l1}_fisher_vs_step1.txt`).

**Criterion for gradients and Fisher matrices** (parts `fd_steps` and `fd_steps_l1`, decided by
the author on 2026-10-18 in place of the 1e-12 gate for these outputs). At the default steps
these derivatives carry their own noise, the same in the step 1 and the step 2 code: per gradient
component, relative to its scale, about 1e-2 to 0.25 in f0, 0.2 to 1.8 in ḟ and 0.6 to 13 in f̈
on these fixtures; per Fisher entry, relative to sqrt(F_ii F_jj), up to 18 in the f̈ entries and
at most 5.2e-4 elsewhere. GBGPU's `information_matrix` already notes the cause: roundoff in the
Fresnel phase, which this step does not touch, would leave the f0 and ḟ diagonals 2 % and 13 %
high at the shared steps, so it widens those two steps 10 and 100 times. The criterion is that the change between the two codes stays
within that noise. `scripts/dffr/finite_difference_steps.py` evaluates `get_ll_grad_stft` and
`information_matrix` (defaults and TDI track) at 1, 10 and 100 times their default steps with
both codes; at factor 1 its outputs equal the golden outputs bit for bit. The noise of a code is
the difference between its values at the default and the tenfold step; the change is the
difference between the codes at the default step. Both are taken per gradient component (largest
over sources) and per Fisher entry, relative to the step 1 values at the tenfold step.

| Output | Within the larger of the two noises | Largest change, relative | Largest ratio change / noise |
| --- | --- | --- | --- |
| Gradients, 7 fixtures | 62 of 63 components | 2.8e-4 (f̈, `astro`) | 1.10 |
| Fisher matrices, defaults and TDI track, 7 fixtures | 4526 of 4536 entries | 5.7e-5 (f̈ diagonal, `tukey_mid`) | 2.47 |

Accepted. The 11 components and entries above the noise are terms that both codes compute almost
exactly, so their noise is itself at the level of rounding: the ι component of the
`esa_aet_diag` gradient (change 9.0e-11 against noise 8.2e-11) and ten entries of the TDI-track
Fisher in amplitude, ι and ψ (changes 1.3e-12 to 4.0e-10 of their scale, at most 2.5 times the
noise). The changes also fall as the steps grow: from the default to the hundredfold step, by
factors of 20 to 1300 relative to the array peaks. As a control, step 1's change against step 0,
which the step 1 review accepted, shows the same picture with 28 terms above the noise and
ratios up to 1.9. Files: `fd_steps_step{0,1,2}_cpu{,_l1}.npz`,
`fd_steps_step2_vs_step1_cpu{,_l1}.txt`, `fd_steps_step1_vs_step0_cpu{,_l1}.txt`. The captures
were first made by `$OUT/regression/step2/fd_steps_chain.sh`, which also ran the step 0 code for
the control (`--compare` of the step 0 and step 1 captures gives the control files); the parts
`fd_steps` and `fd_steps_l1` reproduced the step 1 and step 2 captures bit for bit. The swap gradients follow the
same rule; the GPU runs, which keep them, check it there.

The step 2 captures `$OUT/regression/step2/golden_cpu.npz` and `golden_cpu_l1.npz` are the new
baseline, without the swap kernels.

**Accuracy before and after** (part `accuracy`). `GBGPU/tests/test_stft_gb_accuracy.py` gives the
same numbers in the frozen step 1 environment and after step 2, to every printed digit: end-to-end
mismatch 9.049e-3 at 10 side bins (3.950e-3 at 20), on-bin mismatch 1.409e-5. The tests pass in
both environments. Files: `accuracy_base_cpu.log`, `accuracy_cpu.log`.

**Test suites** (part `tests`). Criterion: the same outcomes as step 1. GBGPU: 39 passed, 1 failed
and 1 module not importable, the same two GPU-only cases. LAT: 291 tests, the 282 of step 1 plus
the 9 new ones, with the same 2 failures and 15 errors and 58 skipped (53 plus the 5 new CUDA
cases). Comparing the outcome of every test name between the step 1 and step 2 logs shows only
the new tests. Pass. Files: `gbgpu_tests_cpu.{log,xml}`, `lat_tests_cpu.log`.

### Benchmarks

**CPU** (parts `benchmarks` and `benchmarks_l1`), alone in the allocation, with the fixtures and
settings of steps 0 and 1 and without the swap kernels: one warm-up call and five timed calls per
point, three runs in separate processes. Compared with step 1 by `compare_benchmarks.py`, as in
step 1. These runs are the timing baseline of step 3. Files:
`$OUT/benchmarks/step2/cpu_{esa,l1}_run{1,2,3}.json`, `compare_cpu_{esa,l1}.txt`.

ESA fixture (`esa_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 1, three batches | Largest spread of the step 2 runs |
| --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.32 | 49.35 | 449.32 | 0.993 to 1.005 | 0.43 % |
| `get_fstat_ll_stft` | 80.39 | 758.09 | 7027.45 | 0.997 to 1.009 | 1.06 % |
| `fill_global_stft` | 5.18 | 48.15 | 438.61 | 0.994 to 1.005 | 0.38 % |
| `get_ll_grad_stft` | 83.98 | 767.86 | 6909.32 | 0.985 to 0.992 | 0.43 % |
| `information_matrix` | 42.44 | 361.03 | 3060.61 | 0.996 to 0.998 | 0.60 % |

`L1Orbits` fixture (`l1_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 1, three batches | Largest spread of the step 2 runs |
| --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.31 | 49.62 | 449.54 | 1.000 to 1.005 | 0.29 % |
| `get_fstat_ll_stft` | 80.21 | 762.45 | 7025.15 | 1.000 to 1.010 | 1.58 % |
| `fill_global_stft` | 5.16 | 48.31 | 437.27 | 0.996 to 1.006 | 0.23 % |
| `get_ll_grad_stft` | 83.76 | 773.44 | 6937.35 | 0.996 to 1.002 | 1.04 % |
| `information_matrix` | 42.23 | 360.42 | 3052.90 | 0.995 to 0.997 | 0.22 % |

No threshold applies in this step. Over the 15 points of each fixture the ratio lies between 0.985
and 1.009 (median 0.997) on the ESA fixture and between 0.995 and 1.010 (median 1.000) on the
`L1Orbits` fixture, against run-to-run spreads of up to 2.94 % (ESA) and 1.74 % (`L1Orbits`) in
step 1. Only the gradient on the ESA fixture is faster in all three batches, by 0.8 % to 1.5 %;
on the `L1Orbits` fixture it is not. The cheaper special functions thus cost or save nothing
measurable on the CPU, where the TDI sampling and the seven windowed terms of each pixel
dominate. Their effect on the GPU is measured with the GPU runs.

### Validation runs

None in this step.

### Pending on the GPU

After the step 1 GPU runs, so that `golden_gpu*.npz` of step 1 exist for the comparison. The
shared environment holds the step 3 code since 2026-10-18, so these runs come from the frozen
environment of the step 2 code, `$OUT/step2_env`, which reproduces the step 2 CPU golden outputs
bit for bit (step 3 records how it was built and checked). On a GPU node, with `S2=$OUT/step2_env`:
`PYTHON=$S2/.venv/bin/python BASE_ENV=$OUT/step1_env MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" $S2/lisa-analysis-tools/scripts/dffr/run_step2.sh --device gpu $OUT environment unit golden golden_l1 fd_steps fd_steps_l1 tests benchmarks benchmarks_l1`.
The GPU order is therefore: step 0 from `$OUT/base_env`, step 1 from `$OUT/step1_env`, step 2
from `$OUT/step2_env`, then step 3 from the shared environment.

## Step 3: tile seam

Code: LAT `feat-dffr` at 732ecf42 with the step 3 changes uncommitted (`cutils/lat_stft_kernels.hh`,
`scripts/dffr/`); GBGPU `feat-dffr` at 62c8d4c, unchanged. GBGPU instantiates the kernel templates
of the header, so its kernels are the new ones. Rebuilt on 2026-10-18 into the shared environment
with `JAX_PLATFORMS=cpu ONLY="lisa-analysis-tools GBGPU" ./install_erebor_spider.sh`, CPU and CUDA
backends. Without `JAX_PLATFORMS=cpu` the build completes but its smoke test crashes at
`import phentax`, whose JAX CUDA plugin fails on a node without an NVIDIA driver
(`$OUT/regression/step3/rebuild_main_env_first.log`); the step 3 code plays no part in it.

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41563985, 16 CPUs, 2026-10-18. The parts
ran inside that allocation as
`srun --jobid=<job> --overlap -n 1 -c 16 env PYTHON=$PYTHON BASE_ENV=$OUT/step2_env MOJITO_L1=<file> scripts/dffr/run_step3.sh --device cpu $OUT <part>`:
`environment reference` first, then `golden oracle accuracy`, `golden_l1` and `tests` side by
side, then `benchmarks benchmarks_l1` alone, and `resources` last
(`$OUT/regression/step3/step3_cpu_chain.sh`). The GPU runs are pending.

The gate of this step is bit identity (`numpy.array_equal`) of every output against the step 2
code, and timings within run-to-run noise. Unlike step 2, the CPU captures keep the two swap
kernels, because this step rewrites them and the GPU nodes are still down.

The last change of the code, the fill kernel constructing its template once per source instead of
once per kernel (two registers fewer on the H100, GPU resources below), came after a first full
CPU run. It changes no arithmetic and no sequence of calls, and the CPU runs were repeated for it
after a rebuild (`$OUT/regression/step3/rerun_after_fill_change.sh`). The results below are those
of the repeat, the final code; the run before the change gave the same bit identity and test
outcomes (`$OUT/regression/step3/before_fill_change/`).

### Frozen environment of the step 2 code

The pending GPU runs of step 2 must come from the step 2 code, and so must the step 2 outputs that
this step is compared with but that the step 2 files do not hold (the swap kernels on the CPU).
`scripts/dffr/build_base_env.sh` built `$OUT/step2_env` from LAT 732ecf42 and GBGPU 62c8d4c, the
committed step 2, with GPUBackendTools b07bd18, Eryn 26dd14a and BBHx b7b0ae5 plus the uncommitted
changes of the shared clones, and the third-party versions of the shared environment: 16 minutes
in job 41563985, 6.1 GB. Command:
`LAT_COMMIT=732ecf42 GBGPU_COMMIT=62c8d4c JAX_PLATFORMS=cpu scripts/dffr/build_base_env.sh $OUT/step2_env`.

Criterion: the environment must recompute the step 2 CPU golden outputs bit for bit. Result: 84 of
84 arrays of `golden_cpu.npz` and 15 of 15 of `golden_cpu_l1.npz` are bit-identical, pass
(`$OUT/regression/step2/step2_env_recheck_cpu{,_l1}.txt`). The check is part of the part
`reference` below.

### Step 2 references on the CPU

The part `reference` captures the step 2 outputs again in `$OUT/step2_env`, the swap kernels
included: 144 arrays on the six single-grid fixtures, 60 of them from the swap kernels, and 26 on
`l1_tdi2_envelope`, 11 from the swap kernels
(`$OUT/regression/step2/golden_cpu{,_l1}_with_swap.npz`), and the 54 arrays of
`scripts/validation/stft_column_policy_oracle.py` (`$OUT/regression/step2/column_oracle_cpu.npz`).
The arrays that the step 2 files also hold come back bit for bit (above).

### Unit tests

None: the step adds no numerics. The bit-identity regression below is the test of the seam.

### Regression tests

**Golden outputs against step 2** (parts `golden` and `golden_l1`, swap kernels included).
Criterion: bit identity of every array against the step 2 references, and of two captures in
separate processes. Result: 144 of 144 and 26 of 26 arrays bit-identical to step 2, and the
repeats bit-identical to the first captures, pass. Files:
`$OUT/regression/step3/golden_cpu{,_l1}_vs_step2.txt` and `golden_cpu{,_l1}_repeat_compare.txt`.

**Column-policy oracle** (part `oracle`). `stft_column_policy_oracle.py --compare` of the step 3
code against its capture in `$OUT/step2_env`: all 54 arrays bit-identical, pass
(`column_oracle_vs_step2_cpu.txt`).

**Accuracy before and after** (part `accuracy`). `GBGPU/tests/test_stft_gb_accuracy.py` gives the
same numbers in `$OUT/step2_env` and after step 3, to every printed digit: end-to-end mismatch
9.049e-3 at 10 side bins (3.950e-3 at 20), on-bin mismatch 1.409e-5. The tests pass in both
environments. Files: `accuracy_base_cpu.log`, `accuracy_cpu.log`.

**Test suites** (part `tests`). Criterion: the same outcomes as step 2. GBGPU: 39 passed, 1 failed
and 1 module not importable, the same two GPU-only cases; the outcome of each of the 41 test cases
is the same as in step 2. LAT: 291 tests with the same 2 failures, 15 errors and 58 skips as in
step 2; the 289 tests that the comparison pairs by name between the two logs have the same
outcomes. Pass. Files: `gbgpu_tests_cpu.{log,xml}`, `lat_tests_cpu.log`.

**GPU resources, without a GPU** (part `resources`). `scripts/dffr/kernel_resources.py` compiles
GBGPU's `gb_tdi_on_the_fly.cu`, which instantiates every STFT kernel, with the command of the build
tree and ptxas statistics, once with the step 2 header (from the clone in `$OUT/step2_env`) and
once with the step 3 header, for the A100 (sm_80, the GPU of the timing gate) and the H100 (sm_90).
Both have 65 536 registers per SM, allocated in steps of 8 per thread, so 4 blocks of 128 threads
fit at up to 128 registers per thread and 3 blocks at up to 168. The GPU timings depend on how
many blocks run at once. No threshold; compared with step 2. Files:
`$OUT/regression/step3/kernel_resources_sm{80,90}.txt`.

| Kernel | Own registers, sm_80, step 2 / step 3 | Blocks per SM at those counts, sm_80 | Own registers, sm_90 | Blocks per SM at those counts, sm_90 |
| --- | --- | --- | --- | --- |
| `stft_get_ll_kernel` | 120 / 120 | 4 / 4 | 128 / 122 | 4 / 4 |
| `stft_fill_global_kernel` | 128 / 120 | 4 / 4 | 128 / 127 | 4 / 4 |
| `stft_swap_ll_kernel` | 113 / 120 | 4 / 4 | 105 / 128 | 4 / 4 |
| `stft_get_fstat_ll_kernel` | 152 / 145 | 3 / 3 | 122 / 122 | 4 / 4 |
| `stft_get_ll_grad_kernel` | 128 / 127 | 4 / 4 | 144 / 122 | 3 / 4 |
| `stft_swap_ll_grad_kernel` | 168 / 162 | 3 / 3 | 144 / 122 | 3 / 4 |
| `stft_information_matrix_kernel` (unchanged) | 162 / 162 | 3 / 3 | 128 / 128 | 4 / 4 |

On both devices every kernel keeps at least the blocks per SM of step 2, without spills; on the
H100 the two gradient kernels gain one. The same source gives different register counts on the
two devices, in step 2 already, because each is compiled for its own instruction set, so the
check has to be made per device. The stack frame grows by 32 to 48 bytes per thread, mostly the
domain pointer and the two link-table pointers of the column context, and by 112 bytes in the
F-statistic kernel, whose two evaluation sites hold their templates apart. The fill kernel reaches
127 registers on sm_90 only with its template constructed per source: held for the whole kernel,
it needed 129, one above the 128 that allow 4 blocks, while sm_80 stayed at 120 either way.

**Correction of 2026-10-18 (step 4).** The registers in this table are those of each kernel's own
code, which ptxas reports for one file. GBGPU links its CUDA files with relocatable device code,
and at the device link a kernel that calls a function of another file gets the largest count
among itself and the functions it calls. Every kernel above calls
`STFTFresnelGPU::get_fourier_value` of `domains.cu`, which uses 244 registers on the A100 and 222
on the H100. In the linked module (`cuobjdump -res-usage`) every kernel has 244 registers on the
A100 in step 2 and in step 3 (the modules of `$OUT/step2_env` and `$OUT/step3_env`), and 222 on
the H100 in step 3 (step 4, part `resources`): 2 blocks of 128 threads per SM throughout. The
conclusion, that no kernel loses blocks per SM, holds; the blocks per SM in the table, and the
gains on the H100, do not describe the linked kernels. The linked stack frames grow from step 2 to
step 3 by 32 to 112 bytes per thread, as above. `kernel_resources.py` reads the linked module from
step 4 on.

### Benchmarks

**CPU** (parts `benchmarks` and `benchmarks_l1`), alone in the allocation, with the fixtures and
settings of steps 0 to 2 and without the swap kernels, as the step 2 runs they are compared with:
one warm-up call and five timed calls per point, three runs in separate processes. Criterion
(set by the author on 2026-10-18): at each point, the ratio of the median times of step 3 and step
2 lies within the larger run-to-run spread of the two steps at that point, (largest - smallest) /
median of the three run medians. Reference: the step 2 runs of 2026-10-18 morning. Files:
`$OUT/benchmarks/step3/cpu_{esa,l1}_run{1,2,3}.json`, `compare_cpu_{esa,l1}.txt`.

ESA fixture (`esa_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 2, three batches | Largest spread of the step 3 runs | Within noise |
| --- | --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.33 | 49.44 | 450.18 | 1.002 | 0.35 % | 3 of 3 |
| `get_fstat_ll_stft` | 80.31 | 755.26 | 6996.82 | 0.996 to 0.999 | 0.67 % | 3 of 3 |
| `fill_global_stft` | 5.18 | 48.09 | 436.64 | 0.996 to 0.999 | 0.39 % | 2 of 3 |
| `get_ll_grad_stft` | 84.15 | 771.03 | 6945.12 | 1.002 to 1.005 | 0.23 % | 2 of 3 |
| `information_matrix` | 42.34 | 361.66 | 3059.33 | 0.998 to 1.002 | 0.24 % | 3 of 3 |

`L1Orbits` fixture (`l1_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 2, three batches | Largest spread of the step 3 runs | Within noise |
| --- | --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.31 | 49.78 | 450.87 | 0.999 to 1.003 | 1.68 % | 3 of 3 |
| `get_fstat_ll_stft` | 80.59 | 765.79 | 7045.75 | 1.003 to 1.005 | 0.53 % | 3 of 3 |
| `fill_global_stft` | 5.17 | 48.55 | 438.17 | 1.001 to 1.005 | 0.99 % | 3 of 3 |
| `get_ll_grad_stft` | 83.86 | 779.23 | 6959.95 | 1.001 to 1.007 | 0.70 % | 2 of 3 |
| `information_matrix` | 42.15 | 360.81 | 3054.67 | 0.998 to 1.001 | 0.41 % | 3 of 3 |

27 of the 30 points lie within the noise. Of the three outside, one is faster, `fill_global_stft`
at batch 64 on the ESA fixture (ratio 0.996, spreads 0.35 % in step 2 and 0.21 % in step 3), and
two are slower, both `get_ll_grad_stft`: batch 64 on the ESA fixture (ratio 1.005, spreads 0.18 %
and 0.20 %) and batch 8 on `L1Orbits` (ratio 1.007, spreads 0.74 % and 0.70 %). The ratios lie
between 0.996 and 1.005 (median 1.000) on the ESA fixture and between 0.998 and 1.007 (median
1.003) on the `L1Orbits` fixture. The two slower points are drift between runs of the same code,
not a cost of this step: the CPU machine code of `stft_get_ll_grad_kernel` and of
`stft_eval_block_ll` is identical, instruction for instruction, before and after the last change
of the fill kernel (`objdump -d` of the two CPU objects, `$OUT/regression/step3/compile_check/`),
and in the run before that change the same two points came out at 0.996 and 0.993, inside the
noise (`$OUT/benchmarks/step3/before_fill_change/`). Runs of identical code an hour apart thus
differ by up to 1 %, more than the spreads within one run (0.2 % to 0.7 % at these points), so a
comparison with runs of another day cannot resolve changes below about 1 %. The kernel of
`information_matrix` did not change in this step, and its ratios, 0.998 to 1.002, show the same
drift. This reading of the three points goes to the author's review of the step.

**A first version of the seam**, kept in `$OUT/regression/step3/first_version/` and
`$OUT/benchmarks/step3/first_version/`, was already bit-identical (144 of 144 and 26 of 26 arrays,
the same test outcomes), but cost time and registers. On the CPU, `Template::values` stayed out of
line under `-Os`, one more call per pixel: `get_ll_stft` was 1.1 % and 0.5 % slower at batches 1
and 8 on the ESA fixture, outside the spreads of 0.56 % and 0.44 %. On sm_80 its templates lived in
the kernels and its launch settings in the column context, which raised the gradient kernels to 170
and 232 registers, and the `get_ll` gradient to 129 once the templates had moved. Three changes
removed these costs, none of which touches the arithmetic: `values()` is forced inline on the CPU,
the templates are locals of the evaluation functions, and the launch settings reach
`Template::build` as arguments instead of members of the column context. The results above are
those of the final code. That run also showed the drift between days: the unchanged
`information_matrix` came out at 1.005 on `L1Orbits` at batch 8, outside its spreads of 0.22 % and
0.19 %.

### Validation runs

None in this step.

### Pending on the GPU

After the step 2 GPU runs, so that `golden_gpu*.npz` and the GPU benchmarks of step 2 exist for the
comparison. Since step 4 the shared environment holds step 4 code, so the step 3 runs come from
the frozen environment of the step 3 code (step 4, below), with its own clone of the scripts. On a
GPU node, from `$OUT/step3_env/lisa-analysis-tools`:
`PYTHON=$OUT/step3_env/.venv/bin/python BASE_ENV=$OUT/step2_env MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step3.sh --device gpu $OUT environment golden golden_l1 tests benchmarks benchmarks_l1`.
On the GPU the captures and the benchmarks keep the swap kernels, and the reference is the step 2
GPU capture itself. `fill_global_stft` adds into its template rows with `atomicAdd`, so the GPU
captures of steps 0 to 2 first show whether that output is reproducible from run to run; bit
identity against step 2 can only be asked of it if it is.

## Step 4: DFFR kernels

Code: LAT `feat-dffr` at 7617da1a and GBGPU `feat-dffr` at 62c8d4c, with the changes of parts 4a
and 4b uncommitted; part 4c, the DFFR Fisher, follows. Rebuilt on 2026-10-18 into the shared
environment with `JAX_PLATFORMS=cpu ONLY="lisa-analysis-tools GBGPU" ./install_erebor_spider.sh`,
CPU and CUDA backends. It was rebuilt again after `lat_dffr_tile.hh` was merged into
`lat_dffr_kernels.hh`. The merge left the CPU machine code of every function unchanged in the
DFFR unit, the unit of the existing response and the LAT diagnostics (58, 86 and 14 functions,
`objdump -d`, `$OUT/regression/step4/compile_check/cpu_machine_code_after_merge.txt`). The DFFR
tests and the exchange gave the same results after it.

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41563985, 16 CPUs, 2026-10-18. The parts
ran inside that allocation as
`srun --jobid=<job> --overlap -n 1 -c 16 env PYTHON=$PYTHON BASE_ENV=$OUT/step3_env DFFR_PYTHON=<Python of the dffr environment> MOJITO_L1=<file> scripts/dffr/run_step4.sh --device cpu $OUT <part>`:
every part but the benchmarks side by side, then `benchmarks benchmarks_l1` alone
(`$OUT/regression/step4/step4_cpu_chain.sh`). The GPU runs are pending.

The existing response must stay bit-identical to the step 3 code, with timings within run-to-run
noise. The DFFR code is checked in four ways:

- against quadrature and `TDIConfig`;
- against itself, through the identities of the consumers;
- against standalone dffr;
- against the existing response.

### Frozen environment of the step 3 code

The pending GPU runs of step 3 need the step 3 code after the rebuild, and so do the comparisons
of this step. `scripts/dffr/build_base_env.sh` built `$OUT/step3_env` before the first rebuild with
step 4 code: 16 minutes in job 41563985, 6.1 GB. It holds LAT 7617da1a and GBGPU 62c8d4c, the
committed step 3, with GPUBackendTools b07bd18, Eryn 26dd14a and BBHx b7b0ae5 plus the
uncommitted changes of the shared clones. Command:
`LAT_COMMIT=7617da1a GBGPU_COMMIT=62c8d4c JAX_PLATFORMS=cpu scripts/dffr/build_base_env.sh $OUT/step3_env`.

Criterion: the environment must recompute the step 3 CPU golden outputs bit for bit, the swap
kernels included. Result: 144 of 144 and 26 of 26 arrays bit-identical, pass
(`$OUT/regression/step3/step3_env_recheck_cpu{,_l1}.txt`).

### Unit tests

**`tests/test_dffr.py` (LAT).** The source-independent DFFR stages, through three diagnostic
methods on existing wraps: `STFTFresnelWrap.compute_dffr_sub_intervals` and
`compute_dffr_windowed_kernels`, and `KinematicsTableWrap.compute_dffr_link_coefficients`. The
measured values come from part `unit_numbers` (`scripts/dffr/dffr_unit_numbers.py`,
`$OUT/regression/step4/unit_numbers_cpu.txt`). The transform errors are in DFFR's normalisations:
the kernel as a fraction of the segment duration, the moment as a fraction of the square of half
of it.

| Test | Cases | Bound | Largest measured | Result |
| --- | --- | --- | --- | --- |
| Sub-interval integral against adaptive quadrature | 16: chirp rates ±4e-13, 1e-20 and -1e-22 Hz/s; detunings 0, 0.3, 7 and -18.5 bins | 2e-10 | 7.3e-11 | pass |
| Its first moment | the same | 5e-8 | 2.9e-8 | pass |
| Windowed transform against piecewise quadrature | 132: tapers 0, 0.1157 and 0.5; 4 chirp rates; 11 bins from -15 to +15 about a carrier 0.37 bins off the grid | 4e-10 | 4.5e-12 | pass |
| Its first moment | the same | 1e-7 | 6.4e-13 | pass |
| Closed-form TDI coefficients against the unit sums of `TDIConfig`, absolute | 64 segments; carriers 0.1 to 50 mHz; light travel times 8.33 ± 0.03 s per link and segment | 1e-13 | 2.5e-16 (1st generation), 1.1e-15 (2nd) | pass |
| `TDIConfig.generation` | the stock combinations in any order; one unit removed | exact | 1, 2 and `None` | pass |
| Argument checks | generation 3; a short output; a zero floor; df dt ≠ 1 | `ValueError` | raised | pass |

The sub-interval bounds are those of the standalone package's `tests/test_fresnel.py`. The windowed
bounds are twice those, since the six sub-intervals carry window weights that sum to 2. The
largest sub-interval errors sit 0.3 bins from the carrier at the Doppler-dominated rates, where the
rational fits of `lat_fresnel.hh` (about 2e-10) set them. The CUDA class of the file is skipped
without a GPU: 5 tests, CUDA against CPU among them, with a provisional bound of 1e-13.

**`GBGPU/tests/test_stft_gb_dffr.py`.** The `gb_dffr_*` entry points, called with the arguments
the comp will pass in step 5. Fixture: 24 six-hour segments of 10 s samples, the band 3.9 to 4.6
mHz, `n_side_bins = 3`, equal-arm orbits, seeded data and an identity inverse covariance. All 10
CPU tests pass:

- fill_global followed by the domain's inner product equals get_ll (rtol 1e-10), for tapers 0
  and 0.5 and both TDI generations;
- the compact template equals the fill_global grid bit for bit on its band;
- swap(A, A) equals get_ll(A) term by term (rtol 1e-10);
- exchanging add and remove exchanges the terms of swap and conjugates the cross term;
- swap on the union band of two sources 2.6 bins apart equals the inner products of compact
  templates on that band (rtol 1e-10);
- the N and M of the F-statistic equal the get_ll and swap terms of its four filters (rtol 1e-9);
- both gradients equal host central differences (rtol 1e-6), and eps ≤ 0 freezes a component;
- DFFR against the existing response on the same fixture, both generations: overlap above 0.99
  and norm ratio in (0.9, 1.1). These bounds were set before measuring, to catch a sign, a
  conjugation or a factor 2;
- rows_per_pass 24, generation 3, a kinematics table of another grid and an AET domain raise
  `ValueError`.

The CUDA class of the file is skipped without a GPU: 9 tests, among them CUDA against CPU and the
16-row tiles against the 32-row tiles, with a provisional bound of 1e-12 of each peak.

### Regression tests

**Golden outputs against step 3** (parts `golden` and `golden_l1`, swap kernels included).
Criterion: bit identity against the step 3 captures (`$OUT/regression/step3/golden_cpu{,_l1}.npz`,
which `$OUT/step3_env` reproduces), and between two captures in separate processes. Result: 144
of 144 and 26 of 26 arrays bit-identical to step 3, and the repeats bit-identical to the first
captures, pass. Files: `$OUT/regression/step4/golden_cpu{,_l1}_vs_step3.txt` and
`golden_cpu{,_l1}_repeat_compare.txt`.

**Column-policy oracle** (part `oracle`). Against its capture in `$OUT/step2_env`, as in step 3:
all 54 arrays bit-identical, pass (`column_oracle_vs_step2_cpu.txt`).

**Accuracy** (part `accuracy`). `GBGPU/tests/test_stft_gb_accuracy.py` gives the same numbers in
`$OUT/step3_env` and after step 4, to every printed digit: end-to-end mismatch 9.049e-3 at 10 side
bins (3.950e-3 at 20), on-bin mismatch 1.409e-5. The 4 tests pass in both. Files:
`accuracy_base_cpu.log`, `accuracy_cpu.log`.

**Test suites** (part `tests`). Criterion: every test case of step 3 keeps its outcome. Result:
pass. Files: `gbgpu_tests_cpu.{log,xml}`, `lat_tests_cpu.log`.

- GBGPU: the 41 cases of step 3 keep theirs (39 passed; 1 failed and 1 module not importable, the
  two GPU-only cases). The 10 CPU tests of `test_stft_gb_dffr.py` pass and its 9 CUDA tests are
  skipped: 49 passed and 9 skipped in all.
- LAT: 301 tests. The 289 that pair by name with step 3 keep their outcomes (the same 2 failures
  and 15 errors). The 5 CPU tests of `test_dffr.py` pass and its 5 CUDA tests are skipped: 63
  skips in all.

**Exchange with standalone dffr** (part `exchange`). Three scripts:

- `dffr_exchange_lat.py --capture` stores the inputs of a DFFR evaluation and LAT's outputs on
  them;
- `dffr_exchange_dffr.py` recomputes the outputs in the environment of the dffr repository at
  01530a5, from the same kinematics arrays (`OrbitKinematics.from_arrays`);
- `dffr_exchange_lat.py --compare` reads both.

Fixture: ESA orbits; 32 one-day segments from 10 days after their start; the production band (0.1
to 29 mHz) as one domain, so that both sides read the same f_min; Tukey taper 0.1157 with the
segment midpoints; `n_side_bins = 10`; TDI 2nd generation; linear envelope and first-order delays
on; the 24 sources of the production fixtures; seeded data and inverse covariance. Criterion, set
before measuring: per output, the largest difference relative to the peak of the dffr array is at
most 1e-13, and integer arrays are equal. Result: pass.

| Output | Largest difference over peak |
| --- | --- |
| Compact templates; fill_global grid; segment table (effective frequencies, chirp rates, reference phasors, link amplitudes, envelope slopes, emission phasors and frequencies, TDI weights) | 0, bit-identical |
| Band starts of the compact templates and of the segment table | equal |
| (d\|h) | 1.6e-15 |
| (h\|h) | 6.3e-15 |

The inner products differ by the order of their sums only. Files: `exchange_{lat,dffr}_cpu.npz`,
`exchange_compare_cpu.txt`.

**DFFR against the existing response** (part `against_existing`). There is no threshold: the
numbers say how far apart the two models are, not which is closer to the truth (step 6). Per
source, the templates of both responses over the band of the exchange fixture (fill_global, factor
one). a is the DFFR template and b the existing one, with unit weights over the three channels. The
settings are the two production settings of the existing response on ESA orbits and the two
crossed ones. DFFR takes the same TDI generation and envelope, with its first-order delay terms on.
Files: `against_existing_cpu.{txt,npz}`.

| Setting | Median mismatch, 1 - Re(a\|b) / sqrt((a\|a)(b\|b)) | Largest | Largest phase of (a\|b) [rad] | Norm ratio (a\|a) / (b\|b) |
| --- | --- | --- | --- | --- |
| TDI 2nd generation, linear envelope (production) | 5.6e-10 | 2.6e-8 | 2.2e-5 | 0.99998 to 1.00008 |
| TDI 1st generation, linear envelope | 5.6e-10 | 2.6e-8 | 2.2e-5 | 0.99998 to 1.00008 |
| TDI 2nd generation, constant envelope | 8.6e-6 | 4.3e-5 | 2.7e-5 | 0.99995 to 1.00006 |
| TDI 1st generation, constant envelope (production) | 8.6e-6 | 4.3e-5 | 1.7e-5 | 0.99997 to 1.00003 |

The TDI generation changes nothing; the envelope sets the distance. With the linear envelope the
two responses agree to a median mismatch of 5.6e-10. With a constant envelope each holds the
amplitude constant over a segment in its own way, and they differ by about 1e-5.

**GPU resources, without a GPU** (part `resources`). `scripts/dffr/kernel_resources.py`, rewritten
in this step, works as follows:

- it compiles every CUDA file of GBGPU's GPU module for one device, with the commands of the
  build tree;
- it links them as the build does;
- it reads each kernel's registers, stack frame and static shared memory from the linked code
  (`cuobjdump -res-usage`).

With relocatable device code, a kernel gets at the link the largest register count among itself
and the functions it calls. ptxas on one file, as in step 3, reports the kernel's own count only.
A small host program, built with the flags of `gb_dffr.cu`, gives the dynamic shared memory of a
DFFR template: 32 672 bytes at 32 rows, 16 480 at 16. Blocks of 128 threads per SM are counted two
ways:

- by registers: 65 536 per SM, allocated in steps of 8 per thread;
- by shared memory: 164 kB per SM on the A100 and 228 kB on the H100, with the 1 kB the system
  keeps per block.

The reference is the step 3 module, compiled and linked from the build tree of `$OUT/step3_env`.
There is no threshold. The record and the register caps and device link-time optimisation below
come from the final code, after the merge of the headers (`$OUT/regression/step4/merge_chain.sh`);
the forced inline was measured before it. Files:
`$OUT/regression/step4/kernel_resources_sm{80,90}.txt`.

Existing kernels: on both devices, every field equals step 3: 244 registers on the A100 and 222 on
the H100, the same stack frames and static shared memory, 2 blocks per SM.

DFFR kernels, at 32 rows unless stated; the two-template consumers hold twice the template:

| Kernel | Registers, A100 / H100 | Blocks per SM by registers | Blocks per SM by shared memory, A100 / H100 |
| --- | --- | --- | --- |
| `stft_get_ll_kernel` | 158 / 154 | 3 | 4 / 6 |
| `stft_fill_global_kernel` | 156 / 156 | 3 | 4 / 6 |
| `stft_get_ll_grad_kernel` | 164 / 162 | 3 | 4 / 6 |
| `stft_swap_ll_kernel` | 220 / 222 | 2 | 2 / 3 at 32 rows; 3 / 5 at 16 |
| `stft_get_fstat_ll_kernel` | 220 / 222 | 2 | 2 / 3 at 32 rows; 3 / 5 at 16 |
| `stft_swap_ll_grad_kernel` | 254 / 254 | 2 | 2 / 3 at 32 rows; 3 / 5 at 16 |
| `dffr_fill_template_kernel`, `dffr_segment_table_kernel` | 142 / 142 | 3 | 4 / 6 |

Called functions set these counts on the A100:

- `STFTFresnelGPU::get_fourier_value` of `domains.cu`, 244, for every existing kernel;
- `STFTDomainGPU::add_ip_swap_contrib`, 220, for the swap kernels and the F-statistic of both
  responses;
- in the DFFR swap gradient, `DFFRTile::Template::values` and `build`, which nvcc keeps out of
  line there (254 each).

The H100 code reports no registers for called functions. On both devices the registers set the
occupancy of every DFFR kernel; at 32 rows on the A100, the shared memory of the two-template
kernels gives the same 2 blocks. So 16 and 32 rows give the same occupancy today.

Variants measured on the same module, not adopted (files in `$OUT/regression/step4/`):

| Variant | Effect on the linked registers, A100 / H100 | Decision of the author |
| --- | --- | --- |
| `Template::values` and `build` of the DFFR tile forced inline on the GPU (`kernel_resources_forced_inline_sm*.txt`) | DFFR swap gradient 254 → 220 / 222, stack frame 1048 → 168 bytes; but the `get_ll` gradient 164 → 254, because its evaluation function then stays out of line. Forcing `values` alone gives the same. | reverted; to be timed in step 6 |
| `domains.cu` capped at 168 registers (`-maxrregcount`; `compile_check/register_cap_168_sm*.txt`) | every existing kernel 244 / 222 → 166: 3 blocks per SM. DFFR swap → 172 / 166: 3 blocks on the H100 only, since its own code needs 172 on the A100. `get_fourier_value` spills 164 bytes per call. At 128 (`register_cap_128_sm*.txt`) the existing kernels reach 134 to 162 on the A100 and 134 on the H100, still 3 blocks, with 248 and 224 bytes of spills. | step 6, A/B timing on an A100 |
| Device link-time optimisation, `-dlto` (`device_lto_sm*.txt`) | the DFFR kernels 128 to 168 on the A100 and 130 to 172 on the H100: 3 blocks (4 for the compact template on the A100), except the swap gradient on the H100 (172, 2 blocks). The existing kernels go to 255, still 2: once inlined, the Fresnel evaluation needs these registers. | step 6 |

The `-dlto` link needs `nvvm/lib64/libnvvm.so` next to nvlink. The pip wheel of the CUDA 13
toolkit keeps it in `lib/` as `libnvvm.so.4`, and nvlink ignores `-nvvmpath`. The measurement used a
copy of the toolkit's `bin/` with that path linked to the library
(`$OUT/regression/step4/device_lto/toolkit`, `kernel_resources.py --toolkit`).

### Benchmarks

**CPU** (parts `benchmarks` and `benchmarks_l1`), alone in the allocation, with the fixtures,
settings and criterion of step 3, against the step 3 runs of 2026-10-18. Files:
`$OUT/benchmarks/step4/cpu_{esa,l1}_run{1,2,3}.json`, `compare_cpu_{esa,l1}.txt`.

ESA fixture (`esa_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 3, three batches | Largest spread of the step 4 runs | Within noise |
| --- | --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.36 | 49.31 | 449.98 | 0.997 to 1.006 | 1.07 % | 3 of 3 |
| `get_fstat_ll_stft` | 80.40 | 754.11 | 6982.84 | 0.998 to 1.001 | 1.55 % | 3 of 3 |
| `fill_global_stft` | 5.17 | 48.00 | 436.79 | 0.998 to 1.000 | 0.66 % | 2 of 3 |
| `get_ll_grad_stft` | 84.19 | 765.62 | 6889.92 | 0.992 to 1.000 | 0.64 % | 1 of 3 |
| `information_matrix` | 42.40 | 361.07 | 3052.65 | 0.998 to 1.001 | 0.67 % | 2 of 3 |

`L1Orbits` fixture (`l1_tdi2_envelope`):

| Method | Batch 1 [ms] | Batch 8 [ms] | Batch 64 [ms] | Ratio to step 3, three batches | Largest spread of the step 4 runs | Within noise |
| --- | --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 5.33 | 49.92 | 452.40 | 1.003 to 1.005 | 1.57 % | 3 of 3 |
| `get_fstat_ll_stft` | 80.41 | 765.37 | 7055.05 | 0.998 to 1.001 | 1.29 % | 3 of 3 |
| `fill_global_stft` | 5.14 | 48.64 | 439.25 | 0.994 to 1.002 | 1.14 % | 3 of 3 |
| `get_ll_grad_stft` | 83.78 | 776.10 | 6957.79 | 0.996 to 1.000 | 1.24 % | 3 of 3 |
| `information_matrix` | 42.10 | 360.50 | 3051.54 | 0.999 | 0.96 % | 3 of 3 |

26 of the 30 points lie within the noise. The 4 outside are all faster, all on the ESA fixture:
`fill_global_stft` at batch 8 (ratio 0.998), `get_ll_grad_stft` at batches 8 and 64 (0.993 and
0.992) and `information_matrix` at batch 64 (0.998).

The CPU machine code of the existing response is that of step 3. Of the 86 functions of GBGPU's
CPU unit of the existing response, 84 are identical instruction for instruction. The other two,
`ucb_f` and `ucb_phase`, now call `gb_physics.hh`. They differ only in register names and, in
`ucb_f`, in the order of the two factors of one fused multiply-add, which gives the same bits
(`compile_check/cpu_machine_code_vs_step3.txt`, `cpu_machine_code_changed_functions.txt`). The four
points are therefore drift between runs, which the unchanged `information_matrix` also shows. Step
3 saw drift of up to 1 % between runs of identical code.

### Validation runs

None in this step.

### Pending on the GPU

After the step 3 GPU runs. On a GPU node, from the shared environment:
`PYTHON=$PYTHON BASE_ENV=$OUT/step3_env DFFR_PYTHON=<Python of the dffr environment> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step4.sh --device gpu $OUT environment golden golden_l1 tests unit_numbers exchange benchmarks benchmarks_l1`.
This runs:

- the CUDA classes of both test files, CUDA against CPU and 16 against 32 rows among them;
- the exchange on the GPU backends of both packages;
- the existing response against the step 3 GPU captures.

The first GPU run also reaches code that no CPU run reaches: the launch above 48 kB, the band-bin
map at 32 rows and the device copy of the domain.

## Step 5: wiring

Code: LAT `feat-dffr` at de526e1b and GBGPU `feat-dffr` at e6fb653, the committed step 4, with the
changes of step 5 uncommitted. Step 5 changes Python only: `gbcomps.py` in GBGPU, `recipe.py` in
LAT, tests and scripts. The editable builds of the step 4 code load Python from the source trees,
so nothing was rebuilt.

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41576984, 16 CPUs, 2026-10-19. The parts
ran inside that allocation as
`srun --jobid=<job> --overlap -n 1 -c 16 env PYTHON=$PYTHON BASE_ENV=$OUT/step4_env DFFR_PYTHON=<Python of the dffr environment> MOJITO_L1=<file> scripts/dffr/run_step5.sh --device cpu $OUT <part>`:
every part but the benchmarks side by side, then `benchmarks benchmarks_l1` alone
(`$OUT/regression/step5/step5_cpu_chain.sh`). The GPU runs are pending.

The existing response must stay bit-identical to the step 4 code, with timings within run-to-run
noise. The DFFR response, called through `STFTGBComputations`, must give the outputs of its entry
points bit for bit.

### Frozen environment of the step 4 code

The pending GPU runs of step 4 need the step 4 tests and scripts, which step 5 changes in the
trees of the shared environment. `scripts/dffr/build_base_env.sh` built `$OUT/step4_env` before
the first change: 14 minutes in job 41576984, 6.1 GB. It holds LAT de526e1b and GBGPU e6fb653, the
committed step 4, with GPUBackendTools, Eryn and BBHx as in the shared clones, with their
uncommitted changes. Command:
`LAT_COMMIT=de526e1b GBGPU_COMMIT=e6fb653 JAX_PLATFORMS=cpu scripts/dffr/build_base_env.sh $OUT/step4_env`
(chain: `$OUT/regression/step5/step4_env_chain.sh`).

Criterion: the environment must recompute the step 4 CPU captures bit for bit: the golden outputs
of the existing response, swap kernels included, and the LAT side of the exchange with dffr.
Result: 144 of 144 and 26 of 26 arrays, and 29 of 29 exchange entries, bit-identical, pass
(`$OUT/regression/step5/step4_env_recheck_cpu{,_l1}.txt`, `step4_env_recheck_exchange_cpu.txt`).

### Unit tests

**`GBGPU/tests/test_stft_gb_dffr.py`.** `_DFFRCase` now calls the methods of
`STFTGBComputations(response="dffr")`, bound to a stand-in group that carries the domain, the
Fresnel object, `linear_envelope` and the kinematics table. The step 4 identities therefore test
the dispatch. The entry points are still called directly for three things: to compare them with
the comp, for their own argument checks, and for the 16-row tiles on the GPU. All 17 CPU tests
pass; the 10 CUDA tests are skipped without a GPU. New in this step:

- Wiring. Each of the six methods equals its entry point called with the expected arguments, bit
  for bit, for two sources per call, in two settings that differ in every argument the comp
  passes:
  - TDI generation 1 and 2;
  - envelope off and on, first-order delays off and on;
  - chirp-rate floor 1e-15 and 1e-22 Hz/s;
  - `t_ref` 1234.5 s and 0;
  - taper 0 and 0.5;
  - a window start of 2 bins and none.

  The floor of 1e-15 Hz/s moves (d|h) and (h|h) of these sources by 6.3e-9 of their peak
  (`$OUT/regression/step5/floor_check.py`), so a misplaced floor would show. The test also checks
  that the comp asks the group for the table with its own orbits.
- Envelope. `get_ll` follows the group's `linear_envelope`. Both gradients follow it by default.
  `linear_envelope=False` on a group with the envelope on equals a group without it, bit for bit,
  and `True` the other way round.
- Replicas. A comp rebuilt from its recorded arguments, directly and through LAT's replica builder
  (`devicereplicas._build_gb_comp_replica`, which takes its multi-GPU path for device 1 against
  primary 0), keeps the response, the delays, the floor and the TDI generation. Its outputs equal
  those of the original bit for bit.
- Arguments. A custom `TDIConfig` (one unit removed), a floor of 0 and a group of AET channels
  raise `ValueError` at construction. A custom `TDIConfig` set later on a DFFR comp raises as well,
  while the default response still takes one. A group without `cpp_kinematics` raises
  `TypeError`, and `information_matrix` raises `NotImplementedError`.

**`GBGPU/tests/test_stft_gb.py`.** Three tests of the default response, all pass:

- `response="tdi_on_the_fly"` given explicitly, with DFFR settings it must ignore, gives every
  output of the seven methods bit for bit, the Fisher matrix included;
- on a group whose evaluator has the envelope on, the default gradients equal
  `linear_envelope=False`, and `None` gives other values;
- an unknown response raises `ValueError`.

**`tests/test_gb_likelihood_engine.py` (LAT).** `STFTEngineNumericTest` runs for both responses on
a real container of two bands. It fills band 0 through the engine, then runs `get_ll` on both
bands in one call, the degenerate swap, and the removal. For DFFR it also checks that the group
built its table from the comp's orbits. Both pass. The stand-in container of
`test_engine_calls_bind_to_STFTGBComputations_signatures` gained the per-split routing tables that
the engine reads (`split_map_by_split`, `ac_to_intra_by_split`). The test had failed since step 0;
it passes now.

### Regression tests

**Golden outputs of the existing response against step 4** (parts `golden` and `golden_l1`, swap
kernels included). Criterion: bit identity against the step 4 captures, and between two captures
in separate processes. Result: 144 of 144 and 26 of 26 arrays bit-identical to step 4, and the
repeats bit-identical, pass. Files: `$OUT/regression/step5/golden_cpu{,_l1}_vs_step4.txt` and
`golden_cpu{,_l1}_repeat_compare.txt`.

**Column-policy oracle** (part `oracle`). Against its capture in `$OUT/step2_env`: all arrays
bit-identical, pass (`column_oracle_vs_step2_cpu.txt`).

**Accuracy** (part `accuracy`). `GBGPU/tests/test_stft_gb_accuracy.py` gives the same numbers in
`$OUT/step4_env` and after step 5, to every printed digit: end-to-end mismatch 9.049e-3 at 10 side
bins (3.950e-3 at 20), on-bin mismatch 1.409e-5. The tests pass in both. Files:
`accuracy_base_cpu.log`, `accuracy_cpu.log`.

**Test suites** (part `tests`). Criterion: every test case of step 4 keeps its outcome. Result:
pass. Files: `gbgpu_tests_cpu.{log,xml}`, `lat_tests_cpu.log`.

- GBGPU: the 60 cases of step 4 keep their outcomes. 11 cases are new: 10 pass, and the CUDA
  wiring test is skipped. In all: 59 passed and 10 skipped, plus the failure and the error of step
  4, the two cases that need a GPU (`test_thirdbody.py`, and `test_gbgpu.py`, which does not
  collect).
- LAT: 302 tests. Every test that pairs by name with step 4 keeps its outcome, except
  `test_engine_calls_bind_to_STFTGBComputations_signatures`, which went from error to pass. The
  DFFR engine test is new and passes. In all: 2 failures and 14 errors (15 in step 4), 63 skips.

**DFFR through the comp: first golden capture** (parts `golden_dffr` and `golden_dffr_l1`). The
outputs of `stft_golden_outputs.py --response dffr` on the production configurations with XYZ
channels (`esa_tdi2_envelope`, `esa_tdi1_const`, `l1_tdi2_envelope`): 21 arrays per
configuration, every output of the seven methods but the Fisher matrix. Two captures in separate
processes are bit-identical, 42 of 42 and 21 of 21. They are the reference of the DFFR response
for step 6 (`$OUT/regression/step5/golden_dffr_cpu{,_l1}.npz`).

**Exchange with standalone dffr, through the comp** (part `exchange`). As in step 4, but the
fill_global grid, (d|h) and (h|h) of the LAT side now come from `STFTGBComputations`. Criteria: per
output at most 1e-13 of the peak of the dffr array, as in step 4; and bit identity of the LAT
capture against that of step 4, which called the entry points. Result: 14 outputs within the gate,
and 29 of 29 entries bit-identical to step 4, pass (`exchange_compare_cpu.txt`,
`exchange_lat_cpu_vs_step4.txt`).

**DFFR against the existing response** (part `against_existing`), with both templates now from
the comp. Criterion: bit identity against the step 4 results. Result: 13 of 13 entries
bit-identical, pass (`against_existing_cpu_vs_step4.txt`), so the numbers of step 4 stand: median
mismatch 5.6e-10 with the linear envelope.

**Finite-difference noise of the gradients, with the envelope on and off** (part
`envelope_noise`). This measurement backs the default `linear_envelope` of the gradients of DFFR,
the group's setting. `finite_difference_steps.py --envelope-variants` evaluates `get_ll_grad_stft`
on the three XYZ production configurations at the default steps times 1, 10 and 100. It does so
with the envelope forced on, forced off, and at the default, for both responses. A response's own
noise per component is the difference between its gradients at the default and the tenfold step,
largest over the 24 sources, over the component's largest value at the tenfold step, the measure
of step 2. There is no threshold: the question is whether the envelope raises the noise. Files:
`envelope_noise_{tdi_on_the_fly,dffr}_cpu.{npz,txt}`.

Ratio of the noise with the envelope on to the noise with it off, per component:

| Response, configuration | amp | f0 | ḟ | f̈ | φ0 | ι | ψ | λ | β |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DFFR, `esa_tdi1_const` | 0.31 | 1.00 | 1.00 | 1.00 | 1.00 | 0.48 | 1.25 | 1.00 | 1.00 |
| DFFR, `esa_tdi2_envelope` | 0.72 | 1.00 | 1.00 | 1.00 | 1.00 | 1.61 | 4.57 | 1.00 | 1.00 |
| DFFR, `l1_tdi2_envelope` | 1.52 | 1.00 | 1.00 | 1.00 | 1.00 | 1.23 | 1.10 | 1.00 | 1.00 |
| existing, `esa_tdi1_const` | 2.18 | 1.01 | 1.00 | 1.02 | 0.81 | 1.24 | 1.63 | 0.99 | 1.15 |
| existing, `esa_tdi2_envelope` | 0.70 | 1.03 | 1.10 | 0.99 | 1.03 | 0.43 | 0.70 | 0.96 | 1.18 |
| existing, `l1_tdi2_envelope` | 1.31 | 0.90 | 1.01 | 1.00 | 0.97 | 1.79 | 1.44 | 1.40 | 1.11 |

- DFFR: the envelope leaves the noise of f0, ḟ, f̈, φ0, λ and β unchanged to two digits. It
  changes that of the amplitude, ι and ψ by factors from 0.31 to 4.57, on noise levels of at most
  1.5e-8 of their scale. The largest noise of any component is 1.8 in f̈. Result: the envelope does
  not raise the noise that matters, so the default stays as designed.
- The default gradient of DFFR equals the forced-on gradient bit for bit on the two configurations
  whose group has the envelope on, and the forced-off one on `esa_tdi1_const`, whose group has it
  off.
- Observation 1, the existing response. Its envelope changes the noise by factors from 0.43 to
  2.18 here. The f0, ḟ and f̈ components, whose noise is largest, change by at most 10 %. The
  reason its derivatives default to the envelope off, a noise four to seven times higher
  (`STFTComputationGroup.envelope_override`), does not show in this measure on these fixtures.
  Nothing changes in this step.
- Observation 2, both responses. At the default steps, DFFR's noise is larger than the existing
  response's on the same fixture and envelope (`esa_tdi2_envelope`, envelope off):

  | Component | DFFR | existing |
  | --- | --- | --- |
  | f0 | 4.6e-2 | 2.0e-2 |
  | ḟ | 0.43 | 0.24 |
  | f̈ | 1.8 | 0.63 |
  | φ0 | 6.6e-2 | 9.4e-3 |

  In both responses the difference between the tenfold and the hundredfold step is about ten
  times smaller than that between the default and the tenfold step. The noise therefore falls as
  1/step: it is rounding in the evaluation, not a difference of the modelled signal. In DFFR's
  log-likelihood it amounts to a few 1e-12 of (h|h): for f0, ḟ and φ0, a median over the sources
  of 3e-12 to 6e-12 and at most 2.2e-11, for sources whose sqrt((h|h)) is 0.6 to 221. The steps of
  the DFFR Fisher matrix are reviewed in step 4c.

**Kinematics table** (`$OUT/regression/step5/kinematics_table_timing.py`). On the CPU, a group's
first DFFR call builds the table in 0.35 ms at 32 one-day segments and 2.4 ms at 365. Later calls
find it in 2 to 3 µs. For comparison, a `get_ll_stft` call of one source of the existing response
takes 5.4 ms on the ESA fixture (step 4 benchmarks).

### Benchmarks

**CPU** (parts `benchmarks` and `benchmarks_l1`), alone in the allocation, with the fixtures,
settings and criterion of step 3. The existing response is read against the step 4 runs; DFFR runs
through the comp, without a threshold, since the timing gate of DFFR is step 6 on an A100. Files:
`$OUT/benchmarks/step5/cpu_{esa,l1}{,_dffr}_run{1,2,3}.json`, `compare_cpu_{esa,l1}.txt`,
`compare_cpu_{esa,l1}_dffr_vs_existing.txt`.

The existing response against the step 4 runs of 2026-10-18, with the criterion of step 3: each
point within the larger run-to-run spread of the two sets.

| Fixture | Ratio step 5 / step 4, all points | Median | Within noise |
| --- | --- | --- | --- |
| ESA | 1.006 to 1.023 | 1.010 | 6 of 15 |
| `L1Orbits` | 0.998 to 1.010 | 1.005 | 14 of 15 |

Step 5 changes no compiled code. To tell a change of the code from drift of the node, two more
measurements ran alone in the same allocation (`$OUT/regression/step5/benchmark_ab_chain.sh` and
`python_only_ab.sh`; files `compare_cpu_{esa,l1}_ab_step4_vs_step4.txt`,
`compare_cpu_{esa,l1}_ab.txt`, `compare_cpu_l1_python_only.txt`):

| Measurement | ESA: ratio range, median, within noise | `L1Orbits`: ratio range, median, within noise |
| --- | --- | --- |
| The step 4 code (`$OUT/step4_env`) today, against its runs of 2026-10-18 | 1.003 to 1.012, 1.006, 7 of 15 | 0.997 to 1.009, 1.001, 14 of 15 |
| Step 5 against the step 4 code, the two environments in turn, three runs each | 0.998 to 1.008, 1.003, 14 of 15 | 1.004 to 1.013, 1.007, 13 of 15 |
| The step 5 Python against the step 4 Python, both on the binaries of `$OUT/step4_env`, in turn, four runs each (`get_ll_stft`, `fill_global_stft`, `get_ll_grad_stft` and `information_matrix` at batches 1 and 8) | not run | 0.994 to 1.002, 1.000, 8 of 8 |

- The step 4 code itself is 0.6 % slower on the ESA fixture than on 2026-10-18. This is drift of
  the node between the two days, and it explains most of the points outside the noise in the
  first table.
- Run in turn, the two environments differ by 0.3 % (ESA) and 0.7 % (`L1Orbits`). 29 of the 30
  ratios exceed 1, and 27 of the 30 points lie within the spread. The two environments hold
  separately compiled binaries of the same code: the shared environment was built from the
  step 4 working tree, `$OUT/step4_env` from its commits.
- On the same binaries, the step 5 Python changes nothing measurable: every point lies within the
  spread, with a median ratio of 1.000. In the last row, the second arm imports a copy of the
  frozen environment's `gbgpu` package that holds the step 5 `gbcomps.py`.

Result: pass. The step 5 code leaves the timings of the existing response unchanged. The
differences in the first two tables come from drift of the node and from the two builds.

DFFR through the comp against the existing response, the median of three runs of each, on the
same fixture, with one CPU thread (32 one-day segments, 10 side bins, so 21 bins per segment,
production taper; the CPU backend uses one segment row per pass). There is no threshold: the timing
gate of DFFR is the A100 of step 6.

ESA fixture (`esa_tdi2_envelope`):

| Method | Existing, batch 1 / 8 / 64 [ms] | DFFR, batch 1 / 8 / 64 [ms] | DFFR / existing |
| --- | --- | --- | --- |
| `get_ll_stft` | 5.40 / 49.90 / 452.91 | 1.61 / 15.88 / 156.00 | 0.299 to 0.344 |
| `fill_global_stft` | 5.28 / 48.54 / 442.18 | 1.47 / 14.51 / 142.49 | 0.278 to 0.322 |
| `get_ll_grad_stft` | 84.95 / 774.91 / 6977.06 | 27.66 / 282.02 / 2779.99 | 0.326 to 0.398 |
| `get_fstat_ll_stft` | 80.87 / 763.32 / 7049.87 | 25.19 / 257.15 / 2542.80 | 0.312 to 0.361 |

`L1Orbits` fixture (`l1_tdi2_envelope`):

| Method | Existing, batch 1 / 8 / 64 [ms] | DFFR, batch 1 / 8 / 64 [ms] | DFFR / existing |
| --- | --- | --- | --- |
| `get_ll_stft` | 5.36 / 50.10 / 452.86 | 1.61 / 15.81 / 155.92 | 0.300 to 0.344 |
| `fill_global_stft` | 5.19 / 48.72 / 441.36 | 1.45 / 14.49 / 142.69 | 0.280 to 0.323 |
| `get_ll_grad_stft` | 84.50 / 782.80 / 6999.16 | 27.85 / 283.19 / 2801.94 | 0.330 to 0.400 |
| `get_fstat_ll_stft` | 80.25 / 766.86 / 7071.30 | 25.21 / 256.54 / 2540.67 | 0.314 to 0.359 |

On the CPU, DFFR takes 0.28 to 0.40 of the time of the existing response, for every method and
batch size on both fixtures. The swap kernels are left out, as in the CPU runs of steps 2 to 4,
and so is the Fisher matrix, which DFFR does not have yet.

### Validation runs

None in this step.

### Pending on the GPU

After the step 4 GPU runs, which come from `$OUT/step4_env`. Part 4c changed the trees of the
shared environment, so these runs come from the frozen environment of the step 5 code,
`$OUT/step5_env` (part 4c), with its own scripts. On a GPU node:
`cd $OUT/step5_env/lisa-analysis-tools && PYTHON=$OUT/step5_env/.venv/bin/python BASE_ENV=$OUT/step4_env DFFR_PYTHON=<Python of the dffr environment> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step5.sh --device gpu $OUT environment golden golden_l1 golden_dffr golden_dffr_l1 tests exchange against_existing envelope_noise benchmarks benchmarks_l1`.
This runs the CUDA classes of the test files, the CUDA wiring test among them, and the first GPU
captures and timings of DFFR through the comp.

## Part 4c: DFFR Fisher matrix

Part 4c of step 4 ran after step 5, on the committed step 5. Code: LAT `feat-dffr` at 568bb265 and
GBGPU `feat-dffr` at 5f8c1db, with the changes of part 4c uncommitted: LAT `lat_dffr_kernels.hh`
and scripts; GBGPU's DFFR source model, entry points, binding, `gbcomps.py` and tests. Rebuilt on
2026-10-19 into the shared environment with `ONLY="lisa-analysis-tools GBGPU"
./install_erebor_spider.sh`, CPU and CUDA backends, last at 11:07 to 11:15.

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41576984, 16 CPUs, 2026-10-19. The parts
ran inside that allocation as
`srun --jobid=<job> --overlap -n 1 -c 16 env PYTHON=$PYTHON BASE_ENV=$OUT/step5_env DFFR_PYTHON=<Python of the dffr environment> MOJITO_L1=<file> scripts/dffr/run_step4c.sh --device cpu $OUT <part>`:
the regression parts side by side, then the timing parts alone
(`$OUT/regression/step4c/part3_chain.sh`). The GPU runs are pending.

The existing response and the likelihood outputs of DFFR must stay bit-identical to step 5. The
DFFR Fisher matrix must equal a reference built from `fill_global` templates to 1e-13 of
sqrt(F_ii F_jj), lie on the plateau of its steps, and be no slower than the existing Fisher on the
CPU.

### Frozen environment of the step 5 code

Part 4c changes the trees of the shared environment, so its baseline and the pending step 5 GPU
runs need the step 5 code frozen. `scripts/dffr/build_base_env.sh` built `$OUT/step5_env` before
the first change: 14 minutes in job 41576984, 6.1 GB. It holds LAT 568bb265 and GBGPU 5f8c1db,
with GPUBackendTools, Eryn and BBHx as in the shared clones. Command:
`LAT_COMMIT=568bb265 GBGPU_COMMIT=5f8c1db JAX_PLATFORMS=cpu scripts/dffr/build_base_env.sh $OUT/step5_env`
(chain: `$OUT/regression/step4c/step5_env_chain.sh`).

Criterion: the environment must recompute the step 5 CPU captures bit for bit: the golden outputs
of the existing response, swap kernels included; those of DFFR through the comp; and the LAT side
of the exchange with dffr. Result: 144 of 144, 26 of 26, 42 of 42 and 21 of 21 arrays, and 29 of
29 exchange entries, bit-identical, pass (`$OUT/regression/step4c/step5_env_recheck_*.txt`).

### Unit tests

**`GBGPU/tests/test_stft_gb_dffr.py`.** `DFFRFisherTest` checks the DFFR Fisher matrix on the
fixture of the file: 24 six-hour segments, 3 side bins, two sources, each with a random Hermitian
positive-definite inverse CSD of its own, so that the cross-channel contraction carries weight.
Two settings differ in the TDI generation, the window (taper 0.5 or boxcar), the envelope and the
first-order delays. All 10 tests pass on the CPU:

- Parity with `gbgpu.testing.stft_information_matrix_reference`, which differences the
  `fill_global_stft` templates and contracts them in NumPy: on all 9 parameters, on the 8 of the
  global fit, and with the two-point stencil. Criterion: per entry, at most 1e-13 of
  sqrt(F_ii F_jj). Measured: 2.3e-15 to 3.3e-15 (`$OUT/regression/step4c/parity_numbers.py`). The
  matrices are exactly symmetric.
- Exact entries. h is linear in A, and φ0 multiplies it by exp(-i φ0), so F_AA A² and F_φ0φ0
  equal (h|h) of `get_ll_stft`, and F_Aφ0 vanishes. Criterion: 1e-5 relative. Measured: F_AA A²
  3.7e-14 to 3.9e-14; F_φ0φ0 1.2e-7 to 2.1e-7; |F_Aφ0| / sqrt(F_AA F_φ0φ0) 9.4e-7 to 1.0e-6. The
  phase-offset step of 1e-6 rad is added to template phases of up to 1.4e4 rad on this fixture,
  and rounding the sum to a double changes the realised step by up to 9e-7 of itself; this sets
  the values in φ0 and the criterion.
- A subset of the parameters gives the submatrix, a window start of one bin the full-grid matrix,
  and a zero step a zero row and column, bit for bit.
- Carrier crossing. A source whose carrier sits on a half-bin boundary in one segment, placed with
  the segment-table diagnostic so that the f0 stencil moves it across the bin there (checked), and
  the same source a quarter bin away: every diagonal ratio lies in (0.5, 2).
- A source whose reception chirp changes sign within the fixture, found by a scan in longitude,
  matches the reference to 1e-13.
- Declared parameters. Shifting A, φ0, ι or ψ leaves the effective frequencies, chirp rates and
  band starts of every reception state unchanged, bit for bit; shifting f0, ḟ, f̈, λ or β moves
  the effective frequencies.
- Arguments: an index outside [0, 9) raises `ValueError` in the comp and in the entry point, and so
  do rows per pass other than 4 or 8.

The wiring test compares `information_matrix` with `gb_dffr_information_matrix` called with the
expected arguments, bit for bit, in both of its settings. `DFFRGPUFisherTest` compares the CUDA
Fisher with the reference, with the CPU, and the 4-row with the 8-row tables, within a provisional
1e-7 of sqrt(F_ii F_jj); it is skipped without a GPU. `test_information_matrix_raises` is removed.

**`GBGPU/tests/test_stft_gb.py`.** `test_exact_amplitude_and_phase_entries` checks the same
identities for the existing response, with (h|h) from its astro track, the chirp its Fisher
differences. It passes: F_φ0φ0 equals (h|h) to 1.4e-6, within the criterion of 1e-5; the TDI
combination of the existing response enlarges the rounding of the phase step.

### Regression tests

**Golden outputs of the existing response against step 5** (parts `golden` and `golden_l1`, swap
kernels included). Criterion: bit identity against the step 5 captures, and between two captures
in separate processes. Result: 144 of 144 and 26 of 26 arrays bit-identical to step 5, and the
repeats bit-identical, pass. Files: `$OUT/regression/step4c/golden_cpu{,_l1}_vs_step5.txt` and
`golden_cpu{,_l1}_repeat_compare.txt`.

**DFFR through the comp against step 5, and the first capture of its Fisher matrix** (parts
`golden_dffr` and `golden_dffr_l1`). The captures now hold the DFFR Fisher matrices of four
sources per configuration: defaults, two-point stencil, the subset f0, ḟ, λ, β, and the envelope
forced on. DFFR has no TDI-track variant. Criterion: the arrays of step 5 bit-identical, and the
repeats bit-identical. Result: 42 of 42 and 21 of 21 arrays bit-identical to step 5, and the
repeats 50 of 50 and 25 of 25 bit-identical, pass. The 8 and 4 Fisher arrays are the reference of
the DFFR Fisher for step 6 (`$OUT/regression/step4c/golden_dffr_cpu{,_l1}.npz`).

**Exchange with standalone dffr** (part `exchange`). Criteria of step 5. Result: 14 outputs within
the gate of 1e-13 of the peak of the dffr array, and 29 of 29 entries of the LAT capture
bit-identical to step 5, pass (`exchange_compare_cpu.txt`, `exchange_lat_cpu_vs_step5.txt`).

**Steps of the DFFR Fisher matrix** (part `fd_steps`). `finite_difference_steps.py --all-sources`
evaluates the Fisher matrices of all 24 sources of the three XYZ production fixtures, at 1, 10,
100 and 1000 times the default steps (the shared table, with f0 ten times and ḟ a hundred times
wider), for both responses. `--plateau` prints per parameter the largest change between a factor
and the tenfold one, relative to sqrt(F_ii F_jj), over the sources and the parameter's row,
outside f̈. Criterion (author's decision of 2026-10-19): at most 1e-3 at the steps in use. Files:
`fd_steps_{dffr,tdi_on_the_fly}_cpu.npz`, `plateau_{dffr,tdi_on_the_fly}_cpu.txt`.

Largest change between the default and the tenfold steps:

| Fixture | DFFR | Existing response | Existing response, TDI track |
| --- | --- | --- | --- |
| `esa_tdi1_const` | 1.1e-5 | 3.2e-5 | 3.5e-5 |
| `esa_tdi2_envelope` | 1.1e-5 | 7.5e-5 | 9.1e-5 |
| `l1_tdi2_envelope` | 8.8e-6 | 1.6e-3 (β) | 2.0e-3 (β) |
| f̈ diagonal, recorded only | 5.7e-3 | 1.8e-2 to 4.3e-2 | 0.23 to 0.99 |

- Result: the DFFR Fisher lies on the plateau at the steps in use, by two orders of magnitude, on
  every fixture: pass. Its steps stay those of the existing response, and
  `_information_matrix_steps` has no branch for DFFR. The changes fall by a factor of 6 to 10 per
  decade of step, as rounding does.
- Observation: at its own steps, the existing response's Fisher on the L1 fixture changes by up
  to 1.6e-3 in β and 4.3e-4 in λ, above the criterion. Noted for the validation of the Fisher
  matrix, not changed.

**DFFR against the existing Fisher matrix** (part `fd_steps`, `fisher_against_existing.py`), on
the same captures at the default steps, 8 parameters. No threshold: the two responses model the
signal differently.

| Fixture | Largest / median over sources of max \|ΔF\| / sqrt(F_ii F_jj) | Diagonal ratio | Largest change of F_ij / sqrt(F_ii F_jj) | Condition numbers | Marginal standard deviation ratio, median (range) |
| --- | --- | --- | --- | --- | --- |
| `esa_tdi1_const` | 1.5e-2 / 5.4e-4 | 0.985 to 1.003 | 1.4e-3 | 6.2e3 to 3.8e10 | 1.014 (0.99 to 3.7) |
| `esa_tdi2_envelope` | 1.5e-2 / 5.7e-4 | 0.985 to 1.003 | 1.4e-3 | 6.3e3 to 3.8e10 | 1.022 (0.99 to 7.5) |
| `l1_tdi2_envelope` | 4.8e-3 / 7.7e-4 | 0.996 to 1.005 | 1.1e-3 | 4.8e3 to 3.5e10 | 1.011 (0.99 to 7.2) |

- The largest differences belong to sources 0 to 3, which the fixture places next to the edges
  of their store windows (one and two bins inside), so that 12 of the 21 bins of their band lie
  in the window. There they are a common scale of the source's whole matrix: its eight diagonal
  ratios agree to 0.1 % (0.986 for the largest), so they come from (h|h). DFFR's (h|h) agrees with
  that of the existing response's production template, on the TDI-phase track, to 0.99992 to
  1.00037, and with that of the astro-track template its Fisher differences to 0.986 to 1.001 on
  ESA orbits and 0.998 to 1.005 on L1 orbits (`$OUT/regression/step4c/norm_tracks_quick.py`).
  The astro track has no Doppler shift in its carrier, so at a window edge it puts another part
  of the band inside the window.
- For the 20 sources inside their windows: largest 2.5e-3 to 3.5e-3, median 4.2e-4 to 5.5e-4, and
  diagonal ratios 0.996 to 1.003. The steps leave each Fisher's own noise at most 1e-4 on these
  fixtures, except the sky entries of the existing response on L1, so these are differences of
  the models.
- The marginal standard deviations from the inverses agree to a median of 1 to 2 %. Their range
  reaches a factor of 7 for sources whose normalised Fisher matrix has a condition number of up
  to 4e10 over 32 days; there the inverse turns small differences of the entries into large ones.
- On the test fixture of `test_stft_gb_dffr.py`, envelope off: the entries agree to 2.7e-3 and the
  diagonals to 0.27 %, with condition numbers of 1.6e5 and of 3e7 to 4e7 for the two sources
  (`$OUT/regression/step4c/dffr_vs_existing_fisher_quick.py`, `fisher_conditioning_quick.py`).

**Test suites** (part `tests`). Criterion: every test case of step 5 keeps its outcome. Result:
pass. Files: `gbgpu_tests_cpu.{log,xml}`, `lat_tests_cpu.log`.

- GBGPU: the 71 cases of step 5 keep their outcomes, except `test_information_matrix_raises`,
  which is removed. 11 cases are new: 10 pass, and the CUDA Fisher test is skipped. In all: 68
  passed and 11 skipped, plus the failure and the error of step 5, the two cases that need a GPU.
- LAT: 302 tests. The 299 that pair by name with step 5 keep their outcomes. In all: 2 failures
  and 14 errors, 63 skips, as in step 5.

**GPU resources, without a GPU** (part `resources`). `kernel_resources.py`, now with a second
host program for the dynamic shared memory of the DFFR Fisher, which depends on the parameters it
differences. Reference: the step 5 module, from `$OUT/step5_env`. Files:
`kernel_resources_sm{80,90}.txt`.

| Kernel | Linked registers, sm_80 / sm_90 | Stack [B] | Spills | Static shared [B], sm_80 / sm_90 | Dynamic shared [B] | Blocks per SM by registers and by shared memory, A100 / H100 |
| --- | --- | --- | --- | --- | --- | --- |
| DFFR Fisher, 4 rows, 8 parameters | 170 / 174 | 1848 | none | 9696 / 10720 | 70 464 | 2, 2 / 2, 2 |
| DFFR Fisher, 8 rows, 8 parameters | 170 / 174 | 1848 | none | 10016 / 11040 | 140 640 | 2, 1 / 2, 1 |
| DFFR Fisher, 4 rows, 9 parameters | 170 / 174 | 1848 | none | 9696 / 10720 | 81 280 | 2, 1 / 2, 2 |
| DFFR Fisher, 8 rows, 9 parameters | 170 / 174 | 1848 | none | 10016 / 11040 | 162 272 | 2, 0 / 2, 1 |
| Existing Fisher, unchanged | 244 / 222 | 14 048 | none | 9424 / 10448 | none | 2, 16 / 2, 22 |

- 8 rows of 9 parameters do not fit an A100 block, and that launch raises; `STFTGBComputations`
  uses 4 rows.
- The existing kernels keep their resources of step 5. The DFFR kernels keep their linked
  registers and blocks per SM, except the fill kernel, with one register fewer (155); the own
  registers of the two-template DFFR kernels fall by 2 to 4, from the split of the pixel
  functions.

### Benchmarks

**Timing gate of the DFFR Fisher matrix on the CPU** (parts `fisher_timing` and
`fisher_timing_l1`), alone in the allocation. The Fisher of both responses on all 9 parameters and
on the 8 of the global fit, at batches 1, 8 and 64, three runs of each response in turn, on the
ESA and L1 production fixtures (32 one-day segments). Criterion (author's decision): DFFR no
slower than the existing response at every point. Files:
`$OUT/benchmarks/step4c/cpu_{esa,l1}_fisher_{existing,dffr}_run{1,2,3}.json`,
`compare_cpu_{esa,l1}_fisher_dffr_vs_existing.txt`.

| Fixture, parameters | Existing, batch 1 / 8 / 64 [ms] | DFFR, batch 1 / 8 / 64 [ms] | DFFR / existing |
| --- | --- | --- | --- |
| ESA, 9 | 42.58 / 363.32 / 3075.62 | 35.62 / 355.78 / 3473.96 | 0.837 / 0.979 / 1.130 |
| ESA, 8 | 36.40 / 307.62 / 2576.70 | 29.89 / 297.95 / 2900.01 | 0.821 / 0.969 / 1.125 |
| L1, 9 | 42.44 / 362.36 / 3067.56 | 35.58 / 354.92 / 3469.78 | 0.838 / 0.979 / 1.131 |
| L1, 8 | 36.15 / 306.38 / 2573.24 | 29.84 / 297.23 / 2895.88 | 0.825 / 0.970 / 1.125 |

Result: fail, by 13 % at batch 64 on both fixtures.

- Why the ratio grows with the batch. The DFFR Fisher spends nearly all its time in windowed
  transforms, 63 per pixel on 9 parameters (21 sets of three reception states); the existing
  Fisher spends most of its time sampling the TDI, 36 times per segment, whatever the pixels.
  Sources 0 to 3 sit at the edges of their windows, with 12 of their 21 bins inside. One source
  at a time (`$OUT/regression/step4c/fisher_cost_quick.py`), the DFFR Fisher takes 35.5 ms for
  these four and about 56 ms for most others, the existing Fisher 41 to 50 ms. Batch 1 holds only
  source 0.
- Compilation. GBGPU's CPU module compiles with `-Os` (plan, Risks and observations). Two variants
  were built in `$OUT/regression/step4c/o3_variant/` and loaded in place of the installed module,
  without changing it (`build_o3_variant.sh`, `build_full_o3.sh`, `run_with_overlay.py`):
  - `gb_dffr.cu` alone at `-O3`: every DFFR method is 2.3 to 2.9 times faster. At batch 64,
    `get_ll_stft` takes 0.42 of its time with `-Os`, the Fisher 0.35. The DFFR outputs change at
    rounding level: none of the 50 arrays of `golden_dffr` is bit-identical; templates and inner
    products stay within 8e-15 of their peaks, gradients within 6e-11.
  - The whole module at `-O3`, both responses in turn: the DFFR Fisher takes 0.30 to 0.42 of the
    time of the existing one (9 parameters: 12.7 / 125 / 1207 ms against 40.8 / 343 / 2872 ms).
    The existing Fisher gains 4 to 7 % against its `-Os` runs of the same day.
- Decision of the author (2026-10-19): both results recorded, the build unchanged in this part.
  The `-Os` of the CPU module is an open item, and the A100 gate of step 6 remains the gate for
  production.

**The existing response against the step 5 code** (parts `existing_timing` and
`existing_timing_l1`), alone, the two environments in turn, three runs each, every method but the
swap kernels. Criterion of step 3: each point within the larger run-to-run spread of the two sets.
Files: `compare_cpu_{esa,l1}_vs_step5_env.txt`.

| Fixture | Ratio, all points | Median | Within noise |
| --- | --- | --- | --- |
| ESA | 0.996 to 1.013 | 1.003 | 14 of 15 |
| `L1Orbits` | 0.995 to 1.007 | 0.999 | 13 of 15 |

The three points outside the spread differ by at most 0.7 %, the drift between runs of identical
code seen in steps 3 to 5. Result: pass; part 4c leaves the timings of the existing response, whose
compiled code it does not change, as they were.

### Validation runs

None in this part.

### Pending on the GPU

On a GPU node, from the shared environment, after the step 5 GPU runs:
`PYTHON=$PYTHON BASE_ENV=$OUT/step5_env DFFR_PYTHON=<Python of the dffr environment> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step4c.sh --device gpu $OUT environment golden golden_l1 golden_dffr golden_dffr_l1 exchange tests fd_steps fisher_timing fisher_timing_l1 existing_timing existing_timing_l1`.
This runs the CUDA Fisher test and the first GPU captures and timings of the DFFR Fisher.

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
