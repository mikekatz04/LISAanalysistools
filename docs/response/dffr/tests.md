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
the square root of F_ii F_jj, is largest in the row and column of f̈ (fddot): up to 8e-6 on the
four-day equal-arm fixtures (`rect`, `tukey_mid`) and up to 1e-7 on the 32-day ESA fixtures. All
other entries change by at most 1.5e-8 and 4e-10 of their scale. The f̈ derivative is a finite
difference of templates that f̈ barely changes over four days (its phase grows as T^3), so
changes at the level of rounding in the templates are enlarged; the ratio of the two cases, about
160, is of the order of (32 / 4)^3. On the equal-arm fixtures the f0 and ḟ entries change by
1e-8, the size of the removed truncation error. Once the author has reviewed these changes, the
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

From the shared environment, on a GPU node:
`PYTHON=$PYTHON BASE_ENV=$OUT/base_env MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step1.sh --device gpu $OUT environment unit golden golden_l1 dffr l1 tests benchmarks benchmarks_l1`,
after the GPU baseline of step 0, so that `golden_gpu*.npz` of step 0 exist for the comparison.

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
