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
- GPU runs use one full A100 (A100-PCIE-40GB) on `wn-gb-02` of partition `gpu_a100_22c`, in
  Slurm job 41556413 (22 CPUs, driver 615.71.10), from 2026-10-19. The node's other A100 runs
  another user's job. Every environment holds CUDA toolkit 13.2.2 (nvcc 13.2.86), CuPy 14.2.0
  (`cupy-cuda13x`) and JAX 0.11.2. The A100 MIG slices of the first plan were dropped on
  2026-10-19 (`plan.md`, step 6).
- On the GPU, every part of a step driver runs as a call of its own, from a chain script in
  `$OUT/regression/<step>/`: a comparison that finds a differing array exits with status 1, and
  under the drivers' `set -e` that would skip every later part.

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

**Golden outputs on the GPU** (parts `golden` and `golden_l1`, 2026-10-19, the full A100 of the
conventions above). Every GPU node of SURF Spider was down from 2026-10-16 14:56 to 2026-10-19
about 17:00, and step 1 had rebuilt the shared environment, so these captures come from the frozen
environment of the step 0 code (LAT 71b3f351, GBGPU a156dc3), which reproduces both CPU baselines
above bit for bit (step 1, "Frozen environment of the step 0 code"). Each capture ran twice, in
separate processes.

- Six single-grid fixtures: 141 of the 144 arrays are bit-identical between the two captures. The
  other three are the `fill` outputs of `rect`, `esa_tdi2_envelope` and `esa_tdi1_const`; they
  differ by at most 3.4e-17 of their peaks, or 8 to 64 units in the last place of small elements.
- `l1_tdi2_envelope`: 25 of the 26 arrays are bit-identical; `fill` differs by 1.1e-16 of its
  peak.

`fill_global_stft` adds into its template rows with `atomicAdd` (`lat_stft_kernels.hh:815`), so
where sources share pixels the order of the additions changes from run to run. On the GPU, `fill`
is therefore reproducible to rounding only; every other output is reproducible bit for bit. In
the steps that ask bit identity (3, 4, 5 and part 4c), `fill` on the GPU must agree with the
previous step to 1e-15 of its peak, about ten times the largest run-to-run difference here
(author's decision of 2026-10-19); every other output must stay bit-identical. Baseline: `$OUT/regression/step0/golden_gpu.npz`
and `golden_gpu_l1.npz`; comparisons in `golden_gpu_repeat_compare.txt` and
`golden_gpu_l1_repeat_compare.txt`.

Command, one driver call per part (`$OUT/regression/step0/step0_gpu_chain.sh`), with
`BASE=$OUT/base_env`, for each part of `environment golden golden_l1 benchmarks benchmarks_l1
tests`:
`PYTHON=$BASE/.venv/bin/python LAT_ROOT=$BASE/lisa-analysis-tools MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_baseline.sh --device gpu $OUT <part>`.

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

**GPU** (part `benchmarks`, 2026-10-19, the full A100), fixture `esa_tdi2_envelope` with a year
of one-day segments (365) and batch sizes from 1 to 10 000, otherwise as on the CPU: one warm-up
call and five timed calls per point, each closed by a device synchronisation, three runs in
separate processes. Files: `$OUT/benchmarks/step0/gpu_esa_run{1,2,3}.json`.

| Method | Batch 1 [ms] | Batch 10 [ms] | Batch 100 [ms] | Batch 1000 [ms] | Batch 10 000 [ms] | Per source at 10 000 [µs] | Largest spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 15.34 | 10.00 | 10.20 | 66.51 | 660.29 | 66.03 | 0.74 % |
| `get_swap_ll_stft` | 22.87 | 34.75 | 36.55 | 199.19 | 1883.38 | 188.34 | 60.18 % |
| `get_fstat_ll_stft` | 131.18 | 158.00 | 165.06 | 1194.83 | 11292.06 | 1129.21 | 1.07 % |
| `fill_global_stft` | 8.18 | 9.70 | 9.97 | 62.20 | 605.75 | 60.58 | 1.05 % |
| `get_ll_grad_stft` | 139.13 | 162.69 | 166.15 | 1074.38 | 10121.46 | 1012.15 | 1.09 % |
| `get_swap_ll_grad_stft` | 559.37 | 1067.80 | 1109.35 | 5996.97 | 55774.14 | 5577.41 | 1.72 % |
| `information_matrix` | 80.34 | 87.76 | 94.63 | 610.56 | 6205.82 | 620.58 | 1.61 % |

The largest spread of every method lies at or below 1.72 %, except one point: `get_swap_ll_stft`
at batch 1, whose three run medians are 16.6, 30.3 and 22.9 ms. Up to 100 sources a call costs
about the same as for one, so the device is not yet filled; from 1000 sources on the time grows
in proportion to the batch.

**GPU on `L1Orbits`** (part `benchmarks_l1`): fixture `l1_tdi2_envelope`, otherwise as above.
Files: `$OUT/benchmarks/step0/gpu_l1_run{1,2,3}.json`.

| Method | Batch 1 [ms] | Batch 10 [ms] | Batch 100 [ms] | Batch 1000 [ms] | Batch 10 000 [ms] | Per source at 10 000 [µs] | Largest spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `get_ll_stft` | 15.22 | 10.05 | 10.17 | 66.25 | 666.07 | 66.61 | 0.52 % |
| `get_swap_ll_stft` | 23.06 | 35.01 | 36.43 | 200.34 | 1903.03 | 190.30 | 60.03 % |
| `get_fstat_ll_stft` | 132.37 | 159.47 | 165.02 | 1199.19 | 11379.38 | 1137.94 | 0.27 % |
| `fill_global_stft` | 8.13 | 9.71 | 9.85 | 61.72 | 602.85 | 60.28 | 0.43 % |
| `get_ll_grad_stft` | 138.12 | 163.08 | 166.28 | 1080.45 | 10197.67 | 1019.77 | 0.31 % |
| `get_swap_ll_grad_stft` | 561.06 | 1075.64 | 1115.64 | 6075.64 | 56384.48 | 5638.45 | 0.59 % |
| `information_matrix` | 81.24 | 88.96 | 96.02 | 596.85 | 6170.26 | 617.03 | 0.65 % |

Again only `get_swap_ll_stft` at batch 1 spreads widely (60.03 %). At 10 000 sources every method
lies within 1.1 % of the ESA orbits (ratios 0.994 to 1.011).

### Test suites

Both suites ran on the CPU node through the driver parts `gbgpu_tests` and `lat_tests`, with the
two uncommitted changes above in place. Neither run left files in the repositories
(`git_status_after_*_cpu.txt`). Failures that already exist at the base commits are recorded
here, not fixed. The GPU runs follow the CPU tables.

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

**On the GPU** (part `tests`, 2026-10-19, the full A100, from `$OUT/base_env`). These outcomes are
the GPU baseline that later steps must keep. Neither run left files in the frozen clones
(`git_status_after_*_gpu.txt`).

GBGPU (109 s): 33 passed, 4 failed, 6 errors. Log and JUnit file:
`$OUT/regression/step0/gbgpu_tests_gpu.{log,xml}`.

| Outcome | Tests | Cause |
| --- | --- | --- |
| passed (33) | `test_stft_gb` (23), `test_intra_split_index` (6), `test_stft_gb_accuracy` (4) | |
| error (6) | every test of `test_stft_gb_crossdomain` | its orbits are built on the CPU backend and receive CuPy arrays (`OrbitsWrapCPU.__init__`, incompatible arguments); on the CPU node all 6 pass |
| failed (4) | `test_gbgpu` (`test_circ`, `test_likelihood`, `test_information_matrix`), `test_thirdbody::test_ecc_third_body` | out of date against the GBGPU class: no `XYZf` or `inject_signal`, and `information_matrix` asks for `psd` |

LAT (`python -m unittest discover -v`, 289 s): 271 tests, 223 passed, 28 skipped, 7 failures,
13 errors. Log: `$OUT/regression/step0/lat_tests_gpu.log`.

| Outcome | Tests | Cause |
| --- | --- | --- |
| error (2) | `test_detector` (`test_get_vel`, `test_orbits`) | they ask for the CUDA 12 backend by name; the environment has the CUDA 13 backend |
| error (8) | `test_sensitivity` (seven `test_sensitivity_matrix_*`), `test_quintic_response::test_quintic_batched_heterogeneous_shift_equals_per_source` | a CuPy array reaches NumPy (`Implicit conversion to a NumPy array is not allowed`) |
| failure (3) | `test_response_alignment` (`test_batched_min_trim_preserves_leading_signal`, `test_off_grid_data_t0_no_one_sample_desync`, `test_times_land_on_data_grid`) | assertion errors; skipped on the CPU node |
| failure (2) | `test_domain_backend_threading` (`test_transform_target_backend_mismatch_raises`, `test_wdm_to_stft_layered_chain`) | they expect the backend name `gpu` where the backend reports `cuda13x`, and a `ValueError` that is not raised |
| error (2), failure (2), error (1) | the five cases of the CPU table above | as on the CPU |

| Skipped | Reason |
| --- | --- |
| 14 | two or more CUDA devices needed; the job holds one |
| 9 | developer checks, run only with `STFT_MULTI_GPU_CHECK=1` |
| 3 | bbhx, phentax or mojito not importable |
| 1 | needs a fully initialised band buffer on a GPU host |
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
the driver sets the thread counts and `JAX_PLATFORMS` as in step 0. The GPU runs (2026-10-19)
repeat the same parts with `--device gpu` ("On the GPU" below).

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
| CUDA against CPU | absolute difference of every field | rounding bound | 0 in all 11 fields: on the A100 the CUDA table equals the CPU table bit for bit (2026-10-19) | pass |

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

### On the GPU

Run on 2026-10-19 from 20:39 to 21:56 on the full A100 of the conventions, from the frozen
environment of the step 1 code, `$OUT/step1_env`, after the GPU baseline of step 0. Every part
runs as a call of its own (`$OUT/regression/gpu_chain_steps1_4c.sh`), with `S1=$OUT/step1_env`,
for each part of `environment unit golden golden_l1 dffr l1 tests benchmarks benchmarks_l1`:
`PYTHON=$S1/.venv/bin/python BASE_ENV=$OUT/base_env DFFR_REPOSITORY=<dffr clone> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" $S1/lisa-analysis-tools/scripts/dffr/run_step1.sh --device gpu $OUT <part>`.
The driver compares its captures with step 0 only after the repeat check, which on the GPU always
finds `fill` differing, so the part ends first; `$OUT/regression/gpu_step_comparisons.sh` makes
these comparisons with the step's own scripts, for this and every later step.

- **Unit tests** (part `unit`): 13 of 13 pass, the six CUDA cases included. The CUDA table equals
  the CPU table bit for bit in all 11 fields, and every other number equals the CPU's
  (`unit_numbers_gpu.txt`).
- **Golden outputs.** The two captures are bit-identical except `fill`, as in step 0 (143 of 144
  and 25 of 26; `fill` differs by 3.4e-17 and 3.4e-18 of its peak). Against the step 0 GPU
  captures the outputs change as on the CPU: the same arrays stay bit-identical (all of `astro`,
  and the plain Fisher estimators everywhere), and the largest changes per kind of output are of
  the same size (files `golden_gpu{,_l1}_vs_step0.txt`; table by
  `$OUT/regression/gpu_review/step_changes.py`):

| Fixture | Unchanged arrays | Inner products, F-statistic, templates | Gradients | Fisher, TDI track | Fisher, other estimators |
| --- | --- | --- | --- | --- | --- |
| `astro` | 22 of 22 | 0 | 0 | 0 | 0 |
| `rect` | 3 of 22 | 4.7e-11 | 1.8e-9 | 1.5e-8 | 0 |
| `tukey_mid` | 3 of 22 | 2.2e-14 | 1.1e-9 | 2.6e-9 | 0 |
| `esa_tdi2_envelope` | 4 of 26 | 4.4e-11 | 1.7e-10 | 1.7e-11 | 0 |
| `esa_tdi1_const` | 4 of 26 | 4.4e-11 | 5.0e-10 | 7.0e-12 | 0 |
| `esa_aet_diag` | 4 of 26 | 3.3e-11 | 1.9e-10 | 3.3e-11 | 0 |
| `l1_tdi2_envelope` | 4 of 26 | 5.9e-11 | 3.0e-10 | 2.3e-11 | 0 |

  Both criteria of this step hold on the GPU. Pass.
- **Exchange with standalone dffr** (part `dffr`) **and the Mojito orbits** (part `l1`): every
  number equals the CPU's, since the CUDA table equals the CPU table
  (`kinematics_esa_compare_gpu.txt`, `kinematics_l1_gpu.txt`). Pass.
- **Test suites** (part `tests`): every test keeps its step 0 outcome on the GPU, and the 13 new
  tests pass (`$OUT/regression/gpu_review/test_outcomes.py $OUT/regression/step0
  $OUT/regression/step1`). Pass.
- **Benchmarks** (parts `benchmarks` and `benchmarks_l1`): `$OUT/benchmarks/step1/gpu_{esa,l1}_run{1,2,3}.json`.
  The timings of different steps are compared through the interleaved round ("GPU timings of the
  steps", step 6).

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
runs are recorded below ("On the GPU").

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
| CUDA against CPU | absolute difference for C, S and g', relative for f, g and f'; the phase kernel as a fraction of dffr's normalisation | 8 roundings; 1e-14, ten times the largest A100 value (the provisional 1e-13 was set to this on 2026-10-20) | on the A100: C 2.2e-16, S 1.1e-16; f, g, f', g' bit-identical; phase kernel 9.0e-17, moment 3.0e-16 | pass, with the bound of 1e-14 too |

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

### On the GPU

Run on 2026-10-19 from 21:56 to 23:13 on the full A100, from the frozen environment of the step 2
code, `$OUT/step2_env`, after the step 1 GPU runs, one call per part
(`$OUT/regression/gpu_chain_steps1_4c.sh`), with `S2=$OUT/step2_env`, for each part of
`environment unit golden golden_l1 fd_steps fd_steps_l1 tests benchmarks benchmarks_l1`:
`PYTHON=$S2/.venv/bin/python BASE_ENV=$OUT/step1_env MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" $S2/lisa-analysis-tools/scripts/dffr/run_step2.sh --device gpu $OUT <part>`.
The comparisons with step 1 come from `$OUT/regression/gpu_step_comparisons.sh` (step 1, "On the
GPU").

- **Unit tests** (part `unit`): 9 of 9 pass, the five CUDA cases included. CUDA against CPU, on
  the A100: C 2.2e-16 and S 1.1e-16 absolute, f, g, f' and g' bit-identical, the phase kernel and
  its moment 9.0e-17 and 3.0e-16 of dffr's normalisation (table above; `unit_numbers_gpu.txt`).
- **Golden outputs.** The two captures are bit-identical except `fill` (141 of 144 and 25 of 26;
  `fill` at most 1.1e-16 of its peak). Against the step 1 GPU captures, with the swap kernels,
  which the GPU keeps: every output computed directly from templates changes by at most 7.2e-14
  of its peak (`astro`; 4.9e-14 `rect`, at most 1.4e-14 elsewhere), within the gate of 1e-12.
  Pass. The 54 arrays above the gate are all gradients and Fisher matrices, as on the CPU
  (`golden_gpu{,_l1}_vs_step1.txt`).
- **Gradients and Fisher matrices** (parts `fd_steps` and `fd_steps_l1`, the criterion of the CPU
  runs above): 62 of 63 gradient components lie within the larger noise of the two codes
  (largest ratio change / noise 1.43, the amplitude component of `esa_aet_diag`), and 4519 of
  4536 Fisher entries (largest ratio 2.14). The entries above the noise are again terms in
  amplitude, ι and ψ that both codes compute almost exactly, with changes of 1e-12 to 1e-9 of
  their scale. Accepted, as on the CPU (CPU: 62 of 63 and 4526 of 4536, ratios up to 2.47).
  Files: `fd_steps_step{1,2}_gpu{,_l1}.npz`, `fd_steps_step2_vs_step1_gpu{,_l1}.txt`.
- **Swap gradients.** `finite_difference_steps.py` does not evaluate them, so the noise criterion
  above does not cover them, although the CPU record said the GPU runs would. Per component,
  relative to its largest value over the sources, they change as much as the `get_ll` gradient:
  f0 1.5e-5 (`get_ll` gradient 1.0e-5), ḟ 4.5e-5 (7.5e-5), f̈ 1.4e-3 (5.4e-4), φ0 2.0e-6
  (6.2e-6), angles at most 1.6e-9, amplitude 1.1e-12, all far below the noise this step measured
  at the default steps (f̈ 0.6 to 13, ḟ 0.2 to 1.8, f0 1e-2 to 0.25). Recorded.
- **Test suites** (part `tests`): every test keeps its step 1 outcome on the GPU, and the 9 new
  tests pass. Pass.
- **Benchmarks**: `$OUT/benchmarks/step2/gpu_{esa,l1}_run{1,2,3}.json`; compared between steps
  through the interleaved round (step 6).

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
(`$OUT/regression/step3/step3_cpu_chain.sh`). The GPU runs are recorded below ("On the GPU").

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

### On the GPU

Run on 2026-10-19 from 23:13 to 00:30 on the full A100, from the frozen environment of the step 3
code (`$OUT/step3_env`, step 4 below), with its own scripts, one call per part, for each part of
`environment golden golden_l1 tests benchmarks benchmarks_l1`, from
`$OUT/step3_env/lisa-analysis-tools`:
`PYTHON=$OUT/step3_env/.venv/bin/python BASE_ENV=$OUT/step2_env MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step3.sh --device gpu $OUT <part>`.
On the GPU the captures and the benchmarks keep the swap kernels, and the reference is the step 2
GPU capture itself.

- **Golden outputs.** The steps 0 to 2 showed that on the GPU `fill` is reproducible to rounding
  only, so the gate of this step is bit identity of every other output and 1e-15 of its peak for
  `fill` (step 0). Against step 2: 141 of 144 arrays bit-identical, the other three the `fill`
  outputs of `astro`, `esa_tdi1_const` and `esa_tdi2_envelope`, at most 8.6e-17 of their peaks;
  `l1_tdi2_envelope` 25 of 26, `fill` 1.4e-17. Pass. The two captures of this step differ only in
  `fill` too (at most 3.4e-17), and on `L1Orbits` not at all
  (`golden_gpu{,_l1}_vs_step2.txt`, `golden_gpu{,_l1}_repeat_compare.txt`).
- **Test suites** (part `tests`): every test keeps its step 2 outcome on the GPU. Pass.
- **Benchmarks**: `$OUT/benchmarks/step3/gpu_{esa,l1}_run{1,2,3}.json`. The timing criterion of
  this step reads from the interleaved round (step 6): 27 of 35 points within the noise on both
  fixtures, the other 8 within 0.7 %, accepted by the author as drift. Pass.

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
(`$OUT/regression/step4/step4_cpu_chain.sh`). The GPU runs are recorded below ("On the GPU").

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

**`tests/test_dffr.py` (LAT; it left LAT in step 6, item 6, with its diagnostics and
`dffr_unit_numbers.py`).** The source-independent DFFR stages, through three diagnostic
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
without a GPU: 5 tests, CUDA against CPU among them. On the A100 all 5 pass ("On the GPU" below);
the provisional bound of 1e-13 of CUDA against CPU became 1e-14 on 2026-10-20, and the CUDA class
passes with it on the A100, at 0.004 to 0.10 of the bound (`$OUT/regression/gpu_review/`).

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
16-row tiles against the 32-row tiles, with a provisional bound of 1e-12 of each peak. On the A100
7 of the 9 pass and two fail ("On the GPU" below).

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

### On the GPU

Run on 2026-10-20 from 00:30 to 01:53 on the full A100, from the frozen environment of the step 4
code, `$OUT/step4_env` (step 5 records how it was built and checked), after the step 3 GPU runs,
one call per part, for each part of `environment golden golden_l1 tests unit_numbers exchange
benchmarks benchmarks_l1`, from `$OUT/step4_env/lisa-analysis-tools`:
`PYTHON=$OUT/step4_env/.venv/bin/python BASE_ENV=$OUT/step3_env DFFR_PYTHON=<Python of the dffr CUDA 13 environment> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step4.sh --device gpu $OUT <part>`.
The exchange calls standalone dffr on its CUDA 13 backend (step 6, "Standalone dffr against DFFR
in `STFTGBComputations`"), in `.venv-gpu13` of the dffr clone.

- **Existing response against step 3** (parts `golden` and `golden_l1`): every output
  bit-identical except `fill` (141 of 144; `fill` at most 3.4e-17 of its peak), and 26 of 26 on
  `L1Orbits`; in this step the two captures of each fixture were bit-identical, `fill` included.
  Pass.
- **LAT tests** (`tests/test_dffr.py`): 10 of 10 pass, the 5 CUDA cases included. CUDA against CPU
  at most 3.8e-16 for the transforms and 1.0e-15 for the coefficients; every other number equals
  the CPU's (`unit_numbers_gpu.txt`). Pass.
- **GBGPU tests** (`tests/test_stft_gb_dffr.py`): 17 of the 19 new tests pass, among them CUDA
  against CPU and 16 against 32 rows. So the code that no CPU run reaches runs: the launch above
  48 kB (swap at 32 rows), the band-bin map at 32 rows and the device copy of the domain. Two
  CUDA cases fail; both assert a property that the CPU build has and the GPU build need not have.
  The author decided on 2026-10-20 to leave them failing and record them:
  - `test_compact_template_equals_fill_global` asks for bit equality of the compact template and
    the `fill_global` grid. On the A100, 375 of 1080 elements differ, by at most 3.8e-15 of their
    values. The two come from different kernels, which nvcc compiles separately; with one source,
    the additions into the grid are exact, so the difference is the rounding of two builds.
  - `test_swap_ll_grad_matches_host_differences` compares the swap gradient with host central
    differences of `get_swap_ll_stft` at rtol 1e-6. On the A100 only f̈ of the added source
    misses, by 5.2e-4. The test's f̈ step, 1e-28 Hz/s² over six days, changes the phase far less
    than rounding, so this component measures rounding: on the CPU the kernel and the host round
    alike, on the GPU the swap-gradient kernel and the swap kernel do not.
- **Every other test** keeps its step 3 outcome on the GPU.
- **Exchange with standalone dffr on the GPU** (part `exchange`): 14 of 14 outputs within the
  gate of 1e-13: compact templates 1.7e-14, `fill_global` 1.2e-14, (d|h) 5.0e-16, (h|h) 1.5e-16,
  the segment table identical except the TDI weights (2.7e-16) (`exchange_compare_gpu.txt`). Pass.
- **Benchmarks**: `$OUT/benchmarks/step4/gpu_{esa,l1}_run{1,2,3}.json`; compared between steps
  through the interleaved round (step 6).

## Step 5: wiring

Code: LAT `feat-dffr` at de526e1b and GBGPU `feat-dffr` at e6fb653, the committed step 4, with the
changes of step 5 uncommitted. Step 5 changes Python only: `gbcomps.py` in GBGPU, `recipe.py` in
LAT, tests and scripts. The editable builds of the step 4 code load Python from the source trees,
so nothing was rebuilt.

CPU node: `wn-la-01` (AMD EPYC 9754, Bergamo), Slurm job 41576984, 16 CPUs, 2026-10-19. The parts
ran inside that allocation as
`srun --jobid=<job> --overlap -n 1 -c 16 env PYTHON=$PYTHON BASE_ENV=$OUT/step4_env DFFR_PYTHON=<Python of the dffr environment> MOJITO_L1=<file> scripts/dffr/run_step5.sh --device cpu $OUT <part>`:
every part but the benchmarks side by side, then `benchmarks benchmarks_l1` alone
(`$OUT/regression/step5/step5_cpu_chain.sh`). The GPU runs are recorded below ("On the GPU").

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

### On the GPU

Run on 2026-10-20 from 01:53 to 03:39 on the full A100, from the frozen environment of the step 5
code, `$OUT/step5_env`, after the step 4 GPU runs, one call per part, for each part of
`environment golden golden_l1 golden_dffr golden_dffr_l1 tests exchange against_existing
envelope_noise benchmarks benchmarks_l1`:
`cd $OUT/step5_env/lisa-analysis-tools && PYTHON=$OUT/step5_env/.venv/bin/python BASE_ENV=$OUT/step4_env DFFR_PYTHON=<Python of the dffr CUDA 13 environment> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step5.sh --device gpu $OUT <part>`.

- **Existing response against step 4** (parts `golden` and `golden_l1`): every output
  bit-identical except `fill` (142 of 144 and 25 of 26; `fill` at most 3.4e-17 of its peak). Pass.
- **First GPU captures of DFFR through the comp** (parts `golden_dffr` and `golden_dffr_l1`, 42
  and 21 arrays): the two captures are bit-identical except `fill` (at most 1.2e-16 of its peak),
  the same reproducibility as the existing response. Against the CPU captures of this step:
  (d|h) and (h|h) at most 1.4e-15 of their peaks, the swap terms 8.0e-15 (the cross term), the
  F-statistic 4.1e-15, `fill` 2.3e-14, and gradients 1.1e-10 to 7.2e-10, the finite differences
  enlarging the rounding. These captures are the GPU baseline of DFFR for part 4c.
- **Test suites** (part `tests`): every test keeps its step 4 outcome on the GPU, the two failing
  CUDA cases of step 4 included; the 11 new tests pass, the CUDA wiring test among them, and
  `test_engine_calls_bind_to_STFTGBComputations_signatures` now passes, as on the CPU. Pass.
- **Exchange through the comp on the GPU** (part `exchange`): 14 of 14 outputs within the gate of
  1e-13, with the numbers of step 4. LAT's side equals step 4's in 28 of 29 entries; `fill_global`
  differs by 6.1e-17 of its peak, the rounding of its atomic additions. Pass.
- **DFFR against the existing response** (part `against_existing`): the same numbers as on the CPU
  to every printed digit: median mismatch 5.6e-10, largest 2.6e-8, phases within 2.2e-5 rad, norm
  ratios 0.99998 to 1.00008 (`against_existing_gpu.txt`).
- **Envelope noise** (part `envelope_noise`): with the envelope on, DFFR's gradient noise is
  unchanged in f0, ḟ, f̈, φ0, λ and β (ratio on / off 1.00 on all three fixtures), as on the CPU;
  the existing response's ratios lie between 0.81 and 1.40 (`envelope_noise_*_gpu.txt`).
- **Benchmarks**: `$OUT/benchmarks/step5/gpu_{esa,l1}{,_dffr}_run{1,2,3}.json`, both responses;
  compared between steps and between the responses through the interleaved round (step 6).

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
(`$OUT/regression/step4c/part3_chain.sh`). The GPU runs are recorded below ("On the GPU").

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
1e-7 of sqrt(F_ii F_jj); it is skipped without a GPU, and passes on the A100 ("On the GPU" below).
`test_information_matrix_raises` is removed.

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

### On the GPU

Run on 2026-10-20 from 03:39 to 06:35 on the full A100, from the shared environment (LAT
24b4bd8a, GBGPU 2d52f70), after the step 5 GPU runs, one call per part, for each part of
`environment golden golden_l1 golden_dffr golden_dffr_l1 exchange tests fd_steps fisher_timing
fisher_timing_l1 existing_timing existing_timing_l1`:
`PYTHON=$PYTHON BASE_ENV=$OUT/step5_env DFFR_PYTHON=<Python of the dffr CUDA 13 environment> MOJITO_L1=<file> BENCH_ARGS="--num-segments 365 --batch-sizes 1 10 100 1000 10000" scripts/dffr/run_step4c.sh --device gpu $OUT <part>`.

- **Existing response against step 5**: every output bit-identical except `fill` (143 of 144 and
  25 of 26; `fill` at most 4.2e-18 of its peak). Pass.
- **DFFR against step 5.** On the CPU this part leaves every DFFR output bit-identical; on the GPU
  it does not. 16 of the 42 arrays and 8 of the 21 on `L1Orbits` stay bit-identical: the outputs
  of `get_ll`, its gradients, and the swap terms of the added source. The others are `fill`
  (rounding of its atomic additions) and outputs of the two-template kernels: the swap terms of
  the removed source change by at most 2.7e-15 of their peaks (the cross term), the F-statistic
  by 3.9e-16, and the swap gradients by up to 1.1e-10.
  Per component the swap gradients change 3 to 5 orders of magnitude less than DFFR's own
  gradient noise (part `envelope_noise` of step 5) in f0, ḟ, f̈, φ0, λ, β and ι; in the amplitude
  and ψ the change reaches 2.5 times that noise, which there is itself rounding (1e-13 and 1e-9
  of their scale), as in step 2.

  The machine code for the A100 (`cuobjdump` and `nvdisasm` of the sm_80 images of both GBGPU
  modules) shows the cause. `DFFRTile::Template::values`, which nvcc keeps out of line in the
  two-template kernels, holds two more double-precision multiplies in this part, where the
  windowed sums were split before their final products. The kernels that inline it (`get_ll`,
  `fill_global`, the `get_ll` gradient) only order independent operations differently, which
  changes no rounding. The agreement of the GPU with the CPU is the same before and after: (d|h)
  and (h|h) within 1.4e-15 of their peaks, the swap cross term 8.0e-15 and 6.2e-15 (step 5 and
  this part), `fill` 2.3e-14, gradients 1.1e-10 to 7.2e-10. Files:
  `$OUT/regression/gpu_review/sass/`, `golden_dffr_gpu{,_l1}_vs_step5.txt`.

  The author decided on 2026-10-20 to accept the change: on the GPU these outputs are judged to
  rounding (direct outputs within 1e-14 of their peaks, gradients by their noise) instead of bit
  identity, because the occupancy runs of step 6 change this code again. Pass by that criterion.
- **First GPU captures of the DFFR Fisher** (8 and 4 arrays): the two captures are bit-identical.
  CUDA against CPU, per entry relative to sqrt(F_ii F_jj): at most 9.3e-6 on all nine parameters,
  where the f̈ entries carry their finite-difference noise, and 2.8e-8 on the subset f0, ḟ, λ, β.
- **Test suites** (part `tests`): every test keeps its step 5 outcome on the GPU, the two failing
  CUDA cases of step 4 included; the 11 new tests pass, among them `DFFRGPUFisherTest` (the CUDA
  Fisher against the reference, against the CPU, and 4 against 8 rows, within 1e-7 of
  sqrt(F_ii F_jj)). `test_information_matrix_raises` is removed. Pass.
- **Exchange** (part `exchange`): 14 of 14 outputs within the gate of 1e-13, the numbers of step
  4; LAT's side equals step 5's in 28 of 29 entries, `fill_global` 6.1e-17 of its peak. Pass.
- **Steps of the Fisher matrix** (part `fd_steps`): between the default and the tenfold steps the
  DFFR Fisher changes by at most 8.8e-6 of sqrt(F_ii F_jj) outside the row and column of f̈,
  against the criterion of 1e-3 (CPU 1.1e-5): on the plateau from the default steps. The
  existing Fisher changes by 2.0e-3 there (β), on the plateau from the tenfold steps; recorded,
  as on the CPU. DFFR against the existing Fisher: median over sources 5.7e-4 (ESA) and 7.7e-4
  (`L1Orbits`) of sqrt(F_ii F_jj), largest 1.5e-2 at the sources the fixture places at window
  edges (`plateau_*_gpu.txt`, `fisher_against_existing_gpu.txt`).
- **Timing gate of the DFFR Fisher on the A100** (parts `fisher_timing` and `fisher_timing_l1`;
  the two Fisher kernels in turn, three runs each, a year of one-day segments). Ratio of the time
  of the DFFR Fisher to that of the existing Fisher:

  | Batch | 9 parameters, ESA | 9 parameters, `L1Orbits` | 8 parameters, ESA | 8 parameters, `L1Orbits` |
  | --- | --- | --- | --- | --- |
  | 1 | 0.995 | 0.982 | 0.989 | 0.974 |
  | 10 | 0.931 | 0.927 | 0.942 | 0.939 |
  | 100 | 0.925 | 0.919 | 0.932 | 0.923 |
  | 1000 | 1.435 | 1.469 | 1.062 | 1.087 |
  | 10 000 | 1.336 | 1.343 | 0.987 | 0.994 |

  At 10 000 sources the existing Fisher takes 605 µs per source on nine parameters and 513 µs on
  eight, the DFFR Fisher 809 µs and 507 µs. The run-to-run spreads stay below 2.2 %. The gate,
  not slower than the existing Fisher, fails from 1000 sources on nine parameters, and at 1000
  sources on the eight parameters of the global fit. The GPU builds the stencil tables one at a
  time, which keeps 12 to 24 of a block's 128 threads busy during a build (Risks and observations
  of `plan.md`). The author decided on 2026-10-20 to optimise this in step 6 and to measure the
  gate again then. Files: `$OUT/benchmarks/step4c/gpu_{esa,l1}_fisher_{dffr,existing}_run*.json`,
  `compare_gpu_{esa,l1}_fisher_dffr_vs_existing.txt`.
- **Existing response against the step 5 build** (parts `existing_timing` and
  `existing_timing_l1`, in turn): 32 of 35 and 33 of 35 points within the run-to-run noise, the
  others within 0.5 %, median ratio 1.000 on both fixtures. Unchanged. Files:
  `compare_gpu_{esa,l1}_vs_step5_env.txt`.

## Step 6: validation

Started on 2026-10-19 with the comparison of standalone dffr and the integration on the A100.

### Unit tests

Added with the direct build of the DFFR Fisher tables (2026-10-20), in GBGPU
`tests/test_stft_gb_dffr.py`:

- `DFFRFisherTest.test_parity_on_one_set_of_states`: the Fisher matrix on A and φ0 alone against
  the reference built from `fill_global` templates, at the bound of the other parity tests
  (`FISHER_BOUND`), for both settings of the class. Both parameters leave the reception states
  unchanged, so their eight stencil templates share one set of states: the smallest layout, where
  the build holds one template at a time and enlarges the space that holds it.
- `DFFRGPUFisherTest.test_gpu_matches_reference_and_cpu` gains the same layout: the CUDA Fisher
  against the CPU one (`GPU_FISHER_BOUND`).

Both pass on the CPU and on the A100, with the code before and after the change (below).

### Regression tests

**GPU timings of the steps** (the interleaved round, author's decision of 2026-10-19; run on
2026-10-20 from 06:40 to 15:41 on the full A100). Each of the seven environments (steps 0 to 5 and
part 4c) ran its own `stft_benchmarks.py`: a year of one-day segments, batches 1 to 10 000, one
warm-up and five timed calls per point. The environments took turns, in three passes over both
orbit fixtures, and the environments of step 5 and part 4c also timed DFFR in each pass. Each
comparison is `compare_benchmarks.py` over the three runs: the ratio of the medians, and "within"
when it lies inside the larger run-to-run spread of the two sets (`$OUT/benchmarks/interleaved/`,
chain `$OUT/benchmarks/interleaved_chain.sh`).

| Comparison | ESA: median ratio, points within | `L1Orbits`: median ratio, points within | Outside the noise |
| --- | --- | --- | --- |
| Step 1 / step 0 | 1.003, 17 of 35 | 1.002, 18 of 35 | from 1000 sources on 1.1 % to 2.1 % slower, gradients the most: the five-point acceleration. No threshold in this step. |
| Step 2 / step 1 | 0.960, 2 of 35 | 0.955, 4 of 35 | 3 % to 6 % faster in every method but the Fisher matrix (0.975 to 1.00): the cheaper special functions of step 2, within the noise on the CPU. |
| Step 3 / step 2 | 1.000, 27 of 35 | 0.998, 27 of 35 | the other 8 points per fixture differ by at most 0.7 % (0.991 to 1.007), in both directions, against run-to-run spreads of 0.1 % to 0.4 %. |
| Step 4 / step 3 | 1.000, 33 of 35 | 1.000, 34 of 35 | at most 0.6 %. |
| Step 5 / step 4 | 0.999, 33 of 35 | 1.000, 35 of 35 | at most 0.5 %. |
| Part 4c / step 5 | 1.000, 35 of 35 | 1.000, 35 of 35 | none. |
| DFFR, part 4c / DFFR, step 5 | 0.999, 22 of 30 | 0.999, 25 of 30 | `get_ll_stft` at 10 000 sources 1.0 % slower on both fixtures; the rest calls of about 1 ms at batch 1 or within 0.6 %. |

The only larger ratios, 0.70 to 1.35, are `get_swap_ll_stft` at batch 1, whose run medians spread
by 14 % to 64 % in every environment.

- Step 3, whose criterion is the run-to-run spread per point, meets it at 27 of 35 points on both
  fixtures; the other 8 lie within 0.7 %. On the CPU the same criterion left 3 of 30 points
  outside, by at most 0.7 %, which the author accepted as drift. The author accepted these 8 as
  drift too (2026-10-20): they are below 1 %, in both directions, and the linked kernels of
  steps 2 and 3 have the same registers.
- Steps 4, 5 and part 4c leave the GPU timings of the existing response unchanged.

DFFR against the existing response, both in the part 4c environment, as the speed-up of DFFR
(existing time over DFFR time; ESA, `L1Orbits` within 3 % except at batch 1):

| Method | Batch 1 | 10 | 100 | 1000 | 10 000 | Per source at 10 000, existing / DFFR [µs] |
| --- | --- | --- | --- | --- | --- | --- |
| `fill_global_stft` | 5.2 | 5.7 | 5.8 | 7.9 | 8.8 | 57.9 / 6.6 |
| `get_ll_stft` | 7.2 | 5.0 | 5.1 | 7.2 | 8.2 | 63.5 / 7.7 |
| `get_ll_grad_stft` | 7.5 | 4.9 | 4.3 | 4.9 | 4.9 | 998 / 206 |
| `get_swap_ll_stft` | 4.7 | 3.4 | 3.4 | 4.0 | 4.1 | 179 / 43.7 |
| `get_fstat_ll_stft` | 4.5 | 4.9 | 4.8 | 4.7 | 4.0 | 1077 / 273 |
| `get_swap_ll_grad_stft` | 6.4 | 2.8 | 2.2 | 2.2 | 2.1 | 5457 / 2584 |
| `information_matrix`, 9 parameters | 1.02 | 1.08 | 1.07 | 0.70 | 0.76 | 614 / 812 |
| `information_matrix`, 8 parameters | 1.02 | 1.06 | 1.06 | 0.95 | 1.03 | 521 / 508 |

DFFR is faster in every likelihood method at every batch size, by 2.1 to 8.8 times from 1000
sources on. The Fisher matrix of this build is not; see the next entry for the change of its
reduction.

**Shared code of the exchange scripts.** `dffr_exchange_lat.py` gained `build_dffr_comp` (the comp,
its domain, Fresnel object, kinematics table and group stand-in) and `dffr_exchange_dffr.py`
`build_response`, which the timing scripts below use as well. On 2026-10-20 `dffr_exchange_lat.py`
gained `--inputs` (a capture on the inputs of `dffr_timing_lat.py --write-inputs`, with
`--max-sources`) and `--by-segment` (the compact templates compared per group of segments). On one
node, the committed and the changed scripts give the same exchange outputs on the fixture: LAT
side 29 of 29 arrays (checked after each of the two changes), dffr side 14 of 14, bit-identical.
Captures of one script on two CPU types differ in 8 of the 29 arrays, as the conventions expect.
Files: `$OUT/regression/step6_prep/exchange_refactor_check/`.

**Final reduction of the DFFR Fisher in dynamic shared memory** (2026-10-20, uncommitted in
`lat_dffr_kernels.hh`). The reduction staged its sums in a static array of 128 threads × 9
derivatives (9 216 bytes). It now stages them where the reception states and amplitude parts were,
which are free after a source's last pass, with the same tree and so the same order of the sums.
The author had asked for a warp-shuffle reduction to free the memory; this variant frees it as
well and keeps the GPU results bit for bit. `dffr_information_bytes` takes the larger of the
tables and the staging, so the dynamic shared memory of every layout in use is unchanged.
`kernel_resources.py` against the step 5 module, for the A100 and the H100
(`$OUT/regression/step6_fisher/kernel_resources_sm{80,90}.txt`):

| Kernel | Static shared [B], A100 / H100 | Blocks per SM by registers and by shared memory, A100 / H100 |
| --- | --- | --- |
| DFFR Fisher, 4 rows, 8 parameters | 9 696 → 480 / 10 720 → 1 504 | 2, 2 / 2, 3 (was 2, 2 / 2, 2) |
| DFFR Fisher, 4 rows, 9 parameters | 9 696 → 480 / 10 720 → 1 504 | 2, 2 / 2, 2 (was 2, 1 / 2, 2) |

Registers (170 and 174), stack (1 848 bytes) and dynamic shared memory are unchanged, and so is
every other kernel of the module. On nine parameters the A100 now runs 2 blocks per SM instead of
1, as the existing Fisher does.

Checks after the rebuild of the shared environment with this change (2026-10-20, 15:45 to 16:16,
`$OUT/regression/step6_fisher/staging_chain.sh`):

- CPU, both captures on the A100 node before and after the rebuild: the DFFR outputs, Fisher
  matrices included, bit-identical, 50 of 50 and 25 of 25. Pass.
- GPU, against the part 4c GPU captures: DFFR bit-identical except `fill` (48 of 50 and 24 of 25;
  `fill` at most 1.1e-16 of its peak, the rounding of its atomic additions), so the Fisher
  matrices are bit-identical; the existing response 144 of 144. Pass.
- `DFFRFisherTest` and `DFFRGPUFisherTest`: 10 of 10 pass.
- Timing gate of the DFFR Fisher again, in turn with the existing Fisher, three runs each (ratio
  of the DFFR time to the existing time):

  | Batch | 9 parameters, ESA | 9 parameters, `L1Orbits` | 8 parameters, ESA | 8 parameters, `L1Orbits` |
  | --- | --- | --- | --- | --- |
  | 1 | 1.008 | 0.992 | 0.990 | 0.983 |
  | 10 | 0.945 | 0.940 | 0.940 | 0.936 |
  | 100 | 0.936 | 0.929 | 0.932 | 0.930 |
  | 1000 | 1.046 | 1.074 | 1.058 | 1.078 |
  | 10 000 | 0.974 | 0.980 | 0.987 | 0.993 |

  On nine parameters the ratio at 1000 sources falls from 1.44 to 1.47 to 1.05 to 1.07, and at
  10 000 from 1.34 to 0.97 to 0.98. Eight parameters keep the ratios of part 4c. The gate, not
  slower than the existing Fisher, now fails at 1000 sources only, by 5 % to 8 %, on both
  parameter sets (`compare_gpu_{esa,l1}_fisher_dffr_vs_existing.txt`).

**Where the DFFR Fisher spends its time on the A100** (2026-10-20, on the code with the change
above). A side build, never committed, adds cycle counters to the kernel: thread 0 of each block
reads `clock64()` around the table build of each pass, the band setup, the pixel loop (closed by
an added barrier, so that it is the block's) and the final reduction, and prints its sums for 32
sources per call. GPU profiling counters are open to administrators only on SURF Spider, so
Nsight Compute cannot be used. Fixture `esa_tdi2_envelope`, a year of one-day segments, 4 rows per
pass, 1000 and 10 000 sources, 64 sources per line (`$OUT/validation/fisher_profile/`:
`patch_header.py`, `build_profile.sh`, `run_with_profile.py`, `summary.txt`).

| Derivatives | Table build | Band setup | Pixel loop | Reduction | Cycles per source and block |
| --- | --- | --- | --- | --- | --- |
| 8 | 50.0 % to 50.2 % | 0.1 % | 49.6 % to 49.8 % | below 0.1 % | 149 to 150 million |
| 9 | 48.4 % to 48.6 % | 0.1 % | 51.3 % to 51.5 % | below 0.1 % | 173 to 176 million |

The batch size does not change the split. At the A100's clock of about 1.41 GHz, 150 million
cycles per source and 216 blocks at a time give about 490 µs per source, the measured 507 µs. Half
of the time is the table build, which builds one stencil template at a time and keeps 12 to 24 of
the 128 threads busy. The first profile run failed at import: the module relinked from the
recorded commands kept the build tree's library path, which does not hold `libcudart.so.13`;
`build_profile.sh` now sets the toolkit's path with `patchelf` (`failed_run_1/`).

**Direct build of the DFFR Fisher tables** (2026-10-20, uncommitted in `lat_dffr_kernels.hh`). The
build of part 4c ran the tile's table build once per stencil template, through a scratch table
that the block then copied into the Fisher tables. At 4 rows its stages had 4 to 24 work items,
so one warp did the work, with 4 barriers per template. The author asked for a build straight into
the Fisher tables, with all threads of the block and no scratch table. All templates at once do not
fit: the build also makes values that only its later stages read, per template the shifted source
constants (152 bytes) and per segment row 6 emission states and 6 link amplitudes (336 bytes).
At 4 rows that is 54 kB for the 36 templates of 9 parameters, where 5.4 kB are free before the
second block per SM goes. The author chose these changes instead (2026-10-20):

- One build for the tile and the Fisher matrix. The stages of the tile's table build run over
  (template, row) items and reach every field through a storage type (`dffr_build_rows`). The
  tile's table is the storage of one template, a wave of Fisher templates the other. Each field
  comes from the same function and the same inputs as before; only where it is stored differs.
- Waves in the anchor space. The four endpoint anchors of each reception state moved out of
  `DFFRFisherReception` into an array of their own, `DFFRFisherAnchors`. Only the pixels read them,
  so they are built after the last wave, and until then the waves keep their intermediates in their
  space. A wave holds as many templates as fit there, spread evenly over the waves: at 4 rows 9 of
  the 36 templates on 9 parameters and 8 of the 32 on 8, in 4 waves. Where the anchors take less
  than one template (A and φ0 alone, one set of states), the space is enlarged to hold one. A pass
  has 13 barriers instead of 129 on 8 parameters and 145 on 9.
- The band setup takes each derivative's carrier range from the band starts of its templates, so
  the build keeps no carrier ranges. The scratch table, its copies and the layout field
  `variant_derivative`, now unused, are gone.

Resources (`kernel_resources.py` on the shared environment's build tree, which compiles the edited
header; `$OUT/regression/step6_build/kernel_resources_sm{80,90}.txt`, against the staging
change's report above):

| DFFR Fisher | A100 (sm_80) | H100 (sm_90) |
| --- | --- | --- |
| Registers | 170 → 176 | 174 → 183 |
| Dynamic shared memory at 4 rows, 8 / 9 parameters [B] | 70 464 → 66 272 / 81 280 → 77 088 | same |
| Dynamic shared memory at 8 rows, 8 / 9 parameters [B] | 140 640 → 132 400 / 162 272 → 154 032 | same |
| Blocks per SM by registers and by shared memory at 4 rows, 8 / 9 parameters | 2, 2 / 2, 2 (unchanged) | 2, 3 / 2, 2 (unchanged) |

Stack (1 848 bytes), spills (none) and static shared memory are unchanged. A third block per SM
would need 168 registers or fewer, so the added registers cost nothing. Nine parameters at 8 rows
now fit an A100 block (154 032 bytes and 800 static, below 163 kB), where that launch raised before.
The tile's kernels keep their registers, stack and shared memory; only the own registers of the
DFFR swap gradient on the A100 change, 180 → 184, below its linked 254.

Machine code (`cuobjdump` with `nvdisasm` 13.2, `compare_sass.py` and `compare_fp_ops.py` of
`$OUT/regression/gpu_review/sass/`; listings in `$OUT/regression/step6_build/sass/`): every DFFR
function has other instructions on both devices. The tile's kernels and functions keep their
double-precision operations in number and kind, in another order. The Fisher kernels keep their
double multiplies, adds, fused multiply-adds and reciprocals (83 `MUFU.RCP64H` on the A100), and have
8 more single-precision reciprocals, all in integer divisions (5 → 13, the division of the work
items into template and row), and 6 fewer double comparisons (390 → 384).

Checks (chain `$OUT/regression/step6_build/chain.sh`, job 41556413 on wn-gb-02, 2026-10-20 from
21:00 to 22:31). "Before" is the shared environment, built with the staging change. "After" is
GBGPU's CPU and CUDA modules with `gb_dffr.cu` recompiled from the edited header by the commands the
build tree records, and linked into copies of the installed packages (`build_overlay.sh`, loaded by
`run_with_overlay.py`), so nothing installed changed. A check of the loaded module files confirms
which build each run used (`loaded_{before,after}.log`).

- CPU, both captures on this node: the DFFR outputs through the comp, Fisher matrices included,
  bit-identical, 50 of 50 and 25 of 25. The "before" captures also equal those of the staging
  chain. Pass.
- Exchange with standalone dffr on the CPU: the LAT capture is bit-identical before and after, 29 of
  29 entries, compared entry by entry, because one entry is text, which `--compare-files` cannot
  subtract. Against standalone dffr, none of the 14 outputs is above the gate of 1e-13. Pass.
- GPU, against the staging chain's captures: DFFR 48 of 50 and 24 of 25 bit-identical, Fisher
  matrices included; the other three are `fill`, within 1.1e-16 of its peak (the order of its
  atomic additions, criterion 1e-15). The existing response 143 of 144; the other is its `fill`,
  3.0e-17. Pass.
- `tests/test_stft_gb_dffr.py` with both builds: 36 passed and 2 failed before and after. The 2
  failures are the `DFFRGPUTest` cases recorded in step 4; the new tests pass with both builds.

Timings on the A100 (`$OUT/regression/step6_build/timing/`), in turn: per run and fixture the
existing Fisher, the DFFR Fisher before, after, and after at 8 rows, then the DFFR likelihood
methods before and after. Three runs on both fixtures, a year of one-day segments, batches 1 to
10 000. The DFFR Fisher after the change, over the existing Fisher:

| Batch | 9 parameters, ESA | 9 parameters, `L1Orbits` | 8 parameters, ESA | 8 parameters, `L1Orbits` |
| --- | --- | --- | --- | --- |
| 1 | 0.632 | 0.584 | 0.563 | 0.554 |
| 10 | 0.540 | 0.540 | 0.528 | 0.525 |
| 100 | 0.518 | 0.512 | 0.510 | 0.505 |
| 1000 | 0.555 | 0.562 | 0.560 | 0.567 |
| 10 000 | 0.510 | 0.513 | 0.518 | 0.520 |

- The gate, not slower than the existing Fisher, holds at every point. At 10 000 sources the DFFR
  Fisher takes 309 µs per source on 9 parameters and 266 µs on 8, the existing one 606 and 513 µs.
- Against the code before the change, the DFFR Fisher takes 0.52 to 0.63 of the time (median
  0.55): 1.9 times faster from 1000 sources.
- 8 rows over 4 rows: 0.54 to 0.76 at batch 1, 0.90 to 0.93 at 10 and 100, 1.25 to 1.32 at 1000 and
  10 000. At 8 rows a block fills an SM's shared memory, at 4 rows two blocks share it. While a
  batch has fewer sources than the A100 has SMs (108), no SM holds a second block either way, and
  the 8-row build wins: half the passes, twice the items per stage. From 1000 sources the second
  block per SM wins. At 8 rows the Fisher takes 0.30 to 0.73 of the time of the existing one, so
  it meets the gate as well. The author decided on 2026-10-21 to keep 4 rows for every batch.
- DFFR likelihood methods, after over before: median ratio 1.001 (ESA) and 1.000 (`L1Orbits`), 27
  and 23 of 30 points within the run-to-run spread, the others within 0.8 % in both directions
  (`fill_global_stft` at batch 1 aside, whose run medians spread by 22 % to 70 %).
  `fill_global_stft` at 10 000 sources is 0.8 % to 0.9 % slower on both fixtures, in 5 of 6 pairs
  of runs; its kernel does the same arithmetic in another order, and its outputs are unchanged.
  The author accepted this as drift (2026-10-21).

CPU (the A100 node, ESA fixture, 32 segments, batches 1, 8 and 64, three runs in turn;
`cpu_timing_chain.sh`, `$OUT/regression/step6_build/timing_cpu/`): the DFFR Fisher after over
before 0.993 to 1.006, 6 of 6 points within the spread; the likelihood methods without the swap
kernels 0.970 to 1.005, 10 of 12 within, `get_ll_stft` at batches 8 and 64 2.5 % faster.

Where the time goes now: the profile above, its patch adapted to the new call
(`$OUT/regression/step6_build/profile/`):

| Derivatives | Table build | Band setup | Pixel loop | Reduction | Cycles per source and block |
| --- | --- | --- | --- | --- | --- |
| 8 | 14.8 % | 0.4 % to 0.5 % | 84.7 % | 0.1 % | 80 million (149 to 150 before) |
| 9 | 12.3 % to 12.4 % | 0.4 % | 87.1 % to 87.2 % | 0.1 % | 93 to 94 million (173 to 176 before) |

The build takes 12 and 11 million cycles per source instead of 75 and 85 million, 6.3 and 7.4
times fewer. The pixel loop takes 68 and 81 million cycles instead of 74 and 90 million. The
counters give the elapsed cycles of a block that shares its SM with another block, so this
difference is not separated from the other block's phases. The pixel loop is now 85 % to 87 % of
the kernel; at 4 rows a pass gives 84 to 92 pixels to the 128 threads of a block.

Rebuild of the shared environment with the change (author's decision of 2026-10-21;
`rebuild_chain.sh`, 08:35 to 08:49 on wn-gb-02). The rebuilt environment reproduces the overlay's
captures: on the CPU bit for bit, 50 of 50 and 25 of 25; on the GPU bit for bit but for `fill`
(DFFR at most 1.2e-16 of its peak, the existing response 3.1e-17, the order of the atomic
additions). The DFFR test file gives the same outcomes, 36 passed and the 2 recorded failures. The
installed CUDA module holds the machine code of the overlay that was timed: of the 185 functions
of its sm_80 code, none has other instructions (`sass/installed_vs_overlay_sm80.txt`).

**Occupancy: `domains.cu` capped at 168 registers** (2026-10-21). Of the four settings the plan
lists for this A/B, the author chose to time the cap at 168 only, after an estimate of what each
would gain. The cap at 128 adds spills without a further block per SM, and forcing `values` and
`build` inline costs the `get_ll` gradient a block. Device link-time optimisation would give the
DFFR two-template kernels a third block, but needs a change of the install for the pip toolkit.
The two-template DFFR consumers are timed at 16 rows as well, inside both builds. Further
decisions of the author that day: the ESA fixture, 10 000 sources, three runs in turn; and a
setting goes in when it is faster beyond the run-to-run spread at some point and slower beyond it
at none, over every method of both responses.

Resources with the cap (`register_cap.py`, the script of step 4 pointed at the record of the
direct build, `$OUT/regression/step6_occupancy/register_cap_168_sm{80,90}.txt`). The cap applies to
every function of `domains.cu`, which GBGPU's CUDA module compiles from LAT's tree:

| | A100 (sm_80) | H100 (sm_90) |
| --- | --- | --- |
| Spill stores per call of `get_fourier_value`, `get_fourier_kernel`, `add_ip_swap_contrib` [B] | 164, 52, 96 | 164, 68, 96 |
| Existing response, every kernel, Fisher included | 244 → 166 registers: 2 → 3 blocks per SM | 222 → 166: 2 → 3 |
| DFFR swap | 220 → 168: 3 blocks at 16 rows, 2 at 32 (shared memory) | 222 → 166: 3 at both |
| DFFR F-statistic | 220 → 178 (its own code): 2 | 222 → 168: 3 at both |
| DFFR swap gradient | 254, set by the tile's out-of-line functions: 2 | 254: 2 |

`add_ip_contrib` does not spill, and the other DFFR kernels keep their registers. On the A100 the
DFFR F-statistic and swap gradient gain no block, while `add_ip_swap_contrib`, which they call,
spills 96 bytes per call.

Checks (chain `$OUT/regression/step6_occupancy/chain.sh`, job 41556413 on wn-gb-02, 2026-10-21
from 09:31 to 11:18). The cap build is GBGPU's CUDA module with `domains.cu` recompiled at
`-maxrregcount=168` by the build tree's recorded commands, linked into a copy of the installed
package (`build_cap_overlay.sh`); the baseline is the rebuilt shared environment.

- GPU outputs against the rebuilt environment's captures: not bit-identical, against the premise
  of the plan that a cap changes only the allocation of registers. Under the cap the compiler
  computes 12 products in `STFTDomainGPU::add_ip_contrib` apart from their additions instead of in
  fused multiply-adds (12 fewer `DFMA`, 12 more `DADD`), and `get_fourier_kernel` has one multiply
  less (`compare_fp_ops.py`, `sass/fp_ops_cap168_vs_rebuilt_sm80.txt`). Unchanged: the templates
  (`fill` within 1.2e-16 of its peak, the order of the atomic additions), the swap outputs and the
  Fisher matrices of both responses. Changed in both responses: (d|h), (h|h), the log-likelihood
  and the F-statistic, by at most 4.0e-16 of their peaks, and the gradients built from them, by at
  most 5.4e-10 of their peaks for the existing response and 1.4e-11 for DFFR. Existing response 92
  of 144 arrays bit-identical, DFFR 31 of 50 and 15 of 25.
- `tests/test_stft_gb_dffr.py`: 36 passed and the 2 recorded failures, with both builds.

Timings, ESA fixture, a year of one-day segments, 10 000 sources, three runs in turn
(`$OUT/regression/step6_occupancy/timing/`); ratio of the cap build to the baseline:

| Method | Existing response | DFFR, 32 rows | DFFR, 16 rows (over 32 rows of the baseline) |
| --- | --- | --- | --- |
| `get_ll_stft` | 0.919 | 0.996 (within) | |
| `fill_global_stft` | 0.896 | 1.000 (within) | |
| `get_ll_grad_stft` | 0.929 | 1.002 (within) | |
| `get_swap_ll_stft` | 0.950 | 1.010 | 0.970 (1.166 without the cap) |
| `get_fstat_ll_stft` | 0.937 | 1.004 (within) | 1.048 (1.050 without the cap) |
| `get_swap_ll_grad_stft` | 0.961 | 1.005 | 1.121 (1.116 without the cap) |
| `information_matrix`, 9 parameters | 0.848 | 1.000 (within) | |
| `information_matrix`, 8 parameters | 0.852 | 1.001 | |

- The existing response is faster in every method, by 4 % to 15 %, beyond the run-to-run spread
  (at most 0.6 %).
- DFFR at 32 rows keeps 5 of 8 points within the spread; the swap (1.0 %), the swap gradient
  (0.5 %) and the Fisher matrix on 8 parameters (0.1 %, against a spread of 0.03 %) are slower:
  the spills of `add_ip_swap_contrib`, without a block gained.
- 16 rows pay only for the DFFR swap with the cap, which gives it a third block per SM: 3 % faster
  than the baseline at 32 rows. Without the cap, 16 rows are 5 % to 17 % slower. The comp passes
  one number of rows to all three two-template methods.

Under the author's rule, faster at some point and slower at none, the cap is a mixed result: the
existing response gains in every method, DFFR loses in two to three. The author left the cap out
(2026-10-21), and the two-template DFFR consumers keep their 32 rows. This closes the occupancy
A/B of step 6.

### Validation runs

**Standalone dffr against DFFR in `STFTGBComputations` on the full A100** (2026-10-19). The author
decided that day to run the standalone package's own test suite on the A100 and to time both
codes on the same inputs, on the ESA orbits only. The ratio of the times is recorded and steers
the occupancy runs of this step; it is not a gate.

Code: dffr 01530a5 with an uncommitted change that adds a CUDA 13 backend (below); LAT 24b4bd8a and
GBGPU 2d52f70 (part 4c) with the uncommitted scripts `dffr_timing_lat.py` and `dffr_timing_dffr.py`.
Both codes were compiled by nvcc 13.2.86 for sm_80 and ran with CuPy 14.2.0 and the CUDA 13.2
runtime, on the node and job of the conventions above.

- CUDA 13 backend of standalone dffr. The author asked for it so that both codes use one toolkit:
  the standalone had a CUDA 12 backend only, while the lisatools stack builds with CUDA 13.2, so a
  CUDA 12 build would have timed two compilers. The change: `DFFRCuda13xBackend` in `src/dffr/backends.py`, which runs
  the generic CUDA checks of gpubackendtools 0.1.1 with the CUDA 13 version range, because that
  release has no CUDA 13 class; the extra `gpu-cuda13` (`cupy-cuda13x[ctk]` and the CUDA 13.2
  runtime libraries); and `tests/test_gpu.py` on the first CUDA backend found. CMake already names
  the module after the major version of the toolkit. The environment `.venv-gpu13` of the dffr
  clone was built with the CUDA 13.2 toolkit of the lisatools environment (`DFFR_CUDA_ARCH=80`,
  `DFFR_MARCH=x86-64-v3`). Logs: `$OUT/validation/standalone_cuda13_build*.log`.
- Standalone test suite on the A100 (`pytest -rA`, 73 s): 51 of 51 passed, the CUDA tests on
  `dffr_cuda13x` included. Its timing script `scripts/benchmark.py` and its parity script
  `scripts/backend_parity.py` were not run, because both build Keplerian orbits. Files:
  `$OUT/benchmarks/standalone/dffr_pytest_gpu.{log,xml}`.
- Inputs (`dffr_timing_lat.py --write-inputs`): LAT's kinematics table of the grid, 10 000
  sources, seeded data and a Hermitian 3 × 3 inverse covariance over one domain; a year of one-day
  segments from 10 days after the start of the ESA orbit file; TDI 2nd generation, the linear
  envelope, first-order delays and a chirp-rate floor of 1e-22 Hz/s. Three setups:
  - the setup of the standalone's benchmark, at band 41 and band 21: the band from 1.5 to 28.5 mHz
    (2333 bins), Tukey taper 0.116, sources drawn as that benchmark draws them (2 to 28 mHz, seed 7);
  - the production settings at band 21: the band from 0.1 to 29 mHz (2497 bins), Tukey taper
    0.1157, the sources of the production fixtures. The standalone has no store windows, so both
    codes take the whole band as one domain; the store windows of production are timed by the step
    benchmarks.
- Timing: `fill_global` and `get_ll` of each code on the first B sources, B from 1 to 10 000, one
  warm-up call and five timed calls, each closed by a device synchronisation. Three runs; within a
  run the two codes alternate, one process per code and setup. The standalone's `get_ll` computes
  (d|d) on every call, which the comp reads from its group; that step is timed on its own, and the
  columns "without (d|d)" remove it.

Time per source at 10 000 sources. The ratio is ours over the standalone; the spread is the larger
of the two codes' spreads of their run medians.

| Setup | Method | Ours [µs] | Standalone [µs] | Standalone without (d\|d) [µs] | Ratio | Ratio without (d\|d) | Spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standalone, band 41 | `fill_global` | 11.22 | 10.75 | | 1.044 | | 0.97 % |
| standalone, band 41 | `get_ll` | 14.59 | 13.52 | 12.86 | 1.079 | 1.134 | 0.87 % |
| standalone, band 21 | `fill_global` | 6.56 | 6.19 | | 1.059 | | 0.32 % |
| standalone, band 21 | `get_ll` | 8.45 | 8.16 | 7.50 | 1.035 | 1.126 | 0.71 % |
| production, band 21 | `fill_global` | 6.45 | 6.05 | | 1.066 | | 0.47 % |
| production, band 21 | `get_ll` | 7.72 | 7.94 | 7.27 | 0.972 | 1.061 | 0.22 % |

From 1 to 1000 sources the run medians of either code spread by up to 76 %, so those points are not
quoted; they are in `report.txt`. At 10 000 sources, DFFR in the comp takes 4 to 7 % more time per
source than standalone dffr in `fill_global`, and 6 to 13 % more in `get_ll` once the standalone's
(d|d) is removed.

Agreement of the outputs at 10 000 sources, the largest difference relative to the standalone's
peak: (d|h) at most 1.4e-15 and (h|h) 7.3e-16 in all three setups; the `fill_global` grids
1.0e-12 (standalone setup, band 41), 6.2e-13 (band 21) and 1.1e-13 (production settings). Two
checks place the grid difference (`fill_check/`):

- Two runs of the same code differ by 2.4e-16 of the peak (band 41), the order of `atomicAdd`.
  Between the codes the difference is 1.04e-12 in both pairs of runs, so it is systematic.
- The exchange on the GPU (24 production sources, 32 segments, `dffr_exchange_*.py --device gpu`)
  passes its gate of 1e-13 in every output: compact templates 1.7e-14, `fill_global` 1.2e-14, (d|h)
  5.0e-16, (h|h) 1.5e-16, the segment table identical except the TDI weights (2.7e-16).

On the CPU the two codes agree bit for bit (step 4), so on the GPU the two builds round
differently. Why the difference grows to 1e-12 over a year is not measured. One candidate: the
template phase reaches about 5e6 rad over a year, where one unit in the last place is about
1e-9 rad. The author decided on 2026-10-19 to locate it in this step, before deciding anything
else: the per-source templates of both codes at 32, 120 and 365 segments.

**Where the two codes' GPU templates differ** (2026-10-20). Compact templates of both codes on the
A100, on identical inputs: the setup of the standalone's benchmark at band 41, 100 sources, a grid
of 32, 120 and 365 one-day segments from the same start (`dffr_timing_lat.py --write-inputs`, then
`dffr_exchange_lat.py --capture --inputs`, `dffr_exchange_dffr.py --device gpu`, and
`dffr_exchange_lat.py --compare --by-segment`; `$OUT/validation/grid_difference/`). Largest
difference per segment, relative to the largest template value of that segment:

| Days after the data start | 32 segments | 120 segments | 365 segments |
| --- | --- | --- | --- |
| 0 to 73 | 1.3e-14 to 2.7e-14 | 1.3e-14 to 2.7e-14 | 2.2e-14 to 2.7e-14 |
| 74 to 95 | | 2.5e-14 to 5.8e-14 | 1.2e-13 |
| 96 to 119 | | 1.1e-13 to 1.2e-13 | 1.1e-13 to 1.2e-13 |
| 120 to 292 | | | 8.2e-14 to 1.2e-13 |
| 293 to 364 | | | 2.7e-14 |

The difference does not grow with time, as rounding of the template phase would make it: it rises
at the same days in the 120-day and the year-long grids and falls again after day 292. So it
belongs to the orbit's configuration over the year, which sets the arguments of the special
functions, not to the phase accumulated since the reference time. The segment tables of the two
codes agree bit for bit except the TDI weights (3.5e-16), so the difference arises in the
evaluation of the pixels, which the two builds compile differently. With 100 sources the
`fill_global` grids differ by 8.7e-14 of their peak, with 10 000 by 1.0e-12, about the square root
of the ratio of the source counts. The year-long grid difference is thus these per-template
roundings adding up over overlapping sources. (d|h) and (h|h) agree to 1.3e-15 and 3.7e-16. At 120
segments the templates differ by 1.01e-13 of the peak, just above the gate of 1e-13 of the
exchange, which was set for its 32 segments.

Files: `$OUT/benchmarks/standalone/` (`report.txt`, `inputs_*.npz`, `ours_*_run*.npz`,
`dffr_*_run*.npz`, `fill_check/`). Chain: `$OUT/benchmarks/standalone/standalone_chain.sh`; the
commands of a single run are in the docstring of `scripts/dffr/dffr_timing_lat.py`.

**Item 1 of the validation: both responses against the STFT of a time-domain truth, and the window
convention** (2026-10-21). The author set the scope and order of the validation on 2026-10-21;
this is its first item. It measures how well the templates of both responses reproduce the
transform of time-domain data, and how much of the difference comes from the window: the global
fit windows the data with LAT's symmetric Tukey window, while both responses integrate the
periodic one. The numbers are recorded; there is no gate.

Code: LAT 0195b391 and GBGPU 11d5159, with the uncommitted scripts `stft_accuracy.py`,
`td_truth.py`, `td_injection.py` and `run_step6.sh`; dffr 2b61cdd for the truth, in its own
environment (lisagwresponse 3.0.2, pytdi 2.2.1, lisainstrument 2.3.0, lisaorbits 3.1.0). The comps
ran on the A100 of the conventions above, the truth on the CPUs of the same node.

- Sources, chosen by the author for items 1 to 3: 0.5 and 20 mHz; at each, the largest and the
  smallest ḟ of the global fit's GB prior at that frequency (`get_fdot_mojito(f, "+")` and
  `(f, "-")`: 1.1e-18 and -6.6e-20 Hz/s at 0.5 mHz, 8.2e-13 and -2.3e-11 Hz/s at 20 mHz); each at
  the two sky positions of the step 0 truth, (λ, β) = (2.1, -0.35) and (5.0, 0.9) rad. A = 1e-22,
  φ0 = 1.2, ι = 0.9 and ψ = 0.7, as in the step 0 truth, and f̈ = 0, as the global fit sets it. The
  parameters refer to the start of the ESA orbit file.
- Truth: X2, Y2 and Z2 every 5 s from the chain of the standalone package's truth
  (`dffr.comparison.TimeDomainTruth`): lisagwresponse on the strain at the exact retarded times,
  then pytdi, on the ESA orbits that LAT ships (the dffr environment ships the same file; the
  scripts compare its MD5). A year: 365 one-day segments from 10 days after the start of the orbit
  file. The step 0 truth put lisainstrument, with every noise off, between the two. Over 2 days
  the two chains agree to 6.3e-7 of the peak at 0.5 mHz and 4.2e-7 at 20 mHz (`truth_chains/`),
  while a year of the longer chain needs about 74 GB per source (6.1 GB per 30 days in step 0), so
  the truth leaves lisainstrument out.
- Data: `TDSignal.stft` of the truth, dt·rfft(w·x) per segment, with two windows: LAT's
  `windowfun("tukey", N, 0.1157)`, as `globalfit/engine.py` builds the data, and the periodic
  Tukey window (`scipy.signal.windows.tukey(N, 0.1157, sym=False)`), which both responses
  integrate.
- Templates: `STFTGBComputations` with each response and the production settings (XYZ, TDI 2nd
  generation, taper 0.1157, segment midpoints, linear envelope, `freq_from_tdi_phase`), but 60
  side bins (author's decision, below; production takes 10). Each source has a domain over its
  frequency track for the year, plus the side bins and 20 bins on each side, cut at the start of
  the production band (0.1 mHz). Inverse covariance: the XYZ instrument noise of
  `XYZSensitivityBackend` (TDI 2nd generation, its default noise levels). The template comes from
  `fill_global_stft`, (d|h) and (h|h) from `get_ll_stft`, and (d|d) from the group.
- Metrics, per source, response and window, with d the data and h the template. Over the pixels
  that the template fills, the metrics of the step 0 truth: the flat mismatch
  1 - |<d, h>| / (|d| |h|), with a free phase, and per segment and channel max |h - d| / max |d|.
  With the noise weights of the likelihood, over the whole domain and without a free phase: the
  mismatch 1 - Re(d|h) / sqrt((d|d)(h|h)), the residual (d-h|d-h)/(d|d) (the loss in ln L at the
  true parameters is SNR² times half of it), the amplitude ratio sqrt((h|h)/(d|d)) and the phase
  of (d|h). Also the share of (d|d) outside the pixels that the template fills.
- Checks: the truth against `GBTDIonTheFly` in the time domain (2 days from day 100 of the data,
  spline knots every 300 s): 1.6e-7 to 3.3e-7 of the peak at 20 mHz, with a least-squares scale
  within 2.2e-8 of 1, and 1.3e-5 to 1.7e-5 at 0.5 mHz. The noise-weighted products of the script
  agree with those of the comp and the group to 1.8e-15 at 20 mHz and 3.8e-11 at 0.5 mHz, where
  the 3 × 3 XYZ noise covariance has a condition number of 3e4 to 1.1e5 (2.6 at 20 mHz).
- Command, from the LAT root inside the job:
  `PYTHON=$PYTHON DFFR_PYTHON=<dffr repository>/.venv/bin/python scripts/dffr/run_step6.sh --device gpu $OUT accuracy truth_chains`.
  The truth takes about 12 minutes (8 processes side by side), the comps about 2 s per source.
  Files: `$OUT/validation/accuracy/`: `truth/`, `year_pytdi_n60/` (`report.txt`, `summary.json`,
  a figure per source, `summary.png`), `year_pytdi_n10/` (`N_SIDE_BINS=10`), `truth_chains/`.

Results with 60 side bins, on data with the periodic window (the author's choice for the
benchmark, below). Ranges over the 4 sources of each frequency.

| Frequency | Response | Flat mismatch, template pixels | Median segment error | Largest segment error | Noise-weighted mismatch | Residual |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 mHz | DFFR | 2.7e-8 | 1.0e-4 | 2.5e-4 to 9.0e-4 | 1.3e-7 | 2.5e-7 to 2.7e-7 |
| 0.5 mHz | existing | 2.7e-8 | 1.0e-4 | 3.2e-4 to 6.2e-3 | 7.2e-6 to 1.4e-5 | 1.4e-5 to 2.8e-5 |
| 20 mHz | DFFR | 4.8e-10 to 6.8e-10 | 2.2e-5 to 3.2e-5 | 4.0e-4 to 4.7e-4 | 2.4e-8 to 3.6e-8 | 4.8e-8 to 7.2e-8 |
| 20 mHz | existing | 7.6e-10 to 1.2e-9 | 1.4e-5 to 1.9e-5 | 4.1e-3 to 4.6e-3 | 2.5e-8 to 3.7e-8 | 5.0e-8 to 7.3e-8 |

- At 20 mHz DFFR matches the truth at least as well as the standalone package did in step 0
  (mismatch 7.5e-10 to 1.2e-9 and median segment errors 3.2e-5 to 4.1e-5 at 20 mHz, over 30
  days). The largest segment errors of both responses fall where one channel's signal is weak: in
  the worst segment that channel's peak is 6 % to 7 % of its median over the year. There DFFR
  reaches 4.0e-4 to 4.7e-4, as in step 0 (4.8e-4 at 20 mHz in sky 0), and the existing response
  4.1e-3 to 4.6e-3, while its median error is the lower one. The noise-weighted mismatch of both is
  set by the power beyond 60 bins, 4.6e-8 to 7.0e-8 of (d|d).
- At 0.5 mHz the band of 60 side bins reaches the start of the production band. There the data
  hold the leakage of the signal's negative-frequency image, which neither response computes:
  both give the transform of the positive-frequency part. It sets a floor of 1.0e-4 in the
  segment error and of 2.7e-8 in the flat mismatch of both. With 10 side bins these pixels lie
  outside the band, and the median segment errors are 3.7e-5 (existing) and 4.9e-5 to 5.4e-5
  (DFFR).
- At 0.5 mHz the noise-weighted mismatch of the existing response, 7.2e-6 to 1.4e-5, is 50 to 100
  times that of DFFR. Its errors sit in the combination T = (X + Y + Z)/√3, which carries 0.9 % of
  the signal of A there and whose noise is far below that of A and E, so the likelihood weights it
  heavily. Relative error in T over the template pixels: 2.6e-3 to 3.7e-3 for the existing
  response, 3.4e-4 for DFFR; in A and E both have 2.3e-4, set by the image above. With 10 side
  bins the band truncation (below) hides this.
- With 10 side bins the sampling of the ESA orbit file, every 1.87 days, shows at 0.5 mHz: both
  responses follow the same day-to-day ripple and share an error floor of about 3.7e-5 per
  segment, and the truth differs from `GBTDIonTheFly`, which reads LAT's orbits as both responses
  do, by 1.3e-5 to 1.7e-5. LAT and lisagwresponse interpolate the file's samples differently; that
  is the likely cause, not tested directly.

The window convention. The data window of the global fit, LAT's symmetric Tukey window, divides
by N - 1 in its taper and is zero at the last sample of a segment. It equals scipy's symmetric
Tukey window to 8e-9 and is the periodic window of the models squeezed in time by (N - 1)/N. In
the trailing taper it is lower by up to 1.53e-3 (2465 s before the segment end, 9.7e-4 on average
over the last 5000 s), in the leading taper by at most 5.3e-5, and its sum is 5.8e-5 smaller
(`window_explained/windows.png`, made with the script's window formulas). On the truth, the same
in every segment and for every source, the LAT-window data differ from the periodic-window data
by:

| Quantity | 0.5 mHz | 20 mHz |
| --- | --- | --- |
| Flat mismatch | 3.6e-8 | 3.6e-8 |
| Median segment error | 6.3e-5 | 6.3e-5 to 6.6e-5 |
| Amplitude ratio minus 1 | -2.9e-5 | -2.9e-5 |
| Phase [rad] | -6.4e-6 | -9.7e-8 to -5.8e-8 |
| Residual, of (d\|d) | 7.5e-8 | 7.3e-8 |

Against LAT-window data both responses carry this on top of their own error. With 60 side bins:
flat mismatch 3.7e-8 to 3.8e-8 at 20 mHz and 6.2e-8 to 6.4e-8 at 0.5 mHz, median segment errors
7.5e-5 to 9.3e-5 and 1.0e-4 to 1.1e-4, residual 1.2e-7 to 1.5e-7 at 20 mHz. In ln L at the true
parameters the window costs SNR² × 3.6e-8: 4e-4 at SNR 100. The author decided on 2026-10-21 to
benchmark the accuracy on data with the periodic window and to leave production as it is: there
the 10 side bins leave a mismatch four orders of magnitude larger (below).

Side bins. With the 10 side bins of production (`year_pytdi_n10/`), 1.3e-3 (20 mHz) to 1.7e-3
(0.5 mHz) of (d|d) lies outside the template's pixels, with either window. That sets the
noise-weighted mismatch of both responses at 6.4e-4 to 8.8e-4 and their residual at 1.3e-3 to
1.8e-3: at SNR 100, a loss of 6 to 9 in ln L at the true parameters. The window causes it: under
the taper 0.1157 a tone keeps 1.2e-3 to 1.4e-3 of its power beyond 10 bins, 1.3e-6 to 5.8e-6
beyond 30 and 3.8e-8 to 1.1e-7 beyond 60. On data with the periodic window, the in-band metrics at
20 mHz equal those of the 60-bin run to two digits. The author decided to run the validation with
60 side bins.

LAT's full time-domain response. The first injection was LAT's `pyResponseTDI`
(`td_injection.py`; `stft_accuracy.py` without `--truth`, `year_pyresponsetdi_n10/`). Its check
against `GBTDIonTheFly` showed an error that grows as f^-3 below 5 mHz: 3.0e-3 of the peak at
0.5 mHz, 3.7e-4 at 1 mHz, 4.3e-5 at 2 mHz, 2.7e-6 at 5 mHz and 3.4e-7 at 20 mHz. `pyResponseTDI`
interpolates the factor A(e) of its fractional-delay Lagrange weights linearly from a table of
1001 values (`MAX_A_VALS` in `LISAResponse.cu`), which underestimates A by up to 2.1e-7. At
20 mHz that is a scale error of 2.4e-7; at low frequency TDI 2 amplifies it, because each link term
is a small difference of two strain values and the combination cancels the links again. In the
template band the effect is smaller: the two injections agree to 2.5e-7 to 2.9e-7 per segment
(median) at 20 mHz and to 1.2e-5 at 0.5 mHz, so most of the time-domain error lies outside the
band. The author then chose the truth chain of the standalone package and decided to record this
finding only.

**Item 2 of the validation: the cases of the step 0 truth through both responses** (2026-10-21).
The second item of the author's scope, as the author defined it after item 1: the truth of step 0
for both responses of `STFTGBComputations`, on the cases and the grid of step 0, over 30 days with
the chain of step 0 and over a year without lisainstrument. It ties the accuracy that `README.md`
quotes for the standalone package to the code of production. The numbers are recorded; there is
no gate.

- Code: as item 1, with `stft_accuracy.py --source-set step0`.
- Cases, those of step 0: 3, 20 and 40 mHz with ḟ = 3e-16 Hz/s, and 20 mHz with ḟ = 1e-13 Hz/s,
  carriers moved to the nearest bin, each at the two sky positions; A = 1e-22, φ0 = 1.2, ι = 0.9,
  ψ = 0.7 and f̈ = 0.
- Grid of step 0: one-day segments from 5 days after the start of the ESA orbit file; data
  windowed with the periodic Tukey window of taper 0.116, which both responses take as well; flat
  metrics over the 51 bins within 3e-4 Hz of the carrier, in every segment. Both responses use 60
  side bins, which hold that band in every segment (the script checks it).
- Truth: the chain of step 0 (lisagwresponse, lisainstrument with every noise off, pytdi) over 30
  days, and the chain of item 1 (without lisainstrument) over a year and over its first 30 days.
- Checks: the truth against `GBTDIonTheFly` (2 days from day 28 of the 30-day runs and from day
  100 of the year): at most 2.8e-6 of the peak at 3 mHz and 1.3e-6 at 20 and 40 mHz. The products
  of the script agree with those of the comp and the group to 3e-13 at 3 mHz and 4e-15 at 20 and
  40 mHz.
- Command, from the LAT root inside the job:
  `PYTHON=$PYTHON DFFR_PYTHON=<dffr repository>/.venv/bin/python scripts/dffr/run_step6.sh --device gpu $OUT step0`.
  The 16 truth runs go side by side: about 3 minutes per case for 30 days with lisainstrument and
  15 minutes for a year without; then three runs of `stft_accuracy.py`. Files:
  `$OUT/validation/accuracy/step0/` (`truth/`, `truth_instrument_30d/`, `30d_instrument_n60/`,
  `30d_gws_n60/`, `year_n60/`).

Over 30 days with the chain of step 0 (`30d_instrument_n60/`), next to the standalone package of
step 0 on the same cases. Each entry gives the flat mismatch, the median and the largest segment
error.

| Case | Sky | Standalone dffr, step 0 | DFFR in `STFTGBComputations` | Existing response |
| --- | --- | --- | --- | --- |
| 3 mHz | 0 | 1.23e-9, 4.65e-5, 1.41e-4 | 1.23e-9, 4.65e-5, 1.41e-4 | 8.72e-10, 1.85e-5, 2.83e-4 |
| 3 mHz | 1 | 6.47e-10, 3.17e-5, 9.89e-5 | 6.47e-10, 3.17e-5, 9.89e-5 | 1.04e-9, 2.87e-5, 8.66e-5 |
| 20 mHz | 0 | 1.24e-9, 4.06e-5, 4.79e-4 | 1.24e-9, 4.06e-5, 4.79e-4 | 1.04e-8, 1.93e-5, 4.08e-3 |
| 20 mHz | 1 | 7.49e-10, 3.23e-5, 5.63e-5 | 7.49e-10, 3.23e-5, 5.64e-5 | 9.61e-10, 2.99e-5, 8.16e-5 |
| 20 mHz, ḟ = 1e-13 Hz/s | 0 | 1.24e-9, 4.05e-5, 4.75e-4 | 1.24e-9, 4.04e-5, 4.75e-4 | 1.04e-8, 1.92e-5, 4.05e-3 |
| 20 mHz, ḟ = 1e-13 Hz/s | 1 | 7.49e-10, 3.25e-5, 5.66e-5 | 7.49e-10, 3.24e-5, 5.66e-5 | 9.61e-10, 3.01e-5, 8.19e-5 |
| 40 mHz | 0 | 9.92e-10, 3.68e-5, 8.94e-5 | 9.93e-10, 3.68e-5, 8.93e-5 | 2.97e-10, 1.43e-5, 7.78e-5 |
| 40 mHz | 1 | 1.08e-9, 4.53e-5, 7.41e-5 | 1.08e-9, 4.53e-5, 7.41e-5 | 5.86e-10, 2.42e-5, 7.37e-5 |

- DFFR in the comp reproduces the standalone package to two or three digits in every case,
  although it runs on the GPU from LAT's kinematics table, where step 0 ran the standalone CPU path
  from its own. The accuracy that `README.md` quotes thus holds for the integrated response.
- The existing response has the lower median errors, 1.4e-5 to 3.0e-5 against 3.2e-5 to 4.7e-5,
  and the smaller mismatch at 40 mHz and at 3 mHz in sky 0. At 20 mHz in sky 0 its mismatch is
  1.04e-8, 8 times that of DFFR, from the segments where the Y channel is weak (largest error
  4.1e-3).
- The chain without lisainstrument changes these numbers by at most 1 % (`30d_gws_n60/`).

Over a year without lisainstrument (`year_n60/`): flat mismatch 3.7e-10 to 1.3e-9 and median
segment errors 2.2e-5 to 4.0e-5 for DFFR, 5.8e-10 to 1.4e-9 and 1.5e-5 to 2.0e-5 for the existing
response. The largest segment errors come from segments where one channel nearly vanishes: at
3 mHz in sky 1, on day 352, the peak of Y is 1 % of its median over the year, and the errors reach
2.4e-3 (DFFR) and 7.8e-2 (existing); the 99th percentiles of the segment errors of that case are
3.5e-4 and 1.8e-3. With the noise weights of the likelihood, both responses reach 1.7e-8 to 2.4e-8
at 20 and 40 mHz, set by the power beyond 60 bins (3.3e-8 to 4.6e-8 of (d|d)). At 3 mHz the
existing response reaches 5.8e-8 to 1.2e-7 against 2.2e-8 to 2.3e-8 for DFFR: the effect of the
combination X + Y + Z that item 1 found at 0.5 mHz, smaller at the higher frequency.

The window convention on these cases (LAT's window against the periodic one, taper 0.116): flat
mismatch 3.5e-8 to 3.6e-8 and median segment errors 5.8e-5 to 6.6e-5, as in item 1.

**Item 3 of the validation: likelihood surfaces of four waveforms against a time-domain truth**
(2026-10-21). The third item of the author's scope. Each of four waveforms is evaluated in its
own domain against the transform of the same noiseless time-domain data, one parameter at a time.
The scans show where each waveform puts the peak of its likelihood, and how wide the peak is, in
units of the widths of DFFR's Fisher matrix. The numbers are recorded; there is no gate.

- Code: LAT 0195b391 with the scripts of items 1 and 2 and the new `likelihood_surfaces.py`;
  GBGPU 11d5159. The comps and FastGB ran on the A100 of the conventions above, the FD
  heterodyne on the CPUs of the same node.
- Waveforms: DFFR and TDI on the fly in `STFTGBComputations` (STFT); TDI on the fly in the
  frequency domain, `GBFDComputations` (`GBTDIonTheFly` sampled on N points over the year,
  shifted by the carrier and Fourier transformed); and FastGB, `gbgpu.gbgpu.GBGPU`.
- Data: the truth of item 1 for its eight sources (lisagwresponse and pytdi, X2, Y2 and Z2 every
  5 s over a year from 10 days after the start of the ESA orbit file), scaled to an optimal SNR of
  100, where SNR² is the STFT (d|d) with the periodic window and 60 side bins. The truth is linear
  in the amplitude, so the scaling needs no new run: A = 5.7e-21 to 6.2e-21 at 0.5 mHz and 1.2e-22
  to 1.3e-22 at 20 mHz.
- STFT: the data and settings of item 1 (periodic Tukey window of taper 0.1157 on the data, 60
  side bins, production settings otherwise), with one CSD per segment.
- FD: dt rfft of the same 365 days with the window of the global fit's FD path, LAT's symmetric
  Tukey with a taper of 1e4 s (α = 3.17e-4). Both FD codes run without a window: their windows act
  on whole samples of T/N, 3850 s at N = 8192, so they cannot represent a taper of 5000 s at each
  end. The products cover a band of N bins about the carrier at the data start, with N = 8192 at
  0.5 mHz and at 20 mHz with the largest ḟ and N = 262 144 at 20 mHz with the smallest ḟ: four
  times the smallest power of two, at least 2048, that holds the Doppler-shifted frequency track
  with a margin of 64 bins. One CSD for the year, from `XYZSensitivityBackend` with its transfer
  functions averaged over the orbit.
- Likelihoods: on X, Y and Z with the full complex 3 × 3 CSD in every domain (author's rule), as
  ln L(θ) = -(d - h(θ)|d - h(θ))/2, so that an exact waveform gives 0 at the true parameters. STFT:
  `get_ll_stft`. FD: one product in the script for both FD codes, since the own kernel of
  `GBFDComputations` (`get_ll_fd`) contracts with the real part of C⁻¹ only (checks below).
- Parameters: those of the truth, at t_ref = 0. FastGB returns the transform of [t0, t0 + T) with
  its time origin at t0, so its parameters must be those at its t0, which must be the data start
  to match the FD data. It runs with t0 at the data start, f0, φ0 and A carried there (exact in
  phase; the amplitude evolution, which FastGB lacks, to first order), and `flip_ref_phase=True`,
  since its phase is +φ0 + 2πf0ξ + … where LAT's is -φ0 + 2πf0τ + …; with TDI 2.0, XYZ, its
  compiled kernel up to N = 2048 and its Python path beyond. `GBFDComputations` requires
  t_start = t_ref, so it gets the same carried parameters with both at the data start.
- Scans: each of the eight parameters of the global fit (the nine of `GBTDIonTheFly` without f̈),
  the others at the truth, over ±4σ in 81 points, where σ = 1/√F_ii of DFFR's information matrix at
  the truth: the width of the likelihood along the scan with the other parameters fixed. A peak
  farther than 3σ from the centre of its scan gets up to three more scans centred on it.
- Metrics, per source, waveform and parameter: the peak offset (θ_peak - θ_true)/σ and the width
  relative to σ, from a parabola fitted within ±1σ of the largest grid value; ln L at the truth.
- Checks at the truth, before the scans: overlap, amplitude ratio and phase of (d|h) of every
  waveform; FastGB's Python path against its compiled kernel; `get_ll_fd` against the script's
  products; both FD codes at 4 and 16 times their N, cut to the same band, and against the data
  without a window; the FD window against no window; the weight of T = (X + Y + Z)/√3 in both
  domains, with the backend's other FD CSD for comparison.
- The author's decisions of 2026-10-21, in the sub-plan and at the review of the first source:
  FastGB and `GBFDComputations` as the FD waveforms; the global fit's FD window on the FD data;
  N four times the base length, after N = 2048 left the FD heterodyne at up to 2.2 in ln L at the
  truth from its own sampling; the CSD with averaged transfer functions (below); σ conditional;
  the physical parameters; X, Y, Z and the full CSD for data and likelihoods everywhere.
- Command, from the LAT root inside the job, after part `accuracy`:
  `PYTHON=$PYTHON scripts/dffr/run_step6.sh --device gpu $OUT surfaces`. At most four processes
  run side by side (`SURFACE_PROCESSES`): the first run, with all eight, ran out of the A100's
  40 GB for three sources, which were rerun; the script now returns CuPy's memory pool between its
  phases. A source takes 2 minutes, a 20 mHz source with the smallest ḟ 50 minutes, 46 of them for
  the FD heterodyne's 648 templates of 262 144 samples on the CPU. Files:
  `$OUT/validation/surfaces/` (`report.txt`, `summary.json`, per source `NAME.json`, `NAME.npz`
  and `NAME.png`, `summary_offsets.png`, `summary_widths.png`, `summary_levels.png`); the checks of
  the review in `dev/`.

Results. The largest peak offset is over the 8 parameters; widths are in units of DFFR's σ; the
mismatch is 1 - Re(d|h)/√((d|d)(h|h)) at the truth.

| Sources | Waveform | Largest peak offset [σ] | Width / σ | ln L at the truth | Mismatch at the truth |
| --- | --- | --- | --- | --- | --- |
| 0.5 mHz (4) | DFFR | 0.002 | 1.000 | -1.3e-3 | 1.3e-7 |
| | TDI on the fly, STFT | 0.010 | 0.998 to 1.000 | -0.07 to -0.14 | 7.2e-6 to 1.4e-5 |
| | TDI on the fly, FD | 0.016 | 0.70 to 0.86 | -0.49 to -0.77 | 2.8e-5 to 4.4e-5 |
| | FastGB | 6.6 | 0.89 to 0.99 | -3.1e3 to -3.2e3 | 0.20 to 0.21 |
| 20 mHz, largest ḟ (2) | DFFR | 0.002 | 1.000 | -2.4e-4 to -2.7e-4 | 2.4e-8 to 2.7e-8 |
| | TDI on the fly, STFT | 0.001 | 1.000 | -2.5e-4 to -2.8e-4 | 2.5e-8 to 2.8e-8 |
| | TDI on the fly, FD | 0.010 | 0.962 to 0.965 | -0.34 to -0.44 | 3.2e-5 to 4.1e-5 |
| | FastGB | 2.2 | 0.965 to 0.978 | -36 | 3.3e-3 to 3.4e-3 |
| 20 mHz, smallest ḟ (2) | DFFR | 0.002 | 1.000 | -3.6e-4 | 3.6e-8 |
| | TDI on the fly, STFT | 0.001 | 1.000 | -3.6e-4 to -3.7e-4 | 3.6e-8 to 3.7e-8 |
| | TDI on the fly, FD | 0.009 | 0.961 to 0.965 | -0.30 to -0.39 | 2.8e-5 to 3.6e-5 |
| | FastGB | 5.7 | 0.971 to 0.988 | -45 to -49 | 4.1e-3 to 4.5e-3 |

- DFFR puts every peak within 0.002σ of the truth, with widths equal to its Fisher matrix to three
  digits, at all eight sources; its error at the truth costs at most 1.3e-3 in ln L. At SNR 100
  its model error does not show in the surfaces, and its Fisher matrix gives the curvature of its
  likelihood against a time-domain truth.
- TDI on the fly in the STFT stays within 0.010σ. At 0.5 mHz it loses 0.07 to 0.14 in ln L at the
  truth, the error in X + Y + Z that item 1 found.
- The FD widths differ from the STFT ones because the domains hold different information, not
  because of the waveforms. At 20 mHz they are 0.962 to 0.965, √(1 - 5α/8) = 0.963: the year
  without a window keeps the 7 % of the power that the STFT's Tukey removes. At 0.5 mHz they are
  0.70 to 0.86: the FD CSD weights T more than the CSD per segment (FD SNR 132, below), and the
  widths in A and φ0, 0.756 to 0.759, equal the ratio of the SNRs, 100/132.
- TDI on the fly in the FD stays within 0.016σ. Its loss at the truth, 0.30 to 0.77 in ln L, has
  two causes. The data window, which no FD template models: the tapered year differs from the
  untapered one by a residual of 2.3e-5 to 7.1e-5 of (d|d) in the band. And at 20 mHz with the
  largest ḟ its own sampling on N = 8192 points: against untapered data it loses 0.34 to 0.44,
  0.019 to 0.025 at 4N and 0.0011 to 0.0015 at 16N, cut to the same band. That error sits mostly
  at the band's edges; the check at the review, on the middle 2048 bins only, had given at most
  0.1. Its own kernel (`get_ll_fd`) moves ln L at the truth by at most 2.1e-3: its (h|h) differs
  from the product with the full CSD by up to 2.9e-3 at 0.5 mHz and equals the script's product
  with Re(C⁻¹) to 1e-11.
- FastGB is off by up to 2.2σ at 20 mHz with the largest ḟ, 5.7σ with the smallest and 6.6σ at
  0.5 mHz, mostly in φ0 and ψ (its phase error) and in f0, ḟ and the sky; every final peak lies
  inside its scan, after 18 rescans at 0.5 mHz and 9 at 20 mHz. At 0.5 mHz it produces no T: its A
  and E match the truth to an unweighted mismatch of 3e-5 to 5e-5, but its T is zero where the
  truth's is 0.9 % of A, and the FD CSD gives T much weight (the data without T keep 0.64 of
  (d|d)). That costs 3.1e3 to 3.2e3 in ln L at the truth. At 20 mHz its A and E differ from the
  truth by an unweighted mismatch of 3e-4 to 1e-3, its T is 11 % low with the opposite sign (sky
  0), its amplitude 0.6 % to 1.8 % low and its phase 0.02 to 0.06 rad off; the largest errors are
  those of the strongly chirping sources. Its Python path equals its compiled kernel to 2.8e-10 of
  the peak, and its error hardly depends on N (ln L at the truth -36.0 to -36.2 at N = 8192,
  -35.8 to -36.0 at 131 072). The nominal arm length in its transfer function, its TDI 2 factor taken at f0, and its
  missing amplitude evolution are likely causes, not tested.
- FastGB's reference time, checked at the review: with t0 at the start of the orbit file and the
  parameters there, or with t0 at the data start and the parameters not carried, ln L at the truth
  falls to -1.1e4 at 20 mHz with the largest ḟ (template 22 bins off, ḟΔ/df). At 0.5 mHz, t0 at
  the start of the file takes it from -3.1e3 and -3.2e3 to -3.4e3 and -3.8e3, while not carrying
  the parameters changes nothing, since f0Δ is a whole number of cycles and ḟΔ is 3e-5 bins.
  Notebook: `$OUT/notebooks/fastgb_reference_time.ipynb`. `GBFDComputations` built with t_ref = 0 and its
  t_start set to the data start afterwards gives ln L at the truth within 0.001 of the carried
  version: its kernel takes the GB phase from t_ref.
- The FD CSD decides how much T counts at low frequency. At 0.5 mHz the FD data have an SNR of
  132 with the transfer functions averaged over the orbit, 182 with the CSD at the orbit-averaged
  arm lengths (the backend's default, which the global fit's FD path builds), against 100 in the
  STFT, about 104 without its window. Removing T from the data changes (d|d) by a factor of 0.64
  in the FD and 3.0 in the STFT, where the CSD per segment ties T to A and E. At 20 mHz T carries
  0.1 % of (d|d) in both domains, and the two FD CSDs agree to 0.01 %.

**Item 4 of the validation: six waveforms on the Mojito-light data** (2026-10-21). The fourth item
of the author's scope: the four waveforms of item 3, and the two of jaxgb, against the Mojito-light
simulation for its five galactic binaries of highest frequency. Each waveform runs in its own
domain against the transform of the same preprocessed data. The script gives the mismatch and the
SNRs at the true parameters and the one-parameter scans of item 3. The numbers are recorded; there
is no gate.

- Code: LAT b0cc2743 with the new `mojito_dffr_validation.py` and the part `mojito` of
  `run_step6.sh`; GBGPU 11d5159; jaxgb 0.2.2 and lisaorbits 3.1.0 from PyPI. The comps, FastGB and
  jaxgb ran on the A100 of the conventions above, the FD heterodyne on the CPUs of the same node.
- Data: the Mojito-light L1 files of the galaxy (`data/GB/L1`, the white-dwarf binaries of
  `catalogues/wdwd_cat_mojito_lite_processed.hdf5`, 15 539 324 of them) and of the instrument noise
  (`data/INSTRUMENT/L1`): X2, Y2 and Z2 every 2.5 s over 731 days, divided by the laser frequency.
  They are read and preprocessed as `globalfit/engine.py` does with the settings of
  `mojito_input/gb_high_freq_global_fit_settings.py`: `L1ProcessingStep`, a high-pass at 1e-5 Hz
  and a low-pass at 0.1 Hz (order-2 Butterworth, each run forward and backward), downsampling to
  5 s (`resample_poly` with a Kaiser window of β = 31) and 200 h trimmed from each end; then the
  first 1.92 years, where the settings file takes 9 months. Both domains use the 701 whole one-day
  segments of that span, which starts 720 850.5 s after the catalogue's reference time. (Corrected
  on 2026-10-22: this entry called 1.92 years the longest span the trim leaves. The trim leaves
  713.8 of the 730.5 days, 713 whole segments, which item 5 uses.)
- Orbits: LAT's `L1Orbits` from the noise file (ICRS, `MOJITO_AVERAGE_ARMLENGTH`), as the global
  fit loads them, for the comps, FastGB and the noise weights. The signal codes take the file's
  time-varying light travel times; the constant arm length enters only FastGB. jaxgb takes
  lisaorbits orbits: `InterpolatedOrbits` on the spacecraft positions and velocities of the same
  file, whose light travel times stay within 2.5e-7 s of the file's. The galaxy and the noise files
  hold the same positions, and light travel times within 2.5e-7 s of each other, 3e-8 rad of phase
  at 20 mHz.
- STFT: the periodic Tukey window of taper 0.1157 (1e4 s on a one-day segment) on the data and 60
  side bins, as in items 1 to 3, in place of the settings file's taper of 1/(14 mHz) = 71 s and the
  10 side bins of production. FD: dt rfft of the same 701 days with LAT's symmetric Tukey of taper
  1e4 s, as in item 3, on a band of 8192 bins about each carrier.
- Sources: the five binaries of highest gravitational-wave frequency, all between 15.6 and 20.4
  mHz, at the catalogue's amplitudes. Their parameters are those of the catalogue at its reference
  time (`MOJITO_REFERENCE_TIME`), mapped as the global fit maps them
  (`gb_catalogue_to_sampling_basis` and the stock GB transform: φ0 = +TrueAnomaly, sky and
  polarisation angles in ICRS). The SNR is DFFR's optimal SNR at the truth with the noise weights
  below; the gaps are those of the carriers in the middle of the span, in bins of 1/(1 day).

| Source | f [mHz] | ḟ [Hz/s] | A | SNR | Nearest catalogue binary |
| --- | --- | --- | --- | --- | --- |
| gb0 | 20.380 | 1.02e-13 | 1.37e-22 | 223 | gb1, 61.6 bins away |
| gb1 | 19.668 | 8.26e-14 | 1.36e-22 | 95 | gb0, 61.6 bins away |
| gb2 | 16.716 | 3.36e-14 | 1.15e-22 | 163 | gb3, 92.3 bins away |
| gb3 | 15.652 | -1.07e-13 | 4.26e-23 | 67 | gb4, 1.9 bins away |
| gb4 | 15.626 | 3.86e-14 | 1.57e-22 | 193 | gb3, 1.9 bins away |

- gb3 and gb4 lie inside each other's templates in the STFT, an overlapping pair, and 1595 bins
  apart in the FD. Of the 40 binaries of highest frequency, down to 10.9 mHz, only the three above
  16.7 mHz have no neighbour within their 60 side bins; each of the others has one with at least
  54 % of its amplitude.
- Waveforms, each in its own domain, on X, Y and Z with the full complex CSD: DFFR and TDI on the
  fly in `STFTGBComputations` with t_ref = `MOJITO_REFERENCE_TIME`, as the global fit sets it;
  `GBFDComputations` and FastGB with the parameters carried to the data start, FastGB with
  `flip_ref_phase=True`, as in item 3; JaxGB and JaxGBaccurate through jaxgb's `get_tdi` (TDI 2.0)
  with frequency and phase carried to the data start and the amplitude kept, since jaxgb has no
  amplitude evolution (as its `GBObject.to_jaxgb_array`). The FD heterodyne, FastGB and JaxGB give
  the transform of the data span without a window, as in item 3. JaxGBaccurate transforms a span
  twice as long, [t0 - T/2, t0 + 3T/2), under its own window, which is flat over the data; its
  template goes back to the time domain, where the data span takes the FD data window, as the data
  did.
- Noise weights: `XYZSensitivityBackend` at its default levels with the options of the settings
  file (TDI 2.0, transfer-function dips masked within 2 %, transfer functions averaged over the
  orbit in the FD) and, as its `normalize_window` asks, the CSD scaled by the mean square of the
  data window.
- Three data sets in every product: the galaxy plus the noise, the data of the global fit; the
  galaxy alone; and the galaxy without each source's neighbours, the catalogue binaries whose
  template reaches its STFT band. The neighbours leave the galaxy at their catalogue parameters:
  in the STFT as DFFR templates written by `fill_global_stft`, whose (h|h) and (d|h) equal those of
  `get_ll_stft` to 1.8e-15; in the FD as templates of `GBFDComputations` on 16 times their sample
  count, with the FD window applied in the time domain. In both domains the templates first take
  the gain of the preprocessing filters at their frequencies, as the neighbours in the data did.
- Metrics at the truth, against the galaxy without neighbours: the mismatch
  1 - Re(d|h)/√((d|d)(h|h)), the amplitude ratio √((h|h)/(d|d)), the phase of (d|h) and ln L; the
  optimal SNR √(h|h), the matched-filter SNR Re(d|h)/√(h|h) on the data, and the noise level
  (n|n)/(6N) of the noise alone over the N pixels or bins of the band, which is 1 when the CSD
  describes the noise. Scans as in item 3, on all three data sets: the eight parameters of the
  global fit, ±4σ in 81 points, σ = 1/√F_ii of DFFR's information matrix at the truth.
- The author's decisions of 2026-10-21: the settings of `gb_high_freq_global_fit_settings.py` with
  all of its preprocessing, over 1.92 years; the galaxy plus the noise as data; the STFT window of
  items 1 to 3 and 60 side bins; the catalogue's amplitudes; the five binaries of highest frequency,
  all of them, with gb3 and gb4 reported as an overlapping pair; FastGB with the conventions of
  item 3; and jaxgb's JaxGB and JaxGBaccurate as two more waveforms.
- Command, from the LAT root inside the job:
  `PYTHON=$PYTHON scripts/dffr/run_step6.sh --device gpu $OUT mojito`. The first run reads and
  preprocesses the files (44 s) into `$OUT/validation/mojito/data/`, which later runs reuse. The
  run took 11 min 44 s at a peak of 15.4 GB of memory; a source takes about 2 minutes, 1.5 of
  them for the FD heterodyne on the CPU.
  Files: `$OUT/validation/mojito/` (`report.txt`, `summary.json`, per source `NAME.json`,
  `NAME.npz` and `NAME.png`, `summary_offsets_{data,isolated}.png`,
  `summary_widths_{data,isolated}.png`, `summary_mismatch.png`); the review checks below in `dev/`.

Results. Mismatch at the truth against the galaxy without neighbours; for the FD heterodyne on
8192 points, and on 16 times as many in brackets.

| Source | DFFR | TDI on the fly, STFT | TDI on the fly, FD | FastGB | JaxGB | JaxGBaccurate |
| --- | --- | --- | --- | --- | --- | --- |
| gb0 | 3.1e-8 | 3.3e-8 | 1.4e-5 (7.6e-6) | 2.2e-3 | 2.2e-3 | 2.1e-3 |
| gb1 | 1.8e-7 | 1.8e-7 | 4.0e-5 (1.7e-5) | 2.5e-3 | 2.5e-3 | 2.4e-3 |
| gb2 | 5.8e-8 | 5.9e-8 | 1.1e-4 (2.2e-5) | 1.1e-3 | 1.1e-3 | 9.7e-4 |
| gb3 | 4.5e-7 | 4.5e-7 | 6.2e-5 (5.5e-5) | 7.6e-4 | 7.6e-4 | 6.8e-4 |
| gb4 | 5.4e-8 | 5.4e-8 | 1.9e-5 (3.3e-6) | 1.0e-3 | 1.0e-3 | 9.7e-4 |

Scans against the galaxy without neighbours: the largest peak offset over f0, ḟ, φ0, ψ, λ and β,
in σ, and the offset of A for DFFR.

| Source | DFFR | TDI on the fly, STFT | TDI on the fly, FD | FastGB | JaxGB | JaxGBaccurate | A, DFFR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gb0 | 0.003 | 0.001 | 0.004 | 0.88 | 1.05 | 0.96 | -0.177 |
| gb1 | 0.003 | 0.003 | 0.004 | 0.25 | 0.25 | 0.26 | -0.066 |
| gb2 | 0.002 | 0.001 | 0.010 | 1.32 | 1.33 | 1.27 | -0.059 |
| gb3 | 0.001 | 0.001 | 0.001 | 0.44 | 0.45 | 0.43 | -0.007 |
| gb4 | 0.002 | 0.001 | 0.002 | 1.05 | 1.09 | 0.90 | -0.056 |

- DFFR and TDI on the fly in the STFT agree to 2 % in mismatch at every source and put every peak
  in f0, ḟ, φ0, ψ, λ and β within 0.003σ of the truth, with widths of 0.997 to 1.000 σ. Their
  mismatch is 3.1e-8 to 5.8e-8 where no louder binary is near. Beside a louder one it rises to
  1.8e-7 (gb1) and 4.5e-7 (gb3): the neighbour leaves the data with DFFR's own error of 3e-8 to 5e-8
  of its power, which counts (SNR of the neighbour / SNR)² times in the weaker source's (d|d),
  5.5 and 8.3 times here.
- Amplitude: a waveform without the preprocessing filters sees the data lower by the low-pass's
  gain, 0.99932 at 20.4 mHz to 0.99977 at 15.6 mHz. DFFR's amplitude ratios, 1.00011 to 1.00079,
  are that inverse gain plus 6e-5 to 1.1e-4, or -1.3e-4 for gb3, whose ḟ is negative: the amplitude
  evolution A ∝ f^(2/3) of the model of LAT and GBGPU, which the Mojito binaries lack (review check
  4). In the scans this moves A, and ι through its correlation with A, by up to 0.18σ at gb0 (SNR
  223), the shift that the amplitude ratio predicts to three digits.
- The FD heterodyne: 1.4e-5 to 1.1e-4 on 8192 points and 3.3e-6 to 5.5e-5 on 16 times as many;
  most of the rest is the data window, which it does not model (with the window applied, review
  check 4, 3e-10 to 2.4e-9). Peaks within 0.010σ; ln L at the truth -0.28 to -2.9 on 8192 points
  and -0.12 to -0.60 on 16 times as many.
- FastGB, JaxGB and JaxGBaccurate: mismatch 6.8e-4 to 2.5e-3, peaks up to 1.3σ off (λ of gb2,
  β of gb4, f0 of gb0), ln L at the truth -3.1 to -109. JaxGBaccurate's time-varying arm lengths
  gain only 2 % to 11 % over JaxGB. Review check 4 traces this to their TDI step.
- The pair: with the neighbour left in the noiseless galaxy, the peaks of gb3 and gb4 move by at
  most 0.031σ in the STFT and 0.033σ in the FD, and the A of gb3 by 0.115σ in the STFT. Their STFT
  templates overlap by -5.6e-4 (normalised), a cross term of -7.3 in ln L; in the FD the cross
  terms are +0.35 and +0.58. For gb0 and gb1, 61.6 bins apart, the neighbour changes no peak in the
  third digit.
- With the noise the peaks move by up to 2.5σ (ι of gb4), alike for all six waveforms; the STFT
  and the FD weight the noise differently, so their offsets differ by up to 0.6σ. The
  widths stay within 0.973 and 1.031 σ. The noise level is 0.9968 to 0.9998 in the STFT and 0.9946
  to 1.0068 in the FD: the CSD at its default levels describes the Mojito noise to 0.7 % at these
  frequencies. No peak needed a rescan.

Review checks of the item, in `$OUT/validation/mojito/dev/`:

1. Neighbour templates without the preprocessing gain (`check_without_gain/`): the filtered
   remainder of the neighbours set the mismatch of gb1 to 1.9e-6 and of gb3 to 8.4e-7, against
   1.8e-7 and 4.5e-7 with the gain (`check_with_gain/`).
2. The preprocessing on a tone: the low-pass at 0.1 Hz, run forward and backward, scales the
   amplitude by 0.99937 at 20 mHz and 0.99712 at 29 mHz; the high-pass at 1e-5 Hz leaves it at 1 to
   six digits, and the downsampling filter changes it by at most 7.4e-6 up to 29 mHz. The
   processed sum of galaxy and noise and the sum of the processed parts agree to 2.4e-9 of the peak.
3. The taper of `gb_high_freq_global_fit_settings.py`: `window_taper_duration = 1/start_freq` is
   71 s for its band from 14 mHz, a Tukey α of 8.3e-4 on a one-day segment. On a pure tone it
   leaves up to 1.9e-2 of the power outside 10 side bins and 2.9e-3 outside 60 (a tone half a bin
   off; 3.6e-4 and 3.4e-4 on a bin), against 1.4e-3 and 1.1e-7 with the taper of 1e4 s. The author
   sets the file to 1e4 s (decision of 2026-10-21); this item ran with 1e4 s.
4. The Mojito signal chain (`jaxgb_tests/`, a script that the author keeps out of the repository;
   pytdi 2.2.1). On the raw 2.5 s data without filters, over the same span, in each source's FD
   band with the 1e4 s window:

   | Source | (a) JaxGBaccurate links vs the file's η_ij | (b) pytdi TDI 2.0 of those links vs the file's X2, Y2, Z2 | jaxgb's `get_tdi` | `GBFDComputations`, 16 times N, window applied |
   | --- | --- | --- | --- | --- |
   | gb0 | 7.8e-16 | 0 to rounding | 2.1e-3 | 1.6e-9 |
   | gb1 | 2.3e-14 | 2.6e-14 | 2.4e-3 | 1.2e-9 |
   | gb2 | 1.3e-12 | 8.5e-13 | 9.7e-4 | 3.0e-10 |
   | gb3 | 3.9e-12 | 5.6e-12 | 9.4e-4 | 2.4e-9 |
   | gb4 | 2.0e-12 | 3.0e-12 | 9.4e-4 | 2.4e-9 |

   Before the sources, pytdi's TDI 2.0 (`X2_ETA`, `Y2_ETA` and `Z2_ETA`, with the Doppler factors)
   of the file's own intermediary variables η_ij and light travel times gives the file's X2, Y2
   and Z2 to 3.3e-15 of their peak, so the check rebuilds the pipeline's TDI step. The mismatches
   are those of the table above; each source is modelled with its neighbours. The Mojito galaxy is
   thus JaxGBaccurate's link responses (column a, phases within 3e-9 rad, which also confirms the
   parameter mapping of the catalogue) combined by pytdi with the time-varying light travel times
   (column b). jaxgb's `get_tdi`, like FastGB, combines the links with one arm length, the mean
   light travel time (its `tdi.py`: "assuming equal armlengths"). Over the 701 days the six light
   travel times move within -0.081 s and +0.080 s of their mean of 8.3172 s, up to 1.0e-2 rad of
   delay phase at 20.4 mHz. `GBFDComputations`, which forms TDI 2.0 from the time-varying light
   travel times, gives 3e-10 to 2.4e-9 once the window is applied, with an amplitude 2.6e-5 to
   9.9e-5 high: the amplitude evolution of its model, which the Mojito binaries lack. pytdi ran in
   chunks of 2^20 samples, each with 400 samples of its neighbours on both sides, on 20 processes;
   the rebuild of the pipeline's TDI above shows that the chunks give the series of one evaluation.

**Item 5 of the validation: a global fit of the high-frequency galactic binaries with each
response** (2026-10-21 to 2026-10-23). The fifth item of the author's scope, in place of the first
plan's smoke run on the Mojito STFT settings: the parameter estimation of the galactic-binary block
of the global fit, started at the injected values, run once with TDI on the fly and once with
DFFR on the same data, settings and random seed, each for 1.45 days on one A100. The two posteriors
are compared with each other and with the injected values; the numbers are recorded, there is no
gate.

- Code: LAT 764a94ed with the new `scripts/dffr/gb_high_freq_globalfit_settings.py`,
  `run_globalfit.sbatch` and `globalfit_report.py` and the parts `globalfit_*` of `run_step6.sh`;
  GBGPU 11d5159.
- Data: the galaxy and the instrument noise of Mojito light, read and preprocessed as in item 4,
  but over all 713 whole one-day segments that the trim of 200 h at each end leaves (713.8 days),
  from 720 850.5 s after the catalogue's reference time. STFT: LAT's symmetric Tukey window of
  taper 1e4 s (α = 0.1157), as production; 16 side bins; noise weights of `XYZSensitivityBackend`
  at its default levels with the options of item 4, fixed (no noise branch).
- Grid and bands: the active bins run from 9.514 mHz (bin 822 of the 1/(1 day) grid) to 23.0 mHz.
  The band layout of the GB block gives 38 bands, 36 of them sampled (f0 from 9.578 to 22.898 mHz)
  and the two at the ends as buffers. The sampled range ends at the last band edge below the top of
  the grid, and the bands above 17 mHz are 3 to 11 mHz wide over 713 days: one bin more at the
  start moves that end to 19.92 mHz, and the settings file's grid of 10 to 22 mHz samples only up
  to 19.53 mHz, without the two binaries of highest frequency. The settings file therefore fixes
  the start by its bin and stops if a catalogue binary above 9.75 mHz lies outside the sampled
  range.
- Start: every catalogue binary in the sampled range, 56 (45 above 10 mHz), at its catalogue values
  in every walker and temperature, mapped as in item 4. A sky-averaged estimate with each binary's
  inclination gives optimal SNRs of 16 to 288; it reads 16 % and 19 % below DFFR's SNR for two
  binaries of item 4. 25 of the 36 sampled bands hold binaries, at most 4 each. The list of
  iteratively resolved binaries that the settings file reads is not on SURF Spider.
- Sampler: the production move `rj_prior` (`GBSpecialRJPriorMove`): each iteration, every leaf slot
  of every chain gets one birth or death proposal from the prior, and every live binary 50 in-model
  jumps along its Fisher matrix; then swaps between the temperatures of each band. 64 leaf slots,
  32 walkers, 24 temperatures (the settings file's ladder 1/1.2^k, the last at 1e-4), random seed
  5701424 in both runs.
- Batches: 64 cells (temperature, walker, band) per proposal batch and 2 (band, walker) rows of 24
  temperatures per tempering batch. A cell holds its band's window of 555 bins of the residual
  (18.1 MiB), of the inverse 3×3 CSD (54.3 MiB) and of a template twin (18.1 MiB). The settings
  file's 512 and 128 ran out of the A100's 40 GiB in the first test.
- Stop: a recipe step of the settings file ends a run after the first iteration that completes
  1.45 days after the start, inside the Slurm limit of 1.5 days.
- Test before the runs (job 41556413, A100 of wn-gb-02, 2026-10-21 23:47 to 2026-10-22 02:19,
  part `globalfit_smoke`, 3 iterations each): both responses kept all 56 binaries in every cold
  walker and stopped through the recipe step. TDI on the fly took 31.1 min per iteration (its
  in-model likelihood evaluations 1008 s, tempering 409 s), DFFR 18.3 min (301 s and 452 s); the
  GPU memory peaked at 31.2 and 28.6 GiB.
- Runs: jobs 41632927 (TDI on the fly) and 41632928 (DFFR), one A100 each on wn-gb-01, from
  2026-10-22 02:20. Over their first 7 and 14 iterations: 1894 s and 1028 s per iteration (the
  in-model likelihood evaluations 1003 s and 300 s), a GPU memory peak of 31.1 and 28.7 GiB.
- The author's decisions of 2026-10-21 and 2026-10-22: the settings of
  `gb_high_freq_global_fit_settings.py` with the taper of 1e4 s and 16 side bins for both runs; all
  713 segments; every binary above 9.75 mHz sampled, guards allowed, the upper limit raised until
  the two binaries of highest frequency lie inside; the start at the injection of every catalogue
  binary in the sampled range; 64 leaf slots with the reversible jump on; 32 walkers and 24
  temperatures; one A100 per run and 1.45 days of wall clock; the test of a few iterations on the
  development job before the runs; the run folders under `global_fit_runs/` with a copy of the
  settings file, the report in `$OUT/validation/globalfit/`.
- Commands, from the LAT root. Test, inside the job:
  `PYTHON=$PYTHON scripts/dffr/run_step6.sh --device gpu $OUT globalfit_smoke`. Runs, per response
  R in `tdi_on_the_fly` and `dffr`, after creating `<runs>/gb_high_freq_R`:
  `PYTHON=$PYTHON GLOBAL_FIT_RUNS_DIR=<runs> sbatch --job-name=gf_R
  --output=<runs>/gb_high_freq_R/slurm_%j.out scripts/dffr/run_globalfit.sbatch R $OUT`. Report:
  `python scripts/dffr/globalfit_report.py --runs <runs>/gb_high_freq_tdi_on_the_fly
  <runs>/gb_high_freq_dffr --output $OUT/validation/globalfit`.
  Files: the run folders `<runs>/gb_high_freq_{tdi_on_the_fly,dffr}/` (backend, logs, settings copy,
  GPU memory log), outside the repository; the test in `$OUT/validation/globalfit_smoke/`; the
  report in `$OUT/validation/globalfit/` (`report.txt`, `summary.json`, `health.png`,
  `speed_memory.png`, `convergence.png`, `truth_offsets.png`, `comparison_{bias,sd_ratio,ks}.png`,
  and per binary `traces/` and `corners/`), written by the closing chain `final_chain.sh` there.
- Metrics, cold chain only: each run's posterior per binary from the walkers after the first 25 %
  of its iterations, the leaves matched to the injected binaries by f0; (mean - injected)/σ per
  run; the two runs compared over the first iterations both reached, as the runs stop on a wall
  clock and not on an iteration count, by the difference of the means in σ of TDI on the fly, the
  ratio of the widths and a two-sample KS test of thinned samples; the MCMC noise of these from
  each run's integrated autocorrelation time. Convergence: the spread of the cold walkers per
  iteration, median over the binaries, and the dispersion of (mean - injected)/σ over the
  binaries, which is 1 for converged posteriors in data with noise.

Results. The runs stopped on 2026-10-23 at 13:25 (DFFR) and 13:41 (TDI on the fly). Run health
and speed, medians per iteration:

| | TDI on the fly | DFFR |
| --- | --- | --- |
| Iterations | 67 | 122 |
| Time per iteration | 31.6 min | 17.2 min |
| In-model likelihood evaluations | 1001 s | 298 s |
| In-model Fisher matrices | 22.4 s | 8.5 s |
| RJ step | 27.0 s | 10.2 s |
| Tempering | 454 s | 390 s |
| GPU memory, peak | 31.1 GiB | 28.7 GiB |
| Injected binaries alive in every cold walker | 56 of 56 at every iteration | 56 of 56 at every iteration |
| In-model acceptance, cold chain | 0.463 | 0.463 |
| ln L rebuilt after the move's bookkeeping drifted, times (largest drift) | 134 (233) | 244 (160) |

Comparison of the two posteriors over the first 67 iterations of each run (16 dropped), over the 56
binaries; the difference of the means in σ of TDI on the fly:

| Parameter | Median of the difference | Largest difference | Width ratio DFFR / TDI on the fly, median (5 % to 95 %) | KS above its 95 % level |
| --- | --- | --- | --- | --- |
| ln A | 0.033 | 0.20 | 0.998 (0.940 to 1.093) | 16 % |
| f0 | 0.065 | 0.23 | 0.993 (0.908 to 1.065) | 30 % |
| ḟ | 0.075 | 0.24 | 0.989 (0.901 to 1.070) | 32 % |
| φ0 | 0.059 | 0.22 | 0.998 (0.927 to 1.093) | 39 % |
| cos ι | 0.040 | 0.19 | 1.003 (0.950 to 1.081) | 14 % |
| ψ | 0.037 | 0.20 | 1.003 (0.923 to 1.092) | 25 % |
| α | 0.026 | 0.08 | 0.990 (0.938 to 1.026) | 5 % |
| sin δ | 0.023 | 0.10 | 0.996 (0.937 to 1.042) | 7 % |

Convergence, per parameter:

| Parameter | ln A | f0 | ḟ | φ0 | cos ι | ψ | α | sin δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spread growth over the last quarter of the first 67 iterations, TDI on the fly | 1.026 | 1.060 | 1.023 | 1.023 | 1.043 | 1.055 | 1.006 | 0.997 |
| The same, DFFR | 0.992 | 1.066 | 1.076 | 1.052 | 1.000 | 1.073 | 0.982 | 0.967 |
| Spread at iteration 122 over iteration 67, DFFR | 0.979 | 1.059 | 1.031 | 1.109 | 0.976 | 1.054 | 0.976 | 1.032 |
| Dispersion of (mean - injected)/σ, TDI on the fly | 0.99 | 0.69 | 0.61 | 0.74 | 0.95 | 0.59 | 0.92 | 1.05 |
| The same, DFFR | 1.05 | 0.81 | 0.73 | 0.80 | 1.00 | 0.59 | 0.94 | 1.04 |

- At equal iterations the two responses give the same posteriors within the sampling noise of the
  chains: the widths agree to 1 % at the median, the means differ by 0.02σ to 0.08σ at the median
  and by at most 0.24σ. In the sky angles, which have settled, the differences and the KS tests
  are what independent samples give (5 % to 7 % beyond the 95 % level). In f0, ḟ, φ0 and ψ, 25 % to
  45 % lie beyond twice their noise: the walkers, started at one point, still spread along these
  correlated directions (next block), and the noise from the autocorrelation time of a spreading
  chain is too small. The deterministic difference of the two responses is far below: item 3 puts
  their peaks within 0.003σ of each other.
- Both runs follow the same spread curves iteration by iteration. ln A, cos ι and the sky angles
  settle within about 30 iterations; f0, ḟ, φ0 and ψ still spread by 2 % to 8 % over the last
  quarter of the first 67 iterations and, in the DFFR run, by 3 % to 11 % more up to iteration 122.
  Their posterior means stay closer to the injection than converged posteriors do (dispersion 0.59
  to 0.81 instead of 1). The comparison therefore shows that DFFR gives the sampler the same
  likelihood as TDI on the fly, not converged posteriors of the phase parameters.
- A first look over each run's own iterations made DFFR 13 % to 15 % wider in f0, ḟ and φ0: an
  effect of its larger iteration count while the walkers still spread, gone at equal iterations.
- Speed: with identical settings an iteration takes 1.84 times less with DFFR, its in-model
  likelihood evaluations 3.4 times less and its Fisher matrices 2.6 times less. Tempering, mostly
  the building of the swap buffers, gains little (1.16).

Observations, alike in both runs (the galactic-binary block, not the response):

- Spurious binaries in the cold chains. The ḟ prior is one range for all bands, down to -4.7e-11
  Hz/s, while each band's store window holds the tracks of its own frequency. Prior births in hot
  chains with large negative ḟ have tracks that leave their window (1.1 % of the evaluated
  templates in both runs), and tempering swaps carry some into cold walkers: up to 4 (TDI on the
  fly) and 3 (DFFR) in one walker, with amplitudes at the prior's lower edge (6 to 8e-24) and ḟ of
  -1e-11 to -4.5e-11 Hz/s. After such swaps the move's rebuild of ln L differs from its own
  bookkeeping by up to 233 and 160, against 0.4 to 1 otherwise. The report matches leaves to the
  injected binaries and counts the others.
- Slow spread of the phase parameters: from a start at one point, the in-model jumps along each
  binary's Fisher matrix take more than 120 iterations of 50 jumps to spread the walkers over the
  posterior in f0, ḟ, φ0 and ψ, while ln A, cos ι and the sky angles settle within about 30. The
  author recorded this as a property of the move (2026-10-23) and closed the item on the runs as
  they are.
- Leaf order: the move writes the leaves of a walker back in band order, not in the order they
  were injected; the report matches leaves to binaries by f0, within 1e-5 mHz (the binaries lie
  8e-4 mHz apart or more, posterior widths of f0 are below 3e-7 mHz).

**Item 6 of the validation: the source-independent DFFR diagnostics leave LAT** (2026-10-22). The
three diagnostics of step 4 served the unit tests of the DFFR stages and the measured errors behind
them. LAT keeps no code that only those tests use, so they leave the repository, as the plan
foresaw; copies of the four files stay in the author's work folder
(`$OUT/removed/lat_dffr_diagnostics/`).

- Removed: `src/lisatools/cutils/lat_dffr_diagnostics.{hh,cu}`, their copy rule and their two
  entries in `cutils/CMakeLists.txt`; `STFTFresnelWrap.compute_dffr_sub_intervals` and
  `compute_dffr_windowed_kernels` (`binding_domains.hpp`) and
  `KinematicsTableWrap.compute_dffr_link_coefficients` (`binding_detector.hpp`), with their
  bindings in `binding_detector.cxx` and the type alias `kinematics_complex_array_type`, which only
  the last one used; `tests/test_dffr.py`, its test of `TDIConfig.generation` included;
  `scripts/dffr/dffr_unit_numbers.py` and part `unit_numbers` of `run_step4.sh`. The header comment
  of `lat_dffr_kernels.hh` no longer names the file.
- Kept: `dffr_link_coefficients` in `lat_dffr_kernels.hh`, which the DFFR kernels use, and
  `TDIConfig.generation`, which the comp needs; the DFFR tests of GBGPU still exercise it.
- Code: LAT 764a94ed with the removal (`$OUT/regression/step6_item6/item6.diff`) and GBGPU
  11d5159, built into the frozen environment `$OUT/item6_env` by `build_base_env.sh` (20 min). The
  reference is the shared environment, which still holds the diagnostics and is rebuilt only after
  the global-fit runs of item 5, which use it.
- Checks on the A100 of job 41556413 (wn-gb-02), both environments in turn; for the shared one the
  LAT tests of a clean clone at 764a94ed:

| Check | Shared environment (before) | Frozen environment (after) | Result |
| --- | --- | --- | --- |
| The three methods bound, CPU and CUDA 13 backends | all three on both | none | as intended |
| Golden outputs, existing response, CPU (6 configurations) | | 144 of 144 bit-identical | pass |
| The same on the GPU | | 141 of 144 bit-identical; `fill` within 4.2e-18 to 4.3e-17 of its peak | pass (`fill` held to 1e-15) |
| Golden outputs, DFFR, CPU (2 configurations on the ESA orbits; L1 orbits) | | 50 of 50 and 25 of 25 bit-identical | pass |
| The same on the GPU | | 48 of 50 and 24 of 25 bit-identical; `fill` within 4.3e-17 to 1.2e-16 of its peak | pass |
| LAT suite | 302 tests: 256 passed, 7 failed, 12 errors, 27 skipped | 292 tests: 246, 7, 12, 27 | the 10 tests of `test_dffr.py`, all passed before, are gone; no other outcome changed |
| GBGPU suite | 84 tests: 72 passed, 6 failed, 6 errors | the same | no outcome changed |

- The failures and errors are the same in both environments and do not touch the diagnostics:
  among them the six cross-domain tests of `test_stft_gb_crossdomain.py`, waveform tests of
  `test_gbgpu.py` and `test_thirdbody.py`, `DFFRGPUTest.test_compact_template_equals_fill_global`
  (recorded in part 4c), the flow tests of the unused `gbspecialstretch` family, and tests of
  `test_sensitivity.py`, `test_detector.py` and `test_response_alignment.py`.
- Command, inside the job: `$OUT/regression/step6_item6/item6_chain.sh` (it applies `item6.diff`
  to a clone of LAT at 764a94ed, builds the environment with `scripts/dffr/build_base_env.sh`, and
  compares with `stft_golden_outputs.py --compare-files` and the work folder's `test_outcomes.py`).
  Files: `$OUT/regression/step6_item6/` (`bindings.log`, `golden_*_after_vs_before.txt`,
  `test_outcomes.txt`, `before/` and `after/` with the captures and the suite logs).
- The shared environment, rebuilt after the runs of item 5 ended (2026-10-23, 13:48 to 14:03,
  `ONLY="lisa-analysis-tools GBGPU" ./install_erebor_spider.sh` in the closing chain of item 5),
  against the frozen environment: none of the three methods bound on either backend; the golden
  outputs bit-identical on the CPU (144 of 144, DFFR 50 of 50 and 25 of 25) and on the GPU but for
  `fill` (within 1.1e-16 of its peak); the LAT suite (292 tests) and the GBGPU suite (84) with the
  same outcome for every test. Files: `$OUT/regression/step6_item6/rebuilt/`.

## Step 7: review of documentation and scripts

Not started.

### Unit tests

### Regression tests

### Validation runs
