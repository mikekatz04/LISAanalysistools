# DFFR as a selectable response of STFTGBComputations: plan

Author: Coen Rondeel. The design below was agreed on 2026-10-16. This document is updated at the
end of every step and rewritten at the end of the effort to describe what was built. File and line
references point to LAT `l2d-dev` at 7c12a36d and GBGPU `l2d-dev` at ed22789, the base of the
branch `feat-dffr` in both repositories.

## Status

| Step | Scope | Status | Date | Notes |
| --- | --- | --- | --- | --- |
| 0 | Branches, documents, baseline of the existing response | Done, CPU and GPU, `L1Orbits` included | 2026-10-17, GPU on 2026-10-19 | The GPU part ran on a full A100 from a frozen environment of the step 0 code, which reproduces every CPU golden output bit for bit. On the GPU every output is reproducible bit for bit except `fill_global_stft`, whose atomic additions change their order from run to run (at most 1.1e-16 of the peak). At a year of one-day segments and 10 000 sources the existing response takes 66 µs per source in `get_ll_stft` and 61 µs in `fill_global_stft`. Results in `tests.md`. |
| 1 | Orbits cache | Done, CPU and GPU | 2026-10-17, GPU on 2026-10-19 | Existing outputs change by at most 6e-11 (templates, inner products), 2e-9 (gradients) and 2e-8 (Fisher on the TDI track, whose f̈ entries change by up to 8e-6); the step 1 captures are the new baseline. On the A100 the outputs change in the same arrays by the same amounts, and the CUDA kinematics table equals the CPU table bit for bit. Results in `tests.md`. |
| 2 | Fresnel split | Done, CPU and GPU | 2026-10-18, GPU on 2026-10-19 | Templates, inner products and the F-statistic change by at most 1.1e-14 of their peaks (gate 1e-12). Gradients and Fisher matrices change within the finite-difference noise that both codes carry, the criterion the author set for them in place of the gate. The step 2 captures are the new baseline. On the A100 the direct outputs, swap included, change by at most 7.2e-14, and gradients and Fisher matrices meet the same noise criterion; CUDA and CPU special functions agree to 3e-16. Results in `tests.md`. |
| 3 | Tile seam | Done, CPU and GPU | 2026-10-18, GPU on 2026-10-19 | Every output of the seven kernels, swap included, is bit-identical to the step 2 code on the CPU (144 and 26 arrays, against a frozen environment of the step 2 code), with the same test outcomes. CPU timings lie within run-to-run noise at 27 of 30 points; the other three differ by at most 0.7 %, within the drift between runs of identical code. ptxas gives every kernel at least the blocks per SM of step 2 on the A100 (sm_80) and the H100 (sm_90). On the A100 every output is bit-identical to step 2 except `fill_global_stft`, whose atomic additions round differently from run to run (at most 8.6e-17 of its peak, within the GPU criterion of 1e-15), with the same test outcomes. In the interleaved GPU timings of step 6, 27 of 35 points lie within the run-to-run noise and the other 8 within 0.7 %, accepted by the author as drift, as on the CPU. Results in `tests.md`. Corrected on 2026-10-18 in step 4: the register counts of step 3 were those of each kernel's own file; in the linked module every kernel has 244 registers on the A100 and 222 on the H100, in step 2 and step 3 alike, so 2 blocks per SM, and the conclusion holds. |
| 4 | DFFR kernels | 4a, 4b and 4c: done, CPU and GPU; the DFFR Fisher misses its A100 timing gate (step 6) | 2026-10-18, 4c on 2026-10-19, GPU on 2026-10-20 | The DFFR numerics, tile and galactic-binary model are built. The existing response stays bit-identical to step 3 on the CPU (144 and 26 arrays), with the same test outcomes and CPU timings within run-to-run noise or faster. Against standalone dffr on the same kinematics table, templates, global fill and segment table are bit-identical and the inner products agree to 6.3e-15. With the linear envelope of production, DFFR and the existing response differ by a median mismatch of 5.6e-10. Linked registers give the DFFR kernels 2 to 3 blocks of 128 threads per SM and the existing kernels 2; register caps and device link-time optimisation are timed in step 6. Part 4c, after step 5, adds the DFFR Fisher matrix as a kernel of its own: it matches a reference built from `fill_global` templates to 3.3e-15 of sqrt(F_ii F_jj), lies on the plateau of its steps (1.1e-5 against the criterion of 1e-3), and leaves every output of step 5 bit-identical. On the CPU it misses its timing gate at batch 64 by 13 % with the installed `-Os` build of GBGPU's CPU module, and takes 0.30 to 0.42 of the time of the existing Fisher with the module at `-O3`. On the A100 the existing response stays bit-identical to step 3 except `fill_global_stft` (rounding of its atomic additions); the exchange with standalone dffr on its GPU backend agrees within 1.7e-14; two DFFR CUDA tests that ask for CPU-level agreement fail and are recorded (author's decision). Part 4c on the A100 changes DFFR's swap, F-statistic and swap gradient at rounding level against step 5 (nvcc compiles an out-of-line function differently), which the author accepted. The DFFR Fisher misses its A100 gate from 1000 sources (1.34 to 1.47 times the existing Fisher on nine parameters, 1.06 to 1.09 at 1000 sources on eight); it is optimised in step 6. Results in `tests.md`. |
| 5 | Wiring into `STFTGBComputations` | Done, CPU and GPU; GPU timings in step 6 | 2026-10-19, GPU on 2026-10-20 | `STFTGBComputations(response="dffr")` runs DFFR in its six likelihood methods and gives the outputs of the DFFR entry points bit for bit; through it, the exchange with standalone dffr is bit-identical to step 4. The existing response stays bit-identical to step 4 (144 and 26 arrays), with the same test outcomes, and keeps its CPU timings: on the same binaries, the step 5 Python changes none of them measurably. With the envelope on, DFFR's gradient noise is unchanged in f0, ḟ, f̈, φ0, λ and β, so its gradients follow the group's envelope. On the CPU, DFFR takes 0.28 to 0.40 of the time of the existing response. On the A100 the existing response stays bit-identical to step 4 except `fill_global_stft`, DFFR through the comp is reproducible like the existing response and agrees with its CPU outputs to 8.0e-15 (templates and inner products), and the comparisons with standalone dffr and with the existing response give the CPU numbers. Results in `tests.md`. |
| 6 | Validation | Started: standalone dffr against the integration, GPU timings of the steps, the DFFR Fisher optimised, the occupancy A/B closed, all on the A100; validation items 1 (accuracy against a time-domain truth, window convention), 2 (the step 0 cases through both responses) and 3 (likelihood surfaces of four waveforms) done | 2026-10-19 to 2026-10-21 | The standalone package passes its own suite on the A100 (51 of 51) with a CUDA 13 backend added for this comparison. On the same inputs, a year of one-day segments and 10 000 sources on the ESA orbits, DFFR in `STFTGBComputations` takes 4 to 7 % more time per source than standalone dffr in `fill_global` and 6 to 13 % more in `get_ll`; (d\|h) and (h\|h) agree to 1.4e-15 of their peaks. In the interleaved GPU timings of the steps, DFFR is 2.1 to 8.8 times faster than the existing response in every likelihood method from 1000 sources. The DFFR Fisher, with its final reduction staged in dynamic shared memory and its tables built in waves straight into the Fisher tables, takes 0.51 to 0.63 of the time of the existing Fisher and meets its A100 gate at every batch size; its outputs are unchanged bit for bit on the CPU and the GPU, `fill_global` aside. The occupancy A/B changed nothing in the build: a register cap of 168 on `domains.cu` makes the existing response 4 to 15 % faster but two DFFR methods up to 1 % slower and moves inner products at rounding level, so the author left it out. Against a year of a time-domain truth (lisagwresponse and pytdi) at 0.5 and 20 mHz, with 60 side bins and the periodic window on the data, DFFR's templates reach a flat mismatch of 4.8e-10 to 6.8e-10 at 20 mHz, as the standalone package did in step 0, and at 0.5 mHz a noise-weighted mismatch 50 to 100 times below that of the existing response. LAT's symmetric data window adds a mismatch of 3.6e-8, which the author judged insignificant against the 6.4e-4 to 8.8e-4 that production's 10 side bins leave, so production keeps it. On the eight cases of the step 0 truth, DFFR in the comp reproduces the standalone package's accuracy to two or three digits. In one-parameter likelihood scans against a year of that truth at SNR 100, each waveform in its own domain with X, Y, Z and the full CSD, DFFR puts every peak within 0.002σ of the truth with the widths of its Fisher matrix; TDI on the fly stays within 0.016σ in the STFT and the FD; FastGB is off by up to 6.6σ, lacking the combination X + Y + Z, which carries much of the weight at 0.5 mHz. Results in `tests.md`. |
| 7 | Review of documentation and scripts | Not started | | |

## Context

`STFTGBComputations` (GBGPU `src/gbgpu/gbcomps.py:986`) computes the STFT likelihood of galactic
binaries with the `gb_stft_*` kernels, which LAT's `lat_stft_kernels.hh` templates on
`GBTDIonTheFly` and on a column policy (`FresnelColumn`). For every segment and channel it samples
the TDI on the fly three times and evaluates a seven-term windowed Fresnel transform per pixel.

The Direct Fast Fresnel Response (DFFR, standalone package `direct-fast-fresnel-response`)
computes the same pixels from a source-independent kinematics table at the segment midpoints:
three spacecraft reception states per pixel, folded with eighteen TDI weights per segment.
Measured: 6.5 µs per source and year at a band of 41 bins on one H100; median per-segment error
3.2e-5 to 4.7e-5 and whole-template mismatch 6.5e-10 to 1.2e-9 against a time-domain TDI 2.0
truth over 30 one-day segments on the ESA orbits that LAT ships.

Goal: DFFR becomes a response of `STFTGBComputations`, selected with one flag, built into the
existing STFT infrastructure (computation groups, `STFTDomain`, window starts, per-GPU replicas,
the likelihood consumer kernels), with no loss of accuracy or speed, and with a source-model
contract that can carry higher-order modes the way the TDI-on-the-fly family carries them.

The work runs in steps 0 to 7 below. Every step ends with regression tests, a review by the
author and an update of the documents in this folder. The changes land on the branch `feat-dffr`
of LAT and GBGPU.

## Documentation and scripts

| Path (in `lisa-analysis-tools`) | Content |
| --- | --- |
| `docs/response/dffr/README.md` | Overview for collaborators: the response, where its pieces live in LAT and GBGPU, how to select it, and links to the standalone package and its theory, conventions and validation documents. |
| `docs/response/dffr/plan.md` | This plan with a status table per step; updated at the end of every step and rewritten after the implementation to describe what was built and any deviations. |
| `docs/response/dffr/tests.md` | Per step, the unit tests, regression tests and validation runs performed: what was compared, on which backend, node and commits, the tolerances, the results, and the command that reproduces each. |
| `scripts/dffr/` | The scripts behind `tests.md`: golden-output capture, the exchange with the standalone dffr environment, benchmarks, validation runs. |

Unit tests stay in the test suites of each repository (LAT `tests/`, GBGPU `tests/`). Script
outputs (golden `.npz` files, timings, figures) go to a folder outside git, for the author
`~/projects/erebor/dffr_integration/` on SURF Spider; `tests.md` records the numbers.

## Design decisions (2026-10-16)

| Topic | Decision |
| --- | --- |
| Kernel structure | Tile seam: the `stft_*` consumers are generalised from per-thread columns to cooperative tiles; FresnelColumn keeps today's mapping; DFFR plugs in an optimised shared-memory tile. The Fisher stays per response. |
| Step order | The tile-seam refactor is its own step, before any DFFR code. |
| Orbits cache | 5-point stencils as `Orbits` methods in `Detector.cu` (`get_velocity`, `get_acceleration`, `get_ltt_rate`, `get_ltt_acceleration`) and `Orbits::get_kinematics_table`; `stft_get_spacecraft_acc` switches to them. No Python API for these methods. |
| Step size | Multiple of each grid's spacing closest to 2000 s, per grid; centre clamped so all five samples stay inside the table. |
| Table holder | `STFTComputationGroup` owns the table as xp arrays on its device and holds a LAT `KinematicsTableWrap`; the comp passes its `cpp_orbits` at call time. |
| TDI weights | DFFR's closed forms for 2nd generation, plus a 1st-generation form, selected from the comp's TDIConfig; custom combinations raise; a test checks each form against the TDIConfig units. |
| Fresnel split | One shared LAT header for the special functions, with DFFR's numerics; STFTFresnel delegates to it. |
| Modes | Component axis in the source-model contract; only the GB model (one component) is built. No spline source in this effort. |
| Flag | `response="tdi_on_the_fly"` (default) or `"dffr"`, stored in the constructor kwargs; DFFR knobs `first_order_delays`, `chirp_rate_floor`. |
| Channels | XYZ only; DFFR raises on an AET domain. |
| Derivatives | For `response="dffr"`, gradients and the Fisher inherit the group's `linear_envelope` (a sentinel default keeps `False` for the existing response). |
| Oracle | Standalone dffr on the same kinematics arrays, the existing GBGPU accuracy tests, and dffr's time-domain truth. |
| Window | The data use LAT's symmetric Tukey window, while STFTFresnel and DFFR model the periodic one. The effect, estimated at 5e-5 per segment and so comparable to DFFR's own error, is measured in step 6 before either side changes. Measured in step 6: 6.3e-5 per segment and a mismatch of 3.6e-8; the author kept both sides (2026-10-21), since production's 10 side bins leave a mismatch four orders of magnitude larger, and the validation runs on data with the periodic window. |
| Documentation | README, plan and tests record in LAT `docs/response/dffr/`, scripts in LAT `scripts/dffr/`, updated every step; a final step reviews both. |

## Findings the design relies on

- Orbits on l2d-dev (`Detector.hpp:117`, `Detector.cu:83-172`): separate position grid
  (`sc_t0/sc_dt/sc_N`) and light-travel-time grid (`ltt_t0/ltt_dt/ltt_N`), linear interpolation,
  times in the orbit frame (absolute TCB for `L1Orbits`). Grids: 50 s for the base orbit classes;
  500 s positions and native light travel times (about 2.5 s) for `L1Orbits`. Standalone DFFR
  builds against the released 1.2.8 class (one grid, times relative to the file start).
- The ESA orbit file that LAT ships starts at t0 = 2 073 211 130.8 s (TCB) and holds 1721
  samples 161 790 s (1.87 days) apart, 8.8 years; LAT's base classes count times from its start.
  The Mojito L1 file holds positions every 500 000 s (5.8 days) and light travel times every
  2.5 s.
- The only ad-hoc orbit derivative in the C++ of LAT, GBGPU and BBHx is
  `stft_get_spacecraft_acc` (`lat_stft_kernels.hh:177`, 3-point, 2000 s).
- Production STFT (`mojito_input/galfor_psd_multigpu_stft_settings.py`, `recipe.py:1352`):
  one-day contiguous segments, Tukey alpha 0.1157, `n_side_bins=10`, XYZ, TDI 2nd generation,
  `L1Orbits` in ICRS, `t_ref = MOJITO_REFERENCE_TIME`, `use_midpoint=True`,
  `linear_envelope=True`. The comp's default TDIConfig is 1st generation.
- Conventions checked equal between DFFR and LAT/GBGPU: GB phase, amplitude growth and strain
  signs (`gb_tdi_on_the_fly.cu:52-92`, `lat_tdi_on_the_fly.cu:724`); STFT `dt * rfft(w * x)`
  with the Fourier origin at each segment start (`domains.py:843`); data and inverse-covariance
  layouts of `STFTDomain`; chirp-rate floor 1e-22.
- Multi-GPU: `STFTBandLikelihoodEngine` rebinds `comp.stft_comps` per split
  (`gb_likelihood.py:1092`) and resolves a per-device replica rebuilt from `comp.args` /
  `comp.kwargs` (`devicereplicas.py:311`). Band buffers share the time grid and differ only in
  their frequency window (`start_freq_inds`).
- The store windows already pad `(n_side_bins + guard)` bins plus a Doppler reach
  (`gbbandstructure.py:390`); DFFR's band centre lies within one bin of the current one.
- Build: GBGPU copy-compiles LAT's `Detector.cu`, `domains.cu`, `lat_tdi_on_the_fly.cu` and
  instantiates LAT's header-only kernel templates; LAT registers every shared wrap
  (`binding_detector.cxx:592`), GBGPU receives wrap pointers (`binding_gbgpu.cxx:343`).

## Architecture

### Where the pieces live

LAT (source-independent):

- `cutils/Detector.{hpp,cu}`: the four stencil methods, `KinematicsTable` (POD, CPU/GPU name
  alias), `Orbits::get_kinematics_table` (host launcher and three build kernels ported from
  dffr's `kinematics.cu`). Methods only: the `Orbits` layout, and so `OrbitsView`, is unchanged.
- LAT binding: `KinematicsTableWrap` (holds pointers to the group's arrays, method
  `build(OrbitsWrap*)`), registered in `pycppdetector`; listed in `cutils/__init__.py`.
- `domaincomputation.py`: `STFTComputationGroup.cpp_kinematics(cpp_orbits)` builds the table in
  the group's device context on first use and caches it.
- `cutils/lat_fresnel.hh` (new): Maclaurin series of C and S, auxiliary f and g and their
  slopes, with DFFR's numerics. `STFTFresnel` delegates to it.
- `cutils/lat_stft_kernels.hh`: tile contract, `ColumnTile<ColumnT>` adapter, tile-generic
  consumers; from step 4 also the band-bin-first pixel map (`StftBandBinMap<Rows>`) and the launch
  path above 48 kB of dynamic shared memory.
- `cutils/lat_dffr_kernels.hh` (step 4): DFFR settings, source-model contract, windowed
  transform, closed-form TDI weights of both generations with first-order delay terms, segment
  table, `DFFRTile<SourceT, Rows>` with its launch inputs, two diagnostics of a batch of sources
  (compact templates, segment table), and since part 4c the DFFR Fisher matrix
  (`dffr_information_matrix_kernel<SourceT, Rows>` and its host launcher).
- `cutils/lat_dffr_diagnostics.{hh,cu}` (step 4): three source-independent diagnostics (bounded
  sub-interval integral, windowed transform, closed-form TDI coefficients), methods of
  `STFTFresnelWrap` and `KinematicsTableWrap` for `tests/test_dffr.py`. They leave LAT after step
  6; a copy stays in the author's work folder.
- `response/tdiconfig.py`: `TDIConfig.generation`, 1 or 2 for the stock combinations, `None`
  otherwise, so that the comp can pick DFFR's closed form.

GBGPU (GB-specific):

- `cutils/gb_physics.hh` (step 4): the GB phase, frequency, frequency derivative and amplitude
  as inline free functions; `GBTDIonTheFly`'s `ucb_*` methods and the DFFR GB model both call
  them, so there is one source of truth.
- `cutils/gb_dffr_source.hh` (step 4): `GBDFFRSource`, the GB model (one component).
- `cutils/gb_dffr.{hh,cu}` (step 4): the entry points `gb_dffr_*_wrap`, free functions in a
  translation unit of their own, bound as the `gb_dffr_*` methods of `GBComputationGroupWrap` in
  `binding_gbgpu.{hpp,cxx}`. Nothing DFFR enters `gb_tdi_on_the_fly.{hh,cu}`.
- `gbcomps.py` (step 5): the `response` flag and the DFFR dispatch, of `information_matrix` too
  since part 4c.
- `cutils/gb_dffr_source.hh` and `cutils/gb_dffr.{hh,cu}` gain in part 4c the parameters that leave
  the reception states unchanged and the entry point `gb_dffr_information_matrix_wrap`.

### Tile contract (the seam)

A tile policy supplies: the shared bytes per row and the rows per pass for a given number of
templates; a cooperative `build` over a block of rows (barriers allowed); the band start of each
row; the pixel mapping (one row per thread, or band bin first); and `values` returning all three
channels of one pixel. `ColumnTile<FresnelColumn>` reproduces today's mapping (thread `tid` owns
rows `tid, tid + 128, ...`, all side bins, same summation order), so the existing outputs stay
bit-identical. `DFFRTile` holds 32-row segment tables in shared memory and maps a warp to one
band bin. The 32 is the warp size, so that the 32 threads of a warp take the 32 segments of one
band bin; 32 rows of 1012 bytes also fit the 48 kB of shared memory a launch gets by default.
`get_ll`, `fill_global`, `swap_ll`, the F-statistic and both gradients become tile-generic; the
existing Fisher kernel stays as it is for FresnelColumn. The DFFR Fisher (part 4c) is a kernel of
its own in `lat_dffr_kernels.hh`, with the estimator of the existing one; whether one kernel can
serve both responses is decided after the A100 timings of step 6.

### Source-model contract (modes)

Per source: constants loaded once (sky basis, amplitude); a component count with a compile-time
maximum; per component a quadratic phase expansion at a retarded time (phase, frequency, rate,
amplitude ratio), the amplitude log-slope, and complex plus and cross polarisation weights. The
segment table and the pixel loop run over the components; a GB has one and pays nothing. This
mirrors TDI-on-the-fly, where each mode is a sub-source with its own amplitude, phase and angles,
but here the components of one source are summed per pixel, so `(h|h)` keeps the cross terms.

As built in step 4 (author's decision of 2026-10-18), the axis is in the contract only:
`GBDFFRSource` speaks in components (`num_components = 1`, component 0), while the segment table
and the pixel loop evaluate one component, and `DFFRTile` asserts `num_components == 1`. The
harmonics of a source lie hundreds of bins apart, outside one band of `2 n_side_bins + 1` bins, so
they belong in separate templates rather than in one table (`README.md`).

### Call path of a DFFR likelihood call

`comp.get_ll_stft` -> `self.stft_comps.cpp_kinematics(self.cpp_orbits)` ->
`GBComputationGroupWrap().gb_dffr_get_ll(d_h, h_h, kinematics, cpp_fresnel, cpp_domain, params,
data_index, noise_index, num_bin, nparams, t_ref, n_side_bins, tdi_generation, linear_envelope,
first_order_delays, chirp_rate_floor, start_freq_inds)` ->
`stft_get_ll_launch<DFFRTile<GBDFFRSource, 32>>` (one row on the CPU). Window settings come from
the bound `STFTFresnel`, the grid and data from `STFTDomain`, window starts from
`start_freq_inds` as today, and `linear_envelope` from the computation group. The dispatch in the
comp was built in step 5.

## Steps

### Step 0: branches, documents and baseline

- Local branches `feat-dffr` in LAT (from `l2d-dev` 7c12a36d) and GBGPU (from `l2d-dev`
  ed22789).
- On LAT `feat-dffr`: `docs/response/dffr/` with `README.md`, this plan and `tests.md`, and
  `scripts/dffr/`.
- An output folder outside git with `regression/`, `benchmarks/` and `validation/`.
- Baseline of the existing response, before any code changes: golden outputs of every
  `gb_stft_*` method (CPU and GPU) on fixed fixtures; timings of every method against batch size;
  the full GBGPU and LAT test suites, recorded.

### Step 1: orbits cache

Done on the CPU on 2026-10-17 and on the GPU on 2026-10-19.

- `Orbits::get_velocity`, `get_acceleration`, `get_ltt_rate` and `get_ltt_acceleration`
  (`Detector.cu`): five-point central differences of the linearly interpolated tables. Each grid
  uses the whole number of its cells closest to 2000 s (`ORBIT_STENCIL_STEP`): 4 cells of the
  500 s position grid, 800 cells of the 2.5 s light-travel-time grid of `L1Orbits`. With whole
  cells, the five samples share one position within their cells, so the interpolation error of
  the table cancels in the difference. The stencil centre is clamped so that every sample lies
  between t0 and t0 + (N - 2) dt of its grid.
- `KinematicsTable` and `Orbits::get_kinematics_table`: the three build kernels of dffr's
  `kinematics.cu`, with the four methods as stencils, absolute times in the orbit frame, and the
  link order, receivers and senders of the `Orbits` link arrays. A host-side check raises
  `ValueError` when the midpoints, widened by two stencil steps (and by 20 s of light travel time
  for the senders), leave the position or the light-travel-time grid.
- `KinematicsTableWrap` in `binding_detector.{hpp,cxx}`: it holds the table by value, takes
  C-contiguous float64 arrays without conversion and checks their lengths on both backends,
  because it keeps raw pointers. It is a required field of `LISAToolsBackendMethods`; the GBGPU
  and BBHx loaders fill every field of that table themselves, so each needs one line that passes
  it, as for the STFT domain wraps. GBGPU `feat-dffr` has that line. BBHx does not: the line was
  applied only to the local BBHx clone used for the tests, so BBHx fails to load against this
  branch until its maintainers add it.
- `STFTComputationGroup.cpp_kinematics(cpp_orbits)` builds the table at the group's segment
  midpoints, on the group's device, on the first call and keeps it; another orbits object gets
  its own table. `kinematics_fields` returns the arrays with their natural shapes, for tests and
  for the exchange with standalone dffr.
- `stft_get_spacecraft_acc` (three points, 2000 s) is removed; the TDI-phase frequency estimator
  calls `Orbits::get_acceleration`. The truncation error of the old difference, (Omega h)^2 / 12
  = 1.3e-8 of the annual acceleration, is gone; what remains is the rounding of the absolute
  positions, about 5e-9.
- Deviations: `get_window` is not changed in this effort; its last accepted cell reads one row
  past a table, and no new method reaches that cell. The NumPy mirror is compared
  within a rounding bound instead of bit for bit, because the CPU backend is built for x86-64-v3,
  where GCC fuses multiply-adds that NumPy rounds separately. The exchange with standalone dffr
  uses both of its builders: the C builder with the same stencil checks the port, the lisaorbits
  builder the physics.
- Tests: `tests/test_orbit_kinematics.py` (synthetic orbits on two offset grids against exact
  and analytic derivatives, a NumPy mirror, the ESA file velocities, coverage errors, argument
  checks, the group cache, CPU against GPU); regressions and the exchange with dffr in
  `scripts/dffr/run_step1.sh`. Results in `tests.md`.

### Step 2: Fresnel split

Done on the CPU on 2026-10-18 and on the GPU on 2026-10-19.

- `cutils/lat_fresnel.hh` (new), with the numerics of dffr's `fresnel.hh` at 01530a5:
  `lat_fresnel_series` (the Maclaurin series of C and S for |x| <= 1.6, 17 terms in x^4 summed by
  Horner, its coefficients in the constant bank on the GPU), `lat_fresnel_aux` (the auxiliary
  functions f and g and their derivatives: minimax rational fits for 1.6 < |x| < 8, a five-term
  asymptotic series above) and `lat_fresnel_integrals` (C and S for any x). The header records
  the definitions, references, the accuracy of each branch, its assumptions and its costs.
- The former evaluator of `domains.cu` used the same series and the same rational fits, so the
  accuracy is unchanged and the values differ by rounding only. DFFR's evaluation is cheaper: two
  divisions instead of five on the rational branch, one instead of two and no separate derivative
  polynomials on the asymptotic branch, and a fixed Horner sum instead of a loop with four
  divisions per term in the series.
- `STFTFresnel::get_fresnel_integrals` and `get_fresnel_aux` keep their signatures and delegate
  to the header. `get_phase_kernel_core` calls `lat_fresnel_series` and `lat_fresnel_aux`
  directly: it has already tested the seam |v| > 1.6, and the full integral would test it again
  and carry the auxiliary branch into the series branch.
- `STFTFresnelWrap.compute_fresnel_functions` (new diagnostic): C, S, f, g and the derivatives of
  f and g at given arguments, through the two members. It is a method of an existing wrap class,
  so `LISAToolsBackendMethods` and the GBGPU and BBHx loaders are unchanged.
- The step 1 GPU runs, and the step 1 GPU baseline that step 2 is compared with, need the step 1
  code after the rebuild, so they come from a frozen environment of it (`build_base_env.sh` with
  LAT 2863df4e and GBGPU 62c8d4c). It reproduces the step 1 CPU golden outputs bit for bit.
- Deviations: the gate of 1e-12 of each array's peak holds for every output computed directly
  from templates (at most 1.1e-14), but not for gradients and Fisher matrices (up to 1.8e-9 and
  9.6e-9). These are central differences of templates, which divide a change in rounding by
  their steps, and at their default steps they already carry noise, the same in both codes: up
  to 0.25 (f0), 1.8 (ḟ) and 13 (f̈) of each gradient component's scale, and up to 18 of
  sqrt(F_ii F_jj) in the f̈ entries of the Fisher matrix. On 2026-10-18 the author set their
  criterion instead: the change
  must stay within each code's own finite-difference noise, per gradient component and Fisher
  entry (`scripts/dffr/finite_difference_steps.py`). The references of f, g and their
  derivatives are quadratures of their integral representations (DLMF 7.7), which have smooth,
  positive integrands; `scipy` has no f and g.
- Tests: `tests/test_fresnel.py`; regressions and benchmarks in `scripts/dffr/run_step2.sh`.
  Results in `tests.md`.

### Step 3: tile seam

Done on the CPU on 2026-10-18 and on the GPU on 2026-10-19; its GPU timings are compared in step 6.

- Only `cutils/lat_stft_kernels.hh` changes; GBGPU is rebuilt, not changed. The consumers
  `get_ll`, `fill_global`, `swap_ll`, the F-statistic and both gradients are templates on one tile
  type and are launched by `stft_*_launch<TileT>`. They no longer name the source object, the link
  tables, the sky vectors, `T`, `t_ref`, the window factor or the frequency track. The contract
  (`Inputs`, `BlockShared`, `table_bytes`, `Map`, `Context`, `Template`) is listed at the seam in
  the header and explained in `README.md`.
- `ColumnTile<ColumnT>` adapts the column policy. A pass holds one segment per thread
  (`rows_per_pass` is the block size), and its pixel map, `StftRowPerThreadMap`, gives each thread
  the band of its own segment in increasing frequency. The column states stay in registers, so a
  launch reserves no dynamic shared memory. `FresnelColumn` only gains the alias `Source`.
- The seven `stft_*_impl<SourceT, ColumnT = FresnelColumn<SourceT>>` keep their signatures. Six
  resolve the device copies of the structs as before (`stft_column_tile_inputs`) and call the
  launchers with `ColumnTile`; `stft_information_matrix_impl` and its kernel are unchanged. The
  launchers pass the tile's dynamic shared memory, `table_bytes()` per template, to the launch,
  and on the CPU a host buffer of that size.
- Why the outputs stay bit-identical: each thread evaluates the same segments, side bins and
  channels in the same order as before, with the same calls and arguments, and the consumers keep
  the form of every floating-point expression (`t0 + time_i * dt`, `f_min + freq_j_here * df`,
  the factors 0.5 and `factor * 0.5`, the scaling by 4 df, the gradient differences). The Fresnel
  values and the inner-product sums are functions of `domains.cu`, which GBGPU compiles as a
  separate translation unit on both backends (separate compilation without device link-time
  optimisation on the GPU, no link-time optimisation on the CPU), so the restructured callers
  cannot change their rounding.
- Resources. Three choices keep the kernels at the cost of step 2, as the first version of the
  seam did not. The templates are locals of the evaluation functions, as the column states were,
  so their state is dead between the repeated evaluations of the gradient and F-statistic
  kernels; held in the kernels, they raised the two gradient kernels from 128 and 168 to 170 and
  232 registers (ptxas, sm_80). For the same reason the fill kernel constructs its template once
  per source: held for the whole kernel, it needed 129 registers on the H100 (sm_90), one above
  the 128 that allow 4 blocks of 128 threads per SM. The launch settings reach `Template::build`
  as arguments, not as members of the context; as members, the gradient kernel of `get_ll` needed
  129 registers, which on sm_80 allows 3 blocks of 128 threads per SM instead of 4. On the CPU,
  `Template::values` is forced inline (`STFT_TILE_PIXEL_INLINE`): GBGPU's CPU module compiles
  with `-Os`, where GCC kept it out of line, one more call per pixel, which made `get_ll_stft`
  0.5 % to 1.1 % slower.
  Correction of 2026-10-18 (step 4): the register counts above are those of each kernel's own
  file, as ptxas reports them. With separate compilation, a kernel that calls a function of another
  file gets, at the device link, the largest count among itself and the functions it calls. Every
  kernel of the existing response calls `STFTFresnelGPU::get_fourier_value` of `domains.cu`, which
  uses 244 registers on the A100 and 222 on the H100. In the linked module every kernel has those
  counts, in step 2 and step 3 alike, and runs 2 blocks of 128 threads per SM. The three choices
  therefore did not change the occupancy of step 3. They keep the kernels' own counts low, which
  matters once the count of the Fresnel function drops (step 6). `kernel_resources.py` now reads
  the linked module (`tests.md`, step 4).
- Deviations, decided by the author on 2026-10-18. The band-bin-first pixel map and the launch
  path for more than 48 kB of dynamic shared memory move to step 4, with the DFFR tile as their
  first user, so that every line of this step runs in its tests. The swap kernels, which this step
  rewrites, are part of the CPU gate after all: the frozen step 2 code captures them again on the
  CPU (`run_step3.sh`, part `reference`), because the GPU nodes are still down. The timing
  criterion is per point: the ratio of the median times must lie within the larger run-to-run
  spread of the step 2 and step 3 runs, read against the step 2 runs of 2026-10-18. Two smaller
  ones: the contract asks for the carrier bin of each row (`carrier_bin`), not the band start,
  because the consumers build both the single band and the union band of `swap_ll` from the
  carriers, as before; and the DFFR entry points of step 4 call `stft_*_launch<DFFRTile<...>>`,
  not `stft_*_impl`.
- Tests: no new unit tests, since the step adds no numerics. The gate is the bit identity of every
  output of the seven kernels against the frozen step 2 code, the swap kernels included, plus the
  column-policy oracle, the accuracy test and both test suites, and the CPU timings. The registers
  of every kernel for the A100 (sm_80) and the H100 (sm_90) come from ptxas
  (`scripts/dffr/kernel_resources.py`), since the GPU timings depend on them and no GPU node is
  up. Results in `tests.md`;
  `scripts/dffr/run_step3.sh`.

### Step 4: DFFR kernels

Parts 4a and 4b done on the CPU on 2026-10-18, and part 4c, the DFFR Fisher, on 2026-10-19 after
step 5; the GPU runs followed on 2026-10-20. The author split the step into 4a (LAT core), 4b (pixel map,
launch path, tile and galactic-binary model), 4c (Fisher) and 4d (documents, scripts, commits),
with a review after each.

- 4a, in `cutils/lat_dffr_kernels.hh`, ported from dffr 01530a5 with the same operations in the
  same order:
  - settings from `STFTFresnel` and the knobs, with host checks: `df dt = 1` to 1e-12, a taper in
    [0, 1], generation 1 or 2, a positive chirp-rate floor;
  - a check of the kinematics table against the grid: segment count, first midpoint t0 + dt/2 to
    1e-6 s, spacing;
  - the bounded sub-interval integral and the windowed transform, on `lat_fresnel.hh`;
  - the closed-form TDI coefficients of both generations, with first-order delay terms; the 2nd
    generation multiplies each channel pair by 1 - R R;
  - the segment table, 1012 bytes per row, and the pixel.

  `TDIConfig.generation` names the stock combinations. Three diagnostics on the existing wraps
  (`lat_dffr_diagnostics.{hh,cu}`) feed `tests/test_dffr.py`.
- 4b, three pieces:
  - In `lat_stft_kernels.hh`: the band-bin-first pixel map `StftBandBinMap<Rows>`;
    `stft_prepare_dynamic_shared`, which raises a kernel's limit on dynamic shared memory above
    48 kB within the device's opt-in limit (`cudaDevAttrMaxSharedMemoryPerBlockOptin`) and raises
    `std::runtime_error` beyond it; and `stft_device_domain`, the device copy of a domain for
    tiles that read no other struct.
  - In `lat_dffr_kernels.hh`: `DFFRTile<SourceT, Rows>`; `dffr_tile_inputs`, which raises
    `std::invalid_argument` for a domain other than XYZ, or for a Fresnel object and a domain of
    different grids; and the two diagnostics of a batch of sources.
  - In GBGPU: `gb_physics.hh`, `GBDFFRSource` and the `gb_dffr_*` entry points.

  One template takes 32 672 bytes at 32 rows. The two-template consumers (swap, its gradient and
  the F-statistic) are built for 32 and for 16 rows (65 344 and 32 960 bytes) and take
  `rows_per_pass`. `get_ll`, `fill_global` and the `get_ll` gradient use 32 rows.
- Deviations, decided by the author on 2026-10-18:
  - The DFFR Fisher moves to 4c. As planned then, it was to generalise the existing Fisher kernel
    instead of copying it. The derivative of a template splits into two parts:
    - a grid part, the three reception kernels and moments, which only f0, ḟ, f̈, λ and β change;
    - an amplitude part, the eighteen coefficients (TDI weights times reference phasors), which A,
      φ0, ι and ψ change.

    Those four parameters share the grid slot of the central template, so 21 grid slots replace
    36. The column Fisher must stay bit-identical. Part 4c below keeps the split but builds the
    DFFR Fisher as a kernel of its own.
  - Components: in the contract only (see Source-model contract above).
  - File layout:
    - one LAT header, `lat_dffr_kernels.hh`, holds the numerics, the tile and the batch
      diagnostics; at the author's request it absorbed a first `lat_dffr_tile.hh`;
    - GBGPU's entry points are free functions in `gb_dffr.{hh,cu}`;
    - the shared GB formulas are in `gb_physics.hh`.
  - The three LAT diagnostics stay until step 6, then leave LAT; a copy stays in the author's
    work folder.
  - Splitting `GBComputationGroup` out of `gb_tdi_on_the_fly.{hh,cu}` is left to the GBGPU
    maintainers (see Risks and observations).
  - At the author's request, `Template::values` and `build` were forced inline on the GPU. That
    cost the `get_ll` gradient one block per SM and gained none elsewhere (`tests.md`), so it was
    reverted.
- Resources (linked module, A100 and H100; details in `tests.md`):
  - the one-template DFFR kernels use 154 to 164 registers: 3 blocks of 128 threads per SM;
  - swap and the F-statistic use 220 and 222, set by `STFTDomainGPU::add_ip_swap_contrib` of
    `domains.cu`: 2 blocks;
  - the swap gradient uses 254, because nvcc keeps `values` and `build` out of line there: 2
    blocks;
  - the existing kernels use 244 and 222: 2 blocks.

  Two remedies, measured without a GPU, go to step 6:
  - capping the registers of `domains.cu` at 168 lifts every existing kernel to 3 blocks, with
    164 bytes of spills per Fresnel evaluation, and the DFFR swap on the H100;
  - device link-time optimisation lifts the DFFR kernels to 3 blocks, except the swap gradient on
    the H100, but leaves the existing kernels at 2 (255 registers).
- Tests: `tests/test_dffr.py` (LAT) and `tests/test_stft_gb_dffr.py` (GBGPU); the exchange with
  standalone dffr; DFFR against the existing response; and the bit identity of the existing
  response against the frozen step 3 code. Driver: `scripts/dffr/run_step4.sh`. The tests of the
  Fisher matrix are those of part 4c below. Results in `tests.md`.

#### Part 4c: the DFFR Fisher matrix

Done on the CPU on 2026-10-19, after step 5 and on its commits; the GPU runs followed on 2026-10-20. Decisions
of the author on 2026-10-19:

- A kernel of its own. `dffr_information_matrix_kernel<SourceT, Rows>` in `lat_dffr_kernels.hh`
  uses the estimator of the existing Fisher kernel: its parameter shifts, four-point stencil
  (two-point with `easy_central_difference`), common support of the templates of one derivative,
  cross-channel contraction and reduction. The existing Fisher kernel is not changed. Whether one
  kernel can serve both responses is decided after the A100 timings of step 6, because that tuning
  may still change the structure of the DFFR kernel. This replaces the plan of 2026-10-18, which
  had the DFFR Fisher generalise the existing kernel.
- Shared reception states. Each stencil template, a variant, gets its own segment table through
  one scratch table and keeps what its pixels read: its 3 reference phasors and 18 TDI weights, and
  its reception states unless the source model lists its parameter in
  `reception_invariant_params`. For a galactic binary these are A, φ0, ι and ψ, which enter only the
  link amplitudes and the phasors. Their variants share the reception states of the unperturbed
  source, so a pixel evaluates their windowed sums once: 17 sets of sums for 32 variants on the 8
  parameters of the global fit, 21 for 36 on 9.
- Each variant applies its own reference phasors and weights to the shared sums per pixel, in the
  order of the fill path, so every stencil template equals its `fill_global` template bit for bit.
  For this, `dffr_evaluate_windowed_kernel` and `dffr_evaluate_reception_state` are split before
  their final products (`dffr_windowed_sums`, `dffr_reception_sums`, `dffr_add_reception_state`),
  with the same operations in the same order; the DFFR outputs stay bit-identical.
- Compact tables in dynamic shared memory, sized at launch from the variants: 70 464 bytes at 4
  rows on 8 parameters and 81 280 on 9, about 16.5 and 19.2 kB per segment row. A pass holds 4
  rows on the GPU, or 8; both are instantiated, and `STFTGBComputations` passes 4
  (`_DFFR_FISHER_ROWS_PER_PASS`), because 8 rows of 9 parameters do not fit an A100 block. The CPU
  uses one row. The list of variants is built on the host from `inds` and the steps and passed to
  the kernel by value; on the GPU the launcher copies `inds` and `param_eps` from the device for it.
- Steps. The DFFR Fisher keeps the steps of the existing one: on the three XYZ production fixtures,
  every entry outside the row and column of f̈ changes by at most 1.1e-5 of sqrt(F_ii F_jj) between
  the default and the tenfold steps, against the criterion of 1e-3 the author set.
- `information_matrix` dispatches to the new entry point `gb_dffr_information_matrix` for
  `response="dffr"`. `astro_track` does not apply to DFFR's single chirp model, and
  `linear_envelope` defaults to the group's setting, as for the gradients.
- Resources (linked module): 170 registers on the A100 and 174 on the H100, a stack of 1848 bytes,
  no spills; with the shared memory, 2 blocks of 128 threads per SM at 4 rows on 8 parameters on
  both devices. The existing Fisher kernel has 244 and 222 registers and a stack of 14 048 bytes.
- CPU timing. The DFFR Fisher spends nearly all its time in windowed transforms, 63 per pixel on 9
  parameters, the existing one in its 36 TDI samplings per segment. With the installed build
  (`-Os`) the DFFR Fisher takes 0.82 to 0.84 of the time of the existing one at batch 1, whose
  only source sits at a window edge with 12 of its 21 bins inside, and 1.13 at batch 64: the CPU
  gate fails there. Compiled at `-O3`, the DFFR code runs 2.9 times faster and the existing code 4
  to 7 %; with the whole CPU module at `-O3` the DFFR Fisher takes 0.30 to 0.42 of the time of the
  existing one. The author decided to record both and keep the build in this part.
- Tests: `DFFRFisherTest` and the wiring test in GBGPU `tests/test_stft_gb_dffr.py`, the exact
  amplitude and phase entries of the existing response in `tests/test_stft_gb.py`, and a CUDA class
  for step 6. Driver: `scripts/dffr/run_step4c.sh`. Results in `tests.md`.
- Questions left open, and the accuracy of the Fisher matrix, which a separate validation takes up:
  see Risks and observations.

### Step 5: wiring

Done on the CPU on 2026-10-19 and on the GPU on 2026-10-20. On 2026-10-19 the author moved the wiring
before part 4c, the DFFR Fisher. Until 4c, `information_matrix` raised for DFFR; part 4c replaced
the error with the DFFR Fisher matrix.

- `STFTGBComputations` (`gbcomps.py`):
  - Constructor arguments `response` (`"tdi_on_the_fly"`, the default, or `"dffr"`),
    `first_order_delays` (default `True`) and `chirp_rate_floor` (default 1e-22 Hz/s). They are
    part of the recorded constructor arguments, from which the per-device replicas are rebuilt,
    so a replica keeps them.
  - Checks at construction, which raise `ValueError`: an unknown response; for DFFR, a
    `TDIConfig` with combinations other than the stock 1st or 2nd generation (also when the
    `TDIConfig` is set later), a bound group of another TDI set, and a floor that is not positive.
    The kernels keep their own checks on every call. The TDI generation is kept when the
    `TDIConfig` is set, because `TDIConfig.generation` compares the combinations on every read.
  - The six likelihood methods (`get_ll_stft`, `get_swap_ll_stft`, `get_fstat_ll_stft`,
    `fill_global_stft` and both gradients) call the `gb_dffr_*` entry points with the group's
    kinematics table (`cpp_kinematics(cpp_orbits)`), `t_ref`, `n_side_bins`, the TDI generation,
    the envelope, the delays, the floor and `start_freq_inds`. Their outputs and the attributes
    they store are those of the existing response, so the band engine is unchanged.
  - The envelope of a DFFR call is the group's `linear_envelope`, read as the existing response
    reads it: the group's attribute, else the comp's own argument. The comp cannot read it from
    the Fresnel object, because `STFTFresnelWrap` exposes none of its settings to Python. The
    envelope is an argument of the DFFR kernels, so a value given per call applies exactly,
    without the evaluator swap that the existing response needs.
  - `RESPONSE_DEFAULT` is the default `linear_envelope` of both gradients and the Fisher. It
    means `False` for the existing response, as before, and the group's setting for DFFR, as the
    design decided. `True`, `False` and `None` keep their meaning for both.
  - `information_matrix` raises `NotImplementedError` for DFFR, until part 4c.
- `recipe.py`: the optional keys `response`, `first_order_delays` and `chirp_rate_floor` of
  `gb_stft_kwargs`; without them the comp's defaults hold.
- Usage, settings, what DFFR does not read and its limits: `README.md`, "How to select it". The
  `STFTGBComputations` docstring points there.
- Deviations, decided by the author on 2026-10-19:
  - Tests. `test_stft_gb_dffr.py` calls the comp's methods, so the identities of step 4 now test
    the dispatch, and a new test compares every comp method with its entry point called with the
    expected arguments, bit for bit. `test_stft_gb.py` tests that the default response is
    unchanged. The plan had `test_stft_gb.py` run over both responses, which would have checked
    the DFFR identities twice.
  - Rows per pass. The comp passes 32 rows to the two-template DFFR kernels, as a class constant,
    not a setting: the choice matters on the GPU only, and only to the timing of step 6, which
    calls the entry points with 16 and 32 rows.
  - `window_factor` is not read by DFFR, as planned. `README.md` says what it does and when the
    two responses then differ. `use_midpoint` joins the settings DFFR does not read.
  - The pending step 4 GPU runs come from a frozen environment of the step 4 code, because step
    5 changes tests and scripts in the trees of the shared environment.

### Step 6: validation

- Accuracy against the brute-force STFT of a time-domain GB TDI injection built with the LAT
  window, both responses (`test_stft_gb_accuracy.py` recipe); size the window-convention error.
  Done on 2026-10-21 (`scripts/dffr/stft_accuracy.py`, `td_truth.py`, `run_step6.sh`; results in
  `tests.md`). The author's decisions: the injection comes from the truth chain of the
  standalone package (lisagwresponse and pytdi), after LAT's `pyResponseTDI` proved inaccurate
  below 5 mHz (its Lagrange weights come from a linearly interpolated table; recorded only); a
  year on the ESA orbits; the eight sources of the likelihood surfaces below; 60 side bins for
  every validation run, since under the production taper 10 side bins leave 1.2e-3 to 1.4e-3 of
  a tone's power outside the template and 60 leave 4e-8 to 1e-7; and the periodic window on the
  data of every validation run, while production keeps LAT's symmetric window.
- dffr time-domain truth (lisagwresponse, lisainstrument, pytdi) on the ESA trailing orbits, 30
  days to months, on a compute node, both responses.
  Done on 2026-10-21 on the cases and the grid of step 0, as the author defined the item after
  item 1 (`run_step6.sh` part `step0`; results in `tests.md`): over 30 days with the chain of
  step 0, DFFR in `STFTGBComputations` reproduces the numbers of the standalone package to two or
  three digits in all eight cases; the existing response has lower median errors and, at 20 mHz in
  sky 0, an 8 times larger mismatch. A year without lisainstrument gives the same picture.
- Likelihood surfaces against a full time-domain simulation (author's decision of 2026-10-21, in
  place of the cross-domain SNR^2 against FD and WDM and of the Mojito FD mismatch of the first
  plan). The surfaces of four waveforms for every parameter, on one year of data: FD GBGPU, FD
  TDI-on-the-fly, STFT TDI-on-the-fly and DFFR. Eight sources: one at low and one at high
  frequency, each at the largest and the smallest ḟ at that frequency, each at two sky positions.
  Fixed in item 1 (author's decisions of 2026-10-21): 0.5 and 20 mHz, the two bounds of the
  global fit's GB prior at that frequency (`get_fdot_mojito`), and the two sky positions of the
  step 0 truth.
  A surface is a scan of one parameter with every other parameter at its true value, repeated for
  each parameter, over a range of its own.
  Further decisions of the author (2026-10-21, after items 1 and 2):
  - No scan of the time-domain likelihood itself, whose cost per point is that of a full
    time-domain simulation: each waveform's surface is computed in its own domain against the
    transform of the same time-domain data.
  - The data are the noiseless time-domain truth of items 1 and 2 (lisagwresponse and pytdi, a
    year on the ESA orbits). The likelihood weights them with the noise of `XYZSensitivityBackend`
    at the global fit's default levels: optical metrology 15e-12 m/√Hz and test-mass acceleration
    3e-15 m s⁻²/√Hz, the noise injection of the global-fit settings and the defaults of
    `set_sensitivity_matrix`.
  - Every source at an optimal SNR of 100, by its amplitude.
  - The 8 parameters of the global fit: the nine of `GBTDIonTheFly` without f̈.
  - Each scan covers about 4 σ on both sides of the injection, with σ from DFFR's information
    matrix (`STFTGBComputations.information_matrix`) at the injection.
  - As in items 1 and 2: 60 side bins for the STFT waveforms and the periodic Tukey window on the
    STFT data.

  Done on 2026-10-21 (`scripts/dffr/likelihood_surfaces.py`, `run_step6.sh` part `surfaces`;
  results in `tests.md`). The author's decisions in the sub-plan and at the review of the first
  source:
  - FD GBGPU is FastGB (`gbgpu.gbgpu.GBGPU`), FD TDI on the fly is `GBFDComputations`.
  - The FD data carry the window of the global fit's FD path (LAT's symmetric Tukey, 1e4 s of
    taper), which neither FD code can sample, so both run without a window.
  - The FD codes sample the year on N = 8192 points (262 144 for the 20 mHz sources with the
    smallest ḟ), four times the length that holds the frequency track, after 2048 points left
    the FD heterodyne at up to 2.2 in ln L at the truth from its own sampling.
  - The FD likelihood weights with the CSD of `XYZSensitivityBackend` with its transfer functions
    averaged over the orbit, rather than the default CSD at the orbit-averaged arm lengths, which
    at 0.5 mHz makes T far quieter (FD SNR 182 against 132, and 100 in the STFT).
  - σ is the conditional width 1/√F_ii; the scans use the physical parameters.
  - Data and likelihoods use X, Y, Z and the full complex CSD everywhere. `GBFDComputations`'
    own kernel, which keeps only the real part of C⁻¹, is evaluated at the truth only.
  - FastGB takes its parameters at its t0, the data start, carried from t_ref = 0, with
    `flip_ref_phase=True`; the author checked the reference time at the review.

  Results: DFFR puts every peak within 0.002σ of the truth with the widths of its Fisher matrix,
  at all eight sources; TDI on the fly stays within 0.010σ in the STFT and 0.016σ in the FD;
  FastGB is off by up to 2.2σ, 5.7σ and 6.6σ at 20 mHz with the largest and the smallest ḟ and at
  0.5 mHz, where it lacks the combination T. The FD widths differ from the STFT ones by the
  information each domain holds (the STFT window, the weight of T).
- Likelihood surfaces of the same four waveforms against the Mojito-light data (in the data
  folder of `lisa_nlddpc`), for its five galactic binaries of highest frequency.
- Benchmarks per method and batch size, both responses, on a full A100 and on the CPU. Gate:
  DFFR is not slower than the existing response for any method the global fit calls. On
  2026-10-19 the author dropped the A100 MIG slice of the first plan: the global fit will run on
  several full A100s, so the code is optimised and judged on a full A100 first.
- Standalone dffr against the integration on the A100 (author's decisions of 2026-10-19): the
  standalone's own test suite, then `fill_global` and `get_ll` of both codes on the same inputs
  on the ESA orbits, in the setup of the standalone's benchmark (bands 41 and 21) and with the
  production settings, interleaved runs (`scripts/dffr/dffr_timing_lat.py`). The ratio of the
  times is recorded, not a gate; the occupancy runs below aim to close any gap. For it, standalone
  dffr gained a CUDA 13 backend, so that both codes are built by the same nvcc.
- On the GPU the `fill_global` grids of the two codes differ by up to 1e-12 of their peak over a
  year of 10 000 sources, a systematic difference, while the inner products agree to 1.4e-15.
  Before anything else is decided, the per-source templates of both codes are compared at 32, 120
  and 365 segments, to see whether the difference grows with the template phase (author's
  decision of 2026-10-19).
- Occupancy, from the linked registers of step 4 (author's decisions of 2026-10-18). A/B timings
  on an A100, for both responses; the faster settings go in:
  - `domains.cu` without a register cap, and capped at 168 and at 128. The cap changes the
    register allocation, not the arithmetic, so the outputs must stay bit-identical;
  - device link-time optimisation (`-dlto`). It needs `nvvm/lib64/libnvvm.so` next to nvlink,
    which the pip wheel of the CUDA toolkit does not hold (`tests.md`, step 4);
  - 16 against 32 rows for the two-template DFFR consumers;
  - `Template::values` and `build` of the DFFR tile forced inline on the GPU, or not.

  Done on 2026-10-21, after an estimate of each setting's gain from the linked registers. The
  author chose to time the cap at 168 only, with 16 rows inside both builds, on the ESA fixture
  at 10 000 sources, three runs in turn, and to adopt a setting only when it is faster at some
  point and slower at none, over every method of both responses. The cap at 128 adds spills
  without a further block, forced inlining costs the `get_ll` gradient a block, and device
  link-time optimisation would need the install to supply `libnvvm` for the pip toolkit. Results
  (`tests.md`, step 6): the cap makes every method of the existing response 4 % to 15 % faster,
  but the DFFR swap and swap gradient 0.5 % to 1 % slower; and it is not bit-identical, against
  the premise above: the compiler computes 12 products of `add_ip_contrib` apart from their
  additions, which moves the inner products of both responses by up to 4e-16 of their peaks and
  their gradients by up to 5e-10. 16 rows pay only for the DFFR swap with the cap. The author left
  the cap out, and the two-template DFFR consumers keep their 32 rows.
- The DFFR Fisher of part 4c on an A100: its CUDA tests (against the reference Fisher, against the
  CPU, 4 against 8 rows per pass) and the timing gate, which covers `information_matrix` since the
  GB special move calls it once per repeat block. The A100 timings of both Fisher kernels decide 4
  against 8 rows, and whether one Fisher kernel can serve both responses (author's decision of
  2026-10-19). The GPU runs of part 4c (2026-10-20) passed the CUDA tests and missed the timing
  gate from 1000 sources, so the DFFR Fisher is optimised before the occupancy runs below
  (author's decisions of 2026-10-20):
  - First, where its time goes: a side build with cycle counters around the table build, the band
    setup, the pixel loop and the final reduction, never committed (profiling counters of the GPU
    are open to administrators only on SURF Spider).
  - The final reduction no longer stages its sums in a static array of 9 kB. On nine parameters
    at 4 rows the kernel then needed 81 280 bytes of dynamic and 9 696 of static shared memory,
    more than the 82 944 bytes per block that 2 blocks per SM allow on an A100, so it ran 1 block
    per SM, the existing Fisher 2. With the sums staged in dynamic shared memory, the gate held at
    every batch size but 1000 sources, where it missed by 5 % to 8 %.
  - The table build straight into the Fisher tables, with all threads of the block. The table build
    took half of the kernel's time on the A100, one stencil template after the other through a
    scratch table (`tests.md`, step 6). The author's design, all templates of a pass at once and no
    scratch table, does not fit: besides the tables the build makes, per template, its shifted
    source constants and, per segment row, 6 emission states and 6 link amplitudes, 54 kB on nine
    parameters at 4 rows, where 5.4 kB are free before the second block per SM goes. Deviation,
    decided by the author on 2026-10-20: the templates are built in waves, whose intermediates sit
    in the space of the endpoint anchors. The anchors moved into an array of their own and are
    built after the last wave. The tile and the Fisher share one build (`dffr_build_rows`), the
    tile as the build of one template. The outputs stay bit-identical on the CPU and the GPU
    (`fill_global` aside), the build takes 12 % to 15 % of the kernel instead of half, and the DFFR
    Fisher takes 0.51 to 0.63 of the time of the existing Fisher, 1.9 times less than before from
    1000 sources.
  - 4 against 8 rows, timed with the gate in turn with the existing Fisher. 8 rows are faster up to
    100 sources (0.54 to 0.93 of the 4-row time) and slower from 1000 (1.25 to 1.32), because their
    tables leave room for one block per SM instead of two. Both meet the gate at every batch size.
    The author decided on 2026-10-21 that `STFTGBComputations` keeps its 4 rows for every batch.
- The three source-independent diagnostics of step 4 (`lat_dffr_diagnostics.{hh,cu}`, their wrap
  methods and `tests/test_dffr.py`) leave LAT after this step. A copy stays in the author's work
  folder.
- Global-fit smoke run on the Mojito STFT settings with `response="dffr"`: resolved-GB
  subtraction and log-likelihoods against the existing response; optionally, a short GB
  parameter-estimation comparison with trace and corner plots and a convergence table.

### Step 7: review of documentation and scripts

- `README.md` brought up to date with what was built, how to select the response, and its
  measured accuracy and speed.
- `plan.md` rewritten to describe the implementation as built: final file layout, interfaces,
  deviations from this plan and why, open items.
- `tests.md` completed: every test, regression and validation run with backend, node, commits,
  tolerances, results and the reproducing command.
- `scripts/dffr/` reviewed: each script runs from its documented command, writes to the output
  folder, and reproduces the numbers quoted in `tests.md`; scripts that served only one step
  are removed or folded into others.
- The documentation is checked against the code (names, paths, defaults) and against the
  project writing rules; the author's review closes the effort.

## Verification

- Rebuild after C++ changes, CPU and CUDA backends, on any compute node: nvcc comes with the
  environment, so the CUDA backends build without a GPU, and the install script then skips its
  GPU checks. The author's setup on SURF Spider:
  `ONLY="lisa-analysis-tools GBGPU" ./install_erebor_spider.sh` from `~/projects/erebor`.
- Every script in `scripts/dffr/` takes `--device cpu|gpu`, so a run made on the CPU repeats on
  a GPU with that one flag.
- From step 2 on, the CPU golden captures and CPU benchmarks leave out `get_swap_ll_stft` and
  `get_swap_ll_grad_stft` (`--without-swap` in both scripts). Both build their terms from the
  same column evaluation as `get_ll_stft` and `get_ll_grad_stft`, which stay, and
  `GBGPU/tests/test_stft_gb.py` keeps checking their own parts on the CPU: swap(A, A) against
  get_ll(A), the add and remove symmetry with the conjugated cross term, and the swap gradient
  against central differences. The two swap kernels took about 70 % of a CPU benchmark run. The
  GPU runs keep both kernels. Step 3, which rewrites the swap kernels, keeps them in its CPU
  golden captures and compares them with a capture of the frozen step 2 code; its CPU benchmarks
  still leave them out.
- A baseline that must come from the code before a step, but cannot be captured before the
  rebuild, comes from a frozen environment of that code (`scripts/dffr/build_base_env.sh`). Such
  an environment is accepted only if it reproduces the CPU golden outputs of that code bit for
  bit.
- When a step changes rounding on purpose, gradients and Fisher matrices are judged by
  `scripts/dffr/finite_difference_steps.py` against the frozen environment of the previous code:
  per gradient component and Fisher entry, the change must stay within each code's own
  finite-difference noise. Outputs computed directly from templates keep the tolerance of the
  step. Steps that must leave the existing response bit-identical (3 and 4) ask bit identity of
  every output. On the GPU, `fill_global_stft` adds with `atomicAdd` and is reproducible to
  rounding only, so there it must agree to 1e-15 of its peak instead (author's decision of
  2026-10-19).
- CPU tests: `python -m pytest GBGPU/tests/test_stft_gb.py GBGPU/tests/test_stft_gb_accuracy.py
  GBGPU/tests/test_stft_gb_crossdomain.py` and
  `python -m unittest tests.test_orbit_kinematics tests.test_fresnel` from the LAT root.
- GPU tests and benchmarks with `force_backend="gpu"`, on one full A100 (partition
  `gpu_a100_22c` of SURF Spider; the MIG slices were dropped, see step 6). Each step's GPU run
  keeps its own benchmark parts, but the GPU timings of different steps are compared through one
  interleaved round, which alternates the seven environments (steps 0 to 5 and part 4c), three
  runs each, on both orbit fixtures. Runs of one step after the other span many hours, and drift
  over that time would enter the comparison (author's decision of 2026-10-19).
- dffr environment: `cd direct-fast-fresnel-response && uv sync --extra compare`; the regression
  scripts in `scripts/dffr/` exchange arrays as `.npz` files in the output folder's
  `regression/`. The LAT side counts times from the start of the orbit file (`t0 = 0`, as the
  base orbit classes do); the dffr side adds the file's `t0` to whole-second offsets, which is
  exact, so both evaluate the orbits at the same times.
- Every step ends with its entries in `tests.md` (what ran, where, tolerances, results,
  reproducing command) and its status row in this plan.

## Risks and observations

- Window convention: the data use LAT's symmetric Tukey (`utils/utility.py:60`, used at
  `globalfit/engine.py:388`), while STFTFresnel and DFFR model the periodic window. Estimated
  effect of order 5e-5 per segment, comparable to DFFR's own error; measured in step 6, and any
  change is decided then. Measured in step 6 on a year of a time-domain truth: the symmetric window
  is the periodic one squeezed in time by (N - 1)/N, up to 1.5e-3 lower in the trailing taper; it
  changes every segment by 6.3e-5 of its peak, the amplitude by -2.9e-5 and the mismatch by
  3.6e-8, the same for every source. That is the largest error of both responses inside the band,
  but four orders of magnitude below what the 10 side bins of production leave (next entry). The
  author kept both sides as they are (2026-10-21); the validation runs on data with the periodic
  window.
- Side bins of production: with the Tukey taper 0.1157, the 10 side bins of the production
  templates leave 1.3e-3 (20 mHz) to 1.7e-3 (0.5 mHz) of a source's noise-weighted power outside
  the template, for both responses. That sets their mismatch against time-domain data at 6.4e-4
  to 8.8e-4, and the loss in ln L at the true parameters at 6 to 9 for a source of SNR 100
  (`tests.md`, step 6, item 1). A tone keeps 1.3e-6 to 5.8e-6 of its power beyond 30 bins and
  4e-8 to 1e-7 beyond 60. Recorded for the owners of the production settings; the validation
  uses 60 side bins.
- Negative-frequency image: both responses give the transform of the positive-frequency part of
  the signal. At 0.5 mHz, 60 side bins reach the start of the production band, 34 bins below the
  carrier, where the data hold the leakage of the image at minus the carrier frequency: a floor of
  1.0e-4 in the segment error and 2.7e-8 in the flat mismatch of both responses. With 10 side bins
  it does not reach the band. Noted, not changed; adding the image term would remove it.
- The combination X + Y + Z at low frequency: at 0.5 mHz it carries 0.9 % of the signal of A and
  far less noise, so the likelihood weights it heavily. The existing response gets it wrong by
  2.6e-3 to 3.7e-3 of its size, DFFR by 3.4e-4, and the existing response's noise-weighted
  mismatch against the truth is 50 to 100 times DFFR's (7.2e-6 to 1.4e-5 against 1.3e-7, with 60
  side bins). Recorded; the likelihood surfaces of the next validation items show what it does to
  parameter estimates.
- LAT's full time-domain response (`pyResponseTDI`) takes the factor A(e) of its fractional-delay
  Lagrange weights from a table of 1001 values with linear interpolation (`MAX_A_VALS` in
  `LISAResponse.cu`), which underestimates it by up to 2.1e-7. TDI 2 amplifies this at low
  frequency: against `GBTDIonTheFly` its X, Y and Z are off by 3.0e-3 of the peak at 0.5 mHz, 4.3e-5
  at 2 mHz and 3.4e-7 at 20 mHz. Found in step 6 and recorded only (author's decision); computing
  A(e) exactly in the two interpolation functions would remove it.
- FastGB (`gbgpu.gbgpu.GBGPU`) produces no combination T = (X + Y + Z)/√3 at 0.5 mHz and a T
  11 % low with the opposite sign at 20 mHz, while its A and E match a time-domain truth to
  mismatches of 3e-5 to 1e-3. With X, Y, Z and the full CSD that costs 3.1e3 to 3.2e3 in ln L at
  the truth of a 0.5 mHz source of SNR 100, and its peaks lie up to 6.6σ off (step 6, item 3).
  Recorded for the users of FastGB with XYZ likelihoods.
- `GBFDComputations` contracts its FD inner products with the real part of the XYZ inverse CSD
  only, which moved (h|h) by up to 2.9e-3 at 0.5 mHz against the full CSD (4.8e-3 with the CSD at
  the orbit-averaged arm lengths). Its constructor requires t_start = t_ref, while its kernel
  takes the GB phase from t_ref at absolute times: with t_start set apart afterwards it gave ln L
  within 0.001 of the version with the parameters carried to t_start. Recorded for its owners.
- `XYZSensitivityBackend` in the frequency domain: its default CSD, at the orbit-averaged arm
  lengths, makes T much quieter at low frequency than the CSD averaged over the orbit
  (`average_transfer_functions=True`): at 0.5 mHz the FD SNR of a source is 182 against 132, where
  the STFT's CSD per segment gives about 104. The global fit's FD path uses the default. Recorded.
- Sky frame of the shipped ESA file: LAT's `Orbits` docstring calls it ecliptic, the dffr docs
  call it ICRS. The regressions read raw file positions on both sides (`frame="ecliptic"`, no
  rotation); the question itself is checked in step 6.
- `Orbits::get_window` accepts the last cell of a table, index N - 1, where `interpolate` reads
  row N, one past the end of the array: a lookup in the final dt of a table returns a value
  computed from whatever memory follows the array (shown on the CPU backend with marker values
  placed after a table). The new methods and the coverage check stay below t0 + (N - 2) dt.
  `get_window` is left as it is in this effort; production data end well before the orbit files.
- The base orbit classes (`Orbits._configure`) append the file's last sample after their
  uniform 50 s grid, although the C++ class assumes row i at t0 + i dt. On the ESA file the last
  cell then spans 74.7 s instead of 50 s, and positions in it are off by up to about 740 km. With
  `t0` not zero, the same function builds its grid from t0 to about 2 t0 + span: 4.7e7 rows,
  about 16 GB, for the ESA file's own t0. Neither reaches the new methods; both are noted, not
  fixed.
- Orbit representation: the base classes and `L1Orbits` pass a cubic spline through the file's
  position samples (1.87 days apart on the ESA file, 5.8 days on the Mojito L1 file) and do not
  use the file velocities. The derivatives of the interpolated positions differ from the file
  velocities by 2.7e-7 (ESA) and 8.1e-6 (Mojito L1) of the orbital speed, and from the analytic
  accelerations of lisaorbits by 8.4e-5 (ESA). Both responses read these positions. On the ESA
  orbits, at 20 mHz, the position difference (up to 486 m) and the acceleration difference over
  half a day each shift the phase by about 2e-7 rad. Step 6 measures the effect against the
  time-domain truth. Measured in step 6 (item 1): at 20 mHz the truth, which interpolates the file
  with lisagwresponse, and `GBTDIonTheFly`, which reads LAT's orbits, agree to 3.3e-7 of the peak;
  at 0.5 mHz, where TDI 2 cancels most of the signal, they differ by 1.3e-5 to 1.7e-5, and both
  responses follow a day-to-day ripple with the 1.87-day sampling of the file. The orbit
  representation is the likely cause; not tested directly.
- GBT `CubicSplineSegment` derivative helpers return `c1 + c2 dx + c3 dx^2` and `c2 + c3 dx`,
  not the power-basis derivatives. Nothing calls them; noted only.
- At their default steps, the gradients and Fisher matrices of the existing response carry
  finite-difference noise, measured on the step 2 fixtures as the difference between the default
  and the tenfold step. Per gradient component, relative to its scale: about 1e-2 to 0.25 in f0,
  0.2 to 1.8 in ḟ and 0.6 to 13 in f̈. `information_matrix` notes the cause, roundoff in the
  Fresnel phase, and widens its f0 and ḟ steps 10 and 100 times for it; per Fisher entry,
  relative to sqrt(F_ii F_jj), the noise is then at most 5.2e-4 except in the row and column of
  f̈: up to 18 on its diagonal and 0.2 to 0.4 elsewhere. Noted, not changed. The DFFR Fisher of
  part 4c uses the same estimator and steps; on the three XYZ production fixtures, every entry
  outside the row and column of f̈ changes by at most 1.1e-5 of sqrt(F_ii F_jj) between the default
  and the tenfold steps, so the steps were kept for it (`tests.md`, part 4c).
- Shared memory: one DFFR template takes 32 672 bytes at 32 rows, so the two of swap (65 344
  bytes) exceed the 48 kB a launch gets on an A100 and on an H100. With the opt-in of step 4,
  which takes its limit from the device, an A100 allows 163 kB per block and 164 kB per SM, an
  H100 227 kB and 228 kB. The DFFR Fisher of part 4c holds the tables of every stencil template of
  a pass: about 16.5 kB per segment row on 8 parameters and 19.2 kB on 9, so a pass takes 4 rows,
  or 8 at one block per SM. Before the direct build of step 6, 8 rows did not fit 9 parameters on
  an A100; since then they take 154 kB and fit.
- Newer devices than the A100, the H100 (sm_90) in particular. The timing gate of this effort is
  measured on A100s, the GPUs of SURF Spider, while the standalone DFFR timings in the README were
  measured on an H100 NVL. Both devices have 65 536 registers per SM and at most 255 per thread,
  so their steps in occupancy fall at the same register counts (4 blocks of 128 threads up to 128
  registers, 3 blocks up to 168), and both have warps of 32 threads, so the 32 rows of the DFFR
  tile suit both. In the linked module, every STFT kernel of the existing response has 244
  registers on the A100 and 222 on the H100, in steps 2 to 4 alike, so 2 blocks per SM on both
  (step 3, correction). What should change for H100 nodes:
  - The build. The install script of SURF Spider compiles machine code for its own GPUs and PTX of
    the A100 generation. An H100 would run code that its driver translates from that PTX when the
    module loads. A build for H100 nodes adds 90 to `CUDA_ARCH`, so that the code comes from PTX
    of its own generation, the code that the ptxas check describes.
  - The checks. The same source gives different register counts on the two devices: in step 2
    already, the own code of `get_ll` used 120 registers on the A100 and 128 on the H100, and the
    linked kernel 244 and 222. A change of the kernels is therefore checked with
    `kernel_resources.py --arch 90` as well as `--arch 80`, which since step 4 compiles and links
    the whole module for that device. Claims of speed on an H100 need the benchmarks repeated on
    one (`--device gpu`).

  What could change, to use what an H100 has more of:
  - Shared memory: with the two templates of the DFFR swap at 32 rows (65 344 bytes, and about
    11 kB of static shared memory), an H100 fits 3 blocks per SM by shared memory where an A100
    fits 2; today the registers hold both at 2. The launch path of step 4 takes its limit from
    the device (`cudaDevAttrMaxSharedMemoryPerBlockOptin`).
  - Double precision: an H100 SM has 64 FP64 units against 32 on an A100, and the Fresnel and TDI
    arithmetic is all in double precision, so an H100 needs more work in flight per SM to keep
    those units busy. A launch bound that trades registers for blocks, or a block size other than
    the 128 threads of `NUM_THREADS_HERE`, which is one value for every device today, may pay on
    an H100 and not on an A100; to be measured.
  - The grid: one block per source, and 132 SMs on an H100 SXM against 108 on an A100, so at 4
    blocks per SM a batch needs about 530 sources instead of about 430 to occupy every SM.
  - H100 features that the kernels do not use: clusters of blocks with distributed shared memory
    would let the blocks of a cluster read each other's segment tables, for example the add and
    remove tables of swap.
- Bit identity on the GPU. `fill_global_stft` adds with `atomicAdd`, so on the GPU it is
  reproducible to rounding only, and the gate for it there is 1e-15 of its peak. And nvcc decides
  per kernel whether to inline a function and which products to fuse: in part 4c the split of the
  windowed sums left the CPU code alone but added two multiplies to the out-of-line
  `DFFRTile::Template::values` that the two-template kernels call, which moved their GPU outputs at
  rounding level (accepted by the author on 2026-10-20). Changes of the DFFR code are therefore
  judged on the GPU to rounding, and the GPU machine code of the affected kernels is compared
  when outputs move (`tests.md`, part 4c, "On the GPU"). For the same reason two CUDA tests of
  step 4 that ask for agreement at CPU level fail on the A100; they are recorded, not changed.
- Multi-device runs are not checked in this effort (author's decision of 2026-10-19): the LAT
  tests that need two CUDA devices, the developer checks behind `STFT_MULTI_GPU_CHECK`, and a
  DFFR replica on a second device, which builds its own kinematics table there. Step 5 checks the
  multi-GPU path of the replica builder on the CPU only (`tests.md`).
- Proposal for the GBGPU maintainers, recorded at the author's request (2026-10-18), not done:
  `GBComputationGroup`, whose 23 method definitions sit among the kernels of the existing response
  in `gb_tdi_on_the_fly.cu`, could move to a file of its own, generic over the response. The
  `gb_stft_*` and `gb_dffr_*` methods would then live in one group without either response's file
  holding the other's code. The class is under active development by others, so this effort binds
  the DFFR entry points as methods of the existing wrap and leaves the class where it is.
- GBGPU's CPU module, which holds the STFT kernels of the CPU backend, compiles with `-Os`:
  `nanobind_add_module` optimises a module's sources for size unless it is given `NOMINSIZE`,
  and GCC applies the last `-O` option, so the `-O3` of the release build has no effect there.
  The GPU code is compiled by nvcc at `-O3`. Noted, not changed: a change would move the CPU
  timings and could change rounding through other inlining. Measured in part 4c: at `-O3` the DFFR
  code runs 2.3 to 2.9 times faster and the existing response's Fisher 4 to 7 %, and the DFFR
  outputs move at rounding level (templates within 8e-15 of their peaks, gradients 6e-11). Whether
  the CPU module, or `gb_dffr.cu` alone, moves to `-O3` is open; it would replace the bit identity
  of the outputs against the earlier steps, and of DFFR against standalone dffr, by agreement to
  rounding.
- Open questions of the DFFR Fisher matrix (part 4c), left to later sessions by the author on
  2026-10-19. A validation of the Fisher matrix itself is a separate effort, not part of this
  integration.
  - Layout copy on the GPU. `dffr_information_matrix` builds the list of stencil templates on the
    host from `inds` and the steps. On the GPU it copies these two arrays, about 100 bytes, from
    the device on every call, which waits for the stream. Open: keep the copy, or pass host arrays
    through the binding.
  - Eight rows and nine parameters. At 8 segment rows per pass the tables of nine parameters took
    about 162 kB, which with the kernel's static shared memory exceeded the 163 kB an A100 block
    can hold, so that launch raised `std::runtime_error`. Resolved in step 6: the direct build
    removed the scratch table, and the launch now takes 154 kB. The error stays for layouts beyond
    a block of the device.
  - Table build on the GPU. The kernel built the table of one stencil template at a time, so at 4
    rows per pass the stages of a build gave 12 to 24 of the 128 threads of a block work, and the
    DFFR Fisher was 1.34 to 1.47 times slower than the existing Fisher from 1000 sources on nine
    parameters (A100, 2026-10-20). Resolved in step 6 by the staged reduction and the direct build
    in waves: the DFFR Fisher takes 0.51 to 0.63 of the time of the existing one.
  - Rows per pass of the DFFR Fisher. `STFTGBComputations` passes 4 rows. 8 rows are 7 % to 46 %
    faster while a batch has fewer sources than the device has SMs, and 25 % to 32 % slower from
    1000 sources (`tests.md`, step 6). The author decided on 2026-10-21 to keep 4 rows for every
    batch; a choice by batch size would need the device's SM count at launch.
  - Source-model contract. The list in `lat_dffr_kernels.hh` gives `load(Constants*, params)`; the
    source models also take the launch inputs `Model`, and the list names neither `Model` nor
    `num_params`.
  - Accuracy of the Fisher matrix, for the validation:
    - Phase-offset step. The step of 1e-6 rad is added to template phases of up to 1.4e4 rad on the
      test fixture of `test_stft_gb_dffr.py`, and of about 1e6 rad in production. Rounding the sum
      to a double changes the realised step by up to half a unit in the last place of the phase:
      9e-7 of the step on the test fixture, about 1e-4 in production. On the test fixture F_φ0φ0
      equals (h|h) to a relative difference of 1.2e-7 to 2.1e-7 for DFFR, and of 1.4e-6 for the
      existing response, whose TDI combination enlarges the rounding.
    - DFFR against the existing Fisher. On the test fixture, with the envelope off and the 8
      parameters of the global fit, the entries agree to 2.7e-3 of sqrt(F_ii F_jj) and the
      diagonals to 0.27 %. The difference of the models (the Doppler shift in DFFR's reception
      chirps, the astro track of the existing Fisher) and the finite-difference noise of each are
      not separated. A comparison of each Fisher with the curvature of its own likelihood,
      -2 Δln L against δᵀ F δ, would separate them.
    - Conditioning. On the same fixture the normalised Fisher matrix of one of the two sources has
      a condition number of 3e7 to 4e7 in both responses, against 1.6e5 for the other. Its marginal
      standard deviations from the inverse then differ by up to 17 % between the responses while
      its entries agree to 0.3 %. On the production fixtures the condition numbers reach 4e10.
    - Astro track of the existing Fisher. On the production fixtures, for the four sources the
      fixture places at the edges of their store windows, the template the existing Fisher
      differences has an (h|h) up to 1.4 % away from DFFR's, while DFFR and the existing response's
      production template agree to 4e-4: without the Doppler shift in its carrier, the astro track
      puts another part of the band inside the window. The two Fisher matrices differ there by up
      to 1.5e-2 of sqrt(F_ii F_jj), and by at most 3.5e-3 (median 4e-4 to 6e-4) for the sources
      inside their windows.
    - Steps of the existing Fisher on L1 orbits: between its default and tenfold steps it changes
      by up to 1.6e-3 of sqrt(F_ii F_jj) in β, above the criterion of 1e-3 that DFFR meets with
      1.1e-5.
