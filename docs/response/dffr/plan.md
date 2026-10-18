# DFFR as a selectable response of STFTGBComputations: plan

Author: Coen Rondeel. The design below was agreed on 2026-10-16. This document is updated at the
end of every step and rewritten at the end of the effort to describe what was built. File and line
references point to LAT `l2d-dev` at 7c12a36d and GBGPU `l2d-dev` at ed22789, the base of the
branch `feat-dffr` in both repositories.

## Status

| Step | Scope | Status | Date | Notes |
| --- | --- | --- | --- | --- |
| 0 | Branches, documents, baseline of the existing response | CPU part done, `L1Orbits` included; GPU part pending | 2026-10-17 | The GPU part runs from a frozen environment of the step 0 code, which reproduces every CPU golden output bit for bit; every GPU node of SURF Spider has been down since 2026-10-16. Results in `tests.md`. |
| 1 | Orbits cache | CPU part done; GPU part pending | 2026-10-17 | Existing outputs change by at most 6e-11 (templates, inner products), 2e-9 (gradients) and 2e-8 (Fisher on the TDI track, whose f̈ entries change by up to 8e-6); the step 1 captures are the new baseline. Results in `tests.md`. |
| 2 | Fresnel split | CPU part done; GPU part pending | 2026-10-18 | Templates, inner products and the F-statistic change by at most 1.1e-14 of their peaks (gate 1e-12). Gradients and Fisher matrices change within the finite-difference noise that both codes carry, the criterion the author set for them in place of the gate. The step 2 captures are the new baseline; the pending step 1 GPU runs come from a frozen environment of the step 1 code. Results in `tests.md`. |
| 3 | Tile seam | CPU part done; GPU part pending | 2026-10-18 | Every output of the seven kernels, swap included, is bit-identical to the step 2 code on the CPU (144 and 26 arrays, against a frozen environment of the step 2 code), with the same test outcomes. CPU timings lie within run-to-run noise at 27 of 30 points; the other three differ by at most 0.7 %, within the drift between runs of identical code. ptxas gives every kernel at least the blocks per SM of step 2 on the A100 (sm_80) and the H100 (sm_90). The pending step 2 GPU runs come from the frozen step 2 environment. Results in `tests.md`. |
| 4 | DFFR kernels | Not started | | |
| 5 | Wiring into `STFTGBComputations` | Not started | | |
| 6 | Validation | Not started | | |
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
| Window | The data use LAT's symmetric Tukey window, while STFTFresnel and DFFR model the periodic one. The effect, estimated at 5e-5 per segment and so comparable to DFFR's own error, is measured in step 6 before either side changes. |
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
  consumers.
- `cutils/lat_dffr_*.hh` (new, names settled in step 4): DFFR settings, source-model contract,
  segment table, closed-form TDI weights with first-order delay terms, windowed transform,
  `DFFRTile<SourceModel>`, DFFR Fisher kernel, diagnostic kernels.

GBGPU (GB-specific):

- GB phase and amplitude formulas moved into inline free functions used by both
  `GBTDIonTheFly` and the DFFR GB model, so there is one source of truth.
- `cutils/gb_dffr_source.hh` (new): `GBDFFRSource`, the GB model (one component).
- `GBComputationGroup::gb_dffr_*_wrap` and their bindings in `binding_gbgpu.{hpp,cxx}`.
- `gbcomps.py`: the `response` flag and the DFFR dispatch.

### Tile contract (the seam)

A tile policy supplies: the shared bytes per row and the rows per pass for a given number of
templates; a cooperative `build` over a block of rows (barriers allowed); the band start of each
row; the pixel mapping (one row per thread, or band bin first); and `values` returning all three
channels of one pixel. `ColumnTile<FresnelColumn>` reproduces today's mapping (thread `tid` owns
rows `tid, tid + 128, ...`, all side bins, same summation order), so the existing outputs stay
bit-identical. `DFFRTile` holds 32-row segment tables in shared memory and maps a warp to one
band bin. The 32 is the warp size, so that the 32 threads of a warp take the 32 segments of one
band bin; 32 rows of 1012 bytes also fit the 48 kB of shared memory a launch gets by default. `get_ll`, `fill_global`, `swap_ll`, the F-statistic and both gradients become
tile-generic; the existing Fisher kernel stays as it is for FresnelColumn, and DFFR gets its own.

### Source-model contract (modes)

Per source: constants loaded once (sky basis, amplitude); a component count with a compile-time
maximum; per component a quadratic phase expansion at a retarded time (phase, frequency, rate,
amplitude ratio), the amplitude log-slope, and complex plus and cross polarisation weights. The
segment table and the pixel loop run over the components; a GB has one and pays nothing. This
mirrors TDI-on-the-fly, where each mode is a sub-source with its own amplitude, phase and angles,
but here the components of one source are summed per pixel, so `(h|h)` keeps the cross terms.

### Call path of a DFFR likelihood call

`comp.get_ll_stft` -> `self.stft_comps.cpp_kinematics(self.cpp_orbits)` ->
`GBComputationGroupWrap().gb_dffr_get_ll(..., kinematics, cpp_fresnel, cpp_domain,
tdi_generation, n_side_bins, linear_envelope, first_order_delays, ...)` ->
`stft_get_ll_launch<DFFRTile<GBDFFRSource>>`. Window settings come from the bound `STFTFresnel`,
the grid and data from `STFTDomain`, window starts from `start_freq_inds` as today.

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

Done on the CPU on 2026-10-17; the GPU runs are pending.

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

Done on the CPU on 2026-10-18; the GPU runs are pending.

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

Done on the CPU on 2026-10-18; the GPU runs are pending.

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

- LAT DFFR headers: settings from `STFTFresnel`, `STFTDomain` and the knobs; segment table
  (reception and emission states, link amplitudes and slopes, carrier and band start, closed-form
  TDI weights, first-order delay terms, endpoint anchors); windowed transform on `lat_fresnel.hh`;
  `DFFRTile`; DFFR Fisher (same estimator, steps and support rules as the existing Fisher,
  reusing the reception-state Fresnel kernels across amplitude, phase, inclination and
  polarisation perturbations, which do not change them); compact-template and segment-table
  diagnostics. Swap holds two tables per tile, so its row count (or the dynamic shared-memory
  limit) follows the budget.
- Moved here from step 3: the band-bin-first pixel map of the DFFR tile, which puts the 32
  segments of one band bin on the 32 threads of a warp, and the launch path for more than 48 kB of
  dynamic shared memory (32 rows of 1012 bytes per table: 32.4 kB for one table, 64.8 kB for the
  two of swap).
- GBGPU: shared GB formulas, `GBDFFRSource`, `gb_dffr_*` wraps and bindings; the TDI generation
  is detected in Python from the comp's TDIConfig.
- Tests: against standalone dffr on the same table (compact templates, band starts, global fill,
  `(d|h)`, `(h|h)` to about 1e-14 of the peak; segment-table stages); closed forms against the
  TDIConfig units; the self-consistency suite of `test_stft_gb.py` for DFFR (fill then domain
  inner product equals `get_ll`, swap identities, F-statistic against `get_ll` and swap, gradients
  against host central differences, Fisher against a Python reference built from `fill_global`);
  CPU against GPU. The existing response stays bit-identical.

### Step 5: wiring

- `STFTGBComputations`: `response`, `first_order_delays`, `chirp_rate_floor` in `_ctor_kwargs`;
  dispatch per method; the `linear_envelope` sentinel for gradients and Fisher; `window_factor`,
  `freq_from_tdi_phase` and `astro_track` documented as unused by DFFR; AET raises.
- `recipe.py`: optional `gb_stft_kwargs["response"]`, default unchanged.
- Tests: `test_stft_gb.py` over both responses; a replica keeps the response; engine smoke test
  (`fill_template`, `get_ll`, `get_swap_ll`) with `response="dffr"`.
- Docs: a usage section in `docs/response/dffr/README.md` (the flag, the DFFR knobs, what is
  ignored);
  the `STFTGBComputations` docstring points to it. Long explanations go in these docs, not in
  code comments.

### Step 6: validation

- Accuracy against the brute-force STFT of a time-domain GB TDI injection built with the LAT
  window, both responses (`test_stft_gb_accuracy.py` recipe); size the window-convention error.
- dffr time-domain truth (lisagwresponse, lisainstrument, pytdi) on the ESA trailing orbits, 30
  days to months, on a compute node, both responses.
- Cross-domain SNR^2 against FD and WDM (`test_stft_gb_crossdomain.py`) for DFFR.
- Mojito: `scripts/validation/gb_mojito_stft_fd_mismatch.py` for both responses.
- Benchmarks per method and batch size, both responses, on a full A100 and one A100 MIG slice,
  and on the CPU. Gate: DFFR is not slower than the existing response for any method the global
  fit calls.
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
  every output.
- CPU tests: `python -m pytest GBGPU/tests/test_stft_gb.py GBGPU/tests/test_stft_gb_accuracy.py
  GBGPU/tests/test_stft_gb_crossdomain.py` and
  `python -m unittest tests.test_orbit_kinematics tests.test_fresnel` from the LAT root.
- GPU tests and benchmarks with `force_backend="gpu"`, on one A100 MIG slice or a full A100 node.
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
  change is decided then.
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
  time-domain truth.
- GBT `CubicSplineSegment` derivative helpers return `c1 + c2 dx + c3 dx^2` and `c2 + c3 dx`,
  not the power-basis derivatives. Nothing calls them; noted only.
- At their default steps, the gradients and Fisher matrices of the existing response carry
  finite-difference noise, measured on the step 2 fixtures as the difference between the default
  and the tenfold step. Per gradient component, relative to its scale: about 1e-2 to 0.25 in f0,
  0.2 to 1.8 in ḟ and 0.6 to 13 in f̈. `information_matrix` notes the cause, roundoff in the
  Fresnel phase, and widens its f0 and ḟ steps 10 and 100 times for it; per Fisher entry,
  relative to sqrt(F_ii F_jj), the noise is then at most 5.2e-4 except in the row and column of
  f̈: up to 18 on its diagonal and 0.2 to 0.4 elsewhere. Step 4 plans the same estimator and
  steps for the DFFR Fisher; the steps are to be reviewed there. Noted, not changed.
- Shared memory: two segment tables for swap exceed 48 kB at 32 rows; the Fisher needs the
  reception-state sharing to fit. Without the opt-in a block gets 48 kB on an A100 and on an
  H100; with it, an A100 allows 163 kB per block and 164 kB per SM, an H100 227 kB and 228 kB.
- Newer devices than the A100, the H100 (sm_90) in particular. The timing gate of this effort is
  measured on A100s, the GPUs of SURF Spider, while the standalone DFFR timings in the README were
  measured on an H100 NVL. Both devices have 65 536 registers per SM and at most 255 per thread,
  so their steps in occupancy fall at the same register counts (4 blocks of 128 threads up to 128
  registers, 3 blocks up to 168), and both have warps of 32 threads, so the 32 rows of the DFFR
  tile suit both. After step 3, ptxas gives every STFT kernel at least the blocks per SM of step
  2 on both. What should change for H100 nodes:
  - The build. The install script of SURF Spider compiles machine code for its own GPUs and PTX of
    the A100 generation. An H100 would run code that its driver translates from that PTX when the
    module loads. A build for H100 nodes adds 90 to `CUDA_ARCH`, so that the code comes from PTX
    of its own generation, the code that the ptxas check describes.
  - The checks. The same source gives different register counts on the two devices: in step 2
    already, `get_ll` used 120 registers on the A100 and 128 on the H100, and the gradient of
    `get_ll` 128 and 144. A change of the kernels is therefore checked with
    `kernel_resources.py --arch 90` as well as `--arch 80`. Claims of speed on an H100 need the
    benchmarks repeated on one (`--device gpu`).

  What could change, to use what an H100 has more of:
  - Shared memory: with the two segment tables of the DFFR swap (64.8 kB at 32 rows), an H100
    fits 3 blocks per SM where an A100 fits 2. The launch path of step 4 for more than 48 kB
    should take its limit from the device (`cudaDevAttrMaxSharedMemoryPerBlockOptin`), not from a
    constant.
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
- MIG slices have no peer access; multi-split tests use slices of one job and host-routed checks.
- GBGPU's CPU module, which holds the STFT kernels of the CPU backend, compiles with `-Os`:
  `nanobind_add_module` optimises a module's sources for size unless it is given `NOMINSIZE`,
  and GCC applies the last `-O` option, so the `-O3` of the release build has no effect there.
  The GPU code is compiled by nvcc at `-O3`. Noted, not changed: a change would move the CPU
  timings and could change rounding through other inlining.
