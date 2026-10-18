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
| 3 | Tile seam | CPU part done; GPU part pending | 2026-10-18 | Every output of the seven kernels, swap included, is bit-identical to the step 2 code on the CPU (144 and 26 arrays, against a frozen environment of the step 2 code), with the same test outcomes. CPU timings lie within run-to-run noise at 27 of 30 points; the other three differ by at most 0.7 %, within the drift between runs of identical code. ptxas gives every kernel at least the blocks per SM of step 2 on the A100 (sm_80) and the H100 (sm_90). The pending step 2 GPU runs come from the frozen step 2 environment. Results in `tests.md`. Corrected on 2026-10-18 in step 4: the register counts of step 3 were those of each kernel's own file; in the linked module every kernel has 244 registers on the A100 and 222 on the H100, in step 2 and step 3 alike, so 2 blocks per SM, and the conclusion holds. |
| 4 | DFFR kernels | 4a and 4b: CPU part done; 4c (DFFR Fisher) next; GPU part pending | 2026-10-18 | The DFFR numerics, tile and galactic-binary model are built. The existing response stays bit-identical to step 3 on the CPU (144 and 26 arrays), with the same test outcomes and CPU timings within run-to-run noise or faster. Against standalone dffr on the same kinematics table, templates, global fill and segment table are bit-identical and the inner products agree to 6.3e-15. With the linear envelope of production, DFFR and the existing response differ by a median mismatch of 5.6e-10. Linked registers give the DFFR kernels 2 to 3 blocks of 128 threads per SM and the existing kernels 2; register caps and device link-time optimisation are timed in step 6. Results in `tests.md`. |
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
  consumers; from step 4 also the band-bin-first pixel map (`StftBandBinMap<Rows>`) and the launch
  path above 48 kB of dynamic shared memory.
- `cutils/lat_dffr_kernels.hh` (step 4): DFFR settings, source-model contract, windowed
  transform, closed-form TDI weights of both generations with first-order delay terms, segment
  table, `DFFRTile<SourceT, Rows>` with its launch inputs, and two diagnostics of a batch of
  sources (compact templates, segment table). The DFFR Fisher follows in step 4c.
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
- `gbcomps.py` (step 5): the `response` flag and the DFFR dispatch.

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
existing Fisher kernel stays as it is for FresnelColumn. The DFFR Fisher (step 4c) generalises it
with the amplitude and grid split of step 4.

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
`start_freq_inds` as today. The comp's part of this path, the dispatch in `get_ll_stft`, comes in
step 5; until then the tests call the entry points directly.

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

Parts 4a and 4b done on the CPU on 2026-10-18; part 4c, the DFFR Fisher, follows; the GPU runs
are pending. The author split the step into 4a (LAT core), 4b (pixel map, launch path, tile and
galactic-binary model), 4c (Fisher) and 4d (documents, scripts, commits), with a review after
each.

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
  - The DFFR Fisher moves to 4c. It generalises the existing Fisher kernel instead of copying it.
    The derivative of a template splits into two parts:
    - a grid part, the three reception kernels and moments, which only f0, ḟ, f̈, λ and β change;
    - an amplitude part, the eighteen coefficients (TDI weights times reference phasors), which A,
      φ0, ι and ψ change.

    Those four parameters share the grid slot of the central template, so 21 grid slots replace
    36. The column Fisher must stay bit-identical.
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
  response against the frozen step 3 code. Driver: `scripts/dffr/run_step4.sh`. The Fisher
  against a Python reference comes with 4c. Results in `tests.md`.

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
- Occupancy, from the linked registers of step 4 (author's decisions of 2026-10-18). A/B timings
  on an A100, for both responses; the faster settings go in:
  - `domains.cu` without a register cap, and capped at 168 and at 128. The cap changes the
    register allocation, not the arithmetic, so the outputs must stay bit-identical;
  - device link-time optimisation (`-dlto`). It needs `nvvm/lib64/libnvvm.so` next to nvlink,
    which the pip wheel of the CUDA toolkit does not hold (`tests.md`, step 4);
  - 16 against 32 rows for the two-template DFFR consumers;
  - `Template::values` and `build` of the DFFR tile forced inline on the GPU, or not.
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
  f̈: up to 18 on its diagonal and 0.2 to 0.4 elsewhere. Step 4c plans the same estimator and
  steps for the DFFR Fisher; the steps are to be reviewed there. Noted, not changed.
- Shared memory: one DFFR template takes 32 672 bytes at 32 rows, so the two of swap (65 344
  bytes) exceed the 48 kB a launch gets on an A100 and on an H100; the Fisher needs the
  reception-state sharing to fit. With the opt-in of step 4, which takes its limit from the
  device, an A100 allows 163 kB per block and 164 kB per SM, an H100 227 kB and 228 kB.
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
- MIG slices have no peer access; multi-split tests use slices of one job and host-routed checks.
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
  timings and could change rounding through other inlining.
