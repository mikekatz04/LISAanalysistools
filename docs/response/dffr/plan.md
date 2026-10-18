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
| 2 | Fresnel split | Not started | | |
| 3 | Tile seam | Not started | | |
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
band bin. `get_ll`, `fill_global`, `swap_ll`, the F-statistic and both gradients become
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
`stft_get_ll_impl<DFFRTile<GBDFFRSource>>`. Window settings come from the bound `STFTFresnel`,
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

- `lat_fresnel.hh` with DFFR's series, auxiliary functions and slopes; `STFTFresnel`'s
  `get_fresnel_integrals` and `get_fresnel_aux` keep their signatures and delegate.
- Tests: C and S against `scipy.special.fresnel`, f and g and their slopes against references,
  the phase kernel and its moment against quadrature (ported from dffr `tests/test_fresnel.py`).
- Regression: existing outputs agree with the step-1 baseline to 1e-12 relative; new baseline.

### Step 3: tile seam

- Tile contract, `ColumnTile`, tile-generic consumers and launchers with dynamic shared memory;
  the `*_impl<SourceT>` entry points keep their signatures through a default tile, so GBGPU's
  `gb_stft_*` wraps compile unchanged.
- Gate: bit-identical outputs on CPU and GPU against the step-2 baseline
  (`scripts/validation/stft_column_policy_oracle.py`, golden files, GBGPU STFT tests), and
  timings within run-to-run noise of the baseline. The swap kernels, which this step rewrites
  for two tables per tile, are left out of the CPU captures from step 2 on, so the gate covers
  them on the GPU, plus the swap checks of `test_stft_gb.py` on the CPU.

### Step 4: DFFR kernels

- LAT DFFR headers: settings from `STFTFresnel`, `STFTDomain` and the knobs; segment table
  (reception and emission states, link amplitudes and slopes, carrier and band start, closed-form
  TDI weights, first-order delay terms, endpoint anchors); windowed transform on `lat_fresnel.hh`;
  `DFFRTile`; DFFR Fisher (same estimator, steps and support rules as the existing Fisher,
  reusing the reception-state Fresnel kernels across amplitude, phase, inclination and
  polarisation perturbations, which do not change them); compact-template and segment-table
  diagnostics. Swap holds two tables per tile, so its row count (or the dynamic shared-memory
  limit) follows the budget.
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
  GPU runs keep both kernels.
- A baseline that must come from the code before a step, but cannot be captured before the
  rebuild, comes from a frozen environment of that code (`scripts/dffr/build_base_env.sh`). Such
  an environment is accepted only if it reproduces the CPU golden outputs of that code bit for
  bit.
- CPU tests: `python -m pytest GBGPU/tests/test_stft_gb.py GBGPU/tests/test_stft_gb_accuracy.py
  GBGPU/tests/test_stft_gb_crossdomain.py` and `python -m unittest tests.test_orbit_kinematics`
  from the LAT root.
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
- Shared memory: two segment tables for swap exceed 48 kB at 32 rows; the Fisher needs the
  reception-state sharing to fit. Blackwell (sm_120) allows less dynamic shared memory than A100.
- MIG slices have no peer access; multi-split tests use slices of one job and host-routed checks.
