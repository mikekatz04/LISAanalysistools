# DFFR as a selectable response of STFTGBComputations: plan

Author: Coen Rondeel. The design below was agreed on 2026-10-16. This document is updated at the
end of every step and rewritten at the end of the effort to describe what was built. File and line
references point to LAT `l2d-dev` at 7c12a36d and GBGPU `l2d-dev` at ed22789, the base of the
branch `feat-dffr` in both repositories.

## Status

| Step | Scope | Status | Date | Notes |
| --- | --- | --- | --- | --- |
| 0 | Branches, documents, baseline of the existing response | CPU part done, `L1Orbits` included; GPU part pending | 2026-10-17 | The GPU part must run from the base build before the first C++ change of step 1; every GPU node of SURF Spider has been down since 2026-10-16. Results in `tests.md`. |
| 1 | Orbits cache | Not started | | |
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

- Port the 5-point stencils from dffr `kinematics.cu` into the four `Orbits` methods; port the
  three build kernels into `Orbits::get_kinematics_table`, with a host-side coverage check that
  raises when the midpoints plus margins leave either grid.
- `KinematicsTableWrap`, `STFTComputationGroup.cpp_kinematics`, backend method lists.
- `stft_get_spacecraft_acc` calls `orbits->get_acceleration`; this changes the existing outputs
  slightly (3-point to 5-point), so the change is measured, reviewed, and becomes the new baseline.
- Tests (`lisa-analysis-tools/tests/test_orbit_kinematics.py`): stencil methods against
  analytic derivatives and against the file velocities; table against a NumPy mirror of the same
  interpolation and stencils (bit-level on the CPU); CPU against GPU; coverage errors; against
  dffr's lisaorbits builder on the ESA orbits that LAT ships (dffr environment). Existing STFT
  tests pass.

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
  timings within run-to-run noise of the baseline.

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

- Rebuild after C++ changes, CPU and CUDA backends, on a GPU node. The author's setup on SURF
  Spider: `ONLY="lisa-analysis-tools GBGPU" ./install_erebor_spider.sh` from `~/projects/erebor`.
- CPU tests: `python -m pytest GBGPU/tests/test_stft_gb.py GBGPU/tests/test_stft_gb_accuracy.py
  GBGPU/tests/test_stft_gb_crossdomain.py` and `python -m unittest tests.test_orbit_kinematics`
  from the LAT root.
- GPU tests and benchmarks with `force_backend="gpu"`, on one A100 MIG slice or a full A100 node.
- dffr environment: `cd direct-fast-fresnel-response && uv sync --extra compare`; the regression
  scripts in `scripts/dffr/` exchange arrays as `.npz` files in the output folder's
  `regression/`. Both sides use absolute times: the LAT orbits get `t0` equal to the file's `t0`.
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
- `Orbits::get_window` accepts the last cell, where `interpolate` reads one row past the table;
  the new methods stay below `t0 + (N - 2) dt`. A one-line fix of `get_window` is proposed for
  step 1.
- GBT `CubicSplineSegment` derivative helpers return `c1 + c2 dx + c3 dx^2` and `c2 + c3 dx`,
  not the power-basis derivatives. Nothing calls them; noted only.
- Shared memory: two segment tables for swap exceed 48 kB at 32 rows; the Fisher needs the
  reception-state sharing to fit. Blackwell (sm_120) allows less dynamic shared memory than A100.
- MIG slices have no peer access; multi-split tests use slices of one job and host-routed checks.
