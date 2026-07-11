# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## LISA Analysis Tools conventions & maps (this repo is the canonical hub)

LISAanalysistools (LAT) is the project's central LISA-physics library and the
canonical home for cross-repo Claude guidance — the umbrella workspace's root
directory (`lisa_sprint_2026/CLAUDE.md`) is now just a stub that points here.

The rules below apply to every LISA Analysis Tools repo (`lisa-on-gpu`, LAT,
GPUBackendTools, BBHx, FastEMRIWaveforms, GBGPU, Eryn). Compact gist per
rule — **full detail: [`docs/conventions.md`](docs/conventions.md).**

- **Backend implementation hierarchy.** GPU C++ (CUDA) leads as the
  reference implementation; CPU C++ mirrors it via `#ifdef`/shared macros
  and must match to machine precision; JAX may diverge internally with
  JAX-native idioms but must match at the inner-product level (reldiff
  ≲ 1e-12).
- **No new global-fit settings files.** Run configurations are installed
  `StockGlobalFit` classes under `lisatools.globalfit.stock`, not settings
  files — see "Stock global fits" below for the LAT-specific API.
- **No backend strings as function kwargs.** Backend choice is made at
  object instantiation (`force_backend="cpu"/"cuda12x"/"jax"`), never as a
  per-method kwarg (`backend=`, `use_jax=`, ...); dispatch internally via
  `self.backend`.
- **Host→device upload of class-wrapper objects.** Host-`new`'d wrapper
  structs (`OrbitsWrap`, `WDMSettingsWrap`, ...) whose pointer fields a CUDA
  kernel dereferences must be `cudaMalloc`+`cudaMemcpy`'d to device before
  the kernel launch, then freed after sync — never pass the host pointer
  straight into a device kernel.
- **CPU/GPU class-name aliasing.** Every C++ class compiled into both the
  CPU and GPU shared objects (wrapper classes and the underlying classes
  they point to) needs a per-backend `#define` alias (`FooGPU`/`FooCPU`) so
  the two builds emit distinct C++ type names and don't collide in
  pybind11's/nanobind's typeid registry.
- **Deepcopy / pickle safety.** Never store an array module (`self.xp = cp`)
  as an instance attribute — expose it as a property derived from a flag or
  `self.backend`. Guard `__getattr__` delegators against dunder/pre-`__init__`
  probing. Settings-tree objects must survive
  `pickle.loads(pickle.dumps(copy.deepcopy(obj)))`.
- **Narrowband mismatches mm2 / mm5.** Canonical chunked-het/WDM accuracy
  metrics: `mm5` over a 5-m-layer band around the carrier (primary metric),
  `mm2` over the 2 carrier layers (tighter check). Both are
  `1 - normalized overlap` via `AnalysisContainer.template_inner_product`.
- **No nested OpenMP in compute kernels.** Threading is owned at the run
  level (`OMP_NUM_THREADS`; multi-GPU splits via `AnalysisContainerArray`
  `gpus`/`run_threaded`); fix slow CPU kernels algorithmically or move them
  to GPU, never `#pragma omp` inside a kernel.
- **Cross-wheel C++/CUDA sharing: recompile-in-place.** Downstream wheels
  (GBGPU, BBHx, FEW) recompile against upstream (GBT, LAT) headers rather
  than link against upstream's compiled archive; prefer POD `*View` structs
  as the cross-wheel interface; bump `LISATOOLS_HEADER_ABI_VERSION` on any
  struct-layout change.

Maps: [`docs/codebase-map.md`](docs/codebase-map.md) (this repo's internal
layout) and [`docs/architecture-map.md`](docs/architecture-map.md)
(cross-repo dependency graph, capability→module table, backend-wheel model).

## Project Overview

`lisaanalysistools` (imported as `lisatools`) is a Python package for LISA (Laser Interferometer Space Antenna) data analysis, including building the LISA Global Fit. It is a hybrid Python / C++ / CUDA project built with `scikit-build-core` and CMake. Python 3.12+ is required.

## Build & Install

The project uses `scikit-build-core` (configured in `pyproject.toml`) and CMake (`CMakeLists.txt`) — **do not use `setup.py` directly**.

```sh
# Editable dev install (CPU; auto-detects CUDA toolkit if present)
pip install -e '.[dev,testing]'

# From-source install (release)
pip install .
```

Build behavior is controlled by the CMake option `LISATOOLS_WITH_GPU` (`AUTO` / `ON` / `OFF` / `ONLY` / `BARE`). On `AUTO` (default), the GPU backend is built only if `nvcc` and the CUDA toolkit are found. CUDA architectures default to `70;75;80;86;90` (70 is dropped automatically for CUDA ≥ 13). Override via `-DLISATOOLS_CUDA_ARCH=...` or `-DLISATOOLS_WITH_GPU=...` passed through `pip install --config-settings=cmake.define.<OPT>=<VAL> .`.

CUDA-enabled builds are published as separate plugin wheels: `lisaanalysistools-cuda11x`, `-cuda12x`, `-cuda13x`. Each ships a `lisatools_backend_cudaXXx` Python module that the main package imports lazily.

## Testing

Tests live in `tests/` and use `unittest`:

```sh
# Run the full suite (two equivalent forms)
python -m lisatools.tests
python -m unittest discover

# Run a single test file / case / method
python -m unittest tests.test_detector
python -m unittest tests.test_detector.DetectorTest
python -m unittest tests.test_detector.DetectorTest.test_orbits
```

The `[testing]` extras (`eryn`, `gbgpu`, `corner`, `gpubackendtools`, `matplotlib`) must be installed.

## Lint / Format

Pre-commit is configured with `black` (line-length 100, target py312) and `isort` (black profile). `mypy` and `pylint` are available as dev extras.

```sh
pre-commit install         # one-time
pre-commit run --all-files # run on the whole tree
```

## Backend Architecture

This is the most important architectural concept in the codebase.

LISAanalysistools delegates all GPU/CPU dispatch to **`gpubackendtools`** (a separate package, also a build dependency). Backends are registered at import in `src/lisatools/__init__.py` by adding `LISAToolsCpuBackend`, `LISAToolsCuda11xBackend`, `LISAToolsCuda12xBackend`, `LISAToolsCuda13xBackend` to the global `Globals().backends_manager`.

- `lisatools.get_backend("cpu" | "cuda11x" | "cuda12x" | "cuda13x" | "cuda" | "gpu")` returns a `Backend` object. The `"cuda"` and `"gpu"` names are aliases for the first available CUDA backend.
- `lisatools.has_backend(name)` reports availability without raising.
- A backend object exposes `xp` (either `numpy` or `cupy`) plus C-level functions/classes loaded from the corresponding native module (`lisatools_backend_<name>.pycppdetector`).
- Backend definitions: `src/lisatools/cutils/__init__.py`. Methods are declared in the `LISAToolsBackendMethods` dataclass — when adding a new C++/CUDA function, expose it on this dataclass and on each concrete backend's `*_module_loader`.

In Python code, the `xp` pattern is used widely — modules do `try: import cupy as cp / except: import numpy as cp` and pick the right module via `get_array_module(arr)` from `lisatools.utils.utility`. Don't assume `numpy`; arrays may be `cupy.ndarray`.

## Native Sources (`src/lisatools/cutils/`)

- `Detector.cu` / `Detector.hpp` — LISA orbit / detector geometry (light travel times, normals, spacecraft positions). The `.cu` file is **copied to `.cxx` at build time** and compiled by the C++ compiler for the CPU backend; the same `.cu` is compiled by `nvcc` for the GPU backend. Code must be valid as both.
- `PSD.cu` / `PSD.hpp` — PSD-related kernels (treated the same way).
- `LISAResponse.cu` / `LISAResponse.hh` — LISA arm-projection / TDI machinery. **Absorbed from lisa-on-gpu at Phase 3E** (2026-06-02). lisa-on-gpu's CMake references `${LISATOOLS_DIR}/cutils/LISAResponse.cu` so its tdionthefly module still consumes the same source.
- `fd_domain.hh`, `wdm_settings.hh`, `wdm_domain.hh` — deprecated include shims forwarding to `domains.hpp` (2026-06-12 consolidation; kept so lisa-on-gpu-era include paths compile). The POD-style FD/WDM domain descriptors they declared (absorbed from lisa-on-gpu at Phase 3L.1/3L.2/3L.4) now live in `domains.hpp`; their nanobind registrations stay in `binding_fd_domain.hpp` / `binding_wdm_settings.hpp` / `binding_wdm_domain.hpp`.
- `lat_tdi_on_the_fly.{hh,cu}` — `LISATDIonTheFly` base class + `OrbitsSplineCache` struct + 4 cache eval helpers. **Absorbed from lisa-on-gpu at Phase 3L.5** (2026-06-03). `.hh + .cu` split because of method volume (~26 methods, ~1300 lines); lisa-on-gpu's CMake copy-compiles the `.cu`. LAT's own detector_static archive also compiles the `.cu` so the virtual-class typeinfo is present in LAT's `.so` for the pybind11 registration to dlopen cleanly.
- `lat_spline_tdi_waveform.{hh,cu}` + `binding_lat_spline_tdi.hpp` — `FDSplineTDIWaveform` + `TDSplineTDIWaveform` (LISATDIonTheFly subclasses for spline-fed intrinsic waveforms) + their `*Wrap` + the `LISATDIonTheFlyWrap` pybind11 base shared by all Wrap subclasses. **Absorbed from lisa-on-gpu at Phase 3L.6** (2026-06-03). Same `.hh + .cu` + copy-compile pattern; wrap-side `run_wave_tdi_wrap` bodies inline in the `.hpp`.
- `binding_detector.cxx` / `binding_detector.hpp` — **nanobind** module exposing the C++/CUDA functions to Python (renamed from `binding.{cxx,hpp}` at the 2026-06-11 stft_tof merge; the module is still `pycppdetector`). The module body calls `detector_part(m)` then `response_part(m)` so OrbitsWrap, LISAResponseWrap, TDIConfigWrap all register here, plus the stft_tof-reactivated `XYZSensitivityMatrixWrap` (galactic grid, FD time-averaged transfer functions, `run_async`), `GalacticGridWrap`/`GalacticGridSetup`, and the legacy `psd_likelihood`/`compute_logpdf` kernels. (CubicSplineWrap moved to GBT's `interp` module 2026-06-10 — GBT is its single registrant.)
- `domains.{hpp,cu}` — stft_tof STFT time-frequency C++ (`STFTSettings`, `FDSettings`, `STFTDomain`, `STFTFresnel`, and `FDDomainForStft` — renamed from the incoming `FDDomain` to avoid colliding with dev's `fd_domain.hh` class; unify later). TDI flavor ints are re-based to the canonical `TDI_XYZ=1 / TDI_AET=2 / TDI_AE=3` behind `#ifndef` guards — Python must pass `backend.TDITypeDict` values, never literals. Also hosts the consolidated WDM/FD domain descriptors (2026-06-12); the STFT wraps are bound via `binding_domains.hpp` (`STFTDomainWrap`, `FDDomainForStftWrap`, `STFTFresnelWrap`).
- `binding_flr.cxx` / `binding_flr.hpp` — `response_part()` implementation + shared wrapper classes (`ReturnPointerBase`, `TDIConfigWrap`, `LISAResponseWrap`). **Absorbed from lisa-on-gpu at Phase 3E**. The `PYBIND11_MODULE(responselisa, m)` block was stripped; LAT's pycppdetector is the sole entry point.
- `orbits_view.hpp` — `OrbitsView` POD struct, the stable-layout cross-wheel interface downstream packages consume in place of typed `Orbits*` pointers. `binding_detector.cxx` runs `static_assert(sizeof + 15 offsetofs)` confirming layout matches `class Orbits` at every build. See plan section "POD-view side-channel".
- `lisatools_header_abi.hpp` — `LISATOOLS_HEADER_ABI_VERSION` macro + `LISATOOLS_IS_WRAPPER_OWNER` toggle. **`binding_detector.cxx` sets the toggle to 1** (LAT is the owner); downstream binding TUs leave it at the default 0 and add `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)`. Compile-time enforcement of the single-registrant rule. See `tools/check_single_registrant.sh` in the umbrella workspace root for the CI-side grep complement.
- `LISAanalysisToolsConfig.cmake` — `find_package(LISAanalysisTools CONFIG REQUIRED)` → `LISAanalysisTools::headers` interface target for downstream CMake consumers.
- `pycppdetector.pyx` is a legacy Cython file; the active path is the nanobind module `pycppdetector` produced from `binding_detector.cxx`.

## Python Package Layout

`src/lisatools/`:
- `detector.py` — `Orbits` / `DefaultOrbits` Python frontends wrapping the backend `OrbitsWrap` class.
- `sensitivity.py` — `SensitivityMatrix`, `get_sensitivity`, noise/PSD models.
- `datacontainer.py` — `DataResidualArray`: container abstracting data/residual/template arrays and their connection to sensitivity matrices.
- `analysiscontainer.py` — `AnalysisContainer`: combines a `DataResidualArray` + `SensitivityMatrix` + signal generator; central object for likelihood computations.
- `diagnostic.py` — inner products and likelihood terms used by `AnalysisContainer`.
- `domains.py` — frequency / time-frequency domain settings (`DomainSettingsBase` etc.).
- `stochastic.py` — stochastic foreground models (e.g. `FittedHyperbolicTangentGalacticForeground`).
- **`response/`** — LISA-response Python frontends, **absorbed from `fastlisaresponse` at Phase 3B+C**:
  - `parallelbase.py` — `FastLISAResponseParallelModule` backend-dispatch base.
  - `tdiconfig.py` — `TDIConfig` configuration.
  - `directresponse.py` — was `fastlisaresponse.response.py`; `pyResponseTDI`, `ResponseWrapper`, `ecliptic_to_icrs`.
  - `tdionfly.py` — TDI-on-the-fly family.
- **`jax/`** — pure-JAX backend + LISA-response/WDM JAX implementations:
  - `backend.py`, `orbits.py` — LAT-native (existing).
  - `response/` — `base.py` (JaxAmpPhaseSource), `projection.py`, `tdi_config.py`, `amp_phase_extract.py`. **Absorbed at Phase 3D.**
  - `wdm/` — `wavelet_lookup.py`, `wdm_settings.py`, `wdm_domain.py`, `fast_inner.py`. **Absorbed at Phase 3D.** GB-specific (heterodyne) variants live in `gbgpu.jax.wdm`.
- `sources/` — waveform generators per source class: `bbh/`, `emri/`, `gb/`, plus `defaultresponse.py` and `waveformbase.py`.
- `sampling/` — MCMC pieces built on top of `eryn`: priors, likelihood wrappers, custom moves, stopping criteria, GMM utilities.
- `globalfit/` — the LISA global fit pipeline: `engine.py`, `run.py`, `recipe.py`, `hdfbackend.py`, the moves in `globalfit/moves/`, and stock recipes in `globalfit/stock/` (the legacy `pipeline.py` + per-component `mbhglobal/galaxyglobal/psdglobal` movers were deleted 2026-07, parallel-resources plan P0/P4; `mbhsearch.py` remains for the dev search script). Post stft_tof merge (2026-06-12) conventions:
  - **Engine-side template generation**: each branch registers a params-based generator on `Settings.signal_gen` (`fn(*sampling_params) -> DomainBase`, wrapping transform + waveform + domain projection); `run.py::setup_acs(state, rebuild_residuals=True)` builds/subtracts the state's templates from the residuals under the hood. Settings-file recipes build **moves only** — never write residuals directly.
  - **ICRS run frame**: catalogue sky/polarization parameters are sampled raw (`alpha`/RA, `sin_delta`, `psi` ICRS); orbits are loaded with `frame="icrs"`. The stock MBH transform (`make_mbh_transform_container`, forward + inverse) is direct-ICRS with `Q = m1/m2`.
  - **Domains are never communicated by string** — `GeneralSettings.domain_settings` takes a `DomainSettingsBase` instance or a `(times, dt, force_backend)` factory (`FDSettings/STFTSettings/WDMSettings.make_factory`); all dispatch is `isinstance` on the settings class.
- `utils/` — `constants.py` (re-exports `lisaconstants`-derived values like `YRSID_SI`), array helpers (`get_array_module`, `AET`), exceptions.
- `orbit_files/` — packaged orbit data files.
- `scripts/` — dev / validation / benchmark / diagnostics scripts (`gb_chunked_het/`, `gb_lookup/`, `sobbh/`, `mbh/`, `emri/`, `wdm/`, `validation/`, `benchmark/`, `diagnostics/`, `notes/`). Migrated from the umbrella workspace root at Phase 2.

## Stock global fits — no new settings files (LISA Analysis Tools–wide rule)

Run configurations live as **installed stock classes**, not settings files
(reorg-top-layer, 2026-07-09). The API:

```python
from lisatools.globalfit.stock import erebor

erebor.get_stock_options()             # [(name, description), ...]
fit = erebor.gb_no_fg(nwalkers=4)      # or erebor.get_stock("gb_no_fg", ...)
fit.gb.center_freq = 8e-3              # plain attribute access on the blocks
fit.recipe.pop_move("rj_refit")        # named move stacks per recipe stage
fit.remove_branch("galfor")            # compose whole objects in/out
curr = fit.build()                     # heavy stage (data load), on command
fit.run()                              # build -> GlobalFit -> run_global_fit
```

Every variant's data pipeline swaps with one knob: `fit.general.data_mode`
(`"mojito"` default everywhere; `"synthetic"` builds all streams in-process
with no external data; all_sources also keeps legacy `"sangria"`). Env:
`DATA_PROCESSOR=<mode>`. An explicit `data_processor_class` swap always wins.

Architecture (building-block pyramid, `stock/base.py` + `stock/erebor/`):
`StockGlobalFit` **inherits `CurrentInfoGlobalFit`** with the heavy
`super().__init__` deferred to `.build()`; the per-branch knob layer is the
existing `*Settings` dataclasses (variants fill defaults via subclasses);
the recipe is a declarative `RecipeSpec` of stages/`MoveSpec`s materialized
by the variant's module-level `setup_recipe`. Env vars resolve as field
defaults (*explicit kwarg > env var > hard default*). Waveform-path
defaults: SOBBH TDI-on-the-fly, MBH legacy phentax, EMRI legacy — per-branch
`use_tdionfly` knobs; `USE_TDIONFLY` env flips both.

Rules:

1. **No new global-fit settings files.** A new run variant is a
   `StockGlobalFit` subclass in `lisatools.globalfit.stock.<family>/variants/`
   registered with the family registry; knobs are documented dataclass
   fields, users get the class (good docstrings replace file-editing).
2. **Nothing heavy in `__init__`.** Construction is validation + defaults
   only; data loads, waveform builds, HDF backends, and directory creation
   happen in `.build()` (prove with the cheapness tests in
   `tests/test_stock_globalfit.py`).
3. **The pre-build fit must pickle/deepcopy** (LISA Analysis Tools–wide rule): named
   module-level functions/classes only on the config; runtime-only objects
   attach post-deepcopy via `attach_runtime_objects`.
4. The `global_fit_input/*.py` files for migrated variants are minimal
   stubs (default `get_global_fit_settings()` + legacy re-exports) — do not
   grow them back.

`scripts/run_global.py` accepts `--stock <name>` alongside the legacy
`-sfp <path>`.

## V2 signal-heterodyne likelihood path

A second WDM-domain likelihood path — polyphase per-active-m-layer iFFT +
carrier de-rotation producing sparse complex WDM coefficients without the full
dense `TDSignal.transform` — has been developed and validated alongside the
chunked-heterodyne (`gb_wdm_het_*`) family (~130× faster; GB mm5 ≈ 1.6e-9
median). Source-agnostic helpers live in `signal_het.py` / `chunked_het.py`;
the GB implementation is `gbgpu.gbsignalhetcomputations.GBSignalHetComputations`
(currently CPU-only, wired into the global-fit GB path via `for_band_engine()`),
and the SOBBH counterpart lives in `bbhx`. See the per-repo `docs/codebase-map.md`
for exact current locations; a JAX mirror is a documented follow-up.

## Downstream consumption

LAT exposes its public C++/CUDA headers via:
- `lisatools.get_include() → str` (Python entry point used by downstream CMake via shell-out)
- `lisatools.get_cmake_module_path() → str` returning the directory containing `LISAanalysisToolsConfig.cmake`.

Downstream waveform packages (GBGPU, BBHx, FastEMRIWaveforms) consume LAT headers (`Detector.hpp`, `LISAResponse.hh`, `binding_flr.hpp`, `orbits_view.hpp`, `lisatools_header_abi.hpp`, ...) but **do not register the shared wrapper classes with pybind11** — that's LAT's sole responsibility (single-registrant rule, enforced by `LISATOOLS_IS_WRAPPER_OWNER`).

## Key External Dependencies

- `gpubackendtools` — backend dispatch framework (CPU/CUDA selection, module loading). Build- and runtime-required.
- `lisaconstants` — pinned to `==2.0.2`, source of physical constants.
- `eryn` — MCMC sampler used by `sampling/` and `globalfit/`.
- `gbgpu`, `mojito`, `mojito-processor`, `cudakima` — source / response models pulled in via `pyproject.toml`. `mojito` is fetched from a custom GitLab index (see `[tool.uv.sources]`).
- `cupy` is **not** declared in `dependencies` — it must be installed separately matching the chosen CUDA backend (`cupy-cuda12x` etc.).

## Notes for Working in This Tree

- The repository root contains many untracked dev artifacts (notebooks, `.npy`, `.png`, `back_*.py`, `fix_*.py`, scratch `.txt` files). These are personal scratch and not part of the package — don't modify or rely on them. Package code lives under `src/lisatools/` and `tests/`.
- `Makefile` at the repo root is leftover from a previous project (`lisacattools`) — ignore it; build with `pip` / `cmake` as described above.
- When editing native code, remember the `.cu → .cxx` copy step: a change to `Detector.cu` rebuilds both CPU and GPU targets.
- When adding Python code that needs to work on both CPU and GPU, follow the `xp` pattern (resolve the array module from an input array via `get_array_module`) rather than importing `cupy` unconditionally.
