# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- `binding.cxx` / `binding.hpp` — pybind11 module exposing the C++/CUDA functions to Python (one shared binding source for both CPU and GPU builds).
- `pycppdetector.pyx` is a legacy Cython file; the active path is the pybind11 module `pycppdetector` produced from `binding.cxx`.

## Python Package Layout

`src/lisatools/`:
- `detector.py` — `Orbits` / `DefaultOrbits` Python frontends wrapping the backend `OrbitsWrap` class.
- `sensitivity.py` — `SensitivityMatrix`, `get_sensitivity`, noise/PSD models.
- `datacontainer.py` — `DataResidualArray`: container abstracting data/residual/template arrays and their connection to sensitivity matrices.
- `analysiscontainer.py` — `AnalysisContainer`: combines a `DataResidualArray` + `SensitivityMatrix` + signal generator; central object for likelihood computations.
- `diagnostic.py` — inner products and likelihood terms used by `AnalysisContainer`.
- `domains.py` — frequency / time-frequency domain settings (`DomainSettingsBase` etc.).
- `stochastic.py` — stochastic foreground models (e.g. `FittedHyperbolicTangentGalacticForeground`).
- `sources/` — waveform generators per source class: `bbh/`, `emri/`, `gb/`, plus `defaultresponse.py` and `waveformbase.py`.
- `sampling/` — MCMC pieces built on top of `eryn`: priors, likelihood wrappers, custom moves, stopping criteria, GMM utilities.
- `globalfit/` — the LISA global fit pipeline: `pipeline.py`, `engine.py`, `run.py`, `recipe.py`, plus per-component modules (`mbhglobal.py`, `galaxyglobal.py`, `psdglobal.py`, `mbhsearch.py`, …) and stock recipes in `globalfit/stock/`.
- `utils/` — `constants.py` (re-exports `lisaconstants`-derived values like `YRSID_SI`), array helpers (`get_array_module`, `AET`), exceptions, multi-GPU data holders.
- `orbit_files/` — packaged orbit data files.

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

## Backend implementation hierarchy (sprint-wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START`,
   `BLOCK_INCR`, …) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.



## Narrowband mismatches mm2 / mm5 (chunked-het / WDM validation)

When verifying a chunked-heterodyne or other narrowband WDM template
against a lisatools reference signal, the canonical narrowband
mismatches are:

- **`mm5`** -- "5-layer" mismatch over a 5-m-layer band around the
  carrier `f0`. The band is defined by frequency bounds
  `[f0 - 3*layer_df, f0 + 2*layer_df]` (slightly asymmetric to cover
  the spectral tails on the side where the WDM transform spreads). Use
  this as the **primary** chunked-het accuracy metric -- it captures
  the dominant carrier + first-neighbour m-layers.

- **`mm2`** -- "2-layer" mismatch over just `m_floor` and `m_floor + 1`
  (the two layers that hold the bulk of a near-monochromatic GB
  signal). Band bounds: `[(m_floor - 0.5)*layer_df,
  (m_floor + 1 + 0.5)*layer_df]`. Use this as a tighter check
  isolating the carrier itself; it strips away spectral-tail
  contributions.

Both are **`1 - normalized overlap`**:

```python
mm = 1 - <d|h> / sqrt(<d|d> <h|h>)
```

via `AnalysisContainer.template_inner_product(..., normalize=True)`,
after slicing both `data` and `template` to the same narrow band by
building a per-binary `WDMSettings(min_freq=..., max_freq=...)` and
reusing the parent grid for layer-index alignment.

The canonical implementation lives in
`gb_chunked_prior_draws.py:283-340` (the `mm5` and `mm2` blocks).
SOBBH and other source-class versions should mirror the same band
definition for direct cross-source comparison.

Acceptance thresholds (current chunked-het with N_cp_sig=48,
N_cp_orbit=32, half-day wavelets, full angular prior):
- median mm5 ~ 1e-9, 90% < 8e-9, 99% < 3e-7
- low-frequency (m_floor < 100) sources occasionally show mm5 ~ 1e-7
  due to spectral-tail extension below ind_min_f -- documented
  systematic, not a bug.

