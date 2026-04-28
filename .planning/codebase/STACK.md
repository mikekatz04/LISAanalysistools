# Technology Stack

**Analysis Date:** 2026-04-28

## Languages

**Primary:**
- Python 3.12+ — all application logic, sampling, analysis, and glue code (`src/lisatools/`)
- C++17 — performance-critical kernels shared between CPU and GPU paths (`src/lisatools/cutils/Detector.cu`, `PSD.cu`, `binding.cxx`)
- CUDA — GPU-accelerated versions of the same C++ kernels compiled with `nvcc`; same `.cu` sources are compiled as plain C++ for the CPU backend by copying to `.cxx` (`src/lisatools/cutils/CMakeLists.txt`)

**Secondary:**
- Cython — used in the build system's `requires` block (`pyproject.toml`) and referenced in `pycppdetector.pyx` stub; active Cython sources are minimal compared to the pybind11 path

## Runtime

**Environment:**
- CPython 3.12+ (classifiers also list 3.10, 3.11, 3.13 but `requires-python = ">= 3.12"`)

**Package Manager:**
- `uv` — used for all virtual-environment management and script execution; `uv run python` is the canonical invocation (do NOT use raw `python3`)
- Lockfile: `uv.lock` present and committed

## Build System

**Primary:**
- `scikit-build-core` (build backend declared in `pyproject.toml [build-system]`) delegates to CMake
- `CMakeLists.txt` at project root drives the full build
- `src/lisatools/cutils/CMakeLists.txt` declares CPU and GPU backend targets

**CMake Details:**
- Minimum CMake 3.23, tested up to 3.31
- GPU support controlled by `LISATOOLS_WITH_GPU` option (`AUTO` | `ON` | `OFF` | `ONLY` | `BARE`)
  - `AUTO`: detects CUDA toolchain at configure time
  - `ONLY`: GPU only (used for plugin wheel builds)
  - `BARE`: no compiled extensions (pure-Python release)
- CUDA architectures: `70;75;80;86;90` by default; arch 70 dropped automatically for CUDA 13+
- CPU build: copies `.cu` → `.cxx`, compiles with standard C++ compiler via `add_library(...STATIC Detector.cxx PSD.cxx)`
- GPU build: compiles `.cu` directly with `nvcc`; `CMAKE_CUDA_SEPARABLE_COMPILATION ON`
- `-march=native` applied to CXX if supported (configurable via `LISATOOLS_MARCH`)
- CMake functions from `gpubackendtools` (`apply_cpu_backend_common_options`, `apply_gpu_backend_common_options`) control target properties

**pybind11:**
- Used to expose C++ classes and functions to Python via `pybind11_add_module` in CMake
- Entry point: `src/lisatools/cutils/binding.cxx` / `binding.hpp`
- On GPU builds, uses `pybind11_cuda_array_interface.hpp` for zero-copy array passing via CUDA Array Interface
- On CPU builds, uses standard `py::array_t<T>`
- Key bindings exported: `OrbitsWrapCPU`/`OrbitsWrapGPU`, `OrbitsCPU`/`OrbitsGPU`, `XYZSensitivityMatrixWrapCPU`/`XYZSensitivityMatrixWrapGPU`, `psd_likelihood`, `compute_logpdf`

## Versioning

- `setuptools_scm` provides dynamic version from git tags (`src/lisatools/_version.py`)
- `version_scheme = "no-guess-dev"`, `local_scheme = "node-and-date"`

## GPU Backend Architecture

The backend system is provided by the `gpubackendtools` package (≥ 0.0.15). At package import time (`src/lisatools/__init__.py`), four backend classes are registered:

| Backend name | Class | Array module |
|---|---|---|
| `lisatools_cpu` | `LISAToolsCpuBackend` | `numpy` |
| `lisatools_cuda11x` | `LISAToolsCuda11xBackend` | `cupy` (cupy-cuda11x) |
| `lisatools_cuda12x` | `LISAToolsCuda12xBackend` | `cupy` (cupy-cuda12x) |
| `lisatools_cuda13x` | `LISAToolsCuda13xBackend` | `cupy` (cupy-cuda13x) |

Each backend class inherits from both a `gpubackendtools` base class (`CpuBackend`, `Cuda11xBackend`, etc.) and the project-local `LISAToolsBackend`. At instantiation, it imports the matching compiled wheel (`lisatools_backend_cpu.pycppdetector`, `lisatools_backend_cuda11x.pycppdetector`, etc.) and wraps its symbols in a `LISAToolsBackendMethods` dataclass. Definition: `src/lisatools/cutils/__init__.py`.

**Backend selection API** (`src/lisatools/__init__.py`):
```python
import lisatools
lisatools.has_backend("cpu")       # check availability
lisatools.get_backend("cuda12x")   # get backend object
lisatools.get_first_backend(...)   # first available from list
```

**Array-module pattern** (CPU/GPU-agnostic NumPy/CuPy switching):
```python
from lisatools.utils.utility import get_array_module
xp = get_array_module(arr)   # returns numpy or cupy based on array type
result = xp.sum(arr)
```
Implementation: `src/lisatools/utils/utility.py` lines 196–207.

`Globals()` from `gpubackendtools` is a singleton that manages backend registration. CUDA macros (`CUDA_CALLABLE_MEMBER`, `CUDA_DEVICE`, `CUDA_KERNEL`) come from `gpubackendtools/cutils/gbt_global.h` (external). The same C++ class is aliased: `Orbits` → `OrbitsCPU` or `OrbitsGPU` by preprocessor macro in `src/lisatools/cutils/Detector.hpp`.

## Key Dependencies

**Core Scientific Computing:**
- `numpy` — primary CPU array library; used throughout
- `scipy` — interpolation, signal processing, special functions (`sensitivity.py`, `domains.py`, `detector.py`)
- `cupy` (optional, cuda11x/12x/13x variant) — GPU array library; mirrors NumPy API; required by GPU backends

**Interpolation / Splines:**
- `cudakima` — GPU/CPU-compatible Akima spline interpolation; used in `sensitivity.py` and `mojito_detector.py` for smooth PSD curves (`from cudakima import AkimaInterpolant1D`)
- `multispline` — listed as a core dependency in `pyproject.toml`; no direct import found in source (likely transitive / future use)

**Data I/O:**
- `h5py` — HDF5 file I/O; used for orbit files, MCMC chain storage, state persistence (`detector.py`, `globalfit/engine.py`, `globalfit/hdfbackend.py`, `globalfit/stock/erebor.py`)
- `pyyaml` — YAML parsing (file registry)
- `jsonschema` — validates file registry content
- `requests` — downloads orbit/data files from GitHub if missing (`detector.py` line 134)
- `platformdirs` — locates config/data directories cross-platform
- `rich` — displays download progress bars
- `tqdm` — progress bars for iterative tasks

**MCMC / Sampling** (optional group `sampling`):
- `eryn` — primary MCMC sampler; `EnsembleSampler`, `HDFBackend`, `State`, `TransformContainer`, `ProbDistContainer` are all imported extensively (`globalfit/engine.py`, `globalfit/hdfbackend.py`, `sampling/`)
- `mpi4py` — MPI parallelism for distributed runs (`globalfit/run.py`, `globalfit/pipeline.py`)

**Source Models / Waveforms** (optional, used in `sources/` and `globalfit/`):
- `fastlisaresponse` — LISA TDI response wrapper; `ResponseWrapper` used in `sources/emri/waveform.py`, `sources/gb/waveform.py`
- `few` (Fast EMRI Waveforms) — EMRI waveform generation; imported in `sources/gb/waveform.py` as `from few.waveform import GenerateEMRIWaveform`
- `gbgpu` — galactic binary GPU waveforms; `GBGPU` used in `sources/gb/waveform.py`; utilities in `globalfit/stock/erebor.py`
- `bbhx` — massive black hole binary waveforms + likelihood; `BBHWaveformFD`, `HeterodynedLikelihood`, `LISATDIResponse` imported in `globalfit/mbhsearch.py`

**LISA-Specific:**
- `lisaconstants` (==2.0.2, pinned) — physical and LISA mission constants; `lisaconstants.indexing.link2sc` used in `mojito_detector.py`
- `mojito` — reads Mojito L1 orbit files (ESA format); `MojitoL1File` used in `detector.py` for `L1Orbits` class (sourced from ESA GitLab index)
- `mojito-processor` — orbit pre-processing; listed as dependency (sourced from TestPyPI; pending PyPI publication)

**Performance / Compilation:**
- `numba` (≥ 0.63.1) — JIT compilation + CUDA kernels for PSD likelihood; `@cuda.jit` used in `cutils/psd_likelihood_utils.py` (`psd_likelihood_xyz_numba_fused`, `psd_likelihood_numba`)
- `nvidia-ml-py` — detects installed CUDA version to select correct backend wheel at install time

**Configuration / Validation:**
- `pydantic` — listed as core dependency; used for advanced dataclasses and citation/reference handling (no direct import found in current source scan; likely used via `gpubackendtools` or prepared for future use)
- `astropy` — coordinate transforms (ICRS → ecliptic in `detector.py`)
- `scikit-learn` — GMM (Gaussian Mixture Model) for proposals in `globalfit/gmm.py`

**Visualization** (optional `doc`/`sampling` groups):
- `matplotlib` — plotting throughout (`sensitivity.py`, `diagnostic.py`, `domains.py`)
- `corner` — posterior corner plots (`diagnostic.py`)
- `chainconsumer` — chain convergence visualization (optional)

## Configuration Files

**Build:**
- `pyproject.toml` — project metadata, dependencies, build backend, uv index configuration
- `CMakeLists.txt` — top-level CMake configuration
- `src/lisatools/cutils/CMakeLists.txt` — backend compilation targets
- `src/lisatools/cutils/lisatools_sources.cmake` — additional CMake source definitions

**Code Style:**
- `black` formatter: line-length 100, target Python 3.12 (configured in `pyproject.toml [tool.black]`)
- `isort` profile "black", line_length 100 (`pyproject.toml [tool.isort]`)
- `pylint`, `mypy` in dev dependencies
- `.pre-commit-config.yaml` — pre-commit hooks

**Runtime:**
- `global_fit_input/*.py` — Python-based configuration scripts for different global fit scenarios (not YAML/TOML; imported as modules)

## Platform Requirements

**Development:**
- Python 3.12+
- `uv` package manager
- CMake 3.23+
- C++17-capable compiler
- Optional: CUDA Toolkit (11.x, 12.x, or 13.x) for GPU backends

**Production / HPC:**
- CPU-only or CUDA-enabled nodes
- MPI environment for distributed runs (`mpi4py`)
- HDF5 support (via `h5py`)

---

*Stack analysis: 2026-04-28*
