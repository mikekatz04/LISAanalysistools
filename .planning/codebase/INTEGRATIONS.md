# External Integrations

**Analysis Date:** 2026-04-28

## LISA-Specific Libraries

### lisaconstants (==2.0.2, pinned)

- **Purpose:** Physical constants and LISA mission parameters (arm length, speed of light, orbital period, etc.)
- **Usage:** `from lisaconstants.indexing import link2sc` in `src/lisatools/mojito_detector.py`; star-imported constants in `src/lisatools/utils/constants.py` (re-exported through `from .utils.constants import *` in many modules)
- **Source index:** PyPI
- **Version pinned** — breaking changes in the constants schema would silently corrupt science results, hence the exact pin

### mojito

- **Purpose:** Reads ESA Mojito L1 orbit files; provides the `MojitoL1File` class for accessing spacecraft positions, velocities, and light-travel times from the official LISA orbit format
- **Usage:** `from mojito import MojitoL1File` in `src/lisatools/detector.py` (inside `L1Orbits` class, line ~666); lazy import — only triggered when constructing `L1Orbits`
- **Source index:** ESA GitLab (`https://gitlab.esa.int/api/v4/groups/29349/-/packages/pypi/simple`) — not on public PyPI
- **Integration point:** `src/lisatools/detector.py` — `L1Orbits` class and `LISAToolsJAXL1Orbits` class

### mojito-processor

- **Purpose:** Orbit pre-processing (resampling, filtering of Mojito orbit data)
- **Usage:** Listed in `pyproject.toml` core dependencies; no direct import found in current source scan
- **Source index:** TestPyPI (pending migration to PyPI)

### fastlisaresponse

- **Purpose:** Computes LISA TDI (Time-Delay Interferometry) response functions for gravitational wave sources; wraps source waveforms in the `ResponseWrapper` class
- **Usage:**
  - `from fastlisaresponse import ResponseWrapper` in `src/lisatools/sources/emri/waveform.py`
  - `from fastlisaresponse import ResponseWrapper` in `src/lisatools/sources/gb/waveform.py`
- **Optional dependency:** in `[optional-dependencies.sampling]` group
- **Integration pattern:** `ResponseWrapper` wraps a waveform generator and an `EqualArmlengthOrbits` object, producing TDI A/E/T channel data

### eryn

- **Purpose:** MCMC ensemble sampler tailored for trans-dimensional problems; handles multi-source models where the number of sources is unknown
- **Key classes used:**
  - `EnsembleSampler` — main sampler (`globalfit/engine.py`)
  - `HDFBackend` — HDF5-backed chain storage (`globalfit/hdfbackend.py`, base class for `GBHDFBackend`, `MBHHDFBackend`, etc.)
  - `State` — sampler state container
  - `TransformContainer` — parameter transformations for proposals
  - `ProbDistContainer` — prior distributions
  - `StretchMove` — proposal move
  - `make_ladder` — temperature ladder for parallel tempering (`globalfit/stock/erebor.py`)
- **Usage files:** `src/lisatools/globalfit/engine.py`, `src/lisatools/globalfit/hdfbackend.py`, `src/lisatools/globalfit/stock/erebor.py`, `src/lisatools/sampling/likelihood.py`, `src/lisatools/sampling/prior.py`, `src/lisatools/sampling/stopping.py`
- **Optional dependency:** in `[optional-dependencies.sampling]` and `[optional-dependencies.testing]` groups

### few (Fast EMRI Waveforms)

- **Purpose:** Generates EMRI (Extreme Mass Ratio Inspiral) gravitational waveforms rapidly using GPU-accelerated trajectory integration
- **Usage:** `from few.waveform import GenerateEMRIWaveform` in `src/lisatools/sources/gb/waveform.py` (active); commented out in `src/lisatools/sources/emri/waveform.py`
- **Not listed** in `pyproject.toml` — expected to be installed separately by the user; import is guarded implicitly by its absence causing `ImportError` at use time

### gbgpu

- **Purpose:** GPU-accelerated galactic binary (GB) waveform generation
- **Usage:**
  - `from gbgpu.gbgpu import GBGPU` in `src/lisatools/sources/gb/waveform.py`
  - `from gbgpu.utils.utility import get_fdot, get_N` in `src/lisatools/globalfit/stock/erebor.py`
- **Not listed** in `pyproject.toml` — expected separate installation

### bbhx

- **Purpose:** Massive Black Hole (MBH) binary frequency-domain waveforms and likelihood for LISA; includes heterodyned likelihood for fast evaluation
- **Usage:**
  - `from bbhx.waveformbuild import BBHWaveformFD` in `src/lisatools/globalfit/mbhsearch.py`
  - `from bbhx.likelihood import HeterodynedLikelihood, Likelihood as MBHLikelihood` in `src/lisatools/globalfit/mbhsearch.py`
  - `from bbhx.response.fastfdresponse import LISATDIResponse` in `src/lisatools/globalfit/mbhsearch.py`
  - `from bbhx.utils.transform import mT_q, LISA_to_SSB` in `src/lisatools/globalfit/stock/erebor.py`
- **Not listed** in `pyproject.toml` — expected separate installation

### cudakima

- **Purpose:** GPU/CPU-compatible Akima spline interpolation (1D); used for smooth PSD evaluation
- **Usage:**
  - `from cudakima import AkimaInterpolant1D` in `src/lisatools/sensitivity.py` (line 29)
  - `from cudakima import AkimaInterpolant1D` in `src/lisatools/mojito_detector.py` (line 23)
- **Listed** in `pyproject.toml` core dependencies

---

## GPU Backend System (gpubackendtools)

- **Purpose:** Provides the abstract backend registry, base classes for CPU/CUDA backends, and CUDA array interface utilities
- **Version requirement:** ≥ 0.0.15 (build); ≥ 0.0.16 (testing)
- **Key symbols consumed:**
  - `Globals` — singleton backend registry (`src/lisatools/__init__.py` line 42)
  - `CpuBackend`, `Cuda11xBackend`, `Cuda12xBackend`, `Cuda13xBackend` — base classes (`src/lisatools/cutils/__init__.py`)
  - `BackendMethods`, `BackendUnavailableException`, `MissingDependencies`
  - `get_backend`, `has_backend`, `get_first_backend` — re-exported from `src/lisatools/__init__.py` with `lisatools_` prefix normalization
- **CMake integration:** `gpubackendtools` provides `apply_cpu_backend_common_options` and `apply_gpu_backend_common_options` CMake functions (loaded from `gpubackendtools/cutils/cmake_functions.cmake`) and the shared `gbt_global.h` header with `CUDA_CALLABLE_MEMBER`, `CUDA_DEVICE`, `CUDA_KERNEL` macros
- **Compiled backend packages:** At runtime the four separate wheel packages (`lisatools_backend_cpu`, `lisatools_backend_cuda11x`, `lisatools_backend_cuda12x`, `lisatools_backend_cuda13x`) each ship a compiled `pycppdetector` pybind11 module; `LISAToolsBackend.__init__` imports whichever is available

---

## HDF5 File I/O

HDF5 is used pervasively for both input (orbit files) and output (MCMC chains, states).

### Orbit Files

- **Library:** `h5py`
- **Format:** HDF5 files containing spacecraft positions, normal unit vectors, and light-travel times on a regular time grid
- **Reading:** `src/lisatools/detector.py` — `Orbits._load_orbit_data()` reads groups and datasets; field names follow a conventional schema (`n_arr`, `ltt_arr`, `x_arr`)
- **Auto-download:** If the orbit file is not present locally, `detector.py` (line 134) fetches it from GitHub using `requests.get(github_file)` and saves it before opening with `h5py`

### MCMC Chain / Backend Files

- **Library:** `h5py` (via `eryn.backends.HDFBackend`)
- **Format:** Eryn HDF5 backend format; extended by project-local subclasses
- **Key files:**
  - `src/lisatools/globalfit/hdfbackend.py` — defines `GFHDFBackend`, `GBHDFBackend`, `MBHHDFBackend`, `EMRIHDFBackend`; each extends `eryn.backends.HDFBackend`
  - `src/lisatools/globalfit/stock/erebor.py` — reads `reset_kwargs` and band metadata from HDF5 at startup
- **Stored data:** sampler states, log-likelihoods, chain samples, band edges, reset kwargs, per-source model branches
- **Access pattern:** Direct `h5py` dataset reads for metadata (e.g., `num_bands`, `band_edges`); `eryn` `HDFBackend` API for chain samples with optional thinning/discarding

---

## External Data Downloads

- **Mechanism:** `requests.get()` in `src/lisatools/detector.py`
- **Source:** GitHub (orbit data files)
- **Trigger:** Missing orbit file at the path expected by `Orbits.__init__`
- **Storage:** Saved locally before `h5py.File()` opens it; location determined by `platformdirs` config directory

---

## MPI Parallelism

- **Library:** `mpi4py`
- **Usage:** Distributed global fit runs; imported in `src/lisatools/globalfit/run.py` and `src/lisatools/globalfit/pipeline.py`
- **Optional dependency:** `[optional-dependencies.sampling]` group
- **Pattern:** Standard MPI communicator (`MPI.COMM_WORLD`) for splitting work across nodes; Eryn sampler is MPI-aware

---

## JAX Integration

- **Library:** `jax`, `jax.numpy`
- **Usage:** `src/lisatools/detector.py` and `src/lisatools/mojito_detector.py` — `LISAToolsJAXL1Orbits` class uses JAX for differentiable orbit computations
- **Pattern:** Lazy import guarded by `try/except ImportError`; only used when JAX is available
- **Not listed** in `pyproject.toml` — optional / experimental

---

## Data Formats Produced

| Format | Purpose | Produced by |
|---|---|---|
| HDF5 (`.hdf`, `.h5`) | MCMC chain files | `globalfit/hdfbackend.py` via eryn |
| HDF5 | State snapshots | `globalfit/engine.py` |
| NumPy / CuPy arrays | In-memory data passing between components | Throughout |
| Matplotlib figures (`.png`, `.pdf`) | Diagnostic plots | `diagnostic.py`, `globalfit/plot.py`, `globalfit/diagnosticplot.py` |

---

## Python Index Sources

Three PyPI indexes are configured in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"

[[tool.uv.index]]
name = "gitlab-esa-commons"
url = "https://gitlab.esa.int/api/v4/groups/29349/-/packages/pypi/simple"

[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
```

- `mojito` comes from ESA GitLab (requires ESA GitLab credentials or network access)
- `lisaconstants` comes from public PyPI
- `mojito-processor` comes from TestPyPI

---

*Integration audit: 2026-04-28*
