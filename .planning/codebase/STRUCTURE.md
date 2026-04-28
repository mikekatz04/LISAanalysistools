# Codebase Structure

**Analysis Date:** 2026-04-28

## Directory Layout

```
LISAanalysistools/
├── src/lisatools/              # Main Python package source
│   ├── __init__.py             # Package init; backend registration
│   ├── _version.py             # Auto-generated version string
│   ├── _editable.py            # Editable install marker
│   ├── analysiscontainer.py    # AnalysisContainer: data + sensitivity + likelihood orchestration
│   ├── datacontainer.py        # DataResidualArray: multi-channel data management
│   ├── detector.py             # Orbits, L1Orbits: LISA orbital geometry
│   ├── sensitivity.py          # Sensitivity, SensitivityMatrix: PSD and noise models
│   ├── diagnostic.py           # Likelihood calculations and inner products
│   ├── domains.py              # Frequency/time domain settings and transforms
│   ├── stochastic.py           # Stochastic background modeling
│   ├── mojito_detector.py      # Mojito-based detector model interface
│   ├── CMakeLists.txt          # C++ build config for this package
│   ├── cutils/                 # C++/CUDA performance-critical extensions
│   ├── globalfit/              # Global fit pipeline
│   ├── orbit_files/            # Bundled HDF5 orbit data files
│   ├── sampling/               # MCMC sampling infrastructure
│   ├── sources/                # Gravitational wave source waveform models
│   └── utils/                  # Shared Python utilities
├── tests/                      # Pytest test suite
├── examples/                   # Jupyter notebooks demonstrating usage
├── global_fit_input/           # Global fit scenario configuration scripts
├── scripts/                    # Dev/build/CI helper scripts
├── docs/                       # Sphinx documentation source
├── mojito_input/               # Mojito detector input files
├── pyproject.toml              # Package metadata, dependencies, build config
├── CMakeLists.txt              # Root C++ build configuration
└── uv.lock                     # Locked dependency manifest
```

## Directory Purposes

**`src/lisatools/cutils/`:**
- Purpose: Performance-critical code compiled as a Python extension module
- Contains: CUDA kernels (`Detector.cu`, `PSD.cu`), C++ headers (`Detector.hpp`, `PSD.hpp`, `cuda_complex.hpp`, `binding.hpp`), pybind11 binding entry point (`binding.cxx`), CMake source list (`lisatools_sources.cmake`), and a pure-Python PSD likelihood helper (`psd_likelihood_utils.py`)
- Key files: `src/lisatools/cutils/binding.cxx`, `src/lisatools/cutils/Detector.cu`, `src/lisatools/cutils/PSD.cu`

**`src/lisatools/globalfit/`:**
- Purpose: Full global fit pipeline from data ingestion through posterior sampling and postprocessing
- Contains: pipeline orchestration (`pipeline.py`, `run.py`, `engine.py`), per-source fitting modules (`psdglobal.py`, `galaxyglobal.py`, `mbhglobal.py`, `mbhsearch.py`), catalog building (`buildcatalog.py`, `gathergalaxy.py`), waveform generation helpers (`generatefuncs.py`), state management (`state.py`), HDF5 backend (`hdfbackend.py`), recipe system (`recipe.py`, `recipe_steps.py`), pre/postprocessing (`preprocessing.py`, `postprocessing.py`), diagnostics and plotting (`diagnosticplot.py`, `plot.py`, `loginfo.py`), shared utilities (`utils.py`)
- Key subdirectories: `moves/` (custom MCMC proposal moves), `stock/` (pre-built pipeline configurations, e.g. `erebor.py`)

**`src/lisatools/globalfit/moves/`:**
- Purpose: Custom MCMC moves for the global fit sampler
- Contains: `globalfitmove.py` (base), `addremovemove.py`, `psdmove.py`, `gbspecialstretch.py`, `mbhspecialmove.py`

**`src/lisatools/globalfit/stock/`:**
- Purpose: Pre-assembled pipeline configurations for standard runs
- Contains: `erebor.py` (Erebor cluster configuration)

**`src/lisatools/sampling/`:**
- Purpose: Generic MCMC sampling infrastructure wrapping Eryn
- Contains: `likelihood.py`, `prior.py`, `gmm.py`, `stopping.py`, `utility.py`, `moves/` (generic sampling moves, `skymodehop.py`)

**`src/lisatools/sources/`:**
- Purpose: Gravitational wave source waveform models
- Contains: `waveformbase.py` (base class), `utils.py`, `defaultresponse.py`, per-source subdirectories
- Subdirectories:
  - `bbh/`: Binary black hole waveforms (`waveform.py`)
  - `emri/`: Extreme mass ratio inspiral waveforms (`waveform.py`)
  - `gb/`: Galactic binary waveforms (`waveform.py`, `chebyshevwave.py`)

**`src/lisatools/utils/`:**
- Purpose: Shared Python utilities used across modules
- Contains: `utility.py` (array module helpers, `get_array_module()`), `constants.py`, `exceptions.py`, `multigpudataholder.py`, `parallelbase.py`

**`src/lisatools/orbit_files/`:**
- Purpose: Bundled LISA orbit data shipped with the package
- Contains: HDF5 orbit files (`equalarmlength-orbits.h5`, `equalarmlength-orbits-best-fit-to-esa.h5`, `esa-trailing-orbits.h5`)
- Generated: No (static data files, committed to repo)

**`tests/`:**
- Purpose: Pytest test suite
- Key files: `test_detector.py`, `test_orbits.py`, `test_sensitivity.py`, `test_sources_utils.py`, `test_fresnel.py`, `test_get_amp_phase_shifting.py`

**`global_fit_input/`:**
- Purpose: Python scripts that define run-specific global fit configuration objects passed to the pipeline
- Key files: `global_fit_settings.py`, `emri_and_psd_global_fit_setting.py`, `mbh_only_global_fit_settings.py`, `gb_and_foreground_global_fit_settings.py`, `psd_only_global_fit_settings.py`, `emri_only_global_fit_settings.py`, `emri_mbh_psd_global_fit_setting.py`

**`scripts/`:**
- Purpose: Dev and CI helpers
- Key files: `run-tests.bash`, `run_global.py`, `prebuild.py`, `install-hooks.bash`, `data_download.py`

**`examples/`:**
- Purpose: Jupyter notebooks illustrating package usage
- Key files: `lisatools_tutorial.ipynb`, `Sensitivity.ipynb`, `Diagnostics.ipynb`

## Key File Locations

**Entry Points:**
- `src/lisatools/__init__.py`: Package init; registers CPU/CUDA backends via `gpubackendtools.Globals()`
- `scripts/run_global.py`: CLI entry point for launching a global fit run
- `src/lisatools/globalfit/run.py`: Core run function invoked by the global fit CLI

**Configuration:**
- `pyproject.toml`: Package metadata, dependencies, build system settings
- `CMakeLists.txt`: Root C++ build configuration
- `src/lisatools/CMakeLists.txt`: Per-package C++ build config
- `src/lisatools/cutils/lisatools_sources.cmake`: CMake source list for C++ extensions
- `global_fit_input/*.py`: Per-scenario global fit settings

**Core Scientific Logic:**
- `src/lisatools/analysiscontainer.py`: Combines data + sensitivity; orchestrates likelihoods
- `src/lisatools/datacontainer.py`: Multi-channel data arrays, residual management
- `src/lisatools/sensitivity.py`: PSD and sensitivity curves
- `src/lisatools/detector.py`: Orbital geometry and armlengths
- `src/lisatools/diagnostic.py`: Inner products and likelihood calculations
- `src/lisatools/domains.py`: Domain transform infrastructure
- `src/lisatools/stochastic.py`: Stochastic background models

**Global Fit Pipeline:**
- `src/lisatools/globalfit/pipeline.py`: High-level pipeline orchestration
- `src/lisatools/globalfit/engine.py`: Sampler engine
- `src/lisatools/globalfit/recipe.py` / `recipe_steps.py`: Modular run recipe system
- `src/lisatools/globalfit/state.py`: Run state management
- `src/lisatools/globalfit/postprocessing.py`: Post-run analysis
- `src/lisatools/globalfit/stock/erebor.py`: Pre-built Erebor cluster configuration

**Testing:**
- `tests/test_detector.py`, `tests/test_orbits.py`, `tests/test_sensitivity.py`
- `tests/test_sources_utils.py`, `tests/test_fresnel.py`, `tests/test_get_amp_phase_shifting.py`

## Naming Conventions

**Files:**
- Lowercase with underscores: `analysiscontainer.py`, `datacontainer.py`, `psd_likelihood_utils.py`
- Module names match the primary class or concern they expose: `sensitivity.py` → `Sensitivity`

**Directories:**
- Lowercase, no separators: `globalfit/`, `cutils/`, `sources/`
- Source subdirectories named after source type: `bbh/`, `emri/`, `gb/`

**Classes:**
- PascalCase: `DataResidualArray`, `AnalysisContainer`, `SensitivityMatrix`, `BaseDomainComputationGroup`

**Functions and Variables:**
- `snake_case` throughout

## Where to Add New Code

**New gravitational wave source type:**
- Create a subdirectory under `src/lisatools/sources/<source_type>/`
- Add `__init__.py` and `waveform.py` implementing a class derived from `src/lisatools/sources/waveformbase.py`
- Register in `src/lisatools/sources/__init__.py`

**New global fit module (per-source fitter):**
- Add `<source>global.py` in `src/lisatools/globalfit/`
- Follow pattern of `src/lisatools/globalfit/psdglobal.py` or `galaxyglobal.py`

**New MCMC move for the global fit:**
- Add file in `src/lisatools/globalfit/moves/`
- Derive from base in `src/lisatools/globalfit/moves/globalfitmove.py`

**New generic sampling move:**
- Add file in `src/lisatools/sampling/moves/`

**New C++ / CUDA kernel:**
- Add `.cu` / `.hpp` sources in `src/lisatools/cutils/`
- Register in `src/lisatools/cutils/lisatools_sources.cmake`
- Add pybind11 binding in `src/lisatools/cutils/binding.cxx` using helpers from `src/lisatools/cutils/binding.hpp`

**New utility function:**
- Shared helpers with no domain dependency: `src/lisatools/utils/utility.py`
- Domain-specific helpers: relevant module file (e.g., `src/lisatools/globalfit/utils.py`)

**New test:**
- Add `tests/test_<module>.py` following existing patterns in `tests/`
- Run with: `uv run pytest tests/`

**New global fit scenario configuration:**
- Add `<scenario>_global_fit_settings.py` in `global_fit_input/`
- Follow pattern of `global_fit_input/psd_only_global_fit_settings.py`

## Special Directories

**`src/lisatools/orbit_files/`:**
- Purpose: Bundled HDF5 LISA orbit data
- Generated: No — Committed: Yes

**`build/`:**
- Purpose: C++ extension build artifacts
- Generated: Yes — Committed: No

**`.venv/`:**
- Purpose: uv-managed virtual environment
- Generated: Yes — Committed: No

**`.planning/codebase/`:**
- Purpose: GSD codebase map documents
- Generated: Yes (by mapping agents) — Committed: Optional

**`docs/`:**
- Purpose: Sphinx documentation source and build output
- Generated: Partially (build output generated) — Source files committed

---

*Structure analysis: 2026-04-28*
