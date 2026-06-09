# LISA Analysis Tools — Development Information

This page is the entry point for developers working on the LISA Analysis
Tools (LAT) software stack. It explains how the repos relate, where to
find what, and how the end-to-end LISA global-fit pipeline is built from
the same primitives used for one-line SNR computations.

## Repository map

The organization hosts six repos. The flagship repo at the org-name URL
— `lisa-analysis-tools/lisa-analysis-tools` — is the central LAT
library and also hosts the cross-stack installer and this development
page. Every other sub-package can be used standalone; together they
form the full LISA-data-analysis stack.

```
github.com/lisa-analysis-tools/                  (this org)
├── lisa-analysis-tools/   ← LAT — central library + install.sh + DEVELOPMENT.md
├── GPUBackendTools/       ← GBT — GPU primitives (splines, backends, cuda_complex)
├── Eryn/                  ←       trans-dimensional MCMC sampler
├── BBHx/                  ←       MBH + SOBBH waveforms & physics
├── GBGPU/                 ←       Galactic-binary (UCB) waveforms & physics
└── LATW/                  ←       LISA Analysis Tools Workshop tutorials
```

Two repos that the LAT stack interoperates with but that live outside the
org are `BlackHolePerturbationToolkit/FastEMRIWaveforms` (EMRIs) and
`asantini29/phentax` (MBH IMRPhenomTHM). The central `install.sh` (in
this repo) brings these in alongside the org-owned packages.

### Full dev install

```bash
git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
cd lisa-analysis-tools
./install.sh        # clones siblings, editable-installs everything
```

The installer is re-runnable, clones sibling repos into the parent
directory of this clone, applies the sprint-wide pybind11 pin via
`constraints/sprint.txt`, and installs in dependency order
(GBT → Eryn → LAT → BBHx + GBGPU → LATW + externals).

## What lives where

### `lisa-analysis-tools` (LAT) — the central LISA library

LAT holds everything that is LISA-physics-generic. It spans the full
range of computations a user might want, from a one-line SNR to a
multi-source global fit, and the same objects work in both regimes
because the computational organization is hidden behind the same API.

What LAT owns:

- **Detector + response**: orbits, TDI configs, on-the-fly TDI, FD/WDM/TD domains.
- **Inner-product machinery**: sensitivity matrices, `AnalysisContainer`, swap-likelihood kernels.
- **Sampling-adjacent infrastructure** that is source-agnostic: residual moves, likelihood-engine base classes, settings loaders, global-fit run harness.
- **The JAX mirror** of all of the above (in-progress).

```
src/lisatools/
├── response/        ← directresponse, tdionfly, tdiconfig, parallelbase
├── jax/             ← JAX backend (response/, wdm/, sources/)
├── cutils/          ← C++/CUDA source (LISAResponse, TDIonTheFly, kernels)
├── sources/         ← source-class base classes (not source-specific physics)
├── sampling/        ← residual moves, likelihood engines
├── analysiscontainer.py
└── ...
```

The rule that follows from this: any LISA computation that is not
specific to one source class (GB / SOBBH / MBH / EMRI) lives in LAT. The
per-source-class repos depend on LAT, never the other way around.

### `GPUBackendTools` (GBT) — GPU primitives + backend management

GBT is the foundation. It provides the GPU/CPU primitives that everyone
else builds on:

- `CubicSpline` evaluators (used by orbits, sensitivity, response splines).
- `cuda_complex.hpp` — the single sprint-wide complex header.
- The backend resolver (`get_backend("cpu" | "cuda12x" | ...)`) that loads
  per-backend wheels at runtime.
- LAPACKE / cuBLAS / cuSolver glue.

```
GPUBackendTools/src/gpubackendtools/
├── cutils/
│   ├── cuda_complex.hpp        ← sole source for cuda_complex
│   ├── Interpolate.cu          ← cubic-spline kernels
│   ├── InterpolateDevice.hh    ← header-only evaluators
│   └── GPUBackendToolsConfig.cmake
└── ...
```

### `Eryn` — the sampler

Eryn is a trans-dimensional ensemble MCMC sampler. The LAT global fit
runs on top of Eryn — every PE step (per-source MCMC, RJ-MCMC for source
counts, PSD updates) is an Eryn move acting on an LAT-provided likelihood.

### `BBHx` — Massive-black-hole and SOBBH source classes

BBHx is source-class-specific: it owns MBH (PhenomHM, PhenomTHM via
phentax) and SOBBH waveforms, their parameter spaces, and the
SOBBH-specific TDIonTheFly + ComputationGroup wraps. Its C++ depends on
LAT (templated chunked-het kernels) and GBT (splines, complex).

### `GBGPU` — Galactic-binary source class

Same shape as BBHx but for GB (UCB) waveforms. Owns the GB TDIonTheFly
+ ComputationGroup wraps and the UCB physics layer. Its C++ depends on
LAT and GBT.

### `LATW` — LISA Analysis Tools Workshop

LATW carries the older, simpler tutorial set — single-source SNR /
likelihood examples, a walkthrough of the response, an introductory
sampling notebook. It is the on-ramp before users dive into the
global-fit settings files. LATW continues to ship as a companion repo
alongside the code packages.

## C-code dependency direction

C/C++ source flows in one direction only:

```
        GBT
         │
         ▼
        LAT
         │
   ┌─────┴─────┐
   ▼           ▼
  BBHx       GBGPU
```

In practice this means: source-agnostic kernels that started in
`lisa-on-gpu` or in a source-class repo have been moved up the chain into
LAT (e.g. all the templated chunked-heterodyne kernels) so the
source-class repos consume them via include. The reverse — LAT pulling
in BBHx- or GBGPU-specific code — is forbidden.

## Backend hierarchy (GPU C++ → CPU C++ → JAX)

Every algorithm exists in up to three backends. The implementation
order is fixed:

1. **GPU C++ (CUDA) leads.** Canonical performance target and reference
   implementation. New algorithms are designed for the GPU first.
2. **CPU C++ mirrors GPU C++ as closely as possible** — same kernel
   structure, same algorithm, bridged by `#ifdef __CUDACC__` and shared
   macros (`CUDA_SHARED`, `THREAD_START_X`, `BLOCK_INCR_X`). The CPU
   path exists for testing, validation, and CPU-only environments. It
   must not diverge in algorithm beyond floating-point order of
   operations.
3. **JAX is allowed to diverge internally.** It does *not* mechanically
   translate the C++ kernel structure (shared memory, block tiling).
   Instead it uses JAX-native idioms — `jax.lax.scan` for outer loops,
   `jax.vmap` for inner batched work, static-shape `dynamic_slice`s, and
   functional carries. JAX is held to a single contract: end-to-end
   inner-product outputs (`<d|h>`, `<h|h>`, the 5 swap_ll terms) must
   match the CPU/GPU C++ to ≤ 1e-12 relative on representative cases.

Workflow for a new feature:

```
write GPU CUDA  →  mirror CPU under #ifdef  →  port to JAX (JAX-native)
                                                  └──> validate at <d|h>
```

## Same objects, small scale to large scale

The same primitives drive a 10-line SNR script and a multi-thousand-core
global-fit run. A user composes them differently — the library does not
have a "small" mode and a "large" mode.

Small-scale example (single template, one inner product):

```python
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sources.gb import GBWaveform

ac = AnalysisContainer(data, settings, signal_gen=GBWaveform(...))
snr = ac.snr(params)
ll  = ac.template_inner_product(params)
```

Large-scale global fit (same building blocks, wrapped in Eryn moves):

```python
# global_fit_input/full_year_combined_global_fit_settings.py defines:
#   - per-source likelihood engines (each holds an AnalysisContainer)
#   - per-source Eryn move (StretchMove inside ResidualAddOneRemoveOneMove)
#   - shared data (synthetic noise + galactic foreground)

python run_mpi_global_fit.py \
    --settings global_fit_input/full_year_combined_global_fit_settings.py
```

The global-fit harness's job is plumbing: route templates to the right
likelihood, manage residual subtraction between sources, hand
trans-dimensional state to Eryn. The physics objects are unchanged.

## Running things

### Scripts

Reusable development scripts live in `scripts/`,
organized by topic:

```
scripts/
├── benchmark/        ← perf measurements
├── diagnostics/      ← waveform / response sanity checks
├── emri/             ← EMRI-specific dev scripts
├── gb_chunked_het/   ← chunked-heterodyne GB development (incl. signal_het_cpp/)
├── gb_lookup/        ← lookup-table GB development
├── mbh/              ← MBH dev scripts
├── sobbh/            ← SOBBH dev scripts
├── notes/            ← scratch notes per topic
├── validation/       ← cross-backend validation scripts
├── wdm/              ← WDM-domain dev scripts
└── run_global.py     ← global-fit launcher (single-process)
```

Source-class-specific scripts that depend on physics from BBHx or GBGPU
may live in those repos instead — convention is "where the source class
lives, the scripts live too."

### Global-fit runs

A global-fit run is fully specified by **one settings file** plus a
runner. Settings files live in `global_fit_input/`:

```
global_fit_input/
├── full_year_combined_global_fit_settings.py    ← EMRI + SOBBH + MBH
├── combined_gb_psd_emri_sobbh_global_fit_settings.py
├── mbh_only_global_fit_settings.py
├── sobbh_only_global_fit_settings.py
├── emri_only_global_fit_settings.py
├── gb_and_foreground_global_fit_settings.py
├── psd_only_global_fit_settings.py
└── global_fit_settings.py                       ← shared defaults
```

Each settings file declares: data source (mojito loader or synthetic
generator), which source classes participate, per-source Eryn move
configuration, instrument / foreground noise model, and GPU device list.

To launch:

```bash
# single-process (dev / smoke)
python scripts/run_global.py \
    --settings global_fit_input/full_year_combined_global_fit_settings.py

# MPI (production)
mpirun -n <N> python LISAanalysistools/run_mpi_global_fit.py \
    --settings global_fit_input/full_year_combined_global_fit_settings.py
```

GPUs are picked up automatically when cupy is installed and the settings
file sets `GPUS = [<dev_id>]`.

### Installing the full development environment

This repo ships `install.sh`, the central one-shot installer for the
full LAT stack. Quickstart:

```bash
# inside a fresh virtualenv / conda env
git clone https://github.com/lisa-analysis-tools/lisa-analysis-tools.git
cd lisa-analysis-tools
./install.sh
```

What `install.sh` does, in order:

1. Sets `PIP_CONSTRAINT=$PWD/constraints/sprint.txt` so every subsequent
   `pip install` honors the sprint-wide pybind11 pin (the L1 enforcement
   for the single-registrant rule).
2. Installs base build dependencies: `scikit_build_core`, `setuptools_scm`,
   `pybind11`, `numpy`, `scipy`, `astropy`, `lisaconstants`, `Cython`, etc.
3. **Clones each sibling repo into the parent directory of this clone**,
   then editable-installs in dependency order:

   ```
   GPUBackendTools  →  Eryn  →  lisa-analysis-tools (this repo)
                       →  BBHx  →  GBGPU
                       →  LATW (clone only; no compile)
   ```

4. Installs the external collaborators: `phentax` (MBH PhenomTHM) and
   `FastEMRIWaveforms` (EMRIs).

The resulting on-disk layout next to your clone:

```
<dev_root>/
├── lisa-analysis-tools/     ← this repo (LAT)
├── GPUBackendTools/         ← GBT
├── Eryn/
├── BBHx/
├── GBGPU/
├── LATW/                    ← tutorials (no build)
└── FastEMRIWaveforms/       ← optional
```

`install.sh` is **re-runnable**. Existing sibling clones are reused,
not re-cloned — so a re-run after `git pull` in any sub-repo just
reinstalls everything against the new state.

#### Skipping optional pieces

```bash
SKIP_FEW=1      ./install.sh    # skip FastEMRIWaveforms
SKIP_PHENTAX=1  ./install.sh    # skip phentax (MBH PhenomTHM)
```

The retiring `lisa-on-gpu` is **off by default**; set
`SKIP_LISA_ON_GPU=0` to opt back in for backwards-compat testing.

#### Pre-flight notes

- **CUDA / GPU**: the editable installs auto-detect CUDA. To target a
  specific CUDA major, install the matching `cupy-cudaXXx` wheel
  separately. CUDA arches default to `70;75;80;86;90`.
- **LAPACKE**: passed via
  `--config-settings=cmake.define.GBT_LAPACKE_FETCH=ON`, which makes
  the build fetch a vendored LAPACKE if the system copy isn't found.
  On macOS + brew, uncomment the `PKG_CONFIG_PATH` block at the top of
  `install.sh` to point at your `lapack` keg.
- **Python ≥ 3.12** is required by LAT's `pyproject.toml`.

#### Verifying the install

```bash
python -c "import lisatools, eryn, bbhx, gbgpu, gpubackendtools; \
           print('all import OK')"
```

For a fuller smoke-test, run any of the LATW tutorial notebooks
(`../LATW/notebooks/01_basics/`).

## Tutorials (LATW)

The Workshop repo (`LATW`) hosts the on-ramp examples. Layout:

```
LATW/
├── notebooks/        ← topic-ordered tutorial notebooks
│   ├── 01_basics/        SNR, inner products, sensitivity
│   ├── 02_response/      LISA response + TDI walkthrough
│   ├── 03_sources/       per-source-class quickstarts (GB, MBH, EMRI, SOBBH)
│   ├── 04_sampling/      single-source MCMC with Eryn
│   └── 05_global_fit/    minimal global-fit example
└── data/             ← small example data files
```

LATW tutorials use the same objects as `scripts/` and the global-fit
harness — they are deliberately the simpler entry path, not a separate
API. Once a user is past the LATW examples, the leap to a real global
fit is changing which settings file is loaded, not learning new code.

## Sprint-wide rules of the road

For full detail see `CLAUDE.md`. The short version, in order of how
often they bite:

1. **Backend is fixed at instantiation, never a method kwarg.** Pass
   `force_backend="cpu" | "cuda12x" | "jax"` to the constructor; use
   `self.backend` / `self.backend.xp` to dispatch. No `backend=` kwargs
   on methods.
2. **CPU/GPU class-name aliasing.** Every class compiled into both the
   CPU and GPU shared object needs `#define FooWrap FooWrapCPU/GPU`
   blocks at the top of its header — otherwise the two backends collide
   in pybind11's type registry.
3. **Host→device wrapper upload.** Wrapper objects (`Orbits`,
   `TDIConfig`, `WDMSettings`, …) are `new`'d on the host. Every CUDA
   kernel that dereferences one needs an explicit `cudaMalloc` +
   `cudaMemcpy(..., cudaMemcpyHostToDevice)` of the struct before
   launch.
4. **Single registrant for pybind11 wrappers.** Each `*Wrap` is
   registered in exactly one .so. `LISATOOLS_IS_WRAPPER_OWNER` + the
   `tools/check_single_registrant.sh` grep gate enforce this.
5. **C-code direction is GBT → LAT → BBHx/GBGPU.** Never the reverse.
