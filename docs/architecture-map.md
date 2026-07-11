_Last mapped: 8d05c32 · 2026-07-10 · regenerate when repo structure changes_

# LISA Analysis Tools — Cross-Repo Architecture Map

> Verified against source (pyproject deps, top-level `__init__.py`, `import`
> statements, CMakeLists include dirs), **not** against docs. Where existing
> docs contradict the code, the code wins — see §8.

## 1. What LISA Analysis Tools is

A GPU/CPU-accelerated toolchain for **LISA gravitational-wave data analysis**:
generate LISA TDI waveforms for every source class (Galactic Binaries, massive
black-hole binaries, stellar-origin BBHs, EMRIs), run the LISA instrument
response, form WDM-wavelet / frequency-domain likelihoods, and drive Bayesian
global-fit MCMC. It is 7 separate git repos developed together. The project just
finished a large multi-repo reorg (Phases 3B–3M) that made **LISAanalysistools
(LAT)** the central LISA-physics library, pushed all shared compute
infrastructure down into **GPUBackendTools (GBT)**, moved source-specific C++
kernels into their waveform repos, and hollowed out `lisa-on-gpu` into a
pure-Python deprecation husk.

## 2. The repos

| Repo | Python pkg | Role / status |
|---|---|---|
| **GPUBackendTools** | `gpubackendtools` | **Foundation library.** Backend registry + `Backend`/`has_backend`/`get_backend`, cubic-spline interpolation C++ (sole registrant of `CubicSplineWrap`), `cuda_complex.hpp`, `ParallelModuleBase`. Everyone depends on it. (branch `spline`) |
| **LISAanalysistools** | `lisatools` | **Central LISA-physics library, active.** Response/TDI (Python+JAX+C++), WDM domains, TDI-on-the-fly base C++, sensitivity/PSD, stock source wrappers, global-fit orchestration. (branch `dev`) |
| **GBGPU** | `gbgpu` | **Active waveform repo** — Galactic-Binary waveforms + GB TDI-on-the-fly / signal-het / chunked-het C++. (branch `dev`) |
| **BBHx** | `bbhx` | **Active waveform repo** — MBH (PhenomHM/PhenomT) + SOBBH waveforms, MBH/SOBBH TDI-on-the-fly C++. (branch `dev`) |
| **FastEMRIWaveforms** | `few` | **Active waveform repo** — EMRI waveforms (Cython cutils; kept its own diverged `cuda_complex.hpp`). (branch `gpu_backend`) |
| **Eryn** | `eryn` | **Active sampler library** — ensemble/parallel-tempered MCMC, RJMCMC, NUTS, `paraensemble`. (branch `dev`) |
| **lisa-on-gpu** | `fastlisaresponse` | **DEPRECATED pure-Python husk.** CMake is `LANGUAGES NONE`; `find … -name '*.cu/*.cxx/*.hpp/*.hh/*.pyx'` under `src/` returns **0**. Only `__init__.py` (DeprecationWarning → `lisatools.response`/`gbgpu`/`bbhx`) + packaging stubs remain. (The `.cxx` files under `bench/build/` are untracked build artifacts, not source.) (branch `tdi_on_fly`) |

## 3. Dependency graph

Hard deps (from `pyproject.toml` `dependencies`) and runtime imports:

```
                         gpubackendtools   (foundation: backends, spline, cuda_complex)
                          ▲   ▲   ▲   ▲  ▲
        ┌─────────────────┘   │   │   │  └──────────────┐
        │                     │   │   └────────┐        │
      eryn            lisaanalysistools │       │       few
   (paraensemble         (lisatools)   │       │   (imports gpubackendtools.
    backend registry)     ▲    ▲       │       │    globals at runtime;
                          │    │       │       │    lisatools optional)
                    ┌─────┘    └────┐  │       │
                  gbgpu           bbhx │       │
              (→ lisatools,   (→ lisatools,    │
                 gpubackendtools) gpubackendtools)
                    ▲                  ▲
                    └── soft/optional ─┘
   lisatools.sources.{gb,sobbh,bbh,emri} import gbgpu / bbhx / few
   lazily (stock waveform wrappers) → soft back-edge, not a pyproject dep.
```

- **Native C++ link edges (CMake):** GBGPU and BBHx `cutils/CMakeLists.txt`
  resolve `${GBT_CUTILS}` (via `gpubackendtools.get_include()`) and
  **copy-compile GBT's `Interpolate.cu`**; they also compile LAT-side
  TDIonTheFly `.cu` so the nanobind virtual-class typeinfo resolves. LAT ships
  `cutils/LISAanalysisToolsConfig.cmake` for downstreams.
- **Soft cycle:** `gbgpu/bbhx → lisatools` (hard) while
  `lisatools.sources.* → gbgpu/bbhx/few` (lazy import inside stock wrappers).
  Not a real cycle at install time because the LAT→waveform edge is optional.
- `lisa-on-gpu/fastlisaresponse` has **no runtime deps** — it only warns.

## 4. Capability → location map (verified against code)

| Capability | Module path(s) |
|---|---|
| **LISA response / projection** | Py: `lisatools.response.directresponse` (`pyResponseTDI`, `ResponseWrapper`), `lisatools.response.parallelbase`. JAX: `lisatools.jax.response`. C++: `lisatools/cutils/LISAResponse.{cu,hh}`, `Detector.{cu,hpp}` |
| **TDI** | Py: `lisatools.response.tdiconfig` (`TDIConfig`). C++: `LISAResponse.cu` + `lat_tdi_on_the_fly.{cu,hh}` |
| **WDM wavelet transforms** | Py: `lisatools.domains` (`WDMSettings`), `lisatools.wdm_het`. C++: `lisatools/cutils/wdm_settings.hh`, `wdm_domain.hh`, `lat_wdm_fft.hh`. JAX: `lisatools.jax.wdm` |
| **TDI-on-the-fly (base)** | Py: `lisatools.response.tdionfly`. C++: `lisatools/cutils/lat_tdi_on_the_fly.{cu,hh}`, `lat_spline_tdi_waveform.{cu,hh}`, `lat_chunked_het_kernels.hh` |
| **GB waveform** | `gbgpu.gbgpu.GBGPU` (+ `GBGPUBase`); C++ `gbgpu/cutils/gb_tdi_on_the_fly.{cu,hh}`, `SharedMemoryGBGPU.*`. Stock TDI wrapper: `lisatools.sources.gb.waveform.GBAETWaveform` |
| **MBH waveform** | `bbhx` (`bbhx/cutils/PhenomHMWaveform.{cu,hh}`, `Response.*`, `WaveformBuild.*`), `bbhx.mbhtdionfly`, `bbhx.mbhphentax`. Stock wrapper: `lisatools.sources.bbh.waveform` |
| **SOBBH waveform** | `bbhx.sobbhtdionfly`, `bbhx.sobbhcomps`; C++ `bbhx/cutils/sobbh_tdi_on_the_fly.{cu,hh}`. PN math + wrapper: `lisatools.sources.sobbh.waveform` |
| **EMRI waveform** | `few.waveform.waveform.GenerateEMRIWaveform`; C++/Cython in `few/cutils/`. LISA-TDI wrappers: `lisatools.sources.emri.{waveform,emritdionfly,response}` |
| **Samplers / MCMC** | `eryn.ensemble`, `eryn.paraensemble`, `eryn.moves.*` (stretch, nuts, rj, tempering, distgen, …), `eryn.backends.*`. Global-fit orchestration: `lisatools.globalfit.*` (`engine`, `run`, `recipe`, `moves/`, `stock/`) |
| **Interpolation + compute backends** | `gpubackendtools.interpolate` (`CubicSplineInterpolant`); C++ `gpubackendtools/cutils/Interpolate.{cu,hh}` (SOLE `CubicSplineWrap` registrant). Backend machinery: `gpubackendtools.gpubackendtools` (`Backend`, `BackendsManager`, `CpuBackend`/`Cuda1{1,2,3}xBackend`) |
| **Signal-het / chunked-het likelihood** | Py base: `lisatools.signal_het`, `lisatools.chunked_het` (`WDMComputationsBase`). GB: `gbgpu.gbcomps` (`GBWDMComputations`, `GBFDComputations`), `gbgpu.gbsignalhetcomputations`; JAX kernels `gbgpu.jax.wdm.signal_het_kernels`. SOBBH: `bbhx.sobbhcomps` |

## 5. Backend-wheel model

Each native package ships as a **pure-Python "core" wheel** + **separate
per-backend plugin wheels**. Verified in `gpubackendtools.gpubackendtools` and
each `lisatools/gbgpu/bbhx/few` `cutils/__init__.py`:

- **Plugin wheel names:** `lisaanalysistools-cpu`, `-cuda11x/-cuda12x/-cuda13x`
  (same pattern for `gbgpu-*`, `bbhx-*`, `fastemriwaveforms-*`). Each installs a
  **top-level import module** `<pkg>_backend_<flavor>` containing the compiled
  nanobind module `.pycppdetector` (LAT) / `.cgbgpu` (GBGPU) / `.cbbhx` (BBHx) /
  Cython modules (FEW).
- **Registry:** `gpubackendtools` owns a global `BackendsManager` (`_registry`
  keyed by full name like `lisatools_backend_cuda12x`), reached via
  `Globals().backends_manager`. Each package registers its flavors at import:
  `Globals().backends_manager.add_backends({"few_cpu": FEWCpuBackend, ...})`.
- **Per-package facade:** `lisatools.get_backend("cpu")` →
  `gpubackendtools.get_backend("lisatools_cpu")`; likewise `gbgpu_*`, `bbhx_*`,
  `few_*`. `has_backend(...)` mirrors it.
- **Selection at construction, never per-method** (LISA Analysis Tools–wide rule): pass
  `force_backend="cpu"|"cuda12x"|"jax"`; `ParallelModuleBase` (GBT) exposes
  `self.backend` / `self.backend.xp` / `self.backend.name`.
- **CPU + CUDA co-load in one interpreter:** the CPU/GPU `#define FooWrap{CPU,GPU}`
  aliasing (CLAUDE.md rule) gives each backend distinct mangled C++ symbols, so
  `has_backend("cpu")` and `has_backend("cuda12x")` can both be True at once.
- `Backend.__reduce__` pickles by registry name (deepcopy/pickle-safe); array
  module `xp` is a property, never stored.

## 6. Build order / how to build the stack

Bottom-up (each layer needs the previous compiled + installed):

1. **GPUBackendTools** — provides `get_include()` cutils dir (`${GBT_CUTILS}`),
   `Interpolate.cu`, `cuda_complex.hpp`, `gbt_global.h`, backend registry.
2. **LISAanalysistools** — needs GBT. Compiles LISA-response + TDIonTheFly base
   `.cu`; installs `LISAanalysisToolsConfig.cmake`. Binding layer is **nanobind**
   (Phase 3M).
3. **GBGPU** and **BBHx** — need GBT **and** LAT. Include `${GBT_CUTILS}`,
   copy-compile GBT `Interpolate.cu`, and compile LAT-side TDIonTheFly `.cu` for
   typeinfo. (`gbgpu` also has a hard `lisaanalysistools>=1.0.17` pin.)
4. **FastEMRIWaveforms** — needs GBT (imports `gpubackendtools.globals`);
   otherwise self-contained Cython cutils.
5. **Eryn** — pure Python; needs GBT for `paraensemble` backend access.
6. **lisa-on-gpu / fastlisaresponse** — optional, install last; pure-Python
   deprecation husk, nothing to compile.

LISA Analysis Tools–wide: export `PIP_CONSTRAINT=<repo>/LISAanalysistools/constraints/sprint.txt`
before every `pip install` (nanobind/pybind11 pin). Single-registrant grep gate:
`LISAanalysistools/tools/check_single_registrant.sh`.

## 7. Per-repo maps (pointers)

Verified-against-source internal maps (this doc is the cross-repo architecture map):

- ✅ [`LISAanalysistools/docs/codebase-map.md`](codebase-map.md)
- ✅ [`GBGPU/docs/codebase-map.md`](../../GBGPU/docs/codebase-map.md)
- ✅ [`BBHx/docs/codebase-map.md`](../../BBHx/docs/codebase-map.md)
- ✅ [`GPUBackendTools/docs/codebase-map.md`](../../GPUBackendTools/docs/codebase-map.md)
- ✅ [`Eryn/docs/codebase-map.md`](../../Eryn/docs/codebase-map.md)
- `FastEMRIWaveforms` — active sibling repo (see §2); no map written yet.
- `lisa-on-gpu` — husk; its `CLAUDE.md` deprecation notice suffices, no map needed.

Each repo also has a root `CLAUDE.md` carrying the LISA Analysis Tools–wide
rules relevant to it. Regenerate any map when its repo's structure changes —
a stale map is worse than none.

## 8. Stale-doc warnings (code contradicts docs)

1. **Root `CLAUDE.md` says the umbrella workspace root has `constraints/sprint.txt` + `tools/…`.**
   Neither exists at the umbrella workspace root. They live at
   `LISAanalysistools/constraints/sprint.txt` and
   `LISAanalysistools/tools/check_single_registrant.sh` (matches the LAT
   constraints-migration history note). Update the `PIP_CONSTRAINT` snippet and
   the tooling-pointer bullet.
2. **Root `CLAUDE.md` Phase-3L sub-phase table marks 3L.7 / 3L.8 "pending".**
   They shipped: `gbgpu/cutils/gb_tdi_on_the_fly.{cu,hh}` and
   `bbhx/cutils/sobbh_tdi_on_the_fly.{cu,hh}` exist. (The prose paragraphs below
   the table do acknowledge this — the table header is the stale part.)
3. **Convention-rule code pointers reference now-deleted files.** The Host→device and
   CPU/GPU-aliasing rules cite
   `lisa-on-gpu/src/fastlisaresponse/cutils/LISAResponse.cu:419-433` and
   `.../binding_tof.hpp` / `TDIonTheFly.hh` as canonical. Those files no longer
   exist in `lisa-on-gpu` (it's a husk with 0 C++ files). The canonical copies
   now live in `LISAanalysistools/src/lisatools/cutils/LISAResponse.cu` and the
   GBGPU/BBHx `cutils/` headers. Re-point these references.
4. **Root `CLAUDE.md` architecture-table phrase "lisa-on-gpu is being
   deprecated into LAT".** Already fully done — `lisa-on-gpu` is a finished husk,
   not mid-migration. Tense is stale.
