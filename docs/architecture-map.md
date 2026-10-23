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
  resolve `${GBT_CUTILS}` (a Python shell-out to the installed `gpubackendtools`
  package directory + `cutils`, the directory `gpubackendtools.get_include()`
  returns; LAT's CMake does the same), include its `cmake_functions.cmake`, and
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
| **WDM wavelet transforms** | Py: `lisatools.domains` (`WDMSettings`), `lisatools.wdm_het`. C++: `lisatools/cutils/domains.hpp` (the WDM / FD descriptors; `wdm_settings.hh` / `wdm_domain.hh` are deprecated include shims onto it), `lat_wdm_fft.hh`. JAX: `lisatools.jax.wdm` |
| **TDI-on-the-fly (base)** | Py: `lisatools.response.tdionfly`. C++: `lisatools/cutils/lat_tdi_on_the_fly.{cu,hh}`, `lat_spline_tdi_waveform.{cu,hh}`, `lat_chunked_het_kernels.hh` |
| **GB waveform** | `gbgpu.gbgpu.GBGPU` (+ `GBGPUBase`); C++ `gbgpu/cutils/gb_tdi_on_the_fly.{cu,hh}`, `SharedMemoryGBGPU.*`. Stock TDI wrapper: `lisatools.sources.gb.waveform.GBAETWaveform` |
| **MBH waveform** | `bbhx` (`bbhx/cutils/PhenomHMWaveform.{cu,hh}`, `Response.*`, `WaveformBuild.*`), `bbhx.mbhtdionfly`, `bbhx.mbhphentax`. Stock wrapper: `lisatools.sources.bbh.waveform`; the production batched windowed scoring: `lisatools.sources.bbh.gridaligned` + `lisatools.sources.batching` (`MBH_LIKELIHOOD=batched`, `globalfit.moves.mbhbatchedmove`) |
| **SOBBH waveform** | `bbhx.sobbhtdionfly`, `bbhx.sobbhcomps`; C++ `bbhx/cutils/sobbh_tdi_on_the_fly.{cu,hh}`. PN math + wrapper: `lisatools.sources.sobbh.waveform`; the production direct-to-WDM lookup scorer: `lisatools.sources.sobbh.wdm_direct` + C++ `lisatools/cutils/sobbh_lookup_kernel.{hpp,cu}` (`SOBBH_LIKELIHOOD=lookup`, [`docs/sobbh-wdm-lookup.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/sobbh-wdm-lookup.md)) |
| **EMRI waveform** | `few.waveform.waveform.GenerateEMRIWaveform`; C++/Cython in `few/cutils/`. LISA-TDI wrappers: `lisatools.sources.emri.{waveform,emritdionfly,response}`; the production direct-to-WDM template: `lisatools.sources.emri.{wdm_direct,direct_signal_gen}` (`EMRI_LIKELIHOOD=direct`, [`docs/emri-direct-wdm.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/emri-direct-wdm.md)) |
| **Samplers / MCMC** | `eryn.ensemble`, `eryn.paraensemble`, `eryn.moves.*` (stretch, nuts, rj, tempering, distgen, …), `eryn.backends.*`. Global-fit orchestration: `lisatools.globalfit.*` (`engine`, `run`, `recipe`, `moves/`, `stock/`) |
| **Interpolation + compute backends** | `gpubackendtools.interpolate` (`CubicSplineInterpolant`); C++ `gpubackendtools/cutils/Interpolate.{cu,hh}` (SOLE `CubicSplineWrap` registrant). Backend machinery: `gpubackendtools.gpubackendtools` (`Backend`, `BackendsManager`, `CpuBackend`/`Cuda1{1,2,3}xBackend`), `gpubackendtools.globals` (`Globals`, `get_backend`/`has_backend`/`get_first_backend`), `gpubackendtools.parallelbase` (`ParallelModuleBase`) |
| **Proposal information matrices** (eigen tables, GB in-model factors) | LAT: `lisatools.globalfit.moves.eigen_refresh` (tables), `addremovemove` (`{BRANCH}_EIGEN_INFO` ll / gram), `psdmove` (`{PSD,GALFOR}_EIGEN_INFO` ll / fisher), `gb_chol_cache` (`GB_CHOL_CACHE`), `lisatools.info_matrix_ll` (second differences). GBGPU: `gbgpu.gbsignalhetcomputations` / `gbgpu.gblookupcomputations` `information_matrix` (`SIGHET_INFOMAT`, `SIGHET_INFOMAT_ENGINE=lookup`). Eryn: `eryn.moves.EigenAxisMove`. Route map: [`docs/eigen-info-routes.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/eigen-info-routes.md) |
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
  keyed by the registry name `<pkg>_<flavor>`, e.g. `lisatools_cuda12x`;
  `lisatools_backend_cuda12x` is that backend's plugin MODULE, the class's
  `_backend_name`), reached via `Globals().backends_manager`. Each package
  registers its flavors at import:
  `Globals().backends_manager.add_backends({"few_cpu": FEWCpuBackend, ...})`.
  Every `add_backends` call rebuilds the registry for all packages (backends
  loaded earlier are rebuilt on their next lookup).
- **Per-package facade:** `lisatools.get_backend("cpu")` →
  `gpubackendtools.get_backend("lisatools_cpu")`; likewise `gbgpu_*`, `bbhx_*`
  (FEW has no package-level facade; its `few_*` backends are reached through
  `FewBackendConsumer` or GBT's functions). `get_backend` also resolves the `"cuda"` / `"gpu"` aliases (first
  available of `cuda13x`, `cuda12x`, `cuda11x`). `has_backend(...)` adds the
  same prefix but resolves no alias: `lisatools.has_backend("cuda")` raises
  `ValueError`; probe the flavors (`"cuda13x"`, ...) instead.
- **Selection at construction, never per-method** (LISA Analysis Tools–wide rule): pass
  `force_backend="cpu"|"cuda12x"|"jax"`; `ParallelModuleBase` (GBT) exposes
  `self.backend` / `self.xp` / `self.backend_name`. Bare flavors work because each
  package base (`LISAToolsParallelModule`, `GBGPUParallelModule`,
  `BBHxParallelModule`, `FewBackendConsumer`, GBT's `GBTParallelModuleBase`) adds
  its prefix; GBT's plain `ParallelModuleBase` needs a full registry name, and a
  bare flavor in `gpubackendtools.get_first_backend` matches the first registered
  package (`gbt_*`), so `supported_backends()` must return prefixed names.
- **CPU + CUDA co-load in one interpreter:** the CPU/GPU `#define FooWrap{CPU,GPU}`
  aliasing (CLAUDE.md rule) gives each backend distinct mangled C++ symbols, so
  `lisatools.has_backend("cpu")` and `lisatools.has_backend("cuda12x")` can both
  be True at once.
- `Backend.__reduce__` pickles by registry name (deepcopy/pickle-safe); array
  module `xp` is a property, never stored.

## 6. Build order / how to build the stack

Bottom-up (each layer needs the previous compiled + installed):

1. **GPUBackendTools** — provides `get_include()` cutils dir (`${GBT_CUTILS}`),
   `Interpolate.cu`, `cuda_complex.hpp`, `gbt_global.h`, `cmake_functions.cmake`
   (the `apply_{cpu,gpu}_backend_common_options` / LAPACKE helpers the LAT, GBGPU
   and BBHx CMake include), backend registry.
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
- ✅ [`GBGPU/docs/codebase-map.md`](https://github.com/lisa-analysis-tools/GBGPU/blob/dev/docs/codebase-map.md)
- ✅ [`BBHx/docs/codebase-map.md`](https://github.com/lisa-analysis-tools/BBHx/blob/dev/docs/codebase-map.md)
- ✅ [`GPUBackendTools/docs/codebase-map.md`](https://github.com/lisa-analysis-tools/GPUBackendTools/blob/spline/docs/codebase-map.md)
- ✅ [`Eryn/docs/codebase-map.md`](https://github.com/lisa-analysis-tools/Eryn/blob/dev/docs/codebase-map.md)
- `FastEMRIWaveforms` — active sibling repo (see §2); no map written yet.
- `lisa-on-gpu` — husk; its `CLAUDE.md` deprecation notice suffices, no map needed.

Each repo also has a root `CLAUDE.md` carrying the LISA Analysis Tools–wide
rules relevant to it. Regenerate any map when its repo's structure changes —
a stale map is worse than none.

## 8. Stale-doc warnings (code contradicts docs)

_(As of 2026-10-22 all four are resolved: the umbrella root `CLAUDE.md` is now
a short stub that points into LAT, and `docs/conventions.md` cites LAT's
`cutils/LISAResponse.cu` for the host->device pattern. Kept as a record.)_

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
