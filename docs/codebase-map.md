# LISAanalysistools (`lisatools`) — Codebase Map

_Last mapped: 6e7739e · 2026-07-10 · regenerate when structure changes_

Central LISA-physics library for LISA Analysis Tools. All structural claims below were
verified against current source under `src/lisatools/` (imports, class defs,
CMake, `__init__` wiring), not just docs. Where a doc lagged the code it is
flagged in the closing notes.

---

## 1. What this is

`lisaanalysistools` (imported as `lisatools`) is a hybrid Python / C++ / CUDA
package for LISA data analysis — sensitivity/PSD modeling, the LISA TDI
response, time/frequency/wavelet (WDM) domain transforms, per-source waveform
generators, likelihood inner products, and the full **LISA Global Fit** MCMC
pipeline. Built with `scikit-build-core` + CMake, Python 3.12+. It is the
project's central LISA-physics library: `lisa-on-gpu` was deprecated into it,
and downstream waveform packages (GBGPU, BBHx, FEW) consume its C++ headers.
GPU/CPU/JAX dispatch is delegated to the separate `gpubackendtools` (GBT)
package.

---

## 2. Layout (`src/lisatools/`)

| Path | Role |
|---|---|
| `__init__.py` | Registers CPU/CUDA/JAX backends into `Globals().backends_manager`; `get_backend/has_backend/get_first_backend` (prefix `lisatools_`); `get_include()` / `get_cmake_module_path()` for downstream C++. |
| `analysiscontainer.py` | `AnalysisContainer` (central likelihood object) + `AnalysisContainerArray` + `BandView`. |
| `datacontainer.py` | `DataResidualArray` — **deprecated** thin shim over `domains.DomainBase`. |
| `domains.py` | Domain settings + signal containers: `DomainSettingsBase`/`DomainBase`, `TD/FD/STFT/WDM` `Settings`+`Signal` pairs, `DomainBaseArray`. ~2900 lines; the domain hub. |
| `domaincomputation.py` | `BaseDomainComputationGroup` + `STFT/FD/WDMComputationGroup` — Python front to C++ batched `<d\|h>`/`<h\|h>`. |
| `sensitivity.py` | `Sensitivity`/`SensitivityMatrix` hierarchy, `get_sensitivity`, TDI noise channels, `CompositeSensitivityMatrix`/`Backend`, `XYZSensitivityBackend`. ~4000 lines. |
| `detector.py` | `Orbits` (+ `EqualArmlength/ESA/L1/DefaultOrbits`), `LISAModel`. Python frontend over backend `OrbitsWrap`. |
| `mojito_detector.py` | Mojito-dataset orbit/detector interop. |
| `diagnostic.py` | `inner_product`, likelihood terms, Fisher `info_matrix`/`covariance`, SNR, Cutler–Vallisneri bias. |
| `stochastic.py` | Stochastic foregrounds/SGWB (`FittedHyperbolicTangentGalacticForeground`, `PowerLawSGWB`, …). |
| `chunked_het.py` | `WDMComputationsBase` — source-agnostic chunked-heterodyne WDM likelihood base (GB/SOBBH subclass it downstream). |
| `wdm_het.py` / `signal_het.py` | WDM chunk geometry / window / layer-grouping helpers; sparse-time signal-heterodyne helpers. |
| `response/` | LISA-response Python frontends (absorbed from `fastlisaresponse`): `parallelbase`, `tdiconfig`, `directresponse` (`pyResponseTDI`, `ResponseWrapper`), `tdionfly` (`TDIonTheFly` family + GB/SOBBH/FD variants). |
| `jax/` | Pure-JAX backend (`backend.py`, `jaxbase.py`, `orbits.py`) + `response/` and `wdm/` JAX mirrors. |
| `cutils/` | All C++/CUDA sources + nanobind bindings + public headers (see §5). |
| `sources/` | Per-source waveform generators: `bbh/`, `emri/`, `gb/`, `sobbh/`, plus `waveformbase.py`, `defaultresponse.py`, `utils.py`. |
| `sampling/` | Eryn-based MCMC pieces: `prior.py`, `likelihood.py`, `gmm.py`, `stopping.py`, `moves/`. |
| `globalfit/` | The global-fit pipeline (engine, run, recipe, per-branch modules, `moves/`, `priors/`, `stock/`). See §3–4. |
| `utils/` | `parallelbase.py` (`LISAToolsParallelModule`), `constants.py`, `utility.py` (`get_array_module`, `AET`, `asnumpy`), `typing.py`. |
| `orbit_files/` | Packaged orbit data. |

---

## 3. Core abstractions

**Domains (the data-model spine).** Every array lives in a *domain*. A
`DomainSettingsBase` subclass (`TDSettings`, `FDSettings`, `STFTSettings`,
`WDMSettings`) describes the grid; its paired `DomainBase` subclass
(`TDSignal`, `FDSignal`, `STFTSignal`, `WDMSignal`) wraps the actual array +
settings and knows how to `.transform()` between domains and add/subtract
templates. **Domains are never passed by string** — dispatch is `isinstance`
on the settings class.

**AnalysisContainer.** Combines `data (DomainBase)` + `SensitivityMatrix` +
`signal_gen` (a `Callable` or `{name: Callable}` dict) into the object that
computes likelihoods (`.likelihood`, `.calculate_signal_likelihood`,
`template_inner_product`). Legacy `DataResidualArray` inputs are transparently
unwrapped to their `DomainBase`.

**Sensitivity.** `Sensitivity` (per-channel PSD models: `X/A/E/T` gen-1/2) →
`SensitivityMatrixBase` → `SensitivityMatrix` (stock: `AET1/2`, `XYZ1/2`,
`AE1/2`, `LISASens`). `CompositeSensitivityMatrix` sums `InstrumentNoise` +
`GalacticForeground` + `SGWB` components; `XYZSensitivityBackend` is the C++
galactic-grid path.

**Response / TDI.** `pyResponseTDI` + `ResponseWrapper` (direct time-domain
projection+TDI); `TDIonTheFly` family (`TDTDIonTheFly`, `FDTDIonTheFly`,
`GBTDIonTheFly`, `SOBBHTDIonTheFly`, `GBFDTDIonTheFly`) generate TDI on the fly
via C++ `*Wrap` classes. `TDIConfig` holds the TDI combination.

**WDM likelihood paths.** `WDMComputationsBase` (chunked-heterodyne) and the
`*ComputationGroup` classes route batched inner products through backend C++
kernels; source-specific subclasses (`GBWDMComputations`,
`SOBBHWDMComputations`) live in GBGPU/BBHx.

**Stock global fits.** `StockGlobalFit` (in `globalfit/stock/base.py`) is a
picklable, unbuilt run config; the `erebor` family (`EreborFit`,
`EreborGeneralSettings`) plus per-branch `*Settings`/`*Setup` dataclasses
(`GBSettings`, `MBHSettings`, `EMRISettings`, `SOBBHSettings`, `PSDSettings`,
`GalForSettings`) and a declarative `RecipeSpec` of `StageSpec`/`MoveSpec`.
`.build()` → `CurrentInfoGlobalFit`; `GlobalFit(curr, comm).run_global_fit()`.

```
DomainSettingsBase ──paired──> DomainBase (TD/FD/STFT/WDM Signal)
        │                            │
        └── force_backend            └──> AnalysisContainer(data, sens_mat, signal_gen)
                                              │  uses diagnostic.inner_product
             SensitivityMatrix ──────────────┘
             (Composite / XYZBackend)

signal_gen  ← response.* (pyResponseTDI / *TDIonTheFly) ← sources.* waveforms

Global fit:  *Settings (branch blocks) + RecipeSpec (stages/moves)
             └─ StockGlobalFit.build() ─> CurrentInfoGlobalFit
                └─ GlobalFit.run_global_fit()  [eryn EnsembleSampler + moves/]
```

---

## 4. Public API / entry points

**Top-level imports** (`import lisatools`):
`AnalysisContainer`, `DataResidualArray`, `SensitivityMatrix`,
`get_sensitivity`, `get_backend`/`has_backend`/`get_first_backend`,
`get_include`/`get_cmake_module_path`.

**Typical analysis workflow.** Build orbits (`EqualArmlengthOrbits`) → make a
waveform/response generator (`sources.*` or `response.*TDIonTheFly`) →
`get_sensitivity` / a `SensitivityMatrix` → wrap in `AnalysisContainer(data,
sens_mat, signal_gen)` → call `.likelihood(params)` /
`.template_inner_product(...)`. Domain conversions via `DomainBase.transform`.

**Global fit (stock, current API).**
```python
from lisatools.globalfit.stock import erebor
erebor.get_stock_options()                 # [(name, description), ...]
fit = erebor.gb_no_fg(nwalkers=4)          # or erebor.get_stock("gb_no_fg", ...)
fit.gb.min_freq = 9.8e-3                    # plain attribute access on blocks
fit.recipe.add_move(MoveSpec("rj_fstat_mcmc", branch="gb"), stage="gb_pe")
fit.build()                                 # heavy: load + pour data
fit.run()                                   # build -> GlobalFit -> run_global_fit
```
Registered variants: `gb_no_fg`, `all_sources`, `full_year_combined`
(`globalfit/stock/erebor/variants/`). `scripts/run_global.py --stock <name>`
is the CLI entry (single-process or under `mpiexec` — rank layout and GPU
knobs in [`docs/global-fit-launch.md`](global-fit-launch.md); the legacy
`run_mpi_global_fit.py`/`pipeline.py` stack was removed 2026-07). The legacy
`global_fit_input/*.py` and `mojito_input/*.py` settings files are
compatibility stubs — **do not grow them back**.

**Examples**: `examples/*.ipynb` (analysis_container, sensitivity, lisa_response,
wdm_transform, time_frequency_domains, stock_global_fit tutorials).

---

## 5. Backend structure

**Dispatch model.** LAT delegates all CPU/CUDA/JAX selection to
`gpubackendtools`. Backend classes are defined in `cutils/__init__.py`:
`LISAToolsCpuBackend`, `LISAToolsCuda11x/12x/13xBackend` (each a `CpuBackend`/
`CudaNNxBackend` + `LISAToolsBackend` mixin), plus `jax/backend.py`’s
`LISAToolsJaxBackend`. `__init__.py` registers them; `get_backend("cpu"|
"cuda12x"|"cuda"|"gpu"|"jax")` returns a `Backend` exposing `.xp` (numpy/cupy/
jax.numpy) and the native symbols.

**How a class picks a backend.** Subclass `LISAToolsParallelModule`
(`utils/parallelbase.py`, `_BACKEND_PREFIX = "lisatools"`) and pass
`force_backend="cpu"|"cuda12x"|"jax"|...` **at construction**. Use
`self.backend`, `self.backend.xp`, `self.backend.name` internally. Bare
strings are rewritten to `("lisatools", name)`. Some classes override the
prefix (e.g. `GBTDIonTheFly._BACKEND_PREFIX = "gbgpu"`,
`WDMComputationsBase` subclasses).

**Method table.** `LISAToolsBackendMethods` (dataclass in `cutils/__init__.py`)
enumerates every native symbol a backend module must supply (`OrbitsWrap`,
`Orbits`, `LISAResponseWrap`, `TDIConfigWrap`, `WDMSettingsWrap`,
`WDMDomainWrap`, `FDDomainWrap`, `FDSpline/TDSplineTDIWaveformWrap`,
`XYZSensitivityMatrixWrap`, `GalacticGridWrap/Setup`, `STFTDomainWrap`,
`FDDomainForStftWrap`, `STFTFresnelWrap`, `CubicSplineWrap` [sourced from GBT],
`psd_likelihood`, `compute_logpdf`, `TDITypeDict`). Add a symbol → add a field
here + populate each `*_module_loader`.

**Native sources (`cutils/`).** The nanobind module is `pycppdetector`, built
from `binding_detector.cxx` + `binding_flr.cxx`; kernels compile into a static
archive (`Detector.cu PSD.cu LISAResponse.cu lat_tdi_on_the_fly.cu
lat_spline_tdi_waveform.cu galactic_response.cu domains.cu`). Each `.cu` is
**copied to `.cxx`** and compiled by the host C++ compiler for the CPU wheel;
the same `.cu` is compiled by `nvcc` for the GPU wheel — code must be valid as
both (`#ifdef __CUDACC__`).
- `Detector.{cu,hpp}` — orbit/detector geometry (`Orbits`).
- `PSD.{cu,hpp}` — PSD kernels.
- `LISAResponse.{cu,hh}` + `binding_flr.{cxx,hpp}` — arm projection / TDI (`response_part()`, `LISAResponseWrap`, `TDIConfigWrap`).
- `domains.{cu,hpp}` — STFT/FD/WDM domain descriptors + STFT machinery; TDI flavor ints `TDI_XYZ/AET/AE`.
- `lat_tdi_on_the_fly.{cu,hh}` — `LISATDIonTheFly` base + `OrbitsSplineCache`.
- `lat_spline_tdi_waveform.{cu,hh}` + `binding_lat_spline_tdi.hpp` — `FD/TDSplineTDIWaveform`.
- `lat_chunked_het_kernels.hh`, `lat_wdm_fft.hh` — templated chunked-het kernels + WDM FFT helpers (shared with GB/SOBBH downstream).
- `galactic_response.{cu,hpp}` — galactic-grid transfer functions.
- Public headers for downstream: `orbits_view.hpp` (`OrbitsView` POD), `lisatools_header_abi.hpp` (ABI version + `LISATOOLS_IS_WRAPPER_OWNER`), `Detector.hpp`, `PSD.hpp`, `LISAResponse.hh`, `binding_flr.hpp`.
- `fd_domain.hh`/`wdm_settings.hh`/`wdm_domain.hh` are **deprecated include shims** → `domains.hpp`. `pycppdetector.pyx` is legacy Cython; nanobind is the live path.

**JAX subpackage.** `jax/response/` (`JaxAmpPhaseSource`, projection, tdi_config,
amp_phase_extract) + `jax/wdm/` (wdm_settings/domain, wavelet_lookup,
fast_inner). Pure-Python; no compiled wheel. Designed with JAX-native idioms
and validated against C++ at the inner-product level (see §7).

---

## 6. Cross-repo dependencies

**LAT imports from:**
- `gpubackendtools` (GBT) — backend dispatch (`Globals`, `Backend`,
  `ParallelModuleBase`, `BackendMethods`, `CpuBackend`/`CudaNNxBackend`),
  `interpolate.CubicSplineInterpolant`, `wrapper`; also `get_include()` for
  GBT’s `gbt_global.h`/`cuda_complex.hpp`/`InterpolateDevice.hh`. CUDA backends
  import `gbt_backend_<flavor>.interp` (GBT is the **single registrant** for
  `CubicSplineWrap`).
- `lisaconstants==2.0.2`, `eryn` (MCMC), `cudakima` (`AkimaInterpolant1D`),
  `multispline`, `mojito` + `mojito-processor` (data), `numba`, `scipy`,
  `h5py`.
- Waveform packages used *by* LAT’s source generators: `gbgpu.gbgpu.GBGPU`
  (`sources/gb`), `few.waveform` (`sources/emri`), `bbhx.waveformbuild`
  (`globalfit/stock/erebor/wrappers.py`). These are `[testing]`/runtime source
  deps, not build deps.

**Depends on LAT (downstream):** GBGPU, BBHx, FastEMRIWaveforms consume LAT’s
C++ headers. Verified: `GBGPU/src/gbgpu/cutils/CMakeLists.txt` and
`BBHx/src/bbhx/cutils/CMakeLists.txt` both shell out to
`python -c "import lisatools; print(...cutils)"` for `LISATOOLS_CUTILS` on
their include path (pulling `orbits_view.hpp`, `lisatools_header_abi.hpp`,
`Detector.hpp`, `ReturnPointerBase`, etc.). They **do not** re-register the
shared wrapper classes — LAT is the sole registrant (single-registrant rule).
GB/SOBBH `*ComputationGroupWrap` shells and the source-specific chunked-het/
signal-het subclasses live in those repos and dispatch back through LAT’s
templated kernels.

---

## 7. Non-obvious invariants / gotchas (pointers; full rules in `CLAUDE.md`)

- **No backend strings as method kwargs.** Backend is chosen at construction
  (`force_backend=`), never `backend="jax"` per-call. (see `CLAUDE.md`)
- **No array module as an instance attribute.** Never `self.xp = cp`; expose
  `xp` as a property off `self.backend`/a flag. Objects deepcopy/pickle
  (settings tree, MPI). (Deepcopy/pickle rule)
- **Host→device wrapper upload.** CUDA kernels must `cudaMalloc`+`cudaMemcpy`
  host-`new`’d wrapper structs (`OrbitsWrap`, `WDMDomainWrap`, …) before
  launch; IMA at `0x55555...`/TB-range delta ⇒ missing upload, not an index
  bug. (Host→device rule)
- **CPU/GPU class-name aliasing.** Every class in both wheels needs a per-
  backend `#define Foo → FooGPU/FooCPU` block (wrappers *and* underlying
  classes; both `#if/#else` branches symmetric). Prevents `typeid` collisions
  across side-loaded plugin wheels. (CPU/GPU aliasing rule)
- **Single-registrant rule.** Only LAT registers the shared wrapper classes;
  enforced by `LISATOOLS_IS_WRAPPER_OWNER` static_asserts +
  `tools/check_single_registrant.sh`. `CubicSplineWrap` is GBT-owned.
- **`OrbitsView` POD layout** asserted at every build in `binding_detector.cxx`
  (`sizeof` + 15 offsetofs).
- **nanobind pinned exactly** (`==2.12.0`) for cross-wheel type sharing.
- **`.cu → .cxx` copy step**: editing `Detector.cu` rebuilds CPU *and* GPU.
- **No new global-fit settings files**: new run = `StockGlobalFit` subclass;
  nothing heavy in `__init__` (data load in `.build()`); pre-build fit must
  pickle/deepcopy.
- **Backend implementation hierarchy**: GPU C++ leads → CPU C++ mirrors via
  `#ifdef` → JAX diverges internally but must match C++ inner products
  (reldiff ≲ 1e-12). Narrowband WDM validation via `mm5`/`mm2`.
- **`xp` pattern everywhere**: arrays may be cupy; resolve with
  `get_array_module` (`utils/utility.py`), don’t assume numpy.

---

## 8. Where to look for X

| I want to change/understand… | Start in |
|---|---|
| Likelihood / inner product of data vs template | `analysiscontainer.py` (`AnalysisContainer`), `diagnostic.py` |
| Add/rework a sensitivity or PSD model | `sensitivity.py`; foregrounds in `stochastic.py` |
| A time/freq/WDM domain transform or grid | `domains.py`; batched C++ path `domaincomputation.py` |
| Orbits / detector geometry | `detector.py` (Py) + `cutils/Detector.{cu,hpp}` |
| LISA response / TDI generation | `response/directresponse.py`, `response/tdionfly.py`, `cutils/LISAResponse.{cu,hh}` + `binding_flr.*` |
| Per-source waveforms (GB/EMRI/SOBBH/MBH) | `sources/{gb,emri,sobbh,bbh}/`; base `sources/waveformbase.py` |
| Chunked-heterodyne WDM likelihood | `chunked_het.py`, `wdm_het.py`; kernels `cutils/lat_chunked_het_kernels.hh`, `cutils/lat_wdm_fft.hh` |
| Add a native C++/CUDA symbol to a backend | `cutils/__init__.py` (`LISAToolsBackendMethods` + each `*_module_loader`), `cutils/CMakeLists.txt` |
| How a backend is chosen at runtime | `utils/parallelbase.py`, `cutils/__init__.py`, `__init__.py`, `jax/backend.py` |
| Global-fit run config / stock variants | `globalfit/stock/base.py`, `globalfit/stock/erebor/`, `.../variants/` |
| Global-fit engine / run loop / recipe | `globalfit/engine.py`, `globalfit/run.py`, `globalfit/recipe.py`, `globalfit/hdfbackend.py` |
| MCMC moves (GB/MBH/PSD special moves) | `globalfit/moves/`, `sampling/moves/`, `sampling/prior.py` |
| Downstream C++ header consumption (GBGPU/BBHx) | `cutils/orbits_view.hpp`, `cutils/lisatools_header_abi.hpp`, `get_include()` in `__init__.py` |
| JAX response / WDM implementations | `jax/response/`, `jax/wdm/` |
| Tests for a subsystem | `tests/` (`test_detector`, `test_sensitivity`, `test_wdm_domain_cpp`, `test_stock_globalfit`, `test_gb_likelihood_engine`, …) |
