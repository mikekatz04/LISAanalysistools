# LISAanalysistools (`lisatools`) — Codebase Map

_Last mapped: 6e7739e · 2026-07-10 · regenerate when structure changes_
_(targeted additions since: the eigen / information-matrix routes,
`GB_CHOL_CACHE`, the PSD_BATCH method rule, the F-stat refit memory
fixes and the PE refit reference walker, 2026-10-06; stale facts corrected
in the 2026-10-22 docs audit: variants, ladder defaults, the search refit
walker, the deleted `.pyx`, file sizes)_

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
| `domains.py` | Domain settings + signal containers: `DomainSettingsBase`/`DomainBase`, `TD/FD/STFT/WDM` `Settings`+`Signal` pairs, `DomainBaseArray`. ~6,000 lines; the domain hub. |
| `domaincomputation.py` | `BaseDomainComputationGroup` + `STFT/FD/WDMComputationGroup` — Python front to C++ batched `<d\|h>`/`<h\|h>`. |
| `sensitivity.py` | `Sensitivity`/`SensitivityMatrix` hierarchy, `get_sensitivity`, TDI noise channels, `CompositeSensitivityMatrix`/`Backend`, `XYZSensitivityBackend`; mojito NOISE-brick model (`MojitoNoiseEstimates` tabulated component, `MojitoNoiseSensitivityMatrix`, `estimate_noise_params_from_file`). ~7,900 lines. |
| `detector.py` | `Orbits` (+ `EqualArmlength/ESA/L1/DefaultOrbits`), `LISAModel`. Python frontend over backend `OrbitsWrap`. |
| `mojito_detector.py` | Mojito-dataset orbit/detector interop. |
| `diagnostic.py` | `inner_product`, likelihood terms, information matrix `info_matrix`/`covariance`, SNR, Cutler–Vallisneri bias. |
| `stochastic.py` | Stochastic foregrounds/SGWB (`FittedHyperbolicTangentGalacticForeground`, `PowerLawSGWB`, …). |
| `chunked_het.py` | `WDMComputationsBase` — source-agnostic chunked-heterodyne WDM likelihood base (GB/SOBBH subclass it downstream). |
| `wdm_lookup_eval.py` | `WDMLookupEvaluator` — vectorized (numpy/cupy) evaluation of an `n_ref` WDM lookup table with the quarter-turn rule (linear / Keys cubic). |
| `sources/sobbh/wdm_direct.py` | SOBBH direct-to-WDM: batched 3.5PN, batched TDI-on-the-fly, tracer, sparse lookup template, sparse inner products / fill, and `SOBBHLookupComputations` (the `SOBBH_LIKELIHOOD=lookup` comp for `SOBBHChunkedLikeMove`).; `SOBBHLookupRouter` (per-device replicas; the stock fast comp since `SOBBH_LIKELIHOOD=lookup` became the default, 2026-10-03) |
| `cutils/sobbh_lookup_kernel.{hpp,cu}` | The fused SOBBH direct-to-WDM lookup (`sobbh_lookup`, bound in `binding_detector.cxx`, `backend.sobbh_lookup`): response splines -> per-row `<d|h>`, `<h|h>` (mode 0) or the fill (mode 1) in one launch, on the shared `wdm_lookup_kernels.hh` helpers (B-spline table, quarter turn, spline derivatives). Python driver `SOBBHDirectWDM.kernel_inner_products` / `kernel_fill`; `SOBBH_LOOKUP_KERNEL` (auto / kernel / python). Tests `tests/test_sobbh_lookup_kernel.py`. |
| `wdm_het.py` / `signal_het.py` | WDM chunk geometry / window / layer-grouping helpers; sparse-time signal-heterodyne helpers. |
| `response/` | LISA-response Python frontends (absorbed from `fastlisaresponse`): `parallelbase`, `tdiconfig`, `directresponse` (`pyResponseTDI`, `ResponseWrapper`), `tdionfly` (`TDIonTheFly` family + GB/SOBBH/FD variants). |
| `jax/` | Pure-JAX backend (`backend.py`, `jaxbase.py`, `orbits.py`) + `response/` and `wdm/` JAX mirrors. |
| `cutils/` | All C++/CUDA sources + nanobind bindings + public headers (see §5). |
| `sources/` | Per-source waveform generators: `bbh/`, `emri/`, `gb/`, `sobbh/`, plus `waveformbase.py`, `defaultresponse.py`, `utils.py`. `bbh/gridaligned.py` (`GridAlignedPhenomTHMTDIWaveform`, the only generator with `supports_batch`) evaluates every walker on one shared absolute data lattice so `pyResponseTDI` can batch them; `batching.py` (`BatchedDomainSignalGen`) adapts it to a `signal_gen` returning one `DomainBase` with a leading source axis. Batching is opt-in via `supports_batch` (default-deny at the read site) and refusals raise `BatchNotLaunchable` (PR #81/#82, 2026-09). |
| `sampling/` | Eryn-based MCMC pieces: `prior.py`, `likelihood.py`, `gmm.py`, `stopping.py`, `fstat_proposal.py` (`FStatProposal4D` grid+inverse-CDF proposal over GB intrinsics + RJ-birth container helpers; `StackedFStatProposal4D` keeps its cell log-weights on the grid's device but its flattened sampling CDF on the HOST since 2026-10-06 -- only `rvs` searchsorts it, same draws), `f0_mchirp_prior.py` (`F0McGMMSampling` astrophysical joint (f0, Mc) GB prior from the population heatmap GMM, box-truncated + renormalized), `moves/`. |
| `globalfit/` | The global-fit pipeline (engine, run, recipe, per-branch modules, `moves/`, `priors/`, `stock/`). See §3–4. `globalfit/communication/` (`ranks.py`, `fanout.py`, `fakecomm.py`, `walkerslice.py`, `rowfanout.py`) is the multi-rank walker-block layer — rank roles, layout, device pinning, and the head↔compute fan-out; `rowfanout.py` (`RowFanout`) is the one-walker-replica row-scatter primitive (`run(op, rows, *, local_body)` splits row batches across compute ranks and concatenates in row order, `replay(op, payload, *, local_body)` broadcasts to every rank) used by the addremove/PSD scoring seams; see [`docs/global-fit-launch.md`](global-fit-launch.md) (roles/knobs) and [`docs/multirank-cluster-gates.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/multirank-cluster-gates.md) (cluster validation runbook). `moves/mbhbatchedmove.py` (`MBHBatchedLikeMove`): MBH add/remove move scored through one batched grid-aligned response launch per chunk on a per-leaf 90 d / 10 d window with a segment WDM transform, per-walker residual+PSD; the DEFAULT MBH scoring path since 2026-09-30 (`MBH_LIKELIHOOD=auto` -> batched on WDM legacy-response runs, `full` with one INFO reason line for `USE_TDIONFLY=1` / a non-WDM domain / a conflicting `MBH_WAVEFORM_DURATION`; `MBH_BATCH_MAX_SIZE` 8; a data span shorter than the window is clamped by `mbh_window_layers`, not refused; `MBH_WINDOW_DECIMATE` = the window's lattice decimation factor (default 1) generates / responds / WDM-transforms the window at decimation*dt on Nf/decimation layers -- same pixels -- with the epoch snapped onto that lattice for both generators, `sources/batching.py` `MBHWindowedWDMSignalGen(decimate=...)`). `moves/emridirectmove.py` (`EMRIDirectLikeMove`): EMRI add/remove move scored through the direct-to-WDM template (`sources/emri/direct_signal_gen.py` `EMRIDirectWDMSignalGen` around `sources/emri/wdm_direct.py` `EMRIDirectWDM`), chunks of `EMRI_BATCH_MAX_SIZE` rows, per-walker residual+PSD; the engine's EMRI templates (fills, residual rebuilds) are the direct template too, the cross-check stays on the production wrap; the default since 10-03 (`EMRI_LIKELIHOOD=auto` -> direct on WDM / XYZ runs, `full` with one INFO reason line otherwise), see [`docs/emri-direct-wdm.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/emri-direct-wdm.md); its n_ref table is found or built once by `wdm_lookup_store.py` (`ensure_lookup_table`: pointer or run-folder canonical name, lock + atomic rename). |
| `globalfit/moves/eigen_refresh.py`, `eigen_table_persist.py` + the information-matrix routes | Eigen inner-move tables `(axes, sigmas)` from an information matrix (`_tables_from_info_batch`: whiten, eigen axes, prior cap) and their run sidecar. Which matrix: `addremovemove.py` `ResidualAddOneRemoveOneMove._build_eigen_table` -- default `"ll"` (likelihood second differences, `{BRANCH}_EIGEN_SCOPE`), opt-in `{BRANCH}_EIGEN_INFO=gram` (`_build_eigen_table_gram` / `_gram_info`: `<dh_a\|dh_b>` of the move's own templates through per-move `_gram_templates` hooks in `sobbhspecialmove.py`, `emridirectmove.py`, `mbhbatchedmove.py`; step tuning `{BRANCH}_EIGEN_GRAM_TARGET`); `psdmove.py` -- opt-in `{PSD,GALFOR}_EIGEN_INFO=fisher` (`_noise_fisher`, the expected Fisher of `_batched_covariance`). Both opt-ins fall back to `"ll"` on any error. Full map, numbers, knobs and status: [`docs/eigen-info-routes.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/eigen-info-routes.md). |
| `globalfit/moves/gb_chol_cache.py` | `GB_CHOL_CACHE` (default off; on in the 6mo / 9mo / 1yr v9 launchers and the flagship template, the 9mo / 12mo runs included; the 3mo v9 launchers leave it off): host-resident GB in-model proposal factors (+ observable-eigen `Gamma_z`), one cache per rank and branch shared by every GB move, entries keyed by walker + coordinates and moved with their source; all alive sources rebuilt every `GB_CHOL_CACHE_EVERY` proposes through the slot-free lookup Gram (GBGPU `SIGHET_INFOMAT_ENGINE=lookup`, which needs `SIGHET_REF_BUILD=lookup`); any error disables it and unsets the engine switch. Hooks in `gbspecialstretch.py` (`_chol_cache`, `_ensure_proposal_tables`, `_proposal_cholesky`). Tests `tests/test_gb_chol_cache.py`. |
| `globalfit/moves/psdmove.py` (PSD_BATCH) | The walker-batched noise scorer (`PSD_BATCH`, default 1) builds one covariance stack per shard (`_batched_covariance`, shared with the noise Fisher). Since 290c1902 (2026-10-06) its galfor / SGWB magnitudes follow the backend's `WDM_PSD_METHOD` (`_batched_stochastic_mag`: `fold` = the batched exact fold, anything else -- production `layer_calibrated` -- per row through the per-walker components' `_wdm_stationary_psd_column`), so the noise moves score the same model as every other move. Parity `tests/test_psd_move_batched.py` (rtol 1e-12, fold and layer_calibrated). |
| `utils/` | `parallelbase.py` (`LISAToolsParallelModule`), `constants.py`, `utility.py` (`get_array_module`, `AET`, `asnumpy`), `typing.py`, `exceptions.py` (`LISAToolsException`, `WaveformDomainError`, `BatchNotLaunchable`), `stagetimer.py` (device-synced stage timers for the MBH TDI-on-the-fly path, inert unless `MBHTDIONFLY_TIMING=1`). |
| `orbit_files/` | Packaged orbit data. |
| `galfor_modulation/` | Packaged galactic-foreground modulation table `modulation_unequal.dat` (unequal-arm TDI-2, mojito mission clock, 2.0 yr from 97729939.83 s). The `all_sources` default on mojito data (`galfor_modulation_path="auto"`, anchored at the data start); `GalForTimeModulation("modulation_unequal.dat")` resolves to it. `scripts/noise/modulation_unequal.dat` is a symlink kept for old launchers. |

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
`GalForSettings`) and a `Recipe` of `Stage`/`Move` blocks (one concept per
level, each with a `setup(ctx)` hook run at materialization; a plain
`fn(model, state)` function wraps as a `FunctionMove`, and `fit.add_branch`
takes plain branch info — no Settings class needed for simple modules).
`.build()` → `GlobalFitSetup` (formerly `CurrentInfoGlobalFit`, still an alias);
`GlobalFit(curr, comm).run_global_fit()`, or single-process
`for model, state in fit.sample(...)` (the generator run mode).

```
DomainSettingsBase ──paired──> DomainBase (TD/FD/STFT/WDM Signal)
        │                            │
        └── force_backend            └──> AnalysisContainer(data, sens_mat, signal_gen)
                                              │  uses diagnostic.inner_product
             SensitivityMatrix ──────────────┘
             (Composite / XYZBackend)

signal_gen  ← response.* (pyResponseTDI / *TDIonTheFly) ← sources.* waveforms

Global fit:  *Settings (branch blocks) + Recipe (Stages of Moves)
             └─ StockGlobalFit.build() ─> GlobalFitSetup
                └─ GlobalFit.run_global_fit()  [eryn EnsembleSampler + moves/]
                   (or fit.sample() — the single-process generator run mode)
```

---

## 4. Public API / entry points

**Top-level imports** (`import lisatools`):
`AnalysisContainer`, `DataResidualArray`, `SensitivityMatrix`,
`get_sensitivity`, `get_backend`/`has_backend`/`get_first_backend`,
`get_include`/`get_cmake_module_path`.

**Typical analysis workflow.** Build orbits (`L1Orbits` from a Mojito L1 file,
the unequal-arm model the global fits use; `EqualArmlengthOrbits` for quick
tests) → make a
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
fit.add_move("rj_fstat_mcmc", branch="gb", stage="gb_pe")
fit.build()                                 # heavy: load + pour data
fit.run()                                   # build -> GlobalFit -> run_global_fit
```
The simple entrances (`erebor.blank` + `fit.add_branch(ndim=..., priors=...,
moves=[my_fn])` + `fit.sample()` generator) are documented in
[`docs/stock-stages-and-moves.md`](stock-stages-and-moves.md).
Registered variants (`globalfit/stock/erebor/variants/`): `blank`, `gb_no_fg`,
`vgb`, `all_sources` (THE production variant of the 9mo / 12mo runs),
`full_year_combined`, `noise_only`, `noise_sgwb`, `noise_mojito`,
`noise_galfor_mojito`, and the `*_lite` presets (`variants/lite.py`).
`scripts/run_global.py --stock <name>` is the generic CLI entry
(single-process or under `mpiexec` — rank layout and GPU knobs in
[`docs/global-fit-launch.md`](global-fit-launch.md); the legacy
`run_mpi_global_fit.py`/`pipeline.py` stack was removed 2026-07). The
production runs use `scripts/fstat_proposal/run_combined_staged.py` instead
([`docs/global-fit-navigation.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/global-fit-navigation.md)). The legacy
`global_fit_input/*.py` and `mojito_input/*.py` settings files are
compatibility stubs — **do not grow them back**.

**Examples**: none in this repo; `examples/README.md` points to the LATW
tutorials repository (branch policy in [`conventions.md`](conventions.md)).

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
- `lat_spline_tdi_waveform.{cu,hh}` + `binding_lat_spline_tdi.hpp` — `FD/TDSplineTDIWaveform`, `TDDenseTDIWaveform` (EMRI dense-phase response), and `wdm_lookup_sum` (the fused sparse-response -> WDM n_ref lookup sum, EMRI direct-to-WDM); `wdm_lookup_kernels.hh` — its header-only device helpers (mirror-mode cubic B-spline table eval, spline value/derivatives, DOPR853 phase derivatives, quarter-turn rule) for other sources' kernels.
- `lat_chunked_het_kernels.hh`, `lat_wdm_fft.hh` — templated chunked-het kernels + WDM FFT helpers (shared with GB/SOBBH downstream).
- `galactic_response.{cu,hpp}` — galactic-grid transfer functions.
- Public headers for downstream: `orbits_view.hpp` (`OrbitsView` POD), `lisatools_header_abi.hpp` (ABI version + `LISATOOLS_IS_WRAPPER_OWNER`), `Detector.hpp`, `PSD.hpp`, `LISAResponse.hh`, `binding_flr.hpp`.
- `fd_domain.hh`/`wdm_settings.hh`/`wdm_domain.hh` are **deprecated include shims** → `domains.hpp`. The legacy Cython `pycppdetector.pyx` was deleted (72c38485; `pyproject.toml` `sdist.include` still names it); nanobind is the live path.

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
- **Global-fit storage is cold-chain-only in the main backend**
  (`gf_format_version=2`): the engine runs `ntemps=1`; each branch owns its
  full tempered ensemble (chain/inds/ll/ladder/counters + per-leaf
  `d_h`/`h_h`) in its `ModuleSubState` / `ModuleSubBackend`
  (`sub_backend/<branch>` in the same HDF file). Per-branch ladder knobs
  `<BRANCH>_NTEMPS` (code defaults gb 24 / vgb 12 / mbh 24 / emri 24 / sobbh 4 /
  psd and galfor 12; the 9mo / 12mo launchers set vgb 8, mbh 8, emri 12,
  sobbh 8); the legacy
  `NTEMPS` env raises on stock variants (`erebor.blank` keeps it — simple-API
  branches temper on the engine ladder). "The sub-state IS the schema": a
  sub-state's setup methods + name lists define its datasets; no Spec layer.
  Cold-row agreement is checked at every stage entry (`GF_SUBSTATE_CHECK=0`
  disables). Pre-rework files cannot be resumed (clean break).
- **`gbspecialstretch.py`** — GB/VGB special moves. One-walker replica mode
  (`layout.replica_mode`): static per-rank band ranges (`replica_band_ranges`,
  `_owned_rows_mask`), a per-unit cold-chain delta ledger (`_ledger_*`), a
  merge by physical source on the head (`merge_owned_sources`) and a 4th
  fan-out command `gb_sync` that rebuilds every replica's residual from the
  merged branch. `GB_OPS` carries three more — `gb_fstat_ref_row`,
  `gb_fstat_stage_b`, `gb_fstat_release` — the NON-session F-stat epoch fit
  (the head issues them from `setup()`, before any session exists): replicate
  the global reference walker's residual + inverse-PSD row to every rank,
  split each Mc group's stage-B sweep by contiguous box range, then drop the
  row, the scorer and its comp-side reference blocks everywhere. Runbook:
  `docs/multirank-cluster-gates.md` Step 5. F-stat refit memory (6mo jobs
  730 / 738 OOMs): before a refit builds epoch k, the move drops its
  installed proposal and centre table, `_evict_fstat_epochs` removes every
  other epoch of its fit dir from the process-wide `_FSTAT_GRID_REGISTRY` /
  `_FSTAT_CTR_TABLE_REGISTRY` (keyed by epoch dir, never evicted before), and
  the memory pool is freed (`tests/test_fstat_refit_releases_grid.py`).
  F-stat epoch reference walker (`_fstat_fit_ref_rule` /
  `_fstat_fit_ref_from`): search refits ROTATE through the cold walkers with
  a finite lnL by epoch number (`GB_FSTAT_SEARCH_REF=rotate`, the default
  since 2026-10-16; `min` restores the min-lnL rule);
  PE refits default to a uniformly RANDOM cold walker per epoch
  (`GB_FSTAT_PE_REF=random`, 2026-10-06), seeded by
  (`GB_FSTAT_PE_REF_SEED`, default 0; epoch) so restarts and every rank
  re-derive the same walker; `GB_FSTAT_PE_REF=min` restores the min-lnL
  rule in PE. The per-propose distance-birth centre keeps the min.
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
| Proposal information matrices / eigen tables (which route, which knob) | [`docs/eigen-info-routes.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/eigen-info-routes.md); `globalfit/moves/eigen_refresh.py`, `addremovemove.py`, `psdmove.py`, `gb_chol_cache.py` |
| Downstream C++ header consumption (GBGPU/BBHx) | `cutils/orbits_view.hpp`, `cutils/lisatools_header_abi.hpp`, `get_include()` in `__init__.py` |
| JAX response / WDM implementations | `jax/response/`, `jax/wdm/` |
| Tests for a subsystem | `tests/` (`test_detector`, `test_sensitivity`, `test_wdm_domain_cpp`, `test_stock_globalfit`, `test_gb_likelihood_engine`, …) |
