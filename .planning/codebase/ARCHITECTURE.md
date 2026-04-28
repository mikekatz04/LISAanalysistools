<!-- refreshed: 2026-04-28 -->
# Architecture

**Analysis Date:** 2026-04-28

## System Overview

LISAanalysistools is a scientific Python package for simultaneous Bayesian parameter estimation of all gravitational-wave sources observed by LISA (the "Global Fit"). Its central design problem is computing the likelihood over a shared, mutable residual (data minus current model) when thousands of sources compete for the same frequency band. A custom container stack (domains → data containers → sensitivity matrices → analysis containers) abstracts the signal representation away from the likelihood math, enabling backend-agnostic CPU/GPU execution.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     Sampling Layer (MCMC)                           │
│  Eryn EnsembleSampler / GlobalFitEngine  `globalfit/engine.py`      │
│  Moves: ResidualAddOneRemoveOne, PSDMove  `globalfit/moves/`        │
└──────────────────────┬──────────────────────────────────────────────┘
                       │  log-likelihood calls
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Analysis Container Layer                           │
│  AnalysisContainer / AnalysisContainerArray  `analysiscontainer.py` │
│  Combines DataResidualArray + SensitivityMatrix + signal generators  │
└───────────┬───────────────────────────┬─────────────────────────────┘
            │                           │
            ▼                           ▼
┌─────────────────────┐   ┌─────────────────────────────────────────┐
│  Data / Residual    │   │  Sensitivity / PSD Layer                │
│  DataResidualArray  │   │  SensitivityMatrix, SensitivityMatrixBase│
│  `datacontainer.py` │   │  `sensitivity.py`                       │
└──────────┬──────────┘   └────────────────┬────────────────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Domain / Signal Layer                          │
│  DomainBase, DomainSettingsBase, TDSignal, FDSignal, STFTSignal      │
│  `domains.py`                                                        │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│              C++ / CUDA Backend (via pybind11)                      │
│  `cutils/__init__.py` → lisatools_backend_{cpu,cuda11x,cuda12x}     │
│  OrbitsWrap, SensitivityMatrixWrap, psd_likelihood, compute_logpdf  │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `DomainSettingsBase` / `DomainBase` | Encapsulate array + grid metadata; perform domain transforms (TD→FD→STFT) | `src/lisatools/domains.py` |
| `DataResidualArray` | Hold multi-channel frequency/STFT data; delegate indexing and shape ops | `src/lisatools/datacontainer.py` |
| `SensitivityMatrix` / `SensitivityMatrixBase` | Store inverse PSD; expose `invC`, `detC`, `differential_component` | `src/lisatools/sensitivity.py` |
| `AnalysisContainer` | Pair data + PSD; compute `template_likelihood`, `inner_product`, `snr` | `src/lisatools/analysiscontainer.py` |
| `AnalysisContainerArray` | Manage a flat array of `AnalysisContainer` objects with shared memory layout for multi-GPU; domain-aware `signal_operation` (add/subtract signals from residuals) | `src/lisatools/analysiscontainer.py` |
| `inner_product` / likelihood functions | Pure math: weighted integral `4 df ∑ a*b / Sn` | `src/lisatools/diagnostic.py` |
| `Likelihood` / `GlobalLikelihood` | Bridge Eryn sampler ↔ `AnalysisContainer`; handle parameter transforms | `src/lisatools/sampling/likelihood.py` |
| `TDWaveformBase` | Abstract base for time-domain waveform + TDI response + domain transform | `src/lisatools/sources/waveformbase.py` |
| `GlobalFitEngine` | Subclass of Eryn `EnsembleSampler`; injects `AnalysisContainerArray` into `GlobalFitInfo` model | `src/lisatools/globalfit/engine.py` |
| `GeneralSetup` | Parse `GeneralSettings`; load+preprocess data; build domain, sensitivity, orbit objects | `src/lisatools/globalfit/engine.py` |
| `GlobalFitMove` / `GFCombineMove` | Mixin on Eryn moves adding MPI rank management and iteration bookkeeping | `src/lisatools/globalfit/moves/globalfitmove.py` |
| `ResidualAddOneRemoveOneMove` | The core global-fit move: remove old template from residual → propose → add new template → compute likelihood | `src/lisatools/globalfit/moves/addremovemove.py` |
| `PSDMove` | Propose new PSD parameters; compute noise likelihood using GPU kernel `psd_likelihood` | `src/lisatools/globalfit/moves/psdmove.py` |
| `LISAToolsBackend` / `LISAToolsCpuBackend` etc. | Dispatch `OrbitsWrap`, `SensitivityMatrixWrap`, `psd_likelihood`, `compute_logpdf` to the right compiled module | `src/lisatools/cutils/__init__.py` |
| `Recipe` | Ordered list of pipeline stages with stopping functions; drives `GlobalFitEngine` through search → PE phases | `src/lisatools/globalfit/recipe.py` |

## Pattern Overview

**Overall:** Layered container architecture with residual-update MCMC

**Key Characteristics:**
- Domain-aware templates: every signal carries its own `DomainSettingsBase` (t0, dt, NT, NF); the residual-update logic intersects time/frequency ranges before adding or subtracting
- Shared mutable residual: `AnalysisContainerArray` holds a flat linear memory buffer that all containers reference via in-place slice views; signal operations mutate this buffer in-place so that likelihood evaluations always see the correct current state
- Backend dispatch via `gpubackendtools`: classes inherit from `LISAToolsParallelModule` → `ParallelModuleBase` and call `self.backend.xp` / `self.backend.psd_likelihood`; the backend is selected at construction via `force_backend`

## Layers

**Domain Layer:**
- Purpose: Represent signals as typed arrays with attached grid metadata; define transforms between TD, FD, STFT, WDM
- Location: `src/lisatools/domains.py`
- Contains: `DomainSettingsBase`, `DomainBase`, `DomainBaseArray`, `TDSettings`, `FDSettings`, `STFTSettings`, `WDMSettings`, `TDSignal`, `FDSignal`, `STFTSignal`
- Depends on: NumPy/CuPy, SciPy signal
- Used by: `datacontainer.py`, `sensitivity.py`, `waveformbase.py`, `analysiscontainer.py`

**Data Container Layer:**
- Purpose: Wrap a `DomainBase` as a multi-channel data/residual/signal array; propagate domain transforms on construction
- Location: `src/lisatools/datacontainer.py`
- Contains: `DataResidualArray`
- Depends on: `domains.py`
- Used by: `analysiscontainer.py`, `diagnostic.py`, `globalfit/`

**Sensitivity Layer:**
- Purpose: Compute and store the noise PSD (or its inverse/determinant) for use in the likelihood inner product
- Location: `src/lisatools/sensitivity.py`
- Contains: `Sensitivity` (ABC), `SensitivityMatrix`, `SensitivityMatrixBase`, channel-specific subclasses (`AE1SensitivityMatrix`, `AET2SensitivityMatrix`, `XYZ1SensitivityMatrix`, etc.), `XYZSensitivityBackend`, `get_sensitivity`
- Depends on: `detector.py`, `domains.py`, C++ backend (`SensitivityMatrixWrap`)
- Used by: `analysiscontainer.py`, `diagnostic.py`

**Analysis Container Layer:**
- Purpose: Combine data + PSD into a single object from which likelihood values, inner products, and SNRs can be computed; manage multi-GPU memory layout
- Location: `src/lisatools/analysiscontainer.py`
- Contains: `AnalysisContainer`, `AnalysisContainerArray`
- Depends on: `datacontainer.py`, `sensitivity.py`, `diagnostic.py`
- Used by: `globalfit/`, `sampling/likelihood.py`

**Likelihood / Diagnostic Layer:**
- Purpose: Pure mathematical functions (inner product, noise likelihood term, source likelihood term, Fisher information matrix, covariance)
- Location: `src/lisatools/diagnostic.py`
- Depends on: `datacontainer.py`, `sensitivity.py`
- Used by: `analysiscontainer.py`, `sampling/likelihood.py`

**Sampling Layer:**
- Purpose: Bridge between Eryn MCMC sampler and `AnalysisContainer`; wrap likelihoods and parameter transforms
- Location: `src/lisatools/sampling/`
- Contains: `Likelihood`, `GlobalLikelihood`, GMM utilities (`gmm.py`), prior helpers (`prior.py`), stopping functions (`stopping.py`), sky-mode-hop move (`moves/skymodehop.py`)
- Depends on: `analysiscontainer.py`, `diagnostic.py`, Eryn
- Used by: `globalfit/`

**Source Layer:**
- Purpose: Waveform generators for each source type; all emit `DomainBase` objects
- Location: `src/lisatools/sources/`
- Contains: `TDWaveformBase`, `AETTDIWaveform`, `SNRWaveform`; concrete waveforms in `emri/`, `gb/`, `bbh/`
- Depends on: `fastlisaresponse`, `gbgpu`, `bbhx`, `few`; domain layer
- Used by: `globalfit/moves/`, `sampling/likelihood.py`

**Global Fit Layer:**
- Purpose: Orchestrate multi-source MCMC over the shared residual using MPI parallelism and a phased recipe
- Location: `src/lisatools/globalfit/`
- Depends on: all layers above, Eryn, MPI (`mpi4py`)

**C++ Backend Layer:**
- Purpose: Compiled CPU/CUDA implementations of orbit calculation, sensitivity matrix computation, PSD likelihood, and log-PDF kernels
- Location: `src/lisatools/cutils/` (Python dispatch) + `lisatools_backend_cpu` / `lisatools_backend_cuda{11,12,13}x` (separate compiled packages)
- Contains: `Detector.cu/.hpp`, `PSD.cu/.hpp`, `binding.cxx/.hpp`, `psd_likelihood_utils.py`

## Data Flow

### Primary Request Path: Single-Source Likelihood

1. User constructs `DataResidualArray` with multi-channel FD/STFT data and a `DomainSettingsBase` (`datacontainer.py`)
2. User constructs `SensitivityMatrix` from a noise model or PSD function (`sensitivity.py`)
3. `AnalysisContainer(data_res_arr, sens_mat, signal_gen)` is created (`analysiscontainer.py`)
4. MCMC sampler calls `ac.eryn_likelihood_function(params)` or `ac.calculate_signal_likelihood(*params)`
5. Signal generator (`signal_gen(*params)`) returns array → wrapped in `DataResidualArray` as `template`
6. `AnalysisContainer._slice_to_template(template)` intersects time/frequency grids if needed
7. Three inner products are computed via `diagnostic.inner_product`: `<d|d>`, `<h|h>`, `<d|h>` → `logL = -½(d_d + h_h - 2*d_h)`
8. `diagnostic.inner_product` accumulates `4 * df * Σ (a*.b) * invC` over channels, using `psd.invC` from the sensitivity matrix

### Global Fit Residual-Update Path

1. `AnalysisContainerArray` holds N `AnalysisContainer` objects whose data arrays share a contiguous memory buffer (`linear_data_arr`)
2. Each MCMC step, a `ResidualAddOneRemoveOneMove` calls `acs.signal_operation(-1, old_templates)` to add old signals back into the residual
3. New parameters are proposed (stretch move or custom move)
4. New waveforms are generated via `TDWaveformBase.__call__` → `DomainBaseArray`
5. `acs.signal_operation(+1, new_templates)` subtracts the new signals
6. `AnalysisContainer.likelihood()` or a custom GPU kernel computes the likelihood on the updated residual
7. Metropolis–Hastings accept/reject; if rejected, the residual is reverted

### PSD Fitting Path

1. `PSDMove` proposes new noise parameters via stretch move
2. `XYZSensitivityBackend` evaluates the PSD given new parameters using the C++ backend (`psd_likelihood` GPU kernel)
3. Full noise + source likelihood is computed; accepted if improved; `SensitivityMatrix.invC` is updated in-place

## Key Abstractions

**DomainBase / DomainSettingsBase:**
- Purpose: Every signal array travels with its grid (t0, dt, df, NT, NF); transforms between TD/FD/STFT are methods on the array
- Examples: `src/lisatools/domains.py` — `TDSignal`, `FDSignal`, `STFTSignal`
- Pattern: Settings dataclass + associated array class; `settings.associated_class` maps settings → signal class

**AnalysisContainerArray:**
- Purpose: The central mutable state of the global fit; exposes `signal_operation(sign, templates)` as the single interface for all residual updates
- Examples: `src/lisatools/analysiscontainer.py` lines 596–1170
- Pattern: Flat linear memory buffer partitioned across GPUs; in-place slice assignment keeps all `AnalysisContainer.data_res_arr._arr` views valid

**GlobalFitMove:**
- Purpose: Mixin that adds `comm`, `ranks`, `gpus` properties to any Eryn move class
- Examples: `src/lisatools/globalfit/moves/globalfitmove.py`, `addremovemove.py`, `psdmove.py`
- Pattern: MPI rank assignment at pipeline init (`assign_ranks`), then consulted at each MCMC step

**Recipe:**
- Purpose: Ordered sequence of pipeline stages each with its own settings adjustments and stopping function
- Examples: `src/lisatools/globalfit/recipe.py`
- Pattern: `add_recipe_component(adjust_fn)` → `Recipe.__call__(iter, state, sampler)` iterates stages

## Entry Points

**Single-source analysis:**
- Location: `src/lisatools/analysiscontainer.py` — `AnalysisContainer`
- Triggers: User scripts, notebooks
- Responsibilities: Likelihood, SNR, inner product for one source

**Global Fit pipeline:**
- Location: `src/lisatools/globalfit/run.py` — `CurrentInfoGlobalFit`; `src/lisatools/globalfit/pipeline.py` — `MBHSearchSegment`, `InitialPSDSearch`, `InitialGBSearchSegment`, `FullPESegment`
- Triggers: `run_mpi_global_fit.py` at repository root via MPI
- Responsibilities: Orchestrate MPI ranks, construct sampler, run recipe stages

**Stock setup:**
- Location: `src/lisatools/globalfit/stock/erebor.py`
- Triggers: `from lisatools.globalfit.stock.erebor import Setup` → GB/MBH/PSD settings objects
- Responsibilities: Provide production-ready configuration for the Erebor run configuration

## Architectural Constraints

- **Threading:** Single-threaded Python event loop per MPI rank. Parallelism is across MPI processes (one process per source type: GB PE, MBH PE, PSD, GB search). GPU operations are CuPy-async on a per-rank GPU.
- **Global state:** `AnalysisContainerArray.linear_data_arr` and `linear_psd_arr` are module-level mutable arrays shared by reference across all `AnalysisContainer` objects in a run. Concurrent writes without coordination would corrupt the state.
- **Circular imports:** Lazy imports are used in `domains.py` (`from ..domains import get_stft_settings` inside functions) to avoid circular import chains between `domains` ↔ `datacontainer` ↔ `sensitivity`.
- **Residual consistency constraint:** After any accepted MCMC step the residual must equal `data - Σ(accepted signals)`. Moves that reject a proposal must revert the residual in the same step. Failure to do so silently corrupts all future likelihoods.
- **Backend selection:** `force_backend` must be propagated consistently from the top-level `GeneralSettings.gpu_backend` all the way down to each `DomainSettingsBase`, `Orbits`, and `SensitivityMatrix`; mixing CPU and GPU objects is not supported.

## Anti-Patterns

### Passing raw ndarrays to `signal_operation`

**What happens:** Calling `acs.signal_operation(sign, raw_array)` bypasses the domain-aware path and uses a legacy index-based approach.
**Why it's wrong:** It ignores time/frequency offsets in the template settings, silently writing into the wrong frequency bins when templates have non-zero `t0` or partial frequency coverage.
**Do this instead:** Wrap templates in `DomainBase` (or produce a `DomainBaseArray` from `TDWaveformBase.__call__`) and pass those to `signal_operation`. The domain-aware path in `AnalysisContainerArray._apply_stft_signal / _apply_fd_signal / _apply_td_signal` computes the correct overlap slices automatically.

### Constructing `DataResidualArray` without `input_signal_domain`

**What happens:** Passing a raw numpy array without specifying `input_signal_domain` raises a `ValueError`.
**Why it's wrong:** Without knowing the domain (FD vs STFT vs TD), the container cannot infer `f_arr`, `df`, `dt`, or perform domain transforms.
**Do this instead:** Always pass a `DomainSettingsBase` instance as `input_signal_domain`, or pass a `DomainBase` object directly as `data_res_in`.

### Calling `noise_likelihood_term` with an incorrectly shaped PSD

**What happens:** `noise_likelihood_term` asserts that the number of NaN/Inf entries in the PSD equals either 0 or the product of the channel-shape dimensions.
**Why it's wrong:** A PSD with unexpected NaN positions (e.g., only some DC bins set to NaN across a 3×3 matrix) will trigger an assertion error mid-run.
**Do this instead:** Ensure DC bin handling is consistent; set the entire DC slice to `1e100` or `NaN` for all channels simultaneously, as done in `Likelihood.inject_signal` (`sampling/likelihood.py` line 186).

## Error Handling

**Strategy:** Assertions and `ValueError`/`NotImplementedError` at construction time; silent degradation (warnings) for non-critical issues such as non-overlapping template time ranges.

**Patterns:**
- Construction-time type/shape assertions in `AnalysisContainer.__init__`, `DataResidualArray.__init__`
- `warnings.warn` for overlapping-range mismatches in `AnalysisContainerArray._apply_stft_signal` (`analysiscontainer.py` line 873)
- `breakpoint()` calls remain in `sampling/likelihood.py` (lines 680, 716, 718) — active debugging artifacts

## Cross-Cutting Concerns

**Logging:** `logging.getLogger(__name__)` pattern used in `globalfit/` modules; `init_logger` utility in `globalfit/loginfo.py` writes to per-component log files inside `artifacts_file_dir`
**Validation:** Pydantic not used in the hot path; input validation is by `isinstance` checks and `assert` statements
**Authentication:** Not applicable (scientific computation)

---

*Architecture analysis: 2026-04-28*
