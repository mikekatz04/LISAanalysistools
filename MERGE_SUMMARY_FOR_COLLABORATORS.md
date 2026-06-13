# stft_tof → dev merge: summary of changes vs `origin/stft_tof`

*(Draft — finalized at the end of the integration; for the Erebor collaborators.
Branch: `merge-stft-tof`; base: dev + your 315 commits since d690a0a.)*

## What we kept from stft_tof (and where it went)

- **`PhenomTHMTDIWaveform` + the `TDWaveformBase` family** (`sources/waveformbase.py`,
  `sources/bbh/waveform.py`) — kept wholesale and is now the MBH path in the
  combined global-fit settings. Extended with `output_domain_settings` so the
  same generator also serves WDM-domain runs (your FD/STFT paths unchanged).
- **`XYZSensitivityBackend` evolution** — your galactic-grid foreground, FD
  time-averaged transfer functions (`average_transfer_functions`, 1024-epoch),
  `f_1/f_knee/f_2` renames, and `run_async` are all active. (Dev had this
  backend disabled due to "symbol issues on Linux" — root cause was missing
  CPU/GPU class-name aliases, now fixed, so it's enabled everywhere.)
- **Multi-GPU orchestration**: `MultiGPUMoveBase` + `DomainComputationGroupArray`
  (per-split computation groups, `ThreadPoolExecutor` threading, `run_async`)
  kept as the orchestration layer on top of dev's `AnalysisContainerArray`
  sharding. `MultiGPUPSDMove` and `TDMBHSpecialMove` are intact.
- **PSD move**: hybrid — dev's domain-agnostic AnalysisContainer route plus your
  C++ kernel fast path (auto-selected for FD/STFT single-shard runs).
- **MBH info-matrix proposal**: your Fisher/Cholesky-cache proposal replaced
  dev's chunked-het NUTS inside the GB/MBH special moves (FD-pinned info
  matrix; mixed per-iteration with the band-aware group stretch via
  `stretch_probability`).
- **Priors structure** (`globalfit/priors/`), `utils/typing.py`, the `JaxBase`
  threading mixin (relocated to `lisatools.jax.jaxbase`), run-metadata fields +
  `postprocessing.py` submission machinery, `mbh_catalogue_to_sampling_basis` /
  `gb_catalogue_to_sampling_basis` / `setup_state_for_injection` /
  `scatter_around_injection` / `subtract_initial_signal` recipe helpers.
- **DDPC analytic sky conversions** (`sources/utils.py`): astropy-free
  `icrs_to_ecliptic` / `ecliptic_to_icrs` (+ analytic psi rotation), validated
  against astropy in `tests/test_sky.py`.

## What changed relative to your branch (please adopt these conventions)

1. **nanobind, not pybind11.** The whole sprint moved to nanobind (pinned via
   `constraints/sprint.txt`); all incoming pybind11 code was ported. The
   detector binding is `binding_detector.{cxx,hpp}` (renamed from
   `binding.{cxx,hpp}`); the Python module is still `pycppdetector`.
2. **Domains are never communicated by strings.** `basis_domain="stft"` /
   `stft_dt=` are gone from `GeneralSettings`; pass
   `domain_settings=STFTSettings.make_factory(big_dt=..., min_freq=..., max_freq=...)`
   (or `FDSettings`/`WDMSettings` factories / instances). All dispatch is
   `isinstance` on `DomainSettingsBase` children. Your `mojito_input/` settings
   files have been ported to this (minimal diffs; `general_set.start_freq`-style
   reads still work).
3. **`DataResidualArray` is deprecated.** Everything flows through
   `DomainBase` children (`FDSignal` / `STFTSignal` / `WDMSignal` / `TDSignal`);
   `.data_res_arr` is a self-referential compat property.
4. **Engine-side template generation.** Branches register a params-based
   `signal_gen` on their `Settings`; `run.py::setup_acs(rebuild_residuals=True)`
   builds and subtracts the state's templates centrally. Settings recipes build
   moves only — no `subtract_initial_signal` / residual writes in settings files
   (legacy recipes without `signal_gen` still work unchanged).
5. **ICRS is the run frame.** Catalogue sky/polarization sampled raw
   (`alpha`/RA, `sin_delta`, `psi` ICRS); orbits loaded with `frame="icrs"`;
   the stock MBH transform (`make_mbh_transform_container`, forward+inverse,
   `Q = m1/m2`) is direct-ICRS. The LISA→SSB→ICRS transform chains were
   removed. (`SkyMove` inner moves are temporarily disabled for MBH until an
   ICRS-basis variant exists.)
6. **One instance = one backend.** No `use_gpu=True` / backend strings as
   method kwargs; backends are chosen at construction
   (`force_backend="cpu"/"cuda12x"/...`).
7. **TDI flavor ints re-based.** Your `domains.hpp` used `TDI_XYZ=0/TDI_AET=1`;
   the canonical sprint values are `XYZ=1, AET=2, AE=3` (`backend.TDITypeDict`).
   Python must pass the dict values, never literals.
8. **`FDDomain` name collision**: your C++ `FDDomain` is temporarily
   `FDDomainForStft` (dev's WDM-era `FDDomain` kept its name); the two will be
   unified into one class.
9. **`propagate_data_res_kwargs`** was removed from the MBH move's
   likelihood kwargs (the DataResidualArray propagation path it configured is
   gone; it now would crash `inner_product`).
10. **Vectorized legacy response pending**: your batched
    `pyResponseTDI.get_projections` (array lam/beta/t0 + `run_async`) lives in
    the lisa-on-gpu `tdi_on_fly` branch and is being ported into
    `lisatools.response.directresponse`; until then `TDWaveformBase` detects
    the API and falls back to a per-source loop (numerically identical).

## Validation status

- LAT test suite green (58 tests; CPU).
- Synthetic full-year combined run (WDM domain): settings construct; the
  engine-side MBH `signal_gen` produces a `WDMSignal` whose overlap with the
  injected data is 1.000000; the `setup_acs` rebuild nulls the residual
  exactly.
- `tests/test_sky.py` validates the analytic ICRS↔ecliptic conversions
  against astropy.
- GPU items are tracked in `GPU_VALIDATION_CHECKLIST.md` (multi-GPU routing,
  XYZ backend kernels, info-matrix on GPU, CUDA wheel build of the renamed
  binding, …).

## TODO(open) — known follow-ups

- Port of the vectorized legacy response (drops the per-source fallback loop).
- ICRS-basis `SkyMove`; re-enable MBH sky-mode hops.
- Unify `FDDomainForStft` + `FDDomain`; finish the C++ domains consolidation.

## Done since the draft

- **WDM counterparts to the STFT C/C++ kernels**:
  `WDMDomainWrap.compute_likelihood_terms` (two-pass batched (d|h)/(h|h)
  kernels in `cutils/domains.cu`, real-valued, integer (m, n) sub-grid
  addressing, XYZ/AET/AE) + `WDMComputationGroup` in `domaincomputation.py`
  (auto-selected by `DomainComputationGroupArray` for `WDMSettings` data).
  Validated against the Python WDM `inner_product` path to ~1e-15
  (`tests/test_wdm_domain_cpp.py`).
- CPU-thread splits through the same DCGA structure
  (`AnalysisContainerArray(n_splits=...)` + `run_threaded=True`; threads ↔
  GPUs via one Python orchestration layer). The pure-compute C++ bindings
  release the GIL so CPU threads parallelize for real.
- Combined STFT+WDM time-frequency tutorial:
  `examples/time_frequency_domains_tutorial.ipynb`.
