# Codebase Concerns

**Analysis Date:** 2026-04-28

## Tech Debt

**TDI2 sensitivity functions unverified:**
- Issue: Multiple sensitivity transfer functions carry explicit `TODO: check these functions for TDI2` comments with no validation tests.
- Files: `src/lisatools/sensitivity.py` (lines 460, 570, 824, 887)
- Impact: Using TDI2 channel data produces silently incorrect likelihood values. TDI1/AET path is the only validated configuration.
- Fix approach: Add dedicated TDI2 unit tests that compare against known analytic or published results; resolve each TODO comment once verified.

**Matrix inversion uses `np.linalg.inv` instead of Cholesky:**
- Issue: `SensitivityMatrix` (in `src/lisatools/sensitivity.py`, line 1296) computes full matrix inverse with `np.linalg.inv` in the off-diagonal (correlated noise) branch. The TODO comment explicitly requests a switch to Cholesky.
- Files: `src/lisatools/sensitivity.py:1296`
- Impact: Numerical instability for ill-conditioned covariance matrices at edge frequency bins; also slower than Cholesky solve on large arrays.
- Fix approach: Replace with `xp.linalg.cholesky` + solve, matching the pattern already used in `src/lisatools/cutils/psd_likelihood_utils.py:671`.

**Legacy raw-ndarray path in `signal_operation`:**
- Issue: `AnalysisContainer.signal_operation` still accepts raw `np.ndarray` / `cp.ndarray` input via a deprecated path that emits a `DeprecationWarning`. The legacy branch complicates every code path in the method.
- Files: `src/lisatools/analysiscontainer.py:1030–1097`
- Impact: Caller code that never migrated silently continues working; the extra branch increases maintenance surface and makes GPU/domain-type checks harder to enforce.
- Fix approach: Remove the raw-array branch after auditing all internal callers (globalfit generators are the likely remaining users). Gate removal behind a version bump.

**Hard-coded year constant in `stochastic.py`:**
- Issue: `year = 365.25 * 24.0 * 3600.0` is defined locally with the comment "hard coded for initial fits" instead of using `lisaconstants.ASTRONOMICAL_YEAR`.
- Files: `src/lisatools/stochastic.py:189`
- Impact: Slight numerical inconsistency (Julian year vs. astronomical year) across modules; potential for divergence if stochastic models are ever compared against other modules that use `YRSID_SI`.
- Fix approach: Replace with `from lisaconstants import ASTRONOMICAL_YEAR as YRSID_SI` matching the pattern in `src/lisatools/sources/waveformbase.py:12`.

**XYZ TDI type not implemented in global-fit template generators:**
- Issue: Both `GetMBHTemplates` and `GetEMRITemplates` hard-code a 2-channel (AE only) output slice and carry `# TODO: change for XYZ` / `# TODO: adjust for AET/XYZ rather than AE` comments.
- Files: `src/lisatools/globalfit/generatefuncs.py:73–100`
- Impact: Running a global fit with `tdi_type="XYZ"` silently drops the third channel, producing wrong likelihoods without an error.
- Fix approach: Parameterize the output channel count based on `tdi_type` and pass it through from `GlobalFitSettings`.

**Stochastic background verification flag:**
- Issue: `src/lisatools/stochastic.py:182` carries `# TODO: need to verify this is still working` with no corresponding test.
- Files: `src/lisatools/stochastic.py:182`
- Impact: Galactic foreground contribution to the PSD may be silently incorrect after recent refactors.
- Fix approach: Add a regression test comparing the stochastic foreground PSD against a pinned numerical output.

**`PSDMove` and `AddRemoveMove` do not support multi-GPU splits:**
- Issue: Per the project memory, moves that update the global data residual (`PSDMove`, `AddRemoveMove`) have not been adapted for the multi-split (one DCG per GPU) design.
- Files: `src/lisatools/globalfit/moves/addremovemove.py`, `src/lisatools/globalfit/psdglobal.py`
- Impact: Multi-GPU runs using more than one GPU split will silently update only one split's residual, leaving the others stale.
- Fix approach: Pass all GPU splits to move `__call__` and apply residual updates to each split, following the pattern established for `STFTComputationGroup`.

**WDM domain transform partially stubbed:**
- Issue: `WDMSignal.transform` raises `NotImplementedError` unconditionally (line 1206). The WDM lookup table class (`WDMLookupTable`) also carries an unimplemented `get_slice` path (domains.py:508).
- Files: `src/lisatools/domains.py:508`, `src/lisatools/domains.py:585`, `src/lisatools/domains.py:1206`
- Impact: Any code path that calls `.transform()` on a `WDMSignal` raises at runtime. The WDM domain is not usable end-to-end.
- Fix approach: Either implement the WDM transform kernel or raise a clear `NotImplementedError("WDM transform not yet implemented")` with a docstring milestone, and guard the class with an explicit "experimental" warning.

**Global fit `run.py` has multiple TODOs around MCMC state initialization:**
- Issue: Several aspects of state loading, nleaves adjustments, and GMM saving are marked as incomplete or require manual adjustment.
- Files: `src/lisatools/globalfit/run.py:240`, `256`, `263`, `291`, `314`, `359`, `372`, `454`, `466`, `565`
- Impact: Production runs may require manual intervention to tune state initialization parameters; GMM distributions are not saved, losing warm-start capability across run restarts.
- Fix approach: Consolidate state-initialization logic into a single method, automate nleaves detection, and add GMM checkpoint save/load at the end of each recipe step.

## Known Bugs

**`emri_params` undefined in `GetEMRITemplates.__call__`:**
- Symptoms: `NameError: name 'emri_params' is not defined` at `src/lisatools/globalfit/generatefuncs.py:101`.
- Files: `src/lisatools/globalfit/generatefuncs.py:101`
- Trigger: Calling the EMRI template generator at runtime.
- Workaround: None; the variable `emri_params` is referenced before it is extracted from `current_state`.

## Security Considerations

**No secrets-handling concerns identified** in source code. All configuration flows through environment variables or HDF5 files.

## Performance Bottlenecks

**Full matrix inverse at every likelihood evaluation:**
- Problem: The correlated-noise branch of `SensitivityMatrix` calls `np.linalg.inv` on an `(N_freqs, C, C)` tensor at evaluation time.
- Files: `src/lisatools/sensitivity.py:1296–1304`
- Cause: Matrix is recomputed on each call rather than pre-factored.
- Improvement path: Pre-factor once with Cholesky when the sensitivity matrix is constructed; cache `L` for reuse in `log_likelihood_matrix`.

**`galaxyglobal.py` is 1963 lines with dense nested loops:**
- Problem: `galaxyglobal.py` and `gathergalaxy.py` (1908 lines) contain deeply nested Python loops over galactic binary groups with manual index arithmetic.
- Files: `src/lisatools/globalfit/galaxyglobal.py`, `src/lisatools/globalfit/gathergalaxy.py`
- Cause: Incremental growth without structural refactoring.
- Improvement path: Profile hot loops; vectorize binary group assignments using NumPy fancy indexing; consider splitting into smaller focused modules.

**`gbspecialstretch.py` is 3986 lines:**
- Problem: The galactic binary MCMC stretch move is the single largest file in the project at nearly 4000 lines.
- Files: `src/lisatools/globalfit/moves/gbspecialstretch.py`
- Cause: Accumulated specializations for band management, GMM proposals, and GPU/CPU dispatch without extraction.
- Improvement path: Extract GMM proposal logic into `sampling/gmm.py`, separate band-management helpers, and isolate GPU dispatch.

## Fragile Areas

**`cutils` C++ extension loading:**
- Files: `src/lisatools/cutils/__init__.py`
- Why fragile: Extensions are imported inside a `try/except (ModuleNotFoundError, ImportError)` block that silently degrades to CPU-only mode. If a CUDA extension is partially built (e.g., wrong CUDA version), the fallback may hide a broken installation.
- Safe modification: Always test with `lisatools.has_backend("cuda12x")` before relying on GPU code. Check `uv run python -c "import lisatools; print(lisatools.has_backend('cpu'))"` after any build change.
- Test coverage: No tests exercise the fallback CPU path explicitly.

**`AnalysisContainer` backend compatibility checks are missing:**
- Files: `src/lisatools/analysiscontainer.py:395`
- Why fragile: The TODO comment explicitly notes that there is no check to ensure TDI channel structure and domain type are equivalent across backends. Mixing an AET `DataResidualArray` with an XYZ `SensitivityMatrix` will produce silently wrong inner products.
- Safe modification: Add domain/channel-type assertions in `AnalysisContainer.__init__` or `set_data`.
- Test coverage: No tests covering mixed-channel or mixed-domain container construction.

**MPI rank count assumptions in `MPIControlGlobalFit`:**
- Files: `src/lisatools/globalfit/run.py:639`
- Why fragile: A `ValueError("Not enough MPI ranks to give.")` is raised late at runtime, not at setup time. If the run is started with fewer ranks than sources, the error surfaces only after initializing all source setups.
- Safe modification: Add a rank-count preflight check in `MPIControlGlobalFit.__init__`.
- Test coverage: No tests for rank assignment logic.

## Scaling Limits

**Single-node MPI model:**
- Current capacity: The MPI communicator in `run.py` and `pipeline.py` is constructed for a single node, with one rank per source type.
- Limit: Scaling beyond ~10 simultaneous source types requires more ranks than typical single-node GPU counts.
- Scaling path: Implement hierarchical communicator groups per source type.

## Dependencies at Risk

**`mpi4py` hard import in `globalfit/run.py`:**
- Risk: `from mpi4py import MPI` at module top-level (line 8) means importing `lisatools.globalfit.run` in any context (including notebooks or tests) requires MPI to be installed and initialized.
- Impact: Any test or notebook that imports `run.py` will fail if `mpi4py` is not installed.
- Migration plan: Move `mpi4py` import inside the class or function that requires MPI, matching the `cupy` guarded-import pattern used elsewhere.

## Missing Critical Features

**No tests for `globalfit`, `analysiscontainer`, `datacontainer`, or `domains`:**
- Problem: The test suite contains only 25 test functions across 6 files, covering `detector`, `sensitivity`, `orbits`, `sources_utils`, `fresnel`, and `amp_phase_shifting`. The entire `globalfit/` package, `domains.py`, `analysiscontainer.py`, and `datacontainer.py` have zero test coverage.
- Blocks: Safe refactoring of the global fit pipeline and core container logic.

**GMM checkpoint save/load not implemented:**
- Problem: `run.py:67` notes `# TODO: save GMM distributions`. Between restarts, warm-start proposal distributions are lost and must be re-fitted from scratch.
- Blocks: Efficient multi-run global fit campaigns.

## Test Coverage Gaps

**`globalfit/` package:**
- What's not tested: All of `run.py`, `pipeline.py`, `recipe.py`, `recipe_steps.py`, `galaxyglobal.py`, `gathergalaxy.py`, `psdglobal.py`, `mbhglobal.py`, `generatefuncs.py`, `hdfbackend.py`.
- Files: `src/lisatools/globalfit/` (entire directory)
- Risk: Regressions in MCMC state management, HDF5 backend serialization, and MPI rank assignment go undetected.
- Priority: High

**`domains.py` and domain computation group:**
- What's not tested: `WDMSignal`, `WDMSettings`, `WDMLookupTable`, `TimeDomainArray.transform`, STFT boundary conditions (`# TODO: need to fix top and bottom layer`, line 564).
- Files: `src/lisatools/domains.py:508–591`, `src/lisatools/domains.py:1189–1206`
- Risk: Domain transform round-trips produce silent numerical errors.
- Priority: High

**`analysiscontainer.py` and `datacontainer.py`:**
- What's not tested: `AnalysisContainer` initialization, `signal_operation`, `DataResidualArray` channel arithmetic.
- Files: `src/lisatools/analysiscontainer.py`, `src/lisatools/datacontainer.py`
- Risk: Channel mismatch and residual sign errors pass undetected.
- Priority: High

**`stochastic.py`:**
- What's not tested: `StochasticContribution` PSD evaluation, galactic foreground fit as a function of time.
- Files: `src/lisatools/stochastic.py`
- Risk: Stochastic background contributions to likelihood are incorrect after refactors.
- Priority: Medium

---

*Concerns audit: 2026-04-28*
