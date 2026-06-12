# GPU validation checklist (post stft_tof merge)

Items discovered during the CPU-only merge/integration work that need
validation on real GPU hardware (single- and multi-GPU). Append as new
items arise; check off with notes + date when validated.

## Multi-GPU (ACA substrate + DCGA orchestration)

- [ ] **DCGA routing on dev's ACA sharding (>1 GPU)**:
  `DomainComputationGroupArray.unpack_indices` / `ac_to_split` /
  `ac_to_intra` consume `acs.gpu_splits` / `split_map` from the dev-side
  AnalysisContainerArray. Verify walker→split→intra-index round trips on
  2+ GPUs (stft's DCGA was written against their own ACA variant).
- [ ] **`run_threaded=True` per-thread device context**: cupy's current
  device is thread-local; each ThreadPoolExecutor worker must enter its
  split's `Device(gpu)` (via the group's `device_context`). Confirm no
  cross-device launches under threading.
- [ ] **`run_async=True` + memory-pool hygiene**: audit
  `free_all_blocks()` placement after threaded/async ops (the stft branch
  had IMA history around pool reuse during async kernels; the defensive
  `.copy()` fixes in `prepare_likelihood_inputs` address aliasing — verify
  under load).
- [ ] **MultiGPUPSDMove parity**: end-to-end logL agreement vs single-GPU
  `PSDMove` (use `_force_parent_path=True` to bisect DCGA-routing vs
  kernel discrepancies).
- [ ] **MultiGPUResidualAddRemoveMove parity**: per-device waveform-gen
  replicas vs dev's general `ResidualAddOneRemoveOneMove` on one GPU.
- [ ] **Cross-device template hygiene**: `AnalysisContainerArray
  ._signal_on_device` (contiguity + host-routed copy, no P2P assumption)
  — exercise a template generated on GPU A applied to a shard on GPU B,
  in both `signal_operation` paths (DomainBase + legacy raw-array).
- [ ] **`num_per_gpu_walker` mapping** in GB init
  (`recipe_steps` `generate_global_template(num_per_gpu=...)`) on >1 GPU.
- [ ] **Threading rollout (follow-up)**: once DCGA is validated, route
  dev's sequential per-device loops (`acs.likelihood()`,
  `signal_operation` shard loop) through `_loop_operation(run_threaded=
  True)` and re-verify parity + scaling.

## Reactivated XYZ sensitivity backend + galactic grid

- [ ] **`psd_likelihood_wrap` GPU kernel** (hybrid PSDMove fast path):
  parity vs the ACA route on FD and STFT bases; `run_async=True` variant.
- [ ] **CUDA wheel build** of the renamed `binding_detector.cxx` module
  (cluster `cuda12x` build; GPU branch of the new alias blocks:
  `XYZSensitivityMatrixWrapGPU`, `GalacticGridWrapGPU`,
  `GalacticGridSetupGPU`, `NoiseLevelsGPU`, `FDDomainForStftGPU`).
- [ ] **GalacticGrid device lifecycle**: `allocate_and_setup` →
  `initialize` → `set_galactic_grid` pointer attachment; foreground term
  in the likelihood on GPU.
- [ ] **FD time-averaged transfer functions**
  (`average_transfer_functions=True`): `_build_and_attach_averaged_tfs`
  GPU path (~1024-epoch TF average) and `set_averaged_tfs_wrap` device
  pointers.

## GB move (info-matrix + stretch mix)

- [ ] **`gb.information_matrix` GPU call** with the merged signature
  (`psd=linear_psd_arr, noise_index=, data_length=, batch_size=10000`)
  — confirm against the GBGPU dev merge (Phase C) and on-GPU behavior.
- [ ] **Cholesky-jump proposal on cupy**: `xp.linalg.cholesky(inv(info_mat))`
  batched on GPU; cache add/remove bookkeeping under RJ churn.
- [ ] **`stretch_probability` mix**: acceptance-rate sanity for both
  proposal types in the in-model loop on GPU runs.

## MBH phentax path (PhenomTHMTDIWaveform + engine-side signal_gen)

- [ ] **Legacy-response fallback loop on GPU**: until the Phase-B port of
  the vectorized legacy response (batched `get_projections` + `run_async`)
  into `lisatools.response.directresponse`, `TDWaveformBase._apply_response`
  loops the batch against the single-source API, mutating
  `response.num_pts` per call. Verify on GPU (cupy strain slices, per-call
  buffer sizing); delete the fallback when Phase B lands.
- [ ] **Engine-side residual rebuild on GPU**: `setup_acs(rebuild_residuals
  =True)` drives `signal_gen` → `build_template` → WDM transform per
  walker; confirm cupy end-to-end and memory-pool behavior for full-year
  grids (Nf*Nt ≈ 12.6M TD samples per waveform window).
- [ ] **`output_domain_settings` WDM placement**: `place_td_signal_on_grid`
  + full-grid `TDSignal.transform(WDMSettings)` on GPU (large rfft).
- [ ] **ICRS orbits end-to-end**: `L1Orbits(frame='icrs')` + raw catalogue
  `(ra, dec, psi_icrs)` through the response on GPU; CPU smoke verified
  injection/template overlap = 1.0 in synthetic mode.

## Cross-cutting

- [ ] **TDI flavor-int re-base**: stft's `domains.hpp` previously used
  `TDI_XYZ=0/TDI_AET=1`; re-based to canonical `1/2/3` (wdm_domain.hh).
  When the STFT domain wraps are bound, verify Python passes
  `backend.TDITypeDict` values into the STFT/FD kernels (never literals).
- [ ] **nanobind leak warnings at exit** (leaked types/instances on
  interpreter shutdown): currently benign module-teardown noise — confirm
  no growth under long GPU runs.
