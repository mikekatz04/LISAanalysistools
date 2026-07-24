# Multi-GPU smoke runbook — all_sources stock global fit

Verifies on real hardware what the CPU fake-shard tests
(`tests/test_gb_shard_router.py`, `tests/test_addremove_multi_shard.py`,
`tests/test_psd_move_multi_shard.py`, `tests/test_gather_scatter_roundtrip.py`,
`tests/test_band_view_multi_shard.py`, GBGPU
`tests/test_intra_split_index.py`) verify structurally: every module of the
all_sources run spreads over `GPUS=<list>`, and the walker groups of
residuals/PSDs shard across devices while staying accessible from every
move. No settings files — everything is env + stock classes.

## 1. Single-GPU baseline

```sh
OMP_NUM_THREADS=1 GPUS=0 DATA_MODE=synthetic NUM_ITERATIONS=2 \
    python scripts/run_global.py --stock all_sources
```

Record from the log:
- initial per-walker lnL (the `setup_acs` / first-iteration likelihood line),
- one completed propose per branch (gb, vgb, psd, mbh, emri, sobbh),
- `SubBandBuffer: ... GPU pool used/total` lines,
- `nvidia-smi --query-gpu=index,memory.used --format=csv` at steady state.

## 2. Two-GPU run, same seeds

```sh
OMP_NUM_THREADS=1 GPUS=0,1 DATA_MODE=synthetic NUM_ITERATIONS=2 \
    python scripts/run_global.py --stock all_sources
```

Assert:

1. **lnL parity**: initial per-walker lnL matches the single-GPU baseline to
   ~1e-8 *relative*. NOT bitwise — shard-parallel reduction order differs.
2. **Every branch proposes cleanly**: one propose per branch completes with
   no cupy cross-device errors (`peer access`, `different device`), and the
   log shows `Built the run-shared DomainComputationGroupArray (2 splits...)`.
3. **Both devices carry the plane**: `cp.cuda.Device(i).mem_info` /
   `nvidia-smi` watermarks show BOTH devices populated with roughly half the
   single-GPU main-ACA + SubBandBuffer load. Cross-check the static
   estimator: `python scripts/diagnostics/gpu_memory_estimate.py`.
4. **Post-production reclamation**: the steady-state watermark should track
   the estimator (persistent ACA/DCGA terms only) — data-production
   transients are swept once at the build→sampling transition
   (`run.py::setup_acs`).
5. **Proposal-exit teardown**: after each GB propose the
   `buffer lifecycle` log line appears and per-device pool totals drop back
   (SubBandBuffer scratch is proposal-scoped).
6. **Uneven-walker warning**: rerun once with `NWALKERS` not divisible by
   the GPU count and confirm the `nwalkers=... not divisible by len(gpus)`
   warning fires.

## 3. Known limits (expected, not failures)

- **Sig-het in-model GB references** (`GB_MODE` sig-het in-model path) raise
  `NotImplementedError` on multi-shard buffers — single-shard state on the
  shared computation object; per-shard comp replicas are the follow-on.
  Run that path on a single GPU.
- **WDM-basis PSD move** stays on the (correct) rebuild fallback — the
  FD/STFT kernel gate keeps the DCGA replica path off.
- **Comp-level device-sticky caches**: kernel config flows through
  per-launch host→device wrapper uploads (LISA Analysis Tools convention),
  which is device-safe; if a comp-level cache (e.g. an orbit spline eval
  cache) turns out to be pinned to the device it was first built on, it
  shows up here as a cross-device error in the GB FD path — report with the
  shard id and the failing kernel name.
