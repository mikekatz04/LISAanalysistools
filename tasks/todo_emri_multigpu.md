# Task: EMRI move multi-GPU utilization

Status: Stages A + B APPROVED by user (2026-07-08); Stage C gated on Stage B
measurements; Stage D last resort. Implementation in progress.
Goal (agreed): root cause first ✔, then wall-clock scaling + simultaneous GPU
utilization for the EMRI move. Constraints: move/scheduling changes and
processes-over-threads acceptable; few (FastEMRIWaveforms) changes last resort
(no dev privilege); async+threading pattern works for PSD/MBH and stays.

## Root cause (evidence-backed)

Live run (PID 1167088): the two per-GPU thread-pool workers share exactly one
CPU core (64%/36%); GPU0/GPU1 utilization anti-correlated, sum ≈ one GPU.
EMRI proposal ≈ 210 s ≈ 85% of each ~255 s iteration.

1. **GIL held through heavy native calls.** `binding_flr.cxx:128-130`
   (`get_response_wrap`, `get_tdi_delays_wrap`, `get_response_quintic_wrap`)
   lack `nb::call_guard<nb::gil_scoped_release>()`; EMRI response runs with
   `run_async=False` default (`waveformbase.py:745`) → internal
   `cudaDeviceSynchronize` (`LISAResponse.cu:982,:459`) with GIL held.
   few's Cython wrappers (no `nogil`) + pure-Python DOPR853 trajectory hold the
   GIL for the rest.
2. **Chunk lock-step.** `batch_size_per_gpu=1` → ~600 serial chunks/iteration,
   each: thread join → unconditional `synchronize()` on both GPUs
   (`analysiscontainer.py:2449`, nullifies `run_async=True`) → serial D2H →
   `free_gpu_memory()` (`emrispecialmove.py:106-108`) → cold allocator.
3. **Single-GPU bookkeeping.** ~40 waveforms/iter regenerated one-at-a-time on
   GPU0's replica only (`addremovemove.py:969-977` `get_waveform_here`), with
   `gc.collect()` + pool free per source (`addremovemove.py:200-207`).
4. **Split imbalance.** `randomize_split` × contiguous walker→GPU map →
   E[max] ≈ 5.9 vs 5 chunks/half (~18% waste). MBH already has the fix:
   `TDMBHSpecialMove.get_split_inds` (`mbhspecialmove.py:173-190`).

## Stage A — cheap, LAT-only (no architecture change)
- [x] Add `nb::call_guard<nb::gil_scoped_release>()` to the 3 response bindings
      in `src/lisatools/cutils/binding_flr.cxx`; backend rebuilt by user
      (LAT 1.2.8.post1.dev828+gd13c82dd9, .so 2026-07-08 15:34).
- [x] Run EMRI response/TDI with `run_async=True` (constructor kwarg via
      `waveform_init_kwargs` in the settings file).
- [x] `DEBUG_MODE` split: full-array NaN/Inf scans now opt-in via
      `LISATOOLS_WAVEFORM_DEBUG=1`; the cheap orbit-range crash guard stays
      always-on with cached orbit extrema.
- [x] `batch_size_per_gpu` 1 -> 5 (settings); per-chunk `free_gpu_memory()`
      removed from `EMRISpecialMove._compute_like_chunk` (pool stays warm
      across chunks/repeats; emptied per proposal by the base move).
- [x] ~~Skip the unconditional `synchronize()` in `_compute_group_likelihood`~~
      DROPPED (user decision): likelihoods must reach the host every chunk for
      the proposal scope; the barrier is shared PSD/MBH code and its cost is
      amortized once batching cuts chunk count. Reduce sync *frequency*, not
      sync points.
- [x] GPU-stratified split: `EMRISpecialMove.get_split_inds` override (indexed
      by split position, not GPU id — the MBH variant silently no-ops for
      non-contiguous device ids like [5,6]; left untouched, flagged to user).
- [x] `TDSettings.t_end` scalar property (used for `merger_time`) +
      `stft_t_arr` built as a strided arange — no more 6.3M `arange` per call.
      Bit-equivalence verified.

## Stage A2 — low-hanging follow-ups (approved 2026-07-08, implemented)
- [x] Split-parallel cold-chain bookkeeping: `_apply_cold_chain_sources`
      override on `MultiGPUResidualAddRemoveMove` (user decision: parent
      class, so MBH and future multi-GPU sources inherit it) — each split's
      thread generates its walkers' templates on the owning device and
      applies them there (single-row `signal_operation` per source; distinct
      residual buffers -> thread-safe). Kills the GPU0-only serial phase, the
      host-routed cross-device template copies, and the per-source
      `gc.collect()`. Tests: `TestEMRIColdChainBookkeeping` (routing/values/
      replica ownership, serial + threaded, EMRI + parent class).
- [x] `waveformbase._apply_response`: removed the `inspect.signature`
      capability probe AND the dead legacy single-source else-branch (user
      confirmed the batched pyResponseTDI API is fully implemented;
      `self.response` is always pyResponseTDI in this class).
- [x] `_data_time_check`: endpoint check (`t_data[-1] + t0`) instead of
      materializing the (batch, N) sum (`directresponse.py`).
- [x] `get_projections`: max-orbital-radius cached per orbits object instead
      of deepcopy + reduce every call (`directresponse.py`).

## Stage B — measure (gate for Stage C)
- [ ] Re-run; repeat the live diagnostics (per-thread CPU%, GPU0/1 series,
      per-proposal seconds from globalfit_run.log). Target check: threads on
      ~2 cores, GPUs correlated, proposal time drop.
- [ ] Add lightweight per-stage timers (traj / few-sum / response / TDI / STFT /
      likelihood) inside `_emri_signal_op` or via nsys to quantify the
      remaining GIL-bound (few) fraction.

## Stage C — if Stage B shows few's GIL floor dominates: per-GPU processes
- [ ] Persistent per-GPU likelihood worker processes (extend ProcessExecutor
      pattern, or dedicated MPI ranks — pipeline already runs under MPI): each
      worker owns waveform replica + data shard, receives coords, returns logl.
      Scales to the 8 A100s. Design doc first.

## Stage D — last resort (no dev privilege): few patches
- [ ] `nogil` in few's pyinterp/pyampinterp2D wrappers + drop per-segment
      device syncs; propose upstream.

## Side issues noted (separate from performance)
- [ ] Replicas share nested `inspiral_kwargs`/`sum_kwargs` dicts (shallow copy
      in `create_waveform_gen_replicas`, `addremovemove.py:902-925`) and few
      mutates them per call — benign today, latent cross-GPU leak. Deep-copy.
- [ ] GPU0 at 39.9/40 GB vs GPU1 at 23.2 GB in live run — understand the
      asymmetry (bookkeeping replica pinned to GPU0? persistent buffers?)
      before raising batch size.

## Review
(to fill after implementation)
