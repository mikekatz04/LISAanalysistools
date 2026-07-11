# Multi-GPU validation runbook (parallel-resources plan P1)

The P1 code landed CPU-verified (single-split behavior identical; pure-logic
unit checks green). This runbook is the **cluster half**: validate the
multi-GPU paths on ≥2 real devices. Run on the GPU cluster in `flr_env`
(module recipe in the sprint build notes: gcc/11.2.0 + cuda12.6 toolkit +
`jax[cuda12_local]` env).

## What changed (all inert at 1 GPU / CPU)

1. **Band-grouped shard assignment** — `band_gpu_assignment(..., group_ids=band_ids)`
   (`analysiscontainer.py`): every cell of a band shares a shard; bands
   round-robin across GPUs *within each even/odd parity class*. Replaces
   plain slot striping, under which every tempering swap pair (a row's
   adjacent temps = consecutive slots) was cross-shard.
2. **Same-shard swap fast path** — `BandView.swap_rows` +
   `SubBandBuffer.swap_template_slots`: same-shard pairs swap in place in
   their device context (no host hop); cross-shard pairs fall back to
   gather/scatter.
3. **Per-GPU temperature permutation** — `run_tempering` /
   `_permute_walkers_for_swaps` (gbspecialstretch.py): swap partners drawn
   within each device's cold-chain walker block.
4. **Load-balanced walker blocks** — ACA `np.array_split` (sizes differ by
   ≤1) + imbalance log.
5. **Mempool churn opt-in** — the per-pick-round `free_all_blocks()` is now
   gated behind `GB_MEMPOOL_FREE_EACH_ROUND=1`.

## Gates (run in order)

### Gate 1 — 1-GPU baseline vs current dev
```sh
GPUS=0 GF_NUM_ITER=50 NWALKERS=8 NTEMPS=4 DATA_PROCESSOR=synthetic MAKE_PLOTS=0 \
  python scripts/run_global.py --stock gb_no_fg
```
Must be statistically identical to pre-P1 dev (same seeds → same chain).
Single-GPU paths were untouched; any diff is a bug.

### Gate 2 — 2-GPU assembly + likelihood identity
```sh
GPUS=0,1 ... python scripts/run_global.py --stock gb_no_fg   # same knobs
```
- Startup log shows the ACA walker split (balanced blocks) and no imbalance
  warning at nwalkers % ngpus == 0.
- Instrument `BandView.swap_rows`: count same-shard vs cross-shard pairs per
  tempering pass (add a temporary counter or logger.debug). **Expect ~100%
  same-shard** under band grouping; a high cross rate means the buffer slot
  ordering violates the band-grouping assumption — investigate
  `unique_band_combos` ordering before trusting timings.
- `acs.likelihood()` at the injected start must match the 1-GPU value to
  ~1e-12 (same data, same PSD; only storage moved).

### Gate 3 — sampling equivalence (the real gate)
Same run, 1 GPU vs 2 GPUs, ~500 iterations:
- per-band tempering acceptance rates statistically indistinguishable
  (the per-GPU walker permutation restricts swap partners; with balanced
  blocks this must not change acceptance in expectation);
- posterior corners for the seeded band overlap;
- the incremental-LL drift-repair rate (`check_ll_inject` warnings) does not
  increase vs 1 GPU.

### Gate 4 — scaling + churn
- Wall-clock per iteration: 1 vs 2 GPUs on a wide-band gb_no_fg config
  (`GB_MIN_FREQ/GB_MAX_FREQ` wide → many bands). Record the split between
  proposal, tempering, and fill stages (`GB_TIMING` lines).
- `GB_MEMPOOL_FREE_EACH_ROUND=0` (default) vs `=1`: the `mempool_free`
  stage time should collapse at 0 with no OOM on the production config.

## Deliberately left for cluster iteration (measure first)

- **BandView index-resolution caching**: `_resolve_array` does host
  argsort/searchsorted per access. Cache per buffer build if Gate 4 shows it
  hot. (Slot→shard maps are static per allocation.)
- **`cudaMemcpyPeer` path** for the residual cross-shard copies
  (`_signal_on_device`, `reset_linear_psd_arr` host hops) — only if Gate 4
  shows them on the critical path; guard with `deviceCanAccessPeer` +
  host-hop fallback.
- **Label indirection for swaps** (plan §2.5): swap (temp,walker) labels
  instead of slab contents — evaluate only if the same-shard in-place swap
  is still hot after Gate 4.
- **Buffer refill drift**: the shard assignment is fixed at allocation while
  the scheduler swaps cells into freed slots; log the same-shard swap rate
  over a long run to see whether band-locality decays across refills (if it
  does, make the scheduler prefer shard-matching slots).
