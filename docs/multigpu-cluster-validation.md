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
GPUS=0 NUM_ITERATIONS=50 NWALKERS=8 NTEMPS=4 DATA_MODE=synthetic MAKE_DIAGNOSTIC_PLOTS=0 \
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

## Addendum (2026-09): multi-rank walker-block port

The gates above validate the **in-process multi-GPU router** — one process
switching cupy contexts across several devices. That router is still live
code and still exercised, but which of its paths run depends on the rank
layout (`docs/global-fit-launch.md`; design spec
[`docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md),
Decision 12):

- **At `-n 1`** (or any layout where a lone compute rank owns its node's
  whole GPU pool, the AUTO `gpus_per_rank` resolution): the router runs
  exactly as validated by Gates 1-4 above, unchanged. A plain single-process
  `GPUS=0,1 python scripts/run_global.py --stock <name>` still exercises
  both devices this way.
- **Under several compute ranks *per node*** (`mpiexec -n <k>` with `k >= 2`
  compute ranks landing on the same node): each rank's
  `AnalysisContainerArray` is single-device by default (`gpus_per_rank=1`),
  so the router runs in its exonerated single-shard passthrough form on
  every rank — the band-grouped shard assignment, same-shard swap fast path,
  and per-GPU temperature permutation this runbook validates are simply not
  exercised (there is nothing to shard).
  Reproducing Gate 2's literal "2-GPU assembly" command as multi-device
  under this layout now additionally requires `GPUS_PER_RANK=2` (or
  `RANKS_PER_GPU`, for sharing rather than multi-device ownership) — a bare
  `GPUS=0,1` at `np=1` still means one rank driving both devices, exactly as
  these gates assume.
  The *per node* qualifier is load-bearing: AUTO resolves per node, so a
  multi-node launch whose nodes host exactly ONE compute rank each gives
  every one of them its node's whole pool and puts the router back in play.
  Pin `GPUS` to a single device per node (or set `GPUS_PER_RANK=1`) when
  that is not what you want — `docs/multirank-cluster-gates.md` Steps 0/1
  do exactly this for their 2-node commands, so that every parity layout is
  single-device.
- **`gpus_per_rank > 1` with several compute ranks** (a rank owning more
  than one device while other ranks exist) is layout-supported but
  **unvalidated** by either this runbook or the multi-rank port: it
  re-enables the in-process cross-device paths *inside* a rank, and the
  09-15 audit that motivated walker-block sharding in the first place
  (ambiguous `cp.asarray` cross-device semantics, an unfixed
  `BandView._scatter` bug, an unguarded SOBBH cross-device assertion) is an
  explicit prerequisite before anyone runs it in that configuration (design
  spec, Risks).

**New runbook for the rank axis itself** — transport parity across layouts,
the statistical gate against today's in-process 2-GPU run, and the
load-balance/payload-size measurements — lives separately in
[`docs/multirank-cluster-gates.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/multirank-cluster-gates.md). Run this
document's Gates 1-4 first (single-rank multi-GPU router still correct),
then that runbook for the multi-rank transport and statistics gates.

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
