# Sig-het v5 — global-memory residency for the reference fold blocks

**Date:** 2026-08-03
**Status:** design approved, not implemented
**Scope:** one experiment. Prove or disprove that moving sig-het's per-pixel
reference arrays out of shared memory buys speed.

## Problem

The sig-het v4-banded scorer runs at **1 block per SM** on an A100. Its
per-block shared footprint at `N_sparse_t = 204` is 143.4 KB against a
163 KB limit, so a second block cannot be resident. With one block per SM
there is almost nothing to hide memory and instruction latency behind.

The footprint splits cleanly:

| part | bytes | at `N_sparse_t = 204` |
|---|---|---|
| constant (node arrays `n_r=64`, knot arrays `K=128`, reduction scratch) | 39,104 | 38.2 KB |
| per-pixel (`2·nch·M·N_sparse_t·16` fold blocks + `nch·N_sparse_t·16` c0 row) | 528 · `N_sparse_t` | 105.2 KB |

**73% of the footprint is per-pixel data, and it scales with the observation
baseline.** It is also what pins `N_sparse_t` — the same ceiling behind the
two-year accuracy wall, though accuracy is explicitly *not* what this
experiment measures.

## What v5 is

v5 is **v4-banded with the per-pixel reference arrays resident in global
memory instead of shared**. Identical mathematics, identical results, one
change: where those arrays live.

Shared retains only the constant 38.2 KB, which is independent of
`N_sparse_t`. Predicted occupancy: **1 → 4 blocks/SM**.

### Why this could win

The fold blocks are *the reference*: `A0/A1/B0/B1` and the `c0` row are
identical for every candidate in the batch — that is the entire premise of
sig-het, which builds one reference per source and scores many candidates
against it.

Today every resident block loads its own private copy into its own shared
memory, so the same ~105 KB is duplicated across every block on every SM.
In global memory there is **one copy** read by all blocks, and at ~105 KB it
sits far inside A100's 40 MB L2. The apparent cost — HBM traffic — is
therefore mostly L2 hits, and the trade is N private copies for one shared
one, plus 4× the occupancy.

### Why this could lose

- Sig-het's kernel is ~90% waveform build, which is arithmetic-heavy. If it
  is **issue-bound rather than latency-bound**, extra occupancy buys little.
  This is the most likely way v5 disappoints.
- L2 is shared with everything else in flight; the residency argument is
  weaker under load than in isolation.
- The pixel-axis access pattern must stay coalesced. If the relocation forces
  a strided read, bandwidth will not behave as projected.

A negative result is a successful outcome. "1 block/SM is not what limits
this kernel" is worth knowing and would redirect effort to Phase 2.

### Relationship to Phase 2

Phase 2 moment contraction removes these same arrays **algebraically** — same
occupancy win, with *less* work rather than more memory traffic. Phase 2
strictly dominates v5 if its algebra holds.

v5's value is that it needs no new mathematics and is testable sooner. If v5
shows the occupancy win is real, Phase 2 is validated as worth the effort
before anyone writes the contraction. If v5 shows no gain, Phase 2's
occupancy argument is undermined too, and its case rests on the reduced work
alone. Either way v5 informs Phase 2 cheaply.

## Architecture

The change is confined to the GPU computation. **Wrapper and binding levels
do not change.**

- **`GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu`** — the v4 section
  (`gb_sighet_v4_shared_bytes`, `gb_signal_het_v4_score_one_source`,
  `gb_signal_het_v4_get_ll_kernel`). The scorer takes its fold blocks from a
  caller-supplied global pointer rather than carving them out of
  `shared_mem`. Indexing arithmetic is otherwise unchanged.
- **`GBComputationGroup`** owns a lazily-sized device slab for the fold
  blocks. This is what keeps the bindings stable: the buffer never crosses
  the binding boundary, so `gb_signal_het_v4_get_ll_wrap` keeps its exact
  signature.
- **Untouched:** every binding file, the `.def`, the nanobind registrations,
  and the Python comp surface.
- **Selection** rides the existing `for_band_engine` path the way `v4_band`
  already does — a constructor argument defaulting to off, so v4 and v5 are
  both live in one build and an A/B is a one-argument swap.
- **`gb_sighet_v4_shared_bytes`** gains a v5 branch returning the constant
  38.2 KB, so the benchmark's shared-memory accounting and the
  `fit_nt_layer` grid solver stay correct.

The existing v2/v3/v4 paths are not modified. v5 is additive.

## Validation

v5 must reproduce v4-banded to **~1e-12 relative** on identical inputs. Same
algebra, different residency: anything above round-off is a bug, not a
trade-off. Ground truth is the existing validated engine — no hand-rolled
likelihood, per the standing rule in this codebase.

Validation runs on CPU first, where `CUDA_SHARED` collapses to stack and the
relocation is a no-op semantically. That proves the indexing rewrite before
any GPU time is spent. **The CPU cannot measure the speed claim at all** —
shared-versus-global is a GPU-only distinction — so CPU is a correctness gate
only.

## Measurement

Via `scripts/gb_chunked_het/gb_engine_benchmark.py`, adding `v5` to
`ALL_ENGINES` so it is selectable through the existing `BENCH_ENGINES` knob.
Follow the established `timed_and_scored` methodology: one scored pre-loop
call supplies accuracy, the timing loop repeats the identical call, and the
first and last iterations are verified equal.

Primary measurement — **matched `N_sparse_t = 204`**, v4-banded against v5,
same build, same references:

1. µs per candidate across the batch ladder to saturation.
2. Achieved occupancy (blocks/SM), to confirm 1 → 4.
3. Accuracy identical to v4-banded, as a correctness check rather than a
   result.

Matched grid is deliberate: it isolates the residency question from the
accuracy question. Whether v5 can then run a finer grid than v4 is a separate
follow-on, not part of this spec.

## Success criteria

- **Correctness:** v5 ≡ v4-banded to ~1e-12.
- **Occupancy:** 4 blocks/SM confirmed, or an explanation of why not.
- **Decision:** a defensible yes/no on whether relocating the fold blocks
  speeds up sig-het, with the measurement supporting it.

## Out of scope

- **Two-size chunk transforms.** A chunked-het concern, tracked in the
  standing `TODO(CPU two-size chunks)` above `compute_chunk_geometry` in
  `src/lisatools/wdm_het.py`. Folding it in would blur this measurement.
- **Raising `N_sparse_t` past 204.** Follow-on once residency is proven.
- **Accuracy work.** The two-year wall is a separate thread.
- **Move integration.** No RJ, no `SubBandBuffer`, no stock-fit wiring.
- **Phase 2 contraction.** Informed by this, not part of it.

## Risks

| risk | mitigation |
|---|---|
| Kernel is issue-bound, occupancy buys nothing | The measurement's purpose. Report it as a finding. |
| Indexing rewrite introduces a subtle error | CPU correctness gate at 1e-12 before any GPU run. |
| Global reads not coalesced | Preserve the pixel-axis-fastest layout the shared version uses. |
| Scope creep into Phase 2 or accuracy | Out-of-scope list above is explicit. |
