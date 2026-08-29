# Unsticking the cold-chain edge deadlock — options exploration (2026-08-29)

Two sources adjacent across a sub-band edge each block the other from the true
configuration; the ladder cannot break it. **Direct split/merge is out of scope.**

## Findings that change the design space

- **F1 — the cell slab is already ~40 sub-bands wide** (`GB_WDM_BAND_SLAB_LAYERS=5`,
  1 layer = 8 sub-bands). Band b+1 is *already inside* band b's cell. **No option
  needs a wider window.** The deadlock is about residual CONTENT, not scope.
- **F2 — killing a neighbour's cold row IS the co-open, free and residual-exact.**
  `add_cold_chain_sources_to_residual` only touches `subset.inds` rows; flip a cold
  row to False and it is never re-subtracted for the rest of the propose. Because the
  parent is addressed by walker, that exposure is visible to *every rung*.
- **F3 — a killed leaf's coords do NOT survive `_write_back_state`** (alive leaves are
  repacked densely). So a *within-propose* hold-out is free and exactly reversible; a
  *multi-propose* hold needs its own persistence. Hard cost boundary.
- **F4 — the `in_model` move (recipe.py:2930) is the ideal host**: no births, no cap
  accounting, no tempering, and PE-inert *by installation* rather than by knob.
- **F5 — `_window_residual_lls(acs, edges)` already exists**, so the gate can be a
  support-widened pair window — neither a too-narrow band nor a noisy whole-walker stat.
- **F6 — `band_temps` includes the cold rung and `_adapt_band_temps` never touches
  index 0**, so a per-band cold-rung β<1 is a stable ~3-line knob.

**Caution that shapes everything:** the measured population is *two real neighbours
interfering*, not one source split (causal 1→2 transition test, p=1.0000 in the split
direction; overlap median 0.0115, max 0.416). **A knockout is the wrong operator for the
majority** and must be gated on a duplicate signature (high mutual overlap AND ladder
extinction) or it destroys real detections.

## Ranking

| # | Option | Effect | Cost | Risk | DB |
|---|---|---|---|---|---|
| 1 | **H. Edge-pair diagnosis census** | enabling | very low | none | inert |
| 2 | **A1+E1. Ladder-oracle single-propose hold-out in `in_model`** | high on duplicates | low | medium | search-only, PE-inert by install |
| 3 | **D1. Per-band cold-rung annealing** | medium | very low | medium | search-only |
| 4 | **B1. Randomized zero-width edge collapse** | high, structural | medium | medium | **DB-safe, PE-safe** |
| 5 | **F2. Best-of-K pair enumeration** | high, safe on real neighbours | low-med | low | argmax / MTM-valid |
| 6 | **B2. Joint pair in-model proposal** | high on the *majority* | high | low | **DB-valid, PE-safe** |
| 7 | **F1. Carlin–Chib indicator over the pair** | high | high | medium | **fully valid, PE-usable** |
| 8 | A2-multi. Multi-propose hold + birth veto | high if A1 shows "almost" | medium | high | search-only |
| 9 | G2. Pair-coupled caps | containment only | low | low | search-only |
| 10 | E2. Hot-rung parameter import | low (topology not fit) | medium | medium | search-only |
| 11 | D2. Tempered transitions | high in principle | very high | low | valid |
| 12 | C1. Per-rung parent context | nil without joint pair swap | very high | high | valid |

## The options

**H. Diagnosis census (do first).** Per-propose `[GB_EDGE]` line: adjacent-band cold
pairs per walker (5-line adaptation of `_ortho_boundary_pairs`), pair-window residual
lnL, per-leaf ⟨d|h⟩/⟨h|h⟩ from existing captures, **mutual overlap ⟨h_i|h_j⟩ — the
duplicate-vs-real-neighbour discriminator**, and occupancy-vs-temperature. Nearly free,
and without it every actuator below is untargeted and therefore dangerous.

**A1. Single-propose solo hold-out in `in_model`.** Restore victim's power to the parent,
set `inds=False`, run the propose (survivor gets its 100 repeats against a residual that
now contains the neighbour), then gate on the pair-window lnL and keep-or-resurrect.
Makes the death free and unpriced, gives the second half 100 repeats, prices the pair
jointly at the end. Restore-on-failure exact within the propose.

*Victim choice:* (a) test both, take the better window lnL; (b) **E1 ladder oracle** — the
side that goes extinct fastest with temperature (one bincount, zero likelihood calls);
(c) lowest ⟨h|h⟩; (d) alternate by parity; (e) coin flip (keeps symmetry if a DB version
is ever wanted). *Hold length:* one `in_model` propose (free) → one iteration (needs
persistence + a birth veto; the band-shutoff machinery is reusable) → R iterations
(highest risk). *Criterion:* strict improvement + a movement interlock (survivor must
actually move, or the "improvement" is noise). *Do not price it* — unpriced, gated on a
true likelihood improvement, same contract as `GB_REPLACE_FSTAT_MAX`.

**A3.** Compose A1 with D1: anneal the survivor during the hold. ~3 lines over A1.

**B1. Randomized zero-width edge collapse.** Set `band_edges[b+1] = band_edges[b+2]` for
one propose. `num_bands` unchanged, so the cap-grid mismatch raise never fires. The two
leaves land in **one cell** — the compound update becomes expressible with *no new
operator*. **DB-safe if the collapsed set is drawn uniformly and state-independently**
(same argument as the unit-scan draw). Must be global across walkers. Needs: `band_N_vals`
max of the two, re-run `check_band_support_separation`, and a collapsed-band guard in the
shutoff valve. ⚠ Resembles the *cancelled* alternating frames — the difference is real
(delete selected edges vs shift the whole grid) but **the user should rule on it
explicitly.** Honest limit: co-residency makes the update expressible but still serial —
converges, no longer stuck; the extra win needs B2.

**B2. Joint pair in-model proposal.** Move both leaves along the dominant eigenvector of
their joint information matrix. **The correct operator for the real-neighbour majority**,
fully DB-valid and PE-safe. Real work: the serial-within-band raise must be relaxed for
this class, and the accept kernel is shaped around one row per source.

**C2. Cell-local pretend co-open.** Inject the neighbour's power into band b's cell only.
Subsumed by F2 (which gives the same exposure at parent level for every rung, free).
Keep as a proposal generator for a delayed-rejection second stage.

**D1. Per-band cold-rung annealing.** `band_temps[b,0] = β<1` for K proposes. The RJ death
at β=1 costs full ΔlnL and is never accepted; at β=0.3 it costs 0.3×. ~3 lines. Risk: the
cold chain *is* the parent context for everything — keep β modest and the pair set small.

**F1(lit). Carlin–Chib product-space.** Retain both leaves' coords with an indicator
z ∈ {both, left, right}; the pair update becomes a fixed-dimension Gibbs draw on z, a
softmax over three window evaluations. **The only fully-valid, PE-usable option here.**
Cost: pseudo-prior design + retained-coord persistence.

**F2(lit). Best-of-K enumeration.** Stage {both, kill-left, kill-right} and take the
argmax. Natural upgrade from A1's binary gate at ~2× staging. **Degrades gracefully: on a
real neighbour "keep both" simply wins.**

**F4. Annexation.** Kill the adjacent cold leaf and re-fit into the freed power, scored on
the widened window. **Be honest — this is a merge in all but name.** Whether the absence of
a dimension-matching bijection puts it inside the ruling is the user's call; ask first.

## Try first

**H, then A1+E1, in the same build.** Census is nearly free and is the readout for
everything; A1 is the user's own idea in its cheapest correct form, with F4 removing its
biggest objection (no births to refill the hole) and F3 making it exactly reversible.
**Arm in measure-and-revert mode first** — reuse the reverted pair-borrow's two-knob
pattern (`_BORROW` measures, `_BORROW_APPLY` keeps).

If the census shows a large *real-neighbour* population A1 correctly refuses to touch, the
follow-on is **B1 then B2** — the only path correct for the majority and clean enough for PE.

## Dead ends

- **B3 pair-atom / non-modulus unit partitions** — two adjacent cells concurrently scored
  is exactly the orthogonality violation the stride prevents. Only viable as B1.
- **Any window widening** — F1: the slab is already ~40 sub-bands.
- **E3 joint pair vertical swap** — blocked by the same trap it escapes; hot rungs have no
  parent-residual representation to price against.
- **C1 alone** — moves nothing without a joint pair swap, and destroys the vertical swap's
  free-exchange argument.
- **Per-walker band grids** — `_tempering_swap_grid`'s cross-walker permutation requires
  band index to denote the same Hz for every walker.
- **Mid-propose re-homing** — the label keys the residual bookkeeping (gbbands.py:4560-4573).
- **Delayed rejection in literal form** — degenerates into F4 (annexation).

## Key files

`gbspecialstretch.py`: `propose` 14042, `run_proposal` 3290,
`adjust_sources_in_residual_buffer` 2270, `_window_residual_lls` 12745,
`_write_back_state` 12608, `_ortho_boundary_pairs` 715, `_band_shutoff_*` 6187-6429,
`_replace_phase_max` 6025 (PE-stamp pattern).
`gbbands.py`: band-label freeze 4560-4573, `get_subset_bool` 4984,
`add/remove_sources_to_band_buffer` 4343, `check_band_support_separation` 171,
`_compute_band_slab_Nf` 2884.
`recipe.py`: `in_model` install 2930, stage stamps 3096-3245.
`git show cb4bff9f` — reference for staging, true-lnL gating, double PE interlock.
