# WDM sub-band leakage & bookkeeping — investigation plan

Status: **COMPLETE (measured 2026-08-06).** Written 2026-08-06 to survive a context compaction;
the "Structural findings" section below was added when the investigation ran
and CORRECTS several premises of the original plan. Read that section first —
where it disagrees with the text underneath, it wins.

---

## Structural findings (code-verified 2026-08-06)

Five facts, each read out of the producer, that change what the measurement is
actually about.

**1. `layer_df` does NOT shrink with `T_obs`.** The stock grid pins the wavelet
pixel duration to `[3600, 4400]` s independent of `T_obs`
(`stock/erebor/fit.py:100-104` -> `derive_wdm_grid` -> `adjust_to_even_bins`),
so `Nf = wavelet_duration/dt` and `layer_df = 1/(2*Nf*dt)` are `T_obs`-
independent; only `Nt` grows. The original plan's "gets worse with `T_obs`:
`layer_df` shrinks" is **wrong**.

**2. The FD minimum band width is `T_obs`-independent beyond 1 yr.**
`2*get_N*df` with `df = 1/T_obs`: `get_N`'s `T_obs` multiplier doubles exactly
when `df` halves, so the width in Hz is flat for `T_obs >= 1 yr` and 4x WIDER
at 3 months. Sub-bands per layer at `wavelet_duration = 3600 s`
(`layer_df = 1.3889e-4 Hz`):

| f0 | 3 mo | 1 yr | 2 yr | 4 yr |
|---|---|---|---|---|
| 0.5 mHz | 17.1 | 68.5 | 68.5 | 68.5 |
| 1-8 mHz | 8.6 | 34.2 | 34.2 | 34.2 |
| 12-22 mHz | 2.1 | 8.6 | 8.6 | 8.6 |

So the pressure to sub-divide is real and largest at LOW frequency, it jumps
once from 3 mo to 1 yr, and then stops.

**3. The per-source likelihood window follows the source's own `f0`, not the
band edges.** `m_active` is built from `m_floor = floor(f0_cand / layer_df)`
spanning `+- m_active_half_width` (`gb_tdi_on_the_fly.cu:2117-2123`, and the
CPU mirrors at `:2380`, `:3456`); `m_band_half_width` defaults to 1, so every
source's likelihood is a **3-layer window centred on its own carrier**.
Band width therefore NEVER truncates a likelihood, and sub-dividing bands does
not change the likelihood computation at all — it only changes which sources
are grouped into a proposal cell and which are concurrently open.

**4. The 7-layer slab is a memory container, not the likelihood window.**
`band_slab_Nf = max_span + 2*(leakage + guard) = 1 + 2*(2+1) = 7` layers
(`gbbands.py:1445-1466`), centred per band. Neighbouring bands' slabs overlap
in ~6 of 7 layers **already**, at any divisor. That overlap is NOT the hazard —
the 3-layer likelihood window from (3) is.

**5. Opening a parity class ADDS the templates back.**
`remove_cold_chain_sources_from_residual` -> `adjust_sources_in_residual_buffer(+1)`
(`gbspecialstretch.py:979-993`), i.e. `r -> r + h`, restoring raw signal. So
within a parity unit, band B's window contains band A's **raw signal as
unmodeled power** — a static bias, not a drift. The error term is
`<h_A | dh_B>` (RJ magnitude), which is the worst case, applied to every
proposal in the unit.

### Consequence: the real hazard, and where the knob is missing

Contamination requires the two 3-layer windows to overlap, i.e.
`|m_A - m_B| <= 2` layers. Parity keys on **band index**
(`gbbands.py:2787`: `inds_keep &= self.band_inds % units == remainder`) with
`band_units` **hardcoded to 2** (`gbspecialstretch.py:418`) and **never wired
to `subband_divisor`** — `recipe.py` does not pass it and `GBSettings` has no
field for it. So:

| config | same-parity separation | 3-layer windows |
|---|---|---|
| `div=1, units=2` (today) | 2 layers | touch at exactly 1 layer |
| `div=D, units=2` (naive sub-division) | `2/D` layers | overlap heavily |
| `div=D, units=2D` | 2 layers | same as today |

**But `units=2D` gives back no parallelism.** Bands per parity pass is
`(D*N)/(2D) = N/2` either way, and total pick rounds are `2D * (max/D) = 2*max`
— unchanged. The parallelism win from sub-dividing comes *precisely* from
holding `band_units` fixed while shrinking bands, which is exactly what erodes
the separation. That trade-off is the decision, and the section-3 number prices
it.

---

## RESULTS (measured 2026-08-06)

Script: `scripts/diagnostics/wdm_subband_leakage.py` (CPU, `--sections 1,2,3,4,5`;
`--replot results.json` redraws without recomputing). Everything goes through
`lisatools.diagnostic.inner_product` + the installed domain transforms — no
hand-rolled likelihood. 3-month stock grid: `Nf=360`, `Nt=2190`,
`layer_df = 1.3889e-4 Hz`, one layer = **1095 bins** of `1/Tobs`; source
amplitude calibrated to `SNR = 20`.

### The algebra, verified

`ll = -1/2 <r|r>`, band B proposing `r -> r - dh_B`, band A having concurrently
moved the residual by `dh_A`:

    dll_B(r_fresh) - dll_B(r_stale) = -<dh_A | dh_B>          (*)

Checked end-to-end against four actual `ll` evaluations (not the algebra) at
every separation: **max relative mismatch 1e-6 at 3 mo, 3.7e-6 at 1 yr** —
catastrophic-cancellation limited (`ll ~ 1e2-1e3`, gap ~1e-6 of it), not an
algebra error.

### 1. Overlap envelope, FD vs WDM (3 mo)

| sep (bins) | layers | FD | WDM |
|---|---|---|---|
| 1.2 | 0.001 | 2.0e-1 | 2.0e-1 |
| 10.6 | 0.010 | 1.8e-2 | 1.8e-2 |
| 77 | 0.071 | 1.9e-3 | 2.3e-3 |
| 290 | 0.265 | 3.0e-4 | 8.3e-4 |
| 563 | 0.514 | 2.0e-5 | 5.1e-4 |
| 1360 | 1.24 | 1.2e-6 | 1.5e-6 |

Two sources are orthogonal **well inside one layer**. WDM tracks FD below ~50
bins; between 0.1 and 1 layer WDM holds a floor ~1e-3 that FD does not,
consistent with the 1.8% WDM-vs-FD normalization difference measured on the
diagonal (`SNR_wdm/SNR_fd = 0.9825`). **All downstream numbers use the WDM
value** — the sampler computes in WDM, so that is the conservative choice.

### 2. WDM frequency localization (energy fraction outside +-k layers)

| Tukey alpha | k=0 | k=1 | k=2 |
|---|---|---|---|
| 0 (none) | 3e-8 | 1.2e-9 | ~2e-9 floor |
| 0.01 | 1.4e-7 | 5e-11 | 3.6e-12 |
| **0.05 (stock)** | **2e-11** | **9e-14** | **6e-16** |
| 0.1 | 1.3e-11 | 6e-16 | <1e-19 |

The "sources overlap in WDM" intuition is a **time**-axis overlap. In frequency
a GB is confined to its own layer at the 1e-11 level. Note `alpha=0` is WORSE
at large `k` (broadband leakage off the hard time edges, flat ~2e-9 floor) —
the Tukey taper is what buys the steep fall-off.

### 3/4. The bookkeeping error (GO/NO-GO), envelope

`|dlnL|` induced in band B, at `rho_A = 20`. Error scales **linearly in
`rho_A`**.

| sep (bins) | | 3 mo RJ | 3 mo in-model | 1 yr RJ | 1 yr in-model |
|---|---|---|---|---|---|
| 1 | | 31 | 7.0 | 48 | 1.5 |
| ~20-30 | | ~1 (crossing) | 0.38 | 0.31 | 0.033 |
| 128 | `2*get_N` | 0.13 | 0.059 | 0.062 | 0.007 |
| 256 | | 0.10 | 0.024 | 0.018 | 0.003 |
| 1095 / 4383 | 1 layer | 3.0e-4 | 4.1e-4 | 1.4e-5 | 6.3e-6 |
| 2 layers | **today** | **3.0e-5** | 1.1e-5 | 7.9e-7 | 1.3e-7 |

RJ envelope crosses `|dlnL| = 1` at **29.7 bins (3 mo)** and **9.6 bins (1 yr)**.

**Today's configuration sits at 3e-5 — five orders of magnitude of headroom.**

**`T_obs` makes it BETTER, not worse.** At a fixed bin separation the 1-yr
error is *below* the 3-month one (0.062 vs 0.13 at 128 bins) despite the 1-yr
SNR being 1.9x higher (38.6 vs 20). Per unit SNR that is a 4x improvement.
The original plan's `T_obs` worry is refuted twice over — by finding (1)
above and by this.

### 5. Is shared-sky conservative? Yes.

Sections 1/3 give A and B the same sky/inc/psi, maximizing correlation. With B
drawn isotropically (24 draws):

| sep (bins) | shared sky | random median | random p90 | random max |
|---|---|---|---|---|
| 128 | 2.2e-1 | 2.6e-2 | 9.5e-2 | 1.5e-1 |
| 256 | 9.8e-2 | 2.2e-2 | 5.2e-2 | 1.0e-1 |
| 1024 | 1.9e-3 | 4.1e-4 | 1.2e-3 | 1.4e-3 |
| 1095 | 4.6e-4 | 2.0e-4 | 4.5e-4 | 6.5e-4 |

Shared-sky sits at or above the random-sky p90 everywhere: the table above is
conservative by **2-8x**.

## RECOMMENDATION

Under the `2*get_N` minimum-band rule with `band_units` left at 2, the
same-parity separation is `band_units * band_width = 4*get_N` bins —
**independent of `layer_df` and (beyond 1 yr) of `T_obs`**:

| f0 | sep | 3-mo RJ error @ rho=20 | @ rho=100 |
|---|---|---|---|
| 0.5 mHz | 128 b | 0.13 | 0.66 |
| 1-8 mHz | 256 b | 0.10 | 0.50 |
| 12-22 mHz | 1024 b | ~2e-3 | ~1e-2 |

All sub-threshold; the thin spot is the bright-neighbour case at low frequency,
and even that is conservative by 2-8x per section 5.

**Do NOT scale `band_units` with the divisor.** It is unnecessary (above) and it
would surrender the entire win: with `units = 2D` the bands per parity pass
(`D*N/2D = N/2`) and the total pick rounds (`2D * max/D = 2*max`) are both
unchanged from today. Holding `band_units = 2` while shrinking bands is exactly
what delivers the ~Dx reduction in pick rounds.

### Before using it

* **`GB_SUBBAND_DIVISOR` has zero tests and zero script usage.** Added
  2026-07-27 (`bb220b5`) alongside the fstat comb work, never exercised. It
  needs a smoke test before a production run — a correctness risk independent
  of the physics measured here.
* `band_units` is hardcoded to 2 (`gbspecialstretch.py:418`) with no
  `GBSettings` field and no `recipe.py` plumbing. Fine for this
  recommendation; there is simply no knob if it is ever wanted.
* `check_ll_inject` (`gbspecialstretch.py:5073`) repairs the **ledger** (full
  residual rebuild + exact lnL, gated at 1e-4), not the **decisions** already
  made inside the unit. It is not a mitigation for a biased `dlnL`; the
  measurement above is what prices that.

---

## The question (user's framing)

Two GB sources that do **not** overlap in frequency **do** overlap in WDM
(their time-frequency supports intersect). Two separate claims must not be
conflated:

* **(a) The inner product is basis-independent.** `<h1|h2>` is the same
  bilinear form in FD and WDM, so non-overlapping-in-frequency sources must
  give `<h1|h2> ~ 0` in WDM too — the overlap cancels by phase even where
  the supports intersect. If this fails numerically, the WDM inner product
  is wrong and everything downstream is suspect.
* **(b) The residual bookkeeping is NOT basis-independent.** If band A
  subtracts its template from pixels band B is concurrently reading, B's
  `Delta ll` is computed against a residual that moved underneath it. The
  likelihoods can be independent while the bookkeeping is still wrong.

**(b) is the real hazard**, and it is what limits how small sub-bands can get.

## Why it matters now

* Full band 0.5–22 mHz at `layer_df = 1.3889e-4` (90 d grid) = **~155 bands**
  at one band per WDM layer.
* User wants minimum band width `2 * get_N` (the FD law), which at low
  frequency is **narrower than one WDM layer** — i.e. multiple sub-bands
  inside a single wavelet layer.
* The parity scheme (odds, then evens) exists so concurrently-processed
  bands do not share pixels. It holds today ONLY because band index ==
  layer index. Sub-dividing a layer breaks that invariant: two same-parity
  bands can land in the same layer and be co-scheduled.
* Gets worse with `T_obs`: `layer_df` shrinks, so a fixed `2*get_N` minimum
  spans more layers.

## Measurements (each with a plot)

1. **`<h1|h2>` in WDM vs FD** vs controlled frequency separation. Confirms
   (a) numerically and exposes the suppression floor. Expect ~0 well below
   the diagonal terms; plot `|<h1|h2>| / sqrt(<h1|h1><h2|h2>)` vs `Δf` in
   both bases on one axis.
2. **Fractional WDM energy outside a source's own band**, vs Tukey `alpha`
   and vs separation. Quantifies the leakage (b) implies. This is the
   number the Tukey window is supposed to suppress — measure how much.
3. **The bookkeeping error itself.** Add/subtract one source's template in
   band A, measure the induced `Delta ll` error in a neighbouring
   same-parity band B. THIS is the go/no-go number for sub-dividing.
4. **(3) vs `T_obs`** (3 mo / 1 yr / 2 yr), since shrinking `layer_df` is
   what makes it worse.

## Then: can we paper over it?

`gbspecialstretch` already runs a check-and-refill (`check_ll_inject`) that
rebuilds the residual. Precedent exists. The question is whether per-iteration
drift stays under the accept/reject noise floor between refills. Same drift
metric already tracked: overnight_2 median 6.4 / p90 17.6 / max 77.6 over
221 proposals.

## Code pointers (verified)

* Band edges, WDM path: `stock/erebor/gb.py:595-615` — one band per WDM
  layer, edges at `k * layer_df / div`, `div = GB_SUBBAND_DIVISOR` (already
  supports sub-division!). `band_edges_override` replaces it wholesale.
* `band_N_vals` computed via `get_N` at `gb.py:617` but the comment says it
  is **unused by the WDM engine** — FD-style widths never reach WDM.
* `get_N`: `GBGPU/src/gbgpu/utils/utility.py:198` — Tobs->mult (<=1yr 1,
  <=2yr 2, <=4yr 4, <=8yr 8), f0->base (>=0.1 1024 / 0.03-0.1 512 /
  0.01-0.03 256 / 0.001-0.01 64 / else 32), SNR->`M = 2^ceil(log2(A
  sqrt(Tobs/Sm))+1)`, `N = max(M, N)`.
* Parity / scheduling: `gbspecialstretch.py::_run_band_unit` +
  `BandScheduler(subset.special_band_inds, self.num_band_preload)`.
  **Check whether parity keys on band index or layer index** — if band
  index, sub-division silently aliases same-layer bands into one parity.
* Neighbour subtraction reaches +-8 layers: `GB_SUBTRACT_BUFFER_LAYERS=8`.
* `_band_klohi` (`variants/gb_no_fg.py:375`) enforces >= 3 whole layers.
* 5-layer chunked-het gating also assumes layer-scale bands.
* Drift check / refill: `check_ll_inject` in `gbspecialstretch.py`.

## Proposed fix (hypothesis, to be confirmed by the measurements)

Key parity on **layer**, not band index, so sub-bands within a layer are
never co-scheduled. Then sweep band width down toward `2*get_N` and measure
(3) at each width.

## Open item carried in from the same session

`SIGHET_V3_NODES=64` is the ratio-EVALUATION node count `n_r` (raw waveform
evals — the expensive part), distinct from `v4_knots` (in-kernel resample,
"not a lever"). v5 consumes BOTH (`gbsignalhetcomputations.py:985`).

**Unresolved conflict:** the v4/v5 shootout concluded "n_r = 64 NOT 32",
while `TODO(T_obs-aware node law)` argues short baselines are OVER-resolved
and ~8 should match at 3 months. Both cannot hold at the same `T_obs`.
GBGPU `28ca7f6` implements the `T_obs` scaling but it is **opt-in**
(adaptive mode `SIGHET_V3_NODES=-1` only) and **unvalidated**. Before
enabling: sweep `n_r in {8,16,32,64}` at 0.25 yr with
`gb_sighet_proof_figure.py` and find where `|Delta lnL|` breaks the tiered
budget (`allowed(T) ~ max(0.1, T/100)`).
