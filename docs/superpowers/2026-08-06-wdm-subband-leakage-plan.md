# WDM sub-band leakage & bookkeeping — investigation plan

Status: **planned, not started.** Written 2026-08-06 to survive a context
compaction. Everything below is either measured or a code pointer already
verified; nothing here is assumed.

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
