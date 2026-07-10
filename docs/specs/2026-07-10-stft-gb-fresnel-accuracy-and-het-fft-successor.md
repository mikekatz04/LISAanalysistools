# STFT GB likelihood: Fresnel accuracy roadmap and the heterodyned-FFT successor

**Date:** 2026-07-10
**Status:** design study (no code changes; phased implementation outline in §5)
**Scope constraints:** per-segment window stays a NON-OVERLAPPING Tukey (window-family /
overlap changes deferred); mismatch target for the STFT GB path is **1e-6 .. 1e-8**
per source against the exact STFT of the data (GBs at detector SNR in the hundreds).

This document answers two questions for the `l2d-dev` STFT/Fresnel GB band-likelihood
path (`lisatools/cutils/lat_stft_kernels.hh` + `lisatools/cutils/domains.{hpp,cu}`,
Python surface `gbgpu.gbcomps.STFTGBComputations`):

- **Part A (§3):** what can or should be changed if the *Fresnel* mismatch needs to be
  lower — ranked from free knobs to kernel changes, with measured gains where available.
- **Part B (§4):** what to use *instead of* the Fresnel if we want both a lower mismatch
  and faster GPU execution — the STFT analog of the WDM domain's evolution
  (lookup table → chunked heterodyne → signal heterodyning + polyphasing).

The numbers quoted below come from the 2026-07 agreement studies
(`make_stft_fft_fd_agreement_plots.py`, `stft_window_alpha_sweep.py`, records in the
workspace task log), the FFT-per-column prototype spec
(`docs/specs/2026-07-01-stft-gb-fft-per-column-design.md`, org_setup root; prototype
lives on `feat-stft-gb`), the STFT test suite (`GBGPU/tests/test_stft_gb_accuracy.py`),
and code inspection of `l2d-dev`. New measurements can be reproduced per source with
`scripts/validation/gb_mojito_stft_fd_mismatch.py` (full / in-stencil / stencil-interior
decomposition).

---

## 1. Executive summary

The Fresnel model itself is *nearly* exact for a galactic binary: the measured
per-column method error is ~7e-6 (6 h segments, grid-aligned taper). The large
mismatches seen in practice are dominated, in order, by:

1. **stencil truncation** — the template populates only ±`n_side_bins` frequency bins
   per segment while the data's per-segment spectrum leaks everywhere (window-driven
   floor; the production default is `n_side_bins = 2`);
2. **edge segments** — first/last STFT segments when the observation is pinned to the
   orbit-file boundary;
3. **the Fresnel-integral evaluation floor** — the A&S 7.3.27/28 rational fits carry
   ~2e-3 per-value error, i.e. a mismatch floor of ~2e-6 (this is the measured 5–7e-6
   "Fresnel saturation" at strong tapers);
4. **structural model error** — constant envelope per segment and pure linear-chirp
   phase (no within-segment response modulation or phase curvature): ~7e-6 at 6 h,
   growing to ~8e-5 at 24 h segments.

**Part A conclusion:** knobs alone (`n_side_bins`, Tukey α, `freq_from_tdi_phase`,
`use_midpoint`, orbit-interior observation) take the path to ~1e-5..2e-6. Reaching
1e-6..1e-7 with the Fresnel requires two kernel changes: a precise Fresnel-integral
evaluation (A1) and an envelope/curvature correction (A2/A3). The cost, however, grows
with stencil width: the windowed evaluator spends ~22 fused sincos *per pixel*, so a
1e-6-grade Fresnel at `n_side ≳ 20` is an expensive kernel.

**Part B conclusion:** the recommended successor is a **heterodyned, spline-fed,
in-register FFT per (source, segment)** — "het-FFT column producer". It reuses the
chunked-het machinery that already exists on `l2d-dev`
(`fast_wdm_inner_heterodyne_spline` pattern: sparse `get_tdi` control points → carrier
de-rotation → cubic spline → windowed slow signal → radix-2 FFT → band extract) applied
per STFT segment. It is *exact* in the window (time-domain multiply), *exact* in the
within-segment physics to spline accuracy (measured mm < 4e-11 for the GB envelope
spline at N_cp=48), needs no Fresnel special functions at all, and its per-segment cost
(≈ 3·N_sub spline evaluations + 1–2 sincos + an N_sub-point in-register FFT,
N_sub = 8–32) undercuts the windowed Fresnel already at `n_side = 2` and scales better
with stencil width. This mirrors the WDM migration exactly: the transcendental-heavy
analytic projection is replaced by carrier removal + a tiny transform over a smooth,
cheaply interpolated envelope.

---

## 2. Where the Fresnel mismatch comes from

### 2.1 Measured error budget

All template-vs-exact-STFT (brute per-segment DFT of the same stream), XYZ, unweighted
complex overlap. From the 2026-07 agreement studies (1 yr, Δ=6 h unless noted):

| term | scaling knob | measured values |
|---|---|---|
| stencil truncation | `n_side_bins` × window α | α=0.1: n_side 10 → 1.25e-3, 15 → 4.5e-4, 20 → 5.2e-5, 25 → ~1e-5 plateau. Window floor at n_side=10: rect → 1.06e-2, α=0.1 → 1.25e-3, α=0.5 → 1.9e-6, Hann → 1.1e-7 |
| edge segments | observation placement | pinned at orbit t0: ~60× inflation of full-grid mm (6e-3 vs ~1e-4 in `test_stft_gb_accuracy`); interior unaffected |
| off-grid taper defect | — | **FIXED** (`get_windowed_fourier_value` off-grid phases): was 2.16e-2 → 9.57e-5 at 24 h / 10⁴ s taper |
| Fresnel-integral floor | evaluation precision | saturation ~5-7e-6 at α ≥ 0.5 where the FFT column stays exact (2.5e-6 → 5.3e-7); consistent with mm ≈ ε²/2 for ε ≈ 2e-3 per-value error |
| const-envelope + linear chirp | segment length Δ | ~7e-6 @ 6 h → ~8e-5 @ 24 h (grid-aligned taper, in-band interior) |
| f/fdot anchor quality | `freq_from_tdi_phase`, `use_midpoint` | tdi-phase estimator 9.53e-5 → 2.53e-5 (256-seg grid test); midpoint 2.09e-5 → 1.48e-5 (32 seg) |

The first two rows are *representation completeness* terms shared by ANY per-segment
stencil method (the exact FFT column sits on the same truncation floor); only the last
three rows are Fresnel-specific.

### 2.2 The code anatomy behind each term (l2d-dev)

- Kernel mapping: **block-per-binary, thread-per-segment**; per segment one direct
  response evaluation `src.get_tdi_Xf_single` at the anchor (`stft_eval_block_ll`,
  `lat_stft_kernels.hh:275`, anchor at :315), then a `(2·n_side_bins+1) × 3-channel`
  loop calling `STFTFresnel::get_fourier_value` per pixel (:324–336), block reduction
  into (d|h), (h|h).
- `freq_from_tdi_phase` (`stft_freq_fdot_from_tdi_phase`, `lat_stft_kernels.hh:150`):
  central difference of the TDI phase via complex-product args at t ± D,
  D = quarter carrier cycle capped at 3600 s → +2 direct response evaluations and
  6 atan2 per segment. Captures orbital Doppler (rate typically > astrophysical fdot)
  and response phase; the astro fallback captures neither.
- Windowed evaluator (`get_windowed_fourier_value`, `domains.cu:752`): the Tukey window
  is decomposed **exactly** into 7 sub-interval terms (full-segment rectangular + roll-on
  {DC, ±f_taper} + roll-off {DC, ±f_taper}, weights 1/−0.5/−0.25), each a Fresnel-kernel
  evaluation → **~22–24 fused sincos per pixel** (the unwindowed path costs ~4).
- **Fresnel-integral floor**: `get_fresnel_integrals` (`domains.cu:672`) uses the
  Abramowitz & Stegun 7.3.27/7.3.28 rational fits — the code comment itself measures
  ~1.9e-3 relative error on the Fourier value of a clean linear chirp and names the
  upgrade path. Since mm ≈ ε²/2 for an incoherent per-value error ε, this is a
  ~2e-6 mismatch floor — precisely the observed α ≥ 0.5 saturation.
- Structural assumptions: amplitude frozen at the anchor sample for the whole segment
  (`lat_stft_kernels.hh:332` feeding a constant prefactor, `domains.cu:767/817`);
  quadratic phase only (`get_v`, `domains.cu:633`; stationary-phase term at :746 —
  `fddot` never enters within-segment); integration bounds pinned to `[t0, t0+dt]`
  (`get_fresnel_kernel`, `domains.cu:726`); the engine's whole-span data taper is not
  represented anywhere in the evaluator.
- **No producer seam**: the pixel loop is written three times — `stft_eval_block_ll`
  (:275, shared by ll / fstat-diagonal / grad), `stft_eval_block_swap` (:624, shared by
  swap / fstat-off-diagonal / swap-grad), and an inlined copy in
  `stft_fill_global_kernel` (:464, atomic-add scatter). A `FresnelColumn/FFTColumn`
  template policy exists only on `feat-stft-gb` (spec §4), not on `l2d-dev`.
- Python surface (`gbcomps.py:STFTGBComputations`, :937): knobs `n_side_bins=2`
  (production default!), `window_factor=1.0`, `freq_from_tdi_phase=True`;
  `window_alpha` lives on the `STFTFresnel`/group, not on the comp;
  `window_factor` is only applied on the *unwindowed* path (`domains.cu:817`).

---

## 3. Part A — lowering the Fresnel mismatch (ranked)

Ordered by gain/effort. Gains quote the measured budget of §2.1.

### A0. Knobs — no code changes

| change | expected gain | cost |
|---|---|---|
| `n_side_bins`: 2 → 10 / 20 / 25 | truncation floor 1.25e-3 → 5.2e-5 → ~1e-5 (at α=0.1) | linear in stencil: ~22 sincos × (2n+1) × 3 per segment |
| Tukey α: 0.1 → 0.3–0.5 (data + evaluator on the SAME knob) | window leakage floor → ~2e-6 at α=0.5 | per-segment SNR² retention 1−5α/8 (0.94 → 0.69); non-overlapping segments, so this is real signal loss |
| `freq_from_tdi_phase=True` (default ON — keep it) | 9.53e-5 → 2.53e-5 class | ~3× anchor response cost per segment |
| `use_midpoint=True` | 2.09e-5 → 1.48e-5 class | free |
| observation interior to the orbit span (or mask edge segments) | removes the ~60× full-grid edge inflation | run configuration |

**Ceiling of knobs alone: ~1e-5 .. 2e-6** — truncation can be bought down with
`n_side`, but the A&S floor (~2e-6) and the const-envelope error (~7e-6 @ 6 h) remain.

### A1. Precise Fresnel integrals (small, local, highest accuracy/effort)

Replace the A&S 7.3.27/28 rational fits in `get_fresnel_integrals` (`domains.cu:672`)
with a Boersma-class evaluation or the standard power-series (|x| small) +
continued-fraction/asymptotic (|x| large) split, targeting ≤1e-9 per value. One device
function; cost stays in the same class (a handful of polynomial terms + the same
sincos). Removes the ~2e-6 saturation → the windowed Fresnel tracks the exact window
model to the next floor down. *This is the single biggest accuracy win per line of code.*

### A2. Linear-envelope correction (kills the const-envelope error)

The estimator stencil already evaluates the response at t ± D (`z₊`, `z₋`); reuse those
samples to linearize the per-channel amplitude (and residual non-chirp phase) across the
segment: A(τ) ≈ A₀(1 + a·τ). The first-moment integral ∫ τ·e^{iφ(τ)}dτ is analytic —
it is (1/2πi)·∂/∂f of the Fresnel kernel already computed — so the correction is one
extra kernel-derivative term per pixel (~1.3–1.5× pixel cost), no new response
evaluations. Expected: removes the ~7e-6 (6 h) / ~8e-5 (24 h) within-segment modulation
term; the 24 h "grid-aligned residual" trend in the studies is exactly this term.

### A3. Cubic-phase (curvature) perturbative term

From the same ±D stencil a third phase derivative is available (or `fddot` +
Doppler-rate drift analytically). First-order expansion
e^{i(π/3)φ⃛τ³} ≈ 1 + i(π/3)φ⃛τ³ adds a third-moment term (second kernel derivative).
Only relevant for long segments / high f0 — gate it on a per-source estimate of
φ⃛·Δ³. Together, A1+A2(+A3) put the Fresnel *interior* ceiling at ~1e-7.

### A4. Model the whole-span data taper

If the run applies the engine's global Tukey (`window_taper_duration`), the evaluator
silently ignores it (§2.2). Cheapest correct fix inside the current model: multiply each
segment's value by w_glob(t_mid) (exact to the const-envelope order already assumed);
exact alternative: Tukey×Tukey product windows on the few overlap segments. Only needed
if the STFT stream inherits the global taper — decide at the data-pipeline level and
keep the two consistent (the measured cost of an *unmodeled* global taper is ~1e-2).

### A5. Hygiene (document-or-fix)

- `window_factor` acts only on the unwindowed path (`domains.cu:817`) — unify or
  document; it silently does nothing once `window_alpha > 0`.
- Expose `window_alpha` on `STFTGBComputations.__init__` so comp and group can't drift.
- The three copies of the pixel loop (§2.2) should collapse behind one column-producer
  seam — that is P1 of the successor plan (§5) and benefits the Fresnel path too.

### Part A bottom line

| configuration | expected mm |
|---|---|
| today's defaults (n_side=2, α=0.1-ish, tdi-phase anchor) | ~1e-3 |
| A0 knobs (n_side 20–25, α 0.3–0.5, midpoint, interior) | ~1e-5 .. 2e-6 |
| + A1 (precise Fresnel integrals) | ~2e-6 → const-envelope-limited |
| + A2/A3 (envelope + curvature corrections) | **~1e-7 interior** |

Cost trend: every step *adds* work to a per-pixel path that already spends ~22 sincos,
and the pixel count grows linearly with `n_side`. A 1e-6-grade Fresnel at n_side ≳ 20 is
~4000+ sincos per (source, segment). That is the motivation for Part B.

---

## 4. Part B — the successor: the het-FFT column producer

**Recommendation:** replace the per-pixel analytic Fresnel projection with a
**heterodyned, spline-fed, in-register FFT per (source, segment)** — the direct STFT
analog of the WDM chunked-heterodyne, built from primitives that already exist on
`l2d-dev`.

### 4.1 Design

**Per proposal batch (shared, once):**
- Orbit geometry: `OrbitsSplineCache` (`lat_tdi_on_the_fly.hh:64`) filled cooperatively
  via `populate_orbit_spline_cache` (`lat_chunked_het_kernels.hh:358`) — exists.
- Per source: evaluate the TDI response at **N_cp control points** over the source's
  live span with `get_tdi_raw_cached`; de-rotate by a **single global carrier**
  f0g = round(f0/df_stft)·df_stft; fit amp/phase cubic splines
  (`wdm_fit_cubic_spline`, `lat_chunked_het_kernels.hh:309`). This is *literally*
  steps 1–3 of `fast_wdm_inner_heterodyne_spline` (`lat_chunked_het_kernels.hh:483`),
  whose measured envelope-spline accuracy is **mm < 4e-11** for GBs at N_cp=48 per
  chunk. One heterodyne serves ALL segments because the orbital Doppler excursion
  (±~0.4 µHz at a few mHz) is far below one STFT column (df_stft ≈ 46 µHz at 6 h):
  after de-rotation the whole signal is a near-DC envelope with ~1/day bandwidth.
  N_cp scales with the span (~2 nodes/day; 90 d → ~180 nodes — negligible memory).

**Per (source, segment) — one thread, keeping today's kernel mapping:**
1. Evaluate the amp/phase splines at **N_sub uniform subsamples** across the segment —
   polynomial evaluation, zero transcendentals.
2. Multiply by (a) the **precomputed per-segment Tukey window vector** — identical for
   every segment, one shared constant array; the window is now handled *exactly*, for
   any window shape, and the entire 7-term analytic machinery disappears — and (b) the
   sub-bin residual carrier e^{−2πiδf·τ} via a **sincos recurrence** (1 sincos + N_sub
   complex multiplies; the "R1 transcendental-free DFT" trick from the FFT-column spec).
3. **Serial in-register radix-2 FFT of length N_sub** (N_sub = 8–32 complex doubles
   fit comfortably in registers; no shared-memory choreography, no cross-thread
   cooperation — the block layout and reductions of `stft_eval_block_*` are untouched).
   For wide stencils (N_sub ≥ 64), fall back to the block-cooperative
   `wdm_spline_radix2_fft` (`lat_wdm_fft.hh:52`) or the cufftdx path
   (`wdm_fft_dispatch`, :245).
4. Apply the FT-origin phase (segment start t0_seg — same convention
   `get_phase_kernel_product` re-anchors today) and scatter/accumulate the
   ±`n_side_bins` bins around k0 exactly as the Fresnel loop does now (keep-mask,
   0.5·factor fill convention, d_h/h_h reductions all unchanged).

**Sampling rule:** `N_sub ≥ 2·(n_side_bins + margin)` — the aliasing cliff measured in
the FFT-column prototype (`n_sub ≳ 2·n_side+1`, spec §12); midpoint-rule subsampling
was the prototype's key quadrature fix and carries over.

### 4.2 Why this wins on BOTH axes

*Accuracy.* Window: exact (time-domain multiply). Within-segment physics: the spline
carries the TRUE TDI amplitude/phase — orbital Doppler curvature, response modulation,
fddot — so the const-envelope and linear-chirp assumptions vanish entirely, and there
are no Fresnel special functions to approximate. Remaining error terms: envelope-spline
interpolation (measured 4e-11-class at chunked-het densities) and N_sub aliasing
(controlled by the sampling rule). Expected per-column method error: **1e-8-class or
below**, to be measured against the brute STFT; the truncation floor then remains the
only knob (n_side × α), same as for any stencil method.

*Speed.* Per (source, segment):

| | windowed Fresnel (today) | het-FFT producer |
|---|---|---|
| response evaluations | 1–3 direct `get_tdi_Xf_single` (full UCB response each) | 0 (spline lookups) |
| transcendentals | ~22 sincos × (2n_side+1) × 3ch (n_side=2 → ~330; n_side=20 → ~2700) | 1–2 sincos + 6 atan2-free spline evals × N_sub; FFT twiddles via recurrence |
| memory traffic | params + anchor | + spline coefficients (contiguous, per source) |
| GPU shape | thread-per-segment, no cooperation | identical (in-register FFT at N_sub ≤ 32) |

The FFT-column prototype was ~10× *slower* than the Fresnel on CPU precisely because it
re-ran the full direct response N_sub times per segment; the spline + single-heterodyne
removes exactly that term (the prototype spec already identified the response spline as
"proven essential"). The chunked-het WDM path demonstrates the same trade at scale.

### 4.3 Reuse map (all on l2d-dev)

| piece | source | change needed |
|---|---|---|
| control-point response + de-rotation + spline fit | `fast_wdm_inner_heterodyne_spline` steps 1–3 (`lat_chunked_het_kernels.hh:483`), `wdm_fit_cubic_spline` (:309) | factor out / call with STFT carrier f0g |
| orbit spline sharing | `OrbitsSplineCache` + `populate_orbit_spline_cache` | none |
| tiny FFT | new: serial in-register radix-2 (≤32); existing `wdm_spline_radix2_fft` / `wdm_fft_dispatch` for large N_sub | small device helper |
| window vector | per-segment Tukey, one constant array | trivial precompute |
| kernel skeleton, reductions, keep-mask, fill scatter, engine/buffer layers | `stft_*_impl` family, `STFTBandLikelihoodEngine`, `gbbands` | **unchanged** — only the column producer swaps |
| producer seam | `FresnelColumn`/`FFTColumn` template policy, validated byte-identical on `feat-stft-gb` (spec §4) | port the refactor to l2d-dev's 3 sites |

### 4.4 The migration, in the WDM analogy the collaboration already made

| WDM domain | STFT domain |
|---|---|
| lookup table (dense, precomputed, inflexible) | analytic Fresnel projection (transcendental-heavy, window-approximate, fixed model) |
| **chunked heterodyne** (`GBWDMComputations`): sparse response nodes → de-rotate → spline → windowed slow signal → FFT → band extract | **het-FFT column producer** (this proposal): same five steps, per STFT segment, in-register FFT |
| signal-het + polyphase (V2, `GBSignalHetComputations`): per-source *reference* c0 + ratio scoring for in-model repeats | optional later: per-source STFT reference + ratio scoring for the in-model repeat blocks (defer — the LAT `SignalHet*` cutils are still in-flight, and the direct producer is already cheap) |

### 4.5 Open items / risks to validate in the prototype

- **Convention parity**: FT-origin phase, the 0.5·factor fill scaling, `use_midpoint`
  anchoring, and the keep-mask must match the Fresnel path bit-for-bit in intent —
  the byte-identical-Fresnel policy refactor (P1) plus template-vs-template mismatches
  (Fresnel | het-FFT) are the guard.
- **N_cp density vs Tobs and per-source live spans** (RJ births see short spans):
  calibrate the nodes/day rule with the accuracy harness.
- **Segment-edge behaviour**: the spline is global, so edge segments have no special
  treatment — expect the same orbit-edge sensitivities as today (observation-interior
  rule still applies).
- **Grad path**: `get_ll_grad_stft` finite-differences through the same producer —
  no extra work, but re-run the grad gates.
- **Register pressure** at N_sub=32 on the production GPU (fallback: cooperative FFT).

### 4.6 What stays true regardless

The truncation floor (window α × n_side) is a property of the *representation*, not the
producer — the het-FFT producer makes wide stencils affordable but does not remove the
floor. With the window fixed at non-overlapping Tukey, hitting the 1e-6..1e-8 target is
a (α, n_side) budget choice: e.g. α=0.5 + n_side ~ 10–15 → window floor 1.9e-6 with the
producer exact beneath it; pushing to 1e-8 within the Tukey family requires larger α
and/or wider stencils (SNR trade documented in §3/A0) — to be measured with the
mismatch script once the producer exists.

---

## 5. Phased implementation outline (future work)

- **P0 — quick wins (Fresnel kept as-is):** flip run defaults per A0 (`n_side_bins`,
  α on one knob, midpoint); implement A1 (precise `get_fresnel_integrals`). Gates:
  STFT trio (23+2) with updated accuracy expectations; α≥0.5 saturation should drop
  below 1e-6 in `test_stft_gb_accuracy`-style checks.
- **P1 — producer seam:** port the `FresnelColumn` template-policy refactor to
  `l2d-dev`'s three sites (`stft_eval_block_ll`, `stft_eval_block_swap`, inlined fill
  loop) with **byte-identical** Fresnel results (the feature branch validated this
  refactor shape). Rebuild both wheels (copy-compile rule).
- **P2 — het-FFT producer, CPU:** implement §4.1 behind the seam; accuracy gates =
  producer-vs-brute-STFT on the accuracy-test grid (target ≤1e-7 in-stencil interior at
  moderate α), Fresnel-vs-hetFFT template mismatches, trio stays green, engine/flow
  suites untouched.
- **P3 — GPU:** in-register FFT path + bench vs Fresnel at n_side ∈ {2, 10, 25} on the
  production card; `wdm_fft_dispatch` fallback for large N_sub. (The old FFT-column’s
  GPU go/no-go criteria in `GPU_AGENT_BRIEF.md` are superseded by this design — its CPU
  cost analysis no longer applies once the spline+heterodyne is in.)
- **P4 — only if a Fresnel fallback must stay production-grade:** A2/A3 envelope and
  curvature corrections; A4 global-taper handling decided at the pipeline level.

---

## 6. References

- Fresnel path: `lisatools/cutils/lat_stft_kernels.hh` (`stft_eval_block_ll` :275,
  `stft_eval_block_swap` :624, `stft_fill_global_kernel` :464,
  `stft_freq_fdot_from_tdi_phase` :150); `lisatools/cutils/domains.cu`
  (`get_fresnel_integrals` :672 + accuracy note :655, `get_windowed_fourier_value`
  :752, `get_fourier_value` :804); `gbgpu/gbcomps.py` (`STFTGBComputations` :937).
- Reused machinery: `lisatools/cutils/lat_chunked_het_kernels.hh`
  (`fast_wdm_inner_heterodyne_spline` :483, `wdm_fit_cubic_spline` :309,
  `populate_orbit_spline_cache` :358), `lisatools/cutils/lat_wdm_fft.hh`
  (`wdm_spline_radix2_fft` :52, `wdm_fft_dispatch` :245),
  `lisatools/cutils/lat_tdi_on_the_fly.hh` (`OrbitsSplineCache` :64).
- Prior art & measurements: `docs/specs/2026-07-01-stft-gb-fft-per-column-design.md`
  (org_setup root; §12 prototype results), 2026-07 agreement studies
  (`make_stft_fft_fd_agreement_plots.py`, `stft_window_alpha_sweep.py`),
  `GBGPU/tests/test_stft_gb_accuracy.py`,
  `scripts/validation/gb_mojito_stft_fd_mismatch.py` (per-source decomposition tool).
