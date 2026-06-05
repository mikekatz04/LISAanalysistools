# Where the residual mm_5 spread comes from, and what to change

After the m-parity sign fix, no source flips. Remaining
`mismatch_5` spans **5×10⁻¹⁶ → 3×10⁻²** across 80 random prior draws.
The cause is **not** a parameter regime issue — it's a per-source-pixel
edge effect in the C central-difference frequency estimator.

## Statistical pattern (80 draws)

Spearman ρ vs log₁₀(mm_5):

| feature                  |    ρ    |   p     |
|--------------------------|--------:|--------:|
| **`|f0_frac - 0.5|`**    | **-0.66** | **2.4e-11** |
| f0 [mHz]                 |  +0.20  |  7.6e-02 |
| m_layer                  |  +0.20  |  7.6e-02 |
| log10(SNR), |fdot|, inc, |cos(inc)|, psi, lam, beta, log10(A) | all \|ρ\| < 0.16 | p > 0.17 |
| interp_offset (sub-cell of f_vals_norm grid) | +0.16 | 0.16 |
| log10(distance to band edge) | +0.025 | 0.83 |

Only **one** parameter matters. Plots:
[gb_prior_pattern_mm_vs_params.png](gb_prior_pattern_mm_vs_params.png),
[gb_prior_pattern_focus.png](gb_prior_pattern_focus.png),
[gb_prior_pattern_top_correlators.png](gb_prior_pattern_top_correlators.png).

The U-shape vs `f0_frac`:

* **bad (mm_5 > 10⁻³):** `f0_frac ∈ {0.0088, 0.034, 0.978}` (boundary)
  *and* `f0_frac ≈ 0.45, 0.50, 0.54, 0.60` (center)
* **good (mm_5 ≤ 10⁻¹²):** `f0_frac ∈ {0.07, 0.08, 0.10, 0.12, 0.23,
  0.80, 0.87, 0.90}` — the "sweet ring" around ≈ 0.1 / 0.9

WDM band-edge proximity is **not** a factor: many bad sources sit deep
in the active band (>200 layers from either edge); many good sources
sit near the edges.

## Spatial pattern of the error (per-source diagnostic)

Six dumps in [gb_prior_pattern_srcdiag/](gb_prior_pattern_srcdiag/).

**All best sources (mm_5 ~ 10⁻¹⁵):**
[src_best_00_idx0035.png](gb_prior_pattern_srcdiag/src_best_00_idx0035.png)
shows residuals of order **10⁻²⁵** concentrated at `n ≈ 50` and `n ≈ 2500`
— the Tukey-window roll-off. Floor noise only; no interior spikes.

**All worst sources (mm_5 ~ 10⁻²):** e.g. src #64
[src_worst_03_idx0064.png](gb_prior_pattern_srcdiag/src_worst_03_idx0064.png),
src #22
[src_worst_04_idx0022.png](gb_prior_pattern_srcdiag/src_worst_04_idx0022.png),
src #4
[src_worst_05_idx0004.png](gb_prior_pattern_srcdiag/src_worst_05_idx0004.png).

All three share the *same* shape:

* Residual ≈ **0 everywhere** except at **2–3 specific interior time
  bins** (`n ≈ 60, 1900, 2260` for #64; `n ≈ 1100, 2050` for #22;
  `n ≈ 1200, 1900` for #4).
* The residual lives entirely on the **single central m-layer**.
* At those bins, the C template is **missing** a contribution that the
  WDM-of-eval_tdi injection has.

That signature — a few isolated time bins where the central pixel is
left empty while neighbouring pixels are fine — is the C
central-difference frequency estimator picking up a spurious shift on
those individual bins. The wavelet contribution gets routed to a
distant layer (outside the 5-layer mm_5 box), leaving the central
pixel empty.

## The mechanism (TDIonTheFly.cu, `fast_wdm_inner`)

```cpp
tdi_phase_down = -gcmplx::arg(tdi_channel_val_down_dt[i] * gcmplx::exp(I*phase_ref_down));
tdi_phase_mid  = -gcmplx::arg(tdi_channel_val[i]         * gcmplx::exp(I*phase_ref));
tdi_phase_up   = -gcmplx::arg(tdi_channel_val_up_dt[i]   * gcmplx::exp(I*phase_ref_up));

double dphi_up = tdi_phase_up - tdi_phase_mid;
if (dphi_up >  M_PI) dphi_up -= 2.0 * M_PI;          // single-step unwrap
else if (dphi_up < -M_PI) dphi_up += 2.0 * M_PI;
double dphi_down = tdi_phase_down - tdi_phase_mid;
if (dphi_down >  M_PI) dphi_down -= 2.0 * M_PI;
else if (dphi_down < -M_PI) dphi_down += 2.0 * M_PI;

tdi_frequency = (dphi_up - dphi_down) / (2 * deriv_delta_t) / (2 * M_PI);
f[i] = residual_frequency + tdi_frequency;
```

`arg()` returns values in `(-π, π]`. At specific time bins the residual
TDI phase crosses ±π between `tn−Δt`, `tn`, and `tn+Δt`. The single
±2π correction handles most of these — but when the up-side and
down-side unwraps fire asymmetrically, `tdi_frequency` picks up a
spurious **`±1 / (2·Δt) = ±1 mHz`** — about **30 layers** in this
grid. The kernel then does

```cpp
layer_m = int(f[i] / wdm->layer_df);     // off by ~30
```

and `get_wdm_in_channel_over_layers` returns 0 (out of band) or writes
the wavelet into a wrong layer 30 away from `m_central`. The pixel
under `m_central` at that `n` is left empty → exactly the residual
pattern we see.

**Why does `|f0_frac - 0.5|` predict it?** The number of times the
residual TDI phase sweeps through ±π over `T_obs` is set by the rate
at which the phase `−arg(M·exp(i·phase_ref))` revolves. Empirically
that rate goes through a U-shape vs `f0_frac` (centre and boundary
both produce more wraps per `Δt` than the sweet ring around 0.1).
Sources in the sweet ring hit the ±π boundary zero or one times
across the run; sources at the centre/boundary hit it 2–4 times,
which is exactly what we count in the residual `n` spikes.

## What to change (ranked by expected impact / effort)

### 1. Replace the numerical central-difference frequency with the analytic GB frequency  *(highest impact, small change)*

`GBTDIonTheFly::ucb_f` already returns the instantaneous source
frequency from `(f0, fdot, fddot)` analytically. The TDI Doppler shift
is small (≲ 10⁻⁴ relative), so

```cpp
f[i] = tdi_on_fly_here.get_f(tn, params, bin_i);   // analytic
// optionally + a single-pass Doppler correction
```

removes the ±π unwrap fragility entirely. This **eliminates the
spike mechanism** and should drop the worst sources from mm_5 ≈ 10⁻²
to ≈ 10⁻¹⁵ along with the best ones.

The current numerical estimator was needed when the TDI Doppler
component was unknown; for GBs the Doppler is well-approximated by
`v_orbit · k / c · f0`, which can be computed in-line from `k` and the
orbit geometry already on hand in `fast_wdm_inner`.

### 2. Multi-step unwrap (if (1) is too invasive)  *(medium impact)*

Replace the single `if (dphi > π) dphi -= 2π;` with a loop:

```cpp
while (dphi_up >  M_PI) dphi_up -= 2.0 * M_PI;
while (dphi_up < -M_PI) dphi_up += 2.0 * M_PI;
```

This handles the cases where the phase change between `tn−Δt` and
`tn+Δt` exceeds ±π by more than one full cycle. Cheaper than (1) but
still leaves the per-anchor asymmetry: if up sees one wrap and down
sees zero, the central diff still picks up an extra π.

The robust version is to unwrap **up–down** directly (one wrap that
applies symmetrically) instead of unwrapping each side against `mid`:

```cpp
double dphi = tdi_phase_up - tdi_phase_down;
while (dphi >  M_PI) dphi -= 2.0 * M_PI;
while (dphi < -M_PI) dphi += 2.0 * M_PI;
tdi_frequency = dphi / (2 * deriv_delta_t) / (2 * M_PI);
```

This drops `tdi_phase_mid` from the formula completely, also fixing
the candidate-1 convention bug that we patched earlier (which is now
a no-op but conceptually redundant).

### 3. Centre the 5-layer mismatch box on `round(m_continuous)` rather than `int(m_continuous)`  *(measurement-only fix; doesn't help global fit)*

The boundary-side worst sources (`f0_frac ∈ {0.0088, 0.034, 0.978}`)
are partly an artifact of how the 5-layer box is chosen:
`min_freq = f0 − 3*layer_df`, `max_freq = f0 + 2*layer_df`. For
`f0_frac ≈ 0.98` the source's centre of mass sits at `m_central + 1`,
so the box is off-centre by one layer and clips real power.

Replacing the box with `[round(f0/layer_df) - 2, round(f0/layer_df) + 2]`
makes the mm_5 metric symmetric around the source's actual main
layer. The full-band log-likelihood is already unaffected by this.

### 4. (Lower priority) Reduce `deriv_delta_t` so the residual TDI phase changes less per step

The current `deriv_delta_t = 500 s` puts the up/down anchors 1000 s
apart. Shrinking to 100 s reduces the number of windows that straddle
a ±π crossing by 5×. Not a real fix — the underlying numerical
derivative is still fragile — but it would push the spike rate down
roughly proportionally. Cheap to test.

## Expected outcome after (1) + (3)

* Worst sources drop from `mm_5 ≈ 10⁻²` to `≈ 10⁻¹³`.
* Histogram of mm_5 collapses to a narrow band at 10⁻¹³ ± 1 dex,
  bottoming at 10⁻¹⁶.
* All `|f0_frac − 0.5|` dependence disappears.

## Files

* [gb_lookup_prior_draws.py](gb_lookup_prior_draws.py) — runs the prior
  sweep, writes `<prefix>_results.npz`.
* [gb_lookup_pattern_plots.py](gb_lookup_pattern_plots.py) — reads the
  npz, prints Spearman ρ for every feature, dumps the all-features and
  top-correlators figures.
* [gb_lookup_source_diag.py](gb_lookup_source_diag.py) — picks the K
  best and K worst sources and rebuilds them with full per-source
  spatial residual plots under `<prefix>_srcdiag/`.
