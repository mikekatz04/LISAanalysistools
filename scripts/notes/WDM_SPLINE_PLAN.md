# Plan — spline-based replacement for `fast_wdm_inner`

## Why

`fast_wdm_inner` currently does, for every WDM time bin `n` of every
source:

1. `get_tdi_Xf_single` at `t_n`, `t_n ± Δt` (3 calls — the most
   expensive thing per pixel).
2. `phase_ref = get_phase_ref(t_n)` analytic (cheap).
3. `tdi_phase = -arg(M·exp(i·phase_ref))` at the three anchors plus a
   single-step `±π` unwrap, then `tdi_frequency = central diff`.

Problems we have already located:

* The per-pixel `f[i]` jitters by ≲ 10⁻⁶ Hz around the analytic value,
  and at sources right at a layer boundary that jitter walks
  `f_scaled = f − (m+m_diff)·layer_df` across a **column boundary in
  the lookup table**, where `linear_interp` mixes values that differ
  by ~10¹⁴ — producing the per-source spikes in `gb_prior_pattern_worst3_residuals.png`.
* Three `get_tdi_Xf_single` evaluations per pixel is the dominant cost
  for the per-source kernel.

## Replacement

Mirror the Python `GBLookupWaveWrap` approach: build a **cubic spline**
of the smooth scalar fields `(tdi_amp_c, tdi_phase_c, phase_ref)` over
a coarse set of knots, then evaluate the spline analytically at every
WDM time bin. Concretely:

* **`N_KNOT` knots, evenly spaced** across the active WDM time range,
  with knot spacing `dt_knot ≫ layer_dt`. Default: `N_KNOT = 128` over
  `Nt_active ≈ 2520`, giving `dt_knot ≈ 20 · layer_dt`.
* Each knot is filled by calling `LISATDIonTheFly::get_tdi` (the
  amplitude/phase-extracted version with sign-flip tracking and
  unwrap) — **not** `get_tdi_Xf_single`. This is the same routine
  that the Python spline builder uses, so it inherits the sign
  tracker and the multi-cycle phase unwrap that `arg()` does *not*
  provide.
* The cubic on each segment is **Hermite (Catmull-Rom style)** — local
  construction, no tridiagonal solve. Per knot we estimate the slope
  with a centered finite difference of the knot values. This keeps
  the build O(N_KNOT) and embarrassingly parallel across knots.
* Spline storage = `7 × N_KNOT` doubles per source (3 amp, 3
  tdi_phase, 1 phase_ref). At `N_KNOT = 128` that's 7 KB per source,
  in **global memory** — the user's constraint that the whole spline
  not be required to fit in shared memory.
* At each WDM pixel:
  * `amp_c(t_n)`, `tdi_phase_c(t_n)`, `phase_ref(t_n)` from spline
    value;
  * `f[c] = (d/dt)(tdi_phase_c + phase_ref) / (2π)` from the
    analytic spline derivative — **no central differencing, no `arg()`
    wrap**;
  * Complex `M_c = amp_c · exp(-i · (tdi_phase_c + phase_ref))`;
  * Apply the same `conj(M·exp(-iπ/2))` rotation `get_w_mn_lookup`
    consumes, and return.

## What this fixes

* **Per-pixel `f[i]` jitter and column-boundary spikes**: gone. `f`
  becomes a smooth cubic, no ±π unwrap chains to misfire.
* **Cost**: drops from 3·Nt_active `get_tdi_Xf_single` calls per source
  to `N_KNOT` `get_tdi_Xf_single` calls (plus a one-time
  `new_extract_amplitude_and_phase` + `new_unwrap_phase` over those
  N_KNOT knots). For Nt_active=2520, N_KNOT=128 that's a ~60× drop.
* **Sign tracker**: the spline build runs the same
  `new_extract_amplitude_and_phase` the Python side does, so the C
  template stops disagreeing with the Python injection on the few
  sources where the sign tracker matters.

## Memory layout (per source)

```
double t_knots [N_KNOT]              # 1 KB at N_KNOT=128
double y      [7 * N_KNOT]           # 7 KB at N_KNOT=128
double m      [7 * N_KNOT]           # 7 KB at N_KNOT=128  (slopes at knots)
double y_lookup_workspace [3 * N_KNOT]    # only during build
```

`y` is laid out `(var, knot)` with var indices

```
0 = tdi_amp   chan 0      4 = tdi_phase chan 1
1 = tdi_amp   chan 1      5 = tdi_phase chan 2
2 = tdi_amp   chan 2      6 = phase_ref
3 = tdi_phase chan 0
```

All buffers are caller-provided global-memory pointers. The block-per-
source kernel allocates them once at block entry (with `cudaMalloc` /
host malloc in the wrapper around the kernel call, since per-block
shared memory cannot hold them).

## New device functions (CPU + CUDA)

```cpp
// Append to TDIonTheFly.{hh,cu} — declared CUDA_DEVICE so the same
// implementation runs in the CPU and the CUDA build.

// 1. Build the spline coefficients for one source.
CUDA_DEVICE
void build_wdm_spline_for_source(
    GBTDIonTheFly &tdi_on_fly,
    double *params, int bin_i,
    int *link_Space_craft_rec, int *link_Space_craft_em,
    double t_lo, double t_hi, int N_knot,
    /* outputs */
    double *t_knots,                 // (N_knot,)
    double *y,                       // (7, N_knot) — values
    double *m,                       // (7, N_knot) — slopes
    /* workspace */
    cmplx *M_buf,                    // (3 * N_knot)
    void  *get_tdi_buf, int get_tdi_buf_len);

// 2. Evaluate the spline at one WDM time bin tn for one source.
CUDA_DEVICE
void eval_wdm_spline(
    double tn,
    const double *t_knots, int N_knot,
    const double *y, const double *m,
    /* outputs */
    double *amp_out,                 // (3,)
    double *tdi_phase_out,           // (3,)
    double *phase_ref_out,           // scalar
    double *tdi_phase_d_out,         // (3,)  d/dt
    double *phase_ref_d_out);        // d/dt

// 3. Drop-in replacement for fast_wdm_inner. Same signature except it
//    takes spline buffers in place of the params-based on-the-fly
//    evaluation.
CUDA_DEVICE
void fast_wdm_inner_spline(
    cmplx *tdi_channel_val,          // (3,) — rotated, ready for lookup
    double *f, double *fdot,         // (3,) — analytic from spline
    double tn,
    const double *t_knots, int N_knot,
    const double *y, const double *m);
```

## Wire-up (not in this PR — separate change)

The two callers that need to switch over are
`gb_wdm_fill_global_kernel` and `gb_wdm_get_ll_kernel`. The new flow
inside each is:

```cpp
for each bin_i:
    build_wdm_spline_for_source(...)        // once per source
    for each n in active band:
        fast_wdm_inner_spline(...)
        // existing layer_m, m_diff loop, get_w_mn_lookup, atomicAdd
```

`gb_wdm_swap_ll_kernel` builds two splines (add + remove) and proceeds
similarly.

## Limits / things to validate

* **Knot spacing**: `dt_knot = (t_hi - t_lo) / (N_knot - 1)`. For
  `tdi_amp`/`tdi_phase` the dominant variation is the LISA orbital
  modulation (~1 year period). 128 knots over Tobs gives knots every
  ≈ 1.4 days — well-sampled. We should add an env-var override and
  scan it on the same 80-draw prior run.
* **Hermite slope estimate** at boundaries: forward/backward difference
  at endpoints; centered everywhere else.
* The sign-flip tracker in `new_extract_amplitude_and_phase` was a
  candidate-4 in `SIGN_FLIP_NOTES.md` — by reusing it identically for
  Python and C, we eliminate the source of disagreement.
* The amp spline can go through zero (a sign-tracker zero-cross point).
  Because we store `tdi_amp` as the signed value the extractor
  produces (and `tdi_phase` carries the matching `+π` offset), the
  cubic interpolation is fine on both sides — the sign discontinuity
  is *not* in the stored data, only in the principal-branch arg of
  the underlying complex M.
