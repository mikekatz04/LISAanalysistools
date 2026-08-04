# The `rj_replace` ll defect is a non-attainable phase-maximised score

**Status:** root-caused locally, fix proposed, not implemented.
**Date:** 2026-08-03
**Repro:** `scripts/gb/gb_search_rj_cycle_smoke.py` with `GB_DEBUG=1` (CPU,
one synthetic GB, 3-layer band, 2 walkers x 2 temps).

## Symptom

`rj_replace` tracked ll changes disagree with a direct recompute:

| quantity | GPU (13 bands, catalogue) | local (1 band, 1 synthetic source) |
|---|---|---|
| old-side delta vs `get_add_ll` | 0.000e+00 | 0.000e+00 |
| new-side delta vs `get_add_ll` | 4.4e-2, 5.2e-2 | 4.3e-2 .. 1.5e-1 |
| tracked vs direct swap dll | 2.2, 3.0 | 1.1 .. 92.7 |
| parent-ll reconcile | 150, 889 | 4.8e-3 .. 3.0e-2 |

The new-side error reproduces with a single isolated source, so it is not a
crowding effect. The parent-ll reconcile discrepancy does **not** reproduce
locally -- that is a separate issue (see "Still open").

## What was ruled out

Each of these was tested, not argued.

1. **The phi0 write-back.** Re-maximising on the final parameters returns a
   residual rotation of `~1e-11 rad`. The write-back
   (`params_new[:, phi0_col] -= phase_angle`, `gbspecialstretch.py:2871`)
   lands the template exactly on its own maximum. Sign and convention correct.

2. **Batch composition.** `get_replace_ll` scores `[old; new]` concatenated on
   a shared `data_index`, and `compute_layer_groups`
   (`wdm_het.py:302-316`) builds each group's m-band as a *union* --
   `m_hi` comes from `sorted_m[j-1]`, the last member's carrier, not each
   row's own. So in principle a source's inner product depends on its
   batch-mates. Measured directly:

   ```
   cat[old;new] new-half vs solo:   0.000e+00
   cat[new;new] first-half vs solo: 0.000e+00
   cat[new;new] halves consistent:  0.000e+00
   ```

   Not the cause here. **The union-window behaviour is still real** and worth
   a separate audit for crowded bands -- it means an inner product is not a
   pure function of its row.

3. **Engine self-consistency.** Re-scoring the exact pre-write-back rows
   reproduces `get_replace_ll`'s value to `0.000e+00`. The engine returns
   what it was asked for.

## Root cause

The two-quadrature phase maximisation returns a value that is **not
invariant to phi0**:

```
PRE-write-back max vs scored:  0.000e+00   (all rounds)
PRE max vs POST max:           3.7e-2, 2.9e-1, 2.0e-1, 1.0e-1, 7.6e-2
```

In exact arithmetic this is impossible. For a GB,
`h = a(t) cos(Phi) - b(t) sin(Phi) = Re[H e^{i Phi}]` with `H = a + i b`
independent of phi0, so `h(phi0 + pi/2) = Re[i H e^{i Phi}]` is an exact
pi/2 rotation of the analytic signal. Then

```
<r|h(phi0)> = Re[ e^{i phi0} <r| H e^{i Phi_0}> ]
```

is exactly sinusoidal in phi0, and `|D| = |d_h_0 + i d_h_90|` is a constant.
`_get_ll_phase_max` (`GBGPU/src/gbgpu/gb_likelihood.py:105-149`) is built on
that identity.

The identity is broken by the **narrow-band m-window truncation**. WDM
wavelet coefficients of a *real* signal carry a parity-dependent phase, so
the fraction of a source's power falling inside the retained
`group_band_layers`-wide m-band is itself phase-dependent. Rotating phi0
redistributes power across the truncation boundary. `<r|h(phi0)>` is
therefore only approximately sinusoidal, and the analytic `|D|` is an
*upper bound* on a curve it no longer exactly describes -- it can exceed the
maximum actually attainable at any phi0.

Consequence: the move scores the proposal at an inflated `delta_new`, then
applies a template whose true delta is lower. Every accepted replacement
mis-reports its ll change by that gap, and the ledger drifts up.

### It is structural, not a discretisation artifact

Re-run at 8x the sparse-grid resolution and 6x the spline control points
(`CHUNKED_N_SPARSE=512 CHUNKED_N_CP_SIG=96 CHUNKED_N_CP_ORBIT=64` vs the
lite preset's `64 / 16 / 16`):

Full samples of the per-round `rj_replace` new-side relative error (scored
`delta_new` vs `get_add_ll` at the applied phi0), which is the tight
measurement of the ledger error:

| preset | n | min | median | max |
|---|---|---|---|---|
| lite (`64 / 16 / 16`)  | 8 | 1.77e-2 | **1.31e-1** | 4.12e-1 |
| fine (`512 / 96 / 64`) | 9 | 4.21e-3 | **1.28e-1** | 4.85e-1 |

And the phi0-invariance gap at fine resolution, 9 rounds:
`2.85e-1, 1.37e-1, 4.20e-3, 1.04e-1, 3.27e-1, 1.14e-1, 1.76e-2, 7.38e-3,
1.13e-1` (median ~1.1e-1) against the lite run's median ~1.0e-1.

**The distributions are indistinguishable.** 8x the sparse grid and 6x the
spline control points move nothing. Time-resolution is not the lever, which
is exactly what the m-band-truncation explanation predicts -- the truncation
is a *frequency-window* effect, independent of `N_sparse` / `N_cp_sig`.

### The scored value is wrong in BOTH directions -- it is scatter, not bias

Scanning phi0 over 24 points, taking the best actual-phase add-delta, and
subtracting it from the scored maximum (8 rounds, `max / median` per round):

```
+1.517e+01 / +4.118e-01
+1.749e+02 / +8.341e+01
-4.209e-01 / -1.363e+00     <- scan beat the analytic "maximum"
+1.414e+01 / +2.119e-02
+1.459e+02 / +1.693e+01
+7.006e-01 / +3.347e-01
-1.426e+00 / -1.426e+00     <-
-1.262e-01 / -2.792e-01     <-
```

**5 positive, 3 negative.** The analytic maximum is not a consistent upper
bound. This is the expected signature of a *non-sinusoidal*
`<r|h(phi0)>`: the two-quadrature formula fits a sinusoid through two
samples of a curve that is not one, so its extremum can land above the true
maximum (unattainable) or below it (the write-back then converges to a
local maximum that is not global -- note the residual rotation is ~1e-11
rad, i.e. a genuine stationary point, in every case).

An earlier draft of this document claimed the sign was always positive.
That was read off the first two rounds before the run completed and is
wrong. The consequence matters: the ledger error is approximately
**zero-mean scatter, not systematic inflation**, so tracked ll drifts as a
random walk (~sqrt(N_accepts)) rather than accumulating linearly.

The tight measurement of the per-replacement error remains the direct one
-- scored `delta_new` vs `get_add_ll` at the applied phi0 -- which is
**4.3e-2 to 1.5e-1 relative**, consistently.

### Relation to the GPU-only reconcile gap

Per-replacement errors of order 1e0..1e2 in ll units can plausibly random-
walk to the observed GPU reconcile discrepancies (150, 889) over many
bands x many accepts, while staying at ~1e-2 in a 2x2 smoke that accepts
only a handful. But diffusion is a weaker accumulator than bias, so this is
now a *candidate* rather than a likely full explanation. The overlapping-
slab hypothesis is not displaced and the slab sweep should still be run.

## Why this is a shared defect, not a replace defect

The same two-quadrature maximisation feeds RJ births. `RJ birth delta vs
get_add_ll` measures **1.4e-1 to 2.4e-1** both locally and on GPU, against
6e-4..9e-3 recorded historically. Births and replaces are one bug.

The old side is exact precisely *because* it already avoids the maximised
value: `get_replace_ll` reads `non_marg_d_h` (the un-maximised `d_h_0`) for
`delta_old_actual` (`gbbands.py:1836-1841`). The new side has no equivalent.

## Proposed fix (not implemented)

Mirror on the new side what the old side already does: **book the ledger at
the actual applied phase, not the analytic maximum.**

After the phi0 write-back in `_run_replace_step`, re-score the new rows with
one non-maximised `get_add_ll` on the still-exposed residual, and use that
value for `delta_new` in both the ledger and the MH ratio. The maximised
value stays as the *proposal* statistic (it is what makes the move find
sources); it is simply not what gets booked.

This adds one `n`-row kernel call to a step that already runs `4n` rows
(two quadratures over `[old; new]`) -- about +25%.

Crucially this **reuses `get_add_ll` and reimplements no likelihood**, per
the standing rule in `feedback_never_rewrite_likelihood_infra`. It is the
same shape as the existing `delta_old_actual` path.

The identical treatment is needed for the RJ birth path.

### Alternatives considered

* *Widen `group_band_layers`* -- reduces the truncation and hence the gap,
  but does not eliminate it, and costs global-memory reads on every kernel.
  Worth measuring as a knob; not a fix.
* *Iterate the maximisation* -- re-maximise at the new phi0 until it
  converges. Converges to the true maximum but costs several extra passes
  and still needs the actual-phase value for the ledger.

## Still open

1. **The parent-ll reconcile discrepancy** (GPU 150 / 889; local 4.8e-3).
   Does not reproduce with one isolated source, so it is plausibly the
   overlapping-slab / edge-leakage bookkeeping. Note
   `band_slab_Nf = band_span + 6` (leakage 2 + guard 1 per side,
   `gbbands.py:80,931-938`) while the band parity stride defaults to 2
   (`gbspecialstretch.py:728-730`) -- with `band_slab_Nf = 7` observed in
   production, simultaneously-updated bands still hold **overlapping copies
   of the same WDM layers**, and an update in one cell does not propagate to
   the other. Decisive test is a one-variable sweep of
   `GB_WDM_BAND_SLAB_LAYERS` / `GB_WDM_SLAB_GUARD_LAYERS`
   (`stock/erebor/gb.py:229-234`) on real data.
2. `GB_SUBTRACT_BUFFER_LAYERS` defaults to **8** while the per-cell slab halo
   is 3 layers. One of the two is wrong; see the TODO in `recipe.py`.

## Correction to an earlier claim

`GB_DEBUG=1` does **not** degrade the chunked-het settings. It only sets the
move's debug flag (`recipe.py:2361`). The coarse
`Nt_sub=64 N_sparse=64 N_cp_sig=16` seen in the smoke comes from
`erebor.gb_no_fg(lite=True)`. Debug runs on a production config are
representative.
