# Fast sig-het information-matrix kernel — design + test plan

**Date:** 2026-08-04
**Status:** design, for review. Nothing implemented yet.
**Goal:** replace the 46.44 ms/source WDM information matrix with a sig-het
kernel that takes its numerical derivatives *inside* the kernel and reuses the
reference the in-model likelihood block has already built.

---

## Why the current one is slow

`chunked_het.information_matrix` (`chunked_het.py:1223`) assembles
`Gamma_ij` from central differences of `get_swap_ll_wdm`:

```
Gamma_ij = [ar(i+,j+) - ar(i+,j-) - ar(i-,j+) + ar(i-,j-)] / (4 eps_i eps_j)
```

**4 kernel launches per unique (i, j) pair.** At 9 sampled parameters that is
45 pairs -> 180 launches, each rebuilding waveforms from scratch. Measured on
the overnight run: **46.44 ms per cold-chain source** (linear fit, R = 0.999,
over 148 table rebuilds spanning 23 -> 2437 sources).

Its own docstring names the fix as unbuilt:

> A direct dh-buffer variant via `fill_global_wdm` can replace this once its
> inner product is factor-calibrated against `<h|h>`.

---

## The design

### D1 — contract derivatives, do NOT second-difference the likelihood

Two routes to an information matrix:

| | expression | waveform builds | definiteness |
|---|---|---|---|
| observed info | `-d_i d_j lnL = <d_i h\|d_j h> - <r\|d_i d_j h>` | ~180 (4/pair) | **can be indefinite** |
| Fisher (Gram) | `Gamma_ij = <d_i h\|d_j h>` | **18** (2/param) | **PSD by construction** |

The Gram form is both cheaper and the only one safe for a proposal
covariance — `_proposal_cholesky` needs a Cholesky factor, and an indefinite
observed-information matrix has none. The two differ only by a term whose
expectation vanishes at the peak.

So: build `d_i h = [h(theta + eps_i) - h(theta - eps_i)] / (2 eps_i)` — the
"two waveforms, subtract" step — and contract pairwise. **Never form the
second derivative of the likelihood.**

### D2 — contract PER PIXEL (the memory constraint)

Storing all `nparams` derivative blocks is impossible in shared memory:
`dr_sparse` is `nch*M*N_sparse_t` complex = **240 B per sparse point**, so 9 of
them at `N_sparse_t = 204` is ~440 KB against A100's 163 KB.

Instead, **loop sparse pixels and contract as you go**:

```
for each sparse pixel p:                       # the v4 scorer already does this
    for i in 0..nparams-1:                     # 2 evaluations each
        d[i] = (h_plus_i[p] - h_minus_i[p]) / (2 eps_i)    # nch complex, registers
    for i <= j:
        Gamma[i][j] += Re{ conj(d[i]) . invC[p] . d[j] }
```

Register cost is `nch * nparams` complex — trivial. `Gamma` is
`nparams*(nparams+1)/2 = 45` doubles per block in shared. **The
`N_sparse_t` scaling that pins v4 does not appear at all**, so this kernel has
no equivalent of the two-year wall.

### D3 — reuse the reference, do not rebuild it

The reference stash (`A0/A1/B0/B1/B0nc/B1nc` + `c0_sparse_all`) is built once
per in-model block by `setup_in_model` and is **candidate-independent**. The
existing gradient path already relies on exactly this:

> Each call regenerates X_het via `gb_run_fd_wave_tdi_wrap` so the FD reflects
> the perturbed params; **the shared bin-fold A0/A1/B0/B1 are reused across
> all perturbations.** (`gb_tdi_on_the_fly.cu:3713-3715`)

So the info-matrix kernel takes the same pointer arguments as
`gb_signal_het_v4_get_ll` and adds `param_eps` + `gamma_out`. **No new
reference machinery.** This is the "one after another" reuse in the request:
call it immediately after `setup_in_model`, before the repeat loop.

### D4 — module layout (reuse, don't duplicate)

```
gb_tdi_on_the_fly.cu
  gb_signal_het_v4_score_one_source(...)          EXISTING  (device, per candidate)
  sighet_v4_build_at_params(...)                  NEW       (device) -- the
      per-pixel candidate build LIFTED OUT of the scorer so both the scorer
      and the info-mat kernel call it. Pure refactor, no behaviour change.
  gb_signal_het_v4_infomat_kernel(...)            NEW       (global)
  GBComputationGroup::gb_signal_het_v4_infomat_wrap(...)     NEW  (host)

gbsignalhetcomputations.py
  GBSignalHetComputations.information_matrix(...)  NEW -- same signature as
      chunked_het.information_matrix so it is a DROP-IN; falls through to
      self.chunked.information_matrix when the kernel is unavailable.
```

The lift in step 2 is the important modularity move: the scorer and the
info-mat kernel must build candidates through **one** function, or they will
drift apart the way the v3/v4/v5 scorers already have.

---

## Adjustable knobs (all env, all live for tomorrow)

| knob | default | what it does |
|---|---|---|
| `SIGHET_INFOMAT` | `0` | master switch; `0` = current chunked-het path |
| `SIGHET_INFOMAT_EPS_SCALE` | `1.0` | scales the whole `param_eps` table |
| `SIGHET_INFOMAT_NPARAMS` | all | freeze trailing params (`eps<=0` freezes a dim, the existing gradient convention) |
| `SIGHET_INFOMAT_ONESIDED` | `0` | `1` = forward difference (9 builds not 18), accuracy check |
| `SIGHET_INFOMAT_VALIDATE` | `0` | compute BOTH paths and log max reldiff per source |

`SIGHET_INFOMAT_VALIDATE=1` is the gate that matters — it is how we find out
whether the kernel is right without trusting it.

---

## Cost model

Per source, `nch=3`, `nparams=9`, sig-het v4 at 14.6 us/candidate (measured,
Tobs-flat):

```
current chunked-het   180 launches, full waveform rebuild each   46.44 ms
sig-het infomat       18 perturbed builds, per-pixel contraction  ~0.26 ms
                                                                  ~175x
```

Consequences if it holds:

* the per-proposal table (2438 sources) goes **115.6 s -> 0.6 s**;
* the per-block scheme becomes affordable: 120 sources -> ~30 ms, so
  **each in-model block can build its own exact matrices** instead of
  borrowing nearest-in-frequency — which removes the documented
  "a loud neighbour hands a quiet source a badly-sized jump" defect
  (`gbspecialstretch.py:3290-3298`) entirely;
* the table-staleness problem disappears with it (no table).

---

## Validation ladder (in order, tomorrow)

1. **CPU parity, one source.** `SIGHET_INFOMAT_VALIDATE=1` against
   `chunked_het.information_matrix`. Target: max reldiff < 1e-6 on every
   entry. This is the factor-calibration the chunked-het docstring warns
   about — a wrong normalization shows up here as a constant ratio.
2. **Eigenvalue check.** All eigenvalues > 0 on every source (the Gram form
   guarantees it; a violation means an indexing bug).
3. **eps sensitivity.** `SIGHET_INFOMAT_EPS_SCALE` in `{0.5, 1, 2}` — the
   matrix must be stable to a few 1e-3. If it moves, the step table is wrong
   for this basis, not the kernel.
4. **GPU parity** vs CPU, same source set.
5. **Timing** at `nparams=9`, batch 1 / 128 / 2048 — confirm the ~175x and
   that it is batch-efficient.
6. **Acceptance-rate A/B** in a short live run: borrowed-table vs per-block
   exact. This is the number that decides whether it is worth adopting, and
   **the code logs no acceptance rates today — that instrumentation has to
   land first.**

Gate 1 is the blocker. Everything else is measurement.

---

## Risks

* **Normalization.** The single most likely failure, and precisely what the
  chunked-het docstring flags. Gate 1 catches it.
* **`param_eps` in the 9-col distance basis.** The existing table is
  GB-oriented for the lnA/fdot basis; slot 0 is now `dist` and slot 2 `Mc`,
  with a 9th `fdot_astro_ratio`. The steps must be re-derived for the run
  basis or gate 3 will fail.
* **Register pressure.** `nch*nparams` complex in registers may spill at
  `nparams=9`, cutting occupancy. Measure at gate 5; if it spills, tile the
  parameter loop (contract in two passes of 5+4) at the cost of a second
  pixel sweep.
* **Scope discipline.** The refactor in D4 step 2 touches the *existing*
  scorer. It must be a pure lift with a before/after parity check on
  `get_ll`, or it risks the validated v4 path.
