# F-stat 4→2 basis-filter fold (chunked-WDM kernel)

**Status: SPEC ONLY — nothing implemented.** Written 2026-08-28.

## Why

`rj_fstat_centers` is the single largest line item in the GB block:
**830 s of a ~2140 s iteration (~39%)**, measured in snapshot 13
(`[GB_TIMING rj_fstat_search] rj_fstat_centers=830.202s` at cluster
11:24:36, in a propose with `untracked=0.276s` — i.e. no epoch refit, so
this is genuinely the per-propose center cost and not the once-per-epoch
`fstat_grid_fit`).

Two prior attempts to reduce it failed for understood reasons:

- The **unit-open cache** (86ed9353) was a wash: measured 725-743 s
  against a 713-799 s pre-fix band. There was no recomputation to
  dedupe — the precompute row count ≈ the picked row count at an
  identical 0.667 ms/row.
- **Drawing (iota, psi)** instead of maximizing (analysed 2026-08-28)
  is a net throughput *loss*: the recovered fraction is a qubit fidelity
  `R = ½(1 + n̂·n̂_true)`, exactly symmetric about ½ and strongly
  bimodal (q25 = 0.077), which kills 43.7% of currently-SNR-8-passing
  candidates. See [[fstat-centers-redesign-scope]].

This fold is different: it is **exact**. Zero physics loss, no sampling
semantics touched, no proposal-quality question. It is the first
genuinely lossless win identified on this cost.

## The mathematics

The chunked-WDM F-stat kernel builds four basis filters
(`lat_chunked_het_kernels.hh:2455-2460`):

```
N_FILTERS = 4
A_arr    = {2.0, 2.0, 2.0, 2.0}
iota_arr = {pi/2, pi/2, pi/2, pi/2}
psi_arr  = {0, pi/4, 0, pi/4}
phi0_arr = {0, pi,   3pi/2, pi/2}
```

`iota = pi/2` everywhere (edge-on: the cross term drops, the plus term
carries a factor 1/2), so the four filters are **2 polarization
directions × 2 phase quadratures** — the standard JKS basis. The
quadrature half is a constant phase rotation of the complex heterodyned
representation, which is exactly the quantity the `d_h_im_out` machinery
already produces.

So only **two** waveform generations are needed. Filters 2 and 3 are
recovered from filters 0 and 1 by an exact unit-modulus constant:

| filter | psi | phi0 | generated stage | rotation alpha |
|---|---|---|---|---|
| 0 | 0    | 0     | stage 0 (psi=0)    | 1  |
| 1 | pi/4 | pi    | stage 1 (psi=pi/4) | -1 |
| 2 | 0    | 3pi/2 | stage 0            | +i |
| 3 | pi/4 | pi/2  | stage 1            | -i |

Both bases have rank 4 and condition number 1.000 — the two-generation
basis spans the identical 4-space.

## The reference implementation already exists

**`gb_signal_het_fstat_score_one_source` in the sig-het kernel already
does this** (`GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu:6323-6352`):

```c
const int n_stages = (fstat_mode == 1) ? 4 : 2;
...
} else {                       // FOLDED path
    psi_st[0]  = 0.0;  psi_st[1]  = M_PI / 4.0;
    phi0_st[0] = 0.0;  phi0_st[1] = 0.0;   // both generated at phi0 = 0
    for (int i = 0; i < GB_FSTAT_N_FILTERS; ++i) {
        s_of[i]  = i & 1;                  // filter -> stage
        alpha[i] = cmplx(cos(PHI0_F[i]), -sin(PHI0_F[i]));   // e^{-i phi0}
    }
}
```

The chunked-WDM kernel simply never received the fold. **Copy the
structure, not the constants blindly** (see the sign trap below).

### Why the centers cannot just be routed to the sig-het kernel instead

Two hard blockers, both correct design:

1. `GBSignalHetComputations.get_fstat_ll_wdm` raises without an active
   reference stash (`gbsignalhetcomputations.py:1342`), and sig-het
   references are built and torn down inside `_run_in_model_repeats` —
   during the RJ phase none is armed.
2. `_fstat_comp_method` deliberately unwraps to the chunked delegate
   (`wdm_comp = getattr(self.gb_wdm_comp, "chunked", self.gb_wdm_comp)`)
   because the F-stat must score against the **parent ACA residual
   passed explicitly by the caller**, never the in-model heterodyne
   reference.

So the fold has to be ported into the chunked kernel. This also means
the change **cannot perturb sig-het** (untouched) and **cannot change
what the F-stat scores against** (same call site, same residual).

## Where the work is

`src/lisatools/cutils/lat_chunked_het_kernels.hh`, the per-chunk loop
at **:2567**:

```c
for (int fi_b = 0; fi_b < N_FILTERS; ++fi_b) {
    double params_basis[16];
    ... params_basis[IDX_A/IDX_IOTA/IDX_PSI/IDX_PHI0] = *_arr[fi_b];
    // 1) TD-build into fd_chunk_buf[fi_b]  <- the expensive part
    //    (get_sky_vectors + get_tdi_Xf_single over N_sparse, then
    //     heterodyne, Tukey, FFT)
}
```

Everything downstream (the per-m window/rearrange → iFFT → parity →
accumulate) consumes `fd_chunk_buf[fi_b]` and does not care how it was
produced.

## Implementation, in three separable phases

**Phase 1 — fold the generation only (do this first).**
Generate stages 0 and 1 (psi = 0 and pi/4, both at phi0 = 0). Fill
`fd_chunk_buf[2]` and `fd_chunk_buf[3]` as `alpha_i * fd_chunk_buf[s_of[i]]`
— a complex scale over the buffer, no waveform work. Leave every
downstream consumer untouched. Lowest risk, and captures the whole
generation saving.

**Phase 2 — drop the redundant buffers (optional, occupancy win).**
Rather than materializing buffers 2 and 3, apply `alpha_i` at the
accumulation site. That halves the filter buffers and should cut the
kernel's shared-memory footprint materially — currently **74 KB against
27 KB for `get_ll`, i.e. 2 resident blocks vs 6**. Occupancy may be worth
as much as the arithmetic.

**Phase 1b — remove the scaffolding.** After the parity gate is green
and one production run confirms the timing, delete the unfolded branch
and its toggle (see the TODO under Validation). Do not let it become
permanent; the parity TEST stays, the device branch does not.

**Phase 3 — exploit the Gram-matrix structure (optional, later).**
With `h_2 = alpha_2 h_0`, the 4×4 `M` is derivable from the 2×2 of the
generated stages plus known constants (`|alpha| = 1`, so `M_22 = M_00`
etc.). Fewer accumulations, but it is the easiest place to get a
convention wrong — do not attempt before Phases 1-2 are proven.

**Two adjacent wins surfaced by the same audit, independent of the fold:**
- Hoist the 6 `invC` values out of the `fi` loops — currently re-read
  **84× per pixel**.
- Give the F-stat kernel the **orbit-spline cache** `get_ll` already
  uses: production sets `N_cp_orbit > 0` and `get_ll` consumes it at
  `:1380`, but the F-stat kernel never does.

## Build impact — this is a GBGPU recompile, not a LAT one

`lat_chunked_het_kernels.hh` lives in the LAT tree but **LAT's own build
does not compile it** — it is not in `lisatools_sources.cmake`. It is
consumed downstream: included by `GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu`
and `binding_gbgpu.cxx`, and listed in `GBGPU/CMakeLists.txt`. That is
the project's recompile-in-place model (downstream wheels compile
against upstream headers rather than linking an upstream archive).

So the object code for this kernel is in the **GBGPU** wheel:

- **Editing this header requires rebuilding GBGPU.** Same path as the
  phase-max fusion (GBGPU `c49fcb1`).
- A LAT rebuild is not required for the kernel change itself. `./install.sh`
  rebuilds the stack anyway, so it is moot if that is the route.
- Cluster-relevant: pulling this change **without** a GBGPU rebuild
  leaves the old kernel running — there is no loader-level fallback to
  warn you here (unlike the routing kernels, which degrade to a python
  chain with a one-time warning). **A silent no-op is the failure mode.**
  Confirm the rebuild took by checking that `rj_fstat_centers` actually
  moved, not by assuming the pull was enough.
- `LISATOOLS_HEADER_ABI_VERSION` only needs bumping if a struct layout
  changes; a pure kernel-body fold does not change any POD interface, so
  no ABI bump and no BBHx/FEW rebuild.

## ⚠ The sign trap — do not copy sig-het's constant on faith

The sig-het comment is explicit that its rotation sign is *empirical*:

> `e^{-i phi0}`: the empirically-fixed rotation sign of the
> constructed-ratio convention (see the header comment; gate (c) of
> `gb_sighet_fstat_parity.py` pins it).

This project has been bitten by exactly this before: the phase-max
fusion work needed per-scorer `_QUAD_SIGN_*` constants pinned
**empirically**, and the FD swap-cross slot came out at **-1** while
every other slot was +1. The chunked kernel has its own normalization
(raw accumulator with `kappa` folded in) and may well need the conjugate
sign or a different quadrature ordering.

**Determine the sign against the 4-filter reference on real data. Do not
assume it matches sig-het's.**

## Validation

1. **Keep the unfolded path as a runtime reference**, exactly as sig-het
   did with `fstat_mode == 1`. Add the equivalent toggle to the chunked
   kernel so the kernel can be A/B'd against *itself* bit-for-bit rather
   than against a reimplementation.

   > **TODO(remove-later): the unfolded path is VALIDATION SCAFFOLDING,
   > not production code.** Once the parity gate is green on the cluster
   > and one production run has confirmed the timing, strip the toggle
   > and the 4-stage branch. Reasons to be disciplined about this:
   > it doubles the constant tables and the `s_of`/`alpha` bookkeeping,
   > it keeps a second code path that nothing exercises (so it rots
   > silently and the next reader cannot tell which branch is live), and
   > it costs registers/shared-memory budget in a kernel whose occupancy
   > is already the constraint (74 KB → 2 resident blocks). Mark the
   > branch with an explicit `// TODO(fstat-fold): remove after the
   > parity gate lands` at the site so it is greppable. Retain the
   > PARITY TEST after removal — the test can pin the folded kernel
   > against a host-side 4-filter reference without keeping the device
   > branch alive.
2. **Parity gate**: mirror `gb_sighet_fstat_parity.py` — compare folded
   vs unfolded `(N, M)` for a batch of real rows. Require agreement to
   the fused-kernel tolerance already in use (`_dyn_atol = 1e-9 × batch
   max`; the fold is exact in exact arithmetic, so any residual is pure
   floating-point reassociation).
3. **End-to-end**: `fstat_maximized_extrinsics` outputs
   `(A_max, phi0_max, iota_max, psi_max, F)` must agree; `F` is the one
   that matters, since `sigma = 1/sqrt(max(2F,1))` and `ln_snr` derive
   from it.
4. **Production check**: `rj_fstat_centers` in `[GB_TIMING]` should
   roughly halve. Everything else in the propose is untouched, so a
   move-total change materially different from the centers delta means
   something else moved.

## Expected gain

Phase 1 alone: the F-stat row is currently **4-6× a `get_ll` row** (four
independent generations, no orbit-spline cache, 74 KB shared). Halving
the generations should take `rj_fstat_centers` from ~830 s toward
~400-450 s — **roughly 18-20% off the whole GB iteration**, exactly, with
no proposal-quality cost.

Phase 2's occupancy effect is unquantified and could add more; measure,
do not assume.
