# Plan — `fdot` in the WDM lookup table (`lisatools.domains.WDMLookupTable`)

Companion to `WDM_SPLINE_PLAN.md` / `WDM_MISMATCH_FINDINGS.md`. Two
implementations are sketched. Plan A is the recommended primary path
(revert to the n_ref-only build, keep the new parity logic on top).
Plan B keeps the per-n / per-m_diff build but actually wires fdot
through the interpolator.

The intent is to land Plan A *and* leave the Plan B build path in
`build_lookup_table` commented out behind a flag so we can swap back
quickly. Both share the same `get_wdm_coeffs` surface, so callers
(`gb_lookup_table_test_script.py`, `gb_lookup_prior_draws.py`,
`compare_template_vs_injection.py`, `diag_*` scripts) keep working.

---

## Current state (what is already in `domains.py`)

`WDMLookupTable.build_lookup_table` ([src/lisatools/domains.py:2234](LISAanalysistools/src/lisatools/domains.py#L2234))
generates, for every `(f_norm, fdot)` and every `m_diff`, the
de-rotated wavelet coefficient at **every** time pixel `n`. The stored
table has shape `(Nt, fdot_steps, f_steps)`. The trick that makes the
per-n storage cheap to interpolate is `get_x_points_no_fdot` (line
2467), which encodes `(f_norm, n)` into a single 1-D coordinate
`f_norm + factor_spacing * n`. That trick only works when fdot is
absent — `build_interpolator` and `get_table_coeffs` both
`raise NotImplementedError` on the fdot path
([domains.py:2453](LISAanalysistools/src/lisatools/domains.py#L2453),
[domains.py:2475](LISAanalysistools/src/lisatools/domains.py#L2475)).

Concretely the constant-frequency case works because for fdot=0 the
source stays in the same `m_floor` layer for every `n`. With fdot ≠ 0
the source's instantaneous frequency at pixel `n` is
`f(n) = f_ref + f_norm + fdot · (n·layer_dt − t_ref)` and that *can*
cross layer boundaries, so the cached `(n, m_diff, f_norm, fdot)`
entry no longer corresponds to a well-defined layer offset for an
arbitrary lookup. The build path that meshes `(f_norm × fdot)` is
fine, but the lookup math + the 1-D encoded interpolator are not set
up for it.

The parity / sign logic in `get_wdm_coeffs`
([domains.py:2489](LISAanalysistools/src/lisatools/domains.py#L2489))
is the part we *must* keep:

* `f_norm = f_arr - ms_to_use * layer_df` — instantaneous offset within
  the looked-up layer.
* `is_m_plus_n_even` swap: if `(m + n) % 2 == 0`, transfer
  most of the energy into the cosine branch (the four-branch swap at
  lines 2533–2543).
* `is_m_even` split (currently identical sin/cos for the two even
  branches — leave as is, both implementations should reproduce it
  bit-for-bit).
* The `(-1)^((ms_to_use - m_ref) parity)` post-multiply (line 2554) —
  this comes from FFT-mirror correction baked at build time. Plan A
  needs to re-derive this; Plan B already has it.

---

## Plan A — revert build to `(m_ref, n_ref)`-only, keep new parity logic

Reference: pre-rewrite shape in commit `26db3c5` (and its successors
`5e8b0d8`, `0064be8`, `aab845b`). The old build stored a single
`(f_steps, fdot_steps)` table by extracting `wave_*_wdm[:, m_ref, n_ref]`
([git diff](#) — see `git show 26db3c5:src/lisatools/domains.py`,
lines 1253–1276).

### Why this is the cleaner fdot story

For a source whose instantaneous frequency at pixel `n_eval` is
`f(n_eval)`, we evaluate the table as if a synthetic source had been
generated with `(f_table = f(n_eval), fdot_table = fdot_eval)` and we
read its `(m_ref, n_ref)` pixel. Because that synthetic source's
instantaneous frequency at *its* `n_ref` pixel is exactly `f_table`,
the layer-crossing bookkeeping collapses — the only `f_norm` we ever
have to query is `f(n_eval) mod layer_df`, which sits in `[0,
layer_df)` by construction. The fdot grid then captures the
chirp-within-wavelet effect through the build itself.

### Build (`build_lookup_table_n_ref`)

```python
# table shape: (fdot_steps, f_steps)   — no Nt, no m_diffs
# f_steps    = len(norm_freq_single_layer)         (one layer worth)
# fdot_steps = len(fdot_vals)

self.sub_settings = WDMSettings(self.Nf, self.Nt, self.data_dt, ...)
self.m_ref = m_ref
self.n_ref = int(self.sub_settings.Nt / 2)
self.is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0

# meshgrid the (f_norm, fdot) grid (single-layer f_norm, full fdot)
_f_vals, _fdot_vals = xp.meshgrid(self.norm_freq_single_layer + self.f_ref,
                                  self.fdot_vals)
_f_vals, _fdot_vals = _f_vals.ravel(), _fdot_vals.ravel()

t_vals  = xp.arange(self.sub_settings.N) * self.data_dt
t_diff  = t_vals - self.t_ref     # centered on n_ref

if td_window is None:
    td_window = xp.ones_like(t_diff)
self.td_window = td_window

table_sin = xp.zeros((self.fdot_steps, self.f_steps))
table_cos = xp.zeros((self.fdot_steps, self.f_steps))

for st, end in batched(...):
    inds  = slice(st, end)
    phase = 2*pi*(_f_vals[inds, None] * t_diff[None, :]
                  + 0.5 * _fdot_vals[inds, None] * t_diff[None, :]**2)
    wave_sin = xp.sin(phase); wave_cos = xp.cos(phase)
    wave_sin_wdm = TDSignal(wave_sin, ...).wdmtransform(settings=self.sub_settings,
                                                        window=self.td_window)
    wave_cos_wdm = TDSignal(wave_cos, ...).wdmtransform(...)
    # ONE pixel: (m_ref, n_ref)
    table_sin.ravel()[inds] = wave_sin_wdm[:, self.m_ref, self.n_ref]
    table_cos.ravel()[inds] = wave_cos_wdm[:, self.m_ref, self.n_ref]

# reshape back to (fdot_steps, f_steps) and store
```

**Windowing rule (important):** in the n_ref-only build the
synthetic source uses the *full duration* of `sub_settings.N`
samples, and the wavelet basis at `(m_ref, n_ref)` only has support
near the centre of that record. If `td_window` is anything other than
`xp.ones_like(t_diff)`, the build silently re-weights the carrier and
the stored coefficient stops matching what the GB pipeline produces.
Reuse the existing API but enforce a hard assertion when the user
passes a non-trivial `td_window` — and document that the test scripts
must drop the Tukey window in `gb_lookup_table_test_script.py:217` and
`gb_lookup_prior_draws.py` once Plan A is on.

### Interpolator

* No fdot: keep the existing `interp1d` over `norm_freq_single_layer`
  (drop the `factor_spacing * n_arr` encoding — there is no n axis
  any more).
* With fdot: a 2-D regular-grid interpolator over `(f_norm, fdot)`.
  Use `scipy.interpolate.RegularGridInterpolator(... method="linear")`
  on CPU and `cupyx.scipy.interpolate.RegularGridInterpolator` on GPU
  — both handle the meshgrid layout produced above directly. This
  replaces the `LinearNDInterpolator` from `26db3c5` (which was
  triangulating an unstructured cloud and was the slow path).

### Lookup (`get_table_coeffs`)

```python
def get_table_coeffs(self, f_norm, fdot_arr, n_arr):
    # n_arr is unused in Plan A but kept in the signature so callers
    # (gb_lookup_table_test_script, GBWDMComputations) don't need to
    # branch.
    if self.run_fdot:
        sin_coeffs = self.table_sin_interpolate((f_norm, fdot_arr))
        cos_coeffs = self.table_cos_interpolate((f_norm, fdot_arr))
    else:
        sin_coeffs = self.table_sin_interpolate(f_norm)
        cos_coeffs = self.table_cos_interpolate(f_norm)
    sin_coeffs[xp.isnan(sin_coeffs)] = 0.0
    cos_coeffs[xp.isnan(cos_coeffs)] = 0.0
    return sin_coeffs, cos_coeffs
```

### `get_wdm_coeffs` — unchanged shape, parity preserved

Keep `get_wdm_coeffs` verbatim
([domains.py:2489–2558](LISAanalysistools/src/lisatools/domains.py#L2489-L2558)):

* `ms = (f_arr / layer_df).astype(int)` then loop over the same
  `range(-num_m_layers, num_m_layers + 1)` block;
* `f_norm = f_arr[keep_now] - ms_to_use[keep_now] * layer_df` — the
  Plan-A table is keyed exactly on this f_norm;
* `is_m_plus_n_even` four-branch swap stays as-is — the table is now
  the same primitive as it was in the old code, so the same swap
  rules apply (this matches the user's instruction "transfer most to
  cosine if (m+n) is odd");
* The `(-1)^((ms_to_use - m_ref) parity)` post-multiply (line 2554)
  has to come from somewhere. In the per-n build it was baked in at
  line 2326 (`if int(m_diff) & 1: sin_coeff = -sin_coeff; cos_coeff =
  -cos_coeff`). With the n_ref-only build there is no `m_diff` loop
  at build time, so the post-multiply at line 2554 *is* the right
  place to apply the FFT-mirror sign — keep it.

### File-format / API touches

* `to_file` / `from_file_internal`: drop the `m_diffs` dataset (or
  keep it as an empty array for backward-compat readers) and store
  the 2-D `(fdot_steps, f_steps)` table. Bump a small `table_kind`
  attribute (`"n_ref_only"` vs `"per_n"`) so loaders can fail loudly
  on a mismatch instead of silently shape-mangling.
* `apply_eps_frequency`: still useful — it returns `(f_norm, m_diffs,
  m_ref)`. Keep the signature but document that `m_diffs` is only
  consumed by the eval-side `num_m_layers` loop in `get_wdm_coeffs`,
  not by the build.

### Caller fixes

* `gb_lookup_table_test_script.py:217` — drop the Tukey
  `td_window`. Pass `td_window=None` so the assertion in Plan A's
  build does not fire. Same change in `build_parity_fix_*`,
  `compare_template_vs_injection.py`, etc. Grep:
  `td_window=xp.asarray(signal.windows.tukey(...))`.
* `gb_lookup_prior_draws.py` already passes `td_window = None` (line
  203), so it is good.
* `analysis_container.transform(output_set, window=window)` — keep
  this `window` (it's the *outer* window used when transforming the
  TD injection into WDM for the residual side); it has nothing to do
  with the lookup build. The constraint is only that the *injected
  synthetic carriers used to build the table* are unwindowed.

### Risks for Plan A

1. **Pixel-only sample = larger interpolation error than per-n.** The
   per-n build smooths over a full wavelet support; n_ref-only only
   samples the centre pixel. Quantify with
   `gb_lookup_prior_draws.py` at fdot=0 first to confirm we are not
   regressing the <1e-8 mismatches you mentioned in commit `ace6a9e`.
2. **fdot grid density needs to be honest.** The chirp-within-wavelet
   effect is captured only as far as `fdot_vals` is dense.
   `apply_eps_fdot(eps=0.2, fdot_max_factor=1.0)` ≈ 10 samples per
   side is probably too sparse — start with `eps=0.05`,
   `fdot_max_factor` matching the prior range from
   `gb_lookup_prior_draws.py` (`FDOT_MAX = 1e-15` env var, line 237).
3. **`m_ref` placement.** The whole table is built at one carrier
   frequency `m_ref * layer_df`. For sources far from `m_ref` the
   chirp-within-wavelet effect scales with `fdot * (Nf*dt)` which
   doesn't care about `m_ref`, so this should be safe — but verify by
   building a second table at a different `m_ref` and confirming the
   coefficient at the same `(f_norm, fdot)` matches.

---

## Plan B — keep the per-n / m_diff build, properly wire fdot

Only worth doing if Plan A's interpolation error turns out to be too
large for the global-fit likelihood budget. Keep the build code in
`build_lookup_table` but rename it `_build_lookup_table_per_n` and
comment it out behind an `if self.table_kind == "per_n":` guard.

### Build

No change from the current code at
[domains.py:2234–2356](LISAanalysistools/src/lisatools/domains.py#L2234-L2356)
except:

* Drop the `breakpoint()` at line 2343.
* The `if int(m_diff) & 1: sin_coeff = -sin_coeff; cos_coeff =
  -cos_coeff` block (line 2326) needs verification under fdot. The
  TODO comment at lines 2322–2325 explicitly flags this — the parity
  bake came from a fdot=0 derivation. With fdot the imaginary
  (sine-coefficient) channel is no longer ~0, so flipping it might
  destroy real information. Two options:
  * (B1) Keep the flip, accept a re-derivation cost: do an analytic
    check by generating a small `(f_norm, fdot)` pair, computing the
    coefficient with and without the flip at adjacent `m_diff`, and
    confirming the cross-`m_diff` linear interpolation is still
    smooth in *both* sin and cos channels. The notes file
    `SIGN_FLIP_NOTES.md` already captures the fdot=0 reasoning — pull
    a parallel derivation for fdot ≠ 0 into a new
    `SIGN_FLIP_FDOT_NOTES.md`.
  * (B2) Drop the bake at build time, compensate at lookup. Cleaner,
    but loses the smooth-cross-boundary property the comment at lines
    2314–2321 relies on.

### Interpolator + lookup

Replace `interp1d(x_points_in, y_points_in)` with a
`RegularGridInterpolator` over `(f_norm_combined, fdot, n)` where
`f_norm_combined` is the *layer-extended* coordinate currently encoded
into `f_vals` by `f_vals` property at
[domains.py:2386](LISAanalysistools/src/lisatools/domains.py#L2386).

Lookup with fdot:

```python
def get_table_coeffs(self, f_norm, fdot_arr, n_arr):
    # f_norm is layer-extended: caller has already added m_diff*layer_df
    # back in, OR the lookup wraps modulo layer_df and shifts m_diff
    # accordingly (see below).
    pts = xp.stack([f_norm, fdot_arr, n_arr.astype(float)], axis=-1)
    sin = self.table_sin_interpolate(pts)
    cos = self.table_cos_interpolate(pts)
    ...
```

The wrap-and-shift logic in `get_wdm_coeffs` is the gnarly bit: the
source's instantaneous f at the eval pixel can fall outside the
single-layer `[0, layer_df)` window for non-zero fdot, and Plan B's
table answers that by walking into a different `m_diff` block. The
caller's `for m_diff in range(-num_m_layers, num_m_layers + 1)` loop
needs to shift `m_diff -> m_diff - k` where `k = floor((f_norm +
fdot*(n_eval - n_ref)*layer_dt) / layer_df)`. Write this as a small
helper and unit-test it against fdot=0 first (it must reduce to the
existing path).

### Storage

Same `(Nt, fdot_steps, f_steps)` table. The size scales as
`Nt * fdot_steps * f_steps` — for `Nt = 2560`, `fdot_steps = 20`,
`f_steps = 100 * (2 * num_layers_diff + 2) ≈ 1200` that's ~ 60M
entries per of {sin, cos} → ~1 GB. Watch the on-disk size before
committing to long fdot grids.

### Risks for Plan B

1. **Wrap-and-shift correctness.** Easy to get wrong at the layer
   boundary; the worst-case interpolation noise from a misshifted
   lookup would look exactly like the pre-fix layer-boundary spikes
   in `gb_prior_pattern_worst3_residuals.png`.
2. **Parity bake under fdot.** The TODO at lines 2322–2325 is real
   and unsolved.
3. **GPU memory.** `RegularGridInterpolator` on cupy keeps the table
   resident; the encoded-coordinate `interp1d` trick was specifically
   to avoid that. If GPU memory bites, fall back to a custom
   tri-linear kernel.

---

## Side-by-side implementations

We'll land Plan A as the active path and keep Plan B in-source but
inert:

```python
class WDMLookupTable(WDMSettings):
    BUILD_KIND = "n_ref_only"      # or "per_n" — switch to fall back

    def build_lookup_table(self, ...):
        if self.BUILD_KIND == "n_ref_only":
            return self._build_lookup_table_n_ref(...)
        elif self.BUILD_KIND == "per_n":
            return self._build_lookup_table_per_n(...)
        raise ValueError(self.BUILD_KIND)

    def _build_lookup_table_n_ref(self, ...):
        # Plan A — primary
        ...

    def _build_lookup_table_per_n(self, ...):
        # Plan B — kept for fallback; currently raises on fdot != 0
        # because the interpolator wiring is not done. Build still
        # runs end-to-end and can be exercised against the fdot=0
        # regression suite.
        ...
```

Same trick for `get_table_coeffs` and `build_interpolator`.

---

## Validation milestones (run in order)

All three reuse `gb_lookup_prior_draws.py`
([gb_lookup_prior_draws.py:193](gb_lookup_prior_draws.py#L193)) — it
already drives `WDMLookupTable.from_file` + `GBWDMComputations` +
`AnalysisContainer.template_inner_product` over a 200/1000-draw prior
sweep and emits hist + scatter + per-source diagnostics.

1. **fdot=0 regression.** Build Plan A with `fdot_vals = np.array([0.0])`,
   run `N_DRAWS=200 FDOT_MAX=0 python gb_lookup_prior_draws.py`. The
   mismatch histogram must stay at the same `<1e-8` floor we have
   today (the ace6a9e baseline). If not, the parity logic isn't being
   reused correctly.
2. **fdot bounded by chirp.** `FDOT_MAX=1e-17` (one wavelet pixel of
   drift over Tobs), `fdot_vals = apply_eps_fdot(eps=0.1,
   fdot_max_factor=2.0)`. Mismatch should stay below `1e-6`.
3. **fdot at prior edge.** `FDOT_MAX=1e-15`,
   `fdot_max_factor=10.0`. Mismatch budget is whatever the
   global-fit likelihood can absorb; the histogram from
   `gb_lookup_prior_draws.log` shows the current
   no-fdot tails — Plan A should not widen them.

Also run `compare_template_vs_injection.py` and the
`diag_pixel_residual*` family on a single off-band source after each
build to confirm no per-pixel residual spikes appeared.

---

## Status (2026-05-19 overnight)

* **Plan A build is done and produces correct stored values.** Verified by
  dumping Plan A's `(fdot_steps, f_steps)` table vs per_n's
  `(Nt, fdot_steps, f_steps)` table at the `(n=n_ref, m_diff=0)` slice
  for the same params — values match (e.g. `table_cos[start..+5] =
  -120.83` in both).
* **`get_wdm_coeffs` eval is the unsolved bit.** With Plan A's table:
  * Per-pixel parity swap (existing per_n eval): mm ≈ 1.0 across all
    f_frac.
  * Per-source net swap matching per_n's `(m_ref - ms)` net rule:
    mm ≈ 0.5 across all f_frac.
  * Added per-pixel carrier-phase rotation `R(-θ)`: mm goes back up to
    1.0 (rotation washes out the alternating pattern that should be
    there).
  * Added per-pixel basis-parity sign `(-1)^(n_eval - n_ref)`:
    helps f_frac=0.05 marginally, hurts f_frac=0.5 / 0.95.

  Per-pixel diagnostic on f_frac=0.05 (no rotation, per-source swap)
  shows the template signs alternate per pixel while the injection
  alternates per *pair* of pixels — every other pixel matches
  exactly, the in-between pixels are sign-flipped. Adjusting the
  per-pixel sign breaks the other f_fracs because the wavelet basis
  has a `(m+n) mod 4` phase cycle (four basis types: `+cos`, `-sin`,
  `-cos`, `+sin`), not a simple `(m+n) mod 2`. The right Plan A eval
  needs a 4-way per-pixel transformation derived from the WDM
  Meyer-Mallat construction. **TODO before this can ship.**
* **per_n baseline holds** on the same small params: mm ≈ 1.5e-12
  on-grid, ≈ 2.3e-4 mid-bin. So the eval-side parity swap + FFT-mirror
  post-multiply machinery in `get_wdm_coeffs` is correct when fed
  per_n-style per-pixel values; Plan A's failure mode is purely the
  missing per-pixel transformation.

* **C side** (`fastlisaresponse/gbcomps.py:85`) still asserts the
  per_n `(Nt, fdot_steps, f_steps)` shape and is gated off by the
  test script's `RUN_C_LOOKUP=auto` (which skips when
  `build_kind == "n_ref_only"`). After Plan A's eval is fixed,
  port the corresponding lookup logic into the C kernel.

## Validation knobs the test script now exposes

* `WDM_BUILD_KIND=n_ref_only` (default) or `per_n`. Auto-routes the
  store_path so the two kinds don't clobber.
* `EPS_FREQ`, `NUM_LAYERS_DIFF`, `FDOT_EPS`, `FDOT_MAX_FACTOR`,
  `BATCH_SIZE_GEN` — table build knobs.
* `F_FRAC=0.05` (single) or `F_FRACS=0.05,0.5,0.95` (sweep).
* `DIAG_PIXEL=1` — dumps per-layer rms/cosine of inj vs template
  plus first-6-pixel side-by-side comparison; the fastest way to see
  which pixels match.
* `RUN_C_LOOKUP=auto` (default — uses C path only for per_n),
  `1` (force on), `0` (force off).

## Open questions for you

* Do we want the Plan A switch behind an env var (`WDM_BUILD_KIND`)
  or a class-level constant? Env var is faster to flip during the
  validation sweep; constant is easier to reason about for the
  C-side.
* The `td_window` API: hard-assert vs warn-and-ignore when a
  non-trivial window is passed in Plan A? The existing test scripts
  all pass a Tukey window — silently ignoring is bug-friendly,
  asserting forces the call sites to be updated.
* `m_ref` chooser: leave it at `int(3e-3 / layer_df)` (today's
  default) or expose as a knob in `apply_eps_frequency`? For the
  n_ref-only build the table is a single 2-D slab so we could afford
  to build one per `m_ref` if needed — but probably not necessary.
