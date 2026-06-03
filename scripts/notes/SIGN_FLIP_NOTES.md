# WDM C-lookup sign-flip investigation

How to use this file: run [gb_lookup_sign_flip_diag.py](gb_lookup_sign_flip_diag.py)
to find which draws flip sign; come back here for the candidate-by-candidate
analysis.

## Verdict (TL;DR)

The full rotation `conj(M · exp(-i π/2))` in
[fast_wdm_inner](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L872)
**is algebraically correct** — see the derivation below. But it is
**redundant**, and its only practical effect is to mutate `tdi_channel_val`
into a different complex convention than the values that get fetched on
the next two lines for the central-difference frequency. **That mutation
is the bug in candidate 1.**

Candidate 1 corrupts `f[i]` (~ ±1 mHz shift, ~ 30 layer routing error)
but does *not* sign-flip any single `w_mn`. Candidate 3 (layer boundary
rounding) likewise does *not* sign-flip per-pixel — the `(m+n)` parity
rule is keyed off the pixel coordinates, not the lookup’s `layer_m`.

So neither candidate 1 nor candidate 3 on its own explains "off by an
overall minus sign". The most plausible mechanism that *does* produce
that signature is candidate 4 (the Python sign-flip tracker in
[`new_extract_amplitude_and_phase`](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L2804)),
which feeds `tdi_amp` and `tdi_phase` into the splines that build the
*Python* TD waveform and the Python lookup wrap. If that tracker misses
a `|M|` zero, it’s the Python side that's flipped, but the comparison
against the C template makes the C template look wrong.

The diagnostic distinguishes the two cases:

* Python lookup vs C lookup: if they **agree** but both disagree with
  the `TDSignal(td).transform(wdm_set)` injection by `-1` ⇒ candidate 4.
* Python lookup vs C lookup: if they **disagree** by `-1` ⇒ bug in C
  (probably not candidate 1 or 3 — keep looking).

## Diagnostic

`gb_lookup_sign_flip_diag.py` reuses the prior, WDM grid, and lookup-table
setup from `gb_lookup_prior_draws.py` and the Python lookup wrap
(`GBLookupWaveWrap`) from `gb_lookup_table_test_script.py`. For each draw
it builds the WDM template both ways and reports

* `rho_unwt`  — Frobenius cosine `<py|C> / |py||C|`
* `rho_wt`    — noise-weighted `<py|C>/sqrt(<py|py><C|C>)` (via
  lisatools `template_inner_product`).

Sign-flipped draws (rho_unwt < -FLIP_TOL, default 0.5) are dumped to
`gb_sign_flip_diag_dumps/flip_*.png` with chan-0 heatmaps of `py`, `C`,
`py - C`, `py + C` so it’s obvious whether the disagreement is a global
`-1`, a per-pixel speckle, or a layer offset.

## Derivation: is the rotation correct?

Convention for the TDI envelope, from
[`TDTDIOutput.eval_tdi`](lisa-on-gpu/src/fastlisaresponse/tdionfly.py#L454):

```
M(t) = tdi_amp(t) · exp(-i · (tdi_phase(t) + phase_ref(t)))
     = (a(t) + i b(t)) · exp(-i · phase(t))
```

i.e. `M = slow · exp(-i·phase_carrier)`. Writing `M = amp · exp(-iα)`
with `α = -arg(M)` and `amp = |M|`, the real and imaginary parts are

```
Re(M) =  amp · cos(α)
Im(M) = -amp · sin(α)
```

The lookup-table content is the WDM transform of the build-time
references `cos(2π f t)` and `sin(2π f t)`, demodulated by the per-bin
phase `phase_n` ([build code](LISAanalysistools/src/lisatools/domains.py#L2272-L2305)):

```
_cos_coeff = Re((C + iS) · exp(-i phase_n)) =  C cos(phase_n) + S sin(phase_n)
_sin_coeff = Im((C + iS) · exp(-i phase_n)) = -C sin(phase_n) + S cos(phase_n)
```

(`C = WDM(cos)` at the reference frequency, `S = WDM(sin)`.) The build
then applies an `(m+n)`-parity *swap* before storing: at `(m+n)` odd,
`table_sin = _cos_coeff` and `table_cos = _sin_coeff`; at `(m+n)` even,
no swap. The lookup applies the *inverse* swap, so after build + lookup,
**both parities** give

```
sin_final =  _cos_coeff = C cos(phase_n) + S sin(phase_n)
cos_final =  _sin_coeff = -C sin(phase_n) + S cos(phase_n)
```

That is, `(C + iS) · exp(-i phase_n)` lives in `(cos_final + i sin_final)`
*after* the dance — the build/lookup swap pair simply identifies
`{table_sin, table_cos}` with `{Re, Im}` of the rotated wavelet block.

Python wrap formula
([gb_lookup_table_test_script.py:102, 130](gb_lookup_table_test_script.py#L102-L130)):

```
phi_t = (tdi_phase + ref_phase) + π/2  = α + π/2
w_py  = amp · (sin_final · sin(phi_t) + cos_final · cos(phi_t))
      = amp · (sin_final · cos(α) - cos_final · sin(α))
      =  Re(M) · sin_final  +  Im(M) · cos_final
      =  Re(M) · _cos_coeff +  Im(M) · _sin_coeff       (substituting)
```

C formula
([TDIonTheFly.cu:872, 549](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L549)):

```
M' = conj(M · exp(-iπ/2)) = i · conj(M)  ⇒  M'.real = Im(M),  M'.imag = Re(M)
w_C = c_nm · M'.real + s_nm · M'.imag
    = c_nm · Im(M) + s_nm · Re(M)
    = _sin_coeff · Im(M) + _cos_coeff · Re(M)
                            (since s_nm pairs with table_sin = _cos_coeff
                             after the build/lookup parity dance)
```

`w_py == w_C` ✓ — **the rotation is correct.**

But the rotation is also **redundant**. Without it (using raw `M`), the
exact same answer comes from a one-line swap of the formula:

```cpp
w_mn = s_nm * tdi_channel_val_raw.real() + c_nm * tdi_channel_val_raw.imag();
```

i.e. swap `c_nm ↔ s_nm` so that `s_nm` (which actually carries
`_cos_coeff`) multiplies `Re(M)`, and `c_nm` (carrying `_sin_coeff`)
multiplies `Im(M)`. No rotation, no `tdi_channel_val` mutation,
candidate 1 disappears for free.

## Candidate-by-candidate

### 1. `tdi_phase_mid` uses the rotated value

[fast_wdm_inner around L870–908](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L846)

```cpp
tdi_on_fly_here.get_tdi_Xf_single(&tdi_channel_val[0], tn, ...);
for (int i = 0; i < 3; i += 1)
    tdi_channel_val[i] = gcmplx::conj((tdi_channel_val[i] * gcmplx::exp(-I * M_PI / 2.)));  // ← mutates in place

// ...later, with raw down/up values...
tdi_phase_down = -arg(tdi_channel_val_down_dt[i] * exp(I phase_ref_down));   // raw M
tdi_phase_mid  = -arg(tdi_channel_val[i]         * exp(I phase_ref));        // rotated M  ← BUG
tdi_phase_up   = -arg(tdi_channel_val_up_dt[i]   * exp(I phase_ref_up));     // raw M
```

`tdi_phase_mid` is shifted from the up/down convention by
`K = π/2 - 2·arg(M_mid)` (mod 2π). The shift cancels exactly in
`dphi_up - dphi_down`, **but** the per-anchor unwraps

```cpp
if (dphi_up >  M_PI) dphi_up -= 2π;
else if (dphi_up < -M_PI) dphi_up += 2π;
```

are single-step and depend on `dphi_x - tdi_phase_mid`. When the
convention shift puts `dphi_up` and `dphi_down` on opposite sides of
the unwrap threshold, the unwrap fires asymmetrically and
`tdi_frequency` picks up a spurious `±1/(2·Δt) ≈ ±1 mHz`.

* Effect: wrong `f[i]` → wrong `layer_m` → contributions routed to
  layers ~30 away from the truth. **Not** a sign flip.
* Fix: drop the rotation and swap the formula (preferred — see above),
  OR compute `tdi_phase_mid` from a *local* raw copy of
  `tdi_channel_val` before the rotation, OR move the rotation after
  the whole frequency-computation block.

### 2. Build-time vs lookup-time `(m+n)` parity swap

Build:
[domains.py:2298-2301](LISAanalysistools/src/lisatools/domains.py#L2298-L2301)
swaps when `(m+n)` is **odd**.

Lookup (Python and C):
[TDIonTheFly.cu:535-546](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L535-L546),
[domains.py:2501-2511](LISAanalysistools/src/lisatools/domains.py#L2501-L2511)
swap when `(m+n)` is **even**.

These compose to a *no-op* on both sides, as worked out in the
derivation above. Both Python and C agree, and the existing
single-source check in
`gb_lookup_table_test_script.py` passes. Rule out unless a dumped flip
heatmap shows an obvious every-other-row pattern.

### 3. `layer_m = int(f / layer_df)` rounding near a layer boundary

If C and Python pick `layer_m` differing by 1 because `f` straddles
`(m+½)·layer_df`, what gets written to a given pixel `(M_pix, n_pix)` is
still:

* Python: loop iteration `diff_py = M_pix - layer_m_py` writes
  `lookup(f_py - M_pix·layer_df, n_pix, parity(M_pix + n_pix))`.
* C: loop iteration `diff_C = M_pix - layer_m_C = diff_py ± 1` writes
  `lookup(f_C - M_pix·layer_df, n_pix, parity(M_pix + n_pix))`.

The parity rule is keyed off `M_pix + n_pix`, **not** off `layer_m`, so
the swap is the same in both paths. The only difference is the
within-layer interpolation offset `f - M_pix·layer_df`, which is a
small numerical perturbation — not a sign flip.

The edge layers (`m_py - num_diff` only-Python, `m_C + num_diff`
only-C) are populated only by one side, so the diff plot would show a
single "extra" row at each end. Still not an overall sign flip.

### 4. `new_extract_amplitude_and_phase` sign-flip tracker — **most likely**

[TDIonTheFly.cu:2804](lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu#L2804)
detects minima of `|M|` and accumulates them as `count`, then sets
`flip = (-1)^count` and `pjump = count·π`. `As[i]` is multiplied by
`flip[i]` and `Dphi[i]` gets `+pjump[i]`. This makes
`amp · exp(-i·(phase + phase_ref))` reconstruct correctly **as long as
every real zero of `|M|` is detected**.

The detector is a finite-tolerance second-derivative test (`abs(dA2/dA1)
< 0.1`, `abs(dA3/dA1) < 0.1`). For sources whose `|M|` zero is shallow
or whose neighboring samples have unfavorable noise, the test can miss
a real crossing → `count` stays at its prior value → `flip` keeps the
wrong sign from then on → `tdi_amp` is `-|M|` instead of `+|M|` and
`tdi_phase` is missing its `+π` → `Re[M_reconstructed] = -Re[M_true]`
for all times past the missed zero.

This affects:

* Python WDM **injection** (built from `inj_spline.eval_tdi(t_arr)` → 
  `TDSignal.transform(wdm_set)`): **time-tail of the signal is flipped.**
* Python lookup wrap (uses `tdi_amp`/`tdi_phase` directly): same flip.
* C lookup template (uses `get_tdi_Xf_single`, no tracker): **correct.**

If the missed zero is *before* `t_start`, the *entire* template is
flipped relative to C. If the missed zero is mid-signal, only the
post-zero portion is flipped.

So "the C WDM is off by a minus sign vs the injection" is exactly the
symptom of this bug, with the **bug actually on the Python side**.

* Confirm with the diagnostic: if `rho_unwt(py_lookup, C_lookup) ≈ -1`
  but `rho(py_lookup, py_injection) ≈ +1`, it's candidate 4.
* Possible fixes: tighten the `0.1` tolerances; switch to a
  sign-blind reconstruction (`Re[M] = tdi_amp · cos(tdi_phase + phase_ref)`
  using the **raw** `|M|` everywhere and not bothering with the count
  tracker); or, for the lookup wrap, pull `Re(M)` and `Im(M)` from
  `eval_tdi` directly instead of going through `tdi_amp · exp(...)`.

## What to do next

1. Run `gb_lookup_sign_flip_diag.py` (no args, ~200 draws) to confirm
   flipped draws exist with the user’s lookup table.
2. For each flipped draw, the script writes `flip_<i>.png`. Look at the
   `py - C` and `py + C` panels:
   * `py + C ≈ 0` everywhere → candidate 4 (or another whole-template
     sign mechanism).
   * Speckle pattern in `py - C`, similar magnitudes → frequency-routing
     error (candidate 1).
   * Edge-only differences in `py - C` → boundary rounding (candidate 3).
3. Pull one flipped draw’s params into a follow-up script and compare
   `tpl_py`, `tpl_C`, and `WDM(TDSignal(eval_tdi))`. The party that
   stands alone is the buggy one.
4. If it’s candidate 4: the fix is in Python, not C. Replace the
   spline-based reconstruction with one that doesn’t depend on the
   `count` tracker (e.g. store `Re(M)` and `Im(M)` directly).
5. If it’s candidate 1 (despite the analysis above), the minimal C
   patch is to compute `tdi_phase_mid` from a raw copy of
   `tdi_channel_val` before the rotation, or to drop the rotation
   entirely and swap `c_nm`/`s_nm` in the lookup formula.
