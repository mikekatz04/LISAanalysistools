# Part B implementation guide — the het-FFT column producer, step by step

**Companion to:** `2026-07-10-stft-gb-fresnel-accuracy-and-het-fft-successor.md`
(design study §4: heterodyned, spline-fed, in-register FFT per (source, STFT
segment) — the chunked-het analog for the STFT basis). This file is the OFFLINE
execution guide.

**Prerequisite:** the column-producer seam, commit `c5cd45a`. The producer plugs in
as a second `ColumnT` policy beside `FresnelColumn` in
`src/lisatools/cutils/lat_stft_kernels.hh`; consumers (ll/swap/fill/fstat/grad
kernels) are NOT touched. Environment/gates/rebuild cheat-sheet: §0 of the Part A
guide (same workspace, same commands).

**Target:** per-column method error 1e-7..1e-9 (window-exact, physics exact to
spline accuracy) at LOWER per-segment cost than the windowed Fresnel; the remaining
mismatch is then the representation's truncation floor (window α × n_side, see the
design study §4.6).

---

## The policy contract you are implementing (from the seam)

```cpp
struct HetFFTColumn</*SourceT, int N_SUB*/> {
  struct State { ... };                       // policy-defined, per column
  static void setup(State&, SourceT&, STFTFresnel*, STFTDomain*, double* params,
                    Vec k, Vec u, Vec v, int* lsr, int* lse, int bin_i,
                    double t_seg, double t_anchor_shift,
                    double window_factor, bool freq_from_tdi_phase);
  static cmplx value(const State&, int j, int freq_j_here, double freq_here);
};
```

Contract notes that MATTER here (documented at the seam):
- `value()` must return the RAW Fourier value (consumers apply the 0.5 real-signal
  convention and any fill `factor`).
- The swap kernel queries `value()` over the UNION of two carriers' stencils —
  up to `|freq_j_here − carrier_j| ≤ n_side_bins + |carrier_add − carrier_remove|`.
  A tabulated-spectrum policy must clamp: return 0 outside its tabulated range
  (physically: the same truncation the stencil already imposes; far cross-terms are
  negligible for well-separated pairs).
- `t_anchor_shift` (use_midpoint) is a FRESNEL expansion-anchor concept; the exact
  producer ignores it (document with `(void)` cast, do not repurpose).
- `freq_from_tdi_phase` likewise: the spectrum carries the true TDI phase; carrier
  PLACEMENT (below) is the only frequency estimate needed.

---

## B1. Python mock first — lock the conventions cheaply (no C++)

**Goal.** Validate the math + every convention (scaling, FT origin, midpoint
quadrature, window handling, carrier snapping) in numpy against the brute STFT
before writing any device code.

**Script** (new, `lisa-analysis-tools/scripts/validation/stft_hetfft_column_mock.py`):

1. Build the reference exactly as `GBGPU/tests/test_stft_gb_accuracy.py::setUpClass`
   does: `GBTDIonTheFly` → `out.eval_tdi(data_t)` TD stream → `TDSignal(...).stft(
   window=seg_win, settings=stft_settings)` = the exact per-segment spectrum `D`.
2. Mock the producer per column (pure numpy):
   - sample amp_j(t), φ_j(t) of the TDI (from the SAME `eval_tdi` samples, or from
     `get_tdi`-style sparse control points + `CubicSpline`) at
     `t_n = t_seg + (n + 0.5)·dt_seg/N_sub`, n = 0..N_sub−1 (MIDPOINT rule — the
     prototype's key quadrature fix);
   - heterodyne: `s_n = amp·exp(i(φ(t_n)) ) · exp(−2πi f_het (t_n − t_seg))` with
     `f_het = k_car·df_stft` (carrier snapped to the grid, see B2.3);
   - window: multiply by `w(t_n)` = the SAME per-segment Tukey used for `D`
     (sampled at midpoints);
   - spectrum: `S_m = (dt_seg/N_sub) · Σ_n s_n · exp(−2πi m n / N_sub)`
     (an N_sub-point DFT with the extra half-sample phase
     `exp(−iπ m / N_sub)` from the midpoint offsets — write it explicitly);
   - bin m ↔ absolute STFT bin `k_car + m` for m ∈ [−N_sub/2, N_sub/2) (fftshift
     bookkeeping); compare `S_m` to `D[ch, seg, k_car + m − ind_min]`.
3. Also compare against `FresnelColumn` values (via
   `STFTGBComputations.fill_global_stft` on a 1-source batch) — on a
   near-monochromatic source the two must agree to the Fresnel's own accuracy
   (~1e-3-field / A&S floor); this locks the amp/phase/conj and 0.5 conventions
   (remember: fill scatters `0.5·factor·value`; the mock's `S_m` corresponds to the
   RAW value, so compare against `2×` the filled pixel at factor=1).
4. Scan: N_sub ∈ {8, 16, 32} × n_side ∈ {2, 4, 10} × window α ∈ {0, 0.3} ×
   N_cp density (if using sparse control points) → print max in-stencil relative
   field error per config.

**Acceptance:** in-stencil field error ≤1e-7 at N_sub ≥ 2·(n_side+2) with
spline-fed sampling at ~2 control points/day; exact-sample variant ≤1e-12
(pure DFT identity check). Conventions locked = the phase of S_m matches D
(not just |S_m|).

Likely convention traps (check them explicitly in the mock):
- the kernels use `arg(conj(TDI))` (Fresnel convention, header comment) — carry the
  conjugation consistently or the phase sign flips;
- the brute STFT's normalization (`TDSignal.stft`) vs the `dt_seg/N_sub` DFT scale;
- `ind_min` offset between absolute bins and active-band storage.

---

## B2. C++ device pieces (LAT header, new file `lat_stft_hetfft.hh`)

Keep the new policy in its own header included by `lat_stft_kernels.hh` users
(GBGPU's TU) to keep the seam file small. Pieces:

### B2.1 Per-source spline prepass (in-kernel, shared memory)

Mirror `fast_wdm_inner_heterodyne_spline` steps 1–3
(`lat_chunked_het_kernels.hh`, `wdm_fit_cubic_spline`,
`populate_orbit_spline_cache`, `OrbitsSplineCache`): at the TOP of the per-bin loop
(block-per-binary — the block already owns one source), threads cooperatively:

1. evaluate `src.get_tdi_raw[_cached]` at `N_cp` control points spanning
   `[t0, t0 + num_times·dt]` (amp_j, phase_un_het_j per channel);
2. de-rotate: `dphi_j(t_cp) = phase_un_het_j − 2π f_het t_cp`
   (`f_het = k_car·df_stft`, B2.3) — unwrap along cp index (adjacent-point
   `remainder` unwrap; the chunked-het code shows the pattern);
3. fit cubic splines for amp_j and dphi_j (`wdm_fit_cubic_spline`) into
   `CUDA_SHARED` storage.

Shared budget: 3ch × 2 (amp, phase) × N_cp × 4 coeffs × 8 B ≈ 24 kB at N_cp=128
(90 d at ~1.4/day) — inside the default 48 kB; for longer spans use the
`cudaFuncSetAttribute` opt-in (pattern at the end of
`GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu::gb_run_fd_wave_tdi_wrap`) or chunk the
span. CPU mirror: plain stack/heap arrays, same code via the `CUDA_SHARED` macro.

N_cp rule of thumb: envelope varies on ~1 day (antenna sweep); start at 2/day and
verify with the B1 mock's N_cp scan + B3 gates. RJ births on short live spans: N_cp
= max(8, 2·span_days).

**Note:** this prepass changes the eval-block structure (a per-bin cooperative
phase before the per-thread column loop). Two clean options:
- (a) put the prepass INSIDE `HetFFTColumn` as a static `prepare_source()` hook and
  add one guarded call at the top of the bin loop in the two eval blocks + fill
  (small consumer touch, one line each, no-op default on `FresnelColumn`); or
- (b) generalize `setup()` to lazily build the spline on `time_i == first` — NOT
  recommended (thread-divergent, needs sync).
Option (a) is the intended shape: add
`ColumnT::prepare_source(shared_ws, src, params, ...)` with an empty default in
`FresnelColumn`, and re-run the byte-oracle to prove Fresnel is untouched.

### B2.2 `HetFFTColumn::State` + `setup()`

```cpp
template <class SourceT, int N_SUB>          // N_SUB compile-time, power of two
struct HetFFTColumn {
  struct State {
    cmplx spec[3][N_SUB];   // per-channel spectrum, fftshift-ordered
    int carrier_j;          // absolute carrier bin (stencil placement)
    int m_lo;               // spec index of bin (carrier_j - N_SUB/2)
  };
  ...
};
```

`setup()` per (source, segment):
1. sample the shared splines at the N_SUB midpoint times (polynomial eval, no
   transcendentals);
2. `s_n = amp · exp(i dphi)` — use ONE sincos for the first sample + the
   recurrence `exp(iΔ)` multiply for uniform-in-n phase increments of the RESIDUAL
   heterodyne only if you also linearize dphi; simplest correct v1: `polar(amp,
   dphi)` per sample (N_SUB sincos — still ≪ Fresnel's per-pixel budget); optimize
   later (the spec's R1 trick);
3. multiply by the precomputed window vector `w[N_SUB]` (CUDA_SHARED constant,
   filled once per kernel from `fresnel->window_alpha` — same Tukey formula as
   `TDSignal.stft`'s window arg; the two must be THE SAME array shape);
4. in-register serial radix-2 FFT of `spec[j]` (write the ~25-line
   `hetfft_serial_fft<N_SUB>` device helper: bit-reverse + butterflies, twiddles via
   recurrence; for N_SUB ≥ 64 call the block-cooperative `wdm_spline_radix2_fft`
   from `lat_wdm_fft.hh` instead — but then setup() is no longer per-thread; keep
   v1 at N_SUB ≤ 32);
5. apply the scale `dt_seg/N_SUB`, the midpoint half-sample phase
   `exp(−iπ m/N_SUB)`, and the FT-origin convention locked in B1;
6. `carrier_j` from the params' f0 (`stft->get_freq_index(src.get_f(t_seg, params,
   bin_i))` — astro f0 is sufficient for PLACEMENT: Doppler ±0.4 µHz ≪ df_stft;
   keep the SAME placement the keep-masks/engine assume).

`value(s, j, freq_j_here, freq_here)`: `(void) freq_here;` translate
`freq_j_here − s.m_lo` to the fftshift index; **return 0 if outside [0, N_SUB)**
(the swap-union clamp).

### B2.3 Carrier snapping

`k_car = llround(f0_astro / df_stft)`; `f_het = k_car · df_stft`. One global
heterodyne for the whole observation (design study §4.1). `carrier_j` and `k_car`
must be the SAME integer — one source of truth in `setup()`/`prepare_source()`.

---

## B3. Wiring + Python surface

1. Instantiate alongside Fresnel in GBGPU's TU
   (`GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu`, the `stft_*_impl<GBTDIonTheFly>`
   call sites): dispatch on a new `int column_policy` argument in the
   `gb_stft_*_wrap` functions —
   `0 → stft_get_ll_impl<GBTDIonTheFly>` (Fresnel default),
   `1 → stft_get_ll_impl<GBTDIonTheFly, HetFFTColumn<GBTDIonTheFly, 16>>`,
   `2 → ... N_SUB=32` (compile-time N_SUB ⇒ enumerate the sizes you support).
2. Bindings: add the `column_policy` (+ `n_cp`) args in
   `GBGPU/src/gbgpu/cutils/binding_gbgpu.{hpp,cxx}` (`gb_stft_get_ll` etc.).
3. Python: `STFTGBComputations.__init__(..., column_policy="fresnel"|"hetfft",
   n_sub=16, n_cp_per_day=2.0)`; forward through each `gb_stft_*` call
   (`GBGPU/src/gbgpu/gbcomps.py`). The band engine
   (`GBGPU/src/gbgpu/gb_likelihood.py::STFTBandLikelihoodEngine`) needs NO change —
   it forwards through the comp.
4. Keep-mask sanity: the engine's central-bin `kept_out` logic uses the params' f0 —
   consistent with B2.3 placement by construction; add an assert in the smoke test.

Rebuild BOTH wheels after every C++ edit (Part A guide §0).

---

## B4. Validation ladder (run in this order)

1. **Byte-oracle with policy=fresnel** (`stft_column_policy_oracle.py --compare`):
   the new code paths must leave the default EXACTLY alone (including the
   `prepare_source` hook addition of B2.1a).
2. **Policy-parity unit test** (new `GBGPU/tests/test_stft_hetfft.py`): one
   near-monochromatic source, fill via both policies at n_side=4 → in-stencil field
   agreement to the Fresnel's own accuracy class (~1e-3 field pre-A1, ~1e-6 with
   Part A's A1 in — this test pins conventions, not exactness).
3. **Brute-STFT accuracy** (port the `STFTEngineAccuracy` harness): in-stencil
   interior mm vs the exact STFT at N_SUB=16/32, α ∈ {0, 0.3}, n_side ∈ {4, 10}:
   target ≤1e-7 (spline-limited); log the N_cp scan.
4. **Aliasing cliff check:** mm vs N_SUB at fixed n_side — must be flat for
   N_SUB ≥ 2(n_side+2) and degrade below (reproduces the prototype's
   `n_sub ≳ 2·n_side+1` rule; guards the clamp/margin logic).
5. **Engine + flow integration:** the trio must stay exactly `23 passed +
   2 subtests` (policy default unchanged); run `test_gbspecial_flow_stft.py` once
   with the comp built as hetfft (fixture kwarg) — all 4 tests green.
6. **Per-source real-data numbers:** add `--stft-policy hetfft` to
   `scripts/validation/gb_mojito_stft_fd_mismatch.py` (plumb through
   `STFTGBComputations`) and compare the full/in-stencil/stencil-interior columns
   against the Fresnel rows on the box data.

---

## B5. GPU bring-up + benchmark (on the box)

1. Build both wheels with CUDA; run the oracle `--capture/--compare` GPU vs CPU
   (expect FP-level, not byte, agreement across devices — compare with a 1e-12
   rtol variant, or capture per-device references).
2. Shared-memory audit: prepass splines (B2.1) + the existing kernel shared arrays
   must fit; use the `cudaFuncSetAttribute` opt-in where needed.
3. Register/local-memory check at N_SUB=32 (`-Xptxas -v`): `spec[3][32]` = 1.5 kB
   per thread will live in local memory — acceptable if coalesced; if occupancy
   drops, prefer N_SUB=16 + wider stencils via the cooperative-FFT variant.
4. Bench protocol (mirror the oracle `--bench` shapes): get_ll/swap/fill medians at
   n_side ∈ {2, 10, 25} × policy ∈ {fresnel, hetfft} × α ∈ {0, 0.3}. Go/no-go:
   hetfft ≥ 1× Fresnel speed at n_side=2 AND ≥ 3× at n_side=25, with accuracy from
   B4.3. (CPU expectation from the design study's op count: parity-or-better
   already at n_side=2.)
5. Only after go: consider flipping the production default and the follow-ups
   (native STFT Fisher can reuse the spectra; the signal-het reference layer for
   in-model repeats is the next rung — design study §4.4).

---

## Pitfalls checklist (from the workspace's recorded lessons)

- **Dangling-pointer rule:** any array handed to a pointer-storing wrap must be
  OWNED by a long-lived Python object and dtype-exact (complex128) — nanobind
  conversion temporaries dangle silently.
- **Copy-compile rule:** `lat_*.hh` / `domains.*` changes rebuild BOTH wheels, LAT
  first; a cold GBGPU rebuild takes >5 min — run rebuilds in the background.
- **Two-pytest rule:** GBGPU and LAT test dirs are both packages named `tests` —
  never mix them in one pytest invocation.
- **Byte-oracle discipline:** capture a fresh reference before every
  numbers-preserving step; never reuse a reference across intentional numeric
  changes.
- **Don't trust `use_midpoint`/`window_factor` semantics blindly** — midpoint is a
  Fresnel-anchor concept (ignore in hetfft), `window_factor` is unwindowed-path-only
  (Part A §A5).
