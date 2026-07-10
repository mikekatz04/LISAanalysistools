# Part B agent plan — het-FFT column producer, end to end (CPU scope)

**Audience:** an implementation agent picking this up cold. Follow tasks IN ORDER.
Background reading first: the design study
(`2026-07-10-stft-gb-fresnel-accuracy-and-het-fft-successor.md`, §4) and the guide
(`2026-07-10-partB-hetfft-column-implementation-guide.md`) — the guide's B-sections
carry the physics; THIS plan fixes every open choice and the exact sequence. The
environment/rebuild/gate command block is §0 of the Part A guide.

## Rules (non-negotiable)

Same rules 1–5 as the Part A agent plan (`2026-07-10-partA-agent-plan.md`), plus:
6. **Decisions are already made — do not redesign:** `prepare_source` hook (guide
   B2.1 option (a)); `N_SUB ∈ {16, 32}` compile-time, serial in-register FFT only
   (no cooperative fallback in this scope); midpoint subsampling; carrier snapped to
   `k_car = llround(f0_astro / df_stft)`; `value()` returns 0 outside the tabulated
   spectrum (swap-union clamp); one global heterodyne per source; N_cp =
   `max(8, (int)ceil(2 * span_days))` control points over the FULL domain span.
7. GPU work is OUT OF SCOPE (it is guide §B5, runs on the box). Stop after Task B-6.

## Task B-0 — baseline

Gates green + `stft_column_policy_oracle.py --capture /tmp/pre_B.npz --bench`
(exact expectations as Part A plan Task A-0). STOP if not green.

## Task B-1 — NumPy convention-locking mock

**File (new, committed):** `lisa-analysis-tools/scripts/validation/stft_hetfft_column_mock.py`

Follow guide §B1 items 1–4 exactly, with these pinned details:
- Reference: copy the `STFTEngineAccuracy.setUpClass` construction from
  `GBGPU/tests/test_stft_gb_accuracy.py` (GBTDIonTheFly → `eval_tdi` → 
  `TDSignal(...).stft(...)`), 6 h segments, `t_start = 10 d`, one source.
- The mock producer per column, per guide B1 step 2 (midpoint samples, snapped
  carrier heterodyne, per-segment Tukey, N_sub-point DFT with the half-sample
  phase `exp(-1j*np.pi*m/N_sub)` and scale `dt_seg/N_sub`, fftshift bin map).
- ALSO transliterate this exact iterative FFT into the mock and assert it matches
  `np.fft.fft` to 1e-12 on random inputs for N ∈ {16, 32} (it becomes the C++
  device FFT in Task B-3 — validate the ALGORITHM here, where testing is easy):

```python
def serial_fft(a):                      # in-place, len power of two, forward
    n = len(a); j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit; bit >>= 1
        j ^= bit
        if i < j: a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = -2.0 * np.pi / length
        wlen = complex(np.cos(ang), np.sin(ang))
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(length // 2):
                t = a[i + k + length // 2] * w
                a[i + k + length // 2] = a[i + k] - t
                a[i + k] = a[i + k] + t
                w *= wlen
        length <<= 1
    return a
```

- Two sampling variants: (i) exact TDI samples (`eval_tdi` at the subsample times)
  — acceptance: in-stencil relative field error vs the brute STFT `<= 1e-10` at
  `N_sub >= 2*(n_side+2)` (pure quadrature/DFT identity); (ii) spline-fed (CubicSpline
  through amp/dphi at N_cp control points, `dphi = unwrap(phase) - 2π f_het t`) —
  acceptance `<= 1e-7` at 2 cp/day. Print a table over
  N_sub ∈ {8,16,32} × n_side ∈ {2,4,10} × α ∈ {0, 0.3} × cp/day ∈ {1,2,4}.
- Also cross-check phase CONVENTION against the Fresnel path exactly as guide B1
  step 3 says (fill via `STFTGBComputations.fill_global_stft`, factor 1, compare
  2× the filled pixel): complex agreement to ~2e-3 field (pre-A1 Fresnel accuracy)
  proves the conj/0.5/origin conventions. If |values| match but phases differ →
  fix the mock's conjugation/origin BEFORE any C++.

**CHECKPOINT COMMIT** (`scripts/validation/stft_hetfft_column_mock.py`):
`feat(validation): numpy mock of the het-FFT STFT column producer (conventions locked)`
— body: the acceptance table.

## Task B-2 — `prepare_source` hook on the seam (byte-identical)

**File:** `lisa-analysis-tools/src/lisatools/cutils/lat_stft_kernels.hh` only.

- Add to `FresnelColumn`: a `struct SourceWorkspace {};` and
  `CUDA_DEVICE static void prepare_source(SourceWorkspace&, SourceT&, STFTFresnel*,
  STFTDomain*, double* params, int* lsr, int* lse, int bin_i) {}` (empty).
- In BOTH eval blocks and the fill kernel, declare
  `typename ColumnT::SourceWorkspace src_ws;` next to the existing `State`
  declarations and call `ColumnT::prepare_source(src_ws, ...)` ONCE per bin —
  immediately after the existing `src.get_sky_vectors(...)` call (ll/fill) and
  after the second `get_sky_vectors` in the swap block — then pass `src_ws` as a
  new first argument to `ColumnT::setup(...)` (FresnelColumn::setup ignores it).
  NOTE (GPU semantics for later policies): every thread constructs/calls on its
  own `src_ws` copy here; a policy that needs SHARED per-source storage will take
  CUDA_SHARED buffers via kernel-level scratch instead — document this in the
  seam comment block.
- Rebuild both wheels. Oracle `--compare /tmp/pre_B.npz` MUST be BYTE-IDENTICAL;
  full gates; bench within ±5%.

**CHECKPOINT COMMIT**: `refactor(gb-stft): prepare_source hook on the column seam (byte-identical)`.

## Task B-3 — `HetFFTColumn` (new header, CPU-correct)

**Files:** new `lisa-analysis-tools/src/lisatools/cutils/lat_stft_hetfft.hh`;
one `#include "lat_stft_hetfft.hh"` added at the end of the include block of
`lat_stft_kernels.hh`.

Implement per guide §B2.2–B2.3 with these pinned specifics:
- `template <class SourceT, int N_SUB> struct HetFFTColumn` with
  `SourceWorkspace` holding: `double t_cp[N_CP_MAX]`, per-channel
  `amp_y/amp_c1/amp_c2/amp_c3` and `dphi_y/dphi_c1/dphi_c2/dphi_c3`
  (`[3*N_CP_MAX]` each), `double B_buf[N_CP_MAX]`, `int n_cp`, `double f_het`,
  `int k_car`; `N_CP_MAX = 256`.
- `prepare_source`: sample times uniform over `[stft->t0, stft->t0 +
  num_times*dt]`; ONE call `src.get_tdi_raw(tdi_buf, phi_ref_buf, params, t_cp,
  n_cp, bin_i, 3)` (`lat_tdi_on_the_fly.hh`; stack buffers `cmplx tdi_buf[3*N_CP_MAX]`,
  `double phi_ref_buf[N_CP_MAX]`); per channel: `amp = abs`, `phase =
  arg(conj(tdi)) + phi_ref`?? — NO: derive the de-rotated phase EXACTLY as the
  chunked-het does — read `fast_wdm_inner_heterodyne_spline`
  (`lat_chunked_het_kernels.hh:483`) and copy its amp/phase/de-rotation block
  (steps: per-channel complex → amp/phase, add phi_ref handling as done there,
  subtract `2π f_het t`, unwrap along cp index). Do NOT invent the convention —
  transliterate that block, then the B-1 mock parity test judges it.
  Spline fits via `wdm_fit_cubic_spline(x, y, c1, c2, c3, B_buf, /*pcr*/ nullptr,
  n_cp, <same spline_type value used at the fast_wdm_inner_heterodyne_spline call
  site — read it there>)` per channel per quantity.
- `State`: `cmplx spec[3][N_SUB]; int carrier_j; int m_lo;`
- `setup`: per guide (midpoint samples from the splines, per-segment window vector,
  residual `exp(i dphi)` via `gcmplx::polar(amp, dphi)` v1, the serial FFT below,
  `dt_seg/N_SUB` scale + half-sample phase + FT origin from the B-1 mock, fftshift
  `m_lo = carrier_j - N_SUB/2`). The per-segment Tukey vector: compute inline per
  sample from `fresnel->window_alpha` with the SAME formula the brute STFT window
  uses (`scipy.signal.windows.tukey` definition — copy the closed form from the
  gbfd slow-part window in `gb_tdi_on_the_fly.cu` (search "tukey") which already
  encodes it in C++).
- `value(s, j, freq_j_here, freq_here)`: `(void) freq_here;`
  `int m = freq_j_here - s.m_lo; if (m < 0 || m >= N_SUB) return cmplx(0.,0.);`
  return the fftshift-mapped entry.
- The device FFT: transliterate `serial_fft` from Task B-1 EXACTLY (same loop
  structure, `cmplx`/`gcmplx` ops), `template <int N> CUDA_DEVICE inline void
  hetfft_serial_fft(cmplx* a)`.
- CPU compile check only at this point: rebuild LAT (GBGPU not needed yet — nothing
  instantiates the policy). Header must compile standalone
  (it is included by the kernels TU): rebuild both anyway and run the oracle
  `--compare` (byte-identical — nothing instantiates HetFFTColumn yet) + gates.

**CHECKPOINT COMMIT**: `feat(gb-stft): HetFFTColumn policy header (not yet wired)`.

## Task B-4 — wiring: `column_policy` switch

**Files:** `GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu` (the six
`stft_*_impl<GBTDIonTheFly>` call sites in the `gb_stft_*_wrap` functions),
`GBGPU/src/gbgpu/cutils/binding_gbgpu.{hpp,cxx}` (add `int column_policy` — LAST
positional arg of each `gb_stft_*` method), `GBGPU/src/gbgpu/gbcomps.py`
(`STFTGBComputations.__init__(..., column_policy="fresnel")`, map
{"fresnel":0, "hetfft16":1, "hetfft32":2}, forward from every `gb_stft_*` call).

Dispatch pattern at each wrap site (0 must remain the default → identical behavior):

```cpp
switch (column_policy) {
  case 0: stft_get_ll_impl<GBTDIonTheFly>( ... ); break;
  case 1: stft_get_ll_impl<GBTDIonTheFly, HetFFTColumn<GBTDIonTheFly, 16>>( ... ); break;
  case 2: stft_get_ll_impl<GBTDIonTheFly, HetFFTColumn<GBTDIonTheFly, 32>>( ... ); break;
  default: throw std::invalid_argument("column_policy");
}
```

Rebuild both. Oracle `--compare /tmp/pre_B.npz` byte-identical (default 0);
full gates green.

**CHECKPOINT COMMIT**: `feat(gb-stft): column_policy switch (fresnel default, hetfft16/32)`.

## Task B-5 — validation ladder

New test file `GBGPU/tests/test_stft_hetfft.py` implementing guide §B4 rungs 2–4
(policy parity vs Fresnel on a near-monochromatic source; brute-STFT in-stencil
interior accuracy `< 1e-6` at N_SUB=32/α=0.3/n_side=10 — use `< 1e-6` not 1e-7 as
the ASSERT, record the actual; aliasing cliff: mm flat for `N_SUB >= 2*(n_side+2)`,
degrades at N_SUB=8/n_side=10). Then:
- trio still EXACTLY `23 passed, 2 subtests`; LAT set `37 passed, 1 skipped`;
- flow smoke with the policy on: run `test_gbspecial_flow_stft.py` once with
  `STFTGBComputations(..., column_policy="hetfft16")` patched into the fixture
  (scratch edit, revert after; record the 4/4 pass);
- oracle `--bench` and a hetfft bench: duplicate the bench block in a scratch copy
  with `column_policy="hetfft16"` — record get_ll/swap/fill medians vs Fresnel at
  n_side 2 and 10 (expect parity-or-better at 2, clearly better at 10).

**CHECKPOINT COMMIT** (tests only):
`test(gb-stft): het-FFT policy validation ladder (parity, brute-STFT accuracy, aliasing cliff)`.

## Task B-6 — wrap-up

Append a review block to `tasks/todo.md` (numbers: accuracy table, bench table),
update the workspace `HANDOFF.md` repo-state + a 5-line summary, list the GPU
follow-ups verbatim from guide §B5 as the next steps. Do NOT push. STOP.
