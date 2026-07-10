# Part A implementation guide — lowering the Fresnel mismatch, step by step

**Companion to:** `2026-07-10-stft-gb-fresnel-accuracy-and-het-fft-successor.md` (the
design study; read §2–§3 first). This file is the OFFLINE, step-by-step execution guide
for Part A. Every step is independent unless stated; do them in order of appearance
(they are ranked by gain/effort).

**Prerequisite:** the column-producer seam, commit `c5cd45a`
(`FresnelColumn` policy in `src/lisatools/cutils/lat_stft_kernels.hh`). All anchors
below are post-seam. Function names, not line numbers, are authoritative.

---

## 0. Environment + gates cheat-sheet (used by every step)

Work from the workspace root (the directory containing `lisa-analysis-tools/`,
`GBGPU/`, `.venv/`). All Python via uv:

```sh
PY () { VIRTUAL_ENV=$PWD/.venv uv run --no-project python "$@"; }
```

Rebuild (needed after ANY C++ change; `lat_stft_kernels.hh` / `domains.{hpp,cu}`
rebuild BOTH wheels — GBGPU copy-compiles LAT's cutils; LAT FIRST; ~5–10 min):

```sh
VIRTUAL_ENV=$PWD/.venv PKG_CONFIG_PATH=/opt/homebrew/opt/lapack/lib/pkgconfig:/opt/homebrew/opt/openblas/lib/pkgconfig \
  uv pip install -e ./lisa-analysis-tools --no-build-isolation --no-deps --no-cache
VIRTUAL_ENV=$PWD/.venv PKG_CONFIG_PATH=/opt/homebrew/opt/lapack/lib/pkgconfig:/opt/homebrew/opt/openblas/lib/pkgconfig \
  uv pip install -e ./GBGPU --no-build-isolation --no-deps --no-cache
```

Gates (two pytest invocations — both test dirs are packages named `tests`):

```sh
PY -m pytest GBGPU/tests/test_stft_gb.py GBGPU/tests/test_stft_gb_accuracy.py \
   GBGPU/tests/test_stft_gb_crossdomain.py -q          # trio: 23 passed + 2 subtests
PY -m pytest lisa-analysis-tools/tests/test_gb_likelihood_engine.py \
   lisa-analysis-tools/tests/test_band_view.py \
   lisa-analysis-tools/tests/test_aca_vectorized_dispatch.py \
   lisa-analysis-tools/tests/test_gbspecial_flow.py \
   lisa-analysis-tools/tests/test_gbspecial_flow_stft.py -q   # 37 passed, 1 skipped
PY repro_stft_fill_then_ll_bug.py                      # all OK
```

Byte-oracle + CPU bench (`lisa-analysis-tools/scripts/validation/stft_column_policy_oracle.py`):

```sh
PY lisa-analysis-tools/scripts/validation/stft_column_policy_oracle.py --capture /tmp/pre.npz --bench
# ... make the change, rebuild ...
PY lisa-analysis-tools/scripts/validation/stft_column_policy_oracle.py --compare /tmp/pre.npz --bench
```

Use `--compare` (byte-identity) ONLY for refactor-style steps. Steps that change
NUMBERS on purpose (A1–A4) must instead show the *expected accuracy movement* in the
measurement harnesses:
- `GBGPU/tests/test_stft_gb_accuracy.py` (template vs brute STFT; edit-run locally
  with your alpha/n_side of interest),
- `lisa-analysis-tools/scripts/validation/gb_mojito_stft_fd_mismatch.py --selftest`
  (or with mojito data: full / in-stencil / stencil-interior decomposition per source).

Per-source rule of thumb linking field error to mismatch: mm ≈ ε²/2 for an
incoherent relative field error ε.

---

## A0. Knob changes (no code, no rebuild)

**Goal.** Move production defaults off the sampler-era settings.

1. `n_side_bins`: production default is 2 (`STFTGBComputations.__init__`,
   `GBGPU/src/gbgpu/gbcomps.py`). For fidelity work set 10–25 at the call site or via
   the run config (`STFT_N_SIDE_BINS` env knob in
   `global_fit_input/gb_no_foreground_global_fit_settings.py`). Expected truncation
   floor at Tukey(0.1): 10 → 1.25e-3, 15 → 4.5e-4, 20 → 5.2e-5, 25 → ~1e-5.
2. Tukey `window_alpha`: keep DATA and EVALUATOR on one knob (`STFT_WINDOW_ALPHA` in
   the settings file drives `acs.domain_group_kwargs`; the group builds the
   `STFTFresnelWrap` with it). α 0.1 → 0.5 moves the leakage floor 1.25e-3 → ~2e-6 at
   the SNR² cost 1−5α/8.
3. `freq_from_tdi_phase=True` (default — keep), `use_midpoint=True` (group knob
   `STFT_USE_MIDPOINT`): measured 2.09e-5 → 1.48e-5 on the 32-seg grid test.
4. Observation interior to the orbit span (avoid the first/last-segment edge
   artifact: pinning at the orbit t0 inflates full-grid mm ~60×).

**Validate.** `gb_mojito_stft_fd_mismatch.py` per-source table before/after;
no gates change (knobs only). **Ceiling of this step alone: ~1e-5..2e-6.**

---

## A1. Precise Fresnel integrals (the ~2e-6 saturation killer)

**Goal.** Replace the A&S 7.3.27/28 rational fits (per-value error ~2e-3 ⇒ mm floor
~2e-6) with an evaluation good to ≤1e-9 per value.

**Files.** `lisa-analysis-tools/src/lisatools/cutils/domains.cu` only:
`STFTFresnel::get_fresnel_integrals` (+ its helpers `get_auxiliary_f`,
`get_auxiliary_g`). The callers (`get_fresnel_kernel_interval`) are unchanged.

**Change.** Two-branch evaluation of C(x), S(x) (defined per A&S 7.3.1/7.3.2, the
convention the current code already returns):

- **|x| ≤ x_split (≈ 1.6–1.8): Maclaurin series** (converges fast, ~12–16 terms to
  <1e-16):

  C(x) = Σ_{n≥0} (−1)^n (π/2)^{2n} x^{4n+1} / [(2n)! (4n+1)]
  S(x) = Σ_{n≥0} (−1)^n (π/2)^{2n+1} x^{4n+3} / [(2n+1)! (4n+3)]

  Implement with a term recurrence (multiply by −(π/2)² x⁴ / [(2n+1)(2n+2)] for C,
  analogous for S), stop when |term| < 1e-17·|sum|.

- **|x| > x_split: auxiliary-function form** (keep the existing structure —
  C = 0.5 + f·sin(πx²/2) − g·cos(πx²/2), S = 0.5 − f·cos(πx²/2) − g·sin(πx²/2)) but
  with HIGH-ORDER rational approximations for f(x), g(x) (Boersma-1960-class or a
  freshly fitted minimax pair; ~1e-9 or better is easy with degree-10/10 rationals in
  u = 1/x² over x ∈ [1.6, ∞)).

**Do not transcribe coefficient tables from memory.** Generate + verify them:

```python
# fit_fresnel_aux.py -- run once, paste the printed coefficients into domains.cu
import numpy as np
from scipy.special import fresnel          # S, C convention: fresnel(x) = (S, C)
x = np.linspace(1.6, 60.0, 20000)
S, C = fresnel(x)
arg = 0.5 * np.pi * x**2
# invert the auxiliary form for the exact f, g on the grid
# (from C = 0.5 + f sin - g cos, S = 0.5 - f cos - g sin):
f = (0.5 - S) * np.cos(arg) + (C - 0.5) * np.sin(arg)
g = (0.5 - S) * np.sin(arg) - (C - 0.5) * np.cos(arg)
# sanity: f ~ 1/(pi x) and g ~ 1/(pi^2 x^3) > 0 at large x; g's leading
# 1/x terms cancel by construction (double precision leaves ~1e-13 abs
# noise on g at x ~ 50 -- fine for a 1e-10 fit target).
u = 1.0 / x**2
# least-squares rational fit f ~ x * P(u)/Q(u) etc.; iterate degree until
# max relative error < 1e-10 on the grid (numpy.polynomial or scipy).
```

**Acceptance test (add to the trio's accuracy file or a new unit test):**
max |ΔC|, |ΔS| < 1e-9 against `scipy.special.fresnel` on x ∈ [0, 50] (10⁵ points,
both signs), plus continuity at x_split < 1e-12.

**Validate.**
- The oracle is NOT byte-identical (numbers change by design) — capture a NEW
  reference after this step for later refactors.
- `test_stft_gb_accuracy`-style check at α ≥ 0.5: the 5–7e-6 saturation must drop
  below ~1e-7 (then envelope-limited).
- Trio + LAT gates green (tolerances are one-sided; improved accuracy passes).
- `--bench`: cost within noise of before (series/rational ≈ same op count as the old
  fits + the same sincos).

**Commit message suggestion:** `feat(gb-stft): precise Fresnel integrals (<=1e-9/value), removes the ~2e-6 mm saturation`.

---

## A2. Linear-envelope correction (kills the const-envelope error)

**Goal.** The kernel freezes each channel's amplitude at the column anchor; the true
TDI amplitude drifts across the segment (measured mm: ~7e-6 @ 6 h → ~8e-5 @ 24 h).
Model A_j(τ) ≈ A_j·(1 + a_j τ) per column and add the analytic first-moment term.

**Where the slope comes from (free).** The estimator stencil already evaluates the
TDI at t ± D (`stft_freq_fdot_from_tdi_phase`, `lat_stft_kernels.hh`): per channel,
a_j = (|z₊_j| − |z₋_j|) / (2 D A_j). Export |z±| per channel from the estimator
(extend its out-params) and store `a[3]` in `FresnelColumn::State` — this is now a
LOCAL policy change thanks to the seam; the consumers don't move.

**The first-moment term.** With F(f) = ∫ e^{i(φ0 + 2π f0 τ + π fdot0 τ² − 2π f τ)} dτ
over the segment (what `get_fourier_value` returns up to the amplitude prefactor),

  ∫ τ e^{i(...)} dτ = (i / 2π) ∂F/∂f.

∂F/∂f is analytic through the existing pieces: ζ = (f0−f)/fdot0 gives
∂v/∂f = −√(2|fdot0|)/fdot0 at both endpoints, and C′(v) = cos(πv²/2),
S′(v) = sin(πv²/2) — the SAME sincos values the kernel evaluation already computes
inside `get_fresnel_integrals` (expose them or recompute: 2 extra sincos per pixel).
Also differentiate the stationary-phase prefactor (−πfdot0ζ² term: ∂/∂f = 2πζ) — one
extra multiply. Assemble:

  value_corrected = (1 + a_j·τ̂) path = value + a_j · (i/2π) ∂(value)/∂f
  (with the τ origin matching the evaluator's t_ref anchoring — midpoint vs start;
  re-derive the τ-shift constant for use_midpoint exactly as
  `get_phase_kernel_product` does for the zeroth moment.)

**Gating.** Add a `linear_envelope` bool on `STFTFresnel` (wrap-plumbed, default
false) so a_j = 0 reproduces today's path BYTE-IDENTICALLY (oracle `--compare` with
the flag off must pass; that is the refactor-safety part of this step).

**Validate.**
- Flag off: oracle byte-identical; gates green.
- Flag on: 24 h-segment accuracy check (the harness in `test_stft_gb_accuracy`
  with `stft_dt = 86400`): interior mm ~8e-5 → target <1e-5;
  6 h: 7e-6 → ~1e-6-class. Windowed path: verify the correction composes with the
  7-term decomposition (each sub-interval term gets its own first moment — the
  τ-window is per sub-interval; implement inside `get_windowed_fourier_value`'s loop).
- Bench: expect ≤1.5× per-pixel cost with the flag on; unchanged off.

---

## A3. Cubic-phase correction (long segments / high f0 only)

**Goal.** Within-segment phase curvature beyond the quadratic (Doppler rate drift,
fddot). First-order perturbation: e^{i(π/3)φ⃛τ³} ≈ 1 + i(π/3)φ⃛τ³ ⇒ a third-moment
term = (i/2π)³-flavored second derivative ∂²F/∂f² (same machinery as A2, one order
deeper; reuse the exposed endpoint sincos).

**φ⃛ estimate.** The 3-point stencil gives only f, fdot. Options (pick 1):
(a) widen the stencil to 5 points (t ± D, t ± 2D → +2 response evals per column), or
(b) analytic: φ⃛ ≈ 2π·(dDoppler-rate/dt from the orbit spline) + 2π fddot — cheaper,
approximate.

**Gate the step:** only implement after measuring, with A2 in, that the residual
scales like Δ³ (run the accuracy harness at Δ = 6/12/24 h). If production stays at
6 h segments this step is likely unnecessary (per-column method error already
~1e-6-class after A1+A2).

**Validate.** Same harnesses as A2; `cubic_phase` flag defaulting off with
byte-identical oracle when off.

---

## A4. Whole-span data taper (config-dependent correctness fix)

**Goal.** If the engine's data pipeline applies a global Tukey
(`GeneralSettings.window_taper_duration`) to the stream that gets STFT'd, the
evaluator currently ignores it entirely (measured cost of an unmodeled global taper:
~1e-2). Decide ONE of:
- (preferred) **don't inherit the global taper on the STFT stream** — engine-side
  config; document in the settings file; nothing to change in the kernel; or
- model it: add `global_taper_alpha`, `global_T`, `global_t0` to `STFTFresnel`
  (constructor + `STFTFresnelWrap` + `STFTComputationGroup` plumbing), have
  `FresnelColumn::setup` compute w_glob(t_seg + dt/2) once per column, and multiply
  in `value()` (exact to the const-envelope order already assumed; the few
  ramp-overlap segments can use the exact Tukey×Tukey product later if needed).

**Validate.** Extend `gb_mojito_stft_fd_mismatch.py` (data-STFT path) to apply the
same global taper to the stream, then check mm(stft|data) recovers the untapered
level. Byte-oracle: identical with alpha_global = 0.

---

## A5. Hygiene (fold into whichever step touches the area first)

1. `window_factor` is applied ONLY on the unwindowed path
   (`get_fourier_value`'s rectangular branch); once `window_alpha > 0` it silently
   does nothing. Either thread it through `get_windowed_fourier_value` or assert
   `window_factor == 1.0` when windowed, and say so in the docstrings
   (`STFTGBComputations.__init__`).
2. Expose `window_alpha` (and `use_midpoint`) on `STFTGBComputations.__init__` so
   comp and group cannot drift — today they are group-only
   (`domain_group_kwargs`). Pure Python; oracle unaffected.
3. When editing near the estimator: the quarter-cycle stencil constants
   (`STFT_FREQ_FDOT_STENCIL_CYCLES`, `STFT_FREQ_FDOT_DT_MAX`) encode a measured
   roundoff-vs-curvature budget (see the header comment) — do not "simplify" them.

---

## Expected end state of Part A

| configuration | expected per-source mm |
|---|---|
| A0 knobs (n_side 20–25, α 0.3–0.5, midpoint, interior) | ~1e-5 .. 2e-6 |
| + A1 | ~2e-6 → envelope-limited (~1e-6 @ 6 h) |
| + A2 (+A3 for long segments) | **~1e-7 interior** |

Cost: grows ~linearly with `n_side_bins` on a ~22-sincos/pixel windowed path (+≤1.5×
from A2). If that cost matters at wide stencils, switch to the Part B producer — the
seam makes both coexist.
