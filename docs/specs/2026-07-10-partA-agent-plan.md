# Part A agent plan — Fresnel accuracy work, end to end

**Audience:** an implementation agent picking this up cold. Follow tasks IN ORDER.
Background reading (do read, don't improvise beyond it):
`2026-07-10-stft-gb-fresnel-accuracy-and-het-fft-successor.md` (design, §2–3) and
`2026-07-10-partA-fresnel-accuracy-implementation-guide.md` (the guide; its §0 has the
environment, rebuild and gate commands used below — memorize that section).

## Rules (non-negotiable)

1. Touch ONLY the files each task lists. No drive-by refactors, no formatting sweeps.
2. Never relax a test tolerance, never delete a test, never push. Commit exactly at
   the marked checkpoints with the given message prefix.
3. Every C++ change: rebuild BOTH wheels (guide §0), then run the FULL gate block
   (trio + LAT set + repro) and the oracle bench. Record numbers in the commit body.
4. If a gate fails and the fix is not obvious within 2 attempts: `git checkout` your
   change away, write what you saw into `tasks/todo.md`, STOP the task, move nothing.
5. Numeric work happens in a NumPy mirror FIRST, validated against scipy; C++ is a
   transliteration of a validated mirror, never fresh derivation.
6. Scope: Tasks A-0 .. A-3 only. A3 (cubic phase) and A4 (global taper kernel work)
   from the guide are OUT OF SCOPE — do not start them.

## Task A-0 — baseline (no changes)

Run: gates (guide §0) + `stft_column_policy_oracle.py --capture /tmp/pre_A.npz --bench`.
EXPECT: `23 passed, 2 subtests`; `37 passed, 1 skipped`; repro lines all `OK`;
bench prints two lines. If ANY differs → STOP (report; the tree is not at baseline).

## Task A-1 — precise Fresnel integrals

**Files:** `lisa-analysis-tools/src/lisatools/cutils/domains.cu` (function
`STFTFresnel::get_fresnel_integrals` + the two helpers it replaces) and a new scratch
mirror script (not committed).

**Recipe (all constants pre-verified; do not refit):** replace the body with three
branches on `ax = |x|` (keep the final sign flip for `x < 0` exactly as-is):

1. `ax <= 1.6` — Maclaurin series with term recurrences, stop at `|term| < 1e-17*|sum|`:
   `C = Σ (-1)^n (π/2)^{2n} ax^{4n+1} / ((2n)!(4n+1))`,
   `S = Σ (-1)^n (π/2)^{2n+1} ax^{4n+3} / ((2n+1)!(4n+3))`.
2. `1.6 < ax < 8` — auxiliary form (same structure as today:
   `S = 0.5 - f*cos(arg) - g*sin(arg)`, `C = 0.5 + f*sin(arg) - g*cos(arg)`,
   `arg = 0.5*π*ax²`) with `u = 1/ax²` and rational fits (Horner, coefficients
   ascending in u; max rel err 8.7e-10 (f) / 2.2e-9 (g), i.e. mm impact ~1e-18):

   f = (1/(π·ax)) · P_F(u)/Q_F(u)
   P_F: 9.9999998788975186e-01, 5.6393336781284269e+00, 1.6530235844406874e+01,
        9.9741235553347174e+00, -1.2714841072845186e+01, -2.8568630009945513e+00
   Q_F: 1.0000000000000000e+00, 5.6393317184256810e+00, 1.6834326314502963e+01,
        1.1683893414197762e+01, -8.5850028996594485e+00, -6.5764635236542199e+00

   g = (1/(π²·ax³)) · P_G(u)/Q_G(u)
   P_G: 9.9999994455231545e-01, 5.6691485084817055e+00, 2.2643028376343988e+01,
        1.4001161175318936e+01, -2.1885461791121589e+01, 1.1588596586401721e+01,
        3.2177684059265470e+00
   Q_G: 1.0000000000000000e+00, 5.6691386346869024e+00, 2.4163564112837470e+01,
        2.2588896377169348e+01, 5.8164191581105191e+00, -1.9482374004004097e+01,
        1.9504389595032652e+01

3. `ax >= 8` — asymptotic series, 5 terms, `w = 1/(π·ax²)` (truncation ≤1e-13 here):
   `f = (1/(π·ax)) Σ_{m=0}^{4} (-1)^m (4m-1)!! w^{2m}`   ((-1)!! = 1)
   `g = (1/(π·ax)) Σ_{m=0}^{4} (-1)^m (4m+1)!! w^{2m+1}`
   Compute the ten double-factorial constants ((4m∓1)!! for m = 0..4) in the NumPy
   mirror (step a below) and paste the PRINTED values as C++ literals — do not
   write them from memory.

**Order of work:**
a. Write the NumPy mirror (all three branches) in a scratch file; acceptance:
   `max |ΔC|, |ΔS| < 2e-9` vs `scipy.special.fresnel` on `x ∈ [0, 1000]`, 2e5 log+lin
   points, both signs, plus branch-boundary continuity `< 1e-11` at 1.6 and 8.
   Print and record the max errors AND the double-factorial constants.
b. Transliterate to `domains.cu` (keep the function signature; delete
   `get_auxiliary_f/g` or leave them unused — prefer delete + update the accuracy
   NOTE comment above the function to describe the new scheme and its ≤2e-9 bound).
c. Rebuild both wheels; run gates. EXPECT trio/LAT/repro unchanged
   (`23 passed, 2 subtests` / `37 passed, 1 skipped` / all OK).
d. Oracle: `--compare /tmp/pre_A.npz` MUST FAIL (numbers change by design) with
   small diffs; sanity: reported `max rel` diffs ≲ 1e-5 across arrays (the old fits
   were 2e-3-accurate per value; if you see O(1) rel diffs you broke a branch).
   Then `--capture /tmp/post_A1.npz --bench`; bench within ±10% of A-0.
e. Accuracy movement check (the point of the task): run the trio's accuracy file
   with a strong window — copy `GBGPU/tests/test_stft_gb_accuracy.py`'s
   `STFTEngineAccuracy` setup into a scratch run with `window_alpha=0.5` on the
   `STFTFresnelWrap` and the SAME Tukey window on the data STFT
   (`scipy.signal.windows.tukey(nperseg, 0.5)`), n_side=10: the in-stencil interior
   mismatch must come out `< 1e-6` (it saturated at ~5e-6 before this task).
   Record the before/after numbers.

**CHECKPOINT COMMIT** (only `domains.cu`):
`feat(gb-stft): precise Fresnel integrals (series + fitted rationals + asymptotics, <=2e-9/value)`
— body: mirror max errors, gate results, bench, the e) numbers.

## Task A-2 — hygiene trio

**Files:** `GBGPU/src/gbgpu/gbcomps.py`, `lisa-analysis-tools/src/lisatools/cutils/domains.cu`.

a. `STFTGBComputations.__init__`: add kwargs `window_alpha=None`, `use_midpoint=None`
   — when not None, ASSERT they equal the bound group's values
   (`self.stft_comps.cpp_fresnel` has no getters? then compare against
   `stft_comps.window_alpha` / `.use_midpoint` attributes if present, else store and
   document as informational). Keep behavior identical when unset.
b. Docstring both in `gbcomps.py` (`window_factor` arg) and as a comment in
   `domains.cu::get_fourier_value`: `window_factor` acts ONLY on the unwindowed
   (`window_alpha == 0`) path; it is ignored when windowed.
c. Python-only → no rebuild; run the LAT gate + trio once.
   Oracle `--compare /tmp/post_A1.npz` must PASS (byte-identical).

**CHECKPOINT COMMIT**: `docs(gb-stft): window_factor/window_alpha surface hygiene`.

## Task A-3 — linear-envelope correction (flag-gated)

Follow the guide's A2 section for the physics; the agent-level order is fixed:

a. **NumPy mirror first**: implement `fourier_value(amp, phase0, f0, fdot0, t0, f)`
   (unwindowed) AND its analytic `d/df` in NumPy from the formulas in
   `domains.cu` (`get_zeta/get_v/get_phase_kernel_product`); validate the derivative
   against a central finite difference in `f` (step `1e-4*df_stft`, rel agreement
   `< 1e-6` across a grid of (f0, fdot0, f) drawn like the oracle's params).
   THEN add the windowed variant (the 7-term loop): d/df of each sub-interval term;
   same FD validation. Do not proceed until both pass.
b. Estimator export: in `lat_stft_kernels.hh::stft_freq_fdot_from_tdi_phase`, add two
   out-params `double* amp_p, double* amp_m` filled with `abs(tdi_p[ch])` /
   `abs(tdi_m[ch])` (3 each) and thread them through `stft_pixel_freq_fdot` and
   `FresnelColumn::setup` (all in the same header). Slope per channel:
   `a[j] = (amp_p[j] - amp_m[j]) / (2*D*amp[j])` with `D` also exported. Astro
   fallback path: set `a[j] = 0`.
c. `STFTFresnel` gains `bool linear_envelope` (default false; constructor +
   `STFTFresnelWrap` binding in `binding_domains.hpp` + plumb from
   `STFTComputationGroup`'s `domain_group_kwargs` in
   `lisa-analysis-tools/src/lisatools/domaincomputation.py`). `FresnelColumn::value`
   adds `a[j] * (first-moment term)` only when the flag is on (transliterate the
   validated mirror).
d. Rebuild both; flag OFF: oracle `--compare /tmp/post_A1.npz` byte-identical + full
   gates + bench (≤5% drift). Flag ON: scratch accuracy run at `stft_dt = 86400`
   (24 h segments, the harness from A-1e): interior in-stencil mismatch must improve
   from ~8e-5 to `< 1e-5`; at 6 h from ~7e-6 (post-A1 value) toward ~1e-6. Record.

**CHECKPOINT COMMIT** (header + domains + binding + domaincomputation.py + any new
test): `feat(gb-stft): linear-envelope correction (flag-gated, byte-identical off)`.

## Done criteria for this plan

- Three checkpoint commits, each with recorded numbers.
- Final state: gates green, oracle byte-identical vs `/tmp/post_A1.npz` with the
  envelope flag off, bench within budget, accuracy records in the commit bodies.
- Update `tasks/todo.md` (append a review block) and the workspace `HANDOFF.md`
  repo-state table. Do NOT push.
