# MBH merger interpolation deep-dive — amp/phase interpolation EXONERATED

Goal (user): *eliminate or convict interpolation* as the cause of the residual
MBH TDI-on-the-fly error (the merger-time <5e-4 Hz "red" transient, worst at
near-equal mass q~2 — see `../mbh_td_residual/`). Standalone; installed code not
modified. Runner: `../../runners/mbh_merger_interp_study.py`.

## Method

The TOF (`TDSplineTDIWaveform`) evaluates PhenomTHM (`phentax.IMRPhenomTHM`) on
phentax's **adaptive** coarse grid and cubic-splines amp & phase per mode to the
output grid before the TDI. We replicate that, *isolated from the response*:
reconstruct `h(t)=Sum_m amp_m e^{i phase_m}` on a common 2.5 s grid from knots at
several `coarse_graining_scale_factor` (cf), with several interpolants, and
mismatch against the finest grid (cf=96) as truth.

Notes learned:
- `coarse_graining_scale_factor` runs **opposite** to intuition: higher cf =
  MORE knots = finer. phentax enforces cf>=8; the **stock run uses cf=48**; cf=8
  is the coarsest allowed (worst case for interpolation).
- phentax's coarse grid is **adaptive** — sub-second knots at the merger, up to
  ~240 s in the inspiral — so the sharp (low-q) merger peak is densely sampled,
  not starved (see the right panel: 285 cf=8 knots resolve the q=1.2 peak).

## Result (mismatch vs cf=96 truth)

| q | stock cf=48, cubic | cf=8 cubic | cf=8 quintic | cf=8 akima | cf=8 pchip |
|-----|-----------|----------|----------|----------|----------|
| 1.2 | 2e-15 | 1.4e-12 | 1.1e-12 | 1.9e-11 | 5.1e-12 |
| 1.5 | 2e-16 | 7.0e-12 | 5.9e-12 | 2.4e-11 | 1.3e-11 |
| 2.0 | 9e-16 | 2.6e-11 | 2.1e-11 | 4.4e-11 | 4.0e-11 |
| 3.0 | 4e-16 | 6.4e-15 | 5.1e-15 | 1.0e-11 | 2.1e-12 |
| 5.0 | 2e-15 | 1.8e-10 | 1.5e-10 | 2.0e-10 | 2.5e-10 |
| 9.0 | 7e-16 | 6.8e-15 | 5.6e-15 | 7.9e-13 | 2.3e-13 |

**Worst case over all q and all methods = 2.5e-10.** Observed TOF null mm is
~1e-3 at low q. Gap: ~4e6x worst-case, ~1e12x at the actual stock cf=48.

## Verdict

- **Amp/phase interpolation is NOT the cause.** Even the coarsest allowed grid
  (cf=8) with cubic reproduces the converged waveform to mm ~1e-12 at the
  sharpest (q=1.2) merger. The stock cf=48 is fully converged (~1e-15).
- **Interpolation method is irrelevant here.** Cubic is already effectively
  exact; quintic is marginally better, and **Akima is marginally WORSE** (its
  C1 kinks cost accuracy where the function is smooth). No method change helps.
- The user's "sharp low-q merger breaks the interpolation" hypothesis is sound in
  spirit — low q IS the sharper merger — but phentax's adaptive knot placement
  already resolves it, so the sharpness does not translate into interpolation
  error.

## What remains (the residual error is elsewhere)

Two interpolations exist in the TOF; this eliminates the first (amp/phase). Still
open: the **TDI-delay / orbit-position spline** (the response evaluates
`h(t - delay(t))` on the coarse grid) and the response computation itself. The
decisive next test is a **cf-sweep of the full TOF TDI output** (build the stock
`PhenomTHMTDIOnFlyWaveform` at cf=8 vs 48 vs 96 and compare the TDI, not just the
strain): if cf-insensitive like the strain, ALL interpolation is eliminated and
the residual is a genuine response/waveform-model difference vs the mojito data
(low-q, merger-localized); if it moves with cf, the delay spline is the culprit.

Files: `mbh_merger_interp_study.png`, `interp_study.log`.
