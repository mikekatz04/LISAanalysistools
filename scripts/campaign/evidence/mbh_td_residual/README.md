# MBH null residual — WDM time-frequency localization (legacy vs TDI-on-the-fly)

Follow-up to `../mbh_response_study/`: *where in the waveform* does the stock MBH
template fail to null the mojito data, inspiral (early / low-freq) vs
merger-ringdown (late), and how does it differ between the **legacy** phentax and
**TDI-on-the-fly** responses?

`erebor.full_year_combined`'s analysis domain is WDM (time x frequency), so the
noise-weighted `<r|r>` that drives the mismatch is a per-(f,t) element sum. We build
the residual through the stock objects (`setup_acs` data-only -> `signal_gen`
template) and decompose that exact `<r|r> = 4 dphi Sum_ij Re[r_i* r_j invC_ij]` per
(f,t). `validate_ratio = 1.00000` for all 12 runs -> the split IS the stock
likelihood, not an approximation. Runner: `../../runners/mbh_td_residual.py`
(extract|analyze); overnight sequencer: `../../runners/run_td_overnight.sh`.

Local mojito MBH L1 files are only 0,1,16,17,18,19 (id4 — the 2x2 worst — is on the
other laptop). CHOP_WINDOW: 28.9-d active window, merger at t-layer 53 (24.7 d, ~85%).

## Per-source split (ordered by mass ratio q)

| id | M_tot | q | pop | legacy mm | TOF mm | TOF merger+rd % | TOF <5e-4 % | legacy <5e-4 % |
|----|-------|------|--------|-----------|-----------|-----|-----|-----|
| 16 | 9.67e6 | 1.91 | excess | 1.48e-4 | 3.06e-3 | 92.3 | 99.1 | 99.8 |
| 18 | 17.2e6 | 2.19 | excess | 1.98e-4 | 2.48e-3 | 91.8 | 99.0 | 97.2 |
| 1  | 0.62e6 | 2.26 | excess | 2.64e-5 | 1.12e-3 | 92.7 | 96.8 |  0.2 |
| 0  | 0.57e6 | 4.39 | clean  | 4.81e-6 | 7.99e-6 | 93.7 | 40.1 |  0.7 |
| 19 | 24.8e6 | 7.12 | trunc  | 6.75e-4 | 6.27e-5 | 93.9 | 99.0 | 99.8 |
| 17 | 11.6e6 | 9.21 | trunc  | 3.72e-5 | 2.61e-7 | 85.6 | 14.8 | 99.2 |

## Findings

1. **The TOF residual is ALWAYS ~92% at the merger** (all 6: 85–94%). Whatever TOF
   gets wrong, it is a merger-time phenomenon, not inspiral.

2. **The excess is spurious LOW-FREQUENCY (<5e-4 Hz, red / 1-over-f-rising spectrum)
   power injected at the coalescence TIME — independent of the merger frequency.**
   *Disambiguated by id1* (low mass 0.62e6, merges at HIGH freq): legacy's residual
   is 93% at the merger and only 0.2% <5e-4 Hz (the real high-f merger), while TOF's
   residual for the same source is 92.7% at the merger AND 96.8% <5e-4 Hz — a
   vertical TF streak at t_c, brightest at the lowest frequencies. TOF adds low-freq
   power the physical (high-freq-merging) source cannot have -> it is spurious.
   High-mass 16/18 look the same but are degenerate (their real merger is also
   <5e-4 Hz); id1 breaks the degeneracy.

3. **The trigger is near-equal mass (low q).** TOF mm scales inversely with mass
   ratio: q≈1.9–2.3 (id16/18/1, + id4 q1.93 / id2 q3.70 from the 2x2) give the large
   excess; q≳4 (id0/19/17) are clean or truncation-only. So the artifact is
   *conditional* — it fires for near-equal-mass mergers (stronger symmetric /
   higher-mode merger structure), which is why id17 (high mass but q9.2) nulls
   cleanly (mm 2.6e-7) where legacy still truncates its inspiral.

4. **Legacy** fails by truncating the low-freq **inspiral** for high-mass sources
   (id16/17/19: 97–99% <5e-4 Hz, inspiral-weighted) and is otherwise fine.

## Implication + next step

The signature — a **time-localized transient at t_c with a red spectrum, gated by low
mass ratio** — points to a **step/discontinuity in the TDI-on-the-fly evaluation at
the merger for near-equal-mass systems** (orbit-spline TDI or coarse-graining edge
ringing into the low-freq band), NOT an inspiral/orbit-spline-edge effect. Concrete
fix work: (1) plot the TOF **time-domain** template around t_c for a step/kink on a
low-q source (id4/id16); (2) check `coarse_graining_scale_factor` (=48) and the
TDI-delay interpolation continuity across the merger sample; (3) confirm the q-trend
by extracting q for the full excess set on the other laptop. This is also why the
>5e-4 Hz cut is clean common ground — it excises exactly this red-spectrum transient
tail (96–99% of the TOF residual).

Plots: `mbh_td_residual_id{0,1,16,17,18,19}.png`. Raw summaries: `td_<id>_<resp>.npz`;
per-source printed splits: `analyze_<id>.log`; run log: `overnight.log`.
