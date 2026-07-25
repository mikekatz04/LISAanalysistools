# MBH response × frequency-band study (null mismatch vs mojito data)

Chain-of-custody evidence for the **MBH branch**: how well the stock MBH template
nulls the mojito data at injection, across the 2×2 of

- **response**: legacy phentax (Lagrange-interp) vs TDI-on-the-fly (TOF)
- **band**: full band vs a low-frequency cut at `min_freq = 5e-4 Hz`

All numbers are `[RESULT] branch=mbh ...` lines from the stock
`mojito_null_check` run inside `erebor.full_year_combined` (CHOP_WINDOW,
47.5-day merger-centered window, one MBH id at a time). Same mojito data and
noise (data SNR identical across configs to 6 figures); only the template
(response) and the inner-product band change.

Regenerate the plot with:

```
python ../../runners/mbh_response_study.py \
  "legacy full=legacy_full.txt" "legacy >5e-4=legacy_cut.txt" \
  "TOF full=tof_full.txt" "TOF >5e-4=tof_cut.txt"
```

## Key finding — two DISJOINT full-band failure populations

Neither full-band response matches the mojito data everywhere; each fails on a
different, non-overlapping set of sources, and the **>5e-4 Hz band is the common
ground** where both agree (both cut curves collapse to a ~1e-7…1e-5 floor):

- **legacy-truncation population** — ids `[10,11,12,13,14,15,17,19]`
  (legacy full ≥5× worse than TOF full). Legacy's Lagrange interpolation
  truncates real <5e-4 Hz merger power that the data contains; TOF recovers it
  (id14: 1.19e-4 → 7.2e-7, id15: 9.2e-5 → 4.3e-8). Mostly the higher-mass /
  lower-merger-frequency sources.

- **TOF <5e-4 excess population** — ids `[1,2,4,16,18]`
  (TOF full ≥5× worse than legacy full). TOF injects *spurious* <5e-4 Hz power
  the data does NOT have — every one shows `⟨h|h⟩ > ⟨d|d⟩` (template louder than
  data) with the excess residual living entirely below 5e-4 Hz (the cut removes
  it). Worst is **id4** (low mass, 0.9e6 M☉): full-band mm = **2.05e-2**
  (rr = 2584, Δlnℓ ≈ −1292 at injection), collapsing to 2.1e-4 with the cut.
  id16/id18 are in BOTH populations — TOF fixes their truncation but adds excess,
  net still bad full-band.

- **>5e-4 Hz cut = common ground**: worst mismatch left is 2.65e-5 (legacy-cut,
  6 sources checked) / 2.12e-4 (TOF-cut, all 20; the id4 residual).

## Implication for running MBH in the global fit

- Full-band with **either** response leaves per-source systematics that dwarf the
  noise floor for a handful of sources (up to Δlnℓ ≈ −1300 at truth) — unsafe.
- The **>5e-4 Hz cut** is the safe short-term choice: brings all 20 sources to
  mm ≤ 2.1e-4, and both responses agree there. But it discards the genuine <5e-4
  Hz merger SNR that TOF correctly recovers for the high-mass sources
  (id14/15/19) — a blunt instrument.
- Correct long-term fix: **root-cause the TOF <5e-4 Hz excess** on the
  `[1,2,4,16,18]` population (start with id4, the worst; compare TOF vs data
  spectra below 5e-4 Hz; check `coarse_graining_scale_factor` and orbit-spline
  edge coverage), then run TOF full-band to also capture the real <5e-4 content
  the cut would throw away.

## Provenance

- `legacy_full.txt` — legacy phentax, full band, all 20 (laptop `/Users/mkatz`).
- `legacy_cut.txt` — legacy phentax, >5e-4 Hz, 6 sources (id 0,1,16,17,18,19).
- `tof_full.txt` / `tof_cut.txt` — TOF, full / >5e-4 Hz, all 20 (laptop
  `/Users/mlkatz/new_dev`). The two checkouts agree on legacy full (data SNRs
  identical), so the legacy-vs-TOF differences are real, not a checkout artifact.
