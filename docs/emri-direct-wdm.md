# EMRI direct-to-WDM templates: status and measurements

Developed on branch `emri-direct-wdm` (LAT), now on `dev`: the direct template is the
production EMRI path of the 9mo / 12mo runs (`EMRI_LIKELIHOOD=direct`,
`EMRI_DIRECT_LOOKUP=kernel`, `EMRI_DIRECT_RESPONSE=dense`, `EMRI_TRAJ_WORKERS=8` in the
launchers). The measurements below are point-in-time. Plan: `~/.claude/plans/scalable-napping-toast.md`.
All numbers below are laptop CPU runs on CD1L (mojito light) data or synthetic sources;
mismatch is `1 - Re(O)` with no time or phase maximisation, per TDI channel X, Y, Z.

## What exists

| Piece | Where |
|---|---|
| TDI-on-the-fly X, Y, Z matching the production response | `sources/emri/emritdionfly.py` (`frame="icrs_special"`, `n_fine`, `t_fine_window`) |
| Restored n_ref lookup table + exact evaluation rule | `domains.py` `WDMLookupTable` (`BASIS_CYCLE="quarter_turn"`) |
| Harmonic tracks from the FEW holder | `sources/emri/wdm_direct.py` `harmonic_tracks_from_holder` |
| Plunge chunk (even-start 128-px chunks) | `wdm_het.py` `wdm_chunk_of_td`, `tail_chunk_plan`, `splice_chunk` |
| The fast template | `sources/emri/wdm_direct.py` `EMRIDirectWDM` |
| Laptop table (layer 3600 s, Nf 180, dt 20 s) | `wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5` (sprint root) |

## Results

**TDI-on-the-fly vs production response** (EMRI 1, 15.2 d): mismatch ~1e-13 per channel at
mode thresholds 1e-3 and 1e-7; dlogL difference <= 4e-7. Signed off 2026-09-30.

**Direct-to-WDM vs production, CD1L EMRI 1** (16 d window, no plunge in the window):

| mode threshold | table interp | modes | direct vs production | norm ratio vs production | direct vs data | production vs data |
|---|---|---|---|---|---|---|
| 1e-3 (production EMRI_EPS) | linear | 38 | 1.05e-6 | 0.99973-0.99976 | 0.10 | 0.10 |
| 1e-3 (production EMRI_EPS) | **cubic** (spline default since 09-30, same accuracy) | 38 | **3.5e-8** | **0.999995-0.999997** | 0.10 | 0.10 |
| 1e-5 | linear | 111 | 1.11e-6 | not measured | 2.6e-3 | 2.6e-3 |

The normalised mismatch alone hid a 2.5e-4 amplitude deficit with linear table interpolation
(the bias of linear interpolation across the table's peaked frequency response); cubic
removes it (EMRIDirectWDM default, `interp="cubic"`; CPU only).

The direct template is as close to the data as the production template; the data
mismatch is the production mode threshold's content loss, not the template method.

**Plunging synthetic source, single (2,2,0,0) harmonic** (42.7 d, plunge at 33.8 d):

| region (t / t_plunge) | direct vs TOF | TOF vs production |
|---|---|---|
| < 0.90 (lookup) | 0.7-1.0e-4 | ~3e-12 |
| 0.90-0.99 (chunk) | 0.9-1.4e-5 | ~5e-8 |
| >= 0.99 (chunk) | ~1e-14 | 1e-6 to 1.2e-5 (was 1.3-1.7e-3, fixed: plunge-end tail) |
| whole window | 1.3-1.5e-5 | 3e-7 to 3.8e-6 |

Direct vs production over the whole window: 1.4-1.9e-5 per channel; norm ratios within
1e-4 (whole window) and 1e-3 (< 0.90 region) of 1.

**Error budget of the lookup** (single harmonic, 170 d, p0 = 10.2): the exact local-chirp
model is at 1.9e-4 (median, before 0.9 t_plunge), the table's f/fdot interpolation at
1.1e-3 (fdot step 0.01 layer_df/layer_dt). The table dominates over the inspiral.

## Defects found and fixed on the way

1. EMRITDIonFly built the -m partner as the +m mode with a negated phase: off by pi for
   some higher-l modes (2% strain error at threshold 1e-7).
2. EMRITDIonFly passed Tobs in seconds where FEW's T is in years (integrated to plunge).
3. FEW merges call-time `inspiral_kwargs` into the generator permanently.
4. The historical lookup evaluation (2-way sign) was wrong; the exact rule is a
   parity-dependent quarter turn, with an odd-(m_ref + n_ref) variant and a block bake on
   the chirp term that must be undone at the table nodes (block seams).
5. `build_wdm_lookup_gpu.py` ignored `--time-layers` for complex builds (32x slower).
6. Lookup support must cover offsets [-2, 3] layers; 5 layers per pixel are needed.
7. An even-start WDM chunk is not exact in its interior: edge contamination decays
   algebraically (0.2 at the edge, 5e-5 at 16 px): discard Nt_sub/4 per side.
8. At a plunge the on-the-fly response grid ended 720 s early (delay trim at the end of the
   fine feed): the feed now continues past the stop with zero amplitude (two delay margins).
9. Retrograde input: FEW maps xI0 < 0 to (-a, +1) before its sign rule; the harmonic tracks
   now do the same (the old code flipped the phase for the user form).

## CD1L campaign: all 8 EMRIs at 6 and 24 months

`scripts/emri/emri_cd1l_campaign.sh` runs `emri_cd1l_campaign.py` serially for every CD1L
EMRI (rows 0-7) at 6mo (180 d, Nt 4320) and 24mo (720 d, Nt 17280), one process per
(source, duration), and writes `results.jsonl`, one log per run and `summary.md`.
Each run scores the production, TDI-on-the-fly and direct-WDM templates against the
source-only mojito L1 stream in the WDM domain (SciRD v1): logL = -1/2<d-h|d-h>,
noise-weighted mismatch, opt/det SNR, flat per-channel mismatch and norm ratio, and the
template-vs-template mismatch and dlogL. A failing template is recorded (error and
traceback in the JSON row) and the run continues.

```bash
export MOJITO_LIGHT_PATH=/path/to/mojito_light_v1_0_0          # catalogues/ + data/EMRI/L1/
export EMRI_WDM_TABLE=/path/to/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5   # 30 MB
bash scripts/emri/emri_cd1l_campaign.sh                          # OUT_DIR=emri_cd1l_campaign_out
python scripts/emri/emri_cd1l_campaign.py --src 1 --duration 4d --table $EMRI_WDM_TABLE   # smoke
```

Knobs: `SOURCES`, `DURATIONS`, `TEMPLATES`, `THRESH` (driver); `TOF_FINE_DT` (fine
trajectory spacing, default 300 s), `MODE_BATCH` (16), `EVAL_CHUNK`, `RSS_LIMIT_GB`,
`START_OFFSET_S` (script). The driver skips pairs with a `.done` marker, so a rerun resumes.
Long windows are untested territory: the TOF and direct paths were validated to 170 d;
memory and wall time at 6 and 24 months are what the campaign measures.

**Mode-threshold sweep (cheap, production template only, ~30 s per run):**

```bash
TEMPLATES=prod THRESH=1e-2,1e-3,1e-4,1e-5 OUT_DIR=emri_cd1l_thresh bash scripts/emri/emri_cd1l_campaign.sh
```

Each row also carries `prod_resid_snr2` (= -2 logL: the power a subtracted template leaves
in the residual), its split over the bands <1, 1-3, 3-10, >10 mHz, and the kept-mode count.
EMRI 1, 6 months (data SNR 21.8), production template:

| threshold | modes kept | residual SNR^2 | of which 3-10 mHz | opt/data SNR |
|---|---|---|---|---|
| 1e-2 | 14 | 26.8 | 23.9 | 1.056 |
| 1e-3 (production) | 38 | 2.94 | 2.64 | 1.011 |
| 1e-4 | 67 | 0.37 | 0.27 | 1.0002 |
| 1e-5 | 112 | 0.037 | 0.027 | 0.9998 |

**Aliasing at 20 s (09-30).** On the plunging gate source, TDI-on-the-fly vs production per
harmonic was 3e-3..0.8 for m >= 3 at dt = 20 s and drops to 3e-5..4e-4 at dt = 5 s ((4,4,0,1)
5.3e-2 -> 2.5e-5, (6,6,0,2) 0.79 -> 4e-5): near the plunge the high harmonics exceed the 25 mHz
Nyquist of the 20 s grid. The response is fine; 20 s grids (the laptop table, the CD1L campaign
script) are unsafe for EMRIs plunging inside the window. Production (2.5 s) is not affected.

**GPU timing (09-30, H100 cuda13x, 6 months, production grid, EMRI 1, batch of 64 in chunks of
16):** direct 157 ms/template at eps 1e-3 (38 modes) and 418 ms at 1e-5 (112 modes) vs production
94 / 101 ms. Per template: FEW 11 ms (knot feed), response kernel 79 / 223 ms, harmonic tracks
(CPU) 21 / 53 ms, tracer 4 / 11 ms, lookup 6 / 16 ms. The response kernel recomputes the full
geometry per (harmonic, time): the next step is a kernel that shares it across the harmonics of a
template (plan: ~/.claude/plans/emri-tof-dense-phase-overnight.md).

## Dense-phase response kernel (`TDDenseTDIonTheFly`, 10-01)

`response="dense"` in `EMRIDirectWDM` feeds a new, opt-in TD TDI-on-the-fly kernel
(`TDDenseTDIWaveform`, lat_spline_tdi_waveform.{hh,cu}; binding `TDDenseTDIWaveformWrap`):

- per template: the integrator's knots and DOPR853 8th-order dense-output phase coefficients
  (re-expressed exactly on FEW's holder knots, whose last knot FEW cuts at T); per harmonic: the
  integers (m, k, n) and a complex amplitude cubic spline over the knots
  (`dense_inputs_from_holder`). The phase is exact everywhere: no fine-grid phase splines.
- kernel 1, one thread per (template, time): the link geometry and the three fundamental phases
  are computed ONCE per TDI unit and reused by every harmonic (unit term
  `sign * pre * (xi_p A_p + xi_c A_c) (z_em - z_rec)`, `z = amp_factor c exp(-i Phi)`); many
  templates per launch. Delayed times outside the trajectory contribute zero (plunge without a
  feed extension); the reference phase continues linearly outside the trajectory.
- kernel 2, one block per harmonic: the existing amplitude/phase extraction + unwrap.

CPU results (laptop): EMRI 1, 16 d: mismatch vs production 3.55e-8 (= spline response), whole
template 1.6 s -> 0.8 s. Plunging single harmonic: direct vs production 1.5-1.7e-5 over the
window, 7e-6 in the last 1% (spline response: 9e-6..1.2e-5). Tests: `tests/test_tdi_dense.py`
(== the spline-fed kernel to 1e-9 on exactly representable input, 2 templates x 5 harmonics,
inc != 0; polarisation and strain-sign mutations caught), `tests/test_emri_dense_inputs.py`.
The CUDA build of the kernel has NOT been compiled or run yet.

More CPU checks (10-01):

| check | result |
|---|---|
| EMRI 1, eps 1e-5 (111-112 modes), dense vs spline response, wall | 16 d 1.6 vs 3.8 s; 60 d 5.4 vs 15.3 s; 180 d 16.0 vs 49.8 s |
| same, dense vs spline template | mismatch <= 4e-13, norm ratio 1.0000000 |
| plunging source, 69 modes, dt 5 s (table NF720_DT5), dense direct vs production | whole window 1.3-2.5e-4; < 0.9 t_p 7.5-9.2e-5 (table); 0.9-0.99 3.8-6.0e-6; last 1% 5.9e-4..1.2e-3 |
| same at dt 20 s | 2.6-4.7e-2 (aliasing of m >= 3 near plunge) |
| per mode, dense response vs production at 5 s | (2,2,0,0) 2.5e-6, (4,4,0,1) 2.0e-5, (3,3,0,2) 1.5e-4, (5,5,0,0) 1.8e-4; the last two entirely in the final 1% before plunge (production's interpolated strain across the abrupt stop) |

Lookup memory: the table call is chunked (`EMRI_DIRECT_LOOKUP_CHUNK`, default 2e6 entries).

## Sparse response grid (10-02, default with `response="dense"`)

The response no longer needs one sample per pixel. `EMRIDirectWDM(response_grid="sparse")`
(default; `EMRI_DIRECT_RESPONSE_GRID=pixels` restores the old grid) evaluates the dense kernel on
the integrator knots plus a point every `sparse_dt` (43200 s; `EMRI_DIRECT_SPARSE_DT`) and
reads the pixels through `ExactPhaseTDIOutput`: channel phase = the harmonic's EXACT dense-output
carrier `Phi_s(t)` (same coefficients as the kernel) + a splined residual `r = tdi_phase +
phase_ref - Phi_s` (Doppler delay to spacecraft 1, polarisation and transfer-function phase, the
mode amplitude's phase), amplitude splined; `f`, `fdot` from the same total phase. Only slow
quantities are splined, so the grid follows the trajectory, not the pixels.

The grid must stay where the response is COMPLETE (`sparse_response_grid`): a delay margin
(600 s) inside the trajectory and inside the orbit tables (`_orbit_span`). Outside either the
kernel zeroes the channel, and a half-day spline across that edge rings into the window
(measured: 2 % amplitude two intervals in at a trajectory end; 4e-4 rad at the first pixels of a
laptop-trimmed L1 table). An in-window plunge keeps the 80 s segment. Batched templates get their
own grids, padded to a common length by halving their widest intervals (`pad_grid`).

| check | result |
|---|---|
| synthetic, 12 h knots vs 60 s grid (phase / f / fdot / amp) | 8e-9 rad / 4e-12 Hz / 4e-15 Hz/s / 1.4e-8 |
| power-law chirp (all polynomial orders), 6 h knots | exact carrier 1.6e-6 rad vs whole-phase spline 7.3e-4 |
| CD1L EMRI 1, 16 d, 20 s, CPU: response points | 388 -> 37 |
| same, mismatch vs production (fit domain, sensitivity-weighted) | 2.386e-8, identical to the pixel grid; dlogL 1.8e-6 |
| same, s/row (fit wiring check, CPU) | direct 0.66 vs production 1.33 |

## Fused lookup-sum kernel (10-02, default `lookup="kernel"`)

The response stays sparse until the WDM sum. `wdm_lookup_sum` (C++/CUDA,
`cutils/lat_spline_tdi_waveform.cu`; device helpers in the header-only
`cutils/wdm_lookup_kernels.hh`, shared with other sources' kernels) runs one thread per
(harmonic, pixel): amplitude and residual phase from the response's cubic splines (GBT
`CubicSplineInterpolant` coefficients), the exact DOPR853 carrier and its first two time
derivatives, `f = phase' / 2 pi` and `fdot = phase'' / 2 pi` ANALYTICALLY, the `-m` mirror, the
`|fdot| <= fdot_axis_max` and `f > 2 layer_df` filters, then the `2 num_m_layers + 1` layers of
the n_ref lookup (the table's prefiltered cubic B-spline in ndimage mirror mode, zero outside;
`get_wdm_coeffs`' quarter-turn rule) summed straight into `out[row, ch, m - m_lo, n]`
(atomicAdd). Nothing per pixel lands in global memory except the output; the per-pixel
tracer arrays, the table's temporaries and the mode blocks are gone. Python driver:
`wdm_direct.lookup_sum_kernel`; `EMRIDirectWDM(lookup=...)`, env `EMRI_DIRECT_LOOKUP`
(`kernel` default; `python` = the analytic tracer + `get_wdm_coeffs` + scatter-add, the reference).
A kernel request on a module built without it, a `response="spline"` output or a table that is
not spline-interpolated falls back to Python with one warning.

Around it:

* the Python tracer also differentiates analytically now (`eval_tracer` on
  `SplinedTDIOutput`/`ExactPhaseTDIOutput`, one evaluation per pixel instead of five stencil
  points; `EMRI_TRACER=stencil` keeps the central differences);
* `f_band=(m_lo, m_hi)`: `batch()` returns those layers only; the fit's adapter asks for the
  run's active band (6 months: 179 of 1440 layers, ~8x less memory per template);
* the per-harmonic tracks (all modes at all pixels, ~7 ms per 6-month template on the GPU)
  are only computed when a bound over the three fundamentals says a harmonic MAY hand off to
  the plunge chunk in the window (`_handoff_may_trip`; `EMRI_DIRECT_HANDOFF_BOUND=0` always
  computes them). The bound cannot hide a handoff, so the template is unchanged.

| check | result |
|---|---|
| kernel vs Python, synthetic (f, fdot) sweeping the table's edges, 5 and 7 layers | 1.1e-12 max rel (phase ~2e4 rad round-off) |
| kernel vs Python, dense response: sparse grid + exact carrier, 2 templates | 1.7e-12 |
| kernel vs Python, dense response: pixel grid, carrier folded into the phase spline | 3.4e-12 |
| analytic vs stencil tracer, synthetic sparse response | 1e-13 Hz in f, 9e-17 Hz/s in fdot |
| CD1L EMRI 1, 16 d, 20 s, fit wiring check (box t[60:324]) | kernel = python: mismatch 2.109e-8, max dlogL 1.14e-6 |

### In-window plunge: exact tail, source-frame stop taper (10-02, taper 10-03)

The first 720-day speed test put EMRI 1's plunge 518 d into the window and scored the direct
template at SNR 3920 against production's 56 (foreground noise; the 180/360 d windows were fine).
The plunge chunk read its time series from the response's amplitude/phase splines, which cannot
follow a channel that stops (each TDI delay term drops out at its own time); the leaked broadband
power has a low-frequency, X+Y+Z-like part the TDI combination would suppress, and the noise
weights it enormously. Now:

* the chunk's time series comes from the dense kernel at the sample times
  (`EMRIDirectWDM._dense_td`: the raw complex channels' real part, `TDDenseTDIonTheFly.channels`,
  the channel kernel alone, summed over the template's harmonics in the kernel);
* the SOURCE is tapered over the last `EMRI_DIRECT_PLUNGE_TAPER_S` (120 s) of emission time before
  the trajectory's end: `stop_taper` weights every delay term by sin^2 of its emission time
  (`TDDenseTDIWaveform::stop_weight`). Why: the stop itself has no net T-like area (every delay
  term is the same truncated function), but SAMPLED, each term's cut falls at its own place
  between samples, and the channel keeps an area that aliases into the lowest frequencies
  (`tests/test_tdi_dense.py`: the net X+Y+Z area of a stopping sinusoid is 8x its pre-stop swing
  at 1 s sampling and shrinks with the step). Production's Lagrange interpolation band-limits the
  stop. A weight on the emission time is the same for every term, so the cancellation holds and
  each term is smooth. (10-02 to 10-03 tapered each CHANNEL over 300 s before its stop instead;
  the cuts stayed inside that window and their aliased area did not go away.)
* every harmonic of a template hands off to the chunk at the earliest harmonic's pixel (the
  summed tail implies the common handoff): one transform per chunk window.

Laptop check (dt 5 s, equal-arm orbits, 60 days around EMRI 1's plunge, against production with the
exact Lagrange prefactor, scirdv1 + foreground):

| stop | mm vs production | 0.25-1 mHz SNR (production 1.6e-4) | 3-10 / 10-25 mHz mm |
|---|---|---|---|
| hard stop | 0.62 | 50 | 1.1e-4 / 2.7e-6 |
| per-channel 300 s taper (10-02) | 8.0e-3 | 2.7 | |
| source taper 60 s | 2.2e-6 | 0.011 | 1.6e-6 / 3.9e-6 |
| **source taper 120 s (default)** | **2.4e-6** | **0.0016** | 2.0e-6 / 4.3e-6 |
| source taper 300 s | 5.1e-6 | 0.0018 | 4.3e-6 / 8.2e-6 |

Against the direct model's own exact time series transformed whole, the common-handoff template
is at mm 1.6e-6 (the per-harmonic handoff of 10-02 was at 3.9e-3: its lookup ran into the last
hours before the plunge).

## In the global fit: `EMRI_LIKELIHOOD=direct` (10-02; the 6-month launcher default and, through `auto`, the library default since 10-03)

`emri_pe` becomes `EMRIDirectLikeMove` (`globalfit/moves/emridirectmove.py`), built by
`build_emri_move_runtime` (`stock/erebor/source_runtime.py`) through `EMRIDirectMoveBuilder`
(`recipe.py`). Same pattern as `MBHBatchedLikeMove` / `SOBBHChunkedLikeMove`: the add/remove
choreography is untouched, only the scoring changes.

* **Scoring.** Each chunk of up to `EMRI_BATCH_MAX_SIZE` (8) rows is one
  `EMRIDirectWDMSignalGen.templates` call (`sources/emri/direct_signal_gen.py`): one FEW call per
  row, one response launch (`EMRI_DIRECT_RESPONSE`, `dense` default), one table pass, built on the
  run's FULL wavelet grid and cropped to its active box. Rows are scored against their own walker's
  residual and PSD, `offset + <r|h> - 1/2 <h|h>` with `offset = acs.likelihood()` on the exposed
  residual; one batched inner-product pair per walker, one host pull per chunk.
* **Template generation follows (10-03).** In direct mode the engine's EMRI template generator
  (`SourceSignalGen`: residual rebuilds at load, the move's expose/fold through the containers'
  installed generator) is the direct template too, so fills, rebuilds and scoring use one template
  family and the shared residual holds exactly what the sampler scored. Only the cross-check and
  the fallback use the production wrap (FEW + ResponseWrapper + dense TD->WDM), which the move
  carries as its own `waveform_gen`. (Until 10-03 the fill stayed on production.)
* **Checks with a tolerance** (cold rung only): the fast-vs-production cross-check (every 10th
  visit, `EMRI_CHECK_LL_EVERY`) and the expose invariant. Per point the tolerance is
  `EMRI_CHECK_LL_TOL + mm <h|h> + 3 sqrt(2 mm <h|h>)` (1 nat, `mm = EMRI_CHECK_LL_MM` = 3e-4):
  the template difference grows with SNR (bias ~ mm SNR^2, scatter ~ sqrt(2 mm) SNR), so a fixed
  1 nat would fire on most visits at 6-month SNRs; an expose-sign bug moves lnL by ~SNR^2.
* **Inspirals ending before the window** (hot-rung rows that plunge before the data start) are an
  exact zero template, no response call.
* **Failures.** A FEW domain refusal scores that row `-1e300` (as on the container path); any other
  failure of a direct chunk (GPU OOM, a missing kernel, a table error) scores the chunk through the
  container path with the production generator (not the installed one, which is the direct
  template), warned once per leaf, counted in `n_batch_fallbacks` and the
  `[EMRI_DIRECT]` telemetry.
* **Getter** `get_emri_direct_gen`: per device, reuses the FEW generator inside that device's
  production wrap (one FEW construction per device), its orbits, the REF epoch (mojito) and the
  `EMRI_EPS` threshold; refuses a table built for another `Nf`/`dt`.
* **Lookup table: found or built once** (`lisatools/wdm_lookup_store.py`). `EMRI_DIRECT_TABLE`
  points to a specific file; unset, it is the canonical file in the run's folder
  (`general.file_store_dir` = the launcher's `STORE_DIR`), e.g.
  `wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5` on the 6-month grid (the recipe of every
  table so far, `EMRI_TABLE_RECIPE`). `ensure_lookup_table` returns it when present (checked
  against the grid, never rebuilt over), else builds and saves it there: written to a temporary
  file and renamed into place, under a `<path>.lock` that makes other callers (MPI ranks, the
  launcher) wait for the one builder. A restart finds the file. The 6-month launcher's preflight
  runs it on the node's GPU before mpiexec, so ranks never build.
* **Resolver** `resolve_emri_direct_cfg`: `direct` needs a WDM domain, XYZ channels, a known
  response, batch >= 1, workers >= 0 (the table need not exist yet) and raises otherwise.
  `auto` (the `SourceEMRISettings.likelihood` default since 10-03, as `MBH_LIKELIHOOD=auto`)
  resolves to `direct` when those hold, else to `full` with ONE INFO line naming the reasons (a
  run domain that cannot be identified as WDM counts as a reason for `auto`); the cfg carries the
  resolved mode only. Every stock EMRI run on a WDM / XYZ grid therefore uses the lookup template
  for scoring and template generation unless `EMRI_LIKELIHOOD=full`.
* **Trajectory pool** (`EMRI_TRAJ_WORKERS`, 0 in the library): spawn workers integrate a chunk's
  trajectories (`few.trajectory.pool`, FEW gpu_backend >= 68bcda54) for chunks of at least that many
  rows (eigen sweeps). Workers start eagerly with `__main__.__file__` hidden, or each spawned child
  would re-import `run_combined_staged.py` (and MPI); a warm-up that did not start every worker is
  refused. Any pool failure disables it for the run. Counters (rows, trajectories computed, cache
  hits/misses) are logged as `[EMRI_DIRECT] trajectory pool:` every 50 pooled batches.
* **Launcher** (`submit_gf_6mo_v9_4gpu.sh`, and the same block in the 9mo v9 and flagship
  launchers, EMRI block after `EMRI_EPS`): default `direct` since
  10-03 (`EMRI_LIKELIHOOD=full` restores production for scoring and template generation); the
  `# EMRI PREFLIGHT.` heredoc resolves the knobs through the settings class and, for direct, checks
  `EDGE_CROP_WAVELETS >= pixel_edge` (8), runs `tests.test_tdi_dense.TDDenseGPUParityTest` and
  (`EMRI_DIRECT_LOOKUP=kernel`) the fused kernel's GPU-vs-CPU test on the node's GPU (a skip
  refuses) and finds or builds the lookup table. The self-dispatch
  passes `--cpus-per-task = 2 + EMRI_TRAJ_WORKERS` when direct (10: 8 workers since 2026-10-07,
  was 4); `OMP_NUM_THREADS` stays 1.

| check (laptop CPU, CD1L EMRI 1, 16 d, dt 20 s, dense, real getters + real move) | result |
|---|---|
| direct vs production template at truth | mismatch 2.4e-8, amplitude ratio 1.000000 |
| move lnL, direct vs production container path, 6 rows | max \|dlogL\| 1.8e-6 |
| `--fill direct` (10-03, the fit's setup): direct batched vs the per-row container path / vs production check | 9.3e-10 / 1.1e-6 |
| wall per row | direct 0.73 s, production 1.38 s; 0 fallbacks |

H100 speed tests (`scripts/emri/emri_speed_durations.sh`, CD1L EMRI 1, L1 orbits, scored against the
mojito stream under scirdv1 + the fitted galactic foreground at each Tobs; sparse grid, fused kernel):

| window | eps | ms/template single / 16 per call | production ms | mm vs production | lnL vs data: direct / production |
|---|---|---|---|---|---|
| 180 d | 1e-3 | 25 / 15 | 88 | 1.6e-5 | -1.308 / -1.305 |
| 180 d | 1e-5 | 38 / 18 | 94 | 1.7e-5 | -0.020 / -0.017 |
| 360 d | 1e-3 | 32 / 21 | 163 | 7.2e-6 | -2.892 / -2.886 |
| 360 d | 1e-5 | 45 / 24 | 175 | 7.3e-6 | -0.050 / -0.043 |
| 720 d (plunge at 518 d) | 1e-3 | 551 / 630 | 398 | 1.5e-5 | -5.557 / -5.522 |

At 720 d every row hands off to the plunge chunk and is built alone. 10-03 H100 rerun with the
channels-only kernel, the common handoff and one FEW call per row: 175 ms per template
(production 396), but mm vs production 6.8e-5 / lnL -5.732: the per-channel taper's aliased
stop area, now on every harmonic; fixed by the source-frame taper above (rerun pending).

Cluster check on the 6-month grid: `scripts/emri/emri_direct_fit_wiring_check.py --backend cuda12x
--days 180 --table-dir <dir holding or receiving the table> --rows 8`.
Tests: `tests/test_emri_direct_move.py`, `test_emri_direct_wiring.py`,
`test_emri_direct_fanout.py` (real proposes on FakeWorld ranks: one-walker replicas and the
v9 1-walker-per-rank blocks), `test_submit_scripts_layout.py::SixMonthEMRIDirectTest`.

Related fix (10-02): the TDI-on-the-fly configured-orbits cache (`_orbits_cache_key`,
`response/tdionfly.py`) is keyed by backend and device; it handed the CPU tables to the GPU
response when one process built both (`TDDenseGPUParityTest` on the cluster), and could hand one
GPU's tables to another. `LISATOOLS_ORBITS_CACHE=0` bypasses the cache.

## Open items

1. (resolved) TOF vs production at the abrupt plunge end.
2. Table resolution (fdot direction; cubic fixed the f-direction bias): finer fdot rows near 0 (where most pixels sit) to reach the model's
   ~1e-4 per-pixel level; production-grid table (Nf 1440, dt 2.5) on the cluster GPU.
3. Speed (09-30): EMRI 1, 16 d, 38 modes, laptop CPU: 27 s -> 2.75 s with the same mismatch vs
   production (3.6e-8). Changes: the response is fed on a 1800 s grid when no harmonic hands off
   inside the window (coarse + 80 s from the earliest handoff otherwise; TOF accepts non-uniform
   times); one vectorised table call + one scatter-add per batch; interpolation `"spline"` (uniform
   cubic B-spline via ndimage, same code on scipy/cupyx; default); one tracer spline call.
   Remaining CPU cost: two FEW calls (~1.4 s) and the response (~0.9 s). Code is numpy/cupy
   agnostic (written before its first GPU run; it has run on the cluster GPUs since, see the
   H100 timing above, and in production since 10-03): `scripts/emri/emri_batch_speed.py --direct-table`.
4. Memory: every FEW EMRI generator reads the whole 5.1 GB amplitude file at construction
   (`few/amplitude/ampinterp2d.py:235`), a ~6 GB transient footprint. One per process.
5. EMRIs 0, 2-7: their L1 bricks are not on the laptop; run
   `scripts/emri/emri_direct_wdm_mismatch.py --src N` on the cluster.
6. Neighbour layers per pixel: 5 (num_m_layers=2) hold an EMRI inspiral (fdot mostly
   < 0.1 layer_df/layer_dt); a chirp at 0.25 units keeps ~1.3% of its power two layers out and
   loses ~2e-2 rel L2 in layers three out. Make num_m_layers grow with fdot if needed.
7. Mode selection: EMRIDirectWDM selects modes on a 256-point trajectory over ITS window;
   the production wrapper selects over its own span from the reference epoch. For a
   like-for-like comparison on another window, pass the production modes via
   `mode_selection=[(l, m, k, n), ...]`.
8. The kappa (intra-chunk sweep) guard was removed: the SOBBH limit is for heterodyned
   chunks; the plunge chunk is a raw TD->WDM transform, exact through the plunge.
