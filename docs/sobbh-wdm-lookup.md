# SOBBH direct-to-WDM lookup scorer: status and measurements

Developed on branch `sobbh-wdm-lookup` (LAT), now on `dev`: `SOBBH_LIKELIHOOD=lookup` with the
fused kernel (`SOBBH_LOOKUP_KERNEL=kernel`, `SOBBH_LOOKUP_EVAL_DT=43200`,
`SOBBH_LOOKUP_ROW_BATCH=32`) is the production SOBBH scorer of the 9mo / 12mo runs; the
sections below are dated status notes. Spec `docs/superpowers/specs/2026-09-30-sobbh-wdm-lookup-design.md`,
plan `docs/superpowers/plans/2026-09-30-sobbh-wdm-lookup.md`. The SOBBH twin of the EMRI
direct-to-WDM template (`docs/emri-direct-wdm.md`), vectorized over proposal rows and slotted into
`SOBBHChunkedLikeMove` as a comp.

## What exists

| Piece | Where |
|---|---|
| Vectorized n_ref table evaluator (quarter-turn rule, linear / Keys cubic) | `lisatools/wdm_lookup_eval.py` `WDMLookupEvaluator` |
| Batched 3.5PN, batched TDI-on-the-fly, tracer, sparse template, inner products, fill | `lisatools/sources/sobbh/wdm_direct.py` |
| The comp for the move / engine (`get_ll_wdm`, `fill_global_wdm`) | `SOBBHLookupComputations` (same file) |
| Stock knob | `SOBBH_LIKELIHOOD=lookup` + `SOBBH_LOOKUP_TABLE_PATH` (and `SOBBH_LOOKUP_{NUM_M_LAYERS,EVAL_DT,INTERP,ROW_BATCH}`) |
| Gate script | `scripts/sobbh/sobbh_lookup_gate.py` |

## The table

The n_ref table depends on the layer duration only: two tables with `layer_dt = 3600 s` built at
(Nf=64, dt=56.25) and (Nf=128, dt=28.125) agree entry by entry to 4e-9 (test
`TablePortabilityTest`). The laptop EMRI table `wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5`
(layer 3600 s, offsets [-3, 3) layers, fdot +-8 layer units in steps of 0.01, 30 MB) therefore
serves the production grid (Nf=1440, dt=2.5). SOBBH chirp rates are tiny in layer units
(|fdot| <= 0.002 for catalogue-like sources; the 6-mo catalogue's worst chirper ~0.005), so
`num_m_layers=2` (5 layers per pixel) is *expected* to be converged by analogy with the EMRI
direct-to-WDM measurement (a chirp at fdot 0.25 layer units keeps ~1.3% of its power two layers
out; SOBBH's fdot is at most ~0.005, two orders of magnitude smaller) — this has not been
separately measured for SOBBH in this gate. No plunge chunk is needed; pixels past the fdot axis
(sources merging in band) are dropped and counted.

## Results (laptop CPU, EqualArmlengthOrbits, 2nd-generation TDI, scirdv1)

Both runs below are `scripts/sobbh/sobbh_lookup_gate.py --rows 8`, laptop table
`wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5`, full printed output pasted verbatim (a
nanobind "leaked instances/types/functions" block printed by every run at interpreter shutdown is
omitted below — it is a known benign refcount-cleanup artifact at process exit, not a correctness
signal).

### What each column measures

- `mm` = flat `1 - Re(O)` per channel, no maximisation, computed on the **interior** pixels only
  (`EDGE=24` grid-end time pixels excluded on each side of the time axis, because the dense TD->WDM
  transform's grid-end pixels are unreliable edge artifacts, not a lookup defect).
- `ratio` = the norm ratio `sqrt(<h|h>/<truth|truth>)` on that same interior.
- `mm_w`, `dlogL`, `snr` = the scirdv1 XYZ noise-weighted mismatch / log-likelihood difference /
  SNR, computed over **all** pixels, including the dense transform's unreliable grid-end pixels.
- `mm_w_int` (new in this revision) = the same noise-weighted mismatch as `mm_w`, but with both
  `truth` and the lookup template zeroed outside `[EDGE, nt-EDGE)` first — i.e. `mm_w` restricted
  to the interior, added specifically to separate the two effects folded into `mm_w`. Measured
  (from `docs/sobbh_lookup_gate.jsonl`): in every one of the 12 source-rows below, `mm_w_int` is
  within about an order of magnitude of the interior flat `mm` (same pixels, same order), while
  the all-pixel `mm_w` is `69x`-`20,700x` above `mm_w_int` (about 1.8-4.3 orders of magnitude).
  This shows the `mm`-vs-`mm_w` gap is driven almost entirely by the dense TD->WDM transform's own
  unreliable grid-end pixels (present in `mm_w`, excluded from both `mm` and `mm_w_int`), not by
  an interior lookup-vs-truth defect.
- In the timing block, `exact` is the dense TD->WDM transform of the lookup's **own** batched
  response (not the production `SOBBHTDIonFly`, not an independent ground truth) — so
  `lookup_vs_exact_max_abs` isolates the lookup table / sparse-template approximation error
  specifically, while `chunked_vs_exact_max_abs` contains **all** of the chunked comp's
  approximations at once (the `m_band_half_width=3` band truncation, its `Nt_sub`/`N_sparse`
  chunking, and its own edge handling). How much of the chunked number comes from band truncation
  alone versus its other approximations is **unmeasured** by this script.
- Timing keys: `*_warm` is the FIRST call on the batch (includes any one-time setup/caching); the
  unsuffixed key (e.g. `lookup_get_ll`) is the SECOND call on the same batch. Both are single
  wall-clock samples, CPU-only, not medians. The two comps are not scoring an identical band: the
  lookup comp's band is `num_m_layers=2` -> 5 layers per pixel, the chunked comp's is
  `m_band_half_width=3` -> 7 layers. By the time the timing block runs, the lookup comp's `direct`
  object has already been exercised by the six `dense()` calls above, while the chunked comp
  (`ch`) is freshly constructed — so the two comps' `_warm`/unsuffixed pairs are not a clean
  apples-to-apples cold-start comparison between the two comps.

### `--nt 1024` (42.7 days)

```
grid Nf=180 Nt=1024 dt=20.0 layer_dt=3600.0 s; table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5
src 0 f_low 0.0045 snr 0.1  mm X/Y/Z 5.26e-08/2.28e-06/5.86e-07  ratio 0.99998/0.99997/0.99998  mm_w 4.56e-04  mm_w_int 6.59e-06  dlogL -8.796e-06  tof-dense 7.4s lookup 4.4s
src 1 f_low 0.0080 snr 0.3  mm X/Y/Z 5.31e-08/8.17e-08/4.77e-08  ratio 1.00000/1.00000/1.00000  mm_w 3.31e-04  mm_w_int 9.06e-08  dlogL -3.840e-05  tof-dense 7.8s lookup 6.6s
src 2 f_low 0.0120 snr 0.3  mm X/Y/Z 1.78e-07/4.95e-08/7.88e-07  ratio 1.00001/1.00001/1.00010  mm_w 1.47e-03  mm_w_int 3.16e-07  dlogL -1.721e-04  tof-dense 5.8s lookup 4.9s
src 3 f_low 0.0150 snr 0.8  mm X/Y/Z 2.76e-07/1.93e-07/1.62e-07  ratio 1.00000/0.99998/1.00000  mm_w 9.32e-04  mm_w_int 3.12e-07  dlogL -5.352e-04  tof-dense 6.0s lookup 5.9s
src 4 f_low 0.0060 snr 0.1  mm X/Y/Z 6.66e-08/5.70e-08/6.08e-08  ratio 1.00000/1.00000/1.00000  mm_w 1.11e-03  mm_w_int 5.39e-08  dlogL -6.334e-06  tof-dense 4.9s lookup 5.2s
src 5 f_low 0.0180 snr 0.5  mm X/Y/Z 1.93e-07/2.60e-07/2.10e-07  ratio 0.99999/1.00000/1.00000  mm_w 1.38e-03  mm_w_int 2.29e-07  dlogL -3.747e-04  tof-dense 4.5s lookup 4.5s
timing: {
 "lookup_get_ll_warm": 6.661789873032831,
 "lookup_get_ll": 5.772116897045635,
 "lookup_fill": 5.717818813980557,
 "lookup_vs_exact_max_abs": 6.449406441786688e-05,
 "chunked_get_ll_warm": 15.221764751011506,
 "chunked_get_ll": 14.695631613954902,
 "chunked_vs_exact_max_abs": 0.0001748772653341578,
 "rows": 8,
 "nt": 1024
}
```

`dropped_pixels == 0` and `merged_rows == 0` for every one of the 6 sources (checked against
`docs/sobbh_lookup_gate.jsonl`). All per-channel `mm` <= 2.28e-6 (well under the 1e-3 gate); all
norm ratios within **1.03e-4** of 1 (recomputed directly from the JSONL: worst `|ratio-1| =
1.0279e-4`, src 2 channel Z; well under the 2e-3 gate). Source SNRs at this duration span
0.075-0.76 (see "SNR context" below). `lookup_vs_exact_max_abs` is 6.45e-5; `chunked_vs_exact_max_abs`
is 1.75e-4 — per the legend above, these are not a clean apples-to-apples split of the same error
source.

### `--nt 4320` (6 months, 20 s)

```
grid Nf=180 Nt=4320 dt=20.0 layer_dt=3600.0 s; table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5
src 0 f_low 0.0045 snr 0.6  mm X/Y/Z 1.52e-07/2.06e-07/2.14e-07  ratio 0.99999/0.99999/0.99999  mm_w 7.57e-05  mm_w_int 6.92e-07  dlogL -2.296e-05  tof-dense 10.0s lookup 4.5s
src 1 f_low 0.0080 snr 0.7  mm X/Y/Z 3.03e-08/4.04e-08/3.00e-08  ratio 1.00000/1.00000/1.00000  mm_w 5.05e-05  mm_w_int 4.07e-08  dlogL -2.767e-05  tof-dense 5.3s lookup 5.6s
src 2 f_low 0.0120 snr 0.6  mm X/Y/Z 1.46e-07/3.49e-08/1.36e-07  ratio 1.00001/1.00000/1.00001  mm_w 2.40e-04  mm_w_int 1.28e-07  dlogL -8.599e-05  tof-dense 7.2s lookup 5.7s
src 3 f_low 0.0150 snr 1.5  mm X/Y/Z 3.99e-07/2.18e-07/2.15e-07  ratio 0.99999/1.00000/1.00000  mm_w 5.34e-04  mm_w_int 3.46e-07  dlogL -1.256e-03  tof-dense 23.6s lookup 13.5s
src 4 f_low 0.0060 snr 0.2  mm X/Y/Z 1.14e-07/1.11e-07/1.11e-07  ratio 1.00001/1.00001/1.00001  mm_w 1.42e-04  mm_w_int 1.00e-07  dlogL -7.685e-06  tof-dense 15.1s lookup 14.0s
src 5 f_low 0.0180 snr 1.2  mm X/Y/Z 1.53e-07/1.66e-07/1.57e-07  ratio 1.00000/0.99999/1.00000  mm_w 1.39e-04  mm_w_int 1.66e-07  dlogL -2.085e-04  tof-dense 14.8s lookup 8.8s
timing: {
 "lookup_get_ll_warm": 10.2268609800376,
 "lookup_get_ll": 9.328296154970303,
 "lookup_fill": 10.599311379948631,
 "lookup_vs_exact_max_abs": 5.2925551063925935e-05,
 "chunked_get_ll_warm": 75.62815951905213,
 "chunked_get_ll": 61.12050140206702,
 "chunked_vs_exact_max_abs": 0.0008285584205466046,
 "rows": 8,
 "nt": 4320
}
```

`dropped_pixels == 0` and `merged_rows == 0` for every source again. All per-channel `mm` <= 4.0e-7;
all norm ratios within **1.44e-5** of 1 (recomputed from the JSONL: worst `|ratio-1| = 1.437e-5`).
The lookup comp's `get_ll_wdm` second call for 8 rows on the 6-month grid is 9.33 s vs the chunked
comp's 61.12 s (~6.5x faster at this `row_batch`/`Nt_sub` pairing, a single-sample wall-time ratio
on a shared laptop at these specific knob settings, not a controlled benchmark or an asymptotic
speed-up claim). `fill_global_wdm` for 8 rows is 10.60 s.

### SNR context

The six catalogue-like gate sources span SNR 0.075-1.53 across both durations (measured range
from `docs/sobbh_lookup_gate.jsonl`) for these masses/frequencies at Gpc-scale distances over
these windows. lnL errors and `dlogL` scale roughly as SNR^2 x mismatch, so the move-level anchor
measured separately in `test_sobbh_lookup_move` — lookup-vs-slow lnL max |diff| 0.081 at SNR^2 =
10.7 (SNR ~3.3) — is the measured anchor at a non-trivial SNR, well above the SNR ~0.1-1.5 sources
scored directly by this gate script. The two gate durations mainly differ in SNR, not in any
duration-dependent accuracy effect: going from 42.7 d to 6 months raises each source's integrated
SNR (more cycles observed within the window), and once an error is read relative to signal power
rather than as a raw lnL-unit number, both comps' agreement with `exact` improves rather than
degrades with the longer duration (the lookup's `lookup_vs_exact_max_abs` even decreases in raw
terms, 6.45e-5 -> 5.29e-5, despite the higher-SNR batch; the chunked number's raw growth,
1.75e-4 -> 8.29e-4, is 4.7x while SNR^2 grew ~15.7x; normalised by `h_h`, both comps improve from
42.7 d to 6 months).

### Measured in the test suite

- Table portability: two tables built at different (Nf, dt) but the same layer duration agree
  entry by entry to 4e-9 (`TablePortabilityTest`).
- Batched response vs production `SOBBHTDIonFly` per row/channel: mismatch ~1e-16 (machine
  precision) at `eval_dt = 600 s` — control: a one-day reference-epoch error gives 1.92
  (`BatchedTOFTest`).
- Lookup template vs the response's own dense TD->WDM, on the tiny `eps_freq=0.01` toy table:
  mismatch 1.4e-8..1.7e-4 per channel, norm ratios within 1.34e-4 of 1 — control without the
  quarter-turn parity rule: 0.50 (`DirectWDMTest`; per-channel numbers below).
- Move-level lookup-vs-slow lnL max |diff| 0.081 (median 0.021) at SNR^2 = 10.7, against the
  0.537 bound — control (no parity turn) 5.39 (`test_sobbh_lookup_move`).
- fdot mid-node error 0.108 on the coarse `eps_fdot=0.1` toy table, for the reference
  `get_wdm_coeffs` and the evaluator alike (table resolution: the production table has
  `eps_fdot=0.01` and SOBBH's `|fdot| <= 0.005` layer units sits in its first cell).
- The TDI-on-the-fly amplitude is **signed** (liveness is `amp != 0`, never `amp > 0`).
- The cubic-spline ringing past merger that the tracer masks measured 9e-23.

### Per-channel numbers, `test_sobbh_wdm_direct.py::DirectWDMTest::test_dense_matches_tof_own_transform`

The test itself only asserts (`mm < 1e-3`, `|ratio - 1| < 2e-3` per row/channel) and does not print;
the numbers below were obtained by re-running the test's exact `setUpClass`/body (same tiny table
from `build_tiny_table`, same `ROWS`, same `td_mismatch`) with a `print` added in a throwaway
driver script (no test or library file was modified):

```
row 0 ch 0: mm 1.378e-08 ratio 1.000001
row 0 ch 1: mm 1.362e-08 ratio 1.000000
row 0 ch 2: mm 1.400e-08 ratio 1.000000
row 1 ch 0: mm 3.118e-06 ratio 0.999949
row 1 ch 1: mm 6.496e-06 ratio 0.999947
row 1 ch 2: mm 2.614e-06 ratio 0.999926
row 2 ch 0: mm 1.647e-04 ratio 0.999884
row 2 ch 1: mm 1.639e-04 ratio 0.999866
row 2 ch 2: mm 1.722e-04 ratio 0.999908
last_stats: {'rows': 3, 'pixels': 256, 'lookup_pixels': 768, 'dropped_pixels': 0, 'merged_rows': 0}
worst mm = 1.722e-04; worst |ratio-1| = 1.341e-04
```

Range: mismatch `1.4e-8..1.7e-4` per row/channel, norm ratios within `1.34e-4` of 1 on this tiny
test table. This table is **not** a scaled-down version of the gate's production-layer table in a
"same accuracy, finer grid" sense — its `eps_freq=0.01` / `eps_fdot=0.1` node spacing is 2x
coarser in the offset axis (`eps_freq` 0.01 vs 0.005) and 10x in fdot (`eps_fdot` 0.1 vs 0.01)
than the production table's (see
the fdot mid-node-error fact above), so these numbers are a resolution-limited upper bound on this
toy table, not a smaller- or larger-scale version of the gate's own 42.7 d / 6-month measurement.

## Conventions pinned by tests

- Chunked-basis rows `(m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)`; the
  response feed is `phase = gw_phase + pi` with the intrinsic amplitude (`SOBBHTDIonFly`).
- Pixel centres `t_n = t_obs_start + n * layer_dt`, `n` the absolute grid index; the quarter-turn
  rule uses the absolute `(m + n)` parity.
- Inner products carry NO `4 * differential_component` factor (the chunked kernel convention);
  `-0.5 (h_h - 2 d_h)` is the container's source term.
- Batched response vs production `SOBBHTDIonFly` per row/channel: mismatch ~1e-16 (machine
  precision) at `eval_dt = 600 s` (`BatchedTOFTest`; the test's own assertion threshold is
  <= 1e-8, looser than what is actually measured); a one-day reference-epoch error is caught as a
  control (mismatch 1.92).

## Known limits

- Python, `xp`-vectorized (numpy / cupy) as first written; since 2026-10-02 the fused C++/CUDA
  kernel `sobbh_lookup` (`cutils/sobbh_lookup_kernel.{hpp,cu}`, `SOBBH_LOOKUP_KERNEL`, "Fused
  lookup kernel" below) does the inner products and the fill, and this Python path is its
  reference.
- A source merging inside the window keeps evaluating up to `tc` with the shared node grid
  (production `SOBBHTDIonFly` zeros the last `buffer_time` before merger instead); its pixels past
  the table's fdot axis are dropped (counted in `last_stats["dropped_pixels"]`, one warning per
  call).
- A lookup comp is single-device (it raises when called off its build device); the stock getter
  returns `SOBBHLookupRouter` (2026-10-03), which builds one replica per device on first use under
  each walker shard's device context and dispatches every call to the current device's, so
  multi-GPU walker shards work (router unit-tested with mocked devices; not yet run on a real
  multi-GPU rank).
- cubic interpolation is Keys cubic convolution, not scipy's global cubic spline.

## Follow-ups

(Items 1 and 3 are done: the fused `sobbh_lookup` kernel, 2026-10-02, and `SOBBHLookupRouter`,
2026-10-03; see the dated sections below.)

1. **CUDA/C++ kernel** in the chunked-het family: per-pixel `SOBBHTDIonTheFly::get_tdi` amplitude /
   phase at the pixel centres, the table resident on the device, the same quarter-turn rule and
   the same `(d_h, h_h)` accumulation; this Python path is its reference. GPU leads per the sprint
   rule; the comp swap is then a comp swap, not a move change.
2. **CD1L data gate**: per-source mismatch / dlogL of the lookup template vs the mojito L1 SOBBH
   streams (ids 0-5 of the 6-mo run) on the production grid — the bricks are not on this laptop.
   Recipe: `L1ProcessingStep(source_types=["sobbh"], source_ids=dict(sobbh=[id]), orbits_class=L1Orbits,
   frame="icrs")` as `scripts/emri/emri_cd1l_campaign.py` does for EMRIs, the stock SOBBH wave wrap as
   the production template, `SOBBHDirectWDM.dense` as the fast one, report `1 - Re(O)` per channel,
   norm ratio, dlogL, SNR.
3. **Per-device replicas** of the lookup comp (table + orbits/tdi wraps per device, routed per
   shard as the GB `_RoutedBandEngine` does) to lift the single-device restriction.
4. JAX mirror; the fused phase-max quadrature; F-stat / gradient methods on the comp.

## One-GPU speed test (cluster runbook)

`scripts/sobbh/sobbh_lookup_speed_gpu.py` times the lookup comp and the chunked comp head to head
on ONE device at the production 6-month grid (Nf=1440, Nt=4320, dt=2.5): `get_ll_wdm` first call /
warm (median of `--repeats`; the lookup comp is already exercised by the setup fill, so its first
call is not a from-scratch cold start), the fill, the lookup's template/inner split (device-
synchronised spans, taken from the last timed call), the memory pool per comp (`lk_pool` /
`ch_pool` = pool total GB after that comp's calls, freed between the comps; the JSON also holds
the in-use figures) and the per-row lnL difference of the two comps on the same residual.
Device-synchronised timings. Each JSON line records the grid, `m_band`, `fill_band`,
`num_m_layers`, `nt_sub` and a timestamp.

    # cluster, one GPU, the stack's python, this branch checked out
    SOBBH_LOOKUP_TABLE_PATH=/path/to/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5 \
    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda12x --rows 4,8,32,96,288 \
        --out sobbh_lookup_speed_gpu.jsonl

Reading the table: `ch/look` is warm chunked time over warm lookup time, so `ch/look < 1` means
the chunked comp was FASTER. Expect (a GPU EXPECTATION from the chunked kernel's one-block-per-row
launch, not a measurement) the chunked wall to be roughly flat in the batch size up to the
device's block capacity; the interesting columns are how `look_warm` scales with `rows` and the
`resp` (batched response) / `trac` (tracer) / `look` (lookup) vs `inner` (gathers + reductions)
split. `--row-batch` bounds
the response-spline memory (default 32); raise it on a large-memory device and watch the pool
columns. `resp`/`trac`/`look`/`inner` are not the warm median but the spans of the last timed call.

### Cluster result (2026-10-01, one GPU, cuda13x, production grid, 2nd-generation TDI, XYZ)

Table `wdm_lookup_sobbh_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5` (built on the cluster with the
production grid; the laptop EMRI table is equivalent at fixed `layer_dt`), comp build 0.6 s,
chunked comp `Nt_sub=32 n_chunks=180`, `--rows 4,8,32,96,288`, default `--row-batch 32`,
`--eval-dt 600`:

    rows  look_first  look_warm     resp     trac     look    inner  look_fill  ch_first   ch_warm   ch_fill  ch/look   max|dll|  lk_pool  ch_pool
       4       1.143      0.181    0.162    0.010    0.007    0.002      0.180     1.912     1.695     1.712     9.35  7.680e-04     2.05     1.99
       8       1.172      0.196    0.176    0.011    0.007    0.002      0.193     1.702     1.700     1.717     8.69  3.287e-03     2.16     2.01
      32       1.262      0.272    0.249    0.013    0.008    0.002      0.270     1.710     1.711     1.728     6.30  3.397e-03     2.70     2.01
      96       0.824      0.816    0.747    0.039    0.024    0.005      0.810     1.706     1.707     1.723     2.09  3.531e-03     2.73     2.01
     288       2.441      2.438    2.227    0.119    0.073    0.015      2.423     3.731     3.724     2.452     1.53  3.672e-03     2.71     2.01

The run also printed, once, the response kernel's shared-memory warning:
`lisatools td_spline TDI-on-the-fly: N=25919, scratch=544299 B exceeds device dynamic-shared
ceiling (230256 B); global-memory scratch fallback (... slower)` — `N` is the per-row response
evaluation count, 6 months / `eval_dt` = 15552000 / 600 ≈ 25919 points, about 21 B of scratch per
point.

Reading:

- The lookup is 6-9x faster than the chunked comp at 4-32 rows and 1.5-2x at 96-288. The
  chunked wall is flat (1.7 s) up to 96 rows and doubles at 288 (its one-block-per-row launch
  exceeds one wave of blocks), as expected from the laptop memo.
- The lookup wall IS the batched response build: `resp` is > 90 % of `look_warm` at every size;
  tracer + lookup + inner together are <= 0.21 s even at 288 rows (so the table evaluation and the
  inner products are solved; the response is the remaining cost).
- `resp` scales per BATCH of `row_batch = 32` rows, ~0.25 s each (288 rows = 9 batches = 2.23 s).
  A linear fit of the 4/8/32-row points gives ~0.15 s fixed per batch (the `TDTDIonTheFly`
  construction: orbits / TDI wraps, input-spline setup, output-spline fit) plus ~3 ms per row
  (the kernel on 25919 points per row under the global-memory scratch fallback, and the per-row
  output-spline fits). Both parts are tunable without new code:
  - `--eval-dt 1800` (`SOBBH_LOOKUP_EVAL_DT=1800`): 8641 points per row, ~181 kB scratch,
    under the 230 kB shared-memory ceiling, 3x less kernel and spline work per row. Accuracy is
    unchanged: the laptop gate at `--nt 1024` reproduces every `eval_dt = 600` column (per-source
    `mm`, `ratio`, `mm_w_int`, `dlogL`) to 6-7 significant digits at `eval_dt = 1800` and 1200
    (`/tmp/gate_evaldt_1800.jsonl`, `/tmp/gate_evaldt_1200.jsonl` vs the 17:45 entries of
    `docs/sobbh_lookup_gate.jsonl`; e.g. source 0 `mm_w_int` 6.5914e-6 vs 6.5915e-6), and the
    6-month gate (`--nt 4320 --rows 4 --eval-dt 1800`, 33 s on the laptop) reproduces every
    printed digit of the 17:48 `eval_dt = 600` entries (all six sources: `mm`, `ratio`,
    `mm_w_int`, `dlogL`). The response splines vary on orbital and chirp timescales, far slower
    than 1800 s. The guard `eval_dt < buffer_time / 2 = 2500 s` is the hard ceiling at the
    default buffer.
  - A finer grid costs proportionally: Mike's `--eval-dt 300` run (51833 points, 1.09 MB scratch,
    same fallback) gave `look_warm` 0.338 / 0.479 s at 8 / 32 rows (`resp` 0.317 / 0.457) vs
    0.196 / 0.272 at 600 — both the per-batch and the per-row parts of `resp` scale with the
    point count. Its `max|dll|` (8.4e-4 / 2.8e-3) is NOT an accuracy comparison across runs: the
    script's residual is the lookup comp's own fill at that `eval_dt`, and its row batches come
    from one seeded RNG consumed in `--rows` order, so a different `--rows` list scores different
    rows. Only the gate (dense TD->WDM reference) measures `eval_dt` accuracy.
  - `--row-batch 96` or `288` (`SOBBH_LOOKUP_ROW_BATCH`): pays the ~0.15 s fixed cost 3x or 1x
    instead of 9x at 288 rows; the pool grows by the response output (N x 3 x n_eval x 16 B, ~0.1 GB
    at 288 rows and `eval_dt = 600`) and the sparse template (~50 MB), both small next to the
    2.7 GB pool.
  - Hoisting the per-batch `TDTDIonTheFly` construction (one generator per comp, padded to the
    batch size) removes the fixed cost; evaluating the response only at the pixel centres plus
    derivative points, or the CUDA lookup kernel (follow-up 1), removes `resp` almost entirely.
- `max|dll|` 8e-4 to 4e-3 between the two comps on the same residual: the chunked comp truncates
  at `m_band_half_width = 3` (`--m-band`) and the lookup uses its 5-layer window; the gate's
  move-level lookup-vs-exact numbers (above) are the accuracy statement, this column is only the
  two fast paths' mutual spread.
- Memory: 2.0-2.7 GB pool per comp, comparable between the two.
- `look_first` at 4-32 rows (1.1-1.3 s) is the cupy kernel-cache warm-up of the first call;
  irrelevant to a run.

Suggested next cluster runs (seconds each; `--no-chunked` skips the 1.7 s chunked calls):

    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda13x --rows 8,32,96,288 --eval-dt 1800 --out speed_e1800_rb32.jsonl
    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda13x --rows 8,32,96,288 --eval-dt 600 --row-batch 288 --no-chunked --out speed_e600_rb288.jsonl
    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda13x --rows 8,32,96,288 --eval-dt 1800 --row-batch 288 --no-chunked --out speed_e1800_rb288.jsonl

The first keeps the chunked comp on so `max|dll|` confirms the coarser response grid on the
production grid; the warning line must be gone from its output.

Result of the first run (`--eval-dt 1800`, row batch 32; the shared-memory warning is gone):

    rows  look_first  look_warm     resp     trac     look    inner  look_fill  ch_first   ch_warm   ch_fill  ch/look   max|dll|  lk_pool  ch_pool
       8       1.072      0.108    0.090    0.010    0.007    0.002      0.107     1.775     1.700     1.716    15.72  8.364e-04     2.10     1.99
      32       1.140      0.137    0.116    0.010    0.008    0.002      0.135     1.711     1.711     1.728    12.50  2.841e-03     2.54     2.01
      96       0.417      0.410    0.349    0.031    0.024    0.005      0.405     1.707     1.707     1.724     4.16  3.531e-03     2.57     2.01
     288       1.232      1.224    1.041    0.093    0.073    0.015      1.212     3.754     3.663     2.444     2.99  3.672e-03     2.59     2.01

Every size halves against `eval_dt = 600` (0.108 vs 0.196, 0.137 vs 0.272, 0.410 vs 0.816,
1.224 vs 2.438); the lookup is 12-16x faster than the chunked comp at 8-32 rows and 3-4x at
96-288. `resp` is now ~0.116 s per 32-row batch: ~0.08 s fixed per batch (generator
construction) + ~1.1 ms per row, so the fixed part is 70 % of it and `--row-batch 288` is the
next lever (expected ~0.4 s at 288 rows). `max|dll|` at 96 / 288 rows is identical to 4 digits
across `eval_dt` 600 / 1800 (3.531e-3 / 3.672e-3): the chunked-vs-lookup spread is the m-band
truncation, row-set independent once the batch samples the jitter densely, and the response grid
does not move the lookup lnL at that level.

Result of the second run (`--eval-dt 600 --row-batch 288 --no-chunked`; the `ch_*` columns are
`nan` and `ch_pool` is meaningless without the chunked comp):

    rows  look_first  look_warm     resp     trac     look    inner  look_fill  lk_pool
       8       1.144      0.195    0.175    0.011    0.007    0.002      0.193     1.30
      32       1.262      0.271    0.250    0.013    0.008    0.002      0.270     1.84
      96       1.349      0.366    0.330    0.016    0.017    0.002      0.364     3.32
     288       1.706      0.580    0.501    0.027    0.046    0.004      0.575     7.72

One batch instead of nine at 288 rows: 0.580 s vs 2.438 (4.2x); 96 rows 0.366 vs 0.816. The
marginal cost per row FALLS with the batch (3.1 ms/row from 8 to 32 rows, 1.25 from 32 to 96,
0.89 from 96 to 288): the response kernel is one block per row and the device is still filling
up, so the per-row part is sub-linear and the per-batch fixed part (~0.16 s at `eval_dt = 600`)
is what the batch size amortises. The price is memory: the pool grows ~23 MB per row at
`eval_dt = 600` (response outputs, their spline coefficients and the per-row intermediates all
scale with the point count), 7.7 GB at 288 rows in one batch vs 2.6 GB with batches of 32;
Pick `SOBBH_LOOKUP_ROW_BATCH` from the device memory, not from the speed alone (see the third
run for the memory split).

Result of the third run (`--eval-dt 1800 --row-batch 288 --no-chunked`):

    rows  look_first  look_warm     resp     trac     look    inner  look_fill  lk_pool
       8       1.066      0.108    0.089    0.010    0.007    0.002      0.106     1.27
      32       1.112      0.135    0.115    0.010    0.008    0.002      0.134     1.71
      96       1.178      0.183    0.152    0.011    0.017    0.002      0.181     2.83
     288       1.451      0.389    0.326    0.016    0.046    0.004      0.388     6.00

Against the chunked comp's 1.70 s (3.7 s at 288 rows) this is 15.7x / 12.6x / 9.3x / 9.6x at
8 / 32 / 96 / 288 rows, and 6.3x over the original `eval_dt = 600`, row batch 32 call at 288
rows (2.438 -> 0.389 s). What remains in `resp` is ~0.08 s of generator construction per call
plus ~0.85 ms per row; tracer + lookup + inner are 0.07 s at 288 rows.

Memory: the pool at 288 rows is 6.0 GB at `eval_dt = 1800` vs 7.7 GB at 600, so only ~1.7 GB of
the one-batch peak is the response's point-count-dependent part (~9 MB per row at 600, ~3 at
1800); the other ~5 GB (~17 MB per row, independent of `eval_dt`) is the lookup's own
temporaries: the Keys cubic 16-neighbour gathers over `(rows, 3 channels, 5 layers, 4320 pixels)`
for the cos and sin tables and the `(amp, phase, f, fdot)` tracer arrays. The cupy pool keeps
that peak. Lowering it is a code change (chunk `coeffs` over pixels or rows inside `sparse`, or
`interp="linear"` for a 4-neighbour gather at the linear-interpolation accuracy); until then the
budget is ~1 GB + ~20 MB per row in one batch at `eval_dt = 1800`.

### How sparse can the response grid go? (laptop gate sweep, 2026-10-01)

Mike's question: evaluate the response sparsely (as the EMRI direct path does) and spline up to
the pixel times. The lookup already never touches the data grid: it evaluates the batched
TDI-on-the-fly response every `eval_dt` and splines (amp, whole phase) to the pixel centres
(3600 s). The sweep below (6 months, `--nt 4320 --rows 4 --no-chunked`, `--buffer-time
2.5 * eval_dt + 5000` so the guard passes and the eval grid stays inside the node grid; the gate
script grew `--buffer-time` for it) reproduces EVERY printed digit of the `eval_dt = 600` baseline
for all six catalogue-like sources (`mm`, `ratio`, `mm_w_int`, `dlogL`) at eval_dt 1800, 3600,
7200, 21600, 43200 and 86400 s, i.e. one response point per DAY (181 points over 6 months
instead of 25919 at 600 s). The per-source lookup time on the laptop CPU fell 0.22 s (1800) ->
0.08 (21600) -> 0.06 (86400); the grid change is real. Why it works for SOBBH: the whole channel
phase is the slow PN carrier plus the orbital Doppler term (~1 rad/day at 20 mHz), both with
tiny fourth derivatives (cubic-spline error ~ h^4/384 * phi''''), so even 1-day nodes are
~1e-6 rad on these sources.

IMPLEMENTED (2026-10-02, `SOBBHBatchedTOF`, tests in `tests/test_sobbh_sparse_response.py`):

- `eval_grid(t_lo, t_hi)`: uniform `linspace`, at most `eval_dt` apart, at least 4 points,
  ending EXACTLY at the window ends (the old `t_lo + k * eval_dt` overshot `t_hi` by up to one
  step), after clipping the window `DELAY_MARGIN = 600 s` inside the orbit tables' coverage
  (`orbits.t_base`): the C++ response zeroes any time it cannot serve and a half-day spline
  across that zero edge rings back into the window (the EMRI session measured 2 % in amplitude
  two intervals in on a laptop-trimmed L1 table). A window entirely outside the coverage raises.
- The guard `eval_dt < buffer_time / 2` is gone; `buffer_time >= DELAY_MARGIN` is the condition
  (the node grid must cover the TDI delays), so the production buffer of 5000 s takes any step.
- Default `eval_dt` 600 -> 43200 s (12 h) everywhere: `SOBBHBatchedTOF`, `SOBBHDirectWDM`,
  `SOBBHLookupComputations`, `SOBBH_LOOKUP_EVAL_DT`, both scripts.
- The whole-phase spline is kept (no exact-carrier residual): for an in-band SOBBH the
  carrier is too slow to need it. Measured on the test grid (10.7 days, 23 nodes at 12 h), the
  worst pixel against the 600-s tracer is the chirpiest in-band row (60 + 55 Msun at 24.5 mHz,
  fdot 5e-10 Hz/s) at the grid's last pixels: 2.9e-6 rad, 6e-11 Hz, 1.8e-13 Hz/s, amplitude
  4e-9 relative; catalogue rows < 1e-7 rad. The error scales as the step to the fourth
  (6 h: 2.0e-7, 12 h: 2.9e-6, 24 h: 4.5e-5 rad), which is why 12 h and not 24 h is the default
  (the test asserts both the 1e-5 rad bound at 12 h and that 24 h exceeds it). A source merging
  inside the window leaves the band (f > 25 mHz) and the table's fdot axis long before the
  spline degrades: a 110 Msun pair at 25 mHz still has 0.57 yr to merger and fdot 5e-10, 12x
  under the production table's 6.2e-9 Hz/s axis. The EMRI form (exact analytic carrier at the
  pixels + splined residual, `sobbh_amp_phase_batch` gives the carrier at any t, and
  `LISATDIonTheFly::get_phase_ref` is the input phase at spacecraft-1 time so the residual is
  slow) remains the fallback if the band or the mass range ever grows; it would cost a host-side
  PN evaluation at every pixel on the GPU path.
- 6-month gate at the new default (`--nt 4320 --rows 4`, buffer 5000): every printed digit of
  the 600-s baseline again.
- Laptop CPU, `--laptop` preset (Nf=180, dt=20), lookup only, `--repeats 2`, before -> after:
  the 8-row 6-month `get_ll` was 9.3 s at 600 s (earlier in this doc), now 0.59 s. The response
  is 20 % of the CPU call; the numpy table gathers (`look`) are the rest and are milliseconds on
  the GPU.

      months  rows  look_warm   resp   trac   look  inner  look_fill
         5.9     4      0.285  0.056  0.035  0.177  0.012      0.275
         5.9     8      0.588  0.094  0.064  0.399  0.025      0.549
        11.8     4      1.097  0.146  0.089  0.824  0.039      0.742
        11.8     8      1.400  0.163  0.138  1.035  0.066      1.178
        23.7     4      1.183  0.159  0.143  0.904  0.061      1.148
        23.7     8      2.378  0.267  0.246  1.695  0.115      2.262

  (`--nt 4320 / 8640 / 17280`; every span scales with the pixel count, as it should.)

Cost on the GPU after the sparse grid: the per-row part of `resp` (~0.85 ms per row at 1800 s)
scales with the point count (8641 -> 361 at 6 months), so it becomes negligible; the ~0.08 s
generator construction per call is then most of the response at <= 32 rows. The laptop stage
profile (`MBHTDIONFLY_TIMING=1`) shows the configured-orbits cache working (one 8-s
configuration per process, then hits), so that fixed cost is the kernel launch, the input-spline
build, the wraps and the output-spline fits; the cluster profile decides what to hoist. The
response part of the pool (~1.7 GB at 288 rows, 1800 s) shrinks ~25x; the lookup's own
temporaries (~17 MB per row) are untouched.

Cluster speed check at 6 / 12 / 24 months (one GPU, `SOBBH_LOOKUP_TABLE_PATH` exported as for
the earlier runs; the stage profile prints at exit):

    MBHTDIONFLY_TIMING=1 python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda13x --nt 4320  --rows 8,32,96,288 --out speed_sparse_6mo.jsonl
    MBHTDIONFLY_TIMING=1 python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda13x --nt 8640  --rows 8,32,96,288 --out speed_sparse_12mo.jsonl
    MBHTDIONFLY_TIMING=1 python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda13x --nt 17280 --rows 8,32,96,288 --out speed_sparse_24mo.jsonl

(default `--eval-dt 43200`, `--row-batch 32`; the chunked comp runs too, so `ch/look` and
`max|dll|` come out per duration. The 24-month residual slab and `invC` are 4x the 6-month
ones, ~2.4 GB; the lookup's per-row temporaries scale with the pixel count, ~70 MB per row at
24 months, so `--row-batch 32` is the safe default there.)

### Cluster result, sparse 12-h grid (2026-10-02, dev df31fe86, one H100, cuda13x)

Speed script defaults (`--eval-dt 43200 --row-batch 32`, Keys cubic at the time), chunked comp
on, `MBHTDIONFLY_TIMING=1`:

| months | rows | look_warm | resp | trac | look | inner | chunked | chunked / lookup |
|---|---|---|---|---|---|---|---|---|
| 6 | 8 | 0.051 | 0.033 | 0.010 | 0.007 | 0.002 | 1.697 | 33 |
| 6 | 32 | 0.063 | 0.041 | 0.013 | 0.008 | 0.002 | 1.703 | 27 |
| 6 | 96 | 0.191 | 0.122 | 0.040 | 0.025 | 0.005 | 1.706 | 9 |
| 6 | 288 | 0.569 | 0.366 | 0.113 | 0.073 | 0.015 | 3.721 | 6.5 |
| 12 | 8 | 0.053 | 0.034 | 0.010 | 0.007 | 0.002 | 3.396 | 65 |
| 12 | 288 | 0.617 | 0.373 | 0.114 | 0.114 | 0.015 | 7.443 | 12 |
| 24 | 8 | 0.059 | 0.040 | 0.010 | 0.007 | 0.002 | 6.828 | 116 |
| 24 | 32 | 0.085 | 0.048 | 0.012 | 0.022 | 0.002 | 6.826 | 80 |
| 24 | 96 | 0.256 | 0.146 | 0.037 | 0.065 | 0.007 | 6.814 | 27 |
| 24 | 288 | 0.757 | 0.433 | 0.107 | 0.197 | 0.022 | 14.875 | 20 |

- The sparse grid took the 8-row 6-month call from 0.194 to 0.051 s; the chunked comp scales
  with the duration (1.7 / 3.4 / 6.8 s at 8 rows), the lookup barely (0.051 / 0.053 / 0.059).
- Stage profile per response build: kernel 2.3 / 4.3 / 8.0 ms (6 / 12 / 24 months), output
  splines 2.5-3.7 ms, input splines 2.7 ms; the orbits line is ONE configuration amortised.
  The rest of the ~33-40 ms `resp` is outside those stages: the PN at the 2048 nodes, the
  uploads, Python -- the speed script now prints a `pn` column to split it.
- `resp` at 96 / 288 rows = 3 / 9 builds of the fixed cost (row batch 32): fixed below.
- `max|dll|` vs chunked 8e-4 .. 7e-3 (the chunked comp's m-band truncation, growing with the
  duration as its chunk count does); pools 2.0-4.5 GB.

### EMRI-style cost reductions (2026-10-02)

Following the EMRI direct-WDM session's path (sparse response, analytic derivatives, blocks,
one fused kernel; `plans/emri-wdm-fused-lookup-kernel.md`):

- Already in place for SOBBH: the sparse response grid, analytic spline derivatives (no
  finite-difference stencils), amplitude read at the pixel centres only, output in the active
  band only. The exact-carrier form is not needed in band (above).
- NEW: ONE response build per call for all rows; `row_batch` (`SOBBH_LOOKUP_ROW_BATCH`) now
  only blocks the tracer / lookup temporaries (`SOBBHDirectWDM.response` + `sparse_from(out,
  rows)`, `sobbh_tracer(..., rows=(lo, hi))` through the splines' `ind_interps`). Expected at
  288 rows / 6 months: `resp` 0.366 -> ~0.04-0.08 s, the call ~0.57 -> ~0.25 s.
  `tests/test_sobbh_lookup_batching.py` pins: one build per get_ll / fill, results independent
  of `row_batch` to 1e-12, the tracer's row range equals the full tracer's slice.
- NEW: `pn` span (the PN amplitude / phase at the node grid) in `last_call_spans` and the speed
  script. Laptop CPU (JAX CPU): 11 / 25 / 171 ms at 8 / 32 / 288 rows; on the cluster the next
  speed run shows it.
- NEW: the table interpolation is the table's own uniform cubic B-spline (`interp="spline"`,
  default everywhere; `SOBBH_LOOKUP_INTERP`): the fused kernels' semantics, and on the 6-month
  gate equal to or better than Keys (sources 3 / 5: mismatch 2.6e-7 / 7e-8 vs 4.0e-7 / 1.5e-7,
  integrated weighted 2.3e-7 / 8.3e-8 vs 3.5e-7 / 1.7e-7; the other four unchanged).
- PLANNED: the SOBBH fused lookup kernel (`docs/superpowers/plans/2026-10-02-sobbh-fused-lookup-kernel.md`),
  response splines -> `(d_h, h_h)` per row (or the fill) in one launch, built on the EMRI
  session's shared device helpers once those are pushed.

### Fused lookup kernel (2026-10-02, night): built, CPU-tested, ready for the cluster

`src/lisatools/cutils/sobbh_lookup_kernel.{hpp,cu}` (`backend.sobbh_lookup`): one launch from
the response splines to `<d|h>`, `<h|h>` per row (scoring) or the fill, nothing per pixel in
global memory, on the EMRI session's shared helpers (`wdm_lookup_kernels.hh`: B-spline table,
quarter turn, spline derivatives). `SOBBH_LOOKUP_KERNEL` = `auto` (default: when the module has
it and `SOBBH_LOOKUP_INTERP=spline`), `kernel` (required) or `python`. Plan + results:
`docs/superpowers/plans/2026-10-02-sobbh-fused-lookup-kernel.md`. Kernel == Python lookup to
1e-10 (scoring, routing, AET diagonal, fill, stats); four kernel mutations each caught. Laptop
CPU at 6 months, the lookup step 0.54 -> 0.09 s at 8 rows (the serial CPU build); on the GPU
the expectation is ~ms for the whole lookup at 288 rows, leaving the response build (`resp`,
with its `pn` part) as the call.

Cluster verification (mirrors the EMRI one: one unit-test line plus one speed/accuracy script):

    git pull origin dev
    pip install -e . --no-build-isolation          # REBUILDS the C++/CUDA modules (both kernels)
    export SOBBH_LOOKUP_TABLE_PATH=/path/you/built/wdm_lookup_sobbh_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5
    python -m unittest tests.test_sobbh_lookup_kernel tests.test_sobbh_lookup_batching tests.test_sobbh_sparse_response tests.test_sobbh_wdm_direct tests.test_wdm_lookup_eval -v
    bash scripts/sobbh/sobbh_speed_durations.sh

The unit line must show `test_gpu_equals_cpu ... ok` (a skip means the GPU module has no
`sobbh_lookup`: the rebuild did not take). `scripts/sobbh/sobbh_speed_durations.sh` (the twin of
`scripts/emri/emri_speed_durations.sh`; knobs BACKEND, DAYS "180 360 720", TABLE, ROWS, LOOKUPS
"kernel,python", REPS, GATE_ROWS, GATE, SPEED_ARGS, OUT) runs per duration (Nt = days * 24):
the speed script on the production grid once per lookup (the chunked comp alongside the first:
ch/look and max|dll|; the `MBHTDIONFLY_TIMING` stage profile), then the accuracy gate per lookup
(per-source mismatch / ratio / dlogL vs the dense transform, and `[gate] ... max|lnL lookup -
exact|` for the comp's scoring path). Logs and JSON lines in `sobbh_speed_durations_<date>/`.
With the kernel the speed table's `look` column is the fused kernel and `trac` / `inner` read 0;
the `[gate]` line must agree between `lookup=kernel` and `lookup=python` (same semantics; on the
laptop smoke both gave 1.178e-05).

### Aligned test setup + mojito bricks (2026-10-02, evening)

The three steps share `scripts/sobbh/_sobbh_testbox.py` with the EMRI / MBH setups: DAYS 180 360
720; the run box 0.25-25 mHz with EDGE_CROP_WAVELETS = 60 layers cropped per end; every SNR and
score against scirdv1 + the fitted tanh galactic foreground at the window's Tobs
(`FOREGROUND=off`: instrument only). `sobbh_speed_durations.sh` gained step 3,
`scripts/sobbh/sobbh_lookup_mojito.py` (MOJITO=auto): every SOBHB source with an L1 brick, the
catalogue row through the stock mapping, the brick's L1Orbits (ICRS), MOJITO_REFERENCE_TIME,
the window from brick start + 5e4 s; the lookup and the production template scored against the
brick (snr, data_snr, mm, logL, snr_ratio, snr_det, mm_vs_production). Laptop, bricks 0 and 1,
8-16 d with the foreground: production vs mojito mm 5e-12..2e-8, lookup vs production
1e-7..3.3e-6 (the harness without the edge crop had read ~0.1: the window edges).

Fixed on the way: the response coverage came from `orbits.t_base`, which for mojito L1Orbits
ends at REF + 449 d although the configured sc / ltt tables run to REF + 730.5 d (24-month
windows would have lost everything after day 449); it is now the configured span
(`SOBBHBatchedTOF.orbit_span`), and pixels outside the response coverage are a zero template on
both lookup paths (`SOBBHDirectWDM.covered_pixels`; `OrbitCoverageTest`).

### THE DEFAULT (2026-10-03)

`SOBBH_LIKELIHOOD` now defaults to `lookup` everywhere (`SourceSOBBHSettings.likelihood`, the
6-month launcher); `chunked` (the 2026-07-30 default) and `full` stay selectable. Multi-GPU
walker shards go through `SOBBHLookupRouter` (one comp per device), so the launcher's
`GPUS_PER_RANK > 1` refusal is gone. A run without `SOBBH_LOOKUP_TABLE_PATH` finds or builds the
canonical n_ref table in its run folder (the EMRI recipe; one build per grid, minutes on a GPU,
much longer on a CPU -- point CPU / laptop runs at an existing table with the run's 3600-s layer,
e.g. `wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5`).

### The table during a global-fit run (2026-10-02, the EMRI way)

`SOBBH_LOOKUP_TABLE_PATH` is now OPTIONAL. `resolve_sobbh_lookup_table(general_info, cfg)`
(`source_runtime.py`) returns `(path, status)`:

- set: that file (`"explicit"`; must exist; the comp getter then checks its LAYER duration
  against the run's, so any `(Nf, dt)` with the same `Nf * dt` serves, e.g. the laptop
  `NF180_DT20` table on the `NF1440_DT2p5` grid);
- unset: the canonical n_ref table of the run folder (`general_info.file_store_dir`),
  `wdm_lookup_store.lookup_table_path(None, file_store_dir, Nf, dt)` -- the EMRI recipe, so it
  is the SAME file `EMRI_LIKELIHOOD=direct` uses
  (`wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5` in production; Mike's cluster SOBBH
  table was built with exactly that recipe, only named differently) -- found there
  (`"found"`) or built and saved there first (`"built"`; one builder under `<path>.lock`, the
  other ranks `"waited"`), so a restart never rebuilds. Neither a path nor a run folder raises.

The 6-month launcher (`scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh`) grew the matching
block after the EMRI one (the 9mo v9 and flagship launchers carry it too): `SOBBH_LIKELIHOOD`
(default `chunked` on 10-02; `lookup` since 10-03, see above), `SOBBH_LOOKUP_TABLE_PATH`
(default unset = the run folder's canonical file), `SOBBH_LOOKUP_EVAL_DT` (43200),
`SOBBH_LOOKUP_ROW_BATCH` (32), and a SOBBH PREFLIGHT that resolves the knobs through
`SourceSOBBHSettings` (an unknown env var is silently ignored), refuses a lisatools without
the lookup comp, refused `GPUS_PER_RANK > 1` (dropped on 10-03 with the router), and finds -- or builds and saves -- the table on
the submitting node's GPU before `mpiexec`, checking its layer duration. Launch:

    SOBBH_LIKELIHOOD=lookup NGPUS=4 ./scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh

Tests: `tests/test_sobbh_lookup_stock.py::LookupTableResolutionTest` (found / built-and-saved
then found on restart / explicit wins / explicit must exist / explicit checked against the
layer duration, same-layer other sampling accepted); the store's own tests cover the lock,
the wait and the atomic build. The launcher preflight was dry-run on the laptop for the
explicit-table, chunked, `GPUS_PER_RANK=2` and missing-table cases.

Production settings from these runs: `SOBBH_LOOKUP_EVAL_DT=1800` (superseded by the 12-h
default above); `SOBBH_LOOKUP_ROW_BATCH` =
the move's row count when ~20 MB per row fits next to the other comps, else 96 (0.18 s per
batch; 288 rows in three batches ~0.55 s vs 0.39 in one).

### Pre-cluster CPU smoke (historical)

GPU results were not measurable on the laptop (no CUDA backend). The CPU smoke below
(`--backend cpu --laptop --nt 256 --rows 2,4 --repeats 1`) has
the lookup SLOWER than the chunked comp, and it must not be read as a GPU result: the smoke grid
is tiny (256 layers), so the lookup's fixed per-call template build (~4-7 s here, independent of
rows and of the grid: it was also ~7 s on the production grid) dominates; another session's Python
process was running during the original smoke, so the CPU numbers are contended (the Task 7 gate
measured 4.6 s per 8-row lookup call on a 1024-layer grid on the same laptop); and none of this
says anything about the GPU, where the per-row work is parallel. The CPU smoke scales with rows
(chunked 1.10 s at 2 rows, 2.19 s at 4), unlike the GPU expectation above. The cluster run is the
measurement.

CPU smoke, laptop preset:

    backend cpu  grid Nf=180 Nt=256 dt=20.0 layer_dt=3600 s  active 178 x 256  table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5  comp build 0.1 s
    chunked comp: Nt_sub=32 n_chunks=11 build 4.5 s
     rows  look_first  look_warm     resp     trac     look    inner  look_fill  ch_first   ch_warm   ch_fill  ch/look   max|dll|  lk_pool  ch_pool
        2       7.570      4.495    4.463    0.003    0.007    0.001      4.448     1.135     1.204     1.669     0.27  3.317e-05      nan      nan

(re-run after the span split: the lookup call's cost is the batched response build, `resp`; the
tracer and the lookup itself are milliseconds on this grid. The 4-row smoke row and the
production-grid table below predate the column rename: `tmpl` there is `resp + trac + look`.)

Production grid on CPU, 2 rows, lookup only (`--backend cpu --rows 2 --repeats 1 --no-chunked`;
completed, no watchdog exit; proves the Nf=1440, Nt=4320, dt=2.5 grid runs):

    backend cpu  grid Nf=1440 Nt=4320 dt=2.5 layer_dt=3600 s  active 179 x 4320  table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5  comp build 0.1 s
     rows  look_first  look_warm     tmpl    inner  look_fill  ch_first   ch_warm   ch_fill  ch/look   max|dll|  lk_pool  ch_pool
        2       6.507      5.443    7.170    0.006      9.528       nan       nan       nan      nan        nan      nan      nan
