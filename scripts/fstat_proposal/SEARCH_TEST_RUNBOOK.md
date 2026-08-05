# F-stat search-in-global-fit test runbook (zero-leaf start)

Runs the reversible-jump **F-stat birth search end-to-end through the stock
`gb_no_fg` pipeline**, starting from **zero GB leaves** — the sampler must
*find* the catalogue sources via F-stat births (`GB_MODE=search`). This is
the real test of the F-stat proposal, distinct from the injection-at-truth
(`GB_MODE=pe`) convention checks.

## How the proposal FITS and DRAWS (post sub-band refactor)

**FIT (one grid-prep command per band configuration —
`plot_fstat_proposal_mojito.py`):**

1. **Band** from the stock knobs (`GB_MIN_FREQ`/`GB_MAX_FREQ`,
   `GB_CENTER_FREQ`+`GB_N_LAYERS`, or `GB_HIGHEST_FREQUENCIES=N`); the stock
   build derives `band_edges` → sub-bands and `f0_lims`. Catalogue sources
   OUTSIDE the band stay in the data by default; `GB_SUBTRACT_OUT_OF_BAND=1`
   removes them as known signals.
2. **Comb scan**: ONE batched kernel sweep over (all f0 nodes at `1/(2*Tobs)`
   spacing spanning every interior sub-band) × (6 golden-spiral sky points),
   Mc held fixed (fdot derives from `(f0, Mc)`, never scanned; the
   extrinsics are analytically maximized inside F).
3. **Peak selection** (vectorized, on-device): two-tier candidate set
   (window champions ± the Doppler `min_sep`, plus EVERY strict local
   maximum — the rescue is absolute, never relative to the window max) →
   SNR floor (`FSTAT_PEAK_MIN_SNR`, or `FSTAT_PEAK_MIN_F` = SNR²/2) →
   per-band cap `FSTAT_PEAKS_PER_BAND` (~35). K peaks, each tagged with
   its sub-band.
4. **Local grids**: each peak's 4-D box — f0 = peak ± half-width CLAMPED to
   its sub-band edges; Mc = `[FSTAT_MC_MIN, m_chirp_lims[1]]`; full
   alpha/sin_delta. f0 nodes default to ~1/Tobs cells (the peak width,
   auto ~40 at 90 d); Mc/sky stay anisotropic-coarse (3/8/8). ALL K boxes
   swept in ONE chunked kernel stream → stacked `(K, n_f0, n_Mc, n_alpha,
   n_sd)` grids — **the production sampling layer**.
5. **Save** one `*_peaks_stacked.npz` (grids + metadata, + the
   `*_comb.npz`). `FSTAT_FIT_GMM=1` additionally fits the optional batched
   per-box GMM (`vec_fit_gmm_min_bic`) into the cache.

**DRAW (search runtime — `run_fstat_rj_search.py`, one global
distribution):**

1. `floor( mixture[ stacked peak grids (w ∝ F or equal), comb (linear-in-F
   f0 × uniform Mc/sky, weight 0.3 by default — see the comb study) ],
   uniform floor eps over the whole f0_lims box )`, wrapped into the
   8-column GB container (uniform lnA/phi0/cos ι/ψ; phi0 re-maximized
   analytically at scoring).
2. `rvs`: one call fills ALL empty leaves across temps/walkers/bands — the
   peak component draws from the stacked grids (exact inverse-CDF;
   on-device for cupy runs).
3. The move bins each draw to its sub-band (`searchsorted(band_edges)`);
   the per-cell gate rejects out-of-band births; the progressive leaf cap
   gates counts.
4. `logpdf`: one vectorized evaluation supplies detailed-balance factors
   for births AND deaths (the floor keeps death factors finite everywhere).
5. Accepted births refine via the in-model machinery (information-matrix
   jumps + group stretch; two-quadrature phase maximization).

`FSTAT_PEAK_SAMPLING=gmm` swaps the peak component for the batched GMM
fitted FROM the grids — the memory-light option (~MBs vs the full stack);
on that path a **fidelity gate** logs GMM-vs-grid logpdf stats at startup
(`[gate] ...`). TODO(fstat-gmm-deprecation): may be retired once the grid
path proves out at full-band GPU scale.

From-zero tests MUST run `GB_MODE=search`: it arms the progressive
per-band leaf cap (start 1; the per-band source count only gradually
increments — plateau-gated by `GB_LEAF_CAP_MIN_ITERS` — so the sampler
climbs the likelihood one recovered source at a time).

## Knobs

Band (SAME knobs for prep + search — one setting sizes both identically):

| knob | meaning |
|---|---|
| `GB_MIN_FREQ` / `GB_MAX_FREQ` | direct band bounds (Hz), snapped to WDM layers |
| `GB_CENTER_FREQ` + `GB_N_LAYERS` | center + layer count (overrides min/max) |
| `GB_HIGHEST_FREQUENCIES=N` | center on the N-th highest catalogue source |
| `GB_SUBTRACT_OUT_OF_BAND=1` | subtract ALL out-of-band catalogue sources (default 0 = keep) |

The band width is the **cost knob**: a few layers (few sub-bands) is
laptop-testable; the full band is GPU scale. `GB_CENTER_FREQ_HZ` is a
deprecated alias for `GB_CENTER_FREQ`.

Grid prep (`plot_fstat_proposal_mojito.py`):

| knob | default | meaning |
|---|---|---|
| `FSTAT_GRID_CACHE` | — | npz base path (writes `*_comb.npz`, `*_peaks_stacked.npz`) |
| `FSTAT_COMB_CACHE_REUSE=1` | off | reuse the comb; selection re-runs with current knobs |
| `FSTAT_PEAK_MIN_SNR` / `FSTAT_PEAK_MIN_F` | SNR 5 (F 12.5) | absolute selection floor (F = SNR²/2; explicit SNR wins over explicit F; gates both tiers) |
| `FSTAT_PEAKS_PER_BAND` | 35 | per-sub-band peak cap (production complement) |
| `FSTAT_PEAKS_TO_FIT` | all | global cap on fitted peaks (laptop brake) |
| `FSTAT_PEAK_HALF_MHZ` | 2.5e-3 | box f0 half-width (clamped to the sub-band) |
| `FSTAT_MC_MIN` | 0.01 | Mc grid-box floor (prior stays full `m_chirp_lims`) |
| `FSTAT_N_F0` | auto ~1/Tobs | f0 nodes/box (auto ≈ 40 at 90 d, clamp [12, 96]) |
| `FSTAT_N_MC/_ALPHA/_SINDELTA` | 3 / 8 / 8 | anisotropic Mc/sky nodes; `FSTAT_N_PER_AXIS` overrides all four |
| `FSTAT_BATCH` | 4096 | kernel rows per call (raise on GPU) |
| `FSTAT_GRID_MEM_MB` | — | K-axis memory budget for the stacked grids |
| `FSTAT_FIT_GMM=1` | off | ALSO fit the optional GMM layer into the cache |
| `FSTAT_GMM_SAMPLES` / `FSTAT_GMM_MAX_COMP` | 4096 / 12 | GMM fit knobs |
| `FSTAT_SAVE_PER_BOX=1` | off | also write legacy per-box `*_peak<i>.npz` |

Laptop brakes: `FSTAT_N_F0=12 FSTAT_PEAKS_TO_FIT=3` reproduce the smoke
cost (12×3×8×8 ≈ 3 min/peak on CPU); the auto f0 density is GPU scale.

Search (`run_fstat_rj_search.py`):

| knob | default | meaning |
|---|---|---|
| `FSTAT_GRID_DIR` | — | dir with the `*_peaks_stacked.npz` (+ `*_comb.npz`) — the standard grid source |
| `FSTAT_PEAK_GRIDS` | — | TEST-ONLY legacy per-box npz list (stacked + GMM-fitted on load) |
| `FSTAT_PEAK_WEIGHTING` | `fstat` | peak-box weights `w ∝ F`; `equal` flattens |
| `FSTAT_COMB_CACHE` / `FSTAT_COMB_WEIGHT` | auto / 0.3 | comb component (0.3 from the 2026-07 study: ~linear un-boxed coverage gain, ~9% boxed dilution, no added spurious mass) |
| `FSTAT_FLOOR_EPS` | 0.1 | uniform-floor weight (finite death factors) |
| `FSTAT_PEAK_SAMPLING` | `grid` | production stacked-grid sampling; `gmm` = memory-light option (gate-checked; may be deprecated) |
| `FSTAT_RESUME=1` | off | keep the backend, continue from its last sample |

## Cost / machine notes

- CPU anchors: F-stat ≈ 76–79 ms/eval; a 12×3×8×8 box = 2304 evals ≈ 3 min;
  a 3-layer comb ≈ 18 min. All sweeps are single chunked streams — on GPU
  (`USE_GPU=1 GPU_BACKEND=cuda12x GPUS=0`, `FSTAT_BATCH` large) the full
  band is the production path.
- The grids are **chirp-mass** basis: run the search with
  `GB_USE_CHIRP_MASS=1` (the default) — do **not** set 0 here. (fdot<0
  interacting DWDs are out of the (f0, Mc) proposal's reach — see the
  `(f0, fdot)` TODO in `lisatools/sampling/fstat_proposal.py`; pick clean
  sub-bands or use `GB_SUBTRACT_NEIGHBORS=1` / `GB_SUBTRACT_OUT_OF_BAND=1`.)
- `NWALKERS`/`NTEMPS`/`NITER`/`GB_NUM_REPEAT_PROPOSALS` + the band knobs are
  the levers; values below are a reasonable full-run baseline. Laptop smoke:
  `NWALKERS=8 NTEMPS=4 NITER=6 GB_NUM_REPEAT_PROPOSALS=10` + a narrow band.

## 0. Sync (pure-Python changes — no rebuild for an editable install)

```bash
cd LISAanalysistools && git pull origin dev
cd ../Eryn         && git pull origin dev
cd ../LISAanalysistools
G=gf_runs_fstat_rj/grids ; mkdir -p "$G"
```

If the mojito data isn't at `~/.mojito_cache/brickmarket/mojito_light_v1_0_0/`,
prefix every command below with `MOJITO_DATA_PATH=/your/path`.

## Test 1 — highest-frequency single source

### 1a. Generate the grids

```bash
OMP_NUM_THREADS=1 \
GB_HIGHEST_FREQUENCIES=1 GB_N_LAYERS=5 \
  FSTAT_GRID_CACHE=$G/fstat_grid_highest.npz \
  python scripts/fstat_proposal/plot_fstat_proposal_mojito.py
ls $G/fstat_grid_highest_*.npz   # -> _comb.npz + _peaks_stacked.npz
```

### 1b. Search from zero leaves

Same band knobs as 1a (that is the point of the unification).

```bash
OMP_NUM_THREADS=1 \
GB_MODE=search GB_HIGHEST_FREQUENCIES=1 GB_N_LAYERS=5 GB_USE_CHIRP_MASS=1 \
TOBS_TARGET=7776000.0 NWALKERS=16 NTEMPS=6 NITER=15 GB_NUM_REPEAT_PROPOSALS=25 \
GB_LEAF_CAP_MIN_ITERS=2 \
FIT_DIR=./gf_runs_fstat_rj/search_highest/ \
FSTAT_GRID_DIR=$G FSTAT_COMB_WEIGHT=0.3 \
  python scripts/fstat_proposal/run_fstat_rj_search.py
```

(If `$G` holds stacked caches from several bands, keep one per directory —
the runner expects exactly one `*_peaks_stacked.npz`.)

### 1c. Diagnose

```bash
python scripts/fstat_proposal/diag_fstat_rj_search.py \
  gf_runs_fstat_rj/search_highest/gb_no_fg_test_2_testing.h5 highest \
  gf_runs_fstat_rj/search_highest/diag.png
```

**Pass:** a leaf is born at ~20.380 mHz within a few iterations, the cold logL
climbs and plateaus, the final leaf's f0 sits on the catalogue source.

## Test 2 — multi-source middle band

Any middle band works; the historic "band75" test is the stock default band
(~7.36–7.78 mHz, 15 sources — no band knobs needed), and a slid band is just
`GB_CENTER_FREQ=<Hz> GB_N_LAYERS=<n>`.

### 2a. Generate the grids

```bash
OMP_NUM_THREADS=1 \
  FSTAT_GRID_CACHE=$G/fstat_grid_band75.npz \
  python scripts/fstat_proposal/plot_fstat_proposal_mojito.py
# with an existing *_comb.npz add: FSTAT_COMB_CACHE_REUSE=1 (selection +
# stacked Stage B + GMM re-run from the cached sweep with current knobs)
```

### 2b. Search from zero leaves

```bash
OMP_NUM_THREADS=1 \
GB_MODE=search GB_USE_CHIRP_MASS=1 \
TOBS_TARGET=7776000.0 NWALKERS=16 NTEMPS=6 NITER=15 GB_NUM_REPEAT_PROPOSALS=25 \
GB_LEAF_CAP_MIN_ITERS=2 \
FIT_DIR=./gf_runs_fstat_rj/search_band75/ \
FSTAT_GRID_DIR=$G FSTAT_COMB_WEIGHT=0.3 \
  python scripts/fstat_proposal/run_fstat_rj_search.py
```

### 2c. Diagnose

```bash
python scripts/fstat_proposal/diag_fstat_rj_search.py \
  gf_runs_fstat_rj/search_band75/gb_no_fg_test_2_testing.h5 band75 \
  gf_runs_fstat_rj/search_band75/diag.png
```

For a generic (non-legacy) band pass any label — the diagnostics annotate
the loudest in-band catalogue source automatically.

**Pass:** births land only on real catalogue sources (zero spurious), walkers
consolidate onto the loudest, the per-band leaf cap advances toward the
in-band complement; the startup `[gate]` line shows a sane GMM-vs-grid match
(|median Δlogp| ≲ 1, tiny outside-box fraction).

## Test 3 — GPU full band (production shape)

On the GPU machine:

```bash
USE_GPU=1 GPU_BACKEND=cuda12x GPUS=0 OMP_NUM_THREADS=1 \
FSTAT_BATCH=65536 \
  FSTAT_GRID_CACHE=$G/fstat_grid_full.npz \
  python scripts/fstat_proposal/plot_fstat_proposal_mojito.py
# then the search with FSTAT_GRID_DIR=$G and the same (full) band knobs
```

**Pass:** peaks populate multiple interior sub-bands (per-band counts in the
log), one chunked sweep line per stage (no per-sky/per-peak kernel loops),
and the from-zero search recovers catalogue sources in every sub-band with
signal.

## Notes

- The `SOURCE-ONLY residual` per-band metric is an **injection-at-truth** debug
  aid (`GB_MODE=pe`); it is not meaningful for a from-zero search, so ignore it
  here — judge the search by the diagnostic plot (births vs catalogue).
- pe-mode regression (validated baseline): `GB_HIGHEST_FREQUENCIES=1
  GB_MODE=pe GB_START_FACTOR=0` → sub-band SOURCE-ONLY residual T0 ≈ 0
  (was −0.015); with all 15 band75 sources + `GB_SUBTRACT_NEIGHBORS=1`,
  T0 ≈ −0.09.
- Legacy per-box caches still load through the TEST-ONLY
  `FSTAT_PEAK_GRIDS=<comma list>` override (they are stacked + GMM-fitted on
  load; all boxes must share one `FSTAT_N_*` shape).

---

## Test 4 — 6–8 mHz, exact mirror of the 2026-08-03 overnight run + the new updates

Reproduces the `overnight_2` run (`gf_runs_fstat_rj/overnight/`,
`gb_no_fg_test_2`) knob-for-knob, changing ONLY the new machinery. Every
setting below that is not under "What changed" was read back out of that
run's `run_settings.log`, so this is a like-for-like.

Same entry point as before — `run_fstat_rj_search.py` — because it is what
supplies the run's noise model (`MojitoNoiseEstimates(noise_file,
which="xyz")` + `fixed_psd_kwargs`, i.e. the empirical NOISE-brick table, not
the analytic sensitivity). `run_global.py --stock gb_no_fg` would NOT
reproduce that.

### What changed vs. overnight_2

| | overnight_2 | now |
|---|---|---|
| F-stat grids | offline prep → `FSTAT_GRID_DIR` | **fitted in-move**, first proposal that needs them (`GB_FSTAT_FIT_IN_MOVE=1`) |
| in-model scoring | chunked-het | **sig-het v5** (`SIGHET_V5=1` + the three knobs it is gated on) |
| info matrices | borrowed from a shared nearest-in-frequency table | **per-block exact, borrowing retired** (`GB_INFOMAT_PER_BLOCK=1`, chunked backend) |
| leaf cap | progressive, start 1 | **off** — all leaves available immediately |
| acceptance | not logged | `[GB_ACCEPT]` per proposal |
| GPUs | `[0]` | `[0]` first, then 2+ |

`num_repeat_proposals` was ALREADY 100 in overnight_2 — unchanged.

### Rebuild

**GBGPU must be rebuilt** in the target env: the v5 kernel and bindings are
new native code (`f4c54dc` → `gb_tdi_on_the_fly.cu/.hh`,
`binding_gbgpu.cxx/.hpp`). **LAT needs no rebuild** — the 2026-08-04 changes
are pure Python.

### Configuration

```bash
export RUN=./gf_runs_fstat_rj/overnight_v5/
mkdir -p $RUN

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export USE_GPU=1 GPU_BACKEND=cuda12x

# ---- identical to overnight_2 ----
export NITER=100 NWALKERS=16 NTEMPS=6
export GB_NTEMPS=12                    # branch ladder (general NTEMPS is 6)
export GB_MODE=search GB_USE_CHIRP_MASS=1
export GB_MIN_FREQ=6.0e-3 GB_MAX_FREQ=8.0e-3   # snaps to 6.1111-7.9167 mHz,
                                               # 13 sub-bands (layers 44..57)
export TOBS_TARGET=7776000.0           # 90 d
export GB_NUM_REPEAT_PROPOSALS=100
export GB_N_SUBBANDS=1024
export GB_SEARCH_PRIOR_REMOVAL=1
export GB_LEAF_CAP_ITER_ONLY=1
export FIT_DIR=$RUN

# ---- new: in-move F-stat fit (no offline grid prep) ----
export GB_FSTAT_FIT_IN_MOVE=1
export GB_FSTAT_FIT_DIR=$RUN/gb_fstat_fit
export FSTAT_PEAKS_PER_BAND=100        # saturating cap (200 == 100)

# ---- new: sig-het v5 in-model scoring ----
# ALL FOUR required: v5 is gated on v4_knots, and v5=1's phase-aliased
# arena is gated on v4_band. A guard raises if they disagree rather than
# silently running v3 or the v5=2 control arm.
export GB_SIGHET_INMODEL=1 SIGHET_V5=1
export SIGHET_V3_NODES=64 SIGHET_V4_KNOTS=128 SIGHET_V4_BAND=16
export SIGHET_NT_LAYER=512

# ---- new: per-block EXACT info matrices, all borrowing retired ----
# Chunked backend. Do NOT set SIGHET_INFOMAT: its fast route is not
# reachable from the move yet (2026-08-04 audit), so it would change
# nothing except to look like it had.
export GB_INFOMAT_PER_BLOCK=1

# ---- new: no source cap ----
export GB_LEAF_CAP_START=              # EMPTY disables the cap.
                                       # "0" would cap at ZERO leaves.
export GB_SEARCH_IN_MODEL=0            # the pure in-model move is redundant
                                       # once every source gets its own info
                                       # matrix; set 1 to keep it.
```

### 4a. One GPU — shakedown

```bash
GPUS=0 NITER=5 \
  python scripts/fstat_proposal/run_fstat_rj_search.py 2>&1 | tee $RUN/shakedown.log
```

**Pass:**
- `GB_FSTAT_FIT_IN_MOVE=1: ... skipping the offline grid load.`
- `GB in-model likelihood: SIGNAL-HET` (the v5 guard did not raise).
- One in-move fit runs, leaving `gb_fstat_fit/<move>/epoch_0000/` with
  `fstat_grid_comb.npz`, `fstat_grid_peaks_stacked.npz`, `DONE.json`;
  iteration 2+ does **not** refit.
- Peaks populate multiple of the 13 interior sub-bands.
- `[GB_ACCEPT ...]` shows non-zero rj and in-model acceptance.
- `[GB_TIMING]` reports `inmodel_cholesky` per block (per-block info
  matrices being computed, not borrowed).

### 4b. One GPU — overnight

Same block with `NITER=100`. The `epoch_0000/` grid from 4a is reused when
`GB_FSTAT_FIT_DIR` is unchanged, so the fit cost is paid once.

### 4c. Two or more GPUs

Only once 4a/4b look right:

```bash
GPUS=0,1 NITER=5 \
  python scripts/fstat_proposal/run_fstat_rj_search.py 2>&1 | tee $RUN/shakedown_2gpu.log
```

**Pass:** the initial log-likelihood matches the 1-GPU run **bit-identically**
(the established multi-GPU gate — sharding changes where work runs, never the
answer), and per-device comp/engine replicas are built once per non-primary
device.

### Diagnose (unchanged)

```bash
python scripts/fstat_proposal/diag_fstat_rj_search.py \
  $RUN/gb_no_fg_test_2_testing.h5 band68 $RUN/diag.png
```
