# Deep dives on a production run (a guide for Claude)

Who this is for: a Claude Code session asked to examine a running or
finished global-fit production run ("process this tar", "how is the gating
going", "why is this walker stuck", "is the run healthy") without the
context of the sessions that did it before. It records how those deep dives
were done, what we look for, what healthy and sick look like, and the rulings
that constrain what you may conclude. Where things live is in
[`global-fit-navigation.md`](global-fit-navigation.md); read its sections 3
to 7 first if the log tags below mean nothing to you.

_Written 2026-10-11 against `dev` 8917eac2, during the 9mo v9 run (jobs
751-754). Every number quoted is a point-in-time reference, not a target._

---

## 1. Working rules (user rulings; in force)

1. **A deep dive is done by the session itself**, not delegated, when the
   user asks for one. Delegation is for raw-number extraction only
   (section 2.4). Give quick takeaways mid-turn on a long dive; the user
   reads them.
2. **A claim that production has been running wrong comes with the evidence
   chain AND a falsification attempt in the same message.** Name the log
   lines, the rows, the code path; say what would disprove it and whether
   you tried.
3. **Quote the `[GB_GATE]` census line for valve state.** Never infer the
   valve from leaf counts. Read `[MAXLOGL]`'s FINAL line per invocation before
   saying "converged"; the per-round lines and the sub-backend counters show
   only the last round.
4. **Check what a stored statistic IS before reading it as a likelihood.**
   (`band_cold_logl_w` was the data term only until da45aa98; `h_h` is SNR^2
   under the walker's own noise; `mm` in the arm caches is the OVERLAP.)
5. **Verify the baseline before blaming new code.** Run the control
   (`git show HEAD:<file>` into a temp path) under the same harness; check the
   store's mtime before theorising about a changed output; a tar's row count
   grows while you work.
6. **A silent absence of errors from code that does not run is not health.**
   Confirm the path executed (a log line, a counter that moved) before
   crediting it.
7. **One reading of a moving system does not license a standing claim.**
   Say "not in danger right now, re-check in N minutes", never "fine".
8. **Judge the galfor CURVE, never `amp`** (the parameters are degenerate:
   exp-vs-tanh, alpha-f_1). The 3-5 mHz foreground level at 6-9 months is a
   GB-completeness readout, not a foreground measurement.
9. **Gating changes need a unit + mutation test layer AND a replay on real
   store data** before shipping; replay a resume with the LAUNCHER's knobs,
   never a test fixture's. Say what counters mean ("cycle 1 of 20" is read
   literally).
10. **Push only on the user's word** (no auto commit / push). When the user
    is waiting on a cluster relaunch, push as soon as what the cluster needs
    is verified; do not hold it for laptop-only smoke tests.
11. **Laptop hygiene** (the originating laptop: 8 GB RAM, several Claude
    sessions at once): one python process at a time across every session,
    thread pools pinned to 1 (`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`). The one check:
    `pgrep -fl "bin/python" | grep -vE "vscode|Code Helper|pet server"`.
    The one wait form: `while kill -0 PID 2>/dev/null; do sleep 10; done`,
    on a PID captured at launch. Never wait with a `pgrep -f` whose pattern
    appears in the waiting shell's own argv (it self-matches and waits
    forever; hit three times). Never read a multi-MB log wholesale; stream
    tars with `tar -xOzf ... | grep`; `rm -rf` the `from_tar` cache when the
    disk fills; swap `used/total >= 0.85` means wait.

---

## 2. Inputs and how to trust them

### 2.1 What arrives

| input | where it lands | notes |
|---|---|---|
| snapshot tar | the LAT root, `gf_prod_<run>_snapshot.tar.gz` (full) or `gf_prod_<run>_short.tar.gz`; browser downloads suffix ` (N)` before `.gz` | built by `make_snapshots.sh` or the saver (`monitor/snapshot.py`). Check it is fresh: mtime, `gzip -t`, md5, size stable (a download in progress has a round size). Compare the md5 with the PREVIOUS tar's: a byte-identical re-download means the cluster-side tar did not refresh (a saver / hook stall on the run), not a download problem. |
| the monitor page the user downloaded | the LAT root, `gf_prod_<run>_monitor (N).html` | the in-run page from the monitor rank; the user reads it, you may be asked why a panel looks wrong. |
| a live run dir (on the cluster) | `/shared/data/global_fit_output/gf_prod_<run>/` | reading the live store races the saver; readers fall back to the running backup copy per row. |

### 2.2 What is in a tar

Full tar: the reduced `*_extract.h5` (keep 5 rows of the big chains, 12 of
the cold ones; full history of everything small), the residual snapshot npz,
eigen pkl, `gpu_util_<job>.csv`, `gb_fstat_fit/*/epoch_*/DONE.json`, setup
logs, the head + rank run logs (grep-filtered above 200 MB, plus a raw
tail), the newest `slurm_stdout_<job>.log` files, the truth npz, the warm
start npz, the html. Short tar: the extract at keep 1, filtered logs only,
no truth npz (so no overlap panels).

Facts that decide what you can read from it:

- **`slurm_stdout_<job>.log` is the superset.** Every run-log line plus the
  `print()` families (`[GF_TIMING]`, `[V9-SEED]`, the `[combined]` stage
  table, `[r<k>/saver]`, python warnings, cupy device warnings). Read it
  raw. The `*_filtered.log` products DROP the gate families (`[GB_GATE]`,
  `[V9-STAGE]`, `[GB_IMCONV]`, `[GF_TIMING]`, `[GB_STAGE]`).
- **The extract is the live store** for the small arrays and a keep-window
  store for the big ones. Keep-window only (`LAST_K_ONLY` in
  `monitor/_store_extract.py`; 5 rows in a full tar, 1 in a short one):
  `chain/gb`, `chain/vgb`, every `sub_backend/gb` per-rung array (`chain`,
  `inds`, `d_h`, `h_h`, `band_num_binaries`, `band_temps`, the `band_num_*`
  and `band_swaps_*` counters, `band_cold_ll`) and `sub_backend/vgb/chain`.
  Full history: the main `inds` / `log_like` / `saved_after`, the valve `_w`
  arrays and `band_rj_shutoff`, the noise chains, the MBH / EMRI / SOBBH
  sub-backends. Rows outside the keep window read back as ZEROS, not NaN,
  with no error. Stage-end extrapolation from a short tar's band arrays is
  impossible (last row only).
- **Two h5 files in a run dir**: the live one is the newest mtime with the
  highest `global_fit.attrs["iteration"]`; name it in the readout. A
  previous format's full h5 can survive a tar refresh.
- **Clocks**: log timestamps are UTC (cluster); tar mtimes print in the
  laptop zone; mtimes inside a pulled zip are the copy time. The dashboard
  anchors ages to the file mtime so it does not cry "clock skew" on a tar.
- **Mid-flush**: sub-backends may lag the main backend by one row. Expected;
  report it.
- **Rows past `attrs["iteration"]`** hold stale values from an earlier job
  (a rewound store). Slice by the attr.

### 2.3 The routine that worked (per tar)

1. Verify freshness (above). Extract into the scratchpad
   (`snap<run>_tNN/`), or stream: `tar -xzf TAR -C DIR <run>/slurm_stdout_<job>.log`
   then grep.
2. Run the dashboard on the extracted run dir:
   `python scripts/diagnostics/gf_last_iteration.py DIR/RUN [--rows N]`. It gives
   the stage, the leg in flight, the valve clock with its trend, empty bands,
   F-stat epochs, GPU peaks, lnL / leaves, the last rows and the cycle table
   in one screen. Start every readout from it.
3. Grep the stdout by family (section 3). The greps that worked on the 9mo:
   `[GF_TIMING]`, `[SAVE]`, `[LEG `, `[MAXLOGL]`, `F-stat grid fit epoch`,
   `[stageA]`, `[stageB] banded Mc`, `[peaks] N peaks` (strip the per-band
   dict with sed), `GB_ACCEPT rj-split`, `Number of active leaves`,
   `[GB_GATE`, `[GB_BAND_SHUTOFF status]`, `[V9-STAGE`, `[eigen_refresh]`,
   `different from the current device`, `[STORE-WRITE]`, `Traceback`,
   `OutOfMemory`, `GPUassert`, `DISABLED`, `RESUM`, and on a run with
   `GB_WARM_SELF=1`, `[GB_WARM_SELF]` (section 3.4).
4. Read the store with h5py (one python at a time): cold lnL max / mean /
   min per row and per walker, leaves per walker per row, `saved_after`,
   the valve arrays at the last rows, `sub_backend/<src>/h_h` per leaf and
   walker, the psd / galfor cold chains. Small targeted scripts in the
   scratchpad, never one monolith.
5. If asked for a page, regenerate it (`from_tar` or `python -m
   lisatools.globalfit.monitor DIR/RUN OUT.html`, ~15-20 min for the full
   page on the laptop, ~20 s for the short page) and publish it (one artifact
   per run kind, republished in place for later tars); report `missing=K`
   and every MISSING reason.
6. Write the readout (section 8). Save what is durable (rulings, defects,
   the next-tar watch list) to your session's memory and, for anything
   another person must see, to a dated handoff note under `docs/`.

### 2.4 Delegating the numbers

The repo skill `processing-gf-snapshots` (`.claude/skills/processing-gf-snapshots/`,
when present in the checkout you are in) dispatches ONE Opus agent per
snapshot with a fixed template: it extracts, measures (trajectory, high-f
assessment, health, truth rebuild, page regeneration) and reports RAW
NUMBERS. It never publishes and never interprets; the parent writes the
takeaways. Its trap table is the distilled history of wrong analyses; read
it before dispatching. The template itself is partly stale and tied to the
originating user's laptop: it assumes zips, `gf_monitor_gen.py` instead of
`python -m lisatools.globalfit.monitor`, the pre-2026-09-26
`build_truth.py --iteration` trap (the default is safe now), a 3-month v4
reference store, fixed artifact ids and a scratchpad path. Keep the trap
rows that still apply (stale zip, two h5 files, the extract semantics,
`--kappa-out` into the run dir, report `missing=K`); rebuild the commands,
the interlock and the publish step for your own setup before dispatching.

---

## 3. The health checklist, in order

Work through these in order; each level's "sick" signs change how you read
the next.

### 3.1 Process level (first 5 minutes)

| look for | healthy | sick / what it meant |
|---|---|---|
| `Traceback`, `MPI-ABORT`, `OutOfMemory`, `GPUassert`, `illegal memory access` | none | a GPUassert at a sync line reports ANY earlier async fault; the line number is the reporter, not the cause. An IMA that vanishes under `CUDA_LAUNCH_BLOCKING=1` is a cross-device race. |
| cupy `device where the array resides (0) is different from the current device (1)` | zero lines | a kernel read a peer device's buffer: a race waiting to happen (fixed site by site: 2ab7c0ac, c7a47a01, 4c7a3f9d, 9e3ab31c). Count them per job; name the python frame in the warning. A lookup-table BUILD line after startup is the same disease. |
| job launches and what each resumed at | one job per intended relaunch; `RESUMING` / `[LEG ..] resume:` lines consistent with the previous job's last `[SAVE]` | several launches with zero stored rows = preemption faster than the first save (the midit checkpoint was built for this). `the stage's stored move order [...] differs from the live composition` (grep `stored move order`) = the composition changed; the cycle restarts at its head (expected after a launcher change; it repeats on EVERY resume of that stage because the stamp is write-once). |
| `[SAVE]` cadence | one per leg in search (3-5 per cycle), one per iteration in PE; save 25-30 s | a 400-s save stall every Nth row = the monitor hook or the backup copy holding the saver (now on its own rank). |
| `gpu_util_<job>.csv` | peak memory well under 95.8 GB (9mo: 44-51 GB); util 70-96 % in GB search | > 90 GB = the F-stat grid or a lookup comp duplicated per rank; see 3.9. The csv samples the HEAD node's two GPUs only. |
| `[GB_CHOL_CACHE] ... DISABLED`, `[GB_STALE_POOL`, `share a cell label` | never | the cache fell back (a crash in a refresh); a slot-map guard tripped (raises). |
| host RSS (from the saver / monitor lines or the csv) | slow growth | 30 -> 38 GB over 45 h on the 9mo was tolerated. |

### 3.2 Stage and cycle level

- Which stage, step k of N, rows in stage, and the `[V9-STAGE <stage>]
  entering` block: the profile applied to each GB move (opt-SNR floor, peak
  floor, phase maximization, prior births), `per-walker RJ valve RELEASED at
  step entry (stored step ...): N pairs reopened`, `reset_valves` at a fresh
  `gb_search_2` entry, the F-stat peak floor line.
- Cycle wall and the per-move walls from `[GF_TIMING]` and `[LEG]`. The
  dashboard prints the last wall per move. 9mo reference, `gb_search_2`,
  cycle ~88 min: `rj_warm_search` 580-610 s, `in_model` 545-714,
  `rj_fstat_search` 1456-2460 (of which the fit 193-272 s once the comb
  skip is in; 2115 s before it; the RJ part 900-1250 and growing with the
  population), `in_model_fstat` 550-640, `rj_prior_removal` 1140-1180 (the
  largest fixed cost: prior births at SNR 5), `in_model_removal` 540-605,
  sobbh 10-16 / mbh 25-52 / emri 57-71, noise block 7.5 s, save 27 s.
- Timings that are NOT a regression: the first call of a batched MBH or
  EMRI move pays JAX / CUDA JIT (minutes per rank per process); a job's
  first source block is 10x its steady state. The first 10 rounds of a
  plateau search are slower than the rest.
- The F-stat epochs: `DONE.json` per epoch (`wall_seconds`, `n_peaks`,
  `clock`), the `[stageA]` line (nodes x sky evaluations; at 9mo 996k f0
  nodes x adaptive sky up to 512 = 22 min before the shut-band skip, 2.3-7.7
  min after) and stage B (per-peak grids, ~1 min at a few hundred peaks, 13
  min at 1772). Peak counts should fall over a stage (9mo search 2: 463 ->
  242); an epoch with 0 peaks skips the F-stat RJ move and its partner.
- Legs: a 9mo search cycle is 3 rows (saved after `in_model`,
  `in_model_fstat`, `in_model_removal`); the source block and the cycle-end
  noise block run in the tail of the last leg, so the third row holds the
  converged noise. `saved_after` tells you which leg each row is; compare
  lnL leg-to-same-leg, not row-to-row. (An extra row right after a noise
  change existed only under the dropped galfor ratchet.)

### 3.3 The likelihood trajectory

Cold lnL per walker per row (`global_fit/log_like[:, 0, 0, :]`), max / mean
/ min and the spread. What we look for:

- Monotone-ish climb with decelerating per-leg gains inside a search stage
  (9mo search 1: +3,185 then +422 / +277 / +146 per leg; search 2 settled at
  +50-100 per cycle). Flat over 3 cycles with a flat leaf count is the
  picture of a stage near its end.
- Walker spread 450-1,000 nats at 9mo with one walker trailing by 750-1,000
  = a stuck source on that walker (3.7), not a GB problem; 500-800 behind
  with 10-30 fewer GB leaves = local 1-4 mHz solutions (normal).
- Expected DROPS, not defects: the `gb_search_2` entry switches phase
  maximization off, so row 56 read 1.1-1.9k below row 55; a galfor nudge
  costs 8-15k on every walker (the ratchet; dropped); a prior-RJ leg with
  births can lose 300-900 per walker (correct detailed-balance RJ accepting
  marginal deaths; the typical-set spread); a noise re-fit moves it by
  hundreds either way; rows with all-rungs vertical swaps can drop 150-700
  and stay real.
- The stored cold lnL is trustworthy: every GB propose rebuilds
  `state.log_like` from the residual (the `incremental ll drift ...
  rebuilding log_like` line). The drift value itself is the ledger error
  (p50 ~1600 per propose in `rj_warm_search` in September, under 1 nat after
  the slot-map fixes); a jump there is a bookkeeping regression.
- A resume's "initial log likelihood (after recipe setup)" must match the
  stored last row within ~1 nat per walker; tens to hundreds = a time /
  frame / residual bug in a changed waveform path (the EMRI full->direct
  switch was ~0.2 nat per walker). Nothing checks this automatically:
  `setup_acs(rebuild_residuals=True)` overwrites `state.log_like` on resume
  with no stored-vs-recomputed guard, so diff the lines against the stored
  row by hand on every relaunch that changed a waveform path.

### 3.4 The GB population

- Leaves per walker per row (`inds/gb` summed over leaves) and the trend:
  9mo search 1 +20/row early, +5/row late; search 2 +70/cycle after the SNR
  5 floor, then +5-10. Counts move without RJ (vertical swaps) by +-20.
- Births and deaths from `[GB_ACCEPT rj-split <move>]` PER RANK (one line
  per compute rank; the head log alone is one walker): proposed, viable,
  prior-gated, SNR-dropped, accepted over rungs, accepted cold. Cold RJ
  acceptance is 0.1-0.6 % of proposals in a mature search; dimension changes
  reach the cold chain mostly through swaps. `[GB_BIRTH_STALE]` 60-90 % = the
  warm / F-stat peaks are already absorbed (normal late in a stage).
- The GPU-index-aligned pair gap (walkers 0+1 hold +40 leaves over 2+3 at
  9mo, diffuse +-1 per band, pairs aligned with the GPU index not the node):
  unexplained, open item; report whether it persists.
- Truth matching when the truth npz is present: phase-maximised overlap
  >= 0.8 (`GF_MONITOR_MATCH_MM`), completeness / purity / median overlap on
  the page (6mo full_pe: 62.7 % / 65.2 % / 0.976). The denominator is FROZEN
  at the row the truth set was built (and under that row's noise); a noise
  change moves the detectable count (a nudge moved it 2495 -> 3168 at SNR 7).
  Misses sit at SNR 7-10, 1.5-6 mHz.
- The warm start of a child run: the first warm propose births tens of
  thousands at once (9mo: +65 leaves per walker in one propose, 2,012 ->
  2,172 by the seed end); the SNR-8 floor of `gb_search_1` drops the parent's
  SNR 5-8 tail ("scored-dropped: snr 74k"). Since 73bb198b the candidate
  sequence differs per rank; identical `viable / prior-gated / SNR-dropped`
  tallies on every rank meant one shared seed (defect, fixed).
- The warm-start SELF refit (`GB_WARM_SELF=1`, 2026-10-13; navigation 2.8),
  the cross-walker GB exchange: grep `[GB_WARM_SELF]`. Healthy: on the saver
  (`[r<k>/saver]`) one `epoch k built from rows a..b (200 samples, K
  components) in T s` every 5 rows of `gb_search_2` and every 250 of
  `full_pe`, `built_at_row` advancing by exactly the cadence (a resume
  continues the clock, it does not rebuild at once); on EVERY compute rank
  one `rank r adopted epoch k` per epoch, the same k on all ranks within a
  propose or two; `this propose used SELF` on 3 proposes in 4 of each warm
  move once an epoch exists (ORIG on every propose before). The point of the
  feature is the `cold` birth acceptance of the `self` tally against the
  `orig` one on the same line: the self mixture should land cold births the
  original no longer does (walker-held SNR 6-12 leaves). Sick: `build
  FAILED` (read its traceback; the epoch dir stays incomplete and is
  ignored; it retries a cadence later), `HOLDING THE SAVER RANK UP` (the
  build wall exceeds the time to the next save: every such build shows as a
  slow leg on the head; quote the read / fit / referee / apply walls),
  `STOPPING THE SELF REFIT` (stored rows were dropped; no more epochs this
  run), an adoption on some ranks only (`epoch k is not complete` / `failed
  to load`: a shared-filesystem lag or a corrupt npz; the rank keeps its
  previous epoch). `warmstart/self/epoch_NNNN/DONE.json` carries the rows,
  counts and walls of every build.

### 3.5 Gating: how to read the valves

This is the question asked most. The per-walker RJ valve is the stage-end
clock; read it from the `[GB_GATE <move>]` census, once per cycle (the judge
runs inside `rj_fstat_search`):

```
[GB_GATE rj_fstat_search] occupied 1999 | shut 1875 (93.8%) | active 124 -- streak p50 0 p90 2 of 3,
   max-minus-now p50 ... | fresh N | chronic N | young N ... (tol 20, gain)
```

- **active** (occupied and not shut) is the number to track; the dashboard
  prints its last 8 judges. Healthy = falling cycle over cycle (9mo search 2:
  1957 -> 377 at the 4th judge after the reset, the predicted "shut wave",
  then 313 -> 286 -> 232 -> 195 -> 183 -> 136 -> 124). "% shut" hides growth
  of the occupied set; births open new pairs (+1-2 per cycle late, +1.3 per
  iteration early in the 3mo).
- **streak p50 / p90 of 3**: a pair shuts after 3 judged cycles without a
  gain > tol over its running best. Nothing can shut before the 3rd judge
  after a reset; after a stage entry or a `reset_valves` the first judge
  re-learns the max (one propose of delay, not a defect). At a stage entry
  expect `shut 0, streak 0, max all -inf` on the first row and finite maxima
  on the next judged row.
- **Streak resets** separate into pairs whose occupancy changed (real births
  / deaths) and pairs whose occupancy did not (a noise-driven gain change).
  Under moving noise (gb_search_2) the dense 0.5-5 mHz bands (gain 14-38k
  nats) move by ~0.4 % per cycle from the cycle-end psd / galfor sampling,
  which is 50-150 nats against tol 20, so their streaks reset chronically
  (113-126 pairs at 9mo). **Ruling 2026-10-11: the gate is working as
  planned; it digs the foreground out slowly; no gate change.** Do not
  re-propose re-baselining or a larger tolerance unless asked. The stage
  ends when the foreground has been dug out.
- **Shut pairs never reopen within a stage.** 0 shut->open transitions in
  every stored row pair is the expected reading.
- The empty-band valve (`[GB_BAND_SHUTOFF status] 647 of 688 eligible bands
  off (f > 10 mHz) . next revive check in N`): bands above 10 mHz with zero
  cold occupancy. Its since-revive counter rolls 10 -> 0 WITHOUT reviving
  unless the noise fingerprint moved (by design); when it does revive
  (`the noise model has moved by 168%` = alpha wandering, floor 0.1 %; grep
  `has moved by`) the next comb
  sweep costs a full stage A (17 min instead of 2.3) for one cycle. Accepted
  (~2 % of wall). Empty bands BELOW 10 mHz never shut (91 of them at 9mo;
  TODO item, not ruled).
- **Persistence**: on a resume the entry line says either `restored` (the
  `_w` datasets were read back) or `fresh` / `RELEASED at step entry (stored
  step != k): 0 pairs reopened`. Before 8e1f5f5a the per-walker valve was
  never persisted (the store was created with the flags off, `save_step`
  skipped missing datasets), so EVERY resume restarted the stage-end clock:
  232 active -> 1942 on the next job. Since 8e1f5f5a the 8 datasets are
  created at the first save; the first resume on the new code still starts
  fresh, later ones persist. Check the entry line on every relaunch.
- **The da45aa98 unit switch** (residual term -> band gain) leaves a trace
  on any store that crossed it (the 9mo at row 57): the stored
  `band_cold_logl_max_w` were in the old units, so every streak reset ONCE at
  the first judge after the change; a store whose `band_cold_logl_w` holds
  residual-unit rows shows a +1,400-nat step at the switch row (the
  2026-10-13 9mo extract holds `-inf`, no statistic, in rows 0-56 instead; the
  monitor's `gb_band_evo` raster blanks both kinds of row when it falls back
  to this array); and the in-model observation site is live since then, so a
  mid-cycle high can reset a streak.
- **The window-rule switch (2026-10-13, `band_window_rule` 1 -> 2)**: on a
  WDM grid the gain is formed on whole layers, and until then a sub-band
  owned only the layers whose CENTRE it contains, i.e. none for 924 of the
  1232 9mo sub-bands (band index 1-6 mod 8): their `band_cold_logl_w` is
  exactly 0 on every row and walker, and their occupied pairs shut after 3
  judges whatever their sources did (9mo row 66, the first gb_search_2 row
  with a shut pair: 40.6 % of ALL blind (walker, band) pairs shut, about
  every occupied one, against 7.7 % of the window-owning pairs; 41.6 % vs
  39.0 % by row 138, the stage's last). Since then every sub-band owns its covering layer (the 8
  sub-bands of a layer tile share its gain; the 308 window owners kept
  theirs). A store that crossed the switch logs `[GB_GATE <move>] window
  geometry v2: ... re-basing` ONCE at its first judge: the running maxima
  take that step's gain, every pair keeps its streak, reset count and shut
  flag (user ruling 2026-10-13: a reset mid stage changes nothing about
  convergence), that one step is not judged, and `band_window_rule` reads
  1 on every row before that judge and 2 after. When reading a store's
  gain per band, expect runs of 8 identical values (one per tile), not a
  per-sub-band statistic; a sum over bands overcounts each layer.
- **Stage end**: `[V9-STAGE <stage>] STAGE COMPLETE on the SHUTOFF rule
  ("every occupied pair has shut off")` (9mo search 1 at row 55, cycle 8
  leg 2) or on the nleaves plateau. Estimate the handover from the active
  trend (10-50 per cycle late) and say so as an estimate.
- Levels 1 and 2 (`[GB_IMCONV]`, `[GB_IMGROUP]`) are read for cost, not for
  the stage end: at-floor fractions near 100 % mean the window, not
  convergence, bounds the repeats (a cost trade, do not read it as
  convergence); `[GB_IMGROUP ... pass 1: cold dlnL +0.000 over N occupied
  sub-band(s); shut 1232 / open 0` with a 1-s runtime is the signature of a
  valve read from the previous stage (fixed c3527ef6).
- Never conflate the two valves: both print "shutoff"; the level-3 one is
  per PAIR and ends the stage, the empty-band one is per BAND and only turns
  births off above 10 mHz.

### 3.6 The noise

- psd `Soms_d` / `Sa_a` per walker vs the injected pair. **The reference is
  the unequal-arm fit** (`MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=1` on the 9mo
  / 1yr; the 6mo keeps 0 for its resume identity and its page needs the
  `honor_resume_pin=False` path, e4fb360c). Healthy: `Soms_d` within 0.1 %,
  `Sa_a` +2-3 % (unresolved low-f galaxy) and random-walking 0.3-0.9 % per
  cycle in gb_search_2.
- galfor: judge the curve at 1 / 2 / 3 / 3.5 / 4 / 4.5 / 5 mHz (the monitor's
  foreground panels; `galfor_curves` in `noise_ratchet.py` is the mean of
  per-walker curves, the curve of the mean parameters misreads it). `amp`,
  `fk`, `f_1`, `f_2` stable to ~1 %; `alpha` wanders 2-15 (unconstrained;
  cap 20 is fine). Two lnL-degenerate parameter modes (alpha ~1.5 vs ~8)
  give the same curve to 0.1 %; walkers moving between them is mixing, not a
  problem.
- In `gb_search_1` psd / galfor are bit-identical row to row (noise fixed).
  In `gb_search_2` the cycle-end block converges in 6-22 rounds, 7-30 s
  (`[MAXLOGL] ... converged=4/4`).
- The 3-5 mHz foreground at 6-9 months carries unresolved GB power (8.5x
  the true confusion at 6mo); it drains as the GB search resolves sources.
  Do not score it against a foreground truth above ~2.5 mHz. Foreground
  self-shielding: a model noise k x too high makes a true-SNR source read
  SNR / sqrt(k); that is what kept SNR 7-10 completeness at 3-5 mHz low and
  motivated (and then outlived) the ratchet.
- The snapshot taken right before `gb_search` starts is the cleanest noise
  certificate (zero leaves).
- The 6mo lesson behind the child-run noise pin: the 6mo launcher pinned
  psd + galfor at a hand-set 3-month estimate and search 1 / 2 held it, so
  the noise had never been fitted to the 6mo data before the stage that
  released it, and the first release moved lnL by ~+10k. Hence
  `noise_pin.py` (the parent's max-lnL walker) for parented runs, and a
  `noise_search` stage first, from an inflated foreground estimate, for the
  no-parent 3mo. A large lnL jump at the first noise release means "the noise
  had never been fitted", not a defect.

### 3.7 The other sources (MBH / EMRI / SOBBH)

- Per leaf per walker at the last row: median cold SNR from
  `sub_backend/<src>/h_h` (SNR = sqrt(h_h); populated only where
  `{BRANCH}_RECORD_DH` is on, the default for the batched MBH / direct EMRI /
  lookup SOBBH moves; NaN on the slow path); the glance table names a walker
  whose SNR is > 20 % and 3 SNR off the median. 9mo reference: MBH 437 / 169
  / 1010 / 171 / 262 / 540; EMRI 18 / 19 / 12 / 44 / 23 / 13 / 31 / 16;
  SOBBH 8.6 / 9.4 / 6.0 / 4.5 / 5.6 / 5.7. SOBBH leaves at SNR ~0.1 on some
  walkers = the template is gone on that walker (seen in the 6mo v8).
- Movement: sources should move ~1 walker-std from the seed in the first
  cycles; the MBHs moved a lot toward a better likelihood in the 6mo
  `full_pe`, which is why the 9mo got a fixed-dimension `source_search`
  stage and a source warm start.
- **The stuck-walker signature (EMRI 3, walker 0, 9mo):** one walker's leaf
  sits at a boundary maximum of the wrong mode (cos qK = -0.99998 against a
  prior edge at -0.99999, SNR 22 vs 45 on the other walkers, 650-1,000 nats
  below) and does not move through dozens of source blocks although the
  move's acceptance is ~70 % overall. Its own 8-rung ladder sits 60-450 nats
  BELOW its cold rung, so no swap can lift it until a hot rung finds the
  right mode. Side effect: the Gram information matrix's stencil stepped
  outside the prior box -> `arccos` NaN -> non-finite Gram (fixed by nudging
  the expansion point inside the box, 4c7a3f9d). Remedy at the next stop:
  `scripts/fstat_proposal/copy_store_leaf.py --store STORE.h5 --branch emri
  --leaf 3 --src-walker 1 --dst-walker 0` (dry run, then `--apply`; the
  resume rebuilds residuals and recomputes lnL). Nothing in the run itself
  un-sticks it.
- `at walker w` in the `[eigen_refresh]` lines and in the Gram non-finite
  errors is the RANK-LOCAL walker index (one walker per rank); map it to the
  global walker through the line's rank prefix (`[r<k>/c<i>]`; the head's
  lines carry no prefix and are walker block 0).
- Gram rebuilds: `[eigen_refresh] <branch> leaf k Gram info matrix at walker
  w in X s` every `{BRANCH}_EIGEN_REFRESH` (10) visits; SOBBH ~0.3-0.7 s, MBH
  ~3 s, EMRI ~1 s. A rebuild that runs off-device shows up as the cupy
  warning plus a second lookup comp built (9e3ab31c was the last such site;
  GPU-unverified at the time of writing: watch the first SOBBH Gram of the
  next run on a GPU-1 rank).
- The slow-path cross-checks (`{SOBBH,MBH,EMRI}_CHECK_LL`) are OFF in
  production: at every 10th visit they cost 26 min per iteration in the 6mo.
- Source block cost per cycle at 9mo: sobbh 10 s, mbh 25-50 s, emri 56-70 s
  (first cycle 14 / 133 / 114 s of setup).

### 3.8 GB accuracy audits

- `ll AUDIT` cold medians and the `DELTA-vs-DELTA` lines (sig-het vs exact):
  6mo production 0.014 / p99 0.15 / max ~0.5 nats after the carrier reference;
  hundreds = a reference-build regression (the Re/Im layout failed at low f
  in September).
- `[GB_ORTHO_LL]` worst and `[GB_CELL_LL]` exceed counts: the baseline is
  98-100 % "exceed" on every move, so only the DELTA against a previous job
  and the COLD count carry information. A worst-offender jump at temp 0
  after a tempering change means mis-credited ledgers (the pre-registered
  response was to turn the new path off).
- `[GB_VERT]`: priced vs unpriceable pairs and acceptance (~50 %). `0/0 rows
  over 0 sweeps` on every line is a dead counter, not a measured zero.
- The in-model acceptance (cold) 0.6-0.7; a drop to ~0.5 with 3-15x larger
  accepted steps was the Gram proposal scales widening correctly.

### 3.9 Performance and memory

- The budget is the cycle table in 3.2; what grows with the population is
  the RJ part of `rj_fstat_search` and `rj_prior_removal`; what grows with
  Tobs is the F-stat fit (stage A nodes x sky) and the lookup / sig-het
  stashes.
- GPU memory: the stage-B F-stat grid (~12 GB raw, ~3x resident, on every
  rank that loads it) is the biggest transient; OOMs there were fixed by
  evicting old epochs and keeping the CDF on the host (09160c8e); host mmap
  / float32 are still open. `FSTAT_STAGEB_NSKY_MAX=512` and
  `FSTAT_STAGEB_GROUP_MAX_GB=1.0` are the 9mo memory levers (accuracy
  preserved by ruling). 9mo peaks 44-51 GB per head-node GPU.
- The monitor / backup copy must never sit on the saver's critical path
  (it did: 400-s stalls, then a crash that took both store copies). It runs
  on its own rank now.
- Two of four ranks sweeping at 19k evals/s vs 80k = a slow node, not code.

---

## 4. Signatures we have met (symptom -> cause -> status)

| symptom in the logs / store | cause | status |
|---|---|---|
| valve census `shut 0` forever, `band_cold_logl_max_w` all -inf while `_peak_w` is finite | the judge wrote into the state object eryn discards (`_arm_search_stage` ran before the state copy) | fixed fe26e09f (09-28) |
| `[GB_IMGROUP in_model] pass 1: cold dlnL +0.000 ... shut 1232 / open 0`, 1-s in_model as the first move of a new stage | the pure in-model move bound the previous stage's all-shut table; release lived only in the RJ moves' bind path | fixed c3527ef6: release at step entry |
| every resume: `RELEASED at step entry (stored step != 2): 0 pairs reopened`, active pairs jump 232 -> 1942 | the per-walker valve datasets were never created in the store; `save_step` skips missing datasets | fixed 8e1f5f5a (10-10); first resume on the new code still fresh |
| 3 in 4 sub-bands (index 1-6 mod 8 on the 9mo grid) hold `band_cold_logl_w` exactly 0 on every row; their occupied pairs shut after exactly 3 judges, a wave of shuts at the first judge after every stage entry | the valve window held only the layers whose centre lies in the band; a 1/8-layer sub-band without a centre had an EMPTY window (`_window_bins`) | fixed 2026-10-13 (covering-layer window, `band_window_rule` 2, one persisted re-learn per store) |
| births on ONE rank only in `[GB_ACCEPT rj-split rj_prior_removal]`; ranks 1-3 at opt-SNR 8 in a stage declared 5 | the stage profile / PE declarations were applied on the head only; compute ranks held their own move objects | fixed 9ebe7d9d (10-02; pushed in the merge ca61955d): live attrs shipped in the fan-out payload |
| `KeyError: 'Unable to synchronously open object (message not aligned)'` on `global_fit/recipe/<step>` at resume; the backup copy identical | head-side stage stamps raced the saver (two HDF5 writers) at a stage transition; the validator never opened the recipe group so the torn header was byte-copied into the backup | fixed: validator 0a8c7d7f, head stamps routed through the saver 5bf31e7b; repair tools `salvage_recipe_group.py` / `repair_recipe_group.py` |
| saver dies with `OSError: Unable to synchronously open file (file is already open for read-only)` after a page build | the in-process monitor generator left the store open after raising (`runpy` returns globals only on success) | fixed 95c1734a; monitor moved to its own rank 12dab4e2 |
| standalone page dies with `filter returned failure during read` on `chain/gb` | a non-SWMR reader racing the saver, not disk damage | fixed b5ddb8b5 (merged as 78b33438): per-row fallback to the running backup copy; 21916c49 for the last `inds/gb` row |
| `GPUassert illegal memory access at LISAResponse.cu:514` at setup, gone under `CUDA_LAUNCH_BLOCKING=1`, cupy "different from the current device" warnings | cross-device peer reads (WDM window / fold map cached on GPU 0; Gram PSD on the walker's GPU; cap / warm tables; JAX leaving GPU 0 current after the SOBBH PN step) | fixed 2ab7c0ac, c7a47a01, 4c7a3f9d, 9e3ab31c (10-08 .. 10-11); the last one GPU-unverified at writing |
| cupy OOM at 88 GB in `StackedFStatProposal4D` cumsum during a PE refit | every earlier epoch's grid stayed registered on the GPU; the global CDF resident | fixed c610fcf2 + 09160c8e; mmap / float32 open |
| `band_occ_streak` pinned at 1, empty-band valve never latching | `GB_FSTAT_REFIT_EVERY=1`: every epoch revived every band | `GB_FSTAT_REFIT_EVERY=2` pinned in the launchers since 2026-09-28 (cadence knob 611ab2d1); the epoch revive is opt-in |
| births above 10 mHz in shut bands right after a (re)launch; the empty-band valve silent on each RJ move's first propose | the empty-band table was bound only at propose END, so every move's first propose after a launch ran without it (permissive) | fixed 2f4b480e (the barren-band valve is adopted at propose start) |
| 3mo `gb_search_1` pending pairs floor at 28-36, never 0 | births open occupied pairs as fast as the valve shuts them; the criterion, not patience | the stage-end rule and the SNR floors were restructured in v9 (per-stage profiles) |
| the galfor ratchet: nudge twice in a row, stage ends after 3 rows, re-equilibration in PE | clock lag on the head's `backend.iteration`, the per-band max earned under fixed noise, a midit checkpoint adopted after a rewind | fixed f44d1073 and the band-max reset; the ratchet itself DROPPED 2026-10-03 ("not truly worth it": ~1 % of the missing sources per 3-h pair, stop rule at its noise floor) |
| a rewind did not take: row 47 holds a mid-iteration state of the previous job | `MIDIT_CHECKPOINT` wins over the store when its stored iteration >= the store's | `reset_recipe_stage.py --apply` sets the pkl aside; `MIDIT_CHECKPOINT=0` on legged search launch lines |
| identical warm-start tallies (viable / prior-gated / SNR-dropped) on every rank | both warm-start containers seeded with the bare run seed on all ranks; device-1 cupy streams entropy-seeded | fixed 73bb198b; `tests/test_cross_rank_seeds.py` |
| EMRI Gram "non-finite" on walker 0 only; `invalid value encountered in arccos` | a leaf 1.3e-5 inside the prior edge; the stencil stepped outside | fixed 4c7a3f9d (stencil nudged inside the box); the walker stays stuck (3.7) |
| monitor time-frequency panel: merger "spikes in the wrong panels" | raster aliasing: one-column merger lines, 347 px per panel for 1272 columns | open (pool DEC to the pixel width; TODO) |
| monitor whitened residual: walker 0 "acting differently" by 5-7x below 1.5 mHz | the page built the min-lnL walker's residual from GB + VGB only | fixed 1f1289cf: the snapshot carries every cold walker's residual |
| page says the fitted `Soms` sits +0.2 % from injection on the 9mo | the monitor rank inherited the equal-arm reference pin | fixed e4fb360c; launchers on unequal-arm a7c7ca7e |
| `prior move` legs lose -361 .. -919 per walker with births on every rank | correct RJ accepting ~75 marginal deaths per unit at cold; the deaths-only heuristic had only removed junk | not a defect (user: the typical-set spread); mitigation `GB_SEARCH_{N}_PRIOR_BIRTHS=0` if a stop rule reads the wrap |
| `[GB_VERT] all-rung measurement unavailable (AttributeError(...slot_specials))` on every block | the measured path read an attribute the real buffer never had; the fake had it | fixed 1cde08f1; lesson in `feedback`: fakes must match the real surface |
| `[GB_IMCONV] armed ... gate = coldest 12/24` while the run gated 4 | the armed line recomputed the gate inline | fixed 8ff55f76 |
| `[GB_INFOMAT] FALL-THROUGH` x 400, all "over 1 rows" | the heuristic misfires on 1-row blocks | cosmetic |
| `gf_arm_<tag>.npz` of another run overwritten | two runs sharing an arm tag; the generator globs every arm cache in the CWD | `_identity.py` rule per run family; run the generator in a clean dir |
| 6mo probe: 5 launches in a day, 0 stored rows | spot preemption every 30-75 min vs 85 min to the first save | the midit checkpoint (08-27); legs (one row per leg, 09-30) |

Still open at the time of writing (the user keeps the live TODO list; ask
for it): neighbour-aware sub-band
shutoff (user's item), the empty-band floor below 10 mHz, the
GPU-index-aligned leaf gap, RNG hygiene within a rank, the chol-cache
refresh touching shut bands, the F-stat fit still running on zero-peak
epochs, host-mmap F-stat grids, the TF panel aliasing.

---

## 5. Reference numbers (9mo v9, jobs 751-754, 2026-10-10/11)

Point-in-time; use them to recognise an order of magnitude, not as a spec.

| quantity | value |
|---|---|
| grid | Nf 1440 x Nt 6480 x dt 2.5 s, `TOBS_TARGET` 23,328,000 s, 1232 GB bands, `EDGE_CROP_WAVELETS` 60 |
| walkers / ranks | 4 walkers, 4 compute ranks (2 nodes x 2 GPUs), saver, monitor rank |
| warm-start build (in-job) | 53 min (merge test 38 min of it) |
| seed stage | 4 rows (`GB_SEARCH_SEED_ITERS=3` gives n+1), 22-31 min each; leaves 1,947 -> 2,172 per walker |
| `source_search` | 52 rounds, 92 min, converged 4/4, +1.9k lnL |
| `gb_search_1` | cycle ~85 min; 8 cycles; complete at row 55 on the shutoff rule; lnL +6.9k; leaves 2,186 -> 2,372 |
| `gb_search_2` | cycle 88 min (143 min with the full comb); active pairs 1957 -> 124 over 19 h; lnL +10.6k; leaves 2,366 -> 2,684; psd `Sa_a` +2.5 % |
| F-stat fit | 23 min per epoch before the shut-band skip (stage A 22 min at 996k nodes x 512 sky over 4 ranks at 75-79k evals/s), 3-4.5 min after; peaks 201 / 41 / 23 / 9 / ... in search 1, 463 -> 242 in search 2 |
| GPU peak | 35-51 GB per head-node GPU (stage B at 850 boxes per group = 50 GB) |
| cold lnL | ~1.580e8; walker spread 450-1,000 |
| 6mo `full_pe` for comparison | 5.5 min per iteration; sig-het audit cold median 0.014; completeness 62.7 % at overlap >= 0.8 |

---

## 6. Changing the run from a deep dive: relaunch mechanics

- A relaunch is the bare launcher (`./scripts/fstat_proposal/submit_gf_9mo_v9.sh`
  or the flagship twin) after `git pull` on the cluster; the store resumes
  automatically; the warm-start npz is reused. Pure-Python changes need no
  rebuild; a `.cu` / binding change in LAT, GBGPU, BBHx or FEW needs
  `pip install -e . --no-build-isolation` in that repo (the launcher's
  preflights run the GPU parity tests when a fast path is selected).
- One-off knobs go on the launch line (`K=v ./submit...`) only for `${K:-v}`
  exports; a hard export swallows the line. After a launcher change the user
  wants the full list of general settings and everything that differs from
  the previous run (ruling 2026-10-07).
- Relaunch after a `[SAVE]`, not mid-leg, when the choice is free; a cut
  inside a leg loses that leg.
- Store surgery (copy a leaf, re-rung, rewind, salvage) only on a STOPPED
  store, dry run first, on a copy when the user says so. Only
  `copy_store_leaf.py` and `migrate_gb_shutoff_persist.py` refuse a store
  written in the last 300 s (`--live-window`); for the others confirm the job
  is dead yourself.
- A `git pull` on the cluster changes a RUNNING job's imports (editable
  install): pull only with the job stopped.
- A composition change inside a search stage makes the leg resume restart
  the cycle at its head ("stored move order differs") on every later
  resume of that stage; `gb_search_2` stamps fresh.
- `NULL_CHECK_ONLY=1` twice (old vs new waveform path) on a store copy is
  the way to prove a path switch is lnL-neutral before it goes live
  (`|delta| <~ 1 nat per walker`).

---

## 7. Rulings that constrain what you conclude or propose

- **Unequal arm only.** Every accuracy gate and every reported noise number
  uses the production noise (`UNEQUAL_ARM=1`, `layer_calibrated`). Equal-arm
  numbers describe nothing the cluster runs.
- **`q` is the mass ratio.** Name a decimation factor "decimation".
- **The gate is working as planned (2026-10-11).** Noise-driven streak
  resets in the dense bands are accepted; no tolerance or re-baseline change.
  Tol 20 nats (the user's judgement, not 0.5 SNR^2). Level-3 shut pairs run
  nothing; level-1 convergence follows the source, never reset on a rung
  change; `in_model_fstat` stays in every search stage.
- **Search errs toward grabbing** (recall over precision): junk dies in the
  RJ, a missed source is invisible downstream. The landed F-stat peak floor
  was the knee (6.25), not the extreme.
- **No maximization in PE**; search needs no detailed balance (maximize,
  then pretend the draw happened, with the MH machinery intact). Phase
  maximization only in `gb_search_1`.
- **One psd + galfor proposal per cycle, at the END of the cycle, run to
  convergence** (gb_search_2). Sources every cycle in `gb_search_1` since
  ddad1304. `gb_search_seed` is GB-only. No replica_pe, no ratchet in the
  9mo / 1yr recipe.
- **MBHB admission**: merger within two weeks of the data end, modelled by
  the in-window inspiral cut (`MBH_MERGER_TIME_BUFFER`); a case that cannot
  be cut properly is left out.
- **Warm start**: the last 300 cold rows x 4 walkers of the parent; sources
  with parent SNR > 10 start from the parent's cold chain (EMRI 3 included,
  plunge or not); below that, truth as at the parent's Tobs. **Self refit**
  (2026-10-13, `GB_WARM_SELF`): from `gb_search_2` on, refit the warm start
  from the run's own last 50 rows x 4 walkers, every 5 rows in search and
  every 250 iterations in `full_pe` (a periodic approximation in PE, like the
  residual F-stat refit every 150), and KEEP the original: the warm moves
  draw 25 % original / 75 % self refit.
- **A 9mo instead of a 1yr** (2026-10-07): keep the 1yr settings; change
  only what Tobs forces (`TOBS_TARGET`, `SIGHET_NT_LAYER`, the MBHB set);
  capacity knobs stay at the 1yr values. Memory levers on the F-stat are
  allowed only without accuracy loss.
- **`MIDIT_CHECKPOINT=0` on legged search launch lines; never change
  `NWALKERS` on a resume; `GF_PERSIST_STAGE_START=1` resets at each stage
  boundary.**
- **"Anything serious in terms of code updates" goes to an Opus agent**
  (2026-10-07); the deep dive itself stays with the session.
- **Everything finished and tested ends up merged into dev**, on the user's
  word; commit only your own hunks when another session edits the same
  checkout.
- **Rejected, do not re-propose**: dropping `in_model_fstat` from search
  stages; per-band replica gate ("leave it as is"); `RJ_FSTAT_DIST_BIRTH`
  (unused, will not be); re-baselining the valve at a noise change.

Design rulings (what a fix, a probe or a deep-dive script may not do):

- **No hand-rolled likelihoods or inner products.** Every lnL, SNR, `<r|r>`
  or overlap goes through `AnalysisContainer` (`.likelihood`,
  `.template_inner_product`, ...) or an installed engine
  (`diagnostic.inner_product`, the GB sig-het / chunked-het engines). A debug
  value computed as `-0.5 sum|r|^2` on raw WDM coefficients was the recorded
  failure (no PSD weighting; the number was not a likelihood).
- **Never rewrite likelihood infrastructure; stick to implemented code.**
  Find where a quantity is PRODUCED and read it; do not write a parallel
  harness that re-derives bin folds, windows, references or waveforms (it
  silently picks the wrong variant of a multi-path primitive). Prefer an
  existing script or installed class end-to-end.
- **GB serial-within-band scheduling is a requirement**: one RJ source per
  sub-band at a time, the next sampled against the UPDATED residual; two
  sources in one sub-band never move in-model simultaneously. Batching that
  breaks it was vetoed.
- **GB RJ default = the F-stat grid move + the prior move** in search and
  PE; the legacy GBGPU frequency-domain `get_ll` / `get_fstat_ll` path must
  never appear on a WDM basis.
- **Start scatter is multiplicative** (`truth * (1 + factor * randn)`), never
  additive.
- **Tukey tapers are equivalent** across chunked-het, sig-het and the TD->FD
  transform; any tapered region is removed by the edge crop (the crop serves
  the taper), and the crop is a constant layer count, not a Tobs fraction.
- **Anything compared against mojito data takes dt, band, window and TDI
  generation from the mojito file**, never a convenience value (naive
  decimation corrupted the T channel by more than the effect under study).
- **Approximate-likelihood accuracy is judged by the tiered spec** (error
  allowed to grow with displacement from the reference; DELTA-vs-DELTA),
  not by worst-case raw error.
- **Commands handed to the user carry no `<angle-bracket>` placeholders**
  (bash reads them as redirects; a `>file` once truncated a script to 0
  bytes); use UPPERCASE tokens or real paths.
- **Minimal prompts**: batch decisions, lead with a recommended default, do
  not stop between plans for approval; ask only what is genuinely blocking.

---

## 8. Writing the readout

What the user expects from a tar readout, in this order:

1. **The tar** (name verbatim including the ` (N)` suffix, md5, mirror time
   in UTC, which jobs and rows it covers, the live h5 named).
2. **Takeaways first**, two to six lines: stage and clock, healthy or not,
   the one or two things to act on.
3. **Health**: section 3.1 as a short list of counts (tracebacks 0, device
   warnings 0, restarts, GPU peak).
4. **Timeline** of jobs in UTC with what each resumed at.
5. **Timings** per move as a table (compare to the previous tar's).
6. **Trajectory**: lnL and leaves per walker, per leg; the noise.
7. **Gating**: the `[GB_GATE]` line quoted, the active trend, the estimate
   for the stage end (as an estimate).
8. **Sources**: the per-walker SNR table; any stuck walker named with the
   mechanism and the remedy.
9. **Defects**, each with the evidence chain, the falsification attempt, the
   fix status (uncommitted / pushed hash / needs relaunch / needs rebuild),
   and whether the running job needs it.
10. **Watch list for the next tar** and the relaunch line if one is due.

Numbers go in tables; times are UTC; per-walker values are listed in walker
order; name commits by hash. Separate "verified in production" from
"GPU-unverified" from "not applied". Multi-tar batches get ONE combined
summary, not a write-up per tar. Save the durable parts (rulings, defects
with status, the next watch list, relaunch lines) to your session's memory
before the session ends, and update this document when a routine, a
signature or a ruling changes.
