# Navigating the global fit (a guide for Claude)

Who this is for: a Claude Code session that has to work on the LISA global
fit without the context of the sessions that built it. It says where each
thing lives, what runs when, and how to find an answer without reading the
whole tree. The companion guide, [`global-fit-deep-dives.md`](global-fit-deep-dives.md),
says how a production run is examined and what we look for.

Rules that apply everywhere: [`conventions.md`](conventions.md) and the repo
`CLAUDE.md`. The library-level layout (domains, response, sensitivity,
backends) is [`codebase-map.md`](codebase-map.md); this guide is only about
the global-fit pipeline on top of it.

_Last verified against `dev` 8917eac2 on 2026-10-11. Line numbers drift;
every pointer below is a name you can grep for._

---

## 0. The three-minute model

A production run is this chain, left to right:

```
launcher (bash; ~500 env knobs)          scripts/fstat_proposal/submit_gf_<run>.sh
                                          scripts/flagship/submit_gf_<run>.sh
  -> driver: composes the recipe          scripts/fstat_proposal/run_combined_staged.py
       from the stock fit                 src/lisatools/globalfit/stock/erebor/
  -> GlobalFit.run_global_fit()           src/lisatools/globalfit/run.py
       one MPI rank per GPU + saver       src/lisatools/globalfit/communication/
       (+ a monitor rank)                 src/lisatools/globalfit/monitor/rank.py
  -> eryn steps a Recipe of Stages        src/lisatools/globalfit/recipe.py
       of Moves, one iteration = one      src/lisatools/globalfit/moves/
       pass through the stage's moves
  -> the store (HDF5) + sidecars + logs   <run_dir>/<base>_testing.h5, *.log, *.npz
  -> the monitor page + the dashboard     src/lisatools/globalfit/monitor/
                                          scripts/diagnostics/gf_last_iteration.py
```

Everything a run does is decided by (a) which **stock variant** it builds
(`full_year_combined` for every production run), (b) the **env knobs** the
launcher exports, and (c) the **recipe** the driver composes from them. A
knob you cannot find in the launcher is at its code default; a code default
you cannot find is a dataclass field in `stock/erebor/*.py`.

Vocabulary (the user's words; use them):

| word | meaning |
|---|---|
| **branch** | one source class in the model: `gb`, `vgb` (verification GBs), `mbh`, `emri`, `sobbh`, `psd`, `galfor` (galactic foreground), optionally `sgwb`. |
| **leaf** | one source instance of a branch on one walker (an RJ dimension). "Leaves per walker" = the GB population. |
| **stage** | a phase of the run (`gb_search_seed`, `source_search`, `gb_search_1`, `gb_search_2`, `gb_search_3`, `replica_pe`, `full_pe`, ...). A stage ends by its own rule. |
| **move** = **proposal** | one eryn move inside a stage (`rj_fstat_search`, `in_model`, `psd_pe`, ...). |
| **cycle** | one pass through a stage's move list. In search stages a cycle is split into **legs**. |
| **leg** | the moves up to and including the next `in_model*` move; one stored row per leg (`GB_SEARCH_LEGS=1`). |
| **row** = **stored iteration** | one saved state in the store; `global_fit.attrs["iteration"]` is the count. |
| **band** / **sub-band** | the GB frequency bands (1232 at 6-9 months); the unit of GB work, tempering and gating. |
| **pair** | a (walker, band) pair; the unit the per-walker RJ valve judges. |
| **rung** | one temperature of a branch's internal ladder (GB 24, PSD 12, MBH/EMRI/SOBBH 8 on the v9 launchers; the code default for the three sources is 4). The main store holds the cold chain only. |
| **valve** / **gate** | a mechanism that stops work in a band or pair once it has converged (section 4). |
| **epoch** | one F-stat grid fit (`gb_fstat_fit/shared_search/epoch_NNNN/`). |
| **q** | always the binary mass ratio. Never reuse it for anything else (user ruling). |

---

## 1. Files by size: what never to open wholesale

| file | lines | how to use it |
|---|---|---|
| `src/lisatools/globalfit/moves/gbspecialstretch.py` | ~33,000 | grep for a method or a log tag, then read that method. Never read it top to bottom. |
| `src/lisatools/globalfit/recipe.py` | ~6,900 | class list in section 2; grep by name. |
| `scripts/fstat_proposal/submit_gf_9mo_v9.sh` (and every launcher) | ~5,900 | grep for the knob; read its comment block (every knob has one with the ruling and the date). |
| `scripts/flagship/launcher.sh.in` | ~5,500 | the same body as a template. |
| `src/lisatools/globalfit/moves/psdmove.py` | ~3,900 | grep. |
| `src/lisatools/globalfit/run.py` | ~3,600 | grep `def run_global_fit`, `prepare_main`, `setup_acs`, `_seed_rank_streams`. |
| `scripts/fstat_proposal/run_combined_staged.py` | ~2,900 | read `V9_SEARCH_STAGE_PROFILES`, `search_stage_profiles`, `build_fit`, `describe_recipe`. |
| `src/lisatools/globalfit/moves/addremovemove.py` | ~2,900 | grep. |
| `src/lisatools/globalfit/monitor/_generator.py` | large | a SCRIPT: it runs on import and exits on import. Helpers are pulled out by name (`_short.py`, `_logs.py`). Never `import` it to inspect it. |
| a `slurm_stdout_<job>.log` | 10^5 to 10^6 lines | grep by tag (section 7). Stream a tar with `tar -xOzf ... | grep` instead of extracting the whole thing. |

The cheap, reliable entry points are the **module docstrings** (every module
added since September opens with a "why this exists" paragraph: `legs.py`,
`noise_ratchet.py`, `residual_snapshot.py`, `midit_checkpoint.py`,
`wdm_lookup_store.py`, every tool under `scripts/fstat_proposal/`, every
monitor module) and the **design specs** under `docs/superpowers/specs/`
(section 9).

---

## 2. The layer map

### 2.1 Launchers (the record of production values)

| what | where |
|---|---|
| the launched 6mo / 1yr / 9mo v9 launchers | `scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh`, `submit_gf_1yr_v9.sh`, `submit_gf_9mo_v9.sh` (the 1yr and 9mo are derived from the 6mo by `derive_1yr_v9_launcher.py` / `derive_9mo_v9_launcher.py`; the twin tests pin the deltas). |
| the flagship series (one template, six observation times, 2026-10-11) | `scripts/flagship/` : `launcher.sh.in` + `runs.py` -> `build.py` -> `submit_gf_{3mo,6mo,9mo,12mo,18mo,24mo}.sh`. Read `scripts/flagship/README.md` first: the parent chain, the three adjustment levels, the guards. The flagship 9mo equals the launched 9mo v9 in every executable line. |
| what a launcher pins | tests `tests/test_submit_scripts_layout.py` (allowlists, twin tests, `V930ProductionDefaultsTest`) and `tests/test_flagship_launchers.py` (DRIFT / IDENTITY / STRUCTURE / VALUES / COMPOSITION). A failing twin test is usually knob drift between runs, not a code bug. Known baseline failure: `ThreeMonthV9TwinTest`. |
| how it launches | run the script DIRECTLY (`NGPUS=4 ./scripts/...sh`), never a bare `sbatch`: the self-dispatch block picks partition, nodes and `--ntasks` and resubmits itself. See `docs/global-fit-launch.md` ("Campaign submit scripts") for the dispatch mechanics only: its examples and defaults are v8-era (`NWALKERS` is 4 in production, and the monitor rank, `GF_MONITOR_RANK`, is not described there). A requeued job re-runs the script spooled at submission; to change a launcher, cancel and resubmit. |
| GPU count and splitting a walker (`NGPUS`, `GF_GPU_ROUTING`, `GPUS_PER_RANK`; 2026-10-15) | `NGPUS` = 2 / 4 / 8 picks the allocation (1 node, 2 nodes, 4 nodes of 2 GPUs). By default one compute rank per GPU and the walkers split into equal blocks, so `NWALKERS` must be a multiple of the compute-rank count (a non-multiple is ROUNDED UP in the job, and the store then locks to it). `GPUS_PER_RANK=k` PACKS walkers onto a rank (its k GPUs shard the rank's walker rows); it never splits a walker, and with fewer walkers per rank than GPUs the launcher and `build_layout` refuse (9mo job 764). To split EACH walker over several GPUs: `NGPUS=8 GF_GPU_ROUTING=1 ./submit_gf_9mo_v9.sh` on the 4-walker store = 8 compute ranks, one GPU each, 4 walker blocks x R = 2 replicas (`n_blocks = gcd(NWALKERS, N_COMPUTE)`; `NWALKERS` is never rounded with the knob on; the `[SUBMIT] GF_GPU_ROUTING=1: replica factorization ...` line and the run's `walker-block layout: ... gpu_routing=on REPLICAS` table say what was built). Inside a block GB/VGB split by band range, MBH/EMRI/SOBBH by likelihood rows, PSD/galfor propose on the block lead and publish to the replica. Wired in the 6mo / 9mo / 1yr v9 launchers, the 3mo v9 twin and the flagship template; the cluster gates G3-G6 are in `docs/gpu-routing-test-campaign.md` ("2026-10-15"). |
| knob form | `export K=${K:-v}` yields to the launch line; a hard `export K=v` does not (a line knob then reaches nothing). When a knob "did not take", check the form first. |
| the resolved recipe | `python scripts/fstat_proposal/run_combined_staged.py --print-recipe` under the launcher's exports (the flagship COMPOSITION test does this). |

### 2.2 The driver: `scripts/fstat_proposal/run_combined_staged.py`

This is what `mpiexec` runs in production, not `scripts/run_global.py`. It:

- builds `erebor.full_year_combined` with the env knobs (`build_fit`);
- composes the stage list: `gb_search_seed` x N (`GB_SEARCH_SEED_ITERS`,
  `FixedIterationStop`), `source_search` (`JointMaxLogLSearch` over the
  MBH/EMRI/SOBBH moves, `SOURCE_SEARCH_POSITION`), the numbered GB search
  stages from `V9_SEARCH_STAGE_PROFILES` (per-stage `phase_maximize`,
  `opt_snr`, `peak_min_snr`, `prior_births`, `reset_band_max`), optional
  `replica_pe`, then `full_pe`;
- resolves the per-stage knobs: `GB_SEARCH_STAGES`, `GB_SEARCH_{N}_OPT_SNR`
  (N = 2, 3 only; stage 1 has none), `GB_SEARCH_{N}_PRIOR_BIRTHS`,
  `GB_SEARCH_{N}_NOISE_MODE` (`fixed` / `interleaved` / `cycle_end`),
  `GB_SEARCH_{N}_NOISE_MAX_ROUNDS`, `GB_SEARCH_{N}_SOURCE_EVERY`,
  `GB_SEARCH_{N}_RESET_VALVES` (at a FRESH entry, not a resume),
  `GALFOR_RATCHET*` (the dropped ratchet, `GatedNoiseSearch`),
  `STAGE_V9_SEARCH=0` (one legacy `gb_search`). The profile row wins over the
  launcher's global exports on stage entry (`_apply_profile` writes
  `opt_snr`, `phase_maximize`, `prior_births` onto every GB move), so
  `GB_OPT_SNR_LIMIT_SEARCH` / `GB_RJ_PHASE_MAXIMIZE` do not reach a
  numbered search stage;
- prints the recipe (`describe_recipe` / `--print-recipe`).

The production recipes as of 2026-10-11:

| run | stages |
|---|---|
| 6mo v9 (store fixed since September) | `gb_search_seed` -> `gb_search_1` -> `gb_search_2` -> `gb_search_3` -> `replica_pe` -> `full_pe` |
| 9mo v9 / 1yr v9 / flagship with a parent | `gb_search_seed` x3 -> `source_search` -> `gb_search_1` -> `gb_search_2` -> `full_pe` |
| flagship 3mo (no parent) | `source_search` -> `noise_search` -> `gb_search_1` -> `gb_search_2` -> `full_pe` |

A store's recipe is FIXED once written (`GFHDFBackend.add_recipe` refuses a
dropped or reordered stage; it can append stages after the active one). A
recipe change therefore reaches only NEW stores.

### 2.3 Stock configuration: `src/lisatools/globalfit/stock/erebor/`

| file | holds |
|---|---|
| `fit.py` | `EreborFit` / the general settings block (`TOBS_TARGET`, `DATA_MODE`, `MIN_FREQ`, walkers, GPUs, ...). |
| `gb.py`, `vgb.py` | `GBSettings` / VGB settings: every `GB_*` knob is a dataclass field with `env_default("GB_...")`. The band grid helper lives here too. |
| `mbh.py`, `emri.py`, `sobbh.py`, `source_runtime.py` | the three source branches: waveform path knobs (`MBH_LIKELIHOOD`, `EMRI_LIKELIHOOD`, `SOBBH_LIKELIHOOD`), batching, ladders, repeats, warm starts; `source_runtime.py` is the runtime selection (which move class, which table) and builds the MBH inner-move stack (`make_mbh_inner_moves`: the eigen/stretch move plus the LISA-frame sky-mode hops of `sampling/moves/skymodehop.py` in the ICRS basis, `MBH_SKY_MOVE_{BOTH,LAT,LONG}_WEIGHT` = 0.02 / 0.04 / 0.02 of the per-leaf draws; the hops read the constellation phase off the run's orbits). |
| `noise.py`, `stochastic.py` | PSD + galfor (+ SGWB) settings, priors (`GALFOR_FK/F1/F2_PRIOR`), start params, the reference-fit pins. |
| `injections.py`, `transforms.py`, `wrappers.py` | catalogue -> sampling basis, parameter transforms, waveform wrappers. |
| `variants/full_year_combined.py` | the production variant (every branch). `all_sources`, `gb_no_fg`, `noise*`, `blank` are the smaller ones. |
| `stock/base.py` | `StockGlobalFit`, `ENV_ALIASES`, `add_move` / `add_branch` / `sample`. The knob-naming rule: **env knob = the capitalized attribute name**, branch-prefixed. |

To find which code reads a knob: `grep -rn '"KNOB_NAME"' src/lisatools scripts/fstat_proposal/run_combined_staged.py`.
Three spellings occur: `env_default("K", ...)` (stock fields),
`os.environ.get("K")` (moves, recipe, driver), and `${K:-v}` (launchers).

### 2.4 The recipe layer: `src/lisatools/globalfit/recipe.py`

| name | role |
|---|---|
| `Recipe`, `Stage` | the declarative stage list AND the runtime step engine (`docs/stock-stages-and-moves.md` for the API; that doc predates the `gb_search` / `replica_pe` stage kinds and the standalone `in_model*` moves, so read it for the mechanics, not the production composition). `Recipe.__call__` is the per-iteration hook (stage advance, residual snapshot). |
| `RecipeStep` -> `SearchRecipeStep` / `PERecipeStep` / `RJRecipeStep` / `ReplicaPERecipeStep` | kind-specific stopping rules. `PERecipeStep.note_recipe_step` applies the PE declarations (RJ flip fraction, in-model repeats). |
| `SearchStageProfileStep` | the numbered GB search stages. `note_recipe_step` (head-only stage entry: `_apply_profile`, valve release / reset, legs resume, ratchet), `setup_run`, `_stopping_rules` (the stage-end rules), `_drive_ratchet`. Every mutation prints a `[V9-STAGE]` line. |
| `band_shutoff_w_pending_total`, `band_shutoff_w_armed` | the stage-end clock: active (occupied, not shut) pairs; 0 ends the stage under `GB_SEARCH_STAGE_END_ON_SHUTOFF`. |
| `release_band_shutoff_window`, `reset_band_logl_max`, `revive_band_shutoff_on_state`, `apply_shutoff_floor_to_state`, `force_fstat_refit` | the valve operations a stage entry can perform on the live state. |
| `build_gb_moves`, `build_vgb_moves`, `build_noise_moves` / `build_psd_moves`, the `SourceMoveBuilder` family (`MBHBatchedMoveBuilder`, `EMRIDirectMoveBuilder`, `SOBBHChunkedMoveBuilder`, ...) | the stock move constructors; `resume_ladder_wins` (a stored rung count beats `{BRANCH}_NTEMPS`); `resolve_resumed_ladders` (a resumed MBH / EMRI / SOBBH store keeps its stored per-leaf ladders unless `{BRANCH}_LADDER_RESET_ON_RESUME=1`; before 2026-10-13 every resume re-tiled the configured ladder); `_warm_start_seed`. The default per-leaf ladder is `make_ladder(ndim, {BRANCH}_NTEMPS)` (`stock/erebor/common.py::default_source_ladder`, user ruling 2026-10-13; was 1/1.2^k). |
| `apply_rj_flip_fraction`, `apply_inmodel_repeats` | PE-stage declarations, applied on entry and shipped to every rank (`GB_RANK_LIVE_ATTRS`). |

### 2.5 Moves: `src/lisatools/globalfit/moves/`

| file | what |
|---|---|
| `globalfitmove.py` | `GlobalFitMove`, `Move`, `MoveBuildContext`; `GFCombineMove` (bundles a stage's moves, runs the legs via `_propose_leg`, prints `[GF_TIMING]` and `[LEG]`); the max-lnL plateau wrapper that prints `[MAXLOGL]` (noise searches, `source_search`). |
| `gbspecialstretch.py` | ALL GB and VGB moves. Classes: `GBSpecialBase` (everything shared), `GBSpecialStretchMove` (`in_model*`), `VGBSpecialStretchMove` (`vgb_pe`), `GBSpecialRJPriorMove` (`rj_prior_removal`, `rj_prior*`), `GBSpecialRJFStatGridMove` (`rj_fstat_search` / `rj_fstat_pe`: owns the F-stat fit call, the per-walker valve judge and the stage-end census), `GBSpecialRJSerialSearchMCMC`, `GBSpecialRJRefitMove`, plus `_ColdBandLnL` (the valve statistic and `judge`), `_InModelGroupState` / `_InModelConvergeState` (levels 2 / 1), `GBRankSession` (the per-rank body under fan-out). The warm-start RJ move (`rj_warm_search` / `rj_warm_pe`) is built around `warmstart/proposal.py`; with `GB_WARM_SELF=1` its birth law is `warmstart/self_refit.WarmSelfMixer` (original vs self refit per propose, section 2.8; hooks `_warm_self_begin` / `_warm_self_apply` / `_warm_self_note`). |
| `gbbands.py`, `gb_likelihood.py`, `gb_chol_cache.py` | band sorter / scheduler / sub-band buffer; the GB likelihood engine; the host-side in-model proposal factor cache (`GB_CHOL_CACHE`). |
| `addremovemove.py` | `ResidualAddOneRemoveOneMove`: the per-leaf add/remove scheme for MBH / EMRI / SOBBH, the eigen tables (`_build_eigen_table`, `_gram_info`), the per-leaf ladders, the eigen step-scale adaptation (`_eigen_adapt_table` + the per-leaf ring buffer `_note_eigen_adapt_sample`; `{BRANCH}_EIGEN_AXIS_ADAPT`, below). |
| `mbhbatchedmove.py`, `emridirectmove.py`, `sobbhspecialmove.py` | the fast scoring paths (batched windowed MBH, direct-to-WDM EMRI, lookup SOBBH); defaults since 2026-09-30 / 10-03. `docs/emri-direct-wdm.md`, `docs/sobbh-wdm-lookup.md`. |
| `psdmove.py` | `PSDMove`: psd / galfor / sgwb (one instance each), `PSD_BATCH`, `forced_noise_step` (ratchet), the noise Fisher. |
| `eigen_refresh.py`, `eigen_table_persist.py` | eigen tables from an information matrix and their sidecar pkl (with `{BRANCH}_EIGEN_AXIS_ADAPT` on, the entry also carries `sigmas_fisher`); the step-scale adaptation helpers (`cold_unit_deviations`, `empirical_axis_widths`, `adapt_axis_sigmas`, `axis_caps`, `carried_axis_widths`). Route map: `docs/eigen-info-routes.md`. |
| `walkerfanout.py`, `multigpumove.py` | the head <-> compute-rank fan-out mixin (`fanout_propose`, live attrs, pooled ladder adaptation). |

**The source eigen step-scale adaptation** (user ruling 2026-10-14;
`docs/eigen-info-routes.md`, "the adapted step scale"). The MBH / EMRI /
SOBBH eigen tables take their DIRECTIONS from the information matrix
(Gram / ll, at one point) and, with the switch on, their per-axis step
SCALE from the chain. The scale is the robust width (1.4826 x MAD) of the
leaf's last K end-of-visit rung ensembles, in cold units (deviation from each
rung's own centre, periodic columns wrapped, x sqrt(beta)), projected
onto the table's axis basis. The floor is the matrix sigma and the cap is the
table's own. Each rank adapts from its own walkers' rungs. Why: on the 9mo
(job 763) the EMRI ridges were 1.4-56x wider than the table sigma along the
table's own widest axis.

| knob | where | default | meaning |
|---|---|---|---|
| `{EMRI,MBH,SOBBH}_EIGEN_AXIS_ADAPT` | Settings field `eigen_axis_adapt` (emri.py / mbh.py / sobbh.py) -> `SingleSourcePEBuilder` -> move kwarg | `False` | master switch; off = every table bit-identical, no buffer, no log line |
| `{EMRI,MBH,SOBBH}_EIGEN_ADAPT_VISITS` | move `eigen_adapt_visits` | 50 | ring-buffer length in leaf visits (x rungs x the rank's walkers) |
| `{EMRI,MBH,SOBBH}_EIGEN_ADAPT_MIN_VISITS` | move `eigen_adapt_min_visits` | 10 | buffered visits before a refresh adapts (until then the information-matrix table) |

The scale changes only at a table refresh (`{BRANCH}_EIGEN_REFRESH`), so
the cadence knob also sets the adaptation cadence.

### 2.6 State and storage

| name | where | note |
|---|---|---|
| `GFState`, `GFBranchInfo`, `AllGFBranchInfo` | `globalfit/state.py` | the full host state (cold chain) plus `sub_states`. |
| `ModuleSubState` -> `GBState`, `PerLeafLadderState` -> `MBHState` / `EMRIState` / `SOBBHState` | `globalfit/state.py` | each branch's full tempered ensemble; "the sub-state IS the schema" (`conventions.md`). `GBState.band_info` holds every per-band / per-pair array (the valves). |
| `GFHDFBackend`, `ModuleSubBackend` | `globalfit/hdfbackend.py` | the store writer/reader; `save_step`, `add_recipe`, `completed_recipe_step` (stage stamps, routed through the saver since 5bf31e7b), `write_saved_after`, `_validate_resume_readable`, the running backup copy. |
| the store layout | section 5 | |
| `midit_checkpoint.py` | `globalfit/` | mid-iteration pickle of the state (`MIDIT_CHECKPOINT`; OFF on the legged search launch lines). The rewind trap is in its docstring. |
| `residual_snapshot.py` | `globalfit/` | `<store>_residual_snapshot.npz` every `GF_RESIDUAL_SNAPSHOT_EVERY` rows (default 3, not in the launcher; 0 = off), every cold walker since 1f1289cf; what the monitor's residual panels draw. |
| `legs.py` | `globalfit/` | `LegCursor`, `leg_ends_from_names`; resume by the saved-after NAME. |
| `noise_ratchet.py` | `globalfit/` | the galfor ratchet (nudge / hold / release). Dropped from production 2026-10-03; the 6mo store's `gb_search_3` still carries it. |
| `wdm_lookup_store.py` | `lisatools/` | find-or-build of the EMRI / SOBBH lookup tables (`EMRI_DIRECT_TABLE`, run-folder canonical name, lock + atomic rename). |

### 2.7 Run loop, ranks, GPUs

`globalfit/run.py` (`GlobalFit.run_global_fit`, `prepare_main`,
`setup_acs(rebuild_residuals=True)` rebuilds every residual from coords on
resume, `_seed_rank_streams`), `globalfit/communication/` (`ranks.py`
roles + `build_layout` + `prepare_rank`; `fanout.py`; `fakecomm.py` for
tests; `rowfanout.py` for one-walker replicas). Everything about `np`,
`NGPUS`, `GPUS_PER_RANK`, `RANKS_PER_GPU`, `GF_GPU_ROUTING`, per-rank log
files and stdout prefixes is in [`global-fit-launch.md`](global-fit-launch.md);
the cluster runbook is [`multirank-cluster-gates.md`](multirank-cluster-gates.md).
Production = one walker per compute rank (4 walkers, 4 GPUs, 2 nodes), a
saver rank and a monitor rank (`GF_MONITOR_RANK`, `GF_MONITOR_ITER`).

### 2.8 Warm start and seeding of a child run: `globalfit/warmstart/`

`fit_from_store.py` (GMM mixture from the parent's last `GB_WARM_START_LAST_K`
cold rows) -> `match_referee.py` -> `referee_apply.py` -> the npz under
`<run_dir>/warmstart/`; `proposal.py` (the `rj_warm_*` container);
`noise_pin.py` (psd + galfor start from the parent's max-lnL walker,
`GF_SEED_STORE`; point it at the parent's `*_running_backup_copy.h5` when the
parent job is still writing); `sources.py` (MBH / EMRI / SOBBH leaves with parent SNR > 10
start at the parent's cold chain, `SOURCE_WARM_START_STORE`); `opt_snr.py`,
`basis.py`. The launcher's `[WARMSTART]` block prints the three by-hand
commands for a login-node pre-build. Design: `docs/superpowers/specs/2026-09-18-warmstart-observable-basis-design.md`.

**The warm-start SELF refit** (`self_refit.py`, user rulings 2026-10-13: "Starting in
gb search 2 (and in full pe). Let's rebuild our warm start proposal with the last 50
samples (x4 walkers)"; "keep the original warm start in the fit"; 25 % original / 75 %
refit in full_pe every 250 rows; 2026-10-14 for gb_search_2: refit EVERY stored row and
draw ONLY from the refit, after the 9mo's first PE epoch carried 84 % of the lnL gain in
its seven self draws; 2026-10-15: build on the head at the warm moves' entry, one build
gives a FULL and a FILTERED refit, search draws only the filtered one, PE one third each
of original / full / filtered, PE every 25 rows). Why: per-walker GB
completeness at SNR > 8 was 91 % on the 9mo while the union of the four cold walkers
was 96.5 %; a mixture fitted to ALL cold walkers proposes to each walker what the
others hold. The pieces:

- **build, ON THE HEAD at the warm move's entry** (default since 2026-10-15,
  `GB_WARM_SELF_AT_ENTRY=1`: `WarmSelfBuffer` + `WarmSelfHeadBuilder`, armed by
  `run.prepare_main` -> `make_head_builder` before the engine is built). The head may not
  read the store the saver writes, so it keeps its own copy of the last
  `GB_WARM_SELF_LAST_K` stored cold rows (all walkers): filled from the store at set-up
  (before the first save; also the active stage and its start row for the clock) and
  appended by the head's `GFHDFBackend.save_step` with every state it saves. When a warm
  move starts a propose (`GBSpecialBase._warm_self_begin` -> `WarmSelfMixer.begin_propose`
  with the live `gf_stage_name`), the registered head builder checks `warm_self_due`
  (stage `gb_search_N` with N >= 1 or `full_pe`; clock = max(stage start, latest complete
  epoch's `built_at_row`); the stage start is the store's on a resume, else the stored row
  count when the head first sees the stage at a warm entry; a stage's FIRST warm entry with no
  refit built in it yet builds at once from the rows already stored). If due it writes the buffer to
  `epoch_NNNN/leaf_rows.h5` (the same layout `copy_leaf_rows` makes from the store), runs
  fit -> referee -> apply exactly as `build.py` does (`_default_runner`, CLI defaults, the
  run's Tobs, `--gmm-gpu` = the head's device), then `consensus_filter`, writes the
  manifest LAST, prunes, and the propose that triggered it draws from the new epoch.
  Every rank waits (the compute ranks are parked on the propose; ~75 s on the 9mo), so the
  `[GF_TIMING]` wall of `rj_warm_search` / `rj_warm_pe` now INCLUDES the build on a build
  propose: not a regression. Only the head rank builds. Rows are counted as the states the
  head handed to the saver (`row_count: head states` in the manifest). A failure logs,
  leaves an incomplete epoch and retries a cadence later; it never raises.
- **the FILTERED container** (`consensus_filter`): every leaf row of the refit is
  assigned to its component with the referee's `assign_members`; a cluster is CONSENSUS
  when EVERY walker holds it in >= `GB_WARM_SELF_CONSENSUS` (0.95) of its rows AND >= that
  share of all samples hold the same number of its leaves (>= 1). The filtered npz is the
  refit minus the consensus clusters (same schema). Measured on the 9mo (rows 82-89 and
  103-110): ~2,160 of ~3,230 source groups are consensus and took 83 % of the full
  mixture's draws; draws landing on a source the walker already holds 93 % -> 57 %
  (useful draws ~7 % -> ~42 %); ~60 blends (all walkers, differing counts) are kept.
- **build, on the SAVER rank** (`GB_WARM_SELF_AT_ENTRY=0` only; `WarmSelfHook`, called by
  `hdfbackend`'s save loop after a batch that saved a row): when the store's active recipe
  stage (`store_stage`) is `gb_search_N` (N >= 1) or `full_pe` and `rows - clock >=
  cadence`, it copies the last `GB_WARM_SELF_LAST_K` cold rows through the saver's own
  `open("r")` (scoped and closed before anything slow: a held read handle makes the
  saver's next `open("a")` fail) into `epoch_NNNN/leaf_rows.h5` and runs fit -> referee
  -> apply on it (no filtered container: the filtered choice serves the full refit). A
  queued save always wins; `WarmSelfWatchdog` reports a build that held a save up and
  switches the builder off if it ever cost stored rows.
- **epochs**: `<run_dir>/warmstart/self/epoch_NNNN/` = `self_fit.npz`,
  `self_fit_referee.npz`, `self_refereed.npz`, `self_refereed_filtered.npz` (head builds)
  and `DONE.json` written LAST (tmp + fsync + rename: `epoch`, `built_at_row`, `stage`,
  `stage_start`, `rows [a, b]`, `n_samples`, `n_clusters`, `n_components`, `npz_filtered`,
  `n_clusters_filtered`, `consensus`, `built_on`, `walls`, `device`). No readable
  `DONE.json` or no npz = not an epoch; the next build takes `1 + max(every epoch dir)`.
  After each successful build `prune_epochs` strips every older epoch to its `DONE.json`
  except the newest 3 complete ones and every 10th.
- **split + adoption, in the warm moves** (`WarmSelfMixer` is the
  `rj_proposal_distribution` of `rj_warm_search` / `rj_warm_pe` when `GB_WARM_SELF=1`;
  built by `recipe._warm_self_wrap`). At each propose the head polls the epoch root,
  adopts a newer complete epoch, DRAWS ORIGINAL / FULL / FILTERED at random with
  probabilities ~ the twin's split (`0,0,1` search = always the filtered refit; `1,1,1`
  PE = one third each, drawn per propose; 100 % original before the first epoch and in
  every stage that never refits -- `gb_search_seed`, `source_search` -- even with epochs
  on disk; an epoch
  without a filtered npz serves its full refit where the filtered one is chosen) and
  ships `{"epoch", "choice"}` in the `gb_run_proposal` payload (`"warm_self"`); every
  rank loads that epoch's two npz files from the shared filesystem (`_warm_self_apply`,
  before its BandSorter is built) with the original's Tobs / floor box / floor_eps /
  p_floor / circ_images / use_cupy and the seeds `_warm_start_seed(gb_info, tag ^ (epoch
  + 1))` (full) and that tag `^ _FILTERED_SEED_XOR` (filtered). One container serves the
  births and every density of a propose. The PE draw-one share of `rj_warm_pe`
  (`GB_PE_RJ_WARM_FRACTION`) is unchanged: the split happens inside the move.

| knob (field on `GBSettings`) | default | meaning |
|---|---|---|
| `GB_WARM_SELF` (`warm_self`) | 0 | master switch; 0 = no builder, no mixer, bit-identical; needs `GB_WARM_START_COMPONENTS` |
| `GB_WARM_SELF_AT_ENTRY` (`warm_self_at_entry`) | 1 | build on the head at the warm moves' entry; 0 = the saver builds after its saves (2026-10-13 design) |
| `GB_WARM_SELF_EVERY_SEARCH` (`warm_self_every_search`) | 1 | stored rows between refits in `gb_search_N`, N >= 1 (= every cycle at the entry; <= 0 off) |
| `GB_WARM_SELF_EVERY_PE` (`warm_self_every_pe`) | 25 | stored rows (= iterations) between refits in `full_pe`, checked when `rj_warm_pe` is drawn (<= 0 off; 150 before 2026-10-15, 250 before 2026-10-14) |
| `GB_WARM_SELF_LAST_K` (`warm_self_last_k`) | 50 | the run's last stored rows fed to a refit, every cold walker of each |
| `GB_WARM_SELF_SPLIT_SEARCH` (`warm_self_split_search`) | `0,0,1` | `rj_warm_search` weights original / full / filtered once an epoch exists |
| `GB_WARM_SELF_SPLIT_PE` (`warm_self_split_pe`) | `1,1,1` | `rj_warm_pe` weights original / full / filtered |
| `GB_WARM_SELF_CONSENSUS` (`warm_self_consensus`) | 0.95 | per-walker row share and same-count share above which a cluster counts as held by every walker (dropped from the filtered refit) |

`GB_WARM_SELF_FRACTION` / `GB_WARM_SELF_FRACTION_SEARCH` (0.75 / 1.0, two containers) are
no longer read (2026-10-15); `recipe._warm_self_wrap` warns if either is set. The fit /
referee options are not knobs here: they are the parent build's (CLI defaults of the
three modules); the referee's own backend choice (`GB_WARM_START_REFEREE_BACKEND`) is
read by the referee. Tests: `tests/test_warmstart_self_refit.py`,
`tests/test_warmstart_self_refit_entry.py`, `tests/test_warmstart_self_refit_prune.py`.

### 2.9 Monitoring: `src/lisatools/globalfit/monitor/` and `scripts/diagnostics/`

| tool | command | what |
|---|---|---|
| the dashboard | `python scripts/diagnostics/gf_last_iteration.py RUN_DIR [--rows N] [--line]` | one screen: last row, stage, pace, the leg in flight, the RJ valve clock, empty bands, F-stat epochs, GPU, lnL, leaves, the stage cycle with per-move walls. Reads the store + the newest `slurm_stdout_<job>.log` (falls back to it when the artifacts log is absent). First thing to run on any run dir or extracted tar. |
| the page (full + short) | `python -m lisatools.globalfit.monitor RUN_DIR [OUT.html] [--short-page] [--snapshot] [--build-truth]` | the html; `_short.build_glance` is the "at a glance" block on top; detail groups collapsed. In-run it is built by the monitor rank every `GF_MONITOR_ITER` saves. `GF_MONITOR_MATCH_STATS=1` for the overlap panels (default 1 since 2026-09-30). Per-sub-band history (GB search group, "Model Size & Likelihood", 2026-10-13; generator section 5a-1, read shared with 5a-0's `BAND_HIST`): `gb_band_evo` = three band x iteration rasters on one y axis (the sub-bands that ever held a cold leaf, one row each): band lnL (sum of `d_h - h_h/2` over its live leaves) minus its final value, GB count minus the final count, share of walkers with the per-walker valve shut; pooled to the pixel grid. `gb_band_movers` = the 16 sub-bands that moved most over the last 100 rows, a line per walker, a valve-shut strip on top. Both need every row's leaf frequencies, i.e. the full store; on a snapshot extract the lnL raster falls back to the valve gain `band_cold_logl_w` (residual-unit rows before da45aa98 blanked, window-sharing sub-band pairs folded) and the count raster says it is unavailable. The older `gb_band_ll` line panel sits below them. |
| page from a tar | `python -m lisatools.globalfit.monitor.from_tar SNAP.tar.gz [OUT.html] [--short-page]` | finds the run dir inside the tar; caches one extraction per tar under the temp dir (fills a laptop disk; `rm -rf` is safe). |
| truth set | `python scripts/diagnostics/build_truth.py STORE.h5 --out RUN_DIR/gb_truth_3to21.npz --kappa-out RUN_DIR/kappa_grid.npz [--iteration ROW]` | the detectable-GB denominator the page matches against. The default row is the latest usable one (filled, not rewound, not torn; since 2026-09-26); pass `--iteration ROW` only to pin a row, e.g. to keep a previous denominator. Always route `--kappa-out` into the run dir (the CWD default clobbers the LAT-root 3-month cache). |
| snapshot tars | `bash scripts/fstat_proposal/make_snapshots.sh RUN_DIR` (shell) and `monitor/snapshot.py` (the saver's in-package twin; both must change together) | a reduced `*_extract.h5` (full history of the small arrays, last `KEEP` rows of the big chains) + everything small in the run dir + the newest job logs. `--add-fstat` keeps the epoch payloads. |
| store extract | `monitor/_store_extract.py` (`scripts/diagnostics/gf_store_extract.py` is a shim) | the reduced store; same layout and `iteration` attr; chain COORDS outside the keep window read back as ZEROS. |
| identity | `monitor/_identity.py` | run kind -> banner / arm-cache tag (`gf_arm_<tag>.npz`); the flagship series tag rule. A new store name family needs a rule here or it lands on another run's cache. |
| log discovery | `monitor/_logs.py::discover_run_logs` | head log, rank logs, and the slurm stdout (a verbatim SUPERSET of the run logs; the only place `[GF_TIMING]`, `[V9-SEED]`, the stage table and python warnings appear). |
| compare page | `scripts/diagnostics/gf_compare_gen.py` | several runs on one page. |
| others | `gb_source_deep_dive.py`, `gb_walker_agreement.py`, `gb_temper_audit.py`, `gf_state_digest.py`, `analyze_rj_trace.py`, `gf_layout_preflight_mpi.py` | per-topic probes; read each docstring. |

### 2.10 Store surgery tools: `scripts/fstat_proposal/*.py`

All dry-run by default, all meant for a STOPPED store, all with a "why this
exists" docstring. `reset_recipe_stage.py` (re-open / rewind a stage; sets
the midit checkpoint aside), `copy_store_leaf.py` (copy one source leaf
between walkers on every rung; built for the stuck EMRI 3 walker-0 case),
`rerung_store_branches.py` (change a branch's rung count; `--repair-unlimited`),
`salvage_recipe_group.py` / `repair_recipe_group.py` (torn recipe headers),
`compact_gf_store.py`, `graft_noise_state.py`, `rescale_store_walkers.py`
(walker scale-up by lag-cloning), `migrate_gb_*.py` (band arrays added to an
old store: a band array added after a store was created is never gained by
running, it has to be migrated; since 8e1f5f5a `save_step` creates missing
valve datasets lazily).

### 2.11 Sibling repos the fit calls into

| repo (branch) | the modules the fit touches | map |
|---|---|---|
| GBGPU (`dev`) | `gbgpu.py` (waveform), `gbcomps.py` (`GBWDMComputations`, chunked-het), `gbsignalhetcomputations.py` (sig-het: the in-model likelihood engine, `SIGHET_*` knobs), `gblookupcomputations.py` (lookup reference / Gram, `SIGHET_REF_BUILD=lookup`), `gb_likelihood.py` | `../GBGPU/docs/codebase-map.md` |
| BBHx (`dev`) | `waveformbuild.py`, `mbhphentax.py` (MBH), `sobbhtdionfly.py`, `sobbhcomps.py` (SOBBH chunked) | `../BBHx/docs/codebase-map.md` |
| FastEMRIWaveforms (`gpu_backend`) | `waveform/`, `trajectory/`, `amplitude/`, `summation/` | no map yet |
| Eryn (`dev`) | `moves/` (`tempering.py`, `rj.py`, `stretch.py`, `eigenaxis.py`, `ridgegibbs.py`, `combine.py`), `backends/`, `state.py` | `../Eryn/docs/codebase-map.md` |
| GPUBackendTools (`spline`) | backends, `Interpolate` | `../GPUBackendTools/docs/codebase-map.md` |

Cross-repo graph and the backend-wheel model: [`architecture-map.md`](architecture-map.md).
A native (`.cu`/binding) change in any of these needs a cluster REBUILD before
it is in production; a pure-Python change needs only a `git pull` (editable
installs).

---

## 3. How one search cycle runs (what you are reading in the logs)

The 9mo `gb_search_2` move order, as the dashboard prints it (`<stage>.move_order`
stamped on the recipe group), with the 2026-10-11 wall times:

```
leg 1   vgb_pe               10 s      VGB in-model
        rj_warm_search       ~10 min   RJ births/deaths from the parent-run warm-start mixture
        in_model             ~10 min   GB in-model polish (leg ender -> row saved)
leg 2   rj_fstat_search      25-50 min owns the F-stat grid fit when due (epoch), RJ births from the
                                       grid, the per-walker valve JUDGE and the stage-end census
        in_model_fstat       ~10 min   (leg ender -> row)
leg 3   rj_prior_removal     ~20 min   RJ from the prior: deaths only in gb_search_1, births+deaths from gb_search_2
        in_model_removal     ~10 min   (leg ender -> row; the LAST leg runs through the tail:)
        [ridge gibbs]        0 s
        sobbh / mbh / emri   10 / 25-50 / 60 s   the source block (every cycle since ddad1304)
        noise_cycle_end_search 7-30 s  ONE psd+galfor max-lnL search to convergence (gb_search_2; frozen in gb_search_1)
```

Stage profiles (`V9_SEARCH_STAGE_PROFILES`): `gb_search_1` = opt-SNR floor 8,
F-stat peak floor 8, phase maximization ON, prior move deaths only, noise
fixed, no source moves in the 6mo (sources every cycle in the 9mo/1yr);
`gb_search_2` = floors 5 / 6.25, phase maximization OFF, births+deaths,
sources every cycle, valves reset at the fresh entry, noise at the cycle end.
`full_pe` = every move every iteration, no maximization anywhere
(`feedback`: no maximizing over parameters in PE), one draw-one GB RJ move per
iteration (`GB_PE_RJ_DRAW_ONE`; the three-way draw `GB_PE_RJ_FSTAT_FRACTION`
0.45 / `GB_PE_RJ_WARM_FRACTION` 0.1 / the prior move the rest; with
`GB_WARM_SELF=1` the warm move itself then draws one third each from the
original warm start, the run's full refit and its filtered refit, section 2.8). The PE
declarations applied on entry by `PERecipeStep.note_recipe_step`:
`PE_INMODEL_REPEATS=25` (`apply_inmodel_repeats`; `GB_NUM_REPEAT_PROPOSALS=25`
is the global export), `GB_PE_RJ_FLIP_FRACTION=0.1` (`apply_rj_flip_fraction`),
`GB_FSTAT_REFIT_EVERY_PE=250`; a PE refit fits a RANDOM cold walker
(`GB_FSTAT_PE_REF`, seeded; search refits fit the min-lnL walker).

Two facts needed to read leaf counts and SNR tails: the opt-SNR floor tests
`sqrt(h_h)` of the PROPOSED template, so a current source that falls below
the floor is frozen alive, not removed; and a production birth draws
(f_mid, fdot, sky) from the F-stat grid and amplitude / phase / inclination /
polarization from the prior, with no maximization (`GB_RJ_FSTAT_DIST_BIRTH=0`,
`GB_RJ_AMP_MAXIMIZE=0`; the F-stat extrinsic maximizers are never used for
births). Phase maximization of the scoring is on in `gb_search_1` only.

In-model GB updates are NOT a separate move in the RJ moves: every RJ move
repeats in-model proposals on the sources it touched (`num_repeat_proposals`,
25 in PE). The pure `in_model*` moves are the standalone passes with the
level-2 group rule.

Things that move leaf counts and lnL WITHOUT an RJ accept: the vertical
(all-rungs) tempering swaps hand band states down the ladder inside every GB
move, so a row saved after `in_model` can show +-20 leaves per walker.

The full step-by-step walkthrough (preamble, stage entry, inside one RJ
propose, stage end, kill switches) is
`docs/superpowers/specs/2026-09-24-v9-recipe-walkthrough.md`.

---

## 4. The gates and valves (where each lives)

Four distinct mechanisms print "shutoff" or "converged". Never conflate them.

| mechanism | unit | statistic | where | log tag | store arrays |
|---|---|---|---|---|---|
| **Level 1**: in-model convergence of ONE source | a sorter row (follows the source through rung swaps; never reset on a rung change, user ruling) | accumulated accepted delta-lnL vs its running max | `_InModelConvergeState`, `_run_in_model_repeats` | `[GB_IMCONV]` (printed by the RJ moves' repeat blocks only; the pure `in_model*` moves print `[GB_IMGROUP]` and no `[GB_IMCONV]`) | none |
| **Level 2**: group rule of the pure in-model passes | a (walker, band) per pass | per-sub-band cold dlnL over a pass window (`GB_INMODEL_GROUP_ITERS`) | `_InModelGroupState` | `[GB_IMGROUP]` | none |
| **Level 3**: the per-walker RJ valve = the stage-end clock | an occupied (walker, band) PAIR | the band GAIN `1/2<d|d> - 1/2<r|r>` under the current noise (da45aa98; before that the residual term) vs a running best, tolerance `GB_SEARCH_BAND_SHUTOFF_LL_TOL` (20 nats), `GB_SEARCH_BAND_SHUTOFF_CONV_ITER` (3) judged cycles; judged once per cycle inside `rj_fstat_search` | `_ColdBandLnL.judge`, `_update_search_band_shutoff`, `_arm_search_stage`, `_release_search_band_shutoff`; stage side `release_band_shutoff_window`, `band_shutoff_w_pending_total` | `[GB_GATE <move>] occupied N \| shut N (x%) \| active N -- streak p50/p90 of 3 ...` and `[V9-STAGE ...] ... pair(s) still active` | `sub_backend/gb/band_rj_shutoff_w`, `band_shutoff_w_step`, `band_shutoff_best_w`, `band_shutoff_streak_w`, `band_cold_logl_max_w`, `band_cold_logl_w`, `band_shutoff_reset_w`, `band_cold_logl_peak_w` (per row, per walker, per band) |
| **Empty-band valve**: births off in barren high-f bands | a BAND (all walkers) | cold occupancy 0 for `GB_RJ_BAND_SHUTOFF_ITERS` iterations; floor `GB_RJ_BAND_SHUTOFF_FMIN_MHZ` (10 mHz); revive check every `GB_RJ_BAND_SHUTOFF_RESET_ITERS` (10) iterations, revives only if the noise fingerprint moved | `_update_band_shutoff`, `_band_shutoff_restore`, `_band_shutoff_revive` | `[GB_BAND_SHUTOFF <move>] ...`, `[GB_BAND_REVIVE]` | `sub_backend/gb/band_rj_shutoff` (per row, per band), `band_occ_streak`, `band_shutoff_since_revive` |

The level-3 window (user ruling 2026-10-13, "Give every sub-band its
covering layer as the valve window"): the gain is formed on whole WDM
layers, `gbspecialstretch.wdm_window_layers` via `GBSpecialBase._window_bins`,
the one geometry the residual term, the data term (`_window_residual_lls` on
the band grid) and the in-model sites' per-slot window (`_slot_band_windows`
-> `_observed_band_gain`) all read. Layer `m` is CENTRED on `m * layer_df`
(tile `[(m - 1/2), (m + 1/2)) * layer_df`; a GB at `f` lands in layer
`round(f / layer_df)`). Since 2026-10-13 every band owns each layer whose
tile it overlaps (`cover=True`), so the 8 sub-bands of a 9mo tile SHARE one
gain (4 at the two grid ends). Before, only the layers whose centre lies in
the band counted: 924 of the 1232 9mo sub-bands (all but band index 0 and 7
mod 8) had an empty window, a gain of exactly 0 on every row, and shut after
3 judges whatever their sources did. The cap-cell grid keeps that centres
rule (`cover=False`; `_cap_cells_resolvable` needs sub-layer cells to come
back empty). The rule is stamped per store in
`sub_backend/gb/band_window_rule` (1 = centres, 2 = covering; `state.py`
`VALVE_WINDOW_RULE*`, `ensure_valve_window_rule`); the first judge on a
record learned under an older rule re-bases it once
(`relearn_valve_window`: running max / best / peak take that step's gain
under the new geometry; streaks, reset counts and the shut set are
untouched and that step is not judged -- user ruling 2026-10-13, "a reset
mid stage does not adjust anything about convergence/turn over"; log
`[GB_GATE <move>] window geometry v2: ...`). Rows written before the stamp
existed read 1 (`GBState.predating_fill_values`). Readers that SUM the gain
over bands now count each layer up to 8 times
(`scripts/diagnostics/walker_lnl_sources.py`), and the monitor's
`gb_band_evo` fallback folds "window-sharing pairs" by adjacency, which
under rule 2 folds whole runs of bands: both written for rule 1.

How a shut level-3 pair behaves (user ruling 2026-09-28): it runs NOTHING
until the next stage: no births, no deaths, no in-model repeats, no pure
in-model passes, no vertical swaps. Shut pairs never reopen within a stage
(`judge` only sets `shut`); a stage entry releases every pair
(`release_band_shutoff_window`, log `per-walker RJ valve RELEASED at step
entry`) and `gb_search_2`'s fresh entry also revives the empty bands
(`reset_valves`). The all-time max persists across stages except where a
profile sets `reset_band_max`.

The F-stat comb sweep skips bands shut on EVERY walker OR empty-band-shut
(dabe3fe8); a shut band therefore also stops costing grid time. A zero-peak
epoch skips `rj_fstat_search` and its in-model partner (843b7ec9).

Stage end (`SearchStageProfileStep._stopping_rules`): `STAGE COMPLETE on the
SHUTOFF rule` when active pairs reach 0 (`GB_SEARCH_STAGE_END_ON_SHUTOFF`),
or the nleaves plateau (`GB_PLATEAU_ITERS`), or both; `full_pe` never ends
(`NUM_ITERATIONS`).

---

## 5. The store and the run directory

```
<run_dir>/                                   /shared/data/global_fit_output/gf_prod_9mo_v9/
  gf_prod_9mo_testing.h5                     the live store (pick the LIVE h5 by mtime AND attrs["iteration"])
  gf_prod_9mo_testing_running_backup_copy.h5 refreshed every BACKUP_ITER saves (the counter resets per job); a byte copy, the
                                             torn-row fallback for readers; after a REWIND it can carry a HIGHER iteration than
                                             the primary, and a restore from it would resurrect the discarded rows
  gf_prod_9mo_testing_extract.h5             only in snapshot tars: same layout, big chains only in the keep window
  gf_prod_9mo_testing_residual_snapshot.npz  data + every cold walker's residual (float32, active WDM box)
  gf_prod_9mo_testing_midit_checkpoint.pkl   only with MIDIT_CHECKPOINT=1
  *_eigen_tables.pkl                          the eigen-table sidecar
  gb_fstat_fit/shared_search/epoch_NNNN/     DONE.json (wall_seconds, n_peaks, clock) + the grid payload (not in tars unless --add-fstat)
  warmstart/<parent>_refereed.npz            the GB warm-start mixture
  warmstart/self/epoch_NNNN/                 GB_WARM_SELF=1 only: the run's own refits (self_refereed.npz + self_refereed_filtered.npz + DONE.json LAST)
  warmstart/self/epoch_NNNN/: older epochs hold DONE.json only (build history); the newest 3 complete epochs and every 10th keep their npz files (prune_epochs, after each successful build)
  gb_truth_3to21.npz, kappa_grid.npz         the truth set (built by build_truth.py; frozen denominator)
  gpu_util_<job>.csv                         head-node GPUs only: ts, idx, name, util, util_mem, mem_used MiB, mem_total, power, temp
  slurm_stdout_<job>.log                     the 30-s mirror of the job's stdout (SLURM_LOG); the superset log
  *_artifacts/globalfit_run.log, .rank<k>.log, global_fit.log, general_setup.log, *_setup.log, run_settings.log
  gf_prod_9mo_v9_monitor.html, _monitor_short.html
```

Inside the h5 (`gf_format_version=2`, cold chain only at the top level):

```
global_fit.attrs["iteration"]            the row count; rows past it hold stale values from an earlier job
global_fit/log_like     (it, 1, 1, nw)   cold lnL per walker      (cold chain index = [it, 0, 0])
global_fit/chain/<br>   (it, 1, 1, nw, nleaves_max, ndim)   gb basis: [dist, f0_mHz, Mc, phi0, cos_iota, psi, alpha, sin_delta, fdot_astro_ratio]
global_fit/inds/<br>    (it, 1, 1, nw, nleaves_max)         alive mask
global_fit/saved_after  (it,) utf-8      the leg-ender name a row was saved after ("" = unknown)
global_fit/recipe/<step>.attrs           status, order num, start_iteration, completed_iteration, move_order (json, write-once)
global_fit/sub_backend/<br>/             the branch's tempered ensemble:
    chain (it, T, nw, nleaves, ndim), inds, log_like, log_prior, betas / betas_all / band_temps, counters,
    d_h, h_h (it, nw, nleaves)           h_h = the leaf's SNR^2 under the walker's noise (always on for GB; {BRANCH}_RECORD_DH
                                         defaults ON for the batched MBH / direct EMRI / lookup SOBBH moves and OFF on the slow
                                         container path, where the datasets stay NaN)
global_fit/sub_backend/gb/band_edges (nbands+1,), the band / valve arrays above (it, nw, nbands) or (it, nbands)
```

Read with h5py; slice rows by `attrs["iteration"]`. An `_extract.h5` reads
like the live store except that chain coords outside its keep window are
zeros (inds, log_like, band arrays, noise chains are full history).

---

## 6. Answering "where is X" questions

1. **A log line**: grep the tag table (section 7) or `grep -rn '\[TAG' src/lisatools/globalfit scripts/fstat_proposal/run_combined_staged.py`.
   Tags are stable; wording inside drifts.
2. **A knob**: `grep -rn 'KNOB' src/lisatools scripts/fstat_proposal/run_combined_staged.py scripts/fstat_proposal/submit_gf_9mo_v9.sh`.
   The launcher comment block carries the ruling and date; the test allowlists
   say what is pinned.
3. **Why something is the way it is**: the module docstring, then the knob's
   launcher comment, then `git log -S'<string>' --oneline dev` and `git log --grep`,
   then the spec under `docs/superpowers/specs/`. Commit messages on `dev`
   are written as one-line rulings and are a good index.
4. **A stage's behaviour**: `--print-recipe` (section 2.1), then
   `V9_SEARCH_STAGE_PROFILES`, then `SearchStageProfileStep`.
5. **What a stored array IS**: the sub-state that allocates it
   (`state.py`, `storage_arrays` / `static_arrays` lists) and the move that
   writes it. Check what a statistic is before reading it as a likelihood
   (`band_cold_logl_w` was the data term only, no log-det, until da45aa98).
6. **Which rank did what**: non-head ranks prefix stdout with `[r<k>/c<i>]`,
   the saver with `[r<k>/saver]`; each rank has its own `globalfit_run.rank<k>.log`.
   `[GB_ACCEPT rj-split]` is printed once per rank: a count from the head log
   alone is one walker's births.
7. **Tests that exercise it**: `tests/test_<topic>_*.py` by name
   (`test_stage_end_rules`, `test_valve_release_at_step_entry`,
   `test_gb_shutoff_persist`, `test_gb_valve_gain`, `test_search_legs`,
   `test_fstat_zero_peak_skip`, `test_cross_device_reads`, ...). Most GB suites
   drive the logic through light fakes that bypass `__init__`; the suites that
   construct a real `GBSpecialStretchMove` are `test_temper_shutoff_bands`
   (cheap) and `test_gbspecial_flow` (cluster only).

---

## 7. Log-tag glossary

Emitted from `src/lisatools/globalfit/` unless noted. Counts of `[TAG` call
sites in the tree are a fair proxy for importance.

| tag | emitted by | read it for |
|---|---|---|
| `[SAVE] save_step X s (...; h5 now N MB)` | `hdfbackend.py` (saver) | row cadence; wall per iteration = consecutive `[SAVE]` timestamps |
| `[GF_TIMING] stage=.. move=.. it=..` | `globalfitmove.py` (stdout only) | per-move wall per iteration; `move=__total__`. `it=` is a PER-PROCESS counter that restarts at every job, as is every other "iteration" in the logs (eryn's progress, `[REPLICA_PE] N row(s) in-stage`); anchor to store rows through `[SAVE]` / `[LEG]` |
| `[LEG <stage>] cycle c leg i/n: ran [...] -> row saved after 'x'` / `[LEG ..] resume: ...` / `the stage's stored move order [...] differs from the live composition` (grep `stored move order`) | `globalfitmove.py`, `recipe.py` | which leg landed; a resume point; a cycle restarted at its head after a composition change |
| `[V9-STAGE <stage>] entering (recipe step k): profile {...}` and every mutation / `RELEASED at step entry` / `reset_valves` / `STAGE COMPLETE ...` | `recipe.py` | stage entry, what the profile set on each GB move, valve release, stage end |
| `[GB_GATE <move>] occupied N \| shut N (x%) \| active N -- streak p50 .. p90 .. of 3 ...` | `gbspecialstretch.py` | THE valve census; quote it, never infer the valve from leaf counts |
| `[GB_GATE <move>] window geometry v2: every sub-band now owns its covering WDM layer(s) ... re-basing the per-(walker, band) running max ... Stamped band_window_rule=2` | `gbspecialstretch.py::_relearn_valve_window_once` | once per store, at the first judge of a store learned under the centres rule (2026-10-13); the maxima take that step's gain, every pair keeps its streak and shut flag, and that one step is not judges |
| `[GB_BAND_SHUTOFF <move>] status ...` / `[GB_BAND_REVIVE <move>]` | `gbspecialstretch.py` | the empty-band valve: bands off, since-revive counter, revive events (`the noise model has moved by x%`; grep `has moved by`) |
| `[GB_ACCEPT rj-split <move>] ...` / `[GB_ACCEPT in_model]` | `gbspecialstretch.py` | births / deaths proposed vs accepted per rung and cold, PER RANK |
| `[GB_IMGROUP <move>] pass k: cold dlnL +x over N occupied sub-band(s); shut/open` | `gbspecialstretch.py` | the level-2 passes of the pure in-model moves |
| `[GB_IMCONV ...]` | `gbspecialstretch.py` | level-1 convergence census (survivors / newborns, at-floor fraction); emitted by the RJ moves' repeat blocks only, never by the pure `in_model*` moves |
| `[GB_VERT] ...` | `gbspecialstretch.py` | vertical swap census (priced / unpriceable pairs, acceptance) |
| `[GB_CELL_LL]`, `[GB_ORTHO_LL]`, `incremental ll drift X after proposal; rebuilding log_like from the residual`, `ll AUDIT`, `DELTA-vs-DELTA` | `gbspecialstretch.py` | GB bookkeeping / sig-het accuracy audits. Stored lnL is rebuilt from the residual every propose, so the drift line measures the ledger, not the store. |
| `[GB_CHOL_CACHE] refreshed N sources in X s` / hits / misses / `DISABLED` | `gb_chol_cache.py` | the proposal-factor cache; `DISABLED` must never appear |
| `[GB_STALE_POOL`, `active buffer slot(s) share a cell label` | `gbspecialstretch.py` | must never appear (slot-map guards) |
| `[GB_BIRTH_STALE <move>]` | `gbspecialstretch.py` | share of scored births whose peak is already absorbed |
| `[GB_TIMING ...]`, `[GB_INMODEL_BATCH]`, `[GB_SWEEP]`, `[GB_UNIT_SCAN]` | `gbspecialstretch.py` | GB-internal timing shares (shares, not a wall budget) |
| `[GB_CAP_PW]`, `[GB_CAP_INC]`, `[GB_CAP_LL_CHECK]` | `gbspecialstretch.py` | per-walker leaf caps (off in v9) |
| `[GB_STAGE <move>]` | `gbspecialstretch.py` | RJ staging (cells excluded from staging). The planned `[GB_STAGE_CENSUS]` line was never built; do not look for it |
| `F-stat grid fit epoch N ... done in X s`, `[stageA] comb + peak selection: N peaks in X m`, `[stageB] banded Mc ...`, `[peaks] N peaks`, `[FSTAT_CTR*]` | `sampling/fstat_gridfit.py` (the fit: `run_fstat_grid_fit`, stage A `run_comb_scan`, stage B `run_stacked_stage_b`), `gbspecialstretch.py` (the call site) | the F-stat epochs: stage A comb (nodes x sky; the lever is `FSTAT_COMB_NSKY_MAX`) vs stage B (per-peak grids; `FSTAT_STAGEB_NSKY_MAX`, `FSTAT_STAGEB_GROUP_MAX_GB`), peak counts. A 0-peak epoch SKIPS `rj_fstat_search` and `in_model_fstat` (843b7ec9; the warning says `NOT falling back to the prior`) |
| `[MAXLOGL] stage=.. iter=.. flat n/m ... converged=k/4` / `done after N iterations` | `globalfitmove.py` | the noise / source plateau searches; only each invocation's FINAL line counts |
| `[NOISE-PIN]`, `[SOURCE-WARM]`, `[WARMSTART-BUILD]`, `[V9-SEED]` | `warmstart/`, driver | the child-run seeding at launch |
| `[GB_WARM_SELF] head builder armed: the buffer holds n of the last K stored rows ...` / `epoch k built ON THE HEAD at the <stage> warm entry from rows a..b (N samples): K components; filtered: kept m of c clusters (... dropped) in T s (read, fit, referee, apply, filter); every rank waited for it` / `head epoch k build FAILED` / `pruned n older epoch(s)` | `warmstart/self_refit.py` (`WarmSelfHeadBuilder`, head) | the head-side self-refit builds (default), their walls (the time every rank waited) and the filtered container's size |
| `[GB_WARM_SELF] builder armed on the saver ...` / `epoch k built from rows a..b (N samples, K components) in T s` / `epoch k build FAILED` / `THE SELF-REFIT BUILD IS HOLDING THE SAVER RANK UP` / `STOPPING THE SELF REFIT` | `warmstart/self_refit.py` (`WarmSelfHook`, saver: `[r<k>/saver]`) | the saver-side builds (`GB_WARM_SELF_AT_ENTRY=0` only) and whether one held a save |
| `[GB_WARM_SELF] rank r adopted epoch k (rows a..b, N samples, K components) into <move>; filtered: m components; split original/full/filtered x/y/z` / `[GB_WARM_SELF <move>] rN this propose used ORIG \| FULL (epoch k) \| FILTERED (epoch k) \| cumulative orig: ... \|\| full: ... \|\| filtered: ...` / `[GB_WARM_SELF] <move>: the original warm start (...), the full self refit and the filtered self refit` | `self_refit.WarmSelfMixer`, every rank; `recipe._warm_self_wrap` at build | adoption per rank, the per-propose container, births viable / accepted (all and cold) per container |
| `[eigen_refresh] <branch> leaf k Gram info matrix at walker w in X s` | `addremovemove.py` | Gram table rebuilds (MBH / EMRI / SOBBH; cadence `{BRANCH}_EIGEN_REFRESH`) and their device |
| `[EMRI_EIGEN] leaf k: step scale adapted on N of 12 axes, max x12.3 (axis 3, ridge), median x1.8 (50 visits, 8 rungs)` (also `[MBH_EIGEN]`, `[SOBBH_EIGEN]`); `... (carried from the persisted table, buffer n of 10 visits)` after a restart; `step-scale adaptation failed` (WARNING) | `addremovemove.py::_log_eigen_adapt` / `_eigen_adapt_table`, every rank for its own table | `{BRANCH}_EIGEN_AXIS_ADAPT=1` only: a table refresh widened some axis by more than 20 % over the information-matrix sigma (ratio adapted / matrix sigma; median over the adapted axes; `ridge` = the table's widest unrailed axis). Silent when nothing moved by 20 % or the switch is off. A failure runs the information-matrix table |
| `[EMRI] inner-move stack: EigenAxisJumpMixMove 1.000 (jump mix 0.10 x 3.0)` (also `[MBH]`, `[SOBBH]`) | `addremovemove.py` (move construction, every rank) | what each source branch proposes with; the `(jump mix f x k)` suffix proves `{BRANCH}_EIGEN_JUMP_MIX` took effect (2026-10-13; `sampling/moves/eigenjumpmix.py`) |
| `[EMRI_LADDER] leaf N: re-anchored K rung(s) to the cold state (walker w rungs a-b, X-Y nats below cold; threshold T + expected tempering drop ...)` (also `[MBH_LADDER]`, `[SOBBH_LADDER]`) | `addremovemove.py::_reanchor_lost_rungs`, each rank for its own walkers (global indices) | the lost-rung guard firing (`{BRANCH}_LADDER_REANCHOR_NATS`, default 200, 0 = off); search and PE. A steady stream at the top rung of a make_ladder ladder means the margin is too small for that source |
| `[MBH] cold->hot rotate event n (stage kind search, every 200 proposals): offset k over W walkers -- N (walker, leaf) cold points copied into rung h` (also `[EMRI]`, `[SOBBH]`) | `addremovemove.py::_maybe_rotate_cold_into_hot`, on the HEAD before the state is sliced (`WalkerFanoutMixin.fanout_prepare_state`) | the search-stage cross-walker exchange (user rulings 2026-10-14): every `{BRANCH}_COLD_TO_HOT_EVERY` proposals (125 = 5 proposes of 25 repeats, "every 5 iterations"; was 200; 0 = off) each walker's cold chain is copied into the hottest sampled rung of walker w + k, k = 1 + event mod (nwalkers - 1), never itself; only in search kinds (source_search, gb_search_seed/1/2), never `full_pe`. In the gb_search stages a source move proposes once per CYCLE, so that is every 5 cycles there and every 5 rounds in source_search. The copied rung is scored at the next leaf visit; a better point swaps down, a lost one is re-anchored by the guard above. The clock restarts at 0 on a resume |
| `[EMRI_LADDER] resume keeps the STORED per-leaf ladders` / `... RESET_ON_RESUME=1: REPLACING ...` + `leaf N ladder reset: old [...] -> new [...]` | `recipe.py::resolve_resumed_ladders` | what a resume did with the stored per-leaf ladders (`{BRANCH}_LADDER_RESET_ON_RESUME`, one-off launch-line knob) |
| `[MBH_BATCH]`, `[EMRI_DIRECT]`, `[SOBBH_FILL]`, `[SOBBH_LOOKUP]` | the source moves | batched MBH window telemetry (a `gen_failed=` field or a `waveform generation FAILED at runtime` WARNING = rows phentax could not generate, scored at the floor; the parameters are in the warning; `outside_box=N (...: rejected at the floor)` = rows whose in-data merger falls outside the locked kept box, scored at the floor since 2026-10-16 so a truncated template can never enter the state -- `MBH_REJECT_OUTSIDE_BOX=0` restores the old truncated scoring; the job 769 phantom is in the deep-dives signature table), EMRI trajectory pool, SOBBH fill / lookup table builds (a build line after startup = a wrong device) |
| `[FANOUT]`, `[FANOUT_DIGEST]` | `communication/`, `walkerfanout.py` | head-vs-rank straggle; the bit-identity digest (`GF_FANOUT_DIGEST`) |
| `[MIDIT_CKPT]` | `midit_checkpoint.py` | checkpoint writes; the resume line is `RESUMING from mid-iteration checkpoint` in `run.py` (no tag) |
| `[GF_MONITOR_RANK]`, `[GF_MONITOR]`, `[STORE-WRITE]`, `[REPLICA_PE]`, `[GALFOR_RATCHET]`, `[REF]`, `[PSD_BATCH]`, `[COARSE_AUDIT]` | monitor rank, saver, recipe, psdmove | as named |
| `different from the current device` (cupy PerformanceWarning), `GPUassert`, `illegal memory access`, `Traceback`, `OutOfMemory`, `MPI-ABORT` | anywhere | the first greps of every health check |

---

## 8. Traps that cost time (navigation and tooling)

- **Editable install + worktrees**: the `deving` env's scikit-build editable
  finder maps every `lisatools.*` import to the MAIN checkout; `PYTHONPATH`
  does not shadow it. The default for a new effort is a separate branch in a
  worktree with its `.wtenv/sitecustomize.py` shim (user rule 2026-09-29);
  working on `dev` in the main checkout was the ruling for the 2026-10
  relaunch effort and applies when the user says so.
- **A `git pull` on the cluster changes a RUNNING job's imports** (editable
  install): pull only with the job stopped, or from a checkout no job imports.
- **Refit on restart**: `GB_FSTAT_FORCE_REFIT=1` (+ `GB_FSTAT_PE_REF=max`,
  d13cd739) is a one-off launch-line knob, deliberately not a launcher export.
- **`python -m unittest ... | tail` loses the verdict AND the exit code**
  (the pipeline's status is `tail`'s). Redirect stderr to a file and echo `$?`.
- **Never run locally** (8 GB laptop): `tests.test_gbspecial_flow` (10-26 GB),
  `tests.test_gb_search_stage_schedule` as a module (SIGKILL), the
  `test_temper_*` SIGKILL suites, the full suite. One python process at a
  time across every session on that machine; thread pools pinned to 1.
- **`_generator.py` runs on import and exits on import.** `build_monitor`
  defaults to IN-PROCESS (`GF_MONITOR_SUBPROCESS=1` for a subprocess); grepping
  a `subprocess.run` in it proves nothing. `GF_MONITOR_TIMEOUT` is inert on the
  in-process path.
- **Two snapshot implementations** (`make_snapshots.sh` and `monitor/snapshot.py`);
  a parity test compares them. Change both.
- **Filtered logs drop the gate families.** A `*_filtered.log` has no
  `[GB_GATE]`, `[V9-STAGE]`, `[GB_IMCONV]`, `[GF_TIMING]`; an absence there
  is evidence of nothing. Read the raw `slurm_stdout_<job>.log`.
- **`gf_monitor_gen.py` has no `--out` flag** (`argv[1]` run dir, `argv[2]`
  output) and caches `gf_arm_<tag>.npz` in the CWD; run it in a clean
  directory. The page silently drops a panel whose sidecar is missing; always
  report its `missing=K` line and every MISSING reason.
- **Launcher harvest**: a config harvest must cover `export VAR=v`,
  `: "${VAR:=d}"` and bare `export VAR`; names carry digits.
- **A resume ignores a changed knob in several places**: `NWALKERS` (the store
  carries it; never change it on a resume), a branch's rung count
  (`resume_ladder_wins`; use `rerung_store_branches.py`), the F-stat peak
  threshold of an already-fitted epoch (delete `fstat_grid_peaks_stacked.npz`
  to re-select), the recipe (section 2.2). `GF_PERSIST_STAGE_START=1` keeps
  stage clocks across a resume and resets them at each boundary.
- **Timestamps**: cluster logs are UTC; a tar's listed mtimes are in the
  laptop's zone; mtimes inside a pulled zip are the COPY time.
- **zsh**: `python -m unittest $MODS` passes the list as ONE argument; use
  `${=MODS}`. `$m[a-z]*` inside a grep pattern is an array subscript.
- **Background Bash from the umbrella root** lands in the wrong CWD: always
  `cd .../LISAanalysistools &&`.
- **A requeued SLURM job re-runs the spooled script**; cancel and resubmit to
  change anything.
- **A fake more capable than the real object tests nothing**, and a big green
  count over fake-based suites says nothing about a constructor. Before
  reporting a change green, revert the line it adds and watch a named test
  fail (`feedback`: mutation must break a test).
- **Test-writing traps**: a test that calls `RecipeStep.note_recipe_step`
  must clear the process-global F-stat peak floor
  (`set_peak_min_F_override(None)` in `tearDown`); a test class that
  `mkdtemp`s needs `addCleanup(rmtree)` and KB-sized fixtures (one suite
  leaked 28 GiB into `/var/folders` and took the laptop disk down); never
  conclude "X is never set" from a grep piped through `head` / `tail`: re-run
  it unbounded over every repo in the stack before a code change rests on a
  negative result (a GPU run was spent on a no-op "fix" that way).

---

## 9. Reading list by task

| I want to ... | read, in order |
|---|---|
| understand what a running job is doing | `gf_last_iteration.py` output -> section 3 -> the `[V9-STAGE]` entry lines of the stage -> [`global-fit-deep-dives.md`](global-fit-deep-dives.md) |
| change a knob for the next launch | the knob's launcher comment block -> who reads it (section 2.3) -> the twin / flagship test that pins it -> `--print-recipe` |
| add or reorder a stage | `run_combined_staged.py` (`search_stage_profiles`, `build_fit`) -> `recipe.py` `Stage` / `SearchStageProfileStep` -> `docs/stock-stages-and-moves.md` -> `tests/test_v9_search_stages.py`, `test_stage_end_rules.py` |
| touch the GB valve / stage end | section 4 -> `_ColdBandLnL.judge`, `_update_search_band_shutoff` -> `tests/test_gb_valve_gain.py`, `test_gb_shutoff_persist.py`, `test_valve_release_at_step_entry.py`, `test_stage_end_rules.py` -> replay on a real store copy before shipping (user rule) |
| touch the F-stat fit | `scripts/fstat_proposal/WALKTHROUGH_fstat_proposal.md`, `FSTAT_FIT_GPU_RUNBOOK.md` -> `sampling/fstat_gridfit.py` (the fit) and `sampling/fstat_proposal.py` (the birth proposal classes, `StackedFStatProposal4D`) -> `docs/superpowers/specs/2026-09-16-parallel-fstat-fit-design.md` -> `tests/test_fstat_*.py` |
| touch MBH / EMRI / SOBBH scoring | `docs/emri-direct-wdm.md`, `docs/sobbh-wdm-lookup.md`, `docs/superpowers/specs/2026-09-29-mbh-batched-windowed-likelihood-design.md` -> `moves/{mbhbatchedmove,emridirectmove,sobbhspecialmove}.py` -> `stock/erebor/source_runtime.py` -> `docs/eigen-info-routes.md` |
| touch the noise moves | `moves/psdmove.py` docstrings (`PSD_BATCH`, `_batched_covariance`) -> `stock/erebor/noise.py` -> `conventions.md` "Noise moves are split" |
| touch ranks / GPUs / fan-out | [`global-fit-launch.md`](global-fit-launch.md) -> `docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md` -> `communication/ranks.py` -> [`multirank-cluster-gates.md`](multirank-cluster-gates.md) |
| make a child run (12mo, ...) | `scripts/flagship/README.md` -> `runs.py` -> `build.py --check` -> the warm-start dry runs on a login node |
| repair or rewind a store | section 2.10 docstrings -> `hdfbackend.py` `_validate_resume_readable` / `add_recipe` -> always on a COPY, job stopped |
| regenerate or change the monitor | `monitor/__main__.py` -> `_short.py` (`build_glance`) -> `_generator.py` by grep only -> `tests/test_monitor_*.py` |
| know the rulings that constrain a design | [`global-fit-deep-dives.md`](global-fit-deep-dives.md) section 7 -> `conventions.md` -> the knob's launcher comment |
