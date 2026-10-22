"""Derive scripts/fstat_proposal/submit_gf_9mo_v9.sh from submit_gf_6mo_v9_4gpu.sh.

The 9-month run (user ruling 2026-10-07): the 1yr run's settings, exactly,
except what Tobs = 270 d itself changes. So the table below is the 1yr
table (derive_1yr_v9_launcher.py) with the Tobs-derived values at 270 d
and the 1yr values everywhere else (GB_NLEAVES_MAX, GB_N_SUBBANDS,
GB_RJ_INMODEL_CHUNK, the 6mo v9 parent for the warm start + noise pin).

Run from the LAT root after ANY edit to the 6mo script (with
derive_1yr_v9_launcher.py), then run
tests/test_submit_scripts_layout.py::NineMonthV9TwinTest (it holds the same
table and cross-checks every export against the 1yr script).

Every replacement is an exact string with an expected match count; anything
else in the file is byte-identical. Re-runnable."""
import os, sys
src = "scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh"
dst = "scripts/fstat_proposal/submit_gf_9mo_v9.sh"
text = open(src).read()

REPL = [
    # ---- naming (store, file names, SLURM) ---------------------------------
    ("#SBATCH --job-name=gf6mo_v9_4gpu     # job name",
     "#SBATCH --job-name=gf9mo_v9          # job name", 1),
    ("/shared/data/global_fit_output/gf6mo_v9_4gpu_%j.log",
     "/shared/data/global_fit_output/gf9mo_v9_%j.log", 1),
    # the in-job LOG MIRROR (cp of the SLURM stdout into the store every 30 s)
    # must name THIS run's log, not the 6mo's (pre-launch check 2026-10-09)
    ("SLURM_LOG=/shared/data/global_fit_output/gf6mo_v9_4gpu_${SLURM_JOB_ID:-manual}.log",
     "SLURM_LOG=/shared/data/global_fit_output/gf9mo_v9_${SLURM_JOB_ID:-manual}.log", 1),
    ("STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/}",
     "STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_9mo_v9/}", 1),
    ("export BASE_FILE_NAME=gf_prod_6mo\n", "export BASE_FILE_NAME=gf_prod_9mo\n", 1),
    # ---- Tobs and the knobs that follow it ---------------------------------
    # 270 d = 23328000 s: WDMSettings.adjust_to_even_bins resolves it to
    # Nf 1440 x Nt 6480 x dt 2.5 exactly (checked 2026-10-07 with the stock
    # wavelet-duration bounds), 1.5x the 6mo Nt; the 1yr grid is Nt 8640.
    ("export TOBS_TARGET=15552000        # 180 d; grid resolves Nf 1440 x Nt 4320 x dt 2.5 (exact factor-2 of 3 mo in Nt)",
     "export TOBS_TARGET=23328000        # 270 d; grid resolves Nf 1440 x Nt 6480 x dt 2.5 (1.5x the 6 mo Nt; the 1yr runs Nt 8640)", 1),
    # the sig-het layer stride keeps the validated 36-h parity: Tobs / 36 h =
    # 60 (3mo) / 120 (6mo) / 180 (9mo) / 240 (1yr); 180 divides Nt = 6480.
    ("export SIGHET_NT_LAYER=120\n", "export SIGHET_NT_LAYER=180\n", 1),
    # ---- the 1yr values, verbatim (user ruling 2026-10-07: "keep all the
    #      accuracy settings for the 1yr run the same") ----------------------
    ("export GB_NLEAVES_MAX=15000        # 6 mo: deeper confusion resolved; 3-mo ran 10000",
     "export GB_NLEAVES_MAX=20000        # 9 mo: the 1yr value (user ruling 2026-10-07, 1yr settings); 6-mo ran 15000, 3-mo 10000", 1),
    ("export GB_N_SUBBANDS=8192   # PER GPU; total = x n_gpus. Slab ~0.5 MB/slot",
     "export GB_N_SUBBANDS=4096   # PER GPU; total = x n_gpus. Slab ~0.75 MB/slot at 9 mo (the 1yr value, user ruling 2026-10-07)", 1),
    ("export GB_RJ_INMODEL_CHUNK=32768  # byte-parity with the 3mo twin's 65536 (6mo cells ~2x bytes); floored to ntemps multiples by the column-atomic staging",
     "export GB_RJ_INMODEL_CHUNK=16384  # the 1yr value (user ruling 2026-10-07, 1yr settings; 9mo cells ~1.5x the 6mo bytes); floored to ntemps multiples by the column-atomic staging", 1),
    # ---- the MBHBs inside a 270 d window + 14 d (user rule 2026-10-07) ----
    # The MBH session's catalogue table (2026-10-03): every catalogue MBHB
    # merging before 367 d is one of the 1yr eleven, so the 9mo set is the
    # subset with t_c < 270 d + 14 d = 284 d. Nothing lands in (270, 284] d:
    # the next mergers are srcs 9 (285.9 d) and 4 (286.4 d), 15.9 / 16.4 d
    # past the end. The 14-d buffer is exported so the rule is on the run
    # record and the t_plunge prior's upper edge follows it; a merger past the
    # data end is modelled by its in-window inspiral, cut at the data end
    # (mbh_window_layers clamps the kept box at the active box; the stock
    # generator lives on the data lattice; tests/test_mbh_batched_move.py::
    # test_telemetry_merger_past_the_data_end).
    ("export MBHB_IDS=2,5,16,18          # t_c 173.3 / 104.7 / 111.4 / 92.0 d",
     "export MBHB_IDS=2,5,7,12,16,18     # t_c 173.3/104.7/263.8/243.8/111.4/92.0 d (MBH session 2026-10-03 table): every catalogue MBHB merging inside 270 d + the 14-d MBH_MERGER_TIME_BUFFER below. None merges in (270, 284] d; the next are srcs 9 (285.9 d) and 4 (286.4 d), 15.9 / 16.4 d past the end, then 0 (300.4), 15 (318.0), 3 (336.8), 10 (369.5)\n"
     "# User rule 2026-10-07: keep any MBHB merging within 2 weeks of the end of the\n"
     "# observation (its in-window inspiral is modelled, cut at the data end by the\n"
     "# batched window's active-box clamp and the stock generator's data lattice).\n"
     "# 14 d replaces the code default of 7 d; the t_plunge prior's upper edge\n"
     "# follows it (obs_end + buffer + t_plunge_pad).\n"
     "export MBH_MERGER_TIME_BUFFER=1209600   # 14 d", 1),
    # ---- the three preflight resolvers name the run's grid -----------------
    ("make_factory(1440, 4320)", "make_factory(1440, 6480)", 2),
    ("Nf 1440 x Nt 4320 at dt 2.5 s", "Nf 1440 x Nt 6480 at dt 2.5 s", 3),
    # ---- the warm-start / noise-pin parent is the 6mo v9 run (as the 1yr's) ----
    ("${STORE_DIR}/warmstart/gf_prod_3mo_v8_10w_refereed.npz}",
     "${STORE_DIR}/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz}", 1),
    ("GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers/gf_prod_3mo_testing.h5}",
     "GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5}", 1),
    ("GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-7776000}",
     "GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-15552000}   # the 6-month parent", 1),
    # the warm-start fit reads the last K WRITTEN cold-chain rows x every walker
    # (user ruling 2026-10-08: "the last 300 samples ... so 1200 total" = 300
    # rows x the 6mo's 4 walkers); the 6mo took 10 rows x 10 walkers from the 3mo
    ("export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-10}\n",
     "export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-300}   # 9mo/1yr (Mike 2026-10-08): the last 300 stored cold-chain rows x 4 walkers = 1,200 samples of the 6mo full_pe (~2.9M leaf rows into the fit; build the npz by hand on a login node if the in-job build is slow)\n", 1),
    # ---- the 9mo / 1yr RECIPE (user ruling 2026-10-07; Tobs-independent, the
    #      same two entries in derive_1yr_v9_launcher.py) ---------------------
    ("export GB_SEARCH_SEED_ITERS=5\n",
     "# ---- THE 9MO / 1YR RECIPE (user ruling 2026-10-07) -------------------------\n"
     "# gb_search_seed (3 iterations) -> gb_search_1 -> gb_search_2 -> full_pe. No\n"
     "# gb_search_3, no replica_pe (STAGE_REPLICA_PE above), no galfor ratchet\n"
     "# (GALFOR_RATCHET stays 0). The seed: the 6mo's, 3 iterations not 5, no\n"
     "# source moves. gb_search_1: its table row (opt SNR 8, phase max, F-stat peak\n"
     "# 8, prior removal only), noise FIXED, NO sobbh/mbh/emri moves. gb_search_2:\n"
     "# its table row (opt SNR 5, no phase max, peak 6.25, prior births AND\n"
     "# deaths), the valves reset at its fresh entry, the sources EVERY iteration,\n"
     "# and ONE psd+galfor search to CONVERGENCE at the END of every cycle (\"only\n"
     "# one noise/galfor proposal ... at the end of the cycle\"): the standalone\n"
     "# noise stage's plateau rule (NOISE_SEARCH_CHECKS flat rounds per walker\n"
     "# within MAXLOGL_TOL), afresh each cycle. Then full_pe, declared as today.\n"
     "# The resolved recipe, no data / no MPI, under this script's exports:\n"
     "#   python scripts/fstat_proposal/run_combined_staged.py --print-recipe\n"
     "export GB_SEARCH_SEED_ITERS=3\n"
     "export GB_SEARCH_STAGES=${GB_SEARCH_STAGES:-1,2}\n"
     "export GB_SEARCH_1_SOURCE_EVERY=${GB_SEARCH_1_SOURCE_EVERY:-1}   # 2026-10-10 ruling: sobbh/mbh/emri every cycle in gb_search_1 too (was 0)\n"
     "export GB_SEARCH_2_SOURCE_EVERY=${GB_SEARCH_2_SOURCE_EVERY:-1}\n"
     "export GB_SEARCH_2_NOISE_MODE=${GB_SEARCH_2_NOISE_MODE:-cycle_end}\n"
     "export GB_SEARCH_2_RESET_VALVES=${GB_SEARCH_2_RESET_VALVES:-1}\n", 1),
    ("export STAGE_REPLICA_PE=${STAGE_REPLICA_PE:-1}\n",
     "export STAGE_REPLICA_PE=${STAGE_REPLICA_PE:-0}   # 9mo/1yr recipe (ruling 2026-10-07): \"No replica pe\" -- gb_search_2 hands over to full_pe\n", 1),
    # ---- the fixed-dim source seed (user ruling 2026-10-08; placed after the
    #      seed by the main session the same day; the same entry in
    #      derive_1yr_v9_launcher.py) ----------------------------------------
    ("export STAGE_SKIP_SOURCE_SEARCH=1\n",
     "# ---- 9MO / 1YR: THE FIXED-DIM SOURCE SEED (user ruling 2026-10-08) ---------\n"
     "# \"since we are seeing the MBHs move quite a bit towards a better likelihood\n"
     "# ... 'fixed-dim source seed' ... the galfor and psd will be fixed just like in\n"
     "# gb search seed and gb search 1. However, it will only run MBHBs, SOBHBs, and\n"
     "# EMRIs, until the maximum likelihood of each walker converges.\" The existing\n"
     "# source_search stage, RUN here (STAGE_SKIP_SOURCE_SEARCH 1 -> 0) and placed\n"
     "# AFTER gb_search_seed (SOURCE_SEARCH_POSITION=after_seed, main session\n"
     "# 2026-10-08: the source maxima are found against a residual with the\n"
     "# warm-started galaxy already subtracted; =first puts it before the seed).\n"
     "# Moves: sobbh_pe / mbh_pe / emri_pe ONLY, from their start points (the 6mo cold chain for SNR > 10 sources, exact truth otherwise: the SOURCE WARM START block below);\n"
     "# psd + galfor FIXED at the start pin, the GB leaves held at the seed's\n"
     "# warm-start births, the VGBs held. Stop: every walker's cold lnL gains no\n"
     "# more than SOURCE_SEARCH_TOL nats over SOURCE_SEARCH_CHECKS consecutive\n"
     "# rounds (a round = one pass of the three moves at their in-model repeats;\n"
     "# MAXLOGL_PER_WALKER=1, the laggard walker decides), at most\n"
     "# SOURCE_SEARCH_MAX_ROUNDS rounds, MAXLOGL_ITERS_PER_STEP (10) rounds per\n"
     "# stored row. MAXLOGL_TOL=20 / NOISE_SEARCH_CHECKS stay the gb_search_2\n"
     "# cycle-end noise slot's.\n"
     "export STAGE_SKIP_SOURCE_SEARCH=${STAGE_SKIP_SOURCE_SEARCH:-0}\n"
     "export SOURCE_SEARCH_POSITION=${SOURCE_SEARCH_POSITION:-after_seed}\n"
     "export SOURCE_SEARCH_CHECKS=${SOURCE_SEARCH_CHECKS:-10}\n"
     "export SOURCE_SEARCH_TOL=${SOURCE_SEARCH_TOL:-10}\n"
     "export SOURCE_SEARCH_MAX_ROUNDS=${SOURCE_SEARCH_MAX_ROUNDS:-200}\n", 1),
    # the galfor frequency prior (user ruling 2026-10-08; 9mo / 1yr only -- the
    # 6mo relaunch keeps the stock box)
    ("export GALFOR_ALPHA_MAX=20.0\n",
     "export GALFOR_ALPHA_MAX=20.0\n"
     "# ---- 9MO / 1YR: GALFOR FREQUENCY PRIOR (user ruling 2026-10-08) -----------\n"
     "# \"adjust the frequency parameters (fk, f1, f2) in the foreground model ...\n"
     "# in its prior ... to go from 1e-4 to 1e-2\": the three frequency columns\n"
     "# share one box, 0.1-10 mHz (stock: fk 0.8-10 mHz, f_1 / f_2 10 uHz-10 mHz).\n"
     "# One resolver (galfor_prior_ranges) feeds the prior, the start-pin window\n"
     "# and the noise-pin refusal, so the GALFOR_START_PARAMS pin (fk 2.53 mHz,\n"
     "# f_1 10 mHz, f_2 1.41 mHz) is inside the box. Fresh store only: a chain\n"
     "# outside the box would price at log_prior = -inf on resume.\n"
     "# 2026-10-09 (Mike): \"adjust the foreground prior for f_k and f_2 to go from\n"
     "# 0.8 mHz to 10 mHz. Keep f_1 as is.\" -- per-column boxes, in Hz; f_1 keeps the\n"
     "# 2026-10-08 box. The start pin (fk 2.53 mHz, f_1 10 mHz, f_2 1.41 mHz) is inside.\n"
     "export GALFOR_FK_PRIOR=${GALFOR_FK_PRIOR:-0.8e-3,1e-2}\n"
     "export GALFOR_F1_PRIOR=${GALFOR_F1_PRIOR:-1e-4,1e-2}\n"
     "export GALFOR_F2_PRIOR=${GALFOR_F2_PRIOR:-0.8e-3,1e-2}\n", 1),
    # the source warm start from the 6mo cold chain (user ruling 2026-10-08;
    # the same entry in both derive tables)
    ("export MBH_START_FACTOR=0.0\nexport EMRI_START_FACTOR=0.0\nexport SOBBH_START_FACTOR=0.0\nexport VGB_START_FACTOR=0.0\n",
     "export MBH_START_FACTOR=0.0\nexport EMRI_START_FACTOR=0.0\nexport SOBBH_START_FACTOR=0.0\nexport VGB_START_FACTOR=0.0\n"
     "# ---- 9MO / 1YR: SOURCE WARM START from the 6mo cold chain (user ruling\n"
     "# 2026-10-08) -------------------------------------------------------------\n"
     "# \"start the 9mo sources from the 6mo cold chain's final positions instead of\n"
     "# catalogue truth ... for any source over SNR 10 at 6 mo. Otherwise start how\n"
     "# we did at 6mo [exact truth] for sources under SNR 10 (computed at six\n"
     "# months)\". lisatools.globalfit.warmstart.sources reads the seed store's last\n"
     "# written cold-chain row (sub_backend/<branch>/chain + h_h; the running backup\n"
     "# copy if the primary is torn), maps leaves by CATALOGUE ID (the seed's own id\n"
     "# lists below, NOT this run's: MBHB 7 and 12 were never fitted at 6mo and\n"
     "# start at truth), takes SNR_6mo = median over the cold walkers of sqrt(h_h),\n"
     "# and for SNR > SOURCE_WARM_START_SNR_MIN copies walker w -> w and the whole\n"
     "# 8-rung ladder (the rung counts match). Below the threshold, or absent from\n"
     "# the seed: exact truth as above. A NaN record REFUSES the launch\n"
     "# (SOURCE_WARM_START_SNR_UNKNOWN=truth|warm overrides); a start outside this\n"
     "# run's prior refuses it. Fresh start only (a resume keeps its chain). DRY RUN\n"
     "# on the cluster BEFORE launching, once per branch (prints each id's 6mo SNR\n"
     "# and the warm / truth decision):\n"
     "#   python -m lisatools.globalfit.warmstart.sources --store $GF_SEED_STORE \\\n"
     "#       --branch mbh --ids 2,5,16,18\n"
     "# OFF BY DEFAULT SINCE 2026-10-22 (Mike: the non-GB sources \"always start at\n"
     "# the catalog injected values rather than the last run's values\"): every\n"
     "# source leaf starts at exact truth (*_START_FACTOR=0 above). Opt back in for\n"
     "# one launch with SOURCE_WARM_START_STORE=$GF_SEED_STORE on the line; to push a\n"
     "# stopped run's sources back to the injection use\n"
     "# scripts/flagship/relaunch_sources_at_injection.sh.\n"
     "export SOURCE_WARM_START_STORE=${SOURCE_WARM_START_STORE-}\n"
     "export SOURCE_WARM_START_SNR_MIN=${SOURCE_WARM_START_SNR_MIN:-10}\n"
     "export MBH_SOURCE_WARM_START_IDS=${MBH_SOURCE_WARM_START_IDS:-2,5,16,18}\n"
     "export EMRI_SOURCE_WARM_START_IDS=${EMRI_SOURCE_WARM_START_IDS:-0,1,2,3,4,5,6,7}\n"
     "export SOBBH_SOURCE_WARM_START_IDS=${SOBBH_SOURCE_WARM_START_IDS:-0,1,2,3,4,5}\n", 1),
    # the PSD reference fit (user rulings 2026-09-26 [3mo] and 2026-10-04 [unequal
    # arm only]; the 6mo line's own comment: "DELETE THIS LINE for a fresh store")
    ("export MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=0\n",
     "# ---- 9MO / 1YR: the reference fit is the UNEQUAL-ARM fit (2026-10-08) --------\n"
     "# The =0 pin above exists ONLY so the resumed 6mo store keeps the equal-arm\n"
     "# pair its chain was started with (coarse_fiducial_digest). A fresh store has\n"
     "# no such debt: =1 (the code default when unset; the 3mo v9 dropped the line,\n"
     "# ruling 2026-09-26) makes general.psd_injection -- the reference / 'truth'\n"
     "# pair the pages and diagnostics compare against, NOT the noise model, which\n"
     "# is UNEQUAL_ARM=1 either way -- the unequal-arm fit 1.500004e-11 /\n"
     "# 3.000107e-15. Ruling 2026-10-04: no default may rely on an equal-arm path.\n"
     "export MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=${MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM:-1}\n", 1),
    # the stage-B F-stat sky cap (pre-launch check 2026-10-09; Mike: "if you think
    # ... for the 1yr run, then yes do that")
    ("export FSTAT_STAGEB_NSKY_MAX=0       # 0 = no node cap (user ruling)\n",
     "# ---- 9MO / 1YR: cap the stage-B sky grid at the 6mo's 8-mHz node count ------\n"
     "# The adaptive sky grid asks for ~(f0 Tobs v/c)^2 nodes per peak: at 6 mo that\n"
     "# is 64 (< 4 mHz) / 256 (8 mHz) / 1024 (15 mHz); at 9 mo 128 / 512 / 2048 and\n"
     "# at 1 yr 256 / 1024 / 4096, on top of the fdot axis (13 -> 28 -> 48 nodes at\n"
     "# 8 mHz) and the f0 axis (79 -> 96). The 6mo stacked grid was ~12 GB raw and\n"
     "# 24-35 GB resident per GPU, so uncapped the 9mo grid lands at 2-3x that and\n"
     "# the 1yr at ~8x (the 1yr v8 OOMed on its first F-stat search). 512 never\n"
     "# touches a peak below 8 mHz at 9 mo (their natural counts are <= 512) and\n"
     "# bounds the > 8 mHz tail; the host-mapped float32 grid (TODO) is the real\n"
     "# fix, and at 1 yr the fdot axis alone needs it. FSTAT_STAGEB_NSKY_MAX=0 on\n"
     "# the line restores the uncapped grid.\n"
     "export FSTAT_STAGEB_NSKY_MAX=${FSTAT_STAGEB_NSKY_MAX:-512}\n", 1),
    # the stage-B fit's per-group transient (pre-launch check 2026-10-09; Mike:
    # "adjust the FSTAT a little bit if you can for memory ... no changes that
    # sacrifice needed accuracy"): the grouping only batches the evaluation
    ("export FSTAT_STAGEB_GROUP_MAX_GB=2.0\n",
     "export FSTAT_STAGEB_GROUP_MAX_GB=${FSTAT_STAGEB_GROUP_MAX_GB:-1.0}   # 9mo/1yr: halve the stage-B per-group transient (batching only, no grid change; 2.0 at 6mo)\n", 1),
    # the 6mo v9 run overlaid on this run's monitor page (user request
    # 2026-10-12; the same entry in derive_1yr_v9_launcher.py; the 6mo itself
    # never adds itself)
    ("export GF_MONITOR_TIMEOUT=1800\n# Requires MOJITO_INFO_PATH, exported above.\n",
     "export GF_MONITOR_TIMEOUT=1800\n# Requires MOJITO_INFO_PATH, exported above.\n"
     "# ---- 9MO / 1YR: THE 6MO RUN ON THIS RUN'S MONITOR PAGE (user request 2026-10-12)\n"
     "# \"add a second run to the same html only in specific places where it is\n"
     "# reasonable visually to compare between runs ... plot the phase maximized\n"
     "# overlap plots, a comparison of the final psd and foreground measurements\n"
     "# (final for the previous run and current for the ongoing run)\". The page stays\n"
     "# this run's; each run directory listed here (':'-joined) is read at ITS last\n"
     "# stored row and drawn onto the overlap CDF / survival, recovered-vs-SNR and\n"
     "# recovered-vs-frequency figures (matched against its own gb_truth_*.npz) and\n"
     "# the noise + foreground posteriors and curves. A missing dir / store / truth\n"
     "# set costs only its overlay (a caption note + a MISSING line). The monitor rank\n"
     "# inherits it; the CLI twin is python -m lisatools.globalfit.monitor --add-info.\n"
     "# GF_MONITOR_ADD_INFO=none on the launch line drops it.\n"
     "export GF_MONITOR_ADD_INFO=${GF_MONITOR_ADD_INFO:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu}\n", 1),
]
for old, new, n in REPL:
    c = text.count(old)
    if c != n:
        sys.exit(f"REFUSING: {old!r} matches {c} times, expected {n}")
    text = text.replace(old, new)

HEADER = """#!/bin/bash
# ============================================================================
# PRODUCTION global fit -- 9mo_v9 (Tobs = 270 d). DERIVED 2026-10-07 FROM
# submit_gf_6mo_v9_4gpu.sh (dev, the 6mo production launcher on the V9-30
# defaults) BY scripts/fstat_proposal/derive_9mo_v9_launcher.py (exact-string
# replacement, re-run it after every 6mo edit): everything not listed here is
# BYTE-IDENTICAL to the 6mo script, and tests/test_submit_scripts_layout.py::
# NineMonthV9TwinTest refuses any other drift AND checks every export against
# submit_gf_1yr_v9.sh. User ruling 2026-10-07: "run a 9mo run instead of
# 1yr ... keep all the accuracy settings for the 1yr run the same ... keep
# everything the same as the 1 yr run except for any specific necessary
# changes due to the observation". No "_4gpu" in the name: NGPUS picks the
# layout (default 4 since V9-30).
#
#   9MO-1. NAMING: --job-name gf9mo_v9, --output gf9mo_v9_%j.log,
#          STORE_DIR gf_prod_9mo_v9/, BASE_FILE_NAME gf_prod_9mo.
#   9MO-2. TOBS_TARGET 15552000 -> 23328000 (270 d; Nf 1440 x Nt 6480 x dt
#          2.5, 1.5x the 6mo Nt) and the ONE knob that follows Tobs itself:
#          SIGHET_NT_LAYER 120 -> 180 (the 36-h stride parity; 180 divides
#          6480). GB_NLEAVES_MAX 20000, GB_N_SUBBANDS 4096 per GPU and
#          GB_RJ_INMODEL_CHUNK 16384 are the 1YR VALUES, verbatim (the
#          ruling), not re-derived for 270 d: all three are capacity /
#          memory knobs, and the 1yr sizing has more headroom at 9 mo
#          (slab ~0.75 MB/slot, sig-het stash product 4096 x 178 = 0.73e6
#          against the 6mo-safe 0.97e6).
#   9MO-3. MBHB_IDS: the catalogue MBHBs merging inside 270 d + 14 d
#          (2,5,7,12,16,18 -- all merge BEFORE the end; none lands in
#          (270, 284] d, the next are srcs 9 / 4 at 285.9 / 286.4 d) and
#          MBH_MERGER_TIME_BUFFER=1209600 (14 d, user rule 2026-10-07: keep
#          a merger up to two weeks past the end, modelled by its in-window
#          inspiral cut at the data end -- supported: the batched window
#          clamps its kept box at the data's active box and the stock
#          generator lives on the data lattice). EMRI_IDS / SOBHB_IDS
#          unchanged; EMRI 3 (plunge 256 d) takes the plunge path inside
#          this window, EMRI 0 (347 d) does not.
#   9MO-4. The MBH / EMRI / SOBBH preflight resolvers name the 9mo grid
#          (make_factory(1440, 6480); comments). The lookup tables are the
#          SAME canonical files (Nf and dt only), built once in the new
#          STORE_DIR on the first launch.
#   9MO-5. Warm start + noise pin parent = the 6mo v9 run, exactly as the
#          1yr (GF_SEED_STORE gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5,
#          GB_WARM_START_SOURCE_TOBS 15552000, refereed npz
#          gf_prod_6mo_v9_4gpu_refereed.npz -- auto-built from the parent
#          store on the first launch when missing). GB_WARM_START_LAST_K
#          10 -> 300 (user ruling 2026-10-08): the fit takes the last 300
#          written cold-chain rows x 4 walkers = 1,200 samples of the 6mo
#          full_pe (the 6mo itself took 10 rows x 10 walkers of the 3mo).
#   9MO-6. THE RECIPE (user ruling 2026-10-07, the SAME as the 1yr's 1YR-6;
#          not a Tobs change -- the running 6mo store's recipe is fixed, so
#          it lands in the 9mo/1yr only): gb_search_seed x3 -> gb_search_1 ->
#          gb_search_2 -> full_pe. GB_SEARCH_SEED_ITERS 5 -> 3;
#          GB_SEARCH_STAGES=1,2 (no gb_search_3); GB_SEARCH_1_SOURCE_EVERY=1
#          (sobbh/mbh/emri every cycle in gb_search_1 -- user ruling
#          2026-10-10, was 0 at the 9mo's first launch); GB_SEARCH_2_SOURCE_EVERY=1
#          (every iteration); GB_SEARCH_2_NOISE_MODE=cycle_end (ONE psd+galfor
#          search to convergence at the END of each gb_search_2 cycle);
#          GB_SEARCH_2_RESET_VALVES=1 (fresh entry: per-(walker, band) lnL max
#          re-learned + per-band barren valve revived); STAGE_REPLICA_PE 1 -> 0
#          ("No replica pe"). GALFOR_RATCHET stays 0 ("no ratcheting").
#          full_pe's declarations unchanged. `run_combined_staged.py
#          --print-recipe` under these exports prints the resolved recipe.
#   9MO-7. THE FIXED-DIM SOURCE SEED (user ruling 2026-10-08, the SAME as the
#          1yr's 1YR-7): the existing source_search stage runs, AFTER the seed
#          (main-session placement, same day): gb_search_seed x3 ->
#          source_search -> gb_search_1 -> gb_search_2 -> full_pe.
#          STAGE_SKIP_SOURCE_SEARCH 1 -> 0; SOURCE_SEARCH_POSITION=after_seed
#          (the sources' maxima found against a residual with the warm-started
#          galaxy already subtracted); sobbh_pe / mbh_pe / emri_pe ONLY at their
#          in-model repeats (25), from exact truth, psd + galfor FIXED at the
#          pin, GB leaves and VGBs held. Stop: every walker's cold lnL gains <=
#          SOURCE_SEARCH_TOL=10 nats over SOURCE_SEARCH_CHECKS=10 consecutive
#          rounds (the laggard walker decides), ceiling SOURCE_SEARCH_MAX_ROUNDS
#          =200; 10 rounds (MAXLOGL_ITERS_PER_STEP) per stored row. The globals
#          MAXLOGL_TOL=20 / NOISE_SEARCH_CHECKS=5 stay the cycle-end noise
#          slot's.
#   9MO-8. GALFOR FREQUENCY PRIOR (user ruling 2026-10-08, shared by the 9mo and
#          1yr), revised 2026-10-09: fk and f_2 0.8e-3..1e-2 Hz, f_1 1e-4..1e-2 Hz
#          (GALFOR_FK_PRIOR / GALFOR_F1_PRIOR / GALFOR_F2_PRIOR; stock: fk
#          0.8e-3..1e-2, f_1 / f_2 1e-5..1e-2). One resolver feeds the prior, the start-pin window
#          and the noise-pin refusal; the GALFOR_START_PARAMS pin is inside
#          the box. The 6mo relaunch keeps the stock box.
#   9MO-9. MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM 0 -> 1 (2026-10-08): the reference
#          / 'truth' PSD pair is the unequal-arm fit on this fresh store (the
#          6mo's =0 pin only keeps its own resumable identity; the 3mo v9 dropped
#          it on 2026-09-26; ruling 2026-10-04: unequal arm only).
#   9MO-10. GF_MONITOR_ADD_INFO defaults to the 6mo v9 run directory
#          /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu (user request
#          2026-10-12, the SAME entry as the 1yr's 1YR-10): that run, read at
#          its last stored row, is overlaid on this run's monitor page on the
#          phase-maximised overlap figures and the noise / foreground
#          posteriors and curves only (monitor/_add_info.py). Not a run knob:
#          the sampler never reads it.
#
# EVERYTHING ELSE = the 1yr script = the 6mo script (NineMonthV9TwinTest
# diffs the resolved exports against submit_gf_1yr_v9.sh: only TOBS_TARGET,
# SIGHET_NT_LAYER, BASE_FILE_NAME, MBHB_IDS and MBH_MERGER_TIME_BUFFER may
# differ). The waveform knobs are the 1yr answers (no export changes): EMRI
# direct, SOBBH lookup (EVAL_DT 43200), MBH batched (decimation 2, 90 d
# window), EDGE_CROP_WAVELETS 60 -- verified at Nt 6480 (2026-10-07, by
# running the build guard): sig-het taper ceil(0.005 x 6480) = 33 + 8
# margin = 41 <= 60 (19 layers spare; the 1yr has 8), the data taper is 2
# fixed wavelets (auto crop 20, subsumed), EMRI pixel_edge 8 <= 60; the
# crop costs 1.85 % of the data.
#
# LAUNCH (from the LAT root; the V9-30 defaults ARE the production line --
# NGPUS=4, MIDIT_CHECKPOINT=0, MBH_NTEMPS=8, EMRI_NTEMPS=8,
# EMRI_TRAJ_WORKERS=8, MBH_WINDOW_DECIMATE=2, gram / fisher eigen tables,
# source cross-checks off):
#   ./scripts/fstat_proposal/submit_gf_9mo_v9.sh
# (a FRESH store: no CLOCK_START, no re-rung. Requires the 6mo v9 store for
#  the seed + warm start. GB_FSTAT_FORCE_REFIT=1 / GB_FSTAT_PE_REF=max stay
#  one-off line knobs, never defaults.)
#
# ---- the 6mo v9 header follows verbatim ------------------------------------
"""
assert text.startswith("#!/bin/bash\n")
text = HEADER + text[len("#!/bin/bash\n"):]
open(dst, "w").write(text)
os.chmod(dst, os.stat(src).st_mode)
print("wrote", dst, len(text.splitlines()), "lines")
