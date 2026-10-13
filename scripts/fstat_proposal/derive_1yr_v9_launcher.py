"""Derive scripts/fstat_proposal/submit_gf_1yr_v9.sh from submit_gf_6mo_v9_4gpu.sh.

Run from the LAT root after ANY edit to the 6mo script, then run
tests/test_submit_scripts_layout.py::OneYearV9TwinTest (it holds the same table).

Every replacement is an exact string with an expected match count; anything
else in the file is byte-identical. Re-runnable."""
import os, sys
src = "scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh"
dst = "scripts/fstat_proposal/submit_gf_1yr_v9.sh"
text = open(src).read()

REPL = [
    # ---- naming (store, file names, SLURM) ---------------------------------
    ("#SBATCH --job-name=gf6mo_v9_4gpu     # job name",
     "#SBATCH --job-name=gf1yr_v9          # job name", 1),
    ("/shared/data/global_fit_output/gf6mo_v9_4gpu_%j.log",
     "/shared/data/global_fit_output/gf1yr_v9_%j.log", 1),
    # the in-job LOG MIRROR (cp of the SLURM stdout into the store every 30 s)
    # must name THIS run's log, not the 6mo's (pre-launch check 2026-10-09)
    ("SLURM_LOG=/shared/data/global_fit_output/gf6mo_v9_4gpu_${SLURM_JOB_ID:-manual}.log",
     "SLURM_LOG=/shared/data/global_fit_output/gf1yr_v9_${SLURM_JOB_ID:-manual}.log", 1),
    ("STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/}",
     "STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_1yr_v9/}", 1),
    ("export BASE_FILE_NAME=gf_prod_6mo\n", "export BASE_FILE_NAME=gf_prod_1yr\n", 1),
    # ---- Tobs and the knobs that scale with it (the 6mo-v8 -> 1yr-v8 precedent) ----
    ("export TOBS_TARGET=15552000        # 180 d; grid resolves Nf 1440 x Nt 4320 x dt 2.5 (exact factor-2 of 3 mo in Nt)",
     "export TOBS_TARGET=31104000        # 360 d; grid resolves Nf 1440 x Nt 8640 x dt 2.5 (exact factor-2 of 6 mo in Nt, as 6 mo was of 3 mo)", 1),
    ("export SIGHET_NT_LAYER=120\n", "export SIGHET_NT_LAYER=240\n", 1),
    ("export GB_NLEAVES_MAX=15000        # 6 mo: deeper confusion resolved; 3-mo ran 10000",
     "export GB_NLEAVES_MAX=20000        # 1 yr: deeper confusion resolved again; 6-mo ran 15000, 3-mo 10000", 1),
    ("export GB_N_SUBBANDS=8192   # PER GPU; total = x n_gpus. Slab ~0.5 MB/slot",
     "export GB_N_SUBBANDS=4096   # PER GPU; total = x n_gpus. Slab ~1.0 MB/slot at 1 yr", 1),
    ("export GB_RJ_INMODEL_CHUNK=32768  # byte-parity with the 3mo twin's 65536 (6mo cells ~2x bytes); floored to ntemps multiples by the column-atomic staging",
     "export GB_RJ_INMODEL_CHUNK=16384  # byte-parity with the 6mo 32768 (1yr cells ~2x bytes); floored to ntemps multiples by the column-atomic staging", 1),
    # ---- the sources inside a 360 d window (MBH session, 2026-10-03) --------
    ("export MBHB_IDS=2,5,16,18          # t_c 173.3 / 104.7 / 111.4 / 92.0 d",
     "export MBHB_IDS=0,2,3,4,5,7,9,12,15,16,18   # t_c 300.4/173.3/336.8/286.4/104.7/263.8/285.9/243.8/318.0/111.4/92.0 d (MBH session 2026-10-03). src 10 (2.8e6 Msun, SNR~1963) merges at 369.5 d, 9.5 d past the window, so the 7-d MBH_MERGER_TIME_BUFFER leaves it out; to model its in-window inspiral add 10 here and export MBH_MERGER_TIME_BUFFER=1209600", 1),
    # ---- the three preflight resolvers name the run's grid -----------------
    ("make_factory(1440, 4320)", "make_factory(1440, 8640)", 2),
    ("Nf 1440 x Nt 4320 at dt 2.5 s", "Nf 1440 x Nt 8640 at dt 2.5 s", 3),
    # ---- the warm-start / noise-pin parent is the 6mo v9 run (as 6mo's was the 3mo) ----
    ("${STORE_DIR}/warmstart/gf_prod_3mo_v8_10w_refereed.npz}",
     "${STORE_DIR}/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz}", 1),
    ("GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers/gf_prod_3mo_testing.h5}",
     "GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5}", 1),
    ("GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-7776000}",
     "GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-15552000}   # the 6-month parent", 1),
    # the warm-start fit reads the last K WRITTEN cold-chain rows x every walker
    # (user ruling 2026-10-08 for the 9mo, shared so the 1yr composes the same:
    # "the last 300 samples ... so 1200 total" = 300 rows x the 6mo's 4 walkers)
    ("export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-10}\n",
     "export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-300}   # 9mo/1yr (Mike 2026-10-08): the last 300 stored cold-chain rows x 4 walkers = 1,200 samples of the 6mo full_pe (~2.9M leaf rows into the fit; build the npz by hand on a login node if the in-job build is slow)\n", 1),
    # ---- the 9mo / 1yr RECIPE (user ruling 2026-10-07; Tobs-independent, the
    #      same two entries in derive_9mo_v9_launcher.py) ---------------------
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
    #      derive_9mo_v9_launcher.py) ----------------------------------------
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
     "# SOURCE_WARM_START_STORE= (explicitly empty) turns it off.\n"
     "export SOURCE_WARM_START_STORE=${SOURCE_WARM_START_STORE-${GF_SEED_STORE}}\n"
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
    # 2026-10-12; the same entry in derive_9mo_v9_launcher.py; the 6mo itself
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
# PRODUCTION global fit -- 1yr_v9 (Tobs = 360 d). DERIVED 2026-10-03 FROM
# submit_gf_6mo_v9_4gpu.sh (dev 93f22bce, the 6mo production relaunch; last
# re-derived 2026-10-07 on the V9-30 defaults) BY
# scripts/fstat_proposal/derive_1yr_v9_launcher.py (exact-string replacement,
# re-run it after every 6mo edit): everything not listed here is
# BYTE-IDENTICAL to the 6mo script, and tests/test_submit_scripts_layout.py::
# OneYearV9TwinTest refuses any other drift. User ruling 2026-10-03: "identical
# runs with the exception of changing the usual Tobs changes ... the most
# minimal amount of changes needed for the waveforms and other areas due to
# 12 mo instead of 6". No "_4gpu" in the name: NGPUS picks the layout.
#
#   1YR-1. NAMING: --job-name gf1yr_v9, --output gf1yr_v9_%j.log,
#          STORE_DIR gf_prod_1yr_v9/, BASE_FILE_NAME gf_prod_1yr.
#   1YR-2. TOBS_TARGET 15552000 -> 31104000 (Nf 1440 x Nt 8640 x dt 2.5) and
#          the knobs that scale with it, exactly as 6mo-v8 -> 1yr-v8 did:
#          SIGHET_NT_LAYER 120 -> 240, GB_NLEAVES_MAX 15000 -> 20000,
#          GB_N_SUBBANDS 8192 -> 4096 per GPU (slot bytes double),
#          GB_RJ_INMODEL_CHUNK 32768 -> 16384 (byte parity).
#   1YR-3. MBHB_IDS: the 11 catalogue MBHBs merging inside 360 d + 7 d
#          (MBH session 2026-10-03); src 10 merges 9.5 d past the window and
#          is left out (see the MBHB_IDS line). EMRI_IDS / SOBHB_IDS unchanged.
#   1YR-4. The MBH / EMRI / SOBBH preflight resolvers name the 1yr grid
#          (make_factory(1440, 8640); comments). The lookup table is the SAME
#          canonical file (it depends on Nf and dt only), built once in the
#          new STORE_DIR on the first launch.
#   1YR-5. Warm start + noise pin parent = the 6mo v9 run
#          (GF_SEED_STORE gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5,
#          GB_WARM_START_SOURCE_TOBS 15552000, refereed npz
#          gf_prod_6mo_v9_4gpu_refereed.npz -- auto-built from the parent store
#          on the first launch when missing, as the 6mo's was from the 3mo).
#          GB_WARM_START_LAST_K 10 -> 300 (user ruling 2026-10-08, shared with
#          the 9mo): the last 300 written cold-chain rows x 4 walkers = 1,200
#          samples of the 6mo full_pe.
#   1YR-6. THE RECIPE (user ruling 2026-10-07, shared with the 9mo; not a Tobs
#          change -- the running 6mo store's recipe is fixed, so it lands
#          here only): gb_search_seed x3 -> gb_search_1 -> gb_search_2 ->
#          full_pe. GB_SEARCH_SEED_ITERS 5 -> 3; GB_SEARCH_STAGES=1,2 (no
#          gb_search_3); GB_SEARCH_1_SOURCE_EVERY=1 (sobbh/mbh/emri every
#          cycle in gb_search_1 -- user ruling 2026-10-10, was 0);
#          GB_SEARCH_2_SOURCE_EVERY=1 (every iteration);
#          GB_SEARCH_2_NOISE_MODE=cycle_end (ONE psd+galfor search to
#          convergence at the END of each gb_search_2 cycle: "the galfor and
#          psd sampling should be required to converge before moving on ...
#          only one noise/galfor proposal ... at the end of the cycle");
#          GB_SEARCH_2_RESET_VALVES=1 (fresh entry: per-(walker, band) lnL max
#          re-learned + per-band barren valve revived; the per-walker window
#          is released at every new step anyway); STAGE_REPLICA_PE 1 -> 0 ("No
#          replica pe"). GALFOR_RATCHET stays 0 ("no ratcheting"). full_pe's
#          declarations (peak floor 6.25, PE repeats, RJ flip 0.1) unchanged.
#          `run_combined_staged.py --print-recipe` under these exports prints it.
#   1YR-7. THE FIXED-DIM SOURCE SEED (user ruling 2026-10-08, shared with the
#          9mo's 9MO-7): the existing source_search stage runs, AFTER the seed
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
#   1YR-8. GALFOR FREQUENCY PRIOR (user ruling 2026-10-08, shared by the 9mo and
#          1yr), revised 2026-10-09: fk and f_2 0.8e-3..1e-2 Hz, f_1 1e-4..1e-2 Hz
#          (GALFOR_FK_PRIOR / GALFOR_F1_PRIOR / GALFOR_F2_PRIOR; stock: fk
#          0.8e-3..1e-2, f_1 / f_2 1e-5..1e-2). One resolver feeds the prior, the start-pin window
#          and the noise-pin refusal; the GALFOR_START_PARAMS pin is inside
#          the box. The 6mo relaunch keeps the stock box.
#   1YR-9. MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM 0 -> 1 (2026-10-08): the reference
#          / 'truth' PSD pair is the unequal-arm fit on this fresh store (the
#          6mo's =0 pin only keeps its own resumable identity; the 3mo v9 dropped
#          it on 2026-09-26; ruling 2026-10-04: unequal arm only).
#   1YR-10. GF_MONITOR_ADD_INFO defaults to the 6mo v9 run directory
#          /shared/data/global_fit_output/gf_prod_6mo_v9_4gpu (user request
#          2026-10-12, shared with the 9mo's 9MO-10): that run, read at its
#          last stored row, is overlaid on this run's monitor page on the
#          phase-maximised overlap figures and the noise / foreground
#          posteriors and curves only (monitor/_add_info.py). Not a run knob:
#          the sampler never reads it.
#
# WAVEFORM SETTINGS AT 1 YR (all three windows, 2026-10-03): NO export line
# changes. EMRI direct: same table; EMRIs 0 (347 d) and 3 (256 d) plunge INSIDE
# the window and take the plunge path (fdot handoff + dense plunge chunk +
# 120 s stop taper, built alone, ~100-175 ms/row vs ~21 ms batched), validated
# on the laptop only -- watch "[EMRI_DIRECT] F fallbacks" = 0 and the
# every-10th cross-check. SOBBH lookup: same table, EVAL_DT 43200 holds
# (phase-tracer error is local), 0.036 s per 8-row call at 360 d (chunked
# 3.40 s). MBH batched: window knobs unchanged, decimation 2 safe for all 11,
# ~45 s per mbh_pe per rank (11 leaves). The exact SOBBH cross-check costs ~2x
# the 6mo one: put SOBBH_CHECK_LL_EVERY=10 on the line only if the 6mo run's
# first check stayed under ~10 % of its SOBBH leg when doubled.
#
# NOT carried (deliberately, "identical runs"): the 2026-09-08 1yr F-stat grid
# reduction (FSTAT_COMB_NSKY_MAX 256, FSTAT_N_ALPHA/N_SINDELTA 4) -- the 1yr v8
# 4-GPU script did not carry it either; revisit if the first F-stat fit's wall
# or dev0 memory says so. GALFOR_START_PARAMS stays the offline 3mo estimate
# (6mo ruling 2026-09-24 "just for now").
#
# LAUNCH (from the LAT root). Since 2026-10-07 (V9-30) the 6mo production
# line's knobs are the launcher DEFAULTS -- NGPUS=4, MIDIT_CHECKPOINT=0,
# MBH_NTEMPS=8, EMRI_NTEMPS=8, EMRI_TRAJ_WORKERS=8, MBH_WINDOW_DECIMATE=2, the
# gram / fisher eigen tables, the source cross-checks off -- so the line is:
#   ./scripts/fstat_proposal/submit_gf_1yr_v9.sh
# (a FRESH store: no CLOCK_START, no re-rung -- MBH_NTEMPS / EMRI_NTEMPS build
#  the 8-rung ladders the 6mo store was re-rung to. GB_FSTAT_FORCE_REFIT=1 /
#  GB_FSTAT_PE_REF=max stay one-off line knobs, never defaults.)
#
# ---- the 6mo v9 header follows verbatim ------------------------------------
"""
assert text.startswith("#!/bin/bash\n")
text = HEADER + text[len("#!/bin/bash\n"):]
open(dst, "w").write(text)
os.chmod(dst, os.stat(src).st_mode)
print("wrote", dst, len(text.splitlines()), "lines")
