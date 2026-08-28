#!/bin/bash
# ============================================================================
# 6-MONTH SOURCE-BRANCH TIMING PROBE -- MBH + EMRI + SOBBH ONLY
# (user request 2026-08-24: "test a 6 mo run version with just MBHs, EMRIs,
#  and SOBHBs to get timing information. mojito data (No GBs, no VGBs),
#  injection psd information. 2 GPUs to make sure multigpu is working.")
#
# Vehicle: the stock ``full_year_combined`` variant -- catalogue-driven
# multi-leaf MBH+EMRI+SOBBH on mojito L1 data with NO gb / vgb / psd /
# galfor branches and a FIXED sensitivity fitted to the mojito NOISE
# brick's tabulated estimates (= the injection PSD; likelihood is
# source-only -1/2 <r|r>). This is the heavily-tested full-year
# configuration, re-gridded to 6 months via TOBS_TARGET (env wins over
# the variant's defaults; EDGE_CROP_WAVELETS has been env-wired here
# since 2026-08-19).
#
# WHAT THIS PROBE ANSWERS (campaign gates C1-C4, docs/6mo-run-prep.md):
#   * per-move walls at 6 mo for mbh_pe / emri_pe / sobbh_pe
#     ([GF_MOVE_TIMING] lines -- read wall_s, rss, gpu_used per move);
#   * multi-GPU: both devices active (gpu_util CSV + [GF_MOVE_TIMING]
#     gpu_pool_mb), no cross-device errors from the all_sources hardening
#     (jax.default_device + cupy ctx + per-device orbits);
#   * the 6-mo detectability census for free: all 8 EMRIs + all 6 SOBHBs
#     are in -- sub-threshold branches show prior-like posteriors;
#   * EMRI stability: the interpolate.cu cusparse hard-exit hardening is
#     NOT yet committed in FEW -- if this probe dies with a bare exit(-1)
#     inside EMRI likelihoods, that is campaign item C0-EMRI firing, not
#     a new bug. A short probe is the acceptable place to hit it.
#
# SOURCE SELECTION (computed 2026-08-24 from
# mojito_light_v1_0_0/catalogues, laptop copy -- same catalogue family as
# the cluster cache):
#   * MBHB: catalogue has 20; exactly 4 MERGE WITHIN the 6-mo window
#     (user ruling: only include MBHs that merge by end of observation):
#     ids 2, 5, 16, 18 (t_coalescence = 173.3, 104.7, 111.4, 92.0 days).
#   * EMRI: all 8 (ids 0-7) -- 6-mo detectability unknown (user), the
#     probe doubles as the census.
#   * SOBHB: all 6 (ids 0-5) -- most expected undetectable at 6 mo
#     (user); 6 sources is cheap and answers which (if any) are not.
#
# RELAUNCH 2026-08-27 (post-preemption). Five launches that day stored ZERO
# iterations: spot kills landed 30-75 min apart against an ~85-min exposure
# to the first [SAVE]. Three changes make this relaunch answer the open
# questions even if it is preempted again:
#   1. SOBBH RUNS FIRST in the PE stage (user ruling). The still-unmeasured
#      SOBBH leaf timings now arrive minutes after sampling starts instead
#      of ~65 min in, behind mbh (~26 min) and emri (~37 min).
#   2. MID-ITERATION CHECKPOINTS (MIDIT_CHECKPOINT below): the state is
#      pickled at each PE leaf / stage sub-move, so a kill costs at most one
#      leaf instead of the whole iteration. The run SELF-TESTS this in its
#      first minute -- grep "[MIDIT_CKPT] SELF-TEST" to see PASSED/FAILED
#      before trusting it (failure is loud, never fatal: the run then simply
#      behaves as it did before the feature).
#   3. The slurm stdout log is MIRRORED into the store dir, so the
#      [GF_TIMING] table finally travels with the pull.
#
# RESUME: resubmitting resumes from the h5, and now also from the newer
# *_midit_checkpoint.pkl sidecar when one is ahead of the store (a config
# change safely invalidates it -- the snapshot is moved to *.rejected and
# the run falls back). Fresh start: move/delete the store dir first. An
# initialized-but-empty store is now re-initialized against the current
# config automatically (it used to survive relaunches carrying the OLD
# ladder shape, which would have crashed the first surviving save).
# ============================================================================

#SBATCH --job-name=gf6mo_src         # job name
#SBATCH --partition=gpu-80-spot   # GPU partition
#SBATCH --gres=gpu:2              # TWO H100s -- the regular production setup
                                  # (user ruling 2026-08-26); single-GPU leg:
                                  # GPUS=0 sbatch --gres=gpu:1 <this script>
#SBATCH --nodes=1
#SBATCH --ntasks=1                # single process (MPI singleton)
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/shared/data/global_fit_output/gf6mo_src_%j.log

set -euo pipefail

# ---- environment ------------------------------------------------------------
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

STORE_DIR=/shared/data/global_fit_output/gf_prod_6mo_sources_probe/

# ---- GPU telemetry (5 s device sampler + 30 s per-process sampler) ----------
# FIRST-LAUNCH CHECK (2026-08-23): current cluster launches have been writing
# 0-BYTE gpu_util CSVs -- VERIFY this file grows a few minutes in; the
# multi-GPU verdict needs it.
mkdir -p ${STORE_DIR}
GPU_SAMPLE_SEC=${GPU_SAMPLE_SEC:-5}
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l ${GPU_SAMPLE_SEC} > "${GPU_LOG}" &
GPU_SMI_PID=$!
GPU_PROC_LOG=${STORE_DIR}/gpu_procs_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits -l $((GPU_SAMPLE_SEC * 6)) > "${GPU_PROC_LOG}" &
GPU_PROC_PID=$!

# ---- slurm stdout MIRROR into the store dir (2026-08-27) --------------------
# Every line that matters for the timing readout -- [GF_TIMING], [SAVE],
# [MIDIT_CKPT], the driver banner, any traceback -- goes to STDOUT, i.e. the
# slurm --output file, which lives OUTSIDE ${STORE_DIR}. The pulls are zips
# of ${STORE_DIR}, so that file has been missing from EVERY pull so far (it
# was asked for by hand each time and never arrived). Mirror it into the
# store every 30 s: zipping the store now captures it automatically, and
# because this is a copy loop rather than an EXIT trap it survives a spot
# preemption (SIGKILL runs no traps).
SLURM_LOG=/shared/data/global_fit_output/gf6mo_src_${SLURM_JOB_ID:-manual}.log
LOG_MIRROR_PID=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ( while true; do
      cp -f "${SLURM_LOG}" "${STORE_DIR}/slurm_stdout_${SLURM_JOB_ID}.log" 2>/dev/null || true
      sleep 30
    done ) &
  LOG_MIRROR_PID=$!
  echo "[LOGMIRROR] ${SLURM_LOG} -> ${STORE_DIR}/slurm_stdout_${SLURM_JOB_ID}.log every 30 s"
fi

trap 'kill ${GPU_SMI_PID} ${GPU_PROC_PID} ${LOG_MIRROR_PID:-} 2>/dev/null || true' EXIT

# ---- threading policy (MPI-only, no OMP) ------------------------------------
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export VERBOSE=1
export PROGRESS=0

# ---- run plumbing -----------------------------------------------------------
REAL_MOJITO=/shared/data/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
# TWO GPUs -- the regular H100 setup (user ruling 2026-08-26, superseding
# the brief 1-GPU default). 24 walkers shard 12/12 across devices with
# threaded per-split overlap; GPUS=0 restores the single-GPU leg.
export GPUS=${GPUS:-0,1}
# Run the synthetic injection build on the GPU too (user request): the
# stream builder is GPU-safe (asnumpy() at every host accumulation) but
# this is the FIRST GPU exercise of that path -- a failure shows up
# loudly in the first minutes of fit.build(); SYNTHETIC_INJECTION_BACKEND
# =cpu restores the validated CPU injections.
export SYNTHETIC_INJECTION_BACKEND=auto

# ---- DATA MODE: synthetic FOR NOW (user 2026-08-26: the mojito data has
#      not been moved to /shared/data/mojito_cache yet) ----------------------
# Synthetic = self-contained in-process streams, injections generated ON
# THE GPUs (SYNTHETIC_INJECTION_BACKEND=auto above); id lists contribute
# only their COUNTS (4 MBHBs / 8 EMRIs / 6 SOBHBs) from the stock tables;
# synthetic MBH mergers land in the window interior by construction;
# fixed analytic PSD; no instrument noise (exact truth nulls -- with the
# 1e-8 start factors the initial lnL line should be ~identical ~0 across
# all 24 walkers).
# ONCE THE DATA IS IN PLACE: DATA_MODE=mojito flips to real L1 data from
# ${REAL_MOJITO} -- catalogue ids become real rows (MBHB 2,5,16,18 = the
# in-window mergers), fixed PSD = LSQ fit to the NOISE brick.
# PRECONDITION there: MBHB L1 bricks for ids 2 and 5 must exist (loader
# raises loudly naming the missing id).
export DATA_MODE=${DATA_MODE:-synthetic}

# ---- CONFUSION FOREGROUND DATA (user request 2026-08-24; then DISABLED
#      same day: "Turn that off for now in our 6 month test" -- the GALFOR
#      output file is not on the laptop and its presence in the cluster
#      cache is unconfirmed; Robbie's generator subtract_resolvable_tdi.py
#      + inputs ARE in the galaxy-noise worktree, so it can be produced
#      when wanted). ADD_CONFUSION_FG=1 re-arms the whole block. ----------
ADD_CONFUSION_FG=${ADD_CONFUSION_FG:-0}
if [ "${ADD_CONFUSION_FG}" = "1" ] && [ "${DATA_MODE}" != "mojito" ]; then
  echo "[GALFOR] ADD_CONFUSION_FG=1 ignored: the confusion brick is mojito"
  echo "         L1 data and DATA_MODE=${DATA_MODE}. Set DATA_MODE=mojito."
  ADD_CONFUSION_FG=0
fi
if [ "${ADD_CONFUSION_FG}" = "1" ]; then
# Add the SELF-GENERATED confusion-foreground mojito file to the data: the
# GB L1 brick with all resolvable binaries regenerated (GBGPU, on the file's
# own orbits) and subtracted -- sample-accurate, format-preserving, TDI-2
# fractional-frequency in the file's Hz units. Generator:
# noise_updates_from_robbie/subtract_resolvable_tdi.py (galaxy-noise
# worktree); output name GALFOR_731d_2.5s_L1.h5. NOT present in the laptop
# cache (verified 2026-08-24) -- expected in the cluster cache; this block
# FAILS LOUDLY if it cannot find it (set GALFOR_FILE explicitly to
# override the search).
#
# Mechanism: the stock loader maps source type -> data/TYPE/L1 and for GB
# loads exactly ONE file matching GB_*source0_* (preprocessing.find_file),
# summing its TDI into the data stream. So we build a SHADOW mojito folder
# -- catalogues + every class symlinked from the real cache, except data/GB
# holds ONLY the confusion file under a GB-pattern symlink name -- and add
# a GB entry to source_ids (driver below). No gb branch exists in this
# variant, so the confusion TDI is pure unmodeled-by-sources data, and the
# FIXED sensitivity already carries the matching confusion component (the
# variant bakes InstrumentNoise + annually-modulated GalacticForeground
# into extra_components unconditionally). We do NOT touch
# ADD_GALACTIC_FOREGROUND -- that knob would add a SYNTHETIC foreground
# draw on top of the real one.
GALFOR_FILE=${GALFOR_FILE:-$(find ${REAL_MOJITO} -iname "GALFOR*L1*.h5" -print -quit)}
if [ -z "${GALFOR_FILE}" ] || [ ! -f "${GALFOR_FILE}" ]; then
  echo "[GALFOR] FATAL: confusion-foreground L1 file not found under"
  echo "         ${REAL_MOJITO} (pattern GALFOR*L1*.h5) and GALFOR_FILE not set."
  echo "         Generate/copy it (subtract_resolvable_tdi.py output) or"
  echo "         export GALFOR_FILE=/path/to/GALFOR_731d_2.5s_L1.h5"
  exit 3
fi
echo "[GALFOR] using confusion foreground file: ${GALFOR_FILE}"
SHADOW=${STORE_DIR}/mojito_shadow
rm -rf "${SHADOW}"
mkdir -p "${SHADOW}/data/GB/L1"
# The dir that directly holds catalogues/ + data/ (handles both a flat
# cache and a brickmarket/version layout):
BRICK_DIR=$(dirname "$(find ${REAL_MOJITO} -type d -name catalogues -print -quit)")
ln -s "${BRICK_DIR}/catalogues" "${SHADOW}/catalogues"
for cls_dir in "${BRICK_DIR}"/data/*/; do
  cls=$(basename "${cls_dir}")
  if [ "${cls}" != "GB" ]; then
    ln -s "${cls_dir%/}" "${SHADOW}/data/${cls}"
  fi
done
# find_file needs startswith "GB_" and "source0_" in the name:
ln -s "${GALFOR_FILE}" "${SHADOW}/data/GB/L1/GB_731d_2.5s_L1_source0_0_confusion_foreground.h5"
echo "[GALFOR] shadow mojito folder built at ${SHADOW} (GB brick = confusion only)"
export MOJITO_DATA_PATH=${SHADOW}
export PROBE_GB_CONFUSION_STREAM=1   # tells the driver to add the GB stream
else
  export MOJITO_DATA_PATH=${REAL_MOJITO}
  echo "[GALFOR] confusion foreground OFF (ADD_CONFUSION_FG=0) -- data = MBH+EMRI+SOBHB only"
fi

# ---- output -----------------------------------------------------------------
export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_6mo_sources

# ---- per-branch tempering + in-model repeats (user rulings 2026-08-27) ------
# Source branches temper internally (engine ntemps is retired to 1).
# Ladder history: 6 rungs too costly (mbh leaf ~24 min) -> 4 (morning) ->
# MBH/EMRI 2 for this relaunch. SOBBH goes the OTHER way: 12 rungs and 25
# repeats, on the expectation that it is the cheap branch -- which is
# precisely what this probe still has to measure, so it is a TEST, not a
# settled setting (user: "assuming sobhb will be fine ... we will test
# that"). It runs FIRST, so the answer lands minutes after sampling starts.
#
# COST ARITHMETIC (per-leaf work is ~ ntemps * nwalkers * repeats
# likelihood rows on top of a fixed expose/fold round trip; the measured
# baselines below were taken at ntemps=4, repeats=2):
#   mbh   4x2=8  -> 2x5=10   = 1.25x  (1534 s -> ~32 min projected)
#   emri  4x2=8  -> 2x5=10   = 1.25x  (2223 s -> ~46 min projected)
#   sobbh 4x2=8  -> 12x25=300 = 37.5x over its default config (no measured
#                               baseline yet -- this is the open question)
# NB the halved ladders do NOT make mbh/emri cheaper: repeats 2->5 more
# than offsets rungs 4->2. Both branches get slightly MORE expensive, and
# the iteration total is dominated by whatever sobbh turns out to cost.
export MBH_NTEMPS=2
export EMRI_NTEMPS=2
export SOBBH_NTEMPS=12
# In-model stretch repeats per leaf visit (env default is 2 for all three).
# More repeats amortize the per-visit expose/fold overhead over more
# proposals -- worth it where the likelihood row is cheap (sobbh), less so
# where it dominates (mbh/emri).
# MBH/EMRI back to the stock default of 2 (user ruling 2026-08-28; the 5
# was chosen believing the default was higher). Repeats amortize a leaf's
# fixed expose/fold cost over more proposals, which is why SOBBH wants
# many -- but MBH/EMRI template builds CACHE (the injection stage showed
# 14 s for the first EMRI then <1 s for the rest), so there is little
# fixed cost to amortize and the extra repeats were close to pure spend.
# Measured at 5 (job 373): mbh 1376 s (4 leaves), emri ~2000 s (8 leaves),
# an iteration of ~57 min against a 10-min build -- job 373 ran 64 min and
# missed the first [SAVE] by ~3 min, on EMRI's last leaf. At 2 the leaf
# cost is a fixed part plus 2/5 of the per-repeat part, so expect mbh
# ~9-14 min and emri ~13-20 min: an iteration around 23-35 min, which
# banks a stored row inside any observed window.
export MBH_NUM_PROP_REPEATS=2
export EMRI_NUM_PROP_REPEATS=2
# SOBBH 25 -> 10 for THIS measurement run (2026-08-28). The chunked fill
# removed ~25 min of per-leaf dense bookkeeping (expose measured at 0.31 s,
# job 372, down from 24 x 32 s), which leaves the repeat loop AS the entire
# leaf -- and unlike the dense floor it removed, that loop is exactly
# proportional to repeats. Job 372 ran 8.4 min of scoring without finishing
# a leaf, against a window of (preemption 20-25 min) - (build 11.5 min) =
# 8.5-13.5 min: right on the edge, which is why every job misses by a hair.
# At 10 the leaf is ~3.4 min, so boundaries land every ~3 min and
# mid-iteration checkpointing can finally bank progress. Cost is LINEAR in
# repeats, so this measures 25 by multiplication -- put it back up once the
# run can survive a kill.
export SOBBH_NUM_PROP_REPEATS=10
# Fancy (walker-permuting) temperature swap every 10 iterations (user
# ruling 2026-08-27): the measured cost was ~17 min of an ~18.5-min MBH
# leaf visit — ~5x the in-model work — when it fired every propose.
# permute_every is a cadence in proposes again; the FIRST propose after
# a (re)start never fires (iteration 1 = build/JIT transients) -- swaps
# land on iterations 10, 20, ... (<=0 disables).
export MBH_PERMUTE_EVERY=10
export EMRI_PERMUTE_EVERY=10
export SOBBH_PERMUTE_EVERY=10

# ---- start scatter (user ruling 2026-08-26) ---------------------------------
# The default 1e-5 multiplicative truth scatter produced wildly spread
# initial lnL across walkers (-5.6e6 .. -2.2e7) -- at 6-mo source SNRs
# even 1e-5 relative offsets carry ~1e7 of lnL. 1e-8 puts every walker at
# ~the truth null (offsets scale as factor^2 -> ~1e-6 of the above), so
# the initial lnL line should read ~identical values ~0 across all 24.
# Convention: x * (1 + factor * randn); 0 = exact injection. NB the
# scatter is MULTIPLICATIVE, so any exactly-zero injected parameter has
# zero ensemble spread at ANY factor (the stretch move cannot create
# spread it never had -- the known VGB ratio lesson); per-branch
# additive_start_widths is the remedy if a dimension needs unfreezing.
export MBH_START_FACTOR=1e-8
export EMRI_START_FACTOR=1e-8
export SOBBH_START_FACTOR=1e-8

# ---- preemption protection (2026-08-27) -------------------------------------
# Mid-iteration checkpointing: 2026-08-27 logged FIVE spot preemptions in
# one day (kills 30-75 min apart) against an ~85-min exposure to the first
# [SAVE] -- zero iterations ever stored. The sampler now pickles the state
# at sub-iteration boundaries (each PE leaf; each stage sub-move) to
# *_midit_checkpoint.pkl next to the h5, and a relaunch resumes from the
# newest snapshot (a config change safely invalidates it). Default is ON
# with a 600 s write throttle; the probe state is small (MBs), so tighten
# to 5 min -> at most one ~leaf of work lost per kill. Look for
# [MIDIT_CKPT] lines in global_fit.log. MIDIT_CHECKPOINT=0 disables.
export MIDIT_CHECKPOINT=1
export MIDIT_CHECKPOINT_MIN_INTERVAL=300

# ---- shape ------------------------------------------------------------------
# 24 walkers = the production 6mo_v1 shape, so the timing transfers to the
# eventual all_sources run (full_year's historical default was 6; drop back
# to 6 to reproduce the legacy full-year cost profile instead). Branches
# temper internally (<BRANCH>_NTEMPS knobs, variant defaults).
export NWALKERS=24
# A timing probe, not a production run: enough iterations for steady-state
# per-move walls after the first-iteration build/JIT transients.
export NUM_ITERATIONS=30

# ---- 6-month domain ---------------------------------------------------------
# env wins over the variant's chop/full-span defaults (env_resolve).
export TOBS_TARGET=15552000
# Match the 6mo_v1 production domain crop (constant-layer ruling; no GB
# sig-het in this probe so the tukey/taper guard is not in play, but keeping
# the domain identical makes the timing directly comparable).
export EDGE_CROP_WAVELETS=60

# ---- source selection (see header for the derivation) -----------------------
# NB in DATA_MODE=synthetic these lists contribute only their COUNTS to
# the stock synthetic injection tables; the specific catalogue ids apply
# when DATA_MODE=mojito.
export MBHB_IDS=2,5,16,18          # mojito: ONLY the 4 that merge inside 6 months
export EMRI_IDS=0,1,2,3,4,5,6,7    # all 8 -- detectability census (mojito)
export SOBHB_IDS=0,1,2,3,4,5       # all 6 -- expected mostly sub-threshold

# ---- SOBBH chunked accuracy on THIS grid (measured 2026-08-25) --------------
# full_year_combined runs 11.1-h WDM layers (wavelet 40-48 ks) -- the
# STRESS regime for the chunked SOBBH path (intra-chunk sweep is quadratic
# in layer duration; ~3.5 layers/chunk at the Nt_sub=32 default, which the
# removal-null sweep measured as the OPTIMUM on this grid: removal residual
# 6e-4 of <h|h>). But the m=1 SCORING band sheds ~15% of <h|h> here
# (lnL bias ~0.076*SNR^2, trips SOBBH_CHECK_LL_TOL at SNR>~2.6); m=2
# recovers 99.9% and m=3 is fully converged (== m=6, residual floor
# 1.6e-4). 3, not 2 (user ruling 2026-08-26): scoring width is nowhere
# near a bottleneck, so take the converged value for safety. The
# production all_sources grid (1-h layers) does not need this.
export SOBBH_M_BAND_HALF_WIDTH=3
# Thin the built-in fast-vs-slow A/B (user ruling 2026-08-27). The SOBBH
# SCORING path is vectorized -- one chunked-het call for all
# ntemps*nwalkers rows of a leaf -- but ``_verify_prev_logl`` re-scores
# those same points through the SLOW per-row container path (one
# TDI-on-the-fly waveform per row) to measure chunked truncation error,
# and it defaults to EVERY leaf visit. That is 12*24 = 288 serial
# waveform generations per leaf, ~1728 per iteration across the 6 SOBHBs,
# and it scales with ntemps -- so tripling the ladder 4 -> 12 tripled the
# diagnostic too. At 30 it fires on the 30th propose, leaving the
# measured SOBBH wall as the vectorized science path.
# NB with NUM_ITERATIONS=30 that is the FINAL iteration only, so the
# accuracy check effectively does not run in this probe (and not at all
# if the job is preempted first). Lower this -- or raise NUM_ITERATIONS
# -- when the goal is validating chunked-het accuracy at 6 months rather
# than timing it. SOBBH_CHECK_LL=0 would disable it outright; left at 1
# so it still fails loudly past SOBBH_CHECK_LL_TOL when it does fire.
export SOBBH_CHECK_LL_EVERY=30
# Residual-integrity check for the NEW chunked fill (2026-08-28). Folding
# the residual through fill_global_wdm puts chunked truncation into the
# residual itself, where the move previously kept it bit-identical -- and
# nothing has verified that in production yet. SWAP_DEBUG is the CHEAP way
# to check: it logs a per-leaf [STAGE] POST-READD line comparing the move's
# believed cold lnL against the ACS likelihood ("spread ~0 = consistent
# bookkeeping; large spread = state/residual corruption"), costing a couple
# of likelihood evaluations per leaf.
# Do NOT use SOBBH_CHECK_LL_EVERY=1 for this: for SOBBH the "slow" A/B path
# is the same 32 s/row dense build the fill just eliminated, so it would
# cost hours per leaf. Turn this back off once the fill is trusted.
export SOBBH_SWAP_DEBUG=1

# ---- timing instrumentation -------------------------------------------------
# [GF_MOVE_TIMING] per-move wall + host RSS + GPU pool for every move; SYNC
# makes each mark carry exactly its own kernel time (cupy is async). Keep
# SYNC for this probe throughout -- attribution is the whole point and the
# ~2-3% tax is irrelevant here.
export GF_MOVE_TIMING=1
export GF_MOVE_TIMING_SYNC=1

# ---- driver ----------------------------------------------------------------
# Not run_global.py: adding the confusion-foreground GB stream needs a
# source_ids key the env knobs don't cover (MBHB/EMRI/SOBHB_IDS only). This
# is the sanctioned class-API driver pattern (run_global.py's own help:
# "adjust anything else through the class API in a driver script").
python - <<'PYEOF'
import os
from lisatools.globalfit.stock import erebor

fit = erebor.full_year_combined()
gs = fit.general
if os.environ.get("PROBE_GB_CONFUSION_STREAM") == "1":
    # The shadow cache's GB brick IS the confusion-foreground file; ids
    # gate nothing for GB (the loader always takes the one GB_*source0_*
    # file and sums its TDI into the data). No gb branch exists in this
    # variant, so the stream is data-only. Armed via ADD_CONFUSION_FG=1.
    gs.mojito_source_ids["GB"] = [0]
print(f"[driver] tobs_target={gs.tobs_target:.6g} s  "
      f"ids={gs.mojito_source_ids}  gpus={gs.gpus}  "
      f"edge_crop={gs.edge_crop_wavelets}  data={gs.mojito_data_path}",
      flush=True)
curr = fit.build()
fit.run()
PYEOF
