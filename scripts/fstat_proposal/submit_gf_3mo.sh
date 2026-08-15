#!/bin/bash
# ============================================================================
# PRODUCTION global fit -- 3-month Tobs (gb + vgb + psd + galfor, mojito)
# Staged recipe: noise_search -> noise_vgb_search -> gb_search -> full_pe
# via scripts/fstat_proposal/run_combined_staged.py  (LAT dev >= 53dc401)
#
# RESUME: re-submitting this script resumes automatically -- the h5 backend
# restores the last saved iteration and completed stage statuses. Keep the
# same FILE_STORE_DIR/BASE_FILE_NAME and env between submissions. For a
# fresh start, move/delete ${STORE_DIR} first.
#
# ############################################################################
# ## PRE-SUBMIT MIGRATION CHECKLIST (2026-08-15 relaunch) -- RUN BOTH ONCE, ##
# ## ON A STOPPED RUN, BEFORE THE FIRST sbatch. Each writes its own .bak.   ##
# ## Resume guards REFUSE loudly if a store is un-migrated, so a missed     ##
# ## step fails fast rather than corrupting -- but it wastes a queue slot.  ##
# ##                                                                        ##
# ##  H5=./gf_prod_3mo/gf_prod_3mo_testing.h5                               ##
# ##                                                                        ##
# ##  # 1. VGB beta ladder: heal the betas=[1e-4] bug + go to 8 rungs       ##
# ##  #    (must match VGB_NTEMPS=8 below)                                  ##
# ##  python scripts/fstat_proposal/fix_vgb_band_temps.py "$H5" 8           ##
# ##                                                                        ##
# ##  # 2. GB leaf caps onto the band/8 cap-cell grid (GB_CAP_DIVISOR=8)    ##
# ##  #    NOTE: --cap-divisor is a FLAG, not a positional argument.        ##
# ##  python scripts/fstat_proposal/migrate_gb_cap_grid.py "$H5" \          ##
# ##      --cap-divisor 8                                                   ##
# ##                                                                        ##
# ##  NOT NEEDED (2026-08-15): migrate_vgb_chirp_basis.py -- the VGB branch ##
# ##  stays on its ESTABLISHED 5-dim distance basis (VGB_CHIRP_MASS_BASIS=0 ##
# ##  below), so the store's sampled columns do not change. If you ALREADY  ##
# ##  ran that migration, restore its .bak before submitting -- a 6-column  ##
# ##  store against the 5-column config trips the ndim resume guard.        ##
# ############################################################################
# ============================================================================

# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf3mo          # job name
#SBATCH --partition=gpu-80-spot   # GPU partition
#SBATCH --gres=gpu:2              # 2 GPUs (GPUS=0,1 below are LOCAL indices)
#SBATCH --nodes=1                 # single node
#SBATCH --ntasks=3                # main + stopped spare + SAVER rank (mpiexec -n 3)
#SBATCH --cpus-per-task=2
#SBATCH --mem=0                   # whole-node memory
#SBATCH --time=24:00:00
#SBATCH --output=gf3mo_%j.log     # combined stdout+stderr (captures [MAXLOGL]/[BENCH])
# ----------------------------------------------------------------------------

set -euo pipefail

# ---- environment (fill in your activation) ---------------------------------
# module load FILLME_cuda_module
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

STORE_DIR=./gf_prod_3mo/

# ---- GPU telemetry ---------------------------------------------------------
# Background nvidia-smi sampler: one CSV row per GPU every 30 s into the run
# store (timestamped per job, so resubmits/resumes append new files rather
# than clobbering). Killed automatically when the job exits. Columns:
# timestamp, index, name, util.gpu [%], util.mem [%], mem.used [MiB],
# mem.total [MiB], power [W], temp [C].
mkdir -p ${STORE_DIR}
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 30 > "${GPU_LOG}" &
GPU_SMI_PID=$!
trap 'kill ${GPU_SMI_PID} 2>/dev/null || true' EXIT

# ---- threading policy (MPI-only, no OMP) -----------------------------------
export OMP_NUM_THREADS=1

# ---- run plumbing ----------------------------------------------------------
export MOJITO_DATA_PATH=/shared/home/mlkatz1/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
export GPUS=0,1

# ---- output ----------------------------------------------------------------
export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_3mo

# ---- sampler shape ---------------------------------------------------------
export NWALKERS=24                 # 24 walkers / 24 GB temps (user ruling)
export NUM_ITERATIONS=2000         # total engine iterations (resume-safe; NITER was a dead name)

# ---- band + domain ---------------------------------------------------------
# EXPLICIT Tobs (2026-08-13): sbatch propagates the submitting shell's env,
# and a stale TOBS_TARGET export (a 3-day one was found live in the shell)
# would silently re-grid this run. Pin the 90-d production value.
export TOBS_TARGET=7776000
export MIN_FREQ=4e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

# ---- GB knobs (everything else rides the flipped defaults: sig-het in-model,
#      fstat-fit-in-move + sig-het fstat, D/2 leaf-cap gate w/ min-iters 5,
#      at-cap RJ skip, cell-lifecycle ll credit, GB_MODE=search +
#      GB_PE_MOVES_STRICT=1 + GB_SEARCH_PRIOR_REMOVAL=1 seeded by the script) --
export GB_NLEAVES_MAX=10000
# FULL parity-unit residency (grouped RJ->in-model scheduling, 2026-08-13):
# one unit = 77 bands x 24 temps x 24 walkers = 44,352 cells; the scheduler
# clamps n_slots to min(GB_N_SUBBANDS, cells), so 50000 means every cell is
# resident (zero mid-unit refills) and the grouped in-model flush runs at
# full grid width. Buffer cost ~255 KB/slot at slab-5 => ~11.3 GB (user
# budget: 10-20 GB is fine). Back off to 2560 (652 MB) if the pool OOMs.
# 16384 (user Q 2026-08-14): slots now amortize ROUNDS, not just memory --
# rounds/unit ~ waves x per-wave depth, so 4x residency ~ 3-4x fewer
# 2.4s host round-trips. ~4.2 GB buffer; post-fix profile at 4096 was
# flat 42-45/31 GB on 96 GB cards. If the unit-open lines stay flat,
# full residency (50000 -> 44,352 slots, ~11.3 GB) is the next step.
export GB_N_SUBBANDS=8192  # PER GPU; TRUE per-slot cost incl. XYZ invC (~1 MB @3mo, ~8 MB @23mo) x 2 move caches -- job-183 sizing   # PER GPU (LAT >= this commit): total = x n_gpus
# RJ pick thinning (user ruling 2026-08-14): each round proposes to a
# 0.3 random subset of eligible slots; in-model repeats still cover
# ALL alive sources (flip gate is rj-only by construction).
export GB_RJ_FLIP_FRACTION=0.2
# In-model info-matrix jump scale: 0.005 default measured 95% cold
# acceptance; 0.2 -> 0.61; 0.4 -> 0.60 (job 196). Job 197 flipped the
# story: with the EXACT per-block SIGHET info matrices live, cold
# acceptance at 0.6 ROSE to 0.71-0.80 (all-T 0.83-0.92, n=27k) -- the
# better-adapted covariance makes the old scale far too timid. 1.2 is
# the next notch (recommended range 1.0-1.5); tune against the
# [GB_ACCEPT] per-proposal-type line toward the 0.15-0.4 target.
export GB_JUMP_FACTOR=1.2
# SPEED-DIAGNOSIS WINDOW #2: CLOSED 2026-08-15 (job-196 05:49 record
# captured the full SYNC-attributed rj readout; export removed -- the
# next [GB_TIMING rj] lines are honest baselines again).
# ---- 2026-08-15 perf batch (ALL code defaults; pinned for the run
#      record; each knob independently revertible) ---------------------
# De-synced in-model repeat loop (device-resident accept chain) rides
# the pull with no knob (bit-identical decisions, tested).
export GB_ROUTER_DEVICE_RESIDENT=1 # params/outputs never host-stage in the
                                   # shard router; =0 restores host staging
export GB_RJ_SNR_TRUNC_DIST=1      # birth distance draw truncated at the
                                   # analytic SNR-5 boundary; truncated
                                   # density in the factors (DB-exact);
                                   # =0 restores the plain lognormal
# Per-class in-model repeats (search mode): newborns polish hard,
# survivors lightly. PE resolves to GB stock num_repeat_proposals.
export GB_INMODEL_REPEATS_NEWBORN=200
export GB_INMODEL_REPEATS_SURVIVOR=25
# Per-block EXACT info matrices through the sig-het fast route
# (~2.4 ms/src vs ~29-46 chunked). The data_index misindex is FIXED and
# multi-GPU slots now route by the BUFFER's slot shards. First
# shakedown: set SIGHET_INFOMAT_VALIDATE=1 for one propose to log the
# fast-vs-chunked reldiff (expect ~1e-4 near-peak, larger off-peak =
# observed-vs-Fisher, fine for a proposal), then remove.
export SIGHET_INFOMAT=1
export GB_INFOMAT_PER_BLOCK=1
# Countable-only F-stat center precompute + lookup-miss fallback rides
# the pull (no knob beyond the existing GB_RJ_FSTAT_CTR_HOIST=1); the
# new [FSTAT_CTR] census line diagnoses the job-195 5x centers blowup.
# Bilinearity bookkeeping monitor (code default ON, user ruling: ~1.5 s
# per propose = negligible): per-unit [GB_ORTHO_LL] line compares the
# realized cold parent-residual delta against the summed per-buffer
# deltas; WARNs above GB_ORTHO_LL_TOL (0.05). The [GB_ORTHO] premise
# check (GB_ORTHO_CHECK) stays OFF until the sub-band shakedown.
export GB_ORTHO_LL_CHECK=1
# Cap-cell grid (user design 2026-08-15): leaf caps on a band/8 grid at
# the confusion scale; scheduling unchanged. RESUME REQUIRES migration
# step 3 in the header checklist. WATCH first propose: leaf growth +
# memory (the band-total throttle is gone -- a band can now reach
# K*cap); GB_CAP_DIVISOR=1 reverts instantly.
export GB_CAP_DIVISOR=8
export GB_CAP_LL_CHECK=1
# Grouped RJ scheduling: accumulate inds=True picks across RJ rounds
# (1 proposal per cell per round), then ONE full-width in-model block.
# Code default since 2026-08-13; pinned for the run record. =0 restores
# the per-round RJ->in-model interleave.
export GB_RJ_GROUPED_INMODEL=1
# ---- 2026-08-14 rj stack (ALL code defaults; pinned for the run record;
#      each =0 reverts that piece independently) --------------------------
export GB_RJ_DIRECT_BATCH=1        # rigid batches -> one end-of-unit in-model
                                   # phase; =0 restores the staged scheduler
export GB_RJ_LIVE_CAP_PICK=1       # live at-cap birth gate + same-unit
                                   # re-entry (freed cells birth again)
export GB_BUFFER_FIXED_CAPACITY=1  # ONE capacity buffer; smaller units
                                   # resize-rebind instead of drop+rebuild
export GB_RJ_FSTAT_CTR_HOIST=1     # F-stat distance centers batched once per
                                   # unit (was 735 s/propose per-round)
# EPOCH CENTERS (user ruling 2026-08-15: compute the center
# distributions ONCE when the fstat distribution is built in setup(),
# smear for inaccuracy, done): per-epoch table over the proposal's
# drawable support; propose-time = nearest-node lookup (centers chain
# 109-953 s -> ~0). Smear defaults 2.0 in epoch mode (covers <=100-
# propose staleness + node mismatch); =unit restores the per-unit hoist.
export GB_FSTAT_CTR_MODE=epoch
# (GB_FSTAT_CTR_SMEAR is unset ON PURPOSE: a 1.5 pin would override the
# epoch-mode 2.0 default, which is what covers the <=100-propose table
# staleness. The smeared sigma feeds BOTH the draw and the densities,
# so detailed balance is exact at any smear.)
export GB_TEMPER_ON_REMOVAL=1      # band swaps run inside rj_prior_removal
# High-f barren-band birth shutoff (search scope): bands above FMIN with
# AFTER consecutive zero-birth-accept proposes stop proposing births
# (deaths + in-model continue; [GB_BAND_SHUTOFF] log line per band).
export GB_RJ_BAND_SHUTOFF_FMIN_MHZ=10.0
export GB_RJ_BAND_SHUTOFF_AFTER=5
export GB_RJ_BAND_SHUTOFF_SCOPE=search
export GB_FSTAT_REFIT_EVERY=100    # production cadence (5 was verify-only)
export FSTAT_PEAKS_PER_BAND=200    # per-sub-band peak cap (code default; explicit)
# Slab 5 (user ruling): measured-safe (+-1 layer holds >=1-1e-7 of tone
# energy; 5 = 2x that need) and ~30%% smaller band buffers than the AUTO 7.
# Smoke 2 exonerated the slab as the VGB [GB_CELL_LL] growth cause (growth
# persisted at slab 7).
export GB_WDM_BAND_SLAB_LAYERS=5
# 3-D Milky Way (dist, alpha, sin_delta) joint prior (user ruling: the
# proper density, not the flat placeholder). Chirp-mass basis + the
# astrophysical f0-Mc GMM prior are already the code defaults. NOTE:
# this knob was NOT exercised in the smokes (they ran the uniform
# placeholder); detailed balance holds either way -- births still draw
# dist from the birth container and the prior enters through logp.
export GB_USE_GALAXY_PRIOR=1

# ---- VGB: exact chunked-het in-model scorer (sig-het accuracy at the
#      loudest-VGB SNRs is unverified -- [GB_CELL_LL] growth in smoke 1) ----
export VGB_SIGHET_INMODEL=0
# VGB RELAUNCH BLOCK (2026-08-15, user rulings). The VGB likelihood was
# OFF all run (betas=[1e-4] bug) and 36/55 leaves were frozen by the GB
# SNR gate -- both fixed in code (76cd3237); pre-fix VGB samples are
# prior-only. Migration 1 in the header checklist is REQUIRED for the
# VGB_NTEMPS=8 ladder below.
# PARAMETERIZATION: the ESTABLISHED 5-dim DISTANCE basis
# [dist, phi0, cos_iota, psi, fdot_astro_ratio] -- what this store has
# been sampling all along. (User ruling 2026-08-15, superseding the
# earlier chirp-basis arming: "revert to the old VGB parameterization
# ... the old regular parameterization we had before" for these runs.)
# Keeping it means NO chirp migration and NO ndim 5->6 change on a live
# store -- one less thing moving while we validate the likelihood fix.
# The 6-dim chirp basis (Mc sampled, un-collapses fdot_astro_ratio)
# stays built and tested for the 6-month run: set 1 there and run
# migrate_vgb_chirp_basis.py first.
# NOTE: fdot_astro_ratio stays a COLLAPSED dimension in this basis
# (truth exactly 0 x multiplicative init = zero spread, and the
# affine-invariant stretch cannot create spread it never had). That is
# the known, accepted cost of staying on the old parameterization.
export VGB_CHIRP_MASS_BASIS=0
# 8-rung ladder (user ruling 2026-08-15). Resume derives the rung count
# from the STORED band_temps shape, so the migration above MUST be run
# with the matching "8" argument (it recreates every rung-dimensioned
# vgb dataset: temps ladder, counters zeroed, 7 swap pairs).
export VGB_NTEMPS=8
# Concurrent per-device shard dispatch (code default since 2026-08-13;
# explicit here for the run record). =0 restores serial dispatch if the
# drift/[GB_CELL_LL] checks ever implicate concurrency.
export GB_ROUTER_THREADED=1

# ============================================================================
# ONE-TIME RELAUNCH PREP (2026-08-15). Guarded by a marker file, so ordinary
# resubmits/resumes skip it entirely and a mid-fit death resumes normally.
# To RE-ARM (e.g. after restoring a .bak): rm ${STORE_DIR}/.relaunch_prep_*
# Runs BEFORE the sampler, uses the knobs exported above so the store and the
# config can never disagree, and fails the job fast (set -e) if a step errors
# rather than launching onto a half-migrated store.
# ============================================================================
PREP_MARK=${STORE_DIR}/.relaunch_prep_2026_08_15
H5=${STORE_DIR}/${BASE_FILE_NAME}_testing.h5
if [ ! -f "${PREP_MARK}" ]; then
  echo "[PREP] one-time relaunch preparation on ${H5}"

  # 1. VGB beta ladder: heal the betas=[1e-4] bug and re-rung to VGB_NTEMPS.
  #    (Recreates every rung-dimensioned vgb dataset; writes its own .bak.)
  python scripts/fstat_proposal/fix_vgb_band_temps.py "${H5}" "${VGB_NTEMPS}"

  # 2. GB leaf caps onto the band/GB_CAP_DIVISOR cap-cell grid. Each band
  #    hands its current cap + min-iters counters to all its cells (inherit,
  #    never tighten). Refuses if its .bak already exists.
  python scripts/fstat_proposal/migrate_gb_cap_grid.py "${H5}" \
      --cap-divisor "${GB_CAP_DIVISOR}"

  # 3. RETIRE THE STALE F-STAT EPOCH so the fit-in-move rebuilds the grid
  #    against the CURRENT residual. Two independent reasons this grid is
  #    stale: (a) it was fitted at epoch 0 (2026-08-13, ~iteration 8) and the
  #    model has since absorbed ~140 GB sources/walker that are now
  #    SUBTRACTED out of the residual it was fitted against; (b) the VGB
  #    likelihood was OFF for the whole run (betas=[1e-4]), so VGBs were
  #    never actually fitted -- with that repaired they now move and subtract
  #    properly, changing the residual again. Archiving (not deleting) the
  #    epoch dirs makes _latest_epoch() return None -> a fresh "fit" at the
  #    first rj_fstat_search setup(); the archive keeps the old grid for
  #    comparison. The epoch table of F-stat CENTERS is rebuilt in the same
  #    sweep (GB_FSTAT_CTR_MODE=epoch), so centers follow the new residual
  #    automatically.
  FSTAT_SHARED=${STORE_DIR}/gb_fstat_fit/shared
  if [ -d "${FSTAT_SHARED}" ] && compgen -G "${FSTAT_SHARED}/epoch_*" > /dev/null; then
    ARCH=${STORE_DIR}/gb_fstat_fit/stale_epochs_$(date +%Y%m%d_%H%M%S)
    mkdir -p "${ARCH}"
    mv "${FSTAT_SHARED}"/epoch_* "${ARCH}/"
    echo "[PREP] archived stale F-stat epoch(s) -> ${ARCH} (fresh fit will run)"
  else
    echo "[PREP] no F-stat epochs found -- the first fit will build epoch_0000"
  fi

  touch "${PREP_MARK}"
  echo "[PREP] complete; marker ${PREP_MARK}"
else
  echo "[PREP] already applied (${PREP_MARK}) -- skipping; normal resume."
fi

# LATER REFITS: GB_FSTAT_REFIT_EVERY=100 proposal-hits (~8 h at the new
# iteration cadence, ~3.5% overhead at a 17.7-min fit). To force an extra
# refit mid-run WITHOUT restarting from a stale grid, stop the job and
# archive the epoch dir again:
#   mv ./gf_prod_3mo/gb_fstat_fit/shared/epoch_* /tmp/  &&  sbatch ...
# Worth doing once the VGBs have visibly converged, since this first fresh
# fit happens after only ONE VGB move (noise_vgb_joint_search runs before
# rj_fstat_search within the gb_search stage, so it is one move, not zero).

# DEDICATED SAVER RANK (armed 2026-08-15, user directive -- the [SAVE]
# math flipped: the ~60 s sync write was 2% of a 55-min iteration but
# is 6-10% of the post-mega-batch 10-17 min iterations). np>=3: rank 0
# samples, the HIGHEST rank becomes the async results/saver rank, the
# middle spare is stopped at startup (run.py GlobalFit role logic).
# FIRST-LAUNCH CHECK (known caveat): run_combined_staged.py builds on
# EVERY rank before roles resolve -- watch nvidia-smi for saver/spare
# device allocations; if the extra ranks hold GPU memory, drop back to
# the plain single-process line below until the rank-gated build lands.
mpiexec -n 3 python scripts/fstat_proposal/run_combined_staged.py
# python scripts/fstat_proposal/run_combined_staged.py   # single-process fallback
