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
# ============================================================================

# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf3mo          # job name
#SBATCH --partition=FILLME        # GPU partition
#SBATCH --gres=gpu:2              # 2 GPUs (GPUS=0,1 below are LOCAL indices)
#SBATCH --nodes=1                 # single node
#SBATCH --ntasks=1                # single process (MPI singleton)
#SBATCH --cpus-per-task=FILLME    # e.g. 8-16 (host-side prep threads)
#SBATCH --mem=FILLME              # e.g. 64G (smoke peaked ~35 GB host RSS)
#SBATCH --time=FILLME             # wall limit, e.g. 24:00:00
#SBATCH --output=gf3mo_%j.log     # combined stdout+stderr (captures [MAXLOGL]/[BENCH])
# ----------------------------------------------------------------------------

set -euo pipefail

# ---- environment (fill in your activation) ---------------------------------
# module load FILLME_cuda_module
# source /shared/home/mlkatz1/envs/gf_env/bin/activate    # or conda activate
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
export GB_N_SUBBANDS=16384
# RJ pick thinning (user ruling 2026-08-14): each round proposes to a
# 0.3 random subset of eligible slots; in-model repeats still cover
# ALL alive sources (flip gate is rj-only by construction).
export GB_RJ_FLIP_FRACTION=0.3
# Grouped RJ scheduling: accumulate inds=True picks across RJ rounds
# (1 proposal per cell per round), then ONE full-width in-model block.
# Code default since 2026-08-13; pinned for the run record. =0 restores
# the per-round RJ->in-model interleave.
export GB_RJ_GROUPED_INMODEL=1
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
# Concurrent per-device shard dispatch (code default since 2026-08-13;
# explicit here for the run record). =0 restores serial dispatch if the
# drift/[GB_CELL_LL] checks ever implicate concurrency.
export GB_ROUTER_THREADED=1

python scripts/fstat_proposal/run_combined_staged.py
