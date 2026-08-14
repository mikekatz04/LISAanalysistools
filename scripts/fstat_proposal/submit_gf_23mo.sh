#!/bin/bash
# ============================================================================
# SCALING RUN -- 23-month Tobs (gb + vgb + psd + galfor, mojito)
# Identical staged recipe to submit_gf_3mo.sh; only the knobs below differ.
#
# !! PREFLIGHT GATES -- do these BEFORE submitting !!
#
# 1. FSTAT KERNEL SHARED-MEMORY CLAMP (blocking) -- FIX LANDED 2026-08-12,
#    GATE STILL REQUIRED. The sig-het shared-memory work on GBGPU dev
#    (checked opt-in + setup-time clamp for the fstat scorer; global-memory
#    twiddle fallback for gb_signal_het_make_reference, whose 287 KB
#    all-shared carve at THIS grid's Nt=16800 was over every device's
#    limit) must be in the wheel: REBUILD GBGPU from dev on the cluster
#    (stale .so = the old "GPUassert: invalid argument
#    gb_tdi_on_the_fly.cu:6747" crash). Then run the one-band gate ON A GPU
#    NODE in this run's env and require "GATE: PASS":
#
#        BACKEND=cuda13x python scripts/gb_chunked_het/gate_sighet_fstat_nt525.py
#
#    (exercises the 23-month grid shape at SIGHET_NT_LAYER=525: reference
#    build + fstat sweep + mode-0/1 parity + 525-vs-420 grid consistency.)
#
# 2. MEMORY SHAKEDOWN (strongly recommended). Per-walker WDM residual is
#    3 x 1440 x 16800 f64 ~ 0.58 GB -- 7.8x the 3-month grid -- plus the
#    invC store at the same scaling. Run NUM_ITERATIONS=2 first and watch the
#    "GPU pool used" lines; if OOM, reduce NWALKERS or add GPUs.
#
# 3. SKIPPED BY USER RULING (2026-08-13): the memory shakedown and the
#    FSTAT_SIGHET_MULTIDEV=check parity gate. The run launches pinned
#    (=0 below) and checkpoint-resumes through any OOM/bug: if the build
#    OOMs, reduce NWALKERS or take gpu:4, and resubmit.
#
# Expected scaling vs 3 mo: sig-het in-model scoring is FLAT in Tobs
# (v4/v5 bench-off), so per-source proposal cost holds; buffer fills,
# band likelihoods, and the fstat grid fit scale ~8x (comb nodes ~ Tobs).
# ============================================================================

# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf23mo         # job name
#SBATCH --partition=FILLME        # GPU partition
#SBATCH --gres=gpu:2              # 2 GPUs; consider 4 if the shakedown is tight
#SBATCH --nodes=1                 # single node
#SBATCH --ntasks=1                # single process (MPI singleton)
#SBATCH --cpus-per-task=FILLME    # e.g. 8-16
#SBATCH --mem=FILLME              # e.g. 128G+ (host arrays scale ~8x vs 3 mo)
#SBATCH --time=FILLME             # wall limit -- iterations are ~2-5x slower
#SBATCH --output=gf23mo_%j.log    # combined stdout+stderr
# ----------------------------------------------------------------------------

set -euo pipefail

# ---- environment (fill in your activation) ---------------------------------
# module load FILLME_cuda_module
# source /shared/home/mlkatz1/envs/gf_env/bin/activate    # or conda activate
cd /shared/home/mlkatz1/lisa-analysis-tools

STORE_DIR=./gf_prod_23mo/

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

export OMP_NUM_THREADS=1

export MOJITO_DATA_PATH=/shared/home/mlkatz1/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
export GPUS=0,1

export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_23mo

export NWALKERS=24
export NUM_ITERATIONS=2000     # total engine iterations (resume-safe; NITER was a dead name)

# ---- Tobs: 23 months = 700 d = 6.048e7 s -----------------------------------
# Snaps exactly to Nf=1440 x Nt=16800 x dt=2.5 with layer_dt=3600 s (same
# layer_df=1.389e-4 Hz as the 3-mo grid, so band/slab geometry is unchanged).
# The 731-d mojito files cover 700 d + 2 x 8.33 d preprocessing trim.
export TOBS_TARGET=60480000

# ---- band (same as 3 mo) ---------------------------------------------------
export MIN_FREQ=4e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

# ---- sig-het accuracy at long baselines (the 2026-08-03/04 studies) --------
# The 2/4-yr accuracy "wall" was the COARSE SPARSE TIME GRID, not the
# engine: at N_sparse_t ~ 510 the 2-yr battery passes every tier. 525
# divides Nt=16800 (stride 32 layers = 32 h -- the same temporal density
# the 3-mo default gives). Runnable ONLY because v5 (default ON) needs
# ~31 KB scratch at this N; v4's 301 KB cannot launch. n_r=64 fit nodes +
# K=128 knots (the ruling from the overnight battery) are already the
# defaults; knots are not a lever.
export SIGHET_NT_LAYER=525

# ---- GB knobs --------------------------------------------------------------
export GB_NLEAVES_MAX=25000        # user ruling for the 23-mo run
# Grouped RJ->in-model scheduling (2026-08-13, code default; pinned for the
# run record): RJ rounds accumulate below-cap inds=True picks, then ONE
# full-width in-model block. =0 restores the per-round interleave.
export GB_RJ_GROUPED_INMODEL=1
# Buffer sized for the grouped scheme (user budget: 10-20 GB of GPU memory
# is fine). Per-slot slab (3, 5, ~Nt_band) scales ~ Tobs: ~255 KB at 3 mo
# -> ~2.0 MB at 23 mo, so 8192 slots ~= 16 GB and in-model flushes run
# 8192 cells wide (full unit residency = 44,352 cells ~= 88 GB does NOT
# fit). Back off to 4096 (~8 GB) or 2560 (~5 GB) if the pool OOMs.
export GB_N_SUBBANDS=8192
# Comb nodes scale ~ Tobs (0.5/Tobs spacing) -> each epoch fit is ~8x the
# 3-mo cost. 100 keeps the same iteration cadence; raise to 200 if the
# [GF_TIMING] lines show refits dominating.
export GB_FSTAT_REFIT_EVERY=100
# 500, not the 200 default (user ruling): ~8x finer comb resolves more
# distinct peaks per sub-band at 23 mo; the cap must not squeeze them out.
export FSTAT_PEAKS_PER_BAND=500
# Slab 5 (user ruling; see submit_gf_3mo.sh) -- the per-cell slab is
# (3, 5, Nt) so the saving matters more at Nt=16800.
export GB_WDM_BAND_SLAB_LAYERS=5
# 3-D Milky Way (dist, alpha, sin_delta) joint prior (user ruling: the
# proper density, not the flat placeholder). Chirp-mass basis + the
# astrophysical f0-Mc GMM prior are already the code defaults. NOTE:
# this knob was NOT exercised in the smokes (they ran the uniform
# placeholder); detailed balance holds either way -- births still draw
# dist from the birth container and the prior enters through logp.
export GB_USE_GALAXY_PRIOR=1

# ---- VGB: exact chunked-het in-model scorer (see submit_gf_3mo.sh) ---------
export VGB_SIGHET_INMODEL=0
# Concurrent per-device shard dispatch (code default since 2026-08-13;
# explicit here for the run record). =0 restores serial dispatch if the
# drift/[GB_CELL_LL] checks ever implicate concurrency.
export GB_ROUTER_THREADED=1
# PINNED fstat scorer (user ruling 2026-08-13): the =check parity gate
# for the multi-device fan-out was skipped as not worth the delay --
# fstat fit speed is far from the bottleneck -- so this run uses the
# single-device pin, the exact path the 3-month production run
# validates. Flip to =1 only after a green FSTAT_SIGHET_MULTIDEV=check
# pass on a 2-GPU allocation.
export FSTAT_SIGHET_MULTIDEV=0

python scripts/fstat_proposal/run_combined_staged.py
