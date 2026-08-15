#!/bin/bash
# ============================================================================
# STRIDE A/B smoke -- GB_BAND_UNIT_STRIDE=2 vs 3 on a COPY of the live store
#
# Purpose (2026-08-15): validate the stride-k unit machinery (LAT >= 0dd45782)
# on production state BEFORE the get_N free-frequency band rework makes
# stride > 2 profitable. On the current uniform whole-layer grid stride 3 is
# a PURE SEPARATION WIDENING: no concurrency gain is expected -- this run
# measures the COST side (3 units/propose instead of 2, 3 tempering
# open/close passes, smaller per-unit fills) and confirms the correctness
# monitors do not move:
#   [GB_ORTHO_LL]  max stays at the ~3e-5/unit floor at BOTH strides
#   [GB_CELL_LL]   worst excesses comparable (stochastic, same family)
#   [GB_ACCEPT]    rj + in-model rates statistically unchanged
#   drift          rebuild floor unchanged
# NOT bit-identical across strides by design (unit membership reorders the
# proposal sequence) -- compare statistically, not sample-by-sample.
#
# USAGE (one job per stride, e.g. on the spare GPU pair while production runs):
#   STRIDE=2 SRC_STORE=/path/to/gf_prod_3mo sbatch scripts/fstat_proposal/submit_stride_ab.sh
#   STRIDE=3 SRC_STORE=/path/to/gf_prod_3mo sbatch scripts/fstat_proposal/submit_stride_ab.sh
# then:
#   python scripts/diagnostics/stride_ab_compare.py \
#       ./stride_ab_s2/gf_prod_3mo_artifacts/globalfit_run.log \
#       ./stride_ab_s3/gf_prod_3mo_artifacts/globalfit_run.log
# ============================================================================

#SBATCH --job-name=strideAB
#SBATCH --partition=gpu-80-spot
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=strideAB_%j.log

set -euo pipefail

source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

: "${STRIDE:?set STRIDE=2 or STRIDE=3}"
: "${SRC_STORE:?set SRC_STORE=/path/to/the/production/store dir}"

# ---- ISOLATED copy of the store (never touch production) -------------------
STORE_DIR=./stride_ab_s${STRIDE}/
if [ ! -d "${STORE_DIR}" ]; then
  mkdir -p "${STORE_DIR}"
  cp -a "${SRC_STORE}"/gf_prod_3mo_testing.h5 "${STORE_DIR}/"
  # fstat grids checkpoint-load in ~3 s; share them read-only via copy
  [ -d "${SRC_STORE}/gb_fstat_fit" ] && cp -a "${SRC_STORE}/gb_fstat_fit" "${STORE_DIR}/"
fi

# ---- GPU telemetry (same columns as production) ----------------------------
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 30 > "${GPU_LOG}" &
GPU_SMI_PID=$!
trap 'kill ${GPU_SMI_PID} 2>/dev/null || true' EXIT

export OMP_NUM_THREADS=1

# ---- identical production env (mirror submit_gf_3mo.sh), short run ---------
export MOJITO_DATA_PATH=/shared/home/mlkatz1/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
export GPUS=0,1

export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_3mo

export NWALKERS=24
# Short: resume-iteration + 4 engine iterations is enough for 4 full search
# proposes per stride (the h5 copy resumes at the production iteration).
export NUM_ITERATIONS=2000
export STRIDE_AB_MAX_NEW_ITERS=4   # honored by the wrapper loop below

export TOBS_TARGET=7776000
export MIN_FREQ=4e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

export GB_NLEAVES_MAX=10000
export GB_N_SUBBANDS=8192
export GB_RJ_FLIP_FRACTION=0.2
export GB_JUMP_FACTOR=0.6
export GB_RJ_GROUPED_INMODEL=1
export GB_RJ_DIRECT_BATCH=1
export GB_RJ_LIVE_CAP_PICK=1
export GB_BUFFER_FIXED_CAPACITY=1
export GB_RJ_FSTAT_CTR_HOIST=1
export GB_FSTAT_CTR_SMEAR=1.5
export GB_TEMPER_ON_REMOVAL=1
export GB_RJ_BAND_SHUTOFF_FMIN_MHZ=10.0
export GB_RJ_BAND_SHUTOFF_AFTER=5
export GB_RJ_BAND_SHUTOFF_SCOPE=search
export GB_FSTAT_REFIT_EVERY=100
export FSTAT_PEAKS_PER_BAND=200
export GB_WDM_BAND_SLAB_LAYERS=5
export GB_USE_GALAXY_PRIOR=1
export VGB_SIGHET_INMODEL=0
export GB_ROUTER_THREADED=1
export GB_ROUTER_DEVICE_RESIDENT=1
export GB_RJ_SNR_TRUNC_DIST=1
export GB_INMODEL_REPEATS_NEWBORN=200
export GB_INMODEL_REPEATS_SURVIVOR=25
export SIGHET_INFOMAT=1
export GB_INFOMAT_PER_BLOCK=1
export GB_ORTHO_LL_CHECK=1

# ---- THE A/B variable ------------------------------------------------------
export GB_BAND_UNIT_STRIDE=${STRIDE}

python scripts/fstat_proposal/run_combined_staged.py
