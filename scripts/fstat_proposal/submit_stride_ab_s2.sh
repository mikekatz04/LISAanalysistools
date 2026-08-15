#!/bin/bash
# ============================================================================
# STRIDE A/B -- arm 2: GB_BAND_UNIT_STRIDE=2 (today's parity scheduling)
#
# Twin of submit_stride_ab_s3.sh -- the two files are IDENTICAL except for
# the three lines marked "A/B ARM" (job name, store dir, stride). Keep them
# in sync; assess with:
#   python scripts/diagnostics/stride_ab_compare.py \
#       ./gf_prod_3mo_stride2/gf_prod_3mo_artifacts/globalfit_run.log \
#       ./gf_prod_3mo_stride3/gf_prod_3mo_artifacts/globalfit_run.log
#
# WHAT IT DOES (2026-08-15 user design):
#   1. Copies the WHOLE live 3-month store folder (SRC_STORE, default
#      ./gf_prod_3mo) into this arm's own store dir -- once, on first
#      launch; re-submitting resumes the copy where it left off. The copy
#      validates that the h5 opens (a copy taken mid-[SAVE] can be torn)
#      and falls back to the .h5.bak if not. The inherited cumulative logs
#      are rotated to *_pre_ab.log so the comparator sees ONLY A/B
#      iterations.
#   2. Runs the EXACT production config for EXACTLY AB_NEW_ITERS new engine
#      iterations (default 6): the job reads the copied h5's iteration
#      counter at start and sets NUM_ITERATIONS = current + AB_NEW_ITERS.
#
# PURPOSE: on the current uniform grid stride 3 is pure separation
# widening -- this A/B measures the stride cost (3 units/propose + 3
# tempering passes vs 2, smaller per-unit fills) and gates the correctness
# monitors ([GB_ORTHO_LL] floor, [GB_CELL_LL] family, [GB_ACCEPT] rates)
# before the get_N free-frequency band rework makes stride > 2 profitable.
# The strides reorder the proposal sequence, so compare rates/statistics,
# never samples.
# ============================================================================

#SBATCH --job-name=strideAB2       # ---- A/B ARM ----
#SBATCH --partition=gpu-80-spot
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=strideAB2_%j.log  # ---- A/B ARM ----

set -euo pipefail

source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

SRC_STORE=${SRC_STORE:-./gf_prod_3mo}
STORE_DIR=./gf_prod_3mo_stride2/   # ---- A/B ARM ----
AB_STRIDE=2                        # ---- A/B ARM ----
AB_NEW_ITERS=${AB_NEW_ITERS:-6}
BASE=gf_prod_3mo

# ---- one-time full-folder copy (isolated; production is never touched) -----
if [ ! -f "${STORE_DIR}/.ab_copy_done" ]; then
  echo "[AB] copying ${SRC_STORE} -> ${STORE_DIR}"
  mkdir -p "${STORE_DIR}"
  cp -a "${SRC_STORE}/." "${STORE_DIR}/"
  # DEEP-validate the h5: a copy taken mid-[SAVE] can carry a truncated
  # gzip chunk that opens fine (attrs OK) but fails only when the torn
  # dataset is READ (job-198 lesson: "filter returned failure during
  # read" at resume). Read EVERY dataset fully; fall back to .bak (also
  # deep-validated); abort if both are torn.
  _deep_check() {
    python - "$1" <<'EOF'
import h5py, sys
def walk(g):
    for k in g:
        o = g[k]
        if isinstance(o, h5py.Group):
            walk(o)
        else:
            o[()]  # full read -> decompresses every chunk
try:
    with h5py.File(sys.argv[1], "r") as f:
        walk(f)
except Exception as e:
    print(f"[AB] deep check FAILED for {sys.argv[1]}: {e!r}")
    sys.exit(1)
EOF
  }
  if ! _deep_check "${STORE_DIR}/${BASE}_testing.h5"; then
    echo "[AB] main h5 torn (copy raced a [SAVE]) -- trying .bak"
    cp -a "${STORE_DIR}/${BASE}_testing.h5.bak" "${STORE_DIR}/${BASE}_testing.h5"
    if ! _deep_check "${STORE_DIR}/${BASE}_testing.h5"; then
      echo "[AB] .bak torn too -- delete ${STORE_DIR} and re-copy right"
      echo "     after the live log shows a completed [SAVE]."
      exit 1
    fi
  fi
  # Rotate inherited logs so this arm's log contains ONLY A/B iterations.
  for lg in globalfit_run.log global_fit.log; do
    if [ -f "${STORE_DIR}/${BASE}_artifacts/${lg}" ]; then
      mv "${STORE_DIR}/${BASE}_artifacts/${lg}" \
         "${STORE_DIR}/${BASE}_artifacts/${lg%.log}_pre_ab.log"
    fi
  done
  touch "${STORE_DIR}/.ab_copy_done"
fi

# ---- EXACT iteration budget: current + AB_NEW_ITERS ------------------------
CUR_ITER=$(python -c "
import h5py
with h5py.File('${STORE_DIR}/${BASE}_testing.h5', 'r') as f:
    print(int(f['global_fit'].attrs['iteration']))")
export NUM_ITERATIONS=$((CUR_ITER + AB_NEW_ITERS))
echo "[AB] stride=${AB_STRIDE}: resuming at iteration ${CUR_ITER}, running to ${NUM_ITERATIONS} (${AB_NEW_ITERS} new)"

# ---- GPU telemetry ---------------------------------------------------------
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 30 > "${GPU_LOG}" &
GPU_SMI_PID=$!
trap 'kill ${GPU_SMI_PID} 2>/dev/null || true' EXIT

export OMP_NUM_THREADS=1

# ---- EXACT production env (mirror submit_gf_3mo.sh @ ff0d747f) -------------
export MOJITO_DATA_PATH=/shared/home/mlkatz1/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
export GPUS=0,1

export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=${BASE}

export NWALKERS=24

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
# VGB ladder pin: the betas bugfix builds vgb.ntemps rungs, but this
# COPIED store carries 1-rung band_temps and the A/B runs NO VGB
# migrations -- pin 1 rung so resume matches (the A/B is stride-only).
export VGB_NTEMPS=1

# ---- THE A/B variable ------------------------------------------------------
export GB_BAND_UNIT_STRIDE=${AB_STRIDE}

python scripts/fstat_proposal/run_combined_staged.py
