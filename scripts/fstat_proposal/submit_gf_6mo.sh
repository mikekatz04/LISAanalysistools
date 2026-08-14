#!/bin/bash
# ============================================================================
# NEXT BIG RUN -- 6-month Tobs (gb + vgb + psd + galfor, mojito)
# Staged recipe via scripts/fstat_proposal/run_combined_staged.py.
#
# DEBUTS in this run (vs the 3-mo production config):
#   * VGB CHIRP-MASS BASIS (VGB_CHIRP_MASS_BASIS=1, LAT >= 526a59d):
#     sampled [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio], fixed
#     [f0, alpha, sin_delta]; fdot = get_fdot(f0, Mc) * (1 + ratio).
#     At 6 months the fast chirpers' ratio posteriors should CONTRACT
#     around 0 -- the built-in GR-consistency gate.
#   * (PENDING -- leave commented until the work lands) WARM-START GB
#     proposal from the 3-mo posterior (clustered components; see
#     docs/warm-start-gb-proposal.md).
#   * (PENDING DECISION -- see the mpiexec block at the bottom) dedicated
#     saver/plot rank once the [SAVE] lines from the 3-mo run size the win.
#
# BEFORE LAUNCH checklist (tracker: project_next_big_run_tracker):
#   1. 3-mo correctness assessment passed (its posterior may also seed the
#      warm start).
#   2. mpiexec -n 3 decision made from [SAVE] data (+ rank-gated build
#      verified if adopted).
#   3. If MBHB/SOBHB/EMRI are to be included: their fold-in items must be
#      closed first (EMRI interpolate.cu hard-exit above all) and the
#      remove_branch lines in run_combined_staged.py revisited. This
#      script as written runs the proven 4-branch configuration.
#
# RESUME: resubmitting resumes from the h5 (same store dir/base name/env).
# Fresh start: move/delete ${STORE_DIR} first.
# ============================================================================

#SBATCH --job-name=gf6mo          # job name
#SBATCH --partition=gpu-80-spot   # GPU partition
#SBATCH --gres=gpu:2              # 2 GPUs (GPUS=0,1 below are LOCAL indices)
#SBATCH --nodes=1                 # single node
#SBATCH --ntasks=1                # single process (MPI singleton; see mpiexec block)
#SBATCH --cpus-per-task=4
#SBATCH --mem=0                   # whole-node memory
#SBATCH --time=24:00:00
#SBATCH --output=gf6mo_%j.log     # combined stdout+stderr ([MAXLOGL]/[GB_ACCEPT]/...)

set -euo pipefail

# ---- environment ------------------------------------------------------------
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

STORE_DIR=./gf_prod_6mo/

# ---- GPU telemetry (30 s nvidia-smi sampler; per-job CSV in the store) ------
mkdir -p ${STORE_DIR}
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 30 > "${GPU_LOG}" &
GPU_SMI_PID=$!
trap 'kill ${GPU_SMI_PID} 2>/dev/null || true' EXIT

# ---- threading policy (MPI-only, no OMP) ------------------------------------
export OMP_NUM_THREADS=1

# ---- run plumbing -----------------------------------------------------------
export MOJITO_DATA_PATH=/shared/home/mlkatz1/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
export GPUS=0,1

# ---- output -----------------------------------------------------------------
export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_6mo

# ---- sampler shape ----------------------------------------------------------
export NWALKERS=24
export NUM_ITERATIONS=2000

# ---- Tobs: 6 months = 180 d = 1.5552e7 s ------------------------------------
# Snaps exactly to Nf=1440 x Nt=4320 x dt=2.5 (Nt = 2x the 3-mo grid; same
# layer_df = 1.389e-4 Hz, so band/slab geometry is unchanged). Pinned
# explicitly: sbatch inherits the submitting shell, and a stale TOBS_TARGET
# export once silently re-gridded a run.
export TOBS_TARGET=15552000

# ---- band (same as 3 mo / 23 mo) --------------------------------------------
export MIN_FREQ=4e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

# ---- sig-het sparse time grid -----------------------------------------------
# 120 divides Nt=4320 -> stride 36 layers = 36 h, the SAME temporal density
# as the validated 3-mo production grid (nt_layer 60 @ Nt 2160). -1 = AUTO
# (~35 h prescription) is the alternative if this ever snaps.
export SIGHET_NT_LAYER=120

# ---- GB knobs ---------------------------------------------------------------
export GB_NLEAVES_MAX=15000
# PER GPU (LAT >= df8771c). TRUE per-slot cost at 6 mo ~ 2 MB (data slab
# (3,5,~4243) ~0.5 MB + XYZ invC ~1.5 MB) -> 4096/GPU = 8192 slots total
# ~ 16.4 GB across both devices, ONE shared buffer across all GB moves
# (66332cf). Fills/likelihoods are chunk-bounded, so raising this trades
# memory for fewer/wider pick rounds -- check the honest sizing INFO line.
export GB_N_SUBBANDS=4096
# RJ pick thinning (standing ruling; NOTE the job-180 marks measured the rj
# kernel LAUNCH-WIDTH-BOUND, so flip < 1 saves little wall -- candidate to
# revisit to 1.0 after the concurrency-era timing lands).
export GB_RJ_FLIP_FRACTION=0.2
# In-model info-matrix jump scale (0.005 default measured 95% cold
# acceptance = steps ~0.5% of a posterior sigma). Tune against the
# [GB_ACCEPT] per-proposal-type line; target ~0.15-0.4.
export GB_JUMP_FACTOR=0.2
export GB_RJ_GROUPED_INMODEL=1
export GB_FSTAT_REFIT_EVERY=100
# Comb nodes scale ~ Tobs (2x the 3-mo fit); the 2-GPU fstat fan-out is
# gate-green and cuts it ~2x.
export FSTAT_PEAKS_PER_BAND=300
export FSTAT_SIGHET_MULTIDEV=1
export GB_WDM_BAND_SLAB_LAYERS=5
export GB_USE_GALAXY_PRIOR=1
export GB_ROUTER_THREADED=1

# ---- VGB: CHIRP-MASS BASIS DEBUT -------------------------------------------
# Sampled [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio] (ndim 6);
# ratio init is ADDITIVE (the zero-truth exception; VGB_RATIO_INIT_WIDTH
# default 0.02) so the stretch ensemble is never collapsed. Fresh store =
# no migration needed; to carry a 3-mo store's VGB state instead, run
# scripts/fstat_proposal/migrate_vgb_chirp_basis.py first.
export VGB_CHIRP_MASS_BASIS=1
# Exact chunked-het VGB scorer (sig-het at VGB SNRs unverified).
export VGB_SIGHET_INMODEL=0

# ---- WARM START (PENDING -- uncomment when the clustered posterior->proposal
#      work lands; see docs/warm-start-gb-proposal.md + the tracker) ---------
# export GB_WARM_START_COMPONENTS=/path/to/gf_prod_3mo_warmstart.npz
# export GB_WARM_START_WEIGHT=...       # mixture weight vs fstat proposal

python scripts/fstat_proposal/run_combined_staged.py

# ---- mpiexec ALTERNATIVE (PENDING the [SAVE] decision) ----------------------
# At np >= 3 the highest rank becomes a dedicated h5-saver + plotter and the
# sampler never blocks on gzip/matplotlib. BEFORE adopting: (1) read the
# [SAVE] sync-write seconds from the 3-mo logs -- switch if they exceed
# ~5-10% of iteration wall; (2) verify spare/saver ranks don't allocate GPU
# memory in fit.build() (rank-gated build may be needed); change --ntasks=3
# above and replace the python line with:
# mpiexec -n 3 python scripts/fstat_proposal/run_combined_staged.py
