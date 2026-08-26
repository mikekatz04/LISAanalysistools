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
# RESUME: resubmitting resumes from the h5. Fresh start: move/delete the
# store dir first.
# ============================================================================

#SBATCH --job-name=gf6mo_src         # job name
#SBATCH --partition=gpu-80-spot   # GPU partition
#SBATCH --gres=gpu:1              # ONE GPU (default GPUS=0); the 2-GPU parity
                                  # leg: sbatch --gres=gpu:2 with GPUS=0,1
#SBATCH --nodes=1
#SBATCH --ntasks=1                # single process (MPI singleton)
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=gf6mo_src_%j.log

set -euo pipefail

# ---- environment ------------------------------------------------------------
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

STORE_DIR=./gf_prod_6mo_sources_probe/

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
trap 'kill ${GPU_SMI_PID} ${GPU_PROC_PID} 2>/dev/null || true' EXIT

# ---- threading policy (MPI-only, no OMP) ------------------------------------
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export VERBOSE=1
export PROGRESS=0

# ---- run plumbing -----------------------------------------------------------
REAL_MOJITO=/shared/home/mlkatz1/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
# ONE GPU by default (user request 2026-08-26); the 2-GPU parity leg of
# gate S5 is the env override:  GPUS=0,1 sbatch <this script>
export GPUS=${GPUS:-0}
# Run the synthetic injection build on the GPU too (user request): the
# stream builder is GPU-safe (asnumpy() at every host accumulation) but
# this is the FIRST GPU exercise of that path -- a failure shows up
# loudly in the first minutes of fit.build(); SYNTHETIC_INJECTION_BACKEND
# =cpu restores the validated CPU injections.
export SYNTHETIC_INJECTION_BACKEND=auto

# ---- DATA MODE: TEMPORARILY SYNTHETIC (user request 2026-08-26) -------------
# "synthetic" builds every stream in-process -- no mojito folder needed.
# What changes vs mojito: injections come from the STOCK SYNTHETIC tables
# (make_*_injections), so the id lists below contribute only their COUNTS
# (4 MBHBs / 8 EMRIs / 6 SOBHBs), not specific catalogue rows; synthetic
# MBH merger times are spread across the interior (0.25-0.85) of the
# window BY CONSTRUCTION, so the merge-in-window ruling holds
# automatically; the fixed sensitivity falls back to the stock analytic
# levels (15e-12 / 3e-15 -- no NOISE brick fit); no instrument noise is
# added to the data (source-only likelihood, exact nulls). TIMING and
# MULTI-GPU evidence (gates S1/S5) remains fully valid; the DETECTABILITY
# CENSUS (gate S4) is catalogue physics and needs the mojito rerun --
# flip back with DATA_MODE=mojito once the cache/bricks are settled.
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
