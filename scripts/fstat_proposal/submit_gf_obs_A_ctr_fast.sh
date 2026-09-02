#!/bin/bash
# ===========================================================================
# OBSERVABLE-BASIS A/B PROBE -- arm A_ctr_fast (2026-09-01 F-stat kernel batch)
#
#   sbatch submit_gf_obs_A_ctr_fast.sh
#
# IDENTICAL configuration to submit_gf_obs_A_ctr.sh -- same band, leaf
# budget, proposals, centering ON -- run on the 2026-09-01 F-stat kernel
# batch. The ONLY intended delta vs the running A_ctr arm is the binary:
#
#   chunked route (rj_fstat_centers / rj replace; LAT f166f344 + GBGPU
#   89eb5ba): per-m fold completion, invC hoist, orbit spline cache
#   (follows CHUNKED_N_CP_ORBIT), exact e = C w factorization.
#   sig-het route (FSTAT_USE_SIGHET=1 epoch grid fit; GBGPU cc3c527):
#   folded pair-loop core 10 -> 4 + exact alpha component picks.
#
# All of it is DEFAULT-ARMED -- the exports below only pin + document the
# state so the arm is attributable. Every physics/proposal knob comes from
# obs_probe_common.sh, unchanged.
#
# READ: compare against BOTH running arms.
#   vs A_ctr   (same config, pre-batch binary): pure kernel-speed delta --
#              rj_fstat_centers s/rows in [GB_TIMING], and the epoch-fit
#              wall. Sampling behaviour should be statistically identical
#              (the batch is exact; reassociation-level FP only, so NOT
#              bit-identical / seed-reproducible against A_ctr).
#   vs A_noctr (centering off): does faster centering close the wall-clock
#              gap while keeping centering's birth quality?
#
# ⚠ REQUIRES the post-cc3c527 GBGPU REBUILD. The N_cp_orbit TypeError
# boundary was already passed by the 89eb5ba compile, so a stale .so now
# runs the OLD sig-het kernel SILENTLY. Pull + ./install.sh (or the GBGPU
# rebuild) BEFORE sbatch; confirm via the gbgpu version suffix
# (dev...+gcc3c527...) in `pip show gbgpu`.
#
# Fresh store required (fdot-axis grid cache refusal), same as every arm.
# ===========================================================================
#SBATCH --job-name=obs_A_highf_ctr_fast
#SBATCH --partition=gpu-40-spot
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/shared/data/global_fit_output/obs_A_highf_ctr_fast_%j.log

# ---- this arm ---------------------------------------------------------
# layers 144.5-149.5; leaf budget 20 is the proven high-f value.
# Band/budget/proposal lines MUST stay identical to submit_gf_obs_A_ctr.sh.
export STORE_DIR=/shared/data/global_fit_output/gf_obs_A_highf_ctr_fast/
export BASE_FILE_NAME=gf_obs_A_highf_ctr_fast
export GB_MIN_FREQ=2.006944e-02
export GB_MAX_FREQ=2.076389e-02
export GB_NLEAVES_MAX=20
export GB_INMODEL_PROPOSAL=observable
export GB_RJ_FSTAT_DIST_BIRTH=1

# ---- 2026-09-01 kernel batch: pin the defaults so the arm is
# ---- self-documenting (none of these change behaviour vs default) -----
export GB_FSTAT_FOLD=1          # 4->2 basis fold (chunked kernel)
export GB_FSTAT_ORBIT_CACHE=1   # F-stat follows the comp's N_cp_orbit
export FSTAT_USE_SIGHET=1       # epoch fit on the sig-het route (default)
echo "  kernel batch    GB_FSTAT_FOLD=${GB_FSTAT_FOLD} GB_FSTAT_ORBIT_CACHE=${GB_FSTAT_ORBIT_CACHE} FSTAT_USE_SIGHET=${FSTAT_USE_SIGHET}"

# LOCATE THE REPO COPY OF THESE SCRIPTS.
# sbatch COPIES the batch file to /var/spool/slurmd/jobNNNNN/slurm_script and
# runs it from there, so ${BASH_SOURCE[0]} points at the spool copy and its
# directory holds nothing else -- dirname "$0" cannot work under Slurm.
# Try, in order: an explicit override, the submit directory, the repo path
# the run script itself already hard-codes, and finally dirname (which does
# work when this file is run directly, outside Slurm).
for _c in "${OBS_PROBE_DIR:-}" \
          "${SLURM_SUBMIT_DIR:-}" \
          "/shared/home/mlkatz1/lisa-analysis-tools/scripts/fstat_proposal" \
          "$( cd "$( dirname "${BASH_SOURCE[0]}" )" 2>/dev/null && pwd )"; do
  if [[ -n "${_c}" && -f "${_c}/obs_probe_common.sh" ]]; then _HERE="${_c}"; break; fi
done
if [[ -z "${_HERE:-}" ]]; then
  echo "ERROR: cannot find obs_probe_common.sh beside this script." >&2
  echo "  Set OBS_PROBE_DIR=<repo>/scripts/fstat_proposal and resubmit." >&2
  exit 5
fi
echo "  script dir      ${_HERE}"

source "${_HERE}/obs_probe_common.sh"
source "${_HERE}/submit_gf_highf_grid.sh"
