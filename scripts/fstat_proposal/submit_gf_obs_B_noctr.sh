#!/bin/bash
# ===========================================================================
# OBSERVABLE-BASIS A/B PROBE -- arm B_noctr
#
#   sbatch submit_gf_obs_B_noctr.sh
#
# LOW-F, F-stat centering OFF. The neutrality control for the
#   no-centering arm.
#
# Shared config for all four arms lives in obs_probe_common.sh; the run
# itself is submit_gf_highf_grid.sh, sourced at the bottom (its own
# #SBATCH header is inert once sourced -- only THIS header is parsed).
# ===========================================================================
#SBATCH --job-name=obs_B_lowf_noctr
#SBATCH --partition=gpu-40-spot
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/shared/data/global_fit_output/obs_B_lowf_noctr_%j.log

# ---- this arm ---------------------------------------------------------
# layers 44.5-49.5; 200 leaves because at 20 the cap would BIND
# and turn a neutrality test into a cap test
export STORE_DIR=/shared/data/global_fit_output/gf_obs_B_lowf_noctr/
export BASE_FILE_NAME=gf_obs_B_lowf_noctr
export GB_MIN_FREQ=6.180556e-03
export GB_MAX_FREQ=6.875000e-03
export GB_NLEAVES_MAX=200
export GB_INMODEL_PROPOSAL=observable
export GB_RJ_FSTAT_DIST_BIRTH=0

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
