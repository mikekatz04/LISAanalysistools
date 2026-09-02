#!/bin/bash
# ===========================================================================
# OBSERVABLE-BASIS A/B PROBE -- arm A_ctr
#
#   sbatch submit_gf_obs_A_ctr.sh
#
# HIGH-F, F-stat centering ON. The band that must improve: 20.07-20.76
#   mHz holds the flagship and NOTHING else (1 catalogue source, SNR 46,
#   nothing within +-300 bins), predicted shear 3.1 bins.
#
# Shared config for all four arms lives in obs_probe_common.sh; the run
# itself is submit_gf_highf_grid.sh, sourced at the bottom (its own
# #SBATCH header is inert once sourced -- only THIS header is parsed).
# ===========================================================================
#SBATCH --job-name=obs_A_highf_ctr_r2
#SBATCH --partition=gpu-40-spot
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/shared/data/global_fit_output/obs_A_highf_ctr_r2_%j.log

# ---- this arm ---------------------------------------------------------
# NOTE (r2): v8 pins centering OFF, so this ctr arm is now a DELIBERATE
# DEVIATION from v8 (the A/B contrast arm), not a v8 representative.
# The noctr arm is the v8-parity run.
# layers 144.5-149.5; leaf budget 20 is the proven high-f value
export STORE_DIR=/shared/data/global_fit_output/gf_obs_A_highf_ctr_r2/
export BASE_FILE_NAME=gf_obs_A_highf_ctr_r2
export GB_MIN_FREQ=2.006944e-02
export GB_MAX_FREQ=2.076389e-02
export GB_NLEAVES_MAX=20
export GB_INMODEL_PROPOSAL=observable
export GB_RJ_FSTAT_DIST_BIRTH=1

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
