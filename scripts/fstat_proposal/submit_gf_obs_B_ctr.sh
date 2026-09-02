#!/bin/bash
# ===========================================================================
# OBSERVABLE-BASIS A/B PROBE -- arm B_ctr
#
#   sbatch submit_gf_obs_B_ctr.sh
#
# LOW-F, F-stat centering ON. The NEUTRALITY control: 6.18-6.88 mHz,
#   108 catalogue / 66 detectable sources, predicted shear 0.04 bins --
#   i.e. zero. The change must do essentially NOTHING here. A material
#   improvement is a BUG, not a win.
#
# Shared config for all four arms lives in obs_probe_common.sh; the run
# itself is submit_gf_highf_grid.sh, sourced at the bottom (its own
# #SBATCH header is inert once sourced -- only THIS header is parsed).
# ===========================================================================
#SBATCH --job-name=obs_B_lowf_ctr
#SBATCH --partition=gpu-40-spot
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/shared/data/global_fit_output/obs_B_lowf_ctr_%j.log

# ---- this arm ---------------------------------------------------------
# layers 44.5-49.5; 200 leaves because at 20 the cap would BIND
# and turn a neutrality test into a cap test
export STORE_DIR=/shared/data/global_fit_output/gf_obs_B_lowf_ctr/
export BASE_FILE_NAME=gf_obs_B_lowf_ctr
export GB_MIN_FREQ=6.180556e-03
export GB_MAX_FREQ=6.875000e-03
export GB_NLEAVES_MAX=200
export GB_INMODEL_PROPOSAL=observable
export GB_RJ_FSTAT_DIST_BIRTH=1

_HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${_HERE}/obs_probe_common.sh"
source "${_HERE}/submit_gf_highf_grid.sh"
