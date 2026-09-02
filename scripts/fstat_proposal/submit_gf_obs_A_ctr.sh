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
#SBATCH --job-name=obs_A_highf_ctr
#SBATCH --partition=gpu-40-spot
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/shared/data/global_fit_output/obs_A_highf_ctr_%j.log

# ---- this arm ---------------------------------------------------------
# layers 144.5-149.5; leaf budget 20 is the proven high-f value
export STORE_DIR=/shared/data/global_fit_output/gf_obs_A_highf_ctr/
export BASE_FILE_NAME=gf_obs_A_highf_ctr
export GB_MIN_FREQ=2.006944e-02
export GB_MAX_FREQ=2.076389e-02
export GB_NLEAVES_MAX=20
export GB_INMODEL_PROPOSAL=observable
export GB_RJ_FSTAT_DIST_BIRTH=1

_HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${_HERE}/obs_probe_common.sh"
source "${_HERE}/submit_gf_highf_grid.sh"
