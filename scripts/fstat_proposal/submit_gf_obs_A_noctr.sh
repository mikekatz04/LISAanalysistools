#!/bin/bash
# ===========================================================================
# OBSERVABLE-BASIS A/B PROBE -- arm A_noctr
#
#   sbatch submit_gf_obs_A_noctr.sh
#
# HIGH-F, F-stat centering OFF. Same band as A_ctr; the birth falls back
#   to the usual phase-maximised d_h/h_h amplitude pin. In v7
#   rj_fstat_centers was 1407.5 s of the 1961.8 s search move -- 54% of
#   EVERY iteration -- so if recovery holds without it the run roughly
#   doubles in speed for free.
#
# Shared config for all four arms lives in obs_probe_common.sh; the run
# itself is submit_gf_highf_grid.sh, sourced at the bottom (its own
# #SBATCH header is inert once sourced -- only THIS header is parsed).
# ===========================================================================
#SBATCH --job-name=obs_A_highf_noctr
#SBATCH --partition=gpu-40-spot
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/shared/data/global_fit_output/obs_A_highf_noctr_%j.log

# ---- this arm ---------------------------------------------------------
# layers 144.5-149.5; leaf budget 20 is the proven high-f value
export STORE_DIR=/shared/data/global_fit_output/gf_obs_A_highf_noctr/
export BASE_FILE_NAME=gf_obs_A_highf_noctr
export GB_MIN_FREQ=2.006944e-02
export GB_MAX_FREQ=2.076389e-02
export GB_NLEAVES_MAX=20
export GB_INMODEL_PROPOSAL=observable
export GB_RJ_FSTAT_DIST_BIRTH=0

_HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${_HERE}/obs_probe_common.sh"
source "${_HERE}/submit_gf_highf_grid.sh"
