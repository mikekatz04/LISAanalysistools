#!/bin/bash
# ============================================================================
# KAPPA PROBE -- measure the 3mo v8 run's OWN residual power per WDM layer
# against its noise model. Single short job, no sampling, no writes to the
# production store.
#
# THE QUESTION. The full fit sits at S_oms 1.3879x / S_tm 1.6017x truth
# (1.93x / 2.57x in POWER) while the noise-only AND noise+galfor runs recover
# the injection to 0.05% on the same grid, band and noise model. That splits
# into eps=0.150 (subtraction residual -> drives S_tm) + kappa=0.840
# (BROADBAND -> drives S_oms). Every low-frequency mechanism tried pushes
# S_oms the WRONG way, and above 22 mHz the mojito bricks carry EXACTLY zero
# signal power with only ~14 leaves/walker above 12 mHz. So the residual up
# there should be pure instrument noise. This job measures whether it is.
#
# THE READOUT is the 18-25 mHz row of the band table:
#   q_true/3 ~ 1.84 -> kappa is REAL and measured. The residual carries ~2x
#                      the instrument-noise power where there is no signal and
#                      almost nothing subtracted => a bookkeeping /
#                      normalization problem in how the residual is assembled.
#   q_true/3 ~ 1.00 -> the stored state does NOT reproduce the run's own
#                      fitted noise => the problem is in state handling, and
#                      the next look is the live-vs-rebuilt residual.
# q_fit/3 (same table) should be ~1 either way -- it is the wiring sanity
# check, since the chain is converged against its own model.
#
# SAFETY: runs against a COPY. Building a GlobalFitSetup OPENS the HDF backend
# and the production run may still be live -- never point this at the live dir.
# ============================================================================
#SBATCH --job-name=kappa_probe
#SBATCH --partition=gpu-80-spot
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/shared/data/global_fit_output/kappa_probe_%j.log

set -euo pipefail

source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

SRC=${SRC:-/shared/data/global_fit_output/gf_prod_3mo_v8}
PROBE=/shared/data/global_fit_output/kappa_probe_store

# ---- take the COPY first ----------------------------------------------------
# Before anything else, because the config harvest below resolves
# ${STORE_DIR} and we want it pointing at the copy from the very first
# expansion -- never at the (possibly live) production dir.
rm -rf "${PROBE}"
mkdir -p "${PROBE}"
echo "[probe] copying ${SRC} -> ${PROBE} (source is read-only; run may be live)"
cp -a "${SRC}/." "${PROBE}/"

# ---- inherit the PRODUCTION config verbatim --------------------------------
# Pull the export lines straight out of the production submit script rather
# than restating them: the probe MUST resolve the same grid, band, noise model
# and branch set as the run it is measuring, and a hand-copied block would
# drift. (The `: "${VAR:=...}"` soft-defaults are picked up too.)
#
# ⚠ The harvest is EXPORT lines only, so any PLAIN assignment they reference
# must be defined HERE first or `set -u` aborts the job. Today that is exactly
# one variable: the production script does
#     STORE_DIR=/shared/data/global_fit_output/gf_prod_3mo_v8/   (plain)
#     export FILE_STORE_DIR=${STORE_DIR}                          (harvested)
# so STORE_DIR must exist. We deliberately bind it to the COPY -- that is what
# redirects the whole run at the probe store, and it is a safety property, not
# a convenience. (`$PWD` is the only other reference and is fine after the cd.)
# If a future edit to the production script adds another such reference the
# job will fail loudly here with "unbound variable"; define it below.
STORE_DIR=${PROBE}/
PROD=scripts/fstat_proposal/submit_gf_3mo_v8.sh
eval "$(grep -E '^[[:space:]]*export [A-Z_]+=' "$PROD")"
eval "$(grep -E '^[[:space:]]*: "\$\{[A-Z_]+:=' "$PROD" || true)"
echo "[probe] inherited config from ${PROD}"
echo "[probe] TOBS_TARGET=${TOBS_TARGET:-unset} MIN_FREQ=${MIN_FREQ:-unset} MAX_FREQ=${MAX_FREQ:-unset}"
echo "[probe] UNEQUAL_ARM=${UNEQUAL_ARM:-unset} WDM_PSD_METHOD=${WDM_PSD_METHOD:-unset}"
echo "[probe] NOTE: all_sources pins min_freq=1e-4 as a PLAIN default, so the"
echo "[probe]       exported MIN_FREQ is IGNORED -- the run really analyses"
echo "[probe]       from WDM layer 1. The probe reproduces that faithfully."

# ---- probe-only overrides ---------------------------------------------------
# FILE_STORE_DIR is already ${PROBE}/ via STORE_DIR above; re-exported here so
# the redirection is explicit and survives a change to the harvest.
export FILE_STORE_DIR=${PROBE}/
echo "[probe] FILE_STORE_DIR=${FILE_STORE_DIR}  (must NOT be ${SRC})"
case "${FILE_STORE_DIR}" in
  "${SRC}"|"${SRC}/") echo "[probe] REFUSING: store dir points at the source."; exit 2;;
esac
export GPUS=0
export USE_GPU=1
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export VERBOSE=1
export PROGRESS=0
export KAPPA_OUT=${PROBE}/kappa_probe.npz
# no sampling happens, but keep a resume from re-fitting anything expensive
export GB_FSTAT_REFIT_EVERY=1000000

python scripts/noise/residual_kappa_probe.py

echo "[probe] done -- npz at ${KAPPA_OUT}"
echo "[probe] pull it with:  zip -r kappa_probe.zip ${PROBE}/kappa_probe.npz /shared/data/global_fit_output/kappa_probe_\${SLURM_JOB_ID}.log"
