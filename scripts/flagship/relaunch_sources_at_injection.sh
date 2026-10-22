#!/usr/bin/env bash
# Relaunch a STOPPED global-fit run with its non-GB sources (VGB, MBHB, SOBBH,
# EMRI) pushed back to the catalogue injection values (Mike 2026-10-22: "We
# want those to always start at the catalog injected values rather than the
# last run's values ... make a script that pushes the ongoing runs (when
# stopped) back to the injection values").
#
# usage:
#   scripts/flagship/relaunch_sources_at_injection.sh LAUNCHER [BRANCHES]
#     LAUNCHER  the run's own launcher, e.g. scripts/flagship/submit_gf_12mo.sh
#               (1yr flagship) or scripts/fstat_proposal/submit_gf_9mo_v9.sh (9mo)
#     BRANCHES  comma list out of vgb,mbh,emri,sobbh (default: all four)
#   Every other launch-line knob passes through, e.g. the 12mo's
#     STAGE_SKIP_NOISE=1 scripts/flagship/relaunch_sources_at_injection.sh \
#         scripts/flagship/submit_gf_12mo.sh
#
# What it does: exports {BRANCH}_RESET_TO_INJECTION_ON_RESUME=1 for the chosen
# branches for THIS launch only and runs the launcher, which submits itself
# with `sbatch --export=ALL`, so the knobs reach the job. At the resume the run
# replaces those branches' stored last-row coordinates -- every leaf, walker
# and rung -- with the injection rows it built for this launch (the same rows a
# fresh start uses: this run's id lists, merger-window filter and sampling
# basis), then rebuilds the residual and the log-likelihood from them
# (lisatools.globalfit.source_reset). The branches' per-leaf d_h / h_h
# records read NaN until their moves run, and their entries in
# <store>_eigen_tables.pkl are dropped (backup <sidecar>.pre_source_reset_<job>)
# so the first visit builds the proposal tables at the injection. Nothing on
# disk is edited before the job starts; the store continues from the reset
# state, and the next BARE relaunch resumes normally (the knobs live in no
# launcher).
#
# Requeues: a SLURM requeue re-runs the job with these knobs still set. The run
# writes <store>_source_reset.json (job id + the stored row it reset at); the
# same job resumed from a LATER row skips the reset, the same job resumed from
# the same row (nothing saved yet) applies it again. If the job dies before its
# first saved row and you relaunch BARE, the reset is lost: run this script
# again instead (a new job applies it afresh).
#
# Run it only when the job is STOPPED (the knobs act at the resume). Check the
# job's first log for one line per branch:
#   [SOURCE-RESET] mbh: 12 leaves x 4 walkers set to the catalogue injection ...
# and, after the setup, `initial log likelihood (after recipe setup)` at the
# lnL of the injected sources (the GB and noise parts unchanged).
set -euo pipefail

usage() {
  sed -n '9,16p' "$0" | sed 's/^# \{0,1\}//'
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage >&2
  exit 2
fi
launcher=$1
branches=${2:-vgb,mbh,emri,sobbh}

if [ ! -f "${launcher}" ]; then
  echo "[SOURCE-RESET] launcher not found: ${launcher}" >&2
  exit 2
fi
if [ ! -x "${launcher}" ]; then
  echo "[SOURCE-RESET] launcher is not executable: ${launcher}" >&2
  exit 2
fi

armed=""
_list=()
IFS=',' read -r -a _list <<< "${branches}" || true
if [ "${#_list[@]}" -eq 0 ]; then
  echo "[SOURCE-RESET] no branch selected" >&2
  exit 2
fi
for b in "${_list[@]}"; do
  case "${b}" in
    vgb|mbh|emri|sobbh) ;;
    *)
      echo "[SOURCE-RESET] unknown branch '${b}' (allowed: vgb,mbh,emri,sobbh)" >&2
      exit 2
      ;;
  esac
  knob="$(printf '%s' "${b}" | tr '[:lower:]' '[:upper:]')_RESET_TO_INJECTION_ON_RESUME"
  export "${knob}=1"
  armed="${armed} ${knob}=1"
done
if [ -z "${armed}" ]; then
  echo "[SOURCE-RESET] no branch selected" >&2
  exit 2
fi

echo "[SOURCE-RESET] one-launch reset armed:${armed}"
# a bare file name would make exec search PATH instead of the current directory
case "${launcher}" in
  */*) ;;
  *) launcher="./${launcher}" ;;
esac
echo "[SOURCE-RESET] launching ${launcher} (the job must be STOPPED; the reset acts at its resume)"
exec "${launcher}"
