#!/bin/bash
# ============================================================================
# OBSERVABLE-BASIS PROPOSAL PROBES -- the v8 gate.
#
# Four cheap single-GPU jobs that decide whether v8 launches. Each is a thin
# retarget of the PROVEN submit_gf_highf_grid.sh (GB-only, injection PSD, no
# noise wait, 24 walkers, ~70-200 s/iteration against ~33 min for the
# 3-month production run) -- this file sets only the band, the leaf budget,
# the store and the proposal knob, then execs it. Nothing about the physics
# config is forked, so a probe result transfers to v8.
#
#   ./submit_gf_obs_probe.sh A obs           # high-f, observable
#   ./submit_gf_obs_probe.sh B obs           # low-f,  observable
#   ./submit_gf_obs_probe.sh A leg           # high-f, legacy   (control)
#   ./submit_gf_obs_probe.sh B leg           # low-f,  legacy   (control)
#
# Third argument selects the F-stat CENTERING arm (default "ctr"):
#   ./submit_gf_obs_probe.sh A obs noctr     # high-f, observable, no centering
#   ./submit_gf_obs_probe.sh B obs noctr     # low-f,  observable, no centering
#
# A and B run CONCURRENTLY, one GPU each, on gpu-40-spot. Fire the two
# "obs" arms first and read them at ~25 min; the "leg" controls are only
# needed if the obs arms show something worth attributing.
#
# WHY FOUR AND NOT TWO. Twice this campaign a control caught a defect in the
# TEST rather than in the code, and an invariance claim without one proves
# nothing. Probe B IS the control for the mechanism; the paired legacy runs
# are the controls for the outcome. The endgame run's high-f numbers are NOT
# an adequate baseline for A -- they were reached through a long hand-tuned
# transport stack, so they do not isolate the proposal. A same-seed legacy
# run at this exact config does.
#
# ---------------------------------------------------------------------------
# THE BANDS (counted against gb_truth_3to21.npz; layer = 0.13889 mHz)
#
#   A  layers 144.5-149.5   20.0694-20.7639 mHz    1 catalogue source, SNR 46
#      The flagship ALONE -- nothing within +-300 bins, and the highest-f
#      source in the catalogue. Ten cold walkers hold exactly one leaf: no
#      cap pressure, no neighbour, no pair, and the climb to truth is smooth
#      and monotone. Any freeze here is the proposal and nothing else.
#      Predicted shear: 3.1 bins. This is where the change must show up.
#
#   B  layers  44.5- 49.5    6.1806- 6.8750 mHz  108 catalogue, 66 at SNR>7
#      Predicted shear: 0.04 bins -- i.e. ZERO. 66 detectable sources give
#      this enough statistical power to see a change if one happens.
#
# THE FALSIFIABLE PREDICTION. The shear scales as fdot*T^2, so the new
# proposal must be essentially NEUTRAL in B and large in A. A material
# improvement in B would mean the move is perturbing a regime where the
# defect it fixes does not exist -- that is a BUG, not a bonus, and it is
# far cheaper to find here than 33 minutes per iteration into v8.
#
# ---------------------------------------------------------------------------
# GATES TO PASS BEFORE LAUNCHING v8
#
#   factors device path      either  no _imk_layout_problem; the trace's
#                                    independent Jacobian recompute prints
#                                    "match", never "*** MISMATCH ***"
#   cold in-model acceptance    A     0.67 -> 0.23-0.44
#   mean |dln_fdot|             A     materially above the legacy control
#   mean |df_mid| (bins)        A     ~0.012 (its own step), NOT ~0.170
#   flagship fdot/truth         A     walks off 1.35 toward 1.0
#   low-f neutrality            B     recovery + timing unchanged vs control
#
# READ THEM FROM (both mirrored into globalfit_run.log):
#   [GB_ACCEPT ...] in-model by proposal type -- obs_basis: cold X/Y (rate)
#   [GB_OBS_BASIS ...] in-model motion -- draws N accepted M (rate);
#       mean |dln_fdot| prop=... acc=...; mean |df_mid| prop=... acc=... bins
#   [GB_INMODEL_TRACE ...] DB: ln|dy/dz| recomputed=... factors=... -> match
#
# The [GB_OBS_BASIS] line is the one that answers "is fdot finally moving".
# It does not exist on the legacy control (no obs_basis draws), so compare
# the two runs on the flagship's fdot/truth trajectory instead.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${HERE}/submit_gf_highf_grid.sh"
[[ -f "${BASE}" ]] || { echo "missing ${BASE}" >&2; exit 1; }

# STRICT ARGUMENT VALIDATION. The version of this script before the
# centering arm existed took only $1 and $2, so "A obs noctr" run against
# it SILENTLY DROPPED the third argument and routed the job into the
# ctr arm's store -- two MPI jobs appending to one gzip HDF5 with
# HDF5_USE_FILE_LOCKING=FALSE, which tore the file and killed both runs
# with "filter returned failure during read". A script that quietly
# ignores an argument it does not understand is the bug generator; refuse
# instead.
if [[ $# -gt 3 ]]; then
  echo "ERROR: too many arguments ($#). usage: $0 {A|B} {obs|leg} [ctr|noctr]" >&2
  exit 2
fi
PROBE="${1:-A}"
MODE="${2:-obs}"
CTR="${3:-ctr}"

case "${PROBE}" in
  A|a|highf)
    # layers 144.5 -> 149.5; the flagship alone. Leaf budget 20 is the
    # proven high-f value: one real source plus room for the split modes.
    export GB_MIN_FREQ=2.006944e-02
    export GB_MAX_FREQ=2.076389e-02
    export GB_NLEAVES_MAX=20
    TAG=A_highf ;;
  B|b|lowf)
    # layers 44.5 -> 49.5; 66 detectable sources. The 20-leaf budget from
    # probe A would BIND here and turn a neutrality test into a cap test,
    # which is the one thing that would make B unreadable -- 66 detectable
    # plus headroom for sub-threshold sources and transient duplicates.
    export GB_MIN_FREQ=6.180556e-03
    export GB_MAX_FREQ=6.875000e-03
    export GB_NLEAVES_MAX=200
    TAG=B_lowf ;;
  *) echo "usage: $0 {A|B} {obs|leg}" >&2; exit 2 ;;
esac

case "${MODE}" in
  obs|observable) export GB_INMODEL_PROPOSAL=observable ;;
  leg|legacy)     export GB_INMODEL_PROPOSAL=legacy ;;
  *) echo "usage: $0 {A|B} {obs|leg} [ctr|noctr]" >&2; exit 2 ;;
esac

# ---- F-STAT CENTERING ARM ------------------------------------------
# "noctr" turns OFF the F-stat distance/amplitude centering and lets the
# birth fall back to the usual phase-maximised d_h/h_h amplitude pin
# (rj_amp_maximize), i.e. plain phase maximisation in the likelihood as
# search/RJ has always done.
#
# WHY IT IS WORTH A PROBE. GB_RJ_FSTAT_DIST_BIRTH=0 gates the ENTIRE
# centers chain, not just the draw -- and in v7 rj_fstat_centers was
# 1407.5 s of the 1961.8 s rj_fstat_search move, i.e. 54% OF EVERY
# ITERATION, comfortably the single largest cost in the run. The question
# this arm answers is whether that 54% is still buying anything now that
# the grid places births on the ridge (fdot axis) and the in-model move
# refines them along it: if recovery holds without centering, the run gets
# roughly twice as fast for free.
#
# The knob normally FOLLOWS rj_amp_maximize, so it must be pinned in both
# arms -- leaving it unset in the "ctr" arm would make the comparison
# depend on a per-move default rather than on this switch.
case "${CTR}" in
  ctr)   export GB_RJ_FSTAT_DIST_BIRTH=1 ;;
  noctr) export GB_RJ_FSTAT_DIST_BIRTH=0 ;;
  *) echo "usage: $0 {A|B} {obs|leg} [ctr|noctr]" >&2; exit 2 ;;
esac

# BOTH new paths ARMED. These are now the code defaults; pinned anyway so
# the probe cannot change meaning if a default is revisited, and so the
# knob diff names what is under test.
#   1. the observable-basis IN-MODEL proposal (set per-arm above), and
#   2. the fdot-axis F-STAT GRID: fdot becomes a first-class grid axis
#      instead of the r = 0 manifold the grid searches today. The two are
#      complementary -- the grid places births on the ridge, the in-model
#      move refines them along it -- so shipping one alone can read as "no
#      effect". Attribution comes from DISJOINT observables, chiefly the
#      fraction of births with fdot < 0, which the old grid cannot produce
#      at all.
# NOTE: the fdot grid REFUSES an *_peaks_stacked.npz fitted in the Mc
# basis, by design. These probes use fresh stores, so they refit.
export FSTAT_FDOT_AXIS=1
export FSTAT_FDOT_RATIO_MAX=5.0
# The four in-model tuning knobs pinned at their defaults, exactly as v8
# pins them, so a probe result transfers without a second variable.
export GB_INMODEL_OBSERVABLE_FIBER_WEIGHT=0.0
export GB_INMODEL_OBSERVABLE_JUMP=1.0
export GB_INMODEL_OBSERVABLE_MC_STEP=0.05
export GB_INMODEL_OBSERVABLE_SHEAR=0.5
# The live detailed-balance check: recomputes the log-Jacobian INDEPENDENTLY
# of the code that produced it and prints MISMATCH on disagreement. One
# traced source per repeat -- negligible, and it is the only in-production
# check of the one term in this path that can be silently wrong.
export GB_INMODEL_TRACE=1
# Same diagnostics in every probe, so wall-clock stays comparable.
export GB_CAP_DIAG=1

# Separate store per (probe, mode): four runs, four stores, no resume
# collisions. A shared store would silently RESUME the other arm's chain,
# which reads as "the change did nothing".
# OBS_PROBE_ROOT exists so this file can be dry-run off-cluster (the default
# is a read-only mount on a laptop) -- do not point a real probe elsewhere
# without also moving the sbatch --output below.
: "${OBS_PROBE_ROOT:=/shared/data/global_fit_output}"
# The centering arm is part of the store name for "noctr" ONLY. The
# "ctr" arm keeps the original unsuffixed name deliberately, so
# re-submitting it RESUMES the runs already in flight rather than starting
# a fresh store beside them; "noctr" gets its own store so it cannot
# resume into them, which would read as "the change did nothing".
_ARM="${TAG}_${MODE}"
[[ "${CTR}" == "noctr" ]] && _ARM="${_ARM}_noctr"
export STORE_DIR="${OBS_PROBE_ROOT}/gf_obsprobe_${_ARM}/"
export BASE_FILE_NAME="gf_obsprobe_${_ARM}"
# 4 h of allocation, but the READ is at 20-30 minutes: at ~70-200 s per
# iteration that is roughly 8-25 iterations, and the flagship sat static
# for 19, so a walk should already be visible by then. The extra hours
# cost nothing on a spot partition and mean a promising arm can simply
# keep going instead of being resubmitted. Resume-safe either way.
export NUM_ITERATIONS=200

# ---- ONE WRITER PER STORE -------------------------------------------
# The real protection, and it does not depend on getting the naming right:
# refuse to submit if a job for this exact arm is already queued or
# running. Two jobs sharing a store corrupt it in minutes and the failure
# surfaces later, in the saver rank, looking like an HDF5 bug rather than
# a duplicate launch.
if command -v squeue >/dev/null 2>&1; then
  _live="$(squeue -h -u "${USER}" -o '%i %j %T' 2>/dev/null \
           | awk -v n="obs_${_ARM}" '$2 == n {print $1" "$3}')"
  if [[ -n "${_live}" ]]; then
    echo "REFUSING TO SUBMIT: a job for this arm is already active:" >&2
    echo "    ${_live}" >&2
    echo "  store: ${STORE_DIR}" >&2
    echo "  Two jobs writing one gzip HDF5 tear it -- that is what killed" >&2
    echo "  the first high-f probes. Cancel it first, or pick another arm." >&2
    exit 3
  fi
fi

mkdir -p "${STORE_DIR}"

cat <<EOF
=== observable-basis probe ${TAG} / ${MODE} / ${CTR} ===
  band            ${GB_MIN_FREQ} - ${GB_MAX_FREQ} Hz
  leaves          ${GB_NLEAVES_MAX}
  proposal        ${GB_INMODEL_PROPOSAL}
  fstat centering ${GB_RJ_FSTAT_DIST_BIRTH}  (1 = on, 0 = phase-max pin)
  store           ${STORE_DIR}
  iterations      ${NUM_ITERATIONS}
  base script     ${BASE}
EOF

# ---- SUBMIT -----------------------------------------------------------
# These flags OVERRIDE the #SBATCH header inside the base script (which
# asks for gpu-80-spot / 24 h): sbatch takes command line > environment >
# header. That is legal but it means the file does not read like what
# runs, so the resolved command is echoed below and the partition is
# checked to exist first -- a bad partition otherwise shows up as a job
# sitting in the wrong place, or with no GPU, long after submit.
: "${OBS_PROBE_PARTITION:=gpu-40-spot}"
: "${OBS_PROBE_TIME:=04:00:00}"

if command -v sinfo >/dev/null 2>&1; then
  if ! sinfo -h -p "${OBS_PROBE_PARTITION}" -o '%P' 2>/dev/null | grep -q .; then
    echo "ERROR: partition '${OBS_PROBE_PARTITION}' does not exist here." >&2
    echo "  available:" >&2
    sinfo -h -o '    %P  (%a, %D nodes, gres=%G)' 2>/dev/null | sort -u >&2
    echo "  set OBS_PROBE_PARTITION=<name> to choose one." >&2
    exit 4
  fi
fi

# SBATCH_* env vars outrank the #SBATCH header and would silently retarget
# this job; the explicit flags below outrank them in turn, but an inherited
# one is worth seeing rather than guessing about.
for _v in SBATCH_PARTITION SBATCH_GRES SBATCH_TIMELIMIT SBATCH_ACCOUNT; do
  [[ -n "${!_v:-}" ]] && echo "  NOTE: ${_v}=${!_v} is set in your environment"
done

set -x
exec sbatch --job-name="obs_${_ARM}" \
  --partition="${OBS_PROBE_PARTITION}" \
  --gres=gpu:1 \
  --time="${OBS_PROBE_TIME}" \
  --output="${OBS_PROBE_ROOT}/obs_${_ARM}_%j.log" \
  --export=ALL "${BASE}"
