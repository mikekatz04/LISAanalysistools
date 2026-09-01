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
#   ./submit_gf_obs_probe.sh A obs     # high-f, observable   <- the fix
#   ./submit_gf_obs_probe.sh A leg     # high-f, legacy       <- its control
#   ./submit_gf_obs_probe.sh B obs     # low-f,  observable   <- neutrality
#   ./submit_gf_obs_probe.sh B leg     # low-f,  legacy       <- its control
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

PROBE="${1:-A}"
MODE="${2:-obs}"

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
  *) echo "usage: $0 {A|B} {obs|leg}" >&2; exit 2 ;;
esac

# The four tuning knobs pinned at their defaults, exactly as v8 pins them,
# so a probe result transfers without a second variable.
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
export STORE_DIR="${OBS_PROBE_ROOT}/gf_obsprobe_${TAG}_${MODE}/"
export BASE_FILE_NAME="gf_obsprobe_${TAG}_${MODE}"
# Enough to see the flagship walk (it sat static for 19) without burning a
# 24 h allocation on a question answered in the first 50.
export NUM_ITERATIONS=200

mkdir -p "${STORE_DIR}"

cat <<EOF
=== observable-basis probe ${TAG} / ${MODE} ===
  band            ${GB_MIN_FREQ} - ${GB_MAX_FREQ} Hz
  leaves          ${GB_NLEAVES_MAX}
  proposal        ${GB_INMODEL_PROPOSAL}
  store           ${STORE_DIR}
  iterations      ${NUM_ITERATIONS}
  base script     ${BASE}
EOF

exec sbatch --job-name="obs_${TAG}_${MODE}" \
  --output="${OBS_PROBE_ROOT}/obs_${TAG}_${MODE}_%j.log" \
  --export=ALL "${BASE}"
