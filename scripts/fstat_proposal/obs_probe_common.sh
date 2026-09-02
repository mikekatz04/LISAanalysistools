# ===========================================================================
# Shared configuration for the observable-basis probe arms — V8 PARITY.
# SOURCED by submit_gf_obs_{A,B}_{ctr,noctr}.sh — not submitted directly.
#
# RULE (user, 2026-09-02): besides the band, the leaf budget, and the RJ
# phase-maximization knob below, these probes must EXACTLY represent
# submit_gf_3mo_v8.sh. Everything here was derived from an export-line
# diff against that script; OBS_PROBE_V8_PARITY=1 tells the base run
# script to stand down its probe-era values so these stand.
#
# Declared, unavoidable deltas from v8 (a probe cannot be a production
# run): GB_ONLY composition + injection PSD (no noise/vgb sampling), the
# confined band + band-scaled leaf budget (arm scripts), NUM_ITERATIONS,
# and GB_INMODEL_TRACE=1 (diagnostic only — traces one source per repeat).
# ===========================================================================
export OBS_PROBE_V8_PARITY=1

# ---- THE ONE EXPERIMENTAL KNOB BEYOND v8 (user ruling 2026-09-02) -----
# RJ PHASE MAXIMIZATION ON for search scoring. History, so this does not
# get lost again: in v7-style search, birth phase maximization came via
# the F-STAT CENTERS path (extrinsics pinned at the F-stat maximizers).
# v8 turns centering off (GB_RJ_FSTAT_DIST_BIRTH=0, c0bd6cf7) — which
# SILENTLY strips that maximization too, because GB_RJ_PHASE_MAXIMIZE
# defaults 0 and rj_amp_maximize follows it. The first probe campaign
# therefore ran births with NO maximization at all. This arms the
# in-likelihood two-quadrature phase max on the search RJ moves.
export GB_RJ_PHASE_MAXIMIZE=1
# Amplitude maximization DEFAULTS TO FOLLOW phase_maximize, so with the
# line above it would silently arm too. User ruling: phase max only, no
# amp max for now — pin it off explicitly.
export GB_RJ_AMP_MAXIMIZE=0

# ---- v8 values that differ from the base script's probe-era defaults --
# (base lines are OBS_PROBE_V8_PARITY-guarded; values verbatim from
# submit_gf_3mo_v8.sh)
export FSTAT_PEAK_WEIGHT_CELLS=1
export GB_CAP_DIVISOR=1
export GB_CAP_OVERLAP_FRAC=0
export GB_INFOMAT_MEMPOOL_FREE=0
export GB_INMODEL_REPEATS_NEWBORN=250
export GB_INMODEL_SETUP_BATCH=4096
export GB_LEAF_CAP_MIN_ITERS=4
export GF_MOVE_TIMING_SYNC=0
export GB_PROP_TIMING_SYNC=0
export GB_RJ_LIVE_CAP_PICK=0
# v8 leaves these four UNSET (recipe defaults / auto); the parity guard
# keeps the base script from exporting its probe-era values, so nothing
# needs setting here: FSTAT_N_MC (auto; ignored under the fdot axis
# anyway), FSTAT_PEAKS_TO_FIT (uncapped), GB_RJ_FLIP_FRACTION (search 0.2
# by recipe default, PE 0.1 — a global env export would flatten PE too),
# GB_RJ_BAND_SHUTOFF_AFTER (legacy alias; canonical knobs below).

# ---- knobs v8 sets that the probe stack never did (verbatim) ----------
export GB_BAND_UNIT_DIR_PER_WALKER=1
export GB_BAND_UNIT_START_PER_WALKER=1
export GB_BAND_WINDOW_STRICT=1
export GB_CAP_CELL_MAX=20
export GB_CAP_DEST_BAND=1
export GB_CAP_DRIFT_GATE_EDGE_LEAK=1
export GB_CAP_INMODEL_HEADROOM=0
export GB_CELL_LABEL_DEFERRED=1
export GB_FSTAT_CTR_BATCH=4096
export GB_FSTAT_FOLD=1
export GB_FSTAT_NM_LANE_WEIGHTS=
export GB_FSTAT_PERROW_UNIT_CACHE=1
export GB_INMODEL_ACCEPT_KERNEL=0
export GB_INMODEL_BATCH_MEMPOOL_FREE=0
export GB_ORTHO_CHECK=1
export GB_PHASE_MAX_FUSED=1
export GB_REPLACE_PHASE_MAX=auto
export GB_RJ_BAND_SHUTOFF_ITERS=5
export GB_RJ_BAND_SHUTOFF_RESET_ITERS=100
export GB_SEARCH_CAP_QUIESCENT=1
# v8 386e3855: the SEARCH replace move is OFF (the PE replace,
# GB_PE_RJ_REPLACE, stays at its default ON — that is v8's config too).
export GB_SEARCH_RJ_REPLACE=0
# PE replace also OFF (user ruling 2026-09-02; v8 pins the same) -- it
# registered but never proposed in the r2 probes.
export GB_PE_RJ_REPLACE=0
export GB_TEMPER_EVERY_PROPOSES=1
export GB_TEMPER_VERTICAL=1

# ---- the changes under test (identical in v8; pinned so the run cannot
#      change meaning if a code default is revisited) -------------------
export GB_INMODEL_PROPOSAL=observable
export FSTAT_FDOT_AXIS=1
export FSTAT_FDOT_RATIO_MAX=5.0
export GB_INMODEL_OBSERVABLE_FIBER_WEIGHT=0.0
export GB_INMODEL_OBSERVABLE_JUMP=1.0
export GB_INMODEL_OBSERVABLE_MC_STEP=0.05
export GB_INMODEL_OBSERVABLE_SHEAR=0.5
# Centering: v8 pins 0 (c0bd6cf7, the probe verdict). Arm scripts may
# preset 1 to run the DELIBERATE-DEVIATION ctr arm; unset falls to v8's 0.
: "${GB_RJ_FSTAT_DIST_BIRTH:=0}"
export GB_RJ_FSTAT_DIST_BIRTH

# ---- probe-only diagnostics (declared deltas) -------------------------
# Live detailed-balance check: recomputes the log-Jacobian independently
# per traced proposal and prints MISMATCH on disagreement. One source per
# repeat — negligible, and the only in-production check of the one term
# that can be silently wrong.
export GB_INMODEL_TRACE=1
export GB_CAP_DIAG=1

# ---- run shape --------------------------------------------------------
# 4 h allocation; the READ is at 20–30 min. Resume-safe.
export NUM_ITERATIONS=200

echo "=== observable-basis probe arm (v8 parity, r2) ==="
echo "  store           ${STORE_DIR}"
echo "  band            ${GB_MIN_FREQ} - ${GB_MAX_FREQ} Hz"
echo "  leaves          ${GB_NLEAVES_MAX}"
echo "  in-model        ${GB_INMODEL_PROPOSAL}"
echo "  fdot grid axis  ${FSTAT_FDOT_AXIS}"
echo "  rj phase max    ${GB_RJ_PHASE_MAXIMIZE}  (amp max ${GB_RJ_AMP_MAXIMIZE})"
echo "  fstat centering ${GB_RJ_FSTAT_DIST_BIRTH}  (v8 = 0)"
echo "  search replace  ${GB_SEARCH_RJ_REPLACE}  (v8 = 0)"
echo "  slurm job       ${SLURM_JOB_ID:-<none>} on ${SLURM_JOB_PARTITION:-<none>}"

# ONE WRITER PER STORE. Two jobs appending to one gzip HDF5 with file
# locking disabled tear it, and the tear only surfaces on the next READ —
# how the first probe campaign lost both high-f runs.
if command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _dup="$(squeue -h -u "${USER}" -o '%i %j %T' 2>/dev/null \
          | awk -v n="${SLURM_JOB_NAME}" -v me="${SLURM_JOB_ID}" \
                '$2 == n && $1 != me && $3 == "RUNNING" {print $1}')"
  if [[ -n "${_dup}" ]]; then
    echo "REFUSING TO RUN: job(s) ${_dup} are already running this arm" >&2
    echo "  and would share ${STORE_DIR}. Cancel one." >&2
    exit 3
  fi
fi
