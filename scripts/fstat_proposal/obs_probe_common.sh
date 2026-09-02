# ===========================================================================
# Shared configuration for the four observable-basis A/B probe arms.
# SOURCED by submit_gf_obs_{A,B}_{ctr,noctr}.sh -- not submitted directly.
#
# Everything here is IDENTICAL across all four arms on purpose. The arms
# differ only in band (A vs B), leaf budget, and F-stat centering (ctr vs
# noctr); if a knob lives in an arm file instead of here, it silently
# becomes a second variable and the comparison stops being attributable.
# ===========================================================================

# ---- THE CHANGE UNDER TEST -------------------------------------------
# 1. Observable-basis in-model proposal. Steps are drawn in
#    z = [lnA, f_mid, fdot, phi0, cos_iota, psi, alpha, sin_delta, Mc]
#    and mapped back; the SAMPLING basis is untouched. On the real
#    flagship Fisher the legacy draw walks an f0-fdot ridge of slope
#    -0.898 T where the chirp geometry demands -T/2, and that excess lands
#    as 0.170 bins of spurious f_mid motion per fdot step -- about 14
#    sigma at rho = 46. Set per-arm (all four currently "observable");
#    =legacy reverts bit-identically at the same seed.
: "${GB_INMODEL_PROPOSAL:=observable}"
export GB_INMODEL_PROPOSAL

# 2. fdot as a FIRST-CLASS F-stat grid axis, replacing the r = 0 manifold
#    the grid searches today. v7 measured 39.6% of low-f and 10.5% of
#    high-f alive leaves carrying fdot < 0, which the old grid CANNOT
#    represent. Also costs fewer nodes (20.38 mHz: 71 -> 53) over 10x the
#    range. A cache fitted in the Mc basis is REFUSED on load by design,
#    so these arms must start from fresh stores.
export FSTAT_FDOT_AXIS=1
export FSTAT_FDOT_RATIO_MAX=5.0

# ---- in-model tuning, pinned at the defaults v8 will use --------------
export GB_INMODEL_OBSERVABLE_FIBER_WEIGHT=0.0
export GB_INMODEL_OBSERVABLE_JUMP=1.0
export GB_INMODEL_OBSERVABLE_MC_STEP=0.05
export GB_INMODEL_OBSERVABLE_SHEAR=0.5

# ---- diagnostics -----------------------------------------------------
# The live detailed-balance check: recomputes the log-Jacobian
# INDEPENDENTLY of the code that produced it and prints MISMATCH on
# disagreement. One traced source per repeat -- negligible, and the only
# in-production check of the one term in this path that can be silently
# wrong.
export GB_INMODEL_TRACE=1
# Same diagnostics in every arm, so wall-clock stays comparable.
export GB_CAP_DIAG=1

# ---- run shape -------------------------------------------------------
# 4 h of allocation, but the READ is at 20-30 minutes: at ~70-200 s per
# iteration that is 8-25 iterations, and the flagship sat static for 19,
# so a walk should already be visible. The extra hours cost nothing on a
# spot partition and let a promising arm keep going. Resume-safe.
export NUM_ITERATIONS=200

echo "=== observable-basis probe arm ==="
echo "  store           ${STORE_DIR}"
echo "  band            ${GB_MIN_FREQ} - ${GB_MAX_FREQ} Hz"
echo "  leaves          ${GB_NLEAVES_MAX}"
echo "  in-model        ${GB_INMODEL_PROPOSAL}"
echo "  fdot grid axis  ${FSTAT_FDOT_AXIS}"
echo "  fstat centering ${GB_RJ_FSTAT_DIST_BIRTH}  (1 = on, 0 = phase-max pin)"
echo "  iterations      ${NUM_ITERATIONS}"
echo "  slurm job       ${SLURM_JOB_ID:-<none>} on ${SLURM_JOB_PARTITION:-<none>}"
echo "  gres            ${SLURM_JOB_GRES:-<none>}"

# ONE WRITER PER STORE. Two jobs appending to one gzip HDF5 with file
# locking disabled tear it, and the tear only surfaces on the next READ --
# which is how the first high-f probes died with "filter returned failure
# during read", looking like an HDF5 bug rather than a duplicate launch.
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
