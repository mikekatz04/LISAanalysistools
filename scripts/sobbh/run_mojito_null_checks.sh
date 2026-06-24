#!/usr/bin/env bash
#
# Re-run the full_year global-fit null-template inner-product check for EVERY
# mojito L1 source that is present on disk (EMRI / MBHB / SOBHB), ONE SOURCE AT
# A TIME. For each source it builds the branch's template at the exact catalogue
# injection (factor=0), no injected noise, and reports the inner products that
# feed the likelihood:  <d|d>, <h|h>, <d|h>, <r|r>, overlap, mismatch,
# <r|r>/<d|d>, and the source term -0.5<r|r>.
#
# Source files are auto-discovered from <MOJITO_DATA_PATH>/data/<CLASS>/L1/
# (filenames carry "source<N>"). The worker is scripts/sobbh/mojito_null_check.py.
#
# Usage:
#   ./run_mojito_null_checks.sh                 # all available EMRI/MBHB/SOBHB
#   ./run_mojito_null_checks.sh EMRI MBHB       # only these classes
#   FULL_TOBS=1 ./run_mojito_null_checks.sh     # EMRI/SOBHB over the FULL dataset
#   TOBS_MONTHS=6 ./run_mojito_null_checks.sh   # EMRI/SOBHB 6-month window
#   DRY_RUN=1 ./run_mojito_null_checks.sh       # just list what would run
#
# Windowing (defaults = exactly the runs done by hand on 2026-06-19):
#   MBHB        -> merger-centered chopped window (CHOP_WINDOW=1); FULL_TOBS N/A.
#   EMRI/SOBHB  -> TOBS_MONTHS-month window from the data start (default 3 mo,
#                 to save memory). Set FULL_TOBS=1 to use the entire data file
#                 (~731 days, span auto-detected from the L1 filename) -- this
#                 builds a large WDM grid, so watch memory (the worker aborts
#                 via its RSS watchdog if it blows up).
# Overridable env: LAT_DIR, PY, MOJITO_DATA_PATH, LOGDIR, TOBS_MONTHS,
#   FULL_TOBS, OMP_NUM_THREADS, DRY_RUN. GB/VGB are skipped (no branch here).
set -uo pipefail

LAT_DIR="${LAT_DIR:-/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools}"
PY="${PY:-/Users/mkatz/miniconda3/envs/deving/bin/python}"
MOJITO_DATA_PATH="${MOJITO_DATA_PATH:-/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/}"
NULL="$LAT_DIR/scripts/sobbh/mojito_null_check.py"
LOGDIR="${LOGDIR:-/tmp/mojito_null_logs}"
TOBS_MONTHS="${TOBS_MONTHS:-3}"
YRSID=31558149.763545603
TOBS_TARGET_VAL=$(awk "BEGIN{printf \"%.6f\", ${TOBS_MONTHS}/12.0*${YRSID}}")

CLASSES=("$@"); [ ${#CLASSES[@]} -eq 0 ] && CLASSES=(EMRI MBHB SOBHB)

if [ "${FULL_TOBS:-0}" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "############################################################################"
  echo "!! FULL_TOBS=1: EMRI/SOBHB will use the FULL ~731-day data file."
  echo "!! That is a VERY large WDM grid and may exhaust this machine's memory."
  echo "!! The worker aborts if RSS exceeds NULL_CHECK_MEM_GB (default ${NULL_CHECK_MEM_GB:-24} GB)."
  echo "!! Prefer the default 3-month window, or TOBS_MONTHS=<n>, unless you are sure."
  echo "############################################################################"
fi

mkdir -p "$LOGDIR"
if command -v brew >/dev/null 2>&1; then
  export PKG_CONFIG_PATH="$(brew --prefix lapack 2>/dev/null)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
export DATA_PROCESSOR=mojito MOJITO_DATA_PATH NWALKERS=1 NTEMPS=1 \
       OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
       NULL_CHECK_MEM_GB="${NULL_CHECK_MEM_GB:-24}"

# class dir -> the settings env var that selects its source ids
# (a function, not a `declare -A` -- macOS ships bash 3.2 with no assoc arrays)
envvar_for() {
  case "$1" in
    EMRI)  echo EMRI_IDS ;;
    MBHB)  echo MBHB_IDS ;;
    SOBHB) echo SOBHB_IDS ;;
    *)     echo "" ;;
  esac
}

ROWS=()
get_val() { sed -n "s/.* $2=\([^ ]*\).*/\1/p" <<<"$1"; }

for CLASS in "${CLASSES[@]}"; do
  L1="$MOJITO_DATA_PATH/data/$CLASS/L1"
  if [ ! -d "$L1" ]; then echo "[skip] $CLASS: no $L1"; continue; fi
  IDS=$(ls "$L1"/*.h5 2>/dev/null | grep -oE 'source[0-9]+' | grep -oE '[0-9]+' | sort -n -u)
  if [ -z "$IDS" ]; then echo "[skip] $CLASS: no source files in $L1"; continue; fi
  echo "[found] $CLASS ids: $(echo $IDS | tr '\n' ' ')"
  # full-dataset span for EMRI/SOBHB (days parsed from the L1 filename, e.g. "731d")
  SPAN_DAYS=$(ls "$L1"/*.h5 2>/dev/null | head -1 | grep -oE '[0-9]+d' | grep -oE '[0-9]+' | head -1)
  [ -z "$SPAN_DAYS" ] && SPAN_DAYS=731
  FULL_SPAN_VAL=$(awk "BEGIN{printf \"%.6f\", ${SPAN_DAYS}*86400.0}")

  for ID in $IDS; do
    echo "=================================================================="
    echo ">>> running $CLASS id=$ID  ($(date '+%H:%M:%S'))"
    # one active class at a time
    export EMRI_IDS="" MBHB_IDS="" SOBHB_IDS=""
    EV=$(envvar_for "$CLASS")
    export "$EV=$ID"
    if [ "$CLASS" = "MBHB" ]; then
      # MBH: always the merger-centered chopped window (a full-span MBH window is
      # pointless -- the signal is a short merger). FULL_TOBS does not apply.
      export CHOP_WINDOW=1; unset TOBS_TARGET 2>/dev/null || true
    elif [ "${FULL_TOBS:-0}" = "1" ]; then
      # EMRI/SOBHB over the FULL dataset (~${SPAN_DAYS} days). WARNING: large WDM
      # grid -> high memory; the worker's RSS watchdog will abort if it exceeds.
      export CHOP_WINDOW=0 TOBS_TARGET="$FULL_SPAN_VAL"
    else
      # EMRI/SOBHB over a TOBS_MONTHS-month window from the data start (default).
      export CHOP_WINDOW=0 TOBS_TARGET="$TOBS_TARGET_VAL"
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
      if [ "$CLASS" = "MBHB" ]; then WIN="merger-centered chop"
      elif [ "${FULL_TOBS:-0}" = "1" ]; then WIN="FULL ${SPAN_DAYS}d from start"
      else WIN="${TOBS_MONTHS}mo from start"; fi
      echo "  [dry-run] would run worker with $EV=$ID  window=$WIN"
      ROWS+=("$(printf '%-6s %-3s %-10s %-12s %-11s %-11s %-13s %-6s' "$CLASS" "$ID" DRYRUN - - - - -)")
      continue
    fi
    LOG="$LOGDIR/${CLASS}_id${ID}.log"
    rm -f "$LAT_DIR/gf_output/full_year_combined_run_testing.h5" 2>/dev/null
    ( cd "$LAT_DIR" && "$PY" "$NULL" ) > "$LOG" 2>&1
    RES=$(grep '^\[RESULT\]' "$LOG" | tail -1)
    if [ -z "$RES" ]; then
      echo "  !! FAILED (see $LOG):"; tail -4 "$LOG" | sed 's/^/     /'
      ROWS+=("$(printf '%-6s %-3s %-10s %-12s %-11s %-11s %-13s %-6s' "$CLASS" "$ID" FAILED - - - - -)")
    else
      printf '  %s\n' "$RES"
      ROWS+=("$(printf '%-6s %-3s %-10s %-12s %-11s %-11s %-13s %-6s' \
        "$CLASS" "$ID" \
        "$(get_val "$RES" data_snr)" "$(get_val "$RES" overlap)" \
        "$(get_val "$RES" mismatch)" "$(get_val "$RES" rr_over_dd)" \
        "$(get_val "$RES" source_logL)" "$(get_val "$RES" xcheck)")")
    fi
  done
done

echo
echo "================================  SUMMARY  ================================"
printf '%-6s %-3s %-10s %-12s %-11s %-11s %-13s %-6s\n' \
  CLASS ID dataSNR overlap mismatch rr/dd src_logL xchk
printf '%s\n' "${ROWS[@]}"
echo "=========================================================================="
echo "logs: $LOGDIR   worker: $NULL"
