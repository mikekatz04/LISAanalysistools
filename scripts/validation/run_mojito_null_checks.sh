#!/usr/bin/env bash
#
# Run the mojito-fidelity checks for EVERY source class present on disk, one
# source at a time, through the CURRENT stock setups:
#
#   EMRI / MBHB / SOBHB -> scripts/validation/mojito_null_check.py: the stock
#       ``erebor.full_year_combined`` pipeline (phentax MBH, TDI-on-the-fly
#       SOBBH, legacy EMRI — the branch's engine-side ``SourceSignalGen``),
#       template built at the exact catalogue injection; reports the inner
#       products feeding the likelihood: <d|d>, <h|h>, <d|h>, <r|r>, overlap,
#       mismatch, <r|r>/<d|d>, source term -0.5<r|r>. Null <=> <r|r>/<d|d> -> 0.
#   GB   -> scripts/gb/gb_mojito_match.py: the GB_TOPN (default 3) HIGHEST
#       (highest-frequency, band-pass-isolatable) galactic binaries from the
#       whole-galaxy stream vs the GBTDIonTheFly template (validated
#       catalogue-injection convention: params @ REF, phi0 = +TrueAnomaly).
#   VGB  -> scripts/gb/vgb_mojito_match.py: the VGB_TOPN (default 2) highest
#       verification binaries from the VGB stream (single-source-clean).
#
# Source files are auto-discovered from <MOJITO_DATA_PATH>/data/<CLASS>/L1/
# (filenames carry "source<N>"; GB/VGB are single summed streams).
#
# Usage:
#   ./run_mojito_null_checks.sh                 # all: EMRI MBHB SOBHB GB VGB
#   ./run_mojito_null_checks.sh EMRI MBHB       # only these classes
#   FULL_TOBS=1 ./run_mojito_null_checks.sh     # EMRI/SOBHB over the FULL dataset
#   TOBS_MONTHS=6 ./run_mojito_null_checks.sh   # EMRI/SOBHB 6-month window
#   GB_TOPN=5 VGB_TOPN=3 ./run_mojito_null_checks.sh GB VGB
#   DRY_RUN=1 ./run_mojito_null_checks.sh       # just list what would run
#
# Windowing (defaults = the validated hand-run configurations):
#   MBHB        -> merger-centered chopped window (CHOP_WINDOW=1); FULL_TOBS N/A.
#   EMRI/SOBHB  -> TOBS_MONTHS-month window from the data start (default 3 mo,
#                 to save memory). Set FULL_TOBS=1 to use the entire data file
#                 (~731 days, span auto-detected from the L1 filename) -- this
#                 builds a large WDM grid, so watch memory (the worker aborts
#                 via its RSS watchdog if it blows up).
#   GB          -> GB_DAYS-day window (gb_mojito_match.py default 365 d).
#   VGB         -> GB_DAYS-day window (vgb_mojito_match.py default 90 d).
# Overridable env: LAT_DIR, PY, MOJITO_DATA_PATH, LOGDIR, TOBS_MONTHS,
#   FULL_TOBS, OMP_NUM_THREADS, DRY_RUN, GB_TOPN, VGB_TOPN, GB_DAYS,
#   GB_BAND_UHZ, NULL_CHECK_MEM_GB.
set -uo pipefail

LAT_DIR="${LAT_DIR:-/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools}"
PY="${PY:-/Users/mkatz/miniconda3/envs/deving/bin/python}"
MOJITO_DATA_PATH="${MOJITO_DATA_PATH:-/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/}"
NULL="$LAT_DIR/scripts/validation/mojito_null_check.py"
GB_MATCH="$LAT_DIR/scripts/gb/gb_mojito_match.py"
VGB_MATCH="$LAT_DIR/scripts/gb/vgb_mojito_match.py"
LOGDIR="${LOGDIR:-/tmp/mojito_null_logs}"
TOBS_MONTHS="${TOBS_MONTHS:-3}"
GB_TOPN_WANT="${GB_TOPN:-3}"
VGB_TOPN_WANT="${VGB_TOPN:-2}"
YRSID=31558149.763545603
TOBS_TARGET_VAL=$(awk "BEGIN{printf \"%.6f\", ${TOBS_MONTHS}/12.0*${YRSID}}")

CLASSES=("$@"); [ ${#CLASSES[@]} -eq 0 ] && CLASSES=(EMRI MBHB SOBHB GB VGB)

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
       OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
       NULL_CHECK_MEM_GB="${NULL_CHECK_MEM_GB:-24}"

# class dir -> the env var that selects its source ids for the stock variant
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

# --- GB / VGB: per-source fidelity vs the summed stream (top-N ranked) ------
run_gb_class() {
  local CLASS="$1" SCRIPT="$2" TOPN="$3"
  local L1="$MOJITO_DATA_PATH/data/$CLASS/L1"
  if [ ! -d "$L1" ] || [ -z "$(ls "$L1"/*.h5 2>/dev/null)" ]; then
    echo "[skip] $CLASS: no L1 stream under $L1"; return
  fi
  echo "=================================================================="
  echo ">>> running $CLASS top-$TOPN fidelity  ($(date '+%H:%M:%S'))"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  [dry-run] would run $SCRIPT with GB_TOPN=$TOPN"
    ROWS+=("$(printf '%-6s %-4s %-10s %-12s %-11s %-11s %-13s %-6s' "$CLASS" - DRYRUN - - - - -)")
    return
  fi
  local LOG="$LOGDIR/${CLASS}_top${TOPN}.log"
  ( cd "$LAT_DIR" && GB_TOPN="$TOPN" "$PY" "$SCRIPT" ) > "$LOG" 2>&1
  local RES_LINES
  RES_LINES=$(grep '^\[RESULT\]' "$LOG")
  if [ -z "$RES_LINES" ]; then
    echo "  !! FAILED (see $LOG):"; tail -4 "$LOG" | sed 's/^/     /'
    ROWS+=("$(printf '%-6s %-4s %-10s %-12s %-11s %-11s %-13s %-6s' "$CLASS" - FAILED - - - - -)")
    return
  fi
  while IFS= read -r RES; do
    printf '  %s\n' "$RES"
    # columns: CLASS ID dataSNR overlap mismatch rr/dd src_logL xchk
    # (for GB/VGB: ID=rank, dataSNR=det_snr, rr/dd=f0[mHz], src_logL=mm@tau*)
    ROWS+=("$(printf '%-6s %-4s %-10s %-12s %-11s %-11s %-13s %-6s' \
      "$CLASS" "$(get_val "$RES" rank)" \
      "$(get_val "$RES" det_snr)" "$(get_val "$RES" overlap)" \
      "$(get_val "$RES" mismatch)" "$(get_val "$RES" f0_mhz)" \
      "$(get_val "$RES" mm_taustar)" "-")")
  done <<<"$RES_LINES"
}

for CLASS in "${CLASSES[@]}"; do
  if [ "$CLASS" = "GB" ];  then run_gb_class GB  "$GB_MATCH"  "$GB_TOPN_WANT";  continue; fi
  if [ "$CLASS" = "VGB" ]; then run_gb_class VGB "$VGB_MATCH" "$VGB_TOPN_WANT"; continue; fi

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
      ROWS+=("$(printf '%-6s %-4s %-10s %-12s %-11s %-11s %-13s %-6s' "$CLASS" "$ID" DRYRUN - - - - -)")
      continue
    fi
    LOG="$LOGDIR/${CLASS}_id${ID}.log"
    rm -f "$LAT_DIR/gf_output/full_year_combined_run_testing.h5" 2>/dev/null
    ( cd "$LAT_DIR" && "$PY" "$NULL" ) > "$LOG" 2>&1
    RES=$(grep '^\[RESULT\]' "$LOG" | tail -1)
    if [ -z "$RES" ]; then
      echo "  !! FAILED (see $LOG):"; tail -4 "$LOG" | sed 's/^/     /'
      ROWS+=("$(printf '%-6s %-4s %-10s %-12s %-11s %-11s %-13s %-6s' "$CLASS" "$ID" FAILED - - - - -)")
    else
      printf '  %s\n' "$RES"
      ROWS+=("$(printf '%-6s %-4s %-10s %-12s %-11s %-11s %-13s %-6s' \
        "$CLASS" "$ID" \
        "$(get_val "$RES" data_snr)" "$(get_val "$RES" overlap)" \
        "$(get_val "$RES" mismatch)" "$(get_val "$RES" rr_over_dd)" \
        "$(get_val "$RES" source_logL)" "$(get_val "$RES" xcheck)")")
    fi
  done
done

echo
echo "================================  SUMMARY  ================================"
echo "EMRI/MBHB/SOBHB rows: dataSNR overlap mismatch rr/dd src_logL xchk"
echo "GB/VGB rows (ID=rank): detSNR overlap mismatch f0[mHz] mm@tau* -"
printf '%-6s %-4s %-10s %-12s %-11s %-11s %-13s %-6s\n' \
  CLASS ID dataSNR overlap mismatch "rr/dd|f0" "srcLL|mm*" xchk
printf '%s\n' "${ROWS[@]}"
echo "=========================================================================="
echo "logs: $LOGDIR"
echo "workers: $NULL"
echo "         $GB_MATCH (GB_TOPN=$GB_TOPN_WANT)"
echo "         $VGB_MATCH (VGB_TOPN=$VGB_TOPN_WANT)"
