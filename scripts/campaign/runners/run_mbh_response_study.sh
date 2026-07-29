#!/usr/bin/env bash
#
# Regenerate the MBH response x band 2x2 null-check study across ALL MBH ids,
# on CPU. Drives the stock null-check worker
#   scripts/validation/mojito_null_check.py
# (erebor.full_year_combined, CHOP_WINDOW 48-day merger-centered snippet, one
# MBH id at a time, template built at the exact catalogue injection) once per
# (id, config) and collects the machine-parseable `[RESULT] branch=mbh ...`
# lines into the four per-config files the plotter ingests:
#
#   legacy_full.txt  legacy_cut.txt  tof_full.txt  tof_cut.txt
#
# The 2x2 is {response} x {band}:
#   response : legacy phentax (USE_TDIONFLY unset) | TDI-on-the-fly (USE_TDIONFLY=1)
#   band     : full (MIN_FREQ unset)               | >5e-4 Hz cut (MIN_FREQ=5e-4)
#
# Nothing here reimplements any numerics -- it is pure orchestration around the
# installed stock worker, exactly how the recorded evidence under
# scripts/campaign/evidence/mbh_response_study/ was produced.
#
# Usage:
#   ./run_mbh_response_study.sh                         # all ids, all 4 configs
#   CONFIGS="legacy_full tof_full" ./run_mbh_response_study.sh   # subset of configs
#   MBHB_ID_LIST="0 1 17 19" ./run_mbh_response_study.sh         # subset of ids
#
# Overridable env: LAT_DIR, PY, MOJITO_DATA_PATH, OUT_DIR, CONFIGS,
#   MBHB_ID_LIST, NULL_CHECK_MEM_GB.
#
# Runtime note: each run loads a multi-GB L1 file and builds a WDM grid, so a
# full 4-config x 20-id sweep is ~80 worker runs (hours). Results are written
# incrementally -- Ctrl-C any time and the partial *.txt files are still valid
# inputs to the plotter. Pick CONFIGS/MBHB_ID_LIST to shorten.
set -uo pipefail

LAT_DIR="${LAT_DIR:-/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools}"
PY="${PY:-/Users/mkatz/miniconda3/envs/deving/bin/python}"
MOJITO_DATA_PATH="${MOJITO_DATA_PATH:-/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/}"
OUT_DIR="${OUT_DIR:-/tmp/mbh_response_study}"
CONFIGS="${CONFIGS:-legacy_full tof_full legacy_cut tof_cut}"
NULL="$LAT_DIR/scripts/validation/mojito_null_check.py"

if command -v brew >/dev/null 2>&1; then
  export PKG_CONFIG_PATH="$(brew --prefix lapack 2>/dev/null)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

mkdir -p "$OUT_DIR/logs"

# --- discover MBH ids (override with MBHB_ID_LIST="0 1 2 ...") ---------------
L1="$MOJITO_DATA_PATH/data/MBHB/L1"
if [ -n "${MBHB_ID_LIST:-}" ]; then
  IDS="$MBHB_ID_LIST"
else
  IDS=$(ls "$L1"/*.h5 2>/dev/null | grep -oE 'source[0-9]+' | grep -oE '[0-9]+' | sort -n -u)
fi
[ -z "$IDS" ] && { echo "no MBHB source files under $L1 (set MBHB_ID_LIST or MOJITO_DATA_PATH)"; exit 1; }
echo "[ids]     $(echo $IDS | tr '\n' ' ')"
echo "[configs] $CONFIGS"
echo "[out]     $OUT_DIR"

# --- CPU + single-walker + thread pins, shared across every run -------------
# (leave USE_TDIONFLY / MIN_FREQ OUT of this block -- set per-config below so
#  legacy/full runs see them genuinely UNSET, not empty.)
export DATA_PROCESSOR=mojito MOJITO_DATA_PATH NWALKERS=1 NTEMPS=1 \
       USE_GPU=0 CHOP_WINDOW=1 EMRI_IDS="" SOBHB_IDS="" \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       NULL_CHECK_MEM_GB="${NULL_CHECK_MEM_GB:-24}"

# config name -> which knobs to set
cfg_tdionfly() { case "$1" in tof_*)  echo 1;;    *) echo "";; esac; }   # "" => unset (legacy)
cfg_minfreq()  { case "$1" in *_cut)  echo 5e-4;; *) echo "";; esac; }   # "" => unset (full band)

# fresh result files for the configs we're about to (re)run
for C in $CONFIGS; do : > "$OUT_DIR/$C.txt"; done

for C in $CONFIGS; do
  TOF=$(cfg_tdionfly "$C"); MF=$(cfg_minfreq "$C")
  echo "=========================================================================="
  echo ">>> config=$C   USE_TDIONFLY='${TOF:-<unset>}'  MIN_FREQ='${MF:-<unset>}'"
  for ID in $IDS; do
    echo "  -- $C id=$ID  ($(date '+%H:%M:%S'))"
    LOG="$OUT_DIR/logs/${C}_id${ID}.log"
    rm -f "$LAT_DIR/gf_output/full_year_combined_run_testing.h5" 2>/dev/null
    (
      cd "$LAT_DIR"
      export MBHB_IDS="$ID"
      if [ -n "$TOF" ]; then export USE_TDIONFLY="$TOF"; else unset USE_TDIONFLY; fi
      if [ -n "$MF"  ]; then export MIN_FREQ="$MF";       else unset MIN_FREQ;      fi
      "$PY" "$NULL"
    ) > "$LOG" 2>&1
    RES=$(grep '^\[RESULT\] branch=mbh' "$LOG" | tail -1)
    if [ -n "$RES" ]; then
      echo "$RES" | tee -a "$OUT_DIR/$C.txt"
    else
      echo "     !! FAILED id=$ID (see $LOG):"; tail -3 "$LOG" | sed 's/^/        /'
    fi
  done
done

echo
echo "=== done -> result files in $OUT_DIR ==="
ls -la "$OUT_DIR"/*.txt 2>/dev/null

echo
echo "plot the 2x2 with:"
echo "  $PY $LAT_DIR/scripts/campaign/runners/mbh_response_study.py \\"
echo "    \"legacy full=$OUT_DIR/legacy_full.txt\" \\"
echo "    \"legacy >5e-4=$OUT_DIR/legacy_cut.txt\" \\"
echo "    \"TOF full=$OUT_DIR/tof_full.txt\" \\"
echo "    \"TOF >5e-4=$OUT_DIR/tof_cut.txt\""
