#!/bin/bash
# Overnight WDM time-frequency null-residual decomposition for every LOCAL MBH
# source x {legacy, tof}. One build at a time (memory fully released between
# each), each guarded by a system-swap kill-switch so a spike aborts only that
# build, never the machine. Then analyze each source -> PNG + printed split.
#
# Launch (prevents idle sleep, detached):
#   nohup caffeinate -i bash scripts/campaign/runners/run_td_overnight.sh \
#         > /tmp/mbh_td/overnight_driver.log 2>&1 &
#
# Progress:  tail -f /tmp/mbh_td/overnight.log
set -u

REPO=/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools
PY=/Users/mkatz/miniconda3/envs/deving/bin/python
OUT=${CAMPAIGN_PLOT_DIR:-/tmp/mbh_td}
LOG=$OUT/overnight.log
# excess-population sources first (16 worst, 18, 1), then controls (17,19,0)
SRCS="${TD_SRCS:-16 18 1 17 19 0}"

export CAMPAIGN_PLOT_DIR=$OUT
export MOJITO_DATA_PATH=${MOJITO_DATA_PATH:-/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NULL_CHECK_MEM_GB=${NULL_CHECK_MEM_GB:-7.5}   # per-process RSS abort (PRIMARY guard)
SWAP_KILL_MB=${SWAP_KILL_MB:-60}                     # system free-swap kill line (last resort)
RAM_GO_PCT=${RAM_GO_PCT:-33}                         # RAM free %% needed to start a build
SETTLE_S=${SETTLE_S:-30}                             # settle after each build (macOS reclaim)

mkdir -p "$OUT"
cd "$REPO" || exit 1

swapfree() { sysctl -n vm.swapusage | sed -E 's/.*free = ([0-9.]+)M.*/\1/'; }
ramfree()  { memory_pressure 2>/dev/null | grep -i "free percentage" | grep -oE "[0-9]+" | head -1; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }   # a<b ?
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== overnight TD run START; sources: $SRCS ; RSS_cap=${NULL_CHECK_MEM_GB}G ==="
for id in $SRCS; do
  for resp in legacy tof; do
    npz=$OUT/td_${id}_${resp}.npz
    if [ -f "$npz" ]; then log "id=$id $resp: npz present, skip"; continue; fi

    # settle, then wait (<=15 min) until RAM has been reclaimed to RAM_GO_PCT
    sleep "$SETTLE_S"
    for _ in $(seq 1 45); do
      rf=$(ramfree); [ -z "$rf" ] && break
      lt "$rf" "$RAM_GO_PCT" || break     # proceed once free RAM % >= threshold
      sleep 20
    done
    log "id=$id $resp: launch (RAM_free=$(ramfree)% swap_free=$(swapfree)M)"

    MBHB_ID=$id RESPONSE=$resp nice -10 "$PY" \
      scripts/campaign/runners/mbh_td_residual.py extract \
      > "$OUT/td_${id}_${resp}.log" 2>&1 &
    pid=$!

    # Last-resort system backstop only: the python RSS watchdog (7.5G) is the
    # primary guard. macOS grows swap on demand, so a low swap_free is not
    # itself OOM -- require it to stay critically low for 3 consecutive checks.
    strikes=0
    while kill -0 $pid 2>/dev/null; do
      f=$(swapfree)
      if lt "$f" "$SWAP_KILL_MB"; then strikes=$((strikes+1)); else strikes=0; fi
      if [ $strikes -ge 3 ]; then
        log "id=$id $resp: SWAP CRITICAL ${f}M x3 -> kill"; kill -9 $pid 2>/dev/null; break
      fi
      sleep 8
    done
    wait $pid 2>/dev/null; rc=$?
    log "id=$id $resp: rc=$rc  $(grep -E '\[RESULT\]' "$OUT/td_${id}_${resp}.log" | tail -1)"
    sleep 5
  done

  if [ -f "$OUT/td_${id}_legacy.npz" ] && [ -f "$OUT/td_${id}_tof.npz" ]; then
    "$PY" scripts/campaign/runners/mbh_td_residual.py analyze --id "$id" \
      > "$OUT/analyze_${id}.log" 2>&1
    log "id=$id: ANALYZED -> $(grep -E 'legacy|TOF|insp %|analyze_ok' "$OUT/analyze_${id}.log" | tr -s ' ' | tail -4 | tr '\n' '|')"
  else
    log "id=$id: missing a config npz, skip analyze"
  fi
done
log "=== overnight TD run COMPLETE; npz+PNG in $OUT ; per-source splits in analyze_*.log ==="