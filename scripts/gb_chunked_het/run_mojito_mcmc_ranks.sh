#!/bin/bash
# Sequential, resumable 3-way (FD / chunked-het / sig-het) MCMC + corner for one
# or more mojito GB ranks. Each method runs as its own invocation so it is
# independently resumable: a method whose saved chain already has >= NSTEPS steps
# is skipped, so if the driver is killed (the 4-core box can't run these in
# parallel) just relaunch it and it picks up where it left off.
#
# Env: RANKS (default "1 2"), NSTEPS, NWALKERS, NTEMPS, BURN_STEPS, FD_N_SPARSE, OUT_DIR.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
PY=/Users/mkatz/miniconda3/envs/deving/bin/python
OUT="${OUT_DIR:-/tmp/gbmcmc_long}"
RANKS="${RANKS:-1 2}"
NSTEPS="${NSTEPS:-2000}"
NWALKERS="${NWALKERS:-20}"
NTEMPS="${NTEMPS:-5}"
BURN_STEPS="${BURN_STEPS:-1000}"
FD_N_SPARSE="${FD_N_SPARSE:-2048}"
mkdir -p "$OUT"

nsteps_of() {  # $1 = h5 path -> echo chain step count (0 if missing/unreadable)
  $PY - "$1" <<'PY' 2>/dev/null || echo 0
import sys, os
from eryn.backends import HDFBackend
p = sys.argv[1]
print(HDFBackend(p).get_chain()["gb"].shape[0] if os.path.exists(p) else 0)
PY
}

run_method() {  # $1 = rank, $2 = METHODS csv, $3 = h5-suffix to gate on
  local rank=$1 methods=$2 suffix=$3
  local h5="$OUT/mcmc_mojito_rank${rank}_${suffix}.h5"
  local n; n=$(nsteps_of "$h5")
  if [ "${n:-0}" -ge "$NSTEPS" ]; then echo "[skip] rank$rank $methods ($n steps)"; return; fi
  echo "[run ] rank$rank METHODS=$methods  (had $n steps, target $NSTEPS)"
  GB_RANK=$rank METHODS="$methods" NWALKERS=$NWALKERS NTEMPS=$NTEMPS NSTEPS=$NSTEPS \
    BURNIN=0 FD_EDGE_MODE=active FD_N_SPARSE=$FD_N_SPARSE OUT_DIR="$OUT" \
    $PY gb_mojito_mcmc_three_ways.py
}

for rank in $RANKS; do
  echo "===== RANK $rank ====="
  run_method $rank wdm_chunked wdm_chunked
  run_method $rank wdm_sighet  wdm_sighet
  run_method $rank fd          fd
  echo "[corner] rank$rank (burn $BURN_STEPS)"
  GB_RANK=$rank BURN_STEPS=$BURN_STEPS OUT_DIR="$OUT" $PY combine_three_way_corner.py
done
echo "ALL DONE for ranks: $RANKS"
