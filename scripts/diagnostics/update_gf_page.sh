#!/bin/bash
# One command from a pulled-down snapshot to a viewable monitor page.
#
#   bash scripts/diagnostics/update_gf_page.sh <snapshot.tar.gz|.zip> [out_dir]
#
# The full loop (v8 monitoring workflow, 2026-09-02):
#   cluster$  cd /shared/data/global_fit_output && \
#             bash <repo>/scripts/fstat_proposal/make_snapshots.sh gf_prod_3mo_v8
#   laptop$   scp cluster:/shared/data/global_fit_output/gf_prod_3mo_v8_snapshot.tar.gz .
#   laptop$   bash scripts/diagnostics/update_gf_page.sh gf_prod_3mo_v8_snapshot.tar.gz
#   -> prints the path of monitor_<run>.html; open it in any browser.
#
# Anyone with the snapshot can run this -- it needs only this repo (with
# its python env) and, for the recovery panels, the 3-month truth set
# gb_truth_3to21.npz in the repo root (the page still builds without it;
# those panels report themselves missing).
#
# Repeatable: re-running with a newer snapshot overwrites the extraction
# in place and regenerates the page. NOTE tar cannot DELETE files -- a
# stale store file from an older layout survives extraction -- but the
# generator picks the live h5 by mtime + iteration attr (431a3df9), so a
# refreshed snapshot always wins. For a truly clean slate, remove the
# out_dir first.
set -euo pipefail

SNAP=${1:?usage: update_gf_page.sh <snapshot.tar.gz|.zip> [out_dir]}
OUT=${2:-gf_pull}
PYTHON=${PYTHON:-python}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
mkdir -p "$OUT"

case "$SNAP" in
  *.tar.gz|*.tgz) tar -xzf "$SNAP" -C "$OUT" ;;
  *.zip)          unzip -oq "$SNAP" -d "$OUT" -x "*fstat_grid_parts/*" ;;
  *) echo "ERROR: $SNAP is neither .tar.gz nor .zip" >&2; exit 2 ;;
esac

# the run dir = the directory holding the live/extract store
RUN_DIR=$(dirname "$(ls -t "$OUT"/*/*testing*.h5 "$OUT"/*/*_extract.h5 2>/dev/null | head -1)")
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: no store h5 found under $OUT after extraction" >&2; exit 3
fi
RUN=$(basename "$RUN_DIR")
echo "== run dir: $RUN_DIR"

if [ ! -f "$HERE/gb_truth_3to21.npz" ]; then
  echo "== NOTE: $HERE/gb_truth_3to21.npz not found -- recovery/match panels"
  echo "         will report themselves missing; the page still builds."
fi

# GF_MONITOR_MATCH_STATS=1 is REQUIRED: without it 5 figures silently
# vanish (documented trap). Run from the repo root so the truth npz and
# the per-run arm cache (gf_arm_*.npz) resolve.
PAGE="$(cd "$(dirname "$RUN_DIR")" && pwd)/monitor_${RUN}.html"
( cd "$HERE" && \
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 GF_MONITOR_MATCH_STATS=1 \
  "$PYTHON" scripts/diagnostics/gf_monitor_gen.py "$RUN_DIR" "$PAGE" )

echo ""
echo "== PAGE: $PAGE"
echo "   open it in any browser (macOS: open \"$PAGE\")"
