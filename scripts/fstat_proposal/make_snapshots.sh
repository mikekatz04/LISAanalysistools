#!/bin/bash
# Build analysis-complete snapshot archives of global-fit run directories:
# a same-layout REDUCED store (gf_store_extract.py) instead of the full h5,
# everything else in the run dir except the large fstat grid parts, plus
# the most recent slurm job logs (they carry [GF_TIMING] and the real
# tracebacks the store log never sees).
#
# Usage (from anywhere; run dirs resolved relative to the repo root):
#   bash scripts/fstat_proposal/make_snapshots.sh [run_dir ...]
#   bash scripts/fstat_proposal/make_snapshots.sh gf_prod_1yr_v5
#   KEEP=8 bash scripts/fstat_proposal/make_snapshots.sh gf_prod_3mo_v6
#
# With no arguments it does the current production trio. Output:
#   <run_dir>_snapshot.tar.gz   (extract locally with: tar -xzf <file>)
#
# Knobs: PYTHON  interpreter for the extract (default: python);
#        INCLUDE_FSTAT=1  keep the fstat epoch caches (default: only
#              DONE.json; the payload npzs are 100s of MB-GBs);
#        KEEP  iterations of the big chains kept in the reduced store
#              (default 5); N_JOB_LOGS  newest *.log files from the repo
#              root to include (default 8).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root (script lives in scripts/fstat_proposal)
KEEP=${KEEP:-5}
PYTHON=${PYTHON:-python}
N_JOB_LOGS=${N_JOB_LOGS:-8}

DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
  DIRS=(gf_prod_3mo_v5 gf_prod_3mo_v6 gf_prod_1yr_v5)
fi

for d in "${DIRS[@]}"; do
  d=${d%/}
  if [ ! -d "$d" ]; then
    echo "== $d: not a directory, skipping"
    continue
  fi
  # live store = newest *testing*.h5 that is not a backup/corrupt/extract
  h5=$(ls -t "$d"/*testing*.h5 2>/dev/null \
       | grep -v -e backup -e CORRUPT -e _extract | head -1 || true)
  if [ -z "$h5" ]; then
    echo "== $d: no live *testing*.h5 found, skipping"
    continue
  fi
  echo "== $d  live store: $h5"

  if ! "$PYTHON" scripts/diagnostics/gf_store_extract.py "$h5" --keep "$KEEP"; then
    echo "== $d: extract FAILED, skipping archive"
    continue
  fi

  # file list: everything in the dir EXCEPT
  #  * h5 files (the *_extract.h5 reduced store stays in),
  #  * migration/backup copies (*.bak -- a FULL store copy that silently
  #    dodged the h5 rule via its extension; 2026-08-22 1.5 GB tars),
  #  * dissect diagnostic dumps,
  #  * the fstat epoch caches (peaks_stacked/centers/comb are the other
  #    heavy family; DONE.json survives so epoch status stays readable.
  #    INCLUDE_FSTAT=1 keeps them, minus grid parts as always).
  list=$(mktemp)
  FSTAT_PRUNE=()
  if [ "${INCLUDE_FSTAT:-0}" != "1" ]; then
    FSTAT_PRUNE=(-not \( -path "*/gb_fstat_fit/*" -a ! -name "DONE.json" \))
  fi
  find "$d" -type f \
       ! -path "*fstat_grid_parts*" \
       ! -path "*/dissect/*" \
       ! -name "*.bak" ! -name "*.h5.bak*" \
       "${FSTAT_PRUNE[@]}" \
       \( ! -name "*.h5" -o -name "*_extract.h5" \) \
       > "$list"
  # newest slurm job logs from the repo root, if any
  ls -t *.log 2>/dev/null | head -"$N_JOB_LOGS" >> "$list" || true

  out=${d}_snapshot.tar.gz
  tar -czf "$out" -T "$list"
  rm -f "$list"
  echo "== wrote $out ($(du -h "$out" | cut -f1))"
done
