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
#              root to include (default 8);
#        LOG_CAP_MB  when the run log exceeds this (default 200), ship a
#              GREP-FILTERED full-history file + the raw tail instead of
#              the whole log (v8 audit 2026-09-02: the DEBUG log grows
#              ~0.5 MB/iteration; at 2000 iterations the raw file alone
#              would blow the "low hundreds of MB" snapshot budget).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root (script lives in scripts/fstat_proposal)
echo "make_snapshots @ $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
KEEP=${KEEP:-5}
PYTHON=${PYTHON:-python}
N_JOB_LOGS=${N_JOB_LOGS:-8}

DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
  DIRS=(gf_prod_3mo_v8)
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
  # LOG HANDLING (v8 audit 2026-09-02): the DEBUG run log is the one file
  # that outgrows the budget (~0.5 MB/iteration). Above LOG_CAP_MB, ship
  # (a) a grep-filtered FULL-HISTORY file carrying every line family the
  # monitor page and the snapshot analyses actually read, and (b) the raw
  # tail for context -- and exclude the raw log from the tar.
  LOG_CAP_MB=${LOG_CAP_MB:-200}
  LOG_EXCL=()
  runlog=$(ls "$d"/*_artifacts/globalfit_run.log 2>/dev/null | head -1 || true)
  if [ -n "$runlog" ]; then
    logmb=$(( $(stat -c %s "$runlog" 2>/dev/null || stat -f %z "$runlog") / 1048576 ))
    if [ "$logmb" -gt "$LOG_CAP_MB" ]; then
      echo "== $d: run log ${logmb} MB > ${LOG_CAP_MB} MB -- filtering"
      grep -aE "\[SAVE\]|\[GB_ACCEPT|\[GB_OBS_BASIS|\[GB_INMODEL_TRACE|MISMATCH|\[GB_TEMPER_CHECK|\[GB_CAP|\[GB_ORTHO|\[stageB\]|epoch .* done in|\[FSTAT_CTR|\[GB_TIMING|\[GB_BAND_SHUTOFF|\[GB_BAND_REVIVE|\[V8-NOISE|\[COARSE_AUDIT|\[GB_TRUST|\[GB_VERT|\[unequal-arm|\[galfor-modulation|\[LADDER|\[MIDIT_CKPT|WARNING|ERROR|CRITICAL|Traceback|ll AUDIT|DELTA-vs-DELTA|sig-het engine resolved|\[GB_SWEEP|\[LOGMIRROR|\[GB_REPLACE" \
        "$runlog" > "${runlog%.log}_filtered.log" || true
      tail -c 20000000 "$runlog" > "${runlog%.log}_tail.log" || true
      LOG_EXCL=(! -path "$runlog")
    fi
  fi
  find "$d" -type f \
       ! -path "*fstat_grid_parts*" \
       ! -path "*/dissect/*" \
       ! -path "*_artifacts/diagnostics/*" \
       ! -name "*.bak" ! -name "*.h5.bak*" ! -name "*.tmp" \
       ! -name "*.tar.gz" ! -name "*.zip" \
       ! -name "*_midit_checkpoint.pkl" \
       "${LOG_EXCL[@]}" \
       "${FSTAT_PRUNE[@]}" \
       \( ! -name "*.h5" -o -name "*_extract.h5" \) \
       > "$list"
  # newest slurm job logs from the repo root, if any
  ls -t *.log 2>/dev/null | head -"$N_JOB_LOGS" >> "$list" || true

  out=${d}_snapshot.tar.gz
  # Live files (the append-only run log kept under LOG_CAP_MB, and the extract
  # h5) can change while tar reads them, making tar exit 1 ("file changed as we
  # read it"). That is BENIGN here: an append-only log gives a consistent
  # prefix and the extract h5 is written atomically, so the archive is sound.
  # Suppress the noisy warning and accept exit 1; only a real failure (>=2)
  # aborts.
  rc=0
  tar --warning=no-file-changed -czf "$out" -T "$list" || rc=$?
  if [ "$rc" -ge 2 ]; then
    echo "== ERROR: tar failed for $out (exit $rc)" >&2
    rm -f "$list"
    exit "$rc"
  fi
  [ "$rc" -eq 1 ] && echo "== note: a live file changed while archiving $out (run log / extract h5) -- benign; the snapshot holds a consistent prefix."
  rm -f "$list"
  echo "== wrote $out ($(du -h "$out" | cut -f1)); largest members:"
  tar -tvzf "$out" | sort -rk3 -n | head -8 | awk '{printf "     %6.1f MB  %s\n", $3/1048576, $NF}'
done
