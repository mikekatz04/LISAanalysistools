#!/usr/bin/env bash
# * Standalone dffr against its time-domain TDI 2.0 truth, one process per source and sky.
# What it measures, and the results, are in docs/response/dffr/tests.md.
#
# Usage: run_dffr_truth.sh DFFR_REPOSITORY OUTPUT_DIR [DAYS] [ORBITS]
#   DAYS defaults to 30; ORBITS to esa, the ESA trailing orbits that lisaanalysistools ships.
# The dffr environment must exist first (on a node that reaches the package index):
#   cd DFFR_REPOSITORY && env -u VIRTUAL_ENV uv sync --frozen --extra compare
set -euo pipefail

usage="usage: run_dffr_truth.sh DFFR_REPOSITORY OUTPUT_DIR [DAYS] [ORBITS]"
dffr_repository=$(cd "${1:?$usage}" && pwd)
output_dir=${2:?$usage}
days=${3:-30}
orbits=${4:-esa}
python_bin="$dffr_repository/.venv/bin/python"
[ -x "$python_bin" ] || { echo "no dffr environment in $dffr_repository/.venv" >&2; exit 2; }
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

# ! One thread per process: the eight cases run side by side, one per CPU.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset VIRTUAL_ENV

echo "[truth] $(date +%T) dffr $(git -C "$dffr_repository" rev-parse --short HEAD) on $(hostname)," \
    "$days days, orbits $orbits"
cd "$dffr_repository"
pids=()
for source in 3mHz 20mHz 20mHz-chirping 40mHz; do
    for sky in 0 1; do
        stem="${source}_sky${sky}"
        "$python_bin" scripts/compare_time_domain.py --days "$days" --orbits "$orbits" \
            --sources "$source" --sky "$sky" --output "$output_dir/$stem.json" \
            > "$output_dir/$stem.log" 2>&1 &
        pids+=($!)
    done
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" = 0 ] || { echo "a case failed; see the logs in $output_dir" >&2; exit 1; }

# * One file and one table for the eight cases.
"$python_bin" - "$output_dir" <<'EOF'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
records = []
for path in sorted(output_dir.glob("*_sky[0-9].json")):
    records.extend(json.loads(path.read_text()))
(output_dir / "summary.json").write_text(json.dumps(records, indent=1))
print(f"{'source':15s} sky {'mismatch':>9s} {'median segment':>14s} {'largest segment':>15s}")
for record in records:
    result = record["compiled"]
    print(f"{record['source']:15s} {record['sky']:3d} {result['mismatch']:9.2e} "
          f"{result['median_segment_error']:14.2e} {result['max_segment_error']:15.2e}")
EOF
echo "[truth] $(date +%T) done"
