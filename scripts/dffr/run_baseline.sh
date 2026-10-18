#!/usr/bin/env bash
# * Baseline of the existing STFT GB response for one backend (step 0 of the DFFR integration).
# What each part records, and the results, are in docs/response/dffr/tests.md.
#
# Usage: run_baseline.sh --device cpu|gpu OUTPUT_ROOT [PART ...]   (the older form drops --device)
#   PART: environment golden oracle benchmarks tests   (the default, in this order)
#         gbgpu_tests lat_tests                         (the two halves of tests)
#         golden_l1 benchmarks_l1                       (need MOJITO_L1, a complete Mojito L1 file)
# Environment: PYTHON (default python), LAT_ROOT (default: the LAT clone of this script; point it
#   at a frozen clone to run that clone's scripts and record its commit), GBGPU_ROOT (default:
#   next to LAT_ROOT), MOJITO_L1,
#   BENCH_ARGS (extra arguments for stft_benchmarks.py, e.g. the GPU segment count and batches).
set -euo pipefail

usage="usage: run_baseline.sh --device cpu|gpu OUTPUT_ROOT [PART ...]"
[ "${1:-}" != "--device" ] || shift
backend=${1:?$usage}
case "$backend" in cpu | gpu) ;; *) echo "$usage" >&2 && exit 2 ;; esac
output_root=${2:?$usage}
shift 2
parts=("$@")
[ ${#parts[@]} -gt 0 ] || parts=(environment golden oracle benchmarks tests)

python_bin=${PYTHON:-python}
mojito_l1=${MOJITO_L1:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lat_root=${LAT_ROOT:-$(cd "$script_dir/../.." && pwd)}
gbgpu_root=${GBGPU_ROOT:-$(cd "$lat_root/../GBGPU" && pwd)}
regression_dir="$output_root/regression/step0"
benchmark_dir="$output_root/benchmarks/step0"
mkdir -p "$regression_dir" "$benchmark_dir"

# * Fixtures that need no Mojito file, and the one that does.
single_grid_configs=(rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag)
l1_configs=(l1_tdi2_envelope)

# ! One thread for OpenMP and BLAS keeps CPU outputs reproducible from run to run.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# ! On a node without an NVIDIA driver the JAX CUDA plugin crashes in its version check.
[ "$backend" = gpu ] || export JAX_PLATFORMS=cpu
export PYTHONWARNINGS=ignore::DeprecationWarning

golden() { "$python_bin" "$lat_root/scripts/dffr/stft_golden_outputs.py" "$@"; }

require_complete_mojito_file() {
    [ -n "$mojito_l1" ] || { echo "MOJITO_L1 is not set" >&2; return 2; }
    # ? A file still being written changes size, and h5py refuses it as truncated.
    local previous_size=-1 size
    while true; do
        size=$(stat -c %s "$mojito_l1")
        if [ "$size" = "$previous_size" ] &&
            "$python_bin" -c "import h5py, sys; h5py.File(sys.argv[1], 'r').close()" "$mojito_l1" 2>/dev/null; then
            return 0
        fi
        previous_size=$size
        echo "[baseline] $(date +%T) waiting for a complete $mojito_l1 ($size bytes)"
        sleep 60
    done
}

capture_twice() {
    # * Two captures in separate processes: equal files show that the outputs are reproducible.
    local stem=$1
    shift
    golden --capture "$regression_dir/$stem.npz" --backend "$backend" "$@"
    golden --capture "$regression_dir/${stem}_repeat.npz" --backend "$backend" "$@"
    golden --compare-files "$regression_dir/$stem.npz" "$regression_dir/${stem}_repeat.npz" \
        | tee "$regression_dir/${stem}_repeat_compare.txt"
}

benchmark_three_times() {
    # * Three runs in separate processes give the run-to-run spread.
    local stem=$1
    shift
    for run in 1 2 3; do
        # shellcheck disable=SC2086
        "$python_bin" "$lat_root/scripts/dffr/stft_benchmarks.py" --backend "$backend" "$@" \
            --output "$benchmark_dir/${stem}_run$run.json" ${BENCH_ARGS:-}
    done
}

run_part() {
    local part=$1 started=$SECONDS
    echo "[baseline] $(date +%T) part $part on $(hostname)"
    case "$part" in
        environment)
            {
                date --iso-8601=seconds
                hostname
                lscpu | grep -E "^Model name|^CPU\(s\)"
                nvidia-smi -L 2>/dev/null || echo "no GPU"
                echo "LAT $(git -C "$lat_root" rev-parse HEAD) on $(git -C "$lat_root" branch --show-current)"
                git -C "$lat_root" status --short
                echo "GBGPU $(git -C "$gbgpu_root" rev-parse HEAD) on $(git -C "$gbgpu_root" branch --show-current)"
                git -C "$gbgpu_root" status --short
                "$python_bin" -c "import importlib.metadata as md
for name in ('lisaanalysistools', 'gbgpu', 'gpubackendtools', 'numpy', 'scipy', 'cupy', 'jax'):
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, 'not installed')"
                env | grep -E "^(OMP|OPENBLAS|MKL)_NUM_THREADS|^JAX_PLATFORMS|^SLURM_JOB_ID|^CUDA_VISIBLE" | sort
            } > "$regression_dir/environment_$backend.txt" 2>&1
            ;;
        golden)
            capture_twice "golden_$backend" --configs "${single_grid_configs[@]}"
            ;;
        golden_l1)
            require_complete_mojito_file
            capture_twice "golden_${backend}_l1" --configs "${l1_configs[@]}" --mojito-l1 "$mojito_l1"
            ;;
        oracle)
            # ! The oracle script runs on the CPU backend only.
            [ "$backend" = cpu ] || { echo "[baseline] oracle part skipped on $backend"; return 0; }
            "$python_bin" "$lat_root/scripts/validation/stft_column_policy_oracle.py" \
                --capture "$regression_dir/column_oracle_cpu.npz"
            "$python_bin" "$lat_root/scripts/validation/stft_column_policy_oracle.py" --bench \
                | tee "$benchmark_dir/column_oracle_bench_cpu.txt"
            golden --compare-files "$regression_dir/column_oracle_cpu.npz" \
                "$regression_dir/golden_cpu.npz" --common-only \
                | tee "$regression_dir/column_oracle_vs_golden_cpu.txt"
            ;;
        benchmarks)
            benchmark_three_times "${backend}_esa" --config esa_tdi2_envelope
            ;;
        benchmarks_l1)
            require_complete_mojito_file
            benchmark_three_times "${backend}_l1" --config l1_tdi2_envelope --mojito-l1 "$mojito_l1"
            ;;
        tests)
            run_part gbgpu_tests
            run_part lat_tests
            ;;
        gbgpu_tests)
            # ? Pre-existing failures are recorded here, not fixed; the exit code goes to the log.
            # * A module that fails to import must not keep pytest from running the others.
            local status=0
            (cd "$gbgpu_root" && "$python_bin" -m pytest tests -rA -p no:cacheprovider \
                --continue-on-collection-errors \
                --junitxml="$regression_dir/gbgpu_tests_$backend.xml") \
                > "$regression_dir/gbgpu_tests_$backend.log" 2>&1 || status=$?
            echo "pytest exit status $status" >> "$regression_dir/gbgpu_tests_$backend.log"
            git -C "$gbgpu_root" status --short \
                > "$regression_dir/git_status_after_gbgpu_tests_$backend.txt"
            ;;
        lat_tests)
            local status=0
            (cd "$lat_root" && "$python_bin" -m unittest discover -v) \
                > "$regression_dir/lat_tests_$backend.log" 2>&1 || status=$?
            echo "unittest exit status $status" >> "$regression_dir/lat_tests_$backend.log"
            git -C "$lat_root" status --short > "$regression_dir/git_status_after_lat_tests_$backend.txt"
            ;;
        *)
            echo "unknown part $part" >&2
            return 2
            ;;
    esac
    echo "[baseline] $(date +%T) part $part done in $((SECONDS - started)) s"
}

for part in "${parts[@]}"; do
    run_part "$part"
done
