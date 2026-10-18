#!/usr/bin/env bash
# * Step 1 of the DFFR integration (orbits cache): unit tests, regressions and benchmarks.
# What each part records, and the results, are in docs/response/dffr/tests.md.
#
# Usage: run_step1.sh --device cpu|gpu OUTPUT_ROOT [PART ...]
#   PART: environment unit golden accuracy dffr tests benchmarks   (the default, in this order)
#         golden_l1 l1 benchmarks_l1                              (need MOJITO_L1, a Mojito L1 file)
#         gbgpu_tests lat_tests                                   (the two halves of tests)
# Environment: PYTHON (default python), BASE_ENV (the frozen environment of build_base_env.sh;
#   accuracy runs there too, for the numbers before the change), DFFR_REPOSITORY (the dffr clone
#   with its .venv, for the dffr part), MOJITO_L1, GBGPU_ROOT (default: next to this LAT clone),
#   BENCH_ARGS (extra arguments for stft_benchmarks.py).
set -euo pipefail

usage="usage: run_step1.sh --device cpu|gpu OUTPUT_ROOT [PART ...]"
[ "${1:-}" = "--device" ] || { echo "$usage" >&2 && exit 2; }
device=${2:?$usage}
case "$device" in cpu | gpu) ;; *) echo "$usage" >&2 && exit 2 ;; esac
output_root=${3:?$usage}
shift 3
parts=("$@")
[ ${#parts[@]} -gt 0 ] || parts=(environment unit golden accuracy dffr tests benchmarks)

python_bin=${PYTHON:-python}
mojito_l1=${MOJITO_L1:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lat_root=$(cd "$script_dir/../.." && pwd)
gbgpu_root=${GBGPU_ROOT:-$(cd "$lat_root/../GBGPU" && pwd)}
regression_dir="$output_root/regression/step1"
step0_dir="$output_root/regression/step0"
benchmark_dir="$output_root/benchmarks/step1"
mkdir -p "$regression_dir" "$benchmark_dir"

single_grid_configs=(rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag)
l1_configs=(l1_tdi2_envelope)

# ! One thread for OpenMP and BLAS keeps CPU outputs reproducible from run to run.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# ! On a node without an NVIDIA driver the JAX CUDA plugin crashes in its version check.
[ "$device" = gpu ] || export JAX_PLATFORMS=cpu
export PYTHONWARNINGS=ignore::DeprecationWarning

golden() { "$python_bin" "$script_dir/stft_golden_outputs.py" "$@"; }
kinematics() { "$python_bin" "$script_dir/kinematics_regression.py" "$@"; }

require_mojito_file() {
    [ -n "$mojito_l1" ] || { echo "MOJITO_L1 is not set" >&2; return 2; }
}

capture_and_compare() {
    # * Two captures in separate processes show the outputs are reproducible; the comparison
    # * with step 0 measures the change, so its non-zero status is recorded, not fatal.
    local stem=$1
    shift
    golden --capture "$regression_dir/$stem.npz" --device "$device" "$@"
    golden --capture "$regression_dir/${stem}_repeat.npz" --device "$device" "$@"
    golden --compare-files "$regression_dir/$stem.npz" "$regression_dir/${stem}_repeat.npz" \
        | tee "$regression_dir/${stem}_repeat_compare.txt"
    if [ -f "$step0_dir/$stem.npz" ]; then
        golden --compare-files "$step0_dir/$stem.npz" "$regression_dir/$stem.npz" \
            > "$regression_dir/${stem}_vs_step0.txt" || true
        tail -n 1 "$regression_dir/${stem}_vs_step0.txt"
    else
        echo "[step1] no step 0 capture $step0_dir/$stem.npz to compare with"
    fi
}

benchmark_three_times() {
    local stem=$1
    shift
    for run in 1 2 3; do
        # shellcheck disable=SC2086
        "$python_bin" "$script_dir/stft_benchmarks.py" --device "$device" "$@" \
            --output "$benchmark_dir/${stem}_run$run.json" ${BENCH_ARGS:-}
    done
    # * Ratios against the step 0 runs of the same fixture, read against the run-to-run spread.
    if ls "$output_root/benchmarks/step0/${stem}_run"*.json > /dev/null 2>&1; then
        "$python_bin" "$script_dir/compare_benchmarks.py" \
            --reference "$output_root/benchmarks/step0/${stem}_run"*.json \
            --candidate "$benchmark_dir/${stem}_run"*.json | tee "$benchmark_dir/compare_$stem.txt"
    fi
}

run_part() {
    local part=$1 started=$SECONDS status
    echo "[step1] $(date +%T) part $part on $(hostname)"
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
            } > "$regression_dir/environment_$device.txt" 2>&1
            ;;
        unit)
            # * The CUDA cases skip themselves when no usable GPU is present.
            status=0
            (cd "$lat_root" && "$python_bin" -m unittest tests.test_orbit_kinematics -v) \
                > "$regression_dir/unit_tests_$device.log" 2>&1 || status=$?
            echo "unittest exit status $status" >> "$regression_dir/unit_tests_$device.log"
            tail -n 4 "$regression_dir/unit_tests_$device.log"
            (cd "$lat_root" && "$python_bin" "$script_dir/kinematics_unit_numbers.py" --device "$device") \
                | tee "$regression_dir/unit_numbers_$device.txt"
            ;;
        golden)
            capture_and_compare "golden_$device" --configs "${single_grid_configs[@]}"
            ;;
        golden_l1)
            require_mojito_file
            capture_and_compare "golden_${device}_l1" --configs "${l1_configs[@]}" --mojito-l1 "$mojito_l1"
            ;;
        accuracy)
            # ! test_stft_gb_accuracy.py runs on the CPU backend only.
            [ "$device" = cpu ] || { echo "[step1] accuracy part runs on cpu only"; return 0; }
            status=0
            (cd "$gbgpu_root" && "$python_bin" -m pytest tests/test_stft_gb_accuracy.py -s -rA \
                -p no:cacheprovider) > "$regression_dir/accuracy_cpu.log" 2>&1 || status=$?
            echo "pytest exit status $status" >> "$regression_dir/accuracy_cpu.log"
            if [ -n "${BASE_ENV:-}" ]; then
                status=0
                (cd "$BASE_ENV/GBGPU" && "$BASE_ENV/.venv/bin/python" -m pytest \
                    tests/test_stft_gb_accuracy.py -s -rA -p no:cacheprovider) \
                    > "$regression_dir/accuracy_base_cpu.log" 2>&1 || status=$?
                echo "pytest exit status $status" >> "$regression_dir/accuracy_base_cpu.log"
            fi
            grep -hE "^\[(e2e|accuracy)\]|exit status" "$regression_dir"/accuracy_*cpu.log
            ;;
        dffr)
            # * dffr runs on its CPU backend; its tables are built once and reused by later runs.
            local dffr_tables="$regression_dir/kinematics_dffr_esa.npz"
            kinematics esa --device "$device" --output "$regression_dir/kinematics_lat_esa_$device.npz"
            if [ ! -f "$dffr_tables" ]; then
                [ -n "${DFFR_REPOSITORY:-}" ] || { echo "DFFR_REPOSITORY is not set" >&2; return 2; }
                local orbit_file
                orbit_file=$("$python_bin" -c "from lisatools.detector import ESAOrbits
print(ESAOrbits(force_backend='cpu').filename)")
                (cd "$DFFR_REPOSITORY" && env -u VIRTUAL_ENV .venv/bin/python \
                    "$script_dir/kinematics_dffr.py" --orbit-file "$orbit_file" --output "$dffr_tables")
            fi
            kinematics compare "$regression_dir/kinematics_lat_esa_$device.npz" "$dffr_tables" \
                --output "$regression_dir/kinematics_esa_compare_$device.json" \
                | tee "$regression_dir/kinematics_esa_compare_$device.txt"
            ;;
        l1)
            require_mojito_file
            kinematics l1 --device "$device" --mojito-l1 "$mojito_l1" \
                --output "$regression_dir/kinematics_l1_$device.npz" \
                | tee "$regression_dir/kinematics_l1_$device.txt"
            ;;
        benchmarks)
            benchmark_three_times "${device}_esa" --config esa_tdi2_envelope
            ;;
        benchmarks_l1)
            require_mojito_file
            benchmark_three_times "${device}_l1" --config l1_tdi2_envelope --mojito-l1 "$mojito_l1"
            ;;
        tests)
            run_part gbgpu_tests
            run_part lat_tests
            ;;
        gbgpu_tests)
            status=0
            (cd "$gbgpu_root" && "$python_bin" -m pytest tests -rA -p no:cacheprovider \
                --continue-on-collection-errors \
                --junitxml="$regression_dir/gbgpu_tests_$device.xml") \
                > "$regression_dir/gbgpu_tests_$device.log" 2>&1 || status=$?
            echo "pytest exit status $status" >> "$regression_dir/gbgpu_tests_$device.log"
            git -C "$gbgpu_root" status --short > "$regression_dir/git_status_after_gbgpu_tests_$device.txt"
            ;;
        lat_tests)
            status=0
            (cd "$lat_root" && "$python_bin" -m unittest discover -v) \
                > "$regression_dir/lat_tests_$device.log" 2>&1 || status=$?
            echo "unittest exit status $status" >> "$regression_dir/lat_tests_$device.log"
            git -C "$lat_root" status --short > "$regression_dir/git_status_after_lat_tests_$device.txt"
            ;;
        *)
            echo "unknown part $part" >&2
            return 2
            ;;
    esac
    echo "[step1] $(date +%T) part $part done in $((SECONDS - started)) s"
}

for part in "${parts[@]}"; do
    run_part "$part"
done
