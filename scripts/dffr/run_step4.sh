#!/usr/bin/env bash
# * Step 4 of the DFFR integration (DFFR kernels): the existing response against step 3, the DFFR
# * tests and their measured errors, the exchange with standalone dffr, and the kernel resources.
# What each part records, and the results, are in docs/response/dffr/tests.md.
#
# Usage: run_step4.sh --device cpu|gpu OUTPUT_ROOT [PART ...]
#   PART: environment golden oracle accuracy tests exchange against_existing benchmarks
#                                                  (the default, in order)
#         golden_l1 benchmarks_l1                  (need MOJITO_L1, a Mojito L1 file)
#         gbgpu_tests lat_tests                    (the two halves of tests)
#         resources                                (linked kernel resources; needs nvcc, not a GPU)
# Environment: PYTHON (default python), BASE_ENV (the frozen environment of the step 3 code, from
#   build_base_env.sh; accuracy and resources compare with it), DFFR_PYTHON (the Python of the
#   standalone dffr environment, for the exchange), MOJITO_L1, GBGPU_ROOT (default: next to this
#   LAT clone), GBGPU_BUILD_DIR (GBGPU build tree, default: env/build/GBGPU next to the clones),
#   BENCH_ARGS (extra arguments for stft_benchmarks.py).
set -euo pipefail

usage="usage: run_step4.sh --device cpu|gpu OUTPUT_ROOT [PART ...]"
[ "${1:-}" = "--device" ] || { echo "$usage" >&2 && exit 2; }
device=${2:?$usage}
case "$device" in cpu | gpu) ;; *) echo "$usage" >&2 && exit 2 ;; esac
output_root=${3:?$usage}
shift 3
parts=("$@")
[ ${#parts[@]} -gt 0 ] ||
    parts=(environment golden oracle accuracy tests exchange against_existing benchmarks)

python_bin=${PYTHON:-python}
mojito_l1=${MOJITO_L1:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lat_root=$(cd "$script_dir/../.." && pwd)
gbgpu_root=${GBGPU_ROOT:-$(cd "$lat_root/../GBGPU" && pwd)}
gbgpu_build_dir=${GBGPU_BUILD_DIR:-$lat_root/../env/build/GBGPU}
regression_dir="$output_root/regression/step4"
step2_dir="$output_root/regression/step2"
step3_dir="$output_root/regression/step3"
benchmark_dir="$output_root/benchmarks/step4"
mkdir -p "$regression_dir" "$benchmark_dir"

single_grid_configs=(rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag)
l1_configs=(l1_tdi2_envelope)
# ? The CPU benchmarks leave out the two swap kernels, as the step 2 and step 3 runs do.
bench_swap_args=()
[ "$device" = gpu ] || bench_swap_args=(--without-swap)

# ! One thread for OpenMP and BLAS keeps CPU outputs reproducible from run to run.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# ! On a node without an NVIDIA driver the JAX CUDA plugin crashes in its version check.
[ "$device" = gpu ] || export JAX_PLATFORMS=cpu
export PYTHONWARNINGS=ignore::DeprecationWarning

golden() { "$python_bin" "$script_dir/stft_golden_outputs.py" "$@"; }

require_mojito_file() {
    [ -n "$mojito_l1" ] || { echo "MOJITO_L1 is not set" >&2; return 2; }
}

require_base_env() {
    [ -n "${BASE_ENV:-}" ] || { echo "BASE_ENV is not set" >&2; return 2; }
}

capture_and_compare() {
    # * Two captures in separate processes show the outputs are reproducible; the gate of this
    # * step is bit identity against step 3, and its status is recorded, not fatal.
    local stem=$1
    shift
    local reference="$step3_dir/$stem.npz"
    golden --capture "$regression_dir/$stem.npz" --device "$device" "$@"
    golden --capture "$regression_dir/${stem}_repeat.npz" --device "$device" "$@"
    golden --compare-files "$regression_dir/$stem.npz" "$regression_dir/${stem}_repeat.npz" \
        | tee "$regression_dir/${stem}_repeat_compare.txt"
    if [ -f "$reference" ]; then
        golden --compare-files "$reference" "$regression_dir/$stem.npz" \
            > "$regression_dir/${stem}_vs_step3.txt" || true
        tail -n 1 "$regression_dir/${stem}_vs_step3.txt"
    else
        echo "[step4] no step 3 reference $reference to compare with"
    fi
}

benchmark_three_times() {
    local stem=$1
    shift
    for run in 1 2 3; do
        # shellcheck disable=SC2086
        "$python_bin" "$script_dir/stft_benchmarks.py" --device "$device" "${bench_swap_args[@]}" \
            "$@" --output "$benchmark_dir/${stem}_run$run.json" ${BENCH_ARGS:-}
    done
    # * The gate: every point within the larger run-to-run spread of the step 3 and step 4 runs.
    if ls "$output_root/benchmarks/step3/${stem}_run"*.json > /dev/null 2>&1; then
        "$python_bin" "$script_dir/compare_benchmarks.py" \
            --reference "$output_root/benchmarks/step3/${stem}_run"*.json \
            --candidate "$benchmark_dir/${stem}_run"*.json | tee "$benchmark_dir/compare_$stem.txt"
    fi
}

run_part() {
    local part=$1 started=$SECONDS status
    echo "[step4] $(date +%T) part $part on $(hostname)"
    case "$part" in
        environment)
            {
                date --iso-8601=seconds
                hostname
                lscpu | grep -E "^Model name|^CPU\(s\)"
                nvidia-smi -L 2>/dev/null || echo "no GPU"
                echo "LAT $(git -C "$lat_root" rev-parse HEAD)" \
                    "on $(git -C "$lat_root" branch --show-current)"
                git -C "$lat_root" status --short
                echo "GBGPU $(git -C "$gbgpu_root" rev-parse HEAD)" \
                    "on $(git -C "$gbgpu_root" branch --show-current)"
                git -C "$gbgpu_root" status --short
                "$python_bin" -c "import importlib.metadata as md
for name in ('lisaanalysistools', 'gbgpu', 'gpubackendtools', 'numpy', 'scipy', 'cupy', 'jax'):
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, 'not installed')"
                env | grep -E "^(OMP|OPENBLAS|MKL)_NUM_THREADS|^JAX_PLATFORMS" | sort
                env | grep -E "^SLURM_JOB_ID|^CUDA_VISIBLE" | sort
            } > "$regression_dir/environment_$device.txt" 2>&1
            ;;
        golden)
            capture_and_compare "golden_$device" --configs "${single_grid_configs[@]}"
            ;;
        golden_l1)
            require_mojito_file
            capture_and_compare "golden_${device}_l1" --configs "${l1_configs[@]}" \
                --mojito-l1 "$mojito_l1"
            ;;
        oracle)
            # ! The oracle script runs on the CPU backend only; its reference is the step 2 capture.
            [ "$device" = cpu ] || { echo "[step4] oracle part runs on cpu only"; return 0; }
            status=0
            "$python_bin" "$lat_root/scripts/validation/stft_column_policy_oracle.py" \
                --compare "$step2_dir/column_oracle_cpu.npz" \
                > "$regression_dir/column_oracle_vs_step2_cpu.txt" 2>&1 || status=$?
            echo "oracle exit status $status" >> "$regression_dir/column_oracle_vs_step2_cpu.txt"
            tail -n 2 "$regression_dir/column_oracle_vs_step2_cpu.txt"
            ;;
        accuracy)
            # ! test_stft_gb_accuracy.py runs on the CPU backend only.
            [ "$device" = cpu ] || { echo "[step4] accuracy part runs on cpu only"; return 0; }
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
        exchange)
            # * LAT stores its inputs and outputs, standalone dffr recomputes the outputs in its own
            # * environment, and the comparison reads both.
            [ -n "${DFFR_PYTHON:-}" ] || { echo "DFFR_PYTHON is not set" >&2; return 2; }
            "$python_bin" "$script_dir/dffr_exchange_lat.py" --device "$device" \
                --capture "$regression_dir/exchange_lat_$device.npz"
            env -u VIRTUAL_ENV "$DFFR_PYTHON" "$script_dir/dffr_exchange_dffr.py" \
                --device "$device" \
                --inputs "$regression_dir/exchange_lat_$device.npz" \
                --output "$regression_dir/exchange_dffr_$device.npz"
            status=0
            "$python_bin" "$script_dir/dffr_exchange_lat.py" --compare \
                "$regression_dir/exchange_lat_$device.npz" \
                "$regression_dir/exchange_dffr_$device.npz" \
                > "$regression_dir/exchange_compare_$device.txt" 2>&1 || status=$?
            echo "exchange exit status $status" >> "$regression_dir/exchange_compare_$device.txt"
            tail -n 2 "$regression_dir/exchange_compare_$device.txt"
            ;;
        against_existing)
            "$python_bin" "$script_dir/dffr_against_existing.py" --device "$device" \
                --output "$regression_dir/against_existing_$device.npz" \
                | tee "$regression_dir/against_existing_$device.txt"
            ;;
        resources)
            # * The registers of the linked module set the blocks per SM; the step 3 module is
            # * compiled and linked from the frozen environment's own build tree.
            # * sm_80 is the A100 of the timing gate, sm_90 the H100.
            require_base_env
            for arch in 80 90; do
                "$python_bin" "$script_dir/kernel_resources.py" --build-dir "$gbgpu_build_dir" \
                    --reference-build-dir "$BASE_ENV/env/build/GBGPU" --arch "$arch" \
                    --output-dir "$regression_dir/kernel_resources" \
                    > "$regression_dir/kernel_resources_sm$arch.txt" 2>&1 &
            done
            wait
            cat "$regression_dir"/kernel_resources_sm*.txt
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
            git -C "$gbgpu_root" status --short \
                > "$regression_dir/git_status_after_gbgpu_tests_$device.txt"
            ;;
        lat_tests)
            status=0
            (cd "$lat_root" && "$python_bin" -m unittest discover -v) \
                > "$regression_dir/lat_tests_$device.log" 2>&1 || status=$?
            echo "unittest exit status $status" >> "$regression_dir/lat_tests_$device.log"
            git -C "$lat_root" status --short \
                > "$regression_dir/git_status_after_lat_tests_$device.txt"
            ;;
        *)
            echo "unknown part $part" >&2
            return 2
            ;;
    esac
    echo "[step4] $(date +%T) part $part done in $((SECONDS - started)) s"
}

for part in "${parts[@]}"; do
    run_part "$part"
done
