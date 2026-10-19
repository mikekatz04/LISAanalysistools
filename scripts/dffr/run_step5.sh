#!/usr/bin/env bash
# * Step 5 of the DFFR integration (wiring into STFTGBComputations): the existing response against
# * step 4, the DFFR response through the comp (first golden capture, exchange with standalone
# * dffr, comparison with the existing response), the finite-difference noise of the gradients
# * with the envelope on and off, and the timings of both responses.
# What each part records, and the results, are in docs/response/dffr/tests.md.
#
# Usage: run_step5.sh --device cpu|gpu OUTPUT_ROOT [PART ...]
#   PART: environment golden oracle accuracy tests golden_dffr exchange against_existing
#         envelope_noise benchmarks                 (the default, in order)
#         golden_l1 golden_dffr_l1 benchmarks_l1    (need MOJITO_L1, a Mojito L1 file)
#         gbgpu_tests lat_tests                     (the two halves of tests)
# Environment: PYTHON (default python), BASE_ENV (the frozen environment of the step 4 code, from
#   build_base_env.sh; accuracy compares with it), DFFR_PYTHON (the Python of the standalone dffr
#   environment, for the exchange), MOJITO_L1, GBGPU_ROOT (default: next to this LAT clone),
#   BENCH_ARGS (extra arguments for stft_benchmarks.py).
set -euo pipefail

usage="usage: run_step5.sh --device cpu|gpu OUTPUT_ROOT [PART ...]"
[ "${1:-}" = "--device" ] || { echo "$usage" >&2 && exit 2; }
device=${2:?$usage}
case "$device" in cpu | gpu) ;; *) echo "$usage" >&2 && exit 2 ;; esac
output_root=${3:?$usage}
shift 3
parts=("$@")
[ ${#parts[@]} -gt 0 ] ||
    parts=(environment golden oracle accuracy tests golden_dffr exchange against_existing
        envelope_noise benchmarks)

python_bin=${PYTHON:-python}
mojito_l1=${MOJITO_L1:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lat_root=$(cd "$script_dir/../.." && pwd)
gbgpu_root=${GBGPU_ROOT:-$(cd "$lat_root/../GBGPU" && pwd)}
regression_dir="$output_root/regression/step5"
step2_dir="$output_root/regression/step2"
step4_dir="$output_root/regression/step4"
benchmark_dir="$output_root/benchmarks/step5"
mkdir -p "$regression_dir" "$benchmark_dir"

single_grid_configs=(rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag)
l1_configs=(l1_tdi2_envelope)
# * The configurations of the DFFR response: the production ones with XYZ channels.
dffr_configs=(esa_tdi2_envelope esa_tdi1_const)
# ? The CPU benchmarks leave out the two swap kernels, as the runs of steps 2 to 4 do.
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

npz_identical() {
    # * Bit identity of every entry of two .npz files, strings included.
    "$python_bin" - "$1" "$2" <<'EOF'
import sys
import numpy as np

reference, candidate = (np.load(path, allow_pickle=False) for path in sys.argv[1:3])
keys = sorted(reference.files)
only_one = sorted(set(keys) ^ set(candidate.files))
equal = [k for k in keys if k in candidate.files and np.array_equal(reference[k], candidate[k])]
for key in keys:
    if key not in equal:
        print(f"differs or missing: {key}")
print(f"{len(equal)} of {len(keys)} entries bit-identical; keys only in one file: {only_one}")
EOF
}

capture_twice() {
    # * Two captures in separate processes show the outputs are reproducible.
    local stem=$1
    shift
    golden --capture "$regression_dir/$stem.npz" --device "$device" "$@"
    golden --capture "$regression_dir/${stem}_repeat.npz" --device "$device" "$@"
    golden --compare-files "$regression_dir/$stem.npz" "$regression_dir/${stem}_repeat.npz" \
        | tee "$regression_dir/${stem}_repeat_compare.txt"
}

capture_and_compare() {
    # * The existing response: the gate of this step is bit identity against step 4, recorded,
    # * not fatal.
    local stem=$1
    shift
    capture_twice "$stem" "$@"
    local reference="$step4_dir/$stem.npz"
    if [ -f "$reference" ]; then
        golden --compare-files "$reference" "$regression_dir/$stem.npz" \
            > "$regression_dir/${stem}_vs_step4.txt" || true
        tail -n 1 "$regression_dir/${stem}_vs_step4.txt"
    else
        echo "[step5] no step 4 reference $reference to compare with"
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
}

compare_runs() {
    local reference=$1 candidate=$2 output=$3
    if ls "${reference}_run"*.json > /dev/null 2>&1; then
        "$python_bin" "$script_dir/compare_benchmarks.py" --reference "${reference}_run"*.json \
            --candidate "${candidate}_run"*.json | tee "$output"
    fi
}

run_part() {
    local part=$1 started=$SECONDS status
    echo "[step5] $(date +%T) part $part on $(hostname)"
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
        golden_dffr)
            # * The first capture of the DFFR response through the comp, the baseline of step 6.
            capture_twice "golden_dffr_$device" --response dffr --configs "${dffr_configs[@]}"
            ;;
        golden_dffr_l1)
            require_mojito_file
            capture_twice "golden_dffr_${device}_l1" --response dffr --configs "${l1_configs[@]}" \
                --mojito-l1 "$mojito_l1"
            ;;
        oracle)
            # ! The oracle script runs on the CPU backend only; its reference is the step 2 capture.
            [ "$device" = cpu ] || { echo "[step5] oracle part runs on cpu only"; return 0; }
            status=0
            "$python_bin" "$lat_root/scripts/validation/stft_column_policy_oracle.py" \
                --compare "$step2_dir/column_oracle_cpu.npz" \
                > "$regression_dir/column_oracle_vs_step2_cpu.txt" 2>&1 || status=$?
            echo "oracle exit status $status" >> "$regression_dir/column_oracle_vs_step2_cpu.txt"
            tail -n 2 "$regression_dir/column_oracle_vs_step2_cpu.txt"
            ;;
        accuracy)
            # ! test_stft_gb_accuracy.py runs on the CPU backend only.
            [ "$device" = cpu ] || { echo "[step5] accuracy part runs on cpu only"; return 0; }
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
            # * LAT's outputs come from the comp now; on the same inputs they must equal the step 4
            # * capture, which called the entry points, bit for bit.
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
            if [ -f "$step4_dir/exchange_lat_$device.npz" ]; then
                npz_identical "$step4_dir/exchange_lat_$device.npz" \
                    "$regression_dir/exchange_lat_$device.npz" \
                    | tee "$regression_dir/exchange_lat_${device}_vs_step4.txt"
            fi
            ;;
        against_existing)
            "$python_bin" "$script_dir/dffr_against_existing.py" --device "$device" \
                --output "$regression_dir/against_existing_$device.npz" \
                > "$regression_dir/against_existing_$device.txt"
            tail -n 1 "$regression_dir/against_existing_$device.txt"
            if [ -f "$step4_dir/against_existing_$device.npz" ]; then
                npz_identical "$step4_dir/against_existing_$device.npz" \
                    "$regression_dir/against_existing_$device.npz" \
                    | tee "$regression_dir/against_existing_${device}_vs_step4.txt"
            fi
            ;;
        envelope_noise)
            # * Each response's own finite-difference noise per gradient component, with the
            # * envelope forced on and forced off, at the default and the tenfold steps.
            local configs=("${dffr_configs[@]}") l1_args=()
            if [ -n "$mojito_l1" ]; then
                configs+=("${l1_configs[@]}")
                l1_args=(--mojito-l1 "$mojito_l1")
            fi
            for response in tdi_on_the_fly dffr; do
                "$python_bin" "$script_dir/finite_difference_steps.py" --device "$device" \
                    --capture "$regression_dir/envelope_noise_${response}_$device.npz" \
                    --configs "${configs[@]}" "${l1_args[@]}" --response "$response" \
                    --envelope-variants
                "$python_bin" "$script_dir/finite_difference_steps.py" \
                    --noise "$regression_dir/envelope_noise_${response}_$device.npz" \
                    | tee "$regression_dir/envelope_noise_${response}_$device.txt"
            done
            ;;
        benchmarks)
            # * The existing response against the step 4 runs (the gate: every point within the
            # * larger run-to-run spread), then DFFR, and DFFR against the existing response.
            benchmark_three_times "${device}_esa" --config esa_tdi2_envelope
            compare_runs "$output_root/benchmarks/step4/${device}_esa" "$benchmark_dir/${device}_esa" \
                "$benchmark_dir/compare_${device}_esa.txt"
            benchmark_three_times "${device}_esa_dffr" --config esa_tdi2_envelope --response dffr
            compare_runs "$benchmark_dir/${device}_esa" "$benchmark_dir/${device}_esa_dffr" \
                "$benchmark_dir/compare_${device}_esa_dffr_vs_existing.txt"
            ;;
        benchmarks_l1)
            require_mojito_file
            benchmark_three_times "${device}_l1" --config l1_tdi2_envelope --mojito-l1 "$mojito_l1"
            compare_runs "$output_root/benchmarks/step4/${device}_l1" "$benchmark_dir/${device}_l1" \
                "$benchmark_dir/compare_${device}_l1.txt"
            benchmark_three_times "${device}_l1_dffr" --config l1_tdi2_envelope \
                --mojito-l1 "$mojito_l1" --response dffr
            compare_runs "$benchmark_dir/${device}_l1" "$benchmark_dir/${device}_l1_dffr" \
                "$benchmark_dir/compare_${device}_l1_dffr_vs_existing.txt"
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
    echo "[step5] $(date +%T) part $part done in $((SECONDS - started)) s"
}

for part in "${parts[@]}"; do
    run_part "$part"
done
