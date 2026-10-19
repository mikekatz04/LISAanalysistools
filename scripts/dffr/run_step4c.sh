#!/usr/bin/env bash
# * Part 4c of the DFFR integration (the DFFR Fisher matrix): the existing response and the DFFR
# * likelihood outputs against step 5, the first golden capture of the DFFR Fisher, the exchange
# * with standalone dffr, both test suites, the plateau of the Fisher steps, the linked GPU
# * resources, the CPU timing gate of the DFFR Fisher, and the timings of the existing response.
# What each part records, and the results, are in docs/response/dffr/tests.md.
#
# Usage: run_step4c.sh --device cpu|gpu OUTPUT_ROOT [PART ...]
#   PART: environment golden golden_dffr exchange tests fd_steps resources   (the default, in order)
#         golden_l1 golden_dffr_l1               (need MOJITO_L1, a Mojito L1 file)
#         fisher_timing fisher_timing_l1         (the gate; run them alone on the node)
#         existing_timing existing_timing_l1     (need BASE_ENV; run them alone on the node)
#         gbgpu_tests lat_tests                  (the two halves of tests)
# Environment: PYTHON (default python), BASE_ENV (the frozen environment of the step 5 code, from
#   build_base_env.sh), DFFR_PYTHON (the Python of the standalone dffr environment, for the
#   exchange), MOJITO_L1, GBGPU_ROOT (default: next to this LAT clone), BENCH_ARGS (extra arguments
#   for stft_benchmarks.py).
set -euo pipefail

usage="usage: run_step4c.sh --device cpu|gpu OUTPUT_ROOT [PART ...]"
[ "${1:-}" = "--device" ] || { echo "$usage" >&2 && exit 2; }
device=${2:?$usage}
case "$device" in cpu | gpu) ;; *) echo "$usage" >&2 && exit 2 ;; esac
output_root=${3:?$usage}
shift 3
parts=("$@")
[ ${#parts[@]} -gt 0 ] ||
    parts=(environment golden golden_dffr exchange tests fd_steps resources)

python_bin=${PYTHON:-python}
mojito_l1=${MOJITO_L1:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lat_root=$(cd "$script_dir/../.." && pwd)
gbgpu_root=${GBGPU_ROOT:-$(cd "$lat_root/../GBGPU" && pwd)}
regression_dir="$output_root/regression/step4c"
step5_dir="$output_root/regression/step5"
benchmark_dir="$output_root/benchmarks/step4c"
mkdir -p "$regression_dir" "$benchmark_dir"

single_grid_configs=(rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag)
l1_configs=(l1_tdi2_envelope)
# * The configurations of the DFFR response: the production ones with XYZ channels.
dffr_configs=(esa_tdi2_envelope esa_tdi1_const)
# ? The CPU benchmarks leave out the two swap kernels, as the runs of steps 2 to 5 do.
bench_swap_args=()
[ "$device" = gpu ] || bench_swap_args=(--without-swap)
# * The timing gate of the DFFR Fisher: all nine parameters, and the eight of the global fit.
fisher_methods=(information_matrix information_matrix_test_inds)

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
    # * Bit identity against step 5 on the arrays both captures hold; the DFFR captures of this
    # * part add the Fisher matrices, which step 5 did not have. Recorded, not fatal.
    local stem=$1
    shift
    capture_twice "$stem" "$@"
    local reference="$step5_dir/$stem.npz"
    if [ -f "$reference" ]; then
        golden --compare-files "$reference" "$regression_dir/$stem.npz" \
            > "$regression_dir/${stem}_vs_step5.txt" || true
        tail -n 1 "$regression_dir/${stem}_vs_step5.txt"
    else
        echo "[step4c] no step 5 reference $reference to compare with"
    fi
}

benchmark_in_turn() {
    # * Three runs of each of two benchmark sets, alternated, so that drift of the node reaches
    # * both alike. Arguments: the stems of the two sets, then a separator --, the arguments of the
    # * first set, a separator --, the arguments of the second.
    local first_stem=$1 second_stem=$2
    shift 2
    [ "$1" = "--" ] && shift
    local first_args=() second_args=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do first_args+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    second_args=("$@")
    for run in 1 2 3; do
        # shellcheck disable=SC2086
        "${first_args[@]}" --output "$benchmark_dir/${first_stem}_run$run.json" ${BENCH_ARGS:-}
        # shellcheck disable=SC2086
        "${second_args[@]}" --output "$benchmark_dir/${second_stem}_run$run.json" ${BENCH_ARGS:-}
    done
}

compare_runs() {
    local reference=$1 candidate=$2 output=$3
    "$python_bin" "$script_dir/compare_benchmarks.py" --reference "${reference}_run"*.json \
        --candidate "${candidate}_run"*.json | tee "$output"
}

fisher_timing() {
    # * The gate: the DFFR Fisher no slower than the existing Fisher, both responses in turn.
    local stem=$1
    shift
    local bench=("$python_bin" "$script_dir/stft_benchmarks.py" --device "$device"
        --methods "${fisher_methods[@]}" "$@")
    benchmark_in_turn "${stem}_fisher_existing" "${stem}_fisher_dffr" \
        -- "${bench[@]}" -- "${bench[@]}" --response dffr
    compare_runs "$benchmark_dir/${stem}_fisher_existing" "$benchmark_dir/${stem}_fisher_dffr" \
        "$benchmark_dir/compare_${stem}_fisher_dffr_vs_existing.txt"
}

existing_timing() {
    # * The existing response of this part against the frozen step 5 code, in turn.
    local stem=$1
    shift
    local base_python="$BASE_ENV/.venv/bin/python"
    local base_script="$BASE_ENV/lisa-analysis-tools/scripts/dffr/stft_benchmarks.py"
    benchmark_in_turn "${stem}" "${stem}_step5_env" \
        -- "$python_bin" "$script_dir/stft_benchmarks.py" --device "$device" \
        "${bench_swap_args[@]}" "$@" \
        -- "$base_python" "$base_script" --device "$device" "${bench_swap_args[@]}" "$@"
    compare_runs "$benchmark_dir/${stem}_step5_env" "$benchmark_dir/${stem}" \
        "$benchmark_dir/compare_${stem}_vs_step5_env.txt"
}

run_part() {
    local part=$1 started=$SECONDS status
    echo "[step4c] $(date +%T) part $part on $(hostname)"
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
            # * The likelihood outputs against step 5 and the first capture of the DFFR Fisher,
            # * the baseline of step 6.
            capture_and_compare "golden_dffr_$device" --response dffr --configs "${dffr_configs[@]}"
            ;;
        golden_dffr_l1)
            require_mojito_file
            capture_and_compare "golden_dffr_${device}_l1" --response dffr \
                --configs "${l1_configs[@]}" --mojito-l1 "$mojito_l1"
            ;;
        exchange)
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
            if [ -f "$step5_dir/exchange_lat_$device.npz" ]; then
                npz_identical "$step5_dir/exchange_lat_$device.npz" \
                    "$regression_dir/exchange_lat_$device.npz" \
                    | tee "$regression_dir/exchange_lat_${device}_vs_step5.txt"
            fi
            ;;
        fd_steps)
            # * The Fisher matrices of every source at 1, 10, 100 and 1000 times the default steps,
            # * both responses; the plateau tables; and DFFR against the existing Fisher.
            local configs=("${dffr_configs[@]}") l1_args=()
            if [ -n "$mojito_l1" ]; then
                configs+=("${l1_configs[@]}")
                l1_args=(--mojito-l1 "$mojito_l1")
            fi
            for response in tdi_on_the_fly dffr; do
                "$python_bin" "$script_dir/finite_difference_steps.py" --device "$device" \
                    --capture "$regression_dir/fd_steps_${response}_$device.npz" \
                    --configs "${configs[@]}" "${l1_args[@]}" --response "$response" \
                    --all-sources
                "$python_bin" "$script_dir/finite_difference_steps.py" \
                    --plateau "$regression_dir/fd_steps_${response}_$device.npz" \
                    | tee "$regression_dir/plateau_${response}_$device.txt"
            done
            "$python_bin" "$script_dir/fisher_against_existing.py" \
                "$regression_dir/fd_steps_tdi_on_the_fly_$device.npz" \
                "$regression_dir/fd_steps_dffr_$device.npz" \
                | tee "$regression_dir/fisher_against_existing_$device.txt"
            ;;
        resources)
            # * Linked registers, stack and shared memory for the A100 and the H100, against the
            # * step 5 module; nvcc only, no GPU.
            require_base_env
            local gbgpu_build_dir
            gbgpu_build_dir=${GBGPU_BUILD_DIR:-$(cd "$lat_root/../env/build/GBGPU" && pwd)}
            for arch in 80 90; do
                "$python_bin" "$script_dir/kernel_resources.py" --build-dir "$gbgpu_build_dir" \
                    --reference-build-dir "$BASE_ENV/env/build/GBGPU" --arch "$arch" \
                    --output-dir "$regression_dir/kernel_resources" \
                    > "$regression_dir/kernel_resources_sm$arch.txt" 2>&1 &
            done
            wait
            cat "$regression_dir"/kernel_resources_sm*.txt
            ;;
        fisher_timing)
            fisher_timing "${device}_esa" --config esa_tdi2_envelope
            ;;
        fisher_timing_l1)
            require_mojito_file
            fisher_timing "${device}_l1" --config l1_tdi2_envelope --mojito-l1 "$mojito_l1"
            ;;
        existing_timing)
            require_base_env
            existing_timing "${device}_esa" --config esa_tdi2_envelope
            ;;
        existing_timing_l1)
            require_base_env
            require_mojito_file
            existing_timing "${device}_l1" --config l1_tdi2_envelope --mojito-l1 "$mojito_l1"
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
    echo "[step4c] $(date +%T) part $part done in $((SECONDS - started)) s"
}

for part in "${parts[@]}"; do
    run_part "$part"
done
