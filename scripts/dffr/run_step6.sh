#!/usr/bin/env bash
# * Step 6 of the DFFR integration (validation), the parts run so far: both responses against
# * the STFT of a time-domain truth with both data windows (items 1 and 2, the second on the cases
# * and grid of the step 0 truth), and the checks behind that truth.
# What each part records, and the results, are in docs/response/dffr/tests.md.
#
# Usage: run_step6.sh --device cpu|gpu OUTPUT_ROOT [PART ...]
#   PART: accuracy           the 8 validation sources, their pytdi truth over DAYS days (one
#                            process per source, side by side) and both responses against it
#                            (the default)
#         truth_chains       the truth with and without lisainstrument, 2 days, the first source
#                            of each frequency
#         accuracy_response  both responses against LAT's pyResponseTDI as the injection
#         step0              the 8 cases of the step 0 truth on its grid, both responses: the
#                            truth without lisainstrument over DAYS days and with it over 30
#                            days (one process per case and chain, side by side), then a run over
#                            DAYS days and two over 30 days, one per chain
# Environment: PYTHON (default python), DFFR_PYTHON (the Python of the standalone dffr
#   environment, for the truth), DAYS (default 365), N_SIDE_BINS (default 60, side bins of the
#   templates; production takes 10).
set -euo pipefail

usage="usage: run_step6.sh --device cpu|gpu OUTPUT_ROOT [PART ...]"
[ "${1:-}" = "--device" ] || { echo "$usage" >&2 && exit 2; }
device=${2:?$usage}
case "$device" in cpu | gpu) ;; *) echo "$usage" >&2 && exit 2 ;; esac
output_root=${3:?$usage}
shift 3
parts=("$@")
[ ${#parts[@]} -gt 0 ] || parts=(accuracy)

python_bin=${PYTHON:-python}
dffr_python=${DFFR_PYTHON:-}
days=${DAYS:-365}
n_side_bins=${N_SIDE_BINS:-60}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
accuracy_dir="$output_root/validation/accuracy"
truth_dir="$accuracy_dir/truth"
mkdir -p "$truth_dir"

# ! One thread for OpenMP and BLAS: the truth runs side by side, one per CPU.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# ! On a node without an NVIDIA driver the JAX CUDA plugin crashes in its version check.
[ "$device" = gpu ] || export JAX_PLATFORMS=cpu
export PYTHONWARNINGS=ignore::DeprecationWarning

require_dffr_python() {
    [ -n "$dffr_python" ] || { echo "DFFR_PYTHON is not set" >&2; return 2; }
}

source_names() {
    "$python_bin" -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['sources']))" \
        "$1"
}

wait_all() {
    local status=0 pid
    for pid in "$@"; do
        wait "$pid" || status=1
    done
    return "$status"
}

write_sources() {
    "$python_bin" "$script_dir/stft_accuracy.py" --write-sources "$truth_dir/sources.json" \
        --days "$days"
}

accuracy() {
    require_dffr_python
    write_sources
    local pids=() name
    for name in $(source_names "$truth_dir/sources.json"); do
        "$dffr_python" "$script_dir/td_truth.py" --sources "$truth_dir/sources.json" \
            --output "$truth_dir" --chain gws --names "$name" > "$truth_dir/$name.log" 2>&1 &
        pids+=($!)
    done
    wait_all "${pids[@]}" || { echo "a truth run failed; see $truth_dir" >&2; return 1; }
    "$python_bin" "$script_dir/stft_accuracy.py" --device "$device" \
        --output "$accuracy_dir/year_pytdi_n$n_side_bins" --truth "$truth_dir" --days "$days" \
        --n-side-bins "$n_side_bins"
}

truth_chains() {
    require_dffr_python
    write_sources
    # * The first source of each frequency, as the convention check of stft_accuracy.py takes.
    "$dffr_python" "$script_dir/td_truth.py" --sources "$truth_dir/sources.json" \
        --output "$accuracy_dir/truth_chains" --chain both --days 2 \
        --names 0.5mHz_fdotmax_sky0 20mHz_fdotmax_sky0
}

accuracy_response() {
    "$python_bin" "$script_dir/stft_accuracy.py" --device "$device" \
        --output "$accuracy_dir/year_pyresponsetdi_n$n_side_bins" --days "$days" \
        --n-side-bins "$n_side_bins"
}

step0() {
    require_dffr_python
    local step0_dir="$accuracy_dir/step0" pids=() name accuracy
    mkdir -p "$step0_dir/truth" "$step0_dir/truth_instrument_30d"
    "$python_bin" "$script_dir/stft_accuracy.py" --source-set step0 \
        --write-sources "$step0_dir/truth/sources.json" --days "$days"
    # * The chain of step 0, with lisainstrument, needs about 6 GB per case and 30 days.
    for name in $(source_names "$step0_dir/truth/sources.json"); do
        "$dffr_python" "$script_dir/td_truth.py" --sources "$step0_dir/truth/sources.json" \
            --output "$step0_dir/truth" --chain gws --names "$name" \
            > "$step0_dir/truth/$name.log" 2>&1 &
        pids+=($!)
        "$dffr_python" "$script_dir/td_truth.py" --sources "$step0_dir/truth/sources.json" \
            --output "$step0_dir/truth_instrument_30d" --chain instrument --days 30 --names "$name" \
            > "$step0_dir/truth_instrument_30d/$name.log" 2>&1 &
        pids+=($!)
    done
    wait_all "${pids[@]}" || { echo "a truth run failed; see $step0_dir" >&2; return 1; }
    accuracy=("$python_bin" "$script_dir/stft_accuracy.py" --source-set step0 --device "$device"
        --n-side-bins "$n_side_bins")
    "${accuracy[@]}" --output "$step0_dir/year_n$n_side_bins" --truth "$step0_dir/truth" \
        --days "$days"
    "${accuracy[@]}" --output "$step0_dir/30d_instrument_n$n_side_bins" \
        --truth "$step0_dir/truth_instrument_30d" --days 30
    "${accuracy[@]}" --output "$step0_dir/30d_gws_n$n_side_bins" --truth "$step0_dir/truth" \
        --days 30
}

for part in "${parts[@]}"; do
    echo "[step6] $(date +%T) part $part"
    case "$part" in
    accuracy | truth_chains | accuracy_response | step0) "$part" ;;
    *) echo "unknown part $part" >&2 && exit 2 ;;
    esac
done
echo "[step6] $(date +%T) done"
