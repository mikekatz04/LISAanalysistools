#!/usr/bin/env bash
# * Step 6 of the DFFR integration (validation), the parts run so far: both responses against
# * the STFT of a time-domain truth with both data windows (items 1 and 2, the second on the cases
# * and grid of the step 0 truth), the checks behind that truth, the likelihood surfaces of four
# * waveforms against it (item 3), and their mismatch, SNR and likelihood surfaces on the
# * Mojito-light data (item 4).
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
#         surfaces           the likelihood surfaces of the 8 validation sources against the
#                            truth of part accuracy (one process per source, SURFACE_PROCESSES
#                            side by side), then the combined report and figures
#         mojito             the 5 galactic binaries of highest frequency in the Mojito-light data
#                            (MOJITO_FOLDER), preprocessed as the global fit does over 1.92 years:
#                            mismatch, SNR and likelihood surfaces of the four waveforms
#         globalfit_smoke    the high-frequency GB global fit of the Mojito-light data
#                            (gb_high_freq_globalfit_settings.py) for a few iterations, both
#                            responses one after the other, into validation/globalfit_smoke/
#         globalfit_tdi_on_the_fly, globalfit_dffr
#                            one full run of that response (1.45 days of wall clock) into
#                            GLOBAL_FIT_RUNS_DIR/gb_high_freq_<response>/ (run_globalfit.sbatch)
# Environment: PYTHON (default python), DFFR_PYTHON (the Python of the standalone dffr
#   environment, for the truth), DAYS (default 365), N_SIDE_BINS (default 60, side bins of the
#   templates; production takes 10), SURFACE_PROCESSES (default 4, part surfaces), MOJITO_FOLDER
#   (default /project/lisa_nlddpc/Data/mojito_light, parts mojito and globalfit_*),
#   GLOBAL_FIT_RUNS_DIR (parts globalfit_tdi_on_the_fly and globalfit_dffr).
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
surface_processes=${SURFACE_PROCESSES:-4}
mojito_folder=${MOJITO_FOLDER:-/project/lisa_nlddpc/Data/mojito_light}
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

surfaces() {
    local surfaces_dir="$output_root/validation/surfaces" running=() name status=0
    [ -f "$truth_dir/sources.json" ] || { echo "run part accuracy first" >&2; return 2; }
    mkdir -p "$surfaces_dir"
    for name in $(source_names "$truth_dir/sources.json"); do
        # ! At most SURFACE_PROCESSES at a time: eight side by side ran out of the A100's 40 GB.
        if [ ${#running[@]} -ge "$surface_processes" ]; then
            wait "${running[0]}" || status=1
            running=("${running[@]:1}")
        fi
        "$python_bin" "$script_dir/likelihood_surfaces.py" --device "$device" --truth "$truth_dir" \
            --output "$surfaces_dir" --days "$days" --sources "$name" --no-summary \
            > "$surfaces_dir/$name.log" 2>&1 &
        running+=($!)
    done
    wait_all "${running[@]}" || status=1
    [ "$status" = 0 ] || { echo "a surface run failed; see $surfaces_dir" >&2; return 1; }
    "$python_bin" "$script_dir/likelihood_surfaces.py" --output "$surfaces_dir" --summarize-only
}

mojito() {
    local mojito_dir="$output_root/validation/mojito"
    mkdir -p "$mojito_dir"
    # * The first run preprocesses the Mojito streams into $mojito_dir/data (about 10 GB of memory).
    "$python_bin" "$script_dir/mojito_dffr_validation.py" --device "$device" \
        --output "$mojito_dir" --mojito "$mojito_folder" > "$mojito_dir/run.log" 2>&1
}

# ! A second MPI start in one Slurm step hangs, so each global-fit run starts as an MPI singleton.
singleton_env=()
for name in $(compgen -e | grep -E '^(SLURM|PMI|PMIX|OMPI)'); do
    singleton_env+=(-u "$name")
done

globalfit_run() {
    local settings_function=$1 runs_dir=$2 name=$3 interval_s=$4 monitor status=0
    local run_dir="$runs_dir/$name"
    local backend="$run_dir/${name}_parameter_estimation_main.h5"
    # ! The global fit resumes from an existing backend, so a new run needs a clean folder.
    [ ! -e "$backend" ] || { echo "$backend exists; move it away first" >&2; return 2; }
    mkdir -p "$run_dir"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l "$interval_s" \
        > "$run_dir/gpu_memory.csv" 2>&1 &
    monitor=$!
    env "${singleton_env[@]}" GLOBAL_FIT_RUNS_DIR="$runs_dir" MOJITO_FOLDER="$mojito_folder" \
        "$python_bin" -u "$script_dir/../run_global.py" \
        -sfp "$script_dir/gb_high_freq_globalfit_settings.py" -sff "$settings_function" \
        > "$run_dir/run.log" 2>&1 || status=1
    kill "$monitor" 2> /dev/null || true
    return "$status"
}

globalfit_smoke() {
    local smoke_dir="$output_root/validation/globalfit_smoke" status=0 response
    for response in tdi_on_the_fly dffr; do
        globalfit_run "get_${response}_smoke_settings" "$smoke_dir" \
            "gb_high_freq_${response}_smoke" 10 || status=1
    done
    return "$status"
}

globalfit_response() {
    local runs_dir=${GLOBAL_FIT_RUNS_DIR:?set GLOBAL_FIT_RUNS_DIR for the full global-fit runs}
    globalfit_run "get_$1_settings" "$runs_dir" "gb_high_freq_$1" 60
}

for part in "${parts[@]}"; do
    echo "[step6] $(date +%T) part $part"
    case "$part" in
    accuracy | truth_chains | accuracy_response | step0 | surfaces | mojito | globalfit_smoke) "$part" ;;
    globalfit_tdi_on_the_fly | globalfit_dffr) globalfit_response "${part#globalfit_}" ;;
    *) echo "unknown part $part" >&2 && exit 2 ;;
    esac
done
echo "[step6] $(date +%T) done"
