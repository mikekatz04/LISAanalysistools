#!/usr/bin/env bash
# * The checks of the DFFR response of STFTGBComputations next to the existing one: golden
# * outputs, timings, kernel resources, DFFR against the existing response, both against a
# * time-domain truth, likelihood surfaces of four waveforms, six waveforms on the Mojito-light
# * data, and the galactic-binary block of the global fit with each response.
# The results of the integration are in docs/response/dffr/results.md.
#
# Usage: run_dffr.sh --device cpu|gpu OUTPUT_ROOT [PART ...]
#   PART: environment        node, commits and package versions, into regression/
#         golden             the outputs of the seven STFT methods and the Fisher matrix of both
#                            responses on the fixtures of stft_fixtures.py (the default); with
#                            GOLDEN_REFERENCE, compared with the captures of that folder
#         against_existing   DFFR against the existing response on the production fixtures
#         benchmarks         three timing runs of each response, DFFR against the existing
#                            response and, with BENCH_REFERENCE, each against that folder's runs
#         resources          registers, stack and shared memory of the linked kernels for the
#                            A100 and the H100 (nvcc, no GPU needed)
#         accuracy           the 8 validation sources, their pytdi truth over DAYS days (one
#                            process per source, side by side) and both responses against it
#         truth_chains       the truth with and without lisainstrument, 2 days, the first source
#                            of each frequency
#         accuracy_response  both responses against LAT's pyResponseTDI as the injection
#         step0              the 8 cases of the standalone package's truth comparison on its
#                            grid, both responses: the truth without lisainstrument over DAYS
#                            days and with it over 30 days, then a run over DAYS days and two
#                            over 30 days, one per chain
#         surfaces           the likelihood surfaces of the sources of part accuracy against its
#                            truth (SURFACE_PROCESSES side by side), then the combined report
#         mojito             the 5 galactic binaries of highest frequency in the Mojito-light data
#                            (MOJITO_FOLDER), preprocessed as the global fit does: mismatch, SNR
#                            and likelihood surfaces of six waveforms
#         globalfit_smoke    the high-frequency GB global fit of the Mojito-light data
#                            (gb_high_freq_globalfit_settings.py) for a few iterations, both
#                            responses one after the other, into validation/globalfit_smoke/
#         globalfit_tdi_on_the_fly, globalfit_dffr
#                            one full run of that response (1.45 days of wall clock) into
#                            GLOBAL_FIT_RUNS_DIR/gb_high_freq_<response>/, on the GPU at hand
#         submit_tdi_on_the_fly, submit_dffr
#                            the same full run submitted to Slurm, on one A100 of gpu_a100_22c
# Environment: PYTHON (default python), DFFR_PYTHON (the Python of the standalone dffr
#   environment, for the truth), MOJITO_L1 (a Mojito L1 file, for the l1 fixtures of golden and
#   benchmarks), GOLDEN_REFERENCE and BENCH_REFERENCE (folders of earlier runs of those parts),
#   GBGPU_BUILD_DIR (GBGPU build tree, default env/build/GBGPU next to the clones),
#   REFERENCE_BUILD_DIR (an older GBGPU build tree, for resources), BENCH_ARGS (extra arguments
#   for stft_benchmarks.py), DAYS (default 365), N_SIDE_BINS (default 60; production takes 10),
#   SURFACE_PROCESSES (default 4), MOJITO_FOLDER (default /project/lisa_nlddpc/Data/mojito_light),
#   GLOBAL_FIT_RUNS_DIR (the full global-fit runs).
set -euo pipefail

usage="usage: run_dffr.sh --device cpu|gpu OUTPUT_ROOT [PART ...]"
[ "${1:-}" = "--device" ] || { echo "$usage" >&2 && exit 2; }
device=${2:?$usage}
case "$device" in cpu | gpu) ;; *) echo "$usage" >&2 && exit 2 ;; esac
output_root=${3:?$usage}
shift 3
parts=("$@")
[ ${#parts[@]} -gt 0 ] || parts=(golden)

python_bin=${PYTHON:-python}
dffr_python=${DFFR_PYTHON:-}
mojito_l1=${MOJITO_L1:-}
days=${DAYS:-365}
n_side_bins=${N_SIDE_BINS:-60}
surface_processes=${SURFACE_PROCESSES:-4}
mojito_folder=${MOJITO_FOLDER:-/project/lisa_nlddpc/Data/mojito_light}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lat_root=$(cd "$script_dir/../.." && pwd)
gbgpu_root=$(cd "$lat_root/../GBGPU" && pwd)
gbgpu_build_dir=${GBGPU_BUILD_DIR:-$lat_root/../env/build/GBGPU}
regression_dir="$output_root/regression"
benchmark_dir="$output_root/benchmarks"
accuracy_dir="$output_root/validation/accuracy"
truth_dir="$accuracy_dir/truth"
mkdir -p "$regression_dir" "$benchmark_dir" "$truth_dir"

# * The fixtures of the existing response, and the production ones with XYZ channels, which are
# * the fixtures of DFFR as well. The l1 fixture needs a Mojito L1 file.
existing_configs=(rect tukey_mid astro esa_tdi2_envelope esa_tdi1_const esa_aet_diag)
dffr_configs=(esa_tdi2_envelope esa_tdi1_const)
l1_configs=(l1_tdi2_envelope)
# ? The CPU benchmarks leave out the two swap kernels, which took 70 % of a CPU run.
bench_swap_args=()
[ "$device" = gpu ] || bench_swap_args=(--without-swap)

# ! One thread for OpenMP and BLAS keeps CPU outputs reproducible from run to run.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# ! On a node without an NVIDIA driver the JAX CUDA plugin crashes in its version check.
[ "$device" = gpu ] || export JAX_PLATFORMS=cpu
export PYTHONWARNINGS=ignore::DeprecationWarning

golden() { "$python_bin" "$script_dir/stft_golden_outputs.py" "$@"; }

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

environment() {
    {
        date --iso-8601=seconds
        hostname
        lscpu | grep -E "^Model name|^CPU\(s\)"
        nvidia-smi -L 2> /dev/null || echo "no GPU"
        echo "LAT $(git -C "$lat_root" rev-parse HEAD) on $(git -C "$lat_root" branch --show-current)"
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
        env | grep -E "^(OMP|OPENBLAS|MKL)_NUM_THREADS|^JAX_PLATFORMS|^SLURM_JOB_ID|^CUDA_VISIBLE" | sort
    } > "$regression_dir/environment_$device.txt" 2>&1
}

capture() {
    # * One capture, compared with the reference folder's capture of the same name when there is
    # * one. On the GPU, fill_global_stft adds atomically and agrees to rounding only, so a
    # * difference is recorded, not fatal.
    local stem=$1 status
    shift
    golden --capture "$regression_dir/$stem.npz" --device "$device" "$@"
    if [ -n "${GOLDEN_REFERENCE:-}" ] && [ -f "$GOLDEN_REFERENCE/$stem.npz" ]; then
        status=0
        golden --compare-files "$GOLDEN_REFERENCE/$stem.npz" "$regression_dir/$stem.npz" \
            > "$regression_dir/${stem}_vs_reference.txt" 2>&1 || status=$?
        echo "compare exit status $status" >> "$regression_dir/${stem}_vs_reference.txt"
        echo "[dffr] $stem against the reference: $(tail -n 2 "$regression_dir/${stem}_vs_reference.txt" | head -n 1)"
    fi
}

golden_part() {
    capture "golden_$device" --configs "${existing_configs[@]}"
    capture "golden_dffr_$device" --response dffr --configs "${dffr_configs[@]}"
    if [ -n "$mojito_l1" ]; then
        capture "golden_${device}_l1" --configs "${l1_configs[@]}" --mojito-l1 "$mojito_l1"
        capture "golden_dffr_${device}_l1" --response dffr --configs "${l1_configs[@]}" \
            --mojito-l1 "$mojito_l1"
    fi
}

against_existing() {
    "$python_bin" "$script_dir/dffr_against_existing.py" --device "$device" \
        --output "$regression_dir/against_existing_$device.npz" \
        | tee "$regression_dir/against_existing_$device.txt"
}

benchmark_three_times() {
    # * Three runs in separate processes give the run-to-run spread a ratio is read against.
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

benchmark_fixture() {
    # * Both responses on one fixture; DFFR against the existing response, and each against the
    # * reference folder's runs of the same name.
    local stem=$1
    shift
    benchmark_three_times "${device}_$stem" "$@"
    benchmark_three_times "${device}_${stem}_dffr" "$@" --response dffr
    compare_runs "$benchmark_dir/${device}_$stem" "$benchmark_dir/${device}_${stem}_dffr" \
        "$benchmark_dir/compare_${device}_${stem}_dffr_vs_existing.txt"
    if [ -n "${BENCH_REFERENCE:-}" ]; then
        for variant in "" _dffr; do
            compare_runs "$BENCH_REFERENCE/${device}_$stem$variant" \
                "$benchmark_dir/${device}_$stem$variant" \
                "$benchmark_dir/compare_${device}_$stem${variant}_vs_reference.txt"
        done
    fi
}

benchmarks() {
    benchmark_fixture esa --config esa_tdi2_envelope
    [ -z "$mojito_l1" ] || benchmark_fixture l1 --config l1_tdi2_envelope --mojito-l1 "$mojito_l1"
}

resources() {
    # * sm_80 is the A100 of the timing gate, sm_90 the H100. The same source gives different
    # * register counts on the two, so a change of the kernels is checked on both.
    local reference_args=()
    [ -z "${REFERENCE_BUILD_DIR:-}" ] || reference_args=(--reference-build-dir "$REFERENCE_BUILD_DIR")
    for arch in 80 90; do
        "$python_bin" "$script_dir/kernel_resources.py" --build-dir "$gbgpu_build_dir" \
            "${reference_args[@]}" --arch "$arch" --output-dir "$regression_dir/kernel_resources" \
            > "$regression_dir/kernel_resources_sm$arch.txt" 2>&1 &
    done
    wait
    cat "$regression_dir"/kernel_resources_sm*.txt
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
    # * The chain with lisainstrument needs about 6 GB per case and 30 days.
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

submit_response() {
    # * One full run on one A100 of gpu_a100_22c: a node has two A100s and 44 CPUs, so one A100
    # * comes with 22 CPUs and 176 000 MB. The job calls this driver with part globalfit_<response>.
    local response=$1
    local runs_dir=${GLOBAL_FIT_RUNS_DIR:?set GLOBAL_FIT_RUNS_DIR for the full global-fit runs}
    local run_dir="$runs_dir/gb_high_freq_$response"
    # ! Slurm does not create the folder of --output.
    mkdir -p "$run_dir"
    sbatch --partition=gpu_a100_22c --gpus=a100:1 --cpus-per-task=22 --mem-per-cpu=8000M \
        --time=1-12:00:00 --job-name="gf_$response" --output="$run_dir/slurm_%j.out" \
        --export=ALL,PYTHON="$python_bin",GLOBAL_FIT_RUNS_DIR="$runs_dir",MOJITO_FOLDER="$mojito_folder" \
        --wrap "echo \"Job \$SLURM_JOB_ID ($response) on \$(hostname), \$(date)\"; nvidia-smi -L;
            '$script_dir/run_dffr.sh' --device gpu '$output_root' globalfit_$response;
            echo \"Finished \$(date)\""
}

for part in "${parts[@]}"; do
    started=$SECONDS
    echo "[dffr] $(date +%T) part $part on $(hostname)"
    case "$part" in
    golden) golden_part ;;
    environment | against_existing | benchmarks | resources | accuracy | truth_chains | \
        accuracy_response | step0 | surfaces | mojito | globalfit_smoke) "$part" ;;
    globalfit_tdi_on_the_fly | globalfit_dffr) globalfit_response "${part#globalfit_}" ;;
    submit_tdi_on_the_fly | submit_dffr) submit_response "${part#submit_}" ;;
    *) echo "unknown part $part" >&2 && exit 2 ;;
    esac
    echo "[dffr] $(date +%T) part $part done in $((SECONDS - started)) s"
done
