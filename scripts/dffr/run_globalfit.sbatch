#!/bin/bash
# * One full run of DFFR validation item 5 (the high-frequency GB global fit, 1.45 days of wall
# * clock) on one A100, through the step 6 driver. Submit from the lisa-analysis-tools root:
#
#   PYTHON=<python> GLOBAL_FIT_RUNS_DIR=<folder> sbatch --job-name=gf_dffr \
#       --output=<folder>/gb_high_freq_dffr/slurm_%j.out scripts/dffr/run_globalfit.sbatch \
#       dffr OUTPUT_ROOT
#
# ! Slurm does not create the folder of --output, so make <folder>/gb_high_freq_<response> first.
# ? A gpu_a100_22c node has two A100s and 44 CPUs, so one A100 comes with 22 CPUs and 176000 MB.
#SBATCH --partition=gpu_a100_22c
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=22
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=1-12:00:00
set -euo pipefail
usage="usage: run_globalfit.sbatch tdi_on_the_fly|dffr OUTPUT_ROOT"
response=${1:?$usage}
output_root=${2:?$usage}

echo "Job $SLURM_JOB_ID ($response) on $(hostname), $(date)"
nvidia-smi -L
# ! sbatch runs a copy of this file, so the driver is found from the submission folder.
"$SLURM_SUBMIT_DIR/scripts/dffr/run_step6.sh" --device gpu "$output_root" "globalfit_$response"
echo "Finished $(date)"
