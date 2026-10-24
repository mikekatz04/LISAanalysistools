#!/bin/bash
# ============================================================================
# FLAGSHIP global fit -- 24mo (Tobs = 720 d). GENERATED FILE, DO NOT
# EDIT: python scripts/flagship/build.py rewrites it from
#   scripts/flagship/launcher.sh.in   (every run  -- level 1)
#   scripts/flagship/runs.py :: RUNS["24mo"]   (this run -- level 2)
# One-off changes (level 3): KNOB=value ./scripts/flagship/submit_gf_24mo.sh for any
# ${K:-v} knob, or  python scripts/flagship/build.py 24mo --set KNOB=value
# --out scripts/flagship/oneoff/<file>.sh  (rewrites a hard export too).
# Guard: tests/test_flagship_launchers.py (committed == rebuilt; the per-run
# deltas; the 9mo == scripts/fstat_proposal/submit_gf_9mo_v9.sh in every line
# bash executes). The 9mo v9 is the base of the series (user ruling 2026-10-11):
# every run follows its pattern and differs ONLY in the lines listed here.
#
# THIS RUN'S VALUES (runs.py):
#   TOBS_TARGET=62208000 (720 d) -> Nf 1440 x Nt 17280 x dt 2.5 s;
#   SIGHET_NT_LAYER=480 (Tobs / 36 h); EDGE_CROP_WAVELETS=120 (sig-het guard needs >= 95);
#   GB_NLEAVES_MAX=25000, GB_N_SUBBANDS=2048 per GPU, GB_RJ_INMODEL_CHUNK=8192;
#   MBHB_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 (t_c < 720 + 14 d); EMRI / SOBHB: all;
#   STORE_DIR /shared/data/global_fit_output/gf_prod_24mo_flagship/, BASE_FILE_NAME gf_prod_24mo,
#   job gf24mo_flagship, log gf24mo_flagship_<jobid>.log; NGPUS default 4.
#   PARENT = the 18mo (540 d): GF_SEED_STORE /shared/data/global_fit_output/gf_prod_18mo_flagship/gf_prod_18mo_testing.h5
#     (GB warm start fitted from its last GB_WARM_START_LAST_K stored rows -> warmstart/gf_prod_18mo_flagship_refereed.npz, auto-built
#     on the first launch when missing; the psd pinned at its maxlogL cold walker (galfor stays the
#     hand-set offline estimate, ruling 2026-09-25); the source
#     warm start copies its SNR > 10 leaves by catalogue id: MBHB 0,2,3,4,5,6,7,9,10,12,13,15,16,18,19, every EMRI / SOBHB; the monitor page overlays
#     it: GF_MONITOR_ADD_INFO=/shared/data/global_fit_output/gf_prod_18mo_flagship).
#     The parent store must be COMPLETE and readable when this launches.
#
# NOTES:
#   All 20 catalogue MBHBs merge inside 720 d (the last, 14, at 712.7 d).
#   EDGE_CROP 120 (guard needs 95). Same unverified-at-this-Tobs list as the
#   18mo; the F-stat host-mapped float32 grid (TODO) is the real fix for the
#   stage-B memory at 1 yr and beyond.
#
# RECIPE (the 9mo pattern, ruling 2026-10-07; resolved by
#   python scripts/fstat_proposal/run_combined_staged.py --print-recipe):
#   gb_search_seed x3 -> source_search -> gb_search_1 -> gb_search_2 -> full_pe
#
# LAUNCH (from the LAT root; a FRESH store on the first submit, a RESUME after):
#   ./scripts/flagship/submit_gf_24mo.sh
# History of every knob below: scripts/fstat_proposal/submit_gf_9mo_v9.sh (the
# launched 9mo, with the 6mo v9 / v8 headers it inherited) and git log.
# ============================================================================
# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf24mo_flagship   # job name (runs.py)
#SBATCH --partition=gpu-80-spot   # DEFAULT partition (2-GPU flow); the
                                  # NGPUS self-dispatch below overrides it
#SBATCH --gres=gpu:2              # DEFAULT 2 GPUs (GPUS below are LOCAL indices)
#SBATCH --nodes=1                 # DEFAULT 1 node; NGPUS=4 -> 2 nodes (below)
#SBATCH --ntasks=3                # DEFAULT/fallback only (legacy layout: main
                                  # + stopped spare + SAVER rank, mpiexec -n 3);
                                  # the NGPUS self-dispatch below computes the
                                  # real rank count from GPUS_PER_RANK/
                                  # RANKS_PER_GPU/GF_LEGACY_RANK_LAYOUT and
                                  # passes --ntasks explicitly on `exec sbatch`
#SBATCH --cpus-per-task=2
#SBATCH --mem=0                   # whole-node memory
#SBATCH --time=24:00:00
#SBATCH --output=/shared/data/global_fit_output/gf24mo_flagship_%j.log     # combined stdout+stderr (captures [MAXLOGL]/[BENCH])
# ----------------------------------------------------------------------------

set -euo pipefail

# ---- GPU-count self-dispatch + rank-layout knobs (2026-09-16: rank count
# ---- now derives from the GPU count) ---------------------------------
# NGPUS=2 -> gpu-80-spot, 1 node x 2 GPUs (the unchanged partition);
# NGPUS=4 -> gpu-80-ondemand, 2 NODES x 2 GPUs each (THE DEFAULT since
# 2026-10-07, Mike's production line) -- the cluster's real 4-GPU
# allocation shape (there is no single 4-GPU node; see the design spec's
# "Context" section). #SBATCH lines are static comments, so neither the
# partition/node/task count can follow an env var through a plain
# `sbatch <script>`. Instead, run this script DIRECTLY to pick the GPU
# count and it submits itself with the matching flags:
#
#     ./submit_gf_24mo.sh         # THE INTENDED INVOCATION (= NGPUS=4)
#     NGPUS=2 ./submit_gf_24mo.sh # gpu-80-spot, 1 node x gpu:2
#     sbatch  ./submit_gf_24mo.sh # legacy flow: header defaults above
#                                  # (2 GPUs, gpu-80-spot, --ntasks=3)
#
# GPUS_PER_RANK (empty = AUTO) / RANKS_PER_GPU (default 1) size the compute
# rank count: N_COMPUTE = NGPUS * RANKS_PER_GPU / GPUS_PER_RANK.
#
# LAYOUT DEFAULT (user ruling 2026-09-16): the walker-block layout
# (GF_LEGACY_RANK_LAYOUT=0) is the default at EVERY NGPUS. At NGPUS=2 that is
# head + 1 compute rank + saver (--ntasks=3, one GPU each, NWALKERS/2 walkers
# per rank); the WP7 transport gates (Steps 0-2, 4) passed on the cluster and
# the 6mo campaign run itself is the statistical read (Step 3 was skipped by
# ruling). GF_LEGACY_RANK_LAYOUT=1 still selects today's single-compute-rank
# layout for a 1-node job (rollback knob). NGPUS=4 cannot use the legacy
# layout (it cannot span nodes), so it forces GF_LEGACY_RANK_LAYOUT=0
# regardless of any pre-set value.
#
# Inside the job, the GPU list further below derives from what slurm
# ACTUALLY granted (SLURM_GPUS_ON_NODE), so a manual
# `sbatch --partition=gpu-80-spot --gres=gpu:2 --nodes=2 <script>` also
# works.
#
# ---- GF_GPU_ROUTING: split EACH WALKER over several GPUs (2026-10-15) -------
# GPUS_PER_RANK>1 PACKS walkers onto one rank (its GPUs shard the rank's
# walker rows); it never splits a walker. 9mo job 764 (NGPUS=8
# GPUS_PER_RANK=2 on the 4-walker store: 1 walker per 2-GPU rank) died at the
# first GB gather on an EMPTY second shard. What splits a walker is the
# replica routing, opt-in with GF_GPU_ROUTING=1: N_COMPUTE = n_blocks x R,
# n_blocks = gcd(NWALKERS, N_COMPUTE), R ranks share a walker block, ONE GPU
# per rank. Inside a block GB/VGB split by band range and MBH/EMRI/SOBBH by
# likelihood rows; PSD/galfor propose on the block lead and publish to the
# other replica(s). The production line on the 4-walker store:
#     NGPUS=8 GF_GPU_ROUTING=1 ./submit_gf_24mo.sh  # 8 ranks = 4 blocks x R=2
# With the knob ON, NWALKERS is NEVER rounded (a resume refuses a changed
# walker count anyway). With it unset every shape is exactly what it was,
# except that the GPUS_PER_RANK trap above is refused before sbatch.
_gf_routing_on() {
  case "$(printf '%s' "${GF_GPU_ROUTING:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
    1|true|yes|on) return 0 ;;
  esac
  return 1
}
_gf_gcd() { local a=$1 b=$2 t; while [ "$b" -ne 0 ]; do t=$b; b=$(( a % b )); a=$t; done; echo "$a"; }
# _gf_gpus_trap NWALKERS_PER_RANK GPUS_PER_RANK NWALKERS N_COMPUTE: refuse
# (return 2) a rank that would drive more GPUs than it holds walkers.
_gf_gpus_trap() {
  local per=$1 k=${2:-1} nw=$3 nc=$4
  if [ "${k}" -gt 1 ] && [ "${per}" -lt "${k}" ]; then
    echo "[SUBMIT] REFUSED: GPUS_PER_RANK=${k} but each compute rank holds ${per} walker(s) (NWALKERS=${nw} over N_COMPUTE=${nc})."
    echo "[SUBMIT]   A rank's GPUs shard its WALKERS (GPUS_PER_RANK packs walkers, it never splits one), so"
    echo "[SUBMIT]   $(( k - per )) GPU(s) per rank would hold no walker and the build would die at the first GB"
    echo "[SUBMIT]   gather ('Per-AC element size mismatch across shards', 9mo job 764). Two routes:"
    echo "[SUBMIT]   (1) split every walker over several GPUs: unset GPUS_PER_RANK and set GF_GPU_ROUTING=1"
    echo "[SUBMIT]       (one GPU per rank, R ranks per walker block);"
    echo "[SUBMIT]   (2) keep GPUS_PER_RANK=${k}: NWALKERS a multiple of $(( nc * k )) (a fresh or rescaled store)."
    return 2
  fi
  return 0
}
# _gf_routing_layout NWALKERS N_COMPUTE GPUS_PER_RANK [RANKS_PER_BLOCK]: print
# the replica factorization; refuse (return 2) a shape build_layout refuses.
_gf_routing_layout() {
  local nw=$1 nc=$2 k=${3:-1} rpb=${4:-} nb R
  if [ "${nw}" -lt 1 ] || [ "${nc}" -lt 1 ]; then
    echo "[SUBMIT] REFUSED: GF_GPU_ROUTING=1 needs NWALKERS >= 1 and N_COMPUTE >= 1 (got ${nw}, ${nc})."
    return 2
  fi
  if [ -n "${rpb}" ]; then
    R=${rpb}
    if [ "${R}" -lt 1 ] || [ $(( nc % R )) -ne 0 ]; then
      echo "[SUBMIT] REFUSED: RANKS_PER_BLOCK=${R} must be >= 1 and divide N_COMPUTE=${nc}."
      return 2
    fi
    nb=$(( nc / R ))
    if [ $(( nw % nb )) -ne 0 ]; then
      echo "[SUBMIT] REFUSED: RANKS_PER_BLOCK=${R} gives ${nb} walker block(s), which does not divide NWALKERS=${nw}."
      return 2
    fi
  else
    nb=$(_gf_gcd "${nw}" "${nc}")
    R=$(( nc / nb ))
  fi
  if [ "${R}" -lt 1 ] || [ $(( nb * R )) -ne "${nc}" ]; then
    echo "[SUBMIT] REFUSED: N_COMPUTE=${nc} is not n_blocks=${nb} x R=${R}."
    return 2
  fi
  _gf_gpus_trap $(( nw / nb )) "${k}" "${nw}" "${nc}" || return 2
  echo "[SUBMIT] GF_GPU_ROUTING=1: replica factorization NWALKERS=${nw} on N_COMPUTE=${nc} -> n_blocks=${nb} walker block(s) of $(( nw / nb )) walker(s) x R=${R} replica rank(s) per block, one GPU per rank (n_blocks = gcd(NWALKERS, N_COMPUTE); NWALKERS is NOT rounded)"
  if [ "${R}" -gt 1 ]; then
    echo "[SUBMIT]   each walker block is split over its ${R} ranks: GB/VGB by band range, MBH/EMRI/SOBBH likelihood rows by walker; PSD/galfor propose on the block lead and publish to the other replica(s)"
  fi
  if [ "${nb}" -eq 1 ] && [ "${nw}" -gt 1 ] && [ "${nc}" -gt 1 ]; then
    echo "[SUBMIT]   WARNING: gcd(${nw}, ${nc}) = 1 -> ONE block replicated on every rank; no walker parallelism."
  fi
  return 0
}
if [ -z "${SLURM_JOB_ID:-}" ]; then
  NGPUS=${NGPUS:-4}                # the run's default (runs.py ngpus_default; 4 = the 2-node on-demand shape since 2026-10-07)
  GPUS_PER_RANK=${GPUS_PER_RANK:-}
  RANKS_PER_GPU=${RANKS_PER_GPU:-1}
  _k=${GPUS_PER_RANK:-1}
  case "${NGPUS}" in
    2) _NGPU_PART=gpu-80-spot; _NODES=1; _GRES=gpu:2 ;;
    # NGPUS=4 -> ON-DEMAND (user ruling 2026-09-18). The 4-GPU shape is
    # 2 nodes x 2 GPUs, and on spot a preemption of EITHER node kills the
    # whole MPI world -- twice the exposure of the 1-node flow for the
    # same work. NGPUS=2 stays on spot: it is a single node and the
    # midit-checkpoint path recovers it cheaply.
    4) _NGPU_PART=${PARTITION:-gpu-80-ondemand}; _NODES=2; _GRES=gpu:2 ;;
    # NGPUS=8 (2026-10-14, timing tests): 4 nodes x 2 GPUs on the same
    # on-demand partition -> N_COMPUTE=8 compute ranks, so the walker
    # count must be a multiple of 8 (a 4-walker store cannot resume on it:
    # clone it to 8 or 16 walkers with scripts/fstat_proposal/
    # rescale_store_walkers.py and launch the clone with NWALKERS=8) --
    # OR keep the 4-walker store and add GF_GPU_ROUTING=1 (2026-10-15): 4
    # walker blocks x R=2 replicas, each walker split over two GPUs.
    8) _NGPU_PART=${PARTITION:-gpu-80-ondemand}; _NODES=4; _GRES=gpu:2 ;;
    *) echo "[SUBMIT] NGPUS=${NGPUS} unsupported (2, 4 or 8)."; exit 2 ;;
  esac
  # NODES=<n> spreads the NGPUS GPUs over n nodes (gres = NGPUS/n per node).
  # One-walker replica gates (2026-09-16 ruling: test ACROSS nodes):
  # NGPUS=2 NODES=2 -> 2 nodes x gpu:1, one replica per node;
  # NGPUS=4 NODES=4 -> 4 nodes x gpu:1. Unset = the NGPUS table above.
  NODES=${NODES:-}
  if [ -n "${NODES}" ]; then
    if [ "${NODES}" -lt 1 ] || [ $(( NGPUS % NODES )) -ne 0 ]; then
      echo "[SUBMIT] NODES=${NODES} must be >= 1 and divide NGPUS=${NGPUS}."; exit 2
    fi
    _NODES=${NODES}; _GRES=gpu:$(( NGPUS / NODES ))
  fi
  if [ "${_NODES}" -gt 1 ]; then
    GF_LEGACY_RANK_LAYOUT=0    # the legacy single-compute-rank layout cannot span nodes
  else
    GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-0}
  fi
  export GF_LEGACY_RANK_LAYOUT
  N_COMPUTE=$(( NGPUS * RANKS_PER_GPU / _k ))
  if [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
    NTASKS=3
  else
    NTASKS=$(( N_COMPUTE + 1 ))
  fi
  # DEDICATED MONITOR RANK (user ruling 2026-10-04: "have a separate rank run
  # the html"). ONE extra task, the HIGHEST world rank, which the driver
  # splits off before the layout is built -- the run's ranks, placement and
  # roles are exactly what they were without it (cyclic over 2 nodes: run
  # ranks A:{0,2,4} B:{1,3} as before, the monitor lands on B). It builds
  # the page + snapshot tar from disk, so the saver never stops receiving
  # (job 717: the save after every in-place build stalled ~400 s with every
  # GPU idle). GF_MONITOR_RANK=0 restores the in-place hook on the saver.
  export GF_MONITOR_RANK=${GF_MONITOR_RANK:-1}
  if [ "${GF_MONITOR_RANK}" = "1" ]; then
    NTASKS=$(( NTASKS + 1 ))
  fi
  if [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
    echo "[SUBMIT] GF_LEGACY_RANK_LAYOUT=1: TODAY's roles (one sampling rank"
    echo "[SUBMIT]   drives all local GPUs; rank 1 stopped spare; rank 2"
    echo "[SUBMIT]   saver) -- set GF_LEGACY_RANK_LAYOUT=0 for the"
    echo "[SUBMIT]   walker-block layout after the WP7 gates."
  else
    echo "[SUBMIT] GF_LEGACY_RANK_LAYOUT=0: walker-block layout: ${N_COMPUTE}"
    echo "[SUBMIT]   compute ranks + 1 saver."
  fi
  # GF_GPU_ROUTING report / the GPUS_PER_RANK trap (functions above), BEFORE
  # sbatch. NWALKERS: the launch line's, else this script's own default.
  _NW_DEFAULT=$(sed -n 's/^export NWALKERS=\${NWALKERS:-\([0-9][0-9]*\)}.*/\1/p' "$0")
  _NW_SUBMIT=${NWALKERS:-${_NW_DEFAULT:-4}}
  _ROUTING_EXPORT=""
  if _gf_routing_on; then
    _ROUTING_EXPORT=",GF_GPU_ROUTING=1"
    if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ]; then
      _gf_routing_layout "${_NW_SUBMIT}" "${N_COMPUTE}" "${_k}" "${RANKS_PER_BLOCK:-}" || exit 2
    fi
  elif [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ "${N_COMPUTE}" -gt 0 ]; then
    # the in-job block rounds NWALKERS UP to a multiple of N_COMPUTE (one-walker
    # replica mode excepted): this is the per-rank count the job would run
    if [ "${_NW_SUBMIT}" -eq 1 ] && [ "${N_COMPUTE}" -gt 1 ]; then
      _NW_PER=1
    else
      _NW_PER=$(( (_NW_SUBMIT + N_COMPUTE - 1) / N_COMPUTE ))
    fi
    _gf_gpus_trap "${_NW_PER}" "${_k}" "${_NW_SUBMIT}" "${N_COMPUTE}" || exit 2
  fi
  echo "[SUBMIT] NGPUS=${NGPUS} -> sbatch --partition=${_NGPU_PART} --gres=${_GRES} --nodes=${_NODES} --ntasks=${NTASKS} (N_COMPUTE=${N_COMPUTE} compute ranks + 1 saver + GF_MONITOR_RANK=${GF_MONITOR_RANK} monitor)"
  _DIST_FLAG=""
  if [ "${_NODES}" -gt 1 ]; then
    _DIST_FLAG="--distribution=cyclic"
  fi
  # CPUs per rank: 2, plus one core per EMRI trajectory-pool worker when emri_pe
  # scores through the direct-to-WDM template (EMRI LIKELIHOOD block below; the
  # two defaults here MUST match that block's exports -- pinned by
  # tests/test_submit_scripts_layout.py::SixMonthEMRIDirectTest).
  _CPT=2
  if [ "${EMRI_LIKELIHOOD:-direct}" = "direct" ]; then
    _CPT=$(( 2 + ${EMRI_TRAJ_WORKERS:-8} ))
  fi
  echo "[SUBMIT] --cpus-per-task=${_CPT} (EMRI_LIKELIHOOD=${EMRI_LIKELIHOOD:-direct}, EMRI_TRAJ_WORKERS=${EMRI_TRAJ_WORKERS:-8})"
  exec sbatch --partition="${_NGPU_PART}" --gres="${_GRES}" --nodes="${_NODES}" \
       --ntasks="${NTASKS}" --cpus-per-task="${_CPT}" ${_DIST_FLAG} \
       --export=ALL,NGPUS="${NGPUS}",GPUS_PER_RANK="${GPUS_PER_RANK}",RANKS_PER_GPU="${RANKS_PER_GPU}",GF_LEGACY_RANK_LAYOUT="${GF_LEGACY_RANK_LAYOUT}",GF_MONITOR_RANK="${GF_MONITOR_RANK}"${_ROUTING_EXPORT} \
       "$0" "$@"
fi

# ---- environment (fill in your activation) ---------------------------------
# module load FILLME_cuda_module
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

# FRESH RUN (2026-08-15, user ruling: "we want to totally restart. This is a
# fresh run now."). A NEW store dir so the previous run's h5/logs/fstat cache
# stay intact for comparison and nothing can silently resume. BASE_FILE_NAME
# stays gf_prod_3mo so every analysis tool (monitor generator, digests) works
# unchanged -- they take the DIRECTORY as their argument.
# env-overridable (2026-09-16) so a COPY of a live run's folder can be continued
# under new code/layout without touching the original (STORE_DIR=<copy> ...).
STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_24mo_flagship/}

# ---- GPU telemetry ---------------------------------------------------------
# Background nvidia-smi sampler: one CSV row per GPU into the run store
# (timestamped per job, so resubmits/resumes add new files rather than
# clobbering). Killed automatically when the job exits. Columns:
# timestamp, index, name, util.gpu [%], util.mem [%], mem.used [MiB],
# mem.total [MiB], power [W], temp [C].
#
# INTERVAL 30 -> 5 s (2026-08-15): 30 s was tuned for 55-minute iterations.
# Post-speedup an iteration is ~5 min and individual moves take SECONDS, so
# 30 s gave ~10 samples/iteration -- far too coarse to attribute utilization
# to a move. At 5 s a 5-min iteration yields ~60 samples/GPU. Cost is
# negligible (a few hundred KB/hour; nvidia-smi polling is cheap) and the
# monitor's gpu_util panel reads it unchanged.
mkdir -p ${STORE_DIR}
GPU_SAMPLE_SEC=${GPU_SAMPLE_SEC:-5}
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l ${GPU_SAMPLE_SEC} > "${GPU_LOG}" &
GPU_SMI_PID=$!
# PER-PROCESS GPU memory sampler. --query-gpu reports only DEVICE totals, so
# it cannot say WHICH process holds the memory. Under `mpiexec -n 3` that is
# exactly the open question: run_combined_staged.py builds on EVERY rank
# before roles resolve, so the saver/spare ranks may be holding GPU
# allocations they never use. Columns: timestamp, gpu_uuid, pid,
# process_name, used_memory [MiB] -- one row per process per GPU. Sampled
# 6x slower than the utilization stream (this changes only when a process
# allocates).
GPU_PROC_LOG=${STORE_DIR}/gpu_procs_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits -l $((GPU_SAMPLE_SEC * 6)) > "${GPU_PROC_LOG}" &
GPU_PROC_PID=$!

# ---- slurm stdout MIRROR into the store dir --------------------------------
# Ported from submit_gf_6mo_sources_probe.sh (noise-merge readout item: the
# multi-GPU probe zip carried NO slurm log, so [GF_TIMING]/[MAXLOGL]/
# [PROBE]/[SMOKE] never travelled). Everything that matters for the timing
# readout goes to STDOUT -- i.e. the --output file above, which lives
# OUTSIDE ${STORE_DIR} -- and the pulls are zips OF ${STORE_DIR}, so that
# file has been missing from every pull. Mirror it in every 30 s: zipping
# the store then captures it automatically, and because this is a copy loop
# rather than an EXIT trap it survives a spot preemption (SIGKILL runs no
# traps). Also note slurm stdout only FLUSHES at job end, so the mirror is
# the only way to see these lines while the job is still running.
SLURM_LOG=/shared/data/global_fit_output/gf24mo_flagship_${SLURM_JOB_ID:-manual}.log
LOG_MIRROR_PID=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ( while true; do
      cp -f "${SLURM_LOG}" "${STORE_DIR}/slurm_stdout_${SLURM_JOB_ID}.log" 2>/dev/null || true
      sleep 30
    done ) &
  LOG_MIRROR_PID=$!
  echo "[LOGMIRROR] ${SLURM_LOG} -> ${STORE_DIR}/slurm_stdout_${SLURM_JOB_ID}.log every 30 s"
fi

trap 'kill ${GPU_SMI_PID} ${GPU_PROC_PID} ${LOG_MIRROR_PID:-} 2>/dev/null || true' EXIT

# ---- threading policy (MPI-only, no OMP) -----------------------------------
export OMP_NUM_THREADS=1

# ---- HDF5 FILE LOCKING (2026-08-15 hang forensics) --------------------------
# Job 210 hung at STARTUP for 11 h: allocation alive, CUDA contexts created
# (10 GB held), 0% GPU on both devices, and the MAIN rank never logged a
# single line -- while a healthy start logs within ~37 s. The first thing the
# main rank does after build() is load_info() -> backend.get_last_sample(),
# i.e. OPEN AND READ the store h5. The previous job had been writing that same
# file 3.5 min earlier, and h5py BLOCKS INDEFINITELY (it does not error) when
# it cannot take the lock -- which is the classic failure on NFS/Lustre, and
# is made likelier here by the saver rank holding the file from a second
# process. Disabling HDF5's own locking is the standard remedy for shared
# filesystems; this run is the only writer, so the lock buys nothing.
export HDF5_USE_FILE_LOCKING=FALSE

# ---- console verbosity ------------------------------------------------------
# WHERE THE DETAIL ALREADY LIVES: the file handler is pinned at DEBUG
# UNCONDITIONALLY, so every per-iteration line ("Number of active leaves
# before proposal", "Current number of active sources in cold chain",
# the [GB_TIMING]/[GB_ACCEPT]/[FSTAT_CTR] records, ...) is ALWAYS written to
#   ${STORE_DIR}/${BASE_FILE_NAME}_artifacts/globalfit_run.log
# regardless of this knob. VERBOSE only MIRRORS them to stdout (the sbatch
# .log). Setting it in the submitting shell is unreliable (it depends on
# sbatch --export propagation), so pin it here.
# Costs no compute: the messages are formatted either way; this only adds a
# second handler writing the same text.
export VERBOSE=1
# PROGRESS defaults to "follow VERBOSE", which would start a tqdm bar. In a
# non-tty sbatch log tqdm emits a line per update and buries the real
# output, so pin it off; VERBOSE stays purely about the log lines.
export PROGRESS=0

# ---- run plumbing ----------------------------------------------------------
export MOJITO_DATA_PATH=/shared/data/mojito_cache
# The SAME directory, named the way the monitor page asks for it: the level
# holding catalogues/ and data/ (user ruling 2026-09-26). The page needed two
# knobs for this one path and the second was never exported here, so a page
# built on the cluster fell through to the laptop default and silently lost
# the residual-spectrum and data/template/residual panels. Required by
# GF_MONITOR_AFTER_SAVE=1; harmless otherwise -- the sampler never reads it.
export MOJITO_INFO_PATH=/shared/data/mojito_cache
# ---- MID-RUN MONITOR (user request 2026-09-27) -----------------------------
# The saver rank (walkers=[0,0), no sampling work) builds the page and the
# snapshot tar in the quiet gap after a save, so the artifact tracks the run
# instead of waiting for a manual pull. Outputs land BESIDE the run folder:
#   <STORE_DIR>/<run>_monitor.html   and   <run>_snapshot.tar.gz
#
# ⚠ THE HOOK CAN DISABLE ITSELF, BY DESIGN. It measures every build against
# the run's own save cadence and, if a build ever causes the saver to DROP a
# state -- which would lose a stored iteration permanently, since the saver
# coalesces to the newest payload when its queue backs up -- it prints a
# banner and stays off for the rest of the run. The run is never affected.
# If that fires, raise GF_MONITOR_ITER or set GF_MONITOR_SNAPSHOT=0.
# MATCH PANELS ON (user request 2026-09-30: "get back to including the phase
# maximized computations of match"). Restores the nine match-criterion panels
# and the Parameter Recovery section. Of those, the F4 three-way split and the
# zoomable-plot marker classes use the REAL phase-maximised, noise-weighted
# overlap (GF_MONITOR_MATCH_MM, default 0.8); the per-iteration
# completeness / purity curves still key off the 2-df-bin f0 proxy so they can
# run on every stored row. Overridable; =0 restores the 2026-08-19 default.
export GF_MONITOR_MATCH_STATS=${GF_MONITOR_MATCH_STATS:-1}
export GF_MONITOR_AFTER_SAVE=${GF_MONITOR_AFTER_SAVE:-1}
# EVERY 10th save (3 -> 10, user 2026-10-04: "every 10 or 20 ... whatever is
# sufficient"). At 1-in-3 the page (~575-600 s) + tar (~175 s) built on the
# SAVER held the next save ~400 s every third iteration (job 717, ~25-30 %
# of the wall). With the dedicated monitor rank (GF_MONITOR_RANK above) no
# save waits on the build at all; 10 saves (~1 h at replica_pe pace) keeps
# the page current without re-reading the 1+ GB live store back to back.
# YIELDING: GF_MONITOR_ITER=20 on the launch line overrides it.
export GF_MONITOR_ITER=${GF_MONITOR_ITER:-10}
# The tar as well as the page -- it is what actually gets downloaded.
# Set to 0 if the watchdog starts complaining; the page alone is much
# cheaper than the multi-GB archive.
export GF_MONITOR_SNAPSHOT=1
# Hard ceiling on one build. Well above the observed cost; exists so a
# pathological build cannot sit on the saver indefinitely.
export GF_MONITOR_TIMEOUT=1800
# Requires MOJITO_INFO_PATH, exported above.
# ---- FLAGSHIP: THE PARENT RUN ON THIS RUN'S MONITOR PAGE (user request 2026-10-12)
# "add a second run to the same html only in specific places where it is
# reasonable visually to compare between runs ... plot the phase maximized
# overlap plots, a comparison of the final psd and foreground measurements
# (final for the previous run and current for the ongoing run)". The page stays
# this run's; each run directory listed here (':'-joined) is read at ITS last
# stored row and drawn onto the overlap CDF / survival, recovered-vs-SNR and
# recovered-vs-frequency figures (matched against its own gb_truth_*.npz) and
# the noise + foreground posteriors and curves. A missing dir / store / truth
# set costs only its overlay (a caption note + a MISSING line). The monitor rank
# inherits it; the CLI twin is python -m lisatools.globalfit.monitor --add-info.
# The default is THIS RUN'S PARENT (runs.py); GF_MONITOR_ADD_INFO=none on the
# launch line drops it. Not a run knob: the sampler never reads it.
export GF_MONITOR_ADD_INFO=${GF_MONITOR_ADD_INFO:-/shared/data/global_fit_output/gf_prod_18mo_flagship}

export USE_GPU=1
export GPU_BACKEND=cuda13x
# GPU list follows what slurm actually granted (self-dispatch block above);
# NGPUS is the pre-submit intent, SLURM_GPUS_ON_NODE the in-job truth.
# NOTE GB_N_SUBBANDS below is PER GPU, so total sub-band residency scales
# with the count automatically; nvidia-smi telemetry samples every device.
_NGPUS_EFF=${SLURM_GPUS_ON_NODE:-${NGPUS:-2}}
# printf join, NOT `seq -s,` (BSD seq leaves a trailing separator, which
# the driver's GPUS parser would choke on -- caught in the local smoke).
GPUS=$(printf ",%d" $(seq 0 $((_NGPUS_EFF - 1)))); GPUS=${GPUS:1}
export GPUS
echo "[GPUS] ${_NGPUS_EFF} GPUs -> GPUS=${GPUS} (partition ${SLURM_JOB_PARTITION:-n/a})"

# ---- rank-layout knobs, re-derived in-job -----------------------------
# The self-dispatch block above already exports these via `--export=ALL,...`
# when the script is run directly; a manual `sbatch --gres=... <script>`
# skips that block entirely (SLURM_JOB_ID is already set at job start), so
# re-apply the identical defaults here rather than depending on an unset
# var under `set -u`.
GPUS_PER_RANK=${GPUS_PER_RANK:-}
RANKS_PER_GPU=${RANKS_PER_GPU:-1}
# Re-exported here for symmetry with GF_LEGACY_RANK_LAYOUT below: all three
# knobs are read from the environment by build_layout, and a manual
# `sbatch <script>` never ran the --export=ALL,... dispatch line.
export GPUS_PER_RANK RANKS_PER_GPU
_k=${GPUS_PER_RANK:-1}
if [ "${NGPUS:-4}" = "4" ]; then
  GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-0}
else
  # walker-block layout by default at NGPUS=2 too (user ruling 2026-09-16;
  # GF_LEGACY_RANK_LAYOUT=1 = the rollback knob for a 1-node job)
  GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-0}
fi
# The legacy layout is per-NODE (one compute rank owns that node's whole GPU
# pool; every other non-saver rank is a stopped spare), so it cannot span an
# allocation: on 2 nodes every rank of node B would idle. The self-dispatch
# block forces GF_LEGACY_RANK_LAYOUT=0 at NGPUS=4, but a manual
# `sbatch --nodes=2 --ntasks=5 <script>` skips that block entirely -- so
# force it here too, off the GRANTED allocation, and say so loudly.
if [ "${SLURM_NNODES:-1}" -gt 1 ] && [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
  echo "[SUBMIT] SLURM_NNODES=${SLURM_NNODES} > 1: FORCING GF_LEGACY_RANK_LAYOUT=0"
  echo "[SUBMIT]   (the legacy single-compute-rank layout is per-node and cannot"
  echo "[SUBMIT]    span an allocation -- every rank on the other node(s) would"
  echo "[SUBMIT]    build, idle and waste the node)."
  GF_LEGACY_RANK_LAYOUT=0
fi
export GF_LEGACY_RANK_LAYOUT
if [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
  N_COMPUTE_EFF=1
else
  N_COMPUTE_EFF=$(( ${SLURM_NNODES:-1} * _NGPUS_EFF * RANKS_PER_GPU / _k ))
  if [ "${N_COMPUTE_EFF}" -lt 1 ]; then
    echo "[SUBMIT] N_COMPUTE=${N_COMPUTE_EFF} from NNODES=${SLURM_NNODES:-1} x"
    echo "[SUBMIT]   GPUS=${_NGPUS_EFF} x RANKS_PER_GPU=${RANKS_PER_GPU} /"
    echo "[SUBMIT]   GPUS_PER_RANK=${_k}: inconsistent knobs (GPUS_PER_RANK larger"
    echo "[SUBMIT]   than the granted pool?). Leaving NWALKERS alone; build_layout"
    echo "[SUBMIT]   will raise with the exact reason."
  fi
fi
# DEDICATED MONITOR RANK, checked against the GRANTED task count. The driver
# takes the highest world rank out of the run whenever GF_MONITOR_RANK=1, so
# an allocation WITHOUT the extra task (a manual `sbatch --ntasks=5`, an
# interactive allocation) would silently lose a compute rank to it. Only
# arm it when the allocation has exactly one task beyond the run's.
export GF_MONITOR_RANK=${GF_MONITOR_RANK:-1}
if [ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then
  _RUN_TASKS=3
else
  _RUN_TASKS=$(( N_COMPUTE_EFF + 1 ))
fi
if [ "${GF_MONITOR_RANK}" = "1" ] && [ -n "${SLURM_NTASKS:-}" ] \
    && [ "${SLURM_NTASKS}" -ne $(( _RUN_TASKS + 1 )) ]; then
  echo "[SUBMIT] GF_MONITOR_RANK=1 but SLURM_NTASKS=${SLURM_NTASKS} != ${_RUN_TASKS} run"
  echo "[SUBMIT]   ranks + 1 monitor: DISARMING it (GF_MONITOR_RANK=0) so no run"
  echo "[SUBMIT]   rank is taken; the page/tar are built on the saver as before."
  export GF_MONITOR_RANK=0
fi

# ---- output ----------------------------------------------------------------
export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_24mo

# ---- v8 noise model (the whole v8-vs-v7 diff) -------------------------------
export UNEQUAL_ARM=1
export UNEQUAL_ARM_STRIDE=200
# Since LAT 290c1902 (2026-10-06) the walker-batched noise scorer (PSD_BATCH,
# code default 1) folds galfor with THIS method too; before, it always used
# the exact fold while every other move scored the quadrature (synthetic
# check on the 6-month grid: 4e-4 lnL at the galfor estimate, up to 0.3 lnL
# across wide prior draws).
export WDM_PSD_METHOD=layer_calibrated
export GALFOR_MODULATION_PATH="$PWD/src/lisatools/galfor_modulation/modulation_unequal.dat"
export GALFOR_MODULATION_T0=data
# PIN THE RESUMABLE NOISE IDENTITY OF THIS STORE (2026-09-23). 36e2319e made
# the NOISE-brick scalar fit use the brick's own link delays, which moves the
# fitted (Soms_d, Sa_a) by 0.26% / 0.59%:
#     equal arms (this store)  1.496182116469e-11  2.982411739286e-15
#     unequal arms (new code)  1.500004011496e-11  3.000107254658e-15
# Those numbers ARE general.psd_injection, and the coarse delayed-acceptance
# fiducial digest is a SHA-256 over their raw float64 bytes, so the new fit
# changes the digest and the resume guard refuses the store:
#     "stored noise-model identity differs from the configured one:
#      {'coarse_fiducial_digest': ('856ddb07f6b46e39', 'b5047f761a4400d5')}"
# =0 restores the equal-arm fit BIT-IDENTICALLY (verified against 36e2319e^
# on the mojito-light brick), so gf_prod_6mo_v8_4gpu resumes unchanged.
# DELETE THIS LINE for a fresh store -- the arm model is the better answer,
# it just is not the answer this chain was started with.
# NOT the noise model: UNEQUAL_ARM=1 above still selects
# UnequalArmInstrumentNoise for the likelihood. This is only the REFERENCE
# fit that produces general.psd_injection.
# ---- FLAGSHIP: the reference fit is the UNEQUAL-ARM fit (2026-10-08) ---------
# The =0 pin above exists ONLY so the resumed 6mo store keeps the equal-arm
# pair its chain was started with (coarse_fiducial_digest). A fresh store has
# no such debt: =1 (the code default when unset; the 3mo v9 dropped the line,
# ruling 2026-09-26) makes general.psd_injection -- the reference / 'truth'
# pair the pages and diagnostics compare against, NOT the noise model, which
# is UNEQUAL_ARM=1 either way -- the unequal-arm fit 1.500004e-11 /
# 3.000107e-15. Ruling 2026-10-04: no default may rely on an equal-arm path.
export MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=${MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM:-1}
echo "[V8-NOISE] UNEQUAL_ARM=${UNEQUAL_ARM} stride=${UNEQUAL_ARM_STRIDE} wdm_psd_method=${WDM_PSD_METHOD} psd_reference_fit_unequal_arm=${MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM}"
echo "[V8-NOISE] modulation=${GALFOR_MODULATION_PATH} t0=${GALFOR_MODULATION_T0}"

# ---- coarse noise likelihood (pinned, not inherited) ------------------------
# all_sources DEFAULTS these on now, but a submit script states its own
# configuration: a reader must be able to tell which noise likelihood the run
# used without cross-referencing the variant.
#
# delayed_acceptance is EXACT -- stage 1 screens PSD/galfor proposals on the
# Q-fold time-coarsened surrogate, stage 2 corrects with the exact fine/coarse
# ratio -- so the SAMPLED target is the fine likelihood in every stage,
# whatever the surrogate's quality. (search_approx is faster but approximate;
# in probe job 376 the search it drove railed galfor against two prior edges.
# The noise block is a small share of wall clock, so that trade is not worth
# taking here -- accuracy ruling 2026-08-28.)
#
# Q=8 measured on the 3-mo grid: Nt_active 2121 -> Ncoarse 266. WS weighting
# (the Welch-Satterthwaite effective dof, frozen at the injection fiducial)
# was exercised on GPU in job 376 -- the unequal-arm coarse basis path works
# there. To fall back: COARSE_GPU_MODE=off restores the exact-fine likelihood
# (job 369's configuration) and COARSE_USE_WS=0 swaps WS for Bartlett.
# ---------------------------------------------------------------------------
# V9-2 (2026-09-24, user ruling): NOISE MODEL BACK FROM "COARSE" TO NORMAL.
# v8 ran mode=delayed_acceptance with the Q=8 time-coarsened surrogate as
# stage 1. v9 runs the plain EXACT-FINE likelihood -- the configuration the
# comment above names as the fallback, and the one probe job 369 ran.
#
# This is NOT a bias change in either direction: delayed acceptance is exact
# (stage 2 corrects with the fine/coarse ratio), so v8 and v9 sample the SAME
# target. What changes is cost and the number of moving parts -- no surrogate
# basis, no WS dof, no fiducial digest, no stage-2 acceptance to audit, and
# the [COARSE_AUDIT] lines simply stop appearing. Expect the noise block to
# get SLOWER; it is a small share of the iteration (28.5 s of 852 s measured
# at v8 3mo), which is what makes the simplification affordable.
#
# ⚠ THE COARSE MODE IS PART OF ``noise_model_identity``. A v8 store CANNOT be
# resumed under this line -- the [V8-NOISE] preflight below and run.py both
# refuse the mismatch by design. STORE_DIR is a fresh v9 path for this reason.
#
# ⚠ COARSE_Q MUST BE 1 HERE, AND "8 + mode=off" WAS NOT A NO-OP.
# JOB 618 (2026-09-25) aborted all 5 ranks at build:
#
#   ValueError: coarse_Q > 1 in an all-source run requires an explicit
#   COARSE_GPU_MODE ('search_approx' for optimization stages,
#   'delayed_acceptance' for production PE): the coarse statistic changes
#   the PSD/GALFOR transition kernel, so it is never an implicit speed knob
#   here.   [stock/erebor/noise.py::validate_coarse_settings]
#
# This comment used to claim COARSE_Q could stay at 8 "so the stamped
# identity and the preflight's wanted values are read from the same env",
# because "with mode=off they select nothing". THAT WAS WRONG, and the
# validator exists to catch precisely it: run.py:1607 reads coarse_Q on its
# own -- `Q = int(getattr(general_info, "coarse_Q", 1) or 1)` -- and builds
# the coarse basis for ANY Q > 1 without consulting the mode at all. So the
# pair (Q=8, mode=off) is not "exact fine with a documentary 8"; it is two
# subsystems disagreeing, which is why it is refused rather than resolved.
#
# Q=1 IS the exact-fine configuration V9-2 asked for -- there is nothing to
# coarsen -- and it is what "the general psd computations for galfor and
# psd, not the coarse version" (user, 2026-09-25) means in this file.
# ⚠ all_sources.py defaults COARSE_Q to 8, so this export must be explicit.
# To go back to the v8 configuration: COARSE_Q=8 AND
# COARSE_GPU_MODE=delayed_acceptance together (and a store created under
# that pair -- the mode is part of noise_model_identity).
export COARSE_Q=1
export COARSE_GPU_MODE=off
export COARSE_USE_WS=1
export COARSE_FIDUCIAL=injection
# Galfor slope-index cap (2026-09-04 diagnostic, KEPT on the 3mo science
# arm; ported): alpha railed at the stock 5.0 cap (~60% of samples at the
# edge). Widened [1e-3, 20] so the slope can explore; revert = drop the
# line. The prior is rebuilt from code each run.
# GALFOR SAMPLING BASIS (user ruling 2026-09-18). amp, fk, f_1, f_2 move to
# log10 over the SAME physical support (galfor_prior_dict + the 10**x
# transform container); alpha is O(1) and stays linear. The foreground model
# still receives linear parameters.
#
# WHY. galfor's linear prior box spans ~42 decades -- amp (1e-47, 1e-41)
# against alpha (1e-3, 20) -- and the eigen proposal's step is scaled to that
# box: the softest whitened eigenvalue puts a ~1-prior-width step on amp,
# i.e. ~190x the parameter's own value, which takes amp negative and the
# prior rejects every draw. That is the measured "galfor does not move".
#
# The two cheaper fixes were tried and REFUTED, in this order:
#   * per-parameter whitening of the info matrix -- ALREADY implemented in
#     eigen_refresh._tables_from_info_batch (diag(w) I diag(w)); rescaling
#     cannot make a likelihood quadratic in a parameter it is wildly
#     non-linear in;
#   * building the eigen tables on the COLD ROW only (512105bd) -- shipped,
#     confirmed live in the 6mo run by its "1 sources x 5 dims" infomat
#     signature, and galfor STILL came back non-positive on 23 of 28 builds,
#     ~29% of them catastrophic (worst lambda/lambda_max -9.7e78).
#
# NEEDS A FRESH STORE: this changes what the stored numbers MEAN, not their
# shape. A resume across the flip is refused by noise_model_identity as of
# 0044b7cc (galfor_log_sampling is stamped and compared) -- pull at least
# that commit before running this, or the refusal is not there to catch you.
export GALFOR_LOG_SAMPLING=${GALFOR_LOG_SAMPLING:-1}
export GALFOR_ALPHA_MAX=20.0
# ---- FLAGSHIP: GALFOR FREQUENCY PRIOR (user ruling 2026-10-08) ------------
# "adjust the frequency parameters (fk, f1, f2) in the foreground model ...
# in its prior ... to go from 1e-4 to 1e-2": the three frequency columns
# share one box, 0.1-10 mHz (stock: fk 0.8-10 mHz, f_1 / f_2 10 uHz-10 mHz).
# One resolver (galfor_prior_ranges) feeds the prior, the start-pin window
# and the noise-pin refusal, so the GALFOR_START_PARAMS pin (fk 2.53 mHz,
# f_1 10 mHz, f_2 1.41 mHz) is inside the box. Fresh store only: a chain
# outside the box would price at log_prior = -inf on resume.
# 2026-10-09 (Mike): "adjust the foreground prior for f_k and f_2 to go from
# 0.8 mHz to 10 mHz. Keep f_1 as is." -- per-column boxes, in Hz; f_1 keeps the
# 2026-10-08 box. The start pin (fk 2.53 mHz, f_1 10 mHz, f_2 1.41 mHz) is inside.
export GALFOR_FK_PRIOR=${GALFOR_FK_PRIOR:-0.8e-3,1e-2}
export GALFOR_F1_PRIOR=${GALFOR_F1_PRIOR:-1e-4,1e-2}
export GALFOR_F2_PRIOR=${GALFOR_F2_PRIOR:-0.8e-3,1e-2}
echo "[V8-NOISE] coarse: Q=${COARSE_Q} mode=${COARSE_GPU_MODE} \
use_ws=${COARSE_USE_WS} fiducial=${COARSE_FIDUCIAL}"

# ---- sampler shape ---------------------------------------------------------
# 10 -> 4 (user ruling 2026-09-18, for the fresh walker-block store). One
# walker per compute rank at NGPUS=4. Chosen for SEARCH SPEED: at fixed GPU
# count the per-rank walker block is what grows, and ~364 s of a ~390 s
# gb_search iteration scales with rows (the three GB RJ moves ~260 s,
# mbh/emri amortized ~62, vgb ~30, noise ~11). Only SOBBH does not -- its
# scorer bills a flat 1.73 s per CALL whatever the batch shape. Doubling to
# 8 walkers therefore roughly doubles the iteration while buying nothing in
# search throughput: births come from the F-stat grid and the warm-start
# components, both independent of walker count, so births per HOUR is a wash
# and iterations per hour is what drives the cap ramp and the tempering
# cadence.
# NOTE the walker count LOCKS for the life of the store (the resume refuses
# any change -- state.py's walker-count mismatch), and it carries into
# full_pe, where 4 is a thin posterior ensemble. Deliberate: this store's job
# is the search.
export NWALKERS=${NWALKERS:-4}
                                   # GB rungs stay GB_NTEMPS=24 -- walkers and
                                   # temps are independent axes. NEVER change
                                   # NWALKERS on a resume: the store carries the
                                   # walker axis. Under the walker-block layout
                                   # NWALKERS must divide by N_COMPUTE, and
                                   # N_COMPUTE = NNODES * GPUS_PER_NODE *
                                   # RANKS_PER_GPU / GPUS_PER_RANK. At the
                                   # intended NGPUS=4 shape that is
                                   # 2 * 2 * 1 / 1 = 4, so NWALKERS=4 divides
                                   # cleanly at 1 walker/rank and NOTHING is
                                   # rounded. (The older note here claimed
                                   # "4 GPUs need 5 compute ranks with
                                   # RANKS_PER_GPU=2" -- that is wrong and was
                                   # left over from the NWALKERS=10 plan:
                                   # RANKS_PER_GPU=2 gives N_COMPUTE=8, not 5,
                                   # and 5 is not reachable from 4 GPUs at all.)
                                   # ⚠ A NON-MULTIPLE IS ROUNDED UP, not
                                   # refused, and the store then LOCKS to the
                                   # rounded value: at NGPUS=4, 2 and 3 both
                                   # become 4, 5 and 6 become 8, 10 becomes 12.
                                   # So test at the value you intend to launch
                                   # -- a resume refuses any change.
                                   # Noise-block floor 2*ndim (galfor ndim 5
                                   # -> 10) still satisfied.
                                   # The dispatch's --export=ALL carries the
                                   # shell's value, so `NWALKERS=1 ./submit_gf_24mo.sh`
                                   # reaches the one-walker branch below.
# GF_GPU_ROUTING=1 (functions at the top of the script): the replica
# factorization of the GRANTED allocation -- NWALKERS is never rounded (the
# rounding branch below is skipped), and a shape build_layout would refuse is
# refused here, before the preflights and mpiexec.
_GF_ROUTING=0
if _gf_routing_on; then
  _GF_ROUTING=1
  export GF_GPU_ROUTING=1
  if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ]; then
    _gf_routing_layout "${NWALKERS}" "${N_COMPUTE_EFF}" "${_k}" "${RANKS_PER_BLOCK:-}" || exit 2
  fi
fi
if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ "${N_COMPUTE_EFF}" -gt 1 ] && [ "${NWALKERS}" -eq 1 ]; then
  echo "[SUBMIT] NWALKERS=1 on N_COMPUTE=${N_COMPUTE_EFF}: one-walker replica mode (every compute rank holds the walker; GB/VGB by band range, addremove/PSD by likelihood rows)"
elif [ "${_GF_ROUTING:-0}" = "0" ] && [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] \
     && [ "${N_COMPUTE_EFF}" -gt 0 ] && [ $(( NWALKERS % N_COMPUTE_EFF )) -ne 0 ]; then
  echo "[SUBMIT] NWALKERS=${NWALKERS} is not a multiple of N_COMPUTE=${N_COMPUTE_EFF}; using NWALKERS=$(( (NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF )) (user decision at the first 4-GPU launch)"
  export NWALKERS=$(( (NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF ))
fi
export NUM_ITERATIONS=${NUM_ITERATIONS:-2000}         # total engine iterations (resume-safe; NITER was a dead name)

# ---- band + domain ---------------------------------------------------------
# EXPLICIT Tobs (2026-08-13): sbatch propagates the submitting shell's env,
# and a stale TOBS_TARGET export (a 3-day one was found live in the shell)
# would silently re-grid this run. Pin this run's value (runs.py: DAYS x
# 86400 s; one WDM layer = Nf 1440 x dt 2.5 s = 1 h, so Nt = Tobs / 3600 --
# exact at every flagship Tobs, checked against WDMSettings.adjust_to_even_bins).
export TOBS_TARGET=62208000        # 720 d; grid resolves Nf 1440 x Nt 17280 x dt 2.5
# Sig-het sparse layer stride = Tobs / 36 h (the validated 36-h stride
# parity: 60 at 3 mo, 120 at 6 mo, 180 at 9 mo, 240 at 1 yr, 360 at 18 mo,
# 480 at 2 yr); it divides Nt by construction (Nt / stride = 36).
export SIGHET_NT_LAYER=480
# ⚠ MIN_FREQ 4e-4 -> 2.5e-4 (2026-09-06). This is the LAYER-1 FIX, and the
# value is set by the VGBs, not by taste.
#
# WHY IT MOVED AT ALL. all_sources pinned min_freq as a PLAIN default 1e-4 that
# shadowed the env-backed field, so THIS EXPORT NEVER TOOK EFFECT (fixed in
# 7ad3b0bd). The run therefore analysed from WDM layer 1 -- support
# 0.069-0.208 mHz, sharing an edge with DC, where the instrument model diverges
# and instrument_fill_nans=0.0 zeroes the covariance. Measured on this store
# (kappa probe, job 459): det(C)=0 there, q = w^T C^-1 w / 3 = 150.3 against an
# expected 1.0, i.e. 43% of the fit's ENTIRE chi^2 budget. That alone produced
# the instrument-PSD bias: the exact ML alpha = sqrt(mean(q/3)) = 1.389 vs the
# 1.3874 observed (0.21%).
#
# WHY 2.5e-4 AND NOT 4e-4. Only layer 1 is pathological; layer 2 measured
# q/3 = 1.0203, as clean as anything in the band. Both choices cure the bias
# identically (alpha 1.0490 at layer 2 vs 1.0493 at layer 3) -- but the VGB
# catalogue's four lowest sources sit at 0.3117/0.3364/0.3365/0.3392 mHz, and
# layer 3's support starts at 0.34722 mHz, so MIN_FREQ=4e-4 would DROP FOUR
# VGBs. 2.5e-4 lands on layer 2 (support from 0.20833 mHz), keeps all 55 VGBs,
# and retains 93.8% of the S_tm information against 87.9% at layer 3.
#
# Valid range for layer 2: 1.3889e-4 < MIN_FREQ <= 2.7778e-4. 2.5e-4 sits
# comfortably inside (2.7778e-4 is the exact boundary and can tip to layer 3 on
# floating-point rounding -- do not use it).
#
# RESUME-SAFE: band_edges/cap_edges are stored in absolute Hz and start at
# layer 4 (5.5556e-4), entirely above the new floor, so they recompute
# bit-identically and the leaf-cap guard passes.
export MIN_FREQ=2.5e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

# ---- GB knobs (everything else rides the flipped defaults: sig-het in-model,
#      fstat-fit-in-move + sig-het fstat, D/2 leaf-cap gate w/ min-iters 5,
#      at-cap RJ skip, cell-lifecycle ll credit, GB_MODE=search +
#      GB_PE_MOVES_STRICT=1 + GB_SEARCH_PRIOR_REMOVAL=1 seeded by the script) --
export GB_NLEAVES_MAX=25000        # per run (runs.py): 3mo 10000 / 6mo 15000 / 9mo + 12mo 20000 (the 1yr value, ruling 2026-10-07) / 18mo + 24mo 25000 (the 23mo ruling)
# FULL parity-unit residency (grouped RJ->in-model scheduling, 2026-08-13):
# one unit = 77 bands x 24 temps x 24 walkers = 44,352 cells; the scheduler
# clamps n_slots to min(GB_N_SUBBANDS, cells), so 50000 means every cell is
# resident (zero mid-unit refills) and the grouped in-model flush runs at
# full grid width. Buffer cost ~255 KB/slot at slab-5 => ~11.3 GB (user
# budget: 10-20 GB is fine). Back off to 2560 (652 MB) if the pool OOMs.
# 16384 (user Q 2026-08-14): slots now amortize ROUNDS, not just memory --
# rounds/unit ~ waves x per-wave depth, so 4x residency ~ 3-4x fewer
# 2.4s host round-trips. ~4.2 GB buffer; post-fix profile at 4096 was
# flat 42-45/31 GB on 96 GB cards. If the unit-open lines stay flat,
# full residency (50000 -> 44,352 slots, ~11.3 GB) is the next step.
# ---- OOM 2026-09-16 (dev0, 3.68 GB En fold at 88.9 GB allocated, run
# died overnight in setup_in_model/bin_fold_real): 16384 violated the
# sig-het stash law documented at the SIGHET_NT_LAYER block below -- the
# stash goes as CELLS x N_sparse_t and the knobs MULTIPLY. The mirror
# halved SLAB bytes (the rationale for 16384) but not the fold stash:
# 6mo N_sparse_t 118 x 16384 = 1.93e6, right at the 2.2e6 product that
# OOM'd v4@270 (and 4x the calibrated 6-mo-safe 118 x 4096 = 4.8e5).
# Residency fills toward capacity as rj/tempering matures, so early
# telemetry (35-45 GB) looks fine and the wall arrives after the first
# long uninterrupted stretch. 8192 -> product 0.97e6, under the 23-mo
# 1.1e6 OK precedent; back off to 4096 (the calibrated-safe value) if
# dev0 max memory trends past ~70 GB in gpu_util_*.csv again.
export GB_N_SUBBANDS=2048   # PER GPU; total = x n_gpus. Per run (runs.py): slot bytes scale with Tobs, so 32768 at 3 mo / 8192 at 6 mo / 4096 at 9 mo + 1 yr (the 1yr value, ruling 2026-10-07) / 2048 at 18 mo + 2 yr (the 23mo precedent)
                            # @6mo (mirror) ~ 4.2 GB/GPU; the binding
                            # constraint is the SIG-HET STASH product above,
                            # not slab bytes. (History: pre-mirror 4096 at
                            # ~2 MB/slot; 16384 mirror-era OOM'd 09-16.)
# PSD SHARED MIRROR (386f25ce, validated jobs 469/470: 3mo dev0 68->44 GB,
# 1yr 85->61 GB with capacity DOUBLED; parity 11k+ IDENTICAL). Parity gate
# disarmed: vgb_pe rebuilds its buffer every unit so the gate never retires
# (+25 s/unit at 1yr); the gate already banked its evidence on 469/470.
export GB_PSD_SHARED_MIRROR=1
export GB_PSD_MIRROR_PARITY_PROPOSES=0

### SOME OTHER GB options
# V9-11: 5.0 -> 8.0. The opt-SNR boundary is now a PER-STAGE PROFILE value
# (V9_SEARCH_STAGE_PROFILES in run_combined_staged.py): gb_search_1 sets 8.0,
# gb_search_2 and gb_search_3 set 5.0, applied to every GB band move at stage
# entry and logged as [V9-STAGE ...] opt_snr_rej_samp_limit lines.
# This export is the value the moves are CONSTRUCTED with, i.e. what is in
# force before the first stage profile applies, so it is set to stage 1's --
# if the profile ever failed to apply, the run would hold stage 1's floor
# rather than silently running stage 1 at stage 2's.
export GB_OPT_SNR_LIMIT_SEARCH=8.0
# 10 -> 5 (user ruling 2026-09-27). Consumed by
# run_combined_staged.py::source_pe (NOT by src/ -- a grep of the
# package alone says this knob is dead, and it is not).
# Measured on 6mo job 650: sobbh_pe 144.0 s + mbh_pe 95.7 s +
# emri_pe 71.1 s = 310.8 s on the iterations they run, ~4.9% of a
# 6377 s iteration. At every-10 that averaged 31 s/iteration; at
# every-5 it averages 62 s. full_pe is untouched -- all three run
# every iteration there.
export GB_SEARCH_SOURCE_EVERY=5

# RJ pick thinning. UNSET as of 2026-08-28 -- the value now lives in code
# (_SEARCH_RJ_FLIP_DEFAULT / _PE_RJ_FLIP_DEFAULT in recipe.py, both 0.2),
# so behavior is UNCHANGED from the 0.2 this line used to export.
#
# Removed rather than kept because {BRANCH}_RJ_FLIP_FRACTION is a GLOBAL
# override: one exported value lands on every RJ move in every stage, so
# it silently collapses any future search/PE split. The five search-named
# RJ moves are now passed the default explicitly at their construction
# sites (LAT bba4219d) -- before that they only reached 0.2 BECAUSE of
# this export, and would have fallen through to a hard-coded 1.0 without
# it. The old comment here also said "0.3 random subset" while exporting
# 0.2; that contradiction is gone with the line.
# In-model repeats are unaffected either way -- they cover ALL alive
# sources; the flip gate is rj-only by construction.
# export GB_RJ_FLIP_FRACTION=0.2   # <- re-export ONLY to force ALL stages
# PER-STAGE flip knobs (2026-09-11 user ruling, values corrected 09-14 --
# ported from the 3mo 10-walker arm): rj_fstat_search never received the
# flip default (recipe wiring miss, fixed 09-14), so the search ran at 1.0
# in jobs 474/479; the intended "half the birth attempts" against what
# actually ran is 0.5 (0.1 would be a 10x cut of the search feed). PE
# moves DID carry 0.2 before, so 0.1 there is the intended halving.
# V9-5 (user ruling 2026-09-24): 1.0 for the SEARCH parts -- every RJ
# candidate row gets a proposal instead of half of them. This reverses the
# 09-11 halving; the search feed is what the three-stage structure is built
# to consume, and stage 1 in particular is meant to be greedy.
export GB_SEARCH_RJ_FLIP_FRACTION=1.0
export GB_PE_RJ_FLIP_FRACTION=0.1
# In-model info-matrix jump scale: 0.005 default measured 95% cold
# acceptance; 0.2 -> 0.61; 0.4 -> 0.60 (job 196). Job 197 flipped the
# story: with the EXACT per-block SIGHET info matrices live, cold
# acceptance at 0.6 ROSE to 0.71-0.80 (all-T 0.83-0.92, n=27k) -- the
# better-adapted covariance makes the old scale far too timid. 1.2 is
# the next notch (recommended range 1.0-1.5); tune against the
# [GB_ACCEPT] per-proposal-type line toward the 0.15-0.4 target.
export GB_JUMP_FACTOR=1.2
# ---- TIMERS: ALL ARMED (user directive 2026-08-15, fresh run: "make sure
#      all the timers are armed so we can study the differences in detail")
# Always-on, FREE (no knob, no sync): the [GB_TIMING <move>] per-propose
# span breakdown, the [FSTAT_CTR] center census, [GB_ACCEPT] +
# [GB_ACCEPT rj-split] per-proposal-type rates, [GB_CELL_LL] credit
# checks, [SAVE] write times, the buffer-lifecycle lines, and the
# nvidia-smi CSV sampler above.
#
# GF_MOVE_TIMING: per-move wall_s + host RSS + GPU-pool MB for EVERY move
# in the stage. This is the one that finally makes psd_pe / galfor_pe
# visible -- they emit no [GB_TIMING] of their own, which is why the
# ~43 s/iteration noise block was untimed "dark matter" until now. Cheap:
# one timestamp + two memory reads per move.
export GF_MOVE_TIMING=1
# GF_MOVE_TIMING_SYNC / GB_PROP_TIMING_SYNC: make every mark carry EXACTLY
# its own kernel time instead of leaking into the next mark (cupy is async,
# so unsynced spans attribute launch time, not execution time). This is
# what makes a detailed before/after comparison trustworthy.
# COST: ~8-10 extra device syncs per get_ll; measured +2.5% wall on the
# GB propose, and it serializes some concurrent shard work.
# >>> TURNED OFF 2026-08-28 (user ruling). The tempering audit found the
# >>> sync cost is much larger than the +2.5% headline in the TEMPERING
# >>> path specifically: `temper_swap_score` opens and closes INSIDE the
# >>> rung loop, so with sync on it adds roughly 27,000 full device syncs
# >>> per move -- a real part of the ~88 s that `run_tempering` could not
# >>> account for. Leaving it on means the instrumentation is measuring
# >>> itself. Detailed per-stage attribution is already banked from
# >>> snapshots 11-13; steady-state numbers now matter more.
# >>> Set BOTH back to 1 for 2-3 iterations if a fresh detailed
# >>> attribution is ever needed again -- the run resumes cleanly.
export GF_MOVE_TIMING_SYNC=0
# GB_PROP_TIMING_SYNC uses the := form to MATCH submit_gf_3mo_v7.sh:467,
# whose comment states the reason: a plain `export VAR=0` CLOBBERS an
# sbatch-time override, so `GB_PROP_TIMING_SYNC=all sbatch ...` would
# silently do nothing -- exactly the re-arm the comment above tells you to
# perform. Resolved value is still 0, so run behaviour is unchanged; this
# only removes a v7/v8 asymmetry and restores the documented escape hatch.
: "${GB_PROP_TIMING_SYNC:=0}"
export GB_PROP_TIMING_SYNC
#
# ---- rj_fstat_centers INTERIOR PROBE (2026-08-29) --------------------
# WHAT IS ALREADY KNOWN (do not re-derive, and do not re-open it as a
# "missing time" question -- that was an arithmetic error):
#   v7 snapshot: total=1955.1s run_proposal=1810.6s
#   rj_fstat_centers=1334.874s (68% of the propose).
#   The "[FSTAT_CTR ...] unit precompute: ... in 149.75s" line is emitted
#   ONCE PER BAND UNIT and the move runs NINE units per propose. The nine
#   lines sum to ~1339 s vs the reported 1334.874 s -- the bucket closes
#   to 0.2%. Cross-checked twice: the other _run_rj_step marks bound the
#   per-round centre chain at <= 3.077 s, and the nine unit row counts sum
#   to exactly picked_sources = 4,546,846.
#   => ~99.8% of the stage is _precompute_fstat_centers -> _fstat_ctr_compute,
#      4.55 M rows/propose at ~0.293 ms/row, ~97% of them dead birth slots,
#      "0 at-cap excluded".
#
# WHAT IS NOT KNOWN, AND WHAT THIS PROBE IS FOR: what that 0.293 ms/row is
# made of. Basis-filter/scorer kernel? Jaranowski-Krol inversion? Host
# staging of 4.55 M rows? The answer decides the attack -- a cheaper
# kernel vs fewer rows. rj_fstat_centers now decomposes (all NESTED, so
# `tracked` is unchanged):
#   unit-open precompute : fstat_ctr_select / fstat_ctr_solve /
#                          fstat_ctr_census / fstat_ctr_audit /
#                          fstat_ctr_pack
#   the scorer itself    : fstat_nm_transform / fstat_nm_lanes |
#                          fstat_nm_routed / fstat_nm_invert /
#                          fstat_ctr_map / fstat_nm_h2d /
#                          fstat_nm_lane_score / fstat_nm_lane_build /
#                          fstat_ctr_miss_fallback
#   per pick round       : rj_ctr_keep_gate / rj_ctr_birth_lookup /
#                          rj_ctr_birth_draw / rj_ctr_death_lookup /
#                          rj_ctr_death_dens
# plus counters fstat_ctr_units / fstat_ctr_rows / fstat_nm_rows /
# rj_ctr_route_{table,cache,direct} / fstat_ctr_audit_rows.
#
# ***READ THE SPLIT FROM A SYNC-ON PROPOSE, NOT A PRODUCTION ONE.*** The
# TOTAL is solid either way (it is confirmed above), but the SHARES inside
# it are not: with GB_PROP_TIMING_SYNC=0 the spans do not synchronize, so
# a sub-stage that contains a sync point is billed for kernels launched by
# earlier ones -- the fill_indmap_data precedent (598 s measured, 45 s
# real). Splitting a 1,331 s interior is exactly where that bites.
#
#     GB_PROP_TIMING_SYNC=all   # 'all' drains EVERY run device;
#                               # '1' drains only the current one, which
#                               # is not enough with GB_FSTAT_NM_MULTIDEV
#                               # fanning the scorer across both GPUs.
#
# Relaunch this script with that exported (and NUM_ITERATIONS small, or
# just kill it after a couple of [GB_TIMING] lines) -- the run resumes
# cleanly from the checkpoint, exactly as the 2026-08-28 note above says.
# Then read fstat_nm_transform / fstat_nm_lanes (with its fstat_nm_h2d +
# fstat_nm_lane_score) / fstat_nm_invert / fstat_ctr_map as shares of
# fstat_ctr_solve. Leave it at 0 for production.
# ---- 2026-08-15 perf batch (ALL code defaults; pinned for the run
#      record; each knob independently revertible) ---------------------
# De-synced in-model repeat loop (device-resident accept chain) rides
# the pull with no knob (bit-identical decisions, tested).
export GB_ROUTER_DEVICE_RESIDENT=1 # params/outputs never host-stage in the
                                   # shard router; =0 restores host staging
export GB_RJ_SNR_TRUNC_DIST=1      # birth distance draw truncated at the
                                   # analytic SNR-5 boundary; truncated
                                   # density in the factors (DB-exact);
                                   # =0 restores the plain lognormal
# Per-class in-model repeats (user ruling 2026-08-27, final:
# 250/25 after a same-day 200/100 hold): concentrate polish AT BIRTH
# (a newborn lands at grid resolution and must climb its peak before
# removal judges it) and buy iteration RATE with the survivor budget
# (highf endgame: survivor polish saturates — stuck walkers took
# 100/round for ~150 rounds without moving; transport fixes those —
# while survivor cost rides all three RJ moves and scales with alive
# count). Faster rows also tick the iteration-clocked cap patience
# faster and give more permuted + vertical swap rounds per hour.
# NOTE these env pins beat the PE mode default as well — both phases
# run 250/25.
# 100 (2026-09-11 user ruling, was 250 -- ported from the 3mo 10-walker
# arm): with the vertical ladder working, hot-rung newborns descend over
# the following iterations and are polished as survivors at every rung
# they visit; the newborn class had become 83% of the 1yr repeat-rows
# (3,347 rows @250) and ~50% at 3mo.
export GB_INMODEL_REPEATS_NEWBORN=100
# SURVIVOR 25 -> 100 (user ruling 2026-08-29, aligned with v7), restoring the
# value the high-f probe ran (200/100). In-model f0 drift is the ONLY mechanism
# that moves a source across a sub-band edge -- there is no merge operator, RJ
# is serial-within-band, and tempering swaps never cross bands -- and the
# measured crossing rate shows how underpowered it is at 25: median NN-matched
# displacement 0.046 bins/iteration, only 1.64% of matched pairs cross a band
# edge, 0.59% clear a 12-bin gap in one step. Trades iteration RATE for the
# ability to close edge-split pairs.
# ⚠ The per-class split applies on the DIRECT-batch path only; the grouped
# scheduler takes ONE budget for the whole pool from _SURVIVOR, so with
# GB_RJ_GROUPED_INMODEL=1 this raises the effective budget for newborns too.
# 50 (2026-09-11 user ruling, was 100 -- ported from the 3mo 10-walker
# arm): vertical sweeps run once per repeat, so this also halves the
# vertical mixing per block -- judge by progress per wall-clock hour with
# the [GB_CELL_LL] / sig-het cold audit as guards.
export GB_INMODEL_REPEATS_SURVIVOR=50

# ===========================================================================
# V9-1: PER-SOURCE IN-MODEL CONVERGENCE (2026-09-24 user design).
# ===========================================================================
# The two budgets above stop being what a newborn gets and become the SCALE
# the ceiling is derived from. Each ROW -- one (temp, walker, band) cell --
# now runs until its OWN cell lnL stops climbing, then freezes; a band leaves
# when its cold rungs are done; a finished band is swapped out for a queued
# one. SURVIVOR (mature) rows are untouched by default and keep the fixed 50.
#
# Full design + the measurements behind every default:
#   docs/superpowers/specs/2026-09-24-gb-inmodel-convergence-design.md
#
# THE RULE, per row: G = running sum of ACCEPTED delta_ll (== the cell's
# whole-model lnL up to a constant); retire once
#   max(G) has not improved by DLL over the last ITERS repeats.
# It is deliberately NOT the leaf cap's D/2 form: that gate's clock is a
# whole sampler iteration, and on a per-REPEAT clock the same code becomes a
# per-step test that retires a row mid-climb (measured: 17.3 lnL left on the
# table). And it is the GAIN, not `L_free + ll_ref`, because ll_ref is a
# sig-het value whose absolute offset is 1-17 lnL cold and is re-anchored
# mid-block every GB_SIGHET_REFRESH_EVERY repeats.
#
# ⚠ READ (thresh, ITERS) AS A RATE: a row is retired below DLL/ITERS lnL per
# repeat = 0.04 here. ⚠ AND NOTE THE FLOOR: a row that converged instantly
# still pays ITERS + 1 = 101 repeats, ABOVE v8's fixed 100 -- so this is a
# SPEND, not a saving, and the [GB_IMCONV] "work =" ratio will print above
# 1.0. That is the intended trade (user ruling: run to convergence, not to a
# fixed 100). ITERS=50 is the setting that makes it pay for itself instead.
export GB_INMODEL_CONVERGE=on
# ⚠⚠ FIRST JOB OF THE CAMPAIGN SHOULD ARGUABLY BE `observe`. It runs the
# ENTIRE rule under the unchanged fixed budget -- not one proposal differs --
# and logs the repeat/gain distribution the rule WOULD have produced. It is
# the paired negative control, and it answers the one question that decides
# whether arming is worth anything: if every row would retire at the floor
# having gained ~nothing, the rule is a constant budget of ITERS and
# GB_INMODEL_REPEATS_NEWBORN=100 already says that for free. Flip this line
# to `observe`, read the [GB_IMCONV ... calibration] line, then flip back.
#   export GB_INMODEL_CONVERGE=observe

# W, the patience window, in UNTHINNED in-model repeats (user ruling
# 2026-09-24, raised from 50; 2026-09-25 raised again 100 -> 250 "just to
# be safe" for births). 100 tracked climbs out to ~300 repeats where 50
# gave up at 76.
#
# MEASURED AT W=100 (job 620): repeats p10/p50/p90/max = 102/260/400/401,
# only 9.7% retiring at the floor and 94% converging on evidence -- so 100
# was NOT truncating. 250 buys margin on the tail rather than fixing a
# known cut-off.
#
# ⚠ THE FLOOR MOVES WITH IT. A row cannot retire before W+1, so the floor
# goes 101 -> 251, and the ceiling below HAS to move with it or the window
# never gets to speak -- see the note there.
export GB_INMODEL_CONVERGE_ITERS=250
# ---- SURVIVOR WINDOW (user ruling 2026-09-25, off job 628) -------------
# 250 stays for BIRTHS; survivors get 100. The window is a FLOOR on block
# length -- a row cannot converge before seen >= window -- so every block
# runs at least that many repeats whatever the physics says. Measured over
# 628's three complete search cycles:
#
#   survivors  median 260, p90 320-375, MAX 420-480, ~50% sat exactly at
#              the 255 floor
#   births     median 275-285, p90 435-500, max 1200-1865  <- these DO
#              use the depth, so 250 stays for them
#
# and the work is not evenly split: survivors are 96.3% of all converging
# in-model work, births 0.9%. Survivors' mean of 276-306 against a floor of
# 255 means ~90% of the in-model bill was the FLOOR, not convergence.
# Expected saving ~54% of survivor repeats -> ~44% of the iteration.
#
# ⚠ (window, thresh) IS A RATE: thresh/window lnL per repeat. Shortening
# the window at a fixed dll does NOT let survivors stop sooner at the same
# bar, it LOWERS the bar -- 250->100 at dll 4.0 would be 2.5x looser. The
# survivor dll is therefore DERIVED (4.0 * 100/250 = 1.6) so the enforced
# rate is IDENTICAL for both classes and only the earliest permitted exit
# moves. tests/test_inmodel_converge.py pins that, with a control showing
# what the unscaled version would have done.
# 0 = off (both classes share GB_INMODEL_CONVERGE_ITERS, pre-2026-09-25).
# Watch: [GB_IMCONV ... armed] now prints both windows, and the survivor
# block length should collapse toward ~105 from ~280.
#
# 100 -> 50 (user ruling 2026-09-26, off job 643). MEASURED over 1,956,023
# mature rows in five iterations: p10 is pinned at EXACTLY 101 -- the floor,
# window+1 -- 30.2% of rows stop there, and p50 is only 111-120. So the
# mature distribution is packed against the floor: the window is not
# measuring these rows, it is a fixed budget they serve out. Their median
# gain over the whole block is 0.08-0.23 lnL, i.e. they arrive converged.
# (Newborns are the opposite and unchanged at 250: median gain 5.18-5.26,
# above the 4.0 dll, once 09ca4819 stopped them exiting at 5 repeats.)
#
# The derived dll follows automatically: 4.0 * 50/250 = 0.8, so the enforced
# rate stays 0.016 lnL/repeat and only the earliest permitted exit moves.
# A row still genuinely climbing is unaffected -- it keeps climbing.
# Mature rows are 89.9% of all in-model row-repeats, so this is the largest
# remaining lever that costs no evidence.
# Watch on the next snapshot: mature p10 should read 51, not 101, and p50
# should fall from ~115 toward ~65. If p10 stays at 101 the knob did not
# reach the move.
# 50 -> 25 (user ruling 2026-09-27). Measured on 6mo job 650 over
# 52,951 survivor blocks: 40.2% of rows terminate AT the floor, i.e.
# they arrived already converged and we paid 51 repeats to prove it.
# The floor is window+1, so 25 proves the same thing in 26. Block
# medians run p50 60 / p75 76, so rows that DO need refinement keep
# room. ⚠ the threshold does NOT scale with this any more -- it stays
# at GB_INMODEL_CONVERGE_DLL=4.0, the same D/2 bar the band valve and
# the leaf cap use (GB_INMODEL_CONVERGE_DLL_SCALE_WITH_WINDOW=1
# restores the old rate-preserving rescale).
export GB_INMODEL_CONVERGE_ITERS_SURVIVOR=25
# The improvement threshold: D/2 = 0.5 * GB_LEAF_CAP_NDIM, the lnL a
# genuinely new D-parameter source has to buy. Same number as the leaf cap
# gate, but see the warning above -- same threshold, different clock.
export GB_INMODEL_CONVERGE_DLL=4.0
# Per-row ceiling. 0 would mean 4x the class budget = 400, which is now
# WRONG twice over: with W=250 the floor alone is 251, and 400 caps rows
# that are still climbing hard.
#
# THE EVIDENCE FOR GOING BIG (job 620 calibration, per rank):
#   lnL gained per row p10/p50/p90 = -12.80 / 21.57 / 222.53, max 1323.1
#   and on another rank max 4988.0
#   at the 400-repeat ceiling: 3.4-4.2% of rows
# The p90 of 222 and those maxima are rows still buying hundreds to
# thousands of lnL when the cap cut them off. Capping a row that is still
# adding logL is the one thing this knob should never do (user ruling
# 2026-09-25: "make the ceiling much larger than 1000 honestly if it is
# still adding logL").
#
# 5000 -> 20000 (user, 2026-09-25). At this size the ceiling stops being
# a budget and becomes a TRIPWIRE: the measured need is p50 260 / p90 400
# repeats, so a row reaching 20000 has run ~50x the p90 without its
# accumulated accepted delta_ll ever going W=250 repeats without a new
# best. A new loud [GB_IMCONV] warning fires whenever ANY row hits it --
# and says not to raise it again, because at that point the suspect is the
# THRESHOLD (DLL) or a row that is not converging at all (a hot rung that
# slipped the ladder gate, a source oscillating between modes).
#
# ⚠ WHY THIS IS NOT A COST BLOWUP. The ceiling is a BACKSTOP, not the
# primary control. Two other things stop a block first:
#   * the convergence rule itself -- 94% of gated rows retired on evidence
#     at W=100, not on the cap;
#   * GB_INMODEL_CONVERGE_STOP_FRAC=0.5 ends the whole block once half the
#     GATED rows have converged, and un-retired rows are released with it.
# So the ceiling only binds inside a block that is already running long,
# and raising it buys exactly the rows that are still paying. If the
# [GB_IMCONV] ceiling fraction climbs anyway, the thing to move is the
# THRESHOLD (GB_INMODEL_CONVERGE_DLL), not this.
export GB_INMODEL_CONVERGE_MAX=20000
# LADDER GATE: only the COLDEST half of the 24 rungs is tested, frozen, and
# allowed to hold a block open; hotter rungs keep sampling (they are the
# transport that feeds the cold rungs) and are reported "released".
# ⚠ Not cosmetic. A hot chain accepts freely, so its gain RANDOM-WALKS and
# the running max of a random walk keeps creeping -- over a 100-repeat window
# that clears D/2, so without the gate the hot rungs would be permanent
# stragglers holding every finished cold source hostage. 1.0 = every rung
# must converge (the pre-gate behaviour).
export GB_INMODEL_CONVERGE_GATE_FRAC=0.5
# COLD-CHAIN-ONLY COLUMN RETIREMENT (user ruling 2026-09-27: "with the
# vertical swaps working well, we should pull the whole group off the
# current running block when its cold chain converges").
#
# An ABSOLUTE rung count that overrides GATE_FRAC above. =1 means the T0
# row is the only one with a vote on when a (walker, band) column may
# leave the in-model block; the whole column -- all 24 rungs -- then goes
# together, and the hot rows are recorded as RELEASED (nothing was
# concluded about them) rather than converged.
#
# WHY NOT just set GATE_FRAC=1/24: the exact ratio does work, but this
# file exports a typed decimal and the answer flips on the fourth place --
# ceil(24 * 0.04) is 1, ceil(24 * 0.0417) is 2. An absolute count says
# what it means.
#
# WHAT IT BUYS. With GATE_FRAC=0.5 a block ran until the slowest of TWELVE
# rungs converged, and the warmer ones converge more slowly because their
# gain random-walks rather than climbs. Longer blocks deliver more lnL per
# iteration, which keeps resetting the per-(walker, band) valve's streak --
# so the hot rungs were holding the search stage open through the back
# door. Measured on job 650: 17.5% of blocks that PASSED the in-model test
# still delivered a median row gain above the valve's 4.0 tolerance.
# 1 -> 4 (user ruling 2026-09-27, refining the 09-27 cold-only
# ruling): "make sure that all 4 of the coldest 4 chains have
# either no picked source or its source has converged to a max
# logL." The coldest FOUR rungs gate the column, not just T0.
#
# Paired with the two-phase block: until those four are done,
# EVERY row runs to its own convergence; once they are, that
# column's remaining rungs get a fixed tail --
# GB_INMODEL_POST_GATE_NEWBORN=100 steps for births,
# GB_INMODEL_POST_GATE_SURVIVOR=0 for survivors.
export GB_INMODEL_CONVERGE_GATE_RUNGS=4
# SWAP TRIGGER: end a block once this fraction of its (walker, band) COLUMNS
# has fully retired, then swap the finished bands out for queued ones.
# ⚠ 1.0 SILENTLY DISABLES THE REFILL -- at 1.0 every column finishes together,
# nothing carries, and the loop degenerates into the fixed chunk loop.
export GB_INMODEL_CONVERGE_STOP_FRAC=0.5
# Retire finished bands and give their buffer slots to queued ones.
# ⚠ COSTS ONE BLOCK SETUP PER GENERATION, measured at ~73 in-model repeats
# (cholesky + sig-het reference + buffer refill). This is the one part of V9-1
# that can plausibly be net-negative on wall clock. A/B it: the [GB_IMCONV]
# line prints the generation count and the work ratio, and 0 turns the loop
# off while keeping the freeze.
#
# ⚠⚠ THE REFILL ONLY DOES ANYTHING IF THE POOL EXCEEDS THE CHUNK WIDTH.
# It swaps finished bands out for QUEUED ones, so with no queue it runs one
# generation and is a pure no-op. Check before crediting or blaming it:
# v8's own line "<N> survivors polished in <K> in-model chunk(s)" -- if K is
# 1 per class, there was never a queue and REFILL=1 changes nothing.
# ⚠ And note the setup accounting at THIS configuration: with
# GB_INMODEL_SETUP_BATCH=2048 a chunk is already split into ~
# (rows / 2040) sub-blocks, each paying its own setup. So a generation costs
# that many setups, not one, and the refill's marginal cost is one setup per
# 2040 CARRIED rows. Price it from the [GB_IMCONV] generation count, not from
# a single 73-repeat figure.
export GB_INMODEL_CONVERGE_REFILL=1
# BOTH CLASSES. User ruling 2026-09-25, final form: "What I want is RJ
# moves to have their sources run to convergence (in the search) whether
# they are birthed or survived."
#
# A pooled row is MATURE when it was alive at pick and survived the RJ
# step -- it was judged for death and KEPT. Under "newborn" those rows took
# the flat GB_INMODEL_REPEATS_SURVIVOR budget (50) while newborns
# converged, so a source that had just proved it was worth keeping got the
# LEAST polish in the pool. Both classes now run the window (250) /
# ceiling (20000) rule and stop on their own evidence.
#
# ⚠ TWO SCOPES, BOTH WANTED -- this is the distinction I got wrong twice.
#   * THIS knob is the per-ROW rule inside an RJ move's own pool. It is
#     what the ruling above is about.
#   * The pure in-model slots (in_model / in_model_fstat /
#     in_model_replace) that follow each RJ move converge per SUB-BAND via
#     the GROUP rule ([GB_IMGROUP]). That is the "let the in model moves
#     mix them up" half, it is a different scope, and it is untouched --
#     _run_in_model_repeats is not passed a converge state on that path.
#   Arming "mature" here therefore does NOT change the in_model* budgets.
#
# ⚠ COST -- the largest single increase in this batch. Newborns are a
# minority of the pool; mature rows are the bulk once the search has found
# most of the catalogue. Bounded by the same two mechanisms as newborns:
# the stop is evidence-based, and GB_INMODEL_CONVERGE_STOP_FRAC=0.5 ends a
# block once half the gated rows are done. Watch [GB_IMCONV ... mature]:
# a high "at floor" fraction means mature rows plateau immediately and the
# convergence is buying them nothing.
export GB_INMODEL_CONVERGE_CLASSES=newborn,mature

# ===========================================================================
# V9-6: IN-GROUP CONVERGENCE FOR THE PURELY IN-MODEL PROPOSALS
# ===========================================================================
# THIS IS A DIFFERENT SCOPE FROM V9-1 ABOVE. Three convergence rules are now
# live and they share vocabulary, so, explicitly:
#
#   unit      statistic                 clock              lifetime
#   row       accepted delta_ll         one in-model REPEAT  one repeat block
#             (V9-1, newborns only, inside the RJ moves)
#   (w,band)  sub-band cold-chain logL  one in-model PASS    ONE proposal   <- V9-6
#   (w,band)  sub-band cold-chain logL  one sampler ITER     one recipe stage
#             (the per-walker band shutoff, other session)
#
# V9-6: the PURE in-model move repeats its whole pass until every
# (walker, band) sub-band's cold-chain logL has converged, shutting each
# sub-band off as it converges, and ends when all are shut. Pure in-model
# moves only -- armed on an RJ move it logs a warning and does nothing,
# because on an RJ pass a flat sub-band logL means the BIRTHS stopped
# paying, which is the stage valve's decision, not this one's.
# ⚠ THE GROUP RULE IS DEAD WITHOUT THIS PAIR. The move it lives on is the
# PURE in-model move, which is built only when search_in_model is set and
# is otherwise absent from the stage entirely -- so with GB_SEARCH_IN_MODEL
# unset the knobs below are exported, ignored, and nothing logs [GB_IMGROUP]
# at all. That is exactly the silent no-op the preflight exists to catch.
export GB_SEARCH_IN_MODEL=1
# Repeats per source per PASS of that move (the group rule then decides how
# many passes). 25 per the user's structure.
#
# ⚠ DO NOT RAISE THIS EXPECTING A SPEEDUP. It was briefly set to 100 on
# 2026-09-25 and reverted the same day, because the group rule's window is
# GB_INMODEL_GROUP_ITERS *PASSES* and one pass is this many repeats per
# source -- so the EFFECTIVE window is 3 x this, against a FIXED
# GB_INMODEL_GROUP_DLL of 4.0:
#
#     25  -> 4.0 over  75 repeats = 0.053 lnL/repeat
#    100  -> 4.0 over 300 repeats = 0.013 lnL/repeat   (4x STRICTER)
#
# A fatter pass posts a proportionally LARGER per-pass gain, so the 3-pass
# total stays above 4.0 for MORE passes, not fewer. Raising this is 4x the
# work per pass AND a tighter bar. (Unlike the per-ROW rule in V9-1, this
# convergence is per SUB-BAND, so there is no per-source averaging to
# absorb it.) If fatter passes are ever wanted, GB_INMODEL_GROUP_DLL must
# scale with them -- 50 pairs with dll 8.0 -- to hold the rate.
export GB_NUM_REPEAT_PROPOSALS=25
export GB_INMODEL_GROUP=1
# W in PASSES (not repeats): one pass is already GB_NUM_REPEAT_PROPOSALS
# repeats per source, so the per-pass gain is a far coarser quantity than
# the per-repeat one and needs nothing like the 100-repeat window V9-1 uses.
# >>> 3 -> 2 (user ruling 2026-09-26, off job 634's gb_search_1 it 1).
# >>> The window is a FLOOR: a sub-band cannot be judged until it has W
# >>> passes of history, so passes 1..W shut NOTHING. Measured:
# >>>
# >>>   pass 1  1013 blocks  3,372,150 rep*src   open 448
# >>>   pass 2  1006 blocks  3,412,575           open 448   <- +shut 0
# >>>   pass 3  1020 blocks  3,310,825           open 448   <- +shut 0
# >>>   pass 4   868 blocks  1,543,025           open 118   <- 90% shut here
# >>>   pass 5+  tail, 57 -> 0 bands, 17.5% of the work
# >>>
# >>> The three warm-up passes are 72% of the move's work and the first
# >>> three quarters of it cannot retire a single band. Dropping to 2
# >>> removes one full warm-up pass (~3.3M rep*src, ~24% of the move) and
# >>> lets bands shut at pass 3 instead of pass 4.
# >>> 2 -> 3 (user ruling 2026-10-01, "for now", for the ratcheted
# >>> gb_search_3 relaunch): the noise is MOVING in that stage (nudge /
# >>> release), so a sub-band's per-pass gain is judged against a
# >>> residual that just changed under it; one more pass of history
# >>> before a sub-band can retire. The per-repeat bar goes back to
# >>> 4.0/(3*25) = 0.0533 lnL/repeat (see the CONSEQUENCE note below).
export GB_INMODEL_GROUP_ITERS=3
# D/2 again, and FLAT per sub-band (user ruling): "When a source is birthed,
# it is per-source. During the special in-model only proposals it is
# per-sub-band. I want flat D/2. This will focus more resources on the
# sub-bands with more sources." The occupancy scaling is the INTENT -- a
# dense band posts a larger per-pass gain, clears the flat bar for longer,
# and so earns more passes. GB_INMODEL_GROUP_SCALE=per_source is the
# alternative; both numbers are logged every pass.
# STAYS AT 4.0 = D/2 WHEN THE WINDOW MOVES (user ruling 2026-09-26).
# I proposed deriving it (2.6667) to hold the per-repeat rate constant
# when ITERS went 3 -> 2. The user kept 4.0, and that is the right call:
# D/2 is a PHYSICAL quantity, not a rate. It is the lnL a genuinely new
# D-parameter source has to buy, and it is deliberately the SAME number
# the leaf cap's improvement gate uses, "so the two features cannot
# disagree about what counts as an improvement worth waiting for".
# Deriving it from the window would have broken that tie and left D/2
# meaning something different here than it means three knobs away.
#
# ⚠ CONSEQUENCE, stated so it is chosen and not discovered: holding dll
# while the window shortens DOES loosen the per-repeat bar.
#   3 passes, dll 4.0 -> 4.0/(3*25) = 0.0533 lnL/repeat
#   2 passes, dll 4.0 -> 4.0/(2*25) = 0.0800 lnL/repeat   (1.5x looser)
# So sub-bands shut EARLIER than the pass-count change alone implies --
# which is the direction we want here, on top of the warm-up pass saved.
# The read is unchanged and still physical: a sub-band retires once its
# whole model gains less than one source's worth over the window.
export GB_INMODEL_GROUP_DLL=4.0
export GB_INMODEL_GROUP_SCALE=flat
# ⚠ COST KNOB, NOT A SAFETY NET. This group is unbounded by construction:
# one pass is a full sweep of every source at 25 repeats each, so 50 passes
# is up to 1250 repeats/source for the slowest sub-band. Watch the
# [GB_IMGROUP] "group done" line's pass distribution and tighten if the p90
# is at the ceiling.
#
# 20 -> 50 (user, 2026-09-25: "this may need to be higher"). Two reasons:
#
#  * SCALE=flat above makes crowded bands structurally slower to shut --
#    that is the INTENT -- so the ceiling truncates exactly the bands the
#    design is trying to favour. At 20 passes a band gets 500
#    repeats/source, and the row-level rule measured on job 620 needs a
#    MEDIAN of 260 and a p90 of 400 repeats for a SINGLE newborn to
#    converge. 20 was ~1.25x the p90 for one source, for a bar the whole
#    band has to clear in aggregate.
#  * raising it is nearly free. A shut sub-band leaves the eligible set
#    (``_group_shutoff_wb`` is the next pass's filter), so late passes only
#    touch what is still open -- with ~61 columns, one stubborn band costs
#    ~1/61 of a pass. The expensive case is MANY bands still open, and that
#    trips the ceiling WARNING, which names the remedy.
#
# ⚠ 50 is reasoned, not measured: the group rule was inert on the fan-out
# path until 2026-09-25, so no run has ever produced a [GB_IMGROUP] pass
# distribution. The first one settles this number.
export GB_INMODEL_GROUP_MAX_PASSES=1000
echo "[V9-IMGROUP] group=${GB_INMODEL_GROUP} iters=${GB_INMODEL_GROUP_ITERS} \
dll=${GB_INMODEL_GROUP_DLL} scale=${GB_INMODEL_GROUP_SCALE} \
max_passes=${GB_INMODEL_GROUP_MAX_PASSES}"
# ---- V9 FEATURE PREFLIGHT --------------------------------------------------
# ⚠ AN UNRECOGNIZED ENV VAR IS SILENTLY IGNORED (the stock-settings rule in
# CLAUDE.md). Every V9-1/V9-6 knob above is new, so running this script
# against an installed lisatools that predates them yields a run that looks
# completely normal, logs nothing unusual, and quietly behaves like v8 with
# caps off. That is the worst possible failure: a wasted allocation that
# produces plausible output. Refuse instead.
#
# ⚠ AND A PRESENCE CHECK IS NOT ENOUGH. The 2026-09-24 audit found the group
# knobs DEAD while every symbol below existed: the resolver built the env
# name ``GB_INMODEL_CONVERGE_GROUP*`` while this script, every docstring and
# every log line said ``GB_INMODEL_GROUP*``. ``hasattr`` cannot see that. So
# the second half of this preflight RESOLVES the knobs from the environment
# this script has actually exported and refuses if a value does not come
# back -- the only check that covers a name mismatch.
python - <<'PYEOF' || exit 2
import sys
from lisatools.globalfit.moves import gbspecialstretch as g

missing = [n for n in ("_InModelConvergeState", "_InModelGroupState",
                       "_converge_stage_allows", "_converge_gate_mask",
                       "_resolve_converge_knob")
           if not hasattr(g, n)]
if missing:
    print("[V9-PREFLIGHT] REFUSING: the installed lisatools has no "
          f"in-model convergence support (missing {missing}). The "
          "GB_INMODEL_CONVERGE / GB_INMODEL_GROUP knobs this script exports "
          "would be SILENTLY IGNORED and the run would quietly be v8. "
          "Install dev at or after the gb-inmodel-converge merge.")
    sys.exit(2)

# RESOLVE, do not merely probe. Each entry is (knob suffix, family, cast,
# a default this script's exported value must DIFFER from).
# The sentinels must be values the cast ACCEPTS (so an unset knob comes back
# as the sentinel and is reported, rather than raising out of the cast and
# hiding which knob was missing) and must differ from what this script sets.
_checks = [
    ("", "INMODEL_GROUP", g._converge_cast_refill, False),
    ("iters", "INMODEL_GROUP", g._converge_cast_window, 999999),
    ("dll", "INMODEL_GROUP", float, -12345.0),
    ("scale", "INMODEL_GROUP", g._converge_cast_scale, "per_source"),
    ("max_passes", "INMODEL_GROUP", int, 999999),
    ("", "INMODEL_CONVERGE", g._converge_cast_mode, "off"),
]
bad = []
for knob, family, cast, sentinel in _checks:
    try:
        got = g._resolve_converge_knob("gb", knob, None, sentinel, cast,
                                       family=family)
    except TypeError:
        bad.append(f"{family}_{knob or '<master>'} (resolver has no 'family' "
                   f"parameter -- this lisatools predates the fix)")
        continue
    except Exception as exc:
        bad.append(f"GB_{family}{'_' + knob.upper() if knob else ''} "
                   f"did not resolve: {exc!r}")
        continue
    if got == sentinel:
        bad.append(f"GB_{family}{'_' + knob.upper() if knob else ''} "
                   f"resolved to the sentinel {sentinel!r}, i.e. the "
                   f"exported value was NOT SEEN")
if bad:
    print("[V9-PREFLIGHT] REFUSING: exported knobs are not reaching the "
          "code:\n  - " + "\n  - ".join(bad))
    sys.exit(2)

# The v9 recipe restructure and its per-stage profile.
from lisatools.globalfit import recipe as _r
from lisatools.sampling import fstat_proposal as _fp
_need = [("recipe.SearchStageProfileStep", hasattr(_r, "SearchStageProfileStep")),
         ("recipe.force_fstat_refit", hasattr(_r, "force_fstat_refit")),
         ("recipe.band_shutoff_w_armed", hasattr(_r, "band_shutoff_w_armed")),
         ("fstat_proposal.set_peak_min_F_override",
          hasattr(_fp, "set_peak_min_F_override")),
         ("GBSpecialRJFStatGridMove.arm_fstat_refit",
          hasattr(g.GBSpecialRJFStatGridMove, "arm_fstat_refit")),
         ("Stage kind 'gb_search'", "gb_search" in _r.Stage._KINDS)]
_absent = [n for n, ok in _need if not ok]
if _absent:
    print("[V9-PREFLIGHT] REFUSING: the installed lisatools has no v9 "
          f"three-stage search support (missing {_absent}). STAGE_V9_SEARCH "
          "would fail at composition, or worse, compose a single legacy "
          "gb_search stage with none of the per-stage profiles applied.")
    sys.exit(2)
# The other two knobs that are NEW on 2026-09-24 and SILENTLY IGNORED on an
# older install. Neither goes through _resolve_converge_knob, so the sentinel
# pass above cannot see them; a plain source probe is enough, because both
# are read with a literal os.environ.get at a known site.
import inspect
# NOTE the fancy-tempering name is built from an f-string
# (f"{BRANCH}_RUN_FANCY_TEMPERING"), so the literal is not in the source --
# probe the SUFFIX, which is what the f-string contains.
_src_checks = [
    ("_RUN_FANCY_TEMPERING", inspect.getsource(g),
     "the permuted swaps would stay ON and the ladder would keep adapting "
     "from them -- v9 expects them OFF with the ladder adapting from the "
     "VERTICAL swaps instead"),
    ("GB_REPLACE_WARM_PASS", inspect.getsource(_r),
     "rj_replace would run SINGLE-pass (F-stat only); the warm pass the "
     "2026-09-24 ruling asked for would silently not happen"),
]
_stale = [f"{n}: not read anywhere in the installed module -- {why}"
          for n, src, why in _src_checks if n not in src]
if _stale:
    print("[V9-PREFLIGHT] REFUSING: knobs this script exports are not "
          "implemented in the installed lisatools:\n  - "
          + "\n  - ".join(_stale))
    sys.exit(2)

print("[V9-PREFLIGHT] in-model convergence + group knobs RESOLVE from the "
      "environment; v9 stage-profile support present; "
      "GB_RUN_FANCY_TEMPERING and GB_REPLACE_WARM_PASS are implemented.")
PYEOF
# WHAT TO WATCH (grep '[GB_IMGROUP]'):
#   * "pass N: cold dlnL +X over M occupied sub-band(s); shut A (+B this
#     pass) / open C (D occupied); mean gain/occupied band ..., per source
#     ..." -- one line PER PASS. The per-source column is what tells you
#     whether flat-vs-per_source would have behaved differently.
#   * "group CONVERGED after N pass(es)" or "group hit the N-pass CEILING
#     with M sub-band(s) still open" -- the ceiling line is a WARNING and
#     means the group is being truncated, not finishing.
#   * "group done: N pass(es) x R repeats/source; shut-off pass
#     p10/p50/p90/max ... over OCCUPIED pairs; K pair(s) never shut.
#     Compare against a fixed 5-pass block: Fx the passes." -- the cost
#     line. F > 1 means this is costing more than the 5-pass block it
#     replaced.
# ===========================================================================
echo "[V9-IMCONV] mode=${GB_INMODEL_CONVERGE} iters=${GB_INMODEL_CONVERGE_ITERS} \
dll=${GB_INMODEL_CONVERGE_DLL} ceiling=${GB_INMODEL_CONVERGE_MAX:-0}(0=4x) \
gate_frac=${GB_INMODEL_CONVERGE_GATE_FRAC} \
stop_frac=${GB_INMODEL_CONVERGE_STOP_FRAC} \
refill=${GB_INMODEL_CONVERGE_REFILL} classes=${GB_INMODEL_CONVERGE_CLASSES}"
# WHAT TO WATCH, in priority order (all at INFO, one set per class per
# in-model phase; grep '[GB_IMCONV]'):
#   1. "... calibration: ... AT FLOOR (<=101) N (X%); lnL gained per row
#      p10/p50/p90 ... rows gaining < dll(4.00) = M (Y%)"
#      THE line. X near 100% with Y near 100% = the rule never fired on
#      evidence and is just a constant budget of ITERS -- recalibrate DLL
#      against a measured delta_ll distribution, do not just ship it.
#      A broad repeat spread with p50 gain of order D/2 and up = working.
#   2. "... work = N.NNx a fixed 100-repeat budget, plus K extra block
#      setup(s) from the refill" -- the cost. Multiply K by ~73 repeats to
#      price the refill against the repeats saved.
#   3. "... by rung: GATED ... | RELEASED ..." -- the two populations are
#      judged by different rules; a pooled number hides which one is paying.
#   4. "... armed (on): window ... gate = coldest 12/24 rung(s) ..." -- one
#      line per move at first use, and it NAMES every move that armed. The
#      mode refuses PE-stage moves (optional stopping is search-only) and
#      says so once; confirm rj_fstat_search/rj_prior_removal are in and
#      nothing named *_pe is.
# ===========================================================================

# VERTICAL TEMPERING ON (2026-08-26 user ruling: "this is crucial").
# Per-repeat vertical band-temperature swaps inside the in-model loop
# (same walker, adjacent rungs) -- built+tested 2026-08-18 (e6ed71e2,
# 30 tests) but never promoted: the flag sat default-off through the
# v6/overlap/replace campaigns. It is ADDITIVE to the permuted swaps
# and never drives ladder adaptation. Directly attacks the measured
# transport bottleneck (correct fdot living at rung 4 while cold holds
# the mosaic): exchange cadence per in-model REPEAT (~100x/block) vs
# once-per-iteration permuted swaps. Known approximation: a mid-block
# vertical swap exchanges occupancy without updating the drift-gate
# census (self-corrects next block). =0 reverts.
# ===========================================================================
# V9-7: PERMUTED ("FANCY") BAND-TEMPERATURE SWAPS OFF (user ruling
# 2026-09-24: "make sure the run_tempering is entirely turned off because
# we have no fancy swaps").
# ===========================================================================
# ⚠ THIS IS THE ONLY RELIABLE OFF SWITCH, and the obvious candidates are
# both traps:
#   * GB_TEMPER_EVERY_PROPOSES only THROTTLES. In _temper_cadence_fire,
#     `n <= 1` returns True unconditionally, so 0 means "fire on EVERY
#     eligible propose" -- the exact opposite of off.
#   * per-move run_swaps= is set from several different expressions in the
#     recipe (`_temper_all_moves or not _temper_on_removal`, hard False on
#     some moves), so there is no single recipe value to flip.
# GB_RUN_FANCY_TEMPERING=0 forces run_swaps False for every GB move at once, at
# the one choke point in the constructor, and logs once that it did.
export GB_RUN_FANCY_TEMPERING=0
# ⚠ AND NOTE WHAT THIS DOES **NOT** TOUCH: the per-repeat VERTICAL rung
# swaps below. They live inside the in-model loop, are ADDITIVE to the
# permuted swaps rather than part of them, and are the transport the
# in-model convergence work depends on -- killing the fancy swaps must not
# silently take them too. They stay ON.
# ===========================================================================
export GB_TEMPER_VERTICAL=${GB_TEMPER_VERTICAL:-1}

# ---- ALL-RUNGS VERTICAL SWAP (user ruling 2026-09-26: turn it on) --------
# Swap EVERY rung of a (walker, band) column, not just the rungs that
# happen to hold a picked source. "Even if it does not have a source in
# that band it has a total likelihood ... and should get swapped
# accordingly." A cold cell whose whole model is net-harmful against the
# bare slab can now be evicted upward -- the route recorded as missing in
# project_sole_occupant_swap_gap.
#
# Inactive rungs are NOT made resident: their total is computed once per
# block and swapped against as a number ("you do not need to load
# information into the buffer for good for bands without an active
# source"). Buffer residency is unchanged.
#
# ⚠ TURNED ON AHEAD OF ITS OWN GATE. The design spec asks for a clean
# [GB_CELL_LL] sampled-vs-realized line on a real run before this stops
# being opt-in; it has unit tests and five mutation controls but no
# production hours. WATCH [GB_CELL_LL] on the first snapshot -- both of
# this path's failure modes (a mis-credited cell ledger, a half-applied
# relabel) are invisible in aggregate counts and show up ONLY there.
# Also watch [GB_VERT] for "N pair(s) dropped as unpriceable" (an
# occupied-but-unpicked rung, whose total nothing measured) and for the
# per-rung acceptances staying BELOW ~1.0 -- an empty<->empty pair would
# be accepted unconditionally and collapse the band's ladder, which is
# why those pairs are dropped.
# GB_TEMPER_ALL_RUNGS=0 restores the picked-row-only sweep exactly.
export GB_TEMPER_ALL_RUNGS=1
# EXACT SWAP PRICING (2026-10-04, 6mo replica_pe job 715): the vertical
# swap prices each rung with L_free + ll_ref, and ll_ref is the sig-het
# RUNNING value -- its anchor-level error cancels inside one rung's MH
# ratio but not between two cells (end-of-block audit: block-max
# |sig-het - exact| p90 40-77 nats, max 43,000 on hot rungs; the cold
# chain lost ~1,400 lnL per walker on every prior-RJ iteration and won it
# back on the F-stat ones). =1 prices every sweep with an exact add-delta
# through the chunked engine (one batched call per sweep); ll_ref and every
# in-rung MH ratio are untouched. DEFAULT 0 until the copy-store comparison
# confirms it. Watch [GB_TIMING] inmodel_vertical_exact_price.
export GB_VERT_EXACT_PRICE=${GB_VERT_EXACT_PRICE:-0}

# PERMUTED-SWAP CADENCE 3 -> 1 (user ruling 2026-08-26): fire the
# permuted band swaps after EVERY GB propose -- 3x/iteration in search
# (was once, on the third move), and every PE iteration (was every ~2-3:
# the measured PE transport drought). Pairs with vertical: permuted
# swaps move whole band contents between rungs, vertical pumps
# per-repeat during polish -- the full transport stack. Probe cost
# ~+40 s/it (tempering block x3); production ~+3%. =3 reverts.
export GB_TEMPER_EVERY_PROPOSES=1
# ---- 10-WALKER-ARM TEMPERING STACK (ported 2026-09-11) --------------------
# COLUMN STAGING (b3f5acbd): every rung of a (walker, band) column staged
# together, pools ordered (band, walker, temp). REQUIRED with
# GB_TEMPER_VERTICAL=1 above -- jobs 465-469 ran order=count and the
# vertical sweep starved/biased (the leaf-shedding root cause pair); job
# 473 on order=band + the whole-cell L_with ratio is the validated config.
export GB_TEMPER_CELL_ORDER=band
# RUNNING-BACKUP CADENCE (2026-09-14, ported from the 3mo 10-walker arm):
# job 479 spent 60-80 s of every 5.4 min PE iteration in [SAVE] save_step
# -- the sampler's blocking handoff waiting for the saver rank to finish
# copying + fsyncing the 14 GB store into *_running_backup_copy.h5 after
# EVERY save. Every 10th save instead; the 600 s mid-iteration checkpoint
# still bounds a torn-store loss to that interval. The 6mo store will be
# larger still, so this matters MORE here.
export BACKUP_ITER=10
export GB_TEMPER_PRELOAD_CELLS=4800
# One occupancy census per unit (assert-guarded exact) + drop inert rows.
export GB_TEMPER_CENSUS_HOIST=1
export GB_TEMPER_COMPACT_ROWS=1
# Adaptive gb_search noise rider (379ae2e1): keep taking MaxLogL rounds
# while a round improves by > tol, stop at the first flat one.
export GB_SEARCH_NOISE_CHECKS=1
export GB_SEARCH_NOISE_ITERS_PER_STEP=100
# PLATEAU TOLERANCE 5 -> 20 lnL (user ruling 2026-09-18). MAXLOGL_TOL is what
# counts as "this round improved"; JointMaxLogLSearch keeps taking rounds while
# a round beats it and stops at the first flat one.
#
# Measured on the 6-month 4-GPU run: the rider took ~11.5 rounds per gb_search
# iteration and ran INTO its 10-round ceiling (GB_SEARCH_NOISE_ITERS_PER_STEP=0
# resolves to MAXLOGL_ITERS_PER_STEP, default 10), costing 101.4 s of a 320 s
# steady-state iteration -- 32%, the single largest line item. The [MAXLOGL]
# trace says where that goes: ROUND 1 carries the real re-tracking (IMPROVED
# jumps of hundreds of lnL), rounds 2+ add ~5 lnL each -- exactly AT the old
# tol, so the plateau rule could never fire and the loop ground to the cap
# chasing the tol-level wobble of an ensemble already near the mode.
#
# 20 keeps round 1 whenever the GB residual genuinely moved and cuts the
# wobble. Against a cold logL of ~1.04e8 gaining ~2e4 per iteration, a 20 lnL
# floor is far below anything that matters.
#
# SCOPE, deliberately noted: MAXLOGL_TOL is GLOBAL to JointMaxLogLSearch, so it
# also loosens the standalone noise_search / noise_vgb_search stages, which
# will now declare plateau sooner. Those stages only need the noise roughly
# converged before the next stage samples it in PE mode (see the tol default's
# own comment), so this is judged acceptable -- but if the searches come out
# under-converged, the rider needs its own tol knob rather than reverting this.
export MAXLOGL_TOL=20
# PE-only exclusive RJ draw (b9aae51f).
export GB_PE_RJ_DRAW_ONE=1
# ONE of rj_fstat_pe / rj_prior_pe / rj_warm_pe per PE iteration (user ruling
# 2026-10-02: "put the warm proposal in with the other two rj moves. Draw them
# at 0.45 prior, 0.45 fstat, 0.1 warm start"); prior takes the remainder.
export GB_PE_RJ_FSTAT_FRACTION=${GB_PE_RJ_FSTAT_FRACTION:-0.45}
# Warm share 0.1 -> 0.4 (user ruling 2026-10-18, with the residual split 0.9
# below): on the 9mo job 773 the warm (self-refit) draw accepted 18.6 cold
# births per draw against 2.5 for the residual grid, 2.5 for the prior and
# 0.4 for the GB-free galaxy grid, so the mix moves toward the channels that
# still find sources; prior takes the remainder (0.15).
export GB_PE_RJ_WARM_FRACTION=${GB_PE_RJ_WARM_FRACTION:-0.4}
# Per-block EXACT info matrices through the sig-het fast route
# (~2.4 ms/src vs ~29-46 chunked). The data_index misindex is FIXED and
# multi-GPU slots now route by the BUFFER's slot shards. First
# shakedown: set SIGHET_INFOMAT_VALIDATE=1 for one propose to log the
# fast-vs-chunked reldiff (expect ~1e-4 near-peak, larger off-peak =
# observed-vs-Fisher, fine for a proposal), then remove.
# ---- V4 sig-het block (2026-08-18) ------------------------------------
# SPARSE TIME GRID. Every production script has been running ~35 h sparse
# spacing -- 3-mo default 64 (snaps to 60, stride 36), 6-mo 120 (stride 36),
# 23-mo 525 (stride 32) -- the "constant temporal density" prescription from
# the accuracy studies. Measured on the high-f probe 2026-08-18, the compiled
# BANDED v5 kernel wants finer than that: the delta-vs-delta likelihood error
# eps/T on the loud block ran 0.131 at 36 h against 0.072 at 8 h, and the
# trust-gate rejection fell from ~15% to 9.0%. The prescription was
# calibrated with gb_sighet_tier_assess.py, which builds its engine with NO
# v3_n_nodes / v4_knots / v4_band / v5 (i.e. the v2 path) and never calls a
# compiled v3/v4/v5 kernel at all -- so it cannot speak to this engine.
# 270 is an EXACT divisor of Nt=2160 (stride 8 -> 8.0 h), so it lands
# cleanly instead of snapping. If it exceeds the device shared budget the
# fstat scorer FAILS AT SETUP naming the largest value that fits (~2 min,
# not a wasted run) -- it is never silently coarsened.
#
# ---- REVERTED 2026-08-18: 270 OOMs, and the accuracy case did not hold ----
# MEMORY. The sig-het stash is (n, nch, nch, Nf_active, N_sparse_t)
# complex128 in _expand_B -- 4x _expand_B + 4x _expand_A per setup -- so it
# goes as CELLS x N_sparse_t and the two knobs MULTIPLY. 270 raised
# N_sparse_t 60 -> 265 while GB_N_SUBBANDS stayed at the 3-month 8192:
#   3-mo default  60 x 8192 = 4.9e5  OK (= v3)   |  6-mo  118 x 4096 = 4.8e5 OK
#   23-mo        525 x 2048 = 1.1e6  OK          |  v4@270 265 x 8192 = 2.2e6 OOM
# It died on a 14.4 GB request at 91.5 GB allocated on a 99.9 GB card --
# 2x the 23-month run's product on the SHORTEST baseline. The confined
# probes could not surface this: n scales with cells and they ran 128
# against production's 16384.
#
# ACCURACY. The 0.131@36h vs 0.072@8h A/B above predates GBGPU b412089, so
# NEITHER run logged its resolved config -- and 0.131 is also the nodes=64
# value from the separate 32/64/128 sweep (0.103/0.131/0.093), i.e. the two
# arms are not a clean pair. With the echo finally live, four temper arms
# CONFIRMED at nt_layer=270 measured eps/T 0.087-0.102 at small
# displacement -- squarely inside that 0.093-0.131 spread, not 0.072. The
# gain did not reproduce.
#
# And it should not have been expected to. project_sighet_v4_plan.md,
# A100 bench-off 2026-08-03: at 3 months a CPU run at STRIDE 1 (maximum
# possible resolution) still gave 2.90 at T=100 vs 3.70 at the 16 h grid --
# "that tail is resolution-independent ... the known deep-null fit tail
# (remedy = SNR-aware / null-densified nodes)". Same note, longer
# baselines: "a 1.6x finer sparse grid changes NOTHING, and where it
# changes anything it is worse." The real lever is NODE PLACEMENT.
#
# The "production default is TOO COARSE" line in that note (which motivated
# 270) argues for matching the BENCHMARK's 16 h configuration; it was
# written against a SHARED-memory ceiling and never costed the global-memory
# stash above. It is not a measured 3-month accuracy deficit.
#
# TO RE-TEST: 270 needs GB_N_SUBBANDS=2048 to fit (265 x 2048 = 5.4e5), at
# a throughput cost -- fewer resident cells means more sequential passes.
# export SIGHET_NT_LAYER=270
#
# Confirm from the log, do not assume -- nothing else echoes these:
#   grep "sig-het engine resolved" <store>/gf_prod_3mo_artifacts/globalfit_run.log
# want: nt_layer=270 (stride 8) ... sparse spacing 8.0 h   (GBGPU b412089)
#
# MULTI-DEVICE F-STAT FAN-OUT. Validated 2026-08-13 (a86c52af): the =check
# gate ran on 2xH100, full comb+stageB in 122.3 s / 224 peaks with ZERO
# diverging batches vs the pinned scorer; the 2026-08-12 divergence was
# closed by the drift-campaign replica fixes. The 23-month script has used
# it since. Without it every refit runs the serialized pinned scorer. Real
# lane overlap needs a GBGPU wheel at/after 4381300 (GIL release); older
# wheels stay CORRECT but serialized, so this is safe either way.
export FSTAT_SIGHET_MULTIDEV=1
#
# SIG-HET REFERENCE REFRESH (user ruling 2026-08-18). The trust gate measures
# drift from ``ref_track`` -- the parameters the sig-het reference was built
# at -- so REFRESHING THE REFERENCE RESETS THE BUDGET. With no refresh a
# source spends all 200 repeats accumulating against one fixed expansion
# point, which is exactly why the drift audit pinned at max 0.47-0.50 of a
# 0.5 budget in EVERY block of every probe.
#
# BOTH knobs are required. ``REFRESH_EVERY`` is only the cadence at which
# FARNESS IS CHECKED; the refresh fires only for sources with
# ``drift > sighet_refresh_dphase``, which defaults to 0.5 -- the trust gate
# itself. The gate stops the drift one step before the refresh would notice,
# so the phase arm can NEVER trigger at the defaults: two knobs that must
# differ ship equal. DPHASE=0 makes it "refresh anything that moved" (a
# source that never accepted a move still has an exact reference).
#
# The counter is REPEATS, not iterations: newborn blocks (200) get 8
# refreshes, survivor blocks (25) get none -- the `move_i + 1 < n_rep` guard
# -- which is right, they barely drift.
#
# Cost, measured on the high-f probe with this exact configuration:
# inmodel_sighet_refresh 0.25 s typical / 6.26 s worst against
# inmodel_repeats 5.2-15.1 s and an 85-98 s propose = 0.3% typical, 6.4%
# worst. In-model is only ~5% of a propose (rj_step is 87%).
#
# WHY THIS RATHER THAN WIDENING THE GATE: refreshing re-linearizes, so it
# buys mixing while PRESERVING accuracy; widening buys the same mixing by
# SPENDING accuracy. Same reason GB_SIGHET_TRUST_PHASE_C stays at 0 here.
# >>> 25 -> 50 (2026-09-25): inmodel_sighet_refresh was the single largest
# >>> line item inside the in-model polish in job 628 (167-186 s per RJ
# >>> move). Halving the cadence halves that. The reference goes staler
# >>> between rebuilds, so the trade is mixing against accuracy -- and both
# >>> sides are instrumented: watch [GB_TRUST] (was rejecting 4-13% of
# >>> candidates) and the end-of-block 'll AUDIT vs exact' COLD median (was
# >>> 0.026-0.073, max 0.14-0.73). If either degrades materially, revert.
export GB_SIGHET_REFRESH_EVERY=25
# 50 -> 25 (2026-09-28, user ruling). Paired with
# GB_TEMPER_VERTICAL_AT_REFIT below: the vertical sweep now runs ONLY on
# repeats that re-anchored the references, so the swap and the refit
# land in the same repeat. A cold row that swaps up and down across
# 25-50 proposals against references anchored somewhere else can lose
# proper dll tracking -- ll_ref is an add-delta against the reference it
# was built on, so the further a row has walked since, the more the swap
# ratio compares two cells through different expansion points.
# Expected cost signature: inmodel_sighet_refresh roughly DOUBLES
# (twice as many refit ticks) while inmodel_vertical_swap COLLAPSES
# (~25x fewer sweeps). If vertical_swap does not fall, the gate is not
# taking -- check [GB_VERT] '(N sweep(s))'.
export GB_TEMPER_VERTICAL_AT_REFIT=1
# DPHASE 0 -> 0.1 rad (2026-09-11, ported from the 10w arm 379ae2e1):
# refresh only sources drifted past 0.1 rad since their reference build.
# Error bound at 0.1 rad = 2.4e-7 x SNR^2/2 = 0.004 lnL at SNR 184
# (Experiment A); at DPHASE=0 job 473 refreshed 89% of sources every
# check for ~20 s/it of pure rebuild. The 2026-08-18 "ALL RUNGS REFRESH"
# ruling below was written against the 0.1 DEFAULT's rung gating -- this
# knob is a drift threshold, not a rung filter, so hot rungs still
# refresh once they drift.
# REVERTED to 0 (2026-09-11 evening, ported from the 3mo 10-walker arm):
# 0.1 rad refreshed 81% of sources instead of 89% at 3mo and 94% instead
# of 97% at 1yr -- no measurable wall-clock saving -- while the 1yr cold
# audit max rose 15 -> 22 in the same relaunch (cause not isolated).
# Accuracy is primary: refresh every check.
export GB_SIGHET_REFRESH_DPHASE=0
# ALL RUNGS REFRESH (user ruling 2026-08-18). The default 0.1 keeps a stale
# reference on everything hotter, justified in the code as "the ll error is
# beta-suppressed". That reasoning covers the WITHIN-rung accept test, where
# the error enters as beta*eps -- but NOT the tempering swap, where it enters
# as (beta_i - beta_j)*eps. On a geometric ladder from 1.0 to 1e-4 over 24
# rungs the adjacent ratio is 0.687, so beta_i - beta_j = 0.313*beta_i, and a
# stale-reference error of 1e3 lnL (the tail the probes measured off the cold
# chain) contributes ~31 at beta=0.1 and ~3 at beta=0.01. Swaps at those
# rungs would be decided by reference staleness rather than by the data.
# Only around beta ~ 1e-4 does it genuinely vanish (~0.03).
#
# Cost: this refreshes every rung instead of the ~top third, so roughly 3x
# the measured refresh time -- ~0.9% of a propose typically, ~17% in the
# heaviest propose observed. Against rj_step at 87% of the propose that is
# ~12% wall clock worst case, and it buys swap ratios that mean something.
export GB_SIGHET_REFRESH_MIN_BETA=0
#
# SNR-SCALED TRUST GATE (measured 2026-08-18, high-f probe A/B). The uniform
# 0.5 rad gate is the wrong SHAPE: the tiered spec places gates at a constant
# TRUE-lnL displacement T, but a fixed phase offset sits at
# T = 0.5*(dphase*SNR/3.456)**2 -- T~0.7 at SNR 8 against a design point of
# T~1000, while being ~9 sigma for a loud source. It strangled exactly the
# faint population the completeness deficit lives in.
# C_phase = 3.456*sqrt(2*T_gate); 49 -> T=100. Clipped BELOW by
# sighet_trust_dphase, so this can never tighten the gate for anyone.
#
# A/B result, same nt_layer, only the gate differing:
#   C_phase=0   gate=[0.5..0.5] rad    -> [GB_TRUST] 13.8-23.8% rejected,
#                                          infomat cold acceptance 0.323
#   C_phase=49  gate=[0.81..9.81] rad  -> [GB_TRUST] 2.7-3.6% rejected,
#                                          infomat cold acceptance 0.404
# The gate stops being an active constraint (5-6x fewer kills) and becomes
# the rare safety net it was meant to be, and the per-walker acceptance line
# goes from many nan walkers (no in-model proposals reaching them at all) to
# full coverage. Cost: the chain travels further from its reference, so the
# end-of-block DELTA-vs-DELTA on the loud block ran 8.46 against 3.65-7.06
# elsewhere -- ~+20%, at the high end of the observed range but inside it.
# Worth it: the accumulated error affects bookkeeping and swaps; the
# rejection rate affects whether the chain moves at all.
#
# Also re-couples the refresh: with the gate up to ~10 rad, drift CAN now
# exceed the stock refresh trigger, so the two knobs stop being mutually
# exclusive (moot here -- REFRESH_DPHASE=0 above -- but it matters elsewhere).
export GB_SIGHET_TRUST_PHASE_C=49
#
# RUNG-COVERAGE AUDIT -- ARMED FOR THE FIRST FEW ITERATIONS, THEN REMOVE.
# Every probe today ran the degenerate 2-rung ladder, so we have NO data on
# 22 of this run's 24 rungs. The probes' delta-vs-delta line showed a tail of
# 1e3-1e4 lnL error OFF the cold chain (cold maxima stayed ~1-17). At
# beta=1e-4 that is suppressed; on this run's geometric ladder the middle
# rungs sit at beta ~ 0.01-0.1, where 1e3 becomes beta*eps ~ 10-100 and would
# corrupt the tempering swap ratio (beta_i - beta_j)(L_i - L_j).
# These two make DELTA-vs-DELTA report across ALL 24 rungs. Watch the "all"
# median: if it stays ~1 with isolated maxima, unset both and carry on; if it
# climbs with rung count, stop and investigate before spending days on it.
# Cost: one extra exact batched call per in-model block (measured 0.053 s
# against inmodel_repeats ~4-5 s).
# SIG-HET ANCHOR AUDIT RETIRED (2026-09-04 gate PASSED on the 3mo arm:
# all-rung delta-vs-delta p50 0.054, cold max 5.48, well inside tolerance
# over the whole run; ~18 s/iter while armed). Re-arm (=1) if a sig-het
# accuracy question reopens. Ported from the 3mo 10-walker arm.
export GB_SIGHET_ANCHOR_CHECK=${GB_SIGHET_ANCHOR_CHECK:-0}
export GB_SIGHET_DRIFT_CHECK=1
# TIER SCAN RETIRED FOR THE CLEAN RESTART (2026-08-19). It has NO iteration
# cap (the "first-few-iterations" note above it was wrong): it ran 13 extra
# scoring passes -- half of them chunked-exact -- on EVERY in-model block,
# ~0.32 s/propose, for the whole run. Its 72-block dissect record is
# captured and analyzed (see GB_SIGHET_DISSECT below); the clean production
# reading must not carry its overhead. The anchor check above stays: one
# cheap exact call per block, logging |dll@anchor| -- the corrupted-refs
# rate stays on the record for the post-fix comparison.
export GB_SIGHET_TIER_SCAN=""

# ============================================================================
# SIG-HET DISSECTION + IN-RUN ENGINE SWEEP (2026-08-19, LAT 749af2e1).
# Motivated by the 13 anchor checks in this run's own log: ll_het ~ -8e3 vs
# ll_exact ~ +3e2 AT THE EXPANSION POINT (r=1, frozen residual) for the SAME
# recurring sources (band 10 @ 1.9727 mHz x6, band 17 @ 2.9603 x3, band 5 @
# 1.3181 x2) -- corrupted references, not accuracy noise, and nothing a
# resolution knob can touch. Both riders live inside the tier scan above.
#
# DISSECT: one npz per in-model block (first 32) -- the anchor through BOTH
# engines with the d_h/h_h split (data-side vs template-side attribution),
# null-depth/masked-row stats from the engine's own c0 stash, and full
# per-source identity. ~1 extra batched call per block.
#
# SWEEP: at the first 2 blocks, every arm below is rebuilt around the SAME
# underlying chunked comp, re-anchored on the SAME frozen residual and
# scored on the SAME 512-source subset (worst anchor offenders + random
# fill -- the band-10 population is guaranteed in-sample) against ONE shared
# exact side. Arms differ by exactly one thing: the engine config. Base
# (production) config is auto-prepended as the control. A failed arm logs
# and the loop continues; the production engine is restored no matter what,
# and the run then continues normally. Budget: ~10 arms x (make_reference
# on 512 refs + 7 batched scores) -- minutes, twice.
#
# Read the verdict locally once "[GB_SWEEP] wrote" appears (the dumps ride
# home in the store-dir zip):
#   python scripts/gb_chunked_het/gb_sighet_dissect_report.py \
#       <unzipped>/gf_prod_3mo_v4/dissect
# Anchor |dll| CANNOT move under a resolution-only knob: flat across the nt
# arms + moving under node arms = deep-null node fit; flat across ALL arms
# = the reference build itself (then the dissect d_h/h_h split names which
# half). v5=0 differing from base = v5-specific; v5=2 is the flat-carve
# control arm.
# Sig-het reference-build taper, PINNED (2026-08-19 Tukey rulings: equal
# alphas across chunked/sig-het/TD->FD; error-created edges REMOVED by the
# [min_time, max_time] crop). 0.01 -> taper 11 layers + 8 margin = 19 <=
# crop 20: the build-time edge-exclusion guard passes with ONE layer to
# spare. The old inherited 0.05 tapered 54 layers against the 20-layer
# crop -- the flat ~1% h_h bias that was the dissect's high-f 0.984,
# fixed in GBGPU 88b278d / LAT 7e9c4c65+454d04bd. Verify on resume:
#   grep "sig-het engine resolved" ...  (tukey_alpha=0.01)
# and the anchor/AUDIT high-f values should move 0.984 -> ~1.000.
export SIGHET_TUKEY_ALPHA=0.01
# THE LOW-F h_h CORRUPTION FIX (2026-08-19, root-caused on the laptop from
# this run's own dissect captures). Mechanism: the make_reference spline
# reconstruction (n_cp_build control points) matches each channel to ~0.1%
# but its per-channel errors are INCOHERENT across X/Y/Z, so the GW
# template's X+Y+Z null cancellation (true null power ~1e-10 of total) is
# broken at ~1e-5 -- and the near-singular low-f XYZ invC amplifies
# exactly that direction (null eigenvalue 54-7500x the differential ones).
# Result: anchor h_h inflated up to 27x (d_h CLEAN -- the measured v4
# signature), worst for edge-on sources at low f. The AUTO n_cp law
# (4-day spacing -> 32 nodes at 3 months) was set by a phase criterion
# that never saw the null direction. Verified on the production grid
# (gb_sighet_bfold_gpu_probe.py): 32 -> 256 nodes takes the scored anchor
# from max |log hh ratio| 0.31 to 1.5e-4 at +2.6% setup cost. 256 = the
# shared-arena ceiling; pinned explicitly (the GBGPU AUTO default now
# also resolves here, this is the belt to that suspenders).
export SIGHET_N_CP=256
# UNIFORM EDGE EXCLUSION (user ruling 2026-08-19): bringing the domain
# [min_time, max_time] in removes edge-created error in EVERY likelihood
# at once (all WDMSettings inherit the one domain; min/max_freq still vary
# per source). NOTE (capture-replay verdict, same day): the crop is NOT
# the cure for the low-f h_h inflation -- the fresh reference build is
# EXACT on this very crop-20 domain; the inflation is live-stash
# corruption (see the dissect block below). The crop remains the POLICY
# knob (taper must be subsumed; constant-layers scaling, pinned in the
# 6-mo/23-mo scripts) -- flip on a FRESH STORE_DIR only: changing the
# crop changes Nt_active and resume/rewind compatibility across a
# domain-shape change is UNVERIFIED.
# PER RUN (runs.py). The sig-het taper is Tobs-scaled -- the build guard
# (check_sighet_build_config) needs crop >= ceil(alpha/2 x Nt) + 8 =
# 19 / 30 / 41 / 52 / 73 / 95 layers at 3 / 6 / 9 / 12 / 18 / 24 mo, and the
# code default (20) only covers 3 mo. 60 is the validated 6 mo / 9 mo / 1 yr
# value (30 / 19 / 8 layers spare); 18 mo takes 90 and 24 mo 120 (the same
# ~1.4 % of the data cropped as at 1 yr). Fresh store only: the crop sets
# Nt_active, and a domain-shape change across a resume is unverified.
export EDGE_CROP_WAVELETS=120

# DISSECT + RAW CAPTURE RETIRED (2026-08-19, mission accomplished). The
# 72-block dissect + 25 raw-slab captures were pulled and replayed locally:
# a FRESH setup_in_model from the captured params/slabs scores EXACT
# (sig-het == direct pixel sum == production's exact side), while the LIVE
# in-run stash scored the same sources 10-35x inflated in h_h with
# wrong-signed d_h. Verdict: the reference BUILD and both engines are
# exonerated; the corruption is in the LIVE slot->reference stash lifecycle
# (prime suspect: multi-GPU router/replica sync -- the [1b] shard-swap
# anomaly), amplified at low f by the near-singular XYZ invC (null-space
# eigenvalue 54x the differential ones). Anchor errors reach the cold chain
# (median |dll| 5.5, low-f 23) but sampler-facing DELTAS track exact
# proportionally (multiplicative distortion): clean sources in spec,
# ~6% corrupted-anchor sources 0.3-2 lnL at 1-2 rad displacement. Same
# condition v3 sampled under -- newly measured, not newly introduced.
# Re-arm all three (TIER_SCAN hosts them) only to re-verify after the
# stash-lifecycle fix lands. Analysis: scratchpad replay_raw.py + the
# dissect report; capture data archived off-cluster.
export GB_SIGHET_DISSECT=""
export GB_SIGHET_DISSECT_RAW=0
# SWEEP RETIRED (2026-08-19, after 8 swept blocks): every arm answered.
# nt_layer=270 differs by 6e-8 (round-off -- resolution DEAD); m_half by
# 1e-4/1e-10 (m-window irrelevant); v5=0/v5=2 bitwise (the v4/v5
# bit-identity holding); c_Nt_sub/c_N_cp_sig bitwise (never enter the
# reference build); n_sparse_fd=2048 device-clamped; n_sparse_fd=512
# negligible; c_N_sparse=512 anomalous (uniform 2x -- suspected
# delegate-rebuild side effect in the sweep harness, NOT an engine
# result). The flat ~1% component was root-caused OFF-engine (tukey
# semantics, fixed); the remaining low-f inflation is being chased by the
# CUDA probes + the dissect below, which stays ON.
export GB_SIGHET_SWEEP=""
export GB_SIGHET_SWEEP_F0="1.9727e-3,2.9603e-3,1.3181e-3,1.6357e-3,2.2107e-3,4.2389e-3,5.1677e-3,1.2269e-2,2.0381e-2"
# 4 qualifying blocks: the low-f targets cluster in nearby bands (one or
# two units) while the high-f pair lives in different units entirely.
export GB_SIGHET_SWEEP_BLOCKS=4
# defaults, pinned for the run record:
export GB_SIGHET_DISSECT_MAX=32
export GB_SIGHET_SWEEP_MAX_SRC=512

export SIGHET_INFOMAT=1
export GB_INFOMAT_PER_BLOCK=1
# ---- LOOKUP-BUILT SIG-HET + CACHED PROPOSAL FACTORS (2026-10-05) --------
# Six knobs; the first four differ from the CODE defaults (fd / chunked /
# unset / 0), EVERY and TOL pin the code defaults. All six yield to the
# launch line. Map of every eigen / information-matrix route:
# docs/eigen-info-routes.md.
#
# SIGHET_REF_BUILD=lookup (Settings field sighet_ref_build; code default
#   "fd"). The sig-het v5 carrier reference (c0 / c1) is read from the GB
#   WDM lookup table instead of an FD transform per source: ~55x cheaper per
#   reference on CPU, scores equal to the FD build to the table's ~1e-6.
#   Attaches the lookup table to the sig-het engine
#   (GB_LOOKUP_TABLE_PATH / GB_LOOKUP_TABLE_DIR, built once under a lock),
#   which the two knobs below need. Needs GBGPU >= 84f06bf.
# SIGHET_ANCHOR_ENGINE=lookup (Settings field sighet_anchor_engine; code
#   default "chunked"). The sig-het anchor from the same table, ~10x cheaper
#   per reference. REFUSES at build without SIGHET_REF_BUILD=lookup.
# SIGHET_INFOMAT_ENGINE=lookup (GBGPU env, read per call; code default
#   unset). The GB in-model proposal factor becomes the Gram / expected
#   Fisher <dh_a|dh_b> from 2 lookup fills per parameter (GBGPU
#   GBLookupComputations.information_matrix; f0 / fdot steps 1e-4/Tobs and
#   1e-4/Tobs^2) instead of the sig-het second differences of lnL
#   (SIGHET_INFOMAT=1 above). No in-model block or buffer slot needed, which
#   is what lets GB_CHOL_CACHE build every alive source in one batch. Needs
#   the attached table; GBGPU >= 376c94f on GPU (with f8073ce alone the
#   first cache refresh raised on cupy and the cache disabled itself, job
#   735). NOTE the `-` (not `:-`) expansion: a set-EMPTY
#   SIGHET_INFOMAT_ENGINE= on the launch line is how to turn it OFF.
# GB_CHOL_CACHE=1 (code default 0). In-model proposal factors (and the
#   observable-eigen Gamma_z) live in HOST memory, one cache per rank and
#   branch shared by every GB move (src/lisatools/globalfit/moves/
#   gb_chol_cache.py), each entry mapped to its living source: same WALKER
#   (never rung -- a vertical swap only relabels it), best of the
#   GB_CHOL_CACHE_WINDOW (default 32) f0-neighbours each side, sampling
#   columns 0-2 (amplitude / f0 / Mc) within GB_CHOL_CACHE_TOL of the entry's
#   own marginal widths; a hit moves the entry's key with the source, and
#   keys are re-tracked at each block's end. Births and unmatched sources
#   are computed per block, in one batch before the repeats. The factor is
#   still frozen across each block (symmetric proposal). GB only (VGB keeps
#   the per-block factors).
# GB_CHOL_CACHE_EVERY=40. ALL alive sources are rebuilt together (in
#   GB_CHOL_CACHE_BATCH, default 4096, chunks) every 40 proposes on a
#   global ticker -- the largest move.time -- i.e. ~1000 in-model steps at
#   25 repeats.
# GB_CHOL_CACHE_TOL=5. Match tolerance, in marginal widths.
# Job 738 (all of it on): misses ~1.5-2 %, inmodel_cholesky 53 -> ~5 s per
# propose, iteration 6.6 -> 5.5 min.
# WATCH: "[GB_CHOL_CACHE] refreshed N sources in X s" (per refresh), the
# "... hits / ... misses" lines, and "DISABLED" -- any error disables the
# cache AND unsets SIGHET_INFOMAT_ENGINE, so the run continues on the old
# direct per-block factors. Back out on the launch line with
# GB_CHOL_CACHE=0 SIGHET_INFOMAT_ENGINE=
export SIGHET_REF_BUILD=${SIGHET_REF_BUILD:-lookup}
export SIGHET_ANCHOR_ENGINE=${SIGHET_ANCHOR_ENGINE:-lookup}
export SIGHET_INFOMAT_ENGINE=${SIGHET_INFOMAT_ENGINE-lookup}
# GB_LIKELIHOOD=lookup (Settings field likelihood, GB and VGB blocks; built by
#   erebor.gb.build_gb_wdm_comp): the per-row GB likelihood -- RJ births /
#   removals and every score outside an in-model block -- runs GBGPU's fused
#   direct-to-WDM lookup scorer (GBLookupWDMComputations) instead of the
#   chunked-het kernel. Fills, swaps, the F-stat and the sig-het delegate stay
#   chunked-het. H100 harness (2026-10-05): 4.4 / 7.0 / 19 us per row at 180 /
#   360 / 720 d vs chunked 88 / 172 / 345; median 1-3e-5 nats, max 5e-3 nats
#   from the chunked value. Reads the GB lookup table of SIGHET_REF_BUILD=lookup.
#   WATCH: "GB per-row likelihood: LOOKUP" and "VGB per-row likelihood: LOOKUP"
#   at build. Back out on the launch line with GB_LIKELIHOOD=chunked.
# GB_LOOKUP_FILL=1 (field lookup_fill): with the lookup likelihood, the adds
#   and removals write the LOOKUP template into the residual too (GBGPU
#   gb_lookup_fill), so every template in the residual is the model the
#   likelihood scored; swaps and the F-stat grid scoring stay chunked-het
#   (no lookup kernel). The build line says "fills LOOKUP". Inert with
#   GB_LIKELIHOOD=chunked.
export GB_LIKELIHOOD=${GB_LIKELIHOOD:-lookup}
export GB_LOOKUP_FILL=${GB_LOOKUP_FILL:-1}
export GB_CHOL_CACHE=${GB_CHOL_CACHE:-1}
export GB_CHOL_CACHE_EVERY=${GB_CHOL_CACHE_EVERY:-40}
export GB_CHOL_CACHE_TOL=${GB_CHOL_CACHE_TOL:-5}
# Countable-only F-stat center precompute + lookup-miss fallback rides
# the pull (no knob beyond the existing GB_RJ_FSTAT_CTR_HOIST=1); the
# new [FSTAT_CTR] census line diagnoses the job-195 5x centers blowup.
# Bilinearity bookkeeping monitor (code default ON, user ruling: ~1.5 s
# per propose = negligible): per-unit [GB_ORTHO_LL] line compares the
# realized cold parent-residual delta against the summed per-buffer
# deltas; WARNs above GB_ORTHO_LL_TOL (0.05). (Stale note removed 2026-09-02: the [GB_ORTHO]
# premise check is ON below -- GB_ORTHO_CHECK=1 -- and had its first
# GPU validation in the r2 probes: 332/332 measured, max boundary
# overlap 2.95e-04, premise sound.)
export GB_ORTHO_LL_CHECK=1
# Cap-cell grid (user design 2026-08-15): leaf caps on a band/8 grid at
# the confusion scale; scheduling unchanged. RESUME REQUIRES migration
# step 3 in the header checklist. WATCH first propose: leaf growth +
# memory (the band-total throttle is gone -- a band can now reach
# K*cap); GB_CAP_DIVISOR=1 reverts instantly.
# V3: 32, not 8. A 17.36 uHz cell (K=8) is 135 FD bins -- far wider than
# the resolution here -- so 47 cells hold 3-5 separable detectable sources
# against a cap of 2 and 170 of 694 detectable sources (24.5%) are
# structurally unrepresentable. K=32 gives a 4.34 uHz / 34-bin cell and
# drops that to 2.6%, with 456 of 566 occupied cells holding exactly ONE
# source. 32 is also the FLOOR: it is the last divisor whose cell still
# spans this run's own observed duplicate-parking distance of ~1 Doppler
# width (15.5 FD bins at 20 mHz); at K=64 the cell is 1.1 Doppler widths
# there and parked duplicates escape into the neighbouring cell.
# V6: SUB-BAND SHRINKAGE (the run's ONE variable; see the V6 header).
# Uniform layer/8 bands: 135 FD bins each, 1232 bands. The startup
# separation diagnostic logs the conservative-envelope verdict; the
# operative independence monitor is [GB_ORTHO_LL] (default ON) exactly
# as on the production 1-layer grid. GB_SUBBAND_DIVISOR=1 reverts.
export GB_SUBBAND_DIVISOR=8
# Stride 9 = same-unit gap of exactly ONE LAYER, mimicking production's
# stride-2-on-1-layer separation. ~137 concurrent bands/unit (v5: 77);
# 9 units per pass instead of 2.
export GB_BAND_UNIT_STRIDE=9
# PER-WALKER BAND-CLASS ROTATION (aligned with v7 2026-08-29). The stride and
# class membership are UNCHANGED -- band b stays in class b % 9 for every
# walker and band_edges stays one global array, so band b means the same Hz
# everywhere. Only the ORDER changes: each walker gets its own random START
# class and its own +/-1 cycle DIRECTION, then visits the classes in order
# from there. Every walker still covers all 9 classes per sweep (gcd(1,9)=1),
# so it is a rotation, never a permutation -- a walker's concurrently-open
# bands stay one residue class apart and the orthogonality argument is
# untouched (it is a per-walker property: cells of different walkers write to
# disjoint parent rows).
#
# APPLIES IN BOTH SEARCH AND PE (user ruling), so the detailed-balance safety
# is load-bearing rather than a search-stage convenience: both draws are
# UNIFORM and STATE-INDEPENDENT, drawn from model.random so they stay
# seed-reproducible, and run_proposal asserts the per-walker partition every
# propose and refuses to sample if it breaks. NEVER weaken the draw into a
# heuristic ("which walker looks stuck", by logL, by occupancy) -- that
# silently converts a DB-safe change into a DB-breaking one.
#
# ARMED AHEAD OF v7 EVIDENCE (user decision 2026-08-29: "assume it will work
# and it will help or be neutral at worst"). v7 is running these now; if its
# logs show a problem, set both to 0 -- knob-OFF is bit-identical to the
# single global start by construction, not by coincidence. Grep [GB_UNIT_SCAN]
# for the schedule actually used.
export GB_BAND_UNIT_START_PER_WALKER=1
export GB_BAND_UNIT_DIR_PER_WALKER=1
# VGB DE-COUPLED from the fine grid (2026-08-22 timing autopsy): the VGB
# branch inherits GB_SUBBAND_DIVISOR through GBSetup.init_band_structure,
# so v6 silently ran the ~30-source VGB move on 1232 narrow bands --
# vgb_pe 34 s/propose vs v5's 8.5 (run_tempering 25 s; fill_slots 55,920
# vs 4,800). VGB_BAND_LAYERS=8 merges 8 fine bands back to the 1-layer
# separations, restoring v5's VGB geometry (VGB has no RJ surface; its
# per-band arrays migrate on the resume that picks this up). Recovers
# ~4.3 min of the 18.3-min iteration.
export VGB_BAND_LAYERS=8
# STAGING BATCH CAP (2026-08-23, full_pe OOM autopsy): v6 died at it=173
# on a cupy OOM (3.09 GB _expand_B request, gpu0 93.06/93.6 GB) during
# rj_prior_pe's sig-het setup -- this script never carried the cap the
# v5/1-yr scripts got, and full_pe's picked pools (779 leaves/walker,
# 1230/1232 bands occupied) finally outgrew the card.
# MEMORY-FOR-SPEED (user ruling 2026-08-27, snapshot-2 telemetry: GPU
# peaks 45.3/18.2 GB of 99.9 -- 50-68 GB idle in gb_search): batch cap
# raised 1024->2048 (halves the sig-het staging setups, the ~2.3 s
# spikes per sub-block) and BOTH mempool sweeps disabled (stop paying
# free/realloc cycles the headroom does not require).
# *** REVERT AT FULL_PE HANDOFF if memory approaches the card: the v6
# OOM above happened at full_pe occupancy (1230/1232 bands); on any
# resume into full_pe set GB_INMODEL_SETUP_BATCH=1024 and both
# *_MEMPOOL_FREE=1 unless telemetry shows margin. ***
# 4096 (2026-08-27, batch-width autopsy): the in-model REPEAT TRAINS
# run at exactly this width (log: "repeats x 1024/2048 sources" tracks
# the cap), and each step's wall is launch-bound (~dozens of kernel
# launches over ~14 ms of physics at width 2048) -- doubling the width
# halves the number of 100/250-step trains and with them the
# launch-train overhead. Same full_pe revert rule as above.
# ONE-BLOCK STAGING (2026-09-11, ported from the 10w arm). The old
# "SETUP_BATCH=2048, do NOT raise" sizing predates the WINDOWED sig-het
# stash (faadf82: stash bytes go per-source window, not per-band extent)
# and the PSD mirror; the 10w arm runs SETUP_BATCH=0 (whole pool, column-
# atomic per b3f5acbd) with the fold cap below. SAFE ONLY on b3f5acbd+
# code: one-block staging is what raised vertical-pair co-residency, and
# the fixed whole-cell swap ratio is what makes that correct.
export GB_SIGHET_INMODEL_WINDOWED=1
# ######################################################################### #
# ## OOM 2026-09-18 -- THE REVERT ABOVE, APPLIED EARLY.                  ## #
# ##                                                                     ## #
# ## The run died in gb_search (NOT full_pe) at ~4288 leaves:            ## #
# ##   setup_in_model -> gbsignalhetcomputations.py:1074 bin_fold_real   ## #
# ##   -> signal_het.py:119  En = c0[...,:,None,:,:]*iC*c0[...,None,:,:] ## #
# ##   OutOfMemoryError: 2,782,742,528 bytes (89,586,791,936 allocated)  ## #
# ##                                                                     ## #
# ## gbsignalhetcomputations.py:1047 sizes each fold chunk to FILL the   ## #
# ## byte cap, so an 8 GiB cap builds an ~8 GiB transient by design:     ## #
# ##   per_src = 2*nch*nch*W*Nt_active*16 (Ec+En) + nch*W*Nt_active*16   ## #
# ##   chunk   = _SIGHET_FOLD_MAX_BYTES // per_src                       ## #
# ## The CODE default is 1 GiB (1<<30); 8 GiB was this file's override,  ## #
# ## and one-block staging (SETUP_BATCH=0) sat on top of it. That is     ## #
# ## ~8 GiB of transient against a card already 89.6 GB resident.        ## #
# ##                                                                     ## #
# ## The block above says to revert AT THE FULL_PE HANDOFF. The 6-month  ## #
# ## run reaches the danger zone EARLIER than that -- in gb_search, as   ## #
# ## the leaf count climbs -- so the revert is now the default here.     ## #
# ## This file's own header (lines ~15-17) already calls the calibrated  ## #
# ## 6-month sizing GB_N_SUBBANDS=4096/GPU and SETUP_BATCH=2048; the     ## #
# ## live values had drifted to 8192 and 0.                              ## #
# ##                                                                     ## #
# ## All four are TRANSIENT/scheduling knobs: no stored number changes,  ## #
# ## so a mid-store resume is safe. Cost is more chunks per unit (launch ## #
# ## overhead), not accuracy. Every one is env-overridable, so the old   ## #
# ## aggressive sizing is one export away if telemetry shows margin.     ## #
# ##                                                                     ## #
# ## STILL TIGHT? Next lever is GB_N_SUBBANDS 8192 -> 4096 (halves the   ## #
# ## slab AND the active-slot count; the header's calibrated value).     ## #
# ## NEVER reach for SIGHET_NT_LAYER: not a mid-store knob, and it       ## #
# ## MULTIPLIES with GB_N_SUBBANDS in the sig-het byte product.          ## #
# ######################################################################### #
export GB_INMODEL_SETUP_BATCH=${GB_INMODEL_SETUP_BATCH:-2048}
export GB_SIGHET_FOLD_MAX_BYTES=${GB_SIGHET_FOLD_MAX_BYTES:-1073741824}
export GB_RJ_INMODEL_CHUNK=8192  # per run (runs.py), byte parity across Tobs: 65536 at 3 mo / 32768 at 6 mo / 16384 at 9 mo + 1 yr (the 1yr value, ruling 2026-10-07) / 8192 at 18 mo + 2 yr; floored to ntemps multiples by the column-atomic staging
export GB_INFOMAT_MEMPOOL_FREE=${GB_INFOMAT_MEMPOOL_FREE:-1}
export GB_INMODEL_BATCH_MEMPOOL_FREE=${GB_INMODEL_BATCH_MEMPOOL_FREE:-1}
# ######################################################################### #
# ## SEAM-STRADDLING CAP CELLS (divisor 2 + stagger, 2026-08-29).        ## #
# ## ⚠ DO NOT "FIX" THE CAP GRID BACK INTO ALIGNMENT WITH THE SUB-BANDS. ## #
# ######################################################################### #
# This REPLACES the 2026-08-26 aligned-cells ruling (divisor 1, stagger 0)
# on direct measurement. Read this before touching GB_CAP_DIVISOR or
# GB_CAP_STAGGER -- alignment looks tidier and is exactly wrong.
#
# THE FINDING. The 3-month v7 run has a persistent bimodality at the
# flagship 20.380377 mHz: leaves split across the band 1141/1142 seam
# (+12.19 bins from the flagship) with an EMPTY GAP between the two
# populations. The dedicated high-f probe SOLVED this on the same band,
# and the probe's BAND grid is bit-identical to v7's -- so the band grid
# is not the discriminator. THE CAP GRID IS. Cap-cell membership of the
# actual flagship leaves at v7 row 5, under both geometries:
#
#   v7    (divisor=1, stagger=0):
#      cell 1141  [-122.8, +12.2] bins   below-seam 24   above-seam  0
#      cell 1142  [ +12.2, +147.2] bins  below-seam  0   above-seam 22
#
#   probe (divisor=2, stagger=1):
#      cell 2284  [ -21.6,  +45.9] bins  below-seam 24   above-seam 22
#                                        ^^^ SPANS THE SEAM
#
# WHY IT MATTERS. Under the aligned grid each band carries an INDEPENDENT
# cap, so the two modes never compete and eight leaves can sit across the
# seam forever. Under the staggered grid BOTH modes fall in ONE cell and
# compete for ONE cap, which gives the RJ death move direct pressure to
# kill the weaker side. This compounds with the block-Gibbs scan: bands
# 1141/1142 are residue classes 7 and 8 mod GB_BAND_UNIT_STRIDE=9 and are
# therefore NEVER co-open, so no single move can even see both modes at
# once. The straddling cell is what reaches across that.
#
# make_cap_edges' own docstring states the intent: stagger exists so "no
# source can sit on a band seam and a cap seam simultaneously", and the
# cell at index b*K "physically STRADDLES the band-(b-1)/b boundary".
#
# GEOMETRY AT K=2. Sub-bands are 135 FD bins (GB_SUBBAND_DIVISOR=8), so
# cells are 67.5 bins. Staggered edges land at the 1/4 and 3/4 points of
# each band, so every band owns one INTERIOR centre cell and SHARES two
# straddling cells with its neighbours. Every cap-cell boundary is
# mid-band; NO cap edge sits on a band seam. 2464 cells, not 1232.
#
# PINNED BY tests/test_cap_stagger.py::SeamStraddlingCellTest, which
# asserts that at (2, 1) two leaves either side of a band seam map to the
# SAME cell index and at (1, 0) they do not. If you change this, that
# test fails and tells you why.
#
# ⚠ CAP EDGES CHANGE -> the resume guard
# (GBState.initialize_band_information) refuses a v7/v8-lineage store. See
# the relaunch block at the top of this file: rewind GB to empty, then
# migrate_gb_cap_grid.py --cap-divisor 2 --stagger.
export GB_CAP_DIVISOR=1
# 2 -> 1 (user design 2026-08-29). WITH GB_CAP_STAGGER=1 this is the
# MIDPOINT-TO-MIDPOINT grid: 1232 cap cells for 1232 sub-bands, each
# interior cell running from the midpoint of one sub-band to the midpoint
# of the next, so the seam sits at the CENTRE of a cell and the two sides
# of a seam compete for ONE cap. Verified against the real 1232-band grid:
# interior cap edges == band midpoints exactly; widths half / 1.7361e-5 Hz
# (= one full sub-band) / 1.5x at the two ends.
#
# WHY NOT 2 (what the 2026-08-29 restart ran): divisor 2 makes cells HALF
# a sub-band, which (a) halves the straddle reach to +/-0.25 sub-band, so
# a seam-straddling pair further apart falls out of the shared cell,
# (b) doubles the cells, halving per-cell occupancy and delaying at-cap
# birth-row exclusion -- measured +39% F-stat candidate rows at matched
# iteration -- and (c) gives each band TWO owned cells, which is what let
# a birth into an already-full straddling cell slip past the old
# band-saturation gate (4 of 24 walkers held 2 leaves in a cap-1 cell at
# rows 5-6 while no sub-band ever held more than one).
#
# REBUILD: not required for THIS config (the CUDA cell lookup is reached
# only via the fused in-model path, and GB_INMODEL_ACCEPT_KERNEL=0 here),
# but rebuild on relaunch anyway so python and kernel stay in step -- the
# disagreement would be silent if anyone ever set that knob to 1.
# ############################################################################
# ## V7 (2026-08-24) = V6 + OVERLAPPING CAP CELLS, nothing else.            ##
# ## GB_CAP_OVERLAP_FRAC=0.25 widens each cap cell's enforcement span on   ##
# ## the SAME staggered edge grid (width 45 bins = 11.25 shared | 22.5     ##
# ## core | 11.25 shared) so any two leaves within 11.25 bins share a      ##
# ## covering cap-1 cell -- the anti-split-source experiment, now at        ##
# ## production scale (motivation: the flagship's persistent -5.7-bin      ##
# ## cap-edge mode; see the confined highf probes). Edges unchanged ->     ##
# ## resume guard passes on a rewound v6-lineage store.                    ##
# ## ALSO ON (user, 2026-08-24): rj_replace (GB_SEARCH_RJ_REPLACE=1,     ##
# ##   the code default made explicit) -- exact-MH F-stat replacement      ##
# ##   after rj_fstat_search in the search cycle. NOTE: the confined       ##
# ##   probe measured ~0 cold acceptance for this move (forensics in      ##
# ##   flight); at worst it costs wall time, it cannot bias (exact MH).   ##
# ##   Resume-safe: the store guard asserts stage NAMES/order only.       ##
# ## EXCLUDED for now: the core-dominant divisor-2 geometry (probe-only). ##
# ## START (v6 pattern): cp -r gf_prod_3mo_v6 gf_prod_3mo_v7 &&           ##
# ##   python scripts/fstat_proposal/reset_recipe_stage.py \              ##
# ##     gf_prod_3mo_v7/gf_prod_3mo_testing.h5 gb_search \                ##
# ##     --rewind-to-empty gb --apply                                      ##
# ## (inherits fitted noise + VGB ladder + fstat epoch cache; only the GB  ##
# ## search reruns under overlap enforcement). Fresh dir works too.        ##
# ############################################################################
# CAP CELLS EXACTLY == SUB-BANDS (user ruling, aligned with v7 2026-08-29):
# 0.25 -> 0. With GB_CAP_DIVISOR=1 and GB_CAP_STAGGER=0 the cap edges are
# already bit-identical to the band edges; dropping the overlap removes the
# widened ENFORCEMENT SPAN so a cap cell is exactly its sub-band, with no lip.
#
# WHY: the overlap makes "at cap" an OR over covering cells, so a leaf near a
# band edge is charged against BOTH neighbours' budgets. Harmless when the cap
# is slack -- but measured on the v7 store the cap BINDS: at the flagship bands
# 1141/1142, max-over-walkers occupancy reached the cap in 55 of 104
# (band, iteration) pairs = 53%. With a binding cap the overlap obstructs the
# cross-edge in-model movement the headroom below is meant to allow: a leaf
# could be vetoed because EITHER side was full, not because its destination
# was. Removing the lip makes each leaf count only where it actually is.
#
# Resume-safe, no migration: GBState.static_names is only
# ("band_edges", "cap_edges"), and make_cap_edge_extensions (state.py:107) is
# computed at runtime rather than persisted, so this touches no stored array.
#
# ⚠ STAYS 0 UNDER THE STAGGERED GRID (user ruling 2026-08-29). The probe
# whose geometry we are adopting ran overlap=0.25 as WELL, so this is
# deliberately NOT an exact probe reproduction. Reasoning:
#
#  * Stagger and overlap are THE SAME MECHANISM from opposite sides: both
#    charge a leaf to a cell it does not sit in, creating competition for
#    one cap. Whether that is good depends on WHERE.
#  * At a BAND SEAM competition is what you want, because non-competition
#    there compounds with the block-Gibbs trap (1141/1142 are residues 7
#    and 8 mod 9, never co-open, so no single move sees both modes).
#    STAGGER FIXES EXACTLY THAT CASE, and it is the case with direct
#    evidence -- see the divisor block above.
#  * At a CAP-CELL BOUNDARY MID-BAND competition is worth much less: those
#    two leaves ARE in the same band, so they are co-open, RJ can propose
#    to both and in-model moves freely. Non-competition there is an
#    inefficiency, not a trap. That is the ONLY extra case overlap covers
#    once staggered -- and it buys it by tightening effective capacity
#    everywhere, against a cap measured as BINDING in 55 of 104
#    (band, iteration) pairs = 53% at the flagship bands.
#  * ATTRIBUTION: one change at a time. Stagger+overlap together and we
#    cannot tell which mattered.
#  * COMPOUNDING: at divisor 2, stagger AND overlap gives 67.5-bin cells
#    widened to 90 with 22.5-bin shared zones -- every leaf near any
#    boundary counted twice, layered on an already-binding cap.
#
# IF STAGGER ALONE DOES NOT CLEAR THE BIMODALITY, restoring
# GB_CAP_OVERLAP_FRAC=0.25 is the obvious next step, and we will then know
# which of the two was load-bearing.
export GB_CAP_OVERLAP_FRAC=0
# NO-OP AT GB_CAP_DIVISOR=2 -- RETAINED DELIBERATELY (verified 2026-08-29).
# The drift gate's short-circuit is
#   (cap_divisor == 1 AND overlap <= 0 AND NOT edge_leak) -> return None
# so at divisor 2 the FIRST clause is already False and the gate is armed
# regardless of this knob. Grep confirms it: `cap_drift_gate_edge_leak` has
# exactly two references in the tree -- its ctor read and that one clause --
# so it can have no other effect here.
#
# KEPT SET ANYWAY because it is only inert while divisor > 1: if anyone
# reverts to GB_CAP_DIVISOR=1, this is what stops cap enforcement on
# cross-edge in-model moves from silently vanishing (the 2026-08-20
# "29 leaves into a cap-1 cell" mode, at the seams).
export GB_CAP_DRIFT_GATE_EDGE_LEAK=0   # V9-4: cap cells off
# N/4 IN-MODEL BAND WINDOW ACTUALLY MEANS N/4 (bug fix, aligned with v7).
# The window was BUILT as band_N_vals * layer_df / 4 Hz (gbbands.py:3397-3401)
# while every consumer divides by the move's df = 1/Tobs -- so it was too wide
# by layer_df*Tobs = Nt/2 = 1080x. Measured: N=256 intended +/-64 bins, actual
# +/-69,120 bins = +/-512 sub-bands, WIDER THAN THE WHOLE 3-21 mHz BAND. The
# per-step leash (|df0| <= N/4 bins) was always unit-correct and is untouched.
# =0 restores the old unbounded window.
export GB_BAND_WINDOW_STRICT=1
# CAP GATE READS THE DESTINATION CELL FROM THE CANDIDATE f0 (bug fix, v7-aligned).
# At cap_divisor == 1 _cap_cell_index returned band_inds and never read f0, so
# current cell == new cell for every row and THE VETO COULD NOT FIRE -- the
# cap+2 destination rule was a tautology. =0 is the escape hatch.
#
# STILL CORRECT AT GB_CAP_DIVISOR=2 + STAGGER=1 (verified 2026-08-29), and the
# earlier "the clip folds an out-of-band proposal back into the leaf's own
# boundary cell" reading DOES NOT APPLY: that clip lives in the NESTED branch
# only. The staggered branch clips solely to the GLOBAL cell range, so sub may
# run past K-1 (or below 0) and the leaf lands in the neighbouring STRADDLING
# cell -- exactly what the geometry wants. Verified against
# searchsorted(cap_edges, f) over +/- 0.9 band widths.
#
# On our UNIFORM band grid the b*K terms cancel algebraically, so the staggered
# lookup gives the same cell for ANY handed-in band index and this knob is a
# no-op -- EXCEPT exactly on a cap edge, where the cancellation is exact in
# real arithmetic but not in floating point and the two band references can
# round the tie to different sides (measured: 2 of 37 sample points, both on
# edges; resolve_band matched searchsorted at both). Keep it ON: one
# searchsorted, removes the tie ambiguity, and load-bearing on a ragged get_n
# grid where the per-band steps differ.
export GB_CAP_DEST_BAND=1
# rj_replace DISABLED (aligned with v7 2026-08-29). Not earning its ~580 s/row
# (cold acceptance 0.033-0.046%, delta-ll flat at ~103 mean across 55 calls),
# and its lnL ACCOUNTING IS BROKEN: 365 of 905 [GB_ORTHO_LL rj_replace] lines
# (40.3%) breach GB_ORTHO_LL_TOL=0.05, max 6.971e+03 -- four orders worse than
# any other move -- with a drift ledger claiming +9,693 against -117,009
# realized. Chain state stays correct (drift is repaired from the residual) but
# the per-cell lnL the MH ratio prices against can be wrong by thousands of
# nats. Re-enable only after the accounting is fixed and re-audited.
# V9-11 RE-ENABLED (user ruling 2026-09-24, reversing the 08-29 disable):
# "clean up the replace move to modernize ... and add it to each GB search
# 1-3 before the prior move." The accounting complaint above was raised
# against the PRE-2026-08-24 move; that redesign replaced the unattainable
# phase-max credit with rotation-on-accept (the scored rows ARE the final
# rows) and made both sides exact add-deltas. The user's position is that the
# original problems may have been the old GPU setup.
#
# ⚠ IT IS BEING WATCHED, NOT TRUSTED. Per pass the move now prints
#   [GB_REPLACE rj_replace] pass i/N: candidate source = WARM|FSTAT container.
#   [GB_REPLACE rj_replace] pass i/N (src) done: max |dlnL| ... = X
# and WARNS above 1.0e3 -- the recorded retirement signature was a
# propose-level drift of 1.5-1.9e3, three orders above every other move.
# Also watch [GB_ACCEPT replace-split] (per-pass acceptance, gated counts,
# cold-accepted dll mean/max) and [GB_ORTHO_LL rj_replace] against
# GB_ORTHO_LL_TOL. If either fires, set GB_SEARCH_RJ_REPLACE=0 and relaunch;
# nothing else in the cycle depends on it.
# >>> 1 -> 0 (user ruling 2026-09-25, off job 628). NOT because the
# >>> watchdogs above fired -- max |dlnL| stayed 298-661, under the 1.0e3
# >>> warn and well under the 1.5-1.9e3 retirement signature. It is being
# >>> switched off for YIELD: six consecutive zero-yield proposes, 8 swaps
# >>> accepted out of ~707,000 proposals across 3 iterations x 2 passes,
# >>> and ZERO at the cold chain -- dead at every temperature, not just
# >>> cold. It was simultaneously the most expensive move in iteration 3
# >>> (1610 s = 19.4% of the cycle), ~98% of which is in-model polish on
# >>> its survivors that in_model_replace then provides anyway.
# >>> in_model_replace STAYS: it is a plain in-model pass and does not
# >>> depend on the RJ move. Reinstate for gb_search_3, where sources are
# >>> mis-seated rather than missing and a replacement has something to
# >>> beat.
export GB_SEARCH_RJ_REPLACE=0
# ---- in_model_replace retires WITH it (user ruling 2026-09-26) --------
# "in_model_replace we do not need this anymore." It was the polish slot
# for rj_replace's survivors; with that move off it is simply a THIRD pure
# in-model pass over rows in_model and in_model_fstat already polished.
# Measured on job 634's gb_search_1 iteration 1: 1156 s = 14% of the
# iteration, reaching IMGROUP pass 14.
# Pinned explicitly even though the CODE default is now 0, so this run
# cannot silently change meaning if that default is ever revisited.
# Independent of GB_SEARCH_RJ_REPLACE on purpose -- reinstating the RJ
# move for gb_search_3 does not have to drag the polish slot with it.
export GB_SEARCH_IN_MODEL_REPLACE=0
# TWO INTERNAL PASSES per propose() (user ruling 2026-09-24: "one internal
# iteration with the refit/warmstart and one internal iteration of the
# fstat"). Pass 1 draws replacement candidates from the warm-start mixture
# (the SAME container object rj_warm_search births from), pass 2 from the
# F-stat grid; each pass prices both sides of its MH ratio against its own
# container, so the two never mix. =0 leaves replace single-pass (F-stat
# only), which is the pre-2026-09-24 behaviour.
export GB_REPLACE_WARM_PASS=1
# PE replace OFF too (user ruling 2026-09-02). It was default-ON and would
# have been the only replace flavor left; in both r2 probes it registered
# but never exercised a single proposal, and with centering off its
# center-table inputs are gone -- installed-but-inert machinery in a
# production run is surface area with no upside. =1 restores it.
export GB_PE_RJ_REPLACE=0
# ORTHOGONALITY PREMISE MONITOR (v7-aligned). NOT GB_ORTHO_LL_CHECK (the lnL
# bookkeeping reconcile, already on). This measures what the band decomposition
# RESTS on: normalized |<h_i|h_j>| between concurrently-open adjacent-band cold
# sources. 8 pairs per unit at unit close, diagnostic only, never mutates state.
# DISARMED (2026-09-04, ported from the 3mo 10-walker arm): this premise
# check DIES on the 2-GPU path (get_swap_ll shards index a numpy
# data_index with a cupy keep_idx -> guarded TypeError skip, ~166/snapshot
# of pure log spam, never runs). The companion GB_ORTHO_LL_CHECK
# (credited-vs-direct) still runs and stays on. Re-arm =1 only after the
# gb_likelihood.py:907 cupy/numpy fix lands.
export GB_ORTHO_CHECK=0
# STAGGER ON (2026-08-29) -- the half-cell shift is the whole point of
# this configuration; see the GB_CAP_DIVISOR block above for the measured
# cell-membership numbers. It is meaningless without divisor > 1 (the move
# forces it off at K == 1), and the pair (2, 1) is what the high-f probe
# ran when it solved the flagship bimodality.
export GB_CAP_STAGGER=1
# Per-cell cap CEILING + entry headroom + stage hold (probe-validated
# 2026-08-26 set):
#   GB_CAP_CELL_MAX=20    -- belt on the cap updater, sized to the
#                            v6-REALIZED envelope (v6 ran NO ceiling --
#                            the knob postdates it -- and its log shows
#                            34-bin cells reaching cap 5; 4 such cells
#                            per 135-bin sub-band -> up to ~20). The
#                            probes pinned 5, but that was a
#                            SINGLE-SOURCE band; a 135-bin confusion
#                            cell legitimately holds 3-5+ separable
#                            detectable sources (v3 analysis above) and
#                            must be able to ramp past 5. NOTE the
#                            regime tension: the ratchet is
#                            STALL-driven (patience without D/2
#                            improvement WHILE occupied at cap
#                            increments), so loud isolated cells lean
#                            on the drift gate / entry veto / removal /
#                            SNR-8 floor rather than this ceiling; if
#                            flagship-style stacking reappears, a
#                            frequency-dependent ceiling is the next
#                            lever, not a global squeeze.
#   GB_CAP_INMODEL_HEADROOM=2 -- in-model/replace f0 moves may enter a
#                            foreign at-cap cell up to cap+2 (peak
#                            handover across an edge); births still
#                            respect the hard cap.
#   GB_SEARCH_CAP_QUIESCENT=1 -- the nleaves plateau cannot end
#                            gb_search while any engaged cap cell is
#                            still mid-ramp (occupied at cap, below
#                            ceiling): the stage holds for the ramp.
# ---- BOTH RE-EXAMINED AT (divisor=2, stagger=1), 2026-08-29 ----
# GB_CAP_CELL_MAX=20 KEPT. Cells now govern HALF a sub-band (67.5 bins,
#   2464 cells), so 20 is loose -- but it is a BELT against the ratchet,
#   not a design target, and it is what the probe effectively ran: the
#   probe set no GB_CAP_CELL_MAX at all, so its ceiling fell back to
#   nleaves_max, which the probe pinned at 20. Same effective per-cell
#   ceiling, reached two different ways. Lowering it now would be a second
#   simultaneous change with no evidence behind it.
# GB_CAP_INMODEL_HEADROOM=2 KEPT, but note its SCOPE SHRANK. The veto only
#   fires on a cell the row is NEWLY entering (_cap_new_entry_veto:
#   `_foreign = _memb & (_cell != c_p) & ...`). With both flagship modes
#   now inside ONE cell, a cross-seam in-model move has new cell == cur
#   cell, is not foreign, and is NOT POLICED AT ALL -- correctly, since the
#   occupancy census does not change when a leaf moves within a cell. What
#   the headroom still governs are the two MID-BAND cap boundaries per band
#   (the 1/4 and 3/4 points), where cap + 2 keeps peak handover possible
#   against a cap measured as binding 53% of the time at the flagship
#   bands. 2 remains right: it is an ADDITIVE allowance on destination
#   occupancy, and the transient double-occupancy it must exceed did not
#   change when the cell halved.
# RAISED 20 -> 50 (user ruling 2026-09-19): "we cannot be up against that
# edge". This is a BELT against a runaway ratchet, not a design target, and
# 20 was only ever inherited from the probe's nleaves_max. Truth-set source
# density per cap cell, scaling SNR by sqrt(Tobs) off the 3-month set,
# detection at SNR 7:
#     Tobs    p99   p99.9   max   cells >=20   cells >=50
#     3 mo      6       8     8        0            0
#     6 mo     11      12    14        0            0
#     1 yr     17      20    21        4            0
#     2 yr     28      34    35       92            0
# So 20 already BINDS at 1 year (four cells exceed it) and badly at 2, while
# 50 is clear of the densest cell at every Tobs through 2 years. It costs
# nothing to raise: the knob gates the increment condition and sizes no
# allocation, and on the live 6mo run the highest cap reached is 17 with
# nothing at the ceiling, so this changes no cell's behaviour today.
export GB_CAP_CELL_MAX=50
#
# ── SUPERSEDED 2026-08-30. The block above reasons about the divisor-2
# geometry (two mid-band cap boundaries per band, "when the cell halved").
# Under GB_CAP_DIVISOR=1 + STAGGER there is ONE cap boundary per band, at
# its midpoint, and the straddling cell is centred on the SEAM.
#
# 2 -> 0: ENFORCE CAP-CELL OCCUPANCY FOR IN-MODEL (user ruling). Sub-band
# boundaries stay PERMEABLE -- the N/4 bin leak in _run_in_model_repeats
# (``new_bin < lo_s - n4_s``) is untouched, and GB_BAND_WINDOW_STRICT
# governs the RJ window, not this. A source may still walk across a
# sub-band edge; what it may no longer do is walk into a cap cell that is
# already at capacity.
#
# MEASURED, first hour of the 2026-08-30 run (row 2, the first GB propose):
# 555 cap cells held two leaves, one either side of a sub-band seam. Those
# doubles CANNOT have been born:
#   - the cap armed 08:28:41, before the grid fit and all births;
#   - only bands c-1 and c can reach cell c, and 555/555 doubles had
#     |delta band| = 1 with 0/555 sharing a residue mod GB_BAND_UNIT_STRIDE
#     -- so the two bands NEVER co-open and two births into one cell inside
#     a single unit is geometrically impossible;
#   - a later unit's birth is blocked: the at-cap mask is rebuilt at every
#     unit open and applied, exclusions growing 213,239 -> 1,462,467.
# By elimination the second leaf WANDERED IN, crossing the cap edge at the
# sub-band midpoint into a cell already at cap -- which headroom 2
# explicitly permits. Occupancy topped out at exactly 2, never 3: a gate
# holding at cap + headroom, not a gate that was missing.
#
# So the birth gate and the staggered grid are working, and this one knob
# manufactured every seam double. 1 does NOT help (cap + 1 = 2 is the
# observed state); only 0 closes it.
#
# ACCEPTED RISK: at 0 a source cannot enter an occupied cell at all, so a
# seam pair resolves by one leaf DYING rather than by the two merging.
# Deaths stay proposable in over-cap cells and a leaf still moves freely
# within its own cell, which is the intended mechanism: the cap forces one
# source per straddling cell, and tempering handles the mid-band pairs.
export GB_CAP_INMODEL_HEADROOM=0
# ===========================================================================
# V9-4: CAP CELLS OFF (user ruling 2026-09-24 -- "So no caps. Keep all those
# options available but turn them off.").
# ===========================================================================
# The progressive per-band/per-cap-cell leaf cap is ARMED ONLY by
# GB_LEAF_CAP_START (recipe.py: `int(os.environ["GB_LEAF_CAP_START"]) if
# os.environ.get("GB_LEAF_CAP_START") else None`), so an EMPTY value is the
# off switch and every knob below stays present and documented, inert,
# ready to be re-armed by setting a start value again.
#
# WHY OFF. The cap existed to stop a band swallowing leaves before the
# search could judge them, and it brought its own failure modes with it --
# the ghost-increment ratchet, the cap-quiescence and cap-headroom vetoes
# that had to be bolted onto stage convergence to stop a cap blockade
# reading as convergence, the half-band cap_edges shift, the at-cap birth
# skip. The v9 structure attacks the same problem from the other end: run
# in-model to convergence per sub-band and let the per-(walker, band)
# shutoff decide when a band is done. The bet is that this makes the cap
# unnecessary; if it is wrong, the symptom is bands running away in leaf
# count and the fix is to set GB_LEAF_CAP_START again.
export GB_LEAF_CAP_START=
# The cap's companion knobs (drift gate, edge leak, the two
# stage-convergence vetoes) are turned off AT THEIR OWN EXPORT LINES below,
# not here -- bash resolves exports in order, so a duplicate here would be
# silently overridden by the later one. Search V9-4 to find them all.
# ===========================================================================
export GB_SEARCH_CAP_QUIESCENT=0   # V9-4: cap off -> nothing to hold for

# ===========================================================================
# V9-9: PER-(WALKER, BAND) RJ SHUTOFF VALVE -- the STAGE convergence signal.
# ===========================================================================
# A (walker, band) pair freezes when its sub-band cold-chain logL converges
# within the current recipe step, and the valve RELEASES when the next step
# begins. A search stage ends when EVERY OCCUPIED pair has shut off (user's
# words) -- empty pairs can never shut off by construction, so "all pairs"
# is unreachable and the occupied-only reading is the operative one.
export GB_SEARCH_BAND_SHUTOFF_PER_WALKER=1
# 5 -> 3 (user ruling 2026-09-25): "let's make sure it's not too many
# iterations. 2 or 3 is fine." This is how many consecutive iterations an
# occupied (walker, band) pair must post a converged sub-band logL before
# it shuts off; the STAGE ends when every occupied pair has. A shorter
# window ends each search stage sooner, which is what we want now that the
# per-leaf cost is climbing (7.4 -> 70.6 -> 101.9 s/leaf over 628's three
# iterations) -- time is better spent in the next stage than re-proving a
# band that already stopped paying.
export GB_SEARCH_BAND_SHUTOFF_CONV_ITER=3
# LEVEL-3 VALVE TOLERANCE. The bar a (walker, band) must clear to keep
# its valve OPEN. Default without this export is 0.5 * leaf_cap_ndim =
# 4.0, which is the LEVEL-1/2 per-row dll -- far too fine for a
# whole-band plateau test, and why 154 pairs sat chronic.
#
# 20, set DIRECTLY by Mike on 2026-09-28. ⚠ This is a JUDGEMENT, not a
# formula: the earlier 32 came from "one floor-SNR source",
# 0.5 * SNR^2 = 0.5 * 8^2, and 20 is not that expression for any floor
# in play (the rule would give 32 at floor 8 and 12.5 at floor 5 in
# gb_search_2/3). Do not "restore" it to a derived value on the
# assumption the rule still holds -- it was superseded.
#
# A CONSTANT rather than derived from the stage profile, deliberately:
# three different SNR attributes are in scope here
# (opt_snr_rej_samp_limit, opt_snr_limit_search_coarse, _fine) and
# picking the wrong one would silently mis-set a gate threshold.
# Change it by hand per stage.
# Read from env AT JUDGE TIME and NOT part of the band_shutoff_w_step
# stamp, so the persisted streak/max survive this change.
export GB_SEARCH_BAND_SHUTOFF_LL_TOL=20
# THE ALL-WALKER RULE (Mike 2026-10-17): the convergence criterion above
# stays per (walker, band), but a sub-band shuts off -- on every walker --
# only at a judge where every walker holding a source in it is converged
# together, and a shut band reopens on every walker as soon as one of them
# stops being converged (its band gain beat the all-time max by LL_TOL
# again). A converged walker in a band the others have not converged stays
# OPEN. Why: on the 1yr job 780 the per-walker valve shut 46 of walker 0's
# 58 pairs at 1.5-2.5 mHz (row 71) while the other walkers kept adding ~5
# sources a row there; a shut pair runs no RJ, so walker 0 could never take
# the 185 leaves it lacked and fell 3,750 nats behind, its own foreground
# absorbing the missing power. Read at judge time; 0 = the per-walker valve.
# Log: "[GB_GATE] ... all-walker rule on: held open N reopened M" and the
# REOPENED line when a band stops being converged everywhere.
export GB_SEARCH_BAND_SHUTOFF_ALL_WALKERS=${GB_SEARCH_BAND_SHUTOFF_ALL_WALKERS:-1}
# A PURE IN-MODEL PASS AFTER rj_prior_removal (2026-09-28, Mike).
# rj_prior_removal is the ONLY RJ move with no in-model partner, so the
# SAVED STATE -- and the residual the next iteration's births are drawn
# against -- was the post-death state with no group-rule pass over it.
# The surviving twin of a killed double is exactly the cell that needs
# readjusting after the death.
# ⚠ NOT A DOUBLES CURE: deaths are still JUDGED on the unadjusted
# residual. This is correctness of the saved state and better births.
# Same move class as in_model / in_model_fstat, so it inherits the
# level-2 group rule, the level-3 shut mask and the census. Cost is the
# same class too: ~1400 s at a fully open population, under 200 s once
# most pairs are shut. GB_SEARCH_DROP_MOVES can name it.
export GB_SEARCH_INMODEL_AFTER_REMOVAL=1
# GB_SEARCH_DROP_MOVES EXISTS BUT IS DELIBERATELY UNSET.
# It omits named pure in-model slots from every search stage. Dropping
# in_model_fstat was proposed on 2026-09-28 on cost grounds (it and
# in_model are the same class -- one polish per RJ move -- and cost
# 1392 s + 1248 s of a 5564 s job-662 iteration, 47%) and MIKE
# REJECTED IT: in_model_fstat stays, so the stage compositions are
# unchanged. Unset is a documented no-op; the knob and its guard are
# kept because the guard is worth having -- rj_fstat_search can never
# be dropped (designated updater, and the only move that runs the
# level-3 judge and the stage-end check), and the knob refuses it with
# a SystemExit rather than leaving a stage unable to end.
# ⚠ OFF, deliberately: this is the PER-BAND stage schedule (SNR limits and
# phase maximization moving independently per band). In v9 those move
# together across RECIPE stages instead, so the per-band schedule must stay
# out of the way. The two are independent flags.
export GB_SEARCH_STAGE_PER_WALKER=0
# RJRecipeStep plateau window (user ruling 2026-09-19). The code default is
# 5, and at FOUR walkers that is too tight. The rule is a running-maximum
# ratchet (recipe.py::RJRecipeStep._stop_fn): max of the cold leaf count
# over the stage so far, against its max over the last `convergence_iter`
# iterations -- so ONE noisy dip ends the search stage. Both 4-walker
# production runs died that way while gb_search was still gaining leaves:
#
#   3 mo: ...771->778, 772->782, 778->782 x4, 782->777 STOP
#   6 mo: ...1674->1683, 1679->1683 x3,      1683->1681 STOP  <-- THIS RUN
#
# The 6 mo stopped on a TWO-leaf dip while its own walker-to-walker spread
# was 51 ([1638 1689 1656 1657]), after +110 leaves over the preceding 30
# checks, with GB_SEARCH_CAP_QUIESCENT armed but firing zero times (caps
# were quiescent, so nothing held the stage open).
#
# 20 gives the ratchet a window several times the observed per-check growth
# (0.5-3.7 leaves) without disarming the test: a genuinely converged stage
# still trips it, just later. Note the stage-scoped guard means the first
# check cannot fire until 2*20 = 41 iterations into the stage, which is
# exactly what a post-rewind resume wants.
export GB_PLATEAU_ITERS=20
# ---- THE TWO v4-POSTMORTEM FIXES (code defaults since 8d926f27; pinned
#      so the store's provenance is unambiguous) ----
# Birth fix (1274a66c): births draw fdot_astro_ratio | (f0, Mc) from the
# tight mixture around the F-stat grid fdot instead of U[-5, 5].
export GB_FSTAT_BIRTH_RATIO_TIGHT=1
# Ridge-Gibbs (8d926f27 / Eryn 6ed5a8b): zero-likelihood-call resample
# along the exact Mc^(5/3)(1+r)=const ridge; unfreezes (Mc, r, dist).
export GB_RIDGE_GIBBS=1
# CAP DRIFT GATE (2026-08-20, root-caused on the high-f probe: births
# respect the per-cell cap but in-model repeats walked 29 leaves into a
# cap-1 cell -- the 2026-08-15 TODO made real). In-model proposals whose
# f0 lands in a FOREIGN at-cap cell are vetoed; within-cell moves and
# drains of over-full cells stay allowed. This is the same mechanism
# that let production mosaics stack leaves past their cell caps. Watch
# [GB_CAPGATE] veto counts; GB_CAP_DRIFT_GATE=0 reverts.
export GB_CAP_DRIFT_GATE=0             # V9-4: cap cells off
# Leaf-cap PATIENCE: consecutive iterations without a sufficient (D/2)
# lnL improvement before a cap CELL advances. Code default is now 3
# (2026-08-16, was 5): caps live on the band/8 cap-cell grid, so 1,232
# cells must each climb from 1, and at ~6 min/iteration the ramp -- not
# the wall -- is what limits how fast the model can fill. Pinned here for
# the run record; raise it if caps ever outrun the likelihood evidence.
# GHOST-INCREMENT GUARD -- ON in v3, and ONLY meaningful with K=32.
# An EMPTY cap cell can never improve its max ll by D/2, so under the bare
# counter it accrued patience on a fixed clock and promoted itself
# alongside cells doing real work. In v2 that ran caps from 1 to a median
# of 14 by iteration 60: the model held 591 sources against a permitted
# 15,619 (3.8% of the allowance) with 0.3% of occupied cells at cap -- the
# cap tightest when the model was empty and needed no protection, absent
# once it was fullest. The guard starts a cell's patience clock only after
# it has improved at least once, mirroring `changed_once` in the PSD
# max-logL search. It must NOT ship at K=8: freezing empty cells at cap 1
# with a 135-bin cell re-imposes the 24.5% exclusion above.
export GB_LEAF_CAP_REQUIRE_IMPROVEMENT=1
# 4 (user ruling 2026-08-28, was 5): the v7 cap-gap analysis showed real
# 2->3 increments arrive at median gap 9 with ZERO at the floor of 5 --
# qualification-bound, but E[step] scales with the threshold, so 5->4 buys
# ramp depth with a wide margin before the floor binds. 3 remains the
# aggressive option if the ramp still lags. (Historical: 5-not-3 was
# probe-validated 2026-08-26 with the one-shot engagement latch +
# occupied-only patience, c251b267.)
# ⚠ NOTE for the v8-vs-v7 comparison: this is the ONE knob that makes the
# v8 GB configuration differ from v7's. The cluster checklist's gate 4
# assumed "same GB config by construction, the noise marginals are the new
# content" -- with this change the GB side moved too, so a v8-vs-v7 cap or
# leaf-count difference is NOT purely a noise effect. Set back to 5 if you
# want the noise comparison fully isolated.
# 4 -> 3 (user ruling 2026-09-16): the v4/v5 3-mo campaigns ran 3; 4 was
# the noise-merge-era compromise kept for v7 isolation, which no longer
# applies. At 6-mo wall (~33 min/it) each increment cost ~2.2 h at 4 —
# measured in production: caps raised on exactly the 4th consecutive
# at-cap iteration (rows 13->16/17). Plateau/occupancy gates, staggered
# cap grid, at-cap RJ skip, D/2 and GB_CAP_LL_CHECK all still stand.
# 2 was considered and deferred: a 2-it stagnation window is weak at 10
# walkers — revisit with the next snapshot's cap trajectory if 3 lags.
# STAYS AT 3 (user ruling 2026-09-18, after considering 8 and 5).
# RECORDED COUNTER-EVIDENCE, so the next person does not re-derive it: the
# gate is MAX over cold walkers -- a band's cap holds while the BEST walker
# keeps improving it by D/2, and increments once none has for this many
# CONSECUTIVE iterations. With W walkers the max gets W independent chances
# per iteration to reset that clock, so the same iteration count is weaker
# evidence of a plateau at small W and the cap ratchets FASTER the fewer
# walkers run. Every previous value (5, 4, 3) was tuned at 10 or 24 walkers;
# this store runs 4. Measured on the 4-walker run, iterations 35 -> 68:
# summed cap 1232 -> 2373 (+34.6 slots/it) against cold leaves 420 -> 1066
# (+19.6/it), i.e. 1307 cap slots of unused headroom, 202 bands at cap >= 4
# with a max of 7 while the median band still held ONE source -- and the
# flagship 20.38 mHz source carried THREE fragment leaves per walker.
# Matching the ~30 walker-iterations that 3 bought at 10 walkers would mean
# 8 here. The knob is NOT walker-aware, so raising it is the operator's
# call; the fresh store's fixed warm start is the other half of that
# fragmentation story and is being addressed separately.
export GB_LEAF_CAP_MIN_ITERS=3
export GB_CAP_LL_CHECK=1
# Grouped RJ scheduling: accumulate inds=True picks across RJ rounds
# (1 proposal per cell per round), then ONE full-width in-model block.
# Code default since 2026-08-13; pinned for the run record. =0 restores
# the per-round RJ->in-model interleave.
export GB_RJ_GROUPED_INMODEL=1
# ---- 2026-08-14 rj stack (ALL code defaults; pinned for the run record;
#      each =0 reverts that piece independently) --------------------------
# ⚠ 1 -> 0 (user ruling 2026-09-25). THE SCHEDULE IS THE POINT.
#
# =1 is the rigid-batch path: it runs ALL RJ rounds -- one source per cell
# per round, ~12 rounds per propose measured on job 621 (360k birth
# proposals over ~29.6k cells) -- and only then does ONE end-of-unit
# in-model phase. So a cell could accept a dozen births with nothing
# polished in between, which is how the 19.66822 mHz source ended up with
# 2-3 leaves in 33 of 96 (rung, walker) cells, the extras at 0.76-0.99x
# amplitude because they were never polished at all.
#
# =0 is the staged scheduler, which is the schedule the user asked for:
#   RJ round -> survivors pool, their cells FREEZE
#            -> flush: in-model TO CONVERGENCE on the pool
#            -> cells UNFREEZE, completed cells retire and new bands stage
#            -> next RJ round
# ("RJ 1 round -> in model converge -> RJ 1 round -> in model converge",
#  and "after each converge, you sub out bands that are wholly complete
#  for new bands" -- the latter is _advance_and_refill, called immediately
#  after every flush.) PE gets the same schedule WITHOUT the convergence
#  for free: _converge_state_for returns None on a PE-stage move.
#
# ⚠ COST, unmeasured on this hardware: the rigid path exists because "the
# fixed, repeating fill -> rounds -> accept shape per batch is what CUDA
# graph capture needs". The staged scheduler's shape changes every round,
# so graph capture is off. Watch the [GB_TIMING] rj_step / inmodel_repeats
# split against the 621 baseline; =1 reverts.
export GB_RJ_DIRECT_BATCH=0        # staged scheduler: RJ round -> converge
                                   # -> refill -> RJ round; =1 = rigid batches
# LIVE-CAP PICK OFF (user ruling 2026-08-27, snapshot-2 timing autopsy):
# under the ALIGNED divisor-1 grid, "saturated across all K cap cells"
# is "all 1" -- every occupied band trips it at cap 1, so the live-cap
# regime staged 879,846 dead at-cap birth slots (all temps) through the
# sig-het in-model staging at ~2.3 s per 1024-slot sub-block ~= 220 s
# per band unit -- THE 14x rj_fstat_search blowup vs v6 (whose
# divisor-4 grid needed all 4 cells saturated, i.e. almost never).
# =0 restores the 2026-08-12 unit-open exclusion: at-cap cells' dead
# slots never stage; a cell freed by a death births next UNIT instead
# of same-unit (immaterial at our removal acceptance).
export GB_RJ_LIVE_CAP_PICK=0
export GB_BUFFER_FIXED_CAPACITY=1  # ONE capacity buffer; smaller units
                                   # resize-rebind instead of drop+rebuild
export GB_RJ_FSTAT_CTR_HOIST=1     # F-stat distance centers batched once per
                                   # unit (was 735 s/propose per-round)
# EPOCH CENTERS (user ruling 2026-08-15: compute the center
# distributions ONCE when the fstat distribution is built in setup(),
# smear for inaccuracy, done): per-epoch table over the proposal's
# drawable support; propose-time = nearest-node lookup (centers chain
# 109-953 s -> ~0). Smear defaults 2.0 in epoch mode (covers <=100-
# propose staleness + node mismatch); =unit restores the per-unit hoist.
export GB_FSTAT_CTR_MODE=epoch
# (GB_FSTAT_CTR_SMEAR is unset ON PURPOSE: a 1.5 pin would override the
# epoch-mode 2.0 default, which is what covers the <=100-propose table
# staleness. The smeared sigma feeds BOTH the draw and the densities,
# so detailed balance is exact at any smear.)
#
# ############################################################################
# ## v7 CHANGE-SET FOLDED IN (2026-08-28). The v8 script was cut from the   ##
# ## PRE-change-set v7, so these were missing. Every one was validated in   ##
# ## v7 production (jobs 352-368, snapshots 8-12) unless noted.             ##
# ############################################################################
# FUSED TWO-QUADRATURE PHASE MAX (GBGPU c49fcb1 / LAT 9704c4a8). One kernel
# call returns both quadratures instead of two evaluations at phi0 and
# phi0+pi/2. GPU-VALIDATED in production job 352: 0 errors over 86k log
# lines, TEMPER_CHECK 297/0, COLD audit medians at baseline. =0 is the
# no-rebuild rollback to the legacy two-call path (bit-identical algorithm;
# the fused path is epsilon-better, NOT bit-identical).
export GB_PHASE_MAX_FUSED=1
# F-STAT 4->2 BASIS-FILTER FOLD (GBGPU 8245ced / LAT ae8cdb87+07bbac97).
# The chunked-WDM F-stat kernel generated four independent waveforms, but
# the four basis filters are 2 polarization directions x 2 phase
# quadratures (iota=pi/2 throughout, psi in {0, pi/4}, phi0 in quadrature
# pairs) -- and the quadrature half is a constant unit-modulus rotation
# {+1,-1,+i,-i} of the complex heterodyned representation, which the
# d_h_im_out machinery already produces. Two generations span the
# identical 4-space (rank 4, condition 1.000), so this is EXACT: no
# physics loss, no sampling-semantics change.
# VALIDATED: sign pinned CPU-side against a deliberately CONJUGATED build
# that failed the gate, then the parity suite passed in full ON GPU.
# The gate is on SIGNED (N, M) and must never be reduced to F -- a
# conjugated sign is N->DN, M->DMD with D=diag(1,1,-1,-1), which leaves F
# EXACTLY invariant (tests/test_fstat_filter_fold.py pins this).
# EXPECT rj_fstat_centers ~830 s -> ~400-450 s, i.e. ~18-20% off the GB
# iteration. The OFF baseline is already banked (snapshot 13, 830.2 s).
# =0 restores the unfolded 4-generation path bit-for-bit.
# REQUIRES A GBGPU REBUILD -- LAT does not compile that header. A stale
# .so is a LOUD TypeError at the first F-stat call, NOT a silent no-op
# (corrected 2026-08-28; the earlier note here said the opposite).
# _FSTAT_FOLD_KERNELS in gbcomps.py is a hard constant rather than a probe
# of the compiled binding, and the trailing fold arg is passed
# unconditionally -- fold ON or OFF -- so a binding that predates the fold
# cannot accept it and raises immediately. rj_fstat_centers ~830 -> ~400-450
# s is still the check that the fold is doing work, but a bad build fails
# first and visibly.
export GB_FSTAT_FOLD=1
# REPLACE PHASE-MAX + ROTATION-ON-ACCEPT. "auto" (not =1): ON for the
# search replace exactly as =1 was, OFF for any PE-stamped replace. A hard
# =1 would force maximization onto PE, which the 2026-08-28 general rule
# forbids ("no maximizing over parameters during PE" -- PE samples a
# posterior; maximize-and-keep biases it). Live since job 352: cold replace
# acceptance ~3x (0.0002-3 -> ~0.0010) at Delta-ll up to ~700.
export GB_REPLACE_PHASE_MAX=auto
# PER-ROW F-STAT CENTERS THROUGH THE UNIT-OPEN CACHE (LAT 86ed9353).
# ⚠ MEASURED A WASH in snapshot 12: rj_fstat_centers 725-743 s vs a
# 713-799 s pre-fix band. There was no recomputation to dedupe -- the
# precompute row count ~= the picked row count, at an identical 0.667
# ms/row. Kept ON because it is the code default and costs nothing either
# way; it is NOT a speed lever. The real centers levers are the
# multi-device lane rebalance and row-count reduction (see the scoping
# notes). Telltale: [FSTAT_CTR] says "perrow (unit-cache)".
export GB_FSTAT_PERROW_UNIT_CACHE=1
# DEFERRED CELL-LABEL RELABELS (LAT 9fa32109; code default flipped ON in
# bcdde159 per the user's "only cells change labels" design invariant).
# Rung-pair/vertical-swap relabels accumulate in a slot+pos composition
# table and flush once per tempering chunk / repeat block. Pinned here for
# the run record. Tripwires: [GB_TEMPER_CHECK] must stay 100% MATCH with
# "unit label checks passed" (340/340 in snapshot 12).
export GB_CELL_LABEL_DEFERRED=1
# FUSED IN-MODEL GATE/ACCEPT KERNEL (LAT 0f0fc73a + the 07634536 nvcc
# guard fix). ~160 CuPy launches per repeat-step -> 3 backend calls.
# ⚠ OFF FOR v8 PENDING THE v7 A/B (user ruling 2026-08-28: "if the
# multi-GPU picture is not clearly better, keep it off for v8"). Expected
# size is small -- ~450k launches removed => single-digit seconds, 0.2-1%
# of a ~1130 s propose; the real hope was multi-GPU overlap, since the
# kernel leaves ONE data-dependent host sync where the python chain had
# many. FLIP TO 1 only if the v7 run's `inmodel_gate` mark plus
# gpu_util_*.csv show a clear multi-GPU improvement. Requires ./install.sh
# to have built the binary; without it the loaders degrade to the python
# chain with a one-line warning (safe, just not faster).
# >>> FLIPPED TO 1 (2026-09-11, ported from the 10w arm de903711): first
# >>> production arming was the 3mo relaunch after job 473; est. -30..-40
# >>> s/it. VERIFY on this run: in-model acceptance rates + [GB_CELL_LL]
# >>> + sig-het audit unchanged vs the python chain. =0 reverts.
# BACK TO 0 (2026-09-11 user ruling, ported from the 3mo 10-walker arm):
# it fuses only the inmodel_gate + inmodel_accept launches, ~18 s/it at
# 250/100 repeats and ~9 s at 100/50 -- not worth a first production
# exposure. Turn on only after tests/test_gb_inmodel_accept_kernel.py has
# been run on a cluster GPU node. (With GB_INMODEL_TRACE armed it stood
# down anyway -- job 474 logged "standing down".)
export GB_INMODEL_ACCEPT_KERNEL=0
# ---- THE v8 EXPERIMENT: OBSERVABLE-BASIS IN-MODEL PROPOSAL ----------
# Pinned EXPLICITLY even though it is now the code default, so this run
# does not silently change meaning if the default is ever revisited, and
# so the v7/v8 knob diff names the thing under test.
#
# WHAT IT CHANGES. In-model steps are drawn in
#   z = [lnA, f_mid, fdot, phi0, cos_iota, psi, alpha, sin_delta, Mc]
# and mapped back; the SAMPLING basis is untouched (f0 stays anchored at
# t_ref, so bands, cap cells, storage and the catalogue conversion are all
# unaffected). Two measured defects motivate it, both invisible in the
# sampling basis:
#   1. (dist, Mc, r) -> (A, fdot) is 3->2, so Mc and r both drive fdot and
#      a joint draw can move both and leave fdot_total put.
#   2. f0 is the frequency at the START of the data while the data
#      constrains the frequency at the MIDDLE. On the real flagship Fisher
#      the legacy joint draw walks an f0-fdot ridge of slope -0.898 T
#      against the geometry's -T/2; the excess is 0.170 bins of spurious
#      f_mid motion per fdot step, ~14 sigma at rho = 46.
#
# WHAT TO WATCH, in [GB_ACCEPT] / [GB_OBS_BASIS] lines:
#   * "in-model by proposal type -- obs_basis:" replaces "infomat:"
#   * cold acceptance 0.67 (v7) should fall toward 0.23-0.44
#   * "[GB_OBS_BASIS ...] in-model motion" gives mean |dln_fdot| and
#     |df_mid| in bins, proposed and accepted -- the direct read of
#     whether fdot is finally moving
#   * the flagship's fdot/truth should walk off 1.35
# FALSIFIABLE PREDICTION: the shear goes as fdot*T^2, so this must be
# essentially NEUTRAL below ~7 mHz (0.04 bins) and large above ~15 mHz
# (3.1 bins). A material low-f change means a BUG, not a win.
#
# =legacy reverts to the v7 proposal bit-identically (same seed), which is
# what makes the comparison readable. Keep that path until v8 is read.
export GB_INMODEL_PROPOSAL=observable
# Fiber (Mc) component of the composite step. 0.0 = the 8-observable step
# only; gb_ridge_gibbs already supplies fiber mixing on the main state for
# free, so this stays an INDEPENDENT A/B rather than a coupled one.
export GB_INMODEL_OBSERVABLE_FIBER_WEIGHT=0.0
# Overall step multiplier for this path. Deliberately its own knob and NOT
# the legacy jump_factor, which was tuned against the eigen-floored draw.
# RAISED 1.0 -> 2.0 (2026-09-19). Under GB_INMODEL_OBSERVABLE_EIGEN=axis
# the GB in-model step came out TOO SMALL: measured on the 6mo run,
# obs_basis acceptance cold 0.71 / all-rung 0.78, with
#   mean |df_mid| proposed 0.0184 bins vs 0.0032 accepted
# against the well-tuned 10-walker diagonal arm's proposed 0.038 /
# accepted 0.021 at acceptance 0.219. A one-axis-per-repeat draw is
# effectively 1-D, where the optimal acceptance is 0.44 (not 0.234), so
# 0.71 says the step wants roughly doubling -- which is this knob, and
# deliberately NOT GB_JUMP_FACTOR (that one was tuned against the
# eigen-floored draw; see the block above).
# 2.0 -> 1.5 (2026-09-28, Mike). The 09-19 doubling to 2.0 was too
# far, and the evidence is per-axis: under
# GB_INMODEL_OBSERVABLE_EIGEN=axis ONE eigen axis is drawn per repeat,
# so an axis's accepted/proposed motion ratio IS its own acceptance.
# Job 663 measured mean |df_mid| accepted/proposed at 0.01-0.34 and
# |dln_fdot| at 0.04-0.10 against a POOLED cold acceptance of
# 0.55-0.79 -- i.e. f_mid was accepting ~7% of its own draws and fdot
# 3-7%, while the other seven axes ran ~80% to make up the pooled
# number. The doubling itself made accepted f motion SMALLER
# (0.0032 -> 0.0020 bins), which is the signature of a step already
# past its optimum on that axis.
#
# ⚠ ONE GLOBAL KNOB CANNOT FIX BOTH HALVES: 1.5 helps f_mid and fdot
# and costs the seven axes that were happy at 2.0. It is an interim
# compromise until the per-axis / per-source scaling lands (that work
# is on the gb-inmodel-axis-adapt branch, deliberately NOT on dev).
# The per-axis census added alongside this is what will show whether
# 1.5 moved the right axes.
export GB_INMODEL_OBSERVABLE_JUMP=1.5
# Mc step as a FRACTION of the m_chirp prior box (only reachable when the
# fiber weight above is non-zero).
export GB_INMODEL_OBSERVABLE_MC_STEP=0.05
# Shear coefficient as a fraction of Tobs; 0.5 => f_mid. The shear has
# determinant 1 for ANY coefficient (verified for 0, T/2, 0.41T, T, -3T),
# so a wrong value here costs acceptance and never correctness.
export GB_INMODEL_OBSERVABLE_SHEAR=0.5
# ---- COMBINED WITH THE EIGENBASIS (user ruling 2026-09-14: "GB in model
# should always be observed basis. We should combine that with the
# eigenbasis.") ------------------------------------------------------------
# The step STAYS in observable z with the same log-Jacobian factors, but
# instead of independent per-coordinate steps it draws along the eigen
# table of each source's own information matrix congruenced into z (exact
# chain rule through the transform -- no extra waveform calls) and
# whitened by the analytic step scales above. "full" = the JOINT
# correlated draw over all axes per repeat, the whitened shear-free modern
# version of the legacy full-covariance infomat draw (the old proposal was
# never diagonal -- user preference 2026-09-14); "axis" = one uniformly
# picked eigen-axis per repeat is the one-knob alternative; empty/0
# reverts to the independent per-coordinate draw BIT-IDENTICALLY, RNG
# stream included. Rows without a table yet (fresh births before their
# first infomat visit) silently take the diagonal draw -- by design.
# GB_INMODEL_OBSERVABLE_EIGEN_SMAX (default 10.0) caps the whitened
# sigmas. Watch the [GB_TIMING] span "infomat_obs_eigen" (the per-block
# Gamma_z stash) and the obs_basis cold acceptance vs the 3mo arm's ~0.2.
# THIS RUN IS THE TEST (user ruling 2026-09-14: "Let's just go to the
# full 6mo run test") -- the 3mo probe script
# (submit_gf_3mo_v8_10w_eigenaxis_probe.sh) remains available as an
# ISOLATION arm if the combined draw needs to be separated from the 6mo
# changes. Launch with GB_INMODEL_OBSERVABLE_EIGEN= (explicitly empty)
# to fall back to the diagonal draw bit-identically if it reads badly.
# DEFAULT CHANGED full -> axis (user ruling 2026-09-19). "full" is a JOINT
# draw over every eigen axis per repeat, so ONE ill-conditioned direction
# contaminates EVERY proposal: an observed-information eigenvalue projected
# onto the PSD cone leaves lambda ~ 0, its whitened sigma (~1/sqrt(lambda))
# rails at GB_INMODEL_OBSERVABLE_EIGEN_SMAX = 10, and the step overshoots
# along an axis the matrix could not measure. Measured cost in full_pe:
#     GB obs_basis cold 0.0139 (full)  vs  0.0864 (10-walker arm, no eigen)
#     mean |df_mid| proposed 0.218 bins vs 0.026 accepted -- 8.5x too wide
#     (the 10-walker arm: proposed 0.038, accepted 0.021, ratio 1.8)
# "axis" draws ONE whitened eigen-axis per repeat, so a bad direction costs
# only the 1-in-ndim repeats that pick it instead of all of them. That is
# exactly the mode EigenAxisMove(mode="axis") uses for mbh / emri / sobbh,
# and those three are healthy (accepted fraction 0.57 / 0.16 / 0.26).
# Escapes, both bit-identical to what they name:
#   GB_INMODEL_OBSERVABLE_EIGEN=      -> the diagonal observable draw
#   GB_INMODEL_OBSERVABLE_EIGEN=full  -> the 2026-09-14 joint draw
export GB_INMODEL_OBSERVABLE_EIGEN=${GB_INMODEL_OBSERVABLE_EIGEN-axis}
echo "[GB-OBS-EIGEN] GB_INMODEL_OBSERVABLE_EIGEN='${GB_INMODEL_OBSERVABLE_EIGEN}' (empty = diagonal draw)"
# ---- AND THE F-STAT GRID IN THE SAME BASIS -------------------------
# fdot becomes a FIRST-CLASS grid axis instead of the r = 0 manifold the
# grid searches today. Measured in v7: 39.6% of low-f and 10.5% of high-f
# alive leaves carry fdot < 0, which the current grid CANNOT represent --
# every one of them arrived from the blind post-hoc r-draw or in-model
# drift, never from the search. The new axis also costs FEWER nodes
# (20.38 mHz: 71 -> 53, and v7's top group used exactly 71) while covering
# 10x the range, because the aligned coherence width is 13.4x coarser.
#
# Complementary to the in-model move above -- the grid places births on
# the ridge, the move refines them along it -- so shipping one alone can
# read as "no effect". ATTRIBUTION comes from disjoint observables, the
# sharpest being the fraction of births with fdot < 0: the old grid cannot
# produce one at all, so any non-zero count is unambiguously this change.
#
# WARNING: an *_peaks_stacked.npz fitted in the Mc basis is REFUSED on
# load (by design -- axis 2 is a chirp mass in one basis and Hz/s in the
# other). v8 starts fresh, so it refits; a resume of an older store would
# need its epoch caches cleared.
export FSTAT_FDOT_AXIS=1
export FSTAT_FDOT_RATIO_MAX=5.0

# PEAK-SELECTION FLOOR 8.0 -> 6.25 (user ruling 2026-09-23: "err on the side
# of grabbing things not missing them. Even if there is excess stuff that is
# fine.")
#
# NOTE ON THE RUN IN FLIGHT: this knob was never exported before, so the
# current 6mo run has been selecting at the stock 8.0. That is what the
# measurement below is against.
#
# WHY THE STOCK 8.0 IS TOO STRICT *ON THIS RUN*. The 8.0 default is a
# false-alarm argument (fstat_proposal.py): under the null 2F ~ chi2_4, so
# F = 32 gives ~4e-5 expected false peaks on a full-band production comb.
# That null assumes the ASSUMED PSD equals the true one. Here it does not --
# galfor absorbs the galaxy the search has not resolved, so the assumed floor
# is k = 1.85-2.00x the truth over 3-4 mHz. The realized null is then
# 2F ~ chi2_4 / k: the whole distribution SHRINKS, the cut becomes orders of
# magnitude stricter than designed, and the cost is paid entirely in missed
# sources. Measured 2026-09-23 (galfor_6mo_figs/): 8.0 is an effective
# TRUE-SNR threshold of 11.3 at 3.5 mHz, hiding 693 findable binaries, 506 of
# them in 3-5 mHz.
#
# WHY 6.25 AND NOT LOWER. Expected false peaks against this run's own k(f),
# with the comb size back-calibrated from the ruling's own 4e-5 number:
#
#     cut 8.00 -> 6e-5 false,     0 of 693 recovered   (the run in flight)
#     cut 7.00 -> 0.06 false,   319 recovered
#     cut 6.50 -> 1.2  false,   478 recovered
#     cut 6.25 -> 5.3  false,   551 recovered          <-- HERE
#     cut 6.00 -> 21.7 false,   628 recovered
#     cut 5.66 -> 134  false,   692 recovered
#
# 6.25 buys 73 more sources than 6.50 for ~4 more expected false peaks. The
# marginal rate collapses just below it -- 6.50->6.40 is 36 sources per false
# peak, 6.25->6.20 is 7, 6.10->6.00 is 3.7 -- so 6.25 sits at the last point
# where the exchange is still clearly favourable. 6.0 and below is the regime
# the 2026-08-17 SNR-8 ruling was written to escape (a peak list that was
# ~98% noise put 2-7 leaves on one source).
#
# THIS IS A COMPENSATION, NOT A FIX, AND IT IS RUN-SPECIFIC. It trades a
# known-inflated floor for a looser cut; do NOT port it to a run whose noise
# model is well calibrated, where it would simply loosen the false-alarm rate
# for nothing. The principled replacement is a per-band self-calibrated
# threshold (8/sqrt(k_hat), k_hat measured from the F-stat grid's own median)
# which recovers ALL 693 at the 4e-5 design rate -- see
# galfor_6mo_figs/FSTAT_SELFCAL_PROPOSAL.md. Revert to 8.0 when that lands,
# or when GB has subtracted enough that galfor is no longer inflated.
#
# CAVEAT the table cannot show: the chi2_4 null only holds where the residual
# is Gaussian. At 1-3 mHz the confusion forest is not, so the real false rate
# there is worse than tabulated at ANY global cut. Watch the per-band peak
# counts and the birth multiplicity per source in the 1-3 mHz bands.
#
# ############################################################################
# ## ⚠⚠ THIS KNOB IS A NO-OP ON A RESUME UNLESS YOU CLEAR THE STAGE-B CACHE ##
# ##                                                                        ##
# ## fstat_gridfit's stage-B load path is "load and return, nothing         ##
# ## recomputed": the peak LIST was selected at whatever threshold was in   ##
# ## force when it was FITTED, and nothing on that path re-reads the knob.  ##
# ## So a resume keeps proposing from the old SNR-8 list while this script  ##
# ## and the run log both say 6.25.                                         ##
# ##                                                                        ##
# ## THE CHEAP MIGRATION -- delete ONLY the stacked file, KEEP the comb:    ##
# ##                                                                        ##
# ##     rm <fit_dir>/fstat_grid_peaks_stacked.npz                          ##
# ##     # keep  <fit_dir>/fstat_grid_comb.npz                              ##
# ##                                                                        ##
# ## The comb stores F_max for EVERY node, so peaks re-select at the new    ##
# ## threshold (cheap, deterministic) and only stage B reruns. Deleting the ##
# ## whole epoch directory also works but pays for the comb again -- and    ##
# ## the comb is the expensive half (monitor_6mo.html was still reporting   ##
# ## "the grid fit is still running" at the Sep-18 snapshot).               ##
# ##                                                                        ##
# ## As of 2026-09-23 the stage-B cache STAMPS peak_min_F and the loader    ##
# ## REFUSES a mismatch (fstat_gridfit._check_cached_peak_threshold). But   ##
# ## caches written BEFORE that stamp -- including this run's -- carry no   ##
# ## stamp and only WARN. Watch for "carries no peak_min_F stamp" in the    ##
# ## log on the first resume; that line means you must clear it by hand.    ##
# ############################################################################
# V9-11: 6.25 -> 8.0. Per-stage again: gb_search_1 selects peaks at SNR 8,
# gb_search_2/3 at 6.25. ⚠ THIS ONE CANNOT BE MOVED BY AN ENV VAR MID-RUN --
# the floor is consumed when the grid is FITTED and stamped into the stage-B
# cache, whose loader REFUSES a mismatch. So the stage entry installs an
# in-code override (fstat_proposal.set_peak_min_F_override) AND forces a
# fresh-epoch refit; the old epoch stays on disk, valid and unread, and
# NOTHING IS DELETED. Watch for:
#   [V9-STAGE gb_search_2] F-stat peak floor -> SNR 6.250 (F 19.531)
#   [V9-STAGE rj_fstat_search] FORCED F-stat refit: opening epoch N
# This export is the value epoch 0 is fitted at, i.e. stage 1's.
export FSTAT_PEAK_MIN_SNR=8.0
# F-STAT CENTERING OFF (probe verdict 2026-09-02, 4-arm A/B, readout
# artifact 2f5d673c): centered births are a stacking engine in BOTH test
# bands (A_ctr multiplicity 4.33/walker on a 1-source band, 23% neg-fdot
# junk; B_ctr cold logL max FELL 1,478 nats after it 65) while OFF is
# clean in both -- and best logL was identical-within-1-nat, so the junk
# bought nothing. Maximizing all F-stat quantities finds local maxima too
# easily. This also retires v7's rj_fstat_centers cost (54%/iteration).
# ⚠ THE ENV ALWAYS WINS over the install-site stamps
# (_fstat_dist_birth_stamp): =0 ALSO reverts the 08-28 "mirror them" PE
# ruling, so rj_fstat_pe births go back to full prior widths
# (bit-identically the pre-stamp path). That is DELIBERATE here: the
# clean probe arms ran exactly this env, so this is the validated
# configuration; a PE-only re-arm needs its own scoped knob first.
export GB_RJ_FSTAT_DIST_BIRTH=0
# RJ PHASE MAXIMIZATION (user ruling 2026-09-02, r2 probe verdict). With
# centering off, birth maximization would otherwise be GONE ENTIRELY: in
# v7-style search the maximizers came VIA the centers path, and
# GB_RJ_PHASE_MAXIMIZE defaults 0. The r2 probes measured the difference
# directly -- with phase-maximized birth scoring the flagship was ON
# TARGET at the FIRST stored iteration (-0.23 bins / fdot 0.92x,
# multiplicity 1.00 all run, 0% negative-fdot leaves) where the
# unmaximized r1 arm needed ~10 iterations; low-f stayed neutral and all
# 979 live detailed-balance traces still matched (phase max touches
# scoring only, never the observable-basis factors).
# V9-11: 0 -> 1. Phase maximization is now a PER-STAGE PROFILE value:
# gb_search_1 ON (hunting -- take the credit, and the r2 probes measured the
# flagship ON TARGET at the first stored iteration with it), gb_search_2 and
# gb_search_3 OFF. The profile sets it on the RJ moves only; the pure
# in-model moves stay at their deliberate False (in-model scoring is at the
# ACTUAL phase). As with the opt-SNR floor, this export is stage 1's value so
# a profile that failed to apply would hold stage 1's configuration rather
# than run stage 1 unmaximized.
export GB_RJ_PHASE_MAXIMIZE=1     # birth lands on target at iteration 0
# Amp max DEFAULTS TO FOLLOW phase_maximize -- it would silently arm with
# the line above. User ruling: phase max only, no amp max for now.
export GB_RJ_AMP_MAXIMIZE=0
# LIVE DETAILED-BALANCE TRACE (audit follow-up 2026-09-02). Recomputes the
# observable-basis log-Jacobian INDEPENDENTLY for one traced source per
# repeat and prints "-> match" / "*** MISMATCH ***". Negligible cost, and
# this run is the FIRST MULTI-GPU exposure of the factors path -- every
# probe ran gres=gpu:1 -- so the one failure class the probes could not
# see (a device-routing fault in factors) is exactly what this catches.
# 979/979 matched in the r2 probes; any MISMATCH line here is a stop
# signal. =0 disarms.
# 0 (2026-09-11, was 1 -- ported from the 3mo 10-walker arm): the
# per-repeat MH trace is a DEBUG knob (DEBUG lines per repeat, host
# syncs) and, while armed, the fused accept kernel stands down. Re-arm
# only for a diagnostic run.
export GB_INMODEL_TRACE=0
# DIFF DISCIPLINE: v7 exported GB_CAP_DIAG=1 and it costs time. Leaving it
# on in v7 and off in v8 would make v8 look faster for reasons unrelated
# to the proposal, so it is pinned ON here -- and the cap census is wanted
# regardless. (GB_FSTAT_CTR_AUDIT stays absent: measured at 0.260 s of a
# ~2000 s propose, 0.013%, so it carries no timing weight, and its verdict
# is already in -- see below.)
export GB_CAP_DIAG=1
# NOT PINNED, ON BY DEFAULT -- recorded so the run log is interpretable:
#  * GB_REPLACE_FSTAT_MAX resolves "auto" = ON for the search replace via
#    the recipe's replace_search_stage stamp (the move is named plain
#    "rj_replace", so the name idiom alone would MISS it). Search replace
#    candidates are the full JKS maximizer: slot 0 pinned AT the per-row
#    F-stat center, then priced through the UNCHANGED RJ densities as if
#    drawn (maximize-then-pretend). Telltale: one [GB_REPLACE_FSTAT_MAX]
#    line. =0 restores the exact-DB draw bit-identically.
#  * GB_PE_RJ_REPLACE: now pinned OFF above (user ruling 2026-09-02) --
#    the default-ON rj_replace_pe registered but never proposed in either
#    r2 probe, so it shipped as inert machinery.
#  * GB_RJ_FSTAT_DIST_BIRTH: the rj_fstat_pe stamp WOULD arm epoch-table
#    centers for PE births (user ruling 2026-08-28 "yes mirror them"),
#    but the explicit =0 export above overrides every stamp -- probe
#    verdict 2026-09-02, see that block. PE births draw at full prior
#    widths in v8.
# GB_FSTAT_CTR_AUDIT is DELIBERATELY ABSENT: it is a v7-only diagnostic
# (table-vs-per-row center deltas). Re-arm it here only if v7 never
# produced a completed propose to read it from.
#
# ITS VERDICT IS ALREADY IN (v7, 2026-08-29, 18 unit lines): the epoch
# table CANNOT serve birth centers. Medians dphi0 1.21-1.29 rad, dpsi
# 0.57-0.63 rad, dcos_iota 0.22-0.29, dln_center 0.81 -> 1.21 (factor ~3
# in amplitude) -- with a TINY f0 node gap (~1.1e-8 Hz ~ 0.08/Tobs), so
# the gap is not resolution: the table is solved against the epoch-0000
# residual and the F-stat maximum is data-dependent, drifting as sources
# are subtracted. That is why v8 does not re-run the audit and does not
# use the table. The per-row solve stands; the two levers below are what
# is left.
#
# ---- F-STAT COST LEVERS (both bit-identical at their defaults) ------
# rj_fstat_centers measured at 61% of a v7 iteration, ~100% inside
# fstat_nm_lane_score. GB_FSTAT_CTR_BATCH is the serial python chunk
# (4096 default => only ~2048 rows per device per launch, ~1,409 launches
# per propose); raising it is pure re-chunking, results unchanged.
# ⚠ GPU0 peaked at 90,698 MiB of 95,830 in v7 -- step 4096 -> 8192 first.
# GB_FSTAT_NM_LANE_WEIGHTS reweights the CONTIGUOUS row lanes across
# devices (v7: GPU0 39.1% util vs GPU1 72.1%, yet split 50/50, so every
# join waited on the busier card). Integer weights in holder.gpus order,
# e.g. "3,2" = 60/40; unset/blank/malformed => the exact equal split.
# Both are overridable from the submitting shell without editing this
# file, e.g.  GB_FSTAT_CTR_BATCH=8192 sbatch <this script>
: "${GB_FSTAT_CTR_BATCH:=4096}"
export GB_FSTAT_CTR_BATCH
: "${GB_FSTAT_NM_LANE_WEIGHTS:=}"
export GB_FSTAT_NM_LANE_WEIGHTS
export GB_TEMPER_ON_REMOVAL=1      # band swaps run inside rj_prior_removal
# ---- PRUNING IS BACK ON (user ruling 2026-09-26) ---------------------
# 0 -> 1, i.e. back to the CODE DEFAULT and to the original deaths-only
# pruning move: every BIRTH row of this move's RJ step is force-rejected
# (curr_logp = -inf routes it through the existing keep machinery, so it
# never reaches the likelihood kernel) and only DEATHS are judged.
#
# The 0 was inherited, not chosen for v9: a 2026-09-21 hand edit in
# recipe.py for the OLD 6mo run from stored iteration 300 on, promoted to
# a knob on 09-22 with the default left at 1.
#
# WHY BACK: with 0 there was NO dedicated pruning anywhere in the cycle --
# rj_prior_removal proposed births too, and rj_replace is now off. Job 634
# gb_search_1 iteration 1 measured +312 leaves/walker with only 8 cold RJ
# accepts from this move, and walker 0 has carried a split source at
# 19.6682 mHz across several iterations of the previous run.
#
# ⚠ NOT SELF-REVERSIBLE, and that is accepted: "a death-only kernel is not
# self-reversible, so this is NOT a valid MH move on its own -- the USER
# has explicitly waived detailed-balance concerns for search mode. Pair it
# with a birth-capable RJ move in the same stage cycle; never use in PE."
# The condition holds here: rj_warm_search and rj_fstat_search are both
# birth-capable and share every gb_search cycle. PE is unaffected --
# GB_PE_RJ_REPLACE / the pe-named moves do not take this.
#
# ★ SIDE BENEFIT, worth knowing when reading the logs: with deaths-only,
# every accept of this move IS a removal, so [GB_ACCEPT rj_prior_removal]
# "rj cold X/Y" finally reads as a straight cold-removal count. At 0 it
# was births and deaths mixed and there was no way to separate them --
# the rj-split line labels everything "births" even here.
#
# One setting, all four gb_search stages: the move is built ONCE
# (recipe.py) and resolved by name in every stage, and the per-stage
# profile writes only opt_snr / phase_maximize / peak_min_snr. It cannot
# vary per stage.
# ⚠ AND IT CANNOT BE MADE TO, cheaply. rj_removal_only IS read at
# propose time (gbspecialstretch :7295 and four more), so flipping it on
# stage entry would work -- but SearchStageProfileStep._apply_profile
# writes its profile to EVERY GB move in the stage tree, and this
# attribute is not safe to broadcast: on rj_fstat_search, True makes
# apply_inds true (alive rows only => NO births at all, silently
# gutting the search), and rj_removal_only with rj_replace RAISES
# (:4383). A per-stage form therefore needs move-SCOPED profile
# entries, which _apply_profile has no concept of -- real plumbing with
# a loud failure mode, not a table row like GB_SEARCH_3_OPT_SNR.
#
# OVERRIDABLE FROM THE COMMAND LINE (2026-09-29, Mike: "let's also
# adjust the gb search 3 stage prior removal to be not just removal
# only"). The hard-coded value meant an exported 0 could not reach the
# run at all. Default unchanged; the stage-3 test passes 0, and because
# the store resumes INSIDE stage 3 the global scope is not a problem
# for it.
export GB_SEARCH_PRIOR_REMOVAL_ONLY=${GB_SEARCH_PRIOR_REMOVAL_ONLY:-1}
echo "[GB-PRIOR-REMOVAL] GB_SEARCH_PRIOR_REMOVAL_ONLY=${GB_SEARCH_PRIOR_REMOVAL_ONLY} (1 = deaths only; 0 = prior births AND deaths, ALL search stages)"
# ---- the galfor RATCHET (user design 2026-09-30) ----------------------------
# GALFOR_RATCHET=1 on the launch line: gb_search_3 carries ONE gated noise
# proposal at the head of the iteration instead of the rider plus the four
# interleaved slots. Stage-local schedule: iteration 0 of a cycle FORCES the
# galfor coordinates down (DLOG10_* in the sampled log10 basis, every rung and
# walker, published through the noise move's own accept path, hard F-stat
# refit armed, then one GB in-model pass), the next HOLD-1 iterations hold the
# noise fixed, then RELEASE iterations run the ordinary joint noise search
# (each followed by the same in-model pass). CYCLES nudges, then permanent
# release. The release is the measurement: a region of the curve that comes
# back was honest, one that stays down had been absorbing resolvable sources.
# STEP SIZE: one nudge takes the 3-5 mHz total noise from 2-5x the add-back
# estimate to ~0.75-1.5x, where the F-stat peak floor (SNR 6.25) admits true
# SNR-8 sources across the band (lisatools.globalfit.noise_ratchet docstring).
# CYCLES=2 (user ruling 2026-09-30, "do at least 2 cycles for now"). ⚠ The
# second nudge applies the same delta to whatever the first release left: on
# a fit that stayed down it lands at 0.2-0.6x the estimate below 4.5 mHz, so
# read its release with the junk indicators (birth truth-partner fraction,
# power share per band) before going further.
# Overridable here, default OFF = today's composition byte-identical.
export GALFOR_RATCHET=${GALFOR_RATCHET:-0}
# SCHEDULE DEFAULTS = THE PRODUCTION PLAN (user design 2026-10-02, jobs 685/695):
# release-first two-step cycles (nudge, release) with no hold, the CYCLES count a
# ceiling, the data-driven stop (MIN_GAIN) the real end. Until 2026-10-02 these
# four lived only on the launch line (the launcher held the 09-30 3/2/2 trial
# schedule), so a line that dropped one silently fell back to a different run:
# RELEASE_CHECKS to the standalone noise search's 5 flat rounds, MIN_GAIN to
# "never stop". Every value still yields to the environment.
export GALFOR_RATCHET_HOLD=${GALFOR_RATCHET_HOLD:-1}
export GALFOR_RATCHET_RELEASE=${GALFOR_RATCHET_RELEASE:-1}
export GALFOR_RATCHET_CYCLES=${GALFOR_RATCHET_CYCLES:-20}
export GALFOR_RATCHET_RELEASE_FIRST=${GALFOR_RATCHET_RELEASE_FIRST:-1}
export GALFOR_RATCHET_DLOG10_AMP=${GALFOR_RATCHET_DLOG10_AMP:--0.05}
export GALFOR_RATCHET_DLOG10_FK=${GALFOR_RATCHET_DLOG10_FK:--0.10}
export GALFOR_RATCHET_DLOG10_F2=${GALFOR_RATCHET_DLOG10_F2:--0.15}
# A RELEASE runs the psd+galfor search afresh to convergence: RELEASE_CHECKS
# consecutive rounds per walker without a RELEASE_TOL-nat gain ("logL diff of 5
# over a 50-round window"), RELEASE_ITERS the safety ceiling ("we do not want it
# to hit that"). MIN_GAIN: a release whose max cold lnL gains less than this over
# the previous release ends the ratchet (stamped in the store); 0 = never.
export GALFOR_RATCHET_RELEASE_CHECKS=${GALFOR_RATCHET_RELEASE_CHECKS:-50}
export GALFOR_RATCHET_RELEASE_TOL=${GALFOR_RATCHET_RELEASE_TOL:-5}
export GALFOR_RATCHET_RELEASE_ITERS=${GALFOR_RATCHET_RELEASE_ITERS:-5000}
export GALFOR_RATCHET_MIN_GAIN=${GALFOR_RATCHET_MIN_GAIN:-200}
# WHAT ENDS THE RATCHET (user ruling 2026-10-03, "let's adjust away from the
# raw lnL gain"): STOP_RULE=galfor (default) ends it after a release whose
# median-walker released foreground over DROP_BAND (Hz) fell by less than
# MIN_DROP (a fraction) of the previous release's -- each walker against
# itself, so the 3-9 % posterior spread between walkers does not enter. With
# the prior RJ move in births+deaths mode the cold chain relaxes toward the
# typical set (job 706: -360..-920 lnL per walker in one prior-removal leg), so
# the max-lnL gain between releases (STOP_RULE=gain, MIN_GAIN above) can fall
# for reasons unrelated to the foreground; it is still logged. STOP_RULE=off =
# the CYCLES ceiling alone. MIN_NUDGES below applies to every rule.
export GALFOR_RATCHET_STOP_RULE=${GALFOR_RATCHET_STOP_RULE:-galfor}
export GALFOR_RATCHET_MIN_DROP=${GALFOR_RATCHET_MIN_DROP:-0.01}
export GALFOR_RATCHET_DROP_BAND=${GALFOR_RATCHET_DROP_BAND:-3e-3,5e-3}
# THE CLOCK ON A RELAUNCH. CLOCK_RESET=1: the schedule restarts at k = 0 on
# this (re-)entry instead of landing wherever the stage's stored start would
# put it. CLOCK_START = the schedule k the gate runs at NEXT, i.e. what the
# cycle in progress at the cut counts as (START - 1): the last
# "[GALFOR_RATCHET gb_search_3] iteration N: ..." line of the cancelled log
# says RELEASE -> START=1 (the next head is a nudge); NUDGE -> START=2 (the
# next head is a release). 0 = the next head is a release regardless.
export GALFOR_RATCHET_CLOCK_RESET=${GALFOR_RATCHET_CLOCK_RESET:-1}
export GALFOR_RATCHET_CLOCK_START=${GALFOR_RATCHET_CLOCK_START:-0}
# The gain rule (GALFOR_RATCHET_MIN_GAIN) may not stop the ratchet before this
# many nudges have run in the process (user ruling 2026-10-02: "I generally
# want it to do minimum 2 nudges total"). Counted per process: a relaunch
# starts the count again. 0 = no floor.
export GALFOR_RATCHET_MIN_NUDGES=${GALFOR_RATCHET_MIN_NUDGES:-2}
# ---- search LEGS (user design 2026-09-30) ----------------------------------
# GB_SEARCH_LEGS=1: the numbered search stages store one row per LEG of the
# cycle -- after in_model, after in_model_fstat, after in_model_removal --
# instead of one per full cycle. The resume position is the NAME of the
# in-model move the last row was saved after (stored per row), so a gated
# move that skipped a cycle cannot mis-align a relaunch; cadences
# (GB_SEARCH_3_WARM_EVERY) and the galfor ratchet then count CYCLES. With a
# row per leg the mid-iteration checkpoint only guards part of one leg;
# MIDIT_CHECKPOINT=0 is reasonable for a legged search. Default ON since
# 2026-10-02: every 6mo production launch from 09-30 on (jobs 675-695) ran
# legged, and a line that dropped it would resume a legged store un-legged.
export GB_SEARCH_LEGS=${GB_SEARCH_LEGS:-1}
# ---- gb_search_3 ends at the ratchet's stop; then REPLICA PE (user design
# 2026-10-03) --------------------------------------------------------------
# "make gb search 3 only the ratcheting and foreground convergence? And then gb
# search 4 will actually be called 'replica pe' which will wait until we start
# seeing the main large-scale metrics (min, max, mean leaf count; min, max,
# mean logL) reaching a reasonable convergence. The only difference between
# replica pe and full pe is that the other sources (emris, mbhs, sobhbs) are
# still only run every five iterations." full_pe's start_iteration stamp is
# then the start of sample taking. The replica's stop: each of the six series
# must trend by less than LEAF_TOL leaves / LNL_TOL nats over the last WINDOW
# stored rows (a least-squares line, not a range). STAGE_REPLICA_PE=0 restores
# the direct gb_search_3 -> full_pe handover.
export GALFOR_RATCHET_END_STAGE_ON_STOP=${GALFOR_RATCHET_END_STAGE_ON_STOP:-1}
export STAGE_REPLICA_PE=${STAGE_REPLICA_PE:-0}   # flagship recipe (ruling 2026-10-07): "No replica pe" -- gb_search_2 hands over to full_pe
export REPLICA_PE_WINDOW=${REPLICA_PE_WINDOW:-10}
export REPLICA_PE_LEAF_TOL=${REPLICA_PE_LEAF_TOL:-10}
export REPLICA_PE_LNL_TOL=${REPLICA_PE_LNL_TOL:-100}
echo "[REPLICA-PE] STAGE_REPLICA_PE=${STAGE_REPLICA_PE} window=${REPLICA_PE_WINDOW} leaf_tol=${REPLICA_PE_LEAF_TOL} lnl_tol=${REPLICA_PE_LNL_TOL} | gb_search_3 ends at the ratchet stop: ${GALFOR_RATCHET_END_STAGE_ON_STOP}"
# ---- STAGE CLOCKS SURVIVE A RESUME (user ask 2026-09-26; ruling 2026-10-07:
# "keep GF_PERSIST_STAGE_START, but it should reset at the end of a stage") --
# A stage's start iteration is stamped in the store (start_iteration on its
# recipe group) and restored on a relaunch, so the plateau / shutoff / replica
# windows count IN-STAGE iterations across restarts instead of restarting the
# clock at every launch (3mo job 640 -> 644: gb_search_1's earliest possible
# end moved 66 -> 71; a run bounced every few hours never reaches the 41
# in-stage iterations the plateau needs). The stamp is RESET at every stage
# boundary: completed_recipe_step rewrites the next stage's start in the same
# open as the completion, and a stored start past the live iteration (a
# rewound store) is ignored and re-stamped. Code default 0; =0 on the line
# restores the per-launch clock. Compute ranks never touch the store for this
# (their shell engine carries eryn's in-memory backend).
export GF_PERSIST_STAGE_START=${GF_PERSIST_STAGE_START:-1}
# While the ratchet is active the per-(walker, band) RJ shutoff valve leaves
# every band below this frequency OPEN (reopening shut pairs with a fresh
# streak) and shuts converged bands above it as usual; lifted at the
# ratchet's stop so the stage can end on the full valve (user design
# 2026-10-02: "maybe don't do RJ shutoff during the ratchet cycles ... you
# can shut off bands above 7 mHz if their likelihoods converge as usual").
# 0 = the valve acts at every frequency throughout.
export GALFOR_RATCHET_SHUTOFF_MIN_FREQ=${GALFOR_RATCHET_SHUTOFF_MIN_FREQ:-7e-3}
echo "[GALFOR-RATCHET] GALFOR_RATCHET=${GALFOR_RATCHET} hold=${GALFOR_RATCHET_HOLD} release=${GALFOR_RATCHET_RELEASE} cycles=${GALFOR_RATCHET_CYCLES} release_first=${GALFOR_RATCHET_RELEASE_FIRST} dlog10 amp/fk/f2=${GALFOR_RATCHET_DLOG10_AMP}/${GALFOR_RATCHET_DLOG10_FK}/${GALFOR_RATCHET_DLOG10_F2} | release to convergence: checks=${GALFOR_RATCHET_RELEASE_CHECKS} tol=${GALFOR_RATCHET_RELEASE_TOL} cap=${GALFOR_RATCHET_RELEASE_ITERS} | stop: rule=${GALFOR_RATCHET_STOP_RULE} min_drop=${GALFOR_RATCHET_MIN_DROP} band=${GALFOR_RATCHET_DROP_BAND} Hz min_gain=${GALFOR_RATCHET_MIN_GAIN} min_nudges=${GALFOR_RATCHET_MIN_NUDGES} | clock: reset=${GALFOR_RATCHET_CLOCK_RESET} start=${GALFOR_RATCHET_CLOCK_START} | valve floor ${GALFOR_RATCHET_SHUTOFF_MIN_FREQ} Hz (0 = off: rider + 4 interleaved noise slots as before)"
# High-f barren-band birth shutoff (search scope): bands above FMIN with
# AFTER consecutive zero-birth-accept proposes stop proposing births
# (deaths + in-model continue; [GB_BAND_SHUTOFF] log line per band).
export GB_RJ_BAND_SHUTOFF_FMIN_MHZ=10.0
# PATIENCE 50 -> 5 (user ruling 2026-08-28). 5 is the CODE DEFAULT; this
# pin is what disabled the valve, and the audit behind the ruling found
# the valve has NEVER FIRED IN PRODUCTION -- zero [GB_BAND_SHUTOFF] lines
# in the whole v7 run log, because AFTER=50 needs 50 consecutive barren
# iterations of the designated move and that run had only reached ~38. So
# this line is both the behaviour change and the first real exercise of
# the machinery.
#
# The 2026-08-16 replay that raised it to 50 still stands on its facts:
# running the EXACT rule over iterations 5-60 at AFTER=5 switches off 74
# bands above 10 mHz, and 9 of them contain a detectable catalogue source
# the run subsequently FOUND -- SNR 45.7 (band 142, 20.278-20.417 mHz,
# silenced at iteration 18), 35.6 (band 90, iter 14), 34.3 (band 72, iter
# 19), 32.7, 26.9, 24.0, 19.8, 15.0, 12.5 -- with observed
# time-to-first-source of 14-21 iterations. What changed is that replay's
# closing clause, "shutoff is PERMANENT for the process, so there is no
# recovery". It is no longer permanent: the shut-off set and the
# occupancy streaks are now REVIVED on every new F-stat epoch (a refit
# brings a new proposal grid AND an updated noise/foreground profile, so
# a band that was unreachable may now be reachable) and, failing that,
# after GB_RJ_BAND_SHUTOFF_RESET_ITERS iterations. With
# GB_FSTAT_REFIT_EVERY=40 below (was 50 when this note was written), a
# band silenced at iteration 5 is open again by ~40 and then has to
# re-earn its shutoff over a fresh 5-window,
# so those 9 bands get repeated chances instead of one. The cost of the
# short clock is now a DELAY on a genuinely barren-looking band, not a
# permanent loss. Revivals log as [GB_BAND_REVIVE <move>].
# 5 -> 2 (user ruling 2026-09-27). The 5 was sized to outlast the
# progressive leaf-CAP ramp -- a band could be empty because its cap
# had not ramped yet. This schedule has no caps, so an empty band is
# empty on the evidence and two consecutive zero-occupancy iterations
# settle it.
export GB_RJ_BAND_SHUTOFF_ITERS=2
export GB_RJ_BAND_SHUTOFF_SCOPE=search
# Backstop revival (new 2026-08-28): iterations with NO new F-stat epoch
# after which the shut-off set is cleared anyway; 0 disables the trigger.
# Even with no refit the noise model keeps evolving, so a long stretch
# should re-open the question on its own. 100 = 2x the refit cadence
# below, so it only bites if refitting stalls or is turned off.
# 100 -> 10 (user ruling 2026-09-27). This is now a periodic CHECK,
# not an unconditional revival: the shut-off set is reconsidered every
# N iterations and reopened ONLY IF THE NOISE HAS MOVED (see
# GB_RJ_BAND_SHUTOFF_NOISE_TOL, default 1e-3). The check is one
# fingerprint compare, so it can run ten times as often. With the 6mo
# noise PINNED it will never revive, which is the intent.
export GB_RJ_BAND_SHUTOFF_RESET_ITERS=10
# USER RULING 2026-08-28: a shut-off band is frozen "for RJ and fancy
# swaps until it resets". The RJ half is enforced in run_proposal; the
# swap half is this knob (default OFF in code) and it had never been
# exported -- shut-off bands kept being built, scored and swapped in the
# horizontal tempering. ON since 2026-09-11 on the 3mo arm; ported here.
# Safe because shutoff revives on every F-stat epoch (REFIT_EVERY=50) and
# after RESET_ITERS. 6MO TODO (memory note): MEASURE the saving -- read
# temper_cells_filled + run_tempering span vs the unskipped baseline.
export GB_TEMPER_SKIP_SHUTOFF_BANDS=1
# 100 -> 50 (2026-08-18): the refit re-derives the peaks against the LIVE
# residual and the UPDATED foreground/PSD, which is the whole point of
# refitting -- and the foreground converges well inside 20 iterations, so a
# 100-propose cadence spends most of the run on a grid fitted to a
# foreground that no longer exists. ~6% overhead at a 17.7-min fit.
# The peak weighting also flattens to w ~ sqrt(SNR) from epoch 1 onward
# (FSTAT_PEAK_WEIGHT_ALPHA_LATE, default 0.25), so this cadence is also
# when that takes effect.
# 50 -> 30 (2026-09-21, user ruling), alongside the SEARCH live-residual
# refit. The clock counts global-fit ITERATIONS in every stage (2026-09-18
# fix), so this is literally "refit every 30 iterations". The search refit
# no longer restores the walker's GBs before sweeping, so its peak list
# now SHRINKS as sources are found -- each refit should get cheaper, which
# is what pays for the faster cadence. WATCH on the first snapshot:
#   grep '\[peaks\]' globalfit_run.log
# Epochs 0/8/9 were 13513 / 19413 / 19586 peaks, all GB-free. Under the
# change the count should start falling, and stage-B wall time with it
# (epoch 8 was 4054 s).
# 40 -> 10 (user ruling 2026-09-25, SEARCH stages only -- the PE cadence
# below is untouched). The grid goes stale as the search subtracts what it
# finds: every accepted birth changes the residual the peaks were fitted
# against, so a 40-iteration-old grid is aiming births at peaks that have
# already been claimed. That is the same staleness the [GB_ACCEPT rj-split]
# birth-staleness ratio measures (s = d_h/h_h << 1 for a claimed peak).
#
# ⚠ COST: 4x the refits. What is supposed to pay for it is that each refit
# gets CHEAPER as sources are found -- the peak list shrinks. Epochs 0/8/9
# were 13513 / 19413 / 19586 peaks, all GB-free, and epoch 8's stage B was
# 4054 s. If the peak count is NOT falling by the third epoch this cadence
# is 4x a fixed 4000 s bill; check with
#   grep '\[peaks\]' globalfit_run.log
# before committing a long allocation. GB_FSTAT_REFIT_EVERY=40 reverts.
# >>> 10 -> 1 (user ruling 2026-09-25, off job 628): refit EVERY search
# >>> iteration, "because everything is happening so quickly with the
# >>> leaves". Job 628 added 113 then 81 leaves/walker per iteration while
# >>> rj_fstat_search's cold yield fell 135 -> 96 -> 54, i.e. the grid was
# >>> proposing against a residual that had moved substantially since it was
# >>> fitted. 628 never refit once in 7h42m -- it reloaded epoch_0000 every
# >>> iteration.
# >>> ⚠ COST, and it is not small: a full fit was 12.0 min stage A + ~18 min
# >>> stage B + 33 s centres = ~30 min. On an 8307 s iteration that is +22%
# >>> BEFORE the peak list starts shrinking. Dropping rj_replace below pays
# >>> back ~1610 s of it. Watch '[peaks]' and '[stageA]' -- if the peak count
# >>> is not falling by the third epoch this is a fixed ~1800 s/iteration
# >>> bill and should go back to 5 or 10.
export GB_FSTAT_REFIT_EVERY=2      # SEARCH stages: every 2nd iteration
# Which cold walker's residual a SEARCH epoch is fitted to (gb_search_1 and
# 2): rotate = cycle through the walkers by epoch number (code default since
# 2026-10-16, Mike: "let's rotate the fits during search. Not random, but
# cycle through"); min = the 2026-09-21 min-lnL rule. Why: on the 1yr job 774
# the min rule fitted walker 1 on five consecutive search-2 epochs while the
# walker with the fewest leaves never got a grid -- at a few hundred nats of
# spread the min-lnL proxy picks one walker over and over. The epoch line
# prints `rule=rotate`. PE keeps its own knob (GB_FSTAT_PE_REF, random).
export GB_FSTAT_SEARCH_REF=${GB_FSTAT_SEARCH_REF:-rotate}
# 1 -> 2 (2026-09-28, Mike: "let's do fstat refit every two
# iterations"; an intermediate 10 was staged and superseded). Measured on job 662
# iteration 5: the refit cost 1129 s of a 5564 s iteration (20%, epochs
# 11-14 at 1127-1134 s each) and bought ~16 cold births per iteration.
# Cold births are exhausted at this point in the stage (rj_warm 25 of
# 253k proposals, rj_fstat 80 of 236k), so a stale peak list costs
# nothing measurable while the refit costs a fifth of the wall clock.
# Revert to 1 if cold birth acceptance climbs back.
# PE-STAGE CADENCE (user ruling 2026-09-24): 250 elapsed iterations.
# Separate from the search value because full_pe is a RANDOM_CHOICE stage --
# the GB move is drawn roughly one iteration in N, so a search-tuned cadence
# there refits far more often in wall-clock terms than the number suggests.
# ⚠ THE CLOCK'S UNITS CHANGED WITH THIS: it counts ELAPSED iterations now,
# not "iterations this branch was proposed in". In a sequential stage
# (gb_search) the two coincide exactly, so 40 above is unchanged in meaning;
# in full_pe the old clock ticked ~1/N as fast and the knob silently meant N
# times more iterations than it said.
# refit the F-stat birth grid every 150 PE ITERATIONS (the clock counts
# iterations; the decision runs when the move proposes, so the actual
# spacing is its first draw at or after 150) -- user ruling
# 2026-10-02: "during PE fstat should refit every 50 iterations" (was 250);
# back to 250 on 2026-10-07 (Mike: "make it default during pe to updating
# fstat every 250 iterations"); 150 on 2026-10-12 (Mike: "PE fstat fit
# should be based on a randomly chosen walker residual. And we should do a
# refit on that every 150 iterations" -- the walker rule is the
# GB_FSTAT_PE_REF=random code default below, unchanged).
# ⚠ SINCE 2026-10-12 (the TWO PE F-STAT MOVES below) THIS CADENCE BELONGS TO
# rj_fstat_pe_residual, the LIVE-residual move; rj_fstat_pe (the GB-free
# galaxy grid, a 3.5-h epoch on the 9mo: 32,715 peaks at F >= 19.5 over all
# 1232 bands) fits once and never again (GB_FSTAT_REFIT_EVERY_PE_GB_FREE).
export GB_FSTAT_REFIT_EVERY_PE=150
# ---------------------------------------------------------------------------
# TWO PE F-STAT MOVES (user ruling 2026-10-12)
# ---------------------------------------------------------------------------
# Mike: "I would like to do two fstat moves that will split their share of
# the total moves. One fstat move fits the GB-free residual so the whole
# galaxy down to SNR 6.5. [...] The second fstat move will fit the
# gb-inclusive residual every 150 iterations. These iteration counts are the
# overall iterations, so the first time after the overall iterations pass a
# number, it should refit the first time that proposal is chosen. This is not
# a 'number of visits', so within the proposal it should check if it has
# passed the overall iteration cadence and refit." Then: "set the gb-free
# residual refit (with the full galaxy found) to not refitting during full
# pe. It should just fit once the first time it is visiting in full pe. The
# residual fstat will take care of low snr adjustments." And: "this should
# not be specific to each walker, just specific to the one walker that was
# randomly drawn."
#   rj_fstat_pe           GB-FREE residual (the reference walker's cold GBs
#                         restored for the sweep) = the whole galaxy. Same
#                         name, fit dir (gb_fstat_fit/shared) and fingerprint
#                         as before, so a resumed store LOADS the PE epoch it
#                         already holds as its one fit. One random cold walker
#                         (GB_FSTAT_PE_REF, unchanged).
#   rj_fstat_pe_residual  LIVE residual (GB templates subtracted, what the PE
#                         model has not absorbed), ONE random cold walker per
#                         epoch (not the search's min-lnL walker, not a fit per
#                         walker), its own root gb_fstat_fit/shared_pe_residual,
#                         refit every GB_FSTAT_REFIT_EVERY_PE (above) elapsed
#                         iterations, peak floor = full_pe's declared 6.25.
# The galaxy move's cadence: 0 = fit ONCE (its first PE visit fits, or loads
# the epoch already on disk), never refit.
export GB_FSTAT_REFIT_EVERY_PE_GB_FREE=0
# The galaxy move's OWN F-stat peak floor, SNR (F = SNR^2/2 = 21.125). NOT part
# of the epoch fingerprint: a FIT selects its peaks at it; a LOAD of an epoch
# selected at a lower floor (the 9mo's PE epoch_0000, F >= 19.53) keeps only
# its cached peak boxes with F >= 21.125 -- exactly the catalog re-selecting
# the cached comb at 6.5 gives (candidates are floor-independent, the per-band
# cap keeps the top F), with no refit. 'none' = full_pe's declared floor.
export GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE=6.5
# The residual move's share of GB_PE_RJ_FSTAT_FRACTION. 0.5 -> 0.9 (user
# ruling 2026-10-18): on the 9mo job 773 the GB-free galaxy grid accepted 0.4
# cold births per draw (cleared out) against 2.5 for the live-residual grid,
# whose yield did not decay over the 160 rows its grid aged, so the F-stat
# share goes to the residual move (0.45 -> 0.405 residual / 0.045 galaxy);
# warm 0.4 above, prior the remaining 0.15. 0 or 1 composes ONE F-stat move.
export GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=${GB_PE_RJ_FSTAT_RESIDUAL_SPLIT:-0.9}
# GB_FSTAT_FORCE_REFIT=1 on a launch line now refits BOTH PE grids (one new
# epoch per root, the 3.5-h galaxy included) -- leave it off a PE resume.
# WHICH WALKER A PE REFIT FITS (code default, NOT exported; user ruling
# 2026-10-06): GB_FSTAT_PE_REF=random (default) fits each PE epoch to a
# uniformly RANDOM cold walker with a finite lnL instead of the min-lnL one --
# in PE every walker is a posterior sample, and refitting the min-lnL walker
# every epoch shaped the birth grid around the ensemble's worst state. The
# draw is seeded by (GB_FSTAT_PE_REF_SEED, default 0, epoch), so a restart
# or another rank re-derives the SAME walker for an epoch (checkpoint
# fingerprint, centre table on a load). GB_FSTAT_PE_REF=min restores the
# min-lnL rule in PE. Search refits always take the min-lnL walker, and the
# per-propose distance-birth centre keeps the min in both.
# GB_FSTAT_PE_REF=max and GB_FSTAT_FORCE_REFIT=1 (refit on a restart) are
# ONE-OFF launch-line knobs (user ruling 2026-10-07, V9-30): never defaults
# here -- the PE refit walker stays the code default, random.
export FSTAT_PEAKS_PER_BAND=200    # per-sub-band peak cap (code default; explicit)
# STAGE-B STACK CHUNKING -- fixes the 2026-09-21 epoch-9 OOM.
# StackedFStatProposal4D.__init__ corner-averages the ENTIRE K-box 4-D grid
# in ONE allocation; unset, k_chunk = K and nothing is chunked. Epoch 9
# (19586 peaks, the largest of the run) asked for 1,708,159,488 bytes on a
# 91.5 GB resident baseline and died. This feeds mem_budget_mb -> k_chunk:
# SAME grid, SAME result, pure chunking, one-time cost, then cached to
# <epoch>/..._peaks_stacked.npz so resumes skip it. 512 rather than the
# 1yr run's proven 1500 because this baseline is tighter. Lower it further
# if it still OOMs; also consider the MEMPOOL_FREE=1 levers.
export FSTAT_GRID_MEM_MB=512
# ---------------------------------------------------------------------------
# STAGE-B SKY GRID: f0-ADAPTIVE (2026-09-24, new this run)
# ---------------------------------------------------------------------------
# WHAT WAS WRONG. Stage A (the comb) has always sized its sky grid from the
# Doppler ridge, n ~ (f0 Tobs v/c)^2, adaptive over 16..512 nodes -- the log
# line is "[comb] ... sky ADAPTIVE [16..512] over 6 level(s)". Stage B, which
# REFINES stage A's peaks, was pinned at a fixed 8 x 8 = 64 full-sky nodes at
# every frequency. So above ~6 mHz the refinement was 4-8x COARSER on the sky
# than the scan that produced its own input, and stage B is the component the
# birth proposal actually draws from.
#
# Worse, of those 64 nodes only 44 were distinct directions:
# linspace(0, 2*pi, 8) repeats the same longitude at both ends, and every
# longitude collapses to one point at sin_delta = +-1. 31% of the sweep
# re-derived values it already had, and the poles were over-weighted by the
# collapse. Stage B now builds alpha half-open and picks the best rectangular
# factorization of its node budget (measured: near-square, 9x7 / 13x10 /
# 18x14 / 26x20), so 95th-percentile miss-to-nearest-node goes 0.399 ->
# 0.368 rad at the same 64 nodes, and 0.187 rad at 252.
#
# WHY IT MATTERS HERE. The 7.444 and 7.599 mHz misses sit exactly in the
# 5-11 mHz range where this run's peak boxes are dense and the old grid was
# 2-4x too coarse. Those groups are the ones that gain most below.
#
# NO NODE CAP (user ruling 2026-09-24): FSTAT_STAGEB_NSKY_MAX stays 0. The
# limit is a per-group BYTE budget instead, which is the right control
# because what breaks is memory and memory goes as
# n_boxes * n_f0 * n_Mc * n_sky -- not as n_sky. Uncapped-with-no-budget on
# the snapshot-19 peak list asks for 45.8 GB of grid with a single 27 GB
# group at 15-20 mHz (438 boxes but n_Mc = 96), and this run already died
# once on a 1.7 GB allocation against a 91.5 GB resident baseline, which is
# what FSTAT_GRID_MEM_MB above exists to work around.
#
# THE BUDGET SELF-ADJUSTS, which is why it beats a node cap: a group's grid
# shrinks as its peak count falls, so as the search subtracts the high-f
# sources the freed bytes are spent on sky resolution there automatically,
# with no knob change. Epoch 0 is the expensive one.
#
# 2.0 GB holds the largest group at 1.82 GB -- within the 1.71 GB this run
# has already survived -- for 1.80x today's stage-B node count. Projected
# per-group sky at epoch 0 (from the snapshot-19 group census):
#     0.57- 5.14 mHz  11630 boxes  nMc  3   sky  64 ->  63   1.39 GB
#     5.14- 6.11 mHz   2447 boxes  nMc  3   sky  64 -> 130   0.60 GB
#     6.11- 7.27 mHz   1547 boxes  nMc  6   sky  64 -> 130   0.76 GB
#     7.28- 7.57 mHz    393 boxes  nMc  6   sky  64 -> 252   0.38 GB
#     7.57- 9.24 mHz   1754 boxes  nMc 12   sky  64 -> 130   1.73 GB
#     9.32-10.29 mHz    435 boxes  nMc 24   sky  64 -> 252   1.66 GB
#    10.29-11.25 mHz    432 boxes  nMc 24   sky  64 -> 252   1.65 GB
#    11.25-13.17 mHz    514 boxes  nMc 43   sky  64 -> 130   1.82 GB
#    15.36-20.38 mHz    438 boxes  nMc 96   sky  64 ->  63   1.67 GB
# Above 13 mHz the sky stays put because n_Mc = 96 eats the budget there.
# Those sources are few and loud; they are the ones the shrinking peak list
# will free up first.
#
# ⚠ THIS FORCES A REFIT. The stage-B cache stores the grid, so an epoch
# fitted at 8x8 cannot be reused under a different sky. Start from a clean
# gb_fstat_fit/ or the loader reuses the old 64-node grids silently.
#
# WATCH on the first snapshot -- one line per group, and the step-downs:
#   grep '\[stageB\] group' globalfit_run.log
#   grep 'stepping the sky down' globalfit_run.log
#   grep 'VARIES BY GROUP' globalfit_run.log
# ESCAPES, in order of bluntness:
#   FSTAT_STAGEB_GROUP_MAX_GB=1.7  hold every group at today's largest
#   FSTAT_STAGEB_NSKY_MAX=128      hard node cap on top of the budget
#   FSTAT_STAGEB_SKY_ADAPT=0       back to the fixed 8x8 exactly
#   FSTAT_N_ALPHA / FSTAT_N_SINDELTA   setting EITHER also pins the sky
export FSTAT_STAGEB_SKY_ADAPT=1
export FSTAT_STAGEB_NSKY_MIN=64      # never coarser than the old fixed 8x8
# ---- FLAGSHIP: cap the stage-B sky grid at the 6mo's 8-mHz node count -------
# The adaptive sky grid asks for ~(f0 Tobs v/c)^2 nodes per peak: at 6 mo that
# is 64 (< 4 mHz) / 256 (8 mHz) / 1024 (15 mHz); at 9 mo 128 / 512 / 2048 and
# at 1 yr 256 / 1024 / 4096, on top of the fdot axis (13 -> 28 -> 48 nodes at
# 8 mHz) and the f0 axis (79 -> 96). The 6mo stacked grid was ~12 GB raw and
# 24-35 GB resident per GPU, so uncapped the 9mo grid lands at 2-3x that and
# the 1yr at ~8x (the 1yr v8 OOMed on its first F-stat search). 512 never
# touches a peak below 8 mHz at 9 mo (their natural counts are <= 512) and
# bounds the > 8 mHz tail; the host-mapped float32 grid (TODO) is the real
# fix, and at 1 yr the fdot axis alone needs it. FSTAT_STAGEB_NSKY_MAX=0 on
# the line restores the uncapped grid.
export FSTAT_STAGEB_NSKY_MAX=${FSTAT_STAGEB_NSKY_MAX:-512}
export FSTAT_STAGEB_GROUP_MAX_GB=${FSTAT_STAGEB_GROUP_MAX_GB:-1.0}   # flagship (9mo pattern): halve the stage-B per-group transient (batching only, no grid change; the 6mo v9 ran 2.0)
# BIRTH-DRAW ALLOCATION (2026-08-16). Peak boxes are weighted w ~ F**alpha,
# and the F-statistic goes like SNR^2 -- so the historical alpha=1 hands an
# SNR-10 source 9x FEWER birth attempts than an SNR-30 one, exactly
# backwards. Measured at iteration 15: 80.3% of the birth mass landed on cap
# cells whose source was ALREADY found (median peak F 125.5) against 7.8% on
# cells holding an unfound detectable source (median F 26.6), while 146 of
# 332 cells with a detectable source held no leaf at all. alpha=0.5 makes
# draws ~ SNR instead of SNR^2 -- still preferring real signal, without
# starving the faint tail by the square. Predicted redistribution: unfound
# cells 7.8% -> ~20% of the birth mass (2.6x).
# NO REFIT NEEDED: the weights are applied when the birth proposal is built
# FROM the cached stage-B grids and are not persisted in them, so a restart
# picks this up against the existing epoch cache.
# alpha=1 restores the previous behaviour bit-identically.
export FSTAT_PEAK_WEIGHT_ALPHA=0.5
# HIERARCHICAL BIRTH DRAW: pick a STRATUM uniformly, then draw within it
# with w ~ F**alpha. K = strata PER SUB-BAND.
#
# K 32 -> 1 (user ruling 2026-08-29): "the fstat proposal should probably
# just group by sub-band". ONE STRATUM PER SUB-BAND. Only one RJ source is
# proposed per sub-band per round (serial-within-band scheduling), so the
# sub-band is the unit a birth actually competes in -- "those are the
# effective rj limits as well". Sub-dividing it over-served the sparse
# parts of a sub-band relative to the dense ones, for a partition that
# corresponds to nothing physical: at K=32 a stratum was 1.7361e-5/32 =
# ~4.2 FD bins, far narrower than a source's ~+-16-bin support, and the
# recorded epoch-0 check (median 3 peaks per occupied cell, max 21) means
# the draw was close to flat-per-peak. At K=1 the anti-starvation motive
# is still met -- mass is equalised across the 1232 occupied SUB-BANDS --
# while the F**alpha preference now does real work among the many peaks
# inside each one, which is what it was for.
#
# NOT A COST KNOB. rj_fstat_centers (61% of an iteration) is a per-row
# solve and does not know this exists; K only sets the birth PROPOSAL
# density.
#
# The strata are derived from band_edges (2026-08-29). They used to be
# np.linspace(be[0], be[-1], nb*K+1) -- identical on this uniform band
# grid, but equal to the sub-bands only by luck and to the STAGGERED cap
# cells never, despite the old comment above calling them cap cells. The
# env default also no longer tracks GB_CAP_DIVISOR: a cap knob must not
# silently re-stratify the RJ birth proposal.
#
# Set to 0 for the historical global w ~ F**alpha mixture.
# Implemented as flat composite weights, w_j = (1/N_occupied) *
# F_j**alpha / sum_stratum(F**alpha), so StackedFStatProposal4D's rvs AND
# logpdf stay mutually exact by construction (both read self.weights) --
# the RJ acceptance ratio is never at the mercy of two implementations.
export FSTAT_PEAK_WEIGHT_CELLS=1
# Slab 5 (user ruling): measured-safe (+-1 layer holds >=1-1e-7 of tone
# energy; 5 = 2x that need) and ~30%% smaller band buffers than the AUTO 7.
# Smoke 2 exonerated the slab as the VGB [GB_CELL_LL] growth cause (growth
# persisted at slab 7).
export GB_WDM_BAND_SLAB_LAYERS=5
# 3-D Milky Way (dist, alpha, sin_delta) joint prior (user ruling: the
# proper density, not the flat placeholder). Chirp-mass basis + the
# astrophysical f0-Mc GMM prior are already the code defaults. NOTE:
# this knob was NOT exercised in the smokes (they ran the uniform
# placeholder); detailed balance holds either way -- births still draw
# dist from the birth container and the prior enters through logp.
export GB_USE_GALAXY_PRIOR=1

# ---- NOISE (psd + galfor) internal repeats: 50 -> 10 (user ruling
#      2026-08-15) ---------------------------------------------------------
# Each PSDMove.propose runs num_prop_repeats internal MCMC repeats, and each
# repeat scores the whole (ntemps x nwalkers) ladder -- one batched build per
# distinct walker. At 50 repeats that is ~660 batched covariance
# build+score calls per move propose, ~1320 per iteration across psd_pe +
# galfor_pe, which is what made the noise block ~43-44 s/iteration. The
# noise model is only 4 (psd) + 5 (galfor) parameters and it converges long
# before 50 repeats, so 10 buys back ~5x of that block for very little
# mixing. WATCH on the first snapshot: the psd/galfor acceptance +
# parameter traces (artifact panels) and whether the noise still tracks the
# injection -- if the chains look under-mixed, 20 is the next notch.
# Combines with today's de-sync work (gated debug guard, sync-free
# sanitization, same-device repack), which cuts the cost of EACH call.
export PSD_NUM_PROP_REPEATS=10
export GALFOR_NUM_PROP_REPEATS=10
# The noise eigen tables' opt-in {PSD,GALFOR}_EIGEN_INFO=fisher route is
# documented with the source branches' gram route (search "OPT-IN: EIGEN
# TABLES" below); default unchanged (ll).

# ---- VGB in-model scorer: sig-het ON (user ruling 2026-09-17), same engine
#      and the SAME knobs as the GB branch by construction (VGBSettings reads
#      the shared SIGHET_* / GB_SIGHET_* env names: windowed refs, N_CP,
#      trust gate thresholds, n_sparse_fd). Was pinned to 0 ("accuracy at
#      the loudest-VGB SNRs unverified -- [GB_CELL_LL] growth in smoke 1"),
#      which routed the per-propose VGB information matrices through the
#      CHUNKED engine at 30-45 ms/source: vgb_pe 130 s per call, the
#      noise_vgb_search stage 131 s/iteration vs 24 s on the pre-basis-change
#      legacy run. WATCH on the first snapshot: [GB_CELL_LL vgb_pe]
#      |sampled-actual| must stay ~1e-7 (growth = the sig-het error at loud
#      VGBs), [GB_TRUST] rejection fraction for vgb_pe (the trust gate is
#      what protects the chain), the ll-drift rebuild count, VGB acceptance.
#      VGB_SIGHET_INMODEL=0 restores the exact chunked scorer.
export VGB_SIGHET_INMODEL=${VGB_SIGHET_INMODEL:-1}
# VGB RELAUNCH BLOCK (2026-08-15, user rulings). The VGB likelihood was
# OFF all run (betas=[1e-4] bug) and 36/55 leaves were frozen by the GB
# SNR gate -- both fixed in code (76cd3237); pre-fix VGB samples are
# prior-only. Migration 1 in the header checklist is REQUIRED for the
# VGB_NTEMPS=8 ladder below.
# PARAMETERIZATION: the 6-dim CHIRP basis
# [dist, phi0, cos_iota, psi, Mc, fdot_astro_ratio] -- Mc SAMPLED, only
# f0 / alpha / sin_delta left as per-leaf fills.
#
# USER RULING 2026-09-16, superseding BOTH the 08-15 "revert to the old
# VGB parameterization" ruling and this script's v8-parity exact-copy
# pin at 0: "The VGBs should now (with stretch removed) have the exact
# same mechanics as the GBs except f0 and sky fixed" -- Mc AND
# fdot_astro_ratio both sampled -- and "really mirror the GBs as much as
# possible ... VGBs get the ridge-gibbs fiber move too".
#
# WHAT THIS FIXES: on the old 5-dim basis fdot_astro_ratio was a
# COLLAPSED dimension (truth exactly 0 x multiplicative init = zero
# spread, and the affine-invariant stretch cannot create spread it never
# had). The chirp basis un-collapses it: the nonzero catalogue Mc gives
# the multiplicative init real spread, and the zero-truth ratio column
# gets the documented ADDITIVE init exception
# (VGB_RATIO_INIT_WIDTH x GB_FDOT_ASTRO_RATIO_MAX, still scaled by
# VGB_START_FACTOR -- so START_FACTOR=0 below still starts at exact
# truth).
#
# STANDALONE VALIDATION (HM Cnc smoke, 2026-09-16):
#   * installed VGBObservableBasis log_jacobian EXACT vs analytic;
#   * observable + eigen=full with fdot-weight 0 freezes Mc BY
#     CONSTRUCTION (the fiber is exactly flat to the info matrix);
#   * obs + eigen fw=0 + RIDGE-GIBBS was the BEST arm, cold acceptance
#     0.593 -- the ridge move is what unfreezes Mc, so the two knobs
#     ship together;
#   * McRatioDistFiber works VERBATIM on the VGB layout: it resolves
#     dist / Mc / fdot_astro_ratio by NAME and never reads f0.
# The vgb_ridge_gibbs move registers AUTOMATICALLY off this flag --
# recipe.build_vgb_moves gates on the basis carrying those three column
# names, and run_combined_staged.py requests it in gb_search + full_pe
# exactly where gb_ridge_gibbs rides. GB_RIDGE_GIBBS=0 kills both.
#
# *** LOUD CAVEAT -- NOT RESUME-COMPATIBLE. ***
# This takes the vgb chain from ndim 5 to ndim 6. An existing store
# CANNOT be resumed across the flip: run.py refuses it at backend
# construction (before any chain load) with a message naming
# VGB_CHIRP_MASS_BASIS. So this requires EITHER a FRESH store, OR a
# vgb-branch migration with
#   python scripts/fstat_proposal/migrate_vgb_chirp_basis.py \
#       <store.h5> <catalogue_dir>
# run BEFORE the first launch. NEVER flip this mid-store.
# LIVE (user ruling 2026-09-18). Staged off since 2026-09-16; flipped on
# after the 4-GPU run measured VGB in-model acceptance at 0.001-0.004 per
# rung under the 5-column basis, with NO vgb_ridge_gibbs registered (the
# ridge gate reads the column names, and the 5-column basis does not carry
# them). The HM Cnc standalone exonerated the map three ways and ranked
# the arms: 6-col observable+eigen with the RIDGE 0.534 acceptance
# (tau 66.6, the best arm) against 0.1596 without it. This run has been
# getting neither half.
#
# *** BEFORE THE NEXT LAUNCH ON AN EXISTING STORE, MIGRATE IT. ***
#   python scripts/fstat_proposal/migrate_vgb_chirp_basis.py \
#       ${STORE_DIR}/${BASE_FILE_NAME}_testing.h5 <catalogue_dir>
# Without that the launch REFUSES to resume (run.py's ndim guard fires at
# backend construction, before any chain load, naming this knob) -- a
# clean stop, not a corrupt run. Set 0 to roll back to the 5-column store.
export VGB_CHIRP_MASS_BASIS=${VGB_CHIRP_MASS_BASIS:-1}
# 8-rung ladder (user ruling 2026-08-15). Resume derives the rung count
# from the STORED band_temps shape, so the migration above MUST be run
# with the matching "8" argument (it recreates every rung-dimensioned
# vgb dataset: temps ladder, counters zeroed, 7 swap pairs).
export VGB_NTEMPS=8
# VGB IN-MODEL PROPOSAL = OBSERVABLE + EIGEN=FULL. SUPERSEDES the 09-15
# morning "=eigen" arming (user ruling, same day, later: "we should be
# sampling the VGBs just like the GBs now (in terms of the basis/proposal
# type not RJ, still fixed model. f0 should be filled. sky location should
# be filled. Everything else is just like the GB setup. We should be
# sampling still in the observed basis for VGBs (just without f0, sky
# coords)"). Both are the CODE defaults now; pinned here anyway because a
# proposal swap is exactly the change a runbook must state.
#
# WHAT IT IS: the SAME composite step the GB branch runs, with the reduced
# map swapped in -- a symmetric draw in z = [lnA, fdot, phi0, cos_iota,
# psi] (f0 / alpha / sin_delta / Mc pinned per leaf; no f_mid because a
# pinned f0 has no shear, no Mc fiber because (dist, r) -> (A, fdot) is
# 2->2), whitened by the per-block information-matrix eigenbasis in z
# (=full: one joint correlated step per repeat, all coordinates moving).
# factors are the map's log-Jacobian difference.
#
# WHY IT SUPERSEDES =eigen -- the r-column verdict. The =eigen draw was in
# the SAMPLING basis, where the ONLY physical quantity fdot_astro_ratio
# drives is fdot = fdot_gr(f0, Mc)(1+r); with Mc a per-leaf FILL, Mc
# occupies the physical fdot output slot through the container key_map, so
# fdot sits OUTSIDE test_inds and r's only scored target is the fddot slot
# -- which McDistFdotAstroQuad emits as exactly f0*0.0. Measured on the
# real stock container: J[:, :, r] == 0 EXACTLY, info_y rank-deficient,
# and the r eigen step set by the prior box (U[-5, 5]) instead of by
# curvature. That is a blind jump in the one coordinate a known-f0 /
# known-sky branch exists to measure, and every VGB run before this one
# had it (commit 2bb484e2 recorded the suspicion; it is now confirmed and
# fixed -- _infomat_phys_inds asks the engine for the live fdot slot).
#
# REQUIRED by the exact-truth flow above: VGB_START_FACTOR=0 gives a
# zero-spread ensemble that a stretch draw could never move; the
# observable step comes from the map + step scales, not the spread.
# COST (accepted, unchanged): VGB_SIGHET_INMODEL=0 above routes the
# matrices through the CHUNKED engine at ~29-46 ms/instance x (55 leaves
# x 8 rungs x 10 walkers ~ 4.4k instances) ~ 2-3.5 min of factor builds
# per vgb propose; validating VGB_SIGHET_INMODEL=1 at the loudest-VGB
# SNRs drops that ~12-19x (~11-17 s) and remains the standing
# optimization.
# FIRST CLUSTER EXPOSURE of the vgb observable path -- watch the first
# vgb_pe [GF_TIMING] wall, any "observable-basis proposal unavailable"
# or "falling back to the stretch proposal" warnings, and the vgb fdot
# spread actually opening up. Escapes: VGB_INMODEL_PROPOSAL=eigen (the
# sampling-basis one-axis draw, now with a live r column) and =stretch
# (the bit-identical legacy).
export VGB_INMODEL_PROPOSAL=${VGB_INMODEL_PROPOSAL:-observable}
# OVERRIDABLE (2026-09-19). Measured, 4 walkers, full_pe:
#   VGB  obs_basis  cold 0/47 (0.0000)  all 0/376 (0.0000)   <- EIGEN=full
#   (the 10-walker arm ran VGB_INMODEL_PROPOSAL=stretch and got
#    cold 182/470 = 0.387, all 1512/3760 = 0.402)
# The VGB in-model move is frozen: 47 cold proposals a propose, zero
# accepted. Same failure as the GB twin (GB_INMODEL_OBSERVABLE_EIGEN=full
# -> obs_basis cold 0.0139 against the 10-walker arm's 0.0864) and worse,
# because VGB fixes f0 and sky, so the information matrix keeps only the
# degenerate directions and a larger fraction of its eigenspectrum is
# ill-conditioned. A "full" draw is JOINT over every axis, so one railed
# direction (PSD-projected eigenvalue -> whitened sigma at
# *_OBSERVABLE_EIGEN_SMAX, default 10) contaminates EVERY proposal --
# unlike "axis", which spends only 1-in-ndim repeats on a bad direction
# and is what the healthy mbh/emri/sobbh EigenAxisMove(mode="axis") uses.
# Empty = the diagonal observable draw, bit-identical fallback.
# NOTE stretch (the 10-walker arm's setting) is NOT reachable at 4 walkers
# on 4 ranks: its red/blue pairing is within the local walker block, which
# needs B >= 2 and even. That is why this run moved VGB to observable.
# DEFAULT CHANGED full -> axis (user ruling 2026-09-19), same reasoning as
# the GB knob above and more urgent: VGB measured cold 0/47 = 0.0000 under
# "full" -- frozen, not slow. VGB fixes f0 and sky, so its information
# matrix keeps only the degenerate directions and a larger share of its
# eigenspectrum is ill-conditioned, which is why the joint draw tips from
# bad to zero here. Escape: VGB_INMODEL_OBSERVABLE_EIGEN= (diagonal draw).
export VGB_INMODEL_OBSERVABLE_EIGEN=${VGB_INMODEL_OBSERVABLE_EIGEN-axis}
# VGB's own step scale, split from GB's 2026-09-19. One shared knob used to
# scale BOTH branches; in the same propose GB was over-accepting at 0.71
# while VGB sat healthy at 0.26-0.43, so GB's correction is the last thing
# VGB wants. 1.0 = unchanged (it is also the code default).
export VGB_INMODEL_OBSERVABLE_JUMP=${VGB_INMODEL_OBSERVABLE_JUMP:-1.0}
echo "[VGB-OBS-EIGEN] VGB_INMODEL_OBSERVABLE_EIGEN='${VGB_INMODEL_OBSERVABLE_EIGEN}' (empty = diagonal draw)"
# GB rung count. 24 is already the code default (stock/erebor/gb.py
# env_default("GB_NTEMPS", 24)) -- pinned here anyway because the rung count
# is the one knob whose failure mode is completely silent: resume derives it
# from the STORED band_temps shape, so a store built at the wrong count runs
# the wrong ladder forever while the script still says 24, and the only hint
# is a single build_gb_moves warning buried in a 200k-line log. Both confined
# probes ran a degenerate [1.0, 1e-4] ladder for days on exactly that. The
# LADDER PREFLIGHT below turns the silent case into a refusal to start.
export GB_NTEMPS=24
# ---- GB LADDER RE-SPACING (user ruling 2026-10-07: "Set these values in the
# v9 script") ----------------------------------------------------------------
# Measured on 6mo job 650 (and the 3mo run): the stock ladder is geometric at
# 1.20 from T0 to T22 (beta 1.765e-2, T 56.7), then T23 is PINNED at beta
# 1e-4 (T 10,000) -- a 176x gap that holds 258 sources against T22's 3,731
# and couples at 15.4 %, while every other adjacent pair couples at 31-71 %
# (the hot half is spaced far too finely; replica exchange wants ~20-30 %).
# Sources/walker against ln(T) over T0..T22 reaches zero near T ~ 1,460, so
# ~a factor of 26 of usable temperature is skipped before the pinned rung
# overshoots it 7x. First step of the two-step plan (1.35, measure the
# acceptance profile, then ~1.5): T0-T10 stay at 1.20 (T10 = 6.19), T11-T23
# at 1.35 -> T23 ~ 306 (beta 3.3e-3), and the top rung is no longer pinned
# (GB_LADDER_PIN_LAST=0 leaves it on the progression; 1.5 would land ~1,205).
# ⚠ TAKES EFFECT ON A FRESH STORE ONLY (the 9mo run): band_temps lives in the
# store and initialize_band_information restores it, and the LADDER PREFLIGHT
# below compares the rung COUNT (24, unchanged), not the values -- so on the
# resumed 6mo store these three are a silent no-op by design (no migration).
# Watch on the 9mo's first tar: [GB_VERT] adjacent swap acceptance per rung
# pair (want 20-30 % across the hot half, no 15 % cliff at the top) and the
# sources-per-walker profile over the ladder.
export GB_LADDER_HOT_RATIO=${GB_LADDER_HOT_RATIO:-1.35}
export GB_LADDER_HOT_FROM=${GB_LADDER_HOT_FROM:-10}
export GB_LADDER_PIN_LAST=${GB_LADDER_PIN_LAST:-0}
# Concurrent per-device shard dispatch (code default since 2026-08-13;
# explicit here for the run record). =0 restores serial dispatch if the
# drift/[GB_CELL_LL] checks ever implicate concurrency.
export GB_ROUTER_THREADED=1

# ============================================================================
# CHANGE 1 OF 3 vs 3mo_v8 -- WARM START (the between-run refit proposal).
# rj_warm_search: the 3mo_v8 posterior, clustered offline into Gaussian
# components, as an RJ birth distribution IMMEDIATELY BEFORE
# rj_fstat_search in gb_search. Built exactly like fstat_search with the
# distribution switched (user spec 2026-09-02): phase-max follows the
# GB_RJ_PHASE_MAXIMIZE=1 exported above; run_swaps/leaf_cap_update stay
# off (cycle invariants -- fstat_search owns tempering + cap counters).
#
# BUILD THE NPZ from the MOST RECENT 3mo run (user ruling 2026-09-14:
# the refit proposal rides the eigenbasis test and is based on the most
# recent 3-month data run) -- that is the 10-WALKER SCIENCE arm this whole
# file is rebased on (gf_prod_3mo_v8_10walkers, the GB_SCIENCE_465
# relaunch of 2026-09-11, jobs 465->479 lineage), NOT the old 24-walker
# gf_prod_3mo_v8 store. Use the FULL FINAL store h5 (NEVER the
# make_snapshots tars -- their chain slabs are keep-window extracts and
# the fitter would warn + fit on ~3 iterations):
#   python -m lisatools.globalfit.warmstart.fit_from_store \
#       --store <...>/gf_prod_3mo_v8_10walkers/gf_prod_3mo_testing.h5 \
#       --last-k 10 --tobs 7776000 --out v8_10w_last10.npz
#   python -m lisatools.globalfit.warmstart.match_referee \
#       --npz v8_10w_last10.npz --store <same h5>
#   python -m lisatools.globalfit.warmstart.referee_apply --fit v8_10w_last10.npz \
#       --referee v8_10w_last10_referee.npz \
#       --out gf_prod_3mo_v8_10w_refereed.npz
# (the machinery was audited end-to-end on the v7 final store,
# 2026-09-02: p-accounting exact, 93-95% of p>0.9 comps on real
# sources, logpdf finite at every leaf under the production floor box.)
# The preflight below REFUSES to start while the npz is missing, so the
# refit proposal cannot silently drop out of the run; explicitly-empty
# GB_WARM_START_COMPONENTS= is the only way to run WITHOUT the warm move
# (stage lists bit-identical to 3mo_v8's).
# The npz LIVES INSIDE THE RUN STORE (user ruling 2026-09-14: "The
# warmstart store should be within the 6mo folder") -- so snapshot zips
# of ${STORE_DIR} capture it, a resume finds it, and a fresh store
# (rm -rf) rebuilds its own. The fit/referee intermediates land next to
# it automatically (same directory as the target).
# The separator is EXPLICIT (2026-09-18). The slash used to be missing, and
# that only lands inside the store when STORE_DIR carries a trailing slash --
# the built-in default does, but every override on the command line does not,
# so `STORE_DIR=/.../gf_prod_6mo_v8_4gpu ./submit...` silently put the npz in a
# SIBLING directory gf_prod_6mo_v8_4gpuwarmstart/, outside the snapshot zips
# and invisible to a fresh-store rebuild. Every other use of STORE_DIR in this
# script already writes ${STORE_DIR}/..., so the doubled slash the default
# produces is the same one they produce, and harmless.
export GB_WARM_START_COMPONENTS=${GB_WARM_START_COMPONENTS-${STORE_DIR}/warmstart/gf_prod_18mo_flagship_refereed.npz}
# AUTO-BUILD (2026-09-14, after the first launch died on the missing npz;
# user: "Check if it is done, if not run it. I would like it to be
# automatic."). When the npz is missing, recipe build now runs the
# fit -> referee -> apply pipeline ITSELF from this store, IN-PROCESS
# (the three scripts are imported and their main() called -- no
# subprocess; one MPI rank builds under a lock, the others wait; watch
# the [WARMSTART-BUILD] lines). SOURCE_TOBS is the SOURCE store's Tobs
# (3 months), not this run's -- the proposal container rescales to the
# run Tobs at load.
# ============================================================================
# V9-12 -- ONE SEED STORE for BOTH the warm start and the noise pin.
#
# USER RULING 2026-09-24: "make sure the warmstart and PSD/GB frozen start
# point come from the same folder (3mo 10 walker noise fix). It should use
# maxlogL for PSD/GB."
#
# GF_SEED_STORE is the single source. The warm-start mixture is fitted from
# it (GB_WARM_START_SOURCE_STORE below) and the psd/galfor start pin is read
# from its maxlogL cold walker (the V9-12 block further down). Overriding one
# without the other is what this variable exists to prevent.
#
# THE SEED IS THE 3mo 10-WALKER RUN. This default was FIRST written as a
# `_noisefix` sibling of that directory (user, 2026-09-24, prefixed "I
# think") and the first v9 launch -- job 616, 2026-09-24 19:52 -- died on
# it before reaching python: no such directory exists under
# /shared/data/global_fit_output. The script's own candidate scan proved it
# in the same log, listing eight gf_prod_3mo* runs and no _noisefix among
# them. User ruling on those forensics, 2026-09-25: "yea it is just
# gf_prod_3mo_v8_10walkers".
#
# ⚠ scripts/diagnostics/gf_monitor_gen_lean.py calls THIS directory the
# PRE-noise-fix arm, and that is still true. What makes it acceptable as
# the v9 seed is that galfor no longer comes from the store at all --
# GALFOR_START_PARAMS below is the offline 3mo estimate (user ruling
# 2026-09-24, "just for now") -- so the store supplies only
# PSD_START_PARAMS and the GB warm-start mixture, and that mixture is the
# very one v8's own 6mo run was launched on.
#
# Override at launch:
#     GF_SEED_STORE=/shared/data/global_fit_output/<dir>/<store>.h5 \
#         ./scripts/flagship/submit_gf_24mo.sh
#
# It may be given as the .h5 OR as the run DIRECTORY; the resolver below
# picks the single .h5 inside a directory. ⚠ It does NOT fall back to a
# DIFFERENT run: a seed is a provenance statement, and quietly substituting
# another store would be the exact "warm start and noise pin from different
# folders" split the ruling forbids. A store that cannot be resolved means
# no pin, which means the noise stages run -- loudly, see below.
export GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_18mo_flagship/gf_prod_18mo_testing.h5}
if [ ! -f "${GF_SEED_STORE}" ]; then
  _seed_dir="${GF_SEED_STORE}"
  [ -f "${_seed_dir}" ] || [ -d "${_seed_dir}" ] || _seed_dir=$(dirname "${GF_SEED_STORE}")
  if [ -d "${_seed_dir}" ]; then
    _n_h5=$(find "${_seed_dir}" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${_n_h5}" = "1" ]; then
      GF_SEED_STORE=$(find "${_seed_dir}" -maxdepth 1 -name '*.h5')
      export GF_SEED_STORE
      echo "[V9-SEED] resolved the seed store inside ${_seed_dir}:"
      echo "[V9-SEED]   ${GF_SEED_STORE}"
    else
      echo "[V9-SEED] ${_seed_dir} holds ${_n_h5} .h5 files -- cannot pick one."
      find "${_seed_dir}" -maxdepth 1 -name '*.h5' 2>/dev/null | sed 's/^/[V9-SEED]   /'
    fi
  else
    echo "[V9-SEED] GF_SEED_STORE does not exist: ${GF_SEED_STORE}"
    echo "[V9-SEED] Candidate 18mo run directories on this filesystem:"
    find /shared/data/global_fit_output -maxdepth 1 -type d -name 'gf_prod_18mo*' \
      2>/dev/null | sort | sed 's/^/[V9-SEED]   /'
    echo "[V9-SEED] Re-launch with GF_SEED_STORE=<one of those>/<store>.h5 to"
    echo "[V9-SEED] pin the noise; otherwise the noise stages will fit it."
    # JOIN THE TWO GATES (2026-09-25, job-616 forensics). On its own a
    # missing seed is SOFT -- no pin, the noise stages run. But the warm
    # start is fitted from this same store, so when the refereed npz is
    # also absent the [WARMSTART] gate below hard-exits on the very same
    # path. Job 616 read the soft wording above, concluded the launch had
    # survived, and then died 10 lines later with no traceback and no
    # stated connection between the two. Say it here.
    if [ -n "${GB_WARM_START_COMPONENTS}" ] && [ ! -f "${GB_WARM_START_COMPONENTS}" ]; then
      echo "[V9-SEED] ⚠ AND THIS WILL BE FATAL BELOW. The warm-start mixture is"
      echo "[V9-SEED]   fitted from this same store, and the refereed npz"
      echo "[V9-SEED]     ${GB_WARM_START_COMPONENTS}"
      echo "[V9-SEED]   does not exist either, so the [WARMSTART] gate is about"
      echo "[V9-SEED]   to refuse the launch (exit 2). Fixing GF_SEED_STORE"
      echo "[V9-SEED]   fixes BOTH gates; there is one problem here, not two."
    fi
  fi
  unset _seed_dir _n_h5
fi
export GB_WARM_START_SOURCE_STORE=${GB_WARM_START_SOURCE_STORE-${GF_SEED_STORE}}
if [ "${GB_WARM_START_SOURCE_STORE}" != "${GF_SEED_STORE}" ]; then
  echo "[V9-SEED] WARNING: GB_WARM_START_SOURCE_STORE was overridden and no"
  echo "[V9-SEED] longer matches GF_SEED_STORE. The warm start and the noise"
  echo "[V9-SEED] pin would then come from DIFFERENT runs, which is exactly"
  echo "[V9-SEED] what the 2026-09-24 ruling forbids."
  echo "[V9-SEED]   warm start: ${GB_WARM_START_SOURCE_STORE}"
  echo "[V9-SEED]   noise pin : ${GF_SEED_STORE}"
fi
export GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-46656000}   # the 18mo parent (540 d)
export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-300}   # flagship (Mike 2026-10-08): the last 300 stored cold-chain rows x the parent's walkers (4 -> 1,200 samples) of the 18mo full_pe (~2.9M leaf rows into the fit at 6 mo; build the npz by hand on a login node if the in-job build is slow)
# GB warm-start SELF REFIT (user rulings 2026-10-13: "rebuild our warm start
# proposal with the last 50 samples (x4 walkers)" / "keep the original warm
# start in the fit"; 2026-10-14: refit every stored row in gb_search_1 and
# gb_search_2; 2026-10-15: "build everything on the head at entry time for
# the refit ... Make that the default", "split the refits three ways: old run
# refit that never changes, new refit with all sources, and new refit removal
# of sources that are already in all walkers ... each 1/3 of the time",
# search "100% refit on sources that do not exist in all four (with high
# confidence)", PE every 25 iterations; later on 2026-10-15, search "20 % old
# run refit warmstart / 80% self refit warmstart filtered" so no leaf of
# the previous run is left out; 2026-10-16 "Let's split them 0.5/0/0.5 in
# search": the 1yr tar 4 found the original still holding ~1,000 sources no
# walker had while the filtered yield was spent). The HEAD keeps this run's last
# GB_WARM_SELF_LAST_K stored cold rows x every walker in memory and, at a
# warm move's entry once GB_WARM_SELF_EVERY_SEARCH rows (gb_search_1/2: every
# cycle) or GB_WARM_SELF_EVERY_PE rows (full_pe) were stored since the last
# build, refits the mixture (fit -> referee -> apply, the parent build's
# pipeline, then the consensus filter) into
# ${STORE_DIR}/warmstart/self/epoch_NNNN/: ONE build gives the FULL refit and
# the FILTERED one (minus the clusters every walker holds in >=
# GB_WARM_SELF_CONSENSUS of its rows with the same leaf count); every rank
# waits for it (~75 s on the 9mo). The warm moves draw "original, full,
# filtered" by GB_WARM_SELF_SPLIT_SEARCH (rj_warm_search) and
# GB_WARM_SELF_SPLIT_PE (rj_warm_pe), all from the parent until the first
# epoch exists -- a cross-walker GB exchange: each walker is proposed what
# the others hold. GB_WARM_SELF_AT_ENTRY=0 returns to the saver-side build
# after each due save. GB_WARM_SELF_FRACTION / _FRACTION_SEARCH are no longer
# read (the recipe warns if they are set).
# Code default OFF (stock runs bit-identical); needs GB_WARM_START_COMPONENTS.
# Log: "[GB_WARM_SELF ...]". Rollback: GB_WARM_SELF=0 on the launch line.
export GB_WARM_SELF=${GB_WARM_SELF:-1}
export GB_WARM_SELF_AT_ENTRY=${GB_WARM_SELF_AT_ENTRY:-1}
export GB_WARM_SELF_EVERY_SEARCH=${GB_WARM_SELF_EVERY_SEARCH:-1}
export GB_WARM_SELF_EVERY_PE=${GB_WARM_SELF_EVERY_PE:-25}
export GB_WARM_SELF_LAST_K=${GB_WARM_SELF_LAST_K:-50}
export GB_WARM_SELF_SPLIT_SEARCH=${GB_WARM_SELF_SPLIT_SEARCH:-0.5,0,0.5}
export GB_WARM_SELF_SPLIT_PE=${GB_WARM_SELF_SPLIT_PE:-1,1,1}
export GB_WARM_SELF_CONSENSUS=${GB_WARM_SELF_CONSENSUS:-0.95}
if [ -n "${GB_WARM_START_COMPONENTS}" ] && [ ! -f "${GB_WARM_START_COMPONENTS}" ]; then
  if [ -f "${GB_WARM_START_SOURCE_STORE}" ]; then
    echo "[WARMSTART] ${GB_WARM_START_COMPONENTS} missing -- it will be BUILT"
    echo "[WARMSTART] automatically at recipe build (fit -> referee -> apply) from"
    echo "[WARMSTART]   ${GB_WARM_START_SOURCE_STORE}"
    echo "[WARMSTART] (last_k=${GB_WARM_START_LAST_K}, source tobs=${GB_WARM_START_SOURCE_TOBS})."
  else
    echo "[WARMSTART] FATAL: GB_WARM_START_COMPONENTS=${GB_WARM_START_COMPONENTS} does not exist"
    echo "[WARMSTART] and the auto-build source store is also missing:"
    echo "[WARMSTART]   GB_WARM_START_SOURCE_STORE=${GB_WARM_START_SOURCE_STORE}"
    echo "[WARMSTART] ^ that path came from GF_SEED_STORE -- read the [V9-SEED] lines"
    echo "[WARMSTART]   above FIRST: if the seed store did not resolve, this is the"
    echo "[WARMSTART]   same single failure and re-launching with a seed that exists"
    echo "[WARMSTART]   clears both gates at once."
    echo "[WARMSTART] Point GB_WARM_START_SOURCE_STORE at the previous run's FULL FINAL h5,"
    echo "[WARMSTART] or build the npz by hand:"
    echo "[WARMSTART]   python -m lisatools.globalfit.warmstart.fit_from_store --store <store.h5> \\"
    echo "[WARMSTART]       --last-k ${GB_WARM_START_LAST_K} --tobs ${GB_WARM_START_SOURCE_TOBS} --out <dir>/gf_prod_18mo_flagship_fit.npz"
    echo "[WARMSTART]   python -m lisatools.globalfit.warmstart.match_referee --npz <...fit.npz> --store <store.h5>"
    echo "[WARMSTART]   python -m lisatools.globalfit.warmstart.referee_apply --fit <...fit.npz> \\"
    echo "[WARMSTART]       --referee <...fit_referee.npz> --out ${GB_WARM_START_COMPONENTS}"
    echo "[WARMSTART] or launch with GB_WARM_START_COMPONENTS= (explicitly empty) to run without it."
    exit 2
  fi
fi
# Uniform-floor weight over the 9-col prior box (keeps death factors
# finite at every leaf) and the p floor: recipe defaults, pinned for the
# run record.
export GB_WARM_START_FLOOR_EPS=${GB_WARM_START_FLOOR_EPS:-0.05}
# The knob also arms the PE twin rj_warm_pe (IMMEDIATELY BEFORE
# rj_fstat_pe in full_pe; user ruling 2026-09-07). Its container charges
# the EXACT wrapped-normal density (detailed balance) via +-K period
# images per wide circular column; the search twin keeps the cheaper
# minimal-image charge (no DB requirement). 0 would restore minimal
# image on the PE side too -- never do that in a real PE run.
export GB_WARM_START_CIRC_IMAGES=${GB_WARM_START_CIRC_IMAGES:-3}

# ============================================================================
# V9-12 (cont.) -- THE NOISE START PIN, from GF_SEED_STORE's maxlogL point.
#
# WHY IT IS MANDATORY FOR v9 AND WAS NOT FOR v8. The v9 recipe's first two
# search stages DO NOT SAMPLE THE NOISE MODEL -- "fixed noise" is simply the
# absence of the psd/galfor moves from those stages. psd and galfor are the
# only sampled branches with no start-coordinate path in run.py, so without a
# pin their chains begin at a PRIOR DRAW and the whole of gb_search_1 and
# gb_search_2 would hunt GBs against a random noise level.
#
# The values are PHYSICAL/LINEAR. run.py converts them into whatever basis
# THIS run samples in (psd ln, galfor log10 on amp/fk/f_1/f_2, alpha linear),
# which is why the interchange format is physical: this run exports
# GALFOR_LOG_SAMPLING=1 and the source run's basis may differ. The extractor
# detects the SOURCE basis from the values themselves -- every log-sampled
# column is strictly positive under its physical prior, so a value <= 0 can
# only be a log -- and refuses rather than emitting an absurd pin.
#
# It ALSO decides the noise stages: with every sampled noise branch pinned,
# run_combined_staged.py drops noise_search / noise_vgb_search (there is
# nothing for a convergence-gated burn-in to find). STAGE_FORCE_NOISE_SEARCH=1
# runs them anyway; leaving the pin unset keeps them, and the driver says
# which it chose.
#
# ⚠ ON FAILURE THIS FALLS BACK, IT DOES NOT ABORT -- but LOUDLY. The
# fallback ("no pin, so run the noise stages") is not a degraded
# configuration, it is exactly what v8 ran for months: the noise stages fit
# psd+galfor to convergence and gb_search_1/2 then freeze them at THAT point
# instead of at the previous run's. Aborting a 2-node on-demand allocation
# over it would be the worse outcome.
#
# The realistic failure is a seed store from an OLDER galfor
# PARAMETERIZATION. Measured 2026-09-24 against a v7 store: its galfor
# columns came back f_1 ~ 2e5, f_2 ~ 4e3 against the current prior range
# 1e-5..1e-2 -- a different model, not a different basis. The extractor's
# physical-window check caught it and refused, which is the whole point: a
# pin from that store would have started the run at a foreground nothing in
# the current model can represent.
#
# Export V9_PIN_REQUIRED=1 to make a failed pin FATAL instead.
# ---- THE GALFOR START POINT: the OFFLINE 3-MONTH ESTIMATE ----------------
# USER RULING 2026-09-24, asked for explicitly after being shown the 6-month
# alternative: use the 3mo offline estimate. "For now" -- the durable
# mechanism is the maxlogL pin from GF_SEED_STORE, which is what the block
# below does for psd.
#
# PHYSICAL / LINEAR, order (amp, fk, alpha, f_1, f_2). run.py converts into
# the sampled basis (log10 on amp/fk/f_1/f_2; alpha stays LINEAR).
# amp is MODULATION-CORRECTED: the raw 90 d Welch fit is 1.160508e-44 and is
# divided by <M_XX> = 0.8080513 over the leading 90 d, because this run puts
# M(t) back on top (GALFOR_MODULATION_PATH + _T0=data, exported above).
# ⚠ The 90 d window sits on a DIM stretch of the annual sweep (<M_XX> < 1,
# unlike 180 d's 1.1031), so the correction RAISES the amplitude here. Using
# the raw number would understate the foreground by 19%.
#
# PROVENANCE: galfor_6mo_figs/addback_fixedpoint_90d.json,
# history[-1].prior_box_refit. Add-back fixed point -- the GALFOR brick has a
# TWO-YEAR resolvable set subtracted (subtract_resolvable_tdi.py output, 8420
# binaries at SNR>7 / Tobs=1.999yr), so the binaries that no longer clear SNR
# 7 at 90 d are regenerated and added back; doing that once is a lower bound
# because the added power demotes more, so it is iterated to a fixed point.
# Converged at iteration 6: n_drop 5227 -> 6792, delta +7, fit rms 0.0755,
# fmax 3e-3. ⚠ fit_galfor's DEFAULT fmax 8e-3 RAILS at every Tobs -- three
# sessions confirmed that independently; 3e-3 is the working value.
#
# ⚠ f_1 IS AT ITS CEILING AND IS NOT A MEASUREMENT. The unconstrained fit
# wants f_1 ~ 0.365 Hz, i.e. exp(-(f/f_1)^alpha) == 1 and the tanh doing all
# the cutting -- a real property of the 3mo spectrum, not a fit that failed
# to settle (f_1 sat at 0.14 .. 0.45 Hz across all six iterations at flat
# rms). Refitting inside the prior support changed the rms by nothing at all
# (0.07547 constrained vs 0.07547 free), which is exactly what an
# unidentified parameter looks like, and the curve moves < 2.24% over
# 0.8-5 mHz. Expect f_1 to sit against 1e-2 and the branch to wander along
# the exp-vs-tanh degeneracy; that is not a pathology. alpha 5.0 is INTERIOR
# here only because GALFOR_ALPHA_MAX=20.0 is exported above.
# Verified through the branch's own prior: total log_prior = -7.077, finite.
#
# ⚠ IT IS THE WRONG Tobs, IN THE UNSAFE DIRECTION. The 3mo foreground is
# louder than the 6mo one (amp 1.436e-44 vs the verified 6mo fixed point's
# 1.163e-44, x1.235), and a floor that is too HIGH hides sources -- the
# search's own SNR gate is computed against it. The 6mo fixed point is
# 1.162927183e-44,0.007332070412,3.371045451,0.002620587616,0.0005505988026
# (all five interior, no translation needed) if this is ever revisited.
export GALFOR_START_PARAMS=${GALFOR_START_PARAMS:-1.436180605904e-44,2.533915614978e-03,5.0,1.0e-02,1.405721657329e-03}

#
# ⚠ A HAND-SET VALUE ALWAYS WINS, PER VARIABLE. The guard used to test
# PSD_START_PARAMS only, so someone who set GALFOR_START_PARAMS by hand --
# which is exactly how an OFFLINE foreground estimate is supplied -- had it
# silently overwritten by the store's. Both are now saved across the eval and
# restored, so pinning one by hand still lets the other come from the store.
# ``${VAR+x}`` is set-ness, not emptiness, so an explicit empty value (the
# documented "no pin for this branch" escape) is honoured too.
_psd_was=${PSD_START_PARAMS+set}; _psd_save=${PSD_START_PARAMS-}
_gal_was=${GALFOR_START_PARAMS+set}; _gal_save=${GALFOR_START_PARAMS-}
if [ -f "${GF_SEED_STORE}" ] && { [ -z "${_psd_was}" ] || [ -z "${_gal_was}" ]; }; then
  # ⚠ NEVER `eval` this output. JOB 617 (2026-09-25) died right here with
  #     slurm_script: eval: line 3572: syntax error near unexpected token `('
  # The CLI itself is clean -- it writes its three diagnostics to STDERR and
  # only the two `export NAME=<digits>` lines to stdout, verified against a
  # real store both bare and under this script's full 191-export environment.
  # Something ELSE in the cluster process wrote to stdout: a banner from a
  # GPU/backend import is the obvious candidate, and it never appears on a
  # laptop with no CUDA, which is why local testing could not have caught it.
  # `eval` of another process's stdout is the defect regardless of who the
  # polluter turns out to be -- ANY stray line is executed as shell. So parse
  # the two values out with an anchored pattern that admits only number
  # characters, and drop everything else on the floor.
  _pin_rc=0
  _pin=$(python -m lisatools.globalfit.warmstart.noise_pin \
           --store "${GF_SEED_STORE}" --export) || _pin_rc=$?
  _psd_pin=$(printf '%s\n' "${_pin}" \
    | sed -n 's/^export PSD_START_PARAMS=\([-+0-9.eE,]*\)$/\1/p' | tail -1)
  _gal_pin=$(printf '%s\n' "${_pin}" \
    | sed -n 's/^export GALFOR_START_PARAMS=\([-+0-9.eE,]*\)$/\1/p' | tail -1)
  # Name the polluter in the log so the next run identifies it instead of
  # leaving this comment guessing.
  _pin_junk=$(printf '%s\n' "${_pin}" \
    | grep -vcE '^export (PSD|GALFOR)_START_PARAMS=|^[[:space:]]*$' || true)
  if [ "${_pin_junk:-0}" != "0" ]; then
    echo "[V9-SEED] note: the pin CLI wrote ${_pin_junk} unexpected stdout"
    echo "[V9-SEED] line(s). They are IGNORED (this is what broke job 617"
    echo "[V9-SEED] when the output was eval'd). First few:"
    printf '%s\n' "${_pin}" \
      | grep -vE '^export (PSD|GALFOR)_START_PARAMS=|^[[:space:]]*$' \
      | head -3 | sed 's/^/[V9-SEED]   | /'
  fi
  # A VARIABLE IS SATISFIED EITHER WAY, INDEPENDENTLY (user ruling 2026-09-25:
  # "the galfor pin should come from our estimate. The PSD estimate can come
  # from the 3mo run"). In the shipped configuration GALFOR_START_PARAMS is
  # already hand-set to the offline foreground estimate, so the store only has
  # to deliver the PSD line -- demanding BOTH would throw away a perfectly good
  # instrument-noise pin over a galfor value this run was never going to use.
  if [ "${_pin_rc}" = "0" ] \
     && { [ -n "${_psd_was}" ] || [ -n "${_psd_pin}" ]; } \
     && { [ -n "${_gal_was}" ] || [ -n "${_gal_pin}" ]; }; then
    if [ -n "${_psd_was}" ]; then
      echo "[V9-SEED] PSD_START_PARAMS was set by hand -- keeping it, not the store's."
    else
      export PSD_START_PARAMS="${_psd_pin}"
      echo "[V9-SEED] PSD_START_PARAMS <- the seed store's maxlogL cold walker."
    fi
    if [ -n "${_gal_was}" ]; then
      echo "[V9-SEED] GALFOR_START_PARAMS was set by hand (the offline foreground"
      echo "[V9-SEED] estimate) -- keeping it, not the store's."
    else
      export GALFOR_START_PARAMS="${_gal_pin}"
      echo "[V9-SEED] GALFOR_START_PARAMS <- the seed store's maxlogL cold walker."
    fi
  else
    echo "[V9-SEED] ############################################################"
    echo "[V9-SEED] # NO NOISE PIN: could not extract one from"
    echo "[V9-SEED] #   ${GF_SEED_STORE}"
    echo "[V9-SEED] # (see the [NOISE-PIN] line above for the exact reason --"
    echo "[V9-SEED] #  an out-of-window value means that store predates the"
    echo "[V9-SEED] #  current galfor parameterization.)"
    echo "[V9-SEED] #"
    echo "[V9-SEED] # FALLING BACK: noise_search / noise_vgb_search WILL RUN"
    echo "[V9-SEED] # and fit psd+galfor from scratch; gb_search_1/2 then"
    echo "[V9-SEED] # freeze them at THAT converged point. This is correct and"
    echo "[V9-SEED] # is what v8 did -- it just costs the noise burn-in."
    echo "[V9-SEED] #"
    echo "[V9-SEED] # To pin instead: point GF_SEED_STORE at a store written"
    echo "[V9-SEED] # with the current galfor model, or set PSD_START_PARAMS /"
    echo "[V9-SEED] # GALFOR_START_PARAMS by hand (PHYSICAL/linear units)."
    echo "[V9-SEED] # V9_PIN_REQUIRED=1 makes this fatal instead."
    echo "[V9-SEED] ############################################################"
    if [ "${V9_PIN_REQUIRED:-0}" = "1" ]; then
      echo "[V9-SEED] V9_PIN_REQUIRED=1 -- refusing to launch without a pin."
      exit 2
    fi
  fi
  unset _pin _pin_rc _psd_pin _gal_pin _pin_junk
elif [ -n "${_psd_was}" ] && [ -n "${_gal_was}" ]; then
  echo "[V9-SEED] both start pins were supplied explicitly -- the seed store is not read."
elif [ ! -f "${GF_SEED_STORE}" ]; then
  echo "[V9-SEED] GF_SEED_STORE=${GF_SEED_STORE} not found -- no noise pin."
  echo "[V9-SEED] noise_search / noise_vgb_search will run and fit it."
fi
unset _psd_was _psd_save _gal_was _gal_save
echo "[V9-SEED] store=${GF_SEED_STORE}"
echo "[V9-SEED] PSD_START_PARAMS=${PSD_START_PARAMS:-<unset: noise stages will run>}"
echo "[V9-SEED] GALFOR_START_PARAMS=${GALFOR_START_PARAMS:-<unset>}"

# ============================================================================
# V9-11 -- THE THREE-STAGE GB SEARCH (user spec 2026-09-24).
#
# One gb_search stage becomes gb_search_1 / _2 / _3, each running the cycle
#
#   rj_warm_search -> in_model -> rj_fstat_search -> in_model_fstat
#     -> rj_replace -> in_model_replace -> rj_prior_removal
#
# The cycle ENDS on the removal judge (user amendment 2026-09-24): every
# birth and every swap gets a full in-model refinement pass before it is
# judged for death, so nothing is ever deleted at coordinates it has not
# been given a chance to walk away from. Each in-model slot is named for
# the RJ move it polishes; watch [GF_TIMING] move=in_model* for their cost.
#
# and differing only in noise (fixed / fixed / sampled), phase max (on / off /
# off), opt SNR (8 / 5 / 5) and F-stat peak floor (8 / 6.25 / 6.25). The three
# knob values live in V9_SEARCH_STAGE_PROFILES in run_combined_staged.py, are
# applied to the shared move objects at stage entry, and every mutation
# prints a [V9-STAGE] line. =0 restores the single legacy stage.
export STAGE_V9_SEARCH=1
# A stage ENDS when the cold-chain nleaves plateau AND every OCCUPIED
# (walker, band) pair has shut off (GB_SEARCH_BAND_SHUTOFF_PER_WALKER, armed
# below). Empty pairs can never shut off -- "all pairs" is unreachable -- so
# the plateau gate is composed IN rather than replaced, and covers them.
# Watch: "[V9-STAGE gb_search_1] nleaves plateau reached but N occupied
# (walker, band) pair(s) have NOT shut off -- holding the stage open."
#
# Warm start every 5th iteration in STAGE 3 only (user ruling 2026-09-24): by
# then the previous run's posterior has been mined, and each hit is expensive.
# Stages 1-2 run it every iteration.
# ---- gb_search_seed (user ruling 2026-09-26) --------------------------
# A new FIRST search stage: warm-start RJ + in-model to convergence ONLY,
# run for a FIXED 5 full sampler iterations, then straight into
# gb_search_1 unchanged. The point is to seed HARD off the warm-start
# posterior before any F-stat grid is fitted -- the stage carries no
# rj_fstat_search, so GB_FSTAT_REFIT_EVERY=1 cannot trigger a fit in it.
#
# NAMED, NOT RENUMBERED (user's choice): gb_search_1/2/3 keep their names,
# their knobs (GB_SEARCH_3_WARM_EVERY above) and their meaning in every
# existing log and snapshot, so old runs stay directly comparable.
#
# Its profile is gb_search_1's verbatim -- opt_snr 8, phase maximization
# on, F-stat peak floor 8 -- per "identical to what is now gb search 2".
# It carries vgb_pe (~8 s) but NOT sobbh/mbh/emri: those were 2694 s of
# job 628's first iteration and are exactly the wait this stage avoids.
#
# ⚠ The per-(walker, band) RJ shutoff valve is DISARMED for this stage
# only. It freezes a band's RJ after GB_SEARCH_BAND_SHUTOFF_CONV_ITER (=3)
# flat iterations within a step; on a 5-iteration stage whose whole job is
# to birth hard, that would cut the seeding short in the bands slowest to
# pay off. Stages 1-3 keep it.
#
# 0 disables the stage entirely, restoring the pre-2026-09-26 recipe.
# ---- THE FLAGSHIP RECIPE (the 9mo pattern, user ruling 2026-10-07) ---------
# gb_search_seed (3 iterations) -> gb_search_1 -> gb_search_2 -> full_pe. No
# gb_search_3, no replica_pe (STAGE_REPLICA_PE above), no galfor ratchet
# (GALFOR_RATCHET stays 0). The seed: the 6mo's, 3 iterations not 5, no
# source moves. gb_search_1: its table row (opt SNR 8, phase max, F-stat peak
# 8, prior removal only), noise FIXED, NO sobbh/mbh/emri moves. gb_search_2:
# its table row (opt SNR 5, no phase max, peak 6.25, prior births AND
# deaths), the valves reset at its fresh entry, the sources EVERY iteration,
# and ONE psd+galfor search to CONVERGENCE at the END of every cycle ("only
# one noise/galfor proposal ... at the end of the cycle"): the standalone
# noise stage's plateau rule (NOISE_SEARCH_CHECKS flat rounds per walker
# within MAXLOGL_TOL), afresh each cycle. Then full_pe, declared as today.
# The resolved recipe, no data / no MPI, under this script's exports:
#   python scripts/fstat_proposal/run_combined_staged.py --print-recipe
export GB_SEARCH_SEED_ITERS=3
export GB_SEARCH_STAGES=${GB_SEARCH_STAGES:-1,2}
export GB_SEARCH_1_SOURCE_EVERY=${GB_SEARCH_1_SOURCE_EVERY:-1}   # 2026-10-10 ruling: sobbh/mbh/emri every cycle in gb_search_1 too (was 0)
export GB_SEARCH_2_SOURCE_EVERY=${GB_SEARCH_2_SOURCE_EVERY:-1}
export GB_SEARCH_2_NOISE_MODE=${GB_SEARCH_2_NOISE_MODE:-cycle_end}
export GB_SEARCH_2_RESET_VALVES=${GB_SEARCH_2_RESET_VALVES:-1}
# ---- gb_search_2 IN-MODEL BUDGET (Mike 2026-10-16: "Gb search 2 should spend
# less time in the in-model mixing but maybe with some quick checker to make
# sure the logL is not seriously improving during those steps anymore";
# gb_search_1 stays as is). The checkers already exist and stay: the pure
# in-model moves' per-sub-band group rule ([GB_IMGROUP]: passes until no open
# sub-band gained D/2 over the window) and the RJ moves' per-row polish rule
# ([GB_IMCONV]: a row retires once its best gain stalls over the window). Both
# windows are FLOORS, so these shorten the floors in gb_search_2 only (applied
# at the stage's entry, a resume included; the other stages run the moves as
# built: 25 repeats/pass, 3 passes, 250). Measured on the 1yr flagship job 774
# (gb_search_2, 146-min cycle, 2026-10-16): the three dedicated in-model legs
# ran 10-12 passes x 25 repeats each, of which the 3 warm-up passes (129-147 s
# each, half the leg) could shut nothing, for a net cold lnL change per
# propose of about zero (the legs are mixing, the gains come from the RJ
# moves); the RJ moves' newborn polish ran 255 repeats per row against a
# 250-repeat floor for a median gain of 0.2 nats (top tenth +22 nats, which
# the rule keeps running). Expected: ~20 min off the in-model legs and ~18
# min off the polish per cycle (~26 %). Rate bars at the same D/2 = 4.0:
# group 4.0/(2 x 15) = 0.133 lnL/repeat (was 0.053), newborn 4.0/100 = 0.04
# (was 0.016) -- looser, which is the direction asked for. Unset any of the
# three on the line to keep that knob as built.
# FEWER, FATTER PASSES (Mike 2026-10-19, "faster ... I really do not want to
# sacrifice accuracy"): 15 x 2 -> 30 x 1. The group test is unchanged -- the
# best cold gain over the last 30 repeats <= D/2 -- and is judged at 60 / 90 /
# 120 repeats instead of 45 / 60 / 75 / 90, a subset, so no band runs fewer
# repeats than before (a flat band 60 instead of 45, a band shut by the pass-1
# static test 30 instead of 15); the vertical swaps (opening sweep + one per
# ~25-repeat refit tick) go from 1 to 2 per pass. MEASURED on the 1yr job 804
# against job 780 (2026-10-21): 4.8 passes per rank instead of 7.8, but a pass
# at 30 repeats costs ~1.6x one at 15 at equal open bands, so the leg's wall is
# UNCHANGED (~12 min). It buys mixing per band at equal cost, not speed. (The
# 10-19 prediction of ~6 min per leg came from a pass-cost fit across jobs 774
# and 780 that read the pass wall as repeat-independent; job 804 shows it is
# not.)
export GB_SEARCH_2_NUM_REPEAT_PROPOSALS=${GB_SEARCH_2_NUM_REPEAT_PROPOSALS:-30}
export GB_SEARCH_2_INMODEL_GROUP_ITERS=${GB_SEARCH_2_INMODEL_GROUP_ITERS:-1}
export GB_SEARCH_2_INMODEL_CONVERGE_ITERS=${GB_SEARCH_2_INMODEL_CONVERGE_ITERS:-100}
# F-STAT REFIT EVERY SECOND CYCLE IN gb_search_2 (Mike 2026-10-19: "Make the
# fstat refit every 2 cycles during gb search 2"); gb_search_1 keeps
# GB_FSTAT_REFIT_EVERY below (every cycle -- its population moves fast, the
# 2026-09-25 ruling). Same units as that knob: refit-clock ticks, one per
# stored row, and under GB_SEARCH_LEGS=1 a cycle is three rows, so the ages
# the one rj_fstat_search propose per cycle sees are 3, 6, 9: 4 refits every
# second cycle and stays there if a relaunch drops a tick or two. Why: on the
# 1yr job 780 the fit was 11 min of rj_fstat_search's 30 (epoch 27, falling
# as bands shut on every walker) while each walker added ~35 of ~1,275 peaks'
# sources per cycle, so a grid one cycle old has lost ~3 % of its peaks; the
# rotation keeps each walker's own grid in use a quarter of the time (two
# cycles in eight). Applied at every entry of the stage; the entry line reads
# "F-stat refit cadence set on entry ... rj_fstat_search.fstat_refit_every 2
# -> 4", and the epoch "starting" lines come every second cycle.
export GB_SEARCH_2_FSTAT_REFIT_EVERY=${GB_SEARCH_2_FSTAT_REFIT_EVERY:-4}
export GB_SEARCH_3_WARM_EVERY=5

# ---- MID-ITERATION CHECKPOINTS (user ask 2026-09-24: "make sure there are
# checkpoints between all the GB moves for resume") ------------------------
# The HOOKS already sit at EVERY sub-move boundary: GFCombineMove writes one
# after each move it proposes, on both its sequential and weighted-cycle
# paths, and the v9 search stages take the sequential one. So across a
# search iteration there is a coherent resume point after each of
#   rj_warm_search / in_model / rj_fstat_search / in_model_fstat /
#   rj_replace / in_model_replace / rj_prior_removal
# plus one BETWEEN the two internal passes of rj_replace (that move is a
# single sub-move running two complete sweeps, so the combine's own boundary
# would have left the second pass unprotected).
#
# ⚠ WHICH hooks actually WRITE is the throttle's decision, not the hook's.
# MIDIT_CHECKPOINT_MIN_INTERVAL is the minimum seconds between writes, so at
# the 600 s default a ~2500-4000 s v9 iteration checkpoints at roughly every
# other boundary, and the exposure on a preemption is bounded by the LONGEST
# gap between two writes -- which for v9 is a whole in-model slot, since one
# of those can run up to GB_INMODEL_GROUP_MAX_PASSES x
# GB_NUM_REPEAT_PROPOSALS per source. Lower this if the [MIDIT_CKPT] write
# cost (it logs MB and seconds) turns out to be small against that exposure;
# 0 writes at every boundary. Both pinned explicitly here rather than left
# to the code defaults, because on a SPOT partition this is the knob that
# decides how much a preemption costs.
# OVERRIDABLE from the command line since 2026-10-01 (it was a hard
# `export MIDIT_CHECKPOINT=1`, so `MIDIT_CHECKPOINT=0 ./submit...` reached
# nothing -- the documented knob-reaches-nothing failure). Mike's ruling for
# the legged + ratcheted search: "MIDIT_CHECKPOINT=0 in the launch command
# is good". Under GB_SEARCH_LEGS=1 a row lands after every leg, so the
# checkpoint protects at most one leg of work, while its adoption rule
# (checkpoint stored_iteration >= store iteration) cannot tell a harmless
# startup copy from a half-nudged state: jobs 673/674 each ran the galfor
# NUDGE and were stopped; ten more minutes and either would have written a
# post-nudge checkpoint at stored iteration 47 that job 675 would have
# adopted and nudged AGAIN (the job-672 double nudge by another door).
# DEFAULT 0 since 2026-10-07 (V9-30: every production relaunch since job 715
# carried MIDIT_CHECKPOINT=0 on the line); MIDIT_CHECKPOINT=1 on the line
# re-arms it for an unlegged run.
export MIDIT_CHECKPOINT=${MIDIT_CHECKPOINT:-0}
export MIDIT_CHECKPOINT_MIN_INTERVAL=${MIDIT_CHECKPOINT_MIN_INTERVAL:-600}
if [ "${MIDIT_CHECKPOINT}" = "1" ]; then
  echo "[V9-CKPT] mid-iteration checkpoints ON, min interval ${MIDIT_CHECKPOINT_MIN_INTERVAL}s"
  echo "[V9-CKPT] watch: [MIDIT_CKPT] wrote ... at stored iteration N (boundary '...')"
else
  echo "[V9-CKPT] mid-iteration checkpoints OFF (MIDIT_CHECKPOINT=${MIDIT_CHECKPOINT}); under GB_SEARCH_LEGS a row lands after every leg instead"
fi

# ============================================================================
# CHANGE 3 OF 3 vs 3mo_v8 -- MBHB + EMRI + SOBHB (campaign S6). Non-empty
# id lists arm the branches in run_combined_staged.py: a source_search
# stage (joint max-lnL over the source PE moves) runs FIRST so the loud
# sources converge + subtract before the noise stages fit the PSD; the
# PE moves then ride gb_search + full_pe (sobbh -> mbh -> emri banking
# order). Ids = the 2026-08-24 census (user "yes in general",
# 2026-09-02): MBHB only the 4 systems with t_merge <= 6 mo; EMRI/SOBHB
# full census pending the S4 readout.
export MBHB_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19   # t_c 300.4/611.7/173.3/336.8/286.4/104.7/507.4/263.8/606.6/285.9/369.5/654.6/243.8/481.3/712.7/318.0/111.4/585.5/92.0/405.9 d from the data start (mojito-lite catalogue anchored on the MBH session table 2026-10-03): every catalogue MBHB merging inside 720 d + the 14-d MBH_MERGER_TIME_BUFFER below; all merge BEFORE the end; none lands in (720, 734] d; the whole catalogue
# User rule 2026-10-07: keep any MBHB merging within 2 weeks of the end of the
# observation (its in-window inspiral is modelled, cut at the data end by the
# batched window's active-box clamp and the stock generator's data lattice).
# 14 d replaces the code default of 7 d; the t_plunge prior's upper edge
# follows it (obs_end + buffer + t_plunge_pad).
export MBH_MERGER_TIME_BUFFER=1209600   # 14 d
export EMRI_IDS=0,1,2,3,4,5,6,7    # all 8 at every Tobs (plunging INSIDE this window, on the direct template's plunge path: 3 (256 d), 0 (347 d), 6 (427 d), 1 (518 d), 7 (666 d), 2 (712 d))
export SOBHB_IDS=0,1,2,3,4,5       # all 6 -- expected mostly sub-threshold
# ---- THE COMBINED DATA SET (user ruling 2026-09-14: the 6mo testing
# campaign runs on the "combined" data) --------------------------------
# COMBINED = mojito's PRE-SUMMED L1 stream (${MOJITO_DATA_PATH}/data/
# COMBINED/L1/): it establishes the data AND the orbits, and it already
# contains EVERYTHING -- instrument noise, the full GB galaxy (resolved +
# confusion), VGBs, and ALL sources of every class. The per-class entries
# after it contribute CATALOGUES ONLY (VGB seeding, nleaves sizing,
# F-stat overlays, source-branch priors/starts); their L1 bricks are NOT
# read, so per-id brick presence stops mattering. The loader REFUSES
# COMBINED+NOISE and COMBINED+GALFOR (double-count guards) -- do not add
# them. The NOISE brick must still EXIST on disk: UNEQUAL_ARM=1 reads its
# /ltts delay table and the psd start estimates read its noise_estimates,
# both resolved by PATH (NOISE_FILE / data/INSTRUMENT/L1/NOISE_*),
# independent of this list.
# ⚠ MODELING CONSEQUENCE vs the per-brick default (NOISE,GB,VGB,+armed):
# the data now also contains the NON-ARMED sources (the other MBHBs
# beyond ids 2,5,16,18, any catalogue rows outside the id lists). Nothing
# samples them, so they sit in the residual as unmodeled content -- the
# realistic full-data-challenge configuration, accepted by the 09-14
# ruling. Explicitly-set SOURCE_TYPES always wins over this line.
export SOURCE_TYPES=${SOURCE_TYPES-COMBINED,GB,VGB,MBHB,EMRI,SOBHB}
echo "[DATA] SOURCE_TYPES=${SOURCE_TYPES} (COMBINED = pre-summed stream; classes after it are catalogue-only)"
# Ladders / repeats / swap cadence / start scatter: the probe's latest
# rulings (submit_gf_6mo_sources_probe.sh, 2026-08-26..28 -- see its
# comment blocks for the measured cost arithmetic).
# Yielding since 2026-10-02: the stored ladder wins on a resume anyway
# (recipe.resume_ladder_wins, with a warning naming both counts), and a store
# re-rung with scripts/fstat_proposal/rerung_store_branches.py is relaunched
# with the matching count on the line (MBH_NTEMPS=12 EMRI_NTEMPS=12) so the
# configured and stored ladders agree and the warning stays quiet. A hard
# export here swallowed that line value.
# DEFAULT 8 since 2026-10-07 (V9-30): the 6mo store was re-rung to 8 on 10-02
# and every relaunch line since carried MBH_NTEMPS=8 EMRI_NTEMPS=8; a FRESH
# store (1yr, 9mo) builds its MBH / EMRI ladders from these.
# WHERE THE LADDER COMES FROM (2026-10-13): {MBH,EMRI,SOBBH}_NTEMPS size the
# DEFAULT per-leaf ladder each branch builds when no explicit betas is set:
# eryn make_ladder(ndim, ntemps) (EMRI ndim 12, MBH / SOBBH 11; the ~25 %
# swap-acceptance geometric spacing), in stock/erebor/{mbh,emri,sobbh}.py via
# common.default_source_ladder (user ruling 2026-10-13; was 1/1.2^k). A
# RESUMED store keeps its stored per-leaf ladders (betas_all);
# {MBH,EMRI,SOBBH}_LADDER_RESET_ON_RESUME=1 on the LAUNCH LINE (one-off, not
# exported here) replaces them with this ladder, logged leaf by leaf
# (recipe.resolve_resumed_ladders). The add/remove move's lost-rung guard
# re-anchors a hot rung more than {BRANCH}_LADDER_REANCHOR_NATS (code default
# 200; 0 = off) below its leaf's cold rung, beyond the rung's expected
# tempering drop, to the cold state ("[EMRI_LADDER] leaf N: re-anchored ...").
export MBH_NTEMPS=${MBH_NTEMPS:-8}
# EMRI 8 -> 12 rungs with T_max 6 -> 625 below (user ruling 2026-10-18, the
# EMRI sampling audit): the loud EMRIs' posteriors are combs of discrete
# islands and a T=6 top rung cannot find islands no walker holds; the top rung
# should see the LOUDEST leaf (EMRI 3, SNR 50) at an effective SNR ~2 ("go
# until it is at SNR ~2"), i.e. T_max = (50/2)^2 = 625; 12 rungs (ratio 1.80,
# one more than Eryn's make_ladder count for a 25 % swap rate at ndim 12 --
# the count the production store was migrated to), at
# ~1.5x the EMRI move (~9 % of the iteration wall). A RESUMED store keeps
# its 8 stored rungs: migrate it first (scripts/fstat_proposal/
# rerung_store_branches.py --set emri=12) and put EMRI_LADDER_RESET_ON_RESUME=1
# on that one relaunch line.
export EMRI_NTEMPS=${EMRI_NTEMPS:-12}
# Ladder caps (user ruling 2026-10-13, second ruling of the day: "I want to
# focus on sampling the main modes (+ the sky move for MBHs) in terms of
# tempering"). The first ruling capped the EMRI ladder at T_max = 100 (an
# SNR-100 EMRI at SNR 10 on the hottest rung) and left MBH on make_ladder's
# own spacing (T_max ~ 150). Measured on the 9mo store (job 759 rows 200-349
# vs rows 110-197 on the old 1/1.2^k ladder) that put 5 of the 8 EMRI rungs
# below beta 0.07, where every EMRI but the two loudest has lost the source
# (hot-rung lnL deficit = the whole SNR^2/2, in-model acceptance 0.9-0.98 =
# diffusing in the prior), cut the cold-pair swap acceptance 0.5-0.74 ->
# 0.13-0.30 and raised the cold-chain autocorrelation time 1.3-2.7 -> 5-16
# iterations for BOTH branches. Main-mode ladders instead: the hottest rung
# keeps an effective SNR (SNR / sqrt(T_max)) that still sees the source --
#   EMRI_LADDER_TMAX=6:  an SNR-15 EMRI keeps SNR 6 on the top rung
#                        ((15 / 6)^2 ~ 6; SNR 12 -> 4.9, SNR 48 -> 20);
#                        make_ladder spaces the 8 rungs 1..6 (ratio 1.29,
#                        swaps ~0.5-0.6 as on the old ladder).
#   MBH_LADDER_TMAX=10:  SNR 180-1000 MBHBs keep SNR 56-330 on the top rung;
#                        the discrete sky / inclination modes are the sky
#                        hops' job (MBH_SKY_MOVE_*), not the ladder's.
#   MBH_LADDER_REANCHOR_NATS=0: the lost-rung guard OFF for MBHBs. An MBHB
#                        rung is never "lost" (that would be 15k-500k nats
#                        down); what the guard caught (61 times in 152
#                        iterations of job 759, deficits 620-1350 nats) were
#                        hot rungs sitting in a secondary sky mode a few
#                        hundred RAW nats down -- 4-8 tempered nats, i.e. it
#                        was undoing the tempering.
#   EMRI_LADDER_REANCHOR_NATS=50 (code default 200): on the T_max 6 ladder a
#                        legitimate rung sits (ndim/2)(T-1) <= 30 nats below
#                        the cold rung with a spread of T sqrt(ndim/2) ~ 15,
#                        while a rung that LOST its EMRI is the whole SNR^2/2
#                        down (72 at SNR 12, 240 at SNR 22, 1150 at SNR 48).
#                        200 only caught the two loudest; 50 (3.4 sigma above
#                        a legitimate rung) catches every EMRI but the
#                        faintest's top rung. Matters at the reset below: the
#                        old T_max 100 rungs 5-7 hold prior states for every
#                        EMRI but 3 and 6, and on the new ladder nothing else
#                        brings them back (a swap into a state 100 nats worse
#                        never happens).
#   EMRI_LADDER_REANCHOR_NATS 50 -> 100 (2026-10-18, the EMRI sampling audit,
#                        docs/emri-sampling-audit.md item 6): on the 9mo (job 773,
#                        rows 840-1007) every leaf's hot-rung deficit sits on the
#                        tempered-Gaussian expectation (ndim/2)(T-1) (30 nats at the
#                        top rung; medians 24-30 on 7 of the 8 leaves), and the 26
#                        firings in 137 iterations were rungs 6-7 at 72-100 nats:
#                        the posterior's own top-rung tail (q99 70-85), on leaves
#                        whose LOST deficit SNR^2/2 is 118-1262 nats (EMRI 3, lost
#                        at 1262, fired twice). 100 (130 at the top rung) still
#                        catches a lost rung of every EMRI above SNR 16 and stops
#                        undoing the top rungs' exploration of the islands; the
#                        two faintest (SNR 12.8 / 15.4, lost at 82 / 118 nats) are
#                        prior-dominated at T = 6 anyway.
#   SOBBH_LADDER_TMAX=6 / SOBBH_LADDER_REANCHOR_NATS=20 (user ruling
#                        2026-10-15, "update the SOBHBs"): until then SOBBH
#                        stayed on the uncapped make_ladder spacing (beta down
#                        to 0.007, T_max ~150), measured on the 1yr tar 2 at
#                        row 28: hot-pair swaps 0.80-0.95 (prior sampling),
#                        cold pairs 0.26-0.56. The EMRI ladder (ratio 1.29)
#                        instead; an SNR-10 SOBBH keeps SNR 4 on the top rung.
#                        Guard 20: a lost SNR-10 SOBBH is 50 nats down against
#                        a 27.5-nat expected drop at the top rung (ndim 11);
#                        the SNR 3-5 leaves (12-27 nats whole) can never trip it.
# A RESUMED store keeps its stored ladders: put EMRI_LADDER_RESET_ON_RESUME=1
# MBH_LADDER_RESET_ON_RESUME=1 SOBBH_LADDER_RESET_ON_RESUME=1 on the relaunch
# line ONCE to replace them (logged leaf by leaf; the old rungs' states stay
# where they are and the guard re-anchors the lost ones at the first block).
# EMRI T_max 6 -> 625 with 12 rungs (user ruling 2026-10-18; see EMRI_NTEMPS).
export EMRI_LADDER_TMAX=${EMRI_LADDER_TMAX:-625}
export EMRI_LADDER_REANCHOR_NATS=${EMRI_LADDER_REANCHOR_NATS:-100}
export MBH_LADDER_TMAX=${MBH_LADDER_TMAX:-10}
export MBH_LADDER_REANCHOR_NATS=${MBH_LADDER_REANCHOR_NATS:-0}
export SOBBH_LADDER_TMAX=${SOBBH_LADDER_TMAX:-6}
# SOBBH guard 20 -> 60 (user ruling 2026-10-18, with the EMRI audit): on the
# 9mo job 773 it fired 60 times on the two bimodal SNR 9-10 leaves (0 and 1),
# re-anchoring hot rungs sitting 45-55 nats below cold = the OTHER mass mode,
# not a lost source; for a source this weak a T=6 rung's tail and a lost
# source are the same number, so the guard yields to the exploration.
export SOBBH_LADDER_REANCHOR_NATS=${SOBBH_LADDER_REANCHOR_NATS:-60}
# 12 -> 8 (user ruling 2026-09-16, with repeats 25 -> 20: sobbh cost
# trim now that the single-call fix landed). NOTE: SOBHB's ladder lives
# in the stored PerLeafLadderState (betas_all per leaf), so on a RESUME
# the stored 12-rung ladder wins and this value is inert -- 8 rungs
# take effect at the next FRESH branch init (the deliberate VGB-chirp
# restart). The persisted eigen sidecar's sobbh tables are
# per-(temp,walker) shaped and will rebuild once on the shape change.
# REVERTED 8 -> 12 (2026-09-16 snapshot 7 decode): the resume-safety
# claim was WRONG -- on resume the LIVE ladder followed the config (8
# betas) while the store/sidecar stayed 12-shaped, triggering mass
# eigen-table rebuilds every launch (231 s leaf windows, 772 rows/call,
# 80/80 non-positive infomats, mbh/emri walls 3-5x, iteration ~80 min).
# Set 8 ONLY at the deliberate fresh restart (with the VGB chirp
# migration), never on a resumed 12-rung store.
# (2026-09-17) Overridable, and the STORED rung count now WINS on resume
# for mbh/emri/sobbh (recipe.resume_ladder_wins, same rule as gb/vgb): a
# store born at 8 rungs (d3c0d6ee era, 09-16 12:52 -> 21:55) resumed under
# 12 built a 12-rung move against an 8-rung state and died in the SOBBH
# per-walker eigen sweep ("cannot reshape array of size 88 into shape
# (12,newaxis)"). The knob is reported-and-ignored on such a resume; a
# fresh store honours it. BACK TO 8 (user ruling 2026-09-17 evening): the
# gf_prod_6mo_v8_4gpu store IS an 8-rung store, and a resumed 12-rung
# store (the _mr copy, the original run) keeps its 12 via the store-wins
# rule with a WARNING, so 8 here is safe for every store.
export SOBBH_NTEMPS=${SOBBH_NTEMPS:-8}
# ONE information matrix per leaf at the max-lnL cold walker (like MBH/EMRI)
# instead of one per (temperature, walker) (user ruling 2026-09-16): the
# per-walker stash is keyed by the walker axis, so every resume under a
# different walker-block width (2 -> 4 GPUs) discarded it and rebuilt 60-120
# matrices per leaf (~2.6 min/leaf/rank); walker_max tables are
# layout-independent and persist across resumes. Watch the per-rung SOBBH
# acceptance -- hot rungs now propose with the best walker's curvature.
# THE ONLY SOBBH_EIGEN_SCOPE EXPORT IN THIS FILE: a second
# `export SOBBH_EIGEN_SCOPE=per_walker` further down (the 2026-09-08
# TABLE SCOPE block) silently overrode this line until 2026-09-17, so the
# ruling never took effect. Escape: SOBBH_EIGEN_SCOPE=per_walker in the env.
export SOBBH_EIGEN_SCOPE=${SOBBH_EIGEN_SCOPE:-walker_max}
# IN-MODEL REPEATS OF THE SOURCE MOVES = 25 IN EVERY STAGE (user ruling
# 2026-10-03: "adjust the in-model repeats during search for mbhs emris and
# sobhbs to 25. same as pe"). These are the BUILT values, so gb_search_3 and
# replica_pe (sources every GB_SEARCH_SOURCE_EVERY=5 iterations) run them at
# 25 too; the PE declaration below then finds them already at 25. Were 2 / 2
# (hard exports: a value on the launch line could not reach them) / 10.
# ⚠ COST under EMRI_LIKELIHOOD=full (the per-row production wrap): rows per
# visit scale as repeats x rungs, so 25 repeats x 8 rungs is 50x the 2 x 2
# rows job 706 ran. EMRI_NUM_PROP_REPEATS=2 on the line keeps the old count.
export MBH_NUM_PROP_REPEATS=${MBH_NUM_PROP_REPEATS:-25}
export EMRI_NUM_PROP_REPEATS=${EMRI_NUM_PROP_REPEATS:-25}
# 25 -> 20 (user ruling 2026-09-16) -> 10 (user ruling 2026-09-18) -> 25
# (user ruling 2026-10-03, above).
# THE lever on the dominant per-iteration cost. [SOBBH_LL_TIMING] on the
# 4-GPU run measured the chunked-het scorer at a FLAT 1.73 s per CALL,
# independent of how many rows the call carries (windows of 138 and 404
# rows both cost 1.72 s/call; 100% of it in the kernel-launch span at
# ~605 ms per WDM layer group, ~2.8 groups/call). Calls come from repeats,
# NOT from walkers or rungs, so the walker-block width does not touch this
# cost and repeats are the only knob that moves it: 22 calls/leaf (20
# repeats + 2) x 6 leaves x 1.73 s = 228 s, plus 12 residual fills at
# 1.70 s = 248 s, i.e. 43% of a 9.2-min iteration. At 10 repeats that is
# ~130 s and the iteration drops to ~7.2 min. The underlying defect is the
# 605 ms group launch: the in-code reference for this configuration is
# 2.78 ms/row (sobbhspecialmove.py, job-373 note), ~58x away.
export SOBBH_NUM_PROP_REPEATS=${SOBBH_NUM_PROP_REPEATS:-25}
# PE stages (replica_pe + full_pe; user ruling 2026-10-02: "in-model repeats
# of 25 for all sources ... This includes VGBs and GBs ... that way we have a
# general 'thinning' factor of 25"): the PE stage declares the in-model
# repeat count for the listed branches on entry, on the built moves it shares
# with the search stages. With the source knobs above at 25 the declaration
# changes VGB 1 / PSD 10 / GALFOR 10 -> 25 and, since 2026-10-04 ("make all
# the pe repeats 25 (for both newborns and survivors)"), the GB RJ moves'
# in-model polish budgets GB_INMODEL_REPEATS_NEWBORN 100 / _SURVIVOR 50 -> 25
# (search stages keep 100 / 50). EMRI joined the list on
# 2026-10-03 ("same as pe" for mbhs, emris and sobhbs); an
# EMRI_NUM_PROP_REPEATS=2 on the line therefore needs
# PE_INMODEL_REPEATS_BRANCHES=gb,vgb,sobbh,mbh,psd,galfor as well to keep EMRI
# at 2 through PE. Unset PE_INMODEL_REPEATS = every move as built.
export PE_INMODEL_REPEATS=${PE_INMODEL_REPEATS:-25}
export PE_INMODEL_REPEATS_BRANCHES=${PE_INMODEL_REPEATS_BRANCHES:-gb,vgb,sobbh,mbh,emri,psd,galfor}
export MBH_PERMUTE_EVERY=10
export EMRI_PERMUTE_EVERY=10
export SOBBH_PERMUTE_EVERY=10
# SEARCH-STAGE COLD -> HOT ROTATION of the source ladders (user ruling
# 2026-10-14, after the 1yr's EMRI 0 was held by one walker at SNR 28.7 and
# lost by the other three at SNR 4-8): "during all search parts of the
# sampler, when source search is run (EMRI, MBHB, SOBHB), we should be
# copying the cold-chain into the hot chain rotating by walker so it does not
# pass to itself ... every 200 proposals for a source (which includes the 25
# repeats so 8 iterations)", then "every 5 iterations during the search stages
# rather than 8"). Every {BRANCH}_COLD_TO_HOT_EVERY proposals (25
# repeats x 5 proposes = 125) in a search stage (source_search, gb_search_seed/1/2;
# never full_pe) each walker's cold point of every leaf is copied into the
# hottest sampled rung of walker w + k, k = 1 + event mod (nwalkers - 1):
# never itself, every other walker in turn. Scored at the next leaf visit
# against the receiving walker's own residual; a better point is swapped
# down the ladder by the per-repeat temperature swaps, a worse one is
# re-anchored by the lost-rung guard. In the gb_search stages the source moves
# propose once per CYCLE, so 125 = every 5 cycles there. Code default 125 (this block is for
# visibility; 0 disables). Log: "[MBH] cold->hot rotate event ...".
export MBH_COLD_TO_HOT_EVERY=${MBH_COLD_TO_HOT_EVERY:-125}
export EMRI_COLD_TO_HOT_EVERY=${EMRI_COLD_TO_HOT_EVERY:-125}
export SOBBH_COLD_TO_HOT_EVERY=${SOBBH_COLD_TO_HOT_EVERY:-125}
# EXACT-TRUTH STARTS + NO SOURCE_SEARCH STAGE (user ruling 2026-09-14
# late): "start all the emri, mbh, vgb, sobhb at the true points exactly
# ... they switched to the eigen proposal, we do not [need] a spread like
# with the stretch proposal." *_START_FACTOR=0 is exact truth (the
# run.py seeders are x*(1+f*randn), no floor; VGB's additive ratio
# jitter also scales by its factor, so 0 = exact truth-null there too).
# With nothing to converge, the source_search stage is SKIPPED: sources
# are subtracted at truth from setup_acs onward, sit frozen through the
# noise stages, and their eigen PE proposals run only in gb_search +
# full_pe (where they already ride).
# VGB pairs with VGB_INMODEL_PROPOSAL=eigen (block below, 09-15 ruling):
# the eigen draw samples fine from identical starts, so VGB is NOT
# frozen at truth -- it PE-samples around it from the first propose.
# ---- FLAGSHIP: THE FIXED-DIM SOURCE SEED (user ruling 2026-10-08) ----------
# "since we are seeing the MBHs move quite a bit towards a better likelihood
# ... 'fixed-dim source seed' ... the galfor and psd will be fixed just like in
# gb search seed and gb search 1. However, it will only run MBHBs, SOBHBs, and
# EMRIs, until the maximum likelihood of each walker converges." The existing
# source_search stage, RUN here (STAGE_SKIP_SOURCE_SEARCH 1 -> 0) and placed
# AFTER gb_search_seed (SOURCE_SEARCH_POSITION=after_seed, main session
# 2026-10-08: the source maxima are found against a residual with the
# warm-started galaxy already subtracted; =first puts it before the seed).
# Moves: sobbh_pe / mbh_pe / emri_pe ONLY, from their start points (the parent's cold chain for SNR > 10 sources, exact truth otherwise: the SOURCE WARM START block below);
# psd + galfor FIXED at the start pin, the GB leaves held at the seed's
# warm-start births, the VGBs held. Stop: every walker's cold lnL gains no
# more than SOURCE_SEARCH_TOL nats over SOURCE_SEARCH_CHECKS consecutive
# rounds (a round = one pass of the three moves at their in-model repeats;
# MAXLOGL_PER_WALKER=1, the laggard walker decides), at most
# SOURCE_SEARCH_MAX_ROUNDS rounds, MAXLOGL_ITERS_PER_STEP (10) rounds per
# stored row. MAXLOGL_TOL=20 / NOISE_SEARCH_CHECKS stay the gb_search_2
# cycle-end noise slot's.
export STAGE_SKIP_SOURCE_SEARCH=${STAGE_SKIP_SOURCE_SEARCH:-0}
export SOURCE_SEARCH_POSITION=${SOURCE_SEARCH_POSITION:-after_seed}
export SOURCE_SEARCH_CHECKS=${SOURCE_SEARCH_CHECKS:-10}
export SOURCE_SEARCH_TOL=${SOURCE_SEARCH_TOL:-10}
export SOURCE_SEARCH_MAX_ROUNDS=${SOURCE_SEARCH_MAX_ROUNDS:-200}
export MBH_START_FACTOR=0.0
export EMRI_START_FACTOR=0.0
export SOBBH_START_FACTOR=0.0
export VGB_START_FACTOR=0.0
# ---- FLAGSHIP: SOURCE WARM START from the parent's cold chain (user ruling
# 2026-10-08) -------------------------------------------------------------
# "start the 9mo sources from the 6mo cold chain's final positions instead of
# catalogue truth ... for any source over SNR 10 at 6 mo. Otherwise start how
# we did at 6mo [exact truth] for sources under SNR 10 (computed at six
# months)". lisatools.globalfit.warmstart.sources reads the seed store's last
# written cold-chain row (sub_backend/<branch>/chain + h_h; the running backup
# copy if the primary is torn), maps leaves by CATALOGUE ID (the seed's own id
# lists below, NOT this run's: an MBHB that merges after the parent's data
# end was never fitted there and starts at truth), takes SNR_parent = median
# over the cold walkers of sqrt(h_h),
# and for SNR > SOURCE_WARM_START_SNR_MIN copies walker w -> w and the whole
# 8-rung ladder (the rung counts match). Below the threshold, or absent from
# the seed: exact truth as above. A NaN record REFUSES the launch
# (SOURCE_WARM_START_SNR_UNKNOWN=truth|warm overrides); a start outside this
# run's prior refuses it. Fresh start only (a resume keeps its chain). DRY RUN
# on the cluster BEFORE launching, once per branch (prints each id's 6mo SNR
# and the warm / truth decision):
#   python -m lisatools.globalfit.warmstart.sources --store $GF_SEED_STORE \
#       --branch mbh --ids 0,2,3,4,5,6,7,9,10,12,13,15,16,18,19
# ⚠ OFF BY DEFAULT SINCE 2026-10-22 (Mike: "We want those [VGB, MBHB, SOBHB,
# EMRI] to always start at the catalog injected values rather than the last
# run's values"): every source leaf starts at exact truth (*_START_FACTOR=0
# above), like the no-parent launchers. Opt back in for one launch with
# SOURCE_WARM_START_STORE=$GF_SEED_STORE on the line; the id lists below are
# kept for that and are inert otherwise. A RESUMED run keeps its chain; to put
# a stopped run's sources back at the injection, edit its store first with
# scripts/fstat_proposal/set_store_sources_to_injection.py, then relaunch.
export SOURCE_WARM_START_STORE=${SOURCE_WARM_START_STORE-}
export SOURCE_WARM_START_SNR_MIN=${SOURCE_WARM_START_SNR_MIN:-10}
export MBH_SOURCE_WARM_START_IDS=${MBH_SOURCE_WARM_START_IDS:-0,2,3,4,5,6,7,9,10,12,13,15,16,18,19}   # the PARENT's MBHB_IDS (runs.py)
export EMRI_SOURCE_WARM_START_IDS=${EMRI_SOURCE_WARM_START_IDS:-0,1,2,3,4,5,6,7}
export SOBBH_SOURCE_WARM_START_IDS=${SOBBH_SOURCE_WARM_START_IDS:-0,1,2,3,4,5}
# SOBBH chunked scoring width: converged value (S3 ruling; the 11-h-layer
# stress result -- cheap insurance on this 1-h-layer production grid too).
export SOBBH_M_BAND_HALF_WIDTH=3
# Thin the slow-path A/B re-score (per-row TDI-on-the-fly x all rungs);
# it still fails loudly past tolerance when it fires. The counter advances
# once per SOBBH move VISIT (not per repeat); on a check visit every leaf
# scores its rows once through the exact full-TD path (unmeasured on the
# GPU; orders of magnitude above the ~31 ms fast call). Yielding since
# 2026-10-03 (was a hard `=30` that silently overwrote a launch-line value):
# the SOBBH session recommends 10 for the first lookup segment, back to 30
# if the check costs more than ~10 % of the SOBBH leg.
export SOBBH_CHECK_LL_EVERY=${SOBBH_CHECK_LL_EVERY:-30}
# The check itself is OFF by default since 2026-10-07 (V9-30: SOBBH_CHECK_LL=0
# on Mike's production line -- on job 727 the per-container slow-path
# re-score took ~525 s x 3 rounds = 26 min and it never warned in 715-738);
# SOBBH_CHECK_LL=1 on the line re-arms it at the SOBBH_CHECK_LL_EVERY cadence.
export SOBBH_CHECK_LL=${SOBBH_CHECK_LL:-0}

# ---- EIGEN INNER MOVE (2026-09-05/08 work; code defaults, pinned) ----------
# The addremove branches' in-model proposal is the eryn EigenAxisMove: a
# one-axis jump along eigenvectors of each source's own information matrix
# (per-axis curvature widths, prior-capped, factors=0 -- tables frozen
# between refreshes). Replaces the stretch as the DEFAULT inner move; the
# laptop A/B on the real chunked SOBBH kernel measured acceptance 0.544 vs
# stretch 0.156 at equal per-step cost. {BRANCH}_INNER_MOVE_KIND=stretch
# is the per-branch escape back to the legacy stretch, exactly.
# GUARDED (2026-09-16) for the scoring-regression A/B: production
# measures 61 ms/row vs the kernel's 2.78 ms/row job-373 reference.
# One-window discriminator: `SOBBH_INNER_MOVE_KIND=stretch sbatch <this>`
# -- leaf time collapsing toward ~15 s indicts the eigen inner-move path;
# staying ~190 s indicts the scoring seam. Default stays eigen (it beat
# stretch 0.544 vs 0.156 acceptance in the laptop A/B).
export SOBBH_INNER_MOVE_KIND=${SOBBH_INNER_MOVE_KIND:-eigen}
export MBH_INNER_MOVE_KIND=eigen
export EMRI_INNER_MOVE_KIND=eigen
# ---- MBH SKY-MODE HOPS (2026-10-12) ----------------------------------------
# The MBH inner-move stack mixes the LISA-frame sky-mode hops back in. The
# production basis is ICRS (alpha / sin_delta / psi), so the SkyMove converts
# each row ICRS -> ecliptic -> constellation frame at the row's OWN orbital
# phase (read off the run's orbits; t_plunge is referenced to
# mbh_waveform_t0), hops there (the 8 modes: plane reflection x 90-degree
# in-plane rotations; the merger keeps its DETECTOR arrival time, so t_plunge
# moves by up to ~2 R/c), converts back, and carries the frame Jacobian in
# the Metropolis factor (exact detailed balance). Verified 2026-10-12 on the
# LAT time-domain response (equal-arm + ESA orbits): every hopped mode
# overlaps the original A/E at > 0.99 with no residual time shift at 3 mHz.
# Draw fractions of the per-leaf inner-move draws (Mike 2026-10-12): 0.02
# the full 8-mode hop, 0.04 the plane reflection only, 0.02 the in-plane
# rotation only; the remaining 0.92 is the eigen move above. 0 disables a
# kind. Code defaults are the same values (this block pins them).
export MBH_SKY_MOVE_BOTH_WEIGHT=${MBH_SKY_MOVE_BOTH_WEIGHT:-0.02}
export MBH_SKY_MOVE_LAT_WEIGHT=${MBH_SKY_MOVE_LAT_WEIGHT:-0.04}
export MBH_SKY_MOVE_LONG_WEIGHT=${MBH_SKY_MOVE_LONG_WEIGHT:-0.02}
# ---- MBH PROPOSALS IN THE LISA FRAME (2026-10-19) --------------------------
# Mike 2026-10-19: "There are clear difficulties in the equatorial frame. Can
# we keep the same sampling/waveform parameters on the MBHB, but when we do
# the sampling moves, can we move in the LISA reference frame (similar to
# what is needed for the MBHB sky move) ... Only the move that produces the
# new coordinates should be moved in the LISA frame." The sampling basis and
# the waveform stay ICRS (alpha / sin_delta / psi / t_plunge); every MBH inner
# move (eigen, jump mixture, cross-walker jump) proposes in (lam_L, sinbeta_L,
# psi_L, t_L) -- the constellation frame at the row's own orbital phase, the
# sky-hops' map -- and the proposal is mapped back with the frame Jacobian in
# the Metropolis factor (|det J| = 1 + O(Omega R / c), ln|det J| +-2-5e-5).
# The eigen tables, the adaptation, the controller and the jump partners all
# work in that frame; the sky hops keep converting themselves. Why (9mo job
# 773, rows 858-1007, 600 cold samples per leaf): in ICRS t_plunge is a
# near-deterministic, partly NONLINEAR function of the sky (R^2 of t on
# alpha / sin_delta / psi: linear 0.87-0.998, quadratic 0.997-1.000; t widths
# 3-243 s); in the LISA frame t_L is 20-75x narrower (0.17-3.2 s) and
# decoupled (R^2 0.02-0.28); the standardized 4x4 condition number drops
# 304 -> 3.7, 2314 -> 119, 2923 -> 556, 421 -> 9.8 (leaves 1-4). A Gaussian
# step along a Fisher axis in ICRS leaves the curved t(sky) relation and is
# rejected. Code default "sampling" (bit-identical); rollback
# MBH_PROPOSAL_FRAME=sampling on the launch line. Log, once per leaf refresh:
#   [MBH_EIGEN] leaf k: proposals in the LISA frame (t_L sigma 0.42 s vs
#   t_plunge 31.00 s in the sampling basis)
# A frame-tagged eigen-table sidecar entry is never adopted in the other
# frame (the first refresh after a switch rebuilds).
export MBH_PROPOSAL_FRAME=${MBH_PROPOSAL_FRAME:-lisa}
# ---- EIGEN JUMP MIXTURE (2026-10-13) ---------------------------------------
# A share of each branch's eigen inner-move proposals draws its axis step at
# {BRANCH}_EIGEN_JUMP_FACTOR x the table width (lisatools.sampling.moves.
# eigenjumpmix.EigenAxisJumpMixMove: a fixed mixture of two symmetric
# kernels, the Metropolis factor stays 0). Why: on the 9mo v9 the loud
# EMRIs' cold walkers sat on separate islands 23-38 nats below the best
# walker with no bridge; a 1-sigma axis step cannot cross that. Mike
# 2026-10-13: "if this is going to improve sampling in general then lets
# try it. We can always rewind. but let's make sure it takes effect for
# EMRIs so we can make this current sampling more efficient." CODE defaults
# stay mix 0 (bit-identical to the plain EigenAxisMove); this block turns it
# ON (10 % of the draws at 3 sigma) for all three branches. The job log
# proves it at move construction, one line per branch and rank:
#   [EMRI] inner-move stack: EigenAxisJumpMixMove 1.000 (jump mix 0.10 x 3.0)
# MBH: the mix rides the eigen part (0.92) only, never the sky hops above.
# Rollback: {BRANCH}_EIGEN_JUMP_MIX=0 on the launch line.
export EMRI_EIGEN_JUMP_MIX=${EMRI_EIGEN_JUMP_MIX:-0.1}
export EMRI_EIGEN_JUMP_FACTOR=${EMRI_EIGEN_JUMP_FACTOR:-3.0}
export MBH_EIGEN_JUMP_MIX=${MBH_EIGEN_JUMP_MIX:-0.1}
export MBH_EIGEN_JUMP_FACTOR=${MBH_EIGEN_JUMP_FACTOR:-3.0}
export SOBBH_EIGEN_JUMP_MIX=${SOBBH_EIGEN_JUMP_MIX:-0.1}
export SOBBH_EIGEN_JUMP_FACTOR=${SOBBH_EIGEN_JUMP_FACTOR:-3.0}
# Per-axis STEP-SCALE ADAPTATION of the source eigen tables (user ruling
# 2026-10-14: "some of the emris are sampling terribly ... look at the eigen
# proposals"). The information matrix at one point sets the DIRECTIONS but
# the move steps exactly one LOCAL sigma; along the EMRIs' degenerate
# intrinsic ridge (ln M, mu, a, p0, e0) the chain's own spread on the 9mo
# was 1.4-56x that sigma (EMRI 7 56x, 4 13x, 0/6 5.5x), so the four cold
# walkers sat stranded along ONE ridge at equal SNR (within-island
# autocorrelation 2-6 iterations, no traversal). With the switch on each
# rank measures the width per table axis from its OWN rung ensemble (the
# last {BRANCH}_EIGEN_ADAPT_VISITS=50 visits, each rung's deviations x
# sqrt(beta) to cold units, robust MAD, periodic columns wrapped), floors
# it at the Fisher sigma and caps it at the prior box; the axes stay; the
# adapted sigmas persist in the eigen-table sidecar. Synthetic 20x ridge:
# autocorrelation along it 917+ -> 48. Log: "[EMRI_EIGEN] leaf k: step
# scale adapted on N of D axes ...". Code default OFF (stock runs
# bit-identical); rollback {BRANCH}_EIGEN_AXIS_ADAPT=0 on the launch line.
export EMRI_EIGEN_AXIS_ADAPT=${EMRI_EIGEN_AXIS_ADAPT:-1}
export MBH_EIGEN_AXIS_ADAPT=${MBH_EIGEN_AXIS_ADAPT:-1}
export SOBBH_EIGEN_AXIS_ADAPT=${SOBBH_EIGEN_AXIS_ADAPT:-1}
# POOLED + TEMPERED source eigen steps and the widening cap (user ruling
# 2026-10-15 after the 9mo EMRI dive: walkers 1 and 3 of EMRI 3 had ladders
# whose EVERY rung stayed within 1e-4 of p0 while walker 0's hot rungs spanned
# the whole fixed-plunge ridge, because every rung stepped with the same
# cold-sized sigma and a per-rank adaptation can only learn 1e-4 from a stuck
# walker). POOL: every rank adapts from the UNION of all walkers' rung
# ensembles, shipped by the head in the propose payload (needs
# _EIGEN_AXIS_ADAPT=1; no-op under GF_GPU_ROUTING=1, where the head already
# holds every walker). TEMPER: rung k steps sigma / sqrt(beta_k), capped at
# the table's own cap and never below the cold sigma; the sidecar keeps the
# cold table. MAX_FACTOR: the adapted width may exceed the Fisher sigma by at
# most this factor -- the 1yr tar 2 showed x600-4900 on the narrowest axis
# where hot rungs straddle sky- / phase-degenerate solutions (one wasted axis
# in ndim), while the 9mo ridges needed at most x56. Synthetic 4-walker x
# 8-rung ridge: the stuck walkers reach +1 sigma at visit 10 instead of 19-38,
# autocorrelation 167 -> 41. Code defaults OFF / OFF / 100 (stock runs
# bit-identical); rollback =0 / =0 / =0 (uncapped) on the launch line.
export EMRI_EIGEN_ADAPT_POOL=${EMRI_EIGEN_ADAPT_POOL:-1}
export MBH_EIGEN_ADAPT_POOL=${MBH_EIGEN_ADAPT_POOL:-1}
export SOBBH_EIGEN_ADAPT_POOL=${SOBBH_EIGEN_ADAPT_POOL:-1}
export EMRI_EIGEN_TEMPER_SIGMAS=${EMRI_EIGEN_TEMPER_SIGMAS:-1}
export MBH_EIGEN_TEMPER_SIGMAS=${MBH_EIGEN_TEMPER_SIGMAS:-1}
export SOBBH_EIGEN_TEMPER_SIGMAS=${SOBBH_EIGEN_TEMPER_SIGMAS:-1}
export EMRI_EIGEN_ADAPT_MAX_FACTOR=${EMRI_EIGEN_ADAPT_MAX_FACTOR:-100}
export MBH_EIGEN_ADAPT_MAX_FACTOR=${MBH_EIGEN_ADAPT_MAX_FACTOR:-100}
export SOBBH_EIGEN_ADAPT_MAX_FACTOR=${SOBBH_EIGEN_ADAPT_MAX_FACTOR:-100}
# ---- THE EMRI SAMPLING AUDIT (user ruling 2026-10-18; docs/emri-sampling-audit.md)
# Mike: "do a fully audited and adjusted EMRI setup to get this to be better"
# and "we do not have enough walkers for the stretch move, so we need to
# improve what we have". The 9mo (job 773, rows 840-1007) EMRI posteriors are
# a COMB of discrete islands along the intrinsic degeneracy (EMRI 3: p0
# 12.3307 / 12.3310 / 12.3329 / 12.3346 / 12.3371, a / logm1 / m2 / e0 sliding
# along, all within ~10 nats; EMRI 0: seven islands in (p0, e0, Phi_r0)); every
# walker's ladder holds one or two islands, nothing moves a walker between
# them (in PE there is no cross-walker channel at all: the walker-permuting
# swap never fires at one walker per rank, the cold -> hot rotation is
# search-only), and the pooled adaptation widened the narrow intrinsic axes
# x14-100 from the BETWEEN-island offsets, where the walkers' own spread is
# 0.04-0.2 of the step: 4 of 12 axes dead on EMRI 3, 3 on EMRI 0.
#   {BRANCH}_EIGEN_AXIS_ACC_LOG=1: per-axis proposals / acceptances of the
#     eigen move, cold vs hot rungs, one "[EMRI_EIGEN] leaf k (walker w): axis
#     acceptance over the last 10 visits, cold ..." line per refresh per rank,
#     and the adaptation line names the widened axes' cold acceptance.
#     Telemetry only (the proposal stream is bit-identical); on for MBH and
#     SOBBH too, nothing else of theirs changes.
#   EMRI_EIGEN_AXIS_ACC_TUNE=1: after each refresh window an axis whose COLD
#     acceptance fell below _ACC_MIN (0.1) has its step divided by _ACC_FACTOR
#     (2; never below the Fisher sigma), one above _ACC_MAX (0.5) multiplied
#     (never above the adapted width); frozen within a visit. Synthetic comb:
#     the dead axes' cold acceptance 0.06 / 0.15 -> 0.32 / 0.38.
#   EMRI_DE_JUMP_MIX=0.2: the cross-walker difference jump x' = x + gamma
#     (x_a - x_b) (lisatools.sampling.moves.walkerdiffjump), partners = the
#     start-of-propose rung ensembles of the OTHER walker group (walkers {0,2}
#     and {1,3} alternate by global iteration, so it stays exact), gamma = 1
#     (island to island) with probability _MODE_FRAC (0.5), else 2.38 /
#     sqrt(2 d) x N(1, 0.1); 20 % of the stack, the eigen move the rest. The
#     partners ride in the propose payload (8 leaves x 8 rungs x 4 walkers x
#     12 floats, shared with the pooled adaptation sample). Synthetic comb (4
#     walkers x 8 rungs, two islands 45 nats apart): 0 island switches in 200
#     visits -> 70-98 per walker. Log: "[EMRI_DE] leaf k (walker w):
#     cross-walker jumps over the last 10 visits (active on 5 of 10): mode cold
#     a/b hot c/d, scaled ...", and the stack line "[EMRI] inner-move stack:
#     EigenAxisJumpMixMove 0.800 (jump mix 0.10 x 3.0) | WalkerDifferenceJumpMove
#     0.200 (cross-walker, mode frac 0.50)".
# Code defaults OFF / OFF / 0 (stock runs bit-identical); rollback =0 on the
# launch line. MBH / SOBBH (user ruling 2026-10-18, "do this for the other
# sources"): the CONTROLLER on for both -- the pooled adaptation had widened
# MBH's intrinsic axes until its cold acceptance fell from 0.4 to 0.07-0.21
# on four leaves, and SOBBH 0/1 sat at 0.09-0.21 -- and the JUMP at 0.2 for
# SOBBH (leaves 0/1 are bimodal in the masses at SNR 9-10, walkers in
# different modes) and 0.1 for MBH (posterior-sized correlated steps; the
# sky reflections already have the sky-hop move). The controller's band
# (0.1-0.5, factor 2) and the mode fraction (0.5) are the code defaults.
export EMRI_EIGEN_AXIS_ACC_LOG=${EMRI_EIGEN_AXIS_ACC_LOG:-1}
export MBH_EIGEN_AXIS_ACC_LOG=${MBH_EIGEN_AXIS_ACC_LOG:-1}
export SOBBH_EIGEN_AXIS_ACC_LOG=${SOBBH_EIGEN_AXIS_ACC_LOG:-1}
export EMRI_EIGEN_AXIS_ACC_TUNE=${EMRI_EIGEN_AXIS_ACC_TUNE:-1}
export MBH_EIGEN_AXIS_ACC_TUNE=${MBH_EIGEN_AXIS_ACC_TUNE:-1}
export SOBBH_EIGEN_AXIS_ACC_TUNE=${SOBBH_EIGEN_AXIS_ACC_TUNE:-1}
export EMRI_EIGEN_AXIS_ACC_MIN=${EMRI_EIGEN_AXIS_ACC_MIN:-0.1}
export EMRI_EIGEN_AXIS_ACC_MAX=${EMRI_EIGEN_AXIS_ACC_MAX:-0.5}
export EMRI_EIGEN_AXIS_ACC_FACTOR=${EMRI_EIGEN_AXIS_ACC_FACTOR:-2}
export EMRI_DE_JUMP_MIX=${EMRI_DE_JUMP_MIX:-0.2}
export MBH_DE_JUMP_MIX=${MBH_DE_JUMP_MIX:-0.1}
export SOBBH_DE_JUMP_MIX=${SOBBH_DE_JUMP_MIX:-0.2}
export EMRI_DE_JUMP_MODE_FRAC=${EMRI_DE_JUMP_MODE_FRAC:-0.5}
# TABLE SCOPE. SOBBH is set ABOVE (SOBBH_EIGEN_SCOPE, walker_max per the
# 2026-09-16 ruling; the 2026-09-08 per-walker ruling it superseded cost
# 12 rungs x 10 walkers = 120 points x ~245 rows ~ 29k batched likelihood
# rows ~ 1.4 min per LEAF REFRESH at 2.78 ms/row, and the per-walker stash
# is discarded on every walker-block change). SOBBH_EIGEN_REFRESH stretches
# the refresh cadence either way. Do NOT re-add an export here: a second
# export silently overrode the ruling above until 2026-09-17.
export SOBBH_EIGEN_REFRESH=10
# MBH/EMRI: ONE table per leaf, built at the max-lnL COLD walker and shared
# by every walker and rung. REFRESH EVERY 10 VISITS since 2026-10-08 (Mike:
# "update them now with fast waveforms to do the same thing as the sobhbs psd
# and galfor"; was 100). The 100 dates from the likelihood ("ll") route,
# whose ~245 dense likelihood rows cost ~6.4 min/leaf (MBH) / ~4.6 min/leaf
# (EMRI). Under {MBH,EMRI}_EIGEN_INFO=gram (the V9-30 default below) a build
# is ~2 x (2 ndim + 1) of the move's own FAST templates -- the batched
# windowed MBH generator (~0.02 s/row) and the EMRI direct-to-WDM adapter --
# so an expected few seconds per leaf (SOBBH's lookup Gram measures ~0.6 s;
# the MBH/EMRI builds were not yet timed on the cluster -- read the
# "[eigen_refresh] <branch> leaf N Gram info matrix ... in X s" lines). The
# first Gram rebuild (job 744,
# store row ~690) widened the chains from 30-50x inside the Fisher width
# toward it: the ll tables' steps were 10-900x too small. Escape:
# {BRANCH}_INNER_MOVE_KIND=stretch above.
export MBH_EIGEN_SCOPE=walker_max
export EMRI_EIGEN_SCOPE=walker_max
export MBH_EIGEN_REFRESH=${MBH_EIGEN_REFRESH:-10}
export EMRI_EIGEN_REFRESH=${EMRI_EIGEN_REFRESH:-10}
# WATCH ON FIRST LAUNCH: any "[eigen_refresh] ... fallback" WARNING means
# a table build failed and that leaf is sampling on identity/1%-width
# tables (correct but slow -- MH corrects the shape); a steady stream of
# them means arm the stretch escape and file the traceback.
#
# ---- EIGEN TABLES FROM THE EXPECTED INFORMATION (2026-10-06; DEFAULT ON
# ---- since 2026-10-07, V9-30) -------------------------------------------
# Exported at the end of this block with Mike's production-line values:
# gram for the three source branches, fisher for the two noise branches.
# {BRANCH}_EIGEN_INFO=ll on the launch line restores the second-difference
# tables above. CPU-validated (2026-10-06); the first GPU run is the launch
# that follows. Route map, numbers and tests: docs/eigen-info-routes.md.
#
# {SOBBH,EMRI,MBH}_EIGEN_INFO=gram (move attribute eigen_info). The table is
#   built from the Gram matrix <dh_a|dh_b> of the move's OWN batched
#   templates (SOBBH: the lookup fill under SOBBH_LIKELIHOOD=lookup; EMRI:
#   the direct template; MBH: the batched windowed template) at the max-lnL
#   cold walker, through that walker's PSD -- the expected information, no
#   residual noise, positive semi-definite. ONE shared table per leaf (the
#   {BRANCH}_EIGEN_SCOPE knob does not apply). Any failure WARNS
#   ("[eigen_refresh] ... Gram build failed") and builds the ll table.
#   Why: on the SOBBH toy the ll route's 1e-4-of-box step gives an f_low
#   curvature ~250x too small, noise-dominated spin entries, and masses
#   ~1.5x off the Gram. Companion knobs:
#   {BRANCH}_EIGEN_GRAM_TARGET (default 1e-3): each column's step is
#     rescaled once so the template moves by this fraction of ||h||;
#     0 = fixed steps.
#   {BRANCH}_EIGEN_GRAM_EPS_REL (default {BRANCH}_EIGEN_EPS_REL, 1e-4): the
#     starting step, fraction of the prior box.
#   SOBBH holds 23 data-shaped WDM slabs during a build (~0.6 GB at 6
#   months, ~1.2 GB at 1 year).
# {PSD,GALFOR}_EIGEN_INFO=fisher (PSDMove attribute eigen_info). The noise
#   tables come from the expected Fisher kappa * sum Tr(C^-1 dC_i C^-1 dC_j)
#   of the SAME PSD_BATCH covariance the noise scorer builds -- positive
#   semi-definite everywhere, which the observed curvature is not (34 of 55
#   galfor builds non-positive on the 3mo run). PSD_BATCH route only; any
#   failure WARNS ("[eigen_refresh] ... noise Fisher failed") and builds the
#   ll table.
#
# e.g.  SOBBH_EIGEN_INFO=ll GALFOR_EIGEN_INFO=ll ./submit...   (back to ll)
# WATCH: "[eigen_refresh] <branch> leaf N Gram info matrix at walker W in
# X s" / "[eigen_refresh] galfor: noise Fisher tables at N point(s) in X s"
# lines (one per refresh), and no "failed" warnings.
export SOBBH_EIGEN_INFO=${SOBBH_EIGEN_INFO:-gram}
export EMRI_EIGEN_INFO=${EMRI_EIGEN_INFO:-gram}
export MBH_EIGEN_INFO=${MBH_EIGEN_INFO:-gram}
export PSD_EIGEN_INFO=${PSD_EIGEN_INFO:-fisher}
export GALFOR_EIGEN_INFO=${GALFOR_EIGEN_INFO:-fisher}

# ============================================================================
# MBH LIKELIHOOD: BATCHED + WINDOWED (2026-09-30). Spec:
#   docs/superpowers/specs/2026-09-29-mbh-batched-windowed-likelihood-design.md
# ============================================================================
# mbh_pe becomes MBHBatchedLikeMove (build_mbh_move_runtime,
# stock/erebor/source_runtime.py; built on EVERY compute rank). Each chunk of
# up to MBH_BATCH_MAX_SIZE rows is ONE grid-aligned phentax + response launch
# on the leaf's own window -- 90 d before to 10 d after the median cold-chain
# merger (+-1 d margin, 4 d discarded pad per side) -- transformed on its own
# WDM segment and scored against each walker's own residual AND PSD. The
# expose/fold, the in-model rows and the 243-row eigen-table sweep all go
# through it. The per-row path it replaces measured 1.16-1.43 s/row (jobs
# 373/487/558). GPU benchmark on THIS run's 6-month grid (H100):
#     B= 8   0.037 s/row    +8.5 GB device memory per launch
#     B=16   0.034 s/row   +18.5 GB
#     B=24/32              ~+30 GB
# B=8 IS PINNED (ruling 2026-09-30): 16 is only 8% faster per row for more than
# twice the memory, and these cards also hold the GB buffers. In this layout
# (NWALKERS=4 on 4 compute ranks = 1 walker/rank, MBH_NTEMPS=2) an in-model
# step scores 2 rows and an expose/fold 1, so the cap binds only on the
# eigen-table sweep -- that is where the +8.5 GB lands. The first call per
# batch SHAPE pays ~25 s of JIT, once per shape per process.
#
# WHAT ELSE MOVES WITH IT (engine residual rebuild AND move, together):
#   * waveform_duration = MBH_WINDOW_BEFORE_DAYS (90 d) instead of the per-row
#     path's 1/12 yr (~30 d), for BOTH the windowed and the stock generator,
#     so the residual rebuild, the scoring and the cross-check agree on the
#     inspiral length. MBH_WAVEFORM_DURATION must stay UNSET and USE_TDIONFLY
#     off: the build REFUSES either (resolve_mbh_batched_cfg; the preflight
#     below runs that same rule before the allocation is spent).
#   * the epoch snaps onto the data lattice (<= dt/2 = 1.25 s); t_plunge moves
#     by the same amount, so absolute merger times do not change.
# RESUME of a store sampled on the per-row path is SUPPORTED: no stored shape
# changes. setup_acs rebuilds every residual with the 90 d snapped stock
# generator and log_like is re-scored from it; MBH coords, nleaves (ids
# 2,5,16,18) and the per-leaf ladder are untouched. The persisted MBH eigen
# tables (<store>_eigen_tables.pkl, "mbh:<leaf>") were built on the OLD
# likelihood and are ADOPTED until their next MBH_EIGEN_REFRESH tick --
# MH-valid, just shaped for a 30 d template. A rebuild now costs ~10 s/leaf
# rather than the 6.4 min quoted above; to force one, drop the "mbh:" entries
# from that pickle before relaunching.
# ESCAPE: MBH_LIKELIHOOD=full is the per-row container path the store ran so
# far (add MBH_RESPONSE_ORDER=30 to match its response order as well).
export MBH_LIKELIHOOD=${MBH_LIKELIHOOD:-batched}
export MBH_BATCH_MAX_SIZE=${MBH_BATCH_MAX_SIZE:-8}
# Response Lagrange order 8 = the code default since the batched merge (PR
# #82: mismatch flat from order 30 down to 4; ~1.6x cheaper). The store ran
# 30 until now, so it is pinned here for the run record.
# MBH_RESPONSE_ORDER=30 on the launch line restores the old order.
export MBH_RESPONSE_ORDER=${MBH_RESPONSE_ORDER:-8}
# Window lattice DECIMATION FACTOR (2026-10-02; DEFAULT 2 since 2026-10-07, V9-30:
# the cluster speed/accuracy run passed and every production job since 715 ran
# MBH_WINDOW_DECIMATE=2 on the line; not the mass ratio): the batched template is generated,
# responded and WDM-transformed at decimation*dt on Nf/decimation layers (same
# 1-h pixels), ~decimation x less phentax + response + transform per row.
# Laptop: decimation 2 mismatch <= 1.5e-9 vs none at 6e5 and 6e6 Msun;
# decimation 4 fails the lightest (ringdown above the 50 mHz coarse Nyquist).
# The epoch snaps onto that lattice for BOTH generators (no stored coordinate
# changes; a resume may switch the factor).
export MBH_WINDOW_DECIMATE=${MBH_WINDOW_DECIMATE:-2}
# MBH_CHECK_LL_EVERY is deliberately NOT exported: the batched move defaults
# it to 10 (every 10th leaf visit re-scores the cold rung through the stock
# 90 d generator; warns past MBH_CHECK_LL_TOL = 0.5 nats). For the FIRST
# segment on the new path, MBH_CHECK_LL_EVERY=1 on the launch line gives a
# cross-check on every visit (one stock waveform per row on top).
# The check itself is OFF by default since 2026-10-07 (V9-30: MBH_CHECK_LL=0
# on Mike's production line; it never warned in jobs 715-738); MBH_CHECK_LL=1
# on the line re-arms it at the cadence above.
export MBH_CHECK_LL=${MBH_CHECK_LL:-0}
# WATCH: [MBH_BATCH] "leaf N: R rows ... s/row ..., F fallbacks,
# outside_box=K" (F and K must stay 0 -- a steady K means the walkers' merger
# times spread past MBH_WINDOW_MARGIN_DAYS); [MBH_FILL]; and the two warnings
# "batched windowed fast path vs slow container path disagree" and
# "EXPOSE INVARIANT VIOLATED". Device memory: gpu_util_*.csv around the first
# mbh_pe eigen sweep (JAX's default allocator keeps what phentax took; if the
# GB side gets tight, MBH_BATCH_MAX_SIZE=4 is the first lever).
#
# MBH PREFLIGHT. An unknown env var is SILENTLY IGNORED, so on an install that
# predates the batched merge this block would quietly run the per-row path
# while every line above says batched. Resolve the knobs through the real
# settings class and the real consistency rule instead, and refuse on any gap.
python - <<'PYEOF' || exit 2
import os
import sys

try:
    from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove  # noqa: F401
    from lisatools.globalfit.stock.erebor.source_runtime import (
        SourceMBHSettings, resolve_mbh_batched_cfg)
except ImportError as exc:
    print("[MBH-PREFLIGHT] REFUSING: the installed lisatools has no batched MBH "
          f"likelihood ({exc}). MBH_LIKELIHOOD would be SILENTLY IGNORED and "
          "mbh_pe would run the per-row path. Pull dev at/after the batched MBH "
          "merge, or launch with MBH_LIKELIHOOD=full.")
    sys.exit(2)
from lisatools.domains import WDMSettings

mbh = SourceMBHSettings()
# The resolver needs the run-domain spec (MBH_LIKELIHOOD=auto/batched check the
# domain is WDM). This run's grid: Nf 1440 x Nt 17280 at dt 2.5 s.
try:
    cfg = resolve_mbh_batched_cfg(
        mbh, domain_settings=WDMSettings.make_factory(1440, 17280))
except ValueError as exc:
    print(f"[MBH-PREFLIGHT] REFUSING: {exc}")
    sys.exit(2)
want = (os.environ["MBH_LIKELIHOOD"], int(os.environ["MBH_BATCH_MAX_SIZE"]),
        int(os.environ["MBH_RESPONSE_ORDER"]))
got = (cfg["mbh_likelihood"], int(cfg["mbh_batch_max_size"]),
       int(mbh.response_order))
if got != want:
    print(f"[MBH-PREFLIGHT] REFUSING: exported (likelihood, batch, order) = "
          f"{want} but the settings resolve {got}.")
    sys.exit(2)
dur = cfg["mbh_waveform_duration"]
dur_txt = "full span" if dur is None else "%.1f d" % (float(dur) / 86400.0)
decim_want = int(os.environ["MBH_WINDOW_DECIMATE"])
decim_got = int(cfg.get("mbh_window_decimate", 1))
if got[0] == "batched" and decim_got != decim_want:
    print(f"[MBH-PREFLIGHT] REFUSING: exported MBH_WINDOW_DECIMATE={decim_want} but the "
          f"settings resolve {decim_got} (an install that predates the knob ignores it).")
    sys.exit(2)
if decim_got > 1 and 1440 % decim_got:
    print(f"[MBH-PREFLIGHT] REFUSING: MBH_WINDOW_DECIMATE={decim_got} does not divide Nf=1440.")
    sys.exit(2)
print(f"[MBH-PREFLIGHT] mbh_pe scoring={got[0]} batch<={got[1]} "
      f"response_order={got[2]} waveform_duration={dur_txt} window "
      f"-{cfg['mbh_window_before'] / 86400.0:g}/+"
      f"{cfg['mbh_window_after'] / 86400.0:g} d decimate={decim_got}")
PYEOF

# EMRI MODE-SELECTION THRESHOLD (user ruling 2026-09-14). FEW's kwarg is
# ``mode_selection_threshold``; ``eps`` is its FEW 1.x name and what we still
# call the knob. The EFFECTIVE value was 1e-5 (the generator default), NOT the
# 1e-2 pinned in lisatools/sources/emri/response.py: that pin is a mode
# SELECTOR CONSTRUCTOR value, and FEW's _generate_waveform hands its own
# call-time default (1e-5, few/waveform/base.py:143) to the selector
# unconditionally, so the constructor value never applied. EMRI_EPS is wired
# at CALL time, through the EMRI wave wrap's runtime_kwargs -- the ONE wrap
# both the engine-side generator (template + residual) and the move-side
# generator resolve through, so it is on every EMRI waveform the run builds.
# It is deliberately NOT stamped into the branch waveform_kwargs: for EMRI
# those double as the move's LIKELIHOOD kwargs, and inner_product has no such
# parameter -- doing so crashed the 6mo run on 2026-09-15 (fixed same day).
# 1e-5 -> 1e-3 LOOSENS the cut: FEW keeps modes until the cumulative mode
# SNR^2 is within (1-threshold)^2 of the total (few/utils/modeselector.py:426),
# so fewer harmonics are kept = direct speedup on the dominant EMRI cost
# (EMRI likelihood rows measure ~1 s each at 6mo). FEW documents the trade as
# "Increasing this value removes modes from consideration and can have a
# considerable affect on the speed of the waveform, albeit at the cost of some
# accuracy (usually an acceptable loss)" (few/utils/modeselector.py class
# docstring), and its tutorial measures the two bracketing endpoints: 455
# modes at 1e-5 vs 25 modes at 1e-2, mismatch 1.5e-3 between them
# (examples/FastEMRIWaveforms_tutorial.ipynb, stored output). 1e-3 sits inside
# that bracket. The tiered-accuracy policy tolerates it, and the campaign IS
# the test.
# RESUME-SAFE: a template/proposal accuracy knob only -- no data, grid, or
# store-layout change, and nothing stamps waveform_kwargs into a store
# identity (the only semantic resume guard is the noise-model identity in
# globalfit/hdfbackend.py). Corollary: a resumed store will NOT flag the
# change, so it has to be recorded with the run.
# Unset or empty reverts to the generator default (1e-5).
export EMRI_EPS=1e-3

# ============================================================================
# EMRI LIKELIHOOD: DIRECT-TO-WDM (2026-10-02). Docs: docs/emri-direct-wdm.md
# ============================================================================
# EMRI_LIKELIHOOD=direct makes emri_pe an EMRIDirectLikeMove
# (build_emri_move_runtime, stock/erebor/source_runtime.py; built on EVERY
# compute rank). Each chunk of up to EMRI_BATCH_MAX_SIZE rows is one FEW call
# per row (trajectory + amplitudes at the integrator knots, modes kept at
# EMRI_EPS), ONE dense TDI-on-the-fly response launch for the chunk
# (TDDenseTDIWaveform: exact DOPR853 dense-output phases, link geometry shared
# across harmonics, on the trajectory knots + a 12 h cap) and ONE fused lookup-sum
# kernel launch (wdm_lookup_sum: sparse response -> analytic f, fdot -> n_ref
# table -> WDM sum, nothing per pixel in memory; EMRI_DIRECT_LOOKUP=python is the
# Python reference path) on this run's active band, cropped to its time box and
# scored against each walker's own residual AND PSD.
# A harmonic that chirps off the table's fdot axis near a plunge hands off to
# even-start 128-layer chunks of the EXACT dense-kernel time series, the source
# tapered over the last EMRI_DIRECT_PLUNGE_TAPER_S (120 s) of emission time
# before the stop (so the sampled stop leaves no low-frequency area). In-model steps,
# eigen-table sweeps and the inner-product record all score through it.
# TEMPLATE GENERATION follows: with direct, the engine's EMRI template generator
# (SourceSignalGen: residual rebuilds at load, the move's expose/fold) is the
# direct template too, so the residual every other branch sees holds exactly what
# the sampler scores. Only the cold-rung cross-check and the (loud) fallback
# stay on the production template (FEW + ResponseWrapper + dense TD->WDM).
# Accuracy vs production (H100 speed tests, CD1L EMRI 1, foreground-weighted
# noise, docs/emri-direct-wdm.md): mismatch 1.6e-5 at 6 months, 7e-6 at 12;
# lnL against the mojito stream -1.308 vs production -1.305 (6 months, eps
# 1e-3); an in-window plunge (720 d) 1.5e-5. The cold-rung cross-check
# warns past a per-point tolerance EMRI_CHECK_LL_TOL + mm<h|h> + 3 sqrt(2 mm<h|h>)
# (1 nat, mm = EMRI_CHECK_LL_MM = 3e-4: the template gap grows with SNR); every
# 10th visit by default -- EMRI_CHECK_LL_EVERY=1 on the launch line for the first
# segment.
# DEFAULT: direct (2026-10-03). H100, 6 months, eps 1e-3: 25 ms single and
# 15 ms/row at 16 rows per call vs production 88 ms (eps 1e-5: 38 / 18 vs 94).
# The library default is EMRI_LIKELIHOOD=auto, which resolves to direct on this
# WDM / XYZ grid; this launcher pins direct so a blocker REFUSES the launch instead
# of falling back to full with one INFO line.
# The per-row production path for both scoring and template generation:
#     EMRI_LIKELIHOOD=full ./submit_gf_24mo.sh
# LOOKUP TABLE: EMRI_DIRECT_TABLE points to a specific table; unset (default) it is
# the canonical file in this run's folder,
#     ${STORE_DIR}/wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5
# Either way, when the file is missing the preflight below BUILDS AND SAVES it there
# on this node's GPU before mpiexec (lisatools.wdm_lookup_store), so a restart finds
# it and never rebuilds; a found table is checked against the grid. To reuse a table
# built elsewhere, copy it into ${STORE_DIR} or point EMRI_DIRECT_TABLE at it.
# RESUME-SAFE: no stored shape changes; the load rebuilds the residual with the
# active template generator, every leaf visit re-scores its prev_logl, and the
# persisted EMRI eigen tables (built on the production likelihood) are adopted
# until their next EMRI_EIGEN_REFRESH tick (MH-valid). Switching a store between
# full and direct moves its EMRI lnL by ~mismatch x SNR^2 (~0.003 nat at SNR 14).
# CPUs: with direct, EMRI_TRAJ_WORKERS spawn processes per compute rank integrate
# a chunk's EMRI trajectories in parallel (few.trajectory.pool; only chunks of
# >= EMRI_TRAJ_WORKERS rows -- the eigen sweeps -- since an in-model step here is
# 2 rows); the self-dispatch above raises --cpus-per-task to
# 2 + EMRI_TRAJ_WORKERS to give them cores. EMRI_TRAJ_WORKERS=0 = serial.
# WATCH: [EMRI_DIRECT] "leaf N: R rows in C chunks ... s/row ..., F fallbacks,
# K rows refused by FEW" (F must stay 0: a fallback is a chunk scored on the
# slow path after a direct failure, warned once per leaf), "[EMRI_DIRECT]
# trajectory pool: ... started (this process may run on N cores)", and the
# warnings "direct-to-WDM fast path vs production container path disagree" and
# "EXPOSE INVARIANT VIOLATED".
export EMRI_LIKELIHOOD=${EMRI_LIKELIHOOD:-direct}
export EMRI_BATCH_MAX_SIZE=${EMRI_BATCH_MAX_SIZE:-8}
export EMRI_DIRECT_TABLE=${EMRI_DIRECT_TABLE:-}
export EMRI_DIRECT_RESPONSE=${EMRI_DIRECT_RESPONSE:-dense}
export EMRI_TRAJ_WORKERS=${EMRI_TRAJ_WORKERS:-8}   # 8 since 2026-10-07 (V9-30; was 4)
export EMRI_DIRECT_LOOKUP=${EMRI_DIRECT_LOOKUP:-kernel}
# The cold-rung cross-check is OFF by default since 2026-10-07 (V9-30:
# EMRI_CHECK_LL=0 on Mike's production line; it never warned in 715-738);
# EMRI_CHECK_LL=1 on the line re-arms it (every 10th visit, see above).
export EMRI_CHECK_LL=${EMRI_CHECK_LL:-0}
#
# EMRI PREFLIGHT. An unknown env var is SILENTLY IGNORED, so resolve the knobs
# through the real settings class and the real consistency rule; for direct,
# also check the edge crop clears the template's zeroed edge layers, run the dense
# kernel's (and, with EMRI_DIRECT_LOOKUP=kernel, the fused lookup kernel's)
# GPU-vs-CPU parity test on this node's GPU, and find -- or build and save -- the
# lookup table (above). Refuses before mpiexec on any gap.
python - <<'PYEOF' || exit 2
import dataclasses
import inspect
import os
import sys

try:
    from lisatools.globalfit.moves.emridirectmove import EMRIDirectLikeMove  # noqa: F401
    from lisatools.globalfit.stock.erebor.source_runtime import (
        SourceEMRISettings, resolve_emri_direct_cfg)
except ImportError as exc:
    if os.environ["EMRI_LIKELIHOOD"] == "full":
        print("[EMRI-PREFLIGHT] emri_pe scoring=full (the installed lisatools predates "
              "EMRI_LIKELIHOOD; full is its only path).")
        sys.exit(0)
    print("[EMRI-PREFLIGHT] REFUSING: the installed lisatools has no direct EMRI "
          f"likelihood ({exc}). EMRI_LIKELIHOOD would be SILENTLY IGNORED. Pull dev "
          "at/after the EMRI direct merge, or launch with EMRI_LIKELIHOOD=full.")
    sys.exit(2)
from lisatools.domains import WDMSettings
from lisatools.globalfit.stock.erebor.variants.all_sources import AllSourcesGeneralSettings

emri = SourceEMRISettings()
want = (os.environ["EMRI_LIKELIHOOD"], int(os.environ["EMRI_BATCH_MAX_SIZE"]),
        os.environ["EMRI_DIRECT_RESPONSE"], int(os.environ["EMRI_TRAJ_WORKERS"]))
got = (emri.likelihood, int(emri.batch_max_size), emri.direct_response,
       int(emri.traj_workers))
if got != want:
    print(f"[EMRI-PREFLIGHT] REFUSING: exported (likelihood, batch, response, "
          f"traj_workers) = {want} but the settings resolve {got}.")
    sys.exit(2)
tdi_chan = {f.name: f for f in dataclasses.fields(AllSourcesGeneralSettings)}["tdi_chan"].default
# this run's grid: Nf 1440 x Nt 17280 at dt 2.5 s
try:
    cfg = resolve_emri_direct_cfg(
        emri, domain_settings=WDMSettings.make_factory(1440, 17280), tdi_chan=tdi_chan)
except ValueError as exc:
    print(f"[EMRI-PREFLIGHT] REFUSING: {exc}")
    sys.exit(2)
if cfg["emri_likelihood"] == "full":
    print("[EMRI-PREFLIGHT] emri_pe scoring=full (per-row production path).")
    sys.exit(0)

from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

edge = inspect.signature(EMRIDirectWDM.__init__).parameters["pixel_edge"].default
crop = int(os.environ.get("EDGE_CROP_WAVELETS", "20"))
if crop < edge:
    print(f"[EMRI-PREFLIGHT] REFUSING: EDGE_CROP_WAVELETS={crop} < the {edge} edge layers "
          "the direct template leaves zero at each end of the grid.")
    sys.exit(2)
if cfg["emri_direct_response"] == "dense":
    import unittest

    try:
        from tests import test_tdi_dense
    except ImportError as exc:
        print(f"[EMRI-PREFLIGHT] REFUSING: cannot import tests/test_tdi_dense.py ({exc}) "
              "to prove the dense TDI kernel on this GPU.")
        sys.exit(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_tdi_dense.TDDenseGPUParityTest)
    res = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    if not res.wasSuccessful() or res.skipped or res.testsRun == 0:
        print("[EMRI-PREFLIGHT] REFUSING: the dense TDI kernel's GPU-vs-CPU parity test "
              f"did not pass on this node (run {res.testsRun}, failures "
              f"{len(res.failures)}, errors {len(res.errors)}, skipped {len(res.skipped)}). "
              "Rebuild lisatools (pip install -e . --no-build-isolation) or launch with "
              "EMRI_LIKELIHOOD=full / EMRI_DIRECT_RESPONSE=spline.")
        sys.exit(2)
    if os.environ["EMRI_DIRECT_LOOKUP"] not in ("kernel", "python"):
        print(f"[EMRI-PREFLIGHT] REFUSING: EMRI_DIRECT_LOOKUP={os.environ['EMRI_DIRECT_LOOKUP']!r} "
              "(kernel or python).")
        sys.exit(2)
    if os.environ["EMRI_DIRECT_LOOKUP"] == "kernel":
        # a module without the kernel would fall back to the Python lookup with only a
        # warning: prove it is here and right on this GPU instead
        try:
            from tests import test_wdm_lookup_sum_kernel as tk
        except ImportError as exc:
            print(f"[EMRI-PREFLIGHT] REFUSING: cannot import tests/test_wdm_lookup_sum_kernel.py ({exc}).")
            sys.exit(2)
        suite = unittest.TestSuite([tk.SyntheticEdgesTest("test_gpu_equals_cpu")])
        res = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
        if not res.wasSuccessful() or res.skipped or res.testsRun == 0:
            print("[EMRI-PREFLIGHT] REFUSING: the fused lookup kernel's GPU-vs-CPU parity test "
                  f"did not pass on this node (run {res.testsRun}, failures {len(res.failures)}, "
                  f"errors {len(res.errors)}, skipped {len(res.skipped)}: {res.skipped}). Rebuild "
                  "lisatools (pip install -e . --no-build-isolation) or launch with "
                  "EMRI_DIRECT_LOOKUP=python.")
            sys.exit(2)
import time

import lisatools
from lisatools import wdm_lookup_store

try:  # build on this node's GPU when there is one (has_backend("gpu") raises without)
    lisatools.get_backend("gpu")
    table_backend = "gpu"
except Exception:
    table_backend = "cpu"
table_path = wdm_lookup_store.lookup_table_path(
    cfg["emri_direct_table"], os.environ["FILE_STORE_DIR"], 1440, 2.5)
t_table = time.time()
try:
    status = wdm_lookup_store.ensure_lookup_table(
        table_path, Nf=1440, dt=2.5, force_backend=table_backend)
except (ValueError, OSError, TimeoutError) as exc:
    print(f"[EMRI-PREFLIGHT] REFUSING: lookup table {table_path}: {exc}")
    sys.exit(2)
print(f"[EMRI-PREFLIGHT] lookup table {table_path}: {status} "
      f"({time.time() - t_table:.0f} s)")
cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if cpus is not None and int(cpus) < 1 + cfg["emri_traj_workers"]:
    print(f"[EMRI-PREFLIGHT] WARNING: --cpus-per-task={cpus} < 1 + EMRI_TRAJ_WORKERS="
          f"{cfg['emri_traj_workers']}: the trajectory workers will share cores (submit "
          "through the self-dispatch, which sizes it).")
print(f"[EMRI-PREFLIGHT] emri_pe scoring=direct batch<={cfg['emri_batch_max_size']} "
      f"response={cfg['emri_direct_response']} lookup={os.environ['EMRI_DIRECT_LOOKUP']} "
      f"traj_workers={cfg['emri_traj_workers']} "
      f"table={table_path} (Nf 1440, dt 2.5) edge_crop={crop}>={edge}")
PYEOF

# ============================================================================
# SOBBH LOOKUP SCORING (2026-10-02; the DEFAULT since 2026-10-03).
# SOBBH_LIKELIHOOD=lookup scores and fills the SOBBH add/remove proposals (the
# existing SOBBHChunkedLikeMove; the comp is a drop-in) with the direct-to-WDM
# n_ref lookup template (docs/sobbh-wdm-lookup.md): one batched TDI-on-the-fly
# response per call on a SPARSE 12-h grid, the fused sobbh_lookup kernel from
# the response splines to <d|h>, <h|h>. Cluster, one H100, production grid,
# foreground on: 0.031-0.117 s per call at 8-288 rows vs 1.7-14.9 s for the
# chunked comp (36-176x); lookup vs production template <= 1.2e-6 on the
# mojito SOBHB bricks. SOBBH_LIKELIHOOD=chunked restores the v8 path.
# MULTI-GPU: one lookup comp per device behind SOBBHLookupRouter (built on
# first use under each walker shard's device), so GPUS_PER_RANK>1 works too.
# LOOKUP TABLE: SOBBH_LOOKUP_TABLE_PATH points to a specific table (any (Nf, dt)
# with this run's 3600-s layer duration); unset (default) it is the canonical
# file in this run's folder -- the SAME file EMRI_LIKELIHOOD=direct uses,
#     ${STORE_DIR}/wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5
# found there or built and saved there by the preflight below on this node's
# GPU before mpiexec (lisatools.wdm_lookup_store), so a restart never rebuilds.
# SOBBH_M_BAND_HALF_WIDTH / SOBBH_FILL_M_BAND_HALF_WIDTH are ignored by it.
# RESUME-SAFE: no stored shape changes.
# KERNEL: SOBBH_LOOKUP_KERNEL=kernel (default since 2026-10-03, SOBBH session's
# production recommendation) scores and fills through the fused C++/CUDA lookup
# (sobbh_lookup_kernel.cu) and REFUSES in the preflight below when this build
# lacks it (0.031 s vs the Python lookup's 0.050 s per 8-row call at 6 months,
# identical numbers); =auto falls back to the Python lookup with only a
# warning; =python forces the Python path.
# WATCH: "[SOBBH_LOOKUP] lookup table ... (found|built|waited|explicit)" at
# build, and the [SOBBH_LL_TIMING] leaf windows: ~30-50 ms per 8-row call at
# 6 months with the kernel; 100-400 ms means the Python fallback or a
# regression.
export SOBBH_LIKELIHOOD=${SOBBH_LIKELIHOOD:-lookup}
export SOBBH_LOOKUP_TABLE_PATH=${SOBBH_LOOKUP_TABLE_PATH:-}
export SOBBH_LOOKUP_EVAL_DT=${SOBBH_LOOKUP_EVAL_DT:-43200}
export SOBBH_LOOKUP_ROW_BATCH=${SOBBH_LOOKUP_ROW_BATCH:-32}
export SOBBH_LOOKUP_KERNEL=${SOBBH_LOOKUP_KERNEL:-kernel}
#
# SOBBH PREFLIGHT. An unknown env var is SILENTLY IGNORED, so resolve the knobs
# through the real settings class; for lookup, refuse a lisatools without the
# lookup comp, and find -- or build and save -- the
# lookup table (above). Refuses before mpiexec on any gap.
python - <<'PYEOF' || exit 2
import os
import sys

try:
    from lisatools.globalfit.stock.erebor.source_runtime import (
        SOBBH_FAST_LIKELIHOODS, SourceSOBBHSettings, resolve_sobbh_lookup_table)
except ImportError as exc:
    if os.environ["SOBBH_LIKELIHOOD"] != "lookup":
        print(f"[SOBBH-PREFLIGHT] sobbh scoring={os.environ['SOBBH_LIKELIHOOD']} (the "
              "installed lisatools predates the lookup comp; not requested).")
        sys.exit(0)
    print("[SOBBH-PREFLIGHT] REFUSING: the installed lisatools has no SOBBH lookup "
          f"likelihood ({exc}). SOBBH_LIKELIHOOD=lookup would be SILENTLY IGNORED. Pull dev "
          "at/after the SOBBH lookup merge, or launch with SOBBH_LIKELIHOOD=chunked.")
    sys.exit(2)
sobbh = SourceSOBBHSettings()
want = (os.environ["SOBBH_LIKELIHOOD"], os.environ["SOBBH_LOOKUP_TABLE_PATH"],
        float(os.environ["SOBBH_LOOKUP_EVAL_DT"]), int(os.environ["SOBBH_LOOKUP_ROW_BATCH"]),
        os.environ["SOBBH_LOOKUP_KERNEL"])
got = (sobbh.likelihood, sobbh.lookup_table_path, float(sobbh.lookup_eval_dt),
       int(sobbh.lookup_row_batch), str(getattr(sobbh, "lookup_kernel", "MISSING")))
if got != want:
    print("[SOBBH-PREFLIGHT] REFUSING: exported (likelihood, table, eval_dt, row_batch, kernel) = "
          f"{want} but the settings resolve {got}.")
    sys.exit(2)
if sobbh.likelihood != "lookup":
    print(f"[SOBBH-PREFLIGHT] sobbh scoring={sobbh.likelihood}.")
    sys.exit(0)
if "lookup" not in SOBBH_FAST_LIKELIHOODS:
    print(f"[SOBBH-PREFLIGHT] REFUSING: SOBBH_FAST_LIKELIHOODS={SOBBH_FAST_LIKELIHOODS}: the "
          "installed lisatools has no lookup comp.")
    sys.exit(2)
import time
from types import SimpleNamespace

import lisatools
from lisatools.domains import WDMLookupTable

try:  # build on this node's GPU when there is one (has_backend("gpu") raises without)
    lisatools.get_backend("gpu")
    table_backend = "gpu"
except Exception:
    table_backend = "cpu"
# this run's grid: Nf 1440 x Nt 17280 at dt 2.5 s, layer 3600 s (the resolver reads only
# these three; the file is the same canonical table EMRI direct uses)
gi = SimpleNamespace(domain_settings=SimpleNamespace(Nf=1440, data_dt=2.5, layer_dt=3600.0),
                     file_store_dir=os.environ["FILE_STORE_DIR"], force_backend=table_backend)
t_table = time.time()
try:
    path, status = resolve_sobbh_lookup_table(
        gi, {"sobbh_lookup_table_path": sobbh.lookup_table_path}, force_backend=table_backend)
    layer = float(WDMLookupTable.from_file(path, force_backend="cpu").layer_dt)
except (ValueError, OSError, TimeoutError) as exc:
    print(f"[SOBBH-PREFLIGHT] REFUSING: lookup table: {exc}")
    sys.exit(2)
if abs(layer - 3600.0) > 1e-6:
    print(f"[SOBBH-PREFLIGHT] REFUSING: lookup table {path} has layer duration {layer:g} s; "
          "this run's is 3600 s.")
    sys.exit(2)
# the fused C++/CUDA lookup (sobbh_lookup_kernel.cu) on this node's backend
have_kernel = getattr(lisatools.get_backend(table_backend), "sobbh_lookup", None) is not None
if sobbh.lookup_kernel == "kernel" and not have_kernel:
    print(f"[SOBBH-PREFLIGHT] REFUSING: SOBBH_LOOKUP_KERNEL=kernel but the {table_backend} "
          "backend module has no sobbh_lookup. Rebuild lisatools (pip install -e . "
          "--no-build-isolation) or launch with SOBBH_LOOKUP_KERNEL=auto / python.")
    sys.exit(2)
if sobbh.lookup_kernel == "auto" and not have_kernel:
    print(f"[SOBBH-PREFLIGHT] WARNING: the {table_backend} backend module has no sobbh_lookup: "
          "SOBBH_LOOKUP_KERNEL=auto falls back to the Python lookup (rebuild lisatools).")
path_used = "kernel" if (have_kernel and sobbh.lookup_kernel != "python"
                         and sobbh.lookup_interp == "spline") else "python"
if path_used == "kernel" and table_backend == "gpu":
    # prove the fused kernel's CUDA build on this node's GPU against its CPU build
    import unittest

    try:
        from tests import test_sobbh_lookup_kernel
    except ImportError as exc:
        print("[SOBBH-PREFLIGHT] REFUSING: cannot import tests/test_sobbh_lookup_kernel.py "
              f"({exc}) to prove the fused lookup kernel on this GPU.")
        sys.exit(2)
    suite = unittest.TestSuite(
        [test_sobbh_lookup_kernel.GPUKernelParityTest("test_gpu_equals_cpu")])
    res = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    if not res.wasSuccessful() or res.skipped or res.testsRun == 0:
        print("[SOBBH-PREFLIGHT] REFUSING: the fused SOBBH lookup kernel's GPU-vs-CPU parity "
              f"test did not pass on this node (run {res.testsRun}, failures "
              f"{len(res.failures)}, errors {len(res.errors)}, skipped {len(res.skipped)}). "
              "Rebuild lisatools (pip install -e . --no-build-isolation) or launch with "
              "SOBBH_LOOKUP_KERNEL=python.")
        sys.exit(2)
print(f"[SOBBH-PREFLIGHT] sobbh scoring=lookup ({path_used}) eval_dt={sobbh.lookup_eval_dt:g} "
      f"row_batch={sobbh.lookup_row_batch} table={path} ({status}, "
      f"{time.time() - t_table:.0f} s)")
PYEOF

# ---- PER-RUN EXTRA EXPORTS (runs.py: Run.extra_exports; level 2) ------------
# New knobs that exist for ONE run only land here, after every shared export
# and the waveform preflights; a knob the base already exports is changed in
# place through Run.overrides instead (build.py rewrites that export line).
# (none for this run)

# ============================================================================
# FRESH-RUN GUARD (2026-08-15). This submission starts a NEW run in a NEW
# store dir: every piece of state -- the VGB beta ladder, the GB cap-cell
# grid, the band grid, the F-stat epoch cache -- is built from the config
# above, so NO migration scripts apply (they operate on an existing h5 and
# would fail the job here). Resubmitting this same script later RESUMES this
# new store normally; only the very first submission is a fresh start.
# Refuse to start fresh on top of an existing store rather than silently
# resuming one and reporting it as a fresh run.
# ============================================================================
if [ -e "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" ]; then
  echo "[FRESH] ${STORE_DIR} already holds a run -- RESUMING it."
  echo "[FRESH] For a genuinely fresh start, point STORE_DIR at a new dir"
  echo "        or move this one aside first."
else
  echo "[FRESH] no store at ${STORE_DIR} -- starting a NEW run from scratch."
  echo "[FRESH] stages run from the top (source_search -> noise_search ->"
  echo "        noise_vgb_search ->"
  echo "        gb_search -> full_pe); the F-stat grid + epoch center table"
  echo "        are fitted fresh against this run's own residual."
fi

# ============================================================================
# OPTIONAL LAUNCH SHORTCUT: graft v3's finished noise_search (2026-08-18)
# ============================================================================
# v4 changes NOTHING on the noise side -- the whole config diff against v3 is
# GB / sig-het / F-stat knobs -- so refitting the PSD and galactic foreground
# from scratch just reproduces a result v3 already has, at ~1.5 h.
#
# But noise_vgb_search MUST re-run: the VGB ladder moved to eryn's
# make_ladder, and a resumed store's stored ladder WINS over the configured
# one. (Measured on the temper probes: arms prepped from an older base kept
# the old 1/1.2**i ladder, while freshly-built arms got make_ladder.)
#
# So: let v4 author its own store -- every grid, shape and ladder correct by
# construction -- and move only the fitted numbers in.
#
#   1. sbatch this script against the fresh STORE_DIR. Let it reach
#      noise_search and SAVE ONE iteration, then scancel. That iteration is
#      throwaway; it exists so the datasets are allocated with >= 1 row.
#   2. python scripts/fstat_proposal/graft_noise_state.py \
#          <v3_store>/gf_prod_3mo_testing.h5 \
#          ${STORE_DIR}/${BASE_FILE_NAME}_testing.h5          # dry run
#      ... then the same command with --apply.
#   3. sbatch this script again. It resumes from the grafted row, sees
#      noise_search complete, and starts noise_vgb_search on the NEW ladder.
#
# The graft tool finds v3's handover row itself (VGB is frozen for the whole
# of noise_search and starts moving on the first noise_vgb iteration), gates
# on both stores having zero GB leaves, and refuses to touch sub_backend/vgb
# -- which is where the ladder lives, and the entire point of the exercise.
# Do NOT rewind a COPY of the v3 store instead: that carries v3's grids and
# rung counts into v4 and needs a migration per array, which is how the three
# earlier band-grid migrations failed.

# LATER REFITS: GB_FSTAT_REFIT_EVERY=40 ITERATIONS (~3.5% overhead at a
# 17.7-min fit). ⚠ THE UNITS CHANGED 2026-09-18 and this note predated it:
# the clock counts distinct global-fit ITERATIONS THIS BRANCH WAS PROPOSED
# IN, not proposal hits. In a SEQUENTIAL stage (gb_search) the GB move
# fires every iteration, so 40 really is 40. In a random_choice stage one
# wrapped move is drawn per step, so the clock ticks at that move's own
# rate and 40 means ~40/(draw probability) iterations -- read the cadence
# per stage, not globally. To force an extra
# refit mid-run, stop the job and archive the epoch dir, then resubmit:
#   mv ${STORE_DIR}/gb_fstat_fit/shared/epoch_* /tmp/  &&  sbatch ...
# (_latest_epoch() then returns None and the fit-in-move rebuilds.)

# DEDICATED SAVER RANK (armed 2026-08-15, user directive -- the [SAVE]
# math flipped: the ~60 s sync write was 2% of a 55-min iteration but
# is 6-10% of the post-mega-batch 10-17 min iterations). np>=3: rank 0
# samples, the HIGHEST rank becomes the async results/saver rank, the
# middle spare is stopped at startup (run.py GlobalFit role logic).
# FIRST-LAUNCH CHECK (known caveat): run_combined_staged.py builds on
# EVERY rank before roles resolve -- watch nvidia-smi for saver/spare
# device allocations; if the extra ranks hold GPU memory, drop back to
# the plain single-process line below until the rank-gated build lands.
# ============================================================================
# LADDER PREFLIGHT. Only fires on a RESUME (a fresh submission has no store
# yet and skips it). Resume derives the GB rung count from the stored
# band_temps shape, NOT from GB_NTEMPS above -- refuse to start rather than
# run a silently-wrong ladder for days.
# ============================================================================
if [ -e "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" ]; then
  python - "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" "${GB_NTEMPS}" <<'PYEOF' || exit 2
import sys, h5py
store, want = sys.argv[1], int(sys.argv[2])
with h5py.File(store, "r") as f:
    bt = f["global_fit"]["sub_backend"]["gb"].get("band_temps")
    if bt is None:
        print("[LADDER] no gb band_temps; nothing to check.")
        raise SystemExit(0)
    have = int(bt.shape[-1])
print(f"[LADDER] stored gb rungs = {have}, GB_NTEMPS = {want}")
if have != want:
    print(f"[LADDER] REFUSING TO START: the store would run {have} rungs, not "
          f"{want}. Resume takes the STORED count. Re-rung it first:\n"
          f"  python scripts/fstat_proposal/reset_recipe_stage.py {store} "
          f"gb_search --rewind-to-empty gb --apply\n"
          f"  python scripts/fstat_proposal/rerunge_gb_ladder.py {store} gb "
          f"{want} --apply")
    raise SystemExit(2)
print("[LADDER] OK.")
PYEOF
fi

# ============================================================================
# V8 NOISE PREFLIGHT. Fail here, not 20 minutes into a slurm allocation:
#   * the modulation table must exist;
#   * the NOISE brick must exist and carry /ltts;
#   * an existing store must have been sampled under THIS noise identity
#     (the run.py resume guard is authoritative; this is the cheap copy).
# ============================================================================
python - "${GALFOR_MODULATION_PATH}" "${MOJITO_DATA_PATH}" "${NOISE_FILE:-}"   "${STORE_DIR}/${BASE_FILE_NAME}_testing.h5" "${WDM_PSD_METHOD}" <<'PYEOF' || exit 2
import glob, os, sys
import h5py
mod, mojito, noise_file, store, method = sys.argv[1:6]
if not os.path.isfile(mod):
    print(f"[V8-NOISE] REFUSING: modulation table {mod!r} not found.")
    raise SystemExit(2)
if not noise_file:
    hits = sorted(glob.glob(os.path.join(mojito, "data", "INSTRUMENT", "L1", "NOISE_*")))
    if not hits:
        print(f"[V8-NOISE] REFUSING: no NOISE_* brick under {mojito!r} and NOISE_FILE unset.")
        raise SystemExit(2)
    noise_file = hits[0]
with h5py.File(noise_file, "r") as f:
    if "ltts" not in f:
        print(f"[V8-NOISE] REFUSING: {noise_file!r} has no /ltts group.")
        raise SystemExit(2)
    n = f["ltts"]["ltt_12"].shape[0]
print(f"[V8-NOISE] delay table OK: {noise_file} (/ltts, {n} samples/link)")
if os.path.exists(store):
    with h5py.File(store, "r") as f:
        grp = f.get("global_fit", {})
        ident = grp.get("noise_model_identity") if hasattr(grp, "get") else None
        if ident is None:
            print(f"[V8-NOISE] REFUSING: {store!r} predates noise-model identity "
                  "records -- it cannot have been sampled under the v8 noise "
                  "model. Use a fresh STORE_DIR.")
            raise SystemExit(2)
        a = dict(ident.attrs)
        # The coarse mode/Q are PART of the noise identity: they change the
        # PSD/galfor transition kernel on identical array shapes, so a resume
        # across them is refused by run.py. Check them here too, or the
        # mismatch only surfaces minutes into the allocation.
        want_mode = os.environ.get("COARSE_GPU_MODE", "delayed_acceptance")
        want_q = int(os.environ.get("COARSE_Q", "8"))
        mismatches = {}
        if not bool(a.get("unequal_arm")):
            mismatches["unequal_arm"] = (a.get("unequal_arm"), True)
        if str(a.get("wdm_psd_method", "")) != method:
            mismatches["wdm_psd_method"] = (a.get("wdm_psd_method"), method)
        if str(a.get("coarse_mode", "")) != want_mode:
            mismatches["coarse_mode"] = (a.get("coarse_mode"), want_mode)
        if int(a.get("coarse_Q", 1)) != want_q:
            mismatches["coarse_Q"] = (a.get("coarse_Q"), want_q)
        if mismatches:
            print(f"[V8-NOISE] REFUSING: stored noise identity does not match this "
                  f"config (stored, wanted): {mismatches}. Full stored identity: {a}. "
                  "Use a fresh STORE_DIR.")
            raise SystemExit(2)
        print(f"[V8-NOISE] resume identity OK: {a}")
PYEOF

# ============================================================================
# SOURCES PREFLIGHT (6mo_v8). Hard-check the data before taking a slurm
# allocation (user ruling 2026-09-02: "script hard-checks brick presence
# at launch"). COMBINED-aware (2026-09-14): when SOURCE_TYPES carries
# COMBINED, the per-class L1 bricks are never read -- the checks become
# (a) the COMBINED file resolves exactly the way the loader's
# find_combined_file will resolve it (single .h5 / COMBINED_ prefix /
# MOJITO_COMBINED_FILE override), and (b) NOISE/GALFOR are not also
# listed (the loader refuses the double-count at build; fail it here in
# seconds instead). Without COMBINED, the old per-class brick check.
# ============================================================================
python - "${MOJITO_DATA_PATH}" "${MBHB_IDS:-}" "${EMRI_IDS:-}" "${SOBHB_IDS:-}" "${SOURCE_TYPES:-}" <<'PYEOF' || exit 2
import glob, os, sys
mojito, mbhb, emri, sobhb, src_types = sys.argv[1:6]
types = [s.strip().upper() for s in src_types.split(",") if s.strip()]
armed = {"MBHB": mbhb, "EMRI": emri, "SOBHB": sobhb}
bad = False
if "COMBINED" in types:
    for clash in ("NOISE", "GALFOR"):
        if clash in types:
            print(f"[SOURCES] REFUSING: SOURCE_TYPES lists COMBINED and "
                  f"{clash} -- the combined stream already contains it "
                  "(the loader refuses this at build).")
            bad = True
    folder = os.path.join(mojito, "data", "COMBINED", "L1")
    override = os.environ.get("MOJITO_COMBINED_FILE", "").strip()
    if override:
        p = override if os.path.isabs(override) else os.path.join(folder, override)
        if not os.path.exists(p):
            print(f"[SOURCES] REFUSING: MOJITO_COMBINED_FILE={override!r} "
                  f"does not exist (looked at {p}).")
            bad = True
        else:
            print(f"[SOURCES] COMBINED file (override): {p}")
    elif not os.path.isdir(folder):
        print(f"[SOURCES] REFUSING: no combined-data folder at {folder!r}.")
        bad = True
    else:
        cands = sorted(f for f in os.listdir(folder)
                       if f.endswith(".h5") and not f.startswith("."))
        named = [f for f in cands if f.upper().startswith("COMBINED_")]
        if not cands:
            print(f"[SOURCES] REFUSING: no .h5 file in {folder!r}.")
            bad = True
        elif len(cands) > 1 and len(named) != 1:
            print(f"[SOURCES] REFUSING: {len(cands)} .h5 files in "
                  f"{folder!r} ({cands}); set MOJITO_COMBINED_FILE.")
            bad = True
        else:
            pick = named[0] if len(cands) > 1 else cands[0]
            print(f"[SOURCES] COMBINED file: {os.path.join(folder, pick)}")
    for cls, ids in armed.items():
        state = f"ids {ids}" if ids.strip() else "not armed"
        print(f"[SOURCES] {cls}: catalogue-only under COMBINED ({state}).")
else:
    for cls, ids in armed.items():
        if not ids.strip():
            print(f"[SOURCES] {cls}: not armed (empty id list).")
            continue
        d = os.path.join(mojito, "data", cls, "L1")
        hits = sorted(glob.glob(os.path.join(d, "*")))
        if not hits:
            print(f"[SOURCES] REFUSING: {cls} armed (ids {ids}) but no bricks "
                  f"under {d!r}.")
            bad = True
        else:
            print(f"[SOURCES] {cls}: {len(hits)} brick file(s) under {d}; "
                  f"ids {ids} (per-id resolution is the loader's).")
if bad:
    raise SystemExit(2)
print("[SOURCES] preflight OK.")
PYEOF

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  # Multi-node launch = Intel MPI's OWN launcher (hydra) bootstrapped from the
  # SLURM allocation, NOT `srun --mpi=...` (WP7 Step 0, 2026-09-16): on this
  # cluster `srun` without PMI gives every rank a size-1 world, and
  # `srun --mpi=pmix` bootstraps but Intel MPI's OFI business-card exchange
  # then fails ("Missing hostname ... in business card"). `-ppn 1` places
  # consecutive ranks round-robin over the hosts (A,B,A,B,...) = the cyclic
  # placement the walker-block layout wants: head + saver on node A, the
  # compute ranks spread over both nodes. UCX finds no cross-node transport
  # here (only self/sysv/posix/cma), so the fabric is pinned to libfabric's
  # tcp provider -- correctness first; a faster provider is a WP7 Step 4
  # measurement, not a launch requirement.
  # SLURM_NTASKS is always set inside a job step; the :-3 default matches the
  # mpiexec branch below so an odd/manual allocation still launches the
  # head+compute+saver shape instead of dying on an unset var under `set -u`.
  export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp
  # Makes `-ppn 1` above actually bind: with the SLURM bootstrap, hydra
  # otherwise honours the scheduler's PER-NODE TASK COUNTS over -ppn. With
  # placement respect OFF the round robin is A,B,A,B,A -- two compute ranks
  # per 2-GPU node, which is what the walker-block layout wants.
  #
  # ⚠ NOT REQUIRED ON THIS SCRIPT'S NORMAL PATH, and I first recorded it as a
  # launch blocker in error (corrected 2026-09-24 by the rank-layout session,
  # with hardware corroboration). The self-dispatch block above passes
  # --distribution=cyclic whenever _NODES > 1, and cyclic already lands 5
  # tasks over 2 nodes as A:{0,2,4} B:{1,3} -- node A takes compute 0,
  # compute 2 and the SAVER, i.e. two compute ranks against a 2-GPU pool.
  # Pin or no pin, the placement is identical. The discriminator: v8's own
  # 1yr 4-GPU script never carried the pin and has launched fine.
  #
  # WHERE IT IS GENUINELY REQUIRED: any path that SKIPS the dispatch block
  # and therefore gets no --distribution=cyclic -- a manual
  # `sbatch --nodes=2 --ntasks=5 <script>` (the script's own comments flag
  # that case), and the INTERACTIVE allocation, where --ntasks-per-node=3
  # forces three tasks onto node A and -ppn 1 has to override it. That is
  # the 2026-09-18 failure the one-walker-replicas note describes.
  #
  # Kept as free insurance: identical placement under cyclic, correct
  # placement without it.
  export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
  mpiexec -n "${SLURM_NTASKS:-3}" -ppn 1 python scripts/fstat_proposal/run_combined_staged.py
else
  mpiexec -n "${SLURM_NTASKS:-3}" python scripts/fstat_proposal/run_combined_staged.py
fi
# python scripts/fstat_proposal/run_combined_staged.py   # single-process fallback
