#!/bin/bash
# ============================================================================
# PRODUCTION global fit -- 3-month Tobs (gb + vgb + psd + galfor, mojito)
# Staged recipe: noise_search -> noise_vgb_search -> gb_search -> full_pe
# via scripts/fstat_proposal/run_combined_staged.py  (LAT dev >= 53dc401)
#
# RESUME: re-submitting this script resumes automatically -- the h5 backend
# restores the last saved iteration and completed stage statuses. Keep the
# same FILE_STORE_DIR/BASE_FILE_NAME and env between submissions. For a
# fresh start, move/delete ${STORE_DIR} first.
#
# ############################################################################
# ## V4 (2026-08-18) -- THE IN-MODEL CORRECTNESS RUN.                       ##
# ##                                                                        ##
# ## v3 could not refine f0: the GB ensemble's between-walker scatter was   ##
# ## 0.798 Fourier bins against 0.051 within-walker. Root-caused to the     ##
# ## in-model proposal covariance, broken two independent ways, both        ##
# ## silent (LAT 09687e4b):                                                 ##
# ##   * the fdot CONDITIONING scale 1e-16 was resolved by matching         ##
# ##     ("fdot","Mc"), so on the distance/chirp-mass basis it landed on    ##
# ##     the Mc column (natural scale O(0.1-1)). That drove the Mc          ##
# ##     eigenvalue under the eigen-floor and the Mc step came out at       ##
# ##     1.27e-15 x the true posterior width. Mc never moved.               ##
# ##   * the physical->sampling Jacobian was DIAGONAL on a map that is not  ##
# ##     separable, so Mc lost its amplitude term and fdot_astro_ratio got  ##
# ##     EXACTLY zero curvature (which is why its proposal row had to be    ##
# ##     zeroed). Now the exact congruence J^T Gamma J.                     ##
# ## Verified against the real GB likelihood: all 9 columns 1.000 (was      ##
# ## 8.5e-40 for Mc, 0 for the ratio). CONFIRMED IN PRODUCTION: in-model    ##
# ## infomat cold acceptance 0.31-0.38 at GB_JUMP_FACTOR=1.2, against 0.95  ##
# ## under the broken covariance. Do NOT retune the jump factor.            ##
# ##                                                                        ##
# ## Also in v4:                                                            ##
# ##   * LAT 71c0bbd1 -- the multi-shard router returned None, so on >= 2   ##
# ##     GPUs sighet_active was ALWAYS False and the anchor check, the ll   ##
# ##     audit, the reference refresh and the trust gate were all silently  ##
# ##     dead. Every diagnostic below exists because of this fix.           ##
# ##   * GB group stretch OFF (c3725f7d): measured cold 2/472 = 0.0042 on   ##
# ##     v3 while the VGB stretch scored 0.4485 on the same run.            ##
# ##   * SIGHET_NT_LAYER stays at the v3 default (36 h). 270 was tried and  ##
# ##     REVERTED: it OOMs (the sig-het stash goes as cells x N_sparse_t)   ##
# ##     and the accuracy gain did not reproduce once the config echo made  ##
# ##     the measurement attributable. See the block at SIGHET_NT_LAYER.    ##
# ##   * F-stat peak weighting flattens to w ~ sqrt(SNR) after the first    ##
# ##     refit (62cd814e), plus a fix for alpha being silently dropped on   ##
# ##     the stage-B reload path.                                          ##
# ##                                                                        ##
# ## START: this expects a COPY of the v3 store rewound to zero GB leaves,  ##
# ## so the fitted noise, the VGB ladder and the F-stat epoch cache are     ##
# ## inherited and only the GB search re-runs:                              ##
# ##   cp -r gf_prod_3mo_v3 gf_prod_3mo_v4                                  ##
# ##   python scripts/fstat_proposal/reset_recipe_stage.py \                ##
# ##       gf_prod_3mo_v4/gf_prod_3mo_testing.h5 gb_search \                ##
# ##       --rewind-to-empty gb --apply                                     ##
# ## For a genuinely fresh start instead, just point STORE_DIR at an empty  ##
# ## dir -- every piece of state then rebuilds from this file.              ##
# ##                                                                        ##
# ## Deploy:  git pull && sbatch scripts/fstat_proposal/submit_gf_3mo_v4.sh ##
# ############################################################################
# ============================================================================

# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf3mo_v4          # job name
#SBATCH --partition=gpu-80-spot   # GPU partition
#SBATCH --gres=gpu:2              # 2 GPUs (GPUS=0,1 below are LOCAL indices)
#SBATCH --nodes=1                 # single node
#SBATCH --ntasks=3                # main + stopped spare + SAVER rank (mpiexec -n 3)
#SBATCH --cpus-per-task=2
#SBATCH --mem=0                   # whole-node memory
#SBATCH --time=24:00:00
#SBATCH --output=/shared/data/global_fit_output/gf3mo_v4_%j.log     # combined stdout+stderr (captures [MAXLOGL]/[BENCH])
# ----------------------------------------------------------------------------

set -euo pipefail

# ---- environment (fill in your activation) ---------------------------------
# module load FILLME_cuda_module
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

# FRESH RUN (2026-08-15, user ruling: "we want to totally restart. This is a
# fresh run now."). A NEW store dir so the previous run's h5/logs/fstat cache
# stay intact for comparison and nothing can silently resume. BASE_FILE_NAME
# stays gf_prod_3mo so every analysis tool (monitor generator, digests) works
# unchanged -- they take the DIRECTORY as their argument.
STORE_DIR=/shared/data/global_fit_output/gf_prod_3mo_v4/

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
trap 'kill ${GPU_SMI_PID} ${GPU_PROC_PID} 2>/dev/null || true' EXIT

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
export USE_GPU=1
export GPU_BACKEND=cuda13x
export GPUS=0,1

# ---- output ----------------------------------------------------------------
export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_3mo

# ---- sampler shape ---------------------------------------------------------
export NWALKERS=24                 # 24 walkers / 24 GB temps (user ruling)
export NUM_ITERATIONS=2000         # total engine iterations (resume-safe; NITER was a dead name)

# ---- band + domain ---------------------------------------------------------
# EXPLICIT Tobs (2026-08-13): sbatch propagates the submitting shell's env,
# and a stale TOBS_TARGET export (a 3-day one was found live in the shell)
# would silently re-grid this run. Pin the 90-d production value.
export TOBS_TARGET=7776000
export MIN_FREQ=4e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

# ---- GB knobs (everything else rides the flipped defaults: sig-het in-model,
#      fstat-fit-in-move + sig-het fstat, D/2 leaf-cap gate w/ min-iters 5,
#      at-cap RJ skip, cell-lifecycle ll credit, GB_MODE=search +
#      GB_PE_MOVES_STRICT=1 + GB_SEARCH_PRIOR_REMOVAL=1 seeded by the script) --
export GB_NLEAVES_MAX=10000
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
export GB_N_SUBBANDS=8192  # PER GPU; TRUE per-slot cost incl. XYZ invC (~1 MB @3mo, ~8 MB @23mo) x 2 move caches -- job-183 sizing   # PER GPU (LAT >= this commit): total = x n_gpus
# RJ pick thinning (user ruling 2026-08-14): each round proposes to a
# 0.3 random subset of eligible slots; in-model repeats still cover
# ALL alive sources (flip gate is rj-only by construction).
export GB_RJ_FLIP_FRACTION=0.2
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
# >>> RECOMMENDED: keep BOTH for the first ~2-3 stored iterations to get
# >>> the detailed attribution, then comment them out and resubmit so the
# >>> steady-state numbers are honest baselines. The run resumes cleanly.
export GF_MOVE_TIMING_SYNC=1
export GB_PROP_TIMING_SYNC=1
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
# Per-class in-model repeats (search mode): newborns polish hard,
# survivors lightly. PE resolves to GB stock num_repeat_proposals.
export GB_INMODEL_REPEATS_NEWBORN=200
export GB_INMODEL_REPEATS_SURVIVOR=25
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
export GB_SIGHET_REFRESH_EVERY=25
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
export GB_SIGHET_ANCHOR_CHECK=1
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
# crop changes Nt_active (2121 -> 2001) and resume/rewind compatibility
# across a domain-shape change is UNVERIFIED.
# export EDGE_CROP_WAVELETS=60

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
# Countable-only F-stat center precompute + lookup-miss fallback rides
# the pull (no knob beyond the existing GB_RJ_FSTAT_CTR_HOIST=1); the
# new [FSTAT_CTR] census line diagnoses the job-195 5x centers blowup.
# Bilinearity bookkeeping monitor (code default ON, user ruling: ~1.5 s
# per propose = negligible): per-unit [GB_ORTHO_LL] line compares the
# realized cold parent-residual delta against the summed per-buffer
# deltas; WARNs above GB_ORTHO_LL_TOL (0.05). The [GB_ORTHO] premise
# check (GB_ORTHO_CHECK) stays OFF until the sub-band shakedown.
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
export GB_CAP_DIVISOR=32
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
export GB_LEAF_CAP_MIN_ITERS=3
export GB_CAP_LL_CHECK=1
# Grouped RJ scheduling: accumulate inds=True picks across RJ rounds
# (1 proposal per cell per round), then ONE full-width in-model block.
# Code default since 2026-08-13; pinned for the run record. =0 restores
# the per-round RJ->in-model interleave.
export GB_RJ_GROUPED_INMODEL=1
# ---- 2026-08-14 rj stack (ALL code defaults; pinned for the run record;
#      each =0 reverts that piece independently) --------------------------
export GB_RJ_DIRECT_BATCH=1        # rigid batches -> one end-of-unit in-model
                                   # phase; =0 restores the staged scheduler
export GB_RJ_LIVE_CAP_PICK=1       # live at-cap birth gate + same-unit
                                   # re-entry (freed cells birth again)
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
export GB_TEMPER_ON_REMOVAL=1      # band swaps run inside rj_prior_removal
# High-f barren-band birth shutoff (search scope): bands above FMIN with
# AFTER consecutive zero-birth-accept proposes stop proposing births
# (deaths + in-model continue; [GB_BAND_SHUTOFF] log line per band).
export GB_RJ_BAND_SHUTOFF_FMIN_MHZ=10.0
# PATIENCE RAISED 5 -> 50 (user ruling 2026-08-16, after a replay against
# truth). Running the EXACT rule over iterations 5-60 of this run at
# AFTER=5 switches off 74 bands above 10 mHz -- and 9 of them contain a
# detectable catalogue source the run subsequently FOUND: SNR 45.7 (band
# 142, 20.278-20.417 mHz, silenced at iteration 18), SNR 35.6 (band 90,
# iter 14), SNR 34.3 (band 72, iter 19), SNR 32.7, 26.9, 24.0, 19.8, 15.0,
# 12.5. Every one of those 9 bands is occupied today. Observed
# time-to-first-source in them is 14-21 iterations, so a 5-iteration
# occupancy clock silences bands the sampler merely has not reached yet --
# and shutoff is PERMANENT for the process, so there is no recovery.
export GB_RJ_BAND_SHUTOFF_AFTER=50
export GB_RJ_BAND_SHUTOFF_SCOPE=search
# 100 -> 50 (2026-08-18): the refit re-derives the peaks against the LIVE
# residual and the UPDATED foreground/PSD, which is the whole point of
# refitting -- and the foreground converges well inside 20 iterations, so a
# 100-propose cadence spends most of the run on a grid fitted to a
# foreground that no longer exists. ~6% overhead at a 17.7-min fit.
# The peak weighting also flattens to w ~ sqrt(SNR) from epoch 1 onward
# (FSTAT_PEAK_WEIGHT_ALPHA_LATE, default 0.25), so this cadence is also
# when that takes effect.
export GB_FSTAT_REFIT_EVERY=50     # production cadence (5 was verify-only)
export FSTAT_PEAKS_PER_BAND=200    # per-sub-band peak cap (code default; explicit)
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
# HIERARCHICAL BIRTH DRAW (v3): pick a CAP CELL uniformly, then draw within
# it with w ~ F**alpha. Unset, this tracks GB_CAP_DIVISOR; pinned here so
# the draw grid is explicit and can be decoupled from the cap grid later.
# Set to 1 to fall back to the historical global w ~ F**alpha mixture.
# Checked on the epoch-0 peak set: at K=32, 95% of peaks still sit in a
# multi-peak cell (median 3, max 21), so the F**0.5 preference keeps real
# work to do INSIDE each cell -- this equalises across the band without
# degenerating into flat weighting.
# Implemented as flat composite weights, w_j = (1/N_occupied) *
# F_j**alpha / sum_cell(F**alpha), so StackedFStatProposal4D's rvs AND
# logpdf stay mutually exact by construction (both read self.weights) --
# the RJ acceptance ratio is never at the mercy of two implementations.
export FSTAT_PEAK_WEIGHT_CELLS=32
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

# ---- VGB: exact chunked-het in-model scorer (sig-het accuracy at the
#      loudest-VGB SNRs is unverified -- [GB_CELL_LL] growth in smoke 1) ----
export VGB_SIGHET_INMODEL=0
# VGB RELAUNCH BLOCK (2026-08-15, user rulings). The VGB likelihood was
# OFF all run (betas=[1e-4] bug) and 36/55 leaves were frozen by the GB
# SNR gate -- both fixed in code (76cd3237); pre-fix VGB samples are
# prior-only. Migration 1 in the header checklist is REQUIRED for the
# VGB_NTEMPS=8 ladder below.
# PARAMETERIZATION: the ESTABLISHED 5-dim DISTANCE basis
# [dist, phi0, cos_iota, psi, fdot_astro_ratio] -- what this store has
# been sampling all along. (User ruling 2026-08-15, superseding the
# earlier chirp-basis arming: "revert to the old VGB parameterization
# ... the old regular parameterization we had before" for these runs.)
# Keeping it means NO chirp migration and NO ndim 5->6 change on a live
# store -- one less thing moving while we validate the likelihood fix.
# The 6-dim chirp basis (Mc sampled, un-collapses fdot_astro_ratio)
# stays built and tested for the 6-month run: set 1 there and run
# migrate_vgb_chirp_basis.py first.
# NOTE: fdot_astro_ratio stays a COLLAPSED dimension in this basis
# (truth exactly 0 x multiplicative init = zero spread, and the
# affine-invariant stretch cannot create spread it never had). That is
# the known, accepted cost of staying on the old parameterization.
export VGB_CHIRP_MASS_BASIS=0
# 8-rung ladder (user ruling 2026-08-15). Resume derives the rung count
# from the STORED band_temps shape, so the migration above MUST be run
# with the matching "8" argument (it recreates every rung-dimensioned
# vgb dataset: temps ladder, counters zeroed, 7 swap pairs).
export VGB_NTEMPS=8
# GB rung count. 24 is already the code default (stock/erebor/gb.py
# env_default("GB_NTEMPS", 24)) -- pinned here anyway because the rung count
# is the one knob whose failure mode is completely silent: resume derives it
# from the STORED band_temps shape, so a store built at the wrong count runs
# the wrong ladder forever while the script still says 24, and the only hint
# is a single build_gb_moves warning buried in a 200k-line log. Both confined
# probes ran a degenerate [1.0, 1e-4] ladder for days on exactly that. The
# LADDER PREFLIGHT below turns the silent case into a refusal to start.
export GB_NTEMPS=24
# Concurrent per-device shard dispatch (code default since 2026-08-13;
# explicit here for the run record). =0 restores serial dispatch if the
# drift/[GB_CELL_LL] checks ever implicate concurrency.
export GB_ROUTER_THREADED=1

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
  echo "[FRESH] stages run from the top (noise_search -> noise_vgb_search ->"
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

# LATER REFITS: GB_FSTAT_REFIT_EVERY=100 proposal-hits (~8 h at the new
# iteration cadence, ~3.5% overhead at a 17.7-min fit). To force an extra
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

mpiexec -n 3 python scripts/fstat_proposal/run_combined_staged.py
# python scripts/fstat_proposal/run_combined_staged.py   # single-process fallback
