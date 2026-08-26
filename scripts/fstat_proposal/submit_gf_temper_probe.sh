#!/bin/bash
# ============================================================================
# VERTICAL-TEMPERING PROBE (2026-08-18)
#
# Derived from submit_gf_highf_probe.sh -- IDENTICAL physics, sampler shape
# and knob stack. The ONLY differences are the two new knobs below and the
# per-arm store dir. Everything else is inherited deliberately so any
# difference between arms is attributable to the change under test.
#
# WHAT IS UNDER TEST
#   GB_TEMPER_CELL_ORDER=band   scheduler orders cells (band, walker, temp)
#                               so a vertical partner pair -- (t,w,b) and
#                               (t-1,w,b) -- lands in ADJACENT buffer slots.
#   GB_TEMPER_VERTICAL=1        per-repeat vertical band-temperature swaps
#                               inside the in-model loop: same walker, so
#                               the two cells share a bit-identical data
#                               slab, the post-swap likelihoods are the
#                               pre-swap ones exchanged, and the acceptance
#                               ratio is closed form:
#                                   paccept = (b_cold - b_hot)(L_hot - L_cold)
#                               No likelihood call. No buffer touch. Pure
#                               relabel via exchange_cell_labels.
#
#   run_tempering (the permuted "fancy" swaps) is UNCHANGED and still runs
#   once per propose. Vertical swaps are ADDITIVE, and they never adapt the
#   ladder -- _adapt_band_temps stays exclusive to run_tempering.
#
# RUN THE MATRIX (separate stores, safe to run in parallel):
#   for W in dense highf; do
#     for A in baseline order vertical; do
#       WINDOW=$W ARM=$A sbatch scripts/fstat_proposal/submit_gf_temper_probe.sh
#     done
#   done
#
# WINDOW=dense is the PRIMARY test (confusion-limited: swaps trade real
# competing models). WINDOW=highf is the CONTROL (one source in the whole
# sub-band: mostly empty cells, so it stresses empty-pair handling and
# shows the machinery stays correct where there is little to trade).
# Store dirs are ./gf_temper_probe_<window>_<arm>/.
#
#   baseline  vertical OFF, order=count  -- today's behaviour, the control
#   order     vertical OFF, order=band   -- ISOLATES the scheduling change
#   vertical  vertical ON,  order=band   -- the full feature
#
#   Run `order` even though it changes no sampling: it is the only way to
#   attribute a cost/mixing change to the scheduler rather than the swaps.
#
# ============================================================================
# HOW TO READ THE OUTPUT  (grep these four tags)
# ============================================================================
# [GB_VERT ...] pair AVAILABILITY  -- THE headline for the `order` change.
#     Fraction of in-model rows that HAD a vertical partner co-resident.
#     Simulated prediction at this probe's GB_N_SUBBANDS=64 (4 bands, 24x24
#     cells): count-order ~0%, band-order ~89%. At production sizing
#     (77 bands, 8192 slots) it is 17.7% -> 95.5%.
#     PASS: band-order availability is high (>50%). If it is near zero the
#     ordering is not delivering and every other vertical number below is
#     measured on a sample too small to mean anything -- stop there.
#
# [GB_VERT ...] proposed/accepted + per-rung  -- the swaps themselves.
#     A healthy ladder trades at every rung pair. All-accept or all-reject
#     at a rung means the betas there are not separating the models.
#
# [GB_TEMPER_EMPTY ...]  -- run_tempering's own cost audit, NEW here.
#     Fraction of permuted swap pairs where BOTH cells are empty. Such a
#     pair has L2-L1 == 0, so paccept == 0, which beats log(u)
#     unconditionally: it is COUNTED as an accepted swap and moves nothing,
#     while still paying the buffer chunk build and the per-cell
#     likelihood. Measured 100% on the CPU fixture. If it is large here,
#     a large share of run_tempering's 80.3 s/propose buys no mixing at
#     all, and simply skipping empty-empty pairs is a smaller and possibly
#     bigger win than this whole scheme.
#
# [GB_CELL_LL ...]  -- the CORRECTNESS gate, and it already exists.
#     Per-cell sampled-vs-realized ll reconciliation against a
#     temperature-scaled allowance. The vertical ratio reads ll_ref (the
#     sampled cell ll), so if relabelling corrupted any ledger this is
#     where it shows. WATCH the `vertical` arm for growth relative to
#     `baseline`. (Do NOT try to audit ll_ref against band_likelihoods
#     mid-block yourself -- that measures the slab with the picked source
#     REMOVED and produces a spurious ~1e7 'error'. [GB_CELL_LL] is the
#     right instrument.)
#
# SIG-HET REFRESH INTERACTION (v4). The refresh re-bases `ll_ref` for the
# drifted subset of rows, and the vertical sweep runs AFTER it in the repeat
# body, so it always reads re-based values. A pair whose two rows were
# refreshed against different expansion points is still sound -- `ll_ref`
# estimates the same physical cell ll either way -- but it is the one place
# the two features touch, so watch [GB_CELL_LL] on the `vertical` arm.
#
# Also: [GB_TIMING] gains an `inmodel_vertical_swap` span (should be ~0 --
# it is a relabel), and `pick`/`advance` carry the scheduler cost of the
# ordering change. Simulation says the ordering costs nothing (rounds
# 43->39, slot utilisation 67.7%->74.7% at production sizing), but that
# models occupancy, not kernel cost -- confirm it here.
#
# A vertical swap that leaves the sorter inconsistent RAISES immediately
# (block-boundary barrier on special_index_check), so a silent corruption
# cannot reach the ledger.
# ============================================================================
#
# ---------------------------------------------------------------------------
# INHERITED PROBE HEADER (submit_gf_highf_probe.sh) follows unchanged.
# ---------------------------------------------------------------------------
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
# ## V3 (2026-08-16) -- A/B AGAINST THE LIVE v2 RUN. Three coupled       ##
# ## changes, run in PARALLEL with gf_prod_3mo_v2 so the difference is   ##
# ## attributable: GB_CAP_DIVISOR 8 -> 32, the ghost-increment guard ON, ##
# ## and the hierarchical uniform-cell x F**0.5 birth draw. The cap      ##
# ## changes are COUPLED -- the guard alone at K=8 re-imposes a 24.5%    ##
# ## structural exclusion, and K=32 alone leaves the ratchet to climb    ##
# ## back to irrelevance -- so they ship together or not at all.         ##
# ## FRESH STORE: this is a new run dir, so no migration is needed.      ##
# ## This starts a new store (STORE_DIR below), so every piece of state is  ##
# ## built from the config in this file: the VGB beta ladder at VGB_NTEMPS, ##
# ## the GB cap-cell grid at GB_CAP_DIVISOR, the band grid, and a fresh     ##
# ## F-stat fit + epoch center table against this run's own residual.       ##
# ## The migration scripts (fix_vgb_band_temps / migrate_gb_cap_grid /      ##
# ## migrate_vgb_chirp_basis) exist ONLY to carry an EXISTING store across  ##
# ## these changes -- do not run them here.                                 ##
# ##                                                                        ##
# ## Deploy:  git pull && sbatch scripts/fstat_proposal/submit_gf_3mo.sh    ##
# ############################################################################
# ============================================================================

# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf_temper_probe         # job name
#SBATCH --partition=gpu-80-spot   # GPU partition
#SBATCH --gres=gpu:2              # 2 GPUs (GPUS=0,1 below are LOCAL indices)
#SBATCH --nodes=1                 # single node
#SBATCH --ntasks=3                # main + stopped spare + SAVER rank (mpiexec -n 3)
#SBATCH --cpus-per-task=2
#SBATCH --mem=0                   # whole-node memory
#SBATCH --time=24:00:00
#SBATCH --output=/shared/data/global_fit_output/gf_temper_%j.log    # combined stdout+stderr ([GB_VERT]/[GB_TEMPER_EMPTY])
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
# ---- ARM SELECTION (the only thing that differs between runs) ------------
# Each arm gets its OWN store so they never resume each other and can run
# concurrently. Default `vertical` = the full feature.
ARM=${ARM:-vertical}
case "${ARM}" in
  baseline) TEMPER_VERTICAL=0; TEMPER_CELL_ORDER=count ;;
  order)    TEMPER_VERTICAL=0; TEMPER_CELL_ORDER=band  ;;
  vertical) TEMPER_VERTICAL=1; TEMPER_CELL_ORDER=band  ;;
  *) echo "ARM must be baseline|order|vertical, got '${ARM}'" >&2; exit 2 ;;
esac
# ---- WINDOW SELECTION ----------------------------------------------------
# WHICH BAND WINDOW decides whether this probe can answer the question at
# all. A vertical swap between two EMPTY cells has L2-L1 == 0, so
# paccept == 0, which passes the Metropolis test unconditionally and moves
# NOTHING. In a sparse window most cells are empty, so most swaps are
# vacuous and the acceptance statistics say little about mixing.
#
#   dense  6.04-6.74 mHz (layers 44-48)   -- confusion-limited. MANY sources
#          per cell, so swaps trade genuinely competing models. This is the
#          window that answers "does vertical tempering help?".
#   highf  20.07-20.76 mHz (layers 145-149) -- ONE catalogue source in the
#          whole sub-band (SNR 45.7 at 20.38038 mHz). Sparse by design, so
#          it is the CONTROL: it exercises empty-pair handling hardest and
#          shows whether the machinery stays correct where there is little
#          to trade.
# Both windows snap to whole WDM layers with half-layer margins (see the
# inherited notes below) and keep the target band INTERIOR so the F-stat
# fit, which uses band_edges[1:-1], does not exclude it.
WINDOW=${WINDOW:-dense}
case "${WINDOW}" in
  dense) WIN_MIN=6.041667e-03; WIN_MAX=6.736111e-03 ;;   # layers 44 -> 48
  highf) WIN_MIN=2.006958e-2;  WIN_MAX=2.076406e-2  ;;   # layers 145 -> 149
  *) echo "WINDOW must be dense|highf, got '${WINDOW}'" >&2; exit 2 ;;
esac
echo "[ARM] ${ARM}: GB_TEMPER_VERTICAL=${TEMPER_VERTICAL} GB_TEMPER_CELL_ORDER=${TEMPER_CELL_ORDER}"
echo "[WINDOW] ${WINDOW}: GB_MIN_FREQ=${WIN_MIN} GB_MAX_FREQ=${WIN_MAX}"
STORE_DIR=/shared/data/global_fit_output/gf_temper_probe_${WINDOW}_${ARM}/

# ---- STORE PREP: the four-arm matrix ------------------------------------
# Run all four; ARM x WINDOW is the whole design.
#
#   WINDOW=dense ARM=vertical   does vertical tempering help, in the
#                               confusion regime where swaps trade genuinely
#                               competing models
#   WINDOW=dense ARM=baseline   the control that makes that difference
#                               attributable
#   WINDOW=highf ARM=baseline   v4 at 24 rungs on ONE isolated source: the
#                               clean read on the hot-rung DELTA-vs-DELTA
#                               risk and per-source eps/T, uncontaminated by
#                               new tempering code. This is the arm that
#                               answers the last open question in v4.
#   WINDOW=highf ARM=vertical   vertical tempering where most cells are
#                               EMPTY -- a swap between two empty cells has
#                               L2-L1 == 0 and so "accepts" while moving
#                               nothing, which is the hardest case for the
#                               empty-pair handling to get right.
#
# A FRESH store costs ~1h33m to the first GB propose (measured), so inherit
# the noise / VGB / F-stat work from the matching existing probe store
# instead -- matching because the BAND GRID must agree (a 154-band production
# store cannot seed a 4-band window; that is the "stored with 154 sub-bands
# but the run config builds 4" error).
#
#   BASE=gf_dense_probe          # or gf_highf_probe for the highf arms
#   for ARM in vertical baseline; do
#     DST=gf_temper_probe_dense_${ARM}
#     cp -r ${BASE} ${DST}
#     python scripts/fstat_proposal/reset_recipe_stage.py \
#         ${DST}/gf_prod_3mo_testing.h5 gb_search --rewind-to-empty gb --apply
#     python scripts/fstat_proposal/rerunge_gb_ladder.py \
#         ${DST}/gf_prod_3mo_testing.h5 gb 24 --apply
#   done
#
# The rewind is REQUIRED before the re-rung (with leaves present there is no
# correct rung to assign each existing source to) and is what you want anyway
# -- the arms should start from zero GB leaves so the search is the thing
# being compared. Both arms of a window then share identical noise, VGB state
# and F-stat cache, so the vertical-vs-baseline difference is attributable
# rather than confounded by two independent noise fits.
#
# The LADDER PREFLIGHT below refuses to start if the re-rung was missed.

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
# REFIT EVERY 5, NOT 50 -- deliberately far below the v4 production cadence,
# because two code paths have NEVER EXECUTED anywhere and both are gated
# behind a refit (epoch >= 1):
#   * the GB-FREE residual window (GB_FSTAT_GB_FREE, default on) -- restores
#     the reference walker's cold GB signals before the sweep so the peak
#     list is walker-INDEPENDENT;
#   * the late peak weighting FSTAT_PEAK_WEIGHT_ALPHA_LATE (w ~ sqrt(SNR)
#     from epoch 1, against w ~ SNR at epoch 0).
# Every probe so far loaded epoch_0000 from cache and stopped well short of a
# refit, so both would otherwise first run inside the production job. At 5 the
# first refit lands within minutes and the log carries "F-stat GB-FREE
# residual: restoring N cold GB signal(s)" and a "[birth] peak-box weighting:
# ... alpha=0.25 ~ SNR**0.5, epoch=1" line to confirm each.
# THIS CADENCE IS A TEST SETTING. The fit is cheap on a 4-band window (stage B
# measured at ~7 s) but is NOT cheap at production band counts -- v4 uses 50.
export GB_FSTAT_REFIT_EVERY=5      # TEST cadence; production v4 uses 50
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
  echo "[FRESH] MEASURED cost of a fresh windowed store: ~1h33m to the first"
  echo "        GB propose (recipe setup -> first GB leaves; the noise/VGB"
  echo "        stages average ~5 min/iteration). See the STORE PREP block in"
  echo "        the header to inherit that work from an existing probe store."
fi

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
# ############################################################################
# ## HIGH-f CONFINED PROBE (2026-08-17). IDENTICAL to submit_gf_3mo_v3.sh   ##
# ## except for the block below. Purpose: watch the RJ + in-model machinery ##
# ## around ONE isolated source, with every temperature and walker live, so ##
# ## the jump/acceptance relationship is observable in minutes rather than  ##
# ## hours.                                                                 ##
# ##                                                                        ##
# ## TARGET: f0 = 20.38038 mHz, SNR 45.7 -- the highest-frequency detectable##
# ## injection, and the ONLY catalogue source in its whole sub-band (142).  ##
# ##   sub-band 142 = [20.27778, 20.41667] mHz  (1080 Fourier bins)         ##
# ##   cap cell 4567 = [20.37760, 20.38194] mHz (34 bins, K=32)             ##
# ##   cells 4566-4568 span [20.37326, 20.38628] mHz                        ##
# ##                                                                        ##
# ## BAND WINDOW: three uniform WDM layers (145,146,147) so the target band ##
# ## is INTERIOR -- the F-stat fit uses band_edges[1:-1], so the target band ##
# ## must not be an edge band or it is excluded from the fit entirely.      ##
# ##                                                                        ##
# ## NOISE IS PINNED, NOT FITTED. A three-sub-band slice contains no galaxy,##
# ## so a foreground fit there is meaningless. The psd/galfor values below  ##
# ## are v3's OWN converged numbers at iteration 82 (walker-median), so the ##
# ## GB machinery sees production-realistic noise without paying for the    ##
# ## noise stages. Everything else -- moves, caps, F-stat, tempering,       ##
# ## repeats, sig-het -- is unchanged from v3.                              ##
# ############################################################################
# FOUR layers, not three. Band shards are assigned by band COUNT, so three
# bands split 2/1 across the two GPUs -- the target band lands on one device
# and the other carries a single near-empty band, which exercises the
# multi-shard router only asymmetrically. Four splits 2/2, keeps the target
# band (146) interior for the F-stat fit (band_edges[1:-1]), and puts real
# work on both devices, so the per-device comp replicas, the sig-het
# reference stash and the cross-device reduction are all covered.
# HALF-LAYER MARGINS, not exact boundaries. _band_klohi snaps the window
# INWARD to whole WDM layers, so a value a hair below a boundary loses that
# layer: 2.013889e-2 and 2.069444e-2 sat just under layers 145 and 149 and
# snapped to 146..148, spanning 2 layers where >=3 are required (an interior
# sampled span must exist). Sitting mid-layer makes the snap unambiguous.
export GB_MIN_FREQ=${WIN_MIN}       # set by WINDOW above
export GB_MAX_FREQ=${WIN_MAX}       # -> 4 whole WDM layers either way
# NOTE: the F-stat fit range follows band_edges[1:-1], so restricting the
# band window above ALREADY confines the fit to sub-band 142 (1080 bins /
# 32 cap cells). Narrowing further to cells 4566-4568 has no env knob:
# FSTAT_PEAK_MIN_F is a minimum F-STATISTIC, not a frequency, and there is
# no MAX counterpart. Two honest options -- (a) accept the whole target
# band, which still costs ~150x less comb than production and contains
# exactly ONE catalogue source, or (b) add an f0-window knob to
# select_comb_peaks. (a) is what this script does; (b) is TODO-2 below.

# NOISE IS PINNED VIA PYTHON, NOT ENV. ``general.fixed_psd_params`` is a
# dataclass field with no env default, so run_combined_staged.py must set:
#     fit.general.fixed_psd_params = [1.522086581e-11, 2.727869920e-15]
#     fit.general.fixed_psd_kwargs = dict(
#         psd_params=[1.522086581e-11, 2.727869920e-15],
#         galfor_params=[3.767774829e-44, 5.107561738e-02,
#                        9.377231534e-01, 2.722106290e-03, 4.449185606e+03])
#     fit.remove_branch("psd"); fit.remove_branch("galfor")
# (v3's OWN converged values at iteration 82, walker-median -- not truth.)

# Trackers. GB_JUMP_TRACE logs, per propose and per temperature rung, the
# proposed |df0| in Fourier bins split by accepted/rejected -- the one thing
# no existing log line reports and the whole reason for this run.
# F-stat peak floor is now the stock default (SNR 8) -- see
# FSTAT_KNOB_DEFAULTS in sampling/fstat_proposal.py for why.
export GB_JUMP_TRACE=1
# Step-by-step MH trace of ONE source (the loudest cold row): every term
# in the ratio per repeat, plus a numeric detailed-balance check.
export GB_INMODEL_TRACE=20
# NOT GB_DEBUG=1. That knob is not instrumentation -- it fires
# apply_debug_preset(), a laptop-smoke preset, and EVERY knob the script does
# not set explicitly then falls back to a smoke default: gb.ntemps 2 (not 24),
# CHUNKED_N_SPARSE 64 (not 256), NT_SUB 64, N_PAD 8, N_CP_* 16. The first run
# of this probe silently used a 2-rung ladder and a 4x-truncated sparse
# window, which invalidates any statement about the info-matrix proposal --
# the info matrix is second differences of that same likelihood. Production
# (submit_gf_3mo_v3.sh) never sets it. The GB special-move band plots it also
# arms are not worth that price; GB_JUMP_TRACE gives what this probe needs.
export GB_NTEMPS=24                 # explicit: match production, never inherit
export GB_SIGHET_DRIFT_CHECK=1      # end-of-block drift vs the trust gate
export GB_SIGHET_ANCHOR_CHECK=1     # sig-het expansion error at the anchor

# Small + fast: one band unit, so iterations are seconds not minutes.
# ---- v4 PRODUCTION PARITY (2026-08-18) ------------------------------------
# This probe was derived from submit_gf_highf_probe.sh, which predates the
# v4 sig-het stack. Production (submit_gf_3mo_v4.sh) now arms the knobs
# below, and FOUR of them act inside the very in-model repeat loop the
# vertical swap lives in -- the reference refresh re-bases ``ll_ref``, and
# ``ll_ref`` is exactly what the closed-form swap ratio reads. Running the
# arms without these would not be apples-to-apples with production, and
# would leave the refresh/swap interaction untested.
# Values copied verbatim from submit_gf_3mo_v4.sh; change them only in
# lockstep with that script.
export GB_SIGHET_REFRESH_EVERY=25      # re-anchor drifted references
export GB_SIGHET_REFRESH_DPHASE=0      # ... on the drift test alone
export GB_SIGHET_REFRESH_MIN_BETA=0    # ... on ALL rungs, not just cold
export GB_SIGHET_TRUST_PHASE_C=49      # SNR-scaled carrier-phase gate
export SIGHET_NT_LAYER=270             # sparse-grid resolution
export FSTAT_SIGHET_MULTIDEV=1         # multi-device F-stat fan-out
# Diagnostic ladders, first iterations only (~0.32 s/propose). Kept so the
# accuracy record matches production's.
export GB_SIGHET_TIER_SCAN=0.05,0.1,0.25,0.5,1,2

# ---- THE CHANGE UNDER TEST ------------------------------------------------
# Set from ARM above; both default to today's behaviour when unset in code
# (vertical 0, order count), so `baseline` is a true control.
export GB_TEMPER_VERTICAL=${TEMPER_VERTICAL}
export GB_TEMPER_CELL_ORDER=${TEMPER_CELL_ORDER}
# Permuted-swap ledger reconciliation per unit. Cheap, and it is the
# independent check that vertical relabelling did not corrupt the
# incremental ll accounting that run_tempering also writes.
export GB_TEMPER_AUDIT=1

# Small + fast: one band unit, so iterations are seconds not minutes.
# NOTE ON SIZING: 4 bands x 24 temps x 24 walkers = 2304 cells against 64
# slots (2.8% residency). That is a HARSHER co-residency test than
# production (8192 slots), and deliberately so -- it is where the two
# orderings separate most (simulated ~0% vs ~89% partner availability).
# If [GB_VERT] shows availability near zero on the `vertical` arm, raise
# this to 1152 so a full 576-cell band column fits and re-check before
# concluding anything about the swaps themselves.
export GB_N_SUBBANDS=64
export NUM_ITERATIONS=300
# NB: this sits HERE, not up by the FRESH guard, because the script runs
# under `set -u` and GB_NTEMPS is not exported until the GB knob block
# below -- referencing it earlier aborted the job with 'GB_NTEMPS:
# unbound variable' (2026-08-18). Every export has happened by this point.
# ============================================================================
# LADDER PREFLIGHT. Resume derives the GB rung count from the STORED
# band_temps shape, NOT from GB_NTEMPS -- so a prepped store that was never
# re-runged runs a 2-rung ladder while this script says 24, and every
# tempering measurement in the run is meaningless. That failure is SILENT
# (the only hint is one build_gb_moves warning buried in a 74k-line log), and
# it is exactly how both confined probes ran a degenerate [1.0, 1e-4] ladder
# for days. Refuse to start instead.
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

# ============================================================================
# POST-RUN TRIAGE (run against the combined log)
# ============================================================================
#   grep -h "GB_VERT"          gf_temper_*.log | tail -20
#   grep -h "GB_TEMPER_EMPTY"  gf_temper_*.log | tail -20
#   grep -h "GB_CELL_LL"       gf_temper_*.log | tail -20
#   grep -hE "\[GB_TIMING .*(inmodel_vertical_swap|pick|advance|run_tempering)" \
#        gf_temper_*.log | tail -40
#
# Compare arms on: pair availability (order vs baseline), [GB_CELL_LL]
# worst-per-repeat (vertical vs baseline -- must not grow), iteration wall
# time, and cold-chain max logL trajectory.
# ============================================================================
