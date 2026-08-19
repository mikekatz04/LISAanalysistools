#!/bin/bash
# ============================================================================
# SCALING RUN -- 23-month Tobs (gb + vgb + psd + galfor, mojito)
# Identical staged recipe to submit_gf_3mo.sh; only the knobs below differ.
#
# 2026-08-15 SYNC WITH THE 3-MONTH SCRIPT. The perf/correctness/diagnostic
# work from that day was copied here; the 23-mo SIZING was deliberately NOT
# touched. Kept 23-mo-specific (do not "fix" these to match 3 mo):
#   TOBS_TARGET=6.048e7 | SIGHET_NT_LAYER=525 | GB_NLEAVES_MAX=25000
#   GB_N_SUBBANDS=2048/GPU (per-slot slab ~8x the 3-mo cost)
#   FSTAT_PEAKS_PER_BAND=500 (~8x finer comb) | FSTAT_SIGHET_MULTIDEV=1
#   STORE_DIR / BASE_FILE_NAME / job name
#
# !! PREFLIGHT GATES -- do these BEFORE submitting !!
#
# 0. RESUMING AN EXISTING gf_prod_23mo STORE? Two migrations are REQUIRED,
#    because this script now turns on the cap-cell grid and the repaired VGB
#    beta ladder, and the resume guards compare the store against the config:
#
#      H5=./gf_prod_23mo/gf_prod_23mo_testing.h5
#      python scripts/fstat_proposal/fix_vgb_band_temps.py "$H5" 8
#      python scripts/fstat_proposal/migrate_gb_cap_grid.py "$H5" --cap-divisor 8
#
#    The FIRST is not optional for correctness: the VGB likelihood was OFF
#    for every run before 2026-08-15 (betas collapsed to [1e-4]). The code
#    fix alone does NOT heal a store -- resume restores band_temps from the
#    h5, and the reconciliation deliberately lets the STORED ladder win, so
#    an un-migrated store keeps sampling VGBs with the likelihood off.
#    STARTING FRESH (new/empty STORE_DIR) needs NEITHER migration: every
#    grid and ladder is built from the config below.
#
# 1. FSTAT KERNEL SHARED-MEMORY CLAMP (blocking) -- FIX LANDED 2026-08-12,
#    GATE STILL REQUIRED. The sig-het shared-memory work on GBGPU dev
#    (checked opt-in + setup-time clamp for the fstat scorer; global-memory
#    twiddle fallback for gb_signal_het_make_reference, whose 287 KB
#    all-shared carve at THIS grid's Nt=16800 was over every device's
#    limit) must be in the wheel: REBUILD GBGPU from dev on the cluster
#    (stale .so = the old "GPUassert: invalid argument
#    gb_tdi_on_the_fly.cu:6747" crash). Then run the one-band gate ON A GPU
#    NODE in this run's env and require "GATE: PASS":
#
#        BACKEND=cuda13x python scripts/gb_chunked_het/gate_sighet_fstat_nt525.py
#
#    (exercises the 23-month grid shape at SIGHET_NT_LAYER=525: reference
#    build + fstat sweep + mode-0/1 parity + 525-vs-420 grid consistency.)
#
# 2. MEMORY SHAKEDOWN (strongly recommended). Per-walker WDM residual is
#    3 x 1440 x 16800 f64 ~ 0.58 GB -- 7.8x the 3-month grid -- plus the
#    invC store at the same scaling. Run NUM_ITERATIONS=2 first and watch the
#    "GPU pool used" lines; if OOM, reduce NWALKERS or add GPUs.
#
# 3. SKIPPED BY USER RULING (2026-08-13): the memory shakedown and the
#    FSTAT_SIGHET_MULTIDEV=check parity gate. The run launches pinned
#    (=0 below) and checkpoint-resumes through any OOM/bug: if the build
#    OOMs, reduce NWALKERS or take gpu:4, and resubmit.
#
# Expected scaling vs 3 mo: sig-het in-model scoring is FLAT in Tobs
# (v4/v5 bench-off), so per-source proposal cost holds; buffer fills,
# band likelihoods, and the fstat grid fit scale ~8x (comb nodes ~ Tobs).
# ============================================================================

# ---- fill these in ---------------------------------------------------------
#SBATCH --job-name=gf23mo         # job name
#SBATCH --partition=gpu-80-spot   # GPU partition
#SBATCH --gres=gpu:2              # 2 GPUs; consider 4 if the shakedown is tight
#SBATCH --nodes=1                 # single node
#SBATCH --ntasks=3                # main + stopped spare + SAVER rank (mpiexec -n 3)
#SBATCH --cpus-per-task=2
#SBATCH --mem=0                   # whole-node memory
#SBATCH --time=24:00:00
#SBATCH --output=gf23mo_%j.log    # combined stdout+stderr
# ----------------------------------------------------------------------------

set -euo pipefail

# ---- environment (fill in your activation) ---------------------------------
# module load FILLME_cuda_module
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

STORE_DIR=./gf_prod_23mo/

# ---- GPU telemetry ---------------------------------------------------------
# Background nvidia-smi sampler: one CSV row per GPU into the run store
# (timestamped per job, so resubmits/resumes add new files rather than
# clobbering). Killed automatically when the job exits. Columns:
# timestamp, index, name, util.gpu [%], util.mem [%], mem.used [MiB],
# mem.total [MiB], power [W], temp [C].
# INTERVAL 30 -> 5 s (2026-08-15): fine enough to attribute utilization to a
# move rather than to a whole iteration. Negligible cost.
mkdir -p ${STORE_DIR}
GPU_SAMPLE_SEC=${GPU_SAMPLE_SEC:-5}
GPU_LOG=${STORE_DIR}/gpu_util_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l ${GPU_SAMPLE_SEC} > "${GPU_LOG}" &
GPU_SMI_PID=$!
# PER-PROCESS GPU memory: --query-gpu reports DEVICE totals only, so it
# cannot say which process holds memory. Under `mpiexec -n 3` that is the
# open question (the staged runner builds on every rank before roles
# resolve). Sampled 6x slower -- it only changes on allocation.
GPU_PROC_LOG=${STORE_DIR}/gpu_procs_${SLURM_JOB_ID:-manual_$(date +%s)}.csv
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits -l $((GPU_SAMPLE_SEC * 6)) > "${GPU_PROC_LOG}" &
GPU_PROC_PID=$!
trap 'kill ${GPU_SMI_PID} ${GPU_PROC_PID} 2>/dev/null || true' EXIT

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
# The file handler is pinned at DEBUG unconditionally, so every per-iteration
# line always lands in ${STORE_DIR}/${BASE_FILE_NAME}_artifacts/globalfit_run.log
# regardless of this knob. VERBOSE only MIRRORS them to stdout; PROGRESS off
# so tqdm does not bury the output in a non-tty sbatch log. No compute cost.
export VERBOSE=1
export PROGRESS=0

export MOJITO_DATA_PATH=/shared/home/mlkatz1/mojito_cache
export USE_GPU=1
export GPU_BACKEND=cuda13x
export GPUS=0,1

export FILE_STORE_DIR=${STORE_DIR}
export BASE_FILE_NAME=gf_prod_23mo

export NWALKERS=24
export NUM_ITERATIONS=2000     # total engine iterations (resume-safe; NITER was a dead name)

# ---- Tobs: 23 months = 700 d = 6.048e7 s -----------------------------------
# Snaps exactly to Nf=1440 x Nt=16800 x dt=2.5 with layer_dt=3600 s (same
# layer_df=1.389e-4 Hz as the 3-mo grid, so band/slab geometry is unchanged).
# The 731-d mojito files cover 700 d + 2 x 8.33 d preprocessing trim.
export TOBS_TARGET=60480000

# ---- band (same as 3 mo) ---------------------------------------------------
export MIN_FREQ=4e-4
export MAX_FREQ=2.5e-2
export GB_MIN_FREQ=5.5e-4
export GB_MAX_FREQ=2.2e-2

# ---- sig-het accuracy at long baselines (the 2026-08-03/04 studies) --------
# The 2/4-yr accuracy "wall" was the COARSE SPARSE TIME GRID, not the
# engine: at N_sparse_t ~ 510 the 2-yr battery passes every tier. 525
# divides Nt=16800 (stride 32 layers = 32 h -- the same temporal density
# the 3-mo default gives). Runnable ONLY because v5 (default ON) needs
# ~31 KB scratch at this N; v4's 301 KB cannot launch. n_r=64 fit nodes +
# K=128 knots (the ruling from the overnight battery) are already the
# defaults; knots are not a lever.
export SIGHET_NT_LAYER=525
# SIG-HET NULL-COHERENCE FIX (2026-08-19, see submit_gf_3mo_v4.sh for the
# full story): the reference reconstruction's per-channel spline errors
# break the X+Y+Z null cancellation and the near-singular low-f invC
# amplifies it into the measured anchor h_h inflation. 256 nodes = the
# shared-arena ceiling; at THIS Tobs that is a COARSER effective spacing
# than the 0.35-day criterion verified at 3 months, so re-verify with
# gb_sighet_bfold_gpu_probe.py on this grid before trusting low-f h_h --
# the durable fix for long Tobs is the amp/phase redesign
# (project_signal_het_amp_phase_redesign).
export SIGHET_N_CP=256
# UNIFORM EDGE EXCLUSION (user ruling 2026-08-19): the WDM time crop is set
# by EDGE PHYSICS (taper ramps, wavelet filter width, reconstruction
# transients) -- a CONSTANT number of layers (1 layer = 1 h on this grid),
# NOT a fraction of Tobs. Be conservative: 60 layers = 2.5 days per side
# covers the old 54-layer taper region where the 3-mo dissect localized the
# sig-het edge error. Relative cost shrinks with Tobs: 5.6% @ 3 mo, 2.8% @
# 6 mo, 0.7% @ 23 mo. Every likelihood-facing WDMSettings inherits this one
# domain crop (min/max_freq still vary per source).
export EDGE_CROP_WAVELETS=60


# ---- GB knobs --------------------------------------------------------------
export GB_NLEAVES_MAX=25000        # user ruling for the 23-mo run
# Grouped RJ->in-model scheduling (2026-08-13, code default; pinned for the
# run record): RJ rounds accumulate below-cap inds=True picks, then ONE
# full-width in-model block. =0 restores the per-round interleave.
export GB_RJ_GROUPED_INMODEL=1
# Buffer sized for the grouped scheme (user budget: 10-20 GB of GPU memory
# is fine). Per-slot slab (3, 5, ~Nt_band) scales ~ Tobs: ~255 KB at 3 mo
# -> ~2.0 MB at 23 mo, so 8192 slots ~= 16 GB and in-model flushes run
# 8192 cells wide (full unit residency = 44,352 cells ~= 88 GB does NOT
# fit). Back off to 4096 (~8 GB) or 2560 (~5 GB) if the pool OOMs.
export GB_N_SUBBANDS=2048  # PER GPU; TRUE per-slot cost incl. XYZ invC (~1 MB @3mo, ~8 MB @23mo) x 2 move caches -- job-183 sizing   # PER GPU (LAT >= this commit): total = x n_gpus
# RJ pick thinning (user ruling 2026-08-14): each round proposes to a
# 0.3 random subset of eligible slots; in-model repeats still cover
# ALL alive sources (flip gate is rj-only by construction).
export GB_RJ_FLIP_FRACTION=0.2
# In-model info-matrix jump scale. COPIED FROM THE 3-MO MEASUREMENT, not
# re-derived: with the EXACT per-block SIGHET info matrices live (below),
# the 3-mo run measured cold acceptance RISING to 0.71-0.80 at 0.6 -- the
# better-adapted covariance makes the old 0.2 far too timid. The jump is
# expressed in units of the proposal covariance, so the reasoning is
# Tobs-independent; the VALUE still needs grading here against the
# [GB_ACCEPT] per-proposal-type line (target 0.15-0.4).
export GB_JUMP_FACTOR=1.2
# Comb nodes scale ~ Tobs (0.5/Tobs spacing) -> each epoch fit is ~8x the
# 3-mo cost. 100 keeps the same iteration cadence; raise to 200 if the
# [GF_TIMING] lines show refits dominating.
export GB_FSTAT_REFIT_EVERY=100
# 500, not the 200 default (user ruling): ~8x finer comb resolves more
# distinct peaks per sub-band at 23 mo; the cap must not squeeze them out.
export FSTAT_PEAKS_PER_BAND=500
# Slab 5 (user ruling; see submit_gf_3mo.sh) -- the per-cell slab is
# (3, 5, Nt) so the saving matters more at Nt=16800.
export GB_WDM_BAND_SLAB_LAYERS=5
# 3-D Milky Way (dist, alpha, sin_delta) joint prior (user ruling: the
# proper density, not the flat placeholder). Chirp-mass basis + the
# astrophysical f0-Mc GMM prior are already the code defaults. NOTE:
# this knob was NOT exercised in the smokes (they ran the uniform
# placeholder); detailed balance holds either way -- births still draw
# dist from the birth container and the prior enters through logp.
export GB_USE_GALAXY_PRIOR=1

# ---- 2026-08-15 perf batch (copied from submit_gf_3mo.sh; all are code
#      defaults pinned for the run record, and all are Tobs-INDEPENDENT --
#      they remove host/sync overhead, they do not resize anything) --------
export GB_RJ_DIRECT_BATCH=1        # rigid batches -> one end-of-unit in-model
export GB_RJ_LIVE_CAP_PICK=1       # live at-cap birth gate + same-unit re-entry
export GB_BUFFER_FIXED_CAPACITY=1  # ONE capacity buffer; resize-rebind
export GB_RJ_FSTAT_CTR_HOIST=1     # centers batched per unit (fallback path)
export GB_TEMPER_ON_REMOVAL=1      # band swaps run inside rj_prior_removal
export GB_ROUTER_DEVICE_RESIDENT=1 # shard router never host-stages; =0 reverts
export GB_RJ_SNR_TRUNC_DIST=1      # birth distance truncated at the analytic
                                   # SNR limit; truncated density in the
                                   # factors (DB-exact). At 23 mo the SNR of a
                                   # given source is ~sqrt(8)x the 3-mo value,
                                   # so FEWER births are clamped -- the lever
                                   # is smaller here, and still free.
# Per-class in-model repeats: newborns polish hard, survivors lightly.
export GB_INMODEL_REPEATS_NEWBORN=200
export GB_INMODEL_REPEATS_SURVIVOR=25
# EPOCH F-STAT CENTERS: the center distributions are built ONCE per epoch,
# inside the fit that already sweeps this grid, and looked up per row at
# propose time. This matters MORE at 23 mo than at 3 mo: the per-row F-stat
# evaluation it replaces scales with Tobs, and it rides a fit that is
# already ~8x. =unit restores the per-unit hoist.
export GB_FSTAT_CTR_MODE=epoch
# Per-block EXACT info matrices through the sig-het fast route. The
# data_index misindex is FIXED and multi-GPU slots route by the BUFFER's
# slot shards. First shakedown: SIGHET_INFOMAT_VALIDATE=1 for one propose.
export SIGHET_INFOMAT=1
export GB_INFOMAT_PER_BLOCK=1
# High-f barren-band birth shutoff (search scope).
export GB_RJ_BAND_SHUTOFF_FMIN_MHZ=10.0
export GB_RJ_BAND_SHUTOFF_AFTER=5
export GB_RJ_BAND_SHUTOFF_SCOPE=search
# Leaf caps on a band/8 cap-cell grid (confusion scale, not the computing
# scale of the sub-bands). The band grid is IDENTICAL to the 3-mo one (same
# layer_df), so this is 154 bands -> 1232 cells here too. Note the physics
# is if anything MORE favourable at 23 mo: a source's posterior is ~8x
# narrower in frequency, so a cell of the same width in Hz is ~8x wider in
# units of the posterior -- 8 stays conservative. RESUME REQUIRES the cap
# migration (gate 0 above). GB_CAP_DIVISOR=1 reverts.
export GB_CAP_DIVISOR=8
# Leaf-cap PATIENCE: consecutive iterations without a sufficient (D/2)
# lnL improvement before a cap CELL advances. Code default is now 3
# (2026-08-16, was 5): caps live on the band/8 cap-cell grid, so 1,232
# cells must each climb from 1, and at ~6 min/iteration the ramp -- not
# the wall -- is what limits how fast the model can fill. Pinned here for
# the run record; raise it if caps ever outrun the likelihood evidence.
export GB_LEAF_CAP_MIN_ITERS=3
export GB_CAP_LL_CHECK=1
# Bilinearity bookkeeping monitor (per-unit [GB_ORTHO_LL]); ~1.5 s/propose.
export GB_ORTHO_LL_CHECK=1

# ---- NOISE (psd + galfor) internal repeats: 50 -> 10 (user ruling
#      2026-08-15) ---------------------------------------------------------
# Each PSDMove.propose runs num_prop_repeats internal MCMC repeats, each
# scoring the whole (ntemps x nwalkers) ladder. The noise model is 4 (psd)
# + 5 (galfor) parameters and converges long before 50 repeats. Tobs
# affects the COST of each scoring call (the WDM grid is ~8x), not the
# number of repeats needed -- so this cut is worth MORE here than at 3 mo.
export PSD_NUM_PROP_REPEATS=10
export GALFOR_NUM_PROP_REPEATS=10

# ---- TIMERS: ALL ARMED (so the scaling readout is attributable) ------------
# Always-on and free: [GB_TIMING] spans, [FSTAT_CTR], [GB_ACCEPT] (+rj-split),
# [GB_CELL_LL], [SAVE], buffer lifecycle, the nvidia-smi CSVs above.
# GF_MOVE_TIMING: per-move wall + RSS + GPU-pool for EVERY move -- the only
# way psd_pe/galfor_pe become visible (they emit no [GB_TIMING] of their own).
export GF_MOVE_TIMING=1
# The SYNC pair makes each mark carry exactly its own kernel time instead of
# leaking into the next (cupy is async). COST ~2.5% wall. For a SCALING run
# this attribution is the entire point, so keep them; drop both if you ever
# want raw end-to-end wall numbers.
export GF_MOVE_TIMING_SYNC=1
export GB_PROP_TIMING_SYNC=1

# ---- VGB: exact chunked-het in-model scorer (see submit_gf_3mo.sh) ---------
export VGB_SIGHET_INMODEL=0
# VGB parameterization: the ESTABLISHED 5-dim distance basis, matching the
# 3-mo run (user ruling 2026-08-15). NOTE for the SCIENCE case here: at 23
# months fdot IS measurable, which is exactly what the 6-dim chirp basis
# (VGB_CHIRP_MASS_BASIS=1, Mc sampled) was built for -- in this basis
# fdot_astro_ratio stays a COLLAPSED dimension (truth 0 x multiplicative
# init = no spread, and pure stretch cannot create spread it never had).
# Left at 0 because this run is a SCALING readout, not a science posterior;
# flipping it on a FRESH store costs nothing, on an existing store it needs
# migrate_vgb_chirp_basis.py.
export VGB_CHIRP_MASS_BASIS=0
# 8-rung VGB ladder. On a RESUME this only takes effect after
# fix_vgb_band_temps.py (gate 0): the reconciliation deliberately lets the
# STORED ladder win, so an un-migrated store keeps its old (broken) one.
export VGB_NTEMPS=8
# Concurrent per-device shard dispatch (code default since 2026-08-13;
# explicit here for the run record). =0 restores serial dispatch if the
# drift/[GB_CELL_LL] checks ever implicate concurrency.
export GB_ROUTER_THREADED=1
# PINNED fstat scorer (user ruling 2026-08-13): the =check parity gate
# GATE PASSED 2026-08-14 05:58 (run_fstat_rj_search 7.0-7.8 mHz,
# 2 H100s, FSTAT_SIGHET_MULTIDEV=check): full comb+stageB fit,
# every batch bit-identical between the 2-lane fan-out and the
# pinned single-device scorer. Lanes only overlap on a GBGPU wheel
# built from >= 4381300 (GIL release); older wheels are correct but
# serialize. =0 restores the single-device pin.
export FSTAT_SIGHET_MULTIDEV=1

# DEDICATED SAVER RANK. The [SAVE] case is STRONGER here than at 3 mo: the
# store writes ~25000-leaf arrays (~1 GB/iteration raw), so the synchronous
# write is a larger slice of the iteration. np>=3: rank 0 samples, the
# HIGHEST rank saves asynchronously, the middle spare stops at startup.
# FIRST-LAUNCH CHECK: run_combined_staged.py builds on EVERY rank before
# roles resolve -- watch gpu_procs_*.csv (above) for saver/spare
# allocations. At 23-mo grid sizes an extra full build would be ~0.6 GB of
# residual per walker, so if the helpers hold memory, fall back to the
# single-process line below until the rank-gated build lands.
mpiexec -n 3 python scripts/fstat_proposal/run_combined_staged.py
# python scripts/fstat_proposal/run_combined_staged.py   # single-process fallback
