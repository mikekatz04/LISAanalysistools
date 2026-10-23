# Launching the global fit (processes, ranks, GPUs)

*(2026-07: parallel-resources plan P0 established the process model below.
2026-09: the multi-rank walker-block port — one MPI rank per GPU, walker
blocks sharded across compute ranks — replaced the single-process
multi-GPU model as the default. Full architecture:
[`docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md).
Cluster validation runbook: [`docs/multirank-cluster-gates.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/multirank-cluster-gates.md).)*

## The model

Each MPI rank owns a **walker block**: a fixed, equal-sized slice of the
ensemble's walkers, its own `AnalysisContainerArray` (one device, by
default), and — for every rank except the saver — a copy of the built recipe.
Fan-out lives inside the moves: a move's `propose` slices the full state into
per-rank blocks, ships each block to its rank, and merges the replies back
(`src/lisatools/globalfit/communication/`, `moves/walkerfanout.py`, the GB/VGB
three-command protocol in `moves/gbspecialstretch.py`). With exactly one
compute rank this collapses to a direct in-process call — no MPI traffic, no
pickling, bit-identical to the pre-port single-rank code
(`communication/ranks.py` module docstring; design spec Decision 9).

## Roles

- **HEAD** (rank 0 by default, i.e. `main_rank`): sequences the recipe via
  eryn's `run_mcmc`, owns the full host `GFState`, and **also computes**
  walker block 0 like any other compute rank.
- **COMPUTE**: owns a device list and one walker block; runs the
  `ComputeService` command loop (`comm.recv` is the clock; `{"op": "stop"}`
  exits). One rank per GPU by default.
- **SAVER**: the highest-numbered rank in the communicator, *only* at
  communicator size ≥ 3 (`resolve_roles`); writes HDF5 asynchronously, off
  the sampler's critical path. Below size 3 the saver role is aliased to the
  head — saves happen synchronously, exactly like a single-rank run.
- **SPARE**: exists only under `GF_LEGACY_RANK_LAYOUT=1` — builds like every
  other rank, then is sent `"stop"` at startup and exits without computing
  anything.
- **MONITOR** (`GF_MONITOR_RANK=1`, set by the 9mo / 12mo launchers): the
  highest WORLD rank is split off before the run is set up
  (`monitor/rank.py::split_monitor_rank`, called at the top of the production
  driver's `main()`, right after the memory recorder starts); it builds the monitor page and snapshot tar every
  `GF_MONITOR_ITER` saves and never joins the run communicator. The roles
  above are then resolved on the remaining ranks, so the saver is the highest
  rank of the RUN communicator. Needs at least 4 world ranks (head + compute
  + saver + monitor).

(`communication/ranks.py::resolve_roles`/`build_layout`; `run.py`'s
`role`/`RankRole` dispatch in `run_global_fit`.)

## The `np` matrix

`n_compute` is the number of COMPUTE ranks (HEAD included). Since
2026-09-23 `NWALKERS` no longer has to divide it: the layout factorizes as
`n_compute = n_blocks x R` (`n_blocks = gcd(nwalkers, n_compute)`), so
**more GPUs than walkers is legal at any walker count** and the R ranks
sharing a block disperse its work. Three settings shape the mapping:
`ranks_per_block` (R, above),
`gpus_per_rank` (a rank owns several devices, sharded in-process by the
existing multi-GPU router) and `ranks_per_gpu` (several ranks share one
device); at most one of the two may exceed 1.

| `np` | Per-node GPU pool | Resulting layout |
|---|---|---|
| 1 | any | One rank: HEAD = SAVER = the only COMPUTE rank. AUTO `gpus_per_rank` resolves to the whole pool (today's single-process multi-GPU behavior, unchanged). |
| 2 | 1 GPU | Capacity fallback: rank 1 is demoted to dedicated SAVER with a `UserWarning` (`build_layout`'s `size == 2` branch); the head computes every walker, identical to `np=1`. |
| 2 | ≥2 GPUs, or 1 GPU with `RANKS_PER_GPU=2` | Two COMPUTE ranks (head + rank 1), **no dedicated saver** — the saver role is aliased to the head, so saves are synchronous. |
| 3 | 2 GPUs | Head + 1 COMPUTE rank + 1 dedicated SAVER (rank 2), one device each. |
| 3 | 1 GPU | **`ValueError`** from `build_layout`: 2 COMPUTE ranks on a 1-device pool over-subscribes it (`capacity = len(pool) * ranks_per_gpu // gpus_per_rank = 1`). No silent fallback above size 2 — set `RANKS_PER_GPU=2` to share the device deliberately. |
| 5 | 2 nodes × 2 GPUs | 4 COMPUTE ranks (one per GPU; AUTO `gpus_per_rank` resolves to 1 per node because more than one compute rank shares each node) + 1 dedicated SAVER. Launch with `mpiexec -n 5 -ppn 1` (hydra, SLURM bootstrap) so ranks land round-robin on both nodes. |

Only `np=2` gets a silent capacity fallback. At `np ≥ 3`, asking for more
COMPUTE ranks on a node than its GPU pool can host
(`len(pool) * ranks_per_gpu // gpus_per_rank`) is a hard `ValueError` from
`build_layout` — there is no silent fallback above size 2
(`ranks.py::build_layout`, the per-node `capacity` check under
`if pool and not legacy`). `NWALKERS % n_compute != 0` is likewise a hard
error, not a rounding fallback, when driven directly through `build_layout`
(the campaign submit script rounds `NWALKERS` up instead — see below).

## Knobs

| Knob | Default | Effect |
|---|---|---|
| `GPUS` | unset (CUDA sees whatever's visible) | The **per-node** device pool (`fit.general.gpus`) that `build_layout` partitions across the node's compute ranks. Not itself a rank count. |
| `GPUS_PER_RANK` | AUTO (`None`) | AUTO: a lone compute rank on a node owns the whole per-node pool (today's `-n 1` multi-GPU behavior, `gpus_per_rank = len(pool)`); with several compute ranks on a node AUTO instead gives each one device (`gpus_per_rank = 1`). An explicit int pins it on every node: that many devices per compute rank, sharded in-process by the existing multi-GPU router (`gpu_splits`/`BandView`). Supported by the layout but **not yet validated** at `> 1` with several ranks (see `docs/multigpu-cluster-validation.md`'s addendum). |
| `RANKS_PER_GPU` | 1 | That many compute ranks share each device, each with its own CUDA context and memory pool (`device_slot` distinguishes them on that device). |
| `GF_GPU_ROUTING` | **`0` (OFF)** | ★ **The opt-in for everything in the two rows below.** OFF reproduces the pre-2026-09-23 layout rule exactly: `nwalkers` must divide the compute-rank count (`R = 1`, the walker-block layout), with the single `nwalkers == 1` carve-out that is one-walker replica mode; any other shape raises, naming this knob. ON enables the unified factorization `n_compute = n_blocks × R` at any walker count, and with it the source-weighted GB band split in replica mode. The two rules agree on **every** shape the legacy rule accepted (`tests/test_gpu_routing_optin.py` sweeps 1–32 × 1–32), so turning it on cannot change a run that was already launchable — it only adds shapes. Parsed as an opt-**in**: `1`/`true`/`yes`/`on` enable, anything else (including a typo) is OFF. `scripts/fstat_proposal/submit_gf_6mo_v8_gpurouting.sh` sets it to `1`; `GF_GPU_ROUTING=0` on that script is the rollback. Resolved once in `build_layout` and carried on `layout.gpu_routing`, so it lands in `describe()`/`digest()` and a rank that set it differently fails the agreement check instead of routing differently. |
| `RANKS_PER_BLOCK` | AUTO (`None`) | **Needs `GF_GPU_ROUTING=1`** above (an explicit value other than `1` is refused without it). **R -- compute ranks sharing ONE walker block**, the axis that makes `n_gpus > nwalkers` legal. AUTO resolves `n_blocks = gcd(nwalkers, n_compute)` and `R = n_compute / n_blocks`, i.e. it maximizes walker BLOCKS and only then replicates -- a block is the cheap axis (GB is sublinear in block width, ~`B^0.65`, and the sub-band buffer saturates above a block width of ~2.5) while a replica costs a whole ACA, inverse-PSD plane and whole-grid `BandSorter`. An explicit int pins R: it must divide the compute-rank count, and `n_compute / R` must divide `nwalkers`. `1` forces the walker-block layout, which then needs `nwalkers` divisible by `n_compute`. The rule is importable as `lisatools.globalfit.communication.factorize_layout` -- nothing should re-derive it. |
| `NWALKERS` | variant default | **No divisibility requirement under `GF_GPU_ROUTING=1`** (2026-09-23; with the knob off the old requirement stands): any pair resolves via `n_blocks = gcd(NWALKERS, n_compute)`. Block *b*'s walkers are `[b*B, (b+1)*B)` with `B = NWALKERS / n_blocks`, and the R ranks of block *b* are `compute_ranks[b*R:(b+1)*R]` -- contiguous, which is why `--distribution=block:block` co-locates them. A `gcd` of 1 with `NWALKERS > 1` is legal but replicates the whole ensemble on every rank (no walker parallelism) and `build_layout` warns. **Never change `NWALKERS` on a resume** -- the store carries it. |
| `GF_LEGACY_RANK_LAYOUT` | `0` everywhere (the layout level and, since the 2026-09-16 ruling after the WP7 transport gates passed, the campaign submit scripts at every `NGPUS`) | `1` restores today's pre-port roles: one compute rank owns the whole per-node pool, every other non-saver rank is a stopped SPARE. Rollback knob (design spec Risks). |
| `GF_LAYOUT_DRY_RUN` | unset | Preflight: every rank prints `layout.describe()` and the process exits **before** `fit.build()` allocates anything; a genuinely bad layout still raises inside `build_layout`/`prepare_rank` (Plan 5 Task 2 of this port — see `docs/multirank-cluster-gates.md`'s Step 0 for the exact invocation). |
| `GF_FANOUT_DIGEST` | unset | Emits a per-iteration `[FANOUT_DIGEST]` state-hash line (`log_like` + coords + inds), the cluster-gate tool for diffing two layouts for bit-identical transport (Plan 5 Task 4 of this port — see `docs/multirank-cluster-gates.md`'s Step 1). Emitted from the recipe's post-iteration hook regardless of the rank count, so a single-rank (`-n 1`) baseline prints it too, but only the three `n_compute=2` layouts are expected bit-identical to each other: single mode reseeds nothing (`run.py::_resolve_seed_base`/`_seed_rank_streams` return `None` when `layout.is_single()`), so the `-n 1` line is for observability and an `it=0` cross-check only, never a per-iteration diff. |
| `GF_ONE_WALKER_REPLICAS` | `1` | Escape hatch, name kept for existing runbooks: `0`/`false`/empty refuses **any** layout with `R > 1` (not just the one-walker case it originally guarded), naming `RANKS_PER_BLOCK=1` as the alternative. Any other value allows replicas. |
| `{BRANCH}_LIKELIHOOD_FANOUT` (`MBH_`, `EMRI_`, `SOBBH_`, `PSD_`, `GALFOR_`, `SGWB_`) | `1` | replica mode only: `0` = the head scores every likelihood row itself for that family (replicas still replay the residual/noise mutations and still serve GB). |
| `{P}_INNER_MOVE_KIND` (`PSD_`, `GALFOR_`, `SGWB_`) | `eigen` when the rank's walker BLOCK holds fewer than 3 walkers (one-walker replica mode, and any walker-block layout with 1-2 walkers per rank, e.g. `NGPUS=4 NWALKERS=4`), `stretch` otherwise | the PSDMove inner proposal. The stretch's complement is the rank's LOCAL block (cross-rank pooling is WP8), so `stretch` with a 1-walker block is a hard error and a 2-walker block would only move along a line; an explicit value always wins. |
| `{P}_EIGEN_REFRESH` | `10` | proposes between per-rung eigen table refreshes (eigen kind). |
| `{P}_EIGEN_EPS_REL` | `1e-4` | finite-difference step for the eigen tables, fraction of the prior box. |
| `{BRANCH}_EIGEN_INFO` (`SOBBH_`, `EMRI_`, `MBH_`: `ll` / `gram`; `PSD_`, `GALFOR_`, `SGWB_`: `ll` / `fisher`) | `ll` (code); the 9mo / 12mo launchers set `gram` for MBH / EMRI / SOBBH and `fisher` for PSD / galfor | which information matrix the eigen tables are built from: likelihood second differences (default), the Gram matrix of the move's own templates (`gram`, tuned by `{BRANCH}_EIGEN_GRAM_TARGET`, default `1e-3`), or the expected noise Fisher (`fisher`). Opt-in in the code; any failure falls back to `ll`. Rank-local like every eigen table. See [`docs/eigen-info-routes.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/eigen-info-routes.md). |

- **`FSTAT_SIGHET_MULTIDEV=1` is inert under the walker-block layout.** It
  fans the F-stat scorer out over a rank's OWN devices, and the current
  pipeline gives each rank exactly one GPU — so it silently does nothing.
  The rank split is what replaces that parallelism: stage B is divided
  across compute ranks by contiguous box range (design spec
  `docs/superpowers/specs/2026-09-16-parallel-fstat-fit-design.md`). Leave
  the variable set; it costs nothing and re-engages if anyone ever runs
  `GPUS_PER_RANK > 1`.

## Every rank builds

`prepare_rank(fit, comm)` runs **before** `fit.build()` in both drivers
(`scripts/run_global.py`, `scripts/fstat_proposal/run_combined_staged.py`):
it resolves the layout, pins `CUDA_VISIBLE_DEVICES` (or falls back to
`cudaSetDevice`) on this rank, and writes `fit.rank_layout` /
`fit.general.gpus` (narrowed to this rank's device list) before any CUDA
call. Every rank then calls `fit.build()` — HEAD, COMPUTE, SAVER, and (under
the legacy layout) SPARE all construct the full data pipeline; only the
walker-block *size* of each rank's `AnalysisContainerArray` differs (design
spec Decision 4). There is no "spare that never builds" any more except
under `GF_LEGACY_RANK_LAYOUT=1`, and even the legacy spare builds, then
releases its device memory and waits for the startup `"stop"`.

## Per-rank log files and stdout prefixes

The head keeps today's log file names: `globalfit_run.log` /
`global_fit.log`. Every other rank gets `globalfit_run.rank<k>.log` /
`global_fit.rank<k>.log` (`run.py::_rank_log_filenames`).

Stdout tag format (`communication/ranks.py::rank_tag`): `r<k>/head` (HEAD),
`r<k>/c<i>` (COMPUTE, `i` = the rank's 0-based position among compute
ranks — usually equal to its world rank), `r<k>/saver` (SAVER), `r<k>/spare`
(legacy SPARE). Only **non-head** ranks actually get their stdout lines
wrapped in `[tag] ` (`run.py`: `if self.rank != self.main_rank:
prefix_stdout(rank_tag(...))`) — the head's own console output is never
prefixed, though its tag still appears inside its own log text. A log/stdout
grep for a specific rank's output should match `^\[r<k>/`.

## Commands

```sh
# single process (laptop / driver-script development)
python scripts/run_global.py --stock <name>

# one rank per GPU on a 2-GPU node: head + 1 compute + 1 saver
GPUS=0,1 mpiexec -n 3 python scripts/run_global.py --stock <name>

# two compute ranks sharing one GPU (no second device required)
GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 python scripts/run_global.py --stock <name>

# 2 nodes x 2 GPUs -> 4 compute ranks + 1 saver (inside a 2-node allocation;
# Intel MPI's hydra launcher, round-robin over the hosts, tcp fabric -- see
# docs/multirank-cluster-gates.md "MPI launcher on this cluster")
export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp
GPUS=0,1 mpiexec -n 5 -ppn 1 python scripts/run_global.py --stock <name>

# preflight only: print the layout on every rank, build nothing
GF_LAYOUT_DRY_RUN=1 mpiexec -n 3 python scripts/run_global.py --stock <name>
```

A python driver is equivalent — rank logic lives in `GlobalFit`, not the CLI:

```python
from lisatools.globalfit.stock import erebor
from lisatools.globalfit.communication.ranks import prepare_rank
from mpi4py import MPI

fit = erebor.all_sources(nwalkers=36)
prepare_rank(fit, MPI.COMM_WORLD)   # BEFORE build(): resolves layout, pins the device
fit.build()
fit.run()          # every rank in the communicator takes its role
```

Common env knobs: `NWALKERS`, `NTEMPS`, `NUM_ITERATIONS`, `DATA_MODE`
(`mojito`/`synthetic`), `TOBS_TARGET`, `MAKE_DIAGNOSTIC_PLOTS`, `GPUS`,
`GPUS_PER_RANK`, `RANKS_PER_GPU`, `USE_GPU`, `GPU_BACKEND`, `VERBOSE`,
`GF_LEGACY_RANK_LAYOUT`.

**Console output is quiet by default.** Everything is still logged to the
run's log files (`…_artifacts/global_fit.log`, `globalfit_run.log`,
`general_setup.log`, and their per-rank counterparts above); only
warnings/errors reach the console. Set `verbose=True` at construction
(`erebor.<variant>(verbose=True)`, a headline knob — or `VERBOSE=1`) to
stream the logs to stdout and turn the progress bars back on.

**Threading policy (2026-07): MPI-only — no OMP.** `run_global.py` pins
`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` /
`VECLIB_MAXIMUM_THREADS` / `NUMEXPR_NUM_THREADS` to 1 before any import
(OMP-threaded kernels have caused OOM kills on dev machines). Parallelism
comes from MPI ranks and, when configured, GPUs. Set the env vars
explicitly to override; python drivers that bypass `run_global.py` should
pin them the same way before importing numpy/lisatools.

## Campaign submit scripts: the NGPUS dispatch

`scripts/fstat_proposal/submit_gf_6mo_v8.sh` (and its `_nogb`/`_nogb_null`
siblings) resolve the rank count from a GPU-count knob rather than a fixed
`--ntasks`. Run the script **directly** (not via a bare `sbatch`) so its
self-dispatch block can pick the partition/node/task-count and resubmit
itself:

```sh
NGPUS=2 ./submit_gf_6mo_v8.sh      # gpu-80-spot, 1 node x gpu:2 (default)
NGPUS=4 ./submit_gf_6mo_v8.sh      # gpu-80-spot, 2 nodes x gpu:2 each
sbatch  ./submit_gf_6mo_v8.sh      # legacy flow: static header defaults
                                    #   (2 GPUs, gpu-80-spot, --ntasks=3)
```

- `NGPUS=2`: `gpu-80-spot`, 1 node, `--gres=gpu:2`, `--ntasks=3`, `mpiexec
  -n 3`. `GF_LEGACY_RANK_LAYOUT` defaults to `0` (user ruling 2026-09-16,
  after WP7 Steps 0-2/4 passed on the cluster): head + 1 compute rank (one
  GPU each, `NWALKERS/2` walkers per rank) + saver. `GF_LEGACY_RANK_LAYOUT=1`
  is the rollback knob and still launches today's pre-port roles (one
  compute rank drives both GPUs, rank 1 a stopped spare, rank 2 the saver).
  `NWALKERS` defaults to `8` (env-overridable): divisible by 2 AND 4, so a run
  started at `NGPUS=2` continues at `NGPUS=4` from the same store.
- `NGPUS=4`: `gpu-80-spot`, **2 nodes** × `--gres=gpu:2`,
  `--distribution=cyclic` — the cluster's real 4-GPU shape (there is no
  single 4-GPU node). `GF_LEGACY_RANK_LAYOUT` is **forced to `0`**
  regardless of any pre-set value, because the legacy layout cannot span
  nodes.
- `N_COMPUTE = NGPUS * RANKS_PER_GPU / GPUS_PER_RANK` (`GPUS_PER_RANK` empty
  counts as 1 for this arithmetic — AUTO resolution itself happens inside
  `build_layout`); the self-resubmission passes `--ntasks=$((N_COMPUTE+1))`
  explicitly (the saver) — the static `#SBATCH --ntasks=3` header line is
  only the legacy/manual-`sbatch` fallback.
- In-job, `_NGPUS_EFF` follows `SLURM_GPUS_ON_NODE` (the granted truth, not
  the pre-submit `NGPUS` intent) and `N_COMPUTE_EFF` is recomputed the same
  way, pinned to `1` under `GF_LEGACY_RANK_LAYOUT=1`.
- `NWALKERS % N_COMPUTE_EFF`: when non-legacy and the check fails, the
  script **rounds `NWALKERS` up** to the next multiple and prints a loud
  `[SUBMIT]` line rather than failing — `build_layout` itself would raise.
  The former default `NWALKERS=10` was not divisible by `N_COMPUTE=4` at `NGPUS=4`,
  so a first 4-GPU launch rounds to 12 unless `NWALKERS` is set explicitly.
  `NWALKERS=1` is exempt from the divisibility rounding: it launches
  one-walker replica mode (every compute rank holds the walker).
- Launch line: `mpiexec -n "${SLURM_NTASKS:-3}" -ppn 1 ...` with
  `I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp` exported
  when `SLURM_NNODES > 1` (Intel MPI's hydra launcher placing ranks
  round-robin over the hosts; `srun --mpi=pmix` does not bring Intel MPI up
  on this cluster, see `docs/multirank-cluster-gates.md`), else
  `mpiexec -n ${SLURM_NTASKS:-3} ...` exactly as before.
- Under the walker-block (non-legacy) layout, `np=3` on a 2-GPU pool puts a
  real compute rank on each GPU; under the legacy layout rank 1 was a
  stopped spare that occupied a device without using it.
- In-job, a `SLURM_NNODES > 1` allocation **forces** `GF_LEGACY_RANK_LAYOUT=0`
  with a loud `[SUBMIT]` line (same rule as `NGPUS=4`, applied to the granted
  allocation rather than the pre-submit intent): the legacy layout's single
  compute rank is per-node, so a manual `sbatch --nodes=2 --ntasks=5` under it
  would leave node B's ranks idle.

## Semantics that change only when several compute ranks exist

These are accepted, intentional differences from single-rank behavior — see
the design spec's "Semantics that change only when `n_compute > 1`" and the
Plan 3/4 fan-out mixins' module docstrings for the code-level detail.

- **Ladder adaptation is pooled, once per propose.** Eryn ladders (addremove
  MBH/EMRI/SOBBH, PSD/galfor/sgwb) adapt ONCE per propose from the pooled
  swap ratio (sum accepted / sum proposed over ranks and repeats); `tc.time`
  advances once per propose instead of once per repeat. Ranks never adapt
  their own copy — only the head does, after merging
  (`moves/walkerfanout.py` module docstring; `pooled_ladder_step`). GB's
  `band_temps` cadence is unchanged (it already adapted once per propose),
  but the swap counters it adapts from are now summed across ranks by the
  head orchestrator before the one `_adapt_band_temps` call (design spec
  line 234; WP5 Architecture, fan-out 2 merge step).
- **Mid-iteration checkpoint granularity** for the addremove family drops
  from per-leaf to per-propose (the per-leaf `midit_checkpoint.maybe_write`
  is guarded off inside a rank body and moved to the head's merge step).
- **Eigen, info-matrix and F-stat reference tables are rank-local
  proposal-shaped tables** (still valid MH); the eigen sidecar is written by
  the head only. `GB_INFOMAT_PER_BLOCK=1` (`infomat_per_block`) retires the
  old cross-device info-matrix borrow, and `GB_CHOL_CACHE=1` keeps each
  rank's GB in-model factors in that rank's host memory (one cache per rank
  and branch; route map in [`docs/eigen-info-routes.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/eigen-info-routes.md)).
  GB's frequency-window friend table
  (`BandSorter.build_friend_table`) is likewise rank-local, built from the
  rank's own walker block only.
- **Stretch complements are the rank's local block.** PSD `RedBlue` splits,
  the optional VGB `stretch` kind, and `MBH_INNER_MOVE_KIND=stretch` all draw
  their complement from the local block, not the full ensemble — stretch
  proposals were explicitly ruled out of scope for this port (design spec
  Decision 7).
- **PSD search-mode termination is per block.** `PSDMove.run_move_max_likelihood`
  plateaus on the RANK's walker-block maximum, not a pooled maximum across
  ranks (pooling it would need an in-body collective; deferred).
- **The PSD ladder's swap population differs by mode.** The pooled
  adaptation consumes `PSDMove.run_move`'s explicit `temperature_swaps`
  tallies (fancy swaps included); single-rank mode instead adapts from
  eryn `RedBlueMove.propose`'s identity `temper_comps` swaps. Both are valid
  acceptance ratios; the swap population feeding the ladder differs between
  the two modes.
- **`move.temperature_control.swaps_accepted`** — what eryn stores to the
  HDF backend — is the HEAD block's count only; the pooled/summed counters
  that actually drive adaptation live in the sub-state. Diagnostic-only
  field; nothing reads the HDF-stored copy back.
- **No cross-rank tempering swaps in the base port.** GB/VGB in-band cell
  tempering (`run_tempering`) stays strictly within a rank — the swap unit
  is device-resident slab state and cross-rank exchange was assessed as very
  hard (design spec, "Cross-rank swaps: assessment"); `[GB_TEMPER_CHECK ...]`
  is logged per rank, each rank checking only its own census/device accept
  counters. Cross-rank fancy swaps for the *non*-GB, non-VGB moves (MBH,
  EMRI, SOBBH, PSD/galfor/sgwb) are WP8's scope (`cross_rank_swaps`, default
  off) — GB and VGB keep a `# TODO(multi-rank)` marker either way.
- **No end-of-run submission dump under several compute ranks.**
  `GlobalFit._write_submission` returns early with a `logger.warning` when
  `not layout.is_single()`: the head's ACA holds only its own walker block,
  so the residual dump would be wrong over the full ensemble. An explicit
  `TODO(multi-rank)` marks the gap in the warning text itself; routing the
  submission writer through the fan-out is unscoped follow-up work.
- **A neutral GB block's `log_like` is not rebuilt from the residual.** When
  the head marks a walker block *neutral* (an alive-only move — one carrying
  `use_prior_removal` or `rj_replace` — over a block with no alive source),
  that rank runs nothing and replies with zeros, and the head SKIPS the merge
  for it: those walkers keep the `log_like` they came in with rather than
  having it re-derived. Same as the legacy behaviour for a zero-source
  ensemble, and every other branch's move still rebuilds the same walkers'
  likelihood, so nothing goes stale in a real recipe — but the zeros in a
  neutral reply are "nothing happened", never state
  (`GBSpecialBase._gb_neutral_finish_reply`).
- **Per-rank accept-split log lines.** `[GB_ACCEPT rj-split ...]` (and the
  analogous replace-split line) is now printed once per compute rank, into
  that rank's own log file — a monitor/digest tool that only reads the
  head's `globalfit_run.log` sees just the head block's births, not the
  whole ensemble's.

## Debug-plot instrumentation (per-move residual tracing)

The GB special-stretch move and the source moves
(``ResidualAddOneRemoveOneMove`` — MBH phentax / EMRI / SOBBH) can dump
per-step figures tracing the "remove source from the residual → sample →
put it back" choreography. Output is a GB-style flip-book: one figure per
moment (source in fit → isolated → refit), each rows = TDI channels (X/Y/Z),
columns = [total template | total data | residual]. The template/data columns
are fixed references and the residual column changes across frames, so
flipping ``_f0`` / ``_f1`` / ``_f2`` animates the source leaving and
re-entering the residual.

Two ways to turn it on (precedence: move-spec > stage-spec > env):

**Env, per branch** — the source moves self-activate from
``{BRANCH}_DEBUG`` (capitalised branch name):

```sh
EMRI_DEBUG=1  SOBBH_DEBUG=1  MBH_DEBUG=1  \
EMRI_DEBUG_DIR=./emri_dbg  MBH_DEBUG_PLOT_WALKER=2  MBH_DEBUG_EVERY=10 \
python scripts/run_global.py --stock all_sources
```
Companion knobs (each prefixed by the branch): ``{B}_DEBUG_DIR``,
``{B}_DEBUG_PLOT_WALKER``, ``{B}_DEBUG_PLOT_LEAF``, ``{B}_DEBUG_EVERY``. GB
uses the analogous ``GB_DEBUG`` / ``GB_DEBUG_DIR`` /
``GB_DEBUG_PLOT_WALKER`` / ``GB_DEBUG_PLOT_BAND``.

**Move / stage level, in code** — via the recipe API (works for GB and the
source moves uniformly, and is picklable):

```python
fit = erebor.all_sources()
fit.set_move_debug("emri_pe", plot_dir="./emri_dbg", every=5)  # one move
fit.set_stage_debug("full_pe", plot_walker=2)                  # whole stage
fit.set_move_debug("psd_pe", False)                            # force off
```
These set ``Move.debug`` / ``Stage.debug``, applied at
materialization through ``GlobalFitMove.set_debug(...)`` (options:
``plot_dir``, ``plot_walker``, ``plot_leaf`` / ``plot_band``, ``every``).

## Notes

- `head_rank` is a retired legacy alias (old multi-stage pipeline); it
  defaults to the main rank and has no role. The saver rank is assigned
  automatically — there is no knob for it.
- The saver currently writes gzip-9 HDF5 and does not yet plot; moving the
  diagnostic plot set onto it (with saves-take-priority backpressure) is
  plan phase P2.
- `GlobalFit.resolve_rank_roles` is kept as a legacy-compat wrapper for old
  callers of the `(main_rank, results_rank, spare_ranks)` shape; its
  `spare_ranks` return is always `[]`, including under the legacy layout —
  use `layout.role_of(rank)` / `layout.compute_ranks` for anything new.
