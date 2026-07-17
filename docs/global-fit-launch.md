# Launching the global fit (processes, ranks, GPUs)

*(2026-07: parallel-resources plan P0. See the approved plan for the full
architecture assessment and later phases.)*

The global fit separates its two parallel axes:

- **GPUs are intra-node and knob-controlled** — one python process (the main
  rank) drives all local devices through the `AnalysisContainerArray` split
  machinery. `GPUS=0,1,...` (or `fit.general.gpus=[...]`) selects them;
  `USE_GPU=0` forces CPU. GPU count never changes the MPI layout.
- **MPI ranks are process roles** — resolved by
  `GlobalFit.resolve_rank_roles(comm, main_rank)`:

| np | layout |
|---|---|
| 1 | main rank does everything; HDF5 saves are synchronous. |
| 2 | rank 1 is a spare: it is sent `"stop"` at startup and exits (no build). Saves stay synchronous on main. |
| ≥3 | the highest rank becomes the **dedicated saver rank** (asynchronous HDF5 writes off the sampler's critical path); remaining spares are stopped at startup. |

Spare ranks never pay the data build: `scripts/run_global.py` resolves the
roles *before* building, so only the main rank (and the saver at np≥3)
construct the heavy `GlobalFitSetup` (formerly `CurrentInfoGlobalFit`).

## Commands

```sh
# single process (laptop / driver-script development)
python scripts/run_global.py --stock <name>

# canonical cluster entry: main + dedicated saver
mpiexec -n 3 python scripts/run_global.py --stock <name>

# Slurm
srun -n 3 --gpus-per-node=<G> python scripts/run_global.py --stock <name>

# multi-GPU on one node (GPU count is independent of -n)
GPUS=0,1 mpiexec -n 3 python scripts/run_global.py --stock all_sources
```

A python driver is equivalent — rank logic lives in `GlobalFit`, not the CLI:

```python
from lisatools.globalfit.stock import erebor
fit = erebor.all_sources(nwalkers=36)
fit.build()
fit.run()          # np=1; run under mpiexec for the saver-rank layout
```

Common env knobs: `NWALKERS`, `NTEMPS`, `NUM_ITERATIONS`, `DATA_MODE`
(`mojito`/`synthetic`), `TOBS_TARGET`, `MAKE_DIAGNOSTIC_PLOTS`, `GPUS`, `USE_GPU`,
`GPU_BACKEND`.

**Threading policy (2026-07): MPI-only — no OMP.** `run_global.py` pins
`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` /
`VECLIB_MAXIMUM_THREADS` / `NUMEXPR_NUM_THREADS` to 1 before any import
(OMP-threaded kernels have caused OOM kills on dev machines). Parallelism
comes from MPI ranks and, when configured, GPUs. Set the env vars
explicitly to override; python drivers that bypass `run_global.py` should
pin them the same way before importing numpy/lisatools.

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
