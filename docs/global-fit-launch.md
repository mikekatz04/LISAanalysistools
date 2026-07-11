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
construct the heavy `CurrentInfoGlobalFit`.

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

Common env knobs: `NWALKERS`, `NTEMPS`, `GF_NUM_ITER`, `DATA_PROCESSOR`
(`mojito`/`synthetic`), `TOBS_TARGET`, `MAKE_PLOTS`, `GPUS`, `USE_GPU`,
`GPU_BACKEND`.

## Notes

- `head_rank` is a retired legacy alias (old multi-stage pipeline); it
  defaults to the main rank and has no role. The saver rank is assigned
  automatically — there is no knob for it.
- The saver currently writes gzip-9 HDF5 and does not yet plot; moving the
  diagnostic plot set onto it (with saves-take-priority backpressure) is
  plan phase P2.
