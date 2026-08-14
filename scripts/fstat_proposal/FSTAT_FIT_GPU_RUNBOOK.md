# In-move F-stat fit — 2-GPU timing + test runbook

Times and validates the F-stat grid fit that runs **inside
`GBSpecialRJFStatGridMove.setup()`** (`GB_FSTAT_FIT_IN_MOVE=1`) on the
2-GPU node. Companion pieces: the CPU plumbing smoke
(`fstat_fit_in_move_smoke.py`, already green), the unit suite
(`tests/test_fstat_gridfit.py`), and the initial-lnL parity gate
(`scripts/gb_chunked_het/gb_multigpu_parity.py`). This runbook is the GPU
measurement the in-move-fit plan left open ("fit wall time vs iteration
time"), now on the sharded 2-GPU ACA.

**What multi-GPU means here.** The fit scores every candidate against ONE
reference walker, so `_RoutedBandEngine.route_fstat_ll` lands 100% of the
rows on that walker's shard. On the chunked path expect **no speedup from
the second GPU** — that 2-GPU leg is a correctness gate (device-local comp
buffers, no peer access, identical grids), not a performance target.
**Exception: the sig-het path** (`FSTAT_USE_SIGHET=1`) has an OPT-IN
fan-out (`FSTAT_SIGHET_MULTIDEV=1`, **default OFF** — its first on-GPU run
FAILED grid parity, 2026-08-12) that row-splits every candidate batch over
ALL run devices against per-device reference replicas
(`_RoutedBandEngine._sighet_fstat_multidevice`). Re-arming procedure:

1. compare `walker_ref` in the two legs' `epoch_0000/DONE.json` — different
   reference walkers reproduce the observed empty-band F_max scatter with
   both legs individually healthy;
2. run one 2-GPU leg with `FSTAT_SIGHET_MULTIDEV=check`: every batch is
   ALSO scored by the pinned single-device scorer and the first diverging
   row raises with per-lane forensics (which lane's row range, which
   device, prototype vs replica comp) — a mismatch confined to the
   non-primary lane convicts that device's comp replica; a pattern crossing
   lane boundaries convicts the merge/transfer machinery;
3. only after a full `check` fit passes clean, flip to
   `FSTAT_SIGHET_MULTIDEV=1`: `nvidia-smi` should show every device busy
   through `[sweep:comb.*]` / `[stageB]`, the fit wall dropping toward
   ~half — and the cross-leg grid parity gate still bit-for-bit.

## 0. Prerequisites

```sh
./install.sh --pull-only        # pure Python; no C/CUDA rebuild needed
```

The tree must contain LAT dev `5972b04` (in-move fit) and `c402d5d`
(`route_fstat_ll` shard routing) — both on `origin/dev`.

## 1. Initial-state parity gate (existing script, run first)

```sh
GPUS=0,1 OMP_NUM_THREADS=1 python \
    scripts/gb_chunked_het/gb_multigpu_parity.py --arms fstat
```

Bit-identical initial lnL 1-vs-2-GPU is what makes the grid-parity gate in
step 2 meaningful (same starting residual on both legs).

## 2. Bench, smoke scale (minutes)

```sh
GPUS=0,1 python scripts/fstat_proposal/fstat_fit_in_move_bench.py \
    --preset smoke --out /path/to/scratch/fstat_bench_smoke
```

Runs two legs (GPU 0 alone, then GPUs 0+1), each: fresh fit dir →
`erebor.gb_no_fg(lite=True)` in `GB_MODE=search` → 2 sampler iterations.
Automated gates:

- the fit runs **exactly once** per leg (iteration 2 takes the skip path;
  a second `rj_*` move reuses the registry, never refits);
- `DONE.json` + both npz caches land in `fitdir/shared/epoch_0000/`;
- finite lnL every iteration; no `NotImplementedError` / peer-access /
  cross-device marks anywhere in the log;
- **grid parity across legs**: `fstat_grid_comb.npz` and
  `fstat_grid_peaks_stacked.npz` match (bit-identical expected; rtol 1e-12
  is the formal gate).

## 3. Bench, production scale (the real timing number)

Set the band + grid knobs **exactly as the main run will** — the ambient
preset forces only `GB_MODE=search GB_FSTAT_FIT_IN_MOVE=1` and defaults
`DATA_MODE=synthetic`; every `GB_*` / `FSTAT_*` scale knob passes through.
For mojito data add `DATA_MODE=mojito MOJITO_DATA_PATH=...`.

```sh
GPUS=0 GB_CENTER_FREQ=... GB_N_LAYERS=... \
python scripts/fstat_proposal/fstat_fit_in_move_bench.py \
    --preset ambient --legs 0 --out /path/to/scratch/fstat_bench_prod
```

One leg is enough for timing (see "no speedup" above); add `--legs 0 0,1`
to re-check parity at scale. A killed run restarts with `--resume` (the
sweep checkpoints under `epoch_0000/*_parts/` resume at the saved row —
this is also the on-GPU test of mid-fit resume).

### Record (into the run notes)

| quantity | source |
|---|---|
| fit wall total, n_peaks, walker_ref | `epoch_0000/DONE.json` |
| comb / stage-B split + evals/s | bench summary table (parsed from `[sweep:comb.*]` / `[stageB]` lines) |
| fit-iteration vs steady-iteration wall | `it0_s` vs `it1_s` + the `[GF_TIMING]` per-move lines |
| GPU watermarks | `gpu_used_mb` / `gpu_pool_mb` on `[GF_TIMING]` lines |

Prior reference points: offline prep at the tested dev band was ~7 s comb
+ ~13 s stage-B vs ~130 s/iteration (plan
`floofy-wandering-melody.md`); full-band offline prep is hours-scale. The
production number decides whether `GB_FSTAT_REFIT_EVERY=10` (the lever
that would actually move recovery past the one-time-fit duplicate problem)
is affordable in-run.

## 4. Triage

| symptom | points at |
|---|---|
| `NotImplementedError` on the 2-GPU leg | `route_fstat_ll` regression (single-shard contract leaked through) |
| `peer access` / `different device` | a comp-level buffer not covered by the per-device replicas — report the class named by the router guard |
| grids differ across legs | legs saw different residuals or reference walkers — compare `walker_ref` in the two `DONE.json`s first, then rerun step 1 |
| fit runs twice in one leg | epoch/registry sharing broke (`_FSTAT_GRID_REGISTRY` / `GB_FSTAT_FIT_PER_MOVE`) |
