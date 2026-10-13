# The flagship launchers

One launcher structure, six observation times. The 9mo v9 run (job 751,
`scripts/fstat_proposal/submit_gf_9mo_v9.sh`) is the base; every run follows
its pattern and differs only where the observation time or the parent run
demands (user ruling 2026-10-11).

```
scripts/flagship/
  launcher.sh.in      the shared body (LEVEL 1: every run)
  runs.py             the run table   (LEVEL 2: one run)
  build.py            table + template -> submit_gf_<run>.sh; --check; --set (LEVEL 3)
  submit_gf_3mo.sh    generated, committed, launched as-is
  submit_gf_6mo.sh
  submit_gf_9mo.sh    == the launched 9mo v9 in every line bash executes
  submit_gf_12mo.sh
  submit_gf_18mo.sh
  submit_gf_24mo.sh
  oneoff/             LEVEL 3 outputs (git-ignored)
tests/test_flagship_launchers.py   the guards (see below)
```

## The chain

| run | days | Nt | sig-het stride | edge crop | GB leaves / sub-bands per GPU / RJ chunk | MBHB_IDS (t_c < days + 14 d) | parent |
|---|---|---|---|---|---|---|---|
| 3mo | 90 | 2160 | 60 | 20 | 10000 / 32768 / 65536 | 18 | none |
| 6mo | 180 | 4320 | 120 | 60 | 15000 / 8192 / 32768 | 2,5,16,18 | flagship 3mo |
| 9mo | 270 | 6480 | 180 | 60 | 20000 / 4096 / 16384 | 2,5,7,12,16,18 | the LIVE 6mo v9 (`gf_prod_6mo_v9_4gpu`) |
| 12mo | 360 | 8640 | 240 | 60 | 20000 / 4096 / 16384 | 0,2,3,4,5,7,9,10,12,15,16,18 | 9mo (`gf_prod_9mo_v9`) |
| 18mo | 540 | 12960 | 360 | 90 | 25000 / 2048 / 8192 | + 6, 13, 19 | flagship 12mo |
| 24mo | 720 | 17280 | 480 | 120 | 25000 / 2048 / 8192 | all 20 | flagship 18mo |

`python scripts/flagship/build.py --list` prints the resolved table. Every
rule (grid, stride, crop guard, the two-week MBHB rule, the capacity values'
provenance) is written next to the field it feeds in `runs.py`.

A parented run reads its parent's store for all three warm starts: the GB
warm-start mixture (`GB_WARM_START_COMPONENTS`, auto-built from the parent's
last 300 cold-chain rows on the first launch), the psd + galfor start pin
(`GF_SEED_STORE`, maxlogL cold walker) and the source warm start
(`SOURCE_WARM_START_STORE`, SNR > 10 leaves by catalogue id; the id lists are
the PARENT's `MBHB_IDS`), and the monitor page overlays the parent's final
overlap / noise / foreground figures (`GF_MONITOR_ADD_INFO`). The parent store
must be complete when the child launches.

The 3mo has no parent (the `{{#else}}` branches of `launcher.sh.in`): no GB
warm start, no `gb_search_seed`, no seed store; psd starts at the unequal-arm
reference fit, galfor at the INFLATED 90-d fitted tanh (amp x1.25, fk x1.10,
f_1 off the prior rail); `source_search` runs first, then the FORCED
`noise_search`, then `gb_search_1` with the noise searched once to convergence
at the end of every cycle (as `gb_search_2`), then `full_pe`.

Store names: `gf_prod_<run>_flagship/` (job `gf<run>_flagship`), except the
9mo, which keeps the live run's `gf_prod_9mo_v9/`. The monitor's run identity
(`lisatools/globalfit/monitor/_identity.py`, `_SERIES_TAGS`) knows the tag;
renaming `SERIES` in `runs.py` means adding the new tag there too.

## The three adjustment levels

1. **Every run** -- edit `launcher.sh.in`, then `python scripts/flagship/build.py`.
   The template is bash with `{{TOKENS}}` on the per-run lines and
   `{{#if HAS_PARENT}} ... {{#else}} ... {{#endif}}` around the parent /
   no-parent blocks (the full token list: `build.py::tokens`).
2. **One run** -- edit its `Run` in `runs.py`: a table field, an
   `overrides={"KNOB": "value"}` entry (rewrites that export line in place,
   keeping its `${K:-v}` form), or an `extra_exports=("export NEW=1",)` line
   (lands at the `RUN_EXTRA_EXPORTS` hook after every shared export and the
   waveform preflights). Then rebuild.
3. **One run, once** -- `KNOB=value ./scripts/flagship/submit_gf_<run>.sh`
   for any `${K:-v}` knob (as always), or for a hard-pinned export
   `python scripts/flagship/build.py <run> --set KNOB=value --out scripts/flagship/oneoff/<name>.sh`
   and launch that file. One-offs are marked in their header and never
   committed.

`build.py --check` exits 1 when a committed launcher differs from what the
table + template build (the test suite runs it too).

## Launching

From the LAT root, on the cluster: `./scripts/flagship/submit_gf_<run>.sh`
(a fresh store on the first submit, a resume after; `NGPUS=2` for the 1-node
spot flow; the dispatch table supports 2 or 4 GPUs -- a larger count is a
level-1 edit of the `case "${NGPUS}"` table).

### A child whose parent is still running (the 12mo off the live 9mo)

Three launch-line knobs decide what the child takes from the parent, all
`${K:-v}`, so they go on the line (or into the child's `overrides` in
`runs.py` once decided):

```
GF_SEED_STORE=/shared/data/global_fit_output/gf_prod_9mo_v9/gf_prod_9mo_testing_running_backup_copy.h5 \
GB_WARM_START_LAST_K=<rows> \
./scripts/flagship/submit_gf_12mo.sh
```

* `GF_SEED_STORE` -- while the parent job is WRITING its primary store, point
  the child at the parent's `*_running_backup_copy.h5` (refreshed every
  `BACKUP_ITER` saves): the GB warm-start fit, the psd pin and the source
  warm start all read this one path (`GB_WARM_START_SOURCE_STORE` and
  `SOURCE_WARM_START_STORE` default to it). The 9mo was launched off the live
  6mo this way. Keep the SAME value on every resubmit: the warm-start npz
  records the store it was fitted from.
* `GB_WARM_START_LAST_K` -- the number of the parent's last STORED ROWS the
  GB warm-start mixture is fitted from (x the parent's walkers). Rows, not
  iterations: with `GB_SEARCH_LEGS=1` a search stage stores THREE rows per
  cycle (after `in_model`, `in_model_fstat`, `in_model_removal`); `full_pe`
  stores one per iteration. 300 (the 9mo's value off the 6mo full_pe) means
  100 search cycles.
* The source warm start copies the parent's SNR > 10 leaves at its LAST
  stored row (`SOURCE_WARM_START_SNR_MIN`); the psd pin reads the parent's
  maxlogL cold walker. Galfor stays the hand-set offline estimate (ruling
  2026-09-25: "the galfor pin should come from our estimate"); the parent's
  foreground is NOT taken unless the template's pin block changes.

Before the first launch, on a login node: dry-run the source warm start once
per branch (`python -m lisatools.globalfit.warmstart.sources --store <seed h5>
--branch mbh --ids <the parent's MBHB_IDS>`, then emri / sobbh) -- a NaN SNR
record refuses the launch -- and pre-build the GB warm-start npz if the in-job
build would be slow (the 6mo -> 9mo build took ~45 min in-job; the
`[WARMSTART]` block prints the three commands with this run's values; the
npz must land at `${STORE_DIR}/warmstart/<parent dirname>_refereed.npz`).

## The guards (`tests/test_flagship_launchers.py`)

* DRIFT -- each committed launcher equals `build.py`'s output.
* IDENTITY -- the flagship 9mo == `scripts/fstat_proposal/submit_gf_9mo_v9.sh`
  in every line bash executes (exports, assignments, control flow, heredoc
  preflights, SBATCH directives, the dispatch argv), except three stale
  operator-hint lines listed in the test.
* STRUCTURE -- tokens appear only on the declared per-run lines of the
  template; each run's resolved exports differ from the 9mo's only in the
  declared Tobs / parent / no-parent sets.
* VALUES -- grid, stride, crop guard, MBHB rule, parent chain, the 3mo
  overlay, NGPUS default 4, the V9-30 production defaults everywhere.
* COMPOSITION -- `run_combined_staged.py --print-recipe` under each
  launcher's exports gives `gb_search_seed -> source_search -> gb_search_1 ->
  gb_search_2 -> full_pe` (parented) or `source_search -> noise_search ->
  gb_search_1 -> gb_search_2 -> full_pe` (3mo).

## History

The per-knob comment blocks in `launcher.sh.in` are the 9mo v9's (the 6mo v9
production launcher's, with the 9mo / 1yr deltas). The stacked derivation
headers (9MO-1..9, V9-1..30, V8, V4-V7) were not carried: read them in
`scripts/fstat_proposal/submit_gf_9mo_v9.sh` and `git log`. The old
`derive_*_launcher.py` twins and their tests stay as the record of the
launched 6mo / 1yr / 9mo files.
