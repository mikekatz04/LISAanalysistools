# Eigen / information-matrix routes

Every proposal in the global fit that is shaped by a local curvature, which
information matrix feeds it, how to switch, and what is production. Status as
of **2026-10-06**.

Three proposal families need an information matrix:

- **GB in-model block proposal** (`gbspecialstretch.py`): per source, the
  Cholesky factor of the inverse information matrix (and, with
  `GB_INMODEL_OBSERVABLE_EIGEN`, the observable-basis `Gamma_z`), frozen across
  each in-model block.
- **Add/remove source branches** (SOBBH / EMRI / MBH,
  `addremovemove.py::ResidualAddOneRemoveOneMove`): one eigen table
  `(axes, sigmas)` per leaf for the eryn `EigenAxisMove` inner move, rebuilt
  every `{BRANCH}_EIGEN_REFRESH` leaf visits.
- **Noise branches** (psd / galfor / sgwb, `psdmove.py::PSDMove`): per-branch
  eigen tables for the same inner move, rebuilt every `{P}_EIGEN_REFRESH`
  proposes.

In all three the matrix only shapes a proposal that is **frozen and symmetric
between refreshes**, so every route is valid Metropolis-Hastings; the routes
differ in cost and in how well the proposal is scaled. The source / noise
opt-ins and the GB cache fall back to the older route on any error, with a
logged warning. `SIGHET_INFOMAT_ENGINE=lookup` by itself has no fallback:
without an attached lookup table GBGPU raises, so it always goes with
`SIGHET_REF_BUILD=lookup` (and, in production, the cache).

## At a glance

| Family | Default (code) | Other route | How to select | Fallback | Status 2026-10-06 |
|---|---|---|---|---|---|
| GB in-model factor | the chunked delegate's matrix; with `SIGHET_INFOMAT=1` + `GB_INFOMAT_PER_BLOCK=1` (set in the v9 launchers) sig-het second differences of lnL per block | **lookup Gram** `<dh_a\|dh_b>` (GBGPU) + **host factor cache** | `SIGHET_INFOMAT_ENGINE=lookup` (needs `SIGHET_REF_BUILD=lookup`), `GB_CHOL_CACHE=1` | cache error: cache disabled AND `SIGHET_INFOMAT_ENGINE` unset, direct per-block factors resume | **Production**: on in the 6mo / 1yr v9 launchers; healthy on the cluster since 6mo job 738 |
| SOBBH / EMRI / MBH eigen tables | `"ll"`: likelihood second differences | `"gram"`: Gram matrix of the move's own templates | `{BRANCH}_EIGEN_INFO=gram` | Gram error: the ll route | **Launcher default** (`gram`) in the 6mo / 1yr / 9mo v9 launchers since 2026-10-07 (V9-30, Mike's production line; `ll` on the line restores the second differences); CPU-validated, first GPU run = the first launch on those defaults |
| SOBBH / EMRI / MBH step SCALE (after either route above) | the information-matrix sigma | **adapted from the chain**: per-axis robust width of the leaf's rung ensemble, floor = the matrix sigma, axes kept (2026-10-14) | `{BRANCH}_EIGEN_AXIS_ADAPT=1` | adaptation error: the information-matrix table | Code default OFF; CPU-tested only, no run yet |
| SOBBH / EMRI / MBH adaptation buffer (walker-block fan-out) | the rank's own walkers | **pooled**: the union of every walker's rung ensemble, shipped by the head per propose (2026-10-15) | `{BRANCH}_EIGEN_ADAPT_POOL=1` (with `{BRANCH}_EIGEN_AXIS_ADAPT=1`) | no shipped sample yet: the own buffer | Code default OFF; CPU-tested only, no run yet |
| SOBBH / EMRI / MBH step per RUNG | the cold sigma at every rung | **tempered**: rung `k` steps `sigma / sqrt(beta_k)`, capped (2026-10-15) | `{BRANCH}_EIGEN_TEMPER_SIGMAS=1` | no ladder: the cold sigma | Code default OFF; CPU-tested only, no run yet |
| PSD / galfor (/ sgwb) eigen tables | `"ll"`: likelihood second differences | `"fisher"`: expected Fisher of the noise covariance | `{PSD,GALFOR,SGWB}_EIGEN_INFO=fisher` | Fisher error: the ll route | **Launcher default** (`fisher` for PSD and GALFOR) in the 6mo / 1yr / 9mo v9 launchers since 2026-10-07 (V9-30); CPU-validated, first GPU run = the first launch on those defaults |

## GB in-model factors: the lookup Gram and `GB_CHOL_CACHE`

The factor for source `x` is `chol(Gamma_y^-1)`, with
`Gamma_y = J^T Gamma_x J` the physical information matrix mapped to the
sampling basis by the transform Jacobian
(`GBSpecialBase._compute_proposal_cholesky`). `Gamma_x` comes from the comp's
`information_matrix`, routed per shard (`gbbands._RoutedBandEngine`). The
routes, in the order GBGPU's `GBSignalHetComputations.information_matrix`
takes them:

1. **`SIGHET_INFOMAT_ENGINE=lookup`** (GBGPU f8073ce, env read on every call):
   the Gram (expected Fisher) `<dh_a|dh_b>` from 2 lookup-table fills per
   parameter and one contraction against the walker's inverse-covariance rows
   (`GBLookupComputations.information_matrix`). f0 / fdot steps are
   `1e-4/Tobs` and `1e-4/Tobs^2`: the chunked defaults move the phase by only
   ~1e-7 / 1e-8 rad, where interpolation noise dominates (fdot-fdot came out
   6x the exact-template Gram at 30 d). Needs the table attached by
   `SIGHET_REF_BUILD=lookup` (else it raises). It needs no in-model block and
   no buffer slot, which is what lets the cache below build every alive
   source in one batch. The switch is process-wide: in the v9 launchers it
   also serves the VGB per-block factors (VGB builds its sig-het engine from
   the same `SIGHET_*` knobs).
2. **`SIGHET_INFOMAT=1`** inside an in-model block with buffer slots: second
   differences of the sig-het in-model scorer (`information_matrix_from_ll`),
   ~2.4 ms/source.
3. Otherwise the **chunked delegate**'s matrix (~46 ms/source).

`GB_INFOMAT_PER_BLOCK=1` (VGB: always) computes exact factors per block for
the block's own sources; with it off, a cold-chain table built once per
proposal is borrowed nearest-in-frequency.

**`GB_CHOL_CACHE=1`** (`src/lisatools/globalfit/moves/gb_chol_cache.py`; GB
only, off by default) keeps the factors and `Gamma_z` in HOST memory, one
cache per rank and branch shared by every GB move, each entry mapped to its
living source:

- **Map: per-source IDs (2026-10-07).** Every living GB source carries a
  persistent int64 ID (`GBState.source_uid`, one per leaf slot beside
  `d_h`/`h_h`; `BandSorter.source_uid` per sorter row), and an entry is keyed
  by **(walker, ID)** -- within the source's own walker, never its rung. No
  f0 window, no tolerance, no end-of-block retrack. The ID follows its source
  through the sorter build, RJ birth (new ID) / death (retired), replace (new
  ID), in-model moves, vertical and tempering swaps (a swap relabels the row
  that carries the ID), the write-back repack, GFState copies, the
  multi-rank walker slices and block merges (`block_source_uid` in the
  `gb_finish` reply), the one-walker replica merge, and the cold->hot reseed
  (reseeded sources get new IDs). Alive leaves without an ID (the first
  propose after a load or relaunch) get one at the sorter build, once,
  before any lookup. IDs are per-process (random per-process prefix over a
  counter, unique across ranks and restarts in practice) and are NOT in the
  HDF5 store; a mid-iteration checkpoint pickle keeps them, harmlessly. A
  source moved to another walker (a tempering walker permutation) is a new
  key there, computed once.
- **Fallback: coordinate matching**, for a row without an ID (and for every
  row with the rollback switch `GB_CHOL_CACHE_UID=0`, which makes the GB moves
  build their sorters without the ID column -- bitwise the pre-ID cache). A
  source matches within its own walker against the best of the
  `GB_CHOL_CACHE_WINDOW` (32) f0-neighbours on each side, accepted when
  sampling columns 0-2 (amplitude, f0, Mc in the production basis) are all
  within `GB_CHOL_CACHE_TOL` (5) of the entry's own marginal widths; a hit
  moves the entry's key with the source, and coordinate-keyed entries are
  re-keyed to their source's final coordinates at the end of each block. Its
  two failure modes (two sources of one walker within 5 sigma trading
  factors; a source moved between blocks by another move missing) cannot
  happen to a row with an ID.
- **Refresh.** Every `GB_CHOL_CACHE_EVERY` (40) proposes on a global ticker
  (the largest `move.time`), ALL alive sources are rebuilt together, in
  `GB_CHOL_CACHE_BATCH` (4096) chunks, at the first in-model block. Births and
  unmatched sources are computed per block in one batch before the repeats.
- **Fallback.** Any exception disables the cache for the rest of the run,
  unsets `SIGHET_INFOMAT_ENGINE` (process-wide: every move and cache on the
  rank leaves the lookup Gram), and logs `[GB_CHOL_CACHE] DISABLED`; the move
  continues on the direct per-block factors. The ID bookkeeping (sorter
  build, RJ / replace accepts, write-back, fan-out) runs outside that net, in
  the GB moves themselves; `GB_CHOL_CACHE_UID=0` on the launch line switches
  it off entirely.

Cluster record: job 735 (GBGPU f8073ce alone) disabled itself on the first
refresh (`np.asarray` of a cupy step array; fixed in GBGPU 376c94f); job 738
(LAT a20072e7 + GBGPU 376c94f): misses ~1.5-2 %, `inmodel_cholesky` 53 -> ~5 s
per propose, iteration 6.6 -> 5.5 min. Larger Gram steps lowered in-model
acceptance (0.67 -> 0.49 in job 736) while moving sources further per
accepted step.

Tests: `tests/test_gb_chol_cache.py` (LAT, CPU); GBGPU
`tests/test_lookup_information_matrix.py` (lookup Gram vs the exact-template
Gram: elements 7.6e-3, marginal widths within 30 %; CPU);
`tests/test_sighet_engine_parity.py` (the `SIGHET_REF_BUILD` /
`SIGHET_ANCHOR_ENGINE` wiring); the per-source IDs:
`tests/test_gb_source_uid.py` (continuity through every path above, CPU
`BandSorter`) and `tests/test_gb_orchestrator_merge.py` (the head's block
merge). Open (1yr TODO list): a GPU unit test of the lookup Gram; the ID
path has not run on a GPU yet (1yr TODO #9 implemented 2026-10-07, CPU-tested
only).

## Add/remove source branches: `{BRANCH}_EIGEN_INFO=gram`

`ResidualAddOneRemoveOneMove._build_eigen_table` picks, in order:

1. an `eigen_table_builder` installed from the branch Settings'
   `info_matrix_gen` (custom; overrides everything);
2. **`"gram"`** when `{BRANCH}_EIGEN_INFO=gram` (move attribute `eigen_info`)
   and the move implements `_gram_templates`;
3. **`"ll"`** (default): second differences of the move's own `compute_like`
   at `{BRANCH}_EIGEN_EPS_REL` (1e-4) of the prior box, shaped by
   `{BRANCH}_EIGEN_SCOPE` (`walker_max`: one table at the max-lnL cold walker;
   `per_walker`: one per (temperature, walker) point). A failed ll build gives
   identity axes with 1 %-of-box steps.

**The Gram route** (`_build_eigen_table_gram`, `_gram_info`) builds
`<dh_a|dh_b>` at the max-lnL cold walker from central differences, in the
sampling basis, of the move's OWN batched templates, through that walker's
container (`_slice_to_template` + `inner_product`: the scorers' PSD and
normalization). One batch of `2 * ndim + 1` templates per pass. Steps start at
`{BRANCH}_EIGEN_GRAM_EPS_REL` (default: `{BRANCH}_EIGEN_EPS_REL`) of the box
and are rescaled once so each column moves the template by
`{BRANCH}_EIGEN_GRAM_TARGET` (1e-3) of `||h||` (factor clipped to
[1e-3, 1e3], step to [1e-12, 1e-2] of the box; `0` keeps the fixed steps).
No fixed relative step works for a chirping source: on the SOBBH toy, f_low
goes nonlinear above ~1e-6 of its box while the masses reach the template's
float noise floor below ~1e-7. The Gram route always builds ONE shared table
(`{BRANCH}_EIGEN_SCOPE` does not apply) and returns the same `(axes, sigmas)`
form as the ll route's `walker_max`.

Hooks (`_gram_templates(phys_rows, walker) -> (arr, box)`, plus an optional
`_gram_context(walker)` device context):

| Branch | Templates | Notes |
|---|---|---|
| SOBBH (`SOBBHChunkedLikeMove`) | `comp.fill_global_wdm`, one zeroed slab per row: lookup fills under the stock `SOBBH_LIKELIHOOD=lookup`, chunked fills under `chunked` | single-shard ACA only; holds 23 data-shaped slabs (~0.6 GB peak at 6 months, ~1.2 GB at 1 year) |
| EMRI (`EMRIDirectLikeMove`) | the direct-to-WDM adapter's templates on the active box, `batch_max_size` chunks | a refused row raises into the fallback |
| MBH (`MBHBatchedLikeMove`) | the batched windowed templates in the leaf's window, on the containers' box | no window / a refused batch raises into the fallback |

**Why** (SOBBH chunked toy, 2026-10-06): at its production step (1e-4 of the
box) the ll route gave an **f_low curvature ~250x too small** (that step is
deep in f_low's nonlinear range; the resulting f_low step ~16x too wide),
**spin entries dominated by lnL noise**, and **masses ~1.5x off** the Gram.
(The ba428237 commit message's "25-250x off in masses / spins / f_low"
overstates the masses.)

Accuracy (CPU): SOBBH chunked vs an independent dense stock-template Gram at
the same steps, diagonal within 1-3 %, marginal widths within 3.3 %; SOBBH
lookup comp, max correlation-normalized difference 1.5e-2; EMRI exact (1e-8)
vs the toy generator; MBH windowed within 2e-2. Step-target convergence:
3e-3 vs 1e-3 differs by 5e-3, 1e-4 vs 1e-3 by 4e-4.

Tests: `tests/test_sobbh_chunked_move.py::SOBBHGramInfoTest`,
`tests/test_sobbh_lookup_move.py::SOBBHLookupGramTest`,
`tests/test_source_gram_info.py` (EMRI + MBH: per-walker PSD with a negative
control, exactly flat columns, routing, fallback). The ll route:
`tests/test_eigen_refresh.py`, `tests/test_eigen_table_persist.py`.

The eigen-table sidecar does not record which route built a table: after a
restart the persisted table serves until the leaf's next refresh, whichever
route is selected (valid for the same frozen-proposal reason as above).

## Add/remove source branches: the adapted step scale (`{BRANCH}_EIGEN_AXIS_ADAPT`)

Added 2026-10-14 (user ruling). It sits after whichever route above built the
table:

```
route (custom / gram / ll)  ->  (axes, sigma_fisher)  ->  [EIGEN_AXIS_ADAPT]  ->  (axes, sigma_adapted)  ->  EigenAxisMove
                                                              ^ ring buffer of the leaf's end-of-visit rung ensemble
```

**Why.** On the 9mo run (job 763, rows 453-561, T_max 6 EMRI ladder) every
EMRI sampled a long, degenerate intrinsic ridge (ln M, mu, a, p0, e0). The
table is the curvature at ONE point, and its widest intrinsic axis was the
ridge direction, so the directions were right. The scale was not: the
eigen move steps one table sigma per draw, while the chain's own cold spread
along the ridge was 1.4-56x that sigma (EMRI 0 5.4x, 1 3.8x, 2 2.0x, 3 3.2x,
4 13x, 5 1.4x, 6 5.5x, 7 56x). A random walk whose step is r times smaller
than the ridge needs about r^2 iterations to cross it. The four cold walkers sat
at different places on one connected ridge, while within each walker the
autocorrelation time was 2-6 iterations.

**What it does** (`addremovemove.py::_eigen_adapt_table`, helpers in
`eigen_refresh.py`), at every table refresh of a leaf:

1. **Samples**: a per-leaf ring buffer (`_note_eigen_adapt_sample`, called at
   the end of every leaf visit in `propose_local`) of `work.coords[:ntemps,
   :, leaf]`, meaning every rung of the walker(s) THIS rank's body sampled (one
   walker per rank in production), with the visit's betas. The buffer keeps the last
   `{BRANCH}_EIGEN_ADAPT_VISITS` (50) visits (400 samples at 8 rungs). It is
   not persisted.
2. **Cold units** (`cold_unit_deviations`): each sample's deviation from
   its rung's OWN robust centre, with the periodic columns wrapped to `[-P/2, P/2)`
   (the move's `PeriodicContainer`) before anything else, x `sqrt(beta)`.
   The rung's own centre rather than the cold rung's: a hot rung sitting
   elsewhere (another stretch of the ridge, another island, a cold->hot
   rotation copy, a re-anchored rung) would otherwise add its centre offset
   to every deviation and size the cold step to the distance between modes.
3. **Width** (`empirical_axis_widths`): the deviations in the table's AXIS
   BASIS, `c = A^-1 dev` (the axes are unit vectors but NOT orthogonal in
   the sampling basis; an axis step changes only its own `c_k`), re-centred
   per rung on the per-axis median in that basis (a column-space centre error
   set by a wide axis would otherwise leak onto a narrow one), pooled,
   `1.4826 x MAD`. The MAD resists transient outliers.
4. **Adapted sigma** (`adapt_axis_sigmas`): `max(sigma_fisher, min(width,
   cap))`. The cap is the one the table is built with (`axis_caps` =
   `sigma_max_frac x axis_prior_bounds`, shared with `_tables_from_info_batch`),
   so a railed axis stays at the cap and the information-matrix sigma is the
   floor. **The axes are not changed.** A `per_walker` stash is adapted
   table by table on its own axes.
5. Until the buffer holds `{BRANCH}_EIGEN_ADAPT_MIN_VISITS` (10) visits, the
   table is the information-matrix one.
6. **Safety cap** (`{BRANCH}_EIGEN_ADAPT_MAX_FACTOR`, default 100, added
   2026-10-15): the adapted sigma never exceeds 100x the information-matrix
   sigma, so step 4 becomes `max(sigma_fisher, min(width, cap, max_factor x
   sigma_fisher))` (`addremovemove.factor_capped_sigmas`, applied to
   `adapt_axis_sigmas`' output). The floor and the prior-box cap are
   unchanged, and the factor cap sits under the floor, so it can never push a
   sigma below the information-matrix one. See "The safety cap" below.

The scale changes only at a refresh, at a leaf-visit boundary, so the
proposal stays frozen and symmetric inside every repeat sweep (the same
status as the table refresh itself). The eryn move is unchanged.

**Persistence.** The sidecar entry's `sigmas` are the adapted scale; a new
key, `sigmas_fisher`, carries the information-matrix floor. It is written only
while the switch is on, so an off-run entry is exactly the old entry, and no
format bump was needed. On a resume the head and the read-only compute ranks
adopt the adapted scale. A refresh that lands before the refilling buffer
reaches the minimum maps the carried scale onto the new axes
(`carried_axis_widths`: the old table read as a Gaussian covariance, exact
when the axes did not change). A resume with the switch OFF runs the
`sigmas_fisher` table.

**Note on rungs.** The source branches' shared table (walker_max / Gram) is
installed as ONE `(ndim, ndim)` table for every rung: the same cold-unit
sigma at every temperature. `eigen_refresh.temper_sigmas` is used by the
noise branches only. The adaptation keeps that behaviour, with cold-unit
widths in a shared table. (Since 2026-10-15, `{BRANCH}_EIGEN_TEMPER_SIGMAS=1`
widens it per rung at install time; see the next section.)

**Measured** (CPU, `tests/test_eigen_axis_adapt.py::MixingTest`, 5-D
Gaussian, long axis 20x the table sigma, 4 rungs x 2 walkers, 25 repeats per
visit): the cold long-axis autocorrelation time is 48 repeats with the
adaptation, against more than 900 without over the test's 10k repeats (a lower
bound) and about 6200 over a 300k-repeat run, a gain of about 130x. The adapted
long-axis sigma is 22.6 for a true 20; a chain started at the mode
bootstraps to 19.0 in 250 visits.

Log: `[EMRI_EIGEN] leaf k: step scale adapted on N of 12 axes, max x12.3
(axis 3, ridge), median x1.8 (50 visits, 8 rungs)` (also `[MBH_EIGEN]`,
`[SOBBH_EIGEN]`), once per refresh when any axis moved by more than 20 %.
The ratio is adapted / information-matrix sigma, the median is over the
adapted axes, and `ridge` marks the table's widest unrailed axis. After a
restart the parenthesis reads `carried from the persisted table, buffer n of
10 visits`. A failure warns `step-scale adaptation failed` and runs the
information-matrix table. Nothing is logged when the switch is off.

Tests: `tests/test_eigen_axis_adapt.py`.

### The safety cap (`{BRANCH}_EIGEN_ADAPT_MAX_FACTOR`)

Added 2026-10-15 (user ruling, after the 12mo's first tar with the
adaptation live). On MBH leaves 4/7/8/9/10/11 and EMRI 0/2/7 the narrowest
Fisher axis (the last axis, the best-measured phase / time combination) came
out x600-4900 wider, at the prior-box cap. The rung chains straddle discrete
degenerate solutions (sky-mode hops, phase degeneracies) inside the 50-visit
window, so the per-rung MAD measures the distance between solutions, not a
width. `draw_axis_step` picks one axis per repeat, so such an axis wastes
~1/ndim of the proposals.

- Per axis: `adapted = max(fisher, min(width, box cap, max_factor x
  fisher))`. A 5000x width becomes 100x, a 56x width is untouched, and an
  axis whose box cap is below `max_factor x fisher` stays at the box cap.
  `max_factor < 1` leaves the floor.
- Knob: kwarg / settings field `eigen_adapt_max_factor` > env
  `{BRANCH}_EIGEN_ADAPT_MAX_FACTOR` > **100**. `<= 0` or `inf` = no cap. An
  unreadable value warns once and keeps 100.
- The default is 100 although the other switches default off. The cap acts
  only inside the adaptation, so with `{BRANCH}_EIGEN_AXIS_ADAPT` off it is
  inert and stock runs stay bit-identical
  (`tests/test_eigen_pool_temper.py::GoldenStreamTest`).
- The cap also applies to an adapted scale ADOPTED from the sidecar (the
  first visit after a restart), so a sidecar written by an uncapped run does
  not serve uncapped until the leaf's next refresh. Log: `[EMRI_EIGEN] leaf
  k: persisted step scale factor-capped at x100 on N entries`.
- Tempering (`{BRANCH}_EIGEN_TEMPER_SIGMAS`) multiplies the ADAPTED,
  factor-capped sigma, with its own box cap unchanged. The sidecar stores the
  capped cold sigma.
- Log: the adaptation line gains `, factor-capped at x100 on N axes` after
  its parenthesis, and its ratios are taken after the cap, so a capped line
  reads `max x100.0`.

Tests: `tests/test_eigen_axis_adapt.py::FactorCapTest`.

## Add/remove source branches: pooled adaptation and tempered steps (`{BRANCH}_EIGEN_ADAPT_POOL`, `{BRANCH}_EIGEN_TEMPER_SIGMAS`)

Added 2026-10-15 (user ruling "just do 1 for now", item 1 of the sampling
list). Both switches are independent, code default OFF, and with both off
every proposal is bit-identical to the code before them (pinned against HEAD
aaace7ab by `tests/test_eigen_pool_temper.py::GoldenStreamTest`).

**Why** (9mo, EMRI leaf 3, SNR 50, plunge in the window). Walker 0's hot
rungs (T >= 2.1) spanned the whole degenerate ridge (p0 12.3309..12.3372,
with M and a sliding along it), while walkers 1 and 3 had ladders whose
EVERY rung stayed within 1e-4. Two causes: every rung proposed with the same
cold-sized sigma (the shared source table is untempered), and the per-rank
adaptation above can only learn 1e-4 from a stuck walker, because in
production each compute rank holds one walker. EMRI leaf 0 shows the same
(one walker crossed the e0 valley, the others creep), and MBH / SOBBH can in
principle too.

### Pooling (`{BRANCH}_EIGEN_ADAPT_POOL=1`, needs `{BRANCH}_EIGEN_AXIS_ADAPT=1`)

```
head, top of propose (fanout_prepare_state, before the state is sliced and
before the cold->hot rotation edits a hot rung):
    coords[:ntemps, :, leaf] of EVERY walker + betas_all[leaf], every alive leaf
        -> propose payload "eigen_pool"   (ntemps x nwalkers x ndim floats per leaf)
every compute rank, the head's own body included (fanout_apply_extra):
    append to its POOLED ring buffer (_eigen_pool_buffer, the same K visits)
every table refresh (_eigen_adapt_table), on the rank's OWN axes:
    pooled ring >= {BRANCH}_EIGEN_ADAPT_MIN_VISITS ? the union : the own ring
```

- The adapted sigma of every walker is the robust width of the UNION of all
  walkers' rung ensembles: the same estimator as before (cold units per
  rung, periodic wrap, axis-basis MAD, floor and cap), run on a sample that
  holds every walker. Each rung is centred on its own median over the union,
  so walkers sitting at different places on the ridge contribute their
  spread. That spread is the point of pooling.
- **Why the raw ensemble is shipped and not widths.** Each rank rebuilds its
  table from its OWN walker at every refresh (walker_max / Gram at the rank's
  max-lnL cold walker). Compute ranks adopt the head's persisted table only at
  start-up, so the axes differ between ranks. Widths measured on the head's
  axes could only be mapped onto a rank's axes through a Gaussian
  approximation (`carried_axis_widths`). The raw ensemble is as small as the
  state slice itself (8 rungs x 4 walkers x 12 dims = 384 floats per EMRI
  leaf per propose), and with it each rank computes the exact union estimator
  on its own axes.
- The state at the top of a propose is the previous propose's end-of-visit
  ensemble, so the pooled sample is what the single-process body buffers at
  the end of each visit. Its betas are one pooled ladder step later, and the
  first propose after a start also buffers the starting state.
- **Layouts.** Pooling only acts under the walker-block fan-out (production:
  one walker per compute rank). Single process and the `GF_GPU_ROUTING=1`
  replica layout run the add/remove body FLAT on one process (the head runs
  the full ensemble, the replicas only score rows), so the own buffer
  already is the union and nothing is shipped. The replicas build and hold no
  table at all.
- **Fallbacks.** Switch off: no sample, no payload key (the payload is
  byte-identical). Pooled ring shorter than the minimum (the first proposes,
  or a rank that never received a payload): the rank's own ring, exactly as
  before. A failed note or append warns (`pooled adaptation sample not
  shipped / not recorded`) and the own ring serves. Never persisted: after a
  restart the pooled ring refills like the own ring.
- Detailed balance: the buffer only feeds the table refresh, which happens at
  a leaf-visit boundary, so every repeat sweep still runs a frozen symmetric
  proposal.

### Tempered steps (`{BRANCH}_EIGEN_TEMPER_SIGMAS=1`)

- Rung `k` steps with `sigma_k = sigma / sqrt(beta_k)` (the target at inverse
  temperature `beta` is `exp(beta lnL)`, `1 / sqrt(beta)` wider than the
  cold one; `EigenAxisMove` applies no temperature scaling of its own). The
  tempered sigma is capped at the table's own cap (`axis_caps`), so a railed
  axis stays at the prior-box step, and its floor is the cold sigma. `beta <=
  0` keeps the cold sigma (`eigen_refresh.temper_sigmas`).
- The multiplier is fixed at the start of each leaf visit from the visit's
  ladder (`betas_all[leaf]`, the leaf's temperature control) and held for the
  whole visit. It depends only on the rung, never on the point, so each
  rung's proposal is a frozen symmetric kernel. Under the fan-out the ranks
  never adapt their ladders. In single-process mode the ladder adapts every
  repeat, but the step keeps the visit-start ladder.
- Installed per split (`_install_eigen_split_table`, the same seam the
  per-walker stash uses): a shared table becomes per-rung rows `(ntemps,
  nwalkers_split, 1, ndim, ndim)`, and a `per_walker` stash is tempered row
  by row at install time (so the lost-rung guard's row edits are respected).
  The random stream is unchanged: the cold rung draws bit-identically, and
  rung `k`'s step is exactly `1 / sqrt(beta_k)` times the untempered one.
- The stored table (`_eigen_tables`) and the sidecar entry stay the COLD
  table (the adapted cold sigma with `{BRANCH}_EIGEN_AXIS_ADAPT`).

### Log lines

- `[EMRI_EIGEN] leaf k: step scale adapted on N of 12 axes, max x12.3 (axis 3,
  ridge), median x1.8 (50 visits, 8 rungs, pooled over 4 walkers)` (also
  `[MBH_EIGEN]`, `[SOBBH_EIGEN]`). `pooled over W walkers` appears when the
  union served: the shipped ring on a block rank, or the own ring of a
  full-ensemble body with the pool switch on.
- With tempering on, the refresh's line carries `, tempered 1/sqrt(beta) over
  8 rungs, x1.00..x2.45` inside the same parenthesis. When the adaptation line
  does not fire (adaptation off, or nothing moved by 20 %) the refresh logs
  `[EMRI_EIGEN] leaf k: step tempered 1/sqrt(beta) over 8 rungs,
  x1.00..x2.45` instead. Either way it is once per leaf refresh.
- DEBUG, on the head, once per propose with the pool on: `[EMRI_EIGEN]
  propose payload: pooled adaptation sample, L leaves x 8 rungs x 4 walkers
  (F floats)`.
- WARNING: `no ladder to temper the eigen step with` (once per move),
  `tempered eigen step failed`, `pooled adaptation sample not shipped / not
  recorded`. Each falls back as above.

### Measured (CPU, synthetic)

`tests/test_eigen_axis_adapt.py::PooledRidgeTest`: the 5-D Gaussian with
its long axis 20x the table sigma, 4 walkers x 8 rungs (T_max 6), 25
repeats per visit, refresh every 5 visits, 160 visits. Walker 0 starts
stationary. Walkers 1-3 start with every rung at -2 sigma on the ridge.
Each walker runs on its own one-walker "rank" behind the real head and rank
hooks. Stuck walkers, seed 33:

| configuration | first visit cold c4 > +1 sigma | IAT over the run (repeats) | IAT, 2nd half |
|---|---|---|---|
| information-matrix table | 100, 83, never | 199 | 11 |
| per-rank adaptation (a523d4c0) | 31, 19, 38 | 167 | 6 |
| per-rank adaptation + tempered | 16, 10, 13 | 57 | 5 |
| pooled adaptation | 21, 18, 20 | 102 | 3 |
| pooled + tempered | 10, 10, 11 | 41 | 4 |

The 10-visit floor of the pooled rows is `{BRANCH}_EIGEN_ADAPT_MIN_VISITS`
plus the refresh cadence. Seeds 1-5 give 11-15 (pooled + tempered) against
16-37 (per-rank) visits, and an over-the-run IAT of 48-65 against 125-255.
In this Gaussian toy a stuck walker's own adaptation still bootstraps within
~30 visits. A ridge on which it cannot (the 1e-4 ladders of the 9mo) is the
case pooling is for, and no synthetic test here reproduces it.

Tests: `tests/test_eigen_pool_temper.py` (pooling across two FakeWorld
ranks with a stuck and a spanning walker, the payload seam and its
fallbacks, the tempered rows and draws, the cap, the per-walker stash, the
sidecar, the log lines, the real EMRI move single-process, under the 4-rank
block fan-out and under the 4 x 2 routed layout, the knobs, and the golden
bit-identity with both off), `tests/test_eigen_axis_adapt.py::PooledRidgeTest`.

## Noise branches: `{PSD,GALFOR}_EIGEN_INFO=fisher`

`PSDMove._refresh_eigen_tables` expands at the points `{P}_EIGEN_SCOPE` picks
(`cold_per_walker`, the default: each walker's cold row, shared up its ladder
with tempered sigmas; `per_temp`: every rung at walker 0) and takes, per
branch:

- **`"ll"`** (default): second differences of the noise likelihood through
  `compute_psd_rows`, step `{P}_EIGEN_EPS_REL` (1e-4) of the box.
- **`"fisher"`** (`PSDMove._noise_fisher`, move attribute `eigen_info`, env per
  branch): the expected Fisher

      F_ij = kappa * sum_pix Tr(C^-1 dC/dx_i C^-1 dC/dx_j)

  with `kappa` the likelihood's `logdet_factor` and `dC` central differences
  (same `{P}_EIGEN_EPS_REL` step) of the SAME PSD_BATCH covariance the noise
  scorer builds (`_batched_covariance`); the other noise branches are held
  where the ll route holds them. It needs no residual, so it is positive
  semi-definite everywhere; the observed curvature is not (34 of 55 galfor
  builds were non-positive on the 3-month run). PSD_BATCH route only (raises
  into the ll route otherwise).

Accuracy (CPU, the synthetic noise_only fit, residual drawn from `C(x0)`, 60k
pixels): psd diagonal within 0.6 % of the observed curvature; galfor (loud)
alpha / f_1 within 4 %, while the observed curvature of fk / f_2 is negative
on the same draw. Tests: `tests/test_psd_move_batched.py::NoiseFisherTest`
(psd and galfor; `SGWB_EIGEN_INFO=fisher` runs the same code but has no test);
the ll route: `tests/test_psd_eigen_inner.py`.

Related (LAT 290c1902): the PSD_BATCH scorer now folds galfor / SGWB with the
backend's `WDM_PSD_METHOD` (production `layer_calibrated`), the same model the
per-walker sensitivity uses, so both noise routes differentiate the
likelihood the rest of the run scores.

## Knob reference

The new knobs follow the naming rule (env = branch prefix + capitalized
attribute). Two older ones do not and are kept because production launch
lines use them: `{BRANCH}_EIGEN_REFRESH` seeds `eigen_refresh_every` and
`{BRANCH}_EIGEN_SCOPE` seeds `eigen_table_scope`.

| Env knob | Attribute | Code default | 6mo / 1yr / 9mo v9 launchers |
|---|---|---|---|
| `SIGHET_INFOMAT` | (GBGPU env) | off | `1` |
| `GB_INFOMAT_PER_BLOCK` | move `infomat_per_block` | `0` (VGB: always on) | `1` |
| `SIGHET_REF_BUILD` | GB / VGB Settings `sighet_ref_build` | `fd` | `lookup` |
| `SIGHET_ANCHOR_ENGINE` | GB / VGB Settings `sighet_anchor_engine` | `chunked` | `lookup` |
| `SIGHET_INFOMAT_ENGINE` | (GBGPU env) | unset | `lookup` (set-empty on the launch line turns it off) |
| `GB_CHOL_CACHE` | (env, `gb_chol_cache.py`) | `0` | `1` |
| `GB_CHOL_CACHE_EVERY` / `_TOL` / `_WINDOW` / `_BATCH` | (env) | `40` / `5` / `32` / `4096` | `40` / `5` / default / default |
| `GB_CHOL_CACHE_UID` | (env, `gb_chol_cache.py`; rollback switch) | `1` (per-source IDs) | not set (`0` = coordinate matching for every source, the pre-ID cache) |
| `{SOBBH,EMRI,MBH}_EIGEN_INFO` | move `eigen_info` | `ll` | `gram` (default since 2026-10-07, V9-30; `${K:-gram}`) |
| `{SOBBH,EMRI,MBH}_EIGEN_GRAM_TARGET` | move `eigen_gram_target` | `1e-3` | not set |
| `{SOBBH,EMRI,MBH}_EIGEN_GRAM_EPS_REL` | move `eigen_gram_eps_rel` | `{BRANCH}_EIGEN_EPS_REL` | not set |
| `{SOBBH,EMRI,MBH}_EIGEN_EPS_REL` | move `eigen_eps_rel` | `1e-4` | not set |
| `{SOBBH,EMRI,MBH}_EIGEN_AXIS_ADAPT` | Settings field + move `eigen_axis_adapt` | `False` (tables bit-identical) | not set as of 2026-10-14 (the coordinator adds the exports) |
| `{SOBBH,EMRI,MBH}_EIGEN_ADAPT_VISITS` | move `eigen_adapt_visits` | `50` (ring-buffer length, visits) | not set |
| `{SOBBH,EMRI,MBH}_EIGEN_ADAPT_MIN_VISITS` | move `eigen_adapt_min_visits` | `10` | not set |
| `{SOBBH,EMRI,MBH}_EIGEN_ADAPT_MAX_FACTOR` | Settings field + move `eigen_adapt_max_factor` | `100` (inert unless `_EIGEN_AXIS_ADAPT` is on; `<= 0` / `inf` = no cap) | not set as of 2026-10-15 (the coordinator adds the exports) |
| `{SOBBH,EMRI,MBH}_EIGEN_ADAPT_POOL` | Settings field + move `eigen_adapt_pool` | `False` (per-rank buffer, bit-identical) | not set as of 2026-10-15 (the coordinator adds the exports) |
| `{SOBBH,EMRI,MBH}_EIGEN_TEMPER_SIGMAS` | Settings field + move `eigen_temper_sigmas` | `False` (cold sigma at every rung, bit-identical) | not set as of 2026-10-15 (the coordinator adds the exports) |
| `{PSD,GALFOR,SGWB}_EIGEN_INFO` | PSDMove `eigen_info` | `ll` | `fisher` for PSD and GALFOR (default since 2026-10-07, V9-30; `${K:-fisher}`); SGWB not set |
| `{PSD,GALFOR,SGWB}_EIGEN_EPS_REL` | Settings / PSDMove `eigen_eps_rel` | `1e-4` | not set |

The `eigen_info` / `eigen_gram_*` move attributes are constructor kwargs too;
`None` (the default) defers to the env knob, read when the table is built, so
an attribute set on a live move wins. Unlike `eigen_eps_rel` /
`eigen_refresh_every` they are not Settings fields: set them on the launch
line (env) or on the built move. The same holds for `eigen_adapt_visits` /
`eigen_adapt_min_visits`. `eigen_axis_adapt` is also a field of the
EMRI / MBH / SOBBH Settings, handed to the move by `SingleSourcePEBuilder`
like `eigen_table_scope`. `eigen_adapt_pool`, `eigen_temper_sigmas` and
`eigen_adapt_max_factor` are Settings fields too, stamped onto the built move by
`stock/erebor/source_runtime.py::build_source_moves`
(`stamp_source_move_settings`; a value the builder already set wins).
