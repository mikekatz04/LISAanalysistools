# Stock global fits: stages vs. moves (and how to set them up)

How the global-fit recipe layer is structured, what the difference between a
**stage** and a **move** is, and how to edit both before (or during) a run.
Covers the recipe API in `lisatools/globalfit/recipe.py` +
`lisatools/globalfit/moves/globalfitmove.py` and the fit-level entrances in
`lisatools/globalfit/stock/base.py`.

## The one-sentence version

A **recipe** is an ordered list of **stages**; each **stage** is an ordered list
of **moves**. A *stage* is a phase of the run (search, then PE, then …) that
materializes into a runtime `RecipeStep` with its own stopping/advance logic; a
*move* (a.k.a. **proposal** — the words are synonymous) is a single sampler
proposal (an eryn move) that runs inside a stage.

```
Recipe
 └─ Stage   "gb_search"  kind="search"   ──▶ SearchRecipeStep
 │    ├─ Move "rj_fstat_search"
 │    └─ Move ...
 └─ Stage   "gb_pe"      kind="pe"        ──▶ PERecipeStep
      ├─ Move "rj_fstat_pe"
      └─ Move "rj_prior_pe"
```

(The production recipe, built by `scripts/fstat_proposal/run_combined_staged.py`
on `erebor.all_sources`, is `gb_search_seed` x3 -> `source_search` ->
`gb_search_1` -> `gb_search_2` -> `full_pe`, using the `gb_search` stage kind
and the standalone `in_model*` moves; `docs/global-fit-navigation.md` section 2.2
has it. This page covers the mechanics.)

One concept per level, each with a required **`setup(ctx)`** hook run at
materialization — there are no separate "spec" classes. Everything is
cheap-construct / heavy-setup, so a configured fit pickles.

## The simplest entrances first

Most users never need more than these. A move is any plain function with the
eryn signature — store your own information in your own closure/object, only
touch `model.analysis_container_arr` (read the residual, adjust it, write it
back):

```python
from lisatools.globalfit.stock import erebor

fit = erebor.blank(nwalkers=4, ntemps=2)      # zero branches, zero data
                                              # (include_noise=True adds a
                                              #  synthetic noise realization)

def my_move(model, state):
    aca = model.analysis_container_arr        # the residual
    return state, None                        # or (new_state, accepted)

# branch info + the move(s) that adjust it, one call — no Settings classes:
fit.add_branch("line", ndim=2, priors={0: ..., 1: ...}, moves=[my_move])

fit.run()                                     # entrance A: run generally

# the added branch was stored by the default storage machinery — read it back:
from lisatools.globalfit.hdfbackend import GFHDFBackend
reader = GFHDFBackend(fit.general_info.main_file_path)
print(reader.iteration, reader.get_chain()["line"].shape)
```

```python
# entrance B: the generator — the loop after build.
fit = erebor.blank(nwalkers=4, ntemps=2)
# adjust the branches like above — or keep it with nothing (a hidden no-op
# "idle" branch keeps the sampler ticking on a bare canvas)
fit.build()

for model, state in fit.sample(iterations=100, store=False):
    ...   # customized storage operation user implements
```

In the loop, in-place mutations of `model.analysis_container_arr` / `state`
feed the next iteration. Storage is yours in entrance B: with ``store=False``
the backend is skipped and you record what you want, in whatever form you
want; drop it and the default machinery also records every step, exactly as
in entrance A. The two entrances compose: any added branches/moves also run
inside the generator, and `fit.add_move(...)` mid-loop starts firing on the
next iteration. You never have to call `build()` — `run()`/`sample()` build
on demand (an explicit `fit.build()` still works and is where the heavy data
load happens).

Bookkeeping the framework owns for a function move: acceptance normalization
and re-syncing `state.log_like` from the residual after each call (opt out
with `sync_log_like=False`), so the saved chain stays consistent with whatever
the function did to the residual buffers. Note the ACS is per-walker and
shared across temperature rungs — a function move sees the cold-chain
residual set.

Branch appends work before build (primary) and after build but before a run
(the branch is built incrementally against the cached data load); every
info-appended branch **must be targeted by at least one move**
(`branch=<name>`) by run start — validated with an actionable error.

## The three classes

**`Move(name, *, branch=None, debug=None, every=1)`** — a move IS a proposal.
`every=N` runs it on every Nth iteration of its stage.

| how you use it | what happens at `setup(ctx)` |
|---|---|
| `Move("rj_fstat_pe", branch="gb")` (or just the string `"rj_fstat_pe"` to `add_move`) | **stock lookup**: the variant's setup function built a move under this name; the base `setup` resolves it. |
| subclass `Move`, override `setup(ctx)` | **custom construction**: build and return your eryn move from the live context (`ctx.acs`, `ctx.priors`, `ctx.curr`, …) — or return `None` to mean "`self` IS the runtime move" (then also implement `propose`). |
| plain `fn(model, state)` to `add_move` | wrapped in a `FunctionMove` (which uses exactly the return-`None` convention). |
| a constructed eryn move to `add_move` | wrapped privately; note a constructed move often holds arrays/device state, so the fit may stop being picklable — prefer the name/subclass paths. |

`ctx` is a `MoveBuildContext(recipe, engine_info, curr, acs, priors, state,
stock_moves, ntemps, nwalkers, layout, fanout, rank, state_local)`; the last
four are the multi-rank context (filled from `curr`, all `None` in a
single-process run). `debug` (`bool` or options dict) is applied
via `move.set_debug(...)` at materialization and wins over the stage's.

**`Stage(name, kind, moves, step_kwargs, combine_kwargs, debug)`**

| field | meaning |
|---|---|
| `name` | unique stage name (e.g. `"gb_pe"`, `"main"`). |
| `kind` | `"search"` \| `"pe"` \| `"rj"` \| `"gb_search"` \| `"replica_pe"` — which runtime step class `setup` produces (`SearchRecipeStep` / `PERecipeStep` / `RJRecipeStep` / `SearchStageProfileStep` / `ReplicaPERecipeStep`), i.e. when the recipe advances past it. `gb_search` (the numbered GB search stages) is an `RJRecipeStep` subclass that also applies a per-stage GB search profile on entry and folds the per-(walker, band) RJ valve into its stopping rule; `replica_pe` is a `PERecipeStep` that ends when the large-scale metrics stop trending. |
| `moves` | ordered moves — each entry is anything `add_move` accepts. |
| `step_kwargs` | passed to the `RecipeStep` constructor (e.g. RJ plateau knobs). |
| `combine_kwargs` | passed to the `GFCombineMove` that bundles the stage's moves (they propose together, sharing the analysis-container array). |
| `debug` | stage-level debug override for every move without its own. |

**`Recipe(stages)`** — the declarative stage list AND the runtime step engine,
one object. `fit.recipe` is literally what runs: before the run you edit it
(`add_stage` / `add_move` / `pop_move` / `set_move_debug` / `list_moves` …); at
run start the variant's setup function calls `recipe.setup(ctx)`, which
materializes every stage onto the same object; the sampler then consults it
each iteration for stage advance.

Everything is validated **as you edit**:

- popping an unknown move/stage lists the available ones;
- duplicate move names (across the whole recipe) and duplicate stage names are
  rejected;
- placement takes **at most one** of `before=` / `after=` / `index=`;
- with several stages, `add_move` requires `stage=` (with zero stages it
  auto-creates a `"main"` PE stage);
- a stage left with zero moves errors at materialization.

## Build → run — what materialization does

The recipe rides on `source_metadata` (re-pointed at the live object after
`build()`, so post-build edits reach the run). When the sampler starts, the
engine calls the variant's module-level `setup_recipe(...)` — the run's
`setup_function`. That function does three things:

1. **Variant pre-work** that can't live on picklable settings — for `gb_no_fg`
   this is building the chunked-het WDM likelihood (`GBWDMComputations`) and
   doing the true-point / prior seeding.
2. **Builds the stock moves for exactly the names the recipe asks for**
   (`recipe.stock_names()`). GB names are forwarded to `build_gb_moves(...)` —
   the GB *reference recipe* — steered by which names survived your edits; noise
   names (psd / galfor / sgwb) go to `build_noise_moves` (`build_psd_moves` is a
   deprecated wrapper of it); single-source branches (`mbh_pe`, `emri_pe`,
   `sobbh_pe`) use the per-branch move builders
   (`stock/erebor/source_runtime.build_source_moves`).
3. **Calls `recipe.setup(ctx)`**: every move's `setup(ctx)` runs (stock lookup
   / custom construction / function-move binding); each stage's moves are
   wrapped in one `GFCombineMove`; and each stage registers its runtime step
   (the class its `kind` selects) on the recipe under the stage's name.

**At run time** the `Recipe` is the sampler's stopping/advance controller: after
each iteration it asks the *current* step whether to move on, so a
`kind="search"` stage runs until its stopping criteria fire and then hands over
to the next stage (e.g. PE). That is exactly how a **search → PE pipeline** is
expressed: two stages, in order. (`fit.sample()` drives the same advance logic
inside the generator.)

## Stock move names (what the builders understand)

| name | what it is |
|---|---|
| `rj_fstat_pe`, `rj_fstat_pe_residual`, `rj_prior_pe`, `rj_warm_pe`, `rj_fstat_mcmc`, `rj_refit` | GB reversible-jump proposals for PE stages: F-stat grid births (GB-free galaxy grid / live-residual grid), prior-draw births, warm-start mixture births (needs `GB_WARM_START_COMPONENTS`), F-stat MCMC births, GMM-refit births (only with a refit file). The **in-model** parameter updates happen *inside* these moves — `fit.gb.num_repeat_proposals` repeats per picked source per iteration. |
| `rj_fstat_search`, `rj_prior_removal`, `rj_warm_search`, `rj_fstat_mcmc_search`, `rj_refit_search` | the search-stage GB proposals. `rj_prior_search` / `rj_prior` are legacy aliases of `rj_fstat_search` / `rj_fstat_pe`, resolved at build time only (a recipe edit by the old name finds nothing). |
| `in_model`, `in_model_fstat`, `in_model_removal` (+ `in_model_replace`, retired: composed only with `GB_SEARCH_IN_MODEL_REPLACE=1`) | standalone GB in-model passes (search stages; the per-band group rule). Built only under `GB_MODE=search` with `GB_SEARCH_IN_MODEL=1`. |
| `gb_ridge_gibbs`, `vgb_pe`, `vgb_ridge_gibbs` | GB ridge Gibbs move; the VGB in-model move and its ridge Gibbs move. |
| `psd_pe`, `psd_search`, `galfor_pe`, `galfor_search` (+ `sgwb_*`) | noise sampling (`all_sources` / `noise_only`), one move per noise branch. |
| `mbh_pe`, `emri_pe`, `sobbh_pe` | single-source per-leaf PE moves on `ResidualAddOneRemoveOneMove` (`all_sources` / `full_year_combined`). |

The key implication: **in-model parameter updates ride inside the RJ moves.** For
GB they live *inside* the RJ moves and repeat `num_repeat_proposals` times per
picked source per iteration — so a GB PE stage with just `rj_fstat_pe` (the
`gb_no_fg` default) already does both births/deaths and in-model refinement.
The standalone `in_model*` moves are extra polish passes the search stages add.

## Worked examples

### Inspect the current recipe

```python
from lisatools.globalfit.stock import erebor

fit = erebor.gb_no_fg(debug=True)      # cheap: validation + defaults only
fit.recipe.stages                      # the ordered Stage list
print(fit.list_moves())                # pretty stage → moves listing
print(fit.recipe.move_names())         # flat list of every move name
```

### Add / remove a move by name (with placement)

```python
# add the F-stat MCMC RJ proposal right after the default F-stat grid one …
# (gb_no_fg's default recipe: one stage gb_pe holding rj_fstat_pe)
fit.add_move("rj_fstat_mcmc", branch="gb", stage="gb_pe", after="rj_fstat_pe")
print("after add:", fit.recipe.move_names())

# … then take it back out
fit.pop_move("rj_fstat_mcmc")
print("after pop:", fit.recipe.move_names())

# unknown names raise, listing the available options
try:
    fit.pop_move("not_a_move")
except KeyError as err:
    print("pop unknown ->", err)
```

Placement for `add_move`: pass at most one of `before="<move>"`,
`after="<move>"`, or `index=<int>` (and `stage="<name>"` when the recipe has
more than one stage, to disambiguate — required then).

### Add / remove a whole stage (build a search → PE pipeline)

```python
from lisatools.globalfit import Move, Stage

demo = fit()   # clone: the real fit is untouched

demo.recipe.add_stage(
    Stage(
        name="gb_search",
        kind="search",
        moves=[Move("rj_fstat_search", branch="gb")],
        combine_kwargs=dict(verbose=True, share_temperature_control=False),
    ),
    before="gb_pe",     # run search first, then the existing PE stage
)
print(demo.list_moves())

demo.recipe.pop_stage("gb_search")
print(demo.list_moves())
```

`add_stage` placement mirrors `add_move`: at most one of `before=` / `after=` /
`index=`.

### Plug in a custom move (the setup-function pattern)

```python
from lisatools.globalfit import Move

class MyMove(Move):
    def __init__(self, scale, **kwargs):
        super().__init__("my_move", **kwargs)   # cheap: just config
        self.scale = scale

    def setup(self, ctx):
        # heavy construction with the live run objects
        return MyErynMove(ctx.priors["gb"], scale=self.scale)

fit.add_move(MyMove(scale=0.1, branch="gb"), stage="gb_pe")
```

Subclass-`setup` keeps the fit picklable/deepcopy-safe, which the stock layer
relies on (nothing heavy until `build()`). For the even simpler
plain-function path, see "The simplest entrances first" above.

## Mental model / gotchas

- **A move IS a proposal** — one vocabulary, one `add_move` entrance for
  everything (names, functions, Move subclasses, built moves).
- **Stage = phase of the run; move = one proposal within it.** Two stages in
  order = a pipeline (search then PE); two moves in one stage = both proposals
  applied each iteration of that phase.
- **Stock names are indirection, not instances.** `Move("rj_fstat_pe")` carries no
  move object — the variant's `setup_recipe` builds it at run start. That is
  what keeps the pre-build fit cheap and picklable.
- **`kind` picks the runtime step** (`search`/`pe`/`rj`) and therefore the
  stopping behavior; `step_kwargs` tune that step, `combine_kwargs` tune how the
  stage's moves are bundled.
- **In-model updates ride inside the GB RJ moves** (`num_repeat_proposals`), so
  a PE stage rarely needs a separate "in-model" move for GB (the search stages
  add the standalone `in_model*` passes).
- **The in-model proposals of `mbh_pe` / `emri_pe` / `sobbh_pe` / `psd_pe` /
  `galfor_pe` (and GB's in-model blocks) are shaped by an information matrix**
  whose route is a knob, not a move: `{BRANCH}_EIGEN_INFO` for the eigen
  tables, `SIGHET_INFOMAT_ENGINE` / `GB_CHOL_CACHE` for GB. See
  [`docs/eigen-info-routes.md`](https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/dev/docs/eigen-info-routes.md).
- **Edit freely, validate early.** Duplicate names, empty stages, bad `kind`,
  and conflicting placement all raise at edit/materialize time with actionable
  messages.

---

*Source: `lisatools/globalfit/recipe.py` (`Recipe`, `Stage`),
`lisatools/globalfit/moves/globalfitmove.py` (`Move`, `MoveBuildContext`),
`lisatools/globalfit/moves/functionmove.py` (`FunctionMove`), and
`lisatools/globalfit/stock/base.py` (`add_move`, `add_branch`, `sample`).*
