# Stock global fits: stages vs. moves (and how to set them up)

How the `lisatools.globalfit.stock` recipe layer is structured, what the
difference between a **stage** and a **move** is, and how to edit both before a
run. Extracted from the stock global-fit tutorial
(`examples/stock_global_fit_tutorial.ipynb`, section 3) and the recipe API in
`lisatools/globalfit/stock/base.py`.

## The one-sentence version

A **recipe** is an ordered list of **stages**; each **stage** is an ordered list
of **moves**. A *stage* is a phase of the run (search, then PE, then …) that
becomes a runtime `RecipeStep` with its own stopping/advance logic; a *move* is
a single sampler proposal (an eryn move) that runs inside a stage.

```
RecipeSpec
 └─ StageSpec   "gb_search"  kind="search"   ──▶ SearchRecipeStep
 │    ├─ MoveSpec "rj_prior_search"
 │    └─ MoveSpec ...
 └─ StageSpec   "gb_pe"      kind="pe"        ──▶ PERecipeStep
      ├─ MoveSpec "rj_prior"
      └─ MoveSpec "rj_fstat_mcmc"
```

## Two lives of the recipe

The recipe layer has two lives: a cheap **declarative spec** you edit before the
build, and the **runtime machinery** it materializes into when the run starts.

### Pre-build — what you edit

`fit.recipe` is a `RecipeSpec`: an ordered list of `StageSpec` blocks, each
holding an ordered list of `MoveSpec`s. Nothing heavy exists yet — it is pure
configuration that pickles/deepcopies.

**`StageSpec(name, kind, moves, step_kwargs, combine_kwargs, debug)`**

| field | meaning |
|---|---|
| `name` | unique stage name (e.g. `"gb_pe"`, `"gb_search"`). |
| `kind` | `"search"` \| `"pe"` \| `"rj"` — selects which runtime step class the stage becomes (`SearchRecipeStep` / `PERecipeStep` / `RJRecipeStep`). Validated in `__post_init__`. |
| `moves` | ordered list of `MoveSpec`s for this stage. |
| `step_kwargs` | passed to the `RecipeStep` constructor (stopping criteria, iteration caps, …). |
| `combine_kwargs` | passed to the `GFCombineMove` that bundles the stage's moves (stock materialization). |
| `debug` | stage-level debug override applied to every move in the stage that has no `MoveSpec.debug` of its own (`bool` or an options dict). |

**`MoveSpec(name, target, kwargs, branch, weight, instance, debug)`**

| field | meaning |
|---|---|
| `name` | unique (per recipe) move name. On the **stock path** this is *just a name* — the variant's setup function knows how to build the move behind each canonical name. |
| `target` | how to build a **custom** move: a callable invoked as `target(ctx, **kwargs)` where `ctx = MoveBuildContext(recipe, engine_info, curr, acs, priors, state)` and which returns an eryn move. (A context-free class is instead called `target(**kwargs)`.) `None` = stock path. |
| `kwargs` | keyword args for `target` (or forwarded to the stock builder where supported). |
| `branch` | which branch this move samples; validated against the enabled branches at build. |
| `weight` | optional weight when the stage combines moves in a weighted eryn move list. |
| `instance` | a **fully constructed** move. Takes precedence over `target`. Note: a constructed move often holds arrays/device state, so a fit configured with an `instance` may stop being picklable — prefer the spec (`name`/`target`) path. |
| `debug` | move-level debug override (`bool` or options dict), applied via `move.set_debug(...)` at materialization; takes precedence over the stage's `debug`. |

Everything is validated **as you edit**:

- popping an unknown move/stage lists the available ones;
- duplicate move names (across the whole recipe) and duplicate stage names are
  rejected;
- placement takes **at most one** of `before=` / `after=` / `index=`;
- a stage left with zero moves errors at build time.

### Build → run — what the spec becomes

The spec rides through the settings deepcopy on `source_metadata`, and when the
sampler starts, the engine calls the variant's module-level `setup_recipe(...)`
— the run's `setup_function`. That function does three things:

1. **Variant pre-work** that can't live on picklable settings — for `gb_no_fg`
   this is building the chunked-het WDM likelihood (`GBWDMComputations`) and
   doing the true-point / prior seeding.
2. **Builds the stock moves for exactly the names present in the spec.** GB
   names are forwarded to `build_gb_moves(...)` — the GB *reference recipe* in
   `lisatools.globalfit.recipe` — steered by which names survived your edits
   (`pe_move_names`, `include_search`, `include_refit`). PSD names go to
   `build_psd_moves`; single-source branches (`mbh_pe`, `emri_pe`, `sobbh_pe`)
   use the per-branch move builders.
3. **Materializes each stage**: every `MoveSpec` resolves in the order
   `instance` > `target` > stock name; the stage's moves are wrapped in one
   `GFCombineMove` (proposed together, sharing the analysis-container array);
   and the stage becomes a `SearchRecipeStep` / `PERecipeStep` / `RJRecipeStep`
   registered on the runtime `Recipe` under the stage's name.

**At run time** the `Recipe` is the sampler's stopping/advance controller: after
each iteration it asks the *current* step whether to move on, so a
`kind="search"` stage runs until its stopping criteria fire and then hands over
to the next stage (e.g. PE). That is exactly how a **search → PE pipeline** is
expressed: two stages, in order.

## Stock move names (what the builders understand)

| name | what it is |
|---|---|
| `rj_prior`, `rj_fstat_mcmc`, `rj_refit` | GB reversible-jump proposals (prior-draw births, F-stat MCMC births, GMM-refit births). The **in-model** parameter updates happen *inside* these moves — `fit.gb.num_repeat_proposals` repeats per picked source per iteration. |
| `rj_prior_search`, `rj_fstat_mcmc_search`, `rj_refit_search` | the same proposals configured for a `kind="search"` stage. |
| `psd_pe`, `psd_search` | instrumental-noise sampling (`all_sources`). |
| `mbh_pe`, `emri_pe`, `sobbh_pe` | single-source stretch/RJ moves (`all_sources` / `full_year_combined`). |

The key implication: **in-model parameter updates are not a separate move.** For
GB they live *inside* the RJ moves and repeat `num_repeat_proposals` times per
picked source per iteration — so a GB PE stage with just `rj_prior` already does
both births/deaths and in-model refinement.

## Worked examples

### Inspect the current recipe

```python
from lisatools.globalfit.stock import erebor

fit = erebor.gb_no_fg(debug=True)      # cheap: validation + defaults only
fit.recipe.stages                      # the ordered StageSpec list
print(fit.list_moves())                # pretty stage → moves listing
print(fit.recipe.move_names())         # flat list of every move name
```

### Add / remove a move by name (with placement)

```python
from lisatools.globalfit.stock import MoveSpec

# add the F-stat RJ proposal right after the prior one …
fit.recipe.add_move(
    MoveSpec("rj_fstat_mcmc", branch="gb"), stage="gb_pe", after="rj_prior"
)
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

Placement for `add_move`: pass exactly one of `before="<move>"`,
`after="<move>"`, or `index=<int>` (and `stage="<name>"` when the recipe has
more than one stage, to disambiguate).

### Add / remove a whole stage (build a search → PE pipeline)

```python
from lisatools.globalfit.stock import MoveSpec, StageSpec

demo = fit()   # clone: the real fit is untouched

demo.recipe.add_stage(
    StageSpec(
        name="gb_search",
        kind="search",
        moves=[MoveSpec("rj_prior_search", branch="gb")],
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

### Plug in a custom move

```python
from lisatools.globalfit.stock import MoveSpec

def build_my_move(ctx, **kwargs):
    # ctx = MoveBuildContext(recipe, engine_info, curr, acs, priors, state)
    return MyErynMove(ctx.priors["gb"], **kwargs)

fit.recipe.add_move(
    MoveSpec("my_move", target=build_my_move, kwargs=dict(scale=0.1), branch="gb"),
    stage="gb_pe",
)
```

Use `target=` (a builder callable) rather than `instance=` whenever you can —
`target` keeps the fit picklable/deepcopy-safe, which the stock layer relies on
(nothing heavy until `build()`).

## Mental model / gotchas

- **Stage = phase of the run; move = one proposal within it.** Two stages in
  order = a pipeline (search then PE); two moves in one stage = both proposals
  applied each iteration of that phase.
- **Stock names are indirection, not instances.** A `MoveSpec("rj_prior")`
  carries no move object — the variant's `setup_recipe` builds it at run start.
  That is what keeps the pre-build fit cheap and picklable.
- **`kind` picks the runtime step** (`search`/`pe`/`rj`) and therefore the
  stopping behavior; `step_kwargs` tune that step, `combine_kwargs` tune how the
  stage's moves are bundled.
- **In-model updates ride inside the GB RJ moves** (`num_repeat_proposals`), so
  you rarely add a separate "in-model" move for GB.
- **Edit freely, validate early.** Duplicate names, empty stages, bad `kind`,
  and conflicting placement all raise at edit/build time with actionable
  messages.

---

*Source: `examples/stock_global_fit_tutorial.ipynb` (§3 "Edit the recipe" and
"How stages and moves actually work") and `lisatools/globalfit/stock/base.py`
(`MoveSpec`, `StageSpec`, `RecipeSpec`).*
