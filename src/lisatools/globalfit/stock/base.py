"""Erebor-agnostic machinery for stock global-fit configurations.

A *stock global fit* is a configured-but-unbuilt run: a
:class:`StockGlobalFit` subclass holds the full building-block pyramid as
cheap, picklable settings objects and defers every heavy step (data load,
domain pour, HDF backend creation, waveform construction) to an explicit
:meth:`StockGlobalFit.build` call.

The pyramid, bottom-up::

    atoms (transforms, priors, MoveSpecs, injection tables)
      -> branch blocks   (GBSettings, MBHSettings, ... dataclasses)
      -> general block   (GeneralSettings) + recipe (RecipeSpec of stages)
      -> the fit         (StockGlobalFit variant) --build()--> CurrentInfoGlobalFit
      -> the run         (GlobalFit(curr, comm).run_global_fit())

Every level is swapped by plain assignment or a paired ``add_*``/``pop_*``
(or ``remove_*``) method; subbing a block replaces everything beneath it and
nothing above it.

"Spec" objects (:class:`MoveSpec`, :class:`StageSpec`, :class:`RecipeSpec`)
are declarative *specifications*: picklable descriptions of the runtime
moves/recipe steps that are materialized only inside the run's
``setup_function`` (compare :class:`importlib.machinery.ModuleSpec`).
"""

from __future__ import annotations

import dataclasses
import os
import typing
from copy import deepcopy

from ..engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
    Settings,
    Setup,
)
from ..recipe import PERecipeStep, Recipe, RJRecipeStep, SearchRecipeStep
from ..run import CurrentInfoGlobalFit

__all__ = [
    "MoveSpec",
    "StageSpec",
    "RecipeSpec",
    "StockGlobalFit",
    "StockRegistry",
    "env_default",
    "env_resolve",
    "materialize_recipe",
]


# ---------------------------------------------------------------------------
# Environment-backed defaults
# ---------------------------------------------------------------------------

_BOOL_TRUE = ("1", "true", "yes", "on")
_BOOL_FALSE = ("0", "false", "no", "off")


def env_resolve(var: str, default, cast=str):
    """Resolve ``default`` against the environment: env var wins over the hard default.

    ``cast`` converts the raw string; ``bool`` accepts 1/0, true/false,
    yes/no, on/off (case-insensitive). Explicit user kwargs always beat this
    (they never consult it).
    """
    raw = os.environ.get(var)
    if raw is None or raw.strip() == "":
        return default
    if cast is bool:
        low = raw.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(f"Environment variable {var}={raw!r} is not a valid boolean.")
    return cast(raw)


def env_default(var: str, default, cast=str):
    """``dataclasses.field(default_factory=...)`` helper for env-backed defaults.

    Resolution order is *explicit kwarg > environment variable > hard
    default* — the factory only runs when the field is not passed
    explicitly, and it reads the environment at construction time.
    """

    def _factory():
        return env_resolve(var, default, cast)

    return _factory


# ---------------------------------------------------------------------------
# Declarative recipe layer: MoveSpec / StageSpec / RecipeSpec
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MoveSpec:
    """Specification of one sampler move inside a recipe stage.

    Args:
        name: Unique (per recipe) move name. Stock variants reuse the
            canonical names of the moves they wrap (``"rj_prior"``,
            ``"rj_fstat_mcmc"``, ``"rj_refit"``, ``"gb_in_model"``,
            ``"psd_pe"``, ``"mbh_pe"``, ...).
        target: How to obtain the runtime move when the stage is
            materialized. ``None`` means the variant's ``setup_function``
            provides the move under ``name`` (the stock path). A callable is
            invoked as ``target(ctx, **kwargs)`` where ``ctx`` is the
            :class:`MoveBuildContext`; a class with a context-free
            constructor may also be given and is called as
            ``target(**kwargs)``.
        kwargs: Keyword arguments for ``target`` (or forwarded to the stock
            builder where supported).
        branch: Branch this move samples (validated against enabled
            branches at build).
        weight: Optional weight used when the containing stage combines
            moves in a weighted eryn move list.
        instance: A fully constructed move. Takes precedence over
            ``target``. NOTE: constructed moves often hold arrays or
            device state — a fit configured with an ``instance`` may not be
            picklable (the spec path is the primary interface).
    """

    name: str
    target: typing.Any = None
    kwargs: dict = dataclasses.field(default_factory=dict)
    branch: typing.Optional[str] = None
    weight: typing.Optional[float] = None
    instance: typing.Any = None


@dataclasses.dataclass
class StageSpec:
    """Specification of one recipe stage (materializes into a RecipeStep).

    Args:
        name: Unique stage name (e.g. ``"gb_pe"``, ``"gb_search"``).
        kind: ``"search"`` | ``"pe"`` | ``"rj"`` — selects
            :class:`SearchRecipeStep` / :class:`PERecipeStep` /
            :class:`RJRecipeStep` at materialization.
        moves: Ordered move specifications for this stage.
        step_kwargs: Extra keyword arguments passed to the RecipeStep
            constructor (stopping criteria, iteration caps, ...).
        combine_kwargs: Extra keyword arguments for the ``GFCombineMove``
            wrapping this stage's moves (stock materialization only).
    """

    name: str
    kind: str = "pe"
    moves: typing.List[MoveSpec] = dataclasses.field(default_factory=list)
    step_kwargs: dict = dataclasses.field(default_factory=dict)
    combine_kwargs: dict = dataclasses.field(default_factory=dict)

    _KINDS = ("search", "pe", "rj")

    def __post_init__(self):
        if self.kind not in self._KINDS:
            raise ValueError(f"StageSpec kind must be one of {self._KINDS}, got {self.kind!r}.")

    def move_names(self) -> typing.List[str]:
        return [m.name for m in self.moves]


class RecipeSpec:
    """Ordered, editable collection of :class:`StageSpec` blocks.

    Supports whole-stage and per-move composition::

        recipe.add_stage(StageSpec("gb_search", kind="search", moves=[...]))
        recipe.pop_stage("gb_search")
        recipe.add_move(MoveSpec("my_move", target=...), stage="gb_pe",
                        after="rj_prior")
        recipe.pop_move("rj_refit")
        print(recipe.list_moves())
    """

    def __init__(self, stages: typing.Optional[typing.List[StageSpec]] = None):
        self.stages: typing.List[StageSpec] = list(stages) if stages is not None else []
        self._check_unique()

    # -- internals ---------------------------------------------------------

    def _check_unique(self):
        stage_names = [s.name for s in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError(f"Duplicate stage names: {stage_names}.")
        move_names = [m.name for s in self.stages for m in s.moves]
        if len(set(move_names)) != len(move_names):
            raise ValueError(f"Duplicate move names across recipe: {move_names}.")

    def _stage(self, name: str) -> StageSpec:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(
            f"Unknown stage {name!r}. Available stages: {[s.name for s in self.stages]}."
        )

    def _find_move(
        self, name: str, stage: typing.Optional[str] = None
    ) -> typing.Tuple[StageSpec, int]:
        hits = []
        for st in self.stages if stage is None else [self._stage(stage)]:
            for i, mv in enumerate(st.moves):
                if mv.name == name:
                    hits.append((st, i))
        if not hits:
            raise KeyError(
                f"Unknown move {name!r}. Available moves: {self.move_names()}."
            )
        if len(hits) > 1:
            raise KeyError(
                f"Move name {name!r} appears in multiple stages "
                f"({[st.name for st, _ in hits]}); pass stage=... to disambiguate."
            )
        return hits[0]

    # -- stages --------------------------------------------------------------

    def add_stage(
        self,
        stage: StageSpec,
        before: typing.Optional[str] = None,
        after: typing.Optional[str] = None,
        index: typing.Optional[int] = None,
    ) -> StageSpec:
        if not isinstance(stage, StageSpec):
            raise TypeError(f"add_stage expects a StageSpec, got {type(stage).__name__}.")
        if sum(x is not None for x in (before, after, index)) > 1:
            raise ValueError("Pass at most one of before=, after=, index=.")
        if any(s.name == stage.name for s in self.stages):
            raise ValueError(f"Stage name {stage.name!r} already present.")
        if before is not None:
            index = self.stages.index(self._stage(before))
        elif after is not None:
            index = self.stages.index(self._stage(after)) + 1
        elif index is None:
            index = len(self.stages)
        self.stages.insert(index, stage)
        self._check_unique()
        return stage

    def pop_stage(self, name: str) -> StageSpec:
        stage = self._stage(name)
        self.stages.remove(stage)
        return stage

    # -- moves ---------------------------------------------------------------

    def move_names(self) -> typing.List[str]:
        return [m.name for s in self.stages for m in s.moves]

    def get_move(self, name: str, stage: typing.Optional[str] = None) -> MoveSpec:
        st, i = self._find_move(name, stage)
        return st.moves[i]

    def pop_move(self, name: str, stage: typing.Optional[str] = None) -> MoveSpec:
        st, i = self._find_move(name, stage)
        return st.moves.pop(i)

    def add_move(
        self,
        move: typing.Union[MoveSpec, typing.Any],
        stage: typing.Optional[str] = None,
        before: typing.Optional[str] = None,
        after: typing.Optional[str] = None,
        index: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
    ) -> MoveSpec:
        """Add a move to the recipe.

        ``move`` is a :class:`MoveSpec` or a constructed move object (wrapped
        into a spec via its ``.name`` attribute or the ``name=`` argument;
        note the pickle caveat on :attr:`MoveSpec.instance`). Placement:
        ``stage=`` names the stage (default: the only stage, error if
        ambiguous) and at most one of ``before=``/``after=`` (move names) or
        ``index=`` positions it inside the stage (default: append).
        """
        if not isinstance(move, MoveSpec):
            mv_name = name or getattr(move, "name", None)
            if not mv_name:
                raise ValueError(
                    "A constructed move needs a name: pass add_move(..., name=...) "
                    "or give the move object a .name attribute."
                )
            move = MoveSpec(name=mv_name, instance=move)
        if move.name in self.move_names():
            raise ValueError(
                f"Move name {move.name!r} already present "
                f"(existing: {self.move_names()}). pop_move it first or rename."
            )
        if sum(x is not None for x in (before, after, index)) > 1:
            raise ValueError("Pass at most one of before=, after=, index=.")

        if stage is not None:
            st = self._stage(stage)
        elif before is not None or after is not None:
            st, _ = self._find_move(before if before is not None else after)
        elif len(self.stages) == 1:
            st = self.stages[0]
        else:
            raise ValueError(
                f"Multiple stages present ({[s.name for s in self.stages]}); "
                "pass stage=... to say where the move goes."
            )

        if before is not None:
            index = [m.name for m in st.moves].index(before)
        elif after is not None:
            index = [m.name for m in st.moves].index(after) + 1
        elif index is None:
            index = len(st.moves)
        st.moves.insert(index, move)
        return move

    def list_moves(self) -> str:
        """Human-readable, stage-grouped summary of the recipe."""
        lines = []
        for st in self.stages:
            lines.append(f"[{st.kind}] {st.name}:")
            if not st.moves:
                lines.append("    (no moves)")
            for mv in st.moves:
                src = (
                    "instance"
                    if mv.instance is not None
                    else (getattr(mv.target, "__name__", None) or mv.target or "stock")
                )
                extra = f" branch={mv.branch}" if mv.branch else ""
                lines.append(f"    {mv.name}  <- {src}{extra}")
        return "\n".join(lines) if lines else "(empty recipe)"

    def __repr__(self):
        return f"RecipeSpec({[s.name for s in self.stages]})"


@dataclasses.dataclass
class MoveBuildContext:
    """Runtime context handed to MoveSpec targets during materialization."""

    recipe: Recipe
    engine_info: typing.Any
    curr: CurrentInfoGlobalFit
    acs: typing.Any
    priors: dict
    state: typing.Any


_STEP_CLASSES = {"search": SearchRecipeStep, "pe": PERecipeStep, "rj": RJRecipeStep}


def materialize_recipe(
    recipe: Recipe,
    recipe_spec: RecipeSpec,
    ctx: MoveBuildContext,
    stock_moves: typing.Dict[str, typing.Any],
    ntemps: int,
    nwalkers: int,
) -> None:
    """Materialize a :class:`RecipeSpec` into runtime recipe components.

    ``stock_moves`` maps canonical move names to constructed move objects —
    the variant's ``setup_function`` builds them (via ``build_gb_moves`` and
    friends) for exactly the names present in the spec. Per move, resolution
    order is ``instance`` > ``target`` > ``stock_moves[name]``.
    """
    import numpy as np

    from ..moves import GFCombineMove

    enabled = set(ctx.curr.engine_info.branch_names)
    for st in recipe_spec.stages:
        moves = []
        weights = []
        for mv in st.moves:
            if mv.branch is not None and mv.branch not in enabled:
                raise ValueError(
                    f"Move {mv.name!r} targets branch {mv.branch!r} which is not an "
                    f"enabled branch ({sorted(enabled)}). remove the move or add the branch."
                )
            if mv.instance is not None:
                move = mv.instance
            elif mv.target is not None:
                try:
                    move = mv.target(ctx, **mv.kwargs)
                except TypeError:
                    move = mv.target(**mv.kwargs)
            elif mv.name in stock_moves:
                move = stock_moves[mv.name]
            else:
                raise ValueError(
                    f"Move {mv.name!r} has no instance/target and no stock builder "
                    f"provided one (stock moves available: {sorted(stock_moves)})."
                )
            moves.append(move)
            weights.append(mv.weight if mv.weight is not None else 1.0)

        if not moves:
            raise ValueError(f"Stage {st.name!r} has no moves; pop_stage it or add moves.")

        combined = GFCombineMove(moves=moves, **st.combine_kwargs)
        if not hasattr(combined, "accepted") or combined.accepted is None:
            combined.accepted = np.zeros((ntemps, nwalkers))
        step_cls = _STEP_CLASSES[st.kind]
        recipe.add_recipe_component(step_cls(moves=[combined], **st.step_kwargs), name=st.name)


# ---------------------------------------------------------------------------
# The deferred-build fit object
# ---------------------------------------------------------------------------

# CurrentInfoGlobalFit attributes that only exist after build(); accessing
# them earlier gets a helpful error instead of a bare AttributeError.
_BUILT_ONLY_ATTRS = ("settings_dict", "current_info", "backend")


class StockGlobalFit(CurrentInfoGlobalFit):
    """Base class for stock (configured-but-unbuilt) global fits.

    Subclasses fill in the default building blocks; ``__init__`` stays cheap
    (no data loads, no directories, no waveform or HDF-backend
    construction) and the object stays picklable until :meth:`build` runs.
    After :meth:`build`, the instance IS the :class:`CurrentInfoGlobalFit`
    that :class:`~lisatools.globalfit.run.GlobalFit` consumes.

    Subclass contract — override:

    - ``option_name`` / ``description`` class attributes (registry entry);
    - :meth:`default_general` returning the variant's
      :class:`~lisatools.globalfit.engine.GeneralSettings` (subclass)
      instance;
    - :meth:`default_branches` returning an ordered ``{name: Settings}``
      dict;
    - :meth:`default_recipe` returning the variant's :class:`RecipeSpec`;
    - ``setup_classes`` mapping branch name -> Setup class;
    - ``default_setup_function`` (a named module-level function);
    - :meth:`make_general_settings` (optional) to resolve derived fields
      (domain factories, processor kwargs) right before the heavy build.
    """

    option_name: typing.ClassVar[str] = ""
    description: typing.ClassVar[str] = ""
    setup_classes: typing.ClassVar[typing.Dict[str, type]] = {}
    default_setup_function: typing.ClassVar[typing.Optional[typing.Callable]] = None

    # Headline knobs settable directly on the fit; everything they alias
    # lives on ``self.general``.
    _HEADLINE_KNOBS = (
        "nwalkers",
        "ntemps",
        "dt",
        "num_iterations",
        "file_store_dir",
        "base_file_name",
        "gpus",
        "gpu_backend",
    )

    def __init__(
        self,
        *,
        general: typing.Optional[GeneralSettings] = None,
        branches: typing.Optional[typing.Dict[str, Settings]] = None,
        recipe: typing.Optional[RecipeSpec] = None,
        setup_function: typing.Optional[typing.Callable] = None,
        head_rank: int = 1,
        main_rank: int = 0,
        **knobs,
    ):
        # NOTE: deliberately does NOT call CurrentInfoGlobalFit.__init__ —
        # that is the heavy stage, deferred to build().
        self.general = general if general is not None else self.default_general()
        self._branch_names: typing.List[str] = []
        for name, settings in (
            branches if branches is not None else self.default_branches()
        ).items():
            self.add_branch(name, settings)
        self.recipe = recipe if recipe is not None else self.default_recipe()
        self.setup_function = (
            setup_function if setup_function is not None else type(self).default_setup_function
        )
        self.head_rank = head_rank
        self.main_rank = main_rank

        self._apply_knobs(knobs)

    # -- defaults hooks ------------------------------------------------------

    def default_general(self) -> GeneralSettings:
        raise NotImplementedError

    def default_branches(self) -> typing.Dict[str, Settings]:
        raise NotImplementedError

    def default_recipe(self) -> RecipeSpec:
        raise NotImplementedError

    # -- knob application ------------------------------------------------------

    def _apply_knobs(self, knobs: dict):
        """Apply constructor kwargs: headline knobs and variant attributes."""
        for key, value in knobs.items():
            if key in self._HEADLINE_KNOBS:
                setattr(self.general, key, value)
            elif hasattr(type(self), key) or key in self.__dict__:
                setattr(self, key, value)
            elif hasattr(self.general, key):
                setattr(self.general, key, value)
            else:
                raise TypeError(
                    f"{type(self).__name__} got an unexpected keyword argument {key!r}. "
                    "Adjust nested blocks directly (fit.general.<field>, fit.gb.<field>, "
                    "fit.recipe...) for anything beyond the headline knobs."
                )

    # -- headline knob delegation ---------------------------------------------

    def __getattr__(self, name):
        # Only called when normal lookup fails. Guard the recursion-prone
        # names first (sprint __getattr__ rule).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in ("general", "_branch_names"):
            raise AttributeError(name)
        if name in type(self)._HEADLINE_KNOBS:
            return getattr(self.general, name)
        if name in _BUILT_ONLY_ATTRS or hasattr(CurrentInfoGlobalFit, name):
            # Either a build-product attribute, or a CurrentInfoGlobalFit
            # property that failed internally on one (property AttributeErrors
            # re-dispatch here under the property's own name).
            raise AttributeError(
                f"{type(self).__name__}.{name} is only available after the fit is "
                "built — call .build() first."
            )
        raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in type(self)._HEADLINE_KNOBS and "general" in self.__dict__:
            setattr(self.general, name, value)
            return
        super().__setattr__(name, value)

    # -- branch composition ----------------------------------------------------

    @property
    def branches(self) -> typing.Dict[str, Settings]:
        """Ordered mapping of branch name -> branch Settings block."""
        return {name: getattr(self, name) for name in self._branch_names}

    def add_branch(self, name: str, settings: Settings):
        """Add (or replace) a branch block. ``fit.<name>`` becomes the settings."""
        if not isinstance(settings, Settings):
            raise TypeError(
                f"Branch {name!r}: expected a Settings dataclass instance, got "
                f"{type(settings).__name__}."
            )
        if name not in self._branch_names:
            if (
                name in self._HEADLINE_KNOBS
                or hasattr(type(self), name)
                or name in self.__dict__  # e.g. "recipe", "general", "setup_function"
            ):
                raise ValueError(
                    f"Branch name {name!r} clashes with an existing "
                    f"{type(self).__name__} attribute."
                )
            self._branch_names.append(name)
        object.__setattr__(self, name, settings)

    def remove_branch(self, name: str) -> Settings:
        """Remove a branch block and return its settings."""
        if name not in self._branch_names:
            raise KeyError(
                f"Unknown branch {name!r}. Present branches: {self._branch_names}."
            )
        self._branch_names.remove(name)
        settings = self.__dict__.pop(name)
        return settings

    # convenience aliases matching the recipe layer's add/pop vocabulary
    def add_move(self, *args, **kwargs):
        return self.recipe.add_move(*args, **kwargs)

    def pop_move(self, *args, **kwargs):
        return self.recipe.pop_move(*args, **kwargs)

    def list_moves(self) -> str:
        return self.recipe.list_moves()

    # -- lifecycle ---------------------------------------------------------------

    @property
    def built(self) -> bool:
        return "settings_dict" in self.__dict__

    def make_general_settings(self) -> GeneralSettings:
        """Resolve the general block into the final GeneralSettings for the build.

        The default returns a deepcopy of ``self.general``; variants override
        to fill derived fields (domain factories, processor kwargs) from the
        fit-level knobs at the last moment.
        """
        return deepcopy(self.general)

    def build_general(self, force: bool = False) -> GeneralSetup:
        """Build the (heavy) :class:`GeneralSetup`: data load, domain pour, sensitivity."""
        if force or getattr(self, "_general_setup", None) is None:
            self._general_setup = GeneralSetup(self.make_general_settings())
        return self._general_setup

    def make_source_setup(self, name: str, settings: Settings, general_setup: GeneralSetup):
        """Turn one branch Settings block into its Setup via ``setup_classes``."""
        try:
            setup_cls = self.setup_classes[name]
        except KeyError:
            raise KeyError(
                f"No Setup class registered for branch {name!r} on "
                f"{type(self).__name__}.setup_classes ({sorted(self.setup_classes)}). "
                "Register one or override make_source_setup."
            ) from None
        return setup_cls(settings)

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        """Hook: resolve a branch block against the built general setup.

        Default fills ``Tobs``/``dt`` from the general setup when unset and
        returns a deepcopy (the fit-level block stays untouched/re-buildable).
        """
        settings = deepcopy(getattr(self, name))
        if settings.Tobs is None:
            settings.Tobs = general_setup.Tobs
        if settings.dt is None:
            settings.dt = general_setup.dt
        return settings

    def build_sources(self, force: bool = False) -> typing.Dict[str, Setup]:
        if force or getattr(self, "_source_info", None) is None:
            general_setup = self.build_general(force=False)
            self._source_info = {
                name: self.make_source_setup(
                    name, self.prepare_branch_settings(name, general_setup), general_setup
                )
                for name in self._branch_names
            }
        return self._source_info

    def build(self, force: bool = False) -> "StockGlobalFit":
        """Run the full heavy build and become a usable CurrentInfoGlobalFit."""
        if self.built and not force:
            return self
        if force:
            self.reset_build()
        general_setup = self.build_general(force=force)
        source_info = self.build_sources(force=False)
        if self.setup_function is None:
            raise ValueError(
                f"{type(self).__name__}.setup_function is None — set a recipe "
                "setup function before building."
            )
        settings = GlobalFitSettings(
            source_info=source_info,
            general_info=general_setup,
            rank_info=RankInfo(head_rank=self.head_rank, main_rank=self.main_rank),
            setup_function=self.setup_function,
        )
        # The recipe spec rides along so the setup_function can materialize it.
        settings.source_metadata["recipe_spec"] = self.recipe
        CurrentInfoGlobalFit.__init__(self, settings)
        # Post-deepcopy hook: attach runtime-only objects (e.g. per-branch
        # ``signal_gen`` adapters bound to the runtime general_info) onto the
        # deepcopied Setups — never onto the pre-build config.
        self.attach_runtime_objects()
        return self

    def attach_runtime_objects(self) -> None:
        """Variant hook, called once at the end of :meth:`build`."""

    def reset_build(self):
        """Drop all built products, returning to the cheap configured state."""
        for attr in ("_general_setup", "_source_info", *_BUILT_ONLY_ATTRS):
            self.__dict__.pop(attr, None)

    def run(self, comm=None, **run_kwargs):
        """One-shot pipeline: build (if needed) -> GlobalFit -> run_global_fit.

        ``comm`` defaults to ``MPI.COMM_WORLD``. ``run_kwargs`` pass through
        to ``GlobalFit.run_global_fit``.
        """
        self.build()
        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        from ..run import GlobalFit

        gf = GlobalFit(self, comm)
        return gf.run_global_fit(**run_kwargs)

    # -- introspection / cloning / pickling ---------------------------------------

    def describe(self) -> str:
        """Multi-line summary of the configured blocks and headline knobs."""
        lines = [
            f"{type(self).__name__} ({self.option_name or 'unregistered'}) — "
            f"{self.description or 'no description'}",
            f"  built: {self.built}",
            "  headline:",
        ]
        for key in self._HEADLINE_KNOBS:
            lines.append(f"    {key} = {getattr(self.general, key, None)!r}")
        lines.append(f"  branches: {self._branch_names}")
        for name in self._branch_names:
            lines.append(f"    {name}: {type(getattr(self, name)).__name__}")
        lines.append("  recipe:")
        for ln in self.list_moves().splitlines():
            lines.append(f"    {ln}")
        lines.append(f"  setup_function: {getattr(self.setup_function, '__name__', None)}")
        return "\n".join(lines)

    def apply_overrides(self, overrides: dict) -> None:
        """Apply override knobs to this instance.

        Variants intercept special constructor knobs here (e.g. gb_no_fg's
        ``debug`` preset) so that cloning via ``__call__`` honors them the
        same way construction does.
        """
        self._apply_knobs(overrides)

    def __call__(self, **overrides) -> "StockGlobalFit":
        """Clone this configuration (config only, never built products) with overrides."""
        new = deepcopy(self)
        new.reset_build()
        new.apply_overrides(overrides)
        return new

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in ("_general_setup", "_source_info", *_BUILT_ONLY_ATTRS):
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        status = "built" if self.built else "unbuilt"
        return (
            f"<{type(self).__name__} [{status}] branches={self._branch_names} "
            f"nwalkers={getattr(self.general, 'nwalkers', None)} "
            f"ntemps={getattr(self.general, 'ntemps', None)}>"
        )

    @classmethod
    def from_env(cls, **overrides) -> "StockGlobalFit":
        """Construct with environment-variable defaults applied (see env_default).

        Plain construction already resolves env-backed field defaults; this
        alias exists so call sites can say explicitly that they want the
        env-driven configuration (legacy settings-file stubs use it).
        """
        return cls(**overrides)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class StockRegistry:
    """Name -> StockGlobalFit-class registry for one stock family (e.g. erebor)."""

    def __init__(self, family: str):
        self.family = family
        self._classes: typing.Dict[str, type] = {}

    def register(self, cls: type) -> type:
        """Class decorator: register ``cls`` under its ``option_name``."""
        name = getattr(cls, "option_name", "")
        if not name:
            raise ValueError(f"{cls.__name__} needs a non-empty option_name to register.")
        if name in self._classes:
            raise ValueError(f"Stock option {name!r} already registered in {self.family}.")
        self._classes[name] = cls
        return cls

    def options(self) -> typing.List[typing.Tuple[str, str]]:
        """List of ``(option_name, description)`` pairs."""
        return [(name, cls.description) for name, cls in self._classes.items()]

    def names(self) -> typing.List[str]:
        return list(self._classes)

    def get(self, option: str, *args, **kwargs) -> StockGlobalFit:
        """Instantiate the stock fit registered under ``option``."""
        try:
            cls = self._classes[option]
        except KeyError:
            raise ValueError(
                f"Unknown stock option {option!r} in family {self.family!r}. "
                f"Available: {self.names()}."
            ) from None
        return cls(*args, **kwargs)
