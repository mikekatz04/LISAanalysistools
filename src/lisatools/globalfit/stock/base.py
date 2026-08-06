"""Erebor-agnostic machinery for stock global-fit configurations.

A *stock global fit* is a configured-but-unbuilt run: a
:class:`StockGlobalFit` subclass holds the full building-block stack as
cheap, picklable settings objects and defers every heavy step (data load,
domain pour, HDF backend creation, waveform construction) to an explicit
:meth:`StockGlobalFit.build` call.

The stack, top-down::

    the run      GlobalFit(curr, comm).run_global_fit()   (or fit.sample())
       ^  .build() -> GlobalFitSetup       (heavy: built config + live state)
    the fit      StockGlobalFit variant    (cheap, picklable config)
       |-- general   (GeneralSettings)                          run-wide
       |-- branches  (GBSettings, MBHSettings, ... dataclasses) per-branch
       +-- recipe    (Recipe of Stages, each of Moves)          run-wide
             each composed of atoms: transforms, priors, moves,
             injection tables

``general``, the branch blocks and ``recipe`` are **peers** — three
independently swappable attributes of the fit. Assigning one never replaces
another (``fit.general = ...`` leaves ``fit.gb`` and ``fit.recipe`` alone);
only containment nests, so swapping a block replaces the atoms inside it and
swapping the fit replaces all three. Every block is swapped by plain
assignment or a paired ``add_*``/``pop_*`` (or ``remove_*``) method.

The one real ordering is **build-time resolution**, not hierarchy:
:meth:`build` resolves ``general`` first (grid, ``Tobs``, domain, data); each
branch then inherits any *unset* ``Tobs``/``dt`` — ``EreborFit`` also fills
``domain_settings``/``log_dir`` — from the built general setup; the recipe is
materialized last, with every ``Move.branch`` validated against the
enabled branches. So general's run-wide choices flow *down* into the branches
as defaults, but that is defaulting, not ownership.

The recipe layer has **one concept per level, each with a required
``setup(ctx)`` hook run at materialization** (no separate "spec" classes):
:class:`~lisatools.globalfit.moves.globalfitmove.Move` (a move IS a
proposal; base = stock-name lookup, subclass+``setup(ctx)`` = custom,
:class:`~lisatools.globalfit.moves.functionmove.FunctionMove` = a plain
``fn(model, state)``), :class:`~lisatools.globalfit.recipe.Stage`, and the
unified :class:`~lisatools.globalfit.recipe.Recipe` — declarative before the
run, materializing itself into the runtime steps at run start. All of it is
cheap-construct / heavy-setup, so a configured fit pickles.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import typing
import warnings
from copy import deepcopy

import numpy as np
from eryn.prior import ProbDistContainer

from ..engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
    Settings,
    Setup,
)
from ..moves import FunctionMove, Move, MoveBuildContext
from ..recipe import Recipe, Stage
from ..run import GlobalFitSetup

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_ALIASES",
    "StockGlobalFit",
    "StockRegistry",
    "env_default",
    "env_is_set",
    "env_resolve",
]


# ---------------------------------------------------------------------------
# Environment-backed defaults
# ---------------------------------------------------------------------------

_BOOL_TRUE = ("1", "true", "yes", "on")
_BOOL_FALSE = ("0", "false", "no", "off")

#: NAMING RULE: an environment knob is the **capitalized attribute name** of the
#: field it seeds — ``general.data_mode`` -> ``DATA_MODE``,
#: ``general.num_iterations`` -> ``NUM_ITERATIONS``. Per-branch blocks add the
#: branch namespace as a prefix (``gb.min_freq`` -> ``GB_MIN_FREQ``) because the
#: bare name would collide with the general block's field of the same name.
#:
#: This table maps each canonical name to the legacy names it replaced. The
#: legacy spellings are still honored (with a DeprecationWarning) so existing
#: runbooks keep working: an unrecognized env var is *silently ignored*, so a
#: hard rename would quietly downgrade ``GF_NUM_ITER=1000`` to the default
#: rather than fail loudly. Prefer the canonical name in new code.
ENV_ALIASES: typing.Dict[str, typing.Tuple[str, ...]] = {
    "NUM_ITERATIONS": ("GF_NUM_ITER",),
    "DATA_MODE": ("DATA_PROCESSOR",),
    "MAKE_DIAGNOSTIC_PLOTS": ("MAKE_PLOTS",),
    "WAVELET_DURATION_MIN": ("WAVELET_DUR_MIN",),
    "WAVELET_DURATION_MAX": ("WAVELET_DUR_MAX",),
    "FIT_SGWB": ("ALL_SOURCES_SGWB",),
    "GB_USE_ASTROPHYSICAL_F0_MC_PRIOR": ("GB_F0MC_GMM_PRIOR",),
}


def _env_lookup(var: str) -> typing.Optional[str]:
    """Raw value for ``var``, falling back to its deprecated aliases.

    A present-but-empty variable counts as unset. Returns ``None`` when
    neither the canonical name nor any alias carries a value.
    """
    for name in (var, *ENV_ALIASES.get(var, ())):
        raw = os.environ.get(name)
        if raw is not None and raw.strip() != "":
            if name != var:
                warnings.warn(
                    f"Environment variable {name} is deprecated; use {var} instead "
                    "(the capitalized attribute name it seeds). The old name still "
                    "works for now.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            return raw
    return None


def env_resolve(var: str, default, cast=str):
    """Resolve ``default`` against the environment: env var wins over the hard default.

    ``var`` is the canonical name — the capitalized attribute name (see
    :data:`ENV_ALIASES`); deprecated aliases are still honored with a warning.
    ``cast`` converts the raw string; ``bool`` accepts 1/0, true/false,
    yes/no, on/off (case-insensitive). Explicit user kwargs always beat this
    (they never consult it).
    """
    raw = _env_lookup(var)
    if raw is None:
        return default
    if cast is bool:
        low = raw.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(f"Environment variable {var}={raw!r} is not a valid boolean.")
    return cast(raw)


def env_is_set(var: str) -> bool:
    """True when ``var`` (or one of its deprecated aliases) holds a non-empty value.

    Mirrors :func:`env_resolve`'s notion of "set" (a present-but-empty var
    counts as unset), so it reports exactly when ``env_resolve`` would return
    the environment's value rather than the hard default. Used by
    :meth:`StockGlobalFit.apply_lite` to let env vars overrule the lite preset.
    """
    return _env_lookup(var) is not None


def env_default(var: str, default, cast=str):
    """``dataclasses.field(default_factory=...)`` helper for env-backed defaults.

    Resolution order is *explicit kwarg > environment variable > hard
    default* — the factory only runs when the field is not passed
    explicitly, and it reads the environment at construction time.
    """

    def _factory():
        return env_resolve(var, default, cast)

    return _factory


def engine_ntemps_default():
    """``default_factory`` for the retired engine-level ``ntemps`` knob.

    Stock variants run the engine cold-chain only (each module tempers
    internally on its own per-branch ladder), so ``general.ntemps`` is
    pinned to 1. A set legacy ``NTEMPS`` environment variable raises here
    rather than being silently ignored — it would otherwise quietly
    downgrade a runbook that expects it to control tempering.
    """

    def _factory():
        if os.environ.get("NTEMPS") is not None:
            raise ValueError(
                "NTEMPS no longer controls stock global fits: the engine "
                "runs cold-chain only (ntemps=1) and each branch tempers on "
                "its own ladder. Set the per-branch knobs instead "
                "(GB_NTEMPS, VGB_NTEMPS, MBH_NTEMPS, EMRI_NTEMPS, "
                "SOBBH_NTEMPS, PSD_NTEMPS), or unset NTEMPS. "
                "(erebor.blank keeps NTEMPS for simple-API branches, which "
                "temper on the engine ladder.)"
            )
        return 1

    return _factory


# ---------------------------------------------------------------------------
# The deferred-build fit object
# ---------------------------------------------------------------------------

# GlobalFitSetup attributes that only exist after build(); accessing
# them earlier gets a helpful error instead of a bare AttributeError.
_BUILT_ONLY_ATTRS = ("settings_dict", "current_info", "backend")


@dataclasses.dataclass
class _InfoBranchSettings(Settings):
    """Internal Settings for branches appended with plain branch info.

    Not user-facing — :meth:`StockGlobalFit.add_branch` builds one under the
    hood from the ``ndim``/``nleaves_max``/``priors``/... kwargs. Adds the
    ``injection`` field the run's generic branch seeding reads (start walkers
    scattered around it; ``<BRANCH>_START_FACTOR=0`` starts exactly there).
    """

    injection: typing.Optional[typing.Any] = None


class StockGlobalFit(GlobalFitSetup):
    """Base class for stock (configured-but-unbuilt) global fits.

    Subclasses fill in the default building blocks; ``__init__`` stays cheap
    (no data loads, no directories, no waveform or HDF-backend
    construction) and the object stays picklable until :meth:`build` runs.
    After :meth:`build`, the instance IS the :class:`GlobalFitSetup`
    that :class:`~lisatools.globalfit.run.GlobalFit` consumes.

    Subclass contract — override:

    - ``option_name`` / ``description`` class attributes (registry entry);
    - :meth:`default_general` returning the variant's
      :class:`~lisatools.globalfit.engine.GeneralSettings` (subclass)
      instance;
    - :meth:`default_branches` returning an ordered ``{name: Settings}``
      dict;
    - :meth:`default_recipe` returning the variant's
      :class:`~lisatools.globalfit.recipe.Recipe`;
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
        "verbose",
    )

    def __init__(
        self,
        *,
        general: typing.Optional[GeneralSettings] = None,
        branches: typing.Optional[typing.Dict[str, Settings]] = None,
        recipe: typing.Optional[Recipe] = None,
        setup_function: typing.Optional[typing.Callable] = None,
        # head_rank is a legacy alias from the retired multi-stage pipeline;
        # it defaults to the main rank (no separate role). GlobalFit assigns
        # the dedicated saver rank automatically at np >= 3.
        head_rank: int = 0,
        main_rank: int = 0,
        **knobs,
    ):
        # NOTE: deliberately does NOT call GlobalFitSetup.__init__ —
        # that is the heavy stage, deferred to build().
        self.general = general if general is not None else self.default_general()
        self._branch_names: typing.List[str] = []
        # Branches appended with plain branch info (must be targeted by >= 1
        # move — validated at recipe materialization).
        self._info_branches: typing.List[str] = []
        # Per-branch Setup-class overrides from add_branch(setup_class=...).
        self._setup_class_overrides: typing.Dict[str, type] = {}
        # Live GlobalFit runner while sample() is active (mid-run adds hook
        # into it; branch appends are forbidden then).
        self._runner = None
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

    def default_recipe(self) -> Recipe:
        raise NotImplementedError

    # -- knob application ------------------------------------------------------

    def _apply_knobs(self, knobs: dict):
        """Apply constructor kwargs: headline knobs and variant attributes.

        ``lite=True`` applies the variant's laptop-smoke preset FIRST, but
        :meth:`apply_lite` skips any preset path whose env var is set (env
        vars overrule the preset), and every explicit kwarg is applied AFTER
        the preset (kwargs win over both). Precedence: explicit kwarg > env
        var > lite preset > class default.
        """
        if knobs.pop("lite", False):
            self.apply_lite()
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

    # -- lite (laptop-smoke) preset ---------------------------------------------

    def lite_overrides(self) -> dict:
        """Dotted-path -> value overrides for the variant's ``lite`` preset.

        Applied by ``lite=True`` on any stock call and by the registered
        ``*_lite`` twins — one table shared by both spellings. Empty in the
        base; variants provide it (Erebor's live in
        ``stock/erebor/variants/lite.py``). Precedence when applied:
        explicit kwarg > env var > lite preset > class default (a set env var
        overrules the preset for its path — see :meth:`lite_env_vars`).
        """
        return {}

    def lite_env_vars(self) -> dict:
        """Dotted-path -> environment-variable name for the ``lite`` preset.

        Consulted by :meth:`apply_lite`: when a preset path's env var is set,
        the env-resolved value (already on the field from construction) is
        kept and the lite override for that path is skipped, so env vars
        overrule the preset. Paths absent from this map are always overridden
        by lite (they are not env-backed). Empty in the base; variants provide
        it alongside :meth:`lite_overrides`.
        """
        return {}

    def apply_lite(self) -> None:
        """Apply :meth:`lite_overrides` to this fit (laptop-smoke preset).

        Dotted paths resolve against the fit (``"general.num_iterations"``,
        ``"gb.num_repeat_proposals"``, ...); paths whose parent object is
        missing (e.g. a removed branch) are skipped silently. A path whose
        :meth:`lite_env_vars` env var is set is also skipped, leaving the
        env-resolved value in place (env vars overrule the preset).
        """
        overrides = self.lite_overrides()
        if not overrides:
            raise ValueError(
                f"{type(self).__name__} has no lite preset (lite_overrides() "
                "is empty) — override lite_overrides() on the variant."
            )
        env_vars = self.lite_env_vars()
        for path, value in overrides.items():
            # Env var overrules the lite preset: if the field's env var is
            # set, construction already put the env value on the field — keep
            # it and skip the preset override for this path.
            var = env_vars.get(path)
            if var is not None and env_is_set(var):
                continue
            obj = self
            *parents, leaf = path.split(".")
            try:
                for parent in parents:
                    obj = getattr(obj, parent)
            except AttributeError:
                continue
            if not hasattr(obj, leaf):
                continue
            setattr(obj, leaf, deepcopy(value))

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
        if name in _BUILT_ONLY_ATTRS or hasattr(GlobalFitSetup, name):
            # Either a build-product attribute, or a GlobalFitSetup
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

    def add_branch(
        self,
        name: str,
        settings: typing.Optional[Settings] = None,
        *,
        ndim: typing.Optional[int] = None,
        nleaves_max: typing.Optional[int] = None,
        nleaves_min: typing.Optional[int] = None,
        priors=None,
        moves: typing.Optional[typing.Sequence] = None,
        stage: typing.Optional[str] = None,
        injection=None,
        signal_gen: typing.Optional[typing.Callable] = None,
        sub_state=None,
        sub_backend=None,
        setup_class: typing.Optional[type] = None,
        **branch_kwargs,
    ):
        """Add (or replace) a branch block. ``fit.<name>`` becomes the settings.

        Two paths:

        * **Stock path** — pass a ``Settings`` dataclass instance (the
          pre-installed mechanism; the variant's ``setup_classes`` builds it).
        * **Simple path** — pass plain branch info instead: ``ndim`` +
          ``priors`` (+ optional ``nleaves_max``/``nleaves_min``/``injection``/
          ``signal_gen``). The branch is stored in the main HDF file
          (coords/inds/log_like) with no Settings/Setup subclassing, and
          **must be targeted by at least one move** (``branch=name``) by run
          start — pass ``moves=[...]`` here (each entry is anything
          :meth:`add_move` accepts; they land in the recipe wrapped in the
          basic single-stage recipe, ``stage=`` disambiguating when several
          stages exist) or add them separately.

        ``sub_state``/``sub_backend`` optionally register a per-branch state /
        HDF sub-backend extension (the same hook stock branches use via their
        Settings ``branch_state``/``branch_backend`` fields). ``setup_class``
        overrides the Setup class used at build for this branch.

        Timing: works before build (primary) and after build but before a run
        — a post-build append incrementally builds just this branch against
        the cached general setup, so the heavy data load is not redone.
        Mid-run appends are not supported.
        """
        if getattr(self, "_runner", None) is not None:
            raise RuntimeError(
                "add_branch is not supported mid-run: the HDF backend and state "
                "shapes are fixed at run start. Add branches before run()/sample()."
            )
        if settings is None:
            if ndim is None or priors is None:
                raise TypeError(
                    f"Branch {name!r}: pass a Settings dataclass instance, or the "
                    "branch info (at least ndim= and priors=)."
                )
            if nleaves_max is None:
                nleaves_max = 1
            if nleaves_min is None:
                # Fixed-leaf by default: the run's generic seeding turns a
                # nleaves_min == nleaves_max branch always-on.
                nleaves_min = nleaves_max
            settings = _InfoBranchSettings(
                ndim=ndim,
                nleaves_max=nleaves_max,
                nleaves_min=nleaves_min,
                priors=self._coerce_priors(name, priors),
                injection=injection,
                signal_gen=signal_gen,
                **branch_kwargs,
            )
            if name not in self._info_branches:
                self._info_branches.append(name)
        else:
            if not isinstance(settings, Settings):
                raise TypeError(
                    f"Branch {name!r}: expected a Settings dataclass instance, got "
                    f"{type(settings).__name__}."
                )
            if branch_kwargs:
                raise TypeError(
                    f"Branch {name!r}: branch-info kwargs {sorted(branch_kwargs)} only "
                    "apply when no Settings instance is passed."
                )
        if sub_state is not None:
            settings.branch_state = sub_state
        if sub_backend is not None:
            settings.branch_backend = sub_backend
        if setup_class is not None:
            self._setup_class_overrides[name] = setup_class

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

        if self.built:
            # Incremental post-build append: build just this branch against the
            # cached general setup and patch it into the built products (the
            # heavy data load is preserved). engine_info derives from
            # source_info, so it picks the branch up automatically.
            general_setup = self.build_general(force=False)
            setup = self.make_source_setup(
                name, self.prepare_branch_settings(name, general_setup), general_setup
            )
            self._source_info[name] = setup  # also settings_dict.source_info
            self.current_info.source_info[name] = deepcopy(setup)

        for mv in moves or ():
            self.add_move(mv, stage=stage, branch=name)

    def _coerce_priors(self, name: str, priors) -> typing.Dict[str, ProbDistContainer]:
        """Coerce the simple-path ``priors`` input to the ``{branch: container}`` form.

        Accepts a :class:`~eryn.prior.ProbDistContainer`, a ``{param_index:
        distribution}`` dict (wrapped in one), or the already-keyed
        ``{branch_name: container}`` dict.
        """
        if isinstance(priors, ProbDistContainer):
            return {name: priors}
        if isinstance(priors, dict):
            if priors and all(isinstance(k, int) for k in priors):
                return {name: ProbDistContainer(priors)}
            return priors
        raise TypeError(
            f"Branch {name!r}: priors must be a ProbDistContainer or a "
            f"{{param_index: distribution}} dict, got {type(priors).__name__}."
        )

    def remove_branch(self, name: str) -> Settings:
        """Remove a branch block, and the recipe moves that sample it.

        Composing a branch out is one call: the moves declared against it go
        with it (a left-behind ``Move(..., branch=name)`` has no stock move to
        resolve at materialization, so it would raise a KeyError there). A
        stage emptied by the removal is dropped too. Re-adding the branch with
        :meth:`add_branch` does NOT restore its moves — pass them via
        ``add_branch(..., moves=[...])``.
        """
        if name not in self._branch_names:
            raise KeyError(
                f"Unknown branch {name!r}. Present branches: {self._branch_names}."
            )
        self._branch_names.remove(name)
        settings = self.__dict__.pop(name)

        dropped = [
            mv.name
            for st in self.recipe.stages
            for mv in st.moves
            if getattr(mv, "branch", None) == name
        ]
        for mv_name in dropped:
            self.recipe.pop_move(mv_name)
        empty = [st.name for st in self.recipe.stages if not st.moves]
        for st_name in empty:
            self.recipe.pop_stage(st_name)
        if dropped:
            logger.info(
                "remove_branch(%r) also dropped move(s) %s%s.",
                name,
                ", ".join(repr(m) for m in dropped),
                f" and now-empty stage(s) {', '.join(repr(s) for s in empty)}"
                if empty
                else "",
            )
        return settings

    # convenience aliases matching the recipe layer's add/pop vocabulary
    def add_move(self, move, **kwargs) -> Move:
        """Add a move (a.k.a. proposal) to the fit — the single entrance.

        ``move`` is a :class:`~lisatools.globalfit.moves.globalfitmove.Move`,
        a stock-move name, a constructed eryn move, or a plain
        ``fn(model, state) -> (new_state, accepted)`` function (wrapped in a
        :class:`~lisatools.globalfit.moves.functionmove.FunctionMove` that
        only needs to touch ``model.analysis_container_arr``). Works before
        build, after build, and mid-run (a live add starts firing on the next
        iteration). See :meth:`Recipe.add_move
        <lisatools.globalfit.recipe.Recipe.add_move>` for placement kwargs
        (``stage=`` required when several stages exist).
        """
        mv = self.recipe.add_move(move, **kwargs)
        self._materialize_live(mv)
        return mv

    def pop_move(self, *args, **kwargs):
        return self.recipe.pop_move(*args, **kwargs)

    def _materialize_live(self, move: Move) -> None:
        """Materialize a just-added move into a live run (no-op pre-run).

        Mid-run (``sample()`` active), the move's ``setup(ctx)`` runs with the
        live context and the runtime move is appended into its stage's live
        ``GFCombineMove`` — it proposes from the next iteration on. A whole
        new stage is materialized and registered on the running recipe
        (stages before the currently running one cannot be inserted mid-run).
        """
        runner = getattr(self, "_runner", None)
        ctx = getattr(runner, "live_ctx", None) if runner is not None else None
        if ctx is None:
            return  # pre-run: declarative only, materialized at run start
        stage, _ = self.recipe._find_move(move.name)
        try:
            step = self.recipe.get_step(stage.name)
        except KeyError:
            # The whole stage is new mid-run: materialize + register it (the
            # runtime Recipe scans forward, so it runs when reached).
            self.recipe.add_recipe_component(stage.setup(ctx), name=stage.name)
            return
        combined = step.moves[0]
        runtime = move.materialize(ctx)
        if getattr(runtime, "accepted", None) is None:
            runtime.accepted = np.zeros((ctx.ntemps, ctx.nwalkers))
        if not hasattr(runtime, "progress"):
            runtime.progress = bool(
                getattr(getattr(ctx.curr, "general_info", None), "verbose", False)
            )
        sampler = getattr(runner, "sampler", None)
        if (
            sampler is not None
            and sampler.periodic is not None
            and getattr(runtime, "periodic", None) is None
        ):
            runtime.periodic = sampler.periodic
        if (
            getattr(combined, "share_temperature_control", False)
            and getattr(combined, "temperature_control", None) is not None
            and getattr(runtime, "temperature_control", None) is None
        ):
            runtime.temperature_control = combined.temperature_control
        combined.moves.append(runtime)

    def set_move_debug(self, *args, **kwargs):
        return self.recipe.set_move_debug(*args, **kwargs)

    def set_stage_debug(self, *args, **kwargs):
        return self.recipe.set_stage_debug(*args, **kwargs)

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
        """Turn one branch Settings block into its Setup.

        Resolution: an :meth:`add_branch` ``setup_class=`` override wins, then
        the variant's ``setup_classes`` registry; a branch with neither falls
        back to the plain generic :class:`~lisatools.globalfit.engine.Setup`
        when its info is complete (``ndim``/``priors`` set) — the simple
        branch-append path.
        """
        override = getattr(self, "_setup_class_overrides", {}).get(name)
        if override is not None:
            return override(settings)
        try:
            setup_cls = self.setup_classes[name]
        except KeyError:
            if settings.ndim is not None and settings.priors is not None:
                return Setup(settings)
            raise KeyError(
                f"No Setup class registered for branch {name!r} on "
                f"{type(self).__name__}.setup_classes ({sorted(self.setup_classes)}), "
                "and the branch info is incomplete (needs at least ndim and priors "
                "for the generic Setup). Register a class, pass "
                "add_branch(setup_class=...), or override make_source_setup."
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
        """Run the full heavy build and become a usable GlobalFitSetup."""
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
        # The recipe rides along so the run can materialize it.
        settings.source_metadata["recipe"] = self.recipe
        GlobalFitSetup.__init__(self, settings)
        # GlobalFitSetup.__init__ deepcopies the settings into current_info,
        # which would freeze the recipe at build time. Re-point the copy at
        # the LIVE recipe object so post-build edits (add_move / add_branch)
        # reach the run.
        self.current_info.source_metadata["recipe"] = self.recipe
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

    def sample(self, iterations=None, **sample_kwargs):
        """Generator run mode: build (if needed), then yield ``(model, state)`` per iteration.

        The emcee-style loop, one level up from the engine's own ``sample``::

            for model, state in fit.sample(iterations=100):
                ...   # inspect/mutate model.analysis_container_arr and state
                      # in place; the next iteration continues from them

        Single-process (no MPI rank roles; ``run()`` owns MPI) — inside the
        loop you may do whatever you need, including your own MPI, as long as
        control returns synchronously. Moves added mid-loop via
        :meth:`add_move` start firing on the next iteration. See
        :meth:`GlobalFit.sample <lisatools.globalfit.run.GlobalFit.sample>`
        for the keyword arguments and the backend-saving cadence.
        """
        self.build()
        from ..run import GlobalFit

        gf = GlobalFit(self, comm=None)
        self._runner = gf
        try:
            yield from gf.sample(iterations=iterations, **sample_kwargs)
        finally:
            self._runner = None

    # -- introspection / cloning / pickling ---------------------------------------

    @staticmethod
    def _describe_value(value, maxlen: int = 100) -> str:
        """Concise, tutorial-friendly repr of a settings value.

        Arrays collapse to shape/dtype, classes/callables to their name, and
        long reprs are truncated — so ``describe(full=True)`` stays readable
        even with domain factories, priors, and injection tables present.
        """
        shape = getattr(value, "shape", None)
        if shape is not None and hasattr(value, "dtype"):
            return f"<{type(value).__name__} shape={tuple(shape)} dtype={value.dtype}>"
        if isinstance(value, type):
            return value.__name__
        if callable(value):
            return f"<callable {getattr(value, '__name__', None) or type(value).__name__}>"
        r = repr(value)
        return r if len(r) <= maxlen else r[: maxlen - 1] + "…"

    def all_settings(self) -> dict:
        """Every configured setting as a nested dict (general + branches + recipe).

        Programmatic companion to ``describe(full=True)``::

            {"option_name": ..., "headline": {knob: value, ...},
             "general": {field: value, ...},
             "branches": {name: {field: value, ...}, ...},
             "recipe": [{"stage": ..., "kind": ..., "moves": [...]}, ...],
             "setup_function": name}

        Values are the live configured objects (use ``describe(full=True)``
        for a printable illustration). Reflects the current config only —
        build-time derived fields (``Tobs``, ``domain_settings``, ...) are
        resolved later, in :meth:`build`.
        """
        general = {f.name: getattr(self.general, f.name) for f in dataclasses.fields(self.general)}
        branches = {
            name: {f.name: getattr(blk, f.name) for f in dataclasses.fields(blk)}
            for name, blk in self.branches.items()
        }
        return {
            "option_name": self.option_name,
            "headline": {k: getattr(self.general, k, None) for k in self._HEADLINE_KNOBS},
            "general": general,
            "branches": branches,
            "recipe": [
                {"stage": st.name, "kind": st.kind, "moves": st.move_names()}
                for st in self.recipe.stages
            ],
            "setup_function": getattr(self.setup_function, "__name__", None),
        }

    def describe(self, full: bool = False) -> str:
        """Multi-line summary of the configured blocks and headline knobs.

        ``full=False`` (default) shows the headline knobs, branch types, and
        recipe. ``full=True`` illustrates EVERY setting: all fields of the
        general block and of each branch block (not just the headline knobs).
        Build-time derived fields (``Tobs``, ``domain_settings``, ...) show
        their pre-build value (often ``None``); build the fit to resolve them.

        Once the fit is built, each branch also reports its resolved product
        (the ``*Setup`` it materialized into, and its sub-band count where the
        branch has one), so this doubles as the post-build inspection view.
        For what the *sampler* produced, see
        :meth:`~lisatools.globalfit.run.GlobalFitSetup.summarize_run`.
        """
        lines = [
            f"{type(self).__name__} ({self.option_name or 'unregistered'}) — "
            f"{self.description or 'no description'}",
            f"  built: {self.built}",
        ]
        if full:
            lines.append(f"  general ({type(self.general).__name__}):")
            for f in dataclasses.fields(self.general):
                lines.append(
                    f"    {f.name} = {self._describe_value(getattr(self.general, f.name))}"
                )
        else:
            lines.append("  headline:")
            for key in self._HEADLINE_KNOBS:
                lines.append(f"    {key} = {getattr(self.general, key, None)!r}")
        lines.append(f"  branches: {self._branch_names}")
        # Once built, each branch's resolved Setup (and its sub-band count where
        # it has one) rides alongside the config block it came from.
        built_info = self.source_info if self.built else {}
        for name in self._branch_names:
            blk = getattr(self, name)
            setup = built_info.get(name)
            extra = ""
            if setup is not None:
                band_edges = getattr(setup, "band_edges", None)
                bands = "" if band_edges is None else f", {len(band_edges) - 1} sub-bands"
                extra = f"  [built: {type(setup).__name__}{bands}]"
            lines.append(f"    {name}: {type(blk).__name__}{extra}")
            if full:
                for f in dataclasses.fields(blk):
                    lines.append(
                        f"      {f.name} = {self._describe_value(getattr(blk, f.name))}"
                    )
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
        for attr in ("_general_setup", "_source_info", "_runner", *_BUILT_ONLY_ATTRS):
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
        """List of ``(option_name, description)`` pairs, sorted by option name."""
        return [(name, self._classes[name].description) for name in sorted(self._classes)]

    def names(self) -> typing.List[str]:
        """Registered option names, alphabetically sorted."""
        return sorted(self._classes)

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
