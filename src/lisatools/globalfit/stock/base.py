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

Main entry points: :class:`StockGlobalFit` (the deferred-build base every
variant subclasses), :class:`StockRegistry` (a family's name -> class table,
behind ``erebor.get_stock`` / ``erebor.get_stock_options``), and the
env-default helpers :func:`env_default`, :func:`env_resolve`,
:func:`env_is_set` and :func:`bool_or_str` with the :data:`ENV_ALIASES`
rename table. Every ``*Settings`` field default of the stock families
resolves through :func:`env_default`: *explicit kwarg > env var > lite
preset > hard default*, the env knob being the capitalized field name with
the branch prefix (``gb.min_freq`` -> ``GB_MIN_FREQ``).

In production (the 9mo / 12mo launchers
``scripts/fstat_proposal/submit_gf_9mo_v9.sh`` and
``scripts/flagship/submit_gf_12mo.sh``, driver
``scripts/fstat_proposal/run_combined_staged.py``) the driver clones
``erebor.all_sources(nwalkers=4)`` (:meth:`StockGlobalFit.__call__`), edits
``fit.general``, replaces ``fit.recipe`` wholesale, pins the rank layout,
then calls :meth:`StockGlobalFit.build` and :meth:`StockGlobalFit.run`.

Example:
    .. code-block:: python

        from lisatools.globalfit.stock import erebor

        fit = erebor.gb_no_fg(nwalkers=4)   # cheap: no data load yet
        fit.gb.min_freq = 9.8e-3            # plain attribute access on a block
        print(fit.describe())
        fit.build()                          # heavy: data, domain, branch Setups
        fit.run()                            # MPI run (GlobalFit.run_global_fit)
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
    (they never consult it). A present-but-empty variable counts as unset.

    Args:
        var: Canonical environment-variable name (e.g. ``"GB_NTEMPS"``).
        default: Value returned when neither ``var`` nor a deprecated alias
            is set. Returned as-is (not passed through ``cast``).
        cast (callable, optional): Converter applied to the raw string.
            Default ``str``. ``bool`` is special-cased (see above); any other
            callable is called as ``cast(raw)``.

    Returns:
        object: The cast environment value, or ``default`` when unset.

    Raises:
        ValueError: When ``cast is bool`` and the value is not one of the
            accepted spellings, or when ``cast`` itself raises on the raw
            string.

    Example:
        >>> import os
        >>> from lisatools.globalfit.stock.base import env_resolve
        >>> os.environ["DEMO_STOCK_FLAG"] = "on"
        >>> env_resolve("DEMO_STOCK_FLAG", False, bool)
        True
        >>> env_resolve("DEMO_STOCK_UNSET_KNOB", 3, int)
        3
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

    Args:
        var: Canonical environment-variable name.

    Returns:
        bool: ``True`` when ``var`` or one of its :data:`ENV_ALIASES` legacy
        names holds a non-empty value.
    """
    return _env_lookup(var) is not None


def bool_or_str(raw: str):
    """``cast`` for env-backed ``Union[bool, str]`` fields.

    A boolean spelling (1/0, true/false, yes/no, on/off) becomes the bool;
    anything else is passed through as the stripped string. Used by knobs
    that take either a flag or a named mode — e.g.
    ``add_instrument_noise`` (``ADD_INSTRUMENT_NOISE``), where ``True`` means
    "auto-resolve" and ``"mojito"``/``"synthetic"`` force a specific source.

    Module level and named (never a lambda) so the settings tree stays
    pickle/deepcopy-safe.

    Args:
        raw: The raw environment string.

    Returns:
        bool | str: ``True`` / ``False`` for a boolean spelling, else
        ``raw.strip()``.
    """
    low = raw.strip().lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    return raw.strip()


def env_default(var: str, default, cast=str):
    """``dataclasses.field(default_factory=...)`` helper for env-backed defaults.

    Resolution order is *explicit kwarg > environment variable > hard
    default* — the factory only runs when the field is not passed
    explicitly, and it reads the environment at construction time (so an env
    var set after a fit was constructed does not reach that fit; clone or
    re-construct it). A ``lite=True`` preset sits between the env var and
    the hard default (see :meth:`StockGlobalFit.apply_lite`).

    Args:
        var: Canonical environment-variable name — the capitalized
            field name, prefixed with the branch for per-branch blocks.
        default: Hard default used when the variable is unset.
        cast (callable, optional): Converter for the raw string (see
            :func:`env_resolve`). Default ``str``.

    Returns:
        callable: A zero-argument factory for ``dataclasses.field(default_factory=...)``.

    Example:
        .. code-block:: python

            import dataclasses
            from lisatools.globalfit.engine import Settings
            from lisatools.globalfit.stock.base import env_default

            @dataclasses.dataclass
            class MyBranchSettings(Settings):
                # env knob MY_NTEMPS (= the capitalized, prefixed field name)
                ntemps: int = dataclasses.field(
                    default_factory=env_default("MY_NTEMPS", 8, int)
                )
    """

    def _factory():
        return env_resolve(var, default, cast)

    return _factory


def ten_to_the_x(x):
    """Return ``10 ** x`` (named 1-arg transform for pickling support).

    Used as a per-parameter :class:`~eryn.utils.TransformContainer` transform
    for PSD / foreground amplitudes and frequency scales sampled in ``log10``
    space; ``np.log10`` is its inverse and is already a picklable ufunc.

    Lives here rather than in ``erebor/transforms.py`` (which re-exports it)
    because the noise branch needs it and that module pulls in ``bbhx`` at
    import — see
    :func:`~lisatools.globalfit.stock.erebor.noise.make_galfor_log_transform_container`.

    Args:
        x (float | np.ndarray): Exponent(s).

    Returns:
        float | np.ndarray: ``10.0 ** x``, same shape as ``x``.
    """
    return 10.0**x


def engine_ntemps_default():
    """``default_factory`` for the retired engine-level ``ntemps`` knob.

    Stock variants run the engine cold-chain only (each module tempers
    internally on its own per-branch ladder), so ``general.ntemps`` is
    pinned to 1. A set legacy ``NTEMPS`` environment variable raises here
    rather than being silently ignored — it would otherwise quietly
    downgrade a runbook that expects it to control tempering. The
    per-branch ladder sizes are the branch fields ``ntemps`` (env
    ``GB_NTEMPS``, ``VGB_NTEMPS``, ``MBH_NTEMPS``, ``EMRI_NTEMPS``,
    ``SOBBH_NTEMPS``, ``PSD_NTEMPS``, ``GALFOR_NTEMPS``, ``SGWB_NTEMPS``).

    Returns:
        callable: A zero-argument factory returning ``1``.

    Raises:
        ValueError: From the factory (i.e. at settings construction) when the
            ``NTEMPS`` environment variable is present (any value, even
            empty).
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

# Per-process multi-rank wiring stamped by communication.ranks.prepare_rank
# (GlobalFitSetup declares all four with a ``None`` class default): dropped by
# reset_build, since another process's layout must never survive a reset.
_RANK_ATTRS = ("rank_layout", "rank", "rank_device_mode", "fanout")


@dataclasses.dataclass
class _InfoBranchSettings(Settings):
    """Internal Settings for branches appended with plain branch info.

    Not user-facing — :meth:`StockGlobalFit.add_branch` builds one under the
    hood from the ``ndim``/``nleaves_max``/``priors``/... kwargs. Adds the
    ``injection`` field the run's generic branch seeding reads (start walkers
    scattered around it; ``<BRANCH>_START_FACTOR=0`` starts exactly there).

    Attributes:
        injection: Optional sampling-basis start/truth rows for the branch
            (``None`` -> walkers start from the prior).
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

    Users rarely construct a variant class directly: the family module holds
    one cheap default instance per registered variant, and calling it clones
    the configuration (``erebor.all_sources(nwalkers=4)``, see
    :meth:`__call__`); ``erebor.get_stock(name, **knobs)`` constructs a fresh
    instance from the class instead.

    Args:
        general: The run-wide block. Default ``None`` ->
            :meth:`default_general`.
        branches: Ordered ``{name: Settings}`` branch blocks, each added with
            :meth:`add_branch`. Default ``None`` -> :meth:`default_branches`.
        recipe: The declarative recipe. Default ``None`` ->
            :meth:`default_recipe`.
        setup_function: Named module-level function that materializes the
            recipe at run start (the variant's ``setup_recipe``). Default
            ``None`` -> the class attribute ``default_setup_function``.
        head_rank: Legacy alias of the retired multi-stage pipeline's head
            rank. Default ``0``; ``GlobalFit`` assigns the saver rank itself
            at ``np >= 3``.
        main_rank: Main (head) MPI rank. Default ``0``.
        **knobs: Applied after the blocks by :meth:`_apply_knobs`, in this
            order of lookup: a headline knob (``_HEADLINE_KNOBS``, delegated to
            ``self.general``), a class or instance attribute of the variant,
            then any field of ``self.general``. ``lite=True`` applies the
            variant's laptop-smoke preset first (:meth:`apply_lite`).

    Raises:
        TypeError: When a ``**knobs`` key matches none of the above.
        ValueError: When a branch name clashes with a fit attribute, or
            ``lite=True`` on a variant without a preset.

    Attributes:
        general: The run-wide :class:`~lisatools.globalfit.engine.GeneralSettings`
            block (Erebor: :class:`~lisatools.globalfit.stock.erebor.fit.EreborGeneralSettings`
            or a variant subclass). Its env-backed fields resolve at construction.
        recipe: The :class:`~lisatools.globalfit.recipe.Recipe` (stages of moves).
            The production driver replaces it wholesale before ``build()``.
        setup_function: Recipe-materialization function run at run start.
        head_rank: See Args.
        main_rank: See Args.
        option_name: Class attribute; registry name (``"all_sources"``, ...).
        description: Class attribute; one-line registry description.
        setup_classes: Class attribute; ``{branch name: Setup class}`` used by
            :meth:`make_source_setup`.
        default_setup_function: Class attribute; default recipe setup function.

    Note:
        Each branch block is also an attribute named after the branch
        (``fit.gb``, ``fit.mbh``, ...; see :attr:`branches`), and every
        headline knob reads and writes through to ``fit.general``
        (``fit.nwalkers`` is ``fit.general.nwalkers``). After :meth:`build`
        the instance carries the :class:`~lisatools.globalfit.run.GlobalFitSetup`
        products (``settings_dict``, ``current_info``, ``backend``, ...);
        reading them before the build raises an ``AttributeError`` that says
        to build first.

    Example:
        .. code-block:: python

            from lisatools.globalfit.stock import erebor

            def nudge(model, state):              # a move IS a proposal
                aca = model.analysis_container_arr  # read / adjust the residual
                return state, None                # (new_state, accepted)

            erebor.get_stock_options()            # [(name, description), ...]
            fit = erebor.gb_no_fg(nwalkers=4)     # clone of the module default
            fit.gb.center_freq = 8e-3             # per-branch field
            fit.general.num_iterations = 100      # general field
            print(fit.list_moves())               # [pe] gb_pe: rj_fstat_pe ...
            fit.add_move(nudge, branch="gb")      # one stage -> lands in gb_pe
            fit.recipe.pop_move("nudge")          # and out again
            fit.build()                           # heavy stage, on command
            fit.run()                             # MPI: GlobalFit.run_global_fit
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
        """Return the variant's default general block (subclasses override; base raises)."""
        raise NotImplementedError

    def default_branches(self) -> typing.Dict[str, Settings]:
        """Return the variant's ordered ``{name: Settings}`` blocks (subclasses override)."""
        raise NotImplementedError

    def default_recipe(self) -> Recipe:
        """Return the variant's default :class:`~lisatools.globalfit.recipe.Recipe` (override)."""
        raise NotImplementedError

    # -- knob application ------------------------------------------------------

    def _apply_knobs(self, knobs: dict):
        """Apply constructor kwargs: headline knobs and variant attributes.

        ``lite=True`` applies the variant's laptop-smoke preset FIRST, but
        :meth:`apply_lite` skips any preset path whose env var is set (env
        vars overrule the preset), and every explicit kwarg is applied AFTER
        the preset (kwargs win over both). Precedence: explicit kwarg > env
        var > lite preset > class default.

        Args:
            knobs: Constructor / clone kwargs; ``"lite"`` is popped.

        Raises:
            TypeError: When a key is neither a headline knob, a variant
                attribute, nor a field of ``self.general``.
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

        Returns:
            dict: Map of dotted attribute paths to values (empty in the
            base); ``gb_no_fg``'s preset includes, e.g.,
            ``"general.tobs_target": 14 * 86400.0`` and
            ``"gb.num_repeat_proposals": 2``.
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

        Returns:
            dict: Dotted path -> canonical env-var name (empty in the base).
        """
        return {}

    def apply_lite(self) -> None:
        """Apply :meth:`lite_overrides` to this fit (laptop-smoke preset).

        Dotted paths resolve against the fit (``"general.num_iterations"``,
        ``"gb.num_repeat_proposals"``, ...); paths whose parent object is
        missing (e.g. a removed branch) are skipped silently. A path whose
        :meth:`lite_env_vars` env var is set is also skipped, leaving the
        env-resolved value in place (env vars overrule the preset). Values are
        deep-copied onto the fit.

        Raises:
            ValueError: When the variant has no preset (empty
                :meth:`lite_overrides`).

        Example:
            .. code-block:: python

                fit = erebor.gb_no_fg(lite=True)   # same as erebor.gb_no_fg_lite()
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

        Args:
            name: Branch name; becomes the attribute ``fit.<name>`` and the
                key in every state/backend. Must not clash with a fit
                attribute or headline knob.
            settings: Stock path: a :class:`~lisatools.globalfit.engine.Settings`
                dataclass instance. ``None`` selects the simple path.
            ndim: Simple path: sampled dimension (required).
            nleaves_max: Simple path: maximum leaves. Default ``None`` -> 1.
            nleaves_min: Simple path: minimum leaves. Default ``None`` ->
                ``nleaves_max`` (fixed-dimensional, always-on).
            priors (ProbDistContainer | dict): Simple path (required): a
                :class:`~eryn.prior.ProbDistContainer`, a ``{param_index:
                distribution}`` dict, or an already-keyed ``{name: container}``
                dict (see :meth:`_coerce_priors`).
            moves: Moves to add for this branch (each anything
                :meth:`add_move` accepts), placed with ``branch=name``.
            stage: Stage for ``moves`` (required when several stages exist).
            injection (np.ndarray, optional): Simple path: sampling-basis
                start rows (walkers start scattered around them;
                ``<BRANCH>_START_FACTOR=0`` exactly).
            signal_gen: Simple path: per-leaf template generator
                ``fn(*sampling_params) -> DomainBase`` registered into every
                analysis container, so the engine builds/subtracts this
                branch's templates itself.
            sub_state (type, optional): Per-branch sub-state class (sets
                ``settings.branch_state``).
            sub_backend (type, optional): Per-branch HDF sub-backend class
                (sets ``settings.branch_backend``).
            setup_class: Optional Setup class overriding the variant's
                ``setup_classes`` entry for this branch at build.
            **branch_kwargs: Simple path only: extra ``Settings`` fields
                (``transform``, ``periodic``, ``betas``, ...).

        Raises:
            RuntimeError: When called while :meth:`sample` is running.
            TypeError: When neither a Settings instance nor ``ndim`` +
                ``priors`` is given, ``settings`` is not a ``Settings``, or
                ``branch_kwargs`` are passed alongside a Settings instance.
            ValueError: When ``name`` clashes with an existing fit attribute.

        Example:
            .. code-block:: python

                from eryn.prior import uniform_dist

                fit = erebor.blank(nwalkers=4, ntemps=2)
                fit.add_branch(
                    "line", ndim=2,
                    priors={0: uniform_dist(0.0, 1.0), 1: uniform_dist(-1.0, 1.0)},
                    signal_gen=my_line_template,   # fn(*params) -> DomainBase
                    moves=[my_line_move],          # fn(model, state)
                )
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
        ``{branch_name: container}`` dict (returned unchanged).

        Raises:
            TypeError: For any other type.
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

        Only the configured fit is edited (the build products are not); call
        it before :meth:`build`. The production driver calls it only for an
        unarmed source branch (empty ``MBHB_IDS`` / ``EMRI_IDS`` /
        ``SOBHB_IDS``) or a ``REMOVE_BRANCHES`` entry.

        Args:
            name: Branch to remove.

        Returns:
            Settings: The removed branch block.

        Raises:
            KeyError: When ``name`` is not a present branch.

        Example:
            .. code-block:: python

                fit = erebor.all_sources(nwalkers=4)
                fit.remove_branch("emri")   # also drops emri_pe from every stage
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

        Args:
            move (Move | str | callable | eryn.moves.Move): The move (see above).
            **kwargs: Forwarded to :meth:`Recipe.add_move
                <lisatools.globalfit.recipe.Recipe.add_move>`: ``stage``,
                ``before`` / ``after`` (move names) or ``index``, ``name``,
                ``branch``, ``sync_log_like``.

        Returns:
            Move: The recipe entry that was added.

        Example:
            .. code-block:: python

                def my_move(model, state):
                    aca = model.analysis_container_arr
                    return state, None

                fit = erebor.all_sources(nwalkers=4)   # stages full_pe, vgb_pe
                fit.add_move(my_move, stage="full_pe", branch="gb")
        """
        mv = self.recipe.add_move(move, **kwargs)
        self._materialize_live(mv)
        return mv

    def pop_move(self, *args, **kwargs):
        """Remove and return a recipe move by name (alias of ``fit.recipe.pop_move``).

        Args:
            *args: Forwarded to :meth:`Recipe.pop_move
                <lisatools.globalfit.recipe.Recipe.pop_move>` (``name``, then
                optionally ``stage``).
            **kwargs: Forwarded likewise (``stage=`` disambiguates a name
                present in several stages).

        Returns:
            Move: The removed entry.

        Raises:
            KeyError: When the name is absent, or present in several stages
                and no ``stage`` is given.
        """
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
        """Toggle one move's debug instrumentation (alias of ``fit.recipe.set_move_debug``).

        Args:
            *args: Forwarded to :meth:`Recipe.set_move_debug
                <lisatools.globalfit.recipe.Recipe.set_move_debug>` (``name``,
                ``enabled=True``).
            **kwargs: Forwarded likewise (``stage=``, plus ``plot_dir`` /
                ``plot_walker`` / ``plot_leaf`` / ``plot_band`` / ``every``).

        Returns:
            Move: The edited recipe entry.
        """
        return self.recipe.set_move_debug(*args, **kwargs)

    def set_stage_debug(self, *args, **kwargs):
        """Toggle debug for every move of a stage (alias of ``fit.recipe.set_stage_debug``).

        Args:
            *args: Forwarded to :meth:`Recipe.set_stage_debug
                <lisatools.globalfit.recipe.Recipe.set_stage_debug>`
                (``stage_name``, ``enabled=True``).
            **kwargs: Forwarded likewise (the debug options).

        Returns:
            Stage: The edited stage.
        """
        return self.recipe.set_stage_debug(*args, **kwargs)

    def list_moves(self) -> str:
        """Stage-grouped, human-readable recipe summary (``fit.recipe.list_moves()``).

        Returns:
            str: One ``[kind] stage:`` header per stage, then one line per move
            (``name  <- stock|ClassName branch=...``).
        """
        return self.recipe.list_moves()

    # -- lifecycle ---------------------------------------------------------------

    @property
    def built(self) -> bool:
        """``True`` once :meth:`build` has run (and until :meth:`reset_build`)."""
        return "settings_dict" in self.__dict__

    def make_general_settings(self) -> GeneralSettings:
        """Resolve the general block into the final GeneralSettings for the build.

        The default returns a deepcopy of ``self.general``; variants override
        to fill derived fields (domain factories, processor kwargs) from the
        fit-level knobs at the last moment.

        Returns:
            GeneralSettings: A build-time copy (``self.general`` stays the
            editable, re-buildable configuration).
        """
        return deepcopy(self.general)

    def build_general(self, force: bool = False) -> GeneralSetup:
        """Build the (heavy) :class:`GeneralSetup`: data load, domain pour, sensitivity.

        Cached on the fit: a second call returns the same object unless
        ``force``.

        Args:
            force: Rebuild even when a cached general setup exists.

        Returns:
            GeneralSetup: Built from :meth:`make_general_settings`.
        """
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

        Args:
            name: Branch name.
            settings: The build-time branch block (from
                :meth:`prepare_branch_settings`).
            general_setup: The built general setup (unused by the base; kept
                for variant overrides).

        Returns:
            Setup: The constructed branch Setup (e.g.
            :class:`~lisatools.globalfit.stock.erebor.gb.GBSetup`).

        Raises:
            KeyError: When no Setup class is registered and the branch info is
                incomplete.
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

        It also stamps ``build_seed`` on any block that declares the field
        (GB/VGB today): the PER-RANK seed for prior/proposal objects the
        Setup constructs, from
        :func:`~lisatools.globalfit.communication.ranks.rank_build_seed`.
        ``None`` there means ``general.random_seed`` is unset, which keeps
        those objects on OS entropy exactly as before.

        Args:
            name: Branch name (``fit.<name>`` is copied).
            general_setup: The built general setup.

        Returns:
            Settings: A deepcopy of the branch block with the defaults filled.
        """
        settings = deepcopy(getattr(self, name))
        if settings.Tobs is None:
            settings.Tobs = general_setup.Tobs
        if settings.dt is None:
            settings.dt = general_setup.dt
        if hasattr(settings, "build_seed") and settings.build_seed is None:
            from ..communication.ranks import rank_build_seed

            settings.build_seed = rank_build_seed(self)
        return settings

    def build_sources(self, force: bool = False) -> typing.Dict[str, Setup]:
        """Build every branch Setup against the (cached) general setup.

        Args:
            force: Rebuild the branch Setups even when cached (the general
                setup is reused either way).

        Returns:
            dict[str, Setup]: ``{branch name: Setup}`` in branch order.
        """
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
        """Run the full heavy build and become a usable GlobalFitSetup.

        ``force=True`` runs :meth:`reset_build` first, which also drops the
        rank layout / device pin a prior ``prepare_rank`` left on the fit --
        call ``prepare_rank`` again afterwards under several ranks, or the run
        falls back to the late (post-build) layout path.

        Steps: :meth:`build_general` (data load, grid, domain, sensitivity),
        :meth:`build_sources` (one Setup per branch), then
        ``GlobalFitSetup.__init__`` on a
        :class:`~lisatools.globalfit.engine.GlobalFitSettings` carrying the
        live recipe (``source_metadata["recipe"]``), and finally
        :meth:`attach_runtime_objects`. A built fit returns immediately.

        Args:
            force: Drop the previous build products and build again.

        Returns:
            StockGlobalFit: ``self``, now usable as a
            :class:`~lisatools.globalfit.run.GlobalFitSetup`.

        Raises:
            ValueError: When ``setup_function`` is ``None``.

        Example:
            .. code-block:: python

                fit = erebor.all_sources(nwalkers=4)
                curr = fit.build()        # curr is fit
                print(fit.describe())     # now lists each branch's built Setup
        """
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
        """Drop all built products, returning to the cheap configured state.

        The rank wiring goes too (``rank_layout`` / ``rank`` /
        ``rank_device_mode`` / ``fanout``, class defaults ``None``): it is
        resolved per process by ``communication.ranks.prepare_rank`` just
        before a build, so a reset config that kept another process's layout
        (or a dead communicator) would rebuild against the wrong walker block.
        """
        for attr in (
            "_general_setup",
            "_source_info",
            *_BUILT_ONLY_ATTRS,
            *_RANK_ATTRS,
        ):
            self.__dict__.pop(attr, None)

    def run(self, comm=None, **run_kwargs):
        """One-shot pipeline: prepare this rank -> build (if needed) -> GlobalFit -> run_global_fit.

        ``comm`` defaults to ``MPI.COMM_WORLD``. Under more than one rank, the
        walker-block layout and this rank's device are resolved BEFORE the
        build (``communication.ranks.prepare_rank``: ``CUDA_VISIBLE_DEVICES``
        must be set before the first CUDA call); a fit that was already
        prepared or built is left alone. ``run_kwargs`` are forwarded to
        ``GlobalFit.run_global_fit``, which takes none (see Args).

        Every rank calls this; :class:`~lisatools.globalfit.run.GlobalFit`
        assigns the role (head / compute / saver). The production driver
        calls ``prepare_rank`` itself, then ``fit.build()`` and
        ``fit.run(comm=run_comm)`` on the communicator left after the monitor
        rank split.

        Args:
            comm (mpi4py.MPI.Comm, optional): Communicator. Default ``None`` ->
                ``MPI.COMM_WORLD``.
            **run_kwargs: Forwarded to
                :meth:`GlobalFit.run_global_fit
                <lisatools.globalfit.run.GlobalFit.run_global_fit>`, which
                currently takes no arguments, so this must stay empty.

        Returns:
            None: ``run_global_fit`` returns nothing; the results are in the
            run's HDF store (``general_info.main_file_path``, under
            ``general.file_store_dir``).

        Example:
            .. code-block:: bash

                # my_driver.py: fit = erebor.gb_no_fg(nwalkers=4); fit.run()
                mpiexec -n 3 python my_driver.py
        """
        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        if (
            int(comm.Get_size()) > 1
            and getattr(self, "rank_layout", None) is None
            and not self.built
        ):
            from ..communication.ranks import prepare_rank

            prepare_rank(self, comm)
        self.build()
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

        Args:
            iterations (int, optional): Iterations to run. Default ``None`` ->
                ``general.num_iterations``.
            **sample_kwargs: Forwarded to :meth:`GlobalFit.sample
                <lisatools.globalfit.run.GlobalFit.sample>`: ``thin_by``,
                ``progress``, ``store``, ``sync_log_like``.

        Yields:
            tuple: ``(model, state)`` once per (thinned) iteration — the
            engine model (``model.analysis_container_arr`` holds the residual)
            and the :class:`~lisatools.globalfit.state.GFState` fed to the next
            iteration.

        Example:
            .. code-block:: python

                fit = erebor.blank(nwalkers=4, ntemps=2, make_diagnostic_plots=False)
                for model, state in fit.sample(iterations=3, store=False, progress=False):
                    print(state.log_like)
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

        Args:
            value (object): Any settings value.
            maxlen: Truncation length of the fallback ``repr``.

        Returns:
            str: The one-line illustration.
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

        Returns:
            dict: The nested mapping shown above.
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

        Args:
            full: List every field of every block instead of the headline
                knobs only. Default ``False``.

        Returns:
            str: The multi-line summary (print it).

        Example:
            .. code-block:: python

                fit = erebor.gb_no_fg_lite()
                print(fit.describe())            # headline knobs + recipe
                print(fit.describe(full=True))   # every field of every block
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

        Args:
            overrides: Same keys as the constructor ``**knobs``.
        """
        self._apply_knobs(overrides)

    def __call__(self, **overrides) -> "StockGlobalFit":
        """Clone this configuration (config only, never built products) with overrides.

        The clone is a ``deepcopy`` followed by :meth:`reset_build` and
        :meth:`apply_overrides`. Field values (including env-resolved ones)
        are copied from ``self``: env vars are NOT re-read, so a knob exported
        after the module default was constructed must be passed as an
        override (or the class re-constructed, e.g. via ``get_stock``).

        Args:
            **overrides: Constructor-style knobs (headline knobs, variant
                attributes, general fields, ``lite=True``).

        Returns:
            StockGlobalFit: A fresh, unbuilt fit.

        Example:
            .. code-block:: python

                fit = erebor.all_sources(nwalkers=4)   # what the production driver does
        """
        new = deepcopy(self)
        new.reset_build()
        new.apply_overrides(overrides)
        return new

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in ("_general_setup", "_source_info", "_runner", *_BUILT_ONLY_ATTRS):
            state.pop(attr, None)
        # ``fanout`` is a LIVE communicator wrapper (mpi4py comm + head-side
        # command state): not picklable, and meaningless in another process.
        # ``rank_layout`` is a plain frozen dataclass and stays.
        state.pop("fanout", None)
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

        Args:
            **overrides: Constructor knobs.

        Returns:
            StockGlobalFit: ``cls(**overrides)``.
        """
        return cls(**overrides)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class StockRegistry:
    """Name -> StockGlobalFit-class registry for one stock family (e.g. erebor).

    The erebor package builds one at import (``erebor._registry``), registers
    every variant class, and exposes :meth:`get` / :meth:`options` as
    ``erebor.get_stock`` / ``erebor.get_stock_options``.

    Args:
        family: Family name used in error messages (``"erebor"``).

    Attributes:
        family: See Args.

    Example:
        .. code-block:: python

            from lisatools.globalfit.stock import erebor

            erebor.get_stock_options()                  # [(name, description), ...]
            fit = erebor.get_stock("gb_no_fg", nwalkers=16)
    """

    def __init__(self, family: str):
        self.family = family
        self._classes: typing.Dict[str, type] = {}

    def register(self, cls: type) -> type:
        """Class decorator: register ``cls`` under its ``option_name``.

        Args:
            cls: A :class:`StockGlobalFit` subclass with a non-empty
                ``option_name``.

        Returns:
            type: ``cls`` unchanged (decorator form).

        Raises:
            ValueError: When ``option_name`` is empty or already registered.
        """
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
        """Instantiate the stock fit registered under ``option``.

        Constructs a NEW instance from the class (env-backed field defaults
        are read now), unlike calling a module-level default instance
        (``erebor.gb_no_fg(...)``), which clones that instance.

        Args:
            option: Registered option name.
            *args: Forwarded to the class constructor.
            **kwargs: Forwarded to the class constructor (the knobs of
                :class:`StockGlobalFit`).

        Returns:
            StockGlobalFit: The unbuilt fit.

        Raises:
            ValueError: For an unknown option name (the message lists the
                available ones).
        """
        try:
            cls = self._classes[option]
        except KeyError:
            raise ValueError(
                f"Unknown stock option {option!r} in family {self.family!r}. "
                f"Available: {self.names()}."
            ) from None
        return cls(*args, **kwargs)
