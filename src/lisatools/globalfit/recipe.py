"""Recipe orchestration for sequencing global-fit sampling stages.

This module is the single home for the recipe machinery: the unified
:class:`Recipe` (declarative stage list + runtime engine in one object), the
declarative :class:`Stage`, the generic recipe-step base classes
(:class:`SearchRecipeStep` / :class:`PERecipeStep` / :class:`RJRecipeStep`), the
per-source move-builder hierarchy (:class:`SourceMoveBuilder` and subclasses),
and the injection / catalogue helpers.

A ``Recipe`` lives two lives on the same object: before the run it is a cheap,
picklable, editable list of :class:`Stage` blocks (each holding
:class:`~lisatools.globalfit.moves.globalfitmove.Move` objects); at run start
the variant's setup function calls :meth:`Recipe.setup` with a
:class:`~lisatools.globalfit.moves.globalfitmove.MoveBuildContext`, which
materializes every stage (running each move's ``setup(ctx)``) into the runtime
recipe steps that drive the sampler.
"""

from __future__ import annotations

import os
import time
import logging
import typing
import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

# DataResidualArray is now a deprecation shim; we pass DomainBase children
# (or raw arrays via the AnalysisContainer/template APIs) directly.
from lisatools.domains import FDSettings, WDMSettings

from bbhx.utils.transform import SSB_to_LISA
from gbgpu.gbgpu import GBGPU
from eryn.moves.tempering import TemperatureControl, make_ladder
from eryn.prior import ProbDistContainer

from ..sources.utils import icrs_to_ecliptic, evolve_galactic_binary
from ..utils.utility import asnumpy
from .moves import (
    FunctionMove,
    GFCombineMove,
    Move,
    MoveBuildContext,
    PSDMove,
    ResidualAddOneRemoveOneMove,
    SOBBHChunkedLikeMove,
    MBHBatchedLikeMove,
    EMRIDirectLikeMove,
    GBSpecialRJPriorMove,
    GBSpecialRJFStatGridMove,
    GBSpecialStretchMove,
    GBSpecialRJSerialSearchMCMC,
    GBSpecialRJRefitMove,
    VGBSpecialStretchMove,
)
from .moves.gbspecialstretch import GBSpecialBase
from .moves.globalfitmove import _RuntimeMove

# Type-only imports. These live under TYPE_CHECKING because ``run`` imports this
# module (``run.py`` -> ``from .recipe import Recipe``); importing ``.run`` /
# ``.engine`` / ``.state`` / ``.stock.erebor`` / ``..analysiscontainer`` at module
# scope here would create an import cycle. ``from __future__ import annotations``
# turns every annotation below into a string, so these names are only needed by
# type checkers, never at runtime.
if typing.TYPE_CHECKING:
    from .engine import Setup, GlobalFitEngine
    from .run import CurrentInfoGlobalFit
    from .state import GFState
    from .stock.erebor import GBSetup, GeneralSetup
    from ..analysiscontainer import AnalysisContainerArray

logger = logging.getLogger(__name__)

MOJITO_REFERENCE_TIME = 97729089.327664

#: ``rj_flip_fraction`` policy: the share of eligible RJ slots a single
#: proposal round flips.
#:
#: These are DEFAULTS IN CODE, deliberately, and that is the only way to
#: hold two different values at once: ``{BRANCH}_RJ_FLIP_FRACTION`` is
#: GLOBAL across stages, so a single exported value lands on every RJ move
#: in every stage and collapses the search/PE distinction. Submit scripts
#: therefore leave the env var unset; an explicit kwarg still overrides
#: per move.
#:
#: Named constants rather than bare literals because the PE value was
#: duplicated at two sites below, and those are exactly the pairs that
#: drift apart. Pinned in tests/test_rj_flip_fraction.py.
#: user ruling 2026-08-28: MATCH v6 -- 0.2 everywhere. v3-v6 all exported
#: GB_RJ_FLIP_FRACTION=0.2, and env beats the per-move default, so every
#: RJ move in those runs ran at 0.2 in BOTH search and PE. Keeping the
#: two names distinct (rather than one constant) so the search/PE split
#: can be reopened without re-deriving which sites feed which stage.
_SEARCH_RJ_FLIP_DEFAULT = 0.2
_PE_RJ_FLIP_DEFAULT = 0.2


def _stage_rj_flip_default(env_name: str, code_default: float) -> float:
    """Per-STAGE flip-fraction default: ``env_name`` if exported, else the
    code constant (user ruling 2026-09-11: the 0.2s stay the defaults, but
    each stage gets its own environment knob so a launch script can thin
    search and PE separately -- ``GB_SEARCH_RJ_FLIP_FRACTION`` /
    ``GB_PE_RJ_FLIP_FRACTION``). Precedence is unchanged above this: an
    explicit per-move kwarg, then the GLOBAL ``{BRANCH}_RJ_FLIP_FRACTION``,
    then this stage default. Validated like the move-level resolver.
    """
    raw = os.environ.get(env_name, None)
    if raw is None or not str(raw).strip():
        return float(code_default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a float in (0, 1], got {raw!r}.") from exc
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{env_name} must be in (0, 1], got {value}.")
    return value


def _search_rj_flip_default() -> float:
    return _stage_rj_flip_default("GB_SEARCH_RJ_FLIP_FRACTION", _SEARCH_RJ_FLIP_DEFAULT)


def _pe_rj_flip_default() -> float:
    return _stage_rj_flip_default("GB_PE_RJ_FLIP_FRACTION", _PE_RJ_FLIP_DEFAULT)


def warm_search_move_overrides() -> dict:
    """``rj_warm_search``'s deltas vs ``rj_fstat_search`` -- the auditable seam.

    User ruling 2026-09-02: the warm move is "built exactly like
    fstat_search, with the distribution switched out", so ``phase_maximize``
    follows the SAME ``GB_RJ_PHASE_MAXIMIZE`` seeding as rj_fstat_search
    (it was hardcoded False on the carried-phases theory; at a longer Tobs
    the carried phase degrades with the fdot uncertainty, and the ruling
    supersedes it). The remaining two are CYCLE INVARIANTS, not proposal
    behavior, and stay pinned: ``run_swaps=False`` (exactly one GB move
    tempers per iteration -- rj_fstat_search/removal own it) and
    ``leaf_cap_update=False`` (rj_fstat_search is the designated cap
    updater; a second updater would double-advance the cap counters).
    """
    return dict(
        phase_maximize=bool(int(os.environ.get("GB_RJ_PHASE_MAXIMIZE", "0"))),
        run_swaps=False,
        leaf_cap_update=False,
    )


def warm_pe_move_overrides() -> dict:
    """``rj_warm_pe``'s deltas vs ``rj_warm_search`` -- the PE seam.

    User ruling 2026-09-07: the warm move gets a PE twin that "mirrors the
    fstat search <-> pe differences". Mirrored deltas: ``phase_maximize``
    is UNCONDITIONALLY False (the no-maximization-in-PE policy -- exact
    detailed balance; ``GB_RJ_PHASE_MAXIMIZE`` never leaks in);
    ``run_swaps=True`` under the PE tempering BUDGET
    (``GB_TEMPER_EVERY_PROPOSES``, the same cadence rj_fstat_pe carries --
    at most one band-swap stage per N total proposes across the PE cycle's
    swap-enabled moves); ``leaf_cap_update`` stays False (never a cap
    updater, same as the search twin). Exactness of the charged density
    itself is the container's job: the PE instance is built with
    ``circ_images > 0`` (wrapped-normal image sums).
    """
    return dict(
        phase_maximize=False,
        run_swaps=True,
        temper_every_proposes=int(
            os.environ.get("GB_TEMPER_EVERY_PROPOSES", "3")),
        leaf_cap_update=False,
    )


def pe_phase_maximize_on(gb_mode_search: bool, pe_strict: bool) -> bool:
    """Whether ``rj_fstat_pe`` may phase-maximize. Default: NEVER.

    User ruling 2026-09-19. ``rj_fstat_pe`` used to read

        phase_maximize = _rj_phase_max if (gb_mode_search and not pe_strict)

    which made it the ONE move in the PE stack that could maximize -- every
    other one (``rj_warm_pe``, ``rj_prior_pe``, ``rj_replace_pe``, ``vgb_pe``)
    is hardcoded False. The intent was the ``GB_MODE=search`` campaign, which
    runs the search THROUGH the pe-named stage, but the effect was that
    setting ``GB_MODE=search`` silently re-armed phase maximization inside
    ``full_pe`` -- a config change in one place rearming machinery in another
    with no line saying so, which is exactly the anti-pattern
    :func:`~lisatools.globalfit.moves.gbspecialstretch._rj_amp_maximize_on`
    was written to avoid.

    ``GB_PE_MOVES_STRICT=1`` was the only documented way to close it, and it
    is NOT a safe door to shut on its own: the same flag also nulls the PE
    leaf caps (``leaf_cap_start=None``, ``leaf_cap_update=False``) and swaps
    the RJ flip fraction and IMR defaults. Reaching for it to disarm phase-max
    would quietly reconfigure the cap machinery of a production PE stage.

    So the knob is now its OWN: ``GB_PE_PHASE_MAXIMIZE``, default off. A
    search campaign that genuinely wants the old behaviour asks for it by
    name, and the search-stage moves (``rj_fstat_search``, ``rj_warm_search``,
    ``rj_prior_removal``) are untouched -- they keep following
    ``GB_RJ_PHASE_MAXIMIZE`` under the standing search waiver.

    ``pe_extrinsic_draw`` deliberately keeps its own mode-following gate: it
    selects between two EXACTLY BALANCED proposals (pinned-and-uniform-washed
    vs maximizer-centered-and-charged), so it is not a maximization and is not
    what this policy is about.
    """
    if os.environ.get("GB_PE_PHASE_MAXIMIZE", "0") != "1":
        return False
    return bool(gb_mode_search) and not bool(pe_strict)


# ---- the two PE F-stat moves (user ruling 2026-10-12) ----------------------
# Mike, 2026-10-12: "I would like to do two fstat moves that will split their
# share of the total moves. One fstat move fits the GB-free residual so the
# whole galaxy down to SNR 6.5. [...] The second fstat move will fit the
# gb-inclusive residual every 150 iterations. These iteration counts are the
# overall iterations, so the first time after the overall iterations pass a
# number, it should refit the first time that proposal is chosen." -- then:
# "set the gb-free residual refit (with the full galaxy found) to not
# refitting during full pe. It should just fit once the first time it is
# visiting in full pe. The residual fstat will take care of low snr
# adjustments." -- and: "this should not be specific to each walker, just
# specific to the one walker that was randomly drawn."
#
#   rj_fstat_pe            the GB-FREE residual (the reference walker's cold GBs
#                          restored for the sweep): the whole galaxy. Root
#                          ``shared`` (unchanged, so a resumed store loads the
#                          PE epochs it already fitted). Fit ONCE:
#                          GB_FSTAT_REFIT_EVERY_PE_GB_FREE (default 0). Floor
#                          GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE (default SNR 6.5).
#   rj_fstat_pe_residual   the LIVE residual (GBs subtracted), one random cold
#                          walker per epoch (GB_FSTAT_PE_REF). Root
#                          ``shared_pe_residual``. Refit every
#                          GB_FSTAT_REFIT_EVERY_PE (150) elapsed iterations,
#                          decided inside its propose. Floor: the stage's.
#
# The PE draw splits GB_PE_RJ_FSTAT_FRACTION between them by
# GB_PE_RJ_FSTAT_RESIDUAL_SPLIT (globalfitmove.pe_rj_fstat_residual_split).

#: the PE live-residual F-stat move's name
PE_FSTAT_RESIDUAL_MOVE = "rj_fstat_pe_residual"
#: the default GB-free galaxy floor (SNR), ruling 2026-10-12
PE_FSTAT_GB_FREE_PEAK_MIN_SNR_DEFAULT = 6.5


def _env_nonneg_int(knob: str, default: int) -> int:
    raw = os.environ.get(knob, "").strip()
    try:
        val = int(raw) if raw else int(default)
    except ValueError:
        raise ValueError(f"{knob}={raw!r} must be an integer (iterations; "
                         "0 = fit once and never refit).") from None
    if val < 0:
        raise ValueError(f"{knob}={raw!r} must be >= 0 (0 = fit once and "
                         "never refit).")
    return val


def pe_fstat_refit_every() -> int:
    """``GB_FSTAT_REFIT_EVERY_PE`` (default 150): the PE refit cadence, in
    ELAPSED iterations on the F-stat refit clock, of the LIVE-residual move
    ``rj_fstat_pe_residual`` (since 2026-10-12; it was ``rj_fstat_pe``'s until
    the galaxy move became fit-once). The decision runs inside the move's
    propose, so the refit lands on its first draw at or after the cadence."""
    return _env_nonneg_int("GB_FSTAT_REFIT_EVERY_PE", 150)


def pe_fstat_gb_free_refit_every() -> int:
    """``GB_FSTAT_REFIT_EVERY_PE_GB_FREE`` (default 0): the GB-free galaxy
    move ``rj_fstat_pe``'s cadence. 0 = fit exactly ONCE (at its first visit
    in PE, or load the epoch a resumed store already holds) and never refit
    (user ruling 2026-10-12)."""
    return _env_nonneg_int("GB_FSTAT_REFIT_EVERY_PE_GB_FREE", 0)


def pe_fstat_gb_free_peak_min_snr():
    """``GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE`` (default 6.5): the GB-free galaxy
    move's OWN F-stat peak floor, as an SNR (F = SNR^2 / 2). ``none`` / ``stage``
    hands it back to the stage's declared floor (full_pe: 6.25). A fit selects
    at it; a LOAD of an epoch selected lower keeps only the boxes at or above
    it (no refit -- see ``fstat_gridfit.stacked_cache_at_floor``)."""
    raw = os.environ.get("GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE", "").strip()
    if raw.lower() in ("none", "stage"):
        return None
    try:
        val = float(raw) if raw else PE_FSTAT_GB_FREE_PEAK_MIN_SNR_DEFAULT
    except ValueError:
        raise ValueError(
            f"GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE={raw!r} must be an SNR (or "
            "'none' for the stage's floor).") from None
    if not (np.isfinite(val) and val > 0.0):
        raise ValueError(
            f"GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE={raw!r} must be a positive SNR.")
    return val


def pe_fstat_move_config(name: str) -> dict:
    """The per-move F-stat configuration :func:`build_gb_moves` gives a PE
    F-stat grid move, by name: ``{fstat_refit_every, fstat_peak_min_snr,
    fstat_pe_residual}``.

    ``rj_replace_pe`` (not in the v9 recipe) shares ``rj_fstat_pe``'s root and
    therefore its configuration: on the old shared cadence it would have
    re-opened the galaxy's epochs every 150 iterations.
    """
    if name in ("rj_fstat_pe", "rj_replace_pe"):
        return dict(fstat_refit_every=pe_fstat_gb_free_refit_every(),
                    fstat_peak_min_snr=pe_fstat_gb_free_peak_min_snr(),
                    fstat_pe_residual=False)
    if name == PE_FSTAT_RESIDUAL_MOVE:
        return dict(fstat_refit_every=pe_fstat_refit_every(),
                    fstat_peak_min_snr=None,
                    fstat_pe_residual=True)
    raise KeyError(f"{name!r} is not a PE F-stat grid move.")


def _stage_kind_of(step):
    """The stage kind (``"search"``/``"rj"``/``"pe"``) stamped on a materialized step.

    :meth:`Stage.setup` stamps ``gf_stage_kind`` on the stage's single
    :class:`~lisatools.globalfit.moves.combinemove.GFCombineMove`, which is the
    step's only move; ``(move, weight)`` entries are unwrapped. Returns ``None``
    for anything that does not carry the stamp -- including a legacy
    ``RecipeStep`` whose ``moves`` property RAISES when it was never given any.
    """
    try:
        moves = list(getattr(step, "moves", None) or [])
    except Exception:  # noqa: BLE001 - RecipeStep.moves raises when unset
        return None
    if not moves:
        return None
    first = moves[0]
    if isinstance(first, tuple):
        first = first[0]
    return getattr(first, "gf_stage_kind", None)


class Recipe:
    """The global-fit recipe: declarative stage list + runtime step engine, one object.

    **Declarative life** (before the run): an ordered, editable list of
    :class:`Stage` blocks, each holding
    :class:`~lisatools.globalfit.moves.globalfitmove.Move` objects. Cheap and
    picklable — this is what rides on a configured
    ``StockGlobalFit`` (``fit.recipe``). Edit it with
    :meth:`add_stage` / :meth:`add_move` / :meth:`pop_move` / ... .

    **Runtime life** (from run start): :meth:`setup` materializes every stage
    into a runtime :class:`RecipeStep`; the driver then iterates the object —
    at each call it asks the current step's stopping function whether to
    advance, and on advance it invokes the next step's ``setup_run`` to
    reconfigure the sampler.

    Args:
        stages: Optional initial list of :class:`Stage` blocks.
    """

    #: Multi-rank fan-out (:class:`~lisatools.globalfit.communication.fanout.WalkerFanout`);
    #: the runner sets it before the first step so the recipe can publish the stage /
    #: iteration clock the compute ranks read off each command. ``None`` in a
    #: single-process run, where both hooks below are no-ops.
    fanout = None

    def __init__(self, stages: typing.Optional[typing.List["Stage"]] = None):
        self.stages: typing.List["Stage"] = list(stages) if stages is not None else []
        self._check_unique()
        self._init_runtime()

    def _init_runtime(self):
        self.recipe = []
        self.backend_added = False
        self._current_iter = 0
        self._current_recipe_step = None
        self._has_setup_first_step = False
        self.stock_moves: typing.Dict[str, typing.Any] = {}
        # Cached once here (not re-read every iteration): GF_FANOUT_DIGEST=1
        # makes __call__ log a head-side [FANOUT_DIGEST] line per iteration
        # (Plan 5 Task 4 of the multi-rank port). Off by default, so a
        # single-process run's behaviour is unchanged.
        self._fanout_digest_enabled = os.environ.get("GF_FANOUT_DIGEST", "0") == "1"

    # -- pickling: runtime products never travel with the config ---------------

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in (
            "recipe",
            "_backend",
            "backend_added",
            "_current_iter",
            "_current_recipe_step",
            "_has_setup_first_step",
            "stock_moves",
            # the fan-out owns an MPI communicator: a runtime product that
            # must never travel with the (picklable, deepcopy-able) config.
            "fanout",
            # the residual-snapshot context holds the live ACA + data arrays
            "snapshot_context",
        ):
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_runtime()

    # -- declarative editing API ------------------------------------------------

    @staticmethod
    def _coerce_move(move, *, name=None, branch=None, sync_log_like=True) -> Move:
        """Coerce anything move-like into a :class:`Move`.

        ``Move`` -> as-is; ``str`` -> stock-name :class:`Move`; an object with
        ``.propose`` -> a private runtime wrapper (pickle caveat); a plain
        callable -> :class:`~lisatools.globalfit.moves.functionmove.FunctionMove`.
        """
        if isinstance(move, Move):
            if branch is not None and move.branch is None:
                move.branch = branch
            return move
        if isinstance(move, str):
            return Move(move, branch=branch)
        if hasattr(move, "propose"):
            return _RuntimeMove(move, name=name, branch=branch)
        if callable(move):
            return FunctionMove(move, name=name, branch=branch, sync_log_like=sync_log_like)
        raise TypeError(
            "add_move expects a Move, a stock-move name, a constructed eryn move, "
            f"or a plain fn(model, state); got {type(move).__name__}."
        )

    def _check_unique(self):
        stage_names = [s.name for s in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError(f"Duplicate stage names: {stage_names}.")
        # Move names are unique WITHIN a stage, not across the recipe: a
        # staged run legitimately installs the same stock move in several
        # stages (e.g. ``psd_pe``/``galfor_pe`` keep sampling the noise model
        # through both the GB search stage and the GB PE stage). Both
        # resolve to the SAME runtime object via the stock-name lookup, and
        # stages run sequentially, so sharing it is safe. ``_find_move`` has
        # always handled the multi-stage case by asking for ``stage=`` to
        # disambiguate — that path was unreachable while this check was
        # global.
        for stage in self.stages:
            names = [m.name for m in stage.moves]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"Duplicate move names within stage {stage.name!r}: {names}."
                )

    def _stage(self, name: str) -> "Stage":
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(
            f"Unknown stage {name!r}. Available stages: {[s.name for s in self.stages]}."
        )

    def _find_move(
        self, name: str, stage: typing.Optional[str] = None
    ) -> typing.Tuple["Stage", int]:
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

    def add_stage(
        self,
        stage: "Stage",
        before: typing.Optional[str] = None,
        after: typing.Optional[str] = None,
        index: typing.Optional[int] = None,
    ) -> "Stage":
        if not isinstance(stage, Stage):
            raise TypeError(f"add_stage expects a Stage, got {type(stage).__name__}.")
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

    def pop_stage(self, name: str) -> "Stage":
        stage = self._stage(name)
        self.stages.remove(stage)
        return stage

    def move_names(self) -> typing.List[str]:
        return [m.name for s in self.stages for m in s.moves]

    def stock_names(self) -> typing.List[str]:
        """Stock-move names this recipe needs built.

        The variant setup functions use this to build exactly the stock moves
        the recipe asks for -- so a name missing here is simply never built,
        and the move that wanted it fails at materialization with an empty
        ``ctx.stock_moves``.

        Two sources:

        * base :class:`Move` entries, which resolve BY name; and
        * ``Move.stock_dependencies()`` on custom subclasses, which resolve by
          their own ``setup`` but may COMPOSE stock moves (e.g. a combine move
          wrapping ``psd_pe`` + ``galfor_pe``). Those dependencies are
          invisible from the move's own name, so a subclass that pulls things
          out of ``ctx.stock_moves`` must declare them here or they will not
          exist when it looks.
        """
        names = []
        for s in self.stages:
            for m in s.moves:
                if m.is_stock:
                    names.append(m.name)
                else:
                    names.extend(m.stock_dependencies())
        return names

    def get_move(self, name: str, stage: typing.Optional[str] = None) -> Move:
        st, i = self._find_move(name, stage)
        return st.moves[i]

    def pop_move(self, name: str, stage: typing.Optional[str] = None) -> Move:
        st, i = self._find_move(name, stage)
        return st.moves.pop(i)

    def set_move_debug(
        self,
        name: str,
        enabled: bool = True,
        *,
        stage: typing.Optional[str] = None,
        **opts,
    ) -> Move:
        """Turn a single move's debug instrumentation on/off (materialize-time).

        ``opts`` (plot_dir / plot_walker / plot_leaf / plot_band / every) are
        forwarded to :meth:`GlobalFitMove.set_debug`. Overrides any
        stage-level debug and the move's env default. Example::

            fit.recipe.set_move_debug("emri_pe", plot_dir="./emri_dbg", every=5)
            fit.recipe.set_move_debug("psd_pe", False)   # force off
        """
        mv = self.get_move(name, stage)
        mv.debug = {"enabled": enabled, **opts} if opts else enabled
        return mv

    def set_stage_debug(
        self, stage_name: str, enabled: bool = True, **opts
    ) -> "Stage":
        """Turn debug on/off for every move in a stage (unless a move overrides).

        ``opts`` are forwarded to each move's :meth:`GlobalFitMove.set_debug`.
        Example::

            fit.recipe.set_stage_debug("full_pe", plot_walker=2)
        """
        st = self._stage(stage_name)
        st.debug = {"enabled": enabled, **opts} if opts else enabled
        return st

    def add_move(
        self,
        move,
        stage: typing.Optional[str] = None,
        before: typing.Optional[str] = None,
        after: typing.Optional[str] = None,
        index: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        branch: typing.Optional[str] = None,
        sync_log_like: bool = True,
    ) -> Move:
        """Add a move (a.k.a. proposal) to the recipe — the single entrance.

        ``move`` is a :class:`Move`, a stock-move name (``str``), a constructed
        eryn move (wrapped; note the pickle caveat), or a plain
        ``fn(model, state) -> (new_state, accepted)`` function (wrapped in a
        :class:`~lisatools.globalfit.moves.functionmove.FunctionMove`;
        ``sync_log_like`` applies to this case). Placement: with no stages an
        initial ``Stage("main", kind="pe")`` is created; with one stage the
        move lands there; with several, ``stage=`` is required. At most one of
        ``before=``/``after=`` (move names) or ``index=`` positions it inside
        the stage (default: append).
        """
        move = self._coerce_move(move, name=name, branch=branch, sync_log_like=sync_log_like)
        if sum(x is not None for x in (before, after, index)) > 1:
            raise ValueError("Pass at most one of before=, after=, index=.")

        if not self.stages:
            self.add_stage(Stage("main", kind="pe"))

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

        # Uniqueness is per-stage (see ``_check_unique``): the same stock move
        # may appear in several stages of a staged run.
        if move.name in [m.name for m in st.moves]:
            raise ValueError(
                f"Move name {move.name!r} already present in stage {st.name!r} "
                f"(existing: {[m.name for m in st.moves]}). pop_move it first "
                "or rename."
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
                src = "stock" if mv.is_stock else type(mv).__name__
                extra = f" branch={mv.branch}" if mv.branch else ""
                lines.append(f"    {mv.name}  <- {src}{extra}")
        return "\n".join(lines) if lines else "(empty recipe)"

    def __repr__(self):
        return f"Recipe({[s.name for s in self.stages]})"

    # -- materialization ---------------------------------------------------------

    def setup(self, ctx: MoveBuildContext) -> None:
        """Materialize the declarative stages into runtime recipe steps.

        Called once at run start (by the variant's setup function) with the
        live :class:`~lisatools.globalfit.moves.globalfitmove.MoveBuildContext`.
        Runs every move's ``setup(ctx)`` via :meth:`Stage.setup` and registers
        the resulting steps on this same object.
        """
        if not self.stages:
            raise ValueError(
                "Recipe has no stages; add a Stage (or add_move, which "
                "auto-creates a 'main' PE stage)."
            )
        self.stock_moves = dict(ctx.stock_moves)
        # Every branch appended with plain branch info must actually be
        # adjusted by at least one move.
        info_branches = tuple(getattr(ctx.curr, "_info_branches", ()) or ())
        targeted = {m.branch for st in self.stages for m in st.moves if m.branch is not None}
        missing = [b for b in info_branches if b not in targeted]
        if missing:
            raise ValueError(
                f"Branch(es) {missing} were added with branch info but no move "
                "targets them (branch=<name>). Add at least one move "
                "(add_move / add_branch(moves=[...])) so the branch is actually "
                "adjusted in the fit."
            )
        for st in self.stages:
            self.add_recipe_component(st.setup(ctx), name=st.name)

    # -- runtime engine ------------------------------------------------------------

    def get_step(self, name: str):
        """Return the materialized runtime step registered under ``name``."""
        for entry in self.recipe:
            if entry["name"] == name:
                return entry["adjust"]
        raise KeyError(
            f"Unknown recipe step {name!r}. Available: {[e['name'] for e in self.recipe]}."
        )

    @property
    def backend(self):
        """Backend object that records recipe-step completion."""
        return self._backend

    @backend.setter
    def backend(self, backend):
        self._backend = backend
        self.backend_added = True

    def add_recipe_component(self, adjust_fn, name=None):
        """Append a recipe step.

        Args:
            adjust_fn: A :class:`RecipeStep` (or compatible object) implementing
                ``setup_run`` and ``stopping_function``.
            name: Optional human-readable name. If ``None``, a default name is
                assigned based on the current recipe length.
        """
        if name is None:
            name = f"recipe step {len(self.recipe) + 1}"
        self.recipe.append({"name": name, "adjust": adjust_fn, "status": False})

    def to_file(self):
        """Return a dict mapping recipe-step names to their completion status."""
        _tmp = {recipe_step["name"]: recipe_step["status"] for recipe_step in self.recipe}
        return _tmp

    def __next__(self):
        """Advance the internal cursor past any completed steps."""
        while self._current_iter < len(self.recipe):
            # False means it is not finished
            if self.recipe[self._current_iter]["status"]:
                self._current_iter += 1

            else:
                break

        if self._current_iter < len(self.recipe):
            self._current_recipe_step = self.recipe[self._current_iter]

    def setup_first_recipe_step(self, iteration, last_sample, sampler):
        """Configure the sampler for the very first (incomplete) recipe step.

        Args:
            iteration: Current iteration index.
            last_sample: Last sampled state object.
            sampler: The :class:`GlobalFitEngine` (or eryn-compatible) sampler.

        Raises:
            ValueError: If the recipe has already been completed.
        """
        assert not self._has_setup_first_step
        # move to next recipe step
        next(self)
        if self._current_iter >= len(self.recipe):
            raise ValueError("Recipe is already finished.")

        self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)
        self._announce_recipe_step()
        if self.fanout is not None:
            # Seed the fan-out clock BEFORE the first propose -- otherwise it
            # sits at its constructed 0 through the whole first iteration.
            # ``__call__`` refreshes it from the stopping-function call that
            # FOLLOWS each iteration, so once sampling is under way the
            # ``iteration`` a propose reads is the index of the PREVIOUS
            # completed iteration: a stable label for tracing, not a counter a
            # rank-side body may key behaviour on.
            self.fanout.note_iteration(iteration)
            self.fanout.enter_stage(
                self._current_recipe_step["name"],
                _stage_kind_of(self._current_recipe_step["adjust"]),
            )
        self._has_setup_first_step = True

    @property
    def current_recipe_step(self):
        """The active recipe-step record (a dict of ``name``/``adjust``/``status``)."""
        return self._current_recipe_step

    def __call__(self, iteration, last_sample, sampler):
        """Evaluate the current step's stopping criterion and advance if met.

        Args:
            iteration: Current iteration index.
            last_sample: Last sampled state object.
            sampler: The active sampler.

        Returns:
            ``True`` if the entire recipe has finished, ``False`` otherwise.
        """
        if self.fanout is not None:
            self.fanout.note_iteration(iteration)
        if self._fanout_digest_enabled:
            # Deliberately OUTSIDE the fan-out guard: ``_make_fanout`` returns
            # None in single mode, and the single-rank run is exactly the
            # baseline the cluster gates diff the multi-rank layouts against
            # (docs/multirank-cluster-gates.md Step 1), so the digest has to
            # fire there too.
            #
            # ``iteration`` here is the index of the iteration whose moves
            # just ran (this call is eryn's post-iteration stopping-fn hook)
            # -- NOT the same clock reading a propose sees via
            # ``note_iteration`` above, which a rank-side body reads one
            # iteration late (see setup_first_recipe_step).
            from .communication.fanout import fanout_digest_line

            hashes = None
            fanout = getattr(self, "fanout", None)
            if (
                fanout is not None
                and getattr(fanout.layout, "replica_mode", False)
                and getattr(fanout, "model", None) is not None
            ):
                try:
                    hashes = fanout.gather_residual_hashes(fanout.model.analysis_container_arr)
                except Exception as exc:  # noqa: BLE001 - a diagnostic hook must never break the run
                    logger.warning("[FANOUT_DIGEST] residual hash gather failed: %r", exc)
                    hashes = None
            logger.info(fanout_digest_line(
                iteration, last_sample, residual_hashes=hashes,
                layout=getattr(fanout, "layout", None) if fanout is not None else None))
        # GF_RESIDUAL_SNAPSHOT_EVERY: every cold walker's residual + the data
        # beside the store for the monitor page (never raises; see the module).
        _snap = getattr(self, "snapshot_context", None)
        if _snap is not None:
            from .residual_snapshot import maybe_snapshot

            maybe_snapshot(iteration=iteration, fanout=getattr(self, "fanout", None),
                           **_snap)
        stop_here = self._current_recipe_step["adjust"].stopping_function(
            iteration, last_sample, sampler
        )
        if stop_here:
            # the next INCOMPLETE step, so its start_iteration can be stamped
            # in the same write as this step's completion (see
            # GFHDFBackend.completed_recipe_step)
            #
            # ORDERING (user ruling 2026-10-07, every store write on the saver
            # rank): with a saver rank this write RUNS THERE, after the row
            # this iteration just handed off, and the call returns only once
            # the saver has acknowledged it -- so the next stage's setup_run
            # below reads the stamped start_iteration and the stored
            # iteration (the row is on disk) rather than racing them.
            _next_name = None
            for _j in range(self._current_iter + 1, len(self.recipe)):
                if not self.recipe[_j]["status"]:
                    _next_name = self.recipe[_j]["name"]
                    break
            self.backend.completed_recipe_step(
                self._current_recipe_step["name"], next_step_name=_next_name)
            self._current_recipe_step["status"] = True
            next(self)

            if self._current_iter >= len(self.recipe):
                return True
            self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)
            self._announce_recipe_step()
            if self.fanout is not None:
                self.fanout.enter_stage(
                    self._current_recipe_step["name"],
                    _stage_kind_of(self._current_recipe_step["adjust"]),
                )

        return False

    def _announce_recipe_step(self) -> None:
        """Tell this step's moves which step they are now running under.

        THE IDENTITY IS THE STEP INDEX (``_current_iter``), deliberately
        NOT the backend iteration. A step's index is stable for as long as
        that step is active, including across a mid-step restart -- the
        cursor is rebuilt from the per-step ``status`` flags, which the
        backend persists. The iteration number is not: on a resume it is
        wherever the run got to, so anything keyed on it would read as a
        NEW step and release state the step had legitimately earned.

        That distinction is the whole reason this lives on ``Recipe``
        rather than in ``RecipeStep.setup_run``. A step does not know its
        own index; the recipe does. It also means a Stage subclass that
        overrides ``setup_run`` without calling ``super()`` cannot silently
        break the announcement.

        Today's one consumer is the per-(walker, band) RJ shutoff valve,
        which must HOLD for the whole of a step and release only when the
        next one begins (user requirement 2026-09-24).
        """
        step = self._current_recipe_step
        if step is None:
            return
        adjust = step["adjust"]
        serial = int(self._current_iter)
        # Second consumer (2026-09-24, v9): a step that carries a per-stage
        # PROFILE applies it here rather than in ``setup_run`` -- it needs the
        # step serial, and only the recipe knows it (same argument as the
        # docstring's). Steps without the hook are unaffected.
        _note = getattr(adjust, "note_recipe_step", None)
        if callable(_note):
            _note(serial)
        begin_search_recipe_step(adjust.moves, serial)


class RecipeStep:
    """Abstract base for a single stage in a :class:`Recipe`.

    Each subclass must define a ``setup_run`` method that configures the
    sampler when the step becomes active and a ``stopping_function`` that
    decides when to advance to the next step.

    Args:
        moves: List of MCMC moves to use during this recipe step.
        weights: List of weights matching ``moves``. Defaults to uniform.
    """

    def __init__(self, moves=None, weights=None):
        if moves is not None:
            self.moves = moves
            if weights is not None:
                self.weights = weights

    def __repr__(self):
        return f"RecipeStep with moves: {self.moves} and weights: {self.weights}"

    @property
    def moves(self):
        """List of MCMC moves used by this step."""
        if not hasattr(self, "_moves"):
            raise ValueError("Must add moves for this recipe step.")
        return self._moves

    @moves.setter
    def moves(self, moves):
        self._moves = moves

    @property
    def weights(self):
        """List of weights corresponding to :attr:`moves`. Uniform by default."""
        if not hasattr(self, "_weights"):
            self._weights = [1.0 / len(self.moves) for _ in self.moves]
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def setup_run(self, iteration, last_sample, sampler):
        """Configure ``sampler`` for the start of this recipe step."""
        raise NotImplementedError

    def stopping_function(self, iteration, last_sample, sampler):
        """Return ``True`` when this recipe step should be considered done."""
        raise NotImplementedError

class BaseRecipeStep(RecipeStep):
    """Default :class:`RecipeStep` that simply assigns moves to the sampler.

    Args:
        moves: List of MCMC moves to use during this recipe step.
        weights: List of weights matching ``moves``.
    """

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        """Install :attr:`moves`/:attr:`weights` on the sampler.

        Each move that lacks an explicit periodicity setting inherits the
        sampler's periodicity.
        """
        for move in self.moves:
            if sampler.periodic is not None and move.periodic is None:
                logger.debug(f"Setting periodicity of move {move} to {sampler.periodic}")
                move.periodic = sampler.periodic

        sampler.moves = self.moves
        sampler.weights = self.weights


# ======================================================================
# Installable recipe steps, injection helpers, and move builders
# (folded in from the retired recipe_steps.py).
# ======================================================================


class SearchRecipeStep(BaseRecipeStep):
    """Recipe step that is done when the search itself completes.

    The stopping criterion is handled **internally**, inside the search move —
    the move runs its own search to completion (e.g. to log-likelihood
    convergence) before the recipe is consulted. So there is nothing left for
    the recipe to wait on and this step reports done on its first call. That is
    a statement about where the criterion lives, not about how much work was
    done: compare :class:`RJRecipeStep`, which owns its criterion at the recipe
    level and watches the leaf count across iterations.
    """

    def stopping_function(self, *args, **kwargs):
        """Done when every wrapped move reports its search complete.

        Chunked searches (:class:`MaxLogLCombineMove`) run a bounded number
        of inner iterations per ``propose`` so the backend checkpoints
        between chunks; they publish ``maxlogl_plateau_done``. Moves without
        the attribute keep the legacy semantics -- their criterion ran to
        completion inside one propose call, so the step is done on its
        first check.
        """
        for combined in self.moves or []:
            for mv in getattr(combined, "moves", None) or [combined]:
                if getattr(mv, "maxlogl_plateau_done", True) is False:
                    return False
        return True


class PERecipeStep(BaseRecipeStep):
    """Recipe step that runs indefinitely (ongoing parameter estimation).

    ``peak_min_snr`` (optional) DECLARES the F-stat peak-selection floor this
    stage runs at. ⚠ It exists because the v9 search stages install a
    PROCESS-GLOBAL in-code override of that floor
    (:class:`SearchStageProfileStep`) and nothing was clearing it: ``full_pe``
    silently inherited ``gb_search_3``'s 6.25 while the submit script
    exported 8.0 and the log's last word on the subject was a stale
    ``[V9-STAGE gb_search_3]`` line. 6.25 happens to be the value the design
    wants here -- an assembled model's faint tail must stay reachable -- but
    inheriting it by accident is not the same as choosing it, and the next
    person to change a search stage's floor would have moved this one too.
    Declared, applied and logged; ``None`` clears the override so the run's
    own ``FSTAT_PEAK_MIN_SNR`` governs again.
    """

    def __init__(self, *args, peak_min_snr=None, stage_name: str = "",
                 pe_repeats=None, pe_rj_flip_fraction=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_min_snr = peak_min_snr
        self.stage_name = stage_name
        # ``pe_repeats``: {branch: in-model repeats} this stage DECLARES for
        # its moves (user ruling 2026-10-02: "in-model repeats of 25 for all
        # sources except emris for right now for PE. This includes VGBs and
        # GBs"). Applied on entry to the built moves, which are shared with
        # the search stages -- hence per stage, not a process-global knob.
        self.pe_repeats = (None if not pe_repeats
                           else {str(k): int(v) for k, v in dict(pe_repeats).items()})
        # ``pe_rj_flip_fraction``: the fraction of AVAILABLE slots every GB RJ
        # move proposes on per propose in this stage (user ruling 2026-10-02:
        # "during full PE all RJ moves sample 0.1 of the available slots. This
        # is the FRAC env variable"). Declared per stage because the recipe
        # BUILDS the PE-named RJ moves with the SEARCH default whenever the GB
        # branch runs in search mode (``_rj_flip_default`` follows
        # ``gb_info.mode``, not the move's name), so in a v9 search run
        # GB_PE_RJ_FLIP_FRACTION=0.1 reached nothing until this.
        self.pe_rj_flip_fraction = (None if pe_rj_flip_fraction is None
                                    else float(pe_rj_flip_fraction))
        if self.pe_rj_flip_fraction is not None and not (
                0.0 < self.pe_rj_flip_fraction <= 1.0):
            raise ValueError(
                f"pe_rj_flip_fraction={self.pe_rj_flip_fraction} must be in (0, 1].")
        self._profile_serial = None

    def note_recipe_step(self, serial) -> None:
        """Apply this stage's declared peak floor and in-model repeats
        (idempotent per step)."""
        if serial is not None and serial == self._profile_serial:
            return
        self._profile_serial = serial
        from lisatools.sampling.fstat_proposal import (
            fstat_peak_min_F, set_peak_min_F_override)

        set_peak_min_F_override(self.peak_min_snr)
        _F = float(fstat_peak_min_F())
        logger.info(
            "[V9-STAGE %s] entering (recipe step %s): F-stat peak floor is "
            "F >= %.3f (SNR %.3f), %s.",
            self.stage_name or "pe", serial, _F, float(np.sqrt(2.0 * _F)),
            "declared by this stage" if self.peak_min_snr is not None
            else "from the run's FSTAT_PEAK_MIN_SNR (any search-stage "
                 "override has been cleared)")
        if self.pe_repeats:
            apply_inmodel_repeats(self.moves, self.pe_repeats,
                                  tag=self.stage_name or "pe")
        if self.pe_rj_flip_fraction is not None:
            apply_rj_flip_fraction(self.moves, self.pe_rj_flip_fraction,
                                   tag=self.stage_name or "pe")

    def stopping_function(self, *args, **kwargs):
        """Never stop on its own -- relies on outer stopping logic.

        ⚠ MUST STAY A METHOD OF THIS CLASS. 3ac014a6 (2026-10-02) inserted
        the module-level ``apply_rj_flip_fraction`` / ``apply_inmodel_repeats``
        above this def, which left it indented INSIDE
        ``apply_inmodel_repeats`` after its ``return`` -- dead code -- so
        full_pe inherited ``RecipeStep.stopping_function`` and raised
        NotImplementedError on its first iteration (6mo, 2026-10-04, right
        after replica_pe converged). Pinned by tests/test_pe_stopping.py.
        """
        return False


def apply_rj_flip_fraction(moves, fraction, tag: str = "pe") -> dict:
    """Set ``rj_flip_fraction`` on every GB RJ move of a built move tree.

    The fraction of AVAILABLE (dead) slots a birth proposal draws per
    propose; 1.0 = every candidate row. Applies to GB-branch moves that
    carry the attribute and propose RJ (``is_rj_prop``); in-model GB moves
    and the fixed-leaf VGB branch (no RJ, fraction pinned to 1.0 by its
    resolver) are left alone. Returns ``{move name: (old, new)}``.
    """
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"rj_flip_fraction must be in (0, 1], got {fraction}.")
    changed = {}
    for m in iter_move_tree(moves):
        if getattr(m, "branch_name", None) != "gb":
            continue
        if not hasattr(m, "rj_flip_fraction") or not getattr(m, "is_rj_prop", False):
            continue
        old = float(getattr(m, "rj_flip_fraction"))
        if old != fraction:
            m.rj_flip_fraction = fraction
            changed[getattr(m, "name", type(m).__name__)] = (old, fraction)
    if changed:
        logger.info("[V9-STAGE %s] GB RJ flip fraction set on entry: %s.", tag,
                    ", ".join(f"{k} {o:g} -> {n:g}" for k, (o, n) in changed.items()))
    return changed


def apply_inmodel_repeats(moves, repeats, tag: str = "pe") -> dict:
    """Set the per-move in-model repeat count, per branch, on a built move tree.

    ``repeats`` is ``{branch_name: n}``. The attribute differs by move family
    and is detected per move: the add/remove source moves (sobbh / mbh /
    emri) and the noise moves carry ``num_repeats``; the GB-family band moves
    (gb, vgb) carry ``num_repeat_proposals``. A move of a listed branch with
    neither attribute is reported and left alone -- never a silent no-op.

    GB RJ MOVES ALSO GET THEIR PER-CLASS BUDGETS (user ruling 2026-10-04:
    "make all the pe repeats 25 (for both newborns and survivors)"). An RJ
    band move runs its in-model polish from ``inmodel_repeats_newborn`` /
    ``inmodel_repeats_survivor`` (GB_INMODEL_REPEATS_NEWBORN / _SURVIVOR,
    100 / 50 on the 6mo launcher), NOT from ``num_repeat_proposals`` -- so
    until now the PE declaration left the GB PE moves at 100 / 50 while
    reporting nothing. Both are set to the declared count too; they reach
    the compute ranks through ``GB_RANK_LIVE_ATTRS``.
    Returns ``{move name: (old, new)}`` for what changed (a per-class change
    is keyed ``"<move>.newborn"`` / ``"<move>.survivor"``).
    """
    want = {str(k): int(v) for k, v in dict(repeats or {}).items()}
    for b, n in want.items():
        if n < 1:
            raise ValueError(f"in-model repeats for branch {b!r} must be >= 1 (got {n}).")
    changed, untouched = {}, []
    for m in iter_move_tree(moves):
        b = getattr(m, "branch_name", None)
        if b not in want:
            continue
        n = want[b]
        name = getattr(m, "name", type(m).__name__)
        if hasattr(m, "num_repeats"):
            attr = "num_repeats"
        elif hasattr(m, "num_repeat_proposals"):
            attr = "num_repeat_proposals"
        else:
            untouched.append(name)
            continue
        old = int(getattr(m, attr))
        if old != n:
            setattr(m, attr, int(n))
            changed[name] = (old, n)
        for cls in ("newborn", "survivor"):
            cattr = f"inmodel_repeats_{cls}"
            if not hasattr(m, cattr):
                continue
            cold = int(getattr(m, cattr))
            if cold != n:
                setattr(m, cattr, int(n))
                changed[f"{name}.{cls}"] = (cold, n)
    if changed:
        logger.info("[V9-STAGE %s] in-model repeats set on entry: %s.", tag,
                    ", ".join(f"{k} {o} -> {n}" for k, (o, n) in changed.items()))
    if untouched:
        logger.warning(
            "[V9-STAGE %s] in-model repeats requested for %s but these moves of "
            "a listed branch carry neither num_repeats nor num_repeat_proposals "
            "and were left alone: %s.", tag, want, untouched)
    return changed


#: The stage profile's ``inmodel_budget`` keys and which GB moves each one
#: reaches: the repeats per pass and the group window (in passes) go to the
#: PURE in-model moves (``is_rj_prop`` False: in_model, in_model_fstat,
#: in_model_removal), the newborn convergence window (in repeats) to the RJ
#: moves, whose in-model polish ``_converge_state_for`` reads it per block.
INMODEL_BUDGET_KEYS = {
    "num_repeat_proposals": "in_model",
    "inmodel_group_iters": "in_model",
    "inmodel_converge_iters": "rj",
}


def apply_inmodel_budget(gb_moves, budget, tag: str = "gb_search") -> dict:
    """Set a stage's in-model budget on the GB moves; restore the built one otherwise.

    Mike, 2026-10-16: "Gb search 2 should spend less time in the in-model
    mixing but maybe with some quick checker to make sure the logL is not
    seriously improving during those steps anymore"; gb_search_1 stays as
    is. The checkers already exist -- the per-sub-band group rule of the
    pure in-model moves (``[GB_IMGROUP]``: passes until no open sub-band
    gained D/2 over the window) and the per-row rule of the RJ moves'
    polish (``[GB_IMCONV]``: a row retires once its best gain stalls over
    the window) -- and both windows are FLOORS on how long a step runs, so
    a stage shortens its in-model time by shortening the windows, not by
    cutting the checkers out. Measured on the 1yr flagship (job 774,
    gb_search_2, 2026-10-16): the three dedicated in-model legs spent 3 x
    25-repeat warm-up passes (half of each 13-15 min leg) before any
    sub-band could shut, for a net cold lnL change per propose of about
    zero; the RJ moves' newborn polish ran 255 repeats per row against a
    250-repeat floor for a median gain of 0.2 nats (the top tenth +22).

    ``budget`` is ``{key: int}`` over :data:`INMODEL_BUDGET_KEYS` (any
    subset) or ``None``. Every key is written on every call: a key the
    stage gives takes its value, a key it omits goes back to the value the
    move was BUILT with (stashed on the move at the first call), so a stage
    without a budget runs the constructor's -- the move objects are shared
    across stages and a later stage must not inherit an earlier one's
    budget. Returns ``{"<move>.<key>": (old, new)}`` for what changed.
    """
    want = {}
    for k, v in dict(budget or {}).items():
        if k not in INMODEL_BUDGET_KEYS:
            raise ValueError(
                f"inmodel_budget key {k!r} is not one of {sorted(INMODEL_BUDGET_KEYS)}.")
        n = int(v)
        if n < 1:
            raise ValueError(f"inmodel_budget {k}={v!r} must be >= 1.")
        want[k] = n
    changed = {}
    for m in gb_moves:
        is_rj = bool(getattr(m, "is_rj_prop", False))
        built = getattr(m, "_inmodel_budget_built", None)
        if built is None:
            built = {k: int(getattr(m, k)) for k in INMODEL_BUDGET_KEYS if hasattr(m, k)}
            try:
                m._inmodel_budget_built = built
            except AttributeError:   # a frozen stand-in: nothing to restore later
                pass
        for k, family in INMODEL_BUDGET_KEYS.items():
            if (family == "rj") != is_rj or k not in built:
                continue
            new = int(want.get(k, built[k]))
            old = int(getattr(m, k))
            if old != new:
                setattr(m, k, new)
                changed[f"{getattr(m, 'name', type(m).__name__)}.{k}"] = (old, new)
    if changed:
        logger.info("[V9-STAGE %s] in-model budget set on entry (%s): %s.", tag,
                    "the stage's values" if want else "the built values restored",
                    ", ".join(f"{k} {o} -> {n}" for k, (o, n) in changed.items()))
    return changed


class ReplicaPERecipeStep(PERecipeStep):
    """full_pe's twin that ENDS when the large-scale metrics converge.

    User design 2026-10-03: after the ratchet stops (a criterion "focused
    entirely on the foreground convergence"), run "an exact replica of pe
    mode (and I mean exact) until the mean leaf count converges over say 10
    replica pe iterations. Then you switch to full PE and mark that as the
    start of sample taking. ... The only difference between replica pe and
    full pe is that the other sources (emris, mbhs, sobhbs) are still only run
    every five iterations to give priority to the gbs."

    This class owns only the STOP. Everything else -- the moves, the PE
    declarations (peak floor, in-model repeats, RJ flip fraction), the
    combine -- is :class:`PERecipeStep`, inherited unchanged; the source
    cadence lives in the stage's move list (``Move(every=N)``), not here.

    THE RULE. Over the last ``window`` stored rows of this stage six series
    are read from the store: the cold leaf count's min / mean / max over
    walkers and the cold log-likelihood's min / mean / max over walkers. Each
    gets a least-squares line; the stage ends when every line's total change
    over the window (slope x (window - 1)) is within ``leaf_tol`` leaves (the
    leaf series) or ``lnl_tol`` nats (the lnL series). A TREND, not a range:
    the per-iteration scatter (band swaps move 20-50 leaves between rungs,
    the in-model moves +-150 nats) is what a window average removes; a drift
    is what the typical-set relaxation shows. At least ``min_iters`` rows
    must be stored in the stage first (default: the window). One log line per
    iteration carries all six values and trends, so the log reads the same
    numbers the stop reads.
    """

    def __init__(self, *args, window: int = 10, leaf_tol: float = 10.0,
                 lnl_tol: float = 100.0, min_iters=None, plateau_branch: str = "gb",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.window = int(window)
        if self.window < 3:
            raise ValueError(f"ReplicaPERecipeStep: window={window} must be >= 3 rows.")
        self.leaf_tol = float(leaf_tol)
        self.lnl_tol = float(lnl_tol)
        if not (self.leaf_tol > 0.0 and self.lnl_tol > 0.0):
            raise ValueError(
                f"ReplicaPERecipeStep: leaf_tol={leaf_tol} and lnl_tol={lnl_tol} must be > 0.")
        self.min_iters = max(self.window, int(min_iters or 0))
        self.plateau_branch = str(plateau_branch)
        self._stage_start_iter = 0

    def setup_run(self, iteration, last_sample, sampler):
        super().setup_run(iteration, last_sample, sampler)
        self._stage_start_iter = int(iteration)

    @staticmethod
    def _cold_rows(arr):
        """``(rows, nwalkers)`` from whatever leading axes the store adds."""
        a = np.asarray(arr, dtype=float)
        while a.ndim > 2:
            a = a[:, 0]
        return a

    @staticmethod
    def trend(y) -> float:
        """Least-squares total change over the series: slope x (n - 1)."""
        y = np.asarray(y, dtype=float)
        n = int(y.size)
        if n < 2:
            return 0.0
        slope = np.polyfit(np.arange(n, dtype=float), y, 1)[0]
        return float(slope * (n - 1))

    def metrics(self, sampler):
        """``(leaves, lnl)``, each ``(rows_in_stage, nwalkers)`` on the cold rung."""
        be = sampler.backend
        start = int(self._stage_start_iter)
        nl = self._cold_rows(
            be.get_nleaves(branch_names=[self.plateau_branch], temp_index=0)[
                self.plateau_branch])[start:]
        ll = self._cold_rows(be.get_log_like())[start:]
        n = min(nl.shape[0], ll.shape[0])
        return nl[:n], ll[:n]

    def stopping_function(self, i, sample, sampler) -> bool:
        tag = self.stage_name or "replica_pe"
        try:
            nl, ll = self.metrics(sampler)
        except Exception as exc:  # noqa: BLE001 -- a stop rule must never kill the run
            logger.warning(
                "[REPLICA_PE %s] metrics unreadable from the store (%r); holding the "
                "stage open.", tag, exc)
            return False
        n = int(nl.shape[0])
        start = int(self._stage_start_iter)
        if n < self.min_iters:
            logger.info("[REPLICA_PE %s] %d row(s) in-stage; the convergence check needs %d.",
                        tag, n, self.min_iters)
            return False
        W = self.window
        nlw, llw = nl[-W:], ll[-W:]
        series = [("leaves min", nlw.min(axis=1), self.leaf_tol),
                  ("leaves mean", nlw.mean(axis=1), self.leaf_tol),
                  ("leaves max", nlw.max(axis=1), self.leaf_tol),
                  ("lnL min", llw.min(axis=1), self.lnl_tol),
                  ("lnL mean", llw.mean(axis=1), self.lnl_tol),
                  ("lnL max", llw.max(axis=1), self.lnl_tol)]
        trends = [self.trend(y) for _, y, _ in series]
        ok = [abs(t) <= tol for t, (_, _, tol) in zip(trends, series)]
        logger.info(
            "[REPLICA_PE %s] stored row %d (%d in-stage): leaves min/mean/max %.0f / %.1f / "
            "%.0f, lnL min/mean/max %.1f / %.1f / %.1f | trend over the last %d rows: leaves "
            "%+.1f / %+.1f / %+.1f (tol %g), lnL %+.1f / %+.1f / %+.1f (tol %g) | converged "
            "%d of 6.", tag, start + n - 1, n, nlw[-1].min(), nlw[-1].mean(), nlw[-1].max(),
            llw[-1].min(), llw[-1].mean(), llw[-1].max(), W, trends[0], trends[1], trends[2],
            self.leaf_tol, trends[3], trends[4], trends[5], self.lnl_tol, sum(ok))
        if all(ok):
            logger.info(
                "[REPLICA_PE %s] CONVERGED: every large-scale metric is flat over the last %d "
                "rows -- the stage ends; the next stage begins at stored row %d, THE START OF "
                "SAMPLE TAKING.", tag, W, start + n)
            return True
        return False


def _cap_ramp_pending_total(moves) -> int:
    """Sum of ``_cap_ramp_pending`` over a move tree (GFCombineMove nests).

    GB moves publish the count of cap cells actively counting toward an
    increment (armed, mid patience window, occupied at cap, below ceiling)
    at the end of every cap update; anything without the attribute
    contributes 0. Weighted ``(move, weight)`` entries are unwrapped.
    """
    total = 0
    for m in list(moves or []):
        if isinstance(m, (tuple, list)) and m:
            m = m[0]
        total += int(getattr(m, "_cap_ramp_pending", 0) or 0)
        total += _cap_ramp_pending_total(getattr(m, "moves", None))
    return total


def _cap_headroom_deficit_total(moves) -> int:
    """Sum of ``_cap_headroom_deficit`` over a move tree (GFCombineMove nests).

    GB moves publish, while ``GB_SEARCH_CAP_HEADROOM`` is armed, the count of
    OCCUPIED sub-bands that lack the required cap headroom (cap - maxocc <
    headroom, below the ceiling); anything without the attribute contributes
    0. Weighted ``(move, weight)`` entries are unwrapped.
    """
    total = 0
    for m in list(moves or []):
        if isinstance(m, (tuple, list)) and m:
            m = m[0]
        total += int(getattr(m, "_cap_headroom_deficit", 0) or 0)
        total += _cap_headroom_deficit_total(getattr(m, "moves", None))
    return total


def band_shutoff_w_pending_total(moves) -> int:
    """Occupied (walker, band) pairs not yet shut off, over a move tree.

    The stage-convergence counter for the per-(walker, band) RJ shutoff
    valve (``GB_SEARCH_BAND_SHUTOFF_PER_WALKER``), in the same shape as
    :func:`_cap_ramp_pending_total`: ``0`` means fully converged.
    ``GFCombineMove`` nests and weighted ``(move, weight)`` entries are
    unwrapped, as there.

    THE SANCTIONED CRITERION, in the user's own words (ruling 2026-09-24):

        **"every occupied (walker, band) pair has shut off"**

    OCCUPIED is load-bearing, and the word is why this ruling was asked for
    explicitly. An EMPTY (walker, band) can never shut -- the valve requires
    a non-zero source count, because a band that has found nothing has not
    plateaued, it has not started -- so the looser phrasing "all bands and
    walkers are shut off", which is what the original request said and what
    will be in people's heads, would be unreachable in any run with an empty
    band, i.e. every run. The difference is invisible until someone wonders
    why a stage never ends, which is exactly why it is written out here.

    Empty pairs remain the nleaves-plateau gate's business, which is why a
    stage composes this WITH that gate rather than replacing it.

    A move that does not publish the counter, or has not computed it yet,
    contributes 0: the feature is off there, so it has no opinion and must
    not hold a stage open. ⚠ THE COROLLARY: with the valve off everywhere
    this returns 0, i.e. "converged", on the very first check. A stage that
    uses it as its ONLY criterion must therefore also confirm the feature
    is armed -- pair it with the nleaves-plateau gate (which is what the
    existing stopping_function already is) rather than replacing that gate
    outright.

    ⚠ COUNTED ONCE PER VALVE, NOT ONCE PER MOVE. Every armed RJ move in the
    cycle publishes its own ``_shutoff_w_pending``, but they are all reading
    the SAME shared ``band_info["band_rj_shutoff_w"]`` table -- so a naive sum
    over a 4-RJ-move stage reports 4x the real number of pending pairs. The
    ``== 0`` gate survives that (4x0 is still 0), which is exactly why it went
    unnoticed; the human-facing count and the monitor's "pending" trace do
    not. Moves are deduplicated by the identity of the table they publish
    against, falling back to the move's own identity when it exposes none.
    """
    return _band_shutoff_w_pending(moves, set())


def _band_shutoff_w_pending(moves, seen) -> int:
    total = 0
    for m in list(moves or []):
        if isinstance(m, (tuple, list)) and m:
            m = m[0]
        pending = getattr(m, "_shutoff_w_pending", None)
        if pending is not None:
            table = getattr(m, "_rj_band_shutoff_w", None)
            key = id(table) if table is not None else id(m)
            if key not in seen:
                seen.add(key)
                total += int(pending)
        total += _band_shutoff_w_pending(getattr(m, "moves", None), seen)
    return total


def band_shutoff_w_armed(moves) -> bool:
    """Is the per-(walker, band) RJ valve live on ANY move in the tree?

    The companion to :func:`band_shutoff_w_pending_total`, and the guard
    that makes it safe to use as a stage criterion: the pending count is 0
    both when everything has converged and when nothing is armed, and only
    this can tell those apart.
    """
    for m in list(moves or []):
        if isinstance(m, (tuple, list)) and m:
            m = m[0]
        if getattr(m, "_shutoff_w_pending", None) is not None:
            return True
        if band_shutoff_w_armed(getattr(m, "moves", None)):
            return True
    return False


def begin_search_recipe_step(moves, serial) -> None:
    """Tell every move in the tree that a new recipe step has begun.

    Walks a move tree the same way :func:`_arm_cap_headroom_grant` does
    (``GFCombineMove`` nests; weighted ``(move, weight)`` entries are
    unwrapped), calling ``begin_recipe_step(serial)`` on anything that
    publishes one. Moves without the method are skipped, so this is inert
    for every non-GB branch and for any GB build predating the feature.

    Today the one consumer is the per-(walker, band) RJ shutoff valve,
    which is SCOPED TO A STEP: a (walker, band) that has stopped paying in
    lnL under this step's moves, caps and floors has said nothing about
    the next step's, so the valve is released here rather than allowed to
    persist.

    ``serial`` identifies the STEP, and :meth:`Recipe._announce_recipe_step`
    passes the step INDEX rather than an iteration number precisely so it
    stays the same across a mid-step restart. Only a CHANGE releases, so
    re-announcing an already-active step does not wipe a valve that step
    earned -- which is what makes "hold for the whole step, release at the
    next one" true across a resume and not only within one process.
    """
    for m in list(moves or []):
        if isinstance(m, (tuple, list)) and m:
            m = m[0]
        fn = getattr(m, "begin_recipe_step", None)
        if callable(fn):
            fn(serial)
        begin_search_recipe_step(getattr(m, "moves", None), serial)


def iter_move_tree(moves):
    """Yield every move in a (possibly nested) move tree, once, in order.

    ``GFCombineMove`` nests its children under ``moves``; a weighted entry is
    a ``(move, weight)`` tuple. Same unwrapping as
    :func:`_cap_ramp_pending_total` and friends, factored out because the v9
    stage profile needs to WRITE to the tree rather than sum over it.
    """
    for m in list(moves or []):
        if isinstance(m, (tuple, list)) and m:
            m = m[0]
        if m is None:
            continue
        yield m
        yield from iter_move_tree(getattr(m, "moves", None))


def gb_moves_in_tree(moves):
    """Every GB-branch GBSpecial move in the tree (VGB and non-GB excluded).

    The test is ``branch_name == "gb"`` plus the presence of the SNR-boundary
    attribute, i.e. "a GB band move", which is what the per-stage profile is
    allowed to write to. ⚠ The VGB moves carry the SAME attribute name with
    their own value (``VGB_OPT_SNR_LIMIT``, default 0.0 = off); writing the GB
    floor onto them would silently arm an SNR gate the VGB branch has never
    had, on 55 KNOWN sources that must not be gated at all.
    """
    for m in iter_move_tree(moves):
        if (getattr(m, "branch_name", None) == "gb"
                and hasattr(m, "opt_snr_rej_samp_limit")):
            yield m


def force_fstat_refit(moves, serial, reason: str = "", hard: bool = False) -> int:
    """Arm a forced fresh-epoch F-stat refit on every grid move in the tree.

    Returns how many moves were armed (0 when the tree carries no F-stat
    grid move, e.g. a GB-less recipe). See
    :meth:`~lisatools.globalfit.moves.gbspecialstretch.GBSpecialRJFStatGridMove.arm_fstat_refit`
    for why a refit, and not an environment change, is the mechanism.

    ``hard=True`` arms the age-ignoring form (the galfor ratchet: the noise
    curve changed, so the latest epoch is wrong however fresh it is).
    """
    n = 0
    for m in iter_move_tree(moves):
        fn = getattr(m, "arm_fstat_refit", None)
        if callable(fn):
            if hard:
                fn(serial, reason, ignore_age=True)
            else:
                fn(serial, reason)
            n += 1
    return n


def reset_band_logl_max(state, branch: str = "gb") -> int:
    """Reset the valve's all-time per-(walker, band) cold-lnL maximum to -inf.

    User ruling 2026-09-30 for gb_search_3 ("instead of starting from the
    stored value, let's reset that"): the max carried in from the fixed-noise
    stages was earned under a DIFFERENT noise curve. Once the foreground
    moves (the stage's own noise search, or a ratchet nudge) the cold lnL of
    every band changes scale, a stored max can sit above anything the band
    can now reach, and the valve reads "no improvement" and shuts pairs that
    were never given a fair window -- 6mo job 672's stage 3 completed on the
    shutoff rule after three iterations for exactly this reason. Written IN
    PLACE into ``state.sub_states[branch].band_info`` (the arrays the
    judge holds live references to and the saver persists), so the very next
    judge re-learns the max under the noise now in force; the streaks reset
    themselves at that first judge because -inf + tol is always beaten.

    Returns the number of entries reset, 0 when the state carries no such
    array (valve off, or a fake state).
    """
    try:
        bi = state.sub_states[branch].band_info
        arr = bi["band_cold_logl_max_w"]
    except (AttributeError, KeyError, TypeError):
        return 0
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0
    arr[...] = -np.inf
    return int(arr.size)


def revive_band_shutoff_on_state(state, branch: str = "gb") -> int:
    """Revive the per-band BARREN valve's persisted record on the state.

    The state-side twin of ``GBSpecialBase._band_shutoff_revive`` (same
    effect on the same fields: shut set cleared, streaks 0, previous
    occupancy -1, iterations-since-revive 0; the F-stat epoch is untouched),
    for ``reset_valves`` at a fresh stage entry: a process that has not yet
    adopted the record adopts it from ``band_info`` at its first propose, and
    would otherwise re-install the previous stage's shut set. Returns how many
    bands were off; 0 when the state carries no record.
    """
    try:
        bi = state.sub_states[branch].band_info
    except (AttributeError, KeyError, TypeError):
        return 0
    if not isinstance(bi, dict) or bi.get("band_rj_shutoff") is None:
        return 0
    shut = np.asarray(bi["band_rj_shutoff"])
    n = int(np.count_nonzero(shut))
    bi["band_rj_shutoff"] = np.zeros(shut.shape, dtype=bool)
    for key, fill in (("band_occ_streak", 0), ("band_occ_last", -1),
                      ("band_shutoff_since_revive", 0)):
        if bi.get(key) is not None:
            bi[key] = np.full(np.shape(bi[key]), fill, dtype=np.int64)
    return n


def release_band_shutoff_window(state, serial, branch: str = "gb"):
    """Release the level-3 valve ON THE STATE at recipe-step entry.

    Returns ``(n_reopened, released)``.

    The per-(walker, band) RJ shutoff valve is scoped to a recipe step: a
    stored table earned under another step is released, not honoured. Until
    2026-10-01 that release lived ONLY in the RJ moves' bind path
    (``_arm_search_stage`` on rj_fstat_search / rj_warm_search, at THEIR
    first propose). The pure in-model moves bind the same table READ-ONLY
    for their pick mask -- no stamp check, no release -- so the first
    non-RJ GB move of a new step read the previous step's table. 6mo job
    675 (the first ratcheted relaunch): gb_search_2 had ended with every
    occupied pair shut (that is the stage-end rule), the gated noise head's
    in-model follow-up was the first GB move of gb_search_3, its pick mask
    excluded every shut pair, and it ran ZERO proposals
    (``[GB_ACCEPT in_model] in-model cold 0/0``, one 1 s "pass" with every
    occupied sub-band static) -- the one pass the ratchet design exists to
    run before any RJ move sees the nudged residual. Job 672 did not show it
    because its state came from a stage-3 checkpoint whose stamp matched.

    Same rule as the bind path, applied earlier and on the live arrays the
    moves will bind: stamp differs from ``serial`` (the unset sentinel
    included) -> clear the shut booleans and the window (peak / streak /
    resets; the all-time max is untouched by the 2026-09-27 ruling) and
    re-stamp; same stamp (a resume inside the step) -> honour it. In place,
    so the saver persists the release and the RJ move's own bind then sees a
    matching stamp and releases nothing twice.
    """
    if serial is None:
        return 0, False
    try:
        bi = state.sub_states[branch].band_info
    except (AttributeError, KeyError, TypeError):
        return 0, False
    if not isinstance(bi, dict):
        return 0, False
    shut = bi.get("band_rj_shutoff_w")
    stamp = bi.get("band_shutoff_w_step")
    if shut is None or stamp is None:
        return 0, False
    shut = np.asarray(shut)
    stamp = np.asarray(stamp)
    if stamp.size == 0 or int(stamp.ravel()[0]) == int(serial):
        return 0, False
    from .moves.gbspecialstretch import cold_band_lnl_from_band_info

    view = cold_band_lnl_from_band_info(bi, shut.shape)
    if view is not None:
        view.release()
    n = int(np.count_nonzero(shut))
    shut[...] = False
    stamp.ravel()[0] = int(serial)
    return n, True


def apply_shutoff_floor_to_state(state, fmin_hz, band_edges=None, branch: str = "gb"):
    """Reopen the live valve below ``fmin_hz`` NOW, on the state the moves bind.

    Returns ``(n_reopened, n_exempt_bands)``; ``(0, 0)`` when there is nothing
    to do or not enough to do it with.

    The recipe-step side of the ratchet's RJ shutoff floor. The move side
    (``_update_search_band_shutoff``) applies the floor at every JUDGMENT,
    which is the END of an rj_fstat_search propose -- after that propose's
    F-stat refit. 6mo job 695 (relaunched with the floor onto a valve 82.9 %
    shut): the first nudge's hard refit (epoch 37) skipped every band shut on
    all four walkers (``fstat_band_skip_for``) and produced 5,701 peaks over
    the bands some walker still had open; the floor then reopened 1,010
    pairs at the judgment forty minutes later, and the next refit (epoch 38)
    found 14,119. The first nudge/release pair ran its F-stat births on a
    sixth of the bands. Same arrays and same in-place rule as
    :func:`release_band_shutoff_window`, so the saver persists it and the
    move's own judgment then finds nothing left to reopen.
    """
    if state is None or fmin_hz is None or float(fmin_hz) <= 0.0:
        return 0, 0
    try:
        bi = state.sub_states[branch].band_info
    except (AttributeError, KeyError, TypeError):
        return 0, 0
    if not isinstance(bi, dict):
        return 0, 0
    shut = bi.get("band_rj_shutoff_w")
    if shut is None:
        return 0, 0
    shut = np.asarray(shut)
    if shut.ndim != 2:
        return 0, 0
    edges = band_edges if band_edges is not None else bi.get("band_edges")
    if edges is None:
        logger.warning(
            "[GALFOR_RATCHET] RJ shutoff floor: no band_edges on the GB moves or in "
            "band_info -- the live valve is NOT reopened now; the move's next "
            "judgment will do it (after that propose's refit).")
        return 0, 0
    _get = getattr(edges, "get", None)           # cupy -> host
    edges = np.asarray(_get() if callable(_get) else edges, dtype=float).reshape(-1)
    if edges.size != shut.shape[1] + 1:
        logger.warning(
            "[GALFOR_RATCHET] RJ shutoff floor: band_edges (%d) do not match the "
            "valve's %d bands -- the live valve is NOT reopened now.",
            edges.size, shut.shape[1])
        return 0, 0
    exempt = (0.5 * (edges[:-1] + edges[1:])) < float(fmin_hz)
    streak = bi.get("band_shutoff_streak_w")
    if streak is not None:
        streak = np.asarray(streak)
        if streak.shape != shut.shape:
            streak = None
    from .moves.gbspecialstretch import apply_shutoff_floor

    converged = np.zeros(shut.shape, dtype=bool)     # the judgment recomputes it
    n = apply_shutoff_floor(converged, shut, streak, exempt)
    return int(n), int(exempt.sum())


def _arm_cap_headroom_grant(moves) -> None:
    """Set the one-shot ``_grant_cap_headroom`` flag on every move in the tree
    that publishes a headroom deficit, so the next cap update grants +1 slot to
    saturated occupied cells (letting the search try to fill the new headroom)."""
    for m in list(moves or []):
        if isinstance(m, (tuple, list)) and m:
            m = m[0]
        if hasattr(m, "_cap_headroom_deficit"):
            m._grant_cap_headroom = True
        _arm_cap_headroom_grant(getattr(m, "moves", None))


class RJRecipeStep(BaseRecipeStep):
    """Reversible-jump recipe step that stops once GB leaf count plateaus.

    Args:
        convergence_iter: Window length used to compare older vs newer
            cold-chain max leaf counts.
        thin_by: Forwarded thinning factor applied to the sampler.
    """

    def __init__(
        self,
        *args,
        convergence_iter: int = 5,
        thin_by: int = 1,
        plateau_branch: str = "gb",
        convergence_fn: typing.Callable | None = None,
        **kwargs
    ):
        BaseRecipeStep.__init__(self, *args, **kwargs)
        self.convergence_iter = convergence_iter
        self.thin_by = thin_by
        # Branch whose cold-chain leaf count is monitored for the plateau test.
        # Defaults to ``"gb"`` (the historical hardcoded value); set it per recipe
        # in a settings file to reuse this step for another RJ branch.
        self.plateau_branch = plateau_branch
        # Optional full override of the stopping criterion (``(i, sample, sampler)
        # -> bool``); lets a settings file define convergence per recipe without
        # editing this class.
        self.convergence_fn = convergence_fn

    def stopping_function(
        self,
        i,
        sample,
        sampler: GlobalFitEngine
    ) -> bool:
        """Stop when the cold chain stops growing in number of leaves."""

        if self.convergence_fn is not None:
            return self.convergence_fn(i, sample, sampler)

        if not hasattr(self, "st"):
            self.st = time.perf_counter()

        current_iter = sampler.backend.iteration

        assert isinstance(current_iter, (int, np.integer))

        stop = False
        _start = int(getattr(self, "_stage_start_iter", 0))
        # The old-vs-new window comparison needs at least one full window on
        # each side WITHIN THIS STAGE.
        if current_iter - _start > 2 * self.convergence_iter:
            #? Actual convergence should be related to the same number of sources above SNR XX for Y itterations
            nleaves_cc = sampler.backend.get_nleaves(
                branch_names=[self.plateau_branch], temp_index=0
            )[self.plateau_branch][_start:]

            # do not include most recent
            nleaves_cc_max_old = nleaves_cc[:-self.convergence_iter].max()
            nleaves_cc_max_new = nleaves_cc[-self.convergence_iter:].max()

            if nleaves_cc_max_old >= nleaves_cc_max_new:
                stop = True

            else:
                stop = False

            # A search that has not found its FIRST source has not
            # plateaued -- it has not started (deliberate: an RJ search
            # stage never advances at zero leaves).
            if nleaves_cc_max_new <= 0:
                stop = False

            
            dur = (time.perf_counter() - self.st) / 3600.0  # hours
            logger.info(f"Previous nleaves: {nleaves_cc_max_old} --> new nleaves: {nleaves_cc_max_new}")
            logger.info(f"TIME SINCE START: {dur} hours")

        # CAP-QUIESCENCE veto (user-approved 2026-08-26): under a cap
        # blockade the cold leaf count plateaus BY CONSTRUCTION, so the
        # plateau alone cannot distinguish "converged" from "cap-starved"
        # (the highf-grid probe's search ended at iteration 11 with the
        # flagship cell 3/5 through its patience window -- the increment
        # that would have unblocked truth-side births never got to fire).
        # While any move reports cap cells mid patience window, the stage
        # holds open: the coming increment can re-open growth and break
        # the plateau honestly. GB_SEARCH_CAP_QUIESCENT=0 restores the
        # bare plateau rule.
        if stop and os.environ.get("GB_SEARCH_CAP_QUIESCENT", "1") == "1":
            _pending = _cap_ramp_pending_total(getattr(sampler, "moves", None))
            if _pending:
                logger.info(
                    "nleaves plateau reached but %d cap cell(s) are mid "
                    "patience window -- holding the stage open for the cap "
                    "ramp.", _pending,
                )
                stop = False

        # CAP-HEADROOM veto (user 2026-09-06): additionally hold the stage
        # while any OCCUPIED sub-band lacks the required cap headroom (cap -
        # maxocc < GB_SEARCH_CAP_HEADROOM, below the ceiling). The lnL-gated
        # ramp can leave a converged-at-cap band saturated and invisible to
        # the ramp-pending veto above; this confirms every occupied band was
        # given >= headroom free slots and chose not to fill them. Arms the
        # one-shot grant so the caps ramp toward the headroom over the next
        # iterations -- the search either fills the new slots (plateau breaks,
        # search continues honestly) or leaves them (converged). Off by
        # default (GB_SEARCH_CAP_HEADROOM=0) -> deficit ignored, bit-identical.
        if stop and int(os.environ.get("GB_SEARCH_CAP_HEADROOM", "0") or 0) > 0:
            _deficit = _cap_headroom_deficit_total(getattr(sampler, "moves", None))
            if _deficit:
                logger.info(
                    "nleaves plateau + cap-quiescent, but %d occupied "
                    "sub-band(s) lack the required cap headroom -- arming a "
                    "+1 grant and holding the stage open.", _deficit,
                )
                _arm_cap_headroom_grant(getattr(sampler, "moves", None))
                stop = False

        return stop
        
    def setup_run(
        self,
        iteration,
        last_sample,
        sampler: GlobalFitEngine
    ):
        """Configure the sampler for this RJ recipe step (moves, weights, thinning)."""
        # Stage-scope the plateau window: the backend's nleaves history
        # spans the WHOLE run, so without this anchor the first check
        # compares pre-stage iterations (0 leaves during the noise stages)
        # against the stage's own start and trips 0 >= 0 immediately --
        # the 3-month run's gb_search advanced after ONE check
        # (found 2026-08-13).
        self._stage_start_iter = int(sampler.backend.iteration)
        # ---- GF_PERSIST_STAGE_START (user ask 2026-09-26) -----------------
        # "Can we track things like _stage_start_iter in the backend, so a
        # resume is a real resume?" -- yes, and this is why it matters: the
        # plateau test above needs current_iter - _start > 2*convergence_iter,
        # i.e. 41 iterations at GB_PLATEAU_ITERS=20, and the line above resets
        # _start to the LIVE iteration on every launch. A run restarted every
        # few hours therefore never accumulates 41 in-stage iterations and the
        # stage can never advance on the plateau. Measured on the 3-month v9
        # run: job 640 started gb_search_1 at iteration 25, job 644 resumed at
        # 30 and reset the clock there, pushing the earliest possible stage end
        # from iteration 66 to 71.
        #
        # CODE DEFAULT OFF (user ruling, same day: "add that tracking as an
        # option default off for now so it does not mess up the current
        # runs"). Unset, this block is a no-op and the clock restarts exactly
        # as before. The v9 production launchers export it ON since
        # 2026-10-07 (user ruling: "keep GF_PERSIST_STAGE_START, but it should
        # reset at the end of a stage" -- the reset is
        # ``GFHDFBackend.completed_recipe_step``, which rewrites the next
        # step's start_iteration at every boundary; here a stored start PAST
        # the live iteration, i.e. a rewound store, is ignored and re-stamped).
        # Compute ranks run this method too (run.py stamps every step against
        # the shell engine), but their engine carries eryn's in-memory Backend
        # (no stage_start_iteration / stamp_stage_start), so the block is
        # inert there and the store is never touched off the head.
        if os.environ.get("GF_PERSIST_STAGE_START", "0").strip() in (
                "1", "true", "True", "yes", "on"):
            _name = getattr(self, "stage_name", None) or getattr(
                self, "_current_step_name", None)
            if _name:
                _stored = None
                if hasattr(sampler.backend, "stage_start_iteration"):
                    _stored = sampler.backend.stage_start_iteration(_name)
                if _stored is not None and int(_stored) > self._stage_start_iter:
                    logger.warning(
                        "[STAGE-START] %s: stored start iteration %d is PAST "
                        "the live iteration %d (a rewound store?) -- ignoring "
                        "it and re-stamping the live iteration.",
                        _name, int(_stored), self._stage_start_iter)
                    _stored = None
                if _stored is not None:
                    self._stage_start_iter = int(_stored)
                    logger.info(
                        "[STAGE-START] %s: restored start iteration %d from "
                        "the backend (live iteration %d) -- the convergence "
                        "clock survives this resume.",
                        _name, self._stage_start_iter,
                        int(sampler.backend.iteration))
                elif hasattr(sampler.backend, "stamp_stage_start"):
                    sampler.backend.stamp_stage_start(
                        _name, self._stage_start_iter)
                    logger.info(
                        "[STAGE-START] %s: stamped start iteration %d.",
                        _name, self._stage_start_iter)
        # TODO: maybe make this the default setup
        sampler.moves = self.moves
        sampler.weights = self.weights
        sampler.yield_step = self.thin_by
        sampler.checkpoint_step = self.thin_by
        # sampler.override_thin_by = self.thin_by --> # TODO check this one
        
        for move in self.moves:
            if sampler.periodic is not None and move.periodic is None:
                logger.debug(f"Setting periodicity of move {move} to {sampler.periodic}")
                move.periodic = sampler.periodic
            if sampler.temperature_control is not None and move.temperature_control is None:
                logger.debug(f"Setting temperature control of move {move} to {sampler.temperature_control}")
                move.temperature_control = sampler.temperature_control
            
            # TODO: do we also need to set these? I think the current settings setup has ntemps covered, not sure about temp_cntrl
            # move.ntemps = sampler.ntemps


class SearchStageProfileStep(RJRecipeStep):
    """An RJ step that also owns a per-stage GB SEARCH PROFILE (v9).

    The v9 restructure (user spec 2026-09-24) replaces the single
    ``gb_search`` stage with three, differing ONLY in four knobs:

    ======================  =========  =========  =========  ===========
    stage                   noise      phase max  opt SNR    F-stat peak
    ======================  =========  =========  =========  ===========
    ``gb_search_1``         fixed      **on**     **8**      **8**
    ``gb_search_2``         fixed      off        5          6.25
    ``gb_search_3``         sampled    off        5          6.25
    ======================  =========  =========  =========  ===========

    Noise fixed-vs-sampled is a MOVE-LIST difference (an unsampled branch
    does not move), so it is handled by the driver. The other three are
    per-move attributes, and they are applied HERE, at stage entry, rather
    than by building three named copies of every GB move: ``build_gb_moves``
    builds each stock move once and ``Move("rj_fstat_search")`` resolves the
    SAME object in all three stages. Three copies would triple the
    F-stat-carrying RJ instances -- each of which owns an epoch cache
    lifecycle -- for no sampling benefit whatsoever.

    WHAT IS WRITTEN, AND TO WHAT:

    * ``opt_snr_rej_samp_limit`` -> **every GB-branch band move** in the
      stage, including the pure in-model ones. It is the GB SNR PRIOR
      BOUNDARY, not a per-move heuristic: a birth move and an in-model move
      disagreeing about where the prior ends would let a source be walked
      into a region the move that birthed it treats as zero-prior.
    * ``phase_maximize`` -> the **RJ moves only** (``is_rj_prop``). The pure
      in-model move is constructed ``phase_maximize=False`` deliberately
      ("in-model scoring is at the actual phase") and that is not a stage
      choice to make; phase maximization is a BIRTH heuristic.
    * ``peak_min_snr`` -> the process-wide F-stat peak floor
      (:func:`lisatools.sampling.fstat_proposal.set_peak_min_F_override`)
      plus, on a CHANGE, a forced fresh-epoch refit
      (:func:`force_fstat_refit`). ⚠ The floor is consumed when the grid is
      FITTED and stamped into the stage-B cache, whose loader refuses a
      mismatch -- so moving it without a refit either silently reuses the
      old peak list or hard-refuses. See ``arm_fstat_refit``.

    Every mutation prints one ``[V9-STAGE]`` line naming stage, knob, old and
    new value. This machinery is otherwise invisible, and a profile that
    silently failed to apply would look exactly like a normal run.

    STOPPING. The stage ends when the nleaves plateau AND the
    per-(walker, band) RJ shutoff valve both say so -- see
    :meth:`stopping_function`.

    Args:
        profile: ``{"phase_maximize": bool|None, "opt_snr": float|None,
            "peak_min_snr": float|None}``. ``None`` for a key means "leave
            it alone", which is how a stage opts out of owning a knob.
            Entry-time keys: ``reset_band_max`` (every entry),
            ``prior_births`` (move-scoped), ``reset_valves`` (fresh entry
            only; :meth:`_reset_valves_at_entry`).
        stage_name: label for the log lines.
    """

    def __init__(self, *args, profile: typing.Optional[dict] = None,
                 stage_name: str = "", ratchet=None, ratchet_delta=None,
                 ratchet_min_gain: float = 0.0, ratchet_min_nudges: int = 0,
                 ratchet_stop_rule: str = "gain", ratchet_min_drop: float = 0.0,
                 ratchet_drop_band=None, ratchet_end_stage_on_stop: bool = False,
                 legs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = dict(profile or {})
        _unknown = sorted(set(self.profile) - {
            "phase_maximize", "opt_snr", "peak_min_snr", "reset_band_max",
            "prior_births", "reset_valves", "inmodel_budget"})
        if _unknown:
            raise ValueError(
                f"SearchStageProfileStep({stage_name!r}): unknown profile "
                f"keys {_unknown}. A misspelled key would be silently "
                f"ignored, which is exactly the failure this refuses."
            )
        self.stage_name = stage_name
        #: Step serial the profile was last applied for (idempotence across a
        #: mid-step resume, which re-announces the ACTIVE step).
        self._profile_serial = None
        #: Whether this step has ever been announced. Read by the
        #: ``stopping_function`` guard below.
        self._profile_applied = False
        # ---- the galfor RATCHET (user design 2026-09-30) -------------------
        # ``ratchet`` is a ``noise_ratchet.RatchetSchedule`` (or None = off);
        # ``ratchet_delta`` the per-nudge shift of the galfor coordinates in
        # the sampled basis. The schedule is keyed on the STAGE-LOCAL
        # iteration (backend iteration - stage start), so a resume lands in
        # the right phase instead of nudging again -- see _drive_ratchet.
        self.ratchet = ratchet
        self.ratchet_delta = (None if ratchet_delta is None
                              else np.asarray(ratchet_delta, dtype=float))
        # ``ratchet_min_gain`` (nats, 0 = off): the data-driven stop -- after a
        # release whose max cold lnL gained less than this over the previous
        # release, no more nudges (user design 2026-10-02; _ratchet_check_gain).
        self.ratchet_min_gain = float(ratchet_min_gain or 0.0)
        # ``ratchet_min_nudges``: the gain rule may not stop the ratchet before
        # this many nudges have run IN THIS PROCESS (user ruling 2026-10-02:
        # "I want to force at least 1 more nudge"). 0 = no floor. Counted on
        # ``_ratchet_nudges``, which _drive_ratchet increments per nudge; a
        # relaunch starts the count again.
        self.ratchet_min_nudges = int(ratchet_min_nudges or 0)
        # ``ratchet_stop_rule`` (user ruling 2026-10-03, "adjust away from the
        # raw lnL gain"): what ends the ratchet before the cycle ceiling --
        # "galfor" = the released foreground must still FALL by
        # ``ratchet_min_drop`` (fraction, median walker, over
        # ``ratchet_drop_band`` Hz) between releases; "gain" = the 10-02 lnL
        # rule above; "off" = the ceiling alone. The constructor default stays
        # "gain" (every earlier caller passed ratchet_min_gain); the launcher's
        # default is "galfor" (noise_ratchet.stop_rule_from_env).
        from .noise_ratchet import STOP_RULES

        rule = str(ratchet_stop_rule or "gain").strip().lower()
        if rule not in STOP_RULES:
            raise ValueError(
                f"SearchStageProfileStep({stage_name!r}): ratchet_stop_rule={rule!r} "
                f"is not one of {STOP_RULES}.")
        self.ratchet_stop_rule = rule
        self.ratchet_min_drop = float(ratchet_min_drop or 0.0)
        if rule == "galfor" and ratchet is not None and not 0.0 < self.ratchet_min_drop < 1.0:
            raise ValueError(
                f"SearchStageProfileStep({stage_name!r}): ratchet_stop_rule='galfor' needs "
                f"0 < ratchet_min_drop < 1 (a fraction), got {self.ratchet_min_drop}.")
        band = (3e-3, 5e-3) if ratchet_drop_band is None else tuple(
            float(x) for x in ratchet_drop_band)
        if len(band) != 2 or not (0.0 < band[0] < band[1]):
            raise ValueError(
                f"SearchStageProfileStep({stage_name!r}): ratchet_drop_band must be "
                f"(f_lo, f_hi) Hz with 0 < f_lo < f_hi, got {ratchet_drop_band!r}.")
        self.ratchet_drop_band = band
        # ``ratchet_end_stage_on_stop`` (user design 2026-10-03: "make gb search
        # 3 only the ratcheting and foreground convergence"): when the stop
        # rule ends the ratchet, the STAGE ends with it instead of waiting for
        # the per-(walker, band) valve; the replica-PE stage follows.
        self.ratchet_end_stage_on_stop = bool(ratchet_end_stage_on_stop)
        self._ratchet_end_stage = False
        self._ratchet_stopped = False
        self._ratchet_last_release_max = None
        self._ratchet_release_maxes = []
        self._ratchet_last_release_galfor = None
        self._ratchet_release_drops = []
        self._ratchet_last_k = None
        self._ratchet_pre_nudge = None
        self._ratchet_pre_nudge_w = None
        # ---- search LEGS (user design 2026-09-30) -------------------------
        # One stored row per leg of the cycle; the stage combine carries the
        # cursor (``gf_legs``), this step positions it at entry from the last
        # row's leg-ender NAME. See lisatools.globalfit.legs.
        self.legs = bool(legs)

    # ---- profile application ----------------------------------------------

    def note_recipe_step(self, serial) -> None:
        """Apply the profile for recipe step ``serial`` (idempotent per step).

        Called by :meth:`Recipe._announce_recipe_step` immediately after
        ``setup_run`` and before the step's first propose. It lives here
        rather than in ``setup_run`` for the same reason the valve's release
        does: a step does not know its own index and the forced refit needs
        one that is stable across a mid-step restart.
        """
        if serial is not None and serial == self._profile_serial:
            return
        self._profile_serial = serial
        self._profile_applied = True
        self._apply_profile(serial)
        # Release the level-3 valve on the LIVE state before the step's first
        # move (6mo job 675: the gated noise head's in-model pass, a non-RJ
        # move, read gb_search_2's all-shut table and picked nothing).
        _n_open, _released = release_band_shutoff_window(
            getattr(self, "_ratchet_last_sample", None), serial)
        if _released:
            logger.info(
                "[V9-STAGE %s] per-walker RJ valve RELEASED at step entry "
                "(stored step != %s): %d (walker, band) pairs reopened and the "
                "window cleared BEFORE the step's first move, so a non-RJ move "
                "running first (the gated noise head's in-model pass) picks "
                "from this step's table, not the previous step's.",
                self.stage_name or "gb_search", serial, _n_open)
        if self.profile.get("reset_band_max"):
            # user ruling 2026-09-30 (gb_search_3): the valve's stored per-band
            # max was earned under another noise curve -- re-learn it here
            _n = reset_band_logl_max(getattr(self, "_ratchet_last_sample", None))
            logger.info(
                "[V9-STAGE %s] band_cold_logl_max_w RESET to -inf (%d entries): "
                "the shutoff valve re-learns each (walker, band)'s best cold lnL "
                "under this stage's noise instead of the stored value.",
                self.stage_name or "gb_search", _n)
        if self.profile.get("reset_valves"):
            self._reset_valves_at_entry()
        self._legs_enter()
        self._ratchet_enter()

    def _fresh_stage_entry(self) -> bool:
        """True when this stage has stored no row of its own yet.

        Rows on the store vs the stage's start: the start stamp
        ``completed_recipe_step`` writes at the boundary (independent of
        GF_PERSIST_STAGE_START), else ``_stage_start_iter``. A stamp past the
        live rows (a rewound store) is ignored. Unreadable = fresh: resetting
        a valve costs a few iterations, keeping a stale one can cost sources.
        """
        be = getattr(self, "_ratchet_backend", None)
        try:
            rows = int(getattr(be, "iteration"))
        except Exception:  # noqa: BLE001 -- no store to read: treat as fresh
            return True
        origin = None
        fn = getattr(be, "stage_start_iteration", None)
        if callable(fn) and self.stage_name:
            try:
                origin = fn(self.stage_name)
            except Exception:  # noqa: BLE001 -- a stamp, never fatal
                origin = None
        if origin is None or int(origin) > rows:
            origin = int(getattr(self, "_stage_start_iter", rows))
        return rows <= int(origin)

    def _reset_valves_at_entry(self) -> None:
        """``reset_valves`` (user ruling 2026-10-07, the 9mo / 1yr gb_search_2:
        "GB search 2 with valves reset"), at a FRESH stage entry only.

        * the per-(walker, band) valve's cold-lnL max -> -inf (as
          ``reset_band_max``); its shut set and window are released by the
          step-serial rule above anyway;
        * the per-band BARREN valve: the moves' own
          ``_band_shutoff_revive`` (the in-memory set of a move that already
          adopted it this process) AND the persisted ``band_info`` record (what
          a fresh process adopts at its first propose) -- both, or the other
          copy re-installs the stale set.

        A RESUME inside the stage resets nothing: the stage ends on these
        valves, and restarting their clocks on every relaunch could hold it
        open indefinitely.
        """
        tag = self.stage_name or "gb_search"
        if not self._fresh_stage_entry():
            logger.info(
                "[V9-STAGE %s] reset_valves: a RESUME inside the stage (it already "
                "has rows of its own) -- the valves were reset at the stage's "
                "entry; nothing is reset now.", tag)
            return
        state = getattr(self, "_ratchet_last_sample", None)
        n_max = reset_band_logl_max(state)
        n_moves, n_mem, seen = 0, 0, set()
        for m in gb_moves_in_tree(self.moves):
            fn = getattr(m, "_band_shutoff_revive", None)
            if id(m) in seen or not callable(fn):
                continue
            seen.add(id(m))
            n_mem += int(fn(f"{tag} entry: reset_valves") or 0)
            n_moves += 1
        n_store = revive_band_shutoff_on_state(state)
        logger.info(
            "[V9-STAGE %s] reset_valves at the stage's FRESH entry: "
            "band_cold_logl_max_w -> -inf (%d entries); barren-band valve revived "
            "on %d GB move(s) (%d band(s) were off in memory) and in the stored "
            "record (%d band(s) were off).", tag, n_max, n_moves, n_mem, n_store)

    # ---- search legs --------------------------------------------------------

    def _stage_combine(self):
        """The stage combine carrying the leg cursor, or None."""
        for m in list(getattr(self, "moves", None) or []):
            mm = m[0] if isinstance(m, (tuple, list)) and m else m
            if getattr(mm, "gf_legs", None) is not None:
                return mm
        return None

    def _legs_enter(self) -> None:
        """Position the leg cursor from the store (HEAD-ONLY announce path).

        The resume position is the NAME of the leg-ender the last row was
        saved after; a fresh stage (no rows of its own yet), an old store or
        a changed composition all start at the head of the list, loudly.
        """
        if not getattr(self, "legs", False):
            return
        cm = self._stage_combine()
        tag = self.stage_name or "gb_search"
        if cm is None:
            logger.warning(
                "[LEG %s] legs=True but the stage combine carries no leg "
                "cursor -- compose the stage with leg_ends (GB_SEARCH_LEGS=1).",
                tag)
            return
        cur = cm.gf_legs
        be = getattr(self, "_ratchet_backend", None)
        live = int(getattr(self, "_ratchet_live_iter",
                           getattr(self, "_stage_start_iter", 0)))
        origin = self._ratchet_stage_origin()
        name, hist, stored = None, [], None
        if live > origin:
            fn = getattr(be, "saved_after", None)
            name = fn(live - 1) if callable(fn) else None
            fn = getattr(be, "saved_after_history", None)
            hist = fn(origin, live) if callable(fn) else []
        fn = getattr(be, "stage_move_order", None)
        stored = fn(self.stage_name) if (callable(fn) and self.stage_name) else None
        if stored is not None and list(stored) != list(cur.order):
            logger.warning(
                "[LEG %s] the stage's stored move order %s differs from the "
                "live composition %s -- resuming at the HEAD of the cycle "
                "rather than trusting a row name from a different list.",
                tag, list(stored), list(cur.order))
            cur.set_after(None)
            cur.cycles = 0
        else:
            ok = cur.set_after(name)
            if name and not ok:
                logger.warning(
                    "[LEG %s] last row was saved after %r, which is not a "
                    "leg-ender of %s -- resuming at the head of the cycle.",
                    tag, name, cur.ends)
            cur.cycles = cur.cycles_from_history(hist)
        logger.info(
            "[LEG %s] resume: rows %d..%d of this stage, last saved after %r -> "
            "next move %r (leg %d/%d), %d cycle(s) completed",
            tag, origin, live - 1, name,
            cur.order[cur.cursor] if cur.cursor < len(cur.order) else "?",
            cur.leg_index + 1, cur.nlegs, cur.cycles)

    # ---- the galfor ratchet -----------------------------------------------

    def setup_run(self, iteration, last_sample, sampler):
        super().setup_run(iteration, last_sample, sampler)
        # Stash what the HEAD-ONLY announce path needs. ⚠ setup_run itself
        # runs on every compute rank too (run.py's build loop calls it with
        # iteration=0 against a shell engine), so nothing here may read the
        # store or drive the gate -- that happens in note_recipe_step, which
        # only the recipe's announce path calls.
        self._ratchet_live_iter = int(iteration)
        self._ratchet_backend = getattr(sampler, "backend", None)
        self._ratchet_last_sample = last_sample

    def _ratchet_stage_origin(self) -> int:
        """The stored iteration this stage STARTED at -- the ratchet's clock.

        Read from the recipe group's ``start_iteration`` (stamped on the
        head's transition path, and by the rewind tool), so a resume lands in
        the right phase of the schedule instead of nudging again. Falls back
        to the live stage start when the store carries no stamp.
        """
        live = int(getattr(self, "_stage_start_iter", 0))
        be = getattr(self, "_ratchet_backend", None)
        fn = getattr(be, "stage_start_iteration", None)
        if not callable(fn) or not self.stage_name:
            return live
        try:
            stored = fn(self.stage_name)
        except Exception:  # noqa: BLE001 -- a clock, never fatal
            stored = None
        if stored is None:
            return live
        stored = int(stored)
        if stored > int(getattr(self, "_ratchet_live_iter", live)):
            logger.warning(
                "[GALFOR_RATCHET %s] stored start_iteration %d is AHEAD of the "
                "live iteration %d -- ignoring it (a stale stamp); the "
                "ratchet clock starts at the live iteration.",
                self.stage_name, stored, self._ratchet_live_iter)
            return live
        return stored

    def _ratchet_enter(self) -> None:
        """Drive the gate for the iteration about to run (stage entry/resume).

        Sets the IN-PROCESS clock ``_ratchet_k`` from the store once, here.
        ⚠ Afterwards the clock is advanced by counting ``stopping_function``
        calls, NOT by re-reading ``backend.iteration``: on the head the save
        is a handoff to the saver rank and the store's iteration attr lags
        one save behind, so a re-read at the end of an iteration still shows
        the row that was just handed off. 6mo job 672 nudged TWICE in a row
        that way (k read as 0 twice), stacking two shifts.
        """
        if getattr(self, "ratchet", None) is None:
            return
        self._ratchet_origin = self._ratchet_stage_origin()
        live = int(getattr(self, "_ratchet_live_iter",
                           getattr(self, "_stage_start_iter", 0)))
        cm = self._stage_combine() if getattr(self, "legs", False) else None
        # CYCLE units under legs (the cursor's completed-cycle count, set
        # from the store by _legs_enter just before this); rows otherwise.
        _raw_k = (int(cm.gf_legs.cycles) if cm is not None
                  else live - self._ratchet_origin)
        # CLOCK RESET (user design 2026-10-02: "we can pick this up in the
        # middle of the run"): GALFOR_RATCHET_CLOCK_RESET=1 restarts the
        # schedule at k = 0 on THIS (re-)entry, so a release-first schedule
        # begins with its release wherever the run stopped, instead of landing
        # on whatever phase the stage's stored start would put it in. k0 is
        # the stage-local iteration at which the gate will NEXT run: under
        # legs a mid-cycle resume has already passed this cycle's head, so the
        # next gate run is the next cycle.
        self._ratchet_k0 = 0
        if os.environ.get("GALFOR_RATCHET_CLOCK_RESET", "0").strip() in (
                "1", "true", "True", "yes", "on"):
            _mid = bool(cm is not None and int(getattr(cm.gf_legs, "cursor", 0)) != 0)
            # GALFOR_RATCHET_CLOCK_START (default 0): the schedule k the gate
            # runs at NEXT. 1 = "the release already happened in the cycle
            # now in progress; continue with the nudge" (6mo job 685 was cut
            # off after its first release, and a plain reset would have run
            # a second release before the first nudge). Under legs the
            # in-progress cycle then counts as schedule k = START - 1, so the
            # wrap reads its gain as that iteration's.
            _start = int(os.environ.get("GALFOR_RATCHET_CLOCK_START", "0").strip() or 0)
            if _start < 0:
                raise ValueError(f"GALFOR_RATCHET_CLOCK_START={_start} must be >= 0.")
            self._ratchet_k0 = _raw_k + (1 if _mid else 0) - _start
            logger.info(
                "[GALFOR_RATCHET %s] clock RESET at entry: stage-local %s %d "
                "becomes schedule k = %d (GALFOR_RATCHET_CLOCK_RESET=1, "
                "GALFOR_RATCHET_CLOCK_START=%d); the gate's next action is %s.",
                self.stage_name or "gb_search",
                "cycle" if cm is not None else "iteration",
                _raw_k + (1 if _mid else 0), _start, _start,
                self.ratchet.action(_start).upper())
        self._ratchet_k = _raw_k - self._ratchet_k0
        # A ratchet that already finished in an earlier process stays finished:
        # the stop is stamped in the store (see _ratchet_check_gain).
        _be = getattr(self, "_ratchet_backend", None)
        _fn = getattr(_be, "stage_flag", None)
        if callable(_fn) and self.stage_name and not getattr(self, "_ratchet_stopped", False):
            try:
                _done = _fn(self.stage_name, "galfor_ratchet_done")
            except Exception:  # noqa: BLE001 -- a stamp, never fatal
                _done = None
            if _done is not None and int(_done) != 0:
                self._ratchet_stopped = True
                if getattr(self, "ratchet_end_stage_on_stop", False):
                    self._ratchet_end_stage = True
                logger.info(
                    "[GALFOR_RATCHET %s] the store says this stage's ratchet already "
                    "FINISHED (galfor_ratchet_done stamp): no nudges on this "
                    "relaunch; the gate stays released and the stage ends %s.",
                    self.stage_name,
                    "at its first check (GALFOR_RATCHET_END_STAGE_ON_STOP)"
                    if getattr(self, "ratchet_end_stage_on_stop", False)
                    else "on its ordinary rule")
        _sample = getattr(self, "_ratchet_last_sample", None)
        # a resume mid-cycle (legs) after the gate already ran keeps the
        # reference captured before that nudge; only a cycle head recaptures
        if cm is None or cm.gf_legs.cursor == 0:
            self._ratchet_capture_reference(self._ratchet_k, _sample)
        self._drive_ratchet(self._ratchet_k, self.moves, sample=_sample)

    def _ratchet_capture_reference(self, k, sample) -> None:
        """Before a NUDGE runs, remember the cold-mean galfor vector it starts
        from: the readout compares every later row against THIS, so a release
        that climbs back reads as ~1 and one that stays down reads low."""
        if sample is None or int(k) < 0 or self.ratchet.action(k) != "nudge":
            return
        if getattr(self, "_ratchet_stopped", False):
            return                      # no nudge is coming: keep the last reference
        try:
            bc = getattr(sample, "branches_coords", None)
            if bc is not None and "galfor" in bc:
                _w = np.asarray(bc["galfor"])[0, :, 0, :]
                self._ratchet_pre_nudge = _w.mean(axis=0).copy()
                # per walker too: the readout compares the MEAN OF THE CURVES
                # (the curve of the mean parameters misreads a split alpha)
                self._ratchet_pre_nudge_w = _w.copy()
        except Exception as e:  # noqa: BLE001 -- a readout reference, never fatal
            logger.debug("[GALFOR_RATCHET] reference capture skipped: %r", e)

    def _ratchet_schedule_pending(self, k_next) -> bool:
        """True while the schedule still has a nudge/hold/release to run.

        The stage must not COMPLETE while the ratchet is mid-schedule (user
        design 2026-09-30: the release IS the measurement). 6mo job 672's
        stage 3 ended on the shutoff rule after nudge, nudge, hold -- before
        a single release -- and handed full_pe a foreground two shifts down.
        ``GALFOR_RATCHET_HOLD_STAGE=0`` restores the plain stopping rule.
        """
        if getattr(self, "ratchet", None) is None:
            return False
        if getattr(self, "_ratchet_stopped", False):
            return False          # the data-driven stop: the stage is free to end
        if os.environ.get("GALFOR_RATCHET_HOLD_STAGE", "1").strip() in (
                "0", "false", "False", "no", "off"):
            return False
        return int(k_next) < self.ratchet.total_iterations

    def _ratchet_check_gain(self, k_done, sample, moves) -> None:
        """The data-driven stop, decided after every completed RELEASE.

        ``ratchet_stop_rule``:

        * ``galfor`` (user ruling 2026-10-03, "adjust away from the raw lnL
          gain"): the released per-walker galfor curve over
          ``ratchet_drop_band`` is divided by the SAME walker's curve at the
          previous release (:func:`noise_ratchet.galfor_band_drop`); when the
          median walker's foreground fell by less than ``ratchet_min_drop``
          (or rose) the last nudge bought nothing and the ratchet ends. The
          lnL gain is still logged, informational. A sample without galfor
          coordinates falls back to the gain rule for that release, with a
          WARNING (never a silent no-op).
        * ``gain`` (2026-10-02): the maximum cold log-likelihood over the
          walkers ("check the max logL achieved") must gain at least
          ``ratchet_min_gain`` nats over the previous release.
        * ``off``: nothing here ends the ratchet; the cycle ceiling does.

        On a stop ``_ratchet_stopped`` latches, the gate is told to finish
        (:meth:`NoiseRatchetGate.finish_ratchet`), the refit cadence and the
        valve floor come back, the decision is stamped in the store and the
        stage ends on its ordinary rule. ``ratchet_min_nudges`` holds every
        rule off until that many nudges have run in this process. The first
        release only records the reference.
        """
        if getattr(self, "ratchet", None) is None or sample is None:
            return
        rule = str(getattr(self, "ratchet_stop_rule", "gain") or "gain")
        gain_min = float(getattr(self, "ratchet_min_gain", 0.0) or 0.0)
        min_drop = float(getattr(self, "ratchet_min_drop", 0.0) or 0.0)
        if getattr(self, "_ratchet_stopped", False) or rule == "off":
            return
        if rule == "gain" and gain_min <= 0:
            return                      # the 10-02 rule, disarmed
        if rule == "galfor" and min_drop <= 0:
            return                      # refused at construction; belt and braces
        if int(k_done) < 0 or self.ratchet.action(k_done) != "release":
            return                      # k < 0: a head that ran before a clock reset
        try:
            ll = np.asarray(sample.log_like, dtype=float)[0]
            mx, mean = float(np.max(ll)), float(np.mean(ll))
        except Exception:  # noqa: BLE001 -- a readout, never fatal
            return
        tag = self.stage_name or "gb_search"
        prev = getattr(self, "_ratchet_last_release_max", None)
        hist = getattr(self, "_ratchet_release_maxes", None)
        if hist is None:
            hist = self._ratchet_release_maxes = []
        hist.append((int(k_done), mx))
        self._ratchet_last_release_max = mx
        # the RELEASED per-walker galfor vectors: the noise moves run only in
        # the gated head under the ratchet, so the wrap's coordinates ARE the
        # release's result (6mo rows 73-76 / 81-84: identical galfor per leg)
        coords = self._ratchet_release_coords(sample)
        prev_coords = getattr(self, "_ratchet_last_release_galfor", None)
        self._ratchet_last_release_galfor = coords
        lo, hi = getattr(self, "ratchet_drop_band", (3e-3, 5e-3))
        if prev is None:
            if rule == "galfor":
                logger.info(
                    "[GALFOR_RATCHET %s] after RELEASE %d: released galfor recorded as the "
                    "reference for the stop rule (GALFOR_RATCHET_STOP_RULE=galfor: the next "
                    "release must lower the median walker's foreground at %g-%g mHz by at "
                    "least %g%% or the ratchet ends); max cold lnL %.3f (walker mean %.3f), "
                    "informational.%s", tag, k_done, 1e3 * lo, 1e3 * hi, 100.0 * min_drop,
                    mx, mean, "" if coords is not None else
                    " WARNING: no galfor coordinates in the sample -- the lnL gain rule "
                    "will decide the next release instead.")
            else:
                logger.info(
                    "[GALFOR_RATCHET %s] after RELEASE %d: max cold lnL %.3f (walker "
                    "mean %.3f) -- the baseline for the ratchet's stop rule "
                    "(GALFOR_RATCHET_MIN_GAIN=%.0f).", tag, k_done, mx, mean, gain_min)
            return
        gain = mx - prev
        # ---- the verdict ---------------------------------------------------
        if rule == "galfor" and coords is not None and prev_coords is not None \
                and np.shape(coords) == np.shape(prev_coords):
            from .noise_ratchet import galfor_band_drop

            stat = galfor_band_drop(coords, prev_coords, lo, hi)
            drops = getattr(self, "_ratchet_release_drops", None)
            if drops is None:
                drops = self._ratchet_release_drops = []
            drops.append((int(k_done), float(stat["drop"])))
            keep_going = stat["drop"] >= min_drop
            what = (f"released foreground at {1e3 * lo:g}-{1e3 * hi:g} mHz / previous release: "
                    f"median walker ratio {stat['ratio_median']:.4f} = drop {100.0 * stat['drop']:+.2f}% "
                    f"(threshold {100.0 * min_drop:g}%; walkers "
                    f"{np.array2string(np.asarray(stat['ratio_walkers']), precision=3)}; "
                    f"mean-of-curves {stat['ratio_mean_curves']:.4f}); max cold lnL {mx:.3f}, "
                    f"{gain:+.1f} over the previous release (informational)")
        else:
            if rule == "galfor":
                logger.warning(
                    "[GALFOR_RATCHET %s] after RELEASE %d: the stop rule is 'galfor' but the "
                    "released galfor coordinates are %s -- deciding this release on the lnL "
                    "gain rule instead (GALFOR_RATCHET_MIN_GAIN=%.0f%s).", tag, k_done,
                    "missing from the sample" if (coords is None or prev_coords is None)
                    else f"shaped {np.shape(coords)} vs {np.shape(prev_coords)}",
                    gain_min, "" if gain_min > 0 else " = no stop possible")
            keep_going = (gain >= gain_min) if gain_min > 0 else True
            what = (f"max cold lnL {mx:.3f} (walker mean {mean:.3f}), {gain:+.1f} over the "
                    f"previous release (threshold {gain_min:.0f})")
        if keep_going:
            logger.info("[GALFOR_RATCHET %s] after RELEASE %d: %s -- a real step; ratcheting "
                        "continues.", tag, k_done, what)
            return
        _n_nudges = int(getattr(self, "_ratchet_nudges", 0) or 0)
        _min_nudges = int(getattr(self, "ratchet_min_nudges", 0) or 0)
        if _n_nudges < _min_nudges:
            logger.info(
                "[GALFOR_RATCHET %s] after RELEASE %d: %s -- below the threshold, but only %d "
                "of the required %d nudge(s) have run in this process "
                "(GALFOR_RATCHET_MIN_NUDGES); ratcheting continues.", tag, k_done, what,
                _n_nudges, _min_nudges)
            return
        self._ratchet_stopped = True
        if getattr(self, "ratchet_end_stage_on_stop", False):
            self._ratchet_end_stage = True       # read by stopping_function this same call
        gate = self._ratchet_gate(moves) if moves is not None else None
        if gate is not None and hasattr(gate, "finish_ratchet"):
            gate.finish_ratchet()
        # the refit cadence and the valve floor come back NOW, not at the next
        # drive: if the stage ends in this same iteration there is no next drive
        self._ratchet_refit_only_on_nudge(moves)
        self._ratchet_shutoff_floor(moves)
        # PERSIST the decision: a relaunch re-enters this stage with a fresh
        # process and would otherwise resume the schedule (clock reset or
        # not) and nudge again. The stamp is read back by _ratchet_enter.
        _be = getattr(self, "_ratchet_backend", None)
        _fn = getattr(_be, "stamp_stage_flag", None)
        _stamped = bool(_fn(self.stage_name, "galfor_ratchet_done", 1)) if (
            callable(_fn) and self.stage_name) else False
        logger.info(
            "[GALFOR_RATCHET %s] after RELEASE %d: %s -- below the threshold: RATCHET DONE "
            "after %d release(s): no more nudges; the gate stays released and "
            "the stage ends on its ordinary per-band shut-off rule. Stop %s "
            "in the store's recipe group.", tag, k_done, what,
            len(hist), "STAMPED" if _stamped else "NOT stamped (in-process only)")

    @staticmethod
    def _ratchet_release_coords(sample):
        """The cold per-walker galfor vectors ``(nwalkers, 5)`` of ``sample``, or None."""
        try:
            bc = getattr(sample, "branches_coords", None)
            if bc is None or "galfor" not in bc:
                return None
            arr = np.asarray(bc["galfor"], dtype=float)
            if arr.ndim != 4 or arr.shape[2] < 1:
                return None
            return np.array(arr[0, :, 0, :], copy=True)
        except Exception:  # noqa: BLE001 -- a readout, never fatal
            return None

    def _ratchet_gate(self, moves):
        from .noise_ratchet import is_noise_ratchet_gate

        for m in iter_move_tree(moves):
            if is_noise_ratchet_gate(m):
                return m
        return None

    def _ratchet_shutoff_floor(self, moves, k=None, force_off=False) -> None:
        """Hold the RJ shutoff valve open below a frequency while the ratchet
        is active (user design 2026-10-02: "maybe don't do RJ shutoff during
        the ratchet cycles ... you can shut off bands above 7 mHz if their
        likelihoods converge as usual"). Sets ``rj_shutoff_min_freq`` on
        every GB band move to ``GALFOR_RATCHET_SHUTOFF_MIN_FREQ`` Hz (default
        7e-3; 0 disables) while a nudge can still come, and back to 0 at the
        stop, the cycle ceiling and stage end. The valve itself reopens and
        re-streaks the exempt pairs at its next judgment.
        """
        if getattr(self, "ratchet", None) is None or moves is None:
            return
        raw = os.environ.get("GALFOR_RATCHET_SHUTOFF_MIN_FREQ", "7e-3").strip()
        fmin = float(raw) if raw else 0.0
        if fmin < 0:
            raise ValueError(f"GALFOR_RATCHET_SHUTOFF_MIN_FREQ={fmin} must be >= 0 Hz.")
        active = (fmin > 0 and not force_off
                  and not getattr(self, "_ratchet_stopped", False)
                  and (k is None or int(k) < self.ratchet.total_iterations))
        want = fmin if active else 0.0
        changed = []
        for m in gb_moves_in_tree(moves):
            if float(getattr(m, "rj_shutoff_min_freq", 0.0) or 0.0) != want:
                m.rj_shutoff_min_freq = want
                changed.append(getattr(m, "name", "?"))
        if changed:
            logger.info(
                "[GALFOR_RATCHET %s] RJ shutoff floor %s on %s: %s.",
                self.stage_name or "gb_search",
                f"{1e3 * want:.3g} mHz" if want else "LIFTED", changed,
                "bands below it stay open while the ratchet is active" if want
                else "the valve may shut converged pairs at every frequency again")
        if want:
            # Reopen the LIVE valve now, not at the next judgment: the nudge's
            # hard refit runs BEFORE that judgment and skips bands shut on
            # every walker (job 695 epoch 37: 5,701 peaks vs 14,119).
            _edges = None
            for m in gb_moves_in_tree(moves):
                _edges = getattr(m, "band_edges", None)
                if _edges is not None:
                    break
            n_open, n_ex = apply_shutoff_floor_to_state(
                getattr(self, "_ratchet_last_sample", None), want, _edges)
            if n_open:
                logger.info(
                    "[GALFOR_RATCHET %s] RJ shutoff floor applied to the LIVE valve: "
                    "%d shut (walker, band) pair(s) in the %d band(s) below %.3g mHz "
                    "reopened now (streaks zeroed), ahead of the next F-stat refit.",
                    self.stage_name or "gb_search", n_open, n_ex, 1e3 * want)

    def _ratchet_refit_only_on_nudge(self, moves, k=None, force_off=False) -> None:
        """Hand the F-stat refit clock to the ratchet while it is active.

        User ruling 2026-10-02: "let's run it only on the nudge step." Job
        675 refit the grid once per cycle (~2 h each); only the first was the
        nudge's forced refit, the rest the stage cadence. While the ratchet
        is active every grid move's cadence is suspended
        (``fstat_refit_only_forced``), so an epoch opens only when a nudge
        HARD-arms one; once the ratchet has finished the cadence is restored
        for the ordinary per-band search phase. ``GALFOR_RATCHET_REFIT_ONLY_ON_NUDGE=0``
        keeps the cadence throughout. Idempotent; logs on change.
        """
        if getattr(self, "ratchet", None) is None or moves is None:
            return
        knob = os.environ.get("GALFOR_RATCHET_REFIT_ONLY_ON_NUDGE", "1").strip()
        # ACTIVE = a nudge can still come: not stopped, not past the schedule's
        # last cycle, and the stage not ending. ⚠ The grid move object is
        # SHARED with the stages after this one (ctx.stock_moves), so the
        # flag must be False whenever the ratchet can no longer force a
        # refit, or full_pe would inherit a grid that never refits.
        want = (knob not in ("0", "false", "False", "no", "off")
                and not force_off
                and not getattr(self, "_ratchet_stopped", False)
                and (k is None or int(k) < self.ratchet.total_iterations))
        changed = []
        for m in iter_move_tree(moves):
            if not callable(getattr(m, "arm_fstat_refit", None)):
                continue
            if bool(getattr(m, "fstat_refit_only_forced", False)) != want:
                m.fstat_refit_only_forced = want
                changed.append(getattr(m, "name", "?"))
        if changed:
            logger.info(
                "[GALFOR_RATCHET %s] F-stat refit cadence %s on %s: %s.",
                self.stage_name or "gb_search",
                "SUSPENDED" if want else "RESTORED", changed,
                "the grid refits only when a nudge forces it" if want
                else "the ordinary GB_FSTAT_REFIT_EVERY cadence is back")

    def _drive_ratchet(self, k, moves, sample=None) -> None:
        """Set the gate's mode for stage-local cycle ``k``.

        Called at stage entry and at the end of every iteration (per cycle
        under legs). A nudge additionally arms a HARD F-stat refit on every
        grid move in the stage (the grid must be re-fitted against the
        lowered noise before the next birth proposal draws from it) and
        RESETS the shutoff valve's per-band cold-lnL max on ``sample`` (the
        state the nudge will run on): a max earned under the old noise
        would otherwise read every band as "no improvement" -- see
        :func:`reset_band_logl_max`.
        """
        if getattr(self, "ratchet", None) is None:
            return
        k = int(k)
        # the refit clock and the valve floor belong to the ratchet while
        # nudges can still come (also on a mid-cycle resume, k < 0, before
        # this cycle's legs run)
        self._ratchet_refit_only_on_nudge(moves, k=k)
        self._ratchet_shutoff_floor(moves, k=k)
        if k < 0:
            return
        gate = self._ratchet_gate(moves)
        tag = self.stage_name or "gb_search"
        if getattr(self, "_ratchet_stopped", False):
            # the data-driven stop latched: a plain release, rider mode
            if gate is not None and hasattr(gate, "finish_ratchet") and not getattr(
                    gate, "ratchet_finished", False):
                gate.finish_ratchet()
            self._ratchet_last_action = "release"
            self._ratchet_last_k = k
            return
        if gate is None:
            if self._ratchet_last_k is None:
                logger.warning(
                    "[GALFOR_RATCHET %s] a ratchet schedule is set but the "
                    "stage carries no NoiseRatchetGate -- the schedule "
                    "reaches NOTHING. Compose the stage with the gated noise "
                    "head (GALFOR_RATCHET=1 in the launcher).", tag)
            self._ratchet_last_k = k
            return
        action = self.ratchet.action(k)
        gate.set_mode(action)
        if action == "nudge":
            # the nudge COUNT is part of the serial: arm_fstat_refit is
            # idempotent per serial, and a repeated k (job 672) silently
            # skipped the second nudge's refit
            self._ratchet_nudges = int(getattr(self, "_ratchet_nudges", 0)) + 1
            serial = ("galfor_ratchet", tag, k, self._ratchet_nudges)
            n = force_fstat_refit(
                moves, serial,
                f"galfor ratchet nudge at stage-local iteration {k}",
                hard=True)
            _nmax = reset_band_logl_max(sample) if sample is not None else 0
            logger.info(
                "[GALFOR_RATCHET %s] iteration %d: NUDGE (cycle %d of %d) -- "
                "galfor shift %s in the sampled basis; hard F-stat refit "
                "armed on %d grid move(s); valve per-band lnL max reset (%d "
                "entries); the noise moves then HOLD for %d iteration(s) and "
                "RELEASE for %d.",
                tag, k, self.ratchet.cycle_of(k), self.ratchet.cycles,
                np.array2string(np.asarray(self.ratchet_delta), precision=3)
                if self.ratchet_delta is not None else "?",
                n, _nmax, self.ratchet.hold - 1, self.ratchet.release)
        elif action != (self._ratchet_last_action if hasattr(
                self, "_ratchet_last_action") else None):
            logger.info("[GALFOR_RATCHET %s] iteration %d: %s", tag, k,
                        action.upper())
        self._ratchet_last_action = action
        self._ratchet_last_k = k

    def _ratchet_readout(self, k_done, sample) -> None:
        """One [GALFOR_RATCHET] line per completed iteration ``k_done`` with
        the cold-mean galfor curve against the PRE-nudge curve at 1..5 mHz --
        the release IS the measurement: a region that comes back was honest,
        one that stays down had been absorbing resolvable power."""
        if getattr(self, "ratchet", None) is None or sample is None:
            return
        if int(k_done) < 0:
            return                      # a head that ran before a clock reset
        try:
            from .noise_ratchet import galfor_curve_ratio, galfor_curves

            bc = getattr(sample, "branches_coords", None)
            if bc is None or "galfor" not in bc:
                return
            cur_w = np.asarray(bc["galfor"])[0, :, 0, :]
            mean = cur_w.mean(axis=0)
            ref = getattr(self, "_ratchet_pre_nudge", None)
            ref_w = getattr(self, "_ratchet_pre_nudge_w", None)
            f = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]) * 1e-3
            if ref is None:
                logger.info(
                    "[GALFOR_RATCHET %s] after iteration %d (%s): cold-mean galfor "
                    "%s; no pre-nudge reference captured yet",
                    self.stage_name or "gb_search", k_done,
                    self.ratchet.action(k_done).upper(),
                    np.array2string(mean, precision=4))
                return
            if ref_w is not None and np.shape(ref_w) == np.shape(cur_w):
                # MEAN OF THE WALKERS' CURVES over mean of the reference
                # curves -- not the curve of the mean parameters, which
                # misreads a split alpha/f1 (job 695 row 81: +2..5 % where
                # the walkers' curves averaged -6..7 % at 3.5-5 mHz).
                _cur = galfor_curves(cur_w, f)
                r = _cur.mean(axis=0) / galfor_curves(ref_w, f).mean(axis=0)
                spread = _cur.max(axis=0) / _cur.min(axis=0)
                logger.info(
                    "[GALFOR_RATCHET %s] after iteration %d (%s): cold-mean galfor "
                    "%s; curve (mean over walkers) / pre-nudge at 1,2,3,3.5,4,4.5,5 "
                    "mHz = %s; walker max/min of the curve = %s",
                    self.stage_name or "gb_search", k_done,
                    self.ratchet.action(k_done).upper(),
                    np.array2string(mean, precision=4),
                    np.array2string(r, precision=3),
                    np.array2string(spread, precision=2))
                return
            r = galfor_curve_ratio(mean, ref, f)
            logger.info(
                "[GALFOR_RATCHET %s] after iteration %d (%s): cold-mean galfor "
                "%s; curve / pre-nudge at 1,2,3,3.5,4,4.5,5 mHz = %s",
                self.stage_name or "gb_search", k_done,
                self.ratchet.action(k_done).upper(),
                np.array2string(mean, precision=4),
                np.array2string(r, precision=3))
        except Exception as e:  # noqa: BLE001 -- a readout must never stop a run
            logger.debug("[GALFOR_RATCHET] readout skipped: %r", e)

    def _apply_profile(self, serial) -> None:
        tag = self.stage_name or "gb_search"
        prof = self.profile
        gb_moves = list(gb_moves_in_tree(self.moves))
        if not gb_moves:
            logger.warning(
                "[V9-STAGE %s] no GB band moves in this stage's move tree -- "
                "the stage profile %s applies to NOTHING. Either the gb "
                "branch was removed or the stage was mis-composed.", tag, prof)
        logger.info(
            "[V9-STAGE %s] entering (recipe step %s): profile %s over %d GB "
            "move(s) %s", tag, serial,
            {k: v for k, v in prof.items() if v is not None},
            len(gb_moves), [getattr(m, "name", "?") for m in gb_moves])

        snr = prof.get("opt_snr")
        if snr is not None:
            for m in gb_moves:
                old = float(getattr(m, "opt_snr_rej_samp_limit", float("nan")))
                if old != float(snr):
                    logger.info(
                        "[V9-STAGE %s] %s.opt_snr_rej_samp_limit %.3f -> "
                        "%.3f", tag, m.name, old, float(snr))
                m.opt_snr_rej_samp_limit = float(snr)
                # The per-(walker, band) stage table, when the COARSE/FINE
                # schedule is armed, SHADOWS the scalar (``_live_snr_lim``
                # returns the table). Say so rather than leaving a log line
                # claiming a value the buffer never sees.
                if getattr(m, "_snr_lim_table", None) is not None:
                    logger.warning(
                        "[V9-STAGE %s] %s carries a per-(walker, band) "
                        "opt-SNR table (GB_SEARCH_STAGE_PER_WALKER=1), which "
                        "SHADOWS the scalar just set. v9 expects that flag "
                        "OFF -- SNR floors move per recipe STAGE, not per "
                        "band.", tag, m.name)

        # MOVE-SCOPED key (user ruling 2026-10-02: "set it specifically for gb
        # search 3. gb search 1 should still be prior removal"): what the
        # prior RJ move proposes in THIS stage. ``prior_births`` True = births
        # AND deaths from the prior container; False = the original pruning
        # move (deaths only). ⚠ Never broadcast: ``rj_removal_only`` on
        # rj_fstat_search means alive rows only, i.e. NO BIRTHS (the 09-29
        # hazard note), so this touches the prior move alone, found by
        # name. The move object is shared by every search stage, which is
        # exactly why each stage's profile states its own value.
        pb = prof.get("prior_births")
        if pb is not None:
            _prior = [m for m in gb_moves
                      if getattr(m, "name", None) == "rj_prior_removal"]
            if not _prior:
                logger.warning(
                    "[V9-STAGE %s] profile asks prior_births=%s but the stage "
                    "carries no 'rj_prior_removal' move -- nothing to set.",
                    tag, bool(pb))
            for m in _prior:
                want_ro = not bool(pb)
                if want_ro and bool(getattr(m, "rj_replace", False)):
                    raise ValueError(
                        f"[V9-STAGE {tag}] {m.name}: rj_removal_only and rj_replace "
                        "are mutually exclusive; the profile cannot make a replace "
                        "move deaths-only.")
                old = bool(getattr(m, "rj_removal_only", False))
                if old != want_ro:
                    logger.info(
                        "[V9-STAGE %s] %s.rj_removal_only %s -> %s (%s)", tag, m.name,
                        old, want_ro,
                        "deaths only" if want_ro else "prior BIRTHS and deaths")
                m.rj_removal_only = want_ro

        pm = prof.get("phase_maximize")
        if pm is not None:
            # EVERY GB MOVE IN THE STAGE, not just the RJ ones. User ruling
            # 2026-09-25: "all the GB proposals in the gb search 1 should
            # be phase maximized" -- so when a stage asks for it, it means
            # the whole GB move list, in-model slots included.
            #
            # WHICH stage asks is a separate question and only gb_search_1
            # does. Stages 2 and 3 carry False and this same loop turns
            # every flag back OFF on entry -- which is load-bearing now
            # that the loop writes all of them: a profile that only ever
            # set True would strand stage 1's arming in the later stages
            # with no way back.
            #
            # This deliberately REVERSES two earlier decisions, both of
            # which are still correct outside stages 1-2 and are preserved
            # by leaving the CONSTRUCTOR defaults False:
            #
            #  * the pure in-model moves were built ``phase_maximize=False``
            #    on the rule "in-model scoring is at the actual phase, phase
            #    maximization is a BIRTH heuristic". A legacy single-stage
            #    recipe (STAGE_V9_SEARCH=0) installs no profile, so it still
            #    gets the old behaviour bit-identically -- only a v9 search
            #    stage that explicitly asks for it flips them.
            #  * ``rj_replace.phase_maximize`` was excluded outright. See
            #    the risk note below; it is now written like any other move.
            #
            # ⚠ WHAT IS *NOT* CHANGED, AND MUST NOT BE. Only the ADDED /
            # refined source is ever maximized. The REMOVED source in a
            # death or a replace swap is always scored at its ACTUAL phase
            # -- that asymmetry is the 2026-08-24 redesign and it is what
            # keeps the MH ratio honest. It lives inside the moves
            # (``_replace_phase_max_scoring``, ``_eval(death_k, False)``)
            # and no stage profile reaches it.
            #
            # ⚠ RISK ON rj_replace, stated rather than hidden. Its recorded
            # failure signature was a propose-level lnL drift of 1.5-1.9e3,
            # root-caused to MAXIMIZED CREDIT WITHOUT the rotation write-back
            # -- a delta unattainable at any actual phi0. The mitigation is
            # in place on this path: ``inmodel_get_add_ll`` subtracts the
            # engine's maximizing rotation from phi0 and re-wraps before the
            # accept test, so the scored rows ARE the kept rows. It has
            # never run ARMED in production, so the [GB_REPLACE] |dlnL|
            # alarm above 1.0e3 is the guard that matters this run.
            for m in gb_moves:
                old = bool(getattr(m, "phase_maximize", False))
                if old != bool(pm):
                    logger.info(
                        "[V9-STAGE %s] %s.phase_maximize %s -> %s%s",
                        tag, m.name, old, bool(pm),
                        " (in-model repeats on its survivors; the SWAP's own "
                        "maximization is GB_REPLACE_PHASE_MAX and is "
                        "unaffected)" if getattr(m, "rj_replace", False)
                        else "")
                m.phase_maximize = bool(pm)

        # The stage's in-model budget, on EVERY entry: the keys the profile
        # gives are set, the ones it omits go back to the built values
        # (apply_inmodel_budget). ``GB_SEARCH_{N}_NUM_REPEAT_PROPOSALS`` /
        # ``_INMODEL_GROUP_ITERS`` / ``_INMODEL_CONVERGE_ITERS`` fill it.
        apply_inmodel_budget(gb_moves, prof.get("inmodel_budget"), tag)

        peak = prof.get("peak_min_snr")
        if peak is not None:
            from lisatools.sampling.fstat_proposal import (
                peak_min_F_override, set_peak_min_F_override)

            want_F = 0.5 * float(peak) ** 2
            have_F = peak_min_F_override()
            set_peak_min_F_override(float(peak))
            changed = have_F is None or abs(have_F - want_F) > 1e-9 * max(
                1.0, abs(want_F))
            logger.info(
                "[V9-STAGE %s] F-stat peak floor -> SNR %.3f (F %.3f); "
                "previous in-code override %s", tag, float(peak), want_F,
                "unset (the run's FSTAT_PEAK_MIN_SNR was in force)"
                if have_F is None else f"F {have_F:.3f}")
            if changed:
                n = force_fstat_refit(
                    self.moves, serial,
                    f"{tag}: peak floor -> SNR {float(peak):.3f}")
                logger.info(
                    "[V9-STAGE %s] armed a forced fresh-epoch F-stat refit on "
                    "%d grid move(s). The stage-B cache stamps the floor its "
                    "peaks were selected at and its loader refuses a "
                    "mismatch, so a new epoch -- not a deleted file -- is how "
                    "the new floor takes effect.", tag, n)
                if n == 0:
                    logger.warning(
                        "[V9-STAGE %s] the peak floor MOVED but no F-stat "
                        "grid move was found to refit. The new floor will "
                        "only bind if something else fits a grid.", tag)

    # ---- stopping ----------------------------------------------------------

    def stopping_function(self, i, sample, sampler) -> bool:
        """The stage's stopping rules (:meth:`_stopping_rules`) with the galfor
        ratchet layered on top: the stage-local clock advances by one per
        call, the readout line is written, the stage is HELD OPEN while the
        ratchet schedule is still pending, and the gate is driven for the
        iteration about to start."""
        stop = self._stopping_rules(i, sample, sampler)
        # getattr: fake-based suites build this step without __init__
        if getattr(self, "ratchet", None) is None:
            return stop
        moves = getattr(sampler, "moves", None)
        # IN-PROCESS clock: one stopping_function call = one completed
        # iteration. Never re-read backend.iteration here -- on the head it
        # lags the handoff by one save (see _ratchet_enter). Under LEGS the
        # clock is the cursor's completed-cycle count, so the gate is driven
        # once per CYCLE (at the wrap), not once per row.
        _k_done = int(getattr(self, "_ratchet_k", 0))
        _cm = self._stage_combine() if getattr(self, "legs", False) else None
        if _cm is not None:
            # schedule k = the cursor's completed cycles minus the clock offset
            _k_next = int(_cm.gf_legs.cycles) - int(getattr(self, "_ratchet_k0", 0))
            _advanced = _k_next > _k_done
        else:
            _k_next = _k_done + 1
            _advanced = True
        self._ratchet_readout(_k_done, sample)
        # ONCE per completed schedule iteration. Under legs this function runs
        # at every leg end; evaluating the gain on each call would compare
        # leg against leg within one cycle and stop the ratchet on the first
        # small within-cycle change. The wrap (``_advanced``) is the end of
        # the full iteration the release opened, which is also the state the
        # user asked to read ("the full iteration following that").
        if _advanced:
            self._ratchet_check_gain(_k_done, sample, moves)
        if getattr(self, "_ratchet_end_stage", False) and not stop:
            # user design 2026-10-03: this stage IS the ratchet; the
            # foreground has converged, so the stage ends here and the
            # replica-PE stage takes over (the valve is not consulted)
            logger.info(
                "[GALFOR_RATCHET %s] STAGE COMPLETE at the ratchet's stop "
                "(GALFOR_RATCHET_END_STAGE_ON_STOP=1): the foreground rule ended the "
                "ratcheting after iteration %d; the per-band valve is not consulted.",
                self.stage_name or "gb_search", _k_done)
            stop = True
        if stop and self._ratchet_schedule_pending(_k_next):
            logger.info(
                "[GALFOR_RATCHET %s] stage would complete after iteration "
                "%d but the ratchet schedule is mid-way (%d of %d "
                "iterations) -- holding the stage open so the release "
                "can be read (GALFOR_RATCHET_HOLD_STAGE=0 disables).",
                self.stage_name or "gb_search", _k_done, _k_next,
                self.ratchet.total_iterations)
            stop = False
        if not stop and _advanced:
            self._ratchet_k = _k_next
            self._ratchet_capture_reference(_k_next, sample)
            self._drive_ratchet(_k_next, moves, sample=sample)
        if stop:
            # the stage is ending: the shared moves leave with their ordinary
            # cadence and an unfloored valve, whatever the ratchet did with them
            self._ratchet_refit_only_on_nudge(moves, force_off=True)
            self._ratchet_shutoff_floor(moves, force_off=True)
        return stop

    def _stopping_rules(self, i, sample, sampler) -> bool:
        """Nleaves plateau AND the per-(walker, band) RJ valve (gate 6).

        COMPOSED, never replaced. The criterion the user sanctioned is
        **"every OCCUPIED (walker, band) pair has shut off"**, and the two
        halves cover disjoint failure modes:

        * ``band_shutoff_w_pending_total`` counts OCCUPIED pairs still paying
          in lnL. An EMPTY pair can never shut off (a band that has found
          nothing has not plateaued, it has not started), so this can never
          speak for the empty bands -- which the nleaves plateau does.
        * ⚠ ``band_shutoff_w_armed`` is MANDATORY. ``pending_total`` returns
          0 both when everything converged and when the feature is off, and
          the second reading would end every stage at its first check.
        """
        stop = super().stopping_function(i, sample, sampler)
        moves = getattr(sampler, "moves", None)
        # ---- GB_SEARCH_STAGE_END_ON_SHUTOFF (user ruling 2026-09-26) ------
        # "We want to end the recipe stage when all the (band, walker) pairs
        # are shutoff from converged logL per (band, walker)." The shutoff
        # becomes the WHOLE criterion instead of being ANDed with the nleaves
        # plateau.
        #
        # WHY IT MATTERS in practice: the plateau needs 41 in-stage iterations
        # before it evaluates at all, and its clock resets on every restart
        # (see GF_PERSIST_STAGE_START), so a stage whose pairs have all shut
        # off can sit there for hours re-proving a converged model.
        #
        # ⚠ TWO GUARDS, BOTH LOAD-BEARING.
        #   * band_shutoff_w_armed: `pending` is 0 BOTH when everything
        #     converged and when the valve is off, and only this tells them
        #     apart. Without it every stage ends at its first check.
        #   * nleaves > 0: an EMPTY stage has no occupied pairs, so pending is
        #     0 at entry. gb_search_1 starts at ZERO leaves -- this rule alone
        #     would end it on its first check, before a single birth. The
        #     plateau rule carries the same guard ("a search that has not
        #     found its FIRST source has not plateaued, it has not started").
        #
        # ⚠ DEFAULT ON (user ruling 2026-09-26, raised from the default-off
        # it shipped with hours earlier). The shutoff rule IS the criterion
        # the v9 search stages were designed around; the nleaves plateau was
        # only ever the half that covered empty bands, and the nleaves > 0
        # guard below covers that case directly. This also applies to the 6mo
        # run's gb_search_3 on its next launch.
        # GB_SEARCH_STAGE_END_ON_SHUTOFF=0 (or empty) restores the composed
        # "plateau AND shutoff" rule bit-identically.
        if (not stop and band_shutoff_w_armed(moves)
                and os.environ.get(
                    "GB_SEARCH_STAGE_END_ON_SHUTOFF", "1").strip()
                not in ("0", "false", "False", "no", "off", "")):
            if band_shutoff_w_pending_total(moves) == 0:
                _n = 0
                try:
                    _nl = sampler.backend.get_nleaves(
                        branch_names=[self.plateau_branch], temp_index=0
                    )[self.plateau_branch][
                        int(getattr(self, "_stage_start_iter", 0)):]
                    _n = int(_nl.max()) if len(_nl) else 0
                except Exception:  # noqa: BLE001 -- fall through to no-stop
                    _n = 0
                if _n > 0:
                    logger.info(
                        "[V9-STAGE %s] STAGE COMPLETE on the SHUTOFF rule "
                        "alone: every occupied (walker, band) pair has shut "
                        "off (%d cold leaves in-stage; the nleaves plateau "
                        "was NOT required -- GB_SEARCH_STAGE_END_ON_SHUTOFF).",
                        self.stage_name or "gb_search", _n)
                    return True
                logger.info(
                    "[V9-STAGE %s] shutoff rule satisfied but the stage has "
                    "found NO leaves yet -- holding open (a search that has "
                    "not found its first source has not converged).",
                    self.stage_name or "gb_search")
        if band_shutoff_w_armed(moves):
            pending = band_shutoff_w_pending_total(moves)
            if stop and pending:
                logger.info(
                    "[V9-STAGE %s] nleaves plateau reached but %d occupied "
                    "(walker, band) pair(s) have NOT shut off -- holding the "
                    "stage open.", self.stage_name or "gb_search", pending)
                stop = False
            elif stop:
                logger.info(
                    "[V9-STAGE %s] STAGE COMPLETE: nleaves plateau AND every "
                    "occupied (walker, band) pair has shut off.",
                    self.stage_name or "gb_search")
            elif i % 10 == 0:
                logger.info(
                    "[V9-STAGE %s] running: %d occupied (walker, band) "
                    "pair(s) still active.",
                    self.stage_name or "gb_search", pending)
        elif stop:
            logger.info(
                "[V9-STAGE %s] STAGE COMPLETE on the nleaves plateau alone "
                "(the per-(walker, band) RJ valve is NOT armed -- export "
                "GB_SEARCH_BAND_SHUTOFF_PER_WALKER=1 to add it).",
                self.stage_name or "gb_search")
        return stop


_STEP_CLASSES = {
    "search": SearchRecipeStep,
    "pe": PERecipeStep,
    "rj": RJRecipeStep,
    # v9: an ``rj`` step that additionally owns a per-stage GB search profile
    # and composes the per-(walker, band) shutoff valve into its stopping rule.
    "gb_search": SearchStageProfileStep,
    # 2026-10-03: a ``pe`` step that ENDS when the large-scale metrics converge
    # (the replica-PE stage between the GB search and full_pe).
    "replica_pe": ReplicaPERecipeStep,
}


class Stage:
    """One declarative phase of the recipe (materializes into a :class:`RecipeStep`).

    A stage groups the moves that run together during one phase of the fit
    (they are wrapped in a single :class:`GFCombineMove` and proposed as a
    unit), and its ``kind`` picks the runtime step class — i.e. when the
    recipe advances past it:

    * ``"search"`` -> :class:`SearchRecipeStep` (the move runs its search to
      completion internally; the stage is done on its first check);
    * ``"pe"`` -> :class:`PERecipeStep` (runs indefinitely);
    * ``"rj"`` -> :class:`RJRecipeStep` (stops when the monitored branch's
      cold-chain leaf count plateaus; knobs via ``step_kwargs``);
    * ``"gb_search"`` -> :class:`SearchStageProfileStep` (v9): an ``"rj"``
      step that also applies a per-stage GB search profile at stage entry
      and composes the per-(walker, band) RJ shutoff valve into its stopping
      rule. ``step_kwargs`` takes ``profile=`` and ``stage_name=`` on top of
      the RJ plateau knobs.

    Args:
        name: Unique stage name (e.g. ``"gb_pe"``, ``"main"``).
        kind: ``"search"`` | ``"pe"`` | ``"rj"``.
        moves: Ordered moves for this stage — each entry is anything
            :meth:`Recipe.add_move` accepts (a
            :class:`~lisatools.globalfit.moves.globalfitmove.Move`, a stock
            name, a constructed eryn move, or a plain function).
        step_kwargs: Extra keyword arguments for the RecipeStep constructor
            (e.g. RJ plateau knobs).
        combine_kwargs: Extra keyword arguments for the ``GFCombineMove``
            wrapping this stage's moves.
        debug: Stage-level debug override applied to every move that does not
            carry its own ``Move.debug`` (same value semantics).
    """

    _KINDS = ("search", "pe", "rj", "gb_search", "replica_pe")

    #: Kinds that are a SPECIALIZATION of another kind, mapped to the kind
    #: they BEHAVE as. ``Stage.kind`` picks the step class and is what the
    #: recipe prints; this is what gets stamped onto the runtime combine as
    #: ``gf_stage_kind``, which is what the rest of the package switches on.
    #:
    #: ⚠ WHY THIS EXISTS. ``gf_stage_kind`` is tested by literal tuples in
    #: several places -- ``psdmove._coarse_mode`` resolves "auto" to
    #: search_approx only for ``("search", "rj")``, and the delayed-acceptance
    #: PE path keys on ``== "pe"``. Adding a fourth kind therefore did not
    #: extend those tests, it fell THROUGH them: v9's search stages would have
    #: silently resolved the coarse noise likelihood to delayed_acceptance,
    #: i.e. the PE behaviour, in the stages where the wall time is. Mapping
    #: here fixes every such site at once and keeps new kinds from having to
    #: find them all. (v9 runs COARSE_GPU_MODE=off, so that particular
    #: resolution was inert -- which is exactly why it would not have been
    #: noticed.)
    _RUNTIME_KIND = {"gb_search": "rj", "replica_pe": "pe"}

    def __init__(
        self,
        name: str,
        kind: str = "pe",
        moves: typing.Optional[typing.List] = None,
        step_kwargs: typing.Optional[dict] = None,
        combine_kwargs: typing.Optional[dict] = None,
        debug: typing.Optional[typing.Union[bool, dict]] = None,
    ):
        if kind not in self._KINDS:
            raise ValueError(f"Stage kind must be one of {self._KINDS}, got {kind!r}.")
        self.name = name
        self.kind = kind
        self.moves = [Recipe._coerce_move(m) for m in (moves or [])]
        self.step_kwargs = dict(step_kwargs or {})
        self.combine_kwargs = dict(combine_kwargs or {})
        self.debug = debug

    @property
    def runtime_kind(self) -> str:
        """The kind this stage BEHAVES as (see :data:`_RUNTIME_KIND`)."""
        return self._RUNTIME_KIND.get(self.kind, self.kind)

    def move_names(self) -> typing.List[str]:
        return [m.name for m in self.moves]

    def setup(self, ctx: MoveBuildContext) -> RecipeStep:
        """Materialize this stage: run every move's ``setup(ctx)``, wrap, return the step."""
        enabled = set(ctx.curr.engine_info.branch_names)
        runtime_moves = []
        for mv in self.moves:
            if mv.branch is not None and mv.branch not in enabled:
                raise ValueError(
                    f"Move {mv.name!r} targets branch {mv.branch!r} which is not an "
                    f"enabled branch ({sorted(enabled)}). remove the move or add the branch."
                )
            runtime = mv.materialize(ctx)
            # Tag the runtime move with its declarative name so run-level
            # instrumentation (GF_MOVE_TIMING) can label output per move.
            runtime.gf_move_name = mv.name
            # ... and with the stage name, so inner-move logging (e.g. the
            # [MAXLOGL] plateau lines) is stage-labeled instead of "?" --
            # previously only the stage's outer GFCombineMove was stamped.
            runtime.gf_stage_name = self.name
            # Move-level debug wins over stage-level; None leaves the move's
            # env-resolved default untouched.
            _dbg = mv.debug if mv.debug is not None else self.debug
            if _dbg is not None and hasattr(runtime, "set_debug"):
                if isinstance(_dbg, dict):
                    _opts = dict(_dbg)
                    runtime.set_debug(_opts.pop("enabled", True), **_opts)
                else:
                    runtime.set_debug(bool(_dbg))
            runtime_moves.append(runtime)

        if not runtime_moves:
            raise ValueError(f"Stage {self.name!r} has no moves; pop_stage it or add moves.")

        # Console verbosity: the run-level knob feeds every move's internal
        # progress bars (moves read ``self.progress``) and the combine bar,
        # unless the stage/move sets its own explicitly.
        run_verbose = bool(
            getattr(getattr(ctx.curr, "general_info", None), "verbose", False)
        )
        for runtime in runtime_moves:
            if not hasattr(runtime, "progress"):
                runtime.progress = run_verbose
        combine_kwargs = dict(self.combine_kwargs)
        combine_kwargs.setdefault("verbose", run_verbose)
        # Per-move iteration cadence (Move(every=N), user ruling
        # 2026-09-15): stage-local by riding THIS stage's combine — the
        # same shared stock runtime move stays uncadenced in a stage
        # whose descriptor says every=1.
        _everies = [int(getattr(mv, "every", 1)) for mv in self.moves]
        if any(e != 1 for e in _everies):
            combine_kwargs.setdefault("move_every", _everies)
        combined = GFCombineMove(moves=runtime_moves, **combine_kwargs)
        combined.gf_stage_name = self.name
        # ... and the stage KIND, which the coarse-noise mode resolver reads
        # ("auto" -> search_approx in a search/rj stage, delayed_acceptance in
        # a pe stage). It must ride the per-stage COMBINE move rather than the
        # sub-moves: stock moves are shared by name, so the very same PSDMove
        # instance serves noise_search and full_pe, and a static stamp on it
        # would be whichever stage materialised last. GFCombineMove re-stamps
        # its children immediately before each propose instead.
        #
        # The RUNTIME kind, not ``self.kind``: a specialized kind behaves as
        # the kind it specializes everywhere downstream (see _RUNTIME_KIND).
        combined.gf_stage_kind = self.runtime_kind
        # This combine IS the stage's per-iteration entry point: eryn calls
        # its propose exactly once per global-fit iteration. It therefore
        # owns the iteration counter that gets stamped down the move tree
        # (see GFCombineMove.propose); nested combines inherit the stamp
        # rather than minting their own, exactly as gf_stage_kind does.
        combined.gf_is_stage_combine = True
        if not hasattr(combined, "accepted") or combined.accepted is None:
            combined.accepted = np.zeros((ctx.ntemps, ctx.nwalkers))
        return _STEP_CLASSES[self.kind](moves=[combined], **self.step_kwargs)

    def __repr__(self):
        return f"Stage({self.name!r}, kind={self.kind!r}, moves={self.move_names()})"


def scatter_around_injection(
    state: GFState,
    branch_name: str,
    injection_params: np.ndarray,
    spread: float | np.ndarray,
    reverse_transform: typing.Callable | None = None,
    betas: np.ndarray | None = None,
    priors: ProbDistContainer | None = None,
    max_resample_tries: int = 50,
):
    """
    Initialize branch coordinates by scattering walkers around injection parameters.

    For each leaf, draws coordinates from a multivariate Gaussian centered on
    the (transformed) injection parameters.  Higher-temperature chains receive
    proportionally wider scatter when ``betas`` is provided.  Initialized
    leaves are marked as active (``inds = True``).

    When ``priors`` is supplied, any draw that lies outside the prior support
    (``logpdf == -inf``) is rejected and redrawn.  This is essential for
    sampling bases that contain ``arcsin``/``arccos`` transforms (e.g. MBH
    ``sin_beta``, ``cos_iota``) where an out-of-support initial coordinate
    silently produces NaN once the transform pipeline runs, eventually
    surfacing as a CUDA illegal-memory-access in downstream kernels.

    The function modifies ``state`` in-place, so it can be called from
    ``setup_recipe`` (before MCMC) or from a ``RecipeStep.setup_run``
    (between recipe phases).

    Parameters
    ----------
    state : GFState
        Sampler state to modify in-place.
    branch_name : str
        Name of the branch to initialize (e.g. ``"mbh"``, ``"emri"``).
    injection_params : ndarray
        True source parameters in the **physical** (waveform) basis.
        Shape ``(ndim_phys,)`` for a single leaf, or
        ``(nleaves, ndim_phys)`` for multiple leaves.
    spread : float, ndarray
        Controls the width of the Gaussian scatter (in sampling basis).

        * *scalar* -- isotropic standard deviation for every parameter.
        * *1-D array* ``(ndim,)`` -- per-parameter standard deviations.
        * *2-D array* ``(ndim, ndim)`` -- full covariance matrix
          (shared across leaves).
        * *3-D array* ``(nleaves, ndim, ndim)`` -- per-leaf covariance
          matrices.
    reverse_transform : callable, optional
        Converts a single parameter vector from physical basis to
        sampling basis: ``(ndim_phys,) -> (ndim_sampling,)``.
        If *None*, ``injection_params`` are assumed to already be in
        the sampling basis.
    betas : ndarray of shape ``(ntemps,)``, optional
        Inverse-temperature ladder.  When provided the covariance for
        temperature index *t* is scaled by ``1 / betas[t]`` so that
        hotter chains start with a wider scatter.
    priors : ProbDistContainer, optional
        Prior container for ``branch_name``.  When given, walker draws
        outside the prior support are rejected and redrawn (up to
        ``max_resample_tries`` per walker).  Without it, this routine can
        seed walkers that lie outside arcsin/arccos domains, producing
        NaNs once the transform pipeline runs.
    max_resample_tries : int, optional
        Hard cap on resampling attempts per walker before raising.  Only
        used when ``priors`` is supplied.  Default 50 — for any reasonable
        scatter and prior, this is wildly more than needed; the cap exists
        only to surface pathological configs (e.g. the entire scatter
        landing outside the prior) instead of looping forever.
    """
    # TODO: make this better
    coords = state.branches_coords[branch_name]
    ntemps, nwalkers, nleaves_max, ndim = coords.shape

    injection_params = np.atleast_2d(np.asarray(injection_params, dtype=float))
    # Physical → sampling basis
    if reverse_transform is not None:
        injection_sampling = np.array([reverse_transform(p) for p in injection_params])
    else:
        injection_sampling = injection_params

    nleaves_init = injection_sampling.shape[0]
    assert (
        nleaves_init <= nleaves_max
    ), f"More injection leaves ({nleaves_init}) than nleaves_max ({nleaves_max})"
    assert (
        injection_sampling.shape[-1] == ndim
    ), f"Injection ndim ({injection_sampling.shape[-1]}) != branch ndim ({ndim})"

    # Build covariance matrix/matrices
    spread = np.asarray(spread, dtype=float)
    if spread.ndim == 0:
        cov = spread.item() ** 2 * np.eye(ndim)
        covs = np.tile(cov, (nleaves_init, 1, 1))
    elif spread.ndim == 1:
        cov = np.diag(spread**2)
        covs = np.tile(cov, (nleaves_init, 1, 1))
    elif spread.ndim == 2:
        covs = np.tile(spread, (nleaves_init, 1, 1))
    elif spread.ndim == 3:
        assert spread.shape == (nleaves_init, ndim, ndim)
        covs = spread
    else:
        raise ValueError(f"spread must be scalar, 1-D, 2-D, or 3-D; got shape {spread.shape}")

    if betas is not None:
        logger.info(f"Scaling initial covariance by betas: {betas}")

    leaf_prior = priors[branch_name] if priors is not None else None

    for leaf in range(nleaves_init):
        center = injection_sampling[leaf]
        leaf_cov = covs[leaf]
        for t in range(ntemps):
            if betas is not None:
                scaled_cov = leaf_cov / max(betas[t], 1e-10)
            else:
                scaled_cov = leaf_cov

            draws = np.random.multivariate_normal(center, scaled_cov, size=nwalkers)

            if leaf_prior is not None:
                bad = ~np.isfinite(leaf_prior.logpdf(draws))
                tries = 0
                while bad.any():
                    if tries >= max_resample_tries:
                        n_bad = int(bad.sum())
                        raise RuntimeError(
                            f"scatter_around_injection: leaf={leaf} temp={t}: "
                            f"{n_bad}/{nwalkers} walkers still outside prior support "
                            f"after {max_resample_tries} resample passes. "
                            f"Injection sampling-basis params = {center.tolist()}. "
                            f"Likely the injection sits on / outside a prior edge, or "
                            f"the scatter is too wide for the prior range."
                            f"Last resampled points (showing up to 10): {draws[bad][:10].tolist()}"
                        )
                    redraws = np.random.multivariate_normal(
                        center, scaled_cov, size=int(bad.sum())
                    )
                    draws[bad] = redraws
                    bad = ~np.isfinite(leaf_prior.logpdf(draws))
                    tries += 1

            coords[t, :, leaf] = draws

        state.branches_inds[branch_name][:, :, leaf] = True


def mbh_catalogue_to_sampling_basis(catalogue_entry: dict, trim_duration: float = 0.0) -> np.ndarray:
    """Convert a single Mojito MBHB catalogue entry to MBH sampling basis.

    The sampling basis is:
    ``[logM, q, s1z, s2z, dist, phi_ref, cos_iota, psi, lam, sin_beta, t_plunge]``

    Sky / polarization / time parameters are returned in the **SSB
    ecliptic frame** — the sprint-wide sampling frame. (LISA-frame
    sampling is handled by the moves themselves, e.g.
    :class:`lisatools.sampling.moves.skymodehop.SkyMove` with
    ``coord_frame="ssb_ecliptic"``.)

    Parameters
    ----------
    catalogue_entry : dict
        Dictionary of catalogue parameters for one MBHB source, as
        stored by ``L1DataLoader.catalogue['MBHB'][source_id]``.

    Returns
    -------
    np.ndarray
        Parameter vector of shape ``(11,)`` in the MBH sampling basis
        (SSB ecliptic frame for sky/time parameters).
    """
    m1 = float(catalogue_entry["PrimaryMassSSBFrame"])
    m2 = float(catalogue_entry["SecondaryMassSSBFrame"])

    # Ensure m1 >= m2
    if m2 > m1:
        m1, m2 = m2, m1

    logM = np.log(m1 + m2)
    q = m2 / m1
    Q = m1 / m2
    logq = np.log(q)

    s1z = float(catalogue_entry["PrimarySpinCompZ"])
    s2z = float(catalogue_entry["SecondarySpinCompZ"])
    dist = float(catalogue_entry["LuminosityDistance"]) / 1e3  # Mpc -> Gpc
    phi_ref = float(catalogue_entry["PhaseReferenceSourceFrame"]) % (2 * np.pi)
    cos_iota = np.cos(float(catalogue_entry["InclinationAngle"]))

    # Sky coordinates: ICRS -> ecliptic -> SSB -> LISA
    ra = float(catalogue_entry["RightAscension"]) % (2 * np.pi)
    dec = float(catalogue_entry["Declination"])
    sin_dec = np.sin(dec)
    psi_icrs = float(catalogue_entry["PolarisationAngle"]) % np.pi  # ensure polarization is within [0, pi]
    lam_ecl, beta_ecl, psi_ssb = icrs_to_ecliptic(ra, dec, psi_icrs)
    t_ssb = float(catalogue_entry["TimeCoalescencePhenomTPHMSSBFrame"])

    # ICRS sampling basis (stft_tof + 2026-06 run-frame directive): sky and
    # polarization are kept in ICRS (ra, sin_dec, psi_icrs); time stays SSB.
    # Erebor's stock MBH transform (MBHSetup.init_sampling_info /
    # make_mbh_transform_container) uses the same direct-ICRS basis.
    # logger.debug(f"Catalogue entry: RA={ra}, Dec={dec}, psi_icrs={psi_icrs}, t_ssb={t_ssb}")

    # t_L, lam_L, beta_L, psi_L = SSB_to_LISA(t_ssb, lam_ecl, beta_ecl, psi_ssb)

    # lam_L = lam_L % (2 * np.pi)
    # psi_L = psi_L % np.pi
    # logger.debug(f"Converted to LISA frame: t_L={t_L}, lambda_L={lam_L}, beta_L={beta_L}, psi_L={psi_L}")
    # sin_beta_L = np.sin(beta_L)

    #return np.array([logM, Q, s1z, s2z, dist, phi_ref, cos_iota, psi_L, lam_L, sin_beta_L, t_L])
    return np.array([logM, Q, s1z, s2z, dist, phi_ref, cos_iota, psi_icrs, ra, sin_dec, t_ssb])


def gb_catalogue_to_sampling_basis(catalogue_entry: dict, trim_duration: float = 0.0) -> np.ndarray:
    """Converts the (V)GB catalogue entries to the sampling basis. 
    The index 0 in f0 and phi0 refer to the frequency and phase at the start of the data.

    The sampling basis is:
    ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``

    Parameters
    ----------
    catalogue_entry : dict
        Dictionary of catalogue parameters for all (V)GBs, as
        stored by ``L1DataLoader.catalogue['(V)GB'][source_id]``.

    Returns
    -------
    np.ndarray
        Parameter vector of shape ``(8,)`` in the (V)GB sampling basis
        (ICRS or LISA frame for sky/time parameters).
    """
    # VALIDATED mojito GB convention (scripts/gb/gb_mojito_match.py +
    # gb_mojito_mcmc_three_ways.py, mm ~ 1e-8 vs band-passed data):
    # catalogue params are consumed AT the catalogue reference epoch
    # (TimeReferenceSSBFrame == MOJITO_REFERENCE_TIME) with NO trim
    # evolution -- the GB kernels' ``t_ref`` is that same epoch. The
    # PHYSICAL phase is phi0 = +TrueAnomaly; the sampling-basis sign (and
    # every other basis convention) is single-sourced in
    # ``make_gb_transform_container`` -- this function only builds the
    # PHYSICAL row and routes through the container's inverse.
    from .stock.erebor.transforms import make_gb_transform_container

    del trim_duration  # accepted for signature compat; anchor is REF

    amp = np.asarray(catalogue_entry["Amplitude"], dtype=float)
    # Physical row in the container's output basis:
    # [A, f0 (Hz), fdot, fddot, phi0 (+TrueAnomaly), iota, psi, alpha (RA),
    #  delta (Dec)] -- output columns keep the sampling-style names but hold
    # the physical values (see the factory docstring).
    physical = np.stack(
        [
            amp,
            np.asarray(catalogue_entry["GW22FrequencySSBFrame"], dtype=float),
            np.asarray(
                catalogue_entry["GW22FrequencyDerivativeSourceFrame"], dtype=float
            ),
            np.zeros_like(amp),  # fddot
            np.asarray(catalogue_entry["TrueAnomaly"], dtype=float),
            np.asarray(catalogue_entry["InclinationAngle"], dtype=float),
            np.asarray(catalogue_entry["PolarisationAngle"], dtype=float),
            np.asarray(catalogue_entry["RightAscension"], dtype=float),
            np.asarray(catalogue_entry["Declination"], dtype=float),
        ],
        axis=-1,
    )

    # fdot basis regardless of the run's chirp-mass mode: this function's
    # contract is fdot in slot 2 (chirp-mass runs convert downstream).
    tc = make_gb_transform_container(use_chirp_mass=False)
    sampled = tc.both_inverse_transforms(physical)

    # Wrap the periodic parameters into the prior support [0, period).
    input_basis = list(tc.input_basis)
    sampled[..., input_basis.index("phi0")] %= 2 * np.pi
    sampled[..., input_basis.index("psi")] %= np.pi
    sampled[..., input_basis.index("alpha")] %= 2 * np.pi
    return sampled


def gb_fdot_rows_to_run_basis(rows, *, use_chirp_mass, use_fdot_astro,
                              m_chirp_lims, use_distance=False):
    """Convert FDOT-basis GB rows (slot 0 = lnA, slot 2 = fdot) to the run basis.

    ``gb_catalogue_to_sampling_basis`` returns rows with physical ``fdot`` in
    slot 2 (and ``lnA`` in slot 0) by contract; the run may sample different
    slot-0/slot-2 axes. This maps the seeding rows onto whatever the run
    actually samples, using the SAME conventions as
    ``make_gb_transform_container`` (single source of truth):

    * ``use_fdot_astro`` (9-col Mc + ``fdot_astro_ratio``): the mirror
      convention of :class:`~...transforms.McFdotAstroRatioTripleInverse` --
      ``Mc = clip(Mc_GW(f0, |fdot|), m_chirp_lims)`` and
      ``r = fdot / fdot_gr(f0, Mc) - 1`` appended as a new last column.
      Represents interacting ``fdot <= 0`` systems EXACTLY (seeds at
      ``r ~ -2``), resolving the historical fdot<0 mis-modeling.
    * ``use_distance`` (with ``use_fdot_astro``): slot 0 ``lnA`` becomes the
      luminosity distance (kpc), inverted from the physical amplitude and the
      mirror ``Mc`` via ``dist = gb_amp_from_dist(f0, Mc, 1) / exp(lnA)`` --
      reproduces the catalogue amplitude exactly through the forward quad.
    * ``use_chirp_mass`` only (8-col Mc, no ratio): fdot cannot carry a sign,
      so ``fdot <= 0`` sources are clamped to the Mc floor (the legacy
      behavior; still mis-modeled -- only the 9-col basis fixes it).
    * legacy fdot basis: returned unchanged.

    ``rows`` is not mutated; a converted copy is returned.
    """
    rows = np.array(rows, dtype=float, copy=True)
    mc_lims = list(m_chirp_lims) if m_chirp_lims else [0.001, 1.0]
    if use_fdot_astro:
        from .stock.erebor.transforms import (
            McFdotAstroRatioTripleInverse, gb_amp_from_dist,
        )

        f0_hz = rows[:, 1] * 1e-3
        fdot = rows[:, 2]
        _, mc, ratio = McFdotAstroRatioTripleInverse(tuple(mc_lims))(
            f0_hz, fdot, np.zeros_like(fdot)
        )
        if use_distance:
            # slot 0: lnA -> distance (kpc) using the mirror Mc, so the
            # forward quad reproduces the catalogue amplitude exactly.
            rows[:, 0] = gb_amp_from_dist(f0_hz, mc, 1.0) / np.exp(rows[:, 0])
        rows[:, 2] = mc
        rows = np.concatenate([rows, ratio[..., None]], axis=-1)
    elif use_chirp_mass:
        from gbgpu.utils.utility import get_chirp_mass_from_f_fdot

        f0_hz = rows[:, 1] * 1e-3
        fdot = rows[:, 2]
        mc = np.full_like(fdot, float(mc_lims[0]))
        _pos = fdot > 0
        mc[_pos] = get_chirp_mass_from_f_fdot(f0_hz[_pos], fdot[_pos])
        rows[:, 2] = np.clip(mc, float(mc_lims[0]), float(mc_lims[-1]))
    return rows


def setup_state_for_injection(curr: CurrentInfoGlobalFit, state: GFState, source_type: str, branch_name: str, spread: float | np.ndarray  = 1e-5, subset_inds = None, priors: ProbDistContainer | None = None, relative_spread: float | None = None):
    """Initialize 'branch_name' walkers from catalogue injection parameters.

    ``relative_spread`` (the START_FACTOR convention) overrides ``spread``
    with the sprint-wide MULTIPLICATIVE form ``x * (1 + factor * randn)``:
    per-leaf diagonal covariances sized to ``factor`` times each injection
    value (0 -> exact truth).
    """

    catalogue = getattr(curr.general_info, "catalogue", {})
    catalogue = catalogue.get(source_type, {})
    if catalogue:
        injection_params_list = []
        for source_id in sorted(catalogue.keys()):
            entry = catalogue[source_id]
            
            func_name = f"{branch_name}_catalogue_to_sampling_basis"
            conversion_func = globals().get(func_name)

            assert conversion_func and callable(conversion_func), f"catalogue_to_sampling_basis function for {branch_name} was not found."
            assert curr.general_info.preprocess_kwargs

            trim_duration = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME # curr.general_info.data_processor.original_t0
            sampling_params = conversion_func(entry, trim_duration=trim_duration)

            injection_params_list.append(sampling_params)

        injection_params = np.array(injection_params_list)

        # Reshape by the PRODUCED row width (the catalogue basis), NOT the
        # state ndim -- for a 9-col fdot_astro run the state is 9-wide while
        # the conversion emits 8-col fdot-basis rows (widened just below).
        src_ndim = injection_params.shape[-1]
        if injection_params.ndim == 3:
            injection_params = injection_params.reshape(-1, src_ndim)

        if subset_inds is not None:
            injection_params = injection_params[subset_inds, :]

        # Map the FDOT-basis catalogue rows onto the run's sampling basis
        # (Mc + optional fdot_astro_ratio) with the transform's own
        # conventions. The 9-col ratio basis RESOLVES the historical
        # fdot<0 mis-modeling (interacting DWDs seed at r ~ -2, exact); the
        # 8-col Mc basis still floor-clamps them.
        if branch_name == "gb" and getattr(
            curr.source_info[branch_name], "use_chirp_mass", False
        ):
            info = curr.source_info[branch_name]
            injection_params = gb_fdot_rows_to_run_basis(
                injection_params,
                use_chirp_mass=True,
                use_fdot_astro=getattr(info, "use_fdot_astro", False),
                use_distance=getattr(info, "use_distance", False),
                m_chirp_lims=info.m_chirp_lims,
            )

        # Store injection truths for diagnostic plots
        try:
            setattr(curr.source_info[branch_name], "injection", injection_params)
        except AttributeError:
            logger.warning(f"No injection data is saved for {branch_name}.")
        
        if relative_spread is not None:
            _rows = np.atleast_2d(np.asarray(injection_params, dtype=float))
            if _rows.size == 0:
                # Empty ``subset_inds`` (e.g. no catalogue source cleared the
                # SNR cut over this window) used to reach np.stack([]) and
                # raise a bare "need at least one array to stack" from deep
                # inside numpy. Say what actually happened.
                raise ValueError(
                    f"Branch {branch_name!r}: no {source_type} injections to "
                    "seed from — the catalogue subset selected 0 sources. "
                    "Over a short window nothing clears the SNR cut; lengthen "
                    "Tobs (nf/nt), widen the branch band, or lower the "
                    "selection threshold."
                )
            spread = np.stack(
                [np.diag((float(relative_spread) * row) ** 2) for row in _rows]
            )
        scatter_around_injection(
            state, branch_name, injection_params, spread, betas=getattr(curr.source_info[branch_name], "betas"), priors=priors
        )


class GBFillGlobalSignalGen:
    """ONE engine-side ``signal_gen`` for the GB-family branches (gb / vgb).

    Pure Python PREPARATION feeding the installed GBGPU ``fill_global``
    machinery -- it organizes (transform -> physical rows -> index/factor
    arrays -> zeroed template buffer) and passes into the C kernels
    (``fill_global_wdm`` on WDM, ``generate_global_template`` on FD); no
    waveform code of its own. Every branch difference (chirp-mass vs fdot
    basis, the VGB per-leaf fixed f0/sky fills) lives in the branch's
    transform container: per-leaf transforms arrive pre-transformed from
    the engine (``apply_transform=False``), everything else is transformed
    here, so the SAME class serves both branches.

    Extra ``*fill_args`` / ``**fill_kwargs`` given at construction are
    forwarded to every ``fill_global`` call (call-time ``**kwargs`` win).

    The heavy likelihood comp is NOT held here: it is read from (or built
    onto) ``source_info.gb_wdm_comp`` lazily on first call, so the
    pre-build fit config stays picklable and the moves reuse the same comp.
    """

    def __init__(self, branch, transform, general_info, source_info,
                 *fill_args, **fill_kwargs):
        self.branch = branch
        self.transform = transform
        self.general_info = general_info
        self.source_info = source_info
        self.fill_args = tuple(fill_args)
        self.fill_kwargs = dict(fill_kwargs)

    # -- lazy comp -------------------------------------------------------
    def _comp(self):
        comp = getattr(self.source_info, "gb_wdm_comp", None)
        if comp is not None:
            return comp
        from lisatools.domains import WDMSettings
        from gbgpu.gbcomps import GBWDMComputations

        gi = self.general_info
        si = self.source_info
        if not isinstance(gi.domain_settings, WDMSettings):
            raise NotImplementedError(
                "GBFillGlobalSignalGen lazy comp build is WDM-only; on FD "
                "the branch installs its own generator handles."
            )
        _wdm = gi.domain_settings
        _wdm.t0 = float(getattr(gi, "data_t0", 0.0))
        tdi_gen = 2 if getattr(si, "use_tdi2", True) else 1
        comp = GBWDMComputations(
            _wdm,
            t_ref=si.t0,
            Nt_sub=int(si.nt_sub),
            n_pad=int(si.n_pad),
            N_sparse=int(si.n_sparse),
            N_cp_sig=int(si.n_cp_sig),
            N_cp_orbit=int(si.n_cp_orbit),
            # SINGLE-SOURCE WINDOW (2026-07-31 audit): the run's resolved
            # data-window alpha, NOT the comp's internal FAST_WDM default --
            # the two only agreed by coincidence of defaults. The sig-het
            # wrapper inherits this via resolved_tukey_alpha.
            tukey_alpha=float(getattr(gi, "window_alpha", 0.0)),
            orbits=gi.gpu_orbits,
            tdi_config=f"{tdi_gen}{'nd' if tdi_gen == 2 else 'st'} generation",
            force_backend=gi.force_backend,
            tdi_type="XYZ",
        )
        si.gb_wdm_comp = comp  # shared with the move setup (same guard there)
        return comp

    def __call__(self, *params, apply_transform=True, leaf_inds=None, **kwargs):
        """Sum-of-sources template for this branch as a ``DomainBase``.

        ``*params``: sampling-basis rows by default; waveform-basis with
        ``apply_transform=False`` (the engine pre-transforms per-leaf-fill
        branches like vgb, passing ``leaf_inds`` upstream).
        """
        from lisatools.domains import WDMSettings

        comp = self._comp()
        xp = comp.xp
        params_arr = np.asarray(params, dtype=float)
        params_in = (
            self.transform.both_transforms(params_arr, leaf_inds=leaf_inds)
            if apply_transform
            else params_arr
        )
        params_in = xp.atleast_2d(xp.asarray(params_in))

        gi = self.general_info
        settings = gi.domain_settings
        assert isinstance(settings, WDMSettings)
        nchannels = int(getattr(gi, "nchannels", 3))
        shape = (nchannels, int(settings.Nf_active), int(settings.Nt_active))
        template = xp.zeros(int(np.prod(shape)), dtype=xp.float64)

        # single-buffer fill: every row lands in template 0 with factor +1
        data_index = xp.zeros(params_in.shape[0], dtype=xp.int32)
        factors = xp.ones(params_in.shape[0], dtype=xp.float64)
        fk = dict(self.fill_kwargs)
        fk.update(kwargs)
        comp.fill_global_wdm(
            params_in, template, *self.fill_args,
            data_index=data_index, factors=factors, **fk,
        )
        return settings.associated_class(template.reshape(shape), settings)


def select_gb_injection_subset_by_snr(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    gb_info,
    gb_wdm_comp,
    snr_threshold: float = 3.0,
    source_type: str = "GB",
    branch_name: str = "gb",
    f0_lims: typing.Optional[typing.Sequence[float]] = None,
) -> np.ndarray:
    """Select in-band GB catalogue sources by optimal SNR for true-point start.

    Enumerates ``curr.general_info.catalogue[source_type]`` in the SAME sorted
    order as :func:`setup_state_for_injection`, converts each to the sampling
    basis (via ``<branch_name>_catalogue_to_sampling_basis``), masks to the GB
    band (``gb_info.f0_lims``), and computes the optimal SNR
    ``sqrt(<h|h>)`` with the WDM likelihood object ``gb_wdm_comp``
    (``get_ll_wdm`` stashes ``h_h_out``; ``<h|h>`` needs only the PSD in
    ``acs``, so it is data-independent). Returns the array of catalogue-row
    indices whose optimal SNR exceeds ``snr_threshold`` — suitable for the
    ``subset_inds`` argument of :func:`setup_state_for_injection`.
    """
    catalogue = getattr(curr.general_info, "catalogue", {}) or {}
    catalogue = catalogue.get(source_type, {})
    if not catalogue:
        logger.warning(
            f"No '{source_type}' catalogue found; GB SNR-cut injection skipped."
        )
        return np.array([], dtype=int)

    keys = sorted(catalogue.keys())
    conversion_func = globals().get(f"{branch_name}_catalogue_to_sampling_basis")
    assert conversion_func is not None and callable(conversion_func), (
        f"catalogue_to_sampling_basis function for '{branch_name}' was not found."
    )

    trim_duration = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME
    sampling = np.array(
        [conversion_func(catalogue[k], trim_duration=trim_duration) for k in keys]
    )
    ndim = int(np.asarray(sampling).shape[-1])
    # GB/VGB catalogue entries are array-valued over the whole galaxy, so the
    # per-entry conversion returns (N_src, ndim); flatten to (N_total, ndim)
    # exactly as setup_state_for_injection does so subset_inds line up.
    if sampling.ndim == 3:
        sampling = sampling.reshape(-1, ndim)

    # In-band on f0 (sampling basis stores f0 in mHz at index 1).
    f0_hz = np.asarray(sampling[:, 1], dtype=float) * 1e-3
    if f0_lims is None:
        f0_lims = gb_info.f0_lims
    f0_lo, f0_hi = float(f0_lims[0]), float(f0_lims[1])
    in_band = (f0_hz >= f0_lo) & (f0_hz <= f0_hi)
    n_in = int(in_band.sum())
    if n_in == 0:
        logger.info(
            f"GB SNR-cut injection: no '{source_type}' sources in band "
            f"[{f0_lo:.6e}, {f0_hi:.6e}] Hz; state stays at prior draws."
        )
        return np.array([], dtype=int)

    # Optimal SNR via the WDM likelihood (consistent with the fit's likelihood).
    # ``<h|h>`` needs only the (shared, fixed) PSD, so a single walker slab
    # suffices: pass walker-0's AnalysisContainer.  ``get_ll_wdm`` wraps a lone
    # AnalysisContainer into a 1-element ACA under the hood, so the data path is
    # identical to the multi-walker band-buffer case.
    #
    # Build the injection params on ``gb_wdm_comp``'s OWN backend, not the
    # module-level ``cp``: the comp is constructed with the run's
    # ``force_backend`` (often "cpu" even on a GPU node), so a hardcoded
    # ``cp.asarray`` would hand a cupy array to a numpy ``self.xp.atleast_2d``
    # inside ``get_ll_wdm`` and raise the implicit-conversion TypeError.
    #
    # ``gb_catalogue_to_sampling_basis`` returns FDOT-basis rows by contract
    # (slot 2 = fdot, negatives preserved for interacting DWDs), so the
    # physical rows must come from the fdot-basis transform -- the RUN's
    # transform would read slot 2 as Mc on chirp-mass runs (fdot ~1e-15 as
    # Mc -> template fdot ~ 0; fdot < 0 -> (neg)^(5/3) = NaN template).
    from .stock.erebor.transforms import make_gb_transform_container

    _tc_fdot = make_gb_transform_container(use_chirp_mass=False)
    _xp = gb_wdm_comp.xp
    params_phys = _tc_fdot.both_transforms(
        _xp.asarray(sampling[in_band]), xp=_xp
    )
    di = _xp.zeros(params_phys.shape[0], dtype=_xp.int32)
    gb_wdm_comp.get_ll_wdm(params_phys, acs[0], data_index=di, noise_index=di)

    h_h_np = asnumpy(gb_wdm_comp.h_h_out).real
    d_h_np = asnumpy(getattr(gb_wdm_comp, "d_h_out", np.zeros_like(h_h_np))).real
    params_np = asnumpy(params_phys)

    # Optional single-template information_matrix validation (GB_INFO_VALIDATE=1).
    # Since h scales linearly with amplitude A, dh/dA = h/A, so the information matrix
    # diagonal Gamma_AA satisfies Gamma_AA * A^2 == <h|h>. Cross-check the new
    # WDM information_matrix (inds=[0] -> only the amplitude derivative, cheap)
    # against the h_h just computed by get_ll_wdm.
    if bool(int(os.environ.get("GB_INFO_VALIDATE", "0"))) and params_np.shape[0]:
        try:
            _fish = asnumpy(gb_wdm_comp.information_matrix(
                params_phys, acs[0], inds=[0]))[:, 0, 0]
            _pred = _fish * params_np[:, 0] ** 2
            _rel = np.abs(_pred - h_h_np) / np.abs(np.where(h_h_np != 0, h_h_np, 1.0))
            _fin = np.isfinite(_rel)
            logger.warning(
                "GB info-matrix validation: Gamma_AA*A^2 vs <h|h> "
                "median reldiff=%.3e max=%.3e over %d sources (expect <~1e-3).",
                float(np.median(_rel[_fin])) if _fin.any() else float("nan"),
                float(np.nanmax(_rel[_fin])) if _fin.any() else float("nan"),
                int(_fin.sum()),
            )
        except Exception as e:  # noqa: BLE001 -- validation must never break setup
            logger.warning("GB info-matrix validation failed: %s: %s",
                           type(e).__name__, e)

    # Show the physical parameters EXACTLY as they enter the C kernel (post
    # both_transforms, which is what get_ll_wdm flattens into params_in). Dump
    # per-column min/max + values so units/magnitudes can be sanity-checked
    # (GB physical convention: amp, f0[Hz], fdot, fddot, phi0, iota, psi, lam, beta).
    _sampling_in = np.asarray(sampling[in_band])
    _cols = ["amp", "f0[Hz]", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta"]
    _lines = [
        f"    [{j}] {(_cols[j] if j < len(_cols) else 'col'+str(j)):8s}: "
        f"min={params_np[:, j].min():+.6e} max={params_np[:, j].max():+.6e} "
        f"vals={np.array2string(params_np[:, j], precision=6, max_line_width=240)}"
        for j in range(params_np.shape[1])
    ]
    logger.warning(
        "GB SNR-cut params ENTERING C kernel (post-transform), "
        f"{params_np.shape[0]} sources x {params_np.shape[1]} cols:\n"
        + "\n".join(_lines)
        + f"\n    sampling[in_band][0] (pre-transform, 8-col)={np.array2string(_sampling_in[0], precision=6)}"
        + f"\n    params_phys[0] (post-transform, {params_np.shape[1]}-col)={np.array2string(params_np[0], precision=6)}"
    )

    # Diagnostics: a NaN optimal SNR means either bad physical params fed to the
    # kernel or a non-finite <h|h> out of the kernel (e.g. inf invC from a zero
    # PSD cell, or a NaN WDM template). Report which so the fix lands at the
    # right layer.
    param_bad = ~np.isfinite(params_np).all(axis=1)
    hh_bad = ~np.isfinite(h_h_np)
    dh_bad = ~np.isfinite(d_h_np)
    if int(param_bad.sum()) or int(hh_bad.sum()):
        bad_cols = np.where(~np.isfinite(params_np).all(axis=0))[0].tolist()
        logger.warning(
            f"GB SNR-cut diagnostics: {int(param_bad.sum())}/{n_in} sources have "
            f"non-finite physical params (bad param columns={bad_cols}); "
            f"{int(hh_bad.sum())}/{n_in} non-finite <h|h>; "
            f"{int(dh_bad.sum())}/{n_in} non-finite <d|h>."
        )
        # Probe the exact arrays the kernel reads (invC = linear_psd_arr, data =
        # linear_data_arr on the under-the-hood 1-element ACA wrap) to localise
        # the NaN to the PSD/invC vs the template path.
        try:
            _holder = gb_wdm_comp._as_wdm_holder(acs[0])
            _psd = asnumpy(_holder.linear_psd_arr[0])
            _dat = asnumpy(_holder.linear_data_arr[0])
            logger.warning(
                f"GB SNR-cut invC probe: linear_psd_arr[nan={int(np.isnan(_psd).sum())}, "
                f"inf={int(np.isinf(_psd).sum())}, min={np.nanmin(_psd):.3e}, "
                f"max={np.nanmax(_psd):.3e}]; linear_data_arr[nan={int(np.isnan(_dat).sum())}, "
                f"inf={int(np.isinf(_dat).sum())}]."
            )
            # Generate the WDM template directly to isolate template-generation
            # from the inner product: if THIS is NaN, the on-the-fly WDM template
            # (orbit / chunk / t_ref path) is the culprit, not <.|.>.
            _templ = _holder.xp.zeros_like(_holder.linear_data_arr[0])
            gb_wdm_comp.fill_global_wdm(params_phys, _templ, data_index=di)
            _tn = asnumpy(_templ)
            _tabs = np.abs(_tn)
            logger.warning(
                f"GB SNR-cut template probe: fill_global_wdm out "
                f"[nan={int(np.isnan(_tn).sum())}, inf={int(np.isinf(_tn).sum())}, "
                f"nonzero={int((_tn != 0).sum())}/{_tn.size}, "
                f"absmax={(np.nanmax(_tabs) if np.isfinite(_tabs).any() else float('nan')):.3e}]."
            )
        except Exception as e:  # noqa: BLE001 -- diagnostics must never crash the run
            logger.warning(f"GB SNR-cut invC probe failed: {type(e).__name__}: {e}")

    opt_snr = np.sqrt(np.clip(h_h_np, 0.0, None))
    keep = np.isfinite(opt_snr) & (opt_snr > snr_threshold)
    subset_inds = np.where(in_band)[0][keep]
    finite_snr = opt_snr[np.isfinite(opt_snr)]
    logger.info(
        f"GB SNR-cut injection: {n_in} in-band, {int(keep.sum())} with optimal "
        f"SNR > {snr_threshold} (max finite SNR in band = "
        f"{float(finite_snr.max()) if finite_snr.size else 0.0:.2f}); injecting "
        f"{subset_inds.size} true-point leaves."
    )
    return subset_inds


def subtract_gb_neighbors_from_data(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    gb_info,
    gb_wdm_comp,
    *,
    exclude_f0_lims: typing.Sequence[float],
    window_hz: float,
    source_type: str = "GB",
    branch_name: str = "gb",
) -> int:
    """Subtract KNOWN neighbor-band GB templates from every walker's data.

    For focused single-band runs: catalogue sources whose ``f0`` falls
    OUTSIDE ``exclude_f0_lims`` (the sampled band) but within ``window_hz``
    of it are treated as known signals and subtracted from the residual of
    every cold-chain walker (``fill_global_wdm`` with ``factors = -1``), so
    their frequency spread does not bias the in-band fit. Returns the number
    of subtracted sources.

    Pair with ``select_gb_injection_subset_by_snr(..., f0_lims=
    exclude_f0_lims)`` so injected leaves and subtracted neighbors are
    disjoint (a source must never be modeled AND pre-subtracted).
    """
    # TODO(verify GB_SUBTRACT_OUT_OF_BAND, 2026-08-03): this path has never
    # been confirmed to work end to end in a real run. Three things are open.
    #
    # 1. IS IT STILL SLOW? The f0 pre-filter below took the basis conversion
    #    from 11.3 s (15,539,324 catalogue sources, ~2.1 GB of temporaries)
    #    to 0.13 s (the 422 that land in an 8-layer window around 6.25-7.78
    #    mHz) with bit-identical output. But 11 s alone never explained the
    #    reported slowness, so the fix may not be the whole story. Remaining
    #    suspects, in order: the ~3 GB catalogue LOAD itself (upstream of
    #    here, and possibly repeated); memory pressure on a node already
    #    holding the catalogue plus a GPU pool; and gather/scatter of the
    #    residual on multi-shard holders, where the gather is a real copy.
    #    Time this function directly before optimising anything else.
    #
    # 2. DOES IT ACTUALLY REMOVE THE FLOOR? The motivating measurement was a
    #    -40,694 residual floor from out-of-band neighbours plus -1,577 of
    #    edge leakage (the latter is what set GB_SUBTRACT_BUFFER_LAYERS=8).
    #    The check is cheap and immediate: the run's INITIAL log-likelihood
    #    at zero leaves should drop noticeably versus the same run with the
    #    flag off (-107,217 on the 6.11-7.92 mHz, 90 d configuration). If it
    #    does not move, the window caught nothing useful -- compare the
    #    "-> %d to subtract" count against the catalogue f0 range in the same
    #    log line before concluding the subtraction itself is at fault.
    #
    # 3. DOES IT FIX THE REPLACE MOVE? The hypothesis this was reached for:
    #    rj_replace draws a candidate anywhere in the band and phase-maximises
    #    it against the EXPOSED residual, so unmodelled neighbour power is
    #    something a draw can latch onto, scoring a delta the accept path
    #    cannot reproduce. A separate ledger bug in get_replace_ll has since
    #    been fixed (gbbands.py, the delta_old_actual phase recovery), so
    #    these two must be tested SEPARATELY or neither conclusion is clean.
    #
    # Until (1) and (2) are answered, treat this flag as unvalidated rather
    # than merely slow.
    catalogue = getattr(curr.general_info, "catalogue", {}) or {}
    catalogue = catalogue.get(source_type, {})
    if not catalogue:
        logger.warning(
            f"No '{source_type}' catalogue found; neighbor subtraction skipped."
        )
        return 0

    keys = sorted(catalogue.keys())
    conversion_func = globals().get(f"{branch_name}_catalogue_to_sampling_basis")
    trim_duration = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME
    lo, hi = float(exclude_f0_lims[0]), float(exclude_f0_lims[1])

    # PRE-FILTER on f0 BEFORE converting. Selection needs only f0, which the
    # catalogue already carries in Hz as ``GW22FrequencySSBFrame`` -- there is
    # no reason to run the basis conversion on sources about to be discarded.
    # This previously converted the ENTIRE catalogue and masked afterwards,
    # and since gb_catalogue_to_sampling_basis constructs a fresh
    # TransformContainer on every call, that was one container build per
    # catalogue source. It dominated build() badly enough to make
    # GB_SUBTRACT_OUT_OF_BAND unusable on a full-band catalogue. Converting
    # only the survivors makes the cost scale with the number of neighbours
    # actually subtracted (hundreds) instead of the catalogue size.
    def _in_window(f0_hz_arr):
        return (((f0_hz_arr >= lo - window_hz) & (f0_hz_arr < lo))
                | ((f0_hz_arr > hi) & (f0_hz_arr <= hi + window_hz)))

    rows = []
    n_total, f0_min, f0_max = 0, np.inf, -np.inf
    for k in keys:
        entry = catalogue[k]
        f0_entry = np.atleast_1d(
            np.asarray(entry["GW22FrequencySSBFrame"], dtype=float))
        n_total += int(f0_entry.size)
        f0_min = min(f0_min, float(f0_entry.min()))
        f0_max = max(f0_max, float(f0_entry.max()))
        hit = _in_window(f0_entry)
        if not bool(hit.any()):
            continue
        # Sub-select the ENTRY, not the converted rows. A mojito GB catalogue
        # is a SINGLE key holding parallel arrays of ~1.5e7 sources, so
        # converting the whole entry and slicing afterwards would still pay
        # the full cost. Every per-source array (length == f0_entry.size) is
        # masked; scalars and anything else pass through untouched.
        sub = {
            kk: (np.asarray(vv)[hit]
                 if np.ndim(vv) >= 1 and np.shape(vv)[0] == f0_entry.size
                 else vv)
            for kk, vv in entry.items()
        }
        conv = np.asarray(conversion_func(sub, trim_duration=trim_duration))
        rows.append(conv.reshape(-1, conv.shape[-1]))

    sampling = (np.concatenate(rows, axis=0) if rows
                else np.zeros((0, 8), dtype=float))
    n_sub = int(sampling.shape[0])
    # Every retained row is in-window by construction, so the downstream
    # ``sampling[mask]`` is a no-op kept for readability.
    mask = np.ones(n_sub, dtype=bool)
    logger.info(
        "Neighbor subtraction: catalogue f0 [%.6e, %.6e] Hz (%d sources); "
        "window [%.6e, %.6e] Hz -> %d to subtract (converted %d of %d).",
        f0_min, f0_max, n_total,
        lo - window_hz, hi + window_hz, n_sub, n_sub, n_total,
    )
    if n_sub == 0:
        logger.info("Neighbor subtraction: no catalogue sources in the "
                    f"window around [{lo:.6e}, {hi:.6e}] Hz.")
        return 0

    # FDOT-basis physical rows (same contract note as in
    # select_gb_injection_subset_by_snr: the run transform would misread
    # slot 2 as Mc on chirp-mass runs and NaN the fdot < 0 sources).
    from .stock.erebor.transforms import make_gb_transform_container

    _tc_fdot = make_gb_transform_container(use_chirp_mass=False)
    xp = gb_wdm_comp.xp
    params_phys = _tc_fdot.both_transforms(
        xp.asarray(sampling[mask]), xp=xp)
    nwalkers = _local_nwalkers(acs)
    params_tiled = xp.tile(xp.asarray(params_phys), (nwalkers, 1))
    data_index = xp.repeat(
        xp.arange(nwalkers, dtype=xp.int32), n_sub).astype(xp.int32)
    factors = -xp.ones(params_tiled.shape[0], dtype=xp.float64)
    # Multi-shard gather is a COPY: fill it, then scatter back so the
    # subtraction actually lands in the per-shard residual buffers
    # (single-shard gather returns the buffer itself; scatter no-ops).
    _flat_res = acs.gather_linear_data_arr()
    gb_wdm_comp.fill_global_wdm(
        params_tiled, _flat_res,
        data_index=data_index, factors=factors,
    )
    acs.scatter_linear_data_arr(_flat_res)
    logger.info(
        "Neighbor subtraction: subtracted %d known catalogue sources "
        "(window %.3e Hz around [%.6e, %.6e] Hz) from %d walkers.",
        n_sub, window_hz, lo, hi, nwalkers,
    )
    return n_sub


def subtract_initial_signal(
    acs: AnalysisContainerArray,
    state: GFState,
    wave_gen: typing.Callable,
    source_name: str,
    source_info: Setup,
):
    """Subtract pre-injected source templates from the residual buffers in ``acs``.

    Used at run start when a recipe seeds branches with known signal
    parameters (e.g. catalog injections); the corresponding template is
    removed from the residual so the sampler sees only the noise + other
    sources.

    Args:
        acs: Shared :class:`AnalysisContainerArray` whose residuals are
            modified in place.
        state: Current sampler state.
        wave_gen: Waveform generator for ``source_name``.
        source_name: Branch name (e.g. ``"mbh"``, ``"emri"``).
        source_info: Per-source :class:`Setup` providing transforms /
            waveform kwargs.
    """
    xp = acs.xp
    if np.any(inds := state.branches_inds[source_name][0]):
        logger.info(f"Subtracting initial signals for {source_name}")
        counter = 0
        for leaf in range(inds.shape[-1]):
            if inds[0, leaf]:
                assert np.all(inds[:, leaf])
                inj_coords = state.branches_coords[source_name][0, :, leaf]
                inj_coords_in = xp.asarray(source_info.transform.both_transforms(inj_coords))

                # logger.debug(f"CUDA device here: {cp.cuda.runtime.getDevice()}")  # Debugging line to check current CUDA device

                # C-order columns are non-contiguous (stride = ndim*8), so ascontiguousarray
                # is forced to allocate a fresh, pool-aligned buffer for each parameter —
                # avoiding the misalignment that arises with F-order when nwalkers is odd.
                signals_in = wave_gen(*[xp.ascontiguousarray(col) for col in inj_coords_in.T], **source_info.waveform_kwargs)
                for w in range(len(signals_in)):
                    ll_here = acs.acs[w].template_likelihood(template=signals_in[w], include_psd_info=False)
                    logger.debug(f"Initial log-likelihood contribution from walker {w}, leaf {leaf}: {ll_here}")
                acs.add_signal_to_residual(signals_in)
                counter += 1
                
                # if acs.gpus is not None:
                #     acs.synchronize()  # Ensure GPU computations are complete before logging
                #     # acs.xp.get_default_memory_pool().free_all_blocks()
                #     cp.cuda.runtime.setDevice(main_device)  # Switch back to main device after subtraction
                #     logger.debug(f"Switched back to main CUDA device {main_device} after subtraction.")
                    
        logger.debug(f"Subtracted {counter} initial signals for {source_name}")
    else:
        logger.info(f"No initial signals for {source_name}")

    #breakpoint()

def _reference_sens_mat(acs):
    """First AC's sensitivity matrix, or ``None`` if not introspectable.

    Used to decide whether the per-device DCGA replica path is viable.
    Returns ``None`` (meaning "don't block the build") when the ACA has no
    inspectable container list, rather than raising -- keeps stubbed test
    ACAs and non-standard containers on the build path.
    """
    inner = getattr(acs, "acs", None)
    if inner is None:
        return None
    try:
        flat = inner.flatten()
    except AttributeError:
        return None
    if len(flat) == 0:
        return None
    return getattr(flat[0], "sens_mat", None)


def _local_nwalkers(acs) -> int:
    """Walkers this rank's ACA holds (== the global count in a single-process run).

    ``acs_total_entries`` is the AnalysisContainerArray's own row count;
    ``len(acs)`` is the fallback for the lighter ACA stand-ins (test fakes,
    plain container sequences) that never grew the attribute.
    """
    n = getattr(acs, "acs_total_entries", None)
    return int(len(acs) if n is None else n)


def _body_nwalkers(curr, acs, move_class) -> int:
    """Walker width of the state a single-source PE move's BODY runs on.

    The local ACA width (:func:`_local_nwalkers`) everywhere except one
    layout: replicas over SEVERAL walker blocks (``GF_GPU_ROUTING``, e.g.
    2 walkers on 4 ranks = 2 blocks x R=2). There a flat-capable family
    (``fanout_flat_body``; the addremove moves) runs ``propose_local`` on
    the head over the WHOLE ensemble -- its rows are routed to the ranks
    holding each walker -- while every rank's ACA holds only its block. A
    move sized off the ACA would tile its ``data_index`` and size its
    ladders for B walkers against an N-walker state. At one block (one-walker
    replica mode, walker-block mode) the two widths are equal.
    """
    n_local = _local_nwalkers(acs)
    layout = getattr(curr, "rank_layout", None)
    if (
        layout is not None
        and getattr(layout, "replica_mode", False)
        and int(getattr(layout, "n_blocks", 1)) > 1
        and getattr(move_class, "fanout_flat_body", False)
    ):
        return int(layout.nwalkers)
    return n_local


def _local_walker_block(curr, acs):
    """(w0, w1) of the global walkers this rank's ACA rows correspond to."""
    layout = getattr(curr, "rank_layout", None)
    rank = getattr(curr, "rank", None)
    n_local = _local_nwalkers(acs)
    if layout is None or rank is None:
        return 0, n_local
    w0, w1 = layout.block_of(rank)
    if w1 - w0 != n_local:
        raise ValueError(
            f"rank {rank}: layout block [{w0}, {w1}) has {w1 - w0} walkers but the ACA holds "
            f"{n_local} rows."
        )
    return int(w0), int(w1)


def _noise_temperature_control(effective_ndim, nwalkers, *, betas, ntemps, Tmax):
    """The ladder shared by the psd/galfor/sgwb search and PE moves.

    ``walker_inds`` is pinned OUT of the fancy (walker-permuting) swap. It is the
    ACA row ``PSDMove.compute_log_like`` scores against (psdmove.py ~679/1881);
    Eryn scores a fancy swap against the UNPERMUTED destination rows
    (tempering.py ~778-817) and would then swap the label together with the
    coords (tempering.py ~607-614), leaving slot ``w`` pointing at another
    walker's residual for the rest of the propose -- and, under several compute
    ranks, at an ACA row that does not exist on the rank.
    """
    return TemperatureControl(
        effective_ndim,
        nwalkers,
        betas=betas,
        ntemps=ntemps,
        Tmax=Tmax,
        permute=False,
        skip_swap_supp_names=["walker_inds"],
    )


def get_shared_dcga(acs):
    """The run's ONE shared :class:`DomainComputationGroupArray`.

    Built lazily on the first move that wants the per-device replica path
    and cached on the ACA, so it persists for the whole run
    (memory-lifecycle rule: DCGA orbit/sensitivity replicas are allocated
    once, never torn down mid-run). Returns ``None`` on CPU / single-GPU
    runs — the plain shard-aware move paths are already correct there and
    replicas would only add memory.
    """
    if acs.gpus is None or len(acs.gpus) <= 1:
        return None

    # The per-device replica path rebuilds the C++ domain objects on each
    # device, which needs an orbits-carrying sensitivity matrix (i.e. the
    # XYZSensitivityBackend family). The stock runs use
    # CompositeSensitivityBackend, whose CompositeSensitivityMatrix has no
    # ``orbits`` -- so fall back to the plain shard-aware move paths rather
    # than asserting deep inside DomainComputationGroupArray. Those paths
    # are already multi-GPU correct (rows are grouped by ``split_map`` and
    # evaluated inside their owning device context); they just don't get
    # per-device replicas. We fall back ONLY when we can positively see a
    # sens_mat that lacks orbits/kwargs; if the ACA isn't introspectable
    # (or the backend does carry orbits) we build as before.
    _sens = _reference_sens_mat(acs)
    if _sens is not None and not (
        hasattr(_sens, "orbits")
        and hasattr(getattr(_sens, "orbits", None), "kwargs")
        and hasattr(_sens, "kwargs")
    ):
        if not getattr(acs, "_shared_dcga_unavailable_logged", False):
            logger.info(
                "Per-device DomainComputationGroupArray replicas unavailable "
                "(%s carries no orbits/kwargs); using the plain shard-aware "
                "move paths, which are already multi-GPU correct.",
                type(_sens).__name__,
            )
            acs._shared_dcga_unavailable_logged = True
        return None

    dcga = getattr(acs, "_shared_dcga", None)
    if dcga is None:
        from ..domaincomputation import DomainComputationGroupArray

        dcga = DomainComputationGroupArray(acs)
        acs._shared_dcga = dcga
        logger.info(
            "Built the run-shared DomainComputationGroupArray "
            "(%d splits over gpus=%s).", dcga.num_splits, list(acs.gpus),
        )
    return dcga


def build_noise_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    *,
    sampled_branches: list,
    num_repeats: int = None,
    permute_every: int = 50,
    Tmax: float = 1e6,
) -> tuple[PSDMove, PSDMove]:
    """Build a search + PE noise-move pair over ``sampled_branches``.

    Noise-move split (2026-07): each call builds ONE independent move pair
    (its own :class:`TemperatureControl`) that samples exactly
    ``sampled_branches`` — e.g. ``["psd"]``, ``["galfor"]``, ``["sgwb"]`` for
    the fully split stock recipes, or several branches for a joint move. The
    likelihood always evaluates the full noise model; the move freezes
    non-sampled branches at their cold rows (see :class:`PSDMove`).

    Both returned moves share the same ``acs``/``priors`` (by reference) and
    the same ladder object.

    Parameters
    ----------
    engine_info :
        Engine info object exposing ``ndims``.
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info[<branch>]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    sampled_branches : list
        Noise branches THIS move pair samples (subset of
        ``["psd", "galfor", "sgwb"]``, all present in the run). The first
        entry is the "lead" branch whose settings size the ladder and the
        default ``num_repeats``.
    num_repeats : int, optional
        Number of internal move repeats. Default: the lead branch's
        ``num_prop_repeats`` setting.
    Tmax : float, optional
        Maximum temperature for ``TemperatureControl``. Default 1e6.

    Returns
    -------
    search_move : PSDMove
    pe_move : PSDMove
    """
    general_info = curr.general_info
    nwalkers: int = _local_nwalkers(acs)
    if not sampled_branches:
        raise ValueError("sampled_branches must name at least one noise branch.")
    missing = [b for b in sampled_branches if b not in curr.source_info]
    if missing:
        raise ValueError(
            f"build_noise_moves: branch(es) {missing} not in this run "
            f"(sampled_branches={sampled_branches})."
        )
    lead = sampled_branches[0]
    lead_info = curr.source_info[lead]
    psd_info = curr.source_info.get("psd", None)
    galfor_info = curr.source_info.get("galfor", None)
    sgwb_info = curr.source_info.get("sgwb", None)

    # this move's OWN ladder, sized by the lead branch (the engine runs
    # cold-chain only); an explicit betas array wins over the ntemps knob
    ntemps: int = (
        len(lead_info.betas)
        if lead_info.betas is not None
        else int(getattr(lead_info, "ntemps", None) or general_info.ntemps)
    )

    # Loud guard replacing the old silent trap: every sampled branch's
    # sub-state is sized by its OWN ntemps/betas (run.py::_branch_ntemps);
    # the move writes its ladder rows into those sub-states, so the two
    # MUST agree.
    for branch in sampled_branches:
        info_b = curr.source_info[branch]
        nt_b = (
            len(info_b.betas)
            if info_b.betas is not None
            else int(getattr(info_b, "ntemps", None) or general_info.ntemps)
        )
        if nt_b != ntemps:
            raise ValueError(
                f"Noise branches sampled by one move must share a ladder "
                f"size: '{lead}' has ntemps={ntemps} but '{branch}' has "
                f"ntemps={nt_b}. Match the knobs (PSD_NTEMPS / GALFOR_NTEMPS "
                f"/ SGWB_NTEMPS or the branches' betas), or sample them with "
                "separate moves."
            )

    if num_repeats is None:
        num_repeats = int(getattr(lead_info, "num_prop_repeats", None) or 50)

    # the ladder dimension is the sum of the SAMPLED branch ndims
    effective_ndim = sum(engine_info.ndims[b] for b in sampled_branches)
    temperature_control = _noise_temperature_control(
        effective_ndim, nwalkers, betas=lead_info.betas, ntemps=ntemps, Tmax=Tmax
    )

    move_kwargs = dict(
        sampled_branches=list(sampled_branches),
        num_repeats=num_repeats,
        permute_every=permute_every,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform if psd_info is not None else None,
        galfor_transform_fn=galfor_info.transform if galfor_info is not None else None,
        sgwb_transform_fn=sgwb_info.transform if sgwb_info is not None else None,
        sensitivity_backend=general_info.sensitivity_backend,
        temperature_control=temperature_control,
        # Match the run's compute setup: hardcoding True makes eryn's
        # StretchMove call .get() on numpy arrays in CPU runs.
        use_gpu=general_info.gpus is not None,
        # Multi-GPU spread: the unified PSDMove takes the run-shared DCGA
        # directly (one move structure); None on CPU/single-GPU runs. The
        # FD/STFT kernel gate keeps WDM runs on the fallback path.
        dcga=get_shared_dcga(acs),
        run_threaded=acs.gpus is not None and len(acs.gpus) > 1,
        # CPU thread count for the ACA route's per-walker sensitivity builds
        # (see PSDMove.build_threads). getattr: only the noise fits' settings
        # carry the knob; every other fit keeps the serial default.
        build_threads=int(getattr(general_info, "psd_build_threads", 1) or 1),
        # All-source coarse sidecar (plan-2): present only when the run
        # opted into COARSE_GPU_MODE; None leaves every route untouched.
        coarse_runtime=getattr(general_info, "coarse_wdm_runtime", None),
        # Inner proposal knobs (Task 7): eigen-axis MH (one-walker default)
        # or the vanilla eryn stretch, from the lead branch's settings block.
        inner_move_kind=getattr(lead_info, "inner_move_kind", None),
        eigen_refresh_every=int(getattr(lead_info, "eigen_refresh_every", 10) or 10),
        eigen_eps_rel=float(getattr(lead_info, "eigen_eps_rel", 1e-4) or 1e-4),
        # tiled per-walker ensemble search; only fires inside a search stage
        # (PSDMove._ensemble_search_active), so the same pair of move objects
        # is still the ordinary proposal everywhere else.
        ensemble_search=bool(getattr(lead_info, "ensemble_search", False)),
        ensemble_repeats=int(getattr(lead_info, "ensemble_repeats", 10) or 10),
        ensemble_spread_lo=float(getattr(lead_info, "ensemble_spread_lo", 1.0) or 1.0),
        ensemble_spread_hi=float(getattr(lead_info, "ensemble_spread_hi", 10.0) or 10.0),
        ensemble_scale0=float(getattr(lead_info, "ensemble_scale0", 1e-3) or 1e-3),
        ensemble_scale_tries=int(getattr(lead_info, "ensemble_scale_tries", 4) or 4),
    )

    tag = "+".join(sampled_branches)
    search_move = PSDMove(
        acs, priors, max_logl_mode=True, name=f"{tag} search move", **move_kwargs
    )
    pe_move = PSDMove(
        acs, priors, max_logl_mode=False, name=f"{tag} pe move", **move_kwargs
    )

    search_move.accepted = np.zeros((ntemps, nwalkers))
    pe_move.accepted = np.zeros((ntemps, nwalkers))
    search_move.fanout_branches = list(sampled_branches)
    pe_move.fanout_branches = list(sampled_branches)
    search_move.install_walker_fanout(curr)
    pe_move.install_walker_fanout(curr)

    return search_move, pe_move


def build_psd_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    *,
    num_repeats: int = 60,
    permute_every: int = 50,
    Tmax: float = 1e6,
) -> tuple[PSDMove, PSDMove]:
    """Deprecated wrapper: the historical JOINT psd(+galfor)(+sgwb) move pair.

    Use :func:`build_noise_moves` with an explicit ``sampled_branches`` list;
    this wrapper samples every noise branch present in the run under one
    ladder (the pre-split behavior).
    """
    warnings.warn(
        "build_psd_moves is deprecated: use build_noise_moves("
        "sampled_branches=[...]) — the stock recipes now build independent "
        "psd/galfor/sgwb moves.",
        DeprecationWarning,
        stacklevel=2,
    )
    sampled = [b for b in ("psd", "galfor", "sgwb") if b in curr.source_info]
    return build_noise_moves(
        engine_info,
        curr,
        acs,
        priors,
        sampled_branches=sampled,
        num_repeats=num_repeats,
        permute_every=permute_every,
        Tmax=Tmax,
    )


def build_mbh_moves_phenom(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state: GFState,
    permute_every: int = 20,
    wave_gen: typing.Callable = None,
    subtract_initial: bool = True,
    ) -> tuple[typing.Callable, ResidualAddOneRemoveOneMove]:
    """Build MBH PE move using ``PhenomTHMTDIWaveform`` + ``ResidualAddOneRemoveOneMove``.

    Sets ``state.sub_states['mbh'].betas_all`` as a side effect.

    Parameters
    ----------
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["mbh"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    state :
        Current sampler state; ``sub_states["mbh"].betas_all`` is set here.
    wave_gen :
        Optional pre-built ``PhenomTHMTDIWaveform`` instance. When the
        settings file already constructed (and cached) the generator —
        e.g. to register ``source_info['mbh'].signal_gen`` for the
        engine-side residual rebuild — pass it here so the move shares
        the same instance. Default ``None`` builds a fresh one from
        ``mbh_info.initialize_kwargs``.
    subtract_initial : bool
        If ``True`` (default), subtract the state's current MBH templates
        from the residuals here (legacy recipe-side path). Settings files
        that register ``source_info['mbh'].signal_gen`` must pass
        ``False`` — the engine already subtracts during
        ``setup_acs(rebuild_residuals=True)`` and doing it twice corrupts
        the residuals.

    Returns
    -------
    wave_gen : PhenomTHMTDIWaveform
    mbh_pe_move : ResidualAddOneRemoveOneMove
    """
    from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

    mbh_info = curr.source_info["mbh"]

    if wave_gen is None:
        wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
    # Legacy pre-``signal_gen`` residual subtraction. When the settings file
    # registers ``source_info['mbh'].signal_gen`` the engine's
    # ``setup_acs(rebuild_residuals=True)`` already subtracted — pass
    # ``subtract_initial=False`` there (doing it twice corrupts the residuals).
    if subtract_initial:
        subtract_initial_signal(acs, state, wave_gen.get_signals_for_residuals, "mbh", mbh_info)

    # The move construction (make_ladder -> betas_all -> coords_shape ->
    # ResidualAddOneRemoveOneMove, plus the ``mbh_info.betas`` /
    # ``state.sub_states['mbh'].betas_all`` side effects) is the shared
    # single-source machinery in :class:`MBHMoveBuilder`.
    _, mbh_pe_moves = MBHMoveBuilder(
        wave_gen=wave_gen.get_signals_for_residuals, permute_every=permute_every
    ).build(None, curr, acs, priors, state)

    return wave_gen, mbh_pe_moves[0]


@dataclass
class GBWaveformDict(typing.TypedDict):
    dt: float
    T: float
    use_c_implementation: bool
    start_freq_ind: int
    tdi_channel_setup: str
    tdi2: bool
    window: None | str
    window_alpha: float


def _fstat_dist_birth_stamp(default: bool = False) -> bool:
    """Resolve ``GBSpecialBase.rj_fstat_dist_birth`` for an install site.

    The move's own constructor defaults this to ``bool(rj_amp_maximize)``
    (= ``bool(phase_maximize)``), which is the SEARCH convention: it
    resolves False on every pe-named move, so a PE birth drew slot 0 and
    the extrinsic angles at FULL PRIOR WIDTHS even though the epoch
    F-stat center table was already built and sitting right there.

    Install sites that want the F-stat distance-birth proposal say so
    explicitly through this helper — epoch-table centers, a truncated +
    normalized lognormal slot 0 priced on both sides, and (in PE)
    ``pe_extrinsic_draw`` angles drawn and priced on both sides. USER
    RULING 2026-08-28: *"rj_fstat_pe get the same stamp? yes mirror them.
    That would be better than drawing from the full prior widths."*

    ``GB_RJ_FSTAT_DIST_BIRTH`` always wins (the project-wide env-knob
    convention); ``=0`` restores the prior-width path bit-identically by
    switching the RJ step's whole F-stat branch off.

    DEFAULT FLIPPED True -> False (user ruling 2026-09-21): production has
    been exporting ``GB_RJ_FSTAT_DIST_BIRTH=0`` all along, so the ``True``
    default described a path no run was taking while every run paid the
    cost of reading past it. The default now MATCHES production, and a run
    that wants the F-stat distance birth opts in by name. The 2026-08-28
    mirroring ruling above is untouched -- it says search and PE resolve
    the SAME way, which they still do; only the value they agree on moved.
    """
    env = os.environ.get("GB_RJ_FSTAT_DIST_BIRTH")
    return bool(int(env)) if env is not None else bool(default)


def ridge_gibbs_eligible(info) -> bool:
    """Is the ridge-Gibbs fiber move buildable for this branch's basis?

    Decided BY COLUMN NAME on the branch's own sampling basis — never by
    branch name. :class:`~lisatools.sampling.ridge_fiber.McRatioDistFiber`
    resolves ``dist`` / ``Mc`` / ``fdot_astro_ratio`` out of the transform
    container's ``input_basis`` and raises loudly when one is missing, so
    this is exactly the predicate "the (Mc, r, dist) degeneracy this move
    resamples EXISTS on the branch":

    * GB 9-column basis -> eligible (unchanged; the historical gate tested
      only ``fdot_astro_ratio``, and the 9-column basis carries all three).
    * GB 8-column (A / fdot) bases -> not eligible.
    * VGB 6-column chirp basis (``VGB_CHIRP_MASS_BASIS=1``: Mc SAMPLED) ->
      eligible. User ruling 2026-09-16, "VGBs get the ridge-gibbs fiber
      move too": with f0 and sky per-leaf fills, the fiber is the whole
      remaining flat direction and the move resamples it in closed form.
    * VGB 5-column legacy distance basis (Mc a per-leaf FILL) -> not
      eligible, so every script still on that basis is bit-unchanged.

    ``GB_RIDGE_GIBBS=0`` disables it everywhere (shared knob, mirroring the
    sig-het reference-policy knobs the VGB move already reads from the GB
    namespace so the two branches audit identically).
    """
    if os.environ.get("GB_RIDGE_GIBBS", "1") != "1":
        return False
    if getattr(info, "fdot_astro_ratio_max", None) is None:
        return False
    from ..sampling.ridge_fiber import McRatioDistFiber

    basis = list(
        getattr(getattr(info, "transform", None), "input_basis", []) or []
    )
    return all(name in basis for name in McRatioDistFiber._REQUIRED)


def _stamp_temper_seed_base(moves, random_seed):
    """Give every GB/VGB move the run's temper-RNG seed base; return it.

    ``GBSpecialBase._make_temper_rng`` seeds the vertical-swap ``Generator``
    from the per-rank ``_rank_rng_seed`` on a MULTI-rank compute rank and
    from THIS integer everywhere else -- a single-rank run through
    ``_propose_legacy`` and the orchestrator at one compute rank alike, which
    is what makes the two bodies draw the identical stream there (fix round
    5; before it the legacy body ran on OS entropy and the orchestrator on a
    derived seed). At one rank that stream is seeded ONCE, at whichever
    propose first arms the vertical swap, and then persists for the rest of
    the run -- NOT reseeded per propose the way the multi-rank
    ``_rank_rng_seed`` path is (review N-2); "identical stream" means the two
    bodies agree with EACH OTHER, not that either refreshes every propose.

    Domain-separated off ``general_info.random_seed`` with a fixed tag so it
    can never collide with the per-rank seeds ``derive_rank_seed`` spawns off
    the same number. ``None`` when the run sets no seed: every stream stays
    on today's entropy. No user/env knob -- the run seed is the only input.
    """
    base = None
    if random_seed is not None:
        base = int(
            np.random.SeedSequence([int(random_seed), 0x7E4B])
            .generate_state(1, dtype=np.uint32)[0]
        )
    for move in moves:
        # GBSpecialBase declares the attribute; a plain eryn move in the same
        # list (the ridge-gibbs fiber) has no Generator of its own and is
        # deliberately left alone.
        if hasattr(move, "gf_temper_seed_base"):
            move.gf_temper_seed_base = base
    return base


#: domain tags for the two warm-start containers' private Generators
#: (search twin / PE twin), spawned off the PER-RANK ``GBSettings.build_seed``
_WARM_SEED_TAG_SEARCH = 0x3A12
_WARM_SEED_TAG_PE = 0x3A13


def _warm_start_seed(gb_info, tag):
    """Per-rank, domain-separated seed for a warm-start container.

    ``WarmStartComponents.rvs`` draws every birth candidate (floor coin,
    mixture component, normals) from ONE private ``np.random.Generator``.
    Until 2026-10-10 both containers were built with the BARE run seed
    (``general_info.random_seed``), identical on every compute rank, so the
    four walkers of the 9mo run proposed the SAME candidate sequence: the
    first warm propose of job 751 logged identical viable / prior-gated /
    SNR-dropped counts (367392 / 38531 / 74077) on all four ranks, with only
    the device-RNG acceptance differing. Every other GB stream was already
    per rank (priors and the F-stat birth seed off ``build_seed``, the
    vertical swaps off ``_rank_rng_seed``); this is the same rule applied
    here: ``SeedSequence([build_seed, tag])``, ``build_seed`` being
    :func:`~lisatools.globalfit.communication.ranks.rank_build_seed`'s
    per-rank value. ``None`` (no run seed) keeps the container on OS
    entropy, which is independent per rank as before.
    """
    base = getattr(gb_info, "build_seed", None)
    if base is None:
        return None
    return int(np.random.SeedSequence([int(base), int(tag)])
               .generate_state(1, dtype=np.uint32)[0])


def _warm_self_wrap(container, *, gb_info, general_info, tag, name,
                    load_kwargs):
    """The warm move's birth law: ``container``, or the SELF-refit mixer around it.

    USER RULINGS 2026-10-13 (``warmstart/self_refit.py``): from gb_search_2
    on, the run refits its warm start from its own last
    ``GB_WARM_SELF_LAST_K`` rows x every cold walker, and the warm moves
    draw from that self refit ``GB_WARM_SELF_FRACTION`` (0.75) of their
    proposes and from the ORIGINAL warm start the rest ("We should keep the
    original warm start in the fit"). ``GB_WARM_SELF=0`` (default) returns
    ``container`` itself: the move is bit-identical to before. The self
    epochs load with the original's ``load_kwargs`` (Tobs, floor box,
    floor_eps, p_floor, circ_images, use_cupy) and the per-rank seed
    ``_warm_start_seed(gb_info, tag ^ (epoch + 1))``.
    """
    if not bool(getattr(gb_info, "warm_self", False)):
        return container
    from .warmstart.self_refit import WarmSelfMixer, self_epoch_root

    root = self_epoch_root(general_info.main_file_path)
    # Three containers (user rulings 2026-10-15): the ORIGINAL parent warm
    # start, the FULL self refit and the FILTERED one (minus the clusters
    # every walker holds), weighted "original,full,filtered". The SEARCH
    # twin (rj_warm_search) takes ``warm_self_split_search`` ("0.5,0,0.5":
    # 2026-10-15 "20 % old run refit warmstart / 80% self refit warmstart
    # filtered", then 2026-10-16 "Let's split them 0.5/0/0.5 in search" --
    # the 1yr tar 4 found the original still holding ~1,000 sources no walker
    # had while the filtered yield was spent), the PE twin
    # ``warm_self_split_pe`` ("1,1,1": "each 1/3 of the time").
    _is_search = str(name).startswith("rj_warm_search")
    _split = (getattr(gb_info, "warm_self_split_search", "0.5,0,0.5") if _is_search
              else getattr(gb_info, "warm_self_split_pe", "1,1,1"))
    for _old in ("GB_WARM_SELF_FRACTION", "GB_WARM_SELF_FRACTION_SEARCH"):
        if os.environ.get(_old) is not None:
            logger.warning(
                "[GB_WARM_SELF] %s is set but no longer read (2026-10-15): the "
                "warm moves use GB_WARM_SELF_SPLIT_SEARCH / "
                "GB_WARM_SELF_SPLIT_PE ('original,full,filtered').", _old)
    mixer = WarmSelfMixer(
        container, epoch_root=root, split=_split,
        load_kwargs=load_kwargs, build_seed=getattr(gb_info, "build_seed", None),
        tag=int(tag), name=str(name),
        # the one-draw adoption lag (ruling 2026-10-16, GB_WARM_SELF_ADOPT_LAG)
        adopt_lag=bool(getattr(gb_info, "warm_self_adopt_lag", True)))
    logger.info(
        "[GB_WARM_SELF] %s: the original warm start (%d components), the full "
        "self refit and the filtered self refit from %s, split original/full/"
        "filtered %s once an epoch exists (the original until then).",
        name, int(container.n_components), root,
        "/".join(f"{w:g}" for w in mixer.split))
    return mixer


def build_gb_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state: GFState,
    *,
    Tmax: float = 1e6,
    include_search: bool = True,
    include_refit: bool = True,
    pe_move_names: typing.List[str] | None = None,
) -> typing.Tuple[typing.List[GBSpecialBase], typing.List[GBSpecialBase]]:
    """Build GB search and PE moves.

    Both moves share the same ``acs``, ``priors``, and
    ``TemperatureControl`` instance, so updates to ``acs`` (e.g. signal
    subtraction by another branch) are visible to both moves at runtime.

    The GB move classes, per-move flags, and list ordering are the GB
    *reference recipe* and stay here; a settings file steers which of them it
    actually installs through the keyword-only design knobs below (this
    replaces the post-hoc ``[m for m in gb_pe_moves if "prior" in m.name]``
    filtering that GB settings files used to duplicate).

    Parameters
    ----------
    engine_info :
        Engine info object exposing ``ndims``.
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["gb"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    Tmax : float, optional
        Maximum temperature for ``TemperatureControl``. Default 1e6.
    include_search : bool, optional
        When ``False``, return an empty search-move list (PE-only recipes).
        Default ``True``.
    include_refit : bool, optional
        When ``False``, never build the GMM-refit moves even if the refit
        file is present. Default ``True`` (refit is still gated on the file
        existing).
    pe_move_names : list of str, optional
        When given, keep only the PE moves whose ``name`` is in this list
        (and in this order-preserving subset). ``None`` keeps all PE moves.

    Returns
    -------
    gb_search_moves : List[GBSpecialBase]
    gb_pe_moves : List[GBSpecialBase]
    """
    gb_info: GBSetup = curr.source_info["gb"]
    general_info: GeneralSetup = curr.general_info
    nwalkers: int = general_info.nwalkers
    # MOVE/ACA-level sizes (ladders, friend windows, per-move accepted arrays)
    # follow this rank's walker block; the band tables on the sub-state stay
    # engine-wide (see initialize_band_information below). Equal single-process.
    nwalkers_local = _local_nwalkers(acs)
    data_start_freq_ind = int(acs.start_freq_ind[0])

    gb_betas = gb_info.betas
    # GB's OWN ladder size (the engine runs cold-chain only); an explicit
    # betas array wins over the ntemps knob
    ntemps: int = (
        len(gb_betas)
        if gb_betas is not None
        else int(getattr(gb_info, "ntemps", None) or general_info.ntemps)
    )
    gpus: list[int] = general_info.gpus

    domain_settings = general_info.domain_settings

    #* Setting up gbgpu on the correct backend and (if any) gpu(s).
    #* CPU path keeps numpy and avoids the cupy-only setDevice call.
    from gbgpu.gbgpu import GBGPU
    import gbgpu

    gb_force_backend = general_info.force_backend
    _gb_backend = gbgpu.get_backend(gb_force_backend)
    if gpus is not None:
        _gb_backend.set_cuda_device(gpus[0])
    # NOTE: ``GBGPU.__init__`` no longer accepts ``t0``; it gets the
    # reference time from the orbits object's t0. We keep ``gb_info.t0``
    # around because the WDM-domain ``GBWDMComputations`` consumes it
    # directly.
    gb = GBGPU(force_backend=gb_force_backend, orbits=general_info.gpu_orbits)
    # Device pinning happened via ``_gb_backend.set_cuda_device(gpus[0])``
    # above — recipe-level device control never touches module-level ``cp``.
    gb.gpus = gpus

    logger.debug(f"GBGPU initialized with gpus: {gb.gpus} and backend: {gb.backend}")

    #* Make sure that priors are evaluated on gpus (when available).
    # On CPU runs we keep ``use_cupy=False`` because eryn's prior.xp reads
    # ``cp`` unconditionally when ``use_cupy=True`` and raises NameError
    # if cupy isn't installed.
    use_gpu_priors = gpus is not None
    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for _, item in gpu_priors_in.items():
        item.use_cupy = use_gpu_priors
    _gpu_gb_prior = ProbDistContainer(gpu_priors_in, use_cupy=use_gpu_priors)
    # ``priors_in`` is the ORIGINAL insertion-order dict, so a fresh container
    # lays multi-column tuple priors (the (dist,alpha,sin_delta) sky/distance
    # joint and the (f0,Mc) GMM) out at CONSECUTIVE columns. The CPU prior
    # ``priors["gb"]`` may have had ``reset_key_order`` applied to scatter them
    # to their true basis columns (dist->0, f0->1, Mc->2, alpha->6,
    # sin_delta->7); re-apply that same key order here or the GPU prior the GB
    # moves evaluate against is column-misaligned and forbids every RJ birth
    # (logp=-inf on in-band F-stat draws). No-op when the CPU prior was never
    # reordered (its key_order already equals insertion order).
    _cpu_key_order = list(getattr(priors["gb"], "key_order", []))
    if _cpu_key_order and list(_gpu_gb_prior.key_order) != _cpu_key_order:
        _gpu_gb_prior.reset_key_order(_cpu_key_order)
    gpu_priors = {"gb": _gpu_gb_prior}

    nleaves_max_gb = state.branches["gb"].shape[-2]
    
    #* Get band information
    band_edges = gb_info.band_edges
    band_N_vals = gb_info.band_N_vals
    assert band_edges is not None
    assert band_N_vals is not None

    #* This checks if the initialization has any gbs in it (when injecting gbs) and adjusts acs accordingly.
    #* Skipped when a GB ``signal_gen`` is registered — the engine's
    #* setup_acs(rebuild_residuals=True) already subtracted the state's GB
    #* templates. No GB signal_gen exists today, so this stays active.
    # Block-gated (Plan 2, Task 5): this rank's ACA holds only walkers
    # [w0, w1) of the FULL ``state``, so every per-walker row below is
    # sliced to that block before use.
    w0, w1 = _local_walker_block(curr, acs)
    inds_loc = state.branches["gb"].inds[0, w0:w1]
    if getattr(gb_info, "signal_gen", None) is None and inds_loc.sum() > 0:

        coords_out_gb = state.branches["gb"].coords[0, w0:w1][inds_loc]
        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)

        check = priors["gb"].logpdf(coords_out_gb)
        if np.any(np.isinf(check)):

            # check which prior is inf
            inf_indices = np.where(np.isinf(check))[0]
            inf_coords = coords_out_gb[inf_indices]
            logger.error(f"Found {len(inf_indices)} coordinates with inf logpdf under GB priors. Example inf coordinates: {inf_coords[:5]}") 

            logger.info("Prior bounds for GB parameters:")
            for param_name, prior in priors["gb"].priors_in.items():
                logger.info(f"  {param_name}: [{prior.min_val},{prior.max_val}]")
            breakpoint()
            raise ValueError("Starting priors are inf. If injecting, try reducing spread.")

        coords_in_in = gb_info.transform.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

        walker_vals = np.tile(
            np.arange(w1 - w0), (nleaves_max_gb, 1)
        ).transpose((1, 0))[inds_loc]

        data_index_1 = walker_vals  # ((band_inds % 2) + 0) * nwalkers + walker_vals

        # Build index/factor arrays on the ACA's backend (== the run's
        # force_backend, shared by ``gb`` / ``gb_wdm_comp``), NOT the
        # module-level ``cp`` (cupy on any node with cupy importable, even a
        # CPU run) — otherwise a cupy array reaches the numpy ``gb_wdm_comp.xp``
        # / numpy ``gb`` consumer below and raises the implicit-conversion
        # TypeError.
        _xp = acs.xp
        data_index = _xp.asarray(data_index_1).astype(_xp.int32)
        # goes in as -h (subtract initial template from data residual)
        factors = -_xp.ones_like(data_index, dtype=_xp.float64)

        N_vals = band_N_vals[band_inds]

        logger.debug("Generating global GB template")
        if gpus is not None:
            gb.gpus = gpus

        if isinstance(domain_settings, FDSettings):
            #* TODO: add test to make sure the generator matches the general information.
            template_in = deepcopy(acs.linear_data_arr)
            # acs lays walkers out in contiguous blocks of ``len(gpu_splits[0])`` per
            # GPU, so ``walker % num_per_gpu_walker`` recovers the intra-split residual
            # index inside generate_global_template. Required (and only valid) for >1
            # GPU; left None for single-GPU so GBGPU keeps its 1-GPU fast path. Mirrors
            # GBSpecialBase.adjust_sources_in_residual_buffer. (stft_tof fix.)
            num_per_gpu_walker = (
                len(acs.gpu_splits[0]) if (acs.gpus is not None and len(acs.gpus) > 1) else None
            )
            gb.generate_global_template(
                coords_in_in,
                data_index,
                acs.linear_data_arr,
                data_length=acs.data_length,
                factors=factors,
                data_splits=acs.gpu_map,
                num_per_gpu=num_per_gpu_walker,
                N=N_vals,
                **waveform_kwargs,
            )
            max_diff_templates = _xp.abs(template_in[0] - acs.linear_data_arr[0]).max()
            del template_in
            logger.debug(
                f"Global GB template generated with max template in/out diff = "
                f"{max_diff_templates:5e}"
            )
        elif isinstance(domain_settings, WDMSettings):
            if gb_info.gb_wdm_comp is None:
                raise ValueError(
                    "WDM-domain GB initialization requires "
                    "gb_info.gb_wdm_comp; build a GBWDMComputations in the "
                    "settings file and pass it via GBSettings.gb_wdm_comp."
                )
            num_bin = coords_in_in.shape[0]
            xp = gb_info.gb_wdm_comp.xp
            factors_arr = xp.asarray(factors).astype(xp.float64)
            # GB WDM init writes templates into a single flat buffer. The
            # multi-shard gather is a COPY, so scatter the result back into
            # the per-shard buffers (single-shard: gather returns the buffer
            # itself and scatter no-ops).
            _flat_res = acs.gather_linear_data_arr()
            gb_info.gb_wdm_comp.fill_global_wdm(
                coords_in_in,
                _flat_res,
                data_index=xp.asarray(data_index),
                factors=factors_arr,
            )
            acs.scatter_linear_data_arr(_flat_res)
        else:
            raise NotImplementedError(
                f"Domain settings {type(domain_settings).__name__} are not "
                f"supported for GB initialization."
            )

    # Optional post-subtraction diagnostic plot. Only FD/STFT signal containers
    # implement ``.plot``; WDMSignal (and other domains) do not, so guard it.
    _post_sub = acs[0].data_res_arr.data_res_arr
    if hasattr(_post_sub, "plot"):
        _post_sub.plot(channel=0, filename=curr.general_info.artifacts_file_dir + "data_post_subtraction.png")

    #* Check if we need to adjust the band temps, and adjust if required
    adjust_temps = False
    state_band_info = getattr(state, "band_info", None)
    if state_band_info is not None:
        band_info_check = deepcopy(state_band_info)
        adjust_temps = True
        #    del state.band_info

    band_temps = np.tile(np.asarray(gb_betas), (len(band_edges) - 1, 1))
    # Leaf-cap cell grid (user design 2026-08-15): a refinement of the band
    # grid by GBSettings.cap_divisor. Stored alongside band_edges so the
    # resume guard can verify it and the migration script can split the
    # per-band cap state into its children.
    from .state import make_cap_edges

    _cap_edges = make_cap_edges(
        band_edges, int(getattr(gb_info, "cap_divisor", 1) or 1),
        stagger=bool(getattr(gb_info, "cap_stagger", False)),
    )
    _resolved_ntemps = state.sub_states["gb"].initialize_band_information(
        nwalkers, ntemps, band_edges, band_temps, cap_edges=_cap_edges,
        branch_name="gb",
        # Per-walker caps must be allocated HERE, at state init, not
        # lazily at first gate access: the module backend derives its
        # dataset set from storage_arrays() when it resets, so an array
        # that first appeared mid-run would have no dataset to be
        # written into.
        leaf_cap_per_walker=bool(
            getattr(gb_info, "leaf_cap_per_walker", False)
        ),
        # Same rule for the per-(walker, band) search-stage record and the
        # per-walker RJ valve: allocated at state init or the backend has
        # no dataset for them.
        search_stage_per_walker=bool(
            getattr(gb_info, "search_stage_per_walker", False)
        ),
        search_shutoff_per_walker=bool(
            getattr(gb_info, "search_shutoff_per_walker", False)
        ),
    )
    # RUNG RECONCILIATION (2026-08-15, mirrors build_vgb_moves): a resumed
    # store's ladder WINS over the configured one (initialize_band_information
    # warns and returns the stored count). Without this, the explicit
    # assignment below would blow up with a raw numpy broadcast error
    # instead of the graceful path -- re-read the stored ladder and carry
    # the resolved rung count into everything sized by it.
    if int(_resolved_ntemps) != int(ntemps):
        logger.warning(
            "build_gb_moves: using the STORED %d-rung gb ladder over the "
            "configured %d (GB_NTEMPS / explicit betas). Re-rung the store "
            "with scripts/fstat_proposal/fix_vgb_band_temps.py <store.h5> "
            "%d if the configured ladder is what you want.",
            int(_resolved_ntemps), int(ntemps), int(ntemps),
        )
        ntemps = int(_resolved_ntemps)
        band_temps = np.asarray(
            state.sub_states["gb"].band_info["band_temps"]
        ).copy()
        gb_info.betas = band_temps[0].copy()
    # initialize_band_information is idempotent (it used to silently
    # re-initialize on every call due to a broken initialized check, which
    # this assignment relied on): the state may arrive here with band_temps
    # zero-initialized by load_info, so set the actual ladder explicitly.
    state.sub_states["gb"].band_info["band_temps"][:] = band_temps
    if adjust_temps:
        state.sub_states["gb"].band_info["band_temps"][:] = band_info_check["band_temps"][0, :]

    # TODO Check if the block below is needed... I.e., do we need band_inds in brach supplemental?
    # band_inds_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)
    # N_vals_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)

    # if state.branches["gb"].inds.sum() > 0:
    #     f_in = state.branches["gb"].coords[state.branches["gb"].inds][:, 1] / 1e3
    #     band_inds_in[state.branches["gb"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
    #     N_vals_in[state.branches["gb"].inds] = band_N_vals[band_inds_in[state.branches["gb"].inds]]

    # branch_supp_base_shape = (ntemps, nwalkers, nleaves_max_gb)
    # state.branches["gb"].branch_supplemental = BranchSupplemental(
    #     {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
    # )

    #* Assembling args and kwargs
    #* ``fd`` is no longer a positional — the move derives it from
    #* ``acs.settings`` so the same call works for FDSettings and
    #* WDMSettings (and any future domain).
    gb_move_args = (
        gb,
        priors,
        data_start_freq_ind,
        acs.end_shape[0],
        acs,
        band_edges,
        band_N_vals,
        gpu_priors,
    )

    effective_ndim = engine_info.ndims["gb"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers_local, ntemps=ntemps, Tmax=Tmax, permute=False
    )
    gb_move_kwargs = dict(
        waveform_kwargs=gb_info.waveform_kwargs,
        parameter_transforms=gb_info.transform,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info.random_seed,
        force_backend=general_info.force_backend,
        nfriends=nwalkers_local,
        temperature_control=temperature_control,
        # ``use_gpu=True`` (stft_tof) dropped: backend choice is fixed at
        # construction via force_backend per the sprint-wide rule.
        num_repeat_proposals=gb_info.num_repeat_proposals,
        search_kwargs=gb_info.search_kwargs,
        # gb_wdm_comp is None for the FD path (default) and a
        # GBWDMComputations instance for the WDM path. The move's Buffer
        # then dispatches on the AC's DomainSettings to pick the right
        # likelihood engine -- no string-level mode flag. On the FD path the
        # move builds a GBFDComputations prototype (the gb_fd_* kernels
        # replaced the legacy SharedMemory family, 2026-07 rework), which
        # needs the orbits / TDI configuration and the phase reference time.
        gb_wdm_comp=gb_info.gb_wdm_comp,
        gb_fd_comp=getattr(gb_info, "gb_fd_comp", None),
        orbits=getattr(gb_info, "orbits", None),
        tdi_config=getattr(gb_info, "tdi_config", None),
        t_ref=float(getattr(gb_info, "t0", 0.0) or 0.0),
        # GB-sampler verification instrumentation (band residual round-trip /
        # get_ll consistency checks + begin/middle/end band plots). Off unless
        # GB_DEBUG=1; direct kwarg on the move class (no GBSettings field).
        debug=bool(int(os.environ.get("GB_DEBUG", "0"))),
        debug_plot_dir=os.environ.get("GB_DEBUG_DIR", "./gf_output/gb_debug/"),
        # Plot ONLY this (walker, band) cell (all temperatures, one figure
        # per plotted step). Band default None -> central band at plot time;
        # gb_no_foreground setdefaults these to walker 0 / the central GB
        # band.
        debug_plot_walker=int(os.environ.get("GB_DEBUG_PLOT_WALKER", "0")),
        debug_plot_band=(int(os.environ["GB_DEBUG_PLOT_BAND"])
                         if os.environ.get("GB_DEBUG_PLOT_BAND") else None),
        # Which of the traced cell's sources the sequence figures follow:
        # "first" (default), "loudest", or a target f0 in mHz.
        debug_seq_pick=os.environ.get("GB_DEBUG_SEQ_PICK", "first"),
        # Per-band progressive leaf cap (search mode). Armed only when
        # GB_LEAF_CAP_START is set (gb_no_foreground sets it under
        # GB_MODE=search): every band starts capped at that many leaves per
        # (temp, walker) cell; caps advance independently per band through
        # the gate selected below (default = the D/2 lnL-improvement gate).
        # See GBSpecialBase._update_band_leaf_caps.
        leaf_cap_start=(int(os.environ["GB_LEAF_CAP_START"])
                        if os.environ.get("GB_LEAF_CAP_START") else None),
        # Default 3 (2026-08-16 user ruling; was 5, was 50). Under the D/2
        # gate this is a PATIENCE: consecutive iterations WITHOUT a
        # sufficient lnL improvement before a cell's cap advances. The old
        # 50 froze growth for hours; 5 was set when an iteration cost
        # ~55 min, so patience was cheap in wall-clock terms and expensive
        # in nothing. Post-speedup an iteration is ~6 min, and caps now live
        # on the band/8 CAP-CELL grid -- 1,232 cells that must each climb
        # from 1 to hold a full galaxy -- so the ramp, not the wall, is the
        # binding cost. 3 shortens every rung by 40% while still requiring
        # a genuine multi-iteration plateau, so a single unlucky iteration
        # cannot promote a cell.
        leaf_cap_min_iters=int(os.environ.get("GB_LEAF_CAP_MIN_ITERS", "3")),
        # Coarse lnL-improvement cap gate -- THE DEFAULT (2026-08-12): a
        # band holds its cap while the cold chain keeps finding a max ll
        # better than the stored best by >= GB_LEAF_CAP_NDIM/2 (D/2 = 4.0
        # for GBs), and increments once it has not for
        # GB_LEAF_CAP_MIN_ITERS consecutive iterations.
        # GB_LEAF_CAP_LL_IMPROVE=0 restores the nsigma-spread + occupancy
        # gate. It wins over the iter-only mode in the move's precedence,
        # so when ``leaf_cap_iter_only`` is explicitly armed the default
        # here flips back to 0 -- the fixed schedule keeps working without
        # also having to set GB_LEAF_CAP_LL_IMPROVE=0 (an explicit env
        # value still beats both).
        leaf_cap_ll_improve=os.environ.get(
            "GB_LEAF_CAP_LL_IMPROVE",
            "0" if getattr(gb_info, "leaf_cap_iter_only", False) else "1",
        ) == "1",
        leaf_cap_ndim=float(os.environ.get("GB_LEAF_CAP_NDIM", "8")),
        leaf_cap_ll_nsigma=float(os.environ.get("GB_LEAF_CAP_LL_NSIGMA", "3.0")),
        leaf_cap_require_occupancy=bool(
            int(os.environ.get("GB_LEAF_CAP_OCCUPANCY", "1"))
        ),
        # Iteration-only cap advancement (GBSettings.leaf_cap_iter_only /
        # GB_LEAF_CAP_ITER_ONLY): the increment gate is ONLY
        # iters >= leaf_cap_min_iters (lnL-plateau + occupancy skipped).
        leaf_cap_iter_only=bool(getattr(gb_info, "leaf_cap_iter_only", False)),
        leaf_cap_update=True,
        # Per-walker leaf caps (GBSettings.leaf_cap_per_walker /
        # GB_LEAF_CAP_PER_WALKER, 2026-09-22): the cap family gains a
        # walker axis so every walker earns and is gated on its own
        # allowance. False -> the shared cap, bit-identically.
        leaf_cap_per_walker=bool(
            getattr(gb_info, "leaf_cap_per_walker", False)
        ),
        # PER-(WALKER, BAND) SEARCH SCHEDULE (2026-09-23). The opt-SNR
        # floor becomes a function of a (walker, band) stage; the valve
        # below freezes a pair whose source count has converged within the
        # current recipe step. ``search_mode`` is passed EXPLICITLY rather
        # than inferred inside the move from GB_MODE or from the move's
        # name: the recipe is what knows which mode it is building, and a
        # knob that silently follows another is how a config change in one
        # place rearms machinery in another with no line saying so.
        search_stage_per_walker=bool(
            getattr(gb_info, "search_stage_per_walker", False)
        ),
        search_stage_min_iters=getattr(
            gb_info, "search_stage_min_iters", None),
        opt_snr_limit_search_coarse=getattr(
            gb_info, "opt_snr_limit_search_coarse", None),
        opt_snr_limit_search_fine=getattr(
            gb_info, "opt_snr_limit_search_fine", None),
        search_shutoff_per_walker=bool(
            getattr(gb_info, "search_shutoff_per_walker", False)
        ),
        search_shutoff_conv_iter=getattr(
            gb_info, "search_shutoff_conv_iter", None),
        search_mode=(getattr(gb_info, "mode", "pe") == "search"),
        # Leaf-cap CELL grid (GBSettings.cap_divisor / GB_CAP_DIVISOR):
        # caps are enforced per 1/K-of-a-sub-band cell, not per sub-band.
        # 1 -> the pre-2026-08-15 per-band behaviour, bit-identically.
        cap_divisor=int(getattr(gb_info, "cap_divisor", 1) or 1),
        # Staggered cap grid (GBSettings.cap_stagger / GB_CAP_STAGGER):
        # cap edges shifted half a cell so no cap edge equals a band edge.
        cap_stagger=bool(getattr(gb_info, "cap_stagger", False)),
        # Overlapping cap cells (GBSettings.cap_overlap_frac /
        # GB_CAP_OVERLAP_FRAC, 2026-08-23): each cell's enforcement span
        # shares this fraction of its width with each neighbour; caps
        # enforce AND-headroom over all covering cells. Edges unchanged;
        # 0 = today's exact partition bit-identically.
        cap_overlap_frac=float(
            getattr(gb_info, "cap_overlap_frac", 0.0) or 0.0
        ),
        # Sig-het reference policy: built once per repeat block and FIXED
        # (default 0 = no mid-block refresh). GB_SIGHET_REFRESH_EVERY=N>0
        # re-enables the legacy per-source drift refresh (diagnostic);
        # GB_SIGHET_DRIFT_CHECK=1 logs the end-of-block drift metric
        # without changing the sampling. Inert on chunked/FD.
        sighet_refresh_every=int(os.environ.get("GB_SIGHET_REFRESH_EVERY", "0")),
        sighet_refresh_dphase=float(os.environ.get("GB_SIGHET_REFRESH_DPHASE", "0.5")),
        sighet_refresh_min_beta=float(
            os.environ.get("GB_SIGHET_REFRESH_MIN_BETA", "0.1")),
        # Trust region: reject in-model candidates beyond these gates from
        # the block's heterodyne anchor (physical |dlnA| e-folds / carrier
        # phase rad); 0 disables. Inert on chunked-het / FD.
        sighet_trust_dlna=float(os.environ.get("GB_SIGHET_TRUST_DLNA", "1.5")),
        sighet_trust_dphase=float(
            os.environ.get("GB_SIGHET_TRUST_DPHASE", "0.5")),
        # SNR scaling of the amplitude gate: per-source dlnA_max =
        # clip(C/snr_ref, dlna_min, GB_SIGHET_TRUST_DLNA); C=0 -> uniform.
        sighet_trust_snr_c=float(
            os.environ.get("GB_SIGHET_TRUST_SNR_C", "30")),
        sighet_trust_dlna_min=float(
            os.environ.get("GB_SIGHET_TRUST_DLNA_MIN", "0.3")),
        # SNR scaling of the PHASE gate, same shape as the amplitude one.
        # C_phase = 3.456*sqrt(2*T_gate) places the gate at a constant TRUE
        # lnL displacement T (49 -> T=100, 155 -> T=1000) instead of at a
        # constant parameter offset; C=0 (default) keeps the uniform gate.
        # Calibrate with GB_SIGHET_TIER_SCAN before arming.
        sighet_trust_phase_c=float(
            os.environ.get("GB_SIGHET_TRUST_PHASE_C", "0")),
        sighet_trust_dphase_max=float(
            os.environ.get("GB_SIGHET_TRUST_DPHASE_MAX", "20")),
        sighet_anchor_check=os.environ.get(
            "GB_SIGHET_ANCHOR_CHECK", "0") == "1",
        sighet_drift_check=os.environ.get("GB_SIGHET_DRIFT_CHECK", "0") == "1",
        **{
            k: v
            for k, v in gb_info.group_proposal_kwargs.items()
            if k != "num_repeat_proposals"
        },
    )

    # Phase-maximised RJ births for the prior moves (two-quadrature
    # analytic maximisation in the band engines; the accepted phi0 is
    # rotated to the maximum). GB_RJ_PHASE_MAXIMIZE=1 turns it on; the
    # default is OFF and no stock variant overrides it any more. LOGGED
    # (user request 2026-09-22): the 6mo submit script mentions the knob
    # only in a comment, so the value in force was invisible in the logs.
    _rj_phase_max_env = os.environ.get("GB_RJ_PHASE_MAXIMIZE")
    _rj_phase_max = bool(int(_rj_phase_max_env or "0"))
    logger.info(
        "GB RJ phase maximization = %s (GB_RJ_PHASE_MAXIMIZE %s) -> "
        "rj_fstat_search / rj_prior_removal / search rj_replace births "
        "%s.",
        "ON" if _rj_phase_max else "OFF",
        "unset, default" if _rj_phase_max_env is None
        else f"= {_rj_phase_max_env!r}",
        "scored at the two-quadrature maximum, phi0 rotated on accept"
        if _rj_phase_max else "scored at the proposed phase",
    )

    # Custom RJ-birth distribution hook (``GBSettings.rj_birth_distribution``):
    # an eryn duck-typed distribution over the full 8-column GB sampling
    # basis (see ``lisatools.sampling.fstat_proposal
    # .make_gb_rj_birth_container``). When set, the prior RJ moves birth
    # from it instead of the global prior; death factors evaluate its
    # logpdf, so it must stay finite wherever leaves can live (wrap narrow
    # proposals in a ``UniformFloorMixture``).
    _custom_birth = getattr(gb_info, "rj_birth_distribution", None)
    # The custom RJ-birth container (fstat / f0_mchirp astro prior) is
    # array-module agnostic: its component proposals dispatch on the input
    # coords via ``get_array_module`` and ``StackedFStatProposal4D`` builds
    # cupy tables under ``from_cache(use_cupy=True)``, so it runs on GPU
    # coords without the "Implicit conversion to a NumPy array" trip. Birth
    # from it directly when set; otherwise the device-correct global prior.
    _rj_birth_prop = (
        {"gb": _custom_birth} if _custom_birth is not None else gpu_priors
    )

    # In-move F-stat grid fit (GB_FSTAT_FIT_IN_MOVE=1): swap the RJ birth
    # move class so its setup() fits the comb/peak grids against the live
    # residual instead of consuming a prebuilt offline npz. NAMES
    # (2026-08-12 rename, user ruling): rj_fstat_* = the F-stat grid birth
    # moves (search / pe); rj_prior_pe = pure prior births;
    # rj_prior_removal = prior-judged removal. Legacy aliases
    # rj_prior_search / rj_prior resolve in setup_gb_moves.
    _fit_in_move = bool(getattr(gb_info, "fstat_fit_in_move", False))
    _RJBirth = GBSpecialRJFStatGridMove if _fit_in_move else GBSpecialRJPriorMove
    _fit_kwargs = {}
    # PE-STAGE REFIT CADENCE: ELAPSED iterations in PE, against the search
    # stages' own value. 250 by user ruling 2026-09-24 (and again
    # 2026-10-07); 150 by user ruling 2026-10-12 ("PE fstat fit should be
    # based on a randomly chosen walker residual. And we should do a refit
    # on that every 150 iterations" -- the walker rule is
    # GB_FSTAT_PE_REF=random, the PE default since 2026-10-06). Separate
    # because full_pe is a random_choice stage -- the GB move is drawn
    # roughly one iteration in N, so a search-tuned cadence there would
    # refit far more often in wall-clock terms than the number suggests
    # (and, before the clock was changed to count elapsed iterations,
    # far LESS often than it said). The launchers export the same value.
    # SINCE 2026-10-12 this cadence belongs to the LIVE-residual PE move
    # (rj_fstat_pe_residual); the GB-free galaxy move rj_fstat_pe fits once
    # (GB_FSTAT_REFIT_EVERY_PE_GB_FREE=0) -- see pe_fstat_move_config.
    _pe_fstat_gb_free = pe_fstat_move_config("rj_fstat_pe")
    _pe_fstat_residual = pe_fstat_move_config(PE_FSTAT_RESIDUAL_MOVE)
    if _fit_in_move:
        _fit_dir = getattr(gb_info, "fstat_fit_dir", "") or os.path.join(
            os.path.dirname(str(general_info.main_file_path)), "gb_fstat_fit")
        _fit_kwargs = dict(
            fstat_fit_dir=_fit_dir,
            fstat_refit_every=int(getattr(gb_info, "fstat_refit_every", 0)),
            fstat_fit_kwargs=dict(
                A_lims=gb_info.A_lims,
                mc_lims=list(getattr(gb_info, "m_chirp_lims", None) or []),
                dist_lims=(gb_info.dist_lims
                           if getattr(gb_info, "use_distance", False) else None),
                fdot_astro_ratio_max=(
                    gb_info.fdot_astro_ratio_max
                    if getattr(gb_info, "use_fdot_astro", False) else None),
                # PER-RANK build seed (GBSettings.build_seed, stamped by
                # stock/base.py::prepare_branch_settings from
                # communication.ranks.rank_build_seed). The move derives its
                # per-epoch RJ birth seed from it (_birth_seed); None keeps
                # every birth stream on OS entropy, as before.
                build_seed=getattr(gb_info, "build_seed", None),
            ),
        )
        if _custom_birth is not None:
            logger.warning(
                "GB_FSTAT_FIT_IN_MOVE=1 supersedes the supplied "
                "rj_birth_distribution; drop a prebuilt grid into "
                "%s/<move name>/epoch_0000/ -- WITH a DONE.json beside it, "
                "since under the multi-rank fan-out every rank (the head "
                "included) refuses an epoch whose manifest is missing -- to "
                "have it loaded instead.",
                _fit_dir,
            )

    #* ============================================= SEARCH MOVES =============================================
    # GB tempering placement (user ruling 2026-08-14): when the
    # prior-removal companion is in the search cycle, the band-temperature
    # swap stage (426 s/propose at 3 mo; temper_buffer 389 s of twin
    # builds + fills) runs INSIDE ``rj_prior_removal`` -- the LAST RJ
    # move of the per-iteration cycle, whose alive-only cell set is also
    # the natural future host for swap-fill reuse -- instead of inside
    # the long fstat-birth propose. Exactly ONE GB move tempers per
    # iteration either way: when the removal move is absent (PE mode /
    # search_prior_removal off) the fstat move keeps the swaps.
    # GB_TEMPER_ON_REMOVAL=0 restores the old placement.
    _temper_on_removal = (
        getattr(gb_info, "mode", "pe") == "search"
        and getattr(gb_info, "search_prior_removal", False)
        and os.environ.get("GB_TEMPER_ON_REMOVAL", "1") == "1"
    )
    # USER RULING 2026-08-26 (supersedes the 2026-08-14 single-carrier
    # rule): in SEARCH, band-temperature swaps run in ALL THREE cycle
    # moves (fstat birth, replace, prior removal), not one designated
    # carrier — with GB_TEMPER_EVERY_PROPOSES=1 that is 3 swap passes
    # per iteration, interleaved BETWEEN the moves so each move acts on
    # post-transport state. The shared cadence census still throttles
    # globally (cadence 3 + all-moves = once per iteration, any
    # carrier). GB_TEMPER_ALL_MOVES=0 restores the single-carrier rule.
    _temper_all_moves = (
        getattr(gb_info, "mode", "pe") == "search"
        and os.environ.get("GB_TEMPER_ALL_MOVES", "1") == "1"
    )
    # Per-class in-model repeat defaults (user ruling 2026-08-15): the
    # recipe's OWN mode flag is authoritative over the {BRANCH}_MODE env
    # fallback inside the move (resolution: explicit kwarg > env knob >
    # *_default kwarg here > env-mode fallback). SEARCH moves polish
    # newborns hard and survivors lightly (200/25); PE keeps the stock
    # num_repeat_proposals for both classes (so lite presets stay cheap).
    # Search-NAMED moves get the search dict unconditionally (they only
    # run in search recipes); the pe-named moves resolve by mode below.
    # GB birth SNR floor (user ruling 2026-08-26): SEARCH-cycle moves 8,
    # PE-cycle moves 5. Search hunts above the noise ("SNR 5 = noise,
    # keep 8" — the peak-floor ruling's birth-side counterpart; the
    # SNR-5 floor measurably ballooned the hot ladder with noise
    # births); PE refines an assembled model where the faint tail must
    # stay reachable. GB_OPT_SNR_LIMIT_SEARCH / GB_OPT_SNR_LIMIT_PE
    # override per stage; generic GB_OPT_SNR_LIMIT overrides both.
    _snr_search = float(os.environ.get(
        "GB_OPT_SNR_LIMIT_SEARCH", os.environ.get("GB_OPT_SNR_LIMIT", "8.0")))
    _snr_pe = float(os.environ.get(
        "GB_OPT_SNR_LIMIT_PE", os.environ.get("GB_OPT_SNR_LIMIT", "5.0")))
    _imr_search = {"inmodel_repeats_newborn_default": 200,
                   "inmodel_repeats_survivor_default": 25,
                   "opt_snr_rej_samp_limit": _snr_search}
    _imr_pe = {
        "inmodel_repeats_newborn_default": gb_info.num_repeat_proposals,
        "inmodel_repeats_survivor_default": gb_info.num_repeat_proposals,
        "opt_snr_rej_samp_limit": _snr_pe,
    }
    gb_search_prune_move = _RJBirth(
        *gb_move_args,
        rj_proposal_distribution=(None if _fit_in_move else _rj_birth_prop),
        **({"is_rj_prop": True} if _fit_in_move else {}),
        **_fit_kwargs,
        name="rj_fstat_search",
        # False (2026-08-13): use_prior_removal=True makes the sorter
        # ALIVE-ONLY (gbbands keep_all_inds gate) -- no dead-slot birth
        # candidates, i.e. NOT a birth engine at all (zero-row sorter crash
        # on the search start once the empty-model guard was fixed). The
        # search birth engine is fstat births + standard death flips (the
        # verify-proven config); prior-judged pruning is the separate
        # rj_prior_removal move in the same stage.
        use_prior_removal=False,
        phase_maximize=_rj_phase_max,
        run_swaps=_temper_all_moves or not _temper_on_removal,
        gpus=[],
        # FLIP DEFAULT WAS MISSING HERE (found 2026-09-14 on the 3mo/1yr
        # overnight runs): every other RJ move carried
        # ``rj_flip_fraction_default``, this one -- the production search
        # birth move -- did not, so it resolved to the ctor default 1.0 and
        # visited EVERY dead row each iteration (jobs 474/476/480: births
        # drawn ~78-90% of all dead rows, the remainder being at-cap cells),
        # while the 08-28 "0.2 everywhere" ruling and the 09-11 stage knob
        # only ever thinned the other moves. Now wired like the rest.
        **{**gb_move_kwargs,
           "rj_flip_fraction_default": _search_rj_flip_default(),
           **_imr_search}
    )
    # SEARCH F-stat refits sweep the LIVE residual of the MIN-lnL cold
    # walker -- the GBs stay SUBTRACTED, so the peaks are what the search
    # has left to find. PE keeps the pre-existing method: add the GBs back
    # in so the data looks un-subtracted, ranked by max lnL (user ruling
    # 2026-09-21). Set as an attribute rather than a ctor kwarg because
    # ``_RJBirth`` is ``GBSpecialRJPriorMove`` when GB_FSTAT_FIT_IN_MOVE=0,
    # which has no such parameter and would raise on it; the flag is a
    # no-op on that class, which never fits a grid.
    #
    # TODO (PE, undecided 2026-09-21): PE deliberately KEEPS the GB-free
    # grid -- an assembled model's births must stay able to reach a source
    # any one walker has already subtracted, which is exactly what the
    # GB-free residual preserves and what fitting the live residual would
    # take away. Revisit once the search-side change has a snapshot behind
    # it: if the shrinking peak list turns out to cost PE nothing (compare
    # completeness/purity either side of the full_pe handoff), the two
    # stages could share one live-residual grid again and the ``shared`` /
    # ``shared_search`` root split could collapse back to one.
    gb_search_prune_move.fstat_search_residual = True
    gb_search_prune_move.accepted = np.zeros((ntemps, nwalkers_local))
    gb_search_prune_move.install_walker_fanout(curr)
    
    # LEGACY FD-ONLY (2026-08-12 user ruling): the serial-MCMC moves score
    # through GBGPU's FD get_fstat_ll/get_ll (para_log_like), which
    # hard-errors on a WDM basis. They are in NO default recipe; the default
    # GB RJ stack in both modes is the F-stat grid birth move
    # (rj_prior / rj_prior_search) + prior RJ.
    gb_search_fstat_mcmc_move = GBSpecialRJSerialSearchMCMC(
        *gb_move_args,
        rj_proposal_distribution=None,
        is_rj_prop=True,
        run_swaps=False,
        name="rj_fstat_mcmc_search",
        phase_maximize=_rj_phase_max,
        gpus=[],
        # Leaf-cap counters advance once per iteration: the prior RJ move is
        # the designated updater; the other RJ moves only enforce the gate.
        **{**gb_move_kwargs, "leaf_cap_update": False,
           "rj_flip_fraction_default": _search_rj_flip_default(), **_imr_search}
    )
    gb_search_fstat_mcmc_move.accepted = np.zeros((ntemps, nwalkers_local))
    gb_search_fstat_mcmc_move.install_walker_fanout(curr)

    # The RJ refit moves load a GMM-refit proposal file (``main_file_path``)
    # produced during a run. When it is absent (fresh run / smoke, or refit
    # disabled) the refit moves are optional and skipped: the search refit move
    # is already excluded from ``gb_search_moves``, and the PE refit move is
    # dropped from ``gb_pe_moves`` below. This keeps the prior + fstat moves
    # (incl. GBSpecialRJPriorMove) buildable without the refit artifact.
    _refit_fp = getattr(general_info, "main_file_path", None)
    _refit_available = include_refit and isinstance(_refit_fp, str) and os.path.exists(_refit_fp)

    if _refit_available:
        gb_search_refit_move = GBSpecialRJRefitMove(
            *gb_move_args,
            rj_proposal_distribution=None,
            is_rj_prop=True,
            run_swaps=False,
            name="rj_refit_search",
            fp=_refit_fp,
            phase_maximize=_rj_phase_max,  # gb_info["pe_info"]["rj_phase_maximize"],
            gpus=[],
            # _imr_search like every other search-named RJ move (user rule
            # 2026-09-26: "our newborn/survivor rules should be the same for
            # all rj moves"). Without it this move alone fell through to the
            # move's own env-mode fallback -- the same silent
            # knob-reaches-nothing shape that has bitten this run before.
            **{**gb_move_kwargs, "leaf_cap_update": False,
               "rj_flip_fraction_default": _search_rj_flip_default(),
               **_imr_search}
        )
        gb_search_refit_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_search_refit_move.install_walker_fanout(curr)

    # gb_search_refit_move, Refit currently not used for search
    gb_search_moves = (
        [gb_search_fstat_mcmc_move, gb_search_prune_move] if include_search else []
    )

    #* ===================== SEARCH RJ CYCLE (GB_MODE=search only) =====================
    # Companions to the fstat-birth move in the gb_search cycle: the
    # fixed-dimension REPLACEMENT move (exact MH, 2026-08-24 redesign) and
    # the removal-only pruning move. The variant's recipe setup inserts
    # them right after the fstat-birth move in its stage, so the stage's
    # GFCombineMove runs the per-iteration cycle fstat-birth ->
    # fstat-REPLACE -> prior-REMOVAL in list order (eryn CombineMove.propose
    # is sequential). ``leaf_cap_update`` stays designated on the birth
    # move alone (the first move of the cycle) so the cap counters advance
    # exactly once per iteration; the companions only enforce the gate.
    #
    # rj_replace (reinstated 2026-08-24, USER directive): draws full
    # 9-column candidates from the F-stat machinery -- intrinsics from the
    # same fitted grid container as rj_fstat_search, extrinsics from the
    # epoch CENTER TABLE (stored phi0/iota/psi maxima + truncated-lognormal
    # slot 0) -- and scores the EXACT likelihood of those concrete
    # parameters (never phase-maximized; the old phase-max acceptance was
    # the root-caused rj_replace ll-drift flaw, so ``phase_maximize`` is
    # forced False here AND hard-coded off at the scoring site). Under
    # fstat_fit_in_move it is a grid move sharing the birth move's fit dir:
    # the _FSTAT_GRID_REGISTRY / epoch caches hand it the identical
    # container + center table with no second fit.
    _gb_mode_search = getattr(gb_info, "mode", "pe") == "search"
    gb_replace_move = None
    if _gb_mode_search and getattr(gb_info, "search_rj_replace", False):
        gb_replace_move = _RJBirth(
            *gb_move_args,
            rj_proposal_distribution=(None if _fit_in_move else _rj_birth_prop),
            **({"is_rj_prop": True} if _fit_in_move else {}),
            **_fit_kwargs,
            name="rj_replace",
            rj_replace=True,
            phase_maximize=False,
            run_swaps=_temper_all_moves,
            gpus=[],
            **{**gb_move_kwargs, "leaf_cap_update": False,
               "rj_flip_fraction_default": _search_rj_flip_default(), **_imr_search},
        )
        # The center-table recentering IS this move's proposal, so it must
        # not depend on the phase-max chain the ctor default follows
        # (phase_maximize=False -> rj_amp_maximize False -> the F-stat
        # distance path off, leaving uniform prior extrinsics). The env
        # knob GB_RJ_FSTAT_DIST_BIRTH still wins when set explicitly.
        gb_replace_move.rj_fstat_dist_birth = _fstat_dist_birth_stamp()
        # Search-stage stamp (user ruling 2026-08-28): this install site is
        # search-only (_gb_mode_search above), and the move's plain
        # "rj_replace" name carries no stage info -- the stamp is what arms
        # _replace_fstat_max's auto mode (slot 0 pinned at the F-stat
        # center, then priced through the unchanged RJ densities:
        # maximize-then-pretend). A PE replace install must NOT set this.
        gb_replace_move.replace_search_stage = True
        # Search-stage companion of rj_fstat_search: it must land on the
        # SAME epoch root, so it carries the same live-residual mode. With
        # the flag left False it would fit a second, GB-free epoch under
        # ``shared`` and the "no second fit" sharing described above would
        # quietly stop holding.
        gb_replace_move.fstat_search_residual = True
        gb_replace_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_replace_move.install_walker_fanout(curr)
    # Pure IN-MODEL move (2026-08-04): no RJ step at all -- ``is_rj_prop=False``
    # skips the birth/death branch in the round loop, so every pick round is
    # just ``num_repeat_proposals`` in-model repeats on the live sources.
    # Placed between the fstat-birth move and the removal move so freshly-born
    # sources get a full refinement pass to climb the likelihood BEFORE
    # ``rj_prior_removal`` judges them for death (a source still sitting at its
    # birth coordinates looks far more deletable than the same source after it
    # has walked onto its peak).
    #: The in-model SLOT names the v9 three-stage search cycle installs, in
    #: cycle order. Each is named for the RJ move it POLISHES: the
    #: warm-start births, the F-stat births, and the replacement swaps
    #: (user spec 2026-09-24, amended the same day to end the cycle on
    #: ``rj_prior_removal`` -- so every birth and every swap gets a full
    #: in-model refinement pass BEFORE the removal judge sees it, and a
    #: source is never judged at coordinates it has not been given a chance
    #: to walk away from).
    #:
    #: THREE INSTANCES, not one listed three times, because a stage's move
    #: names must be unique (``Recipe._check_unique``) and because
    #: ``Stage.setup`` stamps ``gf_move_name`` onto the runtime object -- one
    #: shared object in three slots would report all three passes under
    #: whichever name was stamped last, and pool their ``accepted`` counters,
    #: which is exactly the per-slot accounting the cycle exists to expose.
    #: They are cheap: no birth distribution, no F-stat grid, no cap counters;
    #: ``gb_move_args`` are shared references and ``band_temps`` lives in the
    #: STATE, so the three cannot drift apart.
    #:
    #: The first keeps the historical bare name so single-stage recipes, the
    #: ``gb_no_fg`` auto-insertion and every existing runbook still resolve.
    GB_IN_MODEL_SLOTS = ("in_model", "in_model_fstat", "in_model_replace",
                         "in_model_removal")

    gb_in_model_moves = []
    if _gb_mode_search and getattr(gb_info, "search_in_model", False):
        for _slot in GB_IN_MODEL_SLOTS:
            _mv = GBSpecialStretchMove(
                *gb_move_args,
                rj_proposal_distribution=None,
                is_rj_prop=False,      # THE switch: in-model repeats only
                name=_slot,
                phase_maximize=False,  # in-model scoring is at the actual phase
                run_swaps=False,
                gpus=[],
                # Cap counters advance exactly once per iteration on
                # ``rj_prior``; this move must not touch them (it changes no
                # dimensions).
                **{**gb_move_kwargs, "leaf_cap_update": False},
            )
            _mv.accepted = np.zeros((ntemps, nwalkers_local))
            _mv.install_walker_fanout(curr)
            gb_in_model_moves.append(_mv)
        logger.info(
            "build_gb_moves: %d pure in-model slots registered %s. Each runs "
            "%d repeats per source per PASS, and -- when "
            "GB_INMODEL_GROUP=1 -- an ADAPTIVE number of passes, ending when "
            "every occupied (walker, band) sub-band's cold-chain logL has "
            "plateaued or the pass ceiling hits.",
            len(gb_in_model_moves), list(GB_IN_MODEL_SLOTS),
            int(gb_info.num_repeat_proposals),
        )
    gb_in_model_move = gb_in_model_moves[0] if gb_in_model_moves else None

    gb_prior_removal_move = None
    if _gb_mode_search and getattr(gb_info, "search_prior_removal", False):
        # rj_removal_only: True = the original pruning move (births
        # force-rejected, deaths judged against the prior); False = the
        # prior container proposes BOTH births and deaths. The hand edit
        # of 2026-09-21 (6mo jobs 596+, stored iteration 300 on) ran
        # False; promoted to ``GBSettings.search_prior_removal_only`` /
        # GB_SEARCH_PRIOR_REMOVAL_ONLY on 2026-09-22 -- default True
        # (user ruling), the 6mo submit script exports 0.
        _prior_removal_only = bool(
            getattr(gb_info, "search_prior_removal_only", True))
        logger.info(
            "rj_prior_removal: rj_removal_only = %s (%s) "
            "[GB_SEARCH_PRIOR_REMOVAL_ONLY]",
            _prior_removal_only,
            "deaths only" if _prior_removal_only
            else "prior births AND deaths",
        )
        gb_prior_removal_move = GBSpecialRJPriorMove(
            *gb_move_args,
            rj_proposal_distribution=gpu_priors,  # THE prior container
            name="rj_prior_removal",
            rj_removal_only=_prior_removal_only,
            # Follows GB_RJ_PHASE_MAXIMIZE, same as rj_fstat_search (user
            # ruling 2026-09-02: "phase maximization on for the prior
            # removal just like fstat"). It arms the move's IN-MODEL repeat
            # scoring (two-quadrature, rotation-on-accept) and, when births
            # are enabled (removal_only False), the prior births exactly as
            # in rj_fstat_search; deaths always keep the true phase, so the
            # removal-only mode is unaffected. The old hard False also leaned on
            # amp-pin/dist-birth ctor defaults following it; both are now
            # independent knobs (GB_RJ_AMP_MAXIMIZE decoupled 0e4e45d7,
            # GB_RJ_FSTAT_DIST_BIRTH env-pinned), so nothing else moves.
            phase_maximize=_rj_phase_max,
            # Band-temperature swaps live HERE when _temper_on_removal
            # (see the placement comment at the search-move block).
            run_swaps=_temper_all_moves or _temper_on_removal,
            gpus=[],
            # Removal pools are 100% mature -- the survivor budget (25)
            # is the one that binds here.
            **{**gb_move_kwargs, "leaf_cap_update": False,
               "rj_flip_fraction_default": _search_rj_flip_default(), **_imr_search},
        )
        gb_prior_removal_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_prior_removal_move.install_walker_fanout(curr)

    #* ===================== WARM-START RJ BIRTH MOVE (rj_warm_search) =====================
    # Workstream B (user ruling 2026-08-24: "add this proposal in GB search
    # ... BEFORE the fstat proposal"). GB_WARM_START_COMPONENTS (or
    # ``GBSettings.warm_start_components``) points at the clustered
    # previous-run posterior components npz produced OFFLINE by
    # lisatools.globalfit.warmstart.fit_from_store; when set (and a search recipe
    # is being built) this constructs ``rj_warm_search`` -- the EXISTING
    # prior-RJ move class with its birth distribution swapped for the
    # warm-start f0-windowed Gaussian mixture
    # (lisatools.globalfit.warmstart.proposal.WarmStartComponents):
    #
    #   * births draw FULL 9-column GB parameters from the previous
    #     posterior; phase_maximize follows GB_RJ_PHASE_MAXIMIZE exactly
    #     like rj_fstat_search (user ruling 2026-09-02 -- the carried
    #     phase degrades with fdot uncertainty at a longer Tobs);
    #   * mixture weights ~ inclusion probability p (mult IGNORED --
    #     PROVISIONAL v1 policy); cross-Tobs v1 = stored widths, f0 windows
    #     re-derived against the NEW run's 1/Tobs (no Fisher rescale);
    #   * a uniform floor over the 9-col prior box keeps logpdf finite at
    #     every alive leaf (BandSorter death factors -- the same reason
    #     UniformFloorMixture is mandatory for the F-stat container);
    #   * search config mirrors rj_fstat_search's cap/flip settings, but the
    #     cap counters stay designated on rj_fstat_search
    #     (leaf_cap_update=False here) and exactly ONE GB move tempers per
    #     iteration (run_swaps=False here).
    #
    # The staged search recipes insert Move("rj_warm_search") IMMEDIATELY
    # BEFORE Move("rj_fstat_search") in the gb_search stage
    # (run_combined_staged.py). Unset knob -> no move object exists and the
    # run is bit-identical. GB_WARM_START_WEIGHT is RESERVED (parsed on
    # GBSettings, unused in v1 -- the future in-move mixture weight vs the
    # F-stat proposal; v1 runs them as separate sequential moves).
    gb_warm_move = None
    _warm_floor_lo = _warm_floor_hi = None
    _warm_path = str(
        getattr(gb_info, "warm_start_components", "")
        or os.environ.get("GB_WARM_START_COMPONENTS", "")
        or ""
    ).strip()
    if _warm_path:
        # AUTO-BUILD (user ruling 2026-09-14: "Check if it is done, if not
        # run it. I would like it to be automatic." -- the 6mo campaign's
        # first launch died on the missing npz). When the refereed npz is
        # absent and GB_WARM_START_SOURCE_STORE names the previous run's
        # FULL FINAL store, the fit -> referee -> apply pipeline runs HERE,
        # at the choke point shared by BOTH twins, before either loads it.
        # MPI-safe: one rank builds under a lock, the others wait. With the
        # npz present this is a pure existence check.
        from .warmstart.build import ensure_warm_start_components

        _warm_path = ensure_warm_start_components(_warm_path, log=logger)
        # Shared by BOTH twins (rj_warm_search here, rj_warm_pe in the PE
        # section below): the 9-col basis guard and the floor box.
        if int(engine_info.ndims["gb"]) != 9:
            raise ValueError(
                "GB_WARM_START_COMPONENTS requires the 9-column sampled GB "
                "basis (distance + fdot_astro_ratio; "
                "GB_USE_ASTROPHYSICAL_F0_MC_PRIOR / GB_USE_CHIRP_MASS on) -- "
                f"this run samples ndim={engine_info.ndims['gb']}."
            )
        # 9-D floor box = the run's prior box: f0 over the INTERIOR band
        # (buffer bands excluded, mirroring the F-stat container's floor),
        # full prior ranges elsewhere.
        _mc_lims = list(getattr(gb_info, "m_chirp_lims", None) or [0.001, 1.0])
        _dist_lims = list(getattr(gb_info, "dist_lims", None) or [0.001, 40.0])
        _ratio_max = float(getattr(gb_info, "fdot_astro_ratio_max", None) or 1.0)
        _f0_lo_i, _f0_hi_i = (1, -2) if len(band_edges) >= 4 else (0, -1)
        _warm_floor_lo = [
            _dist_lims[0], float(band_edges[_f0_lo_i]) * 1e3, _mc_lims[0],
            0.0, -1.0, 0.0, 0.0, -1.0, -_ratio_max,
        ]
        _warm_floor_hi = [
            _dist_lims[-1], float(band_edges[_f0_hi_i]) * 1e3, _mc_lims[-1],
            2.0 * np.pi, 1.0, np.pi, 2.0 * np.pi, 1.0, _ratio_max,
        ]
    # The load options BOTH twins share (the PE twin adds circ_images); the
    # warm-start SELF refit loads its epochs with exactly these.
    _warm_load_kwargs = None if not _warm_path else dict(
        new_tobs=float(general_info.Tobs),
        use_cupy=use_gpu_priors,
        floor_box=(_warm_floor_lo, _warm_floor_hi),
        floor_eps=float(os.environ.get("GB_WARM_START_FLOOR_EPS", "0.05")),
        p_floor=float(os.environ.get("GB_WARM_START_P_FLOOR", "0")),
    )
    if include_search and _warm_path:
        from .warmstart.proposal import WarmStartComponents

        _warm_container = WarmStartComponents.from_npz(
            _warm_path,
            # PER-RANK seed (see _warm_start_seed): the bare run seed here
            # gave every walker the identical candidate sequence (job 751)
            seed=_warm_start_seed(gb_info, _WARM_SEED_TAG_SEARCH),
            **_warm_load_kwargs,
        )
        gb_warm_move = GBSpecialRJPriorMove(
            *gb_move_args,
            rj_proposal_distribution={"gb": _warm_self_wrap(
                _warm_container, gb_info=gb_info, general_info=general_info,
                tag=_WARM_SEED_TAG_SEARCH, name="rj_warm_search",
                load_kwargs=_warm_load_kwargs)},
            name="rj_warm_search",
            use_prior_removal=False,
            # Phase maximization follows GB_RJ_PHASE_MAXIMIZE exactly like
            # rj_fstat_search (user ruling 2026-09-02: "built exactly like
            # fstat_search, with the distribution switched out"); the cycle
            # invariants (run_swaps / leaf_cap_update False) ride along --
            # see warm_search_move_overrides.
            gpus=[],
            **{**gb_move_kwargs,
               "rj_flip_fraction_default": _search_rj_flip_default(),
               **_imr_search, **warm_search_move_overrides()},
        )
        gb_warm_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_warm_move.install_walker_fanout(curr)
        gb_search_moves = list(gb_search_moves) + [gb_warm_move]
        logger.info(
            "build_gb_moves: rj_warm_search armed from %s (%d components; "
            "weights ~ p, floor_eps=%s).",
            _warm_path, _warm_container.n_components,
            os.environ.get("GB_WARM_START_FLOOR_EPS", "0.05"),
        )
        # TWO-PASS rj_replace (user ruling 2026-09-24: "one internal
        # iteration with the refit/warmstart and one internal iteration of
        # the fstat ... inside one def propose()"). The SAME container object
        # is shared by reference -- one WarmStartComponents in the process,
        # holding one device-resident mixture -- so the replace pass and
        # rj_warm_search's births can never draw from different laws.
        # GB_REPLACE_WARM_PASS=0 leaves replace single-pass.
        if gb_replace_move is not None and os.environ.get(
                "GB_REPLACE_WARM_PASS", "1").strip() not in ("0", "false",
                                                             "False", "off"):
            gb_replace_move.replace_warm_distribution = {
                "gb": _warm_container}
            logger.info(
                "build_gb_moves: rj_replace is TWO-PASS -- pass 1 draws "
                "replacement candidates from the warm-start mixture, pass 2 "
                "from the F-stat grid, both inside one propose(). Watch "
                "[GB_REPLACE] for the per-pass lnL drift and [GB_ACCEPT "
                "replace-split] for the per-pass acceptance."
            )

    #* ============================================= PARAMETER ESTIMATION MOVES =============================================
    # PE births draw the EXTRINSICS (distance/amplitude, phi0, cos-iota,
    # psi) from the PRIOR distributions (the rj birth container's extrinsic
    # slots ARE the stock prior uniforms; see make_gb_rj_birth_container)
    # with intrinsics still from the fstat grids when a custom container is
    # set: no recentering, no pinning, no phase-max (USER ruling,
    # 2026-08-01). In PE mode phase_maximize is therefore forced off on the
    # birth instance (under GB_MODE=search the seeded GB_RJ_PHASE_MAXIMIZE
    # keeps the search behavior unchanged).
    # RJ flip fraction mode default: search thins to _SEARCH_RJ_FLIP_DEFAULT
    # (0.2), PE to _PE_RJ_FLIP_DEFAULT (0.2) -- constants at module level.
    # These two moves ARE the GB_MODE=search birth path when that mode is
    # on, so their default follows the mode. The search-NAMED moves
    # (rj_fstat_mcmc_search / rj_refit_search / rj_replace /
    # rj_prior_removal / rj_warm_search) are passed _SEARCH_RJ_FLIP_DEFAULT
    # EXPLICITLY at their own construction sites above. They used to fall
    # through to the implicit 1.0 and were only reaching 0.2 because every
    # submit script exported the global env override; with that export
    # removed (2026-08-28) this wiring is what holds them at 0.2.
    # {BRANCH}_RJ_FLIP_FRACTION / an explicit kwarg still override.
    _rj_flip_default = (
        _search_rj_flip_default() if _gb_mode_search else _pe_rj_flip_default()
    )
    # Per-class repeat defaults for the pe-NAMED moves follow the mode the
    # same way the flip fraction does (search campaigns that run through
    # the pe-named stage get 200/25; true PE gets num_repeat_proposals).
    # PE-cycle moves ALWAYS take the PE floor (5 / env), even inside a
    # search-mode recipe where they inherit the search repeat defaults.
    _imr_defaults = {
        **(_imr_search if _gb_mode_search else _imr_pe),
        "opt_snr_rej_samp_limit": _snr_pe,
    }

    # GB_PE_MOVES_STRICT=1 (2026-08-12 user ruling, for STAGED recipes that
    # carry BOTH search-named and pe-named GB stages in ONE process, e.g.
    # run_combined_staged.py): the pe-NAMED instances (rj_prior /
    # rj_fstat_mcmc / rj_refit) are configured strictly for PE -- no leaf
    # caps, no birth phase-max, the PE flip fraction -- regardless of GB_MODE,
    # so GB_MODE=search arms ONLY the search-named stage's moves. Default
    # OFF because the single-stage campaigns (gb_no_fg GB_MODE=search) run
    # the SEARCH through the pe-named stage and rely on the mode-following
    # behavior below.
    # PE tempering cadence (user ruling 2026-08-14): the PE stack carries
    # SEVERAL swap-enabled GB moves (rj_fstat_pe / rj_prior_pe /
    # rj_fstat_mcmc / rj_refit), which historically meant multiple band
    # swap stages per iteration. The cadence makes tempering a per-branch
    # budget: at most once per GB_TEMPER_EVERY_PROPOSES (default 3) TOTAL
    # GBSpecial* propose() calls, whichever swap-enabled move crosses the
    # budget first. Search-named moves keep cadence 1 (temper each
    # iteration, inside rj_prior_removal per GB_TEMPER_ON_REMOVAL).
    _pe_temper_every = int(os.environ.get("GB_TEMPER_EVERY_PROPOSES", "3"))
    _pe_strict = os.environ.get("GB_PE_MOVES_STRICT", "0") == "1"
    _pe_cap_off = (
        {"leaf_cap_start": None, "leaf_cap_update": False} if _pe_strict else {}
    )
    if _pe_strict:
        _rj_flip_default = _PE_RJ_FLIP_DEFAULT
        _imr_defaults = dict(_imr_pe)


    # PE extrinsic DRAW (user design ruling 2026-08-25): in the strict-PE
    # flavor the pe-named RJ moves draw phi0/cos_iota/psi from genuine
    # maximizer-centered distributions and charge the real forward/reverse
    # densities in the RJ factors (GBSettings.pe_extrinsic_draw /
    # GB_PE_EXTRINSIC_DRAW, default ON; =0 restores the pin + uniform-wash
    # PE behavior bit-identically). SEARCH stages never see the flag: the
    # search-named moves are not seeded, and rj_fstat_pe under a
    # GB_MODE=search campaign (which runs the search THROUGH the pe-named
    # stage) is gated to the exact same strict-PE condition as its
    # phase_maximize, so search campaigns keep the pin convention.
    _pe_extr_draw = bool(getattr(gb_info, "pe_extrinsic_draw", True))

    def _pe_fit_kwargs(cfg):
        # the PE F-stat grid moves' fit kwargs from pe_fstat_move_config: the
        # cadence (passed as before, fit-in-move or not) and, on the grid
        # class only, the move's own peak floor
        out = {**_fit_kwargs, "fstat_refit_every": cfg["fstat_refit_every"]}
        if _fit_in_move:
            out["fstat_peak_min_snr"] = cfg["fstat_peak_min_snr"]
        return out

    # THE GB-FREE GALAXY GRID (ruling 2026-10-12): name, fit dir and mode
    # (fstat_search_residual / fstat_pe_residual both False -> root "shared",
    # GB-free window open, fingerprint unchanged) are EXACTLY today's, so a
    # resumed store picks up the PE epoch it already fitted. Fit once
    # (GB_FSTAT_REFIT_EVERY_PE_GB_FREE=0) at its own floor
    # (GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE=6.5, applied to a loaded epoch as a
    # cut on its cached peaks -- never a refit).
    gb_pe_prior_move = _RJBirth(
        *gb_move_args,
        rj_proposal_distribution=(None if _fit_in_move else _rj_birth_prop),
        **({"is_rj_prop": True} if _fit_in_move else {}),
        **_pe_fit_kwargs(_pe_fstat_gb_free),
        name="rj_fstat_pe",
        use_prior_removal=False,  # gb_info["pe_info"]["use_prior_removal"],
        # NEVER in full_pe unless GB_PE_PHASE_MAXIMIZE=1 asks for it by name
        # -- see pe_phase_maximize_on() for why this no longer just follows
        # GB_MODE, and why GB_PE_MOVES_STRICT was never the right off switch.
        phase_maximize=(
            _rj_phase_max
            if pe_phase_maximize_on(_gb_mode_search, _pe_strict)
            else False
        ),
        pe_extrinsic_draw=(
            _pe_extr_draw and not (_gb_mode_search and not _pe_strict)
        ),
        run_swaps=True,
        temper_every_proposes=_pe_temper_every,
        gpus=[],
        **{**gb_move_kwargs, "rj_flip_fraction_default": _rj_flip_default,
           **_imr_defaults, **_pe_cap_off}
    )
    # PE stays on the GB-FREE grid (``fstat_search_residual`` left False):
    # see the TODO beside rj_fstat_search. Spelled out here because the
    # default is silent and this is the move the decision is about. Since
    # 2026-10-12 the LIVE residual has its own PE move, rj_fstat_pe_residual
    # (below), and this one is the galaxy grid, fitted once.
    gb_pe_prior_move.accepted = np.zeros((ntemps, nwalkers_local))
    gb_pe_prior_move.install_walker_fanout(curr)

    # F-STAT DISTANCE-BIRTH ON THE PE SIDE (USER RULING 2026-08-28:
    # *"rj_fstat_pe get the same stamp? yes mirror them. That would be
    # better than drawing from the full prior widths."*). The ctor default
    # follows rj_amp_maximize -> phase_maximize, which is False on every
    # pe-named move, so PE births used to draw slot 0 and the extrinsic
    # angles at full prior widths while the epoch center table -- already
    # built for this same fit root, already read by rj_replace_pe -- sat
    # unused. Stamped, a PE birth gets the epoch-table centers, the
    # truncated + normalized lognormal slot 0, and the pe_extrinsic_draw
    # angles, each priced on BOTH the birth and the death side (exact
    # detailed balance; see _run_rj_step's factor assembly and
    # tests/test_gb_pe_fstat_dist_birth.py).
    #
    # Scoped to the PE FLAVOR only -- the same guard the PE replace build
    # uses -- so a GB_MODE=search campaign that runs its search THROUGH the
    # pe-named moves keeps resolving the ctor default bit-identically.
    # NOT stamped, deliberately: rj_prior_pe (the pure-PRIOR complement --
    # its full-width births are the channel the F-stat grid cannot reach),
    # rj_fstat_mcmc (its own serial-MCMC proposal), rj_refit (births and
    # densities come from the GMM refit file) and rj_prior_removal (prior
    # reverse-density convention, per its docstring).
    if not _gb_mode_search or _pe_strict:
        gb_pe_prior_move.rj_fstat_dist_birth = _fstat_dist_birth_stamp()

    #* =============== PE LIVE-RESIDUAL F-STAT (rj_fstat_pe_residual) ===============
    # USER RULING 2026-10-12: "The second fstat move will fit the gb-inclusive
    # residual every 150 iterations [...] The residual fstat will take care of
    # low snr adjustments" -- "this should not be specific to each walker, just
    # specific to the one walker that was randomly drawn". rj_fstat_pe's twin in
    # every proposal setting (same class, kwargs, PE tempering budget, flavor
    # stamps), differing ONLY in what its grid is fitted to:
    #   * fstat_pe_residual: the LIVE residual (GBs subtracted, the GB-free
    #     window closed), ONE random cold walker per epoch on the PE rule
    #     (GB_FSTAT_PE_REF; NOT the search moves' min-lnL walker, NOT a fit per
    #     walker), epochs under their OWN root shared_pe_residual;
    #   * cadence GB_FSTAT_REFIT_EVERY_PE (150 elapsed iterations; the decision
    #     runs in its own propose, so the refit is its first draw at or after);
    #   * the stage's declared peak floor (no floor of its own).
    # Built unconditionally (a cheap object until its first propose fits); the
    # recipe's pe_move_names filter and GB_PE_RJ_FSTAT_RESIDUAL_SPLIT (the
    # driver composes it only with a positive share) decide whether it runs.
    # Never a cap updater (rj_fstat_pe keeps that role, as before).
    gb_pe_residual_move = _RJBirth(
        *gb_move_args,
        rj_proposal_distribution=(None if _fit_in_move else _rj_birth_prop),
        **({"is_rj_prop": True} if _fit_in_move else {}),
        **_pe_fit_kwargs(_pe_fstat_residual),
        name=PE_FSTAT_RESIDUAL_MOVE,
        use_prior_removal=False,
        phase_maximize=(
            _rj_phase_max
            if pe_phase_maximize_on(_gb_mode_search, _pe_strict)
            else False
        ),
        pe_extrinsic_draw=(
            _pe_extr_draw and not (_gb_mode_search and not _pe_strict)
        ),
        run_swaps=True,
        temper_every_proposes=_pe_temper_every,
        gpus=[],
        **{**gb_move_kwargs, "leaf_cap_update": False,
           "rj_flip_fraction_default": _rj_flip_default,
           **_imr_defaults, **_pe_cap_off}
    )
    gb_pe_residual_move.fstat_pe_residual = _pe_fstat_residual["fstat_pe_residual"]
    gb_pe_residual_move.accepted = np.zeros((ntemps, nwalkers_local))
    gb_pe_residual_move.install_walker_fanout(curr)
    if not _gb_mode_search or _pe_strict:
        gb_pe_residual_move.rj_fstat_dist_birth = _fstat_dist_birth_stamp()

    #* ===================== WARM-START PE TWIN (rj_warm_pe) =====================
    # USER RULING 2026-09-07: "rj_warm_search should also have a pe twin,
    # rj_warm_pe. It should mirror the fstat search <-> pe differences."
    # Mirrored deltas (see warm_pe_move_overrides): phase_maximize
    # unconditionally False (no maximization in PE), run_swaps=True under
    # the PE tempering budget, never a cap updater; flip fraction and
    # in-model repeats follow rj_fstat_pe's mode-following defaults
    # (_rj_flip_default / _imr_defaults / _pe_cap_off). Deliberately NOT
    # stamped with rj_fstat_dist_birth and pe_extrinsic_draw=False -- the
    # warm container's births carry their OWN full-9-column draws with
    # exactly charged densities, the same rationale the stamp comment
    # gives for rj_refit ("births and densities come from the GMM refit
    # file"). EXACTNESS: the PE container is built with circ_images > 0
    # (GB_WARM_START_CIRC_IMAGES, default 3) so the charged logpdf is the
    # wrapped-normal density of the actual draws -- the minimal-image
    # approximation the SEARCH twin uses is not detailed-balance-exact for
    # wide circular components (the container's RuntimeWarning regime).
    gb_warm_pe_move = None
    if _warm_path:
        from .warmstart.proposal import WarmStartComponents

        _warm_pe_load_kwargs = dict(
            _warm_load_kwargs,
            circ_images=int(os.environ.get("GB_WARM_START_CIRC_IMAGES", "3")),
        )
        _warm_pe_container = WarmStartComponents.from_npz(
            _warm_path,
            # PER-RANK seed, its own domain tag (see _warm_start_seed)
            seed=_warm_start_seed(gb_info, _WARM_SEED_TAG_PE),
            **_warm_pe_load_kwargs,
        )
        gb_warm_pe_move = GBSpecialRJPriorMove(
            *gb_move_args,
            rj_proposal_distribution={"gb": _warm_self_wrap(
                _warm_pe_container, gb_info=gb_info,
                general_info=general_info, tag=_WARM_SEED_TAG_PE,
                name="rj_warm_pe", load_kwargs=_warm_pe_load_kwargs)},
            name="rj_warm_pe",
            use_prior_removal=False,
            pe_extrinsic_draw=False,
            gpus=[],
            **{**gb_move_kwargs,
               "rj_flip_fraction_default": _rj_flip_default,
               **_imr_defaults, **_pe_cap_off,
               **warm_pe_move_overrides()},
        )
        gb_warm_pe_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_warm_pe_move.install_walker_fanout(curr)
        logger.info(
            "build_gb_moves: rj_warm_pe armed from %s (%d components; "
            "circ_images=%s, floor_eps=%s).",
            _warm_path, _warm_pe_container.n_components,
            os.environ.get("GB_WARM_START_CIRC_IMAGES", "3"),
            os.environ.get("GB_WARM_START_FLOOR_EPS", "0.05"),
        )

    #* ============ GB PARALLEL RESIDUAL SEARCH (rj_para_search_pe) ============
    # USER RULING 2026-10-16 (docs/gb-para-search.md): one GB per sub-band per
    # cold walker, a PT MCMC per (walker, band) on the walker's OWN residual,
    # the finds above opt SNR 5 turned into GMMs and combined over the walkers
    # into one birth mixture. Built only with GB_PARA_SEARCH=1
    # (gb.para_search); stage-composed by setup_gb_moves into the PE stage(s)
    # carrying rj_prior_pe; drawn in the PE draw-one at GB_PE_RJ_PARA_FRACTION.
    # The rj_warm_pe pattern: use_prior_removal=False, pe_extrinsic_draw=False
    # (the mixture draws all nine columns and prices them), the PE overrides;
    # its container is the newest complete epoch under
    # <run_dir>/warmstart/parasearch, loaded like the warm PE twin's (floor box
    # = the run's prior box, GB_WARM_START_FLOOR_EPS, circ_images =
    # GB_WARM_START_CIRC_IMAGES for exact wrapped-normal densities).
    gb_para_move = None
    if bool(getattr(gb_info, "para_search", False)):
        from .moves.gbspecialstretch import GBParaSearchRJMove
        from .warmstart.para_search import (
            PARA_SEARCH_MOVE,
            ParaSearchConfig,
            para_search_epoch_root,
        )

        if int(engine_info.ndims["gb"]) != 9:
            raise ValueError(
                "GB_PARA_SEARCH=1 requires the 9-column sampled GB basis "
                "(distance + fdot_astro_ratio) -- this run samples "
                f"ndim={engine_info.ndims['gb']}.")
        _ps_mc = list(getattr(gb_info, "m_chirp_lims", None) or [0.001, 1.0])
        _ps_dist = list(getattr(gb_info, "dist_lims", None) or [0.001, 40.0])
        _ps_ratio = float(getattr(gb_info, "fdot_astro_ratio_max", None) or 1.0)
        _ps_lo_i, _ps_hi_i = (1, -2) if len(band_edges) >= 4 else (0, -1)
        _ps_floor = (
            [_ps_dist[0], float(band_edges[_ps_lo_i]) * 1e3, _ps_mc[0],
             0.0, -1.0, 0.0, 0.0, -1.0, -_ps_ratio],
            [_ps_dist[-1], float(band_edges[_ps_hi_i]) * 1e3, _ps_mc[-1],
             2.0 * np.pi, 1.0, np.pi, 2.0 * np.pi, 1.0, _ps_ratio],
        )
        _ps_load = dict(
            new_tobs=float(general_info.Tobs), use_cupy=use_gpu_priors,
            floor_box=_ps_floor,
            floor_eps=float(os.environ.get("GB_WARM_START_FLOOR_EPS", "0.05")),
            p_floor=0.0,
            circ_images=int(os.environ.get("GB_WARM_START_CIRC_IMAGES", "3")),
        )
        _ps_cfg = ParaSearchConfig.from_settings(gb_info)
        _ps_root = para_search_epoch_root(general_info.main_file_path)
        # its own domain tag beside the warm twins' 0x3A12 / 0x3A13: the
        # container seeds are warm_seed(build_seed, tag ^ (epoch + 1)), per
        # rank and epoch (the self refit's rule)
        _PARA_SEED_TAG = 0x3A14
        gb_para_move = GBParaSearchRJMove(
            *gb_move_args,
            rj_proposal_distribution=None,
            is_rj_prop=True,
            name=PARA_SEARCH_MOVE,
            use_prior_removal=False,
            pe_extrinsic_draw=False,
            para_search_cfg=_ps_cfg,
            para_search_root=_ps_root,
            para_search_load_kwargs=_ps_load,
            para_search_seed=getattr(gb_info, "build_seed", None),
            para_search_tag=_PARA_SEED_TAG,
            gpus=[],
            **{**gb_move_kwargs,
               "rj_flip_fraction_default": _rj_flip_default,
               **_imr_defaults, **_pe_cap_off,
               **warm_pe_move_overrides()},
        )
        gb_para_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_para_move.install_walker_fanout(curr)
        logger.info(
            "build_gb_moves: %s armed (epochs under %s; every %d PE iterations; "
            "%d x %d chains per sub-band, SNR >= %g; draw share "
            "GB_PE_RJ_PARA_FRACTION=%s).", PARA_SEARCH_MOVE, _ps_root,
            int(_ps_cfg.every_pe), int(_ps_cfg.ntemps), int(_ps_cfg.nwalkers),
            float(_ps_cfg.snr_min), os.environ.get("GB_PE_RJ_PARA_FRACTION", "0"))

    #* ------------------------- PE REPLACE (rj_replace_pe) -------------------------
    # USER DIRECTIVE 2026-08-28: "we need a PE replace that also uses the
    # same machinery as fstat pe". Same fixed-dimension REPLACEMENT move as
    # the search install above (one alive source swapped for a fresh
    # full-parameter F-stat candidate, EXACT add-deltas both sides), built
    # from the SAME container the PE birth move carries -- under
    # fstat_fit_in_move it is a grid move on the shared fit root, so
    # _FSTAT_GRID_REGISTRY / _FSTAT_CTR_TABLE_REGISTRY hand it the
    # identical birth container and epoch center table with no second fit;
    # otherwise it takes the same ``_rj_birth_prop`` (custom container or
    # the device-correct global prior) that rj_fstat_pe does, which is the
    # container ``band_sorter.rj_prop`` resolves to for both moves.
    #
    # PE FLAVOR (the deltas vs the search install): PE repeat/SNR defaults
    # + pe-strict cap config (``_imr_defaults`` / ``_pe_cap_off``), the PE
    # flip-fraction default, and the shared PE tempering cadence
    # (run_swaps=True + temper_every_proposes) instead of the search
    # all-moves swap rule.
    #
    # EXACT DETAILED BALANCE: this install stamps NEITHER
    # ``replace_search_stage`` NOR a name containing "search", so
    # GBSpecialBase._replace_fstat_max resolves False and slot 0 is
    # genuinely drawn + priced on both sides (the search-only
    # maximize-then-pretend exception, which would maximization-bias the
    # amplitude posterior, never arms in PE).
    #
    # NO MAXIMIZING IN PE (user general rule 2026-08-28): the move is
    # seeded with ``pe_extrinsic_draw`` exactly like the pe-named births,
    # so _run_replace_step DRAWS phi0/cos_iota/psi from the
    # maximizer-centered proposal and charges the real forward/reverse
    # densities (the shared _pe_or_pin_extrinsics / _pe_death_extr_corr
    # helpers) instead of PINNING them at the JKS maximizers; the
    # rotation-on-accept phase-max scoring likewise resolves OFF for a
    # replace_pe_stage-stamped move. SEARCH keeps both maximizations.
    gb_pe_replace_move = None
    # Build for a genuine PE stack: a pe-mode recipe, or a STAGED recipe
    # running its pe-named moves strict-PE inside a GB_MODE=search process
    # (GB_PE_MOVES_STRICT=1). A single-stage search campaign that runs its
    # SEARCH through the pe-named moves keeps ``rj_replace`` and nothing
    # else. The recipe's ``pe_move_names`` filter is the final say either
    # way -- an unrequested move never installs.
    if ((not _gb_mode_search or _pe_strict)
            and getattr(gb_info, "pe_rj_replace", False)):
        gb_pe_replace_move = _RJBirth(
            *gb_move_args,
            rj_proposal_distribution=(None if _fit_in_move else _rj_birth_prop),
            **({"is_rj_prop": True} if _fit_in_move else {}),
            # rj_fstat_pe's ROOT, so rj_fstat_pe's cadence and floor (fit once,
            # SNR 6.5): on the residual move's 150 it would re-open the galaxy
            # epochs (2026-10-12)
            **_pe_fit_kwargs(pe_fstat_move_config("rj_replace_pe")),
            name="rj_replace_pe",
            rj_replace=True,
            # NEVER phase-maximize the RJ birth machinery here (the
            # root-caused rj_replace ll-drift flaw); the separate
            # GB_REPLACE_PHASE_MAX scoring knob resolves off from the PE
            # stage stamp below.
            phase_maximize=False,
            # The pe_extrinsic_draw treatment the PE births get: angles
            # drawn from the maximizer-centered proposal and priced on
            # both sides, never pinned at the maximizer.
            pe_extrinsic_draw=_pe_extr_draw,
            run_swaps=True,
            temper_every_proposes=_pe_temper_every,
            gpus=[],
            **{**gb_move_kwargs, "leaf_cap_update": False,
               "rj_flip_fraction_default": _rj_flip_default,
               **_imr_defaults, **_pe_cap_off},
        )
        # The center pin IS this move's proposal, so it must not depend on
        # the phase-max chain the ctor default follows (identical reasoning
        # to the search install). GB_RJ_FSTAT_DIST_BIRTH still wins.
        gb_pe_replace_move.rj_fstat_dist_birth = _fstat_dist_birth_stamp()
        # PE-stage stamp: "same machinery as fstat pe" == the EPOCH CENTER
        # TABLE for the extrinsic centers (what _rj_birth_perrow hands the
        # pe-named fstat moves). The move's plain name carries no stage
        # info, so the stamp is what makes _replace_ctr_mode resolve
        # "table"; search installs stay "perrow" bit-identically, and an
        # explicit GB_REPLACE_CTR_MODE overrides either way.
        gb_pe_replace_move.replace_pe_stage = True
        gb_pe_replace_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_pe_replace_move.install_walker_fanout(curr)

    # PURE prior-birth RJ move for the PE stage (2026-08-12 rename/split):
    # births drawn from the GLOBAL PRIOR (never the fstat grids), deaths
    # judged the same way -- the complement to rj_fstat_pe in a PE cycle.
    # Always strict-PE flavored: no phase-max, no caps beyond enforcement.
    gb_pe_prior_birth_move = GBSpecialRJPriorMove(
        *gb_move_args,
        rj_proposal_distribution=gpu_priors,
        name="rj_prior_pe",
        use_prior_removal=False,
        phase_maximize=False,
        # Always strict-PE flavored, so the extrinsic-draw knob applies
        # directly (only bites when its F-stat distance-birth path is
        # active, i.e. GB_RJ_FSTAT_DIST_BIRTH=1 for this prior-birth move).
        pe_extrinsic_draw=_pe_extr_draw,
        run_swaps=True,
        temper_every_proposes=_pe_temper_every,
        gpus=[],
        **{**gb_move_kwargs, "leaf_cap_update": False,
           "rj_flip_fraction_default": _rj_flip_default,
           **_imr_defaults, **_pe_cap_off}
    )
    gb_pe_prior_birth_move.accepted = np.zeros((ntemps, nwalkers_local))
    gb_pe_prior_birth_move.install_walker_fanout(curr)

    gb_pe_fstat_mcmc_move = GBSpecialRJSerialSearchMCMC(
        *gb_move_args,
        rj_proposal_distribution=None,
        run_swaps=True,
        temper_every_proposes=_pe_temper_every,
        name="rj_fstat_mcmc",
        phase_maximize=False,
        gpus=[],
        **{**gb_move_kwargs, "leaf_cap_update": False,
           "rj_flip_fraction_default": _rj_flip_default,
           **_imr_defaults, **_pe_cap_off}
    )
    gb_pe_fstat_mcmc_move.accepted = np.zeros((ntemps, nwalkers_local))
    gb_pe_fstat_mcmc_move.install_walker_fanout(curr)

    # Prior + fstat moves always build; the refit move is inserted only when
    # its GMM-refit file is available (see ``_refit_available`` above).
    # The search-cycle companions (GB_MODE=search + knobs) ride in this
    # list too: the current search campaign runs through the pe-NAMED stage
    # (single ``rj_prior`` move under GB_MODE=search), and the
    # ``pe_move_names`` filter below keeps exactly the recipe-requested
    # subset either way.
    # ``rj_replace_pe`` sits directly AFTER the pe-named fstat birth move,
    # the same relative position ``rj_replace`` takes in the search cycle
    # (fstat-birth -> fstat-REPLACE): a freshly born leaf is the natural
    # candidate for a replacement swap in the same iteration. The stage's
    # recipe move list is what actually orders the GFCombineMove cycle;
    # this order is the registration order the filter preserves.
    # F-STAT NODE GRIDS ON THE DEVICE (user ruling 2026-10-12, step 1 of the
    # grid memory plan in StackedFStatProposal4D): after construction the
    # node stack is read only by the trilinear in-cell mode, which only the
    # replace moves use. With no replace move composed, every F-stat grid
    # container built in this process (the in-move fit, every epoch load on
    # every rank) drops its node grids: 24.2 GB of the 9mo PE galaxy grid's
    # ~43 GB per card, with bit-identical draws and densities.
    # GB_FSTAT_KEEP_NODES=1 / 0 overrides either way.
    from ..sampling.fstat_proposal import set_stacked_keep_nodes

    _keep_nodes = (gb_replace_move is not None) or (gb_pe_replace_move is not None)
    set_stacked_keep_nodes(_keep_nodes)
    logger.info(
        "[FSTAT_GRID] node grids %s on the device after construction "
        "(replace moves composed: search=%s pe=%s; env GB_FSTAT_KEEP_NODES=%r).",
        "KEPT" if _keep_nodes else "DROPPED",
        gb_replace_move is not None, gb_pe_replace_move is not None,
        os.environ.get("GB_FSTAT_KEEP_NODES", ""))

    gb_pe_moves = [gb_pe_prior_move, gb_pe_residual_move]
    if gb_warm_pe_move is not None:
        gb_pe_moves.append(gb_warm_pe_move)
    if gb_pe_replace_move is not None:
        gb_pe_moves.append(gb_pe_replace_move)
    gb_pe_moves += [gb_pe_prior_birth_move, gb_pe_fstat_mcmc_move]
    if gb_para_move is not None:
        gb_pe_moves.append(gb_para_move)
    if gb_replace_move is not None:
        gb_pe_moves.append(gb_replace_move)
    gb_pe_moves += gb_in_model_moves
    if gb_prior_removal_move is not None:
        gb_pe_moves.append(gb_prior_removal_move)
    if _refit_available:
        gb_pe_refit_move = GBSpecialRJRefitMove(
            *gb_move_args,
            rj_proposal_distribution=None,
            run_swaps=True,
            temper_every_proposes=_pe_temper_every,
            name="rj_refit",
            fp=_refit_fp,
            phase_maximize=False,  # gb_info["pe_info"]["rj_phase_maximize"],
            gpus=[],
            # _imr_defaults like every other pe-named RJ move -- see the
            # matching note on rj_refit_search.
            **{**gb_move_kwargs, "leaf_cap_update": False,
               **_imr_defaults, **_pe_cap_off}
        )
        gb_pe_refit_move.accepted = np.zeros((ntemps, nwalkers_local))
        gb_pe_refit_move.install_walker_fanout(curr)
        gb_pe_moves.insert(1, gb_pe_refit_move)  # [prior, refit, fstat]

    # Design knob: keep only the requested PE moves (order-preserving subset).
    # Absorbs the ``[m for m in gb_pe_moves if "prior" in m.name]`` filtering the
    # GB settings files used to do post-hoc.
    if pe_move_names is not None:
        gb_pe_moves = [m for m in gb_pe_moves if m.name in pe_move_names]

    # RIDGE-GIBBS fiber move (2026-08-20, user-ruled): resamples the exact
    # Mc^{5/3}(1+r) = const / A = const likelihood degeneracy of the 9-column
    # basis with ZERO likelihood calls (prior x fiber-measure MH only) --
    # without it the (Mc, r, dist) marginals freeze along the curved ridge at
    # high f (~2e5 accepted info-matrix jumps to traverse; measured). Cold
    # chain only (the engine state; hot rungs live inside the band moves and
    # do not need mixed nuisance marginals). Registered for BOTH cycles; the
    # stage lists decide where it actually runs. GB_RIDGE_GIBBS=0 disables.
    # Eligibility is ``ridge_gibbs_eligible`` (by COLUMN NAME) since
    # 2026-09-16, shared with the vgb twin in ``build_vgb_moves``; for the
    # 9-column GB basis it is the same answer the inline gate gave.
    if ridge_gibbs_eligible(gb_info):
        from ..sampling.ridge_fiber import make_gb_ridge_gibbs_move

        _ridge = make_gb_ridge_gibbs_move(
            priors["gb"], gb_info.transform,
            mc_lims=gb_info.m_chirp_lims,
            dist_lims=gb_info.dist_lims,
            ratio_max=float(gb_info.fdot_astro_ratio_max),
            leaf_fraction=float(
                os.environ.get("GB_RIDGE_GIBBS_LEAF_FRACTION", "1.0")),
        )
        _ridge.name = "gb_ridge_gibbs"
        # A plain eryn move (coords only, zero likelihood calls): under several
        # compute ranks it runs HEAD-ONLY on the full engine state, so its
        # accepted array is engine-wide and it takes no fan-out install (the
        # readiness guard lists it as head-only).
        _ridge.accepted = np.zeros((1, nwalkers))
        gb_search_moves = list(gb_search_moves) + [_ridge]
        gb_pe_moves = list(gb_pe_moves) + [_ridge]
        logger.info("build_gb_moves: gb_ridge_gibbs registered (leaf_fraction "
                    "%s; zero-likelihood fiber move on the Mc-ratio-distance "
                    "degeneracy).", _ridge.leaf_fraction
                    if hasattr(_ridge, "leaf_fraction") else "1.0")

    # The vertical-swap Generator's seed base, on every move this builder
    # returns (see :func:`_stamp_temper_seed_base`).
    _stamp_temper_seed_base(
        list(gb_search_moves) + list(gb_pe_moves), general_info.random_seed
    )

    return gb_search_moves, gb_pe_moves

def build_vgb_moves(
    engine_info,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state: GFState,
    *,
    Tmax: float = 1e6,
) -> typing.List[VGBSpecialStretchMove]:
    """Build the VGB in-model stretch move (fixed-dimensional, no RJ).

    The VGB analog of :func:`build_gb_moves`, reduced to what a
    fixed-dimensional known-source branch needs: seed-template subtraction
    (leaf-aware transform through the per-leaf fill container), band-info
    initialization on ``sub_states["vgb"]``, and ONE
    :class:`~lisatools.globalfit.moves.VGBSpecialStretchMove` named
    ``"vgb_pe"`` (plain same-leaf stretch; no friends, no info-matrix, no
    phase maximization; band-temperature swaps run on this move since no RJ
    move exists to carry them).

    Under the 6-column chirp basis (``VGB_CHIRP_MASS_BASIS=1``) a SECOND
    move is registered: ``"vgb_ridge_gibbs"``, the exact GB twin (user
    ruling 2026-09-16). See the registration block at the end of this
    function; on the 5-column legacy basis the return is the historical
    one-element list.

    Returns the PE move list (``["vgb_pe"]``, plus ``"vgb_ridge_gibbs"``
    when the basis carries the (dist, Mc, fdot_astro_ratio) fiber).
    """
    vgb_info = curr.source_info["vgb"]
    general_info: GeneralSetup = curr.general_info
    nwalkers: int = general_info.nwalkers
    # MOVE/ACA-level sizes (ladder, friend window, accepted array) follow
    # this rank's walker block; the band table on the sub-state stays
    # engine-wide (see initialize_band_information below). Equal single-process.
    nwalkers_local = _local_nwalkers(acs)
    # VGB's OWN ladder size (the engine runs cold-chain only); an explicit
    # betas array wins over the ntemps knob
    ntemps: int = (
        len(vgb_info.betas)
        if getattr(vgb_info, "betas", None) is not None
        else int(getattr(vgb_info, "ntemps", None) or general_info.ntemps)
    )
    data_start_freq_ind = int(acs.start_freq_ind[0])
    gpus: list[int] = general_info.gpus
    domain_settings = general_info.domain_settings

    from gbgpu.gbgpu import GBGPU
    import gbgpu

    _gb_backend = gbgpu.get_backend(general_info.force_backend)
    if gpus is not None:
        _gb_backend.set_cuda_device(gpus[0])
    gb = GBGPU(force_backend=general_info.force_backend, orbits=general_info.gpu_orbits)
    gb.gpus = gpus if gpus is not None else None

    use_gpu_priors = gpus is not None
    gpu_priors_in = deepcopy(priors["vgb"].priors_in)
    for _, item in gpu_priors_in.items():
        item.use_cupy = use_gpu_priors
    _gpu_vgb_prior = ProbDistContainer(gpu_priors_in, use_cupy=use_gpu_priors)
    # Same key-order preservation as the GB branch: a fresh container built
    # from insertion-order ``priors_in`` must inherit any ``reset_key_order``
    # the CPU prior applied, or multi-column tuple priors land on the wrong
    # columns. Guarded no-op when the CPU prior was never reordered.
    _cpu_vgb_key_order = list(getattr(priors["vgb"], "key_order", []))
    if _cpu_vgb_key_order and list(_gpu_vgb_prior.key_order) != _cpu_vgb_key_order:
        _gpu_vgb_prior.reset_key_order(_cpu_vgb_key_order)
    gpu_priors = {"vgb": _gpu_vgb_prior}

    band_edges = vgb_info.band_edges
    band_N_vals = vgb_info.band_N_vals
    assert band_edges is not None and band_N_vals is not None

    tc = vgb_info.transform
    input_basis = list(tc.input_basis)
    nleaves_max_vgb = state.branches["vgb"].shape[-2]

    # VGB coords are seeded by the generic fixed-leaf path in run.py
    # (multiplicative ``x*(1 + VGB_START_FACTOR*randn)`` scatter around the
    # injection — magnitude-robust, so fdot ~1e-16 and amp scatter sensibly
    # without a per-dimension prior-width scale; 0 -> exact truth; the
    # chirp basis' zero-truth fdot_astro_ratio column gets ADDITIVE jitter
    # via additive_start_widths — see seed_injection_coords). The
    # periodic wrap + prior-bounds check happen in the subtraction block
    # below.

    # ---- subtract the seeded VGB templates from the residuals ----
    # Block-gated (Plan 2, Task 5): this rank's ACA holds only walkers
    # [w0, w1) of the FULL ``state``, so every per-walker row below is
    # sliced to that block before use.
    w0, w1 = _local_walker_block(curr, acs)
    inds0 = state.branches["vgb"].inds[0, w0:w1]  # (w1 - w0, nleaves)
    if getattr(vgb_info, "signal_gen", None) is None and inds0.sum() > 0:
        coords_out = state.branches["vgb"].coords[0, w0:w1][inds0]
        for _name, _per in (("phi0", 2 * np.pi), ("psi", np.pi)):
            _i = input_basis.index(_name)
            coords_out[:, _i] = coords_out[:, _i] % _per

        check = priors["vgb"].logpdf(coords_out)
        if np.any(np.isinf(check)):
            raise ValueError(
                "VGB starting coordinates fall outside the priors; check the "
                "VGB prior limits against the catalogue values."
            )

        leaf_inds = np.tile(np.arange(nleaves_max_vgb), (w1 - w0, 1))[inds0]
        coords_in_in = tc.both_transforms(coords_out, leaf_inds=leaf_inds)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1
        walker_vals = np.tile(
            np.arange(w1 - w0), (nleaves_max_vgb, 1)
        ).transpose((1, 0))[inds0]

        _xp = acs.xp
        data_index = _xp.asarray(walker_vals).astype(_xp.int32)
        # goes in as -h (subtract initial template from data residual)
        factors = -_xp.ones_like(data_index, dtype=_xp.float64)
        N_vals = band_N_vals[band_inds]

        logger.debug("Subtracting seeded VGB templates from the residuals")
        if isinstance(domain_settings, FDSettings):
            num_per_gpu_walker = (
                len(acs.gpu_splits[0])
                if (acs.gpus is not None and len(acs.gpus) > 1)
                else None
            )
            gb.generate_global_template(
                coords_in_in,
                data_index,
                acs.linear_data_arr,
                data_length=acs.data_length,
                factors=factors,
                data_splits=acs.gpu_map,
                num_per_gpu=num_per_gpu_walker,
                N=N_vals,
                **{
                    k: v
                    for k, v in vgb_info.waveform_kwargs.items()
                    if k != "N"
                },
            )
        elif isinstance(domain_settings, WDMSettings):
            if vgb_info.gb_wdm_comp is None:
                raise ValueError(
                    "WDM-domain VGB initialization requires "
                    "vgb_info.gb_wdm_comp (a GBWDMComputations); the variant "
                    "setup builds it before calling build_vgb_moves."
                )
            xp = vgb_info.gb_wdm_comp.xp
            # Gather is a copy on multi-shard ACAs — scatter the filled
            # buffer back or the VGB subtraction silently no-ops there.
            _flat_res = acs.gather_linear_data_arr()
            vgb_info.gb_wdm_comp.fill_global_wdm(
                coords_in_in,
                _flat_res,
                data_index=xp.asarray(data_index),
                factors=xp.asarray(factors).astype(xp.float64),
            )
            acs.scatter_linear_data_arr(_flat_res)
        else:
            raise NotImplementedError(
                f"Domain settings {type(domain_settings).__name__} are not "
                "supported for VGB initialization."
            )

    # ---- per-band temperature ladders on the vgb sub-state ----
    band_temps = np.tile(np.asarray(vgb_info.betas), (len(band_edges) - 1, 1))
    # ``initialize_band_information`` returns the rung count ACTUALLY in
    # effect: on a resume the STORED ladder wins (it warns and names both
    # counts), because re-rungging live per-band temps + counters is a
    # migration, not a mid-resume side effect. Everything below that is
    # sized by the ladder must therefore use ``resolved_ntemps``, or the
    # move would carry e.g. a 12-rung TemperatureControl against a 1-rung
    # state.
    # leaf_caps=False (user ruling 2026-08-22): leaf caps gate RJ births and
    # VGB has no RJ surface -- the branch carries NO cap-cell state, checks
    # no cap grid on resume, and drops any stored cap keys (so a band-grid
    # migration never needs a cap-grid companion for vgb).
    resolved_ntemps = state.sub_states["vgb"].initialize_band_information(
        nwalkers, ntemps, band_edges, band_temps, branch_name="vgb",
        leaf_caps=False,
    )
    if resolved_ntemps == ntemps:
        # unchanged behaviour: the configured flat ladder is (re)written
        # onto every band.
        state.sub_states["vgb"].band_info["band_temps"][:] = band_temps
    else:
        # Adopt the stored ladder end to end: the branch's ``betas`` is
        # what run.py's ``_branch_ntemps`` and any later consumer read, so
        # reconcile it too rather than leaving a stale 12-rung config on
        # ``curr.source_info["vgb"]``. The stored per-band temps are left
        # exactly as loaded; band 0's ladder is the representative one.
        configured_ntemps, ntemps = ntemps, resolved_ntemps
        band_temps = np.asarray(
            state.sub_states["vgb"].band_info["band_temps"], dtype=float
        )
        vgb_info.betas = band_temps[0].copy()
        logger.warning(
            "VGB branch resumed at the STORED %d-rung ladder %s (the "
            "configured %d-rung ladder was discarded); the vgb move's "
            "TemperatureControl, betas and accepted array follow the store.",
            ntemps, np.array2string(vgb_info.betas, precision=4),
            configured_ntemps,
        )

    effective_ndim = engine_info.ndims["vgb"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers_local, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    vgb_move = VGBSpecialStretchMove(
        gb,
        priors,
        data_start_freq_ind,
        acs.end_shape[0],
        acs,
        band_edges,
        band_N_vals,
        gpu_priors,
        branch_name="vgb",
        name="vgb_pe",
        waveform_kwargs=vgb_info.waveform_kwargs,
        parameter_transforms=tc,
        # BUGFIX (2026-08-15): without this kwarg the VGB move inherited
        # the GB SEARCH default opt_snr_rej_samp_limit=5.0 -- a birth
        # prior for unresolvable-source RJ that is wrong by construction
        # for KNOWN fixed-dimension sources. It froze the 36/55 VGBs with
        # optimal SNR < 5 at their init coords (every proposal force-
        # rejected) and truncated the low-amplitude side of the movers.
        # 0.0 disables; VGB_OPT_SNR_LIMIT overrides if ever needed.
        opt_snr_rej_samp_limit=float(
            os.environ.get("VGB_OPT_SNR_LIMIT", "0.0")),
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info.random_seed,
        force_backend=general_info.force_backend,
        nfriends=nwalkers_local,
        temperature_control=temperature_control,
        num_repeat_proposals=vgb_info.num_repeat_proposals,
        gb_wdm_comp=vgb_info.gb_wdm_comp,
        gb_fd_comp=getattr(vgb_info, "gb_fd_comp", None),
        orbits=getattr(vgb_info, "orbits", None),
        tdi_config=getattr(vgb_info, "tdi_config", None),
        t_ref=float(getattr(vgb_info, "t0", 0.0) or 0.0),
        run_swaps=True,
        gpus=[],
        # Same sig-het reference-policy knobs as the GB move (fixed
        # reference by default; see build_gb_moves) so the two branches
        # audit identically.
        sighet_refresh_every=int(os.environ.get("GB_SIGHET_REFRESH_EVERY", "0")),
        sighet_refresh_dphase=float(os.environ.get("GB_SIGHET_REFRESH_DPHASE", "0.5")),
        sighet_refresh_min_beta=float(
            os.environ.get("GB_SIGHET_REFRESH_MIN_BETA", "0.1")),
        # Trust region: reject in-model candidates beyond these gates from
        # the block's heterodyne anchor (physical |dlnA| e-folds / carrier
        # phase rad); 0 disables. Inert on chunked-het / FD.
        sighet_trust_dlna=float(os.environ.get("GB_SIGHET_TRUST_DLNA", "1.5")),
        sighet_trust_dphase=float(
            os.environ.get("GB_SIGHET_TRUST_DPHASE", "0.5")),
        # SNR scaling of the amplitude gate: per-source dlnA_max =
        # clip(C/snr_ref, dlna_min, GB_SIGHET_TRUST_DLNA); C=0 -> uniform.
        sighet_trust_snr_c=float(
            os.environ.get("GB_SIGHET_TRUST_SNR_C", "30")),
        sighet_trust_dlna_min=float(
            os.environ.get("GB_SIGHET_TRUST_DLNA_MIN", "0.3")),
        # SNR scaling of the PHASE gate, same shape as the amplitude one.
        # C_phase = 3.456*sqrt(2*T_gate) places the gate at a constant TRUE
        # lnL displacement T (49 -> T=100, 155 -> T=1000) instead of at a
        # constant parameter offset; C=0 (default) keeps the uniform gate.
        # Calibrate with GB_SIGHET_TIER_SCAN before arming.
        sighet_trust_phase_c=float(
            os.environ.get("GB_SIGHET_TRUST_PHASE_C", "0")),
        sighet_trust_dphase_max=float(
            os.environ.get("GB_SIGHET_TRUST_DPHASE_MAX", "20")),
        sighet_anchor_check=os.environ.get(
            "GB_SIGHET_ANCHOR_CHECK", "0") == "1",
        sighet_drift_check=os.environ.get("GB_SIGHET_DRIFT_CHECK", "0") == "1",
        **{
            k: v
            for k, v in (vgb_info.group_proposal_kwargs or {}).items()
            if k != "num_repeat_proposals"
        },
    )
    vgb_move.accepted = np.zeros((ntemps, nwalkers_local))
    vgb_move.install_walker_fanout(curr)
    vgb_moves = [vgb_move]

    # RIDGE-GIBBS fiber move, the GB twin (user ruling 2026-09-16: "really
    # mirror the GBs as much as possible ... VGBs get the ridge-gibbs fiber
    # move too"). With f0 and sky per-leaf fills, the (Mc, r, dist)
    # degeneracy -- Mc^{5/3}(1 + r) = const at fixed fdot, Mc^{5/3}/d =
    # const at fixed A -- is the ENTIRE remaining flat direction of the
    # branch, and the in-model observable+eigen draw freezes Mc along it by
    # construction (fdot-weight 0). This move resamples it in closed form
    # with ZERO likelihood calls. Standalone HM Cnc validation (2026-09-16):
    # obs+eigen fw=0 + ridge-gibbs was the best arm, cold acceptance 0.593.
    #
    # ``McRatioDistFiber`` is used VERBATIM -- it resolves dist / Mc /
    # fdot_astro_ratio by NAME and never reads f0 -- so the ONLY vgb-side
    # condition is that the basis carries those three columns, which is
    # exactly ``ridge_gibbs_eligible``: true under VGB_CHIRP_MASS_BASIS=1
    # (6 columns, Mc sampled), false on the 5-column legacy distance basis
    # where Mc is a per-leaf fill. Every script still on the 5-column basis
    # therefore builds the identical one-move stack it always did.
    #
    # Same prior-box convention as the branch's own priors (vgb.py): Mc ~
    # U(m_chirp_lims | (0.001, 1.0)), dist ~ U(dist_lims), r ~ U[-rmax,
    # rmax]. Cold chain only, like the GB one; the move mirrors its
    # accepted row back onto the vgb sub-state (GFRidgeGibbsMove).
    if ridge_gibbs_eligible(vgb_info):
        from ..sampling.ridge_fiber import make_gb_ridge_gibbs_move

        _vgb_ridge = make_gb_ridge_gibbs_move(
            priors["vgb"], tc,
            mc_lims=(tuple(vgb_info.m_chirp_lims)
                     if getattr(vgb_info, "m_chirp_lims", None)
                     else (0.001, 1.0)),
            dist_lims=vgb_info.dist_lims,
            ratio_max=float(vgb_info.fdot_astro_ratio_max),
            leaf_fraction=float(
                os.environ.get("GB_RIDGE_GIBBS_LEAF_FRACTION", "1.0")),
            branch_name="vgb",
        )
        _vgb_ridge.name = "vgb_ridge_gibbs"
        _vgb_ridge.accepted = np.zeros((1, nwalkers))
        vgb_moves.append(_vgb_ridge)
        logger.info(
            "build_vgb_moves: vgb_ridge_gibbs registered (leaf_fraction %s; "
            "zero-likelihood fiber move on the Mc-ratio-distance degeneracy "
            "of the %d-column chirp basis %s).",
            getattr(_vgb_ridge, "leaf_fraction", "1.0"),
            len(input_basis), input_basis,
        )
    else:
        logger.info(
            "build_vgb_moves: vgb_ridge_gibbs NOT registered -- the vgb "
            "basis %s does not carry all of (dist, Mc, fdot_astro_ratio) "
            "(or GB_RIDGE_GIBBS=0). Set VGB_CHIRP_MASS_BASIS=1 for the "
            "6-column chirp basis (NOT resume-compatible with a 5-dim "
            "store).", input_basis,
        )

    # Same temper-RNG seed base the GB twin stamps (fix round 5).
    _stamp_temper_seed_base(vgb_moves, general_info.random_seed)

    return vgb_moves


# ======================================================================
# Source move-builder hierarchy
#
# ``SourceMoveBuilder`` is the installable base class; per-source subclasses
# carry the design knobs. Settings files construct and ``build()`` these inside
# ``setup_recipe`` instead of hand-rolling the move construction. GB / PSD keep
# their richer function form (``build_gb_moves`` / ``build_psd_moves``) and the
# builder classes wrap them for a uniform ``(search_moves, pe_moves)`` return.
# ======================================================================


class SourceMoveBuilder:
    """Base class for building the recipe move(s) of one source branch.

    A builder is a light factory: construction carries the per-recipe *design
    knobs* (which move variants, ordering, thresholds), while :meth:`build`
    consumes the runtime context (``curr`` / ``acs`` / ``priors`` / ``state``)
    and returns ``(search_moves, pe_moves)`` — either list may be empty.
    Settings files construct and call a builder inside ``setup_recipe``; the
    repeatable machinery lives here in :mod:`recipe`.
    """

    #: Branch this builder targets; set on the subclass or via the constructor.
    branch_name: typing.Optional[str] = None

    def __init__(self, *, branch_name: typing.Optional[str] = None):
        if branch_name is not None:
            self.branch_name = branch_name
        assert self.branch_name is not None, "SourceMoveBuilder needs a branch_name"

    def build(self, engine_info, curr, acs, priors, state):
        """Return ``(search_moves, pe_moves)`` for this branch."""
        raise NotImplementedError


def resume_ladder_wins(branch_name, info, sub, ntemps):
    """Reconcile a per-leaf-ladder branch's rung count with a RESUMED state.

    Returns ``(ntemps, betas)``: the rung count the move must be built at
    and, when the store overrode the configuration, the store's own
    representative ladder (leaf 0's ``betas_all`` row) to build it with;
    ``betas`` is ``None`` whenever the configuration stands.

    ``sub`` is the branch's sub-state as loaded (``None`` / not yet
    tempered-initialized on a fresh start: the configuration stands). A
    resumed sub-state carries its coords, per-leaf log_like and counters
    at the rung count the store was written with, so that count is
    authoritative: the configured ``{BRANCH}_NTEMPS`` / ``betas`` is
    reported and ignored, and ``info.ntemps`` / ``info.betas`` are
    rewritten so every later consumer (``run.py::_branch_ntemps``, the
    midit checkpoint gate, the saver) reads the store's count too.
    """
    if sub is None or not getattr(sub, "tempered_initialized", False):
        return int(ntemps), None
    stored_nt = int(getattr(sub, "ntemps", 0) or 0)
    if stored_nt <= 0 or stored_nt == int(ntemps):
        return int(ntemps), None
    stored_all = getattr(sub, "betas_all", None)
    betas = None
    if stored_all is not None and np.shape(stored_all)[-1] == stored_nt:
        betas = np.asarray(stored_all, dtype=float).reshape(-1, stored_nt)[0].copy()
    else:
        betas = make_ladder(int(info.ndim), ntemps=stored_nt)
    configured = int(ntemps)
    info.ntemps = stored_nt
    info.betas = betas
    logger.warning(
        "%s branch resumed at the STORED %d-rung ladder %s -- the configured "
        "%d-rung ladder (%s_NTEMPS / an explicit betas) is IGNORED; the "
        "move's TemperatureControl, coords_shape, betas and accepted array "
        "follow the store. To actually run at %d rungs, re-rung the store "
        "(a migration) or start a fresh backend.",
        branch_name, stored_nt, np.array2string(betas, precision=4),
        configured, str(branch_name).upper(), configured,
    )
    return stored_nt, betas


def _stored_per_leaf_ladders(sub, nleaves, ntemps):
    """The sub-state's per-leaf ladders when they are a usable RESUMED set.

    ``None`` on a fresh start (``load_info`` builds the sub-state with
    ``betas_all=None``; the builder fills it) and for anything that is not a
    valid ``(nleaves, ntemps)`` ladder set -- finite, cold rung exactly 1,
    strictly decreasing, positive -- so a zeros template can never pass as a
    stored ladder.
    """
    if sub is None:
        return None
    stored = getattr(sub, "betas_all", None)
    if stored is None:
        return None
    arr = np.asarray(stored, dtype=float)
    if arr.shape != (int(nleaves), int(ntemps)):
        return None
    if not np.all(np.isfinite(arr)) or not np.all(arr[:, 0] == 1.0):
        return None
    if np.any(arr <= 0.0) or (int(ntemps) > 1 and np.any(np.diff(arr, axis=1) >= 0.0)):
        return None
    return arr.copy()


def resolve_resumed_ladders(branch_name, info, sub, betas, ntemps, configured_betas=None):
    """The per-leaf ``betas_all`` a single-source move is built with.

    * FRESH store (no usable stored ladders): every leaf gets ``betas``.
    * RESUMED store: every leaf KEEPS its stored, adapted ladder (the store's
      ``sub_backend/<branch>/betas_all`` row). Until 2026-10-13 the builder
      re-tiled the configured ladder here on every build, so each resume
      silently reset the per-leaf adaptation to the configuration -- in the
      9mo v9 store every leaf of mbh / emri / sobbh reads exactly 1/1.2^k
      (to 1e-16) at row 113, the first row of job 757's resume, while row 112
      had drifted from it by up to ~2e-3 (EMRI) and later rows by ~1e-2.
    * RESUMED store with ``{BRANCH}_LADDER_RESET_ON_RESUME=1`` (settings
      field ``ladder_reset_on_resume``): every leaf's ladder is REPLACED by
      the configured ladder, logged leaf by leaf (old -> new). The rung count
      stays the store's (``resume_ladder_wins`` ran first): a LONGER
      configured ladder is truncated to it (the builder's own rule; for a
      make_ladder ladder that is make_ladder at the store's count), a SHORTER
      one gives way to ``make_ladder(ndim, ntemps=<the store's count>)`` with
      a warning -- re-rung the store (``rerung_store_branches.py``) to change
      the count.

    ``betas`` is the ladder the builder resolved (configuration, or the
    store's leaf-0 row after a rung-count mismatch); ``configured_betas`` is
    the branch's configured ladder BEFORE ``resume_ladder_wins`` rewrote
    ``info.betas``. Returns a fresh ``(nleaves_max, ntemps)`` array.
    """
    nleaves = int(info.nleaves_max)
    ntemps = int(ntemps)
    tiled = np.tile(np.asarray(betas, dtype=float), (nleaves, 1))
    stored = _stored_per_leaf_ladders(sub, nleaves, ntemps)
    if stored is None:
        return tiled
    tag = f"[{str(branch_name).upper()}_LADDER]"
    knob = f"{str(branch_name).upper()}_LADDER_RESET_ON_RESUME"
    if not bool(getattr(info, "ladder_reset_on_resume", False)):
        logger.info(
            "%s resume keeps the STORED per-leaf ladders (%d leaves x %d rungs; "
            "leaf 0 %s). %s=1 replaces them with the configured ladder.",
            tag, nleaves, ntemps, np.array2string(stored[0], precision=5), knob,
        )
        return stored
    new = None
    if configured_betas is not None:
        cfg = np.asarray(configured_betas, dtype=float).ravel()
        if cfg.size >= ntemps:
            new = cfg[:ntemps].copy()
    if new is None:
        new = np.asarray(make_ladder(int(info.ndim), ntemps=ntemps), dtype=float)
        logger.warning(
            "%s %s=1: the configured ladder (%s rungs) does not fit the store's "
            "%d rungs; resetting to make_ladder(ndim=%d, ntemps=%d) instead. "
            "Re-rung the store to change the rung count.",
            tag, knob,
            "no" if configured_betas is None else len(np.ravel(configured_betas)),
            ntemps, int(info.ndim), ntemps,
        )
    out = np.tile(new, (nleaves, 1))
    logger.info(
        "%s %s=1: REPLACING the stored per-leaf ladders of %d leaves with %s",
        tag, knob, nleaves, np.array2string(new, precision=5),
    )
    for leaf in range(nleaves):
        logger.info(
            "%s leaf %d ladder reset: old %s -> new %s",
            tag, leaf, np.array2string(stored[leaf], precision=5),
            np.array2string(out[leaf], precision=5),
        )
    info.betas = new
    return out


def _branch_knob(info, attr, branch_name, knob, default, cast):
    """Settings field ``attr`` > ``{BRANCH}_<knob>`` env > ``default`` (``cast``-ed)."""
    val = getattr(info, attr, None)
    if val is None:
        env = os.environ.get(f"{str(branch_name).upper()}_{knob}", None)
        val = None if (env is None or not str(env).strip()) else env
    if val is None:
        return default
    return cast(val)


def resolve_snr_keyed_ladders(branch_name, info, sub, betas_all, ntemps, *, fresh):
    """SNR-keyed per-leaf ladders at construction (user ruling 2026-10-18; default OFF).

    With ``{BRANCH}_LADDER_TOP_SNR`` (settings field ``ladder_top_snr``) > 0,
    every leaf's ladder is ``make_ladder(ndim, ntemps, Tmax=T_leaf)`` with
    ``T_leaf = clip((SNR_leaf / top)^2, {BRANCH}_LADDER_TMAX_MIN,
    {BRANCH}_LADDER_TMAX)`` (:mod:`lisatools.sampling.snrladder`; the
    existing ``{BRANCH}_LADDER_TMAX`` is the CEILING then, its absence the
    uncapped make_ladder top). The rung count is the store's / the
    configuration's ``ntemps``, so a leaf below the ceiling gets a denser
    ladder than make_ladder's ~25 % spacing.

    * FRESH store (``fresh``) and a RESUMED store with
      ``{BRANCH}_LADDER_RESET_ON_RESUME=1``: each alive leaf with a recorded
      SNR on the sub-state (``d_h`` / ``h_h`` / ``log_like``, loaded from
      the store's last row on a resume) gets its SNR-keyed ladder now; a
      dead leaf (no source on any cold walker) gets the floor; an alive
      leaf without a recorded SNR keeps ``betas_all``'s (configured) row
      and is PENDING -- the move builds its ladder at the first propose
      that finds one (``ResidualAddOneRemoveOneMove._maybe_update_snr_ladders``).
    * Plain RESUME: the stored ladders stand (``resolve_resumed_ladders``
      already logged that); each leaf's refresh baseline becomes its first
      measured SNR.

    Returns ``(betas_all, plan)``: ``plan`` is ``None`` with the knob off
    (``betas_all`` returned untouched, bit-identical) or the move's
    ``snr_ladder_plan`` (``{"ref": {leaf: SNR built at}, "pending": {...}}``).
    """
    from ..sampling.snrladder import (
        LADDER_TMAX_MIN_DEFAULT,
        ceiling_tmax,
        check_clip,
        describe_leaf,
        leaf_alive,
        snr_keyed_ladder,
        snr_tmax,
        substate_leaf_snrs,
        top_snr_value,
    )

    top = _branch_knob(info, "ladder_top_snr", branch_name, "LADDER_TOP_SNR", 0.0, top_snr_value)
    if not top > 0.0:
        return betas_all, None
    B = str(branch_name).upper()
    tag = f"[{B}_LADDER]"
    ntemps = int(ntemps)
    nleaves = int(info.nleaves_max)
    if ntemps < 2:
        logger.info("%s %s_LADDER_TOP_SNR=%g: a %d-rung ladder has no temperatures to key; "
                    "nothing to do", tag, B, top, ntemps)
        return betas_all, None
    lo = _branch_knob(info, "ladder_tmax_min", branch_name, "LADDER_TMAX_MIN",
                      LADDER_TMAX_MIN_DEFAULT, float)
    cap = _branch_knob(info, "ladder_tmax", branch_name, "LADDER_TMAX", None, float)
    lo, hi = check_clip(lo, ceiling_tmax(int(info.ndim), ntemps, cap))
    plan = {"ref": {}, "pending": set()}
    reset = bool(getattr(info, "ladder_reset_on_resume", False))
    if not fresh and not reset:
        logger.info(
            "%s %s_LADDER_TOP_SNR=%g (floor %g, ceiling %g): the resume keeps the stored "
            "ladders; each leaf's refresh baseline is its first measured SNR "
            "(%s_LADDER_RESET_ON_RESUME=1 rebuilds them from the store's SNRs)",
            tag, B, top, lo, hi, B)
        return betas_all, plan
    why = "fresh store" if fresh else f"{B}_LADDER_RESET_ON_RESUME=1"
    out = np.array(betas_all, dtype=float, copy=True)
    measured = substate_leaf_snrs(sub, nleaves) if sub is not None else (
        [(float("nan"), None, None)] * nleaves)
    inds = None if sub is None else getattr(sub, "inds", None)
    logger.info("%s %s_LADDER_TOP_SNR=%g: per-leaf ladders from each leaf's SNR "
                "(T_max = clip((SNR / %g)^2, %g, %g), %d rungs, %s)",
                tag, B, top, top, lo, hi, ntemps, why)
    for leaf in range(nleaves):
        snr, walker, kind = measured[leaf]
        alive = leaf_alive(inds, leaf)
        if alive and not (snr is not None and np.isfinite(snr)):
            plan["pending"].add(leaf)
            logger.info(
                "%s leaf %d: no SNR recorded yet -> the configured ladder (T_max %.1f) until "
                "its first measurement (%s)", tag, leaf, float(1.0 / out[leaf, -1]), why)
            continue
        if not alive:
            snr, walker, kind = float("nan"), None, None
        t_leaf, raw, clip = snr_tmax(snr, top, lo, hi)
        out[leaf] = snr_keyed_ladder(int(info.ndim), ntemps, t_leaf)
        if alive:
            plan["ref"][leaf] = float(snr)
        logger.info(describe_leaf(tag, leaf, snr, t_leaf, raw, clip, top, lo, hi, out[leaf],
                                  why, kind=kind or "optimal", walker=walker))
    return out, plan


class SingleSourcePEBuilder(SourceMoveBuilder):
    """Build a :class:`ResidualAddOneRemoveOneMove` PE move for one branch.

    MBH / EMRI / SOBBH share the identical ``make_ladder`` -> ``betas_all`` ->
    ``coords_shape`` -> ``ResidualAddOneRemoveOneMove`` construction; this is
    that shared core. Subclasses set :attr:`branch_name` (and, where the source
    differs, :attr:`like_kwargs_from_waveform_kwargs`). Any constructor argument
    left ``None`` falls back to the matching field on
    ``curr.source_info[branch_name]``.
    """

    #: EMRI passes ``waveform_kwargs`` as the likelihood kwargs; MBH/SOBBH pass
    #: an empty dict. Encoded as a class flag so the per-source default lives on
    #: the subclass while the construction stays shared.
    like_kwargs_from_waveform_kwargs: bool = False

    #: Waveform-only keys this source's branch ``waveform_kwargs`` may carry
    #: that the LIKELIHOOD path cannot accept. Empty here, so every source
    #: except the ones that override it assembles an unfiltered copy.
    like_kwargs_strip_keys: typing.FrozenSet[str] = frozenset()

    #: The move class this builder constructs — subclasses override to swap in
    #: a fast-likelihood ResidualAddOneRemoveOneMove subclass.
    move_class: type = ResidualAddOneRemoveOneMove

    #: Whether the builder may hand the run-shared DCGA to the move. Fast
    #: kernel moves with no multi-device path set this False.
    use_dcga: bool = True

    def __init__(
        self,
        *,
        branch_name: typing.Optional[str] = None,
        wave_gen: typing.Callable,
        waveform_gen_kwargs: typing.Optional[dict] = None,
        waveform_like_kwargs: typing.Optional[dict] = None,
        num_repeats: typing.Optional[int] = None,
        inner_moves: typing.Optional[list] = None,
        transform=None,
        betas: typing.Optional[np.ndarray] = None,
        Tmax: float = np.inf,
        permute_every: int = 20,
        move_name: typing.Optional[str] = None,
        **move_kwargs,
    ):
        super().__init__(branch_name=branch_name)
        self.wave_gen = wave_gen
        self.waveform_gen_kwargs = waveform_gen_kwargs
        self.waveform_like_kwargs = waveform_like_kwargs
        self.num_repeats = num_repeats
        self.inner_moves = inner_moves
        self.transform = transform
        self.betas = betas
        self.Tmax = Tmax
        self.permute_every = permute_every
        self.move_name = move_name
        self.move_kwargs = move_kwargs

    def assemble_gen_kwargs(self, info) -> dict:
        """The move's WAVEFORM-generation kwargs (an explicit dict wins)."""
        return (
            self.waveform_gen_kwargs
            if self.waveform_gen_kwargs is not None
            else info.waveform_kwargs
        ).copy()

    def assemble_like_kwargs(self, info) -> dict:
        """The move's LIKELIHOOD kwargs (an explicit dict wins).

        :attr:`like_kwargs_strip_keys` is dropped on the way out: these kwargs
        end up at ``inner_product(**kwargs)`` (via ``compute_like`` ->
        ``compute_acs_like`` -> ``AnalysisContainer.template_likelihood``),
        which has no ``**kwargs`` sink, so a waveform-only key in the branch
        ``waveform_kwargs`` is a TypeError on the run's first likelihood call
        (the 2026-09-15 6mo crash: ``mode_selection_threshold``). Sources that
        strip nothing get an unfiltered copy, exactly as before.
        """
        if self.waveform_like_kwargs is not None:
            like_kw = self.waveform_like_kwargs.copy()
        elif self.like_kwargs_from_waveform_kwargs:
            like_kw = info.waveform_kwargs.copy()
        else:
            return dict()
        for key in self.like_kwargs_strip_keys:
            like_kw.pop(key, None)
        return like_kw

    def build(self, engine_info, curr, acs, priors, state):
        info = curr.source_info[self.branch_name]
        gi = curr.general_info
        nwalkers = _body_nwalkers(curr, acs, self.move_class)
        # this branch's OWN per-leaf ladder size (the engine runs cold-chain
        # only); an explicit betas ladder wins over the ntemps knob
        _info_betas = getattr(info, "betas", None)
        ntemps = (
            len(_info_betas)
            if _info_betas is not None
            else int(getattr(info, "ntemps", None) or gi.ntemps)
        )
        # the CONFIGURED ladder, before resume_ladder_wins may overwrite
        # info.betas with the store's (the {BRANCH}_LADDER_RESET_ON_RESUME
        # path below needs the configuration, not the store)
        _configured_betas = (
            None if _info_betas is None else np.asarray(_info_betas, dtype=float).copy()
        )
        # RESUME: the STORED rung count wins (same rule as the banded
        # branches, GBState.initialize_band_information step 3). A resumed
        # sub-state carries its coords / per-leaf log_like / counters at the
        # rung count the store was written with; sizing the move off the
        # configured {BRANCH}_NTEMPS instead leaves a 12-rung
        # TemperatureControl (and a 12-row coords_shape) against an 8-rung
        # state -- or the reverse -- and the mismatch surfaces later as
        # "cannot reshape array of size 88 into shape (12,newaxis)" in the
        # per-walker eigen sweep (2026-09-17 relaunch) or as the mass
        # eigen-table rebuilds of snapshot 7 (2026-09-16). Re-rungging a
        # live store is a migration, never a mid-resume side effect.
        ntemps, _stored_betas = resume_ladder_wins(
            self.branch_name, info, state.sub_states.get(self.branch_name),
            ntemps,
        )

        # Ladder resolution honors the class contract ("any argument left
        # None falls back to the matching field on source_info"): an explicit
        # builder ladder wins, then the branch-configured ladder (the Setups'
        # make_ladder(ndim, ntemps) default since 2026-10-13, or an explicit
        # betas), then the make_ladder default here.
        # Previously info.betas was skipped entirely, so the configured EMRI
        # ladder never reached the move's per-leaf temperature controls.
        # On a rung-count resume mismatch the store's own ladder is the one
        # the run was actually sampling at, so it wins over all of those.
        betas = _stored_betas if _stored_betas is not None else self.betas
        if betas is None:
            betas = getattr(info, "betas", None)
            if betas is not None:
                betas = np.asarray(betas, dtype=float)
                if len(betas) < ntemps:
                    logger.warning(
                        "%s: configured betas ladder has %d temps < run "
                        "ntemps %d; falling back to make_ladder.",
                        self.branch_name, len(betas), ntemps,
                    )
                    betas = None
                elif len(betas) > ntemps:
                    logger.warning(
                        "%s: truncating configured %d-temp betas ladder to "
                        "the run's ntemps=%d (max T %.3g -> %.3g).",
                        self.branch_name, len(betas), ntemps,
                        1.0 / betas[-1], 1.0 / betas[ntemps - 1],
                    )
                    betas = betas[:ntemps]
        if betas is None:
            betas = make_ladder(int(info.ndim), ntemps=int(ntemps))
        # Side effects (parity with the old build_mbh_moves_phenom): stash the
        # ladder on the source info and the per-leaf ladders on the sub-state.
        info.betas = betas
        _sub_in = state.sub_states.get(self.branch_name)
        _fresh = _stored_per_leaf_ladders(_sub_in, int(info.nleaves_max), int(ntemps)) is None
        betas_all = resolve_resumed_ladders(
            self.branch_name, info, _sub_in,
            betas, ntemps, _configured_betas,
        )
        # {BRANCH}_LADDER_TOP_SNR (default off: betas_all untouched, plan None)
        betas_all, _snr_plan = resolve_snr_keyed_ladders(
            self.branch_name, info, _sub_in, betas_all, ntemps, fresh=_fresh,
        )
        state.sub_states[self.branch_name].betas_all = betas_all
        logger.debug(f"{self.branch_name} betas: {betas}")

        coords_shape = (ntemps, nwalkers, info.nleaves_max, info.ndim)

        wf_gen_kw = self.assemble_gen_kwargs(info)
        wf_like_kw = self.assemble_like_kwargs(info)

        # Multi-GPU spread (one move structure): hand the run-shared DCGA to
        # the unified move when the generator can be replicated per device —
        # a bound method of an object exposing ``.kwargs``. Otherwise the
        # move runs its shard-aware ACA path (correct on any shard count).
        wave_gen_in = self.wave_gen
        dcga_kwargs = {}
        dcga = get_shared_dcga(acs) if self.use_dcga else None
        if dcga is not None:
            gen_obj = getattr(self.wave_gen, "__self__", None)
            if gen_obj is not None and hasattr(gen_obj, "kwargs"):
                wave_gen_in = gen_obj
                dcga_kwargs = dict(
                    dcga=dcga,
                    waveform_gen_method=self.wave_gen.__name__,
                    run_threaded=True,
                )
            else:
                logger.info(
                    "%s: per-device replica path unavailable (wave_gen has "
                    "no .kwargs-bearing object); using the shard-aware ACA "
                    "path.", self.branch_name,
                )

        # eigen inner-move plumbing: the per-branch refresh cadence rides
        # the settings field; an explicit move_kwargs entry wins. A
        # configured ``info_matrix_gen`` becomes the external table builder
        # (``builder(move, leaf, widths) -> (axes, sigmas)``).
        _extra_kwargs = dict(dcga_kwargs)
        _extra_kwargs.update(self.move_kwargs)
        _extra_kwargs.setdefault(
            "eigen_refresh_every",
            getattr(info, "eigen_refresh_every", None),
        )
        _extra_kwargs.setdefault(
            "eigen_table_scope",
            getattr(info, "eigen_table_scope", None),
        )
        # the eigen step-scale adaptation's master switch
        # ({BRANCH}_EIGEN_AXIS_ADAPT via the settings field; None lets the
        # move read the env / its default, off)
        _extra_kwargs.setdefault(
            "eigen_axis_adapt",
            getattr(info, "eigen_axis_adapt", None),
        )
        # the lost-rung guard's threshold ({BRANCH}_LADDER_REANCHOR_NATS via
        # the settings field; None lets the move read the env / its default)
        _extra_kwargs.setdefault(
            "ladder_reanchor_nats",
            getattr(info, "ladder_reanchor_nats", None),
        )
        # the SNR-keyed per-leaf ladders ({BRANCH}_LADDER_TOP_SNR & co. via
        # the settings fields; None lets the move read the env / its default)
        for _attr in ("ladder_top_snr", "ladder_tmax", "ladder_tmax_min",
                      "ladder_top_snr_refresh", "ladder_top_snr_refresh_pe"):
            _extra_kwargs.setdefault(_attr, getattr(info, _attr, None))
        move = self.move_class(
            self.branch_name,
            coords_shape,
            wave_gen_in,
            wf_gen_kw,
            wf_like_kw,
            acs,
            self.num_repeats if self.num_repeats is not None else info.num_prop_repeats,
            self.transform if self.transform is not None else info.transform,
            priors,
            self.inner_moves if self.inner_moves is not None else info.inner_moves,
            Tmax=self.Tmax,
            betas_all=betas_all,
            permute_every=self.permute_every,
            name=self.move_name,
            **_extra_kwargs,
        )
        move.fanout_branches = [self.branch_name]
        if _snr_plan is not None:
            # what the construction resolved: per-leaf build SNRs + pending leaves
            move.snr_ladder_plan = _snr_plan
        if getattr(info, "info_matrix_gen", None) is not None:
            move.eigen_table_builder = info.info_matrix_gen
        # where the eigen tables persist to: the same store path the
        # mid-iteration checkpoint sidecars off (moves.eigen_table_persist),
        # so a restarted process reloads the tables instead of repaying
        # minutes of information-matrix build per leaf
        move.eigen_store_path = getattr(gi, "main_file_path", None)
        # multi-rank: bind the run's fan-out (no-op single-process); clears the
        # sidecar path on non-head ranks (single writer) and freezes the ladders
        # on every rank (the head adapts once per propose from pooled tallies)
        move.install_walker_fanout(curr)
        move.accepted = np.zeros((ntemps, nwalkers))
        return [], [move]


class MBHMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"mbh"`` branch."""

    branch_name = "mbh"


class MBHBatchedMoveBuilder(MBHMoveBuilder):
    """:class:`MBHMoveBuilder` constructing :class:`MBHBatchedLikeMove`.

    ``wave_gen`` stays the SLOW exact generator (residual parity with the
    engine + the fast-vs-slow cross-check); the windowed adapter and its knobs
    pass through ``move_kwargs`` (``batched_gen=``, ``batch_max_size=``,
    ``window_*=``). The DCGA branch is skipped: the move raises if handed one.
    """

    move_class = MBHBatchedLikeMove
    use_dcga = False


class EMRIMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"emri"`` branch."""

    branch_name = "emri"
    like_kwargs_from_waveform_kwargs = True
    #: FEW's per-call mode-selection threshold (the ``EMRI_EPS`` knob) is a
    #: WAVEFORM-build kwarg; it reaches the generator through the wave wrap's
    #: ``runtime_kwargs`` (``stock.erebor.source_runtime.get_emri_wave_wrap``),
    #: never through the likelihood. A hand-written
    #: ``emri.waveform_kwargs["mode_selection_threshold"]`` would otherwise
    #: ride this branch's like-kwargs into ``inner_product`` and TypeError.
    like_kwargs_strip_keys = frozenset({"mode_selection_threshold"})


class EMRIDirectMoveBuilder(EMRIMoveBuilder):
    """:class:`EMRIMoveBuilder` constructing :class:`EMRIDirectLikeMove`.

    ``wave_gen`` stays the PRODUCTION generator (the residual expose/fold and the
    fast-vs-slow cross-check); the direct-to-WDM adapter and the chunk size pass through
    ``move_kwargs`` (``direct_gen=``, ``batch_max_size=``). The DCGA branch is skipped:
    the move raises if handed one.
    """

    move_class = EMRIDirectLikeMove
    use_dcga = False


class SOBBHMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"sobbh"`` branch.

    Like EMRI, SOBBH passes ``waveform_kwargs`` as the likelihood kwargs (this
    matches the EMRI/SOBBH inline builders it replaces).
    """

    branch_name = "sobbh"
    like_kwargs_from_waveform_kwargs = True


class SOBBHChunkedMoveBuilder(SOBBHMoveBuilder):
    """:class:`SOBBHMoveBuilder` constructing :class:`SOBBHChunkedLikeMove`.

    ``wave_gen`` stays the SLOW exact wrap (residual expose/fold + the
    fast-vs-slow cross-check); the chunked comp and its band knob are passed
    through ``move_kwargs`` (``chunked_comp=``, ``m_band_half_width=``).
    The DCGA branch is skipped — the chunked kernel is single-shard by
    contract and the move raises if handed a dcga.
    """

    move_class = SOBBHChunkedLikeMove
    use_dcga = False


class GBMoveBuilder(SourceMoveBuilder):
    """Build the GB search + PE move lists.

    Wraps :func:`build_gb_moves` (which owns the GB machinery + the GB reference
    move recipe) in the :class:`SourceMoveBuilder` interface. Design knobs are
    forwarded so a settings file can steer which moves it installs.
    """

    branch_name = "gb"

    def __init__(
        self,
        *,
        Tmax: float = 1e6,
        include_search: bool = True,
        include_refit: bool = True,
        pe_move_names: typing.Optional[list] = None,
    ):
        super().__init__(branch_name="gb")
        self.Tmax = Tmax
        self.include_search = include_search
        self.include_refit = include_refit
        self.pe_move_names = pe_move_names

    def build(self, engine_info, curr, acs, priors, state):
        return build_gb_moves(
            engine_info,
            curr,
            acs,
            priors,
            state,
            Tmax=self.Tmax,
            include_search=self.include_search,
            include_refit=self.include_refit,
            pe_move_names=self.pe_move_names,
        )


class VGBMoveBuilder(SourceMoveBuilder):
    """Build the VGB in-model stretch move (wraps :func:`build_vgb_moves`).

    No search moves — verification binaries are known sources; the single
    PE move is the fixed-dimensional same-leaf stretch.
    """

    branch_name = "vgb"

    def __init__(self, *, Tmax: float = 1e6):
        super().__init__(branch_name="vgb")
        self.Tmax = Tmax

    def build(self, engine_info, curr, acs, priors, state):
        return [], build_vgb_moves(
            engine_info, curr, acs, priors, state, Tmax=self.Tmax
        )


class PSDMoveBuilder(SourceMoveBuilder):
    """Build a noise search + PE move pair (wraps :func:`build_noise_moves`).

    ``sampled_branches`` selects which noise branches THIS pair samples
    (noise-move split 2026-07). ``None`` keeps the historical joint
    behavior (all noise branches present in the run, one ladder).
    """

    branch_name = "psd"

    def __init__(
        self,
        *,
        sampled_branches: list = None,
        num_repeats: int = None,
        permute_every: int = 50,
        Tmax: float = 1e6,
    ):
        super().__init__(branch_name=(sampled_branches or ["psd"])[0])
        self.sampled_branches = sampled_branches
        self.num_repeats = num_repeats
        self.permute_every = permute_every
        self.Tmax = Tmax

    def build(self, engine_info, curr, acs, priors, state):
        sampled = self.sampled_branches
        if sampled is None:
            sampled = [
                b for b in ("psd", "galfor", "sgwb") if b in curr.source_info
            ]
        search_move, pe_move = build_noise_moves(
            engine_info,
            curr,
            acs,
            priors,
            sampled_branches=sampled,
            num_repeats=self.num_repeats,
            permute_every=self.permute_every,
            Tmax=self.Tmax,
        )
        return [search_move], [pe_move]
