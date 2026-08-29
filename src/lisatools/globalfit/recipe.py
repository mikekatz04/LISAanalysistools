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
        stop_here = self._current_recipe_step["adjust"].stopping_function(
            iteration, last_sample, sampler
        )
        if stop_here:
            self.backend.completed_recipe_step(self._current_recipe_step["name"])
            self._current_recipe_step["status"] = True
            next(self)

            if self._current_iter >= len(self.recipe):
                return True
            self._current_recipe_step["adjust"].setup_run(iteration, last_sample, sampler)

        return False


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
    """Recipe step that runs indefinitely (ongoing parameter estimation)."""

    def stopping_function(self, *args, **kwargs):
        """Never stop on its own — relies on outer stopping logic."""
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


_STEP_CLASSES = {"search": SearchRecipeStep, "pe": PERecipeStep, "rj": RJRecipeStep}


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
      cold-chain leaf count plateaus; knobs via ``step_kwargs``).

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

    _KINDS = ("search", "pe", "rj")

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
        combined = GFCombineMove(moves=runtime_moves, **combine_kwargs)
        combined.gf_stage_name = self.name
        # ... and the stage KIND, which the coarse-noise mode resolver reads
        # ("auto" -> search_approx in a search/rj stage, delayed_acceptance in
        # a pe stage). It must ride the per-stage COMBINE move rather than the
        # sub-moves: stock moves are shared by name, so the very same PSDMove
        # instance serves noise_search and full_pe, and a static stamp on it
        # would be whichever stage materialised last. GFCombineMove re-stamps
        # its children immediately before each propose instead.
        combined.gf_stage_kind = self.kind
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
    nwalkers = int(curr.general_info.nwalkers)
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
    nwalkers: int = general_info.nwalkers
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
    temperature_control = TemperatureControl(
        effective_ndim,
        nwalkers,
        betas=lead_info.betas,
        ntemps=ntemps,
        Tmax=Tmax,
        permute=False,
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


def _fstat_dist_birth_stamp(default: bool = True) -> bool:
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
    """
    env = os.environ.get("GB_RJ_FSTAT_DIST_BIRTH")
    return bool(int(env)) if env is not None else bool(default)


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
    if getattr(gb_info, "signal_gen", None) is None and state.branches["gb"].inds[0].sum() > 0:

        coords_out_gb = state.branches["gb"].coords[0,
            state.branches["gb"].inds[0]
        ]
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
            np.arange(nwalkers), (nleaves_max_gb, 1)
        ).transpose((1, 0))[state.branches["gb"].inds[0]]

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
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )
    gb_move_kwargs = dict(
        waveform_kwargs=gb_info.waveform_kwargs,
        parameter_transforms=gb_info.transform,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info.random_seed,
        force_backend=general_info.force_backend,
        nfriends=nwalkers,
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
    # rotated to the maximum). GB_RJ_PHASE_MAXIMIZE=1 turns it on --
    # gb_no_foreground defaults it ON under GB_MODE=search (the
    # "annealing" configuration) and OFF otherwise.
    _rj_phase_max = bool(int(os.environ.get("GB_RJ_PHASE_MAXIMIZE", "0")))

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
            ),
        )
        if _custom_birth is not None:
            logger.warning(
                "GB_FSTAT_FIT_IN_MOVE=1 supersedes the supplied "
                "rj_birth_distribution; drop a prebuilt grid into "
                "%s/<move name>/epoch_0000/ to have it loaded instead.",
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
        **{**gb_move_kwargs, **_imr_search}
    )
    gb_search_prune_move.accepted = np.zeros((ntemps, nwalkers))
    
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
        phase_maximize=True,
        gpus=[],
        # Leaf-cap counters advance once per iteration: the prior RJ move is
        # the designated updater; the other RJ moves only enforce the gate.
        **{**gb_move_kwargs, "leaf_cap_update": False,
           "rj_flip_fraction_default": _SEARCH_RJ_FLIP_DEFAULT, **_imr_search}
    )
    gb_search_fstat_mcmc_move.accepted = np.zeros((ntemps, nwalkers))

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
            phase_maximize=True,  # gb_info["pe_info"]["rj_phase_maximize"],
            gpus=[],
            **{**gb_move_kwargs, "leaf_cap_update": False,
               "rj_flip_fraction_default": _SEARCH_RJ_FLIP_DEFAULT}
        )
        gb_search_refit_move.accepted = np.zeros((ntemps, nwalkers))

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
               "rj_flip_fraction_default": _SEARCH_RJ_FLIP_DEFAULT, **_imr_search},
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
        gb_replace_move.accepted = np.zeros((ntemps, nwalkers))
    # Pure IN-MODEL move (2026-08-04): no RJ step at all -- ``is_rj_prop=False``
    # skips the birth/death branch in the round loop, so every pick round is
    # just ``num_repeat_proposals`` in-model repeats on the live sources.
    # Placed between the fstat-birth move and the removal move so freshly-born
    # sources get a full refinement pass to climb the likelihood BEFORE
    # ``rj_prior_removal`` judges them for death (a source still sitting at its
    # birth coordinates looks far more deletable than the same source after it
    # has walked onto its peak).
    gb_in_model_move = None
    if _gb_mode_search and getattr(gb_info, "search_in_model", False):
        gb_in_model_move = GBSpecialStretchMove(
            *gb_move_args,
            rj_proposal_distribution=None,
            is_rj_prop=False,          # THE switch: in-model repeats only
            name="in_model",
            phase_maximize=False,      # in-model scoring is at the actual phase
            run_swaps=False,
            gpus=[],
            # Cap counters advance exactly once per iteration on ``rj_prior``;
            # this move must not touch them (it changes no dimensions).
            **{**gb_move_kwargs, "leaf_cap_update": False},
        )
        gb_in_model_move.accepted = np.zeros((ntemps, nwalkers))

    gb_prior_removal_move = None
    if _gb_mode_search and getattr(gb_info, "search_prior_removal", False):
        gb_prior_removal_move = GBSpecialRJPriorMove(
            *gb_move_args,
            rj_proposal_distribution=gpu_priors,  # THE prior container
            name="rj_prior_removal",
            rj_removal_only=True,
            # Deaths never phase-maximize; False also keeps the amp-pin /
            # fstat-dist-birth ctor defaults off, so death factors are the
            # plain prior logpdf.
            phase_maximize=False,
            # Band-temperature swaps live HERE when _temper_on_removal
            # (see the placement comment at the search-move block).
            run_swaps=_temper_all_moves or _temper_on_removal,
            gpus=[],
            # Removal pools are 100% mature -- the survivor budget (25)
            # is the one that binds here.
            **{**gb_move_kwargs, "leaf_cap_update": False,
               "rj_flip_fraction_default": _SEARCH_RJ_FLIP_DEFAULT, **_imr_search},
        )
        gb_prior_removal_move.accepted = np.zeros((ntemps, nwalkers))

    #* ===================== WARM-START RJ BIRTH MOVE (rj_warm_search) =====================
    # Workstream B (user ruling 2026-08-24: "add this proposal in GB search
    # ... BEFORE the fstat proposal"). GB_WARM_START_COMPONENTS (or
    # ``GBSettings.warm_start_components``) points at the clustered
    # previous-run posterior components npz produced OFFLINE by
    # scripts/gb/warmstart_fit_from_store.py; when set (and a search recipe
    # is being built) this constructs ``rj_warm_search`` -- the EXISTING
    # prior-RJ move class with its birth distribution swapped for the
    # warm-start f0-windowed Gaussian mixture
    # (lisatools.sampling.warmstart_proposal.WarmStartComponents):
    #
    #   * births draw FULL 9-column GB parameters -- the warm draws carry
    #     their own phases/amplitudes from the previous posterior, so
    #     phase_maximize stays OFF;
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
    _warm_path = str(
        getattr(gb_info, "warm_start_components", "")
        or os.environ.get("GB_WARM_START_COMPONENTS", "")
        or ""
    ).strip()
    if include_search and _warm_path:
        from ..sampling.warmstart_proposal import WarmStartComponents

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
        _warm_container = WarmStartComponents.from_npz(
            _warm_path,
            new_tobs=float(general_info.Tobs),
            use_cupy=use_gpu_priors,
            floor_box=(_warm_floor_lo, _warm_floor_hi),
            floor_eps=float(os.environ.get("GB_WARM_START_FLOOR_EPS", "0.05")),
            p_floor=float(os.environ.get("GB_WARM_START_P_FLOOR", "0")),
            seed=general_info.random_seed,
        )
        gb_warm_move = GBSpecialRJPriorMove(
            *gb_move_args,
            rj_proposal_distribution={"gb": _warm_container},
            name="rj_warm_search",
            use_prior_removal=False,
            # NO phase maximization: warm draws carry their own
            # phases/amplitudes from the previous posterior.
            phase_maximize=False,
            run_swaps=False,
            gpus=[],
            **{**gb_move_kwargs, "leaf_cap_update": False,
               "rj_flip_fraction_default": _SEARCH_RJ_FLIP_DEFAULT, **_imr_search},
        )
        gb_warm_move.accepted = np.zeros((ntemps, nwalkers))
        gb_search_moves = list(gb_search_moves) + [gb_warm_move]
        logger.info(
            "build_gb_moves: rj_warm_search armed from %s (%d components; "
            "weights ~ p, floor_eps=%s).",
            _warm_path, _warm_container.n_components,
            os.environ.get("GB_WARM_START_FLOOR_EPS", "0.05"),
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
        _SEARCH_RJ_FLIP_DEFAULT if _gb_mode_search else _PE_RJ_FLIP_DEFAULT
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

    # BAND-UNIT REPEATS ARE SEARCH-ONLY (user ruling 2026-08-29). N
    # consecutive passes per band residue class costs LINEARLY in N, and
    # it is a search-stage concentration device -- PE must never inherit
    # it from an exported GB_BAND_UNIT_REPEATS. Pinned to 1 (the ctor
    # kwarg beats the env in _resolve_band_unit_repeats) whenever this
    # move is not a search move:
    #   * pe-named moves under GB_PE_MOVES_STRICT (a staged process
    #     carrying both search- and pe-named GB stages), and
    #   * every GB move outside GB_MODE=search.
    # Under GB_MODE=search WITHOUT strict-PE the pe-named moves ARE the
    # search (the single-stage gb_no_fg campaigns), so they take the env.
    if _pe_strict or not _gb_mode_search:
        _pe_cap_off = {**_pe_cap_off, "band_unit_repeats": 1}

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

    gb_pe_prior_move = _RJBirth(
        *gb_move_args,
        rj_proposal_distribution=(None if _fit_in_move else _rj_birth_prop),
        **({"is_rj_prop": True} if _fit_in_move else {}),
        **_fit_kwargs,
        name="rj_fstat_pe",
        use_prior_removal=False,  # gb_info["pe_info"]["use_prior_removal"],
        phase_maximize=(
            _rj_phase_max if (_gb_mode_search and not _pe_strict) else False
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
    gb_pe_prior_move.accepted = np.zeros((ntemps, nwalkers))

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
            **_fit_kwargs,
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
        gb_pe_replace_move.accepted = np.zeros((ntemps, nwalkers))

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
    gb_pe_prior_birth_move.accepted = np.zeros((ntemps, nwalkers))

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
    gb_pe_fstat_mcmc_move.accepted = np.zeros((ntemps, nwalkers))

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
    gb_pe_moves = [gb_pe_prior_move]
    if gb_pe_replace_move is not None:
        gb_pe_moves.append(gb_pe_replace_move)
    gb_pe_moves += [gb_pe_prior_birth_move, gb_pe_fstat_mcmc_move]
    if gb_replace_move is not None:
        gb_pe_moves.append(gb_replace_move)
    if gb_in_model_move is not None:
        gb_pe_moves.append(gb_in_model_move)
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
            **{**gb_move_kwargs, "leaf_cap_update": False, **_pe_cap_off}
        )
        gb_pe_refit_move.accepted = np.zeros((ntemps, nwalkers))
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
    if (os.environ.get("GB_RIDGE_GIBBS", "1") == "1"
            and getattr(gb_info, "fdot_astro_ratio_max", None) is not None
            and "fdot_astro_ratio" in list(
                getattr(gb_info.transform, "input_basis", []) or [])):
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
        _ridge.accepted = np.zeros((1, nwalkers))
        gb_search_moves = list(gb_search_moves) + [_ridge]
        gb_pe_moves = list(gb_pe_moves) + [_ridge]
        logger.info("build_gb_moves: gb_ridge_gibbs registered (leaf_fraction "
                    "%s; zero-likelihood fiber move on the Mc-ratio-distance "
                    "degeneracy).", _ridge.leaf_fraction
                    if hasattr(_ridge, "leaf_fraction") else "1.0")

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

    Returns a one-element list (the PE move).
    """
    vgb_info = curr.source_info["vgb"]
    general_info: GeneralSetup = curr.general_info
    nwalkers: int = general_info.nwalkers
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
    if (
        getattr(vgb_info, "signal_gen", None) is None
        and state.branches["vgb"].inds[0].sum() > 0
    ):
        inds0 = state.branches["vgb"].inds[0]  # (nwalkers, nleaves)
        coords_out = state.branches["vgb"].coords[0, inds0]
        for _name, _per in (("phi0", 2 * np.pi), ("psi", np.pi)):
            _i = input_basis.index(_name)
            coords_out[:, _i] = coords_out[:, _i] % _per

        check = priors["vgb"].logpdf(coords_out)
        if np.any(np.isinf(check)):
            raise ValueError(
                "VGB starting coordinates fall outside the priors; check the "
                "VGB prior limits against the catalogue values."
            )

        leaf_inds = np.tile(np.arange(nleaves_max_vgb), (nwalkers, 1))[inds0]
        coords_in_in = tc.both_transforms(coords_out, leaf_inds=leaf_inds)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1
        walker_vals = np.tile(
            np.arange(nwalkers), (nleaves_max_vgb, 1)
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
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
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
        nfriends=nwalkers,
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
    vgb_move.accepted = np.zeros((ntemps, nwalkers))
    return [vgb_move]


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

    def build(self, engine_info, curr, acs, priors, state):
        info = curr.source_info[self.branch_name]
        gi = curr.general_info
        nwalkers = gi.nwalkers
        # this branch's OWN per-leaf ladder size (the engine runs cold-chain
        # only); an explicit betas ladder wins over the ntemps knob
        _info_betas = getattr(info, "betas", None)
        ntemps = (
            len(_info_betas)
            if _info_betas is not None
            else int(getattr(info, "ntemps", None) or gi.ntemps)
        )

        # Ladder resolution honors the class contract ("any argument left
        # None falls back to the matching field on source_info"): an explicit
        # builder ladder wins, then the branch-configured ladder
        # (e.g. EMRISetup's dense 1/1.2^k), then the make_ladder default.
        # Previously info.betas was skipped entirely, so the configured EMRI
        # ladder never reached the move's per-leaf temperature controls.
        betas = self.betas
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
            betas = make_ladder(info.ndim, ntemps=ntemps)
        # Side effects (parity with the old build_mbh_moves_phenom): stash the
        # ladder on the source info and the tiled ladder on the sub-state.
        info.betas = betas
        betas_all = np.tile(betas, (info.nleaves_max, 1))
        state.sub_states[self.branch_name].betas_all = betas_all
        logger.debug(f"{self.branch_name} betas: {betas}")

        coords_shape = (ntemps, nwalkers, info.nleaves_max, info.ndim)

        wf_gen_kw = (
            self.waveform_gen_kwargs
            if self.waveform_gen_kwargs is not None
            else info.waveform_kwargs
        ).copy()
        if self.waveform_like_kwargs is not None:
            wf_like_kw = self.waveform_like_kwargs.copy()
        elif self.like_kwargs_from_waveform_kwargs:
            wf_like_kw = info.waveform_kwargs.copy()
        else:
            wf_like_kw = dict()

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
            **dcga_kwargs,
            **self.move_kwargs,
        )
        move.accepted = np.zeros((ntemps, nwalkers))
        return [], [move]


class MBHMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"mbh"`` branch."""

    branch_name = "mbh"


class EMRIMoveBuilder(SingleSourcePEBuilder):
    """:class:`SingleSourcePEBuilder` for the ``"emri"`` branch."""

    branch_name = "emri"
    like_kwargs_from_waveform_kwargs = True


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
