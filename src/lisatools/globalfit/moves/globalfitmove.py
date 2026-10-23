"""Global-fit move base classes and the per-stage combine move.

Home of the declarative :class:`Move` base — the single user-facing
"move" concept of the global-fit recipe layer (a move is a proposal; the
words are synonymous). A ``Move`` is cheap to construct and picklable; the
heavy work happens in its required ``setup(ctx)`` hook, run once at recipe
materialization with a :class:`MoveBuildContext`.

Contents, in pipeline order:

* :class:`Move` / :class:`MoveBuildContext` (and the private ``_RuntimeMove``
  wrapper): what a recipe :class:`~lisatools.globalfit.recipe.Stage` holds and
  what ``Stage.setup`` hands each move at materialization.
* :class:`GlobalFitMove`: the mixin every runtime global-fit move carries
  (GB / VGB, noise, sources). It provides the sub-state helpers
  (``_work_branch``, ``_sync_cold_row``, ``_check_substate_consistency``), the
  fan-out ``gf_serve`` hook and the uniform ``set_debug``.
* :class:`GFCombineMove`: ``Stage.setup`` wraps each stage's runtime moves in ONE
  of these; eryn calls its ``propose`` exactly once per global-fit iteration.
  It owns the iteration counter, the sub-state guards, the search legs
  (``GB_SEARCH_LEGS``), the PE draw-one RJ plan (``GB_PE_RJ_DRAW_ONE``), the
  per-move cadence (``Move.every``) and the ``[GF_TIMING]`` instrumentation.
* :class:`MaxLogLCombineMove`: the plateau-terminated joint search used by the
  production ``source_search`` stage and by ``gb_search_2``'s cycle-end noise
  search (driver ``JointMaxLogLSearch``).

Env knobs read here (all at call time): ``GF_MOVE_TIMING`` / ``GF_MOVE_TIMING_SYNC``
(default ``0``; launchers ``1`` / ``0``), ``GF_SUBSTATE_CHECK`` (default ``1``),
``GB_PE_RJ_DRAW_ONE`` (default ``0``; launchers ``1``), ``GB_PE_RJ_FSTAT_FRACTION``
(``0.8``; launchers ``0.45``), ``GB_PE_RJ_FSTAT_RESIDUAL_SPLIT`` (``0.5``; launchers
``0.9``), ``GB_PE_RJ_WARM_FRACTION`` (``0``; launchers ``0.4``),
``GB_PE_RJ_PARA_FRACTION`` (``0``), and the ``MAXLOGL_*`` knobs of
:class:`MaxLogLCombineMove`.
"""

import dataclasses
import logging
import os
import resource
import sys
import time
import typing

import numpy as np
from eryn.moves import CombineMove

from .. import midit_checkpoint

logger = logging.getLogger(__name__)


def ensure_fine_noise_covariance_current(acs, coarse_runtime) -> None:
    """Assert the canonical fine noise state is what source moves consume.

    Cheap, assertion-only source-move precondition for coarse-sidecar runs
    (plan-2 §6.4): every container's sensitivity matrix must live on the
    run's FINE domain settings with the fine active shape. A violation is a
    programming error in a noise move's publication sequence — fail loudly;
    source moves never rebuild an unknown state or continue on coarse
    covariance. No-op unless a coarse sidecar runtime is active.

    Everything needed is derived from the runtime itself
    (``coarse_settings.fine_settings`` is the run's fine domain), so the
    caller only needs the shared ``acs`` and any move's ``coarse_runtime``.

    Called by :meth:`GFCombineMove._gf_precondition` before every non-noise
    sub-move when a wrapped noise move carries an active sidecar. Not reached
    on the 9mo / 12mo runs (``COARSE_GPU_MODE=off``, ``COARSE_Q=1``).

    Args:
        acs: The run's :class:`~lisatools.analysiscontainer.AnalysisContainerArray`.
        coarse_runtime: A ``CoarseWDMRuntime`` (``mode``, ``coarse_settings``) or
            ``None``; ``None`` or mode ``"off"`` makes this a no-op.

    Raises:
        AssertionError: A container's sensitivity matrix is on the coarse
            settings, is not on the run's fine settings, or does not end in the
            fine active shape.
    """
    runtime = coarse_runtime
    if runtime is None or getattr(runtime, "mode", "off") == "off":
        return
    coarse_settings = runtime.coarse_settings
    fine_settings = coarse_settings.fine_settings
    for w in range(len(acs)):
        sens_mat = acs[w].sens_mat
        basis = getattr(sens_mat, "basis_settings", None)
        if coarse_settings is not None and basis == coarse_settings:
            raise AssertionError(
                f"walker {w}: COARSE sensitivity matrix visible at a "
                "source-move boundary — the noise move failed to publish the "
                "fine state before returning."
            )
        if basis != fine_settings:
            raise AssertionError(
                f"walker {w}: sens_mat basis settings are not the run's fine "
                "domain settings (source moves must see only full fine state)."
            )
        got = tuple(sens_mat.data_shape)
        want = tuple(fine_settings.basis_shape_active)
        if got[-len(want):] != want:
            raise AssertionError(
                f"walker {w}: sens_mat data_shape {got} does not end in the "
                f"fine active shape {want}."
            )


@dataclasses.dataclass
class MoveBuildContext:
    """Runtime context handed to every ``Move.setup`` at materialization.

    Carries everything a move needs to build itself: the runtime
    :class:`~lisatools.globalfit.recipe.Recipe`, the engine/branch metadata,
    the built :class:`~lisatools.globalfit.run.GlobalFitSetup` (``curr``), the
    shared :class:`~lisatools.analysiscontainer.AnalysisContainerArray`
    (``acs``), the priors dict, the initial state, the variant-provided
    ``stock_moves`` name lookup, the sampler shape (``ntemps`` / ``nwalkers``),
    the multi-rank ``layout`` / ``fanout`` / ``rank`` and the head-side
    ``state_local`` slice (all ``None`` in a single-process run).

    Built by the variant's ``setup_recipe`` (e.g. ``all_sources.setup_recipe``) and
    passed to :meth:`Recipe.setup <lisatools.globalfit.recipe.Recipe.setup>`, which
    hands it to every :meth:`Move.setup`. ``__post_init__`` fills ``layout`` /
    ``fanout`` / ``rank`` from ``curr`` when not given and, on a multi-rank layout,
    slices ``state_local`` (coords / inds only, no sub-states) to this rank's
    walker block.

    Attributes:
        recipe: The runtime :class:`~lisatools.globalfit.recipe.Recipe`.
        engine_info: The engine / branch metadata (``branch_names``, ``ndims``, ...).
        curr: The built :class:`~lisatools.globalfit.run.GlobalFitSetup`.
        acs: The shared
            :class:`~lisatools.analysiscontainer.AnalysisContainerArray`.
        priors (dict): Branch name -> prior container.
        state: The initial :class:`~lisatools.globalfit.state.GFState`.
        stock_moves (dict): Stock-move name -> built runtime move (the base
            :meth:`Move.setup` looks names up here).
        ntemps (int or None): Engine temperature count (the engine runs the cold
            chain only, so 1 on stock runs).
        nwalkers (int or None): Engine walker count.
        layout: ``curr.rank_layout`` (walker-block layout) or ``None``.
        fanout: ``curr.fanout`` (the walker fan-out) or ``None``.
        rank (int or None): This process's rank (``curr.rank``).
        state_local: This rank's walker-block slice of ``state`` on a multi-rank
            layout, else ``None``.
    """

    recipe: typing.Any
    engine_info: typing.Any
    curr: typing.Any
    acs: typing.Any
    priors: dict
    state: typing.Any
    stock_moves: dict = dataclasses.field(default_factory=dict)
    ntemps: typing.Optional[int] = None
    nwalkers: typing.Optional[int] = None
    # multi-rank context (Plan 2 populates ``curr.rank_layout`` / ``curr.fanout``
    # / ``curr.rank``; ``state_local`` is the head-side walker slice a
    # builder reads when it must act on this rank's block only); documented in
    # the class docstring's Attributes section
    layout: typing.Any = None
    fanout: typing.Any = None
    rank: typing.Optional[int] = None
    state_local: typing.Any = None

    def __post_init__(self):
        curr = self.curr
        if self.layout is None:
            self.layout = getattr(curr, "rank_layout", None)
        if self.fanout is None:
            self.fanout = getattr(curr, "fanout", None)
        if self.rank is None:
            self.rank = getattr(curr, "rank", None)
        # The rank's OWN walker rows, filled here because the variants build
        # this ctx themselves and never pass ``state_local`` -- without it
        # WP3's block-gated builders would silently read the full state.
        # Sub-states are left out: a builder acting on its block reads
        # coords/inds, and slicing them is the head's job (WP4/WP5 payloads).
        if (
            self.state_local is None
            and self.state is not None
            and self.layout is not None
            and self.rank is not None
            and not self.layout.is_single()
        ):
            from ..communication.walkerslice import slice_state  # lazy: avoids an import cycle

            w0, w1 = self.layout.block_of(self.rank)
            self.state_local = slice_state(self.state, w0, w1, sub_states=[])


class Move:
    """Declarative, picklable recipe move (cheap-construct / heavy-setup).

    A move IS a proposal — one object per adjustment the fit makes each
    iteration. The base class resolves ``name`` against the variant's stock
    moves at materialization; there are exactly two ways to customize:

    * subclass ``Move`` and override :meth:`setup` — build and return the
      runtime eryn move there (or return ``None`` to mean "``self`` is the
      runtime move", in which case the subclass also implements ``propose``);
    * pass a plain ``fn(model, state) -> (new_state, accepted)`` to
      ``add_move`` — it is wrapped in a
      :class:`~lisatools.globalfit.moves.functionmove.FunctionMove`.

    The same stock name may sit in several stages (uniqueness is per stage);
    every stage resolves it to the SAME shared runtime object, which is why
    stage-dependent state (kind, name, iteration, cadence) lives on each stage's
    :class:`GFCombineMove`, not on the move.

    Args:
        name: Unique (per stage) move name. For the base class this is the
            stock-move name to look up (``"rj_prior"``, ``"psd_pe"``, ...).
        branch: Branch this move samples (validated against enabled branches
            at materialization).
        debug: Move-level debug override applied at materialization via
            ``set_debug`` — ``None`` keeps the move's env-resolved default,
            ``True``/``False`` force it, a dict passes options
            (``plot_dir``/``plot_walker``/``plot_leaf``/``plot_band``/
            ``every``). Wins over the stage-level ``Stage.debug``.
        every: Run this move only on every ``every``-th iteration of ITS stage
            (user ruling 2026-09-15). Default 1 (every iteration). The counter
            is the stage combine's, so the same shared runtime move stays
            uncadenced in a stage that declares 1; a skipped iteration leaves
            the branch as it was. Applied as the combine's ``move_every``.

    Raises:
        ValueError: ``name`` is not a non-empty string, or ``every < 1``.

    Attributes:
        name (str): See ``name``.
        branch (str or None): See ``branch``.
        debug (bool, dict or None): See ``debug``.
        every (int): See ``every``.

    Example:
        >>> from lisatools.globalfit import Move
        >>> Move("rj_prior_pe", branch="gb")
        Move('rj_prior_pe', branch='gb')
        >>> Move("mbh_pe", branch="mbh", every=10)
        Move('mbh_pe', branch='mbh', every=10)

        A custom move builds its runtime move in ``setup`` from the live context
        and is added like any other (``docs/stock-stages-and-moves.md``):

        .. code-block:: python

            from eryn.moves import StretchMove
            from lisatools.globalfit import Move

            class MyStretch(Move):
                def setup(self, ctx):
                    return StretchMove(live_dangerously=True)

            fit.add_move(MyStretch("my_stretch", branch="line"), stage="main")
    """

    def __init__(
        self,
        name: str,
        *,
        branch: typing.Optional[str] = None,
        debug: typing.Optional[typing.Union[bool, dict]] = None,
        every: int = 1,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Move needs a non-empty string name, got {name!r}.")
        # ``every``: run this move on every Nth iteration of ITS STAGE
        # (user ruling 2026-09-15: mbh/emri ride gb_search at a 1-in-10
        # cadence). Stage-local by construction — the counter lives on the
        # stage's GFCombineMove, so the SAME shared stock runtime move is
        # uncadenced in another stage that declares every=1. A skipped
        # iteration leaves the branch exactly as it was (still subtracted
        # at its current coords); cyclic deterministic thinning of a
        # kernel cycle preserves detailed balance.
        every = int(every)
        if every < 1:
            raise ValueError(f"Move {name!r}: every must be >= 1, got {every}.")
        self.name = name
        self.branch = branch
        self.debug = debug
        self.every = every
        self._runtime = None

    @property
    def is_stock(self) -> bool:
        """True when this move resolves through the stock-name lookup (base ``setup``)."""
        return type(self).setup is Move.setup

    def stock_dependencies(self) -> typing.List[str]:
        """Stock-move names this move's :meth:`setup` pulls from ``ctx``.

        Only meaningful on subclasses that override :meth:`setup`: a custom
        move resolves by its own code rather than by name, so the variant
        setup functions cannot tell from ``Recipe.stock_names()`` that it
        COMPOSES stock moves. Declare them here and they get built.

        Empty by default -- a custom move that builds everything itself needs
        nothing. Returning a name that the variant does not provide fails at
        materialization with the available list, which is the intended
        behaviour (better than a silent no-op).

        Returns:
            list[str]: Stock-move names; ``[]`` in the base class. (The
            production driver's ``JointMaxLogLSearch`` returns its wrapped
            names here.)
        """
        return []

    def setup(self, ctx: MoveBuildContext):
        """Build/return the runtime eryn move (``None`` -> ``self`` IS the runtime move).

        Default implementation: stock lookup — resolve
        ``ctx.stock_moves[self.name]``, the move the variant's setup function
        built under this name.

        Args:
            ctx: The build context.

        Returns:
            The runtime move (an eryn move, normally a :class:`GlobalFitMove`), or
            ``None`` from a subclass whose instance is itself the runtime move.

        Raises:
            ValueError: No stock move under ``self.name`` (the message lists the
                available names).
        """
        try:
            return ctx.stock_moves[self.name]
        except KeyError:
            raise ValueError(
                f"Move {self.name!r}: no stock move under this name (available: "
                f"{sorted(ctx.stock_moves)}). Subclass Move and override setup(ctx) "
                "to build a custom move, or add a plain fn(model, state) via add_move."
            ) from None

    def materialize(self, ctx: MoveBuildContext):
        """Framework entry point: run :meth:`setup` and record the runtime move.

        Called by ``Stage.setup``. A ``None`` from :meth:`setup` means ``self``.

        Returns:
            The runtime move, also stored on :attr:`runtime`.
        """
        runtime = self.setup(ctx)
        if runtime is None:
            runtime = self
        self._runtime = runtime
        return runtime

    @property
    def runtime(self):
        """The materialized runtime move (``None`` before materialization)."""
        return self._runtime

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime"] = None
        return state

    def __repr__(self):
        extra = f", branch={self.branch!r}" if self.branch else ""
        if getattr(self, "every", 1) != 1:
            extra += f", every={self.every}"
        return f"{type(self).__name__}({self.name!r}{extra})"


class _RuntimeMove(Move):
    """Private wrapper for a fully constructed runtime move added to a recipe.

    Created by the recipe's move coercion when a user passes an
    already-built eryn move to ``add_move``/``Stage(moves=[...])``. NOTE:
    constructed moves often hold arrays or device state — a fit configured
    with one may not be picklable (deliberately not dropped on pickle,
    matching the historical ``instance`` caveat).

    Args:
        move: The constructed eryn move; :meth:`setup` returns it unchanged.
        name (str, optional): Recipe name; defaults to ``move.name``.
        branch (str, optional): Branch the move samples.
        debug (bool or dict, optional): Debug override (see :class:`Move`).

    Raises:
        ValueError: Neither ``name`` nor ``move.name`` is set.
    """

    def __init__(self, move, *, name=None, branch=None, debug=None):
        name = name or getattr(move, "name", None)
        if not name:
            raise ValueError(
                "A constructed move needs a name: pass add_move(..., name=...) "
                "or give the move object a .name attribute."
            )
        super().__init__(name, branch=branch, debug=debug)
        self.move = move

    def setup(self, ctx: MoveBuildContext):
        """Return the wrapped constructed move unchanged."""
        return self.move


class GlobalFitMove:
    """Mix-in of the runtime global-fit moves (GB / VGB, noise, sources, combines).

    Subclasses combine this mix-in with an ``eryn`` move (the ridge-Gibbs moves
    ``gb_ridge_gibbs`` / ``vgb_ridge_gibbs`` are plain eryn moves without it).
    What it provides:

    * the sub-state helpers of the cold-chain storage model
      (``docs/conventions.md``): a branch's full tempered ensemble lives in
      ``state.sub_states[branch]`` and the main (engine) state carries only its
      cold row. :meth:`_work_branch` returns the tempered ensemble a move
      samples, :meth:`_sync_cold_row` mirrors the sub-state's row 0 back into
      the main state after a write, and :meth:`_check_substate_consistency`
      verifies the two agree (``GF_SUBSTATE_CHECK``);
    * :meth:`gf_serve`, the multi-rank fan-out hook (the base raises; the
      walker fan-out mixin and the GB moves override it);
    * :meth:`set_debug`, the uniform move / stage debug switch.

    The ``propose(model, state)`` contract a subclass implements is eryn's: return
    ``(new_state, accepted)`` with ``new_state`` the input state or a new
    :class:`~lisatools.globalfit.state.GFState` (never a plain eryn ``State``:
    :meth:`GFCombineMove.propose` raises when the sub-states are lost), every
    written branch's sub-state updated and its cold row mirrored.

    Args:
        *args: Ignored (accepted for cooperative construction).
        name: Required identifier used in logging and bookkeeping
            (``AssertionError`` when ``None``).
        iters_compare_likelihood (int): Stored cadence; ``-1`` (default) disables.
            Nothing in the package reads it.
        iters_reset_buffers (int): Stored cadence; ``-1`` (default) disables.
            Nothing in the package reads it.
        **kwargs: Ignored.

    Note:
        ``comm`` and ``gpus`` are vestigial (the move -> rank assignment was
        removed, parallel-resources plan P3); no move sets or reads them.
    """

    def _work_branch(self, state, branch_name=None):
        """The ensemble a move samples: the sub-state's tempered Branch view.

        The returned eryn ``Branch`` shares memory with the sub-state's
        coords/inds (the module owns its full ladder; the main state carries
        only the engine's cold chain). Falls back to the main state's branch
        when no tempered sub-state exists (e.g. simple-API branches).

        Args:
            state: A :class:`~lisatools.globalfit.state.GFState`.
            branch_name (str, optional): Branch; default ``self.branch_name``.

        Returns:
            eryn.state.Branch: For a tempered sub-state, coords shaped
            ``(ntemps_module, nwalkers, nleaves_max, ndim)``; else the main
            state's branch.
        """
        name = branch_name if branch_name is not None else self.branch_name
        sub_states = getattr(state, "sub_states", None) or {}
        sub = sub_states.get(name)
        if sub is not None and getattr(sub, "tempered_initialized", False):
            return sub.branch
        return state.branches[name]

    def _sync_cold_row(self, state, branch_name=None):
        """Mirror the sub-state's cold row into the main (engine) state.

        Delegates to :meth:`ModuleSubState.sync_cold_row
        <lisatools.globalfit.state.ModuleSubState.sync_cold_row>` for a
        tempered-initialized sub-state; a no-op otherwise. Every move that
        writes a branch's sub-state calls this before returning, so the
        stage-entry check (:meth:`_check_substate_consistency`) holds.

        Args:
            state: The :class:`~lisatools.globalfit.state.GFState` being returned.
            branch_name (str, optional): Branch; default ``self.branch_name``.
        """
        name = branch_name if branch_name is not None else self.branch_name
        sub_states = getattr(state, "sub_states", None) or {}
        sub = sub_states.get(name)
        if sub is not None and getattr(sub, "tempered_initialized", False):
            sub.sync_cold_row(state, name)

    def _check_substate_consistency(self, state, branch_names=None):
        """Verify main-state cold rows match each sub-state's row 0.

        Delegates to :meth:`ModuleSubState.check_cold_row` for every
        tempered-initialized sub-state (or just ``branch_names``). No-op when
        ``GF_SUBSTATE_CHECK=0`` or the state carries no sub-states. Called at
        every stage entry (:meth:`GFCombineMove.propose`) and at the top of the
        noise and source moves' proposes.

        Args:
            state: The :class:`~lisatools.globalfit.state.GFState` to check.
            branch_names (list[str], optional): Branches to check; default all.

        Raises:
            ValueError: From ``check_cold_row``, when a branch's main-state cold
                row and sub-state row 0 differ.
        """
        if not substate_check_enabled():
            return
        sub_states = getattr(state, "sub_states", None)
        if not sub_states:
            return
        names = branch_names if branch_names is not None else sub_states.keys()
        for name in names:
            sub = sub_states.get(name)
            if sub is None or not getattr(sub, "tempered_initialized", False):
                continue
            sub.check_cold_row(state, name)

    def gf_serve(self, op, payload, clock, model):
        """Serve one fan-out command on a computation rank.

        The multi-rank moves override this (GB: ``gb_run_proposal`` /
        ``gb_run_tempering`` / ``gb_finish``; addremove and PSD: ``propose``).
        ``model`` is this rank's ``GlobalFitInfo`` (local ACA, map, rank RNG).

        Args:
            op (str): The fan-out command name.
            payload: The command payload shipped by the head.
            clock: The head's clock values for this command.
            model: This rank's ``GlobalFitInfo``.

        Raises:
            NotImplementedError: Always, in the base (a move without fan-out
                support; ``run.py`` refuses such moves on a multi-rank run).
        """
        raise NotImplementedError(
            f"move {getattr(self, 'gf_move_name', getattr(self, 'name', type(self).__name__))!r} "
            f"does not serve fan-out command {op!r}"
        )

    def __init__(
        self,
        *args,
        name=None,
        iters_compare_likelihood=-1,
        iters_reset_buffers=-1,
        **kwargs,
    ):
        assert name is not None
        self.name = name

        # should be inside moves so you know where it fails
        # rather than in the engine
        self.iters_compare_likelihood = iters_compare_likelihood
        self.iters_reset_buffers = iters_reset_buffers

    @property
    def iters_reset_buffers(self) -> int:
        """Iteration cadence at which auxiliary buffers are reset."""
        return self._iters_reset_buffers

    @iters_reset_buffers.setter
    def iters_reset_buffers(self, iters_reset_buffers: int):
        assert isinstance(iters_reset_buffers, int)
        self._iters_reset_buffers = iters_reset_buffers

    @property
    def iters_compare_likelihood(self) -> int:
        """Iteration cadence at which inter-rank likelihoods are cross-checked."""
        return self._iters_compare_likelihood

    @iters_compare_likelihood.setter
    def iters_compare_likelihood(self, iters_compare_likelihood: int):
        assert isinstance(iters_compare_likelihood, int)
        self._iters_compare_likelihood = iters_compare_likelihood

    @property
    def comm(self):
        """Vestigial MPI communicator slot (``AttributeError`` until set; nothing sets it)."""
        return self._comm

    @comm.setter
    def comm(self, comm):
        # if hasattr(self, "update_comm_special") and self.update_comm_special and hasattr(self, "moves"):
        #     ranks_needed = []
        #     for move in self.moves:
        #         if isinstance(move, tuple) or isinstance(move, list):
        #             assert len(move) == 2
        #             move = move[0]

        #         move.comm = comm

        self._comm = comm

    # NOTE: the ranks/assign_ranks/ranks_needed move->rank machinery was
    # removed with the dead dispatch (parallel-resources plan P3).

    def set_debug(
        self,
        enabled: bool = True,
        *,
        plot_dir=None,
        plot_walker=None,
        plot_leaf=None,
        plot_band=None,
        every=None,
        log=None,
    ) -> None:
        """Enable/disable this move's debug instrumentation (move/stage level).

        Uniform across the debug-capable moves — the GB special-stretch move
        (band-indexed: ``plot_band``) and ``ResidualAddOneRemoveOneMove``
        (leaf-indexed: ``plot_leaf``). Options left ``None`` keep whatever the
        move already resolved from its env vars (``GB_DEBUG`` /
        ``{BRANCH}_DEBUG``); precedence is move-spec > stage-spec > env.

        Applied by ``Stage.setup`` from the ``debug`` field on a
        :class:`Move`/``Stage``, or callable directly on a built move.
        Moves without a matching hook simply carry the flag inertly
        (``PSDMove`` has its own ``{PSD,GALFOR,SGWB}_DEBUG`` plot hook, which
        reads ``debug`` / ``debug_plot_dir`` / ``debug_plot_walker`` /
        ``debug_every``).

        Args:
            enabled: Turn debug on / off. Default ``True``.
            plot_dir (str, optional): Plot folder; when debug ends up on with
                no folder, ``./gf_output/debug/`` is used.
            plot_walker (int, optional): Walker to plot.
            plot_leaf (int, optional): Leaf to plot (moves with
                ``debug_plot_leaf`` only).
            plot_band (int, optional): Band to plot (moves with
                ``debug_plot_band`` only).
            every (int, optional): Plot every N-th propose (floored at 1;
                moves with ``debug_every`` only).
            log (bool, optional): Log-scale plots (moves with ``debug_log``
                only).
        """
        self.debug = bool(enabled)
        if plot_dir is not None:
            self.debug_plot_dir = str(plot_dir)
        if plot_walker is not None:
            self.debug_plot_walker = int(plot_walker)
        if plot_leaf is not None and hasattr(self, "debug_plot_leaf"):
            self.debug_plot_leaf = int(plot_leaf)
        if plot_band is not None and hasattr(self, "debug_plot_band"):
            self.debug_plot_band = int(plot_band)
        if every is not None and hasattr(self, "debug_every"):
            self.debug_every = max(1, int(every))
        if log is not None and hasattr(self, "debug_log"):
            self.debug_log = bool(log)
        if self.debug and not getattr(self, "debug_plot_dir", None):
            self.debug_plot_dir = "./gf_output/debug/"

    @property
    def gpus(self):
        """Vestigial list of GPU indices assigned to this move (default empty; nothing sets it)."""
        if not hasattr(self, "_gpus"):
            return []
        return self._gpus

    @gpus.setter
    def gpus(self, gpus):
        assert isinstance(gpus, list)
        for tmp in gpus:
            assert isinstance(tmp, int)

        self._gpus = gpus


def _gf_rss_mb() -> float:
    """Peak resident set size of this process in MB (monotone; darwin=bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if sys.platform == "darwin" else ru / 1e3


def _gf_gpu_pool_mb():
    """(used, total) MB of the default cupy memory pool, or (None, None).

    Only queries cupy when the run has already imported it — a CPU run must
    never trigger a cupy/CUDA initialization from instrumentation.
    """
    if "cupy" not in sys.modules:
        return None, None
    try:
        pool = sys.modules["cupy"].get_default_memory_pool()
        return pool.used_bytes() / 1e6, pool.total_bytes() / 1e6
    except Exception:
        return None, None


def substate_check_enabled() -> bool:
    """Whether the cold-chain sub-state consistency check is armed.

    Always on; disable with ``GF_SUBSTATE_CHECK=0`` (read at call time).

    Returns:
        bool: ``False`` only when ``GF_SUBSTATE_CHECK`` is exactly ``"0"``.
    """
    return os.environ.get("GF_SUBSTATE_CHECK", "1") != "0"


def pe_rj_fstat_residual_split() -> float:
    """Resolve ``GB_PE_RJ_FSTAT_RESIDUAL_SPLIT`` (read at call time).

    The share of the PE draw's F-stat fraction that goes to
    ``rj_fstat_pe_residual``; ``rj_fstat_pe`` (the GB-free galaxy grid) takes
    the rest. Default ``0.5`` (user ruling 2026-10-12: "two fstat moves that
    will split their share of the total moves"); the 9mo / 12mo launchers set
    ``0.9``. Unset or blank means the default.

    The ONE resolver for both consumers: the PE draw
    (:meth:`GFCombineMove._pe_rj_draw_one_config`) and the driver's stage
    composition (``run_combined_staged.py``), which leaves a move with no
    share (a split of exactly 0 or 1) out of the PE stage altogether -- so
    the draw and the composition can never disagree about which F-stat moves
    a PE stage carries.

    Returns:
        float: The split, in ``[0, 1]``.

    Raises:
        ValueError: The value is not finite or lies outside ``[0, 1]``.
    """
    raw = os.environ.get("GB_PE_RJ_FSTAT_RESIDUAL_SPLIT", "").strip() or "0.5"
    s = float(raw)
    if not np.isfinite(s) or not (0.0 <= s <= 1.0):
        raise ValueError(
            f"GB_PE_RJ_FSTAT_RESIDUAL_SPLIT={raw!r} must lie in [0, 1] (the "
            "share of GB_PE_RJ_FSTAT_FRACTION that rj_fstat_pe_residual gets; "
            "rj_fstat_pe takes the rest).")
    return s


class GFCombineMove(CombineMove, GlobalFitMove):
    """The per-stage combine move: runs one recipe stage's moves each iteration.

    ``Stage.setup`` wraps each stage's runtime moves in ONE of these (its
    ``combine_kwargs`` are this constructor's keywords), stamps
    ``gf_stage_name`` / ``gf_stage_kind`` / ``gf_is_stage_combine`` (true) on it,
    and hands it to the stage's ``RecipeStep``; eryn then calls :meth:`propose`
    exactly once per global-fit iteration. Nested combines (e.g. the driver's
    ``JointMaxLogLSearch`` -> :class:`MaxLogLCombineMove`) are wrapped moves like
    any other.

    The ``propose(model, state)`` contract: mint the iteration counter
    (``gf_iteration``, stage combine only), check every branch's cold row
    against its sub-state (``GF_SUBSTATE_CHECK``), run the wrapped moves through
    :meth:`_propose_moves`, and refuse a returned plain eryn ``State``. Before each
    sub-move :meth:`_gf_precondition` re-stamps the stage kind, name and
    iteration on it (stock moves are shared between stages). Which moves run,
    in this order of precedence:

    * search LEGS (``leg_ends``; ``GB_SEARCH_LEGS=1`` sets it to ``"auto"`` on
      the production ``gb_search_*`` stages): one propose runs from the leg
      cursor to the next ``in_model*`` move and the row is saved after it
      (:meth:`_propose_leg`, :mod:`lisatools.globalfit.legs`);
    * the PE draw-one RJ plan (``GB_PE_RJ_DRAW_ONE=1``, kind ``"pe"`` stages:
      the production ``full_pe``): one of the RJ members per iteration
      (:meth:`_pe_rj_draw_one_plan`), every other move once in fixed order;
    * the per-move cadence (``move_every``, from each :attr:`Move.every`);
    * ``weighted_cycle`` / ``random_choice`` (below);
    * the fixed sequential order: eryn's ``CombineMove.propose``, or
      :meth:`_run_sequence` when timing, a mid-iteration checkpoint or a coarse
      sidecar is armed (``GF_MOVE_TIMING=1`` on the production runs).

    ``GF_MOVE_TIMING=1`` arms per-move instrumentation in :meth:`propose`:
    one ``[GF_TIMING]`` line per wrapped move per iteration (wall time, peak
    RSS, cupy pool) plus a ``move=__total__`` per-iteration line.  The env var
    is read at call time — construction, pickling, and deepcopy behavior are
    untouched, and with the variable unset the eryn code path runs unchanged.
    ``GF_MOVE_TIMING_SYNC=1`` additionally synchronizes the current cupy
    stream at each move boundary so asynchronous kernel time is attributed to
    the move that launched it.

    ``random_choice=True`` draws ONE wrapped move per step (uniformly, with
    replacement, from ``model.random``) instead of running all of them in a
    fixed order -- the stock eryn move-selection semantics. Default ``False``
    keeps eryn's run-them-all behaviour, so existing stages are untouched.
    Useful for a PE stage where the moves are alternative proposals for the
    same state rather than a pipeline that has to run start to finish.

    ``weighted_cycle=True`` (user ruling 2026-08-26, the GB PE storage
    fix): keep the SEARCH-style single-propose cycle -- every drawn
    sub-move executes inside THIS one propose call, so the backend stores
    ONE row per iteration -- but draw the cycle's composition each
    iteration: ``len(moves)`` draws WITH replacement from the wrapped
    moves, weighted by ``move_weights`` (default equal weights). The draw
    comes from ``model.random`` so a fixed seed reproduces the run. When
    both flags are set, ``weighted_cycle`` governs. Search stages pass
    neither flag and keep the fixed sequential order. (The production driver
    turns ``weighted_cycle`` on for ``full_pe`` via ``FULL_PE_WEIGHTED_CYCLE``,
    default ``1``, but with ``GB_PE_RJ_DRAW_ONE=1`` the draw-one plan returns
    first, so the weighted cycle does not run there.)

    Args:
        *args: Forwarded to :class:`eryn.moves.CombineMove`; the first is the list
            of wrapped moves (or ``(move, weight)`` tuples).
        random_choice: See above. Default ``False``.
        weighted_cycle: See above. Default ``False``.
        move_weights (array-like, optional): Weights of the ``weighted_cycle``
            draw, one per wrapped move (normalized; ``None`` = equal).
        move_every (list[int], optional): Per-move cadence aligned with the
            wrapped moves; all-ones collapses to ``None``. Fixed sequential path
            and legs only.
        leg_ends (str or list[str], optional): ``"auto"`` (every ``in_model*``
            move ends a leg) or explicit move names; ``None`` = one propose runs
            the whole cycle. Incompatible with ``random_choice`` /
            ``weighted_cycle``.
        **kwargs: Forwarded to :class:`eryn.moves.CombineMove` (``verbose``,
            ``share_temperature_control``, eryn ``Move`` keywords).

    Raises:
        ValueError: ``leg_ends`` or ``move_every`` combined with
            ``random_choice`` / ``weighted_cycle``, or a malformed
            ``move_weights`` / ``move_every``.

    Attributes:
        gf_legs: The :class:`~lisatools.globalfit.legs.LegCursor`, or ``None``.
        gf_move_every (list[int] or None): The validated cadence.
        gf_iteration (int): Iteration counter, advanced by the stage combine's
            :meth:`propose` (starts at 0 on the first call).

    Note:
        Cross-repo dependency on ``eryn.moves.CombineMove.accepted``. Its getter
        reads an ``_accepted`` attribute the combine never sets (the setter copies
        the array onto every wrapped move instead), so reading ``accepted`` on a
        combine raises ``AttributeError``. ``Stage.setup`` in
        :mod:`lisatools.globalfit.recipe` relies on exactly that: it tests
        ``hasattr`` on the new combine and then sets the counters. The same
        getter is read by :meth:`_propose_moves` when every wrapped move is
        cadenced off in one iteration (it returns zeros shaped like
        ``self.accepted``), so that branch raises ``AttributeError`` instead of
        returning. It is unreachable on the 9mo / 12mo runs (every ``every`` is
        1, so ``move_every`` is ``None``). Changing eryn's getter changes both.

    Example:
        Built by ``Stage.setup``; a stage's ``combine_kwargs`` are these keywords:

        .. code-block:: python

            from lisatools.globalfit import Move, Stage

            fit.recipe.add_stage(Stage(
                "my_pe", kind="pe",
                moves=[Move("psd_pe", branch="psd"), Move("mbh_pe", branch="mbh", every=5)],
                combine_kwargs=dict(random_choice=False, verbose=False),
            ))
    """

    update_comm_special = True

    def __init__(
        self,
        *args,
        random_choice: bool = False,
        weighted_cycle: bool = False,
        move_weights=None,
        move_every=None,
        leg_ends=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.random_choice = bool(random_choice)
        self.weighted_cycle = bool(weighted_cycle)
        self.move_weights = self._validate_move_weights(move_weights)
        # ---- search LEGS (user design 2026-09-30) ------------------------
        # ``leg_ends`` = "auto" (every in_model* move) or an explicit list of
        # move names; None = the historical one-propose-runs-all cycle. With
        # legs, ONE propose runs from the cursor through the next leg-ender
        # and the state comes back tagged with that name, so eryn's save
        # after it is a row per leg. See lisatools.globalfit.legs.
        self.gf_legs = None
        if leg_ends is not None:
            if self.random_choice or self.weighted_cycle:
                raise ValueError(
                    "leg_ends needs the fixed sequential path (random_choice / "
                    "weighted_cycle draw their own cycle).")
            from ..legs import LegCursor, leg_ends_from_names

            names = self.gf_move_names()
            self.gf_legs = LegCursor(names, leg_ends_from_names(names, leg_ends))
        # Per-sub-move iteration cadence (aligned with ``moves``): sub-move
        # i runs only when (stage propose count) % move_every[i] == 0. The
        # counter is THIS combine's, i.e. stage-local — the same shared
        # stock runtime move stays uncadenced in a stage that declares 1.
        # Only the fixed sequential path supports it (a drawn cycle has no
        # per-iteration slot to skip).
        self.gf_move_every = self._validate_move_every(move_every)
        self._gf_cadence_idx = 0

    def _validate_move_every(self, move_every):
        """Normalize ``move_every`` to a per-move int list, or ``None`` for all-ones.

        Raises ``ValueError`` on a length mismatch, an entry below 1, or a
        cadence > 1 together with ``random_choice`` / ``weighted_cycle``.
        """
        if move_every is None:
            return None
        ev = [int(e) for e in move_every]
        if len(ev) != len(self.moves):
            raise ValueError(
                f"move_every has length {len(ev)}; expected "
                f"{len(self.moves)} to match the wrapped moves."
            )
        if any(e < 1 for e in ev):
            raise ValueError("move_every entries must be >= 1.")
        if all(e == 1 for e in ev):
            return None
        if self.random_choice or self.weighted_cycle:
            raise ValueError(
                "move_every > 1 is only supported on the fixed sequential "
                "path (random_choice/weighted_cycle draw their own cycle)."
            )
        return ev

    def _gf_cadence_plan(self, moves):
        """``(due_moves, skipped_names)`` for this propose under the cadence.

        Advances the stage-local counter once per call. With no cadence
        configured this is a passthrough that never touches the counter.
        """
        every = self.gf_move_every
        if every is None:
            return moves, None
        idx = self._gf_cadence_idx
        self._gf_cadence_idx = idx + 1
        by_id = {}
        for m, e in zip(self.moves, every):
            mm = m[0] if isinstance(m, tuple) else m
            by_id[id(mm)] = e
        due, skipped = [], []
        for m in moves:
            mm = m[0] if isinstance(m, tuple) else m
            if idx % by_id.get(id(mm), 1) == 0:
                due.append(m)
            else:
                skipped.append(
                    getattr(mm, "gf_move_name", type(mm).__name__))
        return due, skipped

    def _validate_move_weights(self, move_weights):
        """Normalize ``move_weights`` to a probability vector (or None).

        None means equal weights (resolved at draw time so the move count
        is always current). Anything else must match ``len(self.moves)``,
        be non-negative, and carry positive total mass.
        """
        if move_weights is None:
            return None
        w = np.asarray(move_weights, dtype=float)
        if w.shape != (len(self.moves),):
            raise ValueError(
                f"move_weights has shape {w.shape}; expected "
                f"({len(self.moves)},) to match the wrapped moves."
            )
        if np.any(w < 0.0) or not np.isfinite(w).all():
            raise ValueError("move_weights must be finite and non-negative.")
        total = w.sum()
        if total <= 0.0:
            raise ValueError("move_weights must have positive total mass.")
        return w / total

    def propose(self, model, state):
        """Run this stage's moves for one global-fit iteration.

        On the stage combine (``gf_is_stage_combine``) advance ``gf_iteration``;
        when the state carries a module sub-state, check every branch's cold
        row first (:meth:`GlobalFitMove._check_substate_consistency`); then
        :meth:`_propose_moves` (see the class docstring for the dispatch order).

        Args:
            model (eryn.model.Model): The eryn model (``analysis_container_arr``,
                ``random``, ...).
            state (GFState): The current :class:`~lisatools.globalfit.state.GFState`.

        Returns:
            tuple: ``(state, accepted)``; ``accepted`` is the element-wise sum of
            the sub-moves' acceptance arrays (``None`` when nothing ran in a leg).

        Raises:
            ValueError: A wrapped move returned a state without ``sub_states``
                although the input carried them (a plain eryn ``State``), or a
                sub-state check failed.
        """
        # Simple-API branches (sub_states all None, e.g. erebor.blank) temper
        # on the engine ladder through plain eryn moves, which legitimately
        # rebuild the State -- only arm the guards when a real module
        # sub-state is present.
        # ---- the run's ITERATION counter ---------------------------------
        # eryn calls the stage combine's propose exactly once per global-fit
        # iteration, so this is the one place in the move tree that can count
        # iterations. Everything below it is proposed a stage-dependent NUMBER
        # of times per iteration (gb_search fires 3 GB-branch moves, full_pe
        # ~1/6 of the time), which is why a per-propose counter cannot stand
        # in for one -- see GBSpecialRJFStatGridMove._fstat_clock.
        # Only the STAGE combine mints the value; nested combines inherit it
        # through _prepare_child, the same way gf_stage_kind propagates.
        if getattr(self, "gf_is_stage_combine", False):
            self.gf_iteration = int(getattr(self, "gf_iteration", -1)) + 1

        _subs = getattr(state, "sub_states", None) or {}
        had_sub_states = any(sub is not None for sub in _subs.values())
        if had_sub_states:
            # stage-entry guard: every branch's cold row must agree between
            # the main state and its sub-state (GF_SUBSTATE_CHECK=0 disables)
            self._check_substate_consistency(state)

        state, accepted = self._propose_moves(model, state)

        if had_sub_states:
            if getattr(state, "sub_states", None) is None:
                raise ValueError(
                    f"stage {getattr(self, 'gf_stage_name', '?')!r}: a wrapped "
                    "move returned a plain eryn State, destroying the "
                    "GFState sub-states. Every global-fit move must return "
                    "the input state or a new GFState (e.g. "
                    "GFState(state, copy=True))."
                )
        return state, accepted

    def _propose_moves_once(self, model, state):
        """One pass over the wrapped moves (the plain GFCombineMove body)."""
        return GFCombineMove._propose_moves(self, model, state)

    def _gf_sidecar_runtime_lookup(self):
        """The active coarse sidecar runtime carried by a wrapped noise move.

        Cached per instance; None when no wrapped move carries one (the
        overwhelmingly common case, and every pre-plan-2 run).
        """
        if not hasattr(self, "_gf_sidecar_runtime"):
            runtime = None
            for m in self.moves:
                m = m[0] if isinstance(m, tuple) else m
                cand = getattr(m, "coarse_runtime", None)
                if cand is not None and getattr(cand, "mode", "off") != "off":
                    runtime = cand
                    break
            self._gf_sidecar_runtime = runtime
        return self._gf_sidecar_runtime

    def _gf_precondition(self, move, model):
        """Per-sub-move preamble: stamp the stage kind, then guard the handoff.

        (1) Propagate this stage's ``kind`` onto the sub-move. Stock moves are
        shared BY NAME, so one PSDMove instance serves both a search stage and
        the PE stage -- the kind cannot be a static attribute, and is
        re-stamped here immediately before each propose. Nested combines
        inherit it and pass it down the same way, so a move always sees the
        stage actually running it. The stage name and the iteration counter are
        re-stamped the same way (attribute errors on exotic moves are ignored).

        (2) Coarse-sidecar guard (plan-2 §6.4), before every SOURCE sub-move.

        Noise moves (they expose ``NOISE_BRANCHES``) own the publication and
        are exempt; everyone else must observe full fine state. Assertion
        cost is a per-walker attribute compare — negligible next to any
        source proposal.
        """
        kind = getattr(self, "gf_stage_kind", None)
        if kind is not None:
            try:
                move.gf_stage_kind = kind
            except AttributeError:  # exotic move objects: never fatal
                pass
        # ... and the stage NAME, for the same reason and with the same
        # lifetime. ``Stage.setup`` stamps it once at materialization, but
        # every stage is materialized up front and the GB moves are SHARED
        # objects -- so the static stamp is whichever stage was built last.
        # Under the v9 three-stage search that means every `[GB_STAGE
        # rj_fstat_search]` / `[MAXLOGL]` line emitted during gb_search_1
        # read "gb_search_3", which is precisely the attribution the
        # restructure needs its diagnostics to get right. Re-stamped per
        # propose, exactly as gf_stage_kind is.
        name = getattr(self, "gf_stage_name", None)
        if name is not None:
            try:
                move.gf_stage_name = name
            except AttributeError:  # exotic move objects: never fatal
                pass
        it = getattr(self, "gf_iteration", None)
        if it is not None:
            try:
                move.gf_iteration = it
            except AttributeError:  # exotic move objects: never fatal
                pass
        runtime = self._gf_sidecar_runtime_lookup()
        if runtime is None or hasattr(move, "NOISE_BRANCHES"):
            return
        acs = getattr(model, "analysis_container_arr", None)
        if acs is not None:
            ensure_fine_noise_covariance_current(acs, runtime)

    # ---- PE-only exclusive RJ draw (user ruling 2026-09-10) ----------------
    #: The PE RJ proposals that become mutually exclusive per iteration under
    #: ``GB_PE_RJ_DRAW_ONE=1`` (matched on the declarative move name the
    #: Stage stamps as ``gf_move_name``). ``rj_warm_pe`` joined the draw on
    #: 2026-10-02 (user ruling: "put the warm proposal in with the other two
    #: rj moves. Draw them at 0.45 prior, 0.45 fstat, 0.1 warm start"); a
    #: stage without it draws between the first two as before.
    #: ``rj_fstat_pe_residual`` joined on 2026-10-12 (user ruling: "two fstat
    #: moves that will split their share of the total moves"): the F-stat
    #: share ``GB_PE_RJ_FSTAT_FRACTION`` is split between the GB-free galaxy
    #: move and the live-residual move by ``GB_PE_RJ_FSTAT_RESIDUAL_SPLIT``.
    #: Listed right after ``rj_fstat_pe`` so a stage WITHOUT it keeps the
    #: pre-2026-10-12 member order, weights and therefore draws bit-for-bit.
    #: ``rj_para_search_pe`` (the GB parallel residual search, user ruling
    #: 2026-10-16) joins LAST with its own share ``GB_PE_RJ_PARA_FRACTION``
    #: (default 0): at 0, or on a stage without it, the member tuple and the
    #: weights are exactly today's, so every draw is bit-identical.
    _PE_RJ_DRAW_ONE_MEMBERS = ("rj_fstat_pe", "rj_fstat_pe_residual",
                               "rj_prior_pe", "rj_warm_pe", "rj_para_search_pe")
    #: members that join the draw only when given probability mass (at 0 they
    #: stay outside it and run every iteration in their fixed slot)
    _PE_RJ_OPTIONAL_MEMBERS = ("rj_warm_pe", "rj_para_search_pe")
    #: the F-stat members of the draw: a stage carrying only one of them
    #: hands it the whole F-stat share (the total stays the launcher's)
    _PE_RJ_FSTAT_MEMBERS = ("rj_fstat_pe", "rj_fstat_pe_residual")

    @staticmethod
    def _pe_rj_draw_one_config():
        """``(enabled, {member: P})`` from the env, read at CALL time.

        ``GB_PE_RJ_DRAW_ONE`` (default ``0``) switches the mode on;
        ``GB_PE_RJ_FSTAT_FRACTION`` (default ``0.8``) is the TOTAL F-stat
        probability per iteration, split between ``rj_fstat_pe`` (the
        GB-free galaxy grid) and ``rj_fstat_pe_residual`` (the live-residual
        grid) by ``GB_PE_RJ_FSTAT_RESIDUAL_SPLIT`` (the residual move's
        share, default ``0.5``; see :func:`pe_rj_fstat_residual_split`);
        ``GB_PE_RJ_WARM_FRACTION`` (default ``0``) is P(rj_warm_pe),
        ``GB_PE_RJ_PARA_FRACTION`` (default ``0``) is P(rj_para_search_pe), and
        ``rj_prior_pe`` takes the remainder. Fractions in [0, 1], the three
        absolute ones summing to at most 1. Read per call like
        ``GF_MOVE_TIMING`` so a resumed run takes the runbook's value with no
        code change. Production (launchers 0.45 / 0.9 / 0.4 / 0): P(rj_fstat_pe)
        0.045, P(rj_fstat_pe_residual) 0.405, P(rj_prior_pe) 0.15,
        P(rj_warm_pe) 0.4.

        Returns:
            tuple[bool, dict or None]: ``(False, None)`` when the mode is off,
            else ``(True, {member name: probability})`` for all five members.

        Raises:
            ValueError: A fraction outside ``[0, 1]`` or non-finite, or the three
                absolute fractions summing above 1 (and see
                :func:`pe_rj_fstat_residual_split`).
        """
        if os.environ.get("GB_PE_RJ_DRAW_ONE", "0") != "1":
            return False, None
        raw_f = os.environ.get("GB_PE_RJ_FSTAT_FRACTION", "0.8")
        raw_w = os.environ.get("GB_PE_RJ_WARM_FRACTION", "0")
        raw_p = os.environ.get("GB_PE_RJ_PARA_FRACTION", "0")
        p_f, p_w, p_p = float(raw_f), float(raw_w), float(raw_p)
        for name, raw, p in (("GB_PE_RJ_FSTAT_FRACTION", raw_f, p_f),
                             ("GB_PE_RJ_WARM_FRACTION", raw_w, p_w),
                             ("GB_PE_RJ_PARA_FRACTION", raw_p, p_p)):
            if not np.isfinite(p) or not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"{name}={raw!r} must lie in [0, 1] (a per-PE-iteration "
                    "draw probability).")
        if p_f + p_w + p_p > 1.0 + 1e-12:
            raise ValueError(
                f"GB_PE_RJ_FSTAT_FRACTION={raw_f!r} + GB_PE_RJ_WARM_FRACTION="
                f"{raw_w!r} + GB_PE_RJ_PARA_FRACTION={raw_p!r} exceed 1; "
                "rj_prior_pe takes the remainder.")
        s = pe_rj_fstat_residual_split()
        return True, {"rj_fstat_pe": p_f * (1.0 - s),
                      "rj_fstat_pe_residual": p_f * s,
                      "rj_prior_pe": 1.0 - p_f - p_w - p_p,
                      "rj_warm_pe": p_w,
                      "rj_para_search_pe": p_p}

    def _pe_rj_draw_one_plan(self, model):
        """Sub-move sequence for THIS propose under the PE-only exclusive
        RJ draw, or ``None`` when the mode does not apply.

        In a stage of kind ``"pe"`` whose wrapped moves include
        ``rj_prior_pe`` and at least one F-stat member (``rj_fstat_pe``,
        ``rj_fstat_pe_residual``), exactly ONE draw member runs per iteration,
        drawn from ``model.random`` with the probabilities of
        :meth:`_pe_rj_draw_one_config` (so a fixed seed reproduces the
        draws). ``rj_warm_pe`` / ``rj_para_search_pe`` are members only when
        given probability mass. Every other wrapped move runs once per
        iteration in its fixed order; the drawn RJ move takes the slot of
        the first member present (``gb_ridge_gibbs`` still follows). Search /
        rj stages are never touched (kind gate), nor is a PE stage without
        the required pair -- both fall through to the stage's configured mode
        (``weighted_cycle`` / ``random_choice`` / sequential). Logs one
        ``[GB_PE_RJ_DRAW_ONE]`` line per stage and, with ``GF_MOVE_TIMING=1``,
        a ``[GF_TIMING] ... pe_rj_draw_one`` line per draw.

        The original two-member form (``rj_fstat_pe`` vs ``rj_prior_pe``, with
        P(rj_fstat_pe) = ``GB_PE_RJ_FSTAT_FRACTION``) is what a stage carrying
        only those two still draws with the warm / para fractions at 0 (a
        fraction given to a member the stage lacks is redistributed over the
        others in proportion).

        TWO F-STAT MOVES (user ruling 2026-10-12): ``rj_fstat_pe_residual``
        is a member beside ``rj_fstat_pe`` and either one stands in for the
        F-stat half of the required pair. With both in the stage they split
        ``GB_PE_RJ_FSTAT_FRACTION`` by ``GB_PE_RJ_FSTAT_RESIDUAL_SPLIT``; with
        one, it takes the whole F-stat fraction. The drawn move still takes
        the slot of the first member in the stage's order.

        Why: with the PE cycle drawn as ``len(moves)`` weighted draws, both
        RJ moves ran ~once per iteration (3-month job 469: 27 + 27 firings
        over 27 PE iterations, ~280 s each = the whole ~9.4 min iteration).
        One RJ move per stored row at 0.8/0.2 halves the PE iteration and
        tilts the birth mix toward the F-stat grid, without changing what
        either move does when it runs.
        """
        enabled, probs = self._pe_rj_draw_one_config()
        if not enabled or getattr(self, "gf_stage_kind", None) != "pe":
            return None
        moves = [m[0] if isinstance(m, tuple) else m for m in self.moves]
        names = [getattr(m, "gf_move_name", type(m).__name__) for m in moves]
        # an F-stat move + rj_prior_pe is required (as before, with either
        # F-stat move -- or both -- counting since 2026-10-12). rj_warm_pe
        # joins the draw only when it is given probability mass
        # (GB_PE_RJ_WARM_FRACTION > 0) AND the stage carries it; at 0 it stays
        # outside the draw and runs every iteration in its fixed slot exactly
        # as before. A warm fraction set on a stage without rj_warm_pe is
        # redistributed over the others in their configured proportion.
        fstat_here = [n for n in self._PE_RJ_FSTAT_MEMBERS if n in names]
        if "rj_prior_pe" not in names or not fstat_here:
            return None
        # ONE F-stat move in the stage takes the WHOLE F-stat share, so the
        # total stays GB_PE_RJ_FSTAT_FRACTION and prior / warm keep theirs (a
        # pre-2026-10-12 composition draws exactly as it did). With both, an
        # F-stat move at zero mass (a split of 0 or 1) stays a member, is
        # never drawn and so never runs: it does NOT fall back to running every
        # iteration the way a zero-mass warm move does.
        if len(fstat_here) == 1:
            probs = dict(probs)
            probs[fstat_here[0]] = sum(probs[n] for n in self._PE_RJ_FSTAT_MEMBERS)
        members = tuple(
            n for n in self._PE_RJ_DRAW_ONE_MEMBERS
            if n in names and (n not in self._PE_RJ_OPTIONAL_MEMBERS
                               or probs.get(n, 0.0) > 0.0))
        weights = np.array([probs[n] for n in members], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(members))
        weights = weights / weights.sum()
        u = float(model.random.uniform())
        pick = members[int(min(np.searchsorted(np.cumsum(weights), u, side="right"),
                               len(members) - 1))]
        plan = []
        placed = False
        for m, n in zip(moves, names):
            if n in members:
                if not placed:
                    plan.append(moves[names.index(pick)])
                    placed = True
                continue
            plan.append(m)
        stage = getattr(self, "gf_stage_name", "?")
        if not getattr(self, "_pe_rj_draw_one_announced", False):
            logger.info(
                "[GB_PE_RJ_DRAW_ONE] stage=%s: ONE of %s per iteration, %s, drawn "
                "from the sampler RNG; the other %d wrapped move(s) run once per "
                "iteration in fixed order [GB_PE_RJ_DRAW_ONE=1]",
                stage, "/".join(members),
                " ".join(f"P({n})={w:.2f}" for n, w in zip(members, weights)),
                len(plan) - 1,
            )
            self._pe_rj_draw_one_announced = True
        if os.environ.get("GF_MOVE_TIMING", "0") == "1":
            print(f"[GF_TIMING] stage={stage} pe_rj_draw_one -> {pick}",
                  flush=True)
        return plan

    def _propose_moves(self, model, state):
        """Dispatch one iteration over the wrapped moves (the body of :meth:`propose`).

        Order of precedence: search legs (:meth:`_propose_leg`), the PE draw-one
        plan (:meth:`_pe_rj_draw_one_plan`, then the cadence over the plan),
        the per-move cadence (``gf_move_every``), ``weighted_cycle``,
        ``random_choice``, and finally the fixed sequence (eryn's
        ``CombineMove.propose`` unless ``GF_MOVE_TIMING=1``, a mid-iteration
        checkpoint or a coarse sidecar routes it through :meth:`_run_sequence`).
        With ``GF_MOVE_TIMING=1`` every branch also prints its ``[GF_TIMING]``
        selection / cadence-skip lines.

        Returns:
            tuple: ``(state, accepted)``.

        Note:
            When the cadence skips EVERY wrapped move in one iteration this
            returns zeros shaped like ``self.accepted``. On a combine, eryn's
            ``CombineMove.accepted`` getter raises ``AttributeError`` (the
            combine's own ``_accepted`` is never set; the setter writes the
            wrapped moves' counters), so that branch raises instead of
            returning. ``recipe.Stage.setup`` depends on the same getter
            raising (its ``hasattr`` test before it initializes the
            counters). Not reached on the 9mo / 12mo runs (no cadence).
        """
        # search LEGS first: the cursor owns the plan AND the cadence (in
        # cycle units), so neither the PE draw-one plan nor the per-propose
        # cadence plan below may run ahead of it
        if getattr(self, "gf_legs", None) is not None:
            return self._propose_leg(model, state)
        plan = self._pe_rj_draw_one_plan(model)
        if plan is not None:
            plan, _skipped = self._gf_cadence_plan(plan)
            if _skipped and os.environ.get("GF_MOVE_TIMING", "0") == "1":
                print(
                    f"[GF_TIMING] stage={getattr(self, 'gf_stage_name', '?')} "
                    f"cadence skip: {','.join(_skipped)}", flush=True)
            return self._run_sequence(model, state, plan)

        if self.gf_move_every is not None:
            due, _skipped = self._gf_cadence_plan(self.moves)
            if _skipped and os.environ.get("GF_MOVE_TIMING", "0") == "1":
                print(
                    f"[GF_TIMING] stage={getattr(self, 'gf_stage_name', '?')} "
                    f"cadence skip: {','.join(_skipped)}", flush=True)
            if not due:  # every wrapped move cadenced off this iteration
                return state, np.zeros_like(self.accepted)
            return self._run_sequence(model, state, due)

        if getattr(self, "weighted_cycle", False) and len(self.moves) > 1:
            # GB PE cycle style (user ruling 2026-08-26): one propose runs
            # a full drawn cycle -- len(moves) sub-moves drawn WITH
            # replacement by weight -- so the backend stores one row per
            # iteration instead of one per move (the random_choice mode's
            # 1-row-per-move storage artifact).
            # TODO: think about replacement -- a weighted draw WITHOUT
            # replacement (a weighted permutation) would guarantee every
            # move runs each iteration while keeping the random order.
            n = len(self.moves)
            w = self.move_weights
            if w is None:
                w = np.full(n, 1.0 / n)
            idx = model.random.choice(n, size=n, replace=True, p=w)
            timing = os.environ.get("GF_MOVE_TIMING", "0") == "1"
            accepted_out = None
            for k in idx:
                move = self.moves[int(k)]
                if isinstance(move, tuple):  # (move, weight) entry
                    move = move[0]
                if timing:
                    name = getattr(move, "gf_move_name", type(move).__name__)
                    print(
                        f"[GF_TIMING] stage="
                        f"{getattr(self, 'gf_stage_name', '?')} "
                        f"weighted_cycle -> {name}",
                        flush=True,
                    )
                self._gf_precondition(move, model)
                state, accepted = move.propose(model, state)
                accepted_out = (
                    accepted.copy() if accepted_out is None
                    else accepted_out + accepted
                )
                # sub-move boundary: coherent resume point (throttled; no-op
                # unless run.py armed the module on this rank)
                midit_checkpoint.maybe_write(
                    state,
                    tag=(
                        f"stage={getattr(self, 'gf_stage_name', '?')} after "
                        f"{getattr(move, 'gf_move_name', type(move).__name__)}"
                    ),
                )
            return state, accepted_out

        if getattr(self, "random_choice", False) and len(self.moves) > 1:
            # One move per step, drawn from the sampler's RNG so the choice
            # is reproducible under a fixed seed. Sub-state guards in
            # propose() still apply to whichever move ran.
            k = int(model.random.randint(len(self.moves)))
            move = self.moves[k]
            if isinstance(move, tuple):  # (move, weight); weight unused here
                move = move[0]
            name = getattr(move, "gf_move_name", type(move).__name__)
            if os.environ.get("GF_MOVE_TIMING", "0") == "1":
                print(f"[GF_TIMING] stage={getattr(self, 'gf_stage_name', '?')} "
                      f"random_choice -> {name}", flush=True)
            self._gf_precondition(move, model)
            return move.propose(model, state)

        timing = os.environ.get("GF_MOVE_TIMING", "0") == "1"
        if (
            not timing
            and not midit_checkpoint.armed()
            and self._gf_sidecar_runtime_lookup() is None
        ):
            return super().propose(model, state)
        return self._run_sequence(model, state, self.moves)

    # ---- search legs --------------------------------------------------------
    def gf_move_names(self):
        """The wrapped moves' declarative names, in order."""
        out = []
        for m in self.moves:
            mm = m[0] if isinstance(m, tuple) else m
            out.append(str(getattr(mm, "gf_move_name", type(mm).__name__)))
        return out

    def _propose_leg(self, model, state):
        """Run ONE leg: cursor -> next leg-ender, cadence gates in CYCLE units.

        The leg-ender itself is never gated: it is what the row is saved
        after, and a leg with no row would break the resume-by-name contract.

        Moves run one at a time, so a move that sets ``gf_leg_end_now`` ends
        the leg early after itself, and an F-stat RJ move with a zero-peak
        epoch (``gf_skip_partner_once``) skips its in-model partner once
        (:meth:`_gf_is_inmodel_partner`). The returned state is tagged with
        ``gf_saved_after`` (the resume position), ``gf_move_order`` and
        ``gf_stage_name`` for the saver; one ``[LEG ...]`` log line per leg.

        Returns:
            tuple: ``(state, accepted)`` (``accepted`` ``None`` when no move ran).
        """
        cur = self.gf_legs
        idx, end_name, _last = cur.plan()
        every = self.gf_move_every
        due, skipped = [], []
        for i in idx:
            m = self.moves[i]
            mm = m[0] if isinstance(m, tuple) else m
            e = 1 if every is None else int(every[i])
            name = cur.order[i]
            if name == end_name or cur.cycles % e == 0:
                due.append(mm)
            else:
                skipped.append(name)
        stage = getattr(self, "gf_stage_name", "?")
        leg_i = cur.leg_index
        cycles0 = cur.cycles
        # One move at a time, so a CONDITIONAL leg-ender can stop the leg:
        # a move that sets ``gf_leg_end_now`` on itself (the gated noise
        # head after it CHANGED the noise and ran its in-model pass -- user
        # ruling 2026-09-30, "a save after the in-model noise step whenever
        # it runs") ends the leg right there; the row is saved after ITS
        # name and the rest of the leg runs in the next propose. On a hold
        # the flag stays False and the leg runs on to its static ender.
        #
        # ZERO-PEAK F-STAT SKIP (user ruling 2026-10-10). An F-stat RJ move
        # whose adopted epoch has zero peaks does bookkeeping only and sets
        # ``gf_skip_partner_once``: the NEXT move of this leg is then skipped
        # when it is that move's in-model partner (rj_fstat_search ->
        # in_model_fstat), and nothing else is. The flag is consumed here, so
        # it covers one cycle. The leg still ENDS at the partner's name when
        # the partner is the leg-ender, so the saved-after row, the cursor
        # and the resume path are exactly what they are on a normal cycle.
        accepted = None
        ran = []
        partner_skipped = []
        skip_partner_of = None
        ended_early = False
        for k_due, mm in enumerate(due):
            name = getattr(mm, "gf_move_name", type(mm).__name__)
            if skip_partner_of is not None:
                owner, skip_partner_of = skip_partner_of, None
                owner_name = getattr(owner, "gf_move_name", type(owner).__name__)
                if self._gf_is_inmodel_partner(owner, mm):
                    partner_skipped.append(name)
                    logger.info(
                        "[LEG %s] %r skipped this cycle: the in-model partner "
                        "of %r, whose F-stat epoch has 0 peaks", stage, name,
                        owner_name)
                    continue
                logger.warning(
                    "[LEG %s] %r asked to skip its in-model partner, but the "
                    "next move %r is not one; nothing skipped", stage,
                    owner_name, name)
            state, acc = self._run_sequence(model, state, [mm])
            accepted = acc.copy() if accepted is None else accepted + acc
            ran.append(name)
            if bool(getattr(mm, "gf_skip_partner_once", False)):
                mm.gf_skip_partner_once = False   # consumed: one cycle only
                skip_partner_of = mm
            flag = bool(getattr(mm, "gf_leg_end_now", False))
            if flag:
                mm.gf_leg_end_now = False        # consumed: never ends a later leg
            if flag and name != end_name:
                end_name = name
                cur.end_early_at(cur.order.index(name))
                ended_early = True
                break
        if skip_partner_of is not None:
            logger.warning(
                "[LEG %s] %r asked to skip its in-model partner, but no move "
                "follows it in this leg; nothing skipped", stage,
                getattr(skip_partner_of, "gf_move_name",
                        type(skip_partner_of).__name__))
        if not ended_early:
            cur.advance()
        # tag the state eryn is about to save: the row's saved-after NAME
        # (the resume position) plus the ordered list the saver stamps once
        state.gf_saved_after = end_name
        state.gf_move_order = list(cur.order)
        state.gf_stage_name = stage
        logger.info(
            "[LEG %s] cycle %d leg %d/%d: ran %s -> row saved after %r%s%s%s",
            stage, cycles0, leg_i + 1, cur.nlegs, ran, end_name,
            " (the noise changed: leg ended early)" if ended_early else "",
            f" (cadence skipped {skipped})" if skipped else "",
            (f" (zero-peak F-stat: in-model partner skipped {partner_skipped})"
             if partner_skipped else ""))
        return state, accepted

    @staticmethod
    def _gf_is_inmodel_partner(owner, move) -> bool:
        """Is ``move`` the in-model partner of the RJ move ``owner``?

        The pure in-model GB slot that follows an RJ move in its leg and is
        named for the RJ move it polishes (``recipe.GB_IN_MODEL_SLOTS``:
        ``in_model_fstat`` polishes ``rj_fstat_search``'s births): an
        ``in_model*`` name (the leg-ender prefix), not itself an RJ move, on
        the owner's branch.
        """
        from ..legs import DEFAULT_LEG_END_PREFIX

        name = str(getattr(move, "gf_move_name", type(move).__name__))
        return (name.startswith(DEFAULT_LEG_END_PREFIX)
                and not bool(getattr(move, "is_rj_prop", False))
                and getattr(move, "branch_name", None)
                == getattr(owner, "branch_name", None))

    def _run_sequence(self, model, state, moves):
        """Run ``moves`` in order -- eryn CombineMove.propose semantics plus
        optional [GF_TIMING] instrumentation and mid-iteration checkpoints
        at the sub-move boundaries (each wrapped move returns a coherent
        state -- cold rows synced -- so every boundary is a resume point).
        ``moves`` is normally ``self.moves``; the PE-only exclusive RJ draw
        passes its per-iteration plan instead.

        With ``GF_MOVE_TIMING=1`` prints one ``[GF_TIMING] stage=... move=...``
        line per move (wall time, peak RSS, cupy pool) and a ``move=__total__``
        line; ``GF_MOVE_TIMING_SYNC=1`` also synchronizes the cupy stream after
        each move. ``midit_checkpoint.maybe_write`` runs after each move (a
        no-op unless the run armed it; launchers ``MIDIT_CHECKPOINT=0``).

        Returns:
            tuple: ``(state, accepted)``, ``accepted`` summed over the moves.
        """
        timing = os.environ.get("GF_MOVE_TIMING", "0") == "1"
        self._gf_timing_iter = getattr(self, "_gf_timing_iter", 0) + 1
        it = self._gf_timing_iter
        stage = getattr(self, "gf_stage_name", "?")
        sync = timing and os.environ.get("GF_MOVE_TIMING_SYNC", "0") == "1"

        accepted_out = None
        t_all = time.perf_counter()
        for move in moves:
            if isinstance(move, tuple):
                move = move[0]
            rss0 = _gf_rss_mb()
            t0 = time.perf_counter()
            self._gf_precondition(move, model)
            state, accepted = move.propose(model, state)
            if sync and "cupy" in sys.modules:
                try:
                    sys.modules["cupy"].cuda.get_current_stream().synchronize()
                except Exception:
                    pass
            name = getattr(move, "gf_move_name", type(move).__name__)
            if timing:
                wall = time.perf_counter() - t0
                rss1 = _gf_rss_mb()
                used, total = _gf_gpu_pool_mb()
                print(
                    f"[GF_TIMING] stage={stage} move={name} it={it} "
                    f"wall_s={wall:.4f} rss_mb={rss1:.0f} d_rss_mb={rss1 - rss0:.0f} "
                    f"gpu_used_mb={used if used is not None else -1:.0f} "
                    f"gpu_pool_mb={total if total is not None else -1:.0f}",
                    flush=True,
                )
            if accepted_out is None:
                accepted_out = accepted.copy()
            else:
                accepted_out += accepted
            midit_checkpoint.maybe_write(
                state, tag=f"stage={stage} after {name}"
            )
        if timing:
            print(
                f"[GF_TIMING] stage={stage} move=__total__ it={it} "
                f"wall_s={time.perf_counter() - t_all:.4f} rss_mb={_gf_rss_mb():.0f}",
                flush=True,
            )
        return state, accepted_out


class MaxLogLCombineMove(GFCombineMove):
    """Combine move whose max-logL convergence spans ALL wrapped moves.

    ``PSDMove(max_logl_mode=True)`` converges each noise branch SEPARATELY:
    every move runs its own ``run_move_max_likelihood`` loop to its own
    plateau. For a joint search that is the wrong criterion -- psd and galfor
    are two parameterizations of ONE noise model, so maximizing them
    independently can stall each at a plateau the pair could climb past
    together (galfor moves change the residual the psd move is fitting, and
    vice versa).

    This promotes the criterion one level: the wrapped moves each run a
    single pass, and THIS object owns the loop, checking the cold-chain max
    log-likelihood over the whole set. Effectively one move, maximizing
    ``logl`` over repeated iterations of psd AND galfor together.

    Wrap the ``*_pe`` moves, not the ``*_search`` ones. ``max_logl_mode`` is
    consulted in exactly one place (:meth:`PSDMove.propose_local
    <lisatools.globalfit.moves.psdmove.PSDMove.propose_local>`) -- it only
    chooses the plateau loop over the ensemble search / a single
    ``run_move_for_loop`` -- so the pe move IS the search move minus its
    private loop, which is precisely the inner behaviour wanted here.
    Wrapping the search moves would nest two plateau loops.

    The plateau rule mirrors ``PSDMove.run_move_max_likelihood`` -- exit
    after ``num_checks`` consecutive non-improving iterations, counting only
    once the likelihood has moved at least once (so a move that has not yet
    taken effect cannot trip the exit immediately) -- with one refinement:
    an improvement only counts (resets the counter) when it beats the
    baseline by more than ``tol`` log-likelihood units. A stretch ensemble
    near the mode produces small strict improvements almost every iteration,
    so a search on a ~1e7-scale lnL would otherwise polish the noise floor
    indefinitely. Sub-``tol`` gains do NOT advance the baseline either, so a
    slow accumulated climb (more than ``tol`` per ``num_checks`` window)
    still registers as progress and keeps the stage alive.

    The plateau is PER WALKER by default (``MAXLOGL_PER_WALKER``, default ``1``):
    each cold walker keeps its own baseline and flat counter, and the move is
    done only when the laggard has plateaued. The loop is CHUNKED: at most
    ``iters_per_step`` inner passes per :meth:`propose` call, with the
    bookkeeping kept on the instance across calls, so the engine saves between
    chunks; ``SearchRecipeStep`` reads :attr:`maxlogl_plateau_done` to advance a
    search stage.

    Production (9mo / 12mo): the driver's ``JointMaxLogLSearch`` builds one for
    ``source_search`` (``[sobbh_pe, mbh_pe, emri_pe]``, ``SOURCE_SEARCH_CHECKS=10``,
    ``SOURCE_SEARCH_TOL=10``, ``SOURCE_SEARCH_MAX_ROUNDS=200``) and one for ``gb_search_2``'s
    ``noise_cycle_end_search`` (``[psd_pe, galfor_pe]``,
    ``restart_each_propose`` on, ``NOISE_SEARCH_CHECKS`` default 5).

    Args:
        *args: Forwarded to :class:`GFCombineMove` (the first is the list of
            wrapped moves).
        num_checks: Consecutive non-improving inner passes that end the loop.
            Default 5.
        max_iter: Hard ceiling on inner passes, counted over the instance's
            life (across chunked calls; 0 = unbounded). Default 0;
            ``MAXLOGL_MAX_ITER`` overrides it when set (the env var WINS over the
            keyword; the driver's ``JointMaxLogLSearch`` sets a per-instance
            ``max_iter`` afterwards for ``source_search``,
            ``SOURCE_SEARCH_MAX_ROUNDS``).
        tol: Improvement (lnL units) a pass must beat the baseline by to count.
            Default 5.0; ``MAXLOGL_TOL`` overrides it when set (the env var WINS
            over the keyword; the driver's ``apply_release_tol`` sets a
            per-instance value afterwards, e.g. ``SOURCE_SEARCH_TOL``). Launchers
            ``MAXLOGL_TOL=20``.
        iters_per_step (int, optional): Inner passes per :meth:`propose` call;
            ``None`` reads ``MAXLOGL_ITERS_PER_STEP`` (default 10).
        restart_each_propose: Forget the previous call's verdict at every
            propose, i.e. converge from scratch on the current residual.
            Default ``False``.
        **kwargs: Forwarded to :class:`GFCombineMove`.

    Attributes:
        maxlogl_plateau_done (bool): Whether the last call ended on a plateau
            (or on ``max_iter``).

    Note:
        Other env knobs, read per call: ``MAXLOGL_LOG_EVERY`` (default ``1``;
        one ``[MAXLOGL]`` line per inner pass, ``0`` silences) and
        ``MAXLOGL_FREEZE_CONVERGED`` (default ``0``; ``1`` freezes a converged
        walker's columns, main state and sub-states, in search-kind stages only;
        that path has not been exercised on a real tempered state).

    Example:
        .. code-block:: python

            from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove

            joint = MaxLogLCombineMove(
                [stock_moves["psd_pe"], stock_moves["galfor_pe"]],
                num_checks=5, share_temperature_control=False,
                restart_each_propose=True,
            )
    """

    def __init__(
        self, *args, num_checks: int = 5, max_iter: int = 0, tol: float = 5.0,
        iters_per_step=None, restart_each_propose: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_checks = int(num_checks)
        # True = EVERY propose runs the plateau rule from scratch (user ruling
        # 2026-10-07, the cycle-end noise convergence of a search stage: "the
        # galfor and psd sampling should be required to converge before moving
        # on"). Default False keeps the chunked semantics below: bookkeeping
        # persists across calls, so a plateaued instance takes one round.
        self.restart_each_propose = bool(restart_each_propose)
        # Per-instance cap on inner iterations per propose() call. ``None``
        # defers to MAXLOGL_ITERS_PER_STEP (10). The joint noise move that
        # rides along inside gb_search never reaches a lasting plateau (the
        # GB residual moves every iteration, so it re-climbs each call) and
        # was taking ~6 x 16 s rounds per GB iteration; the gb_search
        # instance caps this at GB_SEARCH_NOISE_ITERS_PER_STEP (2026-09-11)
        # while the standalone noise stages keep the full budget.
        self.iters_per_step = (
            None if iters_per_step is None else int(iters_per_step)
        )
        # Absolute lnL units. The default is deliberately soft (a search
        # stage only needs the noise model roughly converged before the next
        # stage samples it in PE mode); MAXLOGL_TOL overrides, 0 restores
        # the strict any-improvement rule.
        self.tol = float(os.environ.get("MAXLOGL_TOL", tol))
        # Hard ceiling on the plateau loop. 0 = unbounded, which is what
        # PSDMove.run_move_max_likelihood does -- fine when one move owns a
        # cheap likelihood, dangerous here: this can wrap several moves over
        # a basis with no compiled kernel (the WDM noise likelihood falls
        # back to the ACA route), so a likelihood that keeps twitching by
        # 1e-9 never trips the plateau and the stage never ends.
        # MAXLOGL_MAX_ITER overrides.
        self.max_iter = int(os.environ.get("MAXLOGL_MAX_ITER", max_iter))

    def _propose_moves(self, model, state):
        """One chunk of the joint plateau loop (see the class docstring).

        Repeats :meth:`GFCombineMove._propose_moves_once` until every walker's
        flat counter reaches ``num_checks`` (or ``max_iter``), at most
        ``iters_per_step`` passes this call; sets :attr:`maxlogl_plateau_done`
        and prints ``[MAXLOGL]`` progress / done / chunk-break lines.

        Returns:
            tuple: ``(state, accepted)`` of the last pass.
        """
        # CHUNKED plateau loop (2026-08-12): at most MAXLOGL_ITERS_PER_STEP
        # inner iterations per propose() call, with the plateau bookkeeping
        # persisted on the instance across calls. The engine saves the
        # backend between propose() calls, so a killed run resumes from the
        # last chunk's ensemble instead of restarting the whole search --
        # previously the ENTIRE search ran inside one engine iteration and
        # a stop during it lost everything. SearchRecipeStep consults
        # ``maxlogl_plateau_done`` to advance the stage. Inside a non-search
        # stage (the joint noise move rides along in gb_search), a plateaued
        # instance keeps taking one inner iteration per call, so the noise
        # model keeps sampling underneath.
        # getattr: test harnesses build this move without __init__.
        if getattr(self, "restart_each_propose", False):
            # forget the previous call's verdict: converge on THIS residual
            # (same effect as noise_ratchet.reset_maxlogl_search)
            self.__dict__.pop("_ml_state", None)
            self.maxlogl_plateau_done = False
        _cap = getattr(self, "iters_per_step", None)
        max_inner = (
            int(os.environ.get("MAXLOGL_ITERS_PER_STEP", "10"))
            if _cap is None else int(_cap)
        )
        # ---- PER-WALKER PLATEAU (user ruling 2026-09-26) -------------------
        # ⚠ THIS USED TO BE ONE SCALAR: ``float(state.log_like[0].max())``,
        # i.e. the BEST walker. "Joint" meant joint over the wrapped MOVES,
        # never over the walkers -- so the stage advanced as soon as the
        # luckiest walker stopped climbing, and every other walker was handed
        # to the next stage wherever it happened to be.
        #
        # MEASURED, first 3-month v9 run (job 633/636, stored row 0): the
        # plateau was declared on best=52494067.95 while the other three cold
        # walkers sat 71,050 / 128,080 / 1,600 lnL below it. Because
        # gb_search_1/2 then FROZE the noise, those four walkers searched for
        # GBs against four different noise floors for the rest of the run --
        # S_oms spread 1.74x, S_tm spread 7.81x across walkers, with the leaf
        # counts ordering exactly inversely to each walker's S_tm (678 / 341 /
        # 247 / 235 leaves at 0.66x / 2.51x / 5.03x / 5.12x the injection).
        #
        # Now every walker carries its own baseline, its own flat counter and
        # its own changed_once latch, and the stage advances only when the
        # LAGGARD has plateaued. MAXLOGL_PER_WALKER=0 restores the old
        # best-walker scalar rule bit-identically.
        _pw_env = os.environ.get("MAXLOGL_PER_WALKER", "1").strip()
        per_walker = _pw_env not in ("0", "false", "False", "no", "off", "")
        if not hasattr(self, "_ml_state"):
            _n = 1
            if per_walker:
                try:
                    _n = int(np.atleast_1d(np.asarray(state.log_like)[0]).size)
                except Exception:
                    # A harness state with no usable log_like row: fall back
                    # to the scalar rule rather than raising inside a move.
                    per_walker, _n = False, 1
            self._ml_state = dict(
                num_so_far=np.zeros(_n, dtype=int),
                max_logl=np.full(_n, -np.inf),   # running best (logging only)
                ref_logl=np.full(_n, -np.inf),   # baseline at last reset
                changed_once=np.zeros(_n, dtype=bool),
                n_iter=0,
                per_walker=bool(per_walker),
            )
        ms = self._ml_state
        per_walker = bool(ms["per_walker"])
        # SHUT OFF A CONVERGED WALKER (same ruling): once a walker's counter
        # reaches num_checks it stops moving, so the remaining rounds belong
        # to the walkers still climbing and a converged walker cannot wander
        # back down off its own maximum.
        #
        # ⚠ SEARCH-KIND STAGES ONLY. Restoring a walker's rows is safe while
        # this move owns the likelihood, which is true in the standalone
        # noise stage. Riding inside gb_search the GB residual moves between
        # calls, so a restored log_like would be STALE and would corrupt the
        # very bookkeeping it is meant to protect -- there the per-walker rule
        # still governs the loop, but nothing is frozen.
        # ⚠⚠ DEFAULT OFF, AND IT SHIPPED ON ONCE. The first cut restored only
        # the MAIN state's rows, so the psd branch's sub-state still held the
        # pre-freeze coordinates and the next propose died on
        #   [psd] cold-chain coords mismatch between the main state and its
        #   sub-state (1 of 2 alive leaves differ)
        # -- the dual-representation trap, the same one GFRidgeGibbsMove hit
        # in August. The restore below now covers sub_states too, but that
        # path has NOT been exercised on a real tempered state, so it stays
        # opt-in: MAXLOGL_FREEZE_CONVERGED=1 to try it. The per-walker plateau
        # is independent of this and is ON by default.
        _freeze = (
            per_walker
            and os.environ.get("MAXLOGL_FREEZE_CONVERGED", "0").strip()
            in ("1", "true", "True", "yes", "on")
            and str(getattr(self, "gf_stage_kind", "")) == "search"
        )
        accepted = None
        t0 = time.perf_counter()
        stage = getattr(self, "gf_stage_name", "?")
        # Progress EVERY iteration by default. This loop has no natural
        # output and no fixed length, so without it a long stage is
        # indistinguishable from a hung one -- which is exactly what the
        # first full-band run looked like. MAXLOGL_LOG_EVERY=0 silences it.
        log_every = int(os.environ.get("MAXLOGL_LOG_EVERY", "1"))
        inner = 0
        while inner < max_inner and (
            int(ms["num_so_far"].min()) < self.num_checks or accepted is None
        ):
            frozen = (
                ms["num_so_far"] >= self.num_checks if _freeze
                else np.zeros_like(ms["num_so_far"], dtype=bool)
            )
            snap = self._snapshot_walkers(state, frozen) if frozen.any() else None
            state, accepted = self._propose_moves_once(model, state)
            if snap is not None:
                state = self._restore_walkers(state, snap)
            ms["n_iter"] += 1
            inner += 1
            cur = np.atleast_1d(np.asarray(state.log_like, dtype=float)[0])
            if not per_walker:
                cur = np.array([float(cur.max())])
            # ⚠ THE LATCH IS GLOBAL, NOT PER WALKER, AND THAT IS DELIBERATE.
            # Its job is "the move has not taken effect yet, so nothing may
            # trip the exit" -- a property of the MOVE, not of one walker.
            # Armed per-walker it deadlocks the stage: a walker that is flat
            # from the first round (stuck, or already at its optimum) never
            # arms, so its counter never increments, so the laggard rule never
            # clears and the stage runs to MAXLOGL_MAX_ITER -- which defaults
            # to 0, i.e. unbounded. Caught on a two-walker scripted case where
            # w0 sat flat and the stage would not end. Armed globally, that
            # walker simply counts flat rounds and converges, which is the
            # right answer: it IS plateaued.
            _moved = (cur != ms["ref_logl"]) & ~np.isinf(ms["ref_logl"])
            ms["changed_once"] |= bool(_moved.any())
            ms["max_logl"] = np.maximum(ms["max_logl"], cur)
            improved = (cur > ms["ref_logl"] + self.tol) | np.isinf(ms["ref_logl"])
            # Sub-tol gains do NOT advance the baseline, so a slow accumulated
            # climb still registers as progress -- per walker now.
            ms["ref_logl"] = np.where(improved, cur, ms["ref_logl"])
            ms["num_so_far"] = np.where(
                improved, 0,
                np.where(ms["changed_once"], ms["num_so_far"] + 1,
                         ms["num_so_far"]),
            )
            if log_every and (ms["n_iter"] % log_every == 0):
                _n_conv = int((ms["num_so_far"] >= self.num_checks).sum())
                if per_walker and cur.size > 1:
                    _lag = int(np.argmin(ms["num_so_far"]))
                    detail = (
                        f"logl={cur.min():.6f}..{cur.max():.6f} "
                        f"(walker spread {cur.max() - cur.min():.3f}) "
                        f"best={ms['max_logl'].max():.6f} "
                        f"converged={_n_conv}/{cur.size} walker(s); "
                        f"laggard w{_lag} flat "
                        f"{int(ms['num_so_far'][_lag])}/{self.num_checks}"
                        + (f" frozen={int(frozen.sum())}" if _freeze else "")
                    )
                else:
                    detail = (
                        f"logl={float(cur[0]):.6f} "
                        f"best={float(ms['max_logl'][0]):.6f} "
                        + ("IMPROVED" if bool(improved[0])
                           else f"flat {int(ms['num_so_far'][0])}/{self.num_checks}")
                        + f" changed_once={bool(ms['changed_once'][0])}"
                    )
                print(
                    f"[MAXLOGL] stage={stage} iter={ms['n_iter']} {detail} "
                    f"elapsed_s={time.perf_counter() - t0:.1f}",
                    flush=True,
                )
            if self.max_iter and ms["n_iter"] >= self.max_iter:
                print(
                    f"[MAXLOGL] stage={stage} hit max_iter={self.max_iter} "
                    f"before plateau (best={ms['max_logl'].max():.6f}); "
                    "advancing.",
                    flush=True,
                )
                break
        # EVERY walker, not the best one: the laggard decides.
        done = bool(int(ms["num_so_far"].min()) >= self.num_checks) or bool(
            self.max_iter and ms["n_iter"] >= self.max_iter
        )
        first_done = done and not getattr(self, "maxlogl_plateau_done", False)
        self.maxlogl_plateau_done = bool(done)
        if first_done:
            _spread = ""
            if per_walker and ms["max_logl"].size > 1:
                _b = ms["max_logl"]
                _spread = (f", per-walker best {_b.min():.6f}..{_b.max():.6f}"
                           f" spread {_b.max() - _b.min():.3f}")
            print(
                f"[MAXLOGL] stage={stage} done after {ms['n_iter']} iterations "
                f"(best={ms['max_logl'].max():.6f}{_spread}, "
                f"{time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        elif not done:
            print(
                f"[MAXLOGL] stage={stage} chunk break at iter={ms['n_iter']} "
                f"({inner}/{max_inner} this step; checkpointing).",
                flush=True,
            )
        return state, accepted

    # ---- converged-walker freeze helpers ----------------------------------
    # Whole walker COLUMNS only: no index arithmetic into supps and no
    # reliance on walker_inds, which is where this codebase has twice scored
    # walkers against each other's residuals. A walker not in ``mask`` is
    # never touched.
    #
    # ⚠ BOTH REPRESENTATIONS OR NEITHER. A branch that carries a tempered
    # ModuleSubState is stored TWICE -- the main engine state and
    # ``state.sub_states[name]`` -- and ``_check_substate_consistency``
    # compares the main cold row against the sub-state's row 0 on every
    # propose. Restoring one and not the other is not a silent inconsistency,
    # it is a hard ValueError on the next move, which is exactly how the
    # first version of this died on the cluster ("[psd] cold-chain coords
    # mismatch ... 1 of 2 alive leaves differ").
    @staticmethod
    def _snapshot_walkers(state, mask):
        """Copy the masked walkers' columns: coords / inds of branches and sub-states, lnL, lnP.

        Returns the snapshot dict, or ``None`` for a layout it does not
        understand (nothing is then frozen). Used only with
        ``MAXLOGL_FREEZE_CONVERGED=1``.
        """
        w = np.flatnonzero(np.asarray(mask))
        snap = {"w": w, "branches": {}, "subs": {},
                "log_like": None, "log_prior": None}
        try:
            for name, br in (state.branches or {}).items():
                entry = {}
                for attr in ("coords", "inds"):
                    arr = getattr(br, attr, None)
                    if arr is not None:
                        entry[attr] = np.array(arr[:, w])
                if entry:
                    snap["branches"][name] = entry
            for name, sub in (getattr(state, "sub_states", None) or {}).items():
                if sub is None or not getattr(sub, "tempered_initialized", False):
                    continue
                entry = {}
                for attr in ("coords", "inds"):
                    arr = getattr(sub, attr, None)
                    if arr is not None:
                        entry[attr] = np.array(arr[:, w])
                if entry:
                    snap["subs"][name] = entry
            for key in ("log_like", "log_prior"):
                arr = getattr(state, key, None)
                if arr is not None:
                    snap[key] = np.array(np.asarray(arr)[:, w])
        except Exception:
            # Any layout this does not understand -> freeze NOTHING rather
            # than half-restore. The per-walker plateau rule still applies.
            return None
        return snap

    @classmethod
    def _restore_walkers(cls, state, snap):
        """Write a :meth:`_snapshot_walkers` snapshot back into ``state`` in place; return it."""
        if snap is None:
            return state
        w = snap["w"]
        for name, entry in snap["branches"].items():
            cls._restore_cols((state.branches or {}).get(name), entry, w)
        for name, entry in snap["subs"].items():
            cls._restore_cols(
                (getattr(state, "sub_states", None) or {}).get(name), entry, w)
        for key in ("log_like", "log_prior"):
            arr = snap.get(key)
            cur = getattr(state, key, None)
            if arr is not None and cur is not None and cur[:, w].shape == arr.shape:
                cur[:, w] = arr
        return state

    @staticmethod
    def _restore_cols(holder, entry, w):
        """Restore walker columns ``w`` of each array in ``entry`` on ``holder`` (shape-checked)."""
        if holder is None:
            return
        for attr, arr in entry.items():
            cur = getattr(holder, attr, None)
            if cur is not None and cur[:, w].shape == arr.shape:
                cur[:, w] = arr
