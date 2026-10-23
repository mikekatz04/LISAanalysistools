"""Plain-function global-fit move: ``fn(model, state)`` wrapped as a Move.

The simplest way to put your own proposal inside a global fit: write a
function with the eryn move signature, store whatever you need in your own
closure/object, and only touch ``model.analysis_container_arr`` (read the
residual, adjust it, write it back). The framework owns the bookkeeping —
acceptance normalization and keeping ``state.log_like`` consistent with the
residual so HDF5 persistence stays correct.

Entry point: :class:`FunctionMove`. It is created for you whenever a plain
callable is passed to :meth:`Recipe.add_move <lisatools.globalfit.recipe.Recipe.add_move>`
or :meth:`StockGlobalFit.add_move <lisatools.globalfit.stock.base.StockGlobalFit.add_move>`
(see ``docs/stock-stages-and-moves.md``). The 9mo / 12mo production driver adds
no function moves; this is user-facing API.
"""

import numpy as np
from eryn.moves import Move as ErynMove

from ...utils.utility import asnumpy
from .globalfitmove import GlobalFitMove, Move, MoveBuildContext

__all__ = ["FunctionMove"]


class FunctionMove(Move, GlobalFitMove, ErynMove):
    """Wraps ``fn(model, state) -> (new_state, accepted)`` as a global-fit move.

    One class, two lives: it is the declarative recipe entry (cheap,
    picklable) AND the runtime move (its :meth:`setup` captures the live
    run objects onto itself and returns ``None``, so the stage wraps the
    object itself in its :class:`~lisatools.globalfit.moves.globalfitmove.GFCombineMove`).

    The user contract for ``fn``::

        def my_move(model, state):
            aca = model.analysis_container_arr   # read residual, adjust, write back
            return state, None                    # or (new_state, accepted)

    ``fn`` may return ``(state, accepted)``, a bare state, or ``None`` (keep
    the input state); a missing/None ``accepted`` becomes zeros. After each
    call the framework re-syncs ``state.log_like`` from the residual
    (``acs.likelihood()``, the run convention) unless ``sync_log_like=False``
    — this is what keeps the saved chain consistent with whatever ``fn`` did
    to the residual buffers. A function move does not write any branch
    sub-state and does not mirror cold rows (``sync_cold_row``): if ``fn``
    edits a tempered branch it must keep the sub-state's row 0 equal to the
    main state's cold row itself, or the next stage-entry check
    (``GF_SUBSTATE_CHECK``) raises.

    Picklability: the pre-build fit stays picklable when ``fn`` is a named
    module-level callable (same config rule as everywhere else); the
    setup-attached runtime objects (``acs``, ``fanout``) are dropped on pickle.

    Args:
        fn (callable): The proposal function ``fn(model, state)``.
        name (str, optional): Recipe name; defaults to ``fn.__name__`` (then
            ``"function_move"``).
        branch (str, optional): Branch this move samples. An info-appended
            branch (``fit.add_branch``) must be targeted by at least one move.
        debug (bool or dict, optional): Debug override (see
            :class:`~lisatools.globalfit.moves.globalfitmove.Move`).
        sync_log_like: Re-sync ``state.log_like`` from the residual after
            each call. Default ``True``.
        **move_kwargs: Forwarded to :class:`eryn.moves.Move` (e.g.
            ``periodic``, ``temperature_control``).

    Raises:
        TypeError: ``fn`` is not callable.

    Attributes:
        fn (callable): The wrapped function.
        sync_log_like (bool): See ``sync_log_like``.
        acs: The run's :class:`~lisatools.analysiscontainer.AnalysisContainerArray`,
            set by :meth:`setup` (``None`` before).
        fanout: The run's walker fan-out (``ctx.fanout``; ``None`` single-process).

    Note:
        The ACS is per-walker and shared across temperatures, so the log-like
        sync broadcasts one likelihood per walker over all temperature rungs
        (exactly what the stock run does at startup). Function moves therefore
        see/adjust the cold-chain residual set.

        Under several compute ranks a function move runs on the HEAD only
        (``run.py`` lists it in its head-only warning): ``fn`` sees the head's
        walker-block ACA, and the log-like re-sync gathers every rank's
        block through ``fanout.gather_likelihood``.

    Example:
        >>> from lisatools.globalfit import FunctionMove, Recipe
        >>> def nudge(model, state):
        ...     return state, None
        >>> recipe = Recipe()
        >>> recipe.add_move(nudge, branch="line")
        FunctionMove('nudge', branch='line')
        >>> recipe.move_names()
        ['nudge']

        On a stock fit the same call goes through ``fit.add_move``, which
        also starts a live add on the next iteration:

        .. code-block:: python

            from eryn.prior import uniform_dist
            from lisatools.globalfit.stock import erebor

            fit = erebor.blank(nwalkers=4, ntemps=2)
            fit.add_branch("line", ndim=2,
                           priors={0: uniform_dist(0.0, 1.0), 1: uniform_dist(0.0, 1.0)},
                           moves=[nudge])
            fit.add_move(FunctionMove(nudge, name="nudge_again", sync_log_like=False),
                         branch="line")
            fit.run()
    """

    def __init__(
        self,
        fn,
        *,
        name=None,
        branch=None,
        debug=None,
        sync_log_like: bool = True,
        **move_kwargs,
    ):
        if not callable(fn):
            raise TypeError(f"FunctionMove expects a callable, got {type(fn).__name__}.")
        resolved = name or getattr(fn, "__name__", None) or "function_move"
        Move.__init__(self, resolved, branch=branch, debug=debug)
        GlobalFitMove.__init__(self, name=resolved)
        ErynMove.__init__(self, **move_kwargs)
        self.fn = fn
        self.sync_log_like = sync_log_like
        self.acs = None

    def setup(self, ctx: MoveBuildContext):
        """Capture the live run objects onto ``self``; ``self`` is the runtime move.

        Stores ``ctx.acs``, ``ctx.ntemps``, ``ctx.nwalkers`` and ``ctx.fanout``
        and, when the sampler shape is known, zeroes ``accepted`` at
        ``(ntemps, nwalkers)``.

        Args:
            ctx: The recipe build context.

        Returns:
            None: always (the base :meth:`Move.materialize` then uses ``self``).
        """
        self.acs = ctx.acs
        self.ntemps = ctx.ntemps
        self.nwalkers = ctx.nwalkers
        # multi-rank: the head's ACA holds only its walker block, so the
        # log_like re-sync must gather every rank's block (None single-process)
        self.fanout = getattr(ctx, "fanout", None)
        if ctx.ntemps is not None and ctx.nwalkers is not None:
            self.accepted = np.zeros((ctx.ntemps, ctx.nwalkers))
        return None  # self IS the runtime move

    def propose(self, model, state):
        """Call ``fn`` and normalize its output to eryn's ``(state, accepted)``.

        The ``propose(model, state)`` contract: one call per iteration of the
        owning stage (from its ``GFCombineMove``). ``fn``'s return is
        normalized (see the class docstring), then, with ``sync_log_like``,
        ``new_state.log_like[:]`` is overwritten with the per-walker
        likelihood of the current residual, broadcast over every rung.

        Args:
            model (eryn.model.Model): The eryn model; ``model.analysis_container_arr``
                is the residual (falls back to the ACS captured in :meth:`setup`).
            state (GFState): The current global-fit state.

        Returns:
            tuple: ``(new_state, accepted)``; ``accepted`` is ``np.ndarray``
            (zeros shaped like ``state.log_like`` when ``fn`` gave none).

        Raises:
            RuntimeError: ``sync_log_like`` is on and no
                ``AnalysisContainerArray`` is available.
        """
        out = self.fn(model, state)
        if isinstance(out, tuple):
            new_state = out[0] if len(out) > 0 else None
            accepted = out[1] if len(out) > 1 else None
        else:
            new_state, accepted = out, None
        if new_state is None:
            new_state = state
        if accepted is None:
            accepted = np.zeros_like(np.asarray(state.log_like))
        if self.sync_log_like:
            acs = getattr(model, "analysis_container_arr", None)
            if acs is None:
                acs = self.acs
            if acs is None:
                raise RuntimeError(
                    f"FunctionMove {self.name!r}: no AnalysisContainerArray available "
                    "to sync log_like from (run setup(ctx) first, or pass "
                    "sync_log_like=False)."
                )
            fanout = getattr(self, "fanout", None)
            if fanout is not None and not fanout.single:
                # head-only move under several compute ranks: ``fn`` ran on the
                # head's block; every rank reports its block's current likelihood
                new_state.log_like[:] = fanout.gather_likelihood(acs)[None, :]
            else:
                new_state.log_like[:] = asnumpy(acs.likelihood(complex=False))[None, :]
        return new_state, np.asarray(accepted)

    def __getstate__(self):
        state = super().__getstate__()
        state["acs"] = None
        state["fanout"] = None
        return state
