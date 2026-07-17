"""Plain-function global-fit move: ``fn(model, state)`` wrapped as a Move.

The simplest way to put your own proposal inside a global fit: write a
function with the eryn move signature, store whatever you need in your own
closure/object, and only touch ``model.analysis_container_arr`` (read the
residual, adjust it, write it back). The framework owns the bookkeeping —
acceptance normalization and keeping ``state.log_like`` consistent with the
residual so HDF5 persistence stays correct.
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
    run objects onto itself and returns ``None``).

    The user contract for ``fn``::

        def my_move(model, state):
            aca = model.analysis_container_arr   # read residual, adjust, write back
            return state, None                    # or (new_state, accepted)

    ``fn`` may return ``(state, accepted)``, a bare state, or ``None`` (keep
    the input state); a missing/None ``accepted`` becomes zeros. After each
    call the framework re-syncs ``state.log_like`` from the residual
    (``acs.likelihood()``, the run convention) unless ``sync_log_like=False``
    — this is what keeps the saved chain consistent with whatever ``fn`` did
    to the residual buffers.

    .. note::
       The ACS is per-walker and shared across temperatures, so the log-like
       sync broadcasts one likelihood per walker over all temperature rungs
       (exactly what the stock run does at startup). Function moves therefore
       see/adjust the cold-chain residual set.

    Picklability: the pre-build fit stays picklable when ``fn`` is a named
    module-level callable (same config rule as everywhere else); the
    setup-attached runtime objects are dropped on pickle.

    Args:
        fn: The proposal function ``fn(model, state)``.
        name: Recipe name; defaults to ``fn.__name__``.
        branch: Branch this move samples (optional).
        debug: Debug override (see :class:`Move`).
        sync_log_like: Re-sync ``state.log_like`` from the residual after
            each call. Default ``True``.
        **move_kwargs: Forwarded to :class:`eryn.moves.Move` (e.g.
            ``periodic``, ``temperature_control``).
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
        self.acs = ctx.acs
        self.ntemps = ctx.ntemps
        self.nwalkers = ctx.nwalkers
        if ctx.ntemps is not None and ctx.nwalkers is not None:
            self.accepted = np.zeros((ctx.ntemps, ctx.nwalkers))
        return None  # self IS the runtime move

    def propose(self, model, state):
        """Call ``fn`` and normalize its output to eryn's ``(state, accepted)``."""
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
            new_state.log_like[:] = asnumpy(acs.likelihood(complex=False))[None, :]
        return new_state, np.asarray(accepted)

    def __getstate__(self):
        state = super().__getstate__()
        state["acs"] = None
        return state
