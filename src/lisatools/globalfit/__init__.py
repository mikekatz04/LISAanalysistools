"""The LISA global-fit pipeline.

User-facing recipe-layer API (one concept per level, each with a required
``setup(ctx)`` hook run at materialization):

* :class:`Move` — a move IS a proposal. The base class resolves its name
  against the variant's stock moves; subclass it and override ``setup(ctx)``
  to build a custom move.
* :class:`FunctionMove` — wraps a plain ``fn(model, state) -> (new_state,
  accepted)`` function (created automatically when you ``add_move`` a
  callable).
* :class:`Stage` — one phase of the recipe (its moves propose together).
* :class:`Recipe` — the declarative stage list AND the runtime step engine,
  one object (``fit.recipe`` is literally what runs).
* :class:`MoveBuildContext` — what every ``setup(ctx)`` receives.

Stock (installed, ready-to-adjust) run configurations live under
:mod:`lisatools.globalfit.stock`.
"""

from .moves import FunctionMove, Move, MoveBuildContext
from .recipe import Recipe, Stage

__all__ = [
    "FunctionMove",
    "Move",
    "MoveBuildContext",
    "Recipe",
    "Stage",
]
