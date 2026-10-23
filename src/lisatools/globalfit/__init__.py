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
:mod:`lisatools.globalfit.stock`. The pipeline below them: a stock fit IS a
:class:`~lisatools.globalfit.run.GlobalFitSetup` (``fit.build()`` loads the
data and builds every branch through :mod:`lisatools.globalfit.engine`);
``fit.run()`` hands it to :class:`~lisatools.globalfit.run.GlobalFit`, whose
:meth:`~lisatools.globalfit.run.GlobalFit.run_global_fit` gives each MPI rank
its role (head / compute / saver) and drives the eryn sampler
(:class:`~lisatools.globalfit.engine.GlobalFitEngine`) through the recipe.
The run state is a :class:`~lisatools.globalfit.state.GFState` (cold-chain
main state plus one tempered sub-state per branch), stored by
:class:`~lisatools.globalfit.hdfbackend.GFHDFBackend`.

Example:
    .. code-block:: python

        from lisatools.globalfit.stock import erebor

        fit = erebor.all_sources(nwalkers=4)   # the 9mo/12mo production variant
        fit.general.num_iterations = 50        # knobs are plain attributes
        fit.build()                            # heavy: data load, branch setup
        fit.run()                              # MPI run (GlobalFit.run_global_fit)
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
