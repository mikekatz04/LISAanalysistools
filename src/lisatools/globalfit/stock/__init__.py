"""Stock (installed, ready-to-adjust) global-fit configurations.

Families of stock fits live in subpackages (currently
:mod:`lisatools.globalfit.stock.erebor`); the family-agnostic machinery —
the deferred-build :class:`StockGlobalFit` base, the declarative recipe
layer, and the option registry — lives in :mod:`.base` and is re-exported
here.
"""

from .base import (
    MoveBuildContext,
    MoveSpec,
    RecipeSpec,
    StageSpec,
    StockGlobalFit,
    StockRegistry,
    env_default,
    env_resolve,
    materialize_recipe,
)

__all__ = [
    "MoveBuildContext",
    "MoveSpec",
    "RecipeSpec",
    "StageSpec",
    "StockGlobalFit",
    "StockRegistry",
    "env_default",
    "env_resolve",
    "materialize_recipe",
]
