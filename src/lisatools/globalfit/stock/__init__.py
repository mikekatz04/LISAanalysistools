"""Stock (installed, ready-to-adjust) global-fit configurations.

Families of stock fits live in subpackages (currently
:mod:`lisatools.globalfit.stock.erebor`); the family-agnostic machinery —
the deferred-build :class:`StockGlobalFit` base and the option registry —
lives in :mod:`.base` and is re-exported here. The recipe-layer classes
(``Move``/``FunctionMove``/``Stage``/``Recipe``/``MoveBuildContext``) are
general global-fit machinery and live at :mod:`lisatools.globalfit`.
"""

from .base import (
    StockGlobalFit,
    StockRegistry,
    env_default,
    env_resolve,
)

__all__ = [
    "StockGlobalFit",
    "StockRegistry",
    "env_default",
    "env_resolve",
]
