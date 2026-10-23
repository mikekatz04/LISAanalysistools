"""Stock (installed, ready-to-adjust) global-fit configurations.

Families of stock fits live in subpackages (currently
:mod:`lisatools.globalfit.stock.erebor`); the family-agnostic machinery —
the deferred-build :class:`StockGlobalFit` base and the option registry —
lives in :mod:`.base` and is re-exported here. The recipe-layer classes
(``Move``/``FunctionMove``/``Stage``/``Recipe``/``MoveBuildContext``) are
general global-fit machinery and live at :mod:`lisatools.globalfit`.

The production 9mo / 12mo runs build the ``all_sources`` variant of the
erebor family (``erebor.all_sources``; driver
``scripts/fstat_proposal/run_combined_staged.py``).

Example:
    .. code-block:: python

        from lisatools.globalfit.stock import erebor

        print(erebor.get_stock_options())
        fit = erebor.all_sources(nwalkers=4)
        fit.build()
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
