"""LISA-response Python frontends.

Absorbed from ``lisa-on-gpu`` / ``fastlisaresponse`` at Phase 3 of the
sprint reorg. Public symbols:

- :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`: the
  backend-dispatch base (response classes inherit it directly;
  ``FastLISAResponseParallelModule`` / ``LISAToolsResponseParallelModule``
  are backward-compat aliases of it, see :mod:`lisatools.response.parallelbase`).
- :class:`~lisatools.response.tdiconfig.TDIConfig`: the TDI combination
  configuration (``"1st generation"`` / ``"2nd generation"`` or a custom list).
- :class:`~lisatools.response.directresponse.pyResponseTDI` /
  :class:`~lisatools.response.directresponse.ResponseWrapper` /
  :func:`~lisatools.response.directresponse.ecliptic_to_icrs`: the direct
  time-domain projection + TDI response (was ``fastlisaresponse.response``).
- :mod:`lisatools.response.tdionfly`: the TDI-on-the-fly generator family
  (was ``fastlisaresponse.tdionfly``).

The backends these classes dispatch to (``lisatools_cpu``,
``lisatools_cuda{11,12,13}x``, ``lisatools_jax``) are the ones registered by
``lisatools/__init__``; choose one per object with ``force_backend=``.
"""

# ruff: noqa: E402,F401
# Phase 3L.7k (2026-06-04): fastlisaresponse no longer registers any
# backends -- LISA-response wraps live on lisatools.cutils.LISAToolsBackend
# directly. The previous force-registration ``import fastlisaresponse``
# block was removed at Phase 3L.7l.

from .parallelbase import (
    FastLISAResponseParallelModule,
    LISAToolsParallelModule,
    LISAToolsResponseParallelModule,
)
from .tdiconfig import TDIConfig
from .directresponse import ecliptic_to_icrs, pyResponseTDI, ResponseWrapper
from . import tdionfly

__all__ = [
    "LISAToolsParallelModule",
    "FastLISAResponseParallelModule",
    "LISAToolsResponseParallelModule",
    "TDIConfig",
    "ecliptic_to_icrs",
    "pyResponseTDI",
    "ResponseWrapper",
    "tdionfly",
]
