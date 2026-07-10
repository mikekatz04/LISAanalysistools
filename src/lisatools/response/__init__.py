"""LISA-response Python frontends.

Absorbed from ``lisa-on-gpu`` / ``fastlisaresponse`` at Phase 3 of the
sprint reorg. Public symbols:

- :class:`LISAToolsParallelModule` — backend-dispatch base (response
  classes inherit this directly; ``FastLISAResponseParallelModule`` /
  ``LISAToolsResponseParallelModule`` are backward-compat aliases of it)
- :class:`TDIConfig`                       — TDI generation config
- :class:`Response` / :class:`ResponseWrapper` / :func:`pyResponseTDI`
  — direct (frequency-domain) response (was ``fastlisaresponse.response``)
- :class:`TDIonTheFly` family               — on-the-fly TDI generators
  (was ``fastlisaresponse.tdionfly``)

Backend registration (``fastlisaresponse_cpu``, ``..._cuda{11,12,13}x``,
``..._jax``) still lives in ``fastlisaresponse.__init__``. We trigger it
on first import so users of ``lisatools.response`` do not need to know.
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
