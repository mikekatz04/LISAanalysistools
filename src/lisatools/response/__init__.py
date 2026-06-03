"""LISA-response Python frontends.

Absorbed from ``lisa-on-gpu`` / ``fastlisaresponse`` at Phase 3 of the
sprint reorg. Public symbols:

- :class:`FastLISAResponseParallelModule` — backend-dispatch base
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
try:
    import fastlisaresponse  # noqa: F401 -- forces backend registration
except (ImportError, ModuleNotFoundError):
    # fastlisaresponse is being deprecated; if it is not installed,
    # nothing prevents users from instantiating the moved classes --
    # they will simply get a `BackendUnavailableException` listing the
    # missing fastlisaresponse_<flavor> backend, which is the correct
    # diagnostic in that case.
    pass

from .parallelbase import FastLISAResponseParallelModule
from .tdiconfig import TDIConfig
from .directresponse import ecliptic_to_icrs, pyResponseTDI, ResponseWrapper
from . import tdionfly

__all__ = [
    "FastLISAResponseParallelModule",
    "TDIConfig",
    "ecliptic_to_icrs",
    "pyResponseTDI",
    "ResponseWrapper",
    "tdionfly",
]
