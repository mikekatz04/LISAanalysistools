"""Legacy base class for moves that evaluate likelihoods on multiple devices.

Holds :class:`MultiGPUMoveBase`, a small holder of a
:class:`~lisatools.domaincomputation.DomainComputationGroupArray` (DCGA) and its
async / threaded flags. Nothing subclasses or imports it any more: the 2026-07
multi-GPU pass moved the DCGA plumbing into the moves themselves (the ``dcga``
argument of ``PSDMove`` and ``ResidualAddOneRemoveOneMove``). Not on the
9mo / 12mo production path; kept only because ``docs/source/user/globalfitmoves.rst``
documents the module.
"""
from __future__ import annotations

from logging import getLogger
import numpy as np
from typing import TYPE_CHECKING

from ...domaincomputation import DomainComputationGroupArray

logger = getLogger(__name__)


# TODO check: keep/remove/document -- no subclasses or importers in src/scripts/tests/LATW; only the rst automodule; superseded by the dcga keyword of PSDMove / ResidualAddOneRemoveOneMove
class MultiGPUMoveBase:
    """Holder of a DCGA plus its ``run_async`` / ``run_threaded`` flags (legacy, unused)."""

    def __init__(self, dcga: DomainComputationGroupArray, run_async: bool = False, run_threaded: bool = False):
        self._dcga = dcga
        self._run_async = run_async
        self._run_threaded = run_threaded

    @property
    def dcga(self) -> DomainComputationGroupArray:
        return self._dcga
    
    @property
    def run_async(self) -> bool:
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        return self._run_threaded

    @property
    def xp(self):
        """Return the array library (numpy or cupy) used by the DomainComputationGroupArray."""
        return self.dcga.xp