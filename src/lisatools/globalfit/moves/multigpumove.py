"""
multigpumove.py
=============

Base move to perform likelihood evaluations on multiple devices. 
"""
from __future__ import annotations

from logging import getLogger
import numpy as np
from typing import TYPE_CHECKING

from ...domaincomputation import DomainComputationGroupArray

logger = getLogger(__name__)


class MultiGPUMoveBase:
    def __init__(self, dcga: DomainComputationGroupArray = None, run_async: bool = False, run_threaded: bool = False, *, acs=None):
        # The C++ likelihood coordinator now lives on ``AnalysisContainerArray``
        # (DCGA was absorbed). Accept either an ACA or a (deprecated)
        # ``DomainComputationGroupArray`` shim at the constructor boundary so
        # external settings files that still pass ``dcga=`` keep working, and
        # resolve both to the real ACA. We no longer store the DCGA itself.
        resolved = acs if acs is not None else dcga
        self.acs = resolved.acs if hasattr(resolved, "acs") else resolved
        self._run_async = run_async
        self._run_threaded = run_threaded

    @property
    def dcga(self) -> DomainComputationGroupArray:
        """Transitional handle to the ACA-owned cpp coordinator shim.

        Forwards to ``self.acs``; removed in Phase C once every move drives the
        ACA directly via ``self.acs``.
        """
        return self.acs.cpp_likelihood_backend

    @property
    def run_async(self) -> bool:
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        return self._run_threaded

    @property
    def xp(self):
        """Return the array library (numpy or cupy) used by the analysis container array."""
        return self.acs.xp