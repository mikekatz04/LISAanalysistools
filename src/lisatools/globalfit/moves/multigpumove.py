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