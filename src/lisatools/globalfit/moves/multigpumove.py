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
    def __init__(self, dcga: DomainComputationGroupArray):
        self._dcga = dcga

    @property
    def dcga(self) -> DomainComputationGroupArray:
        return self._dcga

    @property
    def xp(self):
        """Return the array library (numpy or cupy) used by the DomainComputationGroupArray."""
        return self.dcga.xp