"""Tests for :mod:`lisatools.sensitivity` PSD and sensitivity-matrix builders."""

import numpy as np
import time

import unittest

try:
    import cupy as cp
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from lisatools.sensitivity import (
    get_sensitivity,
    AET1SensitivityMatrix,
    XYZ1SensitivityMatrix,
    AET2SensitivityMatrix,
    XYZ2SensitivityMatrix,
    AE1SensitivityMatrix,
    AE2SensitivityMatrix,
)
from lisatools.utils.constants import *
from lisatools import detector as lisa

import sys

class SensitivityTest(unittest.TestCase):
    """Sanity checks for :func:`get_sensitivity` and the TDI sensitivity-matrix classes."""

    def test_get_sen(self):
        """Verify :func:`get_sensitivity` returns finite values for the X1 TDI PSD."""
        xp = cp if gpu_available else np
        
        frqs = xp.logspace(-5., 0., 1000)
        Sn = get_sensitivity(frqs, sens_fn="X1TDISens", model=lisa.sangria)

        self.assertFalse(xp.any(xp.isnan(Sn)))

    def _test_sens_mat(self, sens_mat_class, model):
        """Construct ``sens_mat_class`` over a log-spaced frequency grid for ``model``."""

        xp = cp if gpu_available else np
        
        # TODO: improve this
        force_backend = "gpu" if gpu_available else "cpu"
        frqs = xp.logspace(-5., 0., 1000)
        Sn = sens_mat_class(frqs, model=model)

    def test_sensitivity_matrix_AET1(self):
        """Build an :class:`AET1SensitivityMatrix` for the ``sangria`` noise model."""
        self._test_sens_mat(AET1SensitivityMatrix, lisa.sangria)

    def test_sensitivity_matrix_AET2(self):
        """Build an :class:`AET2SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(AET2SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_XYZ1(self):
        """Build an :class:`XYZ1SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(XYZ1SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_XYZ2(self):
        """Build an :class:`XYZ2SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(XYZ2SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_AE1(self):
        """Build an :class:`AE1SensitivityMatrix` for the ``sangria`` noise model."""
        self._test_sens_mat(AE1SensitivityMatrix, lisa.sangria)

    def test_sensitivity_matrix_AE2(self):
        """Build an :class:`AE2SensitivityMatrix` for the ``sangria_v2`` noise model."""
        self._test_sens_mat(AE2SensitivityMatrix, lisa.sangria_v2)

