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
    def test_get_sen(self):
        xp = cp if gpu_available else np
        
        frqs = xp.logspace(-5., 0., 1000)
        Sn = get_sensitivity(frqs, sens_fn="X1TDISens", model=lisa.sangria)

        self.assertFalse(xp.any(xp.isnan(Sn)))

    def _test_sens_mat(self, sens_mat_class, model):

        xp = cp if gpu_available else np
        
        # TODO: improve this
        force_backend = "gpu" if gpu_available else "cpu"
        frqs = xp.logspace(-5., 0., 1000)
        Sn = sens_mat_class(frqs, model=model)

    def test_sensitivity_matrix_AET1(self):
        self._test_sens_mat(AET1SensitivityMatrix, lisa.sangria)

    def test_sensitivity_matrix_AET2(self):
        self._test_sens_mat(AET2SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_XYZ1(self):
        self._test_sens_mat(XYZ1SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_XYZ2(self):
        self._test_sens_mat(XYZ2SensitivityMatrix, lisa.sangria_v2)

    def test_sensitivity_matrix_AE1(self):
        self._test_sens_mat(AE1SensitivityMatrix, lisa.sangria)

    def test_sensitivity_matrix_AE2(self):
        self._test_sens_mat(AE2SensitivityMatrix, lisa.sangria_v2)

