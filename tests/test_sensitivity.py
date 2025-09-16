import numpy as np
import time

import unittest

try:
    import cupy as cp

    cp.cuda.runtime.setDevice(0)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from lisatools.sensitivity import get_sensitivity, AET1SensitivityMatrix, XYZ1SensitivityMatrix
from lisatools.utils.constants import *
from lisatools.detector import scirdv1

import sys

class SensitivityTest(unittest.TestCase):
    def test_circ(self):
        xp = cp if gpu_available else np
        
        # TODO: improve this
        force_backend = "gpu" if gpu_available else "cpu"
        frqs = xp.logspace(-5., 0., 1000)
        Sn = get_sensitivity(frqs, model=scirdv1)

        self.assertFalse(xp.any(xp.isnan(Sn)))