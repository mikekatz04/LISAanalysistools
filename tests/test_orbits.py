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
from lisatools.detector import EqualArmlengthOrbits

import sys


force_backend = "gpu" if gpu_available else "cpu"


class OrbitsTest(unittest.TestCase):
    def test_orbits(self):
        xp = cp if gpu_available else np
        
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)
        new_t = xp.linspace(orbits.t_base.min(), orbits.t_base.max(), 1000)
        sc = 1
        new_pos = orbits.get_pos(new_t, sc)

        self.assertFalse(xp.any(xp.isnan(new_pos)))