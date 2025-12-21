import numpy as np
import time

import unittest

try:
    import cupy as cp
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from lisatools.utils.constants import *
from lisatools import detector as lisa

import sys

class DetectorTest(unittest.TestCase):
    def test_orbits(self):
        xp = cp if gpu_available else np
        force_backend = "cpu" if not gpu_available else "cuda12x"
        orbits = lisa.DefaultOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)
        dt = 100.0
        _t = xp.arange(0.0, YRSID_SI, dt)
        t_arr_links = xp.tile(_t, (len(orbits.LINKS), 1)).flatten()
        links = xp.repeat(xp.asarray(orbits.LINKS), len(_t))
        breakpoint()
        normal_vec = orbits.get_normal_unit_vec(t_arr_links, links)
        ltt = orbits.get_light_travel_times(t_arr_links, links)

        t_arr_sc = xp.tile(_t, (3, 1)).flatten()
        sc = xp.repeat(xp.array([1, 2, 3]), len(_t))
        pos = orbits.get_pos(t_arr_sc, sc)

        self.assertFalse(xp.any(xp.isnan(normal_vec)))
        self.assertFalse(xp.any(xp.isnan(ltt)))
        self.assertFalse(xp.any(xp.isnan(pos)))