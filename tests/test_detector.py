"""Tests for :mod:`lisatools.detector` orbit and geometry queries."""

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
    """Sanity checks for :class:`lisatools.detector.DefaultOrbits`."""

    def test_orbits(self):
        """Verify normal vectors, light travel times, and positions are finite over a year."""
        xp = cp if gpu_available else np
        force_backend = "cpu" if not gpu_available else "cuda12x"
        orbits = lisa.DefaultOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)
        dt = 100.0
        _t = xp.arange(0.0, YRSID_SI, dt)
        t_arr_links = xp.tile(_t, (len(orbits.LINKS), 1)).flatten()
        links = xp.repeat(xp.asarray(orbits.LINKS), len(_t))
        normal_vec = orbits.get_normal_unit_vec(t_arr_links, links)
        ltt = orbits.get_light_travel_times(t_arr_links, links)
        import lisatools
        _backend = lisatools.get_backend(force_backend)
        t_arr_sc = xp.tile(_t, (3, 1)).flatten()
        sc = xp.repeat(xp.array([1, 2, 3]), len(_t))
        pos = orbits.get_pos(t_arr_sc, sc)
        vel = orbits.get_vel(t_arr_sc, sc)

        self.assertFalse(xp.any(xp.isnan(normal_vec)))
        self.assertFalse(xp.any(xp.isnan(ltt)))
        self.assertFalse(xp.any(xp.isnan(pos)))
        self.assertFalse(xp.any(xp.isnan(vel)))

    def test_get_vel(self):
        """get_vel interpolates the stored velocity grid the way get_pos does for positions."""
        xp = cp if gpu_available else np
        force_backend = "cpu" if not gpu_available else "cuda12x"
        orbits = lisa.DefaultOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        sc_t = xp.asarray(orbits.sc_t)
        v_grid = xp.asarray(orbits.v)  # (N, 3_sc, 3_xyz)
        i = len(sc_t) // 2

        for sc in [1, 2, 3]:
            # node exactness: interpolation at a grid node returns the node value
            t_node = float(sc_t[i])
            vel_node = orbits.get_vel(t_node, sc)
            self.assertEqual(vel_node.shape, (3,))
            xp.testing.assert_allclose(vel_node, v_grid[i, sc - 1, :], rtol=1e-12, atol=0.0)

            # linearity: a midpoint between two nodes == average of the node velocities
            t_mid = 0.5 * (float(sc_t[i]) + float(sc_t[i + 1]))
            vel_mid = orbits.get_vel(t_mid, sc)
            expected_mid = 0.5 * (v_grid[i, sc - 1, :] + v_grid[i + 1, sc - 1, :])
            xp.testing.assert_allclose(vel_mid, expected_mid, rtol=1e-9, atol=0.0)

        # input-mode shapes mirror get_pos
        t_arr = xp.asarray([float(sc_t[i]), float(sc_t[i + 1]), float(sc_t[i + 2])])
        self.assertEqual(orbits.get_vel(t_arr, 2).shape, (3, 3))  # (ndarray, int)
        sc_arr = xp.asarray([1, 2, 3], dtype=np.int32)
        out_arr = orbits.get_vel(t_arr, sc_arr)  # (ndarray, ndarray)
        self.assertEqual(out_arr.shape, (3, 3))
        self.assertFalse(xp.any(xp.isnan(out_arr)))

        # out-of-range spacecraft ids raise instead of silently returning zeros
        with self.assertRaises(ValueError):
            orbits.get_vel(t_arr, xp.asarray([1, 2, 9], dtype=np.int32))
