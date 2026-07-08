"""Tests for :class:`lisatools.detector.EqualArmlengthOrbits` interpolation."""

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
    """Sanity checks for :class:`EqualArmlengthOrbits` spacecraft position queries."""

    def test_orbits(self):
        """Verify interpolated spacecraft positions are finite across the orbit time base."""
        xp = cp if gpu_available else np
        
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        # configuration is lazy (first use); no explicit configure() needed
        new_t = xp.linspace(orbits.t_base.min(), orbits.t_base.max(), 1000)
        sc = 1
        new_pos = orbits.get_pos(new_t, sc)

        self.assertFalse(xp.any(xp.isnan(new_pos)))
    def test_configure_deprecated_but_functional(self):
        """Explicit configure() warns but still (re)builds the grid."""
        import warnings

        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        self.assertFalse(orbits.configured)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            orbits.configure(linear_interp_setup=True)
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        self.assertTrue(orbits.configured)

    def test_lazy_configure_on_first_use(self):
        """Accessing a configured-grid quantity triggers configuration."""
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        self.assertFalse(orbits.configured)
        _ = orbits.pycppdetector_args
        self.assertTrue(orbits.configured)
