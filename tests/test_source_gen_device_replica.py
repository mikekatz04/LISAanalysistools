"""Per-device source-generator replicas (multi-GPU MBH parallelism).

Structural CPU coverage for the device-keyed replica mechanism that makes the
MBH source move generate each walker's template on the walker's OWNING device
(no peer access off the primary device):

- ``jax_device_context`` is a safe no-op on CPU / when JAX can't see the device
  (so single-GPU/CPU behaviour is unchanged and importing never hard-fails).
- ``source_runtime._device_local_orbits`` reuses the shared orbits on the run's
  primary device / CPU (zero extra memory, byte-identical to the pre-multi-GPU
  path) and builds ONE cached ``orbits.__class__(*args, **kwargs)`` replica per
  non-primary device (the DCGA build pattern), so the ``id(orbits)``-keyed
  phentax generator cache yields one generator per device.

The actual on-GPU device placement (phentax JAX tables, cupy response, orbit
grids) can only be validated on the cluster; these tests pin the routing.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._multishard import RecordingXp
except ImportError:
    from _multishard import RecordingXp


class JaxDeviceContextTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.utils.device import jax_device_context
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"device helpers not available: {exc}")
        self.jax_device_context = jax_device_context

    def test_none_is_nullcontext(self):
        # CPU / single-device / primary device -> plain no-op, no jax needed.
        with self.jax_device_context(None):
            pass

    def test_missing_device_is_nullcontext(self):
        # A device index JAX cannot possibly have must degrade to a no-op
        # (never raise), so a GPU run with fewer JAX devices than cupy sees,
        # or JAX-on-CPU, keeps working.
        with self.jax_device_context(9999):
            pass


class _FakeOrbits:
    """Minimal orbits stand-in: reconstructable via
    ``__class__(*args, **kwargs)`` (mirrors ``EqualArmlengthOrbits``)."""

    def __init__(self, tag="base"):
        self.tag = tag

    @property
    def args(self):
        return ()

    @property
    def kwargs(self):
        return {"tag": self.tag}


class DeviceLocalOrbitsTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.globalfit.stock.erebor import source_runtime as sr
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"source_runtime not available: {exc}")
        self.sr = sr
        sr._DEVICE_ORBITS_REPLICAS.clear()
        self.addCleanup(sr._DEVICE_ORBITS_REPLICAS.clear)
        self.xp = RecordingXp()
        self.orbits = _FakeOrbits(tag="shared")

    def test_primary_device_reuses_shared(self):
        # current device (RecordingXp default 0) == primary 0 -> shared orbits,
        # no replica built.
        rep = self.sr._device_local_orbits(
            self.orbits, self.xp, primary_device=0)
        self.assertIs(rep, self.orbits)
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 0)

    def test_cpu_none_reuses_shared(self):
        # numpy xp -> current_device is None -> shared orbits, no replica.
        rep = self.sr._device_local_orbits(
            self.orbits, np, primary_device=None)
        self.assertIs(rep, self.orbits)
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 0)

    def test_nonprimary_builds_and_caches_replica(self):
        with self.xp.cuda.Device(1):  # current device 1 != primary 0
            rep1 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
            rep2 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
        self.assertIsNot(rep1, self.orbits)              # distinct replica
        self.assertIsInstance(rep1, _FakeOrbits)
        self.assertEqual(rep1.kwargs["tag"], "shared")   # rebuilt from kwargs
        self.assertIs(rep1, rep2)                        # built once, cached
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 1)
        # replica construction was routed through the device-1 context
        self.assertIn(1, self.xp.device_log)

    def test_distinct_replica_per_device(self):
        with self.xp.cuda.Device(1):
            rep1 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
        with self.xp.cuda.Device(2):
            rep2 = self.sr._device_local_orbits(
                self.orbits, self.xp, primary_device=0)
        self.assertIsNot(rep1, rep2)
        self.assertEqual(len(self.sr._DEVICE_ORBITS_REPLICAS), 2)


if __name__ == "__main__":
    unittest.main()
