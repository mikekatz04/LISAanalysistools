"""``to_current_device`` -- the multi-GPU cross-device guard used by the
per-walker sensitivity build (2026-09-07: the galfor ``f/f_1`` ufunc raised a
peer-access ValueError on P2P-disabled two-GPU nodes because the cached
frequency nodes lived on GPU0 while a GPU1 walker built in its own device
context). The cross-device GPU behaviour is cluster-validated; here we lock the
CPU contract -- a STRICT no-op -- so the guard can never silently start copying
(or fail) on the CPU / single-GPU path where every array is already "current".
"""
import unittest

import numpy as np

from lisatools.utils.device import to_current_device


class ToCurrentDeviceTest(unittest.TestCase):
    def test_numpy_array_is_identity_noop(self):
        a = np.arange(6.0).reshape(2, 3)
        out = to_current_device(np, a)
        self.assertIs(out, a)  # SAME object: no host round-trip on CPU

    def test_numpy_scalar_and_empty_are_noops(self):
        for a in (np.float64(3.0), np.zeros(0), np.asarray(1.5)):
            self.assertIs(to_current_device(np, a), a)

    def test_non_array_passes_through(self):
        # A plain scalar has no ``.device``; must return unchanged, not raise.
        self.assertEqual(to_current_device(np, 2.5), 2.5)


if __name__ == "__main__":
    unittest.main()
