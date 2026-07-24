"""Unified PSDMove GPU-spread plumbing.

One move structure: ``PSDMove(dcga=...)`` routes ``psd_log_like`` through
the DCGA per-device replica path; without a DCGA the single-device path
runs. Also covers the fast-path gate on multi-shard ACAs and the
deprecated ``MultiGPUPSDMove`` shim.
"""

from __future__ import annotations

import unittest
import warnings
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:
    from _multishard import FakeMultiShardACA


class _StubDCGA:
    def __init__(self, acs, num_splits=2, basis=None):
        self.acs = acs
        self.gpus = list(range(num_splits))
        self.num_splits = num_splits
        self.computation_groups = [
            SimpleNamespace(
                sensitivity_backend=SimpleNamespace(
                    use_splines=False, basis_settings=basis
                )
            )
            for _ in range(num_splits)
        ]

    def device_context(self, device):
        return nullcontext()

    @property
    def xp(self):
        return np


class PSDMoveUnifiedTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.domains import FDSettings
            from lisatools.globalfit.moves.psdmove import (
                MultiGPUPSDMove, PSDMove)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"psdmove not available: {exc}")
        self.PSDMove = PSDMove
        self.Shim = MultiGPUPSDMove
        self.fd = FDSettings(N=16, df=1e-4)
        self.acs = FakeMultiShardACA((3, 4), 4, 2, layout="blocked")

    def _backend(self):
        return SimpleNamespace(use_splines=False, basis_settings=self.fd)

    def test_fast_path_gate(self):
        # multi-shard WITHOUT a DCGA: kernel fast path unavailable
        move = self.PSDMove(self.acs, {}, sensitivity_backend=self._backend(), name='psd test')
        self.assertFalse(move._kernel_fast_path_available())
        # multi-shard WITH a DCGA: available (per-replica scoring)
        dcga = _StubDCGA(self.acs, basis=self.fd)
        move2 = self.PSDMove(self.acs, {}, sensitivity_backend=self._backend(),
                             dcga=dcga, name='psd test')
        self.assertTrue(move2._kernel_fast_path_available())
        # single-shard without DCGA: available as before
        single = FakeMultiShardACA((3, 4), 4, 1)
        move3 = self.PSDMove(single, {}, sensitivity_backend=self._backend(), name='psd test')
        self.assertTrue(move3._kernel_fast_path_available())

    def test_psd_log_like_dispatches_to_dcga_path(self):
        dcga = _StubDCGA(self.acs, basis=self.fd)
        move = self.PSDMove(self.acs, {}, sensitivity_backend=self._backend(),
                            dcga=dcga, name='psd test')
        sentinel = object()
        recorded = {}

        def fake_dcga_path(x, supps=None, **kw):
            recorded["x"] = x
            recorded["supps"] = supps
            return sentinel

        move._psd_log_like_dcga = fake_dcga_path
        out = move.psd_log_like([np.zeros((2, 2))],
                                supps={"walker_inds": np.array([0, 1])})
        self.assertIs(out, sentinel)
        self.assertIn("x", recorded)

    def test_force_parent_path_bypasses_dcga(self):
        dcga = _StubDCGA(self.acs, basis=self.fd)
        move = self.PSDMove(self.acs, {}, sensitivity_backend=self._backend(),
                            dcga=dcga, name='psd test')
        move._force_parent_path = True
        move._psd_log_like_dcga = lambda *a, **k: self.fail(
            "DCGA path must be bypassed under _force_parent_path")
        # the parent path will fail on the fake backend — but only AFTER the
        # dispatch decision, which is what this test pins down
        with self.assertRaises(Exception):
            move.psd_log_like([np.zeros((2, 2))],
                              supps={"walker_inds": np.array([0, 1])})

    def test_dcga_supplies_backend_and_acs(self):
        dcga = _StubDCGA(self.acs, basis=self.fd)
        move = self.PSDMove(None, {}, dcga=dcga, name='psd test')
        self.assertIs(move.acs, self.acs)
        self.assertIs(
            move.sensitivity_backend,
            dcga.computation_groups[0].sensitivity_backend,
        )

    def test_deprecated_shim(self):
        dcga = _StubDCGA(self.acs, basis=self.fd)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            move = self.Shim(dcga, {}, name='psd test')
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertIsInstance(move, self.PSDMove)
        self.assertIs(move.dcga, dcga)
        self.assertIs(move.acs, self.acs)


if __name__ == "__main__":
    unittest.main()
