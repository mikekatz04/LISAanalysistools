"""``L1Orbits(ltt_stride=...)``: the light-travel-time table read strided.

A two-year Mojito brick tabulates the six link delays every 2.5 s (1.2 GB).
They change on orbital timescales and the backend interpolates them
linearly, so callers that do not need the full cadence (the LATW 09
tutorial, the monitor's truth build) kept one sample in 200 by hand after
construction. ``ltt_stride`` does that at read time. A fake L1 file stands
in for mojito so the test runs without the data.
"""

import contextlib
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.detector import L1Orbits


class _Sampling:
    def __init__(self, t0, dt, size):
        self.t0, self.dt, self.size = t0, dt, size

    def t(self, idx=None):
        t = self.t0 + self.dt * np.arange(self.size)
        return t if idx is None else t[idx]


def _fake_file(n_ltt=4001, n_pos=401, t0=1.0e6):
    t_ltt = _Sampling(t0, 2.5, n_ltt)
    t_pos = _Sampling(t0, 25.0, n_pos)
    phase = 2 * np.pi * t_ltt.t() / 3.15e7
    ltt = 8.34 + 0.08 * np.sin(phase[:, None] + np.arange(6))
    ang = 2 * np.pi * t_pos.t() / 3.15e7
    sc = np.arange(3) * 2 * np.pi / 3
    pos = 1.5e11 * np.stack([np.cos(ang[:, None] + 1e-2 * np.cos(sc)),
                             np.sin(ang[:, None] + 1e-2 * np.sin(sc)),
                             1e-3 * np.cos(sc) + 0 * ang[:, None]], axis=-1)
    vel = np.gradient(pos, t_pos.t(), axis=0)
    return types.SimpleNamespace(
        ltts=types.SimpleNamespace(ltts=ltt, time_sampling=t_ltt),
        orbits=types.SimpleNamespace(positions=pos, velocities=vel, time_sampling=t_pos))


class L1OrbitsLttStrideTest(unittest.TestCase):
    def setUp(self):
        self.f = _fake_file()
        p = mock.patch.object(L1Orbits, "open",
                              lambda _self: contextlib.nullcontext(self.f))
        p.start()
        self.addCleanup(p.stop)
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        self.path = os.path.join(d, "fake_l1.h5")   # the file must exist; open() is faked
        open(self.path, "wb").close()

    def _orbits(self, **kw):
        return L1Orbits(self.path, force_backend="cpu", frame="icrs", **kw)

    def test_the_table_and_its_times_are_read_strided(self):
        orb = self._orbits(ltt_stride=4)
        np.testing.assert_array_equal(orb.ltt, self.f.ltts.ltts[::4])
        np.testing.assert_array_equal(orb.ltt_t, self.f.ltts.time_sampling.t()[::4])
        self.assertEqual(orb.ltt_dt, 10.0)
        self.assertEqual(orb.ltt_t0, self.f.ltts.time_sampling.t0)

    def test_the_default_reads_every_sample(self):
        orb = self._orbits()
        np.testing.assert_array_equal(orb.ltt, self.f.ltts.ltts)
        self.assertEqual(orb.ltt_dt, 2.5)

    def test_the_stride_survives_a_rebuild_from_kwargs(self):
        self.assertEqual(self._orbits(ltt_stride=4).kwargs["ltt_stride"], 4)

    def test_a_stride_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            self._orbits(ltt_stride=0)


if __name__ == "__main__":
    unittest.main()
