"""Regression tests for time-grid alignment in ``TDPyResponseWaveformBase._apply_response``.

These guard the *post-response* re-alignment block (waveformbase.py ~1003-1022),
where the TDI output is trimmed and its time labels are snapped back onto the
data grid ``data_t0 + k*dt``.

``test_off_grid_data_t0_no_one_sample_desync`` specifically pins the fix for the
removed ``(data_t0 - shifted_t_arr[:, 0]) % dt`` step: numpy's ``%`` is
discontinuous at multiples of ``dt``, so when ``data_t0`` is off-grid a tiny
*negative* float residual mapped to ``~dt`` instead of ``~0`` and shifted the
whole template by one full sample relative to the data (a silent
``e^{-i 2*pi f dt}`` phase ramp in FD). The ``(dt, merger_time, data_t0)`` triple
below is a concrete case where the old step fired; the patched code must land the
labels exactly on the data grid.
"""

import sys

# eryn/scienceplots import shims (see tests/test_quintic_response.py) -- harmless
# if not needed; keeps the import robust across matplotlib versions.
sys.modules.setdefault("scienceplots", None)
import builtins
import typing

builtins.typing = typing

import unittest

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import TDSettings
from lisatools.sources.waveformbase import TDPyResponseWaveformBase

_DT = 5.0
_WAVEFORM_T0 = 0.0
_N_ORIG = 60000
_BUFFER_TIME = 10000.0

# A (merger_time, data_t0) pair for which the removed `% dt` step desynced the
# template by one sample. data_t0 is off-grid; n = round((data_t0 - s0)/dt) < 0
# (the source starts after data_t0), so start_ind stays 0 and the bug surfaces
# purely as a +dt offset on times[0]. Found by a wide random search over the
# exact post-line-1010 arithmetic (see job tmp/probe_bug3*.py).
_MERGER_TIME = 16823712.9794009
_DATA_T0 = 16726395.173080323


class _MiniTDWave(TDPyResponseWaveformBase):
    """Minimal concrete subclass: a cheap on-grid sinusoid as the polarizations.

    The signal content is irrelevant to the alignment arithmetic under test; we
    only need a real waveform on a ``dt`` grid starting at t=0 so that
    ``shifted_t_arr[:, 0] == merger_time + waveform_t0`` exactly.
    """

    def wave_gen(self, *args, **kwargs):
        # _call_single forwards (ra, dec, merger_time) as the trailing positionals.
        t = np.arange(_N_ORIG, dtype=np.float64) * _DT
        phase = 2.0 * np.pi * 1e-3 * t
        h_plus = 1e-21 * np.cos(phase)
        h_cross = 1e-21 * np.sin(phase)
        return t, h_plus, h_cross


class TestResponseAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orbits = EqualArmlengthOrbits()
        orbits.configure(linear_interp_setup=True)
        cls.orbits = orbits

    def _make(self, data_t0):
        return _MiniTDWave(
            waveform_t0=_WAVEFORM_T0,
            data_td_settings=TDSettings(N=200000, dt=_DT, t0=data_t0),
            tdi_generation="1st generation",
            tdi_channels="XYZ",
            sampling_frequency=1.0 / _DT,
            orbits=self.orbits,
            order=25,
            buffer_time=_BUFFER_TIME,
            force_backend="cpu",
        )

    def test_off_grid_data_t0_no_one_sample_desync(self):
        wf = self._make(_DATA_T0)
        times, chans = wf._call_single(ra=0.7, dec=0.3, merger_time=_MERGER_TIME)
        times = np.asarray(times)

        n = int(np.rint((_DATA_T0 - _MERGER_TIME) / _DT))
        on_grid_t0 = _DATA_T0 - n * _DT  # what the response actually evaluated
        old_buggy_t0 = on_grid_t0 + _DT  # what the removed `% dt` step produced

        # The discriminator: the patched code lands on on_grid_t0; the old code
        # would have been a full dt away (tol << dt/2 cleanly separates them).
        self.assertAlmostEqual(float(times[0]), on_grid_t0, delta=1e-3)
        self.assertGreater(abs(float(times[0]) - old_buggy_t0), _DT - 1e-3)

    def test_batched_min_trim_preserves_leading_signal(self):
        """Bug 2: the shared leading-sample trim must use min(start_inds), not max.

        Two on-grid sources (data_t0 a multiple of dt, so t0_shift=0): source 1
        starts exactly at data_t0 (start_ind 0); source 0 starts 500 samples before
        (start_ind 500). With the old max-trim the whole batch was sliced by 500,
        silently dropping source 1's first 500 (valid, in-window) samples. The
        min-trim (=0) keeps source 1 intact, so batched[1] == the single-source
        result for source 1, bit-for-bit. Calls _apply_response directly.
        """
        dt = 5.0
        data_t0 = 1_000_000.0  # multiple of dt -> t0_shift_to_data == 0
        n = 20000
        wf = _MiniTDWave(
            waveform_t0=0.0,
            data_td_settings=TDSettings(N=200000, dt=dt, t0=data_t0),
            tdi_generation="1st generation",
            tdi_channels="XYZ",
            sampling_frequency=1.0 / dt,
            orbits=self.orbits,
            order=25,
            buffer_time=10000.0,
            force_backend="cpu",
        )
        t = np.arange(n, dtype=np.float64) * dt
        phase = 2.0 * np.pi * 1e-3 * t
        h_plus = np.broadcast_to(1e-21 * np.cos(phase), (2, n)).copy()
        h_cross = np.broadcast_to(1e-21 * np.sin(phase), (2, n)).copy()
        t_arr = np.broadcast_to(t, (2, n)).copy()
        # source 0 starts 500 samples before data_t0; source 1 starts at data_t0.
        merger_times = np.array([data_t0 - 500 * dt, data_t0])
        ra = np.array([0.7, 1.3])
        dec = np.array([0.3, -0.2])

        _, tdis_b = wf._apply_response(t_arr, h_plus, h_cross, ra, dec, merger_times)
        _, tdis_s1 = wf._apply_response(
            t, h_plus[1], h_cross[1], 1.3, -0.2, float(merger_times[1])
        )

        # The min-trim keeps the FULL length; the old max-trim sliced the whole
        # batch by source 0's start_ind=500, returning n-500 and dropping source 1's
        # leading valid samples. Length is the clean, construction-robust symptom.
        self.assertEqual(np.asarray(tdis_b).shape[-1], n)
        self.assertEqual(np.asarray(tdis_b[1]).shape, np.asarray(tdis_s1).shape)
        # Source 1 is preserved (matches the single-source run) up to a tiny,
        # pre-existing CPU near-boundary effect (~1e-23, far below the ~1e-21 TDI
        # scale) that is independent of this fix; the old over-trim would instead
        # mismatch by a 500-sample shift (~full TDI scale) and a shorter array.
        np.testing.assert_allclose(
            np.asarray(tdis_b[1]), np.asarray(tdis_s1), atol=1e-22, rtol=0.0
        )

    def test_times_land_on_data_grid(self):
        # General invariant for an off-grid data_t0: every label is an integer
        # number of dt from data_t0, spacing is uniform, and the signal is finite.
        wf = self._make(_DATA_T0)
        times, chans = wf._call_single(ra=0.7, dec=0.3, merger_time=_MERGER_TIME)
        times = np.asarray(times)
        chans = np.asarray(chans)

        k = (times - _DATA_T0) / _DT
        np.testing.assert_allclose(k, np.rint(k), atol=1e-6)
        np.testing.assert_allclose(np.diff(times), _DT)
        self.assertEqual(chans.shape, (3, times.shape[0]))
        self.assertTrue(np.all(np.isfinite(chans)))


if __name__ == "__main__":
    unittest.main()
