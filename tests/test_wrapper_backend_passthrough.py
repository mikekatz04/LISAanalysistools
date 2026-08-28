"""TD wrapper backend contract: ``raw_td`` must NOT pull the waveform to host.

Root cause of the 2026-08-27 job-353 crash (6-mo sources probe, first time
the SOBBH branch ever executed -- it had been last in the move order and
every prior job was preempted before reaching it)::

    File "lisatools/domains.py", line 804, in fft
      fd_arr = self.xp.fft.rfft(self.arr * window, axis=-1) * self.dt
    TypeError: Unsupported type <class 'numpy.ndarray'>

``SOBBHTDIonFlyWaveWrap.raw_td`` forced the generator's output to the host
with ``asnumpy``, but ``__call__`` then builds ``TDSignal(arr,
self.td_settings)`` with GPU-backed settings. ``TDSignal`` is
``(DomainBase, TDSettings)``, so its ``xp``/``backend`` come from the
SETTINGS while the array is stored unconverted -- a host array carrying a
device ``xp``. ``wdmtransform`` then builds its default window as
``self.xp.ones(...)`` (CuPy) and ``fft`` evaluates ``self.arr * window``:
NumPy * CuPy, which CuPy's ufunc machinery refuses.

The host conversion is redundant: the only consumer that needs a host
array (``injections.py``'s synthetic stream builder) already wraps the
call in its own ``asnumpy(...)``. So ``raw_td`` returns the generator's
NATIVE array and each consumer converts if it needs to.

NB this is a BACKEND fix only -- it deliberately does not touch windowing.
``td_window`` stays ``None`` at every construction site (EMRI, SOBBH and
MBH alike): the template paths apply a flat window and the DATA taper is
excluded by the WDM edge crop, which is the global run's Tukey
convention. The fix changes which module the flat window is built on,
never its values.
"""

import unittest

import numpy as np

from lisatools.globalfit.stock.erebor.wrappers import (
    MBHTDIonFlyWaveWrap,
    SOBBHTDIonFlyWaveWrap,
)
from lisatools.utils.utility import asnumpy


class _FakeDeviceArray:
    """Minimal CuPy stand-in: ``asnumpy`` converts it via ``.get()``.

    ``lisatools.utils.utility.asnumpy`` pulls anything exposing ``.get()``
    to the host, which is exactly how a real ``cupy.ndarray`` is detected,
    so this reproduces the conversion on a CPU-only box.
    """

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def get(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    def __getitem__(self, item):
        return _FakeDeviceArray(self._arr[item])


# 11 SOBBH waveform-basis params (m1, m2, s1, s2, dist, inc, f_low, ra,
# dec, psi, phi0) -- the wrapper reorders them for the generator.
_PARAMS = (30.0, 25.0, 0.1, 0.2, 1.0, 0.5, 0.015, 1.0, 0.3, 0.4, 0.6)


class _RecordingGen:
    def __init__(self, out):
        self.out = out
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return self.out


class RawTDBackendPassthroughTest(unittest.TestCase):
    """``raw_td`` returns what the generator returned, on ITS backend."""

    def _wrap(self, cls, gen):
        return cls(
            gen,
            t_arr=np.linspace(0.0, 10.0, 8),
            td_settings=None,      # unused by raw_td
            target_domain=None,    # unused by raw_td
        )

    def test_sobbh_device_array_is_not_pulled_to_host(self):
        """THE regression: a device-backed generator must stay on device.

        Returning a host array here is what produced the NumPy * CuPy
        multiply inside TDSignal.fft on job 353.
        """
        dev = _FakeDeviceArray(np.ones((3, 8)))
        wrap = self._wrap(SOBBHTDIonFlyWaveWrap, _RecordingGen(dev))
        out = wrap.raw_td(*_PARAMS)
        self.assertIsInstance(
            out, _FakeDeviceArray,
            msg="raw_td pulled the waveform to host; TDSignal would then "
                "carry a host array with device settings.",
        )

    def test_mbh_tdionfly_device_array_is_not_pulled_to_host(self):
        """Same defect in the MBH TDI-on-the-fly wrapper (latent: only
        reachable with USE_TDIONFLY=1, since MBH defaults to phentax)."""
        dev = _FakeDeviceArray(np.ones((3, 8)))
        gen = _RecordingGen(dev)
        wrap = MBHTDIonFlyWaveWrap(
            gen,
            t_arr=np.linspace(0.0, 10.0, 8),
            td_settings=None,
            target_domain=None,
        )
        out = wrap.raw_td(1e6, 0.8, 0.1, 0.2, 1.0, 0.5, 1.0, 0.3, 0.4, 0.6, 0.7)
        self.assertIsInstance(out, _FakeDeviceArray)

    def test_host_array_passes_through_unchanged(self):
        """CPU runs are untouched: a NumPy generator still yields NumPy."""
        host = np.ones((3, 8))
        wrap = self._wrap(SOBBHTDIonFlyWaveWrap, _RecordingGen(host))
        out = wrap.raw_td(*_PARAMS)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(out, host)

    def test_nchannels_slice_still_applies(self):
        """The channel trim must survive on either backend."""
        wrap = SOBBHTDIonFlyWaveWrap(
            _RecordingGen(_FakeDeviceArray(np.ones((3, 8)))),
            t_arr=np.linspace(0.0, 10.0, 8),
            td_settings=None,
            target_domain=None,
            nchannels=2,
        )
        out = wrap.raw_td(*_PARAMS)
        self.assertIsInstance(out, _FakeDeviceArray)
        self.assertEqual(out.shape, (2, 8))

    def test_injection_consumer_still_gets_host(self):
        """The synthetic-injection builder wraps raw_td in its own
        asnumpy (injections.py), so it is unaffected by the change --
        that redundancy is WHY the inner conversion can go."""
        wrap = self._wrap(
            SOBBHTDIonFlyWaveWrap, _RecordingGen(_FakeDeviceArray(np.ones((3, 8))))
        )
        self.assertIsInstance(asnumpy(wrap.raw_td(*_PARAMS)), np.ndarray)


if __name__ == "__main__":
    unittest.main()
