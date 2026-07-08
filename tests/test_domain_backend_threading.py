"""Backend threading through layered domain transforms.

Regression tests for the 2026-07-08 bug: intermediate DomainSettings built
under the hood during layered transforms (e.g. WDM -> STFT runs
``wdm_to_fd(settings=None).ifft(settings=None).stft(...)``) must inherit the
source signal's backend. ``FDSignal.ifft(settings=None)`` constructed a
default-backend ``TDSettings`` mid-chain, so on GPU runs the chain went
gpu -> cpu -> NumPy/CuPy mixing error.

Also covers the new fail-fast guard: ``transform`` raises immediately (with
an actionable message) when the TARGET settings live on a different backend
than the signal, instead of erroring deep inside the chain.
"""
import unittest

import numpy as np

try:
    import cupy as cp

    cp.abs(cp.ones(2))
    gpu_available = True
except (ImportError, ModuleNotFoundError, Exception):
    gpu_available = False

from lisatools.domains import (
    FDSettings,
    STFTSettings,
    TDSettings,
    TDSignal,
    WDMSettings,
    get_stft_settings,
)

FORCE = "gpu" if gpu_available else "cpu"


def _flavor(obj) -> str:
    return obj.backend_name.split("_")[-1]


class DomainBackendThreadingTest(unittest.TestCase):
    def _td(self, n=2048, dt=10.0):
        rng = np.random.default_rng(7)
        arr = rng.standard_normal((3, n))
        settings = TDSettings(n, dt, t0=0.0, force_backend=FORCE)
        return TDSignal(settings.xp.asarray(arr), settings)

    def test_ifft_default_settings_inherit_backend(self):
        """FDSignal.ifft(settings=None) intermediate TDSettings inherits the backend."""
        td = self._td()
        fd = td.fft()
        self.assertEqual(_flavor(fd), _flavor(td))
        back = fd.ifft()  # settings=None branch (was CPU-default)
        self.assertEqual(_flavor(back), _flavor(td))
        self.assertIs(back.xp, td.xp)

    def test_fft_default_settings_inherit_backend(self):
        td = self._td()
        fd = td.fft()  # settings=None branch
        self.assertEqual(_flavor(fd), _flavor(td))

    def test_wdm_to_stft_layered_chain(self):
        """WDM -> STFT threads the backend through BOTH hidden intermediates."""
        n, dt = 2048, 10.0
        td = self._td(n, dt)
        wdm_set = WDMSettings(Nf=32, Nt=64, dt=dt, force_backend=FORCE)
        wdm = td.transform(wdm_set)
        self.assertEqual(_flavor(wdm), FORCE if gpu_available else "cpu")

        times = np.arange(n) * dt
        stft_set = get_stft_settings(times, big_dt=640.0, force_backend=FORCE)
        out = wdm.transform(stft_set)  # wdm_to_fd(None).ifft(None).stft(...)
        self.assertEqual(_flavor(out), _flavor(wdm))
        self.assertIs(out.xp, wdm.xp)

    def test_wdm_to_td_and_wdm_layered_chains(self):
        n, dt = 2048, 10.0
        td = self._td(n, dt)
        wdm = td.transform(WDMSettings(Nf=32, Nt=64, dt=dt, force_backend=FORCE))
        back_td = wdm.transform(TDSettings(n, dt, t0=0.0, force_backend=FORCE))
        self.assertIs(back_td.xp, wdm.xp)
        wdm2 = wdm.transform(WDMSettings(Nf=64, Nt=32, dt=dt, force_backend=FORCE))
        self.assertIs(wdm2.xp, wdm.xp)

    @unittest.skipUnless(gpu_available, "needs a GPU backend to build a true mismatch")
    def test_transform_target_backend_mismatch_raises(self):
        """GPU signal + CPU target settings -> immediate, actionable error."""
        td = self._td()
        cpu_target = FDSettings(td.N // 2 + 1, 1.0 / (td.N * td.dt), force_backend="cpu")
        with self.assertRaises(ValueError) as ctx:
            td.transform(cpu_target)
        self.assertIn("force_backend", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
