"""Validate the batched pyResponseTDI path (feat-batching port, 2026-06).

The batched response runs all sources through a single pair of CUDA kernel
launches (parallel along the grid z dimension). This test asserts the batched
output is bit-for-bit identical to looping the single-source path per source
(the batched kernel is the same device code with a per-source ``batch_ind``
offset, so agreement must be to machine precision), and that the batched output
shapes are ``(batch_size, num_pts)`` per channel.

Ported/adapted from asantini29/lisa-on-gpu @ feat-batching
(tests/test_batched_response.py).
"""

import unittest

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.response.directresponse import pyResponseTDI
from lisatools.response.parallelbase import FastLISAResponseParallelModule

YRSID_SI = 31558149.763545603

# Fixed GB source parameters (sky-independent).
_GB_ARGS = dict(A=1.084702251e-22, f=2.35962078e-3, fdot=1.47197271e-17,
                iota=1.11820901, phi0=4.91128699, psi=2.3290324)

# Distinct sky positions (ecliptic longitude, latitude). Kept small (2 sources)
# so the batched y_gw alloc (batch * 6 * num_pts doubles) stays tiny.
_SKY = [
    (5.22979888, 0.98057429),
    (1.5, -0.3),
]
_LAMBDAS = np.array([p[0] for p in _SKY])
_BETAS = np.array([p[1] for p in _SKY])

# Short observation: ~0.03 yr -> ~95k samples at dt=10, well above the TDI
# buffer (2 * t_buffer/dt = 2000 pts) but cheap on memory for the batch.
_T, _DT = 0.03, 10.0
_ORDER = 25
_T_BUFFER = 10000.0


class _GBWave(FastLISAResponseParallelModule):
    """Minimal analytic GB waveform generator (hp + i hc) on the chosen backend."""

    @property
    def xp(self):
        return self.backend.xp

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)
        fddot = 11.0 / 3.0 * fdot ** 2 / f
        phase = (
            2 * np.pi * (f * t + 0.5 * fdot * t ** 2 + (1.0 / 6.0) * fddot * t ** 3)
            - phi0
        )
        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota ** 2)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi
        return hp + 1j * hc


class TestBatchedResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wave_gen = _GBWave(force_backend="cpu")
        cls.t = np.arange(0.0, _T * YRSID_SI, _DT)

        # one waveform per sky position; sky-independent params -> identical
        # polarizations, so any TDI difference is purely the sky projection.
        pols = [
            cls.wave_gen(
                _GB_ARGS["A"], _GB_ARGS["f"], _GB_ARGS["fdot"],
                _GB_ARGS["iota"], _GB_ARGS["phi0"], _GB_ARGS["psi"],
                T=_T, dt=_DT,
            )
            for _ in _SKY
        ]
        cls.all_pols = np.asarray(pols)  # (nsky, num_time_samples)

        # Force orbits onto the same (CPU) backend as the response. Without this,
        # ``EqualArmlengthOrbits()`` defaults to the first available backend (the
        # GPU on a CUDA box), tripping the ``tdi_orbits`` backend-match assertion.
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        orbits.configure(linear_interp_setup=True)
        cls.orbits = orbits

    def _make_response(self):
        return pyResponseTDI(
            sampling_frequency=1.0 / _DT,
            num_pts=len(self.t),
            orbits=self.orbits,
            order=_ORDER,
            tdi="1st generation",
            tdi_chan="AET",
            force_backend="cpu",
        )

    def test_batched_shapes(self):
        resp = self._make_response()
        resp.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        chans = resp.get_tdi_delays()
        self.assertEqual(len(chans), 3)  # AET
        for ch in chans:
            self.assertEqual(ch.shape, (len(_SKY), len(self.t)))

    def test_single_source_unchanged(self):
        """A scalar-sky call still returns 1D channels (batch_size == 1)."""
        resp = self._make_response()
        resp.get_projections(self.all_pols[0], _LAMBDAS[0], _BETAS[0], t_buffer=_T_BUFFER)
        chans = resp.get_tdi_delays()
        for ch in chans:
            self.assertEqual(ch.shape, (len(self.t),))

    def test_batched_equals_per_source(self):
        # Batched: one kernel launch for all sources.
        resp_b = self._make_response()
        resp_b.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        A_b, E_b, T_b = resp_b.get_tdi_delays()

        # Per-source: loop the single-source path.
        for b in range(len(_SKY)):
            resp_s = self._make_response()
            resp_s.get_projections(
                self.all_pols[b], _LAMBDAS[b], _BETAS[b], t_buffer=_T_BUFFER
            )
            A_s, E_s, T_s = resp_s.get_tdi_delays()
            np.testing.assert_array_equal(np.asarray(A_b[b]), np.asarray(A_s))
            np.testing.assert_array_equal(np.asarray(E_b[b]), np.asarray(E_s))
            np.testing.assert_array_equal(np.asarray(T_b[b]), np.asarray(T_s))

    def test_run_async_matches_sync(self):
        """run_async must not change CPU results (it only gates device streams)."""
        resp_sync = self._make_response()
        resp_sync.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        sync = resp_sync.get_tdi_delays(run_async=False)

        resp_async = self._make_response()
        resp_async.get_projections(
            self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER, run_async=True
        )
        asyncd = resp_async.get_tdi_delays(run_async=True)

        for ch_sync, ch_async in zip(sync, asyncd):
            np.testing.assert_array_equal(np.asarray(ch_sync), np.asarray(ch_async))


if __name__ == "__main__":
    unittest.main()
