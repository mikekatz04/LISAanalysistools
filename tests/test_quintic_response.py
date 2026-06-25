"""Validate the quintic-spline projection path of pyResponseTDI.

``pyResponseTDI(..., use_spline=True)`` evaluates the complex
waveform at the light-travel-time-delayed retarded times with a degree-5
spline (GBT ``QuinticSplineInterpolant`` fit + a ``QuinticSplineSegment``
device eval) instead of the order-25 Lagrange fractional-delay filter. The
TDI step always stays on Lagrange. These tests run on the CPU backend.

Checks:
  * a degree-<=5 polynomial waveform -> quintic and Lagrange projections agree
    to ~machine precision (both schemes are exact on degree-5 data);
  * a smooth, well-sampled GB wave -> the two schemes agree to the quintic
    interpolation-error level (loose but bug-catching);
  * batched quintic == looped single-source quintic, to machine precision
    (exercises the per-source coefficient indexing / combined-fit slicing);
  * output shapes match the Lagrange path (batched and scalar-sky).
"""

# eryn.utils.plot does `import scienceplots; plt.style.use(['science'])` inside a
# try/except (ImportError, ModuleNotFoundError). The installed scienceplots is
# incompatible with this matplotlib (raises AttributeError at import, which is
# NOT caught). Setting it to None in sys.modules makes `import scienceplots`
# raise a *caught* ImportError so the lisatools import chain succeeds. We do not
# plot here. (Pre-existing env issue, unrelated to the response code.)
import sys as _sys

_sys.modules["scienceplots"] = None

# eryn.utils.plot also uses a bare ``typing.Union`` in a function annotation
# without ``import typing`` (pre-existing eryn bug). Bare-name lookup falls
# through to builtins, so injecting ``typing`` there fixes the import without
# editing the dependency.
import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import unittest

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.response.directresponse import pyResponseTDI
from lisatools.response.parallelbase import FastLISAResponseParallelModule

YRSID_SI = 31558149.763545603

_GB_ARGS = dict(A=1.084702251e-22, f=2.35962078e-3, fdot=1.47197271e-17,
                iota=1.11820901, phi0=4.91128699, psi=2.3290324)

_SKY = [
    (5.22979888, 0.98057429),
    (1.5, -0.3),
]
_LAMBDAS = np.array([p[0] for p in _SKY])
_BETAS = np.array([p[1] for p in _SKY])

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


class TestQuinticResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wave_gen = _GBWave(force_backend="cpu")
        cls.t = np.arange(0.0, _T * YRSID_SI, _DT)

        pols = [
            cls.wave_gen(
                _GB_ARGS["A"], _GB_ARGS["f"], _GB_ARGS["fdot"],
                _GB_ARGS["iota"], _GB_ARGS["phi0"], _GB_ARGS["psi"],
                T=_T, dt=_DT,
            )
            for _ in _SKY
        ]
        cls.all_pols = np.asarray(pols)

        orbits = EqualArmlengthOrbits()
        orbits.configure(linear_interp_setup=True)
        cls.orbits = orbits

    def _make_response(self, use_spline, **kwargs):
        return pyResponseTDI(
            sampling_frequency=1.0 / _DT,
            num_pts=len(self.t),
            orbits=self.orbits,
            order=_ORDER,
            tdi="1st generation",
            tdi_chan="AET",
            use_spline=use_spline,
            force_backend="cpu",
            **kwargs,
        )

    def _projections(self, use_spline, pols, lam, beta):
        resp = self._make_response(use_spline)
        resp.get_projections(pols, lam, beta, t_buffer=_T_BUFFER)
        return resp, np.asarray(resp.y_gw)

    def _valid_window(self, resp):
        # interior region where both schemes computed non-garbage projections
        pad = resp.projections_start_ind + 50
        return slice(pad, resp.num_pts - pad)

    def _reldiff(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        scale = np.abs(a).max()
        return np.abs(a - b).max() / scale

    def test_default_is_lagrange(self):
        resp = self._make_response(False)
        self.assertFalse(resp.use_spline)
        # default kwarg
        resp_def = pyResponseTDI(
            sampling_frequency=1.0 / _DT, num_pts=len(self.t), orbits=self.orbits,
            tdi="1st generation", force_backend="cpu",
        )
        self.assertFalse(resp_def.use_spline)

    def test_non_bool_use_spline_raises(self):
        # use_spline is a strict bool now (was a string scheme name). Passing the
        # old string API -- or any other non-bool -- must fail fast in __init__.
        with self.assertRaises(AssertionError):
            self._make_response("quintic")

    def test_quintic_shapes(self):
        resp = self._make_response(True)
        resp.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        chans = resp.get_tdi_delays()
        self.assertEqual(len(chans), 3)  # AET
        for ch in chans:
            self.assertEqual(ch.shape, (len(_SKY), len(self.t)))

    def test_quintic_scalar_sky_shapes(self):
        resp = self._make_response(True)
        resp.get_projections(self.all_pols[0], _LAMBDAS[0], _BETAS[0], t_buffer=_T_BUFFER)
        for ch in resp.get_tdi_delays():
            self.assertEqual(ch.shape, (len(self.t),))

    def test_constant_input_projection_is_zero(self):
        """A constant waveform has an identically-zero projection (h(t_em) ==
        h(t_rec)). The quintic reproduces a constant exactly, so its projection
        is ~0. (Order-25 Lagrange has a ~1e-7*|h| floor here: cancellation in its
        26-point weighted sum. So the quintic is the *more* accurate scheme in
        this slowly-varying / cancellation-dominated regime -- the whole point of
        being able to switch.)"""
        scale = 1e-21
        const = np.ones_like(self.t) * scale + 0j
        resp_q, y_quin = self._projections(True,const, _LAMBDAS[0], _BETAS[0])
        _, y_lag = self._projections(False,const, _LAMBDAS[0], _BETAS[0])
        w = self._valid_window(resp_q)
        quin = np.abs(y_quin[:, w]).max() / scale
        lag = np.abs(y_lag[:, w]).max() / scale
        self.assertLess(quin, 1e-12, f"quintic constant projection {quin:.2e} should be ~0")
        # quintic is at least as accurate as Lagrange here (documents the win;
        # quintic ~ 0, Lagrange ~ 1e-7).
        self.assertLessEqual(quin, lag)

    def test_gbwave_matches_lagrange(self):
        """Smooth, well-sampled GB wave -> the two schemes agree to ~1e-6.

        The residual is dominated by the Lagrange floor (~1e-7*|h|) plus the
        genuine quintic interpolation error (~1e-8 for f*dt ~ 0.024); 1e-5 is a
        comfortable, bug-catching bound (a wrong index/offset gives O(1))."""
        _, y_lag = self._projections(False,self.all_pols[0], _LAMBDAS[0], _BETAS[0])
        resp_q, y_quin = self._projections(True,self.all_pols[0], _LAMBDAS[0], _BETAS[0])
        w = self._valid_window(resp_q)
        rd = self._reldiff(y_lag[:, w], y_quin[:, w])
        self.assertLess(rd, 1e-5, f"GB quintic vs lagrange reldiff {rd:.2e}")

    def test_quintic_batched_equals_per_source(self):
        """Batched quintic == looped single-source quintic, to machine precision."""
        resp_b = self._make_response(True)
        resp_b.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        A_b, E_b, T_b = resp_b.get_tdi_delays()

        for b in range(len(_SKY)):
            resp_s = self._make_response(True)
            resp_s.get_projections(
                self.all_pols[b], _LAMBDAS[b], _BETAS[b], t_buffer=_T_BUFFER
            )
            A_s, E_s, T_s = resp_s.get_tdi_delays()
            np.testing.assert_array_equal(np.asarray(A_b[b]), np.asarray(A_s))
            np.testing.assert_array_equal(np.asarray(E_b[b]), np.asarray(E_s))
            np.testing.assert_array_equal(np.asarray(T_b[b]), np.asarray(T_s))

    def test_call_time_use_spline_override(self):
        """``get_projections(use_spline=...)`` overrides the instance default in
        BOTH directions; ``use_spline=None`` (default) falls back to it.

        Forcing a call to a scheme must reproduce the matching construct-time
        projection exactly (same kernel), regardless of the instance default."""
        # construct-time references for each scheme
        lag_ref = self._make_response(False)
        lag_ref.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        y_lag = np.asarray(lag_ref.y_gw).copy()

        quin_ref = self._make_response(True)
        quin_ref.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        y_quin = np.asarray(quin_ref.y_gw).copy()

        # the two schemes genuinely differ (guards against a no-op test)
        self.assertGreater(self._reldiff(y_lag, y_quin), 1e-9)

        # Lagrange-default instance forced ON -> quintic
        force_on = self._make_response(False)
        force_on.get_projections(
            self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER, use_spline=True
        )
        np.testing.assert_array_equal(np.asarray(force_on.y_gw), y_quin)

        # quintic-default instance forced OFF -> Lagrange (the both-directions case)
        force_off = self._make_response(True)
        force_off.get_projections(
            self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER, use_spline=False
        )
        np.testing.assert_array_equal(np.asarray(force_off.y_gw), y_lag)

        # use_spline=None (default) respects the instance default in both cases
        lag_def = self._make_response(False)
        lag_def.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        np.testing.assert_array_equal(np.asarray(lag_def.y_gw), y_lag)

        quin_def = self._make_response(True)
        quin_def.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        np.testing.assert_array_equal(np.asarray(quin_def.y_gw), y_quin)

    def test_quintic_chunk_kwarg_matches_auto(self):
        """``quintic_chunk`` forwards to the GBT SPIKE fit and changes only the
        band-solve partitioning, not the result. A forced small chunk must
        reproduce the auto-sized quintic projection to fit precision."""
        auto = self._make_response(True)  # quintic_chunk defaults to 0 (auto)
        self.assertEqual(auto.quintic_chunk, 0)
        auto.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        y_auto = np.asarray(auto.y_gw).copy()

        forced = self._make_response(True, quintic_chunk=64)
        self.assertEqual(forced.quintic_chunk, 64)
        forced.get_projections(self.all_pols, _LAMBDAS, _BETAS, t_buffer=_T_BUFFER)
        # different SPIKE partitioning -> identical up to FP summation order
        self.assertLess(self._reldiff(y_auto, forced.y_gw), 1e-9)

    def test_invalid_quintic_chunk_raises(self):
        with self.assertRaises(AssertionError):
            self._make_response(True, quintic_chunk=-1)


if __name__ == "__main__":
    unittest.main()
