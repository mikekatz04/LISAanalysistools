"""JAX-native mirror of the C++ ``OrbitsWrap`` interpolation helpers.

Consumes the same flat tuple of arrays that the C++ wrapper takes
(``Orbits.pycppdetector_args``). On construction we convert all the
data buffers to ``jnp.ndarray``; the two methods downstream consumers
need are:

* ``get_pos(t, sc) -> Vec3`` -- spacecraft position (seconds) at time
  ``t``.
* ``get_light_travel_time(t, link) -> double`` -- LTT for ``link`` at
  ``t``.

Both use linear interpolation on a uniform grid -- identical to the
C++ ``linear_interp_setup=True`` configuration. The methods are pure
functions of ``t`` so they JIT/grad cleanly.

Convention reminders (from ``Detector.cu`` and ``configure()``):

* Spacecraft are 1-indexed (1, 2, 3); positions array ``x`` is shape
  ``(N, NSC, 3)``, with ``NSC = 3``. We store internally as
  ``(N, 3, 3)`` and translate ``sc - 1`` at lookup time.
* Links are 6-arm objects with codes ``12, 23, 31, 13, 32, 21`` (the
  ``links`` array carries the actual order); LTTs and arm normals are
  shape ``(N, NLINKS)``, ``(N, NLINKS, 3)``. ``get_link_ind`` resolves
  a link-code to an index via the ``links`` array.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

NLINKS = 6
NSC = 3


def _to_jnp(arr) -> jnp.ndarray:
    """Convert numpy/cupy arrays to jnp.

    ``cupy`` arrays go through ``cp.asnumpy`` first so we don't
    force a JAX dependency on cupy.
    """
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
    except (ImportError, ModuleNotFoundError):
        pass
    return jnp.asarray(np.asarray(arr))


class OrbitsWrapJAX:
    """Pure-JAX analog of the C++ ``OrbitsWrap`` class.

    Accepts the same 13-element tuple as ``OrbitsWrap(*pycppdetector_args)``:
        (ltt_t0, ltt_dt, ltt_N, sc_t0, sc_dt, sc_N,
         n_flat, ltt_flat, x_flat, links, sc_r, sc_e, armlength)
    """

    def __init__(
        self,
        ltt_t0: float,
        ltt_dt: float,
        ltt_N: int,
        sc_t0: float,
        sc_dt: float,
        sc_N: int,
        n_flat,
        ltt_flat,
        x_flat,
        links,
        sc_r,
        sc_e,
        armlength: float,
    ):
        self.ltt_t0 = float(ltt_t0)
        self.ltt_dt = float(ltt_dt)
        self.ltt_N = int(ltt_N)
        self.sc_t0 = float(sc_t0)
        self.sc_dt = float(sc_dt)
        self.sc_N = int(sc_N)
        self.armlength = float(armlength)

        # Layouts (matches Detector.cu):
        #   n_flat   : (N, NLINKS, 3)
        #   ltt_flat : (N, NLINKS)
        #   x_flat   : (N, NSC, 3)
        n_arr = np.asarray(n_flat).reshape(self.sc_N, NLINKS, 3)
        ltt_arr = np.asarray(ltt_flat).reshape(self.ltt_N, NLINKS)
        x_arr = np.asarray(x_flat).reshape(self.sc_N, NSC, 3)

        self.n = _to_jnp(n_arr)
        self.ltt = _to_jnp(ltt_arr)
        self.x = _to_jnp(x_arr)

        # Ints stay as numpy (small fixed-size, used at trace time).
        self.links = np.asarray(links, dtype=np.int32).copy()
        self.sc_r = np.asarray(sc_r, dtype=np.int32).copy()
        self.sc_e = np.asarray(sc_e, dtype=np.int32).copy()

        # Reverse map: link-code -> index in ``links``. Same as
        # ``Orbits::get_link_ind`` in C++.
        self._link_to_ind = {int(c): i for i, c in enumerate(self.links)}

    # ---- index helpers ----
    def get_link_ind(self, link_code: int) -> int:
        return self._link_to_ind[int(link_code)]

    # ---- interpolation primitives ----
    @staticmethod
    def _linear_interp_scalar(t: jnp.ndarray, t0: float, dt: float, N: int,
                              values: jnp.ndarray) -> jnp.ndarray:
        """Linear interpolation along axis 0 of ``values``.

        Matches the C++ ``Orbits::interpolate`` (linear in a uniform grid)
        rather than the cubic-spline base path. The C++ wrapper uses the
        same linear path when ``linear_interp_setup=True`` was passed to
        ``configure()`` -- which is the only mode we accept.

        ``t`` is any shape ``S``; ``values`` is shape ``(N, ...)``. Returns
        shape ``S + values.shape[1:]``.
        """
        x = (t - t0) / dt
        # Clip indices to valid range; out-of-bounds is the caller's
        # problem (the C++ kernels return zero via the ``is_okay``
        # flag in ``get_tdi_Xf_single``; we just clip so the math
        # doesn't NaN).
        i0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, N - 2)
        i1 = i0 + 1
        frac = x - i0.astype(x.dtype)
        # Broadcast frac to value shape.
        while frac.ndim < values[i0].ndim:
            frac = frac[..., None]
        return (1.0 - frac) * values[i0] + frac * values[i1]

    def get_pos(self, t: jnp.ndarray, sc: int) -> jnp.ndarray:
        """Spacecraft position at time(s) ``t``. ``sc`` is 1-indexed.

        Returns array of shape ``t.shape + (3,)``.
        """
        sc_idx = int(sc) - 1
        return self._linear_interp_scalar(
            t, self.sc_t0, self.sc_dt, self.sc_N, self.x[:, sc_idx, :]
        )

    def get_light_travel_time(self, t: jnp.ndarray, link_code: int) -> jnp.ndarray:
        """Light travel time along ``link_code`` at time(s) ``t``."""
        li = self.get_link_ind(link_code)
        return self._linear_interp_scalar(
            t, self.ltt_t0, self.ltt_dt, self.ltt_N, self.ltt[:, li]
        )

    def get_n(self, t: jnp.ndarray, link_code: int) -> jnp.ndarray:
        """Arm normal vector (3,) at time(s) ``t`` for ``link_code``."""
        li = self.get_link_ind(link_code)
        return self._linear_interp_scalar(
            t, self.ltt_t0, self.ltt_dt, self.ltt_N, self.n[:, li, :]
        )

    # ---- bulk variants for jit/vmap convenience ----
    def get_pos_all(self, t: jnp.ndarray) -> jnp.ndarray:
        """Positions for all 3 spacecraft at ``t``. Shape ``t.shape + (3,3)``."""
        return self._linear_interp_scalar(
            t, self.sc_t0, self.sc_dt, self.sc_N, self.x
        )

    def get_ltt_all(self, t: jnp.ndarray) -> jnp.ndarray:
        """LTTs for all NLINKS arms at ``t``. Shape ``t.shape + (NLINKS,)``."""
        return self._linear_interp_scalar(
            t, self.ltt_t0, self.ltt_dt, self.ltt_N, self.ltt
        )
