"""JAX port of ``LISATDIonTheFly::get_tdi_Xf_single`` and friends.

Goal: byte-for-byte parity (modulo last-bit FMA reorderings) with the
C++ projection, the ``LISATDIonTheFly`` methods ``get_sky_vectors``,
``xi_projections``, ``get_tdi_Xf_single``, ``get_hp_hc`` and
``get_phase_ref`` in ``lisatools/cutils/lat_tdi_on_the_fly.cu``. The
structure here mirrors those methods 1:1. (The ``TDIonTheFly.cu:NNNN`` line
numbers in the section comments refer to the pre-absorption lisa-on-gpu
file and no longer match.)

The functions take a :class:`~lisatools.jax.orbits.OrbitsWrapJAX`, a
:class:`~lisatools.jax.response.tdi_config.TDIConfigWrapJAX`, and a source
object inheriting from :class:`~lisatools.jax.response.base.JaxAmpPhaseSource`.
The TDI configuration is consumed as *Python* lists/arrays (not jnp) so the
per-unit / per-sub-delay loops unroll at trace time -- ``num_units`` is small
(24 for 1st-gen, 48 for 2nd-gen) and unrolling is the simplest way to keep
``jax.jit`` and ``jax.grad`` happy without resorting to ``jax.lax.fori_loop``
over heterogeneous link indices. ``C_INV`` is ``1 / c`` [s/m] from
:mod:`lisatools.utils.constants`.

Consumers: the GBGPU JAX path (``gbgpu.jax.tdi_on_the_fly``,
``gbgpu.jax.wdm``) with its :class:`JaxAmpPhaseSource` plugins
(``gbgpu.jax.sources.ucb.JaxUCBSource``, ``bbhx.jax.sources.sobbh.JaxSOBBHSource``).
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .base import JaxAmpPhaseSource
from lisatools.jax.orbits import OrbitsWrapJAX
from .tdi_config import TDIConfigWrapJAX

from lisatools.utils.constants import C_SI as _C_SI

# Mirrors ``C_inv`` in LISAResponse.hh (the reciprocal of the speed of light
# in s/m, lisaconstants' SPEED_OF_LIGHT).
C_INV = 1.0 / _C_SI


# --------------------------------------------------------------------
# Geometry helpers (TDIonTheFly.cu:3605-3635)
# --------------------------------------------------------------------
def get_sky_vectors(params: jnp.ndarray, source: JaxAmpPhaseSource
                    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the (k, u, v) sky basis from (lam, beta) in ``params``.

    Identical to ``LISATDIonTheFly::get_sky_vectors``. ``k`` is the
    propagation direction (minus the unit vector to the source); ``u`` and
    ``v`` span the transverse plane. ``lam`` / ``beta`` are the sky longitude
    / latitude in the orbits' frame.

    Args:
        params: 1-D source parameters; the sky entries are read at
            ``source.get_sky_indices()``.
        source: The source (only its index layout is used).

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: ``(k, u, v)``, each ``(3,)``.
    """
    inc_i, psi_i, lam_i, beta_i = source.get_sky_indices()
    beta = params[beta_i]
    lam = params[lam_i]
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    cl, sl = jnp.cos(lam), jnp.sin(lam)
    v = jnp.array([-sb * cl, -sb * sl, cb])
    u = jnp.array([sl, -cl, 0.0])
    k = jnp.array([-cb * cl, -cb * sl, -sb])
    return k, u, v


def xi_projections(u: jnp.ndarray, v: jnp.ndarray, n: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the antenna projections ``(xi_p, xi_c)`` for arm unit vector ``n``.

    ``xi_p = ((u.n)^2 - (v.n)^2) / 2`` and ``xi_c = (u.n)(v.n)``; mirrors
    ``LISATDIonTheFly::xi_projections``.

    Args:
        u: Sky basis vector ``(3,)``.
        v: Sky basis vector ``(3,)``.
        n: Arm unit vector(s), ``(..., 3)``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(xi_p, xi_c)``, shape ``n.shape[:-1]``.
    """
    u_dot_n = jnp.sum(u * n, axis=-1)
    v_dot_n = jnp.sum(v * n, axis=-1)
    xi_p = 0.5 * (u_dot_n * u_dot_n - v_dot_n * v_dot_n)
    xi_c = u_dot_n * v_dot_n
    return xi_p, xi_c


# --------------------------------------------------------------------
# get_hp_hc  (TDIonTheFly.cu:3886-3910)
# --------------------------------------------------------------------
def get_hp_hc(t: jnp.ndarray, params: jnp.ndarray,
              source: JaxAmpPhaseSource, phase_change: float
              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the source-frame polarizations rotated by ``psi`` at time(s) ``t``.

    Matches the C++ ``LISATDIonTheFly::get_hp_hc``: the un-projected
    source polarizations are::

        hSp = -cos(phase + phase_change) * amp * (1 + cos^2(iota))
        hSc = -sin(phase + phase_change) * 2 * amp * cos(iota)

    and the rotation by ``psi`` gives::

        hp = hSp * cos(2psi) - hSc * sin(2psi)
        hc = hSp * sin(2psi) + hSc * cos(2psi)

    Args:
        t: Time(s) [s] at which the source amplitude and phase are evaluated.
        params: 1-D source parameters.
        source: The amplitude / phase source.
        phase_change: Phase offset [rad]; ``0`` gives the real part of the
            projection, ``pi / 2`` the imaginary part.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(hp, hc)``, broadcasting with ``t``.
    """
    inc_i, psi_i, _, _ = source.get_sky_indices()
    amp = source.amplitude(t, params)
    phase = source.phase(t, params)
    psi = params[psi_i]
    inc = params[inc_i]

    cos2psi = jnp.cos(2.0 * psi)
    sin2psi = jnp.sin(2.0 * psi)
    cosi = jnp.cos(inc)

    hSp = -jnp.cos(phase + phase_change) * amp * (1.0 + cosi * cosi)
    hSc = -jnp.sin(phase + phase_change) * 2.0 * amp * cosi

    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi
    return hp, hc


# --------------------------------------------------------------------
# Single-time-sample TDI projection (TDIonTheFly.cu:3699-3834)
# --------------------------------------------------------------------
def get_tdi_Xf_single(t: jnp.ndarray, params: jnp.ndarray,
                      source: JaxAmpPhaseSource,
                      orbits: OrbitsWrapJAX,
                      tdi_config: TDIConfigWrapJAX,
                      k: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray
                      ) -> jnp.ndarray:
    """Project the source onto LISA arms and form the TDI variables at ``t``.

    For each TDI unit: the delayed reception time is ``t`` minus the sum of
    the unit's delay-link light travel times (each evaluated at ``t``, not
    chained); the base link's emission time follows from its LTT; the arm
    vector comes from the interpolated spacecraft positions; and the unit
    adds ``sign / (1 - k.n)`` times the emitter-minus-receiver difference of
    :func:`get_hp_hc` projected with :func:`xi_projections`, as a complex
    number (real part at phase offset 0, imaginary at ``pi / 2``).

    The implementation unrolls the ``num_units`` outer loop at Python
    trace time (the TDIConfig arrays are concrete numpy ints). The
    inner per-time work is a single ``jnp`` expression so JAX can fuse
    everything via ``jit`` / ``vmap``.

    Args:
        t: Time(s) [s].
        params: 1-D source parameters.
        source: The amplitude / phase source.
        orbits: The JAX orbits.
        tdi_config: The JAX TDI configuration.
        k: Sky vector from :func:`get_sky_vectors`.
        u: Sky vector from :func:`get_sky_vectors`.
        v: Sky vector from :func:`get_sky_vectors`.

    Returns:
        jnp.ndarray: complex128, shape ``t.shape + (num_channels,)``
        (typically ``(N, 3)`` for one binary).
    """
    num_units = tdi_config.num_units
    num_channels = tdi_config.num_channels

    # Force ``t`` to at least 1D so the reductions broadcast cleanly.
    t = jnp.asarray(t)

    # Per-channel accumulators (complex). Build as a list and stack
    # so we can selectively add into specific channels per unit.
    accum_per_channel = [jnp.zeros_like(t, dtype=jnp.complex128)
                         for _ in range(num_channels)]

    for unit_i in range(num_units):
        unit_start = int(tdi_config.unit_starts[unit_i])
        unit_length = int(tdi_config.unit_lengths[unit_i])
        base_link = int(tdi_config.tdi_base_links[unit_i])
        channel = int(tdi_config.channels[unit_i])
        sign = float(tdi_config.tdi_signs[unit_i])

        # Sum of LTTs of all combination links applied as delays before
        # the base-link projection. Matches TDIonTheFly.cu:3737-3756 --
        # each LTT is evaluated at the original ``t``, not chained.
        total_delay = jnp.zeros_like(t)
        for sub_i in range(unit_length):
            ci = unit_start + sub_i
            comb_link = int(tdi_config.tdi_link_combinations[ci])
            if comb_link == -11:
                # ``-11`` is the C++ sentinel for "no delay in this slot".
                continue
            total_delay = total_delay + orbits.get_light_travel_time(
                t, comb_link
            )

        time_eval = t - total_delay
        time_rec = time_eval

        # Base-link LTT and the receiver/emitter spacecraft for ``base_link``.
        L = orbits.get_light_travel_time(time_rec, base_link)
        time_em = time_rec - L

        bli = orbits.get_link_ind(base_link)
        sc_r = int(orbits.sc_r[bli])
        sc_e = int(orbits.sc_e[bli])

        x_rec = orbits.get_pos(time_rec, sc_r)
        x_em = orbits.get_pos(time_em, sc_e)
        n = x_rec - x_em
        n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)

        k_dot_n = jnp.sum(k * n, axis=-1)
        k_dot_x_rec = jnp.sum(k * x_rec, axis=-1)
        k_dot_x_em = jnp.sum(k * x_em, axis=-1)

        pre_factor = 1.0 / (1.0 - k_dot_n)

        delay_rec = time_rec - k_dot_x_rec * C_INV
        delay_em = time_em - k_dot_x_em * C_INV

        xi_p, xi_c = xi_projections(u, v, n)

        # Real part of the projection: phase_change = 0.
        hp_rec_r, hc_rec_r = get_hp_hc(delay_rec, params, source, 0.0)
        hp_em_r, hc_em_r = get_hp_hc(delay_em, params, source, 0.0)
        large_real = (hp_em_r - hp_rec_r) * xi_p + (hc_em_r - hc_rec_r) * xi_c

        # Imag part: phase_change = pi/2.
        hp_rec_i, hc_rec_i = get_hp_hc(delay_rec, params, source, jnp.pi / 2.0)
        hp_em_i, hc_em_i = get_hp_hc(delay_em, params, source, jnp.pi / 2.0)
        large_imag = (hp_em_i - hp_rec_i) * xi_p + (hc_em_i - hc_rec_i) * xi_c

        contrib = sign * pre_factor * (large_real + 1j * large_imag)
        accum_per_channel[channel] = accum_per_channel[channel] + contrib

    return jnp.stack(accum_per_channel, axis=-1)


# --------------------------------------------------------------------
# get_phase_ref (TDIonTheFly.cu:4406-4420)
# --------------------------------------------------------------------
def get_phase_ref(t: jnp.ndarray, params: jnp.ndarray,
                  source: JaxAmpPhaseSource,
                  orbits: OrbitsWrapJAX) -> jnp.ndarray:
    """Return the reference phase at spacecraft 1.

    Mirrors ``LISATDIonTheFly::get_phase_ref`` exactly: evaluate the
    source phase at the SSB-shifted time
    ``t - k . x_rec(t, sc=1) * C_inv``.

    Args:
        t: Time(s) [s].
        params: 1-D source parameters.
        source: The amplitude / phase source.
        orbits: The JAX orbits.

    Returns:
        jnp.ndarray: Phase [rad], broadcasting with ``t``.
    """
    k, _, _ = get_sky_vectors(params, source)
    x_rec = orbits.get_pos(t, 1)  # spacecraft 1
    k_dot_x_rec = jnp.sum(k * x_rec, axis=-1)
    t_sc = t - k_dot_x_rec * C_INV
    return source.phase(t_sc, params)
