"""SOBBH PN waveform (3.5PN, aligned spins).

Ported from ``sobbhtaylert3.py`` (originally from Diganta Bandopadhyay).
The math is unchanged — this module just packages it as an importable
piece of ``lisatools`` and exposes a small :class:`SOBBHWaveform` class
that mirrors the call signature ``ResponseWrapper`` expects from EMRI
waveform generators.
"""

from __future__ import annotations

from typing import Optional, Tuple

import lisaconstants as lc
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import config as _jax_config

    _jax_config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover — jax-less fallback
    jnp = np
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------- constants
G = lc.NEWTON_CONSTANT
GM_sun = lc.GM_SUN
pc = lc.PARSEC_METER
c = lc.SPEED_OF_LIGHT
MTsun = GM_sun / c**3
EulerGamma = 0.57721566490153286060
Armlength = 2.5e9 / c


# -------------------------------------------------------------- core PN math
def phase(x, sigma, delta, eta, s):
    """3.5PN aligned-spin phase. See PNpedia for the closed form.

    Args:
        x: PN expansion parameter ``(pi M f)**(2/3)``.
        sigma: reduced spin ``(m2 s2 - m1 s1) / M``.
        delta: mass asymmetry ``(m1 - m2) / M``.
        eta: symmetric mass ratio ``m1 m2 / M**2``.
        s: spin parameter ``(m1**2 s1 + m2**2 s2) / M**2``.

    Returns:
        ``Phi_0 - phi`` (waveform phase modulo coalescence offset).
    """
    Phi_0_minus_phi = (
        1
        + x * (3.685515873015873 + (55 * eta) / 12.0)
        + x**1.5 * (-10 * jnp.pi + (235 * s) / 6.0 + (125 * delta * sigma) / 8.0)
        + x**2
        * (
            15.051576475497606
            - 100 * s**2
            + (3085 * eta**2) / 144.0
            - 100 * s * delta * sigma
            - (405 * sigma**2) / 16.0
            + eta * (26.92956349206349 + 100 * sigma**2)
        )
        + x**3.5
        * (
            (-9018232555 * s) / 6.096384e6
            + (125925 * s**3) / 224.0
            - (170978035 * delta * sigma) / 387072.0
            + (379805 * s**2 * delta * sigma) / 448.0
            + (182755 * s * sigma**2) / 448.0
            + (1315 * delta * sigma**3) / 21.0
            + jnp.pi
            * (37.93888721576594 - 200 * s**2 - 200 * s * delta * sigma - (815 * sigma**2) / 16.0)
            + eta**2
            * (
                (-74045 * jnp.pi) / 6048.0
                + (835 * s) / 288.0
                + (7015 * delta * sigma) / 1152.0
                + (285 * s * sigma**2) / 8.0
                + (95 * delta * sigma**3) / 16.0
            )
            + eta
            * (
                (3329545 * s) / 3024.0
                - (95 * s**3) / 8.0
                + (2909765 * delta * sigma) / 5376.0
                - (285 * s**2 * delta * sigma) / 16.0
                - (385825 * s * sigma**2) / 224.0
                - (130615 * delta * sigma**3) / 448.0
                + jnp.pi * (31.292576058201057 + 200 * sigma**2)
            )
        )
        + x**3
        * (
            657.6504345051205
            - (1712 * EulerGamma) / 21.0
            - (160 * jnp.pi**2) / 3.0
            + (7915 * s**2) / 63.0
            - (127825 * eta**3) / 5184.0
            + (2645 * s * delta * sigma) / 56.0
            - (1645 * sigma**2) / 128.0
            + jnp.pi * ((940 * s) / 3.0 + (745 * delta * sigma) / 6.0)
            + eta**2 * (11.003327546296296 - 120 * sigma**2)
            + eta
            * (
                -1290.7459270118156
                + (2255 * jnp.pi**2) / 48.0
                + 120 * s**2
                + 120 * s * delta * sigma
                + (5875 * sigma**2) / 112.0
            )
            - (3424 * jnp.log(2)) / 21.0
            - (856 * jnp.log(x)) / 21.0
        )
        + x**2.5
        * (
            (38645 * jnp.pi) / 1344.0
            - (555605 * s) / 2016.0
            - (15 * s**3) / 4.0
            - (41745 * delta * sigma) / 448.0
            - (45 * s**2 * delta * sigma) / 8.0
            - (45 * s * sigma**2) / 8.0
            - (15 * delta * sigma**3) / 8.0
            + eta
            * (
                (-65 * jnp.pi) / 16.0
                - (45 * s) / 8.0
                + (5 * delta * sigma) / 2.0
                + (45 * s * sigma**2) / 4.0
                + (15 * delta * sigma**3) / 8.0
            )
        )
        * jnp.log(x)
    ) / (32.0 * x**2.5 * eta)

    return Phi_0_minus_phi


def frequency(tau, sigma, delta, eta, s):
    """3.5PN aligned-spin frequency as a function of dimensionless time-to-merger.

    Returns ``M*f`` (dimensionless); divide by ``M`` to get Hz.
    """
    F = (
        (
            (-7729 * jnp.pi) / 172032.0
            + (110869 * s) / 258048.0
            + (8349 * delta * sigma) / 57344.0
            + eta
            * (
                (13 * jnp.pi) / 2048.0
                + (11 * s) / 1024.0
                - (3 * delta * sigma) / 1024.0
            )
        )
        / tau
        + (
            0.016046805176334857
            - (15 * s**2) / 128.0
            + (371 * eta**2) / 16384.0
            - (15 * s * delta * sigma) / 128.0
            - (243 * sigma**2) / 8192.0
            + eta * (0.027599031963045636 + (15 * sigma**2) / 128.0)
        )
        / tau**0.875
        + ((-3 * jnp.pi) / 80.0 + (47 * s) / 320.0 + (15 * delta * sigma) / 256.0)
        / tau**0.75
        + (0.03455171130952381 + (11 * eta) / 256.0) / tau**0.625
        + 1 / (8.0 * tau**0.375)
        + (
            -0.312407295555619
            + (107 * EulerGamma) / 2240.0
            + (53 * jnp.pi**2) / 1600.0
            - (68381 * s**2) / 1.2288e6
            + (235925 * eta**3) / 1.4155776e7
            - (161 * jnp.pi * delta * sigma) / 2048.0
            - (132789 * sigma**2) / 7.340032e6
            + (225 * delta**2 * sigma**2) / 8192.0
            + s * ((-1269 * jnp.pi) / 6400.0 - (763 * delta * sigma) / 49152.0)
            + eta
            * (
                0.7599484976667336
                - (451 * jnp.pi**2) / 16384.0
                - (343 * s**2) / 4096.0
                - (343 * s * delta * sigma) / 4096.0
                + (53647 * sigma**2) / 786432.0
            )
            + eta**2 * (-0.002105781010219029 + (343 * sigma**2) / 4096.0)
            + (107 * jnp.log(2)) / 2240.0
            - (107 * jnp.log(tau)) / 17920.0
        )
        / tau**1.125
    ) / jnp.pi
    return F


def frequency_derivative(tau, sigma, delta, eta, s):
    """Time derivative of :func:`frequency`. Dimensionless; multiply by ``1/M**2`` for Hz/s."""
    dFdt = eta * (
        (
            (-7729 * jnp.pi) / 860160.0
            + (110869 * s) / 1.29024e6
            + (8349 * delta * sigma) / 286720.0
            + eta
            * (
                (13 * jnp.pi) / 10240.0
                + (11 * s) / 5120.0
                - (3 * delta * sigma) / 5120.0
            )
        )
        / tau**2
        + (
            0.0028081909058586
            - (21 * s**2) / 1024.0
            + (2597 * eta**2) / 655360.0
            - (21 * s * delta * sigma) / 1024.0
            - (1701 * sigma**2) / 327680.0
            + eta * (0.004829830593532986 + (21 * sigma**2) / 1024.0)
        )
        / tau**1.875
        + ((-9 * jnp.pi) / 1600.0 + (141 * s) / 6400.0 + (9 * delta * sigma) / 1024.0)
        / tau**1.75
        + (0.004318963913690476 + (11 * eta) / 2048.0) / tau**1.625
        + 3 / (320.0 * tau**1.375)
        + (
            -0.06909744507144285
            + (963 * EulerGamma) / 89600.0
            + (477 * jnp.pi**2) / 64000.0
            - (205143 * s**2) / 1.6384e7
            + (47185 * eta**3) / 1.2582912e7
            - (1449 * jnp.pi * delta * sigma) / 81920.0
            - (1195101 * sigma**2) / 2.9360128e8
            + (405 * delta**2 * sigma**2) / 65536.0
            + s * ((-11421 * jnp.pi) / 256000.0 - (2289 * delta * sigma) / 655360.0)
            + eta
            * (
                0.17098841197501505
                - (4059 * jnp.pi**2) / 655360.0
                - (3087 * s**2) / 163840.0
                - (3087 * s * delta * sigma) / 163840.0
                + (160941 * sigma**2) / 1.048576e7
            )
            + eta**2 * (-0.00047380072729928153 + (3087 * sigma**2) / 163840.0)
            + (963 * jnp.log(2)) / 89600.0
            - (963 * jnp.log(tau)) / 716800.0
        )
        / tau**2.125
    ) / jnp.pi

    return dFdt


def time_to_merger(x, sigma, delta, eta, s):
    """Series-inverted t(x) at 3.5PN. Returns dimensionless ``tc / M``."""
    tc = (
        1
        / eta
        * (
            5 / (256.0 * x**4)
            + (5 * (743 + 924 * eta)) / (64512.0 * x**3)
            + (-48 * jnp.pi + 188 * s + 75 * delta * sigma) / (384.0 * x**2.5)
            + (
                5
                * (
                    -23187 * jnp.pi
                    + 221738 * s
                    + 3276 * jnp.pi * eta
                    + 5544 * s * eta
                    + 75141 * delta * sigma
                    - 1512 * delta * eta * sigma
                )
            )
            / (193536.0 * x**1.5)
            - (
                5
                * (
                    -3058673
                    + 20321280 * s**2
                    - 5472432 * eta
                    - 4353552 * eta**2
                    + 20321280 * s * delta * sigma
                    + 5143824 * sigma**2
                    - 20321280 * eta * sigma**2
                )
            )
            / (1.30056192e8 * x**2)
            + (
                -10052469856691
                + 1530761379840 * EulerGamma
                + 1001432678400 * jnp.pi**2
                - 5883416985600 * jnp.pi * s
                - 2359029657600 * s**2
                + 24236159077900 * eta
                - 882121363200 * jnp.pi**2 * eta
                - 2253223526400 * s**2 * eta
                - 206607970800 * eta**2
                + 462992376000 * eta**3
                - 2331460454400 * jnp.pi * delta * sigma
                - 886871462400 * s * delta * sigma
                - 2253223526400 * s * delta * eta * sigma
                - 492159175200 * sigma**2
                + 733471200000 * delta**2 * sigma**2
                + 1948937760000 * eta * sigma**2
                + 2253223526400 * eta**2 * sigma**2
                + 658084331520 * jnp.log(2)
                + 1201719214080 * jnp.log(4)
                + 765380689920 * jnp.log(x)
            )
            / (1.20171921408e12 * x)
        )
    )

    return tc


def tau_to_x(tau, sigma, delta, eta, s):
    """Direct evaluation of x(tau) used to build :func:`frequency`."""
    x = (
        1
        + (
            (-113868647 * jnp.pi) / 4.3352064e8
            + (24532268147 * s) / 2.60112384e9
            + (21 * jnp.pi * s**2) / 16.0
            - (755 * s**3) / 192.0
            + (281190779 * delta * sigma) / 9.9090432e7
            + (21 * jnp.pi * s * delta * sigma) / 16.0
            - (4499 * s**2 * delta * sigma) / 768.0
            + (1711 * jnp.pi * sigma**2) / 5120.0
            - (33929 * s * sigma**2) / 20480.0
            - (325 * s * delta**2 * sigma**2) / 256.0
            - (24007 * delta * sigma**3) / 49152.0
            + eta**2
            * (
                (294941 * jnp.pi) / 3.87072e6
                + (3641 * s) / 122880.0
                - (6169 * delta * sigma) / 294912.0
            )
            + eta
            * (
                (-31821 * jnp.pi) / 143360.0
                - (33704749 * s) / 5.16096e6
                - (5756657 * delta * sigma) / 1.769472e6
                - (21 * jnp.pi * sigma**2) / 16.0
                + (1259 * s * sigma**2) / 192.0
                + (493 * delta * sigma**3) / 256.0
            )
        )
        / tau**0.875
        + (
            (-11891 * jnp.pi) / 53760.0
            + (357923 * s) / 161280.0
            + (96473 * delta * sigma) / 129024.0
            + eta
            * (
                (109 * jnp.pi) / 1920.0
                - (187 * s) / 5760.0
                - (79 * delta * sigma) / 1536.0
            )
        )
        / tau**0.625
        + (
            0.0770935689090451
            - (5 * s**2) / 8.0
            + (31 * eta**2) / 288.0
            - (5 * s * delta * sigma) / 8.0
            - (81 * sigma**2) / 512.0
            + eta * (0.12607990244708994 + (5 * sigma**2) / 8.0)
        )
        / jnp.sqrt(tau)
        + (-0.2 * jnp.pi + (47 * s) / 60.0 + (5 * delta * sigma) / 16.0) / tau**0.375
        + (0.18427579365079366 + (11 * eta) / 48.0) / tau**0.25
        + (
            -1.6730147506856445
            + (107 * EulerGamma) / 420.0
            + jnp.pi**2 / 6.0
            - (47 * jnp.pi * s) / 48.0
            - (1583 * s**2) / 4032.0
            + (25565 * eta**3) / 331776.0
            - (149 * jnp.pi * delta * sigma) / 384.0
            - (529 * s * delta * sigma) / 3584.0
            - (671 * sigma**2) / 8192.0
            + (125 * delta**2 * sigma**2) / 1024.0
            + eta
            * (
                4.033581021911924
                - (451 * jnp.pi**2) / 3072.0
                - (3 * s**2) / 8.0
                - (3 * s * delta * sigma) / 8.0
                + (2325 * sigma**2) / 7168.0
            )
            + eta**2 * (-0.03438539858217592 + (3 * sigma**2) / 8.0)
            + (107 * jnp.log(2)) / 420.0
            - (107 * jnp.log(tau)) / 3360.0
        )
        / tau**0.75
    ) / (4.0 * tau**0.25)
    return x


def _pn_amp_phase_core(
    m1, m2, D, f_low, s1, s2, times, reference_phase=0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared 3.5PN core: ``(A, Phi, t_subset)`` for an aligned-spin inspiral.

    ``A`` is the GW strain amplitude ``(2 M eta x)/D`` (WITHOUT the
    inclination factors), ``Phi`` is the orbital phase anchored at the
    reference epoch, and ``t_subset`` is the subset of ``times`` strictly
    before the estimated coalescence time ``tc``.

    Both the polarization path (:func:`waveform_generate_h_plus_cross`, which
    folds in ``inc`` and builds ``hp``/``hx``) and the amplitude/phase path
    (:func:`waveform_generate_amp_phase`, which feeds the time-domain
    TDI-on-the-fly response) go through this single core, so the legacy
    ``pyResponse`` and the on-the-fly responses use identical PN math.

    NOTE (2026-06-15): the strain-amplitude leading sign is ``+``. It was once
    ``-``, which made the SOBBH single-link response come out as
    ``-(mojito eta_ij)`` on every link (verified pre-TDI vs the mojito L1
    ``eta_ij``; the shared ``pyResponseTDI`` is correct). Flipped to ``+`` to
    match the LDC / lisagwresponse strain-sign convention mojito uses.
    """
    m1 = m1 * MTsun
    m2 = m2 * MTsun
    D = D * pc / c

    M = m1 + m2
    eta = (m1 * m2) / (M**2)

    v0 = (jnp.pi * M * f_low) ** (1 / 3)
    x0 = v0**2

    sigma = (m2 * s2 - m1 * s1) / M
    s = (m1**2 * s1 + m2**2 * s2) / (M**2)
    delta = (m1 - m2) / M

    # Time-to-coalescence from the reference epoch, computed CONSISTENTLY with
    # the ``tau_to_x`` evolution series used just below: pick tc so that
    # ``tau_to_x(tau_ref) == x0`` (equivalently ``f(reference) == f_low``)
    # EXACTLY. The independent t(x) series ``time_to_merger(x0)`` (used here only
    # as the initial guess) leaves ``f(reference)`` off by ~5e-9 -- a ~0.6 s
    # chirp-timing offset against an external catalogue. (mojito's
    # ``TimeCoalescenceSSBFrame`` agrees with this self-consistent value to
    # ~0.08 s, vs ~0.6 s for ``time_to_merger`` -- i.e. mojito's tc is
    # tau_to_x-consistent too.) A few Newton steps on ``tau_to_x(tau_ref)=x0``
    # from the guess converge immediately (the correction is tiny). tc is a
    # DERIVED quantity, fully fixed by (m1, m2, s1, s2, f_low) -- not a free
    # parameter.
    tc = time_to_merger(x0, sigma, delta, eta, s) * M
    for _ in range(4):
        tau_ref = eta * tc / (5 * M)
        x_ref = tau_to_x(tau_ref, sigma, delta, eta, s)
        h = tau_ref * 1e-6
        dx_dtau = (tau_to_x(tau_ref + h, sigma, delta, eta, s) - x_ref) / h
        tau_ref = tau_ref - (x_ref - x0) / dx_dtau
        tc = 5.0 * M * tau_ref / eta
    tc = float(tc)
    t_subset = times[times < tc]
    tau = eta * (tc - t_subset) / (5 * M)

    x = tau_to_x(tau, sigma, delta, eta, s)
    # Anchor the orbital phase at the REFERENCE epoch (t == reference_time,
    # i.e. tau == tau_ref), NOT at coalescence. `phase(x)` is referenced to
    # merger (phase=0 at x_merger), so `- phase(x) + phase(x_ref)` makes
    # `Phi(reference) == reference_phase` exactly. The catalogue TrueAnomaly
    # is the orbital phase at the reference epoch.
    #
    # We anchor on `phase(tau_to_x(tau_ref))` -- the x the inspiral ACTUALLY
    # has at the reference epoch in its own tau-parametrization. With the
    # tau_to_x-consistent tc above, `tau_to_x(tau_ref) == x0` to machine
    # precision, so this equals `phase(x0)`; the explicit form keeps the anchor
    # exact regardless of the tc solver tolerance.
    tau_ref = eta * tc / (5 * M)  # tau at the reference epoch (t_subset == 0)
    x_ref = tau_to_x(tau_ref, sigma, delta, eta, s)
    Phi = (
        reference_phase
        - phase(x, sigma, delta, eta, s)
        + phase(x_ref, sigma, delta, eta, s)
    )

    A = (2 * M * eta * x) / D

    return A, Phi, t_subset


def waveform_generate_h_plus_cross(
    m1, m2, D, inc, f_low, s1, s2, times, reference_phase=0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``(hp, hx)`` for an aligned-spin SOBBH PN inspiral.

    Args:
        m1: primary mass (solar masses).
        m2: secondary mass (solar masses).
        D: luminosity distance (parsec).
        inc: inclination (rad).
        f_low: starting GW frequency (Hz).
        s1: dimensionless aligned spin of primary.
        s2: dimensionless aligned spin of secondary.
        times: time grid (seconds) on which to evaluate.
        reference_phase: orbital phase at the REFERENCE time (i.e. at
            ``f_low`` / ``x0``), in rad. This is the catalogue convention
            (``TrueAnomaly`` is the phase at the reference epoch), so the
            inspiral is anchored there rather than at coalescence.

    Returns:
        ``(hp, hx, t_subset)`` with ``hp``/``hx`` of length ``len(t_subset)``,
        which is the subset of ``times`` strictly before the estimated tc.
    """
    A, Phi, t_subset = _pn_amp_phase_core(
        m1, m2, D, f_low, s1, s2, times, reference_phase=reference_phase
    )
    C = jnp.cos(inc)
    A_plus = A * (1 + C**2)
    A_cross = A * 2 * C

    hp = A_plus * jnp.cos(2 * Phi)
    hx = A_cross * jnp.cos(2 * Phi - jnp.pi / 2)

    return hp, hx, t_subset


def waveform_generate_amp_phase(
    m1, m2, D, f_low, s1, s2, times, reference_phase=0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``(gw_amp, gw_phase, t_subset)`` for a SOBBH PN inspiral.

    The amplitude/phase representation consumed by the time-domain
    TDI-on-the-fly response
    (:class:`lisatools.response.tdionfly.TDTDIonTheFly`, via
    :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly`). It shares the exact PN core
    (:func:`_pn_amp_phase_core`) with the legacy ``pyResponse`` path
    (:func:`waveform_generate_h_plus_cross`), so the two responses differ only
    in the projection, never in the source physics.

    * ``gw_amp = A = (2 M eta x)/D`` -- the **intrinsic** GW strain amplitude,
      WITHOUT the inclination factors ``(1+cos^2 i)`` / ``2 cos i``. Those are
      applied downstream by the response's ``get_hp_hc`` from the ``inc``
      parameter (exactly as the legacy folds them into ``hp``/``hx``).
    * ``gw_phase = 2*Phi`` -- the GW phase. The response builds
      ``hp ~ A (1+cos^2 i) cos(gw_phase)`` and
      ``hx ~ A (2 cos i) sin(gw_phase)``; with the on-the-fly convention
      ``hSp = -amp (1+cos^2 i) cos(phase)`` the caller passes
      ``phase = gw_phase + pi`` so the signs match
      :func:`waveform_generate_h_plus_cross`.

    Args:
        Same as :func:`waveform_generate_h_plus_cross` minus ``inc`` (which is
        applied by the response, not baked into the amplitude here).

    Returns:
        ``(gw_amp, gw_phase, t_subset)``, each of length ``len(t_subset)``.
    """
    A, Phi, t_subset = _pn_amp_phase_core(
        m1, m2, D, f_low, s1, s2, times, reference_phase=reference_phase
    )
    return A, 2.0 * Phi, t_subset


# ----------------------------------------------------------- LISA-side wrapper


class SOBBHWaveform:
    """Lisa-side SOBBH waveform generator.

    Mirrors the call-signature contract that ``fastlisaresponse.ResponseWrapper``
    expects from EMRI waveform generators:

    * ``__call__(*params, **kwargs) -> (hp, hx)`` arrays evaluated on the
      internal ``times`` grid.
    * Internal time grid built from ``(Tobs, dt, t0)`` at construction time
      so the call stays parameter-only.

    Sampling-basis parameter order (12 params, matching the EMRI structure
    with explicit ``fill_values`` for the two non-sampled entries):

      0  m1   (M_sun)
      1  m2   (M_sun)
      2  s1   (dimensionless aligned spin)
      3  s2   (dimensionless aligned spin)
      4  dist (Gpc)
      5  inc  (rad)
      6  f_low (Hz)              -- starting GW frequency
      7  lam  (ecliptic longitude, rad)
      8  beta (ecliptic latitude, rad)
      9  psi  (polarization, rad)
      10 phi0 (orbital phase at the REFERENCE epoch / f_low, rad --
            i.e. the catalogue ``TrueAnomaly``, NOT the coalescence phase)
      11 t_shift (seconds)       -- time offset, default 0

    The wrapper applies polarization rotation; ``ResponseWrapper`` handles
    the sky-direction projection onto LISA TDI channels.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t0: float = 0.0,
        reference_time: Optional[float] = None,
        force_backend: str = "cpu",
        pad_zeros: bool = True,
    ):
        self.Tobs = Tobs
        self.dt = dt
        self.t0 = t0
        # ``reference_time`` is the absolute epoch at which ``f_low`` is
        # defined (the fixed catalogue reference, e.g. mojito's
        # ``MOJITO_REFERENCE_TIME``). It is deliberately decoupled from the
        # data-window start ``t0``: the data array begins at
        # ``t0 = reference_time + trim_duration``, so at the first sample the
        # binary has already evolved ``trim_duration`` past ``f_low``. The PN
        # inspiral measures time from ``reference_time`` (see ``__call__``).
        # Defaults to ``t0`` (f_low at the window start) for the synthetic
        # path, which has no separate catalogue epoch.
        self.reference_time = t0 if reference_time is None else reference_time
        self.force_backend = force_backend
        self.pad_zeros = pad_zeros
        N = int(round(Tobs / dt))
        self._N = N
        self._times_np = np.arange(N) * dt + t0
        # JAX-friendly copy for the inner waveform call.
        self._times = jnp.asarray(self._times_np)

    @property
    def N(self) -> int:
        return self._N

    @property
    def times(self) -> np.ndarray:
        return self._times_np

    def __call__(
        self,
        m1,
        m2,
        s1,
        s2,
        dist,
        inc,
        f_low,
        lam,
        beta,
        psi,
        phi0,
        t_shift=0.0,
        **kwargs,
    ):
        """Evaluate the polarization-rotated waveform on the internal time grid.

        Returns a single complex array ``h = hp + 1j * hx`` so the call
        signature matches what ``fastlisaresponse.ResponseWrapper`` expects
        (it reads ``h.real`` / ``h.imag`` to get the two polarizations).
        Set ``flip_hx=True`` on ``ResponseWrapper`` to get the standard
        ``h.real - 1j h.imag`` convention.

        ``dist`` is in Gpc; converted to parsec inside.
        ``lam`` / ``beta`` are accepted but unused here — they're forwarded
        through ``ResponseWrapper`` for sky-projection.
        """
        del lam, beta  # consumed by ResponseWrapper, not the source waveform
        # ResponseWrapper injects ``T`` and ``dt`` into kwargs; ignore.
        kwargs.pop("T", None)
        kwargs.pop("dt", None)
        kwargs.pop("convert_to_ra_dec", None)
        # The PN inspiral measures time from ``reference_time`` (the fixed
        # catalogue epoch where ``f_low`` is defined), NOT from the data-window
        # start ``t0``.  ``self._times`` is the absolute data grid
        # (``arange(N)*dt + t0``); subtracting ``reference_time`` yields the PN
        # time, which at the first sample equals ``t0 - reference_time ==
        # trim_duration`` -- i.e. the binary has already evolved past ``f_low``
        # by the time the window starts (mirrors the GB
        # ``evolve_galactic_binary(t_ref -> t_ref + trim)`` convention).
        # Without this, a large absolute grid (mojito mode) pushes every sample
        # past the merger cut ``times < tc`` -> all zeros for short-``tc``
        # sources.  Synthetic mode has ``reference_time == t0``, so this reduces
        # to a relative ``arange(N)*dt`` grid.  Absolute placement on the data
        # grid is handled downstream by ResponseWrapper via its own ``t0``.
        times = self._times - self.reference_time
        if t_shift != 0.0:
            times = times - float(t_shift)

        hp_short, hx_short, _t_subset = waveform_generate_h_plus_cross(
            float(m1),
            float(m2),
            float(dist) * 1.0e9,  # Gpc -> parsec
            float(inc),
            float(f_low),
            float(s1),
            float(s2),
            times,
            reference_phase=float(phi0),
        )

        n_active = hp_short.shape[0]
        if self.pad_zeros and n_active < self._N:
            hp_full = np.zeros(self._N)
            hx_full = np.zeros(self._N)
            hp_full[:n_active] = np.asarray(hp_short)
            hx_full[:n_active] = np.asarray(hx_short)
        else:
            hp_full = np.asarray(hp_short)
            hx_full = np.asarray(hx_short)

        # apply polarization rotation
        c2psi = np.cos(2.0 * float(psi))
        s2psi = np.sin(2.0 * float(psi))
        hp_rot = hp_full * c2psi - hx_full * s2psi
        hx_rot = hp_full * s2psi + hx_full * c2psi

        return hp_rot + 1j * hx_rot

    def compute_amp_phase(
        self,
        m1,
        m2,
        s1,
        s2,
        dist,
        f_low,
        phi0,
        times=None,
        t_shift=0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the intrinsic GW amplitude / phase on a time grid.

        The amplitude/phase counterpart of :meth:`__call__`: instead of the
        polarization-rotated complex strain (which the legacy ``pyResponse``
        path consumes via ``ResponseWrapper``), this returns the inclination-
        free GW amplitude ``A`` and the GW phase ``2*Phi`` that the
        time-domain TDI-on-the-fly response
        (:class:`bbhx.sobbhtdionfly.SOBBHTDIonFly`) consumes. Both paths share
        the exact PN core, so they are the *same* waveform tool differing only
        in output representation -- ``inc``/``psi`` are applied by the
        response, not here.

        The PN time is measured from ``reference_time`` (the fixed catalogue
        epoch where ``f_low`` is defined), identically to :meth:`__call__`.

        Args:
            m1, m2, s1, s2, dist, f_low, phi0: as in the sampling basis
                (``dist`` in Gpc, ``phi0`` the reference-epoch orbital phase).
            times (np.ndarray, optional): absolute time grid (sec). Defaults to
                the internal data grid ``arange(N)*dt + t0``.
            t_shift (float, optional): extra time offset (sec). (default: 0)

        Returns:
            ``(abs_times, gw_amp, gw_phase)`` -- ``abs_times`` are the absolute
            times (same epoch convention as the input grid) strictly before
            ``tc``; ``gw_amp = A``; ``gw_phase = 2*Phi``.
        """
        if times is None:
            times = self._times_np
        times = np.asarray(times, dtype=float)
        # PN time measured from the reference epoch (see __call__).
        pn_times = times - self.reference_time - float(t_shift)

        gw_amp, gw_phase, t_subset_pn = waveform_generate_amp_phase(
            float(m1),
            float(m2),
            float(dist) * 1.0e9,  # Gpc -> parsec
            float(f_low),
            float(s1),
            float(s2),
            pn_times,
            reference_phase=float(phi0),
        )
        # Map the (pre-tc) PN-time subset back onto the input epoch.
        abs_times = np.asarray(t_subset_pn) + self.reference_time + float(t_shift)
        return abs_times, np.asarray(gw_amp), np.asarray(gw_phase)
