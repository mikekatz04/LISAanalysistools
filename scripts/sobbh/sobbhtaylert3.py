############################
### Originally from Diganta Bandopadhyay
############################



import numpy as np
import jax.numpy as jnp

from scipy.optimize import root_scalar

import lisaconstants as lc

from jax import config

config.update("jax_enable_x64", True)


### SPINLESS WAVEFORM ####

import jax

##### Physical constants #####
G = lc.NEWTON_CONSTANT
GM_sun = lc.GM_SUN
pc = lc.PARSEC_METER
c = lc.SPEED_OF_LIGHT
MTsun = GM_sun / c**3

EulerGamma = 0.57721566490153286060  # Euler-Mascheroni constant

# Armlength of LISA in seconds
Armlength = 2.5e9 / c


# TODO: Probably should use proper fractions not hacky floats like this, a consequency of using fortranform.
# Remember x = v^2


def phase(x, sigma, delta, eta, s):
    """'
    3.5PN in aligned spin effects. Circular.

    Link to PNpedia for the expression:
    https://github.com/davidtrestini/PNpedia/blob/275c95c8f6765d5628eeb2549d17250cd16e6617/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform/phase.txt

    Parameters: 
        x (float or array of floats): The PN expansion parameter, defined as (pi*M*f)^(2/3), where M is the total mass of the binary and f is the GW frequency.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns: 
        Phi_0_minus_phi (float or array of floats): The value of Phi_0 - phi at the given x, sigma, delta, eta, and s.
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
            * (
                37.93888721576594
                - 200 * s**2
                - 200 * s * delta * sigma
                - (815 * sigma**2) / 16.0
            )
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
    """
    3.5PN in aligned spin effects. Circular.

    Obtained from x(tau) = F(tau).

    Link to PNpedia for the expression:
    https://github.com/davidtrestini/PNpedia/blob/275c95c8f6765d5628eeb2549d17250cd16e6617/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform/chirp.txt

    NOTE: Dimensionless, should be divided by 1/M to get to Hz. 

    Parameters: 
        tau (float or array of floats): Defined as tau = (eta / 5) * (t-t_0), where t_0 is the initial time and t is the time variable. It is a dimensionless time parameter that measures the time to coalescence in units of the symmetric mass ratio eta.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns: 
        F (float or array of floats): The value of frequency at the given tau, sigma, delta, eta, and s.
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
    """'
    Differentiated form of frequency above.

    With respect to t not tau. (Uses Chain rule to get dF/dt from dF/dtau, where tau is a function of t, using dTau/dt = -eta/5).

    NOTE: Dimensionless, should be divided by 1/M^2 to get to Hz^-2. 

    Parameters: 
        tau (float or array of floats): Defined as tau = (eta / 5) * (t-t_0), where t_0 is the initial time and t is the time variable. It is a dimensionless time parameter that measures the time to coalescence in units of the symmetric mass ratio eta.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns: 
        dFdt (float or array of floats): The value of the derivative of frequency with respect to time at the given tau, sigma, delta, eta, and s.
    """
    dFdt = eta*(
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


def calculate_T(v, v0, eta, beta_15, beta_25, sigma):
    """
    NOTE: Not Exact for the PN order being used here, can be used as a sense check though.

    Calculates the value of T using the (TaylorT2) Equation 6.7b from arXiv:1605.00304v2.
    Uses Phase from arXiv:2108.05861 and SPA to calculate t-f correction from upto 2.5PN spin effects.

    Parameters:
      v (float or array of floats): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      eta (float): The symmetric mass ratio of the binary system.
      beta_15 (float): The 1.5 PN Spin-orbit term.
      beta_25 (float): The 2.5 PN Spin-orbit term.
      sigma (float): The 2 PN spin-spin term.

    Returns:
      T (float or array of floats): The calculated value(s) of T at fs provided into v.
    """
    v_2 = v**2
    v_3 = v**3
    v_4 = v**4
    v_5 = v**5
    v_6 = v**6

    term1 = (
        1
        + ((743 / 252) + (11 / 3) * eta) * v_2
        - (32 / 5) * jnp.pi * v_3
        + ((3058673 / 508032) + (5429 / 504) * eta + (617 / 72) * eta**2) * v_4
        + (-(7729 / 252) + (13 / 3) * eta) * jnp.pi * v_5
    )

    term2 = (
        (-10052469856691 / 23471078400)
        + (6848 / 105) * EulerGamma
        + (128 / 3) * jnp.pi**2
        + ((3147553127 / 3048192) - (451 / 12) * jnp.pi**2) * eta
        - (15211 / 1728) * eta**2
        + (25565 / 1296) * eta**3
        + (3424 / 105) * jnp.log(16 * v_2)
    ) * v_6

    term3 = (
        ((-15419335 / 127008) - (75703 / 756) * eta + (14809 / 378) * eta**2)
        * jnp.pi
        * v**7
    )

    spin_aligned_terms = (
        -2 * v_4 * sigma
        + 1 / 315 * v_3 * (504 + 5 * v_2 * (743 + 924 * eta)) * beta_15
        - (8 * v_5 * beta_25) / 3
    )

    T = term1 + term2 + term3 + spin_aligned_terms
    return T


def time_to_merger(x, sigma, delta, eta, s):
    """
    Series inverstion of x(tau) (used for frequency) to get t(x) in the form (NOTE: not tau, t):

    tc(x) = t_0 - t(x) (I think should be the other way around(?))

    Parameters: 
        x (float or array of floats): The PN expansion parameter, defined as (pi*M*f)^(2/3), where M is the total mass of the binary and f is the GW frequency.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns: 
        tc (float or array of floats): The value of time to merger at the given x, sigma, delta, eta, and s.
    """

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
    """
    Directly using x(tau) used for the frequency function

    PNpedia link:
    https://github.com/davidtrestini/PNpedia/blob/275c95c8f6765d5628eeb2549d17250cd16e6617/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform/chirp.txt

    Parameters: 
        tau (float or array of floats): Defined as tau = (eta / 5) * (t-t_0), where t_0 is the initial time and t is the time variable. It is a dimensionless time parameter that measures the time to coalescence in units of the symmetric mass ratio eta.
        sigma (float): Reduced spin parameter (m2 * s2 - m1 * s1) / M
        delta (float): Mass difference parameter (m1 - m2) / M
        eta (float): Symmetric mass ratio m1 * m2 / M^2
        s (float): Spin parameter (m1**2 * s1 + m2**2 * s2) / (M**2)

    Returns: 
        x (float or array of floats): The value of the dimensionless frequency at the given tau, sigma, delta, eta, and s.
    """

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


def waveform_generate_h_plus_cross_T2_time_correspondence(
    m1, m2, D, inc, f_low, s1, s2, times, coallesence_phase=0
):
    """
    NOTE: INACCURATE FOR THE PN ORDER BEING USED HERE.

    Uses T2 time to merger to estimate tc from f_low.
    - Uses a inverse SPA form of the TF correspondence to get t-f correction from spin effects. Only upto 2.5PN.

    Assuming aligned spins, so only z components of the spins are non-zero.
    I.e s1 = s1z, s2 = s2z, where s1z and s2z are the dimensionless spin parameters of the two black holes.

    f here is *GW* frequency not orbital frequency.

    https://github.com/davidtrestini/PNpedia/tree/main/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform

    Parameters:
        m1 (float): Mass of the first black hole in solar masses.
        m2 (float): Mass of the second black hole in solar masses.
        D (float): Distance to the binary system in parsecs.
        inc (float): Inclination angle of the binary system in radians.
        f_low (float): Initial GW frequency in Hz.
        s1 (float): Dimensionless spin parameter of the first black hole (s1z).
        s2 (float): Dimensionless spin parameter of the second black hole (s2z).
        times (array of floats): Array of time values at which to compute the waveform, in seconds.
        coallesence_phase (float, optional): Coalescence phase in radians. Default is 0.

    Returns:
        h_plus (array of floats): The plus polarization of the gravitational waveform at the given times
        h_cross (array of floats): The cross polarization of the gravitational waveform at the given times
        t_subset (array of floats): The subset of the input times that are before the estimated time to merger tc.

    """

    m1 = m1 * MTsun
    m2 = m2 * MTsun
    D = (D) * pc / c

    M = m1 + m2
    # Symmetric mass ratio
    eta = (m1 * m2) / (M**2)

    v0 = (jnp.pi * M * f_low) ** (1 / 3)

    # Spin quantities to be used in the T2 time to merger function, which is used to estimate tc from f_low.

    # Compute the 1.5 PN term from s1 and s2 (Spin-orbit)
    beta_15 = s1 * (113 / 12 * (m1**2) / (M**2) + 25 / 4 * eta) + s2 * (
        113 / 12 * (m2**2) / (M**2) + 25 / 4 * eta
    )

    # Compute the 2.5 PN term from s1 and s2 (Spin-orbit)
    beta_25 = s1 * (
        (m1**2) / (M**2) * (-31319 / 1008 + 1159 / 24 * eta)
        + eta * (-809 / 84 + 281 / 8 * eta)
    ) + s2 * (
        (m2**2) / (M**2) * (-31319 / 1008 + 1159 / 24 * eta)
        + eta * (-809 / 84 + 281 / 8 * eta)
    )

    # Compute the 2 PN term from s1 and s2 (Spin-spin) (sigma)

    # Standard spin-spin term
    simga_s1s2 = 474 / 48 * eta * s1 * s2

    # Quadrupole - monopole term
    sigma_qm = 5 * (s1**2 * (m1**2) / (M**2) + s2**2 * (m2**2) / (M**2))

    # Self-spin interaction term
    sigma_self_spin = 1 / 16 * (s1**2 * (m1**2) / (M**2) + s2**2 * (m2**2) / (M**2))

    # Add them all together to get the 2PN term
    sigma = simga_s1s2 + sigma_qm + sigma_self_spin

    # Use T2 to estimate tc
    tc = (
        5
        / 256
        * M
        / eta
        * 1
        / (v0) ** 8
        * calculate_T(v0, v0, eta, beta_15, beta_25, sigma)
    )

    # Subset to times before merger.
    t_subset = times[times < tc]

    # One of the reduced spin parameter: sigma
    sigma = (m2 * s2 - m1 * s1) / M

    # The other reduced spin parameter: s
    s = (m1**2 * s1 + m2**2 * s2) / (M**2)

    # relative mass difference
    delta = (m1 - m2) / M

    # tau array derived once we know tc
    tau = eta * (tc - t_subset) / (5 * M)  # eta*(t_subset - tc)/(5*M)

    f_t = 1 / M * frequency(tau, sigma, delta, eta, s)

    x = (jnp.pi * M * f_t) ** (2 / 3)

    Phi = coallesence_phase - phase(x, sigma, delta, eta, s)

    # Remember x = v^2
    A = -(2 * M * eta * x) / D

    C = jnp.cos(inc)

    A_plus = A * (1 + C**2)
    A_cross = A * 2 * C

    h_plus = A_plus * jnp.cos(2 * Phi)
    # pi/2 from polarization convention.
    h_cross = A_cross * jnp.cos(2 * Phi - jnp.pi / 2)

    return (h_plus, h_cross, t_subset)


def waveform_generate_h_plus_cross(
    m1, m2, D, inc, f_low, s1, s2, times, coallesence_phase=0
):
    """
    Note: Uses correct form of the time-to-merger tc for the PN order in phase etc. 
    Assuming aligned spins, so only z components of the spins are non-zero.
    I.e s1 = s1z, s2 = s2z, where s1z and s2z are the dimensionless spin parameters of the two black holes.

    f here is *GW* frequency not orbital frequency.

    https://github.com/davidtrestini/PNpedia/tree/main/Core%20post-Newtonian%20quantities/Circular%20orbits/Spinning/Nonprecessing/Without%20tidal%20effects/Waveform


    Parameters:
        m1 (float): Mass of the first black hole in solar masses.
        m2 (float): Mass of the second black hole in solar masses.
        D (float): Distance to the binary system in parsecs.
        inc (float): Inclination angle of the binary system in radians.
        f_low (float): Initial GW frequency in Hz.
        s1 (float): Dimensionless spin parameter of the first black hole (s1z).
        s2 (float): Dimensionless spin parameter of the second black hole (s2z).
        times (array of floats): Array of time values at which to compute the waveform, in seconds.
        coallesence_phase (float, optional): Coalescence phase in radians. Default is 0.
        
    Returns:
        h_plus (array of floats): The plus polarization of the gravitational waveform at the given times
        h_cross (array of floats): The cross polarization of the gravitational waveform at the given times
        t_subset (array of floats): The subset of the input times that are before the estimated time to merger tc.

    """

    m1 = m1 * MTsun
    m2 = m2 * MTsun
    D = (D) * pc / c

    M = m1 + m2
    # Symmetric mass ratio
    eta = (m1 * m2) / (M**2)

    v0 = (jnp.pi * M * f_low) ** (1 / 3)

    x0 = v0**2

    # One of the reduced spin parameter: sigma
    sigma = (m2 * s2 - m1 * s1) / M

    # The other reduced spin parameter: s
    s = (m1**2 * s1 + m2**2 * s2) / (M**2)

    # relative mass difference
    delta = (m1 - m2) / M

    tc = time_to_merger(x0, sigma, delta, eta, s) * M
    t_subset = times[times < tc]
    # tau array derived once we know tc
    tau = eta * (tc - t_subset) / (5 * M)  # eta*(t_subset - tc)/(5*M)

    # f_t = 1/M*frequency(tau,sigma,delta,eta,s)

    x = tau_to_x(tau, sigma, delta, eta, s)

    Phi = coallesence_phase - phase(x, sigma, delta, eta, s)

    # Remember x = v^2
    A = -(2 * M * eta * x) / D

    C = jnp.cos(inc)

    A_plus = A * (1 + C**2)
    A_cross = A * 2 * C

    h_plus = A_plus * jnp.cos(2 * Phi)
    # pi/2 from polarization convention.
    h_cross = A_cross * jnp.cos(2 * Phi - jnp.pi / 2)

    return (h_plus, h_cross, t_subset)


def waveform_generate_phase_only(
    m1, m2, s1, s2, f_low,f_max, T_obs, coallesence_phase=0
):
    """
    Note: Uses correct form of the time-to-merger tc for the PN order in phase etc. 
    Assuming aligned spins, so only z components of the spins are non-zero.
    I.e s1 = s1z, s2 = s2z, where s1z and s2z are the dimensionless spin parameters of the two black holes.

    f here is *GW* frequency not orbital frequency.

    Assumes t0=0 and computes phase for t0 and the time at which it leaves the band 

    """
    m1 = m1 * MTsun
    m2 = m2 * MTsun

    M = m1 + m2
    # Symmetric mass ratio
    eta = (m1 * m2) / (M**2)

    v0 = (jnp.pi * M * f_low) ** (1 / 3)

    x0 = v0**2

    # One of the reduced spin parameter: sigma
    sigma = (m2 * s2 - m1 * s1) / M

    # The other reduced spin parameter: s
    s = (m1**2 * s1 + m2**2 * s2) / (M**2)

    # relative mass difference
    delta = (m1 - m2) / M


    tc = time_to_merger(x0, sigma, delta, eta, s) * M

    x_max = (jnp.pi * M * f_max) ** (2 / 3)

    time_to_merge_from_leaving_band = time_to_merger(x_max, sigma, delta, eta, s) * M

    time_to_leaving_band = tc - time_to_merge_from_leaving_band

    # Only compute for the times before leaving the band, i.e before the time to merger corresponding to f_max.
    # Min between tobs and time to band just takes whever happens first. 
    t_subset = jnp.array([0, min(time_to_leaving_band, T_obs)])

    # tau array derived once we know tc
    tau = eta * (tc - t_subset) / (5 * M)  # eta*(t_subset - tc)/(5*M)

    x = tau_to_x(tau, sigma, delta, eta, s)

    Phi = coallesence_phase - phase(x, sigma, delta, eta, s)

    return (2 * Phi,time_to_leaving_band)


# ---------------------------------------------------------------------------
# JAX TDI-on-the-fly plug-in API.
#
# The `fastlisaresponse.jax` TDI-on-the-fly engine consumes a source via the
# `JaxAmpPhaseSource` interface — concretely, two scalar functions
# `amplitude(t, params)` and `phase(t, params)` of a single time and a flat
# parameter array. The functions below expose exactly that for SOBBH so
# the JAX `SOBBHTDIonTheFly` path can drive directly off this file —
# matching the C++ `SOBBHTDIonTheFly::sobbh_amplitude` /
# `SOBBHTDIonTheFly::sobbh_phase` in `lisa-on-gpu/.../TDIonTheFly.cu`.
#
# Parameter layout (length 11), identical to the C++ kernel:
#     params[0]  = m1            (solar masses)
#     params[1]  = m2            (solar masses)
#     params[2]  = s1            (dimensionless aligned spin, primary)
#     params[3]  = s2            (dimensionless aligned spin, secondary)
#     params[4]  = distance      (parsecs)
#     params[5]  = f_low         (GW frequency at t_ref, Hz)
#     params[6]  = phi_c         (reference orbital phase, rad)
#     params[7]  = inc           (inclination, rad)         <- consumed by projection
#     params[8]  = psi           (polarization, rad)        <- consumed by projection
#     params[9]  = lam           (ecliptic longitude, rad)  <- consumed by projection
#     params[10] = beta          (ecliptic latitude, rad)   <- consumed by projection
#
# All routines are pure JAX (no Python branching on tracer values), so they
# JIT, vmap, and grad cleanly.
# ---------------------------------------------------------------------------

# IAU 2015 / lisaconstants values, mirroring the C++ MTSUN_SOBBH / PARSEC_SOBBH / C_SI.
_MT_SUN = MTsun
_PARSEC = pc
_C_LIGHT = c

# Parameter indices (named so call-sites stay readable).
M1_INDEX = 0
M2_INDEX = 1
S1_INDEX = 2
S2_INDEX = 3
DISTANCE_INDEX = 4
F_LOW_INDEX = 5
PHI_C_INDEX = 6
INC_INDEX = 7
PSI_INDEX = 8
LAM_INDEX = 9
BETA_INDEX = 10
N_PARAMS = 11


def _pn_quantities(params):
    """Compute ``(M, eta, sigma, s, delta, tc)`` -- the C++ setup.

    Mirrors ``SOBBHTDIonTheFly::sobbh_amplitude/phase/f``
    (``TDIonTheFly.cu:5019-5048`` etc.) exactly.

    Returns ``(M, eta, sigma, s_pn, delta, tc)`` with ``M`` in seconds
    (i.e. M = (m1 + m2) * MTsun) and ``tc`` in seconds.
    """
    m1_sec = params[M1_INDEX] * _MT_SUN
    m2_sec = params[M2_INDEX] * _MT_SUN
    s1 = params[S1_INDEX]
    s2 = params[S2_INDEX]
    f_low = params[F_LOW_INDEX]

    M = m1_sec + m2_sec
    eta = (m1_sec * m2_sec) / (M * M)
    sigma = (m2_sec * s2 - m1_sec * s1) / M
    s_pn = (m1_sec * m1_sec * s1 + m2_sec * m2_sec * s2) / (M * M)
    delta = (m1_sec - m2_sec) / M

    v0 = jnp.power(jnp.pi * M * f_low, 1.0 / 3.0)
    x0 = v0 * v0
    tc = time_to_merger(x0, sigma, delta, eta, s_pn) * M
    return M, eta, sigma, s_pn, delta, tc


def amplitude(t, params):
    """SOBBH scalar amplitude ``2 M eta x / D_sec`` at time(s) ``t``.

    Matches ``SOBBHTDIonTheFly::sobbh_amplitude`` byte-for-byte (the C++
    constants were ported from this module). For ``t >= tc`` returns 0;
    expressed as ``jnp.where`` so autograd keeps a meaningful zero
    gradient past coalescence rather than NaN-ing.
    """
    M, eta, sigma, s_pn, delta, tc = _pn_quantities(params)
    D_pc = params[DISTANCE_INDEX]
    D_sec = D_pc * _PARSEC / _C_LIGHT

    tau_raw = eta * (tc - t) / (5.0 * M)
    tau = jnp.where(tau_raw > 0.0, tau_raw, 1.0)
    x = tau_to_x(tau, sigma, delta, eta, s_pn)
    amp = 2.0 * M * eta * x / D_sec
    return jnp.where(t < tc, amp, 0.0)


def gw_phase(t, params):
    """SOBBH GW phase ``2 (phi_c - phase_fn(x))`` at time(s) ``t``.

    Matches ``SOBBHTDIonTheFly::sobbh_phase``. Returns 0 past coalescence
    (consistent with ``amplitude`` returning 0 there).

    Note the name -- ``phase`` at module level is the PN phase coefficient
    function and we must not shadow it.
    """
    M, eta, sigma, s_pn, delta, tc = _pn_quantities(params)
    phi_c = params[PHI_C_INDEX]

    tau_raw = eta * (tc - t) / (5.0 * M)
    tau = jnp.where(tau_raw > 0.0, tau_raw, 1.0)
    x = tau_to_x(tau, sigma, delta, eta, s_pn)
    phi_orbital = phi_c - phase(x, sigma, delta, eta, s_pn)
    ph = 2.0 * phi_orbital
    return jnp.where(t < tc, ph, 0.0)


def gw_frequency_t(t, params):
    """SOBBH GW frequency ``2 x^(3/2) / (pi M)`` at time(s) ``t`` (Hz).

    Matches ``SOBBHTDIonTheFly::sobbh_f``.
    """
    M, eta, sigma, s_pn, delta, tc = _pn_quantities(params)
    tau_raw = eta * (tc - t) / (5.0 * M)
    tau = jnp.where(tau_raw > 0.0, tau_raw, 1.0)
    x = tau_to_x(tau, sigma, delta, eta, s_pn)
    f_val = 2.0 * jnp.power(x, 1.5) / (jnp.pi * M)
    return jnp.where(t < tc, f_val, 0.0)
