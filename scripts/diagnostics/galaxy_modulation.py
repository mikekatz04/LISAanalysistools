#!/usr/bin/env python3
"""Galaxy foreground modulation for LISA, ported from glass_galaxy.c.

Computes the time-dependent modulation of the X/Y/Z TDI channel auto- and
cross-spectra produced by the (rotating) LISA antenna pattern sweeping across
the anisotropic galactic confusion foreground over one orbit.

Pipeline (mirrors initialize_galaxy_modulation + galaxy_modulation in C):

  1. Build a HEALPix grid (RING, NSIDE) in ecliptic coordinates.
  2. For each pixel, rotate ecliptic -> galactic and integrate the galactic
     binary mass density along the line of sight -> sky brightness map.
  3. Decompose the (ecliptic) map into real spherical-harmonic coefficients
     a_lm up to LMAX, using the same iterative pixel-shape-corrected transform
     as the C code.
  4. Contract the a_lm with precomputed analytic antenna-response kernels
     XX(alpha,beta) / XY(alpha,beta) (functions of orbital phase alpha and
     constellation arm angle beta) at each time sample.
  5. Normalize so that mean_t (XX+YY+ZZ)/3 = 1 and write modulation.dat with
     columns: t  XX  YY  ZZ  XY  XZ  YZ.

Dependencies: numpy, scipy, astropy-healpix.
"""

import argparse
import math

import numpy as np
import astropy.units as u
import astropy_healpix as ah
from scipy.integrate import quad

# --- constants (match glass_galaxy.h / glass_constants.h) ----------------------
NSIDE = 16                       # HEALPix resolution
LMAX = 4                         # max multipole of the modulation
NPIX = 12 * NSIDE * NSIDE        # = 3072
YEAR = 3.15581497632e7           # seconds
RTPI = math.sqrt(math.pi)
SQ3 = math.sqrt(3.0)
PIXAREA = 4.0 * math.pi / NPIX

# sqrt prefactors appearing in the response kernels
sq5, sq10, sq15 = math.sqrt(5.0), math.sqrt(10.0), math.sqrt(15.0)
sq30, sq35, sq70, sq105 = math.sqrt(30.0), math.sqrt(35.0), math.sqrt(70.0), math.sqrt(105.0)

# default Milky Way model (galaxy_params in glass_noise_model.c)
GALAXY_PARAMS = dict(
    A=0.25,     # bulge fraction
    Rb=0.8,     # bulge radius (kpc)
    Rd=2.5,     # disk radius (kpc)
    Zd=0.4,     # disk scale height (kpc)
    Rgc=7.2,    # Sun -> galactic center distance (kpc)
    Rcut=3.5,   # inner cutoff: sources inside are individually resolved (kpc)
)


# --- galactic density and line-of-sight integral -----------------------------
def galaxy_distribution(x0, x1, x2, A, Rb, Rd, Zd):
    """Unnormalized galactic binary mass density (bulge + disk)."""
    rsq = x0 * x0 + x1 * x1 + x2 * x2
    u_ = np.hypot(x0, x1)
    s = 1.0 / np.cosh(x2 / Zd)
    return A * np.exp(-rsq / (Rb * Rb)) + (1.0 - A) * np.exp(-u_ / Rd) * s * s


def galaxy_integration(theta, phi, A, Rb, Rd, Zd, Rgc, Rcut, rmax=200.0):
    """Integrate the density along a line of sight in galactic coordinates.

    The density is even in z and in sin(phi), so |cos theta| / |sin phi| are
    used (as in the C integrand)."""
    st = math.sin(theta)
    ct = math.sqrt(1.0 - st * st)
    cp = math.cos(phi)
    sp = math.sqrt(1.0 - cp * cp)

    def integrand(r):
        x0 = r * cp * st - Rgc
        x1 = r * sp * st
        x2 = r * ct
        return galaxy_distribution(x0, x1, x2, A, Rb, Rd, Zd)

    val, _ = quad(integrand, Rcut, rmax, limit=200)
    return val


# --- ecliptic <-> galactic rotation -------------------------------------------
# rotate_ecliptogal from glass_galaxy.c (ecliptic unit vector -> galactic)
_R_ECL2GAL = np.array([
    [-0.05487556043, -0.99382137890, -0.09647662818],
    [0.4941094278, -0.1109907351, 0.8622858751],
    [-0.8676661492, -0.00035159077, 0.4971471918],
])


def build_sky_map(params):
    """Return (theta_ecl, phi_ecl, sky) for all HEALPix pixels.

    theta_ecl/phi_ecl are the ecliptic pixel centers used by the spherical
    harmonic transform; sky is the galaxy brightness in that direction."""
    ipix = np.arange(NPIX)
    lon, lat = ah.healpix_to_lonlat(ipix, NSIDE, order="ring")
    theta_ecl = np.pi / 2.0 - lat.to_value(u.rad)   # colatitude
    phi_ecl = lon.to_value(u.rad)

    # ecliptic unit vectors -> galactic
    xe = np.array([
        np.sin(theta_ecl) * np.cos(phi_ecl),
        np.sin(theta_ecl) * np.sin(phi_ecl),
        np.cos(theta_ecl),
    ])
    xg = _R_ECL2GAL @ xe
    theta_gal = np.arccos(xg[2])
    phi_gal = np.arctan2(xg[1], xg[0])
    phi_gal[phi_gal < 0.0] += 2.0 * np.pi

    sky = np.array([
        galaxy_integration(theta_gal[i], phi_gal[i], **params)
        for i in range(NPIX)
    ])
    return theta_ecl, phi_ecl, sky


# --- spherical harmonic transform (matches sphharm / map2alm / alm2map) -------
def scaled_legendre(z):
    """Scaled associated Legendre P_lm(z) including pixel area + normalization.

    P_lm carries the sqrt((2l+1)/4pi (l-m)!/(l+m)!) prefactor and the pixel
    area 4pi/Npix, exactly as plms() in the C code."""
    P = np.zeros((LMAX + 1, LMAX + 1, z.size))
    sq = 1.0 / np.sqrt(1.0 - z * z)
    P[0, 0] = 1.0
    P[1, 0] = z
    for l in range(1, LMAX):                       # m = 0 recurrence (Legendre)
        P[l + 1, 0] = ((2 * l + 1) * z * P[l, 0] - l * P[l - 1, 0]) / (l + 1)
    for l in range(1, LMAX + 1):                   # raise m
        for m in range(0, l):
            P[l, m + 1] = sq * ((l - m) * z * P[l, m] - (l + m) * P[l - 1, m])
    for l in range(LMAX + 1):                      # apply prefactor + pixel area
        for m in range(l + 1):
            pref = math.sqrt((2 * l + 1) / (4.0 * math.pi)
                             * math.gamma(l - m + 1) / math.gamma(l + m + 1))
            P[l, m] *= PIXAREA * pref
    return P


def sphharm(sky, P, cosm, sinm):
    """Real SHT with two iterations of pixel-shape error correction."""
    def map2alm(field, almR, almI):                # accumulates into alm
        for l in range(LMAX + 1):
            for m in range(l + 1):
                almR[l, m] += np.sum(P[l, m] * cosm[m] * field)
                almI[l, m] += np.sum(P[l, m] * sinm[m] * field)

    def alm2map(almR, almI):
        rec = np.zeros(NPIX)
        for l in range(LMAX + 1):
            rec += almR[l, 0] * P[l, 0]
        for l in range(LMAX + 1):
            for m in range(1, l + 1):
                rec += 2.0 * (almR[l, m] * P[l, m] * cosm[m]
                              + almI[l, m] * P[l, m] * sinm[m])
        return rec * (NPIX / (4.0 * math.pi))

    almR = np.zeros((LMAX + 1, LMAX + 1))
    almI = np.zeros((LMAX + 1, LMAX + 1))
    map2alm(sky, almR, almI)
    for _ in range(2):
        residual = sky - alm2map(almR, almI)
        map2alm(residual, almR, almI)
    return almR, -almI                             # sign flip: Healpix convention


# --- antenna-response kernels XX(alpha,beta) and XY(alpha,beta) ----------------
def _trig_powers(p):
    """Return (c, s) with c[k] = cos(p)**k (k=1..8) and s = sin(p)."""
    c = np.empty(9)
    c[0] = 1.0
    cp = math.cos(p)
    for k in range(1, 9):
        c[k] = cp ** k
    return c, math.sin(p)


def kernel_XX(ca, sa, cb, sb):
    """Auto-spectrum response kernel (XX/YY/ZZ for beta = 0, 2pi/3, 4pi/3)."""
    R = np.zeros((LMAX + 1, LMAX + 1))
    I = np.zeros((LMAX + 1, LMAX + 1))

    R[0][0] = (12.0 * RTPI) / 5.0

    R[2][0] = -3.0 * sq5 * RTPI / 35.0
    R[2][1] = (9.0 * sq10 * RTPI * ca[1]) / 35.0
    I[2][1] = -(9.0 * sq10 * RTPI * sa) / 35.0
    R[2][2] = (9.0 * sq30 * RTPI * (2.0 * ca[2] - 1.0)) / 70.0
    I[2][2] = -(9.0 * sq30 * RTPI * (2.0 * ca[1] * sa)) / 70.0

    R[4][0] = -9.0 * RTPI * ((ca[2] - 0.5) * (cb[2] - 0.5) * ca[1] * cb[1] * sb * sa + (cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] + cb[4] / 8.0 - cb[2] / 8.0 + 11.0 / 630.0)

    R[4][1] = -1.2 * sq15 * RTPI * (sb * (cb[2] - 0.5) * (ca[4] - 1.5 * ca[2] + 0.25) * cb[1] * sa + ca[1] * ((cb[4] - cb[2] + 0.125) * ca[4] + (-0.25 - 2.0 * cb[4] + 2.0 * cb[2]) * ca[2] + 0.875 * cb[4] - 0.875 * cb[2] + 19.0 / 168.0))

    I[4][1] = -1.2 * sq15 * (((-cb[4] + cb[2] - 0.125) * ca[4] + 0.125 * cb[4] - 0.125 * cb[2] + 1.0 / 84.0) * sa + sb * ca[1] * (cb[2] - 0.5) * (ca[4] - 0.5 * ca[2] - 0.25) * cb[1]) * RTPI

    R[4][2] = -1.2 * (sb * ca[1] * (cb[2] - 0.5) * cb[1] * (ca[4] - ca[2] + 0.75) * sa + (ca[2] - 0.5) * ((cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] + 0.625 * cb[4] - 0.625 * cb[2] + 1.0 / 14.0)) * RTPI * sq10

    I[4][2] = -1.2 * RTPI * sq10 * (-((cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] - 0.375 * cb[4] + 0.375 * cb[2] - 3.0 / 56.0) * ca[1] * sa + (cb[2] - 0.5) * cb[1] * (ca[2] - 0.5) * (ca[4] - ca[2] - 0.5) * sb)

    R[4][3] = -8.0 / 35.0 * (cb[1] * (cb[2] - 0.5) * (ca[6] - 1.25 * ca[4] + 0.375 * ca[2] - 7.0 / 16.0) * sb * sa + (-1.0 / 32.0 + (cb[4] - cb[2] + 0.125) * ca[6] + 7.0 / 4.0 * (-cb[4] + cb[2] - 0.125) * ca[4] + 0.125 * (0.5 + 7.0 * cb[4] - 7.0 * cb[2]) * ca[2] - 17.0 / 32.0 * cb[4] + 17.0 / 32.0 * cb[2]) * ca[1]) * sq105 * RTPI

    I[4][3] = -8.0 / 35.0 * sq105 * RTPI * (((-ca[6] + 1.25 * ca[4] - 0.375 * ca[2] - 13.0 / 32.0) * cb[4] + (ca[6] - 1.25 * ca[4] + 0.375 * ca[2] + 13.0 / 32.0) * cb[2] - 0.125 * ca[6] + 5.0 / 32.0 * ca[4] - 1.0 / 16.0) * sa + cb[1] * (cb[2] - 0.5) * (ca[6] - 7.0 / 4.0 * ca[4] + 7.0 / 8.0 * ca[2] + 5.0 / 16.0) * sb * ca[1])

    R[4][4] = -4.0 / 35.0 * (cb[1] * (cb[2] - 0.5) * (ca[2] - 0.5) * (ca[4] - ca[2] + 0.125) * sb * ca[1] * sa + (ca[8] - 2.0 * ca[6] + 1.25 * ca[4] - 0.25 * ca[2] + 41.0 / 64.0) * cb[4] + (-ca[8] + 2.0 * ca[6] - 1.25 * ca[4] + 0.25 * ca[2] - 41.0 / 64.0) * cb[2] + 0.125 * ca[8] - 0.25 * ca[6] + 1.0 / 64.0 * ca[4] + 7.0 / 64.0 * ca[2] + 1.0 / 16.0) * RTPI * sq70

    I[4][4] = -4.0 / 35.0 * RTPI * sq70 * (-(ca[2] - 0.5) * ((cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] + 0.125 * cb[4] - 0.125 * cb[2] - 0.125) * ca[1] * sa + cb[1] * sb * (ca[8] - 2 * ca[6] + 1.25 * ca[4] - 0.25 * ca[2] - 0.625) * (cb[2] - 0.5))

    return R, I


def kernel_XY(ca, sa, cb, sb):
    """Cross-spectrum response kernel (XY/YZ/XZ for beta = 0, 2pi/3, 4pi/3)."""
    R = np.zeros((LMAX + 1, LMAX + 1))
    I = np.zeros((LMAX + 1, LMAX + 1))

    R[0][0] = -(6.0 * RTPI) / 5.0

    R[2][0] = 3.0 * sq5 * RTPI / 70.0
    R[2][1] = -(9.0 * sq10 * RTPI * ca[1]) / 70.0
    I[2][1] = (9.0 * sq10 * RTPI * sa) / 70.0
    R[2][2] = -(9.0 * sq30 * RTPI * (2.0 * ca[2] - 1.0)) / 140.0
    I[2][2] = (9.0 * sq30 * RTPI * (2.0 * ca[1] * sa)) / 140.0

    R[4][0] = 4.5 * RTPI * (((cb[3] - 0.5 * cb[1]) * sb + SQ3 * (cb[4] - cb[2] + 0.125)) * (ca[2] - 0.5) * ca[1] * sa - cb[1] * SQ3 * (ca[4] - ca[2] + 0.125) * (cb[2] - 0.5) * sb + (cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] + 11.0 / 630.0 - 0.125 * cb[2] + 0.125 * cb[4])

    R[4][1] = 0.6 * sq5 * (3.0 * (1.0 / 3.0 * SQ3 * cb[1] * (cb[2] - 0.5) * sb + cb[4] - cb[2] + 0.125) * (ca[4] - 1.5 * ca[2] + 0.25) * sa + (-3.0 * (ca[4] - 2.0 * ca[2] + 0.875) * cb[1] * (cb[2] - 0.5) * sb + SQ3 * ((cb[4] - cb[2] + 0.125) * ca[4] + (-2.0 * cb[4] + 2.0 * cb[2] - 0.25) * ca[2] + 19.0 / 168.0 + 0.875 * cb[4] - 0.875 * cb[2])) * ca[1]) * RTPI

    I[4][1] = 9.0 / 5.0 * sq5 * ((cb[1] * (ca[4] - 0.125) * (cb[2] - 0.5) * sb - 1.0 / 3.0 * SQ3 * ((cb[4] - cb[2] + 0.125) * ca[4] - 0.125 * cb[4] + 0.125 * cb[2] - 1.0 / 84.0)) * sa + ca[1] * (1.0 / 3.0 * SQ3 * cb[1] * (cb[2] - 0.5) * sb + cb[4] - cb[2] + 0.125) * (ca[4] - 0.5 * ca[2] - 0.25)) * RTPI

    R[4][2] = 0.6 * ((ca[4] - ca[2] + 0.75) * ca[1] * ((cb[3] - 0.5 * cb[1]) * sb + SQ3 * (cb[4] - cb[2] + 0.125)) * sa + (ca[2] - 0.5) * (-cb[1] * (cb[2] - 0.5) * SQ3 * (ca[4] - ca[2] + 0.625) * sb + (cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] + 0.625 * cb[4] - 0.625 * cb[2] + 1.0 / 14.0)) * sq10 * RTPI

    I[4][2] = 0.6 * (-ca[1] * (-cb[1] * (cb[2] - 0.5) * (ca[4] - ca[2] - 0.375) * SQ3 * sb + (cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] - 0.375 * cb[4] + 0.375 * cb[2] - 3.0 / 56.0) * sa + ((cb[3] - 0.5 * cb[1]) * sb + SQ3 * (cb[4] - cb[2] + 0.125)) * (ca[4] - ca[2] - 0.5) * (ca[2] - 0.5)) * sq10 * RTPI

    R[4][3] = 4.0 / 35.0 * sq35 * RTPI * (3.0 * (ca[6] - 1.25 * ca[4] + 0.375 * ca[2] - 7.0 / 16.0) * (1.0 / 3.0 * SQ3 * cb[1] * (cb[2] - 0.5) * sb + cb[4] - cb[2] + 0.125) * sa + (-3.0 * cb[1] * (cb[2] - 0.5) * (ca[6] - 7.0 / 4.0 * ca[4] + 0.875 * ca[2] - 17.0 / 32.0) * sb + SQ3 * (-1.0 / 32.0 + (cb[4] - cb[2] + 0.125) * ca[6] + 7.0 / 4.0 * (-cb[4] + cb[2] - 0.125) * ca[4] + 0.125 * (0.5 + 7.0 * cb[4] - 7.0 * cb[2]) * ca[2] - 17.0 / 32.0 * cb[4] + 17.0 / 32.0 * cb[2])) * ca[1])

    I[4][3] = 4.0 / 35.0 * ((((-ca[6] + 1.25 * ca[4] - 0.375 * ca[2] - 13.0 / 32.0) * cb[4] + (ca[6] - 1.25 * ca[4] + 0.375 * ca[2] + 13.0 / 32.0) * cb[2] - 0.125 * ca[6] + 5.0 / 32.0 * ca[4] - 1.0 / 16.0) * SQ3 + 3.0 * (cb[2] - 0.5) * (ca[6] - 1.25 * ca[4] + 0.375 * ca[2] + 13.0 / 32.0) * cb[1] * sb) * sa + (ca[6] - 7.0 / 4.0 * ca[4] + 0.875 * ca[2] + 5.0 / 16.0) * ca[1] * (0.5 * cb[1] * sb * (2.0 * cb[2] - 1.0) * SQ3 + 3.0 * cb[4] - 3.0 * cb[2] + 0.375)) * sq35 * RTPI

    R[4][4] = -2.0 / 35.0 * (-(SQ3 * (cb[4] - cb[2] + 0.125) + 0.5 * sb * cb[1] * (2.0 * cb[2] - 1.0)) * (ca[2] - 0.5) * ca[1] * (ca[4] - ca[2] + 0.125) * sa + SQ3 * (ca[8] - 2.0 * ca[6] + 1.25 * ca[4] - 0.25 * ca[2] + 41.0 / 64.0) * (cb[2] - 0.5) * cb[1] * sb + (-ca[8] + 2.0 * ca[6] - 1.25 * ca[4] + 0.25 * ca[2] - 41.0 / 64.0) * cb[4] + (ca[8] - 2.0 * ca[6] + 1.25 * ca[4] - 0.25 * ca[2] + 41.0 / 64.0) * cb[2] - 1.0 / 16.0 + 0.25 * ca[6] - 1.0 / 64.0 * ca[4] - 7.0 / 64.0 * ca[2] - 0.125 * ca[8]) * sq70 * RTPI

    I[4][4] = 2.0 / 35.0 * (-(ca[2] - 0.5) * (-cb[1] * SQ3 * (ca[4] - ca[2] + 0.125) * (cb[2] - 0.5) * sb + (cb[4] - cb[2] + 0.125) * ca[4] + (-cb[4] + cb[2] - 0.125) * ca[2] + 0.125 * cb[4] - 0.125 * cb[2] - 0.125) * ca[1] * sa + (ca[8] - 2 * ca[6] + 1.25 * ca[4] - 0.25 * ca[2] - 0.625) * ((cb[3] - 0.5 * cb[1]) * sb + SQ3 * (cb[4] - cb[2] + 0.125))) * sq70 * RTPI

    return R, I


def _contract(almR, almI, kR, kI):
    """Sum_lm w_m (almR*kR + almI*kI), with w_0 = 1, w_{m>=1} = 2."""
    c = almR * kR + almI * kI
    return c[:, 0].sum() + 2.0 * c[:, 1:].sum()


# --- top-level driver ----------------------------------------------------------
def compute_modulation(Tobs, t0=0.0, lambda_0=0.0, kappa_0=0.0, params=None):
    """Return (t, XX, YY, ZZ, XY, XZ, YZ) modulation curves over the orbit."""
    if params is None:
        params = GALAXY_PARAMS

    N = int((Tobs / YEAR) * 100.0)
    dt = Tobs / N
    dt = (Tobs + 2 * dt) / (N - 1)                  # pad time samples for splines
    t = t0 + dt * np.arange(N) - dt

    # sky map -> spherical harmonic coefficients (stationary in time)
    theta_ecl, phi_ecl, sky = build_sky_map(params)
    P = scaled_legendre(np.cos(theta_ecl))
    cosm = [np.cos(m * phi_ecl) for m in range(LMAX + 1)]
    sinm = [np.sin(m * phi_ecl) for m in range(LMAX + 1)]
    almR, almI = sphharm(sky, P, cosm, sinm)

    # constellation arm angles for the three channels
    cb = {}
    sb = {}
    for name, off in (("x", 0.0), ("y", 2.0 * math.pi / 3.0), ("z", 4.0 * math.pi / 3.0)):
        cb[name], sb[name] = _trig_powers(lambda_0 + off)

    out = {k: np.zeros(N) for k in ("XX", "YY", "ZZ", "XY", "XZ", "YZ")}
    for i in range(N):
        # orbital phase tied to the actual time sample (correct for any t0);
        # matches glass_lisa.c convention alpha = 2*pi*t/YEAR (+ kappa_0).
        alpha = 2.0 * math.pi * t[i] / YEAR + kappa_0
        ca, sa = _trig_powers(alpha)
        out["XX"][i] = _contract(almR, almI, *kernel_XX(ca, sa, cb["x"], sb["x"]))
        out["YY"][i] = _contract(almR, almI, *kernel_XX(ca, sa, cb["y"], sb["y"]))
        out["ZZ"][i] = _contract(almR, almI, *kernel_XX(ca, sa, cb["z"], sb["z"]))
        out["XY"][i] = _contract(almR, almI, *kernel_XY(ca, sa, cb["x"], sb["x"]))
        out["YZ"][i] = _contract(almR, almI, *kernel_XY(ca, sa, cb["y"], sb["y"]))
        out["XZ"][i] = _contract(almR, almI, *kernel_XY(ca, sa, cb["z"], sb["z"]))

    av = np.mean((out["XX"] + out["YY"] + out["ZZ"]) / 3.0)
    for k in out:
        out[k] /= av

    return t, out["XX"], out["YY"], out["ZZ"], out["XY"], out["XZ"], out["YZ"]


def main():
    ap = argparse.ArgumentParser(description="Galaxy foreground modulation (port of glass_galaxy.c)")
    # default duration reproduces the bundled modulation.dat (t0 = 0)
    ap.add_argument("--duration", type=float, default=62300160.0, help="observation time Tobs [s]")
    ap.add_argument("--t0", type=float, default=0.0, help="start time [s]")
    ap.add_argument("--lambda0", type=float, default=0.0, help="constellation phase lambda_0 [rad]")
    ap.add_argument("--kappa0", type=float, default=0.0, help="guiding-center phase kappa_0 [rad]")
    ap.add_argument("-o", "--out", default="modulation.dat", help="output file")
    args = ap.parse_args()

    t, XX, YY, ZZ, XY, XZ, YZ = compute_modulation(
        args.duration, t0=args.t0, lambda_0=args.lambda0, kappa_0=args.kappa0)

    data = np.column_stack([t, XX, YY, ZZ, XY, XZ, YZ])
    np.savetxt(args.out, data, fmt="%f %.10f %.10f %.10f %.10f %.10f %.10f")
    print(f"wrote {args.out} ({len(t)} samples)")


if __name__ == "__main__":
    main()
