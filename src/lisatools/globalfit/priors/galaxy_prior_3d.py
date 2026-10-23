"""Minimal Eryn prior for Galactic position in (lambda, beta, distance).

Angles are ecliptic radians and distance is heliocentric kpc.  The underlying
Galactocentric density is the four-component Milky Way mixture -- thin disk,
thick disk, flattened Gaussian bulge, cut-off power-law halo -- fitted to the
WDWD catalog by ``plot_galaxy_density.py --plot fitmulti`` (that fitting
script is not part of this repository).  The best-fit values are copied in
below (:data:`DEFAULT_PARAMS`) so this module needs nothing but numpy and eryn.

Pipeline position: production uses :class:`GalaxyPrior3D` only through
:class:`~lisatools.globalfit.priors.galaxy_sky_dist.GalaxySkyDistPrior`, which
calls :meth:`GalaxyPrior3D.density` (in its ``pdf``) and
:meth:`GalaxyPrior3D.rvs` (in its ``rvs`` and ``norm``) for the GB
``(dist, alpha, sin_delta)`` prior (``GB_USE_GALAXY_PRIOR=1``). The ecliptic
:meth:`GalaxyPrior3D.pdf` is used by ``scripts/gb/check_galaxy_sky_dist.py``
(check 2); :meth:`GalaxyPrior3D.logpdf` completes the eryn prior surface.

Importing the module builds the module-level :data:`prior` container (no
random draws; the generator is created from OS entropy).
"""

import math

import numpy as np
from eryn.prior import ProbDistContainer


# Sun -> galactic center distance [kpc]; matches the Lamberts et al. 2019 /
# Toubiana et al. 2024 WDWD catalog.  The Sun sits at (-R_GC, 0, 0) with the
# galactic center at the origin.
R_GC = 8.178

# Maximum-likelihood fit to the 15.5M-source WDWD catalog (8 restarts,
# loglike = -3.408e6).  Copied from wdwd_bestfit_multicomponent.json; rerun
# ``plot_galaxy_density.py --plot fitmulti`` to regenerate, and pass any
# changed values as keyword arguments rather than editing them here.
DEFAULT_PARAMS = dict(
    f_thin=0.40626829484025956,    # mixture fractions (mass fraction per component)
    f_thick=0.36572698231454515,
    f_bulge=0.08236138405909049,
    f_halo=0.14564333878610472,
    Rd_thin=0.9417183320125694,    # thin disk scale length / height [kpc]
    hz_thin=0.6279602179436976,
    Rd_thick=2.7840957540387703,   # thick disk scale length / height [kpc]
    hz_thick=1.8870759429129158,
    Rb_bulge=0.6802352423305649,   # bulge scale radius [kpc]
    q_bulge=0.34628112691749574,   # bulge flattening (1 = spherical/GLASS)
    n_halo=1.8665556208868934,     # halo power-law index (rho ~ r^-n, n < 3)
    rc_halo=10.727165842582606,    # halo exponential cutoff radius [kpc]
)

_FRACTION_KEYS = ("f_thin", "f_thick", "f_bulge", "f_halo")
_POSITIVE_KEYS = ("Rd_thin", "hz_thin", "Rd_thick", "hz_thick",
                  "Rb_bulge", "q_bulge", "rc_halo")

# ecliptic unit vector -> galactic direction (rotate_ecliptogal, galaxy_modulation.py)
ECLIPTIC_TO_GALACTIC = np.array(
    [
        [-0.05487556043, -0.99382137890, -0.09647662818],
        [0.4941094278, -0.1109907351, 0.8622858751],
        [-0.8676661492, -0.00035159077, 0.4971471918],
    ]
)


def _sech2(u):
    """sech^2(u), written so large |u| underflows smoothly to 0 rather than
    overflowing cosh (which would poison the density with inf/0)."""
    e = np.exp(-2.0 * np.abs(u))
    return 4.0 * e / (1.0 + e) ** 2


def _disk_pdf(x, y, z, Rd, hz):
    """Normalized disk PDF: exp(-R/Rd) sech^2(z/hz) / (4 pi Rd^2 hz).

    sech^2 rather than exp(-|z|/hz) for the vertical profile: it is the
    self-gravitating isothermal-sheet solution, and it is smooth at z = 0
    instead of cusped.  The normalization is unchanged because
    int sech^2(z/hz) dz = 2 hz = int exp(-|z|/hz) dz."""
    R = np.hypot(x, y)
    return np.exp(-R / Rd) * _sech2(z / hz) / (4.0 * np.pi * Rd * Rd * hz)


def _gaussian_bulge_pdf(x, y, z, Rb, qb):
    """Normalized flattened-Gaussian bulge PDF on the spheroidal radius
    m = sqrt(R^2 + (z/qb)^2):   exp(-m^2/Rb^2) / (qb (sqrt(pi) Rb)^3).

    Gaussian rather than Hernquist because the WDWD catalog is *cored*, so a
    Hernquist (rho ~ r^-1) or McMillan-style cusp is the wrong inner shape.
    The flattening metric follows McMillan (2017); the normalization picks up
    exactly one factor of qb, since z = qb*zeta makes the integral spherical.
    At qb = 1 this reduces to the GLASS foreground integrator's bulge."""
    m2 = x * x + y * y + (z / qb) ** 2
    return np.exp(-m2 / (Rb * Rb)) / (qb * (np.sqrt(np.pi) * Rb) ** 3)


def _halo_pdf(x, y, z, n, rc, norm):
    """Normalized power-law halo PDF with an exponential cutoff, rho ~ r^-n e^(-r/rc).

    A hard truncation at rmax would make the mixture density (and so the
    likelihood) discontinuous and would floor the log-density at its value on
    the truncation sphere; this form is smooth and strictly positive."""
    r = np.sqrt(x * x + y * y + z * z)
    r = np.maximum(r, 1e-6)
    return r ** (-n) * np.exp(-r / rc) / norm


class GalaxyPrior3D:
    r"""Joint PDF in ecliptic ``(lambda, beta, distance)`` coordinates.

    Columns (indices 0, 1, 2):

    * 0 ``lambda`` -- ecliptic longitude [rad], in [0, 2*pi)
    * 1 ``beta``   -- ecliptic latitude  [rad], in [-pi/2, +pi/2]
    * 2 ``dist``   -- distance from the Sun [kpc], >= 0

    The galactocentric mixture is a proper PDF in kpc^-3 (each component is
    separately normalized, so the mixture integrates to 1 over all space),
    and the spherical volume element is the only Jacobian needed:

    .. math::

        p(\lambda, \beta, d) = \rho(x, y, z)\, d^2 \cos\beta,
        \qquad
        \rho = f_{\rm thin}\rho_{\rm disk}(R_{\rm thin}, h_{\rm thin})
        + f_{\rm thick}\rho_{\rm disk}(R_{\rm thick}, h_{\rm thick})
        + f_{\rm bulge}\rho_{\rm bulge} + f_{\rm halo}\rho_{\rm halo},

    with the Sun at galactocentric ``(-R_GC, 0, 0)`` (:data:`R_GC` kpc) and
    the ecliptic-to-galactic rotation :data:`ECLIPTIC_TO_GALACTIC`.

    Args:
        rng (int, numpy.random.Generator or None): Seed or generator for
            :meth:`rvs`, through ``numpy.random.default_rng``. ``None``
            (default) uses OS entropy.
        **params: Overrides of individual :data:`DEFAULT_PARAMS` entries
            (``f_thin``, ``f_thick``, ``f_bulge``, ``f_halo``, ``Rd_thin``,
            ``hz_thin``, ``Rd_thick``, ``hz_thick``, ``Rb_bulge``,
            ``q_bulge``, ``n_halo``, ``rc_halo``; lengths in kpc). The mixture
            fractions are renormalized to sum to 1.

    Raises:
        ValueError: An unknown parameter name, a negative fraction (or a
            non-positive fraction sum), a non-positive scale length, height,
            ``q_bulge`` or ``rc_halo``, or ``n_halo >= 3`` (the halo would
            not normalize).

    Attributes:
        params (dict): The resolved parameters.
        fractions (np.ndarray): ``(4,)`` normalized mixture fractions
            (thin, thick, bulge, halo).
        halo_norm (float): Halo normalization
            ``4 pi rc^(3 - n) Gamma(3 - n)``.
        rng (numpy.random.Generator): Generator of :meth:`rvs`.
        use_cupy (bool): Written by the eryn container; not read here
            (:meth:`pdf` / :meth:`logpdf` / :meth:`rvs` always return NumPy).
        return_gpu (bool): Written by the eryn container; not read here.

    Example:
        >>> from lisatools.globalfit.priors.galaxy_prior_3d import GalaxyPrior3D
        >>> galaxy = GalaxyPrior3D(rng=0)
        >>> x = galaxy.rvs(4)
        >>> x.shape, galaxy.logpdf(x).shape
        ((4, 3), (4,))
    """

    def __init__(self, rng=None, **params):
        unknown = sorted(set(params) - set(DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"unknown parameter(s): {unknown}")
        p = dict(DEFAULT_PARAMS, **params)

        fractions = np.array([p[k] for k in _FRACTION_KEYS], dtype=float)
        if np.any(fractions < 0.0) or fractions.sum() <= 0.0:
            raise ValueError("mixture fractions must be non-negative, with a positive sum")
        if any(p[k] <= 0.0 for k in _POSITIVE_KEYS):
            raise ValueError(f"require positive {', '.join(_POSITIVE_KEYS)}")
        if p["n_halo"] >= 3.0:
            raise ValueError("n_halo must be < 3 for the halo profile to normalize")

        self.params = p
        self.fractions = fractions / fractions.sum()
        # int_0^inf r^-n e^(-r/rc) 4 pi r^2 dr = 4 pi rc^(3-n) Gamma(3-n), n < 3
        self.halo_norm = (
            4.0 * np.pi * p["rc_halo"] ** (3.0 - p["n_halo"])
            * math.gamma(3.0 - p["n_halo"])
        )
        self.rng = np.random.default_rng(rng)

        # Eryn's ProbDistContainer writes these attributes on member priors.
        self.use_cupy = False
        self.return_gpu = False

    def density(self, x, y, z):
        """Galactocentric mixture density [kpc^-3]; integrates to 1 over all space.

        Args:
            x (np.ndarray): Galactocentric x [kpc] (the Sun is at
                ``x = -R_GC``); any shape broadcastable with ``y`` and ``z``.
            y (np.ndarray): Galactocentric y [kpc].
            z (np.ndarray): Galactocentric z [kpc] (height above the plane).

        Returns:
            np.ndarray: Density at each point, broadcast shape of the inputs.
        """
        p = self.params
        f_thin, f_thick, f_bulge, f_halo = self.fractions
        return (
            f_thin * _disk_pdf(x, y, z, p["Rd_thin"], p["hz_thin"])
            + f_thick * _disk_pdf(x, y, z, p["Rd_thick"], p["hz_thick"])
            + f_bulge * _gaussian_bulge_pdf(x, y, z, p["Rb_bulge"], p["q_bulge"])
            + f_halo * _halo_pdf(x, y, z, p["n_halo"], p["rc_halo"], self.halo_norm)
        )

    def pdf(self, coordinates):
        """Density ``rho * d^2 * cos(beta)`` in ecliptic ``(lambda, beta, dist)``.

        Args:
            coordinates (array-like): ``(N, 3)`` (or one ``(3,)`` row) of
                ``(lambda [rad], beta [rad], dist [kpc])``; NumPy only.

        Returns:
            np.ndarray: ``(N,)`` density in ``kpc^-1 rad^-2``; zero outside
            ``[0, 2 pi) x [-pi/2, pi/2] x [0, inf)``.

        Raises:
            ValueError: ``coordinates`` does not have 3 columns.
        """
        values = np.atleast_2d(np.asarray(coordinates, dtype=float))
        if values.shape[1] != 3:
            raise ValueError("coordinates must have shape (N, 3)")

        lam, beta, distance = values.T
        valid = (
            (lam >= 0.0)
            & (lam < 2.0 * np.pi)
            & (beta >= -0.5 * np.pi)
            & (beta <= 0.5 * np.pi)
            & (distance >= 0.0)
        )

        cb = np.cos(beta)
        ecliptic = np.vstack(
            [cb * np.cos(lam), cb * np.sin(lam), np.sin(beta)]
        )
        galactic = ECLIPTIC_TO_GALACTIC @ ecliptic
        x = distance * galactic[0] - R_GC
        y = distance * galactic[1]
        z = distance * galactic[2]

        result = self.density(x, y, z) * distance**2 * cb
        return np.where(valid, result, 0.0)

    def logpdf(self, coordinates):
        """Log of :meth:`pdf` (``-inf`` outside the support).

        Args:
            coordinates (array-like): ``(N, 3)`` ecliptic
                ``(lambda, beta, dist)``; NumPy only.

        Returns:
            np.ndarray: ``(N,)`` log-density.
        """
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.log(self.pdf(coordinates))

    def rvs(self, size=1):
        """Draw exact samples by sampling each component in proportion to its fraction.

        Disks: R ~ Gamma(2, Rd) (i.e. exp(-R/Rd) R dR), phi uniform, and z by
        inverse-CDF of the sech^2 profile -- CDF(z) = (1 + tanh(z/hz))/2, so
        z = hz artanh(2U - 1).  Bulge: isotropic Gaussian of sigma = Rb/sqrt(2)
        with z squashed by qb.  Halo: r ~ Gamma(3-n, rc) with isotropic angles.
        Component counts are multinomial in :attr:`fractions`; the rows are
        shuffled, shifted to heliocentric and rotated to the ecliptic frame.
        All draws use :attr:`rng`.

        Args:
            size (int or tuple[int, ...]): Leading output shape. A Python
                ``int`` or a ``tuple`` only (a NumPy integer is rejected).
                Default ``1``.

        Returns:
            np.ndarray: ``size + (3,)`` rows of
            ``(lambda [rad], beta [rad], dist [kpc])``.

        Raises:
            ValueError: ``size`` is neither an ``int`` nor a ``tuple``.
        """
        if isinstance(size, int):
            size = (size,)
        elif not isinstance(size, tuple):
            raise ValueError("size must be an int or tuple of ints")

        p = self.params
        count = int(np.prod(size))
        counts = self.rng.multinomial(count, self.fractions)
        xyz = np.empty((count, 3))
        start = 0

        for n_disk, Rd, hz in [
            (counts[0], p["Rd_thin"], p["hz_thin"]),
            (counts[1], p["Rd_thick"], p["hz_thick"]),
        ]:
            radius = self.rng.gamma(shape=2.0, scale=Rd, size=n_disk)
            azimuth = self.rng.uniform(0.0, 2.0 * np.pi, size=n_disk)
            # sech^2(z/hz)/(2 hz) has this inverse CDF; clip keeps artanh finite
            u = np.clip(self.rng.uniform(-1.0, 1.0, size=n_disk), -1 + 1e-15, 1 - 1e-15)
            xyz[start:start + n_disk] = np.column_stack(
                [radius * np.cos(azimuth), radius * np.sin(azimuth), hz * np.arctanh(u)]
            )
            start += n_disk

        n_bulge = counts[2]
        sigma = p["Rb_bulge"] / np.sqrt(2.0)
        bulge = self.rng.normal(scale=sigma, size=(n_bulge, 3))
        bulge[:, 2] *= p["q_bulge"]
        xyz[start:start + n_bulge] = bulge
        start += n_bulge

        n_halo = counts[3]
        # p(r) dr ~ r^(2-n) e^(-r/rc) dr, i.e. Gamma(3-n, rc)
        radius = self.rng.gamma(
            shape=3.0 - p["n_halo"], scale=p["rc_halo"], size=n_halo
        )
        cos_theta = self.rng.uniform(-1.0, 1.0, size=n_halo)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        azimuth = self.rng.uniform(0.0, 2.0 * np.pi, size=n_halo)
        xyz[start:start + n_halo] = np.column_stack(
            [
                radius * sin_theta * np.cos(azimuth),
                radius * sin_theta * np.sin(azimuth),
                radius * cos_theta,
            ]
        )

        # components are laid down contiguously; shuffle so that reshaping into
        # a walker grid (or slicing the first k rows) doesn't get one component
        self.rng.shuffle(xyz, axis=0)

        heliocentric_galactic = xyz + np.array([R_GC, 0.0, 0.0])
        # the rotation is orthogonal, so right-multiplying applies its inverse
        ecliptic = heliocentric_galactic @ ECLIPTIC_TO_GALACTIC
        distance = np.linalg.norm(ecliptic, axis=1)
        lam = np.arctan2(ecliptic[:, 1], ecliptic[:, 0]) % (2.0 * np.pi)
        beta = np.arcsin(np.clip(ecliptic[:, 2] / distance, -1.0, 1.0))
        return np.column_stack([lam, beta, distance]).reshape(size + (3,))


# TODO check: keep/remove/document -- only caller: the module-level prior below (never imported)
def build_prior(**kwargs):
    """Return the Eryn container over parameter indices lambda=0, beta=1, d=2."""
    return ProbDistContainer({(0, 1, 2): GalaxyPrior3D(**kwargs)})


# Module-level ecliptic container, built at import (no draws, OS-entropy
# generator). Nothing imports it.
prior = build_prior()
