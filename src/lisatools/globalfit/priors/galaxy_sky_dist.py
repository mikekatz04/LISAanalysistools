"""3-D Galaxy prior in the GB sampling basis ``(dist, alpha, sin_delta)``.

:class:`~lisatools.globalfit.priors.galaxy_prior_3d.GalaxyPrior3D` is a proper
joint PDF over ECLIPTIC ``(lambda, beta, distance)``. The GB branch samples
ICRS ``(dist, alpha, sin_delta)`` -- a different ORDER, a different FRAME, and
a different latitude VARIABLE -- so it cannot be dropped into
``GBSettings.sky_dist_distribution`` directly. This module is the adapter.

Pipeline position: :meth:`~lisatools.globalfit.stock.erebor.gb.GBSetup.init_sampling_info`
calls :func:`build_gb_galaxy_sky_dist` when ``GBSettings.use_galaxy_prior`` is on (env
``GB_USE_GALAXY_PRIOR``, default ``True``, launchers 1) and the GB basis is the distance basis
(chirp mass + astrophysical ``(f0, Mc)`` prior, the 9-column production basis
``[dist, f0, Mc, phi0, cos_iota, psi, alpha, sin_delta, fdot_astro_ratio]``). The container is
the ``("dist", "alpha", "sin_delta")`` tuple-key member of the GB prior, remapped to basis
columns 0, 6 and 7; its ``logpdf`` runs on every GB prior evaluation and its ``rvs`` on every
prior-RJ birth.

The Jacobian is the part worth stating explicitly, because it looks like it
should carry a ``cos`` and does not:

``GalaxyPrior3D.pdf`` is a density with respect to ``dlambda dbeta ddist``,
so it carries ``cos(beta)`` to turn the solid angle into a latitude
coordinate::

    p(lambda, beta, d) = rho(x, y, z) * d^2 * cos(beta)

The GB basis samples ``sin_delta``, and ``dalpha * d(sin delta)`` IS the
solid-angle element ``dOmega``. A frame rotation preserves ``dOmega``, so::

    p(d, alpha, sin_delta) = rho(x, y, z) * d^2          # no cos factor

i.e. the ``cos`` cancels rather than composing. Verified numerically by
``scripts/gb/check_galaxy_sky_dist.py`` (check 2: both forms integrate to the
same mass; it also checks the normalization, ``rvs`` against ``pdf`` and the
``logpdf`` support). There is no unit test of this module.

Frames: ICRS -> ecliptic is a fixed rotation about the x-axis by the J2000
mean obliquity (:data:`OBLIQUITY_J2000`), composed here with
:data:`~lisatools.globalfit.priors.galaxy_prior_3d.ECLIPTIC_TO_GALACTIC` into
a single ICRS -> galactic matrix (:data:`ICRS_TO_GALACTIC`). ``astropy.SkyCoord``
would be far too slow inside a ``logpdf`` that runs every proposal; the check
script compares :data:`ICRS_TO_ECLIPTIC` with astropy's
``BarycentricMeanEcliptic`` (check 1, < 1 arcsec).
"""

from __future__ import annotations

import numpy as np

from .galaxy_prior_3d import ECLIPTIC_TO_GALACTIC, R_GC, GalaxyPrior3D

__all__ = ["GalaxySkyDistPrior", "ICRS_TO_ECLIPTIC", "ICRS_TO_GALACTIC",
           "build_gb_galaxy_sky_dist"]

# J2000 mean obliquity of the ecliptic [rad] (IAU 2006).
OBLIQUITY_J2000 = np.deg2rad(23.439279444444445)

# ICRS -> ecliptic: rotation about the x-axis (vernal equinox) by +epsilon.
_ce, _se = np.cos(OBLIQUITY_J2000), np.sin(OBLIQUITY_J2000)
ICRS_TO_ECLIPTIC = np.array([
    [1.0, 0.0, 0.0],
    [0.0, _ce, _se],
    [0.0, -_se, _ce],
])

# One matrix straight from ICRS unit vectors to galactocentric axes.
ICRS_TO_GALACTIC = ECLIPTIC_TO_GALACTIC @ ICRS_TO_ECLIPTIC


def _to_module(a, use_cupy):
    """Host result -> the module eryn's container expects.

    The container stamps ``use_cupy`` onto member priors and accumulates
    ``prior_vals`` on that module; a component returning bare numpy into a
    cupy accumulation is a TypeError (hit the first time the galaxy prior
    ran on GPU, 2026-08-13). Compute stays on host -- only the boundary
    converts."""
    if use_cupy:
        import cupy

        return cupy.asarray(a)
    return a


def _to_host(a):
    """cupy -> numpy without importing cupy."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


class GalaxySkyDistPrior:
    r"""Galaxy prior over the GB basis slot ``("dist", "alpha", "sin_delta")``.

    Columns, in the order the GB tuple-key slot supplies them:

    * 0 ``dist``      -- heliocentric distance [kpc], >= 0
    * 1 ``alpha``     -- ICRS right ascension [rad], in [0, 2*pi)
    * 2 ``sin_delta`` -- sine of ICRS declination, in [-1, 1]

    The density is the galactocentric mixture of :class:`GalaxyPrior3D`
    times the radial volume factor, renormalized to the distance box:

    .. math::

        p(d, \alpha, \sin\delta) = \frac{\rho(x, y, z)\, d^2}{m},
        \qquad m = \int_{d_{\rm lo}}^{d_{\rm hi}} \int_{4\pi} \rho\, d^2
        \,{\rm d}\Omega\, {\rm d}d,

    with ``(x, y, z)`` the galactocentric position (kpc) of the point and
    ``m`` = :attr:`norm`. It is a density in ``kpc^-1 rad^-1`` over the box
    ``dist_lims x [0, 2 pi) x [-1, 1]`` and integrates to 1 there. The sky
    support is always the FULL sky: the GB ``alpha_lims`` / ``delta_lims``
    are not applied by this class.

    The class follows the eryn component contract of
    :class:`~eryn.priors.ProbDistContainer`: :meth:`logpdf` takes ``(N, 3)``
    and returns ``(N,)``; :meth:`rvs` takes ``size`` and returns
    ``size + (3,)``. The container writes :attr:`use_cupy` /
    :attr:`return_gpu` onto it; computation always runs on the host and
    only the returned arrays are converted to CuPy when ``use_cupy`` is set.

    Args:
        dist_lims: ``(lo, hi)`` distance bounds [kpc]. Samples outside get
            zero probability, and :meth:`rvs` rejection-samples into the
            range, so the prior stays normalized over the SAMPLED box rather
            than over all space. Default ``(0.0, 100.0)``; the stock GB
            branch passes ``GBSettings.dist_lims`` (default
            ``[0.001, 40.0]``).
        galaxy: an existing :class:`GalaxyPrior3D`, or ``None`` to build one.
        rng: Seed or ``numpy.random.Generator``, stored as :attr:`rng`
            through ``numpy.random.default_rng``. No method of this class
            draws from it: :meth:`rvs` (whose rejection step is
            deterministic) and :attr:`norm` draw only from the galaxy's own
            generator, so ``rng`` does not change any result.
        galaxy_rng: seed / Generator for the :class:`GalaxyPrior3D` this
            builds when ``galaxy`` is ``None``. This is the generator behind
            every random draw of :meth:`rvs` and :attr:`norm`. ``None``
            (default) leaves it on OS entropy. Ignored when an already-built
            ``galaxy`` is passed in.
        **params: forwarded to :class:`GalaxyPrior3D` (mixture overrides).

    Raises:
        ValueError: ``dist_lims`` does not satisfy ``0 <= lo < hi``, or
            ``**params`` holds a name :class:`GalaxyPrior3D` rejects.

    Attributes:
        galaxy (GalaxyPrior3D): The wrapped galactocentric density.
        dist_lims (tuple[float, float]): Distance box [kpc].
        rng (numpy.random.Generator): Stored generator (unused, see ``rng``).
        use_cupy (bool): Written by the eryn container; ``True`` makes
            :meth:`pdf` / :meth:`logpdf` / :meth:`rvs` return CuPy arrays.
        return_gpu (bool): Written by the eryn container; not read here.

    Example:
        >>> from lisatools.globalfit.priors.galaxy_sky_dist import GalaxySkyDistPrior
        >>> prior = GalaxySkyDistPrior(dist_lims=(0.001, 40.0), galaxy_rng=2)
        >>> x = prior.rvs(5)
        >>> x.shape
        (5, 3)
        >>> prior.logpdf(x).shape
        (5,)
    """

    #: Monte-Carlo draws used to estimate :attr:`norm` (see its docstring).
    _norm_samples = 400_000

    def __init__(self, dist_lims=(0.0, 100.0), galaxy=None, rng=None,
                 galaxy_rng=None, **params):
        self.galaxy = (galaxy if galaxy is not None
                       else GalaxyPrior3D(rng=galaxy_rng, **params))
        self.dist_lims = (float(dist_lims[0]), float(dist_lims[1]))
        if not self.dist_lims[1] > self.dist_lims[0] >= 0.0:
            raise ValueError(f"dist_lims must be 0 <= lo < hi; got {dist_lims}")
        self.rng = np.random.default_rng(rng)
        # Eryn's ProbDistContainer writes these onto member priors.
        self.use_cupy = False
        self.return_gpu = False
        self._norm = None  # lazy: mass inside dist_lims

    # -- geometry ---------------------------------------------------------

    def _galactocentric(self, alpha, sin_delta, dist):
        """ICRS (alpha, sin_delta) + distance -> galactocentric (x, y, z)."""
        cd = np.sqrt(np.clip(1.0 - sin_delta**2, 0.0, 1.0))  # cos(delta)
        icrs = np.stack([cd * np.cos(alpha), cd * np.sin(alpha), sin_delta])
        gal = np.tensordot(ICRS_TO_GALACTIC, icrs, axes=(1, 0))
        return dist * gal[0] - R_GC, dist * gal[1], dist * gal[2]

    # -- normalization ----------------------------------------------------

    @property
    def norm(self) -> float:
        """Prior mass inside ``dist_lims`` over the full sky.

        ``GalaxyPrior3D`` is normalized over ALL space; restricting distance
        makes it a sub-probability, so divide by this to keep ``pdf`` a
        density on the sampled box.

        Estimated by Monte Carlo from :meth:`GalaxyPrior3D.rvs`, which samples
        the mixture EXACTLY -- the mass inside the range is then just the
        fraction of draws (``_norm_samples`` = 400 000) that land there. A
        quadrature grid is a poor tool here: the integrand is a thin great
        circle (the galactic plane) times a sharply peaked radial profile,
        and a uniform grid needs to be enormous before it stops biasing the
        answer by ~1%.

        Computed lazily on the first :meth:`pdf` / :meth:`logpdf` call and
        cached. The draws come from the galaxy's generator, so the value is
        reproducible only when that generator is seeded (``galaxy_rng``, as
        the stock GB build does); with OS entropy it carries Monte-Carlo
        noise between instances. Computing it also advances that generator,
        which shifts the later :meth:`rvs` stream.

        Returns:
            float: Mass fraction in ``(0, 1]``.

        Raises:
            RuntimeError: No draw landed inside ``dist_lims``.
        """
        if self._norm is None:
            n = int(self._norm_samples)
            d = np.atleast_2d(
                self.galaxy.rvs(size=n)
            )[:, 2]
            lo, hi = self.dist_lims
            self._norm = float(((d >= lo) & (d <= hi)).mean())
            if self._norm <= 0.0:
                raise RuntimeError(
                    f"dist_lims={self.dist_lims} contains no Galaxy prior "
                    "mass; nothing would ever be drawn there."
                )
        return self._norm

    # -- Eryn prior surface ----------------------------------------------

    def pdf(self, coordinates):
        """``p(dist, alpha, sin_delta)`` -- see the module docstring.

        Args:
            coordinates (array-like): ``(N, 3)`` (or one ``(3,)`` row) of
                ``(dist [kpc], alpha [rad], sin_delta)``; NumPy or CuPy.

        Returns:
            xp.ndarray: ``(N,)`` density in ``kpc^-1 rad^-1``, zero outside
            ``dist_lims x [0, 2 pi) x [-1, 1]``; CuPy when :attr:`use_cupy`
            is set, else NumPy.

        Raises:
            ValueError: ``coordinates`` does not have 3 columns.
        """
        return _to_module(self._pdf_host(coordinates), self.use_cupy)

    def _pdf_host(self, coordinates):
        """Host-side pdf (always numpy in/out; cupy input accepted)."""
        v = np.atleast_2d(_to_host(coordinates)).astype(float)
        if v.shape[1] != 3:
            raise ValueError(
                f"coordinates must be (N, 3) as (dist, alpha, sin_delta); "
                f"got {v.shape}"
            )
        dist, alpha, sin_delta = v.T
        ok = (
            (dist >= self.dist_lims[0]) & (dist <= self.dist_lims[1])
            & (alpha >= 0.0) & (alpha < 2.0 * np.pi)
            & (sin_delta >= -1.0) & (sin_delta <= 1.0)
        )
        x, y, z = self._galactocentric(alpha, sin_delta, dist)
        # NO cos(delta): d(alpha) d(sin delta) is already the solid angle.
        out = self.galaxy.density(x, y, z) * dist**2 / self.norm
        return np.where(ok, out, 0.0)

    def logpdf(self, coordinates):
        """Log of :meth:`pdf`; the call the eryn container makes per GB prior evaluation.

        Args:
            coordinates (array-like): ``(N, 3)`` (or ``(3,)``) of
                ``(dist [kpc], alpha [rad], sin_delta)``; NumPy or CuPy.

        Returns:
            xp.ndarray: ``(N,)`` log-density; ``-inf`` outside the box. CuPy
            when :attr:`use_cupy` is set, else NumPy.

        Raises:
            ValueError: ``coordinates`` does not have 3 columns.
        """
        with np.errstate(divide="ignore"):
            out = np.log(self._pdf_host(coordinates))
        return _to_module(out, self.use_cupy)

    def rvs(self, size=1):
        """Draw ``(dist, alpha, sin_delta)`` inside ``dist_lims``.

        Delegates to :meth:`GalaxyPrior3D.rvs` (which samples the mixture
        exactly, in batches of ``max(n, 128)``) and rejects draws outside the
        distance range, so the result is distributed as :meth:`pdf`. The
        ecliptic draws are rotated to ICRS (``alpha`` wrapped to
        ``[0, 2 pi)``, ``sin_delta`` clipped to ``[-1, 1]``). Used for the GB
        prior-RJ births.

        Args:
            size (int or tuple[int, ...]): Leading output shape. Default ``1``.

        Returns:
            xp.ndarray: ``size + (3,)`` samples; CuPy when :attr:`use_cupy`
            is set, else NumPy.

        Raises:
            RuntimeError: A whole batch landed outside ``dist_lims``.
        """
        shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        n = int(np.prod(shape))
        lo, hi = self.dist_lims
        keep = np.empty((0, 3))
        while keep.shape[0] < n:
            draw = np.atleast_2d(self.galaxy.rvs(size=max(n, 128)))
            lam, beta, dist = draw.T
            m = (dist >= lo) & (dist <= hi)
            if not m.any():
                raise RuntimeError(
                    f"no GalaxyPrior3D draws landed inside dist_lims="
                    f"{self.dist_lims}; the range excludes essentially all "
                    "of the Galaxy."
                )
            # ecliptic (lam, beta) -> ICRS (alpha, sin_delta)
            cb = np.cos(beta[m])
            ecl = np.stack([cb * np.cos(lam[m]), cb * np.sin(lam[m]),
                            np.sin(beta[m])])
            icrs = np.tensordot(ICRS_TO_ECLIPTIC.T, ecl, axes=(1, 0))
            alpha = np.mod(np.arctan2(icrs[1], icrs[0]), 2.0 * np.pi)
            sin_delta = np.clip(icrs[2], -1.0, 1.0)
            keep = np.vstack([keep, np.column_stack(
                [dist[m], alpha, sin_delta])])
        return _to_module(keep[:n].reshape(shape + (3,)), self.use_cupy)


def build_gb_galaxy_sky_dist(dist_lims, **params):
    """``ProbDistContainer`` for ``GBSettings.sky_dist_distribution``.

    The GB tuple-key slot ``("dist", "alpha", "sin_delta")`` hands its three
    columns to member index 0/1/2, so the joint occupies ``(0, 1, 2)``.
    Called by :meth:`~lisatools.globalfit.stock.erebor.gb.GBSetup.init_sampling_info`
    when ``GB_USE_GALAXY_PRIOR`` is on and no explicit
    ``sky_dist_distribution`` is set.

    ``**params`` reaches :class:`GalaxySkyDistPrior`. The stock GB build
    passes ``rng=`` and ``galaxy_rng=`` from the rank's build sub-seeds;
    only ``galaxy_rng`` changes the draws (see :class:`GalaxySkyDistPrior`),
    and ``None`` keeps it on OS entropy.

    Args:
        dist_lims (tuple[float, float]): Distance box [kpc].
        **params: Forwarded to :class:`GalaxySkyDistPrior` (``galaxy``,
            ``rng``, ``galaxy_rng``, mixture overrides).

    Returns:
        eryn.priors.ProbDistContainer: Integer-keyed 3-column container.

    Example:
        >>> from lisatools.globalfit.priors.galaxy_sky_dist import build_gb_galaxy_sky_dist
        >>> container = build_gb_galaxy_sky_dist((0.001, 40.0), galaxy_rng=3)
        >>> container.ndim
        3
        >>> container.rvs(size=4).shape
        (4, 3)
    """
    from eryn.prior import ProbDistContainer

    return ProbDistContainer(
        {(0, 1, 2): GalaxySkyDistPrior(dist_lims=dist_lims, **params)}
    )
