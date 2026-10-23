"""Global-fit prior components (no re-exports; import the submodules directly).

- :mod:`~lisatools.globalfit.priors.galaxy_sky_dist`: the 3-D Milky Way joint prior over the GB
  ``(dist, alpha, sin_delta)`` slot (production, ``GB_USE_GALAXY_PRIOR=1``).
- :mod:`~lisatools.globalfit.priors.galaxy_prior_3d`: the galactocentric WD-binary density it
  wraps.
- :mod:`~lisatools.globalfit.priors.gbpriors`:
  :func:`~lisatools.globalfit.priors.gbpriors.get_fdot_mojito` (the GB ``fdot`` bounds) plus
  an unused GB prior class.
- :mod:`~lisatools.globalfit.priors.gfpriors` and :mod:`~lisatools.globalfit.priors.base`: generic
  prior infrastructure that no stock fit uses.
"""
