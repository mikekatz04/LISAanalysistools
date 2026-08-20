"""Ridge-Gibbs fiber resampling for the GB ``(Mc, ratio, distance)`` degeneracy.

The production 9-column GB sampling basis is ``[dist, f0 (mHz), Mc, phi0,
cos_iota, psi, alpha, sin_delta, fdot_astro_ratio (= r)]``. The physical
observables are ``fdot = fdot_gr(f0, Mc) * (1 + r)`` with ``fdot_gr propto
Mc^{5/3} f0^{11/3}`` and ``A propto Mc^{5/3} / d`` -- both depend on Mc only
through ``u = Mc^{5/3}``, so at fixed data-constrained ``(f0, fdot, A)``
there is an EXACT 1-D likelihood degeneracy (a fiber):

    { Mc variable;  r(Mc) = Kf / Mc^{5/3} - 1;  d(Mc) = Mc^{5/3} / KA }

with invariants ``Kf = Mc^{5/3} (1 + r)`` and ``KA = Mc^{5/3} / d``
(``KA > 0`` always). A whitened-Fisher analysis at 20 mHz showed this
direction exactly flat and effectively frozen under linear info-matrix
proposals; because the likelihood is exactly invariant along the fiber, a
Gibbs-style resampling needs zero likelihood evaluations.

Reparameterizing ``(Mc, r, d) -> (u, Kf, KA)`` gives
``|det d(Mc, r, d)/d(u, Kf, KA)| propto u^{-2/5}`` (only the u-dependence
matters), so with ``u' ~ Uniform[u_lo, u_hi]`` on the invariant-determined
interval the acceptance is

    log alpha = [logprior(x') - logprior(x)] + (2/5) [ln u - ln u'],

implemented generically by :class:`eryn.moves.RidgeGibbsMove` with the
measure term supplied here as ``log_fiber_measure(u) = -(2/5) ln u``.

This module only makes the pieces importable and tested -- wiring into the
production recipe is a separate reviewed step.
"""

from __future__ import annotations

import numpy as np

from eryn.moves import RidgeGibbsMove

__all__ = ["McRatioDistFiber", "make_gb_ridge_gibbs_move"]

# tiny positive floor on u = Mc^{5/3} (Mc, d must be > 0; a run box floor
# like Mc >= 1e-3 sits ~10 orders above this)
_TINY_U = 1e-300


class McRatioDistFiber:
    """Fiber map for the GB ``(Mc, fdot_astro_ratio, dist)`` exact degeneracy.

    Duck-typed ``fiber_map`` for :class:`eryn.moves.RidgeGibbsMove` on the
    9-column GB sampling basis. The fiber coordinate is ``t = u = Mc^{5/3}``;
    the per-row invariants are ``Kf = u (1 + r)`` and ``KA = u / d``. All
    other columns are untouched, so the physical ``(A, f0, fdot)`` -- and
    therefore the waveform and likelihood -- are exactly invariant along the
    fiber by construction.

    Column indices are derived from the transform container's
    ``input_basis`` (never hard-coded); the 8-column bases without
    ``Mc``/``dist``/``fdot_astro_ratio`` are not eligible and raise loudly.

    Args:
        transform_container: The run's GB :class:`eryn.utils.TransformContainer`
            (e.g. from ``make_gb_transform_container(use_chirp_mass=True,
            use_fdot_astro=True, use_distance=True)``); only its
            ``input_basis`` is consulted.
        mc_lims: ``(mc_lo, mc_hi)`` chirp-mass prior box (Msun).
        dist_lims: ``(d_lo, d_hi)`` distance prior box (kpc).
        ratio_max: ``M`` of the ``r ~ U[-M, M]`` ratio prior box.

    Raises:
        ValueError: The basis is missing a required column or the limits are
            not valid.
    """

    _REQUIRED = ("dist", "Mc", "fdot_astro_ratio")

    def __init__(self, transform_container, mc_lims, dist_lims, ratio_max):
        input_basis = list(getattr(transform_container, "input_basis", []) or [])
        missing = [name for name in self._REQUIRED if name not in input_basis]
        if missing:
            raise ValueError(
                "McRatioDistFiber requires the (dist, Mc, fdot_astro_ratio) "
                f"GB sampling basis; input_basis {input_basis} is missing "
                f"{missing}. 8-column (A / fdot) bases are not eligible for "
                "the ridge-Gibbs fiber move."
            )
        self.input_basis = input_basis
        self.dist_index = input_basis.index("dist")
        self.mc_index = input_basis.index("Mc")
        self.ratio_index = input_basis.index("fdot_astro_ratio")

        mc_lims = tuple(float(v) for v in mc_lims)
        dist_lims = tuple(float(v) for v in dist_lims)
        ratio_max = float(ratio_max)
        if not (0.0 < mc_lims[0] < mc_lims[1]):
            raise ValueError(f"mc_lims must satisfy 0 < lo < hi; got {mc_lims}.")
        if not (0.0 < dist_lims[0] < dist_lims[1]):
            raise ValueError(f"dist_lims must satisfy 0 < lo < hi; got {dist_lims}.")
        if not ratio_max > 0.0:
            raise ValueError(f"ratio_max must be > 0; got {ratio_max}.")
        self.mc_lims = mc_lims
        self.dist_lims = dist_lims
        self.ratio_max = ratio_max

        # Mc box in u = Mc^{5/3}
        self._u_mc_lo = mc_lims[0] ** (5.0 / 3.0)
        self._u_mc_hi = mc_lims[1] ** (5.0 / 3.0)

    def to_fiber(self, coords):
        """``(n, ndim)`` rows -> ``(u, {"Kf", "KA"})``.

        Rows with non-positive or non-finite ``Mc``/``d`` get NaN invariants,
        which propagate to NaN bounds and are skipped by the move.
        """
        rows = np.asarray(coords, dtype=np.float64)
        mc = rows[:, self.mc_index]
        ratio = rows[:, self.ratio_index]
        dist = rows[:, self.dist_index]

        with np.errstate(invalid="ignore", divide="ignore"):
            u = mc ** (5.0 / 3.0)
            k_fdot = u * (1.0 + ratio)
            k_amp = u / dist

        bad = ~(np.isfinite(mc) & (mc > 0.0) & np.isfinite(dist) & (dist > 0.0))
        bad |= ~np.isfinite(ratio)
        if np.any(bad):
            u = np.where(bad, np.nan, u)
            k_fdot = np.where(bad, np.nan, k_fdot)
            k_amp = np.where(bad, np.nan, k_amp)

        return u, {"Kf": k_fdot, "KA": k_amp}

    def fiber_bounds(self, invariants):
        """Intersection of the Mc, distance, and ratio boxes in ``u``.

        Constraints (all functions of the INVARIANTS only, so the interval
        is identical forward/reverse):

        * Mc box: ``u in [mc_lo^{5/3}, mc_hi^{5/3}]``
        * distance box (``d' = u / KA``, ``KA > 0``): ``u in [KA d_lo, KA d_hi]``
        * ratio box (``r' = Kf / u - 1 in [-M, M]``):
            * ``Kf > 0``: ``u >= Kf / (1 + M)`` (from ``r' <= M``); and only
              if ``M < 1``, ``u <= Kf / (1 - M)`` (from ``r' >= -M``).
            * ``Kf < 0``: requires ``M > 1`` (``r' < -1`` always); then
              ``r' >= -M`` gives ``u >= Kf / (1 - M) = |Kf| / (M - 1)`` --
              a LOWER bound (``Kf/u`` increases toward 0 with growing
              ``u``). Empty if ``M <= 1``.
            * ``Kf == 0``: ``r' = -1`` always; empty iff ``M < 1``.

        An empty intersection is signaled by ``u_hi <= u_lo`` (the move skips
        the leaf); NaN invariants give NaN bounds (also skipped).
        """
        k_fdot = np.asarray(invariants["Kf"], dtype=np.float64)
        k_amp = np.asarray(invariants["KA"], dtype=np.float64)
        big_m = self.ratio_max

        with np.errstate(invalid="ignore"):
            u_lo = np.maximum(self._u_mc_lo, k_amp * self.dist_lims[0])
            u_hi = np.minimum(self._u_mc_hi, k_amp * self.dist_lims[1])

            pos = k_fdot > 0.0
            u_lo = np.where(pos, np.maximum(u_lo, k_fdot / (1.0 + big_m)), u_lo)
            if big_m < 1.0:
                u_hi = np.where(pos, np.minimum(u_hi, k_fdot / (1.0 - big_m)), u_hi)

            neg = k_fdot < 0.0
            if big_m > 1.0:
                u_lo = np.where(neg, np.maximum(u_lo, k_fdot / (1.0 - big_m)), u_lo)
            else:
                u_hi = np.where(neg, -np.inf, u_hi)

            if big_m < 1.0:
                u_hi = np.where(k_fdot == 0.0, -np.inf, u_hi)

            # numerical guard: u (hence Mc, d) strictly positive
            u_lo = np.maximum(u_lo, _TINY_U)

        # NaN invariants -> NaN bounds (move-side skip)
        nan_rows = ~(np.isfinite(k_fdot) & np.isfinite(k_amp))
        if np.any(nan_rows):
            u_lo = np.where(nan_rows, np.nan, u_lo)
            u_hi = np.where(nan_rows, np.nan, u_hi)

        return u_lo, u_hi

    def from_fiber(self, u_new, invariants, coords):
        """New rows at fiber coordinate ``u_new``; untouched columns carry through."""
        rows = np.array(np.asarray(coords, dtype=np.float64), copy=True)
        u_new = np.asarray(u_new, dtype=np.float64)
        k_fdot = np.asarray(invariants["Kf"], dtype=np.float64)
        k_amp = np.asarray(invariants["KA"], dtype=np.float64)

        rows[:, self.mc_index] = u_new ** (3.0 / 5.0)
        rows[:, self.ratio_index] = k_fdot / u_new - 1.0
        rows[:, self.dist_index] = u_new / k_amp
        return rows

    def log_fiber_measure(self, u, invariants):
        """``-(2/5) ln u``: the u-dependence of ``|det d(Mc,r,d)/d(u,Kf,KA)|``.

        Only the ``t``-dependence matters (constants and pure-invariant
        factors cancel in the acceptance difference). Omitting this term
        skews the Mc marginal -- see the prior-invariance test.
        """
        return -0.4 * np.log(np.asarray(u, dtype=np.float64))


def make_gb_ridge_gibbs_move(
    priors_container,
    transform_container,
    mc_lims,
    dist_lims,
    ratio_max,
    leaf_fraction=1.0,
    branch_name="gb",
    **move_kwargs,
):
    """Build the configured GB ridge-Gibbs move.

    ``per_leaf_log_prior`` is ``priors_container.logpdf`` -- the run's
    :class:`eryn.priors.ProbDistContainer` returns one log-density per row
    for ``(n, 9)`` input, which is exactly the per-leaf density the sampler's
    prior uses (the summed-delta ``log_prior`` update in the move relies on
    this).

    Args:
        priors_container: The run's GB :class:`eryn.priors.ProbDistContainer`.
        transform_container: The run's GB transform container (supplies the
            ``input_basis`` for column resolution).
        mc_lims: ``(mc_lo, mc_hi)`` chirp-mass prior box (Msun).
        dist_lims: ``(d_lo, d_hi)`` distance prior box (kpc).
        ratio_max: ``M`` of the ``r ~ U[-M, M]`` ratio prior box.
        leaf_fraction: Random fraction of alive leaves resampled per call.
        branch_name: Branch the move operates on.
        **move_kwargs: Forwarded to :class:`eryn.moves.RidgeGibbsMove`
            (and through it to :class:`eryn.moves.Move`).

    Returns:
        :class:`eryn.moves.RidgeGibbsMove`: The configured move.
    """
    fiber_map = McRatioDistFiber(transform_container, mc_lims, dist_lims, ratio_max)
    return RidgeGibbsMove(
        branch_name=branch_name,
        fiber_map=fiber_map,
        per_leaf_log_prior=priors_container.logpdf,
        leaf_fraction=leaf_fraction,
        **move_kwargs,
    )
