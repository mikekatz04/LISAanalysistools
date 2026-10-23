r"""Ridge-Gibbs fiber resampling for the GB ``(Mc, ratio, distance)`` degeneracy.

The production 9-column GB sampling basis is ``[dist (kpc), f0 (mHz), Mc (Msun),
phi0, cos_iota, psi, alpha, sin_delta, fdot_astro_ratio (= r)]`` (ICRS sky). The
physical observables depend on ``Mc`` only through ``u = Mc^{5/3}``:

.. math::

    \dot f = \dot f_{\rm GR}(f_0, M_c)\,(1 + r), \qquad
    \dot f_{\rm GR} \propto M_c^{5/3} f_0^{11/3}, \qquad
    A \propto M_c^{5/3} / d \quad (\text{at fixed } f_0),

so at fixed data-constrained ``(f0, fdot, A)`` there is an EXACT 1-D likelihood
degeneracy (a fiber):

.. math::

    \left\{ M_c \text{ free},\; r(M_c) = \frac{K_f}{M_c^{5/3}} - 1,\;
    d(M_c) = \frac{M_c^{5/3}}{K_A} \right\},
    \qquad K_f = u\,(1 + r), \quad K_A = u / d > 0.

A whitened-Fisher analysis at 20 mHz showed this direction exactly flat and
effectively frozen under linear info-matrix proposals; because the likelihood
is exactly invariant along the fiber, a Gibbs-style resampling needs zero
likelihood evaluations.

Reparameterizing ``(Mc, r, d) -> (u, Kf, KA)`` gives
``|det d(Mc, r, d)/d(u, Kf, KA)| = (3/5) u^{-2/5} / KA^2`` (only the
``u``-dependence matters), so with ``u' ~ Uniform[u_lo, u_hi]`` on the
invariant-determined interval the per-leaf acceptance is

.. math::

    \log\alpha = \left[\log p(x') - \log p(x)\right]
        + \tfrac{2}{5}\left[\ln u - \ln u'\right],

implemented generically by :class:`eryn.moves.RidgeGibbsMove` with the measure
term supplied here as ``log_fiber_measure(u) = -(2/5) ln u``.

Entry points: :class:`McRatioDistFiber` (the fiber map),
:class:`GFRidgeGibbsMove` (the eryn move plus the global-fit sub-state write-back)
and :func:`make_gb_ridge_gibbs_move` (the factory the recipe calls).

Pipeline position: :func:`lisatools.globalfit.recipe.build_gb_moves` registers the
GB move as ``gb_ridge_gibbs`` and :func:`lisatools.globalfit.recipe.build_vgb_moves`
registers the VGB twin as ``vgb_ridge_gibbs``; both are gated by
:func:`lisatools.globalfit.recipe.ridge_gibbs_eligible` (env ``GB_RIDGE_GIBBS``,
default ``"1"``, set to 1 by the 9mo/12mo launchers) and take
``leaf_fraction`` from ``GB_RIDGE_GIBBS_LEAF_FRACTION`` (default ``1.0``, unset
in the launchers). The 9mo/12mo recipes run both moves in ``gb_search_1``,
``gb_search_2`` and ``full_pe``, head-only, on the cold chain of the engine
state. This module reads no env knob itself.
"""

from __future__ import annotations

import numpy as np

from eryn.moves import RidgeGibbsMove

__all__ = ["McRatioDistFiber", "GFRidgeGibbsMove", "make_gb_ridge_gibbs_move"]

# tiny positive floor on u = Mc^{5/3} (Mc, d must be > 0; a run box floor
# like Mc >= 1e-3, i.e. u >= 1e-5, sits far above this)
_TINY_U = 1e-300


class McRatioDistFiber:
    """Fiber map for the GB ``(Mc, fdot_astro_ratio, dist)`` exact degeneracy.

    Duck-typed ``fiber_map`` for :class:`eryn.moves.RidgeGibbsMove` on the
    9-column GB sampling basis (and on the 6-column VGB chirp-mass basis, which
    carries the same three columns). The fiber coordinate is ``t = u = Mc^{5/3}``;
    the per-row invariants are ``Kf = u (1 + r)`` and ``KA = u / d``. All
    other columns are untouched, so the physical ``(A, f0, fdot)`` -- and
    therefore the waveform and likelihood -- are exactly invariant along the
    fiber by construction. ``f0`` is never read.

    Column indices are derived from the transform container's
    ``input_basis`` by NAME (never hard-coded); the 8-column bases without
    ``Mc``/``dist``/``fdot_astro_ratio`` are not eligible and raise loudly.
    The eryn move calls :meth:`to_fiber`, :meth:`fiber_bounds`,
    :meth:`from_fiber` and :meth:`log_fiber_measure` once per propose.

    Args:
        transform_container: The run's GB :class:`eryn.utils.TransformContainer`
            (e.g. from
            :func:`~lisatools.globalfit.stock.erebor.transforms.make_gb_transform_container`
            with ``use_chirp_mass=True, use_fdot_astro=True, use_distance=True``);
            only its ``input_basis`` (list of column names) is consulted.
        mc_lims (sequence of float): ``(mc_lo, mc_hi)`` chirp-mass prior box (Msun),
            ``0 < mc_lo < mc_hi``.
        dist_lims (sequence of float): ``(d_lo, d_hi)`` distance prior box (kpc),
            ``0 < d_lo < d_hi``.
        ratio_max (float): ``M`` of the ``r ~ U[-M, M]`` ratio prior box, ``M > 0``.

    Raises:
        ValueError: The basis is missing a required column or the limits are
            not valid.

    Attributes:
        input_basis (list[str]): The container's sampling-basis column names.
        dist_index (int): Column of ``dist``.
        mc_index (int): Column of ``Mc``.
        ratio_index (int): Column of ``fdot_astro_ratio``.
        mc_lims (tuple[float, float]): Chirp-mass box (Msun).
        dist_lims (tuple[float, float]): Distance box (kpc).
        ratio_max (float): Ratio half-width ``M``.

    Example:
        >>> from types import SimpleNamespace
        >>> import numpy as np
        >>> from lisatools.sampling.ridge_fiber import McRatioDistFiber
        >>> tc = SimpleNamespace(input_basis=["dist", "f0", "Mc", "phi0", "cos_iota",
        ...                                   "psi", "alpha", "sin_delta",
        ...                                   "fdot_astro_ratio"])
        >>> fiber = McRatioDistFiber(tc, mc_lims=(0.001, 1.0), dist_lims=(0.1, 50.0),
        ...                          ratio_max=5.0)
        >>> rows = np.array([[8.0, 3.0, 0.3, 1.0, 0.2, 0.5, 1.0, 0.1, 0.4]])
        >>> u, inv = fiber.to_fiber(rows)
        >>> lo, hi = fiber.fiber_bounds(inv)
        >>> new = fiber.from_fiber(0.5 * (lo + hi), inv, rows)
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
        """Map sampling rows to the fiber coordinate and the per-row invariants.

        Rows with non-positive or non-finite ``Mc``/``d`` (or a non-finite ``r``)
        get NaN for ``u`` and both invariants, which propagate to NaN bounds and
        are skipped by the move.

        Args:
            coords (np.ndarray): ``(n, ndim)`` rows in the sampling basis (host;
                cast to float64).

        Returns:
            tuple[np.ndarray, dict[str, np.ndarray]]: ``u = Mc^{5/3}`` of shape
            ``(n,)`` and ``{"Kf": u (1 + r), "KA": u / d}``, each ``(n,)``.
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

        ``u_lo`` is finally floored at ``1e-300`` so ``Mc`` and ``d`` stay
        strictly positive. An empty intersection is signaled by
        ``u_hi <= u_lo`` (the move skips the leaf); NaN invariants give NaN
        bounds (also skipped).

        Args:
            invariants (dict[str, np.ndarray]): ``{"Kf": (n,), "KA": (n,)}`` as
                returned by :meth:`to_fiber`.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(u_lo, u_hi)``, each ``(n,)`` float64.
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
        """New rows at fiber coordinate ``u_new``; untouched columns carry through.

        Sets ``Mc = u_new^{3/5}``, ``r = Kf / u_new - 1`` and ``d = u_new / KA``
        on a copy of ``coords``.

        Args:
            u_new (np.ndarray): ``(n,)`` new fiber coordinates.
            invariants (dict[str, np.ndarray]): ``{"Kf", "KA"}`` per row, from
                :meth:`to_fiber`.
            coords (np.ndarray): ``(n, ndim)`` current rows (not modified).

        Returns:
            np.ndarray: ``(n, ndim)`` float64 rows on the same fiber.
        """
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
        skews the Mc marginal -- see the prior-invariance test
        (``tests/test_ridge_fiber.py``).

        Args:
            u (np.ndarray): ``(n,)`` fiber coordinates.
            invariants (dict[str, np.ndarray]): Unused (the measure has no
                invariant dependence that survives the difference).

        Returns:
            np.ndarray: ``(n,)`` log measure ``-0.4 * ln(u)``.
        """
        return -0.4 * np.log(np.asarray(u, dtype=np.float64))


class GFRidgeGibbsMove(RidgeGibbsMove):
    """:class:`RidgeGibbsMove` + global-fit sub-state cold-row write-back.

    In the global fit the module's tempered ensemble lives on a
    ``ModuleSubState`` (``state.sub_states[branch]``) and the main state
    carries only the engine's cold chain; the band moves sample the
    sub-state and mirror row 0 into the main state (``sync_cold_row``).
    This move samples the MAIN state directly, so after a propose the
    updated cold row must be mirrored the OTHER way (``pull_cold_row``) —
    without it the next global-fit move's ``_check_substate_consistency``
    MPI-aborts the run (the 2026-08-20 v5/v6 gb_search it=1 failure), and
    the accepted fiber jumps would be silently overwritten by the next
    band move's sync anyway. The fiber leaves the waveform — ``(A, f0,
    fdot)`` and all extrinsics — exactly invariant, so no residual or
    likelihood bookkeeping moves: coords/inds only. Outside the global fit
    (no ``sub_states`` on the state) it degrades to the plain eryn move.

    Built by :func:`make_gb_ridge_gibbs_move`; the constructor arguments are
    those of :class:`eryn.moves.RidgeGibbsMove` (``branch_name``,
    ``fiber_map``, ``per_leaf_log_prior``, ``leaf_fraction``, ``**kwargs``).
    In the 9mo/12mo runs it runs head-only on the full engine state as
    ``gb_ridge_gibbs`` / ``vgb_ridge_gibbs``.
    """

    def propose(self, model, state):
        """Run the eryn fiber resampling, then pull the cold row into the sub-state.

        Args:
            model (eryn.model.Model): Sampler model (only ``model.random`` is
                used by the parent move).
            state (eryn.state.State): Current state; in the global fit a
                :class:`~lisatools.globalfit.state.GFState` with ``sub_states``.

        Returns:
            tuple: ``(state, accepted)`` from
            :meth:`eryn.moves.RidgeGibbsMove.propose`; ``accepted`` is the
            ``(ntemps, nwalkers)`` bool array. When the branch has an
            initialized sub-state, ``sub.pull_cold_row(state, branch_name)``
            has been applied to it.
        """
        state, accepted = super().propose(model, state)
        sub = (getattr(state, "sub_states", None) or {}).get(self.branch_name)
        if sub is not None and getattr(sub, "tempered_initialized", False):
            sub.pull_cold_row(state, self.branch_name)
        return state, accepted


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
    for ``(n, ndim)`` input, which is exactly the per-leaf density the sampler's
    prior uses (the summed-delta ``log_prior`` update in the move relies on
    this). The prior is used as is, so a non-uniform distance prior (the 3-D
    Milky Way sky/distance prior under ``GB_USE_GALAXY_PRIOR=1``) enters the
    acceptance correctly; the boxes below only bound the fiber interval.

    Called by :func:`lisatools.globalfit.recipe.build_gb_moves` (``gb_ridge_gibbs``)
    and :func:`lisatools.globalfit.recipe.build_vgb_moves` (``vgb_ridge_gibbs``,
    ``branch_name="vgb"``), which pass ``leaf_fraction`` from
    ``GB_RIDGE_GIBBS_LEAF_FRACTION`` (default ``1.0``).

    Args:
        priors_container: The run's GB :class:`eryn.priors.ProbDistContainer`.
        transform_container: The run's GB transform container (supplies the
            ``input_basis`` for column resolution).
        mc_lims (sequence of float): ``(mc_lo, mc_hi)`` chirp-mass prior box (Msun).
        dist_lims (sequence of float): ``(d_lo, d_hi)`` distance prior box (kpc).
        ratio_max (float): ``M`` of the ``r ~ U[-M, M]`` ratio prior box.
        leaf_fraction (float, optional): Probability that each alive leaf is
            resampled in a call, in ``(0, 1]``. Default ``1.0``.
        branch_name (str, optional): Branch the move operates on. Default ``"gb"``.
        **move_kwargs: Forwarded to :class:`eryn.moves.RidgeGibbsMove`
            (and through it to :class:`eryn.moves.Move`).

    Returns:
        :class:`GFRidgeGibbsMove`: The configured move (sub-state-aware).

    Raises:
        ValueError: From :class:`McRatioDistFiber` (ineligible basis or bad
            limits) or from :class:`eryn.moves.RidgeGibbsMove` (``leaf_fraction``
            outside ``(0, 1]``).

    Example:
        .. code-block:: python

            from lisatools.sampling.ridge_fiber import make_gb_ridge_gibbs_move

            move = make_gb_ridge_gibbs_move(
                priors["gb"], gb_info.transform,
                mc_lims=gb_info.m_chirp_lims,
                dist_lims=gb_info.dist_lims,
                ratio_max=float(gb_info.fdot_astro_ratio_max),
                leaf_fraction=1.0,
            )
            move.name = "gb_ridge_gibbs"
    """
    fiber_map = McRatioDistFiber(transform_container, mc_lims, dist_lims, ratio_max)
    return GFRidgeGibbsMove(
        branch_name=branch_name,
        fiber_map=fiber_map,
        per_leaf_log_prior=priors_container.logpdf,
        leaf_fraction=leaf_fraction,
        **move_kwargs,
    )
