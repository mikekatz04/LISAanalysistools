# -*- coding: utf-8 -*-
"""Sky-mode hopping Metropolis-Hastings move for LISA samplers.

The discrete sky-mode symmetries of the LISA response are symmetries of the
**constellation (detector) frame**: the reflection through the constellation
plane and the 90-degree rotations about its normal. A sampler whose basis
carries the sky in another frame (SSB ecliptic, or the production ICRS
``alpha / sin_delta / psi`` basis) gets the hop by converting the active rows
to the LISA frame, hopping there, and converting back -- see
:class:`SkyMove`. The frame conversions are the bbhx
:func:`~bbhx.utils.transform.SSB_to_LISA` / :func:`~bbhx.utils.transform.LISA_to_SSB`
pair (Marsat & Baker, arXiv:2003.00357), verified 2026-10 against the
LISA Analysis Tools time-domain response on the stock equal-arm and ESA
orbits: every one of the 8 hopped modes overlaps the original TDI A/E at
> 0.99 with no residual time shift at 3 mHz (the latitudinal pair stays at
1.000 up to 10 mHz, the longitudinal ones fall to ~0.93 there as expected
once ``f L / c`` is no longer small).

The transforms assume a circular 1 AU orbit with the constellation at
ecliptic longitude ``alpha = Omega t + phi0``. For a realistic orbit file the
phase ``alpha(t)`` drifts from ``Omega t + const`` by the orbit's equation of
centre (~0.3 deg over a year for the ESA orbits; the data-start offset alone
is tens of degrees), so :class:`ConstellationPhase` reads ``alpha(t)`` off the
run's own orbits and the move pins each row's frame to the phase at that
row's time.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from eryn.moves import MHMove

from ...utils.constants import EPS_RAD, YRSID_SI

__all__ = ["SkyMove", "ConstellationPhase"]


# ----------------------------------------------------------------------------
# constellation orbital phase from an Orbits object
# ----------------------------------------------------------------------------
class ConstellationPhase:
    """Unwrapped ecliptic longitude ``alpha(t)`` of the constellation centre.

    A (time, alpha) table with linear interpolation inside the grid and linear
    extrapolation (end slopes) outside it -- the MBH ``t_plunge`` prior runs
    past the data end, where the configured orbit grid stops. Picklable (two
    numpy arrays and a float).

    Args:
        t_grid: Absolute times (orbit-file time axis), increasing.
        alpha: Unwrapped constellation longitude at ``t_grid`` (radians).
        t_offset: Added to the SAMPLING-basis time before the lookup, i.e.
            the epoch the sampled time is referenced to (the MBH branch's
            ``mbh_waveform_t0``: ``t_abs = t_plunge + mbh_waveform_t0``).
    """

    def __init__(self, t_grid, alpha, t_offset: float = 0.0):
        t_grid = np.asarray(t_grid, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        if t_grid.ndim != 1 or t_grid.shape != alpha.shape or t_grid.size < 2:
            raise ValueError("t_grid and alpha must be 1D arrays of the same length >= 2.")
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing.")
        self.t_grid = t_grid
        self.alpha = np.unwrap(alpha)
        self.t_offset = float(t_offset)

    @classmethod
    def from_orbits(
        cls,
        orbits,
        t_offset: float = 0.0,
        step: float = 86400.0,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> "ConstellationPhase":
        """Build the table from a configured :class:`~lisatools.detector.Orbits`.

        The constellation centre is the mean of the three spacecraft
        positions; ICRS-frame orbits (``orbits.frame == "icrs"``, the
        production L1 files) are rotated to the ecliptic with the fixed
        obliquity ``EPS_RAD`` -- the same analytic frame as
        :func:`lisatools.sources.utils.icrs_to_ecliptic`, so the sky and the
        orbit phase live in ONE ecliptic convention.

        Args:
            orbits: Orbits object (CPU or GPU backend; positions are pulled
                to the host).
            t_offset: See the class docstring.
            step: Grid spacing in seconds (alpha is linear to ~1e-8 rad
                over a day).
            t_min, t_max: Grid bounds (absolute). Default: the orbits'
                configured spacecraft time grid.
        """
        sc_t = getattr(orbits, "sc_t", None)
        if sc_t is None:
            sc_t = orbits.t
        sc_t = np.asarray(_to_host(sc_t), dtype=float)
        lo = float(sc_t[0]) if t_min is None else float(t_min)
        hi = float(sc_t[-1]) if t_max is None else float(t_max)
        if hi <= lo:
            raise ValueError(f"empty orbit time span [{lo}, {hi}].")
        n = max(int(np.ceil((hi - lo) / float(step))) + 1, 2)
        t_grid = np.linspace(lo, hi, n)
        xp = orbits.xp
        t_in = xp.asarray(t_grid)
        pos = np.stack(
            [np.atleast_2d(_to_host(orbits.get_pos(t_in, sc=i))) for i in (1, 2, 3)], axis=1
        )  # (n, 3 sc, 3 xyz)
        centre = pos.mean(axis=1)
        if getattr(orbits, "frame", "ecliptic") == "icrs":
            centre = _icrs_to_ecliptic_cartesian(centre)
        alpha = np.arctan2(centre[:, 1], centre[:, 0])
        return cls(t_grid, alpha, t_offset=t_offset)

    def __call__(self, t_sampling):
        """``alpha`` (radians, unwrapped) at sampling-basis time(s) ``t_sampling``."""
        t = np.asarray(t_sampling, dtype=float) + self.t_offset
        out = np.interp(t, self.t_grid, self.alpha)
        # linear extrapolation with the end slopes
        lo = t < self.t_grid[0]
        if np.any(lo):
            s = (self.alpha[1] - self.alpha[0]) / (self.t_grid[1] - self.t_grid[0])
            out = np.where(lo, self.alpha[0] + s * (t - self.t_grid[0]), out)
        hi = t > self.t_grid[-1]
        if np.any(hi):
            s = (self.alpha[-1] - self.alpha[-2]) / (self.t_grid[-1] - self.t_grid[-2])
            out = np.where(hi, self.alpha[-1] + s * (t - self.t_grid[-1]), out)
        return out


def _to_host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def _icrs_to_ecliptic_cartesian(xyz: np.ndarray) -> np.ndarray:
    """Rotate equatorial (ICRS) cartesian vectors to the ecliptic frame at the
    fixed obliquity ``EPS_RAD`` (the ``sources.utils`` sky convention:
    ``x`` unchanged, ``y_ecl = cos(eps) y + sin(eps) z``,
    ``z_ecl = -sin(eps) y + cos(eps) z``)."""
    ce, se = np.cos(EPS_RAD), np.sin(EPS_RAD)
    out = np.empty_like(xyz)
    out[..., 0] = xyz[..., 0]
    out[..., 1] = ce * xyz[..., 1] + se * xyz[..., 2]
    out[..., 2] = -se * xyz[..., 1] + ce * xyz[..., 2]
    return out


# ----------------------------------------------------------------------------
# the move
# ----------------------------------------------------------------------------
class SkyMove(MHMove):
    """Metropolis move that jumps between equivalent LISA sky modes.

    The LISA response is approximately invariant under several discrete
    transformations of the sky-position / inclination / polarization
    parameters. This move proposes such a swap as the MCMC step, allowing the
    sampler to mix between otherwise isolated sky modes.

    The discrete symmetries are symmetries of the **LISA constellation
    frame**. When the sampling basis carries the sky position in another
    frame, set ``coord_frame`` accordingly and the move wraps the
    transformation in the appropriate frame conversions:

    - ``"lisa"`` (default): coordinates are already LISA-frame; the
      transforms are applied directly (legacy behavior).
    - ``"ssb_ecliptic"``: coordinates are SSB ecliptic
      ``(lam, sinbeta, psi, t_ref)``. The move converts SSB -> LISA with
      :func:`bbhx.utils.transform.SSB_to_LISA`, applies the mode hop, and
      converts back with :func:`bbhx.utils.transform.LISA_to_SSB`.
    - ``"icrs"``: coordinates are equatorial ``(ra, sin(dec), psi_icrs,
      t_ref)`` -- the stock MBH basis (``alpha``, ``sin_delta``, ``psi``,
      ``t_plunge``). The sky AND the polarization are rotated ICRS <->
      ecliptic together with the analytic fixed-obliquity pair
      :func:`lisatools.sources.utils.icrs_to_ecliptic` /
      :func:`~lisatools.sources.utils.ecliptic_to_icrs` on each side of
      the SSB <-> LISA conversion.

    For the non-LISA frames the merger/reference time participates in the
    frame conversion (the LISA-frame time is the retarded time at the
    constellation, so a hop that keeps the DETECTOR arrival time fixed moves
    the SSB time by up to ~2 R/c ~ 1000 s), so ``ind_map`` must contain a
    ``"t_ref"`` entry.

    **Detailed balance.** In the LISA frame the hops are measure-preserving
    involutions / uniform shifts, so the proposal is symmetric there. In the
    sampling frame the move is the conjugate ``F^-1 o S o F`` and its
    Metropolis factor is ``ln|det J_F(x)| - ln|det J_F(x')|`` (the frame map
    mixes the time with the sky, ``|det J| = 1 + O(Omega R / c) ~ 1 +-
    1e-4``). ``jacobian=True`` (default) computes it by central differences
    on the four frame columns; ``False`` neglects it (the legacy behaviour).

    Args:
        ind_map: Mapping from coordinate names to their indices in the
            coordinate array. Defaults to
            ``{"cosinc": 6, "lam": 7, "sinbeta": 8, "psi": 9, "t_ref": 10}``
            (the legacy LISA-frame MBH basis). The stock ICRS MBH basis is
            ``{"cosinc": 6, "psi": 7, "lam": 8, "sinbeta": 9, "t_ref": 10}``.
            ``"t_ref"`` is only consumed when ``coord_frame != "lisa"``.
        which: Which transformation to apply. Must be one of ``"both"``,
            ``"lat"``, or ``"long"``.
        coord_frame: Frame of the sampled sky coordinates; one of
            ``"lisa"``, ``"ssb_ecliptic"``, ``"icrs"``. (default:
            ``"lisa"``)
        frame_t0: Constant constellation-phase offset in **years**
            (``alpha = Omega (t_ref + frame_t0 YRSID_SI)``), used when no
            ``constellation_phase`` is given. (default: ``0.0``)
        constellation_phase: Callable ``alpha(t_ref) -> radians`` giving the
            constellation's ecliptic longitude at each row's sampling-basis
            time -- normally a :class:`ConstellationPhase` built from the
            run's orbits. Overrides ``frame_t0``. (default: ``None``)
        jacobian: Include the frame-conversion Jacobian in the proposal
            factors (see above). (default: ``True``)
        **kwargs: Additional keyword arguments forwarded to
            :class:`eryn.moves.MHMove`.

    Raises:
        ValueError: If ``ind_map`` is not a ``dict``, ``which`` is not one
            of the allowed strings, ``coord_frame`` is not one of the
            allowed strings, or ``coord_frame != "lisa"`` while ``ind_map``
            lacks a ``"t_ref"`` entry.

    """

    _ALLOWED_FRAMES = ("lisa", "ssb_ecliptic", "icrs")
    _FRAME_COLUMNS = ("lam", "sinbeta", "psi", "t_ref")
    # central-difference steps per frame column (rad, -, rad, s) and the
    # wrap period of the corresponding OUTPUT column (None: not periodic)
    _FD_STEPS = (1e-6, 1e-7, 1e-6, 1.0)
    _FD_PERIODS = (2 * np.pi, None, np.pi, None)

    def __init__(
        self,
        ind_map=None,
        which="both",
        coord_frame="lisa",
        frame_t0=0.0,
        constellation_phase: Optional[Callable] = None,
        jacobian: bool = True,
        **kwargs,
    ):

        if ind_map is None:
            ind_map = dict(cosinc=6, lam=7, sinbeta=8, psi=9, t_ref=10)

        elif isinstance(ind_map, dict) is False:
            raise ValueError("If providing the ind_map kwarg, it must be a dict.")

        if which not in ["both", "lat", "long"]:
            raise ValueError("which kwarg must be 'both', 'lat', or 'long'.")

        if coord_frame not in self._ALLOWED_FRAMES:
            raise ValueError(
                f"coord_frame must be one of {self._ALLOWED_FRAMES}, got {coord_frame!r}."
            )

        if coord_frame != "lisa" and "t_ref" not in ind_map:
            raise ValueError(
                "ind_map must contain a 't_ref' entry when coord_frame != 'lisa' "
                "(the SSB <-> LISA conversion mixes the reference time with the "
                "sky position)."
            )

        if constellation_phase is not None and not callable(constellation_phase):
            raise ValueError("constellation_phase must be callable (alpha(t_ref) -> radians).")

        self.ind_map = ind_map
        self.which = which
        self.coord_frame = coord_frame
        self.frame_t0 = float(frame_t0)
        self.constellation_phase = constellation_phase
        self.jacobian = bool(jacobian)
        self.transform = getattr(self, f"{which}_transform")
        super(SkyMove, self).__init__(**kwargs)

    # ------------------------------------------------------------------
    # Frame conversion machinery (only used when coord_frame != "lisa")
    # ------------------------------------------------------------------
    def _frame_t0_years(self, t_ref):
        """Per-row ``t0`` (years) for the bbhx transforms such that
        ``Omega * (t_ref + t0 * YRSID_SI) == alpha(t_ref)``."""
        from bbhx.utils.constants import ConstOmega

        if self.constellation_phase is None:
            return self.frame_t0
        alpha = np.asarray(self.constellation_phase(t_ref), dtype=float)
        return (alpha - ConstOmega * np.asarray(t_ref, dtype=float)) / ConstOmega / YRSID_SI

    def _to_lisa_frame(self, coords):
        """Convert ``(lam, sinbeta, psi, t_ref)`` columns to the LISA frame.

        No-op when ``coord_frame == "lisa"``. Operates on a copy of the 2D
        ``(n, ndim)`` coordinate array.
        """
        if self.coord_frame == "lisa":
            return coords

        from bbhx.utils.transform import SSB_to_LISA

        temp = coords.copy()
        lam = temp[:, self.ind_map["lam"]]
        beta = np.arcsin(np.clip(temp[:, self.ind_map["sinbeta"]], -1.0, 1.0))
        psi = temp[:, self.ind_map["psi"]]
        t_ref = temp[:, self.ind_map["t_ref"]]

        if self.coord_frame == "icrs":
            from ...sources.utils import icrs_to_ecliptic

            # (ra, dec, psi_icrs) -> (lam, beta, psi_ecl): one analytic
            # fixed-obliquity convention for the sky AND the polarization.
            lam, beta, psi = icrs_to_ecliptic(np.asarray(lam), np.asarray(beta), np.asarray(psi))

        t_L, lam_L, beta_L, psi_L = SSB_to_LISA(
            t_ref, lam, beta, psi, t0=self._frame_t0_years(t_ref)
        )

        temp[:, self.ind_map["lam"]] = lam_L % (2 * np.pi)
        temp[:, self.ind_map["sinbeta"]] = np.sin(beta_L)
        temp[:, self.ind_map["psi"]] = psi_L % np.pi
        temp[:, self.ind_map["t_ref"]] = t_L
        return temp

    def _from_lisa_frame(self, coords):
        """Inverse of :meth:`_to_lisa_frame` (applied after the mode hop)."""
        if self.coord_frame == "lisa":
            return coords

        from bbhx.utils.transform import LISA_to_SSB

        temp = coords.copy()
        lam_L = temp[:, self.ind_map["lam"]]
        beta_L = np.arcsin(np.clip(temp[:, self.ind_map["sinbeta"]], -1.0, 1.0))
        psi_L = temp[:, self.ind_map["psi"]]
        t_L = temp[:, self.ind_map["t_ref"]]

        # the phase is pinned at the LISA-frame time; the SSB time differs by
        # at most R/c ~ 500 s, over which alpha moves by ~1e-4 rad --
        # the same Omega-linear extrapolation the forward call makes.
        t_ref, lam, beta, psi = LISA_to_SSB(
            t_L, lam_L, beta_L, psi_L, t0=self._frame_t0_years(t_L)
        )

        if self.coord_frame == "icrs":
            from ...sources.utils import ecliptic_to_icrs

            lam, beta, psi = ecliptic_to_icrs(np.asarray(lam), np.asarray(beta), np.asarray(psi))

        temp[:, self.ind_map["lam"]] = lam % (2 * np.pi)
        temp[:, self.ind_map["sinbeta"]] = np.sin(beta)
        temp[:, self.ind_map["psi"]] = psi % np.pi
        temp[:, self.ind_map["t_ref"]] = t_ref
        return temp

    def _log_abs_det_jacobian(self, coords, fn=None):
        """``ln|det d fn / d x|`` over the four frame columns, by central
        differences (output angles unwrapped on their periods). ``fn``
        defaults to :meth:`_to_lisa_frame`. Zero for ``coord_frame == "lisa"``.
        """
        if self.coord_frame == "lisa":
            return np.zeros(coords.shape[0])
        fn = self._to_lisa_frame if fn is None else fn
        cols = [self.ind_map[k] for k in self._FRAME_COLUMNS]
        n = coords.shape[0]
        J = np.empty((n, 4, 4))
        for j, (c, h) in enumerate(zip(cols, self._FD_STEPS)):
            xp = coords.copy()
            xm = coords.copy()
            xp[:, c] += h
            xm[:, c] -= h
            yp = fn(xp)
            ym = fn(xm)
            for i, (ci, period) in enumerate(zip(cols, self._FD_PERIODS)):
                d = yp[:, ci] - ym[:, ci]
                if period is not None:
                    d = (d + period / 2) % period - period / 2
                J[:, i, j] = d / (2 * h)
        return np.log(np.abs(np.linalg.det(J)))

    # ------------------------------------------------------------------
    # LISA-frame mode-hop transforms
    # ------------------------------------------------------------------
    def lat_transform(self, coords, random):
        """Reflect coordinates across the LISA-plane latitude symmetry.

        Flips ``sinbeta`` and ``cosinc`` and reflects ``psi`` about
        :math:`\\pi/2`. Assumes ``coords`` is a 2D array indexed by
        ``self.ind_map`` and that the latitude variable is parameterized as
        :math:`\\sin\\beta`.

        Args:
            coords: 2D coordinate array with shape ``(n, ndim)``.
            random: Random state object (unused for this transform).

        Returns:
            Transformed copy of ``coords``.
        """
        temp = coords.copy()

        temp[:, self.ind_map["sinbeta"]] *= -1
        temp[:, self.ind_map["cosinc"]] *= -1
        temp[:, self.ind_map["psi"]] = (np.pi - temp[:, self.ind_map["psi"]]) % np.pi

        return temp

    def long_transform(self, coords, random):
        """Rotate longitude / polarization by a random multiple of :math:`\\pi/2`.

        For each row a value drawn uniformly from ``{0, 1, 2, 3}`` times
        :math:`\\pi/2` is added to ``lam`` and ``psi`` and they are then
        wrapped into their canonical ranges.

        Args:
            coords: 2D coordinate array with shape ``(n, ndim)``.
            random: Random state with a ``randint`` method used to draw the
                rotation amount.

        Returns:
            Transformed copy of ``coords``.
        """
        temp = coords.copy()

        move_amount = random.randint(0, 4, size=coords.shape[0]) * np.pi / 2.0

        temp[:, self.ind_map["psi"]] += move_amount
        temp[:, self.ind_map["lam"]] += move_amount

        temp[:, self.ind_map["psi"]] %= np.pi
        temp[:, self.ind_map["lam"]] %= 2 * np.pi

        return temp

    def both_transform(self, coords, random):
        """Apply :meth:`lat_transform` to a random subset, then :meth:`long_transform` to all.

        Combining the latitude and longitude transforms selects uniformly
        from the eight equivalent LISA sky modes.

        Args:
            coords: 2D coordinate array with shape ``(n, ndim)``.
            random: Random state object.

        Returns:
            Transformed coordinate array.
        """
        # if doing both does not assume it will cross plane, selects from 8 modes
        temp = coords.copy()
        inds_lat_change = random.randint(0, 2, size=coords.shape[0]).astype(bool)
        temp[inds_lat_change] = self.lat_transform(temp[inds_lat_change], random)
        temp = self.long_transform(temp, random)
        return temp

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Build a sky-mode-hop proposal for each branch.

        When ``coord_frame != "lisa"`` the active coordinates are converted
        to the LISA frame, mode-hopped, and converted back, so the sampler
        basis never leaves its declared frame; the returned factors carry
        the frame-conversion Jacobian ``ln|det J_F(x)| - ln|det J_F(x')|``
        summed over a walker's active leaves (zero when ``jacobian=False``
        or in the LISA frame, where the hop is symmetric).

        Args:
            branches_coords: Keys are branch names and values are arrays of
                shape ``(ntemps, nwalkers, nleaves_max, ndim)`` holding the
                current walker coordinates.
            random: Current random state object.
            branches_inds: Keys are branch names and values are boolean
                arrays of shape ``(ntemps, nwalkers, nleaves_max)`` selecting
                which leaves are currently active. If ``None``, all leaves
                are treated as active.
            **kwargs: Unused; accepted for ``eryn`` move compatibility.

        Returns:
            Tuple ``(q, factors)`` where ``q`` is a dict of proposed
            coordinates with the same shape as ``branches_coords`` and
            ``factors`` is an array of shape ``(ntemps, nwalkers)`` of
            log proposal-density factors.

        """

        q = {}
        factors = None
        for name, coords in zip(branches_coords.keys(), branches_coords.values()):

            if branches_inds is None:
                inds = np.ones(coords.shape[:-1], dtype=bool)

            else:
                inds = branches_inds[name]

            ntemps, nwalkers, _, _ = coords.shape
            if factors is None:
                factors = np.zeros((ntemps, nwalkers))
            inds_here = np.where(inds == True)

            q[name] = coords.copy()
            old_coords = coords[inds_here]
            new_coords = self._to_lisa_frame(old_coords)
            new_coords = self.transform(new_coords, random)
            new_coords = self._from_lisa_frame(new_coords)
            q[name][inds_here] = new_coords.copy()

            if self.jacobian and self.coord_frame != "lisa" and old_coords.shape[0] > 0:
                dlog = self._log_abs_det_jacobian(old_coords) - self._log_abs_det_jacobian(
                    new_coords
                )
                np.add.at(factors, (inds_here[0], inds_here[1]), dlog)

        return q, factors
