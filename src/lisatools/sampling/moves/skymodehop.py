# -*- coding: utf-8 -*-
"""Sky-mode hopping Metropolis-Hastings move for LISA samplers."""

import numpy as np
from eryn.moves import MHMove

__all__ = ["SkyMove"]


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
      t_ref)``. The move additionally rotates ICRS <-> ecliptic (sky
      position via astropy, polarization via the local-north rotation
      angle) on each side of the SSB <-> LISA conversion.

    For the non-LISA frames the merger/reference time participates in the
    frame conversion (the LISA-frame time depends on the sky position), so
    ``ind_map`` must contain a ``"t_ref"`` entry.

    Args:
        ind_map: Mapping from coordinate names to their indices in the
            coordinate array. Defaults to
            ``{"cosinc": 6, "lam": 7, "sinbeta": 8, "psi": 9, "t_ref": 10}``
            (the MBH sampling basis). ``"t_ref"`` is only consumed when
            ``coord_frame != "lisa"``.
        which: Which transformation to apply. Must be one of ``"both"``,
            ``"lat"``, or ``"long"``.
        coord_frame: Frame of the sampled sky coordinates; one of
            ``"lisa"``, ``"ssb_ecliptic"``, ``"icrs"``. (default:
            ``"lisa"``)
        frame_t0: Constellation phase reference passed as ``t0`` (in
            **years**) to ``SSB_to_LISA`` / ``LISA_to_SSB``. If ``t_ref``
            is sampled relative to the data start rather than absolute SSB
            time, pass ``data_t0 / YRSID_SI`` here so the constellation
            phase is evaluated correctly. (default: ``0.0``)
        **kwargs: Additional keyword arguments forwarded to
            :class:`eryn.moves.MHMove`.

    Raises:
        ValueError: If ``ind_map`` is not a ``dict``, ``which`` is not one
            of the allowed strings, ``coord_frame`` is not one of the
            allowed strings, or ``coord_frame != "lisa"`` while ``ind_map``
            lacks a ``"t_ref"`` entry.

    """

    _ALLOWED_FRAMES = ("lisa", "ssb_ecliptic", "icrs")

    def __init__(self, ind_map=None, which="both", coord_frame="lisa", frame_t0=0.0, **kwargs):

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

        self.ind_map = ind_map
        self.which = which
        self.coord_frame = coord_frame
        self.frame_t0 = frame_t0
        self.transform = getattr(self, f"{which}_transform")
        super(SkyMove, self).__init__(**kwargs)

    # ------------------------------------------------------------------
    # Frame conversion machinery (only used when coord_frame != "lisa")
    # ------------------------------------------------------------------
    @classmethod
    def _psi_rotation_icrs_to_ecliptic(cls, lam_ecl, beta_ecl):
        """Angle ``chi`` such that ``psi_ecl = psi_icrs - chi`` at this sky position.

        Thin delegation to
        :func:`lisatools.sources.utils.psi_rotation_icrs_to_ecliptic`
        (the canonical implementation, shared with the catalogue
        conversions). Lazy import keeps this module light.
        """
        from ...sources.utils import psi_rotation_icrs_to_ecliptic

        return psi_rotation_icrs_to_ecliptic(lam_ecl, beta_ecl)

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
            from ...response.directresponse import icrs_to_ecliptic

            # (ra, dec) -> ecliptic position, then rotate psi into the
            # ecliptic polarization basis at the (common) sky point.
            lam, beta = icrs_to_ecliptic(lam, beta)
            psi = psi - self._psi_rotation_icrs_to_ecliptic(lam, beta)

        t_L, lam_L, beta_L, psi_L = SSB_to_LISA(t_ref, lam, beta, psi, t0=self.frame_t0)

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

        t_ref, lam, beta, psi = LISA_to_SSB(t_L, lam_L, beta_L, psi_L, t0=self.frame_t0)

        if self.coord_frame == "icrs":
            from ...response.directresponse import ecliptic_to_icrs

            # psi rotation is evaluated at the (new) ecliptic sky point —
            # the exact inverse of the rotation applied on the way in.
            psi = psi + self._psi_rotation_icrs_to_ecliptic(lam, beta)
            lam, beta = ecliptic_to_icrs(lam, beta)

        temp[:, self.ind_map["lam"]] = lam % (2 * np.pi)
        temp[:, self.ind_map["sinbeta"]] = np.sin(beta)
        temp[:, self.ind_map["psi"]] = psi % np.pi
        temp[:, self.ind_map["t_ref"]] = t_ref
        return temp

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
        temp[:, self.ind_map["psi"]] = np.pi - temp[:, self.ind_map["psi"]]

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
        inds_lat_change = random.randint(0, 2, size=coords.shape[0]).astype(bool)
        coords[inds_lat_change] = self.lat_transform(coords[inds_lat_change], random)
        coords = self.long_transform(coords, random)
        return coords

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Build a sky-mode-hop proposal for each branch.

        When ``coord_frame != "lisa"`` the active coordinates are converted
        to the LISA frame, mode-hopped, and converted back, so the sampler
        basis never leaves its declared frame. The frame conversions are
        deterministic bijections and the discrete hop is symmetric, so the
        returned log proposal-density factors are zero (the small Jacobian
        of the SSB <-> LISA time/sky mixing is neglected, consistent with
        standard LISA-frame sky-mode proposals).

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
            log proposal-density factors (zero here because the move is
            symmetric).

        """

        q = {}
        for name, coords in zip(branches_coords.keys(), branches_coords.values()):

            if branches_inds is None:
                inds = np.ones(coords.shape[:-1], dtype=bool)

            else:
                inds = branches_inds[name]

            ntemps, nwalkers, _, _ = coords.shape
            inds_here = np.where(inds == True)

            q[name] = coords.copy()
            new_coords = self._to_lisa_frame(coords[inds_here])
            new_coords = self.transform(new_coords, random)
            new_coords = self._from_lisa_frame(new_coords)
            q[name][inds_here] = new_coords.copy()

        return q, np.zeros((ntemps, nwalkers))
