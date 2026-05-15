# -*- coding: utf-8 -*-
"""Sky-mode hopping Metropolis-Hastings move for LISA samplers."""

import numpy as np
from eryn.moves import MHMove

__all__ = ["GaussianMove"]


class SkyMove(MHMove):
    """Metropolis move that jumps between equivalent LISA sky modes.

    The LISA response is approximately invariant under several discrete
    transformations of the sky-position / inclination / polarization
    parameters. This move proposes such a swap as the MCMC step, allowing the
    sampler to mix between otherwise isolated sky modes.

    Args:
        ind_map: Mapping from coordinate names to their indices in the
            coordinate array. Defaults to
            ``{"cosinc": 6, "lam": 7, "sinbeta": 8, "psi": 9}``.
        which: Which transformation to apply. Must be one of ``"both"``,
            ``"lat"``, or ``"long"``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`eryn.moves.MHMove`.

    Raises:
        ValueError: If ``ind_map`` is not a ``dict`` or ``which`` is not one of
            the allowed strings.

    """

    def __init__(self, ind_map=None, which="both", **kwargs):

        if ind_map is None:
            ind_map = dict(cosinc=6, lam=7, sinbeta=8, psi=9)

        elif isinstance(ind_map, dict) is False:
            raise ValueError("If providing the ind_map kwarg, it must be a dict.")

        if which not in ["both", "lat", "long"]:
            raise ValueError("which kwarg must be 'both', 'lat', or 'long'.")

        self.ind_map = ind_map
        self.which = which
        exec(f"self.transform = self.{which}_transform")
        super(SkyMove, self).__init__(**kwargs)

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
            new_coords = self.transform(coords[inds_here], random)
            q[name][inds_here] = new_coords.copy()

        return q, np.zeros((ntemps, nwalkers))
