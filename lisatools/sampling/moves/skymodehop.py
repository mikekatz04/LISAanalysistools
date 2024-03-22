# -*- coding: utf-8 -*-

import numpy as np

from eryn.moves import MHMove

__all__ = ["GaussianMove"]


class SkyMove(MHMove):
    """A Metropolis step with a Gaussian proposal function.

    Args:


    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

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
        """
        assumes sin beta
        assumes 2d array with all coords
        coords[]
        """
        temp = coords.copy()

        temp[:, self.ind_map["sinbeta"]] *= -1
        temp[:, self.ind_map["cosinc"]] *= -1
        temp[:, self.ind_map["psi"]] = np.pi - temp[:, self.ind_map["psi"]]

        return temp

    def long_transform(self, coords, random):
        """
        assumes sin beta
        assumes 2d array with all coords
        coords[]
        """
        temp = coords.copy()

        move_amount = random.randint(0, 4, size=coords.shape[0]) * np.pi / 2.0

        temp[:, self.ind_map["psi"]] += move_amount
        temp[:, self.ind_map["lam"]] += move_amount

        temp[:, self.ind_map["psi"]] %= np.pi
        temp[:, self.ind_map["lam"]] %= 2 * np.pi

        return temp

    def both_transform(self, coords, random):

        # if doing both does not assume it will cross plane, selects from 8 modes
        inds_lat_change = random.randint(0, 2, size=coords.shape[0]).astype(bool)
        coords[inds_lat_change] = self.lat_transform(coords[inds_lat_change], random)
        coords = self.long_transform(coords, random)
        return coords

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from Gaussian distribution

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            branches_inds (dict): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max] representing which
                leaves are currently being used.
            random (object): Current random state object.

        """

        q = {}
        for name, coords in zip(
            branches_coords.keys(), branches_coords.values()
        ):

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
