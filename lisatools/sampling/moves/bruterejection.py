# -*- coding: utf-8 -*-

import numpy as np
import warnings

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import ReversibleJump
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds

___ = ["BruteRejection"]


class BruteRejection:
    """Generate Revesible-Jump proposals for GBs with brute-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        num_brute,
        take_max_ll=False,
    ):
        # TODO: make priors optional like special generate function? 
        self.num_brute = num_brute
        self.take_max_ll = take_max_ll

    def get_bf_proposal(self, coords, nwalkers, random, args_generate=(), kwargs_generate={}, args_like=(), kwargs_like={}):
        """Make a proposal

        Args:
            coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            inds_for_change (dict): Keys are ``branch_names``. Values are
                dictionaries. These dictionaries have keys ``"+1"`` and ``"-1"``,
                indicating waklkers that are adding or removing a leafm respectively.
                The values for these dicts are ``int`` np.ndarray[..., 3]. The "..." indicates
                the number of walkers in all temperatures that fall under either adding
                or removing a leaf. The second dimension, 3, is the indexes into
                the three-dimensional arrays within ``inds`` of the specific leaf
                that is being added or removed from those leaves currently considered.
            random (object): Current random state of the sampler.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """

        factors = np.zeros((nwalkers,))

        # generate new points and get detailed balance info
        generated_points, generated_factors = self.special_generate_func(coords, nwalkers, *args_generate, random=random, size=self.num_brute, **kwargs_generate)
        ll = self.special_like_func(generated_points, *args_like, **kwargs_like)
        if self.take_max_ll:
            # get max
            inds_keep = np.argmax(ll, axis=-1)
            log_prob_factors = np.full(len(inds_keep), 0.0)

        else:
            if np.any(np.isnan(ll)):
                warnings.warn("Getting nans for ll in brute force.")
                ll[np.isnan(ll)] = -1e300
            temp_probs = np.exp(ll - ll.max(axis=-1, keepdims=True))
            probs = temp_probs / np.sum(temp_probs, axis=-1, keepdims=True)
            # draw based on likelihood
            inds_keep = (probs.cumsum(1) > np.random.rand(probs.shape[0])[:,None]).argmax(1)
            log_prob_factors = -np.log(probs[(np.arange(len(inds_keep)), inds_keep)])

        #log_prob_factors = np.log(probs[:, ind_keep])
        inds_tuple = (np.arange(len(inds_keep)), inds_keep)
        ll_out = ll[inds_tuple]
        generated_points_out = generated_points[inds_tuple].copy()

        self.ll_out = ll_out.copy()
        factors += generated_factors[inds_tuple]
        factors += log_prob_factors

        return generated_points_out, ll_out, factors
