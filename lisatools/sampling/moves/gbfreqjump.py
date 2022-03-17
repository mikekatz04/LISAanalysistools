# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import warnings

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import MHMove
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds
from .gbbruterejectionrj import GBBruteRejectionRJ


__all__ = ["GBFreqJump"]


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBFreqJump(MHMove, GBBruteRejectionRJ):
    """Generate Revesible-Jump proposals for GBs with brute-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        df,
        factor,
        gb_args,
        gb_kwargs,
        spread=1,
        *args,
        **kwargs
    ):

        self.df = df
        self.factor = factor

        self.spread = np.delete(np.arange(-spread, spread + 1), spread)
        gaussian = stats.norm(loc=0.0, scale=float(spread))

        self.p = gaussian.pdf(self.spread.astype(float))
        self.p /= self.p.sum()

        self.name = "gbfreqjump"

        GBBruteRejectionRJ.__init__(self, *gb_args, **gb_kwargs)
        MHMove.__init__(self, *args, **kwargs)

    def special_generate_func(self, coords, nwalkers, inds, current_priors=None, random=None, size:int=1):

        # get stuff from prior to fill in for any values that are not sampled
        #if current_priors is None:
        #    raise ValueError("If generating from the prior, must provide current_priors kwargs.")

        nwalkers_coords, nleaves_max, ndim = coords.shape

        assert nwalkers_coords == nwalkers
        # get frequency switch
        # factors are zero
        
        inds_update = np.zeros((nwalkers,), dtype=int)
        inds_good = np.zeros((nwalkers,), dtype=bool)

        arange_array = np.arange(nwalkers)

        # TODO: update
        max_iter = 1000
        current_iter = 0
        while not np.all(inds_good) and current_iter < max_iter:
            num_new = nwalkers - inds_good.sum()
            inds_new = np.random.randint(0, nleaves_max, size=(num_new,))
            inds_update[~inds_good] = inds_new
            inds_good = inds[(arange_array, inds_update,)].reshape(nwalkers,)
            current_iter += 1

        if current_iter == max_iter:
            raise ValueError("Could not draw necessary indexes.")

        self.inds_turn_off = (arange_array, inds_update,)

        # TODO: probably loop with numba
        generated_points = coords[(np.repeat(arange_array, size), np.repeat(inds_update, size))]

        change = generated_points * self.factor * np.random.randn(*generated_points.shape)
        change[:, 1] = 20 * self.df * 1e3 * np.random.randn(nwalkers * size)  # choice(self.spread, replace=True, size=(nwalkers * size,), p=self.p)
        
        generated_points += change

        for i in [4, 7]:
            generated_points[:, i][generated_points[:, i] > 1.0] -= 2 * np.abs(1.0 - generated_points[:, i][generated_points[:, i] > 1.0])
            generated_points[:, i][generated_points[:, i] < -1.0] += 2 * np.abs(-1.0 - generated_points[:, i][generated_points[:, i] < -1.0])
            if np.any(np.abs(generated_points[:, i]) > 1.0):
                breakpoint()

        # TODO: change fdot accordingly
        prior_draws = current_priors.rvs(size=(nwalkers * size,))
        prior_keep = np.array([2, 3, 4, 5])
        generated_points[:, prior_keep] = prior_draws[:, prior_keep]

        generated_points = generated_points.reshape(nwalkers, size, ndim)

        # TODO: check this
        generate_factors = np.zeros((nwalkers, size))

        return generated_points, generate_factors
      
    def get_proposal(self, branches_coords, branches_inds, random, supps=None, branch_supps=None):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            all_inds_for_change (dict): Keys are ``branch_names``. Values are
                dictionaries. These dictionaries have keys ``"+1"`` and ``"-1"``,
                indicating waklkers that are adding or removing a leafm respectively.
                The values for these dicts are ``int`` np.ndarray[..., 3]. The "..." indicates
                the number of walkers in all temperatures that fall under either adding
                or removing a leaf. The second dimension, 3, is the indexes into
                the three-dimensional arrays within ``all_inds`` of the specific leaf
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

        assert branch_supps is not None 

        q = {}
        new_inds = {}
        factors = {}

        for i, (name, coords, inds) in enumerate(
            zip(
                branches_coords.keys(),
                branches_coords.values(),
                branches_inds.values(),
            )
        ):

            ntemps, nwalkers, nleaves_max, ndim = coords.shape
            new_inds[name] = inds.copy()
            q[name] = coords.copy()

            if i > 0:
                raise NotImplementedError

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            num_inds_change = ntemps * nwalkers

            current_priors = self.priors[name]

            branch_supps_in = {}
            for key, value in branch_supps[name][:].items():
                branch_supps_in[key] = value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:])

            generate_points_out, ll_out, factors_out = self.get_bf_proposal(
                coords.reshape(ntemps * nwalkers, nleaves_max, ndim),
                num_inds_change, 
                random, 
                args_generate=(inds.reshape(ntemps * nwalkers, nleaves_max),),
                kwargs_generate={"current_priors": current_priors}, 
                args_like=(coords.reshape(ntemps * nwalkers, nleaves_max, ndim), inds.reshape(ntemps * nwalkers, nleaves_max)), 
                kwargs_like={"branch_supps": branch_supps_in}
            )
            inds_tuple = (
                np.repeat(np.arange(ntemps), nwalkers), 
                np.tile(np.arange(nwalkers), (ntemps, 1)).flatten(), 
                self.inds_turn_off[1]
            )
            q[name][inds_tuple] = generate_points_out

            inds_changed = np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)
            inds_changed[inds_tuple] = True

            self.global_template_builder(q, inds_keep={"gb": inds_changed}, branch_supps=branch_supps)

            # TODO: make sure detailed balance this will move to detailed balance in brute rejection
            factors[:] = factors_out.reshape(ntemps, nwalkers)

        return q, factors
