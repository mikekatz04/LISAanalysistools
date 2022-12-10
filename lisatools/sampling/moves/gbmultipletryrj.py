# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
from multiprocessing.sharedctypes import Value
import numpy as np
import warnings
import time

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.state import State, BranchSupplimental
from eryn.moves import ReversibleJump, MultipleTryMove
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds

from gbgpu.utils.utility import get_N, get_fdot

from lisatools.sampling.moves.gbspecialstretch import GBSpecialStretchMove

__all__ = ["GBMutlipleTryRJ"]

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

class GBMutlipleTryRJ(MultipleTryMove, ReversibleJump, GBSpecialStretchMove):
    """Generate Revesible-Jump proposals for GBs with multiple try

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        gb_args,
        gb_kwargs,
        m_chirp_lims,
        *args,
        start_ind_limit=10,
        num_try=1,
        point_generator_func=None,
        **kwargs
    ):
        self.point_generator_func = point_generator_func
        self.fixed_like_diff = 0
        self.time = 0
        self.name = "gbgroupstretch"
        self.start_ind_limit = start_ind_limit
        self.num_try = num_try

        GBSpecialStretchMove.__init__(self, *gb_args, **gb_kwargs)
        ReversibleJump.__init__(self, *args, **kwargs)
        MultipleTryMove.__init__(self, self.num_try, take_max_ll=False)

        # setup band edges for priors
        self.band_edges_fdot = np.zeros_like(self.band_edges)
        # lower limit
        self.band_edges_fdot[:-1] = get_fdot(self.band_edges[:-1], Mc=m_chirp_lims[0])
        self.band_edges_fdot[1:] = get_fdot(self.band_edges[1:], Mc=m_chirp_lims[1])

    def special_generate_func(self, coords, nwalkers, current_priors=None, random=None, size:int=1, fill=None, fill_inds=None, band_inds=None):
        """if self.search_samples is not None:
            # TODO: make replace=True ? in PE
            inds_drawn = np.array([random.choice(
                self.search_inds, size=size, replace=False,
            ) for w in range(nwalkers)])
            generated_points = self.search_samples[inds_drawn].copy()  # .reshape(nwalkers, size, -1)
            # since this is in search only, we will pretend these are coming from the prior
            # so that they are accepted based on the Likelihood (and not the posterior)
            generate_factors = current_priors.logpdf(generated_points.reshape(nwalkers * size, -1)).reshape(nwalkers, size)
        """

        if band_inds is None:
            raise ValueError("band_inds needs to be set")

        # elif
        if self.point_generator_func is not None:
            if hasattr(self.point_generator_func, "rvs") and hasattr(self.point_generator_func, "logpdf"):
                generated_points = self.point_generator_func.rvs(size=size * nwalkers)
                generate_factors = self.point_generator_func.logpdf(generated_points)
            else:
                generated_points, generate_factors = self.point_generator_func(size=size * nwalkers)

            generated_points = generated_points.reshape(nwalkers, size, -1)
            generate_factors = generate_factors.reshape(nwalkers, size)
            
            starts = self.band_edges[band_inds]
            ends = self.band_edges[band_inds + 1]

            starts_fdot = self.band_edges_fdot[band_inds]
            ends_fdot = self.band_edges_fdot[band_inds + 1]

            generated_points[:, :, 1] = (generated_points[:, :, 1] * ((ends - starts) + starts)[:, None]) * 1e3
            generated_points[:, :, 2] = generated_points[:, :, 2] * ((ends_fdot - starts_fdot) + starts_fdot)[:, None]

            # logpdf contribution from original distribution is zero = log(1/1)
            generate_factors[:] += (np.log(1 / (ends - starts)))[:, None]
            generate_factors[:] += (np.log(1 / (ends_fdot - starts_fdot)))[:, None]

        else:
            if current_priors is None:
                raise ValueError("If generating from the prior, must provide current_priors kwargs.")

            generated_points = current_priors.rvs(size=nwalkers * size).reshape(nwalkers, size, -1)

            # fill before getting logpdf
            if fill is not None or fill_inds is not None:
                if fill is None or fill_inds is None:
                    raise ValueError("If providing fill_inds or fill, must provide both.")
                generated_points[fill_inds] = fill.copy()

            generate_factors = current_priors.logpdf(generated_points.reshape(nwalkers * size, -1)).reshape(nwalkers, size)

        return generated_points, generate_factors

    def special_like_func(self, generated_points, coords, inds, inds_reverse=None, old_d_h_d_h=None, overall_inds=None):

        if overall_inds is None:
            raise ValueError("overall_inds is None.")

        # group everything

        # GENERATED POINTS MUST BE PASSED IN by reference not copied 
        num_inds_change, nleaves_max, ndim = coords.shape
        num_inds_change_gen, num_try, ndim_gen = generated_points.shape
        assert num_inds_change_gen == num_inds_change and ndim == ndim_gen

        if old_d_h_d_h is None:
            raise NotImplementedError
            self.d_h_d_h = d_h_d_h = 4 * self.df * self.xp.sum((in_vals.conj() * in_vals) / psd, axis=(1, 2))
        else:
            self.d_h_d_h = d_h_d_h = self.xp.asarray(old_d_h_d_h)

        ll_out = np.zeros((num_inds_change, num_try)).flatten()
        # TODO: take out of loop later?

        phase_marginalize = self.search
        generated_points_here = generated_points.reshape(-1, ndim)
        
        back_d_d = self.gb.d_d.copy()
        self.gb.d_d = self.xp.repeat(d_h_d_h, self.num_try)

        data_index_tmp = self.xp.repeat(overall_inds, self.num_try)
        data_index = self.xp.asarray(self.mgh.get_mapped_indices(data_index_tmp).astype(self.xp.int32))
        noise_index = data_index.copy()

        # TODO: Remove batch_size if GPU only ?
        prior_generated_points = generated_points_here

        if self.parameter_transforms is not None:
                prior_generated_points_in = self.parameter_transforms.both_transforms(
                    prior_generated_points.copy()
                )

        N_temp = self.xp.asarray(
            get_N(prior_generated_points_in[:, 0], prior_generated_points_in[:, 1], self.waveform_kwargs["T"], self.waveform_kwargs["oversample"])
        )

        waveform_kwargs_in = self.waveform_kwargs.copy()
        waveform_kwargs_in.pop("N")

        breakpoint()
        ll = self.gb.get_ll(prior_generated_points_in, self.mgh.data_list, self.mgh.psd_list, data_index=data_index, noise_index=noise_index, phase_marginalize=phase_marginalize, data_length=self.data_length,  data_splits=self.mgh.gpu_splits,  N=N_temp, **waveform_kwargs_in)

        if np.any(np.isnan(ll)):
            assert np.isnan(ll).sum() < 10
            ll[np.isnan(ll)] = -1e300

        opt_snr = self.xp.sqrt(self.gb.h_h)

        if self.search:
            phase_maximized_snr = (
                self.xp.abs(self.gb.d_h) / self.xp.sqrt(self.gb.h_h)
            ).real.copy()
            
            
            phase_change = self.xp.angle(self.xp.asarray(self.gb.non_marg_d_h) / self.xp.sqrt(self.gb.h_h.real))

            try:
                phase_maximized_snr = phase_maximized_snr.get()
                phase_change = phase_change.get()
                opt_snr = opt_snr.get()

            except AttributeError:
                pass

            if np.any(np.isnan(prior_generated_points)) or np.any(np.isnan(phase_change)):
                breakpoint()

            # adjust for phase change from maximization
            generated_points_here[:, 3] = (generated_points_here[:, 3] - phase_change) % (2 * np.pi)

            snr_comp = phase_maximized_snr

        else:
            snr_comp = (
                self.gb.d_h.real / self.xp.sqrt(self.gb.h_h)
            ).real.copy()
            try:
                snr_comp = snr_comp.get()
                opt_snr = opt_snr.get()

            except AttributeError:
                pass

        ##print(opt_snr[snr_comp.argmax()].real, snr_comp.max(), ll[snr_comp.argmax()].real - -1/2 * self.gb.d_d[snr_comp.argmax()].real)
        if self.search and self.search_snr_lim is not None:
            ll[
                (snr_comp
                < self.search_snr_lim * 0.8)
                | (opt_snr
                < self.search_snr_lim * self.search_snr_accept_factor)
            ] = -1e300
            """if np.any(~((snr_comp
                < self.search_snr_lim * 0.95)
                | (opt_snr
                < self.search_snr_lim * self.search_snr_accept_factor))):
                breakpoint()"""

        generated_points[:] = generated_points_here.reshape(generated_points.shape)
        ll_out = ll.copy()  
        #if inds_reverse is not None and len(inds_reverse) != 0 and split == num_splits - 1:
        #    breakpoint()  

        if inds_reverse is not None:
            try:
                tmp_d_h_d_h = d_h_d_h.get()
            except AttributeError:
                tmp_d_h_d_h = d_h_d_h

            self.special_aux_ll = (-1./2. * tmp_d_h_d_h[inds_reverse]).real  #  + self.noise_ll[inds_reverse]

        #breakpoint()
        # return gb.d_d
        self.gb.d_d = back_d_d.copy()
        # add noise term
        return ll_out.reshape(num_inds_change, num_try)  #  + self.noise_ll[:, None]

    def special_prior_func(self, generated_points, coords, inds, **kwargs):
        nwalkers, nleaves_max, ndim = coords.shape
    
        lp_new = self.priors["gb"].logpdf(generated_points.reshape(-1, 8)).reshape(nwalkers, self.num_try)
        lp_total = self.lp_old[:, None] + lp_new
        # add noise lp
        return lp_total

    def get_proposal(self, gb_coords, inds, changes, leaf_inds_for_changes, band_inds, random, supps=None, branch_supps=None):
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
  
        name = "gb"
        
        ntemps, nwalkers, nleaves_max, ndim = gb_coords.shape

        # adjust inds

        inds_reverse = tuple(np.where(changes == -1))

        inds_reverse_in = np.arange(len(inds_reverse[0]))
        inds_reverse_individual = leaf_inds_for_changes[changes < 0]

        # adjust births from False -> True
        inds_forward = tuple(np.where(changes == 1))

        num_forward = len(inds_forward[0])

        inds_reverse_in += num_forward

        # keep tmp_inds without added value here

        # add coordinates for new leaves
        current_priors = self.priors[name]
        inds_here = tuple([np.concatenate([inds_forward[jjj], inds_reverse[jjj]]) for jjj in range(len(inds_reverse))])
        
        # allows for adjustment during search
        reset_num_try = False
        if len(inds_here[0]) != 0:
            if self.search and self.search_samples is not None:
                raise not NotImplementedError
                if len(self.search_inds) >= self.num_try:
                    reset_num_try = True
                    old_num_try = self.num_try
                self.num_try = len(self.search_inds)

            num_inds_change = len(inds_here[0])

            if self.provide_betas:  #  and not self.search:
                betas = self.temperature_control.betas[inds_here[0]]
            else:
                betas = None

            ll_here = self.current_state.log_like.copy()[inds_here[:2]]
            lp_here = self.current_state.log_prior[inds_here[:2]]

            self.lp_old = lp_here

            old_d_h_d_h = -2 * ll_here

            rj_info = dict(
                ll=ll_here,
                lp=lp_here
            )

            N_vals = branch_supps.holder["N_vals"]
            N_vals = self.xp.asarray(N_vals)

            self.inds_reverse = inds_reverse
            self.inds_forward = inds_forward

            if len(inds_reverse[0]) > 0:
                parameters_remove = self.parameter_transforms.both_transforms(gb_coords[inds_reverse])

                group_index_tmp = inds_reverse[0] * nwalkers + inds_reverse[1]
                N_vals_in = self.xp.asarray(N_vals[inds_reverse])
                group_index = self.xp.asarray(self.mgh.get_mapped_indices(group_index_tmp).astype(self.xp.int32))

                waveform_kwargs_add = self.waveform_kwargs.copy()
                waveform_kwargs_add.pop("N")
                # removing these so d - (h - r) = d - h + r
                self.gb.generate_global_template(
                    parameters_remove,
                    group_index, 
                    self.mgh.data_list,
                    N=N_vals_in,
                    data_length=self.data_length, 
                    data_splits=self.mgh.gpu_splits, 
                    **waveform_kwargs_add
                )

                # self.checkit4 =  (-1/2 * self.df * 4 * self.xp.sum(data_minus_template.conj() * data_minus_template / self.psd, axis=(2,3))) + self.xp.asarray(self.noise_ll).reshape(ntemps, nwalkers)

            coords_inds = (gb_coords[inds_here[:2]], inds[inds_here[:2]])

            overall_inds = supps.holder["overall_inds"][inds_here[:2]]
            
            assert np.prod(band_inds.shape) == inds_here[0].shape[0]
            generate_points_out, logP_out, factors_out = self.get_mt_proposal(
                gb_coords[inds_here],
                len(inds_here[0]), 
                inds_reverse_in,
                inds_reverse_individual,
                random, 
                args_prior=coords_inds,
                kwargs_generate={"current_priors": current_priors, "band_inds": band_inds.flatten()}, 
                args_like=coords_inds, 
                kwargs_like={"inds_reverse": inds_reverse_in, "old_d_h_d_h": old_d_h_d_h, "overall_inds": overall_inds},
                betas=betas,
                rj_info=rj_info,
            )

            # gb_coords[inds_forward] = generate_points_out.copy()

            # TODO: make sure detailed balance this will move to detailed balance in multiple try
            
            self.logP_out = logP_out
            return  generate_points_out, self.ll_out, self.lp_out, factors_out
        
        else:
            breakpoint()

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        
        # st = time.perf_counter()
        # TODO: keep this?
        # this exposes anywhere in the proposal class to this information
        self.current_state = state
        self.current_model = model

        # Run any move-specific setup.
        self.setup(state.branches)

        # ll_before = model.compute_log_like_fn(state.branches_coords, inds=state.branches_inds, supps=state.supplimental, branch_supps=state.branches_supplimental)

        # if not np.allclose(ll_before[0], state.log_like):
        #    breakpoint()

        new_state = State(state, copy=True)

        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        ntemps, nwalkers, _, _ = new_state.branches[list(new_state.branches.keys())[0]].shape
        
        num_consecutive_rj_moves = 10
        print("starting")
        st = time.perf_counter()
        for rj_move_i in range(num_consecutive_rj_moves):
            accepted = np.zeros((ntemps, nwalkers), dtype=bool)
            
            coords_propose_in = new_state.branches_coords["gb"]
            inds_propose_in = new_state.branches_inds["gb"]
            branches_supp_propose_in = new_state.branches_supplimental["gb"]
            remaining_coords = coords_propose_in[inds_propose_in]
            f0 = remaining_coords[:, 1] / 1e3
            inds_into_full_f0 = np.arange(f0.shape[0])
            
            band_inds = np.searchsorted(self.band_edges, f0) - 1
            temp_inds = np.repeat(np.arange(ntemps)[:, None], np.prod(coords_propose_in.shape[1:3]), axis=1).reshape(ntemps, nwalkers, -1)[inds_propose_in]
            walkers_inds = np.repeat(np.repeat(np.arange(nwalkers)[None, :], ntemps, axis=0)[:, :, None],  coords_propose_in.shape[2], axis=2)[inds_propose_in]
            leaf_inds = np.repeat(np.arange(coords_propose_in.shape[2])[None, :], ntemps * nwalkers, axis=0).reshape(ntemps, nwalkers, -1)[inds_propose_in]

            # TODO: make this weighted somehow to not focus on empty bands
            # we are already leaving out the last one
            bands_per_walker = np.tile((np.arange(self.num_bands - 1)), (ntemps, nwalkers, 1))

            # odds & evens (evens first)
            for i in range(2):
                bands_per_walker_here = bands_per_walker[bands_per_walker % 2 == i].reshape(ntemps, nwalkers, -1)
                max_band_ind = bands_per_walker_here.max().item()
                band_inds_here = int(1e12) * temp_inds[band_inds % 2 == i] + int(1e6) * walkers_inds[band_inds % 2 == i] + band_inds[band_inds % 2 == i]

                inds_into_full_f0_here = inds_into_full_f0[band_inds.flatten() % 2 == i]

                inds_for_shuffle = np.arange(len(inds_into_full_f0_here))
                # shuffle allows the first unique index to be the one that is dropped
                np.random.shuffle(inds_for_shuffle)
                bands_inds_here_tmp = band_inds_here[inds_for_shuffle]

                assert band_inds_here.shape == inds_into_full_f0_here.shape

                unique_bands_inds_here, unique_bands_inds_here_index, unique_bands_inds_here_inverse, unique_bands_inds_here_counts = np.unique(bands_inds_here_tmp, return_index=True, return_counts=True, return_inverse=True)
                
                selected_inds_from_shuffle = inds_for_shuffle[unique_bands_inds_here_index]
                inds_into_full_f0_here_selected = inds_into_full_f0_here[selected_inds_from_shuffle]
                bands_inds_here_selected = band_inds_here[selected_inds_from_shuffle]

                temp_index_count = temp_inds[inds_into_full_f0_here_selected]
                walkers_index_count = walkers_inds[inds_into_full_f0_here_selected]
                leaf_index_count = leaf_inds[inds_into_full_f0_here_selected]

                band_index_count = np.floor((unique_bands_inds_here - temp_index_count * int(1e12) - walkers_index_count * int(1e6))).astype(int)
                band_index_tmp = np.searchsorted(bands_per_walker_here[0, 0], band_index_count, side="left")


                nleaves_here = np.zeros_like(bands_per_walker_here)

                nleaves_here[(temp_index_count, walkers_index_count, band_index_tmp)] = unique_bands_inds_here_counts

                changes = np.random.choice([-1, 1], size=bands_per_walker_here.shape)

                changes[nleaves_here == 0] = +1
                changes[nleaves_here == self.max_k[1]] = -1

                leaf_inds_for_change = np.zeros((ntemps, nwalkers, bands_per_walker_here.shape[-1]), dtype=int)
                
                leaf_inds_for_change[(temp_index_count, walkers_index_count, band_index_tmp)] = leaf_index_count

                leaf_inds_tmp = np.repeat(np.arange(coords_propose_in.shape[2])[None, :], ntemps * nwalkers, axis=0).reshape(ntemps, nwalkers, -1)
                
                leaf_inds_tmp[inds_propose_in] = int(1e6)
                leaf_inds_tmp = np.sort(leaf_inds_tmp, axis=-1)[:, :, :bands_per_walker_here.shape[-1]]
                leaf_inds_for_change[changes > 1] = leaf_inds_tmp[changes > 1]

                #et = time.perf_counter()
                #print("start", et - st)
                # st = time.perf_counter()
                # propose new sources and coordinates
                new_coords, ll_out, lp_out, factors = self.get_proposal(
                    coords_propose_in, inds_propose_in, changes, leaf_inds_for_change, bands_per_walker_here, model.random, branch_supps=branches_supp_propose_in, supps=new_state.supplimental
                )
                
                #et = time.perf_counter()
                #print("proposal", et - st)
                # st = time.perf_counter()
                breakpoint()
                # TODO: check this
                edge_factors = self.xp.zeros_like(factors)
                # get factors for edges
                for (name, branch), min_k, max_k in zip(
                    new_state.branches.items(), self.min_k, self.max_k
                ):

                    # do not work on sources with fixed source count
                    if min_k == max_k:
                        continue

                    # fix proposal asymmetry at bottom of k range
                    inds_min = np.where(nleaves_here == min_k)
                    # numerator term so +ln
                    edge_factors[inds_min] += np.log(1 / 2.0)

                    # fix proposal asymmetry at top of k range
                    inds_max = np.where(nleaves_here == max_k)
                    # numerator term so -ln
                    edge_factors[inds_max] += np.log(1 / 2.0)

                    # fix proposal asymmetry at bottom of k range (kmin + 1)
                    inds_min = np.where(nleaves_here == min_k + 1)
                    # numerator term so +ln
                    edge_factors[inds_min] -= np.log(1 / 2.0)

                    # fix proposal asymmetry at top of k range (kmax - 1)
                    inds_max = np.where(nleaves_here == max_k - 1)
                    # numerator term so -ln
                    edge_factors[inds_max] -= np.log(1 / 2.0)

                factors += edge_factors

                logp = lp_out
                #et = time.perf_counter()
                #print("prior", et - st)
                # st = time.perf_counter()
                logl = ll_out
                #loglcheck, new_blobs = model.compute_log_like_fn(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)
                #if not np.all(np.abs(logl[logl != -1e300] - loglcheck[logl != -1e300]) < 1e-5):
                #    breakpoint()
                
                logP = self.compute_log_posterior(logl, logp)

                assert np.allclose(logP, self.logP_out.reshape(ntemps, nwalkers))

                prev_logl = new_state.log_like

                prev_logp = new_state.log_prior

                # TODO: check about prior = - inf
                # takes care of tempering
                prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

                # TODO: fix this
                # this is where _metropolisk should come in
                lnpdiff = factors + logP - prev_logP

                accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))
                #et = time.perf_counter()
                #print("through accepted", et - st)
                # st = time.perf_counter()

                # bookkeeping

                inds_reverse = tuple(np.where(changes == -1))

                # adjust births from False -> True
                inds_forward = tuple(np.where(changes == 1))

                # not accepted removals
                accepted_reverse = accepted[inds_reverse[:2]]
                accepted_inds_reverse = tuple([ir[accepted_reverse] for ir in inds_reverse])
                not_accepted_inds_reverse = tuple([ir[~accepted_reverse] for ir in inds_reverse])

            if len(not_accepted_inds_reverse[0]) > 0:
                points_not_accepted_removal = new_coords.reshape(ntemps, nwalkers, -1)[not_accepted_inds_reverse[:2]]
                assert np.allclose(points_not_accepted_removal, coords_propose_in[not_accepted_inds_reverse])
            else:
                points_not_accepted_removal = np.empty((0, 8))

            # accepted removals
            new_state.branches["gb"].inds[accepted_inds_reverse] = False
            new_state.log_like[accepted_inds_reverse[:2]] = logl[accepted_inds_reverse[:2]]
            new_state.log_prior[accepted_inds_reverse[:2]] = logp[accepted_inds_reverse[:2]]

            # accepted additions
            inds_forward = tuple(inds_for_change["+1"].T)
            accepted_forward = accepted[inds_forward[:2]]
            accepted_inds_forward = tuple([ir[accepted_forward] for ir in inds_forward])
            not_accepted_inds_forward = tuple([ir[~accepted_forward] for ir in inds_forward])

            if len(accepted_inds_forward[0]) > 0:
                points_accepted_addition = new_coords.reshape(ntemps, nwalkers, -1)[accepted_inds_forward[:2]]
                
            else:
                points_accepted_addition = self.xp.empty((0, 8))

            new_state.branches["gb"].inds[accepted_inds_forward] = True
            new_state.branches["gb"].coords[accepted_inds_forward] = points_accepted_addition
            new_state.log_like[accepted_inds_forward[:2]] = logl[accepted_inds_forward[:2]]
            new_state.log_prior[accepted_inds_forward[:2]] = logp[accepted_inds_forward[:2]]

            points_to_add_to_template = np.concatenate([
                points_accepted_addition, 
                points_not_accepted_removal,  # were removed at the start, need to be put back
            ], axis=0)

            if points_to_add_to_template.shape[0] > 0:
                
                points_to_add_to_template_in = self.parameter_transforms.both_transforms(
                    points_to_add_to_template
                )
                N_temp = get_N(points_to_add_to_template_in[:, 0], points_to_add_to_template_in[:, 1], self.waveform_kwargs["T"], self.waveform_kwargs["oversample"])

                inds_here_for_add = tuple([
                    np.concatenate([accepted_inds_forward[jjj], not_accepted_inds_reverse[jjj]]) for jjj in range(len(not_accepted_inds_reverse))
                ])

                new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][inds_here_for_add] = N_temp

                group_index_tmp_accepted_forward = accepted_inds_forward[0] * nwalkers + accepted_inds_forward[1]
                group_index_tmp_not_accepted_reverse = not_accepted_inds_reverse[0] * nwalkers + not_accepted_inds_reverse[1]
                group_index_tmp =  np.concatenate([
                    group_index_tmp_accepted_forward,
                    group_index_tmp_not_accepted_reverse,
                ])
                group_index = self.xp.asarray(self.mgh.get_mapped_indices(group_index_tmp).astype(self.xp.int32))

                # adding these so d - (h + a) = d - h - a
                factors = -self.xp.ones_like(N_temp, dtype=self.xp.float64)
                waveform_kwargs_add = self.waveform_kwargs.copy()
                waveform_kwargs_add.pop("N")
                self.gb.generate_global_template(
                    points_to_add_to_template_in,
                    group_index, 
                    self.mgh.data_list,
                    N=N_temp,
                    data_length=self.data_length, 
                    data_splits=self.mgh.gpu_splits, 
                    factors=factors,
                    **waveform_kwargs_add
                )

        et = time.perf_counter()
        print("rj", (et - st) / num_consecutive_rj_moves)
        breakpoint()
        if self.temperature_control is not None and not self.prevent_swaps:
             new_state, accepted = self.temperature_control.temper_comps(new_state, accepted, adapt=False)
        if np.any(new_state.log_like > 1e10):
            breakpoint()
        #et = time.perf_counter()
        #print("swapping", et - st)
        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        self.time += 1
        return new_state, accepted

