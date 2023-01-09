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
from eryn.prior import ProbDistContainer
from eryn.utils.utility import groups_from_inds

from gbgpu.utils.utility import get_N, get_fdot

from lisatools.sampling.moves.gbspecialstretch import GBSpecialStretchMove

__all__ = ["GBMutlipleTryRJ"]

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def searchsorted2d_vec(a,b, xp=None, **kwargs):
    if xp is None:
        xp = np
    m,n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num*xp.arange(a.shape[0])[:,None]
    p = xp.searchsorted( (a+r).ravel(), (b+r).ravel(), **kwargs).reshape(m,-1)
    return p - n*(xp.arange(m)[:,None])


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
            generated_points = self.point_generator_func.rvs(size=size * nwalkers)
            
            generate_factors = self.point_generator_func.logpdf(generated_points)

            starts = self.band_edges[band_inds]
            ends = self.band_edges[band_inds + 1]

            starts_fdot = self.band_edges_fdot[band_inds]
            ends_fdot = self.band_edges_fdot[band_inds + 1]
            
            # fill before getting logpdf
            generated_points = generated_points.reshape(nwalkers, size, -1)
            generate_factors = generate_factors.reshape(nwalkers, size)

            # map back per band
            generated_points[:, :, 1] = (generated_points[:, :, 1] * (ends[:, None] - starts[:, None]) + starts[:, None]) * 1e3
            generated_points[:, :, 2] =  (generated_points[:, :, 2] * (ends_fdot[:, None] - starts_fdot[:, None]) + starts_fdot[:, None])

            if fill is not None or fill_inds is not None:
                if fill is None or fill_inds is None:
                    raise ValueError("If providing fill_inds or fill, must provide both.")
                generated_points[fill_inds] = fill.copy()

            generated_points = generated_points.reshape(nwalkers, size, -1)

            # logpdf contribution from original distribution is zero = log(1/1)
            generate_factors[:] += (np.log(1 / (ends - starts)))[:, None]
            generate_factors[:] += (np.log(1 / (ends_fdot - starts_fdot)))[:, None]

            generated_points = generated_points.reshape(nwalkers, size, -1)
            
        else:
            if current_priors is None:
                raise ValueError("If generating from the prior, must provide current_priors kwargs.")

            generated_points = current_priors.rvs(size=nwalkers * size).reshape(nwalkers, size, -1)
            if fill is not None or fill_inds is not None:
                if fill is None or fill_inds is None:
                    raise ValueError("If providing fill_inds or fill, must provide both.")
                generated_points[fill_inds[0]] = fill.copy()

            generate_factors = current_priors.logpdf(generated_points.reshape(nwalkers * size, -1)).reshape(nwalkers, size)

        return generated_points, generate_factors

    def special_like_func(self, generated_points, base_shape, inds_reverse=None, old_d_h_d_h=None, overall_inds=None):

        self.xp.cuda.runtime.setDevice(7)

        if overall_inds is None:
            raise ValueError("overall_inds is None.")

        # group everything

        # GENERATED POINTS MUST BE PASSED IN by reference not copied 
        num_inds_change, nleaves_max, ndim = base_shape
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

        # do not need mapping because it comes in as overall inds already mapped
        data_index = self.xp.asarray(self.xp.repeat(overall_inds, self.num_try).astype(self.xp.int32))
        
        noise_index = data_index.copy()

        self.data_index_check = data_index.reshape(-1, self.num_try)[:, 0]

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
        main_gpu = self.xp.cuda.runtime.getDevice()
        # TODO: do search sorted and apply that to nearest found for new points found with group

        ll = self.gb.get_ll(prior_generated_points_in, self.mgh.data_list, self.mgh.psd_list, data_index=data_index, noise_index=noise_index, phase_marginalize=phase_marginalize, data_length=self.data_length,  data_splits=self.mgh.gpu_splits,  N=N_temp, **waveform_kwargs_in)
        self.xp.cuda.runtime.setDevice(main_gpu)
        self.xp.cuda.runtime.deviceSynchronize()
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
        
        ll_out = ll_out.reshape(num_inds_change, num_try)
        self.old_ll_out_check = ll_out.copy()
        if inds_reverse is not None:
            try:
                tmp_d_h_d_h = d_h_d_h.get()
            except AttributeError:
                tmp_d_h_d_h = d_h_d_h

            # this is special to GBs
            self.special_aux_ll = -ll_out[inds_reverse, 0]
            
            ll_out[inds_reverse, :] += self.special_aux_ll[:, None]

        # return gb.d_d
        self.gb.d_d = back_d_d.copy()
        # add noise term
        self.xp.cuda.runtime.deviceSynchronize()

        return ll_out #  + self.noise_ll[:, None]

    def special_prior_func(self, generated_points, base_shape, inds_reverse=None, **kwargs):
        nwalkers, nleaves_max, ndim = base_shape
        lp_new = self.priors["gb_fixed"].logpdf(generated_points.reshape(-1, 8)).reshape(nwalkers, self.num_try)
        lp_total = lp_new  # _old[:, None] + lp_new

        self.old_lp_total_check = lp_total.copy()
        if inds_reverse is not None:
            # this is special to GBs
            self.special_aux_lp = -lp_total[inds_reverse, 0]
            
            lp_total[inds_reverse, :] += self.special_aux_lp[:, None]

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
  
        name = "gb_fixed"
        
        ntemps, nwalkers, nleaves_max, ndim = gb_coords.shape

        # adjust inds

        inds_reverse_band = np.where(changes == -1)

        inds_reverse = inds_reverse_band[:2] + (leaf_inds_for_changes[inds_reverse_band],)
        inds_reverse_in = np.where(changes.flatten() == -1)[0]
        inds_reverse_individual = leaf_inds_for_changes[changes < 0]

        inds_forward_band = np.where(changes == +1)

        inds_forward = inds_forward_band[:2] + (leaf_inds_for_changes[inds_forward_band],)
        # inds_forward_in = np.where(changes.flatten() == +1)[0]
        # inds_forward_individual = leaf_inds_for_changes[changes < 0]

        # add coordinates for new leaves
        current_priors = self.priors[name]
        
        inds_here_band = np.where((changes == -1) | (changes == +1))

        inds_here = inds_here_band[:2] + (leaf_inds_for_changes[inds_here_band],)
        
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

            rj_info = dict(
                ll=np.zeros_like(ll_here),
                lp=np.zeros_like(lp_here)
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

                try:
                    self.xp.cuda.runtime.deviceSynchronize()
                    # parameters_remove[:, 0] *= 1e-30
                    self.gb.generate_global_template(
                        parameters_remove,
                        group_index, 
                        self.mgh.data_list,
                        N=N_vals_in,
                        data_length=self.data_length, 
                        data_splits=self.mgh.gpu_splits, 
                        **waveform_kwargs_add
                    )
                except ValueError as e:
                    print(e)
                    breakpoint()
                        
                # self.checkit4 =  (-1/2 * self.df * 4 * self.xp.sum(data_minus_template.conj() * data_minus_template / self.psd, axis=(2,3))) + self.xp.asarray(self.noise_ll).reshape(ntemps, nwalkers)

            overall_inds = supps.holder["overall_inds"][inds_here[:2]]
            base_shape = (len(inds_here[0]), 1, ndim)
            assert np.prod(band_inds.shape) == inds_here[0].shape[0]
            old_d_h_d_h = np.zeros_like(ll_here)

            
            generate_points_out, logP_out, factors_out = self.get_mt_proposal(
                gb_coords[inds_here],
                len(inds_here[0]), 
                inds_reverse_in,
                random, 
                args_prior=(base_shape, inds_reverse_in),
                kwargs_generate={"current_priors": current_priors, "band_inds": band_inds.flatten()}, 
                args_like=(base_shape,), 
                kwargs_like={"inds_reverse": inds_reverse_in, "old_d_h_d_h": old_d_h_d_h, "overall_inds": overall_inds},
                betas=betas,
                rj_info=rj_info,
            )

            # gb_coords[inds_forward] = generate_points_out.copy()

            # TODO: make sure detailed balance this will move to detailed balance in multiple try
            
            self.logP_out = logP_out

            return  generate_points_out, self.ll_out, self.lp_out, factors_out, betas, inds_here
        
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
        all_branch_names = list(new_state.branches_coords.keys())

        if np.any(new_state.branches_supplimental["gb_fixed"].holder["N_vals"][new_state.branches_inds["gb_fixed"]] == 0):
            breakpoint()

        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        ntemps, nwalkers, _, _ = new_state.branches[list(new_state.branches.keys())[0]].shape
        
        num_consecutive_rj_moves = 1
        print("starting")
        st = time.perf_counter()
        for rj_move_i in range(num_consecutive_rj_moves):
            accepted = np.zeros((ntemps, nwalkers), dtype=bool)
            
            coords_propose_in = new_state.branches_coords["gb_fixed"]
            inds_propose_in = new_state.branches_inds["gb_fixed"]
            branches_supp_propose_in = new_state.branches_supplimental["gb_fixed"]
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

                # shuffle and then take first index from band 
                # is a good way to get random binary from that band
                unique_bands_inds_here, unique_bands_inds_here_index, unique_bands_inds_here_inverse, unique_bands_inds_here_counts = np.unique(bands_inds_here_tmp, return_index=True, return_counts=True, return_inverse=True)
                
                # selected_inds_from_shuffle is indices reverted back to 
                # inds_into_full_f0_here after shuffle
                selected_inds_from_shuffle = inds_for_shuffle[unique_bands_inds_here_index]

                # frequencies of the first instance found in band according to np.unique call
                # and what bands they belong to
                inds_into_full_f0_here_selected = inds_into_full_f0_here[selected_inds_from_shuffle]
                bands_inds_here_selected = band_inds_here[selected_inds_from_shuffle]
              
                temp_index_count = temp_inds[inds_into_full_f0_here_selected]
                walkers_index_count = walkers_inds[inds_into_full_f0_here_selected]
                leaf_index_count = leaf_inds[inds_into_full_f0_here_selected]

                # get band index for each selected source by maths
                band_index_count = np.floor((unique_bands_inds_here - temp_index_count * int(1e12) - walkers_index_count * int(1e6))).astype(int)
                band_index_tmp = np.searchsorted(bands_per_walker_here[0, 0], band_index_count, side="left")

                nleaves_here = np.zeros_like(bands_per_walker_here)

                # how many leaves are in each band
                nleaves_here[(temp_index_count, walkers_index_count, band_index_tmp)] = unique_bands_inds_here_counts

                # setup leaf count change arrays
                changes = np.random.choice([-1, 1], size=bands_per_walker_here.shape)

                # make sure to add a binary if there are None
                changes[nleaves_here == 0] = +1

                if np.any(nleaves_here >= self.max_k[1]):
                    raise ValueError("nleaves_here higher than max_k.")

                # number of sub bands
                num_sub_bands_here = bands_per_walker_here.shape[-1]

                # build arrays to help properly index and locate within these sub bands
                temp_inds_for_change = np.repeat(np.arange(ntemps)[:, None], nwalkers * num_sub_bands_here, axis=-1).reshape(ntemps, nwalkers, num_sub_bands_here)
                walker_inds_for_change = np.tile(np.arange(nwalkers), (ntemps, num_sub_bands_here, 1)).transpose((0, 2, 1))
                band_inds_for_change = np.tile(np.arange(num_sub_bands_here), (ntemps, nwalkers, 1))

                # leaves will be filled later
                leaf_inds_for_change = -np.ones((ntemps, nwalkers, num_sub_bands_here), dtype=int)
                
                # add leaves that would be removed if count is not currently zero
                # will adjust for count soon
                # len(leaf_index_count) < total number of proposals
                leaf_inds_for_change[(temp_index_count, walkers_index_count, band_index_tmp)] = leaf_index_count

                # band_inds_for_change[(temp_index_count, walkers_index_count, band_index_tmp)] = band_index_tmp

                leaf_inds_tmp = np.repeat(np.arange(coords_propose_in.shape[2])[None, :], ntemps * nwalkers, axis=0).reshape(ntemps, nwalkers, -1)
                
                
                # any binary that exists give fake high value to remove from sort
                leaf_inds_tmp[inds_propose_in] = int(1e7)

                # this gives unused leaves, you take the number you need for each walker / temperature
                leaf_inds_tmp_2 = np.sort(leaf_inds_tmp, axis=-1)[:, :, :num_sub_bands_here]
                leaf_inds_for_change[changes > 0] = leaf_inds_tmp_2[changes > 0]

                assert not np.any(leaf_inds_for_change == -1)

                if np.any(leaf_inds_tmp_2 == int(1e7)):
                    raise ValueError("Not enough spots to allocate for new binaries. Need to increase max leaves.")

                # TODO: check that number of available spots is high enough

                #et = time.perf_counter()
                #print("start", et - st)
                # st = time.perf_counter()
                # propose new sources and coordinates
                new_coords, ll_out, lp_out, factors, betas, inds_here = self.get_proposal(
                    coords_propose_in, inds_propose_in, changes, leaf_inds_for_change, bands_per_walker_here, model.random, branch_supps=branches_supp_propose_in, supps=new_state.supplimental
                )

                #et = time.perf_counter()
                #print("proposal", et - st)
                # st = time.perf_counter()

                # TODO: check this
                edge_factors = np.zeros_like(factors)
                # get factors for edges
                min_k = self.min_k[all_branch_names.index("gb_fixed")]
                max_k = self.max_k[all_branch_names.index("gb_fixed")]

                # fix proposal asymmetry at bottom of k range
                inds_min = np.where(nleaves_here.flatten() == min_k)
                # numerator term so +ln
                edge_factors[inds_min] += np.log(1 / 2.0)

                # fix proposal asymmetry at top of k range
                inds_max = np.where(nleaves_here.flatten() == max_k)
                # numerator term so -ln
                edge_factors[inds_max] += np.log(1 / 2.0)

                # fix proposal asymmetry at bottom of k range (kmin + 1)
                inds_min = np.where(nleaves_here.flatten() == min_k + 1)
                # numerator term so +ln
                edge_factors[inds_min] -= np.log(1 / 2.0)

                # fix proposal asymmetry at top of k range (kmax - 1)
                inds_max = np.where(nleaves_here.flatten()  == max_k - 1)
                # numerator term so -ln
                edge_factors[inds_max] -= np.log(1 / 2.0)

                factors += edge_factors

                prev_logl = new_state.log_like[inds_here[:2]]

                prev_logp = new_state.log_prior[inds_here[:2]]

                #et = time.perf_counter()
                #print("prior", et - st)
                # st = time.perf_counter()
                logl = prev_logl + ll_out
                logp = prev_logp + lp_out
                #loglcheck, new_blobs = model.compute_log_like_fn(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)
                #if not np.all(np.abs(logl[logl != -1e300] - loglcheck[logl != -1e300]) < 1e-5):
                #    breakpoint()

                logP = self.compute_log_posterior(logl, logp, betas=betas)

                # TODO: check about prior = - inf
                # takes care of tempering
                prev_logP = self.compute_log_posterior(prev_logl, prev_logp, betas=betas)

                # TODO: fix this
                # this is where _metropolisk should come in
                lnpdiff = factors + logP - prev_logP

                accepted = lnpdiff > np.log(model.random.rand(*lnpdiff.shape))

                #et = time.perf_counter()
                #print("through accepted", et - st)
                # st = time.perf_counter()
                
                # bookkeeping

                inds_reverse = np.where(changes.flatten() == -1)[0]

                # adjust births from False -> True
                inds_forward = np.where(changes.flatten() == +1)[0]

                # accepted_keep = (temp_inds_for_change.flatten()[inds_forward] == 0) & (walker_inds_for_change.flatten()[inds_forward] == 1)

                """accepted_keep_tmp = np.where(accepted_keep)[0]
                accepted_keep[accepted_keep_tmp[:]] = False
                accepted_keep[accepted_keep_tmp[164679]] = True"""

                # accepted[:] = False
                # accepted[inds_forward] = True
                """accepted[inds_forward[:]] = False
                accepted[inds_reverse[0]] = False"""

                # not accepted removals
                accepted_reverse = accepted[inds_reverse]
                accepted_inds_reverse = inds_reverse[accepted_reverse]
                not_accepted_inds_reverse = inds_reverse[~accepted_reverse]

                tuple_not_accepted_reverse = (
                    temp_inds_for_change.flatten()[not_accepted_inds_reverse], 
                    walker_inds_for_change.flatten()[not_accepted_inds_reverse], 
                    leaf_inds_for_change.flatten()[not_accepted_inds_reverse], 
                )

                tuple_accepted_reverse = (
                    temp_inds_for_change.flatten()[accepted_inds_reverse], 
                    walker_inds_for_change.flatten()[accepted_inds_reverse], 
                    leaf_inds_for_change.flatten()[accepted_inds_reverse], 
                )


                if len(not_accepted_inds_reverse) > 0:
                    points_not_accepted_removal = coords_propose_in[tuple_not_accepted_reverse]

                else:
                    points_not_accepted_removal = np.empty((0, 8))

                # accepted removals
                new_state.branches["gb_fixed"].inds[tuple_accepted_reverse] = False
                delta_logl_trans1 = np.zeros_like(leaf_inds_for_change, dtype=float)
                delta_logl_trans1[(
                    temp_inds_for_change.flatten()[accepted_inds_reverse], 
                    walker_inds_for_change.flatten()[accepted_inds_reverse],
                    band_inds_for_change.flatten()[accepted_inds_reverse],
                )] = logl[accepted_inds_reverse] - prev_logl[accepted_inds_reverse]

                new_state.log_like += delta_logl_trans1.sum(axis=-1)

                # accepted update logp
                delta_logp_trans = np.zeros_like(leaf_inds_for_change, dtype=float)
                delta_logp_trans[(
                    temp_inds_for_change.flatten()[accepted_inds_reverse], 
                    walker_inds_for_change.flatten()[accepted_inds_reverse],
                    band_inds_for_change.flatten()[accepted_inds_reverse],
                )] = logp[accepted_inds_reverse] - prev_logp[accepted_inds_reverse]

                new_state.log_prior += delta_logp_trans.sum(axis=-1)
                
                # accepted additions
                accepted_forward = accepted[inds_forward]
                accepted_inds_forward = inds_forward[accepted_forward]
                not_accepted_inds_forward = inds_forward[~accepted_forward]

                tuple_accepted_forward = (
                    temp_inds_for_change.flatten()[accepted_inds_forward], 
                    walker_inds_for_change.flatten()[accepted_inds_forward], 
                    leaf_inds_for_change.flatten()[accepted_inds_forward], 
                )
                new_state.branches["gb_fixed"].inds[tuple_accepted_forward] = True
                new_state.branches["gb_fixed"].coords[tuple_accepted_forward] = new_coords[accepted_forward]
                
                if len(accepted_inds_forward) > 0:
                    points_accepted_addition = new_coords[accepted_forward]

                    # get group friend finder information
                    f0_accepted_addition = -100 * np.ones_like(leaf_inds_for_change, dtype=float)
                    f0_accepted_addition[(
                        temp_inds_for_change.flatten()[accepted_inds_forward], 
                        walker_inds_for_change.flatten()[accepted_inds_forward],
                        band_inds_for_change.flatten()[accepted_inds_forward],
                    )] = points_accepted_addition[:, 1] / 1e3

                    for t in range(ntemps):
                        # use old state to get supp information
                        f0_old = state.branches["gb_fixed"].coords[t, state.branches["gb_fixed"].inds[t]][:, 1] / 1e3
                        friend_start_inds = state.branches["gb_fixed"].branch_supplimental.holder["friend_start_inds"][t, state.branches["gb_fixed"].inds[t]]

                        f0_old_sorted = np.sort(f0_old, axis=-1)
                        inds_f0_old_sorted = np.argsort(f0_old, axis=-1)
                        f0_accepted_addition_in1 = f0_accepted_addition[t]
                        f0_accepted_addition_in2 = f0_accepted_addition_in1[f0_accepted_addition_in1 > -1.0]
                        temp_inds_in = temp_inds_for_change[t][f0_accepted_addition_in1 > -1.0]
                        walker_inds_in = walker_inds_for_change[t][f0_accepted_addition_in1 > -1.0]
                        leaf_inds_in = leaf_inds_for_change[t][f0_accepted_addition_in1 > -1.0]
                        
                        inds_f0_accepted_addition = np.searchsorted(f0_old_sorted, f0_accepted_addition_in2, side="right") - 1

                        old_inds_f0_old = inds_f0_old_sorted[inds_f0_accepted_addition]
                        comp_f0_old = f0_old[old_inds_f0_old]
                        new_friends_start_inds = friend_start_inds[old_inds_f0_old]

                        # TODO: maybe check this
                        new_state.branches["gb_fixed"].branch_supplimental.holder["friend_start_inds"][(temp_inds_in, walker_inds_in, leaf_inds_in)] = new_friends_start_inds
                    
                    """inds_into_old_array = np.take_along_axis(inds_f0_old_sorted, inds_f0_accepted_addition, axis=-1).reshape(ntemps, nwalkers, -1)

                    new_start_inds = np.take_along_axis(state.branches["gb_fixed"].branch_supplimental.holder["friend_start_inds"], inds_into_old_array, axis=-1)

                    keep_new_start_inds = new_start_inds[f0_accepted_addition.reshape(ntemps, nwalkers, -1) > -10.0]
                    breakpoint()
                    state.branches["gb_fixed"].branch_supplimental.holder["friend_start_inds"][(

                    )] = keep_new_start_inds
                    
                    inds_into_old_array 
                    keep_inds_f0_accepted_addition = inds_f0_accepted_addition > 0



                    inds_into_old_array = inds_f0_old_sorted[inds_f0_accepted_addition]"""
                    
                else:
                    points_accepted_addition = np.empty((0, 8))


                delta_logl_trans2 = np.zeros_like(leaf_inds_for_change, dtype=float)
                delta_logl_trans2[(
                    temp_inds_for_change.flatten()[accepted_inds_forward], 
                    walker_inds_for_change.flatten()[accepted_inds_forward],
                    band_inds_for_change.flatten()[accepted_inds_forward],
                )] = logl[accepted_inds_forward] - prev_logl[accepted_inds_forward]

                new_state.log_like += delta_logl_trans2.sum(axis=-1)

                delta_logp_trans = np.zeros_like(leaf_inds_for_change, dtype=float)
                delta_logp_trans[(
                    temp_inds_for_change.flatten()[accepted_inds_forward], 
                    walker_inds_for_change.flatten()[accepted_inds_forward],
                    band_inds_for_change.flatten()[accepted_inds_forward],
                )] = logp[accepted_inds_forward] - prev_logp[accepted_inds_forward]

                new_state.log_prior += delta_logp_trans.sum(axis=-1)

                accepted_counts = np.zeros_like(leaf_inds_for_change, dtype=bool)
                num_proposals_here = leaf_inds_for_change.shape[-1]

                accepted_counts[(
                    temp_inds_for_change.flatten()[accepted], 
                    walker_inds_for_change.flatten()[accepted],
                    band_inds_for_change.flatten()[accepted],
                )] = True

                accepted_overall = accepted_counts.sum(axis=-1)

                # TODO: if we do not run every band, we need to adjust this
                self.accepted += accepted_overall
                self.num_proposals += num_proposals_here

                points_to_add_to_template = np.concatenate([
                    points_accepted_addition, 
                    points_not_accepted_removal,  # were removed at the start, need to be put back
                ], axis=0)

                if points_to_add_to_template.shape[0] > 0:
                    
                    points_to_add_to_template_in = self.parameter_transforms.both_transforms(
                        points_to_add_to_template, xp=self.xp
                    )
                    N_temp = get_N(points_to_add_to_template_in[:, 0], points_to_add_to_template_in[:, 1], self.waveform_kwargs["T"], self.waveform_kwargs["oversample"])

                    # has to be accepted forward first (sase as points_to_add_to_template)
                    inds_here_for_add = tuple([
                        np.concatenate([tuple_accepted_forward[jjj], tuple_not_accepted_reverse[jjj]]) for jjj in range(len(tuple_not_accepted_reverse))
                    ])

                    # inds_here_for_add = tuple_accepted_forward
                    
                    new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][inds_here_for_add] = N_temp.get()

                    group_index_tmp_accepted_forward = tuple_accepted_forward[0] * nwalkers + tuple_accepted_forward[1]
                    group_index_tmp_not_accepted_reverse = tuple_not_accepted_reverse[0] * nwalkers + tuple_not_accepted_reverse[1]

                    group_index_tmp =  np.concatenate([
                        group_index_tmp_accepted_forward,
                        group_index_tmp_not_accepted_reverse,
                    ])
                    group_index = self.xp.asarray(self.mgh.get_mapped_indices(group_index_tmp).astype(self.xp.int32))

                    # adding these so d - (h + a) = d - h - a
                    factors_multiply = -self.xp.ones_like(N_temp, dtype=self.xp.float64)
                    waveform_kwargs_add = self.waveform_kwargs.copy()
                    waveform_kwargs_add.pop("N")

                    # points_to_add_to_template_in[1:, 0] *= 1e-30 

                
                    self.gb.generate_global_template(points_to_add_to_template_in, group_index,  self.mgh.data_list, N=N_temp, data_length=self.data_length,  data_splits=self.mgh.gpu_splits,  factors=factors_multiply, **waveform_kwargs_add)

                    if np.any(new_state.branches_supplimental["gb_fixed"].holder["N_vals"][new_state.branches_inds["gb_fixed"]] == 0):
                        breakpoint()

        if self.time % 100 == 0:
            ll_after2 = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
            if np.abs(ll_after2 - new_state.log_like).max() > 1e-4:
                    breakpoint()

        et = time.perf_counter()
        print("rj", (et - st) / num_consecutive_rj_moves)

        if self.temperature_control is not None and not self.prevent_swaps:
             new_state = self.temperature_control.temper_comps(new_state, adapt=False)

        #et = time.perf_counter()
        #print("swapping", et - st)
        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()
        accepted = np.zeros_like(new_state.log_like)
        self.time += 1
        return new_state, accepted

