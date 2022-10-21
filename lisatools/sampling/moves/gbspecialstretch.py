# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from lisatools.utils.utility import searchsorted2d_vec, get_groups_from_band_structure
from eryn.moves import StretchMove
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds
from .gbmultipletryrj import GBMutlipleTryRJ
from .gbgroupstretch import GBGroupStretchMove

from ...diagnostic import inner_product
from eryn.state import State


__all__ = ["GBSpecialStretchMove"]

# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialStretchMove(StretchMove):
    """Generate Revesible-Jump proposals for GBs with try-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        gb,
        priors,
        num_try,
        start_freq_ind,
        data_length,
        mgh,
        fd,
        band_edges,
        *args,
        waveform_kwargs={},
        noise_kwargs={},
        parameter_transforms=None,
        search=False,
        search_samples=None,
        search_snrs=None,
        search_snr_lim=None,
        search_snr_accept_factor=1.0,
        take_max_ll=False,
        global_template_builder=None,
        point_generator_func=None,
        psd_func=None,
        provide_betas=False,
        alternate_priors=None,
        batch_size=5,
        **kwargs
    ):
        StretchMove.__init__(self, **kwargs)  

        self.time = 0
        self.greater_than_1e0 = 0
        self.name = "gbgroupstretch"

        # TODO: make priors optional like special generate function? 
        for key in priors:
            if not isinstance(priors[key], PriorContainer):
                raise ValueError("Priors need to be eryn.priors.PriorContainer object.")
        self.priors = priors
        self.gb = gb
        self.provide_betas = provide_betas
        self.batch_size = batch_size
        self.stop_here = True

        # use gpu from template generator
        self.use_gpu = gb.use_gpu
        if self.use_gpu:
            self.xp = xp
            self.mempool = self.xp.get_default_memory_pool()

        else:
            self.xp = np


        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.num_try = num_try
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.noise_kwargs = noise_kwargs
        self.parameter_transforms = parameter_transforms
        self.psd_func = psd_func
        self.fd = fd
        self.df = (fd[1] - fd[0]).item()
        self.mgh = mgh
        self.search = search
        self.global_template_builder = global_template_builder
        self.point_generator_func = point_generator_func

        if search_snrs is not None:
            if search_snr_lim is None:
                search_snr_lim = 0.1

            assert len(search_samples) == len(search_snrs)

        self.search_samples = search_samples
        self.search_snrs = search_snrs
        self.search_snr_lim = search_snr_lim
        self.search_snr_accept_factor = search_snr_accept_factor

        self.take_max_ll = take_max_ll      

    def setup(self, branches):

        if "noise_params" in branches:
            ntemps, nwalkers, nleaves, ndim = branches["noise_params"].shape
            noise_params_here = branches["noise_params"].coords[branches["noise_params"].inds].reshape(ntemps * nwalkers, -1)

            if self.psd_func is None:
                raise ValueError("If providing noise_params, need to provide psd_func to __init__ function.")
            
            psd = self.xp.asarray([self.psd_func(self.fd, *noise_params_here.T, **self.noise_kwargs) for _ in range(2)]).transpose((1, 0, 2))
            noise_ll = -self.xp.sum(self.xp.log(psd), axis=(1, 2))
            
            try:
                noise_ll = noise_ll.get()
            except AttributeError:
                pass
        
            self.noise_ll = noise_ll.copy()
            
            # TODO: check this
            #ll_here -= noise_ll

            noise_lp = self.priors["noise_params"].logpdf(noise_params_here)
            self.noise_lp = noise_lp.copy()
            #lp_here -= noise_lp

            self.noise_ll = self.noise_ll.reshape(ntemps, nwalkers)
            self.noise_lp = self.noise_lp.reshape(ntemps, nwalkers)
        
        else:
            self.noise_ll = 0.0
            self.noise_lp = 0.0

    def setup_gbs(self, branch):
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        all_remaining_coords = coords[inds]
        remaining_wave_info = supps[inds]

        num_remaining = len(all_remaining_coords)
        # TODO: make faster?
        points_out = np.zeros((num_remaining, self.nfriends, ndim))
        
        # info_mat = self.gb.information_matrix()

        freqs = all_remaining_coords[:, 1]

        distances = np.abs(freqs[None, :] - freqs[:, None])
        distances[distances == 0.0] = 1e300
        
        """
        distances = self.xp.full((num_remaining,num_remaining), 1e300)
        for i, coords_here in enumerate(all_remaining_coords):
            A_here = remaining_wave_info["A"][i]
            E_here = remaining_wave_info["E"][i]
            sig_len = len(A_here)
            start_ind_here = remaining_wave_info["start_inds"][i].item()
            freqs_here = (self.xp.arange(sig_len) + start_ind_here) * self.df
            psd_here = self.psd[0][start_ind_here - self.start_freq_ind: start_ind_here - self.start_freq_ind + sig_len]
    
            h_h = inner_product([A_here, E_here], [A_here, E_here], f_arr=freqs_here, PSD=psd_here, use_gpu=self.use_gpu)
            
            for j in range(i, num_remaining):
                if j == i:
                    continue
                A_check = remaining_wave_info["A"][j]
                E_check = remaining_wave_info["E"][j]
                start_ind_check = remaining_wave_info["start_inds"][j].item()
                if abs(start_ind_here - start_ind_check) > self.start_ind_limit:
                    continue
                start_ind = self.xp.max(self.xp.asarray([start_ind_here, start_ind_check])).item()
                end_ind = self.xp.min(self.xp.asarray([start_ind_here + sig_len, start_ind_check + sig_len])).item()
                sig_len_new = end_ind - start_ind

                start_ind_now_here = start_ind - start_ind_here
                slice_here = slice(start_ind_now_here, start_ind_now_here + sig_len_new)

                start_ind_now_check = start_ind - start_ind_check
                slice_check = slice(start_ind_now_check, start_ind_now_check + sig_len_new)
                d_h = inner_product([A_here[slice_here], E_here[slice_here]], [A_check[slice_check], E_check[slice_check]], f_arr=freqs_here[slice_here], PSD=psd_here[slice_here], use_gpu=self.use_gpu)
                
                distances[i, j] = abs(1.0 - d_h.real / h_h.real)
                distances[j, i] = distances[i, j]
            print(i)

        keep = self.xp.argsort(distances)[:self.nfriends]
        try:
            keep = keep.get()
        except AttributeError:
            pass
        

        breakpoint()
        """
        keep = np.argsort(distances, axis=1)[:, :self.nfriends]
        supps[inds] = {"group_move_points": all_remaining_coords[keep]}

    def setup_noise_params(self, branch):
        
        coords = branch.coords
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape

        par0 = coords[:, :, :, 0].flatten()

        distances = np.abs(par0[None, :] - par0[:, None])
        distances[distances == 0.0] = 1e300
        
        keep = np.argsort(distances, axis=1)[:, :self.nfriends]
        supps[:] = {"group_move_points": coords.reshape(-1, 1)[keep].reshape(ntemps, nwalkers, nleaves_max, self.nfriends, ndim)}

    def find_friends(self, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name == "gb_fixed":
                self.setup_gbs(branch)
            elif name == "noise_params":
                self.setup_noise_params(branch)
            else:
                raise NotImplementedError

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        # st = time.perf_counter()
        self.xp.cuda.runtime.setDevice(self.mgh.gpus[0])

        xp.random.seed(10)
        np.random.seed(10)
        #print("start stretch")
        #st = time.perf_counter()
        # Check that the dimensions are compatible.
        ndim_total = 0
        for branch in state.branches.values():
            ntemps, nwalkers, nleaves_, ndim_ = branch.shape
            ndim_total += ndim_ * nleaves_

        # TODO: deal with more intensive acceptance fractions
        # Run any move-specific setup.
        self.setup(state.branches)

        new_state = State(state, copy=True)
        self.mempool.free_all_blocks()
        true_data_map = self.xp.asarray(new_state.supplimental.holder["overall_inds"].flatten())
        
        self.mgh.map = true_data_map.get()

        # data should not be whitened
        
        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb_fixed"].shape
        
        f_test = state.branches_coords["gb_fixed"][:, :, :, 1] / 1e3
        if np.any(f_test == 0):
            breakpoint()

        # TODO: add actual amplitude
        N_vals = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"]

        N_vals = self.xp.asarray(N_vals)

        N_vals_2_times = self.xp.concatenate([N_vals, N_vals], axis=-1)

        gb_fixed_coords = self.xp.asarray(new_state.branches_coords["gb_fixed"].copy())

        log_prob_tmp = self.xp.asarray(new_state.log_prob)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)
      
        self.mempool.free_all_blocks()

        unique_N = np.unique(N_vals)
        
        groups = get_groups_from_band_structure(f_test, self.band_edges, xp=np)

        unique_groups, group_len = np.unique(groups.flatten(), return_counts=True)
        num_groups = len(unique_groups)

        waveform_kwargs_now = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs_now:
            waveform_kwargs_now.pop("N")
        waveform_kwargs_now["start_freq_ind"] = self.start_freq_ind

        group_temp_finder = [
            self.xp.repeat(self.xp.arange(ntemps), nwalkers * nleaves_max).reshape(ntemps, nwalkers, nleaves_max),
            self.xp.tile(self.xp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose((0, 2, 1)),
            self.xp.tile(self.xp.arange(nleaves_max), ((ntemps, nwalkers, 1)))
        ]

        split_inds = np.zeros(nwalkers, dtype=int)
        split_inds[1::2] = 1
        np.random.shuffle(split_inds)
        """et = time.perf_counter()
        print("setup", (et - st))
        st = time.perf_counter()"""
        for split in range(2):
            # st = time.perf_counter()
            split_here = split_inds == split
            walkers_keep = self.xp.arange(nwalkers)[split_here]
                
            points_to_move = gb_fixed_coords[:, split_here]
            points_for_move = gb_fixed_coords[:, ~split_here]

            q_temp, factors_temp = self.get_proposal(
                {"gb_fixed": points_to_move.transpose(0, 2, 1, 3).reshape(ntemps * nleaves_max, int(nwalkers / 2), 1, ndim)},  
                {"gb_fixed": [points_for_move.transpose(0, 2, 1, 3).reshape(ntemps * nleaves_max, int(nwalkers / 2), 1, ndim)]}, 
                model.random
            )

            factors = self.xp.zeros((ntemps, nwalkers, nleaves_max))
            factors[:, split_here] = factors_temp.reshape(ntemps, nleaves_max, int(nwalkers / 2)).transpose(0, 2, 1)

            # use new_state here to get change after 1st round
            q = {"gb_fixed": gb_fixed_coords.copy()}

            q["gb_fixed"][:, split_here] = q_temp["gb_fixed"].reshape(ntemps, nleaves_max, int(nwalkers / 2), ndim).transpose(0, 2, 1, 3)

            """et = time.perf_counter()
            print("prop", (et - st))"""

            for group_iter in range(num_groups):
                # st = time.perf_counter()
                # sometimes you will have an extra odd or even group only
                # the group_iter may not match the actual running group number in this case
                if group_iter not in groups:
                    continue
                group = [grp[groups == group_iter].flatten() for grp in group_temp_finder]

                # st = time.perf_counter()
                temp_inds, walkers_inds, leaf_inds = [self.xp.asarray(grp) for grp in group] 

                keep_inds = self.xp.in1d(walkers_inds, walkers_keep)

                temp_inds_for_gen = temp_inds[~keep_inds]
                walkers_inds_for_gen = walkers_inds[~keep_inds]
                leaf_inds_for_gen = leaf_inds[~keep_inds]

                temp_inds_keep = temp_inds[keep_inds]
                walkers_inds_keep = walkers_inds[keep_inds]
                leaf_inds_keep = leaf_inds[keep_inds]                

                group_here = (temp_inds_keep, walkers_inds_keep, leaf_inds_keep)
                group_here_for_gen = (temp_inds_for_gen, walkers_inds_for_gen, leaf_inds_for_gen)

                factors_here = factors[group_here]

                old_points = gb_fixed_coords[group_here]
                new_points = q["gb_fixed"][group_here]

                if self.use_gpu:
                    new_points_prior = new_points.get()
                    old_points_prior = old_points.get()
                else:
                    new_points_prior = new_points
                    old_points_prior = old_points

                logp = self.xp.asarray(self.priors["gb_fixed"].logpdf(new_points_prior))
                keep_here = self.xp.where((~self.xp.isinf(logp)))

                if len(keep_here[0]) == 0:
                    continue
                        
                points_remove = self.parameter_transforms.both_transforms(old_points[keep_here], xp=self.xp)
                points_add = self.parameter_transforms.both_transforms(new_points[keep_here], xp=self.xp)

                data_index_tmp = self.xp.asarray((temp_inds_keep[keep_here] * nwalkers + walkers_inds_keep[keep_here]).astype(xp.int32))
                noise_index_tmp = self.xp.asarray((temp_inds_keep[keep_here] * nwalkers + walkers_inds_keep[keep_here]).astype(xp.int32))
                
                data_index = self.mgh.get_mapped_indices(data_index_tmp).astype(self.xp.int32)
                noise_index = self.mgh.get_mapped_indices(noise_index_tmp).astype(self.xp.int32)

                assert self.xp.all(data_index == noise_index)
                nChannels = 2

                delta_ll = self.xp.full(old_points.shape[0], -1e300)
                """new_state.log_prob[0, 5] = new_state.log_prob[0, 3]
                new_state.log_prior[0, 5] = new_state.log_prior[0, 3]
                inds_change = np.where((temp_inds_keep[keep_here] == 0) & (walkers_inds_keep[keep_here] == 3))
                inds_check = np.where((temp_inds_keep[keep_here] == 0) & (walkers_inds_keep[keep_here] == 5))
                factors_here[inds_check] = factors_here[inds_change]
                

                points_remove[inds_check, :] = points_remove[inds_change, :].copy()

                points_add[inds_check, :] = points_add[inds_change, :].copy()
                data_minus_template[0, 5] = data_minus_template[0, 3].copy()"""
                
                N_in_group = N_vals[group_here][keep_here]
                """et = time.perf_counter()
                print("before like", (et - st), group_iter, group_len[list(unique_groups).index(group_iter)])
                st = time.perf_counter()"""

                delta_ll[keep_here] = self.gb.swap_likelihood_difference(points_remove, points_add, self.mgh.data_list,  self.mgh.psd_list,  N=N_in_group, data_index=data_index,  noise_index=noise_index,  adjust_inplace=False,  data_length=self.data_length, 
                data_splits=self.mgh.gpu_splits, phase_marginalize=self.search, **waveform_kwargs_now)
                """et = time.perf_counter()
                print("after like", (et - st), group_iter, group_len[list(unique_groups).index(group_iter)])
                st = time.perf_counter()"""
                """dhr = self.gb.d_h_remove.copy()
                dha = self.gb.d_h_add.copy()
                aa = self.gb.add_add.copy()
                rr = self.gb.remove_remove.copy()
                ar = self.gb.add_remove.copy()

                kwargs_tmp = self.waveform_kwargs.copy()
                kwargs_tmp["use_c_implementation"] = False
                check_tmp = self.gb.swap_likelihood_difference(points_remove,  points_add,  data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).copy(),  psd.reshape(ntemps * nwalkers, nChannels, -1).copy(),  data_index=data_index,  noise_index=noise_index,  adjust_inplace=False,  **kwargs_tmp)
                breakpoint()"""
                
                if self.xp.any(self.xp.isnan(delta_ll)):
                    warnings.warn("Getting nan in Likelihood function.")
                    breakpoint()
                    logp[self.xp.isnan(delta_ll)] = -np.inf
                    delta_ll[self.xp.isnan(delta_ll)] = -1e300
                    
                optimized_snr = self.xp.sqrt(self.gb.add_add.real)
                detected_snr = (self.gb.d_h_add + self.gb.add_remove).real / optimized_snr
                # check_d_h_add = self.gb.d_h_add.copy()
                # check_add_remove = self.gb.add_remove.copy()
                if self.search:
                    """et = time.perf_counter()
                    print("before search stuff", (et - st), N_now, group_iter, group_len[group_iter])
                    st = time.perf_counter()"""
                    inds_fix = ((optimized_snr < self.search_snr_lim) | (detected_snr < (0.8 * self.search_snr_lim)))

                    if self.xp == np:
                        inds_fix = inds_fix.get()

                    if self.xp.any(inds_fix):
                        logp[keep_here[0][inds_fix]] = -np.inf
                        delta_ll[keep_here[0][inds_fix]] = -1e300

                    phase_change = self.gb.phase_angle
                    new_points[keep_here, 3] -= phase_change
                    points_add[:, 4] -= phase_change

                    new_points[keep_here, 3] = new_points[keep_here, 3] % (2 * np.pi)
                    points_add[:, 4] = points_add[:, 4] % (2 * np.pi)

                    """
                    check = self.gb.swap_likelihood_difference(points_remove,  points_add,  self.mgh.data_list,  self.mgh.psd_list,  data_index=data_index,  noise_index=noise_index,  adjust_inplace=False,  data_length=self.data_length, data_splits=self.mgh.gpu_splits, phase_marginalize=False, **waveform_kwargs_now)
                    breakpoint()
                    """
                    """et = time.perf_counter()
                    print("after search stuff", (et - st), N_now, group_iter, group_len[group_iter])
                    st = time.perf_counter()"""

                prev_logl = log_prob_tmp[(temp_inds_keep, walkers_inds_keep)]
                logl = delta_ll + prev_logl

                #if np.any(logl - np.load("noise_ll.npy").flatten() > 0.0):
                #    breakpoint()    
                #print("multi check: ", (logl - np.load("noise_ll.npy").flatten()))

                prev_logp = self.xp.asarray(self.priors["gb_fixed"].logpdf(old_points_prior))

                if np.any(np.isinf(prev_logp)):
                    breakpoint()
                
                """prev_logp[inds_check] = prev_logp[inds_change]
                logp[inds_check] = logp[inds_change] """
                betas_in = self.xp.asarray(self.temperature_control.betas)[temp_inds_keep]
                logP = self.compute_log_posterior(logl, logp, betas=betas_in)
                
                # TODO: check about prior = - inf
                # takes care of tempering
                prev_logP = self.compute_log_posterior(prev_logl, prev_logp, betas=betas_in)

                # TODO: think about factors in tempering
                lnpdiff = factors_here + logP - prev_logP

                keep = lnpdiff > self.xp.asarray(self.xp.log(self.xp.random.rand(*logP.shape)))
                """et = time.perf_counter()
                print("keep determination", (et - st), group_iter, group_len[list(unique_groups).index(group_iter)])
                st = time.perf_counter()"""
                if self.xp.any(keep):
                    # for testing
                    #keep = np.zeros_like(keep, dtype=bool)
                    #keep[np.where((temp_inds_keep == 0) & (walkers_inds_keep == 1))] = True
                    #keep[np.where((temp_inds_keep == 0) & (walkers_inds_keep == 1))[0][4]] = False
                    #keep[np.isinf(logp)] = False
                    # keep[inds_check] = keep[inds_change]

                    # if gibbs sampling, this will say it is accepted if
                    # any of the gibbs proposals were accepted

                    accepted_here = keep.copy()

                    # check freq overlap
                    nleaves_max = state.branches["gb_fixed"].nleaves_max
                    if nleaves_max > 1:
                        check_f0 = self.xp.zeros((ntemps, nwalkers, nleaves_max))
                        check_f0_old = self.xp.zeros((ntemps, nwalkers, nleaves_max))
                        #check_f0_old = self.xp.zeros((ntemps, nwalkers, nleaves_max))
                        check_f0[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])] = new_points[keep][:, 1] / 1e3
                        check_f0_old[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])] = old_points[keep][:, 1] / 1e3

                        """check_f0_sorted = self.xp.sort(check_f0, axis=-1)
                        inds_f0_sorted = self.xp.argsort(check_f0, axis=-1)

                        check_f0_old_sorted = self.xp.sort(check_f0_old, axis=-1)
                        inds_f0_old_sorted = self.xp.argsort(check_f0_old, axis=-1)
                        """
                        check_f0_both = self.xp.concatenate([check_f0, check_f0_old], axis=-1)
                        check_f0_both_sorted = self.xp.sort(check_f0_both, axis=-1)
                        inds_f0_both_sorted = self.xp.argsort(check_f0_both, axis=-1)

                        N_vals_both_sorted = self.xp.take_along_axis(N_vals_2_times, inds_f0_both_sorted, axis=-1)

                        diff_f_1 = (check_f0_both_sorted[:, :, 1:] - check_f0_both_sorted[:, :, :-1]) * (((inds_f0_both_sorted[:, :, 1:] % nleaves_max) != (inds_f0_both_sorted[:, :, :-1] % nleaves_max)) & (check_f0_both_sorted[:, :, 1:] != 0.0) & (check_f0_both_sorted[:, :, :-1] != 0.0))
                        diff_f_2 = check_f0_both_sorted[:, :, 2:] - check_f0_both_sorted[:, :, :-2]

                        N_check_both_1 = N_vals_both_sorted[:, :, 1:]

                        fix_1 = self.xp.zeros_like(check_f0_both_sorted, dtype=bool)
                        fix_1[:, :, 1:] = ((diff_f_1 / self.df).astype(int) < N_check_both_1) & (diff_f_1 != 0.0)
                        fix_1[:, :, :-1] = fix_1[:, :, :-1] | (((diff_f_1 / self.df).astype(int) < N_check_both_1) & (diff_f_1 != 0.0))

                        diff_f_2 = (check_f0_both_sorted[:, :, 2:] - check_f0_both_sorted[:, :, :-2]) * (((inds_f0_both_sorted[:, :, 2:] % nleaves_max) != (inds_f0_both_sorted[:, :, :-2] % nleaves_max)) & (check_f0_both_sorted[:, :, 2:] != 0.0) & (check_f0_both_sorted[:, :, :-2] != 0.0))

                        N_check_both_2 = N_vals_both_sorted[:, :, 2:]

                        fix_2 = self.xp.zeros_like(check_f0_both_sorted, dtype=bool)
                        fix_2[:, :, 2:] = ((diff_f_2 / self.df).astype(int) < N_check_both_2) & (diff_f_2 != 0.0)
                        fix_2[:, :, :-2] = fix_2[:, :, :-2] | (((diff_f_2 / self.df).astype(int) < N_check_both_2) & (diff_f_2 != 0.0))

                        fix_all = (fix_1 | fix_2)

                        if self.xp.any(fix_all):
                            # breakpoint()
                            temp_inds_fix, walkers_inds_fix, leaf_map_inds_fix = self.xp.where(fix_all)
                            leaf_inds_fix = inds_f0_both_sorted[(temp_inds_fix, walkers_inds_fix, leaf_map_inds_fix)] % nleaves_max

                            bad_check_val = np.unique((temp_inds_fix * 1e12 + walkers_inds_fix * 1e6 + leaf_inds_fix).astype(int))
                               
                            # we are going to make this proposal not accepted
                            # this so far is only ever an accepted-level problem at high temps 
                            # where large frequency jumps can happen
                            check_val = (temp_inds_keep[keep] * 1e12 + walkers_inds_keep[keep] * 1e6 + leaf_inds_keep[keep]).astype(int)
                            fix_keep = self.xp.arange(len(keep))[keep][self.xp.in1d(check_val, bad_check_val)]
                
                            keep[fix_keep] = False

                    """et = time.perf_counter()
                    print("second check", (et - st), group_iter, group_len[list(unique_groups).index(group_iter)])
                    st = time.perf_counter()"""
                    gb_fixed_coords[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])] = new_points[keep]

                    # parameters were run for all ~np.isinf(logp), need to adjust for those not accepted
                    keep_from_before = (keep * (~np.isinf(logp)))[keep_here]
                    try:
                        group_index = data_index[keep_from_before]
                    except IndexError:
                        breakpoint()

                    waveform_kwargs_fill = waveform_kwargs_now.copy()

                    waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

                    """ll_check_d_h_add = self.gb.get_ll(
                        points_add.T, 
                        data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        data_index=data_index, 
                        noise_index=noise_index, 
                        **self.waveform_kwargs
                    )

                    h_h_d_h_add = self.gb.h_h.copy()
                    d_h_d_h_add = self.gb.d_h.copy()

                    ll_check_d_h_remove = self.gb.get_ll(
                        points_remove.T, 
                        data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        data_index=data_index, 
                        noise_index=noise_index, 
                        **self.waveform_kwargs
                    )

                    h_h_d_h_remove = self.gb.h_h.copy()
                    d_h_d_h_remove = self.gb.d_h.copy()"""

                    #tmp = data_minus_template.copy()
                    # remove templates by multiplying by "adding them to" d - h + remove =  = d - (h - remove)
                    """et = time.perf_counter()
                    print("before generate", (et - st), group_iter, group_len[list(unique_groups).index(group_iter)])
                    st = time.perf_counter()"""
                    
                    N_vals_generate = N_vals[group_here][keep]
                    N_vals_generate_in = self.xp.concatenate([
                        N_vals_generate, N_vals_generate
                    ])
                    points_for_generate = self.xp.concatenate([
                        points_remove[keep_from_before],  # factor = +1
                        points_add[keep_from_before]  # factors = -1
                    ])

                    num_add = len(points_remove[keep_from_before])

                    # check number of points for the waveform to be added
                    """f_N_check = points_for_generate[num_add:, 1].get()
                    N_check = get_N(np.full_like(f_N_check, 1e-30), f_N_check, self.waveform_kwargs["T"], self.waveform_kwargs["oversample"])

                    fix_because_of_N = False
                    if np.any(N_check != N_vals_generate_in[num_add:].get()):
                        fix_because_of_N = True
                        inds_fix_here = self.xp.where(self.xp.asarray(N_check) != N_vals_generate_in[num_add:])[0]
                        N_vals_generate_in[inds_fix_here + num_add] = self.xp.asarray(N_check)[inds_fix_here]"""

                    
                    factors_multiply_generate = self.xp.ones(2 * num_add)
                    factors_multiply_generate[num_add:] = -1.0  # second half is adding
                    group_index_add = self.xp.concatenate(
                        [
                            group_index,
                            group_index,
                        ], dtype=self.xp.int32
                    )

                    self.gb.generate_global_template(
                        points_for_generate,
                        group_index_add, 
                        self.mgh.data_list, 
                        N=N_vals_generate_in,
                        data_length=self.data_length, 
                        data_splits=self.mgh.gpu_splits, 
                        factors=factors_multiply_generate,
                        **waveform_kwargs_fill
                    )

                    #waveform_kwargs_fill["use_c_implementation"] = False
                    #self.gb.generate_global_template(points_remove[keep_from_before],
                    #    group_index, tmp.reshape((-1,) + tmp.shape[2:]), **waveform_kwargs_fill
                    #)

                    """self.gb.d_d = self.xp.asarray(-2 * state.log_prob.flatten())
                    ll_check_add = self.gb.get_ll(
                        points_add.T, 
                        data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        data_index=data_index, 
                        noise_index=noise_index, 
                        **self.waveform_kwargs
                    )

                    h_h_add = self.gb.h_h.copy()
                    d_h_add = self.gb.d_h.copy()

                    ll_check_remove = self.gb.get_ll(
                        points_remove.T, 
                        data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                        data_index=data_index, 
                        noise_index=noise_index, 
                        **self.waveform_kwargs
                    )
                    
                    h_h_remove = self.gb.h_h.copy()
                    d_h_remove = self.gb.d_h.copy()
                    breakpoint()"""

                    # add templates by adding to  -(-(d - h) + add) = d - h - add = d - (h + add)
                    #self.mgh.multiply_data(-1.)
                    #self.gb.generate_global_template(points_add[keep_from_before],
                    #    group_index, self.mgh.data_list, data_length=self.data_length, #data_splits=self.mgh.gpu_splits, **waveform_kwargs_fill
                    #)
                    #self.mgh.multiply_data(-1.)
                    """et = time.perf_counter()
                    print("after generate", (et - st), group_iter, group_len[list(unique_groups).index(group_iter)])
                    st = time.perf_counter()"""
                    # update likelihoods

                    # set unaccepted differences to zero
                    accepted_delta_ll = delta_ll * (keep)
                    accepted_delta_lp = (logp - prev_logp)
                    accepted_delta_lp[self.xp.isinf(accepted_delta_lp)] = 0.0
                    logl_change_contribution = np.zeros_like(new_state.log_prob)
                    logp_change_contribution = np.zeros_like(new_state.log_prior)
                    try:
                        in_tuple = (accepted_delta_ll[keep].get(), accepted_delta_lp[keep].get(), temp_inds_keep[keep].get(), walkers_inds_keep[keep].get())
                    except AttributeError:
                        in_tuple = (accepted_delta_ll[keep], accepted_delta_lp[keep], temp_inds_keep[keep], walkers_inds_keep[keep])
                    for i, (dll, dlp, ti, wi) in enumerate(zip(*in_tuple)):
                        logl_change_contribution[ti, wi] += dll
                        logp_change_contribution[ti, wi] += dlp

                    log_prob_tmp += self.xp.asarray(logl_change_contribution)
                    log_prior_tmp += self.xp.asarray(logp_change_contribution)

                    """if fix_because_of_N:
                        new_log_prob_tmp = self.mgh.get_ll().flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
                        old_log_prob_tmp = log_prob_tmp.copy()
                        log_prob_tmp = self.xp.asarray(new_log_prob_tmp)
                        main_gpu = self.xp.cuda.runtime.getDevice()
                        for gpu in self.mgh.gpus:
                            with self.xp.cuda.device.Device(gpu):
                                self.xp.cuda.runtime.deviceSynchronize()
                                mempool = self.xp.get_default_memory_pool()
                                mempool.free_all_blocks()
                                self.xp.cuda.runtime.deviceSynchronize()

                        self.xp.cuda.runtime.setDevice(main_gpu)"""
                        
                    """et = time.perf_counter()
                    print("bookkeeping", (et - st), group_iter, group_len[list(unique_groups).index(group_iter)])
                    st = time.perf_counter()"""

                    """
                    data_minus_template_c = np.concatenate(
                        [
                            tmp.get().reshape(ntemps, nwalkers, 1, self.data_length) for tmp in data_minus_template_in_swap
                        ],
                        axis=2
                    )

                    psd_c = np.concatenate(
                        [
                            tmp.get().reshape(ntemps, nwalkers, 1, self.data_length) for tmp in psd_in_swap
                        ],
                        axis=2
                    )

                    ll_after = (-1/2 * 4 * self.df * np.sum(data_minus_template_c.conj() * data_minus_template_c / psd_c, axis=(2, 3)))

                    if np.abs(log_prob_tmp.get() - ll_after).max() > 1e-6:
                        breakpoint()
                    
                    
                    """

                    #if np.any(np.abs(new_state.log_prob - (check.get() + self.noise_ll))[:3].max() > 1.0):
                    #    breakpoint()

                    #if np.any(np.abs(new_state.log_prob - (check.get() + self.noise_ll))[3:].max() > 100.0):
                    #    breakpoint()
                    """
                    check_logl = model.compute_log_prob_fn(new_state.branches_coords, inds=new_state.branches_inds, branch_supps=new_state.branches_supplimental, supps=new_state.supplimental)
                    sigll = model.log_prob_fn.f.signal_ll.copy()
                    check_logl2 = model.compute_log_prob_fn(state.branches_coords, inds=state.branches_inds, branch_supps=state.branches_supplimental, supps=state.supplimental)
                    breakpoint()
                    """
        # st = time.perf_counter()
        try:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords.get()
            new_state.log_prob[:] = log_prob_tmp.get()
            new_state.log_prior[:] = log_prior_tmp.get()
        except AttributeError:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords
            new_state.log_prob[:] = log_prob_tmp
            new_state.log_prior[:] = log_prior_tmp

        self.mempool.free_all_blocks()

        if self.time % 200 == 0:
            ll_after = self.mgh.get_ll(use_cpu=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
            
            if np.abs(log_prob_tmp.get() - ll_after).max()  > 1e0:
                if np.abs(log_prob_tmp.get() - ll_after).max() > 1e0:
                    breakpoint()
                breakpoint()
                self.mgh.restore_base_injections()

                for name in new_state.branches.keys():
                    if name not in ["gb", "gb_fixed"]:
                        continue
                    new_state_branch = new_state.branches[name]
                    coords_here = new_state_branch.coords[new_state_branch.inds]
                    ntemps, nwalkers, nleaves_max_here, ndim = new_state_branch.shape
                    try:
                        group_index = self.xp.asarray(
                            self.mgh.get_mapped_indices(
                                np.repeat(np.arange(ntemps * nwalkers).reshape(ntemps, nwalkers, 1), nleaves_max, axis=-1)[new_state_branch.inds]
                            ).astype(self.xp.int32)
                        )
                    except IndexError:
                        breakpoint()
                    coords_here_in = self.parameter_transforms.both_transforms(coords_here, xp=np)

                    waveform_kwargs_fill = self.waveform_kwargs.copy()
                    waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

                    if "N" in waveform_kwargs_fill:
                        waveform_kwargs_fill.pop("N")
                        
                    self.mgh.multiply_data(-1.)
                    self.gb.generate_global_template(coords_here_in, group_index, self.mgh.data_list, data_length=self.data_length, data_splits=self.mgh.gpu_splits, batch_size=1000, **waveform_kwargs_fill)
                    self.mgh.multiply_data(-1.)

                ll_after2 = self.mgh.get_ll(use_cpu=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
                new_state.log_prob = ll_after2
                   
            """
            data_minus_template = self.xp.concatenate(
                [
                    tmp.reshape(ntemps, nwalkers, 1, self.data_length) for tmp in data_minus_template_in_swap
                ],
                axis=2
            )
            del data_minus_template_in_swap

            psd = self.xp.concatenate(
                [
                    tmp.reshape(ntemps * nwalkers, 1, self.data_length) for tmp in psd_in_swap
                ],
                axis=1
            )
            del psd_in_swap
            self.mempool.free_all_blocks()

            new_state.supplimental.holder["data_minus_template"] = data_minus_template

            lp_after = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds)
            
            ll_after = (-1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))).get()  # model.compute_log_prob_fn(new_state.branches_coords, inds=new_state.branches_inds, logp=lp_after, supps=new_state.supplimental, branch_supps=new_state.branches_supplimental)
            #check = -1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))
            #check2 = -1/2 * 4 * self.df * self.xp.sum(tmp.conj() * tmp / self.xp.asarray(self.psd), axis=(2, 3))
            #print(np.abs(new_state.log_prob - ll_after[0]).max())

            # if any are even remotely getting to be different, reset all (small change)
            if np.abs(new_state.log_prob - ll_after).max() > 1e-1:
                if np.abs(new_state.log_prob - ll_after).max() > 1e0:
                    self.greater_than_1e0 += 1
                    print("Greater:", self.greater_than_1e0)
                breakpoint()
                fix_here = np.abs(new_state.log_prob - ll_after) > 1e-6
                data_minus_template_old = data_minus_template.copy()
                data_minus_template = self.xp.zeros_like(data_minus_template_old)
                data_minus_template[:] = self.xp.asarray(self.data)[None, None]
                templates = self.xp.zeros_like(data_minus_template).reshape(-1, 2, data_minus_template.shape[-1])
                for name in new_state.branches.keys():
                    if name not in ["gb", "gb_fixed"]:
                        continue
                    new_state_branch = new_state.branches[name]
                    coords_here = new_state_branch.coords[new_state_branch.inds]
                    ntemps, nwalkers, nleaves_max_here, ndim = new_state_branch.shape
                    try:
                        group_index = np.repeat(np.arange(ntemps * nwalkers).reshape(ntemps, nwalkers, 1), nleaves_max, axis=-1)[new_state_branch.inds]
                    except IndexError:
                        breakpoint()
                    coords_here_in = self.parameter_transforms.both_transforms(coords_here, xp=np)

                    self.gb.generate_global_template(coords_here_in, group_index, templates, batch_size=1000, **self.waveform_kwargs)

                data_minus_template -= templates.reshape(ntemps, nwalkers, 2, templates.shape[-1])

                new_like = -1 / 2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / psd, axis=(2, 3)).real.get()
            
                new_like += self.noise_ll
                new_state.log_prob[:] = new_like.reshape(ntemps, nwalkers)

            self.mempool.free_all_blocks()
            data_minus_template_in_swap = [data_minus_template[:,:, 0, :].flatten().copy(), data_minus_template[:,:, 1, :].flatten().copy()]
            del data_minus_template

            psd_in_swap = [psd[:, 0, :].flatten().copy(), psd[:, 1, :].flatten().copy()]
            self.mempool.free_all_blocks()
            del psd
            self.mempool.free_all_blocks()
            """
        
        self.mempool.free_all_blocks()

        # get accepted fraction 
        accepted_check = np.all(np.abs(new_state.branches_coords["gb_fixed"] - state.branches_coords["gb_fixed"]) > 0.0, axis=-1).sum(axis=(1, 2)) / new_state.branches_inds["gb_fixed"].sum(axis=(1,2))

        # manually tell temperatures how real overall acceptance fraction is
        number_of_walkers_for_accepted = np.floor(nwalkers * accepted_check).astype(int)

        accepted_inds = np.tile(np.arange(nwalkers), (ntemps, 1))

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        accepted[accepted_inds < number_of_walkers_for_accepted[:, None]] = True

        if self.temperature_control is not None:
            new_state, accepted = self.temperature_control.temper_comps(new_state, accepted)
        else:
            self.temperature_control.swaps_accepted = np.zeros((ntemps - 1))
        
        if np.any(new_state.log_prob > 1e10):
            breakpoint()

        self.time += 1
        #self.xp.cuda.runtime.deviceSynchronize()
        #et = time.perf_counter()
        #print("end stretch", (et - st))

        true_data_map = np.asarray(new_state.supplimental.holder["overall_inds"].flatten())
        self.mgh.map = true_data_map

        """et = time.perf_counter()
        print("end", (et - st), group_iter, group_len[group_iter])"""
                    
        # breakpoint()
        return new_state, accepted

