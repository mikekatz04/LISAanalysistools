# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import StretchMove
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds
from .gbmultipletryrj import GBMutlipleTryRJ
from .gbgroupstretch import GBGroupStretchMove

from ...diagnostic import inner_product
from eryn.state import State


__all__ = ["GBSpecialStretchMove"]

def searchsorted2d_vec(a,b, xp=None, **kwargs):
    if xp is None:
        xp = np
    m,n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num*xp.arange(a.shape[0])[:,None]
    p = xp.searchsorted( (a+r).ravel(), (b+r).ravel(), **kwargs).reshape(m,-1)
    return p - n*(xp.arange(m)[:,None])
    
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
        data,
        psd,
        fd,
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
        else:
            self.xp = np

        self.num_try = num_try
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.noise_kwargs = noise_kwargs
        self.parameter_transforms = parameter_transforms
        self.psd = psd
        self.psd_func = psd_func
        self.fd = fd
        self.df = (fd[1] - fd[0]).item()
        self.data = data
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
        
        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb_fixed"].shape
        

        f_test = state.branches_coords["gb_fixed"][:, 0, :, 1] / 1e3
        
        f_test[f_test == 0.0] = 1e300
        f_test_sorted = np.asarray(np.sort(f_test,axis=-1))

        inds_f_sorted = np.asarray(np.argsort(f_test,axis=-1))  # np.tile(np.arange(f_test.shape[1]), (f_test.shape[0], 1)).reshape(ntemps, -1)
        groups = []
        group_len = []

        buffer = 0  # 2 ** 8

        dbin = 2 * self.waveform_kwargs["N"] + buffer
        max_iter = 1000
        i = 0
        total = 0
        while i < max_iter and ~np.all(f_test_sorted > 1e100):
            diff_along_f = np.zeros_like(f_test_sorted)
            diff_along_f[:, 1:] = np.diff(f_test_sorted, axis=-1)
            
            switch_check = np.zeros_like(f_test_sorted, dtype=int)
            tmp1 = (((np.cumsum(diff_along_f, axis=-1) / self.df).astype(int) // dbin)) 
            tmp1[(tmp1) % 2 == 1] = 0
            tmp1[:, 1:] = np.diff(tmp1, axis=-1)
            tmp1[tmp1 < 0] = 0
            switch_check[:, 1:] = np.diff(np.cumsum(tmp1, axis=-1), axis=-1)
            switch_check[:, 0] = 2

            inds_switch = np.where((switch_check > 0) & (f_test_sorted < 1e100))
            temp_inds_in = inds_switch[0]
            leaf_inds_in = inds_switch[1]
            
            """for jj in range(ntemps):
                try:
                    print(jj, (np.diff(np.sort(f_test_sorted[jj, leaf_inds_in[temp_inds_in == jj]])) / self.df).min())
                except ValueError:
                    continue
            """
            inds_switch_here = (
                temp_inds_in, 
                leaf_inds_in
            )
            # add the first entry for each array 
            inds_switch = (
                np.tile(temp_inds_in, (nwalkers,)),
                np.repeat(np.arange(nwalkers), len(temp_inds_in)),
                np.tile(leaf_inds_in, (nwalkers,)),
            )

            indexes_out = inds_f_sorted[(inds_switch[0], inds_switch[2])]

            try:
                groups.append((inds_switch[0].get(), inds_switch[1].get(), indexes_out.get()))
            except AttributeError:
                groups.append((inds_switch[0], inds_switch[1], indexes_out))
                
            group_len.append(len(inds_switch[0]))

            #print(f_test_sorted.shape)
            f_test_sorted[inds_switch_here] = 1e300

            sort_inds = np.argsort(f_test_sorted, axis=-1)
            f_test_sorted = np.take_along_axis(f_test_sorted, sort_inds, axis=-1)

            inds_f_sorted = np.take_along_axis(inds_f_sorted, sort_inds, axis=-1)
            total += len(inds_switch_here[0])
            #print(i, len(inds_switch[0]), (f_test_sorted > 1e100).sum(axis=-1))
            #i += 1
            #if i % 10 == 0:
            #    print(i, (f_test_sorted > 1e100).sum() / np.prod(f_test_sorted.shape))

        gb_fixed_coords = self.xp.asarray(new_state.branches_coords["gb_fixed"].copy())

        log_prob_tmp = self.xp.asarray(new_state.log_prob)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)

        data_minus_template = new_state.supplimental.holder["data_minus_template"]

        """et = time.perf_counter()
        print("groups", (et - st))
        st = time.perf_counter()"""
        for group in groups:
            split_inds = np.zeros(nwalkers, dtype=int)
            split_inds[1::2] = 1
            np.random.shuffle(split_inds)
            for split in range(2):
                # st = time.perf_counter()
                
                split_here = split_inds == split
                walkers_keep = self.xp.arange(nwalkers)[split_here]
                temp_inds, walkers_inds, leaf_inds = [self.xp.asarray(grp) for grp in group] 

                keep_inds = self.xp.in1d(walkers_inds, walkers_keep)

                temp_inds_for_gen = temp_inds[~keep_inds]
                walkers_inds_for_gen = walkers_inds[~keep_inds]
                leaf_inds_for_gen = leaf_inds[~keep_inds]

                temp_inds_keep = temp_inds[keep_inds]
                walkers_inds_keep = walkers_inds[keep_inds]
                leaf_inds_keep = leaf_inds[keep_inds]                

                # use new_state here to get change after 1st round
                q = {"gb_fixed": gb_fixed_coords.copy()}

                group_here = (temp_inds_keep, walkers_inds_keep, leaf_inds_keep)
                group_here_for_gen = (temp_inds_for_gen, walkers_inds_for_gen, leaf_inds_for_gen)

                points_to_move = gb_fixed_coords[:, split_here]
                points_for_move = gb_fixed_coords[:, ~split_here]
                
                q_temp, factors_temp = self.get_proposal(
                    {"gb_fixed": points_to_move.transpose(0, 2, 1, 3).reshape(ntemps * nleaves_max, int(nwalkers / 2), 1, ndim)},  
                    {"gb_fixed": [points_for_move.transpose(0, 2, 1, 3).reshape(ntemps * nleaves_max, int(nwalkers / 2), 1, ndim)]}, 
                    model.random
                )

                factors = self.xp.zeros((ntemps, nwalkers, nleaves_max))
                factors[:, split_here] = factors_temp.reshape(ntemps, nleaves_max, int(nwalkers / 2)).transpose(0, 2, 1)

                factors_here = factors[group_here]
                
                q["gb_fixed"][:, split_here] = q_temp["gb_fixed"].reshape(ntemps, nleaves_max, int(nwalkers / 2), ndim).transpose(0, 2, 1, 3)

                old_points = gb_fixed_coords[group_here]
                new_points = q["gb_fixed"][group_here]

                # data should not be whitened
                if "noise_params" not in q:
                    use_stock_psd = True
                    psd = self.xp.tile(self.xp.asarray(self.psd), (ntemps * nwalkers, 1, 1))

                else:
                    use_stock_psd = False
                    
                    noise_params = q["noise_params"]
                    if self.psd_func is None:
                        raise ValueError("When providing noise_params, psd_func kwargs in __init__ function must be given.")

                    if noise_params.ndim == 3:
                        noise_params = noise_params[0]
                    try:
                        tmp = self.xp.asarray([self.psd_func(self.fd, *noise_params.reshape(-1, noise_params.shape[-1]).T, **self.noise_kwargs) for _ in range(2)])
                        psd = tmp.transpose((1,0,2))
                        breakpoint()
                    except ValueError:
                        breakpoint()

                if self.use_gpu:
                    new_points_prior = new_points.get()
                    old_points_prior = old_points.get()
                else:
                    new_points_prior = new_points
                    old_points_prior = old_points

                logp = self.xp.asarray(self.priors["gb_fixed"].logpdf(new_points_prior))
                keep_here = self.xp.where((~self.xp.isinf(logp)))
                
                points_remove = self.parameter_transforms.both_transforms(old_points[keep_here], xp=self.xp)
                points_add = self.parameter_transforms.both_transforms(new_points[keep_here], xp=self.xp)

                data_index = self.xp.asarray((temp_inds_keep[keep_here] * nwalkers + walkers_inds_keep[keep_here]).astype(xp.int32))
                noise_index = self.xp.asarray((temp_inds_keep[keep_here] * nwalkers + walkers_inds_keep[keep_here]).astype(xp.int32))
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
                
                self.waveform_kwargs["start_freq_ind"] = self.start_freq_ind
                
                delta_ll[keep_here] = self.gb.swap_likelihood_difference(points_remove,  points_add,  data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).copy(),  psd.copy(),  data_index=data_index,  noise_index=noise_index,  adjust_inplace=False,  **self.waveform_kwargs)

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
                if self.search:
                    inds_fix = ((optimized_snr < self.search_snr_lim) | (detected_snr < (0.8 * self.search_snr_lim)))

                    if self.xp == np:
                        inds_fix = inds_fix.get()

                    if self.xp.any(inds_fix):
                        logp[keep_here[0][inds_fix]] = -np.inf
                        delta_ll[keep_here[0][inds_fix]] = -1e300

                prev_logl = log_prob_tmp[(temp_inds_keep, walkers_inds_keep)]
                logl = delta_ll + prev_logl

                #if np.any(logl - np.load("noise_ll.npy").flatten() > 0.0):
                #    breakpoint()    
                #print("multi check: ", (logl - np.load("noise_ll.npy").flatten()))

                prev_logp = self.xp.asarray(self.priors["gb_fixed"].logpdf(old_points_prior))
                
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
                        check_f0[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])] = new_points[keep][:, 1]
                        check_f0_old[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])] = old_points[keep][:, 1]

                        check_f0_old_sorted = self.xp.sort(check_f0_old, axis=-1)
                        inds_f0_old_sorted = self.xp.argsort(check_f0_old, axis=-1)

                        check_f0_tmp = check_f0.reshape(-1, check_f0.shape[-1])
                        check_f0_old_sorted_tmp = check_f0_old_sorted.reshape(-1, check_f0_old_sorted.shape[-1])

                        check_f0_in_old_inds = searchsorted2d_vec(check_f0_old_sorted_tmp, check_f0_tmp, xp=self.xp, side="right").reshape(check_f0.shape)

                        zero_check = check_f0_in_old_inds[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])]
                        f0_new_here = new_points[keep][:, 1]
                        keep_for_after = keep.copy()
                        inds_test_here = [-2, -1, 0, 1]
                        for ind_test_here in inds_test_here:
                            here_check = zero_check + ind_test_here
                            do_check = np.ones_like(here_check, dtype=bool)
                            do_check[here_check < 0] = False
                            do_check[here_check >= check_f0_old.shape[-1]] = False

                            here_vals = check_f0_old_sorted[(temp_inds_keep[keep], walkers_inds_keep[keep], here_check)]
                            here_inds = inds_f0_old_sorted[(temp_inds_keep[keep], walkers_inds_keep[keep], here_check)]
                            here_test = (self.xp.abs(f0_new_here - here_vals) / 1e3 / self.df).astype(int)
                            fix_bad2_tmp = self.xp.arange(len(keep))[keep]
                            fix_bad2 = fix_bad2_tmp[(here_test < dbin) & (leaf_inds_keep[keep] != here_inds) & (do_check)]
                            #if len(fix_bad2) > 0:
                            #    print("NEW BAD", len(fix_bad2), len(fix_bad2) / len(keep[keep]))
                            keep_for_after[fix_bad2] = False

                        keep[:] = keep_for_after[:]

                        # TODO: add bad ones to new group?

                        #check_f0_old[group] = f0_old

                        check_f0_sorted = self.xp.sort(check_f0, axis=-1)
                        inds_f0_sorted = self.xp.argsort(check_f0, axis=-1)
                        check_f0_diff = self.xp.zeros_like(check_f0_sorted, dtype=int)
                        check_f0_diff[:, :, 1:] = (self.xp.diff(check_f0_sorted, axis=-1) / 1e3 / self.df).astype(int)

                        bad = (check_f0_diff < dbin) & (check_f0_sorted != 0.0)
                        if self.xp.any(bad):
                            try:
                                bad_inds = self.xp.where(bad)
                                
                                # fix the last entry of bad inds
                                inds_bad = (bad_inds[0], bad_inds[1], inds_f0_sorted[bad])
                                bad_check_val = (inds_bad[0] * 1e10 + inds_bad[1] * 1e5 + inds_bad[2]).astype(int)
                                # we are going to make this proposal not accepted
                                # this so far is only ever an accepted-level problem at high temps 
                                # where large frequency jumps can happen
                                check_val = (temp_inds_keep[keep] * 1e10 + walkers_inds_keep[keep] * 1e5 + leaf_inds_keep[keep]).astype(int)
                                fix_keep = self.xp.arange(len(keep))[keep][self.xp.in1d(check_val, bad_check_val)]
                                keep[fix_keep] = False
                            except:
                                breakpoint()

                    gb_fixed_coords[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])] = new_points[keep]

                    # parameters were run for all ~np.isinf(logp), need to adjust for those not accepted
                    keep_from_before = (keep * (~np.isinf(logp)))[keep_here]
                    try:
                        group_index = data_index[keep_from_before]
                    except IndexError:
                        breakpoint()

                    waveform_kwargs_fill = self.waveform_kwargs.copy()

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
                    
                    self.gb.generate_global_template(points_remove[keep_from_before],
                        group_index, data_minus_template.reshape((-1,) + data_minus_template.shape[2:]), **waveform_kwargs_fill
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
                    data_minus_template *= -1
                    self.gb.generate_global_template(points_add[keep_from_before],
                        group_index, data_minus_template.reshape((-1,) + data_minus_template.shape[2:]), **waveform_kwargs_fill
                    )
                    data_minus_template *= -1
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
    
                    if np.any(new_state.log_prob < -1e100):
                        breakpoint()

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

        try:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords.get()
            new_state.log_prob[:] = log_prob_tmp.get()
            new_state.log_prior[:] = log_prior_tmp.get()
        except AttributeError:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords
            new_state.log_prob[:] = log_prob_tmp
            new_state.log_prior[:] = log_prior_tmp

        if self.time % 1 == 0:
            lp_after = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds)
            ll_after = model.compute_log_prob_fn(new_state.branches_coords, inds=new_state.branches_inds, logp=lp_after, supps=new_state.supplimental, branch_supps=new_state.branches_supplimental)
            #check = -1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))
            #check2 = -1/2 * 4 * self.df * self.xp.sum(tmp.conj() * tmp / self.xp.asarray(self.psd), axis=(2, 3))
            #print(np.abs(new_state.log_prob - ll_after[0]).max())

            if np.abs(new_state.log_prior - lp_after).max() > 0.1 or np.abs(new_state.log_prob - ll_after[0]).max() > 1e0:
                breakpoint()


            # if any are even remotely getting to be different, reset all (small change)
            elif np.abs(new_state.log_prob - ll_after[0]).max() > 1e-3:
                
                fix_here = np.abs(new_state.log_prob - ll_after[0]) > 1e-6
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
                    group_index = np.repeat(np.arange(ntemps * nwalkers).reshape(ntemps, nwalkers, 1), nleaves_max, axis=-1)[new_state_branch.inds]
                    coords_here_in = self.parameter_transforms.both_transforms(coords_here, xp=np)

                    self.gb.generate_global_template(coords_here_in, group_index, templates, batch_size=1000, **self.waveform_kwargs)

                data_minus_template -= templates.reshape(ntemps, nwalkers, 2, templates.shape[-1])

                new_like = -1 / 2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / psd, axis=(2, 3)).real.get()
            
                new_like += self.noise_ll
                new_state.log_prob[:] = new_like.reshape(ntemps, nwalkers)

        # get accepted fraction 
        accepted_check = np.all(np.abs(new_state.branches_coords["gb_fixed"] - state.branches_coords["gb_fixed"]) > 0.0, axis=-1).sum(axis=(1, 2)) / new_state.branches_inds["gb_fixed"].sum(axis=(1,2))

        # manually tell temperatures how real overall acceptance fraction is
        number_of_walkers_for_accepted = np.floor(nwalkers * accepted_check).astype(int)

        accepted_inds = np.tile(np.arange(nwalkers), (ntemps, 1))

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        accepted[accepted_inds < number_of_walkers_for_accepted[:, None]] = True

        if self.temperature_control is not None:
            new_state, accepted = self.temperature_control.temper_comps(new_state, accepted)
        self.temperature_control.swaps_accepted = np.zeros((ntemps - 1))

        if np.any(new_state.log_prob > 1e10):
            breakpoint()

        self.time += 1
        #self.xp.cuda.runtime.deviceSynchronize()
        #et = time.perf_counter()
        #print("end stretch", (et - st))

        return new_state, accepted

