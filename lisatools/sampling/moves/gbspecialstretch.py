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

        if search_snrs is not None:
            if search_snr_lim is None:
                search_snr_lim = 0.1

            assert len(search_samples) == len(search_snrs)

        self.search_samples = search_samples
        self.search_snrs = search_snrs
        self.search_snr_lim = search_snr_lim
        self.search_snr_accept_factor = search_snr_accept_factor

        self.take_max_ll = take_max_ll     

    def run_swap_ll(self, gb_fixed_coords, old_points, new_points, group_here, N_vals, waveform_kwargs_now, factors_here, log_like_tmp, log_prior_tmp, return_at_logl=False):

        temp_inds_keep, walkers_inds_keep, leaf_inds_keep = group_here

        # st = time.perf_counter()
        if self.use_gpu:
            new_points_prior = new_points.get()
            old_points_prior = old_points.get()
        else:
            new_points_prior = new_points
            old_points_prior = old_points

        logp = self.xp.asarray(self.priors["gb_fixed"].logpdf(new_points_prior))
        keep_here = self.xp.where((~self.xp.isinf(logp)))

        if len(keep_here[0]) == 0:
            return
                
        points_remove = self.parameter_transforms.both_transforms(old_points[keep_here], xp=self.xp)
        points_add = self.parameter_transforms.both_transforms(new_points[keep_here], xp=self.xp)

        data_index_tmp = self.xp.asarray((temp_inds_keep[keep_here] * self.nwalkers + walkers_inds_keep[keep_here]).astype(xp.int32))
        noise_index_tmp = self.xp.asarray((temp_inds_keep[keep_here] * self.nwalkers + walkers_inds_keep[keep_here]).astype(xp.int32))
        
        data_index = self.mgh.get_mapped_indices(data_index_tmp).astype(self.xp.int32)
        noise_index = self.mgh.get_mapped_indices(noise_index_tmp).astype(self.xp.int32)

        assert self.xp.all(data_index == noise_index)
        nChannels = 2

        delta_ll = self.xp.full(old_points.shape[0], -1e300)
        
        N_in_group = N_vals[keep_here]
        """et = time.perf_counter()
        print("before like", (et - st), new_points.shape[0])
        st = time.perf_counter()"""

        delta_ll[keep_here] = self.gb.swap_likelihood_difference(points_remove, points_add, self.mgh.data_list,  self.mgh.psd_list,  N=N_in_group, data_index=data_index,  noise_index=noise_index,  adjust_inplace=False,  data_length=self.data_length, 
        data_splits=self.mgh.gpu_splits, phase_marginalize=self.search, **waveform_kwargs_now)
        """et = time.perf_counter()
        print("after like", (et - st), new_points.shape[0])
        st = time.perf_counter()"""
        
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

            """et = time.perf_counter()
            print("after search stuff", (et - st), N_now, group_iter, group_len[group_iter])
            st = time.perf_counter()"""

        prev_logl = log_like_tmp[(temp_inds_keep, walkers_inds_keep)]
        logl = delta_ll + prev_logl

        if return_at_logl:
            return delta_ll

        #if np.any(logl - np.load("noise_ll.npy").flatten() > 0.0):
        #    breakpoint()    
        #print("multi check: ", (logl - np.load("noise_ll.npy").flatten()))

        prev_logp = self.xp.asarray(self.priors["gb_fixed"].logpdf(old_points_prior))

        if np.any(np.isinf(prev_logp)):
            breakpoint()
        
        betas_in = self.xp.asarray(self.temperature_control.betas)[temp_inds_keep]
        logP = self.compute_log_posterior(logl, logp, betas=betas_in)
        
        # TODO: check about prior = - inf
        # takes care of tempering
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp, betas=betas_in)

        # TODO: think about factors in tempering
        lnpdiff = factors_here + logP - prev_logP

        keep = lnpdiff > self.xp.asarray(self.xp.log(self.xp.random.rand(*logP.shape)))
        """et = time.perf_counter()
        print("keep determination", (et - st), new_points.shape[0])
        st = time.perf_counter()"""
        if self.xp.any(keep):
            

            accepted_here = keep.copy()
            
            gb_fixed_coords[(temp_inds_keep[keep], walkers_inds_keep[keep], leaf_inds_keep[keep])] = new_points[keep]

            # parameters were run for all ~np.isinf(logp), need to adjust for those not accepted
            keep_from_before = (keep * (~np.isinf(logp)))[keep_here]
            try:
                group_index = data_index[keep_from_before]
            except IndexError:
                breakpoint()

            waveform_kwargs_fill = waveform_kwargs_now.copy()

            waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

            """et = time.perf_counter()
            print("before generate", (et - st), new_points.shape[0])
            st = time.perf_counter()"""
            
            N_vals_generate = N_vals[keep]
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

            """et = time.perf_counter()
            print("after generate", (et - st), new_points.shape[0])
            st = time.perf_counter()"""
            # update likelihoods

            # set unaccepted differences to zero
            accepted_delta_ll = delta_ll * (keep)
            accepted_delta_lp = (logp - prev_logp)
            accepted_delta_lp[self.xp.isinf(accepted_delta_lp)] = 0.0
            logl_change_contribution = np.zeros_like(log_like_tmp.get())
            logp_change_contribution = np.zeros_like(log_prior_tmp.get())
            try:
                in_tuple = (accepted_delta_ll[keep].get(), accepted_delta_lp[keep].get(), temp_inds_keep[keep].get(), walkers_inds_keep[keep].get())
            except AttributeError:
                in_tuple = (accepted_delta_ll[keep], accepted_delta_lp[keep], temp_inds_keep[keep], walkers_inds_keep[keep])
            for i, (dll, dlp, ti, wi) in enumerate(zip(*in_tuple)):
                logl_change_contribution[ti, wi] += dll
                logp_change_contribution[ti, wi] += dlp

            log_like_tmp[:] += self.xp.asarray(logl_change_contribution)
            log_prior_tmp[:] += self.xp.asarray(logp_change_contribution)

            """et = time.perf_counter()
            print("bookkeeping", (et - st), new_points.shape[0])"""


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

        np.random.seed(10)
        #print("start stretch")
        #st = time.perf_counter()
        # Check that the dimensions are compatible.
        ndim_total = 0
        for branch in state.branches.values():
            ntemps, nwalkers, nleaves_, ndim_ = branch.shape
            ndim_total += ndim_ * nleaves_

        self.nwalkers = nwalkers
        # TODO: deal with more intensive acceptance fractions
        # Run any move-specific setup.
        self.setup(state.branches)

        new_state = State(state, copy=True)
        self.mempool.free_all_blocks()
        
        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        # data should not be whitened
        
        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb_fixed"].shape

        # TODO: add actual amplitude
        N_vals = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"]

        N_vals = self.xp.asarray(N_vals)

        N_vals_2_times = self.xp.concatenate([N_vals, N_vals], axis=-1)

        gb_fixed_coords = self.xp.asarray(new_state.branches_coords["gb_fixed"].copy())

        log_like_tmp = self.xp.asarray(new_state.log_like)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)
      
        self.mempool.free_all_blocks()

        unique_N = np.unique(N_vals)
        
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

            f_test = points_to_move.reshape(ntemps, int(nwalkers / 2), -1, ndim)[:, :, :, 1].get() / 1e3
            f_test_2 = q["gb_fixed"][:, split_here].reshape(ntemps, int(nwalkers / 2), -1, ndim)[:, :, :, 1].get() / 1e3
            if np.any(f_test == 0):
                breakpoint()

            # f0_2 will remove and suggested frequency jumps of more than one band
            # num_groups_base should be three with the way groups are done now
            # no need to do checks now either
            
            # if suggesting to change frequency by more than twice waveform length, do not run
            fix_f_test = (np.abs(f_test - f_test_2) > (self.df * N_vals[:, split_here].get() * 1.5))
            if hasattr(self, "keep_bands") and self.keep_bands is not None:
                band_indices = np.searchsorted(self.band_edges, f_test.flatten()).reshape(f_test.shape) - 1
                keep_bands = self.keep_bands
                assert isinstance(keep_bands, np.ndarray)
                fix_f_test[~np.in1d(band_indices, keep_bands).reshape(band_indices.shape)] = True

            groups = get_groups_from_band_structure(f_test, self.band_edges, f0_2=f_test_2, xp=np, num_groups_base=4, fix_f_test=fix_f_test)

            unique_groups, group_len = np.unique(groups.flatten(), return_counts=True)

            # remove information about the bad "-1" group
            for check_val in [-1, -2]:
                group_len = np.delete(group_len, unique_groups == check_val)
                unique_groups = np.delete(unique_groups, unique_groups == check_val)

            if len(unique_groups) == 0:
                return state, accepted

            # needs to be max because some values may be missing due to evens and odds
            num_groups = unique_groups.max().item() + 1

            """et = time.perf_counter()
            print("prop", (et - st))"""

            for group_iter in range(num_groups):
                # st = time.perf_counter()
                # sometimes you will have an extra odd or even group only
                # the group_iter may not match the actual running group number in this case
                if group_iter not in groups:
                    continue

                group = [grp[:, split_here][groups == group_iter].flatten() for grp in group_temp_finder]

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

                """et = time.perf_counter()
                print("before delta ll", (et - st), group_iter, group_len[group_iter])"""
                self.run_swap_ll(gb_fixed_coords, old_points, new_points, group_here, N_vals[group_here], waveform_kwargs_now, factors_here, log_like_tmp, log_prior_tmp)
               
        # st = time.perf_counter()
        try:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords.get()
            new_state.log_like[:] = log_like_tmp.get()
            new_state.log_prior[:] = log_prior_tmp.get()
        except AttributeError:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords
            new_state.log_like[:] = log_like_tmp
            new_state.log_prior[:] = log_prior_tmp

        self.mempool.free_all_blocks()

        if self.time % 200 == 0:
            ll_after = self.mgh.get_ll(use_cpu=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
            
            if np.abs(log_like_tmp.get() - ll_after).max()  > 1e0:
                if np.abs(log_like_tmp.get() - ll_after).max() > 1e0:
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
                new_state.log_like = ll_after2
                   
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
            
            ll_after = (-1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))).get()  # model.compute_log_like_fn(new_state.branches_coords, inds=new_state.branches_inds, logp=lp_after, supps=new_state.supplimental, branch_supps=new_state.branches_supplimental)
            #check = -1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))
            #check2 = -1/2 * 4 * self.df * self.xp.sum(tmp.conj() * tmp / self.xp.asarray(self.psd), axis=(2, 3))
            #print(np.abs(new_state.log_like - ll_after[0]).max())

            # if any are even remotely getting to be different, reset all (small change)
            if np.abs(new_state.log_like - ll_after).max() > 1e-1:
                if np.abs(new_state.log_like - ll_after).max() > 1e0:
                    self.greater_than_1e0 += 1
                    print("Greater:", self.greater_than_1e0)
                breakpoint()
                fix_here = np.abs(new_state.log_like - ll_after) > 1e-6
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
                new_state.log_like[:] = new_like.reshape(ntemps, nwalkers)

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
            
            """# new_state, accepted = self.temperature_control.temper_comps(new_state, accepted)
            self.swaps_accepted = np.zeros(ntemps - 1)
            self.attempted_swaps = np.zeros(ntemps - 1)
            betas = self.temperature_control.betas
            for i in range(ntemps - 1, 0, -1):
                bi = betas[i]
                bi1 = betas[i - 1]

                dbeta = bi1 - bi

                iperm = np.random.permutation(nwalkers)
                i1perm = np.random.permutation(nwalkers)

                # need to calculate switch likelihoods

                coords_iperm = new_state.branches["gb_fixed"].coords[i, iperm]
                coords_i1perm = new_state.branches["gb_fixed"].coords[i - 1, i1perm]

                N_vals_iperm = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][i, iperm]

                N_vals_i1perm = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][i - 1, i1perm]

                f_test_i = coords_iperm[None, :, :, 1] / 1e3
                f_test_2_i = coords_i1perm[None, :, :, 1] / 1e3
                
                fix_f_test_i = (np.abs(f_test_i - f_test_2_i) > (self.df * N_vals_iperm * 1.5))

                if hasattr(self, "keep_bands") and self.keep_bands is not None:
                    band_indices = np.searchsorted(self.band_edges, f_test_i.flatten()).reshape(f_test_i.shape) - 1
                    keep_bands = self.keep_bands
                    assert isinstance(keep_bands, np.ndarray)
                    fix_f_test_i[~np.in1d(band_indices, keep_bands).reshape(band_indices.shape)] = True


                groups = get_groups_from_band_structure(f_test_i, self.band_edges, f0_2=f_test_2_i, xp=np, num_groups_base=3, fix_f_test=fix_f_test_i)

                unique_groups, group_len = np.unique(groups.flatten(), return_counts=True)

                # remove information about the bad "-1" group
                for check_val in [-1, -2]:
                    group_len = np.delete(group_len, unique_groups == check_val)
                    unique_groups = np.delete(unique_groups, unique_groups == check_val)

                # needs to be max because some values may be missing due to evens and odds
                num_groups = unique_groups.max().item() + 1

                for group_iter in range(num_groups):
                    # st = time.perf_counter()
                    # sometimes you will have an extra odd or even group only
                    # the group_iter may not match the actual running group number in this case
                    if group_iter not in groups:
                        continue
                        
                    group = [grp[i:i+1][groups == group_iter].flatten() for grp in group_temp_finder]

                    # st = time.perf_counter()
                    temp_inds_back, walkers_inds_back, leaf_inds = [self.xp.asarray(grp) for grp in group] 

                    temp_inds_i = temp_inds_back.copy()
                    walkers_inds_i = walkers_inds_back.copy()

                    temp_inds_i1 = temp_inds_back.copy()
                    walkers_inds_i1 = walkers_inds_back.copy()

                    temp_inds_i[:] = i
                    walkers_inds_i[:] = self.xp.asarray(iperm)[walkers_inds_back]

                    temp_inds_i1[:] = i - 1
                    walkers_inds_i1[:] = self.xp.asarray(i1perm)[walkers_inds_back]

                    group_here_i = (temp_inds_i, walkers_inds_i, leaf_inds)

                    group_here_i1 = (temp_inds_i1, walkers_inds_i1, leaf_inds)

                    # factors_here = factors[group_here]
                    old_points = self.xp.asarray(new_state.branches["gb_fixed"].coords)[group_here_i]
                    new_points = self.xp.asarray(new_state.branches["gb_fixed"].coords)[group_here_i1]

                    N_vals_here_i = N_vals[group_here_i]
                    
                    log_like_tmp = self.xp.asarray(new_state.log_like.copy())
                    log_prior_tmp = self.xp.asarray(new_state.log_prior.copy())

                    delta_logl_i = self.run_swap_ll(None, old_points, new_points, group_here_i, N_vals_here_i, waveform_kwargs_now, None, log_like_tmp, log_prior_tmp, return_at_logl=True)

                    # factors_here = factors[group_here]
                    old_points[:] = self.xp.asarray(new_state.branches["gb_fixed"].coords)[group_here_i1]
                    new_points[:] = self.xp.asarray(new_state.branches["gb_fixed"].coords)[group_here_i]

                    N_vals_here_i1 = N_vals[group_here_i1]
                    
                    log_like_tmp[:] = self.xp.asarray(new_state.log_like.copy())
                    log_prior_tmp[:] = self.xp.asarray(new_state.log_prior.copy())

                    delta_logl_i1 = self.run_swap_ll(None, old_points, new_points, group_here_i1, N_vals_here_i1, waveform_kwargs_now, None, log_like_tmp, log_prior_tmp, return_at_logl=True)

                    paccept = dbeta * 1. / 2. * (delta_logl_i - delta_logl_i1)
                    raccept = np.log(np.random.uniform(size=paccept.shape[0]))

                    # How many swaps were accepted?
                    sel = paccept > self.xp.asarray(raccept)

                    inds_i_swap = tuple([tmp[sel].get() for tmp in list(group_here_i)])
                    inds_i1_swap = tuple([tmp[sel].get() for tmp in list(group_here_i1)])
                    
                    group_index_i = self.xp.asarray(
                        self.mgh.get_mapped_indices(
                            temp_inds_i[sel] + nwalkers * walkers_inds_i[sel]
                        )
                    ).astype(self.xp.int32)

                    group_index_i1 = self.xp.asarray(
                        self.mgh.get_mapped_indices(
                            temp_inds_i1[sel] + nwalkers * walkers_inds_i1[sel]
                        )
                    ).astype(self.xp.int32)

                    N_vals_i = N_vals[inds_i_swap]
                    params_i = self.xp.asarray(new_state.branches["gb_fixed"].coords)[inds_i_swap]
                    params_i1 = self.xp.asarray(new_state.branches["gb_fixed"].coords)[inds_i1_swap]

                    params_generate = self.xp.concatenate([
                        params_i,
                        params_i1,
                        params_i1,  # reverse of above
                        params_i,
                    ], axis=0)

                    params_generate_in = self.parameter_transforms.both_transforms(params_generate, xp=self.xp)

                    group_index_gen = self.xp.concatenate(
                        [
                            group_index_i,
                            group_index_i,
                            group_index_i1,
                            group_index_i1
                        ], dtype=self.xp.int32
                    )

                    factors_multiply_generate = self.xp.concatenate([
                        +1 * self.xp.ones_like(group_index_i, dtype=float),
                        -1 * self.xp.ones_like(group_index_i, dtype=float),
                        +1 * self.xp.ones_like(group_index_i, dtype=float),
                        -1 * self.xp.ones_like(group_index_i, dtype=float),
                    ])

                    N_vals_in_gen = self.xp.concatenate([
                        N_vals_i,
                        N_vals_i,
                        N_vals_i,
                        N_vals_i
                    ])
                    
                    waveform_kwargs_fill = waveform_kwargs_now.copy()
                    waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

                    self.gb.generate_global_template(
                        params_generate_in,
                        group_index_gen, 
                        self.mgh.data_list, 
                        N=N_vals_in_gen,
                        data_length=self.data_length, 
                        data_splits=self.mgh.gpu_splits, 
                        factors=factors_multiply_generate,
                        **waveform_kwargs_fill
                    )

                    # update likelihoods

                    # set unaccepted differences to zero
                    accepted_delta_ll_i = delta_logl_i * (sel)
                    accepted_delta_ll_i1 = delta_logl_i1 * (sel)

                    logl_change_contribution = np.zeros_like(log_like_tmp.get())
                    try:
                        in_tuple = (accepted_delta_ll_i[sel].get(), accepted_delta_ll_i1[sel].get(), temp_inds_i[sel].get(), temp_inds_i1[sel].get(), walkers_inds_i[sel].get(), walkers_inds_i[sel].get())
                    except AttributeError:
                        in_tuple = (accepted_delta_ll_i[sel], accepted_delta_ll_i1[sel], temp_inds_i[sel], temp_inds_i1[sel], walkers_inds_i[sel], walkers_inds_i[sel])
                    for j, (dlli, dlli1, ti, ti1, wi, wi1) in enumerate(zip(*in_tuple)):
                        logl_change_contribution[ti, wi] += dlli
                        logl_change_contribution[ti1, wi1] += dlli1

                    log_like_tmp[:] += self.xp.asarray(logl_change_contribution)

                    tmp_swap = new_state.branches["gb_fixed"].coords[inds_i_swap]
                    new_state.branches["gb_fixed"].coords[inds_i_swap] = new_state.branches["gb_fixed"].coords[inds_i1_swap]

                    new_state.branches["gb_fixed"].coords[inds_i1_swap] = tmp_swap

                    tmp_swap = new_state.branches["gb_fixed"].branch_supplimental[inds_i_swap]

                    new_state.branches["gb_fixed"].branch_supplimental[inds_i_swap] = new_state.branches["gb_fixed"].branch_supplimental[inds_i1_swap]

                    new_state.branches["gb_fixed"].branch_supplimental[inds_i1_swap] = tmp_swap

                    # inds are all non-zero
                    self.swaps_accepted[i - 1] += np.sum(sel)
                    self.attempted_swaps[i - 1] += sel.shape[0]

                    ll_after = self.mgh.get_ll(use_cpu=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
                    breakpoint()
                    """
        else:
            self.temperature_control.swaps_accepted = np.zeros((ntemps - 1))
        
        
        if np.any(new_state.log_like > 1e10):
            breakpoint()

        self.time += 1
        #self.xp.cuda.runtime.deviceSynchronize()
        #et = time.perf_counter()
        #print("end stretch", (et - st))

        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        """et = time.perf_counter()
        print("end", (et - st), group_iter, group_len[group_iter])"""
                    
        # breakpoint()
        return new_state, accepted

