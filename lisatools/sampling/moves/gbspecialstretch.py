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
from eryn.prior import ProbDistContainer
from eryn.utils.utility import groups_from_inds

from eryn.moves import GroupStretchMove

from ...diagnostic import inner_product
from eryn.state import State


__all__ = ["GBSpecialStretchMove"]

# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialStretchMove(GroupStretchMove):
    """Generate Revesible-Jump proposals for GBs with try-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`ProbDistContainer` object that has ``logpdf``
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
        gpu_priors,
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
        GroupStretchMove.__init__(self, *args, return_gpu=True, **kwargs)

        self.gpu_priors = gpu_priors

        self.greater_than_1e0 = 0
        self.name = "gbgroupstretch"

        # TODO: make priors optional like special generate function? 
        for key in priors:
            if not isinstance(priors[key], ProbDistContainer):
                raise ValueError("Priors need to be eryn.priors.ProbDistContainer object.")
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

        self.band_edges = self.xp.asarray(self.band_edges)
        self.take_max_ll = take_max_ll   
 
    def setup_gbs(self, branch):
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        all_remaining_freqs = [coords[i][inds[i]][:, 1] for i in range(coords.shape[0])]

        all_remaining_cords = [coords[i][inds[i]] for i in range(coords.shape[0])]
        
        num_remaining = [len(tmp) for tmp in all_remaining_freqs]
        

        # TODO: improve this?
        self.inds_freqs_sorted = [self.xp.asarray(np.argsort(freqs)) for freqs in all_remaining_freqs]
        self.freqs_sorted = [self.xp.asarray(np.sort(freqs)) for freqs in all_remaining_freqs]
        self.all_coords_sorted = [self.xp.asarray(coords)[sort] for coords, sort in zip(all_remaining_cords, self.inds_freqs_sorted)]
        start_inds_freq_out = np.zeros((ntemps, nwalkers, nleaves_max), dtype=int)
        for t in range(ntemps):
            freqs_sorted_here = self.freqs_sorted[t].get()
            freqs_remaining_here = all_remaining_freqs[t]

            start_ind_best = np.zeros_like(freqs_remaining_here, dtype=int)

            best_index = np.searchsorted(freqs_sorted_here, freqs_remaining_here, side="right") - 1
            best_index[best_index < self.nfriends] = self.nfriends
            best_index[best_index >= len(freqs_sorted_here) - self.nfriends] = len(freqs_sorted_here) - self.nfriends
            check_inds = best_index[:, None] + np.tile(np.arange(2 * self.nfriends), (best_index.shape[0], 1)) - self.nfriends

            check_freqs = freqs_sorted_here[check_inds]
            freq_distance = np.abs(freqs_remaining_here[:, None] - check_freqs)

            keep_min_inds = np.argsort(freq_distance, axis=-1)[:, :self.nfriends].min(axis=-1)
            start_inds_freq = check_inds[(np.arange(len(check_inds)), keep_min_inds)]
            
            start_inds_freq_out[t, inds[t]] = start_inds_freq
        
        if "friend_start_inds" not in supps:
            supps.add_objects({"friend_start_inds": start_inds_freq_out})
        else:
            supps[:] = {"friend_start_inds": start_inds_freq_out}

        self.all_friends_start_inds_sorted = [self.xp.asarray(start_inds_freq_out[t, inds[t]])[sort] for t, sort in enumerate(self.inds_freqs_sorted)]
        
        
    def find_friends(self, name, gb_points_to_move, s_inds=None):
    
        if s_inds is None:
            raise ValueError

        inds_points_to_move = self.xp.asarray(s_inds)

        half_friends = int(self.nfriends / 2)

        gb_points_for_move = gb_points_to_move.copy()

        output_friends = []
        for t in range(self.ntemps):
            freqs_to_move = gb_points_to_move[t][inds_points_to_move[t]][:, 1]
            # freqs_sorted_here = self.freqs_sorted[t]
            # inds_freqs_sorted_here = self.inds_freqs_sorted[t]
            inds_start_freq_to_move = self.current_friends_start_inds[t, inds_points_to_move[t].reshape(self.nwalkers, -1)]
 
            deviation = self.xp.random.randint(0, self.nfriends, size=len(inds_start_freq_to_move))

            inds_keep_friends = inds_start_freq_to_move + deviation

            inds_keep_friends[inds_keep_friends < 0] = 0
            inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted[t])] = len(self.all_coords_sorted[t]) - 1
            
            gb_points_for_move[t, inds_points_to_move[t]]  = self.all_coords_sorted[t][inds_keep_friends]

        return gb_points_for_move

    def setup(self, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name != "gb_fixed":
                continue
            
            if self.time % self.n_iter_update == 0:
                self.setup_gbs(branch)

            self.current_friends_start_inds = self.xp.asarray(branch.branch_supplimental.holder["friend_start_inds"][:])

    def run_ll_part_comp(self, data_index, noise_index, start_inds, lengths):
        assert self.xp.all(data_index == noise_index)
        lnL = self.xp.zeros_like(data_index, dtype=self.xp.float64)
        main_gpu = self.xp.cuda.runtime.getDevice()
        keep_gpu_out = []
        lnL_out = []
        for gpu_i, (gpu, inds_gpu_split) in enumerate(zip(self.mgh.gpus, self.mgh.gpu_splits)):
            with xp.cuda.device.Device(gpu):
                self.xp.cuda.runtime.deviceSynchronize()
                
                min_ind_split, max_ind_split = inds_gpu_split.min(), inds_gpu_split.max()

                keep_gpu_here = self.xp.where((data_index >= min_ind_split) & (data_index <= max_ind_split))[0]
                keep_gpu_out.append(keep_gpu_here)
                num_gpu_here = len(keep_gpu_here)
                lnL_here = self.xp.zeros(num_gpu_here, dtype=self.xp.float64)
                lnL_out.append(lnL_here)
                data_A = self.mgh.data_list[0][gpu_i]
                data_E = self.mgh.data_list[1][gpu_i]
                psd_A = self.mgh.psd_list[0][gpu_i]
                psd_E = self.mgh.psd_list[1][gpu_i]

                data_index_here = data_index[keep_gpu_here].astype(self.xp.int32) - min_ind_split
                noise_index_here = noise_index[keep_gpu_here].astype(self.xp.int32) - min_ind_split

                start_inds_here = start_inds[keep_gpu_here]
                lengths_here = lengths[keep_gpu_here]
                do_synchronize = False
                self.gb.specialty_piece_wise_likelihoods(
                    lnL_here,
                    data_A,
                    data_E,
                    psd_A,
                    psd_E,
                    data_index_here,
                    noise_index_here,
                    start_inds_here,
                    lengths_here,
                    self.df, 
                    num_gpu_here,
                    self.start_freq_ind,
                    self.data_length,
                    do_synchronize,
                )

        for gpu_i, (gpu, inds_gpu_split) in enumerate(zip(self.mgh.gpus, self.mgh.gpu_splits)):
            with xp.cuda.device.Device(gpu):
                self.xp.cuda.runtime.deviceSynchronize()
            with xp.cuda.device.Device(main_gpu):
                self.xp.cuda.runtime.deviceSynchronize()
                lnL[keep_gpu_out[gpu_i]] = lnL_out[gpu_i]

        self.xp.cuda.runtime.setDevice(main_gpu)
        self.xp.cuda.runtime.deviceSynchronize()
        return lnL

    def run_swap_ll(self, num_stack, gb_fixed_coords_new, gb_fixed_coords_old, group_here, N_vals_all, waveform_kwargs_now, factors, log_like_tmp, log_prior_tmp, zz_sampled, return_at_logl=False):
        self.xp.cuda.runtime.deviceSynchronize()
        # st = time.perf_counter()
        temp_inds_keep, walkers_inds_keep, leaf_inds_keep = group_here

        ntemps, nwalkers, nleaves_max, ndim = gb_fixed_coords_new.shape

        new_points = gb_fixed_coords_new[group_here]
        old_points = gb_fixed_coords_old[group_here]
        N_vals = N_vals_all[group_here]
        self.xp.cuda.runtime.deviceSynchronize()
        # st = time.perf_counter()
        new_points_prior = new_points
        old_points_prior = old_points

        # st = time.perf_counter()
        logp = self.xp.asarray(self.gpu_priors["gb_fixed"].logpdf(new_points_prior))
        prev_logp = self.xp.asarray(self.gpu_priors["gb_fixed"].logpdf(old_points_prior))
        """et = time.perf_counter()
        print("prior", et - st)"""
        keep_here = self.xp.where((~self.xp.isinf(logp)))

        ## log likelihood between the points
        if len(keep_here[0]) == 0:
            return

        old_freqs = old_points[:, 1] / 1e3
        new_freqs = new_points[:, 1] / 1e3

        band_indices = self.xp.searchsorted(self.xp.asarray(self.band_edges), old_freqs) - 1
        self.xp.cuda.runtime.deviceSynchronize()
        buffer = 5  # bins

        special_band_inds = int(1e12) * temp_inds_keep + int(1e6) * walkers_inds_keep + band_indices
        special_band_inds_sorted = self.xp.sort(special_band_inds)
        inds_special_band_inds_sorted = self.xp.argsort(special_band_inds)
        self.xp.cuda.runtime.deviceSynchronize()
        unique_special_band_inds, start_special_band_inds, inverse_special_band_inds, counts_special_band_inds = self.xp.unique(special_band_inds_sorted, return_index=True, return_inverse=True, return_counts=True)

        inds_special_map = self.xp.arange(special_band_inds.shape[0])
        inds_special_map_sorted = inds_special_map - inds_special_map[start_special_band_inds][inverse_special_band_inds]
        self.xp.cuda.runtime.deviceSynchronize()
        which_stack_piece = self.xp.zeros_like(inds_special_band_inds_sorted)
        
        which_stack_piece[inds_special_band_inds_sorted] = inds_special_map_sorted

        start_wave_inds_old = -self.xp.ones((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=int)
        end_wave_inds_old = -self.xp.ones((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=int)

        stack_track = self.xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=bool)

        stack_track[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = True

        prior_all_new = self.xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1, num_stack))
        prior_all_new[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = logp
        self.xp.cuda.runtime.deviceSynchronize()
        prior_all_old = self.xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1, num_stack))
        prior_all_old[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = prev_logp

        prior_per_group_new = prior_all_new.sum(axis=-1)
        prior_per_group_old = prior_all_old.sum(axis=-1)

        freqs_diff_arranged = self.xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1, num_stack))
        freqs_diff_arranged[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = (self.xp.abs(new_freqs - old_freqs) / self.df).astype(int) / N_vals

        freqs_diff_arranged_maxs = freqs_diff_arranged.max(axis=-1)
        
        start_wave_inds_old[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = ((old_freqs - ((N_vals / 2).astype(int) + buffer) * self.df) / self.df).astype(int)
        end_wave_inds_old[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = ((old_freqs + ((N_vals / 2).astype(int) + buffer) * self.df) / self.df).astype(int)
        self.xp.cuda.runtime.deviceSynchronize()
        start_wave_inds_new = -self.xp.ones((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=int)
        end_wave_inds_new = -self.xp.ones((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=int)

        leaf_inds_new = -self.xp.ones((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=int)
        leaf_inds_new[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = leaf_inds_keep
        temp_inds_new = -self.xp.ones((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=int)
        temp_inds_new[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = temp_inds_keep
        walkers_inds_new = -self.xp.ones((ntemps, nwalkers, len(self.band_edges) - 1, num_stack), dtype=int)
        walkers_inds_new[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = walkers_inds_keep
        self.xp.cuda.runtime.deviceSynchronize()

        start_wave_inds_new[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = ((new_freqs - ((N_vals / 2).astype(int) + buffer) * self.df) / self.df).astype(int)
        end_wave_inds_new[(temp_inds_keep, walkers_inds_keep, band_indices, which_stack_piece)] = ((new_freqs + ((N_vals / 2).astype(int) + buffer) * self.df) / self.df).astype(int)

        start_wave_inds_all = self.xp.concatenate([start_wave_inds_old, start_wave_inds_new], axis=-1)
        end_wave_inds_all = self.xp.concatenate([end_wave_inds_old, end_wave_inds_new], axis=-1)
        
        start_wave_inds_all[start_wave_inds_all == -1] = int(1e15)

        start_going_in = start_wave_inds_all.min(axis=-1)
        end_going_in = end_wave_inds_all.max(axis=-1)

        """tmp_info = []
        for t in range(ntemps):
            for w in range(nwalkers):
                try:
                    tmp_info.append(self.xp.diff(self.xp.where(start_going_in[t, w] < 1000000)[0]).min())
                except ValueError:
                    continue

        if self.xp.any(self.xp.asarray(tmp_info) < 4):
            breakpoint()"""

        group_part = self.xp.where((end_going_in != -1) & (~self.xp.isinf(prior_per_group_new)) & (freqs_diff_arranged_maxs < 1.0))
        
        temp_part, walker_part, sub_band_part = group_part
        leaf_part_bin = leaf_inds_new[group_part]
        keep2 = leaf_part_bin != -1
        self.xp.cuda.runtime.deviceSynchronize()

        temp_part_bin = temp_inds_new[group_part][keep2]
        walkers_part_bin = walkers_inds_new[group_part][keep2]
        leaf_part_bin = leaf_part_bin[keep2]

        group_part_bin = (temp_part_bin, walkers_part_bin, leaf_part_bin)
        
        num_parts = len(temp_part)

        old_points_in = gb_fixed_coords_old[group_part_bin]
        new_points_in = gb_fixed_coords_new[group_part_bin]

        start_inds_final = start_going_in[group_part].astype(self.xp.int32)
        ends_inds_final = end_going_in[group_part].astype(self.xp.int32)
        lengths = ((ends_inds_final + 1) - start_inds_final).astype(self.xp.int32)
                
        points_remove = self.parameter_transforms.both_transforms(old_points_in, xp=self.xp)
        points_add = self.parameter_transforms.both_transforms(new_points_in, xp=self.xp)


        self.xp.cuda.runtime.deviceSynchronize()

        special_band_inds_2 = int(1e12) * temp_part + int(1e6) * walker_part + sub_band_part
        unique_special_band_inds_2, index_special_band_inds_2 = self.xp.unique(special_band_inds_2, return_index=True)

        temp_part_general = temp_part[index_special_band_inds_2]
        walker_part_general = walker_part[index_special_band_inds_2]

        data_index_tmp = self.xp.asarray((temp_part_general * self.nwalkers + walker_part_general).astype(xp.int32))
        noise_index_tmp = self.xp.asarray((temp_part_general * self.nwalkers + walker_part_general).astype(xp.int32))
        
        data_index = self.mgh.get_mapped_indices(data_index_tmp).astype(self.xp.int32)
        noise_index = self.mgh.get_mapped_indices(noise_index_tmp).astype(self.xp.int32)


        self.xp.cuda.runtime.deviceSynchronize()

        """et = time.perf_counter()
        print("before ll", (et - st))
        st = time.perf_counter()"""
        lnL_old = self.run_ll_part_comp(data_index, noise_index, start_inds_final, lengths)
        self.xp.cuda.runtime.deviceSynchronize()
        """et = time.perf_counter()
        print("after ll", (et - st))
        st = time.perf_counter()"""
        N_vals_single = N_vals_all[group_part_bin]
        
        N_vals_generate_in = self.xp.concatenate([
            N_vals_single, N_vals_single
        ])
        points_for_generate = self.xp.concatenate([
            points_remove,  # factor = +1
            points_add  # factors = -1
        ])

        num_add = len(points_remove)
            
        factors_multiply_generate = self.xp.ones(2 * num_add)
        factors_multiply_generate[num_add:] = -1.0  # second half is adding

        group_index_tmp = self.xp.asarray((temp_part_bin * self.nwalkers + walkers_part_bin).astype(xp.int32))
        group_index = self.mgh.get_mapped_indices(group_index_tmp).astype(self.xp.int32)
        group_index_add = self.xp.concatenate(
            [
                group_index,
                group_index,
            ], dtype=self.xp.int32
        )
        self.xp.cuda.runtime.deviceSynchronize()
        waveform_kwargs_fill = waveform_kwargs_now.copy()

        waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

        """et = time.perf_counter()
        print("before 1 adjustment", (et - st))
        st = time.perf_counter()"""
        self.xp.cuda.runtime.deviceSynchronize()
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
        self.xp.cuda.runtime.deviceSynchronize()
        """et = time.perf_counter()
        print("after 1 adjustment", (et - st))
        st = time.perf_counter()"""
        lnL_new = self.run_ll_part_comp(data_index, noise_index, start_inds_final, lengths)
        self.xp.cuda.runtime.deviceSynchronize()
        """et = time.perf_counter()
        print("after second ll", (et - st))
        st = time.perf_counter()"""
        delta_ll = lnL_new - lnL_old

        logp = prior_per_group_new[(temp_part, walker_part, sub_band_part)]
        prev_logp = prior_per_group_old[(temp_part, walker_part, sub_band_part)]

        prev_logl = log_like_tmp[(temp_part, walker_part)]
        logl = delta_ll + prev_logl

        if return_at_logl:
            return delta_ll
        self.xp.cuda.runtime.deviceSynchronize()
        #if np.any(logl - np.load("noise_ll.npy").flatten() > 0.0):
        #    breakpoint()    
        #print("multi check: ", (logl - np.load("noise_ll.npy").flatten()))

        betas_in = self.xp.asarray(self.temperature_control.betas)[temp_part]
        logP = self.compute_log_posterior(logl, logp, betas=betas_in)
        
        # TODO: check about prior = - inf
        # takes care of tempering
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp, betas=betas_in)

        stack_track_here = stack_track[group_part]
        zz_here = zz_sampled[(temp_part, walker_part)]
        ndim_here_tmp = stack_track_here.sum(axis=-1) * 8
        factors_here = (ndim_here_tmp - 1.0) * self.xp.log(zz_here)
        
        # TODO: think about factors in tempering
        # TODO: adjust scale factor to band?
        lnpdiff = factors_here + logP - prev_logP
        self.xp.cuda.runtime.deviceSynchronize()
        keep = lnpdiff > self.xp.asarray(self.xp.log(self.xp.random.rand(*logP.shape)))

        """for i in range(1, num_stack + 1):
            print(i, keep[ndim_here_tmp / 8 == i].shape[0], keep[ndim_here_tmp / 8 == i].sum() / keep[ndim_here_tmp / 8 == i].shape[0])
        print("\n")"""

        """et = time.perf_counter()
        print("keep determination", (et - st), new_points.shape[0])
        st = time.perf_counter()"""
            
        accepted_here = keep.copy()

        temp_part_accepted = temp_inds_new[group_part][keep]
        walkers_part_accepted = walkers_inds_new[group_part][keep]
        leaf_part_accepted = leaf_inds_new[group_part][keep]
        sub_band_part_accepted = sub_band_part[keep]

        temp_part_accepted = temp_part_accepted[leaf_part_accepted != -1]
        walkers_part_accepted = walkers_part_accepted[leaf_part_accepted != -1]
        leaf_part_accepted = leaf_part_accepted[leaf_part_accepted != -1]

        accepted_mapping = self.xp.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)
        accepted_mapping[(temp_part_accepted, walkers_part_accepted, leaf_part_accepted)] = True

        keep_binaries = accepted_mapping[group_part_bin]

        gb_fixed_coords_old[(temp_part_accepted, walkers_part_accepted, leaf_part_accepted)] = new_points_in[keep_binaries]

        N_vals_single_reverse = N_vals_single[~keep_binaries]

        N_vals_reverse = self.xp.concatenate([
            N_vals_single_reverse, N_vals_single_reverse
        ])

        # reverse the factors to put back what we took out
        points_for_reverse = self.xp.concatenate([
            points_remove[~keep_binaries],  # factor = +1
            points_add[~keep_binaries]  # factors = -1
        ])

        num_reverse = len(points_remove[~keep_binaries])
            
        factors_multiply_reverse = self.xp.ones(2 * num_reverse)
        factors_multiply_reverse[:num_reverse] = -1.0  # second half is adding


        # check number of points for the waveform to be added
        """f_N_check = points_for_generate[num_add:, 1].get()
        N_check = get_N(np.full_like(f_N_check, 1e-30), f_N_check, self.waveform_kwargs["T"], self.waveform_kwargs["oversample"])

        fix_because_of_N = False
        if np.any(N_check != N_vals_generate_in[num_add:].get()):
            fix_because_of_N = True
            inds_fix_here = self.xp.where(self.xp.asarray(N_check) != N_vals_generate_in[num_add:])[0]
            N_vals_generate_in[inds_fix_here + num_add] = self.xp.asarray(N_check)[inds_fix_here]"""
        self.xp.cuda.runtime.deviceSynchronize()
        # group_index is already mapped
        group_index_reverse = self.xp.concatenate(
            [
                group_index[~keep_binaries],
                group_index[~keep_binaries],
            ], dtype=self.xp.int32
        )
        self.xp.cuda.runtime.deviceSynchronize()
        self.gb.generate_global_template(
            points_for_reverse,
            group_index_reverse, 
            self.mgh.data_list, 
            N=N_vals_reverse,
            data_length=self.data_length, 
            data_splits=self.mgh.gpu_splits, 
            factors=factors_multiply_reverse,
            **waveform_kwargs_fill
        )
        self.xp.cuda.runtime.deviceSynchronize()
        """et = time.perf_counter()
        print("after reverse", (et - st))
        st = time.perf_counter()"""
        delta_ll_shaped = self.xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1))
        delta_lp_shaped = self.xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1))
        
        delta_ll_shaped[(temp_part[keep], walker_part[keep], sub_band_part[keep])] = delta_ll[keep]

        delta_lp_shaped[(temp_part[keep], walker_part[keep], sub_band_part[keep])] = (logp - prev_logp)[keep]

        log_like_tmp_check = log_like_tmp.copy()
        log_like_tmp[:] += delta_ll_shaped.sum(axis=-1)
        log_prior_tmp[:] += delta_lp_shaped.sum(axis=-1)
        """et = time.perf_counter()
        print("bookkeeping", (et - st))"""

        """ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[self.current_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)

        if np.abs(log_like_tmp.get() - ll_after).max()  > 1e-5:
            breakpoint()"""


    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        self.xp.cuda.runtime.setDevice(self.mgh.gpus[0])
    
        self.current_state = state
        np.random.seed(10)
        #print("start stretch")
        st = time.perf_counter()
        # Check that the dimensions are compatible.
        ndim_total = 0
        for branch in state.branches.values():
            ntemps, nwalkers, nleaves_, ndim_ = branch.shape
            ndim_total += ndim_ * nleaves_

        #ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
        #if np.abs(state.log_like - ll_after).max()  > 1e-5:
        #    breakpoint()

        self.nwalkers = nwalkers
        # TODO: deal with more intensive acceptance fractions
        # Run any move-specific setup.
        self.setup(state.branches)
        
        # for _ in range(20):

        # st = time.perf_counter()

        new_state = State(state, copy=True)
        self.mempool.free_all_blocks()
        
        # for tmp_run in range(100):
        # st = time.perf_counter()
        """dbeta = new_state.betas[1] - new_state.betas[0]
        x = new_state.branches_coords
        logl = new_state.log_like
        logp = new_state.log_prior
        inds = new_state.branches_inds
        supps = new_state.supplimental
        branch_supps = new_state.branches_supplimental
        logP = new_state.betas[:, None] * logl + logp

        if tmp_run ==0:
            (x, logP, logl, logp, inds, blobs, supps, branch_supps) = self.temperature_control.do_swaps_indexing(1, np.array([0]), np.array([0]), dbeta, x, logP, logl, logp, inds=inds, blobs=None, supps=supps, branch_supps=branch_supps)

            # (x, logP, logl, logp, inds, blobs, supps, branch_supps) = self.temperature_control.do_swaps_indexing(1, np.array([1, 0]), np.array([0, 1]), dbeta, x, logP, logl, logp, inds=inds, blobs=None, supps=supps, branch_supps=branch_supps)

            #breakpoint()

            new_state.branches["gb_fixed"].coords = x["gb_fixed"]
            new_state.log_like = logl
            new_state.log_prior = logp
            new_state.branches["gb_fixed"].inds = inds["gb_fixed"]
            new_state.supplimental = supps
            new_state.branches["gb_fixed"].supplimental = branch_supps["gb_fixed"]
        """
        
        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        # data should not be whitened
        
        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb_fixed"].shape

        # TODO: add actual amplitude
        N_vals = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"]

        N_vals = self.xp.asarray(N_vals)

        gb_fixed_coords = self.xp.asarray(new_state.branches_coords["gb_fixed"].copy())

        log_like_tmp = self.xp.asarray(new_state.log_like)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)
    
        #ll_before = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
        #print("before", np.abs(log_like_tmp.get() - ll_before).max())

        self.mempool.free_all_blocks()

        unique_N = self.xp.unique(N_vals)
        
        waveform_kwargs_now = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs_now:
            waveform_kwargs_now.pop("N")
        waveform_kwargs_now["start_freq_ind"] = self.start_freq_ind

        gb_fixed_coords_into_proposal = gb_fixed_coords.reshape(ntemps, nwalkers * nleaves_max, 1, ndim)
        gb_inds_into_proposal = self.xp.asarray(state.branches["gb_fixed"].inds).reshape(ntemps, nwalkers * nleaves_max, 1)
        """et = time.perf_counter()
        print("before prop", (et - st))
        st = time.perf_counter()"""
    
        # TODO: check detailed balance
        q, factors_temp = self.get_proposal(
                {"gb_fixed": gb_fixed_coords_into_proposal}, model.random, s_inds_all={"gb_fixed": gb_inds_into_proposal}, xp=self.xp, return_gpu=True
        )
        
        gb_fixed_coords_into_proposal = gb_fixed_coords_into_proposal.reshape(ntemps, nwalkers, nleaves_max, ndim)
        
        q["gb_fixed"] = q["gb_fixed"].reshape(ntemps, nwalkers, nleaves_max, ndim)

        gb_inds = gb_inds_into_proposal.reshape(ntemps, nwalkers, nleaves_max)
        remove = np.abs((gb_fixed_coords_into_proposal[:, :, :, 1] - q["gb_fixed"][:, :, :, 1]) / 1e3 / self.df).astype(int) > 1

        gb_inds = gb_inds_into_proposal.reshape(ntemps, nwalkers, nleaves_max)
        gb_inds[remove] = False

        points_curr = gb_fixed_coords_into_proposal[gb_inds]
        points_prop = q["gb_fixed"][gb_inds]

        prior_all_curr = self.gpu_priors["gb_fixed"].logpdf(points_curr)
        prior_all_prop = self.gpu_priors["gb_fixed"].logpdf(points_prop)

        keep_prior = (~self.xp.isinf(prior_all_prop))

        prior_ok = self.xp.ones_like(gb_inds)
        prior_ok[gb_inds] = keep_prior

        gb_inds[~prior_ok] = False

        points_curr = points_curr[keep_prior]
        points_prop = points_prop[keep_prior]

        prior_all_curr = prior_all_curr[keep_prior]
        prior_all_prop = prior_all_prop[keep_prior]

        factors = factors_temp.reshape(ntemps, nwalkers, nleaves_max)[gb_inds]

        random_vals_all = self.xp.log(self.xp.random.rand(points_prop.shape[0]))

        L_contribution = self.xp.zeros_like(random_vals_all, dtype=complex)
        p_contribution = self.xp.zeros_like(random_vals_all, dtype=complex)

        data = self.mgh.data_list
        psd = self.mgh.psd_list
        # do unique for band size as separator between asynchronous kernel launches
        band_indices = self.xp.searchsorted(self.band_edges, points_curr[:, 1] / 1e3) - 1
        
        group_temp_finder = [
            self.xp.repeat(self.xp.arange(ntemps), nwalkers * nleaves_max).reshape(ntemps, nwalkers, nleaves_max),
            self.xp.tile(self.xp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose((0, 2, 1)),
            self.xp.tile(self.xp.arange(nleaves_max), ((ntemps, nwalkers, 1)))
        ]

        temp_inds = group_temp_finder[0][gb_inds]
        walker_inds = group_temp_finder[1][gb_inds]
        leaf_inds = group_temp_finder[2][gb_inds]

        N_vals_in = N_vals[gb_inds]

        special_band_inds = int(1e12) * temp_inds + int(1e6) * walker_inds + band_indices
        sort = self.xp.argsort(special_band_inds)
        
        temp_inds = temp_inds[sort]
        walker_inds = walker_inds[sort]
        leaf_inds = leaf_inds[sort]
        band_indices = band_indices[sort]
        factors = factors[sort]
        points_curr = points_curr[sort]
        points_prop = points_prop[sort]
        N_vals_in = N_vals_in[sort]

        special_band_inds_sorted = special_band_inds[sort]

        uni_special_bands, uni_index_special_bands, uni_count_special_bands = self.xp.unique(special_band_inds_sorted, return_index=True, return_counts=True)

        params_curr = self.parameter_transforms.both_transforms(points_curr, xp=self.xp)
        params_prop = self.parameter_transforms.both_transforms(points_prop, xp=self.xp)

        accepted_out = self.xp.zeros_like(random_vals_all, dtype=bool)

        do_synchronize = False
        device = self.xp.cuda.runtime.getDevice()
        
        for remainder in range(4):
            all_inputs = []
            band_bookkeep_info = []
            prior_info = []
            indiv_info = []
            params_prop_info = []
            # ll_before = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
            for N_now in unique_N:
                N_now = N_now.item()
                if N_now == 0:
                    continue

                checkit = np.where(new_state.supplimental[:]["overall_inds"] == 0)

                keep = (band_indices % 4 == remainder) & (N_vals_in == N_now)  #  & (temp_inds == checkit[0].item()) & (walker_inds == checkit[1].item()) #  & (band_indices > 530) & (band_indices < 540)  #  &  (temp_inds == 0) & (walker_inds == 0)
                
                if keep.sum().item() == 0:
                    continue
                #keep[:] = False
                # keep[tmp_keep] = True
                # keep[3000:3020:1] = True
                
                params_prop_info.append((points_prop[keep]))

                params_curr_in = params_curr[keep]
                params_prop_in = params_prop[keep]

                # switch to polar angle
                params_curr_in[:, -1] = np.pi / 2. - params_curr_in[:, -1]
                params_prop_in[:, -1] = np.pi / 2. - params_prop_in[:, -1]

                accepted_out_here = accepted_out[keep]

                params_curr_in_here = params_curr_in.flatten().copy()
                params_prop_in_here = params_prop_in.flatten().copy()

                prior_all_curr_here = prior_all_curr[keep]
                prior_all_prop_here = prior_all_prop[keep]

                prior_info.append((prior_all_curr_here, prior_all_curr_here))
                factors_here = factors[keep]
                random_vals_here = random_vals_all[keep]
                special_band_inds_sorted_here = special_band_inds_sorted[keep]
                
                uni_special_band_inds_here, uni_index_special_band_inds_here, uni_count_special_band_inds_here = self.xp.unique(special_band_inds_sorted_here, return_index=True, return_counts=True)

                # for finding the final frequency
                finding_final = self.xp.concatenate([uni_index_special_band_inds_here[1:], self.xp.array([len(special_band_inds_sorted_here)])]) - 1
                
                band_start_bin_ind_here = uni_index_special_band_inds_here.astype(np.int32)
                band_num_bins_here = uni_count_special_band_inds_here.astype(np.int32)

                band_inds = band_indices[keep][uni_index_special_band_inds_here]
                band_temps_inds = temp_inds[keep][uni_index_special_band_inds_here]
                band_walkers_inds = walker_inds[keep][uni_index_special_band_inds_here]

                indiv_info.append((temp_inds[keep], walker_inds[keep], leaf_inds[keep]))

                band_bookkeep_info.append((band_temps_inds, band_walkers_inds, band_inds))
                data_index_tmp = band_temps_inds * nwalkers + band_walkers_inds

                L_contribution_here = L_contribution[keep][uni_index_special_band_inds_here]
                p_contribution_here = p_contribution[keep][uni_index_special_band_inds_here]

                buffer = 5  # bins

                # determine starting point for each segment
                special_for_ind_deter1 = ((temp_inds[keep] * nwalkers + walker_inds[keep]) * len(self.band_edges) + band_indices[keep]) * 1e3 + params_curr_in[:, 1] * 1e3
                sort_special_for_ind_deter1 = self.xp.argsort(special_for_ind_deter1)
                start_inds1 = ((params_curr_in[:, 1][sort_special_for_ind_deter1][uni_index_special_band_inds_here] / self.df).astype(int) - (N_now / 2) - buffer).astype(int)
                
                final_inds1 = ((params_curr_in[:, 1][sort_special_for_ind_deter1][finding_final] / self.df).astype(int) + (N_now / 2) + buffer).astype(int)
                
                special_for_ind_deter2 = ((temp_inds[keep] * nwalkers + walker_inds[keep]) * len(self.band_edges) + band_indices[keep]) * 1e3 + params_prop_in[:, 1] * 1e3
                sort_special_for_ind_deter2 = self.xp.argsort(special_for_ind_deter2)
                start_inds2 = ((params_prop_in[:, 1][sort_special_for_ind_deter2][uni_index_special_band_inds_here] / self.df).astype(int) - (N_now / 2) - buffer).astype(int)

                final_inds2 = ((params_prop_in[:, 1][sort_special_for_ind_deter2][finding_final] / self.df).astype(int) + (N_now / 2) + buffer).astype(int)

                start_inds = self.xp.min(self.xp.asarray([start_inds1, start_inds2]), axis=0).astype(np.int32)
                end_inds = self.xp.max(self.xp.asarray([final_inds1, final_inds2]), axis=0).astype(np.int32)

                lengths = (end_inds - start_inds).astype(np.int32)

                max_data_store_size = lengths.max().item()

                band_inv_temp_vals_here = self.xp.asarray(self.temperature_control.betas)[band_temps_inds]

                data_index_here = self.mgh.get_mapped_indices(data_index_tmp).astype(np.int32)
                noise_index_here = data_index_here.copy()
                
                num_bands_here = len(band_inds)

                # ll_before = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)

                assert lengths.min() >= N_now + 2 * buffer
                inputs_now = (
                    L_contribution_here,
                    p_contribution_here,
                    data[0][0],
                    data[1][0],
                    psd[0][0],
                    psd[1][0],
                    data_index_here, 
                    noise_index_here,
                    params_curr_in_here,
                    params_prop_in_here,
                    prior_all_curr_here,
                    prior_all_prop_here,
                    factors_here,
                    random_vals_here,
                    band_start_bin_ind_here, # uni_index
                    band_num_bins_here, # uni_count
                    start_inds,
                    lengths,
                    band_inv_temp_vals_here,  # band_inv_temp_vals
                    accepted_out_here,
                    self.waveform_kwargs["T"],
                    self.waveform_kwargs["dt"], 
                    N_now,
                    0,
                    self.start_freq_ind,
                    self.data_length,
                    num_bands_here,
                    max_data_store_size,
                    device,
                    do_synchronize
                )
                
                all_inputs.append(inputs_now)

                self.gb.SharedMemoryMakeMove_wrap(
                    *inputs_now
                )
                self.xp.cuda.runtime.deviceSynchronize()
                """ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)

                ll_check = np.zeros((ntemps, nwalkers))

                for t in range(ntemps):
                    for w in range(nwalkers):
                        inds_tw = np.where((band_temps_inds == t) & (band_walkers_inds == w))

                        ll_check[t, w] += L_contribution_here[inds_tw].sum().item()
                """
            self.xp.cuda.runtime.deviceSynchronize()
            
            for inputs_now, band_info, prior_info_now, indiv_info_now, params_prop_now in zip(all_inputs, band_bookkeep_info, prior_info, indiv_info, params_prop_info):
                ll_contrib_now = inputs_now[0]
                lp_contrib_now = inputs_now[1]
                accepted_now = inputs_now[19]

                # print(accepted_now.sum(0) / accepted_now.shape[0])
                temp_tmp, walker_tmp, leaf_tmp = (indiv_info_now[0][accepted_now], indiv_info_now[1][accepted_now], indiv_info_now[2][accepted_now])

                gb_fixed_coords[(temp_tmp, walker_tmp, leaf_tmp)] = params_prop_now[accepted_now]

                ll_change = self.xp.zeros((ntemps, nwalkers, len(self.band_edges)))
                lp_change = self.xp.zeros((ntemps, nwalkers, len(self.band_edges)))

                self.xp.cuda.runtime.deviceSynchronize()

                ll_change[band_info] = ll_contrib_now
                
                ll_adjustment = ll_change.sum(axis=-1)
                log_like_tmp += ll_adjustment

                self.xp.cuda.runtime.deviceSynchronize()

                """print(ll_adjustment[0,0])
                ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)[0,0]
                breakpoint()"""

                lp_change[band_info] = lp_contrib_now
                
                lp_adjustment = lp_change.sum(axis=-1)
                log_prior_tmp += lp_adjustment

            self.xp.cuda.runtime.deviceSynchronize()
        
        try:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords.get()
            new_state.log_like[:] = log_like_tmp.get()
            new_state.log_prior[:] = log_prior_tmp.get()
        except AttributeError:
            new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords
            new_state.log_like[:] = log_like_tmp
            new_state.log_prior[:] = log_prior_tmp
            
        """new_state = self.temperature_control.temper_comps(new_state)
        
        ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)

        checkit = np.where(new_state.supplimental[:]["overall_inds"] == 0)
        
        print(checkit, np.abs(new_state.log_like - ll_after).max())
        et = time.perf_counter()
        print("CHECKING", et - st)
        breakpoint()"""
        # ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)

        self.mempool.free_all_blocks()

        if self.time % 1 == 0:
            ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
            # print(np.abs(new_state.log_like - ll_after).max())
            if np.abs(new_state.log_like - ll_after).max()  > 1e-2:
                if np.abs(new_state.log_like - ll_after).max() > 1e0:
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
                    self.xp.cuda.runtime.deviceSynchronize()
                    self.mgh.multiply_data(-1.)

        self.mempool.free_all_blocks()

        # get accepted fraction 
        accepted_check = np.all(np.abs(new_state.branches_coords["gb_fixed"] - state.branches_coords["gb_fixed"]) > 0.0, axis=-1).sum(axis=(1, 2)) / new_state.branches_inds["gb_fixed"].sum(axis=(1,2))

        # manually tell temperatures how real overall acceptance fraction is
        number_of_walkers_for_accepted = np.floor(nwalkers * accepted_check).astype(int)

        accepted_inds = np.tile(np.arange(nwalkers), (ntemps, 1))

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        accepted[accepted_inds < number_of_walkers_for_accepted[:, None]] = True

        tmp1 = np.all(np.abs(new_state.branches_coords["gb_fixed"] - state.branches_coords["gb_fixed"]) > 0.0, axis=-1).sum(axis=(2,))
        tmp2 = new_state.branches_inds["gb_fixed"].sum(axis=(2,))

        # add to move-specific accepted information
        self.accepted += tmp1
        if isinstance(self.num_proposals, int):
            self.num_proposals = tmp2
        else:
            self.num_proposals += tmp2

        # print(self.accepted / self.num_proposals)

        if self.temperature_control is not None:
            #st = time.perf_counter()
            new_state = self.temperature_control.temper_comps(new_state)
            #et = time.perf_counter()
            #print("temps ", (et - st))
            # 
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

                walker_map_iperm = np.repeat(iperm[:, None], coords_iperm.shape[-2], axis=-1)
                walker_map_i1perm = np.repeat(i1perm[:, None], coords_i1perm.shape[-2], axis=-1)

                walker_pre_permute_map_iperm = np.repeat(np.arange(len(iperm))[:, None], coords_iperm.shape[-2], axis=-1)
                walker_pre_permute_map_i1perm = np.repeat(np.arange(len(i1perm))[:, None], coords_i1perm.shape[-2], axis=-1)

                N_vals_iperm = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][i, iperm]

                N_vals_i1perm = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][i - 1, i1perm]

                f_iperm = coords_iperm[:, :, 1] / 1e3
                f_i1perm = coords_i1perm[:, :, 1] / 1e3

                bands_iperm = np.searchsorted(self.band_edges.get(), f_iperm.flatten(), side="right").reshape(f_iperm.shape) - 1
                bands_i1perm = np.searchsorted(self.band_edges.get(), f_i1perm.flatten(), side="right").reshape(f_i1perm.shape) - 1

                bands_iperm[np.isnan(f_iperm)] = -1
                bands_i1perm[np.isnan(f_i1perm)] = -1
                for odds_evens in range(2):
                    keep_here_i = (bands_iperm % 2 == odds_evens) & (bands_iperm >= 0) & (bands_iperm == 0) & (walker_map_iperm == 0)
                    keep_here_i1 = (bands_i1perm % 2 == odds_evens) & (bands_i1perm >= 0) & (bands_i1perm == 0) & (walker_map_i1perm == 39)

                    bands_here_i = bands_iperm[keep_here_i]
                    bands_here_i1 = bands_i1perm[keep_here_i1]

                    walkers_here_i_remove = walker_map_iperm[keep_here_i]
                    walkers_here_i1_remove = walker_map_i1perm[keep_here_i1]

                    walkers_here_i_add = walker_map_iperm[keep_here_i1]  # i walkers with i1 binaries
                    walkers_here_i1_add = walker_map_i1perm[keep_here_i]

                    walker_band_map_i = int(1e6) * walkers_here_i_remove + bands_here_i
                    walker_band_map_i1 = int(1e6) * walkers_here_i1_remove + bands_here_i1

                    walkers_pre_permute_here_i = walker_pre_permute_map_iperm[keep_here_i]
                    walkers_pre_permute_here_i1 = walker_pre_permute_map_i1perm[keep_here_i1]

                    N_here_i = N_vals_iperm[keep_here_i]
                    N_here_i1 = N_vals_i1perm[keep_here_i1]
                    
                    coords_here_i = coords_iperm[keep_here_i]
                    coords_here_i1 = coords_i1perm[keep_here_i1]

                    coords_in_here_i = self.parameter_transforms.both_transforms(coords_here_i)
                    coords_in_here_i1 = self.parameter_transforms.both_transforms(coords_here_i1)
                    
                    params_proposal_in = np.concatenate([
                        coords_in_here_i,  # remove from i
                        coords_in_here_i1, # add to i
                        coords_in_here_i1,  # remove form i - 1
                        coords_in_here_i, # add to i - 1
                    ], axis=0)

                    N_vals_in = np.concatenate([
                        N_here_i, 
                        N_here_i1, 
                        N_here_i1, 
                        N_here_i
                    ])

                    data_index_tmp_i_remove = i * nwalkers + walkers_here_i_remove
                    data_index_tmp_i1_remove = (i - 1) * nwalkers + walkers_here_i1_remove

                    data_index_tmp_i_add = i * nwalkers + walkers_here_i_add
                    data_index_tmp_i1_add = (i - 1) * nwalkers + walkers_here_i1_add

                    data_index_tmp_all = np.concatenate([data_index_tmp_i_remove, data_index_tmp_i_add, data_index_tmp_i1_remove, data_index_tmp_i1_add])
                    data_index_forward = self.xp.asarray(self.mgh.get_mapped_indices(data_index_tmp_all)).astype(self.xp.int32)
                    
                    factors_multiply_forward = self.xp.asarray(np.concatenate([
                        np.full_like(data_index_tmp_i_remove, -1.0, dtype=np.float64),
                        np.full_like(data_index_tmp_i_add, +1.0, dtype=np.float64),
                        np.full_like(data_index_tmp_i1_remove, -1.0, dtype=np.float64),
                        np.full_like(data_index_tmp_i1_add, +1.0, dtype=np.float64), 
                    ]))
                    waveform_kwargs_fill = waveform_kwargs_now.copy()

                    f_find = np.concatenate([coords_here_i[:, 1], coords_here_i1[:, 1]]) / 1e3
                    
                    walker_find = np.concatenate([walkers_pre_permute_here_i, walkers_pre_permute_here_i1])
                    N_here_find = np.concatenate([N_here_i, N_here_i1])
                    band_here_find = np.searchsorted(self.band_edges.get(), f_find, side="right") - 1
                    walker_band_find = walker_find * int(1e6) + band_here_find

                    f_find_min = f_find - (N_here_find / 2) * self.df
                    f_find_min_sort = np.argsort(f_find_min)
                    f_find_min = f_find_min[f_find_min_sort]
                    walker_band_find_min = walker_band_find[f_find_min_sort]
                    walker_band_uni_first, walker_band_first = np.unique(walker_band_find_min, return_index=True)
                    f_min_band = f_find_min[walker_band_first]
                    
                    f_find_max = f_find + (N_here_find / 2) * self.df
                    f_find_max_sort = np.argsort(f_find_max)
                    f_find_max = f_find_max[f_find_max_sort]
                    walker_band_find_max = walker_band_find[f_find_max_sort]
                    walker_band_uni_last, walker_band_last = np.unique(walker_band_find_max[::-1], return_index=True)
                    f_max_band = f_find_max[::-1][walker_band_last]

                    assert np.all(walker_band_uni_first == walker_band_uni_last)
                
                    start_inds_band = (f_min_band / self.df).astype(int)
                    end_inds_band = (f_max_band / self.df).astype(int)
                    lengths_band = end_inds_band - start_inds_band

                    walker_permute = (walker_band_uni_first / 1e6).astype(int)

                    walker_i = iperm[walker_permute]
                    walker_i1 = i1perm[walker_permute]

                    walker_band_in_i = walker_band_uni_first * int(1e6) - walker_permute + walker_i * int(1e6)
                    walker_band_in_i1 = walker_band_uni_first * int(1e6) - walker_permute + walker_i1 * int(1e6)

                    start_inds_all = self.xp.asarray(np.concatenate([start_inds_band, start_inds_band]).astype(np.int32))
                    lengths_all = self.xp.asarray(np.concatenate([lengths_band, lengths_band]).astype(np.int32))

                    data_index_tmp_all = np.concatenate([i * nwalkers + walker_i, (i - 1) * nwalkers + walker_i1])

                    data_index_all = self.xp.asarray(self.mgh.get_mapped_indices(data_index_tmp_all).astype(np.int32))
                    noise_index_all = data_index_all.copy()
                    
                    ll_contrib_before = self.run_ll_part_comp(
                        data_index_all, 
                        noise_index_all, 
                        start_inds_all, 
                        lengths_all
                    )

                    self.gb.generate_global_template(
                        params_proposal_in,
                        data_index_forward, 
                        self.mgh.data_list, 
                        N=self.xp.asarray(N_vals_in),
                        data_length=self.data_length, 
                        data_splits=self.mgh.gpu_splits, 
                        factors=factors_multiply_forward,
                        **waveform_kwargs_fill
                    )

                    ll_contrib_after = self.run_ll_part_comp(
                        data_index_all, 
                        noise_index_all, 
                        start_inds_all, 
                        lengths_all
                    )

                    half = start_inds_band.shape[0]

                    delta_logl_i = ll_contrib_after[:half] - ll_contrib_before[:half]
                    delta_logl_i1 = ll_contrib_after[half:] - ll_contrib_before[half:]

                    paccept = dbeta * 1. / 2. * (delta_logl_i - delta_logl_i1)
                    raccept = np.log(np.random.uniform(size=paccept.shape[0]))

                    # How many swaps were accepted?
                    sel = paccept > self.xp.asarray(raccept)
                    sel[:] = False

                    keep_walker_band_i = walker_band_in_i[sel.get()]
                    keep_walker_band_i1 = walker_band_in_i1[sel.get()]

                    reverse_walker_band_i = walker_band_in_i[~sel.get()]
                    reverse_walker_band_i1 = walker_band_in_i1[~sel.get()]

                    breakpoint()
                    reverse_i = np.in1d(walker_band_map_i, reverse_walker_band_i)
                    reverse_i1 = np.in1d(walker_band_map_i1, reverse_walker_band_i1)
                    
                    reverse_N_here_i = N_vals_iperm[keep_here_i][reverse_i]
                    reverse_N_here_i1 = N_vals_i1perm[keep_here_i1][reverse_i1]
                    
                    reverse_coords_here_i = coords_iperm[keep_here_i][reverse_i]
                    reverse_coords_here_i1 = coords_i1perm[keep_here_i1][reverse_i1]

                    reverse_coords_in_here_i = self.parameter_transforms.both_transforms(reverse_coords_here_i)
                    reverse_coords_in_here_i1 = self.parameter_transforms.both_transforms(reverse_coords_here_i1)
                    
                    reverse_params_proposal_in = np.concatenate([
                        reverse_coords_in_here_i,  # remove from i
                        reverse_coords_in_here_i1, # add to i
                        reverse_coords_in_here_i1,  # remove form i - 1
                        reverse_coords_in_here_i, # add to i - 1
                    ], axis=0)

                    reverse_N_vals_in = np.concatenate([
                        reverse_N_here_i, 
                        reverse_N_here_i1, 
                        reverse_N_here_i1, 
                        reverse_N_here_i
                    ])

                    reverse_data_index_tmp_i_remove = i * nwalkers + walkers_here_i_remove[reverse_i]
                    reverse_data_index_tmp_i1_remove = (i - 1) * nwalkers + walkers_here_i1_remove[reverse_i1]

                    reverse_data_index_tmp_i_add = i * nwalkers + walkers_here_i_add[reverse_i1]
                    reverse_data_index_tmp_i1_add = (i - 1) * nwalkers + walkers_here_i1_add[reverse_i]

                    reverse_data_index_tmp_all = np.concatenate([reverse_data_index_tmp_i_remove, reverse_data_index_tmp_i_add, reverse_data_index_tmp_i1_remove, reverse_data_index_tmp_i1_add])
                    reverse_data_index = self.xp.asarray(self.mgh.get_mapped_indices(reverse_data_index_tmp_all)).astype(self.xp.int32)
                    
                    reverse_factors_multiply = self.xp.asarray(np.concatenate([
                        np.full_like(reverse_data_index_tmp_i_remove, -1.0, dtype=np.float64),
                        np.full_like(reverse_data_index_tmp_i_add, +1.0, dtype=np.float64),
                        np.full_like(reverse_data_index_tmp_i1_remove, -1.0, dtype=np.float64),
                        np.full_like(reverse_data_index_tmp_i1_add, +1.0, dtype=np.float64), 
                    ]))

                    self.gb.generate_global_template(
                        reverse_params_proposal_in,
                        reverse_data_index, 
                        self.mgh.data_list, 
                        N=self.xp.asarray(reverse_N_vals_in),
                        data_length=self.data_length, 
                        data_splits=self.mgh.gpu_splits, 
                        factors=reverse_factors_multiply,
                        **waveform_kwargs_fill
                    )
                    
                    breakpoint()
                
                # MAKE SURE TO MOVE PRIORS !!!!
                # Do adaptations

        else:
            self.temperature_control.swaps_accepted = np.zeros((ntemps - 1))
        
        
        if np.any(new_state.log_like > 1e10):
            breakpoint()

        self.time += 1
        #self.xp.cuda.runtime.deviceSynchronize()
        """et = time.perf_counter()
        print("end stretch", (et - st))"""

        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        et = time.perf_counter()
        # print("in-model end", (et - st))
                    
        # breakpoint()
        return new_state, accepted

