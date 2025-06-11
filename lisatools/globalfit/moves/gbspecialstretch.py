from __future__ import annotations
# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N
from ...detector import sangria
from .globalfitmove import GlobalFitMove, GFCombineMove
from ..galaxyglobal import run_gb_bulk_search, fit_each_leaf, make_gmm
from eryn.state import BranchSupplemental
from typing import Optional      

try:
    import cupy as cp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from ...utils.utility import searchsorted2d_vec, get_groups_from_band_structure
from ...sampling.prior import FullGaussianMixtureModel, GBPriorWrap

from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer
from eryn.utils.utility import groups_from_inds
from eryn.utils import PeriodicContainer

from eryn.moves import GroupStretchMove, Move
from eryn.moves.multipletry import logsumexp, get_mt_computations
from ...utils.utility import get_array_module
from ...diagnostic import inner_product
from lisatools.globalfit.state import GFState
from lisatools.sampling.prior import GBPriorWrap


__all__ = ["GBSpecialStretchMove"]

def gb_search_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    assert comm is not None

    # get current rank and get index into class_ranks_list
    print(f"INSIDE GB search, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    if rank_index == 0:
        comm_info = {"process_ranks_for_fit": class_ranks_list}
        print("waiting to send process ranks")
        comm.send(comm_info, dest=main_rank, tag=232342)
        print("sent process ranks")
        
    fit_each_leaf(rank, curr, main_rank, comm)


# def gb_search_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
#     assert comm is not None

#     # get current rank and get index into class_ranks_list
#     print(f"INSIDE GB search, RANK: {comm.Get_rank()}")
#     rank = comm.Get_rank()
#     rank_index = class_ranks_list.index(rank)
#     gather_rank = class_ranks_list[0]
#     if rank_index == 0:
#         split_remainder = 1  # will fix this setup in the future
#         num_search = 2
#         gpu = class_extra_gpus[0]
#         comm_info = {"process_ranks_for_fit": class_ranks_list[1:]}
#         # run search here
#         run_gb_bulk_search(gpu, curr, comm, comm_info, main_rank, num_search, split_remainder)
#         pass

#     else:
#         # run GMM fit here
#         fit_each_leaf(rank, curr, gather_rank, comm)
#         pass


def fit_gmm(samples, comm, comm_info):

    if len(samples) == 0:
        return None

    keep = np.arange(8)  # array([0, 1, 2, 4, 6, 7])

    if samples.ndim == 4:
        num_keep, num_samp, nwalkers_keep, ndim = samples.shape

        args = []
        for band in range(num_keep): 
            args.append(samples[band].reshape(-1, ndim)[:, keep])

    elif samples.ndim == 2:
        max_groups = samples[:, 0].astype(int).max()

        args = []
        for group in np.unique(samples[:, 0].astype(int)): 
            keep_samp = samples[:, 0].astype(int) == group
            if keep_samp.sum() > 0:
                if np.any(np.isnan(samples[keep_samp, 3:])) or np.any(np.isinf(samples[keep_samp, 3:])):
                    breakpoint()
                args.append(samples[keep_samp, 3:][:, keep])

    else:
        raise ValueError
    
    # for debugging
    # args = args[:1000]

    batch = 10000
    breaks = np.arange(0, len(args) + batch, batch)
    print("BREAKS", breaks)
    if len(breaks) == 1:
        breakpoint()
    process_ranks_for_fit = comm_info["process_ranks_for_fit"]
    gmm_info_all = []
    for i in range(len(breaks) - 1):
        start = breaks[i]
        end = breaks[i + 1]
        args_tmp = args[start:end]
        gmm_info = [None for tmp in args_tmp]
        gmm_complete = np.zeros(len(gmm_info), dtype=bool)
        

        # OPPOSITE
        # send_tags = comm_info["rec_tags"]
        # rec_tags = comm_info["send_tags"]
        outer_iteration = 0
        current_send_arg_index = 0
        current_status = [False for _ in process_ranks_for_fit]

        while np.any(~gmm_complete):
            time.sleep(0.1)
            if current_send_arg_index >= len(args_tmp) and np.all(~np.asarray(current_status)):
                current_send_arg_index = 0

            outer_iteration += 1
            if outer_iteration % 500 == 0:
                print(f"ITERATION: {outer_iteration}, need:", np.sum(~gmm_complete), current_status)

            for proc_i, proc_rank in enumerate(process_ranks_for_fit):
                # time.sleep(0.6)
                if current_status[proc_i]:
                    rec_tag = int(str(proc_rank) + "4545")
                    check_output = comm.irecv(source=proc_rank)

                    if not check_output.get_status():
                        check_output.cancel()
                    else:
                        # first two give some delay for the processor that messes up
                        try:
                            output_info = check_output.wait()
                        except (pickle.UnpicklingError, UnicodeDecodeError, ValueError, OverflowError) as e:
                            current_status[proc_i] = False
                            print("BAD error on return")
                            continue
                        if "BAD" in output_info:
                            current_status[proc_i] = False
                            print("BAD", output_info["BAD"])
                            continue
                        # print(output_info)

                        arg_index = output_info["arg"]
                        rank_recv = output_info["rank"]
                        output_list = output_info["output"]

                        gmm_info[arg_index] = output_list
                        gmm_complete[arg_index] = True
                        current_status[proc_i] = False

                        if gmm_complete.sum() + 25 > len(args):
                            print(proc_i, current_status)
                        
                if not current_status[proc_i]:
                    while current_send_arg_index < len(args_tmp) and gmm_complete[current_send_arg_index]:
                        current_send_arg_index += 1

                    if current_send_arg_index < len(args_tmp):
                        send_info = {"samples": args_tmp[current_send_arg_index], "arg": current_send_arg_index}
                        # print("sending", process_ranks_for_fit[index_add])
                        send_tag = int(str(proc_rank) + "67676")
                        comm.send(send_info, dest=proc_rank, tag=send_tag)
                        current_status[proc_i] = True
                        
                        current_send_arg_index += 1

        gmm_info_all.append(gmm_info)

    weights = [tmp[0] for tmp in gmm_info]
    means = [tmp[1] for tmp in gmm_info]
    covs = [tmp[2] for tmp in gmm_info]
    invcovs = [tmp[3] for tmp in gmm_info]
    dets = [tmp[4] for tmp in gmm_info]
    mins = [tmp[5] for tmp in gmm_info]
    maxs = [tmp[6] for tmp in gmm_info]
    
    output = [weights, means, covs, invcovs, dets, mins, maxs]

    return output


def fit_each_leaf(rank, curr, gather_rank, comm):

    run_process = True

    while run_process:
        try:
            check = comm.recv(source=gather_rank)
        except (pickle.UnpicklingError, UnicodeDecodeError, ValueError, OverflowError) as e:
            # print("BAD BAD ", rank)
            comm.send({"BAD": "receiving issue"}, dest=gather_rank, tag=send_tag)
            continue

        if isinstance(check, str):
            if check == "end":
                run_process = False
            continue

        assert isinstance(check, dict)

        try:
            arg_index = check["arg"]

            # print("INSIDE", rank, arg_index)
            samples = check["samples"]
        except KeyError:
            comm.send({"BAD":  "KeyError"}, dest=gather_rank, tag=send_tag)
            continue

        assert isinstance(samples, np.ndarray)

        gmm = GMMFit(samples)
        output_list = [gmm.keep_mix.weights_, gmm.keep_mix.means_, gmm.keep_mix.covariances_, np.array([np.linalg.inv(gmm.keep_mix.covariances_[i]) for i in range(len(gmm.keep_mix.weights_))]), np.array([np.linalg.det(gmm.keep_mix.covariances_[i]) for i in range(len(gmm.keep_mix.weights_))]), gmm.sample_mins, gmm.sample_maxs]
        comm.send({"output": output_list, "rank": rank, "arg": arg_index}, dest=gather_rank)
    return


def gb_refit_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    assert comm is not None

    # get current rank and get index into class_ranks_list
    print(f"INSIDE GB refit, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    gather_rank = class_ranks_list[0]
    if rank_index == 0:
        split_remainder = 0  # will fix this setup in the future
        num_search = 2
        gpu = class_extra_gpus[0]
        comm_info = {"process_ranks_for_fit": class_ranks_list[1:]}
        # run search here
        # run_gb_bulk_search(gpu, curr, comm, comm_info, main_rank, num_search, split_remainder)
        pass

    else:
        # run GMM fit here
        fit_each_leaf(rank, curr, gather_rank, comm)
        pass

from eryn.state import Branch
from dataclasses import dataclass

    
from eryn.utils import TransformContainer

class Buffer:

    @property
    def xp(self) -> object:
        if self.use_gpu:
            return cp
        else:
            return np
    
    def get_index(self, special_inds_test):
        special_indices_unique_sort, special_indices_unique = self.index_info
        now_index = (special_indices_unique_sort[cp.searchsorted(special_indices_unique[special_indices_unique_sort], special_inds_test, side="right") - 1]).astype(cp.int32)
        return now_index 

    def __init__(self, gb, band_edges, unique_band_combos, params_interest, num_bands_now, nchannels, data_length, uni_info, transform_fn, waveform_kwargs, df, sources_now_map, sources_inject_now_map, index_info, special_band_inds, opt_snr_rej_samp_limit=10.0, use_gpu=True, use_template_arr=False, *args, **kwargs):
        self.use_gpu = use_gpu
        assert self.use_gpu == gb.use_gpu
        self.gb = gb
        self.df = df
        self.sources_now_map, self.sources_inject_now_map = sources_now_map, sources_inject_now_map
        self.band_edges, self.unique_band_combos = band_edges, unique_band_combos
        self.num_bands = len(self.band_edges) - 1
        self.params_interest = params_interest
        
        self.index_info = index_info
        self.num_bands_now, self.nchannels, self.data_length = num_bands_now, nchannels, data_length
        self.uni_special, self.uni_special_index, self.uni_special_reverse, self.uni_special_counts = uni_info
        self.transform_fn = transform_fn
        self.waveform_kwargs = waveform_kwargs
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        self.use_template_arr = use_template_arr
        # load data into buffer for these bands
        # 3 is number of sub-bands to store
        self.band_buffer_tmp = cp.zeros(
            (self.num_bands_now * self.nchannels * self.data_length)
            , dtype=complex
        )
        
        self.psd_buffer_tmp = cp.zeros(
            (self.num_bands_now *  self.nchannels * self.data_length)
            , dtype=np.float64
        )

        # careful here with accessing memory
        self.band_buffer = self.band_buffer_tmp.reshape((self.num_bands_now, self.nchannels, self.data_length))
        self.psd_buffer = self.psd_buffer_tmp.reshape((self.num_bands_now, self.nchannels, self.data_length))
        if self.use_template_arr:
            self.template_buffer_tmp = cp.zeros(
                (self.num_bands_now *  self.nchannels * self.data_length)
                , dtype=complex
            )

            self.template_buffer = self.template_buffer_tmp.reshape((self.num_bands_now, self.nchannels, self.data_length))
        
        # TODO: adjust this
        self.edge_buffer = 2000

        if self.num_bands == 1:
            tmp_buffer_start_index = (self.band_edges[0] / self.df).astype(np.int32) - self.edge_buffer
            assert tmp_buffer_start_index + data_length >= ((self.band_edges[-1] / self.df).astype(np.int32) + self.edge_buffer)
            self.buffer_start_index = self.xp.repeat(tmp_buffer_start_index, self.unique_band_combos.shape[0])
            
        else:
            self.buffer_start_index = (self.band_edges[self.unique_band_combos[:, 2] - 1] / self.df).astype(np.int32)
            self.buffer_start_index[self.unique_band_combos[:, 2] == 0] = (self.band_edges[0] / self.df).astype(np.int32) - self.edge_buffer
            # self.buffer_start_index[self.unique_band_combos[:, 2] == self.num_bands - 1] = (self.band_edges[-1] / self.df).astype(np.int32) - self.edge_buffer
            
        self.start_freq_inds = self.buffer_start_index.copy().astype(np.int32)
        
        # TODO: fix this 4????
        lower_f_lim = self.band_edges[self.unique_band_combos[:, 2]]  #  - self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4
        higher_f_lim = self.band_edges[self.unique_band_combos[:, 2] + 1]  #  + self.band_N_vals[self.unique_band_combos[:, 2]] * self.df / 4
        self.frequency_lims = [lower_f_lim, higher_f_lim]
        self.special_band_inds = special_band_inds
        assert special_band_inds.shape[0] == self.params_interest.shape[0]
        self.now_index = self.get_index(special_band_inds)
        
    def likelihood(self, source_only: bool = False, noise_only: bool = False) -> float:
        assert not (source_only and noise_only) 

        # THIS HAS TO HAVE THE .COPY()
        numerator_in = self.band_buffer.copy()
        if self.use_template_arr:
            numerator_in -= self.template_buffer

        if self.nchannels > 2:
            # TODO: need to change buffers to analysis containers to handle XYZ psd setup
            raise NotImplementedError("Need to adjust the buffer to analysis container to take care of that")
            breakpoint()

        if source_only:
            source_term = -1. / 2. * 4.0 * self.df * cp.sum((numerator_in.conj() * numerator_in) * self.psd_buffer, axis=(1, 2)).real
            return source_term

        elif noise_only:
            psd_term = -cp.sum(cp.log(cp.abs(1/self.psd_buffer[self.psd_buffer != 0.0])))
            return psd_term

        source_term = -1. / 2. * 4.0 * self.df * cp.sum((numerator_in.conj() * numerator_in) * self.psd_buffer, axis=(1, 2)).real
        psd_term = -cp.sum(cp.log(cp.abs(self.psd_buffer[self.psd_buffer != 0.0])))
        cp.get_default_memory_pool().free_all_blocks()
        return source_term + psd_term

    def _get_fill_buffer_ind_map(self, acs):
        assert np.all(acs.start_freq_ind[0] == acs.start_freq_ind)
        start_freq_ind = acs.start_freq_ind[0]
        assert np.all((self.buffer_start_index - start_freq_ind) >= 0)
        assert np.all((self.buffer_start_index - start_freq_ind + self.data_length) <= acs.data_length)
        inds1 = cp.repeat(self.unique_band_combos[:, 1], self.nchannels * self.data_length).reshape((self.num_bands_now,) + self.band_buffer.shape[1:])
        inds2 = cp.repeat(cp.arange(self.nchannels), (self.num_bands_now * self.band_buffer.shape[-1])).reshape(self.nchannels, self.num_bands_now, self.band_buffer.shape[-1]).transpose(1, 0, 2)
        inds3 = (cp.arange(self.band_buffer.shape[-1])[None, None, :] + (cp.repeat(self.buffer_start_index, self.nchannels * self.band_buffer.shape[-1]).reshape((self.num_bands_now,) + self.band_buffer.shape[1:]))) - start_freq_ind
        inds_get = (inds1.flatten(), inds2.flatten(), inds3.flatten())
        return inds_get

    def get_swap_ll(self, params_remove, params_add, data_index, N_vals, phase_maximize=False):

        params_remove_in = self.transform_fn.both_transforms(
            params_remove, xp=cp
        )

        params_add_in = self.transform_fn.both_transforms(
            params_add, xp=cp
        )
        
        # print("NEED TO CHECK THIS")
        # TODO: add inplace (would need to add information for accept/reject)
        # with buffer need to not be in kwargs
        wave_kwargs_tmp = self.waveform_kwargs.copy()
        if "start_freq_ind" in wave_kwargs_tmp:
            wave_kwargs_tmp.pop("start_freq_ind")

        if np.any((params_add_in[:, 1] / self.df).astype(int) - self.start_freq_inds[data_index] + (N_vals / 2) >  self.band_buffer.shape[-1]):
            breakpoint()
        if np.any((params_remove_in[:, 1] / self.df).astype(int) - self.start_freq_inds[data_index] + (N_vals / 2) >  self.band_buffer.shape[-1]):
            breakpoint()
        if np.any((params_add_in[:, 1] / self.df).astype(int) - self.start_freq_inds[data_index] - (N_vals / 2) < 0):
            breakpoint()
        if np.any((params_remove_in[:, 1] / self.df).astype(int) - self.start_freq_inds[data_index] - (N_vals / 2) < 0):
            breakpoint()
        
        ll_diff = cp.asarray(self.gb.swap_likelihood_difference(
            params_remove_in,
            params_add_in,
            self.band_buffer_tmp,
            self.psd_buffer_tmp,
            start_freq_ind=self.start_freq_inds,
            data_index=data_index,
            noise_index=data_index,
            adjust_inplace=False,
            N=N_vals,
            data_length=self.band_buffer.shape[-1],
            data_splits=np.full(self.band_buffer.shape[0], self.gb.gpus[0]),
            phase_marginalize=phase_maximize,
            return_cupy=True,
            **wave_kwargs_tmp,
        ))

        # breakpoint()
        if phase_maximize:
            params_add[:, 3] = params_add[:, 3] - self.gb.phase_angle

        # rejection sampling on SNR
        opt_snr = (self.gb.add_add.real ** (1/2)).copy()
        
        # params_add_in = self.transform_fn.both_transforms(
        #     params_add, xp=cp
        # )
        # ll_4 = cp.asarray(self.gb.get_ll(
        #     params_add_in,
        #     self.band_buffer_tmp,
        #     self.psd_buffer_tmp,
        #     start_freq_ind=self.start_freq_inds,
        #     data_index=data_index,
        #     noise_index=data_index,
        #     N=N_vals,
        #     data_length=self.band_buffer.shape[-1],
        #     data_splits=np.full(self.band_buffer.shape[0], self.gb.gpus[0]),
        #     phase_marginalize=False,  # phase_maximize,
        #     return_cupy=True,
        #     **wave_kwargs_tmp,
        # ))
        # # breakpoint()

        # data_index_tmp = self.xp.zeros(params_remove_in.shape[0], dtype=np.int32)

        # ll_diff_2 = cp.asarray(self.gb.swap_likelihood_difference(
        #     params_remove_in,
        #     params_add_in,
        #     self.acs.linear_data_arr,
        #     self.acs.linear_psd_arr,
        #     start_freq_ind=self.xp.asarray(self.acs.start_freq_ind).astype(np.int32),
        #     data_index=data_index_tmp,
        #     noise_index=data_index_tmp,
        #     adjust_inplace=False,
        #     N=N_vals,
        #     data_length=self.acs.data_length,
        #     data_splits=self.acs.gpu_map,
        #     phase_marginalize=phase_maximize,
        #     return_cupy=True,
        #     **wave_kwargs_tmp,
        # ))
        
        # breakpoint()
        # ll_3 = cp.asarray(self.gb.get_ll(
        #     params_add_in,
        #     self.acs.linear_data_arr,
        #     self.acs.linear_psd_arr,
        #     start_freq_ind=self.xp.asarray(self.acs.start_freq_ind).astype(np.int32),
        #     data_index=data_index_tmp,
        #     noise_index=data_index_tmp,
        #     N=N_vals,
        #     data_length=self.acs.data_length,
        #     data_splits=self.acs.gpu_map,
        #     phase_marginalize=phase_maximize,
        #     return_cupy=True,
        #     **wave_kwargs_tmp,
        # ))
        # breakpoint()
        # # rejection sampling on SNR
        # opt_snr_2 = self.gb.add_add.real ** (1/2)
        
        # TODO: change limit
        ll_diff[opt_snr < self.opt_snr_rej_samp_limit] = -1e300
        return ll_diff

    def reset_residual_buffers(self):
        self.band_buffer[:] = 0.0
    def reset_psd_buffers(self):
        self.psd_buffer[:] = 0.0

    def fill_buffer_residual_from_acs(self, acs):
        inds_get = self._get_fill_buffer_ind_map(acs)
        self.reset_residual_buffers()
        self.band_buffer[:self.num_bands_now] += rest_of_data[:]

    def fill_buffer_psd_from_acs(self, acs):
        inds_get = self._get_fill_buffer_ind_map(acs)
        self.reset_psd_buffers()
        self.psd_buffer[:self.num_bands_now] = acs.psd_shaped[0][inds_get].reshape((self.num_bands_now,) + self.band_buffer.shape[1:])
        
    def fill_buffer_residual_and_psd_from_acs(self, acs):
        inds_get = self._get_fill_buffer_ind_map(acs)
        rest_of_data = acs.data_shaped[0][inds_get].reshape((self.num_bands_now,) + self.band_buffer.shape[1:])
        # load rest of data into buffer (has current sources removed)
        self.reset_residual_buffers()
        self.band_buffer[:self.num_bands_now] += rest_of_data[:]
        self.reset_psd_buffers()
        self.psd_buffer[:self.num_bands_now] = acs.psd_shaped[0][inds_get].reshape((self.num_bands_now,) + self.band_buffer.shape[1:])

    # def adjust_sources_in_template_buffer(self, factor, params, params_index, N_vals, *args, **kwargs) -> None:
        
    #     assert isinstance(factor, int) and (factor == -1 or factor == +1)
        
    #     # inject current sources into buffers

    #     # TODO: check this???
    #     factors_change = factor * cp.ones_like(params_index, dtype=float)
    #     params_in = self.transform_fn.both_transforms(params, xp=cp)
    #     # assign N based on band
    #     # TODO: need to be careful about N changes across band edges?
        
    #     wave_kwargs_tmp = self.waveform_kwargs.copy()
    #     if "start_freq_ind" in wave_kwargs_tmp:
    #         wave_kwargs_tmp.pop("start_freq_ind")
    #     try:
    #         self.gb.generate_global_template(
    #             params_in,
    #             params_index,
    #             self.template_buffer_tmp,
    #             data_length=self.band_buffer.shape[-1],
    #             factors=factors_change,
    #             data_splits=np.full(self.band_buffer.shape[0], self.gb.gpus[0]),
    #             N=N_vals,
    #             start_freq_ind=self.start_freq_inds,
    #             **wave_kwargs_tmp,
    #         )
    #     except AssertionError:
    #         breakpoint()

    def remove_sources_from_template_buffer(self, *args, **kwargs) -> None:
        self.adjust_sources_in_band_buffer(-1, self.template_buffer_tmp, *args, **kwargs)
       
    def add_sources_to_template_buffer(self, *args, **kwargs) -> None:
        self.adjust_sources_in_band_buffer(+1, self.template_buffer_tmp, *args, **kwargs)
 
    def adjust_sources_in_band_buffer(self, factor, input_array, params, params_index, N_vals, *args, **kwargs) -> None:
        
        assert isinstance(factor, int) and (factor == -1 or factor == +1)
        
        # inject current sources into buffers

        # TODO: check this???
        factors_change = factor * cp.ones_like(params_index, dtype=float)
        params_in = self.transform_fn.both_transforms(params, xp=cp)
        # assign N based on band
        # TODO: need to be careful about N changes across band edges?
        wave_kwargs_tmp = self.waveform_kwargs.copy()
        if "start_freq_ind" in wave_kwargs_tmp:
            wave_kwargs_tmp.pop("start_freq_ind")
        try:
            self.gb.generate_global_template(
                params_in,
                params_index,
                input_array,
                data_length=self.band_buffer.shape[-1],
                factors=factors_change,
                data_splits=np.full(self.band_buffer.shape[0], self.gb.gpus[0]),
                N=N_vals,
                start_freq_ind=self.start_freq_inds,
                **wave_kwargs_tmp,
            )
        except AssertionError:
            breakpoint()

    def remove_sources_from_band_buffer(self, *args, **kwargs) -> None:
        self.adjust_sources_in_band_buffer(+1, self.band_buffer_tmp, *args, **kwargs)
       
    def add_sources_to_band_buffer(self, *args, **kwargs) -> None:
        self.adjust_sources_in_band_buffer(-1, self.band_buffer_tmp, *args, **kwargs)

def return_x(x):
    return x

class BandSorter:

    @property
    def xp(self) -> object:
        if self.use_gpu:
            return cp
        else:
            return np

    def __init__(
        self, 
        gb_branch: Branch, 
        band_edges: Optional[np.ndarray] = None, 
        band_N_vals: Optional[np.ndarray] = None, 
        use_gpu: bool=False, 
        transform_fn: Optional[TransformContainer] = None, 
        copy:bool = True,
        inds_subset: Optional[np.ndarray] = None,
        inds_main_band_sorter: Optional[np.ndarray] = None,
        gb = None,
        waveform_kwargs = {},
        main_band_sorter = None,
        max_data_store_size: int = 6000,
        rj_prop = None,
    ):
        
        dc = deepcopy if copy else return_x
        if hasattr(gb_branch, "num_sources"):
            _band_sorter = gb_branch
            self.use_gpu = _band_sorter.use_gpu
            for key, value in _band_sorter.__dict__.items():
                if key[:2] != "__":
                    if key in ["main_band_sorter", "inds_main_band_sorter", "gb", "rj_prop"]:
                        continue
                         
                    elif isinstance(value, self.xp.ndarray) and value.shape[0] == _band_sorter.num_sources:
                        if inds_subset is None:
                            inds_subset = self.xp.arange(_band_sorter.num_sources)
                        else:
                            assert isinstance(inds_subset, self.xp.ndarray) and inds_subset.dtype == int
                            assert inds_subset.max() < (_band_sorter.num_sources)
                        set_value = dc(value[inds_subset])

                    else:
                        set_value = dc(value)

                    setattr(self, key, set_value)

            self.rj_prop = _band_sorter.rj_prop
            self.gb = _band_sorter.gb
            # need to make sure is not mixed up in loop
            self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)
            return

        assert band_edges is not None and band_N_vals is not None
        self.use_gpu = use_gpu
        self.gb = gb
        self.waveform_kwargs = waveform_kwargs
        self.gb_branch_orig = gb_branch
        self.num_bands = len(band_edges) - 1
        self.band_edges = band_edges
        self.band_N_vals = band_N_vals
        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = gb_branch.shape
        self.orig_inds = self.xp.asarray(gb_branch.inds)
        
        self.rj_prop = rj_prop

        if rj_prop is not None:
            self.coords = self.xp.asarray(gb_branch.coords.reshape(-1, 8))
            self.inds = self.orig_inds.flatten()

            new_sources = cp.full_like(self.coords[~self.inds], np.nan)
            fix = cp.full(new_sources.shape[0], True)
            while cp.any(fix):
                new_sources[fix] = rj_prop.rvs(size=fix.sum().item())
                fix = cp.any(cp.isnan(new_sources), axis=-1)

            self.coords[~self.inds] = new_sources

            # if self.name == "rj_prior":
            # proposal_logpdf = self.rj_proposal_distribution["gb"].logpdf(
            #     points_curr[gb_inds]
            # )
            # else:
            proposal_logpdf = cp.zeros(self.coords.shape[0])

            batch_here = int(1e6)
            inds_splitting = np.arange(0, self.coords.shape[0], batch_here)
            if inds_splitting[-1] != self.coords.shape[0] - 1:
                inds_splitting = np.concatenate([inds_splitting, np.array([self.coords.shape[0] - 1])])
            
            for stind, eind in zip(inds_splitting[:-1], inds_splitting[1:]):
                proposal_logpdf[stind: eind] = self.xp.asarray(rj_prop.logpdf(self.coords[stind: eind]))
            self.xp.get_default_memory_pool().free_all_blocks()

            self.factors = (cp.asarray(proposal_logpdf) * -1) * (~self.orig_inds).flatten() + (cp.asarray(proposal_logpdf) * +1) * (self.orig_inds).flatten()
            tmp_inds_shaped = self.xp.full_like(self.orig_inds, True)

            # self.factors[self.coords[:, 1] / 1e3 < self.band_edges[0]] = -np.inf

        else:
            self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
            self.inds = self.xp.ones(self.coords.shape[:-1], dtype=bool)
            self.factors = self.xp.ones_like(self.inds)
            tmp_inds_shaped = self.orig_inds

        self.num_sources = self.coords.shape[0]
        self.set_main_band_sorter_info(main_band_sorter, inds_main_band_sorter)
        
        self.freqs = self.coords[:, 1] / 1e3
        self.band_inds = self.xp.searchsorted(band_edges, self.freqs, side="right") - 1
        self.max_data_store_size = max_data_store_size
        
        self.temp_inds = self.xp.repeat(self.xp.arange(self.ntemps), self.nwalkers * self.nleaves_max).reshape(self.ntemps, self.nwalkers, self.nleaves_max)[tmp_inds_shaped]
        self.walker_inds = self.xp.tile(self.xp.arange(self.nwalkers), (self.ntemps, self.nleaves_max, 1)).transpose((0, 2, 1))[tmp_inds_shaped]
        self.leaf_inds = self.xp.tile(self.xp.arange(self.nleaves_max), ((self.ntemps, self.nwalkers, 1)))[tmp_inds_shaped]
        self.special_band_inds = self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds)
        
        self.orig_temp_inds = self.temp_inds.copy()
        self.orig_walker_inds = self.walker_inds.copy()
        self.orig_leaf_inds = self.leaf_inds.copy()
        self.orig_special_band_inds = self.special_band_inds.copy()
        self.orig_band_inds = self.band_inds.copy()
        self.transform_fn = transform_fn

    def set_main_band_sorter_info(self, main_band_sorter, inds_main_band_sorter):
        if main_band_sorter is None:
            self.inds_main_band_sorter = self.xp.arange(self.num_sources)
        else:
            self.inds_main_band_sorter = inds_main_band_sorter

        self.main_band_sorter = main_band_sorter

    @property
    def coords_in(self) -> np.ndarray:
        return self.transform_fn.both_transforms(self.coords, xp=self.xp)

    def get_special_band_index(self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray) -> np.ndarray:
        special_indices = (temp_inds * self.nwalkers + walker_inds) * int(1e6) + band_inds
        return special_indices

    def get_separate_inds_from_special_index(self, special_band_inds: np.ndarray) -> tuple:
        temp_walker_inds_now = cp.floor(special_band_inds / 1e6).astype(int)
        temp_inds_now = temp_walker_inds_now // self.nwalkers
        walker_inds_now = temp_walker_inds_now % self.nwalkers
        band_inds_now = (special_band_inds - temp_walker_inds_now * int(1e6)).astype(int)
        return (temp_inds_now, walker_inds_now, band_inds_now)

    @property
    def special_index_check(self) -> bool:
        return self.xp.all(self.special_band_inds == self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds))

    @property
    def N_vals(self) -> np.ndarray:
        return self.band_N_vals[self.band_inds]
        
    @property
    def unique_N(self) -> np.ndarray:
        return self.xp.unique(self.N_vals)

    def get_subset(
        self,
        *args, 
        **kwargs
    ):
        subset_inds = self.get_subset_inds(*args, **kwargs)

        if len(subset_inds) == 0:
            return None

        # source information
        subset = BandSorter(
            self, inds_subset=subset_inds, main_band_sorter=self.main_band_sorter, inds_main_band_sorter=self.inds_main_band_sorter[subset_inds]
        )
        # band information
        return subset

    def get_subset_inds(self, *args, **kwargs):
        subset_bool = self.get_subset_bool(*args, **kwargs)
        return self.xp.arange(len(subset_bool))[subset_bool]

    def get_subset_bool(
        self, 
        units: Optional[int] = None,
        remainder: Optional[int] = None,
        temp: Optional[int] = None,
        walker: Optional[int] = None,
        leaf: Optional[int] = None,
        band: Optional[int] = None,
        extra_bool: Optional[np.ndarray] = None,
        full_bool:Optional[np.ndarray] = None
    ) -> np.ndarray:

        inds_keep = self.xp.ones_like(self.band_inds, dtype=bool)

        if full_bool is None:
            if band is not None:
                assert isinstance(band, int)
                inds_keep &= (self.band_inds == band)
            elif units is not None or remainder is not None:
                assert units is not None and remainder is not None
                inds_keep &= (self.band_inds % units == remainder)
            
            # TODO: what to do about this
            # inds_keep &= (self.band_inds < len(self.band_edges) - 2)
            # inds_keep &= (self.band_inds > 1)

            if temp is not None:
                assert isinstance(temp, int)
                inds_keep &= (self.temp_inds == temp)
            if walker is not None:
                assert isinstance(walker, int)
                inds_keep &= (self.walker_inds == walker)
            if leaf is not None:
                assert isinstance(temp, int)
                inds_keep &= (self.leaf_inds == leaf)

            if extra_bool is not None:
                assert isinstance(extra_bool, self.xp.ndarray)
                assert extra_bool.shape == (self.num_sources,)
                inds_keep &= extra_bool
        else:
            assert full_bool.shape[0] == self.num_sources
            inds_keep = full_bool
            
        return inds_keep

    @property
    def main_band_sorter(self):
        main_band_sorter = self if self._main_band_sorter is None else self._main_band_sorter
        return main_band_sorter
    
    @main_band_sorter.setter
    def main_band_sorter(self, main_band_sorter):
        self._main_band_sorter = main_band_sorter

    def get_buffer(self, acs, special_indices_unique, special_indices_index, now_bool_full, num_band_preload : int = None, **kwargs) -> Buffer:

        if num_band_preload is None:
            num_band_preload = len(special_indices_unique)
        
        special_indices_unique, special_indices_index = (special_indices_unique[:num_band_preload], special_indices_index[:num_band_preload])
  
        # CAN USE main_band_sorter TO GET SOURCES IN BANDS OF INTEREST THAT ARE NOT CURRENTLY OF INTEREST THEMSELVES

        # TODO: check the end of this line, is this covered ??
        sources_now_map = cp.arange(self.main_band_sorter.special_band_inds.shape[0])[cp.in1d(self.main_band_sorter.special_band_inds, special_indices_unique) & now_bool_full]
        
        # NOTE: self.main_band_sorter.inds needed to only inject real sources
        # inject sources must include sources that have been turned off in these bands
        sources_inject_now_map = cp.arange(self.main_band_sorter.special_band_inds.shape[0])[cp.in1d(self.main_band_sorter.special_band_inds, special_indices_unique) & self.main_band_sorter.inds]
        
        # separate out inds
        temp_inds_now, walker_inds_now, band_inds_now = self.get_separate_inds_from_special_index(special_indices_unique)
        
        all_unique_band_combos = cp.asarray([temp_inds_now, walker_inds_now, band_inds_now]).T
        num_bands_here_total = all_unique_band_combos.shape[0]
        num_bands_now = special_indices_unique.shape[0]
        
        # sort these sources by band
        inds_sort_tmp = cp.argsort(self.main_band_sorter.special_band_inds[sources_now_map])
        sources_now_map[:] = sources_now_map[inds_sort_tmp]
        special_indices_now = self.main_band_sorter.special_band_inds[sources_now_map].copy()
        uni_info_now = cp.unique(special_indices_now, return_index=True, return_counts=True, return_inverse=True) 
        unique_special_indices_now, unique_special_indices_now_index, unique_special_indices_now_inverse, unique_special_indices_now_counts = uni_info_now
        points_curr_tmp = self.main_band_sorter.coords[sources_now_map].copy()
        special_indices_unique_sort = cp.argsort(special_indices_unique)
        index_info = (special_indices_unique_sort, special_indices_unique)
        if all_unique_band_combos.min() < 0:
            breakpoint()
        buffer_obj = Buffer(self.gb, self.band_edges, all_unique_band_combos, points_curr_tmp, num_bands_now, acs.nchannels, self.max_data_store_size, uni_info_now, self.transform_fn, self.waveform_kwargs, acs.df, sources_now_map, sources_inject_now_map, index_info, self.main_band_sorter.special_band_inds[sources_now_map], **kwargs)
        
        buffer_obj.fill_buffer_residual_and_psd_from_acs(acs)
        buffer_obj.acs = acs
        # includes sources in these sub-bands that are no longer getting proposals
        inj_inds_sort_tmp = cp.argsort(self.main_band_sorter.special_band_inds[sources_inject_now_map])
        sources_inject_now_map[:] = sources_inject_now_map[inj_inds_sort_tmp]
        inj_special_indices_now = self.main_band_sorter.special_band_inds[sources_inject_now_map].copy()
        inj_unique_special_indices_now, inj_unique_special_indices_now_index, inj_unique_special_indices_now_inverse, inj_unique_special_indices_now_counts = cp.unique(inj_special_indices_now, return_index=True, return_counts=True, return_inverse=True) 
        
        coords_to_inject = self.main_band_sorter.coords[sources_inject_now_map].copy()

        inject_index = buffer_obj.get_index(inj_special_indices_now)
        inject_N_vals = self.band_N_vals[self.main_band_sorter.band_inds[sources_inject_now_map]].copy()
        
        if len(inject_index) != len(coords_to_inject):
            breakpoint()

        inj_args = (coords_to_inject, inject_index, inject_N_vals)
        if buffer_obj.use_template_arr:
            buffer_obj.add_sources_to_template_buffer(*inj_args)
        else:
            buffer_obj.add_sources_to_band_buffer(*inj_args)

        return buffer_obj

    def get_band_info(self):

        uni_special, uni_special_counts =  cp.unique(self.special_band_inds[self.inds], return_counts=True)
        uni_temp_inds, uni_walker_inds, uni_band_inds = self.get_separate_inds_from_special_index(uni_special)
        
        num_bands = len(self.band_edges) - 1
        band_counts = np.zeros((self.ntemps, self.nwalkers, num_bands), dtype=int)
        band_counts[uni_temp_inds.get(), uni_walker_inds.get(), uni_band_inds.get()] = uni_special_counts.get()
       
        return {"band_counts": band_counts}



# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialBase(GlobalFitMove, GroupStretchMove, Move):
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
        parameter_transforms=None,
        snr_lim=1e-10,
        rj_proposal_distribution=None,
        num_repeat_proposals=1,
        name=None,
        use_prior_removal=False,
        phase_maximize=False,
        ranks_needed=0,
        gpus=[],
        num_band_preload = 20000,
        run_swaps = True, 
        # TODO: make this adjustable?
        max_data_store_size = 6000,
        **kwargs
    ):
        # return_gpu is a kwarg for the stretch move
        GroupStretchMove.__init__(self, *args, return_gpu=True, **kwargs)
        self.ranks_needed = ranks_needed
        self.gpus = gpus
        self.gpu_priors = gpu_priors
        self.name = name
        self.num_repeat_proposals = num_repeat_proposals
        self.num_band_preload = num_band_preload
        self.band_preload_size = self.max_data_store_size = max_data_store_size
        self.use_prior_removal = use_prior_removal
        self.has_setup_group = False
        # for key in priors:
        #     if not isinstance(priors[key], ProbDistContainer) and not isinstance(priors[key], GBPriorWrap):
        #         raise ValueError(
        #             "Priors need to be eryn.priors.ProbDistContainer object."
        #         )
        
        self.priors = priors
        self.gb = gb
        self.stop_here = True
        self.run_swaps = run_swaps

        # args = [priors["gb"].priors_in[(0, 1)].rho_star]
        # args += [priors["gb"].priors_in[(0, 1)].frequency_prior.min_val, priors["gb"].priors_in[(0, 1)].frequency_prior.max_val]
        # for i in range(2, 8):
        #     args += [priors["gb"].priors_in[i].min_val, priors["gb"].priors_in[i].max_val]
        
        # self.gpu_cuda_priors = self.gb.pyPriorPackage(*tuple(args))
        # self.gpu_cuda_wrap = self.gb.pyPeriodicPackage(2 * np.pi, np.pi, 2 * np.pi)

        # use gpu from template generator
        # self.use_gpu = gb.use_gpu
        if self.use_gpu:
            self.mempool = self.xp.get_default_memory_pool()

        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.parameter_transforms = parameter_transforms
        self.fd = fd
        self.df = (fd[1] - fd[0]).item()
        self.mgh = mgh
        self.phase_maximize = phase_maximize

        self.snr_lim = snr_lim

        self.band_edges = self.xp.asarray(self.band_edges)

        self.rj_proposal_distribution = rj_proposal_distribution
        self.is_rj_prop = self.rj_proposal_distribution is not None
        if self.is_rj_prop:
            if (self.num_repeat_proposals != 1):
                print("Adjusting repeat proposals to 1 for RJ.")

            self.num_repeat_proposals = 1

        # setup N vals for bands
        band_mean_f = (self.band_edges[1:] + self.band_edges[:-1]).get() / 2
        self.band_N_vals = cp.asarray(get_N(np.full_like(band_mean_f, 1e-30), band_mean_f, self.waveform_kwargs["T"], self.waveform_kwargs["oversample"]))

    def setup(self, model, branches):
        return 

    def adjust_sources_in_residual_buffer(self, factor, model, band_sorter: BandSorter, *args, **kwargs) -> None:
        
        assert isinstance(factor, int) and (factor == -1 or factor == +1)

        subset = band_sorter.get_subset(*args, **kwargs)

        if subset is None or subset.inds.sum().item() == 0:
            return

        factors_tmp = factor * cp.ones_like(subset.walker_inds[subset.inds], dtype=float)
        self.gb.generate_global_template(
            subset.coords_in[subset.inds],
            subset.walker_inds[subset.inds].astype(self.xp.int32),
            model.analysis_container_arr.linear_data_arr,
            data_length=model.analysis_container_arr.data_length,
            factors=factors_tmp,
            data_splits=model.analysis_container_arr.gpu_map,
            N=subset.N_vals[subset.inds],
            **self.waveform_kwargs,
        )

    def remove_cold_chain_sources_from_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        self.remove_sources_from_residual(*args, **kwargs)

    def remove_sources_from_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(+1, *args, **kwargs)
       
    def add_cold_chain_sources_to_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        self.add_sources_to_residual(*args, **kwargs)

    def add_sources_to_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(-1, *args, **kwargs)

    def run_proposal(self, model, state, band_sorter, band_temps):
        num_proposals_here = self.num_repeat_proposals if not self.is_rj_prop else 1
        source_prop_counter = cp.zeros(band_sorter.coords.shape[0], dtype=int)
        
        ll_change_log = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))
        total_keep = 0
        units = 2 if not self.is_rj_prop else 2
        if self.num_bands == 1:
            units = 1

        # random start to rotation around 
        start_unit = model.random.randint(units)

        for tmp in range(units):
            # continue
            remainder = (start_unit + tmp) % units
            if self.num_bands == 1:
                remainder = 0

            # add back in all sources in the cold-chain 
            # residual from this group
            self.remove_cold_chain_sources_from_residual(model, band_sorter, units=units, remainder=remainder)
            # keep1 = (
            #     (band_indices % units == remainder) 
            #     & (band_indices < len(self.band_edges) - 2)
            #     & (band_indices > 1)
            #     & (self.band_N_vals[band_indices] < 1024)  # TESTING
            # ) 
            subset_of_interest = band_sorter.get_subset(units=units, remainder=remainder, extra_bool=(source_prop_counter < num_proposals_here))

            if subset_of_interest is None:
                continue
                
            # start all false, then highlight sources of interest
            sources_of_interest = self.xp.zeros_like(source_prop_counter, dtype=bool)
            sources_of_interest[subset_of_interest.inds_main_band_sorter] = True
            iteration_num = 0

            # with open("tmp.dat", "w") as fp:
            #     tmp = f"{iteration_num}, {sources_of_interest.sum()}\n"
            #     fp.write(tmp)
            #     print(tmp)

            # TODO: move sources of interest inside? I do not think so right now
            while np.any(sources_of_interest):
                # st_1 = time.perf_counter()
                # MAKE THIS INTO A GENERATOR
                special_indices_unique, special_indices_index = cp.unique(band_sorter.special_band_inds[sources_of_interest], return_index=True)
                now_bool_full = source_prop_counter < num_proposals_here
                if not cp.any(now_bool_full):
                    continue

                buffer_obj = subset_of_interest.get_buffer(model.analysis_container_arr, special_indices_unique, special_indices_index, now_bool_full, num_band_preload=self.num_band_preload)
                
                # with open("tmp.dat", "a") as fp:
                #     tmp = f"inject: {iteration_num}, {sources_of_interest.sum()}"
                #     fp.write(tmp + "\n")
                #     print(tmp)
                # et_1 = time.perf_counter()
                # print("1st:", et_1 - st_1)
                
                for move_i in range(num_proposals_here):
                    # st_1 = time.perf_counter()
                    choice_fraction = cp.random.rand(buffer_obj.num_bands_now)
                    
                    sources_picked_for_update = buffer_obj.uni_special_index + cp.floor(choice_fraction * buffer_obj.uni_special_counts).astype(int)

                    params_to_update = buffer_obj.params_interest[sources_picked_for_update].copy()
                    # make sure periodic parameters are wrapped
                    params_to_update[:] = self.periodic.wrap({"gb": params_to_update[:, None, :]}, xp=self.xp)["gb"][:, 0]
                    
                    inds_to_update = buffer_obj.sources_now_map[sources_picked_for_update].copy()
                    data_index_to_update = buffer_obj.now_index[sources_picked_for_update].copy()
                    # map is back to full band and coords
                    map_to_update = (band_sorter.temp_inds[inds_to_update], band_sorter.walker_inds[inds_to_update], band_sorter.band_inds[inds_to_update])
                    map_to_update_cpu = (band_sorter.temp_inds[inds_to_update].get(), band_sorter.walker_inds[inds_to_update].get(), band_sorter.band_inds[inds_to_update].get())
                    
                    if not self.is_rj_prop:
                        # custom group stretch
                        # TODO: work into main group stretch somehow
                        params_into_proposal = params_to_update[None, :, None, :]

                        friends_into_proposal = state.branches_supplemental["gb"][map_to_update_cpu]["friend_start_inds"][None, :, None]
                        branch_supps_into_proposal = BranchSupplemental({"friend_start_inds": friends_into_proposal}, base_shape=friends_into_proposal.shape)
                        inds_into_proposal = self.xp.ones(params_into_proposal.shape[:-1], dtype=bool)

                        # TODO: check detailed balance
                        q, update_factors = self.get_proposal({"gb": params_into_proposal}, model.random, s_inds_all={"gb": inds_into_proposal}, cp=self.xp, return_gpu=True, branch_supps=branch_supps_into_proposal)         
                        new_coords = q["gb"][0, :, 0, :]
                        
                        new_coords[:] = self.periodic.wrap({"gb": new_coords[:, None, :]}, xp=self.xp)["gb"][:, 0]
                    
                        prev_logp = cp.asarray(self.gpu_priors["gb"].logpdf(params_to_update))  # , psds=self.mgh.psd_shaped[0][0], walker_inds=curr_index)
                        curr_logp = cp.asarray(self.gpu_priors["gb"].logpdf(new_coords))  # , psds=self.mgh.psd_shaped[0][0], walker_inds=curr_index)

                    else:
                        
                        logp_tmp = cp.asarray(self.gpu_priors["gb"].logpdf(params_to_update))
                        prev_logp = cp.zeros_like(logp_tmp)
                        curr_logp = cp.zeros_like(logp_tmp)

                        new_coords = params_to_update.copy()
                        
                        inds = band_sorter.inds[inds_to_update].copy()
                        update_factors = band_sorter.factors[inds_to_update].copy()
                        params_to_update[~inds, 0] = np.log(1e-80)
                        new_coords[inds, 0] = np.log(1e-80)
                        # wrap in case
                        new_coords[:] = self.periodic.wrap({"gb": new_coords[:, None, :]}, xp=self.xp)["gb"][:, 0]
                    
                        prev_logp[inds] = logp_tmp[inds]
                        curr_logp[~inds] = logp_tmp[~inds]

                    if cp.any(cp.isinf(prev_logp)):
                        breakpoint()
                    # inputs into swap proposal
                    
                    # guard on the edges with too-large frequency proposals out of band that would not be physical
                    curr_logp[(new_coords[:, 1] / 1e3 < buffer_obj.frequency_lims[0]) | (new_coords[:, 1] / 1e3 > buffer_obj.frequency_lims[1])] = -np.inf
                    # TODO: 2 vs 4?
                    curr_logp[(cp.abs(params_to_update[:, 1] / 1e3 - new_coords[:, 1] / 1e3) / self.df).astype(int) > (self.band_N_vals[band_sorter.band_inds[inds_to_update]] / 4).astype(int)] = -np.inf

                    ll_diff = cp.full_like(prev_logp, -1e300)
                    opt_snr = cp.full_like(prev_logp, 0.0)
                    keep2 = ~cp.isinf(curr_logp)
                    # et_1 = time.perf_counter()
                    # print("2nd:", et_1 - st_1)
                
                    # st_1 = time.perf_counter()
                    params_remove = params_to_update[keep2].copy()
                    params_add = new_coords[keep2].copy()

                    # data indexes align with the buffers (1 per buffer except for inf priors)
                    data_index = data_index_to_update[keep2].astype(np.int32)
                    swap_N_vals = self.band_N_vals[band_sorter.band_inds[inds_to_update[keep2]]].copy()

                    # CANNOT COPY PARAMETER ARRAYS, IN PLACE ADJUSTMENT IF PHASE MAXIMIZING
                    ll_diff[keep2] = buffer_obj.get_swap_ll(params_remove, params_add, data_index, swap_N_vals, phase_maximize=self.phase_maximize)

                    # in case there is phase marginalization, need to adjust in new_coords
                    if self.phase_maximize:
                        new_coords[keep2] = params_add[:]

                    curr_beta = band_temps[map_to_update[0], map_to_update[2]]
                    # print("change priors?, need to adjust here")
                    
                    delta_logP = curr_beta * ll_diff + (curr_logp - prev_logp)
                    lnpdiff = delta_logP + update_factors.squeeze()
                    accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))
                    # need to copy to old array before changing in place
                    old_params_to_update = params_to_update.copy()
                    params_to_update[accept] = new_coords[accept]
                    buffer_obj.params_interest[sources_picked_for_update] = params_to_update[:]
                    
                    if cp.any(accept):
                        inds_update_accept = inds_to_update[accept]
                        
                        ll_accept = ll_diff[accept]
                        if self.is_rj_prop:
                            # update indsf
                            band_sorter.inds[inds_update_accept] = (~band_sorter.inds[inds_update_accept])
                        
                        temp_inds_accept = band_sorter.temp_inds[inds_update_accept]
                        walker_inds_accept = band_sorter.walker_inds[inds_update_accept]
                        band_inds_accept = band_sorter.band_inds[inds_update_accept]
                        ll_change_log[temp_inds_accept, walker_inds_accept, band_inds_accept] += ll_accept

                        # switch accepted waveform
                        old_coords_for_change = old_params_to_update[accept].copy()
                        new_coords_for_change = new_coords[accept].copy()

                        old_change_index = data_index_to_update[accept].copy().astype(np.int32)
                        new_change_index = old_change_index.copy()
                        
                        old_change_N_vals = self.band_N_vals[band_sorter.band_inds[inds_to_update[accept]]].copy()
                        new_change_N_vals = old_change_N_vals.copy()

                        # TODO: should we combine this to make faster
                        buffer_obj.remove_sources_from_band_buffer(old_coords_for_change, old_change_index, old_change_N_vals)
                        buffer_obj.add_sources_to_band_buffer(new_coords_for_change, new_change_index, new_change_N_vals)
                        
                    # print(iteration_num, move_i)
                    self.mempool.free_all_blocks()
                    source_prop_counter[inds_to_update] += 1

                    # with open("tmp.dat", "a") as fp:
                    #     tmp = f"move {move_i}: {iteration_num}, {sources_of_interest.sum()}"
                    #     fp.write(tmp + "\n")
                    #     print(tmp)
                    # will recalculate prior anyways so leaving that out

                    # change WAVEFORMS THAT HAVE BEEN ACCEPTED

                    # et_1 = time.perf_counter()
                    # print("3rd:", et_1 - st_1)
                
                band_sorter.coords[buffer_obj.sources_now_map] = buffer_obj.params_interest[:]
                # inds change is taken care of inplace
                sources_of_interest[sources_of_interest] = (source_prop_counter[sources_of_interest] < self.num_repeat_proposals)
                iteration_num += 1
                # with open("tmp.dat", "a") as fp:
                #     tmp = f"{iteration_num}, {sources_of_interest.sum()}"
                #     fp.write(tmp + "\n")
                #     print(tmp)
                self.mempool.free_all_blocks()
                # update prop counter
            
            # add back in all sources in the cold-chain 
            # residual from this group
            self.add_cold_chain_sources_to_residual(model, band_sorter, units=units, remainder=remainder)

            self.xp.cuda.runtime.deviceSynchronize()
        
        return ll_change_log

    def run_tempering(self, model, state, band_sorter, band_temps):
        ll_change_log_temp = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))

        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        current_band_counts = cp.zeros((len(self.band_edges) - 1, self.ntemps), dtype=int)

        units = 2
        tmp_start = np.random.randint(units)
        for tmp in range(units):
            remainder = (tmp_start + tmp) % units
            start = remainder
            self.remove_cold_chain_sources_from_residual(model, band_sorter, units=units, remainder=remainder)                

            num_bands_unit = np.arange(self.num_bands)[start::2].shape[0]
            
            walkers_permuted = cp.asarray([cp.random.permutation(cp.arange(self.nwalkers)) for _ in range(self.ntemps * self.num_bands)]).reshape(self.num_bands, self.ntemps, self.nwalkers).transpose(0, 2, 1)[start::2]
            temp_index = cp.repeat(cp.arange(self.ntemps), self.num_bands * self.nwalkers).reshape(self.ntemps, self.num_bands, self.nwalkers).transpose(1, 2, 0)[start::2]
            band_index = cp.repeat(cp.arange(self.num_bands), self.ntemps * self.nwalkers).reshape(self.num_bands, self.ntemps, self.nwalkers).transpose(0, 2, 1)[start::2]
            special_index = band_sorter.get_special_band_index(temp_index, walkers_permuted, band_index)
            
            num_bands_preload_temp = 200
            num_bands_run = 0
            while num_bands_run < self.nwalkers * num_bands_unit:
                start_ind = num_bands_run
                end_ind = start_ind + num_bands_preload_temp

                band_inds_now = band_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                temp_inds_now = temp_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                walker_inds_now = walkers_permuted.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                special_inds_now = special_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                num_bands_now = band_inds_now.shape[0]
                special_inds_now_flat = special_inds_now.flatten()

                # need to include inds
                now_bool_full = cp.in1d(band_sorter.special_band_inds, special_inds_now_flat)  # & band_sorter.inds
                if not cp.any(now_bool_full):
                    num_bands_run += num_bands_preload_temp
                    # print("num bands", num_bands_run)
                    continue

                _, special_inds_index = cp.unique(band_sorter.special_band_inds[now_bool_full], return_index=True)
                
                buffer_obj = band_sorter.get_buffer(model.analysis_container_arr, special_inds_now_flat, special_inds_index, now_bool_full, num_band_preload=num_bands_now * self.ntemps, use_template_arr=True)

                current_lls = buffer_obj.likelihood(source_only=True).reshape(-1, self.ntemps)
                band_combo_map = buffer_obj.unique_band_combos.reshape(-1, self.ntemps, 3)
                current_lls_orig = current_lls.copy()
                # TODO: CHECK LIKELIHOODS/
                for t in range(self.ntemps)[1:][::-1]:
                    st = time.perf_counter()
                    i1 = t
                    i2 = t - 1

                    buffer_i1 = cp.arange(buffer_obj.num_bands_now)[i1::self.ntemps]
                    buffer_i2 = cp.arange(buffer_obj.num_bands_now)[i2::self.ntemps]

                    # IMPORTANT: MAPPING IMPLICITLY UNDERSTANDS WHERE THINGS WILL BE
                    tmp_i1 = buffer_obj.template_buffer[buffer_i1].copy()
                    buffer_obj.template_buffer[buffer_i1] = buffer_obj.template_buffer[buffer_i2]
                    buffer_obj.template_buffer[buffer_i2] = tmp_i1[:]

                    # TODO: add indices because not every likelihood is needed
                    new_lls = buffer_obj.likelihood(source_only=True).reshape(-1, self.ntemps)[:, i2:i1 + 1]
                    old_lls = current_lls[:, i2:i1 + 1]
                    
                    beta1 = band_temps[(band_inds_now[:, 0], i1)]
                    beta2 = band_temps[(band_inds_now[:, 0], i2)]

                    paccept = beta1 * (new_lls[:, 0] - old_lls[:, 1]) + beta2 * (new_lls[:, 1] - old_lls[:, 0])
                    # paccept = bi * (band_here_i1->swapped_like - band_here_i->current_like) + bi1 * (band_here_i->swapped_like - band_here_i1->current_like);

                    raccept = cp.log(cp.random.uniform(size=paccept.shape))
                    sel = paccept > raccept

                    current_lls[sel, i2:i1 + 1] = new_lls[sel]
                    # reverse not accepted ones
                    
                    buffer_i1_reject = buffer_i1[~sel]
                    buffer_i2_reject = buffer_i2[~sel]

                    tmp_i1 = buffer_obj.template_buffer[buffer_i1_reject].copy()
                    buffer_obj.template_buffer[buffer_i1_reject] = buffer_obj.template_buffer[buffer_i2_reject]
                    buffer_obj.template_buffer[buffer_i2_reject] = tmp_i1[:]
                    
                    band_swaps_accepted[band_inds_now[:, 0], i2] += sel.astype(int)
                    band_swaps_proposed[band_inds_now[:, 0], i2] += 1
                    
                    band_inds_exchange_i1 = band_inds_now[sel, i1]
                    walker_inds_exchange_i1 = walker_inds_now[sel, i1]
                    band_inds_exchange_i2 = band_inds_now[sel, i2]
                    walker_inds_exchange_i2 = walker_inds_now[sel, i2]
                    
                    special_ind_test_1 = band_sorter.get_special_band_index(i1, walker_inds_exchange_i1, band_inds_exchange_i1)
                    special_ind_test_2 = band_sorter.get_special_band_index(i2, walker_inds_exchange_i2, band_inds_exchange_i2)

                    # temp_indices[fix_1] = i2
                    # temp_indices[fix_2] = i1

                    ind_sort_1 = cp.argsort(special_ind_test_1.flatten())
                    ind_keep_1 = cp.in1d(band_sorter.special_band_inds, special_ind_test_1)
                    sorted_map_1 = cp.searchsorted(special_ind_test_1[ind_sort_1], band_sorter.special_band_inds[ind_keep_1], side="left")
                    
                    ind_sort_2 = cp.argsort(special_ind_test_2.flatten())
                    ind_keep_2 = cp.in1d(band_sorter.special_band_inds, special_ind_test_2)
                    sorted_map_2 = cp.searchsorted(special_ind_test_2[ind_sort_2], band_sorter.special_band_inds[ind_keep_2], side="left")
                    
                    band_sorter.special_band_inds[ind_keep_1] = special_ind_test_2[ind_sort_1[sorted_map_1]]
                    band_sorter.temp_inds[ind_keep_1] = i2
                    band_sorter.walker_inds[ind_keep_1] = walker_inds_exchange_i2[ind_sort_1[sorted_map_1]]
                    # do not need to change band index but check it
                    assert cp.all(band_sorter.band_inds[ind_keep_1] == band_inds_exchange_i2[ind_sort_1[sorted_map_1]])
                    
                    band_sorter.special_band_inds[ind_keep_2] = special_ind_test_1[ind_sort_2[sorted_map_2]]
                    band_sorter.temp_inds[ind_keep_2] = i1
                    band_sorter.walker_inds[ind_keep_2] = walker_inds_exchange_i1[ind_sort_2[sorted_map_2]]
                    
                    et = time.perf_counter()
                    # print(et - st, t, num_bands_run, self.nwalkers * num_bands_unit)

                diffs = current_lls - current_lls_orig
                # TODO: this should be = not += (?)
                ll_change_log_temp[(buffer_obj.unique_band_combos[:, 0], buffer_obj.unique_band_combos[:, 1], buffer_obj.unique_band_combos[:, 2])] = diffs.flatten()
                num_bands_run += num_bands_preload_temp

            self.add_cold_chain_sources_to_residual(model, band_sorter, units=units, remainder=remainder)                

        # adapt if desired
        print("change adaptation")
        if self.time > 50:
            ratios = (band_swaps_accepted / band_swaps_proposed).T
            betas0 = band_temps.copy().T
            betas1 = betas0.copy()

            # Modulate temperature adjustments with a hyperbolic decay.
            decay = self.temperature_control.adaptation_lag / (self.time + self.temperature_control.adaptation_lag)
            kappa = decay / self.temperature_control.adaptation_time

            # Construct temperature adjustments.
            dSs = kappa * (ratios[:-1] - ratios[1:])

            # Compute new ladder (hottest and coldest chains don't move).
            deltaTs = cp.diff(1 / betas1[:-1], axis=0)

            deltaTs *= cp.exp(dSs)
            betas1[1:-1] = 1 / (cp.cumsum(deltaTs, axis=0) + 1 / betas1[0])

            # Don't mutate the ladder here; let the client code do that.
            dbetas = betas1 - betas0

            band_temps += self.xp.asarray(dbetas.T)

        print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum_temp = ll_change_log_temp.sum(axis=-1)

        return ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`GFState`): Current state of the sampler.

        Returns:
            :class:`GFState`: GFState of sampler after proposal is complete.

        """

        st_all = time.perf_counter()

        self.xp.cuda.runtime.setDevice(model.analysis_container_arr.gpus[0])
        nchannels = model.analysis_container_arr.nchannels
        data_length = model.analysis_container_arr.data_length
        self.fd = model.analysis_container_arr.f_arr.copy()
        self.df = model.analysis_container_arr.df
        self.current_state = state
        np.random.seed(10)
        # print("start stretch")

        # Check that the dimensions are compatible.
        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb"].shape

        if not self.is_rj_prop and not np.any(state.branches["gb"].inds):
            return state, np.zeros((ntemps, nwalkers), dtype=bool)
        
        self.nwalkers = nwalkers
        
        # Run any move-specific setup.
        self.setup(model, state.branches)

        new_state = GFState(state, copy=True)
        band_temps = cp.asarray(state.sub_states["gb"].band_info["band_temps"].copy())

        if self.is_rj_prop:
            orig_store = new_state.log_like[0].copy()

        gb_coords = cp.asarray(new_state.branches["gb"].coords)

        self.mempool.free_all_blocks()

        waveform_kwargs_now = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs_now:
            waveform_kwargs_now.pop("N")
        # waveform_kwargs_now["start_freq_ind"] = self.start_freq_ind

        # if self.is_rj_prop:
        #     print("START:", new_state.log_like[0])
        log_like_tmp = self.xp.asarray(new_state.log_like)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)

        self.mempool.free_all_blocks()

        gb_inds = self.xp.asarray(new_state.branches["gb"].inds)
        gb_inds_orig = gb_inds.copy()

        data = model.analysis_container_arr.linear_data_arr
        psd = model.analysis_container_arr.linear_psd_arr

        # do unique for band size as separator between asynchronous kernel launches
        # band_indices = self.xp.asarray(new_state.branches["gb"].branch_supplemental.holder["band_inds"])
        band_indices = self.xp.searchsorted(self.band_edges, cp.asarray(new_state.branches["gb"].coords[:, :, :, 1]).flatten() / 1e3, side="right").reshape(new_state.branches["gb"].coords[:, :, :, 1].shape) - 1
            
        # N_vals_in = self.xp.asarray(new_state.branches["gb"].branch_supplemental.holder["N_vals"])
        points_curr = self.xp.asarray(new_state.branches["gb"].coords)
        points_curr_orig = points_curr.copy()
        # N_vals_in_orig = N_vals_in.copy()
        band_indices_orig = band_indices.copy()

        rj_prop = None if not self.is_rj_prop else self.rj_proposal_distribution["gb"]

        # make sure all periodic parameters have been put into their range
        new_state.branches["gb"].coords[:] = self.periodic.wrap({"gb": new_state.branches["gb"].coords[:].reshape(ntemps * nwalkers, nleaves_max, ndim)})["gb"].reshape(ntemps, nwalkers, nleaves_max, ndim)
        
        print("is this okay for rj? I do not think so, check with below use of gb_inds_in")
        band_sorter = BandSorter(new_state.branches["gb"], self.band_edges, self.band_N_vals, use_gpu=self.use_gpu, transform_fn=self.parameter_transforms, max_data_store_size=self.max_data_store_size, gb=self.gb, waveform_kwargs=self.waveform_kwargs, rj_prop=rj_prop)
        
        if self.name == "rj_prior" and self.use_prior_removal:
            band_sorter.factors[~band_sorter.orig_inds.flatten()] = -1e300

        do_synchronize = False
        device = self.xp.cuda.runtime.getDevice()

        ll_after = model.analysis_container_arr.likelihood(source_only=False)  #  - cp.sum(cp.log(cp.asarray(psd[:2])), axis=(0, 2))).get()

        # print(np.abs(new_state.log_like - ll_after).max())
        store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        start_diffs = np.abs(new_state.log_like[0] - ll_after)
        # print("CHECKING 0:", store_max_diff, self.is_rj_prop)
        # self.check_ll_inject(new_state, verbose=True)

        per_walker_band_proposals = cp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_accepted = cp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        
        ll_change_log = self.run_proposal(model, new_state, band_sorter, band_temps)

        print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum = ll_change_log.sum(axis=-1)
        new_state.log_like[0] += ll_change_sum[0].get()

        ll_after = model.analysis_container_arr.likelihood()
        check = ll_after - new_state.log_like[0]
        
        assert np.abs(check).max() < 1e-4

        print("ADD check")
        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        #TODO: move this
        self.nchannels = model.analysis_container_arr.nchannels
        
        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        
        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and self.is_rj_prop
            and self.run_swaps
            # and False
        ):  
            ll_change_sum_temp, band_swaps_accepted, band_swaps_proposed = self.run_tempering(model, new_state, band_sorter, band_temps)
            
            new_state.log_like[0] += ll_change_sum_temp[0].get()

            ll_after = model.analysis_container_arr.likelihood()
            check = ll_after - new_state.log_like[0]
            assert np.abs(check).max() < 1e-4
            self.mempool.free_all_blocks()

        print("make sure this works for rj")
        special_indices_finish = (band_sorter.temp_inds[band_sorter.inds] * nwalkers + band_sorter.walker_inds[band_sorter.inds]) * int(1e6) + band_sorter.coords[band_sorter.inds, 1]
        special_inds_temp_walker = (band_sorter.temp_inds[band_sorter.inds] * nwalkers + band_sorter.walker_inds[band_sorter.inds])
        sorted_inds = cp.argsort(special_indices_finish)

        uni, uni_inds, uni_inverse, uni_counts = cp.unique(special_inds_temp_walker[sorted_inds], return_index=True, return_counts=True, return_inverse=True)

        leaf_inds_new_tmp = cp.arange(special_indices_finish.shape[0]) - uni_inds[uni_inverse]
        leaf_inds_new = cp.zeros_like(leaf_inds_new_tmp)
        leaf_inds_new[sorted_inds] = leaf_inds_new_tmp
        
        print("NEED TO PROPERLY MOVE SUPPLEMENTAL INFO BASED ON OLD LEAVES.")
        inds_new = (band_sorter.temp_inds[band_sorter.inds].get(), band_sorter.walker_inds[band_sorter.inds].get(), leaf_inds_new.get())
        inds_old = (band_sorter.orig_temp_inds[band_sorter.inds].get(), band_sorter.orig_walker_inds[band_sorter.inds].get(), band_sorter.orig_leaf_inds[band_sorter.inds].get())
        new_state.branches["gb"].coords[inds_new] = band_sorter.coords[band_sorter.inds].get()
        new_state.branches["gb"].inds[:] = False
        # turn on all the ones that are there
        new_state.branches["gb"].inds[inds_new] = True

        new_state.branches["gb"].branch_supplemental[inds_new] = state.branches["gb"].branch_supplemental[inds_old]
        et_all = time.perf_counter()
        print(self.name, et_all - st_all)

        # TODO: need to redo the acceptance fraction
        # get accepted fraction
        # # if not self.is_rj_prop:
        # #     accepted_check_tmp = np.zeros_like(
        # #         state.branches_inds["gb"], dtype=bool
        # #     )
        # #     accepted_check_tmp[state.branches_inds["gb"]] = np.all(
        # #         np.abs(
        # #             new_state.branches_coords["gb"][
        # #                 state.branches_inds["gb"]
        # #             ]
        # #             - state.branches_coords["gb"][state.branches_inds["gb"]]
        # #         )
        # #         > 0.0,
        # #         axis=-1,
        # #     )
        # #     proposed = gb_inds.get()
        # #     accepted_check = accepted_check_tmp.sum(
        # #         axis=(1, 2)
        # #     ) / proposed.sum(axis=(1, 2))
        # # else:
        # #     accepted_check_tmp = (
        # #         new_state.branches_inds["gb"] == (~state.branches_inds["gb"])
        # #     )

        # #     proposed = gb_inds.get()
        # #     accepted_check = accepted_check_tmp.sum(axis=(1, 2)) / proposed.sum(axis=(1, 2))
            
        # # manually tell temperatures how real overall acceptance fraction is
        # number_of_walkers_for_accepted = np.floor(nwalkers * accepted_check).astype(int)

        # accepted_inds = np.tile(np.arange(nwalkers), (ntemps, 1))

        # accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        # accepted[accepted_inds < number_of_walkers_for_accepted[:, None]] = True

        # tmp1 = np.all(
        #     np.abs(
        #         new_state.branches_coords["gb"]
        #         - state.branches_coords["gb"]
        #     )
        #     > 0.0,
        #     axis=-1,
        # ).sum(axis=(2,))
        # tmp2 = new_state.branches_inds["gb"].sum(axis=(2,))

        # # add to move-specific accepted information
        # self.accepted += tmp1
        # if isinstance(self.num_proposals, int):
        #     self.num_proposals = tmp2
        # else:
        #     self.num_proposals += tmp2

        new_inds = cp.asarray(new_state.branches_inds["gb"])
        del band_sorter
        self.mempool.free_all_blocks()
        new_band_sorter = BandSorter(new_state.branches["gb"], self.band_edges, self.band_N_vals, use_gpu=self.use_gpu, transform_fn=self.parameter_transforms, max_data_store_size=self.max_data_store_size, gb=self.gb, waveform_kwargs=self.waveform_kwargs)
        
        # in-model inds will not change
        tmp_freqs_find_bands = cp.asarray(new_state.branches_coords["gb"][:, :, :, 1])

        # calculate current band counts
        band_here = (cp.searchsorted(self.band_edges, tmp_freqs_find_bands.flatten() / 1e3, side="right") - 1).reshape(tmp_freqs_find_bands.shape)

        group_temp_finder = [
            cp.repeat(cp.arange(ntemps), nwalkers * nleaves_max).reshape(
                ntemps, nwalkers, nleaves_max
            ),
            cp.tile(cp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose(
                (0, 2, 1)
            ),
            cp.tile(cp.arange(nleaves_max), ((ntemps, nwalkers, 1))),
        ]
        
        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        self.mempool.free_all_blocks()

        self.time += 1
        # self.xp.cuda.runtime.deviceSynchronize()

        band_info = new_band_sorter.get_band_info() 

        new_state.sub_states["gb"].update_band_information(band_temps.get(), per_walker_band_proposals.sum(axis=1).T.get(), per_walker_band_accepted.sum(axis=1).T.get(), band_swaps_proposed.get(), band_swaps_accepted.get(),band_info["band_counts"], self.is_rj_prop)
        # TODO: check rj numbers

        # new_state.log_like[:] = self.check_ll_inject(new_state)

        self.mempool.free_all_blocks()

        # if self.is_rj_prop:
        #     pass  # print(self.name, "2nd count check:", new_state.branches["gb"].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplemental)
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        return new_state, accepted

    def check_ll_inject(self, model, new_state, verbose=False):

        check_ll = model.analysis_container_arr.likelihood().copy()

        nleaves_max = new_state.branches["gb"].shape[-2]
        for i in range(2):
            self.mgh.channel1_data[0][self.nwalkers * self.data_length * i: self.nwalkers * self.data_length * (i + 1)] = self.mgh.channel1_base_data[0][:]
            self.mgh.channel2_data[0][self.nwalkers * self.data_length * i: self.nwalkers * self.data_length * (i + 1)] = self.mgh.channel2_base_data[0][:]
        
        coords_out_gb = new_state.branches["gb"].coords[0, new_state.branches["gb"].inds[0]]
        coords_in_in = self.parameter_transforms.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(self.band_edges.get(), coords_in_in[:, 1], side="right") - 1
        assert np.all(band_inds == new_state.branches["gb"].branch_supplemental.holder["band_inds"][0, new_state.branches["gb"].inds[0]])

        walker_vals = np.tile(np.arange(self.nwalkers), (nleaves_max, 1)).transpose((1, 0))[new_state.branches["gb"].inds[0]]

        data_index_1 = ((band_inds % 2) + 0) * self.nwalkers + walker_vals

        data_index = cp.asarray(data_index_1).astype(cp.int32)

        # goes in as -h
        factors = -cp.ones_like(data_index, dtype=cp.float64)

        waveform_kwargs_tmp = self.waveform_kwargs.copy()

        N_vals = self.band_N_vals[band_inds]
        self.gb.generate_global_template(
            coords_in_in,
            data_index,
            self.mgh.data_list,
            batch_size=1000,
            data_length=self.data_length,
            factors=factors,
            N=N_vals,
            data_splits=self.mgh.gpu_splits,
            **waveform_kwargs_tmp,
        )

        check_ll_new = self.mgh.get_ll(include_psd_info=True)
        check_ll_diff1 = check_ll_new - check_ll
        if verbose:
            print(check_ll_diff1)

        # breakpoint()
        return check_ll_new

    @property
    def ranks_needed(self): 
        if not hasattr(self, "_ranks_needed"):
            raise ValueError("Need to set ranks needed for this class.")

        return self._ranks_needed

    @ranks_needed.setter
    def ranks_needed(self, ranks_needed):
        assert isinstance(ranks_needed, int)
        self._ranks_needed = ranks_needed
    

class GBSpecialStretchMove(GBSpecialBase):
    def setup_gbs(self, model, branch):
        st = time.perf_counter()
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplemental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        all_remaining_freqs = coords[0][inds[0]][:, 1]

        all_remaining_cords = coords[0][inds[0]]

        num_remaining = len(all_remaining_freqs)

        all_temp_fs = self.xp.asarray(coords[inds][:, 1])

        # TODO: improve this?
        self.inds_freqs_sorted = self.xp.asarray(np.argsort(all_remaining_freqs))
        self.freqs_sorted = self.xp.asarray(np.sort(all_remaining_freqs))
        self.all_coords_sorted = self.xp.asarray(all_remaining_cords)[
            self.inds_freqs_sorted
        ]

        total_binaries = inds.sum().item()
        still_going = cp.ones(total_binaries, dtype=bool) 
        inds_zero = cp.searchsorted(self.freqs_sorted, all_temp_fs, side="right") - 1
        left_inds = inds_zero - int(self.nfriends / 2)
        right_inds = inds_zero + int(self.nfriends / 2) - 1

        # do right first here
        right_inds[left_inds < 0] = self.nfriends - 1
        left_inds[left_inds < 0] = 0
        
        # do left first here
        left_inds[right_inds > len(self.freqs_sorted) - 1] = len(self.freqs_sorted) - self.nfriends
        right_inds[right_inds > len(self.freqs_sorted) - 1] = len(self.freqs_sorted) - 1

        assert np.all(right_inds - left_inds == self.nfriends - 1)
        assert not np.any(right_inds < 0) and not np.any(right_inds > len(self.freqs_sorted) - 1) and not np.any(left_inds < 0) and not np.any(left_inds > len(self.freqs_sorted) - 1)
        
        jjj = 0
        while np.any(still_going):
            distance_left = np.abs(all_temp_fs[still_going] - self.freqs_sorted[left_inds[still_going]])
            distance_right = np.abs(all_temp_fs[still_going] - self.freqs_sorted[right_inds[still_going]])

            check_move_right = (distance_right <= distance_left)
            check_left_inds = left_inds[still_going][check_move_right] + 1
            check_right_inds = right_inds[still_going][check_move_right] + 1

            new_distance_right = np.abs(all_temp_fs[still_going][check_move_right] - self.freqs_sorted[check_right_inds])

            change_inds = cp.arange(len(all_temp_fs))[still_going][check_move_right][(new_distance_right < distance_left[check_move_right]) & (check_right_inds < len(self.freqs_sorted))]

            left_inds[change_inds] += 1
            right_inds[change_inds] += 1

            stop_inds_right_1 = cp.arange(len(all_temp_fs))[still_going][check_move_right][(check_right_inds >= len(self.freqs_sorted))]

            # last part is just for up here, below it will remove if it is still equal
            stop_inds_right_2 = cp.arange(len(all_temp_fs))[still_going][check_move_right][(new_distance_right >= distance_left[check_move_right]) & (check_right_inds < len(self.freqs_sorted)) & (distance_right[check_move_right] != distance_left[check_move_right])]
            stop_inds_right = cp.concatenate([stop_inds_right_1, stop_inds_right_2])
            assert np.all(still_going[stop_inds_right])

            # equal to should only be left over if it was equal above and moving right did not help
            check_move_left = (distance_left <= distance_right)
            check_left_inds = left_inds[still_going][check_move_left] - 1
            check_right_inds = right_inds[still_going][check_move_left] - 1

            new_distance_left = np.abs(all_temp_fs[still_going][check_move_left] - self.freqs_sorted[check_left_inds])
            
            change_inds = cp.arange(len(all_temp_fs))[still_going][check_move_left][(new_distance_left < distance_right[check_move_left]) & (check_left_inds >= 0)]

            left_inds[change_inds] -= 1
            right_inds[change_inds] -= 1

            stop_inds_left_1 = cp.arange(len(all_temp_fs))[still_going][check_move_left][(check_left_inds < 0)]
            stop_inds_left_2 = cp.arange(len(all_temp_fs))[still_going][check_move_left][(new_distance_left >= distance_right[check_move_left]) & (check_left_inds >= 0)]
            stop_inds_left = cp.concatenate([stop_inds_left_1, stop_inds_left_2])
            
            stop_inds = cp.concatenate([stop_inds_right, stop_inds_left])
            still_going[stop_inds] = False
            # print(jjj, still_going.sum())
            if jjj >= self.nfriends:
                breakpoint()
            jjj += 1

        start_inds = left_inds.copy().get()

        start_inds_all = np.zeros_like(inds, dtype=np.int32)
        start_inds_all[inds] = start_inds.astype(np.int32)

        if "friend_start_inds" not in supps:
            supps.add_objects({"friend_start_inds": start_inds_all})
        else:
            supps[:] = {"friend_start_inds": start_inds_all}

        self.stretch_friends_args_in = tuple([tmp.copy() for tmp in self.all_coords_sorted.T])
        et = time.perf_counter()
        self.mempool.free_all_blocks()

        self.has_setup_group = True
        # print("SETUP:", et - st)
        # start_inds_freq_out = np.zeros((ntemps, nwalkers, nleaves_max), dtype=int)
        # freqs_sorted_here = self.freqs_sorted.get()
        # freqs_remaining_here = all_remaining_freqs

        # start_ind_best = np.zeros_like(freqs_remaining_here, dtype=int)

        # best_index = (
        #     np.searchsorted(freqs_sorted_here, freqs_remaining_here, side="right") - 1
        # )
        # best_index[best_index < self.nfriends] = self.nfriends
        # best_index[best_index >= len(freqs_sorted_here) - self.nfriends] = (
        #     len(freqs_sorted_here) - self.nfriends
        # )
        # check_inds = (
        #     best_index[:, None]
        #     + np.tile(np.arange(2 * self.nfriends), (best_index.shape[0], 1))
        #     - self.nfriends
        # )

        # check_freqs = freqs_sorted_here[check_inds]
        # breakpoint()

        # # batch_count = 1000
        # # split_inds = np.arange(batch_count, freqs_remaining_here.shape[0], batch_count)

        # # splits_remain = np.split(freqs_remaining_here, split_inds)
        # # splits_check = np.split(check_freqs, split_inds)

        # # out = []
        # # for i, (split_r, split_c) in enumerate(zip(splits_remain, splits_check)):
        # #     out.append(np.abs(split_r[:, None] - split_c))
        # #     print(i)

        # # freq_distance = np.asarray(out)

        # freq_distance = np.abs(freqs_remaining_here[:, None] - check_freqs)
        # breakpoint()

        # keep_min_inds = np.argsort(freq_distance, axis=-1)[:, : self.nfriends].min(
        #     axis=-1
        # )
        # start_inds_freq = check_inds[(np.arange(len(check_inds)), keep_min_inds)]

        # start_inds_freq_out[inds] = start_inds_freq

        # start_inds_freq_out[~inds] = -1

        # if "friend_start_inds" not in supps:
        #     supps.add_objects({"friend_start_inds": start_inds_freq_out})
        # else:
        #     supps[:] = {"friend_start_inds": start_inds_freq_out}

        # self.all_friends_start_inds_sorted = self.xp.asarray(
        #     start_inds_freq_out[inds][self.inds_freqs_sorted.get()]
        # )

    def find_friends(self, name, gb_points_to_move, s_inds=None, branch_supps=None):
        if s_inds is None or branch_supps is None:
            raise ValueError

        inds_points_to_move = self.xp.asarray(s_inds.flatten())

        half_friends = int(self.nfriends / 2)

        gb_points_for_move = gb_points_to_move.reshape(-1, 8).copy()

        if not hasattr(self, "ntemps"):
            self.ntemps = 1

        inds_start_freq_to_move = self.xp.asarray(branch_supps[:]["friend_start_inds"].flatten())

        deviation = self.xp.random.randint(0, self.nfriends, size=len(inds_start_freq_to_move))

        inds_keep_friends = inds_start_freq_to_move + deviation

        inds_keep_friends[inds_keep_friends < 0] = 0
        inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted)] = (len(self.all_coords_sorted) - 1)

        gb_points_for_move[inds_points_to_move] = self.all_coords_sorted[inds_keep_friends]
        return gb_points_for_move[None, :, None, :]

    def new_find_friends(self, name, inds_in):
        inds_start_freq_to_move = self.current_friends_start_inds[tuple(inds_in)]

        deviation = self.xp.random.randint(
            0, self.nfriends, size=len(inds_start_freq_to_move)
        )

        inds_keep_friends = inds_start_freq_to_move + deviation

        inds_keep_friends[inds_keep_friends < 0] = 0
        inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted)] = (
            len(self.all_coords_sorted) - 1
        )

        gb_points_for_move = self.all_coords_sorted[
            inds_keep_friends
        ]

        return gb_points_for_move

    def setup(self, model, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name != "gb":
                continue

            if branch.inds[0].sum() >= self.nfriends and ((self.time % self.n_iter_update == 0) or (not self.has_setup_group)):  # not self.is_rj_prop and 
                self.setup_gbs(model, branch)

            # update any shifted start inds due to tempering (need to do this every non-rj move)
            """if not self.is_rj_prop:
                # fix the ones that have been added in RJ
                fix = (
                    branch.branch_supplemental.holder["friend_start_inds"][:] == -1
                ) & branch.inds

                if np.any(fix):
                    new_freqs = cp.asarray(branch.coords[fix][:, 1])
                    # TODO: is there a better way of doing this?

                    # fill information into friend finder for new binaries
                    branch.branch_supplemental.holder["friend_start_inds"][fix] = (
                        (
                            cp.searchsorted(self.freqs_sorted, new_freqs, side="right")
                            - 1
                        )
                        * (
                            (new_freqs > self.freqs_sorted[0])
                            & (new_freqs < self.freqs_sorted[-1])
                        )
                        + 0 * (new_freqs < self.freqs_sorted[0])
                        + (len(self.freqs_sorted) - 1)
                        * (new_freqs > self.freqs_sorted[-1])
                    ).get()

                # make sure current start inds reflect alive binaries
                self.current_friends_start_inds = self.xp.asarray(
                    branch.branch_supplemental.holder["friend_start_inds"][:]
                )
            """

            self.mempool.free_all_blocks()


class GBSpecialRJPriorMove(GBSpecialBase):
    pass

def para_log_like(x, gb, acs, walker_max, transform_fn, phase_maximize, waveform_kwargs, fstat=True, return_snr=False):
    xp = cp if gb.use_gpu else np
    
    x_tmp = transform_fn.both_transforms(x, xp=xp)
    # need to get just f, fdot, fddot, lam, beta
    data_index = xp.full(x.shape[0], walker_max, dtype=xp.int32)

    if fstat:
        x_in = x_tmp[:, xp.array([1, 2, 3, 7, 8])]
        # TODO: fix for N>256?
        ll = gb.get_fstat_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.data_length,
            data_splits=np.array([gb.gpus[0]]),
            phase_marginalize=phase_maximize,
            return_cupy=True,
            N=256, # 1024 is too much shared memory I think 
            **waveform_kwargs
        )

        x[:, 0] = np.log(gb.A_max)
        x[:, 3] = gb.phi0_max % (2 * np.pi)
        x[:, 4] = np.cos(gb.iota_max % (np.pi))
        x[:, 5] = gb.psi_max % (np.pi)
    
    else:
        # breakpoint()
        x_in = x_tmp[:]
        ll = gb.get_ll(
            x_in,
            acs.linear_data_arr,
            acs.linear_psd_arr,
            data_index=data_index,
            noise_index=data_index,
            data_length=acs.data_length,
            data_splits=np.array([gb.gpus[0]]),
            phase_marginalize=phase_maximize,
            return_cupy=True,
            # N=256,
            **waveform_kwargs
        )
        # breakpoint()

        # params_remove_in = x_in.copy()
        # params_add_in = x_in.copy()

        # params_remove_in[:, 0] *= 1e-50
        # breakpoint()
        # ll_diff_2 = gb.swap_likelihood_difference(
        #     params_remove_in,
        #     params_add_in,
        #     acs.linear_data_arr,
        #     acs.linear_psd_arr,
        #     # start_freq_ind=self.xp.asarray(self.acs.start_freq_ind).astype(np.int32),
        #     data_index=data_index,
        #     noise_index=data_index,
        #     # N=N_vals,
        #     data_length=acs.data_length,
        #     data_splits=np.array([gb.gpus[0]]),
        #     phase_marginalize=phase_maximize,
        #     return_cupy=True,
        #     N=256,
        #     **waveform_kwargs,
        # )
        # breakpoint()

        if phase_maximize:
            x[:, 3] = (x[:, 3] - xp.angle(xp.asarray(gb.non_marg_d_h))) % (2 * np.pi)
    
        if return_snr:
            opt_snr = gb.h_h.real ** (1/2)
            return (ll, opt_snr)

    return ll


class PriorTransformFn:
    def __init__(self, f_min: float, f_max: float, fdot_min: float, fdot_max: float):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = f_min, f_max, fdot_min, fdot_max
        
    def adjust_logp(self, logp, groups_running):

        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1. / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1. / (fdot_max_here - fdot_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]

        return logp

    def transform_to_prior_basis(self, coords, groups_running):
        xp = get_array_module(self.f_min)

        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        try:
            coords[:, :, :, 1] = (coords[:, :, :, 1] - f_min_here[:, None, None]) / (f_max_here[:, None, None] - f_min_here[:, None, None])
        except:
            breakpoint()

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (coords[:, :, :, 2] - fdot_min_here[:, None, None]) / (fdot_max_here[:, None, None] - fdot_min_here[:, None, None])
        
        return

    def transform_from_prior_basis(self, coords, groups_running):
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        assert groups_running.shape[0] == coords.shape[0]
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        coords[:, :, :, 1] = (coords[:, :, :, 1] * (f_max_here[:, None, None] - f_min_here[:, None, None])) + f_min_here[:, None, None]

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (coords[:, :, :, 2] * (fdot_max_here[:, None, None] - fdot_min_here[:, None, None])) + fdot_min_here[:, None, None] 

        return

class BayesGMMFit:
    
    def __init__(self, samples_in):
    
        assert isinstance(samples_in, np.ndarray)
    
        run = True
        min_bic = np.inf
        self.sample_mins = sample_mins = samples_in.min(axis=0)
        self.sample_maxs = sample_maxs = samples_in.max(axis=0)

        samples = self.transform_to_gmm_basis(samples_in)

        mixture = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution",
            n_components=60,
            # reg_covar=0,
            # init_params="random",
            max_iter=5000,
            # mean_precision_prior=0.8,
            # random_state=random_state,
        )
        mixture.fit(samples)

        self.keep_mix = mixture

    def transform_to_gmm_basis(self, samples):
        return ((samples - self.sample_mins[None, :]) / (self.sample_maxs[None, :] - self.sample_mins[None, :])) * 2 - 1
        
    def transform_from_gmm_basis(self, samples):
        return (samples + 1.) / 2. * (self.sample_maxs[None, :] - self.sample_mins[None, :]) + self.sample_mins[None, :]
        

from sklearn.mixture import GaussianMixture

class GMMFit:
    
    def __init__(self, samples_in):

        assert isinstance(samples_in, np.ndarray)
    
        run = True
        min_bic = np.inf
        self.sample_mins = sample_mins = samples_in.min(axis=0)
        self.sample_maxs = sample_maxs = samples_in.max(axis=0)

        samples = self.transform_to_gmm_basis(samples_in)
        bad = False
        for n_components in range(1, 30):
            if not run:
                continue
            #fit_gaussian_mixture_model(n_components, samples)
            #breakpoint()
            try:
                mixture = GaussianMixture(n_components=n_components, verbose=False, verbose_interval=2)
    
                mixture.fit(samples)
                test_bic = mixture.bic(samples)
            except ValueError:
                # print("ValueError", samples)
                run = False
                bad = True
                continue
            # print(n_components, test_bic)
            if test_bic < min_bic:
                min_bic = test_bic
                keep_mix = mixture
                keep_components = n_components
                
            else:
                run = False
    
                # print(leaf, n_components - 1, et - st)
            
            """if keep_components >= 9:
                new_samples = keep_mix.sample(n_samples=100000)[0]
                old_samples = samples
                fig = corner.corner(old_samples, hist_kwargs=dict(density=True, color="r"), color="r", plot_datapoints=False, plot_density=False)
                corner.corner(new_samples, hist_kwargs=dict(density=True, color="b"), color="b", plot_datapoints=False, plot_contours=True, plot_density=False, fig=fig)
                fig.savefig("mix_check.png")
                plt.close()
                breakpoint()"""
    
        if bad:
            print("BAD")
        if keep_components >= 19:
            print(keep_components)
        # output_list = [keep_mix.weights_, keep_mix.means_, keep_mix.covariances_, np.array([np.linalg.inv(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), np.array([np.linalg.det(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), sample_mins, sample_maxs]
        
        self.keep_mix = keep_mix

    def transform_to_gmm_basis(self, samples):
        return ((samples - self.sample_mins[None, :]) / (self.sample_maxs[None, :] - self.sample_mins[None, :])) * 2 - 1
        
    def transform_from_gmm_basis(self, samples):
        return (samples + 1.) / 2. * (self.sample_maxs[None, :] - self.sample_mins[None, :]) + self.sample_mins[None, :]

def gather_gmms(gmms):
    weights = []
    means = []
    covs = []
    inv_covs = []
    dets = []
    sample_mins = []
    sample_maxs = []

    for gmm in gmms:
        weights.append(gmm.keep_mix.weights_)
        means.append(gmm.keep_mix.means_)
        covs.append(gmm.keep_mix.covariances_)
        inv_covs.append(np.array([np.linalg.inv(gmm.keep_mix.covariances_[i]) for i in range(len(gmm.keep_mix.weights_))]))
        dets.append(np.array([np.linalg.det(gmm.keep_mix.covariances_[i]) for i in range(len(gmm.keep_mix.weights_))]))
        sample_mins.append(gmm.sample_mins)
        sample_maxs.append(gmm.sample_maxs)
    
    return (
        weights,
        means,
        covs,
        inv_covs,
        dets,
        sample_mins,
        sample_maxs
    )

from ...sampling.gmm import GMMFit as GPUGMMFit

class GBSpecialRJSerialSearchMCMC(GBSpecialBase):
    comm_info = None
    def get_rank_function(self):
        return gb_search_func

    def setup(self, model, branches):

        # run paraensemble MCMC. 
        max_logl_walker = np.argmax(model.analysis_container_arr.likelihood()).item()
        self.gb.d_d = 0.0  # model.analysis_container_arr.inner_product()[max_logl_walker]
        from eryn.paraensemble import ParaEnsembleSampler
        from eryn.prior import uniform_dist
        ndim = branches["gb"].ndim
        nwalkers = 40  # TODO: adjustable
        ntemps = 10  # TODO: adjustable
        ngroups = self.num_bands  # TODO: is this always ok?
        priors = self.priors if not self.use_gpu else self.gpu_priors
        
        # TODO: make this adjustable to match settings
        m_chirp_lims = [0.001, 1.2]

        f0_max = self.band_edges[1:]
        f0_min = self.band_edges[:-1]

        from gbgpu.utils.utility import get_fdot
        fdot_max = get_fdot(f0_max, Mc=m_chirp_lims[-1])
        fdot_min = -fdot_max
    
        priors_in = priors["gb"].priors_in
        priors_in[1] = uniform_dist(0.0, 1.0, use_cupy=self.use_gpu)
        priors_in[2] = uniform_dist(0.0, 1.0, use_cupy=self.use_gpu)
        priors = {"gb": ProbDistContainer(priors_in, return_gpu=True, use_cupy=self.use_gpu)}
        
        prior_transform_fn = PriorTransformFn(f0_min * 1e3, f0_max * 1e3, fdot_min, fdot_max)
        
        start_params = priors["gb"].rvs(size=(ngroups, ntemps, nwalkers))
        prior_transform_fn.transform_from_prior_basis(start_params, self.xp.arange(ngroups))
        print("phase maximizing here right now (?)")
        ll_args = (
            self.gb, 
            model.analysis_container_arr, 
            max_logl_walker,
            self.parameter_transforms, 
            True, # self.phase_maximize,
            self.waveform_kwargs
        )

        ll_args_2 = (
            self.gb, 
            model.analysis_container_arr, 
            max_logl_walker,
            self.parameter_transforms, 
            False, # self.phase_maximize,
            self.waveform_kwargs
        )

        # test_ll = para_log_like(
        #     test_params, 
        #     *ll_args
        # )
        gibbs_sampling_setup = np.ones(8, dtype=bool)
        gibbs_sampling_setup[np.array([0, 3, 4, 5])] = False
        # TODO: track and shutoff bands after not finding sources 2 or 3 times
        para_sampler = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args,
            # kwargs: dict = {},
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn,
            name="gb",
            gibbs_sampling_setup=gibbs_sampling_setup,
            # provide_supplemental=False,
        )

        from eryn.state import ParaState
        state = ParaState({"gb": start_params}, groups_running=self.xp.ones(ngroups, dtype=bool))
        state.log_prior = para_sampler.compute_log_prior(state.branches_coords)
        state.log_like = para_sampler.compute_log_like(state.branches_coords, logp=state.log_prior)

        nsteps = 500
        para_sampler.run_mcmc(state, nsteps, burn=500, progress=True)

        samples = self.xp.asarray(para_sampler.get_chain()[:, :, 0])
        check_ll = para_sampler.get_log_like()[:, :, 0]
        sample_ll = para_log_like(
            samples.reshape(-1, 8), 
            *ll_args
        ).reshape(samples.shape[:-1]).get()

        check_real_ll_phase_maximized = para_log_like(
            samples.reshape(-1, 8), 
            *ll_args, fstat=False
        ).reshape(samples.shape[:-1]).get()
        check_real_ll, opt_snr = para_log_like(
            samples.reshape(-1, 8), 
            *ll_args_2, fstat=False, return_snr=True
        )
        check_real_ll = check_real_ll.reshape(samples.shape[:-1]).get()
        opt_snr = opt_snr.reshape(samples.shape[:-1]).get()

        # TODO: make cut adjustable
        groups_running_now = opt_snr.min(axis=(0, 2)) > 10.0
        print(f"FOUND {groups_running_now.sum()} out of {groups_running_now.shape[0]}")
        if not np.any(groups_running_now):
            print("Did not find any new sources.")
            return 

        start_params_2 = np.tile(samples[-1][groups_running_now, None], (1, ntemps, 1, 1))

        gibbs_sampling_setup_2 = np.ones(8, dtype=bool)
        gibbs_sampling_setup_2[np.array([3])] = False
        prior_transform_fn_2 = PriorTransformFn(f0_min[groups_running_now] * 1e3, f0_max[groups_running_now] * 1e3, fdot_min[groups_running_now], fdot_max[groups_running_now])
        
        ngroups_2 = groups_running_now.sum().item()
        para_sampler_2 = ParaEnsembleSampler(
            ndim,
            nwalkers,
            ngroups_2,
            para_log_like,
            priors,
            tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
            args=ll_args,
            kwargs=dict(fstat=False),
            gpu=self.gb.gpus[0],
            periodic=self.periodic,
            # backend: ParaBackend = None,  # add ParaHDFBackend
            # update_fn: Callable = None,
            # update_iterations=-1,
            # stopping_fn: Callable = None,
            # stopping_iterations: int=-1,
            prior_transform_fn=prior_transform_fn_2,
            name="gb",
            gibbs_sampling_setup=gibbs_sampling_setup_2,
            # provide_supplemental=False,
        )

        new_state = ParaState({"gb": start_params_2}, groups_running=self.xp.ones(ngroups_2, dtype=bool))
        new_state.log_prior = para_sampler_2.compute_log_prior(new_state.branches_coords)
        new_state.log_like = para_sampler_2.compute_log_like(new_state.branches_coords, logp=new_state.log_prior)

        if np.any(np.isinf(new_state.log_prior)):
            breakpoint()

        nsteps = 500
        para_sampler_2.run_mcmc(new_state, nsteps, burn=500, progress=True)

        samples_2 = self.xp.asarray(para_sampler_2.get_chain()[:, :, 0])
        check_ll_2 = para_sampler_2.get_log_like()[:, :, 0]

        check_real_ll_phase_maximized_2 = para_log_like(
            samples_2.reshape(-1, 8), 
            *ll_args, fstat=False
        ).reshape(samples_2.shape[:-1]).get()
        check_real_ll_2 = para_log_like(
            samples_2.reshape(-1, 8), 
            *ll_args_2, fstat=False
        ).reshape(samples_2.shape[:-1]).get()

        # TODO: add removal of bands that consistently dont find things
        samples_2 = samples_2.transpose(1, 0, 2, 3)
        np.save("samples_examples", samples_2)
        import time
        st = time.perf_counter()
        samples_2_tmp = samples_2.reshape(samples_2.shape[0], -1, samples_2.shape[-1])
        _gmms = GPUGMMFit(samples_2_tmp, use_gpu=True)
        et = time.perf_counter()
        print(f"GPU GMM FIT: {et - st}")
        breakpoint()
            
        if self.ranks_needed == 0:
            gmms = [GMMFit(samples_2[i].get().reshape(-1, 8)) for i in range(samples_2.shape[0])]
            gmm_info = gather_gmms(gmms)

        else:
            if self.comm_info is None:
                # this only happens the first time through
                self.comm_info = self.comm.recv(tag=232342)
                
            gmm_info = fit_gmm(samples_2.get(), self.comm, self.comm_info)
        
        full_gmm = FullGaussianMixtureModel(*gmm_info, use_cupy=self.use_gpu)
        
        gen_samp = self.xp.asarray(full_gmm.rvs(100))
        gen_ll, gen_opt_snr = para_log_like(gen_samp, *ll_args, fstat=False, return_snr=True)
        print(gen_ll, self.gb.d_h / gen_opt_snr, gen_opt_snr)
        # breakpoint()
        self.rj_proposal_distribution = {"gb": full_gmm}
    
   
class GBSpecialRJSearchMove(GBSpecialBase):
    def get_rank_function(self):
        return gb_search_func

    def setup(self, model, branches):
        self.interact_with_search()
        super(GBSpecialRJSearchMove, self).setup(branches)

    def interact_with_search(self):
        search_rank = self.ranks[0]

        search_ch = self.comm.irecv(source=search_rank)
        if search_ch.get_status():
            search_req = search_ch.wait()

            if "receive" in search_req and search_req["receive"]:
                search_dict = self.comm.recv(source=search_rank)
                self.rj_proposal_distribution["gb"] = make_gmm(self.gb, search_dict["search"])

            if "send" in search_req and search_req["send"]:
                # get random instance of residual, psd, lisasens
                # TODO: decide about random versus max ll
                random_ind = np.random.randint(self.nwalkers)

                data = [self.mgh.data_shaped[0][0][random_ind].get(), self.mgh.data_shaped[1][0][random_ind].get()]
                psd = [self.mgh.psd_shaped[0][0][random_ind].get(), self.mgh.psd_shaped[1][0][random_ind].get()]
                lisasens = [self.mgh.psd_shaped[0][0][random_ind].get(), self.mgh.lisasens_shaped[1][0][random_ind].get()]
                
                output_data = dict(
                    data=data,
                    psd=psd,
                    lisasens=lisasens
                )
                self.comm.send(output_data, dest=search_rank)

        else:
            search_ch.cancel()

        print("CHECK INSIDE PROP")

class GBSpecialRJRefitMove(GBSpecialBase):
    def get_rank_function(self):
        return gb_refit_func
