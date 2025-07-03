import time
import cupy as xp
from gbgpu.utils.utility import get_N
from gbgpu.gbgpu import GBGPU
import numpy as np
from gbgpu.utils.constants import *
from datetime import datetime
import pandas as pd
import os
import h5py
import multiprocessing as mp

import time
import cupy as xp
from gbgpu.utils.utility import get_N
from gbgpu.gbgpu import GBGPU
import numpy as np
from gbgpu.utils.constants import *
from datetime import datetime
import pandas as pd
import os
import h5py
from typing import Tuple, Optional, Any, List
from copy import deepcopy

class GBGrouping:
    pass

class GBGrouping:
    best_match_limit = 0.9
    parameters = []
    stop = False
    phase_marginalize = True  # TODO: check
    def __init__(self, gb, current_info, params: np.ndarray, sample: np.ndarray, fake_data: list, psd: list, waveform_kwargs: dict, groups: np.ndarray = None, copy: bool = False, group_indicator: int = None, original_sample_count: int = None, samples_so_far: int = None) -> None:

        dc = deepcopy if copy else (lambda x: x)
        self.current_info = current_info
        self.params = dc(params)
        self.groups = dc(groups)
        self.sample = dc(sample)
        self.fake_data = fake_data
        self.psd = psd
        self.gb = gb
        self.group_indicator = group_indicator
        self.original_sample_count = original_sample_count
        self.waveform_kwargs = waveform_kwargs
        self.samples_so_far = samples_so_far
        self.samples_finished = []


    @property
    def params(self) -> np.ndarray:
        return self._params

    @property
    def params_in(self) -> np.ndarray:
        return self.current_info.gb_info["transform"].both_transforms(self.params)

    @params.setter
    def params(self, params: np.ndarray) -> None:
        assert params.ndim == 2
        assert params.shape[-1] == 8

        self._params = params

    @property
    def groups(self) -> np.ndarray:
        return self._groups

    @groups.setter
    def groups(self, groups: np.ndarray) -> None:

        if groups is None:
            groups = np.arange(len(self.params))

        assert groups.ndim == 1
        assert len(groups) == len(self.params)

        self._groups = groups


    @property
    def sample(self) -> np.ndarray:
        return self._sample

    @sample.setter
    def sample(self, sample: np.ndarray) -> None:

        if sample is None:
            sample = np.zeros(len(self.params), dtype=int)

        assert sample.ndim == 1
        assert len(sample) == len(self.params)

        self._sample = sample

    @property
    def unique_groups(self) -> np.ndarray:
        return np.unique(self.groups)

    @property
    def ngroups(self) -> np.ndarray:
        return len(np.unique(self.groups))

    @property
    def median_sources(self) -> np.ndarray:
        median_sources = np.zeros((len(self.unique_groups), 8))
        for i, group in enumerate(self.unique_groups):
            inds_group = self.groups == group
            
            ind_median = np.argsort(self.params[inds_group, 1])[int(inds_group.sum() / 2)]
            median_sources[i] = self.params[inds_group][ind_median]
        return median_sources

    @property
    def median_sources_in(self) -> np.ndarray:
        return self.current_info.gb_info["transform"].both_transforms(self.median_sources)

    @property
    def min_sources(self) -> np.ndarray:
        min_sources = np.zeros((len(self.unique_groups), 8))
        for i, group in enumerate(self.unique_groups):
            inds_group = self.groups == group
            ind_min = np.argsort(self.params[inds_group, 1])[0]
            min_sources[i] = self.params[inds_group][ind_min]
        return min_sources

    @property
    def min_sources_in(self) -> np.ndarray:
        return self.current_info.gb_info["transform"].both_transforms(self.min_sources)

    @property
    def max_sources(self) -> np.ndarray:
        max_sources = np.zeros((len(self.unique_groups), 8))
        for i, group in enumerate(self.unique_groups):
            inds_group = self.groups == group
            ind_max = np.argsort(self.params[inds_group, 1])[-1]
            max_sources[i] = self.params[inds_group][ind_max]
        return max_sources

    @property
    def max_sources_in(self) -> np.ndarray:
        return self.current_info.gb_info["transform"].both_transforms(self.max_sources)

    def add_single_source(self, params: np.ndarray, group: int) -> None:
        self._params = np.concatenate([self._params, params[None, :]], axis=0)
        self._groups = np.concatenate([self._groups, np.array([group])], axis=0)

    def get_group(self, group: int) -> np.ndarray:
        return self.params[self.groups == group]
    
    def get_group_in(self, group: int) -> np.ndarray:
        return self.params_in[self.groups == group]

    def get_group_median_inds(self, all_samples_in: np.ndarray, assigned_groups: np.ndarray, include_only_more_than_1: bool=False) -> np.ndarray:
        # sort by group and then frequency
        df_tmp = pd.DataFrame({"f": all_samples_in[:, 1], "groups": assigned_groups})
        df_sorted = df_tmp.sort_values(["groups", "f"], ascending=[True, True])
        
        inds_assign_sort = df_sorted.index.to_numpy()

        uni, uni_inds, uni_length = np.unique(assigned_groups[inds_assign_sort], return_index=True, return_counts=True)
        
        # 1: removes -1 assigned groups
        group_median_inds = inds_assign_sort[uni_inds + (uni_length / 2).astype(int)]
        
        if include_only_more_than_1:
            group_median_inds = group_median_inds[uni_length > 1]
        
        return group_median_inds

    def get_group_pop(self, all_samples_in: np.ndarray, assigned_groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # sort by group and then frequency
        df_tmp = pd.DataFrame({"f": all_samples_in[:, 1], "groups": assigned_groups})
        df_sorted = df_tmp.sort_values(["groups", "f"], ascending=[True, True])
        
        inds_assign_sort = df_sorted.index.to_numpy()

        uni, uni_inds, uni_length = np.unique(assigned_groups[inds_assign_sort], return_index=True, return_counts=True)
        
        # 1: removes -1 assigned groups
        return uni, uni_length

    def get_overlap(self, params_base_in: np.ndarray, params_check_in: np.ndarray) -> np.ndarray:
        fake_data_swap = [[self.fake_data[0]], [self.fake_data[1]]]
        psd_in_swap = [[self.psd[0]], [self.psd[1]]]
        
        overlap = np.zeros(params_base_in.shape[0])
        N = xp.asarray(get_N(params_base_in[:, 0], params_base_in[:, 1], YEAR, oversample=4))

        batch_size = int(1e7)

        batch_inds = np.arange(0, params_base_in.shape[0], batch_size)
        if batch_inds[-1] != params_base_in.shape[0] - 1:
            batch_inds = np.concatenate([batch_inds, np.array([params_base_in.shape[0] - 1])])
        
        self.gb.gpus = [xp.cuda.runtime.getDevice()]
        for stind, endind in zip(batch_inds[:-1], batch_inds[1:]):
            _ = self.gb.swap_likelihood_difference(params_base_in[stind:endind], params_check_in[stind:endind], fake_data_swap,psd_in_swap, phase_marginalize=self.phase_marginalize, N=N[stind:endind], start_freq_ind=0,data_length=len(self.fake_data[0]),data_splits=[np.array([0])],**self.waveform_kwargs,)
        
            overlap[stind:endind] = (np.real(self.gb.add_remove) / np.sqrt(self.gb.add_add.real * self.gb.remove_remove.real)).get()
        return overlap

    def get_sample(self, sample_i: int) -> np.ndarray:
        return self.params[self.sample == sample_i]

    @property
    def num_samples(self) -> int:
        num_samples = self.sample.max() + 1
        assert np.unique(self.sample).shape[0] == num_samples
        return num_samples

    @property
    def sample_pop(self) -> np.ndarray:
        uni, uni_inverse, uni_count = np.unique(self.sample, return_counts=True, return_inverse=True)
        sample_pop = uni_count
        return sample_pop

    def prune(self, remove_singles: bool=False):
        success = np.zeros_like(self.groups, dtype=bool)
        tried = np.zeros_like(self.groups, dtype=bool)
        success[self.groups == -1] = False
        tried[self.groups == -1] = True
        
        while not np.all(tried | success):
            inds_now = np.arange(len(self.groups))[(~tried) & (~success)]
            uni, uni_inds, uni_inverse = np.unique(self.groups[inds_now], return_index=True, return_inverse=True)
            uni_inds_base = inds_now[uni_inds]
            uni_inverse_sub = np.delete(uni_inverse, uni_inds)
            if len(uni_inverse_sub) == 0:
                tried[uni_inds_base] = True
                continue

            params_in = self.params_in[uni_inds_base]
            num_here = params_in.shape[0]
            
            base_inds = uni_inds_base[uni_inverse_sub]
            check_inds = inds_now[np.delete(np.arange(len(inds_now)), uni_inds)]
            base_params_in = params_in[uni_inverse_sub]
            check_params_in = self.params_in[check_inds]
            try:
                overlap = self.get_overlap(base_params_in, check_params_in)
            except IndexError:
                breakpoint()
                
            inds_success = np.unique(np.concatenate([base_inds[overlap > 0.9], check_inds[overlap > 0.9]]))
            success[inds_success] = True

            tried[inds_success] = True
            tried[np.unique(base_inds)] = True

            print("prune:", success.sum(), tried.sum(), (success | tried).sum())
    
        self.groups[~success] = -1

        # median_inds = self.get_group_median_inds(self.params_in, self.groups)
        # median_groups = self.groups[median_inds]
        # inds_sort_groups = np.argsort(median_groups)
        # median_inds_sort = median_inds[inds_sort_groups]
        # sort_groups = median_groups[inds_sort_groups]

        # uni, uni_inverse = np.unique(self.groups, return_inverse=True)
        # breakpoint()
        # assert np.all(uni == sort_groups)
        # inds_for_base = median_inds_sort[uni_inverse]

        # inds_base = np.arange(self.groups.shape[0])[np.in1d(np.arange(self.groups.shape[0]), median_inds_sort) | (self.groups == -1) | (self.group_pop <= self.original_sample_count)]
        # inds_check = np.delete(np.arange(self.groups.shape[0]), inds_base)
        # inds_corresponding = np.delete(inds_for_base, inds_base)
        # params_base_in = self.params_in[inds_corresponding]
        # params_check_in = self.params_in[inds_check]
        # overlap = self.get_overlap(params_base_in, params_check_in)

        # base_group_fix = self.groups[inds_corresponding]
        # check_group_fix = self.groups[inds_check]

        # for group in np.unique(base_group_fix):
        #     overlaps_group = overlap[base_group_fix == group]

        

        if remove_singles:
            inds_keep = self.groups != -1

            self.params = self.params[inds_keep]
            self.sample = self.sample[inds_keep]
            self.groups = self.groups[inds_keep]

        self.reset_groups_to_range()
                
    def group_based_on_sample(self, sample_i: int=0, start_with_median: bool=False, store_best_overlap: bool=False, remove_grouped: bool=True) -> None:
        
        # get base group
        if sample_i in self.samples_finished:
            print(f"sample_i: ({sample_i}) already used.")
            return 

        max_leaves = self.sample_pop.max()
        
        # running against injection
        if sample_i == -1:
            keep_base = (self.sample == sample_i)
        else:
            keep_base = (self.sample == sample_i) & (self.groups == -1)
        all_samples_base = self.params[keep_base]  # self.get_sample(sample_i)
        original_inds_base = np.where(keep_base)[0]
        
        self.samples_finished.append(sample_i)
        # must add sample before this
        if remove_grouped:
            inds_keep = (self.sample != sample_i) & (self.groups == -1)
        else:
            inds_keep = self.sample != sample_i

        other_samples = self.params[inds_keep]
        other_original_inds = np.arange(self.params.shape[0])[inds_keep]
        if len(other_samples) == 0:
            print("No other samples to test against.")
            return

        N_base = get_N(all_samples_base[:, 0], all_samples_base[:, 1] / 1e3, YEAR, oversample=4).astype(np.int32)

        sort = np.argsort(other_samples[:, 1])
        other_samples = other_samples[sort]
        other_original_inds = other_original_inds[sort]
        df = self.current_info.general_info["df"]
        width_factor = 2
        
        f_ind = (all_samples_base[:, 1] / 1e3 / df).astype(int)
        max_f = ((f_ind + N_base / width_factor) * df) * 1e3
        min_f = ((f_ind - N_base / width_factor) * df) * 1e3

        max_ind = np.searchsorted(other_samples[:, 1], max_f, side="right") - 1
        min_ind = np.searchsorted(other_samples[:, 1], min_f, side="right") - 1
        
        min_ind[min_ind < 0] = 0
        max_ind[max_ind > other_samples.shape[0] - 1] = all_samples_base.shape[0] - 1
        
        base_params = []
        base_ind = []
        check_ind = []
        check_params = []
        for curr_sample_i, (current_sample, min_ind_i, max_ind_i, original_ind) in enumerate(zip(all_samples_base, min_ind, max_ind, original_inds_base)):
            length = max_ind_i - min_ind_i + 1
            if length <= 0:
                continue

            base_ind.append(np.repeat(original_ind, length))
            base_params.append(np.tile(current_sample, (length, 1)))
            check_params.append(other_samples[min_ind_i:max_ind_i + 1])
            check_ind.append(other_original_inds[min_ind_i: max_ind_i + 1])
            if curr_sample_i % 1000 == 0:
                print("prep:", curr_sample_i, len(all_samples_base))

        try:
            base_params = np.concatenate(base_params, axis=0)
        except ValueError:
            print("no base_params")
            return
        base_ind = np.concatenate(base_ind, axis=0)
        check_params = np.concatenate(check_params, axis=0)
        check_ind = np.concatenate(check_ind, axis=0)
        base_params_in = self.current_info.gb_info["transform"].both_transforms(base_params)
        check_params_in = self.current_info.gb_info["transform"].both_transforms(check_params)
        overlap = self.get_overlap(base_params_in, check_params_in)

        self.groups[original_inds_base] = self.groups.max() + 1 + np.arange(original_inds_base.shape[0])
        
        base_ind_keep = base_ind[overlap > 0.9]
        check_ind_keep = check_ind[overlap > 0.9]
        overlap_keep = overlap[overlap > 0.9]
        base_params_keep = base_params_in[overlap > 0.9]
        check_params_keep = check_params_in[overlap > 0.9]

        uni, uni_count = np.unique(check_ind_keep, return_counts=True)
        fix_check = uni[uni_count > 1]

        need_to_fix = np.in1d(check_ind_keep, fix_check)
        check_ind_fix = check_ind_keep[need_to_fix]
        base_ind_fix = base_ind_keep[need_to_fix]
        overlap_to_fix = overlap_keep[need_to_fix] 
        inds_fix = np.arange(check_ind_keep.shape[0])[need_to_fix]  

        special_map = check_ind_fix * 1e1 + overlap_to_fix
        sort = np.argsort(special_map)[::-1]
        check_ind_fix = check_ind_fix[sort]
        base_ind_fix = base_ind_fix[sort]
        overlap_to_fix = overlap_to_fix[sort]
        inds_fix = inds_fix[sort]

        uni, uni_inds = np.unique(check_ind_fix, return_index=True)

        check_ind_best = check_ind_fix[uni_inds]
        base_ind_best = base_ind_fix[uni_inds]
        inds_remove = np.delete(inds_fix, uni_inds)

        base_ind_keep = np.delete(base_ind_keep, inds_remove, axis=0)
        check_ind_keep = np.delete(check_ind_keep, inds_remove, axis=0)
        base_params_keep = np.delete(base_params_keep, inds_remove, axis=0)
        check_params_keep = np.delete(check_params_keep, inds_remove, axis=0)
        overlap_keep = np.delete(overlap_keep, inds_remove, axis=0)
        
        self.groups[check_ind_keep] = self.groups[base_ind_keep]

        if store_best_overlap:
            special_map = base_ind * 1e2 + overlap
            sort = np.argsort(special_map)[::-1]
            overlap = overlap[sort]
            check_ind = check_ind[sort]
            base_ind = base_ind[sort]
            check_params_in = check_params_in[sort]
            base_params_in = base_params_in[sort]

            uni, uni_inds = np.unique(base_ind, return_index=True)

            overlap_keep = overlap[uni_inds]
            check_ind_keep = check_ind[uni_inds]
            base_ind_keep = base_ind[uni_inds]
            check_params_in_keep = check_params_in[uni_inds]
            base_params_in_keep = base_params_in[uni_inds]

            sort_base = np.argsort(base_ind_keep)
            overlap_keep = overlap_keep[sort_base]
            check_ind_keep = check_ind_keep[sort_base]
            base_ind_keep = base_ind_keep[sort_base]
            check_params_in_keep = check_params_in_keep[sort_base]
            base_params_in_keep = base_params_in_keep[sort_base]
        
            # assumes the 'injection' sample is added last
            self.best_overlap_inds = check_ind_keep
            self.best_overlap_base = base_ind_keep
            self.best_overlap = overlap_keep
            self.best_overlap_check_params = check_params_in_keep
            self.best_overlap_base_params = base_params_in_keep
            
        self.reset_groups_to_range()

    def reset_groups_to_range(self) -> None:
        # reset everything to be incremented by 1
        
        uni, uni_inverse, uni_counts = np.unique(self.groups, return_inverse=True, return_counts=True)
        # adjust any single groups
        count_inv = uni_counts[uni_inverse]
        self.groups[count_inv == 1] = -1
        
        # do reset
        uni, uni_inverse, uni_counts = np.unique(self.groups[self.groups != -1], return_inverse=True, return_counts=True)
        self.groups[self.groups != -1] = np.arange(len(uni))[uni_inverse]


    def consolidate_cat(self, final_consolidation: bool=False) -> None:
        median_inds = self.get_group_median_inds(self.params_in, self.groups)
        median_inds = median_inds[self.groups[median_inds] != -1]

        med_params = self.params[median_inds]
        med_params_in = self.current_info.gb_info["transform"].both_transforms(med_params)

        median_f = med_params_in[:, 1]
        
        base_inds = []
        check_inds = []
        base_params_in = []
        check_params_in = []
        for i in range(med_params_in.shape[0]):
            inds = np.where((np.abs(median_f - med_params_in[i, 1]) < 20 * self.current_info.general_info["df"]) & (median_f != med_params_in[i, 1]))[0]
            check_inds.append(inds)
            base_inds.append(np.full_like(inds, i))
            check_params_in.append(med_params_in[inds])
            base_params_in.append(np.tile(med_params_in[i], (inds.shape[0], 1)))
        
        check_inds = np.concatenate(check_inds)
        base_inds = np.concatenate(base_inds)
        check_params_in = np.concatenate(check_params_in, axis=0)
        base_params_in = np.concatenate(base_params_in, axis=0)

        overlap = self.get_overlap(check_params_in, base_params_in)
        
        base_inds_keep = base_inds[overlap > 0.9]
        check_inds_keep = check_inds[overlap > 0.9]
        overlap_keep = overlap[overlap > 0.9]
        base_params_keep = base_params_in[overlap > 0.9]
        check_params_keep = check_params_in[overlap > 0.9]

        for base_ind, check_ind in zip(base_inds_keep, check_inds_keep):
            base_ind_median = median_inds[base_ind]
            check_ind_median = median_inds[check_ind]

            if self.groups[base_ind_median] != self.groups[check_ind_median]:
                self.groups[self.groups == self.groups[check_ind_median]] = self.groups[base_ind_median]
        
        if final_consolidation:
            # assert (overlap > 0.9).sum() == 0
            further_consider = ((overlap > 0.75) & (overlap < 0.9) & (self.groups[median_inds[check_inds]] != -1 ))
            num_check = further_consider.sum()
            print("start", self.ngroups)

            base_inds_tmp = []
            base_tmp = []
            check_inds_tmp = []
            check_tmp = []
            for jj in range(num_check):  #  in zip(base_groups, check_groups):
                base_ind_here = median_inds[base_inds[further_consider][jj]]
                check_ind_here = median_inds[check_inds[further_consider][jj]]
                base_group =  self.groups[base_ind_here]
                check_group =  self.groups[check_ind_here]
                if base_group == check_group:
                    continue
                base_group_params = self.get_group(base_group)
                check_group_params = self.get_group(check_group)

                smaller_group = base_group_params if base_group_params.shape[0] < check_group_params.shape[0] else check_group_params
                larger_group = base_group_params if base_group_params.shape[0] > check_group_params.shape[0] else check_group_params

                smaller_group_median = smaller_group[np.argsort(smaller_group[:, 1])][int(smaller_group.shape[0] / 2)]

                smaller_group_in = self.current_info.gb_info["transform"].both_transforms(np.tile(smaller_group_median, (larger_group.shape[0], 1)))
                larger_group_in = self.current_info.gb_info["transform"].both_transforms(larger_group)

                base_tmp.append(smaller_group_in)
                base_inds_tmp.append(np.repeat(base_group, smaller_group_in.shape[0]))
                check_tmp.append(larger_group_in)
                check_inds_tmp.append(np.repeat(check_group, smaller_group_in.shape[0]))
                if jj % 300 == 0:
                    print(jj, num_check)

            base_tmp = np.concatenate(base_tmp, axis=0)
            base_inds_tmp = np.concatenate(base_inds_tmp)
            check_tmp = np.concatenate(check_tmp, axis=0)
            check_inds_tmp = np.concatenate(check_inds_tmp)

            overlap_here = self.get_overlap(base_tmp, check_tmp)
            
            if np.any(overlap_here > 0.9):
                base_inds_keep = base_inds_tmp[overlap_here > 0.9]
                check_inds_keep = check_inds_tmp[overlap_here > 0.9]
                
                for i in range(base_inds_keep.shape[0]):
                    if base_inds_keep[i] == check_inds_keep[i]:
                        continue
                    self.groups[self.groups == check_inds_keep[i]] = base_inds_keep[i]
                    base_inds_keep[base_inds_keep == check_inds_keep[i]] = base_inds_keep[i]
                    # must be last
                    check_inds_keep[check_inds_keep == check_inds_keep[i]] = base_inds_keep[i]
                    
        self.reset_groups_to_range()
        
    def perform_grouping(self) -> None:
        all_samples_base = self.params

        # walker_index_base = np.repeat(np.arange(gb_snrs.shape[0])[:, None], gb_snrs.shape[1], axis=-1)[(gb_snrs > 7.0) & (gb_inds)]

        # sort by frequency
        sort = np.argsort(all_samples_base[:, 1])[::-1]
        all_samples = all_samples_base[sort]
        groups = self.groups[sort]
        self.params = all_samples
        self.groups = groups

        # walker_index = walker_index_base[sort]

        all_samples_in = self.params_in

        median_group_inds = self.get_group_median_inds(all_samples_in, groups)

        # iterative neighbor grouping 

        # we fix the lengths because we are not going to remove them 
        # from consideration in case they will match better to 
        # something misisng on the next iteration.
        assigned_groups = self.groups.copy()
        previous_group_number = np.inf
        new_group_number = 0
        fill_minus_one = True
        stopper = 0
        inds_still_going = np.zeros(all_samples_in.shape[0], dtype=bool)
        inds_still_going[median_group_inds] = True

        roll_number = 1
        num_iters_converged = 0
        num_iters_converged_total = 0
        runit = True
        iteration = 0
        while runit:
            params_base_in = all_samples_in.copy()[inds_still_going]
            base_inds = np.arange(all_samples_in.shape[0])[inds_still_going]

            roll_number_here = iteration % roll_number if iteration > 0 else 1
            # the roll actually won't do anything on the other side since the overlap is zero
            check_inds = np.roll(base_inds.copy(), roll_number_here)

            params_check_in = np.roll(params_base_in.copy(), roll_number_here, axis=0)

            overlap = self.get_overlap(params_base_in, params_check_in)
            still_good = (np.abs(params_base_in[:, 1] - params_check_in[:, 1]) / self.current_info.general_info["df"]).astype(int) < (N.get() / 2)    
            
            self.gb.gpus = None

            if roll_number == 1:

                start_inds_potential_group = np.where(overlap < 0.9)[0]
                num_per_group = np.diff(start_inds_potential_group)
                num_per_group = np.concatenate([num_per_group, np.array([overlap.shape[0] - start_inds_potential_group[-1]])])

                groups_now = np.repeat(np.arange(start_inds_potential_group.shape[0]), repeats=num_per_group)
                
                if iteration == 0:
                    assigned_groups[:] = groups_now
                else:
                    uni, uni_inverse = np.unique(assigned_groups, return_inverse=True)
                    assigned_groups[:] = np.arange(uni.shape[0])[uni_inverse]

            else:
                keep = overlap > 0.9
                if np.any(keep):
                    base_group = assigned_groups[base_inds[keep]]
                    check_group = assigned_groups[check_inds[keep]]

                    args = []
                    inds = []
                    bg_vals = []
                    for nn, (bg, cg) in enumerate(zip(base_group, check_group)):
                        args.append([nn, assigned_groups, bg, cg])
            
                    with mp.Pool(10) as pool:
                        output = pool.starmap(para_func, args)
                    
                    inds_tmp = np.concatenate([tmp[0] for tmp in output])
                    base_inds_map = np.concatenate([tmp[1] for tmp in output])
                    assigned_groups[inds_tmp] = base_inds_map
                    uni, uni_inverse = np.unique(assigned_groups, return_inverse=True)
                    assigned_groups[:] = np.arange(uni.shape[0])[uni_inverse]
            
                #check ends

                #group them based on overlap
                # for jj, (base_ind, check_ind)  in enumerate(zip(base_inds, check_inds)):
                #     if overlap[jj] > 0.9:
                #         if assigned_groups[base_ind] == -1 and assigned_groups[check_ind] == -1:
                #             new_group_number = assigned_groups.max().item() + 1
                #             assigned_groups[base_ind] = new_group_number
                #             assigned_groups[check_ind] = new_group_number
                #             continue
                #         elif assigned_groups[base_ind] == -1:
                #             assigned_groups[base_ind] = assigned_groups[check_ind]
                #         elif assigned_groups[check_ind] == -1:
                #             assigned_groups[check_ind] = assigned_groups[base_ind]
                #         elif assigned_groups[base_ind] != assigned_groups[check_ind]:
                #             # TODO: need to check joins, maybe get median frequency binary and then recheck matches
                #             # CHECK THIS: NEED TO LOOK AT WALKER INDEXES AND STUFF RELATED
                #             # joined to same group, use lower group number
                #             lower_group_num = assigned_groups[check_ind] if assigned_groups[check_ind] < assigned_groups[base_ind] else assigned_groups[base_ind]
                #             upper_group_num = assigned_groups[check_ind] if assigned_groups[check_ind] > assigned_groups[base_ind] else assigned_groups[base_ind]

                #             # change to lower group
                #             assigned_groups[assigned_groups == upper_group_num] = lower_group_num
                #             assigned_groups[assigned_groups >= upper_group_num] -= 1
                #     if jj % 10000 == 0:
                #         print("PROGRESS", jj, len(base_inds))

            inds_still_going[:] = False
            median_group_inds = self.get_group_median_inds(all_samples_in, assigned_groups)
            inds_still_going[np.asarray(median_group_inds)] = True
            # inds_still_going[assigned_groups == -1] = True

            # remove any groups that have filled the full amount possible
            group_inds, group_pop = self.get_group_pop(all_samples_in, assigned_groups)
            if self.samples_so_far > 36:
                breakpoint()
            groups_complete = group_inds[group_pop == self.samples_so_far]
            inds_still_going[np.in1d(assigned_groups, groups_complete)] = False

            new_group_number = np.unique(assigned_groups).shape[0]
            new_multi_group_number = (np.unique(assigned_groups, return_counts=True)[1] > 1).sum()
            print(iteration, self.group_indicator, "NUM:", new_group_number, new_multi_group_number, roll_number, roll_number_here, num_iters_converged, num_iters_converged_total, previous_group_number, new_group_number, runit, still_good.sum())
            
            if new_group_number == previous_group_number:
                num_iters_converged += 1
                num_iters_converged_total += 1
                if num_iters_converged == 10 + roll_number:
                    num_iters_converged = 0
                    roll_number += 1

                if num_iters_converged_total >= 30 + roll_number:
                    runit = False
                
            else:
                num_iters_converged = 0
                num_iters_converged_total = 0
                previous_group_number = new_group_number

            iteration += 1

            if iteration > 100000:
                raise ValueError("Did not finish in allowed number of iterations.")

        assigned_groups[assigned_groups == -1] = np.arange((assigned_groups == -1).sum()) + assigned_groups.max() + 1
        self.params = all_samples
        self.groups = assigned_groups
        breakpoint()

    def __add__(self, other_group: GBGrouping) -> GBGrouping:

        this_group_median_inds = self.get_group_median_inds(self.params_in, self.groups)
        other_group_median_inds = self.get_group_median_inds(other_group.params_in, other_group.groups)
        
        both_sets_of_params = np.concatenate([self.params[this_group_median_inds], other_group.params[other_group_median_inds]], axis=0)
        both_sets_of_groups = np.concatenate([self.groups[this_group_median_inds], other_group.groups[other_group_median_inds]], axis=0)
        both_sets_of_samples = np.concatenate([np.zeros_like(this_group_median_inds), np.ones_like(other_group_median_inds)], axis=0)

        this_median_f = self.params_in[this_group_median_inds, 1]
        other_median_f = other_group.params_in[other_group_median_inds, 1]

        this_sort = np.argsort(this_median_f)
        other_sort = np.argsort(other_median_f)

        inds_other_into_this = np.searchsorted(this_median_f[this_sort], other_median_f[other_sort], side="right") - 1
        
        tmp = inds_other_into_this
        tmp[tmp < 0] = 0
        inds_down = this_group_median_inds[this_sort][tmp]
        
        tmp = inds_other_into_this + 1
        tmp[tmp >= len(this_group_median_inds)] = len(this_group_median_inds) - 1
        inds_up = this_group_median_inds[this_sort][tmp]

        source_down = self.params_in[inds_down]
        source_up = self.params_in[inds_up]

        group_down = self.groups[inds_down]
        group_up = self.groups[inds_up]

        other_source = other_group.params_in[other_group_median_inds][other_sort]
        other_groups = other_group.groups[other_group_median_inds][other_sort]
        
        closest_ind = np.argmin(np.abs(other_source[:, 1][:, None] - np.array([source_down[:, 1], source_up[:, 1]]).T), axis=-1)
        closest_group = np.take_along_axis(np.array([group_down, group_up]).T, closest_ind[:, None], axis=1).squeeze()
        closest_source = np.take_along_axis(np.array([source_down, source_up]).transpose(1, 0, 2), closest_ind[:, None, None], axis=1).squeeze()
        
        
        overlap = self.get_overlap(closest_source, other_source)
        

        immediate_keep = (overlap >= 0.9)
        other_new_groups = (other_groups + self.groups.max() + 1) * (~immediate_keep) + closest_group * immediate_keep

        still_here_other_source = other_source[~immediate_keep]

        base_inds = []
        check_inds = []
        base_params_in = []
        check_params_in = []
        for i, shos in zip(other_group_median_inds[other_sort], still_here_other_source):
            inds = np.where(np.abs(this_median_f[this_sort] - shos[1]) < 20 * self.current_info.general_info["df"])[0]
            
            check_inds.append(inds)
            base_inds.append(np.full_like(inds, i))
            check_params_in.append(self.params_in[this_sort][inds])
            base_params_in.append(np.tile(shos, (inds.shape[0], 1)))
        
        check_inds = np.concatenate(check_inds)
        base_inds = np.concatenate(base_inds)
        check_params_in = np.concatenate(check_params_in, axis=0)
        base_params_in = np.concatenate(base_params_in, axis=0)
        

        overlap_2 = self.get_overlap(base_params_in, check_params_in)
        keep_2 = overlap_2 >= 0.9
        bpi_keep = base_params_in[keep_2]
        cpi_keep = check_params_in[keep_2]
        base_inds_keep = base_inds[keep_2]
        check_inds_keep = check_inds[keep_2]
        overlap_2_keep = overlap_2[keep_2]

        uni, uni_count = np.unique(base_inds_keep, return_counts=True)
        fix_base = uni[uni_count > 1]
        
        for fix_base_i in fix_base:
            overlap_fix_base_i = overlap_2_keep[base_inds_keep == fix_base_i]
            inds_fix = np.where(base_inds_keep == fix_base_i)[0]
            inds_remove = inds_fix[np.delete(np.arange(len(inds_fix)), overlap_fix_base_i.argmax())]
            bpi_keep = np.delete(bpi_keep, inds_remove, axis=0)
            cpi_keep = np.delete(cpi_keep, inds_remove, axis=0)
            base_inds_keep = np.delete(base_inds_keep, inds_remove, axis=0)
            check_inds_keep = np.delete(check_inds_keep, inds_remove, axis=0)
            overlap_2_keep = np.delete(overlap_2_keep, inds_remove, axis=0)
        
        other_new_groups[base_inds_keep] = self.groups[check_inds_keep]     
        for base_ind_i in np.unique(base_inds):
            if base_ind_i in base_inds_keep:
                continue
            else:
                best_group_tmp = check_inds[base_inds == base_ind_i][overlap_2[base_inds == base_ind_i].argmax()]
                best_group = self.groups[best_group_tmp]
                test_group_in = self.current_info.gb_info["transform"].both_transforms(self.get_group(best_group))
                tmp_tmp = other_group_median_inds[other_sort][base_ind_i]
                base_group_in = np.tile(other_group.params_in[tmp_tmp], (test_group_in.shape[0], 1))
                overlap_here = self.get_overlap(base_group_in, test_group_in)
                if np.any(overlap_here > 0.9):
                    other_new_groups[base_ind_i] = best_group
        
        if np.all(other_group.group_pop == 1):
            other_group.groups[other_group_median_inds[other_sort]] = other_new_groups

        else:
            for old_group, new_group in zip(other_group, other_new_groups):
                other_group.groups[other_group.groups == old_group] = new_group

        both_sets_of_params = np.concatenate([self.params, other_group.params], axis=0)
        both_sets_of_groups = np.concatenate([self.groups, other_group.groups], axis=0)
        both_sets_of_samples = np.concatenate([self.sample, other_group.sample], axis=0)
  
        uni, uni_inverse = np.unique(both_sets_of_groups, return_inverse=True)
        both_sets_of_groups[:] = np.arange(len(uni))[uni_inverse]

        new_grouping = GBGrouping(self.gb, self.current_info, both_sets_of_params, both_sets_of_samples, self.fake_data, self.psd, self.waveform_kwargs, groups=both_sets_of_groups, copy=True, group_indicator=self.group_indicator, original_sample_count=self.original_sample_count) # , samples_so_far=self.samples_so_far + other_group.samples_so_far)
        print("new grouping")
        
        # map the medians mapped
        return new_grouping 

    @property
    def snr(self) -> np.ndarray:
        waveform_kwargs = self.current_info.gb_info["waveform_kwargs"].copy()
    
        if "N" in waveform_kwargs:
            waveform_kwargs.pop("N")

        gpus_back = self.gb.gpus.copy()
        self.gb.gpus = None
        _ = self.gb.get_ll(self.params_in, self.fake_data, self.psd, **waveform_kwargs)
        self.gb.gpus = gpus_back
        return np.sqrt(self.gb.h_h.real.get())

    @property
    def group_pop(self) -> np.ndarray:
        uni, uni_inverse, uni_count = np.unique(self.groups, return_counts=True, return_inverse=True)
        group_pop = uni_count[uni_inverse]
        return group_pop

    @property
    def confidence(self) -> np.ndarray:
        if self.original_sample_count is None:
            raise ValueError("When requesting confidence measures, need to input original_sample_count when initializing class.")
        confidence = (self.group_pop / self.original_sample_count)
        return confidence

    def remove_low_count_groups(self, count: int=1) -> None:
        keep = self.group_pop > count
        self.params = self.params[keep]
        self.groups = self.groups[keep]
        self.sample = self.sample[keep]

def para_func(i, assigned_groups, bg, cg):
    inds_tmp = np.where(assigned_groups == cg)[0]
    base_out = np.full_like(inds_tmp, bg)
    if i % 10000 == 0:
        print(i)
    return [inds_tmp, base_out]

def gather_gb_samples_cat(current_info, gb_reader, psd_in, gpu, samples_keep=1, thin_by=1):

    gb = GBGPU(use_gpu=True)
    xp.cuda.runtime.setDevice(gpu)
   
    #fake_data = [xp.zeros_like(current_info.general_info["fd"], dtype=complex), xp.zeros_like(current_info.general_info["fd"], dtype=complex)]
    tmp = current_info.get_data_psd(only_max_ll=True)
    psd_in = [xp.asarray(tmp["psd"][0]), xp.asarray(tmp["psd"][1])]
    
    fake_data = [xp.asarray(tmp["data"][0]), xp.asarray(tmp["data"][1])]
    groups_sample = []
    
    gb_file = gb_reader.filename

    import matplotlib.pyplot as plt

    groups = []
    st = time.perf_counter()
    read_in_success = False
    max_tries = 100
    current_try = 0

    while not read_in_success:
        try:
            with h5py.File(gb_file, "r") as fp:
                iteration_h5 = fp["mcmc"].attrs["iteration"]
                gb_samples = fp["mcmc"]["chain"]["gb"][iteration_h5 - samples_keep:iteration_h5, 0, :, :, :]
                gb_inds = fp["mcmc"]["inds"]["gb"][iteration_h5 - samples_keep:iteration_h5, 0, :, :]
            read_in_success = True
        except (BlockingIOError, OSError):
            print("Failed open")
            time.sleep(10.0)
            current_try += 1
            if current_try > max_tries:
                raise BlockingIOError("Tried to read in data file many times, but to no avail.")

    gb_samples = gb_samples.reshape(-1, gb_samples.shape[-2], gb_samples.shape[-1])
    gb_inds = gb_inds.reshape(-1, gb_inds.shape[-1])
    gb_sample_inds = np.repeat(np.arange(gb_inds.shape[0])[:, None], gb_inds.shape[1], axis=1)
    
    transform_fn = current_info.gb_info["transform"]


    test_bins_for_snr = gb_samples[gb_inds]

    transform_fn = current_info.gb_info["transform"]
    test_bins_for_snr_in = transform_fn.both_transforms(test_bins_for_snr)
    gb.d_d = 0.0

    waveform_kwargs = current_info.gb_info["waveform_kwargs"].copy()
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    gb.gpus = None
    _ = gb.get_ll(test_bins_for_snr_in, fake_data, psd_in, **waveform_kwargs)
    
    optimal_snr = gb.h_h.real ** (1/2)

    gb_snrs = np.full(gb_inds.shape, -1e10)
    gb_snrs[gb_inds] = optimal_snr.get()

    st1 = time.perf_counter()
    gb.gpus = [gpu]
    
    groups = GBGrouping(gb, current_info, gb_samples[gb_inds & (gb_snrs > 7.0)], gb_sample_inds[gb_inds & (gb_snrs > 7.0)], fake_data, psd_in, waveform_kwargs, copy=True, groups=-np.ones((gb_inds & (gb_snrs > 7.0)).sum(), dtype=int), group_indicator=-1, original_sample_count=samples_keep * 36)

    for i in range(1, gb_samples.shape[0]):
        groups.group_based_on_sample(i)
        print(i, gb_samples.shape[0], time.perf_counter() - st1)

    previous_ngroups = -100
    new_ngroups = -101
    while True:  # new_ngroups != previous_ngroups:
        groups.consolidate_cat()
        new_ngroups = groups.ngroups
        print("CHECK:", new_ngroups, previous_ngroups)
        if new_ngroups == previous_ngroups:
            groups.stop = True
            groups.consolidate_cat(final_consolidation=True)

            new_ngroups = groups.ngroups

            if new_ngroups == previous_ngroups:
                break
            previous_ngroups = -100
            new_ngroups = -101
    
        previous_ngroups = new_ngroups

    groups.prune(remove_singles=True)
    
    st1 = time.perf_counter()
    ldc_source_file = "LDC2_sangria_training_v2.h5"  # "gb-unblinded.h5"  # 
    with h5py.File(ldc_source_file, "r") as fp:
        dgbs = fp["sky"]["dgb"]["cat"][:]
        vgbs = fp["sky"]["vgb"]["cat"][:]
        igbs = fp["sky"]["igb"]["cat"][:]
        
        dgbs = dgbs[np.argsort(dgbs["Frequency"].squeeze())[::-1]][1:]
        vgbs = vgbs[np.argsort(vgbs["Frequency"].squeeze())[::-1]]
        igbs = igbs[np.argsort(igbs["Frequency"].squeeze())[::-1]]

    dgbs_in = np.array([
        dgbs["Amplitude"],
        dgbs["Frequency"],
        dgbs["FrequencyDerivative"],
        np.zeros_like(dgbs["FrequencyDerivative"]),
        dgbs["InitialPhase"],
        dgbs["Inclination"],
        dgbs["Polarization"],
        dgbs["EclipticLongitude"],
        dgbs["EclipticLatitude"],
    ]).T.squeeze()

    igbs_in = np.array([
        igbs["Amplitude"],
        igbs["Frequency"],
        igbs["FrequencyDerivative"],
        np.zeros_like(igbs["FrequencyDerivative"]),
        igbs["InitialPhase"],
        igbs["Inclination"],
        igbs["Polarization"],
        igbs["EclipticLongitude"],
        igbs["EclipticLatitude"],
    ]).T.squeeze()

    vgbs_in = np.array([
        vgbs["Amplitude"],
        vgbs["Frequency"],
        vgbs["FrequencyDerivative"],
        np.zeros_like(vgbs["FrequencyDerivative"]),
        vgbs["InitialPhase"],
        vgbs["Inclination"],
        vgbs["Polarization"],
        vgbs["EclipticLongitude"],
        vgbs["EclipticLatitude"],
    ]).T.squeeze()

    gbs_check_in = np.concatenate([dgbs_in, igbs_in, vgbs_in], axis=0)

    gb.gpus = [xp.cuda.runtime.getDevice()]
    old_gpus = gb.gpus.copy()
    gb.gpus = None

    _ = gb.get_ll(gbs_check_in, fake_data, psd_in, **waveform_kwargs)
    gb.gpus = old_gpus
    optimal_snr_check = gb.h_h.real.get() ** (1/2)

    gbs_check = gbs_check_in[optimal_snr_check > 7.0][:, np.array([0, 1, 2, 4, 5, 6, 7, 8])]

    gbs_check[:, 1] *= 1e3
    gbs_check[:, 4] = np.cos(gbs_check[:, 4])
    gbs_check[:, 7] = np.sin(gbs_check[:, 7])

    inj_input_params = np.concatenate([groups.params[groups.groups != -1].copy(), gbs_check])
    inj_input_samples = np.concatenate([groups.sample[groups.groups != -1], -np.ones(gbs_check.shape[0], dtype=int)])
    inj_input_groups = np.concatenate([groups.groups[groups.groups != -1], np.arange(gbs_check.shape[0], dtype=int)])
    fake_inj_input_groups = np.concatenate([-np.ones_like(groups.groups[groups.groups != -1]), np.arange(gbs_check.shape[0], dtype=int)])
    
    gb_check_groups = GBGrouping(gb, current_info, inj_input_params, inj_input_samples, fake_data, psd_in, waveform_kwargs, copy=True, groups=fake_inj_input_groups, group_indicator=-1, original_sample_count=samples_keep * 36)
    gb_check_groups.group_based_on_sample(-1, store_best_overlap=True, remove_grouped=False)

    print("save injection comparison")
    
    found_params_out = gb_check_groups.best_overlap_check_params
    found_overlaps_out = gb_check_groups.best_overlap
    inj_params_out = gb_check_groups.best_overlap_base_params

    groups_match_out = groups.groups[gb_check_groups.best_overlap_inds]
    confidence_out = groups.confidence[gb_check_groups.best_overlap_inds]

    injection_check_out = np.concatenate([found_params_out[:, np.array([0, 1, 2, 4, 5, 6, 7, 8])], groups_match_out[:, None], inj_params_out[:, np.array([0, 1, 2, 4, 5, 6, 7, 8])], found_overlaps_out[:, None], confidence_out[:, None]], axis=1)
    keys = ["gf-Amplitude", "gf-Frequency", "gf-Frequency Derivative", "gf-Initial Phase", "gf-Inclination", "gf-Polarization", "gf-Ecliptic Longitude", "gf-Eclipitc Latitude", "gf-group", "inj-Amplitude", "inj-Frequency", "inj-Frequency Derivative", "inj-Initial Phase", "inj-Inclination", "inj-Polarization", "inj-Ecliptic Longitude", "inj-Eclipitc Latitude", "Best Overlap", "Confidence"]

    injection_check_out_dict = {key: tmp for key, tmp in zip(keys, injection_check_out.T)}
    df_inj_check = pd.DataFrame(injection_check_out_dict)
    df_inj_check.to_hdf(current_info.general_info["file_information"]["file_store_dir"] + current_info.general_info["file_information"]["base_file_name"] + "_injection_comp.h5", "gf_results")
    return groups
    # breakpoint()

    # first_sample = gb_samples[0].reshape(-1, 8)
    # first_sample_snrs = gb_snrs[0].flatten()

    # keep_map = []
    # binaries_in_not_transformed = []
    # binaries_for_test_not_transformed = []
    # binaries_in = []
    # binaries_for_test = []
    # num_so_far = 0
    # keep_going_in = []
    # base_snrs_going_in = []
    # test_snrs_going_in = []
    # for i, bin in enumerate(first_sample):
    #     #if i > 100:
    #     #    continue
    #     if first_sample_snrs[i] < 7.0:
    #         continue
    #     freq_dist = np.abs(bin[1] - gb_samples[1:, :, 1])
    #     snr_dist = np.abs(first_sample_snrs[i] - gb_snrs[1:])

    #     keep_going_in.append(i)
    #     keep_i = np.where((freq_dist < 1e-4) & (snr_dist < 20.0))

    #     base_snrs_going_in.append(np.repeat(first_sample_snrs[i], len(keep_i[0])))
    #     test_snrs_going_in.append(gb_snrs[1:][keep_i])
    #     keep_map.append([num_so_far + np.arange(len(keep_i[0])), keep_i])
    #     binaries_for_test_not_transformed.append(gb_samples[1:][keep_i])
    #     binaries_in_not_transformed.append(np.tile(bin, (len(keep_i[0]), 1)))
    #     binaries_for_test.append(transform_fn.both_transforms(gb_samples[1:][keep_i]))
    #     binaries_in.append(transform_fn.both_transforms(np.tile(bin, (len(keep_i[0]), 1))))

    #     num_so_far += len(keep_i[0])
        

    # bins_fin_test_in = np.concatenate(binaries_for_test)
    # bins_fin_base_in = np.concatenate(binaries_in)
    # bins_fin_test_in_not_transformed = np.concatenate(binaries_for_test_not_transformed)
    # bins_fin_base_in_not_transformed = np.concatenate(binaries_in_not_transformed)
    # snrs_fin_test_in = np.concatenate(test_snrs_going_in)
    # snrs_fin_base_in = np.concatenate(base_snrs_going_in)

    # N_vals = get_N(bins_fin_test_in[:, 0], bins_fin_test_in[:, 1], YEAR, oversample=4)
    # fake_data_swap = [[fake_data[0]], [fake_data[1]]]
    # psd_in_swap = [[psd_in[0]], [psd_in[1]]]
    # gb.gpus = [gpu]
    
    # ll_diff = np.zeros(bins_fin_base_in.shape[0])
    # batch_size = int(1e7)

    # inds_split = np.arange(0, bins_fin_base_in.shape[0] + batch_size, batch_size)

    # for jjj, (start_ind, end_ind) in enumerate(zip(inds_split[:-1], inds_split[1:])):
    #     waveform_kwargs["N"] = xp.asarray(N_vals[start_ind:end_ind])
    
    #     _ = gb.swap_likelihood_difference(bins_fin_base_in[start_ind:end_ind],bins_fin_test_in[start_ind:end_ind],fake_data_swap,psd_in_swap,start_freq_ind=0,data_length=len(fake_data[0]),data_splits=[np.array([0])],**waveform_kwargs,)

    #     # ll_diff[start_ind:end_ind] = (-1/2 * (gb.add_add + gb.remove_remove - 2 * gb.add_remove).real).get()
    #     ll_diff[start_ind:end_ind] = (gb.add_remove.real / np.sqrt(gb.add_add * gb.remove_remove)).get()
    #     print(start_ind, len(inds_split) - 1)

    # keep_groups = []
    # keep_group_number = []
    # keep_group_samples = []
    # keep_group_snrs = []
    # keep_group_sample_id = []
    # max_number = 0
    # num_so_far_gather = 0

    # for i, keep_map_i in enumerate(keep_map):
    #     # TODO: check this?
    #     (keep_inds, keep_map_back) = keep_map_i
    #     if len(keep_inds) == 0:
    #         continue
    #     ll_diff_i = ll_diff[keep_inds]
    #     group_test = (ll_diff_i > 0.5)  # .get()
    #     num_grouping = group_test.sum()
        
    #     in_here = keep_going_in[i]
    #     sample_map = np.concatenate([np.array([0]), keep_map_back[0][group_test] + 1])
    #     binary_map = np.concatenate([np.array([in_here]), keep_map_back[1][group_test]])
    #     try:
    #         binary_samples = np.concatenate([np.array([bins_fin_base_in_not_transformed[keep_inds[0]]]), bins_fin_test_in_not_transformed[keep_inds][group_test]])
    #     except IndexError:
    #         breakpoint()

    #     binary_snrs = np.concatenate([np.array([snrs_fin_base_in[keep_inds[0]]]), snrs_fin_test_in[keep_inds][group_test]])
        
    #     if not np.all(gb_inds_left[sample_map, binary_map]):
    #         # TODO: fix this
    #         ind_fix = np.where(~gb_inds_left[sample_map, binary_map])
    #         sample_map = np.delete(sample_map, ind_fix)
    #         binary_map = np.delete(binary_map, ind_fix)

    #     if (num_grouping + 1) > 2:
    #         # remove them from possible future grouping
    #         gb_inds_left[sample_map, binary_map] = False
        
    #         keep_group_number.append(num_grouping + 1)
    #         if num_grouping + 1 > max_number:
    #             max_number = num_grouping + 1


    #         keep_group_samples.append(binary_samples)
    #         keep_group_snrs.append(binary_snrs)
    #         keep_groups.append((num_grouping, sample_map, binary_map))

    #         num_so_far_gather += 1

    #         keep_group_sample_id.append(np.asarray([np.full(binary_samples.shape[0], num_so_far_gather), np.full(binary_samples.shape[0], (num_grouping + 1) / gb_samples.shape[0])]))

    # output_information = []
    # for i in range(len(keep_group_sample_id)):
    #     output_information.append(np.concatenate([keep_group_sample_id[i].T, keep_group_snrs[i][:, None], keep_group_samples[i]], axis=1))
    
    # if len(output_information) > 0:
    #     output_information = np.concatenate(output_information, axis=0)
    # return output_information



def gather_gb_samples(current_info, reader, sens_mat, gpu, samples_keep=1, thin_by=1, snr_lim_first_cut=6.0, snr_lim_second_cut=5.0, overlap_lim=0.5):

    gb = GBGPU(use_gpu=True, gpus=[gpu])
    xp.cuda.runtime.setDevice(gpu)
    fd = current_info.general_info["fd"]
    fake_data = [xp.zeros((2, fd.shape[0]), dtype=complex)]
    psd_in = [xp.asarray(sens_mat.invC.copy())]
    
    gb_samples = reader.get_chain(branch_names=["gb"], temp_index=0, discard=reader.iteration - samples_keep, thin=thin_by)["gb"]
    gb_inds = reader.get_inds(branch_names=["gb"], temp_index=0, discard=reader.iteration - samples_keep, thin=thin_by)["gb"]
                
    gb_samples = gb_samples.reshape(-1, gb_samples.shape[-2], gb_samples.shape[-1])
    gb_inds = gb_inds.reshape(-1, gb_inds.shape[-1])

    test_bins_for_snr = gb_samples[gb_inds]

    transform_fn = current_info.source_info['gb']["transform"]
    test_bins_for_snr_in = transform_fn.both_transforms(test_bins_for_snr)
    gb.d_d = 0.0

    waveform_kwargs = current_info.source_info['gb']["waveform_kwargs"].copy()
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    _ = gb.get_ll(test_bins_for_snr_in, fake_data, psd_in, **waveform_kwargs)

    optimal_snr = gb.h_h.real ** (1/2)

    gb_snrs = np.full(gb_inds.shape, -1e10)
    gb_snrs[gb_inds] = optimal_snr.get()
    gb_inds_tmp = gb_inds.copy()

    keep_groups = []
    random_samples = np.random.choice(np.arange(len(gb_samples)), len(gb_samples) - 1, replace=False)
    for samp_i in range(len(gb_samples) - 1):
        
        first_sample = gb_samples[random_samples[samp_i]].reshape(-1, 8)
        first_sample_snrs = gb_snrs[random_samples[samp_i]].flatten()
        inds_keep_i = np.delete(np.arange(gb_samples.shape[0]), random_samples[:samp_i+1])
        gb_samples_in = gb_samples[inds_keep_i]
        # gb_inds_tmp not gb_inds because we adjust that overtime 
        # to reflect binaries already taken
        gb_inds_in = gb_inds_tmp[inds_keep_i]
        gb_snrs_in = gb_snrs[inds_keep_i]
        
        keep_map = []
        binaries_for_test = []
        binaries_base_sample = []
        num_so_far = 0
        keep_going_in = []

        for i, binary in enumerate(first_sample):
            if first_sample_snrs[i] < snr_lim_first_cut:
                continue
            freq_dist = np.abs(binary[1] - gb_samples_in[:, :, 1])
            # snr_dist covers binaries that have inds False
            snr_dist = np.abs(first_sample_snrs[i] - gb_snrs_in)

            keep_going_in.append(i)
            keep_i = np.where((freq_dist < 1e-4) & (snr_dist < 20.0) & (gb_snrs_in >= snr_lim_second_cut) & gb_inds_in)

            keep_map.append([num_so_far + np.arange(len(keep_i[0])), keep_i])
            binaries_for_test.append(gb_samples_in[keep_i])
            binaries_base_sample.append(np.tile(binary, (len(keep_i[0]), 1)))
            num_so_far += len(keep_i[0])
            
        binaries_for_test = np.concatenate(binaries_for_test, axis=0)
        binaries_base_sample = np.concatenate(binaries_base_sample, axis=0)
        band_inds = np.searchsorted(current_info.source_info["gb"]["band_edges"], binaries_for_test[:, 1] / 1e3, side="right") - 1

        N_vals = current_info.source_info["gb"]["band_N_vals"][band_inds]
        
        batch_size = int(1e7)

        # reset data and psd
        fake_data = [xp.zeros((2, fd.shape[0]), dtype=complex)]
        psd_in = [xp.asarray(sens_mat.invC.copy())]
        
        inds_split = np.arange(0, binaries_for_test.shape[0] + batch_size, batch_size)
        ll_diff = np.zeros(binaries_for_test.shape[0])
        for jjj, (start_ind, end_ind) in enumerate(zip(inds_split[:-1], inds_split[1:])):
            waveform_kwargs["N"] = xp.asarray(N_vals[start_ind:end_ind])

            binaries_for_test_batch_in = transform_fn.both_transforms(binaries_for_test[start_ind:end_ind])
            binaries_base_sample_batch_in = transform_fn.both_transforms(binaries_base_sample[start_ind:end_ind])
            if fd[0] != 0.0:
                raise NotImplementedError("Need to work on if start_freq_ind is not zero.")

            _ = gb.swap_likelihood_difference(binaries_for_test_batch_in, binaries_base_sample_batch_in, fake_data, psd_in, phase_marginalize=True, start_freq_ind=0, **waveform_kwargs)

            # ll_diff[start_ind:end_ind] = (-1/2 * (gb.add_add + gb.remove_remove - 2 * gb.add_remove).real).get()
            ll_diff[start_ind:end_ind] = (gb.add_remove.real / np.sqrt(gb.add_add.real * gb.remove_remove.real)).get()
            print(start_ind, len(inds_split) - 1)

        for i, keep_map_i in enumerate(keep_map):
            # TODO: check this?
            (keep_inds, keep_map_back) = keep_map_i
            if len(keep_inds) == 0:
                continue
            
            ll_diff_i = ll_diff[keep_inds]
            group_test = (ll_diff_i > overlap_lim)  # .get()
            num_grouping = group_test.sum()
            if num_grouping == 0:
                continue

            in_here = keep_going_in[i]

            # gb_inds_in[keep_map_back][group_test] = 
            ind1 = inds_keep_i[keep_map_back[0][group_test]]
            ind2 = keep_map_back[1][group_test]

            if np.any(~gb_inds_tmp[ind1, ind2]):
                _remove_here = np.arange(group_test.shape[0])[group_test][~gb_inds_tmp[ind1, ind2]]
                group_test[_remove_here] = False
                num_grouping = group_test.sum()
                if num_grouping == 0:
                    continue

                ind1 = inds_keep_i[keep_map_back[0][group_test]]
                ind2 = keep_map_back[1][group_test]

            if not np.all(gb_inds_tmp[ind1, ind2]):
                breakpoint()
            gb_inds_tmp[ind1, ind2] = False
            
            group = np.concatenate([first_sample[in_here][None, :], gb_samples_in[keep_map_back][group_test]], axis=0)
            if len(group) == 1:
                breakpoint()
            if not np.any(np.all(((gb_snrs_in > snr_lim_second_cut) & gb_inds_in)[keep_map_back][group_test])):
                breakpoint()
            
            if (num_grouping + 1) > gb_samples.shape[0]:
                # remove them from possible future grouping
                breakpoint()

            keep_groups.append(group)
        print(f"samp_i: {samp_i + 1}, num: {gb_inds_tmp.sum()}")
    num_in_group = [len(group_i) for group_i in keep_groups]
    # need to consolidate
    breakpoint()
    return keep_groups


# np.savetxt("output_samples_from_gbs_grouped.txt", output_information, header="id, confidence, amp, f0, fdot, phi0, inc, psi, lam, beta", delimiter=",")
# exit()
# test_bins = np.asarray([])


# breakpoint()

# where_all = np.where(gb_inds.sum(axis=0) == gb_inds.shape[0])[0]

# st = time.perf_counter()
# i = 0

# batch_size = 100
# templates = xp.zeros((batch_size, 2, fd.shape[0]), dtype=complex)

# batches = np.arange(0, len(where_all) + batch_size, batch_size)
# keep_samples = []
# for j in range(len(batches) - 1):
#     batch_start = batches[j]
#     batch_end = batches[j + 1]

#     batch_inds = np.arange(batch_start, batch_end)
#     try:
#         where_all_inds = where_all[batch_inds]
#     except IndexError:
#         continue 
#     gb.d_d = 0.0
#     templates[:] = 0.0 + 0.0 * 1j

#     mean_fs = np.mean(gb_samples[:, where_all_inds, 1], axis=0)
#     injection_ind = np.argmin(np.abs(gb_samples[:, where_all_inds, 1] - mean_fs[None, :]), axis=0)

#     injections = gb_samples[(injection_ind, where_all_inds)]
#     injections_in = transform_fn.both_transforms(injections)
#     data_index = xp.arange(injections.shape[0], dtype=xp.int32)
#     gb.generate_global_template(injections_in, data_index, templates, **waveform_kwargs)
#     gb.get_ll(injections_in, fake_data, psd_in, **waveform_kwargs)
#     gb.d_d = gb.h_h.copy()

#     # data_in = [templates[:, 0].flatten().copy(), templates[:, 1].flatten().copy()]
#     check = gb.get_ll(injections_in, templates, psd_in, data_index=data_index, **waveform_kwargs)
#     del data_index
#     assert np.allclose(check, np.zeros_like(check))

#     coords_orig = gb_samples[:, where_all_inds].reshape(-1, ndim)
#     coords_in = transform_fn.both_transforms(coords_orig)
#     data_index = xp.repeat(xp.arange(where_all_inds.shape[0], dtype=xp.int32)[None, :], gb_samples.shape[0], axis=0).flatten().astype(xp.int32)

#     gb.d_d = xp.asarray(gb.d_d)[data_index]
#     ll_match = gb.get_ll(coords_in, templates, psd_in, data_index=data_index, **waveform_kwargs).reshape(-1, injections.shape[0])

#     ll_mins = ll_match.min(axis=0)
#     snrs_mins = (gb.h_h.reshape(gb_samples[:, where_all_inds].shape[:-1]).real ** (1/2)).min(axis=0).get()

#     keep = (ll_mins > -20.0) & (np.std(ll_match, axis=0) < 10.0) & (snrs_mins > 7.0)

#     keep_samples.append(where_all_inds[keep])
#     print(j, len(batches) - 1)
#     # gb_samples_here = gb_samples[]
#     continue
#     for k in range(gb_keep_inds.shape[0]):
#         if k == i:
#             continue
        
#         ll_match_best = ll_match.max(axis=-1)
#         ll_match_best_ind = ll_match.argmax(axis=-1)

#         keep_best = ll_match_best > -20.0
#         keep_ll_match_best = ll_match_best[keep_best]
#         keep_ll_math_best_ind = ll_match_best_ind[keep_best]

#         if len(keep_ll_math_best_ind) != len(np.unique(keep_ll_math_best_ind)):
#             breakpoint()
            
#         try:
#             which_source = batch_inds[np.arange(coords_orig.shape[0])[keep_ll_math_best_ind]]
#         except IndexError:
#             breakpoint()
#         map_best.append([which_source, keep_ll_math_best_ind, keep_ll_match_best])
#         print(j, k)

# keep_samples = np.concatenate(keep_samples)

# gb.d_d = 0.0

# gb_inds[:, keep_samples] = False

# gb_inds_flat = gb_inds.flatten()
# kept = np.arange(len(gb_inds_flat))[gb_inds_flat]

# params_in = transform_fn.both_transforms(gb_samples.reshape(-1, ndim)[kept])
# gb.get_ll(params_in, fake_data, psd_in, **waveform_kwargs)
# keep_snr = kept[gb.h_h.real.get() ** (1/2) > 7.0]

# gb_keep_inds = np.zeros_like(gb_inds_flat, dtype=bool)
# gb_keep_inds[keep_snr] = True
# gb_keep_inds = gb_keep_inds.reshape(-1, gb_inds.shape[-1])
# gb_keep_samples = gb_samples.reshape(-1, gb_inds.shape[-1], ndim)




# batch_size = 100

# for i in range(gb_keep_samples.shape[0])[:5]:

#     gb_samples_here = gb_keep_samples[i][gb_keep_inds[i]]

#     batches = np.arange(0, gb_samples_here.shape[0] + batch_size, batch_size)
#     map_best = []
#     for j in range(len(batches) - 1):
#         batch_start = batches[j]
#         batch_end = batches[j + 1]

#         batch_inds = np.arange(batch_start, batch_end)
        
#         gb.d_d = 0.0
#         templates[:] = 0.0 + 0.0 * 1j

#         try:
#             injections = gb_samples_here[batch_inds]
#         except IndexError:
#             breakpoint()
#         injections_in = transform_fn.both_transforms(injections)
#         data_index = xp.arange(injections.shape[0], dtype=xp.int32)
#         gb.generate_global_template(injections_in, data_index, templates, **waveform_kwargs)
#         gb.get_ll(injections_in, fake_data, psd_in, **waveform_kwargs)
#         gb.d_d = gb.h_h.copy()

#         # data_in = [templates[:, 0].flatten().copy(), templates[:, 1].flatten().copy()]
#         check = gb.get_ll(injections_in, templates, psd_in, data_index=data_index, **waveform_kwargs)
#         del data_index
#         assert np.allclose(check, np.zeros_like(check))

#         # gb_samples_here = gb_samples[]

#         for k in range(gb_keep_inds.shape[0]):
#             if k == i:
#                 continue

#             coords_orig = gb_keep_samples[k][gb_keep_inds[k]]
#             coords_repeated = np.repeat(coords_orig, len(batch_inds), axis=0)
#             coords_in = transform_fn.both_transforms(coords_repeated)

#             data_index = xp.repeat(xp.arange(injections.shape[0], dtype=xp.int32)[None, :], coords_orig.shape[0], axis=0).flatten().astype(xp.int32)

#             gb.d_d = xp.asarray(gb.d_d)[data_index]
#             ll_match = gb.get_ll(coords_in, templates, psd_in, data_index=data_index, **waveform_kwargs).reshape(-1, injections.shape[0])

#             ll_match_best = ll_match.max(axis=-1)
#             ll_match_best_ind = ll_match.argmax(axis=-1)

#             ind_inj = np.tile(np.arange(injections.shape[0]), (len(ll_match_best), 1))
#             ll_match_inj_ind = ind_inj[(np.arange(ll_match_best_ind.shape[0]), ll_match_best_ind)]

#             keep_best = ll_match_best > -20.0
#             keep_ll_match_best = ll_match_best[keep_best]
#             keep_ll_match_best_ind = ll_match_best_ind[keep_best]
#             coords_keep = coords_orig[keep_best]
#             keep_inj_ind = ll_match_inj_ind[keep_best]

#             # if len(keep_ll_match_best_ind) != len(np.unique(keep_ll_match_best_ind)):
#             #     breakpoint()
                
#             try:
#                 which_source = np.arange(coords_orig.shape[0])[keep_best]
#             except IndexError:
#                 breakpoint()

#             map_best.append([i, j, k, which_source, keep_ll_match_best_ind, keep_ll_match_best, np.abs(injections[keep_inj_ind][:, 1] - coords_keep[:, 1]) / injections[keep_inj_ind][:, 1]])
#             print(i, gb_keep_samples.shape[0], j, k, len(batches) - 1, k, gb_keep_inds.shape[0])

# np.save("output_check_match", np.asarray(map_best))
# et = time.perf_counter()
# print(et - st)
# breakpoint()
