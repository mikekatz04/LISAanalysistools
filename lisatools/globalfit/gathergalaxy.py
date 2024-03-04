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


def gather_gb_samples(current_info, gb_reader, psd_in, gpu, samples_keep=1, thin_by=1):

    gb = GBGPU(use_gpu=True)
    xp.cuda.runtime.setDevice(gpu)
   
    fake_data = [xp.zeros_like(current_info.general_info["fd"], dtype=complex), xp.zeros_like(current_info.general_info["fd"], dtype=complex)]
    
    step_index = slice(gb_reader.iteration - samples_keep, gb_reader.iteration, 1)
    temp_index = [0]
    slice_vals = (step_index, temp_index)
    discard_val = gb_reader.iteration - samples_keep if gb_reader.iteration - samples_keep >= 0 else 0
    
    gb_file = gb_reader.filename
    read_in_success = False
    max_tries = 100
    current_try = 0
    while not read_in_success:
        try:
            with h5py.File(gb_file, "r") as fp:
                iteration = fp["mcmc"].attrs["iteration"]
                gb_samples = fp["mcmc"]["chain"]["gb"][iteration - samples_keep:iteration, 0, :, :, :]
                gb_inds = fp["mcmc"]["inds"]["gb"][iteration - samples_keep:iteration, 0, :, :]
            read_in_success = True
        except BlockingIOError:
            print("Failed open")
            time.sleep(10.0)
            current_try += 1
            if current_try > max_tries:
                raise BlockingIOError("Tried to read in data file many times, but to no avail.")


    gb_samples = gb_samples.reshape(-1, gb_samples.shape[-2], gb_samples.shape[-1])
    gb_inds = gb_inds.reshape(-1, gb_inds.shape[-1])

    test_bins_for_snr = gb_samples[gb_inds]

    transform_fn = current_info.gb_info["transform"]
    test_bins_for_snr_in = transform_fn.both_transforms(test_bins_for_snr)
    gb.d_d = 0.0

    waveform_kwargs = current_info.gb_info["waveform_kwargs"].copy()
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    _ = gb.get_ll(test_bins_for_snr_in, fake_data, psd_in, **waveform_kwargs)

    optimal_snr = gb.h_h.real ** (1/2)

    gb_snrs = np.full(gb_inds.shape, -1e10)
    gb_snrs[gb_inds] = optimal_snr.get()

    gb_keep = gb_snrs > 8.0
    gb_inds_left = gb_inds.copy()

    first_sample = gb_samples[0].reshape(-1, 8)
    first_sample_snrs = gb_snrs[0].flatten()

    keep_map = []
    binaries_in_not_transformed = []
    binaries_for_test_not_transformed = []
    binaries_in = []
    binaries_for_test = []
    num_so_far = 0
    keep_going_in = []
    base_snrs_going_in = []
    test_snrs_going_in = []
    for i, bin in enumerate(first_sample):
        #if i > 100:
        #    continue
        if first_sample_snrs[i] < 7.0:
            continue
        freq_dist = np.abs(bin[1] - gb_samples[1:, :, 1])
        snr_dist = np.abs(first_sample_snrs[i] - gb_snrs[1:])

        keep_going_in.append(i)
        keep_i = np.where((freq_dist < 1e-4) & (snr_dist < 20.0))

        base_snrs_going_in.append(np.repeat(first_sample_snrs[i], len(keep_i[0])))
        test_snrs_going_in.append(gb_snrs[1:][keep_i])
        keep_map.append([num_so_far + np.arange(len(keep_i[0])), keep_i])
        binaries_for_test_not_transformed.append(gb_samples[1:][keep_i])
        binaries_in_not_transformed.append(np.tile(bin, (len(keep_i[0]), 1)))
        binaries_for_test.append(transform_fn.both_transforms(gb_samples[1:][keep_i]))
        binaries_in.append(transform_fn.both_transforms(np.tile(bin, (len(keep_i[0]), 1))))

        num_so_far += len(keep_i[0])
        

    bins_fin_test_in = np.concatenate(binaries_for_test)
    bins_fin_base_in = np.concatenate(binaries_in)
    bins_fin_test_in_not_transformed = np.concatenate(binaries_for_test_not_transformed)
    bins_fin_base_in_not_transformed = np.concatenate(binaries_in_not_transformed)
    snrs_fin_test_in = np.concatenate(test_snrs_going_in)
    snrs_fin_base_in = np.concatenate(base_snrs_going_in)

    N_vals = get_N(bins_fin_test_in[:, 0], bins_fin_test_in[:, 1], YEAR, oversample=4)
    fake_data_swap = [[fake_data[0]], [fake_data[1]]]
    psd_in_swap = [[psd_in[0]], [psd_in[1]]]
    gb.gpus = [gpu]
    
    ll_diff = np.zeros(bins_fin_base_in.shape[0])
    batch_size = int(1e3)

    inds_split = np.arange(0, bins_fin_base_in.shape[0] + batch_size, batch_size)

    for jjj, (start_ind, end_ind) in enumerate(zip(inds_split[:-1], inds_split[1:])):
        waveform_kwargs["N"] = xp.asarray(N_vals[start_ind:end_ind])
    
        _ = gb.swap_likelihood_difference(bins_fin_base_in[start_ind:end_ind],bins_fin_test_in[start_ind:end_ind],fake_data_swap,psd_in_swap,start_freq_ind=0,data_length=len(fake_data[0]),data_splits=[np.array([0])],**waveform_kwargs,)

        # ll_diff[start_ind:end_ind] = (-1/2 * (gb.add_add + gb.remove_remove - 2 * gb.add_remove).real).get()
        ll_diff[start_ind:end_ind] = (gb.add_remove.real / np.sqrt(gb.add_add * gb.remove_remove)).get()
        print(start_ind, len(inds_split) - 1)

    keep_groups = []
    keep_group_number = []
    keep_group_samples = []
    keep_group_snrs = []
    keep_group_sample_id = []
    max_number = 0
    num_so_far_gather = 0

    for i, keep_map_i in enumerate(keep_map):
        # TODO: check this?
        (keep_inds, keep_map_back) = keep_map_i
        if len(keep_inds) == 0:
            continue
        ll_diff_i = ll_diff[keep_inds]
        group_test = (ll_diff_i > 0.5)  # .get()
        num_grouping = group_test.sum()
        
        in_here = keep_going_in[i]
        sample_map = np.concatenate([np.array([0]), keep_map_back[0][group_test] + 1])
        binary_map = np.concatenate([np.array([in_here]), keep_map_back[1][group_test]])
        try:
            binary_samples = np.concatenate([np.array([bins_fin_base_in_not_transformed[keep_inds[0]]]), bins_fin_test_in_not_transformed[keep_inds][group_test]])
        except IndexError:
            breakpoint()

        binary_snrs = np.concatenate([np.array([snrs_fin_base_in[keep_inds[0]]]), snrs_fin_test_in[keep_inds][group_test]])
        
        if not np.all(gb_inds_left[sample_map, binary_map]):
            # TODO: fix this
            ind_fix = np.where(~gb_inds_left[sample_map, binary_map])
            sample_map = np.delete(sample_map, ind_fix)
            binary_map = np.delete(binary_map, ind_fix)

        if (num_grouping + 1) > 2:
            # remove them from possible future grouping
            gb_inds_left[sample_map, binary_map] = False
        
            keep_group_number.append(num_grouping + 1)
            if num_grouping + 1 > max_number:
                max_number = num_grouping + 1


            keep_group_samples.append(binary_samples)
            keep_group_snrs.append(binary_snrs)
            keep_groups.append((num_grouping, sample_map, binary_map))

            num_so_far_gather += 1

            keep_group_sample_id.append(np.asarray([np.full(binary_samples.shape[0], num_so_far_gather), np.full(binary_samples.shape[0], (num_grouping + 1) / gb_samples.shape[0])]))

    output_information = []
    for i in range(len(keep_group_sample_id)):
        output_information.append(np.concatenate([keep_group_sample_id[i].T, keep_group_snrs[i][:, None], keep_group_samples[i]], axis=1))
    
    if len(output_information) > 0:
        output_information = np.concatenate(output_information, axis=0)
    return output_information


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

#     keep = (ll_mins > -50.0) & (np.std(ll_match, axis=0) < 10.0) & (snrs_mins > 7.0)

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

#             keep_best = ll_match_best > -50.0
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
