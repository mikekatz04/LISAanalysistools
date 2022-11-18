from multiprocessing.sharedctypes import Value
import cupy as xp
import time
import pickle
import shutil

mempool = xp.get_default_memory_pool()

from full_band_global_fit_settings import *
from single_mcmc_run import run_single_band_search
from lisatools.utils.multigpudataholder import MultiGPUDataHolder

import subprocess

import warnings
warnings.filterwarnings("ignore")

stop_here = True

class BasicResidualMGHLikelihood:
    def __init__(self, mgh):
        self.mgh = mgh
    def __call__(self, *args, supps=None, **kwargs):
        ll_temp = self.mgh.get_ll()
        overall_inds = supps["overal_inds"]
        return ll_temp[overall_inds]

data = [
    np.asarray(A_inj),
    np.asarray(E_inj),
]

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

# TODO: fix initial setup for mix where it backs up the likelihood

class PointGeneratorSNR:
    def __init__(self, generate_snr_ladder):
        self.generate_snr_ladder = generate_snr_ladder

    def __call__(self, size=(1,)):
        new_points = self.generate_snr_ladder.rvs(size=size)
        logpdf = self.generate_snr_ladder.logpdf(new_points)
        return (new_points, logpdf)

point_generator = PointGeneratorSNR(generate_snr_ladder)

snr_lim = 10.0
gpus = [4, 5]
gpus_for_equilibrate = [6, 7]

from lisatools.sampling.stopping import SearchConvergeStopping2

generating_priors = deepcopy(priors)

waveform_kwargs["start_freq_ind"] = start_freq_ind

#search_f_bin_lims = f0_lims[0] + df * np.arange(num_f_bins)[lim_inds]
#search_f_bin_lims[-1] = f0_lims[-1]

if sub_band_fails_file not in os.listdir():
    num_sub_band_fails = np.zeros(num_sub_bands, dtype=int)

else:
    num_sub_band_fails = np.load(sub_band_fails_file)

assert num_sub_band_fails.shape[0] == num_sub_bands

run_mix_first = False
# TODO: adjust the starting points each iteration to avoid issues at gaps
print(f"num_sub_bands: {num_sub_bands}")

# TODO:
# put together a class that holds list of 2 entries - flattened residual arrays (and for noise)
# this is for each gpu
# this class can do all the manipulations that would be wanted wthout adding memory!!!
# put this class inside all of the moves and maybe sub out likelihood function for sampler itself
# adjust group_index, data_index, noise_index based on the indexing to temperature and walker
# will need to adjust gbgpu to split properly based on group or dataindex
# make supplimental with indices for the temperature / walker / (temperature * nwalkers + walker maybe)
# move around these indices keeping the arrays in the same place. 

# TODO:
# add phase marginalization carefully to swap likelihoods in proposals
# update the other tools for pe
# one of these updates should probably be speeding up rj if possible (gpu priors? swaps no longer an issue)

num_subs = 8
current_subs = [None for _ in range(num_subs)]
current_subs_running_subprocess = [None for _ in range(num_subs)]
temp_files_dir = "tmp_files_dir"

for sub_i in range(num_subs):
    if f"process_{sub_i}" not in os.listdir(temp_files_dir):
        os.mkdir(temp_files_dir + "/" + f"process_{sub_i}")

    current_subs_running_subprocess[sub_i] = subprocess.Popen(["python", "simultaneous_single_mcmc_run.py", "-pi", str(sub_i), "--dir", temp_files_dir]) # , stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None)
                            #out_all.append(out)

# setup mixing subprocess
args = (snr_lim, gpus_for_equilibrate, priors, search_f_bin_lims)
with open("start_simultaneous_equilibrate_file.pickle", "wb") as fp_tmp:
    pickle.dump(args, fp_tmp, protocol=pickle.HIGHEST_PROTOCOL)

mixing_subprocess = subprocess.Popen(["python", "simultaneous_equilibrate.py"])

try:
    xp.cuda.runtime.setDevice(gpus[0])
    if fp_mix_final not in os.listdir():
        
        data_minus_templates = xp.asarray([A_inj, E_inj])[None, :, :]

        if current_start_points_file not in os.listdir():
            current_start_points = []  
            current_snrs_search = []
            current_found_coords_for_starting_mix = []
            gb.d_d = df * 4 * xp.sum((data_minus_templates.conj() * data_minus_templates) / xp.asarray(psd), axis=(1,2))

        else:
            current_start_points = np.load(current_start_points_file)
            current_start_points_in = transform_fn.both_transforms(current_start_points)
            current_snrs_search = list(np.load(current_start_points_snr_file))
            current_found_coords_for_starting_mix = list(np.load(current_found_coords_for_starting_mix_file))

            groups = xp.zeros(len(current_start_points), dtype=np.int32)

            #  -1 is to do -(-d + h) = d - h  
            data_minus_templates *= -1.
            gb.generate_global_template(current_start_points_in, groups, data_minus_templates, batch_size=1000, **waveform_kwargs)
            data_minus_templates *= -1.

            current_start_points = list(current_start_points)

        data_minus_templates = data_minus_templates.squeeze()

        nwalkers_prep = 100
        ntemps_prep = 10

        max_iter = 1000
        snr_break = 10.0
        num_bin_pause = 100

        num_binaries_needed_to_mix = 1
        num_binaries_current = 0
        max_iter = 1000
        iter_init = 0
        is_first_iter = True
        for iter_i in range(iter_init, max_iter):

            #gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])

            #print(-1/2 * gb.d_d, "ll going in")
            tmp_found_this_iter = 0
            
            start_bin_inds = np.arange(len(search_f_bin_lims) - 1)
            if not is_first_iter or not run_mix_first:
                is_first_iter = False
                for jj in range(2):
                    gb.gpus = None
                    data_temp = [data_minus_templates[0].copy(), data_minus_templates[1].copy()]

                    # odd or even
                    start_bin_inds_here = start_bin_inds[jj::2]
                    end_bin_inds_here = start_bin_inds_here + 1

                    # reverse them so it moves from right to left and gets the high f and snr sources first 
                    # (Not necessary but helpful in tracking the algorithm)
                    lower_f0 = search_f_bin_lims[start_bin_inds_here][::-1]
                    upper_f0 = search_f_bin_lims[end_bin_inds_here][::-1]
                    inds_sub_bands_here = np.arange(num_sub_bands)[start_bin_inds_here][::-1]

                    out_run = []

                    para_args = []
                    sub_bands_to_run = []
                    if current_start_points_file in os.listdir():
                        nleaves_max_from_search = np.load(current_start_points_file).shape[0]
                    else:
                        nleaves_max_from_search = 0

                    if nleaves_max_from_search > 0:
                        # make sure mixing adds these binaries before moving on in search
                        # need to do this every switch between odds and evens
                        while current_save_state_file not in os.listdir():
                            time.sleep(0.5)
                        
                        nleaves_max_from_mix = -1
                        
                        first_iter = True
                        while nleaves_max_from_mix != nleaves_max_from_search:
                            with open(current_save_state_file, "rb") as fp_temp:
                                last_state = pickle.load(fp_temp)
                            nleaves_max_from_mix = last_state.branches["gb_fixed"].nleaves[0, 0].item()
                            if nleaves_max_from_mix != nleaves_max_from_search:
                                if first_iter:
                                    check_time = time.perf_counter()
                                else:
                                    if time.perf_counter() - check_time > 15. * 60.:
                                        print("not yet", nleaves_max_from_mix, nleaves_max_from_search)
                                        check_time = time.perf_counter()
                                time.sleep(10.0)
                            first_iter = False

                    if current_residuals_file_iterative_search in os.listdir():
                        data_minus_templates_for_para = np.load(current_residuals_file_iterative_search)
                    
                    else:
                        data_minus_templates_for_para = data_minus_templates.get()
                        np.save(current_residuals_file_iterative_search, data_minus_templates.get())

                    for (sub_band_i, sub_band_lower_f0, sub_band_upper_f0) in zip(inds_sub_bands_here, lower_f0, upper_f0):

                        np.save(sub_band_fails_file, num_sub_band_fails)
                        if num_sub_band_fails[sub_band_i] >= num_sub_band_fails_limit:
                            continue

                        # TODO: get startup time down on single runs

                        # if iter_i == 0:
                        #     if jj < 1:
                        #         continue
                        #    if sub_band_i < 166 and sub_band_i != 158:
                        #        continue

                        #if iter_i == 5:
                        #    if jj < 1:
                        #        continue
                        #    elif sub_band_i > 109 or sub_band_i in [107, 99]:
                        #        continue

                        # if sub_band_i > 15:
                        #    continue
                        
                        para_args.append(
                            (sub_band_i, nwalkers_prep, ntemps_prep, sub_band_lower_f0, sub_band_upper_f0, deepcopy(generating_priors), Tobs, oversample, deepcopy(waveform_kwargs), data_minus_templates_for_para)
                        )
                        sub_bands_to_run.append(sub_band_i)

                    sub_index = 0
                    num_gpus = len(gpus)
                    # sub_index < len(sub_bands_to_run) -> load all sets
                    # np.all(np.asarray(current_subs) != None) make sure they are cleared out

                    while sub_index < len(sub_bands_to_run) or not np.all(np.asarray(current_subs) == None):
                        time.sleep(0.5)
                    
                        if None in current_subs and sub_index < len(sub_bands_to_run):
                            current_index_update = current_subs.index(None)
                            gpu_index = current_index_update % num_gpus
                            gpu_here = gpus[gpu_index]
                            new_sub = sub_bands_to_run[sub_index]

                            para_arg = para_args[sub_index]
                            # launch run
                            fp_transfer = current_iterative_search_sub_file_base + f"_search_iter_{iter_i}_band_{new_sub}_transfer.pickle"

                            fp_check = current_iterative_search_sub_file_base + f"_search_iter_{iter_i}_band_{new_sub}.pickle"
                            assert para_arg[0] == new_sub

                            if fp_check not in os.listdir(temp_files_dir + "/" + f"process_{current_index_update}/"):

                                with open(temp_files_dir + "/" + f"process_{current_index_update}/" + fp_transfer, "wb") as fp_tmp:
                                    pickle.dump((current_index_update, gpu_here) + para_arg, fp_tmp, protocol=pickle.HIGHEST_PROTOCOL)

                                # print(f"added sub: process: {current_index_update}, gpu: {gpu_here}, sub_band_i: {new_sub}")

                            current_subs[current_index_update] = new_sub

                            print(f"current_subs: {current_subs}\n")

                            sub_index += 1
                        #else:
                        #    breakpoint()
                    
                        for jjj, current_sub in enumerate(current_subs):
                            time.sleep(0.05)
                            if current_sub is None:
                                continue

                            fp_check = current_iterative_search_sub_file_base + f"_search_iter_{iter_i}_band_{current_sub}.pickle"
                            if fp_check in os.listdir(temp_files_dir + "/" + f"process_{jjj}/"):
                                time.sleep(0.5)
                                #try:
                                #    current_subs_running_subprocess[jjj].terminate()
                                #except AttributeError:
                                #    pass

                                # deal with this source and update information
                                with open(temp_files_dir + "/" + f"process_{jjj}/" + fp_check, "rb") as fp_tmp:
                                    result = pickle.load(fp_tmp)
                            
                                sub_band_i = result["sub_band_i"]
                                new_coords = result["new_coords"]
                                det_snr  = result["det_snr"]
                                opt_snr = result["opt_snr"]
                                starting_coords = result["starting_coords"]
                                starting_ll_from_sampler = result["starting_ll_from_sampler"]
                                
                                current_subs[jjj] = None

                                # print(f"removed sub: process: {jjj}, gpu: {gpu_here}, sub_band_i: {sub_band_i}")
                                # print(f"current_subs: {current_subs}")

                                if opt_snr < snr_lim:
                                    print("found source too low in optimal SNR: ", det_snr, opt_snr, f"sub band: {sub_band_i + 1} out of {len(lower_f0)}")
                                    os.remove(temp_files_dir + "/" + f"process_{jjj}/" + fp_check)
                                    num_sub_band_fails[sub_band_i] += 1
                                    np.save(sub_band_fails_file, num_sub_band_fails)
                                    continue

                                current_found_coords_for_starting_mix.append(starting_coords)
                                current_start_points.append(new_coords)
                                current_snrs_search.append([det_snr, opt_snr])

                                print("found source: ", det_snr, opt_snr, f"sub band: {sub_band_i + 1} out of {len(lower_f0)}")

                                num_binaries_current += 1

                                groups = xp.zeros(1, dtype=np.int32)
                                #  -1 is to do -(-d + h) = d - h  
                                waveform_kwargs_sub = waveform_kwargs.copy()
                                data_minus_templates *= -1.

                                new_coords_in = transform_fn.both_transforms(new_coords[None, :])

                                gb.generate_global_template(new_coords_in, groups, data_minus_templates[None, :, :], batch_size=1000, **waveform_kwargs_sub)
                                data_minus_templates *= -1.

                                np.save(current_start_points_snr_file, np.asarray(current_snrs_search))
                                np.save(current_start_points_file, np.asarray(current_start_points))
                                np.save(current_found_coords_for_starting_mix_file, np.asarray(current_found_coords_for_starting_mix))
                                os.remove(temp_files_dir + "/" + f"process_{jjj}/" + fp_check) 
                                
                                ll_check = -1/2 * df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])

                                gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])                

                if num_binaries_current == 0:
                    print("END")
                    breakpoint()
            
                # remove the old mixing file if we are adding binaries
                if fp_mix in os.listdir():
                    os.remove(fp_mix)
            num_binaries_current = 0
    for current_sub in current_subs_running_subprocess:
        current_sub.terminate()
    mixing_subprocess.terminate()

    copyfile(fp_mix, fp_mix_final)

except KeyboardInterrupt:
    for current_sub in current_subs_running_subprocess:
        current_sub.terminate()
    mixing_subprocess.terminate()

    