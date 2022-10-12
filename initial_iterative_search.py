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

# TODO: fix initial setup for mix where it backs up the likelihood

class PointGeneratorSNR:
    def __init__(self, generate_snr_ladder):
        self.generate_snr_ladder = generate_snr_ladder

    def __call__(self, size=(1,)):
        new_points = self.generate_snr_ladder.rvs(size=size)
        logpdf = self.generate_snr_ladder.logpdf(new_points)
        return (new_points, logpdf)

point_generator = PointGeneratorSNR(generate_snr_ladder)

snr_lim = 5.0
gpus = [4, 5, 6]

from lisatools.sampling.stopping import SearchConvergeStopping2

generating_priors = deepcopy(priors)

waveform_kwargs["start_freq_ind"] = start_freq_ind



dbin = 2 * N
"""f_width = f0_lims[1] - f0_lims[0]
num_f_bins = int(np.ceil(f_width / df))
num_sub_bands = num_f_bins // dbin + int((num_f_bins % dbin) != 0)
lim_inds = np.concatenate([np.array([0]), np.arange(dbin, num_f_bins, dbin), np.array([num_f_bins - 1])])
"""
#search_f_bin_lims = f0_lims[0] + df * np.arange(num_f_bins)[lim_inds]
#search_f_bin_lims[-1] = f0_lims[-1]

# adjust for frequencies
f0_lims_in = f0_lims.copy()

# TODO: make wider because this is knowning the limits?
f0_lims_in[0] = 0.3e-3
#f0_lims_in[1] = 0.8e-3
low_fs = np.append(np.arange(f0_lims_in[0], 0.001, 2 * 128 * df), np.array([0.001]))
mid_fs = np.arange(0.001 + 256 * 2 * df, 0.01, 2 * 256 * df)
high_fs = np.append(np.arange(0.01, f0_lims_in[-1], 2 * 1024 * df), np.array([f0_lims_in[-1]]))
search_f_bin_lims = np.concatenate([low_fs, mid_fs, high_fs])

# for testing
# search_f_bin_lims = np.arange(f0_lims_in[0], f0_lims_in[1], 2 * 128 * df)

num_sub_bands = len(search_f_bin_lims)

if sub_band_fails_file not in os.listdir():
    num_sub_band_fails = np.zeros(num_sub_bands, dtype=int)

else:
    num_sub_band_fails = np.load(sub_band_fails_file)

assert num_sub_band_fails.shape[0] == num_sub_bands
num_sub_band_fails_limit = 2

run_mix_first = True
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
    snr_break = 5.0
    num_bin_pause = 100

    num_binaries_needed_to_mix = 1
    num_binaries_current = 0
    max_iter = 1000

    for iter_i in range(max_iter):

        #gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])

        #print(-1/2 * gb.d_d, "ll going in")
        tmp_found_this_iter = 0
        
        start_bin_inds = np.arange(len(search_f_bin_lims) - 1)
        if (iter_i > 0 and run_mix_first) or not run_mix_first:
            
            for jj in range(2):
                gb.gpus = None
                data_temp = [data_minus_templates[0].copy(), data_minus_templates[1].copy()]

                # odd or even
                start_bin_inds_here = start_bin_inds[jj::2]
                end_bin_inds_here = start_bin_inds_here + 1
                lower_f0 = search_f_bin_lims[start_bin_inds_here]
                upper_f0 = search_f_bin_lims[end_bin_inds_here]
                inds_sub_bands_here = np.arange(num_sub_bands)[start_bin_inds_here]
                inds_sub_bands_here = inds_sub_bands_here
                lower_f0 = lower_f0
                upper_f0 = upper_f0
                out_run = []

                para_args = []
                sub_bands_to_run = []

                data_minus_templates_for_para = data_minus_templates.get()
                np.save(current_residuals_file_iterative_search, data_minus_templates.get())
                for (sub_band_i, sub_band_lower_f0, sub_band_upper_f0) in zip(inds_sub_bands_here, lower_f0, upper_f0):

                    np.save(sub_band_fails_file, num_sub_band_fails)
                    if num_sub_band_fails[sub_band_i] >= num_sub_band_fails_limit:
                        continue

                    # TODO: get startup time down on single runs

                    """if iter_i == 0:
                        if jj < 1:
                            #continue

                            if sub_band_i < 490:
                                continue
                            if sub_band_i in [494, 496, 498, 510]:
                                continue"""

                    # if sub_band_i > 15:
                    #    continue
                    
                    para_args.append(
                        (sub_band_i, nwalkers_prep, ntemps_prep, sub_band_lower_f0, sub_band_upper_f0, deepcopy(generating_priors), Tobs, oversample, deepcopy(waveform_kwargs), data_minus_templates_for_para)
                    )
                    sub_bands_to_run.append(sub_band_i)

                num_subs = 12
                sub_index = 0
                current_subs = [None for _ in range(num_subs)]
                current_subs_running_subprocess = [None for _ in range(num_subs)]
                

                num_gpus = len(gpus)
                temp_files_dir = "tmp_files_dir"
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

                        if fp_check not in os.listdir(temp_files_dir + "/"):

                            with open(temp_files_dir + '/' + fp_transfer, "wb") as fp_tmp:
                                pickle.dump((current_index_update, gpu_here) + para_arg, fp_tmp, protocol=pickle.HIGHEST_PROTOCOL)

                            current_subs_running_subprocess[current_index_update] = subprocess.Popen(["python", "single_mcmc_run.py", "-si", str(iter_i), "-bi", str(new_sub), "--dir", temp_files_dir]) # , stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None)
                            #out_all.append(out)
                            print(f"added sub: process: {current_index_update}, gpu: {gpu_here}, sub_band_i: {new_sub}")

                        current_subs[current_index_update] = new_sub

                        print(f"current_subs: {current_subs}\n")

                        sub_index += 1
                    #else:
                    #    breakpoint()
                
                    for jjj, current_sub in enumerate(current_subs):
                        if current_sub is None:
                            continue
                        fp_check = current_iterative_search_sub_file_base + f"_search_iter_{iter_i}_band_{current_sub}.pickle"
                        
                        if fp_check in os.listdir(temp_files_dir + "/"):
                            time.sleep(0.5)
                            try:
                                current_subs_running_subprocess[jjj].terminate()
                            except AttributeError:
                                pass

                            # deal with this source and update information
                            with open(temp_files_dir + '/' + fp_check, "rb") as fp_tmp:
                                result = pickle.load(fp_tmp)
                           
                            sub_band_i = result["sub_band_i"]
                            new_coords = result["new_coords"]
                            det_snr  = result["det_snr"]
                            opt_snr = result["opt_snr"]
                            starting_coords = result["starting_coords"]
                            starting_ll_from_sampler = result["starting_ll_from_sampler"]
                            
                            current_subs[jjj] = None

                            print(f"removed sub: process: {jjj}, gpu: {gpu_here}, sub_band_i: {sub_band_i}\ncurrent_subs: {current_subs}")

                            if opt_snr < snr_lim:
                                print("found source too low in optimal SNR: ", det_snr, opt_snr, f"sub band: {sub_band_i + 1} out of {len(lower_f0)}")
                                os.remove(temp_files_dir + "/" + fp_check)
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
                            np.save(current_residuals_file_iterative_search, data_minus_templates.get())
                            np.save(current_found_coords_for_starting_mix_file, np.asarray(current_found_coords_for_starting_mix))
                            os.remove(temp_files_dir + "/" + fp_check) 
                            
                    
                            ll_check = -1/2 * df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])

                            gb.d_d = df * 4 * xp.sum(data_minus_templates.conj() * data_minus_templates / xp.asarray(psd)[None, :])
                
            
            if num_binaries_current == 0:
                print("END")
                breakpoint()
        
            # remove the old mixing file if we are adding binaries
            if fp_mix in os.listdir():
                os.remove(fp_mix)
        
        np.save(current_start_points_snr_file, np.asarray(current_snrs_search))
        np.save(current_start_points_file, np.asarray(current_start_points))
        np.save(current_found_coords_for_starting_mix_file, np.asarray(current_found_coords_for_starting_mix))
        current_start_points = np.load(current_start_points_file)
        current_found_coords_for_starting_mix = np.load(current_found_coords_for_starting_mix_file)

        nleaves_max_fix = len(current_start_points)
        coords_out = np.zeros((ntemps, nwalkers, nleaves_max_fix, ndim))

        A_going_in = np.zeros((ntemps, nwalkers, A_inj.shape[0]), dtype=complex)
        E_going_in = np.zeros((ntemps, nwalkers, E_inj.shape[0]), dtype=complex)
        A_going_in[:] = np.asarray(A_inj)
        E_going_in[:] = np.asarray(E_inj)

        A_psd_in = np.zeros((ntemps, nwalkers, A_inj.shape[0]), dtype=np.float64)
        E_psd_in = np.zeros((ntemps, nwalkers, E_inj.shape[0]), dtype=np.float64)
        A_psd_in[:] = np.asarray(psd)
        E_psd_in[:] = np.asarray(psd)

        data_minus_templates_mix = [xp.asarray(A_going_in.flatten()), xp.asarray(E_going_in.flatten())]
        psd_mix = [xp.asarray(A_psd_in.flatten()), xp.asarray(E_psd_in.flatten())]
        
        tmp1 = xp.asarray([dmtm.reshape(ntemps * nwalkers, -1).conj() * dmtm.reshape(ntemps * nwalkers, -1) / psdm.reshape(ntemps * nwalkers, -1) for dmtm, psdm in zip(data_minus_templates_mix, psd_mix)])
        gb.d_d = 4 * df * xp.sum(tmp1, axis=(0, 2))

        mempool.free_all_blocks()
        
        # setup data streams to add to and subtract from
        supps_shape_in = (ntemps, nwalkers)

        gb.gpus = None
        reader_mix = HDFBackend(fp_mix)
        if fp_mix in os.listdir():
            mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df, base_injections=[A_inj, E_inj], base_psd=[psd, psd])
            gb.gpus = mgh.gpus
            state_mix = get_last_gb_state(mgh, reader_mix, df, supps_base_shape)

            gb.injection_d_d = mgh.get_injection_inner_product().item()

        else:
            """
            for j, params in enumerate(current_start_points):
                
                factor = 1e-1
                cov = np.ones(8) * 1e-3
                cov[1] = 1e-7

                start_like = np.zeros((nwalkers * ntemps))
                data_index_in = (np.tile(np.arange(nwalkers), (ntemps, 1)) + nwalkers * np.repeat(np.arange(ntemps), nwalkers).reshape(ntemps, nwalkers)).flatten()

                data_index = xp.asarray(data_index_in).astype(xp.int32)  # mgh.get_mapped_indices(data_index_in)).astype(xp.int32)
                
                iter_check = 0
                run_flag = True
                while np.std(start_like) < 1.0 and run_flag:
                    
                    logp = np.full_like(start_like, -np.inf)
                    tmp = np.zeros((ntemps * nwalkers, ndim))
                    fix = np.ones((ntemps * nwalkers), dtype=bool)
                    while np.any(fix):
                        tmp[fix] = (params[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, 8)))[fix]

                        tmp[:, 3] = tmp[:, 3] % (2 * np.pi)
                        tmp[:, 5] = tmp[:, 5] % (np.pi)
                        tmp[:, 6] = tmp[:, 6] % (2 * np.pi)
                        logp = priors["gb_fixed"].logpdf(tmp)

                        fix = np.isinf(logp)
                        if np.all(fix):
                            breakpoint()

                    tmp_in = transform_fn.both_transforms(tmp, return_transpose=True)

                    start_like = gb.get_ll(
                        tmp_in.T,
                        data_minus_templates_mix,
                        psd_mix,
                        phase_marginalize=True,
                        data_index=data_index,
                        noise_index=data_index,
                        data_length=data_length,
                        # data_splits=mgh.gpu_splits,
                        **waveform_kwargs,
                    )  # - np.sum([np.log(psd), np.log(psd)])

                    phase_change = np.angle(gb.non_marg_d_h)
                    tmp[:, 3] -= phase_change
                    tmp[:, 3] = tmp[:, 3] % (2 * np.pi)

                    tmp_in[4] = tmp[:, 3]

                    iter_check += 1
                
                    if j > 0:
                        start_like -= -1/2 * gb.d_d.get().real
                    #print(np.std(start_like))
                    factor *= 1.5

                    if factor > 10.0:
                        run_flag = False

                #  -1 is to do -(-d + h) = d - h
                for dat in data_minus_templates_mix:
                    dat *= -1
                # data_minus_templates_mix *= -1.
                #mgh.multiply_data(-1.)
                gb.generate_global_template(tmp_in.T, data_index, data_minus_templates_mix, batch_size=1000, data_length=data_length, **waveform_kwargs)
                #mgh.multiply_data(-1.)
                for dat in data_minus_templates_mix:
                    dat *= -1

                tmp1 = xp.asarray([dmtm.reshape(ntemps * nwalkers, -1).conj() * dmtm.reshape(ntemps * nwalkers, -1) / psdm.reshape(ntemps * nwalkers, -1) for dmtm, psdm in zip(data_minus_templates_mix, psd_mix)])
                gb.d_d = 4 * df * xp.sum(tmp1, axis=(0, 2))
                
                coords_out[:, :, j] = tmp.reshape(ntemps, nwalkers, 8)
                if (j + 1) % 100 == 0:
                    print((j + 1), len(current_start_points))
            """

            if fp_mix[:-3] + "_old.h5" in os.listdir():
                reader_tmp_old = HDFBackend(fp_mix[:-3] + "_old.h5")
                last_sample = reader_tmp_old.get_last_sample()
                old_coords = last_sample.branches["gb_fixed"].coords

                old_coords_start = old_coords.shape[2]

            else:
                old_coords = None
                old_coords_start = 0

            if current_found_coords_for_starting_mix_file in os.listdir():
                coords_from_search_runs = np.load(current_found_coords_for_starting_mix_file)

                # skip ones already added and limit to only number of walkers
                new_coords = coords_from_search_runs[old_coords_start:, :nwalkers].transpose(1, 0, 2)[None, :, :, :]

                if old_coords is not None:
                    coords_out = np.concatenate([old_coords, new_coords], axis=2)

                else:
                    coords_out = new_coords

            else:
                raise ValueError(f"Needs current_found_coords_for_starting_mix_file in cwd.")

            coords_out[:, :, :, 3] = coords_out[:, :, :, 3] % (2 * np.pi)
            coords_out[:, :, :, 5] = coords_out[:, :, :, 5] % (1 * np.pi)
            coords_out[:, :, :, 6] = coords_out[:, :, :, 6] % (2 * np.pi)
            # mgh = MultiGPUDataHolder(gpus, data_minus_templates_mix[0].get().reshape(ntemps, nwalkers, -1), data_minus_templates_mix[1].get().reshape(ntemps, nwalkers, -1), A_psd_in, E_psd_in, df)
            
            mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df)

            gb.gpus = mgh.gpus

            coords_in_in = transform_fn.both_transforms(coords_out.reshape(-1, coords_out.shape[-1]))

            data_index = xp.asarray(np.repeat(np.arange(ntemps * nwalkers)[:, None], nleaves_max_fix, axis=-1).flatten()).astype(xp.int32)

            mgh.multiply_data(-1.)
            gb.generate_global_template(coords_in_in, data_index, mgh.data_list, batch_size=1000, data_length=data_length, data_splits=mgh.gpu_splits, **waveform_kwargs)
            mgh.multiply_data(-1.)
            
            del data_minus_templates_mix
            del psd_mix
            del data_index
            mempool.free_all_blocks()
            
            ll = mgh.get_ll(use_cpu=True)

            temp_inds = mgh.temp_indices.copy()
            walker_inds = mgh.walker_indices.copy()
            overall_inds = mgh.overall_indices.copy()
            
            supps = BranchSupplimental({ "temp_inds": temp_inds, "walker_inds": walker_inds, "overall_inds": overall_inds,}, obj_contained_shape=supps_base_shape, copy=True)

            state_mix = State({"gb_fixed": coords_out}, log_prob=ll.reshape(ntemps, nwalkers), supplimental=supps)
            check = priors["gb_fixed"].logpdf(coords_out.reshape(-1, 8))

            if np.any(np.isinf(check)):
                breakpoint()

        gb_kwargs = dict(
            waveform_kwargs=waveform_kwargs,
            parameter_transforms=transform_fn,
            search=True,
            search_samples=None,  # out_params,
            search_snrs=None,  # out_snr,
            search_snr_lim=snr_lim,  # 50.0,
            psd_func=get_sensitivity,
            noise_kwargs=dict(sens_fn="noisepsd_AE"),
            provide_betas=True,
            point_generator_func=point_generator,
            batch_size=-1,
            skip_supp_names=["group_move_points"],
            random_seed=10,
        )

        gb_args = (
            gb,
            priors,
            int(1e3),
            start_freq_ind,
            data_length,
            mgh,
            np.asarray(fd),
            # [nleaves_max, 1],
            # [min_k, 1],
        )

        try:
            del moves_mix
            del like_mix
            del sampler_mix
            del stop_converge_mix

        except NameError:
            pass

        moves_mix = GBSpecialStretchMove(
            *gb_args,
            **gb_kwargs,
        )

        stop_converge_mix = SearchConvergeStopping2(n_iters=7, diff=0.5, verbose=True, iter_back_check=7)

        moves_mix.gb.gpus = gpus

        like_mix = BasicResidualMGHLikelihood(mgh)

        sampler_mix = EnsembleSampler(
                nwalkers,
                [ndim],  # assumes ndim_max
                like_mix,
                priors,
                tempering_kwargs={"ntemps": ntemps},
                nbranches=1,
                nleaves_max=[nleaves_max_fix],
                moves=moves_mix,
                kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
                backend=reader_mix,
                vectorize=True,
                periodic=periodic,  # TODO: add periodic to proposals
                branch_names=["gb_fixed"],
                stopping_fn=stop_converge_mix,
                stopping_iterations=1,
                provide_groups=True,
                provide_supplimental=True,
            )

        # equlibrating likelihood check: -4293090.6483655665,
        nsteps_mix = 1000
        print("Starting mix ll best:", state_mix.log_prob.max())
        mempool.free_all_blocks()

        out = sampler_mix.run_mcmc(state_mix, nsteps_mix, progress=True, thin_by=20, save_first_state=True)
        max_ind = np.where(out.log_prob == out.log_prob.max())

        temp_inds = out.supplimental[:]["temp_inds"]
        walker_inds = out.supplimental[:]["walker_inds"]

        real_temp_best = temp_inds[max_ind]
        real_walker_best = walker_inds[max_ind]
        overall_ind_best = (real_temp_best * nwalkers + real_walker_best).item()

        data_minus_templates = []
        for gpu_i, split_now in enumerate(mgh.gpu_splits):
            if overall_ind_best in split_now:
                ind_keep = np.where(split_now == overall_ind_best)[0].item()
                for tmp_data in mgh.data_list:
                    with xp.cuda.device.Device(gpus[0]):
                        data_minus_templates.append(tmp_data[gpu_i].reshape(len(split_now), -1)[ind_keep].copy())
        
        xp.cuda.runtime.setDevice(gpus[0])

        data_minus_templates = xp.concatenate(xp.asarray([data_minus_templates]))

        current_start_points = list(out.branches["gb_fixed"].coords[max_ind][out.branches["gb_fixed"].inds[max_ind]])

        current_found_coords_for_starting_mix = list(current_found_coords_for_starting_mix)
        
        current_start_points_in = transform_fn.both_transforms(np.asarray(current_start_points))

        gb.d_d = mgh.get_inner_product(use_cpu=True)[real_temp_best, real_walker_best].item()

        _ = gb.get_ll(current_start_points_in, mgh.data_list, mgh.psd_list, data_length=mgh.data_length, data_splits=mgh.gpu_splits, **waveform_kwargs)

        det_snr = (gb.d_h.real / np.sqrt(gb.h_h.real)).get()
        opt_snr =  (np.sqrt(gb.h_h.real)).get()

        current_snrs_search = list(np.array([det_snr, opt_snr]).T)
        np.save(current_start_points_snr_file, np.asarray(current_snrs_search))
        np.save(current_start_points_file, np.asarray(current_start_points))
        num_binaries_current = 0

        if fp_mix in os.listdir():
            shutil.copy(fp_mix, fp_mix[:-3] + "_old.h5")
            os.remove(fp_mix)

        del mgh
        mempool.free_all_blocks()
        
        print("DONE MIXING","num sources:", len(current_start_points), "opt snrs:", np.sort(opt_snr))
        #if det_snr_finding < snr_break or len(current_snrs_search) > num_bin_pause:
        #    break

copyfile(fp_mix, fp_mix_final)
    
