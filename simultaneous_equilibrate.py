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

def run_equilibrate(snr_lim, gpus_for_equilibrate, priors, search_f_bin_lims):
    gpus = gpus_for_equilibrate

    # from lisatools.sampling.stopping import SearchConvergeStopping2

    waveform_kwargs["start_freq_ind"] = start_freq_ind

    # for testing
    # search_f_bin_lims = np.arange(f0_lims_in[0], f0_lims_in[1], 2 * 128 * df)

    num_sub_bands = len(search_f_bin_lims)

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
        
        if "save_state_temp.pickle" in os.listdir():
            with open("save_state_temp.pickle", "rb") as fp_out:
                last_sample = pickle.load(fp_out)
            nleaves_max_fix = last_sample.branches["gb_fixed"].nleaves.max().item()
            old_number = nleaves_max_fix

        else:
            old_number = 0
        number_needed = 20
        current_iteration = 0
        while True:
            time.sleep(0.1)
            while current_found_coords_for_starting_mix_file not in os.listdir() or np.load(current_found_coords_for_starting_mix_file).shape[0] < number_needed:
                time.sleep(0.25)

            # to prevent ValueError on read
            max_tries_ValueError = 100
            current_tries_ValueError = 0
            working = False
            while not working:
                try:
                    current_found_coords_for_starting_mix = np.load(current_found_coords_for_starting_mix_file)
                    working = True
                except ValueError as e:
                    current_tries_ValueError += 1
                    if current_tries_ValueError >= max_tries_ValueError:
                        raise ValueError("Tried 100 times and did not get sucess:\n" + e.__str__())
                    time.sleep(0.5)

            starting_new = False
            if current_iteration == 0 or current_found_coords_for_starting_mix.shape[0] > old_number + number_needed:
                
                nleaves_max_fix = len(current_found_coords_for_starting_mix)
                coords_out = np.zeros((ntemps, nwalkers, nleaves_max_fix, ndim))

                A_going_in = np.zeros((ntemps, nwalkers, A_inj.shape[0]), dtype=complex)
                E_going_in = np.zeros((ntemps, nwalkers, E_inj.shape[0]), dtype=complex)
                A_going_in[:] = np.asarray(A_inj)
                E_going_in[:] = np.asarray(E_inj)

                A_psd_in = np.zeros((ntemps, nwalkers, A_inj.shape[0]), dtype=np.float64)
                E_psd_in = np.zeros((ntemps, nwalkers, E_inj.shape[0]), dtype=np.float64)
                A_psd_in[:] = np.asarray(psd)
                E_psd_in[:] = np.asarray(psd)

                try:
                    del mgh
                    del sampler_mix
                    del supps
                    del state_mix
                    main_gpu = xp.cuda.runtime.getDevice()
                    mempool.free_all_blocks()
                    for gpu in gpus:
                        with xp.cuda.device.Device(gpu):
                            xp.cuda.runtime.deviceSynchronize()
                            mempool_tmp = xp.get_default_memory_pool()
                            mempool_tmp.free_all_blocks()
                            xp.cuda.runtime.deviceSynchronize()
                    
                    # back to main GPU
                    xp.cuda.runtime.setDevice(main_gpu)
                    xp.cuda.runtime.deviceSynchronize()
                    
                except NameError:
                    pass
            
                mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df,
                    base_injections=[A_inj, E_inj], base_psd=[psd.copy(), psd.copy()]
                )

                gb.d_d = xp.asarray(mgh.get_inner_product(use_cpu=True).flatten())

                mempool.free_all_blocks()
                
                # setup data streams to add to and subtract from
                supps_shape_in = (ntemps, nwalkers)

                gb.gpus = None
                
                if old_number > 0:
                    with open("save_state_temp.pickle", "rb") as fp_out:
                        last_sample = pickle.load(fp_out)
                    old_coords = last_sample.branches["gb_fixed"].coords
                
                    ntemps_old = old_coords.shape[0]
                    if ntemps_old < ntemps:
                        ntemps_added = ntemps - ntemps_old
                        if ntemps_added > ntemps_old:
                            raise NotImplementedError

                        old_coords_before = old_coords.copy()
                        old_coords = np.zeros((ntemps,) + old_coords_before.shape[1:],)
                        old_coords[:ntemps_old] = old_coords_before[:]
                        old_coords[ntemps_old:] = old_coords[:ntemps_added]

                    elif ntemps_old > ntemps:
                        old_coords = old_coords[:ntemps].copy()
                        
                    old_coords_start = old_coords.shape[2]

                    # move cold chain to highest temperature
                    old_coords[-1] = old_coords[0].copy()

                else:
                    old_coords = None
                    old_coords_start = 0

                if current_found_coords_for_starting_mix_file in os.listdir():
                    coords_from_search_runs = np.load(current_found_coords_for_starting_mix_file)

                    # skip ones already added and limit to only number of walkers

                    # THE SHUFFLE COULD BE VERY IMPORTANT BECAUSE THE LIKELIHOODS
                    # ARE ORDERED
                    new_coords = shuffle_along_axis(
                        np.repeat(coords_from_search_runs[old_coords_start:, :nwalkers].transpose(1, 0, 2)[None, :, :, :], ntemps, axis=0),
                        1
                    )

                    if old_coords is not None:
                        coords_out = np.concatenate([old_coords, new_coords], axis=2)

                    else:
                        coords_out = new_coords

                    nleaves_max_fix = coords_out.shape[2]

                else:
                    raise ValueError(f"Needs current_found_coords_for_starting_mix_file in cwd.")

                coords_out[:, :, :, 3] = coords_out[:, :, :, 3] % (2 * np.pi)
                coords_out[:, :, :, 5] = coords_out[:, :, :, 5] % (1 * np.pi)
                coords_out[:, :, :, 6] = coords_out[:, :, :, 6] % (2 * np.pi)
                # mgh = MultiGPUDataHolder(gpus, data_minus_templates_mix[0].get().reshape(ntemps, nwalkers, -1), data_minus_templates_mix[1].get().reshape(ntemps, nwalkers, -1), A_psd_in, E_psd_in, df)
                
                # mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df)

                gb.gpus = mgh.gpus

                coords_in_in = transform_fn.both_transforms(coords_out.reshape(-1, coords_out.shape[-1]))

                data_index = xp.asarray(np.repeat(np.arange(ntemps * nwalkers)[:, None], nleaves_max_fix, axis=-1).flatten()).astype(xp.int32)

                mgh.multiply_data(-1.)
                gb.generate_global_template(coords_in_in, data_index, mgh.data_list, batch_size=1000, data_length=data_length, data_splits=mgh.gpu_splits, **waveform_kwargs)
                mgh.multiply_data(-1.)
                
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

                points_start = state_mix.branches["gb_fixed"].coords[state_mix.branches["gb_fixed"].inds]
                N_vals = np.zeros((ntemps, nwalkers, nleaves_max_fix), dtype=int)
                points_start_transform = transform_fn.both_transforms(points_start)
                amp_start = points_start_transform[:, 0]
                f0_start = points_start_transform[:, 1]
                from gbgpu.utils.utility import get_N
                N_temp = get_N(amp_start, f0_start, waveform_kwargs["T"], waveform_kwargs["oversample"])

                N_vals[state_mix.branches["gb_fixed"].inds] = N_temp
                branch_supp_base_shape = (ntemps, nwalkers, nleaves_max_fix)
                state_mix.branches["gb_fixed"].branch_supplimental = BranchSupplimental({"N_vals": N_vals}, obj_contained_shape=branch_supp_base_shape, copy=True)
                
                gb_kwargs = dict(
                    waveform_kwargs=waveform_kwargs,
                    parameter_transforms=transform_fn,
                    search=True,
                    search_samples=None,  # out_params,
                    search_snrs=None,  # out_snr,
                    search_snr_lim=snr_break,  # 50.0,
                    psd_func=get_sensitivity,
                    noise_kwargs=dict(sens_fn="noisepsd_AE", model="sangria"),
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
                    search_f_bin_lims,
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

                # stop_converge_mix = SearchConvergeStopping2(n_iters=7, diff=0.5, verbose=True, iter_back_check=7)

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
                        backend=None,
                        vectorize=True,
                        periodic=periodic,  # TODO: add periodic to proposals
                        branch_names=["gb_fixed"],
                        stopping_fn=None,  # stop_converge_mix,
                        stopping_iterations=-1,
                        provide_groups=True,
                        provide_supplimental=True,
                    )

                # equlibrating likelihood check: -4293090.6483655665,
                
                out = state_mix
                old_number = nleaves_max_fix
                starting_new = True
                print("starting new", nleaves_max_fix)

            if ((current_iteration + 1) % 40 == 0 or starting_new) and current_iteration != 0:
                out.branches["gb_fixed"].coords[-1] = out.branches["gb_fixed"].coords[0]
                out.branches["gb_fixed"].inds[-1] = out.branches["gb_fixed"].inds[0]
                out.branches["gb_fixed"].branch_supplimental[-1] = out.branches["gb_fixed"].branch_supplimental[0]
                # state_mix.supplimental[-1] = state_mix.supplimental[0]
                out.log_prob[-1] = out.log_prob[0]
                try:
                    out.log_prior[-1] = out.log_prior[0]
                    out.betas[-1] = out.betas[0]
                except TypeError:
                    pass

                inds_cold = out.supplimental.holder["overall_inds"][0]
                inds_hot = out.supplimental.holder["overall_inds"][-1]
                main_gpu = xp.cuda.runtime.getDevice()
                for gpu_i, split_now in enumerate(mgh.gpu_splits):
                    gpu = mgh.gpus[gpu_i]
                    with xp.cuda.device.Device(gpu):
                        xp.cuda.runtime.deviceSynchronize()
                        for overall_ind_hot, overall_ind_cold in zip(inds_hot, inds_cold):
                            if overall_ind_hot in split_now:
                                ind_keep_hot = np.where(split_now == overall_ind_hot)[0].item()
                                for gpu_i_cold, split_cold in enumerate(mgh.gpu_splits):
                                    if overall_ind_cold in split_cold:
                                        ind_keep_cold = np.where(split_cold == overall_ind_cold)[0].item()
                                        for jjj in range(len(mgh.data_list)):
                                            sliceit_hot = slice(ind_keep_hot * data_length, (ind_keep_hot + 1) * data_length)
                                            sliceit_cold = slice(ind_keep_cold * data_length, (ind_keep_cold + 1) * data_length)
                                            mgh.data_list[jjj][gpu_i][sliceit_hot] = mgh.data_list[jjj][gpu_i_cold][sliceit_cold]

                        xp.cuda.runtime.deviceSynchronize()
                xp.cuda.runtime.setDevice(main_gpu)

            nsteps_mix = 200

            print("Starting mix ll best:", out.log_prob.max(axis=-1))
            mempool.free_all_blocks()

            out = sampler_mix.run_mcmc(out, nsteps_mix, progress=True, thin_by=1, save_first_state=False, store=False)

            print("ending mix ll best:", out.log_prob.max(axis=-1))

            save_state = State(out.branches_coords, inds=out.branches_inds, log_prob=out.log_prob, log_prior=out.log_prior)

            if "save_state_temp.pickle" in os.listdir():
                shutil.copyfile("save_state_temp.pickle", "old_save_state_temp.pickle")

            with open("save_state_temp.pickle", "wb") as fp_out:
                pickle.dump(save_state, fp_out, protocol=pickle.HIGHEST_PROTOCOL)

            # needs to be max from mgh or need to map it
            lp_mgh = mgh.get_ll(use_cpu=True)
            max_ind = np.where(lp_mgh == lp_mgh.max())

            max_temp = max_ind[0][0]
            max_walker = max_ind[1][0]
            overall_ind_best = max_temp * nwalkers + max_walker
            
            data_minus_templates = []
            for gpu_i, split_now in enumerate(mgh.gpu_splits):
                if overall_ind_best in split_now:
                    ind_keep = np.where(split_now == overall_ind_best)[0].item()
                    for tmp_data in mgh.data_list:
                        with xp.cuda.device.Device(gpus[0]):
                            data_minus_templates.append(tmp_data[gpu_i].reshape(len(split_now), -1)[ind_keep].copy())
            
            xp.cuda.runtime.setDevice(gpus[0])

            data_minus_templates = xp.asarray(data_minus_templates)

            np.save(current_residuals_file_iterative_search, data_minus_templates.get())
            
            """# current_start_points = list(out.branches["gb_fixed"].coords[max_ind][out.branches["gb_fixed"].inds[max_ind]])

            # current_found_coords_for_starting_mix = list(current_found_coords_for_starting_mix)
            
            # current_start_points_in = transform_fn.both_transforms(np.asarray(current_start_points))

            gb.d_d = mgh.get_inner_product(use_cpu=True)[real_temp_best, real_walker_best].item()

            _ = gb.get_ll(current_start_points_in, mgh.data_list, mgh.psd_list, data_length=mgh.data_length, data_splits=mgh.gpu_splits, **waveform_kwargs)

            
            # TODO: should check if phase marginalize
            det_snr = (np.abs(gb.d_h) / np.sqrt(gb.h_h.real)).get()
            opt_snr =  (np.sqrt(gb.h_h.real)).get()

            phase_angle = np.angle(gb.d_h)

            try:
                phase_angle = phase_angle.get()
            except AttributeError:
                pass

            tmp_current_start_points = np.asarray(current_start_points)

            tmp_current_start_points[:, 3] -= phase_angle
            current_start_points_in[:, 4] -= phase_angle

            current_start_points = list(tmp_current_start_points)

            current_snrs_search = list(np.array([det_snr, opt_snr]).T)
            # np.save(current_start_points_snr_file, np.asarray(current_snrs_search))
            # np.save(current_start_points_file, np.asarray(current_start_points))"""


            num_binaries_current = 0
            
            print("DONE MIXING","num sources:", nleaves_max_fix)
            #if det_snr_finding < snr_break or len(current_snrs_search) > num_bin_pause:
            #    break
            current_iteration += 1

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    #parser.add_argument('--gpu', type=int,
    #                    help='an integer identifier', required=True)

    #parser.add_argument('--start-file', '-sf', type=int,
    #                    help='an integer identifier', required=True)

    #args = parser.parse_args()
    
    fp_in = "start_simultaneous_equilibrate_file.pickle"
    with open(fp_in, "rb") as fp_tmp:
        info_in = pickle.load(fp_tmp)
            
    output = run_equilibrate(*info_in)
                