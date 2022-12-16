from multiprocessing.sharedctypes import Value
import cupy as xp
import time
import pickle
import shutil

# from lisatools.sampling.moves.gbspecialgroupstretch import GBSpecialGroupStretchMove

mempool = xp.get_default_memory_pool()

from full_band_global_fit_settings import *
from single_mcmc_run import run_single_band_search
from lisatools.utils.multigpudataholder import MultiGPUDataHolder
from eryn.moves import CombineMove

import subprocess

import warnings
warnings.filterwarnings("ignore")

stop_here = True


from eryn.moves import Move
class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros(state.log_like.shape)
        self.temperature_control.swaps_accepted = np.zeros(self.temperature_control.ntemps - 1)
        return state, accepted

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

def run_equilibrate():
    gpus_pe = [7]
    gpus = gpus_pe
    # from lisatools.sampling.stopping import SearchConvergeStopping2

    waveform_kwargs["start_freq_ind"] = start_freq_ind

    # for testing
    # search_f_bin_lims = np.arange(f0_lims_in[0], f0_lims_in[1], 2 * 128 * df)

    num_sub_bands = len(search_f_bin_lims)

    xp.cuda.runtime.setDevice(gpus[0])

    nwalkers_pe = 100
    ntemps_pe = 6

    num_binaries_needed_to_mix = 1
    num_binaries_current = 0
    
    if fp_pe in os.listdir():
        reader = HDFBackend(fp_pe)
        last_sample = reader.get_last_sample()
        nleaves_max_fix = last_sample.branches["gb_fixed"].coords.shape[-2]  # will all be the same

        nleaves_max_fix_new = nleaves_max_fix
        
    elif current_save_state_file in os.listdir():
        with open(current_save_state_file, "rb") as fp_out:
            last_sample = pickle.load(fp_out)

        coords = last_sample.branches_coords
        inds = last_sample.branches_inds

        coords["gb"] = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max, ndim))
        inds["gb"] = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max), dtype=bool)

        last_sample = State(coords, inds=inds, log_like=last_sample.log_like, log_prior=last_sample.log_prior)

        nleaves_max_fix = last_sample.branches["gb_fixed"].coords.shape[-2]  # will all be the same

        nleaves_max_fix_new = nleaves_max_fix + 10000

    else:
        raise FileNotFoundError(current_save_state_file + " and " + fp_pe + " not in current directory")

    coords_new = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_fix_new, ndim)) 
    coords_new[:, :, :nleaves_max_fix, :]= last_sample.branches["gb_fixed"].coords[:]

    inds_new = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_fix_new), dtype=bool) 
    inds_new[:, :, :nleaves_max_fix]= last_sample.branches["gb_fixed"].inds[:]

    last_sample = State({"gb_fixed": coords_new, "gb": last_sample.branches_coords["gb"]}, inds={"gb_fixed": inds_new, "gb": last_sample.branches_inds["gb"]}, log_like=last_sample.log_like, log_prior=last_sample.log_prior)

    A_going_in = np.zeros((ntemps_pe, nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((ntemps_pe, nwalkers_pe, E_inj.shape[0]), dtype=complex)
    A_going_in[:] = np.asarray(A_inj)
    E_going_in[:] = np.asarray(E_inj)

    A_psd_in = np.zeros((ntemps_pe, nwalkers_pe, A_inj.shape[0]), dtype=np.float64)
    E_psd_in = np.zeros((ntemps_pe, nwalkers_pe, E_inj.shape[0]), dtype=np.float64)
    A_psd_in[:] = np.asarray(psd)
    E_psd_in[:] = np.asarray(psd)

    """try:
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
    """

    mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df,
        base_injections=[A_inj, E_inj], base_psd=[psd.copy(), psd.copy()]
    )

    gb.d_d = xp.asarray(mgh.get_inner_product(use_cpu=True).flatten())

    mempool.free_all_blocks()
    
    # setup data streams to add to and subtract from
    supps_shape_in = (ntemps_pe, nwalkers_pe)

    gb.gpus = mgh.gpus

    """waveform_kwargs_in = {'dt': 5.0, 'T': 31457280.0, 'use_c_implementation': True, 'oversample': 4, 'start_freq_ind': 0}
    data_index = xp.asarray(np.load("data_index.npy")).astype(xp.int32)
    noise_index = data_index.copy()
    prior_generated_points_in = np.load("prior_gen_check.npy")
    N_temp = xp.asarray(np.load("N_temp.npy"))

    inds = xp.where(xp.asarray(prior_generated_points_in)[:, 1] > 1e-5)
    inds_cpu = [tmp.get() for tmp in list(inds)]
    gb.d_d = xp.asarray(np.load("gb_d_d.npy"))[inds]
    gb.get_ll(prior_generated_points_in[inds_cpu], mgh.data_list, mgh.psd_list, data_index=data_index[inds], noise_index=noise_index[inds], phase_marginalize=False, data_length=data_length,  data_splits=mgh.gpu_splits,  N=N_temp[inds], **waveform_kwargs_in).shape
    breakpoint()"""
    
    coords_out_gb_fixed = last_sample.branches["gb_fixed"].coords[last_sample.branches["gb_fixed"].inds]

    check = priors["gb_fixed"].logpdf(coords_out_gb_fixed)

    if np.any(np.isinf(check)):
        breakpoint()

    coords_out_gb_fixed[:, 3] = coords_out_gb_fixed[:, 3] % (2 * np.pi)
    coords_out_gb_fixed[:, 5] = coords_out_gb_fixed[:, 5] % (1 * np.pi)
    coords_out_gb_fixed[:, 6] = coords_out_gb_fixed[:, 6] % (2 * np.pi)
    # mgh = MultiGPUDataHolder(gpus, data_minus_templates_mix[0].get().reshape(ntemps_pe, nwalkers_pe, -1), data_minus_templates_mix[1].get().reshape(ntemps_pe, nwalkers_pe, -1), A_psd_in, E_psd_in, df)
    
    # mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df)

    coords_in_in = transform_fn.both_transforms(coords_out_gb_fixed)

    temp_vals = np.repeat(np.arange(ntemps_pe)[:, None], nleaves_max_fix_new * nwalkers_pe, axis=-1).reshape(ntemps_pe, nwalkers_pe, nleaves_max_fix_new)

    walker_vals = np.tile(np.arange(nwalkers_pe), (ntemps, nleaves_max_fix_new, 1)).transpose((0, 2, 1))

    data_index_1 = temp_vals * nwalkers_pe + walker_vals

    data_index = xp.asarray(data_index_1[last_sample.branches["gb_fixed"].inds]).astype(xp.int32)

    # goes in as -h
    factors = -xp.ones_like(data_index, dtype=xp.float64)

    gb.generate_global_template(coords_in_in, data_index, mgh.data_list, batch_size=1000, data_length=data_length, factors=factors, data_splits=mgh.gpu_splits, **waveform_kwargs)
    
    del data_index
    del factors
    mempool.free_all_blocks()
    
    # gb
    if last_sample.branches["gb"].nleaves.max().item() > 0:
        coords_out_gb = last_sample.branches["gb"].coords[last_sample.branches["gb"].inds]

        check = priors["gb_fixed"].logpdf(coords_out_gb)

        if np.any(np.isinf(check)):
            breakpoint()

        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)
        # mgh = MultiGPUDataHolder(gpus, data_minus_templates_mix[0].get().reshape(ntemps_pe, nwalkers_pe, -1), data_minus_templates_mix[1].get().reshape(ntemps_pe, nwalkers_pe, -1), A_psd_in, E_psd_in, df)
        
        # mgh = MultiGPUDataHolder(gpus, A_going_in, E_going_in, A_psd_in, E_psd_in, df)

        coords_in_in = transform_fn.both_transforms(coords_out_gb.reshape(-1, coords_out_gb.shape[-1]))
        data_index_tmp = np.repeat(np.arange(ntemps_pe * nwalkers_pe)[:, None], nleaves_max, axis=-1)[last_sample.branches["gb"].inds]
        data_index = xp.asarray(data_index_tmp).astype(xp.int32)

        # goes in as -h
        factors = -xp.ones_like(data_index, dtype=xp.float64)

        gb.generate_global_template(coords_in_in, data_index, mgh.data_list, batch_size=1000, data_length=data_length, factors=factors, data_splits=mgh.gpu_splits, **waveform_kwargs)
        
        del data_index
        del factors
        mempool.free_all_blocks()
        
    ll = mgh.get_ll(use_cpu=True)

    temp_inds = mgh.temp_indices.copy()
    walker_inds = mgh.walker_indices.copy()
    overall_inds = mgh.overall_indices.copy()
    
    supps = BranchSupplimental({ "temp_inds": temp_inds, "walker_inds": walker_inds, "overall_inds": overall_inds,}, base_shape=supps_base_shape, copy=True)

    state_mix = State(last_sample.branches_coords, inds=last_sample.branches_inds, log_like=ll.reshape(ntemps_pe, nwalkers_pe), supplimental=supps)
    from gbgpu.utils.utility import get_N

    for name in ["gb_fixed", "gb"]:
        ntemps_pe, nwalkers_pe, nleaves_max_here, _ = state_mix.branches[name].shape
        if nleaves_max_here == 0:
            continue
        points_start = state_mix.branches[name].coords[state_mix.branches[name].inds]
        N_vals = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_here), dtype=int)
        points_start_transform = transform_fn.both_transforms(points_start)
        amp_start = points_start_transform[:, 0]
        f0_start = points_start_transform[:, 1]
        
        N_temp = get_N(amp_start, f0_start, waveform_kwargs["T"], waveform_kwargs["oversample"])

        N_vals[state_mix.branches[name].inds] = N_temp
        branch_supp_base_shape = (ntemps_pe, nwalkers_pe, nleaves_max_here)
        state_mix.branches[name].branch_supplimental = BranchSupplimental({"N_vals": N_vals}, base_shape=branch_supp_base_shape, copy=True)

    gb_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=random_seed,
        nfriends=nwalkers,
        a=1.1
    )

    gb_args = (
        gb,
        priors,
        start_freq_ind,
        data_length,
        mgh,
        np.asarray(fd),
        search_f_bin_lims,
    )

    gb_fixed_move = GBSpecialStretchMove(
        *gb_args,
        **gb_kwargs,
    )

    """gb_move = GBSpecialGroupStretchMove(
        gb_args,
        gb_kwargs,
        nfriends=40
    )"""

    # add the other

    # stop_converge_mix = SearchConvergeStopping2(n_iters=7, diff=0.5, verbose=True, iter_back_check=7)

    gb_fixed_move.gb.gpus = gpus
    #gb_move.gb.gb_move = gpus


    moves_in_model = gb_fixed_move  # CombineMove([gb_fixed_move, gb_move])
    
    point_generator_func_tmp = deepcopy(priors["gb"].priors_in)
    point_generator_func_tmp[0] = uniform_dist(0.0, 1.0)
    point_generator_func_tmp[1] = uniform_dist(0.0, 1.0)

    point_generator_func = PriorContainer(point_generator_func_tmp)

    gb_kwargs_rj = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        # skip_supp_names_update=["group_move_points"],
        random_seed=10,
        nfriends=nwalkers,
        n_iter_update=20, 
    )

    gb_args_rj = (
        gb,
        priors,
        start_freq_ind,
        data_length,
        mgh,
        np.asarray(fd),
        search_f_bin_lims,
    )

    rj_moves = GBMutlipleTryRJ(
        gb_args_rj,
        gb_kwargs_rj,
        m_chirp_lims,
        [nleaves_max_fix_new, nleaves_max],
        [0, 0],
        num_try=int(1e1),
        gibbs_sampling_setup=["gb_fixed"],
        point_generator_func=point_generator_func,
    )
    rj_moves.gb.gpus = gpus

    like_mix = BasicResidualMGHLikelihood(mgh)
    branch_names = ["gb_fixed", "gb"]

    from eryn.moves.tempering import make_ladder
    betas = make_ladder(10000 * 8, ntemps=ntemps_pe)

    sampler_mix = EnsembleSampler(
            nwalkers_pe,
            [ndim, ndim],  # assumes ndim_max
            like_mix,
            priors,
            tempering_kwargs={"betas": betas},
            nbranches=len(branch_names),
            nleaves_max=[nleaves_max_fix_new, nleaves_max],
            moves=moves_in_model,
            rj_moves=rj_moves,
            kwargs=None,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
            backend=fp_pe,
            vectorize=True,
            periodic=periodic,  # TODO: add periodic to proposals
            branch_names=branch_names,
            stopping_fn=None,  # stop_converge_mix,
            stopping_iterations=-1,
            provide_groups=True,
            provide_supplimental=True,
        )

    # equlibrating likelihood check: -4293090.6483655665,
    
    """out = state_mix
    old_number = nleaves_max_fix_new
    starting_new = True
    print("starting new", nleaves_max_fix_new)

    if ((current_iteration + 1) % 40 == 0 or starting_new) and current_iteration != 0:
        out.branches["gb_fixed"].coords[-1] = out.branches["gb_fixed"].coords[0]
        out.branches["gb_fixed"].inds[-1] = out.branches["gb_fixed"].inds[0]
        out.branches["gb_fixed"].branch_supplimental[-1] = out.branches["gb_fixed"].branch_supplimental[0]
        # state_mix.supplimental[-1] = state_mix.supplimental[0]
        out.log_like[-1] = out.log_like[0]
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
    """
    nsteps_mix = 200

    print("Starting mix ll best:", state_mix.log_like.max(axis=-1))
    mempool.free_all_blocks()

    out = sampler_mix.run_mcmc(state_mix, nsteps_mix, progress=True, thin_by=20, store=True)
    print("ending mix ll best:", out.log_like.max(axis=-1))


    # needs to be max from mgh or need to map it
    lp_mgh = mgh.get_ll(use_cpu=True)
    max_ind = np.where(lp_mgh == lp_mgh.max())

    max_temp = max_ind[0][0]
    max_walker = max_ind[1][0]
    overall_ind_best = max_temp * nwalkers_pe + max_walker
    
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
    

    save_state = State(out.branches_coords, inds=out.branches_inds, log_like=out.log_like, log_prior=out.log_prior)

    if current_save_state_file in os.listdir():
        shutil.copyfile(current_save_state_file, "old_" + current_save_state_file)

    with open(current_save_state_file, "wb") as fp_out:
        pickle.dump(save_state, fp_out, protocol=pickle.HIGHEST_PROTOCOL)

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
    
    # )
    #if det_snr_finding < snr_break or len(current_snrs_search) > num_bin_pause:
    #    break
    current_iteration += 1



if __name__ == "__main__":
    output = run_equilibrate()
                