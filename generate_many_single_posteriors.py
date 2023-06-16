from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N

import pickle

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
from eryn.utils import PeriodicContainer
from eryn.moves.tempering import TemperatureControl

from eryn.moves import GroupStretchMove

from eryn.state import State

from full_band_global_fit_settings import *


def run_many_posteriors(gpu):
    xp.cuda.runtime.setDevice(gpu)

    #with open("mixing_gb_run.pickle", "rb") as fp_search_state:
    #    last_sample = pickle.load(fp_search_state)

    # ntemps_search, nwalkers_search, nleaves_search, ndim = last_sample.branches["gb_fixed"].shape
    
    # # reshape to group samples from the same binary
    # coords_out_gb_fixed = last_sample.branches_coords["gb_fixed"].transpose(2, 0, 1, 3)
    
    with open("mixing_gb_run_4_85.pickle", "rb") as fp: 
        last_state = pickle.load(fp)

    coords_out_gb_fixed = last_state.branches["gb_fixed"].coords.transpose(2, 0, 1, 3)

    nleaves_search, ntemps_search, nwalkers_search, ndim = coords_out_gb_fixed.shape

    coords_out_gb_fixed_orig2 = xp.asarray(coords_out_gb_fixed.copy())
    # coords_out_gb_fixed[:, :, :] = coords_out_gb_fixed[:, :, 0][:, :, None]
    coords_out_gb_fixed_orig = coords_out_gb_fixed.copy()

    cov = np.full(8, 1e-5)
    cov[1] = 1e-10
    cov[2] = 1e-7

    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for key, item in gpu_priors_in.items():
        item.use_cupy = True

    gpu_priors = {"gb_fixed": ProbDistContainer(gpu_priors_in, use_cupy=True)}
   
    # fix = np.ones(coords_out_gb_fixed.shape[:-1], dtype=bool)
    # while np.any(fix):
    #     coords_out_gb_fixed[fix] = coords_out_gb_fixed_orig[fix]
    #     coords_out_gb_fixed[fix] *= (1 + np.random.multivariate_normal(np.zeros_like(cov), np.diag(cov), size=coords_out_gb_fixed_orig.shape[:-1]))[fix]
    #     prev_logp = gpu_priors["gb_fixed"].logpdf(xp.asarray(coords_out_gb_fixed).reshape(-1, 8)).reshape(coords_out_gb_fixed.shape[:-1])
    #     fix = np.isinf(prev_logp.get())
    #     print(fix.sum())

    prev_logp = gpu_priors["gb_fixed"].logpdf(xp.asarray(coords_out_gb_fixed).reshape(-1, 8)).reshape(coords_out_gb_fixed.shape[:-1])
    assert not xp.any(xp.isinf(prev_logp))

    coords_in = transform_fn.both_transforms(
        coords_out_gb_fixed.reshape(-1, ndim)
    )
    data_index = xp.tile(np.arange(nwalkers_search), (nleaves_search, ntemps_search, 1)).flatten().astype(xp.int32)
    
    xp.get_default_memory_pool().free_all_blocks()
    # NEEDS TO BE -1
    factors = -xp.ones_like(data_index, dtype=xp.float64)
    xp.get_default_memory_pool().free_all_blocks()

    A_going_in = np.zeros((nwalkers_search, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((nwalkers_search, E_inj.shape[0]), dtype=complex)
    A_going_in[:] = np.asarray(A_inj)
    E_going_in[:] = np.asarray(E_inj)

    mbh_inj = np.load("best_logl_mbhs_from_psd_run_4" + ".npy")

    A_mbh_going_in = mbh_inj[0].squeeze()
    E_mbh_going_in = mbh_inj[1].squeeze()

    A_going_in[:] -= A_mbh_going_in
    E_going_in[:] -= E_mbh_going_in

    templates_out = [
        [xp.asarray(A_going_in.flatten())],
        [xp.asarray(E_going_in.flatten())],
    ]

    A_out, E_out = templates_out[0][0], templates_out[1][0]

    data_splits = [np.arange(nwalkers_search)]

    main_gpu = xp.cuda.runtime.getDevice()

    gb.gpus = [gpu]

    # coords_in[0, 0] *= 1e-10
    gb.generate_global_template(
        coords_in,
        data_index,
        templates_out,
        data_length=data_length,
        factors=factors,
        data_splits=data_splits,
        **waveform_kwargs,
    )

    A_psd, E_psd = np.load("best_logl_psd_from_psd_run_4.npy")
    A_psd[0] = A_psd[1]
    E_psd[0] = E_psd[1]
    A_psd = xp.asarray(A_psd)
    E_psd = xp.asarray(E_psd)

    psd_out = [
        [xp.asarray(A_psd.flatten())],
        [xp.asarray(E_psd.flatten())],
    ]

    ll = -1/2 * 4 * df * xp.sum(
        (A_out.reshape(nwalkers_search, -1).conj() * A_out.reshape(nwalkers_search, -1)) / A_psd
        + (E_out.reshape(nwalkers_search, -1).conj() * E_out.reshape(nwalkers_search, -1)) / E_psd,
        axis=-1
    ).real - xp.sum(xp.log(psd_out[0][0]) + xp.log(psd_out[1][0]))

    xp.get_default_memory_pool().free_all_blocks()

    prev_logl = xp.tile(ll, (nleaves_search, ntemps_search, 1))

    logl_original = prev_logl.copy()

    temperature_control = TemperatureControl(ndim, nwalkers_search, ntemps=ntemps_search)
    move_proposal = StretchMove(periodic=PeriodicContainer(periodic), temperature_control=temperature_control, return_gpu=True, use_gpu=True)
    
    coords_out_gb_fixed_in = transform_fn.both_transforms(coords_out_gb_fixed.reshape(-1, ndim), xp=xp)

    noise_index = xp.zeros_like(data_index).astype(np.int32)
 
    gb.d_d = 0.0
    #ll_test1 = gb.get_ll(coords_out_gb_fixed_in, templates_out, psd_out, data_index=data_index, noise_index=noise_index, data_length=data_length, data_splits=data_splits, **waveform_kwargs)

    N_vals = xp.asarray(get_N(np.full_like(coords_out_gb_fixed_in[:, 1], 1e-30), coords_out_gb_fixed_in[:, 1], Tobs=waveform_kwargs["T"], oversample=waveform_kwargs["oversample"]))
    waveform_kwargs.pop("N")

    coords_out_gb_fixed_in_2 = coords_out_gb_fixed_in.copy()
    # coords_out_gb_fixed_in[0, 0] *= 1e-10
    # coords_out_gb_fixed_in_2[1:, 0] *= 1.01
    ll_diff_test = gb.swap_likelihood_difference(
        coords_out_gb_fixed_in,
        coords_out_gb_fixed_in_2,
        templates_out,
        psd_out,
        N=N_vals,
        start_freq_ind=0,
        data_index=data_index,
        noise_index=noise_index,
        data_length=data_length,
        data_splits=data_splits,
        phase_marginalize=False,
        **waveform_kwargs,
    )

    gb.d_d = 0.0
    # ll_test = gb.get_ll(coords_out_gb_fixed_in_2, templates_out, psd_out, N=N_vals, data_index=data_index, noise_index=noise_index, data_length=data_length, data_splits=data_splits, **waveform_kwargs) - gb.get_ll(coords_out_gb_fixed_in, templates_out, psd_out, data_index=data_index, N=N_vals, noise_index=noise_index, data_length=data_length, data_splits=data_splits, **waveform_kwargs)

    N_vals = N_vals.reshape(nleaves_search, ntemps_search, nwalkers_search)

    logl_change = []

    original_logls = prev_logl.copy()

    if xp.any(xp.isnan(prev_logl)):
        breakpoint()
    
    leaf_guide = xp.repeat(xp.arange(nleaves_search)[:, None], ntemps_search * nwalkers_search, axis=-1).reshape(nleaves_search, ntemps_search, nwalkers_search)
    temp_guide = xp.repeat(xp.arange(ntemps_search)[:, None], nleaves_search * nwalkers_search, axis=-1).reshape(ntemps_search, nleaves_search, nwalkers_search).transpose(1, 0, 2)
    walker_guide = xp.repeat(xp.arange(nwalkers_search)[:, None], nleaves_search * ntemps_search, axis=-1).reshape(nwalkers_search, nleaves_search, ntemps_search).transpose(1, 2, 0)

    betas_here = xp.asarray(temperature_control.betas)[temp_guide[:, :, 0]]
    old_points = xp.asarray(coords_out_gb_fixed.copy())

    old_points_that_stay_in_the_data = old_points.copy()

    changeover_iter = 100
    num_max_proposals = 10000

    reader_out_iter = 25

    import tqdm

    num_file_adds = int(num_max_proposals / reader_out_iter) 

    # TODO: check on low snr binaries going in and out
    individual_posterior_fp = "search_find_individual_posteriors_3.h5"

    import os
    if False:  # individual_posterior_fp in os.listdir():
        with h5py.File(individual_posterior_fp, "r") as fp:
            check = fp["samples"][:, 0, 0, 0, 0]
            current_readout_iter = np.where(check == 0.0)[0][0]
            old_points_change = xp.asarray(fp["samples"][90])

        old_points_in = transform_fn.both_transforms(old_points.reshape(-1, ndim), xp=xp)
        old_points_change_in = transform_fn.both_transforms(old_points_change.reshape(-1, ndim), xp=xp)

        ll_diff_start = gb.swap_likelihood_difference(
            old_points_in,
            old_points_change_in,
            templates_out,
            psd_out,
            N=N_vals.flatten(),
            start_freq_ind=0,
            data_index=data_index,
            noise_index=noise_index,
            data_length=data_length,
            data_splits=data_splits,
            phase_marginalize=False,
            **waveform_kwargs,
        )
        breakpoint()
        prev_logl += ll_diff_start.reshape(nleaves_search, ntemps_search, nwalkers_search)

    else:
        # pass
        current_readout_iter = 1
        print("starting file save")
        with h5py.File(individual_posterior_fp, "w") as fp:
            fp.create_dataset("samples", data=np.zeros((num_file_adds,) + old_points.shape), dtype=np.float64, compression="gzip", compression_opts=9)
            fp["samples"][0] = old_points.get()
    print("start", print(np.sort(old_points[0, 0, :, 0])))
    for prop_i in tqdm.tqdm(range(num_max_proposals)):  # tqdm(range(num_max_proposals)):
        # st = time.perf_counter()
        inds_split = np.arange(nwalkers_search)
    
        np.random.shuffle(inds_split)
        
        for split in range(2):
            inds_here = np.arange(nwalkers_search)[inds_split % 2 == split]
            inds_not_here = np.delete(np.arange(nwalkers_search), inds_here)

            inds_here = xp.asarray(inds_here)
            inds_not_here = xp.asarray(inds_not_here)

            s_in = old_points[:, :, inds_here].reshape((nleaves_search * ntemps_search, int(nwalkers_search/2), 1, -1))
            c_in = [old_points[:, :, inds_not_here].reshape((nleaves_search * ntemps_search, int(nwalkers_search/2), 1, -1))]

            leaves_here = leaf_guide[:, :, inds_here]
            temps_here = temp_guide[:, :, inds_here]
            walkers_here = walker_guide[:, :, inds_here]
            # bands_here = band_guide[:, inds_here]

            new_points_dict, factors = move_proposal.get_proposal({"gb_fixed": s_in}, {"gb_fixed": c_in}, xp.random)
            new_points = new_points_dict["gb_fixed"].reshape(nleaves_search, ntemps_search, int(nwalkers_search/2), -1)
            
            logp = gpu_priors["gb_fixed"].logpdf(new_points.reshape(-1, ndim)).reshape(new_points.shape[:-1])
            factors = factors.reshape(logp.shape)
            
            fdiff = xp.abs(new_points[:, :, :, 1] - coords_out_gb_fixed_orig2[:, :, inds_here, 1]) / coords_out_gb_fixed_orig2[:, :, inds_here, 1]

            logp[fdiff > 1e-4] = -xp.inf

            keep_logp = ~xp.isinf(logp)          
            new_points_keep = new_points[keep_logp]

            old_points_that_stay_in_the_data_in = transform_fn.both_transforms(old_points_that_stay_in_the_data[:, :, inds_here][keep_logp], xp=xp)
            new_points_in = transform_fn.both_transforms(new_points_keep, xp=xp)

            ll_diff = xp.full_like(logp, -1e300)

            data_index_in = walkers_here[keep_logp].astype(np.int32)
            noise_index_in = xp.zeros_like(data_index_in).astype(np.int32) 
            N_vals_in = N_vals[:, :, inds_here][keep_logp]

            ll_diff[keep_logp] = gb.swap_likelihood_difference(
                old_points_that_stay_in_the_data_in,
                new_points_in,
                templates_out,
                psd_out,
                N=N_vals_in,
                start_freq_ind=0,
                data_index=data_index_in,
                noise_index=noise_index_in,
                data_length=data_length,
                data_splits=data_splits,
                phase_marginalize=False,
                **waveform_kwargs,
            )

            # ll_test = gb.get_ll(new_points_in, templates_out, psd_out, data_index=data_index_in, noise_index=noise_index_in, data_length=data_length, data_splits=data_splits, **waveform_kwargs) - gb.get_ll(old_points_in, templates_out, psd_out, data_index=data_index_in, noise_index=noise_index_in, data_length=data_length, data_splits=data_splits, **waveform_kwargs)

            # fix any nans that may come up
            ll_diff[xp.isnan(ll_diff)] = -1e300

            logl = logl_original[:, :, inds_here] + ll_diff

            xp.cuda.runtime.deviceSynchronize()
            
            prev_logl_here = prev_logl[:, :, inds_here]
            prev_logp_here = prev_logp[:, :, inds_here]

            lnpdiff = factors + betas_here[:, None] * (logl - prev_logl_here) + (logp - prev_logp_here)
            keep = lnpdiff > xp.asarray(xp.log(xp.random.rand(*lnpdiff.shape)))

            prev_logp[leaves_here[keep], temps_here[keep], walkers_here[keep]] = logp[keep]
            prev_logl[leaves_here[keep], temps_here[keep], walkers_here[keep]] = logl[keep]
            old_points[leaves_here[keep], temps_here[keep], walkers_here[keep]] = new_points[keep]
            
        if (prop_i + 1) % reader_out_iter == 0:
            with h5py.File(individual_posterior_fp, "a") as fp:
                fp["samples"][current_readout_iter] = old_points.get()
                current_readout_iter += 1

        if (prop_i + 1) % changeover_iter == 0:
            fdiff = xp.abs(old_points[:, :, :, 1] - coords_out_gb_fixed_orig2[:, :, :, 1])
            print(fdiff.max(), coords_out_gb_fixed_orig2[:, :, :, 1].flatten()[fdiff.argmax()])
            sorted_amps = np.sort(old_points[150, 0, :, 0])
            sorted_fs = np.sort(old_points[150, 0, :, 1])
            print(sorted_amps.mean(), sorted_amps.max() - sorted_amps.min(), sorted_fs.mean(), sorted_fs.max() - sorted_fs.min())
            
            print("changeover", prop_i + 1)


if __name__ == "__main__":
    gpu = 6
    xp.cuda.runtime.setDevice(gpu)
    
    run_many_posteriors(gpu)
    breakpoint()
    
    

