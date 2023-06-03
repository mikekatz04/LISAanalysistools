from full_band_global_fit_settings import *

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N
import pickle
from tqdm import tqdm

mempool = xp.get_default_memory_pool()

from full_band_global_fit_settings import *
from single_mcmc_run import run_single_band_search
from lisatools.utils.multigpudataholder import MultiGPUDataHolder
from eryn.moves import CombineMove
from lisatools.sampling.moves.specialforegroundmove import GBForegroundSpecialMove
from initial_psd_search import run_psd_search

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from lisatools.utils.utility import searchsorted2d_vec, get_groups_from_band_structure
from eryn.moves import StretchMove
from eryn.moves import TemperatureControl
from eryn.prior import ProbDistContainer
from eryn.utils.utility import groups_from_inds
from eryn.utils import PeriodicContainer

from eryn.moves import GroupStretchMove

from lisatools.diagnostic import inner_product
from eryn.state import State

def shuffle_along_axis(a, axis, xp=None):
    if xp is None:
        xp = np
    idx = xp.random.rand(*a.shape).argsort(axis=axis)
    return xp.take_along_axis(a,idx,axis=axis)

def run_iterative_subtraction_mcmc(iter_i, ndim, nwalkers, ntemps, band_inds_running, evens_odds, priors_good, f0_maxs, f0_mins, fdot_maxs, fdot_mins, data_in, psd_in, binaries_found):
    xp.cuda.runtime.setDevice(xp.cuda.runtime.getDevice())
    temperature_control = TemperatureControl(ndim, nwalkers, ntemps=ntemps)

    num_max_proposals = 100000
    convergence_iter_count = 500

    move_proposal = StretchMove(periodic=PeriodicContainer(periodic), temperature_control=temperature_control, return_gpu=True, use_gpu=True)
    
    band_inds_here = xp.where(xp.asarray(band_inds_running) & (xp.arange(len(band_inds_running)) % 2 == evens_odds))[0]

    new_points = priors_good.rvs(size=(ntemps, nwalkers, len(band_inds_here)))

    fix = xp.any(xp.isinf(new_points), axis=-1) | xp.any(xp.isnan(new_points), axis=-1)
    while xp.any(fix):
        tmp = priors_good.rvs(size=int((fix.flatten() == True).sum()))
        new_points[fix == True] = tmp
        fix = xp.any(xp.isinf(new_points), axis=-1) | xp.any(xp.isnan(new_points), axis=-1)

    # TODO: fix fs stuff
    prev_logp = priors_good.logpdf(new_points.reshape(-1, ndim)).reshape(new_points.shape[:-1])
    assert not xp.any(xp.isinf(prev_logp))
    new_points_with_fs = new_points.copy()
    new_points_with_fs[:, :, :, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * new_points_with_fs[:, :, :, 1] + f0_mins[band_inds_here]
    new_points_with_fs[:, :, :, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * new_points_with_fs[:, :, :, 2] + fdot_mins[band_inds_here]

    new_points_in = transform_fn.both_transforms(new_points_with_fs.reshape(-1, ndim), xp=xp).reshape(new_points_with_fs.shape[:-1] + (ndim + 1,)).reshape(-1, ndim + 1)
    gb.d_d = 0.0

    prev_logl = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs).reshape(prev_logp.shape))

    if xp.any(xp.isnan(prev_logl)):
        breakpoint()

    old_points = new_points.copy()
    
    best_logl = prev_logl.max(axis=(0, 1))
    best_logl_ind = prev_logl.reshape(ntemps * nwalkers, len(band_inds_here)).argmax(axis=0)
    
    best_logl_coords = old_points.reshape(ntemps * nwalkers, len(band_inds_here), ndim)[(best_logl_ind, xp.arange(len(band_inds_here)))]

    start_best_logl = best_logl.copy()

    temp_guide = xp.repeat(xp.arange(ntemps)[:, None], nwalkers * len(band_inds_here), axis=-1).reshape(ntemps, nwalkers, len(band_inds_here))
    walker_guide = xp.repeat(xp.arange(nwalkers)[:, None], ntemps * len(band_inds_here), axis=-1).reshape(nwalkers, ntemps, len(band_inds_here)).transpose(1, 0, 2)
    band_guide = xp.repeat(xp.arange(len(band_inds_here))[None, :], ntemps * nwalkers, axis=0).reshape(ntemps, nwalkers, len(band_inds_here))

    still_going_here = xp.ones(len(band_inds_here), dtype=bool)
    num_proposals_per = np.zeros_like(still_going_here, dtype=int)
    iter_count = np.zeros_like(still_going_here, dtype=int)
    betas = xp.repeat(xp.asarray(temperature_control.betas[:, None].copy()), len(band_inds_here), axis=-1)

    for prop_i in range(num_max_proposals):  # tqdm(range(num_max_proposals)):
        # st = time.perf_counter()
        num_still_going_here = still_going_here.sum().item()
        inds_split = np.arange(nwalkers)
    
        np.random.shuffle(inds_split)
        
        for split in range(2):
            inds_here = np.arange(nwalkers)[inds_split % 2 == split]
            inds_not_here = np.delete(np.arange(nwalkers), inds_here)

            inds_here = xp.asarray(inds_here)
            inds_not_here = xp.asarray(inds_not_here)

            s_in = old_points[:, inds_here][:, :, still_going_here].reshape((ntemps, int(nwalkers/2) * num_still_going_here, 1, -1))
            c_in = [old_points[:, inds_not_here][:, :, still_going_here].reshape((ntemps, int(nwalkers/2) * num_still_going_here, 1, -1))]

            temps_here = temp_guide[:, inds_here][:, :, still_going_here]
            walkers_here = walker_guide[:, inds_here][:, :, still_going_here]
            bands_here = band_guide[:, inds_here][:, :, still_going_here]

            new_points_dict, factors = move_proposal.get_proposal({"gb_fixed": s_in}, {"gb_fixed": c_in}, xp.random)
            new_points = new_points_dict["gb_fixed"].reshape(ntemps, int(nwalkers/2), num_still_going_here, -1)
            logp = priors_good.logpdf(new_points.reshape(-1, ndim)).reshape(new_points.shape[:-1])
            factors = factors.reshape(logp.shape)
            keep_logp = ~xp.isinf(logp)

            new_points_with_fs = new_points.copy()

            new_points_with_fs[:, :, :, 1] = (f0_maxs[None, None, band_inds_here[still_going_here]] - f0_mins[None, None, band_inds_here[still_going_here]]) * new_points_with_fs[:, :, :, 1] + f0_mins[None, None, band_inds_here[still_going_here]]
            new_points_with_fs[:, :, :, 2] = (fdot_maxs[None, None, band_inds_here[still_going_here]] - fdot_mins[None, None, band_inds_here[still_going_here]]) * new_points_with_fs[:, :, :, 2] + fdot_mins[None, None, band_inds_here[still_going_here]]

            new_points_with_fs_keep = new_points_with_fs[keep_logp]
            new_points_in = transform_fn.both_transforms(new_points_with_fs_keep, xp=xp)

            logl = xp.full_like(logp, -1e300)

            logl[keep_logp] = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

            # fix any nans that may come up
            logl[xp.isnan(logl)] = -1e300

            xp.cuda.runtime.deviceSynchronize()
            
            prev_logl_here = prev_logl[:, inds_here][:, :, still_going_here]
            prev_logp_here = prev_logp[:, inds_here][:, :, still_going_here]
            
            prev_logP_here = betas[:, still_going_here][:, None, :] * prev_logl_here + prev_logp_here

            logP = betas[:, still_going_here][:, None, :] * logl + logp

            lnpdiff = factors + logP - prev_logP_here
            keep = lnpdiff > xp.asarray(xp.log(xp.random.rand(*logP.shape)))

            prev_logp[temps_here[keep], walkers_here[keep], bands_here[keep]] = logp[keep]
            prev_logl[temps_here[keep], walkers_here[keep], bands_here[keep]] = logl[keep]
            old_points[temps_here[keep], walkers_here[keep], bands_here[keep]] = new_points[keep]
            
        # prepare information on how many swaps are accepted this time
        swaps_accepted = xp.zeros((ntemps - 1, num_still_going_here), dtype=int)
        swaps_proposed = xp.full_like(swaps_accepted, nwalkers)

        # iterate from highest to lowest temperatures
        for i in range(ntemps - 1, 0, -1):
            # get both temperature rungs
            bi = betas[i, still_going_here]
            bi1 = betas[i - 1, still_going_here]

            # difference in inverse temps
            dbeta = bi1 - bi

            # permute the indices for the walkers in each temperature to randomize swap positions
            iperm = shuffle_along_axis(xp.tile(xp.arange(nwalkers), (num_still_going_here, 1)), -1)
            i1perm = shuffle_along_axis(xp.tile(xp.arange(nwalkers), (num_still_going_here, 1)), -1)
            
            # random draw that produces log of the acceptance fraction
            raccept = xp.log(xp.random.uniform(size=(num_still_going_here, nwalkers)))
            
            # log of the detailed balance fraction
            walker_swap_i = iperm.flatten()
            walker_swap_i1 = i1perm.flatten()

            temp_swap_i = np.full_like(walker_swap_i, i)
            temp_swap_i1 = np.full_like(walker_swap_i1, i - 1)
            band_swap = xp.repeat(xp.arange(len(still_going_here))[still_going_here], nwalkers)

            paccept = dbeta[:, None] * (
                prev_logl[(temp_swap_i, walker_swap_i, band_swap)].reshape(num_still_going_here, nwalkers)
                - prev_logl[(temp_swap_i1, walker_swap_i1, band_swap)].reshape(num_still_going_here, nwalkers)
            )

            # How many swaps were accepted
            sel = paccept > raccept
            swaps_accepted[i - 1] = xp.sum(sel, axis=-1)

            temp_swap_i_keep = temp_swap_i[sel.flatten()]
            walker_swap_i_keep = walker_swap_i[sel.flatten()]
            band_swap_keep = band_swap[sel.flatten()]

            temp_swap_i1_keep = temp_swap_i1[sel.flatten()]
            walker_swap_i1_keep = walker_swap_i1[sel.flatten()]

            coords_tmp_i = old_points[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)].copy()
            logl_tmp_i = prev_logl[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)].copy()
            logp_tmp_i = prev_logp[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)].copy()

            old_points[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)] = old_points[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)]
            prev_logl[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)] = prev_logl[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)]
            prev_logp[(temp_swap_i_keep, walker_swap_i_keep, band_swap_keep)] = prev_logp[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)]

            old_points[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)] = coords_tmp_i
            prev_logl[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)] = logl_tmp_i
            prev_logp[(temp_swap_i1_keep, walker_swap_i1_keep, band_swap_keep)] = logp_tmp_i
        # print(prev_logl.max(axis=(1, 2)))
        
        # print(time.perf_counter() - st)
        ratios = swaps_accepted / swaps_proposed
        # adjust temps 
        betas0 = betas[:, still_going_here].copy()
        betas1 = betas[:, still_going_here].copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = temperature_control.adaptation_lag / (prop_i + temperature_control.adaptation_lag)
        kappa = decay / temperature_control.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = xp.diff(1 / betas1[:-1], axis=0)
        deltaTs *= xp.exp(dSs)
        betas1[1:-1] = 1 / (np.cumsum(deltaTs, axis=0) + 1 / betas1[0])

        dbetas = betas1 - betas0
        betas[:, still_going_here] += dbetas

        new_best_logl = prev_logl.max(axis=(0, 1))

        improvement = (new_best_logl - best_logl > 0.01)

        # print(new_best_logl - best_logl, best_logl)
        best_logl[improvement] = new_best_logl[improvement]

        best_logl_ind = prev_logl.reshape(ntemps * nwalkers, len(band_inds_here)).argmax(axis=0)[improvement]
        best_logl_coords[improvement] = old_points.reshape(ntemps * nwalkers, len(band_inds_here), ndim)[(best_logl_ind, xp.arange(len(band_inds_here))[improvement])]

        if prop_i > 500:
            iter_count[improvement] = 0
            iter_count[~improvement] += 1

        num_proposals_per[still_going_here] += 1
        still_going_here[iter_count >= convergence_iter_count] = False
        
        if prop_i % 500 == 0:
            print(f"Proposal {prop_i}, Still going:", still_going_here.sum().item())
        if still_going_here.sum().item() == 0:
            break

    best_binaries_coords_with_fs = best_logl_coords.copy()

    best_binaries_coords_with_fs[:, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 1] + f0_mins[band_inds_here]
    best_binaries_coords_with_fs[:, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 2] + fdot_mins[band_inds_here]

    best_logl_points_in = transform_fn.both_transforms(best_binaries_coords_with_fs, xp=xp)

    best_logl_check = xp.asarray(gb.get_ll(best_logl_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

    if not xp.allclose(best_logl, best_logl_check):
        breakpoint()

    snr_lim = 7.0
    keep_binaries = gb.d_h / xp.sqrt(gb.h_h.real) > snr_lim
    # TODO: add in based on sensitivity changing
    # band_inds_running[band_inds_here[~keep_binaries].get()] = False
    keep_coords = best_binaries_coords_with_fs[keep_binaries].get()

    # adjust the phase from marginalization
    phase_change = np.angle(gb.non_marg_d_h)[keep_binaries.get()]
    keep_coords[:, 3] -= phase_change
    # best_logl_points_in[keep_binaries, 4] -= xp.asarray(phase_change)

    nwalkers_pe = nwalkers
    ntemps_pe = 1
    
    factor = 1e-5
    cov = np.ones(ndim) * 1e-3
    cov[1] = 1e-8

    still_going_start_like = np.ones(keep_coords.shape[0], dtype=bool)
    starting_points = np.zeros((keep_coords.shape[0], nwalkers_pe * ntemps_pe, ndim))
    iter_check = 0
    max_iter = 1000
    while np.any(still_going_start_like):
        num_still_going_start_like = still_going_start_like.sum().item()
        
        start_like = np.zeros((num_still_going_start_like, nwalkers_pe * ntemps_pe))
    
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((num_still_going_start_like, ntemps_pe * nwalkers_pe, ndim))
        fix = np.ones((num_still_going_start_like, ntemps_pe * nwalkers_pe), dtype=bool)
        while np.any(fix):
            tmp[fix] = (keep_coords[still_going_start_like, None, :] * (1. + factor * cov * np.random.randn(num_still_going_start_like, nwalkers_pe * ntemps_pe, ndim)))[fix]

            tmp[:, :, 3] = tmp[:, :, 3] % (2 * np.pi)
            tmp[:, :, 5] = tmp[:, :, 5] % (np.pi)
            tmp[:, :, 6] = tmp[:, :, 6] % (2 * np.pi)
            logp = priors["gb_fixed"].logpdf(tmp.reshape(-1, ndim)).reshape(tmp.shape[:-1])

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        tmp_in = transform_fn.both_transforms(tmp.reshape(-1, ndim))
        start_like = gb.get_ll(tmp_in, data_in, psd_in, phase_marginalize=False, **waveform_kwargs).reshape(tmp.shape[:-1])

        starting_points[still_going_start_like] = tmp
        
        update = np.arange(still_going_start_like.shape[0])[still_going_start_like][np.std(start_like, axis=-1) > 5.0]
        still_going_start_like[update] = False 

        iter_check += 1
        factor *= 1.5

        # print(np.std(start_like))

        if iter_check > max_iter:
            raise ValueError("Unable to find starting parameters.")

    starting_points = starting_points.reshape((keep_coords.shape[0], ntemps_pe, nwalkers_pe, ndim)).transpose(1, 2, 0, 3)

    if binaries_found is None:
        binaries_found = keep_coords
    else:
        binaries_found = np.concatenate([binaries_found, keep_coords], axis=0)

    binaries_found = binaries_found[np.argsort(binaries_found[:, 1])]

    num_binaries_found_this_iteration = keep_binaries.sum().item()

    return starting_points

def run_gb_mixing(iter_i, gpus, fp_gb_mixing, num_binaries_found_this_iteration, starting_points):
    ntemps_pe, nwalkers_pe = starting_points.shape[:2]
    ndim = 8
    if fp_gb_mixing in os.listdir():
        with open(fp_gb_mixing, "rb") as f_tmp:
            last_sample = pickle.load(f_tmp)
        
        nleaves_max_fix = last_sample.branches["gb_fixed"].coords.shape[-2]  # will all be the same

        nleaves_max_fix_new = nleaves_max_fix + num_binaries_found_this_iteration
        coords_new = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_fix_new, ndim)) 
        coords_new[:, :, :nleaves_max_fix, :]= last_sample.branches["gb_fixed"].coords[:]

        betas_mix = last_sample.betas

        # inds_new = np.ones((ntemps_pe, nwalkers_pe, nleaves_max_fix_new), dtype=bool) 
        # inds_new[:, :, :nleaves_max_fix]= last_sample.branches["gb_fixed"].inds[:]

    else:
        nleaves_max_fix = 0
        nleaves_max_fix_new = num_binaries_found_this_iteration
        coords_new = np.zeros((ntemps_pe, nwalkers_pe, nleaves_max_fix_new, ndim))
        betas_mix = np.linspace(1.0, 0.95, ntemps_pe)

    inds_new = np.ones((ntemps_pe, nwalkers_pe, nleaves_max_fix_new), dtype=bool) 
    coords_new[:, :, nleaves_max_fix:] = starting_points
    
    last_sample = State({"gb_fixed": coords_new}, inds={"gb_fixed": inds_new})

    A_going_in = np.zeros((ntemps_pe, nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_going_in = np.zeros((ntemps_pe, nwalkers_pe, E_inj.shape[0]), dtype=complex)
    A_going_in[:] = np.asarray(A_inj).copy()
    E_going_in[:] = np.asarray(E_inj).copy()

    imported = False

    while not imported:
        try:
            mbh_inj = np.load("best_logl_mbhs_from_psd_run.npy")  # fp_mbh_template_search + ".npy")
            imported = True
        except ValueError:
            time.sleep(1)

    A_mbh_going_in = np.zeros((ntemps_pe, nwalkers_pe, A_inj.shape[0]), dtype=complex)
    E_mbh_going_in = np.zeros((ntemps_pe, nwalkers_pe, E_inj.shape[0]), dtype=complex)

    A_mbh_going_in[:] = mbh_inj[0][None, None, :]
    E_mbh_going_in[:] = mbh_inj[1][None, None, :]

    last_mbh_template = [A_mbh_going_in, E_mbh_going_in]

    A_going_in[:] -= A_mbh_going_in
    E_going_in[:] -= E_mbh_going_in
    
    A_psd_in = np.zeros((ntemps_pe, nwalkers_pe, A_inj.shape[0]), dtype=np.float64)
    E_psd_in = np.zeros((ntemps_pe, nwalkers_pe, E_inj.shape[0]), dtype=np.float64)

    A_psd_in[:] = np.asarray(psd)
    E_psd_in[:] = np.asarray(psd)
    imported = False
    while not imported:
        try:
            psds = np.load("best_logl_psd_from_psd_run.npy")  # fp_psd_residual_search + ".npy" )
            imported = True
        except ValueError:
            time.sleep(1)
    
    psds[:, 0] = psds[:, 1]
    A_psd_in[:] = psds[0][None, None, :]  # A
    E_psd_in[:] = psds[1][None, None, :]  # A

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
        base_injections=[A_inj, E_inj], base_psd=None  # [psd.copy(), psd.copy()]
    )

    # psd_params = last_sample.branches["psd"].coords.reshape(-1, last_sample.branches["psd"].shape[-1])

    # foreground_params = last_sample.branches["galfor"].coords.reshape(-1, last_sample.branches["galfor"].shape[-1])

    # mgh.set_psd_vals(psd_params, foreground_params=foreground_params)

    gb.d_d = xp.asarray(mgh.get_inner_product().flatten())
    check = mgh.get_psd_term()

    mempool.free_all_blocks()
    
    # setup data streams to add to and subtract from
    supps_shape_in = (ntemps_pe, nwalkers_pe)

    gb.gpus = mgh.gpus
    
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

    walker_vals = np.tile(np.arange(nwalkers_pe), (ntemps_pe, nleaves_max_fix_new, 1)).transpose((0, 2, 1))

    data_index_1 = temp_vals * nwalkers_pe + walker_vals

    data_index = xp.asarray(data_index_1[last_sample.branches["gb_fixed"].inds]).astype(xp.int32)

    # goes in as -h
    factors = -xp.ones_like(data_index, dtype=xp.float64)

    # gb.d_d = gb.d_d[data_index]
    # ll = gb.get_ll(coords_in_in, mgh.data_list, mgh.psd_list, data_index=data_index, noise_index=data_index.copy(), phase_marginalize=False, data_length=data_length,  data_splits=mgh.gpu_splits, return_cupy=True, **waveform_kwargs)
    gb.generate_global_template(coords_in_in, data_index, mgh.data_list, batch_size=1000, data_length=data_length, factors=factors, data_splits=mgh.gpu_splits, **waveform_kwargs)

    del data_index
    del factors
    mempool.free_all_blocks()
    
    ll = mgh.get_ll(include_psd_info=True)

    # psd fit
    inds = ll.argmax()
    # A_going_in_psd_fit, E_going_in_psd_fit = mgh.data_shaped[0][0][inds], mgh.data_shaped[1][0][inds]

    # psd_A, psd_E = run_psd_search(A_going_in_psd_fit, E_going_in_psd_fit, gpus[0])
    
    # A_psd_in[:] = np.asarray(psd_A).copy()
    # E_psd_in[:] = np.asarray(psd_E).copy()

    xp.get_default_memory_pool().free_all_blocks()
    mgh.set_psd_from_arrays(
        A_psd_in.reshape(-1, A_psd_in.shape[-1]),
        E_psd_in.reshape(-1, E_psd_in.shape[-1]),
        overall_inds=mgh.map
    )
    
    temp_inds = mgh.temp_indices.copy()
    walker_inds = mgh.walker_indices.copy()
    overall_inds = mgh.overall_indices.copy()
    
    supps = BranchSupplimental({ "temp_inds": temp_inds, "walker_inds": walker_inds, "overall_inds": overall_inds,}, base_shape=supps_base_shape, copy=True)

    lp = priors["gb"].logpdf(last_sample.branches_coords["gb_fixed"].reshape(-1, 8)).reshape(last_sample.branches_coords["gb_fixed"].shape[:-1]).sum(axis=-1)

    state_mix = State(last_sample.branches_coords, inds=last_sample.branches_inds, log_like=ll.reshape(ntemps_pe, nwalkers_pe), supplimental=supps, log_prior=lp)
    from gbgpu.utils.utility import get_N

    for name in ["gb_fixed"]:
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

    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for key, item in gpu_priors_in.items():
        item.use_cupy = True

    gpu_priors = {"gb_fixed": ProbDistContainer(gpu_priors_in, use_cupy=True)}
    
    gb_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=transform_fn,
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=random_seed,
        nfriends=nwalkers,
        a=1.7,
        use_gpu=True
    )

    gb_args = (
        gb,
        priors,
        start_freq_ind,
        data_length,
        mgh,
        np.asarray(fd),
        search_f_bin_lims,
        gpu_priors
    )
    betas_mix = np.array([1.0])
    temperature_control_mix = TemperatureControl(8, nwalkers_pe, betas=betas_mix)
    gb_fixed_move = GBSpecialStretchMove(
        *gb_args,
        **gb_kwargs,
        n_iter_update=15,
        temperature_control=temperature_control_mix
    )
    gb_fixed_move.accepted = np.zeros((ntemps_pe, nwalkers_pe))

    mempool.free_all_blocks()

    class TempClass:
        def __init__(self, random):
            self.random = random

    model_placeholder = TempClass(xp.random)
    print(f"Starting mix for iteration {iter_i}")
    mixing_steps = 10000
    save_every_steps = 10
    iters_maximize = 50
    max_logl = state_mix.log_like.max()
    max_logl_iters = 0
    for mix_step in tqdm(range(mixing_steps)):
        
        state_mix, accepted = gb_fixed_move.propose(model_placeholder, state_mix)

        if state_mix.log_like.max() > max_logl:
            max_logl_iters = 0
            max_logl = state_mix.log_like.max()
        else:
            max_logl_iters += 1

        save = False
        if (mix_step) % save_every_steps == 0:
            save = True
        if max_logl_iters >= iters_maximize:
            # TODO: switch to per band likelihood converge
            save = True
            converged = True
        else:
            converged = False

        print("gb status", mix_step, max_logl, max_logl_iters, converged)
        
        if save:
            
            best_gb = state_mix.log_like[0].argmax()
            """coords_out_gb_fixed = state_mix.branches_coords["gb_fixed"][0, best_gb]
            coords_in = transform_fn.both_transforms(coords_out_gb_fixed[state_mix.branches["gb_fixed"].inds[0, best_gb]])
            ntemps_pe, nwalkers_pe, nleaves_max, ndim = state_mix.branches["gb_fixed"].shape
            N_vals_here = xp.asarray(state_mix.branches_supplimental["gb_fixed"].holder['N_vals'][0, best_gb][state_mix.branches["gb_fixed"].inds[0, best_gb]])
            # NEEDS TO BE +1
            factors = +xp.ones(coords_out_gb_fixed.shape[0], dtype=xp.float64)
            data_index = xp.zeros(coords_out_gb_fixed.shape[0], dtype=xp.int32)
            templates_out = [[xp.zeros((fd.shape[0],), dtype=complex).flatten()], [xp.zeros((fd.shape[0],), dtype=complex).flatten()]]
            A_out, E_out = templates_out[0][0], templates_out[1][0]
            data_splits = [np.arange(1)]

            main_gpu = xp.cuda.runtime.getDevice()
            waveform_kwargs_fill = waveform_kwargs.copy()
            waveform_kwargs_fill.pop("N")

            gb.generate_global_template(coords_in, data_index, templates_out, data_length=data_length, factors=factors, data_splits=data_splits, N=N_vals_here, **waveform_kwargs_fill)

            xp.cuda.runtime.setDevice(main_gpu)
            np.save("best_logl_gbs_from_psd_run", xp.array([A_out, E_out]).get())
            del A_out, E_out
            xp.get_default_memory_pool().free_all_blocks()
            """
            xp.get_default_memory_pool().free_all_blocks()
            
            imported = False
            while not imported:
                try:
                    psds = np.load("best_logl_psd_from_psd_run" + ".npy")
                    imported = True
                except ValueError:
                    time.sleep(1)

            psds[:, 0] = psds[:, 1]
            A_psd_in[:] = psds[0][None, None, :]  # A
            E_psd_in[:] = psds[1][None, None, :]  # E
            xp.get_default_memory_pool().free_all_blocks()
            mgh.set_psd_from_arrays(
                A_psd_in.reshape(-1, psds.shape[-1]),
                E_psd_in.reshape(-1, psds.shape[-1]),
                overall_inds=mgh.map
            )
            xp.get_default_memory_pool().free_all_blocks()
            
            A_mbh_remove = last_mbh_template[0]
            E_mbh_remove = last_mbh_template[1]
            mgh.add_templates_from_arrays_to_residuals(
                -1 * A_mbh_remove.reshape(-1, psds.shape[-1]),
                -1 * E_mbh_remove.reshape(-1, psds.shape[-1]),
                overall_inds=mgh.map
            )

            A_out = -(mgh.data_shaped[0][0][best_gb].get() - A_inj)
            E_out = -(mgh.data_shaped[1][0][best_gb].get() - E_inj)

            A_out[np.abs(A_out) < 1e-27] = 0.0
            E_out[np.abs(E_out) < 1e-27] = 0.0
            np.save("best_logl_gbs_from_psd_run", xp.array([A_out, E_out]))

            # ll_af = mgh.get_ll(include_psd_info=True)
            imported = False
            while not imported:
                try:
                    mbh_inj = np.load("best_logl_mbhs_from_psd_run" + ".npy")
                    imported = True
                except ValueError:
                    time.sleep(1)

            A_mbh_going_in = np.zeros_like(last_mbh_template[0])
            E_mbh_going_in = np.zeros_like(last_mbh_template[1])

            A_mbh_going_in[:] = mbh_inj[0][None, None, :]  # A
            E_mbh_going_in[:] = mbh_inj[1][None, None, :]  # A
            xp.get_default_memory_pool().free_all_blocks()
            # TODO: need to check that everything is aligned
            ll_bef = mgh.get_ll(include_psd_info=True)
 
            mgh.add_templates_from_arrays_to_residuals(
                A_mbh_going_in.reshape(-1, psds.shape[-1]),
                E_mbh_going_in.reshape(-1, psds.shape[-1]),
                overall_inds=mgh.map
            )
            # ll_af = mgh.get_ll(include_psd_info=True)

            last_mbh_template = [A_mbh_going_in, E_mbh_going_in]

            ll = mgh.get_ll(include_psd_info=True)

            state_mix.log_like = ll.flatten()[mgh.map].reshape(ll.shape)
            xp.get_default_memory_pool().free_all_blocks()

            with open(fp_gb_mixing, "wb") as f_tmp:
                pickle.dump(state_mix, f_tmp, pickle.HIGHEST_PROTOCOL)

        if converged:
            break

    print(f"BEST LOGL AFTER MIX: {state_mix.log_like.max()}")
    best_logl_ind_after_mix = np.argmax(state_mix.log_like.flatten())    
    best_mapped_ind = mgh.get_mapped_indices(np.array([best_logl_ind_after_mix]))[0]
    A_best_residuals = mgh.data_shaped[0][0][best_mapped_ind].get()
    E_best_residuals = mgh.data_shaped[1][0][best_mapped_ind].get()

    A_best_psd = mgh.psd_shaped[0][0][best_mapped_ind].get()
    E_best_psd = mgh.psd_shaped[1][0][best_mapped_ind].get()

    gb.gpus = None
    mempool.free_all_blocks()

    removals = []
    

    # TODO: read out residuals for GBs

    del mgh.data_list[0][0]
    del mgh.data_list[1][0]
    del mgh.psd_list[0][0]
    del mgh.psd_list[1][0]
    del mgh
    mempool.free_all_blocks()
    data_in = [xp.asarray(A_best_residuals), xp.asarray(E_best_residuals)]
    psd_in = [xp.asarray(A_best_psd), xp.asarray(E_best_psd)]
    return [data_in, psd_in]


def run_gb_bulk_search():
    gpus = [6]
    xp.cuda.runtime.setDevice(gpus[0])
    # max ll combination of psd and mbhs
    # TODO: adjust this to max ll !!!!
    A_going_in = np.asarray(A_inj).copy()
    E_going_in = np.asarray(E_inj).copy()

    mbh_inj = np.load("best_logl_mbhs_from_psd_run" + ".npy")

    A_mbh_going_in = mbh_inj[0].squeeze()
    E_mbh_going_in = mbh_inj[1].squeeze()

    A_going_in[:] -= A_mbh_going_in
    E_going_in[:] -= E_mbh_going_in

    psds = np.load("best_logl_psd_from_psd_run" + ".npy")
    psds[:, 0] = psds[:, 1]
    A_psd_in = psds[0].squeeze() # A
    E_psd_in = psds[1].squeeze()  # E

    data_in = [xp.asarray(A_going_in), xp.asarray(E_going_in)]
    psd_in = [xp.asarray(A_psd_in), xp.asarray(E_psd_in)]
    
    fp_gb_mixing = "mixing_gb_run.pickle"

    if fp_gb_mixing in os.listdir():
        with open(fp_gb_mixing, "rb") as f_tmp:
            last_sample = pickle.load(f_tmp)

        best_log_ind = np.where(last_sample.log_like == last_sample.log_like.max())
        coords_inj = last_sample.branches_coords["gb_fixed"][best_log_ind][0]

        coords_inj_in = transform_fn.both_transforms(coords_inj)
        factors = -xp.ones(coords_inj_in.shape[0])
        data_index = xp.zeros(coords_inj_in.shape[0], dtype=np.int32)
        data_tmp_in = xp.asarray([data_in])
        gb.generate_global_template(coords_inj_in, data_index, data_tmp_in, data_length=len(A_going_in), factors=factors, **waveform_kwargs)
        data_in[0] = data_tmp_in[0, 0].copy()
        data_in[1] = data_tmp_in[0, 1].copy()
        del data_tmp_in, data_index, factors
        mempool.free_all_blocks()
        num_previously_found = coords_inj_in.shape[0]
        binaries_found = coords_inj
        run_mix_first = True

    else:
        num_previously_found = 0
        binaries_found = None
        run_mix_first = False

    iters = 100
    ntemps = 10
    nwalkers = 100
    ndim = 8

    band_inds_running = np.ones_like(search_f_bin_lims[:-1], dtype=bool)

    priors_here = deepcopy(default_priors_gb)

    priors_here[1] = uniform_dist(0.0, 1.0, use_cupy=True)
    priors_here[2] = uniform_dist(0.0, 1.0, use_cupy=True) 

    priors_good = ProbDistContainer(priors_here, use_cupy=True)
    
    fdot_mins = xp.asarray(get_fdot(search_f_bin_lims, Mc=np.full_like(search_f_bin_lims, m_chirp_lims[0])))
    fdot_maxs = xp.asarray(get_fdot(search_f_bin_lims, Mc=np.full_like(search_f_bin_lims, m_chirp_lims[1])))
    f0_mins = xp.asarray(search_f_bin_lims[:-1] * 1e3)
    f0_maxs = xp.asarray(search_f_bin_lims[1:] * 1e3)

    # do not run the last band
    band_inds_running[-1] = False
    num_binaries_total = num_previously_found
    for iter_i in range(iters):
        for evens_odds in range(2):
            starting_points = run_iterative_subtraction_mcmc(iter_i, ndim, nwalkers, ntemps, band_inds_running, evens_odds, priors_good, f0_maxs, f0_mins, fdot_maxs, fdot_mins, data_in, psd_in, binaries_found)
            num_binaries_found_this_iteration = starting_points.shape[2]
            num_binaries_total += num_binaries_found_this_iteration
            print(iter_i, f"Number of bands running: {band_inds_running.sum().item()}, found {num_binaries_found_this_iteration} binaries. Total binaries: {num_binaries_total}")

            data_in, psd_in = run_gb_mixing(iter_i, gpus, fp_gb_mixing, num_binaries_found_this_iteration, starting_points)

if __name__ == "__main__":
    run_gb_bulk_search()