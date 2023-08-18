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
from gather_gb_samples import gather_gb_samples

from sklearn.mixture import GaussianMixture

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

def fit_gmm(samples, comm, comm_info):

    keep = np.array([0, 1, 2, 4, 6, 7])

    if samples.ndim == 4:
        num_keep, num_samp, nwalkers_keep, ndim = samples.shape

        args = []
        for band in range(num_keep): 
            args.append((band, samples[band].reshape(-1, ndim),))

    elif samples.ndim == 2:
        max_groups = samples[:, 0].astype(int).max()

        args = []
        for group in range(max_groups): 
            keep_samp = samples[:, 0].astype(int) == group
            if keep_samp.sum() > 0:
                args.append(samples[keep_samp, 2:][:, keep])

    else:
        raise ValueError
    # import multiprocessing as mp

    # with mp.Pool(10) as pool:
    #     gmm_info = pool.starmap(fit_each_leaf, args)
    
    gmm_info = [None for tmp in args]

    process_ranks_for_fit = comm_info["process_ranks_for_fit"]

    # OPPOSITE
    # send_tags = comm_info["rec_tags"]
    # rec_tags = comm_info["send_tags"]

    num_args_complete = 0
    current_send_arg_index = 0
    current_status = [False for _ in process_ranks_for_fit]
    while num_args_complete < len(args):
        clear_out = True
        while clear_out:
            check_output = comm.irecv()

            if not check_output.get_status():
                check_output.cancel()
                clear_out = False
            else:
                output_info = check_output.wait()
                if isinstance(output_info, str):
                    breakpoint()
                # print(output_info)
                arg_index = output_info["arg"]
                rank_recv = output_info["rank"]
                output_list = output_info["output"]

                gmm_info[arg_index] = output_list

                which_process = process_ranks_for_fit.index(rank_recv)

                current_status[which_process] = False
                num_args_complete += 1

        add_info = not np.all(current_status)
        while add_info:
            if current_send_arg_index == len(args):
                add_info = False
            else:
                index_add = current_status.index(False)

                send_info = {"samples": args[current_send_arg_index], "arg": current_send_arg_index}
                # print("sending", process_ranks_for_fit[index_add])
                comm.isend(send_info, dest=process_ranks_for_fit[index_add])
                current_status[index_add] = True
                current_send_arg_index += 1
                add_info = not np.all(current_status)

                # print(current_status, current_send_arg_index, num_args_complete)

    breakpoint()
    for rank in process_ranks_for_fit:
        comm.isend("end", dest=rank)

    weights = [tmp[0] for tmp in gmm_info]
    means = [tmp[1] for tmp in gmm_info]
    covs = [tmp[2] for tmp in gmm_info]
    invcovs = [tmp[3] for tmp in gmm_info]
    dets = [tmp[4] for tmp in gmm_info]
    mins = [tmp[5] for tmp in gmm_info]
    maxs = [tmp[6] for tmp in gmm_info]
    
    output = [weights, means, covs, invcovs, dets, mins, maxs]

    return output


def fit_each_leaf(rank, gather_rank, rec_tag, send_tag, comm):
    print("process", rank, gather_rank)

    run_process = True

    while run_process:
        try:
            check = comm.recv(source=gather_rank)  # , tag=tag)
        except:
            print("BAD BAD ", rank)
            comm.isend("BAD", dest=gather_rank)

        if isinstance(check, str):
            if check == "end":
                run_process = False
                continue

        assert isinstance(check, dict)

        arg_index = check["arg"]

        # print("INSIDE", rank, arg_index)
        samples = check["samples"]

        assert isinstance(samples, np.ndarray)

        run = True
        min_bic = np.inf
        sample_mins = samples.min(axis=0)
        sample_maxs = samples.max(axis=0)
        samples[:] = ((samples - sample_mins) / (sample_maxs - sample_mins)) * 2 - 1
        for n_components in range(1, 20):
            if not run:
                continue
            #fit_gaussian_mixture_model(n_components, samples)
            #breakpoint()
            try:
                mixture = GaussianMixture(n_components=n_components, verbose=False, verbose_interval=2)

                mixture.fit(samples)
                test_bic = mixture.bic(samples)
            except ValueError:
                breakpoint()
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

        if keep_components >= 19:
            print(keep_components)
        output_list = [keep_mix.weights_, keep_mix.means_, keep_mix.covariances_, np.array([np.linalg.inv(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), np.array([np.linalg.det(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), sample_mins, sample_maxs]
        comm.isend({"output": output_list, "rank": rank, "arg": arg_index}, dest=gather_rank)
    return

def run_iterative_subtraction_mcmc(iter_i, ndim, nwalkers, ntemps, band_inds_running, evens_odds, priors_good, f0_maxs, f0_mins, fdot_maxs, fdot_mins, data_in, psd_in, binaries_found, comm):
    xp.cuda.runtime.setDevice(xp.cuda.runtime.getDevice())
    temperature_control = TemperatureControl(ndim, nwalkers, ntemps=ntemps)

    num_max_proposals = 100000
    convergence_iter_count = 50

    move_proposal = StretchMove(periodic=PeriodicContainer(periodic), temperature_control=temperature_control, return_gpu=True, use_gpu=True)
    
    band_inds_here = xp.where(xp.asarray(band_inds_running) & (xp.arange(len(band_inds_running)) % 1 == evens_odds))[0]  #  % 2 == evens_odds))[0]

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
    inner_product = 4 * df * (xp.sum(data_in[0].conj() * data_in[0] / psd_in[0]) + xp.sum(data_in[1].conj() * data_in[1] / psd_in[1])).real
    ll = (-1/2 * inner_product - xp.sum(xp.log(xp.asarray(psd_in)))).item()
    gb.d_d = ll

    print(ll)

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

    run_number = 0
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

        if prop_i > convergence_iter_count:
            iter_count[improvement] = 0
            iter_count[~improvement] += 1

        num_proposals_per[still_going_here] += 1
        still_going_here[iter_count >= convergence_iter_count] = False
        
        if prop_i % convergence_iter_count == 0:
            print(f"Proposal {prop_i}, Still going:", still_going_here.sum().item())
        if run_number == 2:
            iter_count[:] = 0
            collect_sample_check_iter += 1
            if collect_sample_check_iter % thin_by == 0:
                coords_with_fs = old_points.transpose(2, 0, 1, 3)[still_going_here, 0, :].copy()
                coords_with_fs[:, :, 1] = (f0_maxs[band_inds_here[still_going_here]] - f0_mins[band_inds_here][still_going_here])[:, None] * coords_with_fs[:, :, 1] + f0_mins[band_inds_here[still_going_here]][:, None]
                coords_with_fs[:, :, 2] = (fdot_maxs[band_inds_here[still_going_here]] - fdot_mins[band_inds_here][still_going_here])[:, None] * coords_with_fs[:, :, 2] + fdot_mins[band_inds_here[still_going_here]][:, None]

                samples_store[:, collect_sample_iter] = coords_with_fs.get()
                collect_sample_iter += 1
                print(collect_sample_iter, num_samples_store)
                if collect_sample_iter == num_samples_store:
                    still_going_here[:] = False

        if still_going_here.sum().item() == 0:
            if run_number < 2:
                betas = xp.repeat(xp.asarray(temperature_control.betas[:, None].copy()), len(band_inds_here), axis=-1)
                
                old_points[:] = best_logl_coords[None, None, :]

                gen_points = old_points.transpose(2, 0, 1, 3).reshape(best_logl_coords.shape[0], -1, ndim).copy()
                iter_count[:] = 0
                still_going_here[:] = True

                factor = 1e-5
                cov = xp.ones(ndim) * 1e-3
                cov[1] = 1e-8

                still_going_start_like = xp.ones(best_logl_coords.shape[0], dtype=bool)
                starting_points = np.zeros((best_logl_coords.shape[0], nwalkers * ntemps, ndim))

                iter_check = 0
                max_iter = 10000
                while np.any(still_going_start_like):
                    num_still_going_start_like = still_going_start_like.sum().item()
                    
                    start_like = np.zeros((num_still_going_start_like, nwalkers * ntemps))
                
                    logp = np.full_like(start_like, -np.inf)
                    tmp = xp.zeros((num_still_going_start_like, ntemps * nwalkers, ndim))
                    fix = xp.ones((num_still_going_start_like, ntemps * nwalkers), dtype=bool)
                    while xp.any(fix):
                        tmp[fix] = (gen_points[still_going_start_like, :] * (1. + factor * cov * xp.random.randn(num_still_going_start_like, nwalkers * ntemps, ndim)))[fix]

                        tmp[:, :, 3] = tmp[:, :, 3] % (2 * np.pi)
                        tmp[:, :, 5] = tmp[:, :, 5] % (np.pi)
                        tmp[:, :, 6] = tmp[:, :, 6] % (2 * np.pi)
                        logp = priors_good.logpdf(tmp.reshape(-1, ndim)).reshape(tmp.shape[:-1])

                        fix = xp.isinf(logp)
                        if xp.all(fix):
                            breakpoint()

                    new_points_with_fs = tmp.copy()

                    new_points_with_fs[:, :, 1] = (f0_maxs[None, band_inds_here[still_going_start_like]] - f0_mins[None, band_inds_here[still_going_start_like]]).T * new_points_with_fs[:, :, 1] + f0_mins[None, band_inds_here[still_going_start_like]].T
                    new_points_with_fs[:, :, 2] = (fdot_maxs[None, band_inds_here[still_going_start_like]] - fdot_mins[None, band_inds_here[still_going_start_like]]).T * new_points_with_fs[:, :, 2] + fdot_mins[None, band_inds_here[still_going_start_like]].T

                    new_points_in = transform_fn.both_transforms(new_points_with_fs.reshape(-1, ndim), xp=xp)

                    start_like = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs)).reshape(new_points_with_fs.shape[:-1])

                    old_points[:, :, still_going_start_like, :] = tmp.transpose(1, 0, 2).reshape(ntemps, nwalkers, -1, ndim)
                    prev_logl[:, :, still_going_start_like] = start_like.T.reshape(ntemps, nwalkers, -1)
                    prev_logp[:, :, still_going_start_like] = logp.T.reshape(ntemps, nwalkers, -1)
                    # fix any nans that may come up
                    start_like[xp.isnan(start_like)] = -1e300
                    
                    update = xp.arange(still_going_start_like.shape[0])[still_going_start_like][xp.std(start_like, axis=-1) > 5.0]
                    still_going_start_like[update] = False 

                    iter_check += 1
                    factor *= 1.5
                    print(iter_check, still_going_start_like.sum())

                if run_number == 1:
                    best_binaries_coords_with_fs = best_logl_coords.copy()

                    best_binaries_coords_with_fs[:, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 1] + f0_mins[band_inds_here]
                    best_binaries_coords_with_fs[:, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 2] + fdot_mins[band_inds_here]

                    best_logl_points_in = transform_fn.both_transforms(best_binaries_coords_with_fs, xp=xp)

                    best_logl_check = xp.asarray(gb.get_ll(best_logl_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

                    if not xp.allclose(best_logl, best_logl_check):
                        breakpoint()

                    snr_lim = 12.0
                    keep_binaries = gb.d_h / xp.sqrt(gb.h_h.real) > snr_lim

                    still_going_here = keep_binaries.copy()

                    thin_by = 25
                    num_samples_store = 30
                    samples_store = np.zeros((still_going_here.sum().item(), num_samples_store, nwalkers, ndim))
                    collect_sample_iter = 0
                    collect_sample_check_iter = 0

                    # # TODO: add in based on sensitivity changing
                    # # band_inds_running[band_inds_here[~keep_binaries].get()] = False
                    # keep_coords = best_binaries_coords_with_fs[keep_binaries].get()

                    # # adjust the phase from marginalization
                    # phase_change = np.angle(gb.non_marg_d_h)[keep_binaries.get()]
                    # keep_coords[:, 3] -= phase_change
                    # # best_logl_points_in[keep_binaries, 4] -= xp.asarray(phase_change)

                    # # check if there are sources near band edges that are overlapping
                    # assert np.all(keep_coords[:, 1] == np.sort(keep_coords[:, 1]))
                    # f_found = keep_coords[:, 1] / 1e3
                    # N = get_N(np.full_like(f_found, 1e-30), f_found, Tobs=waveform_kwargs["T"], oversample=waveform_kwargs["oversample"])
                    # inds_check = np.where((np.diff(f_found) / df).astype(int) < N[:-1])[0]

                    # params_add = keep_coords[inds_check]
                    # params_remove = keep_coords[inds_check + 1]
                    # N_check = N[inds_check]

                    # params_add_in = transform_fn.both_transforms(params_add)
                    # params_remove_in = transform_fn.both_transforms(params_remove)

                    # waveform_kwargs_tmp = waveform_kwargs.copy()
                    # if "N" in waveform_kwargs_tmp:
                    #     waveform_kwargs_tmp.pop("N")
                    # waveform_kwargs_tmp["use_c_implementation"] = False

                    # gb.swap_likelihood_difference(params_add_in, params_remove_in, data_in, psd_in, N=256, **waveform_kwargs_tmp)

                    # likelihood_difference = -1/2 * (gb.add_add + gb.remove_remove - 2 * gb.add_remove).real.get()
                    # overlap = (gb.add_remove.real / np.sqrt(gb.add_add.real * gb.remove_remove.real)).get()

                    # fix = np.where((likelihood_difference > -100.0) | (overlap > 0.4))

                    # if np.any(fix):
                    #     params_comp_add = params_add[fix]
                    #     params_comp_remove = params_remove[fix]

                    #     # not actually in the data yet, just using swap for quick likelihood comp
                    #     snr_add = (gb.d_h_add.real[fix] / gb.add_add.real[fix] ** (1/2)).get()
                    #     snr_remove = (gb.d_h_remove.real[fix] / gb.remove_remove.real[fix] ** (1/2)).get()

                    #     inds_add = inds_check[fix]
                    #     inds_remove = inds_add + 1

                    #     inds_delete = (inds_add) * (snr_add < snr_remove) + (inds_remove) * (snr_remove < snr_add)
                    #     keep_coords = np.delete(keep_coords, inds_delete, axis=0)
                        

                run_number += 1

            else:
                break

    return fit_gmm(samples_store)

    # import pickle
    # with open(f"new_4_gmm_info.pickle", "wb") as fp:
    #     pickle.dump(output, fp, pickle.HIGHEST_PROTOCOL)
    
    
    
    # nwalkers_pe = nwalkers
    # ntemps_pe = 1
    
    # factor = 1e-5
    # cov = np.ones(ndim) * 1e-3
    # cov[1] = 1e-8

    # still_going_start_like = np.ones(keep_coords.shape[0], dtype=bool)
    # starting_points = np.zeros((keep_coords.shape[0], nwalkers_pe * ntemps_pe, ndim))
    # iter_check = 0
    # max_iter = 10000
    # while np.any(still_going_start_like):
    #     num_still_going_start_like = still_going_start_like.sum().item()
        
    #     start_like = np.zeros((num_still_going_start_like, nwalkers_pe * ntemps_pe))
    
    #     logp = np.full_like(start_like, -np.inf)
    #     tmp = np.zeros((num_still_going_start_like, ntemps_pe * nwalkers_pe, ndim))
    #     fix = np.ones((num_still_going_start_like, ntemps_pe * nwalkers_pe), dtype=bool)
    #     while np.any(fix):
    #         tmp[fix] = (keep_coords[still_going_start_like, None, :] * (1. + factor * cov * np.random.randn(num_still_going_start_like, nwalkers_pe * ntemps_pe, ndim)))[fix]

    #         tmp[:, :, 3] = tmp[:, :, 3] % (2 * np.pi)
    #         tmp[:, :, 5] = tmp[:, :, 5] % (np.pi)
    #         tmp[:, :, 6] = tmp[:, :, 6] % (2 * np.pi)
    #         logp = priors["gb_fixed"].logpdf(tmp.reshape(-1, ndim)).reshape(tmp.shape[:-1])

    #         fix = np.isinf(logp)
    #         if np.all(fix):
    #             breakpoint()

    #     tmp_in = transform_fn.both_transforms(tmp.reshape(-1, ndim))
    #     start_like = gb.get_ll(tmp_in, data_in, psd_in, phase_marginalize=False, **waveform_kwargs).reshape(tmp.shape[:-1])

    #     starting_points[still_going_start_like] = tmp
        
    #     update = np.arange(still_going_start_like.shape[0])[still_going_start_like][np.std(start_like, axis=-1) > 5.0]
    #     still_going_start_like[update] = False 

    #     iter_check += 1
    #     factor *= 1.5

    #     # print(np.std(start_like))

    #     if iter_check > max_iter:
    #         raise ValueError("Unable to find starting parameters.")

    # starting_points = starting_points.reshape((keep_coords.shape[0], ntemps_pe, nwalkers_pe, ndim)).transpose(1, 2, 0, 3)

    # if binaries_found is None:
    #     binaries_found = keep_coords
    # else:
    #     binaries_found = np.concatenate([binaries_found, keep_coords], axis=0)

    # binaries_found = binaries_found[np.argsort(binaries_found[:, 1])]

    # num_binaries_found_this_iteration = keep_binaries.sum().item()

    # np.save("starting_points_last_batch", starting_points)
    # return starting_points


def refit_gmm(comm, comm_info, gb_reader, data, psd, number_samples_keep):
    print("GATHER")
    samples_gathered = gather_gb_samples(gb_reader, psd, xp.cuda.runtime.getDevice(), samples_keep=number_samples_keep, thin_by=20)
    
    return fit_gmm(samples_gathered, comm, comm_info)

def run_gb_bulk_search(gpu, comm, comm_info, head_rank):
    gpus = [gpu]
    xp.cuda.runtime.setDevice(gpus[0])

    ntemps = 10
    nwalkers = 100
    ndim = 8

    gf_information = comm.recv(source=head_rank, tag=25)
    base_information = gf_information["general_global_info"]
    band_edges = base_information["band_edges"]

    band_inds_running = np.ones_like(band_edges[:-1], dtype=bool)
    
    priors_here = deepcopy(base_information["priors"]["gb"])

    priors_here[1] = uniform_dist(0.0, 1.0, use_cupy=True)
    priors_here[2] = uniform_dist(0.0, 1.0, use_cupy=True) 

    priors_good = ProbDistContainer(priors_here, use_cupy=True)
    
    fdot_mins = xp.asarray(get_fdot(band_edges, Mc=np.full_like(band_edges, m_chirp_lims[0])))
    fdot_maxs = xp.asarray(get_fdot(band_edges, Mc=np.full_like(band_edges, m_chirp_lims[1])))
    f0_mins = xp.asarray(band_edges[:-1] * 1e3)
    f0_maxs = xp.asarray(band_edges[1:] * 1e3)

    # do not run the last band
    band_inds_running[-1] = False

    print("start run")
    run = True
    while run:
        comm.send({"send": True}, dest=head_rank, tag=20)
        print("waiting for data")
        incoming_data = comm.recv(source=head_rank, tag=27)
        print("received data")
        print(incoming_data.keys())

        generate_class = incoming_data["general_global_info"]["generate_current_state"]

        generated_info = generate_class(incoming_data, only_max_ll=True)
        data = generated_info["data"]
        psd = generated_info["psd"]

        data = [xp.asarray(tmp) for tmp in data]
        psd = [xp.asarray(tmp) for tmp in psd]

        # max ll combination of psd and mbhs and gbs
        
        if incoming_data["gb"]["reader"].iteration > 100:
            gmm_samples_refit = refit_gmm(comm, comm_info, incoming_data["gb"]["reader"], data, psd, 100)

        else:
            gmm_samples_refit = None

        gmm_mcmc_serch_info = run_iterative_subtraction_mcmc(iter_i, ndim, nwalkers, ntemps, band_inds_running, evens_odds, priors_good, f0_maxs, f0_mins, fdot_maxs, fdot_mins, data_in, psd_in, binaries_found, comm)
        
        comm.isend({"search": gmm_mcmc_serch_info, "sample_refit": gmm_samples_refit}, dest=head_rank, tag=20)
        






        
            # refit GMM


            # evens_odds = 0
            # if not "starting_points_last_batch.npy" in os.listdir():
                
            # else:
            #     starting_points = np.load("starting_points_last_batch.npy")
            #     starting_points = np.full((1, 100, 0, 0), np.array([]))

            # num_binaries_found_this_iteration = starting_points.shape[2]
            # num_binaries_total += num_binaries_found_this_iteration

            # # starting_points = None
            # # num_binaries_found_this_iteration = 0
            # print(iter_i, f"Number of bands running: {band_inds_running.sum().item()}, found {num_binaries_found_this_iteration} binaries. Total binaries: {num_binaries_total}")

            # data_in, psd_in = run_gb_mixing(iter_i, gpus, fp_gb_mixing, num_binaries_found_this_iteration, starting_points)

if __name__ == "__main__":
    run_gb_bulk_search()