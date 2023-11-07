# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer
from eryn.state import State


__all__ = ["GBSpecialStretchMove"]

# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBForegroundSpecialMove(StretchMove):
    """
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
        *args,
        waveform_kwargs={},
        noise_kwargs={},
        parameter_transforms=None,
        search=False,
        search_samples=None,
        search_snrs=None,
        search_snr_lim=None,
        search_snr_accept_factor=1.0,
        take_max_ll=False,
        global_template_builder=None,
        psd_func=None,
        provide_betas=False,
        alternate_priors=None,
        batch_size=5,
        **kwargs
    ):
        StretchMove.__init__(self, **kwargs)  

        self.time = 0
        self.greater_than_1e0 = 0
        self.name = "GBForegroundSpecialMove".lower()

        # TODO: make priors optional like special generate function? 
        for key in priors:
            if not isinstance(priors[key], ProbDistContainer):
                raise ValueError("Priors need to be eryn.priors.ProbDistContainer object.")
        self.priors = priors
        self.gb = gb
        self.provide_betas = provide_betas
        self.batch_size = batch_size
        self.stop_here = True

        # use gpu from template generator
        self.use_gpu = gb.use_gpu
        if self.use_gpu:
            self.xp = xp
            self.mempool = self.xp.get_default_memory_pool()

        else:
            self.xp = np


        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.noise_kwargs = noise_kwargs
        self.parameter_transforms = parameter_transforms
        self.psd_func = psd_func
        self.fd = fd
        self.df = (fd[1] - fd[0]).item()
        self.mgh = mgh
        self.search = search
        self.global_template_builder = global_template_builder

        if search_snrs is not None:
            if search_snr_lim is None:
                search_snr_lim = 0.1

            assert len(search_samples) == len(search_snrs)

        self.search_samples = search_samples
        self.search_snrs = search_snrs
        self.search_snr_lim = search_snr_lim
        self.search_snr_accept_factor = search_snr_accept_factor

        self.take_max_ll = take_max_ll     

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        # st = time.perf_counter()
        self.xp.cuda.runtime.setDevice(self.mgh.gpus[0])

        # np.random.seed(10)
        #print("start stretch")
        #st = time.perf_counter()
        # Check that the dimensions are compatible.
        ntemps, nwalkers, nleaves, ndim = state.branches["galfor"].shape

        self.nwalkers = nwalkers
        # TODO: deal with more intensive acceptance fractions
        # Run any move-specific setup.
        self.setup(state.branches)

        new_state = State(state)  # , copy=True)
        self.mempool.free_all_blocks()
        
        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        # data should not be whitened
        
        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb"].shape

        split_inds = np.zeros(nwalkers, dtype=int)
        split_inds[1::2] = 1
        np.random.shuffle(split_inds)

        current_coords_galfor = new_state.branches["galfor"].coords.copy()
        """et = time.perf_counter()
        print("setup", (et - st))
        st = time.perf_counter()"""
        for split in range(2):
            # st = time.perf_counter()
            split_here = split_inds == split
            walkers_keep = self.xp.arange(nwalkers)[split_here]
                
            points_to_move = {key: new_state.branches[key].coords[:, split_here] for key in ["psd", "galfor"]}
            points_for_move = {key: [new_state.branches[key].coords[:, ~split_here]] for key in ["psd", "galfor"]}
            
            q, factors = self.get_proposal(
                points_to_move,  
                points_for_move, 
                model.random
            )

            temp_part_general = np.repeat(np.arange(ntemps)[:, None], nwalkers, axis=-1)[:, split_here].flatten()
            walker_part_general = np.tile(np.arange(nwalkers), (ntemps, 1))[:, split_here].flatten()

            # get logp
            logp_here = (self.priors["psd"].logpdf(q["psd"].reshape(-1, q["psd"].shape[-1])) + self.priors["galfor"].logpdf(q["galfor"].reshape(-1, q["galfor"].shape[-1]))).reshape((ntemps, int(nwalkers / 2)))

            prev_logp_here = (self.priors["psd"].logpdf(new_state.branches_coords["psd"][:, split_here].reshape(-1, q["psd"].shape[-1])) + self.priors["galfor"].logpdf(new_state.branches_coords["galfor"][:, split_here].reshape(-1, q["galfor"].shape[-1]))).reshape((ntemps, int(nwalkers / 2)))

            bad = np.isinf(logp_here.flatten())

            data_index_tmp = np.asarray((temp_part_general * self.nwalkers + walker_part_general).astype(xp.int32))

            data_index = self.mgh.get_mapped_indices(data_index_tmp)

            data_index_in = data_index[~bad]

            psd_params = q["psd"].reshape(-1, q["psd"].shape[-1])[~bad]
            foreground_params = q["galfor"].reshape(-1, q["galfor"].shape[-1])[~bad]
    
            self.mgh.set_psd_vals(psd_params, foreground_params=foreground_params, overall_inds=data_index_in)

            logl_temp = self.mgh.get_ll(include_psd_info=True, overall_inds=data_index_in)

            logl = np.full((ntemps, int(nwalkers / 2)), -1e300)
            logl[~bad.reshape(ntemps, -1)] = logl_temp

            prev_logl = new_state.log_like[:, split_here]
            prev_logp = new_state.log_prior[:, split_here]

            logp = prev_logp + logp_here - prev_logp_here

            logP = self.compute_log_posterior(logl, logp)
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            lnpdiff = factors + logP - prev_logP

            keep = lnpdiff > np.log(model.random.rand(ntemps, int(nwalkers / 2)))

            temp_inds_keep = temp_part_general[keep.flatten()]
            walker_inds_keep = walker_part_general[keep.flatten()]

            accepted[temp_inds_keep, walker_inds_keep] = True

            new_state.log_like[temp_inds_keep, walker_inds_keep] = logl[keep]
            new_state.log_prior[temp_inds_keep, walker_inds_keep] = logp[keep]

            for key in ["psd", "galfor"]:
                new_state.branches[key].coords[temp_inds_keep, walker_inds_keep, np.zeros_like(walker_inds_keep)] = q[key][keep][:, 0]

            temp_inds_fix = temp_part_general[~keep.flatten()]
            walker_inds_fix = walker_part_general[~keep.flatten()]

            # return unaccepted psds
            data_index_fix = data_index[~keep.flatten()]
            psd_params_fix = new_state.branches_coords["psd"][(temp_inds_fix, walker_inds_fix)][:, 0]
            foreground_params_fix = new_state.branches_coords["galfor"][(temp_inds_fix, walker_inds_fix)][:, 0]

            self.mgh.set_psd_vals(psd_params_fix, foreground_params=foreground_params_fix, overall_inds=data_index_fix)

        self.accepted += accepted.astype(int)
        self.num_proposals += 1

        self.mempool.free_all_blocks()

        if self.time % 200 == 0:
            ll_after = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
            check = np.abs(new_state.log_like - ll_after).max()
            if check  > 1e-3:
                breakpoint()
                self.mgh.restore_base_injections()

                for name in new_state.branches.keys():
                    if name not in ["gb", "gb"]:
                        continue
                    new_state_branch = new_state.branches[name]
                    coords_here = new_state_branch.coords[new_state_branch.inds]
                    ntemps, nwalkers, nleaves_max_here, ndim = new_state_branch.shape
                    try:
                        group_index = self.xp.asarray(
                            self.mgh.get_mapped_indices(
                                np.repeat(np.arange(ntemps * nwalkers).reshape(ntemps, nwalkers, 1), nleaves_max, axis=-1)[new_state_branch.inds]
                            ).astype(self.xp.int32)
                        )
                    except IndexError:
                        breakpoint()
                    coords_here_in = self.parameter_transforms.both_transforms(coords_here, xp=np)

                    waveform_kwargs_fill = self.waveform_kwargs.copy()
                    waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

                    if "N" in waveform_kwargs_fill:
                        waveform_kwargs_fill.pop("N")
                        
                    self.mgh.multiply_data(-1.)
                    self.gb.generate_global_template(coords_here_in, group_index, self.mgh.data_list, data_length=self.data_length, data_splits=self.mgh.gpu_splits, batch_size=1000, **waveform_kwargs_fill)
                    self.mgh.multiply_data(-1.)

                ll_after2 = self.mgh.get_ll(use_cpu=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
                new_state.log_like = ll_after2
                   
            """
            data_minus_template = self.xp.concatenate(
                [
                    tmp.reshape(ntemps, nwalkers, 1, self.data_length) for tmp in data_minus_template_in_swap
                ],
                axis=2
            )
            del data_minus_template_in_swap

            psd = self.xp.concatenate(
                [
                    tmp.reshape(ntemps * nwalkers, 1, self.data_length) for tmp in psd_in_swap
                ],
                axis=1
            )
            del psd_in_swap
            self.mempool.free_all_blocks()

            new_state.supplimental.holder["data_minus_template"] = data_minus_template

            lp_after = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds)
            
            ll_after = (-1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))).get()  # model.compute_log_like_fn(new_state.branches_coords, inds=new_state.branches_inds, logp=lp_after, supps=new_state.supplimental, branch_supps=new_state.branches_supplimental)
            #check = -1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))
            #check2 = -1/2 * 4 * self.df * self.xp.sum(tmp.conj() * tmp / self.xp.asarray(self.psd), axis=(2, 3))
            #print(np.abs(new_state.log_like - ll_after[0]).max())

            # if any are even remotely getting to be different, reset all (small change)
            if np.abs(new_state.log_like - ll_after).max() > 1e-1:
                if np.abs(new_state.log_like - ll_after).max() > 1e0:
                    self.greater_than_1e0 += 1
                    print("Greater:", self.greater_than_1e0)
                breakpoint()
                fix_here = np.abs(new_state.log_like - ll_after) > 1e-6
                data_minus_template_old = data_minus_template.copy()
                data_minus_template = self.xp.zeros_like(data_minus_template_old)
                data_minus_template[:] = self.xp.asarray(self.data)[None, None]
                templates = self.xp.zeros_like(data_minus_template).reshape(-1, 2, data_minus_template.shape[-1])
                for name in new_state.branches.keys():
                    if name not in ["gb", "gb"]:
                        continue
                    new_state_branch = new_state.branches[name]
                    coords_here = new_state_branch.coords[new_state_branch.inds]
                    ntemps, nwalkers, nleaves_max_here, ndim = new_state_branch.shape
                    try:
                        group_index = np.repeat(np.arange(ntemps * nwalkers).reshape(ntemps, nwalkers, 1), nleaves_max, axis=-1)[new_state_branch.inds]
                    except IndexError:
                        breakpoint()
                    coords_here_in = self.parameter_transforms.both_transforms(coords_here, xp=np)

                    self.gb.generate_global_template(coords_here_in, group_index, templates, batch_size=1000, **self.waveform_kwargs)

                data_minus_template -= templates.reshape(ntemps, nwalkers, 2, templates.shape[-1])

                new_like = -1 / 2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / psd, axis=(2, 3)).real.get()
            
                new_like += self.noise_ll
                new_state.log_like[:] = new_like.reshape(ntemps, nwalkers)

            self.mempool.free_all_blocks()
            data_minus_template_in_swap = [data_minus_template[:,:, 0, :].flatten().copy(), data_minus_template[:,:, 1, :].flatten().copy()]
            del data_minus_template

            psd_in_swap = [psd[:, 0, :].flatten().copy(), psd[:, 1, :].flatten().copy()]
            self.mempool.free_all_blocks()
            del psd
            self.mempool.free_all_blocks()
            """
        
        self.mempool.free_all_blocks()

        if self.temperature_control is not None:
            new_state = self.temperature_control.temper_comps(new_state, adapt=False)
            
            """# new_state, accepted = self.temperature_control.temper_comps(new_state, accepted)
            self.swaps_accepted = np.zeros(ntemps - 1)
            self.attempted_swaps = np.zeros(ntemps - 1)
            betas = self.temperature_control.betas
            for i in range(ntemps - 1, 0, -1):
                bi = betas[i]
                bi1 = betas[i - 1]

                dbeta = bi1 - bi

                iperm = np.random.permutation(nwalkers)
                i1perm = np.random.permutation(nwalkers)

                # need to calculate switch likelihoods

                coords_iperm = new_state.branches["gb"].coords[i, iperm]
                coords_i1perm = new_state.branches["gb"].coords[i - 1, i1perm]

                N_vals_iperm = new_state.branches["gb"].branch_supplimental.holder["N_vals"][i, iperm]

                N_vals_i1perm = new_state.branches["gb"].branch_supplimental.holder["N_vals"][i - 1, i1perm]

                f_test_i = coords_iperm[None, :, :, 1] / 1e3
                f_test_2_i = coords_i1perm[None, :, :, 1] / 1e3
                
                fix_f_test_i = (np.abs(f_test_i - f_test_2_i) > (self.df * N_vals_iperm * 1.5))

                if hasattr(self, "keep_bands") and self.keep_bands is not None:
                    band_indices = np.searchsorted(self.band_edges, f_test_i.flatten()).reshape(f_test_i.shape) - 1
                    keep_bands = self.keep_bands
                    assert isinstance(keep_bands, np.ndarray)
                    fix_f_test_i[~np.in1d(band_indices, keep_bands).reshape(band_indices.shape)] = True


                groups = get_groups_from_band_structure(f_test_i, self.band_edges, f0_2=f_test_2_i, xp=np, num_groups_base=3, fix_f_test=fix_f_test_i)

                unique_groups, group_len = np.unique(groups.flatten(), return_counts=True)

                # remove information about the bad "-1" group
                for check_val in [-1, -2]:
                    group_len = np.delete(group_len, unique_groups == check_val)
                    unique_groups = np.delete(unique_groups, unique_groups == check_val)

                # needs to be max because some values may be missing due to evens and odds
                num_groups = unique_groups.max().item() + 1

                for group_iter in range(num_groups):
                    # st = time.perf_counter()
                    # sometimes you will have an extra odd or even group only
                    # the group_iter may not match the actual running group number in this case
                    if group_iter not in groups:
                        continue
                        
                    group = [grp[i:i+1][groups == group_iter].flatten() for grp in group_temp_finder]

                    # st = time.perf_counter()
                    temp_inds_back, walkers_inds_back, leaf_inds = [self.xp.asarray(grp) for grp in group] 

                    temp_inds_i = temp_inds_back.copy()
                    walkers_inds_i = walkers_inds_back.copy()

                    temp_inds_i1 = temp_inds_back.copy()
                    walkers_inds_i1 = walkers_inds_back.copy()

                    temp_inds_i[:] = i
                    walkers_inds_i[:] = self.xp.asarray(iperm)[walkers_inds_back]

                    temp_inds_i1[:] = i - 1
                    walkers_inds_i1[:] = self.xp.asarray(i1perm)[walkers_inds_back]

                    group_here_i = (temp_inds_i, walkers_inds_i, leaf_inds)

                    group_here_i1 = (temp_inds_i1, walkers_inds_i1, leaf_inds)

                    # factors_here = factors[group_here]
                    old_points = self.xp.asarray(new_state.branches["gb"].coords)[group_here_i]
                    new_points = self.xp.asarray(new_state.branches["gb"].coords)[group_here_i1]

                    N_vals_here_i = N_vals[group_here_i]
                    
                    log_like_tmp = self.xp.asarray(new_state.log_like.copy())
                    log_prior_tmp = self.xp.asarray(new_state.log_prior.copy())

                    delta_logl_i = self.run_swap_ll(None, old_points, new_points, group_here_i, N_vals_here_i, waveform_kwargs_now, None, log_like_tmp, log_prior_tmp, return_at_logl=True)

                    # factors_here = factors[group_here]
                    old_points[:] = self.xp.asarray(new_state.branches["gb"].coords)[group_here_i1]
                    new_points[:] = self.xp.asarray(new_state.branches["gb"].coords)[group_here_i]

                    N_vals_here_i1 = N_vals[group_here_i1]
                    
                    log_like_tmp[:] = self.xp.asarray(new_state.log_like.copy())
                    log_prior_tmp[:] = self.xp.asarray(new_state.log_prior.copy())

                    delta_logl_i1 = self.run_swap_ll(None, old_points, new_points, group_here_i1, N_vals_here_i1, waveform_kwargs_now, None, log_like_tmp, log_prior_tmp, return_at_logl=True)

                    paccept = dbeta * 1. / 2. * (delta_logl_i - delta_logl_i1)
                    raccept = np.log(np.random.uniform(size=paccept.shape[0]))

                    # How many swaps were accepted?
                    sel = paccept > self.xp.asarray(raccept)

                    inds_i_swap = tuple([tmp[sel].get() for tmp in list(group_here_i)])
                    inds_i1_swap = tuple([tmp[sel].get() for tmp in list(group_here_i1)])
                    
                    group_index_i = self.xp.asarray(
                        self.mgh.get_mapped_indices(
                            temp_inds_i[sel] + nwalkers * walkers_inds_i[sel]
                        )
                    ).astype(self.xp.int32)

                    group_index_i1 = self.xp.asarray(
                        self.mgh.get_mapped_indices(
                            temp_inds_i1[sel] + nwalkers * walkers_inds_i1[sel]
                        )
                    ).astype(self.xp.int32)

                    N_vals_i = N_vals[inds_i_swap]
                    params_i = self.xp.asarray(new_state.branches["gb"].coords)[inds_i_swap]
                    params_i1 = self.xp.asarray(new_state.branches["gb"].coords)[inds_i1_swap]

                    params_generate = self.xp.concatenate([
                        params_i,
                        params_i1,
                        params_i1,  # reverse of above
                        params_i,
                    ], axis=0)

                    params_generate_in = self.parameter_transforms.both_transforms(params_generate, xp=self.xp)

                    group_index_gen = self.xp.concatenate(
                        [
                            group_index_i,
                            group_index_i,
                            group_index_i1,
                            group_index_i1
                        ], dtype=self.xp.int32
                    )

                    factors_multiply_generate = self.xp.concatenate([
                        +1 * self.xp.ones_like(group_index_i, dtype=float),
                        -1 * self.xp.ones_like(group_index_i, dtype=float),
                        +1 * self.xp.ones_like(group_index_i, dtype=float),
                        -1 * self.xp.ones_like(group_index_i, dtype=float),
                    ])

                    N_vals_in_gen = self.xp.concatenate([
                        N_vals_i,
                        N_vals_i,
                        N_vals_i,
                        N_vals_i
                    ])
                    
                    waveform_kwargs_fill = waveform_kwargs_now.copy()
                    waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

                    self.gb.generate_global_template(
                        params_generate_in,
                        group_index_gen, 
                        self.mgh.data_list, 
                        N=N_vals_in_gen,
                        data_length=self.data_length, 
                        data_splits=self.mgh.gpu_splits, 
                        factors=factors_multiply_generate,
                        **waveform_kwargs_fill
                    )

                    # update likelihoods

                    # set unaccepted differences to zero
                    accepted_delta_ll_i = delta_logl_i * (sel)
                    accepted_delta_ll_i1 = delta_logl_i1 * (sel)

                    logl_change_contribution = np.zeros_like(log_like_tmp.get())
                    try:
                        in_tuple = (accepted_delta_ll_i[sel].get(), accepted_delta_ll_i1[sel].get(), temp_inds_i[sel].get(), temp_inds_i1[sel].get(), walkers_inds_i[sel].get(), walkers_inds_i[sel].get())
                    except AttributeError:
                        in_tuple = (accepted_delta_ll_i[sel], accepted_delta_ll_i1[sel], temp_inds_i[sel], temp_inds_i1[sel], walkers_inds_i[sel], walkers_inds_i[sel])
                    for j, (dlli, dlli1, ti, ti1, wi, wi1) in enumerate(zip(*in_tuple)):
                        logl_change_contribution[ti, wi] += dlli
                        logl_change_contribution[ti1, wi1] += dlli1

                    log_like_tmp[:] += self.xp.asarray(logl_change_contribution)

                    tmp_swap = new_state.branches["gb"].coords[inds_i_swap]
                    new_state.branches["gb"].coords[inds_i_swap] = new_state.branches["gb"].coords[inds_i1_swap]

                    new_state.branches["gb"].coords[inds_i1_swap] = tmp_swap

                    tmp_swap = new_state.branches["gb"].branch_supplimental[inds_i_swap]

                    new_state.branches["gb"].branch_supplimental[inds_i_swap] = new_state.branches["gb"].branch_supplimental[inds_i1_swap]

                    new_state.branches["gb"].branch_supplimental[inds_i1_swap] = tmp_swap

                    # inds are all non-zero
                    self.swaps_accepted[i - 1] += np.sum(sel)
                    self.attempted_swaps[i - 1] += sel.shape[0]

                    ll_after = self.mgh.get_ll(use_cpu=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
                    breakpoint()
                    """
        else:
            self.temperature_control.swaps_accepted = np.zeros((ntemps - 1))
        
        
        if np.any(new_state.log_like > 1e10):
            breakpoint()

        self.time += 1
        #self.xp.cuda.runtime.deviceSynchronize()
        #et = time.perf_counter()
        #print("end stretch", (et - st))

        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        """et = time.perf_counter()
        print("end", (et - st), group_iter, group_len[group_iter])"""
                    
        # breakpoint()
        return new_state, accepted

