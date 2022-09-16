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

from eryn.moves import GroupStretchMove
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds
from .gbmultipletryrj import GBMutlipleTryRJ
from .gbgroupstretch import GBGroupStretchMove

from ...diagnostic import inner_product
from eryn.state import State


__all__ = ["GBGroupStretchMove"]


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialGroupStretchMove(GBGroupStretchMove, GBMutlipleTryRJ):
    """Generate Revesible-Jump proposals for GBs with try-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        gb_args,
        gb_kwargs,
        start_ind_limit=10,
        *args,
        **kwargs
    ):
        self.fixed_like_diff = 0
        self.time = 0
        self.name = "gbgroupstretch"
        self.start_ind_limit = start_ind_limit
        GBMutlipleTryRJ.__init__(self, *gb_args, **gb_kwargs)
        GroupStretchMove.__init__(self, *args, **kwargs)

    def setup_gbs(self, branch):
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        all_remaining_coords = coords[inds]
        remaining_wave_info = supps[inds]

        num_remaining = len(all_remaining_coords)
        # TODO: make faster?
        points_out = np.zeros((num_remaining, self.nfriends, ndim))
        
        # info_mat = self.gb.information_matrix()

        freqs = all_remaining_coords[:, 1]

        # TODO: improve this?
        inds_freqs_sorted = np.argsort(freqs)
        freqs_sorted = freqs[np.argsort(freqs)]
        
        inds_reverse = np.empty_like(inds_freqs_sorted)
        inds_reverse[inds_freqs_sorted] = np.arange(inds_freqs_sorted.size)

        left = np.full(len(freqs_sorted), self.nfriends)
        right = np.full(len(freqs_sorted), self.nfriends)
        indexes = np.arange(len(freqs_sorted))
        
        left = left * (indexes >= self.nfriends) + indexes * (indexes < self.nfriends)
        right = right * (indexes < len(freqs_sorted) - self.nfriends) + (len(freqs_sorted) - 1 - indexes) * (indexes >= len(freqs_sorted) - self.nfriends)

        left = left * ((left + right == self.nfriends * 2) | (right == self.nfriends)) + (left + self.nfriends - right) * ((left + right < 2 * self.nfriends) & (right != self.nfriends))

        right = right * ((left + right == self.nfriends * 2) | (left == self.nfriends)) + (right + self.nfriends - left) * ((left + right < self.nfriends * 2) & (left != self.nfriends))
        inds_keep = (np.tile(np.arange(2 * self.nfriends), (len(freqs_sorted), 1)) + np.arange(len(freqs_sorted))[:, None] -  left[:, None]).astype(int)
        
        #distances = np.abs(freqs[None, :] - freqs[:, None])
        #distances[distances == 0.0] = 1e300
        """
        distances = self.xp.full((num_remaining,num_remaining), 1e300)
        for i, coords_here in enumerate(all_remaining_coords):
            #A_here = remaining_wave_info["A"][i]
            #E_here = remaining_wave_info["E"][i]
            #sig_len = len(A_here)
            #start_ind_here = remaining_wave_info["start_inds"][i].item()
            #freqs_here = (self.xp.arange(sig_len) + start_ind_here) * self.df
            #psd_here = self.psd[0][start_ind_here - self.start_freq_ind: start_ind_here - self.start_freq_ind + sig_len]
    
            h_h = inner_product([A_here, E_here], [A_here, E_here], f_arr=freqs_here, PSD=psd_here, use_gpu=self.use_gpu)
            
            for j in range(i, num_remaining):
                if j == i:
                    continue
                A_check = remaining_wave_info["A"][j]
                E_check = remaining_wave_info["E"][j]
                start_ind_check = remaining_wave_info["start_inds"][j].item()
                if abs(start_ind_here - start_ind_check) > self.start_ind_limit:
                    continue
                start_ind = self.xp.max(self.xp.asarray([start_ind_here, start_ind_check])).item()
                end_ind = self.xp.min(self.xp.asarray([start_ind_here + sig_len, start_ind_check + sig_len])).item()
                sig_len_new = end_ind - start_ind

                start_ind_now_here = start_ind - start_ind_here
                slice_here = slice(start_ind_now_here, start_ind_now_here + sig_len_new)

                start_ind_now_check = start_ind - start_ind_check
                slice_check = slice(start_ind_now_check, start_ind_now_check + sig_len_new)
                d_h = inner_product([A_here[slice_here], E_here[slice_here]], [A_check[slice_check], E_check[slice_check]], f_arr=freqs_here[slice_here], PSD=psd_here[slice_here], use_gpu=self.use_gpu)
                
                distances[i, j] = abs(1.0 - d_h.real / h_h.real)
                distances[j, i] = distances[i, j]
            print(i)

        keep = self.xp.argsort(distances)[:self.nfriends]
        try:
            keep = keep.get()
        except AttributeError:
            pass
        

        breakpoint()
        """
        """
        """
        try:
            keep = inds_freqs_sorted[inds_keep][inds_reverse]
        except IndexError:
            breakpoint()
        suggested_friends = all_remaining_coords[keep]
        distances = np.abs(suggested_friends[:, :, 1] - all_remaining_coords[:, 1][:, None])
        distances_pretend = np.zeros_like(distances)
        distances_pretend[distances == 0.0] = 1e300
        keep = np.argsort(distances_pretend, axis=1)[:, :self.nfriends]
        friends = np.take_along_axis(suggested_friends, keep[:, :, None], axis=1)

        supps[inds] = {"group_move_points": friends}

    def setup_noise_params(self, branch):
        
        coords = branch.coords
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape

        par0 = coords[:, :, :, 0].flatten()

        distances = np.abs(par0[None, :] - par0[:, None])
        distances[distances == 0.0] = 1e300
        
        keep = np.argsort(distances, axis=1)[:, :self.nfriends]
        supps[:] = {"group_move_points": coords.reshape(-1, 1)[keep].reshape(ntemps, nwalkers, nleaves_max, self.nfriends, ndim)}

    def find_friends(self, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name == "gb":
                self.setup_gbs(branch)
            #elif name == "noise_params":
            #    self.setup_noise_params(branch)
            #else:
            #    raise NotImplementedError

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        
        # st = time.perf_counter()
        # Check that the dimensions are compatible.
        ndim_total = 0
        ntemps, nwalkers, nleaves_, ndim_ = state.branches["gb"].shape 
        
        if state.branches["gb"].nleaves.sum() < 2 * self.nfriends:
            print("Not enough friends yet.")
            accepted = np.zeros((ntemps, nwalkers), dtype=bool)
            self.temperature_control.swaps_accepted = np.zeros(ntemps - 1, dtype=int)
            return state, accepted

        # TODO: deal with more intensive acceptance fractions
        # Run any move-specific setup.
        self.setup(state.branches)
        # st = time.perf_counter()

        new_state = State(state, copy=True)

        # ll_before = model.compute_log_prob_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplimental, branch_supps=new_state.branches_supplimental)
        
        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        f_test = self.xp.asarray(state.branches_coords["gb"][:, :, :, 1]) / 1e3

        # set any inds == False binary to zero 
        f_test[~self.xp.asarray(state.branches_inds["gb"])] = 0.0
        
        f_test[f_test == 0.0] = 1e300
        
        f_test_sorted = self.xp.asarray(self.xp.sort(f_test,axis=-1))

        inds_f_sorted = self.xp.asarray(self.xp.argsort(f_test,axis=-1))
        groups = []
        group_len = []

        buffer = 0  # 2 ** 8

        dbin = 2 * self.waveform_kwargs["N"] + buffer
        max_iter = 1000
        i = 0
        total = 0
        while i < max_iter and ~self.xp.all(f_test_sorted > 1e100):
            diff_along_f = self.xp.zeros_like(f_test_sorted)
            diff_along_f[:, :, 1:] = self.xp.diff(f_test_sorted, axis=-1)
            
            tmp1 = (self.xp.cumsum(diff_along_f, axis=-1) / self.df).astype(int) // dbin
            tmp1[(tmp1) % 2 == 1] = 0
            tmp1[:, :, 1:] = self.xp.diff(tmp1, axis=-1)
            tmp1[tmp1 < 0] = 0

            switch_check = self.xp.zeros_like(f_test_sorted, dtype=int)
            switch_check[:, :, 1:] = self.xp.diff(self.xp.cumsum(tmp1, axis=-1), axis=-1)
            switch_check[:, :, 0] = 2
            inds_switch = self.xp.where((switch_check > 0) & (f_test_sorted < 1e100))

            groups.append((inds_switch[0], inds_switch[1], inds_f_sorted[inds_switch]))
                
            group_len.append(len(inds_switch[0]))

            #print(f_test_sorted.shape)
            f_test_sorted[inds_switch] = 1e300

            sort_inds = self.xp.argsort(f_test_sorted, axis=-1)
            f_test_sorted = self.xp.take_along_axis(f_test_sorted, sort_inds, axis=-1)

            inds_f_sorted = self.xp.take_along_axis(inds_f_sorted, sort_inds, axis=-1)
            total += len(inds_switch[0])
            #print(i, len(inds_switch[0]), (f_test_sorted > 1e100).sum(axis=-1))
            #i += 1
            #if i % 10 == 0:
            #    print(i, (f_test_sorted > 1e100).sum() / np.prod(f_test_sorted.shape))

        gb_coords = self.xp.asarray(new_state.branches_coords["gb"].copy())

        log_prob_tmp = self.xp.asarray(new_state.log_prob)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)

        data_minus_template = new_state.supplimental.holder["data_minus_template"]

        """et = time.perf_counter()
        print("group groups", (et - st))
        st = time.perf_counter()"""
        for group in groups:
            temp_inds, walkers_inds, leaf_inds = group
            if self.use_gpu:
                group_cpu = (temp_inds.get(), walkers_inds.get(), leaf_inds.get())
            else:
                group_cpu = group

            q = q = {"gb": gb_coords.copy()}
            # new_inds = deepcopy(new_state.branches_inds)

            points_to_move = q["gb"][group]
            group_branch_supp_info = new_state.branches_supplimental["gb"][group_cpu]
            points_for_move = self.xp.asarray(group_branch_supp_info["group_move_points"])

            """et = time.perf_counter()
            print("before prop", (et - st))
            st = time.perf_counter()"""
            q_temp, factors_temp = self.get_proposal(
                {"gb": points_to_move},  {"gb": points_for_move}, model.random
            )

            """et = time.perf_counter()
            print("after prop", (et - st))
            st = time.perf_counter()"""

            #breakpoint()

            q["gb"][group] = q_temp["gb"]

            # data should not be whitened
            if "noise_params" not in q:
                use_stock_psd = True
                psd = self.xp.tile(self.xp.asarray(self.psd), (ntemps * nwalkers, 1, 1))

            else:
                use_stock_psd = False
                noise_params = q["noise_params"]
                if self.psd_func is None:
                    raise ValueError("When providing noise_params, psd_func kwargs in __init__ function must be given.")

                if noise_params.ndim == 3:
                    noise_params = noise_params[0]
                try:
                    tmp = self.xp.asarray([self.psd_func(self.fd, *noise_params.reshape(-1, noise_params.shape[-1]).T, **self.noise_kwargs) for _ in range(2)])
                    psd = tmp.transpose((1,0,2))
                    breakpoint()
                except ValueError:
                    breakpoint()

            if self.use_gpu:
                new_points_prior = q["gb"][group].get()
                old_points_prior = gb_coords[group].get()
            else:
                new_points_prior = q["gb"][group]
                old_points_prior = gb_coords[group]

            # TODO: GPUize prior
            logp = self.xp.asarray(self.priors["gb"].logpdf(new_points_prior))

            if self.xp.all(self.xp.isinf(logp)):
                continue

            keep_here = self.xp.where(~self.xp.isinf(logp))

            points_remove = self.parameter_transforms.both_transforms(points_to_move[keep_here], xp=self.xp)
            points_add = self.parameter_transforms.both_transforms(q_temp["gb"][keep_here], xp=self.xp)

            data_index = self.xp.asarray((temp_inds[keep_here] * nwalkers + walkers_inds[keep_here]).astype(xp.int32))
            noise_index = self.xp.asarray((temp_inds[keep_here] * nwalkers + walkers_inds[keep_here]).astype(xp.int32))
            nChannels = 2
        
            delta_ll = self.xp.full(points_to_move.shape[0], -1e300)
            
            delta_ll[keep_here] = self.gb.swap_likelihood_difference(
                points_remove, 
                points_add, 
                data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).copy(), 
                psd.copy(), 
                data_index=data_index, 
                noise_index=noise_index, 
                adjust_inplace=False, 
                **self.waveform_kwargs
            )
            """dhr = self.gb.d_h_remove.copy()
            dha = self.gb.d_h_add.copy()
            aa = self.gb.add_add.copy()
            rr = self.gb.remove_remove.copy()
            ar = self.gb.add_remove.copy()

            kwargs_tmp = self.waveform_kwargs.copy()
            kwargs_tmp["use_c_implementation"] = False
            check = self.gb.swap_likelihood_difference(
                    points_remove, 
                    points_add, 
                    data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).copy(), 
                    psd.reshape(ntemps * nwalkers, nChannels, -1).copy(), 
                    data_index=data_index, 
                    noise_index=noise_index, 
                    adjust_inplace=False, 
                    **kwargs_tmp
                )
            breakpoint()"""

            
            optimized_snr = self.xp.sqrt(self.gb.add_add.real)
            detected_snr = (self.gb.d_h_add + self.gb.add_remove).real / optimized_snr
            
            if self.search:
                inds_fix = ((optimized_snr < self.search_snr_lim) | (detected_snr < (0.8 * self.search_snr_lim)))

                """try:
                    inds_fix = inds_fix.get()
                except AttributeError:
                    pass"""
                if self.xp.any(inds_fix):
                    delta_ll[keep_here[0][inds_fix]] = -1e300

            prev_logl = log_prob_tmp[(temp_inds, walkers_inds)]
            logl = delta_ll + prev_logl

            #if np.any(logl - np.load("noise_ll.npy").flatten() > 0.0):
            #    breakpoint()    
            #print("multi check: ", (logl - np.load("noise_ll.npy").flatten()))
            prev_logp = self.xp.asarray(self.priors["gb"].logpdf(old_points_prior))

            betas_in = self.xp.asarray(self.temperature_control.betas)[temp_inds]
            logP = self.compute_log_posterior(logl, logp, betas=betas_in)
            
            # TODO: check about prior = - inf
            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp, betas=betas_in)

                        # TODO: think about factors in tempering
            lnpdiff = factors_temp + logP - prev_logP

            # TODO: think about random states
            keep = lnpdiff > self.xp.log(self.xp.random.rand(*logP.shape))

            if self.xp.any(keep):
                # if gibbs sampling, this will say it is accepted if
                # any of the gibbs proposals were accepted
                accepted_here = keep.copy()

                # check freq overlap
                f0_new = q_temp["gb"][keep, 1]
                #f0_old = gb_coords[(temp_inds[keep], walkers_inds[keep], leaf_inds[keep])][:, 1]
                nleaves_max = state.branches["gb"].nleaves_max
                check_f0 = self.xp.zeros((ntemps, nwalkers, nleaves_max))
                #check_f0_old = self.xp.zeros((ntemps, nwalkers, nleaves_max))
                check_f0[(temp_inds[keep], walkers_inds[keep], leaf_inds[keep])] = f0_new
                #check_f0_old[group] = f0_old

                check_f0_sorted = self.xp.sort(check_f0, axis=-1)
                inds_f0_sorted = self.xp.argsort(check_f0, axis=-1)
                check_f0_diff = self.xp.zeros_like(check_f0_sorted, dtype=int)
                check_f0_diff[:, :, 1:] = (self.xp.diff(check_f0_sorted, axis=-1) / 1e3 / self.df).astype(int)

                bad = (check_f0_diff < dbin) & (check_f0_sorted != 0.0)
                if self.xp.any(bad):
                    try:
                        bad_inds = self.xp.where(bad)
                        if self.xp.any(bad_inds[0] < 2):
                            breakpoint()
                        # fix the last entry of bad inds
                        inds_bad = (bad_inds[0], bad_inds[1], inds_f0_sorted[bad])
                        bad_check_val = (inds_bad[0] * 1e10 + inds_bad[1] * 1e5 + inds_bad[2]).astype(int)
                        # we are going to make this proposal not accepted
                        # this so far is only ever an accepted-level problem at high temps 
                        # where large frequency jumps can happen
                        check_val = (temp_inds[keep] * 1e10 + walkers_inds[keep] * 1e5 + leaf_inds[keep]).astype(int)
                        fix_keep = self.xp.arange(len(keep))[keep][self.xp.in1d(check_val, bad_check_val)]
                        keep[fix_keep] = False
                    except:
                        breakpoint()

                gb_coords[(temp_inds[keep], walkers_inds[keep], leaf_inds[keep])] = q_temp["gb"][keep]

                # parameters were run for all ~np.isinf(logp), need to adjust for those not accepted
                keep_from_before = (keep * (~self.xp.isinf(logp)))[~self.xp.isinf(logp)]
                group_index = data_index[keep_from_before]

                waveform_kwargs_fill = self.waveform_kwargs.copy()
                waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind
                """ll_check_d_h_add = self.gb.get_ll(
                    points_add.T, 
                    data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    data_index=data_index, 
                    noise_index=noise_index, 
                    **self.waveform_kwargs
                )

                h_h_d_h_add = self.gb.h_h.copy()
                d_h_d_h_add = self.gb.d_h.copy()

                ll_check_d_h_remove = self.gb.get_ll(
                    points_remove.T, 
                    data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    data_index=data_index, 
                    noise_index=noise_index, 
                    **self.waveform_kwargs
                )

                h_h_d_h_remove = self.gb.h_h.copy()
                d_h_d_h_remove = self.gb.d_h.copy()"""

            
                # remove templates by multiplying by "adding them to" d - h
                try:
                    self.gb.generate_global_template(points_remove[keep_from_before],
                        group_index, data_minus_template.reshape((-1,) + data_minus_template.shape[2:]), **waveform_kwargs_fill
                    )
                except ValueError:
                    breakpoint()

                
                """self.gb.d_d = self.xp.asarray(-2 * state.log_prob.flatten())
                ll_check_add = self.gb.get_ll(
                    points_add.T, 
                    data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    data_index=data_index, 
                    noise_index=noise_index, 
                    **self.waveform_kwargs
                )

                h_h_add = self.gb.h_h.copy()
                d_h_add = self.gb.d_h.copy()

                ll_check_remove = self.gb.get_ll(
                    points_remove.T, 
                    data_minus_template.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    psd.reshape(ntemps * nwalkers, nChannels, -1).transpose(1, 0, 2), 
                    data_index=data_index, 
                    noise_index=noise_index, 
                    **self.waveform_kwargs
                )
                
                h_h_remove = self.gb.h_h.copy()
                d_h_remove = self.gb.d_h.copy()
                breakpoint()"""

                # add templates by adding to -(-(d - h) + add) = d - (h + add)
                data_minus_template *= -1
                try:
                    self.gb.generate_global_template(points_add[keep_from_before],
                        group_index, data_minus_template.reshape((-1,) + data_minus_template.shape[2:]), **waveform_kwargs_fill
                    )
                except ValueError:
                    breakpoint()
                data_minus_template *= -1

                # set unaccepted differences to zero
                accepted_delta_ll = delta_ll * (keep)
                accepted_delta_lp = (logp - prev_logp)
                accepted_delta_lp[self.xp.isinf(accepted_delta_lp)] = 0.0
                logl_change_contribution = np.zeros_like(new_state.log_prob)
                logp_change_contribution = np.zeros_like(new_state.log_prior)

                try:
                    in_tuple = (accepted_delta_ll[keep].get(), accepted_delta_lp[keep].get(), temp_inds[keep].get(), walkers_inds[keep].get())
                except AttributeError:
                    in_tuple = (accepted_delta_ll[keep], accepted_delta_lp[keep], temp_inds[keep], walkers_inds[keep])

                for i, (dll, dlp, ti, wi) in enumerate(zip(*in_tuple)):

                    logl_change_contribution[ti, wi] += dll
                    logp_change_contribution[ti, wi] += dlp

                log_prob_tmp += self.xp.asarray(logl_change_contribution)
                log_prior_tmp += self.xp.asarray(logp_change_contribution)

                if np.any(accepted_delta_ll > 1e8):
                    breakpoint()

        try:
            new_state.branches["gb"].coords[:] = gb_coords.get()
            new_state.log_prob[:] = log_prob_tmp.get()
            new_state.log_prior[:] = log_prior_tmp.get()
        except AttributeError:
            new_state.branches["gb"].coords[:] = gb_coords
            new_state.log_prob[:] = log_prob_tmp
            new_state.log_prior[:] = log_prior_tmp

        if self.time % 1 == 0:
            lp_after = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds)
            ll_after = model.compute_log_prob_fn(new_state.branches_coords, inds=new_state.branches_inds, logp=lp_after, supps=new_state.supplimental, branch_supps=new_state.branches_supplimental)
            #check = -1/2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / self.xp.asarray(self.psd), axis=(2, 3))
            #check2 = -1/2 * 4 * self.df * self.xp.sum(tmp.conj() * tmp / self.xp.asarray(self.psd), axis=(2, 3))
            #print(np.abs(new_state.log_prob - ll_after[0]).max(), np.abs(new_state.log_prior - lp_after).max())
            if np.abs(new_state.log_prior - lp_after).max() > 0.1 or np.abs(new_state.log_prob - ll_after[0]).max() > 1e0:
                breakpoint()

            # if any are even remotely getting to be different, reset all (small change)
            elif np.abs(new_state.log_prob - ll_after[0]).max() > 1e-3:
                
                fix_here = np.abs(new_state.log_prob - ll_after[0]) > 1e-6
                data_minus_template_old = data_minus_template.copy()
                data_minus_template = self.xp.zeros_like(data_minus_template_old)
                data_minus_template[:] = self.xp.asarray(self.data)[None, None]
                templates = self.xp.zeros_like(data_minus_template).reshape(-1, 2, data_minus_template.shape[-1])
                for name in new_state.branches.keys():
                    if name not in ["gb", "gb_fixed"]:
                        continue
                    new_state_branch = new_state.branches[name]
                    coords_here = new_state_branch.coords[new_state_branch.inds]
                    ntemps, nwalkers, nleaves_max_here, ndim = new_state_branch.shape
                    group_index = np.repeat(np.arange(ntemps * nwalkers).reshape(ntemps, nwalkers, 1), nleaves_max, axis=-1)[new_state_branch.inds]
                    coords_here_in = self.parameter_transforms.both_transforms(coords_here, xp=np)

                    self.gb.generate_global_template(coords_here_in, group_index, templates, batch_size=1000, **self.waveform_kwargs)

                data_minus_template -= templates.reshape(ntemps, nwalkers, 2, templates.shape[-1])

                new_like = -1 / 2 * 4 * self.df * self.xp.sum(data_minus_template.conj() * data_minus_template / psd, axis=(2, 3)).real.get()
            
                new_like += self.noise_ll
                new_state.log_prob[:] = new_like.reshape(ntemps, nwalkers)
                
            """elif np.abs(new_state.log_prior - lp_after).max() > 1e-6 or np.abs(new_state.log_prob - ll_after[0]).max() > 0.1:
                # TODO: need to investigate when this fails
                self.fixed_like_diff += 1
                print("Fixing like diff for now.", self.fixed_like_diff)
                fix_here = np.abs(new_state.log_prob - ll_after[0]) > 0.1
                new_state.log_prob[fix_here] = ll_after[0][fix_here]"""

            """
            check_logl = model.compute_log_prob_fn(new_state.branches_coords, inds=new_state.branches_inds, branch_supps=new_state.branches_supplimental, supps=new_state.supplimental)
            sigll = model.log_prob_fn.f.signal_ll.copy()
            check_logl2 = model.compute_log_prob_fn(state.branches_coords, inds=state.branches_inds, branch_supps=state.branches_supplimental, supps=state.supplimental)
            breakpoint()
            """
        """et = time.perf_counter()
        print("group middle", (et - st))
        st = time.perf_counter()"""
        # get accepted fraction 
        accepted_check = np.all(np.abs(new_state.branches_coords["gb"] - state.branches_coords["gb"]) > 0.0, axis=-1).sum(axis=(1, 2)) / new_state.branches_inds["gb"].sum(axis=(1,2))

        # manually tell temperatures how real overall acceptance fraction is
        number_of_walkers_for_accepted = np.floor(nwalkers * accepted_check).astype(int)

        accepted_inds = np.tile(np.arange(nwalkers), (ntemps, 1))

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        accepted[accepted_inds < number_of_walkers_for_accepted[:, None]] = True

        if self.temperature_control is not None:
            new_state, accepted = self.temperature_control.temper_comps(new_state, accepted)
        # self.temperature_control.swaps_accepted = np.zeros((ntemps - 1))

        if np.any(new_state.log_prob > 1e10):
            breakpoint()
        self.time += 1

        #et = time.perf_counter()
        #print("group end", (et - st))

        return new_state, accepted

