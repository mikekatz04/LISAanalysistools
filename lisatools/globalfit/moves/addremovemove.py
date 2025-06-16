import numpy as np
import cupy as xp
from copy import deepcopy
import time

from eryn.moves import Move, StretchMove, TemperatureControl
# from eryn.state import State
from lisatools.globalfit.state import GFState
from lisatools.sampling.moves.skymodehop import SkyMove
from bbhx.likelihood import NewHeterodynedLikelihood
from tqdm import tqdm
from .globalfitmove import GlobalFitMove


class ResidualAddOneRemoveOneMove(GlobalFitMove, StretchMove, Move):
    def __init__(self, branch_name, coords_shape, waveform_gen, tempering_kwargs, waveform_gen_kwargs, waveform_like_kwargs, acs, num_repeats, transform_fn, priors, inner_moves, df, 
        Tmax=np.inf, betas_all = None, **kwargs):
        
        # Move.__init__(self, **kwargs)
        StretchMove.__init__(self, **kwargs)

        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = coords_shape
        
        self.branch_name = branch_name
        self.acs = acs
        self.waveform_gen = waveform_gen
        self.num_repeats = num_repeats
        self.transform_fn = transform_fn
        self.priors = priors
        self.waveform_gen_kwargs = waveform_gen_kwargs
        self.waveform_like_kwargs = waveform_like_kwargs
        moves_tmp = [move[0] for move in inner_moves]
        move_weights = [move[1] for move in inner_moves]
        self.moves = moves_tmp
        self.move_weights = move_weights
        self.df = acs.df
        # get data frequency array on gpu 
        self.fd = xp.asarray(acs.f_arr)

        self.temperature_controls = [None for _ in range(self.nleaves_max)]
        for i in range(self.nleaves_max):
            if betas_all is not None:
                assert betas_all.shape == (self.nleaves_max, self.ntemps)
                betas_in = betas_all[i]
            else:
                betas_in = None

            self.temperature_controls[i] = TemperatureControl(
                self.ndim,
                self.nwalkers,
                betas=betas_in,
                permute=False,
                ntemps=self.ntemps,
                Tmax=Tmax,
                skip_swap_branches=None  # will fill in after first run through move
            )

    def check_add_skip_swap_info(self, state):
        if len(state.branches) > 1:
            if self.temperature_controls[0].skip_swap_branches is None:
                skip_swap_branches = [key for key in state.branches.keys()]
                skip_swap_branches.remove(self.branch_name)
                for i in range(self.nleaves_max):
                    self.temperature_controls[i].skip_swap_branches = skip_swap_branches
    
    def add_back_in_cold_chain_sources(self, coords):

        # TODO: fix T channel 
        # d - h -> need to add removal waveforms
        # ll_tmp1 = (-1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        removal_waveforms = self.get_waveform_here(coords)
        ll_tmp2 = self.acs.likelihood(source_only=True)  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        self.acs.remove_signal_from_residual(
            removal_waveforms, data_index=None, start_index=None
        )
        del removal_waveforms
        xp.get_default_memory_pool().free_all_blocks()
        
        ll_tmp3 = self.acs.likelihood(source_only=True)  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

    def remove_cold_chain_sources(self, coords):
        # TODO: fix T channel 
        # d - h -> need to add removal waveforms
        # ll_tmp1 = (-1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        removal_waveforms = self.get_waveform_here(coords)
        ll_tmp2 = self.acs.likelihood(source_only=True)  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        self.acs.add_signal_to_residual(
            removal_waveforms, data_index=None, start_index=None
        )
        del removal_waveforms
        xp.get_default_memory_pool().free_all_blocks()
        
        ll_tmp3 = self.acs.likelihood(source_only=True)  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

    def get_waveform_here(self, coords):
        xp.get_default_memory_pool().free_all_blocks()
        waveforms = xp.zeros((coords.shape[0], self.acs.nchannels, self.acs.data_length), dtype=complex)
        
        for i in range(coords.shape[0]):
            waveforms[i] = self.waveform_gen(*coords[i], **self.waveform_gen_kwargs)
        
        return waveforms

    def setup_likelihood_here(self, coords):
        pass

    def compute_like(self, old_coords_in, data_index):
        # TODO: we should probably move the prior in here even though 
        # in general with current setup it should only be points in the prior
        # that make it here
        ll = np.full_like(data_index.get(), -1e300, dtype=float)

        for i, (coords_in_now, data_index_now) in enumerate(zip(old_coords_in, data_index.get())):
            ll[i] = self.acs[data_index_now].calculate_signal_likelihood(*coords_in_now, signal_gen=self.waveform_gen)
        return ll

    def propose(self, model, state):
        print("PROPOSING")
        print("------" * 20)
        tic = time.time()   

        new_state = deepcopy(state)

        self.acs = model.analysis_container_arr
        self.check_add_skip_swap_info(state)

        # mapping information
        temp_inds_base = np.repeat(np.arange(self.ntemps)[:, None], self.nwalkers, axis=-1)
        walker_inds_base = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))

        # randomize order
        start_leaf = np.random.randint(0, self.nleaves_max)
        for base_leaf in range(self.nleaves_max):
            # second step of randomizing order (making sure it does not run over)
            leaf = (base_leaf + start_leaf) % self.nleaves_max
            print(f"Processing leaf {leaf}")

            # fill this temperature control with temperatures from current state
            temperature_control_here = self.temperature_controls[leaf]

            temperature_control_here.betas[:] = new_state.sub_states[self.branch_name].betas_all[leaf]

            ndim = new_state.branches[self.branch_name].coords.shape[-1]

            # remove cold chain sources
            removal_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            self.add_back_in_cold_chain_sources(removal_coords_in)
           
            self.setup_likelihood_here(removal_coords_in)

            old_coords = new_state.branches[self.branch_name].coords[:, :, leaf].reshape(-1, ndim)
            old_coords_in = self.transform_fn.both_transforms(old_coords)
            
            data_index_in = xp.tile(xp.arange(self.nwalkers), (self.ntemps, 1)).flatten().astype(xp.int32)
            # TODO: fix this
            # prev_logl = self.waveform_gen.get_direct_ll(fd, data_residuals.flatten(), psd.flatten(), self.df, *old_coords_in.T, noise_index=noise_index, data_index=data_index, **self.waveform_kwargs).reshape((ntemps, nwalkers)).real.get()
            # TODO: check if psd term is included properly here at each step
            # TODO: and check data index here
            prev_logl = self.compute_like(
                old_coords_in, 
                data_index=data_index_in,
            ).reshape((self.ntemps, self.nwalkers)).real
            
            print(f"prev_logl: {prev_logl}. elapsed: {time.time() - tic}")

            prev_logp = self.priors[self.branch_name].logpdf(old_coords).reshape((self.ntemps, self.nwalkers))

            prev_logP = temperature_control_here.compute_log_posterior_tempered(prev_logl, prev_logp)
            
            # fix this need to compute prev_logl for all walkers
            xp.get_default_memory_pool().free_all_blocks()
            for repeat in range(self.num_repeats):

                # pick move
                move_here = self.moves[model.random.choice(np.arange(len(self.moves)), p=self.move_weights)]

                # Split the ensemble in half and iterate over these two halves.
                accepted = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
                all_inds = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))
                inds = all_inds % self.nsplits
                if self.randomize_split:
                    [np.random.shuffle(x) for x in inds]

                # prepare accepted fraction
                accepted_here = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
                for split in range(self.nsplits):
                    # get split information
                    S1 = inds == split
                    num_total_here = np.sum(inds == split)
                    nwalkers_here = np.sum(S1[0])

                    temp_inds_here = temp_inds_base[inds == split]
                    walker_inds_here = walker_inds_base[inds == split]

                    # prepare the sets for each model
                    # goes into the proposal as (ntemps * (nwalkers / subset size), nleaves_max, ndim)
                    sets = [
                        new_state.branches[self.branch_name].coords[inds == j][:, leaf].reshape(self.ntemps, -1, 1, ndim)
                        for j in range(self.nsplits)
                    ]

                    old_points = sets[split].reshape((self.ntemps, nwalkers_here, ndim))

                    # setup s and c based on splits
                    s = {self.branch_name: sets[split]}
                    c = {self.branch_name: sets[:split] + sets[split + 1 :]}
                    
                    # Get the move-specific proposal.
                    if isinstance(move_here, StretchMove):
                        q, factors = move_here.get_proposal(
                            s, c, model.random
                        )

                    else:
                        q, factors = move_here.get_proposal(s, model.random)

                    new_points = q[self.branch_name].reshape((self.ntemps, nwalkers_here, ndim))

                    # Compute prior of the proposed position
                    # new_inds_prior is adjusted if product-space is used
                    logp = self.priors[self.branch_name].logpdf(new_points.reshape(-1, ndim))

                    new_points_in = self.transform_fn.both_transforms(new_points.reshape(-1, ndim)[~np.isinf(logp)])
                    
                    # Compute the lnprobs of the proposed position.
                    data_index = xp.asarray(walker_inds_here[~np.isinf(logp)].astype(np.int32))
                    # noise_index = walker_inds_here[~np.isinf(logp)].astype(np.int32)

                    # self.waveform_gen.d_d = xp.asarray(d_d_store[(temp_inds_here[~np.isinf(logp)], walker_inds_here[~np.isinf(logp)])])
                    
                    logl = np.full_like(logp, -1e300)

                    # logl[~np.isinf(logp)] = self.waveform_gen.get_direct_ll(fd, data_residuals.flatten(), psd.flatten(), self.df, *new_points_in.T, noise_index=noise_index, data_index=data_index, **self.waveform_kwargs).real.get()
                    logl[~np.isinf(logp)] = self.compute_like(
                        new_points_in, 
                        data_index=data_index,
                        #constants_index=data_index,
                    )

                    print(f"new logl: {logl}. elapsed: {time.time() - tic}")

                    logl = logl.reshape(self.ntemps, nwalkers_here)
                    
                    logp = logp.reshape(self.ntemps, nwalkers_here)
                    prev_logp_here = prev_logp[inds == split].reshape(self.ntemps, nwalkers_here)

                    prev_logl_here = prev_logl[inds == split].reshape(self.ntemps, nwalkers_here)

                    prev_logP_here = temperature_control_here.compute_log_posterior_tempered(prev_logl_here, prev_logp_here)
                    logP = temperature_control_here.compute_log_posterior_tempered(logl, logp)

                    lnpdiff = factors + logP - prev_logP_here

                    keep = lnpdiff > np.log(model.random.rand(self.ntemps, nwalkers_here))

                    temp_inds_update = temp_inds_here[keep.flatten()]
                    walker_inds_update = walker_inds_here[keep.flatten()]

                    accepted[(temp_inds_update, walker_inds_update)] = True

                    # update state informatoin
                    new_state.branches[self.branch_name].coords[(temp_inds_update, walker_inds_update, np.full_like(walker_inds_update, leaf))] = new_points[keep].reshape(len(temp_inds_update), ndim)

                    prev_logl[(temp_inds_update, walker_inds_update)] = logl[keep].flatten()
                    prev_logp[(temp_inds_update, walker_inds_update)] = logp[keep].flatten()
                    prev_logP[(temp_inds_update, walker_inds_update)] = logP[keep].flatten()

                # acceptance tracking
                self.accepted += accepted
                # print(self.accepted[0])
                self.num_proposals += 1

                # TODO: include PSD likelihood in swaps?
                # temperature swaps
                # make swaps
                coords_for_swap = {self.branch_name: new_state.branches_coords[self.branch_name][:, :, leaf].copy()[:, :, None]}
                
                # TODO: check permute make sure it is okay
                coords_for_swap, prev_logP, prev_logl, prev_logp, inds, blobs, supps, branch_supps = temperature_control_here.temperature_swaps(
                    coords_for_swap,
                    prev_logP.copy(),
                    prev_logl.copy(),
                    prev_logp.copy(),
                    branch_supps={self.branch_name: None}  # TODO: adjust this to be flexible
                )

                temperature_control_here.adapt_temps()

                new_state.branches_coords[self.branch_name][:, :, leaf] = coords_for_swap[self.branch_name][:, :, 0]

            # ll_tmp1 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            # add back cold chain sources
            xp.get_default_memory_pool().free_all_blocks()
            
            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)
            self.remove_cold_chain_sources(add_coords_in)
            
            # read out all betas from temperature controls
            new_state.sub_states[self.branch_name].betas_all[leaf] = temperature_control_here.betas[:]
            # print(leaf)

            # ll_tmp2 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

        # udpate at the end
        # new_state.log_like[(temp_inds_update, walker_inds_update)] = logl.flatten()
        # new_state.log_prior[(temp_inds_update, walker_inds_update)] = logp.flatten()
        print("before computing current likelihood. elapsed: ", time.time() - tic)
        current_ll = self.acs.likelihood()  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        print("after computing current likelihood. elapsed: ", time.time() - tic)
        xp.get_default_memory_pool().free_all_blocks()
        # TODO: add check with last used logl

        current_lp = self.priors[self.branch_name].logpdf(new_state.branches[self.branch_name].coords[0, :, :].reshape(-1, ndim)).reshape(new_state.branches[self.branch_name].shape[1:-1]).sum(axis=-1)

        new_state.log_like[0] = current_ll
        # new_state.log_prior[0] = current_lp
        xp.get_default_memory_pool().free_all_blocks()
        if not hasattr(self, "best_last_ll"):
            self.best_last_ll = current_ll.max()
            self.low_last_ll = current_ll.min()
        # print(self.branch_name, self.best_last_ll, current_ll.max(), current_ll.max() - self.best_last_ll)
        # print(current_ll.max(), self.best_last_ll, current_ll.min(), self.low_last_ll)
        self.best_last_ll = current_ll.max()
        self.low_last_ll = current_ll.min()

        self.temperature_control.swaps_accepted = self.temperature_controls[0].swaps_accepted
        
        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplimental)
        new_state.log_like[:] = self.acs.likelihood()  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
            
        # assert np.abs(new_state.log_like[0] - self.acs.get_ll(include_psd_info=True)).max() < 1e-4
        # breakpoint()
        return new_state, accepted

    def replace_residuals(self, old_state, new_state):
        fd = xp.asarray(self.acs.fd)
        old_contrib = [None, None]
        new_contrib = [None, None]
        for leaf in range(old_state.branches[self.branch_name].shape[-2]):
            removal_coords = old_state.branches[self.branch_name].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            removal_waveforms = self.waveform_gen(*removal_coords_in.T, fill=True, freqs=fd, **self.waveform_gen_kwargs).transpose(1, 0, 2)
            
            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)
            add_waveforms = self.waveform_gen(*add_coords_in.T, fill=True, freqs=fd, **self.waveform_gen_kwargs).transpose(1, 0, 2)

            if leaf == 0:
                old_contrib[0] = removal_waveforms[0]
                old_contrib[1] = removal_waveforms[1]
                new_contrib[0] = add_waveforms[0]
                new_contrib[1] = add_waveforms[1]
            else:
                old_contrib[0] += removal_waveforms[0]
                old_contrib[1] += removal_waveforms[1]
                new_contrib[0] += add_waveforms[0]
                new_contrib[1] += add_waveforms[1]
            
        self.acs.swap_out_in_base_data(old_contrib, new_contrib)
        xp.get_default_memory_pool().free_all_blocks()

