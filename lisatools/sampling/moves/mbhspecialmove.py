import numpy as np
import cupy as xp
from copy import deepcopy

from eryn.moves import RedBlueMove, StretchMove
from eryn.state import State

from lisatools.sampling.moves.skymodehop import SkyMove

from tqdm import tqdm


class MBHSpecialMove(RedBlueMove):
    def __init__(self, waveform_gen, fd, data_residuals, psd, num_repeats, transform_fn, mbh_priors, mbh_kwargs, moves, df, **kwargs):

        RedBlueMove.__init__(self, **kwargs)
        self.fd = fd
        self.data_residuals = data_residuals
        self.psd = psd
        self.waveform_gen = waveform_gen
        self.num_repeats = num_repeats
        self.transform_fn = transform_fn
        self.mbh_priors = mbh_priors
        self.mbh_kwargs = mbh_kwargs
        moves_tmp = [move[0] for move in moves]
        move_weights = [move[1] for move in moves]
        self.moves = moves_tmp
        self.move_weights = move_weights
        self.df = df

    def get_logl(self):
        return -1/2 * (4 * self.df * self.xp.sum((self.data_residuals[:2].conj() * self.data_residuals[:2]) / self.psd[:2], axis=-1)).get()

    def propose(self, model, state):

        new_state = deepcopy(state)

        # TODO: in TF, can we do multiple together?
        # TODO: switch to heterodyning

        ntemps, nwalkers, nleaves, ndim = new_state.branches["mbh"].shape

        temp_inds_base = np.repeat(np.arange(ntemps)[:, None], nwalkers, axis=-1)
        walker_inds_base = np.tile(np.arange(nwalkers), (ntemps, 1))

        start_leaf = np.random.randint(0, nleaves)
        for base_leaf in range(nleaves):
            leaf = (base_leaf + start_leaf) % nleaves
            # remove cold chain sources
            xp.get_default_memory_pool().free_all_blocks()
            removal_coords = new_state.branches["mbh"].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)

            removal_waveforms = self.waveform_gen(*removal_coords_in.T, fill=True, freqs=self.fd, **self.mbh_kwargs).transpose(1, 0, 2)
            assert removal_waveforms.shape == self.data_residuals.shape

            # TODO: fix T channel 
            # d - h -> need to add removal waveforms
            # ll_tmp1 = (-1/2 * 4 * self.df * xp.sum(self.data_residuals[:2].conj() * self.data_residuals[:2] / self.psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(self.psd[:2])), axis=(0, 2))).get()

            self.data_residuals[:2] += removal_waveforms[:2]
            del removal_waveforms
            xp.get_default_memory_pool().free_all_blocks()
            ll_tmp2 = (-1/2 * 4 * self.df * xp.sum(self.data_residuals[:2].conj() * self.data_residuals[:2] / self.psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(self.psd[:2])), axis=(0, 2))).get()

            old_coords = new_state.branches["mbh"].coords[:, :, leaf].reshape(-1, ndim)
            old_coords_in = self.transform_fn.both_transforms(old_coords)
            data_index = walker_inds_base.astype(np.int32)
            noise_index = walker_inds_base.astype(np.int32)

            self.waveform_gen.d_d = xp.asarray(-2 * np.tile(ll_tmp2, (ntemps, 1)).flatten())

            del ll_tmp2
            xp.get_default_memory_pool().free_all_blocks()
            d_d_store = self.waveform_gen.d_d.reshape(ntemps, nwalkers).get()

            # TODO: fix this
            prev_logl = self.waveform_gen.get_direct_ll(self.fd, self.data_residuals.flatten(), self.psd.flatten(), self.df, *old_coords_in.T, noise_index=noise_index, data_index=data_index, **self.mbh_kwargs).reshape((ntemps, nwalkers)).real.get()

            prev_logp = self.mbh_priors["mbh"].logpdf(old_coords).reshape((ntemps, nwalkers))

            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)
            
            # fix this need to compute prev_logl for all walkers
            xp.get_default_memory_pool().free_all_blocks()
            for repeat in range(self.num_repeats):

                # pick move
                move_here = self.moves[model.random.choice(np.arange(len(self.moves)), p=self.move_weights)]

                # Split the ensemble in half and iterate over these two halves.
                accepted = np.zeros((ntemps, nwalkers), dtype=bool)
                all_inds = np.tile(np.arange(nwalkers), (ntemps, 1))
                inds = all_inds % self.nsplits
                if self.randomize_split:
                    [np.random.shuffle(x) for x in inds]

                # prepare accepted fraction
                accepted_here = np.zeros((ntemps, nwalkers), dtype=bool)
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
                        new_state.branches["mbh"].coords[inds == j][:, leaf].reshape(ntemps, -1, 1, ndim)
                        for j in range(self.nsplits)
                    ]

                    old_points = sets[split].reshape((ntemps, nwalkers_here, ndim))
                    
                    # setup s and c based on splits
                    s = {"mbh": sets[split]}
                    c = {"mbh": sets[:split] + sets[split + 1 :]}
                    
                    # Get the move-specific proposal.
                    if isinstance(move_here, StretchMove):
                        q, factors = move_here.get_proposal(
                            s, c, model.random
                        )

                    else:
                        q, factors = move_here.get_proposal(s, model.random)

                    new_points = q["mbh"].reshape((ntemps, nwalkers_here, ndim))

                    # Compute prior of the proposed position
                    # new_inds_prior is adjusted if product-space is used
                    logp = self.mbh_priors["mbh"].logpdf(new_points.reshape(-1, ndim))

                    new_points_in = self.transform_fn.both_transforms(new_points.reshape(-1, ndim)[~np.isinf(logp)])
                    
                    # Compute the lnprobs of the proposed position.
                    data_index = walker_inds_here[~np.isinf(logp)].astype(np.int32)
                    noise_index = walker_inds_here[~np.isinf(logp)].astype(np.int32)

                    self.waveform_gen.d_d = xp.asarray(d_d_store[(temp_inds_here[~np.isinf(logp)], walker_inds_here[~np.isinf(logp)])])
                    
                    logl = np.full_like(logp, -1e300)

                    logl[~np.isinf(logp)] = self.waveform_gen.get_direct_ll(self.fd, self.data_residuals.flatten(), self.psd.flatten(), self.df, *new_points_in.T, noise_index=noise_index, data_index=data_index, **self.mbh_kwargs).real.get()

                    logl = logl.reshape(ntemps, nwalkers_here)
                    
                    logp = logp.reshape(ntemps, nwalkers_here)
                    prev_logp_here = prev_logp[inds == split].reshape(ntemps, nwalkers_here)

                    prev_logl_here = prev_logl[inds == split].reshape(ntemps, nwalkers_here)

                    prev_logP_here = self.compute_log_posterior(prev_logl_here, prev_logp_here)
                    logP = self.compute_log_posterior(logl, logp)

                    lnpdiff = factors + logP - prev_logP_here

                    keep = lnpdiff > np.log(model.random.rand(ntemps, nwalkers_here))

                    temp_inds_update = temp_inds_here[keep.flatten()]
                    walker_inds_update = walker_inds_here[keep.flatten()]

                    accepted[(temp_inds_update, walker_inds_update)] = True

                    # update state informatoin
                    new_state.branches["mbh"].coords[(temp_inds_update, walker_inds_update, np.full_like(walker_inds_update, leaf))] = new_points[keep].reshape(len(temp_inds_update), ndim)

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
                coords_for_swap = {"mbh": new_state.branches_coords["mbh"][:, :, leaf].copy()[:, :, None]}
                
                # TODO: check permute make sure it is okay
                coords_for_swap, prev_logP, prev_logl, prev_logp, inds, blobs, supps, branch_supps = self.temperature_control.temperature_swaps(
                    coords_for_swap,
                    prev_logP.copy(),
                    prev_logl.copy(),
                    prev_logp.copy(),
                    branch_supps={"mbh":None}
                )

                self.temperature_control.adapt_temps()

                new_state.branches_coords["mbh"][:, :, leaf] = coords_for_swap["mbh"][:, :, 0]

            # ll_tmp1 = -1/2 * 4 * self.df * xp.sum(self.data_residuals[:2].conj() * self.data_residuals[:2] / self.psd[:2], axis=(0, 2)).get()

            # add back cold chain sources
            xp.get_default_memory_pool().free_all_blocks()

            add_coords = new_state.branches["mbh"].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)

            add_waveforms = self.waveform_gen(*add_coords_in.T, fill=True, freqs=self.fd, **self.mbh_kwargs).transpose(1, 0, 2)
            assert add_waveforms.shape == self.data_residuals.shape

            # d - h -> need to subtract added waveforms
            self.data_residuals[:2] -= add_waveforms[:2]

            del add_waveforms
            xp.get_default_memory_pool().free_all_blocks()
            # ll_tmp2 = -1/2 * 4 * self.df * xp.sum(self.data_residuals[:2].conj() * self.data_residuals[:2] / self.psd[:2], axis=(0, 2)).get()

            # print(leaf)

            # ll_tmp2 = -1/2 * 4 * self.df * xp.sum(self.data_residuals[:2].conj() * self.data_residuals[:2] / self.psd[:2], axis=(0, 2)).get()

        # udpate at the end
        # new_state.log_like[(temp_inds_update, walker_inds_update)] = logl.flatten()
        # new_state.log_prior[(temp_inds_update, walker_inds_update)] = logp.flatten()

        current_ll = (-1/2 * 4 * self.df * xp.sum(self.data_residuals[:2].conj() * self.data_residuals[:2] / self.psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(self.psd[:2])), axis=(0, 2))).get()
        xp.get_default_memory_pool().free_all_blocks()
        # TODO: add check with last used logl

        current_lp = self.mbh_priors["mbh"].logpdf(new_state.branches["mbh"].coords[0, :, :].reshape(-1, ndim)).reshape(new_state.branches["mbh"].shape[1:-1]).sum(axis=-1)

        new_state.log_like[0] = current_ll
        new_state.log_prior[0] = current_lp
        xp.get_default_memory_pool().free_all_blocks()
        if not hasattr(self, "best_last_ll"):
            self.best_last_ll = current_ll.max()
        # print("mbh", self.best_last_ll, current_ll.max(), current_ll.max() - self.best_last_ll)
        self.best_last_ll = current_ll.max()
        return new_state, accepted