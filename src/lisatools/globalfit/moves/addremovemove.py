import logging
import time
from copy import deepcopy
from typing import Callable

import cupy as xp
import numpy as np
from eryn.moves import Move, StretchMove, TemperatureControl
from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer
# from bbhx.likelihood import NewHeterodynedLikelihood
from tqdm import tqdm

# from lisatools.globalfit.state import GFState
# from lisatools.sampling.moves.skymodehop import SkyMove
from ...analysiscontainer import AnalysisContainerArray
from .globalfitmove import GlobalFitMove

logger = logging.getLogger(__name__)


class ResidualAddOneRemoveOneMove(GlobalFitMove, StretchMove, Move):
    """
    Move that handles adding and removing sources to and from the residuals stored in the analysis container array.
    This is done by first removing the contribution of the current sources in the cold chain from the residual, 
    then proposing new sources for this leaf, and then adding back in the contribution of the new sources to the residual.
    This way we can make sure that the likelihoods are computed correctly for each proposed source and that the likelihoods are consistent with the current state of the residuals in the analysis container array.


    """

    def __init__(
        self,
        branch_name: str,
        coords_shape: tuple,
        waveform_gen: Callable,
        waveform_gen_kwargs: dict,
        waveform_like_kwargs: dict,
        acs: AnalysisContainerArray,
        num_repeats: int,
        transform_fn: TransformContainer,
        priors: ProbDistContainer,
        inner_moves: list,
        Tmax: float = np.inf,
        betas_all: np.ndarray = None,
        **kwargs,
    ):

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
        # self.df = acs.df
        # # get data frequency array on gpu
        # self.fd = xp.asarray(acs.f_arr)

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
                skip_swap_branches=None,  # will fill in after first run through move
            )

    def check_add_skip_swap_info(self, state):

        if self.temperature_controls[0].skip_swap_branches is not None:
            return

        if len(state.branches) > 1:
            skip_swap_branches = [key for key in state.branches.keys()]
            skip_swap_branches.remove(self.branch_name)

        else:
            skip_swap_branches = []

        for i in range(self.nleaves_max):
            self.temperature_controls[i].skip_swap_branches = skip_swap_branches

    def add_back_in_cold_chain_sources(self, coords):
        """
        Remove the contribution of the current sources in the cold chain from the residual.

        Args:
            coords: coordinates of the sources in the cold chain that we want to add back in to the residual.
        """

        # TODO: fix T channel
        # d - h -> need to add removal waveforms
        # ll_tmp1 = (-1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        removal_waveforms = self.get_waveform_here(coords)
        ll_tmp2 = self.acs.likelihood(
            source_only=True
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        self.acs.remove_signal_from_residual(
            removal_waveforms, data_index=None, start_index=None
        )
        del removal_waveforms
        xp.get_default_memory_pool().free_all_blocks()

        ll_tmp3 = self.acs.likelihood(
            source_only=True
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

    def remove_cold_chain_sources(self, coords):
        """
        Add the contribution of the current sources in the cold chain from the residual.

        Args:
            coords: coordinates of the sources in the cold chain that we want to remove from the residual.
        """

        # TODO: fix T channel
        # d - h -> need to add removal waveforms
        # ll_tmp1 = (-1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        removal_waveforms = self.get_waveform_here(coords)
        ll_tmp2 = self.acs.likelihood(
            source_only=True
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        self.acs.add_signal_to_residual(
            removal_waveforms, data_index=None, start_index=None
        )
        del removal_waveforms
        xp.get_default_memory_pool().free_all_blocks()

        ll_tmp3 = self.acs.likelihood(
            source_only=True
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

    def get_waveform_here(self, coords: np.ndarray):
        """
        Get the waveform for the given coordinates.

        Args:
            coords: coordinates of the sources for which we want to get the waveform. Shape is (n_sources, ndim).

        Returns:    
            waveforms: waveforms for the given coordinates. Shape is (n_sources, n_channels, (data_shape)).
        """
    
        xp.get_default_memory_pool().free_all_blocks()
        # change this to be more memory efficient if some waveforms are shorter than the full data array
        # waveforms = xp.zeros(
        #     (coords.shape[0], self.acs.nchannels, self.acs.data_length), dtype=complex
        # )

        # for i in range(coords.shape[0]):
        #     waveforms[i] = self.waveform_gen(*coords[i], **self.waveform_gen_kwargs)

        waveforms = []
        for i in range(coords.shape[0]):
            waveforms.append(
                self.waveform_gen(*coords[i], **self.waveform_gen_kwargs)
            )
        
        # now concatenate along axis to get shape (n_sources, n_channels, data_length)
        waveforms = xp.stack(waveforms, axis=0) 

        return waveforms

    def setup_likelihood_here(self, coords):
        pass

    def compute_like(self, coords_in, data_index):
        """
        Compute the likelihood for the given coordinates and data index.

        Args:
            coords_in: coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        # TODO: we should probably move the prior in here even though
        # in general with current setup it should only be points in the prior
        # that make it here
        ll = np.full_like(data_index.get(), -1e300, dtype=float)

        for i, (coords_in_now, data_index_now) in enumerate(
            zip(coords_in, data_index.get())
        ):
            ll[i] = self.acs[data_index_now].calculate_signal_likelihood(
                *coords_in_now,
                waveform_kwargs=self.waveform_like_kwargs,
                signal_gen=self.waveform_gen,
            )

        return ll

    def setup(self, model, state):
        return

    def log_like_for_fancy_swaping(self, x, supps=None, branch_supps=None, **kwargs):
        """
        Compute the log likelihood for the given coordinates and data index for use in fancy swapping. 
        This is needed because when permuting the coordinates during tempering, we need to recompute the likelihood against the new set of residuals and covariance matrix.

        Args:
            x: Dictionary of coordinates of the sources for which we want to compute the likelihood. 
                The coordinates are expected to be in the shape (ntemps, nwalkers, nleaves_max, ndim).
            supps: supplimental information for the likelihood computation. #todo add
            branch_supps: Branch supplimental. #todo add

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (ntemps, nwalkers).
            blobs: blobs for the given coordinates and data index. Default is None.
        """
        assert (
            x[self.branch_name].ndim == 4
            and x[self.branch_name].shape[1] == self.nwalkers
        )
        # shape is (nwalkers, 1 (nleaves_max), ndim)
        ntemps = x[self.branch_name].shape[0]

        coords = x[self.branch_name].reshape(-1, x[self.branch_name].shape[-1])
        data_index_in = (
            xp.tile(xp.arange(self.nwalkers), (ntemps, 1)).flatten().astype(xp.int32)
        )

        coords_in = self.transform_fn.both_transforms(coords)

        # TODO: need to be careful here when heterodyning about if it is "close"
        output = (
            self.compute_like(
                coords_in,
                data_index=data_index_in,
            )
            .reshape((ntemps, self.nwalkers))
            .real
        )
        return output, None  # AS: match psd? I'm not sure

    def propose(self, model, state):
        logger.debug("PROPOSING")
        logger.debug("------" * 20)

        self.setup(model, state)
        tic = time.time()

        if not np.any(state.branches[self.branch_name].inds):
            ntemps, nwalkers = state.branches[self.branch_name].shape[:2]
            _accepted = np.zeros((ntemps, nwalkers), dtype=bool)
            return state, _accepted

        new_state = deepcopy(state)

        self.acs = model.analysis_container_arr
        self.check_add_skip_swap_info(state)

        # mapping information
        temp_inds_base = np.repeat(
            np.arange(self.ntemps)[:, None], self.nwalkers, axis=-1
        )
        walker_inds_base = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))

        # randomize order
        leaves_random_order = np.random.permutation(np.arange(self.nleaves_max))
        for leaf in leaves_random_order:
            logger.debug(f"Processing leaf {leaf}")

            # guard against leaves with False
            assert np.all(
                state.branches[self.branch_name].inds[0, 0, leaf]
                == state.branches[self.branch_name].inds[:, :, leaf]
            )
            if not state.branches[self.branch_name].inds[0, 0, leaf]:
                continue
            # second step of randomizing order (making sure it does not run over)

            # fill this temperature control with temperatures from current state
            temperature_control_here = self.temperature_controls[leaf]

            temperature_control_here.betas[:] = new_state.sub_states[
                self.branch_name
            ].betas_all[leaf][
                : self.ntemps
            ]  # as: make sure only local ntemps are used
            ntemps_full = (
                new_state.sub_states[self.branch_name].betas_all[leaf].shape[0]
            )

            ndim = new_state.branches[self.branch_name].coords.shape[-1]

            # remove cold chain sources
            removal_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            self.add_back_in_cold_chain_sources(removal_coords_in)

            self.setup_likelihood_here(removal_coords_in)

            old_coords = (
                new_state.branches[self.branch_name]
                .coords[: self.ntemps, :, leaf]
                .reshape(-1, ndim)
            )
            old_coords_in = self.transform_fn.both_transforms(old_coords)

            data_index_in = (
                xp.tile(xp.arange(self.nwalkers), (self.ntemps, 1))
                .flatten()
                .astype(xp.int32)
            )
            # TODO: fix this
            # prev_logl = self.waveform_gen.get_direct_ll(fd, data_residuals.flatten(), psd.flatten(), self.df, *old_coords_in.T, noise_index=noise_index, data_index=data_index, **self.waveform_kwargs).reshape((ntemps, nwalkers)).real.get()
            # TODO: check if psd term is included properly here at each step
            # TODO: and check data index here
            prev_logl = (
                self.compute_like(
                    old_coords_in,
                    data_index=data_index_in,
                )
                .reshape((self.ntemps, self.nwalkers))
                .real
            )

            logger.debug(f"prev_logl: {prev_logl}. elapsed: {time.time() - tic}")

            prev_logp = (
                self.priors[self.branch_name]
                .logpdf(old_coords)
                .reshape((self.ntemps, self.nwalkers))
            )

            prev_logP = temperature_control_here.compute_log_posterior_tempered(
                prev_logl, prev_logp
            )

            # fix this need to compute prev_logl for all walkers
            xp.get_default_memory_pool().free_all_blocks()
            for repeat in tqdm(range(self.num_repeats)):

                # pick move
                move_here = self.moves[
                    model.random.choice(np.arange(len(self.moves)), p=self.move_weights)
                ]

                # Split the ensemble in half and iterate over these two halves.
                accepted = np.zeros((ntemps_full, self.nwalkers), dtype=bool)
                all_inds = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))
                inds = all_inds % self.nsplits
                if self.randomize_split:
                    [np.random.shuffle(x) for x in inds]

                # prepare accepted fraction
                # accepted_here = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
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
                        new_state.branches[self.branch_name]
                        .coords[: self.ntemps][inds == j][:, leaf]
                        .reshape(self.ntemps, -1, 1, ndim)
                        for j in range(self.nsplits)
                    ]

                    old_points = sets[split].reshape((self.ntemps, nwalkers_here, ndim))

                    # setup s and c based on splits
                    s = {self.branch_name: sets[split]}
                    c = {self.branch_name: sets[:split] + sets[split + 1 :]}

                    # Get the move-specific proposal.
                    if isinstance(move_here, StretchMove):
                        q, factors = move_here.get_proposal(s, c, model.random)

                    else:
                        q, factors = move_here.get_proposal(s, model.random)

                    new_points = q[self.branch_name].reshape(
                        (self.ntemps, nwalkers_here, ndim)
                    )

                    # Compute prior of the proposed position
                    # new_inds_prior is adjusted if product-space is used
                    logp = self.priors[self.branch_name].logpdf(
                        new_points.reshape(-1, ndim)
                    )

                    new_points_in = self.transform_fn.both_transforms(
                        new_points.reshape(-1, ndim)[~np.isinf(logp)]
                    )

                    # Compute the lnprobs of the proposed position.
                    data_index = xp.asarray(
                        walker_inds_here[~np.isinf(logp)].astype(np.int32)
                    )
                    # noise_index = walker_inds_here[~np.isinf(logp)].astype(np.int32)

                    # self.waveform_gen.d_d = xp.asarray(d_d_store[(temp_inds_here[~np.isinf(logp)], walker_inds_here[~np.isinf(logp)])])

                    logl = np.full_like(logp, -1e300)

                    # logl[~np.isinf(logp)] = self.waveform_gen.get_direct_ll(fd, data_residuals.flatten(), psd.flatten(), self.df, *new_points_in.T, noise_index=noise_index, data_index=data_index, **self.waveform_kwargs).real.get()
                    logl[~np.isinf(logp)] = self.compute_like(
                        new_points_in,
                        data_index=data_index,
                        # constants_index=data_index,
                    )

                    # print(f"new logl: {logl}. elapsed: {time.time() - tic}")

                    logl = logl.reshape(self.ntemps, nwalkers_here)

                    logp = logp.reshape(self.ntemps, nwalkers_here)
                    prev_logp_here = prev_logp[inds == split].reshape(
                        self.ntemps, nwalkers_here
                    )

                    prev_logl_here = prev_logl[inds == split].reshape(
                        self.ntemps, nwalkers_here
                    )

                    prev_logP_here = (
                        temperature_control_here.compute_log_posterior_tempered(
                            prev_logl_here, prev_logp_here
                        )
                    )
                    logP = temperature_control_here.compute_log_posterior_tempered(
                        logl, logp
                    )

                    lnpdiff = factors + logP - prev_logP_here

                    keep = lnpdiff > np.log(
                        model.random.rand(self.ntemps, nwalkers_here)
                    )

                    temp_inds_update = temp_inds_here[keep.flatten()]
                    walker_inds_update = walker_inds_here[keep.flatten()]

                    accepted[: self.ntemps][
                        (temp_inds_update, walker_inds_update)
                    ] = True

                    # update state informatoin
                    new_state.branches[self.branch_name].coords[
                        (
                            temp_inds_update,
                            walker_inds_update,
                            np.full_like(walker_inds_update, leaf),
                        )
                    ] = new_points[keep].reshape(len(temp_inds_update), ndim)

                    prev_logl[(temp_inds_update, walker_inds_update)] = logl[
                        keep
                    ].flatten()
                    prev_logp[(temp_inds_update, walker_inds_update)] = logp[
                        keep
                    ].flatten()
                    prev_logP[(temp_inds_update, walker_inds_update)] = logP[
                        keep
                    ].flatten()

                # acceptance tracking
                self.accepted += accepted
                # print(self.accepted[0])
                self.num_proposals += 1

                # TODO: include PSD likelihood in swaps?
                # temperature swaps
                # make swaps
                coords_for_swap = {
                    self.branch_name: new_state.branches_coords[self.branch_name][
                        :, :, leaf
                    ].copy()[:, :, None]
                }

                # TODO: make adjustable rate of fancy swaps
                # fancy_swap = (repeat % 20 == 0)
                fancy_swap = False

                compute_log_like = self.log_like_for_fancy_swaping

                # TODO: check permute make sure it is okay
                (
                    coords_for_swap,
                    prev_logP,
                    prev_logl,
                    prev_logp,
                    inds,
                    blobs,
                    supps,
                    branch_supps,
                ) = temperature_control_here.temperature_swaps(
                    coords_for_swap,
                    prev_logP.copy(),
                    prev_logl.copy(),
                    prev_logp.copy(),
                    branch_supps={
                        self.branch_name: None
                    },  # TODO: adjust this to be flexible
                    fancy_swap=fancy_swap,
                    compute_log_like=compute_log_like,
                    permute_here=True,
                )

                temperature_control_here.adapt_temps()

                new_state.branches_coords[self.branch_name][:, :, leaf] = (
                    coords_for_swap[self.branch_name][:, :, 0]
                )

            # ll_tmp1 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            # add back cold chain sources
            xp.get_default_memory_pool().free_all_blocks()

            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)
            self.remove_cold_chain_sources(add_coords_in)

            # read out all betas from temperature controls
            new_state.sub_states[self.branch_name].betas_all[leaf][
                : self.ntemps
            ] = temperature_control_here.betas
            # print(leaf)

            # ll_tmp2 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

        # udpate at the end
        # new_state.log_like[(temp_inds_update, walker_inds_update)] = logl.flatten()
        # new_state.log_prior[(temp_inds_update, walker_inds_update)] = logp.flatten()
        # print("before computing current likelihood. elapsed: ", time.time() - tic)
        current_ll = (
            self.acs.likelihood()
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        # print("after computing current likelihood. elapsed: ", time.time() - tic)
        xp.get_default_memory_pool().free_all_blocks()
        # TODO: add check with last used logl

        current_lp = (
            self.priors[self.branch_name]
            .logpdf(
                new_state.branches[self.branch_name].coords[0, :, :].reshape(-1, ndim)
            )
            .reshape(new_state.branches[self.branch_name].shape[1:-1])
            .sum(axis=-1)
        )

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

        if self.temperature_control is None:
            # this really does not matter
            self.temperature_control = self.temperature_controls[0]

        self.temperature_control.swaps_accepted = self.temperature_controls[
            0
        ].swaps_accepted

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplimental)
        # breakpoint()
        new_state.log_like[:] = (
            self.acs.likelihood()
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

        # assert np.abs(new_state.log_like[0] - self.acs.get_ll(include_psd_info=True)).max() < 1e-4
        # breakpoint()
        return new_state, accepted

    def replace_residuals(self, old_state, new_state):
        raise NotImplementedError
        fd = xp.asarray(self.acs.fd)
        old_contrib = [None, None]
        new_contrib = [None, None]
        for leaf in range(old_state.branches[self.branch_name].shape[-2]):
            removal_coords = old_state.branches[self.branch_name].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            removal_waveforms = self.waveform_gen(
                *removal_coords_in.T, fill=True, freqs=fd, **self.waveform_gen_kwargs
            ).transpose(1, 0, 2)

            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)
            add_waveforms = self.waveform_gen(
                *add_coords_in.T, fill=True, freqs=fd, **self.waveform_gen_kwargs
            ).transpose(1, 0, 2)

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
