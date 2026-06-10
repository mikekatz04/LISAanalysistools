from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any, Callable, TYPE_CHECKING

from lisatools.datacontainer import DataResidualArray


try:
    import cupy as xp
except (ImportError, ModuleNotFoundError):
    import numpy as xp 
import numpy as np
from eryn.moves import Move, StretchMove, TemperatureControl
from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer

from tqdm import tqdm

from ...analysiscontainer import AnalysisContainerArray
from ...domaincomputation import DomainComputationGroupArray
from ...domains import DomainBase, DomainBaseArray
from .globalfitmove import GlobalFitMove
from .multigpumove import MultiGPUMoveBase

logger = logging.getLogger(__name__)
DEBUG_MODE = False

if TYPE_CHECKING:
    from ...sources.waveformbase import TDWaveformBase
    from ...domaincomputation import DomainComputationGroupArray
    
class ResidualAddOneRemoveOneMove(GlobalFitMove, StretchMove, Move):
    """
    Move that handles adding and removing sources to and from the residuals stored in the analysis container array.
    This is done by first removing the contribution of the current sources in the cold chain from the residual,
    then proposing new sources for this leaf, and then adding back in the contribution of the new sources to the residual.
    This way we can make sure that the likelihoods are computed correctly for each proposed source and that the likelihoods are consistent with the current state of the residuals in the analysis container array.

    Args:
        branch_name: name of the branch that this move will operate on.
        coords_shape: shape of the coordinates of the sources in the branch that this move will operate on.
        waveform_gen: function that generates the waveforms for the sources given their coordinates.
        waveform_gen_kwargs: keyword arguments for the waveform generator function.
        waveform_like_kwargs: keyword arguments for the likelihood computation function.
        acs: analysis container array that contains the residuals and other information needed for the likelihood computation.
        num_repeats: number of times to repeat the proposal step for each leaf.
        transform_fn: transform container that contains the transforms to be applied to the coordinates before generating waveforms and computing likelihoods.
        priors: prior distribution container that contains the prior distributions for the sources in the branch.
        inner_moves: list of moves and their corresponding weights to be used for proposing new sources for the leaf.
        Tmax: maximum temperature for the temperature control.
        betas_all: array of betas for all leaves and temperatures. Shape is (nleaves_max, ntemps). If None, betas will be initialized as in TemperatureControl.
        permute_every: number of repeats after which to permute the walkers during a temperature swap. This helps with the mixing of the chains.
        pad_out_of_prior: whether to pad proposed sources that are out of the prior bounds to avoid JIT compilation issues. If True, proposed sources that are out of the prior bounds will be replaced with the first in-prior point. 
        **kwargs: additional keyword arguments for the Move class.
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
        permute_every: int = 20,
        pad_out_of_prior: bool = False,
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
        moves_tmp = [move[0] if isinstance(move, tuple) else move for move in inner_moves]
        move_weights = [move[1] if isinstance(move, tuple) else 1.0 for move in inner_moves]
        self.moves = moves_tmp
        self.move_weights = move_weights / np.sum(move_weights)

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
        
        self.permute_every = permute_every
        self.pad_out_of_prior = pad_out_of_prior
        
        # make sure to propagate the periodic information to the inner moves if it is included in kwargs
        if 'periodic' in kwargs:
            self.periodic = kwargs['periodic']

    @property
    def periodic(self):
        return self._periodic
    
    @periodic.setter
    def periodic(self, periodic):
        self._periodic = periodic
        if periodic is not None:
            for tmp_move in self.moves:
                if tmp_move.periodic is None:
                    tmp_move.periodic = periodic

    def free_gpu_memory(self):
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()

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
        # d - h -> need to add removal waveforms
        # ll_tmp1 = (-1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)) - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        removal_waveforms = self.get_waveform_here(coords)
        # ll_tmp2 = self.acs.likelihood(
        #     source_only=True
        # )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        self.acs.remove_signal_from_residual(removal_waveforms, data_index=None)

        self.acs.synchronize()                                                                                                                                                                                                                                              
                
        del removal_waveforms
        #if xp is not np:
        self.free_gpu_memory()

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
        # ll_tmp2 = self.acs.likelihood(
        #     source_only=True
        # )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        self.acs.add_signal_to_residual(removal_waveforms, data_index=None)
        
        self.acs.synchronize()

        del removal_waveforms
        #if xp is not np:
        self.free_gpu_memory()

        # ll_tmp3 = self.acs.likelihood(
        #     source_only=True
        # )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

    def get_waveform_here(self, coords: np.ndarray) -> DomainBaseArray | list[DomainBase]:
        """Get the waveforms for the given source coordinates.

        Each call to ``waveform_gen`` returns a :class:`~lisatools.domains.DomainBase`
        (e.g. :class:`~lisatools.domains.STFTSignal` or
        :class:`~lisatools.domains.FDSignal`).  The results are collected into a
        :class:`~lisatools.domains.DomainBaseArray`, which batches them for
        vectorized downstream operations when all signals share the same
        domain settings.

        Args:
            coords: Source coordinates, shape ``(n_sources, ndim)``.

        Returns:
            :class:`~lisatools.domains.DomainBaseArray` of length ``n_sources``.

        """
        #if xp is not np:
        self.free_gpu_memory()

        waveforms = []
        for i in range(coords.shape[0]):
            waveforms.append(self.waveform_gen(*coords[i], **self.waveform_gen_kwargs))

        return DomainBaseArray(waveforms)

    def setup_likelihood_here(self, coords):
        pass

    def compute_acs_like(self, coords_in, data_index, signal_gen, **kwargs):
        """
        Compute the likelihood for the given coordinates and data index using the analysis container array.

        Args:
            coords_in: coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).
            signal_gen: waveform generator function to use for computing the likelihood. This is needed because in some cases we need to compute the likelihood with a different waveform generator than the one used for proposing new sources, for example when using heterodyned likelihoods.
            kwargs: additional keyword arguments for the likelihood computation function.

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        # TODO: we should probably move the prior in here even though
        # in general with current setup it should only be points in the prior
        # that make it here
        ll = np.full_like(data_index, -1e300, dtype=float)
        #data_index = xp.asarray(data_index.astype(np.int32)) # make sure data index is on the same device as the likelihood computation
        source_only = kwargs.pop("source_only", False)
        all_templates = signal_gen(*coords_in.T, **self.waveform_gen_kwargs)
        for i in range(coords_in.shape[0]):
            ll[i] = self.acs[data_index[i]].template_likelihood(
                DataResidualArray(all_templates[i]),
                include_psd_info=not source_only,
                **kwargs,
            )
        # for i, (coords_in_now, data_index_now) in enumerate(zip(coords_in, data_index)):
        #     ll[i] = self.acs[data_index_now].calculate_signal_likelihood(
        #         *coords_in_now,
        #         waveform_kwargs=self.waveform_gen_kwargs,
        #         signal_gen=signal_gen,
        #         **kwargs,
        #     )

        return ll

    def compute_like(self, coords_in, data_index):
        """
        Compute the likelihood for the given coordinates and data index.

        Args:
            coords_in: coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        return self.compute_acs_like(coords_in, data_index, signal_gen=self.waveform_gen, **self.waveform_like_kwargs)


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
        assert x[self.branch_name].ndim == 4 and x[self.branch_name].shape[1] == self.nwalkers
        # shape is (nwalkers, 1 (nleaves_max), ndim)
        ntemps = x[self.branch_name].shape[0]

        coords = x[self.branch_name].reshape(-1, x[self.branch_name].shape[-1])
        data_index_in = np.tile(np.arange(self.nwalkers), (ntemps, 1)).flatten().astype(np.int32)

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

    def get_split_inds(self):
        all_inds = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))
        inds = all_inds % self.nsplits
        if self.randomize_split:
            [np.random.shuffle(x) for x in inds]

        return inds

    def propose(self, model, state):

        self.setup(model, state)
        tic = time.time()

        if not np.any(state.branches[self.branch_name].inds):
            ntemps, nwalkers = state.branches[self.branch_name].shape[:2]
            _accepted = np.zeros((ntemps, nwalkers), dtype=bool)
            return state, _accepted

        new_state = deepcopy(state)

        # self.acs = model.analysis_container_arr
        self.check_add_skip_swap_info(state)

        # mapping information
        temp_inds_base = np.repeat(np.arange(self.ntemps)[:, None], self.nwalkers, axis=-1)
        walker_inds_base = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))

        # randomize order
        leaves_random_order = np.random.permutation(np.arange(self.nleaves_max))
        for leaf in leaves_random_order:
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

            temperature_control_here.betas[:] = new_state.sub_states[self.branch_name].betas_all[
                leaf
            ][
                : self.ntemps
            ]  # as: make sure only local ntemps are used
            ntemps_full = new_state.sub_states[self.branch_name].betas_all[leaf].shape[0]

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
                np.tile(np.arange(self.nwalkers), (self.ntemps, 1)).flatten().astype(np.int32)
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

            # if hasattr(self, "waveform_gen_method"):
            #     signal_gen = getattr(self.waveform_gen, self.waveform_gen_method)
            # else:
            #     signal_gen = self.waveform_gen
            
            # acs_like_here = self.compute_acs_like(old_coords_in, data_index=data_index_in, signal_gen=signal_gen, source_only=True).reshape((self.ntemps, self.nwalkers)).real
            # diff = prev_logl - acs_like_here

            # if np.any(np.abs(diff) > 1e-1):
            #         logger.warning(f"acs likelihood: {acs_like_here.flatten()}. proposed likelihood: {prev_logl.flatten()}. This could be a sign of numerical issues.")
            #         if DEBUG_MODE:
            #             breakpoint()
            #         else:
            #             raise ValueError(f"Large difference in log likelihood encountered: {np.abs(diff).max()}. This could be a sign of numerical issues.")

            if np.any(prev_logl < -1e10) or np.any(prev_logl > 1e30):
                logger.warning(f"Very low log likelihood encountered in propose: min = {prev_logl.min()}, max = {prev_logl.max()}. This could be a sign of numerical issues.")
                if DEBUG_MODE:
                    breakpoint()

            prev_logp = (
                self.priors[self.branch_name]
                .logpdf(old_coords)
                .reshape((self.ntemps, self.nwalkers))
            )

            prev_logP = temperature_control_here.compute_log_posterior_tempered(
                prev_logl, prev_logp
            )

            # fix this need to compute prev_logl for all walkers
            self.free_gpu_memory()
            for repeat in tqdm(range(self.num_repeats), desc=f"{self.branch_name} update, leaf {leaf}"):

                # pick move
                move_here = self.moves[
                    model.random.choice(np.arange(len(self.moves)), p=self.move_weights)
                ]

                # logger.debug(f"move here: {move_here.__class__.__name__}")

                # Split the ensemble in half and iterate over these two halves.
                accepted = np.zeros((ntemps_full, self.nwalkers), dtype=bool)
                inds = self.get_split_inds()

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

                    new_points = q[self.branch_name].reshape((self.ntemps, nwalkers_here, ndim))

                    # Compute prior of the proposed position
                    # new_inds_prior is adjusted if product-space is used
                    logp = self.priors[self.branch_name].logpdf(new_points.reshape(-1, ndim))
                    in_prior = ~np.isinf(logp)
                    logl = np.full_like(logp, -1e300)

                    if np.any(in_prior):
                        if self.pad_out_of_prior and np.any(~in_prior):
                            padded = new_points.reshape(-1, ndim).copy()
                            padded[~in_prior] = new_points.reshape(-1, ndim)[in_prior][0]
                            new_points_in = self.transform_fn.both_transforms(padded)

                            data_index = np.asarray(walker_inds_here.astype(np.int32))

                            all_logl = self.compute_like(new_points_in, data_index=data_index)
                            logl = np.where(in_prior, all_logl, -1e300)

                        else:
                            new_points_in = self.transform_fn.both_transforms(
                                new_points.reshape(-1, ndim)[in_prior]
                            )

                            # Compute the lnprobs of the proposed position.
                            data_index = np.asarray(walker_inds_here[in_prior].astype(np.int32))
                
                            logl[in_prior] = self.compute_like(
                                    new_points_in,
                                    data_index=data_index,
                                )
                    
                    if DEBUG_MODE:
                        logger.debug(f"average proposed logl: {logl[in_prior].mean()}.")
    
                    if np.any(logl[in_prior] < -1e10) or np.any(logl[in_prior] > 1e30):
                        logger.warning(f"Suspicious likelihood encountered in propose: min = {logl[~np.isinf(logp)].min()}, max = {logl[~np.isinf(logp)].max()}. This could be a sign of numerical issues.")
                        if DEBUG_MODE:
                            breakpoint()
                    # print(f"new logl: {logl}. elapsed: {time.time() - tic}")

                    logl = logl.reshape(self.ntemps, nwalkers_here)

                    logp = logp.reshape(self.ntemps, nwalkers_here)
                    prev_logp_here = prev_logp[inds == split].reshape(self.ntemps, nwalkers_here)

                    prev_logl_here = prev_logl[inds == split].reshape(self.ntemps, nwalkers_here)

                    prev_logP_here = temperature_control_here.compute_log_posterior_tempered(
                        prev_logl_here, prev_logp_here
                    )
                    logP = temperature_control_here.compute_log_posterior_tempered(logl, logp)

                    lnpdiff = factors + logP - prev_logP_here

                    keep = lnpdiff > np.log(model.random.rand(self.ntemps, nwalkers_here))

                    temp_inds_update = temp_inds_here[keep.flatten()]
                    walker_inds_update = walker_inds_here[keep.flatten()]

                    accepted[: self.ntemps][(temp_inds_update, walker_inds_update)] = True

                    # update state information
                    new_state.branches[self.branch_name].coords[
                        (
                            temp_inds_update,
                            walker_inds_update,
                            np.full_like(walker_inds_update, leaf),
                        )
                    ] = new_points[keep].reshape(len(temp_inds_update), ndim)

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
                coords_for_swap = {
                    self.branch_name: new_state.branches_coords[self.branch_name][
                        :, :, leaf
                    ].copy()[:, :, None]
                }

                fancy_swap = (repeat % self.permute_every == 0) and (repeat > 0)
                #if fancy_swap:
                    # logger.debug(f"Permuting walkers before swap.")
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
                    branch_supps={self.branch_name: None},  # TODO: adjust this to be flexible
                    fancy_swap=fancy_swap,
                    compute_log_like=compute_log_like,
                    permute_here=fancy_swap,
                )

                temperature_control_here.adapt_temps()

                new_state.branches_coords[self.branch_name][:, :, leaf] = coords_for_swap[
                    self.branch_name
                ][:, :, 0]

            # ll_tmp1 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            # add back cold chain sources
            self.free_gpu_memory()

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
        if np.any(current_ll < 0.0):
            logger.warning(f"The ACS likelihood should always be positive given the psd contribution, but got {current_ll.min()}")
            logger.warning(f"The minimum proposed likelihood was {prev_logl.min()}.")
            if DEBUG_MODE:
                breakpoint()
            # else:
            #     raise ValueError(f"The ACS likelihood should always be positive given the psd contribution, but got {current_ll.min()}")

        # TODO: add check with last used logl

        current_lp = (
            self.priors[self.branch_name]
            .logpdf(new_state.branches[self.branch_name].coords[0, :, :].reshape(-1, ndim))
            .reshape(new_state.branches[self.branch_name].shape[1:-1])
            .sum(axis=-1)
        )

        new_state.log_like[0] = current_ll
        # new_state.log_prior[0] = current_lp
        self.free_gpu_memory()
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

        self.temperature_control.swaps_accepted = self.temperature_controls[0].swaps_accepted

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplimental)
        # breakpoint()
        new_state.log_like[:] = (
            current_ll #self.acs.likelihood()
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

        self.free_gpu_memory()

        # assert np.abs(new_state.log_like[0] - self.acs.get_ll(include_psd_info=True)).max() < 1e-4
        # breakpoint()
        logger.debug(f"mean accepted fraction: {np.mean(self.accepted[0] / self.num_proposals)}. elapsed: {time.time() - tic}")
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
        self.free_gpu_memory()

class MultiGPUResidualAddRemoveMove(ResidualAddOneRemoveOneMove, MultiGPUMoveBase):
    """
    Wrapper around ResidualAddOneRemoveOneMove that runs the waveform generation and likelihood computation on multiple GPUs.

    Args:
    dcga: DomainComputationGroupArray that contains the information about the domain computation groups and the GPUs to use for each group.
    waveform_gen: waveform generator class that generates the waveforms for the sources given their coordinates.
    branch_name: name of the branch that this move will operate on.
    coords_shape: shape of the coordinates of the sources in the branch that this move will operate on.
    waveform_gen_method: name of the method of the waveform generator class to use for generating the waveforms for residual operations.
    waveform_gen_kwargs: keyword arguments for the waveform generator method.
    waveform_like_kwargs: keyword arguments for the likelihood computation function.
    num_repeats: number of times to repeat the proposal step for each leaf.
    transform_fn: transform container that contains the transforms to be applied to the coordinates before generating waveforms and computing likelihoods.
    priors: prior distribution container that contains the prior distributions for the sources in the branch.
    inner_moves: list of moves and their corresponding weights to be used for proposing new sources for each leaf.
    Tmax: maximum temperature for the temperature control.
    betas_all: array of betas for all leaves and temperatures. Shape is (nleaves_max, ntemps). If None, betas will be initialized as in TemperatureControl.
    permute_every: number of repeats after which to permute the walkers during a temperature swap. 
    pad_out_of_prior: whether to pad proposed sources that are out of the prior bounds to avoid JIT compilation issues. If True, proposed sources that are out of the prior bounds will be replaced with the first in-prior point. 
    run_async: whether to run the waveform generation and likelihood computation asynchronously for each GPU. If True, the synchronization will happen on the python side after the kernel calls. 
    run_threaded: whether to run the waveform generation and likelihood computation in separate threads for each GPU.
    waveform_like_method: name of the method of the waveform generator class to use for generating the waveforms for likelihood computation. If None, will use the same method as waveform_gen_method.
    """
    def __init__(
        self, 
        dcga: DomainComputationGroupArray,
        waveform_gen: Any,
        branch_name: str,
        coords_shape: tuple,
        waveform_gen_method: str,
        waveform_gen_kwargs: dict,
        waveform_like_kwargs: dict,
        num_repeats: int,
        transform_fn: TransformContainer,
        priors: ProbDistContainer,
        inner_moves: list,
        Tmax: float = np.inf,
        betas_all: np.ndarray = None,
        permute_every: int = 20,
        pad_out_of_prior: bool = False,
        run_async: bool = False,
        run_threaded: bool = False,
        waveform_like_method: str = None,
        **kwargs
    ):
        ResidualAddOneRemoveOneMove.__init__(
            self,
            branch_name=branch_name,
            coords_shape=coords_shape,
            waveform_gen=getattr(waveform_gen, waveform_gen_method),
            waveform_gen_kwargs=waveform_gen_kwargs,
            waveform_like_kwargs=waveform_like_kwargs,
            acs=dcga.acs,
            num_repeats=num_repeats,
            transform_fn=transform_fn,
            priors=priors,
            inner_moves=inner_moves,
            Tmax=Tmax,
            betas_all=betas_all,
            permute_every=permute_every,
            pad_out_of_prior=pad_out_of_prior,
            **kwargs
        )

        MultiGPUMoveBase.__init__(self, dcga, run_async=run_async, run_threaded=run_threaded)

        self.waveform_gen = waveform_gen
        self.waveform_gen_method = waveform_gen_method
        self.waveform_like_method = waveform_like_method or waveform_gen_method

        self.create_waveform_gen_replicas()

    def create_waveform_gen_replicas(self, ):
        """
        Create replicas of the waveform generator for each GPU.
        """

        self._waveform_generators = []
        for i, device in enumerate(
            self.dcga.gpus if self.dcga.gpus is not None else [None] * self.dcga.num_splits
        ):
            if not hasattr(self.waveform_gen, "kwargs"):
                raise ValueError("Waveform generator must have a 'kwargs' attribute that contains the keyword arguments to initialize the waveform generator.")    
            
            with self.dcga.device_context(device):
                # if i == 0:
                #     # Reuse the initial waveform generator for the first split to save memory
                #     self._waveform_generators.append(self.waveform_gen)
                # else:
                init_kwargs = self.waveform_gen.kwargs.copy()
                if "orbits" in init_kwargs:
                    init_kwargs["orbits"] = self.dcga.computation_groups[i].orbits

                self._waveform_generators.append(
                    self.waveform_gen.__class__(**init_kwargs)
                )

    def free_gpu_memory(self):
        self.dcga.free_gpu_memory()

    @property
    def waveform_generators(self) -> list:
        return self._waveform_generators
    
    def make_args_tuple(self, coords) -> tuple:
        """
        Make a tuple of arguments for the waveform generator from the given coordinates.
        """
        # unpack the coordinates into separate arguments for the waveform generator
        return tuple(xp.array(coords[:, i]) for i in range(coords.shape[1])) # had to do in this way to give JAX the right memory addresses

    def prepare_inputs(self, coords, data_index):
        """
        Prepare the inputs for the waveform generator from the given coordinates and data index.
        """

        positions_per_split, data_intra_index_per_split, _ = self.dcga.unpack_indices(data_index)
        coords_per_split = self.dcga.unpack_coords(positions_per_split, coords, keep_tuple=True)

        data_intra_index_per_split, coords_per_split = self.dcga.place_on_device(
            items=(data_intra_index_per_split, coords_per_split)
        )

        waveform_args_per_split = self.dcga._loop_operation(
            operation=[self.make_args_tuple for _ in self.dcga.computation_groups],
            operation_args_per_split=coords_per_split,
        )

        return positions_per_split, data_intra_index_per_split, waveform_args_per_split
    
    def aggregate_waveforms(self, waveforms_per_split: list[DomainBaseArray | list[DomainBase]]) -> list[DomainBase]:
        """
        Aggregate the waveforms from each split into a single list of DomainBase objects.
        """
        waveforms = []
        for waveforms_split in waveforms_per_split:
            waveforms.extend(waveforms_split)
        return waveforms

    def get_waveform_here(self, coords: np.ndarray) -> list[DomainBase]:
        """Get the waveforms for the given source coordinates.

        """
        self.free_gpu_memory()

        data_index = np.arange(coords.shape[0], dtype=np.int32)

        _, _, waveform_args_per_split = self.prepare_inputs(coords, data_index)

        operations = [getattr(waveform_gen, self.waveform_gen_method) for waveform_gen in self.waveform_generators]

        waveforms_out = self.dcga._loop_operation(
            operation=operations,
            operation_args_per_split=waveform_args_per_split,
            operation_kwargs=self.waveform_gen_kwargs,
            aggregate_fn=self.aggregate_waveforms,
            run_threaded=self.run_threaded,
        )

        self.dcga.synchronize()

        return waveforms_out

    def setup_likelihood_here(self, coords: np.ndarray) -> None:
        """
        Set up the likelihood computation. In the general case, this means computing the :math:\\langle d | d \\rangle term.
        """

        self.dcga.compute_d_d_terms()

    def compute_like(self, coords_in: np.ndarray, data_index: np.ndarray) -> np.ndarray:
        """
        Compute the likelihood for the given coordinates and data index.

        Args:
            coords_in: coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim).
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).
        
        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """

        positions_per_split, data_intra_index_per_split, waveform_args_per_split = self.prepare_inputs(coords_in, data_index)

        waveform_like_operations = [getattr(waveform_gen, self.waveform_like_method) for waveform_gen in self.waveform_generators]

        likelihood_args_per_split = self.dcga._loop_operation(
            operation=waveform_like_operations,
            operation_args_per_split=waveform_args_per_split,
            operation_kwargs=self.waveform_like_kwargs,
            positions_per_split=positions_per_split,
            run_threaded=self.run_threaded,
        ) 

        if not self.run_async:
            self.dcga.synchronize()

        likelihoods = self.dcga.compute_signal_likelihood(
            positions_per_split=positions_per_split,
            data_intra_per_split=data_intra_index_per_split,
            noise_intra_per_split=data_intra_index_per_split,
            likelihood_args_per_split=likelihood_args_per_split,
            likelihood_kwargs={'run_async': self.run_async},
            run_threaded=self.run_threaded,
        )

        # Release GPU arrays before freeing the pool.
        # waveform_args_per_split / data_intra_index_per_split are small coord arrays
        # from place_on_device/make_args_tuple; likelihood_args_per_split holds the
        # large template arrays.  All must be dereferenced before free_all_blocks() so
        # those blocks are actually returned to CUDA rather than staying "owned" in the pool.
        del likelihood_args_per_split, waveform_args_per_split, data_intra_index_per_split
        
        self.free_gpu_memory()

        if np.any(~np.isfinite(likelihoods)):
                logger.warning(f"Non-finite likelihoods encountered: {likelihoods}. This could be a sign of numerical issues.")
                if DEBUG_MODE:
                    breakpoint()

        likelihoods = np.where(np.isfinite(likelihoods), likelihoods, -1e300)

        return likelihoods