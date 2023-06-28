# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N

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

from eryn.moves import GroupStretchMove
from eryn.moves.multipletry import logsumexp, get_mt_computations

from ...diagnostic import inner_product
from lisatools.globalfit.state import State


__all__ = ["GBSpecialStretchMove"]


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialStretchMove(GroupStretchMove):
    """Generate Revesible-Jump proposals for GBs with try-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`ProbDistContainer` object that has ``logpdf``
            and ``rvs`` methods.

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
        gpu_priors,
        *args,
        waveform_kwargs={},
        parameter_transforms=None,
        snr_lim=1e-10,
        rj_proposal_distribution=None,
        **kwargs
    ):
        # return_gpu is a kwarg for the stretch move
        GroupStretchMove.__init__(self, *args, return_gpu=True, **kwargs)

        self.gpu_priors = gpu_priors

        self.name = "gbgroupstretch"

        for key in priors:
            if not isinstance(priors[key], ProbDistContainer):
                raise ValueError(
                    "Priors need to be eryn.priors.ProbDistContainer object."
                )

        self.priors = priors
        self.gb = gb
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
        self.parameter_transforms = parameter_transforms
        self.fd = fd
        self.df = (fd[1] - fd[0]).item()
        self.mgh = mgh

        self.snr_lim = snr_lim

        self.band_edges = self.xp.asarray(self.band_edges)

        self.rj_proposal_distribution = rj_proposal_distribution
        self.is_rj_prop = self.rj_proposal_distribution is not None

        if self.is_rj_prop:
            self.get_special_proposal_setup = self.rj_proposal

        else:
            self.get_special_proposal_setup = self.in_model_proposal

    def setup_gbs(self, branch):
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        all_remaining_freqs = coords[inds][:, 1]

        all_remaining_cords = coords[inds]

        num_remaining = len(all_remaining_freqs)

        # TODO: improve this?
        self.inds_freqs_sorted = self.xp.asarray(np.argsort(all_remaining_freqs))
        self.freqs_sorted = self.xp.asarray(np.sort(all_remaining_freqs))
        self.all_coords_sorted = self.xp.asarray(all_remaining_cords)[
            self.inds_freqs_sorted
        ]

        start_inds_freq_out = np.zeros((ntemps, nwalkers, nleaves_max), dtype=int)
        freqs_sorted_here = self.freqs_sorted.get()
        freqs_remaining_here = all_remaining_freqs

        start_ind_best = np.zeros_like(freqs_remaining_here, dtype=int)

        best_index = (
            np.searchsorted(freqs_sorted_here, freqs_remaining_here, side="right") - 1
        )
        best_index[best_index < self.nfriends] = self.nfriends
        best_index[best_index >= len(freqs_sorted_here) - self.nfriends] = (
            len(freqs_sorted_here) - self.nfriends
        )
        check_inds = (
            best_index[:, None]
            + np.tile(np.arange(2 * self.nfriends), (best_index.shape[0], 1))
            - self.nfriends
        )

        check_freqs = freqs_sorted_here[check_inds]
        freq_distance = np.abs(freqs_remaining_here[:, None] - check_freqs)

        keep_min_inds = np.argsort(freq_distance, axis=-1)[:, : self.nfriends].min(
            axis=-1
        )
        start_inds_freq = check_inds[(np.arange(len(check_inds)), keep_min_inds)]

        start_inds_freq_out[inds] = start_inds_freq

        start_inds_freq_out[~inds] = -1

        if "friend_start_inds" not in supps:
            supps.add_objects({"friend_start_inds": start_inds_freq_out})
        else:
            supps[:] = {"friend_start_inds": start_inds_freq_out}

        self.all_friends_start_inds_sorted = self.xp.asarray(
            start_inds_freq_out[inds][self.inds_freqs_sorted.get()]
        )

    def find_friends(self, name, gb_points_to_move, s_inds=None):
        if s_inds is None:
            raise ValueError

        inds_points_to_move = self.xp.asarray(s_inds.flatten())

        half_friends = int(self.nfriends / 2)

        gb_points_for_move = gb_points_to_move.reshape(-1, 8).copy()

        if not hasattr(self, "ntemps"):
            self.ntemps = 1

        inds_start_freq_to_move = self.current_friends_start_inds[
            inds_points_to_move.reshape(self.ntemps, self.nwalkers, -1)
        ]

        deviation = self.xp.random.randint(
            0, self.nfriends, size=len(inds_start_freq_to_move)
        )

        inds_keep_friends = inds_start_freq_to_move + deviation

        inds_keep_friends[inds_keep_friends < 0] = 0
        inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted)] = (
            len(self.all_coords_sorted) - 1
        )

        gb_points_for_move[inds_points_to_move] = self.all_coords_sorted[
            inds_keep_friends
        ]

        return gb_points_for_move.reshape(self.ntemps, -1, 1, 8)

    def setup(self, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name != "gb_fixed":
                continue

            if not self.is_rj_prop and self.time % self.n_iter_update == 0:
                self.setup_gbs(branch)

            # update any shifted start inds due to tempering (need to do this every non-rj move)
            if not self.is_rj_prop:
                # fix the ones that have been added in RJ
                fix = (
                    branch.branch_supplimental.holder["friend_start_inds"][:] == -1
                ) & branch.inds

                if np.any(fix):
                    new_freqs = xp.asarray(branch.coords[fix][:, 1])
                    # TODO: is there a better way of doing this?

                    # fill information into friend finder for new binaries
                    branch.branch_supplimental.holder["friend_start_inds"][fix] = (
                        (
                            xp.searchsorted(self.freqs_sorted, new_freqs, side="right")
                            - 1
                        )
                        * (
                            (new_freqs > self.freqs_sorted[0])
                            & (new_freqs < self.freqs_sorted[-1])
                        )
                        + 0 * (new_freqs < self.freqs_sorted[0])
                        + (len(self.freqs_sorted) - 1)
                        * (new_freqs > self.freqs_sorted[-1])
                    ).get()

                # make sure current start inds reflect alive binaries
                self.current_friends_start_inds = self.xp.asarray(
                    branch.branch_supplimental.holder["friend_start_inds"][:]
                )

            self.mempool.free_all_blocks()

    def rj_proposal(self, model, new_state, state):
        gb_coords = self.xp.asarray(new_state.branches_coords["gb_fixed"].copy())
        gb_inds = self.xp.asarray(new_state.branches_inds["gb_fixed"].copy())

        N_vals = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"]
        N_vals = self.xp.asarray(N_vals)

        gb_keep_inds_special = self.xp.ones_like(gb_inds, dtype=bool)

        gb_keep_inds_special[~gb_inds] = self.xp.random.choice(
            self.xp.asarray([True, False]),
            p=self.xp.asarray([1.0, 0.0]),
            size=(~gb_inds).sum().item(),
        )

        # for testing removal only
        # gb_keep_inds_special[~gb_inds] = False

        print("num not there yet:", gb_keep_inds_special[~gb_inds].sum())
        gb_coords_orig = gb_coords.copy()

        # original binaries that are not there need miniscule amplitude
        # this is the approximate snr
        gb_coords_orig[~gb_inds, 0] = 1e-20

        # setup changes to all slots
        gb_coords_change = gb_coords.copy()

        # where we have binaries, we are going to propose the same point
        # but with a miniscule amplitude
        # this is the approximate snr
        gb_coords_change[gb_inds, 0] = 1e-20

        # proposed coordinates for binaries that are not there fron proposal distribution
        gb_coords_change[~gb_inds] = self.rj_proposal_distribution["gb_fixed"].rvs(
            size=int((~gb_inds).sum())
        )

        # get priors/proposals for each binary being added/removed
        all_priors_curr = self.xp.zeros_like(gb_inds, dtype=float)
        all_priors_prop = self.xp.zeros_like(gb_inds, dtype=float)
        all_priors_curr[gb_inds] = self.gpu_priors["gb_fixed"].logpdf(
            gb_coords_orig[gb_inds]
        )
        all_priors_prop[~gb_inds] = self.gpu_priors["gb_fixed"].logpdf(
            gb_coords_change[~gb_inds]
        )

        all_proposals = self.xp.zeros_like(gb_inds, dtype=float)
        all_proposals[gb_inds] = self.rj_proposal_distribution["gb_fixed"].logpdf(
            gb_coords_orig[gb_inds]
        )
        all_proposals[~gb_inds] = self.rj_proposal_distribution["gb_fixed"].logpdf(
            gb_coords_change[~gb_inds]
        )

        # TODO: check this ordering
        all_factors = (+all_proposals) * gb_inds + (-all_proposals) * (~gb_inds)

        gb_keep_inds_special[self.xp.isnan(all_factors)] = False
        all_factors[self.xp.isnan(all_factors)] = -np.inf

        # remove nans from binaries that are not there in the original coordinates
        # just copy all non-amplitude coordinates
        gb_coords_orig[~gb_inds, 1:] = gb_coords_change[~gb_inds, 1:]

        points_curr = gb_coords_orig[gb_keep_inds_special]
        points_prop = gb_coords_change[gb_keep_inds_special]

        prior_all_curr = all_priors_curr[gb_keep_inds_special]
        prior_all_prop = all_priors_prop[gb_keep_inds_special]

        f_new = gb_coords_change[~gb_inds][:, 1].get() / 1e3
        A_new = np.full_like(f_new, 1e-30)  # points_curr[:, 0] /

        # TODO: add amplitude to this computation
        N_vals[~gb_inds] = self.xp.asarray(
            get_N(
                A_new,
                f_new,
                self.waveform_kwargs["T"],
                self.waveform_kwargs["oversample"],
            )
        )
        N_vals_in = N_vals[gb_keep_inds_special]

        # all_factors[~gb_inds] = -1e10
        factors = all_factors[gb_keep_inds_special]
        
        # for testing
        # factors[:] = -1e10

        return (
            gb_coords,
            gb_inds,
            points_curr,
            points_prop,
            prior_all_curr,
            prior_all_prop,
            gb_keep_inds_special,
            N_vals_in,
            factors,
        )

    def in_model_proposal(self, model, new_state, state):
        ntemps, nwalkers, nleaves_max, ndim = new_state.branches_coords[
            "gb_fixed"
        ].shape

        # TODO: add actual amplitude
        N_vals = new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"]

        N_vals = self.xp.asarray(N_vals)

        gb_fixed_coords = self.xp.asarray(new_state.branches_coords["gb_fixed"].copy())

        gb_fixed_coords_into_proposal = gb_fixed_coords.reshape(
            ntemps, nwalkers * nleaves_max, 1, ndim
        )
        gb_inds_into_proposal = self.xp.asarray(
            state.branches["gb_fixed"].inds
        ).reshape(ntemps, nwalkers * nleaves_max, 1)
        
        # TODO: check detailed balance
        q, factors_temp = self.get_proposal(
            {"gb_fixed": gb_fixed_coords_into_proposal},
            model.random,
            s_inds_all={"gb_fixed": gb_inds_into_proposal},
            xp=self.xp,
            return_gpu=True,
        )

        gb_fixed_coords_into_proposal = gb_fixed_coords_into_proposal.reshape(
            ntemps, nwalkers, nleaves_max, ndim
        )

        q["gb_fixed"] = q["gb_fixed"].reshape(ntemps, nwalkers, nleaves_max, ndim)

        gb_inds = gb_inds_into_proposal.reshape(ntemps, nwalkers, nleaves_max)
        # if it moves move than the 
        remove = (
            np.abs((gb_fixed_coords_into_proposal[:, :, :, 1] - q["gb_fixed"][:, :, :, 1])/ 1e3/ self.df).astype(int) / N_vals
            > 0.25
        )

        gb_inds = gb_inds_into_proposal.reshape(ntemps, nwalkers, nleaves_max)
        gb_inds[remove] = False

        points_curr = gb_fixed_coords_into_proposal[gb_inds]
        points_prop = q["gb_fixed"][gb_inds]

        prior_all_curr = self.gpu_priors["gb_fixed"].logpdf(points_curr)
        prior_all_prop = self.gpu_priors["gb_fixed"].logpdf(points_prop)

        keep_prior = ~self.xp.isinf(prior_all_prop)

        prior_ok = self.xp.ones_like(gb_inds)
        prior_ok[gb_inds] = keep_prior

        gb_inds[~prior_ok] = False

        points_curr = points_curr[keep_prior]
        points_prop = points_prop[keep_prior]

        prior_all_curr = prior_all_curr[keep_prior]
        prior_all_prop = prior_all_prop[keep_prior]

        factors = factors_temp.reshape(ntemps, nwalkers, nleaves_max)[gb_inds]

        N_vals_in = N_vals[gb_inds]

        return (
            gb_fixed_coords,
            gb_inds,
            points_curr,
            points_prop,
            prior_all_curr,
            prior_all_prop,
            gb_inds,
            N_vals_in,
            factors,
        )
        
    def run_ll_part_comp(self, data_index, noise_index, start_inds, lengths):
        assert self.xp.all(data_index == noise_index)
        lnL = self.xp.zeros_like(data_index, dtype=self.xp.float64)
        main_gpu = self.xp.cuda.runtime.getDevice()
        keep_gpu_out = []
        lnL_out = []
        for gpu_i, (gpu, inds_gpu_split) in enumerate(
            zip(self.mgh.gpus, self.mgh.gpu_splits)
        ):
            with xp.cuda.device.Device(gpu):
                self.xp.cuda.runtime.deviceSynchronize()

                min_ind_split, max_ind_split = (
                    inds_gpu_split.min(),
                    inds_gpu_split.max(),
                )

                keep_gpu_here = self.xp.where(
                    (data_index >= min_ind_split) & (data_index <= max_ind_split)
                )[0]
                keep_gpu_out.append(keep_gpu_here)
                num_gpu_here = len(keep_gpu_here)
                lnL_here = self.xp.zeros(num_gpu_here, dtype=self.xp.float64)
                lnL_out.append(lnL_here)
                data_A = self.mgh.data_list[0][gpu_i]
                data_E = self.mgh.data_list[1][gpu_i]
                psd_A = self.mgh.psd_list[0][gpu_i]
                psd_E = self.mgh.psd_list[1][gpu_i]

                data_index_here = (
                    data_index[keep_gpu_here].astype(self.xp.int32) - min_ind_split
                )
                noise_index_here = (
                    noise_index[keep_gpu_here].astype(self.xp.int32) - min_ind_split
                )

                start_inds_here = start_inds[keep_gpu_here]
                lengths_here = lengths[keep_gpu_here]
                do_synchronize = False
                self.gb.specialty_piece_wise_likelihoods(
                    lnL_here,
                    data_A,
                    data_E,
                    psd_A,
                    psd_E,
                    data_index_here,
                    noise_index_here,
                    start_inds_here,
                    lengths_here,
                    self.df,
                    num_gpu_here,
                    self.start_freq_ind,
                    self.data_length,
                    do_synchronize,
                )

        for gpu_i, (gpu, inds_gpu_split) in enumerate(
            zip(self.mgh.gpus, self.mgh.gpu_splits)
        ):
            with xp.cuda.device.Device(gpu):
                self.xp.cuda.runtime.deviceSynchronize()
            with xp.cuda.device.Device(main_gpu):
                self.xp.cuda.runtime.deviceSynchronize()
                lnL[keep_gpu_out[gpu_i]] = lnL_out[gpu_i]

        self.xp.cuda.runtime.setDevice(main_gpu)
        self.xp.cuda.runtime.deviceSynchronize()
        return lnL
        
    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        st = time.perf_counter()

        self.xp.cuda.runtime.setDevice(self.mgh.gpus[0])

        self.current_state = state
        np.random.seed(10)
        # print("start stretch")

        # Check that the dimensions are compatible.
        ndim_total = 0
        for branch in state.branches.values():
            ntemps, nwalkers, nleaves_, ndim_ = branch.shape
            ndim_total += ndim_ * nleaves_

        self.nwalkers = nwalkers
        
        # Run any move-specific setup.
        self.setup(state.branches)

        new_state = State(state, copy=True)
        band_temps = xp.asarray(new_state.band_info["band_temps"])

        self.mempool.free_all_blocks()

        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        # data should not be whitened

        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb_fixed"].shape

        (
            gb_fixed_coords,
            gb_inds_orig,
            points_curr,
            points_prop,
            prior_all_curr,
            prior_all_prop,
            gb_inds,
            N_vals_in,
            factors,
        ) = self.get_special_proposal_setup(model, new_state, state)

        waveform_kwargs_now = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs_now:
            waveform_kwargs_now.pop("N")
        waveform_kwargs_now["start_freq_ind"] = self.start_freq_ind

        log_like_tmp = self.xp.asarray(new_state.log_like)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)

        self.mempool.free_all_blocks()

        unique_N = self.xp.unique(N_vals_in)
        # remove 0
        unique_N = unique_N[unique_N != 0]

        random_vals_all = self.xp.log(self.xp.random.rand(points_prop.shape[0]))

        L_contribution = self.xp.zeros_like(random_vals_all, dtype=complex)
        p_contribution = self.xp.zeros_like(random_vals_all, dtype=complex)

        data = self.mgh.data_list
        psd = self.mgh.psd_list

        # do unique for band size as separator between asynchronous kernel launches
        band_indices = (
            self.xp.searchsorted(self.band_edges, points_curr[:, 1] / 1e3) - 1
        )

        group_temp_finder = [
            self.xp.repeat(self.xp.arange(ntemps), nwalkers * nleaves_max).reshape(
                ntemps, nwalkers, nleaves_max
            ),
            self.xp.tile(self.xp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose(
                (0, 2, 1)
            ),
            self.xp.tile(self.xp.arange(nleaves_max), ((ntemps, nwalkers, 1))),
        ]

        temp_inds = group_temp_finder[0][gb_inds]
        walker_inds = group_temp_finder[1][gb_inds]
        leaf_inds = group_temp_finder[2][gb_inds]

        # randomly permute everything to ensure random ordering
        randomize = self.xp.random.permutation(leaf_inds.shape[0])

        temp_inds = temp_inds[randomize]
        walker_inds = walker_inds[randomize]
        leaf_inds = leaf_inds[randomize]
        band_indices = band_indices[randomize]
        factors = factors[randomize]
        points_curr = points_curr[randomize]
        points_prop = points_prop[randomize]
        N_vals_in = N_vals_in[randomize]
        random_vals_all = random_vals_all[randomize]
        prior_all_curr = prior_all_curr[randomize]
        prior_all_prop = prior_all_prop[randomize]

        special_band_inds = (
            int(1e12) * temp_inds + int(1e6) * walker_inds + band_indices
        )
        sort = self.xp.argsort(special_band_inds)

        temp_inds = temp_inds[sort]
        walker_inds = walker_inds[sort]
        leaf_inds = leaf_inds[sort]
        band_indices = band_indices[sort]
        factors = factors[sort]
        points_curr = points_curr[sort]
        points_prop = points_prop[sort]
        N_vals_in = N_vals_in[sort]
        random_vals_all = random_vals_all[sort]
        prior_all_curr = prior_all_curr[sort]
        prior_all_prop = prior_all_prop[sort]

        special_band_inds_sorted = special_band_inds[sort]

        (
            uni_special_bands,
            uni_index_special_bands,
            uni_count_special_bands,
        ) = self.xp.unique(
            special_band_inds_sorted, return_index=True, return_counts=True
        )

        params_curr = self.parameter_transforms.both_transforms(points_curr, xp=self.xp)
        params_prop = self.parameter_transforms.both_transforms(points_prop, xp=self.xp)

        accepted_out = self.xp.zeros_like(random_vals_all, dtype=bool)

        do_synchronize = False
        device = self.xp.cuda.runtime.getDevice()

        units = 4 if not self.is_rj_prop else 2
        # random start to rotation around 
        start_unit = model.random.randint(units)

        for tmp in range(units):
            remainder = (start_unit + tmp) % units
            all_inputs = []
            band_bookkeep_info = []
            prior_info = []
            indiv_info = []
            params_prop_info = []
            fix_temp_check_info = []
            
            for N_now in unique_N:
                N_now = N_now.item()
                if N_now == 0:
                    continue

                checkit = np.where(new_state.supplimental[:]["overall_inds"] == 0)

                # TODO; check the maximum allowable band
                keep = (
                    (band_indices % units == remainder)
                    & (N_vals_in == N_now)
                    & (band_indices < len(self.band_edges) - 2)
                )  #  & (band_indices == 501) #  # & (N_vals_in <= 256) & (temp_inds == checkit[0].item()) & (walker_inds == checkit[1].item()) #    & (band_indices < 540)  #  &  (temp_inds == 0) & (walker_inds == 0)

                if keep.sum().item() == 0:
                    continue

                # for testing
                # keep[:] = False
                # keep[tmp_keep] = True
                # keep[3000:3020:1] = True

                params_prop_info.append((points_prop[keep]))

                params_curr_in = params_curr[keep]
                params_prop_in = params_prop[keep]

                # switch to polar angle
                params_curr_in[:, -1] = np.pi / 2.0 - params_curr_in[:, -1]
                params_prop_in[:, -1] = np.pi / 2.0 - params_prop_in[:, -1]

                accepted_out_here = accepted_out[keep]

                params_curr_in_here = params_curr_in.flatten().copy()
                params_prop_in_here = params_prop_in.flatten().copy()

                prior_all_curr_here = prior_all_curr[keep]
                prior_all_prop_here = prior_all_prop[keep]

                prior_info.append((prior_all_curr_here, prior_all_curr_here))
                factors_here = factors[keep]
                random_vals_here = random_vals_all[keep]
                special_band_inds_sorted_here = special_band_inds_sorted[keep]

                (
                    uni_special_band_inds_here,
                    uni_index_special_band_inds_here,
                    uni_count_special_band_inds_here,
                ) = self.xp.unique(
                    special_band_inds_sorted_here, return_index=True, return_counts=True
                )

                # for finding the final frequency
                finding_final = (
                    self.xp.concatenate(
                        [
                            uni_index_special_band_inds_here[1:],
                            self.xp.array([len(special_band_inds_sorted_here)]),
                        ]
                    )
                    - 1
                )

                band_start_bin_ind_here = uni_index_special_band_inds_here.astype(
                    np.int32
                )
                band_num_bins_here = uni_count_special_band_inds_here.astype(np.int32)

                band_inds_indiv = band_indices[keep]
                band_inds_final_indiv = self.xp.searchsorted(self.band_edges, params_prop_in[:, 1]) - 1
                temp_inds_indiv = temp_inds[keep]

                band_inv_temp_vals_here = band_temps[(band_inds_indiv, temp_inds_indiv)]  # self.xp.asarray(self.temperature_control.betas)[temp_inds_indiv]  #

                # anything that moves across bands
                fix_change_temps = self.xp.where((band_inds_final_indiv != band_inds_indiv))[0]  #  & (temp_inds_indiv >= 0))[0]
                
                if len(fix_change_temps) > 0:
                    temp_current = band_inv_temp_vals_here[fix_change_temps]
                    temp_inds_indiv_current = temp_inds_indiv[fix_change_temps]
                    temp_prop_same_index = band_temps[(band_inds_final_indiv[fix_change_temps], temp_inds_indiv[fix_change_temps])]
                    average_temp = (temp_current + temp_prop_same_index) / 2.
                    band_inv_temp_vals_here[fix_change_temps] = average_temp

                    # factors_here[fix_change_temps] = -1e300
                    
                band_inds = band_indices[keep][uni_index_special_band_inds_here]
                band_temps_inds = temp_inds[keep][uni_index_special_band_inds_here]
                band_walkers_inds = walker_inds[keep][uni_index_special_band_inds_here]

                indiv_info.append((temp_inds[keep], walker_inds[keep], leaf_inds[keep]))

                band_bookkeep_info.append(
                    (band_temps_inds, band_walkers_inds, band_inds)
                )
                data_index_tmp = band_temps_inds * nwalkers + band_walkers_inds

                L_contribution_here = L_contribution[keep][
                    uni_index_special_band_inds_here
                ]
                p_contribution_here = p_contribution[keep][
                    uni_index_special_band_inds_here
                ]

                buffer = 5  # bins

                # determine starting point for each segment
                special_for_ind_deter1 = (
                    (temp_inds[keep] * nwalkers + walker_inds[keep])
                    * len(self.band_edges)
                    + band_indices[keep]
                ) * 1e3 + params_curr_in[:, 1] * 1e3
                sort_special_for_ind_deter1 = self.xp.argsort(special_for_ind_deter1)
                start_inds1_tmp = (params_curr_in[:, 1][sort_special_for_ind_deter1][uni_index_special_band_inds_here]/ self.df).astype(int)
                    
                final_inds1_tmp = (params_curr_in[:, 1][sort_special_for_ind_deter1][finding_final]/ self.df).astype(int)

                inds1_tmp = self.xp.sort(self.xp.asarray([start_inds1_tmp, final_inds1_tmp]).T, axis=-1)

                start_inds1 = (inds1_tmp[:, 0] - (N_now / 2) - buffer).astype(int)
                final_inds1 = (inds1_tmp[:, 1] + (N_now / 2) + buffer).astype(int)

                special_for_ind_deter2 = (
                    (temp_inds[keep] * nwalkers + walker_inds[keep])
                    * len(self.band_edges)
                    + band_indices[keep]
                ) * 1e3 + params_prop_in[:, 1] * 1e3
                sort_special_for_ind_deter2 = self.xp.argsort(special_for_ind_deter2)

                start_inds2_tmp = (params_prop_in[:, 1][sort_special_for_ind_deter2][uni_index_special_band_inds_here]/ self.df).astype(int)
                final_inds2_tmp = (params_prop_in[:, 1][sort_special_for_ind_deter2][finding_final]/ self.df).astype(int)

                inds2_tmp = self.xp.sort(self.xp.asarray([start_inds2_tmp, final_inds2_tmp]).T, axis=-1)

                start_inds2 = (inds2_tmp[:, 0] - (N_now / 2) - buffer).astype(int)
                final_inds2 = (inds2_tmp[:, 1] + (N_now / 2) + buffer).astype(int)

                start_inds = self.xp.min(
                    self.xp.asarray([start_inds1, start_inds2]), axis=0
                ).astype(np.int32)
                end_inds = self.xp.max(
                    self.xp.asarray([final_inds1, final_inds2]), axis=0
                ).astype(np.int32)

                lengths = (end_inds - start_inds).astype(np.int32)

                max_data_store_size = lengths.max().item()

                data_index_here = self.mgh.get_mapped_indices(data_index_tmp).astype(
                    np.int32
                )
                noise_index_here = data_index_here.copy()

                num_bands_here = len(band_inds)

                assert lengths.min() >= N_now + 2 * buffer
                
                # makes in-model effectively not tempered 
                # if not self.is_rj_prop:
                #    band_inv_temp_vals_here[:] = 1.0

                inputs_now = (
                    L_contribution_here,
                    p_contribution_here,
                    data[0][0],
                    data[1][0],
                    psd[0][0],
                    psd[1][0],
                    data_index_here,
                    noise_index_here,
                    params_curr_in_here,
                    params_prop_in_here,
                    prior_all_curr_here,
                    prior_all_prop_here,
                    factors_here,
                    random_vals_here,
                    band_start_bin_ind_here,  # uni_index
                    band_num_bins_here,  # uni_count
                    start_inds,
                    lengths,
                    band_inv_temp_vals_here,  # band_inv_temp_vals
                    accepted_out_here,
                    self.waveform_kwargs["T"],
                    self.waveform_kwargs["dt"],
                    N_now,
                    0,
                    self.start_freq_ind,
                    self.data_length,
                    num_bands_here,
                    max_data_store_size,
                    device,
                    do_synchronize,
                    self.is_rj_prop,
                    self.snr_lim,
                )

                all_inputs.append(inputs_now)
                self.gb.SharedMemoryMakeMove_wrap(*inputs_now)
                
                self.xp.cuda.runtime.deviceSynchronize()
                
            self.xp.cuda.runtime.deviceSynchronize()
            new_point_info = []
            for (
                inputs_now,
                band_info,
                prior_info_now,
                indiv_info_now,
                params_prop_now,
            ) in zip(
                all_inputs, band_bookkeep_info, prior_info, indiv_info, params_prop_info,
            ):
                ll_contrib_now = inputs_now[0]
                lp_contrib_now = inputs_now[1]
                accepted_now = inputs_now[19]

                # print(accepted_now.sum(0) / accepted_now.shape[0])
                temp_tmp, walker_tmp, leaf_tmp = (
                    indiv_info_now[0][accepted_now],
                    indiv_info_now[1][accepted_now],
                    indiv_info_now[2][accepted_now],
                )

                gb_fixed_coords[(temp_tmp, walker_tmp, leaf_tmp)] = params_prop_now[
                    accepted_now
                ]

                # updates related to newly added sources
                if self.is_rj_prop:
                    gb_inds_orig_check = gb_inds_orig.copy()
                    gb_inds_orig[(temp_tmp, walker_tmp, leaf_tmp)] = (~gb_inds_orig[(temp_tmp, walker_tmp, leaf_tmp)])
                    new_state.branches_supplimental["gb_fixed"].holder["N_vals"][
                        (temp_tmp.get(), walker_tmp.get(), leaf_tmp.get())
                    ] = inputs_now[22]

                    is_new_rj = params_prop_now[accepted_now][:, 0] > 1e-7
                    new_point_info.append([params_prop_now[accepted_now][is_new_rj], temp_tmp[is_new_rj], walker_tmp[is_new_rj], leaf_tmp[is_new_rj], self.xp.asarray(new_state.branches_supplimental["gb_fixed"].holder["N_vals"])[(temp_tmp[is_new_rj], walker_tmp[is_new_rj], leaf_tmp[is_new_rj])]])

                ll_change = self.xp.zeros((ntemps, nwalkers, len(self.band_edges)))
                lp_change = self.xp.zeros((ntemps, nwalkers, len(self.band_edges)))

                self.xp.cuda.runtime.deviceSynchronize()

                ll_change[band_info] = ll_contrib_now

                ll_adjustment = ll_change.sum(axis=-1)
                log_like_tmp += ll_adjustment

                self.xp.cuda.runtime.deviceSynchronize()

                lp_change[band_info] = lp_contrib_now

                lp_adjustment = lp_change.sum(axis=-1)
                log_prior_tmp += lp_adjustment

            self.xp.cuda.runtime.deviceSynchronize()
            
        new_state.branches["gb_fixed"].coords[:] = gb_fixed_coords.get()
        if self.is_rj_prop:
            new_state.branches["gb_fixed"].inds[:] = gb_inds_orig.get()
        new_state.log_like[:] = log_like_tmp.get()
        new_state.log_prior[:] = log_prior_tmp.get()
        
        self.mempool.free_all_blocks()

        # get accepted fraction
        if not self.is_rj_prop:
            assert np.all(
                new_state.branches_inds["gb_fixed"] == state.branches_inds["gb_fixed"]
            )
            accepted_check_tmp = np.zeros_like(
                new_state.branches_inds["gb_fixed"], dtype=bool
            )
            accepted_check_tmp[new_state.branches_inds["gb_fixed"]] = np.all(
                np.abs(
                    new_state.branches_coords["gb_fixed"][
                        new_state.branches_inds["gb_fixed"]
                    ]
                    - state.branches_coords["gb_fixed"][state.branches_inds["gb_fixed"]]
                )
                > 0.0,
                axis=-1,
            )
            proposed = gb_inds.get()
            accepted_check = accepted_check_tmp.sum(
                axis=(1, 2)
            ) / proposed.sum(axis=(1, 2))
        else:
            accepted_check_tmp = (
                new_state.branches_inds["gb_fixed"] == (~state.branches_inds["gb_fixed"])
            )

            proposed = gb_inds.get()
            accepted_check = accepted_check_tmp.sum(axis=(1, 2)) / proposed.sum(axis=(1, 2))
            
        # manually tell temperatures how real overall acceptance fraction is
        number_of_walkers_for_accepted = np.floor(nwalkers * accepted_check).astype(int)

        accepted_inds = np.tile(np.arange(nwalkers), (ntemps, 1))

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        accepted[accepted_inds < number_of_walkers_for_accepted[:, None]] = True

        tmp1 = np.all(
            np.abs(
                new_state.branches_coords["gb_fixed"]
                - state.branches_coords["gb_fixed"]
            )
            > 0.0,
            axis=-1,
        ).sum(axis=(2,))
        tmp2 = new_state.branches_inds["gb_fixed"].sum(axis=(2,))

        # add to move-specific accepted information
        self.accepted += tmp1
        if isinstance(self.num_proposals, int):
            self.num_proposals = tmp2
        else:
            self.num_proposals += tmp2

        new_inds = xp.asarray(new_state.branches_inds["gb_fixed"])
            
        # in-model inds will not change
        tmp_freqs_find_bands = xp.asarray(new_state.branches_coords["gb_fixed"][:, :, :, 1])
        if not self.is_rj_prop:
            count_inds = new_inds
            proposed_inds = new_inds
            accepted_inds = accepted_check_tmp

        else:
            count_inds = new_inds
            proposed_inds = gb_inds
            accepted_inds = accepted_check_tmp

        # calculate current band counts
        band_here = (xp.searchsorted(self.band_edges, tmp_freqs_find_bands.flatten() / 1e3, side="right") - 1).reshape(tmp_freqs_find_bands.shape)

        # get binaries per band
        special_band_here_num_per_band = ((group_temp_finder[0] * nwalkers + group_temp_finder[1]) * int(1e6) + band_here)[count_inds]
        unique_special_band_here_num_per_band, unique_special_band_here_num_per_band_count = xp.unique(special_band_here_num_per_band, return_counts=True)
        temp_walker_index_num_per_band = (unique_special_band_here_num_per_band / 1e6).astype(int)
        temp_index_num_per_band = (temp_walker_index_num_per_band / nwalkers).astype(int)
        walker_index_num_per_band = temp_walker_index_num_per_band - temp_index_num_per_band * nwalkers
        band_index_num_per_band = unique_special_band_here_num_per_band - temp_walker_index_num_per_band * int(1e6)

        per_walker_band_counts = xp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_counts[temp_index_num_per_band, walker_index_num_per_band, band_index_num_per_band] = unique_special_band_here_num_per_band_count
        
        # repeat now for num_proposals per band
        special_band_here_num_props = ((group_temp_finder[0] * nwalkers + group_temp_finder[1]) * int(1e6) + band_here)[proposed_inds]
        unique_special_band_here_num_props, unique_special_band_here_num_props_count = xp.unique(special_band_here_num_props, return_counts=True)
        temp_walker_index_num_props = (unique_special_band_here_num_props / 1e6).astype(int)
        temp_index_num_props = (temp_walker_index_num_props / nwalkers).astype(int)
        walker_index_num_props = temp_walker_index_num_props - temp_index_num_props * nwalkers
        band_index_num_props = unique_special_band_here_num_props - temp_walker_index_num_props * int(1e6)

        per_walker_band_proposals = xp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_proposals[temp_index_num_props, walker_index_num_props, band_index_num_props] = unique_special_band_here_num_props_count
        
        # repeat now for accepted proposals per band
        special_band_here_num_accepted = ((group_temp_finder[0] * nwalkers + group_temp_finder[1]) * int(1e6) + band_here)[accepted_inds]
        unique_special_band_here_num_accepted, unique_special_band_here_num_accepted_count = xp.unique(special_band_here_num_accepted, return_counts=True)
        temp_walker_index_num_accepted = (unique_special_band_here_num_accepted / 1e6).astype(int)
        temp_index_num_accepted = (temp_walker_index_num_accepted / nwalkers).astype(int)
        walker_index_num_accepted = temp_walker_index_num_accepted - temp_index_num_accepted * nwalkers
        band_index_num_accepted = unique_special_band_here_num_accepted - temp_walker_index_num_accepted * int(1e6)

        per_walker_band_accepted = xp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_accepted[temp_index_num_accepted, walker_index_num_accepted, band_index_num_accepted] = unique_special_band_here_num_accepted_count
        
        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        band_swaps_accepted = np.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = np.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        current_band_counts = np.zeros((len(self.band_edges) - 1, self.ntemps), dtype=int)
        
        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and self.is_rj_prop
            # and False
        ):

            betas = self.temperature_control.betas
            for i in range(ntemps - 1, 0, -1):
                if self.temperature_control.permute:
                    iperm = xp.random.permutation(nwalkers)
                    i1perm = xp.random.permutation(nwalkers)
                else:
                    iperm = xp.arange(nwalkers)  
                    i1perm = xp.arange(nwalkers) 

                # need to do this to remove inds == False binaries
                coords_tmp = new_state.branches["gb_fixed"].coords.copy()

                coords_tmp[~new_state.branches["gb_fixed"].inds] = np.nan

                coords_iperm = xp.asarray(
                    coords_tmp[i, iperm.get()]
                )
                coords_i1perm = xp.asarray(
                    coords_tmp[i - 1, i1perm.get()]
                )

                leaves_i = xp.asarray(new_state.branches["gb_fixed"].inds[i])
                leaves_i1 = xp.asarray(new_state.branches["gb_fixed"].inds[i - 1])

                walker_map_iperm = xp.repeat(
                    iperm[:, None], coords_iperm.shape[-2], axis=-1
                )
                walker_map_i1perm = xp.repeat(
                    i1perm[:, None], coords_i1perm.shape[-2], axis=-1
                )

                leaf_map_iperm = xp.tile(
                    xp.arange(coords_iperm.shape[-2]), (len(iperm), 1)
                )
                leaf_map_i1perm = xp.tile(
                    xp.arange(coords_iperm.shape[-2]), (len(i1perm), 1)
                )

                walker_pre_permute_map_iperm = xp.repeat(
                    xp.arange(len(iperm))[:, None], coords_iperm.shape[-2], axis=-1
                )
                walker_pre_permute_map_i1perm = xp.repeat(
                    xp.arange(len(i1perm))[:, None], coords_i1perm.shape[-2], axis=-1
                )

                N_vals_iperm = xp.asarray(
                    new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][
                        i, iperm.get()
                    ]
                )

                N_vals_i1perm = xp.asarray(
                    new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][
                        i - 1, i1perm.get()
                    ]
                )

                f_iperm = coords_iperm[:, :, 1] / 1e3
                f_i1perm = coords_i1perm[:, :, 1] / 1e3

                bands_iperm = (
                    xp.searchsorted(
                        self.band_edges, f_iperm.flatten(), side="right"
                    ).reshape(f_iperm.shape)
                    - 1
                )
                bands_i1perm = (
                    xp.searchsorted(
                        self.band_edges, f_i1perm.flatten(), side="right"
                    ).reshape(f_i1perm.shape)
                    - 1
                )

                bands_iperm[xp.isnan(f_iperm)] = -1
                bands_i1perm[xp.isnan(f_i1perm)] = -1
                # can probably go to 2 iterations, but will need to check Likelihood difference

                unit_temps = 3
                for odds_evens in range(unit_temps):
                    keep_here_i = (bands_iperm % unit_temps == odds_evens) & (bands_iperm >= 0)  #  & (bands_i1perm == 501)  #  & (walker_map_iperm == 0)  # & ((bands_i1perm == 2500) | (bands_i1perm == 2501))  #  & ((walker_map_iperm == 18)) # | (walker_map_iperm == 0) | (walker_map_iperm == 83))  #  & (bands_iperm == 0)
                
                    keep_here_i1 = (bands_i1perm % unit_temps == odds_evens) & (
                        bands_i1perm >= 0
                    )  # & (bands_i1perm == 501) #  & (walker_map_i1perm == 0) # & ((bands_i1perm == 2500) | (bands_i1perm == 2501))  # & ((walker_map_i1perm == 51)) # | (walker_map_i1perm == 39) | (walker_map_i1perm == 0))  #  & (bands_i1perm == 0)

                    if not xp.any(keep_here_i) and not xp.any(keep_here_i1):
                        continue
                    elif not xp.any(keep_here_i) or not xp.any(keep_here_i1):
                        breakpoint()

                    bands_here_i = bands_iperm[keep_here_i]
                    bands_here_i1 = bands_i1perm[keep_here_i1]

                    walkers_here_i_remove = walker_map_iperm[keep_here_i]
                    walkers_here_i1_remove = walker_map_i1perm[keep_here_i1]

                    walkers_here_i_add = walker_map_iperm[
                        keep_here_i1
                    ]  # i walkers with i1 binaries
                    walkers_here_i1_add = walker_map_i1perm[keep_here_i]

                    leaf_map_here_iperm = leaf_map_iperm[keep_here_i]
                    leaf_map_here_i1perm = leaf_map_i1perm[keep_here_i1]

                    walker_band_map_i = int(1e6) * walkers_here_i_remove + bands_here_i
                    walker_band_map_i1 = (
                        int(1e6) * walkers_here_i1_remove + bands_here_i1
                    )

                    walkers_pre_permute_here_i = walker_pre_permute_map_iperm[
                        keep_here_i
                    ]
                    walkers_pre_permute_here_i1 = walker_pre_permute_map_i1perm[
                        keep_here_i1
                    ]

                    N_here_i = N_vals_iperm[keep_here_i]
                    N_here_i1 = N_vals_i1perm[keep_here_i1]

                    coords_here_i = coords_iperm[keep_here_i]
                    coords_here_i1 = coords_i1perm[keep_here_i1]

                    coords_in_here_i = self.parameter_transforms.both_transforms(
                        coords_here_i, xp=xp
                    )
                    coords_in_here_i1 = self.parameter_transforms.both_transforms(
                        coords_here_i1, xp=xp
                    )

                    params_proposal_in = xp.concatenate(
                        [
                            coords_in_here_i,  # remove from i
                            coords_in_here_i1,  # add to i
                            coords_in_here_i1,  # remove form i - 1
                            coords_in_here_i,  # add to i - 1
                        ],
                        axis=0,
                    )

                    N_vals_in = xp.concatenate(
                        [N_here_i, N_here_i1, N_here_i1, N_here_i]
                    )

                    data_index_tmp_i_remove = i * nwalkers + walkers_here_i_remove
                    data_index_tmp_i1_remove = (
                        i - 1
                    ) * nwalkers + walkers_here_i1_remove

                    data_index_tmp_i_add = i * nwalkers + walkers_here_i_add
                    data_index_tmp_i1_add = (i - 1) * nwalkers + walkers_here_i1_add

                    data_index_tmp_all = xp.concatenate(
                        [
                            data_index_tmp_i_remove,
                            data_index_tmp_i_add,
                            data_index_tmp_i1_remove,
                            data_index_tmp_i1_add,
                        ]
                    )
                    data_index_forward = self.xp.asarray(
                        self.mgh.get_mapped_indices(data_index_tmp_all)
                    ).astype(self.xp.int32)

                    factors_multiply_forward = self.xp.asarray(
                        xp.concatenate(
                            [
                                xp.full_like(
                                    data_index_tmp_i_remove, +1.0, dtype=xp.float64
                                ),
                                xp.full_like(
                                    data_index_tmp_i_add, -1.0, dtype=xp.float64
                                ),
                                xp.full_like(
                                    data_index_tmp_i1_remove, +1.0, dtype=xp.float64
                                ),
                                xp.full_like(
                                    data_index_tmp_i1_add, -1.0, dtype=xp.float64
                                ),
                            ]
                        )
                    )
                    waveform_kwargs_fill = waveform_kwargs_now.copy()

                    f_find = (
                        xp.concatenate([coords_here_i[:, 1], coords_here_i1[:, 1]])
                        / 1e3
                    )

                    walker_find = xp.concatenate(
                        [walkers_pre_permute_here_i, walkers_pre_permute_here_i1]
                    )
                    N_here_find = xp.concatenate([N_here_i, N_here_i1])
                    band_here_find = (
                        xp.searchsorted(self.band_edges, f_find, side="right") - 1
                    )

                    band_here_i = xp.searchsorted(self.band_edges, coords_here_i[:, 1] / 1e3, side="right") - 1
                    band_here_i1 = xp.searchsorted(self.band_edges, coords_here_i1[:, 1] / 1e3, side="right") - 1
                    
                    walker_band_find = walker_find * int(1e6) + band_here_find

                    f_find_min = f_find - (N_here_find / 2) * self.df
                    f_find_min_sort = xp.argsort(f_find_min)
                    f_find_min = f_find_min[f_find_min_sort]
                    walker_band_find_min = walker_band_find[f_find_min_sort]
                    walker_band_uni_first, walker_band_first = xp.unique(
                        walker_band_find_min, return_index=True
                    )
                    f_min_band = f_find_min[walker_band_first]

                    f_find_max = f_find + (N_here_find / 2) * self.df
                    f_find_max_sort = xp.argsort(f_find_max)
                    f_find_max = f_find_max[f_find_max_sort]
                    walker_band_find_max = walker_band_find[f_find_max_sort]
                    walker_band_uni_last, walker_band_last = xp.unique(
                        walker_band_find_max[::-1], return_index=True
                    )
                    f_max_band = f_find_max[::-1][walker_band_last]

                    assert xp.all(walker_band_uni_first == walker_band_uni_last)

                    start_inds_band = (f_min_band / self.df).astype(int) - 0
                    end_inds_band = (f_max_band / self.df).astype(int) + 1
                    lengths_band = end_inds_band - start_inds_band

                    walker_permute = (walker_band_uni_first / 1e6).astype(int)

                    walker_i = iperm[walker_permute]
                    walker_i1 = i1perm[walker_permute]

                    band_in = (walker_band_uni_first - walker_permute * int(1e6)).astype(int)

                    walker_band_in_i = (
                        walker_band_uni_first
                        - walker_permute * int(1e6)
                        + walker_i * int(1e6)
                    )
                    walker_band_in_i1 = (
                        walker_band_uni_first
                        - walker_permute * int(1e6)
                        + walker_i1 * int(1e6)
                    )

                    start_inds_all = self.xp.asarray(
                        xp.concatenate([start_inds_band, start_inds_band]).astype(
                            xp.int32
                        )
                    )
                    lengths_all = self.xp.asarray(
                        xp.concatenate([lengths_band, lengths_band]).astype(xp.int32)
                    )

                    data_index_tmp_all2 = xp.concatenate(
                        [i * nwalkers + walker_i, (i - 1) * nwalkers + walker_i1]
                    )

                    data_index_all = self.xp.asarray(
                        self.mgh.get_mapped_indices(data_index_tmp_all2).astype(xp.int32)
                    )
                    noise_index_all = data_index_all.copy()

                    ll_contrib_before = self.run_ll_part_comp(
                        data_index_all, noise_index_all, start_inds_all, lengths_all
                    )

                    # ll_before = self.mgh.get_ll(include_psd_info=True).flatten()[data_index_all.get()]
                    self.gb.generate_global_template(
                        params_proposal_in,
                        data_index_forward,
                        self.mgh.data_list,
                        N=self.xp.asarray(N_vals_in),
                        data_length=self.data_length,
                        data_splits=self.mgh.gpu_splits,
                        factors=factors_multiply_forward,
                        **waveform_kwargs_fill
                    )
                    xp.cuda.runtime.deviceSynchronize()
                    # ll_after3 = self.mgh.get_ll(include_psd_info=True).flatten()[new_state.supplimental[:]["overall_inds"]].reshape(ntemps, nwalkers)
                    # check_here = xp.abs(new_state.log_like - ll_after3)

                    ll_contrib_after = self.run_ll_part_comp(
                        data_index_all, noise_index_all, start_inds_all, lengths_all
                    )
                    
                    half = start_inds_band.shape[0]

                    logl_i_initial = ll_contrib_before[:half]
                    logl_i_final = ll_contrib_after[:half]

                    logl_i1_initial = ll_contrib_before[half:]
                    logl_i1_final = ll_contrib_after[half:]

                    delta_logl_i = (logl_i_final - logl_i_initial)
                    delta_logl_i1 = (logl_i1_final - logl_i1_initial)

                    # dbeta = bi1 - bi
                    bi = band_temps[band_in, i]
                    bi1 = band_temps[band_in, i - 1]
                    
                    paccept = bi * delta_logl_i + bi1 * delta_logl_i1
                    raccept = xp.log(xp.random.uniform(size=paccept.shape[0]))

                    # How many swaps were accepted?
                    sel = paccept > self.xp.asarray(raccept)

                    self.temperature_control.swaps_proposed[i - 1] += sel.shape[0]
                    self.temperature_control.swaps_accepted[i - 1] += sel.sum()

                    walkers_proposed_i = (walker_band_in_i / 1e6).astype(int)
                    walkers_proposed_i1 = (walker_band_in_i1 / 1e6).astype(int)

                    bands_proposed_i = walker_band_in_i - int(1e6) * walkers_proposed_i
                    bands_proposed_i1 = walker_band_in_i1 - int(1e6) * walkers_proposed_i1

                    keep_walker_band_i = walker_band_in_i[sel.get()]
                    keep_walker_band_i1 = walker_band_in_i1[sel.get()]

                    keep_walker_i_band_level = (keep_walker_band_i / 1e6).astype(int)
                    keep_walker_i1_band_level = (keep_walker_band_i1 / 1e6).astype(int)

                    keep_band_i_band_level = (
                        keep_walker_band_i - int(1e6) * keep_walker_i_band_level
                    )
                    keep_band_i1_band_level = (
                        keep_walker_band_i1 - int(1e6) * keep_walker_i1_band_level
                    )

                    unique_band_i_proposed, unique_band_i_proposed_count = np.unique(bands_proposed_i.get(), return_counts=True)
                    unique_band_i1_proposed, unique_band_i1_proposed_count = np.unique(bands_proposed_i1.get(), return_counts=True)
                    
                    unique_keep_band_i_proposed, unique_keep_band_i_proposed_count = np.unique(keep_band_i_band_level.get(), return_counts=True)
                    unique_keep_band_i1_proposed, unique_keep_band_i1_proposed_count = np.unique(keep_band_i1_band_level.get(), return_counts=True)
                    
                    unique_all_bands_in_prop, unique_all_bands_in_count_prop = np.unique(band_in, return_counts=True)

                    unique_all_bands_in_accept, unique_all_bands_in_count_accept = np.unique(band_in[sel], return_counts=True)
                    
                    band_swaps_accepted[unique_all_bands_in_accept.get(), i - 1] += unique_all_bands_in_count_accept.get()
                    band_swaps_proposed[unique_all_bands_in_prop.get(), i - 1] += unique_all_bands_in_count_prop.get()
                    
                    band_ll_diff_i = xp.zeros((nwalkers, len(self.band_edges) - 1))
                    band_ll_diff_i1 = xp.zeros((nwalkers, len(self.band_edges) - 1))

                    band_ll_diff_i[
                        (keep_walker_i_band_level, keep_band_i_band_level)
                    ] = delta_logl_i[sel].get()
                    band_ll_diff_i1[
                        (keep_walker_i1_band_level, keep_band_i1_band_level)
                    ] = delta_logl_i1[sel].get()
                    
                    new_state.log_like[i] += band_ll_diff_i.sum(axis=-1).get()
                    new_state.log_like[i - 1] += band_ll_diff_i1.sum(axis=-1).get()

                    keep_i = xp.in1d(walker_band_map_i, keep_walker_band_i)
                    keep_i1 = xp.in1d(walker_band_map_i1, keep_walker_band_i1)

                    keep_walkers_here_i_remove = walkers_here_i_remove[keep_i]
                    keep_walkers_here_i1_remove = walkers_here_i1_remove[keep_i1]

                    keep_leaf_here_i_remove = leaf_map_here_iperm[keep_i]
                    keep_leaf_here_i1_remove = leaf_map_here_i1perm[keep_i1]

                    mapping = xp.tile(
                        xp.arange(leaves_i.shape[-1]), (leaves_i.shape[0], 1)
                    )

                    new_state.branches["gb_fixed"].inds[
                        i,
                        keep_walkers_here_i_remove.get(),
                        keep_leaf_here_i_remove.get(),
                    ] = False
                    new_state.branches["gb_fixed"].inds[
                        i - 1,
                        keep_walkers_here_i1_remove.get(),
                        keep_leaf_here_i1_remove.get(),
                    ] = False

                    leaves_i = xp.asarray(new_state.branches["gb_fixed"].inds[i]).copy()
                    leaves_i1 = xp.asarray(
                        new_state.branches["gb_fixed"].inds[i - 1]
                    ).copy()

                    num_leaves_i_to_add_arr = xp.zeros_like(leaves_i)
                    num_leaves_i1_to_add_arr = xp.zeros_like(leaves_i1)

                    keep_walkers_here_i1_add = walkers_here_i1_add[keep_i]
                    keep_walkers_here_i_add = walkers_here_i_add[keep_i1]

                    # opposite i <-> i1
                    num_leaves_i1_to_add_arr[
                        (keep_walkers_here_i1_add, keep_leaf_here_i_remove)
                    ] = True
                    num_leaves_i_to_add_arr[
                        (keep_walkers_here_i_add, keep_leaf_here_i1_remove)
                    ] = True

                    walker_mapping = xp.repeat(
                        xp.arange(leaves_i.shape[0])[:, None],
                        leaves_i.shape[-1],
                        axis=-1,
                    )
                    walker_add_i1 = walker_mapping[num_leaves_i1_to_add_arr]
                    walker_add_i = walker_mapping[num_leaves_i_to_add_arr]

                    num_leaves_i1_to_add = num_leaves_i1_to_add_arr.sum(axis=-1)
                    num_leaves_i_to_add = num_leaves_i_to_add_arr.sum(axis=-1)

                    leaves_add_i = xp.concatenate(
                        [
                            xp.arange(leaves_i.shape[-1])[~leaves_i[w]][
                                : num_leaves_i_to_add[w]
                            ]
                            for w in iperm
                        ]
                    )
                    leaves_add_i1 = xp.concatenate(
                        [
                            xp.arange(leaves_i1.shape[-1])[~leaves_i1[w]][
                                : num_leaves_i1_to_add[w]
                            ]
                            for w in i1perm
                        ]
                    )

                    # update coords
                    tmp_coords_i1 = new_state.branches["gb_fixed"].coords[
                        i - 1,
                        keep_walkers_here_i1_remove.get(),
                        keep_leaf_here_i1_remove.get(),
                    ]
                    tmp_N_vals_i1 = new_state.branches[
                        "gb_fixed"
                    ].branch_supplimental.holder["N_vals"][
                        i - 1,
                        keep_walkers_here_i1_remove.get(),
                        keep_leaf_here_i1_remove.get(),
                    ]
                    tmp_start_inds_i1 = new_state.branches[
                        "gb_fixed"
                    ].branch_supplimental.holder["friend_start_inds"][
                        i - 1,
                        keep_walkers_here_i1_remove.get(),
                        keep_leaf_here_i1_remove.get(),
                    ]
                    
                    tmp_coords_i = new_state.branches["gb_fixed"].coords[
                        i,
                        keep_walkers_here_i_remove.get(),
                        keep_leaf_here_i_remove.get(),
                    ]
                    tmp_N_vals_i = new_state.branches[
                        "gb_fixed"
                    ].branch_supplimental.holder["N_vals"][
                        i,
                        keep_walkers_here_i_remove.get(),
                        keep_leaf_here_i_remove.get(),
                    ]
                    tmp_start_inds_i = new_state.branches[
                        "gb_fixed"
                    ].branch_supplimental.holder["friend_start_inds"][
                        i,
                        keep_walkers_here_i_remove.get(),
                        keep_leaf_here_i_remove.get(),
                    ]
                    assert np.all(
                        new_state.branches["gb_fixed"].inds[
                            i - 1, keep_walkers_here_i1_add.get(), leaves_add_i1.get()
                        ]
                        == False
                    )
                    assert np.all(
                            new_state.branches["gb_fixed"].inds[
                                i, keep_walkers_here_i_add.get(), leaves_add_i.get()
                            ]
                            == False
                        )

                    new_state.branches["gb_fixed"].coords[
                        i - 1, keep_walkers_here_i1_add.get(), leaves_add_i1.get()
                    ] = tmp_coords_i
                    new_state.branches["gb_fixed"].inds[
                        i - 1, keep_walkers_here_i1_add.get(), leaves_add_i1.get()
                    ] = True
                    new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][
                        i - 1, keep_walkers_here_i1_add.get(), leaves_add_i1.get()
                    ] = tmp_N_vals_i
                    new_state.branches["gb_fixed"].branch_supplimental.holder[
                        "friend_start_inds"
                    ][
                        i - 1, keep_walkers_here_i1_add.get(), leaves_add_i1.get()
                    ] = tmp_start_inds_i

                    new_state.branches["gb_fixed"].coords[
                        i, keep_walkers_here_i_add.get(), leaves_add_i.get()
                    ] = tmp_coords_i1
                    new_state.branches["gb_fixed"].inds[
                        i, keep_walkers_here_i_add.get(), leaves_add_i.get()
                    ] = True
                    new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][
                        i, keep_walkers_here_i_add.get(), leaves_add_i.get()
                    ] = tmp_N_vals_i1
                    new_state.branches["gb_fixed"].branch_supplimental.holder[
                        "friend_start_inds"
                    ][
                        i, keep_walkers_here_i_add.get(), leaves_add_i.get()
                    ] = tmp_start_inds_i1

                    reverse_walker_band_i = walker_band_in_i[~sel.get()]
                    reverse_walker_band_i1 = walker_band_in_i1[~sel.get()]

                    reverse_i = xp.in1d(walker_band_map_i, reverse_walker_band_i)
                    reverse_i1 = xp.in1d(walker_band_map_i1, reverse_walker_band_i1)

                    reverse_N_here_i = N_vals_iperm[keep_here_i][reverse_i]
                    reverse_N_here_i1 = N_vals_i1perm[keep_here_i1][reverse_i1]

                    reverse_coords_here_i = coords_iperm[keep_here_i][reverse_i]
                    reverse_coords_here_i1 = coords_i1perm[keep_here_i1][reverse_i1]

                    reverse_coords_in_here_i = (
                        self.parameter_transforms.both_transforms(
                            reverse_coords_here_i, xp=xp
                        )
                    )
                    reverse_coords_in_here_i1 = (
                        self.parameter_transforms.both_transforms(
                            reverse_coords_here_i1, xp=xp
                        )
                    )

                    reverse_params_proposal_in = xp.concatenate(
                        [
                            reverse_coords_in_here_i1,  # remove from i
                            reverse_coords_in_here_i,  # add to i
                            reverse_coords_in_here_i,  # remove form i - 1
                            reverse_coords_in_here_i1,  # add to i - 1
                        ],
                        axis=0,
                    )

                    reverse_N_vals_in = xp.concatenate(
                        [
                            reverse_N_here_i1,
                            reverse_N_here_i,
                            reverse_N_here_i,
                            reverse_N_here_i1,
                        ]
                    )

                    reverse_data_index_tmp_i_remove = (
                        i * nwalkers + walkers_here_i_add[reverse_i1]
                    )
                    reverse_data_index_tmp_i1_remove = (
                        i - 1
                    ) * nwalkers + walkers_here_i1_add[reverse_i]

                    reverse_data_index_tmp_i_add = (
                        i * nwalkers + walkers_here_i_remove[reverse_i]
                    )
                    reverse_data_index_tmp_i1_add = (
                        i - 1
                    ) * nwalkers + walkers_here_i1_remove[reverse_i1]

                    reverse_data_index_tmp_all = xp.concatenate(
                        [
                            reverse_data_index_tmp_i_remove,
                            reverse_data_index_tmp_i_add,
                            reverse_data_index_tmp_i1_remove,
                            reverse_data_index_tmp_i1_add,
                        ]
                    )
                    reverse_data_index = self.xp.asarray(
                        self.mgh.get_mapped_indices(reverse_data_index_tmp_all)
                    ).astype(self.xp.int32)

                    reverse_factors_multiply = self.xp.asarray(
                        xp.concatenate(
                            [
                                xp.full_like(
                                    reverse_data_index_tmp_i_remove,
                                    +1.0,
                                    dtype=xp.float64,
                                ),
                                xp.full_like(
                                    reverse_data_index_tmp_i_add, -1.0, dtype=xp.float64
                                ),
                                xp.full_like(
                                    reverse_data_index_tmp_i1_remove,
                                    +1.0,
                                    dtype=xp.float64,
                                ),
                                xp.full_like(
                                    reverse_data_index_tmp_i1_add,
                                    -1.0,
                                    dtype=xp.float64,
                                ),
                            ]
                        )
                    )

                    self.gb.generate_global_template(
                        reverse_params_proposal_in,
                        reverse_data_index,
                        self.mgh.data_list,
                        N=self.xp.asarray(reverse_N_vals_in),
                        data_length=self.data_length,
                        data_splits=self.mgh.gpu_splits,
                        factors=reverse_factors_multiply,
                        **waveform_kwargs_fill
                    )
                    
                new_state.branches["gb_fixed"].coords[
                    ~new_state.branches["gb_fixed"].inds
                ] = self.xp.nan

            # adjust priors accordingly
            log_prior_new_per_bin = xp.zeros_like(
                new_state.branches_inds["gb_fixed"], dtype=xp.float64
            )
            # self.gpu_priors
            log_prior_new_per_bin[
                new_state.branches_inds["gb_fixed"]
            ] = self.gpu_priors["gb_fixed"].logpdf(
                xp.asarray(
                    new_state.branches_coords["gb_fixed"][
                        new_state.branches_inds["gb_fixed"]
                    ]
                )
            )

            new_state.log_prior = log_prior_new_per_bin.sum(axis=-1).get()
            
            ratios = (band_swaps_accepted / band_swaps_proposed).T #  self.swaps_accepted / self.swaps_proposed
            ratios[np.isnan(ratios)] = 0.0

            # adapt if desired
            if self.time > 50:
                betas0 = band_temps.copy().T.get()
                betas1 = betas0.copy()

                # Modulate temperature adjustments with a hyperbolic decay.
                decay = self.temperature_control.adaptation_lag / (self.time + self.temperature_control.adaptation_lag)
                kappa = decay / self.temperature_control.adaptation_time

                # Construct temperature adjustments.
                dSs = kappa * (ratios[:-1] - ratios[1:])

                # Compute new ladder (hottest and coldest chains don't move).
                deltaTs = np.diff(1 / betas1[:-1], axis=0)

                deltaTs *= np.exp(dSs)
                betas1[1:-1] = 1 / (np.cumsum(deltaTs, axis=0) + 1 / betas1[0])

                # Don't mutate the ladder here; let the client code do that.
                dbetas = betas1 - betas0

                band_temps += self.xp.asarray(dbetas.T)

            # band_temps[:] = band_temps[553, :][None, :]

            # only increase time if it is adaptive.
            new_state.betas = self.temperature_control.betas.copy()
            
            self.mempool.free_all_blocks()
            # print(
            #     self.is_rj_prop,
            #     band_swaps_accepted[350] / band_swaps_proposed[350],
            #     band_swaps_accepted[450] / band_swaps_proposed[450],
            #     band_swaps_accepted[501] / band_swaps_proposed[501]
            # )

        self.mempool.free_all_blocks()

        if self.time % 1 == 0:
            ll_after = (
                self.mgh.get_ll(include_psd_info=True)
                .flatten()[new_state.supplimental[:]["overall_inds"]]
                .reshape(ntemps, nwalkers)
            )
            # print(np.abs(new_state.log_like - ll_after).max())
            store_max_diff = np.abs(new_state.log_like - ll_after).max()
            if np.abs(new_state.log_like - ll_after).max() > 1e-2:
                breakpoint()
                
        self.time += 1
        # self.xp.cuda.runtime.deviceSynchronize()

        self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        new_state.update_band_information(
            band_temps.get(), per_walker_band_proposals.sum(axis=1).get().T, per_walker_band_accepted.sum(axis=1).get().T, band_swaps_proposed, band_swaps_accepted,
            per_walker_band_counts.get(), self.is_rj_prop
        )
        # TODO: check rj numbers
        # if self.is_rj_prop:
        #     breakpoint()

        self.mempool.free_all_blocks()
        return new_state, accepted


        # breakpoint()

        # # print(self.accepted / self.num_proposals)

        # # MULTIPLE TRY after RJ

        # if False: # self.is_rj_prop:
            
        #     inds_added = np.where(new_state.branches["gb_fixed"].inds.astype(int) - state.branches["gb_fixed"].inds.astype(int) == +1)

        #     new_coords = xp.asarray(new_state.branches["gb_fixed"].coords[inds_added])
        #     temp_inds_add = xp.repeat(xp.arange(ntemps)[:, None], nwalkers * nleaves_max, axis=-1).reshape(ntemps, nwalkers, nleaves_max)[inds_added]
        #     walker_inds_add = xp.repeat(xp.arange(nwalkers)[:, None], ntemps * nleaves_max, axis=-1).reshape(nwalkers, ntemps, nleaves_max).transpose(1, 0, 2)[inds_added]
        #     leaf_inds_add = xp.repeat(xp.arange(nleaves_max)[:, None], ntemps * nwalkers, axis=-1).reshape(nleaves_max, ntemps, nwalkers).transpose(1, 2, 0)[inds_added]
        #     band_inds_add = xp.searchsorted(self.band_edges, new_coords[:, 1] / 1e3, side="right") - 1
        #     N_vals_add = xp.asarray(new_state.branches["gb_fixed"].branch_supplimental.holder["N_vals"][inds_added])
            
        #     #### RIGHT NOW WE ARE EXPERIMENTING WITH NO OR SMALL CHANGE IN FREQUENCY
        #     # because if it is added in RJ, it has locked on in frequency in some form
            
        #     # randomize order
        #     inds_random = xp.random.permutation(xp.arange(len(new_coords)))
        #     new_coords = new_coords[inds_random]
        #     temp_inds_add = temp_inds_add[inds_random]
        #     walker_inds_add = walker_inds_add[inds_random]
        #     leaf_inds_add = leaf_inds_add[inds_random]
        #     band_inds_add = band_inds_add[inds_random]
        #     N_vals_add = N_vals_add[inds_random]

        #     special_band_map = leaf_inds_add * int(1e12) + walker_inds_add * int(1e6) + band_inds_add
        #     inds_sorted_special = xp.argsort(special_band_map)
        #     special_band_map = special_band_map[inds_sorted_special]
        #     new_coords = new_coords[inds_sorted_special]
        #     temp_inds_add = temp_inds_add[inds_sorted_special]
        #     walker_inds_add = walker_inds_add[inds_sorted_special]
        #     leaf_inds_add = leaf_inds_add[inds_sorted_special]
        #     band_inds_add = band_inds_add[inds_sorted_special]
        #     N_vals_add = N_vals_add[inds_sorted_special]

        #     unique_special_bands, unique_special_bands_index, unique_special_bands_inverse = xp.unique(special_band_map, return_index=True, return_inverse=True)
        #     group_index = xp.arange(len(special_band_map)) - xp.arange(len(special_band_map))[unique_special_bands_index][unique_special_bands_inverse]
            
        #     band_splits = 3
        #     num_try = 1000
        #     for group in range(group_index.max().item()):
        #         for split_i in range(band_splits):
        #             group_split = (group_index == group) & (band_inds_add % band_splits == split_i)
                    
        #             new_coords_group_split = new_coords[group_split]
        #             temp_inds_add_group_split = temp_inds_add[group_split]
        #             walker_inds_add_group_split = walker_inds_add[group_split]
        #             leaf_inds_add_group_split = leaf_inds_add[group_split]
        #             band_inds_add_group_split = band_inds_add[group_split]
        #             N_vals_add_group_split = N_vals_add[group_split]

        #             coords_remove = xp.repeat(new_coords_group_split, num_try, axis=0)
        #             coords_add = coords_remove.copy()

        #             inds_params = xp.array([0, 2, 3, 4, 5, 6, 7])
        #             coords_add[:, inds_params] = self.gpu_priors["gb_fixed"].rvs(size=coords_remove.shape[0])[:, inds_params]
                    
        #             log_proposal_pdf = self.gpu_priors["gb_fixed"].logpdf(coords_add)
        #             # remove logpdf from fchange for now
        #             log_proposal_pdf -= self.gpu_priors["gb_fixed"].priors_in[1].logpdf(coords_add[:, 1])

        #             priors_remove = self.gpu_priors["gb_fixed"].logpdf(coords_remove)
        #             priors_add = self.gpu_priors["gb_fixed"].logpdf(coords_add)

        #             if xp.any(coords_add[:, 1] != coords_remove[:, 1]):
        #                 raise NotImplementedError("Assumes frequencies are the same.")
        #                 independent = False
        #             else:
        #                 independent = True

        #             coords_remove_in = self.parameter_transforms.both_transforms(coords_remove, xp=xp)
        #             coords_add_in = self.parameter_transforms.both_transforms(coords_add, xp=xp)
                    
        #             waveform_kwargs_tmp = self.waveform_kwargs.copy()
        #             waveform_kwargs_tmp.pop("N")

        #             data_index_in = xp.repeat(temp_inds_add_group_split * nwalkers + walker_inds_add_group_split, num_try).astype(xp.int32)
        #             noise_index_in = data_index_in.copy()
        #             N_vals_add_group_split_in = self.xp.repeat(N_vals_add_group_split, num_try)

        #             ll_diff = self.xp.asarray(self.gb.swap_likelihood_difference(coords_remove_in, coords_add_in, self.mgh.data_list, self.mgh.psd_list, data_index=data_index_in, noise_index=noise_index_in, N=N_vals_add_group_split_in, data_length=self.data_length, data_splits=self.mgh.gpu_splits, **waveform_kwargs_tmp))
                    
        #             ll_diff[self.xp.isnan(ll_diff)] = -1e300

        #             band_inv_temps = self.xp.repeat(band_temps[(band_inds_add_group_split, temp_inds_add_group_split)], num_try)
                    
        #             logP = band_inv_temps * ll_diff + priors_add

        #             from eryn.moves.multipletry import logsumexp, get_mt_computations

        #             band_inv_temps = band_inv_temps.reshape(-1, num_try)
        #             ll_diff = ll_diff.reshape(-1, num_try)
        #             logP = logP.reshape(-1, num_try)
        #             priors_add = priors_add.reshape(-1, num_try)
        #             log_proposal_pdf = log_proposal_pdf.reshape(-1, num_try)
        #             coords_add = coords_add.reshape(-1, num_try, ndim)
                    
        #             log_importance_weights, log_sum_weights, inds_group_split = get_mt_computations(logP, log_proposal_pdf, symmetric=False, xp=self.xp)
                    
        #             inds_tuple = (self.xp.arange(len(inds_group_split)), inds_group_split)

        #             ll_diff_out = ll_diff[inds_tuple]
        #             logP_out = logP[inds_tuple]
        #             priors_add_out = priors_add[inds_tuple]
        #             coords_add_out = coords_add[inds_tuple]
        #             log_proposal_pdf_out = log_proposal_pdf[inds_tuple]

        #             if not independent:
        #                 raise NotImplementedError
        #             else:
        #                 aux_coords_add = coords_add.copy()
        #                 aux_ll_diff = ll_diff.copy()
        #                 aux_priors_add = priors_add.copy()
        #                 aux_log_proposal_pdf = log_proposal_pdf.copy()

        #                 aux_coords_add[:, 0] = new_coords_group_split
        #                 aux_ll_diff[:, 0] = 0.0  # diff is zero because the points are already in the data
        #                 aux_priors_add[:, 0] = priors_remove[::num_try]

        #                 initial_log_proposal_pdf = self.gpu_priors["gb_fixed"].logpdf(new_coords_group_split)
        #                 # remove logpdf from fchange for now
        #                 initial_log_proposal_pdf -= self.gpu_priors["gb_fixed"].priors_in[1].logpdf(new_coords_group_split[:, 1])

        #                 aux_log_proposal_pdf[:, 0] = initial_log_proposal_pdf

        #                 aux_logP = band_inv_temps * aux_ll_diff + aux_priors_add

        #                 aux_log_importane_weights, aux_log_sum_weights, _ = get_mt_computations(aux_logP, aux_log_proposal_pdf, symmetric=False, xp=self.xp)

        #                 aux_logP_out = aux_logP[:, 0]
        #                 aux_log_proposal_pdf_out = aux_log_proposal_pdf[:, 0]

        #             factors = ((aux_logP_out - aux_log_sum_weights)- aux_log_proposal_pdf_out + aux_log_proposal_pdf_out) - ((logP_out - log_sum_weights) - log_proposal_pdf_out + log_proposal_pdf_out)

        #             lnpdiff = factors + logP_out - aux_logP_out

        #             keep = lnpdiff > self.xp.asarray(self.xp.log(self.xp.random.rand(*logP_out.shape)))

        #             coords_remove_keep = new_coords_group_split[keep]
        #             coords_add_keep = coords_add_out[keep]
        #             temp_inds_add_keep = temp_inds_add_group_split[keep]
        #             walker_inds_add_keep = walker_inds_add_group_split[keep]
        #             leaf_inds_add_keep = leaf_inds_add_group_split[keep]
        #             band_inds_add_keep = band_inds_add_group_split[keep]
        #             N_vals_add_keep = N_vals_add_group_split[keep]
        #             ll_diff_keep = ll_diff_out[keep]
        #             priors_add_keep = priors_add_out[keep]
        #             priors_remove_keep = priors_remove[::num_try][keep]

        #             # adjust everything
        #             ll_band_diff = xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1))
        #             ll_band_diff[temp_inds_add_keep, walker_inds_add_keep, band_inds_add_keep] = ll_diff_keep
        #             lp_band_diff = xp.zeros((ntemps, nwalkers, len(self.band_edges) - 1))
        #             lp_band_diff[temp_inds_add_keep, walker_inds_add_keep, band_inds_add_keep] = priors_add_keep - priors_remove_keep

        #             new_state.branches["gb_fixed"].coords[temp_inds_add_keep.get(), walker_inds_add_keep.get(), band_inds_add_keep.get()] = coords_add_keep.get()
        #             new_state.log_like += ll_band_diff.sum(axis=-1).get()
        #             new_state.log_prior += ll_band_diff.sum(axis=-1).get()

        #             waveform_kwargs_tmp = self.waveform_kwargs.copy()
        #             waveform_kwargs_tmp.pop("N")
        #             coords_remove_keep_in = self.parameter_transforms.both_transforms(coords_remove_keep, xp=self.xp)
        #             coords_add_keep_in = self.parameter_transforms.both_transforms(coords_add_keep, xp=self.xp)
                   
        #             coords_in = xp.concatenate([coords_remove_keep_in, coords_add_keep_in], axis=0)
        #             factors = xp.concatenate([+xp.ones(coords_remove_keep_in.shape[0]), -xp.ones(coords_remove_keep_in.shape[0])])
        #             data_index_tmp = (temp_inds_add_keep * nwalkers + walker_inds_add_keep).astype(xp.int32)
        #             data_index_in = xp.concatenate([data_index_tmp, data_index_tmp], dtype=xp.int32)
        #             N_vals_in = xp.concatenate([N_vals_add_keep, N_vals_add_keep])
        #             self.gb.generate_global_template(
        #                 coords_in,
        #                 data_index_in,
        #                 self.mgh.data_list,
        #                 N=N_vals_in,
        #                 data_length=self.data_length,
        #                 data_splits=self.mgh.gpu_splits,
        #                 factors=factors,
        #                 **waveform_kwargs_tmp
        #             )
        #             self.xp.cuda.runtime.deviceSynchronize()

        #     ll_after = (
        #         self.mgh.get_ll(include_psd_info=True)
        #         .flatten()[new_state.supplimental[:]["overall_inds"]]
        #         .reshape(ntemps, nwalkers)
        #     )
        #     # print(np.abs(new_state.log_like - ll_after).max())
        #     store_max_diff = np.abs(new_state.log_like - ll_after).max()
        #     breakpoint()
        #     self.mempool.free_all_blocks()

                # self.mgh.restore_base_injections()

                # for name in new_state.branches.keys():
                #     if name not in ["gb", "gb_fixed"]:
                #         continue
                #     new_state_branch = new_state.branches[name]
                #     coords_here = new_state_branch.coords[new_state_branch.inds]
                #     ntemps, nwalkers, nleaves_max_here, ndim = new_state_branch.shape
                #     try:
                #         group_index = self.xp.asarray(
                #             self.mgh.get_mapped_indices(
                #                 np.repeat(
                #                     np.arange(ntemps * nwalkers).reshape(
                #                         ntemps, nwalkers, 1
                #                     ),
                #                     nleaves_max,
                #                     axis=-1,
                #                 )[new_state_branch.inds]
                #             ).astype(self.xp.int32)
                #         )
                #     except IndexError:
                #         breakpoint()
                #     coords_here_in = self.parameter_transforms.both_transforms(
                #         coords_here, xp=np
                #     )

                #     waveform_kwargs_fill = self.waveform_kwargs.copy()
                #     waveform_kwargs_fill["start_freq_ind"] = self.start_freq_ind

                #     if "N" in waveform_kwargs_fill:
                #         waveform_kwargs_fill.pop("N")

                #     self.mgh.multiply_data(-1.0)
                #     self.gb.generate_global_template(
                #         coords_here_in,
                #         group_index,
                #         self.mgh.data_list,
                #         data_length=self.data_length,
                #         data_splits=self.mgh.gpu_splits,
                #         **waveform_kwargs_fill
                #     )
                #     self.xp.cuda.runtime.deviceSynchronize()
                #     self.mgh.multiply_data(-1.0)


