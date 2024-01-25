# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N
from lisatools.sensitivity import Soms_d_all, Sa_a_all

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

from eryn.moves import GroupStretchMove, Move
from eryn.moves.multipletry import logsumexp, get_mt_computations

from ...diagnostic import inner_product
from lisatools.globalfit.state import State
from lisatools.sampling.prior import GBPriorWrap


__all__ = ["GBSpecialStretchMove"]


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialStretchMove(GroupStretchMove, Move):
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
        num_repeat_proposals=1,
        name=None,
        use_prior_removal=False,
        phase_maximize=False, 
        **kwargs
    ):
        # return_gpu is a kwarg for the stretch move
        GroupStretchMove.__init__(self, *args, return_gpu=True, **kwargs)

        self.gpu_priors = gpu_priors
        self.name = name
        self.num_repeat_proposals = num_repeat_proposals
        self.use_prior_removal = use_prior_removal

        for key in priors:
            if not isinstance(priors[key], ProbDistContainer) and not isinstance(priors[key], GBPriorWrap):
                raise ValueError(
                    "Priors need to be eryn.priors.ProbDistContainer object."
                )
        
        self.priors = priors
        self.gb = gb
        self.stop_here = True

        args = [priors["gb"].priors_in[(0, 1)].rho_star]
        args += [priors["gb"].priors_in[(0, 1)].frequency_prior.min_val, priors["gb"].priors_in[(0, 1)].frequency_prior.max_val]
        for i in range(2, 8):
            args += [priors["gb"].priors_in[i].min_val, priors["gb"].priors_in[i].max_val]
        
        self.gpu_cuda_priors = self.gb.pyPriorPackage(*tuple(args))
        self.gpu_cuda_wrap = self.gb.pyPeriodicPackage(2 * np.pi, np.pi, 2 * np.pi)

        # use gpu from template generator
        # self.use_gpu = gb.use_gpu
        if self.use_gpu:
            self.mempool = self.xp.get_default_memory_pool()

        self.band_edges = band_edges
        self.num_bands = len(band_edges) - 1
        self.start_freq_ind = start_freq_ind
        self.data_length = data_length
        self.waveform_kwargs = waveform_kwargs
        self.parameter_transforms = parameter_transforms
        self.fd = fd
        self.df = (fd[1] - fd[0]).item()
        self.mgh = mgh
        self.phase_maximize = phase_maximize

        self.snr_lim = snr_lim

        self.band_edges = self.xp.asarray(self.band_edges)

        self.rj_proposal_distribution = rj_proposal_distribution
        self.is_rj_prop = self.rj_proposal_distribution is not None

        # setup N vals for bands
        band_mean_f = (self.band_edges[1:] + self.band_edges[:-1]).get() / 2
        self.band_N_vals = xp.asarray(get_N(np.full_like(band_mean_f, 1e-30), band_mean_f, self.waveform_kwargs["T"], self.waveform_kwargs["oversample"]))

    def setup_gbs(self, branch):
        st = time.perf_counter()
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        all_remaining_freqs = coords[0][inds[0]][:, 1]

        all_remaining_cords = coords[0][inds[0]]

        num_remaining = len(all_remaining_freqs)

        all_temp_fs = self.xp.asarray(coords[inds][:, 1])

        # TODO: improve this?
        self.inds_freqs_sorted = self.xp.asarray(np.argsort(all_remaining_freqs))
        self.freqs_sorted = self.xp.asarray(np.sort(all_remaining_freqs))
        self.all_coords_sorted = self.xp.asarray(all_remaining_cords)[
            self.inds_freqs_sorted
        ]

        total_binaries = inds.sum().item()
        still_going = xp.ones(total_binaries, dtype=bool) 
        inds_zero = xp.searchsorted(self.freqs_sorted, all_temp_fs, side="right") - 1
        left_inds = inds_zero - int(self.nfriends / 2)
        right_inds = inds_zero + int(self.nfriends / 2) - 1

        # do right first here
        right_inds[left_inds < 0] = self.nfriends - 1
        left_inds[left_inds < 0] = 0
        
        # do left first here
        left_inds[right_inds > len(self.freqs_sorted) - 1] = len(self.freqs_sorted) - self.nfriends
        right_inds[right_inds > len(self.freqs_sorted) - 1] = len(self.freqs_sorted) - 1

        assert np.all(right_inds - left_inds == self.nfriends - 1)
        assert not np.any(right_inds < 0) and not np.any(right_inds > len(self.freqs_sorted) - 1) and not np.any(left_inds < 0) and not np.any(left_inds > len(self.freqs_sorted) - 1)
        
        jjj = 0
        while np.any(still_going):
            distance_left = np.abs(all_temp_fs[still_going] - self.freqs_sorted[left_inds[still_going]])
            distance_right = np.abs(all_temp_fs[still_going] - self.freqs_sorted[right_inds[still_going]])

            check_move_right = (distance_right <= distance_left)
            check_left_inds = left_inds[still_going][check_move_right] + 1
            check_right_inds = right_inds[still_going][check_move_right] + 1

            new_distance_right = np.abs(all_temp_fs[still_going][check_move_right] - self.freqs_sorted[check_right_inds])

            change_inds = xp.arange(len(all_temp_fs))[still_going][check_move_right][(new_distance_right < distance_left[check_move_right]) & (check_right_inds < len(self.freqs_sorted))]

            left_inds[change_inds] += 1
            right_inds[change_inds] += 1

            stop_inds_right_1 = xp.arange(len(all_temp_fs))[still_going][check_move_right][(check_right_inds >= len(self.freqs_sorted))]

            # last part is just for up here, below it will remove if it is still equal
            stop_inds_right_2 = xp.arange(len(all_temp_fs))[still_going][check_move_right][(new_distance_right >= distance_left[check_move_right]) & (check_right_inds < len(self.freqs_sorted)) & (distance_right[check_move_right] != distance_left[check_move_right])]
            stop_inds_right = xp.concatenate([stop_inds_right_1, stop_inds_right_2])
            assert np.all(still_going[stop_inds_right])

            # equal to should only be left over if it was equal above and moving right did not help
            check_move_left = (distance_left <= distance_right)
            check_left_inds = left_inds[still_going][check_move_left] - 1
            check_right_inds = right_inds[still_going][check_move_left] - 1

            new_distance_left = np.abs(all_temp_fs[still_going][check_move_left] - self.freqs_sorted[check_left_inds])
            
            change_inds = xp.arange(len(all_temp_fs))[still_going][check_move_left][(new_distance_left < distance_right[check_move_left]) & (check_left_inds >= 0)]

            left_inds[change_inds] -= 1
            right_inds[change_inds] -= 1

            stop_inds_left_1 = xp.arange(len(all_temp_fs))[still_going][check_move_left][(check_left_inds < 0)]
            stop_inds_left_2 = xp.arange(len(all_temp_fs))[still_going][check_move_left][(new_distance_left >= distance_right[check_move_left]) & (check_left_inds >= 0)]
            stop_inds_left = xp.concatenate([stop_inds_left_1, stop_inds_left_2])
            
            stop_inds = xp.concatenate([stop_inds_right, stop_inds_left])
            still_going[stop_inds] = False
            # print(jjj, still_going.sum())
            if jjj >= self.nfriends:
                breakpoint()
            jjj += 1

        start_inds = left_inds.copy().get()

        start_inds_all = np.zeros_like(inds, dtype=np.int32)
        start_inds_all[inds] = start_inds.astype(np.int32)

        if "friend_start_inds" not in supps:
            supps.add_objects({"friend_start_inds": start_inds_all})
        else:
            supps[:] = {"friend_start_inds": start_inds_all}

        self.stretch_friends_args_in = tuple([tmp.copy() for tmp in self.all_coords_sorted.T])
        et = time.perf_counter()
        self.mempool.free_all_blocks()
        # print("SETUP:", et - st)
        # start_inds_freq_out = np.zeros((ntemps, nwalkers, nleaves_max), dtype=int)
        # freqs_sorted_here = self.freqs_sorted.get()
        # freqs_remaining_here = all_remaining_freqs

        # start_ind_best = np.zeros_like(freqs_remaining_here, dtype=int)

        # best_index = (
        #     np.searchsorted(freqs_sorted_here, freqs_remaining_here, side="right") - 1
        # )
        # best_index[best_index < self.nfriends] = self.nfriends
        # best_index[best_index >= len(freqs_sorted_here) - self.nfriends] = (
        #     len(freqs_sorted_here) - self.nfriends
        # )
        # check_inds = (
        #     best_index[:, None]
        #     + np.tile(np.arange(2 * self.nfriends), (best_index.shape[0], 1))
        #     - self.nfriends
        # )

        # check_freqs = freqs_sorted_here[check_inds]
        # breakpoint()

        # # batch_count = 1000
        # # split_inds = np.arange(batch_count, freqs_remaining_here.shape[0], batch_count)

        # # splits_remain = np.split(freqs_remaining_here, split_inds)
        # # splits_check = np.split(check_freqs, split_inds)

        # # out = []
        # # for i, (split_r, split_c) in enumerate(zip(splits_remain, splits_check)):
        # #     out.append(np.abs(split_r[:, None] - split_c))
        # #     print(i)

        # # freq_distance = np.asarray(out)

        # freq_distance = np.abs(freqs_remaining_here[:, None] - check_freqs)
        # breakpoint()

        # keep_min_inds = np.argsort(freq_distance, axis=-1)[:, : self.nfriends].min(
        #     axis=-1
        # )
        # start_inds_freq = check_inds[(np.arange(len(check_inds)), keep_min_inds)]

        # start_inds_freq_out[inds] = start_inds_freq

        # start_inds_freq_out[~inds] = -1

        # if "friend_start_inds" not in supps:
        #     supps.add_objects({"friend_start_inds": start_inds_freq_out})
        # else:
        #     supps[:] = {"friend_start_inds": start_inds_freq_out}

        # self.all_friends_start_inds_sorted = self.xp.asarray(
        #     start_inds_freq_out[inds][self.inds_freqs_sorted.get()]
        # )

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

    def new_find_friends(self, name, inds_in):
        inds_start_freq_to_move = self.current_friends_start_inds[tuple(inds_in)]

        deviation = self.xp.random.randint(
            0, self.nfriends, size=len(inds_start_freq_to_move)
        )

        inds_keep_friends = inds_start_freq_to_move + deviation

        inds_keep_friends[inds_keep_friends < 0] = 0
        inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted)] = (
            len(self.all_coords_sorted) - 1
        )

        gb_points_for_move = self.all_coords_sorted[
            inds_keep_friends
        ]

        return gb_points_for_move

    def setup(self, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name != "gb":
                continue

            if not self.is_rj_prop and self.time % self.n_iter_update == 0:
                self.setup_gbs(branch)

            elif self.is_rj_prop and self.time == 0:
                ndim = branch.shape[-1]
                self.stretch_friends_args_in = tuple([xp.array([]) for _ in range(ndim)])

            # update any shifted start inds due to tempering (need to do this every non-rj move)
            """if not self.is_rj_prop:
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
            """

            self.mempool.free_all_blocks()

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
        band_temps = xp.asarray(state.band_info["band_temps"].copy())

        if self.is_rj_prop:
            orig_store = new_state.log_like[0].copy()
        gb_coords = xp.asarray(new_state.branches["gb"].coords)

        self.mempool.free_all_blocks()

        # self.mgh.map = new_state.supplimental.holder["overall_inds"].flatten()

        # data should not be whitened

        ntemps, nwalkers, nleaves_max, ndim = state.branches_coords["gb"].shape

        group_temp_finder = [
            self.xp.repeat(self.xp.arange(ntemps), nwalkers * nleaves_max).reshape(
                ntemps, nwalkers, nleaves_max
            ),
            self.xp.tile(self.xp.arange(nwalkers), (ntemps, nleaves_max, 1)).transpose(
                (0, 2, 1)
            ),
            self.xp.tile(self.xp.arange(nleaves_max), ((ntemps, nwalkers, 1))),
        ]

        """(
            gb_coords,
            gb_inds_orig,
            points_curr,
            prior_all_curr,
            gb_inds,
            N_vals_in,
            prop_to_curr_map,
            factors,
            inds_curr,
            proposal_specific_information
        ) = self.get_special_proposal_setup(model, new_state, state, group_temp_finder)"""

        waveform_kwargs_now = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs_now:
            waveform_kwargs_now.pop("N")
        waveform_kwargs_now["start_freq_ind"] = self.start_freq_ind

        # if self.is_rj_prop:
        #     print("START:", new_state.log_like[0])
        log_like_tmp = self.xp.asarray(new_state.log_like)
        log_prior_tmp = self.xp.asarray(new_state.log_prior)

        self.mempool.free_all_blocks()

        gb_inds = self.xp.asarray(new_state.branches["gb"].inds)
        gb_inds_orig = gb_inds.copy()

        data = self.mgh.data_list
        base_data = self.mgh.base_data_list
        psd = self.mgh.psd_list
        lisasens = self.mgh.lisasens_list

        # do unique for band size as separator between asynchronous kernel launches
        # band_indices = self.xp.asarray(new_state.branches["gb"].branch_supplimental.holder["band_inds"])
        band_indices = self.xp.searchsorted(self.band_edges, xp.asarray(new_state.branches["gb"].coords[:, :, :, 1]).flatten() / 1e3, side="right").reshape(new_state.branches["gb"].coords[:, :, :, 1].shape) - 1
            
        # N_vals_in = self.xp.asarray(new_state.branches["gb"].branch_supplimental.holder["N_vals"])
        points_curr = self.xp.asarray(new_state.branches["gb"].coords)
        points_curr_orig = points_curr.copy()
        # N_vals_in_orig = N_vals_in.copy()
        band_indices_orig = band_indices.copy()
        
        if self.is_rj_prop:
            if isinstance(self.rj_proposal_distribution["gb"], list):
                raise NotImplementedError
                assert len(self.rj_proposal_distribution["gb"]) == ntemps
                proposal_logpdf = xp.zeros((points_curr.shape[0], np.prod(points_curr.shape[1:-1])))
                for t in range(ntemps):
                    new_sources = xp.full_like(points_curr[t, ~gb_inds[t]], np.nan)
                    fix = xp.full(new_sources.shape[0], True)
                    while xp.any(fix):
                        new_sources[fix] = self.rj_proposal_distribution["gb"][t].rvs(size=fix.sum().item())
                        fix = xp.any(xp.isnan(new_sources), axis=-1)
                    points_curr[t, ~gb_inds[t]] = new_sources
                    band_indices[t, ~gb_inds[t]] = self.xp.searchsorted(self.band_edges, new_sources[:, 1] / 1e3, side="right") - 1
                    # change gb_inds
                    gb_inds[t, :] = True
                    assert np.all(gb_inds[t])
                    proposal_logpdf[t] = self.rj_proposal_distribution["gb"][t].logpdf(
                        points_curr[t, gb_inds[t]]
                    )

                proposal_logpdf = proposal_logpdf.flatten().copy()
            
            else:
                new_sources = xp.full_like(points_curr[~gb_inds], np.nan)
                fix = xp.full(new_sources.shape[0], True)
                while xp.any(fix):
                    if self.name == "rj_prior":
                        new_sources[fix] = self.rj_proposal_distribution["gb"].rvs(size=fix.sum().item(), psds=self.mgh.psd_shaped[0][0], walker_inds=group_temp_finder[1][~gb_inds][fix])
                    else:
                        new_sources[fix] = self.rj_proposal_distribution["gb"].rvs(size=fix.sum().item())
                    
                    fix = xp.any(xp.isnan(new_sources), axis=-1)
                points_curr[~gb_inds] = new_sources

                band_indices[~gb_inds] = self.xp.searchsorted(self.band_edges, new_sources[:, 1] / 1e3, side="right") - 1
                # N_vals_in[~gb_inds] = self.xp.asarray(
                #     get_N(
                #         xp.full_like(new_sources[:, 1], 1e-30),
                #         new_sources[:, 1] / 1e3,
                #         self.waveform_kwargs["T"],
                #         self.waveform_kwargs["oversample"],
                #         xp=self.xp
                #     )
                # )

                # change gb_inds
                gb_inds[:] = True
                assert np.all(gb_inds)
                if self.name == "rj_prior":
                    proposal_logpdf = self.rj_proposal_distribution["gb"].logpdf(
                        points_curr[gb_inds], psds=self.mgh.psd_shaped[0][0], walker_inds=group_temp_finder[1][gb_inds]
                    )
                else:
                    proposal_logpdf = self.rj_proposal_distribution["gb"].logpdf(
                        points_curr[gb_inds]
                    )

            factors = (proposal_logpdf * -1) * (~gb_inds_orig).flatten() + (proposal_logpdf * +1) * (gb_inds_orig).flatten()

            if self.name == "rj_prior" and self.use_prior_removal:
                factors[~gb_inds_orig.flatten()] = -1e300
        else:
            factors = xp.zeros(gb_inds_orig.sum().item())

        start_inds_all = xp.asarray(new_state.branches["gb"].branch_supplimental.holder["friend_start_inds"], dtype=xp.int32)[gb_inds]
        points_curr = points_curr[gb_inds]
        # N_vals_in = N_vals_in[gb_inds]
        band_indices = band_indices[gb_inds]
        gb_inds_in = gb_inds_orig[gb_inds]

        temp_indices = group_temp_finder[0][gb_inds]
        walker_indices = group_temp_finder[1][gb_inds]
        leaf_indices = group_temp_finder[2][gb_inds]
        
        unique_N = self.xp.unique(self.band_N_vals)
        # remove 0
        unique_N = unique_N[unique_N != 0]

        do_synchronize = False
        device = self.xp.cuda.runtime.getDevice()

        units = 2 if not self.is_rj_prop else 2
        # random start to rotation around 
        start_unit = model.random.randint(units)

        ll_after = (
            self.mgh.get_ll(include_psd_info=True)
        )
        # print(np.abs(new_state.log_like - ll_after).max())
        store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        # print("CHECKING 0:", store_max_diff, self.is_rj_prop)
        self.check_ll_inject(new_state)

        per_walker_band_proposals = xp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_accepted = xp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        
        total_keep = 0
        for tmp in range(units):
            remainder = (start_unit + tmp) % units
            all_inputs = []
            band_bookkeep_info = []
            indiv_info = []
            current_parameter_arrays = []
            N_vals_list = []
            output_info = []
            inds_list = []
            inds_orig_list = []
            bands_list = []
            for N_now in unique_N:
                N_now = N_now.item()
                if N_now == 0:  #  or N_now != 1024:
                    continue

                # old_data_1 = self.mgh.data_shaped[0][0][11].copy()
                # checkit = np.where(new_state.supplimental[:]["overall_inds"] == 0)

                # TODO; check the maximum allowable band
                keep = (
                    (band_indices % units == remainder)
                    # & (temp_indices == 0)  #  & (walker_indices == 2)  #  & (band_indices < 50)
                    & (self.band_N_vals[band_indices] == N_now)
                    & (band_indices < len(self.band_edges) - 2)
                    # & (band_indices == 1)
                )  #  & (band_indices == 501) #  # & (N_vals_in <= 256) & (temp_inds == checkit[0].item()) & (walker_inds == checkit[1].item()) #    & (band_indices < 540)  #  &  (temp_inds == 0) & (walker_inds == 0)

                # if self.time > 0:
                #     keep[np.where(keep)[0][1:]] = False
                """if self.time > 0:
                    keep = (
                        (band_indices % units == remainder)
                        & (band_indices == 346) & (walker_indices == 9) & (temp_indices == 0)
                        & (self.band_N_vals[band_indices] == N_now)
                        & (band_indices < len(self.band_edges) - 2)
                    )  #  & (band_indices == 501) #  # & (N_vals_in <= 256) & (temp_inds == checkit[0].item()) & (walker_inds == checkit[1].item()) #    & (band_indices < 540)  #  &  (temp_inds == 0) & (walker_inds == 0)
                    if np.any(keep):
                        breakpoint()"""

                if keep.sum().item() == 0:
                    continue

                total_keep += keep.sum().item()
                # for testing
                # keep[:] = False
                # keep[tmp_keep] = True
                # keep[3000:3020:1] = True

                # permutate for rj just in case
                permute_inds = xp.random.permutation(xp.arange(keep.sum()))
                params_curr = points_curr[keep][permute_inds]
                start_inds_here = start_inds_all[keep][permute_inds]
                inds_here = gb_inds_in[keep][permute_inds]
                factors_here = factors[keep][permute_inds]

                prior_all_curr_here = self.gpu_priors["gb"].logpdf(params_curr, psds=self.mgh.psd_shaped[0][0], walker_inds=group_temp_finder[1][gb_inds][keep][permute_inds])
                if not self.is_rj_prop:
                    assert xp.all(~xp.isinf(prior_all_curr_here))

                temp_inds_here = temp_indices[keep][permute_inds]
                walker_inds_here = walker_indices[keep][permute_inds]
                leaf_inds_here = leaf_indices[keep][permute_inds]
                band_inds_here = band_indices[keep][permute_inds]

                special_band_inds = (temp_inds_here * nwalkers + walker_inds_here) * int(1e6) + band_inds_here

                sort_special = xp.argsort(special_band_inds)

                params_curr = params_curr[sort_special]
                inds_here = inds_here[sort_special]
                factors_here = factors_here[sort_special]
                start_inds_here = start_inds_here[sort_special]
                prior_all_curr_here = prior_all_curr_here[sort_special]
                temp_inds_here = temp_inds_here[sort_special]
                walker_inds_here = walker_inds_here[sort_special]
                leaf_inds_here = leaf_inds_here[sort_special]
                band_inds_here = band_inds_here[sort_special]
                special_band_inds = special_band_inds[sort_special]

                # assert np.all(np.sort(special_band_inds_sorted_here) == special_band_inds_sorted_here)
                
                (
                    uni_special_band_inds_here,
                    uni_index_special_band_inds_here,
                    uni_count_special_band_inds_here,
                ) = self.xp.unique(
                    special_band_inds, return_index=True, return_counts=True
                )
                
                band_start_bin_ind_here = uni_index_special_band_inds_here.astype(
                    np.int32
                )

                band_num_bins_here = uni_count_special_band_inds_here.astype(np.int32)

                band_inds = band_inds_here[uni_index_special_band_inds_here]
                band_temps_inds = temp_inds_here[uni_index_special_band_inds_here]
                band_walkers_inds = walker_inds_here[uni_index_special_band_inds_here]
                
                band_inv_temp_vals_here = band_temps[band_inds, band_temps_inds]

                indiv_info.append((temp_inds_here, walker_inds_here, leaf_inds_here))

                band_bookkeep_info.append(
                    (band_temps_inds, band_walkers_inds, band_inds)
                )

                data_index_here = (((band_inds + 1) % 2) * nwalkers + band_walkers_inds).astype(np.int32)
                noise_index_here = (band_walkers_inds).astype(np.int32)

                # for updates
                update_data_index_here = (((band_inds + 0) % 2) * nwalkers + band_walkers_inds).astype(np.int32)
                
                L_contribution_here = xp.zeros_like(band_inds, dtype=complex)
                p_contribution_here = xp.zeros_like(band_inds, dtype=complex)

                buffer = 5  # bins

                start_band_index = ((self.band_edges[band_inds] / self.df).astype(int) - N_now - 1).astype(np.int32)
                end_band_index = (((self.band_edges[band_inds + 1]) / self.df).astype(int) + N_now + 1).astype(np.int32)
                
                start_band_index[start_band_index < 0] = 0
                end_band_index[end_band_index >= len(self.fd)] = len(self.fd) - 1
                
                band_lengths = end_band_index - start_band_index

                start_interest_band_index = ((self.band_edges[band_inds] / self.df).astype(int)).astype(np.int32)  #  - N_now).astype(np.int32)
                end_interest_band_index = (((self.band_edges[band_inds + 1]) / self.df).astype(int)).astype(np.int32)  #  + N_now).astype(np.int32)
                
                start_interest_band_index[start_interest_band_index < 0] = 0
                end_interest_band_index[end_interest_band_index >= len(self.fd)] = len(self.fd) - 1
                
                band_interest_lengths = end_interest_band_index - start_interest_band_index

                if False:  # self.is_rj_prop:
                    fmin_allow = ((self.band_edges[band_inds] / self.df).astype(int) + 1) * self.df
                    fmax_allow = ((self.band_edges[band_inds + 1] / self.df).astype(int) - 1) * self.df
                    
                else:
                    # proposal limits in a band
                    fmin_allow = ((self.band_edges[band_inds] / self.df).astype(int)- (N_now / 2)) * self.df
                    fmax_allow = ((self.band_edges[band_inds + 1] / self.df).astype(int) + (N_now / 2 )) * self.df

                # if self.is_rj_prop:
                #     breakpoint()
                fmin_allow[fmin_allow < self.band_edges[0]] = self.band_edges[0]
                fmax_allow[fmax_allow > self.band_edges[-1]] = self.band_edges[-1]

                max_data_store_size = band_lengths.max().item()

                num_bands_here = len(band_inds)

                # makes in-model effectively not tempered 
                # if not self.is_rj_prop:
                #    band_inv_temp_vals_here[:] = 1.0

                params_curr_separated = tuple([params_curr[:, i].copy() for i in range(params_curr.shape[1])])
                params_curr_separated_orig = tuple([params_curr[:, i].copy() for i in range(params_curr.shape[1])])
                params_extra_params = (
                    self.waveform_kwargs["T"],
                    self.waveform_kwargs["dt"],
                    N_now,
                    params_curr.shape[0],
                    self.start_freq_ind, 
                    Soms_d_all["sangria"] ** (1/2),
                    Sa_a_all["sangria"] ** (1/2),
                    1e-100, 0.0, 0.0, 0.0, 0.0,  # foreground params for snr -> amp transform
                )

                gb_params_curr_in = self.gb.pyGalacticBinaryParams(
                    *(params_curr_separated + params_curr_separated_orig + params_extra_params)
                )
                
                current_parameter_arrays.append(params_curr_separated)

                data_package = self.gb.pyDataPackage(
                    data[0][0],
                    data[1][0],
                    base_data[0][0],
                    base_data[1][0],
                    psd[0][0],
                    psd[1][0],
                    lisasens[0][0],
                    lisasens[1][0],
                    self.df,
                    self.data_length,
                    self.nwalkers * self.ntemps,
                    self.nwalkers * self.ntemps
                )

                loc_band_index = xp.arange(num_bands_here, dtype=xp.int32)
                band_package = self.gb.pyBandPackage(
                    loc_band_index,
                    data_index_here,
                    noise_index_here,
                    band_start_bin_ind_here,  # uni_index
                    band_num_bins_here,  # uni_count
                    start_band_index,
                    band_lengths,
                    start_interest_band_index,
                    band_interest_lengths,
                    num_bands_here,
                    max_data_store_size,
                    fmin_allow,
                    fmax_allow,
                    update_data_index_here,
                    self.ntemps,
                    band_inds.astype(np.int32),
                    band_walkers_inds.astype(np.int32),
                    band_temps_inds.astype(np.int32),
                    xp.zeros_like(band_temps_inds, dtype=np.int32),  # swaps propsoed
                    xp.zeros_like(band_temps_inds, dtype=np.int32)     # swaps accepted
                )

                accepted_out_here = xp.zeros_like(band_start_bin_ind_here, dtype=xp.int32)

                mcmc_info = self.gb.pyMCMCInfo(
                    L_contribution_here,
                    p_contribution_here,
                    prior_all_curr_here,
                    accepted_out_here,
                    band_inv_temp_vals_here,  # band_inv_temp_vals
                    self.is_rj_prop,
                    self.phase_maximize,
                    self.snr_lim,
                )

                if not self.is_rj_prop:
                    num_proposals_here = self.num_repeat_proposals
                    
                else:
                    num_proposals_here = 1

                num_proposals_per_band = band_num_bins_here * num_proposals_here

                assert start_inds_here.dtype == np.int32

                proposal_info = self.gb.pyStretchProposalPackage(
                    *(self.stretch_friends_args_in + (self.nfriends, len(self.stretch_friends_args_in[0]), num_proposals_here, self.a, ndim, inds_here, factors_here, start_inds_here))
                )

                output_info.append([L_contribution_here, p_contribution_here, accepted_out_here, num_proposals_per_band])

                inputs_now = (
                    data_package,
                    band_package,
                    gb_params_curr_in,
                    mcmc_info,
                    self.gpu_cuda_priors,
                    proposal_info,
                    self.gpu_cuda_wrap,
                    device,
                    do_synchronize,
                )

                # if self.is_rj_prop:
                #     prior_check = xp.zeros(params_curr.shape[0])
                #     self.gb.check_prior_vals(prior_check, self.gpu_cuda_priors, gb_params_curr_in, 100)
                #     prior_check2 = self.gpu_priors["gb"].logpdf(params_curr)
                #     breakpoint()


                N_vals_list.append(N_now)
                inds_list.append(inds_here)
                inds_orig_list.append(inds_here.copy())
                bands_list.append(band_inds_here)
                all_inputs.append(inputs_now)
                st = time.perf_counter()
                # print(params_curr_separated[0].shape[0])
                # if self.is_rj_prop:
                # if self.time > 0:
                #     breakpoint()

                # tmp_check = self.mgh.channel1_data[0][11 * self.data_length + 3911].real + self.mgh.channel1_data[0][29 * self.data_length + 3911].real - self.mgh.channel1_base_data[0][11 * self.data_length + 3911].real
                # print(f"BEFORE {tmp_check}, {self.mgh.channel1_data[0][11 * self.data_length + 3911].real} , {self.mgh.channel1_data[0][29 * self.data_length + 3911].real} , {self.mgh.channel1_base_data[0][11 * self.data_length + 3911].real} ")
                # print(f"BEFORE2 {params_curr_separated[1].min().item()} ")
                self.gb.SharedMemoryMakeNewMove_wrap(*inputs_now)
                self.xp.cuda.runtime.deviceSynchronize()
                # tmp_check = self.mgh.channel1_data[0][11 * self.data_length + 3911].real + self.mgh.channel1_data[0][29 * self.data_length + 3911].real - self.mgh.channel1_base_data[0][11 * self.data_length + 3911].real
                # print(f"After {tmp_check}, {self.mgh.channel1_data[0][11 * self.data_length + 3911].real} , {self.mgh.channel1_data[0][29 * self.data_length + 3911].real} , {self.mgh.channel1_base_data[0][11 * self.data_length + 3911].real} ")
                # print(f"After2 {params_curr_separated[1].min().item()} ")
                
                et = time.perf_counter()
                # print(et - st, N_now)
                # breakpoint()
                
            self.xp.cuda.runtime.deviceSynchronize()
            new_point_info = []
            ll_diff_info = []
            for (
                band_info,
                indiv_info_now,
                N_now,
                outputs,
                current_parameters,
                inds,
                inds_prev,
                bands
            ) in zip(
                band_bookkeep_info, indiv_info, N_vals_list, output_info, current_parameter_arrays, inds_list, inds_orig_list, bands_list
            ):
                ll_contrib_now = outputs[0]
                lp_contrib_now = outputs[1]
                accepted_now = outputs[2]
                num_proposals_now = outputs[3]
            
                per_walker_band_proposals[band_info] += num_proposals_now
                per_walker_band_accepted[band_info] += accepted_now

                # remove accepted
                # print(accepted_now.sum(0) / accepted_now.shape[0])
                temp_tmp, walker_tmp, leaf_tmp = (
                    indiv_info_now[0],
                    indiv_info_now[1],
                    indiv_info_now[2],
                )

                # updates related to newly added sources
                if self.is_rj_prop:
                    gb_inds_orig_check = gb_inds_orig.copy()
                    gb_coords[(temp_tmp[inds], walker_tmp[inds], leaf_tmp[inds])] = xp.asarray(current_parameters).T[inds]

                    gb_inds_orig[(temp_tmp, walker_tmp, leaf_tmp)] = inds
                    new_state.branches_supplimental["gb"].holder["N_vals"][
                        (temp_tmp[inds].get(), walker_tmp[inds].get(), leaf_tmp[inds].get())
                    ] = N_now
                    new_state.branches_supplimental["gb"].holder["band_inds"][
                        (temp_tmp[inds].get(), walker_tmp[inds].get(), leaf_tmp[inds].get())
                    ] = bands[inds].get()

                else:
                    gb_coords[(temp_tmp, walker_tmp, leaf_tmp)] = xp.asarray(current_parameters).T

                    new_band_inds = self.xp.searchsorted(self.band_edges, xp.asarray(current_parameters).T[:, 1] / 1e3, side="right") - 1

                    # if np.any(new_band_inds != bands):
                    #     print(new_band_inds, bands)
                ll_change = self.xp.zeros((ntemps, nwalkers, len(self.band_edges)))
                lp_change = self.xp.zeros((ntemps, nwalkers, len(self.band_edges)))

                self.xp.cuda.runtime.deviceSynchronize()

                ll_change[band_info] = ll_contrib_now

                ll_diff_info.append(ll_change.copy())
                
                ll_adjustment = ll_change.sum(axis=-1)
                log_like_tmp += ll_adjustment

                self.xp.cuda.runtime.deviceSynchronize()

                lp_change[band_info] = lp_contrib_now

                lp_adjustment = lp_change.sum(axis=-1)
                log_prior_tmp += lp_adjustment

            self.xp.cuda.runtime.deviceSynchronize()
            if True:  # self.time > 0:
                ll_after = (
                    self.mgh.get_ll(include_psd_info=True)
                )
                # print(np.abs(new_state.log_like - ll_after).max())
                store_max_diff = np.abs(log_like_tmp[0].get() - ll_after).max()
                # print("CHECKING in:", tmp, store_max_diff)
                if store_max_diff > 3e-4:
                    print("LARGER ERROR:", store_max_diff)
                    self.check_ll_inject(new_state)
                    # self.mgh.get_ll(include_psd_info=True, stop=True)

        new_state.branches["gb"].coords[:] = gb_coords.get()
        if self.is_rj_prop:
            new_state.branches["gb"].inds[:] = gb_inds_orig.get()
        new_state.log_like[:] = log_like_tmp.get()
        new_state.log_prior[:] = log_prior_tmp.get()

        # get updated bands inds ONLY FOR COLD CHAIN and 
        # propogate changes to higher temperatures
        new_freqs = gb_coords[gb_inds_orig, 1]
        new_band_inds = (xp.searchsorted(self.band_edges, new_freqs / 1e3, side="right") - 1)
        new_state.branches["gb"].branch_supplimental.holder["band_inds"][gb_inds_orig.get()] = new_band_inds.get()

        ll_after = (
            self.mgh.get_ll(include_psd_info=True)
        )
        # print(np.abs(new_state.log_like - ll_after).max())
        store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        # print("CHECKING 1:", store_max_diff, self.is_rj_prop)
        # if self.time > 0:
        #     breakpoint()
        #     self.check_ll_inject(new_state)

        
        if not self.is_rj_prop:
            # if self.time > 0:
            #     breakpoint()
            # check2 = self.mgh.get_ll()
            old_band_inds_cold_chain = state.branches["gb"].branch_supplimental.holder["band_inds"][0] * state.branches["gb"].inds[0]
            new_band_inds_cold_chain = new_state.branches["gb"].branch_supplimental.holder["band_inds"][0] * state.branches["gb"].inds[0]
            inds_band_change_cold_chain = np.where(new_band_inds_cold_chain != old_band_inds_cold_chain)
            # when adjusting temperatures, be careful here
            if len(inds_band_change_cold_chain[0]) > 0:
                check2 = self.mgh.get_ll(include_psd_info=True)
                # print("SWITCH", len(inds_band_change_cold_chain[0]))
                walker_inds_change_cold_chain = np.tile(inds_band_change_cold_chain[0], (self.ntemps - 1, 1)).flatten()
                old_leaf_inds_change_cold_chain = np.tile(inds_band_change_cold_chain[1], (self.ntemps - 1, 1)).flatten()
                new_temp_inds_change_cold_chain = np.repeat(np.arange(1, self.ntemps), len(inds_band_change_cold_chain[0]))

                special_check = new_temp_inds_change_cold_chain * self.nwalkers + walker_inds_change_cold_chain

                uni_special_check, uni_special_check_count = np.unique(special_check, return_counts=True)

                # get new leaf positions
                temp_leaves = np.ones_like(group_temp_finder[2].reshape(self.ntemps * self.nwalkers, -1).get()[uni_special_check], dtype=int) * (~new_state.branches["gb"].inds.reshape(self.ntemps * self.nwalkers, -1)[uni_special_check])
                temp_leaves_2 = np.cumsum(temp_leaves, axis=-1)
                temp_leaves_2[new_state.branches["gb"].inds.reshape(self.ntemps * self.nwalkers, -1)[uni_special_check]] = -1
                
                leaf_guide_here = np.tile(np.arange(nleaves_max), (len(uni_special_check), 1))
                new_leaf_inds_change_cold_chain = leaf_guide_here[((temp_leaves_2 >= 0) & (temp_leaves_2 <= uni_special_check_count[:, None]))]
                try:
                    assert np.all(~new_state.branches["gb"].inds[new_temp_inds_change_cold_chain, walker_inds_change_cold_chain, new_leaf_inds_change_cold_chain])
                except IndexError:
                    breakpoint()
                    
                new_state.branches["gb"].inds[new_temp_inds_change_cold_chain, walker_inds_change_cold_chain, new_leaf_inds_change_cold_chain] = True
                
                new_state.branches["gb"].coords[new_temp_inds_change_cold_chain, walker_inds_change_cold_chain, new_leaf_inds_change_cold_chain] = new_state.branches["gb"].coords[np.zeros_like(walker_inds_change_cold_chain), walker_inds_change_cold_chain, old_leaf_inds_change_cold_chain]
                new_state.branches["gb"].branch_supplimental.holder["band_inds"][new_temp_inds_change_cold_chain, walker_inds_change_cold_chain, new_leaf_inds_change_cold_chain] = new_state.branches["gb"].branch_supplimental.holder["band_inds"][np.zeros_like(walker_inds_change_cold_chain), walker_inds_change_cold_chain, old_leaf_inds_change_cold_chain]
                new_state.branches["gb"].branch_supplimental.holder["N_vals"][new_temp_inds_change_cold_chain, walker_inds_change_cold_chain, new_leaf_inds_change_cold_chain] = new_state.branches["gb"].branch_supplimental.holder["N_vals"][np.zeros_like(walker_inds_change_cold_chain), walker_inds_change_cold_chain, old_leaf_inds_change_cold_chain]
            
                # adjust data
                adjust_binaries = xp.asarray(new_state.branches["gb"].coords[0, inds_band_change_cold_chain[0], inds_band_change_cold_chain[1]])
                adjust_binaries_in = self.parameter_transforms.both_transforms(
                    adjust_binaries, xp=xp
                )
                adjust_walker_inds = inds_band_change_cold_chain[0]
                adjust_band_new = new_state.branches["gb"].branch_supplimental.holder["band_inds"][0, inds_band_change_cold_chain[0], inds_band_change_cold_chain[1]]
                adjust_band_old = state.branches["gb"].branch_supplimental.holder["band_inds"][0, inds_band_change_cold_chain[0], inds_band_change_cold_chain[1]]
                # N_vals_in = new_state.branches["gb"].branch_supplimental.holder["N_vals"][0, inds_band_change_cold_chain[0], inds_band_change_cold_chain[1]]
                
                adjust_binaries_in_in = xp.concatenate([
                    adjust_binaries_in,
                    adjust_binaries_in
                ], axis=0)

                data_index = xp.concatenate([
                    xp.asarray(((adjust_band_old + 0) % 2) * nwalkers + adjust_walker_inds),
                    xp.asarray(((adjust_band_new + 0) % 2) * nwalkers + adjust_walker_inds)
                ]).astype(xp.int32)

                N_vals_in_in = xp.concatenate([
                    xp.asarray(self.band_N_vals[adjust_band_old]),
                    xp.asarray(self.band_N_vals[adjust_band_new])
                ])

                factors = xp.concatenate([
                    +xp.ones_like(adjust_band_old, dtype=xp.float64),  # remove
                    -xp.ones_like(adjust_band_old, dtype=xp.float64)  # add
                ])

                self.gb.generate_global_template(
                    adjust_binaries_in_in,
                    data_index,
                    self.mgh.data_list,
                    N=N_vals_in_in,
                    factors=factors,
                    data_length=self.mgh.data_length,
                    data_splits=self.mgh.gpu_splits,
                    **waveform_kwargs_now
                )
                check3 = self.mgh.get_ll(include_psd_info=True)

                # print(check3 - check2)

                if np.any(
                    self.band_N_vals[adjust_band_old] !=
                    self.band_N_vals[adjust_band_new]
                ):
                    walkers_focus = np.unique(inds_band_change_cold_chain[0][
                        self.band_N_vals.get()[adjust_band_old] !=
                        self.band_N_vals.get()[adjust_band_new]
                    ])
                    # print(f"specific change in N across boundary: {walkers_focus}")
                    new_state.log_like[0, walkers_focus] = self.mgh.get_ll(include_psd_info=True)[walkers_focus]

        # self.check_ll_inject(new_state)
        # breakpoint()
        ll_after = (
            self.mgh.get_ll(include_psd_info=True)
        )
        # print(np.abs(new_state.log_like - ll_after).max())
        store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        # print("CHECKING 2:", store_max_diff, self.is_rj_prop)

        self.mempool.free_all_blocks()
        # get accepted fraction
        if not self.is_rj_prop:
            accepted_check_tmp = np.zeros_like(
                state.branches_inds["gb"], dtype=bool
            )
            accepted_check_tmp[state.branches_inds["gb"]] = np.all(
                np.abs(
                    new_state.branches_coords["gb"][
                        state.branches_inds["gb"]
                    ]
                    - state.branches_coords["gb"][state.branches_inds["gb"]]
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
                new_state.branches_inds["gb"] == (~state.branches_inds["gb"])
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
                new_state.branches_coords["gb"]
                - state.branches_coords["gb"]
            )
            > 0.0,
            axis=-1,
        ).sum(axis=(2,))
        tmp2 = new_state.branches_inds["gb"].sum(axis=(2,))

        # add to move-specific accepted information
        self.accepted += tmp1
        if isinstance(self.num_proposals, int):
            self.num_proposals = tmp2
        else:
            self.num_proposals += tmp2

        new_inds = xp.asarray(new_state.branches_inds["gb"])
            
        # in-model inds will not change
        tmp_freqs_find_bands = xp.asarray(new_state.branches_coords["gb"][:, :, :, 1])

        # calculate current band counts
        band_here = (xp.searchsorted(self.band_edges, tmp_freqs_find_bands.flatten() / 1e3, side="right") - 1).reshape(tmp_freqs_find_bands.shape)

        # get binaries per band
        special_band_here_num_per_band = ((group_temp_finder[0] * nwalkers + group_temp_finder[1]) * int(1e6) + band_here)[new_inds]
        unique_special_band_here_num_per_band, unique_special_band_here_num_per_band_count = xp.unique(special_band_here_num_per_band, return_counts=True)
        temp_walker_index_num_per_band = (unique_special_band_here_num_per_band / 1e6).astype(int)
        temp_index_num_per_band = (temp_walker_index_num_per_band / nwalkers).astype(int)
        walker_index_num_per_band = temp_walker_index_num_per_band - temp_index_num_per_band * nwalkers
        band_index_num_per_band = unique_special_band_here_num_per_band - temp_walker_index_num_per_band * int(1e6)

        per_walker_band_counts = xp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_counts[temp_index_num_per_band, walker_index_num_per_band, band_index_num_per_band] = unique_special_band_here_num_per_band_count
        
        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        band_swaps_accepted = np.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = np.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        current_band_counts = np.zeros((len(self.band_edges) - 1, self.ntemps), dtype=int)
        
        # if self.is_rj_prop:
        #     print("1st count check:", new_state.branches["gb"].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])
        
        # if self.time > 0:
        #     self.check_ll_inject(new_state)
        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and self.is_rj_prop
            # and False
        ):

            new_coords_after_tempering = xp.asarray(np.zeros_like(new_state.branches["gb"].coords))
            new_inds_after_tempering = xp.asarray(np.zeros_like(new_state.branches["gb"].inds))

            # TODO: check if N changes / need to deal with that
            betas = self.temperature_control.betas

            # cannot find them yourself because of higher temps moving across band edge / need supplimental band inds
            band_inds_temp = new_state.branches["gb"].branch_supplimental.holder["band_inds"][new_state.branches["gb"].inds]
            temp_inds_temp = group_temp_finder[0].get()[new_state.branches["gb"].inds]
            walker_inds_temp = group_temp_finder[1].get()[new_state.branches["gb"].inds]

            bands_guide = np.tile(np.arange(self.num_bands), (self.ntemps, self.nwalkers, 1)).transpose(2, 1, 0).flatten()
            temps_guide = np.repeat(np.arange(self.ntemps)[:, None], self.nwalkers * self.num_bands).reshape(self.ntemps, self.nwalkers, self.num_bands).transpose(2, 1, 0).flatten()
            walkers_guide = np.repeat(np.arange(self.nwalkers)[:, None], self.ntemps * self.num_bands).reshape(self.nwalkers, self.ntemps, self.num_bands).transpose(2, 0, 1).flatten()
            
            walkers_permuted = np.asarray([np.random.permutation(np.arange(self.nwalkers)) for _ in range(self.ntemps * self.num_bands)]).reshape(self.num_bands, self.ntemps, self.nwalkers).transpose(0, 2, 1).flatten()

            # special_inds_guide = bands_guide * int(1e6) + walkers_permuted * int(1e3) + temps_guide

            coords_in = new_state.branches["gb"].coords[new_state.branches["gb"].inds]
            
            # N_vals_in = new_state.branches["gb"].branch_supplimental.holder["N_vals"][new_state.branches["gb"].inds]
            unique_N = np.unique(self.band_N_vals).get()
            # remove 0
            unique_N = unique_N[unique_N != 0]

            bands_in = band_inds_temp
            temps_in = temp_inds_temp
            walkers_in = walker_inds_temp

            main_gpu = self.xp.cuda.runtime.getDevice()

            walkers_info = walkers_permuted if self.temperature_control.permute else walkers_guide
            units = 2
            start_unit = np.random.randint(0, units)
            for unit in range(units):
                current_band_remainder = (start_unit + unit) % units
                for N_now in unique_N:
                    keep = (bands_in % units == current_band_remainder) & (self.band_N_vals[xp.asarray(bands_in)].get() == N_now)
                    keep_base = (bands_guide % units == current_band_remainder) & (self.band_N_vals[xp.asarray(bands_guide)].get() == N_now)
                    bands_in_tmp = bands_in[keep]
                    temps_in_tmp = temps_in[keep]
                    walkers_in_tmp = walkers_in[keep]
                    coords_in_tmp = coords_in[keep]

                    # adjust for N_val
                    special_inds_bins = bands_in_tmp * int(1e8) + walkers_in_tmp * int(1e4) + temps_in_tmp

                    bands_guide_keep = bands_guide[keep_base]
                    temps_guide_keep = temps_guide[keep_base]
                    walkers_info_keep = walkers_info[keep_base]

                    special_inds_guide_keep = bands_guide_keep * int(1e8) + walkers_info_keep * int(1e4) + temps_guide_keep
                    sort_tmp = np.argsort(special_inds_guide_keep)
                    sorted_special_inds_guide_keep = special_inds_guide_keep[sort_tmp]
                    sorting_info_need_to_revert = np.searchsorted(sorted_special_inds_guide_keep, special_inds_bins, side="left")
                    
                    sorting_info = sort_tmp[sorting_info_need_to_revert]

                    assert np.all(special_inds_guide_keep[sorting_info] == special_inds_bins)

                    sort_bins = np.argsort(sorting_info)

                    # p1 = self.mgh.psd_shaped[0][0][8, 19748:20132]
                    # p2 = self.mgh.psd_shaped[1][0][8, 19748:20132]
                    # c2 =  self.mgh.data_shaped[1][0][26, 19748:20132] + self.mgh.data_shaped[1][0][8, 19748:20132]- self.mgh.channel2_base_data[0][8 * self.data_length + 19748:8 * self.data_length + 20132]
                    # c1 =  self.mgh.data_shaped[0][0][26, 19748:20132] + self.mgh.data_shaped[0][0][8, 19748:20132]- self.mgh.channel1_base_data[0][8 * self.data_length + 19748:8 * self.data_length + 20132]
                    # args_sorting = np.arange(len(sorting_info))  # np.argsort(sorting_info)
                    bands_in_tmp = bands_in_tmp[sort_bins]
                    temps_in_tmp = temps_in_tmp[sort_bins]
                    walkers_in_tmp = walkers_in_tmp[sort_bins]
                    coords_in_tmp = coords_in_tmp[sort_bins]
                    sorting_info = sorting_info[sort_bins]

                    uni_sorted, uni_index, uni_inverse, uni_counts = xp.unique(xp.asarray(sorting_info), return_index=True, return_counts=True, return_inverse=True)
                    
                    band_info_start_index_bin = xp.zeros_like(bands_guide_keep, dtype=np.int32)
                    band_info_num_bin = xp.zeros_like(bands_guide_keep, dtype=np.int32)

                    band_info_start_index_bin[uni_sorted.astype(np.int32)] = uni_index.astype(np.int32)
                    band_info_num_bin[uni_sorted.astype(np.int32)] = uni_counts.astype(np.int32)

                    params_curr_separated = tuple([xp.asarray(coords_in_tmp[:, i].copy()) for i in range(coords_in_tmp.shape[1])])
                    params_curr_separated_orig = tuple([xp.asarray(coords_in_tmp[:, i].copy()) for i in range(coords_in_tmp.shape[1])])
                    params_extra_params = (
                        self.waveform_kwargs["T"],
                        self.waveform_kwargs["dt"],
                        N_now,
                        coords_in.shape[0],
                        self.start_freq_ind, 
                        Soms_d_all["sangria"] ** (1/2),
                        Sa_a_all["sangria"] ** (1/2),
                        1e-100, 0.0, 0.0, 0.0, 0.0,  # foreground params for snr -> amp transform
                    )

                    gb_params_curr_in = self.gb.pyGalacticBinaryParams(
                        *(params_curr_separated + params_curr_separated_orig + params_extra_params)
                    )

                    # current_parameter_arrays.append(params_curr_separated)

                    data_package = self.gb.pyDataPackage(
                        data[0][0],
                        data[1][0],
                        base_data[0][0],
                        base_data[1][0],
                        psd[0][0],
                        psd[1][0],
                        lisasens[0][0],
                        lisasens[1][0],
                        self.df,
                        self.data_length,
                        self.nwalkers * self.ntemps,
                        self.nwalkers * self.ntemps
                    )

                    data_index_here = xp.asarray((((bands_guide_keep + 1) % 2) * self.nwalkers + walkers_info_keep)).astype(np.int32)
                    noise_index_here = xp.asarray(walkers_info_keep.copy()).astype(np.int32)
                    update_data_index_here = xp.asarray((((bands_guide_keep + 0) % 2) * self.nwalkers + walkers_info_keep)).astype(np.int32)
                    
                    start_band_index = ((self.band_edges[bands_guide_keep] / self.df).astype(int) - N_now - 1).astype(np.int32)
                    end_band_index = (((self.band_edges[bands_guide_keep + 1]) / self.df).astype(int) + N_now + 1).astype(np.int32)
                    
                    start_band_index[start_band_index < 0] = 0
                    end_band_index[end_band_index >= len(self.fd)] = len(self.fd) - 1
                    
                    band_lengths = end_band_index - start_band_index

                    start_interest_band_index = ((self.band_edges[bands_guide_keep] / self.df).astype(int)).astype(np.int32)  #  - N_now).astype(np.int32)
                    end_interest_band_index = (((self.band_edges[bands_guide_keep + 1]) / self.df).astype(int)).astype(np.int32)  #  + N_now).astype(np.int32)
                    
                    start_interest_band_index[start_interest_band_index < 0] = 0
                    end_interest_band_index[end_interest_band_index >= len(self.fd)] = len(self.fd) - 1
                    
                    band_interest_lengths = end_interest_band_index - start_interest_band_index


                    # proposal limits in a band
                    fmin_allow = xp.asarray(((self.band_edges[bands_guide_keep] / self.df).astype(int) - (N_now / 2))) * self.df
                    fmax_allow = xp.asarray(((self.band_edges[bands_guide_keep + 1] / self.df).astype(int) + (N_now / 2))) * self.df

                    fmin_allow[fmin_allow < self.band_edges[0]] = self.band_edges[0]
                    fmax_allow[fmax_allow > self.band_edges[-1]] = self.band_edges[-1]

                    max_data_store_size = band_lengths.max().item()

                    num_bands_here = len(band_info_start_index_bin)

                    num_swap_setups = np.unique(bands_guide_keep).shape[0] * self.nwalkers

                    swaps_proposed_here = xp.zeros(num_swap_setups * (self.ntemps - 1), dtype=np.int32)
                    swaps_accepted_here = xp.zeros(num_swap_setups * (self.ntemps - 1), dtype=np.int32)
                    
                    bands_guide_keep = xp.asarray(bands_guide_keep).astype(np.int32)
                    walkers_info_keep = xp.asarray(walkers_info_keep).astype(np.int32)
                    temps_guide_keep = xp.asarray(temps_guide_keep).astype(np.int32)
                    loc_band_index = xp.arange(num_bands_here, dtype=xp.int32)
                    
                    before_loc_band_index = loc_band_index.copy()
                    before_band_info_start_index_bin = band_info_start_index_bin.copy()
                    before_band_info_num_bin = band_info_num_bin.copy()

                    band_package = self.gb.pyBandPackage(
                        loc_band_index,
                        data_index_here,
                        noise_index_here,
                        band_info_start_index_bin,  # uni_index
                        band_info_num_bin,  # uni_count
                        start_band_index,
                        band_lengths,
                        start_interest_band_index,
                        band_interest_lengths,
                        num_bands_here,
                        max_data_store_size,
                        fmin_allow,
                        fmax_allow,
                        update_data_index_here,
                        self.ntemps,
                        bands_guide_keep, 
                        walkers_info_keep,
                        temps_guide_keep,
                        swaps_proposed_here,
                        swaps_accepted_here
                    )

                    band_inv_temp_vals_here = band_temps[bands_guide_keep, temps_guide_keep]

                    accepted_out_here = xp.zeros_like(bands_guide_keep, dtype=xp.int32)
                    L_contribution_here = xp.zeros_like(bands_guide_keep, dtype=complex)
                    p_contribution_here = xp.zeros_like(bands_guide_keep, dtype=complex)
                    prior_all_curr_here = xp.zeros_like(bands_guide_keep, dtype=np.float64)
                    
                    mcmc_info = self.gb.pyMCMCInfo(
                        L_contribution_here,
                        p_contribution_here,
                        prior_all_curr_here,
                        accepted_out_here,
                        band_inv_temp_vals_here,  # band_inv_temp_vals
                        self.is_rj_prop,
                        False,  # phased maximize
                        self.snr_lim,
                    )

                    inds_here = xp.ones_like(temps_in_tmp, dtype=bool)
                    factors_here = xp.zeros_like(temps_in_tmp, dtype=float)
                    start_inds_here = xp.zeros_like(temps_in_tmp, dtype=np.int32)
                    
                    num_proposals_here = 1
                    
                    assert start_inds_here.dtype == np.int32
                    proposal_info = self.gb.pyStretchProposalPackage(
                        *(self.stretch_friends_args_in + (self.nfriends, len(self.stretch_friends_args_in[0]), num_proposals_here, self.a, ndim, inds_here, factors_here, start_inds_here))
                    )

                    inputs_now = (
                        data_package,
                        band_package,
                        gb_params_curr_in,
                        mcmc_info,
                        self.gpu_cuda_priors,
                        proposal_info,
                        self.gpu_cuda_wrap,
                        num_swap_setups,
                        device,
                        do_synchronize,
                        -1, 
                        100000
                    )

                    self.gb.SharedMemoryMakeTemperingMove_wrap(*inputs_now)

                    self.xp.cuda.runtime.deviceSynchronize()

                    walkers_info_keep_per_bin = xp.repeat(walkers_info_keep, list(band_info_num_bin.get()))
                    temps_guide_keep_per_bin = xp.repeat(temps_guide_keep, list(band_info_num_bin.get()))
                    start_bin_ind_mapping = xp.repeat(band_info_start_index_bin, list(band_info_num_bin.get()))
                    uni_after, uni_after_start_index, uni_after_inverse = xp.unique(start_bin_ind_mapping, return_index=True, return_inverse=True)
                    ind_map = xp.arange(len(start_bin_ind_mapping)) - xp.arange(len(start_bin_ind_mapping))[uni_after_start_index][uni_after_inverse] + start_bin_ind_mapping
                    
                    # out = []
                    # for i in range(len(band_info_num_bin_has)):
                    #     num_bins = band_info_num_bin_has[i].item()
                    #     start_index = band_info_start_index_bin_has[i].item()
                    #     for j in range(num_bins):
                    #         out.append(start_index + j)

                    # breakpoint()
                    new_coords_mapped = coords_in_tmp[ind_map.get()]

                    special_after_tempering = temps_guide_keep_per_bin * self.nwalkers + walkers_info_keep_per_bin
                    
                    sorted_after = xp.argsort(special_after_tempering)
                    special_after_tempering = special_after_tempering[sorted_after]
                    walkers_info_keep_per_bin = walkers_info_keep_per_bin[sorted_after]
                    temps_guide_keep_per_bin = temps_guide_keep_per_bin[sorted_after]
                    new_coords_mapped = new_coords_mapped[sorted_after.get()]

                    uni_after, uni_index_after, uni_inverse_after = xp.unique(special_after_tempering, return_index=True, return_inverse=True)

                    relative_leaf_info_after_tempering = xp.arange(len(special_after_tempering)) - uni_index_after[uni_inverse_after]

                    leaf_start_per_walker = new_inds_after_tempering.argmin(axis=-1)[temps_guide_keep_per_bin[uni_index_after], walkers_info_keep_per_bin[uni_index_after]]
                    
                    absolute_leaf_info_after_tempering = leaf_start_per_walker[uni_inverse_after] + relative_leaf_info_after_tempering

                    assert (np.all(~new_inds_after_tempering[temps_guide_keep_per_bin, walkers_info_keep_per_bin, absolute_leaf_info_after_tempering]))
                    new_coords_after_tempering[temps_guide_keep_per_bin, walkers_info_keep_per_bin, absolute_leaf_info_after_tempering] = new_coords_mapped
                    new_inds_after_tempering[temps_guide_keep_per_bin, walkers_info_keep_per_bin, absolute_leaf_info_after_tempering] = True

                    bands_unique_here = np.unique(bands_guide_keep).get()
                    band_swaps_accepted[bands_unique_here] += swaps_accepted_here.reshape(bands_unique_here.shape[0], self.nwalkers, self.ntemps - 1).sum(axis=1).get()
                    band_swaps_proposed[bands_unique_here] += swaps_proposed_here.reshape(bands_unique_here.shape[0], self.nwalkers, self.ntemps - 1).sum(axis=1).get()

                    ll_change = self.xp.zeros((ntemps, nwalkers, len(self.band_edges)))

                    self.xp.cuda.runtime.deviceSynchronize()

                    ll_change[(temps_guide_keep,walkers_info_keep,bands_guide_keep)] = L_contribution_here
                    
                    ll_adjustment = ll_change.sum(axis=-1)
                    log_like_tmp += ll_adjustment
                    
            new_state.branches["gb"].coords[:] = new_coords_after_tempering.get()
            new_state.branches["gb"].inds[:] = new_inds_after_tempering.get()
            new_state.log_like[:] = log_like_tmp.get()

            new_freqs = new_coords_after_tempering[new_inds_after_tempering, 1]
            new_band_inds = (xp.searchsorted(self.band_edges, new_freqs / 1e3, side="right") - 1)
            new_state.branches["gb"].branch_supplimental.holder["band_inds"][new_inds_after_tempering.get()] = new_band_inds.get()

            # breakpoint()
            # self.check_ll_inject(new_state)
            # breakpoint()
            
            # adjust priors accordingly
            log_prior_new_per_bin = xp.zeros_like(
                new_state.branches_inds["gb"], dtype=xp.float64
            )
            # self.gpu_priors
            log_prior_new_per_bin[
                new_state.branches_inds["gb"]
            ] = self.gpu_priors["gb"].logpdf(
                xp.asarray(
                    new_state.branches_coords["gb"][
                        new_state.branches_inds["gb"]
                    ]
                ),
                psds=self.mgh.psd_shaped[0][0],
                walker_inds=group_temp_finder[1].get()[new_state.branches_inds["gb"]]
            )

            new_state.log_prior = log_prior_new_per_bin.sum(axis=-1).get()
            
            ratios = (band_swaps_accepted / band_swaps_proposed).T #  self.swaps_accepted / self.swaps_proposed
            ratios[np.isnan(ratios)] = 0.0

            # only change those with a binary in them

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
            )
            # print(np.abs(new_state.log_like - ll_after).max())
            store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
            # print("CHECKING:", store_max_diff, self.is_rj_prop)
            if store_max_diff > 1e-5:
                ll_after = (
                    self.mgh.get_ll(include_psd_info=True, stop=True)
                )

                if store_max_diff > 1.0:
                    breakpoint()

                # reset data and fix likelihood
                new_state.log_like[0] = self.check_ll_inject(new_state)
            

        self.time += 1
        # self.xp.cuda.runtime.deviceSynchronize()

        new_state.update_band_information(
            band_temps.get(), per_walker_band_proposals.sum(axis=1).get().T, per_walker_band_accepted.sum(axis=1).get().T, band_swaps_proposed, band_swaps_accepted,
            per_walker_band_counts.get(), self.is_rj_prop
        )
        # TODO: check rj numbers

        # new_state.log_like[:] = self.check_ll_inject(new_state)

        self.mempool.free_all_blocks()

        if self.is_rj_prop:
            pass  # print(self.name, "2nd count check:", new_state.branches["gb"].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])
        return new_state, accepted

    def check_ll_inject(self, new_state):

        check_ll = self.mgh.get_ll(include_psd_info=True).copy()

        nleaves_max = new_state.branches["gb"].shape[-2]
        for i in range(2):
            self.mgh.channel1_data[0][self.nwalkers * self.data_length * i: self.nwalkers * self.data_length * (i + 1)] = self.mgh.channel1_base_data[0][:]
            self.mgh.channel2_data[0][self.nwalkers * self.data_length * i: self.nwalkers * self.data_length * (i + 1)] = self.mgh.channel2_base_data[0][:]
        
        coords_out_gb = new_state.branches["gb"].coords[0, new_state.branches["gb"].inds[0]]
        coords_in_in = self.parameter_transforms.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(self.band_edges.get(), coords_in_in[:, 1], side="right") - 1
        assert np.all(band_inds == new_state.branches["gb"].branch_supplimental.holder["band_inds"][0, new_state.branches["gb"].inds[0]])

        walker_vals = np.tile(np.arange(self.nwalkers), (nleaves_max, 1)).transpose((1, 0))[new_state.branches["gb"].inds[0]]

        data_index_1 = ((band_inds % 2) + 0) * self.nwalkers + walker_vals

        data_index = xp.asarray(data_index_1).astype(xp.int32)

        # goes in as -h
        factors = -xp.ones_like(data_index, dtype=xp.float64)

        waveform_kwargs_tmp = self.waveform_kwargs.copy()

        N_vals = self.band_N_vals[band_inds]
        self.gb.generate_global_template(
            coords_in_in,
            data_index,
            self.mgh.data_list,
            batch_size=1000,
            data_length=self.data_length,
            factors=factors,
            N=N_vals,
            data_splits=self.mgh.gpu_splits,
            **waveform_kwargs_tmp,
        )

        check_ll_new = self.mgh.get_ll(include_psd_info=True)
        check_ll_diff1 = check_ll_new - check_ll
        # print(check_ll_diff1)

        # breakpoint()
        return check_ll_new

        # breakpoint()

        # # print(self.accepted / self.num_proposals)

        # # MULTIPLE TRY after RJ

        # if False: # self.is_rj_prop:
            
        #     inds_added = np.where(new_state.branches["gb"].inds.astype(int) - state.branches["gb"].inds.astype(int) == +1)

        #     new_coords = xp.asarray(new_state.branches["gb"].coords[inds_added])
        #     temp_inds_add = xp.repeat(xp.arange(ntemps)[:, None], nwalkers * nleaves_max, axis=-1).reshape(ntemps, nwalkers, nleaves_max)[inds_added]
        #     walker_inds_add = xp.repeat(xp.arange(nwalkers)[:, None], ntemps * nleaves_max, axis=-1).reshape(nwalkers, ntemps, nleaves_max).transpose(1, 0, 2)[inds_added]
        #     leaf_inds_add = xp.repeat(xp.arange(nleaves_max)[:, None], ntemps * nwalkers, axis=-1).reshape(nleaves_max, ntemps, nwalkers).transpose(1, 2, 0)[inds_added]
        #     band_inds_add = xp.searchsorted(self.band_edges, new_coords[:, 1] / 1e3, side="right") - 1
        #     N_vals_add = xp.asarray(new_state.branches["gb"].branch_supplimental.holder["N_vals"][inds_added])
            
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
        #             coords_add[:, inds_params] = self.gpu_priors["gb"].rvs(size=coords_remove.shape[0])[:, inds_params]
                    
        #             log_proposal_pdf = self.gpu_priors["gb"].logpdf(coords_add)
        #             # remove logpdf from fchange for now
        #             log_proposal_pdf -= self.gpu_priors["gb"].priors_in[1].logpdf(coords_add[:, 1])

        #             priors_remove = self.gpu_priors["gb"].logpdf(coords_remove)
        #             priors_add = self.gpu_priors["gb"].logpdf(coords_add)

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

        #                 initial_log_proposal_pdf = self.gpu_priors["gb"].logpdf(new_coords_group_split)
        #                 # remove logpdf from fchange for now
        #                 initial_log_proposal_pdf -= self.gpu_priors["gb"].priors_in[1].logpdf(new_coords_group_split[:, 1])

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

        #             new_state.branches["gb"].coords[temp_inds_add_keep.get(), walker_inds_add_keep.get(), band_inds_add_keep.get()] = coords_add_keep.get()
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
                #     if name not in ["gb", "gb"]:
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


