# -*- coding: utf-8 -*-

from copy import deepcopy
from inspect import Attribute
import numpy as np
from scipy import stats
import warnings
import time
from gbgpu.utils.utility import get_N
from ...detector import sangria
from .globalfitmove import GlobalFitMove
from ..galaxyglobal import run_gb_bulk_search, fit_each_leaf, make_gmm
from eryn.state import BranchSupplemental
from typing import Optional      

try:
    import cupy as cp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as cp

    gpu_available = False

from lisatools.utils.utility import searchsorted2d_vec, get_groups_from_band_structure
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer
from eryn.utils.utility import groups_from_inds
from eryn.utils import PeriodicContainer

from eryn.moves import GroupStretchMove, Move
from eryn.moves.multipletry import logsumexp, get_mt_computations

from ...diagnostic import inner_product
from lisatools.globalfit.state import GFState
from lisatools.sampling.prior import GBPriorWrap


__all__ = ["GBSpecialStretchMove"]

def gb_search_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    assert comm is not None

    # get current rank and get index into class_ranks_list
    print(f"INSIDE GB search, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    gather_rank = class_ranks_list[0]
    if rank_index == 0:
        split_remainder = 1  # will fix this setup in the future
        num_search = 2
        gpu = class_extra_gpus[0]
        comm_info = {"process_ranks_for_fit": class_ranks_list[1:]}
        # run search here
        run_gb_bulk_search(gpu, curr, comm, comm_info, main_rank, num_search, split_remainder)
        pass

    else:
        # run GMM fit here
        fit_each_leaf(rank, curr, gather_rank, comm)
        pass


def gb_refit_func(comm, curr, main_rank, class_extra_gpus, class_ranks_list):
    assert comm is not None

    # get current rank and get index into class_ranks_list
    print(f"INSIDE GB refit, RANK: {comm.Get_rank()}")
    rank = comm.Get_rank()
    rank_index = class_ranks_list.index(rank)
    gather_rank = class_ranks_list[0]
    if rank_index == 0:
        split_remainder = 0  # will fix this setup in the future
        num_search = 2
        gpu = class_extra_gpus[0]
        comm_info = {"process_ranks_for_fit": class_ranks_list[1:]}
        # run search here
        # run_gb_bulk_search(gpu, curr, comm, comm_info, main_rank, num_search, split_remainder)
        pass

    else:
        # run GMM fit here
        # fit_each_leaf(rank, curr, gather_rank, comm)
        pass

from eryn.state import Branch
from dataclasses import dataclass

    
from eryn.utils import TransformContainer

class Buffer:
    pass


def return_x(x):
    return x

class BandSorter:

    @property
    def xp(self) -> object:
        if self.use_gpu:
            return cp
        else:
            return np

    def __init__(
        self, 
        gb_branch: Branch, 
        band_edges: Optional[np.ndarray] = None, 
        band_N_vals: Optional[np.ndarray] = None, 
        use_gpu: bool=False, 
        transform_fn: Optional[TransformContainer] = None, 
        copy:bool = True,
        inds_subset: Optional[np.ndarray] = None,
        inds_main_band_sorter: Optional[np.ndarray] = None,
        main_band_sorter = None,
    ):
        
        dc = deepcopy if copy else return_x
        if hasattr(gb_branch, "num_sources"):
            _band_sorter = gb_branch
            self.use_gpu = _band_sorter.use_gpu
            for key, value in _band_sorter.__dict__.items():
                if key[:2] != "__":
                    if key in ["main_band_sorter"]:
                        set_value = value
                         
                    elif isinstance(value, self.xp.ndarray) and value.shape[0] == _band_sorter.num_sources:
                        if inds_subset is None:
                            inds_subset = self.xp.arange(_band_sorter.num_sources)
                        else:
                            assert isinstance(inds_subset, self.xp.ndarray) and inds_subset.dtype == int
                            assert inds_subset.max() < (self.num_sources)
                        set_value = dc(value[inds_subset])

                    else:
                        set_value = dc(value)

                    setattr(self, key, set_value)
            return

        assert band_edges is not None and band_N_vals is not None
        self.use_gpu = use_gpu
        self.gb_branch_orig = gb_branch
        self.num_bands = len(band_edges) - 1
        self.band_edges = band_edges
        self.band_N_vals = band_N_vals
        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = gb_branch.shape
        self.orig_inds = self.xp.asarray(gb_branch.inds)
        self.num_sources = self.orig_inds.sum().item()
        self.main_band_sorter = main_band_sorter
        if self.main_band_sorter is None:
            self.inds_main_band_sorter = self.xp.arange(self.num_sources)
        else:
            self.inds_main_band_sorter = inds_main_band_sorter

        self.coords = self.xp.asarray(gb_branch.coords[gb_branch.inds])
        self.freqs = self.coords[:, 1] / 1e3
        self.band_inds = self.xp.searchsorted(band_edges, self.freqs, side="right") - 1
        
        self.temp_inds = self.xp.repeat(self.xp.arange(self.ntemps), self.nwalkers * self.nleaves_max).reshape(self.ntemps, self.nwalkers, self.nleaves_max)[self.orig_inds]
        self.walker_inds = self.xp.tile(self.xp.arange(self.nwalkers), (self.ntemps, self.nleaves_max, 1)).transpose((0, 2, 1))[self.orig_inds]
        self.leaf_inds = self.xp.tile(self.xp.arange(self.nleaves_max), ((self.ntemps, self.nwalkers, 1)))[self.orig_inds]
        self.special_band_inds = self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds)
        
        self.transform_fn = transform_fn

    @property
    def coords_in(self) -> np.ndarray:
        return self.transform_fn.both_transforms(self.coords, xp=self.xp)

    def get_special_band_index(self, temp_inds: np.ndarray, walker_inds: np.ndarray, band_inds: np.ndarray) -> np.ndarray:
        special_indices = (temp_inds * self.nwalkers + walker_inds) * int(1e6) + band_inds
        return special_indices

    @property
    def special_index_check(self) -> bool:
        return self.xp.all(self.special_band_inds == self.get_special_band_index(self.temp_inds, self.walker_inds, self.band_inds))

    @property
    def N_vals(self) -> np.ndarray:
        return self.band_N_vals[self.band_inds]
        
    @property
    def unique_N(self) -> np.ndarray:
        return self.xp.unique(self.N_vals)

    def get_subset(
        self,
        *args, 
        **kwargs
    ):
        subset_inds = self.get_subset_inds(*args, **kwargs)

        main_band_sorter = self if self.main_band_sorter is None else self.main_band_sorter
        
        # source information
        subset = BandSorter(
            self, inds_subset=subset_inds, main_band_sorter=main_band_sorter
        )
        # band information
        return subset

    def get_subset_inds(self, *args, **kwargs):
        subset_bool = self.get_subset_bool(*args, **kwargs)
        return self.xp.arange(len(subset_bool))[subset_bool]

    def get_subset_bool(
        self, 
        units: Optional[int] = None,
        remainder: Optional[int] = None,
        temp: Optional[int] = None,
        walker: Optional[int] = None,
        leaf: Optional[int] = None,
        band: Optional[int] = None,
        extra_bool: Optional[np.ndarray] = None,
        full_bool:Optional[np.ndarray] = None
    ) -> np.ndarray:

        inds_keep = self.xp.ones_like(self.band_inds, dtype=bool)

        if full_bool is None:
            if band is not None:
                assert isinstance(band, int)
                inds_keep &= (self.band_inds == band)
            elif units is not None or remainder is not None:
                assert units is not None and remainder is not None
                inds_keep &= (self.band_inds % units == remainder)
            
            inds_keep &= (self.band_inds < len(self.band_edges) - 2)
            inds_keep &= (self.band_inds > 1)

            if temp is not None:
                assert isinstance(temp, int)
                inds_keep &= (self.temp_inds == temp)
            if walker is not None:
                assert isinstance(walker, int)
                inds_keep &= (self.walker_inds == walker)
            if leaf is not None:
                assert isinstance(temp, int)
                inds_keep &= (self.leaf_inds == leaf)

            if extra_bool is not None:
                assert isinstance(extra_bool, self.xp.ndarray)
                assert extra_bool.shape == (self.num_sources,)
                inds_keep &= extra_bool
        else:
            assert full_bool.shape[0] == self.num_sources
            inds_keep = full_bool
            
        return inds_keep

    def get_buffer(self, sources_of_interest: np.ndarray, num_band_preload : int = 1000) -> Buffer:

        special_indices_unique, special_indices_index = cp.unique(self.special_band_inds[sources_of_interest], return_index=True)
        if num_band_preload is None:
            num_band_preload = len(special_indices_unique)
        
        special_indices_unique, special_indices_index = (special_indices_unique[:num_band_preload], special_indices_index[:num_band_preload])
                
        # CAN USE self.main_band_sorter TO GET SOURCES IN BANDS OF INTEREST THAT ARE NOT CURRENTLY OF INTEREST THEMSELVES
        breakpoint()




# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBSpecialBase(GlobalFitMove, GroupStretchMove, Move):
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
        ranks_needed=0,
        gpus=[],
        psd_like=None,
        **kwargs
    ):
        # return_gpu is a kwarg for the stretch move
        GroupStretchMove.__init__(self, *args, return_gpu=True, **kwargs)
        self.psd_like = psd_like
        assert psd_like is not None
        self.ranks_needed = ranks_needed
        self.gpus = gpus
        self.gpu_priors = gpu_priors
        self.name = name
        self.num_repeat_proposals = num_repeat_proposals

        self.use_prior_removal = use_prior_removal

        # for key in priors:
        #     if not isinstance(priors[key], ProbDistContainer) and not isinstance(priors[key], GBPriorWrap):
        #         raise ValueError(
        #             "Priors need to be eryn.priors.ProbDistContainer object."
        #         )
        
        self.priors = priors
        self.gb = gb
        self.stop_here = True

        # args = [priors["gb"].priors_in[(0, 1)].rho_star]
        # args += [priors["gb"].priors_in[(0, 1)].frequency_prior.min_val, priors["gb"].priors_in[(0, 1)].frequency_prior.max_val]
        # for i in range(2, 8):
        #     args += [priors["gb"].priors_in[i].min_val, priors["gb"].priors_in[i].max_val]
        
        # self.gpu_cuda_priors = self.gb.pyPriorPackage(*tuple(args))
        # self.gpu_cuda_wrap = self.gb.pyPeriodicPackage(2 * np.pi, np.pi, 2 * np.pi)

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
        if self.is_rj_prop:
            if (self.num_repeat_proposals != 1):
                print("Adjusting repeat proposals to 1 for RJ.")

            self.num_repeat_proposals = 1

        # setup N vals for bands
        band_mean_f = (self.band_edges[1:] + self.band_edges[:-1]).get() / 2
        self.band_N_vals = cp.asarray(get_N(np.full_like(band_mean_f, 1e-30), band_mean_f, self.waveform_kwargs["T"], self.waveform_kwargs["oversample"]))

    def setup_gbs(self, branch):
        st = time.perf_counter()
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplemental
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
        still_going = cp.ones(total_binaries, dtype=bool) 
        inds_zero = cp.searchsorted(self.freqs_sorted, all_temp_fs, side="right") - 1
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

            change_inds = cp.arange(len(all_temp_fs))[still_going][check_move_right][(new_distance_right < distance_left[check_move_right]) & (check_right_inds < len(self.freqs_sorted))]

            left_inds[change_inds] += 1
            right_inds[change_inds] += 1

            stop_inds_right_1 = cp.arange(len(all_temp_fs))[still_going][check_move_right][(check_right_inds >= len(self.freqs_sorted))]

            # last part is just for up here, below it will remove if it is still equal
            stop_inds_right_2 = cp.arange(len(all_temp_fs))[still_going][check_move_right][(new_distance_right >= distance_left[check_move_right]) & (check_right_inds < len(self.freqs_sorted)) & (distance_right[check_move_right] != distance_left[check_move_right])]
            stop_inds_right = cp.concatenate([stop_inds_right_1, stop_inds_right_2])
            assert np.all(still_going[stop_inds_right])

            # equal to should only be left over if it was equal above and moving right did not help
            check_move_left = (distance_left <= distance_right)
            check_left_inds = left_inds[still_going][check_move_left] - 1
            check_right_inds = right_inds[still_going][check_move_left] - 1

            new_distance_left = np.abs(all_temp_fs[still_going][check_move_left] - self.freqs_sorted[check_left_inds])
            
            change_inds = cp.arange(len(all_temp_fs))[still_going][check_move_left][(new_distance_left < distance_right[check_move_left]) & (check_left_inds >= 0)]

            left_inds[change_inds] -= 1
            right_inds[change_inds] -= 1

            stop_inds_left_1 = cp.arange(len(all_temp_fs))[still_going][check_move_left][(check_left_inds < 0)]
            stop_inds_left_2 = cp.arange(len(all_temp_fs))[still_going][check_move_left][(new_distance_left >= distance_right[check_move_left]) & (check_left_inds >= 0)]
            stop_inds_left = cp.concatenate([stop_inds_left_1, stop_inds_left_2])
            
            stop_inds = cp.concatenate([stop_inds_right, stop_inds_left])
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

    def find_friends(self, name, gb_points_to_move, s_inds=None, branch_supps=None):
        if s_inds is None or branch_supps is None:
            raise ValueError

        inds_points_to_move = self.xp.asarray(s_inds.flatten())

        half_friends = int(self.nfriends / 2)

        gb_points_for_move = gb_points_to_move.reshape(-1, 8).copy()

        if not hasattr(self, "ntemps"):
            self.ntemps = 1

        inds_start_freq_to_move = self.xp.asarray(branch_supps[:]["friend_start_inds"].flatten())

        deviation = self.xp.random.randint(0, self.nfriends, size=len(inds_start_freq_to_move))

        inds_keep_friends = inds_start_freq_to_move + deviation

        inds_keep_friends[inds_keep_friends < 0] = 0
        inds_keep_friends[inds_keep_friends >= len(self.all_coords_sorted)] = (len(self.all_coords_sorted) - 1)

        gb_points_for_move[inds_points_to_move] = self.all_coords_sorted[inds_keep_friends]
        return gb_points_for_move[None, :, None, :]

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

            if self.time % self.n_iter_update == 0:  # not self.is_rj_prop and 
                self.setup_gbs(branch)

            elif self.is_rj_prop and self.time == 0:
                ndim = branch.shape[-1]
                self.stretch_friends_args_in = tuple([cp.array([]) for _ in range(ndim)])

            # update any shifted start inds due to tempering (need to do this every non-rj move)
            """if not self.is_rj_prop:
                # fix the ones that have been added in RJ
                fix = (
                    branch.branch_supplemental.holder["friend_start_inds"][:] == -1
                ) & branch.inds

                if np.any(fix):
                    new_freqs = cp.asarray(branch.coords[fix][:, 1])
                    # TODO: is there a better way of doing this?

                    # fill information into friend finder for new binaries
                    branch.branch_supplemental.holder["friend_start_inds"][fix] = (
                        (
                            cp.searchsorted(self.freqs_sorted, new_freqs, side="right")
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
                    branch.branch_supplemental.holder["friend_start_inds"][:]
                )
            """

            self.mempool.free_all_blocks()

    def adjust_sources_in_residual_buffer(self, factor, model, band_sorter: BandSorter, *args, **kwargs) -> None:
        
        assert isinstance(factor, int) and (factor == -1 or factor == +1)

        subset = band_sorter.get_subset(*args, **kwargs)

        factors_tmp = factor * cp.ones_like(subset.walker_inds, dtype=float)
        self.gb.generate_global_template(
            subset.coords_in,
            subset.walker_inds.astype(self.xp.int32),
            model.analysis_container_arr.linear_data_arr,
            data_length=model.analysis_container_arr.data_length,
            factors=factors_tmp,
            data_splits=model.analysis_container_arr.gpu_map,
            N=subset.N_vals,
            **self.waveform_kwargs,
        )

    def remove_cold_chain_sources_from_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        self.remove_sources_from_residual(*args, **kwargs)

    def remove_sources_from_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(+1, *args, **kwargs)
       
    def add_cold_chain_sources_to_residual(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        self.add_sources_to_residual(*args, **kwargs)

    def add_sources_to_residual(self, *args, **kwargs) -> None:
        self.adjust_sources_in_residual_buffer(-1, *args, **kwargs)


    def adjust_sources_in_temporary_buffer(self, factor, model, band_sorter: BandSorter, *args, **kwargs) -> None:
        
        assert isinstance(factor, int) and (factor == -1 or factor == +1)
        breakpoint()
        subset = band_sorter.get_subset(*args, **kwargs)

        factors_tmp = factor * cp.ones_like(subset.walker_inds, dtype=float)
        
        self.gb.generate_global_template(
            subset.coords_in,
            subset.walker_inds.astype(self.xp.int32),
            model.analysis_container_arr.linear_data_arr,
            data_length=model.analysis_container_arr.data_length,
            factors=factors_tmp,
            data_splits=model.analysis_container_arr.gpu_map,
            N=subset.N_vals,
            **self.waveform_kwargs,
        )

    def remove_cold_chain_sources_from_temporary_buffer(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        self.remove_sources_from_temporary_buffer(*args, **kwargs)

    def remove_sources_from_temporary_buffer(self, *args, **kwargs) -> None:
        self.adjust_sources_in_temporary_buffer(+1, *args, **kwargs)
       
    def add_cold_chain_sources_to_temporary_buffer(self, *args, **kwargs) -> None:
        kwargs["temp"] = 0
        self.add_sources_to_temporary_buffer(*args, **kwargs)

    def add_sources_to_temporary_buffer(self, *args, **kwargs) -> None:
        self.adjust_sources_in_temporary_buffer(-1, *args, **kwargs)

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`GFState`): Current state of the sampler.

        Returns:
            :class:`GFState`: GFState of sampler after proposal is complete.

        """
        st_all = time.perf_counter()

        self.xp.cuda.runtime.setDevice(model.analysis_container_arr.gpus[0])
        nchannels = model.analysis_container_arr.nchannels
        data_length = model.analysis_container_arr.data_length
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

        new_state = GFState(state, copy=True)
        band_temps = cp.asarray(state.sub_states["gb"].band_info["band_temps"].copy())

        if self.is_rj_prop:
            orig_store = new_state.log_like[0].copy()
        gb_coords = cp.asarray(new_state.branches["gb"].coords)

        self.mempool.free_all_blocks()

        # self.mgh.map = new_state.supplemental.holder["overall_inds"].flatten()

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

        data = model.analysis_container_arr.linear_data_arr
        # base_data = self.mgh.base_data_list
        psd = model.analysis_container_arr.linear_psd_arr
        # lisasens = self.mgh.lisasens_list

        # do unique for band size as separator between asynchronous kernel launches
        # band_indices = self.xp.asarray(new_state.branches["gb"].branch_supplemental.holder["band_inds"])
        band_indices = self.xp.searchsorted(self.band_edges, cp.asarray(new_state.branches["gb"].coords[:, :, :, 1]).flatten() / 1e3, side="right").reshape(new_state.branches["gb"].coords[:, :, :, 1].shape) - 1
            
        # N_vals_in = self.xp.asarray(new_state.branches["gb"].branch_supplemental.holder["N_vals"])
        points_curr = self.xp.asarray(new_state.branches["gb"].coords)
        points_curr_orig = points_curr.copy()
        # N_vals_in_orig = N_vals_in.copy()
        band_indices_orig = band_indices.copy()
        
        if self.is_rj_prop:
            if isinstance(self.rj_proposal_distribution["gb"], list):
                raise NotImplementedError
                assert len(self.rj_proposal_distribution["gb"]) == ntemps
                proposal_logpdf = cp.zeros((points_curr.shape[0], np.prod(points_curr.shape[1:-1])))
                for t in range(ntemps):
                    new_sources = cp.full_like(points_curr[t, ~gb_inds[t]], np.nan)
                    fix = cp.full(new_sources.shape[0], True)
                    while cp.any(fix):
                        new_sources[fix] = self.rj_proposal_distribution["gb"][t].rvs(size=fix.sum().item())
                        fix = cp.any(cp.isnan(new_sources), axis=-1)
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
                new_sources = cp.full_like(points_curr[~gb_inds], np.nan)
                fix = cp.full(new_sources.shape[0], True)
                while cp.any(fix):
                    if self.name == "rj_prior":
                        new_sources[fix] = self.rj_proposal_distribution["gb"].rvs(size=fix.sum().item())
                    else:
                        new_sources[fix] = self.rj_proposal_distribution["gb"].rvs(size=fix.sum().item())
                    
                    fix = cp.any(cp.isnan(new_sources), axis=-1)
                points_curr[~gb_inds] = new_sources

                band_indices[~gb_inds] = self.xp.searchsorted(self.band_edges, new_sources[:, 1] / 1e3, side="right") - 1
                # N_vals_in[~gb_inds] = self.xp.asarray(
                #     get_N(
                #         cp.full_like(new_sources[:, 1], 1e-30),
                #         new_sources[:, 1] / 1e3,
                #         self.waveform_kwargs["T"],
                #         self.waveform_kwargs["oversample"],
                #         cp=self.xp
                #     )
                # )

                # change gb_inds
                gb_inds[:] = True
                assert np.all(gb_inds)
                if self.name == "rj_prior":
                    proposal_logpdf = self.rj_proposal_distribution["gb"].logpdf(
                        points_curr[gb_inds]
                    )
                else:
                    tmp_points = points_curr[gb_inds]
                    proposal_logpdf = cp.zeros(tmp_points.shape[0])

                    batch_here = int(1e6)
                    inds_splitting = np.arange(0, tmp_points.shape[0], batch_here)
                    if inds_splitting[-1] != tmp_points.shape[0] - 1:
                        inds_splitting = np.concatenate([inds_splitting, np.array([tmp_points.shape[0] - 1])])
                    
                    for stind, eind in zip(inds_splitting[:-1], inds_splitting[1:]):
                        proposal_logpdf[stind: eind] = self.rj_proposal_distribution["gb"].logpdf(tmp_points[stind: eind])
                    self.mempool.free_all_blocks()

            factors = (cp.asarray(proposal_logpdf) * -1) * (~gb_inds_orig).flatten() + (cp.asarray(proposal_logpdf) * +1) * (gb_inds_orig).flatten()

            if self.name == "rj_prior" and self.use_prior_removal:
                factors[~gb_inds_orig.flatten()] = -1e300
        else:
            factors = cp.zeros(gb_inds_orig.sum().item())

        band_sorter = BandSorter(new_state.branches["gb"], self.band_edges, self.band_N_vals, use_gpu=self.use_gpu, transform_fn=self.parameter_transforms)
        
        start_inds_all = cp.asarray(new_state.branches["gb"].branch_supplemental.holder["friend_start_inds"], dtype=cp.int32)[gb_inds]
        points_curr = points_curr[gb_inds]
        # N_vals_in = N_vals_in[gb_inds]
        band_indices = band_indices[gb_inds]
        gb_inds_in = gb_inds_orig[gb_inds]

        temp_indices = group_temp_finder[0][gb_inds]
        walker_indices = group_temp_finder[1][gb_inds]
        leaf_indices = group_temp_finder[2][gb_inds]
        special_indices = (temp_indices * nwalkers + walker_indices) * int(1e6) + band_indices

        unique_N = self.xp.unique(self.band_N_vals)
        # remove 0
        unique_N = unique_N[unique_N != 0]

        do_synchronize = False
        device = self.xp.cuda.runtime.getDevice()

        units = 2 if not self.is_rj_prop else 2
        # random start to rotation around 
        start_unit = model.random.randint(units)

        ll_after = model.analysis_container_arr.likelihood(source_only=False)  #  - cp.sum(cp.log(cp.asarray(psd[:2])), axis=(0, 2))).get()

        # print(np.abs(new_state.log_like - ll_after).max())
        store_max_diff = np.abs(new_state.log_like[0] - ll_after).max()
        start_diffs = np.abs(new_state.log_like[0] - ll_after)
        # print("CHECKING 0:", store_max_diff, self.is_rj_prop)
        # self.check_ll_inject(new_state, verbose=True)

        per_walker_band_proposals = cp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_accepted = cp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        
        num_band_preload = 20000
        # TODO: make this adjustable?
        band_preload_size = max_data_store_size = 6000
        
        num_proposals_here = self.num_repeat_proposals if not self.is_rj_prop else 1
        source_prop_counter = cp.zeros(points_curr.shape[0], dtype=int)
        
        ll_change_log = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))
        total_keep = 0
        for tmp in range(units):
            remainder = (start_unit + tmp) % units

            # add back in all sources in the cold-chain 
            # residual from this group
            self.remove_cold_chain_sources_from_residual(model, band_sorter, units=units, remainder=remainder)
            # keep1 = (
            #     (band_indices % units == remainder) 
            #     & (band_indices < len(self.band_edges) - 2)
            #     & (band_indices > 1)
            #     & (self.band_N_vals[band_indices] < 1024)  # TESTING
            # ) 
            
            subset_of_interest = band_sorter.get_subset(units=units, remainder=remainder, extra_bool=(source_prop_counter < self.num_repeat_proposals))
            breakpoint()
            sources_of_interest = self.xp.ones_like(subset_of_interest.band_inds, dtype=bool)
            iteration_num = 0

            with open("tmp.dat", "w") as fp:
                tmp = f"{iteration_num}, {sources_of_interest.sum()}\n"
                fp.write(tmp)
                print(tmp)

            # TODO: move sources of interest inside? I do not think so right now
            while np.any(sources_of_interest):
                # st_1 = time.perf_counter()
                # MAKE THIS INTO A GENERATOR
                
                subset_of_interest.get_buffer(sources_of_interest, num_band_preload=num_band_preload)
                sources_now_map = cp.arange(special_indices.shape[0])[cp.in1d(special_indices, special_indices_unique) & (source_prop_counter < self.num_repeat_proposals)]
                
                # inject sources must include sources that have been turned off in these bands
                sources_inject_now_map = cp.arange(special_indices.shape[0])[cp.in1d(special_indices, special_indices_unique)]
                
                all_unique_band_combos = cp.asarray([temp_indices[sources_now_map], walker_indices[sources_now_map], band_indices[sources_now_map]]).T[special_indices_index]
                num_bands_here_total = all_unique_band_combos.shape[0]
                num_bands_now = special_indices_unique.shape[0]
                
                # sort these sources by band
                inds_sort_tmp = cp.argsort(special_indices[sources_now_map])
                sources_now_map[:] = sources_now_map[inds_sort_tmp]
                special_indices_now = special_indices[sources_now_map].copy()
                unique_special_indices_now, unique_special_indices_now_index, unique_special_indices_now_inverse, unique_special_indices_now_counts = cp.unique(special_indices_now, return_index=True, return_counts=True, return_inverse=True) 

                points_curr_tmp = points_curr[sources_now_map].copy()
                
                inj_special_indices_now = special_indices[sources_inject_now_map].copy()
                inj_unique_special_indices_now, inj_unique_special_indices_now_index, inj_unique_special_indices_now_inverse, inj_unique_special_indices_now_counts = cp.unique(inj_special_indices_now, return_index=True, return_counts=True, return_inverse=True) 

                # load data into buffer for these bands
                # 3 is number of sub-bands to store
                band_buffer_tmp = cp.zeros(
                    (num_bands_now * nchannels * max_data_store_size)
                    , dtype=complex
                )
                psd_buffer_tmp = cp.zeros(
                    (num_bands_now * nchannels * max_data_store_size)
                    , dtype=np.float64
                )

                # careful here with accessing memory
                band_buffer = band_buffer_tmp.reshape((num_bands_now, nchannels, max_data_store_size))
                psd_buffer = psd_buffer_tmp.reshape((num_bands_now, nchannels, max_data_store_size))
                
                buffer_start_index = (self.band_edges[all_unique_band_combos[:, 2] - 1] / self.df).astype(np.int32)
                # TODO: fix this 4????
                lower_f_lim = self.band_edges[all_unique_band_combos[:, 2]]  #  - self.band_N_vals[all_unique_band_combos[:, 2]] * self.df / 4
                higher_f_lim = self.band_edges[all_unique_band_combos[:, 2] + 1]  #  + self.band_N_vals[all_unique_band_combos[:, 2]] * self.df / 4
                frequency_lims = [lower_f_lim, higher_f_lim]
                inds1 = cp.repeat(all_unique_band_combos[:, 1], nchannels * band_buffer.shape[-1]).reshape((num_bands_now,) + band_buffer.shape[1:])
                inds2 = cp.repeat(cp.arange(nchannels), (num_bands_now * band_buffer.shape[-1])).reshape(nchannels, num_bands_now, band_buffer.shape[-1]).transpose(1, 0, 2)
                inds3 = cp.arange(band_buffer.shape[-1])[None, None, :] + (cp.repeat(buffer_start_index, nchannels * band_buffer.shape[-1]).reshape((num_bands_now,) + band_buffer.shape[1:]))
                inds_get = (inds1.flatten(), inds2.flatten(), inds3.flatten())
                rest_of_data = model.analysis_container_arr.data_shaped[0][inds_get].reshape((num_bands_now,) + band_buffer.shape[1:])
                # load rest of data into buffer (has current sources removed)
                band_buffer[:num_bands_now] += rest_of_data[:]
                psd_buffer[:num_bands_now] = model.analysis_container_arr.psd_shaped[0][inds_get].reshape((num_bands_now,) + band_buffer.shape[1:])
                
                # inject current sources into buffers
                params_change_coords = points_curr[sources_inject_now_map].copy()
                # TODO: check this???
                params_change_index = cp.arange(num_bands_now)[inj_unique_special_indices_now_inverse].copy().astype(np.int32)
                factors_params_change = -cp.ones_like(params_change_index, dtype=float)
                params_change_coords_in = self.parameter_transforms.both_transforms(
                    params_change_coords, cp=cp
                )
                params_change_N_vals = self.band_N_vals[band_indices[sources_inject_now_map]].copy()
                start_freq_inds = buffer_start_index.copy().astype(np.int32)
                
                try:
                    self.gb.generate_global_template(
                        params_change_coords_in,
                        params_change_index,
                        band_buffer_tmp,
                        data_length=band_buffer.shape[-1],
                        factors=factors_params_change,
                        data_splits=np.full(band_buffer.shape[0], self.gb.gpus[0]),
                        N=params_change_N_vals,
                        start_freq_ind=start_freq_inds,
                        **self.waveform_kwargs,
                    )
                except AssertionError:
                    breakpoint()
                
                # with open("tmp.dat", "a") as fp:
                #     tmp = f"inject: {iteration_num}, {sources_of_interest.sum()}"
                #     fp.write(tmp + "\n")
                #     print(tmp)
                # et_1 = time.perf_counter()
                # print("1st:", et_1 - st_1)
                
                for move_i in range(num_proposals_here):
                    # st_1 = time.perf_counter()
                    choice_fraction = cp.random.rand(num_bands_now)
                    
                    sources_picked_for_update = unique_special_indices_now_index + cp.floor(choice_fraction * unique_special_indices_now_counts).astype(int)
                    
                    params_to_update = points_curr_tmp[sources_picked_for_update].copy()
                    inds_to_update = sources_now_map[sources_picked_for_update].copy()
                    map_to_update = (temp_indices[inds_to_update], walker_indices[inds_to_update], band_indices[inds_to_update])
                    map_to_update_cpu = (temp_indices[inds_to_update].get(), walker_indices[inds_to_update].get(), band_indices[inds_to_update].get())
                    
                    if not self.is_rj_prop:
                        # custom group stretch
                        # TODO: work into main group stretch somehow
                        params_into_proposal = params_to_update[None, :, None, :]

                        friends_into_proposal = state.branches_supplemental["gb"][map_to_update_cpu]["friend_start_inds"][None, :, None]
                        branch_supps_into_proposal = BranchSupplemental({"friend_start_inds": friends_into_proposal}, base_shape=friends_into_proposal.shape)
                        inds_into_proposal = self.xp.ones(params_into_proposal.shape[:-1], dtype=bool)

                        # TODO: check detailed balance
                        q, update_factors = self.get_proposal({"gb": params_into_proposal}, model.random, s_inds_all={"gb": inds_into_proposal}, cp=self.xp, return_gpu=True, branch_supps=branch_supps_into_proposal)         
                        new_coords = q["gb"][0, :, 0, :]
                        
                        prev_logp = cp.asarray(self.gpu_priors["gb"].logpdf(params_to_update))  # , psds=self.mgh.psd_shaped[0][0], walker_inds=curr_index)
                        curr_logp = cp.asarray(self.gpu_priors["gb"].logpdf(new_coords))  # , psds=self.mgh.psd_shaped[0][0], walker_inds=curr_index)

                    else:
                        
                        logp_tmp = cp.asarray(self.gpu_priors["gb"].logpdf(params_to_update))
                        prev_logp = cp.zeros_like(logp_tmp)
                        curr_logp = cp.zeros_like(logp_tmp)

                        new_coords = params_to_update.copy()
                        inds = gb_inds_in[inds_to_update].copy()
                        update_factors = factors[inds_to_update].copy()
                        params_to_update[~inds, 0] = np.log(1e-80)
                        new_coords[inds, 0] = np.log(1e-80)
                        
                        prev_logp[inds] = logp_tmp[inds]
                        curr_logp[~inds] = logp_tmp[~inds]

                    assert not cp.any(cp.isinf(prev_logp))
                    # inputs into swap proposal
                    
                    # guard on the edges with too-large frequency proposals out of band that would not be physical
                    curr_logp[(new_coords[:, 1] / 1e3 < frequency_lims[0]) | (new_coords[:, 1] / 1e3 > frequency_lims[1])] = -np.inf
                    # TODO: 2 vs 4?
                    curr_logp[(cp.abs(params_to_update[:, 1] / 1e3 - new_coords[:, 1] / 1e3) / self.df).astype(int) > (self.band_N_vals[band_indices[inds_to_update]] / 4).astype(int)] = -np.inf

                    ll_diff = cp.full_like(prev_logp, -1e300)
                    opt_snr = cp.full_like(prev_logp, 0.0)
                    keep2 = ~cp.isinf(curr_logp)
                    # et_1 = time.perf_counter()
                    # print("2nd:", et_1 - st_1)
                
                    # st_1 = time.perf_counter()
                    params_remove = params_to_update[keep2].copy()
                    params_add = new_coords[keep2].copy()

                    params_remove_in = self.parameter_transforms.both_transforms(
                        params_remove, cp=cp
                    )

                    params_add_in = self.parameter_transforms.both_transforms(
                        params_add, cp=cp
                    )
                    
                    swap_N_vals = self.band_N_vals[band_indices[inds_to_update[keep2]]].copy()

                    # data indexes align with the buffers (1 per buffer except for inf priors)
                    data_index = cp.arange(num_bands_now, dtype=np.int32)[keep2].astype(np.int32)

                    # print("NEED TO CHECK THIS")
                    ll_diff[keep2] = cp.asarray(self.gb.new_swap_likelihood_difference(
                        params_remove_in,
                        params_add_in,
                        band_buffer_tmp,
                        psd_buffer_tmp,
                        start_freq_ind=start_freq_inds,
                        data_index=data_index,
                        noise_index=data_index,
                        adjust_inplace=False,
                        N=swap_N_vals,
                        data_length=band_buffer.shape[-1],
                        data_splits=model.analysis_container_arr.gpu_map,
                        phase_marginalize=self.phase_maximize,
                        return_cupy=True,
                        **self.waveform_kwargs,
                    ))

                    # rejection sampling on SNR
                    opt_snr[keep2] = self.gb.add_add.real ** (1/2)

                    # TODO: change limit
                    curr_logp[opt_snr < 3.0] = -np.inf
                    curr_beta = band_temps[map_to_update[0], map_to_update[2]]
                    # print("change priors?, need to adjust here")
                    
                    delta_logP = curr_beta * ll_diff + (curr_logp - prev_logp)
                    lnpdiff = delta_logP + update_factors.squeeze()
                    accept = lnpdiff >= cp.log(cp.random.rand(*lnpdiff.shape))
                    # need to copy to old array before changing in place
                    old_params_to_update = params_to_update.copy()
                    params_to_update[accept] = new_coords[accept]
                    points_curr_tmp[sources_picked_for_update] = params_to_update[:]
                    
                    if cp.any(accept):
                        inds_update_accept = inds_to_update[accept]
                        
                        ll_accept = ll_diff[accept]
                        if self.is_rj_prop:
                            gb_inds_in[inds_update_accept] = (~gb_inds_in[inds_update_accept])
                        
                        temp_inds_accept = temp_indices[inds_update_accept]
                        walker_inds_accept = walker_indices[inds_update_accept]
                        band_inds_accept = band_indices[inds_update_accept]
                        ll_change_log[temp_inds_accept, walker_inds_accept, band_inds_accept] += ll_accept

                        # switch accepted waveform
                        old_coords_for_change = old_params_to_update[accept].copy()
                        new_coords_for_change = new_coords[accept].copy()

                        old_change_index = cp.arange(num_bands_now)[accept].copy().astype(np.int32)
                        new_change_index = old_change_index.copy()
                        old_factors_change = +cp.ones_like(old_change_index, dtype=float)
                        new_factors_change = -cp.ones_like(new_change_index, dtype=float)
                        old_coords_for_change_in = self.parameter_transforms.both_transforms(
                            old_coords_for_change, cp=cp
                        )
                        new_coords_for_change_in = self.parameter_transforms.both_transforms(
                            new_coords_for_change, cp=cp
                        )
                        
                        old_change_N_vals = self.band_N_vals[band_indices[inds_to_update[accept]]].copy()
                        new_change_N_vals = old_change_N_vals.copy()

                        both_params_in = cp.concatenate([old_coords_for_change_in, new_coords_for_change_in], axis=0)
                        both_change_N_vals = cp.concatenate([old_change_N_vals, new_change_N_vals], axis=0).astype(np.int32)
                        both_factors_change = cp.concatenate([old_factors_change, new_factors_change], axis=0)
                        both_change_index = cp.concatenate([old_change_index, new_change_index], axis=0).astype(np.int32)
                        
                        # check_ll_1 = -1/2 * 4 * self.df * cp.sum(band_buffer.conj() * band_buffer * psd_buffer, axis=(-1, -2))

                        if cp.any(both_params_in[:, 1] / self.df - start_freq_inds[both_change_index] + both_change_N_vals > max_data_store_size):
                            raise ValueError("stretching past buffer")

                        # TODO: could combine this with the swap part to reduce waveform gen time
                        self.gb.generate_global_template(
                            both_params_in,
                            both_change_index,
                            band_buffer_tmp,
                            data_length=band_buffer.shape[-1],
                            factors=both_factors_change,
                            data_splits=np.full(band_buffer.shape[0], self.gb.gpus[0]),
                            N=both_change_N_vals,
                            start_freq_ind=start_freq_inds,
                            **self.waveform_kwargs,
                        )

                        # accepted_ll_diff = ll_diff[accept]
                        # check_ll_2 = -1/2 * 4 * self.df * cp.sum(band_buffer.conj() * band_buffer * psd_buffer, axis=(-1, -2))
                        # try:
                        #     assert cp.all(cp.abs((check_ll_2 - check_ll_1)[accept].real - ll_diff[accept]) < 1e-4)
                        # except AssertionError:
                        #     breakpoint()
                    # print(iteration_num, move_i)
                    self.mempool.free_all_blocks()
                    source_prop_counter[inds_to_update] += 1

                    # with open("tmp.dat", "a") as fp:
                    #     tmp = f"move {move_i}: {iteration_num}, {sources_of_interest.sum()}"
                    #     fp.write(tmp + "\n")
                    #     print(tmp)
                    # will recalculate prior anyways so leaving that out

                    # change WAVEFORMS THAT HAVE BEEN ACCEPTED

                    # et_1 = time.perf_counter()
                    # print("3rd:", et_1 - st_1)
                
                points_curr[sources_now_map] = points_curr_tmp[:]
                sources_of_interest[:] = (keep1 & (source_prop_counter < self.num_repeat_proposals))
                iteration_num += 1
                with open("tmp.dat", "a") as fp:
                    tmp = f"{iteration_num}, {sources_of_interest.sum()}"
                    fp.write(tmp + "\n")
                    print(tmp)
                self.mempool.free_all_blocks()
                # update prop counter
            
            # add back in all sources in the cold-chain 
            # residual from this group
            remove = (
                (band_indices % units == remainder)
                & (temp_indices == 0)
                & (band_indices < len(self.band_edges) - 2)
            ) 

            remove_coords = points_curr[remove].copy()
            remove_index = walker_indices[remove].copy().astype(np.int32)
            factors_remove = -1 * cp.ones_like(remove_index, dtype=float)
            remove_coords_in = self.parameter_transforms.both_transforms(
                remove_coords, cp=cp
            )
            
            remove_N_vals = self.band_N_vals[band_indices[remove]].copy()

            self.gb.generate_global_template(
                remove_coords_in,
                remove_index,
                model.analysis_container_arr.linear_data_arr,
                data_length=model.analysis_container_arr.data_length,
                factors=factors_remove,
                data_splits=model.analysis_container_arr.gpu_map,
                N=remove_N_vals,
                **self.waveform_kwargs,
            )

            self.xp.cuda.runtime.deviceSynchronize()
            # if False:  # self.time > 0:
            #     breakpoint()
            #     ll_after = (
            #         self.mgh.get_ll(include_psd_info=True)
            #     )
            #     # print(np.abs(new_state.log_like - ll_after).max())
            #     store_max_diff = np.abs(log_like_tmp[0].get() - ll_after).max()
            #     # print("CHECKING in:", tmp, store_max_diff)
            #     if store_max_diff > 3e-4:
            #         print("LARGER ERROR:", store_max_diff)
            #         breakpoint()
            #         self.check_ll_inject(new_state, verbose=True)
            #         # self.mgh.get_ll(include_psd_info=True, stop=True)

        print("NEED TO FIX ANALYSIS CONTAINER extra factor")
        ll_change_sum = ll_change_log.sum(axis=-1)
        new_state.log_like[0] += ll_change_sum[0].get()

        ll_after = model.analysis_container_arr.likelihood()
        
        print("ADD check")
        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        band_swaps_accepted = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        band_swaps_proposed = cp.zeros((len(self.band_edges) - 1, self.ntemps - 1), dtype=int)
        current_band_counts = cp.zeros((len(self.band_edges) - 1, self.ntemps), dtype=int)

        if (
            self.temperature_control is not None
            and self.time % 1 == 0
            and self.ntemps > 1
            and self.is_rj_prop
            # and False
        ):  
            ll_change_log_temp = cp.zeros((self.ntemps, self.nwalkers, self.num_bands))
        
            tmp_start = np.random.randint(2)
            for unit in range(2):
            
                start = (tmp_start + unit) % 2

                num_bands_unit = np.arange(self.num_bands)[start::2].shape[0]
                remove = (
                    (band_indices % 2 == start)
                    & (temp_indices == 0)
                ) 

                remove_coords = points_curr[remove].copy()
                remove_index = walker_indices[remove].copy().astype(np.int32)
                factors_remove = +1 * cp.ones_like(remove_index, dtype=float)
                remove_coords_in = self.parameter_transforms.both_transforms(
                    remove_coords, cp=cp
                )
                
                remove_N_vals = self.band_N_vals[band_indices[remove]].copy()

                self.gb.generate_global_template(
                    remove_coords_in,
                    remove_index,
                    model.analysis_container_arr.linear_data_arr,
                    data_length=model.analysis_container_arr.data_length,
                    factors=factors_remove,
                    data_splits=model.analysis_container_arr.gpu_map,
                    N=remove_N_vals,
                    **self.waveform_kwargs,
                )
                walkers_permuted = cp.asarray([cp.random.permutation(cp.arange(self.nwalkers)) for _ in range(self.ntemps * self.num_bands)]).reshape(self.num_bands, self.ntemps, self.nwalkers).transpose(0, 2, 1)[start::2]
                temp_index = cp.repeat(cp.arange(self.ntemps), self.num_bands * self.nwalkers).reshape(self.ntemps, self.num_bands, self.nwalkers).transpose(1, 2, 0)[start::2]
                band_index = cp.repeat(cp.arange(self.num_bands), self.ntemps * self.nwalkers).reshape(self.num_bands, self.ntemps, self.nwalkers).transpose(0, 2, 1)[start::2]
                special_index = (temp_index * nwalkers + walkers_permuted) * int(1e6) + band_index
                
                num_bands_preload_temp = 200
                num_bands_run = 0
                while num_bands_run < self.nwalkers * num_bands_unit:
                    start_ind = num_bands_run
                    end_ind = start_ind + num_bands_preload_temp

                    band_inds_now = band_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                    temp_inds_now = temp_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                    walker_inds_now = walkers_permuted.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                    special_inds_now = special_index.reshape(-1, self.ntemps)[start_ind:end_ind].copy()
                    num_bands_now = band_inds_now.shape[0]
                    # load data into buffer for these bands
                    # 3 is number of sub-bands to store
                    band_buffer_tmp = cp.zeros(
                        (num_bands_now * self.ntemps * nchannels * max_data_store_size)
                        , dtype=complex
                    )

                    residual_buffer_tmp = cp.zeros(
                        (num_bands_now * self.ntemps * nchannels * max_data_store_size)
                        , dtype=complex
                    )

                    psd_buffer_tmp = cp.zeros(
                        (num_bands_now * self.ntemps * nchannels * max_data_store_size)
                        , dtype=np.float64
                    )

                    # careful here with accessing memory
                    band_buffer = band_buffer_tmp.reshape((num_bands_now, self.ntemps, nchannels, max_data_store_size))
                    residual_buffer = residual_buffer_tmp.reshape((num_bands_now, self.ntemps, nchannels, max_data_store_size))
                    psd_buffer = psd_buffer_tmp.reshape((num_bands_now, self.ntemps, nchannels, max_data_store_size))
                    
                    buffer_start_index = (self.band_edges[band_inds_now - 1] / self.df).astype(np.int32)
                    
                    inds1 = cp.repeat(walker_inds_now.flatten(), nchannels * band_buffer.shape[-1]).reshape((num_bands_now, self.ntemps) + band_buffer.shape[2:])
                    inds2 = cp.repeat(cp.arange(nchannels), (num_bands_now * self.ntemps * band_buffer.shape[-1])).reshape(nchannels, num_bands_now, self.ntemps, band_buffer.shape[-1]).transpose(1, 2, 0, 3)
                    inds3 = cp.arange(band_buffer.shape[-1])[None, None, :] + (cp.repeat(buffer_start_index.flatten(), nchannels * band_buffer.shape[-1]).reshape((num_bands_now, self.ntemps) + band_buffer.shape[2:]))
                    inds_get = (inds1.flatten(), inds2.flatten(), inds3.flatten())
                    rest_of_data = model.analysis_container_arr.data_shaped[0][inds_get].reshape((num_bands_now,) + band_buffer.shape[1:])
                    # load rest of data into buffer (has current sources removed)
                    residual_buffer[:num_bands_now] = rest_of_data[:]
                    psd_buffer[:num_bands_now] = model.analysis_container_arr.psd_shaped[0][inds_get].reshape((num_bands_now,) + band_buffer.shape[1:])
                    
                    # gb_inds_in makes rj okay even though there are many other sources in points_curr
                    inds_keep = gb_inds_in & cp.arange(len(special_indices))[cp.in1d(special_indices, special_inds_now.flatten())]
                    if not cp.any(inds_keep):
                        num_bands_run += num_bands_preload_temp
                        continue

                    # inject current sources into buffers
                    params_inj = points_curr[inds_keep].copy()
                    ind_sort = cp.argsort(special_inds_now.flatten())
                    sorted_map = cp.searchsorted(special_inds_now.flatten()[ind_sort], special_indices[inds_keep], side="left")
                    assert not cp.any((sorted_map > num_bands_now * self.ntemps) | (special_indices[inds_keep] < special_inds_now.min()))
                    params_inj_index = ind_sort[sorted_map].astype(np.int32).copy()
                    factors_params_inj = +cp.ones_like(params_inj_index, dtype=float)
                    params_inj_in = self.parameter_transforms.both_transforms(
                        params_inj, cp=cp
                    )
                    params_inj_N_vals = self.band_N_vals[band_indices[inds_keep]].copy()
                    start_freq_inds = buffer_start_index.flatten().copy().astype(np.int32)
        
                    if cp.any(params_inj_in[:, 1] / self.df - start_freq_inds[params_inj_index] + params_inj_N_vals > max_data_store_size):
                        raise ValueError("stretching past buffer")

                    self.gb.generate_global_template(
                        params_inj_in,
                        params_inj_index,
                        band_buffer_tmp,
                        data_length=band_buffer.shape[-1],
                        factors=factors_params_inj,
                        data_splits=np.full(band_buffer.shape[0], self.gb.gpus[0]),
                        N=params_inj_N_vals,
                        start_freq_ind=start_freq_inds,
                        **self.waveform_kwargs,
                    )

                    current_lls = -1/2 * 4 * self.df * cp.sum((residual_buffer - band_buffer).conj() * (residual_buffer - band_buffer) * psd_buffer, axis=(-1, -2)).real
                    current_lls_orig = current_lls.copy()
                    for t in range(self.ntemps)[1:][::-1]:
                        st = time.perf_counter()
                        i1 = t
                        i2 = t - 1

                        tmp_i1 = band_buffer[:, i1].copy()
                        band_buffer[:, i1] = band_buffer[:, i2]
                        band_buffer[:, i2] = tmp_i1[:]
                        
                        new_lls = -1/2 * 4 * self.df * cp.sum((residual_buffer[:, i2:i1 + 1] - band_buffer[:, i2:i1 + 1]).conj() * (residual_buffer[:, i2:i1 + 1] - band_buffer[:, i2:i1 + 1]) * psd_buffer[:, i2:i1 + 1], axis=(-1, -2)).real
                        old_lls = current_lls[:, i2:i1 + 1]
                        
                        beta1 = band_temps[(band_inds_now[:, 0], i1)]
                        beta2 = band_temps[(band_inds_now[:, 0], i2)]

                        paccept = beta1 * (new_lls[:, 0] - old_lls[:, 1]) + beta2 * (new_lls[:, 1] - old_lls[:, 0])
                        # paccept = bi * (band_here_i1->swapped_like - band_here_i->current_like) + bi1 * (band_here_i->swapped_like - band_here_i1->current_like);

                        raccept = cp.log(cp.random.uniform(size=paccept.shape))
                        sel = paccept > raccept

                        current_lls[sel, i2:i1 + 1] = new_lls[sel]
                        
                        # reverse not accepted ones
                        tmp_i1 = band_buffer[~sel, i1].copy()
                        band_buffer[~sel, i1] = band_buffer[~sel, i2]
                        band_buffer[~sel, i2] = tmp_i1[:]
                        
                        band_swaps_accepted[band_inds_now[:, 0], i2] += sel.astype(int)
                        band_swaps_proposed[band_inds_now[:, 0], i2] += 1
                        
                        band_inds_exchange_i1 = band_inds_now[sel, i1]
                        walker_inds_exchange_i1 = walker_inds_now[sel, i1]
                        band_inds_exchange_i2 = band_inds_now[sel, i2]
                        walker_inds_exchange_i2 = walker_inds_now[sel, i2]
                        
                        special_ind_test_1 = (i1 * nwalkers + walker_inds_exchange_i1) * int(1e6) + band_inds_exchange_i1
                        special_ind_test_2 = (i2 * nwalkers + walker_inds_exchange_i2) * int(1e6) + band_inds_exchange_i2

                        # temp_indices[fix_1] = i2
                        # temp_indices[fix_2] = i1

                        ind_sort_1 = cp.argsort(special_ind_test_1.flatten())
                        ind_keep_1 = cp.in1d(special_indices, special_ind_test_1)
                        sorted_map_1 = cp.searchsorted(special_ind_test_1[ind_sort_1], special_indices[ind_keep_1], side="left")
                        
                        ind_sort_2 = cp.argsort(special_ind_test_2.flatten())
                        ind_keep_2 = cp.in1d(special_indices, special_ind_test_2)
                        sorted_map_2 = cp.searchsorted(special_ind_test_2[ind_sort_2], special_indices[ind_keep_2], side="left")
                        
                        special_indices[ind_keep_1] = special_ind_test_2[ind_sort_1[sorted_map_1]]
                        temp_indices[ind_keep_1] = i2
                        walker_indices[ind_keep_1] = walker_inds_exchange_i2[ind_sort_1[sorted_map_1]]
                        # do not need to change band index but check it
                        assert cp.all(band_indices[ind_keep_1] == band_inds_exchange_i2[ind_sort_1[sorted_map_1]])
                        
                        special_indices[ind_keep_2] = special_ind_test_1[ind_sort_2[sorted_map_2]]
                        temp_indices[ind_keep_2] = i1
                        walker_indices[ind_keep_2] = walker_inds_exchange_i1[ind_sort_2[sorted_map_2]]
                        
                        et = time.perf_counter()
                        # print(et - st, t, num_bands_run, self.nwalkers * num_bands_unit)
                
                    diffs = current_lls - current_lls_orig
                    # TODO: this should be = not += (?)
                    ll_change_log_temp[(temp_inds_now.flatten(),walker_inds_now.flatten(), band_inds_now.flatten())] = diffs.flatten()
                    num_bands_run += num_bands_preload_temp

                add = (
                    (band_indices % 2 == start)
                    & (temp_indices == 0)
                ) 

                add_coords = points_curr[add].copy()
                add_index = walker_indices[add].copy().astype(np.int32)
                factors_add = -1 * cp.ones_like(add_index, dtype=float)
                add_coords_in = self.parameter_transforms.both_transforms(
                    add_coords, cp=cp
                )
                
                add_N_vals = self.band_N_vals[band_indices[add]].copy()

                self.gb.generate_global_template(
                    add_coords_in,
                    add_index,
                    model.analysis_container_arr.linear_data_arr,
                    data_length=model.analysis_container_arr.data_length,
                    factors=factors_add,
                    data_splits=model.analysis_container_arr.gpu_map,
                    N=add_N_vals,
                    **self.waveform_kwargs,
                )
            # adapt if desired
            if self.time > 50:
                ratios = (band_swaps_accepted / band_swaps_proposed).T
                betas0 = band_temps.copy().T
                betas1 = betas0.copy()

                # Modulate temperature adjustments with a hyperbolic decay.
                decay = self.temperature_control.adaptation_lag / (self.time + self.temperature_control.adaptation_lag)
                kappa = decay / self.temperature_control.adaptation_time

                # Construct temperature adjustments.
                dSs = kappa * (ratios[:-1] - ratios[1:])

                # Compute new ladder (hottest and coldest chains don't move).
                deltaTs = cp.diff(1 / betas1[:-1], axis=0)

                deltaTs *= cp.exp(dSs)
                betas1[1:-1] = 1 / (cp.cumsum(deltaTs, axis=0) + 1 / betas1[0])

                # Don't mutate the ladder here; let the client code do that.
                dbetas = betas1 - betas0

                band_temps += self.xp.asarray(dbetas.T)

            print("NEED TO FIX ANALYSIS CONTAINER extra factor")
            ll_change_sum_temp = ll_change_log_temp.sum(axis=-1)
            new_state.log_like[0] += ll_change_sum_temp[0].get()

            ll_after = model.analysis_container_arr.likelihood()
            
            self.mempool.free_all_blocks()

        
        special_indices_finish = (temp_indices * nwalkers + walker_indices) * int(1e6) + points_curr[:, 1]
        special_inds_temp_walker = (temp_indices * nwalkers + walker_indices)
        sorted_inds = cp.argsort(special_indices_finish)

        uni, uni_inds, uni_inverse = cp.unique(special_inds_temp_walker[sorted_inds], return_index=True, return_inverse=True)

        leaf_inds_new_tmp = cp.arange(special_indices_finish.shape[0]) - uni_inds[uni_inverse]
        leaf_inds_new = cp.zeros_like(leaf_inds_new_tmp)
        leaf_inds_new[sorted_inds] = leaf_inds_new_tmp
        
        new_state.branches["gb"].coords[(temp_indices.get(), walker_indices.get(), leaf_inds_new.get())] = points_curr.get()
        new_state.branches["gb"].inds[(temp_indices.get(), walker_indices.get(), leaf_inds_new.get())] = gb_inds_in.get()
        et_all = time.perf_counter()
        print(et_all - st_all)

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

        new_inds = cp.asarray(new_state.branches_inds["gb"])
            
        # in-model inds will not change
        tmp_freqs_find_bands = cp.asarray(new_state.branches_coords["gb"][:, :, :, 1])

        # calculate current band counts
        band_here = (cp.searchsorted(self.band_edges, tmp_freqs_find_bands.flatten() / 1e3, side="right") - 1).reshape(tmp_freqs_find_bands.shape)

        # get binaries per band
        special_band_here_num_per_band = ((group_temp_finder[0] * nwalkers + group_temp_finder[1]) * int(1e6) + band_here)[new_inds]
        unique_special_band_here_num_per_band, unique_special_band_here_num_per_band_count = cp.unique(special_band_here_num_per_band, return_counts=True)
        temp_walker_index_num_per_band = (unique_special_band_here_num_per_band / 1e6).astype(int)
        temp_index_num_per_band = (temp_walker_index_num_per_band / nwalkers).astype(int)
        walker_index_num_per_band = temp_walker_index_num_per_band - temp_index_num_per_band * nwalkers
        band_index_num_per_band = unique_special_band_here_num_per_band - temp_walker_index_num_per_band * int(1e6)

        per_walker_band_counts = cp.zeros((ntemps, nwalkers, self.num_bands), dtype=int)
        per_walker_band_counts[temp_index_num_per_band, walker_index_num_per_band, band_index_num_per_band] = unique_special_band_here_num_per_band_count
        
        # TEMPERING
        self.temperature_control.swaps_accepted = np.zeros(ntemps - 1)
        self.temperature_control.swaps_proposed = np.zeros(ntemps - 1)

        self.mempool.free_all_blocks()

        self.time += 1
        # self.xp.cuda.runtime.deviceSynchronize()
        breakpoint()
        new_state.sub_states["gb"].update_band_information(
            band_temps, per_walker_band_proposals.sum(axis=1).T, per_walker_band_accepted.sum(axis=1).T, band_swaps_proposed, band_swaps_accepted,
            per_walker_band_counts, self.is_rj_prop
        )
        # TODO: check rj numbers

        # new_state.log_like[:] = self.check_ll_inject(new_state)

        self.mempool.free_all_blocks()

        # if self.is_rj_prop:
        #     pass  # print(self.name, "2nd count check:", new_state.branches["gb"].inds.sum(axis=-1).mean(axis=-1), "\nll:", new_state.log_like[0] - orig_store, new_state.log_like[0])

        new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplemental)
        
        return new_state, accepted

    def check_ll_inject(self, new_state, verbose=False):

        check_ll = self.mgh.get_ll(include_psd_info=True).copy()

        nleaves_max = new_state.branches["gb"].shape[-2]
        for i in range(2):
            self.mgh.channel1_data[0][self.nwalkers * self.data_length * i: self.nwalkers * self.data_length * (i + 1)] = self.mgh.channel1_base_data[0][:]
            self.mgh.channel2_data[0][self.nwalkers * self.data_length * i: self.nwalkers * self.data_length * (i + 1)] = self.mgh.channel2_base_data[0][:]
        
        coords_out_gb = new_state.branches["gb"].coords[0, new_state.branches["gb"].inds[0]]
        coords_in_in = self.parameter_transforms.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(self.band_edges.get(), coords_in_in[:, 1], side="right") - 1
        assert np.all(band_inds == new_state.branches["gb"].branch_supplemental.holder["band_inds"][0, new_state.branches["gb"].inds[0]])

        walker_vals = np.tile(np.arange(self.nwalkers), (nleaves_max, 1)).transpose((1, 0))[new_state.branches["gb"].inds[0]]

        data_index_1 = ((band_inds % 2) + 0) * self.nwalkers + walker_vals

        data_index = cp.asarray(data_index_1).astype(cp.int32)

        # goes in as -h
        factors = -cp.ones_like(data_index, dtype=cp.float64)

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
        if verbose:
            print(check_ll_diff1)

        # breakpoint()
        return check_ll_new

    @property
    def ranks_needed(self): 
        if not hasattr(self, "_ranks_needed"):
            raise ValueError("Need to set ranks needed for this class.")

        return self._ranks_needed

    @ranks_needed.setter
    def ranks_needed(self, ranks_needed):
        assert isinstance(ranks_needed, int)
        self._ranks_needed = ranks_needed
    

class GBSpecialStretchMove(GBSpecialBase):
    pass

class GBSpecialRJPriorMove(GBSpecialBase):
    pass

class GBSpecialRJSearchMove(GBSpecialBase):
    def get_rank_function(self):
        return gb_search_func

    def setup(self, branches):
        self.interact_with_search()
        super(GBSpecialRJSearchMove, self).setup(branches)

    def interact_with_search(self):
        search_rank = self.ranks[0]

        search_ch = self.comm.irecv(source=search_rank)
        if search_ch.get_status():
            search_req = search_ch.wait()

            if "receive" in search_req and search_req["receive"]:
                search_dict = self.comm.recv(source=search_rank)
                self.rj_proposal_distribution["gb"] = make_gmm(self.gb, search_dict["search"])

            if "send" in search_req and search_req["send"]:
                # get random instance of residual, psd, lisasens
                # TODO: decide about random versus max ll
                random_ind = np.random.randint(self.nwalkers)

                data = [self.mgh.data_shaped[0][0][random_ind].get(), self.mgh.data_shaped[1][0][random_ind].get()]
                psd = [self.mgh.psd_shaped[0][0][random_ind].get(), self.mgh.psd_shaped[1][0][random_ind].get()]
                lisasens = [self.mgh.psd_shaped[0][0][random_ind].get(), self.mgh.lisasens_shaped[1][0][random_ind].get()]
                
                output_data = dict(
                    data=data,
                    psd=psd,
                    lisasens=lisasens
                )
                self.comm.send(output_data, dest=search_rank)

        else:
            search_ch.cancel()
            
        print("CHECK INSIDE PROP")

class GBSpecialRJRefitMove(GBSpecialBase):
    def get_rank_function(self):
        return gb_refit_func
