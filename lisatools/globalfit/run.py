import numpy as np
from copy import deepcopy

from .generatefuncs import GenerateCurrentState
import numpy as np
from mpi4py import MPI
import sys, os
import warnings
from copy import deepcopy
from ..analysiscontainer import AnalysisContainer, AnalysisContainerArray
from ..datacontainer import DataResidualArray
from ..sensitivity import AE1SensitivityMatrix
from .state import GFState
from ..detector import EqualArmlengthOrbits
from ..stochastic import HyperbolicTangentGalacticForeground
from .hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend, EMRIHDFBackend
from eryn.backends import HDFBackend
from eryn.moves import Move
from ..detector import LISAModel
from eryn.state import BranchSupplemental
from eryn.ensemble import EnsembleSampler
from .engine import GlobalFitEngine
# from global_fit_input.global_fit_settings import get_global_fit_settings
from ..utils.multigpudataholder import MultiGPUDataHolder
import cupy as cp
from ..sampling.prior import GBPriorWrap
from .psdglobal import log_like as psd_log_like
from .psdglobal import PSDwithGBPriorWrap
from .moves import MBHSpecialMove

from eryn.state import State as eryn_State
from eryn.ensemble import _FunctionWrapper
from .moves import GlobalFitMove
from .hdfbackend import save_to_backend_asynchronously_and_plot
from .utils import new_sens_mat, BasicResidualacsLikelihood
from .utils import SetupInfoTransfer, AllSetupInfoTransfer

from ..sensitivity import get_sensitivity
from .hdfbackend import GFHDFBackend
from .state import GFState
from bbhx.waveformbuild import BBHWaveformFD

from .mbhsearch import ParallelMBHSearchControl
from .galaxyglobal import run_gb_pe, run_gb_bulk_search, fit_each_leaf
from .psdglobal import run_psd_pe
from .mbhglobal import run_mbh_pe

from ..sampling.stopping import SearchConvergeStopping, MPICommunicateStopping
from .plot import RunResultsProduction
from .hdfbackend import save_to_backend_asynchronously_and_plot

from gbgpu.gbgpu import GBGPU

from eryn.backends import HDFBackend

import time
import os
import pickle

from abc import ABC
import logging

# from global_fit_input.global_fit_settings import get_global_fit_settings

def init_logger(filename=None, level=logging.DEBUG, name='GlobalFit'):
    """ Initialize a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if (len(logger.handlers) < 2):
        formatter = logging.Formatter("%(asctime)s - %(name)s - "
                                      "%(levelname)s - %(message)s")
        if filename:
            rfhandler = logging.FileHandler(filename)
            logger.addHandler(rfhandler)
            rfhandler.setFormatter(formatter)
        if level:
            shandler = logging.StreamHandler(sys.stdout)
            shandler.setLevel(level)
            shandler.setFormatter(formatter)
            logger.addHandler(shandler)
    return logger

class CurrentInfoGlobalFit:
    def __init__(self, settings, get_last_state_info=True):

        self.settings_dict = settings
        self.current_info = deepcopy(settings)

        backend_path = self.general_info["file_information"]["fp_main"]
        self.backend = GFHDFBackend(backend_path)

        mbh_search_file = settings["general"]["file_information"]["fp_mbh_search_base"] + "_output.pickle"
        
        if os.path.exists(mbh_search_file):
            with open(mbh_search_file, "rb") as fp:
                mbh_output_point_info = pickle.load(fp)

            if "output_points_pruned" in mbh_output_point_info:
                self.initialize_mbh_state_from_search(mbh_output_point_info)
              
        # gmm info
        if os.path.exists(settings["general"]["file_information"]["fp_gb_gmm_info"]):
            with open(settings["general"]["file_information"]["fp_gb_gmm_info"], "rb") as fp_gmm:
                gmm_info_dict = pickle.load(fp_gmm)
                gb_search_proposal_gmm_info = gmm_info_dict["search"]
                gb_refit_proposal_gmm_info = gmm_info_dict["refit"]
        else:
            gb_search_proposal_gmm_info = None
            gb_refit_proposal_gmm_info = None

        # TODO: remove this?
        if "gb" in self.source_info:
            self.source_info["gb"]["search_gmm_info"] = gb_search_proposal_gmm_info
            self.source_info["gb"]["refit_gmm_info"] = gb_refit_proposal_gmm_info

    def initialize_mbh_state_from_search(self, mbh_output_point_info):
        output_points_pruned = np.asarray(mbh_output_point_info["output_points_pruned"]).transpose(1, 0, 2)
        coords = np.zeros((self.source_info["gb"]["pe_info"]["ntemps"], self.source_info["gb"]["pe_info"]["nwalkers"], output_points_pruned.shape[1], self.source_info["mbh"]["pe_info"]["ndim"]))
        assert output_points_pruned.shape[0] >= self.source_info["mbh"]["pe_info"]["nwalkers"]
        
        coords[:] = output_points_pruned[None, :self.source_info["mbh"]["pe_info"]["nwalkers"]]
        self.source_info["mbh"]["mbh_init_points"] = coords.copy()
    
    def get_data_psd(self, **kwargs):
        # self passed here to access all current info
        return self.general_info["generate_current_state"](self, **kwargs) 

    @property
    def settings(self):
        return self.settings_dict

    @property
    def all_info(self):
        return self.current_info

    @property
    def general_info(self):
        return self.current_info["general"]

    @property
    def source_info(self):
        return self.current_info["source_info"]

    @property
    def rank_info(self):
        return self.current_info["rank_info"]

    @property
    def gpu_assignments(self):
        return self.current_info["gpu_assignments"]

import numpy as np
from mpi4py import MPI
import os
import warnings
from copy import deepcopy
from ..analysiscontainer import AnalysisContainer, AnalysisContainerArray
from ..datacontainer import DataResidualArray
from ..sensitivity import AE1SensitivityMatrix
from .state import GFState
from ..detector import EqualArmlengthOrbits
from ..stochastic import HyperbolicTangentGalacticForeground
from .hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend
from eryn.backends import HDFBackend
from eryn.moves import Move
from ..detector import LISAModel
from eryn.state import BranchSupplemental
from eryn.ensemble import EnsembleSampler
from .engine import GlobalFitEngine
# from global_fit_input.global_fit_settings import get_global_fit_settings
from ..utils.multigpudataholder import MultiGPUDataHolder
import cupy as cp
from ..sampling.prior import GBPriorWrap
from .psdglobal import log_like as psd_log_like
from .psdglobal import PSDwithGBPriorWrap
from eryn.model import Model
from eryn.state import State as eryn_State
from eryn.ensemble import _FunctionWrapper
from .run import CurrentInfoGlobalFit
from .moves import GlobalFitMove, GFCombineMove
from .hdfbackend import save_to_backend_asynchronously_and_plot
from eryn.moves import CombineMove
from .moves import GBSpecialStretchMove, GBSpecialRJRefitMove, GBSpecialRJSearchMove, GBSpecialRJPriorMove, PSDMove
from .galaxyglobal import make_gmm
from .utils import new_sens_mat, BasicResidualacsLikelihood
from .recipe import Recipe

class GlobalFit:
    def __init__(self, gf_branch_information, curr, comm):
        self.gf_branch_information = gf_branch_information
        self.comm = comm
        self.curr = curr
        self.rank = comm.Get_rank()
        self.nwalkers = self.curr.general_info["nwalkers"]
        self.ntemps = self.curr.general_info["ntemps"]
        self.all_ranks = list(range(self.comm.Get_size()))
        self.used_ranks = []
        self.head_rank = self.curr.rank_info["head_rank"]
        self.main_rank = self.curr.rank_info["main_rank"]
        self.used_ranks.append(self.head_rank)
        self.used_ranks.append(self.main_rank)

        self.ranks_to_give =  deepcopy(self.all_ranks)
        if self.head_rank in self.ranks_to_give:
            self.ranks_to_give.remove(self.head_rank)
        self.ranks_to_give.remove(self.main_rank)

        if comm.Get_size() < 3:
            self.results_rank = self.main_rank
        else:
            self.results_rank = self.ranks_to_give.pop()
            self.used_ranks.append(self.results_rank)

        level = logging.DEBUG
        name = "GlobalFit"
        self.logger = init_logger(filename="global_fit.log", level=level, name=name)

    def load_info(self, priors):
        self.logger.debug("need to adjust file path")
        # TODO: update to generalize
        backend_path = self.curr.general_info["file_information"]["fp_main"]
        if os.path.exists(backend_path):
            state = GFHDFBackend(backend_path, sub_state_bases=self.gf_branch_information.branch_state, sub_backend=self.gf_branch_information.branch_backend).get_last_sample()  # .get_a_sample(0)

        else:
            self.logger.debug("update this somehow")
            # print("update this somehow")
            # # breakpoint()
            # start from priors by default
            coords = {key: priors[key].rvs(size=(self.ntemps, self.nwalkers, self.gf_branch_information.nleaves_max[key])) for key in self.gf_branch_information.branch_names}
            inds = {key: np.zeros((self.ntemps, self.nwalkers, self.gf_branch_information.nleaves_max[key]), dtype=bool) for key in self.gf_branch_information.branch_names}
            # TODO: make this more generic to anything
            inds["psd"][:] = True
            inds["galfor"][:] = True
            state = GFState(coords, inds=inds, random_state=np.random.get_state(), sub_state_bases=self.gf_branch_information.branch_state)
            
            band_temps = np.zeros((len(self.curr.source_info["gb"]["band_edges"]) - 1, self.ntemps))
            state.sub_states["gb"].initialize_band_information(self.nwalkers, self.ntemps, self.curr.source_info["gb"]["band_edges"], band_temps)
            state.log_like = np.zeros((self.ntemps, self.nwalkers))
            state.log_prior = np.zeros((self.ntemps, self.nwalkers))
            # self.logger.debug("pickle state load success")
        return state

    def setup_acs(self, state):
    
        cp.cuda.runtime.setDevice(self.curr.general_info["gpus"][0])

        df = self.curr.general_info["df"]
        N = self.curr.general_info["A_inj"].shape[-1]

        f_arr = (np.arange(N) + self.curr.general_info["start_freq_ind"]) * df

        acs_tmp = []
        for w in range(self.nwalkers):
            data_res_arr = DataResidualArray([
                self.curr.general_info["A_inj"].copy(), 
                self.curr.general_info["E_inj"].copy(), 
            ], f_arr=f_arr)
            psd_params = state.branches_coords["psd"][0, w, 0]
            galfor_params = state.branches_coords["galfor"][0, w, 0]
            sens_AE = new_sens_mat(f"walker_{w}", psd_params, galfor_params, data_res_arr.f_arr)
            # sens_AE[0] = psd[0][w]
            # sens_AE[1] = psd[1][w]
            acs_tmp.append(AnalysisContainer(deepcopy(data_res_arr), deepcopy(sens_AE)))
        
        gpus = self.curr.general_info["gpus"]
        acs = AnalysisContainerArray(acs_tmp, gpus=gpus)   

        for name, source_info in self.curr.source_info.items():
            if name not in self.curr.all_info["gf_branch_information"].branch_names:
                continue

            print("want to remove this emri thing eventually")
            if True:  # name == "psd" or name == "emri":
                continue

            templates_tmp = cp.asarray(source_info["get_templates"](state, source_info, self.curr.general_info))

            acs.add_signal_to_residual(templates_tmp)  # no need to adjust data index or start_freq_ind as it is taken care of

            del templates_tmp
            cp.get_default_memory_pool().free_all_blocks()
            print(f"added {name} to acs.")
        # generated_info = generate(state, self.curr.settings_dict, include_gbs=False, include_mbhs=False, include_psd=True, include_lisasens=True, include_ll=True, include_source_only_ll=True, n_gen_in=self.nwalkers, return_prior_val=False, fix_val_in_gen=["gb", "psd", "mbh"])
        # generated_info_with_gbs = generate(state, self.curr.settings_dict, include_psd=True, include_mbhs=False, include_lisasens=True, include_ll=True, include_source_only_ll=True, n_gen_in=self.nwalkers, return_prior_val=False, fix_val_in_gen=["gb", "psd", "mbh"])

        # data = generated_info["data"]
        # psd = generated_info["psd"]
        # lisasens = generated_info["lisasens"]

        return acs 

    def run_global_fit(self):
        
        backend_path = self.curr.general_info["file_information"]["fp_main"]
        
        backend = GFHDFBackend(backend_path, sub_backend=self.gf_branch_information.branch_backend, sub_state_bases=self.gf_branch_information.branch_state)
        if self.rank == self.curr.settings_dict["rank_info"]["main_rank"]: 

            general_info = self.curr.settings_dict["general"]
            
            branch_names = self.gf_branch_information.branch_names
            ndims = self.gf_branch_information.ndims
            nleaves_max = self.gf_branch_information.nleaves_max
            nleaves_min = self.gf_branch_information.nleaves_min
            nwalkers = general_info["nwalkers"]
            ntemps = general_info["ntemps"]

            priors = {}
            periodic = {}
            for name in branch_names:
                if name not in self.curr.source_info:
                    continue
                for key, value in self.curr.source_info[name]["priors"].items():
                    priors[key] = value
                
                if "periodic" in self.curr.source_info[name] and self.curr.source_info[name]["periodic"] is not None:
                    for key, value in self.curr.source_info[name]["periodic"].items():
                        periodic[key] = value
                # breakpoint()
           
            state = self.load_info(priors)
            self.logger.debug("state loaded")

            supps_base_shape = (ntemps, nwalkers)
            walker_vals = np.tile(np.arange(nwalkers), (ntemps, 1))
            supps = BranchSupplemental({"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True)
            state.supplemental = supps

            # backend.reset(
            #     nwalkers,
            #     ndims,
            #     nleaves_max=nleaves_max,
            #     ntemps=ntemps,
            #     branch_names=branch_names,
            #     nbranches=len(branch_names),
            #     rj=True,
            #     moves=None,
            #     num_mbhs=nleaves_max["mbh"],
            #     num_bands=state.sub_states["gb"].band_info["num_bands"],
            #     band_edges=state.sub_states["gb"].band_info["band_edges"],
            # )
           
            # backend.grow(1, None)

            # gb_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_gb.h5")
            # psd_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_psd.h5")
            # mbh_backend = HDFBackend("global_fit_output/eighth_run_through_parameter_estimation_mbh.h5")

            # last_gb = gb_backend.get_last_sample()
            # last_psd = psd_backend.get_last_sample()
            # last_mbh = mbh_backend.get_last_sample()

            # state.branches["gb"] = deepcopy(last_gb.branches["gb"])
            # state.branches["psd"].coords[:] = last_psd.branches["psd"].coords[0, :nwalkers]
            # # order of call function changed for galfor 
            # galfor_coords_orig = last_psd.branches["galfor"].coords[0, :nwalkers]
            # galfor_coords = np.zeros_like(galfor_coords_orig)
            # galfor_coords[:, :, 0] = galfor_coords_orig[:, :, 0]
            # galfor_coords[:, :, 1] = galfor_coords_orig[:, :, 3]
            # galfor_coords[:, :, 2] = galfor_coords_orig[:, :, 1]
            # galfor_coords[:, :, 3] = galfor_coords_orig[:, :, 2]
            # galfor_coords[:, :, 4] = galfor_coords_orig[:, :, 4]
            # state.branches["galfor"].coords[:] = galfor_coords
            # state.branches["mbh"].coords[:] = last_mbh.branches["mbh"].coords[0, :nwalkers]

            # # FOR TESTING
            # state.branches["gb"].coords[:] = state.branches["gb"].coords[0, 0][None, None, :, :]
            # state.branches["gb"].inds[:] = state.branches["gb"].inds[0, 0][None, None, :]
            # state.branches["mbh"].coords[:] = state.branches["mbh"].coords[0, 0][None, None, :, :]
            # state.branches["psd"].coords[:] = state.branches["psd"].coords[0, 0][None, None, :, :]
            # state.branches["galfor"].coords[:] = state.branches["galfor"].coords[0, 0][None, None, :, :]

            # accepted = np.zeros((ntemps, nwalkers), dtype=int)
            # swaps_accepted = np.zeros((ntemps - 1,), dtype=int)
            # state.log_like = np.zeros((ntemps, nwalkers))
            # state.log_prior = np.zeros((ntemps, nwalkers))
            # state.betas = np.ones((ntemps,))

            # backend.save_step(state, accepted, rj_accepted=accepted, swaps_accepted=swaps_accepted)

            A_inj = general_info["A_inj"].copy()
            E_inj = general_info["E_inj"].copy()

            generate = GenerateCurrentState(A_inj, E_inj)
            self.logger.debug("generate function created")

            acs = self.setup_acs(state)
            self.logger.debug("acs setup done")

            state.log_like[:] = acs.likelihood()

            like_mix = BasicResidualacsLikelihood(acs)

            backend = GFHDFBackend(
                backend_path,   # self.curr.settings_dict["general"]["file_information"]["fp_main"],
                compression="gzip",
                compression_opts=9,
                comm=self.comm,
                save_plot_rank=self.results_rank,
                sub_backend=self.gf_branch_information.branch_backend,
                sub_state_bases=self.gf_branch_information.branch_state
            )

            extra_reset_kwargs = {}
            # TODO: fix this somehow
            for name in branch_names:
                if name in state.sub_states and state.sub_states[name] is not None:
                    extra_reset_kwargs = {**extra_reset_kwargs, **state.sub_states[name].reset_kwargs}

            if not backend.initialized:
                backend.reset(
                    nwalkers,
                    ndims,
                    nleaves_max=nleaves_max,
                    ntemps=ntemps,
                    branch_names=branch_names,
                    nbranches=len(branch_names),
                    rj=False,
                    moves=None,
                    **extra_reset_kwargs
                )

            # setup_info_all = None
            # for name in branch_names:
            #     if name not in self.curr.source_info:
            #         setup_info = SetupInfoTransfer(name=name)
                    
            #     elif "setup_func" in self.curr.source_info[name]:
            #         setup_info = self.curr.source_info[name]["setup_func"](self.gf_branch_information, self.curr, acs, priors, state)
            #     else:
            #         setup_info = SetupInfoTransfer(name=name)

            #     if setup_info_all is None:
            #         setup_info_all = setup_info
            #     else:
            #         setup_info_all += setup_info

            self.recipe = Recipe()
            setup_info_all = self.curr.settings_dict["setup_function"](self.recipe, self.gf_branch_information, self.curr, acs, priors, state)
            print("need to setup moves that use parallel resources")

            # backend.grow(1, None)
            # accepted = np.zeros((self.ntemps, self.nwalkers), dtype=int)
            # swaps_accepted = np.zeros((self.ntemps - 1), dtype=int)
            # backend.save_step(state, accepted, swaps_accepted=swaps_accepted)
            # exit()

            rank_instructions = {}
            print("NEED TO FIX")
            for move_tmp in []:  # setup_info_all.in_model_moves + setup_info_all.rj_moves:  
                if isinstance(move_tmp, tuple) or isinstance(move_tmp, list):
                    move_tmp = move_tmp[0]

                # adjust for combine move
                if isinstance(move_tmp, GFCombineMove):
                    moves_list = move_tmp.moves
                else:
                    moves_list = [move_tmp]

                for move in moves_list:
                    if not isinstance(move, GlobalFitMove):
                        raise ValueError("All moves must be a subclass of GlobalFitMove.")
                    
                    move.comm = self.comm
                    if len(move.gpus) > 0 or (move.ranks_needed > 0 and not move.ranks_initialized):
                        if len(move.gpus) > 0:
                            assert move.ranks_needed > 0

                        tmp_ranks = []
                        for _ in range(move.ranks_needed):
                            try:
                                tmp_ranks.append(self.ranks_to_give.pop())
                            except IndexError:
                                raise ValueError("Not enough MPI ranks to give.")
                        self.used_ranks += tmp_ranks
                        move.assign_ranks(tmp_ranks)
                        for rank in tmp_ranks:
                            rank_instructions[rank] = {"function": move.get_rank_function(), "class_rank_list": tmp_ranks, "gpus": move.gpus}
                
            # stop unneeded processes
            for rank in self.all_ranks:
                if rank in self.used_ranks:
                    continue
                self.comm.send("stop", dest=rank)
                
            for rank, tmp in rank_instructions.items():
                self.comm.send({"rank": rank, **tmp}, dest=rank)
            
            self.recipe.backend = backend
            backend.add_recipe(self.recipe)

            from eryn.moves import StretchMove
            _tmp_move = StretchMove(live_dangerously=True)
            # permute False is there for the PSD sampling for now
            sampler_mix = GlobalFitEngine(
                acs,
                self.nwalkers,
                ndims,  # assumes ndim_max
                like_mix,
                priors,
                tempering_kwargs={"ntemps": self.ntemps},
                nbranches=len(branch_names),
                nleaves_max=nleaves_max,
                nleaves_min=nleaves_min,
                moves=_tmp_move,  # setup_info_all.in_model_moves_input,
                rj_moves=None,  # setup_info_all.rj_moves_input,
                kwargs=None,
                backend=backend,
                vectorize=True,
                periodic=periodic,
                branch_names=branch_names,
                # update_fn=recipe,  # stop_converge_mix,
                # update_iterations=1,  # TODO: change this?
                provide_groups=True,
                provide_supplemental=True,
                track_moves=False,
                stopping_fn=self.recipe,
                stopping_iterations=1,
            )
            _tmp_move.temperature_control.swaps_accepted = np.zeros((self.ntemps, self.nwalkers), dtype=int)
            
            state.log_like[:] = acs.likelihood(sum_instead_of_trapz=False)[None, :]
            state.log_prior = np.zeros_like(state.log_like)  # sampler_mix.compute_log_prior(state.branches_coords, inds=state.branches_inds, supps=supps)
            self.recipe.setup_first_recipe_step(sampler_mix.iteration, state, sampler_mix)
            sampler_mix.run_mcmc(state, 100, thin_by=1, progress=True, store=True)
            self.comm.send({"finish_run": True}, dest=self.results_rank)

        elif self.rank == self.results_rank:
            save_to_backend_asynchronously_and_plot(backend, self.comm, self.main_rank, self.head_rank, self.curr.general_info["plot_iter"], self.curr.general_info["backup_iter"])

        else:
            info = self.comm.recv(source=self.main_rank)
            if isinstance(info, dict):
                launch_rank = info["rank"]
                assert launch_rank == self.rank
                launch_function = info["function"]
                launch_function(self.comm, self.curr, self.main_rank, info["gpus"], info["class_rank_list"])

            print(f"Process {self.rank} finished.")
            
            
class GlobalFitSegment(ABC):
    def __init__(self, comm, copy_settings_file=False):
        self.comm = comm
        # self.base_settings = get_global_fit_settings(copy_settings_file=copy_settings_file)
        settings = deepcopy(self.base_settings)
        self.gpus = settings["general"]["gpus"]
        self.adjust_settings(settings)

        self.current_info = CurrentInfoGlobalFit(settings)

    def adjust_settings(self, settings):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
        

class MPIControlGlobalFit:
    def __init__(self, current_info, comm, gpus, run_results_update=True, **run_results_update_kwargs):

        ranks = np.arange(comm.Get_size())

        self.rank = comm.Get_rank()
        self.run_results_update = run_results_update
        self.run_results_update_kwargs = run_results_update_kwargs
        
        assert len(gpus) >= 2
        
        self.current_info = current_info
        self.comm = comm
        self.gpus = gpus

        self.head_rank = self.current_info.rank_info["head_rank"]
        self.main_rank = self.current_info.rank_info["main_rank"]
        self.other_ranks = list(range(comm.Size()))[2:]

        self.main_gpu = self.current_info.gpu_assignments["main_gpu"]
        self.other_gpus = self.current_info.gpu_assignments["other_gpus"]
        
        # assign results saving rank
        self.run_results_rank = self.other_ranks.pop(0)
        
        for gpu_assign in self.all_gpu_assignments:
            assert gpu_assign in self.gpus

    def run_global_fit(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):
        if self.rank == self.head_rank:
            # send to refit 
            print("sending data")

            self.share_start_info(run_psd=run_psd, run_mbhs=run_mbhs, run_gbs_pe=run_gbs_pe, run_gbs_search=run_gbs_search)

            print("done sending data")

            # populate which runs are going
            runs_going = ["main"]
            
            if len(runs_going) == 0:
                raise ValueError("No runs are going to be launched because all options for runs are False.")

            while len(runs_going) > 0:
                time.sleep(1)

        elif self.rank == self.main_rank:
            fin = run_main_function(self.main_gpu, self.comm, self.head_rank, self.run_results_rank)

        elif self.run_results_update and self.rank == self.run_results_rank:
            save_to_backend_asynchronously_and_plot(self.current_info.backend, self.comm, self.main_rank, self.head_rank, self.current_info.general_info["plot_iter"], self.current_info.general_info["backup_iter"])
            
def run_gf_progression(global_fit_progression, comm, head_rank, status_file):
    rank = comm.Get_rank()
    
    for g_i, global_fit_segment in enumerate(global_fit_progression):
        # check if this segment has completed
        if os.path.exists(status_file):
            with open(status_file, "r") as fp:
                lines = fp.readlines()
                if global_fit_segment["name"] + "\n" in lines:
                    print(f"continue {global_fit_segment['name']}")
                    continue

        class_name = global_fit_segment["name"]
        if rank == head_rank:
            print(f"\n\nStarting {class_name}\n\n")
            st = time.perf_counter()
        print(f"\n\nStarting {class_name}, {rank}\n\n")

        segment = global_fit_segment["segment"](*global_fit_segment["args"], **global_fit_segment["kwargs"]) 
        
        print("start", g_i, rank)
        segment.run()
        print("finished:", rank)
        comm.Barrier()
        print("end", g_i, rank)
        if rank == head_rank:
            class_name = global_fit_segment["name"]
            et = time.perf_counter()
            print(f"\n\nEnding {class_name}({et - st} sec)\n\n")
            with open(segment.current_info.general_info["file_information"]["status_file"], "a") as fp:
                fp.write(global_fit_segment["name"] + "\n")