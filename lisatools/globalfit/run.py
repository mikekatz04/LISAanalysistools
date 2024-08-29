import numpy as np
from copy import deepcopy

from ..sensitivity import get_sensitivity
from .hdfbackend import HDFBackend as GBHDFBackend
from .state import State
from bbhx.waveformbuild import BBHWaveformFD

from .mbhsearch import ParallelMBHSearchControl
from .galaxyglobal import run_gb_pe, run_gb_bulk_search, fit_each_leaf
from .psdglobal import run_psd_pe
from .mbhglobal import run_mbh_pe

from lisatools.sampling.stopping import SearchConvergeStopping, MPICommunicateStopping
from lisatools.globalfit.plot import RunResultsProduction
from lisatools.globalfit.hdfbackend import save_to_backend_asynchronously_and_plot

from gbgpu.gbgpu import GBGPU

from eryn.backends import HDFBackend

import time
import os
import pickle

from abc import ABC

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    pass

from global_fit_input.global_fit_settings import get_global_fit_settings


class CurrentInfoGlobalFit:
    def __init__(self, settings, get_last_state_info=True):

        self.settings_dict = settings
        self.current_info = deepcopy(settings)

        self.backend = GBHDFBackend(settings["general"]["file_information"]["fp_main"])

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

        self.current_info["gb"]["search_gmm_info"] = gb_search_proposal_gmm_info
        self.current_info["gb"]["refit_gmm_info"] = gb_refit_proposal_gmm_info
    
    def initialize_mbh_state_from_search(self, mbh_output_point_info):
        output_points_pruned = np.asarray(mbh_output_point_info["output_points_pruned"]).transpose(1, 0, 2)
        coords = np.zeros((self.current_info["gb"]["pe_info"]["ntemps"], self.current_info["gb"]["pe_info"]["nwalkers"], output_points_pruned.shape[1], self.current_info["mbh"]["pe_info"]["ndim"]))
        assert output_points_pruned.shape[0] >= self.current_info["mbh"]["pe_info"]["nwalkers"]
        
        coords[:] = output_points_pruned[None, :self.current_info["mbh"]["pe_info"]["nwalkers"]]
        self.current_info["mbh"]["mbh_init_points"] = coords.copy()
    
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
    def gb_info(self):
        return self.current_info["gb"]

    @property
    def mbh_info(self):
        return self.current_info["mbh"]

    @property
    def psd_info(self):
        return self.current_info["psd"]

    @property
    def general_info(self):
        return self.current_info["general"]

    @property
    def rank_info(self):
        return self.current_info["rank_info"]

    @property
    def gpu_assignments(self):
        return self.current_info["gpu_assignments"]


class GlobalFitSegment(ABC):
    def __init__(self, comm, copy_settings_file=False):
        self.comm = comm
        self.base_settings = get_global_fit_settings(copy_settings_file=copy_settings_file)
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