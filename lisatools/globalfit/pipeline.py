
import numpy as np
from copy import deepcopy

from ..sensitivity import get_sensitivity
from .hdfbackend import HDFBackend as GBHDFBackend
from .state import State
from bbhx.waveformbuild import BBHWaveformFD

from .run import *
from .mbhsearch import ParallelMBHSearchControl
from .galaxyglobal import run_gb_pe, run_gb_bulk_search, fit_each_leaf
from .psdglobal import run_psd_pe
from .mbhglobal import run_mbh_pe

from lisatools.sampling.stopping import SearchConvergeStopping, MPICommunicateStopping, NLeavesSearchStopping, GBBandLogLConvergeStopping

from gbgpu.gbgpu import GBGPU

from eryn.backends import HDFBackend

import time
import os
import pickle



class MBHSearchSegment(GlobalFitSegment):
    def __init__(self, *args, head_rank=0, **kwargs):

        super().__init__(*args, **kwargs)

        self.head_rank = head_rank
        self.para_mbh_search = ParallelMBHSearchControl(self.current_info.settings_dict, self.comm, self.gpus, head_rank=self.head_rank, max_num_per_gpu=self.current_info.settings_dict["mbh"]["search_info"]["max_num_per_gpu"], verbose=self.current_info.settings_dict["mbh"]["search_info"]["verbose"])

    def adjust_settings(self, settings):
        pass

    def run(self):
        self.para_mbh_search.run_parallel_mbh_search() 

class InitialPSDSearch(GlobalFitSegment):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus, run_results_update=False)

    def adjust_settings(self, settings):
        settings["psd"]["pe_info"]["update_iterations"] = -1
        settings["psd"]["pe_info"]["stopping_iterations"] = 4
        settings["psd"]["pe_info"]["stopping_function"] = SearchConvergeStopping(**settings["psd"]["pe_info"]["stop_kwargs"])

    def run(self):

        self.mpi_controller.run_global_fit(run_psd=True, run_mbhs=False, run_gbs_pe=False, run_gbs_search=False)


class InitialMBHMixSegment(GlobalFitSegment):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus, run_results_update=False)

    def adjust_settings(self, settings):
        pass
        
    def run(self):

        stopper_rank = self.mpi_controller.mbh_rank
        other_ranks = [self.mpi_controller.psd_rank]

        # had to go after initialization of mpi because it needs the ranks
        stop_fn = SearchConvergeStopping(**self.current_info.mbh_info["pe_info"]["stop_kwargs"])
        self.current_info.mbh_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=stop_fn)

        self.current_info.psd_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        
        self.mpi_controller.run_global_fit(run_psd=True, run_mbhs=True, run_gbs_pe=False, run_gbs_search=False)


class InitialGBSearchSegment(GlobalFitSegment):
    def __init__(self, *args, snr_lim=10.0, **kwargs):
        self.snr_lim = snr_lim
        super().__init__(*args, **kwargs)
        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus, run_results_update=True)

    def adjust_settings(self, settings):
        
        settings["gb"]["pe_info"]["use_prior_removal"] = True
        settings["gb"]["pe_info"]["rj_refit_fraction"] = 0.1
        settings["gb"]["pe_info"]["rj_search_fraction"] = 0.7
        settings["gb"]["pe_info"]["rj_prior_fraction"] = 0.2
        settings["gb"]["pe_info"]["update_iterations"] = 1
        settings["gb"]["pe_info"]["stopping_iterations"] = 1
        settings["gb"]["pe_info"]["rj_phase_maximize"] = True
        settings["gb"]["pe_info"]["start_resample_iter"] = 100
        settings["gb"]["pe_info"]["group_proposal_kwargs"]["num_repeat_proposals"] = 50
        # settings["gb"]["pe_info"]["thin_by"] = 1

        settings["gb"]["search_info"]["stopping_iterations"] = 1
        settings["gb"]["search_info"]["snr_lim"] = self.snr_lim

        settings["mbh"]["pe_info"]["stopping_iterations"] = 1
        settings["psd"]["pe_info"]["stopping_iterations"] = 4
        
        settings["gpu_assignments"]["gb_search_gpu"] = settings["general"]["gpus"][1:3]
        settings["gpu_assignments"]["gb_pe_gpu"] = settings["general"]["gpus"][0]
        settings["gpu_assignments"]["psd_gpu"] = settings["general"]["gpus"][3]
        settings["gpu_assignments"]["mbh_gpu"] = settings["general"]["gpus"][3]

        settings["rank_info"]["gb_search_rank"] = [2, 3, 4]
        settings["rank_info"]["gb_pe_rank"] = 1
        settings["rank_info"]["psd_rank"] = 5
        settings["rank_info"]["mbh_rank"] = 6
        
    def run(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):

        stopper_rank = self.mpi_controller.gb_pe_rank
        other_ranks = [
            self.mpi_controller.psd_rank,
            self.mpi_controller.mbh_rank,
        ] + self.mpi_controller.gb_search_rank  # gb_search_rank is a list

        print(f"Stopper {stopper_rank}, other: {other_ranks}")
        # had to go after initialization of mpi because it needs the ranks
        stop_fn = NLeavesSearchStopping(**self.current_info.gb_info["pe_info"]["stop_search_kwargs"])
        self.current_info.gb_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=stop_fn)
        self.current_info.psd_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        self.current_info.gb_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        self.current_info.mbh_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)

        self.mpi_controller.run_global_fit(run_psd=run_psd, run_mbhs=run_mbhs, run_gbs_pe=run_gbs_pe, run_gbs_search=run_gbs_search)


class FullPESegment(GlobalFitSegment):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus, run_results_update=True)

    def adjust_settings(self, settings):
        pass

    def run(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):
        
        stopper_rank = self.mpi_controller.gb_pe_rank
        other_ranks = [
            self.mpi_controller.psd_rank,
            self.mpi_controller.mbh_rank,
        ] + self.mpi_controller.gb_search_rank

        # had to go after initialization of mpi because it needs the ranks
        stop_fn = GBBandLogLConvergeStopping(self.current_info.general_info["fd"], self.current_info.gb_info["band_edges"], **self.current_info.gb_info["pe_info"]["stop_kwargs"])
        self.current_info.gb_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=stop_fn)
        self.current_info.psd_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        self.current_info.gb_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        self.current_info.mbh_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)

        self.mpi_controller.run_global_fit(run_psd=run_psd, run_mbhs=run_mbhs, run_gbs_pe=run_gbs_pe, run_gbs_search=run_gbs_search)
