
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

from lisatools.sampling.stopping import SearchConvergeStopping, MPICommunicateStopping

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

        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus)

    def adjust_settings(self, settings):
        settings["psd"]["pe_info"]["update_iterations"] = -1
        settings["psd"]["search_info"]["run_search"] = True
        settings["psd"]["search_info"]["stopping_iterations"] = 4
        settings["psd"]["search_info"]["stopping_function"] = SearchConvergeStopping(**settings["psd"]["search_info"]["stop_kwargs"])

    def run(self):

        self.mpi_controller.run_global_fit(run_psd=True, run_mbhs=False, run_gbs_pe=False, run_gbs_search=False)


class InitialMBHMixSegment(GlobalFitSegment):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus)

    def adjust_settings(self, settings):
        settings["mbh"]["search_info"]["stop_kwargs"] = dict(
            n_iters=3,
            diff=0.01,
            verbose=True
        )
        settings["mbh"]["search_info"]["thin_by"] = 5
        settings["mbh"]["search_info"]["stopping_iterations"] = 1
        settings["mbh"]["search_info"]["run_search"] = True
        
        settings["psd"]["search_info"]["run_search"] = True
        settings["psd"]["search_info"]["stopping_iterations"] = 4

    def run(self):

        stopper_rank = self.mpi_controller.mbh_rank
        other_ranks = [self.mpi_controller.psd_rank]

        # had to go after initialization of mpi because it needs the ranks
        stop_fn = SearchConvergeStopping(**self.current_info.mbh_info["search_info"]["stop_kwargs"])
        self.current_info.mbh_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=stop_fn)

        self.current_info.psd_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        
        self.mpi_controller.run_global_fit(run_psd=True, run_mbhs=True, run_gbs_pe=False, run_gbs_search=False)


class InitialGBSearchSegment(GlobalFitSegment):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus)

    def adjust_settings(self, settings):
        
        #  settings["gb"]["search_info"]["snr_lim"] = 9.0
        settings["gb"]["search_info"]["snr_lim"] = 7.0
        settings["gb"]["pe_info"]["use_prior_removal"] = True
        settings["gb"]["pe_info"]["rj_refit_fraction"] = 0.1
        settings["gb"]["pe_info"]["rj_search_fraction"] = 0.7
        settings["gb"]["pe_info"]["rj_prior_fraction"] = 0.2
        settings["gb"]["pe_info"]["update_iterations"] = 1
        settings["gb"]["pe_info"]["stop_kwargs"] = dict(
            n_iters=5,
            diff=0.01,
            verbose=True
        )
        settings["gb"]["pe_info"]["stopping_iterations"] = 1
        settings["gb"]["pe_info"]["run_search"] = True

        settings["gb"]["search_info"]["stopping_iterations"] = 1
        settings["gb"]["search_info"]["run_search"] = True
        settings["gb"]["search_info"]["snr_lim"] = 9.0

        settings["mbh"]["search_info"]["thin_by"] = 5
        settings["mbh"]["search_info"]["stopping_iterations"] = 1
        settings["mbh"]["search_info"]["run_search"] = True
        
        settings["psd"]["search_info"]["run_search"] = True
        settings["psd"]["search_info"]["stopping_iterations"] = 4
        
    def run(self):

        stopper_rank = self.mpi_controller.gb_pe_rank
        other_ranks = [
            self.mpi_controller.psd_rank,
            self.mpi_controller.mbh_rank,
            self.mpi_controller.gb_search_rank
        ]

        # had to go after initialization of mpi because it needs the ranks
        stop_fn = SearchConvergeStopping(**self.current_info.gb_info["pe_info"]["stop_kwargs"])
        self.current_info.gb_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=stop_fn)
        self.current_info.psd_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        self.current_info.gb_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        self.current_info.mbh_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)

        self.mpi_controller.run_global_fit(run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True)


class FullPESegment(GlobalFitSegment):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(self.current_info, self.comm, self.gpus)

    def adjust_settings(self, settings):
        pass

    def run(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):

        self.mpi_controller.run_global_fit(run_psd=run_psd, run_mbhs=run_mbhs, run_gbs_pe=run_gbs_pe, run_gbs_search=run_gbs_search)
