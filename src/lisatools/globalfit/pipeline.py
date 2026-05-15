"""Pre-built :class:`GlobalFitSegment` pipelines used to assemble end-to-end runs."""

import os
import pickle
import time
from copy import deepcopy

import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
from eryn.backends import HDFBackend
from gbgpu.gbgpu import GBGPU

from lisatools.sampling.stopping import (
    GBBandLogLConvergeStopping,
    MPICommunicateStopping,
    NLeavesSearchStopping,
    SearchConvergeStopping,
)

from ..sensitivity import get_sensitivity
from .galaxyglobal import fit_each_leaf, run_gb_bulk_search, run_gb_pe
from .hdfbackend import HDFBackend as GBHDFBackend
from .mbhglobal import run_mbh_pe
from .mbhsearch import ParallelMBHSearchControl
from .psdglobal import run_psd_pe
from .run import *
from .state import State


class MBHSearchSegment(GlobalFitSegment):
    """Segment that runs the standalone parallel MBH search.

    Args:
        head_rank: MPI rank that drives the search controller.
    """

    def __init__(self, *args, head_rank=0, **kwargs):

        super().__init__(*args, **kwargs)

        self.head_rank = head_rank
        self.para_mbh_search = ParallelMBHSearchControl(
            self.current_info.settings_dict,
            self.comm,
            self.gpus,
            head_rank=self.head_rank,
            max_num_per_gpu=self.current_info.settings_dict["mbh"]["search_info"][
                "max_num_per_gpu"
            ],
            verbose=self.current_info.settings_dict["mbh"]["search_info"]["verbose"],
        )

    def adjust_settings(self, settings):
        """No segment-level overrides applied for the MBH search."""
        pass

    def run(self):
        """Launch :meth:`ParallelMBHSearchControl.run_parallel_mbh_search`."""
        self.para_mbh_search.run_parallel_mbh_search()


class InitialPSDSearch(GlobalFitSegment):
    """Segment that searches for the initial PSD before any source PE."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(
            self.current_info, self.comm, self.gpus, run_results_update=False
        )

    def adjust_settings(self, settings):
        """Configure PSD-only stopping/convergence for the initial search."""
        settings["psd"]["pe_info"]["update_iterations"] = -1
        settings["psd"]["pe_info"]["stopping_iterations"] = 4
        settings["psd"]["pe_info"]["stopping_function"] = SearchConvergeStopping(
            **settings["psd"]["pe_info"]["stop_kwargs"]
        )

    def run(self):
        """Run the global fit with only the PSD branch active."""
        self.mpi_controller.run_global_fit(
            run_psd=True, run_mbhs=False, run_gbs_pe=False, run_gbs_search=False
        )


class InitialMBHMixSegment(GlobalFitSegment):
    """Segment that mixes PSD and MBH PE prior to galactic-binary search."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(
            self.current_info, self.comm, self.gpus, run_results_update=False
        )

    def adjust_settings(self, settings):
        """No segment-level setting overrides for the MBH-mix segment."""
        pass

    def run(self):
        """Run PSD + MBH PE; PSD waits on the MBH stopper via MPI."""
        stopper_rank = self.mpi_controller.mbh_rank
        other_ranks = [self.mpi_controller.psd_rank]

        # had to go after initialization of mpi because it needs the ranks
        stop_fn = SearchConvergeStopping(**self.current_info.mbh_info["pe_info"]["stop_kwargs"])
        self.current_info.mbh_info["pe_info"]["stopping_function"] = MPICommunicateStopping(
            stopper_rank, other_ranks, stop_fn=stop_fn
        )

        self.current_info.psd_info["pe_info"]["stopping_function"] = MPICommunicateStopping(
            stopper_rank, other_ranks, stop_fn=None
        )

        self.mpi_controller.run_global_fit(
            run_psd=True, run_mbhs=True, run_gbs_pe=False, run_gbs_search=False
        )


class InitialGBSearchSegment(GlobalFitSegment):
    """Segment that runs the initial galactic-binary search alongside PSD/MBH PE.

    Args:
        snr_lim: SNR threshold passed into the GB search settings.
    """

    def __init__(self, *args, snr_lim=10.0, **kwargs):
        self.snr_lim = snr_lim
        super().__init__(*args, **kwargs)
        self.mpi_controller = MPIControlGlobalFit(
            self.current_info, self.comm, self.gpus, run_results_update=True
        )

    def adjust_settings(self, settings):
        """Override settings with GB-search-specific RJ fractions and rank/GPU layout."""
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
        """Run the configured global fit; the GB PE rank is the MPI stopper."""
        stopper_rank = self.mpi_controller.gb_pe_rank
        other_ranks = [
            self.mpi_controller.psd_rank,
            self.mpi_controller.mbh_rank,
        ] + self.mpi_controller.gb_search_rank  # gb_search_rank is a list

        print(f"Stopper {stopper_rank}, other: {other_ranks}")
        # had to go after initialization of mpi because it needs the ranks
        stop_fn = NLeavesSearchStopping(
            **self.current_info.gb_info["pe_info"]["stop_search_kwargs"]
        )
        self.current_info.gb_info["pe_info"]["stopping_function"] = MPICommunicateStopping(
            stopper_rank, other_ranks, stop_fn=stop_fn
        )
        self.current_info.psd_info["pe_info"]["stopping_function"] = MPICommunicateStopping(
            stopper_rank, other_ranks, stop_fn=None
        )
        self.current_info.gb_info["search_info"]["stopping_function"] = MPICommunicateStopping(
            stopper_rank, other_ranks, stop_fn=None
        )
        self.current_info.mbh_info["pe_info"]["stopping_function"] = MPICommunicateStopping(
            stopper_rank, other_ranks, stop_fn=None
        )

        self.mpi_controller.run_global_fit(
            run_psd=run_psd,
            run_mbhs=run_mbhs,
            run_gbs_pe=run_gbs_pe,
            run_gbs_search=run_gbs_search,
        )


class FullPESegment(GlobalFitSegment):
    """Segment that runs the full multi-source PE without imposing extra stopping logic."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mpi_controller = MPIControlGlobalFit(
            self.current_info, self.comm, self.gpus, run_results_update=True
        )

    def adjust_settings(self, settings):
        """No settings overrides applied for the full PE segment."""
        pass

    def run(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):
        """Run the full global fit with optional per-component toggles.

        Args:
            run_psd: Toggle PSD branch.
            run_mbhs: Toggle MBH branch.
            run_gbs_pe: Toggle GB parameter-estimation branch.
            run_gbs_search: Toggle GB search branch.
        """

        stopper_rank = self.mpi_controller.main_rank

        # had to go after initialization of mpi because it needs the ranks
        # stop_fn = GBBandLogLConvergeStopping(self.current_info.general_info["fd"], self.current_info.gb_info["band_edges"], **self.current_info.gb_info["pe_info"]["stop_kwargs"])
        # self.current_info.gb_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=stop_fn)
        # self.current_info.psd_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        # self.current_info.gb_info["search_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)
        # self.current_info.mbh_info["pe_info"]["stopping_function"] = MPICommunicateStopping(stopper_rank, other_ranks, stop_fn=None)

        self.mpi_controller.run_global_fit(
            run_psd=run_psd,
            run_mbhs=run_mbhs,
            run_gbs_pe=run_gbs_pe,
            run_gbs_search=run_gbs_search,
        )
