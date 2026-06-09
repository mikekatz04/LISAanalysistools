print("START")
from mpi4py import MPI
print("END")
from copy import deepcopy

import time
# from full_band_global_fit_settings import *
import os

from lisatools.globalfit.pipeline import *
from lisatools.globalfit.run import run_gf_progression

from copy import deepcopy
import pickle

from bbhx.utils.constants import *
from bbhx.utils.transform import *

from eryn.backends import HDFBackend
from global_fit_input.global_fit_settings import get_global_fit_settings

if __name__ == "__main__":

    settings = get_global_fit_settings()
    status_file = settings["general"]["file_information"]["status_file"] 
    # TODO: add command line args
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    head_rank = 0

    global_fit_progression = [
        {"name": "initial psd search", "segment": InitialPSDSearch, "args": (comm,), "kwargs": dict(copy_settings_file=True)},
        {"name": "mbhb search", "segment": MBHSearchSegment, "args": (comm,), "kwargs": dict(head_rank=head_rank, copy_settings_file=True)},
        {"name": "mbhb + psd mix", "segment": InitialMBHMixSegment, "args": (comm,), "kwargs": dict(copy_settings_file=True)},
        # {"name": "gb search 1", "segment": InitialGBSearchSegment, "args": (comm,), "kwargs": dict(snr_lim=10.0)},
        {"name": "gb search 2", "segment": InitialGBSearchSegment, "args": (comm,), "kwargs": dict(snr_lim=7.0, copy_settings_file=True)},
        {"name": "all pe", "segment": FullPESegment, "args": (comm,), "kwargs": dict(copy_settings_file=True)},
    ]
    
    # debug_psd_search = InitialPSDSearch(comm)
    # debug_psd_search.run()

    # debug_seg = FullPESegment(comm)
    # debug_seg.run(run_psd=False, run_gbs_pe=True, run_gbs_search=False, run_mbhs=False)

    # debug_search = MBHSearchSegment(comm, head_rank=head_rank)
    # debug_search.run()

    # debug_mix = InitialMBHMixSegment(comm)
    # debug_mix.run()

    # debug_gb_search = InitialGBSearchSegment(comm)
    # debug_gb_search.run(run_psd=True, run_gbs_pe=False, run_gbs_search=False, run_mbhs=False)

    # if rank == head_rank:
    #     debug_search.para_mbh_search.run_parallel_mbh_search(testing_time_split=7)
    run_gf_progression(global_fit_progression, comm, head_rank, status_file)
     

