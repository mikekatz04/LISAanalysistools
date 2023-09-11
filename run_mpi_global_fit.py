from mpi4py import MPI
from copy import deepcopy

import time
# from full_band_global_fit_settings import *
import os

from lisatools.globalfit.pipeline import *

from copy import deepcopy
import pickle

from bbhx.utils.constants import *
from bbhx.utils.transform import *

from eryn.backends import HDFBackend


if __name__ == "__main__":

    # TODO: add command line args
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    head_rank = 0

    global_fit_progression = [
        {"segment": InitialPSDSearch, "args": (comm,), "kwargs": dict()},
        {"segment": MBHSearchSegment, "args": (comm,), "kwargs": dict(head_rank=head_rank)},
        {"segment": InitialMBHMixSegment, "args": (comm,), "kwargs": dict()},
        {"segment": InitialGBSearchSegment, "args": (comm,), "kwargs": dict()},
        {"segment": FullPESegment, "args": (comm,), "kwargs": dict()},
    ]
    
    # debug_psd_search = InitialPSDSearch(comm)
    # debug_psd_search.run()

    debug_seg = FullPESegment(comm)
    debug_seg.run(run_psd=True, run_gbs_pe=True, run_gbs_search=True, run_mbhs=True)

    # debug_search = MBHSearchSegment(comm, head_rank=head_rank)
    # debug_search.run()

    # debug_mix = InitialMBHMixSegment(comm)
    # debug_mix.run()

    # debug_gb_search = InitialGBSearchSegment(comm)
    # debug_gb_search.run()

    # if rank == head_rank:
    #     debug_search.para_mbh_search.run_parallel_mbh_search(testing_time_split=7)

    # for global_fit_segment in global_fit_progression:
    #     segment = global_fit_segment["segment"](*global_fit_segment["args"], **global_fit_segment["kwargs"]) 
    #     segment.run()
    

