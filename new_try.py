import numpy as np
from mpi4py import MPI
import os
import warnings
from copy import deepcopy

from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
from global_fit_input.global_fit_settings import get_global_fit_settings

if __name__ == "__main__":
    curr = CurrentInfoGlobalFit(get_global_fit_settings())
    gf_branch_information = curr.current_info["gf_branch_information"]
    gf = GlobalFit(gf_branch_information, curr, MPI.COMM_WORLD)
    gf.run_global_fit()