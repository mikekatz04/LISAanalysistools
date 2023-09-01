from mpi4py import MPI
from copy import deepcopy

import time
# from full_band_global_fit_settings import *
from global_fit_input.global_fit_settings import get_global_fit_settings
import os

from lisatools.globalfit.run import CurrentInfoGlobalFit, MPIControlGlobalFit

import pickle

from bbhx.utils.constants import *
from bbhx.utils.transform import *

from eryn.backends import HDFBackend


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    gpus = [4, 5, 6, 7]
    settings = get_global_fit_settings()

    np.random.seed(settings["general"]["random_seed"])
    
    current_info = CurrentInfoGlobalFit(settings)
    mpi_controller = MPIControlGlobalFit(current_info, comm, gpus)
    mpi_controller.run_global()

