from mpi4py import MPI
from global_fit_input.global_fit_settings import get_global_fit_settings
from lisatools.globalfit.mbhsearch import ParallelMBHSearchControl

if __name__ == "__main__":
    # set parameters'
    comm = MPI.COMM_WORLD
    
    settings = get_global_fit_settings()
    gpus = [3, 7]
    # print("start", fp)
    para_mbh_search = ParallelMBHSearchControl(settings, comm, gpus, head_rank=0, max_num_per_gpu=2, verbose=False)

    # para_mbh_search.run_parallel_mbh_search()  # testing_time_split=5)
    para_mbh_search.prune_via_matching()
    
    # print("end", fp)
    # frequencies to interpolate to