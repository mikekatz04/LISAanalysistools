from mpi4py import MPI
from copy import deepcopy
from simultaneous_gb_pe import run_gb_pe
from bulk_gb_search import run_gb_bulk_search, fit_each_leaf
import time
# from full_band_global_fit_settings import *
from lisatools.globalfit.hdfbackend import HDFBackend as GBHDFBackend
from global_fit_input.global_fit_settings import get_global_fit_settings
import os

import pickle

from bbhx.utils.constants import *
from bbhx.utils.transform import *

from eryn.backends import HDFBackend


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if __name__ == "__main__":

    settings = get_global_fit_settings()
    print(rank)

    np.random.seed(settings["general"]["random_seed"])
    
    head_rank = 1
    gb_pe_rank = 0
    gb_search_rank = 2

    current_info = deepcopy(settings)

    # assert comm.size > 3
    gmm_ranks = [i for i in range(3, comm.size)]
    print(gmm_ranks)
    if rank == head_rank:
        gb_reader = GBHDFBackend(settings["general"]["file_information"]["fp_gb_pe"])
        gb_last_sample = gb_reader.get_last_sample()

        # with open("save_state_new_gmm.pickle", "rb") as fp:
        #     gb_last_sample = pickle.load(fp)
        #     gb_last_sample.log_prior = np.zeros_like(gb_last_sample.log_like)
        #     gb_last_sample.branches["gb_fixed"].coords[~gb_last_sample.branches["gb_fixed"].inds] = np.nan
        # breakpoint()

        mbh_reader = HDFBackend(settings["general"]["file_information"]["fp_mbh_pe"])
        mbh_last_sample = mbh_reader.get_last_sample()

        psd_reader = HDFBackend(settings["general"]["file_information"]["fp_psd_pe"])
        psd_last_sample = psd_reader.get_last_sample()

        current_info["psd"]["reader"] = psd_reader
        current_info["psd"]["cc_A"] = psd_last_sample.branches["psd"].coords[0, :, 0, :2]
        current_info["psd"]["cc_E"] = psd_last_sample.branches["psd"].coords[0, :, 0, 2:]
        current_info["psd"]["cc_foreground_params"] = psd_last_sample.branches["galfor"].coords[0, :, 0, :]
        current_info["psd"]["cc_ll"] = psd_last_sample.log_like[0]
        current_info["psd"]["last_state"] = psd_last_sample
        
        current_info["mbh"]["reader"] = mbh_reader
        current_info["mbh"]["cc_params"] = mbh_last_sample.branches["mbh"].coords[0, :, :]
        current_info["mbh"]["cc_ll"] = mbh_last_sample.log_like[0]
        current_info["mbh"]["last_state"] = mbh_last_sample

        # TODO: add all to GMM
        current_info["gb"]["reader"] = gb_reader
        current_info["gb"]["cc_params"] = gb_last_sample.branches["gb_fixed"].coords[0, :, :]
        current_info["gb"]["cc_inds"] = gb_last_sample.branches["gb_fixed"].inds[0, :, :]
        current_info["gb"]["cc_ll"] = gb_last_sample.log_like[0]
        current_info["gb"]["cc_lp"] = gb_last_sample.log_prior[0]
        current_info["gb"]["last_state"] = gb_last_sample

        # gmm info
        if os.path.exists(settings["general"]["file_information"]["fp_gb_gmm_info"]):
            with open(settings["general"]["file_information"]["fp_gb_gmm_info"], "rb") as fp_gmm:
                gmm_info_dict = pickle.load(fp_gmm)
                gb_search_proposal_gmm_info = gmm_info_dict["search"]
                gb_sample_refit_gmm_info = gmm_info_dict["refit"]
        else:
            gb_search_proposal_gmm_info = None
            gb_sample_refit_gmm_info = None

        # send to refit 
        print("sending data")

        comm.send(current_info, dest=gb_pe_rank, tag=255)
        if gb_search_proposal_gmm_info is not None or gb_sample_refit_gmm_info is not None:
            comm.send({"search_gmm": gb_search_proposal_gmm_info, "refit_gmm": gb_sample_refit_gmm_info}, dest=gb_pe_rank, tag=55)

        update_refit = False
        have_started_refit = False
        while True:
            time.sleep(1)

            # print("checking again", update_refit)
            # check if refit is ready
            refit_check = comm.irecv(source=gb_search_rank, tag=20)
            if refit_check.get_status() and update_refit:
                refit_dict = refit_check.wait()
                print("CHECK", refit_dict)
                if "send" in refit_dict and refit_dict["send"]:
                    comm.send(current_info, dest=gb_search_rank, tag=27)

                if "receive" in refit_dict and refit_dict["receive"]:
                    update_refit = False
                    gmm_info = comm.recv(source=gb_search_rank, tag=29)

                    if "search" in gmm_info and gmm_info["search"] is not None:
                        if gb_search_proposal_gmm_info is None:
                            gb_search_proposal_gmm_info = gmm_info["search"]
                        else:
                            # search keeps adding to what it already has
                            gb_search_proposal_gmm_info = [tmp1 + tmp2 for tmp1, tmp2 in zip(gb_search_proposal_gmm_info, gmm_info["search"])]
                        
                    if "sample_refit" in gmm_info and gmm_info["sample_refit"] is not None:
                        # refit always just replaces it
                        gb_sample_refit_gmm_info = gmm_info["sample_refit"]
                        
                    with open(settings["general"]["file_information"]["fp_gb_gmm_info"], "wb") as fp_gmm:
                        gmm_info_dict = {"search": gb_search_proposal_gmm_info, "refit": gb_sample_refit_gmm_info}
                        pickle.dump(gmm_info_dict, fp_gmm, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                refit_check.cancel()

            # check for update from GBs
            gb_check = comm.irecv(source=gb_pe_rank, tag=50)
            if gb_check.get_status():
                gb_req = gb_check.wait()
                print(gb_req)
                if "receive" in gb_req and gb_req["receive"]:
                    gb_dict = comm.recv(source=gb_pe_rank, tag=58)
                
                    if "gb_update" in gb_dict:
                        for key in ["cc_params", "cc_inds", "cc_ll", "cc_lp", "last_state"]:
                            if key == "last_state":
                                current_info["gb"][key] = gb_dict["update"][key]
                            else:
                                current_info["gb"][key][:] = gb_dict["update"][key][:]
                    update_refit = True

                if "send" in gb_req and gb_req["send"]:
                    comm.send(current_info, dest=gb_pe_rank, tag=51)
                    comm.send({"search_gmm": gb_search_proposal_gmm_info, "refit_gmm": gb_sample_refit_gmm_info}, dest=gb_pe_rank, tag=52)

                if "start_refit" in gb_req and gb_req["start_refit"]:
                    update_refit = True
                    if not have_started_refit:
                        comm.send(True, dest=gb_search_rank, tag=1010)
                        have_started_refit = True
                   
            else:
                gb_check.cancel()

       
    elif rank == gb_pe_rank:
        gpu = 5
        fin = run_gb_pe(5, comm, head_rank)

    elif rank == gb_search_rank:
        gpu = 4
        comm_info = {"process_ranks_for_fit": gmm_ranks}
        refit_check = comm.recv(source=head_rank, tag=1010)
        # block until it is told to go
        # fin = run_gb_bulk_search(gpu, comm, comm_info, head_rank)

    elif rank in gmm_ranks:
        rec_tag = int(str(rank) + "300")
        send_tag = int(str(rank) + "400")
        fit_each_leaf(rank, gb_search_rank, rec_tag, send_tag, comm)

