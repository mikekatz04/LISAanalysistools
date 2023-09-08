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
    def __init__(self, settings):

        self.settings_dict = settings
        self.current_info = deepcopy(settings)

        gb_reader = GBHDFBackend(settings["general"]["file_information"]["fp_gb_pe"])
        # with open("save_state_new_gmm.pickle", "rb") as fp:
        #     gb_last_sample = pickle.load(fp)
        #     gb_last_sample.log_prior = np.zeros_like(gb_last_sample.log_like)
        #     gb_last_sample.branches["gb_fixed"].coords[~gb_last_sample.branches["gb_fixed"].inds] = np.nan
        # # breakpoint()

        psd_reader = HDFBackend(settings["general"]["file_information"]["fp_psd_pe"])
        self.current_info["psd"]["reader"] = psd_reader
        
        if os.path.exists(psd_reader.filename):
            psd_last_sample = psd_reader.get_last_sample()
            self.current_info["psd"]["cc_A"] = psd_last_sample.branches["psd"].coords[0, :, 0, :2]
            self.current_info["psd"]["cc_E"] = psd_last_sample.branches["psd"].coords[0, :, 0, 2:]
            self.current_info["psd"]["cc_foreground_params"] = psd_last_sample.branches["galfor"].coords[0, :, 0, :]
            
            if self.current_info["general"]["begin_new_likelihood"]:
                del psd_last_sample.log_like
                del psd_last_sample.log_prior
            else:
                self.current_info["psd"]["cc_ll"] = psd_last_sample.log_like[0]
                self.current_info["psd"]["cc_lp"] = psd_last_sample.log_prior[0]
                
            self.current_info["psd"]["last_state"] = psd_last_sample

        mbh_reader = HDFBackend(settings["general"]["file_information"]["fp_mbh_pe"])
        mbh_search_file = settings["general"]["file_information"]["fp_mbh_search_base"] + "_output.pickle"
        
        self.current_info["mbh"]["reader"] = mbh_reader    

        if os.path.exists(mbh_search_file) and not os.path.exists(mbh_reader.filename):
            with open(mbh_search_file, "rb") as fp:
                mbh_output_point_info = pickle.load(fp)

            if "output_points_pruned" in mbh_output_point_info:
                self.initialize_mbh_state_from_search(mbh_output_point_info)
          
        elif os.path.exists(self.current_info["mbh"]["reader"].filename):
            mbh_last_sample = mbh_reader.get_last_sample()
            self.current_info["mbh"]["cc_params"] = mbh_last_sample.branches["mbh"].coords[0, :, :]
            if self.current_info["general"]["begin_new_likelihood"]:
                del mbh_last_sample.log_like
                del mbh_last_sample.log_prior
            else:
                self.current_info["mbh"]["cc_ll"] = mbh_last_sample.log_like[0]
                self.current_info["mbh"]["cc_lp"] = mbh_last_sample.log_prior[0]
            self.current_info["mbh"]["last_state"] = mbh_last_sample

              
        # TODO: add all to GMM
        self.current_info["gb"]["reader"] = gb_reader
        if os.path.exists(self.current_info["gb"]["reader"].filename):
            gb_last_sample = gb_reader.get_last_sample()

            self.current_info["gb"]["cc_params"] = gb_last_sample.branches["gb_fixed"].coords[0, :, :]
            self.current_info["gb"]["cc_inds"] = gb_last_sample.branches["gb_fixed"].inds[0, :, :]
            if self.current_info["general"]["begin_new_likelihood"]:
                del gb_last_sample.log_like
                del gb_last_sample.log_prior
            else:
                self.current_info["gb"]["cc_ll"] = gb_last_sample.log_like[0]
                self.current_info["gb"]["cc_lp"] = gb_last_sample.log_prior[0]
            self.current_info["gb"]["last_state"] = gb_last_sample

        else:
            coords = np.zeros((
                self.current_info["gb"]["pe_info"]["ntemps"],
                self.current_info["gb"]["pe_info"]["nwalkers"],
                self.current_info["gb"]["pe_info"]["nleaves_max"],
                self.current_info["gb"]["pe_info"]["ndim"]
            ))
            inds = np.zeros((
                self.current_info["gb"]["pe_info"]["ntemps"],
                self.current_info["gb"]["pe_info"]["nwalkers"],
                self.current_info["gb"]["pe_info"]["nleaves_max"]
            ), dtype=bool)
            ll = np.zeros((
                self.current_info["gb"]["pe_info"]["ntemps"],
                self.current_info["gb"]["pe_info"]["nwalkers"]
            ))
            lp = np.zeros((
                self.current_info["gb"]["pe_info"]["ntemps"],
                self.current_info["gb"]["pe_info"]["nwalkers"]
            ))
            self.current_info["gb"]["cc_params"] = coords[0]
            self.current_info["gb"]["cc_inds"] = inds[0]
            self.current_info["gb"]["cc_ll"] = ll[0]
            self.current_info["gb"]["cc_lp"] = lp[0]
            self.current_info["gb"]["last_state"] = State(
                {"gb_fixed": coords}, inds={"gb_fixed": inds}, log_like=ll, log_prior=ll
            )

        # check = self.current_info["general"]["generate_current_state"](self.current_info, include_ll=True, include_source_only_ll=True, n_gen_in=18)
        # print(check["ll"])

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

        print("DONE with SETUP")

    def get_data_psd(self, **kwargs):
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


class GlobalFitSegment(ABC):
    def __init__(self, comm):
        self.comm = comm
        self.base_settings = get_global_fit_settings()
        settings = deepcopy(self.base_settings)
        self.gpus = settings["general"]["gpus"]
        self.adjust_settings(settings)

        self.current_info = CurrentInfoGlobalFit(settings)

    @classmethod
    def adjust_settings(self, settings):
        raise NotImplementedError

    @classmethod
    def run(self):
        raise NotImplementedError
        

class MPIControlGlobalFit:
    def __init__(self, current_info, comm, gpus):

        ranks = np.arange(comm.Get_size())

        self.rank = comm.Get_rank()
        
        assert len(gpus) == 4
        assert len(ranks) >= 5
        
        self.current_info = current_info
        self.comm = comm
        self.gpus = gpus

        self.head_rank = ranks[1]

        self.gb_pe_rank = ranks[2]
        self.gb_pe_gpu = gpus[0]

        self.gb_search_rank = ranks[3]
        self.gb_search_gpu = gpus[1]

        self.psd_rank = ranks[0]
        self.psd_gpu = gpus[2]

        self.mbh_rank = ranks[4]
        self.mbh_gpu = gpus[3]

        self.gmm_ranks = ranks[5:]
        self.have_started_refit = False
    

    def update_refit(self, gmm_info):
        if "search" in gmm_info and gmm_info["search"] is not None:
            # TODO: check if we want this
            self.current_info.gb_info["search_gmm_info"] = gmm_info["search"]
            
            # if self.current_info.gb_info["search_gmm_info"] is None:
            #     self.current_info.gb_info["search_gmm_info"] = gmm_info["search"]
            # else:
            #     # search keeps adding to what it already has
            #     self.current_info.gb_info["search_gmm_info"] = [tmp1 + tmp2 for tmp1, tmp2 in zip(self.current_info.gb_info["search_gmm_info"], gmm_info["search"])]
            
        if "sample_refit" in gmm_info and gmm_info["sample_refit"] is not None:
            # refit always just replaces it
            self.current_info.gb_info["refit_gmm_info"] = gmm_info["sample_refit"]
            
        with open(self.current_info.general_info["file_information"]["fp_gb_gmm_info"], "wb") as fp_gmm:
            gmm_info_dict = {"search": self.current_info.gb_info["search_gmm_info"], "refit": self.current_info.gb_info["refit_gmm_info"]}
            pickle.dump(gmm_info_dict, fp_gmm, protocol=pickle.HIGHEST_PROTOCOL)

    def update_gbs(self, gb_dict):
        assert "gb_update" in gb_dict
        for key in ["cc_params", "cc_inds", "cc_ll", "cc_lp", "last_state"]:
            if key == "last_state" or key not in self.current_info.gb_info:
                self.current_info.gb_info[key] = gb_dict["gb_update"][key]
            else:
                self.current_info.gb_info[key][:] = gb_dict["gb_update"][key][:]

    def update_psd(self, psd_dict):
        assert "psd_update" in psd_dict
        for key in ["cc_A", "cc_E", "cc_foreground_params", "cc_ll", "cc_lp", "last_state"]:
            if key == "last_state" or key not in self.current_info.psd_info:
                self.current_info.psd_info[key] = psd_dict["psd_update"][key]
            else:
                self.current_info.psd_info[key][:] = psd_dict["psd_update"][key][:]

    def update_mbhs(self, mbh_dict):
        assert "mbh_update" in mbh_dict
        for key in ["cc_params", "cc_ll", "cc_lp", "last_state"]:
            if key == "last_state" or key not in self.current_info.mbh_info:
                self.current_info.mbh_info[key] = mbh_dict["mbh_update"][key]
            else:
                self.current_info.mbh_info[key][:] = mbh_dict["mbh_update"][key][:]

    def share_start_info(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):
        
        if run_gbs_pe:
            self.comm.send(self.current_info, dest=self.gb_pe_rank, tag=255)
    
        if run_psd:
            self.comm.send(self.current_info, dest=self.psd_rank, tag=46)
       
        if run_mbhs:
            self.comm.send(self.current_info, dest=self.mbh_rank, tag=76)
        
        if run_gbs_search:
            self.comm.send(self.current_info, dest=self.gb_search_rank, tag=2929)

    def refit_check(self, update_refit):
        # check if refit is ready
        refit_check = self.comm.irecv(source=self.gb_search_rank, tag=20)
        if refit_check.get_status():
            refit_dict = refit_check.wait()
            print("CHECK", refit_dict)
            if "receive" in refit_dict and refit_dict["receive"]:
                update_refit = False
                gmm_info = self.comm.recv(source=self.gb_search_rank, tag=29)
                self.update_refit(gmm_info)

            if "send" in refit_dict and refit_dict["send"]:
                self.comm.send(self.current_info, dest=self.gb_search_rank, tag=27)

        else:
            refit_check.cancel()

        return update_refit

    def gb_check(self, update_refit):
        gb_check = self.comm.irecv(source=self.gb_pe_rank, tag=50)
        if gb_check.get_status():
            gb_req = gb_check.wait()

            if "receive" in gb_req and gb_req["receive"]:
                gb_dict = self.comm.recv(source=self.gb_pe_rank, tag=58)
                self.update_gbs(gb_dict)
                update_refit = True

            if "send" in gb_req and gb_req["send"]:
                self.comm.send(self.current_info, dest=self.gb_pe_rank, tag=51)
                
            if "start_refit" in gb_req and gb_req["start_refit"]:
                update_refit = True
                # if not self.have_started_refit:
                    # self.comm.send(True, dest=self.gb_search_rank, tag=1010)
                    # self.comm.send(self.current_info, dest=self.gb_search_rank, tag=2929)
                self.have_started_refit = True
        else:
            gb_check.cancel()
    
        return update_refit

    def psd_check(self):
        psd_check = self.comm.irecv(source=self.psd_rank, tag=60)
        if psd_check.get_status():
            psd_req = psd_check.wait()

            if "receive" in psd_req and psd_req["receive"]:
                psd_dict = self.comm.recv(source=self.psd_rank, tag=68)
            
                self.update_psd(psd_dict)

            if "send" in psd_req and psd_req["send"]:
                self.comm.send(self.current_info, dest=self.psd_rank, tag=61)

        else:
            psd_check.cancel()

    def mbh_check(self):
        mbh_check = self.comm.irecv(source=self.mbh_rank, tag=70)
        if mbh_check.get_status():
            mbh_req = mbh_check.wait()

            if "receive" in mbh_req and mbh_req["receive"]:
                mbh_dict = self.comm.recv(source=self.mbh_rank, tag=78)
            
                self.update_mbhs(mbh_dict)

            if "send" in mbh_req and mbh_req["send"]:
                self.comm.send(self.current_info, dest=self.mbh_rank, tag=71)

        else:
            mbh_check.cancel()

    def initialize_mbh_state_from_search(self, mbh_output_point_info):
        output_points_pruned = np.asarray(mbh_output_point_info["output_points_pruned"]).reshape(1, 0, 2)
        coords = np.zeros((self.current_info["mbh"]["pe_info"]["ntemps"], self.current_info["mbh"]["pe_info"]["nwalkers"], output_points_pruned.shape[1], self.current_info["mbh"]["pe_info"]["ndim"]))
        coords[:] = output_points_pruned[None, :]

        self.current_info["mbh"]["cc_params"] = coords[0]
        self.current_info["mbh"]["last_state"] = State({"mbh": coords})

    def run_global_fit(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):
        if self.rank == self.head_rank:

            # send to refit 
            print("sending data")

            self.share_start_info(run_psd=run_psd, run_mbhs=run_mbhs, run_gbs_pe=run_gbs_pe, run_gbs_search=run_gbs_search)

            update_refit = True
            while True:
                time.sleep(1)

                # print("checking again", update_refit)
                
                if update_refit:
                    update_refit = self.refit_check(update_refit)

                # check for update from GBs
                update_refit = self.gb_check(update_refit)

                # check for update from PSD
                self.psd_check()

                # check for update from MBH
                self.mbh_check()
        
        elif self.rank == self.gb_pe_rank and run_gbs_pe:
            fin = run_gb_pe(self.gb_pe_gpu, self.comm, self.head_rank)

        elif self.rank == self.gb_search_rank and run_gbs_search:
            comm_info = {"process_ranks_for_fit": self.gmm_ranks}
            # refit_check = self.comm.recv(source=self.head_rank, tag=1010)
            # # block until it is told to go
            print("STARTING REFIT")
            fin = run_gb_bulk_search(self.gb_search_gpu, self.comm, comm_info, self.head_rank)
        
        elif self.rank == self.psd_rank and run_psd:
            fin = run_psd_pe(self.psd_gpu, self.comm, self.head_rank)

        elif self.rank == self.mbh_rank and run_mbhs:
            fin = run_mbh_pe(self.mbh_gpu, self.comm, self.head_rank)

        elif self.rank in self.gmm_ranks and run_gbs_search:
            rec_tag = int(str(self.rank) + "300")
            send_tag = int(str(self.rank) + "400")
            fit_each_leaf(self.rank, self.gb_search_rank, rec_tag, send_tag, self.comm)

        self.comm.Barrier()

