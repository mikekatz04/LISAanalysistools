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
from lisatools.globalfit.plot import RunResultsProduction
from lisatools.globalfit.hdfbackend import save_to_backend_asynchronously_and_plot

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
    def __init__(self, settings, get_last_state_info=True):

        self.settings_dict = settings
        self.current_info = deepcopy(settings)

        gb_reader = GBHDFBackend(settings["general"]["file_information"]["fp_gb_pe"])
        # with open("save_state_new_gmm.pickle", "rb") as fp:
        #     gb_last_sample = pickle.load(fp)
        #     gb_last_sample.log_prior = np.zeros_like(gb_last_sample.log_like)
        #     gb_last_sample.branches["gb"].coords[~gb_last_sample.branches["gb"].inds] = np.nan
        # # breakpoint()

        psd_reader = HDFBackend(settings["general"]["file_information"]["fp_psd_pe"])
        self.current_info["psd"]["reader"] = psd_reader
        
        if os.path.exists(psd_reader.filename) and get_last_state_info:
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
          
        elif os.path.exists(self.current_info["mbh"]["reader"].filename) and get_last_state_info:
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
        if os.path.exists(self.current_info["gb"]["reader"].filename) and get_last_state_info:
            gb_last_sample = gb_reader.get_last_sample()

            self.current_info["gb"]["cc_params"] = gb_last_sample.branches["gb"].coords[0, :, :]
            self.current_info["gb"]["cc_inds"] = gb_last_sample.branches["gb"].inds[0, :, :]
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
                {"gb": coords}, inds={"gb": inds}, log_like=ll, log_prior=ll
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
    
    def initialize_mbh_state_from_search(self, mbh_output_point_info):
        output_points_pruned = np.asarray(mbh_output_point_info["output_points_pruned"]).transpose(1, 0, 2)
        coords = np.zeros((self.current_info["mbh"]["pe_info"]["ntemps"], self.current_info["mbh"]["pe_info"]["nwalkers"], output_points_pruned.shape[1], self.current_info["mbh"]["pe_info"]["ndim"]))
        assert output_points_pruned.shape[0] >= self.current_info["mbh"]["pe_info"]["nwalkers"]
        
        coords[:] = output_points_pruned[None, :self.current_info["mbh"]["pe_info"]["nwalkers"]]

        self.current_info["mbh"]["cc_params"] = coords[0]
        self.current_info["mbh"]["last_state"] = State({"mbh": coords}, log_like=np.zeros((self.current_info["mbh"]["pe_info"]["ntemps"], self.current_info["mbh"]["pe_info"]["nwalkers"])))


    def get_data_psd(self, **kwargs):
        # self passed here to access all current info
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

    @property
    def rank_info(self):
        return self.current_info["rank_info"]

    @property
    def gpu_assignments(self):
        return self.current_info["gpu_assignments"]


class GlobalFitSegment(ABC):
    def __init__(self, comm, copy_settings_file=False):
        self.comm = comm
        self.base_settings = get_global_fit_settings(copy_settings_file=copy_settings_file)
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
    def __init__(self, current_info, comm, gpus, run_results_update=True, **run_results_update_kwargs):

        ranks = np.arange(comm.Get_size())

        self.rank = comm.Get_rank()
        self.run_results_update = run_results_update
        self.run_results_update_kwargs = run_results_update_kwargs
        
        assert len(gpus) in [3, 4]
        
        self.current_info = current_info
        self.comm = comm
        self.gpus = gpus

        self.head_rank = self.current_info.rank_info["head_rank"]

        self.gb_pe_rank = self.current_info.rank_info["gb_pe_rank"]
        self.gb_pe_gpu = self.current_info.gpu_assignments["gb_pe_gpu"]

        self.gb_search_rank = self.current_info.rank_info["gb_search_rank"]

        if isinstance(self.gb_search_rank, int):
            self.gb_search_rank = [self.gb_search_rank]

        self.gb_search_gpu = self.current_info.gpu_assignments["gb_search_gpu"]
        if isinstance(self.gb_search_gpu, int):
            self.gb_search_gpu = [self.gb_search_gpu]

        assert len(self.gb_search_gpu) == len(self.gb_search_rank) - 1

        self.psd_rank = self.current_info.rank_info["psd_rank"]
        self.psd_gpu = self.current_info.gpu_assignments["psd_gpu"]

        self.mbh_rank = self.current_info.rank_info["mbh_rank"]
        self.mbh_gpu = self.current_info.gpu_assignments["mbh_gpu"]

        self.all_gpu_assignments = []
        for tmp in [
            self.gb_pe_gpu,
            self.gb_search_gpu,
            self.mbh_gpu,
            self.psd_gpu
        ]:
            if isinstance(tmp, int):
                self.all_gpu_assignments.append(tmp)
            elif isinstance(tmp, list):
                for tmp1 in tmp:
                    assert isinstance(tmp1, int)
                self.all_gpu_assignments += tmp

        self.all_non_gmm_ranks = []
        for tmp in [
            self.head_rank,
            self.gb_pe_rank,
            self.gb_search_rank,
            self.mbh_rank,
            self.psd_rank
        ]:
            if isinstance(tmp, int):
                self.all_non_gmm_ranks.append(tmp)
            elif isinstance(tmp, list):
                for tmp1 in tmp:
                    assert isinstance(tmp1, int)
                self.all_non_gmm_ranks += tmp

        for gpu_assign in self.all_gpu_assignments:
            assert gpu_assign in self.gpus

        other_ranks = [i for i in ranks if i not in self.all_non_gmm_ranks]

        if run_results_update:
            assert len(other_ranks) >= 2
            self.run_results_rank = other_ranks[0]
            self.gmm_ranks = other_ranks[1:]
        else:
            assert len(other_ranks) >= 1
            self.run_results_rank = -1
            self.gmm_ranks = other_ranks[:]

        # setup for split search
        # must be rank because we want to include the refit
        split_ind = int(len(other_ranks) / len(self.gb_search_rank))
        gmm_rank_inds = np.arange(len(self.gmm_ranks))

        self.gmm_ranks = [[other_ranks[j] for j in gmm_rank_inds if (j % len(self.gb_search_rank)) == i] for i in range(len(self.gb_search_rank))]

        self.have_started_refit = False

        self.updated_search_gmm_info = [None for _ in range(len(self.gb_search_gpu))]

    def update_refit(self, search_index: int, gmm_info: dict):
        if "sample_refit" in gmm_info and gmm_info["sample_refit"] is not None:
            # refit always just replaces it
            self.current_info.gb_info["refit_gmm_info"] = gmm_info["sample_refit"]
            
        if "search" in gmm_info and gmm_info["search"] is not None:
            # TODO: check if we want this
            assert search_index > 0
            self.updated_search_gmm_info[search_index - 1] = gmm_info["search"]
            
            # if self.current_info.gb_info["search_gmm_info"] is None:
            #     self.current_info.gb_info["search_gmm_info"] = gmm_info["search"]
            # else:
            #     # search keeps adding to what it already has
            #     self.current_info.gb_info["search_gmm_info"] = [tmp1 + tmp2 for tmp1, tmp2 in zip(self.current_info.gb_info["search_gmm_info"], gmm_info["search"])]
        
        if np.all(np.asarray(self.updated_search_gmm_info) != None):
            # combine info
            out_lists = [[] for i in range(len(self.updated_search_gmm_info[0]))]
            for i in range(len(self.updated_search_gmm_info[0])):
                for j in range(len(self.updated_search_gmm_info)):
                    out_lists[i] = out_lists[i] + self.updated_search_gmm_info[j][i]

            # save update
            self.current_info.gb_info["search_gmm_info"] = out_lists

            with open(self.current_info.general_info["file_information"]["fp_gb_gmm_info"], "wb") as fp_gmm:
                gmm_info_dict = {"search": self.current_info.gb_info["search_gmm_info"], "refit": self.current_info.gb_info["refit_gmm_info"]}
                pickle.dump(gmm_info_dict, fp_gmm, protocol=pickle.HIGHEST_PROTOCOL)

            if not hasattr(self, "gmm_output_iter"):
                self.gmm_output_iter = 0
            else:
                self.gmm_output_iter += 1

            with open(self.current_info.general_info["file_information"]["fp_gb_gmm_info"][:-7] + f"_{self.gmm_output_iter}.pickle", "wb") as fp_gmm:
                pickle.dump(gmm_info_dict, fp_gmm, protocol=pickle.HIGHEST_PROTOCOL)
                
            self.updated_search_gmm_info = [None for _ in range(len(self.gb_search_gpu))]

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
            for i, gb_search_rank in enumerate(self.gb_search_rank):
                tag = int(str(2929) + str(gb_search_rank))
                print(f"YAYA {i}", gb_search_rank, tag)
                self.comm.send(self.current_info, dest=gb_search_rank, tag=tag)

    def refit_check(self, runs_going, update_refit):
        if "gbs_search" not in runs_going:
            return

        for i, gb_search_rank in enumerate(self.gb_search_rank):
            # only need to stop this on the search operations
            if i > 0 and self.updated_search_gmm_info[i - 1] is not None:
                # cannot allow this to run through the loop
                # will confuse "send"
                continue

            # check if refit is ready
            refit_ch = self.comm.irecv(source=gb_search_rank, tag=20)
            if refit_ch.get_status():
                refit_dict = refit_ch.wait()
                if "receive" in refit_dict and refit_dict["receive"]:
                    gmm_info = self.comm.recv(source=gb_search_rank, tag=29)
                    self.update_refit(i, gmm_info)
                
                    # received and both cleared out (but avoid for just sample refit)
                    if np.any(np.asarray(self.updated_search_gmm_info) == None) and "sample_refit" not in gmm_info or ("sample_refit" in gmm_info and gmm_info["sample_refit"] is None):
                        update_refit = False

                if "send" in refit_dict and refit_dict["send"] and np.all(np.asarray(self.updated_search_gmm_info) == None):
                    self.comm.send(self.current_info, dest=gb_search_rank, tag=27)

                if "finish_run" in refit_dict and refit_dict["finish_run"]:
                    runs_going.remove("gbs_search")
                    # for rank in self.gmm_ranks:
                    #     rec_tag = int(str(rank) + "67676")
                    #     self.comm.send("end", dest=rank, tag=rec_tag)

            else:
                refit_ch.cancel()

        return update_refit

    def gb_check(self, runs_going, update_refit):
        if "gbs_pe" not in runs_going:
            return
        gb_ch = self.comm.irecv(source=self.gb_pe_rank, tag=50)
        if gb_ch.get_status():
            gb_req = gb_ch.wait()

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
            
            if "no_binaries" in gb_req and gb_req["no_binaries"]:
                update_refit = True

            if "finish_run" in gb_req and gb_req["finish_run"]:
                runs_going.remove("gbs_pe")

        else:
            gb_ch.cancel()
    
        return update_refit

    def psd_check(self, runs_going):
        if "psd" not in runs_going:
            return
        psd_check = self.comm.irecv(source=self.psd_rank, tag=60)
        if psd_check.get_status():
            psd_req = psd_check.wait()
            if "receive" in psd_req and psd_req["receive"]:
                psd_dict = self.comm.recv(source=self.psd_rank, tag=68)
            
                self.update_psd(psd_dict)

            if "send" in psd_req and psd_req["send"]:
                self.comm.send(self.current_info, dest=self.psd_rank, tag=61)

            if "finish_run" in psd_req and psd_req["finish_run"]:
                runs_going.remove("psd")

        else:
            psd_check.cancel()

    def mbh_check(self, runs_going):
        if "mbhs" not in runs_going:
            return
            
        mbh_check = self.comm.irecv(source=self.mbh_rank, tag=70)
        if mbh_check.get_status():
            mbh_req = mbh_check.wait()

            if "receive" in mbh_req and mbh_req["receive"]:
                mbh_dict = self.comm.recv(source=self.mbh_rank, tag=78)
            
                self.update_mbhs(mbh_dict)

            if "send" in mbh_req and mbh_req["send"]:
                self.comm.send(self.current_info, dest=self.mbh_rank, tag=71)

            if "finish_run" in mbh_req and mbh_req["finish_run"]:
                runs_going.remove("mbhs")

        else:
            mbh_check.cancel()

    def plot_check(self):

        plot_ch = self.comm.irecv(source=self.run_results_rank, tag=91)
        if plot_ch.get_status():
            plot_req = plot_ch.wait()

            if "send" in plot_req and plot_req["send"]:
                self.comm.send(self.current_info, dest=self.run_results_rank, tag=92)

        else:
            plot_ch.cancel()

    def run_global_fit(self, run_psd=True, run_mbhs=True, run_gbs_pe=True, run_gbs_search=True):
        if self.rank == self.head_rank:
            # send to refit 
            print("sending data")

            self.share_start_info(run_psd=run_psd, run_mbhs=run_mbhs, run_gbs_pe=run_gbs_pe, run_gbs_search=run_gbs_search)

            update_refit = True

            print("done sending data")

            # populate which runs are going
            runs_going = []
            if run_psd:
                runs_going.append("psd")
            if run_gbs_pe:
                runs_going.append("gbs_pe")
            if run_gbs_search:
                runs_going.append("gbs_search")
            if run_mbhs:
                runs_going.append("mbhs")

            if len(runs_going) == 0:
                raise ValueError("No runs are going to be launched because all options for runs are False.")

            while len(runs_going) > 0:
                time.sleep(1)

                # print("checking again", update_refit)
                if update_refit:
                    update_refit = self.refit_check(runs_going, update_refit)

                # check for update from GBs
                update_refit = self.gb_check(runs_going, update_refit)

                # check for update from PSD
                self.psd_check(runs_going)

                # check for update from MBH
                self.mbh_check(runs_going)

                self.plot_check()
        
        elif self.rank == self.gb_pe_rank and run_gbs_pe:
            fin = run_gb_pe(self.gb_pe_gpu, self.comm, self.head_rank, self.run_results_rank)

        elif self.rank in self.gb_search_rank and run_gbs_search:
            num_search = len(self.gb_search_rank)
            split = list(range(num_search))
            assert num_search == len(self.gb_search_gpu) + 1
            index_search = self.gb_search_rank.index(self.rank)
            # one GPU will do both search and refit
            gpu_here = self.gb_search_gpu[index_search] if index_search == 0 else self.gb_search_gpu[index_search - 1]
            comm_info = {"process_ranks_for_fit": self.gmm_ranks[index_search]}
            # refit_check = self.comm.recv(source=self.head_rank, tag=1010)
            # # block until it is told to go
            print("STARTING REFIT", split, self.rank, index_search, self.gmm_ranks[index_search])
            fin = run_gb_bulk_search(gpu_here, self.comm, comm_info, self.head_rank, num_search, index_search)
        
        elif self.rank == self.psd_rank and run_psd:
            fin = run_psd_pe(self.psd_gpu, self.comm, self.head_rank)

        elif self.rank == self.mbh_rank and run_mbhs:
            fin = run_mbh_pe(self.mbh_gpu, self.comm, self.head_rank)

        elif self.run_results_update and self.rank == self.run_results_rank:
            save_to_backend_asynchronously_and_plot(self.current_info.gb_info["reader"], self.comm, self.gb_pe_rank, self.head_rank, self.current_info.general_info["plot_iter"], self.current_info.general_info["backup_iter"])
            #fin = run_results_production()

        elif run_gbs_search:
            assert len(self.gmm_ranks) == len(self.gb_search_rank)
            for gmm_ranks_here, gb_search_rank_here in zip(self.gmm_ranks, self.gb_search_rank):
                if self.rank in gmm_ranks_here:
                    rec_tag = int(str(self.rank) + "300")
                    send_tag = int(str(self.rank) + "400")
                    print(f"GOING IN {self.rank, gb_search_rank_here}")
                    fit_each_leaf(self.rank, gb_search_rank_here, rec_tag, send_tag, self.comm)



def run_gf_progression(global_fit_progression, comm, head_rank):
    rank = comm.Get_rank()
    
    for g_i, global_fit_segment in enumerate(global_fit_progression):
        segment = global_fit_segment["segment"](*global_fit_segment["args"], **global_fit_segment["kwargs"]) 

        # check if this segment has completed
        if segment.current_info.general_info["file_information"]["status_file"].split("/")[-1] in os.listdir(segment.current_info.general_info["file_information"]["file_store_dir"]):
            with open(segment.current_info.general_info["file_information"]["status_file"], "r") as fp:
                lines = fp.readlines()
                if global_fit_segment["name"] + "\n" in lines:
                    continue
        if rank == head_rank:
            class_name = global_fit_segment["name"]
            print(f"\n\nStarting {class_name}\n\n")
            st = time.perf_counter()
        print("start", g_i, rank)
        segment.run()
        print("finished:", rank)
        comm.Barrier()
        print("end", g_i, rank)
        if rank == head_rank:
            class_name = global_fit_segment["name"]
            et = time.perf_counter()
            print(f"\n\nEnding {class_name}({et - st} sec)\n\n")
            with open(segment.current_info.general_info["file_information"]["status_file"], "a") as fp:
                fp.write(global_fit_segment["name"] + "\n")