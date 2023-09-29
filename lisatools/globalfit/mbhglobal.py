from multiprocessing.sharedctypes import Value
import cupy as xp
import time
import pickle
import shutil
import numpy as np

# from lisatools.sampling.moves.gbspecialgroupstretch import GBSpecialGroupStretchMove

mempool = xp.get_default_memory_pool()

from eryn.ensemble import EnsembleSampler
from single_mcmc_run import run_single_band_search
from lisatools.utils.multigpudataholder import MultiGPUDataHolder
from eryn.moves import CombineMove
from eryn.moves.tempering import make_ladder
from eryn.state import State, BranchSupplimental
from lisatools.sampling.moves.specialforegroundmove import GBForegroundSpecialMove
from lisatools.sampling.moves.mbhspecialmove import MBHSpecialMove

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.likelihood import Likelihood as MBHLikelihood
from bbhx.likelihood import HeterodynedLikelihood
from bbhx.utils.constants import *
from bbhx.utils.transform import *

import subprocess

import warnings
warnings.filterwarnings("ignore")

from eryn.moves import Move

from eryn.utils.updates import Update


class UpdateNewResidualsMBH(Update):
    def __init__(
        self, comm, head_rank, verbose=False
    ):
        self.comm = comm
        self.head_rank = head_rank
        self.verbose = verbose

    def __call__(self, iter, last_sample, sampler):
        
        if self.verbose:
            print("Sending psd update to head process.")
 
        update_dict = {
            "cc_params": last_sample.branches["mbh"].coords[0].copy(),
            "cc_ll": last_sample.log_like[0].copy(),
            "cc_lp": last_sample.log_prior[0].copy(),
            "last_state": last_sample
        }

        self.comm.send({"send": True, "receive": True}, dest=self.head_rank, tag=70)

        self.comm.send({"mbh_update": update_dict}, dest=self.head_rank, tag=78)

        if self.verbose:
            print("MBH: requesting updated data from head process.")

        new_info = self.comm.recv(source=self.head_rank, tag=71)
        
        if self.verbose:
            print("Received new data from head process.")

        nwalkers_pe = last_sample.log_like.shape[1]
        
        assert np.all(new_info.mbh_info["cc_params"] == last_sample.branches["mbh"].coords[0])
        
        generated_info = new_info.get_data_psd(n_gen_in=nwalkers_pe)
    
        data, psd = generated_info["data"], generated_info["psd"]

        data_fin = xp.asarray([data[0], data[1], np.zeros_like(data[0])])
        psds_fin = xp.asarray([psd[0], psd[1], np.full_like(psd[0], 1e10)])

        start_ll_check = np.zeros((data_fin.shape[1]))
 
        for i in range(data_fin.shape[1]):
            start_ll_check[i] = (-1/2 * 4 * new_info.general_info["df"] * xp.sum(data_fin[:2, i].conj() * data_fin[:2, i] / psds_fin[:2, i]) - xp.sum(xp.log(xp.asarray(psds_fin[:2, i])))).get().real

        xp.get_default_memory_pool().free_all_blocks()

        last_sample.log_like[0] = start_ll_check[:]

        for move in sampler.moves:
            move.data_residuals[:] = data_fin
            move.psd[:] = psds_fin

        
        if self.verbose:
            print("Finished subbing in new data.")

        del data_fin, psds_fin
        xp.get_default_memory_pool().free_all_blocks()
        
        return

class BasicResidualMGHLikelihood:
    def __init__(self, mgh):
        self.mgh = mgh
    def __call__(self, *args, supps=None, **kwargs):
        ll_temp = self.mgh.get_ll()
        overall_inds = supps["overal_inds"]
        
        return ll_temp[overall_inds]



def run_mbh_pe(gpu, comm, head_rank):

    gpus = [gpu]
    
    gf_information = comm.recv(source=head_rank, tag=76)

    mbh_info = gf_information.mbh_info
    xp.cuda.runtime.setDevice(gpus[0])

    last_sample = mbh_info["last_state"]

    nwalkers_pe = mbh_info["pe_info"]["nwalkers"]
    ntemps_pe = mbh_info["pe_info"]["ntemps"]

    assert np.all(mbh_info["cc_params"] == last_sample.branches["mbh"].coords[0])

    generated_info = gf_information.get_data_psd(include_ll=True, include_source_only_ll=True, n_gen_in=nwalkers_pe)
    fd = xp.asarray(gf_information.general_info["fd"])
    
    data_fin = xp.asarray([generated_info["data"][0], generated_info["data"][1], np.zeros_like(generated_info["data"][1])])
    psds_fin = xp.asarray([generated_info["psd"][0], generated_info["psd"][1], np.full_like(generated_info["psd"][1], 1e10)])

    df = gf_information.general_info["df"]

    xp.get_default_memory_pool().free_all_blocks()

    start_ll_check = np.zeros((data_fin.shape[1]))
 
    for i in range(data_fin.shape[1]):
        start_ll_check[i] = (-1/2 * 4 * df * xp.sum(data_fin[:2, i].conj() * data_fin[:2, i] / psds_fin[:2, i]) - xp.sum(xp.log(xp.asarray(psds_fin[:2, i])))).get().real

    xp.get_default_memory_pool().free_all_blocks()

    priors = mbh_info["priors"]

    if not hasattr(last_sample, "log_like"):
        last_sample.log_like = np.zeros((ntemps_pe, nwalkers_pe))
    if not hasattr(last_sample, "log_prior"):
        last_sample.log_prior = np.zeros((ntemps_pe, nwalkers_pe))

    start_state = State(last_sample, copy=True)
    
    # start_state.branches["mbh"].shape = (ntemps_pe, nwalkers_pe) + start_state.branches["mbh"].shape[2:]
    # start_state.branches["mbh"].coords = start_state.branches["mbh"].coords[:, :nwalkers_pe]
    # start_state.branches["mbh"].inds = start_state.branches["mbh"].inds[:, :nwalkers_pe]
    # start_state.log_prior = start_state.log_prior[:, :nwalkers_pe]
    # start_state.log_like = start_state.log_like[:, :nwalkers_pe]

    start_state.log_like[0] = start_ll_check
    branch_names = mbh_info["pe_info"]["branch_names"]

    if hasattr(start_state, "betas") and start_state.betas is not None:
        betas = start_state.betas
    else:
        betas = make_ladder(mbh_info["pe_info"]["ndim"], ntemps=ntemps_pe)

    start_state.betas = betas

    wave_gen = BBHWaveformFD(
        **mbh_info["initialize_kwargs"]
    )

    # TODO: fix this?
    wave_gen.d_d = 0.0

    # for transforms
    waveform_kwargs = mbh_info["waveform_kwargs"].copy()

    transform_fn = mbh_info["transform"]

    # sampler treats periodic variables by wrapping them properly
    periodic = mbh_info["periodic"]
    
    xp.get_default_memory_pool().free_all_blocks()

    # TODO: start ll needs to be done carefully
    inner_moves = mbh_info["pe_info"]["inner_moves"]
    print("MBH CHECK")
    move = MBHSpecialMove(wave_gen, fd, data_fin, psds_fin, mbh_info["pe_info"]["num_prop_repeats"], transform_fn, priors, waveform_kwargs, inner_moves, df)

    update = UpdateNewResidualsMBH(comm, head_rank, verbose=False)
    print("MBH CHECK 2")

    ndims = {"mbh": mbh_info["pe_info"]["ndim"]}

    like_mix = BasicResidualMGHLikelihood(None)
    # key permute is False

    stopping_fn = mbh_info["pe_info"]["stopping_function"]
    if hasattr(stopping_fn, "add_comm"):
        stopping_fn.add_comm(comm)
    stopping_iterations = mbh_info["pe_info"]["stopping_iterations"]
    thin_by = mbh_info["pe_info"]["thin_by"]

    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        ndims,  # assumes ndim_max
        like_mix,
        priors,
        moves=move,
        tempering_kwargs={"betas": betas, "permute": False},
        nbranches=len(branch_names),
        nleaves_max={"mbh": mbh_info["pe_info"]["nleaves_max"]},
        nleaves_min={"mbh": mbh_info["pe_info"]["nleaves_max"]},
        backend=mbh_info["reader"],  # mbh_info["reader"],
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        branch_names=branch_names,
        update_fn=update,  # stop_converge_mix,
        update_iterations=1,
        stopping_fn=stopping_fn,
        stopping_iterations=stopping_iterations,
        provide_groups=False,
        provide_supplimental=False,
    )

    # equlibrating likelihood check: -4293090.6483655665,
    nsteps_mix = 10000
    mempool.free_all_blocks()

    print("MBH CHECK 3")
    out = sampler_mix.run_mcmc(start_state, nsteps_mix, progress=mbh_info["pe_info"]["progress"], thin_by=thin_by, store=True)
    print("ending mix ll best:", out.log_like.max(axis=-1))
    # communicate end of run to head process
    comm.send({"finish_run": True}, dest=head_rank, tag=70)
    return

if __name__ == "__main__":
    import argparse
    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)

    args = parser.parse_args()"""

    output = run_mbh_pe(3)
                