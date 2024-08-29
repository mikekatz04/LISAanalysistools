from multiprocessing.sharedctypes import Value
import cupy as xp
import time
import pickle
import shutil
import numpy as np
from copy import deepcopy

# from lisatools.sampling.moves.gbspecialgroupstretch import GBSpecialGroupStretchMove

mempool = xp.get_default_memory_pool()

from eryn.ensemble import EnsembleSampler
from lisatools.utils.multigpudataholder import MultiGPUDataHolder
from eryn.moves import CombineMove
from eryn.moves.tempering import make_ladder
from eryn.state import State, BranchSupplimental
from lisatools.sampling.prior import GBPriorWrap
from eryn.prior import ProbDistContainer
from gbgpu.gbgpu import GBGPU

import subprocess

import warnings
warnings.filterwarnings("ignore")

from lisatools.sampling.stopping import SearchConvergeStopping


from eryn.moves import Move
class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros(state.log_like.shape)
        self.temperature_control.swaps_accepted = np.zeros(self.temperature_control.ntemps - 1)
        return state, accepted

def log_like(x, freqs, data, gb, df, data_length, supps=None, **sens_kwargs):
    if supps is None:
        raise ValueError("Must provide supps to identify the data streams.")

    wi = supps["walker_inds"]
    psd_pars = x[0]
    galfor_pars = x[1]
    A_data = data[0]
    E_data = data[1]
    
    data_index_all = xp.asarray(wi).astype(np.int32)
    ll = xp.zeros(psd_pars.shape[0]) 
    A_Soms_d_in_all = xp.asarray(psd_pars[:, 0])
    A_Sa_a_in_all = xp.asarray(psd_pars[:, 1])
    E_Soms_d_in_all = xp.asarray(psd_pars[:, 2])
    E_Sa_a_in_all = xp.asarray(psd_pars[:, 3])
    Amp_all = xp.asarray(galfor_pars[:, 0])
    kn_all = xp.asarray(galfor_pars[:, 1])
    alpha_all = xp.asarray(galfor_pars[:, 2])
    sl1_all = xp.asarray(galfor_pars[:, 3])
    sl2_all = xp.asarray(galfor_pars[:, 4])
    num_data = 1
    num_psds = psd_pars.shape[0]

    gb.psd_likelihood(ll, freqs, A_data, E_data, data_index_all,  A_Soms_d_in_all,  A_Sa_a_in_all,  E_Soms_d_in_all,  E_Sa_a_in_all, 
                     Amp_all,  alpha_all,  sl1_all,  kn_all, sl2_all, df, data_length, num_data, num_psds)
    
    # # galfor_pars = None
    # ll2 = xp.zeros_like(ll)
    # for i, (psd_pars_i, galfor_pars_i) in enumerate(zip(psd_pars, galfor_pars)):
    #     psd = [
    #         get_sensitivity(freqs, model=psd_pars_i[:2], foreground_params=galfor_pars_i, **sens_kwargs),
    #         get_sensitivity(freqs, model=psd_pars_i[2:], foreground_params=galfor_pars_i, **sens_kwargs)
    #     ]
    #     psd[0][0] = psd[0][1]
    #     psd[1][0] = psd[1][1]

    #     # inner_product = 4 * df * (xp.sum(data[0][wi].conj() * data[0][wi] / psd[0]) + xp.sum(data[1][wi].conj() * data[1][wi] / psd[1])).real
    #     inner_product = 4 * df * (xp.sum(data[0].conj() * data[0] / psd[0]) + xp.sum(data[1].conj() * data[1] / psd[1])).real
    #     ll2[i] = -1/2 * inner_product - xp.sum(xp.log(xp.asarray(psd)))
    # assert np.allclose(ll.get(), ll2.get())
    return ll.get()


class PSDwithGBPriorWrap:
    def __init__(self, nwalkers, gb, priors):
        self.gb = gb
        self.priors = priors 
        self.nwalkers = nwalkers
        
    @property
    def full_state(self):
        if not hasattr(self, "_full_state") or self._full_state is None:
            raise ValueError("Need to provide a full state when working with calculations of nwalkers < self.nwalkers.")
        return self._full_state

    @full_state.setter
    def full_state(self, full_state):
        assert full_state.log_like.shape[-1] == self.nwalkers
        self._full_state = full_state

    def logpdf(self, coords, inds, supps=None, branch_supps=None):

        psd_pars = coords["psd"].reshape(-1, coords["psd"].shape[-1])
        galfor_pars = coords["galfor"].reshape(-1, coords["galfor"].shape[-1])

        ntemps, nwalkers = coords["psd"].shape[:2]
        # qwalker_inds = np.arange(nwalkers)

        psd_logpdf = self.priors["psd"].logpdf(psd_pars)
        galfor_logpdf = self.priors["galfor"].logpdf(galfor_pars)

        if nwalkers == self.nwalkers:
            mbh_params = coords["mbh"][0].copy()
            gb_params = coords["gb"][0]
            gb_inds = inds["gb"][0]
        else:
            mbh_params = self.full_state.branches["mbh"].coords[0].copy()
            gb_params = self.full_state.branches["gb"].coords[0].copy()
            gb_inds = self.full_state.branches["gb"].inds[0].copy()

        mbh_logpdf_tmp = self.priors["mbh"].logpdf(mbh_params.reshape(-1, mbh_params.shape[-1])).reshape(mbh_params.shape[:-1]).sum(axis=-1)
        mbh_logpdf = mbh_logpdf_tmp[supps[:]["walker_inds"].flatten()]
        
        if gb_inds.sum() > 0:  # more than zero binaries
            assert supps is not None
            # walker_inds_in = walker_inds[supps[:]["walker_inds"].flatten()]
            gb_inds_in = gb_inds[supps[:]["walker_inds"].flatten()]
            gb_params_in = gb_params[supps[:]["walker_inds"].flatten()][gb_inds_in]
            # current_group_inds = np.repeat(np.arange(psd_pars.shape[0])[:, None], walker_inds_in.shape[1], axis=-1)
            # noise_index_all = xp.asarray(current_group_inds[gb_inds_in]).astype(np.int32)
            noise_index_all = xp.repeat(xp.arange(ntemps * nwalkers)[:, None], gb_inds.shape[-1], axis=-1)[gb_inds_in].astype(np.int32)

            A_Soms_d_in_all = xp.asarray(psd_pars[:, 0])
            A_Sa_a_in_all = xp.asarray(psd_pars[:, 1])
            E_Soms_d_in_all = xp.asarray(psd_pars[:, 2])
            E_Sa_a_in_all = xp.asarray(psd_pars[:, 3])
            Amp_all = xp.asarray(galfor_pars[:, 0])
            alpha_all = xp.asarray(galfor_pars[:, 1])
            sl1_all = xp.asarray(galfor_pars[:, 2])
            kn_all = xp.asarray(galfor_pars[:, 3])
            sl2_all = xp.asarray(galfor_pars[:, 4])
            num_f = len(gb_params_in)

            Sn_A = xp.zeros(num_f, dtype=xp.float64)
            Sn_E = xp.zeros(num_f, dtype=xp.float64)
            f0 = xp.asarray(gb_params_in[:, 1]) / 1e3
            
            if len(f0) > 0:
                self.gb.get_lisasens_val(Sn_A, Sn_E, f0, noise_index_all, A_Soms_d_in_all,  A_Sa_a_in_all,  E_Soms_d_in_all,  E_Sa_a_in_all, Amp_all,  alpha_all,  sl1_all,  kn_all, sl2_all, num_f)
            
            gb_logpdf_contrib = self.priors["gb"].logpdf(gb_params_in, Sn_f=Sn_A)
            logpdf_contribution = np.zeros_like(gb_inds_in, dtype=np.float64)

            try:
                tmp = gb_logpdf_contrib.get()
            except AttributeError:
                tmp = gb_logpdf_contrib
            logpdf_contribution[gb_inds_in] = tmp
            gb_logpdf = logpdf_contribution.sum(axis=-1)
            
        else:
            gb_logpdf = 0.0

        all_logpdf = (gb_logpdf + psd_logpdf + galfor_logpdf + mbh_logpdf).reshape(coords["psd"].shape[:2])
        if np.all(np.isnan(all_logpdf)):
            raise ValueError("All log prior are inf")
        elif np.any(np.isnan(all_logpdf)):
            all_logpdf[np.isnan(all_logpdf)] = -np.inf
        return all_logpdf

    #     self.base_prior.logpdf(x, psds=self.mgh.)

# def log_like(x, freqs, data, supps=None, **sens_kwargs):
#     if supps is None:
#         raise ValueError("Must provide supps to identify the data streams.")

#     wi = supps["walker_inds"]

#     if isinstance(x, list):
#         psd_pars = x[0]
#         galfor_pars = x[1]
#     else:
#         psd_pars = x
#         galfor_pars = None

#     psd = [
#         get_sensitivity(freqs, model=psd_pars[:2], foreground_params=galfor_pars, **sens_kwargs),
#         get_sensitivity(freqs, model=psd_pars[2:], foreground_params=galfor_pars, **sens_kwargs)
#     ]
#     psd[0][0] = psd[0][1]
#     psd[1][0] = psd[1][1]
#     breakpoint
#     inner_product = 4 * df * (xp.sum(data[0].reshape(-1, 786433)[wi].conj() * data[0].reshape(-1, 786433)[wi] / psd[0]) + xp.sum(data[1].reshape(-1, 786433)[wi].conj() * data[1].reshape(-1, 786433)[wi] / psd[1])).real
#     breakpoint()
#     ll = -1/2 * inner_product - xp.sum(xp.log(xp.asarray(psd)))
#     return ll.get()


from eryn.utils.updates import Update

class UpdateNewResidualsPSD(Update):
    def __init__(
        self, comm, head_rank, last_mbh_prior_val, verbose=False
    ):
        self.comm = comm
        self.head_rank = head_rank
        self.verbose = verbose
        self.last_mbh_prior_val = last_mbh_prior_val

    def __call__(self, iter, last_sample, sampler):
        
        if self.verbose:
            print("Sending psd update to head process.")

        update_dict = {
            "cc_A": last_sample.branches["psd"].coords[0, :, 0, :2].copy(),
            "cc_E": last_sample.branches["psd"].coords[0, :, 0, 2:].copy(),
            "cc_foreground_params": last_sample.branches["galfor"].coords[0, :, 0, :].copy(),
            "cc_ll": last_sample.log_like[0].copy(),
            "cc_lp": last_sample.log_prior[0].copy(),
            "last_state": last_sample
        }

        self.comm.send({"send": True, "receive": True}, dest=self.head_rank, tag=60)

        self.comm.send({"psd_update": update_dict}, dest=self.head_rank, tag=68)

        if self.verbose:
            print("PSD: requesting updated data from head process.")

        new_info = self.comm.recv(source=self.head_rank, tag=61)
        
        if self.verbose:
            print("Received new data from head process.")

        nwalkers_pe = last_sample.log_like.shape[1]
        generated_info = new_info.get_data_psd(n_gen_in=nwalkers_pe, return_prior_val=True, include_lisasens=True, fix_val_in_gen=["psd"])
    
        data = generated_info["data"]
        psd = generated_info["psd"]
        
        # copy old information to restore if needed
        old_data_0 = sampler.log_like_fn.args[1][0][:].copy()
        old_data_1 = sampler.log_like_fn.args[1][1][:].copy()

        sampler.log_like_fn.args[1][0][:] = xp.asarray(data[0])
        sampler.log_like_fn.args[1][1][:] = xp.asarray(data[1])

        gb_params_old = sampler._priors["all_models_together"].gb_params.copy()
        walker_inds_old = sampler._priors["all_models_together"].walker_inds.copy()
        walker_inds_map_old = sampler._priors["all_models_together"].walker_inds_map.copy()
        gb_inds_old = sampler._priors["all_models_together"].gb_inds.copy()

        if self.verbose:
            print("Finished subbing in new data.")

        last_sample_copy = State(last_sample, copy=True)

        # need new prior
        gb_inds_generate = generated_info["gb_inds"]
        gb_nleaves_max = new_info.gb_info["cc_params"].shape[1]
        sampler._priors["all_models_together"].gb_params = new_info.gb_info["cc_params"][gb_inds_generate] # [new_info.gb_info["cc_inds"][gb_inds_generate]]
        sampler._priors["all_models_together"].walker_inds = np.repeat(np.arange(nwalkers_pe)[:, None], gb_nleaves_max, axis=-1)# [new_info.gb_info["cc_inds"][gb_inds_generate]]
        sampler._priors["all_models_together"].walker_inds_map = gb_inds_generate
        sampler._priors["all_models_together"].gb_inds = new_info.gb_info["cc_inds"][gb_inds_generate] 
        
        if "mbh_prior_vals" in generated_info:
            new_mbh_prior_vals =  generated_info["mbh_prior_vals"].copy()

        else:
            new_mbh_prior_vals = np.zeros(nwalkers_pe)
        # add in mbh prior value (gb is already included)
        new_lp_here = sampler.compute_log_prior(last_sample.branches_coords, inds=last_sample.branches_inds, supps=last_sample.supplimental)
        logp = new_mbh_prior_vals + new_lp_here[0]
        logl = sampler.compute_log_like(last_sample.branches_coords, inds=last_sample.branches_inds, supps=last_sample.supplimental, logp=new_lp_here)[0]
        logP = logp + logl[0]  # ALWAYS COLD CHAIN SO beta = 1
        
        # add MBH prior val that does not affect the rest of the sampler
        prev_logp = last_sample_copy.log_prior[0].copy() + self.last_mbh_prior_val
        prev_logl = last_sample_copy.log_like[0].copy()
        prev_logP = prev_logp + prev_logl  # ALWAYS COLD CHAIN SO beta = 1
        
        # factors = 0.0
        accept = (logP - prev_logP) > np.log(sampler.get_model().random.rand(*logl[0].shape))
        # TODO: this was not right in the end. Need to think about more. 
        # so adding this:
        accept[:] = True

        print(f"Acceptance fraction: {accept.sum() / accept.shape[0]}")
        # was not stored before --> update 
        last_sample.log_prior[:, accept] = new_lp_here[:, accept]
        last_sample.log_like[:, accept] = logl[:, accept]
        self.last_mbh_prior_val[accept] = new_mbh_prior_vals[accept]

        # stored before --> reset
        sampler.log_like_fn.args[1][0][~accept] = old_data_0[~accept]
        sampler.log_like_fn.args[1][1][~accept] = old_data_1[~accept]

        sampler._priors["all_models_together"].gb_params[~accept] = gb_params_old[~accept]
        sampler._priors["all_models_together"].walker_inds[~accept] = walker_inds_old[~accept]
        sampler._priors["all_models_together"].walker_inds_map[~accept] = walker_inds_map_old[~accept]
        sampler._priors["all_models_together"].gb_inds[~accept] = gb_inds_old[~accept]
        
        xp.get_default_memory_pool().free_all_blocks()

        return


def run_psd_pe(gpu, comm, head_rank):

    gpus = [gpu]
    
    gf_information = comm.recv(source=head_rank, tag=46)

    psd_info = gf_information.psd_info
    xp.cuda.runtime.setDevice(gpus[0])

    nwalkers_pe = psd_info["pe_info"]["nwalkers"]
    ntemps_pe = psd_info["pe_info"]["ntemps"]

    priors = {"all_models_together": psd_info["priors"]}

    generated_info = gf_information.get_data_psd(include_ll=False, include_source_only_ll=False, n_gen_in=nwalkers_pe, return_prior_val=True, fix_val_in_gen=["psd"])
    
    if "gb_inds" in generated_info:

        gb_inds_generate = generated_info["gb_inds"]
        gb_nleaves_max = gf_information.gb_info["cc_params"].shape[1]
        gpu_priors_in = deepcopy(gf_information.gb_info["priors"].priors_in)
        for key, item in gpu_priors_in.items():
            item.use_cupy = True

        gpu_priors = {"gb": GBPriorWrap(gf_information.gb_info["pe_info"]["ndim"], ProbDistContainer(gpu_priors_in, use_cupy=True))}

        gb_params = gf_information.gb_info["cc_params"][gb_inds_generate] # [gf_information.gb_info["cc_inds"][gb_inds_generate]]
        walker_inds = np.repeat(np.arange(nwalkers_pe)[:, None], gb_nleaves_max, axis=-1)# [gf_information.gb_info["cc_inds"][gb_inds_generate]]
        walker_inds_map = gb_inds_generate
        gb_inds = gf_information.gb_info["cc_inds"][gb_inds_generate] 

    else:
        gb_params = None
        walker_inds = None
        walker_inds_map = None
        gpu_priors = None
        gb_inds = None

    gb = GBGPU(use_gpu=True)
    prior_wrap = {"all_models_together": PSDwithGBPriorWrap(
        gb, 
        priors["all_models_together"], 
        gb_prior=gpu_priors, 
        gb_params=gb_params, 
        walker_inds=walker_inds, 
        walker_inds_map=walker_inds_map,
        gb_inds=gb_inds,
    )}

    if "last_state" in psd_info:
        last_sample = psd_info["last_state"]
    else:
        coords_psd = prior_wrap["all_models_together"].psd_prior["psd"].rvs(size=(ntemps_pe, nwalkers_pe, 1))
        inds_psd = np.ones(coords_psd.shape[:-1], dtype=bool)
        coords_galfor = prior_wrap["all_models_together"].psd_prior["galfor"].rvs(size=(ntemps_pe, nwalkers_pe, 1))
        inds_galfor = np.ones(coords_galfor.shape[:-1], dtype=bool)
        last_sample = State({"psd": coords_psd, "galfor": coords_galfor}, inds={"psd": inds_psd, "galfor": inds_galfor})

    fd = xp.asarray(gf_information.general_info["fd"])
    data = [xp.asarray(generated_info["data"][0]), xp.asarray(generated_info["data"][1])]
    walker_vals = np.tile(np.arange(nwalkers_pe), (ntemps_pe, 1))

    supps_base_shape = (ntemps_pe, nwalkers_pe)
    supps = BranchSupplimental({"walker_inds": walker_vals}, base_shape=supps_base_shape, copy=True)

    check = prior_wrap["all_models_together"].logpdf(last_sample.branches_coords, last_sample.branches_inds, supps=supps)

    sens_kwargs = psd_info["psd_kwargs"].copy()
 
    if hasattr(last_sample, "betas") and last_sample.betas is not None:
        betas = last_sample.betas
    else:
        betas = make_ladder(sum(list(psd_info["pe_info"]["ndims"].values())), ntemps=ntemps_pe)

        last_sample.betas = betas

    assert not np.all(betas == 0.0)
    assert len(betas) == ntemps_pe

    branch_names = psd_info["pe_info"]["branch_names"]
    
    state_mix = State(
        {name: last_sample.branches_coords[name] for name in branch_names}, 
        inds={name: last_sample.branches_inds[name] for name in branch_names}, 
        supplimental=supps, 
        betas=betas
    )

    mbh_priors_in = np.full(nwalkers_pe, np.inf)if "mbh_prior_vals" not in generated_info else generated_info["mbh_prior_vals"].copy()
    update = UpdateNewResidualsPSD(comm, head_rank, mbh_priors_in, verbose=False)

    ndims_in = psd_info["pe_info"]["ndims"]
    nleaves_max_in = psd_info["pe_info"]["nleaves_max"]

    # TODO: fix this 
    df = gf_information.general_info["df"]
    data_length = gf_information.general_info["data_length"]

    stopping_fn = psd_info["pe_info"]["stopping_function"]
    if hasattr(stopping_fn, "add_comm"):
        stopping_fn.add_comm(comm)
    stopping_iterations = psd_info["pe_info"]["stopping_iterations"]
    thin_by = psd_info["pe_info"]["thin_by"]

    sampler_mix = EnsembleSampler(
        nwalkers_pe,
        ndims_in,  # assumes ndim_max
        log_like,
        prior_wrap,
        tempering_kwargs={"betas": betas, "permute": False, "skip_swap_supp_names": ["walker_inds"]},
        nbranches=len(branch_names),
        nleaves_max=nleaves_max_in,
        kwargs=sens_kwargs,  # {"start_freq_ind": start_freq_ind, **waveform_kwargs},
        args=(fd, data, gb, df, data_length),
        backend=psd_info["reader"],
        vectorize=True,
        periodic=psd_info["periodic"],  # TODO: add periodic to proposals
        branch_names=branch_names,
        update_fn=update,  # sttop_converge_mix,
        update_iterations=psd_info["pe_info"]["update_iterations"],
        stopping_fn=stopping_fn,
        stopping_iterations=stopping_iterations,
        provide_groups=False,
        provide_supplimental=True,
    )

    lp = sampler_mix.compute_log_prior(state_mix.branches_coords, inds=state_mix.branches_inds, supps=state_mix.supplimental)
    ll = sampler_mix.compute_log_like(state_mix.branches_coords, inds=state_mix.branches_inds, supps=state_mix.supplimental, logp=lp)[0]

    state_mix.log_like = ll
    state_mix.log_prior = lp

    # equlibrating likelihood check: -4293090.6483655665,
    nsteps_mix = 50000

    print("Starting psd ll best:", state_mix.log_like.max(axis=-1))
    mempool.free_all_blocks()
    
    out = sampler_mix.run_mcmc(state_mix, nsteps_mix, progress=psd_info["pe_info"]["progress"], thin_by=thin_by, store=True)
    print("ending psd ll best:", out.log_like.max(axis=-1))

    # communicate end of run to head process
    comm.send({"finish_run": True}, dest=head_rank, tag=60)

    del data, fd
    
    # free memory
    mempool.free_all_blocks()
    return

if __name__ == "__main__":
    import argparse
    """parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int,
                        help='which gpu', required=True)

    args = parser.parse_args()"""

    output = run_psd_pe(6)
                