import numpy as np
import cupy as cp
from copy import deepcopy
from eryn.model import Model
from eryn.state import State as eryn_State
from ..state import GFState
from tqdm import tqdm
from .globalfitmove import GlobalFitMove
import warnings
from eryn.moves import RedBlueMove, StretchMove
from ..moves import GlobalFitMove
from ..utils import new_sens_mat
from ...cutils.psd_gpu import psd_likelihood
from tqdm import tqdm
import time

def psd_log_like(x, freqs, data, gb, df, data_length, supps=None, **sens_kwargs):
    if supps is None:
        raise ValueError("Must provide supps to identify the data streams.")

    wi = supps["walker_inds"]
    psd_pars = x[0]
    galfor_pars = x[1]
    A_data = data[0]
    E_data = data[1]
    
    data_index_all = cp.asarray(wi).astype(np.int32)
    ll = cp.zeros(psd_pars.shape[0]) 
    A_Soms_d_in_all = cp.asarray(psd_pars[:, 0])
    A_Sa_a_in_all = cp.asarray(psd_pars[:, 1])
    E_Soms_d_in_all = cp.asarray(psd_pars[:, 2])
    E_Sa_a_in_all = cp.asarray(psd_pars[:, 3])
    Amp_all = cp.asarray(galfor_pars[:, 0])
    kn_all = cp.asarray(galfor_pars[:, 1])
    alpha_all = cp.asarray(galfor_pars[:, 2])
    sl1_all = cp.asarray(galfor_pars[:, 3])
    sl2_all = cp.asarray(galfor_pars[:, 4])
    num_data = 1
    num_psds = psd_pars.shape[0]
    
    psd_likelihood(ll, freqs, data, data_index_all,  A_Soms_d_in_all,  A_Sa_a_in_all,  E_Soms_d_in_all,  E_Sa_a_in_all, 
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

class PSDMove(GlobalFitMove, StretchMove):
    def __init__(self, gb, acs, priors, *args, num_repeats=1, max_logl_mode=False, psd_kwargs={}, **kwargs):
        GlobalFitMove.__init__(self, *args, **kwargs)
        StretchMove.__init__(self, *args, **kwargs)
        self.acs = acs
        self.gb = gb
        self.psd_kwargs = psd_kwargs
        self.priors = priors
        self.num_repeats = num_repeats
        self.max_logl_mode = max_logl_mode
        self.starting_now = True
        
    def compute_log_like(
        self, coords, inds=None, logp=None, supps=None, branch_supps=None
    ):  
        assert logp is not None
        logl = np.full_like(logp, -1e300)

        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            warnings.warn("All points entering likelihood have a log prior of minus inf.")
            return logl, None
        psd_coords = coords["psd"][logp_keep][:, 0]
        galfor_coords = coords["galfor"][logp_keep][:, 0]

        supps = supps[logp_keep]
        
        tmp_logl = psd_log_like([psd_coords, galfor_coords], cp.asarray(self.acs.f_arr), self.acs.linear_data_arr[0], self.gb, self.acs.df, self.acs.data_length, supps=supps, **self.psd_kwargs)

        logl[logp_keep] = tmp_logl

        self.prev_logl = logl.copy()

        return logl, None

    # def compute_log_prior(self, coords, inds=None, supps=None, branch_supps=None):

    #     ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape
 
    #     logp = np.zeros((ntemps, nwalkers))
    #     for key in ["galfor", "psd"]:
    #         ntemps, nwalkers, nleaves_max, ndim = coords[key].shape
    #         if nleaves_max > 1:
    #             raise NotImplementedError

    #         logp_contrib = self.priors[key].logpdf(coords[key].reshape(-1, ndim)).reshape(ntemps, nwalkers, nleaves_max).sum(axis=-1)
    #         logp[:] += logp_contrib

    #     # now that everything is lined up
    #     breakpoint()
    #     nleaves_max_gb = inds["gb"].shape[-1]
    #     gb_inds_tiled = cp.tile(cp.asarray(inds["gb"][0][None, :]), (ntemps, 1, 1))
    #     gb_coords = cp.tile(cp.asarray(coords["gb"][0]), (ntemps, 1, 1, 1))[gb_inds_tiled]
    #     walker_inds = cp.repeat(cp.arange(ntemps * nwalkers)[:, None], nleaves_max_gb, axis=-1).reshape(ntemps, nwalkers, nleaves_max_gb)[gb_inds_tiled]
    #     logp_per_bin = cp.zeros((ntemps, nwalkers, nleaves_max_gb))
    #     logp_per_bin[gb_inds_tiled] = self.gpu_priors["gb"].logpdf(gb_coords, psds=self.acs.lisasens_list[0][0],walker_inds=walker_inds)
    #     logp[:] += logp_per_bin.sum(axis=-1).get()
        

    #     cp.get_default_memory_pool().free_all_blocks()
    #     return logp

    def compute_log_prior(self, branches_coords, *args, **kwargs):
        ntemps, nwalkers, _, ndim_psd = branches_coords["psd"].shape
        ntemps, nwalkers, _, ndim_galfor = branches_coords["galfor"].shape
        ndims = {"psd": ndim_psd, "galfor": ndim_galfor}
        logp = np.zeros((ntemps, nwalkers))
        for key in ["psd", "galfor"]:
            logp[:] += self.priors[key].logpdf(branches_coords[key].reshape(-1, ndims[key])).reshape(ntemps, nwalkers)
        return logp

    def run_move(self, model, state):
        new_state, accepted = super(PSDMove, self).propose(model, state)
        return new_state, accepted
    
    def run_move_for_loop(self, model, state, num_repeats):
        for i in tqdm(range(num_repeats)):
            state, accepted = self.run_move(model, state)
        return state, accepted

    def run_move_max_likelihood(self, model, state):

        num_checks = 5
        num_so_far = 0
        max_logl = -np.inf
        changed_once = False
        while num_so_far < num_checks:
            state, accepted = self.run_move_for_loop(model, state, self.num_repeats)

            if state.log_like[0].max() != max_logl and not np.isinf(max_logl):
                changed_once = True
                
            if state.log_like[0].max() > max_logl:
                max_logl = state.log_like[0].max()
                num_so_far = 0
            else:
                if changed_once:
                    num_so_far += 1

            print(max_logl, num_so_far, num_checks)
            # breakpoint()

        return state, accepted

    def propose(self, model, state):
        # setup model framework for passing necessary 
        # self.priors["all_models_together"].full_state = state

        tmp_branches_coords = {key: state.branches_coords[key] for key in ["psd", "galfor"]}
        
        tmp_state = GFState(tmp_branches_coords, copy=True, supplemental=state.supplemental)
        
        # ensuring it is up to date. Should not change anything.
        # eryn_state_in = eryn_State(state.branches_coords, inds=state.branches_inds, supplemental=state.supplemental, branch_supplemental=state.branches_supplemental, betas=state.betas, log_like=state.log_like, log_prior=state.log_prior, copy=True)
        before_vals = model.analysis_container_arr.likelihood().copy()
        
        # TODO: check this
        # if self.starting_now:
        tmp_state.log_prior = self.compute_log_prior(tmp_branches_coords)
        tmp_state.log_like = self.compute_log_like(tmp_branches_coords, logp=tmp_state.log_prior, supps=tmp_state.supplemental)[0]
        self.starting_now = False
        # if np.any(np.abs(before_vals - tmp_state.log_like[0]) > 1e-4) :
        #     breakpoint()

        # breakpoint()
        # logp = model.compute_log_prior_fn(state.branches_coords, inds=state.branches_inds, supps=state.supplemental)
        # logl_test = self.compute_log_like(state.branches_coords, inds=state.branches_inds, supps=state.supplemental, logp=logp)
        tmp_coords_check = state.branches["psd"].coords[0,:,0].copy()
        tmp_model = Model(
            state,
            self.compute_log_like,
            self.compute_log_prior,
            self.temperature_control,
            model.map_fn,
            model.random,
        )

        # state.acs.set_psd_vals(
        #     state.branches["psd"].coords[0, :, 0], 
        #     overall_inds=np.arange(state.branches["psd"].shape[1]), 
        #     foreground_params=state.branches["galfor"].coords[0, :, 0]
        # )
        # avs_vals = state.acs.get_ll(include_psd_info=True).copy()
        if self.max_logl_mode:
            tmp_state, accepted = self.run_move_max_likelihood(tmp_model, tmp_state)
        
        else:
            tmp_state, accepted = self.run_move_for_loop(tmp_model, tmp_state, self.num_repeats)
        
        # CHECK THIS STATE SETUP
        new_state = GFState(
            state,
            copy=True
        )

        for key in ["psd", "galfor"]:
            new_state.branches[key].coords[:] = tmp_state.branches[key].coords[:]

        new_state.log_like[:] = tmp_state.log_like[:]
        new_state.log_prior[:] = tmp_state.log_prior[:]
        
        nwalkers = len(self.acs)
        for w in range(nwalkers):
            psd_params = new_state.branches_coords["psd"][0, w, 0]
            galfor_params = new_state.branches_coords["galfor"][0, w, 0]
            sens_AE = new_sens_mat(f"walker_{w}", psd_params, galfor_params, self.acs.f_arr)
            self.acs[w].sens_mat = sens_AE

        self.acs.reset_linear_psd_arr()
        after_vals = self.acs.likelihood()
        
        new_state.log_like[0] = after_vals
        return new_state, accepted

