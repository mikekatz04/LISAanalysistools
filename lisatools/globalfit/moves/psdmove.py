import numpy as np
import cupy as cp
from copy import deepcopy
from eryn.model import Model
from eryn.state import State as eryn_State
from ..state import GFState
from tqdm import tqdm
from .globalfitmove import GlobalFitMove
from ..psdglobal import log_like as psd_log_like
import warnings
from eryn.moves import RedBlueMove, StretchMove
from ..moves import GlobalFitMove
from ..utils import new_sens_mat

class PSDMove(GlobalFitMove, StretchMove):
    def __init__(self, gb, acs, priors, *args, psd_kwargs={}, **kwargs):
        super(PSDMove, self).__init__(*args, **kwargs)
        self.acs = acs
        self.gb = gb
        self.psd_kwargs = psd_kwargs
        self.priors = priors
        
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

    def propose(self, model, state):
        # setup model framework for passing necessary 
        # self.priors["all_models_together"].full_state = state

        # ensuring it is up to date. Should not change anything.
        eryn_state_in = eryn_State(state.branches_coords, inds=state.branches_inds, supplemental=state.supplemental, branch_supplemental=state.branches_supplemental, betas=state.betas, log_like=state.log_like, log_prior=state.log_prior, copy=True)
        before_vals = model.analysis_container_arr.likelihood(sum_instead_of_trapz=True).copy()
        
        if np.any(np.abs(before_vals - state.log_like[0]) > 1e-4) :
            breakpoint()

        # TODO: separate temp control from ensemble sampler
  
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
        tmp_state, accepted = super(PSDMove, self).propose(tmp_model, state)
        # CHECK THIS STATE SETUP
        new_state = GFState(
            tmp_state,
            copy=True
        )
        new_state.sub_states = state.sub_states
        new_state.sub_state_bases = state.sub_state_bases

        nwalkers = len(self.acs)
        for w in range(nwalkers):
            psd_params = state.branches_coords["psd"][0, w, 0]
            galfor_params = state.branches_coords["galfor"][0, w, 0]
            sens_AE = new_sens_mat(f"walker_{w}", psd_params, galfor_params, self.acs.f_arr)
            self.acs[w].sens_mat = sens_AE

        self.acs.reset_linear_psd_arr()
        after_vals = self.acs.likelihood(sum_instead_of_trapz=True)
        if np.any(np.abs(after_vals - new_state.log_like[0]) > 1e-4) :
            breakpoint()
                   
        # TODO: NEED TO ADJUST ACS WITH NEW PSDS AND RESET LINEAR ARRAY
        return new_state, accepted

