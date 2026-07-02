import numpy as np
import logging
from copy import deepcopy
from typing import Dict, Any, List, Tuple
from eryn.moves import Move
from eryn.state import State
from eryn.model import Model
from .globalfitmove import GlobalFitMove
from ..state import GFState
from ...analysiscontainer import AnalysisContainerArray
from ..priors.sourceconfigs import BaseSourceConfig
from ...utils.typing import NDArrayLike
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_N

logger = logging.getLogger(__name__)

class HyperMove(GlobalFitMove, Move):
    """
    A model informed reversible jump move to jump between discrete models.
    The setup: 
        This move handles the changing in model index. This model index is propagated to other relevant 
        branches in the model using the ``model_index`` key in the branch supplemental information. This move is 
        designed to be used with a custom proposal that proposes a change in the model index. The branch supplemental
        informatin of each breach is propagated to prior calculation such that the correct population prior can be used.
        

    Args:
        
    """

    def __init__(
        self,
        acs: AnalysisContainerArray,
        wave_gen: GBGPU, # can be any waveform generator
        waveform_kwargs: Dict[str, Any],
        source_setups: Dict[str, BaseSourceConfig],
        branch_name_map: Dict[str, str],
        catalogues: List[NDArrayLike],
        snr_threshold: float = 7.0,
        num_repeats: int = 1,
        **kwargs
    ):
        Move.__init__(self, is_rj=True, **kwargs)

        self.acs = acs
        self.wave_gen = wave_gen
        self.waveform_kwargs = waveform_kwargs
        self.source_setups = source_setups
        self.branch_name_map = branch_name_map
        self.catalogues = catalogues
        self.nmodels = len(catalogues)
        
        self.N_tot_model = {}
        for model in range(self.nmodels):
            assert self.catalogues[model].shape[0] > self.catalogues[model].shape[1], (
                "The number of sources should be the first axis of the catalog."
            )
            self.N_tot_model[model] = self.catalogues[model].shape[0]
        
        self.snr_threshold = snr_threshold
        self.num_repeats = num_repeats
        
        self.first_catalogue_itteration = True      


    def setup(self, coords):
        
        if self.first_catalogue_itteration:
            logger.info("Starting first snr loop of catalogues")           
            
            max_logl_walker = np.argmax(self.acs.likelihood()).item()
            acs_max_logl = AnalysisContainerArray([deepcopy(self.acs[max_logl_walker])], gpus=self.acs.gpus)
            xp = acs_max_logl.xp
            f_min_global = acs_max_logl.settings.f_arr.min().get()
            df = acs_max_logl.settings.df 
            oversample = self.waveform_kwargs["oversample"] if "oversample" in self.waveform_kwargs else 1
            f_min_filter = f_min_global + get_N(1e-30, f_min_global, 1/df, oversample) * df
            
            for i, catalogue in enumerate(self.catalogues):
                
                # filter for out_of_bounds; currently assume f0_idx = 1
                mask = catalogue[:,1] > f_min_filter
                catalogue = catalogue[mask, :]
                
                ncat = catalogue.shape[0]
                data_index = xp.asarray(
                    np.repeat(0, ncat), dtype=np.int32
                )
                
                self.wave_gen.get_ll(
                    catalogue, 
                    acs_max_logl.linear_data_arr, # we are only interested in h_h contribution for opt_snr, so this can be anything
                    acs_max_logl.linear_psd_arr, 
                    data_index=data_index, 
                    noise_index=data_index, 
                    data_length=acs_max_logl.data_length, 
                    data_splits=np.array([0]), 
                    phase_marginalize=False, 
                    **self.waveform_kwargs
                )
                h_h_raw = self.wave_gen.h_h 
                h_h = h_h_raw.get() if hasattr(h_h_raw, "get") else h_h_raw

                opt_snrs = np.sqrt(h_h.real)
                catalogue_filtered = catalogue[opt_snrs > 0.5] # TODO cut adjustable?
                logger.info(f"For catalogue {i}, {catalogue_filtered.shape[0]}/{ncat} binaries are above SNR=0.5")
                self.catalogues[i] = catalogue_filtered
            
            del acs_max_logl, data_index
            self.first_catalogue_itteration = False

        return
    
    def compute_source_contribution(
        self, 
        model_coords: NDArrayLike, 
        resolved_coords: NDArrayLike, 
        resolved_inds: NDArrayLike,
        stochastic_coords: NDArrayLike
    ) -> NDArrayLike:
        xp = self.acs.xp
        ntemps, nwalkers, nleaves_max, ndim_res = resolved_coords.shape
        _, _, _, ndim_stoc = stochastic_coords.shape

        resolved_priors = self.source_setups["resolved"].priors[self.branch_name_map["resolved"]]
        stochastic_priors = self.source_setups["stochastic"].priors[self.branch_name_map["stochastic"]]
        
        model_coords_2d = model_coords[..., 0, 0].astype(int)
        
        resolved_reshaped = xp.asarray(resolved_coords.reshape(-1, ndim_res))
        model_flat_1 = xp.asarray(
            np.broadcast_to(
                model_coords_2d[:,:, np.newaxis], 
                (ntemps, nwalkers, nleaves_max)
            ).flatten()
        )
        inds_flat = resolved_inds.flatten()
        resolved_in = resolved_reshaped[inds_flat, :]
        model_flat_1 = model_flat_1[inds_flat]
        
        stochastic_reshaped = xp.asarray(stochastic_coords.reshape(-1, ndim_stoc))
        model_flat_2 = xp.asarray(model_coords_2d.flatten())# [:,None]
        
        active_pdfs = resolved_priors.logpdf(resolved_in, model_index=model_flat_1)
        resolved_pdfs = xp.zeros(ntemps * nwalkers * nleaves_max, dtype=xp.float64)
        resolved_pdfs[inds_flat] = active_pdfs
        resolved_pdfs = resolved_pdfs.reshape((ntemps, nwalkers, nleaves_max))   
        logp_resolved = resolved_pdfs.sum(axis=-1)

        logp_stochastic = xp.asarray(stochastic_priors.logpdf(
            stochastic_reshaped, 
            model_index=model_flat_2
        ).reshape((ntemps, nwalkers)))
        
        return logp_resolved + logp_stochastic
    
    
    def compute_number_contribution(
        self,
        model_coords: NDArrayLike,
        num_resolved_sources: NDArrayLike
    ) -> NDArrayLike:
        xp = self.acs.xp
        nwalkers = len(self.acs)

        Nexpected_resolved_array = np.zeros((self.nmodels, nwalkers))
        for i, catalogue in enumerate(self.catalogues):
            ncat = catalogue.shape[0]
            
            coords_in = xp.asarray(
                np.broadcast_to(catalogue, (nwalkers,)+catalogue.shape).reshape(-1, 9)
            )
            data_index = xp.asarray(
                np.repeat(np.arange(nwalkers), ncat), dtype=np.int32
            )
            # we are only interested in h_h contribution for opt_snr, so this can be anything
            self.wave_gen.get_ll(
                coords_in, 
                self.acs.linear_data_arr, 
                self.acs.linear_psd_arr, 
                data_index=data_index, 
                noise_index=data_index, 
                data_length=self.acs.data_length, 
                data_splits=np.array([0]), 
                phase_marginalize=False, 
                **self.waveform_kwargs
            )
            h_h_raw = self.wave_gen.h_h 
            h_h = h_h_raw.get() if hasattr(h_h_raw, "get") else h_h_raw

            opt_snrs = np.sqrt(h_h.real).reshape(nwalkers, ncat)
            Nexpected_resolved_array[i] = np.sum(opt_snrs > self.snr_threshold, axis=-1)
        
        model_coords_2d = model_coords[..., 0, 0].astype(int)
        
        term_1 = Nexpected_resolved_array[model_coords_2d, np.arange(nwalkers)]
        
        N_tot_array = np.array([self.N_tot_model[m] for m in range(self.nmodels)])
        term_2 = N_tot_array[model_coords_2d]
        
        return - term_1 + num_resolved_sources * np.log(term_2)
    
    
    def get_proposal(self, coords, random, supps=None, branch_supps=None):
        if self.nmodels <= 1:
            raise ValueError("nmodels must be strictly greater than 1 to propose a change.")
        
        ntemps, nwalkers, _, _ = coords.shape

        # all leaves of each walker and temperature have the same model
        current_indices = coords[..., 0, 0].copy().astype(int)
        
        proposed_indices = random.randint(1, self.nmodels, size=(ntemps, nwalkers))
        
        new_indices = (current_indices + proposed_indices) % self.nmodels
        
        new_coords = coords.copy()
        new_coords[..., 0, 0] = new_indices
        
        factors = np.zeros((ntemps, nwalkers))
        
        return new_coords, factors
        
    
    def run_hyper_tempering(self, state):
        """
        This tempering function does the tempering operations of the usual temperature control.
        However, it overrides share_temperature=False (represented by skip_swap_branches) 
        for the relevant branches in self.branch_name_map and 'hyper'.
        Thus, this swaps the coords for all relevant branches when the swap is accepted.
        """
        if self.temperature_control is None or self.temperature_control.ntemps <= 1 or self.prevent_swaps:
            return state

        tc = self.temperature_control
        
        target_branches = list(self.branch_name_map.values())
        if "hyper" not in target_branches:
            target_branches.append("hyper")
            
        original_skip_swap = list(tc.skip_swap_branches)
        
        try:
            # This makes do_swaps_indexing() swap their coordinates and parameters
            tc.skip_swap_branches = [
                branch for branch in original_skip_swap 
                if branch not in target_branches
            ]
            swapped_state = tc.temper_comps(state, adapt=False)
            
        finally:
            tc.skip_swap_branches = original_skip_swap
            
        return swapped_state
    
    
    def propose(self, model, state):
        
        self.setup(state.branches_coords)
        
        all_branch_names = list(state.branches.keys())
        
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape
        
        # setup supplemental information
        if not np.all(
            np.asarray(list(state.branches_supplemental.values())) == None
        ):
            new_branch_supps = deepcopy(state.branches_supplemental)
        else:
            new_branch_supps = None

        if state.supplemental is not None:
            new_supps = deepcopy(state.supplemental)
        else:
            new_supps = None
            
        self.current_model = model
        self.current_state = state

        resolved_inds = state.branches[self.branch_name_map["resolved"]].inds[:]
        num_resolved_sources = deepcopy(resolved_inds.sum(axis=-1))
        old_coords_model = deepcopy(state.branches_coords["hyper"])    
        coords_resolved = deepcopy(state.branches_coords[self.branch_name_map["resolved"]])
        coords_stochastic = deepcopy(state.branches_coords[self.branch_name_map["stochastic"]])

        # calculate prior contribution old state
        logp_source_prev = self.compute_source_contribution(
            old_coords_model, 
            coords_resolved, 
            resolved_inds, 
            coords_stochastic
        ) # self.prior at init
        logp_source_prev = logp_source_prev.get() if hasattr(logp_source_prev, "get") else logp_source_prev
        
        logp_number_prev = self.compute_number_contribution(
            old_coords_model, 
            num_resolved_sources
        ) # Ntot(M), catalog and psd set at init
        
        logp_prev = logp_source_prev + logp_number_prev

        # get new model coords
        new_coords_model, factors = self.get_proposal(
            old_coords_model,
            model.random,
            supps=new_supps,
            branch_supps=new_branch_supps,
        )
        
        # calculate prior contribution new state
        logp_source_curr = self.compute_source_contribution(
            new_coords_model, 
            coords_resolved, 
            resolved_inds,
            coords_stochastic
        ) # self.prior at init
        logp_source_curr = logp_source_curr.get() if hasattr(logp_source_curr, "get") else logp_source_curr

        logp_number_curr = self.compute_number_contribution(
            new_coords_model, 
            num_resolved_sources
        ) # Ntot(M), catalog and psd set at init
        
        logp_curr = logp_source_curr + logp_number_curr
        
        # acceptance fraction
        delta_logp = factors + logp_curr - logp_prev
        accepted = delta_logp > np.log(model.random.rand(ntemps, nwalkers))
        
        
        logger.debug(f"Old model coords are     {old_coords_model[0,:,0,0]}")
        logger.debug(f"Proposed model coords are {new_coords_model[0,:,0,0]}")
        model_coords_tmp = deepcopy(old_coords_model)
        model_coords_tmp[accepted] = new_coords_model[accepted]
        logger.debug(f"New model coords are     {model_coords_tmp[0,:,0,0]}")
        
        new_state = GFState(state, copy=True)
        new_state.branches_coords["hyper"][accepted] = new_coords_model[accepted]
        assert new_state.log_prior is not None
        new_state.log_prior[:] = logp_curr

        new_state = self.run_hyper_tempering(new_state)
            
        # add to move-specific accepted information
        self.accepted += accepted
        self.num_proposals += 1

        return new_state, accepted
        
   