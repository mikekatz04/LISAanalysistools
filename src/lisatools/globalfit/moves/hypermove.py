import numpy as np
from copy import deepcopy
from typing import Dict, Any, List, Tuple
from eryn.moves import Move
from eryn.state import State
from eryn.model import Model
from .globalfitmove import GlobalFitMove
from ...analysiscontainer import AnalysisContainerArray
from ..priors.sourceconfigs import BaseSourceConfig
from ...utils.typing import NDArrayLike
from gbgpu.gbgpu import GBGPU



class HyperMove(GlobalFitMove, Move):
    """
    A model informed reversible jump move to jump between discrete models.
    The setup: 
        This move handles the changing in model index. This model index is propagated to other relevant 
        branches in the model using the ``model_index`` key in the branch supplemental information. This move is 
        designed to be used with a custom proposal that proposes a change in the model index. The branch supplemental
        informatin of each breach is propagated to prior calculation such that the correct population prior can be used.
        

    Args:
        nleaves_max (dict): Maximum number(s) of leaves for each model.
            Keys are ``branch_names`` and values are ``nleaves_max`` for each branch.
            This is a keyword argument, nut it is required.
        nleaves_min (dict): Minimum number(s) of leaves for each model.
            Keys are ``branch_names`` and values are ``nleaves_min`` for each branch.
            This is a keyword argument, nut it is required.
        tune (bool, optional): If True, tune proposal. (Default: ``False``)
        fix_change (int or None, optional): Fix the change in the number of leaves. Make them all
            add a leaf or remove a leaf. This can be useful for some search functions. Options
            are ``0`` or ``1``. (default: ``None``)

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
        self.snr_threshold = snr_threshold
        self.num_repeats = num_repeats


    def setup(self) -> None:
        """Calculates the expected number of resolved sources for the current model and sets up 
        any necessary information for the proposal. This is called at the beginning of each proposal
        step, so it can be used to adapt the proposal based on the current state of the sampler.

        Args:
            branches_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.

        """
        Nexpected_resolved_dict = {}
        for i, catalogue in enumerate(self.catalogues):
            self.wave_gen.run_wave(*catalogue, **self.waveform_kwargs)
            data_index = ... # setup data_index
            acs = ... # setup the analysis container array for this catalogue and the current data and psd
            # new analysis containers should be generated (or something similar to acs.linear_data_arr to save memory). 
            # This can be done with self.wave_gen.generate_global_template() (see gb moves file)
            # psd should be grabbed by usual acs
            self.wave_gen.get_ll(
                catalogue, 
                acs.linear_data_arr, 
                acs.linear_psd_arr, 
                data_index=data_index, 
                noise_index=data_index, 
                data_length=acs.data_length, 
                data_splits=np.array([0]), 
                phase_marginalize=False, 
                **self.waveform_kwargs
            )
            opt_snrs = gb.h_h.real ** (1 / 2)
            Nexpected_resolved = np.sum(opt_snrs > self.snr_threshold)
            Nexpected_resolved_dict[i] = Nexpected_resolved
            
        self.Nexpected_resolved_dict = Nexpected_resolved_dict
        
        return   


    def get_model_change_proposal(
        self, 
        coords: np.ndarray, 
        random: np.random.RandomState, 
        nmodels: int, 
    ) -> np.ndarray:       
        """Helper function for changing the model index.
        
        This proposal strictly selects a DIFFERENT model uniformly, perserving detailed balance.

        Args:
            coords (np.ndarray): The coordinates of all walkers for this specific branch with shape 
                ``(ntemps, nwalkers, nleaves_max, ndims)``.
            random (object): Current random state of the sampler.
            nmodels (int): The total number of models, must be greater than 1.

        Returns:
            np.ndarray: A new array of model indices.
        
        """
        if nmodels <= 1:
            raise ValueError("nmodels must be strictly greater than 1 to propose a change.")
        
        ntemps, nwalkers, _, _ = coords.shape

        # all leaves of each walker and temperature have the same model
        current_indices = coords[..., 0, 0].copy().astype(int)
        
        proposed_indices = random.randint(1, nmodels, size=(ntemps, nwalkers))
        
        new_indices = (current_indices + proposed_indices) % nmodels
        
        new_coords = coords.copy()
        new_coords[..., 0, 0] = new_indices[..., np.newaxis]
                
        return new_coords
        
    def get_pop_posterior(
        self,
        branches_coords: Dict[str, np.ndarray], 
        branches_inds: Dict[str, np.ndarray], 
        branch_name_map: Dict[str, str], 
        source_setups: Dict[str, BaseSourceConfig],
        snr_array: np.ndarray,          # <-- Passed in from the sampler state
        N_tot_dict: Dict[int, float],   # <-- N_tot(M) for each model
    ) -> np.ndarray:
        """
        Calculates the exact population posterior derived from the marginalized RJ-MCMC formalism.
        """
        # 1. Extract the current model indices from the hyper branch
        # Shape: (ntemps, nwalkers)
        model_indices = branches_coords["hyper"][..., 0, 0].astype(int)
        
        # 2. Extract GB parameters and active indices
        gb_coords = branches_coords["gb"]
        gb_inds = branches_inds["gb"]
        k1 = gb_inds.sum(axis=-1)  # Number of resolved sources per walker (ntemps, nwalkers)
        
        # 3. Get the expected total and resolved counts dynamically
        # Map the model indices to the constants
        N_tot = np.vectorize(N_tot_dict.get)(model_indices)
        N_1 = np.vectorize(self.Nexpected_resolved_dict.get)(model_indices)

        # 4. Evaluate the modified Poisson term: k1 * ln(N_tot) - N_1
        log_poisson_mod = k1 * np.log(N_tot) - N_1

        # 5. Evaluate the Normalizing Flow (p_pop) and Resolvability (alpha)
        gb_prior = source_setups["gb"].priors["gb"]
        resolv_prior = source_setups["gb"].priors["resolv_gb"]
        
        # Evaluate for all leaves. Eryn priors return -inf for invalid coords, 
        # but we multiply by gb_inds anyway to zero out inactive leaves.
        logp_pop = gb_prior.logpdf(gb_coords, model_index=model_indices)
        logp_res = resolv_prior.logpdf(snr_array)
        
        # Sum over the active leaves (axis=-1)
        logp_sources = np.sum((logp_pop + logp_res) * gb_inds, axis=-1)
        
        # 6. Total Population Posterior
        total_pop_prior = log_poisson_mod + logp_sources
        
        return total_pop_prior

    def get_proposal(
        self, 
        all_coords: Dict[str, np.ndarray], 
        all_inds: Dict[str, np.ndarray], 
        nleaves_min_all: Dict[str, int], 
        nleaves_max_all: Dict[str, int], 
        random: np.random.RandomState, 
        **kwargs: Any
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the current
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            nleaves_min_all (dict): Minimum values of leaf ount for each model. Must have same order as ``all_coords``.
            nleaves_max_all (dict): Maximum values of leaf ount for each model. Must have same order as ``all_coords``.
            random (object): Current random state of the sampler.
            **kwargs (ignored): For modularity.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        Raises:
            ValueError: leave set-up is incorrect.

        """
        # prepare the output dictionaries
        q = {}
        new_inds = all_inds.copy()

        # loop over the models included here
        assert len(nleaves_min_all)
        assert len(all_coords.keys()) == len(nleaves_max_all.keys())
        for i, (name, coords) in enumerate(zip(all_coords.keys(), all_coords.values())):
            # check for proper RJ setup
            nleaves_max = nleaves_max_all[name]
            nleaves_min = nleaves_min_all[name]
            if nleaves_min == nleaves_max:
                continue
            elif nleaves_min > nleaves_max:
                raise ValueError("nleaves_min is greater than nleaves_max. Not allowed.")
            
            # get new coordinates
            q[name] = self.get_model_change_proposal(
                coords, random, self.nmodels
            )
            
            ntemps, nwalkers, _, _ = coords.shape
            
            if i == 0:
                factors = np.zeros((ntemps, nwalkers))
                
            #! since the usual implementation is performed using a uniform proposal 
            #! in the model index, the factors are zero. This can be changed in the
            #! future if necessary

        return q, new_inds, factors
    
        
    def propose(
        self, 
        model: Model, 
        state: State
    ) -> Tuple[State, Any]:
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        self.setup()
        
        current_snrs = state.branches_supplemental["gb"]["snr"]
        
        try:
            prev_tot_pop_prior = state.branches_supplemental["hyper"]["total_pop_prior"]
        except AttributeError:
            prev_tot_pop_prior = self.get_pop_posterior(
                state.branches_coords, 
                state.branches_inds, 
                self.branch_name_map, 
                self.source_setups,
                current_snrs,
                self.N_tot_dict
            )
        
        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        all_branch_names = list(state.branches.keys())

        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # gibbs sampling is only over branches so pick out that info
            coords_propose_in = {
                key: state.branches_coords[key] for key in branch_names_run
            }
            inds_propose_in = {
                key: state.branches_inds[key] for key in branch_names_run
            }
            branches_supp_propose_in = {
                key: state.branches_supplemental[key] for key in branch_names_run
            }

            if len(list(coords_propose_in.keys())) == 0:
                raise ValueError(
                    "Right now, no models are getting a reversible jump proposal. Check nleaves_min and nleaves_max or do not use rj proposal."
                )

            # get min and max leaf information
            nleaves_max_all = {brn: self.nleaves_max[brn] for brn in branch_names_run}
            nleaves_min_all = {brn: self.nleaves_min[brn] for brn in branch_names_run}

            self.current_model = model
            self.current_state = state
            # propose new sources and coordinates
            q, new_inds, factors = self.get_proposal(
                coords_propose_in,
                inds_propose_in,
                nleaves_min_all,
                nleaves_max_all,
                model.random,
                branch_supps=branches_supp_propose_in,
                supps=state.supplemental,
            )

            new_tot_pop_prior = self.get_pop_posterior(
                q, 
                new_inds, 
                self.branch_name_map, 
                self.source_setups,
                current_snrs, # SNR doesn't change during an Out-Model move!
                self.N_tot_dict
            )
            factors += (new_tot_pop_prior - prev_tot_pop_prior)
            
            branches_supps_new = {
                key: item for key, item in branches_supp_propose_in.items()
            }
            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # put back any branches that were left out from Gibbs split
            for name, branch in state.branches.items():
                if name not in q:
                    q[name] = state.branches[name].coords[:].copy()
                if name not in new_inds:
                    new_inds[name] = state.branches[name].inds[:].copy()

                if name not in branches_supps_new:
                    branches_supps_new[name] = state.branches_supplemental[name]

            # fix any ordering issues
            q, new_inds, branches_supps_new = self.ensure_ordering(
                list(state.branches.keys()), q, new_inds, branches_supps_new
            )

            # setup supplemental information

            if state.supplemental is not None:
                # TODO: should there be a copy?
                new_supps = deepcopy(state.supplemental)

            else:
                new_supps = None

            logp = np.zeros((ntemps, nwalkers))
            
            self.fix_logp_gibbs(branch_names_run, inds_run, logp, new_inds)

            # Compute the ln like of the proposed position.
            logl, new_blobs = model.compute_log_like_fn(
                q,
                inds=new_inds,
                logp=logp,
                supps=new_supps,
                branch_supps=branches_supps_new,
            )

            # posterior and previous info
            logP = self.compute_log_posterior(logl, logp)

            prev_logl = state.log_like

            prev_logp = state.log_prior

            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # acceptance fraction
            lnpdiff = factors + logP - prev_logP

            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))
            
            if new_supps is not None and "hyper" in new_supps:
                # Store the updated population prior for accepted walkers
                updated_pop_prior = np.where(accepted, new_tot_pop_prior, prev_tot_pop_prior)
                new_supps["hyper"]["total_pop_prior"] = updated_pop_prior
                
            # update with new state
            new_state = GFState(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=None,
                inds=new_inds,
                supplemental=new_supps,
                branch_supplemental=branches_supps_new,
            )
            state = self.update(state, new_state, accepted)


        if self.temperature_control is not None and not self.prevent_swaps:
            state = self.temperature_control.temper_comps(state, adapt=False)

        # add to move-specific accepted information
        self.accepted += accepted
        self.num_proposals += 1

        return state, accepted
    
    

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

    num_resolved_sources = deepcopy(state.branches[self.branch_name_map["resolved"]].inds[:].sum(axis=-1))
    old_coords_model = deepcopy(state.branches_coords["hyper"])    
    coords_resolved = deepcopy(state.branches_coords[self.branch_name_map["resolved"]])
    coords_stochastic = deepcopy(state.branches_coords[self.branch_name_map["stochastic"]])

    # calculate prior contribution old state
    logp_source_prev = self.compute_source_contribution(old_coords_model, coords_resolved, coords_stochastic) # self.prior at init
    logp_number_prev = self.compute_number_contribution(old_coords_model, num_resolved_sources) # Ntot(M), catalog and psd set at init
    
    logp_prev = logp_source_prev + logp_number_prev

    # get new model coords
    new_coords_model, factors = self.get_proposal(
        old_coords_model,
        model.random,
        supps=new_supps,
        branch_supps=new_branch_supps,
    )
    
    # calculate prior contribution new state
    logp_source_curr = self.compute_source_contribution(new_coords_model, coords_resolved, coords_stochastic) # self.prior at init
    logp_number_curr = self.compute_number_contribution(new_coords_model, num_resolved_sources) # Ntot(M), catalog and psd set at init
    
    logp_curr = logp_source_curr + logp_source_curr
    
    # acceptance fraction
    delta_logp = factors + logp_curr - logp_prev
    
    accepted = delta_logp > np.log(model.random.rand(ntemps, nwalkers))
    
    new_coords = deepcopy(state.coords)
    new_coords["hyper"] = new_coords_model
    
    # TODO check agains psdmove or gbmove
    new_state = GFState( 
        new_coords,
        log_like=state.log_like,
        log_prior=logp_curr, #? i think this only works if everything else is model independent
        blobs=None,
        inds=state.inds,
        supplemental=new_supps,
        branch_supplemental=new_branch_supps,
    )

    if self.temperature_control is not None and not self.prevent_swaps:
        new_state = self.temperature_control.temper_comps(new_state, adapt=False)
        
    # add to move-specific accepted information
    self.accepted += accepted
    self.num_proposals += 1

    return new_state, accepted
            
            