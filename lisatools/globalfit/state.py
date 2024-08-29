import numpy as np
from copy import deepcopy

from eryn.state import State as eryn_State

class GBState(eryn_State):

    @property
    def band_initialized(self):
        if hasattr(self, "band_info") and "initialized" in self.band_info:
            return self.band_info["initialized"]
        else:
            return False

    def __init__(self, possible_state, band_info=None, copy=False, **kwargs):

        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else lambda x: x
            if possible_state.band_initialized and hasattr(possible_state, "band_info"):
                self.band_info = dc(possible_state.band_info)
        elif band_info is not None:
            self.band_info = band_info

    @property
    def band_info_keys(self):
        return ["initialized", "band_edges", "band_temps", 'band_swaps_proposed', 'band_swaps_accepted', 'band_num_proposed', 'band_num_accepted', 'band_num_proposed_rj', 'band_num_accepted_rj', 'band_num_binaries']
    @property
    def band_info(self):
        return self._band_info

    @band_info.setter
    def band_info(self, band_info):
        assert isinstance(band_info, dict)
        for key in self.band_info_keys:
            if key not in band_info and key != "initialized":
                raise ValueError(f"Missing required key: {key}, for band information.")
        self._band_info = band_info
        self._band_info["initialized"] = True

    def initialize_band_information(self, nwalkers, ntemps, band_edges, band_temps):
        
        if not hasattr(self, "intialized"):
            band_info = {}
            band_info["nwalkers"], band_info["ntemps"], band_info["band_edges"] = nwalkers, ntemps, band_edges

            band_info["num_bands"] =  len(band_info["band_edges"]) - 1

            assert band_temps.shape == (band_info["num_bands"], band_info["ntemps"])
            band_info["band_temps"] = band_temps

            band_info["band_swaps_proposed"] =  np.zeros((band_info["num_bands"], band_info["ntemps"] - 1), dtype=int)
            band_info["band_swaps_accepted"] =  np.zeros((band_info["num_bands"], band_info["ntemps"] - 1), dtype=int)

            band_info["band_num_proposed"] =  np.zeros((band_info["num_bands"], band_info["ntemps"]), dtype=int)
            band_info["band_num_accepted"] =  np.zeros((band_info["num_bands"], band_info["ntemps"]), dtype=int)
            
            band_info["band_num_proposed_rj"] =  np.zeros((band_info["num_bands"], band_info["ntemps"]), dtype=int)
            band_info["band_num_accepted_rj"] =  np.zeros((band_info["num_bands"], band_info["ntemps"]), dtype=int)

            band_info["band_num_binaries"] =  np.zeros((band_info["ntemps"], band_info["nwalkers"], band_info["num_bands"]), dtype=int)
            band_info["initialized"] =  True
            self.band_info = band_info

        else:
            assert nwalkers == band_info["nwalkers"]
            assert ntemps == band_info["ntemps"]
            assert np.all(band_edges == band_info["band_edges"])

    def update_band_information(self, band_temps, band_num_proposed, band_num_accepted, band_swaps_proposed, band_swaps_accepted,
                                band_num_binaries, is_rj):
        self.band_info["band_temps"][:] = band_temps
        self.band_info["band_num_binaries"][:] = band_num_binaries
        
        if not is_rj:
            self.band_info["band_num_proposed"] += band_num_proposed
            self.band_info["band_num_accepted"] += band_num_accepted
        else:
            self.band_info["band_num_proposed_rj"] += band_num_proposed
            self.band_info["band_num_accepted_rj"] += band_num_accepted

        self.band_info["band_swaps_proposed"] += band_swaps_proposed
        self.band_info["band_swaps_accepted"] += band_swaps_accepted

    def reset_band_counters(self):
        self.band_info["band_num_proposed"][:] = 0
        self.band_info["band_num_accepted"][:] = 0
        self.band_info["band_num_proposed_rj"][:] = 0
        self.band_info["band_num_accepted_rj"][:] = 0
        self.band_info["band_swaps_proposed"][:] = 0
        self.band_info["band_swaps_accepted"][:] = 0



class MBHState(eryn_State):
    remove_kwargs = ["betas_all"]
    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else lambda x: x
            self.betas_all = dc(possible_state.betas_all)

        elif betas_all is not None:
            self.betas_all = betas_all

class State(GBState, MBHState, eryn_State):
    # TODO: bandaid fix this

    remove_kwargs = ["betas_all", "band_info"]
    def __init__(self, possible_state, *args, mgh=None, **kwargs):
        MBHState.__init__(self, possible_state, *args, **kwargs)
        GBState.__init__(self, possible_state, *args, **kwargs)
        for key in self.remove_kwargs:
            if key in kwargs:
                kwargs.pop(key)
        eryn_State.__init__(self, possible_state, *args, **kwargs)
        
        if isinstance(possible_state, type(self)):
            if mgh is not None:
                raise ValueError("When inputing a State object, cannot provide mbh kwarg as well.")
            self.mgh = possible_state.mgh
        else:
            self.mgh = mgh

    def from_eryn(self, state):
        breakpoint()
        breakpoint()
