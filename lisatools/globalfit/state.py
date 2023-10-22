import numpy as np
from copy import deepcopy

from eryn.state import State as eryn_State

class State(eryn_State):

    @property
    def band_initialized(self):
        if hasattr(self, "band_info") and "initialized" in self.band_info:
            return self.band_info["initialized"]
        else:
            return False

    def __init__(self, possible_state, *args, band_info=None, **kwargs):

        super().__init__(possible_state, *args, **kwargs)

        if isinstance(possible_state, self.__class__):
            copy = "copy" in kwargs and kwargs["copy"] is True

            dc = deepcopy if copy else lambda x: x
            if possible_state.band_initialized and hasattr(possible_state, "band_info"):
                self.band_info = dc(possible_state.band_info)
        elif band_info is not None:
            self.band_info = band_info

    def initialize_band_information(self, nwalkers, ntemps, band_edges, band_temps):
        
        if not hasattr(self, "intialized"):
            self.band_info = {}
            self.band_info["nwalkers"], self.band_info["ntemps"], self.band_info["band_edges"] = nwalkers, ntemps, band_edges

            self.band_info["num_bands"] =  len(self.band_info["band_edges"]) - 1

            assert band_temps.shape == (self.band_info["num_bands"], self.band_info["ntemps"])
            self.band_info["band_temps"] = band_temps

            self.band_info["band_swaps_proposed"] =  np.zeros((self.band_info["num_bands"], self.band_info["ntemps"] - 1), dtype=int)
            self.band_info["band_swaps_accepted"] =  np.zeros((self.band_info["num_bands"], self.band_info["ntemps"] - 1), dtype=int)

            self.band_info["band_num_proposed"] =  np.zeros((self.band_info["num_bands"], self.band_info["ntemps"]), dtype=int)
            self.band_info["band_num_accepted"] =  np.zeros((self.band_info["num_bands"], self.band_info["ntemps"]), dtype=int)
            
            self.band_info["band_num_proposed_rj"] =  np.zeros((self.band_info["num_bands"], self.band_info["ntemps"]), dtype=int)
            self.band_info["band_num_accepted_rj"] =  np.zeros((self.band_info["num_bands"], self.band_info["ntemps"]), dtype=int)

            self.band_info["band_num_binaries"] =  np.zeros((self.band_info["ntemps"], self.band_info["nwalkers"], self.band_info["num_bands"]), dtype=int)
            self.band_info["initialized"] =  True

        else:
            assert nwalkers == self.band_info["nwalkers"]
            assert ntemps == self.band_info["ntemps"]
            assert np.all(band_edges == self.band_info["band_edges"])

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

    def __init__(self, possible_state, *args, betas_all=None, **kwargs):

        super().__init__(possible_state, *args, **kwargs)

        if isinstance(possible_state, self.__class__):
            copy = "copy" in kwargs and kwargs["copy"] is True

            dc = deepcopy if copy else lambda x: x
            self.betas_all = dc(possible_state.betas_all)

        elif betas_all is not None:
            self.betas_all = betas_all

    