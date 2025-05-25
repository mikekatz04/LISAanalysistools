import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from eryn.state import State as eryn_State
from eryn.state import Branch as eryn_Branch

def return_x(x):
    return x

class GBState(eryn_State):

    # copy this still for each. At general hdf5 function to deal with these setups rather than specific
    @property
    def band_initialized(self):
        if hasattr(self, "band_info") and "initialized" in self.band_info:
            return self.band_info["initialized"]
        else:
            return False

    def __init__(self, possible_state, band_info=None, copy=False, **kwargs):

        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
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

    def reset_backend(self, h5_group, h5_kwargs, nwalkers, *args, ntemps=1, **kwargs):

        assert self.band_initialized

        band_group = h5_group.create_group("gb_sub_state")

        band_group.create_dataset(
            "band_edges",
            data=self.band_info["band_edges"],
            **h5_kwargs
        )
        num_bands = self.band_info["num_bands"]
        band_group.attrs["num_bands"] = num_bands

        band_group.create_dataset(
            "band_temps",
            (0, num_bands, ntemps),
            maxshape=(None, num_bands, ntemps),
            **h5_kwargs
        )

        band_group.create_dataset(
            "band_swaps_proposed",
            (0, num_bands, ntemps - 1),
            maxshape=(None, num_bands, ntemps - 1),
            **h5_kwargs
        )

        band_group.create_dataset(
            "band_swaps_accepted",
            (0, num_bands, ntemps - 1),
            maxshape=(None, num_bands, ntemps - 1),
            **h5_kwargs
        )

        band_group.create_dataset(
            "band_num_proposed",
            (0, num_bands, ntemps),
            maxshape=(None, num_bands, ntemps),
            **h5_kwargs
        )

        band_group.create_dataset(
            "band_num_accepted",
            (0, num_bands, ntemps),
            maxshape=(None, num_bands, ntemps),
            **h5_kwargs
        )

        band_group.create_dataset(
            "band_num_proposed_rj",
            (0, num_bands, ntemps),
            maxshape=(None, num_bands, ntemps),
            **h5_kwargs
        )

        band_group.create_dataset(
            "band_num_accepted_rj",
            (0, num_bands, ntemps),
            maxshape=(None, num_bands, ntemps),
            **h5_kwargs
        )

        band_group.create_dataset(
            "band_num_binaries",
            (0, ntemps, nwalkers, num_bands),
            maxshape=(None, ntemps, nwalkers, num_bands),
            **h5_kwargs
        )

    def grow_backend(self, h5_group, ngrow, *args):
        band_group = h5_group["gb_sub_state"]
        for key in band_group:
            if key == "band_edges":
                continue
            band_group[key].resize(ngrow, axis=0)

    def save_step(self, iteration, h5_group, state, *args, **kwargs):
        # make sure the backend has all the information needed to store everything
        gb_group = h5_group["gb_sub_state"]
        for key in [
            "num_bands",
        ]:
            if not hasattr(self, key):
                setattr(self, key, gb_group.attrs[key])

        # branch-specific
        for name, dat in state.sub_states["gb"].band_info.items():
            if not isinstance(dat, np.ndarray) or name == "band_edges":
                continue
            gb_group[name][iteration] = dat

        # reset the counter for band info
        state.sub_states["gb"].reset_band_counters()

    @property
    def reset_kwargs(self):
        # TODO: this okay for future?
        return dict(
            num_bands=len(self.band_info["band_edges"]) - 1,
            band_edges=self.band_info["band_edges"],
        )

class MBHState(eryn_State):
    remove_kwargs = ["betas_all"]
    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            self.betas_all = dc(possible_state.betas_all)
        else:
            self.betas_all = betas_all

    @property
    def reset_kwargs(self):
        # TODO: this okay for future?
        return dict(
            num_mbhs=self.betas_all.shape[0]
        )


class EMRIState(eryn_State):
    remove_kwargs = ["betas_all"]
    def __init__(self, possible_state, betas_all=None, copy=False, **kwargs):
        if isinstance(possible_state, self.__class__):
            dc = deepcopy if copy else return_x
            self.betas_all = dc(possible_state.betas_all)
        else:
            self.betas_all = betas_all

    @property
    def reset_kwargs(self):
        # TODO: this okay for future?
        return dict(
            num_emris=self.betas_all.shape[0]
        )

class GFState(eryn_State):
    # TODO: bandaid fix this
    def __init__(self, possible_state, *args, is_eryn_state_input:bool=False, sub_state_bases: dict=None, **kwargs):
        
        eryn_State.__init__(self, possible_state, *args, **kwargs)
        self.sub_states = {}
        if isinstance(possible_state, type(self)) and not is_eryn_state_input:
            self.sub_state_bases = possible_state.sub_state_bases
            for name in self.branches:
                sub_state_base = self.sub_state_bases.get(name, None)
                if sub_state_base is not None:
                    self.sub_states[name] = sub_state_base(
                        possible_state.sub_states[name],
                        *args,
                        **kwargs
                    )
                else:
                    self.sub_states[name] = None

        else:
            self.sub_state_bases = sub_state_bases
            for name in self.branches:
                sub_state_base = sub_state_bases.get(name, None)
                if sub_state_base is not None:
                    self.sub_states[name] = sub_state_base(
                        possible_state, # this is just coords in the first input
                        *args,
                        **kwargs
                    )
                else:
                    self.sub_states[name] = None
                
        # elif sub_state_bases is None and is_eryn_state_input:
        #     raise ValueError
        
        # elif is_eryn_state_input:
        #     self.sub_state_bases = sub_state_bases
        #     for name in self.branches:
        #         sub_state_base = sub_state_bases.get(name, None)
        #         if sub_state_base is not None:
        #             self.sub_states[name] = sub_state_base(
        #                 None,
        #                 *args, 
        #                 **kwargs
        #             )
        #         else:
        #             self.sub_states[name] = None


class AllGFBranchInfo:
    def __init__(self, branch_1, branch_2):
        
        for key in ["name", "ndims", "nleaves_max", "nleaves_min", "branch_state", "branch_backend"]:
            if isinstance(branch_1, AllGFBranchInfo) and isinstance(branch_2, AllGFBranchInfo):
                if key == "name":
                    self.branch_names = branch_1.branch_names + branch_2.name
                    continue
                setattr(self, key, {**getattr(branch_1, key), **getattr(branch_2, key)})
                
            elif isinstance(branch_1, GFBranchInfo) and isinstance(branch_2, GFBranchInfo):
                if key == "name":
                    self.branch_names = [branch_1.name, branch_2.name]
                    continue
                setattr(self, key, {branch_1.name: getattr(branch_1, key), branch_2.name: getattr(branch_2, key)})
            else:
                if not isinstance(branch_2, GFBranchInfo):
                    # switch so all branch is in position 1
                    tmp = branch_1
                    branch_1 = branch_2
                    branch_2 = tmp
                if key == "name":
                    self.branch_names = branch_1.branch_names + [branch_2.name]
                    continue
                setattr(self, key, {**getattr(branch_1, key), branch_2.name: getattr(branch_2, key)})
        
    def __add__(self, branch_2):
        return AllGFBranchInfo(self, branch_2)
    
    @property
    def ndims(self):
        return self._ndims
    
    @ndims.setter
    def ndims(self, ndims):
        assert isinstance(ndims, dict)
        self._ndims = ndims

    @property
    def branch_names(self):
        return self._branch_names
    
    @branch_names.setter
    def branch_names(self, branch_names):
        assert isinstance(branch_names, list)
        self._branch_names = branch_names

    @property
    def nleaves_max(self):
        return self._nleaves_max
    
    @nleaves_max.setter
    def nleaves_max(self, nleaves_max):
        assert isinstance(nleaves_max, dict)
        self._nleaves_max = nleaves_max

    @property
    def nleaves_min(self):
        return self._nleaves_min
    
    @nleaves_min.setter
    def nleaves_min(self, nleaves_min):
        assert isinstance(nleaves_min, dict)
        self._nleaves_min = nleaves_min

    @property
    def branch_state(self):
        return self._branch_state

    @branch_state.setter
    def branch_state(self, branch_state):
        self._branch_state = branch_state

    @property
    def branch_backend(self):
        return self._branch_backend

    @branch_backend.setter
    def branch_backend(self, branch_backend):
        self._branch_backend = branch_backend

from eryn.backends import backend as eryn_Backend

@dataclass
class GFBranchInfo:
    name: str
    ndims: int
    nleaves_max: int
    nleaves_min: int
    branch_state: eryn_State = None
    branch_backend: eryn_Backend = None
    
    def __add__(self, branch_2):
        return AllGFBranchInfo(self, branch_2)



    

