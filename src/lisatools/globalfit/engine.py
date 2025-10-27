from __future__ import annotations
from eryn.ensemble import EnsembleSampler
from collections import namedtuple
from typing import Optional

__all__ = ["GlobalFitInfo", "GlobalFitEngine"]


import typing 


from eryn.backends import backend as eryn_Backend
from eryn.state import State as eryn_State

import dataclasses
@dataclasses.dataclass
class EngineInfo:
    branch_names: typing.List[str]
    ndims: typing.Dict[str, int]
    nleaves_max: typing.Dict[str, int]
    nleaves_min: typing.Dict[str, int]
    branch_states: typing.Dict[str, eryn_State] = None
    branch_backends: typing.Dict[str, eryn_Backend] = None


GlobalFitInfo = namedtuple(
    "GlobalFitInfo",
    (
        "analysis_container_arr",
        "map_fn",
        "random",
    ),
)

class GlobalFitEngine(EnsembleSampler):

    def __init__(self, analysis_container_arr: AnalysisContainerArray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_container_arr = analysis_container_arr
        
    def get_model(self):
        """Get ``Model`` object from sampler

        The model object is used to pass necessary information to the
        proposals. This method can be used to retrieve the ``model`` used
        in the sampler from outside the sampler.

        Returns:
            :class:`Model`: ``Model`` object used by sampler.

        """
        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        # setup model framework for passing necessary items
        model = GlobalFitInfo(
            self.analysis_container_arr,
            map_fn,
            self._random,
        )
        return model
