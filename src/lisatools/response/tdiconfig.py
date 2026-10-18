from multiprocessing.sharedctypes import Value
import numpy as np
from typing import Optional, List
import warnings
from typing import Tuple
from copy import deepcopy

import time
import h5py

from scipy.interpolate import CubicSpline

from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.utils.utility import AET

from .parallelbase import FastLISAResponseParallelModule

# Stock combinations of channel X in single-link measurements y_ij, each delayed by the links in
# links_for_delay; Y and Z follow by cyclic permutation. TDI 2.0 is TDI 1.0 plus the extra terms.
TDI_FIRST_GENERATION = [
    {"link": 13, "links_for_delay": [], "sign": +1},
    {"link": 31, "links_for_delay": [13], "sign": +1},
    {"link": 12, "links_for_delay": [13, 31], "sign": +1},
    {"link": 21, "links_for_delay": [13, 31, 12], "sign": +1},
    {"link": 12, "links_for_delay": [], "sign": -1},
    {"link": 21, "links_for_delay": [12], "sign": -1},
    {"link": 13, "links_for_delay": [12, 21], "sign": -1},
    {"link": 31, "links_for_delay": [12, 21, 13], "sign": -1},
]
TDI_SECOND_GENERATION_EXTRA = [
    {"link": 12, "links_for_delay": [13, 31, 12, 21], "sign": +1},
    {"link": 21, "links_for_delay": [13, 31, 12, 21, 12], "sign": +1},
    {"link": 13, "links_for_delay": [13, 31, 12, 21, 12, 21], "sign": +1},
    {"link": 31, "links_for_delay": [13, 31, 12, 21, 12, 21, 13], "sign": +1},
    {"link": 13, "links_for_delay": [12, 21, 13, 31], "sign": -1},
    {"link": 31, "links_for_delay": [12, 21, 13, 31, 13], "sign": -1},
    {"link": 12, "links_for_delay": [12, 21, 13, 31, 13, 31], "sign": -1},
    {"link": 21, "links_for_delay": [12, 21, 13, 31, 13, 31, 12], "sign": -1},
]
TDI_SECOND_GENERATION = TDI_FIRST_GENERATION + TDI_SECOND_GENERATION_EXTRA


def _tdi_unit_key(unit: dict) -> tuple:
    """A TDI unit in a form that compares equal for equal contributions."""
    return (
        int(unit["link"]),
        tuple(int(link) for link in unit["links_for_delay"]),
        float(unit["sign"]),
        unit.get("type", "delay"),
    )


def tdi_generation(tdi_combinations: List[dict]) -> Optional[int]:
    """1 or 2 when the combinations are those of the stock 1st or 2nd generation, else None.

    The order of the units does not matter, since a combination is their sum.
    """
    units = sorted(_tdi_unit_key(unit) for unit in tdi_combinations)
    for generation, stock in ((1, TDI_FIRST_GENERATION), (2, TDI_SECOND_GENERATION)):
        if units == sorted(_tdi_unit_key(unit) for unit in stock):
            return generation
    return None


class TDIConfig(FastLISAResponseParallelModule):
    """

    Args:
        tdi (str or list, optional): TDI setup. Currently, the stock options are
            :code:`'1st generation'` and :code:`'2nd generation'`. Or the user can provide
            a list of tdi_combinations of the form
            :code:`{"link": 12, "links_for_delay": [21, 13, 31], "sign": 1, "type": "delay"}`.
            :code:`'link'` (`int`) the link index (12, 21, 13, 31, 23, 32) for the projection (:math:`y_{ij}`).
            :code:`'links_for_delay'` (`list`) are the link indexes as a list used for delays
            applied to the link projections.
            ``'sign'`` is the sign in front of the contribution to the TDI observable. It takes the value of `+1` or `-1`.
            ``type`` is either ``"delay"`` or ``"advance"``. It is optional and defaults to ``"delay"``.
            (default: ``"1st generation"``)
    
    """

    @classmethod
    def supported_backends(cls):
        # Phase 3L.7l holdout (fixed 2026-06-05): use `_BACKEND_PREFIX`
        # ("lisatools" via FastLISAResponseParallelModule).
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED_WITH_JAX()]

    def __init__(self, tdi: str | List[dict], force_backend: Optional[str] = None):
        super().__init__(force_backend=force_backend)

        # setup the actual TDI combination
        if tdi == "1st generation":
            tdi_combinations = deepcopy(TDI_FIRST_GENERATION)
        elif tdi == "2nd generation":
            tdi_combinations = deepcopy(TDI_SECOND_GENERATION)

        elif isinstance(tdi, list):
            tdi_combinations = tdi

        else:
            raise ValueError(
                "tdi kwarg should be '1st generation', '2nd generation', or a list with a specific tdi combination."
            )
        
        self.tdi_combinations = tdi_combinations
        self.nchannels = 3

        self.pytdiconfig_args = [
            self.xp.asarray(self.unit_starts).copy(),
            self.xp.asarray(self.unit_lengths).copy(),
            self.xp.asarray(self.tdi_base_links).copy(),
            self.xp.asarray(self.tdi_link_combinations).copy(),
            self.xp.asarray(self.tdi_signs).copy(),
            self.xp.asarray(self.channels).copy(),
            self.num_units,
            self.nchannels
        ]

    @property
    def pytdiconfig(self) -> object:
        """C++ class"""
        if self._pytdiconfig_args is None:
            raise ValueError(
                "Asking for c++ class. Need to set linear_interp_setup = True when configuring."
            )
        self._pytdiconfig = self.backend.pyTDIConfig(*self._pytdiconfig_args)
        return self._pytdiconfig

    @property
    def pytdiconfig_args(self) -> tuple:
        """args for the c++ class."""
        return self._pytdiconfig_args

    @pytdiconfig_args.setter
    def pytdiconfig_args(self, pytdiconfig_args: tuple) -> None:
        self._pytdiconfig_args = pytdiconfig_args
        
    @property
    def generation(self) -> Optional[int]:
        """1 or 2 when the combinations are those of the stock 1st or 2nd generation, else None.

        The DFFR response has closed forms for these two and raises on any other combination.
        """
        return tdi_generation(self.tdi_combinations)

    @property
    def tdi_combinations(self) -> List:
        """TDI Combination setup"""
        return self._tdi_combinations

    @tdi_combinations.setter
    def tdi_combinations(self, tdi_combinations: List) -> None:
        """Set TDI combinations and fill out setup."""

        self._tdi_combinations = tdi_combinations
        tdi_base_links = []
        tdi_link_combinations = []
        tdi_signs = []
        tdi_operation_index = []
        channels = []

        tdi_index = 0
        for permutation_number in range(3):
            for tmp in tdi_combinations:
                tdi_base_links.append(
                    self._cyclic_permutation(tmp["link"], permutation_number)
                )
                tdi_signs.append(float(tmp["sign"]))
                channels.append(permutation_number)
                if len(tmp["links_for_delay"]) == 0:
                    tdi_link_combinations.append(-11)
                    tdi_operation_index.append(tdi_index)

                else:
                    for link_delay in tmp["links_for_delay"]:

                        tdi_link_combinations.append(
                            self._cyclic_permutation(link_delay, permutation_number)
                        )
                        tdi_operation_index.append(tdi_index)

                tdi_index += 1

        self.tdi_operation_index = self.xp.asarray(tdi_operation_index).astype(
            self.xp.int32
        )
        self.tdi_base_links = self.xp.asarray(tdi_base_links).astype(self.xp.int32)
        self.tdi_link_combinations = self.xp.asarray(tdi_link_combinations).astype(
            self.xp.int32
        )
        self.tdi_signs = self.xp.asarray(tdi_signs).astype(self.xp.float64)
        self.channels = self.xp.asarray(channels).astype(self.xp.int32)
        assert len(self.tdi_link_combinations) == len(self.tdi_operation_index)

        assert (
            len(self.tdi_base_links)
            == len(np.unique(self.tdi_operation_index))
            == len(self.tdi_signs)
            == len(self.channels)
        )

        self.num_units = int(self.tdi_operation_index.max() + 1)

        assert np.all(
            (np.diff(self.tdi_operation_index) == 0)
            | (np.diff(self.tdi_operation_index) == 1)
        )

        _, unit_starts, unit_lengths = np.unique(
            self.tdi_operation_index,
            return_index=True,
            return_counts=True,
        )

        self.unit_starts = unit_starts.astype(np.int32)
        self.unit_lengths = unit_lengths.astype(np.int32)

    def _cyclic_permutation(self, link, permutation):
        """permute indexes by cyclic permutation"""
        link_str = str(link)

        out = ""
        for i in range(2):
            sc = int(link_str[i])
            temp = sc + permutation
            if temp > 3:
                temp = temp % 3
            out += str(temp)

        return int(out)

    @property
    def ptr(self) -> int:
        """pointer to c++ class"""
        return self.pytdiconfig.ptr
    
    