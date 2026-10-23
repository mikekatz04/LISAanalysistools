"""Time-delay interferometry (TDI) configuration for the LISA response.

Holds :class:`TDIConfig`, which turns a TDI generation string or an explicit
list of delayed link terms into the flat index arrays the C++/CUDA response
kernels read (``pyTDIConfig`` / ``TDIConfigWrap``). In the global fit it is
built by :mod:`lisatools.globalfit.stock.erebor.source_runtime` (from the
general block's ``tdi_gen_str``) and passed to the source wrappers of
:mod:`lisatools.globalfit.stock.erebor.wrappers`; it is consumed by
:class:`~lisatools.response.directresponse.pyResponseTDI` and the
TDI-on-the-fly generators of :mod:`lisatools.response.tdionfly`. The 9mo /
12mo runs use ``"2nd generation"`` (XYZ TDI-2).
"""

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

from ..utils.parallelbase import LISAToolsParallelModule


class TDIConfig(LISAToolsParallelModule):
    """Time-delay interferometry combination (Michelson X, Y, Z) for the LISA response.

    The combination is given as the list of terms of the X channel; Y and Z
    are its cyclic permutations of the spacecraft indices (1 -> 2 -> 3 -> 1),
    so there are always three channels. ``"1st generation"`` is the 8-term
    first-generation Michelson X; ``"2nd generation"`` adds the 8 extra terms
    of second-generation X (16 terms). Each term is one link projection
    :math:`y_{ij}` with a sign and the ordered list of links whose delays are
    applied to it. The class flattens the 3 x ``len(terms)`` permuted terms
    ("units") into the index arrays the C++ ``pyTDIConfig`` class takes
    (:attr:`pytdiconfig_args`).

    Args:
        tdi: ``"1st generation"``, ``"2nd generation"``, or a list of term
            dicts for the X channel, each of the form
            ``{"link": 12, "links_for_delay": [21, 13, 31], "sign": 1}``.
            ``"link"`` (int) is the link index (12, 21, 13, 31, 23, 32) of the
            projection :math:`y_{ij}`; ``"links_for_delay"`` (list of int) the
            links whose delays are applied, in order (empty for none);
            ``"sign"`` (``+1`` or ``-1``) the sign of the term. Any other key
            (e.g. a ``"type"``) is ignored: every term is a delay. Required
            (no default).
        force_backend: Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda13x"``,
            ...; prefixed with ``"lisatools_"`` when needed). ``None``
            (default) takes the first available backend.

    Raises:
        ValueError: ``tdi`` is neither a supported string nor a list.

    Attributes:
        nchannels (int): ``3``.
        num_units (int): Number of (channel, term) units, ``3 * len(terms)``.
        tdi_base_links (xp.ndarray): ``(num_units,)`` int32 projected link per unit.
        tdi_link_combinations (xp.ndarray): int32 delay links, units
            concatenated (``-11`` for a unit with no delay).
        tdi_operation_index (xp.ndarray): int32 unit index of each entry of
            ``tdi_link_combinations``.
        tdi_signs (xp.ndarray): ``(num_units,)`` float64 signs.
        channels (xp.ndarray): ``(num_units,)`` int32 channel (0 X, 1 Y, 2 Z).
        unit_starts (xp.ndarray): int32 start of each unit in
            ``tdi_link_combinations``.
        unit_lengths (xp.ndarray): int32 number of delay entries per unit.

    Example:
        >>> from lisatools.response.tdiconfig import TDIConfig
        >>> tdi = TDIConfig("2nd generation", force_backend="cpu")
        >>> tdi.num_units, tdi.nchannels
        (48, 3)
        >>> terms = [
        ...     {"link": 12, "links_for_delay": [], "sign": 1},
        ...     {"link": 21, "links_for_delay": [12], "sign": -1},
        ... ]
        >>> TDIConfig(terms, force_backend="cpu").num_units
        6
    """

    @classmethod
    def supported_backends(cls):
        """Return the backend names this class can run on, in preference order.

        Returns:
            list[str]: ``"lisatools_<tag>"`` for ``cuda13x``, ``cuda12x``,
            ``cuda11x``, ``cpu`` and ``jax``.
        """
        # Phase 3L.7l holdout (fixed 2026-06-05): use `_BACKEND_PREFIX`
        # ("lisatools" via LISAToolsParallelModule).
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED_WITH_JAX()]

    def __init__(self, tdi: str | List[dict], force_backend: Optional[str] = None):
        super().__init__(force_backend=force_backend)

        # setup the actual TDI combination
        if tdi in ["1st generation", "2nd generation"]:
            # tdi 1.0
            tdi_combinations = [
                {"link": 13, "links_for_delay": [], "sign": +1},
                {"link": 31, "links_for_delay": [13], "sign": +1},
                {"link": 12, "links_for_delay": [13, 31], "sign": +1},
                {"link": 21, "links_for_delay": [13, 31, 12], "sign": +1},
                {"link": 12, "links_for_delay": [], "sign": -1},
                {"link": 21, "links_for_delay": [12], "sign": -1},
                {"link": 13, "links_for_delay": [12, 21], "sign": -1},
                {"link": 31, "links_for_delay": [12, 21, 13], "sign": -1},
            ]

            if tdi == "2nd generation":
                # tdi 2.0 is tdi 1.0 + additional terms
                tdi_combinations += [
                    {"link": 12, "links_for_delay": [13, 31, 12, 21], "sign": +1},
                    {"link": 21, "links_for_delay": [13, 31, 12, 21, 12], "sign": +1},
                    {
                        "link": 13,
                        "links_for_delay": [13, 31, 12, 21, 12, 21],
                        "sign": +1,
                    },
                    {
                        "link": 31,
                        "links_for_delay": [13, 31, 12, 21, 12, 21, 13],
                        "sign": +1,
                    },
                    {"link": 13, "links_for_delay": [12, 21, 13, 31], "sign": -1},
                    {"link": 31, "links_for_delay": [12, 21, 13, 31, 13], "sign": -1},
                    {
                        "link": 12,
                        "links_for_delay": [12, 21, 13, 31, 13, 31],
                        "sign": -1,
                    },
                    {
                        "link": 21,
                        "links_for_delay": [12, 21, 13, 31, 13, 31, 12],
                        "sign": -1,
                    },
                ]

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
        """object: A new backend ``pyTDIConfig`` C++ object built from :attr:`pytdiconfig_args`.

        A fresh object is constructed (and cached on ``_pytdiconfig``) on every
        access. Raises ``ValueError`` when the arguments are ``None`` (the
        message names a ``linear_interp_setup`` flag that this class does not
        have).
        """
        if self._pytdiconfig_args is None:
            raise ValueError(
                "Asking for c++ class. Need to set linear_interp_setup = True when configuring."
            )
        self._pytdiconfig = self.backend.pyTDIConfig(*self._pytdiconfig_args)
        return self._pytdiconfig

    @property
    def pytdiconfig_args(self) -> tuple:
        """list: Constructor arguments of the C++ class.

        ``[unit_starts, unit_lengths, tdi_base_links, tdi_link_combinations,
        tdi_signs, channels, num_units, nchannels]``, the arrays as copies in
        the backend's array module.
        """
        return self._pytdiconfig_args

    @pytdiconfig_args.setter
    def pytdiconfig_args(self, pytdiconfig_args: tuple) -> None:
        self._pytdiconfig_args = pytdiconfig_args
        
    @property
    def tdi_combinations(self) -> List:
        """list[dict]: The X-channel TDI term list; setting it rebuilds the index arrays."""
        return self._tdi_combinations

    @tdi_combinations.setter
    def tdi_combinations(self, tdi_combinations: List) -> None:
        """Set the TDI terms and rebuild the flat index arrays (see the class Attributes)."""

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
        """Cyclically permute the two spacecraft digits of a link index.

        Each digit ``d`` becomes ``d + permutation``, wrapped into 1..3.

        Args:
            link (int): Two-digit link index, e.g. ``12``.
            permutation (int): Shift, 0, 1 or 2 (the channel).

        Returns:
            int: The permuted link, e.g. ``12 -> 23`` for ``permutation=1``.
        """
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
        """int: Pointer of a newly built ``pyTDIConfig`` object (see :attr:`pytdiconfig`)."""
        return self.pytdiconfig.ptr
    
    