from __future__ import annotations
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List
from copy import deepcopy
import os
from lisatools.utils.utility import get_array_module

import math
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d as np_gaussian_filter1d
import matplotlib.pyplot as plt
from . import domains

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

except (ModuleNotFoundError, ImportError):
    import numpy as cp
    cp_gaussian_filter1d = np_gaussian_filter1d

from cudakima import AkimaInterpolant1D
NUM_SPLINE_THREADS = 256

from . import detector as lisa_models
from .utils.utility import AET, get_array_module
from .utils.constants import *
from .stochastic import (
    StochasticContribution,
    FittedHyperbolicTangentGalacticForeground,
    check_stochastic,
)
from .utils.parallelbase import LISAToolsParallelModule
from .detector import L1Orbits
from .domains import DomainSettingsBase

"""
The sensitivity code is heavily based on an original code by Stas Babak, Antoine Petiteau for the LDC team.
"""


class Sensitivity(ABC):
    """Base Class for PSD information.

    The initialization function is only needed if using a file input.

    """

    channel: str = None

    @staticmethod
    def get_xp(array: np.ndarray) -> object:
        """Numpy or Cupy (or float)"""
        try:
            return get_array_module(array)
        except ValueError:
            if isinstance(array, float):
                return np
            raise ValueError(
                "array must be a numpy or cupy array (it can be a float as well)."
            )

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        """Transform from the base sensitivity functions to the TDI PSDs.

        Args:
            f: Frequency array.
            noise_levels: Current noise levels at frequency ``f``.
            **kwargs: For interoperability.

        Returns:
            Transformed TDI PSD values.

        """
        raise NotImplementedError

    @classmethod
    def get_Sn(
        cls,
        f: float | np.ndarray,
        model: Optional[lisa_models.LISAModel | str] = lisa_models.sangria,
        **kwargs: dict,
    ) -> float | np.ndarray:
        """Calculate the PSD

        Args:
            f: Frequency array.
            model: Noise model. Object of type :class:`lisa_models.LISAModel`.
                It can also be a string corresponding to one of the stock models.
                The model object must include attributes for ``Soms_d`` (shot noise)
                and ``Sa_a`` (acceleration noise) or a spline as attribute ``Sn_spl``.
                In the case of a spline, this must be a dictionary with
                channel names as keys and callable PSD splines. For example,
                if using ``scipy.interpolate.CubicSpline``, an input option
                can be:

                ```
                noise_model.Sn_spl = {
                    "A": CubicSpline(f, Sn_A)),
                    "E": CubicSpline(f, Sn_E)),
                    "T": CubicSpline(f, Sn_T))
                }
                ```
            **kwargs: For interoperability.

        Returns:
            PSD values.

        """
        # spline or stock computation
        if hasattr(model, "Sn_spl") and model.Sn_spl is not None:
            spl = model.Sn_spl
            if cls.channel not in spl:
                raise ValueError("Calling a channel that is not available.")

            Sout = spl[cls.channel](f)

        else:
            model = lisa_models.check_lisa_model(model)
            # assert hasattr(model, "Soms_d") and hasattr(model, "Sa_a")

            # get noise values
            noise_levels = model.lisanoises(f)

            # transform as desired for TDI combination
            Sout = cls.transform(f, noise_levels, **kwargs)

        # will add zero if ignored
        stochastic_contribution = cls.stochastic_transform(
            f, cls.get_stochastic_contribution(f, **kwargs), **kwargs
        )

        try:
            Sout += stochastic_contribution
        except:
            breakpoint()
        return Sout

    @classmethod
    def get_stochastic_contribution(
        cls,
        f: float | np.ndarray,
        stochastic_params: Optional[tuple] = (),
        stochastic_kwargs: Optional[dict] = {},
        stochastic_function: Optional[StochasticContribution | str] = None,
    ) -> float | np.ndarray:
        """Calculate contribution from stochastic signal.

        This function directs and wraps the calculation of and returns
        the stochastic signal. The ``stochastic_function`` calculates the
        sensitivity contribution. The ``transform_factor`` can transform that
        output to the correct TDI contribution.

        This function has GPU capabilities if a Cupy frequency array is entered.

        Args:
            f: Frequency array. If a Cupy array is provided, the GPU is used. 
            stochastic_params: Parameters (arguments) to feed to ``stochastic_function``.
            stochastic_kwargs: Keyword arguments to feeed to ``stochastic_function``.
            stochastic_function: Stochastic class or string name of stochastic class. Takes ``stochastic_args`` and ``stochastic_kwargs``.
                If ``None``, it uses :class:`FittedHyperbolicTangentGalacticForeground`.

        Returns:
            Contribution from stochastic signal.


        """
        xp = cls.get_xp(f)
        if isinstance(f, float):
            f = xp.ndarray([f])
            squeeze = True
        else:
            squeeze = False

        sgal = xp.zeros_like(f)

        if (
            (tuple(stochastic_params) != tuple() and stochastic_params is not None)
            or (stochastic_kwargs != {} and stochastic_kwargs is not None)
            or stochastic_function is not None
        ):
            if stochastic_function is None:
                stochastic_function = FittedHyperbolicTangentGalacticForeground
                assert len(stochastic_params) == 1
            
            stochastic_function = check_stochastic(stochastic_function)

            sgal[:] = stochastic_function.get_Sh(
                f, *stochastic_params, **stochastic_kwargs
            )

        if squeeze:
            sgal = sgal.squeeze()
        return sgal

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        """Transform from the base stochastic functions to the TDI PSDs.

        **Note**: If not implemented, the transform will return the input.

        Args:
            f: Frequency array.
            Sh: Power spectral density in stochastic term.
            **kwargs: For interoperability.

        Returns:
            Transformed TDI PSD values.

        """
        return Sh


class X1TDISens(Sensitivity):
    channel: str = "X"

    @staticmethod
    def Cxx(f: float | np.ndarray) -> float | np.ndarray:
        """Common TDI transform factor.
        
        Args:
            f: Frequencyies to evaluate.

        Returns:
            Cxx: Transform factor.
        
        """
        x = 2 * np.pi * f * L_SI / C_SI
        return 16.0 * np.sin(x) ** 2
                                 
    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        assert noise_levels.units == "relative_frequency"
        Cxx = X1TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI
        # TODO: need to check these
        isi_rfi_readout_transfer = Cxx
        tmi_readout_transfer = Cxx * (2.0 * (1.0 + np.cos(x) ** 2))
        tm_transfer = Cxx * (2.0 * (1.0 + np.cos(x) ** 2))
        rfi_backlink_transfer = Cxx
        tmi_backlink_transfer = Cxx * (2.0 * (1.0 + np.cos(x) ** 2))
        
        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        x = 2.0 * np.pi * lisaLT * f
        t = 4.0 * x**2 * np.sin(x) ** 2
        return Sh * t

class Y1TDISens(X1TDISens):
    channel: str = "Y"
    __doc__ = X1TDISens.__doc__
    pass

class Z1TDISens(X1TDISens):
    channel: str = "Z"
    __doc__ = X1TDISens.__doc__
    pass


class XY1TDISens(Sensitivity):
    channel: str = "XY"

    @staticmethod
    def Cxy(f: float | np.ndarray) -> float | np.ndarray:
        """Common TDI transform factor for CSD.
        
        Args:
            f: Frequencyies to evaluate.

        Returns:
            Cxy: Transform factor.
        
        """
        x = 2 * np.pi * f * L_SI / C_SI
        return -4.0 * np.sin(2 * x) * np.sin(x)

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        assert noise_levels.units == "relative_frequency"
        Cxy = XY1TDISens.Cxy(f)

        isi_rfi_readout_transfer = Cxy
        tmi_readout_transfer = 4 * Cxy
        tm_transfer = 4 * Cxy
        rfi_backlink_transfer = Cxy
        tmi_backlink_transfer = 4 * Cxy

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        x = 2.0 * np.pi * lisaLT * f
        # TODO: check these functions
        # GB = -0.5 of X
        t = -0.5 * (4.0 * x**2 * np.sin(x) ** 2)
        return Sh * t


class ZX1TDISens(XY1TDISens):
    channel: str = "ZX"
    __doc__ = XY1TDISens.__doc__
    pass


class YZ1TDISens(XY1TDISens):
    channel: str = "YZ"
    __doc__ = XY1TDISens.__doc__
    pass


class X2TDISens(Sensitivity):
    channel: str = "X"

    @staticmethod
    def Cxx(f: float | np.ndarray) -> float | np.ndarray:
        """Common TDI transform factor.

        `arXiv:2211.02539 <https://arxiv.org/pdf/2211.02539>`_. 
        
        Args:
            f: Frequencyies to evaluate.

        Returns:
            Cxx: Transform factor.
        
        """
        x = 2 * np.pi * f * L_SI / C_SI
        return 16. * np.sin(x) ** 2 * np.sin(2 * x) ** 2  # np.abs(1. - np.exp(-2j * np.pi * f * L_SI / C_SI) ** 2) ** 2

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        assert noise_levels.units == "relative_frequency"
        Cxx = X2TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        isi_rfi_readout_transfer = 4. * Cxx
        tmi_readout_transfer = Cxx * (3 + np.cos(2 * x)) 
        tm_transfer = 4 * Cxx * (3 + np.cos(2 * x)) 
        rfi_backlink_transfer = 4 * Cxx
        tmi_backlink_transfer = Cxx * (3 + np.cos(2 * x)) 
 
        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        x = 2.0 * np.pi * lisaLT * f
        # TODO: check these functions for TDI2
        t = 4.0 * x**2 * np.sin(x) ** 2
        return Sh * t


class Y2TDISens(X2TDISens):
    channel: str = "Y"
    __doc__ = X2TDISens.__doc__
    pass


class Z2TDISens(X2TDISens):
    channel: str = "Z"
    __doc__ = X2TDISens.__doc__
    pass

class XY2TDISens(Sensitivity):
    """
    Cross-spectral density (CSD) between X and Y channels for TDI2.

    From Table II of Nam et al. (2023) for uncorrelated noises:
    - Common factor: C_XY(ω) = -16 sin(ωL) sin³(2ωL)
    - Acceleration contribution: 4 * C_XY * S_pm
    - Optical path contribution (ISI/RFI): C_XY * S_op

    Total CSD: C_XY * (4*S_pm + S_op)

    Notes:
        - By circular symmetry, YZ and ZX CSDs have identical transfer functions
        - For equal armlengths, the CSD is real-valued
        - This implements the uncorrelated noise case
    """

    channel: str = "XY"

    @staticmethod
    def Cxy(f: float | np.ndarray) -> float | np.ndarray:
        """Common TDI transform factor for CSD.

        `arXiv:2211.02539 <https://arxiv.org/pdf/2211.02539>`_. 
        
        Args:
            f: Frequencyies to evaluate.

        Returns:
            Cxy: Transform factor.
        
        """
        x = 2 * np.pi * f * L_SI / C_SI

        return -16.0 * np.sin(x) * np.sin(2.0 * x) ** 3

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        """
        Transform from base sensitivity functions (S_pm, S_op) to TDI2 XY CSD.

        Args:
            f: Frequency array [Hz].
            noise_levels: Current noise levels at frequency ``f``.
            **kwargs: For interoperability.

        Returns:
            Cross-spectral density between X and Y channels.

        Mathematical form:
            x = 2π(L/c)f  [dimensionless frequency]
            C_XY = -16 sin(x) sin³(2x)
            CSD_XY = C_XY * (4*S_pm + S_op)
        """
        assert noise_levels.units == "relative_frequency"
        Cxy = XY2TDISens.Cxy(f)

        isi_rfi_readout_transfer = Cxy
        tmi_readout_transfer = Cxy
        tm_transfer = 4 * Cxy
        rfi_backlink_transfer = Cxy
        tmi_backlink_transfer = Cxy
 
        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        """
        Transform stochastic background to TDI2 XY CSD.

        Note: For now, using same transform as TDI1 (placeholder).
        TODO: Verify correct stochastic transform for TDI2 CSDs.

        Args:
            f: Frequency array [Hz].
            Sh: Stochastic signal PSD.
            **kwargs: For interoperability.

        Returns:
            Stochastic contribution to CSD.
        """
        x = 2.0 * np.pi * lisaLT * f
        # Placeholder - using TDI1 form scaled by -0.5
        t = -0.5 * (4.0 * x**2 * np.sin(x) ** 2)
        return Sh * t


class YZ2TDISens(XY2TDISens):
    """
    Cross-spectral density (CSD) between Y and Z channels for TDI2.

    By circular symmetry of the LISA constellation (for equal armlengths),
    this has the same transfer function as XY2TDISens.
    """

    channel: str = "YZ"
    __doc__ = XY2TDISens.__doc__
    pass


class ZX2TDISens(XY2TDISens):
    """
    Cross-spectral density (CSD) between Z and X channels for TDI2.

    By circular symmetry of the LISA constellation (for equal armlengths),
    this has the same transfer function as XY2TDISens.
    """

    channel: str = "ZX"
    __doc__ = XY2TDISens.__doc__
    pass

class A1TDISens(X1TDISens, Sensitivity):
    channel: str = "A"

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the A,E TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        # these are WRONG
        if np.any(np.asarray([
            noise_levels.rfi_backlink_noise,
            noise_levels.tmi_backlink_noise,
            noise_levels.rfi_oms_noise,
            noise_levels.tmi_oms_noise
        ]) != 0.0):
            raise NotImplementedError("ExtendedLISAModel has not been implemented yet for A1/E1/T1.")

        assert noise_levels.units == "relative_frequency"
        Cxx = X1TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        # these are WRONG
        tmi_readout_transfer = Cxx * (2.0 * (1.0 + np.cos(x) ** 2))
        rfi_backlink_transfer = Cxx
        tmi_backlink_transfer = Cxx * (2.0 * (1.0 + np.cos(x) ** 2))
        
        # these are right and were changed accordingly
        # Need to find a citation for these 1st gen stuff
        # all that is needed for old model type
        isi_rfi_readout_transfer = 1/2 * Cxx * (2.0 + np.cos(x))
        tm_transfer = Cxx * (3.0 + 2.0 * np.cos(x) + np.cos(2 * x))

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        x = 2.0 * np.pi * lisaLT * f
        t = 4.0 * x**2 * np.sin(x) ** 2
        return 1.5 * (Sh * t)


class E1TDISens(A1TDISens):
    channel: str = "E"
    __doc__ = A1TDISens.__doc__
    pass


class T1TDISens(Sensitivity):
    channel: str = "T"

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the T TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        assert noise_levels.units == "relative_frequency"
        
        Cxx = X1TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        # these are WRONG
        if np.any(np.asarray([
            noise_levels.rfi_backlink_noise,
            noise_levels.tmi_backlink_noise,
            noise_levels.rfi_oms_noise,
            noise_levels.tmi_oms_noise
        ]) != 0.0):
            raise NotImplementedError("ExtendedLISAModel has not been implemented yet for A1/E1/T1.")
        tmi_readout_transfer = Cxx * (2.0 * (1.0 + np.cos(x) ** 2))
        rfi_backlink_transfer = Cxx
        tmi_backlink_transfer = Cxx * (2.0 * (1.0 + np.cos(x) ** 2))
        
        # these are right and were changed accordingly
        # Need to find a citation for these 1st gen stuff
        # all that is needed for old model type
        isi_rfi_readout_transfer = Cxx * (1 - np.cos(x))
        tm_transfer = 8.0 * Cxx * np.sin(x / 2.) ** 4

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise
    
    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        x = 2.0 * np.pi * lisaLT * f
        t = 4.0 * x**2 * np.sin(x) ** 2
        return 0.0 * (Sh * t)



class A2TDISens(X2TDISens, Sensitivity):
    channel: str = "A"

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        assert noise_levels.units == "relative_frequency"
        Cxx = X2TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI
        
        isi_rfi_readout_transfer = 2. * Cxx * (2 + np.cos(x))
        tmi_readout_transfer = Cxx * (3 + 2 * np.cos(x) + np.cos(2 * x)) 
        tm_transfer = 4 * Cxx * (3 + 2 * np.cos(x) + np.cos(2 * x)) 
        
        rfi_backlink_transfer = 2 * Cxx * (2 * np.cos(x))
        tmi_backlink_transfer = Cxx * (3 + 2 * np.cos(x) + np.cos(2 * x)) 
 
        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise 

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        x = 2.0 * np.pi * lisaLT * f
        # TODO: check these functions for TDI2
        t = 4.0 * x**2 * np.sin(x) ** 2
        return Sh * t


class E2TDISens(A2TDISens):
    channel: str = "E"
    __doc__ = A2TDISens.__doc__
    pass


class T2TDISens(X2TDISens, Sensitivity):
    channel: str = "T"

    @staticmethod
    def transform(
        f: float | np.ndarray,
        noise_levels: lisa_models.CurrentNoises,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        assert noise_levels.units == "relative_frequency"
        Cxx = X2TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI
        
        isi_rfi_readout_transfer = 4. * Cxx * (1 - np.cos(x))
        tmi_readout_transfer = 8 * Cxx * np.sin(x / 2.) ** 4
        tm_transfer = 32 * Cxx * np.sin(x / 2.) ** 4
        rfi_backlink_transfer = 4. * Cxx * (1 - np.cos(x))
        tmi_backlink_transfer = 8 * Cxx * np.sin(x / 2.) ** 4
 
        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise
        
        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = tm_noise_ffd + isi_oms_ffd + rfi_oms_ffd + tmi_oms_ffd + rfi_backlink_ffd + tmi_backlink_ffd
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        x = 2.0 * np.pi * lisaLT * f
        # TODO: check these functions for TDI2
        t = 4.0 * x**2 * np.sin(x) ** 2
        return Sh * t



class LISASens(Sensitivity):
    @classmethod
    def get_Sn(
        cls,
        f: float | np.ndarray,
        model: Optional[lisa_models.LISAModel | str] = lisa_models.sangria,
        average: bool = True,
        **kwargs: dict,
    ) -> float | np.ndarray:
        """Compute the base LISA sensitivity function.

        Args:
            f: Frequency array.
            model: Noise model. Object of type :class:`lisa_models.LISAModel`. It can also be a string corresponding to one of the stock models.
            average: Whether to apply averaging factors to sensitivity curve.
                Antenna response: ``av_resp = np.sqrt(5) if average else 1.0``
                Projection effect: ``Proj = 2.0 / np.sqrt(3) if average else 1.0``
            **kwargs: Keyword arguments to pass to :func:`get_stochastic_contribution`. # TODO: fix

        Returns:
            Sensitivity array.

        """
        model = lisa_models.check_lisa_model(model)
        
        if not isinstance(model, lisa_models.LISAModel):
            raise NotImplementedError("This function has not been implemented for ExtendedLISAModel yet.")

        # get noise values
        noise_values = model.lisanoises(f, unit="displacement")

        Sa_d = noise_values.tm_noise
        Sop = noise_values.isi_oms_noise

        all_m = np.sqrt(4.0 * Sa_d + Sop)
        ## Average the antenna response
        av_resp = np.sqrt(5) if average else 1.0

        ## Projection effect
        Proj = 2.0 / np.sqrt(3) if average else 1.0

        ## Approximative transfer function
        f0 = 1.0 / (2.0 * lisaLT)
        a = 0.41
        T = np.sqrt(1 + (f / (a * f0)) ** 2)
        sens = (av_resp * Proj * T * all_m / lisaL) ** 2

        # will add zero if ignored
        sens += cls.get_stochastic_contribution(f, **kwargs)
        return sens


class CornishLISASens(LISASens):
    """PSD from https://arxiv.org/pdf/1803.01944.pdf

    Power Spectral Density for the LISA detector assuming it has been active for a year.
    I found an analytic version in one of Niel Cornish's paper which he submitted to the arXiv in
    2018. I evaluate the PSD at the frequency bins found in the signal FFT.

    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf

    """

    @staticmethod
    def get_Sn(
        f: float | np.ndarray, average: bool = True, **kwargs: dict
    ) -> float | np.ndarray:
        # TODO: documentation here

        sky_averaging_constant = 20.0 / 3.0 if average else 1.0

        L = 2.5 * 10**9  # Length of LISA arm
        f0 = 19.09 * 10 ** (-3)  # transfer frequency

        # Optical Metrology Sensor
        Poms = ((1.5e-11) * (1.5e-11)) * (1 + np.power((2e-3) / f, 4))

        # Acceleration Noise
        Pacc = (
            (3e-15)
            * (3e-15)
            * (1 + (4e-4 / f) * (4e-4 / f))
            * (1 + np.power(f / (8e-3), 4))
        )

        # constants for Galactic background after 1 year of observation
        alpha = 0.171
        beta = 292
        k = 1020
        gamma = 1680
        f_k = 0.00215

        # Galactic background contribution
        Sc = (
            9e-45
            * np.power(f, -7 / 3)
            * np.exp(-np.power(f, alpha) + beta * f * np.sin(k * f))
            * (1 + np.tanh(gamma * (f_k - f)))
        )

        # PSD
        PSD = (sky_averaging_constant) * (
            (10 / (3 * L * L))
            * (Poms + (4 * Pacc) / (np.power(2 * np.pi * f, 4)))
            * (1 + 0.6 * (f / f0) * (f / f0))
            + Sc
        )

        return PSD


class FlatPSDFunction(LISASens):
    """White Noise PSD function."""

    @classmethod
    def get_Sn(
        cls, f: float | np.ndarray, val: float, **kwargs: dict
    ) -> float | np.ndarray:
        # TODO: documentation here
        xp = cls.get_xp(f)
        out = xp.full_like(f, val)
        if isinstance(f, float):
            out = out.item()
        return out

class SensitivityMatrixBase:
    """Base Container to hold sensitivity information.

    Args:
        basis_x: Frequency array in FD. Time array in TD. Wavelet basis in WDM. Etc.

    """

    def __init__(
        self,
        settings: domains.DomainSettingsBase,
        
    ) -> None:
        self.basis_settings = settings
        self.data_shape = self.basis_settings.basis_shape

    @property
    def basis_settings(self) -> np.ndarray:
        return self._basis_settings

    @basis_settings.setter
    def basis_settings(self, basis_settings: np.ndarray) -> None:
        assert isinstance(basis_settings, domains.DomainSettingsBase)
        self._basis_settings = basis_settings

    def check_update(self):
        if not self.can_redo:
            raise ValueError("Cannot update sensitivities because original input was arrays rather than functions.")

    def update_basis_settings(self, basis_settings: domains.DomainSettingsBase) -> None:
        """Update class with new frequency array.

        Args:
            basis_settings: Domain information.

        """
        self.check_update()
        self.basis_settings = basis_settings
        self.sens_mat = self.sens_mat_input

    def update_model(self, model: lisa_models.LISAModel | list | np.ndarray) -> None:
        """Update class with new sensitivity model.

        Args:
            model: Noise model. Object of type :class:`lisa_models.LISAModel`. It can also be a string corresponding to one of the stock models.

        """
        self.check_update()
        for tmp_kwargs in self.sens_kwargs.flatten():
            tmp_kwargs["model"] = model
        self.sens_mat = self.sens_mat_input

    def update_stochastic(self, **kwargs: dict) -> None:
        """Update class with new stochastic function.

        Args:
            **kwargs: Keyword arguments update for :func:`lisatools.sensitivity.Sensitivity.get_stochastic_contribution`.
                This operation will combine the new and old kwarg dictionaries, updating any
                old information with any added corresponding new information. **Note**: any old information
                that is not updated will remain in place.

        """
        self.check_update()
        tmptmp = self.sens_kwargs.flatten()
        for i, tmp_kwargs in tmptmp:
            tmptmp[i] = {**tmp_kwargs, **kwargs}
        self.sens_kwargs = tmptmp.reshape(self.sens_kwargs.shape)
        self.sens_mat = self.sens_mat_input

    @property
    def sens_mat(self) -> np.ndarray:
        """Get sensitivity matrix."""
        return self._sens_mat

    @sens_mat.setter
    def sens_mat(
        self,
        sens_mat: (
            List[List[np.ndarray | Sensitivity]]
            | List[np.ndarray | Sensitivity]
            | np.ndarray
            | Sensitivity
        ),
    ) -> None:
        """Set sensitivity matrix."""
        
        if (isinstance(sens_mat, np.ndarray) or isinstance(
            sens_mat, cp.ndarray)
        ) and sens_mat.dtype != object:
            assert sens_mat.shape[-len(self.data_shape):] == self.data_shape
            
            self._sens_mat = sens_mat
            if not hasattr(self, "sens_mat_input"):
                self.can_redo = False
            else:
                self.can_redo = True

        elif isinstance(sens_mat, list) or (isinstance(sens_mat, np.ndarray) and sens_mat.dtype == object):
            self.sens_mat_input = deepcopy(sens_mat)
            _run = True
            _layer = self.sens_mat_input
            outer_shape = [len(_layer)]
            while _run:
                _test_length = None
                _type_1 = None
                for tmp in _layer:
                    # check each entry is the same type
                    if _type_1 is None:
                        _type_1 = type(tmp)
                    else:
                        if _type_1 != type(tmp):
                            raise ValueError("List inputs must be all of the same type.")
                        
                    if isinstance(tmp, list):
                        if _test_length is None:
                            _test_length = len(tmp)
                        else:
                            if len(tmp) != _test_length:
                                raise ValueError("Input list structure is not Rectangular.")
                    elif isinstance(tmp, np.ndarray) or isinstance(tmp, cp.ndarray):
                        if tmp.ndim > 1:
                            raise ValueError("If entering a list of arrays, arrays must be 1D on the last dimension of the list structure.")
                        if _test_length is None:
                            _test_length = len(tmp)
                        else:
                            if len(tmp) != _test_length:
                                raise ValueError("Input list/array structure is not Rectangular.")

                if isinstance(_layer[0], list):
                    outer_shape.append(len(_layer[0]))
                    _layer = _layer[0]
                    continue
                        
                elif isinstance(_layer[0], np.ndarray) or isinstance(_layer[0], cp.ndarray):
                    # hit the array, must be last layer
                    _run = False
                    self.can_redo = False
                    self.is_array_base = True
                    continue

                # TODO: better way to do this?
                elif hasattr(_layer[0], "get_Sn"):
                    _run = False
                    self.can_redo = True
                    self.is_array_base = False
                    continue

                elif isinstance(_layer[0], str):
                    _run = False
                    self.can_redo = True
                    self.is_array_base = False
                    sensitivity = check_sensitivity(_layer[0])
                    assert hasattr(sensitivity, "get_Sn")
                    continue

                else:
                    raise ValueError("Matrix element must be Sensitivity object, string representing a sensitivity object, or an array with values.")
                
        
            if isinstance(self.sens_kwargs, np.ndarray) or isinstance(self.sens_kwargs, list):
                tmp_kwargs = np.asarray(self.sens_kwargs, dtype=object)
                assert tmp_kwargs.shape == tuple(outer_shape)

            elif isinstance(self.sens_kwargs, dict):
                tmp_kwargs = np.full(outer_shape, self.sens_kwargs, dtype=object)
            else:
                raise ValueError("sens_kwargs Must be numpy object array, list, or dict.")
            
            # TODO: sens_kwargs property setup
            self.sens_kwargs = tmp_kwargs
            
            num_components = np.prod(outer_shape).item()
            # xp = get_array_module(self.frequency_arr)
            xp = np
            if self.is_array_base:
                _sens_mat = xp.asarray(sens_mat)
            
            else:
                _flattened_arr = np.asarray(sens_mat, dtype=object).flatten()
                _sens_mat = xp.zeros((num_components,) + self.basis_settings.basis_shape)
                for i, matrix_member in enumerate(_flattened_arr):
                    # calculate it
                    if hasattr(matrix_member, "get_Sn") or isinstance(matrix_member, str):
                        _sens_mat[i, :] = get_sensitivity(
                            self.basis_settings,
                            *self.sens_args,
                            sens_fn=matrix_member,
                            **self.sens_kwargs.flatten()[i],
                        )

                    else:
                        raise ValueError

            # setup in array form
            self._sens_mat = _sens_mat.reshape(tuple(outer_shape) + self.basis_settings.basis_shape)
            
        else:
            raise ValueError("Must input array or list.")
        
        self.channel_shape = self._sens_mat.shape[:-len(self.data_shape)]
        self._setup_det_and_inv()

    @property
    def differential_component(self) -> float:
        return self.basis_settings.differential_component

    def _setup_det_and_inv(self):
        """Determinant and inverse of TDI matrix."""

        # setup detC
        xp = get_array_module(self.sens_mat)

        # setup detC
       
        if self.sens_mat.ndim < 3:
            self.detC = xp.prod(self.sens_mat, axis=0)
            self.invC = 1 / self.sens_mat

        else:
            full_shape = tuple(range(len(self.sens_mat.shape)))

            basis_axes = full_shape[-len(self.data_shape):]
            mat_axes = full_shape[:-len(self.data_shape)]
            transpose_shape = basis_axes + mat_axes
            self.detC = xp.linalg.det(self.sens_mat.transpose(transpose_shape))
            invC = xp.zeros_like(self.sens_mat.transpose(transpose_shape))
            invC[self.detC != 0.0] = xp.linalg.inv(
                self.sens_mat.transpose(transpose_shape)[self.detC != 0.0]
            )
            invC[self.detC == 0.0] = 1e-100

            # switch them after they were effectively switched above
            _mat_axes = mat_axes
            mat_axes = basis_axes
            basis_axes = _mat_axes
            back_transpose_shape = mat_axes + basis_axes
            self.invC = invC.transpose(back_transpose_shape)

    def __getitem__(self, index: Any) -> np.ndarray:
        """Indexing the class indexes the array."""
        return self.sens_mat[index]

    def __setitem__(self, index: Any, value: np.ndarray) -> np.ndarray:
        """Indexing the class indexes the array."""
        self.sens_mat[index] = value
        self._setup_det_and_inv()

    @property
    def ndim(self) -> int:
        """Dimensionality of sens mat array."""
        return self.sens_mat.ndim

    def flatten(self) -> np.ndarray:
        """Flatten sens mat array."""
        return self.sens_mat.reshape(-1, self.sens_mat.shape[-1])

    @property
    def shape(self) -> tuple:
        """Shape of sens mat array."""
        return self.sens_mat.shape

    def loglog(
        self,
        ax: Optional[plt.Axes] = None,
        fig: Optional[plt.Figure] = None,
        inds: Optional[int | tuple] = None,
        char_strain: Optional[bool] = False,
        **kwargs: dict,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Produce a log-log plot of the sensitivity.

        Args:
            ax: Matplotlib Axes objects to add plots. Either a list of Axes objects or a single Axes object.
            fig: Matplotlib figure object.
            inds: Integer index to select out which data to add to a single access.
                A list can be provided if ax is a list. They must be the same length.
            char_strain: If ``True``, plot in characteristic strain representation. **Note**: assumes the sensitivity
                is input as power spectral density.
            **kwargs: Keyword arguments to be passed to ``loglog`` function in matplotlib.

        Returns:
            Matplotlib figure and axes objects in a 2-tuple.


        """
        if (ax is None and fig is None) or (
            ax is not None and (isinstance(ax, list) or isinstance(ax, np.ndarray))
        ):
            if ax is None and fig is None:
                outer_shape = self.shape[:-1]
                if len(outer_shape) == 2:
                    nrows = outer_shape[0]
                    ncols = outer_shape[1]
                elif len(outer_shape) == 1:
                    nrows = 1
                    ncols = outer_shape[0]

                fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
                try:
                    ax = ax.ravel()
                except AttributeError:
                    ax = [ax]  # just one axis object, no list

            else:
                assert len(ax) == np.prod(self.shape[:-1])

            for i in range(np.prod(self.shape[:-1])):
                plot_in = self.flatten()[i]
                if char_strain:
                    plot_in = np.sqrt(self.frequency_arr * plot_in)
                ax[i].loglog(self.frequency_arr, plot_in, **kwargs)

        elif fig is not None:
            raise NotImplementedError

        elif isinstance(ax, plt.axes):
            if inds is None:
                raise ValueError(
                    "When passing a single axes object for `ax`, but also pass `inds` kwarg."
                )
            plot_in = self.sens_mat[inds]
            if char_strain:
                plot_in = np.sqrt(self.frequency_arr * plot_in)
            ax.loglog(self.frequency_arr, plot_in, **kwargs)

        else:
            raise ValueError(
                "ax must be a list of axes objects or a single axes object."
            )

        return (fig, ax)

class SensitivityMatrix(SensitivityMatrixBase):
    """Container to hold sensitivity information.

    Args:
        basis_x: Frequency array in FD. Time array in TD. Wavelet basis in WDM. Etc.
        sens_mat: Input sensitivity list. The shape of the nested lists should represent the shape of the
            desired matrix. Each entry in the list must be an array, :class:`Sensitivity`-derived object,
            or a string corresponding to the :class:`Sensitivity` object.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(
        self,
        settings: domains.DomainSettingsBase,
        sens_mat: (
            List[List[np.ndarray | Sensitivity]]
            | List[np.ndarray | Sensitivity]
            | np.ndarray
            | Sensitivity
        ),
        *sens_args: tuple,
        sens_kwargs_mat = None,
        **sens_kwargs: dict,
    ) -> None:
        super().__init__(settings)
        self.sens_args = sens_args
        if sens_kwargs_mat is None:
            self.sens_kwargs = sens_kwargs
        else:
            self.sens_kwargs = sens_kwargs_mat

        self.sens_mat = sens_mat

class XYZ1SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for XYZ (TDI 1)

    This is 3x3 symmetric matrix.

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, settings: domains.DomainSettingsBase, **sens_kwargs: dict) -> None:
        sens_mat = [
            [X1TDISens, XY1TDISens, ZX1TDISens],
            [XY1TDISens, Y1TDISens, YZ1TDISens],
            [ZX1TDISens, YZ1TDISens, Z1TDISens],
        ]
        super().__init__(settings, sens_mat, **sens_kwargs)

class XYZ2SensitivityMatrix(SensitivityMatrix):
    """
    Default sensitivity matrix for XYZ channels using TDI2 transfer functions.

    This creates a 3×3 Hermitian covariance matrix accounting for correlations
    between the X, Y, and Z TDI channels due to shared noise sources (S_pm and S_op).

    Matrix structure:
        Σ(f) = [ Σ_XX   Σ_XY   Σ_XZ ]
               [ Σ_YX   Σ_YY   Σ_YZ ]  at each frequency
               [ Σ_ZX   Σ_ZY   Σ_ZZ ]

    Args:
        f: Frequency array [Hz].
        **sens_kwargs: Keyword arguments to pass to Sensitivity.get_Sn()
            (e.g., model=lisa_models.sangria).

    Notes:
        - Inherits matrix inversion and determinant computation from SensitivityMatrix
        - The invC attribute provides Σ⁻¹(f) for likelihood computations
        - The detC attribute provides det[Σ(f)] for normalization
    """

    def __init__(self, settings: domains.DomainSettingsBase, **sens_kwargs: dict) -> None:
        """
        Initialize TDI2 sensitivity matrix.

        Args:
            settings: Domain settings containing frequency array and other parameters.
            **sens_kwargs: Keyword arguments for Sensitivity.get_Sn()
                Common kwargs:
                    - model: LISA noise model (e.g., sangria, sangria)
                    - stochastic_params: Parameters for galactic foreground
                    - stochastic_function: Custom stochastic function
        """
        # Define 3×3 matrix structure
        # Diagonal: X2, Y2, Z2 PSDs 
        # Off-diagonal: XY2, YZ2, ZX2 CSDs 
        sens_mat = [
            [X2TDISens,   XY2TDISens,  ZX2TDISens],
            [XY2TDISens,  Y2TDISens,   YZ2TDISens],
            [ZX2TDISens,  YZ2TDISens,  Z2TDISens],
        ]

        super().__init__(settings, sens_mat, **sens_kwargs)

class AET1SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for AET (TDI 1)

    This is just an array because no cross-terms.

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, settings: domains.DomainSettingsBase, **sens_kwargs: dict) -> None:
        sens_mat = [A1TDISens, E1TDISens, T1TDISens]
        super().__init__(settings, sens_mat, **sens_kwargs)


class AET2SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for AET (TDI 2)

    This is just an array because no cross-terms.

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, settings: domains.DomainSettingsBase, **sens_kwargs: dict) -> None:
        sens_mat = [A2TDISens, E2TDISens, T2TDISens]
        super().__init__(settings, sens_mat, **sens_kwargs)


class AE1SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for AE (no T) (TDI 1)

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, settings: domains.DomainSettingsBase, **sens_kwargs: dict) -> None:
        sens_mat = [A1TDISens, E1TDISens]
        super().__init__(settings, sens_mat, **sens_kwargs)


class AE2SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for AE (no T) (TDI 1)

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, f: np.ndarray, **sens_kwargs: dict) -> None:
        sens_mat = [A2TDISens, E2TDISens]
        super().__init__(f, sens_mat, **sens_kwargs)


class LISASensSensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix adding :class:`LISASens` for the specified number of channels.

    Args:
        f: Frequency array.
        nchannels: Number of channels.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, f: np.ndarray, nchannels: int, **sens_kwargs: dict) -> None:
        sens_mat = [LISASens for _ in range(nchannels)]
        super().__init__(f, sens_mat, **sens_kwargs)


def get_sensitivity(
    basis_settings: domains.DomainSettingsBase,
    *args: tuple,
    sens_fn: Optional[Sensitivity | str] = LISASens,
    return_type="PSD",
    fill_nans: float = 1e10,
    args_list: Optional[List[tuple]] = None,
    kwargs_list: Optional[List[dict]] = None,
    **kwargs,
) -> float | np.ndarray:
    """Generic sensitivity generator

    Same interface to many sensitivity curves.

    Args:
        f: Frequency array.
        *args: Any additional arguments for the sensitivity function ``get_Sn`` method.
        sens_fn: String or class that represents the name of the desired PSD function.
        return_type: Described the desired output. Choices are ASD,
            PSD, or char_strain (characteristic strain). Default is ASD.
        fill_nans: Value to fill nans in sensitivity (at 0 frequency). 
            If ``None``, thens nans will be left in the array.
        **kwargs: Keyword arguments to pass to sensitivity function ``get_Sn`` method.

    Return:
        Sensitivity values.

    """

    if isinstance(sens_fn, str):
        sensitivity = check_sensitivity(sens_fn)

    elif hasattr(sens_fn, "get_Sn"):
        sensitivity = sens_fn

    else:
        raise ValueError(
            "sens_fn must be a string for a stock option or a class with a get_Sn method."
        )

    if isinstance(basis_settings, domains.FDSettings):
        PSD = sensitivity.get_Sn(basis_settings.f_arr, *args, **kwargs)

    elif isinstance(basis_settings, domains.TDSettings):
        raise NotImplementedError
    elif isinstance(basis_settings, domains.STFTSettings):
        raise NotImplementedError
        PSD = sensitivity.get_Sn(basis_settings.f_arr, *args, **kwargs)
    elif isinstance(basis_settings, domains.WDMSettings):
        if kwargs_list is None:
            kwargs_list = [kwargs for _ in range(basis_settings.NT)]
        else:
            assert isinstance(kwargs_list, list)
            assert len(kwargs_list) == basis_settings.NT
            for tmp in kwargs_list:
                if not isinstance(tmp, dict):
                    raise ValueError("Value in kwargs_list is not a dictionary. Must be a dictionary.")
            
        if args_list is None:
            args_list = [args for _ in range(basis_settings.NT)]
        else:
            assert isinstance(args_list, list)
            assert len(args_list) == basis_settings.NT
            for tmp in args_list:
                if not isinstance(tmp, tuple) and not isinstance(tmp, list):
                    raise ValueError("Value in args_list is not a tuple. Must be a tuple.")
            
        # equation for stationary noise (https://arxiv.org/pdf/2009.00043; eq. 19)
        PSD = np.asarray([basis_settings.df * sensitivity.get_Sn(basis_settings.f_arr, *_args, **_kwargs) for _args, _kwargs in zip(args_list, kwargs_list)])

    else:
        raise ValueError(f"Domain type entered ({type(basis_settings)}). Needs to be one of {domains.get_available_domains()}")
    
    if fill_nans is not None:
        assert isinstance(fill_nans, float)
        PSD[np.isnan(PSD)] = fill_nans

    if return_type == "PSD":
        return PSD

    elif return_type == "ASD":
        return PSD ** (1 / 2)

    elif return_type == "char_strain":
        return (basis_settings.f_arr * PSD) ** (1 / 2)

    else:
        raise ValueError("return_type must be PSD, ASD, or char_strain.")


__stock_sens_options__ = [
    "X1TDISens",
    "Y1TDISens",
    "Z1TDISens",
    "XY1TDISens",
    "YZ1TDISens",
    "ZX1TDISens",
    "A1TDISens",
    "E1TDISens",
    "T1TDISens",
    "X2TDISens",
    "Y2TDISens",
    "Z2TDISens",
    "XY2TDISens",
    "YZ2TDISens",   
    "ZX2TDISens",
    "LISASens",
    "CornishLISASens",
    "FlatPSDFunction",
]


def get_stock_sensitivity_options() -> List[Sensitivity]:
    """Get stock options for sensitivity curves.

    Returns:
        List of stock sensitivity options.

    """
    return __stock_sens_options__


__stock_sensitivity_mat_options__ = [
    "XYZ1SensitivityMatrix",
    "XYZ2SensitivityMatrix",
    "AET1SensitivityMatrix",
    "AE1SensitivityMatrix",
]


def get_stock_sensitivity_matrix_options() -> List[SensitivityMatrix]:
    """Get stock options for sensitivity matrix.

    Returns:
        List of stock sensitivity matrix options.

    """
    return __stock_sensitivity_mat_options__


def get_stock_sensitivity_from_str(sensitivity: str) -> Sensitivity:
    """Return a LISA sensitivity from a ``str`` input.

    Args:
        sensitivity: Sensitivity indicated with a ``str``.

    Returns:
        Sensitivity associated to that ``str``.

    """
    if sensitivity not in __stock_sens_options__:
        raise ValueError(
            "Requested string sensitivity is not available. See lisatools.sensitivity documentation."
        )
    return globals()[sensitivity]


def check_sensitivity(sensitivity: Any) -> Sensitivity:
    """Check input sensitivity.

    Args:
        sensitivity: Sensitivity to check.

    Returns:
        Sensitivity checked. Adjusted from ``str`` if ``str`` input.

    """
    if isinstance(sensitivity, str):
        sensitivity = get_stock_sensitivity_from_str(sensitivity)

    if not issubclass(sensitivity, Sensitivity):
        raise ValueError("sensitivity argument not given correctly.")

    return sensitivity



class XYZSensitivityBackend:
    pass

class XYZSensitivityBackend(LISAToolsParallelModule, SensitivityMatrixBase):
    """Helper class for sensitivity matrix with c++ backend."""
    
    def __init__(self, 
                 orbits: L1Orbits,
                 settings: DomainSettingsBase,
                 tdi_generation: int = 2,
                 use_splines: bool = False,
                 force_backend: Optional[str] = 'cpu',
                 mask_percentage: Optional[float] = None,
                 ):
        
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        SensitivityMatrixBase.__init__(self, settings)

        assert self.backend.xp == orbits.xp, "Orbits and Sensitivity backend mismatch."
        
        self.orbits = orbits
        if not self.orbits.configured:
            self.orbits.configure(linear_interp_setup=True)

        self.tdi_generation = tdi_generation
        self.channel_shape = (3, 3) 

        _use_gpu = force_backend != 'cpu'

        self.use_splines = use_splines
        self.spline_interpolant = AkimaInterpolant1D(use_gpu=_use_gpu, threadsperblock=NUM_SPLINE_THREADS, order='cubic')

        self.mask_percentage = mask_percentage if mask_percentage is not None else 0.05

        self._setup()

    @property
    def kwargs(self):
        """Keyword arguments for class initialization."""
        return {
            "orbits": self.orbits,
            "settings": self.basis_settings,
            "tdi_generation": self.tdi_generation,
            "use_splines": self.use_splines,
            "force_backend": 'cpu' if self.backend.xp == np else 'gpu',
            "mask_percentage": self.mask_percentage,
        }


    @property
    def xp(self):
        """Array module."""
        return self.backend.xp
    
    @property
    def time_indices(self):
        return self._time_indices
    @time_indices.setter
    def time_indices(self, x):
        self._time_indices = x
    
    def get_averaged_ltts(self):
        # first, compute the average ltts and their differences. 
        # check if we need multiple time points
        if hasattr(self.basis_settings, 't_arr'):
            t_arr = self.xp.asarray(self.basis_settings.t_arr)
            tiled_times = self.xp.tile(
                t_arr[:, self.xp.newaxis], (1, 6)
            ).flatten()  # compute ltts at these times with orbits

            links = self.xp.tile(self.xp.asarray(self.orbits.LINKS), (t_arr.shape[0],))

            ltts = self.orbits.get_light_travel_times(
                tiled_times, links
            ).reshape(len(t_arr), 6)

            self.time_indices = self.xp.arange(len(t_arr), dtype=self.xp.int32)
        
        else:
            ltts = self.xp.mean(self.orbits.ltt, axis=0)[self.xp.newaxis, :]
            self.time_indices = self.xp.array([0], dtype=self.xp.int32)

        # with orbits.LINKS order: 12, 23, 31, 13, 32, 21, we need averages between pairs
        # pairs: (12,21), (23,32), (31,13)
        # Use direct indexing to avoid assignment issues with shape (1, 6) arrays
        indices = [0, 1, 2, 3, 4, 5]
        opposite_indices = indices[::-1]

        avg_ltts = 0.5 * (ltts[:, indices] + ltts[:, opposite_indices])
        delta_ltts = ltts[:, indices] - ltts[:, opposite_indices]

        return avg_ltts, delta_ltts

    def _setup(self):
        """Setup the arguments for the c++ backend."""
        
        avg_ltts, delta_ltts = self.get_averaged_ltts()
        
        self.pycppsensmat_args = [
            self.xp.asarray(avg_ltts.flatten().copy()),
            self.xp.asarray(delta_ltts.flatten().copy()),
            avg_ltts.shape[0],  # n_times
            self.orbits.armlength,
            self.tdi_generation,
            self.use_splines,
        ]

        self.pycpp_sensitivity_matrix = self.backend.SensitivityMatrixWrap(*self.pycppsensmat_args)

        self._init_basis_settings()

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle unpicklable backend objects."""
        from copy import copy
        
        # Create a new instance without calling __init__
        cls = self.__class__
        new_obj = cls.__new__(cls)
        
        # Copy the memo to avoid infinite recursion
        memo[id(self)] = new_obj
        
        # Manually copy attributes
        for key, value in self.__dict__.items():
            if key in ('_backend', 'pycpp_sensitivity_matrix'):
                # Don't deepcopy backend objects - just reference
                setattr(new_obj, key, value)
            elif key == 'orbits':
                # Shallow copy orbits (share the same backend)
                setattr(new_obj, key, copy(value))
            elif key == 'spline_interpolant':
                # Shallow copy spline interpolant
                setattr(new_obj, key, copy(value))
            else:
                # Deepcopy everything else
                setattr(new_obj, key, deepcopy(value, memo))
        
        return new_obj

    def _init_basis_settings(self):
        """Initialize basis settings from domain settings."""
        self.f_arr = self.xp.asarray(self.basis_settings.f_arr)
        
        if hasattr(self.basis_settings, 't_arr'):
            self.t_arr = self.xp.asarray(self.basis_settings.t_arr)

        self.num_times = len(self.t_arr) if hasattr(self, 't_arr') else 1
        self.num_freqs = len(self.f_arr)

        dips_indices = self._get_dips_indices()

        dips_mask = self.xp.zeros((self.num_times, self.num_freqs), dtype=bool)
        for t_idx in range(self.num_times):
            dips_mask[t_idx, dips_indices[t_idx]] = True
        
        self.dips_mask = dips_mask.flatten()

    def _find_dips_with_percentage(self, tf, mask_percentage=0.05):

        if hasattr(self.f_arr, 'get'):
            f_arr = self.f_arr.get()
            tf = tf.get()

        peaks = find_peaks(-tf)[0]
        
        all_indices = set()
        for peak in peaks:
            freq = self.f_arr[peak]
            df = self.f_arr[1] - self.f_arr[0]
            
            lower_freq = freq - mask_percentage * freq
            upper_freq = freq + mask_percentage * freq

            lower_idx = int(self.xp.searchsorted(self.f_arr, lower_freq - df/2))
            upper_idx = int(self.xp.searchsorted(self.f_arr, upper_freq + df/2))
            
            all_indices.update(range(lower_idx, upper_idx))
        
        return self.xp.array(sorted(all_indices), dtype=self.xp.int32)

    def _get_dips_indices(self,):

        transfer_functions = self.compute_transfer_functions(self.f_arr)

        tf = transfer_functions[0]

        dips_indices = [
            self._find_dips_with_percentage(tf[t_idx], mask_percentage=self.mask_percentage)
            for t_idx in range(self.num_times)
        ]

        return dips_indices


    def _compute_matrix_elements(self, 
                                 freqs, 
                                 Soms_d_in=15e-12, 
                                 Sa_a_in=3e-15, 
                                 Amp=0, 
                                 alpha=0, 
                                 sl1=0, 
                                 kn=0, 
                                 sl2=0, 
                                 knots_position_all: np.ndarray | cp.ndarray = None,
                                 knots_amplitude_all: np.ndarray | cp.ndarray = None,
                                 ):
        """Compute the 6 sensitivity matrix terms using the c++ backend."""
        
        xp = self.xp
        total_terms = self.basis_settings.total_terms
        
        c00 = xp.empty(total_terms, dtype=xp.float64)
        c11 = xp.empty(total_terms, dtype=xp.float64)
        c22 = xp.empty(total_terms, dtype=xp.float64)
        c01 = xp.empty(total_terms, dtype=xp.complex128)
        c02 = xp.empty(total_terms, dtype=xp.complex128)
        c12 = xp.empty(total_terms, dtype=xp.complex128)

        if self.use_splines:
            assert knots_position_all is not None and knots_amplitude_all is not None
            splines_out = self.spline_interpolant(freqs, knots_position_all, knots_amplitude_all)
            splines_in_isi_oms = splines_out[0]
            spline_in_testmass = splines_out[1]
        else:
            splines_in_isi_oms = xp.zeros(len(freqs), dtype=xp.float64)
            spline_in_testmass = xp.zeros(len(freqs), dtype=xp.float64)

        self.pycpp_sensitivity_matrix.get_noise_covariance_wrap(
            xp.asarray(freqs),
            self.time_indices,
            float(Soms_d_in),
            float(Sa_a_in),
            float(Amp),
            float(alpha),
            float(sl1),
            float(kn),
            float(sl2),
            splines_in_isi_oms,
            spline_in_testmass,
            c00, c01, c02, c11, c12, c22,
            len(freqs),
            len(self.time_indices)
        )

        return c00, c11, c22, c01, c02, c12
    
    def _fill_matrix(self, c00, c11, c22, c01, c02, c12):
        """Fill the full 3x3 sensitivity matrix from its 6 unique elements."""
        xp = self.xp    
        shape = self.basis_settings.basis_shape

        # Reshape views (no copy)
        c00 = c00.reshape(shape)
        c11 = c11.reshape(shape)
        c22 = c22.reshape(shape)
        c01 = c01.reshape(shape)
        c02 = c02.reshape(shape)
        c12 = c12.reshape(shape)

        # Direct assignment is faster than stack (no intermediate copies)
        matrix = xp.empty(self.channel_shape + shape, dtype=xp.complex128)
        matrix[0, 0] = c00
        matrix[1, 1] = c11
        matrix[2, 2] = c22
        matrix[0, 1] = c01
        matrix[1, 0] = c01.conj()
        matrix[0, 2] = c02
        matrix[2, 0] = c02.conj()
        matrix[1, 2] = c12
        matrix[2, 1] = c12.conj()
        
        return matrix
    
    def _extract_matrix_elements(self, matrix_in, flatten=False):
        """Extract the 6 unique sensitivity matrix elements from the full 3x3 matrix."""

        c00 = matrix_in[0, 0].real
        c11 = matrix_in[1, 1].real
        c22 = matrix_in[2, 2].real
        c01 = matrix_in[0, 1]
        c02 = matrix_in[0, 2]
        c12 = matrix_in[1, 2]

        if flatten:
           return c00.flatten(), c11.flatten(), c22.flatten(), c01.flatten(), c02.flatten(), c12.flatten()

        return c00, c11, c22, c01, c02, c12

    
    def compute_sensitivity_matrix(self, freqs, Soms_d_in=15e-12, Sa_a_in=3e-15, Amp=0, alpha=0, sl1=0, kn=0, sl2=0, knots_position_all: np.ndarray | cp.ndarray = None,
                                   knots_amplitude_all: np.ndarray | cp.ndarray = None,):
        """Compute the full 3x3 sensitivity matrix using the c++ backend."""
        c00, c11, c22, c01, c02, c12 = self._compute_matrix_elements(
            freqs, Soms_d_in, Sa_a_in, Amp, alpha, sl1, kn, sl2, knots_position_all, knots_amplitude_all
        )
        matrix = self._fill_matrix(c00, c11, c22, c01, c02, c12)
        return matrix

    def set_sensitivity_matrix(self, 
                               Soms_d_in: float = 15e-12, 
                               Sa_a_in: float = 3e-15, 
                               knots_position_all: np.ndarray | cp.ndarray = None,
                               knots_amplitude_all: np.ndarray | cp.ndarray = None,
                               Amp: float = 0., 
                               alpha: float = 0., 
                               sl1: float = 0., 
                               kn: float = 0., 
                               sl2: float = 0., 
                               ):
        """Internally store the sensitivity matrix computed at the basis frequencies."""

        
        
        c00, c11, c22, c01, c02, c12 = self._compute_matrix_elements(
            self.f_arr, Soms_d_in, Sa_a_in, Amp, alpha, sl1, kn, sl2, knots_position_all, knots_amplitude_all
        )

        sens_mat = self._fill_matrix(c00, c11, c22, c01, c02, c12)

        self.sens_mat = self.smooth_sensitivity_matrix(sens_mat, sigma=5)

    
    def _setup_det_and_inv(self):
        """use the c++ backend to compute the log-determinant and inverse of the sensitivity matrix."""
        c00, c11, c22, c01, c02, c12 = self._extract_matrix_elements(self.sens_mat, flatten=True)
        self.invC, self.detC = self._inverse_det_wrapper(c00, c11, c22, c01, c02, c12)

    def _inverse_det_wrapper(self, 
                             c00: np.ndarray | cp.ndarray, 
                             c11: np.ndarray | cp.ndarray, 
                             c22: np.ndarray | cp.ndarray, 
                             c01: np.ndarray | cp.ndarray, 
                             c02: np.ndarray | cp.ndarray, 
                             c12: np.ndarray | cp.ndarray 
                             ) -> tuple:
        
        """Wrapper to call c++ backend for inverse log-determinant computation."""
        
        xp = self.xp
        total_terms = self.basis_settings.total_terms

        i00 = xp.empty(total_terms, dtype=xp.float64)
        i11 = xp.empty(total_terms, dtype=xp.float64)
        i22 = xp.empty(total_terms, dtype=xp.float64)
        i01 = xp.empty(total_terms, dtype=xp.complex128)
        i02 = xp.empty(total_terms, dtype=xp.complex128)
        i12 = xp.empty(total_terms, dtype=xp.complex128)

        det = xp.empty(total_terms, dtype=xp.float64)

        self.pycpp_sensitivity_matrix.get_inverse_det_wrap(
            c00, c01, c02, c11, c12, c22,
            i00, i01, i02, i11, i12, i22,
            det,
            total_terms
        )
        
        inverse_matrix = self._fill_matrix(i00, i11, i22, i01, i02, i12)

        return inverse_matrix, det.reshape(self.basis_settings.basis_shape)

    def compute_inverse_det(self, 
                            matrix_in: np.ndarray | cp.ndarray 
                            ) -> tuple:
        """
        Invert the 3x3 sensitivity matrix and compute its log-determinant with the c++ backend.

        Args:
            matrix_in: Input sensitivity matrix. Shape (3, 3, ...)
        
        Returns:
            inverse_matrix: Inverted sensitivity matrix. Shape (3, 3, ...)
            det: Determinant of the sensitivity matrix. Shape (...)
        """
        c00, c11, c22, c01, c02, c12 = self._extract_matrix_elements(matrix_in, flatten=True)
        inverse_matrix, det = self._inverse_det_wrapper(c00, c11, c22, c01, c02, c12)
        return inverse_matrix, det

    def compute_transfer_functions(self, 
                                   freqs: np.ndarray | cp.ndarray 
                                   ) -> tuple:
        
        """Compute transfer functions using the c++ backend."""

        xp = self.xp
        num_freqs = len(freqs)

        total_shape = self.num_times * num_freqs

        oms_xx = xp.empty(shape=(total_shape,), dtype=xp.float64)
        oms_yy = xp.empty(shape=(total_shape,), dtype=xp.float64)
        oms_zz = xp.empty(shape=(total_shape,), dtype=xp.float64)
        oms_xy = xp.empty(shape=(total_shape,), dtype=xp.complex128)
        oms_xz = xp.empty(shape=(total_shape,), dtype=xp.complex128)
        oms_yz = xp.empty(shape=(total_shape,), dtype=xp.complex128)

        tm_xx = xp.empty(shape=(total_shape,), dtype=xp.float64)
        tm_yy = xp.empty(shape=(total_shape,), dtype=xp.float64)
        tm_zz = xp.empty(shape=(total_shape,), dtype=xp.float64)
        tm_xy = xp.empty(shape=(total_shape,), dtype=xp.complex128)
        tm_xz = xp.empty(shape=(total_shape,), dtype=xp.complex128)
        tm_yz = xp.empty(shape=(total_shape,), dtype=xp.complex128)

        self.pycpp_sensitivity_matrix.get_noise_tfs_wrap(
            xp.asarray(freqs),
            oms_xx, oms_xy, oms_xz, oms_yy, oms_yz, oms_zz,
            tm_xx, tm_xy, tm_xz, tm_yy, tm_yz, tm_zz,
            num_freqs, self.num_times,
            self._time_indices
        )

        return (oms_xx.reshape(self.num_times, num_freqs), 
                oms_xy.reshape(self.num_times, num_freqs),
                oms_xz.reshape(self.num_times, num_freqs),
                oms_yy.reshape(self.num_times, num_freqs), 
                oms_yz.reshape(self.num_times, num_freqs),
                oms_zz.reshape(self.num_times, num_freqs), 
                tm_xx.reshape(self.num_times, num_freqs), 
                tm_xy.reshape(self.num_times, num_freqs),
                tm_xz.reshape(self.num_times, num_freqs),
                tm_yy.reshape(self.num_times, num_freqs), 
                tm_yz.reshape(self.num_times, num_freqs),
                tm_zz.reshape(self.num_times, num_freqs)
                )

    def compute_log_like(self,
                         data_in_all: np.ndarray | cp.ndarray, 
                         data_index_all: np.ndarray | cp.ndarray,
                         Soms_in_all: np.ndarray | cp.ndarray, 
                         Sa_in_all: np.ndarray | cp.ndarray,
                         Amp_in_all: np.ndarray | cp.ndarray,
                         alpha_in_all: np.ndarray | cp.ndarray,
                         sl1_in_all: np.ndarray | cp.ndarray,
                         kn_in_all: np.ndarray | cp.ndarray,
                         sl2_in_all: np.ndarray | cp.ndarray,
                         knots_position_all: np.ndarray | cp.ndarray = None,
                         knots_amplitude_all: np.ndarray | cp.ndarray = None,
                         ) -> np.ndarray | cp.ndarray:
        """
        Compute log-likelihood using the c++ backend.

        Args:
            data_in_all: Input data array. Shape (num_psds, num_freqs * num_times)
            data_index_all: Data indices array to keep track of which data corresponds to which PSD. Shape (num_psds)
            Soms_in_all: Displacement noise levels for each walker. Shape (num_psds)
            Sa_in_all: Acceleration noise levels for each walker. Shape (num_psds)
            Amp_in_all: Galactic foreground amplitude for each walker. Shape (num_psds)
            alpha_in_all: Galactic foreground alpha for each walker. Shape (num_psds)
            sl1_in_all: First galactic foreground slope parameter for each walker. Shape (num_psds)
            kn_in_all: Galactic foreground knee frequency parameter for each walker. Shape (num_psds)
            sl2_in_all: Second galactic foreground slope parameter for each walker. Shape (num_psds)
            knots_position_all: Positions of spline knots for noise modeling. Shape (2 * num_psds, num_knots)
            knots_amplitude_all: Amplitudes of spline knots for noise modeling. Shape (2 * num_psds, num_knots)

        Returns:
            log_like_out: Computed log-likelihoods for each PSD. Shape (num_psds,)
        """

        xp = self.xp

        num_psds = len(Soms_in_all)

        log_like_out = xp.zeros(shape=(num_psds,), dtype=xp.float64)

        if self.use_splines:
            splines_weights = self.spline_interpolant(xp.log10(self.f_arr), knots_position_all, knots_amplitude_all)

            splines_weights_isi_oms = splines_weights[:num_psds].flatten()
            splines_weights_testmass = splines_weights[num_psds:].flatten()

        else:
            splines_weights_isi_oms = xp.zeros(shape=(num_psds * self.num_freqs))
            splines_weights_testmass = xp.zeros(shape=(num_psds * self.num_freqs))
    
        self.pycpp_sensitivity_matrix.psd_likelihood_wrap(
            log_like_out,
            self.f_arr,
            xp.asarray(data_in_all.flatten()),
            xp.asarray(data_index_all.flatten()),
            xp.asarray(self.time_indices),
            xp.asarray(Soms_in_all),
            xp.asarray(Sa_in_all),
            xp.asarray(Amp_in_all),
            xp.asarray(alpha_in_all),
            xp.asarray(sl1_in_all),
            xp.asarray(kn_in_all),
            xp.asarray(sl2_in_all),
            xp.asarray(splines_weights_isi_oms),
            xp.asarray(splines_weights_testmass), 
            self.basis_settings.differential_component,
            self.num_freqs,
            self.num_times,
            self.dips_mask,
            num_psds
        )

        return log_like_out
    
    def smooth_sensitivity_matrix(self,
                                  matrix_in: np.ndarray | cp.ndarray,
                                  sigma: float = 5.0,
                                  ) -> np.ndarray | cp.ndarray:
        
        """
        Perform log-frequency smoothing of the sensitivity matrix to get rid of the very sharp dips.

        Args:
            matrix_in: Input sensitivity matrix. Shape (3, 3, num_times, num_freqs)
            sigma: Width of the Gaussian smoothing kernel in frequency bins.
        """
        filter_func = np_gaussian_filter1d if self.xp == np else cp_gaussian_filter1d
        
        smoothed_matrix = matrix_in.copy()
        mask = self.dips_mask.reshape(self.num_times, self.num_freqs)
        _smoothed = filter_func(matrix_in, sigma=sigma, axis=-1)

        smoothed_matrix[..., mask] = _smoothed[..., mask]

        return smoothed_matrix
        


    def __call__(self, 
                name: str,
                psd_params: np.ndarray, 
                galfor_params: np.ndarray=None
                ) -> XYZSensitivityBackend:
        """
        Update the internal sensitivity matrix with new noise parameters and return to be used in a AnalysisContainer.

        Args:
            psd_params: Array of PSD parameters in order [Soms_d, Sa_a, (optional spline params...)]
            galfor_params: Array of galactic foreground parameters in order [Amp, alpha, sl1, kn, sl2].
        
        Returns:
            self: a configured copy of the sensitivity matrix backend.
        """

        new_sens_mat = XYZSensitivityBackend(**self.kwargs)

        self.name = name

        Soms_d = psd_params[0]
        Sa_a = psd_params[1]

        if self.use_splines:
            # todo add a container for the noise
            spline_params = psd_params[2:]
            spline_knots_position = spline_params[::2]
            spline_knots_amplitude = spline_params[1::2]

        else:
            spline_knots_position = None
            spline_knots_amplitude = None
        
        if galfor_params is None:
            galfor_params = np.zeros(5)
    
        new_sens_mat.set_sensitivity_matrix(
            Soms_d,
            Sa_a,
            spline_knots_position,
            spline_knots_amplitude,
            *galfor_params
        )

        return new_sens_mat