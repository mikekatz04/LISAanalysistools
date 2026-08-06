"""LISA PSD / sensitivity matrices, TDI noise channels, and PSD utilities.

The sensitivity code is heavily based on an original code by
Stas Babak and Antoine Petiteau for the LDC team.
"""

from __future__ import annotations

import functools
import math
import operator
import os
import warnings
from abc import ABC
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from logging import getLogger
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d as np_gaussian_filter1d
from scipy.signal import find_peaks

from .utils.utility import asnumpy, get_array_module

from . import domains

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d

except (ModuleNotFoundError, ImportError):
    import numpy as cp

    cp_gaussian_filter1d = np_gaussian_filter1d

from cudakima import AkimaInterpolant1D

from . import detector as lisa_models
from . import _unequal_arm_expressions as _ua_expr
from .detector import L1Orbits, Orbits
from .domains import DomainSettingsBase
from .stochastic import (
    FittedHyperbolicTangentGalacticForeground,
    HyperbolicTangentGalacticForeground,
    StochasticContribution,
    check_stochastic,
)
from .utils.constants import *
from .utils.parallelbase import LISAToolsParallelModule
from .utils.utility import AET, get_array_module

if TYPE_CHECKING:
    from .utils.typing import NDArrayLike, ArrayModule

logger = getLogger(__name__)

#: Counts already reported by :func:`_warn_zeroed_invc`. The analytic noise
#: model diverges at f -> 0, so the same pixels are re-zeroed on every
#: sensitivity rebuild (per walker, per PSD proposal). Warn once per distinct
#: count -- a *changed* count is new information and warns again; a repeat is
#: demoted to debug so long runs aren't buried in identical warnings.
_INVC_ZEROED_REPORTED: set = set()


def _warn_zeroed_invc(n_bad: int) -> None:
    """Report zeroed non-finite inverse-covariance elements (once per count)."""
    msg = (
        "sensitivity invC: zeroed %d non-finite element(s) (infinite-noise / "
        "singular-covariance pixels -> zero weight; expected for the "
        "analytic-PSD f=0 WDM layer)."
    )
    if n_bad in _INVC_ZEROED_REPORTED:
        logger.debug(msg, n_bad)
    else:
        _INVC_ZEROED_REPORTED.add(n_bad)
        logger.warning(msg + " Further identical reports are logged at DEBUG.", n_bad)

def _mat3x3_det_inv(C: np.ndarray, xp) -> tuple:
    """Determinant and inverse of a stack of 3x3 matrices, via the adjugate.

    ``C`` has shape ``(3, 3, *data_shape)``; the returned determinant has shape
    ``data_shape`` and the inverse ``(3, 3, *data_shape)``.

    This replaces a ``transpose -> reshape -> xp.linalg.det/inv`` round trip
    over ``prod(data_shape)`` tiny matrices, which is dominated by LAPACK
    per-matrix overhead (~40x slower here on a 59x1024 TDI grid). It is not a
    precision compromise: on the stock XYZ TDI covariances the adjugate's
    residual ``||C^-1 C - I||`` measures *better* than LAPACK's, both sitting
    at the ~1e-9 floor set by the matrices' own conditioning (max cond ~1e7).

    The general (not symmetric-specialised) cofactor expansion is used so this
    stays correct for any 3x3 stack, symmetric or not.

    NaN off-diagonals are zeroed before inversion, and non-invertible pixels
    fall out as zero inverse-covariance weight / unit determinant -- matching
    the general path below.
    """
    det = (
        C[0, 0] * (C[1, 1] * C[2, 2] - C[1, 2] * C[2, 1])
        - C[0, 1] * (C[1, 0] * C[2, 2] - C[1, 2] * C[2, 0])
        + C[0, 2] * (C[1, 0] * C[2, 1] - C[1, 1] * C[2, 0])
    )

    # adjust for nans in off-diagonals (the reference path does this AFTER
    # taking the determinant, so a NaN off-diagonal still poisons detC and is
    # sanitised to 1 below -- preserved here).
    M = C
    off_nan = xp.isnan(C)
    for i in range(3):
        off_nan[i, i] = False
    if bool(xp.any(off_nan)):
        M = xp.where(off_nan, xp.zeros_like(C), C)

    adj_det = (
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        idet = 1.0 / adj_det
        inv = xp.empty_like(M)
        # adjugate transpose: inv[i, j] = cofactor[j, i] / det
        inv[0, 0] = (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) * idet
        inv[0, 1] = (M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2]) * idet
        inv[0, 2] = (M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]) * idet
        inv[1, 0] = (M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2]) * idet
        inv[1, 1] = (M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]) * idet
        inv[1, 2] = (M[0, 2] * M[1, 0] - M[0, 0] * M[1, 2]) * idet
        inv[2, 0] = (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]) * idet
        inv[2, 1] = (M[0, 1] * M[2, 0] - M[0, 0] * M[2, 1]) * idet
        inv[2, 2] = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) * idet

    return det, inv


NUM_SPLINE_THREADS = 256

class Sensitivity(ABC):
    """Base Class for PSD information.

    The initialization function is only needed if using a file input.

    """

    channel: str = None

    #: Whether :meth:`transform` is a linear map of the
    #: :class:`~lisatools.detector.CurrentNoises` levels -- i.e. a sum of
    #: ``transfer_i(f) * noise_level_i`` with parameter-free transfer functions.
    #: Every stock TDI transform in this module is. Consumers use this to
    #: precompute the covariance at unit noise levels and recombine it linearly
    #: instead of re-evaluating the model per proposal (see
    #: :meth:`InstrumentNoise.base_covariance`). A subclass whose ``transform``
    #: mixes the levels non-linearly MUST set this to ``False``.
    linear_in_noise_levels: bool = True

    @staticmethod
    def get_xp(array: np.ndarray) -> object:
        """Numpy or Cupy (or float)"""
        try:
            return get_array_module(array)
        except ValueError:
            if isinstance(array, float):
                return np
            raise ValueError("array must be a numpy or cupy array (it can be a float as well).")

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
        include_instrument: bool = True,
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
            include_instrument: If ``True`` (default), include the instrument
                noise term. If ``False``, return only the (transformed) stochastic
                contribution — ``model`` is then unused. Used to build
                stochastic-only covariance components (galactic foreground, SGWB).
            **kwargs: For interoperability.

        Returns:
            PSD values.

        """
        if include_instrument:
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
        else:
            # stochastic-only: skip the instrument term entirely (no model needed)
            Sout = 0.0

        # The stochastic term is genuinely zero unless a stochastic model was
        # requested, but evaluating it is not free -- ``stochastic_transform``
        # builds its transfer function over the whole frequency array. Skip it
        # outright rather than adding a computed zero (this is ~6% of a WDM
        # noise-PE run, where the instrument covariance is rebuilt per
        # proposal). ``_has_stochastic`` is the same condition
        # ``get_stochastic_contribution`` uses to decide whether to fill.
        if cls._has_stochastic(**kwargs):
            stochastic_contribution = cls.stochastic_transform(
                f, cls.get_stochastic_contribution(f, **kwargs), **kwargs
            )
            Sout += stochastic_contribution
        elif not include_instrument:
            # no instrument term AND no stochastic term: still return an array
            # shaped like ``f`` rather than the scalar 0.0.
            Sout = cls.get_xp(f).zeros_like(f)
        return Sout

    @staticmethod
    def _has_stochastic(
        stochastic_params: Optional[tuple] = (),
        stochastic_kwargs: Optional[dict] = {},
        stochastic_function: Optional[StochasticContribution | str] = None,
        **kwargs: dict,
    ) -> bool:
        """Whether :meth:`get_stochastic_contribution` would produce a non-zero term."""
        return bool(
            (stochastic_params is not None and tuple(stochastic_params) != tuple())
            or (stochastic_kwargs is not None and stochastic_kwargs != {})
            or stochastic_function is not None
        )

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

            sgal[:] = stochastic_function.get_Sh(f, *stochastic_params, **stochastic_kwargs)

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
    """Sensitivity for the TDI 1.5 X channel."""

    channel: str = "X"

    @staticmethod
    def Cxx(f: float | np.ndarray) -> float | np.ndarray:
        """Common TDI 1.5 X auto-spectrum transfer factor.

        Args:
            f: Frequencyies to evaluate.

        Returns:
            Cxx: Transform factor.

        """
        xp = Sensitivity.get_xp(f)
        x = 2 * np.pi * f * L_SI / C_SI
        return 16.0 * xp.sin(x) ** 2

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

        xp = Sensitivity.get_xp(f)
        assert noise_levels.units == "relative_frequency"
        Cxx = X1TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI
        # TODO: need to check these
        isi_rfi_readout_transfer = Cxx
        tmi_readout_transfer = Cxx * (2.0 * (1.0 + xp.cos(x) ** 2))
        tm_transfer = Cxx * (2.0 * (1.0 + xp.cos(x) ** 2))
        rfi_backlink_transfer = Cxx
        tmi_backlink_transfer = Cxx * (2.0 * (1.0 + xp.cos(x) ** 2))

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise

        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        t = 4.0 * x**2 * xp.sin(x) ** 2
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
    """Sensitivity for the TDI 1.5 XY cross-spectrum channel."""

    channel: str = "XY"

    @staticmethod
    def Cxy(f: float | np.ndarray) -> float | np.ndarray:
        """Common TDI transform factor for CSD.

        Args:
            f: Frequencyies to evaluate.

        Returns:
            Cxy: Transform factor.

        """
        xp = Sensitivity.get_xp(f)
        x = 2 * np.pi * f * L_SI / C_SI
        return -4.0 * xp.sin(2 * x) * xp.sin(x)

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

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        # TODO: check these functions
        # GB = -0.5 of X
        t = -0.5 * (4.0 * x**2 * xp.sin(x) ** 2)
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
    """Sensitivity for the TDI 2.0 X channel."""

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
        xp = Sensitivity.get_xp(f)
        x = 2 * np.pi * f * L_SI / C_SI
        return (
            16.0 * xp.sin(x) ** 2 * xp.sin(2 * x) ** 2
        )  # xp.abs(1. - xp.exp(-2j * np.pi * f * L_SI / C_SI) ** 2) ** 2

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

        xp = Sensitivity.get_xp(f)
        assert noise_levels.units == "relative_frequency"
        Cxx = X2TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        isi_rfi_readout_transfer = 4.0 * Cxx
        tmi_readout_transfer = Cxx * (3 + xp.cos(2 * x))
        tm_transfer = 4 * Cxx * (3 + xp.cos(2 * x))
        rfi_backlink_transfer = 4 * Cxx
        tmi_backlink_transfer = Cxx * (3 + xp.cos(2 * x))

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise

        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        # TDI-2. The stochastic term carries the SAME generation as this
        # class's instrument ``transform`` (Cxx = 16 sin^2(x) sin^2(2x)), so
        # the TDI-1.5 kernel 4 x^2 sin^2(x) needs the 1.5 -> 2 conversion
        # factor 4 sin^2(2x) on top of ldasoft's x^2 sin^2(x) base:
        #     4 x^2 sin^2(x) * sin^2(2x)  ==  [x^2 sin^2(x)] * [4 sin^2(2x)]
        # Fixed 2026-08 (was the bare TDI-1.5 kernel, behind a "check these
        # functions for TDI2" TODO). sin^2(2x) ~ 1/41 at 1.5 mHz, so the old
        # form made the foreground ~1.6 decades too loud for a given amp and
        # drove the galfor amplitude down into its prior floor.
        # Verified against the GALFOR 731 d brick through both X (median
        # residual -0.011 dec) and A (-0.167 dec, vs -0.176 predicted by
        # C_AA = C_XX - C_XY = 3/2 C_XX for an isotropic background).
        # NOTE: the TDI-1.5 siblings (X1/A1/T1/XY1) keep 4 x^2 sin^2(x). Under
        # the same ldasoft convention that base should be x^2 sin^2(x), i.e.
        # 4x smaller -- unverified here for lack of TDI-1.5 data.
        t = 4.0 * x**2 * xp.sin(x) ** 2 * xp.sin(2.0 * x) ** 2
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
        xp = Sensitivity.get_xp(f)
        x = 2 * np.pi * f * L_SI / C_SI

        return -16.0 * xp.sin(x) * xp.sin(2.0 * x) ** 3

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

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
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
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        # TDI-2. The stochastic term carries the SAME generation as this
        # class's instrument ``transform`` (Cxx = 16 sin^2(x) sin^2(2x)), so
        # the TDI-1.5 kernel 4 x^2 sin^2(x) needs the 1.5 -> 2 conversion
        # factor 4 sin^2(2x) on top of ldasoft's x^2 sin^2(x) base:
        #     4 x^2 sin^2(x) * sin^2(2x)  ==  [x^2 sin^2(x)] * [4 sin^2(2x)]
        # Fixed 2026-08 (was the bare TDI-1.5 kernel, behind a "check these
        # functions for TDI2" TODO). sin^2(2x) ~ 1/41 at 1.5 mHz, so the old
        # form made the foreground ~1.6 decades too loud for a given amp and
        # drove the galfor amplitude down into its prior floor.
        # Verified against the GALFOR 731 d brick through both X (median
        # residual -0.011 dec) and A (-0.167 dec, vs -0.176 predicted by
        # C_AA = C_XX - C_XY = 3/2 C_XX for an isotropic background).
        # NOTE: the TDI-1.5 siblings (X1/A1/T1/XY1) keep 4 x^2 sin^2(x). Under
        # the same ldasoft convention that base should be x^2 sin^2(x), i.e.
        # 4x smaller -- unverified here for lack of TDI-1.5 data.
        t = -0.5 * (4.0 * x**2 * xp.sin(x) ** 2 * xp.sin(2.0 * x) ** 2)
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
    """Sensitivity for the TDI 1.5 A channel (orthogonal A/E/T basis)."""

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
        if np.any(
            np.asarray(
                [
                    noise_levels.rfi_backlink_noise,
                    noise_levels.tmi_backlink_noise,
                    noise_levels.rfi_oms_noise,
                    noise_levels.tmi_oms_noise,
                ]
            )
            != 0.0
        ):
            raise NotImplementedError(
                "ExtendedLISAModel has not been implemented yet for A1/E1/T1."
            )

        xp = Sensitivity.get_xp(f)
        assert noise_levels.units == "relative_frequency"
        Cxx = X1TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        # these are WRONG
        tmi_readout_transfer = Cxx * (2.0 * (1.0 + xp.cos(x) ** 2))
        rfi_backlink_transfer = Cxx
        tmi_backlink_transfer = Cxx * (2.0 * (1.0 + xp.cos(x) ** 2))

        # these are right and were changed accordingly
        # Need to find a citation for these 1st gen stuff
        # all that is needed for old model type
        isi_rfi_readout_transfer = 1 / 2 * Cxx * (2.0 + xp.cos(x))
        tm_transfer = Cxx * (3.0 + 2.0 * xp.cos(x) + xp.cos(2 * x))

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise

        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        t = 4.0 * x**2 * xp.sin(x) ** 2
        return 1.5 * (Sh * t)


class E1TDISens(A1TDISens):
    channel: str = "E"
    __doc__ = A1TDISens.__doc__
    pass


class T1TDISens(Sensitivity):
    """Sensitivity for the TDI 1.5 T channel (null channel of A/E/T basis)."""

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

        xp = Sensitivity.get_xp(f)
        assert noise_levels.units == "relative_frequency"

        Cxx = X1TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        # these are WRONG
        if np.any(
            np.asarray(
                [
                    noise_levels.rfi_backlink_noise,
                    noise_levels.tmi_backlink_noise,
                    noise_levels.rfi_oms_noise,
                    noise_levels.tmi_oms_noise,
                ]
            )
            != 0.0
        ):
            raise NotImplementedError(
                "ExtendedLISAModel has not been implemented yet for A1/E1/T1."
            )
        tmi_readout_transfer = Cxx * (2.0 * (1.0 + xp.cos(x) ** 2))
        rfi_backlink_transfer = Cxx
        tmi_backlink_transfer = Cxx * (2.0 * (1.0 + xp.cos(x) ** 2))

        # these are right and were changed accordingly
        # Need to find a citation for these 1st gen stuff
        # all that is needed for old model type
        isi_rfi_readout_transfer = Cxx * (1 - xp.cos(x))
        tm_transfer = 8.0 * Cxx * xp.sin(x / 2.0) ** 4

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise

        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        t = 4.0 * x**2 * xp.sin(x) ** 2
        return 0.0 * (Sh * t)

class A2TDISens(X2TDISens, Sensitivity):
    """Sensitivity for the TDI 2.0 A channel."""

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

        xp = Sensitivity.get_xp(f)
        assert noise_levels.units == "relative_frequency"
        Cxx = X2TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        isi_rfi_readout_transfer = 2.0 * Cxx * (2 + xp.cos(x))
        tmi_readout_transfer = Cxx * (3 + 2 * xp.cos(x) + xp.cos(2 * x))
        tm_transfer = 4 * Cxx * (3 + 2 * xp.cos(x) + xp.cos(2 * x))

        rfi_backlink_transfer = 2 * Cxx * (2 * xp.cos(x))
        tmi_backlink_transfer = Cxx * (3 + 2 * xp.cos(x) + xp.cos(2 * x))

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise

        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        # TDI-2. The stochastic term carries the SAME generation as this
        # class's instrument ``transform`` (Cxx = 16 sin^2(x) sin^2(2x)), so
        # the TDI-1.5 kernel 4 x^2 sin^2(x) needs the 1.5 -> 2 conversion
        # factor 4 sin^2(2x) on top of ldasoft's x^2 sin^2(x) base:
        #     4 x^2 sin^2(x) * sin^2(2x)  ==  [x^2 sin^2(x)] * [4 sin^2(2x)]
        # Fixed 2026-08 (was the bare TDI-1.5 kernel, behind a "check these
        # functions for TDI2" TODO). sin^2(2x) ~ 1/41 at 1.5 mHz, so the old
        # form made the foreground ~1.6 decades too loud for a given amp and
        # drove the galfor amplitude down into its prior floor.
        # Verified against the GALFOR 731 d brick through both X (median
        # residual -0.011 dec) and A (-0.167 dec, vs -0.176 predicted by
        # C_AA = C_XX - C_XY = 3/2 C_XX for an isotropic background).
        # NOTE: the TDI-1.5 siblings (X1/A1/T1/XY1) keep 4 x^2 sin^2(x). Under
        # the same ldasoft convention that base should be x^2 sin^2(x), i.e.
        # 4x smaller -- unverified here for lack of TDI-1.5 data.
        t = 4.0 * x**2 * xp.sin(x) ** 2 * xp.sin(2.0 * x) ** 2
        return Sh * t


class E2TDISens(A2TDISens):
    channel: str = "E"
    __doc__ = A2TDISens.__doc__
    pass


class T2TDISens(X2TDISens, Sensitivity):
    """Sensitivity for the TDI 2.0 T (null) channel."""

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

        xp = Sensitivity.get_xp(f)
        assert noise_levels.units == "relative_frequency"
        Cxx = X2TDISens.Cxx(f)

        x = 2 * np.pi * f * L_SI / C_SI

        isi_rfi_readout_transfer = 4.0 * Cxx * (1 - xp.cos(x))
        tmi_readout_transfer = 8 * Cxx * xp.sin(x / 2.0) ** 4
        tm_transfer = 32 * Cxx * xp.sin(x / 2.0) ** 4
        rfi_backlink_transfer = 4.0 * Cxx * (1 - xp.cos(x))
        tmi_backlink_transfer = 8 * Cxx * xp.sin(x / 2.0) ** 4

        isi_oms_ffd = isi_rfi_readout_transfer * noise_levels.isi_oms_noise
        rfi_oms_ffd = isi_rfi_readout_transfer * noise_levels.rfi_oms_noise
        tmi_oms_ffd = tmi_readout_transfer * noise_levels.tmi_oms_noise
        tm_noise_ffd = tm_transfer * noise_levels.tm_noise

        rfi_backlink_ffd = rfi_backlink_transfer * noise_levels.rfi_backlink_noise
        tmi_backlink_ffd = tmi_backlink_transfer * noise_levels.tmi_backlink_noise

        total_noise = (
            tm_noise_ffd
            + isi_oms_ffd
            + rfi_oms_ffd
            + tmi_oms_ffd
            + rfi_backlink_ffd
            + tmi_backlink_ffd
        )
        return total_noise

    @staticmethod
    def stochastic_transform(
        f: float | np.ndarray, Sh: float | np.ndarray, **kwargs: dict
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base stochastic functions to the XYZ stochastic TDI information.\n\n"
            + Sensitivity.stochastic_transform.__doc__.split("PSDs.\n\n")[-1]
        )
        xp = Sensitivity.get_xp(f)
        x = 2.0 * np.pi * lisaLT * f
        # TDI-2. The stochastic term carries the SAME generation as this
        # class's instrument ``transform`` (Cxx = 16 sin^2(x) sin^2(2x)), so
        # the TDI-1.5 kernel 4 x^2 sin^2(x) needs the 1.5 -> 2 conversion
        # factor 4 sin^2(2x) on top of ldasoft's x^2 sin^2(x) base:
        #     4 x^2 sin^2(x) * sin^2(2x)  ==  [x^2 sin^2(x)] * [4 sin^2(2x)]
        # Fixed 2026-08 (was the bare TDI-1.5 kernel, behind a "check these
        # functions for TDI2" TODO). sin^2(2x) ~ 1/41 at 1.5 mHz, so the old
        # form made the foreground ~1.6 decades too loud for a given amp and
        # drove the galfor amplitude down into its prior floor.
        # Verified against the GALFOR 731 d brick through both X (median
        # residual -0.011 dec) and A (-0.167 dec, vs -0.176 predicted by
        # C_AA = C_XX - C_XY = 3/2 C_XX for an isotropic background).
        # NOTE: the TDI-1.5 siblings (X1/A1/T1/XY1) keep 4 x^2 sin^2(x). Under
        # the same ldasoft convention that base should be x^2 sin^2(x), i.e.
        # 4x smaller -- unverified here for lack of TDI-1.5 data.
        t = 4.0 * x**2 * xp.sin(x) ** 2 * xp.sin(2.0 * x) ** 2
        return Sh * t


class LISASens(Sensitivity):
    """Base sensitivity curve for LISA (sky-/polarisation-averaged strain)."""

    @classmethod
    def get_Sn(
        cls,
        f: float | np.ndarray,
        model: Optional[lisa_models.LISAModel | str] = lisa_models.sangria,
        average: bool = True,
        include_instrument: bool = True,
        **kwargs: dict,
    ) -> float | np.ndarray:
        """Compute the base LISA sensitivity function.

        Args:
            f: Frequency array.
            model: Noise model. Object of type :class:`lisa_models.LISAModel`. It can also be a string corresponding to one of the stock models.
            average: Whether to apply averaging factors to sensitivity curve.
                Antenna response: ``av_resp = np.sqrt(5) if average else 1.0``
                Projection effect: ``Proj = 2.0 / np.sqrt(3) if average else 1.0``
            include_instrument: If ``True`` (default), include the instrument
                noise term. If ``False``, return only the stochastic contribution
                (``model`` is then unused).
            **kwargs: Keyword arguments to pass to :func:`get_stochastic_contribution`. # TODO: fix

        Returns:
            Sensitivity array.

        """
        xp = Sensitivity.get_xp(f)
        if include_instrument:
            model = lisa_models.check_lisa_model(model)

            if not isinstance(model, lisa_models.LISAModel):
                raise NotImplementedError(
                    "This function has not been implemented for ExtendedLISAModel yet."
                )

            # get noise values
            noise_values = model.lisanoises(f, unit="displacement")

            Sa_d = noise_values.tm_noise
            Sop = noise_values.isi_oms_noise

            all_m = xp.sqrt(4.0 * Sa_d + Sop)
            ## Average the antenna response (scalar constants -> host np is fine)
            av_resp = np.sqrt(5) if average else 1.0

            ## Projection effect
            Proj = 2.0 / np.sqrt(3) if average else 1.0

            ## Approximative transfer function
            f0 = 1.0 / (2.0 * lisaLT)
            a = 0.41
            T = xp.sqrt(1 + (f / (a * f0)) ** 2)
            sens = (av_resp * Proj * T * all_m / lisaL) ** 2
        else:
            # stochastic-only: skip the instrument term entirely (no model needed)
            sens = 0.0

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
    def get_Sn(f: float | np.ndarray, average: bool = True, **kwargs: dict) -> float | np.ndarray:
        """Cornish (2018) LISA PSD evaluated at ``f`` (Hz).

        Args:
            f: Frequency array in Hz.
            average: If ``True``, include the sky-averaging factor ``20/3``.
            **kwargs: Ignored; kept for interface compatibility.

        Returns:
            PSD values.
        """
        # TODO: documentation here
        xp = Sensitivity.get_xp(f)
        sky_averaging_constant = 20.0 / 3.0 if average else 1.0

        L = 2.5 * 10**9  # Length of LISA arm
        f0 = 19.09 * 10 ** (-3)  # transfer frequency

        # Optical Metrology Sensor
        Poms = ((1.5e-11) * (1.5e-11)) * (1 + xp.power((2e-3) / f, 4))

        # Acceleration Noise
        Pacc = (3e-15) * (3e-15) * (1 + (4e-4 / f) * (4e-4 / f)) * (1 + xp.power(f / (8e-3), 4))

        # constants for Galactic background after 1 year of observation
        alpha = 0.171
        beta = 292
        k = 1020
        gamma = 1680
        f_k = 0.00215

        # Galactic background contribution
        Sc = (
            9e-45
            * xp.power(f, -7 / 3)
            * xp.exp(-xp.power(f, alpha) + beta * f * xp.sin(k * f))
            * (1 + xp.tanh(gamma * (f_k - f)))
        )

        # PSD
        PSD = (sky_averaging_constant) * (
            (10 / (3 * L * L))
            * (Poms + (4 * Pacc) / (xp.power(2 * np.pi * f, 4)))
            * (1 + 0.6 * (f / f0) * (f / f0))
            + Sc
        )

        return PSD


class FlatPSDFunction(LISASens):
    """White Noise PSD function."""

    @classmethod
    def get_Sn(cls, f: float | np.ndarray, val: float, **kwargs: dict) -> float | np.ndarray:
        """Return ``val`` broadcast to the shape of ``f`` (a flat / white PSD).

        Args:
            f: Frequency array (or scalar).
            val: The constant PSD value.
            **kwargs: Ignored; kept for interface compatibility.

        Returns:
            Either an array of shape ``f.shape`` filled with ``val``, or a
            Python ``float`` if ``f`` was a scalar.
        """
        # TODO: documentation here
        xp = cls.get_xp(f)
        out = xp.full_like(f, val)
        if isinstance(f, float):
            out = out.item()
        return out


class SensitivityMatrixBase:
    """Base Container to hold sensitivity information.

    Args:
        basis_settings: Frequency array in FD. Time array in TD. Wavelet basis in WDM. Etc.
        skip_inv_det: Whether to skip the determinant check when updating sensitivities. This is relevant for slicing operations.
    """

    def __init__(
        self,
        settings: domains.DomainSettingsBase,
        skip_inv_det: bool = False,
    ) -> None:
        self.basis_settings = settings
        self.data_shape = self.basis_settings.basis_shape_active

        self.do_inv_det = not skip_inv_det
        # invC / detC are evaluated lazily on first read; this flag tracks
        # whether sens_mat has changed since the last computation. Subsequent
        # in-place updates (``__setitem__``) and arithmetic ops (``__add__`` /
        # ``__sub__``) flip this back to True so the inverse is recomputed only
        # when the caller actually needs it.
        self._inv_det_dirty = False

    @property
    def basis_settings(self) -> domains.DomainSettingsBase:
        """Domain settings (frequency / time / TF) the matrix is evaluated on."""
        return self._basis_settings

    @basis_settings.setter
    def basis_settings(self, basis_settings: domains.DomainSettingsBase) -> None:
        """Set the domain settings (must be a :class:`~lisatools.domains.DomainSettingsBase`)."""
        assert isinstance(basis_settings, domains.DomainSettingsBase)
        self._basis_settings = basis_settings

    def check_update(self):
        """Raise if the original input was raw arrays (rather than callables) and cannot be re-evaluated."""
        if not self.can_redo:
            raise ValueError(
                "Cannot update sensitivities because original input was arrays rather than functions."
            )

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

        elif isinstance(sens_mat, list) or (
            isinstance(sens_mat, np.ndarray) and sens_mat.dtype == object
        ):
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
                            raise ValueError(
                                "If entering a list of arrays, arrays must be 1D on the last dimension of the list structure."
                            )
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
                    raise ValueError(
                        "Matrix element must be Sensitivity object, string representing a sensitivity object, or an array with values."
                    )

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
            xp = get_array_module(self.basis_settings.f_arr)
            # xp = np
            if self.is_array_base:
                _sens_mat = xp.asarray(sens_mat)

            else:
                _flattened_arr = np.asarray(sens_mat, dtype=object).flatten()
                _sens_mat = xp.zeros((num_components,) + self.basis_settings.basis_shape_active)
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
            self._sens_mat = _sens_mat.reshape(tuple(outer_shape) + self.basis_settings.basis_shape_active)

        else:
            raise ValueError("Must input array or list.")

        self.channel_shape = self._sens_mat.shape[: -len(self.data_shape)]

        # Defer inv/det computation: a subsequent read of ``invC`` / ``detC``
        # will trigger ``_setup_det_and_inv`` once. This lets a chain like
        # ``A + B + C`` pay the inverse cost only at the final access, not
        # after every intermediate operation.
        self._inv_det_dirty = True

    @property
    def differential_component(self) -> float:
        """Pass-through to :attr:`basis_settings.differential_component` (df / dt / etc.)."""
        return self.basis_settings.differential_component

    # use the getitem to get a slice of the sensitivity matrix, then use that to get the corresponding slice of the determinant and inverse
    def get_slice(self, index: tuple | slice) -> SensitivityMatrixBase:
        """
        Get a time and frequency slice of the sensitivity matrix, and corresponding slices of the determinant and inverse.

        Args:
            index (tuple | slice): Slice, or tuple of slices, to apply to the sensitivity matrix.
                                   The slice(s) should select part of the time and frequency dimensions of the sensitivity matrix, which are the last dimensions of the array.

        Returns:
            A new SensitivityMatrixBase object with the sliced sensitivity matrix, and corresponding sliced determinant and inverse.
        """
        new_settings = self.basis_settings.get_slice(index)
        new_mat = SensitivityMatrixBase(new_settings, skip_inv_det=True)

        # Normalize index to a tuple so that multi-dimensional basis slices
        # (e.g. (time_slice, freq_slice) for STFT) unpack correctly when
        # combined with Ellipsis for the channel dimensions.
        basis_idx = index if isinstance(index, tuple) else (index,)

        new_mat.sens_mat = self.sens_mat[(Ellipsis,) + basis_idx]
        new_mat.detC = self.detC[basis_idx]
        new_mat.invC = self.invC[(Ellipsis,) + basis_idx]

        # now set skip_inv_det to False
        new_mat.do_inv_det = True

        return new_mat

    # def _setup_det_and_inv(self):
    #     """Determinant and inverse of TDI matrix."""

    #     # setup detC
    #     xp = get_array_module(self.sens_mat)

    #     # setup detC
    #     if self.sens_mat.ndim < 3:
    #         self.detC = xp.prod(self.sens_mat, axis=0)
    #         self.invC = 1 / self.sens_mat

    #     else:
    #         full_shape = tuple(range(len(self.sens_mat.shape)))

    #         basis_axes = full_shape[-len(self.data_shape) :]
    #         mat_axes = full_shape[: -len(self.data_shape)]
    #         transpose_shape = basis_axes + mat_axes
    #         self.detC = xp.linalg.det(self.sens_mat.transpose(transpose_shape))
    #         invC = xp.zeros_like(self.sens_mat.transpose(transpose_shape))
    #         invC[self.detC != 0.0] = xp.linalg.inv(
    #             self.sens_mat.transpose(transpose_shape)[self.detC != 0.0]
    #         )
    #         invC[self.detC == 0.0] = 1e-100

    #         # switch them after they were effectively switched above
    #         self.invC = invC.transpose(transpose_shape)
    def _setup_det_and_inv(self) -> None:
        """Determinant and inverse of TDI matrix. (Patched version)"""
        
        # Check if a custom array module is used (like cupy), fallback to numpy
        try:
            xp = get_array_module(self.sens_mat)
        except NameError:
            xp = np

        # setup detC
        if len(self.channel_shape) == 1:
            self._detC = xp.prod(self.sens_mat, axis=0)
            self._invC = 1 / self.sens_mat

        elif tuple(self.channel_shape) == (3, 3):
            self._detC, self._invC = _mat3x3_det_inv(self.sens_mat, xp)

        # TODO switch to Cholesky decomposition and inversion!
        else:
            assert len(self.channel_shape) == 2
            full_shape = tuple(range(len(self.sens_mat.shape)))

            basis_axes = full_shape[-len(self.data_shape):]
            mat_axes = full_shape[:-len(self.data_shape)]
            transpose_shape = basis_axes + mat_axes
            self._detC = xp.linalg.det(self.sens_mat.transpose(transpose_shape))

            tmp = self.sens_mat.transpose(transpose_shape).reshape((-1,) + self.channel_shape)

            _invC = xp.zeros_like(tmp)

            # adjust for nans in off-diagonals (xp.isnan so this works on cupy;
            # ``inds``/``inds_bad`` below stay host arrays — they are Python-loop
            # batch/index bookkeeping, not device data)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        tmp[xp.isnan(tmp[:, i, j]), i, j] = 0.0

            batch = 100000
            inds = np.arange(0, tmp.shape[0], batch)
            if inds[0] < tmp.shape[0]:
                inds = np.concatenate([inds, np.array([tmp.shape[0]])])
            inds_bad = []
            for ind_st, ind_end in zip(inds[:-1], inds[1:]):
                try:
                    _invC[ind_st:ind_end] = xp.linalg.inv(tmp[ind_st:ind_end])
                except np.linalg.LinAlgError:
                    for i in range(ind_st, ind_end):
                        try:
                            _invC[i] = xp.linalg.inv(tmp[i])
                        except np.linalg.LinAlgError:
                            _invC[i] = 1e-100
                            inds_bad.append(i)
                # print(ind_st)

            inds_bad = np.asarray(inds_bad)

            invC = _invC.reshape(self.data_shape + self.channel_shape)

            # switch them after they were effectively switched above

            full_shape_rev = tuple(range(len(invC.shape)))

            basis_axes_rev = full_shape_rev[:len(self.data_shape)]
            mat_axes_rev = full_shape_rev[len(self.data_shape):]
            transpose_shape_rev = mat_axes_rev + basis_axes_rev
            self._invC = invC.transpose(transpose_shape_rev)

        # Sanitize pixels the noise model / matrix inverse cannot represent.
        # The analytic instrument-noise model diverges as f -> 0 (Sa_d ~
        # (2*pi*f)^-4, Soms_d ~ (2e-3/f)^4), so the WDM f=0 layer carries inf
        # covariance entries. ``xp.linalg.inv`` does NOT raise on inf/NaN
        # input -- it silently returns inf/NaN -- so the LinAlgError guard
        # above never fires and the non-finite inverse reaches the
        # likelihood as -inf. Physically these are infinite-noise pixels
        # carrying zero information: give them zero inverse-covariance
        # weight (and a unit determinant, i.e. no log-det contribution) so
        # the likelihood stays finite and they simply drop out. On empirical
        # (mojito NOISE-brick) PSDs every pixel is finite -> no-op.
        bad = ~xp.isfinite(self._invC)
        if bool(xp.any(bad)):
            n_bad = int(xp.count_nonzero(bad))
            self._invC = xp.where(bad, xp.zeros_like(self._invC), self._invC)
            _warn_zeroed_invc(n_bad)
        det_bad = ~xp.isfinite(self._detC)
        if bool(xp.any(det_bad)):
            self._detC = xp.where(det_bad, xp.ones_like(self._detC), self._detC)

        self._inv_det_dirty = False
            
    @property
    def invC(self) -> np.ndarray:
        """Inverse covariance Σ⁻¹.

        Computed lazily: a fresh ``sens_mat`` (from construction, ``__setitem__``,
        or an arithmetic op) just flips a dirty flag; the actual matrix inverse
        runs on the first read of ``invC`` (or ``detC``) afterwards. A chain
        ``A + B + C + …`` therefore pays the inverse cost exactly once, at the
        final access. ``do_inv_det=False`` (e.g. after slicing) suppresses the
        auto-recompute and returns whatever was last assigned.
        """
        if self._inv_det_dirty and self.do_inv_det:
            self._setup_det_and_inv()
        return self._invC

    @invC.setter
    def invC(self, value: np.ndarray) -> None:
        # Explicit assignment wins: clear the dirty flag so subsequent reads
        # return the just-assigned value rather than recomputing over it.
        self._invC = value
        self._inv_det_dirty = False

    @property
    def detC(self) -> np.ndarray:
        """Determinant det[Σ]. Lazily computed; see :attr:`invC` for semantics."""
        if self._inv_det_dirty and self.do_inv_det:
            self._setup_det_and_inv()
        return self._detC

    @detC.setter
    def detC(self, value: np.ndarray) -> None:
        self._detC = value
        self._inv_det_dirty = False

    def compute_inv_det(self) -> None:
        """Force-compute ``invC`` / ``detC`` now (rather than waiting for first read)."""
        self.do_inv_det = True
        if self._inv_det_dirty:
            self._setup_det_and_inv()

    def _combine(
        self,
        other: "SensitivityMatrixBase | np.ndarray | float",
        op: Callable,
    ) -> "SensitivityMatrixBase":
        """Combine ``self.sens_mat`` with ``other`` via ``op`` and return a new instance.

        The returned matrix shares ``basis_settings`` with ``self`` and starts
        out *dirty* — ``invC`` / ``detC`` are not computed until first read.
        """
        if isinstance(other, SensitivityMatrixBase):
            other_arr = other.sens_mat
        elif isinstance(other, (np.ndarray, cp.ndarray)):
            other_arr = other
        else:
            return NotImplemented

        if other_arr.shape != self.sens_mat.shape:
            raise ValueError(
                f"Shape mismatch combining SensitivityMatrixBase: "
                f"self.sens_mat.shape={self.sens_mat.shape} vs "
                f"other.shape={other_arr.shape}."
            )

        new = SensitivityMatrixBase(self.basis_settings)
        new.sens_mat = op(self.sens_mat, other_arr)
        return new

    def __add__(self, other):
        return self._combine(other, operator.add)

    def __sub__(self, other):
        return self._combine(other, operator.sub)

    def add(self, other) -> "SensitivityMatrixBase":
        """Object-oriented equivalent of ``self + other``. Returns a new matrix."""
        return self._combine(other, operator.add)

    def subtract(self, other) -> "SensitivityMatrixBase":
        """Object-oriented equivalent of ``self - other``. Returns a new matrix."""
        return self._combine(other, operator.sub)

    def __getitem__(self, index: Any) -> np.ndarray:
        """Indexing the class indexes the array."""
        return self.sens_mat[index]

    def __setitem__(self, index: Any, value: np.ndarray) -> np.ndarray:
        """Indexing the class indexes the array."""
        self.sens_mat[index] = value
        self._inv_det_dirty = True

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
            if not isinstance(self.basis_settings, domains.FDSettings):
                raise NotImplementedError("Needs to be frequency domain for automatic plotting.")
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
                    plot_in = np.sqrt(self.basis_settings.f_arr * plot_in)
                ax[i].loglog(self.basis_settings.f_arr, plot_in, **kwargs)

        elif fig is not None:
            raise NotImplementedError

        elif isinstance(ax, plt.axes):
            if inds is None:
                raise ValueError(
                    "When passing a single axes object for `ax`, but also pass `inds` kwarg."
                )
            plot_in = self.sens_mat[inds]
            if char_strain:
                plot_in = np.sqrt(self.basis_settings.f_arr * plot_in)
            ax.loglog(self.basis_settings.f_arr, plot_in, **kwargs)

        else:
            raise ValueError("ax must be a list of axes objects or a single axes object.")

        return (fig, ax)


class SensitivityMatrix(SensitivityMatrixBase):
    """Container to hold sensitivity information.

    Args:
        settings: Domain settings (a :class:`~lisatools.domains.DomainSettingsBase`,
            e.g. :class:`~lisatools.domains.FDSettings`) describing the basis the
            sensitivity is evaluated on (its ``f_arr`` in FD, etc.).
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
        sens_kwargs_mat=None,
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
        settings: Domain settings (e.g. :class:`~lisatools.domains.FDSettings`)
            whose ``f_arr`` sets the frequency grid.
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
        settings: Domain settings (e.g. :class:`~lisatools.domains.FDSettings`)
            whose ``f_arr`` sets the frequency grid [Hz].
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
            [X2TDISens, XY2TDISens, ZX2TDISens],
            [XY2TDISens, Y2TDISens, YZ2TDISens],
            [ZX2TDISens, YZ2TDISens, Z2TDISens],
        ]

        super().__init__(settings, sens_mat, **sens_kwargs)


class AET1SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for AET (TDI 1)

    This is just an array because no cross-terms.

    Args:
        settings: Domain settings (e.g. :class:`~lisatools.domains.FDSettings`)
            whose ``f_arr`` sets the frequency grid.
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

    def __init__(self, settings: domains.DomainSettingsBase, **sens_kwargs: dict) -> None:
        sens_mat = [A2TDISens, E2TDISens]
        super().__init__(settings, sens_mat, **sens_kwargs)


class LISASensSensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix adding :class:`LISASens` for the specified number of channels.

    Args:
        settings: Domain settings (e.g. :class:`~lisatools.domains.FDSettings`)
            whose ``f_arr`` sets the frequency grid.
        nchannels: Number of channels.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(
        self, settings: domains.DomainSettingsBase, nchannels: int, **sens_kwargs: dict
    ) -> None:
        sens_mat = [LISASens for _ in range(nchannels)]
        super().__init__(settings, sens_mat, **sens_kwargs)

def randc(shape):
    """Return complex Gaussian noise with the given ``shape`` (real + imaginary unit-variance)."""
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

def get_sensitivity(
    basis_settings: domains.DomainSettingsBase,
    *args: tuple,
    sens_fn: Optional[Sensitivity | str] = LISASens,
    return_type="PSD",
    fill_nans: float = np.nan,
    args_list: Optional[List[tuple]] = None,
    kwargs_list: Optional[List[dict]] = None,
    wdm_psd_method: str = "fold",
    stationary: bool = True,
    **kwargs,
) -> float | np.ndarray:
    """Generic sensitivity generator

    Same interface to many sensitivity curves.

    Args:
        basis_settings: Domain settings (a
            :class:`~lisatools.domains.DomainSettingsBase`, e.g.
            :class:`~lisatools.domains.FDSettings`) whose ``f_arr`` sets the
            evaluation grid. A raw frequency scalar/array is also accepted for
            backward compatibility (the sensitivity is then evaluated directly
            on those frequencies).
        *args: Any additional arguments for the sensitivity function ``get_Sn`` method.
        sens_fn: String or class that represents the name of the desired PSD function.
        return_type: Describes the desired output. Choices are ``"ASD"``,
            ``"PSD"``, or ``"char_strain"`` (characteristic strain). Default is ``"PSD"``.
        fill_nans: Value to fill nans in sensitivity (at 0 frequency).
            If ``None``, thens nans will be left in the array.
        wdm_psd_method: How to build the WDM (wavelet) noise PSD (ignored for
            non-WDM domains). ``"fold"`` (default) folds the full-resolution
            Fourier-domain PSD into the wavelet basis (matches the forward WDM
            transform; ``E[w_mn^2] == S_wdm[m]``). ``"layer_constant"`` is the
            faster approximation that treats the PSD as constant across a
            wavelet layer, ``S_wdm[m] = (1/2) Sn(f_layer_center)``.
        stationary: For WDM, whether the noise PSD is the same for every time
            pixel. When ``True`` (default) the Fourier-domain PSD is evaluated
            once and broadcast across all time pixels. When ``False``
            (time-varying noise) the stationary ``wdm_psd_method`` calculation is
            repeated per wavelet time column, each using its own Fourier-domain
            PSD supplied through ``args_list`` / ``kwargs_list`` (length ``Nt``).
        include_instrument: Forwarded to ``get_Sn`` (in ``kwargs``). ``True``
            (default) returns instrument + stochastic; ``False`` returns only the
            stochastic contribution (``model`` then unused) — folded through the
            same domain dispatch, so it works in FD, WDM, etc.
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

    # Back-compat: callers from ``gbgpu`` (and other downstreams that pre-date
    # the DomainSettings-first signature) pass a raw frequency scalar/array
    # here. Dispatch straight to ``sensitivity.get_Sn`` on those inputs so
    # neighbouring packages don't break on the new contract.
    if not isinstance(basis_settings, domains.DomainSettingsBase):
        PSD = sensitivity.get_Sn(basis_settings, *args, **kwargs)
        if fill_nans is not None:
            PSD = np.nan_to_num(PSD, nan=fill_nans)
        if return_type == "PSD":
            return PSD
        if return_type == "ASD":
            return np.sqrt(PSD)
        if return_type == "char_strain":
            return np.sqrt(np.asarray(basis_settings) * PSD)
        raise ValueError(f"return_type {return_type!r} not supported.")

    if isinstance(basis_settings, domains.FDSettings):
        PSD = sensitivity.get_Sn(basis_settings.f_arr, *args, **kwargs)

    elif isinstance(basis_settings, domains.TDSettings):
        raise NotImplementedError
    elif isinstance(basis_settings, domains.STFTSettings):
        raise NotImplementedError
        PSD = sensitivity.get_Sn(basis_settings.f_arr, *args, **kwargs)
    elif isinstance(basis_settings, domains.WDMSettings):
        if kwargs_list is None:
            kwargs_list = [kwargs for _ in range(basis_settings.Nt)]
        else:
            assert isinstance(kwargs_list, list)
            assert len(kwargs_list) == basis_settings.Nt
            for tmp in kwargs_list:
                if not isinstance(tmp, dict):
                    raise ValueError(
                        "Value in kwargs_list is not a dictionary. Must be a dictionary."
                    )

        if args_list is None:
            args_list = [args for _ in range(basis_settings.Nt)]
        else:
            assert isinstance(args_list, list)
            assert len(args_list) == basis_settings.Nt
            for tmp in args_list:
                if not isinstance(tmp, tuple) and not isinstance(tmp, list):
                    raise ValueError("Value in args_list is not a tuple. Must be a tuple.")
            
        xp = get_array_module(basis_settings.f_arr)
        # equation for stationary noise (https://arxiv.org/pdf/2009.00043; eq. 19)
        # npts = 3
        # x = np.linspace(basis_settings.f_arr_edges[:-1],  basis_settings.f_arr_edges[1:], num=npts, axis=-1)
        # integrand = xp.asarray([sensitivity.get_Sn(x, *_args, **_kwargs) for _args, _kwargs in zip(args_list, kwargs_list)]).transpose(1, 0, 2)

        # # this is to match tyson's code. I have questions
        # h = 1.0
        # f0 = integrand[:, :, 0]
        # f1 = integrand[:, :, 1]
        # f2 = integrand[:, :, 2]
        # PSD = simpson_3_integral = h*(f0 + 4.0*f1 + f2)/6.0
        # 0.25 is fudge factor from tysons code
        # f_c = np.fft.rfftfreq(basis_settings.N, basis_settings.data_dt)
        # psd = sensitivity.get_Sn(f_c, *args_list[0], **kwargs_list[0])

        # psd_fd = domains.FDSignal(psd, settings=domains.FDSettings(f_c.shape[0], f_c[1] - f_c[0]))
        # PSD = psd_fd.wdmtransform(settings=basis_settings, is_psd=True)[0]

        if wdm_psd_method not in ("fold", "layer_constant"):
            raise ValueError(
                f"wdm_psd_method must be 'fold' or 'layer_constant', got {wdm_psd_method!r}."
            )

        def _wdm_layer_psd(_args, _kwargs):
            """Per-layer wavelet PSD column (length ``Nf_active``) for a single
            Fourier-domain noise spectrum. For locally stationary noise the
            folded PSD is independent of the wavelet time pixel, so one column
            fully describes it; the non-stationary path calls this once per
            time column with that column's own spectrum."""
            if wdm_psd_method == "layer_constant":
                # approximation: PSD constant across each wavelet layer,
                # evaluated at the layer centre frequencies.
                f_c = basis_settings.f_arr
                return 1 / 2 * sensitivity.get_Sn(f_c, *_args, **_kwargs)

            # exact: fold the full-resolution Fourier-domain PSD into the
            # wavelet basis. Validated so that E[w_mn^2] == S_wdm[m] against the
            # forward WDM transform (see wdm_noise_validation.py). The fold is
            # time-column independent, so we keep a single representative column.
            f_full = xp.fft.rfftfreq(basis_settings.N, basis_settings.data_dt)
            df = float(f_full[1] - f_full[0])
            # The fold only ever reads the bins in ``fold_frequency_indices``
            # (a small fraction of the rFFT grid for a narrow active band), and
            # evaluating the noise model is far more expensive than the gather
            # itself -- so score just those bins and leave the rest zero. The
            # folded output is bit-identical to evaluating the full grid.
            idx = basis_settings.fold_frequency_indices
            psd_active = xp.asarray(sensitivity.get_Sn(f_full[idx], *_args, **_kwargs))
            psd_full = xp.zeros(f_full.shape, dtype=psd_active.dtype)
            psd_full[idx] = psd_active
            psd_fd = domains.FDSignal(
                psd_full,
                domains.FDSettings(
                    f_full.shape[0], df, force_backend=basis_settings.backend
                ),
            )
            folded = xp.real(psd_fd.wdmtransform(settings=basis_settings, is_psd=True)[0])
            return folded[:, 0]

        if stationary:
            # STATIONARY: evaluate the Fourier-domain PSD once and broadcast the
            # single folded layer column across every wavelet time pixel.
            col = _wdm_layer_psd(args_list[0], kwargs_list[0])
            PSD = xp.repeat(col[:, None], basis_settings.Nt_active, axis=-1)

        else:
            # NON-STATIONARY (time-varying): repeat the stationary fold /
            # layer_constant calculation per wavelet time column, each with its
            # own Fourier-domain PSD supplied through args_list / kwargs_list
            # (both length Nt). active_slice_t selects the active columns.
            cols = [
                _wdm_layer_psd(args_list[g], kwargs_list[g])
                for g in range(basis_settings.ind_min_t, basis_settings.ind_max_t + 1)
            ]
            PSD = xp.stack(cols, axis=-1)

    else:
        raise ValueError(
            f"Domain type entered ({type(basis_settings)}). Needs to be one of {domains.get_available_domains()}"
        )

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


# Number of epochs the FD transfer-function average is decimated to, spanning the
# full orbit. The constellation breathing is smooth on day-to-month scales (dominant
# period ~1 yr), so ~daily resolution (1024 pts over a ~2 yr orbit) reproduces the
# average over the full native LTT grid (~25M pts) to well below 0.1% -- while the
# full grid is infeasible to evaluate the transfer functions on.
_N_AVERAGE_EPOCHS = 1024


class SensitivityBackendBase(LISAToolsParallelModule):
    """Shared base for per-walker sensitivity-matrix backends.

    A *sensitivity backend* is a callable that maps a walker's noise parameters
    to a :class:`SensitivityMatrixBase`. Both the native-kernel
    :class:`XYZSensitivityBackend` and the pure-Python
    :class:`CompositeSensitivityBackend` derive from this base so they share the
    **same backend-dispatch machinery** (via
    :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`: ``.xp`` /
    ``.backend`` / ``force_backend``) and the **same** ``__call__`` contract;
    each backend supplies only its matrix construction in :meth:`_build_matrix`.

    Args:
        settings: Domain settings the matrix is evaluated on (FD, WDM, ...).
        tdi_generation: 1 (TDI 1.5) or 2 (TDI 2.0).
        force_backend: Backend selector (``"cpu"`` / a CUDA name / ``"jax"``);
            see :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`.
    """

    def __init__(
        self,
        settings: DomainSettingsBase,
        *,
        tdi_generation: int = 2,
        force_backend: Optional[str] = None,
    ):
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        self.basis_settings = settings
        self.tdi_generation = tdi_generation

    @classmethod
    def supported_backends(cls):
        """Backends this sensitivity backend can dispatch to (CPU / CUDA / JAX).

        Mirrors :class:`~lisatools.chunked_het.WDMComputationsBase`: JAX is listed
        last, so the default "first available" pick stays CPU/GPU. (JAX *compute*
        for the composite path is a deferred follow-up; the name is advertised so
        an explicit ``force_backend="jax"`` resolves once that lands.)
        """
        return [cls._BACKEND_PREFIX + "_" + t for t in cls.GPU_RECOMMENDED_WITH_JAX()]

    @property
    def xp(self):
        """Array module of the resolved backend (numpy / cupy / jax.numpy)."""
        return self.backend.xp

    def __call__(
        self,
        name: str,
        psd_params,
        galfor_params=None,
        sgwb_params=None,
        transform_fn: Optional[TransformContainer] = None,
    ) -> "SensitivityMatrixBase":
        """Build a per-walker sensitivity matrix from noise parameters.

        Applies the optional PSD ``transform_fn`` (sampling -> physical params),
        then delegates matrix construction to :meth:`_build_matrix`. The shared
        signature means a run's ``sensitivity_backend`` can be either subclass
        interchangeably (``run.py`` always passes ``transform_fn=`` /
        ``galfor_params=`` / ``sgwb_params=``).

        Args:
            name: Identifier recorded on the produced matrix.
            psd_params: ``[Soms_d, Sa_a, ...]`` (sampling basis if ``transform_fn``),
                or ``None`` for a fixed-PSD run whose instrument noise lives
                entirely in the backend's ``extra_components`` (e.g. a
                :class:`MojitoNoiseEstimates` table) — no parametric
                instrument component is built then.
            galfor_params: Optional galactic-foreground parameters.
            sgwb_params: Optional SGWB spectral-template parameters.
            transform_fn: Optional PSD :class:`TransformContainer`.
        """
        if psd_params is None:
            return self._build_matrix(name, None, galfor_params, sgwb_params)
        params = np.asarray(psd_params, dtype=float)
        if transform_fn is not None:
            params = transform_fn.both_transforms(
                params, copy=True, return_transpose=False
            )
            params = np.atleast_1d(np.asarray(params).squeeze())
        return self._build_matrix(name, params, galfor_params, sgwb_params)

    def _build_matrix(self, name, params, galfor_params, sgwb_params):
        """Construct the sensitivity matrix for the (already-transformed) params.

        Subclass hook. ``params`` is the physical-basis ``[Soms_d, Sa_a, ...]``.
        """
        raise NotImplementedError


class XYZSensitivityBackend(SensitivityBackendBase, SensitivityMatrixBase):
    """3x3 XYZ TDI sensitivity matrix backed by the C++/CUDA detector kernels.

    Wraps :class:`LISAToolsParallelModule` (for backend dispatch) and
    :class:`SensitivityMatrixBase` (for the matrix interface). The matrix is
    computed at the basis frequencies via the native ``SensitivityMatrixWrap``
    using averaged light-travel-times derived from the supplied orbits, with
    optional spline interpolation and galactic-foreground contributions.
    Currently supports TDI generations 1 and 2, but only XYZ channels.

    Args:
        orbits: Configured :class:`~lisatools.detector.L1Orbits` instance providing
            light travel times and spacecraft positions.
        settings: Domain settings (frequency or time-frequency) the matrix is
            evaluated on.
        tdi_generation: 1 for TDI 1.5 or 2 for TDI 2.0.
        use_splines: If ``True``, use Akima spline interpolation for the noise
            knots passed in :meth:`set_sensitivity_matrix`.
        spline_order: Order of the Akima interpolant (passed through to
            :class:`cudakima.AkimaInterpolant1D`).
        force_backend: Backend selector (``"cpu"`` or a CUDA name); see
            :class:`LISAToolsParallelModule`.
        mask_percentage: Fractional bandwidth around transfer-function dips that
            is masked out (defaults to ``0.05``).
        galactic_grid_kwargs: Optional dictionary of keyword arguments to set up
            the galactic grid for foreground contributions. If ``None`` or empty,
            the galactic grid is not included in the sensitivity matrix;
            otherwise the dictionary is passed to :meth:`_setup_galactic_grid`.
        window_values: Optional window applied to the time-domain data; used for
            normalising the resulting PSD (accounts for windowing-induced loss
            of power).
        average_transfer_functions: Whether to average the TDI transfer functions
            over the orbit (``True``) or use the values at a single average epoch
            (``False``), in the case of a frequency-domain basis. Default is
            ``False``.
    """

    def __init__(
        self,
        orbits: Orbits | L1Orbits,
        settings: DomainSettingsBase,
        tdi_generation: int = 2,
        use_splines: bool = False,
        spline_order: Optional[str] = "cubic",
        force_backend: Optional[str] = "cpu",
        mask_percentage: Optional[float] = None,
        galactic_grid_kwargs: Optional[dict] = None,
        window_values: Optional[NDArrayLike] = None,
        average_transfer_functions: bool = False,
    ):
        SensitivityBackendBase.__init__(
            self, settings, tdi_generation=tdi_generation, force_backend=force_backend
        )
        SensitivityMatrixBase.__init__(self, settings)

        assert self.backend.xp == orbits.xp, "Orbits and Sensitivity backend mismatch."

        self.orbits = orbits  # configures lazily on first use

        self.tdi_generation = tdi_generation
        self.channel_shape = (3, 3)

        _use_gpu = force_backend != "cpu"

        self.use_splines = use_splines
        self.spline_order = spline_order
        self.spline_interpolant = AkimaInterpolant1D(
            use_gpu=_use_gpu, threadsperblock=NUM_SPLINE_THREADS, order=spline_order
        )

        self.mask_percentage = mask_percentage if mask_percentage is not None else 0.05

        self.window_values = window_values

        self.average_transfer_functions = average_transfer_functions
        self._averaging_active = False   # set True by get_averaged_ltts() in FD averaged mode

        self._setup()
        
        self.galactic_grid_kwargs = galactic_grid_kwargs # for propagation to copies
        include_galaxy = (isinstance(galactic_grid_kwargs, dict) and len(galactic_grid_kwargs) > 0)
        
        if include_galaxy:
            self._sanitize_galactic_grid_kwargs(galactic_grid_kwargs)
            self._setup_galactic_grid(**galactic_grid_kwargs)

    @property
    def kwargs(self):
        return {
            "orbits": self.orbits,
            "settings": self.basis_settings,
            "tdi_generation": self.tdi_generation,
            "use_splines": self.use_splines,
            "spline_order": self.spline_order,
            "force_backend": self.backend.backend_name.split("_")[-1],
            "mask_percentage": self.mask_percentage,
            "galactic_grid_kwargs": self.galactic_grid_kwargs,  # propagate to copies
            "window_values": self.window_values,
            "average_transfer_functions": self.average_transfer_functions,
        }

    @property
    def xp(self):
        """Array module."""
        return self.backend.xp

    @property
    def smoothing_sigma(self):
        """Sigma for smoothing the sensitivity matrix around the zero dips."""
        return 5

    @property
    def time_indices(self):
        """Integer indices into the time axis used by the C++ backend."""
        return self._time_indices
    

    @time_indices.setter
    def time_indices(self, x):
        """Set the time-index array used by the C++ backend."""
        self._time_indices = x

    def get_averaged_ltts(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute averaged and differential light-travel times across LISA links.

        Reads orbital light-travel times at the segment centre times (STFT/WDM) or
        at the appropriate FD epoch(s), then forms per-arm averages and differences
        needed by the C++ sensitivity kernel.

        Link ordering follows ``orbits.LINKS``: [12, 23, 31, 13, 32, 21].
        Averages are taken between opposite-direction pairs (12↔21, 23↔32, 31↔13).

        Returns:
            avg_ltts: Mean light-travel times per arm. Shape ``(n_epochs, 6)``.
            delta_ltts: Signed difference (forward − backward) per arm. Shape ``(n_epochs, 6)``.

        Note:
            ``n_epochs`` (the first axis of the returned arrays — it becomes the C++
            wrap's ``n_times`` in :meth:`_setup`) is **not** always equal to
            ``len(self.time_indices)``:

            * STFT/WDM and FD non-averaging: ``n_epochs == len(self.time_indices)``
              (one per segment, or 1 for FD).
            * **FD averaging mode** (``average_transfer_functions=True``): the returned
              arrays hold ``N ≈ _N_AVERAGE_EPOCHS`` decimated orbit epochs while
              ``self.time_indices == [0]``. This mismatch is deliberate. Those N epochs
              exist **only** to feed the one-time transfer-function average in
              :meth:`_build_and_attach_averaged_tfs` (which evaluates the 12 TFs at all
              N epochs via ``get_noise_tfs_wrap`` and means them). The likelihood and
              :meth:`compute_sensitivity_matrix` instead read the precomputed averaged
              TFs — ``get_noise_covariance`` indexes ``*_avg[f_idx]`` and ignores
              ``time_index`` — so they iterate a SINGLE effective time. Once the average
              is built, the wrap's N-epoch LTT array is no longer read in averaged mode.
        """
        # first, compute the average ltts and their differences.
        # check if we need multiple time points
        if hasattr(self.basis_settings, "t_arr"):
            t_arr = self.xp.asarray(self.basis_settings.t_arr)
        
            tiled_times = self.xp.tile(
                t_arr[:, self.xp.newaxis], (1, 6)
            ).flatten()  # compute ltts at these times with orbits

            links = self.xp.tile(self.xp.asarray(self.orbits.LINKS), (t_arr.shape[0],))

            ltts = self.orbits.get_light_travel_times(tiled_times, links).reshape(len(t_arr), 6)

            self.time_indices = self.xp.arange(len(t_arr), dtype=self.xp.int32)


        else:
            if self.average_transfer_functions:
                # Average the transfer functions over the FULL orbit span. orbits.ltt_t is
                # the native LTT time grid (fine cadence, ~25M pts); the breathing is smooth
                # on day-to-month scales, so we decimate to ~daily resolution -> numerically
                # identical to the full-grid average but tractable. No user-provided
                # epoch count needed.
                t_full = self.xp.asarray(self.orbits.ltt_t)
                stride = max(1, int(len(t_full) // _N_AVERAGE_EPOCHS))
                t_arr = t_full[::stride]
                tiled_times = self.xp.tile(t_arr[:, self.xp.newaxis], (1, 6)).flatten()
                links = self.xp.tile(self.xp.asarray(self.orbits.LINKS), (t_arr.shape[0],))
                ltts = self.orbits.get_light_travel_times(tiled_times, links).reshape(len(t_arr), 6)
                # NOTE: ltts holds N decimated epochs -> the wrap is built with n_times=N
                # (see _setup), but time_indices=[0]. The N epochs feed the one-time TF
                # average only (_build_and_attach_averaged_tfs); the likelihood reads the
                # averaged TFs and iterates a single effective time. See the
                # get_averaged_ltts docstring for the full rationale.
                self.time_indices = self.xp.array([0], dtype=self.xp.int32)
                self._averaging_active = True
            else:
                # single effective epoch: the orbit-averaged LTTs, i.e. C(E[L])
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

        self._setup_window()

        self.pycppsensmat_args = [
            self.xp.asarray(avg_ltts.flatten().copy()),
            self.xp.asarray(delta_ltts.flatten().copy()),
            avg_ltts.shape[0],  # n_times (= N decimated epochs in FD averaged mode; != len(time_indices) there)
            self.orbits.armlength,
            self.tdi_generation,
            self.use_splines,
            self.window_normalization,
        ]

        # XYZBackend disabled (symbol issues on Linux): SensitivityMatrixWrap may be absent.
        _SensitivityMatrixWrap = getattr(self.backend, "SensitivityMatrixWrap", None)
        if _SensitivityMatrixWrap is None:
            self.pycpp_sensitivity_matrix = None
        else:
            self.pycpp_sensitivity_matrix = _SensitivityMatrixWrap(*self.pycppsensmat_args)

        self._init_basis_settings()

        if self._averaging_active:
            self._build_and_attach_averaged_tfs()

    def _build_and_attach_averaged_tfs(self):
        """Precompute epoch-averaged transfer functions and attach them to the
        c++ object so the in-kernel covariance assembly (and the diagnostic path)
        use E_t[C(f;L(t))].  Parameter-free -> computed once; arrays kept alive on
        self and shared across walker copies (like gal_R_avg)."""
        xp = self.xp
        nf = self.num_freqs
        # the epochs to average the transfer functions over (NOT likelihood time
        # points; the likelihood uses time_indices=[0]). See get_averaged_ltts docstring.
        N = self.pycppsensmat_args[2]
        f_arr = xp.asarray(self.f_arr)

        # order MUST match get_noise_tfs_wrap: oms_xx,xy,xz,yy,yz,zz, tm_xx,xy,xz,yy,yz,zz
        real_flags = (True, False, False, True, False, True,
                      True, False, False, True, False, True)
        acc = [xp.zeros(nf, dtype=xp.float64 if r else xp.complex128) for r in real_flags]

        # Accumulate the per-epoch transfer functions in CHUNKS: the materialised
        # (n_epochs x n_freqs) buffer would be tens of GB at production n_freqs, so we
        # bound the transient buffer to ~1 GB regardless of n_freqs / n_epochs.
        chunk = max(1, min(N, int(1e9 // (nf * 16 * 12))))
        for start in range(0, N, chunk):
            cs = int(min(chunk, N - start))
            bufs = [xp.empty(cs * nf, dtype=xp.float64 if r else xp.complex128) for r in real_flags]
            self.pycpp_sensitivity_matrix.get_noise_tfs_wrap(
                f_arr, *bufs, nf, cs, xp.arange(start, start + cs, dtype=xp.int32))
            for k in range(12):
                acc[k] += bufs[k].reshape(cs, nf).sum(axis=0)

        # epoch mean -> 12 contiguous (nf,) arrays, KEPT ALIVE on self (non-owned in c++)
        self._avg_tf_arrays = [xp.ascontiguousarray(a / N) for a in acc]
        self.pycpp_sensitivity_matrix.set_averaged_tfs_wrap(*self._avg_tf_arrays, nf)

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
            if key in ("_backend", "pycpp_sensitivity_matrix", "_galactic_grid", "_avg_tf_arrays"):
                # Don't deepcopy backend objects - just reference.
                # _avg_tf_arrays is referenced by raw pointers inside the (shared)
                # pycpp_sensitivity_matrix, so copies MUST share these arrays.
                setattr(new_obj, key, value)
            elif key == "orbits":
                # Shallow copy orbits (share the same backend)
                setattr(new_obj, key, copy(value))
            elif key == "spline_interpolant":
                # Shallow copy spline interpolant
                setattr(new_obj, key, copy(value))
            else:
                # Deepcopy everything else
                setattr(new_obj, key, deepcopy(value, memo))
        
        return new_obj

    def _setup_window(self):
        """Setup window values for the c++ backend."""
        if self.window_values is not None:
            assert isinstance(self.window_values, np.ndarray) or isinstance(self.window_values, cp.ndarray)
            assert self.window_values.ndim == 1

            self.window_values = self.xp.asarray(self.window_values)

            num_points = self.window_values.shape[0]
            self.window_normalization = float(
                self.xp.sum(self.window_values ** 2) / num_points
            )
        else:
            self.window_normalization = 1.0
                
    def _init_basis_settings(self):
        """Initialize basis settings from domain settings."""
        self.f_arr = self.xp.asarray(self.basis_settings.f_arr)

        if hasattr(self.basis_settings, "t_arr"):
            self.t_arr = self.xp.asarray(self.basis_settings.t_arr)

        self.num_times = len(self.t_arr) if hasattr(self, "t_arr") else 1
        self.num_freqs = len(self.f_arr)

        dips_indices = self._get_dips_indices()

        dips_mask = self.xp.zeros((self.num_times, self.num_freqs), dtype=bool)
        for t_idx in range(self.num_times):
            dips_mask[t_idx, dips_indices[t_idx]] = True

        self.dips_mask = dips_mask.flatten()

    def _find_dips_with_percentage(self, tf, mask_percentage=0.05):
        """Return indices of bins within ``mask_percentage`` of every transfer-function dip."""
        f_arr = asnumpy(self.f_arr)
        tf = asnumpy(tf)

        peaks = find_peaks(-tf)[0]

        all_indices = set()
        for peak in peaks:
            freq = self.f_arr[peak]
            df = self.f_arr[1] - self.f_arr[0]

            lower_freq = freq - mask_percentage * freq
            upper_freq = freq + mask_percentage * freq

            lower_idx = int(self.xp.searchsorted(self.f_arr, lower_freq - df / 2))
            upper_idx = int(self.xp.searchsorted(self.f_arr, upper_freq + df / 2))

            all_indices.update(range(lower_idx, upper_idx))

        return self.xp.array(sorted(all_indices), dtype=self.xp.int32)

    def _get_dips_indices(
        self,
    ):
        """Compute per-time-slice indices of frequency bins around transfer-function dips."""
        transfer_functions = self.compute_transfer_functions(self.f_arr)

        tf = transfer_functions[0]

        dips_indices = [
            self._find_dips_with_percentage(tf[t_idx], mask_percentage=self.mask_percentage)
            for t_idx in range(self.num_times)
        ]

        return dips_indices

    def _sanitize_galactic_grid_kwargs(self, kwargs: dict) -> None:
        """
        Check that the galactic grid kwargs are valid and contain the necessary parameters.
        
        Args:
            kwargs: Dictionary of galactic grid parameters to check. Expected keys include:
                - R_d: Disk radial scale length [kpc]
                - z_d: Disk vertical scale height [kpc]
                - t0: Reference time at which to compute the initial LISA orbital phase and rotation angle.
                - N_lambda: Number of ecliptic longitude points for quadrature (optional, default 90)
                - N_beta: Number of ecliptic latitude points for quadrature (optional, default 60)
                - galactic_grid: Optional pre-computed galactic grid object (e.g., from another instance) to reuse
        """
        required_keys = ["R_d", "z_d", "t0"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required galactic_grid_kwargs parameter: {key}")
            if not isinstance(kwargs[key], (int, float)):
                raise ValueError(f"Galactic grid parameter {key} must be a number (int or float).")

        optional_keys = ["N_lambda", "N_beta", "galactic_grid"]
        for key in optional_keys:
            if key in kwargs and key == "galactic_grid":
                # galactic_grid can be any object, so we won't check its type here
                continue
            elif key in kwargs:
                if not isinstance(kwargs[key], int):
                    raise ValueError(f"Galactic grid parameter {key} must be an integer.")

    def _setup_galactic_grid(
            self,
            R_d: float,
            z_d: float,
            t0: float,
            N_lambda: Optional[int] = 90,
            N_beta: Optional[int] = 60,
            galactic_grid: Optional[Any] = None
        ) -> None:
        """
        Compute the fixed galactic sky geometry if not provided, and attach it to the sensitivity backend.

        Called once during setup.  After this call, sensitivity_backend.pycpp_sensitivity_matrix
        has gal_R_avg wired in and will include the galactic foreground in every likelihood
        evaluation automatically, scaled by the per-walker spectral parameters passed via
        Amp_all, alpha_all, f_1_all, f_knee_all, f_2_all.

        Args:
            R_d: Disk radial scale length [kpc]
            z_d: Disk vertical scale height [kpc]
            t0: Reference time at which to compute the initial LISA orbital phase and rotation angle.
            N_lambda: Number of ecliptic longitude points for quadrature (default 90)
            N_beta: Number of ecliptic latitude points for quadrature (default 60)
            galactic_grid: Optional pre-computed galactic grid object (e.g., from another instance) to reuse
        """
        if galactic_grid is not None:
            self._galactic_grid = galactic_grid

            # logger.info("Using provided galactic grid object, skipping re-initialization.")
            
        else:
            alpha0, beta0 = self.orbits.get_constellation_angles(t0)

            # logger.debug(
            #     f"Initializing galactic grid: R_d={R_d} kpc, z_d={z_d} kpc, "
            #     f"alpha0={alpha0:.4f} rad, beta0={beta0:.4f} rad"
            # )

            # Build host-side quadrature geometry
            setup = self.backend.GalacticGridSetup()
            setup.compute(
                N_lambda=N_lambda,
                N_beta=N_beta,
            )
            
            # logger.debug(f"Galactic sky grid: N_sky={setup.N_sky}, N_quad={setup.N_quad}")

            if hasattr(self.basis_settings, "t_arr"):
                _t_arr = self.basis_settings.t_arr.copy()
            else:
                _t_arr = np.array([t0])
            #     logger.warning(
            #     f"FD domain detected — using t=t0={t0} for galactic sky average. "
            #     "This is correct only for stationary (non-cyclostationary) analyses."
            # )

            self._initialize_galactic_grid(
                times=self.xp.asarray(_t_arr),
                R_d=float(R_d),
                z_d=float(z_d),
                R_vals_quad=self.xp.asarray(setup.R_vals_quad),
                z_vals_quad=self.xp.asarray(setup.z_vals_quad),
                quad_weights=self.xp.asarray(setup.quad_weights),
                cos_beta_ecl=self.xp.asarray(setup.cos_beta_ecl),
                lam_ecl=self.xp.asarray(setup.lam_ecl),
                beta_ecl=self.xp.asarray(setup.beta_ecl),
                N_quad=setup.N_quad,
                N_sky=setup.N_sky,
                alpha0=float(alpha0),
                beta0=float(beta0),
                t0=float(t0)
            )

            # logger.info("Galactic grid initialized.")

        self.pycpp_sensitivity_matrix.set_galactic_grid(self._galactic_grid)
        # logger.info("Galactic grid attached to sensitivity backend.")

    def _initialize_galactic_grid(
        self,
        times: np.ndarray,
        R_d: float,
        z_d: float,
        R_vals_quad: np.ndarray,
        z_vals_quad: np.ndarray,
        quad_weights: np.ndarray,
        cos_beta_ecl: np.ndarray,
        lam_ecl: np.ndarray,
        beta_ecl: np.ndarray,
        N_quad: int,
        N_sky: int,
        alpha0: float,
        beta0: float,
        t0: float
    ) -> None:
        """
        Build the GalacticGridWrap, compute fixed sky weights and R_avg, and
        attach to the C++ sensitivity matrix.  Call once before inference.

        The grid is stored on self and propagated to any copies made via __call__.

        Args:
            times:        Segment centre times (N_times,)
            R_d:          Disk radial scale length [kpc]
            z_d:          Disk vertical scale height [kpc]
            R_vals_quad:  (N_quad * N_sky,) galactocentric radii
            z_vals_quad:  (N_quad * N_sky,) heights above disk
            quad_weights: (N_quad,) Gauss-Legendre weights
            cos_beta_ecl: (N_sky,) cos(beta) for solid-angle weighting
            lam_ecl:      (N_sky,) ecliptic longitudes
            beta_ecl:     (N_sky,) ecliptic latitudes
            N_quad:       Number of quadrature nodes (16)
            N_sky:        Number of sky pixels
            alpha0:       LISA orbit initial phase (rad)
            beta0:        LISA orbit inclination (rad)
            t0:           Reference time for constellation angles (s)
        """
        GalWrap = self.backend.GalacticGridWrap

        self._galactic_grid = GalWrap(
            self.xp.asarray(R_vals_quad),
            self.xp.asarray(z_vals_quad),
            self.xp.asarray(quad_weights),
            self.xp.asarray(cos_beta_ecl),
            self.xp.asarray(lam_ecl),
            self.xp.asarray(beta_ecl),
            N_quad,
            N_sky,
            alpha0,
            beta0,
            t0,
            self.num_times,
            self.num_freqs,
        )

        self._galactic_grid.initialize_wrap(
            self.xp.asarray(times),
            R_d,
            z_d,
            len(times),
        )

    def disable_galactic_grid(self) -> None:
        """Detach the galactic foreground from all subsequent likelihood evaluations.

        Clears ``self._galactic_grid`` and instructs the C++ backend to stop adding
        the galactic foreground term to the covariance matrix.  To re-enable, call
        :meth:`_setup_galactic_grid` again with the appropriate parameters.
        """
        self._galactic_grid = None
        self.pycpp_sensitivity_matrix.disable_galactic_grid()

    def _compute_matrix_elements(
        self,
        freqs,
        Soms_d_in=15e-12,
        Sa_a_in=3e-15,
        Amp=0,
        alpha=0,
        f_1=0,
        kn=0,
        f_2=0,
        knots_position_all: NDArrayLike = None,
        knots_amplitude_all: NDArrayLike = None,
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
            splines_out = self.spline_interpolant(xp.log10(freqs), knots_position_all, knots_amplitude_all)
            splines_in_isi_oms = splines_out[0]
            spline_in_testmass = splines_out[1]
        else:
            splines_in_isi_oms = xp.zeros(len(freqs), dtype=xp.float64)
            spline_in_testmass = xp.zeros(len(freqs), dtype=xp.float64)

        if self.pycpp_sensitivity_matrix is None:
            raise RuntimeError("XYZBackend disabled (symbol issues on Linux): get_noise_covariance unavailable.")
        self.pycpp_sensitivity_matrix.get_noise_covariance_wrap(
            xp.asarray(freqs),
            self.time_indices,
            float(Soms_d_in),
            float(Sa_a_in),
            float(Amp),
            float(alpha),
            float(f_1),
            float(kn),
            float(f_2),
            splines_in_isi_oms,
            spline_in_testmass,
            c00,
            c01,
            c02,
            c11,
            c12,
            c22,
            len(freqs),
            len(self.time_indices),
        )

        return c00, c11, c22, c01, c02, c12

    def _fill_matrix(self, c00, c11, c22, c01, c02, c12):
        """Fill the full 3x3 sensitivity matrix from its 6 unique elements."""
        xp = self.xp
        shape = self.basis_settings.basis_shape_active

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
            return (
                c00.flatten(),
                c11.flatten(),
                c22.flatten(),
                c01.flatten(),
                c02.flatten(),
                c12.flatten(),
            )

        return c00, c11, c22, c01, c02, c12

    def compute_sensitivity_matrix(
        self,
        freqs: NDArrayLike,
        Soms_d_in: float = 15e-12,
        Sa_a_in: float = 3e-15,
        Amp: float = 0.0,
        alpha: float = 0.0,
        f_1: float = 0.0,
        kn: float = 0.0,
        f_2: float = 0.0,
        knots_position_all: NDArrayLike = None,
        knots_amplitude_all: NDArrayLike = None,
        smooth: bool = False,
    ) -> NDArrayLike:
        """Compute the full 3×3 XYZ covariance matrix at arbitrary frequencies.

        Calls the C++ kernel to evaluate all six independent matrix elements
        (XX, YY, ZZ and the complex cross-terms XY, XZ, YZ) at ``freqs``, then
        assembles the full Hermitian matrix.  If the galactic grid has been
        initialised, the foreground contribution is added automatically via the
        stored ``gal_R_avg``.

        Unlike :meth:`set_sensitivity_matrix`, this method does **not** update the
        internal ``sens_mat`` attribute; it is for one-off evaluations (e.g.,
        diagnostics, plotting).

        Args:
            freqs: Frequency array at which to evaluate the matrix [Hz].
                   Shape ``(n_freqs,)`` or ``(n_times, n_freqs)`` depending on the domain.
            Soms_d_in: Displacement (OMS) noise amplitude ``S_oms`` [m/√Hz]. Default 15 pm/√Hz.
            Sa_a_in: Test-mass acceleration noise amplitude ``S_acc`` [m s⁻²/√Hz]. Default 3 fm s⁻²/√Hz.
            Amp: Galactic foreground spectral amplitude ``A`` [Hz⁻¹]. Pass 0 to omit.
            alpha: Galactic foreground spectral index ``α`` (dimensionless).
            f_1: Galactic foreground low-frequency roll-off scale ``f₁`` [Hz].
            kn: Galactic foreground knee frequency ``f_knee`` [Hz].
            f_2: Galactic foreground high-frequency roll-off scale ``f₂`` [Hz].
            knots_position_all: Log10-frequency positions of spline knots for
                noise residuals. Shape ``(2, n_knots)``. ``None`` if not using splines.
            knots_amplitude_all: Spline knot amplitudes for noise residuals.
                Shape ``(2, n_knots)``. ``None`` if not using splines.
            smooth: If ``True``, apply Gaussian smoothing around the TDI notches
                (zeros of the transfer function) before returning. Default ``False``.

        Returns:
            Hermitian covariance matrix ``Σ(f)``. Shape ``(3, 3, n_times, n_freqs)``
            (or ``(3, 3, n_freqs)`` for FD analyses with a single time point),
            dtype ``complex128``.
        """
        c00, c11, c22, c01, c02, c12 = self._compute_matrix_elements(
            freqs,
            Soms_d_in,
            Sa_a_in,
            Amp,
            alpha,
            f_1,
            kn,
            f_2,
            knots_position_all,
            knots_amplitude_all,
        )
        matrix = self._fill_matrix(c00, c11, c22, c01, c02, c12)
        
        if smooth:
            matrix = self.smooth_sensitivity_matrix(matrix, sigma=self.smoothing_sigma)

        return matrix

    def set_sensitivity_matrix(
        self,
        Soms_d_in: float = 15e-12,
        Sa_a_in: float = 3e-15,
        knots_position_all: NDArrayLike = None,
        knots_amplitude_all: NDArrayLike = None,
        Amp: float = 0.0,
        alpha: float = 0.0,
        f_1: float = 0.0,
        kn: float = 0.0,
        f_2: float = 0.0,
    ) -> None:
        """Evaluate and store the covariance matrix at the domain's basis frequencies.

        Computes the 3×3 XYZ covariance matrix at ``self.f_arr`` via the C++ kernel,
        applies Gaussian smoothing around the TDI transfer-function notches, and
        stores the result in ``self.sens_mat``.  The inverse and log-determinant
        (``self.invC``, ``self.detC``) are recomputed immediately via
        :meth:`_setup_det_and_inv`.

        This is the method called by :meth:`__call__` to update a per-walker copy of
        the backend with new PSD/foreground parameters at each MCMC step.

        Args:
            Soms_d_in: Displacement (OMS) noise amplitude ``S_oms`` [m/√Hz]. Default 15 pm/√Hz.
            Sa_a_in: Test-mass acceleration noise amplitude ``S_acc`` [m s⁻²/√Hz]. Default 3 fm s⁻²/√Hz.
            knots_position_all: Log10-frequency positions of spline knots for noise
                residuals. Shape ``(2, n_knots)``. ``None`` if not using splines.
            knots_amplitude_all: Spline knot amplitudes for noise residuals.
                Shape ``(2, n_knots)``. ``None`` if not using splines.
            Amp: Galactic foreground spectral amplitude ``A`` [Hz⁻¹]. Pass 0 to omit.
            alpha: Galactic foreground spectral index ``α`` (dimensionless).
            f_1: Galactic foreground low-frequency roll-off scale ``f₁`` [Hz].
            kn: Galactic foreground knee frequency ``f_knee`` [Hz].
            f_2: Galactic foreground high-frequency roll-off scale ``f₂`` [Hz].
        """

        c00, c11, c22, c01, c02, c12 = self._compute_matrix_elements(
            self.f_arr,
            Soms_d_in,
            Sa_a_in,
            Amp,
            alpha,
            f_1,
            kn,
            f_2,
            knots_position_all,
            knots_amplitude_all,
        )

        sens_mat = self._fill_matrix(c00, c11, c22, c01, c02, c12)

        self.sens_mat = self.smooth_sensitivity_matrix(sens_mat, sigma=self.smoothing_sigma)

    def _setup_det_and_inv(self):
        """use the c++ backend to compute the log-determinant and inverse of the sensitivity matrix."""
        c00, c11, c22, c01, c02, c12 = self._extract_matrix_elements(self.sens_mat, flatten=True)
        invC, detC = self._inverse_det_wrapper(c00, c11, c22, c01, c02, c12)

        # Sanitize pixels the noise model / 3x3 inverse cannot represent.
        # The analytic instrument-noise model diverges as f -> 0 (Sa_d ~
        # (2*pi*f)^-4, Soms_d ~ (2e-3/f)^4), so the WDM f=0 layer inverts to
        # inf/NaN; a singular XYZ covariance does the same. Physically these
        # are infinite-noise pixels carrying zero information, so give them
        # zero inverse-covariance weight and a unit determinant (zero
        # log-det contribution) -- the likelihood stays finite and those
        # pixels simply don't contribute. On empirical (mojito NOISE-brick)
        # PSDs every pixel is finite, so this is a no-op there.
        xp = self.xp
        bad = ~xp.isfinite(invC)
        if bool(xp.any(bad)):
            n_bad = int(xp.count_nonzero(bad))
            invC = xp.where(bad, xp.zeros_like(invC), invC)
            _warn_zeroed_invc(n_bad)
        det_bad = ~xp.isfinite(detC) | (detC <= 0)
        if bool(xp.any(det_bad)):
            detC = xp.where(det_bad, xp.ones_like(detC), detC)

        self.invC, self.detC = invC, detC

    def _inverse_det_wrapper(
        self,
        c00: NDArrayLike,
        c11: NDArrayLike,
        c22: NDArrayLike,
        c01: NDArrayLike,
        c02: NDArrayLike,
        c12: NDArrayLike,
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

        if self.pycpp_sensitivity_matrix is None:
            raise RuntimeError("XYZBackend disabled (symbol issues on Linux): get_inverse_det unavailable.")
        self.pycpp_sensitivity_matrix.get_inverse_det_wrap(
            c00, c01, c02, c11, c12, c22, i00, i01, i02, i11, i12, i22, det, total_terms
        )

        inverse_matrix = self._fill_matrix(i00, i11, i22, i01, i02, i12)

        return inverse_matrix, det.reshape(self.basis_settings.basis_shape_active)

    def compute_inverse_det(self, matrix_in: NDArrayLike) -> tuple:
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

    def compute_transfer_functions(self, freqs: NDArrayLike) -> tuple[NDArrayLike]:
        """Compute the OMS and test-mass noise transfer functions at arbitrary frequencies.

        Evaluates the 12 independent transfer-function elements (6 OMS + 6 TM,
        covering XX, XY, XZ, YY, YZ, ZZ for each) at ``freqs`` using the C++
        kernel.  These are the pure geometric transfer functions without any noise
        amplitude applied; multiply by ``Soms_d`` / ``Sa_a`` to get the noise PSD
        contribution.  The results are also used internally to locate the TDI notches
        (zeros) that should be masked during smoothing.

        Args:
            freqs: Frequency array [Hz]. Shape ``(n_freqs,)``.

        Returns:
            12-tuple of arrays, each of shape ``(n_times, n_freqs)``:
            ``(oms_xx, oms_xy, oms_xz, oms_yy, oms_yz, oms_zz,
               tm_xx,  tm_xy,  tm_xz,  tm_yy,  tm_yz,  tm_zz)``.
            Diagonal elements (``*_xx``, ``*_yy``, ``*_zz``) are real-valued;
            off-diagonal elements are complex.
        """

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

        if self.pycpp_sensitivity_matrix is None:
            raise RuntimeError("XYZBackend disabled (symbol issues on Linux): get_noise_tfs unavailable.")
        self.pycpp_sensitivity_matrix.get_noise_tfs_wrap(
            xp.asarray(freqs),
            oms_xx,
            oms_xy,
            oms_xz,
            oms_yy,
            oms_yz,
            oms_zz,
            tm_xx,
            tm_xy,
            tm_xz,
            tm_yy,
            tm_yz,
            tm_zz,
            num_freqs,
            self.num_times,
            self._time_indices,
        )

        return (
            oms_xx.reshape(self.num_times, num_freqs),
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
            tm_zz.reshape(self.num_times, num_freqs),
        )

    def compute_log_like(
        self,
        data_in_all: NDArrayLike,
        data_index_all: NDArrayLike,
        Soms_in_all: NDArrayLike,
        Sa_in_all: NDArrayLike,
        Amp_in_all: NDArrayLike,
        alpha_in_all: NDArrayLike,
        f_1_in_all: NDArrayLike,
        kn_in_all: NDArrayLike,
        f_2_in_all: NDArrayLike,
        knots_position_all: NDArrayLike = None,
        knots_amplitude_all: NDArrayLike = None,
        run_async: bool = False,
    ) -> NDArrayLike:
        """
        Compute log-likelihood using the c++ backend.

        Args:
            data_in_all: Input data array. Shape (num_psds, num_freqs * num_times)
            data_index_all: Data indices array to keep track of which data corresponds to which PSD. Shape (num_psds)
            Soms_in_all: Displacement noise levels for each walker. Shape (num_psds)
            Sa_in_all: Acceleration noise levels for each walker. Shape (num_psds)
            Amp_in_all: Galactic foreground amplitude for each walker. Shape (num_psds)
            alpha_in_all: Galactic foreground alpha for each walker. Shape (num_psds)
            f_1_in_all: First galactic foreground scale-frequency parameter for each walker. Shape (num_psds)
            kn_in_all: Galactic foreground knee frequency parameter for each walker. Shape (num_psds)
            f_2_in_all: Second galactic foreground scale-frequency parameter for each walker. Shape (num_psds)
            knots_position_all: Positions of spline knots for noise modeling. Shape (2 * num_psds, num_knots)
            knots_amplitude_all: Amplitudes of spline knots for noise modeling. Shape (2 * num_psds, num_knots)
            run_async: Whether to run the CUDA computation asynchronously. Default is False.

        Returns:
            log_like_out: Computed log-likelihoods for each PSD. Shape (num_psds,)
        """

        xp = self.xp

        # sanitize input
        Soms_in_all = xp.atleast_1d(Soms_in_all)
        Sa_in_all = xp.atleast_1d(Sa_in_all)

        Amp_in_all = xp.atleast_1d(Amp_in_all)
        alpha_in_all = xp.atleast_1d(alpha_in_all)
        f_1_in_all = xp.atleast_1d(f_1_in_all)
        kn_in_all = xp.atleast_1d(kn_in_all)
        f_2_in_all = xp.atleast_1d(f_2_in_all)
        
        # same for splines?

        num_psds = len(Soms_in_all)

        log_like_out = xp.zeros(shape=(num_psds,), dtype=xp.float64)

        if self.use_splines:
            splines_weights = self.spline_interpolant(
                xp.log10(self.f_arr), knots_position_all, knots_amplitude_all
            )
            splines_weights_isi_oms = splines_weights[0].flatten()
            splines_weights_testmass = splines_weights[1].flatten()
            # splines_weights_isi_oms = splines_weights[:num_psds].flatten()
            # splines_weights_testmass = splines_weights[num_psds:].flatten()

        else:
            splines_weights_isi_oms = xp.zeros(shape=(num_psds * self.num_freqs))
            splines_weights_testmass = xp.zeros(shape=(num_psds * self.num_freqs))

        if self.pycpp_sensitivity_matrix is None:
            raise RuntimeError("XYZBackend disabled (symbol issues on Linux): psd_likelihood unavailable.")
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
            xp.asarray(f_1_in_all),
            xp.asarray(kn_in_all),
            xp.asarray(f_2_in_all),
            xp.asarray(splines_weights_isi_oms),
            xp.asarray(splines_weights_testmass),
            self.basis_settings.differential_component,
            self.num_freqs,
            self.num_times,
            self.dips_mask,
            num_psds,
            run_async
        )

        return log_like_out

    def smooth_sensitivity_matrix(
        self,
        matrix_in: NDArrayLike,
        sigma: float = 5.0,
    ) -> NDArrayLike:
        """Smooth the sensitivity matrix around TDI transfer-function notches.

        The TDI transfer functions have sharp zeros at multiples of ``f = c/(2L)``
        (~0.1 Hz for LISA).  Near these notches the covariance matrix becomes
        nearly singular, causing numerical issues in the inversion.  This method
        replaces the matrix values at the masked notch frequencies with values
        from a Gaussian-smoothed version of the matrix, leaving all other
        frequencies untouched.

        Smoothing is applied along the last axis (frequency) with
        ``scipy.ndimage.gaussian_filter1d`` (CPU) or its CuPy equivalent (GPU).

        Args:
            matrix_in: Input sensitivity matrix. Trailing axes match
                ``basis_shape_active`` — ``(num_freqs,)`` for FD,
                ``(num_freqs, num_times)`` for WDM. The array is not modified
                in-place.
            sigma: Width of the Gaussian smoothing kernel in frequency bins.
                Default 5.

        Returns:
            Smoothed sensitivity matrix, same shape and dtype as ``matrix_in``.
        """
        filter_func = np_gaussian_filter1d if self.xp == np else cp_gaussian_filter1d

        smoothed_matrix = matrix_in.copy()
        # ``dips_mask`` is stored flat as ``(num_times, num_freqs)`` rows;
        # the sensitivity matrix's trailing axes are
        # ``basis_shape_active`` which is ``(num_freqs,)`` for FD and
        # ``(num_freqs, num_times)`` for WDM. Reshape and transpose so the
        # mask matches the matrix layout.
        mask = self.dips_mask.reshape(self.num_times, self.num_freqs)
        if self.num_times == 1:
            # FD: squeeze the time axis; mask is (num_freqs,).
            mask = mask[0]
        else:
            # WDM: swap to (num_freqs, num_times) to match basis_shape_active.
            mask = mask.T

        _smoothed = filter_func(matrix_in, sigma=sigma, axis=-1)

        smoothed_matrix[..., mask] = _smoothed[..., mask]

        return smoothed_matrix

    def build_spline_arrays(self, spline_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack interleaved spline parameters into separate position and amplitude arrays.

        The MCMC state stores spline parameters in an interleaved layout::

            [amp₀, pos₀, amp₁, pos₁, ...,   # OMS knots
             amp₀, pos₀, amp₁, pos₁, ...]   # TM knots

        This method splits that flat vector into two ``(2, n_walkers, n_knots)``
        arrays ready for the C++ Akima spline kernel.

        Args:
            spline_params: Interleaved spline parameters for one or more walkers.
                Shape ``(n_walkers, 2 * n_knots_oms + 2 * n_knots_tm)`` or
                ``(2 * n_knots_oms + 2 * n_knots_tm,)`` for a single walker.

        Returns:
            spline_knots_position: Log10-frequency positions of knots.
                Shape ``(2, n_walkers, n_knots)`` — axis 0 indexes [OMS, TM].
            spline_knots_amplitude: Knot amplitudes (multiplicative noise residuals).
                Shape ``(2, n_walkers, n_knots)`` — axis 0 indexes [OMS, TM].
        """
        
        spline_params = self.xp.atleast_2d(spline_params)

        spline_knots_position = spline_params[:, 1::2]
        spline_knots_amplitude = spline_params[:, 0:-1:2]
        half = spline_knots_position.shape[1] // 2
        spline_knots_amplitude = self.xp.stack((spline_knots_amplitude[:, :half], spline_knots_amplitude[:, half:]))
        spline_knots_position = self.xp.stack((spline_knots_position[:, :half], spline_knots_position[:, half:]))

        #todo should we sort the knots

        return spline_knots_position, spline_knots_amplitude

    def _build_matrix(
        self, name: str, params: np.ndarray, galfor_params=None, sgwb_params=None
    ) -> "XYZSensitivityBackend":
        """Create a configured copy of this backend with updated noise parameters.

        Backend hook (see :meth:`SensitivityBackendBase.__call__`). ``params`` is
        the physical-basis noise vector; ``sgwb_params`` is accepted for signature
        parity but ignored (the native XYZ kernel has no SGWB term).

        Used by :class:`~lisatools.globalfit.moves.psdmove.PSDMove` to produce a
        per-walker sensitivity matrix at each MCMC step without re-initialising
        expensive objects (orbits, galactic grid, spline interpolant).

        The returned object shares the same C++ kernel and galactic grid as the
        parent but has its ``sens_mat``, ``invC``, and ``detC`` attributes set
        to the values implied by the supplied parameters.

        Args:
            name: Identifier label attached to the returned copy (``new_sens_mat.name``).
            psd_params: Noise parameters for this walker.

                - Without splines: ``[Soms_d, Sa_a]`` — shape ``(2,)``.
                - With splines: ``[Soms_d, Sa_a, amp₀, pos₀, amp₁, pos₁, ...]``
                  where the remaining elements are interleaved OMS + TM knot
                  amplitudes and positions; see :meth:`build_spline_arrays`.

            galfor_params: Galactic foreground parameters ``[Amp, alpha, f_1, kn, f_2]``
                in physical (not log) units.  If ``None``, the foreground contribution
                is zeroed out.

        Returns:
            A new :class:`XYZSensitivityBackend` instance with the sensitivity matrix,
            its inverse, and log-determinant set to reflect the supplied parameters.

        Notes:
            The copy is a **shallow** :func:`copy.copy`, not a re-construction.  All
            walker-independent state — orbits, spline interpolant, the C++ kernel
            (``pycpp_sensitivity_matrix``), the galactic grid, ``f_arr``, ``dips_mask``,
            window normalisation, basis settings — is shared by reference with the
            parent.  Only the per-walker arrays differ, and they are *rebound* (not
            mutated in place): :meth:`set_sensitivity_matrix` assigns a fresh
            ``sens_mat`` whose setter recomputes ``invC``/``detC`` into new arrays, so
            the parent's arrays are never touched.  This avoids re-allocating the C++
            kernel / interpolant and recomputing the (walker-independent) light-travel
            times and transfer-function dip mask on every MCMC step.
        """
        from copy import copy

        # Shallow copy: shares the kernel, interpolant, orbits, galactic grid and
        # all immutable basis arrays with the parent. set_sensitivity_matrix below
        # rebinds sens_mat/invC/detC, so the shared template is never mutated.
        new_sens_mat = copy(self)
        new_sens_mat.name = name

        Soms_d = params[0]
        Sa_a = params[1]
        if self.use_splines:  # assume transformed input.
            spline_knots_position, spline_knots_amplitude = self.build_spline_arrays(params[2:])
        else:
            spline_knots_position = None
            spline_knots_amplitude = None

        if galfor_params is None:
            galfor_params = self.xp.zeros(5)

        new_sens_mat.set_sensitivity_matrix(
            Soms_d, Sa_a, spline_knots_position, spline_knots_amplitude, *galfor_params
        )

        return new_sens_mat


# =============================================================================
# Composite (additive, optionally time-modulated) sensitivity matrices.
# =============================================================================
#
# The total noise covariance is built as a sum of independent components. Each
# component contributes a covariance in the domain's basis; a component may also
# carry a per-element time modulation, so that
#
#     C_{ij}[basis] = sum_c ( base_{c,ij}[basis] ) * M_{c,ij}[time]
#
# where ``M_c`` is ``1`` for stationary components. This mirrors the GLASS noise
# model (``generate_full_dynamic_covariance_matrix``): a stationary instrument
# term, a time-modulated galactic foreground, and a stationary SGWB summed into
# one covariance.
#
# **Domain-agnostic by design.** The spectral part of every component goes
# through :func:`get_sensitivity`, which dispatches on the domain settings
# (FD -> ``Sn(f_arr)``, WDM -> folded wavelet PSD, ...), so the same component
# classes work in any domain ``get_sensitivity`` supports. A *constant*
# modulation works in every domain; a *time-varying* modulation requires the
# domain to have a time axis (WDM, STFT, TD) -- see :func:`_basis_time_axis`.
#
# The assembled ``(nch, nch, *basis_shape_active)`` array is handed to
# :class:`SensitivityMatrixBase`, which computes ``detC`` / ``invC`` -- so
# :func:`~lisatools.diagnostic.inner_product` and
# :func:`~lisatools.diagnostic.noise_likelihood_term` consume the result with no
# changes.

# Upper-triangle covariance elements, in the order used throughout this section.
ELEMENTS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
ELEMENT_NAMES = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]

# TDI XYZ element sensitivity classes per generation, matching ELEMENTS order.
_XYZ_ELEMENT_SENS = {
    1: [X1TDISens, Y1TDISens, Z1TDISens, XY1TDISens, ZX1TDISens, YZ1TDISens],
    2: [X2TDISens, Y2TDISens, Z2TDISens, XY2TDISens, ZX2TDISens, YZ2TDISens],
}


def _basis_time_axis(settings: domains.DomainSettingsBase) -> Optional[int]:
    """Index of the time axis within ``basis_shape_active``, or ``None``.

    Domains without a time axis (e.g. frequency-domain) return ``None``; only a
    constant modulation is meaningful for those.
    """
    if isinstance(settings, domains.WDMSettings):
        return 1  # basis_shape_active = (Nf, Nt)
    if isinstance(settings, domains.STFTSettings):
        return 0  # basis_shape_active = (NT, NF)
    if isinstance(settings, domains.TDSettings):
        return 0  # basis_shape_active = (N,)
    return None  # FDSettings and anything else with no time axis


def modulation_from_elements(
    elements: dict, nchannels: int = 3
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a ``(nch, nch, Ntime)`` modulation callable from six per-element entries.

    Args:
        elements: Mapping from element name (``"XX"``, ``"YY"``, ``"ZZ"``,
            ``"XY"``, ``"XZ"``, ``"YZ"``) to a callable ``t_arr -> (Ntime,)`` or a
            precomputed length-``Ntime`` array (or a scalar).
        nchannels: Number of channels (3 for XYZ).

    Returns:
        A callable ``t_arr -> (nch, nch, Ntime)`` filling the symmetric matrix.
    """

    def _mod(t_arr):
        xp = get_array_module(t_arr)
        nt = t_arr.shape[0]
        M = xp.zeros((nchannels, nchannels, nt))
        for name, (i, j) in zip(ELEMENT_NAMES, ELEMENTS):
            val = elements[name]
            arr = val(t_arr) if callable(val) else xp.asarray(val)
            M[i, j] = arr
            M[j, i] = arr
        return M

    return _mod


class GalForTimeModulation:
    """Per-element galactic-foreground time modulation loaded from a table file.

    This is the general, data-driven time-modulation provider for
    :class:`GalacticForeground` (``modulation=GalForTimeModulation(path)``): a
    picklable callable ``t_arr -> (3, 3, Ntime)``. The table columns are
    ``t, XX, YY, ZZ, XY, XZ, YZ`` (e.g. a GLASS anisotropy fit tuned to a given
    galaxy orientation); the symmetric per-element covariance modulation is
    interpolated onto the requested time grid.

    Prefer this over the analytic annual model (:func:`annual_modulation_matrix`
    / :class:`AnnualCovarianceModulation`) when a measured/tabulated modulation
    is available. The file is (re)loaded lazily on each call, so the object holds
    only its path and pickles cleanly across MPI ranks.

    Args:
        path: Path to the whitespace-delimited modulation table.
    """

    def __init__(self, path: str, t0: float = 0.0):
        self.path = str(path)
        # Subtracted from the table's time column, i.e. the absolute epoch the
        # table is written against. Tables tabulated on an ABSOLUTE mission
        # clock need this, because the domains hand out a 0-based ``t_arr``
        # (``WDMSettings.t_arr = t0 + arange(NT)*dt``, and the noise runs leave
        # that t0 at 0). Pass the first sample time of the data. Default 0.0
        # keeps a table already written relative to the data start unchanged.
        self.t0 = float(t0)

    def _table(self):
        """``(199, 7)``-style array: time column first, then XX YY ZZ XY XZ YZ.

        Accepts the transposed layout (7 rows x Ntime columns) as well, because
        ``np.savetxt`` of a ``(7, N)`` stack writes it that way and the two are
        trivially distinguishable. Getting this wrong used to be SILENT: for a
        (7, N) file ``glass[:, 0]`` is the first time-sample's column
        ``[t, XX, YY, ZZ, XY, XZ, YZ]``, which is non-monotonic but usually
        still brackets the requested times, so ``interp1d`` interpolated
        nonsense instead of raising.
        """
        g = np.loadtxt(self.path)
        if g.ndim != 2:
            raise ValueError(
                f"{self.path}: expected a 2-D modulation table, got shape {g.shape}."
            )
        if g.shape[1] == 7 and g.shape[0] != 7:
            pass                      # (Ntime, 7): the documented layout
        elif g.shape[0] == 7 and g.shape[1] != 7:
            g = g.T                   # (7, Ntime): transposed, accept it
        elif g.shape == (7, 7):
            raise ValueError(
                f"{self.path}: 7x7 table is ambiguous (cannot tell columns from "
                "rows). Write it as (Ntime, 7) with Ntime != 7."
            )
        else:
            raise ValueError(
                f"{self.path}: modulation table must have 7 columns "
                f"(t, XX, YY, ZZ, XY, XZ, YZ) or be its transpose; got {g.shape}."
            )
        t = g[:, 0]
        if not np.all(np.diff(t) > 0):
            raise ValueError(
                f"{self.path}: the time column is not strictly increasing after "
                "layout detection -- the table is malformed or the columns are "
                "not in the order t, XX, YY, ZZ, XY, XZ, YZ."
            )
        return g

    def __call__(self, t_arr):
        glass = self._table()
        t_tab = glass[:, 0] - self.t0
        mod = np.array(
            [
                [glass[:, 1], glass[:, 4], glass[:, 5]],
                [glass[:, 4], glass[:, 2], glass[:, 6]],
                [glass[:, 5], glass[:, 6], glass[:, 3]],
            ]
        )
        t_req = np.asarray(asnumpy(t_arr))
        if t_req.min() < t_tab.min() or t_req.max() > t_tab.max():
            raise ValueError(
                f"{self.path}: requested times [{t_req.min():.6g}, "
                f"{t_req.max():.6g}] fall outside the table's coverage "
                f"[{t_tab.min():.6g}, {t_tab.max():.6g}] (after subtracting "
                f"t0={self.t0:.6g}). A table written on an absolute mission "
                "clock needs t0 set to the data's first sample time."
            )
        return interpolate.interp1d(t_tab, mod)(t_req)


class NoiseComponent:
    """Base class for an additive contribution to the noise covariance.

    Subclasses return the full ``(nch, nch, *basis_shape_active)`` contribution
    for the given domain settings. Use :class:`SeparableComponent` for the common
    factorised case (a base covariance times a per-element modulation).
    """

    name: str = "component"
    nchannels: int = 3

    def covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        raise NotImplementedError


class SeparableComponent(NoiseComponent):
    """A component whose covariance factorises as ``base_ij[basis] * M_ij[time]``.

    Subclasses provide :meth:`base_covariance` (the stationary covariance in the
    domain basis) and may override :meth:`time_modulation` (per-element factor;
    defaults to ``None`` = stationary). The modulation may be:

    * ``None`` — stationary (the base covariance is returned unchanged);
    * a ``(nch, nch)`` constant matrix — applied in any domain;
    * a ``(nch, nch, Ntime)`` array — requires a time axis; broadcast along it.
    """

    def base_covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        """Stationary covariance in the domain basis: ``(nch, nch, *basis_shape_active)``."""
        raise NotImplementedError

    def time_modulation(self, settings: domains.DomainSettingsBase):
        """Per-element modulation: ``None``, ``(nch,nch)``, or ``(nch,nch,Ntime)``."""
        return None

    def covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        base = self.base_covariance(settings)  # (nch, nch, *basis)
        mod = self.time_modulation(settings)
        if mod is None:
            return base

        xp = settings.xp
        mod = xp.asarray(mod)
        nbasis = len(settings.basis_shape_active)

        if mod.ndim == 2:
            # constant matrix: broadcast over all basis axes (works in any domain)
            idx = (slice(None), slice(None)) + (None,) * nbasis
            return base * mod[idx]

        if mod.ndim == 3:
            time_axis = _basis_time_axis(settings)
            if time_axis is None:
                raise ValueError(
                    f"{type(settings).__name__} has no time axis; a time-varying "
                    "modulation (nch, nch, Ntime) is not allowed — use a constant "
                    "(nch, nch) modulation instead."
                )
            ntime = mod.shape[2]
            if ntime != settings.basis_shape_active[time_axis]:
                raise ValueError(
                    f"modulation time length {ntime} != basis time length "
                    f"{settings.basis_shape_active[time_axis]}."
                )
            # place Ntime on the (channel-offset) time axis, size-1 elsewhere
            shape = list(base.shape[:2]) + [1] * nbasis
            shape[2 + time_axis] = ntime
            return base * mod.reshape(shape)

        raise ValueError("modulation must be 2D (nch,nch) or 3D (nch,nch,Ntime).")


class InstrumentNoise(SeparableComponent):
    """Stationary TDI instrument-noise covariance (no time modulation).

    Args:
        tdi_generation: 1 (TDI 1.5) or 2 (TDI 2.0).
        model: LISA noise model (name or :class:`~lisatools.detector.LISAModel`).
        fill_nans: Passed to :func:`get_sensitivity` (default ``np.nan``, matching
            the stock matrices, leaves the ``f=0`` bin non-finite).
        basis_cache: Optional caller-owned dict enabling the two-basis fast
            path (see :meth:`base_covariance`). ``None`` (default) computes the
            covariance from scratch on every call, exactly as before. Pass a
            dict that outlives the component -- typically owned by the object
            that rebuilds the matrix per MCMC proposal, e.g.
            :class:`CompositeSensitivityBackend` -- to reuse the bases.
    """

    name = "instrument"

    def __init__(
        self,
        tdi_generation: int = 2,
        model="sangria",
        fill_nans: float = np.nan,
        basis_cache: Optional[dict] = None,
    ):
        if tdi_generation not in _XYZ_ELEMENT_SENS:
            raise ValueError(f"tdi_generation must be 1 or 2, got {tdi_generation!r}.")
        self.tdi_generation = tdi_generation
        self.model = model
        self.fill_nans = fill_nans
        self.element_sens_fns = _XYZ_ELEMENT_SENS[tdi_generation]
        self.basis_cache = basis_cache

    def _linear_in_noise_levels(self) -> bool:
        """Whether this component's covariance is linear in ``(Soms_d, Sa_a)``.

        True for the stock analytic model: :meth:`LISAModel.lisanoises
        <lisatools.detector.LISAModel.lisanoises>` scales ``Sop`` / ``Spm`` by
        ``Soms_d`` / ``Sa_a`` times a parameter-free shape function, and every
        stock TDI ``transform`` is a sum of ``transfer(f) * noise_level``. A
        model that overrides ``lisanoises``, a tabulated ``Sn_spl`` model, or a
        sensitivity class that declares ``linear_in_noise_levels = False``
        breaks that and takes the direct path.
        """
        model = self.model
        if isinstance(model, str) or getattr(model, "Sn_spl", None) is not None:
            return False
        if type(model).lisanoises is not lisa_models.LISAModel.lisanoises:
            return False
        return all(
            getattr(fn, "linear_in_noise_levels", False) for fn in self.element_sens_fns
        )

    def _direct_base_covariance(self, settings, model) -> np.ndarray:
        xp = settings.xp
        nch = self.nchannels
        elems = [
            get_sensitivity(settings, sens_fn=fn, model=model, fill_nans=self.fill_nans)
            for fn in self.element_sens_fns
        ]
        C = xp.zeros((nch, nch) + tuple(settings.basis_shape_active), dtype=elems[0].dtype)
        for (i, j), arr in zip(ELEMENTS, elems):
            C[i, j] = arr
            C[j, i] = arr
        return C

    def _basis_cache_key_extra(self) -> tuple:
        """Extra hashable state distinguishing this component's cached bases.

        The default component is fully described by the fields already in
        :meth:`_bases`' key. A subclass whose covariance depends on more than
        the noise levels (e.g. :class:`UnequalArmInstrumentNoise`, whose bases
        depend on the light-travel times) MUST extend the key here — the cache
        is shared across every component built from one backend, so two
        components differing only in that state would otherwise collide and
        silently serve each other's bases.
        """
        return ()

    def _bases(self, settings: domains.DomainSettingsBase):
        """The two unit-level covariances ``(C|Soms_d=1,Sa_a=0, C|Soms_d=0,Sa_a=1)``."""
        key = (
            id(settings),
            self.tdi_generation,
            self.fill_nans,
            tuple(self.element_sens_fns),
            type(self.model),
            self._basis_cache_key_extra(),
        )
        # ``id(settings)`` can be recycled after a settings object is freed, so
        # pin the object in the entry and confirm identity on lookup.
        hit = self.basis_cache.get(key)
        if hit is not None and hit[0] is settings:
            return hit[1], hit[2]

        orbits = getattr(self.model, "orbits", None) or lisa_models.DefaultOrbits()
        bases = tuple(
            self._direct_base_covariance(
                settings,
                lisa_models.LISAModel(soms_d, sa_a, orbits, "instrument_basis"),
            )
            for soms_d, sa_a in ((1.0, 0.0), (0.0, 1.0))
        )
        self.basis_cache[key] = (settings, bases[0], bases[1])
        return bases

    def base_covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        """Dense ``(nch, nch, *basis_shape_active)`` instrument covariance.

        Always returns the full dense matrix. When a ``basis_cache`` was
        supplied and the model is the stock analytic one, the result is
        assembled as ``Soms_d * B_oms + Sa_a * B_acc`` from two cached
        unit-level covariances instead of re-evaluating the noise model and
        the TDI transfer functions over the whole frequency grid. That is an
        algebraic identity, not an approximation -- it reproduces the direct
        path to machine precision (~1e-16 relative), and it is what makes a
        noise MCMC (which rebuilds this per proposal, per walker, per
        temperature) affordable on CPU.
        """
        if self.basis_cache is None or not self._linear_in_noise_levels():
            return self._direct_base_covariance(settings, self.model)

        B_oms, B_acc = self._bases(settings)
        # ``LISAModel`` stores the SQUARED levels, and the covariance is linear
        # in exactly those stored values.
        return float(self.model.Soms_d) * B_oms + float(self.model.Sa_a) * B_acc


# ---------------------------------------------------------------------------
# Unequal-arm (orbit-informed) instrument noise
# ---------------------------------------------------------------------------
#
# The stock ``X2TDISens`` family above assumes a single, constant armlength
# ``L_SI`` -- every TDI transfer function is built from ``x = 2 pi f L/c``. The
# real constellation breathes (the three arms differ by ~1%) and rotates (the
# Sagnac splitting makes ``d_ij != d_ji``), which shifts the transfer-function
# nulls per arm and gives the cross-spectra a non-zero imaginary part. Fitting
# an equal-arm model to unequal-arm data leaves exactly that structure in the
# residual.
#
# The closed forms in ``_unequal_arm_expressions`` carry all six link delays
# independently, so they capture both effects. They reduce to the stock
# equal-arm elements exactly when all six delays coincide.


#: Link ordering for every ``ltts`` array on the unequal-arm path. Matches
#: :data:`lisatools.detector.LINKS`, so an array built by evaluating
#: :meth:`~lisatools.detector.Orbits.get_light_travel_times` over ``LINKS``
#: needs no reordering.
UNEQUAL_ARM_LINKS = [12, 23, 31, 13, 32, 21]


def _ltt_kwargs(ltts) -> dict:
    """Map a length-6 LTT array in :data:`UNEQUAL_ARM_LINKS` order to ``d_ij`` kwargs."""
    return {f"d_{link}": ltts[i] for i, link in enumerate(UNEQUAL_ARM_LINKS)}


class _UnequalArmSensMixin:
    """Shared ``get_Sn`` for the six unequal-arm TDI-2 XYZ covariance elements.

    Overrides :meth:`Sensitivity.get_Sn` rather than
    :meth:`Sensitivity.transform` because the closed forms need the *raw*
    ``Soms_d`` / ``Sa_a`` levels -- they build their own shape functions
    internally -- whereas ``transform`` receives a
    :class:`~lisatools.detector.CurrentNoises` whose shapes are already applied.
    The stochastic branch is inherited unchanged from the equal-arm sibling
    each subclass also derives from, so a stochastic background added through
    this class behaves exactly as before.
    """

    #: ``"XX"`` ... ``"YZ"``; the auto elements are realified (see below).
    element: str = None
    #: The generated closed form, as a ``staticmethod``.
    _expr = None

    @classmethod
    def get_Sn(
        cls,
        f: float | np.ndarray,
        model: Optional[lisa_models.LISAModel | str] = lisa_models.sangria,
        include_instrument: bool = True,
        ltts: Optional[np.ndarray] = None,
        **kwargs: dict,
    ) -> float | np.ndarray:
        """PSD / CSD element at ``f`` for the supplied per-link light travel times.

        Args:
            f: Frequency array (Hz).
            model: :class:`~lisatools.detector.LISAModel` carrying the squared
                levels ``Soms_d`` / ``Sa_a``. Spline (``Sn_spl``) models are not
                supported on this path.
            include_instrument: ``False`` drops the instrument term, leaving only
                any stochastic contribution (same contract as the base class).
            ltts: Length-6 light travel times (s) in :data:`UNEQUAL_ARM_LINKS`
                order. Required when ``include_instrument`` is ``True``.
            **kwargs: Stochastic-contribution arguments, forwarded unchanged.
        """
        xp = cls.get_xp(f)
        if include_instrument:
            if ltts is None:
                raise ValueError(
                    f"{cls.__name__} needs per-link light travel times: pass "
                    "ltts=<length-6 array in UNEQUAL_ARM_LINKS order>. Build "
                    "one with UnequalArmInstrumentNoise.ltts_from_orbits()."
                )
            model = lisa_models.check_lisa_model(model)
            if getattr(model, "Sn_spl", None) is not None:
                raise ValueError(
                    f"{cls.__name__} models the covariance analytically from the "
                    "orbit delays; a tabulated Sn_spl model has no unequal-arm "
                    "form. Use the stock equal-arm sensitivity for spline models."
                )
            ltts = xp.asarray(ltts)
            # f == 0 divides by zero in the closed forms exactly as it does in
            # the stock elements; ``get_sensitivity``'s ``fill_nans`` cleans the
            # DC bin afterwards, so only silence the warning here.
            with np.errstate(divide="ignore", invalid="ignore"):
                Sout = cls._expr(
                    f,
                    Soms_d=float(model.Soms_d),
                    Sa_a=float(model.Sa_a),
                    **_ltt_kwargs(ltts),
                )
            if cls.element in ("XX", "YY", "ZZ"):
                # Hermitian => the autos are real; the closed forms carry a
                # ~1e-14 relative imaginary residue from the phase bookkeeping.
                Sout = xp.real(Sout)
        else:
            Sout = 0.0

        if cls._has_stochastic(**kwargs):
            Sout = Sout + cls.stochastic_transform(
                f, cls.get_stochastic_contribution(f, **kwargs), **kwargs
            )
        elif not include_instrument:
            Sout = xp.zeros_like(f)
        return Sout


class UnequalArmXX2TDISens(_UnequalArmSensMixin, X2TDISens):
    """Unequal-arm TDI-2 ``C_XX``."""

    element = "XX"
    _expr = staticmethod(_ua_expr.noise_cov_XX)


class UnequalArmYY2TDISens(_UnequalArmSensMixin, Y2TDISens):
    """Unequal-arm TDI-2 ``C_YY``."""

    element = "YY"
    _expr = staticmethod(_ua_expr.noise_cov_YY)


class UnequalArmZZ2TDISens(_UnequalArmSensMixin, Z2TDISens):
    """Unequal-arm TDI-2 ``C_ZZ``."""

    element = "ZZ"
    _expr = staticmethod(_ua_expr.noise_cov_ZZ)


class UnequalArmXY2TDISens(_UnequalArmSensMixin, XY2TDISens):
    """Unequal-arm TDI-2 ``C_XY`` (complex)."""

    element = "XY"
    _expr = staticmethod(_ua_expr.noise_cov_XY)


class UnequalArmXZ2TDISens(_UnequalArmSensMixin, ZX2TDISens):
    """Unequal-arm TDI-2 ``C_XZ`` (complex).

    Note the element is ``C_XZ`` -- the ``(0, 2)`` entry, matching this class's
    slot in :data:`ELEMENTS` -- even though the equal-arm sibling it derives
    from is spelled ``ZX``. For the real equal-arm CSD the two coincide; here
    they are conjugates, so the distinction matters.
    """

    element = "XZ"
    _expr = staticmethod(_ua_expr.noise_cov_XZ)


class UnequalArmYZ2TDISens(_UnequalArmSensMixin, YZ2TDISens):
    """Unequal-arm TDI-2 ``C_YZ`` (complex)."""

    element = "YZ"
    _expr = staticmethod(_ua_expr.noise_cov_YZ)


#: Unequal-arm XYZ element classes in :data:`ELEMENTS` order.
_UNEQUAL_ARM_ELEMENT_SENS = [
    UnequalArmXX2TDISens,
    UnequalArmYY2TDISens,
    UnequalArmZZ2TDISens,
    UnequalArmXY2TDISens,
    UnequalArmXZ2TDISens,
    UnequalArmYZ2TDISens,
]


class LinkDelayTable:
    """Per-link light travel times tabulated against an absolute clock.

    The delays breathe by ~1.5-1.8% over a two-year run -- far more than the
    ~0.2% spread between the time-averaged arms -- so collapsing them to one
    epoch throws away the dominant variation. This holds the delays as a time
    series and averages them **over each WDM time slice**, giving
    :class:`UnequalArmInstrumentNoise` one delay set per wavelet time column
    that represents the whole column rather than a point sample of it.

    Holds plain numpy arrays only, so it rides through the settings-tree
    deepcopy / pickle round trip (sprint deepcopy/pickle rule).

    Args:
        t: Sample times in seconds on the **absolute** clock the delays are
            tabulated against (for a mojito L1 brick, ``/ltts/sampling`` ``t0``
            plus ``k*dt``). Must be increasing.
        ltts: ``(len(t), 6)`` light travel times in :data:`UNEQUAL_ARM_LINKS`
            order.
        data_t0: Absolute time corresponding to domain time zero. Domains hand
            out a 0-based ``t_arr``, so this is what lines the two clocks up --
            normally the first sample time of the data being analysed.
    """

    def __init__(self, t, ltts, data_t0: float = 0.0):
        self.t = np.asarray(t, dtype=float).ravel()
        self.ltts = np.asarray(ltts, dtype=float)
        if self.ltts.ndim != 2 or self.ltts.shape != (self.t.size, 6):
            raise ValueError(
                f"ltts must have shape (len(t), 6) = ({self.t.size}, 6); "
                f"got {self.ltts.shape}."
            )
        if self.t.size > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError("t must be strictly increasing.")
        self.data_t0 = float(data_t0)
        # Per-domain slice averages are reused across the two basis builds and
        # every per-walker rebuild; keyed by id + identity like the parent's
        # basis cache.
        self._slice_cache: dict = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_slice_cache"] = {}  # derived; rebuilt on demand in the copy
        return state

    @property
    def digest(self) -> tuple:
        """Cheap hashable identity, for the component's basis-cache key."""
        return (self.t.size, float(self.t[0]), float(self.t[-1]),
                self.data_t0, hash(self.ltts.tobytes()))

    def run_average(self) -> np.ndarray:
        """``(6,)`` mean over the whole table (the stationary approximation)."""
        return self.ltts.mean(axis=0)

    def slice_averages(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        """``(Nt, 6)`` delays, each the mean over one domain time slice.

        Slices are centred on the domain's ``t_arr`` and one column-spacing
        wide. Columns the table does not cover (a table coarser than the grid,
        or a grid running past the tabulated span) fall back to linear
        interpolation at the column centre, so the result is always finite.
        """
        t_arr = getattr(settings, "t_arr", None)
        if t_arr is None:
            raise ValueError(
                f"{type(settings).__name__} has no time axis, so per-slice "
                "delays are undefined. Use run_average() for a stationary "
                "(6,) array instead."
            )
        hit = self._slice_cache.get(id(settings))
        if hit is not None and hit[0] is settings:
            return hit[1]

        centres = asnumpy(t_arr).astype(float).ravel()
        nt = centres.size
        width = float(centres[1] - centres[0]) if nt > 1 else 1.0
        edges = np.concatenate([centres - 0.5 * width, [centres[-1] + 0.5 * width]])
        edges = edges + self.data_t0

        idx = np.searchsorted(edges, self.t, side="right") - 1
        keep = (idx >= 0) & (idx < nt)
        counts = np.bincount(idx[keep], minlength=nt)
        out = np.empty((nt, 6), dtype=float)
        for k in range(6):
            sums = np.bincount(
                idx[keep], weights=self.ltts[keep, k], minlength=nt
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, k] = sums / counts
        empty = counts == 0
        if empty.any():
            abs_centres = centres[empty] + self.data_t0
            for k in range(6):
                out[empty, k] = np.interp(abs_centres, self.t, self.ltts[:, k])

        self._slice_cache[id(settings)] = (settings, out)
        return out

    @classmethod
    def from_l1_file(
        cls, path: str, stride: int = 200, data_t0: Optional[float] = None
    ) -> "LinkDelayTable":
        """Build from a mojito L1 brick's ``/ltts`` group.

        Args:
            path: Path to the L1 file.
            stride: Decimation of the tabulated delays. They vary on month
                timescales while the full table is one sample per 2.5 s, so a
                stride of a few hundred keeps every feature at a fraction of
                the memory and read cost.
            data_t0: Absolute time of domain time zero. ``None`` (default) uses
                the file's own ``t0``, which is right whenever the analysed data
                is this file's own span.
        """
        import h5py

        with h5py.File(path, "r") as fh:
            grp = fh["ltts"]
            samp = grp["sampling"].attrs
            t0, dt = float(samp["t0"]), float(samp["dt"])
            cols = [np.asarray(grp[f"ltt_{link}"][::stride]) for link in UNEQUAL_ARM_LINKS]
        ltts = np.stack(cols, axis=1)
        t = t0 + np.arange(ltts.shape[0], dtype=float) * (dt * stride)
        return cls(t, ltts, data_t0=t0 if data_t0 is None else data_t0)


class UnequalArmInstrumentNoise(InstrumentNoise):
    """Orbit-informed TDI-2 instrument covariance with independent arm delays.

    Drop-in replacement for :class:`InstrumentNoise` inside a
    :class:`CompositeSensitivityMatrix`: same ``(nch, nch, *basis_shape_active)``
    contribution, same linear-in-``(Soms_d, Sa_a)`` basis caching, but the six
    link light travel times are carried independently instead of collapsing to
    one constant ``L_SI``.

    Unlike :class:`XYZSensitivityBackend` (which also models unequal arms, in
    C++) this keeps the whole composite machinery -- galactic foreground, SGWB,
    time modulation -- so it can be swapped in without giving any of that up.

    The covariance is **complex Hermitian** in a frequency-domain basis: the
    cross-spectra acquire an imaginary part once ``d_ij != d_ji``. In a WDM
    (real wavelet) basis the fold keeps only ``Re[C_ij]``, which is the correct
    same-pixel covariance -- the antisymmetric part of a CSD contributes to
    cross-*pixel* correlations, which this diagonal-per-pixel framework does
    not represent either way.

    Args:
        ltts: Per-link light travel times in seconds, in
            :data:`UNEQUAL_ARM_LINKS` order. One of

            * :class:`LinkDelayTable` -- **the recommended form.** A delay time
              series that is averaged over each domain time slice at first use,
              giving one delay set per wavelet time column. This tracks the
              ~1.5-1.8% breathing over a two-year run, which a single epoch
              cannot.
            * ``(6,)`` -- one epoch, i.e. stationary noise. Captures the
              arm-to-arm asymmetry but not the breathing. Cheapest to build.
            * ``(Nt, 6)`` -- explicit per-column delays, if you want to supply
              the slice averages yourself. Requires a domain with a time axis.

            Both time-resolved forms fold once per column, which dominates
            setup; it happens once and is cached, so the per-proposal MCMC cost
            is the same as the stationary case either way.
        tdi_generation: Must be 2; the generated closed forms are TDI-2 only.
        model: Noise model supplying the squared levels (name or
            :class:`~lisatools.detector.LISAModel`).
        fill_nans: Forwarded to :func:`get_sensitivity` for the ``f=0`` bin.
        basis_cache: Caller-owned dict enabling the two-basis fast path, exactly
            as on :class:`InstrumentNoise`.
    """

    name = "instrument_unequal_arm"

    def __init__(
        self,
        ltts,
        tdi_generation: int = 2,
        model="sangria",
        fill_nans: float = np.nan,
        basis_cache: Optional[dict] = None,
    ):
        if tdi_generation != 2:
            raise ValueError(
                "UnequalArmInstrumentNoise is TDI-2 only (the closed forms in "
                f"_unequal_arm_expressions are generated for gen 2); got "
                f"tdi_generation={tdi_generation!r}. Use InstrumentNoise for TDI 1.5."
            )
        super().__init__(
            tdi_generation=2, model=model, fill_nans=fill_nans, basis_cache=basis_cache
        )
        if not isinstance(ltts, LinkDelayTable):
            ltts = np.asarray(ltts, dtype=float)
            if ltts.ndim not in (1, 2) or ltts.shape[-1] != 6:
                raise ValueError(
                    f"ltts must be a LinkDelayTable or have shape (6,) / (Nt, 6) "
                    f"in UNEQUAL_ARM_LINKS order {UNEQUAL_ARM_LINKS}; "
                    f"got {ltts.shape}."
                )
        self.ltts = ltts
        # Replaces the stock equal-arm element classes the parent installed.
        self.element_sens_fns = _UNEQUAL_ARM_ELEMENT_SENS

    @staticmethod
    def ltts_from_orbits(orbits, t=None, mode: str = "averaged") -> np.ndarray:
        """Per-link light travel times off a configured orbits object.

        Args:
            orbits: A configured :class:`~lisatools.detector.Orbits` (e.g.
                :class:`~lisatools.detector.L1Orbits`).
            t: Epoch(s) in seconds to evaluate at. ``None`` uses the orbits'
                native LTT grid.
            mode: ``"averaged"`` returns a ``(6,)`` mean over ``t`` -- the
                stationary case. ``"per_epoch"`` returns ``(len(t), 6)``.

        Returns:
            ``(6,)`` or ``(n_epochs, 6)`` light travel times in
            :data:`UNEQUAL_ARM_LINKS` order.

        Note:
            Returns a plain numpy array on purpose. The orbits object holds a
            C++/nanobind wrap and must not enter a settings tree that gets
            deepcopied or pickled (sprint deepcopy/pickle rule); the LTT array
            is the picklable summary that does.
        """
        xp = orbits.xp
        if t is None:
            t_arr = xp.asarray(orbits.ltt_t)
        else:
            t_arr = xp.atleast_1d(xp.asarray(t))
        n = t_arr.shape[0]
        tiled = xp.tile(t_arr[:, None], (1, 6)).flatten()
        links = xp.tile(xp.asarray(UNEQUAL_ARM_LINKS), (n,))
        ltts = orbits.get_light_travel_times(tiled, links).reshape(n, 6)
        ltts = asnumpy(ltts)
        if mode == "averaged":
            return ltts.mean(axis=0)
        if mode == "per_epoch":
            return ltts
        raise ValueError(f"mode must be 'averaged' or 'per_epoch', got {mode!r}.")

    @staticmethod
    def ltts_from_l1_file(
        path: str, mode: str = "averaged", t=None, stride: int = 1
    ) -> np.ndarray:
        """Per-link light travel times read straight out of a mojito L1 brick.

        The brick tabulates every directed link at the full TDI cadence under
        ``/ltts/ltt_<ij>`` -- the delays the data was actually generated with,
        with no orbit interpolation in between. Prefer this over
        :meth:`ltts_from_orbits` when the file is at hand: it agrees with an
        :class:`~lisatools.detector.L1Orbits` built from the same file to ~1e-11
        relative, and it needs no configured orbits object (so nothing
        unpicklable is constructed just to read six numbers).

        Args:
            path: Path to the L1 file.
            mode: ``"averaged"`` returns the ``(6,)`` run-mean, read in blocks so
                the six 25M-sample columns are never all in memory.
                ``"per_epoch"`` returns ``(len(t), 6)`` sampled at ``t``.
            t: Epochs in seconds on the file's own clock (``/ltts/sampling``
                ``t0``); required for ``"per_epoch"``, ignored otherwise.
            stride: Decimation for the averaged pass. The delays vary smoothly
                on month timescales, so a stride of a few hundred changes the
                mean in the 12th digit while reading a fraction of the data.

        Returns:
            ``(6,)`` or ``(len(t), 6)`` light travel times in
            :data:`UNEQUAL_ARM_LINKS` order.
        """
        import h5py

        with h5py.File(path, "r") as fh:
            grp = fh["ltts"]

            if mode == "averaged":
                out = np.empty(6, dtype=float)
                block = max(1, 4_000_000 // max(1, stride)) * max(1, stride)
                for k, link in enumerate(UNEQUAL_ARM_LINKS):
                    dset = grp[f"ltt_{link}"]
                    total, count = 0.0, 0
                    for i in range(0, dset.shape[0], block):
                        chunk = dset[i : i + block : stride]
                        total += float(chunk.sum())
                        count += chunk.size
                    out[k] = total / count
                return out

            if mode == "per_epoch":
                if t is None:
                    raise ValueError("mode='per_epoch' needs t=<epochs in seconds>.")
                samp = grp["sampling"].attrs
                t0, dt = float(samp["t0"]), float(samp["dt"])
                size = int(samp["size"])
                idx = np.rint(
                    (np.atleast_1d(np.asarray(t, dtype=float)) - t0) / dt
                ).astype(np.int64)
                if idx.min() < 0 or idx.max() >= size:
                    raise ValueError(
                        f"requested epochs fall outside the file's span: sample "
                        f"indices {idx.min()}..{idx.max()} vs 0..{size - 1}. Note "
                        f"t is on the file's clock (t0={t0})."
                    )
                # h5py fancy indexing wants strictly increasing, duplicate-free
                # indices; go through the unique set and scatter back.
                uniq, inv = np.unique(idx, return_inverse=True)
                out = np.empty((idx.size, 6), dtype=float)
                for k, link in enumerate(UNEQUAL_ARM_LINKS):
                    out[:, k] = np.asarray(grp[f"ltt_{link}"][uniq])[inv]
                return out

        raise ValueError(f"mode must be 'averaged' or 'per_epoch', got {mode!r}.")

    def _linear_in_noise_levels(self) -> bool:
        """Always linear for a level-carrying model.

        The closed forms take ``Soms_d`` / ``Sa_a`` directly and bake in their
        own shape functions -- they never call ``model.lisanoises`` -- so the
        parent's check (which inspects that method) does not apply here. What
        matters is only that the model actually carries the two levels.
        """
        model = self.model
        return not isinstance(model, str) and getattr(model, "Sn_spl", None) is None

    def _basis_cache_key_extra(self) -> tuple:
        """Distinguish cached bases built from different light travel times."""
        if isinstance(self.ltts, LinkDelayTable):
            return self.ltts.digest
        return (self.ltts.shape, self.ltts.tobytes())

    def _resolve_ltts(self, settings) -> np.ndarray:
        """The delays this domain actually needs: ``(6,)`` or ``(Nt, 6)``.

        A :class:`LinkDelayTable` collapses here -- to per-slice means on a
        domain with a time axis, or to the run mean on one without.
        """
        if not isinstance(self.ltts, LinkDelayTable):
            return self.ltts
        if getattr(settings, "t_arr", None) is None:
            return self.ltts.run_average()
        return self.ltts.slice_averages(settings)

    def _sensitivity_kwargs(self, settings, model) -> dict:
        """``get_sensitivity`` kwargs selecting the stationary / per-column path."""
        ltts = self._resolve_ltts(settings)
        if ltts.ndim == 1:
            return dict(model=model, ltts=ltts)

        # (Nt, 6): one LTT row per wavelet time column.
        time_axis = _basis_time_axis(settings)
        if time_axis is None:
            raise ValueError(
                f"{type(settings).__name__} has no time axis, so per-epoch light "
                "travel times (Nt, 6) cannot be used. Pass a (6,) orbit-averaged "
                "ltts array instead (ltts_from_orbits(..., mode='averaged'))."
            )
        nt = getattr(settings, "Nt", None)
        if nt is None or ltts.shape[0] != nt:
            raise ValueError(
                f"ltts has {ltts.shape[0]} epochs but the domain has Nt={nt}; "
                "the non-stationary path needs one LTT row per wavelet time column."
            )
        # get_sensitivity does NOT merge the top-level kwargs into kwargs_list,
        # so every entry has to carry the model as well as its own delays.
        return dict(
            model=model,
            stationary=False,
            kwargs_list=[dict(model=model, ltts=row) for row in ltts],
        )

    def _direct_base_covariance(self, settings, model) -> np.ndarray:
        """Hermitian ``(nch, nch, *basis)`` covariance from the closed forms.

        Differs from the parent in two ways: the per-link delays are threaded
        through to the element classes, and the lower triangle gets the
        **conjugate** (the parent mirrors the value, which is only right for the
        real equal-arm CSDs).
        """
        xp = settings.xp
        nch = self.nchannels
        kw = self._sensitivity_kwargs(settings, model)
        elems = [
            xp.asarray(
                get_sensitivity(
                    settings, sens_fn=fn, fill_nans=self.fill_nans, **kw
                )
            )
            for fn in self.element_sens_fns
        ]
        # FD keeps the complex cross-spectra; the WDM fold has already taken the
        # real part, so there the promotion is a no-op.
        dtype = xp.result_type(*[e.dtype for e in elems])
        C = xp.zeros((nch, nch) + tuple(settings.basis_shape_active), dtype=dtype)
        for (i, j), arr in zip(ELEMENTS, elems):
            C[i, j] = arr
            C[j, i] = arr if i == j else xp.conj(arr)
        return C


class GalacticForeground(SeparableComponent):
    """Galactic confusion foreground with a per-element time modulation.

    Base covariance : the foreground *magnitude*
    ``Sgal_mag[basis]`` — the auto-channel (XX) foreground in the domain basis —
    is placed on every element, computed domain-agnostically as the X-channel
    *stochastic-only* sensitivity (``include_instrument=False``), so no instrument
    term is included here. The per-element structure (including the off-diagonal
    sign) and the slow time variation live entirely in the modulation.

    Args:
        foreground_params: Parameters for ``stochastic_fn`` (for the default
            :class:`HyperbolicTangentGalacticForeground` these are
            ``(amp, fk, alpha, s1, s2)``).
        modulation: One of: ``None`` (the isotropic/stationary limit — diagonals
            ``1``, off-diagonals ``-1/2`` — which reproduces the stationary
            foreground); a ``(nch, nch)`` constant matrix; a ``(nch, nch, Ntime)``
            array; or a callable ``t_arr -> (nch, nch, Ntime)`` (build one from
            six per-element functions with :func:`modulation_from_elements`).
        tdi_generation: 1 or 2 (used to pick the X-channel sensitivity used to
            extract the foreground magnitude).
        stochastic_fn: Stochastic foreground model (class or name).
    """

    name = "galactic_foreground"

    def __init__(
        self,
        foreground_params: Sequence[float],
        modulation: Optional[object] = None,
        tdi_generation: int = 2,
        stochastic_fn=HyperbolicTangentGalacticForeground,
    ):
        if tdi_generation not in _XYZ_ELEMENT_SENS:
            raise ValueError(f"tdi_generation must be 1 or 2, got {tdi_generation!r}.")
        self.foreground_params = tuple(foreground_params)
        self._modulation = modulation
        self.tdi_generation = tdi_generation
        self.stochastic_fn = check_stochastic(stochastic_fn)

    def base_covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        xp = settings.xp
        nch = self.nchannels
        Xsens = _XYZ_ELEMENT_SENS[self.tdi_generation][0]
        # foreground magnitude in the domain basis: stochastic contribution only
        # (no instrument term), folded through the same domain dispatch.
        mag = get_sensitivity(
            settings,
            sens_fn=Xsens,
            stochastic_params=self.foreground_params,
            stochastic_function=self.stochastic_fn,
            include_instrument=False,
            fill_nans=0.0,
        )
        C = xp.zeros((nch, nch) + tuple(settings.basis_shape_active), dtype=mag.dtype)
        for (i, j) in ELEMENTS:
            C[i, j] = mag
            C[j, i] = mag
        return C

    def time_modulation(self, settings: domains.DomainSettingsBase):
        xp = settings.xp
        nch = self.nchannels

        if self._modulation is None:
            # isotropic / stationary limit: diag = 1, off-diag = -1/2 (constant)
            M = xp.full((nch, nch), -0.5)
            for i in range(nch):
                M[i, i] = 1.0
            return M

        if callable(self._modulation):
            time_axis = _basis_time_axis(settings)
            if time_axis is None:
                raise ValueError(
                    f"{type(settings).__name__} has no time axis; cannot evaluate a "
                    "callable (time-varying) foreground modulation here."
                )
            return self._modulation(settings.t_arr)

        return xp.asarray(self._modulation)


class SGWB(SeparableComponent):
    """Stochastic gravitational-wave background component (stationary by default).

    The SGWB spectral template ``Sgw(f)`` is folded through the equal-arm TDI
    response (``R_XX = 4 x^2 sin^2 x``, off-diagonals ``-1/2 R_XX``) and summed
    into the covariance. Like the galactic foreground, the *magnitude* — the
    auto-channel (XX) response in the domain basis — is extracted
    domain-agnostically as the X-channel *stochastic-only* sensitivity
    (``include_instrument=False``, no instrument term), and placed on every
    element; the per-element structure lives in the modulation. The isotropic
    default (diag ``1``, off-diag ``-1/2``) reproduces the equal-arm covariance
    ``C_XY = -1/2 C_XX``. Pass a ``modulation`` for an anisotropic / time-varying
    background.

    This uses the analytic equal-arm response (the equal-arm limit of GLASS's
    precomputed ``sgwb_response_xyz2.dat``); a tabulated unequal-arm response is
    a possible later enhancement.

    Args:
        sgwb_params: Parameters for ``stochastic_fn`` — e.g.
            :class:`~lisatools.stochastic.PowerLawSGWB` ``(log10_A, alpha)``,
            :class:`~lisatools.stochastic.LogNormalSGWB`
            ``(log10_A, log10_fstar, log10_D)``,
            :class:`~lisatools.stochastic.PhaseTransitionSGWB`
            ``(rb, b, log10_Ap, log10_fp)``.
        stochastic_fn: SGWB spectral template (class or stock name).
        modulation: ``None`` (stationary isotropic — the usual case); a
            ``(nch, nch)`` constant matrix; a ``(nch, nch, Ntime)`` array; or a
            callable ``t_arr -> (nch, nch, Ntime)``.
        tdi_generation: 1 or 2 (used to pick the X-channel sensitivity used to
            extract the SGWB magnitude).
    """

    name = "sgwb"

    def __init__(
        self,
        sgwb_params: Sequence[float],
        stochastic_fn,
        modulation: Optional[object] = None,
        tdi_generation: int = 2,
    ):
        if tdi_generation not in _XYZ_ELEMENT_SENS:
            raise ValueError(f"tdi_generation must be 1 or 2, got {tdi_generation!r}.")
        self.sgwb_params = tuple(sgwb_params)
        self._modulation = modulation
        self.tdi_generation = tdi_generation
        self.stochastic_fn = check_stochastic(stochastic_fn)

    def base_covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        xp = settings.xp
        nch = self.nchannels
        Xsens = _XYZ_ELEMENT_SENS[self.tdi_generation][0]
        # SGWB magnitude in the domain basis: stochastic contribution only
        # (no instrument term), folded through the same domain dispatch.
        mag = get_sensitivity(
            settings,
            sens_fn=Xsens,
            stochastic_params=self.sgwb_params,
            stochastic_function=self.stochastic_fn,
            include_instrument=False,
            fill_nans=0.0,
        )
        C = xp.zeros((nch, nch) + tuple(settings.basis_shape_active), dtype=mag.dtype)
        for (i, j) in ELEMENTS:
            C[i, j] = mag
            C[j, i] = mag
        return C

    def time_modulation(self, settings: domains.DomainSettingsBase):
        xp = settings.xp
        nch = self.nchannels

        if self._modulation is None:
            # isotropic / stationary limit: diag = 1, off-diag = -1/2 (constant)
            M = xp.full((nch, nch), -0.5)
            for i in range(nch):
                M[i, i] = 1.0
            return M

        if callable(self._modulation):
            time_axis = _basis_time_axis(settings)
            if time_axis is None:
                raise ValueError(
                    f"{type(settings).__name__} has no time axis; cannot evaluate a "
                    "callable (time-varying) SGWB modulation here."
                )
            return self._modulation(settings.t_arr)

        return xp.asarray(self._modulation)


def _interp1d_along_axis(
    x_new: np.ndarray, x_old: np.ndarray, y: np.ndarray, axis: int
) -> np.ndarray:
    """Linear interpolation of ``y`` along ``axis`` with edge clamping.

    ``x_old`` must be strictly increasing. Values of ``x_new`` outside the
    tabulated range are clamped to the end values (matching ``np.interp``).
    """
    idx = np.searchsorted(x_old, x_new, side="right") - 1
    idx = np.clip(idx, 0, len(x_old) - 2)
    x_lo = x_old[idx]
    x_hi = x_old[idx + 1]
    w = np.clip((x_new - x_lo) / (x_hi - x_lo), 0.0, 1.0)
    y_lo = np.take(y, idx, axis=axis)
    y_hi = np.take(y, idx + 1, axis=axis)
    shape = [1] * y.ndim
    shape[axis] = len(x_new)
    w = w.reshape(shape)
    return y_lo * (1.0 - w) + y_hi * w


class MojitoNoiseEstimates(NoiseComponent):
    """Tabulated instrument-noise covariance read from a mojito NOISE L1 brick.

    The mojito NOISE brick stores daily-estimated ``(nch, nch)`` noise
    covariance matrices on a log-uniform frequency grid
    (``noise_estimates/XYZ`` etc., units Hz^2/Hz of the raw laser-frequency
    TDI). This component divides by ``laser_frequency**2`` — the PSD of the
    dimensionless ``tdis.xyz_doppler`` data the global fit ingests — and
    projects onto the domain basis:

    * frequency: linear interpolation in ``log10(f)`` (the tabulated grid is
      log-uniform); bins outside the tabulated range get ``fill_value``;
    * FD basis: the interpolated (time-averaged) one-sided PSD, matching
      :func:`get_sensitivity`'s FD convention;
    * WDM basis: the interpolated full-resolution PSD is folded into the
      wavelet basis through the same validated
      ``FDSignal.wdmtransform(is_psd=True)`` path :class:`InstrumentNoise`
      uses, so the two components are drop-in comparable. With
      ``time_dependent=True`` (default) the slow daily variation is applied
      as a per-layer ratio on top of the time-averaged fold. Domain times
      are interpreted as seconds since the brick's TDI data start (plus
      ``t_offset`` for runs whose data window does not begin at the file
      start; the estimates vary by only a few percent over the year, so
      modest offsets are benign).

    Pickle/deepcopy-safe: only the path and scalar knobs are carried; the
    loaded table is a lazy cache dropped on pickling (LISA Analysis Tools
    settings-tree rule).

    Args:
        path: Path to the mojito NOISE L1 ``.h5`` file.
        which: ``"xyz"`` (default) or ``"aet"`` — which TDI estimate table to
            read. Use the one matching the run's TDI channels.
        time_dependent: If ``True`` (default), apply the daily estimates'
            time variation along the domain's time axis (WDM). ``False``
            time-averages the estimates into a stationary covariance.
        t_offset: Seconds added to the domain's time axis before matching it
            to the brick's estimate times (e.g. a data-window start offset).
        fill_value: Value assigned to frequency bins outside the tabulated
            range (default ``0.0``; the zeroed cells are filtered by the
            ``detC`` mask downstream, matching the composite-backend
            convention).
    """

    name = "mojito_noise_estimates"

    def __init__(
        self,
        path: str,
        *,
        which: str = "xyz",
        time_dependent: bool = True,
        t_offset: float = 0.0,
        fill_value: float = 0.0,
    ):
        if which not in ("xyz", "aet"):
            raise ValueError(f"which must be 'xyz' or 'aet', got {which!r}.")
        self.path = str(path)
        self.which = which
        self.time_dependent = bool(time_dependent)
        self.t_offset = float(t_offset)
        self.fill_value = float(fill_value)
        self._tab = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tab"] = None  # lazy cache; reload from ``path`` after unpickle
        return state

    def _load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(est_times_rel, est_freqs, cov)`` with ``cov`` shape ``(Nd, Nq, nch, nch)``.

        ``est_times_rel`` is seconds since the brick's TDI data start;
        ``cov`` is real, symmetrized, in fractional-frequency units.
        """
        if self._tab is None:
            from mojito import MojitoL1File

            with MojitoL1File(self.path) as f:
                lam = float(f.laser_frequency)
                est = f.noise_estimates
                freqs = np.asarray(est.freq_sampling.f(), dtype=np.float64)
                est_t = np.asarray(est.time_sampling.t(), dtype=np.float64)
                data_t0 = float(f.tdis.time_sampling.t0)
                cov = np.asarray(getattr(est, self.which)[:]).real / lam**2
            cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
            self._tab = (est_t - data_t0, freqs, np.ascontiguousarray(cov))
        return self._tab

    def _interp_freq(self, f_arr: np.ndarray, tab: np.ndarray, axis: int) -> np.ndarray:
        """Interpolate the table onto ``f_arr`` (linear in log10 f) along ``axis``.

        Frequencies outside the tabulated range get ``fill_value``.
        """
        _, est_f, _ = self._load()
        in_range = (f_arr >= est_f[0]) & (f_arr <= est_f[-1])
        f_safe = np.where(in_range, f_arr, est_f[0])
        vals = _interp1d_along_axis(np.log10(f_safe), np.log10(est_f), tab, axis=axis)
        mask_idx = [slice(None)] * vals.ndim
        mask_idx[axis] = ~in_range
        vals[tuple(mask_idx)] = self.fill_value
        return vals

    def covariance(self, settings: domains.DomainSettingsBase) -> np.ndarray:
        est_t, est_f, cov = self._load()
        xp = settings.xp
        nch = self.nchannels
        if cov.shape[-1] != nch:
            raise ValueError(
                f"tabulated estimates are {cov.shape[-1]}x{cov.shape[-2]}; "
                f"expected {nch} channels."
            )
        cov_avg = cov.mean(axis=0)  # (Nq, nch, nch)

        if isinstance(settings, domains.FDSettings):
            f_arr = np.asarray(asnumpy(settings.f_arr), dtype=np.float64)
            vals = self._interp_freq(f_arr, cov_avg, axis=0)  # (Nf, nch, nch)
            return xp.asarray(np.moveaxis(vals, (0, 1, 2), (2, 0, 1)))

        if not isinstance(settings, domains.WDMSettings):
            raise NotImplementedError(
                f"MojitoNoiseEstimates supports FDSettings and WDMSettings; "
                f"got {type(settings).__name__}."
            )

        # --- WDM: fold the interpolated full-resolution PSD into the wavelet
        # basis exactly the way InstrumentNoise does (one fold per element on
        # the time-averaged table), then apply the slow daily variation as a
        # per-layer ratio. ---
        f_full = np.asarray(
            asnumpy(settings.xp.fft.rfftfreq(settings.N, settings.data_dt)),
            dtype=np.float64,
        )
        full = self._interp_freq(f_full, cov_avg, axis=0)  # (Nfull, nch, nch)
        basis_shape = tuple(settings.basis_shape_active)
        fd_settings = domains.FDSettings(
            f_full.shape[0],
            float(f_full[1] - f_full[0]),
            force_backend=settings.backend,
        )
        folded = np.empty((nch, nch, basis_shape[0]), dtype=np.float64)
        for (i, j) in ELEMENTS:
            psd_fd = domains.FDSignal(settings.xp.asarray(full[:, i, j]), fd_settings)
            col = asnumpy(
                np.real(psd_fd.wdmtransform(settings=settings, is_psd=True)[0])
            )[:, 0]
            folded[i, j] = col
            folded[j, i] = col

        if not self.time_dependent:
            C = np.broadcast_to(folded[..., None], (nch, nch) + basis_shape)
            return xp.asarray(np.ascontiguousarray(C))

        # daily variation at the layer centres, as a ratio to the time average
        f_layers = np.asarray(asnumpy(settings.f_arr), dtype=np.float64)
        t_arr = np.asarray(asnumpy(settings.t_arr), dtype=np.float64) + self.t_offset
        by_day = _interp1d_along_axis(t_arr, est_t, cov, axis=0)  # (Nt, Nq, nch, nch)
        num = self._interp_freq(f_layers, by_day, axis=1)  # (Nt, Nf, nch, nch)
        den = self._interp_freq(f_layers, cov_avg, axis=0)  # (Nf, nch, nch)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = num / den[None]
        ratio[~np.isfinite(ratio)] = 1.0
        # (Nt, Nf, nch, nch) -> (nch, nch, Nf, Nt)
        ratio = np.moveaxis(ratio, (0, 1, 2, 3), (3, 2, 0, 1))
        return xp.asarray(folded[..., None] * ratio)

    def fit_scalar_params(
        self,
        band: Tuple[float, float] = (1e-4, 2.5e-2),
        tdi_generation: int = 2,
    ) -> Tuple[float, float]:
        """Estimate scalar ``(Soms_d, Sa_a)`` from the tabulated estimates.

        The TDI instrument PSD is linear in ``(Soms_d**2, Sa_a**2)``, so a
        weighted linear least-squares of the analytic X-channel model against
        the time-averaged tabulated diagonals over ``band`` recovers the
        parameters exactly (1/S weighting so each decade counts equally).
        Only meaningful for ``which="xyz"``.

        Returns:
            ``(Soms_d, Sa_a)`` in linear (square-root) units, the convention
            of :class:`CompositeSensitivityBackend` / the psd-branch priors.
        """
        if self.which != "xyz":
            raise ValueError("fit_scalar_params requires which='xyz'.")
        if tdi_generation not in _XYZ_ELEMENT_SENS:
            raise ValueError(f"tdi_generation must be 1 or 2, got {tdi_generation!r}.")
        _, est_f, cov = self._load()
        avg = cov.mean(axis=0)  # (Nq, nch, nch)
        mask = (est_f >= band[0]) & (est_f <= band[1])
        if not mask.any():
            raise ValueError(f"band {band} has no overlap with the tabulated grid.")
        fb = est_f[mask]
        Xsens = _XYZ_ELEMENT_SENS[tdi_generation][0]
        orbits = lisa_models.DefaultOrbits()
        # LISAModel carries the SQUARED levels; unit-basis responses give the
        # linear-model columns.
        resp_oms = Xsens.get_Sn(fb, model=lisa_models.LISAModel(1.0, 0.0, orbits, "oms_basis"))
        resp_acc = Xsens.get_Sn(fb, model=lisa_models.LISAModel(0.0, 1.0, orbits, "acc_basis"))
        rows, rhs = [], []
        for i in range(self.nchannels):
            target = avg[mask, i, i]
            w = 1.0 / target
            rows.append(np.stack([resp_oms * w, resp_acc * w], axis=1))
            rhs.append(target * w)
        coef, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
        if coef[0] < 0 or coef[1] < 0:
            raise ValueError(
                f"noise-parameter fit returned negative squared levels {coef}; "
                "the tabulated estimates do not look like an instrument PSD."
            )
        return float(np.sqrt(coef[0])), float(np.sqrt(coef[1]))


def estimate_noise_params_from_file(
    path: str,
    *,
    band: Tuple[float, float] = (1e-4, 2.5e-2),
    tdi_generation: int = 2,
) -> Tuple[float, float]:
    """``(Soms_d, Sa_a)`` fit to a mojito NOISE brick's tabulated estimates.

    Thin wrapper over :meth:`MojitoNoiseEstimates.fit_scalar_params`.
    """
    return MojitoNoiseEstimates(path).fit_scalar_params(
        band=band, tdi_generation=tdi_generation
    )


class CompositeSensitivityMatrix(SensitivityMatrixBase):
    """Sensitivity matrix built as a sum of :class:`NoiseComponent` objects.

    Args:
        settings: Domain settings the matrix is evaluated on (FD, WDM, …).
        components: :class:`NoiseComponent` list to sum. All must produce the same
            ``(nch, nch, *basis_shape_active)`` shape.
        skip_inv_det: Skip determinant/inverse computation (e.g. for slicing).
    """

    def __init__(
        self,
        settings: domains.DomainSettingsBase,
        components: Sequence[NoiseComponent],
        skip_inv_det: bool = False,
    ):
        SensitivityMatrixBase.__init__(self, settings, skip_inv_det=skip_inv_det)
        self.components = list(components)
        if not self.components:
            raise ValueError("CompositeSensitivityMatrix needs at least one component.")
        # Per-component contribution cache: a param change only recomputes the
        # touched component before re-summing (the expensive det/inv runs once).
        self._contrib_cache: dict[int, np.ndarray] = {}
        self.rebuild()

    def rebuild(self, indices: Optional[Sequence[int]] = None) -> None:
        """(Re)compute component contributions and re-sum into ``sens_mat``.

        Args:
            indices: Component indices to recompute. ``None`` recomputes all;
                cached contributions are reused for the rest.
        """
        if indices is None:
            indices = range(len(self.components))
        for i in indices:
            self._contrib_cache[i] = self.components[i].covariance(self.basis_settings)

        # Object-oriented sum: every intermediate ``+`` returns a
        # :class:`SensitivityMatrixBase` with dirty ``invC`` / ``detC``, so the
        # matrix inverse runs exactly once -- when this method's caller (or the
        # likelihood code) first reads off the inverse.
        accum = SensitivityMatrixBase(self.basis_settings)
        accum.sens_mat = self._contrib_cache[0]
        for i in range(1, len(self.components)):
            accum = accum + self._contrib_cache[i]
        # adopt the summed array onto self; the assignment marks self dirty so
        # invC / detC will be lazily computed on first read off the composite.
        self.sens_mat = accum.sens_mat

    def update_component(self, index: int) -> None:
        """Recompute a single component (after changing its params) and re-sum."""
        self.rebuild(indices=[index])

    def draw_correlated_td_noise(
        self, dt: float, *, seed: Optional[int | np.random.Generator] = None
    ) -> np.ndarray:
        """Draw a ``(nchannels, N)`` real TD noise realization from this matrix.

        The matrix must be evaluated on an :class:`~lisatools.domains.FDSettings`
        rFFT grid. A per-frequency Cholesky factor of ``self.sens_mat`` (the
        ``(nch, nch, Nf)`` covariance) is applied to a unit complex-Gaussian draw
        and inverse-rFFT'd (LDC convention ``td = irfft(fd) / dt``). Frequencies
        whose covariance is not positive-definite (e.g. the ``f = 0`` bin) fall
        back to an independent diagonal-``sqrt`` draw.

        The realization is computed on the host (NumPy) so a fixed integer
        ``seed`` gives a reproducible, backend-independent stream.

        Args:
            dt: Time-domain sample spacing in seconds. ``N`` is derived as
                ``round(1 / (df * dt))`` from the grid ``df``.
            seed: Seed or :class:`numpy.random.Generator` for the draw.

        Returns:
            ``(nchannels, N)`` ``float64`` array of correlated TD noise.
        """
        if not isinstance(self.basis_settings, domains.FDSettings):
            raise ValueError(
                "draw_correlated_td_noise requires an FDSettings basis; got "
                f"{type(self.basis_settings).__name__}."
            )
        cov = asnumpy(self.sens_mat)  # (nch, nch, Nf_rfft)
        nch = cov.shape[0]
        df = self.basis_settings.df
        N = int(round(1.0 / (df * dt)))
        Nf_rfft = N // 2 + 1
        if cov.shape[-1] != Nf_rfft:
            raise ValueError(
                f"covariance length {cov.shape[-1]} != N//2+1 = {Nf_rfft} implied "
                f"by df={df!r}, dt={dt!r}."
            )

        rng = np.random.default_rng(seed)
        norm = 0.5 * (1.0 / df) ** 0.5
        z = rng.normal(0, norm, (nch, Nf_rfft)) + 1j * rng.normal(
            0, norm, (nch, Nf_rfft)
        )

        n_fd = np.zeros_like(z)
        eye = np.eye(nch) * 1e-60  # regularise the DC / non-PD bins
        for k in range(Nf_rfft):
            C_k = cov[..., k] + eye
            try:
                L = np.linalg.cholesky(C_k)
            except np.linalg.LinAlgError:
                diag = np.maximum(np.diag(cov[..., k]), 0.0)
                n_fd[:, k] = np.sqrt(diag) * z[:, k]
                continue
            n_fd[:, k] = L @ z[:, k]

        n_td = np.fft.irfft(n_fd, n=N, axis=-1) / dt
        return n_td.astype(np.float64)


class MojitoNoiseSensitivityMatrix(CompositeSensitivityMatrix):
    """Stock-style sensitivity matrix driven by a mojito NOISE brick.

    One-liner counterpart of the parametric stock matrices
    (:class:`XYZ2SensitivityMatrix` etc.) for the tabulated file-driven noise
    model: a :class:`CompositeSensitivityMatrix` whose instrument component is
    a :class:`MojitoNoiseEstimates` table, plus any ``extra_components``
    (galactic foreground, SGWB, ...). Use it anywhere a sensitivity matrix is
    accepted — e.g. SNRs via :class:`~lisatools.analysiscontainer.AnalysisContainer`
    or :func:`~lisatools.diagnostic.inner_product` / ``snr``::

        sens = MojitoNoiseSensitivityMatrix(settings, noise_file)
        snr = inner_product(sig, sig, psd=sens).real ** 0.5

    Args:
        settings: Domain settings the matrix is evaluated on (FD, WDM, STFT).
        path: Path to the mojito NOISE L1 ``.h5`` file.
        extra_components: Additional :class:`NoiseComponent` instances summed
            on top of the tabulated instrument noise.
        skip_inv_det: Skip determinant/inverse computation (e.g. for slicing).
        **component_kwargs: Forwarded to :class:`MojitoNoiseEstimates`
            (``which``, ``time_dependent``, ``t_offset``, ``fill_value``).
    """

    def __init__(
        self,
        settings: domains.DomainSettingsBase,
        path: str,
        *,
        extra_components: Sequence[NoiseComponent] = (),
        skip_inv_det: bool = False,
        **component_kwargs,
    ):
        component = MojitoNoiseEstimates(path, **component_kwargs)
        super().__init__(
            settings,
            [component, *extra_components],
            skip_inv_det=skip_inv_det,
        )


def generate_correlated_instrument_noise_td(
    N: int,
    dt: float,
    *,
    Soms_d: float,
    Sa_a: float,
    tdi_generation: int,
    seed: Optional[int | np.random.Generator] = None,
    model_name: str = "synthetic_noise_model",
    orbits: Optional[Orbits] = None,
) -> np.ndarray:
    """Sample a ``(nchannels, N)`` TD instrument-noise realization.

    Builds an :class:`InstrumentNoise` covariance for a
    ``LISAModel(Soms_d**2, Sa_a**2, orbits, model_name)`` on an rFFT
    :class:`~lisatools.domains.FDSettings` grid and delegates to
    :meth:`CompositeSensitivityMatrix.draw_correlated_td_noise`.

    Args:
        N: Number of TD samples.
        dt: Sample spacing in seconds.
        Soms_d / Sa_a: Linear instrument-noise levels (the covariance uses the
            squared values internally, matching the stock model convention).
        tdi_generation: 1 (TDI 1.5) or 2 (TDI 2.0).
        seed: RNG seed / generator.
        model_name: Name recorded on the :class:`LISAModel` (does not affect the
            realization).
        orbits: Orbits carrier for the model (``DefaultOrbits()`` if ``None``;
            ``LISAModel.lisanoises`` only reads the noise levels).

    Returns:
        ``(nchannels, N)`` ``float64`` array of correlated TD noise.
    """
    Nf_rfft = N // 2 + 1
    df = 1.0 / (N * dt)
    fd_settings = domains.FDSettings(N=Nf_rfft, df=df, force_backend="cpu")
    if orbits is None:
        orbits = lisa_models.DefaultOrbits()
    model = lisa_models.LISAModel(Soms_d ** 2, Sa_a ** 2, orbits, model_name)
    instrument = InstrumentNoise(
        tdi_generation=tdi_generation, model=model, fill_nans=0.0
    )
    sens = CompositeSensitivityMatrix(fd_settings, [instrument])
    return sens.draw_correlated_td_noise(dt, seed=seed)


def annual_amplitude_envelope(
    t_arr, *, amp: float, phase0: float, period: float = YRSID_SI
) -> np.ndarray:
    """Per-sample amplitude envelope ``1 + amp*cos(2*pi*t/period + phase0)``."""
    return 1.0 + amp * np.cos(2.0 * np.pi * np.asarray(t_arr) / period + phase0)


def annual_modulation_matrix(
    t_arr, *, amp: float, phase0: float, period: float = YRSID_SI
) -> np.ndarray:
    """``(nch, nch, Nt)`` isotropic-foreground modulation with an annual envelope.

    Standard isotropic per-element pattern (diag ``1``, off-diag ``-1/2``) times
    the *power-domain* annual envelope (the square of
    :func:`annual_amplitude_envelope`, since the modulation multiplies the
    covariance / power, not the amplitude).
    """
    t_arr = np.asarray(t_arr)
    base = np.array(
        [
            [1.0, -0.5, -0.5],
            [-0.5, 1.0, -0.5],
            [-0.5, -0.5, 1.0],
        ]
    )
    env = (
        annual_amplitude_envelope(t_arr, amp=amp, phase0=phase0, period=period) ** 2
    )
    return base[:, :, None] * env[None, None, :]


class AnnualModulatedGalacticForeground(GalacticForeground):
    """:class:`GalacticForeground` with an annual amplitude modulation pre-bound.

    The per-element modulation is :func:`annual_modulation_matrix` with ``amp`` /
    ``phase0`` / ``period`` bound at construction, so the covariance acquires a
    ``(1 + amp*cos(...))**2`` annual envelope. ``foreground_params`` are forwarded
    to ``stochastic_fn`` (default
    :class:`~lisatools.stochastic.FittedHyperbolicTangentGalacticForeground`, for
    which ``(Tobs,)`` is the expected parameter tuple).
    """

    def __init__(
        self,
        foreground_params: Sequence[float],
        *,
        amp: float,
        phase0: float,
        tdi_generation: int = 2,
        period: float = YRSID_SI,
        stochastic_fn=None,
    ):
        warnings.warn(
            "AnnualModulatedGalacticForeground is deprecated; use the general "
            "modulation framework directly: GalacticForeground(foreground_params, "
            "modulation=functools.partial(annual_modulation_matrix, amp=..., "
            "phase0=...)) — or GalForTimeModulation(path) for a tabulated "
            "modulation.",
            DeprecationWarning,
            stacklevel=2,
        )
        if stochastic_fn is None:
            stochastic_fn = FittedHyperbolicTangentGalacticForeground
        modulation = functools.partial(
            annual_modulation_matrix, amp=amp, phase0=phase0, period=period
        )
        super().__init__(
            foreground_params=foreground_params,
            modulation=modulation,
            tdi_generation=tdi_generation,
            stochastic_fn=stochastic_fn,
        )


def generate_foreground_td(
    N: int,
    dt: float,
    *,
    Tobs: float,
    foreground_params: Optional[Sequence[float]],
    tdi_generation: int,
    seed: Optional[int | np.random.Generator] = None,
    stochastic_fn=None,
    envelope: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Sample a ``(nchannels, N)`` TD galactic-foreground realization.

    Builds a stationary :class:`GalacticForeground` covariance on an rFFT
    :class:`~lisatools.domains.FDSettings` grid, draws a correlated TD
    realization via
    :meth:`CompositeSensitivityMatrix.draw_correlated_td_noise`, and (optionally)
    multiplies each channel by a per-sample amplitude-domain ``envelope(t)``
    (e.g. ``sqrt`` of :func:`annual_amplitude_envelope`) to make the realization
    non-stationary in the same way an annual-modulated model covariance is.

    Args:
        N: Number of TD samples.
        dt: Sample spacing in seconds.
        Tobs: Observation span; used as the default ``foreground_params`` tuple
            ``(Tobs,)`` when ``foreground_params`` is ``None``.
        foreground_params: Parameters for ``stochastic_fn``; ``None`` -> ``(Tobs,)``.
        tdi_generation: 1 or 2.
        seed: RNG seed / generator.
        stochastic_fn: Stochastic foreground model (default
            :class:`~lisatools.stochastic.FittedHyperbolicTangentGalacticForeground`).
        envelope: Optional callable ``t_arr -> (N,)`` amplitude-domain multiplier.

    Returns:
        ``(nchannels, N)`` ``float64`` array of correlated TD foreground.
    """
    if stochastic_fn is None:
        stochastic_fn = FittedHyperbolicTangentGalacticForeground
    Nf_rfft = N // 2 + 1
    df = 1.0 / (N * dt)
    fd_settings = domains.FDSettings(N=Nf_rfft, df=df, force_backend="cpu")
    fg_params = foreground_params if foreground_params is not None else (Tobs,)
    fg = GalacticForeground(
        foreground_params=fg_params,
        modulation=None,  # stationary base; the envelope is applied in TD below
        tdi_generation=tdi_generation,
        stochastic_fn=stochastic_fn,
    )
    sens = CompositeSensitivityMatrix(fd_settings, [fg])
    n_td = sens.draw_correlated_td_noise(dt, seed=seed)
    if envelope is not None:
        t_arr = np.arange(N) * dt  # caller adds any t0 separately
        n_td = n_td * np.asarray(envelope(t_arr))[None, :]
    return n_td


_TDI_CHAN_TO_GENERATION = {
    "XYZ": 2,
    "AET": 2,
    "XYZ2": 2,
    "AET2": 2,
    "XYZ1": 1,
    "AET1": 1,
}


def tdi_generation_from_channel(tdi_chan: str) -> int:
    """Map a TDI channel label to its TDI generation (1 or 2).

    ``"XYZ"/"AET"/"XYZ2"/"AET2" -> 2``, ``"XYZ1"/"AET1" -> 1``. Raises
    :class:`ValueError` on an unrecognised label.
    """
    try:
        return _TDI_CHAN_TO_GENERATION[tdi_chan]
    except KeyError:
        raise ValueError(
            f"TDI channel {tdi_chan!r} not recognised; expected one of "
            f"{sorted(_TDI_CHAN_TO_GENERATION)}."
        )


class CompositeSensitivityBackend(SensitivityBackendBase):
    """Callable wrapper that produces :class:`CompositeSensitivityMatrix` instances
    parameterised by per-walker PSD (and optional galactic-foreground / SGWB)
    coordinates.

    Shares :class:`SensitivityBackendBase` (hence the backend dispatch + the
    ``__call__`` contract) with :class:`XYZSensitivityBackend`, so either can be
    slotted into ``GeneralSetup.sensitivity_backend`` interchangeably. Each call
    (via the base ``__call__`` -> :meth:`_build_matrix`) returns a fresh
    :class:`CompositeSensitivityMatrix` that sums an :class:`InstrumentNoise`
    component (rebuilt with the walker's Soms_d / Sa_a) and optionally a
    :class:`GalacticForeground` component (when ``galfor_params`` is supplied),
    plus any extra stationary components passed at construction.

    Args:
        settings: Domain settings the matrix is evaluated on (FD, WDM, ...).
        tdi_generation: 1 (TDI 1.5) or 2 (TDI 2.0).
        model_name: Name to record on the constructed :class:`LISAModel`.
        instrument_fill_nans: ``fill_nans`` value forwarded to
            :class:`InstrumentNoise`. Defaults to ``0.0`` so the WDM fold of
            the FD ``f=0`` divergence doesn't leave NaNs in the matrix; the
            zeroed cells get filtered by :func:`noise_likelihood_term`'s
            ``detC`` mask downstream.
        galfor_stochastic_fn: Stochastic-model class used for the optional
            :class:`GalacticForeground` component (only used when the caller
            supplies ``galfor_params``).
        galfor_modulation: Per-element time modulation forwarded to the
            :class:`GalacticForeground` component: ``None`` (stationary
            isotropic limit), a ``(nch, nch)`` constant matrix, a
            ``(nch, nch, Ntime)`` array, or a callable ``t_arr -> (nch, nch,
            Ntime)`` (e.g. :class:`GalForTimeModulation`), evaluated lazily on
            the domain's active time grid.
        sgwb_stochastic_fn: SGWB spectral-template class or stock name used for
            the optional :class:`SGWB` component (only used when the caller
            supplies ``sgwb_params``).
        instrument_component_kwargs: Extra constructor arguments for
            ``instrument_component_cls``, merged into the per-walker rebuild.
            Use for instrument models needing more than the noise levels — e.g.
            ``dict(ltts=...)`` for :class:`UnequalArmInstrumentNoise`. Keep the
            contents plain data: this rides along in the settings tree, which is
            deepcopied and pickled.
        extra_components: Additional :class:`NoiseComponent` instances added
            to every constructed matrix — e.g. a stationary SGWB. These are
            held by reference so they're built once and reused.
    """

    def __init__(
        self,
        settings: DomainSettingsBase,
        *,
        tdi_generation: int = 2,
        model_name: str = "sangria",
        instrument_fill_nans: float = 0.0,
        galfor_stochastic_fn=HyperbolicTangentGalacticForeground,
        galfor_modulation: Optional[object] = None,
        sgwb_stochastic_fn="PowerLawSGWB",
        instrument_component_cls=None,
        instrument_model_cls=None,
        instrument_component_kwargs: Optional[dict] = None,
        extra_components: Optional[Sequence[NoiseComponent]] = None,
        force_backend: Optional[str] = None,
        cache_instrument_basis: bool = True,
    ):
        SensitivityBackendBase.__init__(
            self, settings, tdi_generation=tdi_generation, force_backend=force_backend
        )
        self.model_name = model_name
        self.instrument_fill_nans = instrument_fill_nans
        # Two unit-level instrument covariances, reused across the per-walker
        # rebuilds this backend exists to serve (see
        # :meth:`InstrumentNoise.base_covariance`). Backend-owned rather than
        # component-owned because ``_build_matrix`` builds a fresh component
        # per call; a per-component cache would never be hit. Purely derived
        # state -- dropped on pickle, rebuilt on demand.
        self._instrument_basis_cache = {} if cache_instrument_basis else None
        # Swappable instrument-noise model (defaults preserve current behavior).
        self.instrument_component_cls = instrument_component_cls or InstrumentNoise
        self.instrument_model_cls = instrument_model_cls or lisa_models.LISAModel
        # Extra constructor arguments for the instrument component, for models
        # that need more than the levels -- e.g.
        # :class:`UnequalArmInstrumentNoise` needs ``ltts=``. Kept as plain data
        # (the LTT array, not the orbits object) so the backend still survives
        # the settings-tree deepcopy / pickle round trip.
        self.instrument_component_kwargs = dict(instrument_component_kwargs or {})
        self.galfor_stochastic_fn = galfor_stochastic_fn
        self.galfor_modulation = galfor_modulation
        self.sgwb_stochastic_fn = sgwb_stochastic_fn
        self.extra_components = list(extra_components) if extra_components else []
        # ``LISAModel.lisanoises`` only reads Soms_d / Sa_a — the orbits field
        # is just a carrier here, so one shared instance is fine.
        self._orbits = lisa_models.DefaultOrbits()

    def __getstate__(self):
        """Drop the derived instrument-basis cache from pickles / deepcopies.

        Sprint deepcopy/pickle rule: this backend can ride along in a settings
        tree that gets deepcopied, and the cache holds two full dense
        covariances (tens of MB on a production grid). It is purely derived
        state, so it is rebuilt on first use in the copy.
        """
        state = dict(self.__dict__)
        if state.get("_instrument_basis_cache") is not None:
            state["_instrument_basis_cache"] = {}
        return state

    def _build_matrix(
        self, name: str, params, galfor_params=None, sgwb_params=None
    ) -> CompositeSensitivityMatrix:
        """Build a per-walker :class:`CompositeSensitivityMatrix`.

        Backend hook (see :meth:`SensitivityBackendBase.__call__`, which applies
        the optional ``transform_fn`` before this is called).

        Args:
            name: Identifier (e.g. ``"walker_3"``) recorded on the LISAModel.
            params: ``[Soms_d, Sa_a]`` in linear (square-root) units (physical
                basis), matching :class:`XYZSensitivityBackend`. ``None`` skips
                the parametric instrument component entirely — the instrument
                noise must then come from ``extra_components`` (fixed-PSD runs
                driven by e.g. a :class:`MojitoNoiseEstimates` table).
            galfor_params: Optional galactic-foreground parameters. When given, a
                :class:`GalacticForeground` component is added (with the backend's
                ``galfor_modulation``).
            sgwb_params: Optional SGWB spectral-template parameters. When given,
                an :class:`SGWB` component is added.

        Returns:
            A freshly built :class:`CompositeSensitivityMatrix`.
        """
        components: list[NoiseComponent] = []
        if params is not None:
            Soms_d = float(params[0])
            Sa_a = float(params[1])
            model = self.instrument_model_cls(
                Soms_d ** 2, Sa_a ** 2, self._orbits, f"{self.model_name}:{name}"
            )
            component_kwargs = dict(
                tdi_generation=self.tdi_generation,
                model=model,
                fill_nans=self.instrument_fill_nans,
                **self.instrument_component_kwargs,
            )
            if self._instrument_basis_cache is not None and issubclass(
                self.instrument_component_cls, InstrumentNoise
            ):
                component_kwargs["basis_cache"] = self._instrument_basis_cache
            components.append(self.instrument_component_cls(**component_kwargs))
        if galfor_params is not None:
            components.append(
                GalacticForeground(
                    foreground_params=np.asarray(galfor_params, dtype=float),
                    modulation=self.galfor_modulation,
                    tdi_generation=self.tdi_generation,
                    stochastic_fn=self.galfor_stochastic_fn,
                )
            )
        if sgwb_params is not None:
            components.append(
                SGWB(
                    sgwb_params=np.asarray(sgwb_params, dtype=float),
                    stochastic_fn=self.sgwb_stochastic_fn,
                    tdi_generation=self.tdi_generation,
                )
            )
        components.extend(self.extra_components)
        if not components:
            raise ValueError(
                "psd_params=None with no galfor/sgwb params and no "
                "extra_components — the backend has no noise component to "
                "build. Pass psd_params or construct the backend with "
                "extra_components (e.g. a MojitoNoiseEstimates table)."
            )
        _tmp = CompositeSensitivityMatrix(self.basis_settings, components)
        return _tmp
