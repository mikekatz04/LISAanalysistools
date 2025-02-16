from __future__ import annotations
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List
from copy import deepcopy
import os

import math
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import detector as lisa_models
from .utils.utility import AET, get_array_module
from .utils.constants import *
from .stochastic import (
    StochasticContribution,
    FittedHyperbolicTangentGalacticForeground,
    check_stochastic,
)

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
        Spm: float | np.ndarray,
        Sop: float | np.ndarray,
        **kwargs: dict,
    ) -> float | np.ndarray:
        """Transform from the base sensitivity functions to the TDI PSDs.

        Args:
            f: Frequency array.
            Spm: Acceleration term.
            Sop: OMS term.
            **kwargs: For interoperability.

        Returns:
            Transformed TDI PSD values.

        """
        raise NotImplementedError

    @classmethod
    def get_Sn(
        cls,
        f: float | np.ndarray,
        model: Optional[lisa_models.LISAModel | str] = lisa_models.scirdv1,
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
            assert hasattr(model, "Soms_d") and hasattr(model, "Sa_a")

            # get noise values
            Spm, Sop = model.lisanoises(f)

            # transform as desired for TDI combination
            Sout = cls.transform(f, Spm, Sop, **kwargs)

        # will add zero if ignored
        stochastic_contribution = cls.stochastic_transform(
            f, cls.get_stochastic_contribution(f, **kwargs), **kwargs
        )

        Sout += stochastic_contribution
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

        Args:
            f: Frequency array.
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
            (stochastic_params != () and stochastic_params is not None)
            or (stochastic_kwargs != {} and stochastic_kwargs is not None)
            or stochastic_function is not None
        ):
            if stochastic_function is None:
                stochastic_function = FittedHyperbolicTangentGalacticForeground

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
    def transform(
        f: float | np.ndarray,
        Spm: float | np.ndarray,
        Sop: float | np.ndarray,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        x = 2.0 * np.pi * lisaLT * f
        return 16.0 * np.sin(x) ** 2 * (2.0 * (1.0 + np.cos(x) ** 2) * Spm + Sop)

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
    def transform(
        f: float | np.ndarray,
        Spm: float | np.ndarray,
        Sop: float | np.ndarray,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        x = 2.0 * np.pi * lisaLT * f
        ## TODO Check the acceleration noise term
        return -4.0 * np.sin(2 * x) * np.sin(x) * (Sop + 4.0 * Spm)

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
    def transform(
        f: float | np.ndarray,
        Spm: float | np.ndarray,
        Sop: float | np.ndarray,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the XYZ TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        x = 2.0 * np.pi * lisaLT * f
        ## TODO Check the acceleration noise term
        return (64.0 * np.sin(x) ** 2 * np.sin(2 * x) ** 2 * Sop) + (
            256.0 * (3 + np.cos(2 * x)) * np.cos(x) ** 2 * np.sin(x) ** 4 * Spm
        )

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


class A1TDISens(Sensitivity):
    channel: str = "A"

    @staticmethod
    def transform(
        f: float | np.ndarray,
        Spm: float | np.ndarray,
        Sop: float | np.ndarray,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the A,E TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        x = 2.0 * np.pi * lisaLT * f
        Sa = (
            8.0
            * np.sin(x) ** 2
            * (
                2.0 * Spm * (3.0 + 2.0 * np.cos(x) + np.cos(2 * x))
                + Sop * (2.0 + np.cos(x))
            )
        )

        return Sa

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
        Spm: float | np.ndarray,
        Sop: float | np.ndarray,
        **kwargs: dict,
    ) -> float | np.ndarray:
        __doc__ = (
            "Transform from the base sensitivity functions to the T TDI PSDs.\n\n"
            + Sensitivity.transform.__doc__.split("PSDs.\n\n")[-1]
        )

        x = 2.0 * np.pi * lisaLT * f
        return (
            16.0 * Sop * (1.0 - np.cos(x)) * np.sin(x) ** 2
            + 128.0 * Spm * np.sin(x) ** 2 * np.sin(0.5 * x) ** 4
        )

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


class LISASens(Sensitivity):
    @classmethod
    def get_Sn(
        cls,
        f: float | np.ndarray,
        model: Optional[lisa_models.LISAModel | str] = lisa_models.scirdv1,
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
        assert hasattr(model, "Soms_d") and hasattr(model, "Sa_a")

        # get noise values
        Sa_d, Sop = model.lisanoises(f, unit="displacement")

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


class SensitivityMatrix:
    """Container to hold sensitivity information.

    Args:
        f: Frequency array.
        sens_mat: Input sensitivity list. The shape of the nested lists should represent the shape of the
            desired matrix. Each entry in the list must be an array, :class:`Sensitivity`-derived object,
            or a string corresponding to the :class:`Sensitivity` object.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(
        self,
        f: np.ndarray,
        sens_mat: (
            List[List[np.ndarray | Sensitivity]]
            | List[np.ndarray | Sensitivity]
            | np.ndarray
            | Sensitivity
        ),
        *sens_args: tuple,
        **sens_kwargs: dict,
    ) -> None:
        self.frequency_arr = f
        self.data_length = len(self.frequency_arr)
        self.sens_args = sens_args
        self.sens_kwargs = sens_kwargs
        self.sens_mat = sens_mat

    @property
    def frequency_arr(self) -> np.ndarray:
        return self._frequency_arr

    @frequency_arr.setter
    def frequency_arr(self, frequency_arr: np.ndarray) -> None:
        assert frequency_arr.dtype == np.float64 or frequency_arr.dtype == float
        assert frequency_arr.ndim == 1
        self._frequency_arr = frequency_arr

    def update_frequency_arr(self, frequency_arr: np.ndarray) -> None:
        """Update class with new frequency array.

        Args:
            frequency_arr: Frequency array.

        """
        self.frequency_arr = frequency_arr
        self.sens_mat = self.sens_mat_input

    def update_model(self, model: lisa_models.LISAModel) -> None:
        """Update class with new sensitivity model.

        Args:
            model: Noise model. Object of type :class:`lisa_models.LISAModel`. It can also be a string corresponding to one of the stock models.

        """
        self.sens_kwargs["model"] = model
        self.sens_mat = self.sens_mat_input

    def update_stochastic(self, **kwargs: dict) -> None:
        """Update class with new stochastic function.

        Args:
            **kwargs: Keyword arguments update for :func:`lisatools.sensitivity.Sensitivity.get_stochastic_contribution`.
                This operation will combine the new and old kwarg dictionaries, updating any
                old information with any added corresponding new information. **Note**: any old information
                that is not updated will remain in place.

        """
        self.sens_kwargs = {**self.sens_kwargs, **kwargs}
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
        self.sens_mat_input = deepcopy(sens_mat)
        self._sens_mat = np.asarray(sens_mat, dtype=object)

        # not an
        new_out = np.full(len(self._sens_mat.flatten()), None, dtype=object)
        self.return_shape = self._sens_mat.shape
        for i in range(len(self._sens_mat.flatten())):
            current_sens = self._sens_mat.flatten()[i]
            if hasattr(current_sens, "get_Sn") or isinstance(current_sens, str):
                new_out[i] = get_sensitivity(
                    self.frequency_arr,
                    *self.sens_args,
                    sens_fn=current_sens,
                    **self.sens_kwargs,
                )

            elif isinstance(current_sens, np.ndarray) or isinstance(
                current_sens, cp.ndarray
            ):
                new_out[i] = current_sens
            else:
                raise ValueError

        xp = get_array_module(new_out[0])
        # setup in array form
        self._sens_mat = xp.asarray(list(new_out), dtype=float).reshape(
            self.return_shape + (-1,)
        )

        xp = get_array_module(self.sens_mat)

        # setup detC
        """Determinant and inverse of TDI matrix."""
        if self.sens_mat.ndim < 3:
            self.detC = xp.prod(self.sens_mat, axis=0)
            self.invC = 1 / self.sens_mat

        else:
            self.detC = xp.linalg.det(self.sens_mat.transpose(2, 0, 1))
            invC = xp.zeros_like(self.sens_mat.transpose(2, 0, 1))
            invC[self.detC != 0.0] = xp.linalg.inv(
                self.sens_mat.transpose(2, 0, 1)[self.detC != 0.0]
            )
            invC[self.detC == 0.0] = 1e-100
            self.invC = invC.transpose(1, 2, 0)

    def __getitem__(self, index: Any) -> np.ndarray:
        """Indexing the class indexes the array."""
        return self.sens_mat[index]

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


class XYZ1SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for XYZ (TDI 1)

    This is 3x3 symmetric matrix.

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, f: np.ndarray, **sens_kwargs: dict) -> None:
        sens_mat = [
            [X1TDISens, XY1TDISens, ZX1TDISens],
            [XY1TDISens, Y1TDISens, YZ1TDISens],
            [ZX1TDISens, YZ1TDISens, Z1TDISens],
        ]
        super().__init__(f, sens_mat, **sens_kwargs)


class AET1SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for AET (TDI 1)

    This is just an array because no cross-terms.

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, f: np.ndarray, **sens_kwargs: dict) -> None:
        sens_mat = [A1TDISens, E1TDISens, T1TDISens]
        super().__init__(f, sens_mat, **sens_kwargs)


class AE1SensitivityMatrix(SensitivityMatrix):
    """Default sensitivity matrix for AE (no T) (TDI 1)

    Args:
        f: Frequency array.
        **sens_kwargs: Keyword arguments to pass to :func:`Sensitivity.get_Sn`.

    """

    def __init__(self, f: np.ndarray, **sens_kwargs: dict) -> None:
        sens_mat = [A1TDISens, E1TDISens]
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
    f: float | np.ndarray,
    *args: tuple,
    sens_fn: Optional[Sensitivity | str] = LISASens,
    return_type="PSD",
    fill_nans: float = 1e10,
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

    PSD = sensitivity.get_Sn(f, *args, **kwargs)

    if fill_nans is not None:
        assert isinstance(fill_nans, float)
        PSD[np.isnan(PSD)] = fill_nans

    if return_type == "PSD":
        return PSD

    elif return_type == "ASD":
        return PSD ** (1 / 2)

    elif return_type == "char_strain":
        return (f * PSD) ** (1 / 2)

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
