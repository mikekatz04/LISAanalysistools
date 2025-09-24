from __future__ import annotations
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List, Dict

import math
import numpy as np
from scipy import interpolate

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import detector as lisa_models
from .utils.utility import AET
from .utils.constants import *


class StochasticContribution(ABC):
    """Base Class for Stochastic Contributions to the PSD."""

    ndim = None
    added_stochastic_list = []

    @classmethod
    def _check_ndim(cls, params: np.ndarray | list) -> None:
        """Check the dimensionality of the parameters matches the model.

        Args:
            params: Parameters for stochastic model.

        """
        if cls.ndim is None:
            raise ValueError(
                "When subclassing the StochasticContribution class, must set `ndim` as a static attribute."
            )

        if len(params) != cls.ndim:
            raise ValueError("length of parameters is not equivalent to class ndim.")

    @classmethod
    def get_Sh(
        cls, f: float | np.ndarray, *params: np.ndarray | list, **kwargs: Any
    ) -> float | np.ndarray:
        """Calculate the power spectral density of the stochastic contribution.

        Args:
            f: Frequency array.
            *params: Parameters for the stochastic model.
            **kwargs: Keyword arguments for the stochastic model.

        """
        if len(cls.added_stochastic_list) > 0:
            cls._check_ndim(params[0])
        return cls.specific_Sh_function(f, *params, **kwargs)

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, *args: Any, **kwargs: Any
    ) -> float | np.ndarray:
        """Calculate the power spectral density contained in a stochastic signal contribution.

        Args:
            f: Frequency array.
            *args: Any arguments for the function.
            **kwargs: Any keyword arguments for the function.

        Returns:
            Power spectral density contained in stochastic signal.

        """
        raise NotImplementedError


class StochasticContributionContainer:
    """Container for multiple Stochastic Contributions

    Args:
        stochastic_contribution_dict: Dictionary with multiple Stochastic entries.
            Keys are the names and values are of type :class:`StochasticContribution`.

    """

    def __init__(
        self, stochastic_contribution_dict: dict[StochasticContribution]
    ) -> None:
        self.stochastic_contribution_dict = stochastic_contribution_dict

    @property
    def stochastic_contribution_dict(self) -> dict[StochasticContribution]:
        """Stochastic contribution storage."""
        return self._stochastic_contribution_dict

    @stochastic_contribution_dict.setter
    def stochastic_contribution_dict(
        self, stochastic_contribution_dict: dict[StochasticContribution]
    ) -> None:
        """Set stochastic_contribution_dict."""
        assert isinstance(stochastic_contribution_dict, dict)
        for key, value in stochastic_contribution_dict.items():
            if not isinstance(value, StochasticContribution):
                raise ValueError(
                    f"Stochastic model {key} is not of type StochasticContribution."
                )
        self._stochastic_contribution_dict = stochastic_contribution_dict

    def get_Sh(
        self, f: float | np.ndarray, params_dict: dict[tuple], kwargs_dict: dict[dict]
    ) -> np.ndarray:
        """Calculate Sh for stochastic contribution.

        Args:
            f: Frequency array.
            params_dict: Dictionary with keys equivalent to ``self.stochastic_contribution_dict.keys()``.
                Values are the parameters for each associated model.
            kwargs_dict: Dictionary with keys equivalent to ``self.stochastic_contribution_dict.keys()``.
                Values are the keyword argument dicts for each associated model.

        Returns:
            Stochastic contribution.

        """
        Sh_out = np.zeros_like(f)
        for key in params_dict:
            stochastic_contrib = self.stochastic_contribution_dict[key]
            Sh_out += stochastic_contrib.get_Sh(
                f, params_dict[key], **(kwargs_dict.get(key, {}))
            )
        return Sh_out

    def __setitem__(self, key: str | int | tuple, val: StochasticContribution) -> None:
        """Set an item by directly indexing the class object."""
        self.stochastic_contribution_dict[key] = val

    def __getitem__(self, key: str | int | tuple) -> StochasticContribution:
        """Get item directly from dictionary."""
        return self.stochastic_contribution_dict[key]


class HyperbolicTangentGalacticForeground(StochasticContribution):
    """Hyperbolic Tangent-based foreground fitting function."""

    ndim = 5

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, amp: float, fk: float, alpha: float, s1: float, s2: float
    ) -> float | np.ndarray:
        """Hyperbolic tangent model 1 for the Galaxy foreground noise

        This model for the PSD contribution from the Galactic foreground noise is given by

        .. math::

            S_\\text{gal} = \\frac{A_\\text{gal}}{2}e^{-s_1f^\\alpha}f^{-7/3}\\left[ 1 + \\tanh{\\left(-s_2 (f - f_k)\\right)} \\right],

        where :math:`A_\\text{gal}` is the amplitude of the stochastic signal, :math:`f_k` is the knee frequency at which a bend occurs,
        math:`\\alpha` is a power law parameter, :math:`s_1` is a slope parameter below the knee,
        and :math:`s_2` is a slope parameter after the knee.:

        Args:
            f: Frequency array.
            amp: Amplitude parameter for the Galaxy.
            fk: Knee frequency in Hz.
            alpha: Power law parameter.
            s1: Slope parameter below knee.
            s2: Slope parameter above knee.

        Returns:
            PSD of the Galaxy foreground noise

        """
        Sgal = (
            amp
            * np.exp(-(f**alpha) * s1)
            * (f ** (-7.0 / 3.0))
            * 0.5
            * (1.0 + np.tanh(-(f - fk) * s2))
        )

        return Sgal


class FittedHyperbolicTangentGalacticForeground(HyperbolicTangentGalacticForeground):
    # TODO: need to verify this is still working
    ndim = 1
    amp = 3.26651613e-44
    alpha = 1.18300266e00
    # Tobs should be in sec.
    day = 86400.0
    month = day * 30.5
    year = 365.25 * 24.0 * 3600.0  # hard coded for initial fits

    Xobs = [
        1.0 * day,
        3.0 * month,
        6.0 * month,
        1.0 * year,
        2.0 * year,
        4.0 * year,
        10.0 * year,
    ]
    knee = [
        1.15120924e-02,
        4.01884128e-03,
        3.47302482e-03,
        2.77606177e-03,
        2.41178384e-03,
        2.09278117e-03,
        1.57362626e-03,
    ]
    Slope1 = [
        9.41315118e02,
        1.36887568e03,
        1.68729474e03,
        1.76327234e03,
        2.32678814e03,
        3.01430978e03,
        3.74970124e03,
    ]

    Slope2 = [
        1.03239773e02,
        1.03351646e03,
        1.62204855e03,
        1.68631844e03,
        2.06821665e03,
        2.95774596e03,
        3.15199454e03,
    ]
    Tmax = 10 * YRSID_SI

    @classmethod
    def specific_Sh_function(
        cls, f: float | np.ndarray, Tobs: float
    ) -> float | np.ndarray:
        """Fitted hyperbolic tangent model 1 for the Galaxy foreground noise.

        This class fits the parameters for :class:`HyperbolicTangentGalacticForeground`
        using analytic estimates from (# TODO). The fit is a function of time, so the user
        inputs ``Tobs``.

        # Sgal_1d = 2.2e-44*np.exp(-(fr**1.2)*0.9e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-1.4e-2)*0.7e2))
        # Sgal_3m = 2.2e-44*np.exp(-(fr**1.2)*1.7e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-4.8e-3)*5.4e2))
        # Sgal_1y = 2.2e-44*np.exp(-(fr**1.2)*2.2e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-3.1e-3)*1.3e3))
        # Sgal_2y = 2.2e-44*np.exp(-(fr**1.2)*2.2e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-2.3e-3)*1.8e3))
        # Sgal_4y = 2.2e-44*np.exp(-(fr**1.2)*2.9e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-2.0e-3)*1.9e3))

        Args:
            f: Frequency array.
            Tobs: Observation time in seconds.

        Returns:
            PSD of the Galaxy foreground noise

        """

        if Tobs > cls.Tmax:
            raise ValueError(
                "Tobs is greater than the maximum allowable fit which is 10 years."
            )

        # Interpolate
        tck1 = interpolate.splrep(cls.Xobs, cls.Slope1, s=0, k=1)
        tck2 = interpolate.splrep(cls.Xobs, cls.knee, s=0, k=1)
        tck3 = interpolate.splrep(cls.Xobs, cls.Slope2, s=0, k=1)
        s1 = interpolate.splev(Tobs, tck1, der=0).item()
        fk = interpolate.splev(Tobs, tck2, der=0).item()
        s2 = interpolate.splev(Tobs, tck3, der=0).item()

        return HyperbolicTangentGalacticForeground.specific_Sh_function(
            f, cls.amp, fk, cls.alpha, s1, s2
        )


__stock_gb_stochastic_options__ = [
    "HyperbolicTangentGalacticForeground",
    "FittedHyperbolicTangentGalacticForeground",
]


def get_stock_gb_stochastic_options() -> List[StochasticContribution]:
    """Get stock options for stochastic contributions.

    Returns:
        List of stock stochastic options.

    """
    return __stock_gb_stochastic_options__


def get_default_stochastic_from_str(stochastic: str) -> StochasticContribution:
    """Return a LISA stochastic from a ``str`` input.

    Args:
        stochastic: Stochastic contribution indicated with a ``str``.

    Returns:
        Stochastic contribution associated to that ``str``.

    """
    if stochastic not in __stock_gb_stochastic_options__:
        raise ValueError(
            "Requested string stochastic is not available. See lisatools.stochastic documentation."
        )
    return globals()[stochastic]


def check_stochastic(stochastic: Any) -> StochasticContribution:
    """Check input stochastic contribution.

    Args:
        stochastic: Stochastic contribution to check.

    Returns:
        Stochastic contribution checked. Adjusted from ``str`` if ``str`` input.

    """
    if isinstance(stochastic, str):
        stochastic = get_default_stochastic_from_str(stochastic)

    if not issubclass(stochastic, StochasticContribution):
        raise ValueError("stochastic argument not given correctly.")

    return stochastic
