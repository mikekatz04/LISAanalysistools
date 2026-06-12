"""Stochastic foreground contributions to the LISA noise budget."""

from __future__ import annotations

import math
import warnings
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate
from scipy.special import erfc

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import detector as lisa_models
from .utils.constants import *
from .utils.utility import AET


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

    def __init__(self, stochastic_contribution_dict: dict[StochasticContribution]) -> None:
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
                raise ValueError(f"Stochastic model {key} is not of type StochasticContribution.")
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
            Sh_out += stochastic_contrib.get_Sh(f, params_dict[key], **(kwargs_dict.get(key, {})))
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
        f: float | np.ndarray, amp: float, fk: float, alpha: float, f_1: float, f_2: float
    ) -> float | np.ndarray:
        """Hyperbolic tangent model 1 for the Galaxy foreground noise

        This model for the PSD contribution from the Galactic foreground noise is given by

        .. math::

            S_\\text{gal} = \\frac{A_\\text{gal}}{2}e^{-\\left(f/f_1\\right)^\\alpha}f^{-7/3}\\left[ 1 + \\tanh{\\left(-(f - f_k)/f_2\\right)} \\right],

        where :math:`A_\\text{gal}` is the amplitude of the stochastic signal, :math:`f_k` is the knee frequency at which a bend occurs,
        math:`\\alpha` is a power law parameter, :math:`f_1` sets the exponential roll-off scale,
        and :math:`f_2` sets the transition width around the knee.

        Args:
            f: Frequency array.
            amp: Amplitude parameter for the Galaxy.
            fk: Knee frequency in Hz.
            alpha: Power law parameter.
            f_1: Exponential scale-frequency parameter.
            f_2: Hyperbolic-tangent transition scale-frequency parameter.

        Returns:
            PSD of the Galaxy foreground noise

        """
        Sgal = (
            amp
            * np.exp(-((f / f_1) ** alpha))
            * (f ** (-7.0 / 3.0))
            * 0.5
            * (1.0 + np.tanh(-(f - fk) / f_2))
        )

        return Sgal


class FittedHyperbolicTangentGalacticForeground(HyperbolicTangentGalacticForeground):
    """Time-dependent fit of the Galactic confusion-foreground PSD.

    Specializes :class:`HyperbolicTangentGalacticForeground` by interpolating
    pre-fit values of the knee frequency and slope parameters as a function of
    observation time ``Tobs``. The amplitude and power-law index are held fixed
    at the values stored in the class attributes ``amp`` and ``alpha``.

    The fit is only valid up to ``Tmax`` (10 years). The single free parameter
    accepted by :meth:`specific_Sh_function` is the observation time in seconds.
    """

    # TODO: need to verify this is still working
    # TODO/DOCS: the time-dependent Galactic-foreground fit has not been re-validated
    # against current data; the original TODO above flags this. Treat the numerical
    # fit coefficients (knee, Slope1, Slope2, amp, alpha) as legacy values pending review.
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
    _Slope1 = [
        9.41315118e02,
        1.36887568e03,
        1.68729474e03,
        1.76327234e03,
        2.32678814e03,
        3.01430978e03,
        3.74970124e03,
    ]

    _Slope2 = [
        1.03239773e02,
        1.03351646e03,
        1.62204855e03,
        1.68631844e03,
        2.06821665e03,
        2.95774596e03,
        3.15199454e03,
    ]
    F1 = [s ** (-1.0 / 1.18300266e00) for s in _Slope1]
    F2 = [1.0 / s for s in _Slope2]
    Tmax = 10 * YRSID_SI

    @classmethod
    def specific_Sh_function(cls, f: float | np.ndarray, Tobs: float) -> float | np.ndarray:
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
            raise ValueError("Tobs is greater than the maximum allowable fit which is 10 years.")

        # Interpolate
        tck1 = interpolate.splrep(cls.Xobs, cls.F1, s=0, k=1)
        tck2 = interpolate.splrep(cls.Xobs, cls.knee, s=0, k=1)
        tck3 = interpolate.splrep(cls.Xobs, cls.F2, s=0, k=1)
        f_1 = interpolate.splev(Tobs, tck1, der=0).item()
        fk = interpolate.splev(Tobs, tck2, der=0).item()
        f_2 = interpolate.splev(Tobs, tck3, der=0).item()

        return HyperbolicTangentGalacticForeground.specific_Sh_function(
            f, cls.amp, fk, cls.alpha, f_1, f_2
        )


# --------------------------------------------------------------------------- #
# Stochastic gravitational-wave background (SGWB) spectral templates.
# --------------------------------------------------------------------------- #

# Hubble constant [1/s]: H0 = 70 km/s/Mpc * 3.24078e-20 Mpc/km
_SGWB_H0_SI = 70.0 * 3.24078e-20

# common cosmology units to PSD factor
SGWB_HSCALE = 3.0 * _SGWB_H0_SI**2 / (4.0 * np.pi**2)

# c_g * Omega_{r,0} (radiation d.o.f. factor times present-day radiation energy density)
# change the denominator if you change H0!
SGWB_CGOR0 = 1.6e-5 / (0.7 * 0.7)

class PowerLawSGWB(StochasticContribution):
    """Power-law SGWB spectral template

    .. math::

        S_\\text{gw}(f) =  A\\,
                          \\left(\\frac{f}{f_\\text{ref}}\\right)^{\\alpha},

    with :math:`A = 10^{\\log_{10}A}` and :math:`f_\\text{ref}` =
    :data:`SGWB_FREF`.
    """

    ndim = 2

    @staticmethod
    def specific_Sh_function(
            f: float | np.ndarray, log10_A: float, alpha: float, fref: float = 25.0 #Hz
    ) -> float | np.ndarray:
        """Power-law SGWB.

        Args:
            f: Frequency array [Hz].
            log10_A: Base-10 log of the amplitude at ``SGWB_FREF``.
            alpha: Power-law spectral index.

        Returns:
            GW spectral density ``Sgw(f)`` (pre-response).
        """
        A = 10.0**log10_A
        # Sgw ~ 1/f^3 diverges at f=0; return NaN there.
        with np.errstate(divide="ignore", invalid="ignore"):
            prefactor = SGWB_HSCALE / (f * f * f)
            Sgw = prefactor * A * (f / fref) ** alpha
        return np.where(np.asarray(f) > 0.0, Sgw, np.nan)


class LogNormalSGWB(StochasticContribution):
    """Log-normal (scalar-induced) SGWB template

    Pi & Sasaki, JCAP 2020 (arXiv:2005.12306), wide-:math:`\\Delta` limit,
    eq. (3.29). For :math:`D \\geq 9` the closed form suffers catastrophic
    cancellation, so the asymptotic numerical value (eq. 3.33) is used instead.

    Parameters are ``(log10_A, log10_fstar, log10_D)``.
    """

    ndim = 3

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, log10_A: float, log10_fstar: float, log10_D: float
    ) -> float | np.ndarray:
        """Log-normal SGWB (see class docstring for the reference).

        Args:
            f: Frequency array [Hz].
            log10_A: Base-10 log of the (scalar) amplitude.
            log10_fstar: Base-10 log of the peak frequency [Hz].
            log10_D: Base-10 log of the (dimensionless) width :math:`D`.

        Returns:
            GW spectral density ``Sgw(f)`` (pre-response).
        """
        A = 10.0**log10_A
        fstar = 10.0**log10_fstar
        D = 10.0**log10_D
        # Sgw ~ 1/f^3 (and log f) diverge at f=0; return NaN there (GLASS zeroes f=0).
        f_pos = np.asarray(f) > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            prefactor = SGWB_HSCALE / (f * f * f)
            ft = f / fstar
            if D < 9.0:
                logft = np.log(ft)
                logK = logft + 1.5 * D * D
                sqrtpi = np.sqrt(np.pi)
                half_log32 = 0.5 * np.log(1.5)
                D2 = D * D
                t1 = (
                    4.0 / 5.0 / sqrtpi
                    * ft**3
                    * np.exp(9.0 * D2 / 4.0) / D
                    * (
                        (logK * logK + 0.5 * D2) * erfc((logK + half_log32) / D)
                        - D / sqrtpi
                        * np.exp(-((logK + half_log32) ** 2) / D2)
                        * (logK - half_log32)
                    )
                )
                t2 = (
                    0.0659 / D2
                    * ft**2
                    * np.exp(D2)
                    * np.exp(-((logft + D2 - 0.5 * np.log(4.0 / 3.0)) ** 2) / D2)
                )
                t3 = (
                    (1.0 / 3.0) * np.sqrt(2.0) / sqrtpi
                    * ft ** (-4)
                    * np.exp(8.0 * D2) / D
                    * np.exp(-(logft * logft) / (2.0 * D2))
                    * erfc((4.0 * D2 - logft + np.log(4.0)) / (D * np.sqrt(2.0)))
                )
                Sgw = prefactor * SGWB_CGOR0 * A * A * (t1 + t2 + t3)
            else:
                # Numerical asymptote from Pi & Sasaki eq. (3.33).
                Sgw = prefactor * SGWB_CGOR0 * A * A * 0.783 / 1e3
        return np.where(f_pos, Sgw, np.nan)


class PhaseTransitionSGWB(StochasticContribution):
    """Phase-transition SGWB template.

    Parameters are ``(rb, b, log10_Ap, log10_fp)`` with a double-broken
    power-law shape.

    See https://arxiv.org/abs/2209.13277

    """

    ndim = 4

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, rb: float, b: float, log10_Ap: float, log10_fp: float
    ) -> float | np.ndarray:
        """Phase-transition SGWB.

        Args:
            f: Frequency array [Hz].
            rb: Low-/high-frequency slope ratio parameter.
            b: Spectral shape parameter.
            log10_Ap: Base-10 log of the peak amplitude.
            log10_fp: Base-10 log of the peak frequency [Hz].

        Returns:
            GW spectral density ``Sgw(f)`` (pre-response).
        """
        Ap = 10.0**log10_Ap
        fp = 10.0**log10_fp
        rb4 = rb * rb * rb * rb
        m = (9.0 * rb4 + b) / (rb4 + 1.0)
        # Sgw ~ 1/f^3 diverges at f=0 (and M is ill-defined there); return NaN
        # at f=0 (GLASS zeroes f=0).
        with np.errstate(divide="ignore", invalid="ignore"):
            s = f / fp
            s4 = s * s * s * s
            s9 = s4 * s4 * s
            M = (
                s9
                * ((1.0 + rb4) / (rb4 + s4)) ** ((9.0 - b) / 4.0)
                * ((b + 4.0) / (b + 4.0 - m + m * s * s)) ** ((b + 4.0) / 2.0)
            )
            prefactor = SGWB_HSCALE / (f * f * f)
            Sgw = Ap * M * prefactor
        return np.where(np.asarray(f) > 0.0, Sgw, np.nan)


__stock_sgwb_options__ = [
    "PowerLawSGWB",
    "LogNormalSGWB",
    "PhaseTransitionSGWB",
]


def get_stock_sgwb_options() -> List[StochasticContribution]:
    """Get stock options for SGWB spectral templates.

    Returns:
        List of stock SGWB template names.

    """
    return __stock_sgwb_options__


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
    if stochastic not in (__stock_gb_stochastic_options__ + __stock_sgwb_options__):
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
