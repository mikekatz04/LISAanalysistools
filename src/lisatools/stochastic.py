"""Stochastic contributions to the LISA noise budget: the Galactic foreground and SGWB templates.

Each model is a :class:`StochasticContribution` subclass used as a class (never instantiated by
the sensitivity code): :meth:`StochasticContribution.get_Sh` returns the one-sided strain PSD
of the stochastic signal at frequency ``f`` *before* the LISA/TDI response, which the
sensitivity layer (:meth:`lisatools.sensitivity.Sensitivity.stochastic_transform`, the
:class:`~lisatools.sensitivity.GalacticForeground` and :class:`~lisatools.sensitivity.SGWB`
components) applies afterwards.

Models:

- :class:`HyperbolicTangentGalacticForeground`: five-parameter Galactic confusion foreground.
  This is the foreground model of the production global fit (the ``galfor`` branch, through
  :class:`~lisatools.sensitivity.GalacticForeground`).
- :class:`FittedHyperbolicTangentGalacticForeground`: the same shape with its parameters
  interpolated from a legacy fit as a function of the observation time only.
- :class:`PowerLawSGWB`, :class:`LogNormalSGWB`, :class:`PhaseTransitionSGWB`: SGWB spectral
  templates.

Name lookup: :func:`get_stock_gb_stochastic_options`, :func:`get_stock_sgwb_options`,
:func:`get_default_stochastic_from_str`, and :func:`check_stochastic` (accepts a class or its
name).
"""

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
from .utils.utility import AET, get_array_module


class StochasticContribution(ABC):
    """Base class for stochastic contributions to the PSD.

    Subclasses set the class attribute ``ndim`` and implement :meth:`specific_Sh_function`;
    they are used as classes (``Model.get_Sh(f, *params)``), not instances.
    :func:`check_stochastic` accepts any subclass or the name of a stock one.

    Attributes:
        ndim (int | None): Number of model parameters. Must be set by subclasses.
        added_stochastic_list (list): Hook for composite models. When non-empty,
            :meth:`get_Sh` checks ``len(params[0]) == ndim`` first. No stock model fills it,
            so the check never runs for them.

    Example:
        >>> import numpy as np
        >>> from lisatools.stochastic import StochasticContribution
        >>> class FlatBackground(StochasticContribution):
        ...     ndim = 1
        ...     @staticmethod
        ...     def specific_Sh_function(f, level):
        ...         return level * np.ones_like(f)
        >>> FlatBackground.get_Sh(np.array([1e-3, 2e-3]), 1e-40)
        array([1.e-40, 1.e-40])
    """

    ndim = None
    added_stochastic_list = []

    @classmethod
    def _check_ndim(cls, params: np.ndarray | list) -> None:
        """Check that ``len(params)`` equals the class ``ndim``.

        Args:
            params: Parameters for stochastic model.

        Raises:
            ValueError: ``ndim`` is not set on the class, or the length does not match.
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
        """Evaluate the PSD of the stochastic contribution (before the LISA response).

        Args:
            f: Frequency in Hz (scalar or array).
            *params: Model parameters, forwarded positionally to :meth:`specific_Sh_function`.
            **kwargs: Forwarded to :meth:`specific_Sh_function`.

        Returns:
            float | np.ndarray: One-sided strain PSD (1/Hz), same shape as ``f``.
        """
        if len(cls.added_stochastic_list) > 0:
            cls._check_ndim(params[0])
        return cls.specific_Sh_function(f, *params, **kwargs)

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, *args: Any, **kwargs: Any
    ) -> float | np.ndarray:
        """Model PSD of the stochastic signal; subclasses override this.

        Args:
            f: Frequency in Hz (scalar or array).
            *args: Model parameters.
            **kwargs: Model keyword arguments.

        Returns:
            float | np.ndarray: One-sided strain PSD (1/Hz), same shape as ``f``.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError


class StochasticContributionContainer:
    """Named collection of stochastic contributions whose PSDs are summed.

    Supports ``container[key]`` get/set on the underlying dict. Not used by the
    sensitivity classes, which take a single model class.

    Args:
        stochastic_contribution_dict: Mapping name -> :class:`StochasticContribution`
            *instance* (the setter checks ``isinstance``, so classes are rejected).

    Raises:
        AssertionError: The input is not a ``dict``.
        ValueError: A value is not a :class:`StochasticContribution` instance.

    Warning:
        :meth:`get_Sh` passes each entry's parameters as ONE positional argument
        (``model.get_Sh(f, params_dict[key])``), so it only works with models whose
        :meth:`~StochasticContribution.specific_Sh_function` takes a single parameter
        (e.g. :class:`FittedHyperbolicTangentGalacticForeground` given a scalar ``Tobs``);
        the multi-parameter stock models take unpacked parameters and raise
        :class:`TypeError`.
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
        """Sum the PSDs of the contributions named in ``params_dict``.

        Args:
            f: Frequency array in Hz (numpy or cupy; the output follows it).
            params_dict: Name -> parameters of that model. Only the keys present here are
                evaluated; each value is passed as a single positional argument.
            kwargs_dict: Name -> keyword dict for that model. Missing keys mean no keywords.

        Returns:
            np.ndarray: Summed one-sided strain PSD (1/Hz), same shape as ``f``.
        """
        xp = get_array_module(f)
        Sh_out = xp.zeros_like(f)
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
    r"""Five-parameter hyperbolic-tangent model of the Galactic confusion foreground.

    .. math::

        S_\text{gal}(f) = \frac{A_\text{gal}}{2}\, e^{-\left(f/f_1\right)^\alpha} f^{-7/3}
            \left[ 1 + \tanh{\left(-(f - f_k)/f_2\right)} \right].

    This is the foreground shape of the production global fit: the ``galfor`` branch samples
    these five parameters, in the order ``(amp, fk, alpha, f_1, f_2)`` (optionally with
    ``log10`` of all but ``alpha``, see
    :func:`~lisatools.globalfit.stock.erebor.noise.make_galfor_log_transform_container`), and
    :class:`~lisatools.sensitivity.GalacticForeground` applies the TDI response
    (``ndim = 5``).

    Example:
        >>> import numpy as np
        >>> from lisatools.stochastic import HyperbolicTangentGalacticForeground as HTGF
        >>> f = np.array([1e-3, 3e-3])
        >>> Sgal = HTGF.get_Sh(f, 1.15e-44, 2.5e-3, 1.2, 1.3e-3, 3e-4)
        >>> Sgal.shape
        (2,)
    """

    ndim = 5

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, amp: float, fk: float, alpha: float, f_1: float, f_2: float
    ) -> float | np.ndarray:
        r"""Hyperbolic-tangent PSD of the Galactic foreground.

        .. math::

            S_\text{gal} = \frac{A_\text{gal}}{2}e^{-\left(f/f_1\right)^\alpha}f^{-7/3}
                \left[ 1 + \tanh{\left(-(f - f_k)/f_2\right)} \right],

        where :math:`A_\text{gal}` is the amplitude, :math:`f_k` the knee frequency at
        which the spectrum bends down, :math:`\alpha` the exponent of the exponential
        roll-off, :math:`f_1` its frequency scale, and :math:`f_2` the width of the
        transition around the knee.

        Args:
            f: Frequency in Hz (numpy or cupy array, or scalar; the output follows it).
            amp: Amplitude :math:`A_\text{gal}` (strain PSD units times
                :math:`\text{Hz}^{7/3}`).
            fk: Knee frequency in Hz.
            alpha: Exponent of the exponential roll-off.
            f_1: Roll-off frequency scale in Hz.
            f_2: Knee transition width in Hz.

        Returns:
            float | np.ndarray: One-sided strain PSD (1/Hz) of the foreground, before the
            LISA response, same shape as ``f``.
        """
        xp = get_array_module(f)
        Sgal = (
            amp
            * xp.exp(-((f / f_1) ** alpha))
            * (f ** (-7.0 / 3.0))
            * 0.5
            * (1.0 + xp.tanh(-(f - fk) / f_2))
        )

        return Sgal


class FittedHyperbolicTangentGalacticForeground(HyperbolicTangentGalacticForeground):
    """Galactic foreground PSD as a function of the observation time only.

    Specializes :class:`HyperbolicTangentGalacticForeground`: the knee ``fk`` and the
    frequency scales ``f_1``, ``f_2`` are linearly interpolated in ``Tobs`` between seven
    tabulated fit nodes (``Xobs``: 1 day, 3, 6 months, 1, 2, 4, 10 years), while ``amp`` and
    ``alpha`` stay at the class values. The single model parameter is ``Tobs`` in seconds
    (``ndim = 1``), up to ``Tmax`` (10 sidereal years).

    It is the default ``stochastic_function`` of
    :meth:`~lisatools.sensitivity.Sensitivity.get_stochastic_contribution` (stochastic
    parameters given without a function); in the global-fit code it supplies the foreground
    of the synthetic injections and the monitor's reference curve. The production fit
    itself samples :class:`HyperbolicTangentGalacticForeground`.

    Attributes:
        amp (float): Fixed amplitude.
        alpha (float): Fixed roll-off exponent.
        Xobs (list[float]): Fit nodes in observation time (s), 365.25-day years.
        knee (list[float]): Knee frequency ``fk`` (Hz) at each node.
        F1 (list[float]): ``f_1`` (Hz) at each node, ``Slope1 ** (-1 / alpha)``.
        F2 (list[float]): ``f_2`` (Hz) at each node, ``1 / Slope2``.
        Tmax (float): Largest accepted ``Tobs`` (s), ``10 * YRSID_SI``.

    Example:
        >>> import numpy as np
        >>> from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground as FHT
        >>> f = np.logspace(-4, -2, 5)
        >>> FHT.get_Sh(f, 3.15e7).shape
        (5,)
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
    # the FIT's own year unit (Julian, 365.25 d), not a physical constant --
    # deliberately not lisaconstants' ASTRONOMICAL_YEAR (the fit coefficients
    # below were derived with it)
    year = 365.25 * 24.0 * 3600.0

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
        """Foreground PSD with ``fk``, ``f_1``, ``f_2`` interpolated at ``Tobs``.

        The three shape parameters are interpolated linearly (``scipy`` ``splrep`` with
        ``k=1``) over the class nodes ``Xobs``, then passed with the fixed ``amp`` and
        ``alpha`` to :meth:`HyperbolicTangentGalacticForeground.specific_Sh_function`. The
        source of the tabulated fit is not recorded in the code. For reference, older
        closed-form versions of the same fit were
        ``Sgal = 2.2e-44 * exp(-(f**1.2) * s1) * f**(-7/3) * 0.5 * (1 + tanh(-(f - fk) * s2))``
        with:

        =======  =======  =======  =======
        Tobs     s1       fk (Hz)  s2
        =======  =======  =======  =======
        1 day    0.9e3    1.4e-2   0.7e2
        3 month  1.7e3    4.8e-3   5.4e2
        1 year   2.2e3    3.1e-3   1.3e3
        2 year   2.2e3    2.3e-3   1.8e3
        4 year   2.9e3    2.0e-3   1.9e3
        =======  =======  =======  =======

        Args:
            f: Frequency in Hz (scalar or array).
            Tobs: Observation time in seconds (a scalar). Values below the first node
                (1 day) are linearly extrapolated.

        Returns:
            float | np.ndarray: One-sided strain PSD (1/Hz) of the foreground, before the
            LISA response, same shape as ``f``.

        Raises:
            ValueError: ``Tobs > Tmax`` (10 sidereal years).
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

# Omega_gw -> strain PSD factor: S_gw(f) = SGWB_HSCALE * Omega_gw(f) / f**3,
# SGWB_HSCALE = 3 H0^2 / (4 pi^2)
SGWB_HSCALE = 3.0 * _SGWB_H0_SI**2 / (4.0 * np.pi**2)

# c_g * Omega_{r,0} (radiation d.o.f. factor times present-day radiation energy density)
# change the denominator if you change H0!
SGWB_CGOR0 = 1.6e-5 / (0.7 * 0.7)

class PowerLawSGWB(StochasticContribution):
    r"""Power-law SGWB spectral template.

    .. math::

        S_\text{gw}(f) = \frac{3 H_0^2}{4\pi^2 f^3}\, \Omega_\text{gw}(f), \qquad
        \Omega_\text{gw}(f) = A \left(\frac{f}{f_\text{ref}}\right)^{\alpha},

    with :math:`A = 10^{\log_{10}A}`, :math:`H_0 = 70` km/s/Mpc (the prefactor
    :math:`3 H_0^2 / (4\pi^2)` is the module constant ``SGWB_HSCALE``), and
    :math:`f_\text{ref}` the ``fref`` keyword of
    :meth:`specific_Sh_function` (default 25 Hz). Parameters are ``(log10_A, alpha)``
    (``ndim = 2``).

    Example:
        >>> import numpy as np
        >>> from lisatools.stochastic import PowerLawSGWB
        >>> Sgw = PowerLawSGWB.get_Sh(np.array([0.0, 1e-3, 1e-2]), -12.0, 2.0 / 3.0)
        >>> bool(np.isnan(Sgw[0])), bool(np.all(Sgw[1:] > 0))
        (True, True)
    """

    ndim = 2

    @staticmethod
    def specific_Sh_function(
            f: float | np.ndarray, log10_A: float, alpha: float, fref: float = 25.0 #Hz
    ) -> float | np.ndarray:
        r"""Power-law SGWB strain PSD (see the class docstring for the formula).

        Works with numpy and cupy ``f`` (the array module follows ``f``).

        Args:
            f: Frequency in Hz (scalar or array).
            log10_A: Base-10 log of the energy-density amplitude :math:`\Omega_\text{gw}`
                at ``fref``.
            alpha: Power-law index of :math:`\Omega_\text{gw}(f)`.
            fref: Reference frequency in Hz. Default ``25.0``.

        Returns:
            float | np.ndarray: One-sided strain PSD ``Sgw(f)`` (1/Hz) before the LISA
            response; ``NaN`` where ``f <= 0``.
        """
        # Backend-generic (numpy / cupy): resolve xp from f. cupy has no
        # ``errstate``; avoid the f=0 divide entirely by masking f<=0 to a
        # dummy 1.0 before the division and NaN-ing those bins afterwards
        # (bit-identical result: Sgw at f>0, NaN at f<=0).
        xp = get_array_module(f)
        A = 10.0**log10_A
        f_arr = xp.asarray(f, dtype=float)
        pos = f_arr > 0.0
        safe_f = xp.where(pos, f_arr, 1.0)
        prefactor = SGWB_HSCALE / (safe_f * safe_f * safe_f)
        Sgw = prefactor * A * (safe_f / fref) ** alpha
        return xp.where(pos, Sgw, xp.nan)


class LogNormalSGWB(StochasticContribution):
    r"""Log-normal (scalar-induced) SGWB template.

    Pi & Sasaki, JCAP 2020 (arXiv:2005.12306), wide-:math:`\Delta` limit, eq. (3.29):
    :math:`\Omega_\text{gw}(f) = c_g \Omega_{r,0} A^2 \,[t_1 + t_2 + t_3](f/f_*, D)`,
    converted to a strain PSD with :math:`S_\text{gw} = 3 H_0^2 \Omega_\text{gw} / (4\pi^2 f^3)`.
    For :math:`D \geq 9` the closed form suffers catastrophic cancellation, so the
    asymptotic numerical value (eq. 3.33), a frequency-independent
    :math:`\Omega_\text{gw} = 0.783 \times 10^{-3}\, c_g \Omega_{r,0} A^2`, is used instead.

    Parameters are ``(log10_A, log10_fstar, log10_D)`` (``ndim = 3``). numpy only.
    """

    ndim = 3

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, log10_A: float, log10_fstar: float, log10_D: float
    ) -> float | np.ndarray:
        """Log-normal SGWB strain PSD (see the class docstring for the reference).

        Args:
            f: Frequency in Hz (numpy scalar or array).
            log10_A: Base-10 log of the scalar (curvature) amplitude :math:`A`; the
                spectrum scales as :math:`A^2`.
            log10_fstar: Base-10 log of the peak frequency :math:`f_*` in Hz.
            log10_D: Base-10 log of the (dimensionless) width :math:`D`.

        Returns:
            float | np.ndarray: One-sided strain PSD ``Sgw(f)`` (1/Hz) before the LISA
            response; ``NaN`` where ``f <= 0``.
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
    r"""Phase-transition SGWB template (double-broken power law).

    With :math:`s = f/f_p`, :math:`m = (9 r_b^4 + b)/(r_b^4 + 1)`:

    .. math::

        \Omega_\text{gw}(f) = A_p\, s^9 \left(\frac{1 + r_b^4}{r_b^4 + s^4}\right)^{(9-b)/4}
            \left(\frac{b + 4}{b + 4 - m + m s^2}\right)^{(b+4)/2},

    and :math:`S_\text{gw} = 3 H_0^2 \Omega_\text{gw} / (4\pi^2 f^3)`. Read off the formula,
    :math:`\Omega_\text{gw}` rises as :math:`f^9` below the first break at
    :math:`f \approx r_b f_p`, goes as :math:`f^b` between the breaks, and falls as
    :math:`f^{-4}` above :math:`f_p`.

    Parameters are ``(rb, b, log10_Ap, log10_fp)`` (``ndim = 4``). numpy only.
    See https://arxiv.org/abs/2209.13277.
    """

    ndim = 4

    @staticmethod
    def specific_Sh_function(
        f: float | np.ndarray, rb: float, b: float, log10_Ap: float, log10_fp: float
    ) -> float | np.ndarray:
        r"""Phase-transition SGWB strain PSD (see the class docstring for the formula).

        Args:
            f: Frequency in Hz (numpy scalar or array).
            rb: Ratio of the lower break frequency to ``fp``.
            b: Spectral slope of :math:`\Omega_\text{gw}` between the two breaks.
            log10_Ap: Base-10 log of the amplitude :math:`A_p` (the value of
                :math:`\Omega_\text{gw}` at ``f = fp``).
            log10_fp: Base-10 log of the upper break (peak) frequency in Hz.

        Returns:
            float | np.ndarray: One-sided strain PSD ``Sgw(f)`` (1/Hz) before the LISA
            response; ``NaN`` where ``f <= 0``.
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
    """Names of the stock SGWB spectral templates.

    Returns:
        list[str]: ``["PowerLawSGWB", "LogNormalSGWB", "PhaseTransitionSGWB"]`` (the module
        list itself, not a copy). Each name resolves with :func:`check_stochastic`.

    Example:
        >>> from lisatools.stochastic import get_stock_sgwb_options
        >>> get_stock_sgwb_options()
        ['PowerLawSGWB', 'LogNormalSGWB', 'PhaseTransitionSGWB']
    """
    return __stock_sgwb_options__


__stock_gb_stochastic_options__ = [
    "HyperbolicTangentGalacticForeground",
    "FittedHyperbolicTangentGalacticForeground",
]


def get_stock_gb_stochastic_options() -> List[StochasticContribution]:
    """Names of the stock Galactic-foreground stochastic models.

    Returns:
        list[str]: The module list itself (not a copy):
        ``HyperbolicTangentGalacticForeground``, ``FittedHyperbolicTangentGalacticForeground``.
        Each name resolves with :func:`check_stochastic`.

    Example:
        >>> from lisatools.stochastic import get_stock_gb_stochastic_options
        >>> get_stock_gb_stochastic_options()
        ['HyperbolicTangentGalacticForeground', 'FittedHyperbolicTangentGalacticForeground']
    """
    return __stock_gb_stochastic_options__


def get_default_stochastic_from_str(stochastic: str) -> StochasticContribution:
    """Return the stock stochastic model class with the given name.

    Args:
        stochastic: Class name, one of :func:`get_stock_gb_stochastic_options` or
            :func:`get_stock_sgwb_options`.

    Returns:
        type[StochasticContribution]: The model class (not an instance).

    Raises:
        ValueError: The name is not a stock option.

    Example:
        >>> from lisatools.stochastic import get_default_stochastic_from_str
        >>> get_default_stochastic_from_str("PowerLawSGWB").ndim
        2
    """
    if stochastic not in (__stock_gb_stochastic_options__ + __stock_sgwb_options__):
        raise ValueError(
            "Requested string stochastic is not available. See lisatools.stochastic documentation."
        )
    return globals()[stochastic]


def check_stochastic(stochastic: Any) -> StochasticContribution:
    """Resolve and validate a stochastic model given as a class or a stock name.

    Called by the sensitivity layer (:class:`~lisatools.sensitivity.GalacticForeground`,
    :class:`~lisatools.sensitivity.SGWB`,
    :meth:`~lisatools.sensitivity.Sensitivity.get_stochastic_contribution`) on every
    ``stochastic_fn`` / ``stochastic_function`` argument.

    Args:
        stochastic: A :class:`StochasticContribution` subclass, or the name of a stock one
            (resolved with :func:`get_default_stochastic_from_str`).

    Returns:
        type[StochasticContribution]: The model class.

    Raises:
        ValueError: A string that is not a stock name, or a class that is not a
            :class:`StochasticContribution` subclass.
        TypeError: ``stochastic`` is neither a string nor a class (from ``issubclass``).

    Example:
        >>> from lisatools.stochastic import check_stochastic
        >>> from lisatools.stochastic import HyperbolicTangentGalacticForeground as HTGF
        >>> check_stochastic("HyperbolicTangentGalacticForeground") is HTGF
        True
    """
    if isinstance(stochastic, str):
        stochastic = get_default_stochastic_from_str(stochastic)

    if not issubclass(stochastic, StochasticContribution):
        raise ValueError("stochastic argument not given correctly.")

    return stochastic
