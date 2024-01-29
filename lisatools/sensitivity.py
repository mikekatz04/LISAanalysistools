import warnings
from typing import Tuple, Optional

import math
import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from .utils.utility import get_array_module

"""
This script defines the parameters of the LISA instrument
"""

"""
    Copyright (C) 2017 Stas Babak, Antoine Petiteau for the LDC team

    This file is part of LISA Data Challenge.

    LISA Data Challenge is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""

##################################################
#                                                #
#                LISA Parameters                 #
#                  version 1.0                   #
#                                                #
#         S. Babak, A. Petiteau, ...             #
#      for the LISA Data Challenge team          #
#                                                #
##################################################

from .utils.constants import *

#### Armlength
lisaL = 2.5e9  # LISA's arm meters
lisaLT = lisaL / C_SI  # LISA's armn in sec

#### Noise levels
### Optical Metrology System noise
## Decomposition
Sloc = (1.7e-12) ** 2  # m^2/Hz
Ssci = (8.9e-12) ** 2  # m^2/Hz
Soth = (2.0e-12) ** 2  # m^2/Hz
## Global
Soms_d_all = {
    "Proposal": (10.0e-12) ** 2,
    "SciRDv1": (15.0e-12) ** 2,
    "MRDv1": (10.0e-12) ** 2,
    "sangria": (7.9e-12) ** 2,
}  # m^2/Hz

### Acceleration
Sa_a_all = {
    "Proposal": (3.0e-15) ** 2,
    "SciRDv1": (3.0e-15) ** 2,
    "MRDv1": (2.4e-15) ** 2,
    "sangria": (2.4e-15) ** 2,
}  # m^2/sec^4/Hz


lisaD = 0.3  # TODO check it
lisaP = 2.0  # TODO check it


"""
References for noise models:
  * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
  * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
  * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
"""


def AET(
    X: float | np.ndarray, Y: float | np.ndarray, Z: float | np.ndarray
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Transform to AET from XYZ

    .. math::

        A = (Z - X) / \\sqrt(2)
        E = (X - 2Y + Z) / \\sqrt(6)
        T = (X + Y + Z) / \\sqrt(3)

    Args:
        X: X-channel information.
        Y: Y-channel information.
        Z: Z-channel information.

    Returns:
        A, E, T Channels.

    """
    return (
        (Z - X) / math.sqrt(2.0),
        (X - 2.0 * Y + Z) / math.sqrt(6.0),
        (X + Y + Z) / math.sqrt(3.0),
    )


def lisasens(
    f: float | np.ndarray,
    model: Optional[str | ArrayLike] = "SciRDv1",
    includewd: Optional[float] = None,
    foreground_args: Optional[tuple] = (),
    foreground_kwargs: Optional[dict] = {},
    foreground_function: Optional[callable] = None,
    average: Optional[bool] = True,
) -> float | np.ndarray:
    """Compute LISA sensitivity

    Args:
        f: Frequency array.
        model: Noise model. Can be a string. String options are:
                * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
                * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
                * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
                * 'Sangria': Sangria LDC2A Challenge.
            If an ArrayLike object is provided (list, tuple, array), passed as arguments for the acceleration an OMS noise. TODO: improve description. TODO: fix ArrayLike
        includewd: Time in years of observation of foreground noise. If None, it is not included.
            If either ``foreground_args`` or ``foreground_kwargs`` are given, then ``includewd`` is ignored.
        foreground_args: Parameters (arguments) to feed to ``foreground_function``.
        foreground_kwargs: Keyword arguments to feeed to ``foreground_function``.
        foreground_function: Function to calculate foreground. Takes ``foreground_args`` and ``foreground_kwargs``. If ``None``, it uses :func:`GalConf`.
        average: If ``True``, use an averaging factor of [sqrt(5)  *  2 / sqrt(3)] {response factor * projection effect}.

    Returns:
        LISA sensitivity value.

    """

    if isinstance(f, float):
        f = np.ndarray([f])
        squeeze = True
    else:
        squeeze = False

    xp = get_array_module(f)

    Sa_d, Sop = lisanoises(f, model, "displacement")

    all_m = xp.sqrt(4.0 * Sa_d + Sop)
    ## Average the antenna response
    av_resp = xp.sqrt(5) if average else 1.0

    ## Projection effect
    Proj = 2.0 / xp.sqrt(3) if average else 1.0

    ## Approximative transfer function
    f0 = 1.0 / (2.0 * lisaLT)
    a = 0.41
    T = xp.sqrt(1 + (f / (a * f0)) ** 2)
    sens = (av_resp * Proj * T * all_m / lisaL) ** 2

    if includewd is not None or foreground_args != () or foreground_kwargs != {}:
        if foreground_function is None:
            foreground_function = GalConf

        sgal = foreground_function(
            f, *foreground_args, Tobs=includewd, **foreground_kwargs
        )
        sens += sgal

    return sens


# FIXME the model for GB stochastic signal is hardcoded
def noisepsd_X(
    f: float | np.ndarray,
    model: Optional[str | ArrayLike] = "SciRDv1",
    includewd: Optional[float] = None,
    foreground_args: Optional[tuple] = (),
    foreground_kwargs: Optional[dict] = {},
    foreground_function: Optional[callable] = None,
) -> float | np.ndarray:
    """Compute PSD for channel X

    Args:
        f: Frequency array.
        model: Noise model. Can be a string. String options are:
                * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
                * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
                * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
                * 'Sangria': Sangria LDC2A Challenge.
            If an ArrayLike object is provided (list, tuple, array), passed as arguments for the acceleration an OMS noise. TODO: improve description. TODO: fix ArrayLike
        includewd: Time in years of observation of foreground noise. If None, it is not included.
            If either ``foreground_args`` or ``foreground_kwargs`` are given, then ``includewd`` is ignored.
        foreground_args: Parameters (arguments) to feed to ``foreground_function``.
        foreground_kwargs: Keyword arguments to feeed to ``foreground_function``.
        foreground_function: Function to calculate foreground. Takes ``foreground_args`` and ``foreground_kwargs``. If ``None``, it uses :func:`GalConf`.

    Returns:
        LISA PSD value for channel X.

    """

    x = 2.0 * np.pi * lisaLT * f

    # get noise values
    Spm, Sop = lisanoises(f, model)

    # geometric functions to first order
    Sx = 16.0 * np.sin(x) ** 2 * (2.0 * (1.0 + np.cos(x) ** 2) * Spm + Sop)

    if includewd is not None or foreground_args != () or foreground_kwargs != {}:
        Sx += WDconfusionX(
            f,
            foreground_args=foreground_args,
            includewd=includewd,
            foreground_kwargs=foreground_kwargs,
        )
    return Sx


def noisepsd_X2(
    f: float | np.ndarray,
    model: Optional[str | ArrayLike] = "SciRDv1",
    includewd: Optional[float] = None,
    foreground_args: Optional[tuple] = (),
    foreground_kwargs: Optional[dict] = {},
    foreground_function: Optional[callable] = None,
) -> float | np.ndarray:
    """Compute PSD for channel X2

    Args:
        f: Frequency array.
        model: Noise model. Can be a string. String options are:
                * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
                * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
                * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
                * 'Sangria': Sangria LDC2A Challenge.
            If an ArrayLike object is provided (list, tuple, array), passed as arguments for the acceleration an OMS noise. TODO: improve description. TODO: fix ArrayLike
        includewd: Time in years of observation of foreground noise. If None, it is not included.
            If either ``foreground_args`` or ``foreground_kwargs`` are given, then ``includewd`` is ignored.
        foreground_args: Parameters (arguments) to feed to ``foreground_function``.
        foreground_kwargs: Keyword arguments to feeed to ``foreground_function``.
        foreground_function: Function to calculate foreground. Takes ``foreground_args`` and ``foreground_kwargs``. If ``None``, it uses :func:`GalConf`.

    Returns:
        LISA PSD value for channel X.

    """

    x = 2.0 * np.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sx = 64.0 * np.sin(x) ** 2 * np.sin(2 * x) ** 2 * Sop
    ## TODO Check the acceleration noise term
    Sx += 256.0 * (3 + np.cos(2 * x)) * np.cos(x) ** 2 * np.sin(x) ** 4 * Spm

    ### TODO Incule Galactic Binaries
    if includewd is not None or foreground_args != () or foreground_kwargs != {}:
        raise NotImplementedError

        Sx += WDconfusionX(
            f,
            foreground_args=foreground_args,
            includewd=includewd,
            foreground_kwargs=foreground_kwargs,
        )

    return Sx


def noisepsd_XY(f, model="SciRDv1", includewd=None, foreground_params=None):
    """
    Compute and return analytic PSD of noise for the correlation between TDI X-Y
     @param frequencydata  numpy array.
     @param model is the noise model:
         * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
         * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
         * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
     @param includewd whether to include  GB confusion, if yes should give a duration of observations in years.
         example: includewd=2.3 - 2.3 yeras of observations
         if includewd == None: includewd = model.lisaWD
    """
    x = 2.0 * math.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sxy = -4.0 * np.sin(2 * x) * np.sin(x) * (Sop + 4.0 * Spm)
    # Sa = Sx - Sxy
    # GB = -0.5 of X

    if includewd is not None or foreground_params is not None:
        Sxy += -0.5 * WDconfusionX(
            f, duration=includewd, foreground_params=foreground_params, model=model
        )  # TODO To be checked

    return Sxy


def noisepsd_AE(
    f, model="SciRDv1", includewd=None, foreground_params=None, use_gpu=False
):
    """
    Compute and return analytic PSD of noise for TDI A and E
     @param frequencydata  numpy array.
     @param model is the noise model:
         * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
         * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
         * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
     @param includewd whether to include  GB confusion, if yes should give a duration of observations in years.
         example: includewd=2.3 - 2.3 yeras of observations
         if includewd == None: includewd = model.lisaWD
    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    x = 2.0 * math.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sa = (
        8.0
        * xp.sin(x) ** 2
        * (
            2.0 * Spm * (3.0 + 2.0 * xp.cos(x) + xp.cos(2 * x))
            + Sop * (2.0 + xp.cos(x))
        )
    )

    if includewd is not None or foreground_params is not None:
        Sa += WDconfusionAE(
            f,
            duration=includewd,
            foreground_params=foreground_params,
            model=model,
            use_gpu=use_gpu,
        )
        # Sa += makewdnoise(f,includewd,'AE')

    return Sa


def noisepsd_AE2(f, model="SciRDv1", includewd=None, foreground_params=None):
    """
    Compute and return analytic PSD of noise for TDI A and E
     @param frequencydata  numpy array.
     @param model is the noise model:
         * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
         * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
         * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
     @param includewd whether to include  GB confusion, if yes should give a duration of observations in years.
         example: includewd=2.3 - 2.3 yeras of observations
         if includewd == None: includewd = model.lisaWD
    """
    x = 2.0 * math.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sa = (
        32.0
        * np.sin(x) ** 2
        * np.sin(2 * x) ** 2
        * (
            2.0 * Spm * (3.0 + 2.0 * np.cos(x) + np.cos(2 * x))
            + Sop * (2.0 + np.cos(x))
        )
    )

    if includewd is not None or foreground_params is not None:
        raise NotImplementedError
        Sa += WDconfusionAE(f, includewd, model=model)
        # Sa += makewdnoise(f,includewd,'AE')

    return Sa


# TODO: currently not including WD background here... probably OK
def noisepsd_T(f, model="SciRDv1", includewd=None, foreground_params=None):
    """
    Compute and return analytic PSD of noise for TDI T
     @param frequencydata  numpy array.
     @param model is the noise model:
         * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
         * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
         * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
     @param includewd whether to include  GB confusion, if yes should give a duration of observations in years.
         example: includewd=2.3 - 2.3 yeras of observations
         if includewd == None: includewd = model.lisaWD
    """
    x = 2.0 * math.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    St = (
        16.0 * Sop * (1.0 - np.cos(x)) * np.sin(x) ** 2
        + 128.0 * Spm * np.sin(x) ** 2 * np.sin(0.5 * x) ** 4
    )

    return St


def SGal(fr, pars, use_gpu=False):
    """
    TODO To be described
    """
    if use_gpu:
        xp = cp
    else:
        xp = np
    # {{{
    Amp = pars[0]
    alpha = pars[1]
    sl1 = pars[2]
    kn = pars[3]
    sl2 = pars[4]
    Sgal = (
        Amp
        * xp.exp(-(fr**alpha) * sl1)
        * (fr ** (-7.0 / 3.0))
        * 0.5
        * (1.0 + xp.tanh(-(fr - kn) * sl2))
    )

    return Sgal
    # }}}


def GalConf(fr, Tobs=None, foreground_params=None, use_gpu=False):
    """
    TODO To be described
    """

    # {{{
    # Tobs should be in sec.
    day = 86400.0
    month = day * 30.5
    year = 365.25 * 24.0 * 3600.0

    # Sgal_1d = 2.2e-44*np.exp(-(fr**1.2)*0.9e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-1.4e-2)*0.7e2))
    # Sgal_3m = 2.2e-44*np.exp(-(fr**1.2)*1.7e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-4.8e-3)*5.4e2))
    # Sgal_1y = 2.2e-44*np.exp(-(fr**1.2)*2.2e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-3.1e-3)*1.3e3))
    # Sgal_2y = 2.2e-44*np.exp(-(fr**1.2)*2.2e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-2.3e-3)*1.8e3))
    # Sgal_4y = 2.2e-44*np.exp(-(fr**1.2)*2.9e3)*(fr**(-7./3.))*0.5*(1.0 + np.tanh(-(fr-2.0e-3)*1.9e3))

    if Tobs is not None:
        Amp = 3.26651613e-44
        alpha = 1.18300266e00

        Xobs = [
            1.0 * day,
            3.0 * month,
            6.0 * month,
            1.0 * year,
            2.0 * year,
            4.0 * year,
            10.0 * year,
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
        knee = [
            1.15120924e-02,
            4.01884128e-03,
            3.47302482e-03,
            2.77606177e-03,
            2.41178384e-03,
            2.09278117e-03,
            1.57362626e-03,
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

        # Slope1 = [9.0e2, 1.7e3, 2.2e3, 2.2e3, 2.9e3]
        # knee = [1.4e-2, 4.8e-3, 3.1e-3, 2.3e-3, 2.0e-3]
        # Slope2 = [0.7e2, 5.4e2, 1.3e3, 1.8e3, 1.9e3]

        Tmax = 10.0 * year
        if Tobs > Tmax:
            print("I do not do extrapolation, Tobs > Tmax:", Tobs, Tmax)
            sys.exit(1)

        # Interpolate
        tck1 = interpolate.splrep(Xobs, Slope1, s=0, k=1)
        tck2 = interpolate.splrep(Xobs, knee, s=0, k=1)
        tck3 = interpolate.splrep(Xobs, Slope2, s=0, k=1)
        sl1 = interpolate.splev(Tobs, tck1, der=0)
        kn = interpolate.splev(Tobs, tck2, der=0)
        sl2 = interpolate.splev(Tobs, tck3, der=0)
        # print "interpolated values: slope1, knee, slope2", sl1, kn, sl2

    elif foreground_params is not None:
        assert len(foreground_params) == 5
        Amp, alpha, sl1, kn, sl2 = foreground_params

    else:
        raise ValueError("Must provide either Tobs or foreground_params.")
    Sgal_int = SGal(fr, [Amp, alpha, sl1, kn, sl2], use_gpu=use_gpu)

    return Sgal_int


# TODO check it against the old LISA noise
def WDconfusionX(
    f, foreground_params=None, duration=None, model="SciRDv1", use_gpu=False
):
    """
    TODO To be described
    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    if foreground_params is None and duration is None:
        raise ValueError(
            "Must provide either duration or direct background parameters."
        )
    # duration is assumed to be in years
    day = 86400.0
    year = 365.25 * 24.0 * 3600.0

    if duration is not None and ((duration < day / year) or (duration > 10.0)):
        raise NotImplementedError

    if True:  # (
        # model == "Proposal" or model == "SciRDv1" or model == "sangria"
        # ):  ## WANRNING: WD should be regenrate for SciRD
        x = 2.0 * math.pi * lisaLT * f
        t = 4.0 * x**2 * xp.sin(x) ** 2
        if duration is not None:
            duration_in = duration * year
        else:
            duration_in = None

        Sg_sens = GalConf(
            f, Tobs=duration_in, foreground_params=foreground_params, use_gpu=use_gpu
        )

        # t = 4 * x**2 * xp.sin(x)**2 * (1.0 if obs == 'X' else 1.5)
        return t * Sg_sens

        ##return t * ( N.piecewise(f,(f >= 1.0e-4  ) & (f < 1.0e-3  ),[lambda f: 10**-44.62 * f**-2.3, 0]) + \
        #             N.piecewise(f,(f >= 1.0e-3  ) & (f < 10**-2.7),[lambda f: 10**-50.92 * f**-4.4, 0]) + \
        #             N.piecewise(f,(f >= 10**-2.7) & (f < 10**-2.4),[lambda f: 10**-62.8  * f**-8.8, 0]) + \
        #             N.piecewise(f,(f >= 10**-2.4) & (f < 10**-2.0),[lambda f: 10**-89.68 * f**-20.0,0])     )
    else:
        if ".txt" in wdstyle:
            conf = np.loadtxt(wdstyle)
            conf[np.isnan(conf[:, 1]), 1] = 0

            return np.interp(f, conf[:, 0], conf[:, 1])
        else:
            raise NotImplementedError


def WDconfusionAE(
    f, foreground_params=None, duration=None, model="SciRDv1", use_gpu=False
):
    """
    TODO To be described
    """
    SgX = WDconfusionX(
        f,
        foreground_params=foreground_params,
        duration=duration,
        model=model,
        use_gpu=use_gpu,
    )
    return 1.5 * SgX


def lisanoises(f, model="SciRDv1", unit="relativeFrequency"):
    """
    Return the analytic approximation of the two components of LISA noise,
    i.e. the acceleration and the
    @param f is the frequency array
    @param model is the noise model:
        * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
        * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
        * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
    @param unit is the unit of the output: 'relativeFrequency' or 'displacement'
    """
    if model == "mldc":
        Spm = 2.5e-48 * (1.0 + (f / 1.0e-4) ** -2) * f ** (-2)
        defaultL = 16.6782
        Sop = 1.8e-37 * (lisaLT / defaultL) ** 2 * f**2

    elif model == "newdrs":  # lisalight, to be used with lisaL = 1Gm, lisaP = 2
        Spm = 6.00314e-48 * f ** (-2)  # 4.6e-15 m/s^2/sqrt(Hz)
        defaultL = 16.6782
        defaultD = 0.4
        defaultP = 1.0
        Sops = (
            6.15e-38
            * (lisaLT / defaultL) ** 2
            * (defaultD / lisaD) ** 4
            * (defaultP / lisaP)
        )  # 11.83 pm/sqrt(Hz)
        Sopo = 2.81e-38  # 8 pm/sqrt(Hz)
        Sop = (Sops + Sopo) * f**2

    elif model == "LCESAcall":
        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_all["Proposal"] * (1.0 + (0.4e-3 / frq) ** 2 + (frq / 9.3e-3) ** 4)
        ## In displacement
        Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
        ## In relative frequency unit
        Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
        Spm = Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_all["Proposal"] * (1.0 + (2.0e-3 / f) ** 4)
        ## In relative frequency unit
        Soms_nu = Soms_d * (2.0 * np.pi * frq / C_SI) ** 2
        Sop = Soms_nu

    elif model == "Proposal":
        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = (
            Sa_a_all["Proposal"]
            * (1.0 + (0.4e-3 / frq) ** 2)
            * (1.0 + (frq / 8e-3) ** 4)
        )
        ## In displacement
        Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
        ## In relative frequency unit
        Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
        Spm = Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_all["Proposal"] * (1.0 + (2.0e-3 / f) ** 4)
        ## In relative frequency unit
        Soms_nu = Soms_d * (2.0 * np.pi * frq / C_SI) ** 2
        Sop = Soms_nu

    elif (
        isinstance(model, list)
        or isinstance(model, np.ndarray)
        or model == "SciRDv1"
        or model == "MRDv1"
        or model == "sangria"
    ):
        if isinstance(model, str):
            Soms_d_in = Soms_d_all[model]
            Sa_a_in = Sa_a_all[model]

        else:
            # square root of the actual value
            Soms_d_in = model[0] ** 2
            Sa_a_in = model[1] ** 2

        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_in * (1.0 + (0.4e-3 / frq) ** 2) * (1.0 + (frq / 8e-3) ** 4)
        ## In displacement
        Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
        ## In relative frequency unit
        Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
        Spm = Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_in * (1.0 + (2.0e-3 / f) ** 4)
        ## In relative frequency unit
        Soms_nu = Soms_d * (2.0 * np.pi * frq / C_SI) ** 2
        Sop = Soms_nu

    else:
        raise NotImplementedError(model)

    if unit == "displacement":
        return Sa_d, Soms_d
    elif unit == "relativeFrequency":
        return Spm, Sop
    else:
        raise NotImplementedError(unit)


def flat_psd_function(f, val, *args, use_gpu=False, **kwargs):
    if use_gpu:
        xp = cp
    else:
        xp = np
    val = xp.atleast_1d(xp.asarray(val))
    out = xp.repeat(val[:, None], len(f), axis=1)
    return out


def cornish_lisa_psd(f, sky_averaged=False, use_gpu=False):
    """PSD from https://arxiv.org/pdf/1803.01944.pdf

    Power Spectral Density for the LISA detector assuming it has been active for a year.
    I found an analytic version in one of Niel Cornish's paper which he submitted to the arXiv in
    2018. I evaluate the PSD at the frequency bins found in the signal FFT.

    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf

    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    if sky_averaged:
        sky_averaging_constant = 20 / 3

    else:
        sky_averaging_constant = 1.0  # set to one for one source

    L = 2.5 * 10**9  # Length of LISA arm
    f0 = 19.09 * 10 ** (-3)  # transfer frequency

    # Optical Metrology Sensor
    Poms = ((1.5e-11) * (1.5e-11)) * (1 + xp.power((2e-3) / f, 4))

    # Acceleration Noise
    Pacc = (
        (3e-15)
        * (3e-15)
        * (1 + (4e-4 / f) * (4e-4 / f))
        * (1 + xp.power(f / (8e-3), 4))
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


def get_sensitivity(f, *args, sens_fn="lisasens", return_type="PSD", **kwargs):
    """Generic sensitivity generator

    Same interface to many sensitivity curves.

    Args:
        f (1D double np.ndarray): Array containing frequency  values.
        sens_fn (str, optional): String that represents the name of the desired
            SNR function. Options are "cornish_lisa_psd" or any sensitivity
            function found in tdi.py from the MLDC gitlab. Default is the
            LISA sensitivity from the tdi.py.
        return_type (str, optional): Described the desired output. Choices are ASD,
            PSD, or char_strain (characteristic strain). Default is ASD.

        *args (list or tuple, optional): Any additional arguments for the sensitivity function.
        **kwargs (dict, optional): Keyword arguments to pass to sensitivity function.

    """

    try:
        sensitivity = globals()[sens_fn]
    except KeyError:
        raise ValueError("{} sensitivity is not available.".format(sens_fn))

    PSD = sensitivity(f, *args, **kwargs)

    if return_type == "PSD":
        return PSD

    elif return_type == "ASD":
        return PSD ** (1 / 2)

    elif return_type == "char_strain":
        return (f * PSD) ** (1 / 2)

    else:
        raise ValueError("return_type must be PSD, ASD, or char_strain.")
