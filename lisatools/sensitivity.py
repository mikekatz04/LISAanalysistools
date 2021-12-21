import warnings

import numpy as np

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp


import math
import numpy as np
from scipy import interpolate

# import matplotlib.pyplot as plt

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

from lisatools.utils.constants import *

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
}  # m^2/Hz

### Acceleration
Sa_a_all = {
    "Proposal": (3.0e-15) ** 2,
    "SciRDv1": (3.0e-15) ** 2,
    "MRDv1": (2.4e-15) ** 2,
}  # m^2/sec^4/Hz


lisaD = 0.3  # TODO check it
lisaP = 2.0  # TODO check it


"""
References for noise models:
  * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
  * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
  * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
"""


def AET(X, Y, Z):
    return (
        (Z - X) / math.sqrt(2.0),
        (X - 2.0 * Y + Z) / math.sqrt(6.0),
        (X + Y + Z) / math.sqrt(3.0),
    )
    # return (2.*X - Y - Z)/3., (Z-Y)/np.sqrt(3.0), (X + Y + Z)/3.0


class TDIf(object):
    """TDI triple-observable object. Can be initialized with X,Y,Z or A,E,T, but
       will keep A,E,T internally."""

    def __init__(self, aet=None, xyz=None):
        """  ...
        """
        # TODO : describe the method
        if aet != None:
            self.Af, self.Ef, self.Tf = aet
        elif xyz != None:
            self.Af, self.Ef, self.Tf = AET(*xyz)
            self.Xf = xyz[0]
            self.Yf = xyz[1]
            self.Zf = xyz[2]  # keep also X (temporary?)

        ## Initalize the PSD of various TDI
        self._Sae, self._St, self._Sx, self._Sxy = None, None, None, None

    def pad(self, leftpad=1, rightpad=1):
        """  ...
        """
        # TODO : describe the method
        self.Af = self.Af.pad(leftpad, rightpad)
        self.Ef = self.Ef.pad(leftpad, rightpad)
        self.Tf = self.Tf.pad(leftpad, rightpad)

        self.Xf = self.Xf.pad(leftpad, rightpad)
        self.Yf = self.Yf.pad(leftpad, rightpad)
        self.Zf = self.Zf.pad(leftpad, rightpad)

        self._Sae, self._St, self._Sx = None, None, None

        return self

    # TO DO: memoize the properties?
    @property
    def Sae(self):
        """
        Return the PSD of TDI A and E. They are identical.
        """
        if self._Sae is None:
            self._Sae = noisepsd_AE(self.Af)
        return self._Sae

    @property
    def St(self):
        """
        Return the PSD of TDI T.
        """
        if self._St is None:
            self._St = noisepsd_T(self.Tf)
        return self._St

    @property
    def Sx(self):
        """
        Return the PSD of TDI X.
        """
        if self._Sx is None:
            self._Sx = noisepsd_X(self.Xf)
        return self._Sx

    @property
    def Sxy(self):
        """
        Return the PSD of TDI X.
        """
        if self._Sxy is None:
            self._Sxy = noisepsd_XY(self.Xf)
        return self._Sxy

    @property
    def kmin(self):
        """
        TODO To be described
        """
        return self.Af.kmin

    def __len__(self):
        """
        Return the number of frequency points
        """
        return len(self.Af)

    # FIXME The operators below require that XYZ are defined for TDI (let's see if it is a problem)
    def __add__(self, other):
        """
        Add others TDI data to the TDI data
        @param other is the other TDI data
        """
        # ret = TDIf(aet=(self.Af + other.Af,self.Ef + other.Ef,self.Tf + other.Tf))
        ret = TDIf(xyz=(self.Xf + other.Xf, self.Yf + other.Yf, self.Zf + other.Zf))
        # ret.Xf = self.Xf + other.Xf
        return ret

    def __sub__(self, other):
        """
        Subtract others TDI data to the TDI data
        @param other is the other TDI data
        """
        # ret = TDIf(aet=(self.Af - other.Af,self.Ef - other.Ef,self.Tf - other.Tf))
        ret = TDIf(xyz=(self.Xf - other.Xf, self.Yf - other.Yf, self.Zf - other.Zf))
        # ret.Xf = self.Xf - other.Xf
        return ret

    def __mul__(self, other):
        """
        Multiply the TDI data by others TDI data
        @param other is the other TDI data
        """
        if isinstance(other, TDIf):
            # ret = TDIf(aet=(self.Af*other.Af,self.Ef*other.Ef,self.Tf*other.Tf))
            ret = TDIf(xyz=(self.Xf * other.Xf, self.Yf * other.Yf, self.Zf * other.Zf))
            # ret.Xf = self.Xf * other.Xf
        else:
            # ret = TDIf(aet=(self.Af*other,self.Ef*other,self.Tf*other))
            ret = TDIf(xyz=(self.Xf * other, self.Yf * other, self.Zf * other))
            # ret.Xf = YDIf(xyz=(self.Xf * other, self.Yf*other, self.Zf*other))

        return ret

    def __rmul__(self, other):
        """
        Multiply the TDI data by real value
        @param other is the real value
        """
        # ret = TDIf(aet=(self.Af*other,self.Ef*other,self.Tf*other))
        ret = TDIf(xyz=(self.Xf * other, self.Yf * other, self.Zf * other))
        # ret.Xf = self.Xf * other
        return ret

    def __div__(self, other):
        """
        Divide the TDI data by others TDI data
        @param other is the other TDI data
        """
        if isinstance(other, TDIf):
            # ret = TDIf(aet=(self.Af/other.Af,self.Ef/other.Ef,self.Tf/other.Tf))
            ret = TDIf(xyz=(self.Xf / other.Xf, self.Yf / other.Yf, self.Zf / other.Zf))
            # ret.Xf = self.Xf / other.Xf
        else:
            # ret = TDIf(aet=(self.Af/other,self.Ef/other,self.Tf/other))
            ret = TDIf(xyz=(self.Xf / other, self.Yf / other, self.Zf / other))
            # ret.Xf = self.Xf / other

        return ret

    def __iadd__(self, other):
        """
        TODO To be described
        @param other is the other TDI data
        """
        self.Af += other.Af
        self.Ef += other.Ef
        self.Tf += other.Tf
        self.Xf += other.Xf
        self.Yf += other.Yf
        self.Zf += other.Zf

    def __isub__(self, other):
        """
        TODO To be described
        @param other is the other TDI data
        """
        self.Af -= other.Af
        self.Ef -= other.Ef
        self.Tf -= other.Tf
        self.Xf += other.Xf
        self.Yf -= other.Yf
        self.Zf -= other.Zf

    def normsq(self, noisepsd=None, extranoise=[0, 0, 0]):
        """
        Return the squared norm of TDI data A,E,T
        """
        if noisepsd == None:
            return (4.0 * self.Af.df) * (
                np.sum(np.abs(self.Af) ** 2 / (self.Sae + extranoise[0]))
                + np.sum(np.abs(self.Ef) ** 2 / (self.Sae + extranoise[1]))
                + np.sum(np.abs(self.Tf) ** 2 / (self.St + extranoise[2]))
            )
        else:
            return (4.0 * self.Af.df) * (
                np.sum(np.abs(self.Af) ** 2 / noisepsd[0])
                + np.sum(np.abs(self.Ef) ** 2 / noisepsd[1])
                + np.sum(np.abs(self.Tf) ** 2 / noisepsd[2])
            )

    def normsqx(self, noisepsd=None):
        """
        Return the squared norm of TDI data X
        """
        if noisepsd == None:
            # return (4.0 / self.Xf.df) * ( np.sum(np.abs(self.Xf)**2 / (self.Sx)) )
            im = self.Xf.kmin
            return (4.0 * self.Xf.df) * (
                np.sum(np.abs(self.Xf[im:]) ** 2 / (self.Sx[im:]))
            )
        else:
            return (4.0 * self.Xf.df) * (np.sum(np.abs(self.Xf) ** 2 / noisepsd))
            # return (4.0 / self.Xf.df) * ( np.sum(np.abs(self.Xf)**2 / noisepsd) )

    def cprod(self, other):
        """
        Return the cross product of TDI data with other TDI data
        @param other is the other TDI data
        """
        return (4.0 / self.Af.df) * (
            np.sum(np.conj(self.Af) * other.Af / self.Sae)
            + np.sum(np.conj(self.Ef) * other.Ef / self.Sae)
            + np.sum(np.conj(self.Tf) * other.Tf / self.St)
        )

    def dotprod(self, other):
        """
        Return the scalar product of TDI data with other TDI data
        @param other is the other TDI data
        """
        return (4.0 / self.Af.df) * np.real(
            np.sum(np.conj(self.Af) * other.Af / self.Sae)
            + np.sum(np.conj(self.Ef) * other.Ef / self.Sae)
            + np.sum(np.conj(self.Tf) * other.Tf / self.St)
        )

    def cprodx(self, other):
        return (4.0 / self.Xf.df) * (np.sum(np.conj(self.Xf) * other.Xf / self.Sx))

    def dotprodx(self, other):
        """
        Return the scalar product of TDI data X with other TDI data X
        @param other is the other TDI data
        """
        return (4.0 / self.Xf.df) * np.real(
            np.sum(np.conj(self.Xf) * other.Xf / self.Sx)
        )

    def logL(self, other):
        """
        Return the log likelihood of other TDI data (model) wrt TDI data
        @param other is the other TDI data
        """
        return (
            -0.5
            * (4.0 / self.Af.df)
            * (
                np.sum(np.abs(self.Af.rsub(other.Af)) ** 2 / self.Sae)
                + np.sum(np.abs(self.Ef.rsub(other.Ef)) ** 2 / self.Sae)
                + np.sum(np.abs(self.Tf.rsub(other.Tf)) ** 2 / self.St)
            )
        )


# TODO need to implement this properly
# def SNR(center,gettdi):
#     return math.sqrt(gettdi(center).normsq())
#
#
# def fisher(center,scale,gettdi):
#     x0, dx = np.asarray(center), np.asarray(scale)
#     d = len(x0)
#
#     # is there a better numpy builtin for the *vector* delta_ij?
#     delta = lambda i: np.identity(d)[i,:]
#
#     derivs = [ ( gettdi(x0 + delta(i)*dx[i]) - gettdi(x0 - delta(i)*dx[i]) ) / (2.0*dx[i]) for i in range(d) ]
#
#     prods = N.zeros((d,d),'d')
#     for i in range(d):
#         for j in range(i,d):
#             prods[i,j] = prods[j,i] = derivs[i].dotprod(derivs[j])
#
#     return prods


def simplesnr(f, h, i=None, years=1.0, noisemodel="SciRDv1", includewd=None):
    """
    TODO To be described
    @param other is the other TDI data
    """
    if i == None:
        h0 = h * np.sqrt(16.0 / 5.0)  # rms average over inclinations
    else:
        h0 = h * np.sqrt((1 + np.cos(i) ** 2) ** 2 + (2.0 * np.cos(i)) ** 2)

    snr = (
        h0
        * np.sqrt(years * 365.25 * 24 * 3600)
        / np.sqrt(lisasens(f, noisemodel, years))
    )
    # print "snr = ", snr, np.sqrt(years * 365.25*24*3600), h0 * np.sqrt(years * 365.25*24*3600) , np.sqrt(lisasens(f,noisemodel,years))
    return snr


def lisasens(f, noiseModel="SciRDv1", includewd=None):
    """
    Compute LISA sensitivity
    @param f is the frequency array
    @param model is the noise model:
        * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
        * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
        * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
    @param unit is the unit of the output: 'relativeFrequency' or 'displacement'
    """
    # noide model variable is ignored now, and can be used if we add more noise models
    # includewd - should be duration of observation in years (if not None)
    # Sa_a = Sa_a_all[noiseModel]*(1.0 +(0.4e-3/f)**2)*(1.0+(f/8e-3)**4)
    # Sa_d = Sa_a*(2.*np.pi*f)**(-4.)

    Sop = Soms_d_all[noiseModel] * (1.0 + (2.0e-3 / f) ** 4)

    Sa_d, Sop = lisanoises(f, noiseModel, "displacement")

    ALL_m = np.sqrt(4.0 * Sa_d + Sop)
    ## Average the antenna response
    AvResp = np.sqrt(5)
    ## Projection effect
    Proj = 2.0 / np.sqrt(3)
    ## Approximative transfert function
    f0 = 1.0 / (2.0 * lisaLT)
    a = 0.41
    T = np.sqrt(1 + (f / (a * f0)) ** 2)
    Sens = (AvResp * Proj * T * ALL_m / lisaL) ** 2

    if includewd != None:
        day = 86400.0
        year = 365.25 * 24.0 * 3600.0
        if (includewd < day / year) or (includewd > 10.0):
            raise NotImplementedError
        Sgal = GalConf(f, includewd * year)
        Sens = Sens + Sgal

    return Sens


# FIXME the model for GB stochastic signal is hardcoded
def noisepsd_X(f, model="SciRDv1", includewd=None):
    """
    Compute and return analytic PSD of noise for TDI X
     @param frequencydata  numpy array.
     @param model is the noise model:
         * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
         * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
         * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
     @param includewd whether to include  GB confusion, if yes should give a duration of observations in years.
         example: includewd=2.3 - 2.3 yeras of observations
         if includewd == None: includewd = model.lisaWD
    """

    x = 2.0 * np.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sx = 16.0 * np.sin(x) ** 2 * (2.0 * (1.0 + np.cos(x) ** 2) * Spm + Sop)

    if includewd != None:
        Sx += WDconfusionX(f, includewd, model=model)
        # Sx += makewdnoise(f,includewd,'X')

    return Sx


def noisepsd_X2(f, model="SciRDv1"):
    """
    Compute and return analytic PSD of noise for TDI X
     @param frequencydata  numpy array.
     @param model is the noise model:
         * 'Proposal': LISA Consortium Proposal for L3 mission: LISA_L3_20170120 (https://atrium.in2p3.fr/13414ec1-c9ac-44b4-bace-7004468f684c)
         * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
         * 'MRDv1': Mission Requirement Document: ESA-L3-EST-MIS-RS-001 08/12/2017
    """

    x = 2.0 * np.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sx = 64.0 * np.sin(x) ** 2 * np.sin(2 * x) ** 2 * Sop
    ## TODO Check the acceleration noise term
    Sx += 256.0 * (3 + np.cos(2 * x)) * np.cos(x) ** 2 * np.sin(x) ** 4 * Spm

    ### TODO Incule Galactic Binaries
    # if includewd != None:
    #     Sx += WDconfusionX(f, includewd, model=model)
    #     #Sx += makewdnoise(f,includewd,'X')

    return Sx


def noisepsd_XY(f, model="SciRDv1", includewd=None):
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

    if includewd != None:
        Sxy += -0.5 * WDconfusionX(f, includewd, model=model)  # TODO To be checked

    return Sxy


def noisepsd_AE(f, model="SciRDv1", includewd=None):
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
        8.0
        * np.sin(x) ** 2
        * (
            2.0 * Spm * (3.0 + 2.0 * np.cos(x) + np.cos(2 * x))
            + Sop * (2.0 + np.cos(x))
        )
    )

    if includewd != None:
        Sa += WDconfusionAE(f, includewd, model=model)
        # Sa += makewdnoise(f,includewd,'AE')

    return Sa


def noisepsd_AE2(f, model="SciRDv1", includewd=None):
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

    if includewd != None:
        raise NotImplementedError
        Sa += WDconfusionAE(f, includewd, model=model)
        # Sa += makewdnoise(f,includewd,'AE')

    return Sa


# TODO: currently not including WD background here... probably OK
def noisepsd_T(f, model="SciRDv1", includewd=None):
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


def SGal(fr, pars):
    """
    TODO To be described
    """
    # {{{
    Amp = pars[0]
    alpha = pars[1]
    sl1 = pars[2]
    kn = pars[3]
    sl2 = pars[4]
    Sgal = (
        Amp
        * np.exp(-(fr ** alpha) * sl1)
        * (fr ** (-7.0 / 3.0))
        * 0.5
        * (1.0 + np.tanh(-(fr - kn) * sl2))
    )

    return Sgal
    # }}}


def GalConf(fr, Tobs):
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
    Sgal_int = SGal(fr, [Amp, alpha, sl1, kn, sl2])

    return Sgal_int


# TODO check it against the old LISA noise
def WDconfusionX(f, duration, model="SciRDv1"):
    """
    TODO To be described
    """
    # duration is assumed to be in years
    day = 86400.0
    year = 365.25 * 24.0 * 3600.0
    if (duration < day / year) or (duration > 10.0):
        raise NotImplementedError

    if (
        model == "Proposal" or model == "SciRDv1"
    ):  ## WANRNING: WD should be regenrate for SciRD
        x = 2.0 * math.pi * lisaLT * f
        t = 4.0 * x ** 2 * np.sin(x) ** 2
        Sg_sens = GalConf(f, duration * year)
        # t = 4 * x**2 * np.sin(x)**2 * (1.0 if obs == 'X' else 1.5)
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


def WDconfusionAE(f, duration, model="SciRDv1"):
    """
    TODO To be described
    """
    SgX = WDconfusionX(f, duration, model)
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
        Sop = 1.8e-37 * (lisaLT / defaultL) ** 2 * f ** 2

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
        Sop = (Sops + Sopo) * f ** 2

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

    elif model == "SciRDv1" or model == "MRDv1":
        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_all[model] * (1.0 + (0.4e-3 / frq) ** 2) * (1.0 + (frq / 8e-3) ** 4)
        ## In displacement
        Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
        ## In relative frequency unit
        Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
        Spm = Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_all[model] * (1.0 + (2.0e-3 / f) ** 4)
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


# """


"""
class model(object):
    c = 299792458

    defaultnoise = 'lisareq'
    noisemodel = defaultnoise

    defaultL = 16.6782
    lisaL = defaultL

    defaultD = 0.4
    lisaD = defaultD

    defaultP = 1.0
    lisaP = defaultP

    defaultWD = None
    lisaWD = defaultWD

    @staticmethod
    def setL(Lm):
        # set L in meters (same as in fastsource/fastbinary)
        model.lisaL = Lm / model.c

    @staticmethod
    def setmodel(mymodel,myarm=None):
        model.noisemodel = model.defaultnoise
        model.lisaL  = model.defaultL
        model.lisaD  = model.defaultD
        model.lisaP  = model.defaultP
        model.lisaWD = model.defaultWD

        if myarm != None:
            model.setL(myarm) # will be overridden by models that assume arm

        if mymodel in ['lisa-classic','default']:
            pass
        elif mymodel == 'CLISA1_P005c_LPF':
            model.noisemodel = 'newlpf'
            model.lisaL  = 1e9 / model.c
            model.lisaP  = 0.05
        elif mymodel == '10LISA1_P2_DRS':
            model.noisemodel = 'newdrs-wrong'
            model.lisaL  = 1e9 / model.c
            model.lisaP  = 2
        elif mymodel == '10LISA1_P07_D25_DRS_4L':
            model.noisemodel = 'newdrs'
            model.lisaL  = 1e9 / model.c
            model.lisaP  = 0.7
            model.lisaD  = 0.25
        elif mymodel == '10LISA1_P2_D25_DRS_4L':
            model.noisemodel = 'newdrs'
            model.lisaL  = 1e9 / model.c
            model.lisaP  = 2
            model.lisaD  = 0.25
        elif mymodel == '10LISA1_P07_D25_RDRS_4L':
            model.noisemodel = 'reddrs'
            model.lisaL  = 1e9 / model.c
            model.lisaP  = 0.7
            model.lisaD  = 0.25
        elif mymodel == 'lagrange':
            model.noisemodel = 'wind'
            model.lisaL  = 21e9 / model.c
        elif mymodel == 'lagrange-smallmirror':
            model.noisemodel = 'wind'
            model.lisaL  = 21e9 / model.c
            model.lisaD  = 0.2
        elif mymodel in ['mldc','mldc-nominal','lisareq','toy','newlpf','newdrs','reddrs','lpf','wind','ax50']:
            model.noisemodel = mymodel
        else:
            raise NotImplementedError(mymodel)



# currently only the lpf model allows for the part of optical noise that does not change with armlength
def lisanoises(f,noisemodel=None):
    if noisemodel == None: noisemodel = model.noisemodel

    if   noisemodel == 'mldc':
        Spm = 2.5e-48 * (1.0 + (f/1.0e-4)**-2) * f**(-2)
        Sop = 1.8e-37 * (model.lisaL/model.defaultL)**2 * f**2
    elif noisemodel == 'mldc-nominal':
        Spm = 2.53654e-48 * (1.0 + (f/1.0e-4)**-2) * f**(-2)        # 3e-15 m/s^2/sqrt(Hz), divided by (2 pi c) and squared
        Sop = 1.75703e-37 * (model.lisaL/model.defaultL)**2 * f**2  # 20 pm/sqrt(Hz), mult. by (2 pi / c) and squared; all optical noise scales as shot noise
    elif noisemodel == 'lisareq':
        Spm = 2.53654e-48 * (1.0 + (f/1.0e-4)**-1) * (1.0 + (f/0.008)**4) * f**(-2)
        Sop = 1.42319e-37 * (model.lisaL/model.defaultL)**2 * (1.0 + (f/0.002)**-4) * f**2
    elif noisemodel == 'toy':
        Spm = 2.53654e-48 * f**(-2)                                 # 3e-15 m/s^2/sqrt(Hz), no reddening

        Sops = 1.1245e-37 * (model.lisaL/model.defaultL)**2 * (model.defaultD/model.lisaD)**4 * (model.defaultP/model.lisaP)   # 16 pm/sqrt(Hz)
        Sopo = 6.3253e-38                                                                                                      # 12 pm/sqrt(Hz)
        Sop = (Sops + Sopo) * f**2
    elif noisemodel == 'newlpf':  # lisalight, to be used with lisaL = 1Gm, lisaP = 0.05
        Spm = 8.17047e-48 * (1.0 + (f/1.8e-4)**-1)**2 * f**(-2)     # 5.3e-15 m/s^2/sqrt(Hz)

        Sops = 6.15e-38 * (model.lisaL/model.defaultL)**2 * (model.defaultD/model.lisaD)**4 * (model.defaultP/model.lisaP)      # 11.83 pm/sqrt(Hz)
        Sopo = 2.81e-38                                                                                                         # 8 pm/sqrt(Hz)
        Sop = (Sops + Sopo) * f**2
    elif noisemodel == 'newdrs-wrong':  # lisalight, to be used with lisaL = 1Gm, lisaP = 2
        Spm = 6.00314e-48 * f**(-2)                                 # 4.6e-15 m/s^2/sqrt(Hz)

        Sops = 3.07e-38 * (model.lisaL/model.defaultL)**2 * (model.defaultD/model.lisaD)**4 * (model.defaultP/model.lisaP)      # 8.36 pm/sqrt(Hz) - was scaled wrong with P
        Sopo = 2.81e-38                                                                                                         # 8 pm/sqrt(Hz)
        Sop = (Sops + Sopo) * f**2
    elif noisemodel == 'newdrs':  # lisalight, to be used with lisaL = 1Gm, lisaP = 2
        Spm = 6.00314e-48 * f**(-2)                                 # 4.6e-15 m/s^2/sqrt(Hz)

        Sops = 6.15e-38 * (model.lisaL/model.defaultL)**2 * (model.defaultD/model.lisaD)**4 * (model.defaultP/model.lisaP)      # 11.83 pm/sqrt(Hz)
        Sopo = 2.81e-38                                                                                                         # 8 pm/sqrt(Hz)
        Sop = (Sops + Sopo) * f**2
    elif noisemodel == 'reddrs':  # used for lisalight C6
        Spm = 6.0e-48 * (1 + (1e-4/f)) * f**(-2)    # 4.61e-15 m/s^2/sqrt(Hz)

        Sops = 6.17e-38 * (model.lisaL/model.defaultL)**2 * (model.defaultD/model.lisaD)**4 * (model.defaultP/model.lisaP)
        Sopo = 2.76e-38
        Sop = (Sops + Sopo) * f**2
    elif noisemodel == 'lpf':
        # LPF CBE curve from ? via Oliver; the coefficient in front is [10^-14.09 * c / (2 pi)]^2
        Spm = 1.86208e-47 * (1.0 + (f/10**-3.58822)**-1.79173) * (1.0 + (f/10**-2.21652)**3.74838) * f**(-2)

        # see LISA-variant Mathematica notebook; include only shot-noise correction due to armlength
        # constants are 7.7 and 5.15 pm (formerly 9.25) squared and multiplied by (2 pi / c)^2
        # notice the standard curve includes 18 pm of which 35% is margin
        Sop = (1.16502e-38 + 2.60435e-38*(model.lisaL/model.defaultL)**2) * f**2
    elif noisemodel == 'wind':
        Spm = 1.76e-50 * f**-0.75 * f**(-2)
        Sop = 1.42319e-37 * (model.lisaL/model.defaultL)**2 * (1.0 + (f/0.002)**-4) * f**2
    elif noisemodel == 'windnew':
        Spm = 1.76e-50 / 12 * f**-0.75 * f**(-2)
        Sop = 1.42319e-37 * (model.lisaL/model.defaultL)**2 * (model.defaultD/model.lisaD)**4 * (model.defaultP/model.lisaP) * (1.0 + (f/0.002)**-4) * f**2
    elif noisemodel == 'ax50':
        Spm = 50 * 2.53654e-48 * (1.0 + (f/1.0e-4)**-1) * (1.0 + (f/0.008)**4) * f**(-2)
        Sop = 1.42319e-37 * (model.lisaL/model.defaultL)**2 * (1.0 + (f/0.002)**-4) * f**2
    else:
        raise NotImplementedError(noisemodel)

    return Spm, Sop


def phinneyswitch(Sinst,Sgwdb,switch):
    return N.minimum(Sinst*switch,Sinst + Sgwdb)


class phinneybackground(object):
    def __init__(self,Sh=1.4e-44,dNdf=2e-3,koverT=1.5,Sh_exp=-7.0/3.0,dNdf_exp=-11.0/3.0,dNdf_func=N.exp,dNdf_switch=phinneyswitch):
        self.Sh,   self.Sh_exp   = Sh,   Sh_exp
        self.dNdf, self.dNdf_exp = dNdf, dNdf_exp
        self.koverT = koverT / (365.25 * 24 * 3600)
        self.dNdf_func, self.dNdf_switch = dNdf_func, dNdf_switch

    def __call__(self,f,Sinst = None):
        Sgwdb = self.Sh * f**self.Sh_exp
        dNdf  = self.dNdf * f**self.dNdf_exp

        if Sinst != None:
            return self.dNdf_switch(Sinst,Sgwdb,self.dNdf_func(self.koverT*dNdf))
        else:
            return Sgwdb



# note: not updated for shorter armlengths, LPF noise, better optical noise model
def lisanoise(f,noisemodel=None,includewd=None):
    if noisemodel == None: noisemodel = model.noisemodel
    if includewd == None: includewd = model.lisaWD

    if noisemodel == 'cutler':
        # compare to Eq. (25) of Barack and Cutler Phys.Rev. D 70, 122002 (2004)
        # their Sh, defined by <n n> = 3/40 Sh, is 6.12e-51 f**-4,
        # so the <n n> noise is enhanced (as "seen" by signals) because of signal averaging
        #
        # that's the same as defining Sh by <n n> = 1/2 Sh as 9.18e-52 f**-4 (as we do)
        # and then enhancing it by 20/3 because of signal averaging
        #
        # either way the S_h used in averaged SNR expressions is 6.12e-51
        # (and I hope I never have to think about this again)

        Sh = (20.0/3.0)*(9.18e-52 * f**-4 + 1.59e-41 + 9.18e-38 * f**2)

        if includewd == True:
            pb = phinneybackground()
            return pb(f,Sh)
        elif includewd == None:
            return Sh
        else:
            raise NotImplementedError
    else:
        optscale = (model.lisaL/model.defaultL)**2 * (model.defaultD/model.lisaD)**4 * (model.defaultP/model.lisaP)

        if noisemodel == 'lisareq':
            Sa = 3e-15 * N.sqrt(1.0 + (f/1.0e-4)**-1) * N.sqrt(1.0 + (f/0.008)**4)
            So = 18e-12 * optscale * N.sqrt(1 + (f/0.002)**-4)
        elif noisemodel == 'lpf':
            Sa = 10**-14.09 * N.sqrt((1.0 + (f/10**-3.58822)**-1.79173) * (1.0 + (f/10**-2.21652)**3.74838))
            So = N.sqrt((7.7e-12)**2  * optscale + (5.15e-12)**2)
        elif noisemodel == 'toy':
            Sa = 3e-15
            So = N.sqrt((1.6e-11)**2  * optscale + (1.2e-11)**2)
        elif noisemodel == 'newtoy':
            Sa = 3e-15
            So = 2e-11
        elif noisemodel == 'newlpf':
            Sa = 5.3e-15 * (1.0 + (f/1.8e-4)**-1)
            So = N.sqrt((1.18e-11)**2 * optscale + (8.0e-12)**2)
        elif noisemodel == 'newdrs-wrong':
            Sa = 4.6e-15
            So = N.sqrt((8.36e-12)**2 * optscale + (8.0e-12)**2)
        elif noisemodel == 'newdrs':
            Sa = 4.6e-15
            So = N.sqrt((1.18e-11)**2 * optscale + (8.0e-12)**2)
        elif noisemodel == 'wind':
            Sa = 2.5e-16 * f**-0.75
            So = 18e-12 * optscale * N.sqrt(1 + (f/0.002)**-4)
        elif noisemodel == 'windnew':
            Sa = 2.5e-16 / 3.464 * f**-0.75     # with Curt's reduced model of noise (2 sqrt(3) in rms due to wind angle fluctuations)
            So = 18e-12 * optscale * N.sqrt(1 + (f/0.002)**-4)
        elif noisemodel == 'ax50':
            Sa = 50 * 3e-15 * N.sqrt(1.0 + (f/1.0e-4)**-1) * N.sqrt(1.0 + (f/0.008)**4)
            So = 18e-12 * optscale * N.sqrt(1 + (f/0.002)**-4)
        else:
            raise NotImplementedError(noisemodel)

        Sac = Sa * 2.0 / (2.0 * math.pi * f)**2

        c = 299792458; L = model.lisaL * c

        # how to fit WD noise in here? Let's reason for X first...
        # TDI-X WD noise could be added to Sop (as defined in lisanoises) after dividing by 16 sin(x)^2
        # now Sop = So^2 * (2*pi/c)^2 f^2; hence we can add Swd/(16 sin(x)^2) / (2*pi*f/c)^2 to So
        # note that confusion noise is L dependent because of subtraction... but it probably still
        # makes sense to factor the L and add L^2 * Swd / (16 sin(x)^2 x^2/c^2) with x = 2 pi L f

        # transfer function
        ft = 0.5 / model.lisaL
        T2 = 1.0 + (f/(0.41 * ft))**2

        if includewd == None:
            Swd = 0
        elif includewd == 'cutler':
            pb = phinneybackground()
            return pb(f,(20.0/3.0) * T2 * (Sac**2 + So**2) / L**2)
        elif isinstance(includewd,phinneybackground):
            return includewd(f,(20.0/3.0) * T2 * (Sac**2 + So**2) / L**2)
        else:
            x = 2.0 * math.pi * model.lisaL * f
            Swd = makewdnoise(f,includewd,obs='X') * L**2 / (16.0 * N.sin(x)**2 * x**2)

        return (20.0/3.0) * T2 * (Sac**2 + So**2 + Swd) / L**2


def simplesnr(f,h,i=None,years=1,noisemodel=None,includewd=None):
    if i == None:
        h0 = h * math.sqrt(16.0/5.0)    # rms average over inclinations
    else:
        h0 = h * math.sqrt((1 + math.cos(i)**2)**2 + (2*math.cos(i))**2)

    return h0 * math.sqrt(years * 365.25*24*3600) / math.sqrt(lisanoise(f,noisemodel,includewd))


wdnoise = {}

# fit between 1e-4 and 5e-3 for X, between 1e-4 and 4e-4 for AET; all SNR = 5
wdnoise['tau2']   = (('rat42',[-1.2503, -13.3508, -94.1852, -296.6416, -313.8596, 4.9418, 6.1323]),
                     ('rat42',[-1.2599, -13.8309, -97.7703, -311.5419, -336.4092, 5.0691, 6.4637]))
wdnoise['opt']    = (('rat42',[-1.0865, -11.2113, -83.9764, -271.5378, -287.9153, 4.8456, 5.8931]),
                     ('rat42',[-1.0781, -11.3477  -85.3638, -279.6701, -301.9440, 4.9496, 6.1504]))
wdnoise['pess']   = (('rat42',[-1.2649, -13.5895, -95.5196, -301.0872, -319.7566, 4.9740, 6.2117]),
                     ('rat42',[-1.2813, -14.1556, -99.5091, -316.7877, -342.7881, 5.1004, 6.5392]))
wdnoise['hybrid'] = (('poly4',[-2.4460, -33.4121,-171.5341, -390.7209, -373.5341]),
                     ('poly4',[-2.7569, -38.0938,-197.8030, -455.9119, -433.8260]))

def makewdnoise(f,wdstyle,obs='X'):
    if wdstyle == 'mldc':
        x = 2.0 * math.pi * model.lisaL * f
        t = 4 * x**2 * N.sin(x)**2 * (1.0 if obs == 'X' else 1.5)

        return t * ( N.piecewise(f,(f >= 1.0e-4  ) & (f < 1.0e-3  ),[lambda f: 10**-44.62 * f**-2.3, 0]) + \
                     N.piecewise(f,(f >= 1.0e-3  ) & (f < 10**-2.7),[lambda f: 10**-50.92 * f**-4.4, 0]) + \
                     N.piecewise(f,(f >= 10**-2.7) & (f < 10**-2.4),[lambda f: 10**-62.8  * f**-8.8, 0]) + \
                     N.piecewise(f,(f >= 10**-2.4) & (f < 10**-2.0),[lambda f: 10**-89.68 * f**-20.0,0])     )
    elif wdstyle in wdnoise:
        mod, p = wdnoise[wdstyle]
        p = p[0] if obs == 'X' else p[1] # assume AE if not X
        y = N.log10(f)

        if mod == 'rat42':
            return 10.0**( (p[0]*y**4+p[1]*y**3+p[2]*y**2+p[3]*y+p[4])/(y**2+p[5]*y+p[6]) )
        elif mod == 'poly4':
            return 10.0**( p[0]*y**4+p[1]*y**3+p[2]*y**2+p[3]*y+p[4] )
        else:
            raise NotImplementedError
    else:
        if '.txt' in wdstyle:
            conf = N.loadtxt(wdstyle)
            conf[N.isnan(conf[:,1]),1] = 0

            return N.interp(f,conf[:,0],conf[:,1])
        else:
            raise NotImplementedError


# from lisatools makeTDIsignal-synthlisa2.py
def noisepsd_X(frequencydata,includewd=None):
    if includewd == None: includewd = model.lisaWD

      x = 2.0 * math.pi * model.lisaL * f

    Spm, Sop = lisanoises(f)

    Sx = 16.0 * N.sin(x)**2 * (2.0 * (1.0 + N.cos(x)**2) * Spm + Sop)
    # Sxy = -4.0 * N.sin(2*x) * N.sin(x) * (Sop + 4.0*Spm)
    # Sa = Sx - Sxy

    if includewd != None:
        Sx += makewdnoise(f,includewd,'X')

    return Sx


def noisepsd_AE(f,includewd=None):
    if includewd == None: includewd = model.lisaWD

    x = 2.0 * math.pi * model.lisaL * f

    Spm, Sop = lisanoises(f)

    Sa = 8.0 * N.sin(x)**2 * (2.0 * Spm * (3.0 + 2.0*N.cos(x) + N.cos(2*x)) +
                              Sop * (2.0 + N.cos(x)))

    if includewd != None:
        Sa += makewdnoise(f,includewd,'AE')

    return Sa

# TO DO: currently not including WD background here... probably OK
def noisepsd_T(f):
    x = 2.0 * math.pi * model.lisaL * f

    Spm, Sop = lisanoises(f)

    St = 16.0 * Sop * (1.0 - N.cos(x)) * N.sin(x)**2 + 128.0 * Spm * N.sin(x)**2 * N.sin(0.5*x)**4

    return St

"""
# === note on FFT normalization: it's always fun, isn't it? ===
#
# numpy.fft.fft and numpy.fft.ifft are inverses of each other, but
# numpy.fft.fft(X)/sqrt(N) has the same sum-squared as X
# numpy.fft.rfft is a subset of numpy.fft.fft, and numpy.fft.irfft is its inverse
#
# now Parseval says
#   int |h(f)|^2 df = int h(t)^2 dt
# and discretizing
#   sum |h(f_i)|^2 / T = sum h(t_i)^2 T / N
# hence
#   sum |h(f_i)|^2 (N/T^2) = sum |fft(X)_i|^2 / N
# whence
#   h(f_i) = fft(X)_i / (N df) -> int |h(f)|^2 df = sum |h(f_i)|^2 df
#                                                 = sum |fft(X)_i|^2 / (N^2 df)
#
# we're using a compact representation of the frequency-domain signal
# with no negative frequencies, and only 2*N - 1 components; if we were really
# integrating over all frequencies, we should take half the DC and Nyquist contributions,
# but that won't be the case here (and the trapezoidal rule would take care of it anyway)
#
# last question is what is returned by FastBinary.cpp. Since
# irfft(ret(f_i)) * N seems to be the correct time-domain signal,
# then ret(f_i) must be h(f_i) * df, and therefore the SNR integral is
#
# 4 Re int |h(f)|^2 / Sn(f) df = 4 sum |ret(f_i) / df|^2 / Sn(f_i) df = (4/df) sum |ret(f_i)|^2 / Sn(f_i)
#
# now how's h(f_i) related to S_h(f)? I have
#
# int_0^\infty S_h(f) = (1/T) \int_0^T |h(t)|^2 dt = (2/T) \int_0^\infty |h(f)|^2 df
# so S_h(f_i) = (2/T) |h(f_i)|^2 = (2/T) |ret(f_i)|^2 / df^2 = (2/df) |ret(f_i)^2|


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

    L = 2.5 * 10 ** 9  # Length of LISA arm
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


def get_sensitivity(f, sens_fn="lisasens", return_type="PSD", *args, **kwargs):
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
