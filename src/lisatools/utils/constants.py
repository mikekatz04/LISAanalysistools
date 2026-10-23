"""Physical and LISA-specific numeric constants used throughout the package.

The physical constants (``MSUN_SI``, ``YRSID_SI``, ``AU_SI``, ``C_SI``, ``G_SI``, ``GMSUN``,
``MTSUN_SI``, ``MRSUN_SI``, ``PC_SI``, ``EPS_RAD``) are sourced from ``lisaconstants``
(pinned 2.0.2) so every package in the tree shares one set of definitions; the trailing
comments show the lisaconstants 2.0.2 values. The C/C++/CUDA side reads the same numbers from
the generated ``cutils/lisaconstants_values.h`` (:mod:`lisatools.utils.lisaconstants_header`).

The rest are literals: mathematical constants (``PI``, ``SQRT2``, ...), legacy orbit
parameters (``F0``, ``Omega0``, ``eorbit``, ``ConstOmega``), the nominal 2.5e9 m arm length
(``L_SI`` / ``lisaL``, light time ``lisaLT``), the transfer frequency ``fstar`` and legacy
optical-metrology noise levels. All values are SI unless noted.

Example:
    >>> from lisatools.utils.constants import YRSID_SI, C_SI
    >>> C_SI
    299792458.0
"""

import lisaconstants as lc

# Values in the comments are lisaconstants 2.0.2's (the pinned version). The
# C/C++/CUDA side reads the same numbers from the generated
# cutils/lisaconstants_values.h (lisatools.utils.lisaconstants_header).
MSUN_SI = lc.SOLAR_MASS  # 1.9884754153381438e30
YRSID_SI = lc.ASTRONOMICAL_YEAR  # 31558149.763545595
AU_SI = lc.ASTRONOMICAL_UNIT  # 149597870700.0
C_SI = lc.SPEED_OF_LIGHT  # 299792458.0
G_SI = lc.GRAVITATIONAL_CONSTANT  # 6.67408e-11
GMSUN = lc.SOLAR_MASS_PARAMETER  # 1.3271244e20
MTSUN_SI = lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**3  # 4.9254909476412675e-06
MRSUN_SI = lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**2  # 1476.6250380501249
PC_SI = lc.PARSEC  # 3.085677581491367e16
PI = 3.141592653589793238462643383279502884
PI_2 = 1.570796326794896619231321691639751442
PI_3 = 1.047197551196597746154214461093167628
PI_4 = 0.785398163397448309615660845819875721
SQRTPI = 1.772453850905516027298167483341145183
SQRTTWOPI = 2.506628274631000502415765284811045253
INVSQRTPI = 0.564189583547756286948079451560772585
INVSQRTTWOPI = 0.398942280401432677939946059934381868
GAMMA = 0.577215664901532860606512090082402431
SQRT2 = 1.414213562373095048801688724209698079
SQRT3 = 1.732050807568877293527446341505872367
SQRT6 = 2.449489742783178098197284074705891392
INVSQRT2 = 0.707106781186547524400844362104849039
INVSQRT3 = 0.577350269189625764509148780501957455
INVSQRT6 = 0.408248290463863016366214012450981898
# Legacy orbit literals: F0 ~ 1 / YRSID_SI [Hz]; Omega0 = 2 pi F0 [rad/s];
# eorbit = L / (2 sqrt(3) AU), the orbital eccentricity of the nominal
# 2.5e9 m constellation; ConstOmega = Omega0 rounded.
F0 = 3.168753578687779e-08
Omega0 = 1.9909865927683788e-07
# Nominal arm length [m].
L_SI = 2.5e9
eorbit = 0.004824185218078991
ConstOmega = 1.99098659277e-7

#### Armlength
lisaL = 2.5e9  # LISA's arm meters
lisaLT = lisaL / C_SI  # LISA's armn in sec

#### Noise levels
### Optical Metrology System noise
## Decomposition
Sloc = (1.7e-12) ** 2  # m^2/Hz
Ssci = (8.9e-12) ** 2  # m^2/Hz
Soth = (2.0e-12) ** 2  # m^2/Hz

######################
# Physical constants #
######################

# Mass of Jupiter
Mjup = 1.898e27

# Obliquity of the ecliptic in radians (lc.OBLIQUITY is in degrees, IAU 2006).
EPS_RAD = PI / 180. * lc.OBLIQUITY

#################
# LISA constant #
#################

# Transfer frequency c / (2 pi L) [Hz]
fstar = C_SI / (lisaL * 2 * PI)
