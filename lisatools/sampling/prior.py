import numpy as np
from scipy import stats

from ..utils.constants import *
from ..sensitivity import get_sensitivity

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    pass


class SNRPrior:
    def __init__(self, rho_star, use_cupy=False):
        self.rho_star = rho_star
        self.use_cupy = use_cupy

    def pdf(self, rho):
        
        xp = np if not self.use_cupy else cp

        p = xp.zeros_like(rho)
        good = rho > 0.0
        p[good] = 3 * rho[good] / (4 * self.rho_star ** 2 * (1 + rho[good] / (4 * self.rho_star)) ** 5)
        return p

    def logpdf(self, rho):
        xp = np if not self.use_cupy else cp
        return xp.log(self.pdf(rho))

    def cdf(self, rho):
        xp = np if not self.use_cupy else cp
        c = xp.zeros_like(rho)
        good = rho > 0.0
        c[good] = 768 * self.rho_star ** 3 * (1 / (768. * self.rho_star ** 3) - (rho[good] + self.rho_star)/(3. * (rho[good] + 4 * self.rho_star) ** 4))
        return c

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        u = xp.random.rand(*size)

        rho = (-4*self.rho_star + xp.sqrt(-32*self.rho_star**2 - (32*(-self.rho_star**2 + u*self.rho_star**2))/(1 - u) + 
      (3072*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)*
         (self.rho_star**4 - u*self.rho_star**4))/
       ((-1 + u)**2*xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
            xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
              3131031158784*u**3*self.rho_star**12))) + 
      xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
          xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
            3131031158784*u**3*self.rho_star**12))/
       (3.*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)))/2.
     + xp.sqrt(32*self.rho_star**2 + (32*(-self.rho_star**2 + u*self.rho_star**2))/(1 - u) - 
      (3072*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)*
         (self.rho_star**4 - u*self.rho_star**4))/
       ((-1 + u)**2*xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
            xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
              3131031158784*u**3*self.rho_star**12))) - 
      xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
          xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
            3131031158784*u**3*self.rho_star**12))/
       (3.*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)) + 
      (2048*self.rho_star**3 - (2048*u*self.rho_star**3)/(-1 + u))/
       (4.*xp.sqrt(-32*self.rho_star**2 - (32*(-self.rho_star**2 + u*self.rho_star**2))/(1 - u) + 
           (3072*2**0.3333333333333333*
              xp.cbrt(-1 + 3*u - 3*u**2 + u**3)*(self.rho_star**4 - u*self.rho_star**4)
              )/
            ((-1 + u)**2*xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
                 xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
                   3131031158784*u**3*self.rho_star**12))) + 
           xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
               xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
                 3131031158784*u**3*self.rho_star**12))/
            (3.*2**0.3333333333333333*
              xp.cbrt(-1 + 3*u - 3*u**2 + u**3)))))/2.)

        return rho


class AmplitudeFromSNR:
    def __init__(self, L, Tobs, **noise_kwargs):
        self.f_star = 1 / (2. * np.pi * L) * C_SI
        self.Tobs = Tobs
        self.noise_kwargs = noise_kwargs

    def __call__(self, rho, f0):
        factor = 1./2. * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / get_sensitivity(f0, sens_fn="noisepsd_AE", **self.noise_kwargs))
        amp = rho / factor
        return (amp, f0)

    def forward(self, amp, f0):
        factor = 1./2. * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / get_sensitivity(f0, sens_fn="noisepsd_AE", **self.noise_kwargs))
        rho = amp * factor
        return (rho, f0)