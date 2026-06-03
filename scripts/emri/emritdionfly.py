from few.utils.utility import get_polarization_angle, get_viewing_angles
from typing import Optional

from gpubackendtools.interpolate import CubicSplineInterpolant
from fastlisaresponse.tdionfly import TDTDIonTheFly
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform
from astropy.coordinates import SkyCoord
import astropy.units as u
import h5py
import numpy as np
try:
    import cupy as xp
    backend = 'cuda12x'
except (ImportError, ModuleNotFoundError):
    import numpy as xp
    backend = 'cpu'

# from lisatools.globalfit.preprocessing import L1ProcessingStep
from preprocessing import L1ProcessingStep
from phentax.waveform import IMRPhenomTHM
from lisatools.detector import L1Orbits

from lisaconstants import ASTRONOMICAL_YEAR
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "notebook"])

FIGSIZE = (10, 6)


import logging
import sys

# Configure logging to display messages in the notebook
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)


# ...existing code...
def ecliptic_to_icrs(lambda_ecl, beta_ecl):
    """
    Convert ecliptic coordinates (lambda, beta) in radians to ICRS (RA, Dec) in radians.

    Parameters
    ----------
    lambda_ecl, beta_ecl : float or array-like
        Ecliptic longitude (lambda) and latitude (beta) in radians.

    Returns
    -------
    ra, dec : tuple
        Right ascension and declination in radians (same shape as inputs).
    """
    ecl = SkyCoord(lon=lambda_ecl * u.rad, lat=beta_ecl * u.rad, frame='barycentrictrueecliptic')
    icrs = ecl.transform_to('icrs')
    return icrs.ra.rad, icrs.dec.rad

        
class EMRITDIonFly:

    def __init__(
        self,
        wave_gen,
        orbits,
        tdi_config,
        dt,
        Tobs,
        t0
    ):
        self.wave_gen = wave_gen
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.dt = dt
        self.T = Tobs
        self.t0 = t0
    
    @property
    def dt(self) -> float:
        """dt value from data."""
        return self._dt
    
    @dt.setter
    def dt(self, dt: float):
        self._dt = dt
        self.sampling_frequency = 1 / dt

    def __call__(self, 
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        x0: float,
        dist: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        Phi_phi0: float,
        Phi_theta0: float,
        Phi_r0: float,
        *add_args: Optional[tuple],
        include_minus_mkn=True,
        **kwargs: Optional[dict],
    ):
        theta, phi = get_viewing_angles(qS, phiS, qK, phiK)
        psi = get_polarization_angle(qS, phiS, qK, phiK)

        lam = phiS
        beta = np.pi / 2 - qS
        
        ra, dec = ecliptic_to_icrs(lam, beta)

        # psi = np.pi - psi
        Kerr_wave = self.wave_gen(
            m1,
            m2,
            a,
            p0,
            e0,
            x0,
            theta,
            phi,
            dist=dist,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            T=self.T,
            dt=self.dt,
            return_sparse_holder=True,
            include_minus_mkn=include_minus_mkn
        )

        mode_amp_phase = np.unwrap(np.angle(Kerr_wave.teuk_modes), axis=0)
        mode_amp_amp = np.abs(Kerr_wave.teuk_modes)

        ylm_phase = np.angle(Kerr_wave.ylms)
        ylm_amp = np.abs(Kerr_wave.ylms)
        _mode_phase = (
            Kerr_wave.ms[None, :] * Kerr_wave.phases[:, 0][:, None]
            + Kerr_wave.ks[None, :] *  Kerr_wave.phases[:, 1][:, None]
            + Kerr_wave.ns[None, :] *  Kerr_wave.phases[:, 2][:, None]
        )

        AMP_FACTOR = 1/2. # CHECK THIS
        if include_minus_mkn:
            # m >= 0
            keep_minus_m = Kerr_wave.ms != 0
            phase_m_zero_and_above = (_mode_phase - ylm_phase[:Kerr_wave.ms.shape[0]] - mode_amp_phase)
            phase_m_below_zero = -phase_m_zero_and_above[:, keep_minus_m]
            mode_phase = np.concatenate([phase_m_zero_and_above, phase_m_below_zero], axis=-1).T

            amp_m_zero_and_above = AMP_FACTOR * mode_amp_amp * ylm_amp[:_mode_phase.shape[1]]    
            amp_m_below_zero = (AMP_FACTOR * mode_amp_amp * ylm_amp[_mode_phase.shape[1]:])[:, keep_minus_m]
            mode_amp = np.concatenate([amp_m_zero_and_above, amp_m_below_zero], axis=-1).T
        
        else:
            mode_phase = (_mode_phase - ylm_phase[:Kerr_wave.ms.shape[0]] - mode_amp_phase).T
            mode_amp = AMP_FACTOR * (mode_amp_amp * ylm_amp[:Kerr_wave.ms.shape[0]]).T

        t_arr_in = self.t0 + np.repeat(Kerr_wave.t_arr[:, None], mode_phase.shape[0], axis=-1).T
        t_arr_tdi = t_arr_in[:, 1:-1]
        dt = 1.0
        sampling_frequency = 1 / dt
        num_sub = mode_amp.shape[0]

        self.tdi_gen = TDTDIonTheFly(t_arr_tdi, mode_amp, mode_phase, sampling_frequency, num_sub, t_input=t_arr_in, tdi_config=self.tdi_config, orbits=self.orbits)

        inc = np.zeros(num_sub)
        psi_in = np.full(num_sub, psi)
        ra_in = np.full(num_sub, ra)
        dec_in = np.full(num_sub, dec)
        # beta = np.full(num_sub, qS)
        # output_tdi_fly = self.tdi_gen(inc, psi_in, lam, beta, return_spline=True)
        output_tdi_fly = self.tdi_gen(inc, psi_in, ra_in, dec_in, return_spline=True)

        return output_tdi_fly
