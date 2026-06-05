from typing import Optional

from gpubackendtools.interpolate import CubicSplineInterpolant
from fastlisaresponse.tdionfly import TDTDIonTheFly
import os 

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
from phentax.waveform import IMRPhenomTHM

from lisaconstants import ASTRONOMICAL_YEAR
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "notebook"])
except ModuleNotFoundError:
    pass

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

        
class MBHTDIonFly:

    def __init__(
        self,
        wave_gen,
        orbits,
        tdi_config,
        dt,
        Tobs,
        t0, 
        dt_min=0.1,
        t_min=0.0, 
        waveform_duration=None,
    ):
        self.wave_gen = wave_gen
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.dt = dt
        self.dt_min = dt_min
        self.t_min = t_min
        self.T = Tobs
        self.t0 = t0
        self.waveform_duration = waveform_duration
    
    @property
    def dt(self) -> float:
        """dt value from data."""
        return self._dt
    
    @dt.setter
    def dt(self, dt: float):
        self._dt = dt
        self.sampling_frequency = 1 / dt

    def __call__(self, 
        m1,
        m2, 
        s1z,
        s2z,
        distance,
        phi_ref,
        inclination,
        ra,
        dec,
        psi,
        t_merge,
        upsample_t_arr: np.ndarray = None,
        combine: bool = False,
        dt_tdi_eval = 10.0,  # tunable: denser than the adaptive grid, sparser than dt=2.5s
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ):
        dt_in = min(self.dt_min, self.dt)

        if self.waveform_duration is None:
            waveform_duration = t_merge
        else:
            waveform_duration = self.waveform_duration

        new_times, new_mask, sc_amp, sc_phase = self.wave_gen.compute_strain_components_amp_phase(
            m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
            delta_t=self.dt_min, t_min=-waveform_duration, t_ref=0.0, 
        )

        mode_amp = sc_amp / 2.0          # AMP_FACTOR = 1/2
        mode_phase = np.pi - sc_phase   # negate and add π

        # this will be close by an integer multiple of dt_min
        _new_times = xp.asarray(new_times[new_mask] + t_merge + self.t0)

        nmodes = self.wave_gen.num_modes
        new_times_arr = xp.repeat(_new_times[None, :], nmodes, axis=0)

        amp = xp.asarray(mode_amp[0][:, new_mask[0]])
        phase = xp.asarray(mode_phase[0][:, new_mask[0]])

        # coarse_times, coarse_mask = wave_gen.get_coarse_grained_time_array()
        # coarse_times = xp.asarray(coarse_times + t_merge)
        # coarse_times_arr = xp.repeat(coarse_times, nmodes, axis=0)

        sampling_frequency = 1 / self.dt

        tdi_buffer = int(1000 / self.dt)  # seconds #todo @Mike: how many samples do we have to discard here? I kept getting out of splines error for smaller values

        eval_t_arr = new_times_arr[:, tdi_buffer:-tdi_buffer]
        #eval_t_arr = coarse_times_arr[:, 1:-10][:, -buffer_size:]
        # eval_t_arr = xp.repeat(eval_t_uniform[None, :], nmodes, axis=0)

        tdi_gen = TDTDIonTheFly(eval_t_arr, amp, phase, sampling_frequency=sampling_frequency, num_sub=nmodes, t_input=new_times_arr, tdi_config=self.tdi_config, orbits=self.orbits)

        inc = xp.full(nmodes, 0.0)  # inclination is already applied in the spherical harmonic
        polarization = xp.full(nmodes, psi)
        ra_arr = xp.full(nmodes, ra)
        dec_arr = xp.full(nmodes, dec)

        output = tdi_gen(inc, polarization, ra_arr, dec_arr, return_spline=True)

        if upsample_t_arr is None:
            return output
        
        new_tdi = np.zeros((output.t_arr.shape[0], 3, upsample_t_arr.shape[-1]))
        keep = (upsample_t_arr >= output.t_arr.min().item()) & (upsample_t_arr <= output.t_arr.max().item())
        new_tdi[:, :, keep] = output.eval_tdi(upsample_t_arr[keep])

        if combine:
            return new_tdi.sum(axis=0)
        
        return new_tdi
        

if __name__ == "__main__":
    from preprocessing import L1ProcessingStep
    from lisatools.detector import L1Orbits
    from lisaconstants import ASTRONOMICAL_YEAR
    from lisatools.utils.constants import YRSID_SI
    from fastlisaresponse import ResponseWrapper
    from fastlisaresponse.tdiconfig import TDIConfig
    from fastlisaresponse.response import icrs_to_ecliptic
    from phentax.waveform import IMRPhenomTHM 

    hms = [21, 33, 44]
    negative_modes = True
    tlowfit = True # use a fit to set the starting time of the root finder used in t(f)
    tol = 1e-12 # root finding tolerance
    Tobs = 1 * ASTRONOMICAL_YEAR / 12
    delta_t = 2.5

    wave_gen = IMRPhenomTHM(
        higher_modes=hms,
        include_negative_modes=negative_modes, # negative m modes will be produced by simmetry
        t_low_fit=tlowfit,
        coarse_grain=True, # if false it will generate the waveform on a dense time grid with the specified timestep
        atol=tol,
        rtol=tol,
        T=Tobs,
    )

    path = "/Users/mlkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/"

    source_types = ['mbhb']
    ID = 0
    source_ids = [ID]

    backend = "cpu"
        
    loader = L1ProcessingStep(
        L1_folder=path,
        source_types=source_types,
        source_ids=dict(mbhb=source_ids),
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=backend, frame="icrs"),  # equatorial coords (ra/dec)
        verbose=True,
        do_plots=False
    )
    full_catalogue = loader.catalogue['MBHB']
    orbits = loader.orbits 
    dt = loader.dt
    tdi_config = TDIConfig('2nd generation')
    t0 = full_catalogue[ID]["TimeReferenceSSBFrame"]
    mbh_tdi_fly_gen = MBHTDIonFly(
        wave_gen,
        orbits,
        tdi_config,
        dt,
        Tobs,
        t0
    )

    m1 = float(full_catalogue[ID]['PrimaryMassSSBFrame'][()].squeeze())
    m2 = float(full_catalogue[ID]['SecondaryMassSSBFrame'][()].squeeze())
    s1z = float(full_catalogue[ID]['PrimarySpinCompZ'][()].squeeze())
    s2z = float(full_catalogue[ID]['SecondarySpinCompZ'][()].squeeze())
    distance = float(full_catalogue[ID]['LuminosityDistance'][()].squeeze())
    redshift = float(full_catalogue[ID]['Redshift'][()].squeeze())
    inclination = float(full_catalogue[ID]['InclinationAngle'][()].squeeze())
    phi_ref = float(full_catalogue[ID]['PhaseReferenceSourceFrame'][()].squeeze())
    f22_start =  float(full_catalogue[ID]['GW22FrequencySSBFrame'][()].squeeze())
    f22_ref =  float(full_catalogue[ID]['GW22FrequencySSBFrame'][()].squeeze())
    ra = float(full_catalogue[ID]['RightAscension'][()].squeeze())
    dec = float(full_catalogue[ID]['Declination'][()].squeeze())
    psi = float(full_catalogue[ID]['PolarisationAngle'][()].squeeze())
    snr = float(full_catalogue[ID]['EstimatedSNR'][()].squeeze())
    tmerg = float(full_catalogue[ID]['TimeCoalescencePetersSSBFrame'][()].squeeze())
    tmerg_tphm = float(full_catalogue[ID]['TimeCoalescencePhenomTPHMSSBFrame'][()].squeeze())
    merger_time = full_catalogue[ID]["TimeReferenceSSBFrame"] + full_catalogue[ID]["TimeCoalescencePhenomTPHMSSBFrame"]


    tdifly_output = mbh_tdi_fly_gen(
        m1,
        m2, 
        s1z,
        s2z,
        distance,
        inclination,
        phi_ref,
        f22_start,
        f22_ref,
        ra,
        dec,
        psi,
        merger_time,
    )

    t_min = tdifly_output.t_arr[0, 0]
    t_max = tdifly_output.t_arr[0, -1]
    keep = (loader.times >= t_min) & (loader.times <= t_max)
    t_new = loader.times[keep]
    tdi_new = np.zeros((3, loader.times.shape[0]))
    tdi_new[:, keep] = tdifly_output.eval_tdi(t_new).sum(axis=0)

    from lisatools.datacontainer import DataResidualArray
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.sensitivity import XYZ2SensitivityMatrix, AET2SensitivityMatrix
    from lisatools.domains import TDSettings, TDSignal, FDSignal, FDSettings
    from lisatools.utils.utility import AET
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('default')

    dt = 2.5
    backend = "cpu"
    ind_choice = 1
    # data_inj_all = np.asarray(np.load(f"../cd1l-validation/notebooks/inj_arr_mbh.npy"))
    N = loader.data.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    Nf = len(freqs)
    from scipy import signal
    window = signal.windows.tukey(N, alpha=0.05)
    keep_2 = np.where((freqs > 0.5e-3) & (freqs < 25e-3))[0]
    assert len(keep_2) > 2
    ind_min = keep_2[0]
    ind_max = keep_2[-1]
    fd_set = FDSettings(Nf, df, ind_min=ind_min, ind_max=ind_max, force_backend=backend)
    data_inj_all_fd = TDSignal(loader.data, settings=td_set).fft(settings=fd_set, window=window)
    injection = DataResidualArray(data_inj_all_fd)
    tdi_output_fd = TDSignal(tdi_new, settings=td_set).fft(settings=fd_set, window=window)
    template = DataResidualArray(tdi_output_fd)
    sens_mat = XYZ2SensitivityMatrix(template.data_res_arr.settings, model="scirdv1")

    analysis = AnalysisContainer(injection, sens_mat)
    # fig, ax = analysis.loglog()
    # plt.show()
    check = analysis.template_inner_product(template)

    check = analysis.template_inner_product(template)
    check_snr = analysis.template_snr(template)
    check_ll = analysis.template_likelihood(template)
    overlap = analysis.template_inner_product(template, normalize=True)
    mismatch = 1.0 - overlap
    print(f"Binary {ind_choice} highest frequency, Log Likelihood:", check_ll, "Mismatch:", mismatch, "Overlap:", overlap, "SNR (observed, optimal):", check_snr)

    breakpoint()
