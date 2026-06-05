#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    pass

from lisatools.detector import ESAOrbits, EqualArmlengthOrbits
from lisaconstants import ASTRONOMICAL_YEAR
from lisatools.utils.constants import YRSID_SI
from fastlisaresponse import ResponseWrapper
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.response import icrs_to_ecliptic

from phentax.waveform import IMRPhenomTHM 
from mbhtdionfly import MBHTDIonFly

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.domains import TDSettings, TDSignal, FDSettings, FDSignal, WDMSettings, WDMSignal

from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform

from eryn.moves import StretchMove
from eryn.ensemble import EnsembleSampler
from eryn.utils import PeriodicContainer

from eryn.state import State
from eryn.backends import HDFBackend
import os
# credit Michael Katz and Alessandro Santini (with internal code contrubtions in docs)


    # HIGHLY RECOMMEND RUNNING THESE THINGS IN A SCRIPT IN THE TERMINAL, OTHERWISE BE CAREFUL TO RUN CELLS IN ORDER AS MUCH AS POSSIBLE


class MBHWaveWrap:
    def __init__(self, t_arr, td_set, output_set, td_window):
        self.t_arr = t_arr
        self.td_set, self.output_set = td_set, output_set
        self.td_window = td_window

    def __call__(self, *params):
        assert len(params) == 11
        wave_tmp = mbh_tdi_fly_gen(
            *params,
            upsample_t_arr=self.t_arr,
            combine=True
        )
        wave = TDSignal(wave_tmp, self.td_set).transform(self.output_set, window=self.td_window)
        return wave


if __name__ == "__main__":
    backend = "cpu"

    xp = np if backend == "cpu" else cp

    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0  # mojito
    _Tobs = 1. * YRSID_SI
    # between half day and 3/4 day. Will be very close to half day
    (Nf, Nt, wavelet_duration) = WDMSettings.adjust_to_even_bins(0.5 * 24 * 3600.0, 0.75 * 24 * 3600.0, dt, _Tobs)
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig('2nd generation')  # mojito

    t_start = int(1 / 2 * YRSID_SI / dt) * dt  # 6 months
    t_arr = np.arange(Nobs) * dt + t_start

    t_ref = t_start

    hms = [21, 33, 44]
    negative_modes = True
    tlowfit = True # use a fit to set the starting time of the root finder used in t(f)
    tol = 1e-12 # root finding tolerance

    wave_gen = IMRPhenomTHM(
        higher_modes=hms,
        include_negative_modes=negative_modes, # negative m modes will be produced by simmetry
        t_low_fit=tlowfit,
        coarse_grain=True, # if false it will generate the waveform on a dense time grid with the specified timestep
        coarse_graining_scale_factor=12.0,
        atol=tol,
        rtol=tol,
        T=Tobs,
    )

    # waveform_duration = None  # duration is from t_start through merger
    waveform_duration = 1 / 12 * YRSID_SI  # month before merger
    
    mbh_tdi_fly_gen = MBHTDIonFly(
        wave_gen,
        orbits,
        tdi_config,
        dt,
        Tobs,
        waveform_duration=waveform_duration,
        t0 = t_ref,
    )

    m1 = 1e6
    m2 = 5e5
    s1z = 0.9
    s2z = 0.8
    distance = 10.0e3
    phi_ref = 0.12984324823423
    inclination = 0.302001
    f22_ref = 1e-4
    f22_start = 1e-4
    lam = 0.423234242
    beta = 0.88762349812312
    psi = np.pi/3.
    t_merger = (1.2 * YRSID_SI) - t_start  # RELATIVE

    assert t_merger > t_arr[0] and t_merger < t_arr[-1]

    inj_params = np.array([
        m1,
        m2, 
        s1z,
        s2z,
        distance,
        phi_ref,
        inclination,
        lam,
        beta,
        psi,
        t_merger,
    ])

    injection_arr = mbh_tdi_fly_gen(
        *inj_params,
        upsample_t_arr=t_arr,
        combine=True
    )

    N = injection_arr.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)
    window = xp.asarray(signal.windows.tukey(N, alpha=0.05))
    min_freq = 0.0005
    max_freq = 0.03
    fd_set = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)

    # shave edges?
    min_time = 5 * wavelet_duration
    max_time = (Nt - 5) * wavelet_duration

    wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq, min_time=min_time, max_time=max_time)
    
    output_set = wdm_set  # fd_set
    
    mbh_gen_wrap = MBHWaveWrap(t_arr, td_set, output_set, window)

    template_input = mbh_gen_wrap(*inj_params)

    # plot
    # plt.rcParams['text.usetex'] = False
    # template_input.heatmap()
    # plt.show()
    # plt.close()

    data_inj_all = TDSignal(injection_arr, settings=td_set).transform(output_set, window=window)
    injection = DataResidualArray(data_inj_all)
    template = DataResidualArray(template_input)
    sens_mat = XYZ2SensitivityMatrix(template.data_res_arr.settings, model="scirdv1")

    analysis = AnalysisContainer(injection, sens_mat, signal_gen=mbh_gen_wrap)

    check = analysis.template_inner_product(template)
    check_snr = analysis.template_snr(template)
    check_ll = analysis.template_likelihood(template)
    check_ll_2 = analysis.calculate_signal_likelihood(*inj_params)
    overlap = analysis.template_inner_product(template, normalize=True)
    mismatch = 1.0 - overlap

    print(f"Log Likelihood:", check_ll, "Mismatch (noise-weighted):", mismatch, "Overlap (noise-weighted):", overlap, "SNR (observed, optimal):", check_snr)
    ## mcmc functions

    analysis_mcmc = AnalysisContainer(injection, sens_mat, signal_gen=mbh_gen_wrap)

    ntemps = 2
    nwalkers = 2

    # The order here defines full_basis — must never change
    full_basis = [
        "m1", "m2", "s1z", "s2z", "dist", "phi_ref", "inc", "lam", "beta", "psi", "t_merger"
    ]

    # 12 sampled parameters — order matches priors_in keys
    sampled_basis = [
        "mT", "q", "s1z", "s2z", "dist", "phi_ref", "cosinc", "lam", "sinbeta", "psi", "t_merger"
    ]

    key_map = {
        "mT": "m1",
        "q": "m2", 
        "cosinc": "inc",
        "sinbeta": "beta"
    }

    parameter_transforms = {
        ("mT", "q"): lambda mT, q: (mT / (1 + q), mT * q / (1 + q)),
        'cosinc': np.arccos,
        'sinbeta': np.arcsin,
    }

    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms=parameter_transforms,
        key_map=key_map
    )

    priors = {"mbh": ProbDistContainer({
        "mT": uniform_dist(1e4, 1e8),
        "q": uniform_dist(0.01, 0.99999),
        "s1z": uniform_dist(-0.999999, +0.999999),  
        "s2z": uniform_dist(-0.999999, +0.999999),  
        "dist": uniform_dist(1e2, 1e5),  # Mpc  
        "phi_ref":     uniform_dist(0.0, 2*np.pi),    
        "cosinc":    uniform_dist(-1.0, 1.0),         
        "lam": uniform_dist(0.0, 2*np.pi),    
        "sinbeta":  uniform_dist(-1.0, 1.0), 
        "psi":     uniform_dist(0.0, np.pi),  
        "t_merger": uniform_dist(t_arr[0].item(), t_arr[-1].item())        
    })}


    factor_gen = 1e-8

    mT = m1 + m2
    q = m2 / m1
    assert q < 1.0
    gen_dist = {"mbh": ProbDistContainer({
        "mT": uniform_dist(mT * (1.0 - factor_gen), mT * (1.0 + factor_gen)),
        "q": uniform_dist(q * (1.0 - 1e-8), q * (1.0 + 1e-8)),  # different value for frequency
        "s1z":  uniform_dist(s1z *  (1.0 - factor_gen), s1z * (1.0 + factor_gen)),       # dimensionless spin
        "s2z":    uniform_dist(s2z *  (1.0 - factor_gen), s2z * (1.0 + factor_gen)),       # semi-latus rectum
        "dist":       uniform_dist(distance *  (1.0 - factor_gen), distance * (1.0 + factor_gen)),         # eccentricity
        "phi_ref":     uniform_dist(phi_ref *  (1.0 - factor_gen), phi_ref * (1.0 + factor_gen)),        # luminosity distance [Gpc]
        "cosinc":    uniform_dist(np.cos(inclination) *  (1.0 - factor_gen), np.cos(inclination) * (1.0 + factor_gen)),        # cos(polar spin angle)
        "lam":     uniform_dist(lam *  (1.0 - factor_gen), lam * (1.0 + factor_gen)),     # azimuthal spin angle [rad]
        "sinbeta":     uniform_dist(np.sin(beta) *  (1.0 - factor_gen), np.sin(beta) * (1.0 + factor_gen)),     # azimuthal spin angle [rad]
        "psi":     uniform_dist(psi *  (1.0 - factor_gen), psi * (1.0 + factor_gen)),     # azimuthal spin angle [rad]
        "t_merger":     uniform_dist(t_merger *  (1.0 - factor_gen), t_merger * (1.0 + factor_gen)),     # azimuthal spin angle [rad]
    })}

    ndims = {"mbh": len(sampled_basis)}

    periodic_container = PeriodicContainer({"mbh": {"phi_ref": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}}, key_order={"mbh": sampled_basis})
    fp = f"test_mbh_pe.h5"
    if os.path.exists(fp):
        file_backend = HDFBackend(fp)
        start_state = file_backend.get_last_sample()
    else:
        start_state = State({"mbh": gen_dist["mbh"].rvs(size=(ntemps, nwalkers, 1))})
        
    sampler = EnsembleSampler(
        nwalkers,
        ndims,                              # list of ndims, one per branch
        analysis_mcmc.eryn_likelihood_function,
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        kwargs=dict(
            transform_fn=tc,
            source_only=True,
        ),
        moves=StretchMove(live_dangerously=True),  # allows for testing
        branch_names=["mbh"],
        periodic=periodic_container,
        backend=fp
    )

    print("starting like comp")
    if start_state.log_like is None:
        start_state.log_prior = sampler.compute_log_prior(start_state.branches_coords)
        start_state.log_like = sampler.compute_log_like(start_state.branches_coords, logp=start_state.log_prior)[0]

    print("start log_like: ", start_state.log_like)
    # assert inj_params.shape[0] == len(full_basis)
    # inj_params_in = inj_params[tc.test_inds].copy()
    # inj_params_in[sampled_basis.index("cosinc")] = np.cos(inj_params[full_basis.index("inc")])
    # inj_params_in[sampled_basis.index("sinbeta")] = np.sin(inj_params[full_basis.index("beta")])
    # inj_params_in[sampled_basis.index("mT")] = mT
    # inj_params_in[sampled_basis.index("q")] = q

    # tmp_state = State({"mbh": np.tile(inj_params_in, (ntemps, nwalkers, 1, 1))})
    # should be close to zero if inj matches template model and no noise
    # best_like = sampler.compute_log_like(tmp_state.branches_coords)[0]
    # breakpoint()

    nsteps = 2000
    burn = 0
    output_state = sampler.run_mcmc(start_state, nsteps=nsteps, burn=burn, thin_by=5, progress=True)


