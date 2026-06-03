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

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

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

from emritdionfly import EMRITDIonFly
# credit Michael Katz and Alessandro Santini (with internal code contrubtions in docs)


    # HIGHLY RECOMMEND RUNNING THESE THINGS IN A SCRIPT IN THE TERMINAL, OTHERWISE BE CAREFUL TO RUN CELLS IN ORDER AS MUCH AS POSSIBLE


class EMRIWaveWrap:
    def __init__(self, emri_gen, runtime_kwargs, td_set, output_set, td_window):
        self.td_set, self.output_set = td_set, output_set
        self.td_window = td_window
        self.emri_gen = emri_gen
        self.runtime_kwargs = runtime_kwargs

    def __call__(self, *params):
        assert len(params) == 14
        wave_tmp = np.asarray(self.emri_gen(
            *params,
            convert_to_ra_dec = False,
            **self.runtime_kwargs
        ))
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


    # SETUP EMRI WAVEFORM WITH LEGACY RESPONSE AND TOF
    sum_kwargs = {
        "pad_output": True,
    }

    # This defines the time grid for TDI on the fly. There are improvements we will make here that will not be hard. This setup is fine for initial testing
    # we know there are small fixable issues. 
    N_pts = 16384

    inspiral_kwargs_main = {
        "DENSE_STEPPING": 0,  # sparsely sampled trajectory
        "max_init_len": int(1e4),  # length of trajectories well under 1000
        "force_backend": "cpu"
    }

    t_buffer = 3e4
    inspiral_kwargs_tof = {
        "DENSE_STEPPING": 0,  # sparsely sampled trajectory
        "max_init_len": int(1e4),  # length of trajectories well under 1000
        "upsample": True, 
        "fix_t": True,
        "new_t": np.linspace(t_start + t_buffer, t_start + Tobs - t_buffer, N_pts),
    }

    # amplitude_kwargs = {}

    mode_selector_kwargs_injection = {
        # "mode_selection": "all",  # this is to match data exactly, but takes long
        'mode_selection_threshold': 1e-5
    }

    mode_selector_kwargs_template = {
        # "mode_selection": "all",  # this is to match data exactly, but takes long
        'mode_selection_threshold': 1e-2
    }

    ##======================= Response set-up 

    index_lambda = 8
    index_beta = 7
    waveform_model = 'Kerr'

    response_kwargs = {
        'Tobs': Tobs / YRSID_SI,
        'dt': dt,
        'index_lambda': index_lambda,
        'index_beta': index_beta,
        'flip_hx': True,
        'force_backend': backend,
        'tdi': tdi_config,
        'tdi_chan': 'XYZ',
        'order': 40,
        'remove_garbage': "zero",
        'is_ecliptic_latitude': False,
        't_buffer': t_buffer,
    }

    few_generator_injection = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        return_list=False,    # returns hp - i*hx as a complex cupy array
        inspiral_kwargs=inspiral_kwargs_main,
        sum_kwargs=sum_kwargs,
        # amplitude_kwargs=amplitude_kwargs,
        frame="detector",
        mode_selector_kwargs=mode_selector_kwargs_injection,
        force_backend=backend
    )

    few_generator_template = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        return_list=False,    # returns hp - i*hx as a complex cupy array
        inspiral_kwargs=inspiral_kwargs_main,
        sum_kwargs=sum_kwargs,
        # amplitude_kwargs=amplitude_kwargs,
        frame="detector",
        mode_selector_kwargs=mode_selector_kwargs_template,
        force_backend=backend
    )

    wave_generator_kerr = FastKerrEccentricEquatorialFlux(
        force_backend=backend,
        inspiral_kwargs=inspiral_kwargs_tof,  # different from legacy for now
        sum_kwargs=sum_kwargs,
        # amplitude_kwargs=amplitude_kwargs,
        mode_selector_kwargs=mode_selector_kwargs_template,  # different from legacy for now
    )

    fly_kwargs = {
        'Tobs': response_kwargs['Tobs'],
        'dt': response_kwargs['dt'],
        # 'force_backend': response_kwargs['force_backend'],
        'tdi': response_kwargs["tdi"],
        'tdi_chan': 'XYZ',
    }

    emri_gen_tof = EMRITDIonFly(
        wave_generator_kerr,
        orbits,
        tdi_config,
        dt,
        Tobs,
        t_start,
    )

    m1 = 1e6
    m2 = 1e1
    a = 0.99
    p0 = 6.1
    e0 = 0.3
    xI0= +1.0  # prograde (-) for retrograde
    dist = 2.0 # convert to Gpc
    lam = 0.2538922432234
    beta = -0.418762312
    
    # transform to coordinates needed for FEW Waveform 
    qS = np.pi / 2 - beta
    phiS = lam

    qK = 0.2340980298542
    phiK = 4.098234232

    Phi_phi0 = 1.0803123123
    Phi_theta0 = 1.9823423423  # it will ignore this in equatorial
    Phi_r0 = 4.32094823423

    # this is the necessary parameter order for input into the FEW legacy waveform
    inj_params = np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])

    # EMRI TOF
    # tdi_channels_tof = emri_gen_tof(*inj_params)
    # new_t = t_start + np.arange(10000) * dt
    # _tdi_here_tof = tdi_channels_tof.eval_tdi(new_t)
    # tdi_here_tof = np.sum(_tdi_here_tof, axis=0)
    # plt.plot(tdi_here_tof[0])
    # plt.show()
    # plt.close()
    # breakpoint()

    # just in case the time grids do not align
    _fake_data = (np.arange(10000) - int(10000 / 2)) * dt + t_arr[0]
    assert np.abs(t_start - t_arr[0]) / dt < 10000
    diff = np.abs(_fake_data - t_start)
    _fake_data_closest = _fake_data[diff.argmin()]
    t0_shift_to_data = _fake_data_closest - t_start
    # should be zero with how we have set it up

    legacy_tdi_generator_injection = ResponseWrapper(
        few_generator_injection,
        orbits=orbits,
        t0=t_start,
        t0_shift_to_data=t0_shift_to_data,
        **response_kwargs
    )

    legacy_tdi_generator_template = ResponseWrapper(
        few_generator_template,
        orbits=orbits,
        t0=t_start,
        t0_shift_to_data=t0_shift_to_data,
        **response_kwargs
    )

    runtime_kwargs = {}

    injection_arr = np.asarray(legacy_tdi_generator_injection(*inj_params, convert_to_ra_dec=False, **runtime_kwargs))

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
    emri_gen_wrap = EMRIWaveWrap(legacy_tdi_generator_template, runtime_kwargs, td_set, output_set, window)
    template_input = emri_gen_wrap(*inj_params, **runtime_kwargs)

    # plot
    # plt.rcParams['text.usetex'] = False
    # template_input.heatmap()
    # plt.show()
    # plt.close()

    data_inj_all = TDSignal(injection_arr, settings=td_set).transform(output_set, window=window)
    injection = DataResidualArray(data_inj_all)
    template = DataResidualArray(template_input)
    sens_mat = XYZ2SensitivityMatrix(template.data_res_arr.settings, model="scirdv1")

    analysis = AnalysisContainer(injection, sens_mat, signal_gen=emri_gen_wrap)

    check = analysis.template_inner_product(template)
    check_snr = analysis.template_snr(template)
    check_ll = analysis.template_likelihood(template)
    check_ll_2 = analysis.calculate_signal_likelihood(*inj_params)
    overlap = analysis.template_inner_product(template, normalize=True)
    mismatch = 1.0 - overlap

    print(f"Log Likelihood:", check_ll, "Mismatch (noise-weighted):", mismatch, "Overlap (noise-weighted):", overlap, "SNR (observed, optimal):", check_snr)
    ## mcmc functions

    analysis_mcmc = AnalysisContainer(injection, sens_mat, signal_gen=emri_gen_wrap)

    ntemps = 2
    nwalkers = 2

    # The order here defines full_basis — must never change
    full_basis = [
        'M', 'mu', 'a', 'p0', 'e0', 'x0', 'dist',
        'qS', 'phiS','qK', 'phiK',
        'Phi_phi0', 'Phi_theta0', 'Phi_r0'
    ]

    # 12 sampled parameters — order matches priors_in keys
    sampled_basis = [
        'M', 'mu', 'a', 'p0', 'e0', 'dist',
        'cosqS', 'phiS','cosqK', 'phiK',
        'Phi_phi0', 'Phi_r0'
    ]
    
    key_map = {
        "cosqK": "qK",
        "cosqS": "qS"
    }
    parameter_transforms = {
        'cosqK': np.arccos,
        'cosqS': np.arccos,
    }

    tc = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms=parameter_transforms,
        fill_dict={
        'x0': inj_params[5],
        "Phi_theta0": inj_params[12],
        },
        key_map=key_map
    )

    factor = 1e-3

    priors = {"emri": ProbDistContainer({
        "M": uniform_dist(m1 * (1.0 - factor), m1 * (1.0 + factor)),
        "mu": uniform_dist(m2 * (1.0 - factor), m2 * (1.0 + factor)),
        "a":        uniform_dist(a *  (1.0 - factor), a * (1.0 + factor)),       # dimensionless spin
        "p0":       uniform_dist(p0 *  (1.0 - factor), p0 * (1.0 + factor)),       # semi-latus rectum
        "e0":       uniform_dist(e0 *  (1.0 - factor), e0 * (1.0 + factor)),         # eccentricity
        "dist":     uniform_dist(0.1, 10.0),        # luminosity distance [Gpc]
        "cosqS":    uniform_dist(-1.0, 1.0),        # cos(polar sky angle)
        "phiS":     uniform_dist(0.0, 2*np.pi),     # azimuthal sky angle [rad]
        "cosqK":    uniform_dist(-1.0, 1.0),        # cos(polar spin angle)
        "phiK":     uniform_dist(0.0, 2*np.pi),     # azimuthal spin angle [rad]
        "Phi_phi0": uniform_dist(0.0, 2*np.pi),     # initial azimuthal phase [rad]
        "Phi_r0":   uniform_dist(0.0, 2*np.pi),     # initial radial phase [rad]
    })}


    factor_gen = 1e-8

    gen_dist = {"emri": ProbDistContainer({
        "M": uniform_dist(m1 * (1.0 - factor_gen), m1 * (1.0 + factor_gen)),
        "mu": uniform_dist(m2 * (1.0 - factor_gen), m2 * (1.0 + factor_gen)),
        "a":        uniform_dist(a *  (1.0 - factor_gen), a * (1.0 + factor_gen)),       # dimensionless spin
        "p0":       uniform_dist(p0 *  (1.0 - factor_gen), p0 * (1.0 + factor_gen)),       # semi-latus rectum
        "e0":       uniform_dist(e0 *  (1.0 - factor_gen), e0 * (1.0 + factor_gen)),         # eccentricity
        "dist":     uniform_dist(dist *  (1.0 - factor_gen), dist * (1.0 + factor_gen)),        # luminosity distance [Gpc]
        "cosqS":    uniform_dist(np.cos(qS) *  (1.0 - factor_gen), np.cos(qS) * (1.0 + factor_gen)),        # cos(polar sky angle)
        "phiS":     uniform_dist(phiS *  (1.0 - factor_gen), phiS * (1.0 + factor_gen)),     # azimuthal sky angle [rad]
        "cosqK":    uniform_dist(np.cos(qK) *  (1.0 - factor_gen), np.cos(qK) * (1.0 + factor_gen)),        # cos(polar spin angle)
        "phiK":     uniform_dist(phiK *  (1.0 - factor_gen), phiK * (1.0 + factor_gen)),     # azimuthal spin angle [rad]
        "Phi_phi0": uniform_dist(Phi_phi0 *  (1.0 - factor_gen), Phi_phi0 * (1.0 + factor_gen)),     # initial azimuthal phase [rad]
        "Phi_r0":   uniform_dist(Phi_r0 *  (1.0 - factor_gen), Phi_r0 * (1.0 + factor_gen)),     # initial radial phase [rad]
    })}

    ndims = {"emri": len(sampled_basis)}

    periodic_container = PeriodicContainer({"emri": {"Phi_phi0": 2 * np.pi, "Phi_r0": 2 * np.pi}}, key_order={"emri": sampled_basis})
    fp = f"test_emri_pe.h5"
    if os.path.exists(fp):
        file_backend = HDFBackend(fp)
        start_state = file_backend.get_last_sample()
    else:
        start_state = State({"emri": gen_dist["emri"].rvs(size=(ntemps, nwalkers, 1))})
        
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
        branch_names=["emri"],
        periodic=periodic_container,
        backend=fp
    )

    print("start log_like: ", start_state.log_like)
    assert inj_params.shape[0] == len(full_basis)
    inj_params_in = inj_params[tc.test_inds].copy()
    inj_params_in[sampled_basis.index("cosqS")] = np.cos(inj_params[full_basis.index("qS")])
    inj_params_in[sampled_basis.index("cosqK")] = np.cos(inj_params[full_basis.index("qK")])

    tmp_state = State({"emri": np.tile(inj_params_in, (ntemps, nwalkers, 1, 1))})

    print("starting like comp")
    if start_state.log_like is None:
        start_state.log_prior = sampler.compute_log_prior(start_state.branches_coords)
        start_state.log_like = sampler.compute_log_like(start_state.branches_coords, logp=start_state.log_prior)[0]

    # should be close to zero if inj matches template model and no noise
    best_like = sampler.compute_log_like(tmp_state.branches_coords)[0]

    print("start log_like: ", start_state.log_like)
    nsteps = 2000
    burn = 0
    output_state = sampler.run_mcmc(start_state, nsteps=nsteps, burn=burn, thin_by=5, progress=True)


