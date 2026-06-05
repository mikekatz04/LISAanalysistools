#!/usr/bin/env python
# coding: utf-8
"""SOBBHTDIonTheFly verification + Eryn parameter-estimation harness.

Mirrors gb_test_script_td_wave.py end-to-end, only swapping the GB
parameterisation (amp, f0, fdot, fddot, phi0) for the 7 SOBBH intrinsic
parameters (m1, m2, s1, s2, distance, f_low, phi_c). Extrinsic params
(inc, psi, lam, beta) keep the same conventions as the GB script.
"""

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
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.directresponse import icrs_to_ecliptic
from lisatools.response.tdionfly import SOBBHTDIonTheFly


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


class SOBBHWaveWrap:
    def __init__(self, t_arr, t_tdi_sparse, Tobs, t_ref, dt, num_bin, sobbh_tdi_kwargs, td_set, output_set, td_window):
        self.t_arr, self.t_tdi_sparse = t_arr, t_tdi_sparse
        self.Tobs, self.t_ref, self.dt, self.num_bin, self.sobbh_tdi_kwargs = Tobs, t_ref, dt, num_bin, sobbh_tdi_kwargs
        self.td_set, self.output_set = td_set, output_set
        self.td_window = td_window

    def __call__(self, *params):
        params = np.asarray([params])
        assert params.shape[-1] == 11
        sobbh_gen_tmp = SOBBHTDIonTheFly(
            self.t_tdi_sparse, self.Tobs, self.t_ref, self.dt, params.shape[0],
            **self.sobbh_tdi_kwargs
        )
        wave_tmp = sobbh_gen_tmp(*params.T, convert_to_ra_dec=False, return_spline=True)
        wave = TDSignal(wave_tmp.eval_tdi(self.t_arr)[0], self.td_set).transform(self.output_set, window=self.td_window)
        return wave


if __name__ == "__main__":
    backend = "cpu"

    xp = np if backend == "cpu" else cp

    orbits = ESAOrbits(force_backend=backend)
    dt = 10.0  # mojito
    _Tobs = 1. * YRSID_SI
    # between half day and 3/4 day. Will be very close to half day
    (Nf, Nt, wavelet_duration) = WDMSettings.adjust_to_even_bins(2 * 3600.0, 3 * 24 * 3600.0, dt, _Tobs)
    Tobs = Nt * wavelet_duration
    Nobs = Nf * Nt

    tdi_config = TDIConfig('2nd generation')  # mojito

    t_start = int(1 / 2 * YRSID_SI / dt) * dt  # 6 months
    t_arr = np.arange(Nobs) * dt + t_start
    data_inj = np.zeros((3, t_arr.shape[-1]))
    template = np.zeros((3, t_arr.shape[-1]))

    t_ref = t_start

    N_inj = 16384

    sobbh_tdi_kwargs = dict(
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=backend,
    )
    t_tdi_inj = xp.linspace(t_arr[0], t_arr[-1], N_inj)
    sobbh_gen_inj = SOBBHTDIonTheFly(
        t_tdi_inj,
        Tobs,
        t_ref,
        1. / dt,
        1,
        **sobbh_tdi_kwargs
    )


    N_sparse = 16384

    t_tdi_sparse = xp.linspace(t_arr[0], t_arr[-1], N_sparse)

    N = data_inj.shape[-1]
    td_set = TDSettings(N, dt, force_backend=backend)
    freqs = np.fft.rfftfreq(N, dt)
    df = freqs[1] - freqs[0]
    N_fd = len(freqs)
    window = xp.asarray(signal.windows.tukey(N, alpha=0.05))
    min_freq = 0.0005
    max_freq = 0.03
    fd_set = FDSettings(N_fd, df, min_freq=min_freq, max_freq=max_freq, force_backend=backend)

    # shave edges?
    min_time = 20 * wavelet_duration
    max_time = (Nt - 20) * wavelet_duration

    wdm_set = WDMSettings(Nf, Nt, dt, min_freq=min_freq, max_freq=max_freq, min_time=min_time, max_time=max_time)

    num_bin = 1
    # Intrinsic
    m1 = np.full(num_bin, 30.0)
    m2 = np.full(num_bin, 20.0)
    s1 = np.full(num_bin, 0.0)
    s2 = np.full(num_bin, 0.0)
    distance = np.full(num_bin, 1.0e9)        # parsecs (1 Gpc)
    f_low = np.full(num_bin, 2.0e-3)          # Hz
    phi_c = np.full(num_bin, 0.0)
    # Extrinsic
    inc = np.full(num_bin, 0.23984234)
    psi = np.full(num_bin, 1.234019814)
    lam = np.full(num_bin, 4.09808143)
    beta = np.full(num_bin, 0.090)
    params = np.array([m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta]).T

    inj_tmp = sobbh_gen_inj(m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta, convert_to_ra_dec=False, return_spline=True)
    data_inj[:] = inj_tmp.eval_tdi(t_arr)

    output_set = wdm_set

    sobbh_gen_wrap = SOBBHWaveWrap(
        t_arr,
        t_tdi_sparse,
        Tobs,
        t_ref,
        dt,
        params.shape[0],
        sobbh_tdi_kwargs,
        td_set,
        output_set,
        window
    )

    template_sparse = sobbh_gen_wrap(*params[0])
    template_sparse = sobbh_gen_wrap(*params[0])

    data_inj_all = TDSignal(data_inj, settings=td_set).transform(output_set, window=window)
    injection = DataResidualArray(data_inj_all)
    template = DataResidualArray(template_sparse)
    sens_mat = XYZ2SensitivityMatrix(template.data_res_arr.settings, model="scirdv1")

    analysis = AnalysisContainer(injection, sens_mat, signal_gen=sobbh_gen_wrap)

    check = analysis.template_inner_product(template)
    check_snr = analysis.template_snr(template)
    check_ll = analysis.template_likelihood(template)
    check_ll_2 = analysis.calculate_signal_likelihood(*params[0])
    overlap = analysis.template_inner_product(template, normalize=True)
    mismatch = 1.0 - overlap

    print(f"Log Likelihood:", check_ll, "Mismatch (noise-weighted):", mismatch, "Overlap (noise-weighted):", overlap, "SNR (observed, optimal):", check_snr)
    ## mcmc functions
    breakpoint()
    analysis_mcmc = AnalysisContainer(injection, sens_mat, signal_gen=sobbh_gen_wrap)

    ntemps = 2
    nwalkers = 2

    # The order here defines full_basis — must never change
    full_basis = [
        "m1", "m2", "s1", "s2", "distance", "f_low", "phi_c", "inc", "psi", "lam", "beta"
    ]

    # Sampled parameters — order matches priors_in keys
    sampled_basis = [
        "m1", "m2", "s1", "s2", "distance", "f_low", "phi_c", "cosinc", "psi", "lam", "sinbeta"
    ]

    key_map = {
        "cosinc": "inc",
        "sinbeta": "beta"
    }

    parameter_transforms = {
        'cosinc': np.arccos,
        'sinbeta': np.arcsin,
    }

    tc_transform = TransformContainer(
        input_basis=sampled_basis,
        output_basis=full_basis,
        parameter_transforms=parameter_transforms,
        fill_dict={},
        key_map=key_map
    )

    priors = {"sobbh": ProbDistContainer({
        "m1": uniform_dist(5.0, 80.0),
        "m2": uniform_dist(5.0, 80.0),
        "s1": uniform_dist(-0.99, 0.99),
        "s2": uniform_dist(-0.99, 0.99),
        "distance": uniform_dist(1.0e8, 1.0e10),
        "f_low": uniform_dist(5.0e-4, 1.0e-2),
        "phi_c": uniform_dist(0.0, 2 * np.pi),
        "cosinc": uniform_dist(-1.0, 1.0),
        "psi": uniform_dist(0.0, np.pi),
        "lam": uniform_dist(0.0, 2 * np.pi),
        "sinbeta": uniform_dist(-1.0, 1.0),
    })}


    factor_gen = 1e-5

    gen_dist = {"sobbh": ProbDistContainer({
        "m1": uniform_dist(m1[0] * (1.0 - factor_gen), m1[0] * (1.0 + factor_gen)),
        "m2": uniform_dist(m2[0] * (1.0 - factor_gen), m2[0] * (1.0 + factor_gen)),
        "s1": uniform_dist(s1[0] - factor_gen, s1[0] + factor_gen),
        "s2": uniform_dist(s2[0] - factor_gen, s2[0] + factor_gen),
        "distance": uniform_dist(distance[0] * (1.0 - factor_gen), distance[0] * (1.0 + factor_gen)),
        "f_low": uniform_dist(f_low[0] * (1.0 - 1e-8), f_low[0] * (1.0 + 1e-8)),
        "phi_c": uniform_dist(phi_c[0] - factor_gen, phi_c[0] + factor_gen),
        "cosinc": uniform_dist(np.cos(inc[0]) * (1.0 - factor_gen), np.cos(inc[0]) * (1.0 + factor_gen)),
        "psi": uniform_dist(psi[0] * (1.0 - factor_gen), psi[0] * (1.0 + factor_gen)),
        "lam": uniform_dist(lam[0] * (1.0 - factor_gen), lam[0] * (1.0 + factor_gen)),
        "sinbeta": uniform_dist(np.sin(beta[0]) * (1.0 - factor_gen), np.sin(beta[0]) * (1.0 + factor_gen)),
    })}

    ndims = {"sobbh": len(sampled_basis)}

    periodic_container = PeriodicContainer({"sobbh": {"phi_c": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}}, key_order={"sobbh": sampled_basis})
    fp = f"test_sobbh_pe.h5"
    if os.path.exists(fp):
        file_backend = HDFBackend(fp)
        start_state = file_backend.get_last_sample()
    else:
        start_state = State({"sobbh": gen_dist["sobbh"].rvs(size=(ntemps, nwalkers, 1))})

    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        analysis_mcmc.eryn_likelihood_function,
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        kwargs=dict(
            transform_fn=tc_transform,
            source_only=True,
        ),
        moves=StretchMove(live_dangerously=True),
        branch_names=["sobbh"],
        periodic=periodic_container,
        backend=fp
    )

    if start_state.log_like is None:
        start_state.log_prior = sampler.compute_log_prior(start_state.branches_coords)
        start_state.log_like = sampler.compute_log_like(start_state.branches_coords, logp=start_state.log_prior)[0]

    print("start log_like: ", start_state.log_like)

    nsteps = 2000
    burn = 0
    output_state = sampler.run_mcmc(start_state, nsteps=nsteps, burn=burn, thin_by=5, progress=True)
