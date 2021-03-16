import numpy as np
import matplotlib.pyplot as plt

from lisatools.sensitivity import get_sensitivity
from lisatools.diagnostic import (
    inner_product,
    snr,
    fisher,
    covariance,
    mismatch_criterion,
    cutler_vallisneri_bias,
    scale_snr,
)

from few.waveform import FastSchwarzschildEccentricFlux, GenerateEMRIWaveform

from lisatools.sampling.likelihood import Likelihood

from lisatools.sampling.samplers.emcee import EmceeSampler
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler
from lisatools.utils.utility import uniform_dist

from lisatools.utils.transform import TransformContainer

import warnings

warnings.filterwarnings("ignore")


use_gpu = False
fast = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True),
    use_gpu=use_gpu,
    return_list=True,
)

fast_not_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True),
    use_gpu=use_gpu,
    return_list=False,
)


# define injection parameters
M = 1.00000000e06
mu = 3.00000000e01
p0 = 1.29984395e01
e0 = 3.00000000e-01
dist = 4.10864264e00
Phi_phi0 = 3.2302777624860943
Phi_r0 = 4.720221760052107

# define other parameters necessary for calculation
a = 0.0
Y0 = 1.0
qS = 0.5420879369091457
phiS = 5.3576560705195275
qK = 1.7348119514252445
phiK = 3.2004167279159637
Phi_theta0 = 0.0

# injection array
injection_params = np.array(
    [
        np.log(M),
        mu,
        a,  # will ignore
        p0,
        e0,
        Y0,  # will ignore
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,  # will ignore
        Phi_r0,
    ]
)

# define other quantities
T = 14/365  # years
dt = 15.0

snr_goal = 30.0

# for SNR and covariance calculation
inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd")

# transformation of arguments from sampling basis to waveform basis
transform_fn_in = {
    "base": {0: (lambda x: np.exp(x)),},
}

# use the special transform container
transform_fn = TransformContainer(transform_fn_in)

# copy parameters to check values
check_params = injection_params.copy()

# this transforms but also gives the transpose of the input dimensions due to inner sampler needs
check_params = transform_fn.transform_base_parameters(check_params).T

# INJECTION kwargs
waveform_kwargs = {"T": T, "dt": dt, "eps": 1e-5}

check_sig = fast(*check_params, **waveform_kwargs)

# adjust distance for SNR goal
check_sig, snr_orig = scale_snr(snr_goal, check_sig, return_orig_snr=True,**inner_product_kwargs) #, dt=dt

print("orig_dist:", injection_params[6])
injection_params[6] *= snr_orig / snr_goal
print("new_dist:", injection_params[6])

# copy again
params_test = injection_params.copy()

# define sampling quantities
nwalkers = 32  # per temperature
ntemps = 10

ndim_full = 14  # full dimensionality of inputs to waveform model

# which of the injection parameters you actually want to sample over
test_inds = np.array([0, 1, 3, 4, 11, 13])

# ndim for sampler
ndim = len(test_inds)

# need to get values to fill arrays for quantities that we are not sampling
fill_inds = np.delete(np.arange(ndim_full), test_inds)
fill_values = injection_params[fill_inds]

# to store in sampler file, get injection points we are sampling
injection_test_points = injection_params[test_inds]

# instantiate the likelihood class
nchannels = 2
like = Likelihood(
    fast, nchannels, dt=dt, parameter_transforms=transform_fn, use_gpu=use_gpu,
)

# inject
like.inject_signal(
    params=injection_params.copy(),
    waveform_kwargs=waveform_kwargs,
    noise_fn=get_sensitivity,
    noise_kwargs=dict(sens_fn="cornish_lisa_psd"),
    add_noise=False,
)

# for checks
check_params = injection_params.copy()

check_params = np.tile(check_params, (6, 1))
check_params[0][0] *= 1.0004
check_ll = like.get_ll(check_params, waveform_kwargs=waveform_kwargs)

# get covariance
eps = 1e-7
cov, fish = covariance(
    fast_not_list,
    injection_params,
    eps,
    deriv_inds=test_inds,
    parameter_transforms=transform_fn,
    waveform_kwargs=waveform_kwargs,
    inner_product_kwargs=inner_product_kwargs,
    diagonalize=False,
    return_fisher=True,
)

sig_diag = np.sqrt(cov.diagonal())

# define priors, it really can only do uniform cube at the moment
priors = [
    uniform_dist(np.log(1e5), np.log(3e6)),
    uniform_dist(1.0, 100.0),
    uniform_dist(10.0, 15.0),
    uniform_dist(0.01, 0.5),
    uniform_dist(0.0, 2 * np.pi),
    uniform_dist(0.0, 2 * np.pi),
]

# can add extra temperatures of 1 to have multiple temps accessing the target distribution
ntemps_target_extra = 0
# define max temperature (generally should be inf if you want to sample prior)
Tmax = np.inf

# not all walkers can fit in memory. subset says how many to do at one time
subset = 4

# set kwargs for the templates
waveform_kwargs_templates = waveform_kwargs.copy()
waveform_kwargs_templates["eps"] = 1e-2

# sampler starting points around true point
factor = 1e-2
start_points = injection_params[
    np.newaxis, test_inds
] + factor * np.random.multivariate_normal(np.zeros(len(test_inds)), cov, size=nwalkers*ntemps)

"""
# random starts
np.random.seed(3000)
start_points = np.zeros((nwalkers * ntemps, ndim))
for i in range(ndim):
    start_points[:, i] = priors[i].rvs(nwalkers * ntemps)
"""

# check the starting points
start_test = np.zeros((nwalkers * ntemps, ndim_full))

# need to set sampling and non-sampling quantities
start_test[:, test_inds] = start_points
start_test[:, fill_inds] = fill_values

split_inds = np.split(np.arange(len(start_test)), int(len(start_test) / subset))

start_ll = np.asarray(
    [
        like.get_ll(start_test[split], waveform_kwargs=waveform_kwargs)
        for split in split_inds
    ]
)

# set periodic parameters
periodic = {
    4: 2 * np.pi,
    5: 2 * np.pi,
}  # the indexes correspond to the index within test_inds

# setup sampler
sampler = PTEmceeSampler(
    nwalkers,
    ndim,
    ndim_full,
    like,
    priors,
    subset=subset,
    lnlike_kwargs={"waveform_kwargs": waveform_kwargs_templates},
    test_inds=test_inds,
    fill_values=fill_values,
    ntemps=ntemps,
    autocorr_multiplier=100, # automatic stopper, be careful with this since the parallel tempering 
    autocorr_iter_count=50, # how often it checks the autocorrelation
    ntemps_target_extra=ntemps_target_extra,
    Tmax=Tmax,
    injection=injection_test_points,
    plot_iterations=100,
    plot_source="emri",
    periodic=periodic,
    fp="search_2yr_snr_{:d}_no_noise_{}_{}_{}_eps_1e-2_large_prior.h5".format(
        int(snr_goal), M, mu, e0
    ),
)

thin_by = 1
max_iter = 10000
sampler.sample(
    start_points,
    iterations=max_iter,
    progress=True,
    skip_initial_state_check=False,
    thin_by=thin_by,
)
