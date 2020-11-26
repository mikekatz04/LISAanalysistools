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

from few.waveform import FastSchwarzschildEccentricFlux

from lisatools.sampling.likelihood import Likelihood

from lisatools.sampling.samplers.emcee import EmceeSampler
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler

import warnings

warnings.filterwarnings("ignore")

fast = FastSchwarzschildEccentricFlux(sum_kwargs=dict(pad_output=True), use_gpu=False)


def wrap_func(*args, **kwargs):
    temp = fast(*args, **kwargs)
    return [temp.real, -temp.imag]


M = 1e6
mu = 1e2
p0 = 10.601813660750054
e0 = 0.2
phi = 0.0
theta = np.pi / 3.0
dist = 5.0

T = 0.01
dt = 10.0

inner_product_kwargs = dict(frequency_domain=False, PSD="cornish_lisa_psd")

transform_fn = {0: np.exp}

injection_params = np.array([np.log(M), mu, p0, e0, theta, phi, dist])

params_test = injection_params.copy()

waveform_kwargs = {"T": T, "dt": dt, "eps": 1e-5}

nwalkers = 16

ndim_full = 7

test_inds = np.array([0, 1, 2, 3])

ndim = len(test_inds)
fill_inds = np.delete(np.arange(ndim_full), test_inds)
fill_values = injection_params[fill_inds]

like = Likelihood(
    wrap_func,
    2,
    frequency_domain=False,
    parameter_transforms=transform_fn,
    use_gpu=False,
)


like.inject_signal(
    dt,
    params=injection_params.copy(),
    waveform_kwargs=waveform_kwargs,
    noise_fn=get_sensitivity,
    noise_kwargs=dict(sens_fn="cornish_lisa_psd"),
    add_noise=False,
)

snr_check = snr(
    wrap_func(M, mu, p0, e0, theta, phi, dist, dt=dt, T=T, eps=1e-2),
    dt,
    **inner_product_kwargs
)

print("snr:", snr_check)
params_test[0] *= 1.0000001

params_test = np.tile(params_test, (2, 1))

check = like.get_ll(params_test, waveform_kwargs=waveform_kwargs)


prior_ranges = [
    [injection_params[i] * 0.9, injection_params[i] * 1.1] for i in test_inds
]

waveform_kwargs_templates = waveform_kwargs.copy()
waveform_kwargs_templates["eps"] = 1e-5
sampler = PTEmceeSampler(
    nwalkers,
    ndim,
    ndim_full,
    like,
    prior_ranges,
    subset=4,
    lnlike_kwargs={"waveform_kwargs": waveform_kwargs_templates},
    test_inds=test_inds,
    fill_values=fill_values,
    fp="test_full_2yr.h5",
)


"""
eps = 1e-9
cov = covariance(
    fast,
    injection_params,
    eps,
    dt,
    deriv_inds=test_inds,
    parameter_transforms=transform_fn,
    waveform_kwargs=waveform_kwargs,
    inner_product_kwargs=inner_product_kwargs,
    diagonalize=False,
)
"""

factor = 1e-2
start_points = injection_params[
    np.newaxis, test_inds
] + factor * np.random.multivariate_normal(np.zeros(len(test_inds)), cov, size=nwalkers)


max_iter = 40000
sampler.sample(start_points, max_iter, show_progress=True)
