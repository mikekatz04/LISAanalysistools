import numpy as np

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False
# import matplotlib.pyplot as plt

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

from gbgpu.gbgpu import GBGPU

from gbgpu.utils.constants import *

from lisatools.sampling.likelihood import Likelihood

from lisatools.sensitivity import get_sensitivity

from lisatools.sampling.samplers.emcee import EmceeSampler
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler
from lisatools.utils.utility import uniform_dist
import warnings


warnings.filterwarnings("ignore")

use_gpu = gpu_available
gb = GBGPU(use_gpu=use_gpu)

num_bin = 100
amp = 1e-19
f0 = 2e-3
fdot = 1e-16
fddot = 0.0
phi0 = 0.1
iota = 0.2
psi = 0.3
lam = 0.4
beta_sky = 0.5
A2 = 19.5
omegabar = 0.0
e2 = 0.3
P2 = 0.6
T2 = 0.0

amp_in = np.full(num_bin, amp)
f0_in = np.full(num_bin, f0)
fdot_in = np.full(num_bin, fdot)
fddot_in = np.full(num_bin, fddot)
phi0_in = np.full(num_bin, phi0)
iota_in = np.full(num_bin, iota)
psi_in = np.full(num_bin, psi)
lam_in = np.full(num_bin, lam)
beta_sky_in = np.full(num_bin, beta_sky)
A2_in = np.full(num_bin, A2)
P2_in = np.full(num_bin, P2)
omegabar_in = np.full(num_bin, omegabar)
e2_in = np.full(num_bin, e2)
T2_in = np.full(num_bin, T2)
N = int(1024)

Tobs = 4.0 * YEAR
dt = 15.0

n = int(Tobs / dt)

df = 1 / (n * dt)

waveform_kwargs = dict(N=N, dt=dt)

transform_fn = {
    0: (lambda x: 1 * np.exp(x)),
    1: (lambda x: x * 1e-3),
    2: (lambda x: 1 * np.exp(x)),
    5: (lambda x: np.arccos(x)),
    8: (lambda x: np.arcsin(x)),
}

like = Likelihood(gb, 2, df=df, parameter_transforms=transform_fn, use_gpu=use_gpu,)

params = np.array(
    [
        np.log(amp),
        f0 * 1e3,
        np.log(fdot),
        fddot,
        phi0,
        np.cos(iota),
        psi,
        lam,
        np.sin(beta_sky),
        # A2,
        # omegabar,
        # e2,
        # P2,
        # T2,
    ]
)

fish_params = np.array([params.copy()]).T

inds_test = np.delete(np.arange(len(fish_params)), 3)

fish = gb.fisher(
    fish_params, inds=inds_test, parameter_transforms=transform_fn, **waveform_kwargs
).squeeze()

cov = np.linalg.pinv(fish)

injection_params = params.copy()
for ind, trans_fn in transform_fn.items():
    injection_params[ind] = trans_fn(injection_params[ind])

A_inj, E_inj = gb.inject_signal(*injection_params, **waveform_kwargs,)

check_snr = snr([A_inj, E_inj], df=df, PSD="noisepsd_AE")

like.inject_signal(
    data_stream=[A_inj, E_inj],
    noise_fn=get_sensitivity,
    noise_kwargs={"sens_fn": "noisepsd_AE"},
    add_noise=False,
)

amp_in[0] *= 1.1
f0_in[1] = 2.001e-3
params_test = np.array(
    [
        np.log(amp_in),
        f0_in * 1e3,
        np.log(fdot_in),
        fddot_in,
        phi0_in,
        np.cos(iota_in),
        psi_in,
        lam_in,
        np.sin(beta_sky_in),
    ]
).T

check = like.get_ll(params_test, waveform_kwargs=waveform_kwargs)

nwalkers = 100

ndim_full = len(params)

test_inds = np.delete(np.arange(ndim_full), np.array([3]))

ndim = len(test_inds)
fill_inds = np.delete(np.arange(ndim_full), test_inds)
fill_values = params[fill_inds]

priors = [
    uniform_dist(np.log(1e-22), np.log(1e-17)),
    uniform_dist(8e-4 * 1e3, 1e-2 * 1e3),
    uniform_dist(np.log(1e-18), np.log(1e-14)),
    uniform_dist(0, 2 * np.pi),
    uniform_dist(-0.9999, 0.99999),
    uniform_dist(0.0, np.pi),
    uniform_dist(0.0, 2 * np.pi),
    uniform_dist(-0.9999, 0.99999),
]

ntemps = 20
Tmax = np.inf
ntemps_target_extra = 10
# betas = np.array([1.0])

sampler = PTEmceeSampler(
    nwalkers,
    ndim,
    ndim_full,
    like,
    priors,
    subset=None,
    lnlike_kwargs={"waveform_kwargs": waveform_kwargs},
    test_inds=test_inds,
    fill_values=fill_values,
    ntemps=ntemps,
    ntemps_target_extra=ntemps_target_extra,
    Tmax=Tmax,
    # betas=betas,
    autocorr_multiplier=50,
    autocorr_iter_count=250,
    fp="test_gb_pt_11_zero_temps.h5",
)

factor = 1e-6
start_points = (
    params[np.newaxis, test_inds]
    + factor * np.random.randn(nwalkers * ntemps, ndim) * params[np.newaxis, test_inds]
)

max_iter = 5000
sampler.sample(start_points, max_iter, show_progress=True)
