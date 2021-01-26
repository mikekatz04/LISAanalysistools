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

# from lisatools.sampling.samplers.emcee import EmceeSampler
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler

import warnings


warnings.filterwarnings("ignore")

use_gpu = gpu_available
gb = GBGPU(use_gpu=use_gpu, shift_ind=2)

num_bin = 100
amp = 1.4811552270333745e-19 * 6
f0 = 2e-3
fdot = 3.143115116217208e-17
fddot = 0.0
phi0 = 0.1
iota = 0.2
psi = 0.3
lam = 0.4
beta_sky = 0.5
e1 = 0.0
beta1 = 0.5
A2 = 227.49224525104734
omegabar = 0.0
e2 = 0.4
P2 = 1.3230498230
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
e1_in = np.full(num_bin, e1)
beta1_in = np.full(num_bin, beta1)
A2_in = np.full(num_bin, A2)
P2_in = np.full(num_bin, P2)
omegabar_in = np.full(num_bin, omegabar)
e2_in = np.full(num_bin, e2)
T2_in = np.full(num_bin, T2)
N = None

modes = np.array([2])

Tobs = 4.0 * YEAR
dt = 15.0
Tobs = int(Tobs / dt) * dt
df = 1 / Tobs

waveform_kwargs = dict(modes=modes, N=N, dt=dt)
fish_kwargs = dict(modes=modes, N=1024, dt=dt)

transform_fn_in = {
    "base": {
        0: (lambda x: np.exp(x)),
        1: (lambda x: x * 1e-3),
        2: (lambda x: np.exp(x)),
        5: (lambda x: np.arccos(x)),
        8: (lambda x: np.arcsin(x)),
    },
}

from lisatools.utils.transform import TransformContainer

transform_fn = TransformContainer(transform_fn_in)

like = Likelihood(gb, 2, df=df, parameter_transforms=transform_fn, use_gpu=use_gpu,)

injection_params = np.array(
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
        A2,
        omegabar,
        e2,
        P2,
        T2,
    ]
)


inds_test = np.delete(np.arange(len(injection_params)), 3)
fish = gb.fisher(
    np.array([injection_params]).T,
    inds=inds_test,
    parameter_transforms=transform_fn_in["base"],
    **fish_kwargs,
)

A_inj, E_inj = gb.inject_signal(
    amp,
    f0,
    fdot,
    fddot,
    phi0,
    iota,
    psi,
    lam,
    beta_sky,
    A2,
    omegabar,
    e2,
    P2,
    T2,
    **waveform_kwargs,
)

check_snr = snr([A_inj, E_inj], df=df, PSD="noisepsd_AE")

print("snr:", check_snr)

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
        A2_in,
        omegabar_in,
        e2_in,
        P2_in,
        T2_in,
    ]
).T

check = like.get_ll(params_test, waveform_kwargs=waveform_kwargs)

nwalkers = 500

ndim_full = 14

test_inds = np.delete(np.arange(ndim_full), np.array([3]))

ndim = len(test_inds)
fill_inds = np.delete(np.arange(ndim_full), test_inds)
fill_values = injection_params[fill_inds]

from lisatools.utils.utility import uniform_dist

priors = [
    uniform_dist(np.log(1e-23), np.log(1e-17)),
    uniform_dist(1.0, 5.0),
    uniform_dist(np.log(1e-18), np.log(1e-12)),
    uniform_dist(0.0, 2 * np.pi),
    uniform_dist(-1, 1),
    uniform_dist(0.0, np.pi),
    uniform_dist(0.0, 2 * np.pi),
    uniform_dist(-1, 1),
    uniform_dist(1.0, 1000.0),
    uniform_dist(0.0, np.pi * 2),
    uniform_dist(0.0001, 0.9),
    uniform_dist(0.5, 8.0),
    uniform_dist(0.0, 10.0),
]
ntemps = 10
ntemps_target_extra = 0
Tmax = np.inf
injection_test_points = injection_params[test_inds]

labels = [
    r"$A$",
    r"$f_0$ (mHz)",
    r"$\dot{f}_0$",
    r"$\phi_0$",
    r"cos$\iota$",
    r"$\psi$",
    r"$\lambda$",
    r"sin$\beta$",
    r"$A_2$",
    r"$\bar{\omega}$",
    r"$e_2$",
    r"$P_2$",
    r"$T_2$",
]

plot_kwargs = dict(
    include_titles=True,
    print_diagnostics=False,
    thin_chain_by_ac=True,
    corner_kwargs=dict(labels=labels, truths=injection_test_points),
    test_inds=None,
    sub_keys={},
    parameter_transforms=None,
)

# from lisatools.sampling.plot import PlotContainer

# plotter = PlotContainer(
#    "GB_sampler_test_with_third_injection_with_third_SNR_300.h5", "gb", **plot_kwargs
# )
# plotter.generate_corner(burn=2000)

# breakpoint()
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

cov = np.linalg.pinv(fish.squeeze())

factor = 1e3
start_points = np.random.multivariate_normal(
    injection_params[test_inds], cov * factor, size=nwalkers * ntemps
)

start_points[:, -1] = np.abs(start_points[:, -1])
start_points[:, -4] = np.abs(start_points[:, -4])
start_points[:, 3] = start_points[:, 3] % (2 * np.pi)
start_points[:, 5] = start_points[:, 5] % (np.pi)
start_points[:, 6] = start_points[:, 6] % (2 * np.pi)
start_points[:, 9] = start_points[:, 9] % (2 * np.pi)


start_check = np.zeros((start_points.shape[0], ndim_full))
start_check[:, test_inds] = start_points
start_check[:, fill_inds] = fill_values

check2 = like.get_ll(start_check, waveform_kwargs=waveform_kwargs)

max_iter = 10000
thin_by = 5

print(start_check, check2)
# breakpoint()
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
    injection=injection_test_points,
    plot_iterations=20,
    plot_source="gb",
    plot_kwargs=plot_kwargs,
    # betas=betas,
    periodic={3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi, 9: 2 * np.pi},
    autocorr_multiplier=5000,
    autocorr_iter_count=100,
    burn=10,
    fp="temp.h5",
)

sampler.sample(
    start_points,
    iterations=max_iter,
    progress=True,
    skip_initial_state_check=True,
    thin_by=thin_by,
)
