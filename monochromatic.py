import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# LISA TOOLs
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

use_gpu = False

from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.samplers.emcee import EmceeSampler
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler
from lisatools.sampling.prior import *
from lisatools.utils.transform import TransformContainer
import warnings

warnings.filterwarnings("ignore")

def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')


# MY waveform class
class Monochromatic:

    def __init__(self, return_list=False , inspiral_kwargs={}):
        # something
        self.inspiral_kwargs = inspiral_kwargs
        self.return_list = return_list

    def __call__(
        self,
        A,
        f,
        fdot,
        fddot,
        phi,
        dt=10.0,
        T=1.0,
    ):
        time_vec = np.arange(0,T*31536000,dt)
        h = zero_pad(A*np.exp(1j* 2 * np.pi * (f + fdot * time_vec/2 + fddot/6 * time_vec * time_vec) * time_vec + phi))

        if self.return_list is False:
            return h

        else:
            hp = h.real
            hx = -h.imag
            return [hp, hx]

# start of the program
mon = Monochromatic(return_list=True)
mon_not_list = Monochromatic(return_list=False)

# injection array
injection_params = np.array(
    [
        np.log(1e-19),
        1e-3,
        1e-11,
        0,
        0
    ]
)

# which of the injection parameters you actually want to sample over
test_inds = np.array([0, 1, 2])

# transformation of arguments from sampling basis to waveform basis
transform_fn_in = {
    0: (lambda x: np.exp(x))
}
# use the special transform container
transform_fn = TransformContainer(transform_fn_in)
#
check_params = transform_fn.transform_base_parameters(injection_params.copy()).T


# time stuff
dt = 10
T = 10/365
minf = 1/(T*3600*24*365)
waveform_kwargs = {"T": T, "dt": dt}

# generate waveform
h = mon(*check_params,**waveform_kwargs)

# for SNR and covariance calculation
inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd")

#%% another check
tot_snr_0 = snr(h,**inner_product_kwargs)

#############
#%% Fisher and derivatives
eps = [1e-7, 1e-10, 1e-15,]
cov, fish, dh = covariance(
    mon_not_list,
    injection_params,
    eps,
    deriv_inds=test_inds,
    waveform_kwargs=waveform_kwargs,
    inner_product_kwargs=inner_product_kwargs,
    parameter_transforms=transform_fn,
    diagonalize=False,
    return_fisher=True,
    return_derivs=True
)

print("ratio numerical theoretical ", fish[0,0]/(tot_snr_0**2) )
print("ratio numerical theoretical ", fish[1,1]/(4*(np.pi*tot_snr_0*T*365*24*3600)**2 / 3) )
print("ratio numerical theoretical ", fish[2,2]/((np.pi*tot_snr_0*(T*365*24*3600)**2)**2 / 5) )

############################################
#%% MCMC parameters to sample over

# define sampling quantities
nwalkers = 8  # per temperature
ntemps = 2

ndim_full = 5  # full dimensionality of inputs to waveform model


# ndim for sampler
ndim = len(test_inds)

# need to get values to fill arrays for quantities that we are not sampling
fill_inds = np.delete(np.arange(ndim_full), test_inds)
fill_values = injection_params[fill_inds]

# to store in sampler file, get injection points we are sampling
injection_test_points = injection_params[test_inds]


######################################################
#%% Likelihood Definition and injection
# instantiate the likelihood class
nchannels = 2
like = Likelihood(
    mon, nchannels, dt=dt, parameter_transforms=transform_fn, use_gpu=use_gpu,
)

# inject
like.inject_signal(
    # in this way I provide the data stream
    #data_stream=h,
    params=injection_params.copy(),
    waveform_kwargs=waveform_kwargs,
    noise_fn=get_sensitivity,
    noise_kwargs=dict(sens_fn="cornish_lisa_psd"),
    add_noise=False,
)

#%% MCMC priors
# define priors, it really can only do uniform cube at the moment
priors_in = {0: uniform_dist(np.log(1e-20), np.log(1e-18)), 
             1: uniform_dist(5e-5, 1e-1),
             2: uniform_dist(5e-15, 0.5e-10)}

priors = PriorContainer(priors_in)
# can add extra temperatures of 1 to have multiple temps accessing the target distribution
ntemps_target_extra = 0
# define max temperature (generally should be inf if you want to sample prior)
Tmax = np.inf

# not all walkers can fit in memory. subset says how many to do at one time
subset = 2

# set kwargs for the templates
waveform_kwargs_templates = waveform_kwargs.copy()

# random starts
np.random.seed(3000)
start_points = np.zeros((nwalkers * ntemps, ndim))
for i in range(ndim):
    start_points[:, i] = np.random.multivariate_normal(injection_params[test_inds], cov, size=nwalkers*ntemps)[:,i] #priors[i].rvs(nwalkers * ntemps) #


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


#%% setup sampler
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
    plot_source="gb",
    fp="mono_no_noise.h5",
    resume=False
)

thin_by = 1
max_iter = 1000
sampler.sample(
    start_points,
    iterations=max_iter,
    progress=True,
    skip_initial_state_check=False,
    thin_by=thin_by,
)

# %%
