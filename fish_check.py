#%% Check with Fisher Matrix
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
from lisatools.sampling.prior import uniform_dist

from lisatools.utils.transform import TransformContainer

import warnings

warnings.filterwarnings("ignore")


use_gpu = False
fast = GenerateEMRIWaveform(
    "Pn5AAKWaveform",
    sum_kwargs=dict(pad_output=True),
    use_gpu=use_gpu,
    return_list=True,
)

fast_not_list = GenerateEMRIWaveform(
    "Pn5AAKWaveform",
    sum_kwargs=dict(pad_output=True),
    use_gpu=use_gpu,
    return_list=False,
)

###############################################
def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')

def window_zero(sig,dt):
    from scipy import signal

    Length = len(sig)
    return zero_pad(sig * signal.tukey(Length, 0.05))
###########################


# injection parameters
M = 1e6
mu = 3.012e01
p0 = 10.0
e0 = 0.267

print("M,mu,p0,e0", M,mu,p0,e0)

# other params
Phi_phi0 = np.pi / 4.0
Phi_r0 = np.pi / 3.0
dist = 4.10864264e00
a = 0.5
Y0 = 0.7
qS = np.pi / 3.0
phiS = np.pi / 5.0
qK = np.pi / 6.0
phiK = 5 * np.pi / 12.0
Phi_theta0 = 0.0

#
T = 12/12 # years
dt = 10.0  # seconds

# injection array
injection_params = np.array(
    [
        np.log(M),
        np.log(mu),
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


##############################################
# Fix SNR
snr_goal = 30.0

# for SNR and covariance calculation
inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd")


# transformation of arguments from sampling basis to waveform basis
transform_fn_in = {0: (lambda x: np.exp(x)), 1: (lambda x: np.exp(x))}


# use the special transform container
transform_fn = TransformContainer(transform_fn_in)

# copy parameters to check values
check_params = injection_params.copy()

# this transforms but also gives the transpose of the input dimensions due to inner sampler needs
check_params = transform_fn.transform_base_parameters(check_params).T

#########################################
# Waveform
# INJECTION kwargs
waveform_kwargs = {"T": T, "dt": dt}#, "mode_selection": [(2,2,0)]}

check_sig = fast(*check_params, **waveform_kwargs)

#########################################
# adjust distance for SNR goal
check_sig, snr_orig = scale_snr(snr_goal, check_sig, return_orig_snr=True,**inner_product_kwargs) #, dt=dt

print("orig_dist:", injection_params[6])
injection_params[6] *= snr_orig / snr_goal
print("new_dist:", injection_params[6])

# copy again
params_test = injection_params.copy()
params_test = transform_fn.transform_base_parameters(params_test).T

check_sig = fast(*params_test, **waveform_kwargs)

print("snr goal reached",snr(check_sig,**inner_product_kwargs))

########################################
#%% Fisher matrix

# which of the injection parameters you actually want to sample over
test_inds = np.array([0, 1, 2, 3, 4, 5, 6])

check_step = False
if check_step:

    # get covariance
    plt.figure()
    vec_eps = 10**np.arange(-9,-2.5, 0.5) #[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    for eps in vec_eps:
        #eps = [1e-5, 1e-5, 1e-8, 1e-8]
        cov, fish, dh = covariance(
            fast_not_list,
            injection_params,
            eps,
            deriv_inds=test_inds,
            parameter_transforms=transform_fn,
            waveform_kwargs=waveform_kwargs,
            inner_product_kwargs=inner_product_kwargs,
            diagonalize=False,
            return_fisher=True,
            return_derivs=True,
            accuracy=True
        )
        plt.loglog(eps, np.sqrt(fish.diagonal()[0]),'.')#, label=str(eps))
        plt.loglog(eps, np.sqrt(fish.diagonal()[1]),'x')
        plt.loglog(eps, np.sqrt(fish.diagonal()[2]),'*')
        plt.loglog(eps, np.sqrt(fish.diagonal()[3]),'o')

        #plt.plot(dh[0], alpha=0.7,label=str(eps))
    #print(np.diag(fish))
    plt.legend()
    plt.show()

eps = 1e-6
cov, fish, dh = covariance(
    fast_not_list,
    injection_params,
    eps,
    deriv_inds=test_inds,
    parameter_transforms=transform_fn,
    waveform_kwargs=waveform_kwargs,
    inner_product_kwargs=inner_product_kwargs,
    diagonalize=False,
    return_fisher=True,
    return_derivs=True,
    accuracy=True
)

sig_diag = np.sqrt(cov.diagonal())

#print(inner_product(np.real(dh[0]),np.real(dh[0]),**inner_product_kwargs) + inner_product(np.imag(dh[0]),np.imag(dh[0]),**inner_product_kwargs), fish[0,0])
likelike = True
if likelike:

    ############################################
    #%% MCMC check

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

    # set kwargs for the templates
    waveform_kwargs_templates = waveform_kwargs.copy()
    #waveform_kwargs_templates["eps"] = 1e-2

    #%% Check inner product
    # single value likelihood
    like.get_ll(np.array([injection_params]),waveform_kwargs=waveform_kwargs)
    # copy again
    true_params = injection_params.copy()
    true_params = transform_fn.transform_base_parameters(true_params).T

    true_sig = fast(*true_params, **waveform_kwargs)

    Tick = ['$\log M$', '$\log \mu$', '$e$', '$p$']
    for i in range(len(test_inds)):
        
        plt.figure()
        plt.title('Fisher matrix width prediction against likelihood  for parameter '+str(i))
        for j in np.arange(-2, 2, 1e-1):
            pert_params = injection_params.copy()
            pert_params[test_inds[i]] = pert_params[test_inds[i]] + j * fish[i,i]**(-0.5)
            p1 = plt.plot(j,np.exp(-like.get_ll(np.array([pert_params]),waveform_kwargs=waveform_kwargs)),'r.')

        x = np.arange(-2,2,1e-1)
        gaus = np.exp(-x*x/(2))
        plt.plot(x,gaus,label='theoretical likelihood')
        plt.legend()
        plt.savefig('plot_check_fish_var'+str(i) +'_aak.png')
        #plt.show()

    #
    print('all right')

