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

from few.utils.constants import *
from few.waveform import FastSchwarzschildEccentricFlux

fast = FastSchwarzschildEccentricFlux(sum_kwargs=dict(pad_output=True))
M = 1e6
mu = 5e1
p0 = 12.0
e0 = 0.3
phi = 0.0
theta = np.pi / 3.0
dist = 10.0

T = 1.0
dt = 10.0

transform_fn = {0: np.exp}

params = np.array([np.log(M), mu, p0, e0, theta, phi, dist])

params_copy = params.copy()

waveform_kwargs = {"T": T, "dt": dt, "eps": 1e-5}

eps = 1e-9

inner_product_kwargs = dict(frequency_domain=False, PSD="cornish_lisa_psd")

sig1 = fast(M, mu, p0, e0, theta, phi, dist=dist, T=T, dt=dt)

inner = inner_product(
    [sig1.real, sig1.imag], [sig1.real, sig1.imag], dt=dt, PSD="cornish_lisa_psd",
)

sig1_fft = [np.fft.rfft(sig1.real)[1:] * dt, np.fft.rfft(sig1.imag)[1:] * dt]
df = 1 / (dt * len(sig1))

inner_FD = inner_product(sig1_fft, sig1_fft, df=df, PSD="cornish_lisa_psd",)
# sig1 = scale_snr(
#    20.0, [sig1.real, sig1.imag], dt, frequency_domain=False, PSD="cornish_lisa_psd"
# )
# check2 = snr(sig1, dt, frequency_domain=False, PSD="cornish_lisa_psd")
breakpoint()

deriv_inds = np.array([0, 2, 3])
# fish = fisher(
#    fast,
#    params,
#    eps,
#    dt,
#    deriv_inds=deriv_inds,
#    parameter_transforms=transform_fn,
#    waveform_kwargs=waveform_kwargs,
#    inner_product_kwargs=inner_product_kwargs,
# )

"""
cov = covariance(
    fast,
    params,
    eps,
    dt,
    deriv_inds=deriv_inds,
    parameter_transforms=transform_fn,
    waveform_kwargs=waveform_kwargs,
    inner_product_kwargs=inner_product_kwargs,
    diagonalize=False,
)


out = mismatch_criterion(
    fast,
    params,
    eps,
    dt,
    deriv_inds=deriv_inds,
    parameter_transforms=transform_fn,
    waveform_kwargs=waveform_kwargs,
    inner_product_kwargs=inner_product_kwargs,
)
"""

waveform_approx_kwargs = waveform_kwargs.copy()

waveform_approx_kwargs["eps"] = 1e-2

out = cutler_vallisneri_bias(
    fast,
    fast,
    params_copy,
    eps,
    dt,
    return_fisher=True,
    return_derivs=False,
    return_cov=False,
    deriv_inds=deriv_inds,
    parameter_transforms=transform_fn,
    waveform_true_kwargs=waveform_kwargs,
    waveform_approx_kwargs=waveform_approx_kwargs,
    inner_product_kwargs=inner_product_kwargs,
)

breakpoint()


p02 = 10.0001
sig1 = fast(M, mu, p0, e0, theta, phi, dist, T=T, dt=dt)
sig2 = fast(M, mu, p02, e0, theta, phi, dist, T=T, dt=dt)

check1 = inner_product(
    [sig1.real, sig1.imag],
    [sig2.real, sig2.imag],
    dt,
    frequency_domain=False,
    PSD="cornish_lisa_psd",
)

breakpoint()
check2 = snr([sig1.real, sig1.imag], dt, frequency_domain=False, PSD="cornish_lisa_psd")

check3 = inner_product(
    [sig1.real, -sig1.imag],
    [sig2.real, -sig2.imag],
    dt,
    frequency_domain=False,
    PSD="cornish_lisa_psd",
    normalize=True,
)
breakpoint()

f = np.logspace(-7, 0, 1000)

plt.loglog(f, get_sensitivity(f, sens_fn="cornish_lisa_psd"))

plt.loglog(f, get_sensitivity(f, sens_fn="lisasens"))

plt.loglog(f, get_sensitivity(f, sens_fn="noisepsd_X"))

plt.loglog(f, get_sensitivity(f, sens_fn="noisepsd_X2", model="SciRDv1"))
plt.show()
