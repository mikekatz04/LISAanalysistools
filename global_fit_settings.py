from tkinter import Place
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from shutil import copyfile
import h5py

from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.utils.utility import groups_from_inds
from eryn.utils import TransformContainer


from eryn.prior import uniform_dist
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR
from lisatools.sampling.moves.gbspecialstretch import GBSpecialStretchMove

try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice
    setDevice(2)

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
from lisatools.sensitivity import get_sensitivity

from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_fdot

from gbgpu.utils.constants import *

from lisatools.sampling.likelihood import GlobalLikelihood, Likelihood

from lisatools.sensitivity import get_sensitivity
from lisatools.sampling.utility import DetermineGBGroups, GetLastGBState

from eryn.prior import PriorContainer

from eryn.state import State
from eryn.moves import StretchMoveRJ
from eryn.state import BranchSupplimental

from lisatools.sampling.moves.gbmultipletryrj import GBMutlipleTryRJ
from lisatools.sensitivity import flat_psd_function
# from lisatools.sampling.moves.globalfish import MultiSourceFisherProposal

import warnings

from lisatools.sampling.moves.placeholder import PlaceHolder

random_seed = 1025
np.random.seed(random_seed)
#warnings.filterwarnings("ignore")

min_k = 0

current_param_file = "out_params_3.5_to_4Hz.npy"
current_injections_file = "current_injections.npy"
current_snrs_file = "current_snr_individual.npy"
fp_mix = "current_mix_stepping_temp_file.h5"
fp_mix_final = fp_mix[:-3] + "_final.h5"
current_start_points_file = "current_start_points.npy"
current_start_points_snr_file = "current_snrs_search.npy"

use_gpu = gpu_available
xp = xp if use_gpu else np

gb = GBGPU(use_gpu=use_gpu)

ndim_full = 9

A_lims = [7e-24, 1e-21]
f0_lims = [3.5e-3, 4e-3]
m_chirp_lims = [0.01, 1.0]
fdot_lims = [get_fdot(f0_lims[i], Mc=m_chirp_lims[i]) for i in range(len(f0_lims))]
phi0_lims = [0.0, 2 * np.pi]
iota_lims = [0.0, np.pi]
psi_lims = [0.0, np.pi]
lam_lims = [0.0, 2 * np.pi]
beta_sky_lims = [-np.pi / 2.0, np.pi / 2.0]

Tobs = 1.0 * YEAR
dt = 15.0
Tobs = int(Tobs / dt) * dt
df = 1 / Tobs

nleaves_max = 2000
ndim = 8
ntemps = 1
ntemps_pe = 8
nwalkers = 100
branch_names = ["gb", "gb_fixed", "noise_params"]

fp = "full_half_mHz_band_pe_output_after_prior_extension.h5"
folder = "./"
import os
fp_old = "full_half_mHz_band_pe_output.h5"

buffer = 2 ** 12
fmin = f0_lims[0] - buffer * df
fmax = f0_lims[1] + buffer * df
start_freq_ind = int(fmin / df)
end_freq_ind = int(fmax / df)
data_length = int(fmax / df) - start_freq_ind + 1

base_psd_val = np.mean(get_sensitivity(np.mean(f0_lims), **{"sens_fn": "noisepsd_AE"}))

fd = np.arange(start_freq_ind, end_freq_ind + 1) * df

psd = flat_psd_function(fd, base_psd_val, xp=np).squeeze()

psd_in = [
    psd.copy(),
    psd.copy(),
]

supps_base_shape = (ntemps, nwalkers)

out_params = np.load(current_param_file)

f0 = out_params[:, 1]
#keep = np.where((f0 > f0_lims[0]) & (f0 < f0_lims[1]))
#breakpoint()
#out_params = out_params[keep]
assert out_params.shape[1] == 9
out_params[:, 3] = 0.0
check_injection = out_params.copy()

periodic = {"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}, "gb_fixed": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}, "noise_params": {}}

from gbgpu.utils.utility import get_N

N = get_N(out_params[:, 0], out_params[:, 1], Tobs, oversample=4).max()

amp_in, f0_in, fdot_in, fddot_in, phi0_in, iota_in, psi_in, lam_in, beta_sky_in = out_params.T.copy()
# phi0 is flipped !
phi0_in *= -1.

# TODO: check beta versus theta
waveform_kwargs = dict(N=N, dt=dt, T=Tobs, use_c_implementation=True)

nfriends = 40
L = 2.5e9

amp_transform = AmplitudeFromSNR(L, Tobs)
transform_fn_in = {
    #0: (lambda x: np.exp(x)),
    1: (lambda x: x * 1e-3),
    5: (lambda x: np.arccos(x)),
    8: (lambda x: np.arcsin(x)),
    #(1, 2, 3): (lambda f0, fdot, fddot: (f0, fdot, 11 / 3.0 * fdot ** 2 / f0)),
    (0, 1): amp_transform
}


fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}
transform_fn = TransformContainer(
    parameter_transforms=transform_fn_in, fill_dict=fill_dict
)

rho_star = 5.0
snr_prior = SNRPrior(rho_star)

det_gb_groups = DetermineGBGroups(gb, transform_fn={"gb": transform_fn, "gb_fixed": transform_fn}, waveform_kwargs=waveform_kwargs)
waveform_kwargs_tmp = waveform_kwargs.copy()
waveform_kwargs_tmp["start_freq_ind"] = start_freq_ind
get_last_gb_state = GetLastGBState(gb, transform_fn={"gb": transform_fn, "gb_fixed": transform_fn}, waveform_kwargs=waveform_kwargs_tmp)

injection_params = np.array(
    [
        amp_transform.forward(amp_in, f0_in)[0],
        f0_in * 1e3,
        fdot_in,
        fddot_in,
        phi0_in,
        np.cos(iota_in),
        psi_in,
        lam_in,
        np.sin(beta_sky_in),
    ]
)

default_priors_gb = {
    0: snr_prior,
    1: uniform_dist(*(np.asarray(f0_lims) * 1e3)),
    2: uniform_dist(*fdot_lims),
    3: uniform_dist(*phi0_lims),
    4: uniform_dist(*np.cos(iota_lims)),
    5: uniform_dist(*psi_lims),
    6: uniform_dist(*lam_lims),
    7: uniform_dist(*np.sin(beta_sky_lims)),
    #(0, 1): SNRPrior(10.0, Tobs),
}

generate_dists = deepcopy(default_priors_gb)

snr_lim = inital_snr_lim = 45.0 # 11.0  # 85.0
dSNR = 40.0
generate_dists[0] = uniform_dist(snr_lim, snr_lim + dSNR)
generate_snr_ladder = PriorContainer(generate_dists)

priors_noise = {
    0: uniform_dist(0.1 * base_psd_val, 10.0 * base_psd_val)
}

priors = {"gb": PriorContainer(default_priors_gb), "gb_fixed": PriorContainer(default_priors_gb), "noise_params": PriorContainer(priors_noise)}

# temp = injection_params[:, 0].copy()
# injection_params[:, 0] = injection_params[:, -1].copy()
# injection_params[:, -1] = temp.copy()

injection_temp = transform_fn.transform_base_parameters(
    injection_params.T, return_transpose=False
).reshape(-1, ndim_full)

num_bin = out_params.shape[0]
A_temp_all = []
num_bin_injected = num_bin - num_bin
snrs_individual = []

if current_injections_file in os.listdir():
    A_inj, E_inj = np.load(current_injections_file)
    snrs_individual = list(np.load(current_snrs_file))

else:
    # inject first signal to get a data stream
    A_inj, E_inj = gb.inject_signal(*injection_temp[0], **waveform_kwargs,)
    A_inj, E_inj = A_inj[start_freq_ind:end_freq_ind + 1], E_inj[start_freq_ind:end_freq_ind + 1]
    
    A_temp_all.append(A_inj)

    snrs_individual.append(snr([A_inj, E_inj], f_arr=fd, PSD="noisepsd_AE",))

    for i in range(1, num_bin):
        A_temp, E_temp = gb.inject_signal(*injection_temp[i], **waveform_kwargs,)
        A_temp, E_temp = A_temp[start_freq_ind:end_freq_ind + 1], E_temp[start_freq_ind:end_freq_ind + 1]
        A_inj += A_temp
        E_inj += E_temp

        A_temp_all.append(A_temp)

        snrs_individual.append(snr([A_temp, E_temp], f_arr=fd, PSD="noisepsd_AE",))
        
    np.save(current_injections_file[:-4], np.array([A_inj, E_inj]))
    np.save(current_snrs_file[:-4], np.asarray(snrs_individual))

from lisatools.utils.utility import generate_noise_fd
A_noise = generate_noise_fd(fd, df, base_psd_val, sens_fn=flat_psd_function).squeeze()
E_noise = generate_noise_fd(fd, df, base_psd_val, sens_fn=flat_psd_function).squeeze()

A_inj_orig = A_inj.copy()
E_inj_orig = E_inj.copy()

A_inj = A_inj  # + A_noise
E_inj = E_inj  # + E_noise

"""
plt.semilogy(np.abs(A_inj), color="C0", lw=2)
#for tmp in A_temp_all:
#    plt.semilogy(np.abs(tmp))

#plt.semilogy(np.abs(A_inj_orig))
reader = HDFBackend(fp_old)
last = reader.get_last_sample()

ind_max = np.where(last.log_prob == last.log_prob.max())

best_coords_gb = last.branches["gb"].coords[ind_max][last.branches["gb"].inds[ind_max]]
best_coords_gb_fixed = last.branches["gb_fixed"].coords[ind_max][last.branches["gb_fixed"].inds[ind_max]]

best_coords = np.concatenate([best_coords_gb, best_coords_gb_fixed], axis=0)
best_coords_in = transform_fn.both_transforms(best_coords)

data_index = xp.zeros(len(best_coords), dtype=np.int32)
templates = xp.zeros((1, 2, psd.shape[0]), dtype=complex)
gb.generate_global_template(best_coords_in, data_index, templates, start_freq_ind=start_freq_ind, **waveform_kwargs)
A_best = templates[0, 0].get()
plt.semilogy(np.abs(A_best), color="C1", lw=2, ls="--")

plt.xlim(4100,6000)
plt.savefig("plot101.png")
plt.close()
np.save("As_out", np.array([fd, A_inj, A_best]))
"""

"""check = snr_prior.rvs(size=(1000000))
rho = np.linspace(0.0, 1000, 100000)
pdf = snr_prior.pdf(rho)
plt.close()
plt.hist(check, bins=np.arange(1000), density=True)
plt.plot(rho, pdf)
plt.savefig("plot1.png")
breakpoint()
"""

