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


from eryn.prior import uniform_dist, MappedUniformDistribution
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR
from lisatools.sampling.moves.gbspecialstretch import GBSpecialStretchMove
from lisatools.utils.utility import AET

try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice

    # setDevice(0)

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

from eryn.prior import ProbDistContainer

from eryn.state import State
from eryn.state import BranchSupplimental

from lisatools.sampling.moves.gbmultipletryrj import GBMutlipleTryRJ
from lisatools.sensitivity import flat_psd_function, noisepsd_AE

# from lisatools.sampling.moves.globalfish import MultiSourceFisherProposal

import warnings

warnings.filterwarnings("ignore")

from lisatools.sampling.moves.placeholder import PlaceHolder


try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice

    # setDevice(7)

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

random_seed = 1024
np.random.seed(random_seed)
# warnings.filterwarnings("ignore")

oversample = 4

min_k = 0

current_param_file = "LDC2_sangria_training_v2.h5"
current_injections_file = "full_band_current_injections.npy"
current_snrs_file = "full_band_current_snr_individual.npy"
fp_mix = "full_band_current_mix_stepping_temp_file.h5"
fp_mix_final = fp_mix[:-3] + "_final.h5"
current_start_points_file = "full_band_current_start_points.npy"
current_start_points_snr_file = "full_band_current_snrs_search.npy"
sub_band_fails_file = "full_band_sub_band_fails.npy"
current_residuals_file_iterative_search = (
    "full_band_current_residuals_file_iterative_search.npy"
)
current_iterative_search_sub_file_base = (
    "full_band_current_iterative_search_sub_file_base"
)
current_found_coords_for_starting_mix_file = (
    "full_band_current_found_coords_for_starting_mix_file.npy"
)
current_save_state_file = "save_state_temp.pickle"
current_save_state_file_psd = "psd_save_state_temp.pickle"

num_sub_band_fails_limit = 2

use_gpu = gpu_available
xp = xp if use_gpu else np

gb = GBGPU(use_gpu=use_gpu)

ndim_full = 9

delta_safe = 1e-5

A_lims = [7e-24, 1e-21]
f0_lims = [0.05e-3, 2.5e-2]
m_chirp_lims = [0.001, 1.0]
fdot_lims = [get_fdot(f0_lims[i], Mc=m_chirp_lims[i]) for i in range(len(f0_lims))]
phi0_lims = [0.0, 2 * np.pi]
iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
psi_lims = [0.0, np.pi]
lam_lims = [0.0, 2 * np.pi]
beta_sky_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

nleaves_max = 2000
ndim = 8
ntemps = 6
ntemps_pe = 8
nwalkers = 100
branch_names = ["gb", "gb_fixed", "psd", "galfor"]


fp_gb = "last_gb_cold_chain_info"
fp_psd = "last_psd_cold_chain_info"
fp_mbh = "last_mbh_cold_chain_info"
fp_mbh_template_search = "last_mbh_cold_chain_info_search"
fp_psd_residual_search = "last_psd_cold_chain_info_search"
fp_gb_template_search = "last_gb_cold_chain_info_search"

fp_psd_search_initial = "initial_search_develop_full_band_psd_2.h5"
fp_psd_search = "search_develop_full_band_psd_3.h5"
fp_mbh_search = "search_develop_full_band_mbh.h5"

fp_pe = "develop_zero_temps_full_band_19.h5"
fp_psd_pe = "develop_full_band_psd_3.h5"
fp_mbh_pe = "develop_full_band_mbh.h5"
folder = "./"
import os

fp_old = fp_pe  # "full_half_mHz_band_pe_output_after_prior_extension.h5"

with h5py.File("LDC2_sangria_training_v2.h5") as f:
    tXYZ = f["obs"]["tdi"][:]

    # remove mbhb and igb
    for source in ["igb", "vgb"]:  # "mbhb",
        change_arr = f["sky"][source]["tdi"][:]
        for change in ["X", "Y", "Z"]:
            tXYZ[change] -= change_arr[change]

    # tXYZ = f["sky"]["dgb"]["tdi"][:]

t, X, Y, Z = (
    tXYZ["t"].squeeze(),
    tXYZ["X"].squeeze(),
    tXYZ["Y"].squeeze(),
    tXYZ["Z"].squeeze(),
)
dt = t[1] - t[0]

Nobs = len(t)
if Nobs > int(YEAR / dt):
    Nobs = int(YEAR / dt)
    t = t[:Nobs]
    X = X[:Nobs]
    Y = Y[:Nobs]
    Z = Z[:Nobs]

Tobs = Nobs * dt
df = 1 / Tobs

buffer = 2**12
fmin = 0.0  # f0_lims[0] - buffer * df
fmax = f0_lims[1]
start_freq_ind = int(fmin / df)
end_freq_ind = int(fmax / df)
data_length = int(fmax / df) - start_freq_ind + 1

# fucking dt
Xf, Yf, Zf = (np.fft.rfft(X) * dt, np.fft.rfft(Y) * dt, np.fft.rfft(Z) * dt)
Af, Ef, Tf = AET(Xf, Yf, Zf)
A_inj, E_inj = (
    Af[start_freq_ind : end_freq_ind + 1],
    Ef[start_freq_ind : end_freq_ind + 1],
)

fd = np.arange(start_freq_ind, end_freq_ind + 1) * df

# np.save("check_sens", np.array([fd, A_inj, E_inj]))

psd = get_sensitivity(fd, **{"sens_fn": "noisepsd_AE", "model": "sangria"}).squeeze()

psd[0] = 1e100

psd_in = [
    psd.copy(),
    psd.copy(),
]

supps_base_shape = (ntemps, nwalkers)

# TODO: check beta versus theta
waveform_kwargs = dict(
    N=None, dt=dt, T=Tobs, use_c_implementation=True, oversample=oversample
)

nfriends = 40
L = 2.5e9

amp_transform = AmplitudeFromSNR(L, Tobs, model="sangria")

# TODO: special move storage of information in backend
# TODO: think about if there is anything else that would nice to store in the backend.

transform_fn_in = {
    # 0: (lambda x: np.exp(x)),
    1: (lambda x: x * 1e-3),
    5: (lambda x: np.arccos(x)),
    8: (lambda x: np.arcsin(x)),
    # (1, 2, 3): (lambda f0, fdot, fddot: (f0, fdot, 11 / 3.0 * fdot ** 2 / f0)),
    (0, 1): amp_transform,
}


fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}
transform_fn = TransformContainer(
    parameter_transforms=transform_fn_in, fill_dict=fill_dict
)

rho_star = 5.0
snr_prior = SNRPrior(rho_star)

det_gb_groups = DetermineGBGroups(
    gb,
    transform_fn={"gb": transform_fn, "gb_fixed": transform_fn},
    waveform_kwargs=waveform_kwargs,
)
waveform_kwargs_tmp = waveform_kwargs.copy()
waveform_kwargs_tmp["start_freq_ind"] = start_freq_ind
get_last_gb_state = GetLastGBState(
    gb,
    transform_fn={"gb": transform_fn, "gb_fixed": transform_fn},
    waveform_kwargs=waveform_kwargs_tmp,
)
default_priors_gb = {
    0: snr_prior,
    1: uniform_dist(
        *(np.asarray(f0_lims) * 1e3)
    ),  # special mapping for RJ (we care about changes in prior, uniform there are no changes)
    2: uniform_dist(*fdot_lims),
    3: uniform_dist(*phi0_lims),
    4: uniform_dist(*np.cos(iota_lims)),
    5: uniform_dist(*psi_lims),
    6: uniform_dist(*lam_lims),
    7: uniform_dist(*np.sin(beta_sky_lims)),
    # (0, 1): SNRPrior(10.0, Tobs),
}

generate_dists = deepcopy(default_priors_gb)

snr_lim = inital_snr_lim = 45.0  # 11.0  # 85.0
dSNR = 40.0
generate_dists[0] = uniform_dist(snr_lim, snr_lim + dSNR)
generate_snr_ladder = ProbDistContainer(generate_dists)

priors_psd = {
    0: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
    1: uniform_dist(2.0e-15, 20.0e-15),  # Sa_a
    2: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
    3: uniform_dist(2.0e-15, 20.0e-15),  # Sa_a
}

priors_galfor = {
    0: uniform_dist(1e-45, 2e-43),  # amp
    1: uniform_dist(1.0, 3.0),  # alpha
    2: uniform_dist(5e1, 1e7),  # Slope1
    3: uniform_dist(1e-4, 5e-2),  # knee
    4: uniform_dist(5e1, 8e3),  # Slope2
}

priors = {
    "gb": ProbDistContainer(default_priors_gb),
    "gb_fixed": ProbDistContainer(default_priors_gb),
    "psd": ProbDistContainer(priors_psd),
    "galfor": ProbDistContainer(priors_galfor),
}

# temp = injection_params[:, 0].copy()
# injection_params[:, 0] = injection_params[:, -1].copy()
# injection_params[:, -1] = temp.copy()

periodic = {
    "gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi},
    "gb_fixed": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi},
    "psd": {},
    "galfor": {},
}


# adjust for frequencies
f0_lims_in = f0_lims.copy()

# TODO: make wider because this is knowning the limits?
f0_lims_in[0] = 0.3e-3
# f0_lims_in[1] = 0.8e-3

low_fs = np.arange(f0_lims_in[0], 0.001 - 4 * 128 * df, 2 * 128 * df)
mid_fs = np.arange(0.001, 0.01 - 4 * 512 * df, 2 * 256 * df)
high_fs = np.append(
    np.arange(0.01, f0_lims_in[-1], 2 * 1024 * df), np.array([f0_lims_in[-1]])
)

# breakpoint()
"""
# low_fs = np.append(np.arange(f0_lims_in[0], 0.001, 2 * 128 * df), np.array([0.001]))
# mid_fs = np.arange(0.001 + 256 * 2 * df, 0.01, 2 * 256 * df)
# high_fs = np.append(np.arange(0.01, f0_lims_in[-1], 2 * 1024 * df), np.array([f0_lims_in[-1]]))
"""
search_f_bin_lims = np.concatenate([low_fs, mid_fs, high_fs])

low_fs_propose = np.arange(f0_lims_in[0], 0.001 - 2 * 128 * df, 128 * df)
mid_fs_propose = np.arange(0.001, 0.01 - 2 * 512 * df, 256 * df)
high_fs_propose = np.append(
    np.arange(0.01, f0_lims_in[-1], 1024 * df), np.array([f0_lims_in[-1]])
)

propose_f_bin_lims = np.concatenate([low_fs_propose, mid_fs_propose, high_fs_propose])
search_f_bin_lims = propose_f_bin_lims
# for testing
# search_f_bin_lims = np.arange(f0_lims_in[0], f0_lims_in[1], 2 * 128 * df)

num_sub_bands = len(search_f_bin_lims)


if current_snrs_file in os.listdir():
    snrs_individual = list(np.load(current_snrs_file))

else:
    with h5py.File("LDC2_sangria_training_v2.h5") as f:
        dgb = f["sky"]["dgb"]["cat"][:]

    # out_params = np.load(current_param_file)
    params_keys = [
        "Amplitude",
        "Frequency",
        "FrequencyDerivative",
        "InitialPhase",
        "Inclination",
        "Polarization",
        "EclipticLongitude",
        "EclipticLatitude",
    ]
    f0 = dgb["Frequency"].squeeze()
    keep_inds = np.where((f0 < f0_lims[-1]) & (f0 > f0_lims[0]))[0]

    params = [dgb[key][keep_inds] for key in params_keys]
    params.insert(3, np.zeros_like(params[0]))
    out_params = np.asarray(params).squeeze().T

    f0 = out_params[:, 1]

    # keep = np.where((f0 > f0_lims[0]) & (f0 < f0_lims[1]))
    # breakpoint()
    # out_params = out_params[keep]
    assert out_params.shape[1] == 9
    out_params[:, 3] = 0.0
    check_injection = out_params.copy()
    num_bin = out_params.shape[0]

    (
        amp_in,
        f0_in,
        fdot_in,
        fddot_in,
        phi0_in,
        iota_in,
        psi_in,
        lam_in,
        beta_sky_in,
    ) = out_params.T.copy()
    # phi0 is flipped !
    phi0_in *= -1.0

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

    injection_temp = transform_fn.transform_base_parameters(
        injection_params.T, return_transpose=False
    ).reshape(-1, ndim_full)

    xp.cuda.runtime.setDevice(7)
    gb.d_d = 0.0

    _ = gb.get_ll(
        out_params,
        xp.asarray([A_inj, E_inj])[None, :],
        xp.asarray(psd_in)[None, :],
        start_freq_ind=start_freq_ind,
        **waveform_kwargs
    )

    optimal_snrs = gb.h_h.real.get() ** (1 / 2)

    np.save(current_snrs_file[:-4], np.asarray(optimal_snrs))
    snrs_individual = list(np.load(current_snrs_file))

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
"""with h5py.File("LDC2_sangria_training_v2.h5") as f:
    dgb = f["sky"]["dgb"]["cat"][:]
    # out_params = np.load(current_param_file)
    params_keys = ["Amplitude", "Frequency", "FrequencyDerivative", "InitialPhase", "Inclination", "Polarization", "EclipticLongitude", "EclipticLatitude"]
    f0 = dgb["Frequency"].squeeze()
    keep_inds = np.where((f0 < f0_lims[-1]) & (f0 > f0_lims[0]))[0]

    params = [dgb[key][keep_inds] for key in params_keys]
    params.insert(3, np.zeros_like(params[0]))
    out_params = np.asarray(params).squeeze().T

f0_inj = out_params[:, 1]
snrs_inj = np.load("full_band_current_snr_individual.npy")
check_points = np.load("max_ll_points.npy")  # full_band_current_start_points.npy")
check_points_in = transform_fn.both_transforms(check_points)
sub_band_fails = np.load(sub_band_fails_file)

assert len(sub_band_fails) == len(search_f_bin_lims)
f0_found = check_points_in[:, 1]
for i, (start_freq, end_freq) in enumerate(zip(search_f_bin_lims[:-1], search_f_bin_lims[1:])):
    plt.close()
    fig = plt.figure()
    fig.set_size_inches(8, 7)
    inds_found = np.where((f0_found > start_freq) & (f0_found < end_freq))
    num_found = len(inds_found[0])
    if num_found == 0:
        continue

    inds_inj_keep1 = np.where((f0_inj > start_freq) & (f0_inj < end_freq) & (snrs_inj >= 10.0))
    inds_inj_keep2 = np.where((f0_inj > start_freq) & (f0_inj < end_freq) & (snrs_inj < 10.0) & (snrs_inj >= 7.0))
    inds_inj_not = np.where((f0_inj > start_freq) & (f0_inj < end_freq) & (snrs_inj < 7.0))
    plt.scatter(out_params[inds_inj_keep1, 1] * 1e3, np.log10(out_params[inds_inj_keep1, 0]), s=snrs_inj[inds_inj_keep1] * 2, color="C0", label="inj (snr >= 10)")
    plt.scatter(out_params[inds_inj_keep2, 1] * 1e3, np.log10(out_params[inds_inj_keep2, 0]), marker="^", s=snrs_inj[inds_inj_keep2] * 2, color="C0", label="inj (snr >= 7)")
    plt.scatter(out_params[inds_inj_not, 1] * 1e3, np.log10(out_params[inds_inj_not, 0]), s=3,  marker="x", color="k", label="inj (snr < 10)")
    plt.scatter(check_points_in[inds_found, 1] * 1e3, np.log10(check_points_in[inds_found, 0]), s=15, color="C1", label="found")
    plt.legend(loc="upper center", ncol=4, bbox_to_anchor=[0.5, 1.05], fancybox=True, framealpha=1.0)
    plt.xlim(start_freq * 1e3, end_freq * 1e3)
    plt.title(f"{start_freq} -- {end_freq}\nnum found: {num_found} / not found times: {sub_band_fails[i]}", y=1.05, ha="center", va="center")
    plt.xlabel(r"$f$ (mHz)")
    plt.ylabel(r"$\log_10(A)$ (mHz)")
    plt.savefig(f"figs/band_output_max_ll_{start_freq}_{end_freq}.png", dpi=100)
    print(i, start_freq, end_freq)

"""

"""reader = HDFBackend("test_full_band_posterior_0.h5")
last = reader.get_last_sample()
ind_max = np.where(last.log_prob == last.log_prob.max())
coords_out = transform_fn.both_transforms(last.branches["gb_fixed"].coords[ind_max].squeeze())[:, np.array([0, 1, 2, 4, 5, 6, 7, 8])]
breakpoint()
np.save("coords_for_video", coords_out)
"""

"""print("START")
fd_here = fd[0::10]
num = 100

psd_pars = priors["psd"].rvs(size=(num, 2))
galfor_pars = priors["galfor"].rvs(size=(num,))

psds_out = np.zeros((num, 2, len(fd)))
for i, (psd_par, galfor_par) in enumerate(zip(psd_pars, galfor_pars)):
    for j in range(2):
        psds_out[i, j] = get_sensitivity(fd, sens_fn="noisepsd_AE", model=psd_par[j], foreground_params=galfor_par)
    print(i)

np.save(fp_psd, psds_out)

#plt.loglog(fd, 2 * df * np.abs(A_inj) ** 2)
# plt.loglog(fd, 2 * df * np.abs(E_inj) ** 2)
#Sn = get_sensitivity(fd_here, sens_fn="noisepsd_AE", model=np.array([7.79e-12, 3.418e-15]))
#plt.loglog(fd_here, Sn)
#Sn = get_sensitivity(fd_here, sens_fn="noisepsd_AE", model="sangria", includewd=1.0)
#plt.loglog(fd_here, Sn, color="k", lw=2, ls="--")
plt.savefig("check_match_A.png")
breakpoint()"""
