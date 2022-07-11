from tkinter import Place
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler

from eryn.prior import uniform_dist
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR

try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice
    setDevice(5)

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

from lisatools.sampling.likelihood import GlobalLikelihood

from lisatools.sensitivity import get_sensitivity

from eryn.prior import PriorContainer

from eryn.state import State
from eryn.moves import StretchMoveRJ

from lisatools.sampling.moves.gbmultipletryrj import GBMutlipleTryRJ

# from lisatools.sampling.moves.globalfish import MultiSourceFisherProposal

import warnings

def flat_psd_function(f, val, *args, xp=None, **kwargs):  
    if xp is None:
        xp = np
    val = xp.atleast_1d(xp.asarray(val))
    out = xp.repeat(val[:, None], len(f), axis=1)
    return out

np.random.seed(100)
#warnings.filterwarnings("ignore")

use_gpu = gpu_available
xp = xp if use_gpu else np

gb = GBGPU(use_gpu=use_gpu)

ndim_full = 9

num_bin = 12

A_lims = [7e-24, 1e-21]
f0_lims = [3.986e-3, 4e-3]
m_chirp_lims = [0.05, 0.75]
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

nleaves_max = 70
ndim = 8
ntemps = 10
nwalkers = 100
branch_names = ["gb", "noise_params"]

buffer = 2 ** 12
fmin = f0_lims[0] - buffer * df
fmax = f0_lims[1] + buffer * df
start_freq_ind = int(fmin / df)
end_freq_ind = int(fmax / df)
data_length = int(fmax / df) - start_freq_ind + 1

true_base_psd_val = np.mean(get_sensitivity(np.mean(f0_lims), **{"sens_fn": "noisepsd_AE"}))
base_psd_val = true_base_psd_val * 1

fd = np.arange(start_freq_ind, end_freq_ind + 1) * df

psd = flat_psd_function(fd, base_psd_val, xp=np).squeeze()

psd_in = [
    psd.copy(),
    psd.copy(),
]

"""
amp_in = np.exp(np.random.uniform(*np.log(A_lims), size=num_bin))
f0_in = np.random.uniform(*f0_lims, size=num_bin)
fdot_in = np.random.uniform(*fdot_lims, size=num_bin)
fddot_in = 11 / 3 * fdot_in ** 2 / f0_in
phi0_in = np.random.uniform(*phi0_lims, size=num_bin)
iota_in = np.arccos(np.random.uniform(*np.cos(iota_lims), size=num_bin))
psi_in = np.random.uniform(*psi_lims, size=num_bin)
lam_in = np.random.uniform(*lam_lims, size=num_bin)
beta_sky_in = np.arcsin(np.random.uniform(*np.sin(beta_sky_lims), size=num_bin))
"""
out_params = np.load("out_params2.npy")
assert out_params.shape[1] == 9
out_params[:, 3] = 0.0
check_injection = out_params.copy()

from gbgpu.utils.utility import get_N

N = get_N(out_params[:, 0], out_params[:, 1], Tobs, oversample=4).max()

breakpoint()
amp_in, f0_in, fdot_in, fddot_in, phi0_in, iota_in, psi_in, lam_in, beta_sky_in = out_params.T.copy()
# phi0 is flipped !
phi0_in *= -1.

#N = get_N(1e-23, max(f0_lims), Tobs, oversample=2).item()

waveform_kwargs = dict(N=N, dt=dt, T=Tobs)
# fish_kwargs = dict(N=1024, dt=dt)


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

from eryn.utils import TransformContainer

fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}
transform_fn = TransformContainer(
    parameter_transforms=transform_fn_in, fill_dict=fill_dict
)

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
# inject first signal to get a data stream
A_inj, E_inj = gb.inject_signal(*injection_temp[0], **waveform_kwargs,)
A_inj, E_inj = A_inj[start_freq_ind:end_freq_ind + 1], E_inj[start_freq_ind:end_freq_ind + 1]

A_temp_all.append(A_inj)
snrs_individual.append(snr([A_inj, E_inj], f_arr=fd, PSD="noisepsd_AE",))
if num_bin_injected >= 1:
    A_sub, E_sub = A_inj.copy(), E_inj.copy()
    A_find, E_find = np.zeros_like(A_inj), np.zeros_like(E_inj)
else:
    A_sub, E_sub = np.zeros_like(A_inj), np.zeros_like(E_inj)
    A_find, E_find = A_inj.copy(), E_inj.copy()


for i in range(1, num_bin):
    A_temp, E_temp = gb.inject_signal(*injection_temp[i], **waveform_kwargs,)
    A_temp, E_temp = A_temp[start_freq_ind:end_freq_ind + 1], E_temp[start_freq_ind:end_freq_ind + 1]
    A_inj += A_temp
    E_inj += E_temp

    A_temp_all.append(A_temp)

    snrs_individual.append(snr([A_temp, E_temp], f_arr=fd, PSD="noisepsd_AE",))
    if i < num_bin_injected:
        print("sub")
        A_sub += A_temp
        E_sub += E_temp

    else:
        A_find += A_temp
        E_find += E_temp
        print(i, "find", snr([A_temp, E_temp], f_arr=fd, PSD="noisepsd_AE",))

from lisatools.utils.utility import generate_noise_fd
A_noise_orig = generate_noise_fd(fd, df, true_base_psd_val, sens_fn=flat_psd_function).squeeze()
E_noise_orig = generate_noise_fd(fd, df, true_base_psd_val, sens_fn=flat_psd_function).squeeze()

A_noise = generate_noise_fd(fd, df, base_psd_val, sens_fn=flat_psd_function).squeeze()
E_noise = generate_noise_fd(fd, df, base_psd_val, sens_fn=flat_psd_function).squeeze()

A_inj_orig = A_inj.copy()
E_inj_orig = E_inj.copy()

A_inj = A_inj + A_noise_orig  #  + A_noise
E_inj = E_inj + E_noise_orig  #  + E_noise

plt.semilogy(np.abs(A_inj))
for tmp in A_temp_all:
    plt.semilogy(np.abs(tmp))
#plt.semilogy(np.abs(A_inj_orig))
plt.xlim(3900,4700)
plt.savefig("plot101.png")

# breakpoint()

d_d = inner_product([A_inj, E_inj], [A_inj, E_inj], f_arr=fd, PSD="noisepsd_AE",)
d_h_d_h = inner_product(
    [A_inj - A_sub, E_inj - E_sub],
    [A_inj - A_sub, E_inj - E_sub],
    f_arr=fd,
    PSD="noisepsd_AE",
)
d_h_h1 = inner_product(
    [A_inj - A_sub, E_inj - E_sub], [A_find, E_find], f_arr=fd, PSD="noisepsd_AE"
)
h1_h1 = inner_product([A_find, E_find], [A_find, E_find], f_arr=fd, PSD="noisepsd_AE")

ll = d_h_d_h + h1_h1 - 2 * d_h_h1
print(ll)

data_sub = [
    xp.asarray(A_inj - A_sub),
    xp.asarray(E_inj - E_sub),
]
params = np.tile(injection_temp[8], (100, 1)).T
factor = 1e-8
params *= 1 + factor * np.random.randn(*params.shape)
gb.d_d = d_h_d_h

d_h_d_h_c1 = d_h_d_h.copy()


ll = gb.get_ll(
    params,
    data_sub,
    [xp.asarray(psd), xp.asarray(psd)],
    phase_marginalize=False,
    start_freq_ind=start_freq_ind,
    **waveform_kwargs,
) - np.sum([np.log(psd), np.log(psd)])
# state = State({"gb": coords_out, "noise_params": coords_start_noise}, inds=dict(gb=inds_out, noise_params=inds_noise))


rho_star = 5.0
snr_prior = SNRPrior(rho_star)
"""check = snr_prior.rvs(size=(1000000))
rho = np.linspace(0.0, 1000, 100000)
pdf = snr_prior.pdf(rho)
plt.close()
plt.hist(check, bins=np.arange(1000), density=True)
plt.plot(rho, pdf)
plt.savefig("plot1.png")
breakpoint()
"""
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

snr_lim = inital_snr_lim = 57.5  # 85.0
dSNR = 5.0
generate_dists[0] = uniform_dist(snr_lim, snr_lim + dSNR)
generate_snr_ladder = PriorContainer(generate_dists)

priors_noise = {
    0: uniform_dist(0.1 * base_psd_val, 10.0 * base_psd_val)
}

priors = {"gb": PriorContainer(default_priors_gb), "noise_params": PriorContainer(priors_noise)}

# generate initial search information
num_total = int(1e7)
num_per = int(1e5)
num_rounds = num_total // num_per

data = [
    xp.asarray(A_inj),
    xp.asarray(E_inj),
]

import time

st = time.perf_counter()

gb.d_d = d_h_d_h  #  + np.sum([np.log(psd), np.log(psd)])  # d_d
d_h_d_h_c2 = d_h_d_h.copy()

out_ll = []
out_snr = []
out_params = []
for i in range(num_rounds):
    params = generate_snr_ladder.rvs(size=num_per)
    params_in = transform_fn.both_transforms(params, return_transpose=True)

    phase_maximized_ll = gb.get_ll(
        params_in,
        data_sub,
        [xp.asarray(psd), xp.asarray(psd)],
        phase_marginalize=True,
        start_freq_ind=start_freq_ind,
        **waveform_kwargs,
    ) - np.sum([np.log(psd), np.log(psd)])
    
    phase_maximized_snr = (xp.abs(gb.d_h) / xp.sqrt(gb.h_h.real)).real.copy()
    phase_change = np.angle(gb.non_marg_d_h)

    try:
        phase_maximized_snr = phase_maximized_snr.get()
        phase_change = phase_change.get()

    except AttributeError:
        pass

    params[:, 3] -= phase_change
    params_in[4] -= phase_change

    params[:, 3] %= (2 * np.pi)
    params_in[4] %= (2 * np.pi)

    phase_maximized_ll_check = gb.get_ll(
        params_in,
        data_sub,
        [xp.asarray(psd), xp.asarray(psd)],
        phase_marginalize=False,
        start_freq_ind=start_freq_ind,
        **waveform_kwargs,
    )

    inds_keep = np.where((phase_maximized_snr > snr_lim) & (xp.sqrt(gb.h_h.real).get() > snr_lim) & (phase_maximized_ll == phase_maximized_ll.max()))
    out_ll.append(phase_maximized_ll[inds_keep])  # temp_ll[inds_keep]
    out_snr.append(phase_maximized_snr[inds_keep])
    out_params.append(params[inds_keep])

    if np.any(phase_maximized_ll[inds_keep] < 0.0):
        breakpoint()

    if (i + 1) % 100:
        print(i + 1, num_rounds)

et = time.perf_counter()

print(f"time: {et - st}")
out_ll = np.concatenate(out_ll)
out_snr = np.concatenate(out_snr)
out_params = np.concatenate(out_params, axis=0)
"""
inds = out_ll < -1/2 * d_h_d_h
plt.scatter(out_params[inds, 1], out_params[inds, 0], c="C0", s=15, label="Binary not found")
plt.scatter(out_params[~inds, 1], out_params[~inds, 0], c="C1", s=40, label="Potential binary")
plt.axhline(injection_params[0, -1], c="C2", label="True missing binary parameters")
plt.axvline(injection_params[1, -1], c="C2")
plt.xlabel("Frequency (Hz)", fontsize=16)
plt.ylabel(r"$\log{A}$", fontsize=16)

#plt.axhline(injection_params[0, -2], c="C3", lw=3)
#plt.axvline(injection_params[1, -2], c="C3", lw=3)
plt.legend(loc="upper left")
plt.show()
breakpoint()
"""

inds_sort_snr = np.argsort(out_ll)[::-1]
out_ll = out_ll[inds_sort_snr]
out_snr = out_snr[inds_sort_snr]
out_params = out_params[inds_sort_snr]

#np.save(
#    "check_pars",
#    np.concatenate([out_params, np.array([out_snr.real, out_ll]).T], axis=1),
#)

best_start_point = out_params[0]
best_start_ll = out_ll[0]
best_start_snr = out_snr[0]


factor = 0.0  # 1e-10
coords_out = np.zeros((ntemps, nwalkers, nleaves_max, ndim))
inds_out = np.zeros((ntemps, nwalkers, nleaves_max)).astype(bool)

inds_out[:, :, 0] = True
beginning_snr = []

factor = 1e-7
cov = np.ones(8) * 1e-3
cov[1] = 1e-7

start_like = np.zeros((nwalkers * ntemps))

while np.std(start_like) < 5.0:
    tmp = best_start_point[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, 8))

    tmp_in = transform_fn.both_transforms(tmp, return_transpose=True)
    
    start_like = gb.get_ll(
        tmp_in,
        data_sub,
        [xp.asarray(psd), xp.asarray(psd)],
        phase_marginalize=False,
        start_freq_ind=start_freq_ind,
        **waveform_kwargs,
    ) - np.sum([np.log(psd), np.log(psd)])
    print(np.std(start_like))
    factor *= 2
    
coords_out[:, :, 0] = tmp.reshape(ntemps, nwalkers, 8)

##### Add in waveforms and data streams
class ProduceWaveforms:
    def __init__(self, gb, data, psd, start_freq_ind, use_gpu=False, **waveform_kwargs):
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.gb = gb

        self.data_clean = self.xp.asarray(data).copy()

        self.start_freq_ind = start_freq_ind
        self.fd = df * np.arange(start_freq_ind, start_freq_ind + len(data[0]))
        psd_here = psd
        self.data = data
        self.psd = psd
        self.start_freq_ind = start_freq_ind
        self.waveform_kwargs = waveform_kwargs
        self.use_gpu = use_gpu
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = [self.xp.asarray(data)[i].copy() for i in range(2)]
        return

    @property
    def psd(self):
        return self._psd

    @psd.setter
    def psd(self, psd):
        self._psd = [self.xp.asarray(psd)[i].copy() for i in range(2)]
        return

    def __call__(self, q, inds=None, logp=None, supps=None, branch_supps=None, inds_keep=None):
        if supps is None and branch_supps is None:
            raise ValueError("supps and branch_supps cannot both be None.")

        # TODO: maybe guard against logp
        ntemps, nwalkers, nleaves_max, ndim = q[list(q.keys())[0]].shape

        if inds_keep is None:
            # TODO: adjust this
            raise ValueError("inds_keep cannot be None.")

        params = transform_fn.both_transforms(q["gb"][inds_keep["gb"]])

        self.gb.get_ll(params.T, self.data, self.psd, start_freq_ind=self.start_freq_ind, **self.waveform_kwargs)
        A_temps_start, E_temps_start, start_inds = gb.A.copy(), gb.E.copy(), gb.start_inds.copy()
        N_here = A_temps_start.shape[1]

        # TODO: add factor for d_h term?
        # TODO: IMPROVE THIS !!
        # set waveforms to ones if not satisfying snr threshold
        inds_fix = ((self.gb.h_h.real)**(1/2) < snr_lim) | ((self.gb.d_h.real / self.gb.h_h.real ** (1/2)) < (0.95 * snr_lim))
        if np.any(inds_fix):
            breakpoint()
        A_temps_start[inds_fix] = 1.0
        E_temps_start[inds_fix] = 1.0

        N_here = A_temps_start.shape[1]
        branch_supps["gb"][inds_keep["gb"]] = {"A": A_temps_start, "E": E_temps_start, "start_inds": start_inds}

        """
        for i in range(len(A_temps_start)):
            Ac, Ec, start_indc = A_temps_start[i], E_temps_start[i], start_inds[i].item()
            start_indc -= self.start_freq_ind
            A_inj_in = self.data[0][start_indc:start_indc + len(Ac)]
            E_inj_in = self.data[1][start_indc:start_indc + len(Ac)]
            psd_here = self.psd[0][start_indc:start_indc + len(Ac)]
            freqs_here = fd[start_indc:start_indc + len(Ac)]

            check_h = snr([Ac, Ec], f_arr=xp.asarray(freqs_here), PSD=psd_here, use_gpu=True)
            check_d = inner_product([Ac, Ec], [A_inj_in, E_inj_in], f_arr=xp.asarray(freqs_here), PSD=psd_here, use_gpu=True)/check_h
            if check_h < 1e10 and (check_h < snr_lim or check_d < 0.95 * snr_lim):
                breakpoint()
        """

    def generate_global_template(self, params, groups, templates_all, *args, branch_supps=None, **kwargs):
        assert branch_supps is not None

        if "inds_keep" in branch_supps:
            inds_keep = branch_supps["inds_keep"]
            if np.any(inds_keep):
                self.gb.get_ll(params[inds_keep].T, self.data, self.psd, start_freq_ind=self.start_freq_ind, **self.waveform_kwargs)
                A_temps_start, E_temps_start, start_inds = gb.A.copy(), gb.E.copy(), gb.start_inds.copy()
                N_here = A_temps_start.shape[1]

                # TODO: add factor for d_h term?
                # set waveforms to ones if not satisfying snr threshold
                inds_fix = ((self.gb.h_h.real)**(1/2) < snr_lim) | ((self.gb.d_h.real / self.gb.h_h.real ** (1/2)) < (0.95 * snr_lim))
                A_temps_start[inds_fix] = 1.0
                E_temps_start[inds_fix] = 1.0

                # TODO: check if it has made it out the otherside
                branch_supps["A"][inds_keep] = A_temps_start
                branch_supps["E"][inds_keep] = E_temps_start
                branch_supps["start_inds"][inds_keep] = start_inds
                
                """
                for i in range(len(A_temps_start)):
                    Ac, Ec, start_indc = A_temps_start[i], E_temps_start[i], start_inds[i].item()
                    start_indc -= self.start_freq_ind
                    A_inj_in = self.data[0][start_indc:start_indc + len(Ac)]
                    E_inj_in = self.data[1][start_indc:start_indc + len(Ac)]
                    psd_here = self.psd[0][start_indc:start_indc + len(Ac)]
                    freqs_here = fd[start_indc:start_indc + len(Ac)]

                    check_h = snr([Ac, Ec], f_arr=xp.asarray(freqs_here), PSD=psd_here, use_gpu=True)
                    check_d = inner_product([Ac, Ec], [A_inj_in, E_inj_in], f_arr=xp.asarray(freqs_here), PSD=psd_here, use_gpu=True)/check_h
                    if check_h < 1e10 and (check_h < snr_lim or check_d < 0.95 * snr_lim):
                        breakpoint()
                """
                
                """
                for i in range(len(branch_supps["A"])):
                    Ac, Ec, start_indc = branch_supps["A"][i], branch_supps["E"][i], branch_supps["start_inds"][i].item()

                    if np.all(Ac == 0.0):
                        continue

                    start = start_indc - self.start_freq_ind
                    end = start_indc - self.start_freq_ind + len(Ac)

                    check_h = snr([Ac, Ec], f_arr=self.xp.asarray(self.fd[start:end]), PSD="noisepsd_AE", use_gpu=True)
                    check_d = inner_product([Ac, Ec], list(self.data_clean[:, start:end]), f_arr=self.xp.asarray(self.fd[start:end]), PSD="noisepsd_AE", use_gpu=True) / check_h
                    if check_h < 1e10:
                        if inds_keep.flatten()[i]:
                            print(check_h, check_d, self.gb.h_h.real[i]**(1/2), self.gb.d_h.real[i]/self.gb.h_h.real[i]**(1/2))
                        else:
                            print(check_h, check_d)

                    if (check_h < snr_lim or check_d < 0.95 * snr_lim) and (check_h < 1e10):
                        breakpoint()

                
                for i in np.arange(len(inds_keep))[~inds_keep]:
                    Ac, Ec, start_indc = branch_supps["A"][i], branch_supps["E"][i], branch_supps["start_inds"][i].item()
                    freqs_here = fd[start_indc:start_indc + len(Ac)]

                    check_h = inner_product([Ac, Ec], [Ac, Ec], f_arr=xp.asarray(freqs_here), PSD="noisepsd_AE", use_gpu=True)
                    check_d = inner_product([Ac, Ec], [xp.asarray(A_inj[start_indc:start_indc + len(Ac)]), xp.asarray(E_inj[start_indc:start_indc + len(Ac)])], f_arr=xp.asarray(freqs_here), PSD="noisepsd_AE", use_gpu=True)
                    observed_snr = check_d / xp.sqrt(check_h.real)
                    if observed_snr < snr_lim or xp.sqrt(check_h.real) < snr_lim:
                        print(i, check_h, check_d)
                """
            
        N_here = branch_supps["A"].shape[1]
        self.gb.fill_global_template(groups.astype(self.xp.int32), templates_all, branch_supps["A"], branch_supps["E"], branch_supps["start_inds"], [N_here], start_freq_ind=start_freq_ind)
        """
        for group_ij in np.unique(groups):
            inds1 = np.where(groups == group_ij)[0]
            for i in inds1:
                start = branch_supps["start_inds"][i] - self.start_freq_ind
                end = branch_supps["start_inds"][i] + N_here  - self.start_freq_ind

                Ac, Ec, start_indc = branch_supps["A"][i], branch_supps["E"][i], branch_supps["start_inds"][i].item()
                freqs_here = fd[start_indc:start_indc + len(Ac)]

                if group_ij == 0: 
                    breakpoint()

                check_h = inner_product([Ac, Ec], [Ac, Ec], f_arr=xp.asarray(freqs_here), PSD="noisepsd_AE", use_gpu=True)
                check_d = inner_product([Ac, Ec], [xp.asarray(A_inj[start_indc:start_indc + len(Ac)]), xp.asarray(E_inj[start_indc:start_indc + len(Ac)])], f_arr=xp.asarray(freqs_here), PSD="noisepsd_AE", use_gpu=True)
                observed_snr = check_d / xp.sqrt(check_h.real)
                if observed_snr < snr_lim or xp.sqrt(check_h.real) < snr_lim:
                    #print(i, check_h, check_d)
                    templates_all[group_ij, :, start:end] = 1.0

                try:
                    templates_all[group_ij, 0, start:end] += branch_supps["A"][i]
                except ValueError:
                    breakpoint()
                templates_all[group_ij, 1, start:end] += branch_supps["E"][i]
        """
        return

start_waveform_kwargs = deepcopy(waveform_kwargs)

build_waves = ProduceWaveforms(gb, data_sub, psd_in, start_freq_ind, use_gpu=use_gpu, **start_waveform_kwargs)

min_k = 1

gb_args = (
    gb,
    priors,
    int(5e3),
    start_freq_ind,
    data_length,
    data_sub,
    psd_in,
    xp.asarray(fd),
    [nleaves_max, 1],
    [min_k, 1],
)

class PointGeneratorSNR:
    def __init__(self, generate_snr_ladder):
        self.generate_snr_ladder = generate_snr_ladder

    def __call__(self, size=(1,)):
        new_points = self.generate_snr_ladder.rvs(size=size)
        logpdf = self.generate_snr_ladder.logpdf(new_points)
        return (new_points, logpdf)

point_generator = PointGeneratorSNR(generate_snr_ladder)

gb_kwargs = dict(
    waveform_kwargs=waveform_kwargs,
    parameter_transforms=transform_fn,
    search=False,
    search_samples=None,  # out_params,
    search_snrs=None,  # out_snr,
    search_snr_lim=None,  # snr_lim,  # 50.0,
    global_template_builder=build_waves,
    psd_func=flat_psd_function,
    noise_kwargs=dict(xp=xp),
    provide_betas=True,
    point_generator_func=point_generator,
)
bf = GBMutlipleTryRJ(
    *gb_args,
    **gb_kwargs,
)

# sort snr_values

n_noise_params = 1
noise_start_factor = 0.0 * 1e-1
coords_start_noise = np.full((ntemps, nwalkers, 1, n_noise_params), base_psd_val) * (1 + noise_start_factor * np.random.randn(ntemps, nwalkers, 1, n_noise_params))
inds_noise = np.full((ntemps, nwalkers, 1), True)
state = State({"gb": coords_out, "noise_params": coords_start_noise}, inds=dict(gb=inds_out, noise_params=inds_noise))

#state = State({"gb": coords_out}, inds=dict(gb=inds_out))

fp = "test_new_search_for_expanded_bands.h5"
folder = "./"
import os
fp_old = fp  # "test_global_fit_on_ldc_2.h5"  # "for_fix_test_global_fit_on_ldc_1.h5"
if fp_old in os.listdir(folder):
    #raise NotImplementedError("need to add noise params to here")
    print("reload", fp)
    backend = HDFBackend(folder + fp_old)
    #state = backend.get_a_sample(22550)
    state = backend.get_last_sample()
    fix = state.branches_coords["gb"][0, 0, 0, 0] < 0.0
    if fix:
        amps = np.exp(state.branches_coords["gb"][:, :, :, 0])
        f0 = state.branches_coords["gb"][:, :, :, 1] / 1e3
        state.branches_coords["gb"][:, :, :, 0] = amp_transform.forward(amps, f0)[0]

    for name, coords in state.branches_coords.items():
        coords[np.isnan(coords)] = 0.0

    """
    upsample = 1  # int(nwalkers / 20)
    state.branches["gb"].coords[np.isnan(state.branches_coords["gb"])] = 0.0
    new_coords = np.tile(state.branches_coords["gb"], (1, upsample, 1, 1))
    new_inds = np.tile(state.branches_inds["gb"], (1, upsample, 1))
    new_log_prob = np.tile(state.log_prob, (1, upsample))
    new_log_prior = np.tile(state.log_prior, (1, upsample))
    state = State({"gb": new_coords}, inds={"gb": new_inds}, log_prob=new_log_prob, log_prior=new_log_prior)
    """

# state, accepted = bf.propose(model, state)

from eryn.moves import Move
class PlaceHolder(Move):
    def __init__(self, *args, **kwargs):
        super(PlaceHolder, self).__init__(*args, **kwargs)

    def propose(self, model, state):
        accepted = np.zeros(state.log_prob.shape)
        return state, accepted

rj_moves = bf  # [(bf, 0.1), (PlaceHolder(), 0.9)]

from lisatools.sampling.moves.gbfreqjump import GBFreqJump
from lisatools.sampling.moves.gbgroupstretch import GBGroupStretchMove

factor = 1e-3

df_freq_jump = 1 / YEAR
# TODO: adjust these settings over time during search
nfriends = 40
moves = [
    #(StretchMoveRJ(live_dangerously=True, a=2.0, gibbs_sampling_leaves_per=1, adjust_supps_pre_logl_func=build_waves), 1.0),
    (GBGroupStretchMove(
        gb_args,
        gb_kwargs,
        nfriends=nfriends,
        live_dangerously=True, 
        a=2.0, 
        gibbs_sampling_leaves_per=1, 
        n_iter_update=30,
        adjust_supps_pre_logl_func=build_waves,
        skip_supp_names=["group_move_points"]
        ), 1.0),  # 0.1666666667),
    (
        GBFreqJump(
            df_freq_jump,
            factor,
            gb_args,
            gb_kwargs,
            spread=4
        ),
        0.0,  # 0.1
    ),
    (PlaceHolder(), 0.0)  # 5 * 0.1666666667)
]
#moves = moves[:1]
"""
import warnings
class HackyTemperatureInfo:
    def __init__(self, ntemps, nwalkers, nsteps):
        self.ntemps, self.nwalkers, self.nsteps = ntemps, nwalkers, nsteps
        self.rj_acceptance_fraction_over_time = np.zeros((self.nsteps, self.ntemps, self.nwalkers))
        self.acceptance_fraction_over_time = np.zeros((self.nsteps, self.ntemps, self.nwalkers))
        self.iteration = 0

    def __call__(self, i, last_result, sampler):
        if self.iteration < self.acceptance_fraction_over_time.shape[0]:
            self.acceptance_fraction_over_time[self.iteration] = sampler.acceptance_fraction.copy()
            self.rj_acceptance_fraction_over_time[self.iteration] = sampler.rj_acceptance_fraction.copy()
            self.iteration += 1
        else:
            warnings.warn("Not adding any more acceptance fraction information because max iterations has been met.")

update_fn = HackyTemperatureInfo(ntemps, nwalkers, 20)
"""

like = GlobalLikelihood(
    build_waves,
    2,
    f_arr=fd,
    parameter_transforms=transform_fn,
    fill_templates=True,
    vectorized=True,
    use_gpu=use_gpu,
    adjust_psd=True
)

like.inject_signal(
    data_stream=[A_inj.copy(), E_inj].copy(),
    noise_fn=flat_psd_function,
    noise_args=[base_psd_val],
    noise_kwargs={"xp": xp},
    add_noise=False,
)

d_d = 4.0 * df * xp.sum(xp.asarray([(temp.conj() * temp) / xp.asarray(psd) for temp in like.injection_channels]))

#### MUST DO THIS #####???
params_test = injection_params.T.copy()[:, np.array([0, 1, 2, 4, 5, 6, 7, 8])]

#check = like(
#    params_test,
#    np.zeros(len(params_test), dtype=np.int32),
#    kwargs_list=[waveform_kwargs],
#    data_length=data_length,
#    start_freq_ind=start_freq_ind,
#)

periodic = {"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}, "noise_params": {}}

backend = HDFBackend(fp)

from lisatools.sampling.stopping import SearchConvergeStopping
#stop1 = SearchConvergeStopping(n_iters=10, verbose=True)


class SearchRJProposalSuccess: 
    def __init__(self, n_iter=4, verbose=False):
        self.n_iter = n_iter
        self.verbose = verbose
        self.old_rj_accepted = None
        self.iters_consecutive = 0

    def __call__(self, iter, sample, sampler):
        if self.iters_consecutive >= self.n_iter:
            self.iters_consecutive = 0

        if self.old_rj_accepted is None:
            self.old_rj_accepted = sampler.backend.rj_accepted.copy()
            return False
        
        diff = sampler.backend.rj_accepted[0] - self.old_rj_accepted[0]
        if np.all(diff < 1e-10):
            self.iters_consecutive += 1
        else:
            self.iters_consecutive = 0
            breakpoint()

        self.old_rj_accepted = sampler.backend.rj_accepted.copy()
        if self.iters_consecutive >= self.n_iter:
            return True
        else:
            return False

stop1 = SearchRJProposalSuccess(verbose=True)

def update_with_snr(i, last_sample, sampler):
    base_psd_val = globals()["base_psd_val"]
    true_base_psd_val = globals()["true_base_psd_val"]
    A_inj = globals()["A_inj"]
    E_inj = globals()["E_inj"]
    fd = globals()["fd"]
    psd = globals()["psd"]
    A_noise = globals()["A_noise"]
    E_noise = globals()["E_noise"]
    generate_snr_ladder = globals()["generate_snr_ladder"]

    snr_lim_tmp = globals()["snr_lim"]
    snr_lim_old = snr_lim_tmp
    
    stop = stop1(i, last_sample, sampler)
    
    if stop:
        if snr_lim_tmp > 20.0:
            snr_lim_tmp -= 2.5
        elif snr_lim_tmp > 10.0:
            snr_lim_tmp -= 0.5
        elif snr_lim > 0.0:
            breakpoint()
            snr_lim_tmp -= 0.5
        else:
            print("converged")
            breakpoint()

        bf.search_snr_lim = snr_lim_tmp
        del bf.point_generator_func
        generate_dists[0] = uniform_dist(snr_lim_tmp, snr_lim_tmp + dSNR)
        generate_snr_ladder = PriorContainer(generate_dists)
        bf.point_generator_func = PointGeneratorSNR(generate_snr_ladder)
        build_waves_temp = globals()["build_waves"]

        # TODO: use this?
        #last_sample.branches_coords["gb"][:] = last_sample.branches_coords["gb"][0].copy()
        #last_sample.branches_inds["gb"][:] = last_sample.branches_inds["gb"][0].copy()
        #last_sample.log_prob[:]  = last_sample.log_prob[0].copy()
        #last_sample.log_prior[:]  = last_sample.log_prior[0].copy()
        #last_sample.branches_supplimental["gb"][:] = last_sample.branches_supplimental["gb"][0]

    globals()["snr_lim"] = snr_lim_tmp

    nleaves = last_sample.branches["gb"].nleaves
    keep = nleaves[0].argmax()
    for j in range(nwalkers):
        tmp =  last_sample.branches_supplimental["gb"][0,j]
        for i in range(len(tmp["A"])):
            Ac, Ec, start_indc = tmp["A"][i], tmp["E"][i], tmp["start_inds"][i].item()

            if np.all(Ac == 0.0):
                continue

            start_indc -= start_freq_ind
            freqs_here = fd[start_indc:start_indc + len(Ac)]
            psd_here = xp.asarray(psd)[start_indc:start_indc + len(Ac)]
            check_h = snr([Ac, Ec], f_arr=xp.asarray(freqs_here), PSD=psd_here, use_gpu=True)
            check_d = inner_product([Ac, Ec], [xp.asarray(A_inj[start_indc:start_indc + len(Ac)]), xp.asarray(E_inj[start_indc:start_indc + len(Ac)])], f_arr=xp.asarray(freqs_here), PSD=psd_here, use_gpu=True)/check_h
            print(j, i, check_h, check_d)
            if check_h < snr_lim_old or check_d < 0.95 * snr_lim_old:
                breakpoint()
    stop = False
    print(globals()["fp"], "stop", stop, base_psd_val, true_base_psd_val, globals()["snr_lim"], bf.search_snr_lim, stop1.iters_consecutive)
    print("update max leaves", nleaves[0].max(), last_sample.log_prob[0].min(), last_sample.log_prob[0].mean(), last_sample.log_prob[0].max())

    
from eryn.moves.tempering import make_ladder

betas = make_ladder(10 * 8 + 1, ntemps=ntemps)
# betas = np.array([1.0])
sampler = EnsembleSampler(
    nwalkers,
    [ndim, 1],  # assumes ndim_max
    like,
    priors,
    provide_groups=True,  # TODO: improve this
    provide_supplimental=True,
    tempering_kwargs={"betas": betas},
    nbranches=len(branch_names),
    nleaves_max=[nleaves_max, 1],
    moves=moves,
    rj_moves=rj_moves,
    kwargs=dict(
        kwargs_list=[waveform_kwargs, None],
        data_length=data_length,
        start_freq_ind=start_freq_ind,
    ),
    backend=backend,
    vectorize=True,
    plot_iterations=-1,
    periodic=periodic,  # TODO: add periodic to proposals
    branch_names=branch_names,
    verbose=False,
    update_fn=update_with_snr,
    update_iterations=5,
    #stopping_fn=stop1,
    #stopping_iterations=10
)

lp = sampler.compute_log_prior(state.branches_coords, inds=state.branches_inds)
state.log_prior = lp

model = sampler.get_model()
num_search_iters = 10

search_snr_lim = 68.0
search_snr_accept_factor = 0.9
# adjust max_k
"""
bf_here = BruteRejection(
    gb,
    priors,
    int(1e5),
    start_freq_ind,
    data_length,
    data_in,
    psd_in,
    [nleaves_max],
    [min_k],
    waveform_kwargs=waveform_kwargs,
    parameter_transforms=transform_fn,
    search=True,
    search_samples=out_params,
    search_snrs=out_snr,
    search_snr_lim=10.0,  # search_snr_lim,
    search_snr_accept_factor=search_snr_accept_factor,
    take_max_ll=True,
    temperature_control=sampler.temperature_control,
    global_template_builder=build_waves
)
stretch_here = moves[0]
num_stretches = 10
num_rjs = 10
search_snr_lim_min = 10.0

accepted_out = np.zeros((ntemps, nwalkers))
accepted_out_st = np.zeros((ntemps, nwalkers))
initial_state = State(state, copy=True)
"""

branch_supps_in = {
    "A": xp.zeros((ntemps, nwalkers, nleaves_max, N), dtype=np.complex128), 
    "E": xp.zeros((ntemps, nwalkers, nleaves_max, N), dtype=np.complex128), 
    "start_inds": xp.zeros((ntemps, nwalkers, nleaves_max), dtype=np.int32),
    "group_move_points": np.zeros((ntemps, nwalkers, nleaves_max, nfriends, ndim))
}
from eryn.state import BranchSupplimental
obj_contained_shape = (ntemps, nwalkers, nleaves_max)

branch_supps = BranchSupplimental(branch_supps_in, obj_contained_shape=obj_contained_shape, copy=True)
#inds_checking = np.arange(np.prod(obj_contained_shape)).reshape(obj_contained_shape)[initial_state.branches_inds["gb"]]

branch_supps_in_noise = {
    "group_move_points": np.zeros((ntemps, nwalkers, 1, nfriends, n_noise_params))
}
obj_contained_shape_noise = (ntemps, nwalkers, 1)

branch_supps_noise = BranchSupplimental(branch_supps_in_noise, obj_contained_shape=obj_contained_shape_noise, copy=True)


# setup data streams to add to and subtract from
supps_shape_in = xp.asarray(data_sub).shape

supps_base_shape = (ntemps, nwalkers)
supps = BranchSupplimental({"data_streams": xp.zeros(supps_base_shape + supps_shape_in, dtype=complex)}, obj_contained_shape=supps_base_shape, copy=True)

state.branches["gb"].branch_supplimental = branch_supps
state.branches["noise_params"].branch_supplimental = branch_supps_noise
state.supplimental = supps

# add inds_keep
state.branches_supplimental["gb"].add_objects({"inds_keep": state.branches_inds["gb"]})

build_waves(state.branches_coords, inds=state.branches_inds, supps=state.supplimental, branch_supps=state.branches_supplimental, inds_keep=state.branches_inds)

# remove inds_keep
state.branches_supplimental["gb"].remove_objects("inds_keep")

import time
num = 10
st = time.perf_counter()
for _ in range(num):
    ll = sampler.compute_log_prob(state.branches_coords, inds=state.branches_inds, supps=state.supplimental, branch_supps=state.branches_supplimental, logp=lp)[0]
et = time.perf_counter()
print("timing:", (et - st)/num)

# group everything
from eryn.utils.utility import groups_from_inds
groups = groups_from_inds({"gb": state.branches_inds["gb"]})["gb"]
# TODO: adjust to cupy

templates = xp.zeros(
    (nwalkers * ntemps, 2, data_length), dtype=xp.complex128
)  
build_waves.generate_global_template(None, groups, templates, branch_supps=state.branches_supplimental["gb"][state.branches_inds["gb"]])


in_vals = (templates - xp.asarray(data_sub)[None, :, :])
d_h_d_h = 4 * xp.sum((in_vals.conj() * in_vals)/ xp.asarray(psd_in)[None, :, :], axis=(1, 2)) * df

state.log_prob = ll


"""
for i in range(100):
    state, accepted = bf_here.propose(model, state)
    if i % 10 == 0:
        breakpoint()
    print(i)
breakpoint()

ks = np.arange(2, nleaves_max + 1)
snr_lims = np.linspace(search_snr_lim_min, search_snr_lim, len(ks))[::-1]
for ii, (k, current_search_snr_lim) in enumerate(zip(ks, snr_lims)):
    bf_here.max_k = [k]
    bf_here.update_with_new_snr_lim(current_search_snr_lim)
    if len(bf_here.search_inds) > int(1e5):
        bf_here.num_brute = int(5e4)
    else:
        bf_here.num_brute = int(len(bf_here.search_inds) / 2.0)

    if bf_here.num_brute == 0:
        bf_here.num_brute = 1

    print(f"starting {k}: {bf_here.search_snr_lim}, {len(bf_here.search_inds)}")
    search_i = 0
    while search_i < num_search_iters:
        for s in range(num_stretches):
            state, accepted_st = stretch_here.propose(model, state)
            accepted_out_st += accepted_st.astype(int)

        for s in range(num_rjs):
            if search_i == num_search_iters:
                continue

            state, accepted = bf_here.propose(model, state)
            accepted_out += accepted.astype(int)
            if not bf_here.last_run_ok:
                search_i = num_search_iters
            print(search_i, num_search_iters,  s)

        search_i += 1

    #np.save(f"log_prob_{k}", state.log_prob)

breakpoint()
"""

#state, accepted = bf.propose(sampler.get_model(), state)
nsteps = 30000


check_injection[:, 0] = np.log(check_injection[:, 0])
check_injection[:, 1] = check_injection[:, 1] * 1e3
check_injection[:, 5] = np.cos(check_injection[:, 5])
check_injection[:, 6] = check_injection[:, 6] % (np.pi)
check_injection[:, 8] = np.sin(check_injection[:, 8])

inj_in = check_injection[:, np.array([0, 1, 2, 4, 5, 6, 7, 8])]
"""
out = []
for i in range(1000):
    state, accepted = moves[0][0].propose(sampler.get_model(), state)
    state, accepted = moves[1][0].propose(sampler.get_model(), state)

    if i == 0:
        state.branches["gb"].coords[0, ::4, 1:, :] = 0.0
        #state.branches["gb"].coords[0, 0, 0, :] = inj_in[8]
        state.branches["gb"].coords[0, ::4, 1, :] = inj_in[2]

        state.branches["gb"].inds[0, ::4, 1:] = False
        state.branches["gb"].inds[0, ::4, :2] = True

        inds_keep = np.zeros_like(state.branches["gb"].inds)
        inds_keep[0, ::4, :2] = True
        state.branches_supplimental["gb"].add_objects({"inds_keep": inds_keep})

        print(state.log_prob.max(), state.log_prob[0,23])
        ll = sampler.compute_log_prob(state.branches_coords, inds=state.branches_inds, supps=state.supplimental, branch_supps=state.branches_supplimental)[0]
        lp = sampler.compute_log_prior(state.branches_coords, inds=state.branches_inds)

        state.log_prob = ll
        state.log_prior = lp

        state.branches_supplimental["gb"].remove_objects(["inds_keep"])
   

    if i > 100:
        tmp = state.branches["gb"].coords[0][state.branches["gb"].inds[0]]
        out.append(tmp)
    
    max_f = state.branches["gb"].coords[0, :, :, 1].max()
    print(i, accepted.sum(axis=-1),  state.log_prob.max(), max_f, max_f - inj_in[2, 1])

out = np.asarray(out).squeeze().reshape(-1, 8)
plt.scatter(inj_in[2, 1], inj_in[2, 0])
plt.scatter(out[:, 6], out[:, 7]) 
plt.savefig("plot2.png")
breakpoint()
"""
out = sampler.run_mcmc(state, nsteps, burn=0, progress=True, thin_by=1)
print(out.log_prob)
breakpoint()
