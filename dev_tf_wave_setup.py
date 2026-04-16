import numpy as np
from scipy import signal
from lisatools.domains import *
import matplotlib.pyplot as plt
import numpy as np

# Ensure LaTeX is disabled
plt.rcParams['text.usetex'] = False

# dt = 5.0

# Nf = 1536
# Nt = 100
# N =  Nf * Nt # int(1e6 / nperseg) * nperseg
# t_arr = np.arange(N) * dt
# phi0 = 0.782340988
# f0 = 3.9e-3
# wave = 1e-24 * np.sin(2 * np.pi * f0 * t_arr + phi0)
# y = np.tile(wave, (3, 1))


# df = 1. / (N * dt)

from fastlisaresponse.tdionfly import GBTDIonTheFly

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError) as e:
    pass

from fastlisaresponse.tdionfly import GBTDIonTheFly
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import DefaultOrbits
from lisatools.utils.constants import *

from lisatools.domains import WAVELET_DURATION, TDSignal, TDSettings, FDSignal, FDSettings, WDMSignal, WDMSettings, WDMLookupTable
from fastlisaresponse.gbcomps import GBWDMComputations
from scipy.signal.windows import tukey

force_backend = "cpu"
xp = np if force_backend == "cpu" else cp
orbits = DefaultOrbits(force_backend=force_backend)
orbits.configure(linear_interp_setup=True)
tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
dt = 2.5
Tobs = 2 * YRSID_SI
Nf = -1
Nt = -1
for tmp in np.linspace(4., 5, 1000):
    wavelet_duration = int(tmp / 365 * YRSID_SI / dt) * dt
    Nt = int(Tobs / wavelet_duration)
    Tobs = Nt * wavelet_duration
    N = int(Tobs / dt)
    Nf = int(N / Nt)
    print(tmp, Nf, Nt)
    if (Nt % 2 == 0) and (Nf % 2 == 0):
        break

N_sparse = 256 
t_tdi = xp.linspace(0.0, Tobs, N_sparse + 2)[1:-1]

wdm_settings = WDMSettings(Nf, Nt, dt, force_backend=force_backend)
time_layers = wdm_settings.Nt
tukey_alpha = 0.00
td_window = tukey(wdm_settings.Nf * time_layers, alpha=tukey_alpha)
#wdm_lookup_table = WDMLookupTable(wdm_settings, 0.01, 0.1, 3, store_path="./wdm_lookup_table_with_fdot.pkl", num_layers_diff=1, fdot_max_factor=8.0, time_layers=time_layers, batch_size_gen=200)
wdm_lookup_table = WDMLookupTable(wdm_settings, 0.01, 0.1, 3, store_path="./wdm_lookup_table_without_fdot.pkl", num_layers_diff=2, fdot_max_factor=0.0, time_layers=time_layers, batch_size_gen=1, td_window=td_window)
f_arr = np.linspace(wdm_lookup_table.f_vals.min(), wdm_lookup_table.f_vals.max(), 100)
#xp.random.uniform(wdm_settings.f_arr.min(), wdm_settings.f_arr.max(), 10)
# fdot_arr = xp.random.uniform(wdm_lookup_table.fdot_vals.min(), wdm_lookup_table.fdot_vals.max(), 10)

t_wdm = wdm_settings.t_arr

amp0 = 1e-22
f0_check = 3e-3  # wdm_lookup_table.m_ref * wdm_settings.layer_df
fdot0_check = 0.0
phi0 = 0.887249823409

td_set = TDSettings(wdm_settings.N, dt, force_backend=force_backend)
t_check = np.arange(wdm_settings.N) * dt
wave_check = amp0 * np.sin(2 * np.pi * (f0_check * t_check + 1/2 * fdot0_check * t_check ** 2) + phi0)

wave_check_wdm = TDSignal(wave_check[None, :], td_set).wdmtransform(wdm_settings, window=tukey(wdm_settings.N, alpha=tukey_alpha))
phi_t = 2 * np.pi * (f0_check * t_wdm + 1/2 * fdot0_check * t_wdm ** 2) + phi0
freq_t = f0_check + fdot0_check * t_wdm
fdot_t = np.full_like(freq_t, 0.0)  # fdot0_check)
amp_t = amp0 * xp.ones_like(t_wdm)

n_arr = np.arange(wdm_settings.Nt)
wdm_coeffs, m_layers = wdm_lookup_table.get_wdm_coeffs(amp_t, phi_t, freq_t, fdot_t, n_arr, num_m_layers=1)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
_check_fill_wave = np.zeros_like(wave_check_wdm[0])
_check_fill_wave[m_layers.flatten(), np.repeat(n_arr[:, None], m_layers.shape[-1], axis=-1).flatten()] = wdm_coeffs.flatten()
_check_fill_wdm = WDMSignal(_check_fill_wave[None, :], wdm_settings)
min_val = np.min([wave_check_wdm[:].min(), _check_fill_wdm[:].min()])
max_val = np.max([wave_check_wdm[:].max(), _check_fill_wdm[:].max()])

cax1 = fig.add_axes([0.9, 0.55, 0.05, 0.25])
cax2 = fig.add_axes([0.9, 0.2, 0.05, 0.25])

ind_check = int(f0_check / wdm_settings.layer_df)
wave_check_wdm.heatmap(index=0, fig=fig, ax=ax1, vmin=min_val, vmax=max_val, cax=cax1)
_check_fill_wdm.heatmap(index=0, fig=fig, ax=ax2, vmin=min_val, vmax=max_val)
difference = WDMSignal(np.log10(np.abs((wave_check_wdm[:] - _check_fill_wdm[:]))), wdm_settings)  #  / wave_check_wdm[:]
difference.heatmap(index=0, fig=fig, ax=ax3, vmin=difference[:, ind_check, 20:-20].min().item(), vmax=difference[:, ind_check, 20:-20].max().item(), cax=cax2, cmap=cm.Blues)
ax1.set_ylim(f0_check * 0.997, f0_check * 1.003)
ax2.set_ylim(f0_check * 0.997, f0_check * 1.003)
# fig.savefig(f"wdm_check_3_alpha_{tukey_alpha}.png")

plt.show()
breakpoint()
ax1.plot(wave_check_wdm[0, m_layers[0,0]], lw=3)
ax2.plot(wdm_coeffs.squeeze(), "--", lw=2)
plt.show()
breakpoint()
# gb_comps = GBWDMComputations(wdm_lookup_table, Tobs, orbits=orbits, tdi_config=tdi_config, force_backend=force_backend)

num_bin = 1

data_t_arr = np.arange(N) * dt
keep = (data_t_arr > t_tdi[0]) & (data_t_arr < t_tdi [-1])
tdi_t_arr = data_t_arr[keep]

ind = int(3e-3 / wdm_settings.layer_df) + 3
num = 10
for i in range(0, num):
    amp = np.full(num_bin, 1.0)
    f0 = np.full(num_bin, (ind + i / num) * wdm_settings.layer_df)
    fdot = np.full(num_bin, 0.0)
    fddot = np.full(num_bin, 0.0)
    phi0 = np.full(num_bin, 0.0)
    inc = np.full(num_bin, np.pi / 2)
    psi = np.full(num_bin, 0.0)
    lam = np.full(num_bin, 0.0)
    beta = np.full(num_bin, np.pi / 2)

    t_ref = int(Nt / 2) * wdm_settings.layer_dt
    gb_gen = GBTDIonTheFly(
        t_tdi, 
        Tobs,
        t_ref,
        1. / dt,
        num_bin,
        n_params=9,
        tdi_config=tdi_config,
        orbits=orbits,
        tdi_chan="XYZ",
        force_backend=force_backend,
    )

    output = gb_gen(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta, return_spline=True)
    # FD 
    # bin_i = int(f0[0] * Tobs) - int(output.t_arr.shape[-1] / 4)
    # carrier_frequency = bin_i / Tobs

    # heterodyned_phase = output.tdi_phase + output.phase_ref - 2 * np.pi * carrier_frequency * output.t_arr
    # heterodyned_signal = -output.tdi_amp * np.sin(heterodyned_phase)
    # gb_fd = np.fft.fftshift(np.fft.fft(heterodyned_signal, axis=-1))

    # breakpoint()
    tdi_output = np.zeros((num_bin, 3, len(data_t_arr))) 

    # t_tdi = 
    tdi_output[:, :, keep]= output.eval_tdi(tdi_t_arr)
    from scipy.signal.windows import tukey
    t_diff = (np.arange(tdi_output.shape[-1])[:] * dt - t_ref)
    fdot = 0.0  # wdm_settings.layer_df / wdm_settings.layer_dt * 0.01
    tdi_output[:,0, :] = np.sin(2 * np.pi * (f0 * t_diff + 1/2 * fdot * t_diff ** 2 ))
    tdi_output[:,1, :] = np.cos(2 * np.pi * (f0 * t_diff + 1/2 * fdot * t_diff ** 2 ))
    # tdi_output[:] *= tukey(tdi_output.shape[-1], alpha=0.7)[None, :]
    
    # np.save("td_check", tdi_output[0])
    # import matplotlib.pyplot as plt
    # plt.plot(data_t_arr[:num_points], tdi_output[0,0])
    # plt.show()

    td = TDSignal(tdi_output[0, :2], settings=TDSettings(tdi_output.shape[-1], dt, force_backend=force_backend))



    wdm_set = WDMSettings(Nf, Nt, dt, force_backend="cpu")

    # y[:, 0] = 1.0
    # y[:, 1:] = 0.0
    # td = TDSignal(y, TDSettings(N, dt, force_backend="cpu"))
    # fd = FDSignal(np.fft.rfft(y), FDSettings(df))
    # stft = STFTSignal(signal.stft(y, fs=(1 / dt), nperseg=nperseg), STFTSettings(big_dt, big_df))
    # new_fd = td.transform(FDSettings(df), window=tukey(y.shape[-1], alpha=0.05))
    # new_td = fd.transform(TDSettings(dt))

    Tobs = N * dt

    # wdm_set.frequency_layer_mask = ((wdm_set.f_arr >= 5e-5) &(wdm_set.f_arr <= 40e-3))

    # fd_from_td = td.fft(apply_dt=True)

    # fd_set = fd_from_td.settings
    # fd_set.frequency_layer_mask = ((fd_set.f_arr >= 5e-5) &(fd_set.f_arr <= 40e-3))
    # fd_from_td.frequency_layer_mask = ((fd_from_td.f_arr >= 5e-5) &(fd_from_td.f_arr <= 40e-3))
    # wdm_from_fd = fd_from_td.transform(wdm_set)
    wdm_from_td = td.transform(wdm_set)
    # new_td = TDSignal(np.fft.irfft(td.fft()[:] * np.exp(-1j * np.pi / 2.) / dt, axis=-1), td.settings)
    # wdm_from_td_new = new_td.transform(wdm_set)
    wdm_comples_val = wdm_from_td[1, ind, int(Nt / 2):int(Nt / 2) + 5] + 1j * wdm_from_td[0, ind, int(Nt / 2):int(Nt / 2) + 5]
    print(f0[0], ind * wdm_settings.layer_df, f0[0] - ind * wdm_settings.layer_df, i)
    # print(wdm_from_td[0, ind+ i, int(Nt /2)] ** 2 + wdm_from_td_new[0, ind+ i, int(Nt /2)] ** 2)
    print("\n")
    breakpoint()
    # exit()
breakpoint()
# assert np.allclose(wdm_from_fd[:] / np.abs(wdm_from_fd[:]).max(), wdm_from_td[:] / np.abs(wdm_from_td[:]).max())

olitas_check = np.genfromtxt("olitas_wdm_impulse.dat")

# wdm_from_td = td.transform(wdm_set)


# fd_from_wdm = wdm_from_fd.transform(fd_set)

# td_from_td = wdm_from_fd.transform(fd_set)

from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.detector import sangria, scirdv1
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer

sens_mat_fd = XYZ2SensitivityMatrix(fd_set, model=scirdv1)
sens_mat_wdm = XYZ2SensitivityMatrix(wdm_set, model=scirdv1)
data_res_wdm = DataResidualArray(wdm_from_td, signal_domain=wdm_set)
data_res_fd = DataResidualArray(fd_from_td, signal_domain=fd_set)

analysis_wdm = AnalysisContainer(data_res_wdm, sens_mat_wdm)
analysis_fd = AnalysisContainer(data_res_fd, sens_mat_fd)
ip_wdm = analysis_wdm.inner_product()
ip_fd = analysis_fd.inner_product()
breakpoint()