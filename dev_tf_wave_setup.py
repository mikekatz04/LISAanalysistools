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


force_backend = "cpu"

orbits = DefaultOrbits(force_backend=force_backend)
orbits.configure(linear_interp_setup=True)
tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
dt = 2.5
Tobs = 2 * YRSID_SI

wavelet_duration = int(1 / 12 * YRSID_SI / dt) * dt
Nt = int(Tobs / wavelet_duration)
Tobs = Nt * wavelet_duration
N = int(Tobs / dt)
Nf = int(N / Nt)

N_sparse = 256 
t_tdi = np.linspace(0.0, Tobs, N_sparse + 2)[1:-1]

wdm_settings = WDMSettings(Nf, Nt, dt, force_backend="cpu")

wdm_lookup_table = WDMLookupTable(wdm_settings, 20, 20, 3, store_path="./wdm_lookup_table.pkl", sub_setting_full_time_layers=True)
f_arr = np.random.uniform(wdm_settings.f_arr.min(), wdm_settings.f_arr.max(), 10)
fdot_arr = np.random.uniform(wdm_lookup_table.fdot_vals.min(), wdm_lookup_table.fdot_vals.max(), 10)
amp_arr = np.ones_like(f_arr)
phase_arr = np.ones_like(f_arr)
wdm_coeffs, m_layers = wdm_lookup_table.get_wdm_coeffs(amp_arr, phase_arr, f_arr, fdot_arr, num_m_layers=4)
breakpoint()
# gb_comps = GBWDMComputations(wdm_lookup_table, Tobs, orbits=orbits, tdi_config=tdi_config, force_backend=force_backend)

num_bin = 1

data_t_arr = np.arange(N) * dt
keep = (data_t_arr > t_tdi[0]) & (data_t_arr < t_tdi [-1])
tdi_t_arr = data_t_arr[keep]

amp = np.full(num_bin, 1e-22)
f0 = np.full(num_bin, 15.2300812341e-3)
fdot = np.full(num_bin, 1e-15)
fddot = np.full(num_bin, 0.0)
phi0 = np.full(num_bin, 0.892342342342)
inc = np.full(num_bin, 1.2309804223)
psi = np.full(num_bin, 3.00908098)
lam = np.full(num_bin, 4.827342308)
beta = np.full(num_bin, -0.50923423)

gb_gen = GBTDIonTheFly(
    t_tdi, 
    Tobs,
    0.0,
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
tdi_output[:] *= tukey(tdi_output.shape[-1], alpha=0.05)[None, :]

np.save("td_check", tdi_output[0])
# import matplotlib.pyplot as plt
# plt.plot(data_t_arr[:num_points], tdi_output[0,0])
# plt.show()

td = TDSignal(tdi_output[0], settings=TDSettings(tdi_output.shape[-1], dt, force_backend=force_backend))



wdm_set = WDMSettings(Nf, Nt, dt, force_backend="cpu")

# y[:, 0] = 1.0
# y[:, 1:] = 0.0
# td = TDSignal(y, TDSettings(N, dt, force_backend="cpu"))
# fd = FDSignal(np.fft.rfft(y), FDSettings(df))
# stft = STFTSignal(signal.stft(y, fs=(1 / dt), nperseg=nperseg), STFTSettings(big_dt, big_df))
# new_fd = td.transform(FDSettings(df), window=tukey(y.shape[-1], alpha=0.05))
# new_td = fd.transform(TDSettings(dt))

Tobs = N * dt

wdm_set.frequency_layer_mask = ((wdm_set.f_arr >= 5e-5) &(wdm_set.f_arr <= 40e-3))

fd_from_td = td.fft(apply_dt=True)

fd_set = fd_from_td.settings
fd_set.frequency_layer_mask = ((fd_set.f_arr >= 5e-5) &(fd_set.f_arr <= 40e-3))
fd_from_td.frequency_layer_mask = ((fd_from_td.f_arr >= 5e-5) &(fd_from_td.f_arr <= 40e-3))
# wdm_from_fd = fd_from_td.transform(wdm_set)
wdm_from_td = td.transform(wdm_set)
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