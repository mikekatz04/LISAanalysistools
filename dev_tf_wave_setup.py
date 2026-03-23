import numpy as np
from scipy import signal
from lisatools.domains import *
import matplotlib.pyplot as plt
import numpy as np

# Ensure LaTeX is disabled
plt.rcParams['text.usetex'] = False

dt = 5.0

Nf = 1536
Nt = 10
N =  Nf * Nt # int(1e6 / nperseg) * nperseg
t_arr = np.arange(N) * dt
phi0 = 0.782340988
f0 = 3.9e-3
wave = 1e-22 * np.sin(2 * np.pi * f0 * t_arr + phi0)
y = np.tile(wave, (3, 1))


df = 1. / (N * dt)

wdm_set = WDMSettings(Nf, Nt, dt, force_backend="cpu")

y[:, 0] = 1.0
y[:, 1:] = 0.0
td = TDSignal(y, TDSettings(N, dt, force_backend="cpu"))
# fd = FDSignal(np.fft.rfft(y), FDSettings(df))
# stft = STFTSignal(signal.stft(y, fs=(1 / dt), nperseg=nperseg), STFTSettings(big_dt, big_df))
# from scipy.signal.windows import tukey
# new_fd = td.transform(FDSettings(df), window=tukey(y.shape[-1], alpha=0.05))
# new_td = fd.transform(TDSettings(dt))

Tobs = N * dt

# wdm_set.frequency_layer_mask = ((wdm_set.f_arr >= 5e-5) &(wdm_set.f_arr <= 25e-3))

fd_from_td = td.fft(apply_dt=False)

fd_set = fd_from_td.settings
wdm_from_fd = fd_from_td.transform(wdm_set)

olitas_check = np.genfromtxt("olitas_wdm_impulse.dat")

# wdm_from_td = td.transform(wdm_set)


fd_from_wdm = wdm_from_fd.transform(fd_set)
breakpoint()
# td_from_td = wdm_from_fd.transform(fd_set)

from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.detector import sangria
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer

sens_mat_fd = XYZ2SensitivityMatrix(fd_set, model=sangria)
sens_mat_wdm = XYZ2SensitivityMatrix(wdm_set, model=sangria)
data_res = DataResidualArray(wdm_from_td, signal_domain=wdm_set)

analysis = AnalysisContainer(data_res, sens_mat_wdm)
ll = analysis.likelihood()
breakpoint()