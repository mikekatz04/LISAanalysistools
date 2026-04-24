import numpy as np
from scipy import signal
from lisatools.domains import *


dt = 10.0
nperseg = 1024
big_dt = nperseg * dt

N = 999936 # int(1e6 / nperseg) * nperseg
t_arr = np.arange(N) * dt
phi0 = 0.782340988
f0 = 3.9e-3
wave = 1e-22 * np.sin(2 * np.pi * f0 * t_arr + phi0)
y = np.tile(wave, (3, 1))

big_df = 1 / big_dt

df = 1. / (N * dt)
td = TDSignal(y, TDSettings(N, dt))
# fd = FDSignal(np.fft.rfft(y), FDSettings(df))
# stft = STFTSignal(signal.stft(y, fs=(1 / dt), nperseg=nperseg), STFTSettings(big_dt, big_df))
# from scipy.signal.windows import tukey
# new_fd = td.transform(FDSettings(df), window=tukey(y.shape[-1], alpha=0.05))
# new_td = fd.transform(TDSettings(dt))

Tobs = N * dt
wdm_set = WDMSettings(Tobs, dt)

fd_from_td = td.fft()
fd_set = fd_from_td.settings
wdm_from_fd = fd_from_td.transform(wdm_set)
wdm_from_td = td.transform(wdm_set)
fd_from_wdm = wdm_from_td.transform(fd_set)
td_from_td = wdm_from_fd.transform(fd_set)

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