import warnings
warnings.simplefilter('ignore', DeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt

from lisatools.domains import TDSettings, WDMSettings, TDSignal, WDMSignal
from lisatools.utils.constants import YRSID_SI

dt   = 10.0                       # seconds
Tobs = 1.0 * YRSID_SI             # half year
N    = int(round(Tobs / dt))
#N    = 1 << (N.bit_length() - 1)  # round to power of two (required by WDM)
t    = np.arange(N) * dt

f0   = 3.0e-3                     # 3 mHz -- mid-LISA band
amp  = 1.0e-21                    # GB-scale amplitude

phi0 = 0.0                        # initial phase (rad)
h_sin = amp * np.sin(2 * np.pi * f0 * t + phi0)

print(f'N = {N:,} samples ({N*dt/YRSID_SI:.3f} yr at dt = {dt} s)')
print(f'f0 = {f0*1e3} mHz   amp = {amp:.1e}')

Nf, Nt, wavelet_duration = WDMSettings.adjust_to_even_bins(12 * 3600.0, 13 * 3600, dt, Tobs)
wdm_set = WDMSettings(Nf=Nf, Nt=Nt, dt=dt)
breakpoint()
# redo time domain to match right length for WDM
td_set = TDSettings(N=wdm_set.N, dt=wdm_set.data_dt)
td_sig_sine = TDSignal(h_sin[None, :wdm_set.N], settings=td_set)
print('TDSettings:', td_set)
print('TDSignal.arr shape:', td_sig_sine.arr.shape, '  (nchannels, N)')
print(f'WDM grid: Nf={wdm_set.Nf}  Nt={wdm_set.Nt}  '
      f'layer_df={wdm_set.layer_df*1e3:.4f} mHz  '
      f'layer_dt={wdm_set.layer_dt/3600:.2f} hr')

wdm_sine = td_sig_sine.transform(wdm_set)
wdm_arr  = wdm_sine.arr[0]    # drop the (single) channel axis -> (Nf, Nt)
print('WDMSignal.arr shape:', wdm_sine.arr.shape, '  (nchannels, Nf, Nt)')

# Which m-layer holds the carrier?
m_floor = int(np.floor(f0 / wdm_set.layer_df))
print(f'carrier f0 = {f0*1e3} mHz  ->  m_floor = {m_floor} '
      f'(layer center {m_floor * wdm_set.layer_df * 1e3:.3f} mHz)')
