from __future__ import annotations
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List

import math
import numpy as np
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError):
    import numpy as cp

from . import detector as lisa_models
from .utils.utility import AET, get_array_module
from .utils.constants import *

import dataclasses

@dataclasses.dataclass
class DomainSettingsBase:
    pass

class DomainBase:

    def __init__(self, arr):
        self.arr = arr

    @property
    def arr(self) -> np.ndarray:
        return self._arr
    
    @arr.setter
    def arr(self, arr: np.ndarray):
        xp = get_array_module(arr)
        assert len(arr.shape) >= len(self.basis_shape)
        if len(arr.shape) == len(self.basis_shape):
            arr = arr[None, :]

        channel_shape = arr.shape[:-len(self.basis_shape)]
        if len(channel_shape) > 1:
            raise ValueError("Too many dimensions outside of basis_shape.")
        
        self.nchannels = channel_shape[0]
        self._arr = arr

    def __getitem__(self, index):
        return self.arr[index]
    def __setitem__(self, index, value):
        self.arr[index] = value

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        raise NotImplementedError("Transform needs to be implemented for this signal type.")

    @property
    def shape(self) -> tuple:
        return self.arr.shape
    
@dataclasses.dataclass
class TDSettings(DomainSettingsBase):
    N: int
    dt: float

    @staticmethod
    def get_associated_class():
        return TDSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        return dict()
    
    @property
    def args(self) -> tuple:
        return (self.N, self.dt)   
    
    @property
    def t_arr(self) -> np.ndarray:
        return np.arange(self.N) * self.dt
    
    @property
    def basis_shape(self) -> tuple:
        return (self.N,)
    
    def __eq__(self, value):
        return (value.N == self.N) and (value.dt == self.dt)
    
    @property
    def differential_component(self) -> float:
        return self.dt
    
    @property
    def total_terms(self) -> int:
        return self.N


class TDSignal(DomainBase, TDSettings):
    def __init__(self, arr, settings: TDSettings):
        TDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> TDSettings:
        return TDSettings(*self.args, **self.kwargs)

    def fft(self, settings=None, window=None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        df = 1 / (self.N * self.dt)
        
        if settings is not None:
            assert isinstance(settings, FDSettings)
            assert settings.df == df

        fd_arr = np.fft.rfft(self.arr * window)
        fd_settings = FDSettings(fd_arr.shape[-1], df)
        return FDSignal(fd_arr, fd_settings)

    def stft(self, settings=None, window=None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide STFTSettings for stft transform.")
        assert isinstance(settings, STFTSettings)
        big_dt = settings.dt

        assert float(int(big_dt / self.dt)) == big_dt / self.dt
        big_df = settings.df
        nperseg = int(big_dt / self.dt)

        stft_arr = signal.stft(self.arr * window, fs=(1/self.dt), nperseg=nperseg)
        return STFTSignal(stft_arr, settings)
    
    def wdmtransform(self, settings=None, window=None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # go to frequency domain then wavelets
        return self.fft(settings=None, window=window).transform(settings)

# static void wavelet_window_time(struct Wavelets *wdm)
# {
#     *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
    
#     //zero frequency
#     REAL(DX,0) =  wdm->inv_root_dOmega
#     IMAG(DX,0) =  0.0
    
#     for(int i=1 i<= wdm->N/2 i++)
#     {
#         int j = wdm->N-i
#         omega = (double)(i)*wdm->domega
        
#         // postive frequencies
#         REAL(DX,i) = phitilde(wdm,omega)
#         IMAG(DX,i) =  0.0
        
#         // negative frequencies
#         REAL(DX,j) =  phitilde(wdm,-omega)
#         IMAG(DX,j) =  0.0
#     }
        
#     glass_inverse_complex_fft(DX, wdm->N)

#     wdm->window = (double*)malloc(sizeof(double)* (wdm->N))
#     for(int i=0 i < wdm->N/2 i++)
#     {
#         wdm->window[i] = REAL(DX,wdm->N/2+i)
#         wdm->window[wdm->N/2+i] = REAL(DX,i)
#     }
    
#     wdm->norm = sqrt((double)wdm->N * wdm->cadence / wdm->domega)

#     free(DX)
# }

# void wavelet_transform(struct Wavelets *wdm, *data)
# {
#     //array index for tf pixel
#     int k
    
#     //total data size
#     int ND = wdm->NT*wdm->NF
    
#     //windowed data packets
#     *wdata = double_vector(wdm->N)

#     //wavelet wavepacket transform of the signal
#     **wave = double_matrix(wdm->NT,wdm->NF)
    
#     //normalization factor
#     fac = M_SQRT2*sqrt(wdm->cadence)/wdm->norm
    
#     //normalization fudge factor
#     fac *= sqrt(wdm->cadence)/2
        
#     //do the wavelet transform by convolving data w/ window and FFT
#     for(int i=0 i<wdm->NT i++)
#     {
        
#         for(int j=0 j<wdm->N j++)
#         {
#             int n = i*wdm->NF - wdm->N/2 + j
#             if(n < 0)   n += ND  // periodically wrap the data
#             if(n >= ND) n -= ND  // periodically wrap the data
#             wdata[j] = data[n] * wdm->window[j]  // apply the window
#         }
                
#         glass_forward_real_fft(wdata, wdm->N)

#         //unpack Fourier transform
#         wave[i][0] = wdata[0]
#         for(int j=1 j<wdm->NF j++)
#         {
#             int n = j*wdm->oversample
#             if((i+j)%2 ==0)
#                 wave[i][j] = wdata[2*n]
#             else
#                 wave[i][j] = -wdata[2*n+1]
#         }
#     }
    
#     //replace data vector with wavelet transform mapped from pixel to index
#     for(int i=0 i<wdm->NT i++)
#     {
#         for(int j=0 j<wdm->NF j++)
#         {
#             //get index number k for tf pixel {i,j}
#             wavelet_pixel_to_index(wdm,i,j,&k)
            
#             //replace data array
#             data[k] = wave[i][j]*fac
#         }
#     }
    
#     free_double_vector(wdata)
#     free_double_matrix(wave,wdm->NT)
# }

#         breakpoint()

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, TDSettings):
            return self.settings.associated_class(self.arr * window, self.settings)
        
        elif isinstance(new_domain, FDSettings):
            return self.fft(settings=new_domain, window=window)
        
        elif isinstance(new_domain, STFTSettings):
            return self.stft(settings=new_domain, window=window)
        
        elif isinstance(new_domain, WDMSettings):
            return self.wdmtransform(settings=new_domain, window=window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")


@dataclasses.dataclass
class FDSettings(DomainSettingsBase):
    N: int
    df: float
    ind_min : Optional[int] = None 
    ind_max : Optional[int] = None 

    @property
    def differential_component(self) -> float:
        return self.df
    
    @property
    def ind_min_actual(self) -> int:
        if self.ind_min is None:
            return 0
        return self.ind_min
    
    @property
    def ind_max_actual(self) -> int:
        if self.ind_max is None:
            return self.N - 1
        return self.ind_max
    
    @staticmethod
    def get_associated_class():
        return FDSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def kwargs(self) -> dict:
        return dict(ind_min=self.ind_min_actual, ind_max=self.ind_max_actual)
    
    @property
    def args(self) -> tuple:
        return (self.N, self.df)  
    
    @property
    def basis_shape(self) -> tuple:
        return (self.N,)
    
    @property
    def f_arr(self) -> np.ndarray:
        return np.arange(self.N)[self.ind_min_actual:self.ind_max_actual + 1] * self.df
    
    def __eq__(self, value):
        return (value.N == self.N) and (value.df == self.df)

    @property
    def total_terms(self) -> int:
        return self.N
    

# from pywavelet.transforms.phi_computer import phitilde_vec_norm
# from pywavelet.transforms.numpy.forward.from_freq import (
#     transform_wavelet_freq_helper
# )

# from pywavelet.transforms.numpy.inverse.to_freq import (
#     inverse_wavelet_freq_helper_fast as inverse_wavelet_freq_helper,
# )

class FDSignal(FDSettings, DomainBase):
    def __init__(self, arr, settings: FDSettings):
        FDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> FDSettings:
        return FDSettings(*self.args, **self.kwargs)

    def ifft(self, settings=None, window=None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        Tobs = 1 / self.df
        td_arr = np.fft.irfft(self.arr * window)
        N = td_arr.shape[-1]
        dt = Tobs / N
        assert N == int(Tobs / dt)
        if settings is not None:
            assert isinstance(settings, TDSettings)
            assert settings.dt == dt

        td_settings = TDSettings(dt)
        return TDSignal(td_arr, td_settings)
    
    def get_fd_window_for_wdm(self, settings):

        N = self.settings.N

        # solve for window
        N = (settings.NF+1)

        # mini wavelet structure for basis covering just N layers
        T = settings.dt*settings.NT
        domega = 2 * np.pi / T

        window = np.zeros(self.N, dtype=complex)

        # wdm window function
        for i in range(0, int(settings.NT / 2)):  # (i=0; i<=wdm->NT/2; i++)
            omega = i*domega
            window[i] = settings.phitilde(omega)
    
        raise NotImplementedError
    
        # normalize
        # for(i=-wdm->NT/2; i<= wdm->NT/2; i++) norm += window[abs(i)]*window[abs(i)];
        # norm = sqrt(norm/wdm_temp->cadence);
        
        # for(i=0; i<=wdm->NT/2; i++) window[i] /= norm;
        
        # free(wdm_temp);
        
    def wdmtransform(self, settings=None, window=None, return_transpose_time_axis_first: bool = False):
        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # phif = phitilde_vec_norm(settings.NF, settings.NT, 4.0)
        m = np.repeat(np.arange(0, settings.NF)[:, None], settings.NT, axis=-1)
        n = np.tile(np.arange(settings.NT), (settings.NF, 1))
        

        # removed zero frequency and mirrored
        k = settings.get_shift_map(m)

        base_window = settings.window[:-1]  # TODO: compared to Tyson's code he does i=-N/2; i<N/2; i++
        dc_window = settings.dc_layer_window
        # TODO: check if this is right?!?!
        max_freq_window = settings.max_freq_layer_window

        # k[0] += int(settings.NT / 2)
        # k[-1] -= int(settings.NT / 2)
        # it is 2 because the max frequency would be at 1, but it removes that (?)
        assert k.max().item() == self.N - 2
        tmp = self.arr[:, k]
        
        tmp[:, 1:-1] *= base_window[None, None, :]
        # tmp[0] *= dc_window
        # tmp[-1] *= max_freq_window

        after_ifft = np.fft.ifft(tmp, axis=-1)
        
        is_m_plus_n_even = ((m + n) % 2 == 0)
        _new_arr = np.zeros((self.nchannels, settings.NF, settings.NT), dtype=float)
        _new_arr[:, is_m_plus_n_even] = np.sqrt(2) * np.real(after_ifft)[:, is_m_plus_n_even]
        _new_arr[:, (~is_m_plus_n_even)] = (-1) ** ((m * n)[(~is_m_plus_n_even)] + 1) * np.sqrt(2) * np.imag(after_ifft)[:, (~is_m_plus_n_even)]
        
        # TODO: need to fix top and bottom layer
        _new_arr[:, np.array([0, -1])] = 0.0
        if return_transpose_time_axis_first:
            output = _new_arr.transpose(0, 2, 1).copy()
        else:
            output = _new_arr

        return WDMSignal(output, settings=settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, FDSettings):
            return self.settings.associated_class(self.arr * window, self.settings)
        
        elif isinstance(new_domain, TDSettings):
            return self.ifft(settings=new_domain, window=window)
        
        elif isinstance(new_domain, STFTSettings):
            raise NotImplementedError
            return self.stft()

        elif isinstance(new_domain, WDMSettings):
            return self.wdmtransform(settings=new_domain, window=new_domain.window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")



@dataclasses.dataclass
class STFTSettings(DomainSettingsBase):
    dt: float
    df: float 
    NT: int
    NF: int
    
    @staticmethod
    def get_associated_class():
        return STFTSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def t_arr(self) -> np.ndarray:
        return np.arange(self.NT) * self.dt

    @property
    def f_arr(self) -> np.ndarray:
        return np.arange(self.NF) * self.df
    
    @property
    def f_arr_edges(self) -> np.ndarray:
        return np.arange(self.NF + 1) * self.df
    @property
    def t_arr_edges(self) -> np.ndarray:
        return np.arange(self.NT + 1) * self.dt

    def __eq__(self, value):
        return (value.NT == self.NT) and (value.NF == self.NF) and (value.dt == self.dt) and (value.df == self.df)

class STFTSignal(STFTSettings, DomainBase):
    def __init__(self, arr, settings: STFTSettings):
        STFTSettings.__init__(self, settings.dt, settings.df)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> STFTSettings:
        return STFTSettings(self.dt, self.df)
    
    @property
    def differential_component(self) -> float:
        return 1.0
    
    @property
    def total_terms(self) -> int:
        return self.NT * self.NF


WAVELET_BANDWIDTH = 6.51041666666667e-5
WAVELET_DURATION = 7680.0
WAVELET_FILTER_CONSTANT = 6

from scipy import special

class WDMSettings(DomainSettingsBase):

    def __init__(
        self,
        Tobs: float, 
        dt: float,
        oversample: int = 16,
        window: Optional[np.ndarray] = None,
    ):
        self.Tobs = Tobs
        self.NT = int(np.ceil(Tobs/WAVELET_DURATION).astype(int))
        self.NF = int(WAVELET_DURATION/dt)
        self.data_dt = dt

        self.df = WAVELET_BANDWIDTH
        self.dt = WAVELET_DURATION
        self.oversample = oversample

        self.cadence = WAVELET_DURATION/self.NF
        self.Omega = np.pi/self.cadence
        self.dOmega = self.Omega/self.NF
        self.domega = 2 * np.pi / self.Tobs
        self.inv_root_dOmega = 1.0/np.sqrt(self.dOmega)
        self.B = self.Omega / (2 * self.NF)
        self.A = (self.dOmega-self.B)/2.0
        
        self.BW = (self.A+self.B)/np.pi

        self.N = self.oversample * 2 * self.NF
        self.T = self.N * self.cadence
        if window is None:
            self.setup_window()
        else:
            assert len(window) == self.NT + 1
            self.window = window
            T = self.dt * self.NT
            domega = 2 * np.pi / T
            self.omega = (np.arange(self.NT + 1) - int(self.NT / 2)) * domega
            self.norm = np.sqrt(self.N * self.cadence / self.dOmega)
   
    @property
    def basis_shape(self) -> tuple:
        return (self.NF, self.NT)
    
    @property
    def t_arr(self) -> np.ndarray:
        return np.arange(self.NT) * self.dt

    @property
    def f_arr(self) -> np.ndarray:
        return np.arange(self.NF) * self.df
    
    @property
    def f_arr_edges(self) -> np.ndarray:
        return np.arange(self.NF + 1) * self.df
    @property
    def t_arr_edges(self) -> np.ndarray:
        return np.arange(self.NT + 1) * self.dt

    def phitilde(self, omega):
        insDOM = self.inv_root_dOmega
        A = self.A
        B = self.B
        
        z = np.zeros(omega.shape[0])
        beta_inc_calc = (np.abs(omega) >= A) & (np.abs(omega) <= A+B)
        x = (np.abs(omega[beta_inc_calc])-A)/B
        y = special.betainc(WAVELET_FILTER_CONSTANT, WAVELET_FILTER_CONSTANT, x)
        z[beta_inc_calc] = insDOM*np.cos(y*np.pi/2.0)
        z[(np.abs(omega) < A)] = insDOM
        #breakpoint()
        return z
    
    def wavelet(self, N: int, in_fd: Optional[bool] = True) -> np.ndarray:

        # NT * NF is even 
        # assert (self.NT * self.NF) % 2 == 0
        base_window = self.window[:-1]
        omega_N = (np.arange(self.N) - int(self.N / 2)) * self.domega
        wavelet_N = 1 / np.sqrt(2.) * self.phitilde(omega_N)
        
        if in_fd:
            return wavelet_N
        else:
            return np.fft.ifft(wavelet_N) / self.norm
        breakpoint()
        wavelets_rfft = np.zeros((len(m), int((self.NT * self.NF) / 2 + 1)))

        np.put_along_axis(wavelets_rfft, k, base_window * 1 / np.sqrt(2.), axis=-1)
        freq = np.fft.fftshift(np.fft.fftfreq(self.NT * self.NF, self.data_dt))
        wavelets_fft = np.exp(-1j * 2 * np.pi * freq[None, :] * n[:, None] * self.dt) * np.concatenate([wavelets_rfft[:, ::-1][:, :-1], wavelets_rfft[:, :-1]], axis=-1)
        if in_fd:
            return wavelets_fft
        else:
            wavelets_time = np.fft.ifft(wavelets_fft, axis=-1) / self.norm
            return wavelets_time
        
    def get_shift_map(self, m: np.ndarray[int]) -> np.ndarray:
        if m.ndim == 1:
            m_in = m[:, None]
        elif m.ndim == 2:
            m_in = m
        else:
            raise ValueError("m must be 1D or 2D array.")
        return (m_in - 1) * int(self.NT / 2) + np.arange(self.NT)[None, :]
        
    def setup_window(self):

        # *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
        # zero frequency
        # REAL(DX,0) =  wdm->inv_root_dOmega
        # IMAG(DX,0) =  0.0
        T = self.dt * self.NT
        domega = 2 * np.pi / T
        self.omega = omega = (np.arange(self.NT + 1) - int(self.NT / 2)) * domega
        window = self.phitilde(omega)

        self.norm = np.sqrt(self.N * self.cadence / self.dOmega)
        self.window = window / self.norm
        assert 0.0 in omega

        self.ind_middle = np.argwhere(omega == 0.0).squeeze().item()

        omega_for_edge_layers = np.concatenate([omega[self.ind_middle:], domega * (self.ind_middle + np.arange(1, omega[:self.ind_middle].shape[0]))])
        assert (np.diff(omega_for_edge_layers).min() > 0.0) and np.allclose(np.diff(omega_for_edge_layers).max(), domega)
        self.dc_layer_window = np.sqrt(2) * self.phitilde(omega_for_edge_layers)
        self.max_freq_layer_window = self.dc_layer_window[::-1]
        
    @staticmethod
    def get_associated_class():
        return WDMSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def kwargs(self) -> dict:
        return dict(
            oversample=self.oversample, window=self.window
        )
    
    @property
    def args(self) -> tuple:
        return (self.Tobs, self.data_dt)   
    
    @property
    def differential_component(self) -> float:
        return 1.0
    
    @property
    def total_terms(self) -> int:
        return self.NT * self.NF

class WDMSignal(WDMSettings, DomainBase):
    def __init__(self, arr, settings: WDMSettings):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> WDMSettings:
        return WDMSettings(*self.args, **self.kwargs)
    
    def wdm_to_fd(self, settings=None, window=None):
        raise NotImplementedError
        phif = phitilde_vec_norm(self.NT, self.NF, 4.0)

        # determine FD parameters
        total_pixels = self.NT * self.NF
        Tobs = total_pixels * self.data_dt
        df = 1 / Tobs
        N = int((total_pixels / 2 + 1) if total_pixels % 2 == 0 else ((total_pixels + 1) / 2))
        check_settings = FDSettings(N, df)
        
        if settings is not None:
            if check_settings != settings:
                breakpoint()
                raise ValueError("Entered FD settings do not correspond to valid transform. Better to leave them blank if possible.")
        else:
            settings = check_settings
            
        # Perform the inverse transform
        new_arr = np.zeros((self.nchannels, settings.N), dtype=complex)

        for i in range(self.nchannels):
            new_arr[i] = inverse_wavelet_freq_helper(self.arr[i], phif, self.NF, self.NT)

        return FDSignal(new_arr, settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, TDSettings):
            return self.wdm_to_fd(settings=None, window=None).ifft(settings=new_domain, window=window)
        
        elif isinstance(new_domain, FDSettings):
            return self.wdm_to_fd(settings=new_domain, window=window)
        
        elif isinstance(new_domain, STFTSettings):
            return self.wdm_to_fd(settings=None, window=None).ifft(settings=None, window=None).stft(settings=new_domain, window=window)
        
        elif isinstance(new_domain, WDMSettings):
            if new_domain == self.settings:
                return self
            else:
                return self.wdm_to_fd(settings=None, window=None).wdmtransform(settings=new_domain, window=window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")

    def heatmap(self, **kwargs):
        # if fig is not None or ax is not None:
        #     if fig is None or ax is None:
        #         raise ValueError("If providing fig or ax, must provide both.")
            
        # else:
        #     # fig and ax are None
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
        
        if "cmap" not in kwargs:
            kwargs["cmap"] = cm.RdBu

        for i, (ax_i, channel)  in enumerate(zip(ax, ["X", "Y", "Z"])):
            z = self.arr[i]
            x, y = self.t_arr_edges, self.f_arr_edges
            sc = ax_i.pcolormesh(
                x, y, z, 
                # extent=[self.t_arr.min(), self.t_arr.max(), self.f_arr.min(), self.f_arr.max()], 
                **kwargs
            )
            ax_i.set_ylabel(channel)
        
        cax = fig.add_axes([0.9, 0.2, 0.05, 0.6])
        fig.colorbar(sc, cax=cax)

        plt.subplots_adjust(right=0.85, hspace=0.1)
        return fig, ax

class WDMLookupTable(WDMSettings):
    def __init__(self, settings: WDMSettings, f_steps: int, fdot_steps: int):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        d_fdot = 0.1
        fdot_step = self.df/self.T*d_fdot
        self.f_steps = f_steps
        self.fdot_steps = fdot_steps
        self.deltaf = self.BW/(self.f_steps)
        # The odd wavelets coefficienst can be obtained from the even.
        # odd cosine = -even sine, odd sine = even cosine
        # each wavelet covers a frequency band of width DW
        # execept for the first and last wasvelets
        # there is some overlap. The wavelet pixels are of width
        # DOM/PI, except for the first and last which have width
        # half that
        ref_layer = int(self.NF/2)
    
        f0 = ref_layer * self.df
        self.fdot_vals = (np.arange(fdot_steps) - int(fdot_steps / 2)) * fdot_step

        wave = self.wavelet(self.N, in_fd=False)
        
        self.f_vals = f0 + ((np.arange(self.f_steps)-self.f_steps/2)+0.5)*self.deltaf
            
        t = (np.arange(self.N) - int(self.N/2)) * self.cadence
        phase = 2 * np.pi * self.f_vals[:, None, None] * t[None, None, :] + np.pi * self.fdot_vals[None, :, None]* (t * t)[None, None, :]
        
        real_coeff = np.sum(wave * np.cos(phase)*self.cadence, axis=-1)  # TODO: trapz?
        imag_coeff = np.sum(wave * np.sin(phase)*self.cadence, axis=-1)  
        self.table = real_coeff + 1j * imag_coeff

    @property
    def table(self) -> np.ndarray:
        return self._table
    
    @table.setter
    def table(self, table: np.ndarray):
        self._table = table
        points = np.asarray([tmp.ravel() for tmp in np.meshgrid(self.f_vals, self.fdot_vals)]).T
        self._interpolant = interpolate.LinearNDInterpolator(points, table.flatten(), rescale=True)

    def get_table_coeffs(self, f_arr: np.ndarray, fdot_arr: np.ndarray):
        assert np.all((f_arr > self.f_vals.min()) & (f_arr < self.f_vals.max()))
        assert np.all((fdot_arr > self.fdot_vals.min()) & (fdot_arr < self.fdot_vals.max()))
        return self._interpolant(np.array([f_arr, fdot_arr]).T)


__available_domains__ = [TDSettings, FDSettings, STFTSettings, WDMSettings]


def get_available_domains() -> List[DomainSettingsBase]:
    return __available_domains__

# from .detector import LISAModel, ExtendedLISAModel


# class WDMSensitivityMatrix(WDMSettings, SensitivityMatrix):
#     def __init__(self, models, settings, base_sens_mat, psd_kwargs=None):
#         WDMSettings.__init__(self, *settings.args, **settings.kwargs)

#         if isinstance(models, LISAModel) or isinstance(models, ExtendedLISAModel):
#             models = [models for _ in range(self.NT)]

#         for _tmp in models:
#             assert isinstance(_tmp, LISAModel) or isinstance(_tmp, ExtendedLISAModel)

#         if psd_kwargs is None:
#             psd_kwargs = [{} for _ in range(self.NT)]
#         elif isinstance(psd_kwargs, dict):
#             psd_kwargs = [psd_kwargs for _ in range(self.NT)]

#         for _tmp in psd_kwargs:
#             assert isinstance(_tmp, dict)

#         assert isinstance(models, list) and isinstance(psd_kwargs, list)
#         assert len(models) == len(psd_kwargs) == self.NT

#         sens_mats = [base_sens_mat(settings.f_arr, model=model, **kwargs) for model, kwargs in zip(models, psd_kwargs)]
#         self.models = models
#         self.psd_kwargs = psd_kwargs

#         tmp_arr = np.asarray([tmp_mat.sens_mat for tmp_mat in sens_mats])
        
#         SensitivityMatrix.__init__(self, settings.f_arr, tmp_arr)
