from __future__ import annotations
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List
import os
import pickle
import math
import numpy as np

try:
    from cupyx.scipy import interpolate as interpolate_gpu
except (ModuleNotFoundError, ImportError):
    pass
from scipy import interpolate as interpolate_cpu
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
from .utils.parallelbase import LISAToolsParallelModule
import dataclasses


class DomainSettingsBase(LISAToolsParallelModule):
    force_backend: str = None
    def __init__(self, force_backend: str = None):
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]


class DomainBase:

    def __init__(self, arr):
        self.arr = arr

    @staticmethod
    def get(x: np.ndarray) -> np.ndarray:
        try:
            return x.get()
        except AttributeError:
            return x

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
    
    def __add__(self, other: DomainBase):
        if not isinstance(other, DomainBase):
            raise ValueError("Can only add another DomainBase object.")
        if self.settings != other.settings:
            raise ValueError("Can only add another DomainBase object with the same settings.")
        return self.__class__(self.arr + other.arr, settings=self.settings)
    
    def __sub__(self, other: DomainBase):
        if not isinstance(other, DomainBase):
            raise ValueError("Can only subtract another DomainBase object.")
        if self.settings != other.settings:
            raise ValueError("Can only subtract another DomainBase object with the same settings.")
        return self.__class__(self.arr - other.arr, settings=self.settings) 
    
    def __mul__(self, other: float):
        if not isinstance(other, (int, float)):
            raise ValueError("Can only multiply by a scalar.")
        return self.__class__(self.arr * other, settings=self.settings)

    def __truediv__(self, other: float):
        if not isinstance(other, (int, float)):
            raise ValueError("Can only divide by a scalar.")
        return self.__class__(self.arr / other, settings=self.settings)

class TDSettings(DomainSettingsBase):
    N: int
    dt: float

    def __init__(self, N: int, dt: float, **kwargs):
        self.N = N
        self.dt = dt
        super().__init__(**kwargs)

    @staticmethod
    def get_associated_class():
        return TDSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        return dict(force_backend=self.force_backend)
    
    @property
    def args(self) -> tuple:
        return (self.N, self.dt)   
    
    @property
    def t_arr(self) -> np.ndarray:
        return self.xp.arange(self.N) * self.dt
    
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

    def fft(self, settings=None, window=None, apply_dt=True):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        df = 1 / (self.N * self.dt)

        factor = 1.0 if not apply_dt else self.dt
        fd_arr = self.xp.fft.rfft(self.arr * window) * factor
        if settings is not None:
            assert isinstance(settings, FDSettings)
            assert settings.df == df
            assert settings.N == fd_arr.shape[-1]
            fd_settings = settings
        
        else:
            fd_settings = FDSettings(fd_arr.shape[-1], df, force_backend=self.backend)
        
        fd_arr_in = fd_arr[..., fd_settings.ind_min:fd_settings.ind_max + 1]
        return FDSignal(fd_arr_in, fd_settings)

    def stft(self, settings=None, window=None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

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
            window = self.xp.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # go to frequency domain then wavelets
        return self.fft(settings=None, window=window, apply_dt=False).transform(settings)

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
#     int ND = wdm->Nt*wdm->Nf
    
#     //windowed data packets
#     *wdata = double_vector(wdm->N)

#     //wavelet wavepacket transform of the signal
#     **wave = double_matrix(wdm->Nt,wdm->Nf)
    
#     //normalization factor
#     fac = M_SQRT2*sqrt(wdm->cadence)/wdm->norm
    
#     //normalization fudge factor
#     fac *= sqrt(wdm->cadence)/2
        
#     //do the wavelet transform by convolving data w/ window and FFT
#     for(int i=0 i<wdm->Nt i++)
#     {
        
#         for(int j=0 j<wdm->N j++)
#         {
#             int n = i*wdm->Nf - wdm->N/2 + j
#             if(n < 0)   n += ND  // periodically wrap the data
#             if(n >= ND) n -= ND  // periodically wrap the data
#             wdata[j] = data[n] * wdm->window[j]  // apply the window
#         }
                
#         glass_forward_real_fft(wdata, wdm->N)

#         //unpack Fourier transform
#         wave[i][0] = wdata[0]
#         for(int j=1 j<wdm->Nf j++)
#         {
#             int n = j*wdm->oversample
#             if((i+j)%2 ==0)
#                 wave[i][j] = wdata[2*n]
#             else
#                 wave[i][j] = -wdata[2*n+1]
#         }
#     }
    
#     //replace data vector with wavelet transform mapped from pixel to index
#     for(int i=0 i<wdm->Nt i++)
#     {
#         for(int j=0 j<wdm->Nf j++)
#         {
#             //get index number k for tf pixel {i,j}
#             wavelet_pixel_to_index(wdm,i,j,&k)
            
#             //replace data array
#             data[k] = wave[i][j]*fac
#         }
#     }
    
#     free_double_vector(wdata)
#     free_double_matrix(wave,wdm->Nt)
# }

#         breakpoint()

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, TDSettings):
            return self.settings.associated_class(self.arr * window, self.settings)
        
        elif isinstance(new_domain, FDSettings):
            return self.fft(settings=new_domain, window=window, apply_dt=True)
        
        elif isinstance(new_domain, STFTSettings):
            return self.stft(settings=new_domain, window=window)
        
        elif isinstance(new_domain, WDMSettings):
            return self.wdmtransform(settings=new_domain, window=window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")

# TODO: dataclass setup?
@dataclasses.dataclass
class FDSettings(DomainSettingsBase):
    N: int
    df: float
    ind_min : Optional[int] = None 
    ind_max : Optional[int] = None  # inclusive
    
    def __init__(self,
        N: int,
        df: float,
        ind_min : Optional[int] = None,
        ind_max : Optional[int] = None,
        **kwargs,
    ):
        self.N, self.df = N, df
        self.ind_min, self.ind_max = ind_min, ind_max
        super().__init__(**kwargs)

    @property
    def frequency_layer_mask(self) -> Optional[np.ndarray]:
        mask = self.xp.zeros(self.N, dtype=bool)
        mask[self.f_ind_array] = True
        return mask
        
    @property
    def f_ind_array(self) -> np.ndarray:
        return self.xp.arange(self.ind_min, self.ind_max + 1)

    @property
    def differential_component(self) -> float:
        return self.df
    
    @property
    def ind_min(self) -> int:
        return self._ind_min
    
    @ind_min.setter
    def ind_min(self, ind_min: int):
        if ind_min is None:
            ind_min = 0
        self._ind_min = ind_min

    @property
    def ind_max(self) -> int:
        return self._ind_max
    
    @ind_max.setter
    def ind_max(self, ind_max: int):
        if ind_max is None:
            ind_max = self.N - 1
        self._ind_max = ind_max

    @property
    def clipped_N(self) -> int:
        return self.ind_max - self.ind_min + 1
    
    @staticmethod
    def get_associated_class():
        return FDSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def kwargs(self) -> dict:
        return dict(ind_min=self.ind_min, ind_max=self.ind_max, force_backend=self.force_backend)

    @property
    def args(self) -> tuple:
        return (self.N, self.df)  
    
    @property
    def basis_shape(self) -> tuple:
        return (self.clipped_N,)
    
    @property
    def f_arr(self) -> np.ndarray:
        return self.f_ind_array * self.df
    
    def __eq__(self, value):
        return (value.N == self.N) and (value.df == self.df)
    
    def apply_frequency_layer_mask(self, arr: np.ndarray) -> np.ndarray:
        if self.frequency_layer_mask is None or arr.shape[-1] == self.clipped_N:
            return arr

        if arr.ndim == 1:
            return arr[self.frequency_layer_mask]
        elif arr.ndim > 1:
            assert arr.shape[-1] == self.frequency_layer_mask.shape[0], "Last dimension of arr must match length of frequency_layer_mask."
            dims_transpose = tuple(np.roll(np.arange(arr.ndim)))
            _arr = arr.transpose(dims_transpose)
            new_arr = _arr[self.frequency_layer_mask]
            dims_back = tuple(np.roll(np.arange(arr.ndim), -1))
            new_arr = new_arr.transpose(dims_back)
            return new_arr

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
        assert arr.shape[-1] == self.clipped_N

    @property
    def settings(self) -> FDSettings:
        return FDSettings(*self.args, **self.kwargs)

    def pad_array(self, arr: np.ndarray) -> np.ndarray:
        assert arr.ndim == 2
        _arr = np.pad(arr, ((0, 0), (self.ind_min - 1, self.N - 1 - self.ind_max)), mode="constant", constant_values=0.0)
        return _arr

    def ifft(self, settings=None, window=None, apply_dt=True):

        arr_in = self.arr.copy()
        
        if self.ind_min != 0 or self.ind_max != self.N - 1:
            warnings.warn("Doing an ifft with a trimmed frequency domain array. Zero-padding.")
            arr_in = self.pad_array(arr_in)

        if window is None:
            window = self.xp.ones(arr_in.shape, dtype=float)

        Tobs = 1 / self.df
        factor = 1.0 if not apply_dt else self.dt
        td_arr = self.xp.fft.irfft(arr_in * window) / factor
        N = td_arr.shape[-1]
        dt = Tobs / N
        assert N == int(Tobs / dt)
        if settings is not None:
            assert isinstance(settings, TDSettings)
            assert settings.dt == dt

        td_settings = TDSettings(N, dt, force_backend=self.force_backend)
        return TDSignal(td_arr, td_settings)
    
    def get_fd_window_for_wdm(self, settings):

        N = self.settings.N

        # solve for window
        N = (settings.Nf+1)

        # mini wavelet structure for basis covering just N layers
        T = settings.dt*settings.Nt
        domega = 2 * np.pi / T

        window = self.xp.zeros(self.N, dtype=complex)

        # wdm window function
        for i in range(0, int(settings.Nt / 2)):  # (i=0; i<=wdm->Nt/2; i++)
            omega = i*domega
            window[i] = settings.phitilde(omega)
    
        raise NotImplementedError
    
        # normalize
        # for(i=-wdm->Nt/2; i<= wdm->Nt/2; i++) norm += window[abs(i)]*window[abs(i)];
        # norm = sqrt(norm/wdm_temp->cadence);
        
        # for(i=0; i<=wdm->Nt/2; i++) window[i] /= norm;
        
        # free(wdm_temp);
        
    def wdmtransform(self, settings=None, window=None, return_transpose_time_axis_first: bool = False, is_psd: bool = False):
        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # phif = phitilde_vec_norm(settings.Nf, settings.Nt, 4.0)
        m = self.xp.repeat(self.xp.arange(0, settings.Nf)[:, None], settings.Nt, axis=-1)
        n = self.xp.tile(self.xp.arange(settings.Nt), (settings.Nf, 1))

        m_special = self.xp.repeat(self.xp.arange(0, settings.Nf + 1)[:, None], settings.Nt - 1, axis=-1)
        
        # removed zero frequency and mirrored
        # TODO: WITH ROBBIE CHECK SECOND TO TOP INDEX START AND END
        k = settings.get_shift_map(m_special)
        k = self.xp.concatenate([self.xp.zeros((k.shape[0], 1), dtype=int), k], axis=1)
        
        k[-1] -= 1

        keep_dc_layer = (k[0] >= 0)
        keep_dc_layer[0] = False
        keep_nyquist_layer = (k[-1] < self.N - 1)
        keep_nyquist_layer[0] = False


        # multiply by  2 / settings.layer_df for forward transform
        exponent = 1 if not is_psd else 2
        base_window = (settings.window[:] * 2 / settings.Nf) ** exponent
        dc_window = (settings.dc_layer_window * 2 / settings.Nf) ** exponent
        # TODO: check if this is right?!?!
        max_freq_window = (settings.max_freq_layer_window * 2 / settings.Nf) ** exponent

        k_in = k.copy()

        # this to make indexing work
        k[0, ~keep_dc_layer] = 0
        k[-1, ~keep_nyquist_layer] = 0

        assert np.sum(keep_dc_layer) == int(settings.Nt / 2)
        assert np.sum(keep_nyquist_layer) == int(settings.Nt / 2)

        arr_in = self.arr.copy()
        
        if self.ind_min != 0 or self.ind_max != self.N - 1:
            warnings.warn("Doing an ifft with a trimmed frequency domain array. Zero-padding.")
            arr_in = self.pad_array(arr_in)

        # it is 2 because the max frequency would be at 1, but it removes that (?)
        before_ifft = self.xp.zeros((self.nchannels, settings.Nf, settings.Nt), dtype=complex)
        before_ifft = arr_in[:, k]
        before_ifft[:, 0, ~keep_dc_layer] = 0.0
        before_ifft[:, -1, ~keep_nyquist_layer] = 0.0
        before_ifft[:, :, 0] = 0.0
        before_ifft2 = before_ifft.copy()

        before_ifft[:, 1:-1, 1:] *= base_window[None, None, :]
        before_ifft[:, 0, 1:] *= dc_window
        before_ifft[:, -1, 1:] *= max_freq_window

        if is_psd:
            # eq. 19 in arxiv.org/pdf/2009.00043
            # window is squared
            before_ifft[:, 1:-1, 1:] *= base_window[None, None, :]
            before_ifft[:, 0, 1:] *= dc_window
            before_ifft[:, -1, 1:] *= max_freq_window
            psd_out = self.xp.zeros(self.Nf, self.Nt)
            psd_sum_tmp = before_ifft.sum(axis=-1) / (self.data_dt * self.Nt * self.Nf)
            wdmpsd = self.xp.zeros((self.nchannels, self.Nf, self.Nt))
            breakpoint()
            wdmpsd[:, 1:] = psd_sum_tmp[1:self.Nf]          # regular layers
            wdmpsd[0::2, 0] = psd_sum_tmp[0]           # DC at even rows
            wdmpsd[1::2, 0] = psd_sum_tmp[self.Nf]     
            return wdmpsd

        after_ifft = self.xp.fft.ifft(before_ifft, axis=-1)
        
        is_m_plus_n_even = (((m + n) % 2 == 0)) 
        _new_arr = self.xp.zeros((self.nchannels, settings.Nf, settings.Nt), dtype=float)

        # TODO: fix this

        if self.backend.uses_cupy:
            # some issue with cupy and xp.real/imag
            cache = self.xp.fft.config.get_plan_cache()
            cache.clear()

        _new_arr[:, is_m_plus_n_even] = self.xp.real(after_ifft[:, :-1][:, is_m_plus_n_even])
        _new_arr[:, (~is_m_plus_n_even)] = (-1) ** (m[(~is_m_plus_n_even)]) * self.xp.imag(after_ifft[:, :-1][:, (~is_m_plus_n_even)])
        
        # Robbie says this is okay
        # TODO: fix this 
        _new_arr[:, 0, 0::2] = self.xp.asarray(np.real(self.get(after_ifft[:, 0, 0::2]))) * np.sqrt(2.)
        _new_arr[:, 0, 1::2] = self.xp.asarray(np.real(self.get(after_ifft[:, -1, 0::2]))) * np.sqrt(2.)
        
        if return_transpose_time_axis_first:
            output = _new_arrn.transpose(0, 2, 1).copy()
        else:
            output = _new_arr

        return WDMSignal(output, settings=settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, FDSettings):
            return self.settings.associated_class(self.arr * window, self.settings)
        
        elif isinstance(new_domain, TDSettings):
            return self.ifft(settings=new_domain, window=window, apply_dt=True)
        
        elif isinstance(new_domain, STFTSettings):
            raise NotImplementedError
            return self.stft()

        elif isinstance(new_domain, WDMSettings):
            return self.wdmtransform(settings=new_domain, window=new_domain.window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")


class STFTSettings(DomainSettingsBase):
    dt: float
    df: float 
    Nt: int
    Nf: int

    def __init__(self, 
        dt: float,
        df: float, 
        Nt: int,
        Nf: int,
        **kwargs
    ):
        self.dt, self.df = dt, df
        self.Nt, self.Nf = Nt, Nf
        super().__init__(**kwargs)
    
    @staticmethod
    def get_associated_class():
        return STFTSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def t_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nt) * self.dt

    @property
    def f_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nf) * self.df
    
    @property
    def f_arr_edges(self) -> np.ndarray:
        return self.xp.arange(self.Nf + 1) * self.df
    @property
    def t_arr_edges(self) -> np.ndarray:
        return self.xp.arange(self.Nt + 1) * self.dt

    def __eq__(self, value):
        return (value.Nt == self.Nt) and (value.Nf == self.Nf) and (value.dt == self.dt) and (value.df == self.df)

class STFTSignal(STFTSettings, DomainBase):
    def __init__(self, arr, settings: STFTSettings):
        STFTSettings.__init__(self, settings.dt, settings.df)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> STFTSettings:
        return STFTSettings(self.dt, self.df, force_backend=self.force_backend)
    
    @property
    def differential_component(self) -> float:
        return 1.0
    
    @property
    def total_terms(self) -> int:
        return self.Nt * self.Nf


WAVELET_BANDWIDTH = 6.51041666666667e-5
WAVELET_DURATION = 7680.0
WAVELET_FILTER_CONSTANT = 6

from scipy import special

class WDMSettings(DomainSettingsBase):

    def __init__(
        self,
        Nf: float, 
        Nt: float,
        dt: float,
        oversample: int = 16,
        window: Optional[np.ndarray] = None,
        dc_layer_window: Optional[np.ndarray] = None,
        max_freq_layer_window: Optional[np.ndarray] = None,
        norm: Optional[float] = None, 
        omega: Optional[np.ndarray] = None,
        ind_min: Optional[int] = None,
        ind_max: Optional[int] = None,
        **kwargs
    ):

        DomainSettingsBase.__init__(self, **kwargs)
        self.Nt = Nt
        self.Nf = Nf
        self.data_dt = dt
        self.N = self.Nt * self.Nf
        self.data_dt = dt
        self.Tobs = self.N * self.data_dt
        self.ind_min = ind_min
        self.ind_max = ind_max
        self.layer_dt = self.Nf * self.data_dt
        self.layer_df = 1. / (2. * self.Nf * self.data_dt)
        self.Nthalf = int(self.Nt / 2)
        self.oversample = oversample

        self.dOmega = 2 * np.pi * self.layer_df
        self.A = 0.0
        self.WAVELET_FILTER_CONSTANT = 6  # window roll-off parameter
        
        if window is None:
            self.setup_window()
        else:
            assert norm is not None, "Must provide norm if providing window."
            assert omega is not None, "Must provide omega if providing window."
            assert dc_layer_window is not None, "Must provide dc_layer_window if providing window."
            assert max_freq_layer_window is not None, "Must provide max_freq_layer_window if providing window."
            assert len(window) == self.Nt - 1
            self.window = window
            self.dc_layer_window = dc_layer_window
            self.max_freq_layer_window = max_freq_layer_window
            self.norm = norm
            self.omega = omega

    @staticmethod
    def adjust_to_even_bins(t_min: float, t_max: float, dt: float, Tobs: float, num_linspace: Optional[int]=1000, verbose: Optional[bool] = False) -> Tuple[int, int, float]:
        Nf = -1
        Nt = -1

        found_wavelet = False
        wavelet_duration = -1.0
        for tmp in np.linspace(t_min, t_max, num_linspace):
            wavelet_duration = int(tmp / dt) * dt
            Nt = int(Tobs / wavelet_duration)
            Tobs = Nt * wavelet_duration
            N = int(Tobs / dt)
            Nf = int(N / Nt)
            if verbose:
                print(f"Attempting wavelet duration: {tmp}, Nf: {Nf}, Nt: {Nt}")
            if (Nt % 2 == 0) and (Nf % 2 == 0):
                found_wavelet = True
                break
        
        if not found_wavelet:
            raise ValueError(f"Could not find suitable wavelet parameters for even numbered Nf and Nt in the range given ({t_min}, {t_max}).")
        
        return (Nf, Nt, wavelet_duration)
    
    def __eq__(self, value):
        return (value.Nt == self.Nt) and (value.Nf == self.Nf) and (value.layer_dt == self.layer_dt) and (value.layer_df == self.layer_df) and (value.data_dt == self.data_dt)

    @property
    def basis_shape(self) -> tuple:
        return (self.Nf, self.Nt)
    
    @property
    def t_td_arr(self) -> np.ndarray:
        return self.xp.arange(self.N) * self.data_dt
    
    @property
    def t_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nt) * self.layer_dt

    @property
    def f_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nf) * self.layer_df
    
    @property
    def f_arr_edges(self) -> np.ndarray:
        return self.xp.arange(self.Nf + 1) * self.layer_df
    @property
    def t_arr_edges(self) -> np.ndarray:
        return self.xp.arange(self.Nt + 1) * self.layer_dt

    def phitilde(self, omega, dOmega):
        insDOM = 1. / np.sqrt(dOmega)
        A = self.A
        B = dOmega - 2 * A

        z = self.xp.zeros(omega.shape[0])
        beta_inc_calc = (np.abs(omega) >= A) & (np.abs(omega) <= A+B)
        x = (np.abs(omega[beta_inc_calc])-A)/B
        y = special.betainc(self.WAVELET_FILTER_CONSTANT, self.WAVELET_FILTER_CONSTANT, x)
        z[beta_inc_calc] = insDOM*np.cos(y*np.pi/2.0)
        z[(np.abs(omega) < A)] = insDOM
        #breakpoint()
        return z
    
    def wavelet(self, N: int, in_fd: Optional[bool] = True) -> np.ndarray:
        raise NotImplementedError
        # Nt * Nf is even 
        # assert (self.Nt * self.Nf) % 2 == 0
        base_window = self.window[:-1]
        omega_N = (self.xp.arange(self.N) - int(self.N / 2)) * self.domega
        wavelet_N = 1 / np.sqrt(2.) * self.phitilde(omega_N)
        
        if in_fd:
            return wavelet_N
        else:
            return self.xp.fft.ifft(wavelet_N) / self.norm
        breakpoint()
        wavelets_rfft = self.xp.zeros((len(m), int((self.Nt * self.Nf) / 2 + 1)))

        self.xp.put_along_axis(wavelets_rfft, k, base_window * 1 / np.sqrt(2.), axis=-1)
        freq = self.xp.fft.fftshift(self.xp.fft.fftfreq(self.Nt * self.Nf, self.data_dt))
        wavelets_fft = self.xp.exp(-1j * 2 * np.pi * freq[None, :] * n[:, None] * self.data_dt) * self.xp.concatenate([wavelets_rfft[:, ::-1][:, :-1], wavelets_rfft[:, :-1]], axis=-1)
        if in_fd:
            return wavelets_fft
        else:
            wavelets_time = self.xp.fft.ifft(wavelets_fft, axis=-1) / self.norm
            return wavelets_time
        
    def get_shift_map(self, m: np.ndarray[int]) -> np.ndarray:
        if m.ndim == 1:
            m_in = m[:, None]
        elif m.ndim == 2:
            m_in = m
        else:
            raise ValueError("m must be 1D or 2D array.")

        return m_in * int(self.Nt / 2) + self.xp.arange( -int(self.Nt / 2) + 1,  int(self.Nt / 2))[None, :]
        
    def window_norm(self) -> float:
        dOmega_s = np.pi / self.Nf
        (2 * np.pi) / self.N 
    def setup_window(self):  # , forward: bool= True):

        # *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
        # zero frequency
        # REAL(DX,0) =  wdm->inv_root_dOmega
        # IMAG(DX,0) =  0.0
        T = self.data_dt * self.N
        dOmega_s = np.pi / self.Nf
        self.omega = omega =  2 * np.pi / self.N * (self.xp.arange(self.Nt - 1) - int(self.Nt / 2) + 1)
        phif = self.phitilde(omega, dOmega_s)
        self.norm = np.sqrt((np.sum(phif ** 2)) * 2. / self.N)
        self.window = phif / self.norm

        # we apply this outside the window setup so the window is the same for forward and backward
        # if forward:
        #     self.window *= 2. / self.Nf

        assert 0.0 in omega

        self.ind_middle = self.xp.argwhere(omega == 0.0).squeeze().item()

        self.dc_layer_window = self.window.copy()
        self.dc_layer_window[self.ind_middle] /= 2.0
        self.max_freq_layer_window = self.window.copy()
        self.max_freq_layer_window[self.ind_middle] /= 2.0

    @property
    def ind_min(self) -> int:
        return self._ind_min
    
    @ind_min.setter
    def ind_min(self, ind_min: int):
        if ind_min is None:
            ind_min = 0
        self._ind_min = ind_min

    @property
    def ind_max(self) -> int:
        return self._ind_max
    
    @ind_max.setter
    def ind_max(self, ind_max: int):
        if ind_max is None:
            ind_max = self.Nf - 1
        self._ind_max = ind_max

    @property
    def frequency_layer_mask(self) -> Optional[np.ndarray]:
        mask = self.xp.zeros(self.Nf, dtype=bool)
        mask[self.f_ind_array] = True
        return mask
        
    @property
    def f_ind_array(self) -> np.ndarray:
        return self.xp.arange(self.ind_min, self.ind_max + 1)

    @staticmethod
    def get_associated_class():
        return WDMSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def kwargs(self) -> dict:
        return dict(
            oversample=self.oversample, 
            window=self.window, 
            dc_layer_window=self.dc_layer_window,
            max_freq_layer_window=self.max_freq_layer_window,
            norm=self.norm, 
            omega=self.omega, 
            ind_min=self.ind_min, 
            ind_max=self.ind_max,
            force_backend=self.force_backend
        )
    
    @property
    def args(self) -> tuple:
        return (self.Nf, self.Nt, self.data_dt)   
    
    @property
    def differential_component(self) -> float:
        return self.layer_df

    @property
    def total_terms(self) -> int:
        return self.Nt * self.Nf
    
    def apply_frequency_layer_mask(self, arr: np.ndarray) -> np.ndarray:
        if self.frequency_layer_mask is None:
            return arr
        elif arr.ndim == 1:
            raise ValueError("arr must be at least 2D to apply frequency layer mask.")
        elif arr.ndim == 2:
            return arr[self.frequency_layer_mask]
        elif arr.ndim > 2:
            raise NotImplementedError
            assert arr.shape[-2] == self.frequency_layer_mask.shape[0], "Last dimension of arr must match length of frequency_layer_mask."
            dims_transpose = tuple(np.roll(np.arange(arr.ndim)))
            _arr = arr.transpose(dims_transpose)
            new_arr = _arr[self.frequency_layer_mask]
            dims_back = tuple(np.roll(np.arange(arr.ndim), -1))
            new_arr = new_arr.transpose(dims_back)
            return new_arr

class WDMSignal(WDMSettings, DomainBase):
    def __init__(self, arr, settings: WDMSettings):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> WDMSettings:
        return WDMSettings(*self.args, **self.kwargs)
    
    def wdm_to_fd(self, settings=None, window=None):
        
        if settings is None:
            _tmp_fd = np.fft.rfftfreq(self.N, self.data_dt)
            Nfd = len(_tmp_fd)
            df = _tmp_fd[1] - _tmp_fd[0]
            settings = FDSettings(Nfd, df, force_backend=self.backend)

        else:
            Nfd = len(_tmp_fd := np.fft.rfftfreq(self.N, self.data_dt))
            df = _tmp_fd[1] - _tmp_fd[0]
            assert settings.N == Nfd
            assert settings.df == df

        base_window = self.window.copy()

        m = self.xp.repeat(self.xp.arange(0, self.Nf)[:, None], self.Nt, axis=-1)
        n = self.xp.tile(self.xp.arange(self.Nt), (self.Nf, 1))

        m_special = self.xp.repeat(self.xp.arange(0, self.Nf + 1)[:, None], self.Nt - 1, axis=-1)
        
        _new_arr = self.arr.copy()


        # we are going to try to write this as the reverse of the forward
       
        after_ifft = self.xp.zeros((self.nchannels, self.Nf + 1, self.Nt), dtype=complex)

        is_m_plus_n_even = (((m + n) % 2 == 0))[None, :]
        # _new_arr = self.xp.zeros((self.nchannels, settings.Nf, settings.Nt), dtype=float)
        # _new_arr[:, is_m_plus_n_even] = self.xp.real(after_ifft)[:, :-1][:, is_m_plus_n_even]
        # _new_arr[:, (~is_m_plus_n_even)] = (-1) ** (m[(~is_m_plus_n_even)]) * self.xp.imag(after_ifft)[:, :-1][:, (~is_m_plus_n_even)]
        
        # # Robbie says this is okay
        # _new_arr[:, 0, 0::2] = np.real(after_ifft[:, 0, 0::2]) * np.sqrt(2.)
        # _new_arr[:, 0, 1::2] = np.real(after_ifft[:, -1, 0::2]) * np.sqrt(2.)
        _n_arr = np.arange(0, self.Nt)
        after_ifft[:, 1:-1] = (_new_arr[:, 1:] * (is_m_plus_n_even[:, 1:])) + (1j * _new_arr[:, 1:] * (~is_m_plus_n_even[:, 1:]) * (-1.) ** (-m[None, 1:]))
        after_ifft[:, 0, :] = _new_arr[:, 0, ((2 * _n_arr) % self.Nt)] / np.sqrt(2.)
        after_ifft[:, -1, :] = _new_arr[:, 0, ((2 * _n_arr) % self.Nt) + 1] / np.sqrt(2.)
        
        before_ifft = self.xp.fft.fft(after_ifft, axis=-1)

        # leave out 2 / self.layer_df for reverse transform
        base_window = self.window[:]  # * 2 / self.layer_df  
        dc_window = self.dc_layer_window  # * 2 / self.layer_df
        # TODO: check if this is right?!?!
        max_freq_window = self.max_freq_layer_window  # * 2 / settings.layer_df
        
        back_before_ifft = before_ifft.copy()
        
        keep_roll = ((_x := np.arange(self.Nf + 1)) % 2 == 0) & (_x != 0) & (_x != self.Nf)
        # roll ifft
        before_ifft[:, keep_roll] = np.roll(before_ifft[:, keep_roll], self.Nthalf, axis=-1)
        before_ifft[:, 1:-1, 1:] *= base_window[None, None, :]
        before_ifft[:, 1:-1, 0] = 0.0  # set DC component to zero

        # dc window
        fft_result_keep_dc = (np.arange(self.Nt) % self.Nt)[0::2]
        window_keep_dc = base_window[self.Nthalf - 1:]  # right half of window
        before_ifft[:, 0, fft_result_keep_dc] *= window_keep_dc

        # nyquist window
        fft_result_keep_ny = (np.arange(self.Nt + 1) % self.Nt)[0::2][1:]
        window_keep_ny = base_window[:self.Nthalf]  # left half of window
        before_ifft[:, -1, fft_result_keep_ny] *= window_keep_ny

        # before_ifft = self.arr[:, k]
        output_arr = self.xp.zeros((self.nchannels, settings.N), dtype=complex)

        k = self.get_shift_map(m_special)
        k = self.xp.concatenate([-self.xp.ones((k.shape[0], 1), dtype=int), k], axis=1)
        
        # to match Robbie/Tyson implementation
        fft_result_keep_ny = (np.arange(self.Nt + 1) % self.Nt)[0::2][1:]
        k[0] = -1
        k[0, 0::2] = np.arange(self.Nthalf)

        k[-1] = -1
        k[-1, 0::2] = np.roll(np.arange(settings.N - self.Nthalf, settings.N), 1)

        # TODO: vectorize
        for j in range(self.nchannels):
            for k_i, val in zip(k.flatten(), before_ifft[j].flatten()):
                if k_i >= 0 and k_i < settings.N:
                    output_arr[j, k_i] += val

        return FDSignal(output_arr, settings)
        breakpoint()
        prefactors = self.xp.zeros((self.nchannels, self.Nf + 1, self.Nt), dtype=complex)
        is_m_plus_n_even = (((m + n) % 2 == 0))[None, :]
        prefactors[:, 1:-1] = self.arr[:, 1:] * is_m_plus_n_even[:, 1:] - 1j * self.arr[:, 1:] * (~is_m_plus_n_even)[:, 1:]
        
        # dc layer
        prefactors[:, 0] = self.arr[:, 0, (2*(n[0])) % self.Nt+0] / np.sqrt(2.)
        prefactors[:, -1] = self.arr[:, 0, (2*(n[-1])) % self.Nt+1] / np.sqrt(2.)

        Nthalf = int(self.Nt / 2)
        after_fft = self.xp.fft.ifft(prefactors, axis=-1)

        after_fft_with_window = after_fft[:, :, 1:] * base_window[None, None, :]
        
        k = self.get_shift_map(m_special)[:, :Nthalf]  # center - Nthalf + self.xp.arange(0, self.Nt)[None, :    ]

        # this makes the indexing work for  the first piece
        assert len(_tmp_2_k := ((_tmp_k := np.unique(k))[(_tmp_k >= 0) & (_tmp_k < self.N)])) == settings.N
        assert self.xp.all(_tmp_2_k == self.xp.arange(settings.N))

        keep_k = (k >= 0)
        pre_transform = after_fft_with_window[:, :, 1:Nthalf + 1]

        new_arr = pre_transform[:, keep_k]

        assert new_arr.shape[-1] == settings.N
        assert new_arr.shape == (self.nchannels, settings.N)
        return FDSignal(new_arr, settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, TDSettings):
            return self.wdm_to_fd(settings=None, window=None).ifft(settings=new_domain, window=window, apply_dt=False)
        
        elif isinstance(new_domain, FDSettings):
            return self.wdm_to_fd(settings=new_domain, window=window)
        
        elif isinstance(new_domain, STFTSettings):
            return self.wdm_to_fd(settings=None, window=None).ifft(settings=None, window=None, apply_dt=False).stft(settings=new_domain, window=window)
        
        elif isinstance(new_domain, WDMSettings):
            if new_domain == self.settings:
                return self
            else:
                return self.wdm_to_fd(settings=None, window=None).wdmtransform(settings=new_domain, window=window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")

    def heatmap(self, index: int = None, fig=None, ax=None, cax=None, add_cax=False, **kwargs):
        # if fig is not None or ax is not None:
        #     if fig is None or ax is None:
        #         raise ValueError("If providing fig or ax, must provide both.")
            
        # else:
        #     # fig and ax are None
        if "cmap" not in kwargs:
            kwargs["cmap"] = cm.RdBu

        if index is None:
            if fig is None and ax is None:
                fig, ax = plt.subplots(self.nchannels, 1, sharex=True, sharey=True)
            else:
                assert ax is not None

            for i, (ax_i, channel)  in enumerate(zip(ax, ["X", "Y", "Z"])):
                z = self.arr[i]
                x, y = self.t_arr_edges, self.f_arr_edges
                x = self.get(x)
                y = self.get(y)
                z = self.get(z)

                sc = ax_i.pcolormesh(
                    x, y, z, 
                    # extent=[self.t_arr.min(), self.t_arr.max(), self.f_arr.min(), self.f_arr.max()], 
                    **kwargs
                )
                ax_i.set_ylabel(channel)

        else:
            assert index is not None and fig is not None and ax is not None
            z = self.arr[index]
            x, y = self.t_arr_edges, self.f_arr_edges
            x = self.get(x)
            y = self.get(y)
            z = self.get(z)
            sc = ax.pcolormesh(
                x, y, z, 
                # extent=[self.t_arr.min(), self.t_arr.max(), self.f_arr.min(), self.f_arr.max()], 
                **kwargs
            )

        if add_cax:
            cax = fig.add_axes([0.9, 0.2, 0.05, 0.6])

        if add_cax or cax is not None:
            fig.colorbar(sc, cax=cax)
        
        plt.subplots_adjust(right=0.85, hspace=0.1)

        return fig, ax

class WDMLookupTable(WDMSettings):
    def __init__(self, settings: WDMSettings, eps_f: int, eps_fdot: int, nchannels: int, num_layers_diff: int=2, fdot_max_factor: float= 8.0, store_path: Optional[str] = None, batch_size_gen: Optional[int] = 20, time_layers: Optional[int] = None, td_window: Optional[np.ndarray] = None):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        # TODO: CHECK FIRST AND LAST TIME LAYERS DUE TO TIME WINDOWING?

        self.store_path = store_path
        if time_layers is None:
            time_layers = self.Nt

        assert isinstance(time_layers, int)
        
        self.nchannels = nchannels
        self.sub_settings = WDMSettings(self.Nf, time_layers, self.data_dt, force_backend=self.force_backend)
        self.num_layers_diff = num_layers_diff
        self.m_ref = int(3e-3 / self.sub_settings.layer_df)  # int(self.sub_settings.Nf / 2)
        self.n_ref = int(self.sub_settings.Nt / 2)
        self.is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        self.f_ref = self.m_ref * self.sub_settings.layer_df
        
        assert 0.0 < eps_f < 1.0
        assert 0.0 < eps_fdot < 1.0
        
        self.eps_f = eps_f
        self.delta_f = eps_f * self.layer_df

        _freq = self.xp.arange(self.f_ref, self.f_ref + num_layers_diff * self.sub_settings.layer_df, self.delta_f)
        _norm_freq = _freq - self.f_ref
        self.f_vals_norm = self.xp.concatenate([-_norm_freq[::-1][:-1], _norm_freq])
        self.f_vals = self.f_vals_norm + self.f_ref

        self.f_min = self.f_vals.min().item()
        self.f_max = self.f_vals.max().item()

        self.f_steps = len(self.f_vals)

        self.eps_fdot = eps_fdot
        
        if fdot_max_factor == 0.0:
            self.run_fdot = False
            self.fdot_steps = 1
            self.fdot_vals = self.xp.array([0.0])
            self.delta_fdot = 0.0

        else:
            self.run_fdot = True
            self.delta_fdot = eps_fdot * self.layer_df / self.layer_dt
            self.fdot_max_val = fdot_max_factor * self.layer_df / self.layer_dt
            _fdot = self.xp.arange(0.0, self.fdot_max_val, self.delta_fdot)
            self.fdot_vals = self.xp.concatenate([-_fdot[::-1][:-1], _fdot])
            
            self.fdot_min = self.fdot_vals.min().item()
            self.fdot_max = self.fdot_vals.max().item()

            self.fdot_steps = len(self.fdot_vals)
            
        run_table_gen = True
        if store_path is not None:
            if os.path.exists(self.store_path):
                with open(self.store_path, "rb") as fp:
                    check_input = pickle.load(fp)
                if (
                    check_input["basis_settings"] == self.sub_settings and
                    check_input["f_steps"] == self.f_steps and
                    check_input["fdot_steps"] == self.fdot_steps and
                    check_input["delta_f"] == self.delta_f and
                    check_input["delta_fdot"] == self.delta_fdot
                ):
                    run_table_gen = False
                    self.table_sin = self.xp.asarray(check_input["table_sin"])
                    self.table_cos = self.xp.asarray(check_input["table_cos"])

        if run_table_gen:
            self.t_ref = self.n_ref * self.sub_settings.layer_dt
            
            total_f_fdot_vals = self.f_steps * self.fdot_steps
            
            if self.run_fdot:
                _f_vals, _fdot_vals = self.points.T
            else:
                _f_vals = self.points
                _fdot_vals = self.xp.zeros_like(_f_vals)

            t_vals = self.xp.arange(self.sub_settings.N) * self.data_dt

            t_diff = t_vals - self.t_ref

            if batch_size_gen == -1:
                batch_size_gen = total_f_fdot_vals
                
            batches = np.arange(0, total_f_fdot_vals, batch_size_gen)
            if batches[-1] < total_f_fdot_vals:
                batches = np.append(batches, np.array([total_f_fdot_vals]))

            _table_sin = self.xp.zeros((total_f_fdot_vals,))
            _table_cos = self.xp.zeros((total_f_fdot_vals,))
            if td_window is None:
                td_window = self.xp.ones_like(t_diff)

            self.td_window = td_window
            for st_batch, end_batch in zip(batches[:-1], batches[1:]):
                inds = slice(st_batch, end_batch)
                # if not self.xp.allclose(_f_vals[inds] - self.f_ref, 0.0) or self.xp.any(_fdot_vals[inds] != 0.0):
                #     continue
                
                wave_sin = self.xp.sin(2 * np.pi * (_f_vals[inds, None] * t_diff[None, :] + 1. / 2. * _fdot_vals[inds, None] * t_diff[None, :] ** 2))
                wave_cos = self.xp.cos(2 * np.pi * (_f_vals[inds, None] * t_diff[None, :] + 1. / 2. * _fdot_vals[inds, None] * t_diff[None, :] ** 2))
                
                wave_sin_wdm = TDSignal(wave_sin, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)
                wave_cos_wdm = TDSignal(wave_cos, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)
                try:
                    _table_sin[inds] = wave_sin_wdm[:, self.m_ref, self.n_ref] 
                    _table_cos[inds] = wave_cos_wdm[:, self.m_ref, self.n_ref]
                except:
                    breakpoint()
                print(inds, total_f_fdot_vals)
                
            self.table_sin = _table_sin.reshape((self.f_steps, self.fdot_steps)).copy()
            self.table_cos = _table_cos.reshape((self.f_steps, self.fdot_steps)).copy()

            if store_path is not None:
                output_dict = {
                    "basis_settings": self.get(self.sub_settings),
                    "f_steps": self.get(self.f_steps),
                    "fdot_steps": self.get(self.fdot_steps),
                    "delta_f": self.get(self.delta_f),
                    "delta_fdot": self.get(self.delta_fdot),
                    "table_sin": self.get(self.table_sin),
                    "table_cos": self.get(self.table_cos)
                }
                with open(self.store_path, "wb") as fp:
                    pickle.dump(output_dict, fp, pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def get(x: np.ndarray) -> np.ndarray:
        try:
            return x.get()
        except AttributeError:
            return x

    @property
    def points(self) -> np.ndarray:
        if self.run_fdot:
            return self.xp.asarray([tmp.ravel() for tmp in self.xp.meshgrid(self.f_vals, self.fdot_vals)]).T
        else:
            return self.f_vals
        
    @property
    def norm_points(self) -> np.ndarray:
        if self.run_fdot:
            return self.xp.asarray([tmp.ravel() for tmp in self.xp.meshgrid(self.f_vals_norm, self.fdot_vals)]).T
        else:
            return self.f_vals_norm
        
    @property
    def table_sin(self) -> np.ndarray:
        return self._table_sin
    
    @property
    def table_sin_interpolate(self) -> np.ndarray:
        return self._table_sin_interpolate

    @table_sin.setter
    def table_sin(self, table_sin: np.ndarray):
        self._table_sin = table_sin
        self._table_sin_interpolate = self.build_interpolator(table_sin)

    @property
    def table_cos(self) -> np.ndarray:
        return self._table_cos
    
    @property
    def table_cos_interpolate(self) -> np.ndarray:
        return self._table_cos_interpolate
    
    @table_cos.setter
    def table_cos(self, table_cos: np.ndarray):
        self._table_cos = table_cos
        self._table_cos_interpolate = self.build_interpolator(table_cos)

    @property
    def settings(self) -> TDSettings:
        return WDMSettings(*self.args, **self.kwargs)

    def build_interpolator(self, table: np.ndarray):
        if self.backend.uses_cupy:
            interpolate = interpolate_gpu
        else:
            interpolate = interpolate_cpu

        if self.run_fdot:
            return interpolate.LinearNDInterpolator(self.norm_points, table.flatten(), rescale=True)
        else:
            return interpolate.interp1d(self.norm_points, table.flatten())
        
    def get_table_coeffs(self, f_arr: np.ndarray, fdot_arr: np.ndarray, ms: np.ndarray):
        # ms = (f_arr // self.layer_df).astype(int)
        assert ms.max() <= self.Nf + 1
        assert ms.min() >= 0
        assert self.xp.all((f_arr >= 0.0) & (f_arr <= self.f_arr.max()))
        assert self.xp.all((fdot_arr >= self.fdot_vals.min()) & (fdot_arr <= self.fdot_vals.max()))
        f_norm = (f_arr - ms * self.layer_df)

        if self.run_fdot:
            sin_coeffs = self.table_sin_interpolate(f_norm, fdot_arr)
            cos_coeffs = self.table_cos_interpolate(f_norm, fdot_arr)
        else:
            sin_coeffs = self.table_sin_interpolate(f_norm)
            cos_coeffs = self.table_cos_interpolate(f_norm)
        
        sin_coeffs[np.isnan(sin_coeffs)] = 0.0
        cos_coeffs[np.isnan(cos_coeffs)] = 0.0
        return (sin_coeffs, cos_coeffs)

    def get_wdm_coeffs(self, amp_arr: np.ndarray, phi_arr: np.ndarray, f_arr: np.ndarray, fdot_arr: np.ndarray, n_arr: np.ndarray, num_m_layers: int = 1):
        ms = (f_arr // self.layer_df).astype(int)
        wdm_coeffs_out = self.xp.zeros((amp_arr.shape[0], num_m_layers * 2 + 1))
        m_map = -self.xp.ones((amp_arr.shape[0], num_m_layers * 2 + 1), dtype=int)
        is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        for i, m_diff in enumerate(range(-num_m_layers, num_m_layers + 1)):
            ms_to_use = (ms + m_diff).astype(int)
            keep_now = self.xp.arange(ms_to_use.shape[0])[(ms_to_use >= 0) & (ms_to_use <= self.Nf + 1)]
            _sin_coeffs, _cos_coeffs = self.get_table_coeffs(f_arr[keep_now], fdot_arr[keep_now], ms_to_use[keep_now])
            
            is_m_plus_n_even = (((ms_to_use[keep_now] + n_arr[keep_now]) % 2 == 0)) 

            sin_coeffs = self.xp.zeros_like(_sin_coeffs)
            cos_coeffs = self.xp.zeros_like(_cos_coeffs)
            if self.is_m_ref_n_ref_even:
                sin_coeffs[~is_m_plus_n_even] = _sin_coeffs[~is_m_plus_n_even]
                cos_coeffs[~is_m_plus_n_even] = _cos_coeffs[~is_m_plus_n_even]

                sin_coeffs[is_m_plus_n_even] = _cos_coeffs[is_m_plus_n_even]
                cos_coeffs[is_m_plus_n_even] = -_sin_coeffs[is_m_plus_n_even]
                
                # keep1 = (~is_m_plus_n_even & ~is_m_odd)
                # sin_coeffs[keep1] = _sin_coeffs[keep1]
                # cos_coeffs[keep1] = _cos_coeffs[keep1]

                # keep2 = (is_m_plus_n_even & ~is_m_odd)
                # sin_coeffs[keep2] = _cos_coeffs[keep2]
                # cos_coeffs[keep2] = -_sin_coeffs[keep2]

                # keep3 = (~is_m_plus_n_even & is_m_odd)
                # sin_coeffs[keep3] = _cos_coeffs[keep3]
                # cos_coeffs[keep3] = _sin_coeffs[keep3]

                # keep4 = (is_m_plus_n_even & is_m_odd)
                # sin_coeffs[keep4] = _sin_coeffs[keep4]
                # cos_coeffs[keep4] = _cos_coeffs[keep4]


            else:
                print("Need to explicitly check this.")
                sin_coeffs[is_m_plus_n_even] = -_cos_coeffs[is_m_plus_n_even]
                cos_coeffs[is_m_plus_n_even] = _sin_coeffs[is_m_plus_n_even]

                sin_coeffs[~is_m_plus_n_even] = _sin_coeffs[~is_m_plus_n_even]
                cos_coeffs[~is_m_plus_n_even] = _cos_coeffs[~is_m_plus_n_even]
            
            # TODO: idk if this is right NEED TO CHECK
            wdm_coeffs_out[keep_now, i] = amp_arr[keep_now] * (sin_coeffs * self.xp.sin(phi_arr[keep_now]) + cos_coeffs * self.xp.cos(phi_arr[keep_now]))
            m_map[keep_now, i] = ms_to_use[keep_now]
        return wdm_coeffs_out, m_map
__available_domains__ = [TDSettings, FDSettings, STFTSettings, WDMSettings]


def get_available_domains() -> List[DomainSettingsBase]:
    return __available_domains__

# from .detector import LISAModel, ExtendedLISAModel


# class WDMSensitivityMatrix(WDMSettings, SensitivityMatrix):
#     def __init__(self, models, settings, base_sens_mat, psd_kwargs=None):
#         WDMSettings.__init__(self, *settings.args, **settings.kwargs)

#         if isinstance(models, LISAModel) or isinstance(models, ExtendedLISAModel):
#             models = [models for _ in range(self.Nt)]

#         for _tmp in models:
#             assert isinstance(_tmp, LISAModel) or isinstance(_tmp, ExtendedLISAModel)

#         if psd_kwargs is None:
#             psd_kwargs = [{} for _ in range(self.Nt)]
#         elif isinstance(psd_kwargs, dict):
#             psd_kwargs = [psd_kwargs for _ in range(self.Nt)]

#         for _tmp in psd_kwargs:
#             assert isinstance(_tmp, dict)

#         assert isinstance(models, list) and isinstance(psd_kwargs, list)
#         assert len(models) == len(psd_kwargs) == self.Nt

#         sens_mats = [base_sens_mat(settings.f_arr, model=model, **kwargs) for model, kwargs in zip(models, psd_kwargs)]
#         self.models = models
#         self.psd_kwargs = psd_kwargs

#         tmp_arr = np.asarray([tmp_mat.sens_mat for tmp_mat in sens_mats])
        
#         SensitivityMatrix.__init__(self, settings.f_arr, tmp_arr)
