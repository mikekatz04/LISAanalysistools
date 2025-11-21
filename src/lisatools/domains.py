from __future__ import annotations
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List

import math
import numpy as np
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt

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

    @property
    def arr(self) -> np.ndarray:
        return self._arr
    
    @arr.setter
    def arr(self, arr: np.ndarray):
        self._arr = arr

    def fft(self, settings=None, window=None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        N = self.arr.shape[0]
        df = 1 / (N * self.dt)
        
        if settings is not None:
            assert isinstance(settings, FDSettings)
            assert settings.df == df

        fd_arr = np.fft.rfft(self.arr * window)
        fd_settings = FDSettings(df)
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

        # windowed data packets
        wdata = np.zeros((self.arr.shape[0], settings.N,)) # , double_vector(wdm->N)
        
        # wavelet wavepacket transform of the signal
        wave = np.zeros((self.arr.shape[0], settings.NT, settings.NF), dtype=complex)
            
        # normalization factor
        fac = np.sqrt(2) * np.sqrt(settings.cadence)/settings.norm

        # normalization fudge factor
        fac *= np.sqrt(settings.cadence) / 2.
        
        ND = settings.NT * settings.NF

        wdm_window = settings.window # 
        
        # np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=0, arr=a)
        for i in range(settings.NT):
            for j in range(settings.N):
                n = i*settings.NF - int(settings.N/2) + j
                if n < 0:
                    n += ND  # periodically wrap the data
                if n >= ND:
                    n -= ND  # periodically wrap the data
                wdata[:, j] = self.arr[:, n] * wdm_window[j]  # apply the window
                
            tmp = np.fft.rfft(wdata, axis=-1)
            wave[:, i, 0] = tmp[:, 0]
            for j in range(settings.NF):  #(int j=1 j<wdm->NF j++)
                n = j*settings.oversample
                wave[:, i, j] = wdata[:, n].conj()
                # if((i+j)%2 ==0):
                #     wave[i][j] = wdata[2*n]
                # else:
                #     wave[i][j] = -wdata[2*n+1]

        wave *= fac
        return WDMSignal(wave, settings)

# static void wavelet_window_time(struct Wavelets *wdm)
# {
#     double *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
    
#     //zero frequency
#     REAL(DX,0) =  wdm->inv_root_dOmega
#     IMAG(DX,0) =  0.0
    
#     for(int i=1 i<= wdm->N/2 i++)
#     {
#         int j = wdm->N-i
#         double omega = (double)(i)*wdm->domega
        
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

# void wavelet_transform(struct Wavelets *wdm, double *data)
# {
#     //array index for tf pixel
#     int k
    
#     //total data size
#     int ND = wdm->NT*wdm->NF
    
#     //windowed data packets
#     double *wdata = double_vector(wdm->N)

#     //wavelet wavepacket transform of the signal
#     double **wave = double_matrix(wdm->NT,wdm->NF)
    
#     //normalization factor
#     double fac = M_SQRT2*sqrt(wdm->cadence)/wdm->norm
    
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

    @property
    def differential_component(self) -> float:
        return self.df

    @staticmethod
    def get_associated_class():
        return FDSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def kwargs(self) -> dict:
        return dict()
    
    @property
    def args(self) -> tuple:
        return (self.N, self.df)  
    
    @property
    def basis_shape(self) -> tuple:
        return (self.N,)
    
    @property
    def f_arr(self) -> np.ndarray:
        return np.arange(self.N) * self.df
    
    def __eq__(self, value):
        return (value.N == self.N) and (value.df == self.df)

    @property
    def total_terms(self) -> int:
        return self.N
    

class FDSignal(FDSettings, DomainBase):
    def __init__(self, arr, settings: FDSettings):
        FDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> FDSettings:
        return FDSettings(*self.args, **self.kwargs)

    @property
    def arr(self) -> np.ndarray:
        return self._arr
    
    @arr.setter
    def arr(self, arr: np.ndarray):
        self._arr = np.atleast_2d(arr)

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
    
    def wdmtransform(self, settings=None, window=None):
        if window is None:
            window = np.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        breakpoint()

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
            return self.wdmtransform(settings=new_domain, window=window)
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
        oversample: int = 8,
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
        self.inv_root_dOmega = 1.0/np.sqrt(self.dOmega)
        self.B = self.dOmega
        self.A = (self.dOmega-self.B)/2.0
        self.BW = (self.A+self.B)/np.pi

        self.N = self.oversample * 2 * self.NF
        if window is None:
            self.setup_window()
        else:
            assert len(window) == self.N
            self.window = window

    @property
    def basis_shape(self) -> tuple:
        return (self.NT, self.NF)
    
    @property
    def t_arr(self) -> np.ndarray:
        return np.arange(self.NT) * self.dt

    @property
    def f_arr(self) -> np.ndarray:
        return np.arange(self.NF) * self.df

    def phitilde(self, omega):
        insDOM = self.inv_root_dOmega
        A = self.A
        B = self.B
        
        z = 0.0
        
        if np.abs(omega) >= A and np.abs(omega) < A+B:
            x = (np.abs(omega)-A)/B
            y = special.betainc(WAVELET_FILTER_CONSTANT, WAVELET_FILTER_CONSTANT, x)
            z = insDOM*np.cos(y*np.pi/2.0)
        
        elif np.abs(omega) < A:
            z = insDOM
        
        return z

    def setup_window(self):

        # double *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
        DX = np.zeros(self.N, dtype=complex)
        
        # zero frequency
        # REAL(DX,0) =  wdm->inv_root_dOmega
        # IMAG(DX,0) =  0.0
        DX[0] = self.inv_root_dOmega
    
        for i in range(1, int(self.N / 2) + 1):  # (int i=1 i<= wdm->N/2 i++)
            j = self.N - i 
            omega = i*self.dOmega
            
            # // postive frequencies
            # REAL(DX,i) = phitilde(wdm,omega)
            # IMAG(DX,i) =  0.0
            DX[i] = self.phitilde(omega)
            # // negative frequencies
            # REAL(DX,j) =  phitilde(wdm,-omega)
            # IMAG(DX,j) =  0.0
            DX[j] = self.phitilde(-omega)
        
        dx_copy = DX.copy()
        DX[:] = np.fft.ifft(DX)

        window = np.zeros(self.N)
        for i in range(int(self.N / 2)):
            window[i] = DX[int(self.N/2)+i].real
            window[int(self.N/2) + i] = DX[i].real
    
        self.window = window
        self.norm = np.sqrt(self.N * self.cadence / self.dOmega)

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
