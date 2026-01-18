from __future__ import annotations
import warnings
from abc import ABC
from typing import Any, Tuple, Optional, List

import math
import numpy as np
from scipy import interpolate
from scipy import special as scipy_special
from scipy import signal
import matplotlib.pyplot as plt

try:
    import cupy as cp
    import cupyx.scipy.signal as cupyx_signal
    from cupyx.scipy import special as cupy_special
    CUPY_AVAILABLE = True

except (ModuleNotFoundError, ImportError):
    import numpy as cp  # type: ignore
    CUPY_AVAILABLE = False
    
from . import detector as lisa_models
from .utils.utility import AET, get_array_module, tukey
from .utils.constants import *

import dataclasses

@dataclasses.dataclass
class DomainSettingsBase:
    pass
class DomainBase:

    def __init__(self, arr):
        self.arr = arr

    @property
    def arr(self) -> np.ndarray | cp.ndarray:
        return self._arr
    
    @arr.setter
    def arr(self, arr: np.ndarray | cp.ndarray):
        self.xp = get_array_module(arr)
        assert len(arr.shape) >= len(self.basis_shape)
        if len(arr.shape) == len(self.basis_shape):
            arr = arr[None, ...]

        self.outer_shape = arr.shape[:-len(self.basis_shape)]
        if len(self.outer_shape) > 1:
            #raise ValueError("Too many dimensions outside of basis_shape.")
            # allow batching. In this case, the shape is (nbatch, nchannels, basis_shape...)
            self.nchannels = self.outer_shape[1]
        else:
            self.nchannels = self.outer_shape[0]
        self._arr = arr

    @property
    def xp(self):
        return self._xp
    
    @xp.setter
    def xp(self, xp):
        self._xp = xp
        if CUPY_AVAILABLE and xp != np:
            self._stft = cupyx_signal.stft
        else:
            self._stft = signal.stft

    def __getitem__(self, index):
        return self.arr[index]
    def __setitem__(self, index, value):
        self.arr[index] = value

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        raise NotImplementedError("Transform needs to be implemented for this signal type.")

    @property
    def shape(self) -> tuple:
        return self.arr.shape
    
@dataclasses.dataclass
class TDSettings(DomainSettingsBase):
    t0: float
    N: int
    dt: float
    xp: Any = np


    @staticmethod
    def get_associated_class():
        return TDSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        return dict(xp=self.xp)
    
    @property
    def args(self) -> tuple:
        return (self.t0, self.N, self.dt)   
    
    @property
    def t_arr(self) -> np.ndarray | cp.ndarray:
        return self.t0 + self.xp.arange(self.N) * self.dt
    
    @property
    def basis_shape(self) -> tuple:
        return (self.N,)
    
    def __eq__(self, value):
        return (value.N == self.N) and (value.dt == self.dt) and (value.t0 == self.t0)
    
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
            window = self.xp.ones(self.arr.shape, dtype=float)

        df = 1 / (self.N * self.dt)
        
        if settings is not None:
            assert isinstance(settings, FDSettings)
            assert settings.df == df

        fd_arr = self.xp.fft.rfft(self.arr * window) * self.dt
        fd_settings = FDSettings(fd_arr.shape[-1], df, xp=self.xp)
        return FDSignal(fd_arr, fd_settings)

    def stft(self, settings=None, window=None):

        if settings is None:
            raise ValueError("Must provide STFTSettings for stft transform.")
        assert isinstance(settings, STFTSettings)
        big_dt = settings.dt

        # Validate that big_dt is an integer multiple of self.dt
        nperseg = round(big_dt / self.dt)
        assert abs(nperseg * self.dt - big_dt) < 1e-10 * big_dt, \
            f"big_dt={big_dt} must be an integer multiple of dt={self.dt}"

        if window is None:
            window = self.xp.ones(nperseg, dtype=float)

        # Use NT from settings directly to ensure consistency
        Nsegments = settings.NT

        # Check we have enough data
        required_samples = Nsegments * nperseg
        if self.N < required_samples:
            raise ValueError(f"Not enough data: have {self.N} samples, need {required_samples} for {Nsegments} segments")

        _arr = self.arr[..., :Nsegments * nperseg]

        stft_arr = self.dt * self.xp.fft.rfft(
            window[None, :] * _arr.reshape(self.outer_shape + (Nsegments, nperseg)),
            axis=-1
        )

        return STFTSignal(stft_arr, settings)  # (nchannels, NT, NF)
    
    def wdmtransform(self, settings=None, window=None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # go to frequency domain then wavelets
        return self.fft(settings=None, window=window).transform(settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

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
    xp: Any = np


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
        return dict(xp=self.xp)
    
    @property
    def args(self) -> tuple:
        return (self.N, self.df)  
    
    @property
    def basis_shape(self) -> tuple:
        return (self.N,)
    
    @property
    def f_arr(self) -> np.ndarray | cp.ndarray:
        return self.xp.arange(self.N) * self.df
    
    def __eq__(self, value):
        return (value.N == self.N) and (value.df == self.df)

    @property
    def total_terms(self) -> int:
        return self.N
    

from pywavelet.transforms.phi_computer import phitilde_vec_norm
from pywavelet.transforms.numpy.forward.from_freq import (
    transform_wavelet_freq_helper
)

from pywavelet.transforms.numpy.inverse.to_freq import (
    inverse_wavelet_freq_helper_fast as inverse_wavelet_freq_helper,
)

class FDSignal(FDSettings, DomainBase):
    def __init__(self, arr, settings: FDSettings):
        FDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> FDSettings:
        return FDSettings(*self.args, **self.kwargs)

    def ifft(self, settings=None, window=None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        Tobs = 1 / self.df
        td_arr = self.xp.fft.irfft(self.arr * window)
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
        domega = 2 * self.xp.pi / T

        window = self.xp.zeros(self.N, dtype=complex)

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
        

    def wdmtransform(self, settings=None, window=None):
        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # phif = phitilde_vec_norm(settings.NF, settings.NT, 4.0)


        # removed zero frequency and mirrored
        m = self.xp.repeat(self.xp.arange(0, settings.NF)[:, None], settings.NT, axis=-1)
        n = self.xp.tile(self.xp.arange(settings.NT), (settings.NF, 1))
        k = (m - 1) * int(settings.NT / 2) + self.xp.arange(settings.NT)[None, :]
        
        
        base_window = settings.window[:-1]  # TODO: compared to Tyson's code he does i=-N/2; i<N/2; i++
        dc_window = settings.dc_layer_window
        # TODO: check if this is right?!?!
        max_freq_window = settings.max_freq_layer_window

        k[0] += int(settings.NT / 2)
        # k[-1] -= int(settings.NT / 2)
        # it is 2 because the max frequency would be at 1, but it removes that (?)
        assert k.max().item() == self.N - 2
        tmp = self.arr[:, k]
        
        tmp[:, 1:-1] *= base_window[None, None, :]
        tmp[0] *= dc_window
        tmp[-1] *= max_freq_window

        after_ifft = self.xp.fft.ifft(tmp, axis=-1)
        
        is_m_plus_n_even = ((m + n) % 2 == 0)
        _new_arr = self.xp.zeros((self.nchannels, settings.NF, settings.NT), dtype=float)
        _new_arr[:, is_m_plus_n_even] = self.xp.sqrt(2) * self.xp.real(after_ifft)[:, is_m_plus_n_even]
        _new_arr[:, (~is_m_plus_n_even)] = (-1) ** ((m * n)[(~is_m_plus_n_even)] + 1) * self.xp.sqrt(2) * self.xp.imag(after_ifft)[:, (~is_m_plus_n_even)]

        return WDMSignal(_new_arr.transpose(0, 2, 1).copy(), settings=settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

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
    t0: float
    dt: float
    df: float 
    NT: int
    NF: int
    xp: Any = np
    
    @staticmethod
    def get_associated_class():
        return STFTSignal
    
    @property
    def associated_class(self):
        return self.get_associated_class()
    
    @property
    def basis_shape(self) -> tuple:
        return (self.NT, self.NF)
    
    @property
    def total_terms(self) -> int:
        return self.NT * self.NF
    
    @property
    def t_arr(self) -> np.ndarray | cp.ndarray:
        return self.t0 + self.xp.arange(self.NT) * self.dt

    @property
    def f_arr(self) -> np.ndarray | cp.ndarray:
        return self.xp.arange(self.NF) * self.df
    
    @property
    def args(self) -> tuple:
        return (self.t0, self.dt, self.df, self.NT, self.NF)
    
    @property
    def kwargs(self) -> dict:
        return dict(xp=self.xp)
    
    def __eq__(self, value):
        return (value.NT == self.NT) and (value.NF == self.NF) and (value.dt == self.dt) and (value.df == self.df) and (value.t0 == self.t0)

class STFTSignal(STFTSettings, DomainBase):
    def __init__(self, arr, settings: STFTSettings):
        STFTSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> STFTSettings:
        return STFTSettings(*self.args, **self.kwargs)
    
    @property
    def differential_component(self) -> float:
        return self.df


WAVELET_BANDWIDTH = 6.51041666666667e-5
WAVELET_DURATION = 7680.0
WAVELET_FILTER_CONSTANT = 6

class WDMSettings(DomainSettingsBase):
    xp: Any = np

    def __init__(
        self,
        Tobs: float, 
        dt: float,
        t0: float = 0.0,
        oversample: int = 8,
        window: Optional[np.ndarray | cp.ndarray] = None,
    ):
        self.Tobs = Tobs
        self.NT = int(np.ceil(Tobs/WAVELET_DURATION).astype(int))
        self.NF = int(WAVELET_DURATION/dt)
        self.data_dt = dt

        self.df = WAVELET_BANDWIDTH
        self.dt = WAVELET_DURATION
        self.t0 = t0
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
            assert len(window) == self.NT + 1
            self.window = window

    @property
    def special(self):
        if self.xp is np:
            return scipy_special
        else:
            return cupy_special

    @property
    def basis_shape(self) -> tuple:
        return (self.NT, self.NF)
    
    @property
    def t_arr(self) -> np.ndarray | cp.ndarray:
        return self.t0 + self.xp.arange(self.NT) * self.dt

    @property
    def f_arr(self) -> np.ndarray | cp.ndarray:
        return self.xp.arange(self.NF) * self.df

    def phitilde(self, omega):
        insDOM = self.inv_root_dOmega
        A = self.A
        B = self.B
        
        z = self.xp.zeros(omega.shape[0])
        beta_inc_calc = (self.xp.abs(omega) >= A) & (self.xp.abs(omega) <= A+B)
        x = (self.xp.abs(omega[beta_inc_calc])-A)/B
        y = self.special.betainc(WAVELET_FILTER_CONSTANT, WAVELET_FILTER_CONSTANT, x)
        z[beta_inc_calc] = insDOM*self.xp.cos(y*self.xp.pi/2.0)
        z[(self.xp.abs(omega) < A)] = insDOM
        
        return z

    def setup_window(self):

        # double *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
        # zero frequency
        # REAL(DX,0) =  wdm->inv_root_dOmega
        # IMAG(DX,0) =  0.0
        T = self.dt * self.NT
        domega = 2 * self.xp.pi / T
        self.omega = omega = (self.xp.arange(self.NT + 1) - int(self.NT / 2)) * domega
        window = self.phitilde(omega)
        self.norm = self.xp.sqrt(self.N * self.cadence / self.dOmega)
        self.window = window / self.norm
        assert 0.0 in omega

        self.ind_middle = self.xp.argwhere(omega == 0.0).squeeze().item()

        omega_for_edge_layers = self.xp.concatenate([omega[self.ind_middle:], domega * (self.ind_middle + self.xp.arange(1, omega[:self.ind_middle].shape[0]))])
        assert (self.xp.diff(omega_for_edge_layers).min() > 0.0) and self.xp.allclose(self.xp.diff(omega_for_edge_layers).max(), domega)
        self.dc_layer_window = self.xp.sqrt(2) * self.phitilde(omega_for_edge_layers)
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
            oversample=self.oversample, window=self.window, xp=self.xp
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
        
        phif = phitilde_vec_norm(self.NF, self.NT, 4.0)

        # determine FD parameters
        total_pixels = self.NT * self.NF
        Tobs = total_pixels * self.data_dt
        df = 1 / Tobs
        N = int((total_pixels / 2 + 1) if total_pixels % 2 == 0 else ((total_pixels + 1) / 2))
        check_settings = FDSettings(N, df, xp=self.xp)
        
        if settings is not None:
            if check_settings != settings:
                breakpoint()
                raise ValueError("Entered FD settings do not correspond to valid transform. Better to leave them blank if possible.")
        else:
            settings = check_settings
            
        # Perform the inverse transform
        new_arr = np.zeros((self.nchannels, settings.N), dtype=complex)
        tmp_arr = self.arr.get() if hasattr(self.arr, "get") else self.arr
        for i in range(self.nchannels):
            new_arr[i] = inverse_wavelet_freq_helper(tmp_arr[i], phif, self.NF, self.NT)

        return FDSignal(new_arr, settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

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

#         tmp_arr = self.xp.asarray([tmp_mat.sens_mat for tmp_mat in sens_mats])
        
#         SensitivityMatrix.__init__(self, settings.f_arr, tmp_arr)
