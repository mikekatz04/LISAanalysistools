from __future__ import annotations

import copy
import math
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
import numpy as np
from matplotlib import cm
from scipy import interpolate, signal, special

try:
    import cupy as cp
    import cupyx.scipy.signal as cupyx_signal
    from cupyx.scipy import special as cupy_special

    CUPY_AVAILABLE = True

except (ModuleNotFoundError, ImportError):
    import numpy as cp  # type: ignore

from . import detector as lisa_models
from .utils.utility import AET, get_array_module
from .utils.constants import *
from .utils.parallelbase import LISAToolsParallelModule
import dataclasses


class DomainSettingsBase(LISAToolsParallelModule):
    force_backend: str = None
    def __init__(self, force_backend: str = None):
        self.force_backend = force_backend


@dataclasses.dataclass
class DomainSettingsBase(LISAToolsParallelModule):
    force_backend: str = None

    def __init__(self, force_backend: str = None):
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_slice(self, index: tuple) -> DomainSettingsBase:
        raise NotImplementedError("get_slice needs to be implemented for this signal type.")


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
    def arr(self) -> np.ndarray | cp.ndarray:
        return self._arr

    @arr.setter
    def arr(self, arr: np.ndarray | cp.ndarray):
        
        if self.backend.uses_cupy:
            self._stft = cupyx_signal.stft
        else:
            self._stft = signal.stft

        assert len(arr.shape) >= len(self.basis_shape)
        if len(arr.shape) == len(self.basis_shape):
            arr = arr[None, ...]

        self.outer_shape = arr.shape[: -len(self.basis_shape)]
        if len(self.outer_shape) > 2:
            raise ValueError(
                f"Too many dimensions outside of basis_shape. "
                f"Expected at most 2 outer dims (batch, channels), got {len(self.outer_shape)}: {self.outer_shape}."
            )
        elif len(self.outer_shape) == 2:
            # batched: shape is (nbatch, nchannels, *basis_shape)
            self._nbatch = self.outer_shape[0]
            self.nchannels = self.outer_shape[1]
        else:
            # unbatched: shape is (nchannels, *basis_shape)
            self._nbatch = None
            self.nchannels = self.outer_shape[0]
        self._arr = arr

    def __getitem__(self, index):
        return self.arr[index]

    def __setitem__(self, index, value):
        self.arr[index] = value

    @property
    def is_batched(self) -> bool:
        """Whether this signal has a batch dimension."""
        return self._nbatch is not None

    @property
    def nbatch(self) -> int | None:
        """Number of batch elements, or None if unbatched."""
        return self._nbatch

    def flatten(self) -> np.ndarray | cp.ndarray:
        return self.arr.flatten()

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
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

    def get_array_slice(self, index: tuple) -> DomainBase:
        new_arr = self.arr[(Ellipsis,) + index]
        new_settings = self.settings.get_slice(index)
        return self.settings.associated_class(new_arr, new_settings)


class TDSettings(DomainSettingsBase):
    N: int
    dt: float
    t0: float = 0.0

    def __init__(self, N: int, dt: float, t0: float = 0.0, **kwargs):
        self.t0 = t0
        self.N = N
        self.dt = dt
        # TODO: include kwargs in kwargs property?
        super().__init__(**kwargs)

    @staticmethod
    def get_associated_class():
        return TDSignal

    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        return dict(t0=self.t0, force_backend=self.force_backend)
    
    @property
    def args(self) -> tuple:
        return (self.N, self.dt)

    @property
    def t_arr(self) -> np.ndarray:
        return self.t0 + self.xp.arange(self.N) * self.dt

    @property
    def basis_shape(self) -> tuple:
        return (self.N,)

    def __repr__(self) -> str:
        return (
            f"TDSettings(t0={self.t0}, N={self.N}, dt={self.dt}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

    def __eq__(self, value):
        if not isinstance(value, TDSettings):
            return False

        return (
            (value.N == self.N) and (value.dt == self.dt) and (self.xp.isclose(value.t0, self.t0))
        )

    @property
    def differential_component(self) -> float:
        return self.dt

    @property
    def total_terms(self) -> int:
        return self.N

    def compute_slice_indices(self, tmin: float, tmax: float) -> slice:
        if tmin < self.t0:
            raise ValueError("tmin must be greater than or equal to t0.")
        if tmax > self.t0 + self.N * self.dt:
            raise ValueError("tmax must be less than or equal to t0 + N*dt.")

        start_idx = int(self.xp.round((tmin - self.t0) / self.dt))
        end_idx = int(self.xp.round((tmax - self.t0) / self.dt))

        return slice(start_idx, end_idx)

    def get_slice(self, index=None, tmin: float = None, tmax: float = None) -> TDSettings:
        """
        Return a new TDSettings object corresponding to the slice of the time points specified by index.

        Args:
            index: A slice object for the time dimension, e.g. slice(0, 10). If provided, this will be used to compute the new t0 and N for the sliced settings.
            tmin: Minimum time value for the slice. If provided, this will be used to compute the new t0 and N for the sliced settings.
            tmax: Maximum time value for the slice. If provided, this will be used to compute the new t0 and N for the sliced settings.

        If both index and (tmin, tmax) are provided, they must be consistent with each other (i.e. the time range specified by index must match the time range specified by tmin and tmax).

        Returns:
            A new TDSettings object corresponding to the slice of the time points specified by index or (tmin, tmax).
        """

        if index is None:
            if tmin is None or tmax is None:
                raise ValueError("If index is not provided, both tmin and tmax must be provided.")
            index = self.compute_slice_indices(tmin, tmax)

        if not isinstance(index, slice):
            raise TypeError("index must be a slice object")

        start, stop, step = index.indices(self.N)
        new_N = (stop - start) // step
        new_t0 = self.t0 + start * self.dt

        return TDSettings(new_t0, new_N, self.dt, force_backend=self.backend)


class TDSignal(DomainBase, TDSettings):
    def __init__(self, arr, settings: TDSettings):
        TDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> TDSettings:
        return TDSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"TDSignal(t0={self.t0}, N={self.N}, dt={self.dt}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )
    
    def fft(self, settings=None, window=None, apply_dt=True):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        df = 1 / (self.N * self.dt)

        factor = 1.0 if not apply_dt else self.dt
        fd_arr = self.xp.fft.rfft(self.arr * window) * factor
        if settings is not None:
            assert isinstance(settings, FDSettings)
            assert settings.df == df, f"Provided FDSettings has df={settings.df}, but expected df={df} based on TDSettings."
            assert settings.N == fd_arr.shape[-1]
            fd_settings = settings
        
        else:
            fd_settings = FDSettings(fd_arr.shape[-1], df, force_backend=self.backend)
        
        fd_arr_in = fd_arr[..., fd_settings.ind_min:fd_settings.ind_max + 1]
        return FDSignal(fd_arr_in, fd_settings)

    def stft(self, settings=None, window=None):

        if settings is None:
            raise ValueError("Must provide STFTSettings for stft transform.")
        assert isinstance(settings, STFTSettings)
        big_dt = settings.dt

        # Validate that big_dt is an integer multiple of self.dt
        nperseg = settings.get_nperseg(self.dt)

        if window is None:
            window =self.xp.ones(nperseg, dtype=float)

        # Use NT from settings directly to ensure consistency
        Nsegments = settings.NT
        Nsegments_available = self.N // nperseg

        # Check we have enough data
        required_samples = Nsegments * nperseg

        if self.N < required_samples:
            raise ValueError(
                f"Not enough data: have {self.N} samples, need {required_samples} for {Nsegments} segments"
            )

        if Nsegments > Nsegments_available:
            # Need to pad
            pad_samples = required_samples - self.N

            # Pad with zeros at the end
            # pad_width format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
            pad_width = [(0, 0)] * len(self.outer_shape) + [(0, pad_samples)]
            _arr =self.xp.pad(self.arr, pad_width, mode="constant", constant_values=0)
        else:
            # Truncate to use only what we need
            _arr = self.arr[..., :required_samples]

        stft_arr = self.dt *self.xp.fft.rfft(
            window[None, :] * _arr.reshape(self.outer_shape + (Nsegments, nperseg)),
            axis=-1,
        )

        return STFTSignal(stft_arr[..., settings.active_slice], settings)  # (nchannels, NT, NF)

    def wdmtransform(self, settings=None, window=None):
        if window is None:
            window =self.xp.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # go to frequency domain then wavelets
        return self.fft(settings=None, window=window, apply_dt=False).transform(settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, TDSettings):
            if window is None:
                window = self.xp.ones(self.arr.shape, dtype=float)
            return self.settings.associated_class(self.arr * window, self.settings)

        elif isinstance(new_domain, FDSettings):
            return self.fft(settings=new_domain, window=window, apply_dt=True)
        
        elif isinstance(new_domain, STFTSettings):

            return self.stft(settings=new_domain, window=window)

        elif isinstance(new_domain, WDMSettings):
            return self.wdmtransform(settings=new_domain, window=window)
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")


class FDSettings(DomainSettingsBase):
    N: int
    df: float
    min_freq: Optional[float] = 0.0
    max_freq: Optional[float] = None

    def __init__(
        self,
        N: int,
        df: float,
        min_freq: Optional[float] = 0.0,
        max_freq: Optional[float] = None,
        **kwargs,
    ):
        self.N = N
        self.df = df
        self.min_freq = min_freq
        self.max_freq = max_freq
        super().__init__(**kwargs)

    @property
    def differential_component(self) -> float:
        return self.df
   
    @property
    def min_freq(self) -> float:
        return self._min_freq

    @min_freq.setter
    def min_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_freq must be non-negative.")

        # self._min_freq = value
        # set it to the closest frequency bin
        self.min_freq_input = value
        if value is not None:
            self.ind_min = int(np.ceil(value / self.df))
        else:
            self.ind_min = 0
        self._min_freq = self.ind_min * self.df

    @property
    def max_freq(self) -> Optional[float]:
        return self._max_freq

    @max_freq.setter
    def max_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_freq must be non-negative.")

        # self._max_freq = value
        # set it to the closest frequency bin
        self.max_freq_input = value
        if value is not None:
            self.ind_max = int(value / self.df)
        else:
            self.ind_max = (self.N - 1)
        self._max_freq = self.ind_max * self.df

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

    @staticmethod
    def get_associated_class():
        return FDSignal

    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        return dict(
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            force_backend=self.force_backend,
        )

    @property
    def args(self) -> tuple:
        return (self.N, self.df)

    @property
    def basis_shape(self) -> tuple:
        return (self.N_active,)

    @property
    def f_arr(self) -> np.ndarray:
        _all_freqs = self.xp.arange(0, self.N) * self.df

        return _all_freqs[self.active_slice]

    def __repr__(self) -> str:
        return (
            f"FDSettings(N={self.N}, df={self.df}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

    def __eq__(self, value):
        if not isinstance(value, FDSettings):
            return False
        return (
            (value.N == self.N)
            and (value.df == self.df)
            and (self.xp.isclose(value.min_freq, self.min_freq))
            and (self.xp.isclose(value.max_freq, self.max_freq))
        )

    @property
    def total_terms(self) -> int:
        return self.N_active

    @property
    def active_slice(
        self,
    ) -> slice:
        return slice(self.ind_min, self.ind_max + 1)

    @property
    def N_active(self) -> int:
        sl = self.active_slice
        return sl.stop - sl.start


# try:
#     from pywavelet.transforms.phi_computer import phitilde_vec_norm
#     from pywavelet.transforms.numpy.forward.from_freq import (
#         transform_wavelet_freq_helper
#     )

# from pywavelet.transforms.numpy.inverse.to_freq import (
#     inverse_wavelet_freq_helper_fast as inverse_wavelet_freq_helper,
# )


class FDSignal(FDSettings, DomainBase):
    def __init__(self, arr, settings: FDSettings):
        FDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

        if self.arr.shape[-1] != self.N_active:
            assert arr.shape[-1] == self.N
            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[:, self.active_slice]
            # self.arr = 

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
    
    def __repr__(self) -> str:
        return (
            f"FDSignal(N={self.N}, df={self.df}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

        td_settings = TDSettings(N, dt, t0=0.0, force_backend=self.force_backend)
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

        # for(i=0; i<=wdm->NT/2; i++) window[i] /= norm;

        # free(wdm_temp);

    def wdmtransform(
        self, settings=None, window=None, return_transpose_time_axis_first: bool = False, is_psd: bool = False
    ):
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
        base_window = (settings.window[:] * 2 / settings.Nf)
        dc_window = (settings.dc_layer_window * 2 / settings.Nf)
        # TODO: check if this is right?!?!
        max_freq_window = (settings.max_freq_layer_window * 2 / settings.Nf)
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
            before_ifft[:, 1:-1, 1:] *= (base_window[None, None, :] * settings.Nf / 2.)  # remove factor of Nf over 2
            before_ifft[:, 0, 1:] *= (dc_window * settings.Nf / 2.)  # remove factor of Nf over 2
            before_ifft[:, -1, 1:] *= (max_freq_window * settings.Nf / 2.)  # remove factor of Nf over 2
            psd_sum_tmp = before_ifft.sum(axis=-1) / (settings.data_dt * settings.Nt * settings.Nf)
            wdmpsd = self.xp.zeros((self.nchannels, settings.Nf, settings.Nt), dtype=complex)

            wdmpsd[:, 1:] = psd_sum_tmp[:, 1:settings.Nf, None]          # regular layers
            wdmpsd[:, 0, 0::2] = psd_sum_tmp[:, 0, None]           # DC at even rows
            wdmpsd[:, 0, 1::2] = psd_sum_tmp[:, settings.Nf, None]     
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
            output = _new_arr.transpose(0, 2, 1).copy()
        else:
            output = _new_arr

        return WDMSignal(output, settings=settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
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

    def plot(self, 
             channel: int = 0, 
             ax: plt.Axes | None = None, 
             filename: Optional[str] = None,
             **kwargs) -> plt.Axes:
        """
        Plot the squared amplitude of the FD signal for a given channel.

        Args:
            channel: The channel index to visualize.
            ax: An optional matplotlib Axes object to plot on. If None, a new figure and axes will be created.  
            filename: An optional filename to save the plot to. If provided, the plot will be saved to this file.
            **kwargs: Additional keyword arguments to pass to the underlying plotting functions.
        
        Returns:
            The matplotlib Axes object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        f_arr = self.f_arr if get_array_module(self.f_arr) == np else self.f_arr.get()
        arr_here = self.arr[channel].get() if self.backend.uses_cupy else self.arr[channel]

        ax.loglog(f_arr, np.abs(arr_here) ** 2, **kwargs)

        ax.set_title(f"Frequency Spectrum for Channel {channel}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(self.min_freq, self.max_freq)
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        return ax
        
class STFTSettings(DomainSettingsBase):
    t0: float
    dt: float
    df: float
    NT: int
    NF: int
    min_freq: Optional[float] = 0.0
    max_freq: Optional[float] = None

    def __init__(
        self,
        t0: float,
        dt: float,
        df: float,
        NT: int,
        NF: int,
        min_freq: Optional[float] = 0.0,
        max_freq: Optional[float] = None,
        **kwargs,
    ):
        self.t0 = t0
        self.dt = dt
        self.df = df
        self.NT = NT
        self.NF = NF
        self.min_freq = min_freq
        self.max_freq = max_freq
        super().__init__(**kwargs)

    def __repr__(self):
        return (
            f"STFTSettings(t0={self.t0}, dt={self.dt}, df={self.df}, NT={self.NT}, NF={self.NF}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, backend={self.backend_name.split('_')[-1]})"
        )

    @staticmethod
    def get_associated_class():
        return STFTSignal

    @property
    def associated_class(self):
        return self.get_associated_class()

    @property
    def basis_shape(self) -> tuple:
        return (
            self.NT,
            self.NF_active,
        )  #! in the STFT domain, the basis shape is (# number of times segments, # number of frequencies)

    @property
    def total_terms(self) -> int:
        return self.NT * self.NF_active

    @property
    def t_arr(self) -> np.ndarray:
        return self.t0 + self.xp.arange(self.NT) * self.dt
   
    @property
    def min_freq(self) -> float:
        return self._min_freq

    @min_freq.setter
    def min_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_freq must be non-negative.")

        # self._min_freq = value
        # set it to the closest frequency bin
        self.min_freq_input = value
        if value is not None:
            self.ind_min = int(np.ceil(value / self.df))
        else:
            self.ind_min = 0
        self._min_freq = self.ind_min * self.df

    @property
    def max_freq(self) -> Optional[float]:
        return self._max_freq

    @max_freq.setter
    def max_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_freq must be non-negative.")

        # self._max_freq = value
        # set it to the closest frequency bin
        self.max_freq_input = value
        if value is not None:
            self.ind_max = int(value / self.df)
        else:
            self.ind_max = (self.NF - 1)
        self._max_freq = self.ind_max * self.df

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
    def f_arr(self) -> np.ndarray:

        _all_freqs = self.xp.arange(0, self.NF) * self.df
        return _all_freqs[self.active_slice]

    @property
    def args(self) -> tuple:
        return (self.t0, self.dt, self.df, self.NT, self.NF)

    @property
    def kwargs(self) -> dict:
        return dict(
            min_freq=self.min_freq, max_freq=self.max_freq, force_backend=self.backend
        )

    @property
    def f_arr_edges(self) -> np.ndarray:
        return self.xp.arange(self.NF + 1) * self.df

    @property
    def t_arr_edges(self) -> np.ndarray:
        return self.xp.arange(self.NT + 1) * self.dt

    def __eq__(self, value):
        if not isinstance(value, STFTSettings):
            return False
        return (
            (value.NT == self.NT)
            and (value.NF == self.NF)
            and (value.dt == self.dt)
            and (value.df == self.df)
            and (self.xp.isclose(value.t0, self.t0))
            and (self.xp.isclose(value.min_freq, self.min_freq))
            and (self.xp.isclose(value.max_freq, self.max_freq))
        )

    @property
    def differential_component(self) -> float:
        return self.df

    @property
    def active_slice(
        self,
    ) -> slice:
        return slice(self.ind_min, self.ind_max + 1)

    @property
    def NF_active(self) -> int:
        sl = self.active_slice
        return sl.stop - sl.start

    def get_nperseg(self, small_dt: float):

        nperseg = round(self.dt / small_dt)

        assert (
            abs(nperseg * small_dt - self.dt) < 1e-10 * self.dt
        ), f"big_dt={self.dt} must be an integer multiple of dt={small_dt}"

        return nperseg

    def compute_slice_indices(
        self, tmin: float, tmax: float, fmin: float, fmax: float
    ) -> Tuple[slice, slice]:
        """
        Compute the slice indices for the time and frequency dimensions based on the provided min and max values.

        Args:
            tmin: Minimum time value for the slice.
            tmax: Maximum time value for the slice.
            fmin: Minimum frequency value for the slice.
            fmax: Maximum frequency value for the slice.

        Returns:
            A tuple of slices for the time and frequency dimensions, e.g. (slice(0, 10), slice(5, 15)).
        """

        if tmin < self.t0:
            raise ValueError("tmin must be greater than or equal to t0.")
        if tmax > self.t0 + self.NT * self.dt:
            raise ValueError("tmax must be less than or equal to t0 + NT*dt.")
        f_min_active = self.f_arr[0]
        f_max_active = self.f_arr[-1]
        if fmin < f_min_active:
            raise ValueError(
                f"fmin ({fmin}) must be >= the active minimum frequency ({f_min_active})."
            )
        if fmax > f_max_active:
            raise ValueError(
                f"fmax ({fmax}) must be <= the active maximum frequency ({f_max_active})."
            )

        time_start_idx = int(self.xp.round((tmin - self.t0) / self.dt))
        time_end_idx = int(self.xp.round((tmax - self.t0) / self.dt))
        freq_start_idx = int(self.xp.round((fmin - self.f_arr[0]) / self.df))
        freq_end_idx = int(self.xp.floor((fmax - self.f_arr[0]) / self.df)) + 1

        return slice(time_start_idx, time_end_idx), slice(freq_start_idx, freq_end_idx)

    def get_slice(self, index: tuple) -> STFTSettings:
        """
        Return a new STFTSettings object corresponding to the slice of the time and frequency points specified by index.

        Args:
            index: A tuple of slices for the time and frequency dimensions, e.g. (slice(0, 10), slice(5, 15)).

        Returns:
            STFTSettings: A new STFTSettings object corresponding to the slice.
        """
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError(
                "Index must be a tuple of two slices for time and frequency dimensions."
            )

        time_slice, freq_slice = index

        new_t0 = self.t0 + time_slice.start * self.dt
        new_NT = time_slice.stop - time_slice.start
        new_NF = self.NF

        new_min_freq = float(self.f_arr[freq_slice.start])
        new_max_freq = float(self.f_arr[freq_slice.stop - 1])

        return STFTSettings(
            t0=new_t0,
            dt=self.dt,
            df=self.df,
            NT=new_NT,
            NF=new_NF,
            min_freq=new_min_freq,
            max_freq=new_max_freq,
            force_backend=self.backend,
        )


def get_stft_settings(
    times: np.ndarray | cp.ndarray,
    big_dt: float,
    min_freq: Optional[float] = 0.0,
    max_freq: Optional[float] = None,
    **kwargs,
) -> STFTSettings:
    """
    Get STFT settings from time array and desired big_dt.

    Args:
        times: Time array.
        big_dt: Desired time resolution for STFT segments.
        min_freq: Minimum frequency to consider.
        max_freq: Maximum frequency to consider.
        **kwargs: Additional keyword arguments to pass to STFTSettings.

    Returns:
        STFTSettings: The settings for the STFT.
    """

    t0 = float(times[0])
    N = len(times)
    dt = float(times[1] - times[0])

    big_dt = int(big_dt / dt) * dt  # make sure big_dt is an integer multiple of dt
    NT = int(np.floor(N / (big_dt / dt)))
    DF = 1 / big_dt
    nperseg = int(big_dt / dt)
    NF = nperseg // 2 + 1

    return STFTSettings(
        t0=t0, dt=big_dt, df=DF, NT=NT, NF=NF, min_freq=min_freq, max_freq=max_freq, **kwargs
    )


class STFTSignal(STFTSettings, DomainBase):
    def __init__(self, arr, settings: STFTSettings):
        STFTSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

        # freq layers
        if self.arr.shape[-2] != self.N_active:
            assert arr.shape[-2] == self.NF
            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[:, self.active_slice]

    @property
    def settings(self) -> STFTSettings:
        return STFTSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"STFTSignal(t0={self.t0}, dt={self.dt}, df={self.df}, NT={self.NT}, NF={self.NF}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, backend={self.backend_name.split('_')[-1]})"
        )

    def _plot_stft(self, channel=0, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t_arr = self.t_arr if get_array_module(self.t_arr) == np else self.t_arr.get()
        f_arr = self.f_arr if get_array_module(self.f_arr) == np else self.f_arr.get()

        arr_here = self.arr[channel].get() if self.backend.uses_cupy else self.arr[channel]
        cb = ax.pcolormesh(
            t_arr, f_arr, (np.abs(arr_here) ** 2).T, shading="auto", cmap="cividis", **kwargs
        )

        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_ylim(self.min_freq, self.max_freq)
        plt.colorbar(cb, ax=ax, label="Magnitude")
        return ax

    def _plot_fd(self, channel=0, ax=None, time_bin=0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        f_arr = self.f_arr if get_array_module(self.f_arr) == np else self.f_arr.get()
        arr_here = self.arr[channel].get() if self.backend.uses_cupy else self.arr[channel]

        ax.loglog(f_arr, np.abs(arr_here[time_bin]) ** 2, **kwargs)

        ax.set_title(f"STFT Frequency Spectrum for Time Bin {time_bin} (Time = {self.t_arr[time_bin]:.2f})")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(self.min_freq, self.max_freq)
        return ax
    
    def _plot_td(self, channel=0, ax=None, freq_bin=0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t_arr = self.t_arr if get_array_module(self.t_arr) == np else self.t_arr.get()
        arr_here = self.arr[channel].get() if self.backend.uses_cupy else self.arr[channel]

        ax.plot(t_arr, arr_here[:, freq_bin].real, label="real part", **kwargs)
        ax.plot(t_arr, arr_here[:, freq_bin].imag, label="imag part", **kwargs)

        ax.legend()
        ax.set_title(f"STFT Time Series for Frequency Bin {freq_bin} (Frequency = {self.f_arr[freq_bin]:.2f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Magnitude")
        return ax

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
    
    def plot(self, 
             channel: int = 0, 
             ax: plt.Axes | None = None, 
             plot_type: str = "stft", 
             filename: Optional[str] = None,
             **kwargs) -> plt.Axes:
        """
        Visualize the STFT signal in either the time-frequency domain (stft), frequency domain (fd), or time domain (td).

        Args:
            channel: The channel index to visualize.
            ax: An optional matplotlib Axes object to plot on. If None, a new figure and axes will be created.
            plot_type: The type of plot to create. Must be one of 'stft', 'fd', or 'td'. 'stft' will create a time
                vs frequency plot of the magnitude squared of the STFT coefficients. 'fd' will create a log-log plot of the magnitude squared of the STFT coefficients for a single time bin. 'td' will create a plot of the magnitude squared of the real and imaginary parts of the STFT coefficients for a single frequency bin.
            filename: An optional filename to save the plot to. If provided, the plot will be saved to this file.
            **kwargs: Additional keyword arguments to pass to the underlying plotting functions.
        
        Returns:
            The matplotlib Axes object containing the plot.
        """
        if plot_type == "stft":
            ax = self._plot_stft(channel=channel, ax=ax, **kwargs)
        elif plot_type == "fd":
            ax = self._plot_fd(channel=channel, ax=ax, **kwargs)
        elif plot_type == "td":
            ax = self._plot_td(channel=channel, ax=ax, **kwargs)
        else:
            raise ValueError(f"Invalid plot_type {plot_type}. Must be one of 'stft', 'fd', or 'td'.")

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")

        return ax


WAVELET_BANDWIDTH = 6.51041666666667e-5
WAVELET_DURATION = 7680.0
WAVELET_FILTER_CONSTANT = 6


class WDMSettings(DomainSettingsBase):
    def __init__(
        self,
        Nf: float, 
        Nt: float,
        dt: float,
        t0: float = 0.0,
        oversample: int = 16,
        window: Optional[np.ndarray] = None,
        dc_layer_window: Optional[np.ndarray] = None,
        max_freq_layer_window: Optional[np.ndarray] = None,
        norm: Optional[float] = None, 
        omega: Optional[np.ndarray] = None,
        min_freq: Optional[int] = None,
        max_freq: Optional[int] = None,
        **kwargs
    ):
        DomainSettingsBase.__init__(self, **kwargs)
        self.Nt = Nt
        self.Nf = Nf
        self.data_dt = dt
        self.N = self.Nt * self.Nf
        self.data_dt = dt
        self.Tobs = self.N * self.data_dt
        self.layer_dt = self.Nf * self.data_dt
        self.layer_df = 1. / (2. * self.Nf * self.data_dt)

        # these have to come after layer_df b/c setters
        # sets ind_min and ind_max
        self.min_freq = min_freq
        self.max_freq = max_freq
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
    
    @property
    def active_slice(
        self,
    ) -> slice:
        return slice(self.ind_min, self.ind_max + 1)
    
    @property
    def Nf_active(self) -> int:
        sl = self.active_slice
        return sl.stop - sl.start

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
    def min_freq(self) -> float:
        return self._min_freq

    @min_freq.setter
    def min_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_freq must be non-negative.")

        # self._min_freq = value
        # set it to the closest frequency bin
        self.min_freq_input = value
        if value is not None:
            self.ind_min = int(np.ceil(value / self.layer_df))
        else:
            self.ind_min = 0
        self._min_freq = self.ind_min * self.layer_df

    @property
    def max_freq(self) -> Optional[float]:
        return self._max_freq

    @max_freq.setter
    def max_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_freq must be non-negative.")

        # self._max_freq = value
        # set it to the closest frequency bin
        self.max_freq_input = value
        if value is not None:
            self.ind_max = int(value / self.layer_df)
        else:
            self.ind_max = (self.N - 1)
        self._max_freq = self.ind_max * self.layer_df

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
    def frequency_layer_mask(self) -> Optional[np.ndarray]:
        mask = self.xp.zeros(self.Nf, dtype=bool)
        mask[self.active_slice] = True
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
            min_freq=self.min_freq, 
            max_freq=self.max_freq, 
            force_backend=self.force_backend
        )

    @property
    def args(self) -> tuple:
        return (self.Nf, self.Nt, self.data_dt)   
    
    @property
    def differential_component(self) -> float:
        return 1.0

    @property
    def total_terms(self) -> int:
        return self.Nt * self.Nf
    
    def apply_frequency_layer_mask(self, arr: np.ndarray) -> np.ndarray:
        if self.frequency_layer_mask is None or arr.shape[-2] == self.Nf_active:
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

        # freq layers
        if self.arr.shape[-2] != self.Nf_active:
            assert arr.shape[-2] == self.Nf
            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[:, self.active_slice]

    @property
    def settings(self) -> WDMSettings:
        return WDMSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"WDMSignal(Tobs={self.Tobs}, dt={self.data_dt}, t0={self.t0}, "
            f"NT={self.NT}, NF={self.NF}, oversample={self.oversample}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

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

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
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
                return self.wdm_to_fd(settings=None, window=None).wdmtransform(
                    settings=new_domain, window=window
                )
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")

    def heatmap(self, index: int = None, mag: bool = False, fig=None, ax=None, cax=None, add_cax=False, **kwargs):
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
                x, y = self.t_arr_edges, self.f_arr_edges[self.ind_min:self.ind_max + 2]
                x = self.get(x)
                y = self.get(y)
                z = self.get(z)
                if mag:
                    z = np.abs(z)
                sc = ax_i.pcolormesh(
                    x, y, z, 
                    # extent=[self.t_arr.min(), self.t_arr.max(), self.f_arr.min(), self.f_arr.max()], 
                    **kwargs
                )
                ax_i.set_ylabel(channel)

        else:
            assert index is not None and fig is not None and ax is not None
            z = self.arr[index]
            x, y = self.t_arr_edges, self.f_arr_edges[self.ind_min:self.ind_max + 2]
            x = self.get(x)
            y = self.get(y)
            z = self.get(z)
            if mag:
                z = np.abs(z)
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

import h5py

class WDMLookupTable(WDMSettings):
   
    def to_file(self, fp: str):
        if os.path.exists(fp):
            raise ValueError("Trying to write to file that exists.")
        
        with h5py.File(fp, "w") as fp:
            g = fp.create_group("wdm")

            g.attrs["Nf"] = self.Nf
            g.attrs["Nt"] = self.Nt
            g.attrs["Nt_generate"] = self.sub_settings.Nt
            g.attrs["data_dt"] = self.data_dt
            g.attrs["max_freq"] = self.ind_min
            g.attrs["min_freq"] = self.ind_max
            g.attrs["m_ref"] = self.m_ref
            g.attrs["n_ref"] = self.n_ref
            g.attrs["nchannels"] = self.nchannels
            g.create_dataset("table_sin", data=self.get(self.table_sin))
            g.create_dataset("table_cos", data=self.get(self.table_cos))
            g.create_dataset("fdot_vals", data=self.get(self.fdot_vals))
            g.create_dataset("norm_freq_single_layer", data=self.get(self.norm_freq_single_layer))
            g.create_dataset("m_diffs", data=self.get(self.m_diffs))

    @staticmethod
    def from_file(fp: str, force_backend: Optional[str] = None):
        with h5py.File(fp, "r") as f:
            g = f["wdm"]

            if "ind_min" in g.attrs.keys():
                # backwards compatibility for now
                input_kwargs = dict(
                    ind_min = g.attrs["ind_min"],
                    ind_max = g.attrs["ind_max"],
                    force_backend=force_backend,
                )

                ind_min = input_kwargs.pop("ind_min")
                ind_max = input_kwargs.pop("ind_max")

                input_args = (
                    g.attrs["Nf"],
                    g.attrs["Nt"],
                    g.attrs["data_dt"] 
                )
                nchannels = g.attrs["nchannels"]
                _settings = WDMSettings(*input_args, **input_kwargs)
                
                max_freq = _settings.layer_df * ind_max
                min_freq = _settings.layer_df * ind_min

                input_kwargs["min_freq"] = min_freq
                input_kwargs["max_freq"] = max_freq

            settings = WDMSettings(*input_args, **input_kwargs)
            return WDMLookupTable(settings, nchannels, store_path=fp)

    def from_file_internal(self, fp: str):
        with h5py.File(fp, "r") as fp:
            g = fp["wdm"]
            self.sub_settings = WDMSettings(g.attrs["Nf"], g.attrs["Nt_generate"], g.attrs["data_dt"])
            self.fdot_vals = self.xp.asarray(g["fdot_vals"][:])

            self.m_ref = g.attrs["m_ref"]
            self.n_ref = int(self.sub_settings.Nt / 2)  # g.attrs["n_ref"]

            self.nchannels = g.attrs["nchannels"]
            self.norm_freq_single_layer = self.xp.asarray(g["norm_freq_single_layer"][:])
            self.m_diffs = self.xp.asarray(g["m_diffs"][:])
            self.table_sin = self.xp.asarray(g["table_sin"][:])
            self.table_cos = self.xp.asarray(g["table_cos"][:])
            
    @staticmethod
    def apply_eps_fdot(eps: float, settings: WDMSettings, fdot_max_factor: float= 8.0) -> np.ndarray:
        delta_fdot = eps * settings.layer_df / settings.layer_dt
        fdot_max_val = fdot_max_factor * settings.layer_df / settings.layer_dt
        _fdot = np.arange(0.0, fdot_max_val, delta_fdot)
        fdot_vals = np.concatenate([-_fdot[::-1][:-1], _fdot])
        return fdot_vals
            
    @staticmethod 
    def apply_eps_frequency(eps: float, settings: WDMSettings, m_ref: Optional[int] = None, num_layers_diff: Optional[int] = 2) -> tuple:
        delta_f = eps * settings.layer_df

        if m_ref is None:
            m_ref = int(settings.Nt / 2)

        norm_freq_single_layer = np.arange(0.0, settings.layer_df, delta_f)
        m_diffs = (_tmp := np.arange(2 * num_layers_diff + 2)) - _tmp[int(len(_tmp) / 2)]

        return norm_freq_single_layer, m_diffs, m_ref

    def __init__(self, settings: WDMSettings, nchannels: int, m_ref: int = None, norm_freq_single_layer: np.ndarray = None, m_diffs: np.ndarray = None, fdot_vals: np.ndarray = None, store_path: Optional[str] = None, batch_size_gen: Optional[int] = 20, time_layers: Optional[int] = None, td_window: Optional[np.ndarray] = None):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        # TODO: CHECK FIRST AND LAST TIME LAYERS DUE TO TIME WINDOWING?

        self.nchannels = nchannels
        
        self.store_path = store_path
        if os.path.exists(self.store_path):
            self.from_file_internal(self.store_path)
        else:
            if time_layers is None:
                time_layers = self.Nt

            assert isinstance(time_layers, int)
            
            self.sub_settings = WDMSettings(self.Nf, time_layers, self.data_dt, force_backend=self.force_backend)
            self.m_ref = m_ref  # int(3e-3 / self.sub_settings.layer_df)  # int(self.sub_settings.Nf / 2)
            self.n_ref = int(self.sub_settings.Nt / 2)
            self.is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
            self.m_diffs = self.xp.asarray(m_diffs).astype(self.xp.int32)
            self.fdot_vals = self.xp.asarray(fdot_vals)
            self.norm_freq_single_layer = self.xp.asarray(norm_freq_single_layer)
            
            total_f_fdot_vals = self.norm_f_steps * self.fdot_steps
            if self.run_fdot:
                _f_vals, _fdot_vals = self.xp.asarray([tmp.ravel() for tmp in self.xp.meshgrid(self.norm_freq_single_layer + self.f_ref, self.fdot_vals)])
            else:
                _f_vals = self.norm_freq_single_layer.copy() + self.f_ref
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
            
            _table_sin = self.xp.zeros((len(m_diffs), total_f_fdot_vals,))
            _table_cos = self.xp.zeros((len(m_diffs), total_f_fdot_vals,))
            
            self.td_window = td_window
            for st_batch, end_batch in zip(batches[:-1], batches[1:]):
                inds = np.arange(st_batch, end_batch)
                
                # if not self.xp.allclose(_f_vals[inds] - (self.f_ref + 0 * self.layer_df), 0.0) or not _fdot_vals[inds][0] == 0.0:  # -3.257427471431767e-11:
                #     continue
                wave_sin = self.xp.sin(2 * np.pi * (_f_vals[inds, None] * t_diff[None, :] + 1. / 2. * _fdot_vals[inds, None] * t_diff[None, :] ** 2))
                wave_cos = self.xp.cos(2 * np.pi * (_f_vals[inds, None] * t_diff[None, :] + 1. / 2. * _fdot_vals[inds, None] * t_diff[None, :] ** 2))
                
                wave_sin_wdm = TDSignal(wave_sin, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)
                wave_cos_wdm = TDSignal(wave_cos, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)

                for m_i, m_diff in enumerate(m_diffs):
                    m_current = self.m_ref + m_diff
                    sin_coeff = wave_sin_wdm[:, self.m_ref - m_diff, self.n_ref] 
                    cos_coeff = wave_cos_wdm[:, self.m_ref - m_diff, self.n_ref] 

                    _f_norm = _f_vals[inds] - (self.m_ref - m_diff) * self.layer_df
                    if (m_current + self.n_ref) % 2 == 1:
                        _tmp = sin_coeff
                        sin_coeff = cos_coeff
                        cos_coeff = _tmp

                    try:
                        _table_sin[m_i, inds] = sin_coeff
                        _table_cos[m_i, inds] = cos_coeff
                    except:
                        breakpoint()
                print(inds, total_f_fdot_vals)

            # TODO: verify if there is a minus sign needed here and below
            freqs =  (_f_vals.reshape(-1, 2) +  self.xp.asarray(m_diffs)[:, None, None] * self.layer_df).transpose(1, 0, 2).reshape(self.fdot_steps, -1)
            _table_sin = _table_sin.reshape(len(m_diffs), self.fdot_steps, self.norm_f_steps).transpose(1, 0, 2).reshape(self.fdot_steps, self.f_steps).copy()
            _table_cos = _table_cos.reshape(len(m_diffs), self.fdot_steps, self.norm_f_steps).transpose(1, 0, 2).reshape(self.fdot_steps, self.f_steps).copy()

            assert _table_sin.shape == (self.fdot_vals.shape[0], self.f_vals.shape[0])
            assert _table_cos.shape == (self.fdot_vals.shape[0], self.f_vals.shape[0])
            
            self.table_sin = _table_sin
            self.table_cos = _table_cos

            # freqs = _f_vals[None, :] - m_diffs[:, None] * self.layer_df
            # freqs_norm = self.f_ref - freqs
            if store_path is not None:
                self.to_file(store_path)

    @property
    def run_fdot(self) -> bool:
        return not (len(self.fdot_vals) == 1 and self.fdot_vals[0] == 0.0)
    
    @property 
    def f_vals_norm(self) -> np.ndarray:
        return self.f_vals - self.f_ref
    
    @property 
    def f_steps(self) -> np.ndarray:
        return len(self.f_vals)
    
    @property 
    def norm_f_steps(self) -> np.ndarray:
        return len(self.norm_freq_single_layer)
    
    @property 
    def fdot_steps(self) -> np.ndarray:
        return len(self.fdot_vals)
    
    @property 
    def t_ref(self) -> np.ndarray:
        return self.n_ref * self.layer_dt
    
    @property 
    def f_ref(self) -> np.ndarray:
        return self.m_ref * self.layer_df
    
    @property
    def f_vals(self) -> np.ndarray:
        freq_single_layer = self.norm_freq_single_layer + self.f_ref 
        f_vals = self.xp.concatenate([freq_single_layer + m_tmp * self.layer_df for m_tmp in self.m_diffs])
        assert self.xp.allclose(_tmp := self.xp.diff(f_vals), _tmp[0])
        return f_vals
        
    
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
        
    def get_table_coeffs(self, f_norm: np.ndarray, fdot_arr: np.ndarray):
        # ms = (f_arr // self.layer_df).astype(int)

        if self.run_fdot:
            sin_coeffs = self.table_sin_interpolate(f_norm, fdot_arr)
            cos_coeffs = self.table_cos_interpolate(f_norm, fdot_arr)
            #breakpoint()
        else:
            sin_coeffs = self.table_sin_interpolate(f_norm)
            cos_coeffs = self.table_cos_interpolate(f_norm)

        sin_coeffs[np.isnan(sin_coeffs)] = 0.0
        cos_coeffs[np.isnan(cos_coeffs)] = 0.0
        return (sin_coeffs, cos_coeffs)

    def get_wdm_coeffs(self, amp_arr: np.ndarray, phi_arr: np.ndarray, f_arr: np.ndarray, fdot_arr: np.ndarray, n_arr: np.ndarray, num_m_layers: int = 1):
        ms = (f_arr / self.layer_df).astype(int)
        wdm_coeffs_out = self.xp.zeros((amp_arr.shape[0], num_m_layers * 2 + 1))
        m_map = -self.xp.ones((amp_arr.shape[0], num_m_layers * 2 + 1), dtype=int)
        is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        for i, m_diff in enumerate(range(-num_m_layers, num_m_layers + 1)):
            ms_to_use = (ms + m_diff).astype(int)
            keep_now = self.xp.arange(ms_to_use.shape[0])[(ms_to_use >= 0) & (ms_to_use <= self.Nf + 1)]
            
            assert ms[keep_now].max() <= self.Nf + 1
            assert ms[keep_now].min() >= 0
            try:
                assert self.xp.all((f_arr[keep_now] >= 0.0) & (f_arr[keep_now] <= self.f_arr.max()))
            except AssertionError:
                breakpoint()
            assert self.xp.all((fdot_arr[keep_now] >= self.fdot_vals.min()) & (fdot_arr[keep_now] <= self.fdot_vals.max()))
            f_norm = (f_arr[keep_now] - ms_to_use[keep_now] * self.layer_df)
            m_diff = (f_norm / self.layer_df).astype(int)

            _sin_coeffs, _cos_coeffs = self.get_table_coeffs(f_norm, fdot_arr[keep_now])
            is_m_plus_n_even = (((ms_to_use[keep_now] + n_arr[keep_now]) % 2 == 0)) 

            sin_coeffs = self.xp.zeros_like(_sin_coeffs)
            cos_coeffs = self.xp.zeros_like(_cos_coeffs)
            
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

            # TODO: idk if this is right NEED TO CHECK
            wdm_coeffs_out[keep_now, i] = amp_arr[keep_now] * (sin_coeffs * self.xp.sin(phi_arr[keep_now]) + cos_coeffs * self.xp.cos(phi_arr[keep_now]))
            m_map[keep_now, i] = ms_to_use[keep_now]
        return wdm_coeffs_out, m_map
    

class DomainBaseArray:
    """Container for a collection of :class:`DomainBase` objects.

    When all signals share identical settings (uniform case), the signals are
    stacked into a single batched :class:`DomainBase`, enabling vectorized
    domain transforms (e.g. a single batched FFT instead of N sequential ones).
    Otherwise the class falls back to per-element processing.

    Args:
        signals: List of :class:`DomainBase` objects.

    """

    def __init__(self, signals: List[DomainBase]) -> None:
        if not all(isinstance(s, DomainBase) for s in signals):
            raise TypeError("All elements of DomainBaseArray must be DomainBase instances.")
        self.signals = list(signals)

        if len(signals) > 1:
            s0 = signals[0].settings
            self.is_uniform = all(s.settings == s0 for s in signals[1:])
        else:
            self.is_uniform = True

        if self.is_uniform and len(signals) > 0:
            xp = get_array_module(signals[0].arr)
            arr_stacked = xp.stack([s.arr for s in signals], axis=0)
            settings = signals[0].settings
            self._batched = settings.associated_class(arr_stacked, settings)
        else:
            self._batched = None

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    def __getitem__(self, index):
        return self.signals[index]

    @property
    def batched(self) -> Optional[DomainBase]:
        """Batched :class:`DomainBase` (shape ``(nbatch, nchannels, *basis_shape)``),
        or ``None`` when settings are non-uniform."""
        return self._batched

    @property
    def settings(self) -> List[DomainSettingsBase]:
        """List of settings for each signal."""
        return [s.settings for s in self.signals]

    def transform(
        self,
        target_settings: DomainSettingsBase,
        window: Optional[np.ndarray] = None,
    ) -> "DomainBaseArray":
        """Transform all signals to *target_settings*.

        Uses a single vectorized call when all signals share the same settings
        (uniform case); otherwise transforms each signal individually.

        Args:
            target_settings: Target domain settings.
            window: Optional window to apply during the transform.

        Returns:
            :class:`DomainBaseArray` of transformed signals.

        """
        if self.is_uniform and self._batched is not None:
            transformed_batched = self._batched.transform(target_settings, window=window)
            # unstack along the batch axis
            transformed_signals = [
                target_settings.associated_class(transformed_batched.arr[i], target_settings)
                for i in range(len(self.signals))
            ]
            return DomainBaseArray(transformed_signals)
        else:
            return DomainBaseArray(
                [s.transform(target_settings, window=window) for s in self.signals]
            )

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

#         tmp_arr = xp.asarray([tmp_mat.sens_mat for tmp_mat in sens_mats])

#         SensitivityMatrix.__init__(self, settings.f_arr, tmp_arr)
