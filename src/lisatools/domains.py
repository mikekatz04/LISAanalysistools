from __future__ import annotations

import copy
import math
import warnings
from abc import ABC
from typing import Any, List, Optional, Tuple

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

    CUPY_AVAILABLE = False

import dataclasses

from .utils.constants import *
from .utils.utility import AET, get_array_module, tukey
from .utils.parallelbase import LISAToolsParallelModule


@dataclasses.dataclass
class DomainSettingsBase(LISAToolsParallelModule):
    force_backend: str = None

    def __init__(self, force_backend: str = None):
        self.force_backend = force_backend
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_slice(self, index: tuple) -> DomainSettingsBase:
        raise NotImplementedError(
            "get_slice needs to be implemented for this signal type."
        )


class DomainBase:

    def __init__(self, arr):
        self.arr = arr

    @property
    def arr(self) -> np.ndarray | cp.ndarray:
        return self._arr

    @arr.setter
    def arr(self, arr: np.ndarray | cp.ndarray):
        xp = get_array_module(arr)

        if CUPY_AVAILABLE and xp != np:
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

    def transform(
        self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None
    ):
        raise NotImplementedError(
            "Transform needs to be implemented for this signal type."
        )

    @property
    def shape(self) -> tuple:
        return self.arr.shape

    def get_array_slice(self, index: tuple) -> DomainBase:
        new_arr = self.arr[(Ellipsis,) + index]
        new_settings = self.settings.get_slice(index)
        return self.settings.associated_class(new_arr, new_settings)


class TDSettings(DomainSettingsBase):
    t0: float
    N: int
    dt: float

    def __init__(self, t0: float, N: int, dt: float, **kwargs):
        self.t0 = t0
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
        return (self.t0, self.N, self.dt)

    @property
    def t_arr(self) -> np.ndarray:
        return self.t0 + self.xp.arange(self.N) * self.dt

    @property
    def basis_shape(self) -> tuple:
        return (self.N,)

    def __repr__(self) -> str:
        return (
            f"TDSettings(t0={self.t0}, N={self.N}, dt={self.dt}, "
            f"force_backend={self.force_backend})"
        )

    def __eq__(self, value):
        if not isinstance(value, TDSettings):
            return False

        return (value.N == self.N) and (value.dt == self.dt) and (value.t0 == self.t0)

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

        return TDSettings(new_t0, new_N, self.dt, force_backend=self.force_backend)


class TDSignal(DomainBase, TDSettings):
    def __init__(self, arr, settings: TDSettings):
        TDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)


    @property
    def settings(self) -> TDSettings:
        return TDSettings(*self.args, **self.kwargs)

    def fft(self, settings=None, window=None):
        xp = get_array_module(self.arr)
        if window is None:
            window = xp.ones(self.arr.shape, dtype=float)

        df = 1 / (self.N * self.dt)

        fd_arr = xp.fft.rfft(self.arr * window) * self.dt

        if settings is not None:
            assert isinstance(settings, FDSettings)
            assert settings.df == df
            fd_settings = settings
        else:
            fd_settings = FDSettings(
                fd_arr.shape[-1],
                df,
                force_backend=self.force_backend,  
            )

        return FDSignal(fd_arr[..., fd_settings.active_slice], fd_settings)

    def stft(self, settings=None, window=None):

        if settings is None:
            raise ValueError("Must provide STFTSettings for stft transform.")
        assert isinstance(settings, STFTSettings)
        big_dt = settings.dt

        xp = get_array_module(self.arr)

        # Validate that big_dt is an integer multiple of self.dt
        nperseg = settings.get_nperseg(self.dt)

        if window is None:
            window = xp.ones(nperseg, dtype=float)

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
            _arr = xp.pad(self.arr, pad_width, mode="constant", constant_values=0)
        else:
            # Truncate to use only what we need
            _arr = self.arr[..., :required_samples]

        stft_arr = self.dt * xp.fft.rfft(
            window[None, :] * _arr.reshape(self.outer_shape + (Nsegments, nperseg)),
            axis=-1,
        )

        return STFTSignal(
            stft_arr[..., settings.active_slice], settings
        )  # (nchannels, NT, NF)

    def wdmtransform(self, settings=None, window=None):
        xp = get_array_module(self.arr)
        if window is None:
            window = xp.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # go to frequency domain then wavelets
        return self.fft(settings=None, window=window).transform(settings)

    def transform(
        self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None
    ):
        xp = get_array_module(self.arr)

        if isinstance(new_domain, TDSettings):
            if window is None:
                window = xp.ones(self.arr.shape, dtype=float)
            return self.settings.associated_class(self.arr * window, self.settings)

        elif isinstance(new_domain, FDSettings):
            return self.fft(settings=new_domain, window=window)

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

    def __init__(self, N: int, df: float, min_freq: Optional[float] = 0.0, max_freq: Optional[float] = None, **kwargs):
        self.N = N
        self.df = df
        self.min_freq = min_freq
        self.max_freq = max_freq
        super().__init__(**kwargs)

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
            f"force_backend={self.force_backend})"
        )

    def __eq__(self, value):
        if not isinstance(value, FDSettings):
            return False
        return (
            (value.N == self.N)
            and (value.df == self.df)
            and (value.min_freq == self.min_freq)
            and (value.max_freq == self.max_freq)
        )

    @property
    def total_terms(self) -> int:
        return self.N_active

    @property
    def active_slice(
        self,
    ) -> slice:
        if self.min_freq is not None and self.min_freq < 0:
            raise ValueError("min_freq must be non-negative.")
        if self.max_freq is not None and self.max_freq < 0:
            raise ValueError("max_freq must be non-negative.")
        if self.min_freq is None:
            start_idx = 0
        else:
            start_idx = int(np.ceil(self.min_freq / self.df))
        if self.max_freq is None:
            end_idx = self.N
        else:
            end_idx = int(np.floor(self.max_freq / self.df)) + 1

        return slice(start_idx, end_idx)

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

    @property
    def settings(self) -> FDSettings:
        return FDSettings(*self.args, **self.kwargs)

    def ifft(self, settings=None, window=None):
        xp = get_array_module(self.arr)
        if window is None:
            window = xp.ones(self.arr.shape, dtype=float)

        Tobs = 1 / self.df
        td_arr = xp.fft.irfft(self.arr * window)
        N = td_arr.shape[-1]
        dt = Tobs / N
        assert N == int(Tobs / dt)
        if settings is not None:
            assert isinstance(settings, TDSettings)
            assert settings.dt == dt

        td_settings = TDSettings(t0=0.0, N=N, dt=dt, force_backend=self.force_backend)
        return TDSignal(td_arr, td_settings)

    def get_fd_window_for_wdm(self, settings):

        xp = get_array_module(self.arr)

        N = self.settings.N

        # solve for window
        N = settings.NF + 1

        # mini wavelet structure for basis covering just N layers
        T = settings.dt * settings.NT
        domega = 2 * xp.pi / T

        window = xp.zeros(self.N, dtype=complex)

        # wdm window function
        for i in range(0, int(settings.NT / 2)):  # (i=0; i<=wdm->NT/2; i++)
            omega = i * domega
            window[i] = settings.phitilde(omega)

        raise NotImplementedError

        # normalize
        # for(i=-wdm->NT/2; i<= wdm->NT/2; i++) norm += window[abs(i)]*window[abs(i)];
        # norm = sqrt(norm/wdm_temp->cadence);

        # for(i=0; i<=wdm->NT/2; i++) window[i] /= norm;

        # free(wdm_temp);

    def wdmtransform(
        self, settings=None, window=None, return_transpose_time_axis_first: bool = False
    ):
        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # phif = phitilde_vec_norm(settings.NF, settings.NT, 4.0)
        xp = get_array_module(self.arr)

        # removed zero frequency and mirrored
        m = xp.repeat(xp.arange(0, settings.NF)[:, None], settings.NT, axis=-1)
        n = xp.tile(xp.arange(settings.NT), (settings.NF, 1))
        k = (m - 1) * int(settings.NT / 2) + xp.arange(settings.NT)[None, :]

        # removed zero frequency and mirrored
        k = settings.get_shift_map(m)

        base_window = settings.window[
            :-1
        ]  # TODO: compared to Tyson's code he does i=-N/2; i<N/2; i++
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

        after_ifft = xp.fft.ifft(tmp, axis=-1)

        is_m_plus_n_even = (m + n) % 2 == 0
        _new_arr = xp.zeros((self.nchannels, settings.NF, settings.NT), dtype=float)
        _new_arr[:, is_m_plus_n_even] = (
            xp.sqrt(2) * xp.real(after_ifft)[:, is_m_plus_n_even]
        )
        _new_arr[:, (~is_m_plus_n_even)] = (
            (-1) ** ((m * n)[(~is_m_plus_n_even)] + 1)
            * xp.sqrt(2)
            * xp.imag(after_ifft)[:, (~is_m_plus_n_even)]
        )

        # TODO: need to fix top and bottom layer
        _new_arr[:, np.array([0, -1])] = 0.0
        if return_transpose_time_axis_first:
            output = _new_arr.transpose(0, 2, 1).copy()
        else:
            output = _new_arr

        return WDMSignal(output, settings=settings)

    def transform(
        self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None
    ):
        xp = get_array_module(self.arr)
        if window is None:
            window = xp.ones(self.arr.shape, dtype=float)

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
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, force_backend={self.force_backend})"
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
        if value is not None:
            self._min_freq = np.ceil(value / self.df) * self.df
        else:
            self._min_freq = 0.0

    @property
    def max_freq(self) -> Optional[float]:
        return self._max_freq

    @max_freq.setter
    def max_freq(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_freq must be non-negative.")

        # self._max_freq = value
        # set it to the closest frequency bin
        if value is not None:
            self._max_freq = np.floor(value / self.df) * self.df
        else:
            self._max_freq = (self.NF - 1) * self.df

    @property
    def f_arr(self) -> np.ndarray:

        _all_freqs = self.xp.arange(0, self.NF) * self.df
        return _all_freqs[self.active_slice]

    @property
    def args(self) -> tuple:
        return (self.t0, self.dt, self.df, self.NT, self.NF)

    @property
    def kwargs(self) -> dict:
        return dict(min_freq=self.min_freq, max_freq=self.max_freq, force_backend=self.force_backend)

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
            and (value.t0 == self.t0)
        )

    @property
    def differential_component(self) -> float:
        return self.df

    @property
    def active_slice(
        self,
    ) -> slice:
        if self.min_freq is not None and self.min_freq < 0:
            raise ValueError("min_freq must be non-negative.")
        if self.max_freq is not None and self.max_freq < 0:
            raise ValueError("max_freq must be non-negative.")

        if self.min_freq is None:
            start_idx = 0
        else:
            start_idx = int(self.xp.ceil(self.min_freq / self.df))
        if self.max_freq is None:
            end_idx = self.NF
        else:
            end_idx = int(self.xp.floor(self.max_freq / self.df)) + 1

        return slice(start_idx, end_idx)

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
            force_backend=self.force_backend,
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

    @property
    def settings(self) -> STFTSettings:
        return STFTSettings(*self.args, **self.kwargs)
    
    def plot(self, channel=0, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        xp = get_array_module(self.arr)
        t_arr = self.t_arr if get_array_module(self.t_arr) == np else self.t_arr.get()
        f_arr = self.f_arr if get_array_module(self.f_arr) == np else self.f_arr.get()

        arr_here = self.arr[channel].get() if xp != np else self.arr[channel]
        cb = ax.pcolormesh(t_arr, f_arr, (np.abs(arr_here)**2).T, shading='auto', cmap='cividis', **kwargs)

        ax.set_yscale('log')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_ylim(self.min_freq, self.max_freq)
        plt.colorbar(cb, ax=ax, label='Magnitude')
        return ax


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
        oversample: int = 16,
        window: Optional[np.ndarray | cp.ndarray] = None,
        **kwargs,
    ):
        self.Tobs = Tobs
        self.NT = int(np.ceil(Tobs / WAVELET_DURATION).astype(int))
        self.NF = int(WAVELET_DURATION / dt)
        self.data_dt = dt

        self.df = WAVELET_BANDWIDTH
        self.dt = WAVELET_DURATION
        self.t0 = t0
        self.oversample = oversample

        self.cadence = WAVELET_DURATION / self.NF
        self.Omega = np.pi / self.cadence
        self.dOmega = self.Omega / self.NF
        self.domega = 2 * np.pi / self.Tobs
        self.inv_root_dOmega = 1.0 / np.sqrt(self.dOmega)
        self.B = self.Omega / (2 * self.NF)
        self.A = (self.dOmega - self.B) / 2.0

        self.BW = (self.A + self.B) / np.pi

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

        super().__init__(**kwargs)

    @property
    def basis_shape(self) -> tuple:
        return (self.NF, self.NT)

    @property
    def t_arr(self, xp=np) -> np.ndarray | cp.ndarray:
        return self.t0 + xp.arange(self.NT) * self.dt

    @property
    def f_arr(self, xp=np) -> np.ndarray | cp.ndarray:
        return xp.arange(self.NF) * self.df

    @property
    def f_arr_edges(self) -> np.ndarray:
        return np.arange(self.NF + 1) * self.df

    @property
    def t_arr_edges(self) -> np.ndarray:
        return np.arange(self.NT + 1) * self.dt

    def phitilde(self, omega, xp=np):
        insDOM = self.inv_root_dOmega
        A = self.A
        B = self.B

        z = xp.zeros(omega.shape[0])
        beta_inc_calc = (xp.abs(omega) >= A) & (xp.abs(omega) <= A + B)
        x = (xp.abs(omega[beta_inc_calc]) - A) / B
        y = special.betainc(WAVELET_FILTER_CONSTANT, WAVELET_FILTER_CONSTANT, x)
        z[beta_inc_calc] = insDOM * xp.cos(y * xp.pi / 2.0)
        z[(xp.abs(omega) < A)] = insDOM

        return z

    def wavelet(self, N: int, in_fd: Optional[bool] = True) -> np.ndarray:

        # NT * NF is even
        # assert (self.NT * self.NF) % 2 == 0
        base_window = self.window[:-1]
        omega_N = (np.arange(self.N) - int(self.N / 2)) * self.domega
        wavelet_N = 1 / np.sqrt(2.0) * self.phitilde(omega_N)

        if in_fd:
            return wavelet_N
        else:
            return np.fft.ifft(wavelet_N) / self.norm
        breakpoint()
        wavelets_rfft = np.zeros((len(m), int((self.NT * self.NF) / 2 + 1)))

        np.put_along_axis(wavelets_rfft, k, base_window * 1 / np.sqrt(2.0), axis=-1)
        freq = np.fft.fftshift(np.fft.fftfreq(self.NT * self.NF, self.data_dt))
        wavelets_fft = np.exp(
            -1j * 2 * np.pi * freq[None, :] * n[:, None] * self.dt
        ) * np.concatenate(
            [wavelets_rfft[:, ::-1][:, :-1], wavelets_rfft[:, :-1]], axis=-1
        )
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
        return (m_in - 1) * int(self.NT / 2) + self.xp.arange(self.NT)[None, :]

    def setup_window(self, xp=np):

        # *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
        # zero frequency
        # REAL(DX,0) =  wdm->inv_root_dOmega
        # IMAG(DX,0) =  0.0
        T = self.dt * self.NT
        domega = 2 * xp.pi / T
        self.omega = omega = (xp.arange(self.NT + 1) - int(self.NT / 2)) * domega
        window = self.phitilde(omega)
        self.norm = xp.sqrt(self.N * self.cadence / self.dOmega)
        self.window = window / self.norm
        assert 0.0 in omega

        self.ind_middle = xp.argwhere(omega == 0.0).squeeze().item()

        omega_for_edge_layers = xp.concatenate(
            [
                omega[self.ind_middle :],
                domega
                * (self.ind_middle + xp.arange(1, omega[: self.ind_middle].shape[0])),
            ]
        )
        assert (xp.diff(omega_for_edge_layers).min() > 0.0) and xp.allclose(
            xp.diff(omega_for_edge_layers).max(), domega
        )
        self.dc_layer_window = xp.sqrt(2) * self.phitilde(omega_for_edge_layers)
        self.max_freq_layer_window = self.dc_layer_window[::-1]

    def __repr__(self) -> str:
        return (
            f"WDMSettings(Tobs={self.Tobs}, dt={self.data_dt}, t0={self.t0}, "
            f"NT={self.NT}, NF={self.NF}, oversample={self.oversample}, "
            f"force_backend={self.force_backend})"
        )

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
        N = int(
            (total_pixels / 2 + 1)
            if total_pixels % 2 == 0
            else ((total_pixels + 1) / 2)
        )
        check_settings = FDSettings(
            N,
            df,
        )

        if settings is not None:
            if check_settings != settings:
                breakpoint()
                raise ValueError(
                    "Entered FD settings do not correspond to valid transform. Better to leave them blank if possible."
                )
        else:
            settings = check_settings

        # Perform the inverse transform
        new_arr = np.zeros((self.nchannels, settings.N), dtype=complex)
        tmp_arr = self.arr.get() if hasattr(self.arr, "get") else self.arr
        for i in range(self.nchannels):
            new_arr[i] = inverse_wavelet_freq_helper(tmp_arr[i], phif, self.NF, self.NT)

        return FDSignal(new_arr, settings)

    def transform(
        self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None
    ):
        if window is None:
            window = xp.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, TDSettings):
            return self.wdm_to_fd(settings=None, window=None).ifft(
                settings=new_domain, window=window
            )

        elif isinstance(new_domain, FDSettings):
            return self.wdm_to_fd(settings=new_domain, window=window)

        elif isinstance(new_domain, STFTSettings):
            return (
                self.wdm_to_fd(settings=None, window=None)
                .ifft(settings=None, window=None)
                .stft(settings=new_domain, window=window)
            )

        elif isinstance(new_domain, WDMSettings):
            if new_domain == self.settings:
                return self
            else:
                return self.wdm_to_fd(settings=None, window=None).wdmtransform(
                    settings=new_domain, window=window
                )
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

        for i, (ax_i, channel) in enumerate(zip(ax, ["X", "Y", "Z"])):
            z = self.arr[i]
            x, y = self.t_arr_edges, self.f_arr_edges
            sc = ax_i.pcolormesh(
                x,
                y,
                z,
                # extent=[self.t_arr.min(), self.t_arr.max(), self.f_arr.min(), self.f_arr.max()],
                **kwargs,
            )
            ax_i.set_ylabel(channel)

        cax = fig.add_axes([0.9, 0.2, 0.05, 0.6])
        fig.colorbar(sc, cax=cax)

        plt.subplots_adjust(right=0.85, hspace=0.1)
        return fig, ax


class WDMLookupTable(WDMSettings):
    def __init__(
        self, settings: WDMSettings, f_steps: int, fdot_steps: int, num_channel: int
    ):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        d_fdot = self.d_fdot = 0.1
        fdot_step = self.df / self.T * d_fdot
        self.f_steps = f_steps
        self.fdot_steps = fdot_steps
        self.deltaf = self.BW / (self.f_steps)
        self.num_channel = num_channel

        # The odd wavelets coefficienst can be obtained from the even.
        # odd cosine = -even sine, odd sine = even cosine
        # each wavelet covers a frequency band of width DW
        # execept for the first and last wasvelets
        # there is some overlap. The wavelet pixels are of width
        # DOM/PI, except for the first and last which have width
        # half that
        ref_layer = int(self.NF / 2)

        f0 = ref_layer * self.df
        self.fdot_vals = (np.arange(fdot_steps) - int(fdot_steps / 2)) * fdot_step

        wave = self.wavelet(self.N, in_fd=False)

        self.f_scaled_vals = (
            (np.arange(self.f_steps) - self.f_steps / 2) + 0.5
        ) * self.deltaf
        self.f_vals = f0 + self.f_scaled_vals
        self.min_f_scaled = self.f_scaled_vals.min().item()
        self.max_f_scaled = self.f_scaled_vals.max().item()
        self.min_fdot = self.fdot_vals.min().item()
        self.max_fdot = self.fdot_vals.max().item()
        t = (np.arange(self.N) - int(self.N / 2)) * self.cadence
        phase = (
            2 * np.pi * self.f_vals[:, None, None] * t[None, None, :]
            + np.pi * self.fdot_vals[None, :, None] * (t * t)[None, None, :]
        )

        real_coeff = np.sum(
            wave * np.cos(phase) * self.cadence, axis=-1
        )  # TODO: trapz?
        imag_coeff = np.sum(wave * np.sin(phase) * self.cadence, axis=-1)
        self.table = real_coeff + 1j * imag_coeff

    @property
    def table(self) -> np.ndarray:
        return self._table

    @table.setter
    def table(self, table: np.ndarray):
        self._table = table
        points = np.asarray(
            [tmp.ravel() for tmp in np.meshgrid(self.f_vals, self.fdot_vals)]
        ).T
        self._interpolant = interpolate.LinearNDInterpolator(
            points, table.flatten(), rescale=True
        )

    def get_table_coeffs(self, f_arr: np.ndarray, fdot_arr: np.ndarray):
        assert np.all((f_arr > self.f_vals.min()) & (f_arr < self.f_vals.max()))
        assert np.all(
            (fdot_arr > self.fdot_vals.min()) & (fdot_arr < self.fdot_vals.max())
        )
        return self._interpolant(np.array([f_arr, fdot_arr]).T)


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
                target_settings.associated_class(
                    transformed_batched.arr[i], target_settings
                )
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

#         tmp_arr = xp.asarray([tmp_mat.sens_mat for tmp_mat in sens_mats])

#         SensitivityMatrix.__init__(self, settings.f_arr, tmp_arr)
