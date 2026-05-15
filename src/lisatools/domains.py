"""Time, frequency, and time-frequency domain settings and array wrappers.

Defines :class:`DomainSettingsBase` (with its concrete subclasses :class:`TDSettings`,
:class:`FDSettings`, :class:`STFTSettings`, :class:`WDMSettings`) and the matching
:class:`DomainBase` array wrappers used throughout the package to keep arrays
tagged with their basis information so that transforms (FFT/iFFT, STFT, WDM)
can be performed automatically.
"""

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


# TODO/DOCS: this stub appears to be a forward declaration that the
# ``@dataclass``-decorated definition below shadows; verify whether it's still
# needed.
class DomainSettingsBase(LISAToolsParallelModule):
    force_backend: str = None
    def __init__(self, force_backend: str = None):
        self.force_backend = force_backend


@dataclasses.dataclass
class DomainSettingsBase(LISAToolsParallelModule):
    """Base class for domain settings (TD, FD, STFT, WDM, ...).

    Carries the ``force_backend`` selector that decides whether NumPy or CuPy
    is used for arrays in this domain. Subclasses must implement
    :meth:`get_slice` and define an ``associated_class`` mapping the settings
    to a concrete :class:`DomainBase` subclass.

    Args:
        force_backend: Backend name passed to
            :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`
            (e.g. ``"cpu"``, ``"cuda12x"``).
    """

    force_backend: str = None

    def __init__(self, force_backend: str = None):
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @classmethod
    def supported_backends(cls):
        """Return the list of backend names this settings class supports."""
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_slice(self, index: tuple) -> DomainSettingsBase:
        """Return a new settings object describing a sliced view of this domain."""
        raise NotImplementedError("get_slice needs to be implemented for this signal type.")


class DomainBase:
    """Base wrapper for an array tagged with its domain settings.

    The array's last ``len(basis_shape_active)`` dimensions correspond to the
    domain's basis (e.g. frequency bins, time bins, time-frequency cells); any
    leading dimensions are interpreted as ``(nbatch?, nchannels)``.

    Args:
        arr: The underlying NumPy or CuPy array.
    """

    def __init__(self, arr):
        self.arr = arr

    @staticmethod
    def get(x: np.ndarray) -> np.ndarray:
        """Return ``x`` as a NumPy array (calls ``.get()`` for CuPy arrays)."""
        try:
            return x.get()
        except AttributeError:
            return x

    @property
    def arr(self) -> np.ndarray | cp.ndarray:
        """Underlying NumPy or CuPy array."""
        return self._arr

    @arr.setter
    def arr(self, arr: np.ndarray | cp.ndarray):
        """Set the underlying array and infer batch / channel dimensions."""
        if self.backend.uses_cupy:
            self._stft = cupyx_signal.stft
        else:
            self._stft = signal.stft

        assert len(arr.shape) >= len(self.basis_shape_active)
        if len(arr.shape) == len(self.basis_shape_active):
            arr = arr[None, ...]

        self.outer_shape = arr.shape[: -len(self.basis_shape_active)]
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
        """Return :attr:`arr` flattened to 1D."""
        return self.arr.flatten()

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        """Transform this signal into ``new_domain`` (subclasses must implement)."""
        raise NotImplementedError("Transform needs to be implemented for this signal type.")

    @property
    def shape(self) -> tuple:
        """Shape of :attr:`arr`."""
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
        """Return a sliced copy with both the array and its settings restricted to ``index``."""
        new_arr = self.arr[(Ellipsis,) + index]
        new_settings = self.settings.get_slice(index)
        return self.settings.associated_class(new_arr, new_settings)


class TDSettings(DomainSettingsBase):
    """Time-domain basis settings.

    Args:
        N: Number of time samples.
        dt: Sample spacing in seconds.
        t0: Start time in seconds. Defaults to ``0.0``.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).
    """

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
        """Return the :class:`DomainBase` subclass that pairs with these settings."""
        return TDSignal

    @property
    def associated_class(self):
        """The :class:`DomainBase` subclass that pairs with these settings."""
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        """Keyword arguments needed to reconstruct this settings object."""
        return dict(t0=self.t0, force_backend=self.backend)

    @property
    def args(self) -> tuple:
        """Positional arguments needed to reconstruct this settings object."""
        return (self.N, self.dt)

    @property
    def t_arr(self) -> np.ndarray:
        """Array of sample times: ``t0 + arange(N) * dt``."""
        return self.t0 + self.xp.arange(self.N) * self.dt

    @property
    def basis_shape(self) -> tuple:
        """Total basis shape ``(N,)``."""
        return (self.N,)

    @property
    def basis_shape_active(self) -> tuple:
        """Active basis shape (same as :attr:`basis_shape` for TD)."""
        # TODO: adjust this
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
        """Differential element used in inner-product summations (``dt`` in TD)."""
        return self.dt

    @property
    def total_terms(self) -> int:
        """Total number of basis elements (``N``)."""
        return self.N

    def compute_slice_indices(self, tmin: float, tmax: float) -> slice:
        """Return a ``slice`` along the time axis covering ``[tmin, tmax]``."""
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
    """Time-domain array wrapper paired with :class:`TDSettings`.

    Args:
        arr: NumPy or CuPy array with shape ``(..., N)``.
        settings: :class:`TDSettings` describing the time grid.
    """

    def __init__(self, arr, settings: TDSettings):
        TDSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

    @property
    def settings(self) -> TDSettings:
        """A fresh :class:`TDSettings` matching this signal's time grid."""
        return TDSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"TDSignal(t0={self.t0}, N={self.N}, dt={self.dt}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )
    
    def fft(self, settings=None, window=None):
        """Forward FFT of the (optionally windowed) time-domain signal.

        Args:
            settings: Optional target :class:`FDSettings`; if ``None``, one is built
                from the inferred ``df`` and FFT length.
            window: Optional window applied before the transform.

        Returns:
            :class:`FDSignal` containing the (possibly trimmed) frequency-domain array.
        """
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        df = 1 / (self.N * self.dt)

        fd_arr = self.xp.fft.rfft(self.arr * window, axis=-1) * self.dt
        if settings is not None:
            assert isinstance(settings, FDSettings)
            assert settings.df == df, f"Provided FDSettings has df={settings.df}, but expected df={df} based on TDSettings."
            assert settings.N == fd_arr.shape[-1]
            fd_settings = settings
        
        else:
            fd_settings = FDSettings(fd_arr.shape[-1], df, force_backend=self.backend)
        
        fd_arr_in = fd_arr[..., fd_settings.active_slice]
        return FDSignal(fd_arr_in, fd_settings)

    def stft(self, settings=None, window=None):
        """Short-time Fourier transform of the time-domain signal.

        Args:
            settings: :class:`STFTSettings` describing the segment grid (required).
            window: Optional per-segment window (length ``nperseg``).

        Returns:
            :class:`STFTSignal` containing the time-frequency array sliced to
            ``settings.active_slice``.
        """
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
        """Transform to the WDM wavelet basis via FFT then :meth:`FDSignal.wdmtransform`."""
        if window is None:
            window =self.xp.ones(self.arr.shape, dtype=float)

        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # go to frequency domain then wavelets
        return self.fft(settings=None, window=window).transform(settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray = None):
        """Dispatch to :meth:`fft`, :meth:`stft`, or :meth:`wdmtransform` based on ``new_domain``."""
        if window is None:
            window = self.xp.ones(self.arr.shape, dtype=float)

        if isinstance(new_domain, TDSettings):
            if window is None:
                window = self.xp.ones(self.arr.shape, dtype=float)
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
    """Frequency-domain basis settings on a uniform grid.

    Args:
        N: Total number of frequency bins on the underlying ``arange(0, N) * df`` grid.
        df: Frequency spacing in Hz.
        min_freq: Lower edge of the active band; bins below are masked out via
            :attr:`active_slice`. Defaults to ``0.0``.
        max_freq: Upper edge of the active band; if ``None``, the full range is used.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).
    """

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
        """Differential element used in inner-product summations (``df`` in FD)."""
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
        self._min_freq = value

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
        self._max_freq = value

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
        """Return the :class:`DomainBase` subclass that pairs with these settings."""
        return FDSignal

    @property
    def associated_class(self):
        """The :class:`DomainBase` subclass that pairs with these settings."""
        return self.get_associated_class()

    @property
    def kwargs(self) -> dict:
        """Keyword arguments needed to reconstruct this settings object."""
        return dict(
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            force_backend=self.backend,
        )

    @property
    def args(self) -> tuple:
        """Positional arguments needed to reconstruct this settings object."""
        return (self.N, self.df)

    @property
    def basis_shape(self) -> tuple:
        """Total basis shape ``(N,)``."""
        return (self.N,)

    @property
    def basis_shape_active(self) -> tuple:
        """Active basis shape (after applying ``min_freq``/``max_freq`` masking)."""
        return (self.N_active,)

    @property
    def f_arr(self) -> np.ndarray:
        """Active-band frequency array (Hz)."""
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
            and ((value.min_freq is None) or (self.min_freq is None) or (self.xp.isclose(value.min_freq, self.min_freq)))
            and ((value.max_freq is None) or (self.max_freq is None) or (self.xp.isclose(value.max_freq, self.max_freq)))
        )

    @property
    def total_terms(self) -> int:
        """Total number of basis elements in the active band."""
        return self.N_active

    @property
    def active_slice(
        self,
    ) -> slice:
        """Slice along the frequency axis that selects ``[min_freq, max_freq]``."""
        return slice(self.ind_min, self.ind_max + 1)

    @property
    def N_active(self) -> int:
        """Number of frequency bins in the active band."""
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
    """Frequency-domain array wrapper paired with :class:`FDSettings`.

    Args:
        arr: NumPy or CuPy array. The trailing axis must have length ``N_active``
            (or ``N`` if no min/max masking is applied).
        settings: :class:`FDSettings` describing the frequency grid and band.
    """

    def __init__(self, arr, settings: FDSettings):
        try:
            FDSettings.__init__(self, *settings.args, **settings.kwargs)
        except:
            breakpoint()
        DomainBase.__init__(self, arr)
        if self.arr.shape[-1] != self.N_active:
            assert arr.shape[-1] == self.N
            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[:, self.active_slice]
            # self.arr = 

    @property
    def settings(self) -> FDSettings:
        """A fresh :class:`FDSettings` matching this signal's frequency grid."""
        return FDSettings(*self.args, **self.kwargs)

    def pad_array(self, arr: np.ndarray) -> np.ndarray:
        """Zero-pad ``arr`` (2D) back to the full ``N``-bin grid before an inverse transform."""
        assert arr.ndim == 2
        _arr = np.pad(arr, ((0, 0), (self.ind_min - 1, self.N - 1 - self.ind_max)), mode="constant", constant_values=0.0)
        return _arr

    def ifft(self, settings=None, window=None):
        """Inverse FFT back to the time domain (zero-padding the active band if trimmed)."""

        arr_in = self.arr.copy()
        
        if self.ind_min != 0 or self.ind_max != self.N - 1:
            warnings.warn("Doing an ifft with a trimmed frequency domain array. Zero-padding.")
            arr_in = self.pad_array(arr_in)

        if window is None:
            window = self.xp.ones(arr_in.shape, dtype=float)

        _tmp = self.xp.fft.irfft(arr_in * window, axis=-1)
        
        if settings is None:
            Tobs = 1 / self.df
            Nobs = _tmp.shape[-1]
            dt = Tobs / Nobs
            settings = TDSettings(Nobs, dt)

        td_arr = _tmp / settings.dt
        return TDSignal(td_arr, settings)
    
    def __repr__(self) -> str:
        return (
            f"FDSignal(N={self.N}, df={self.df}, "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )

        td_settings = TDSettings(N, dt, t0=0.0, force_backend=self.force_backend)
        return TDSignal(td_arr, td_settings)

    def get_fd_window_for_wdm(self, settings):
        """Build the WDM analysis window in frequency space (currently unimplemented)."""
        # TODO/DOCS: this method computes only the first half of the WDM window
        # and then raises NotImplementedError before normalising; the active
        # WDM path uses ``settings.window`` instead. Verify whether this helper
        # is still needed.
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
        """Transform the FD signal into the WDM wavelet basis.

        Args:
            settings: :class:`WDMSettings` describing the wavelet grid (required).
            window: Unused in the current implementation; the WDM analysis window
                is taken from ``settings.window``.
            return_transpose_time_axis_first: Currently has no effect (the
                transposed branch is commented out below).
            is_psd: If ``True``, treat the input as a PSD and follow the
                stationary-PSD shortcut (returns a raw NumPy/CuPy array instead
                of a :class:`WDMSignal`).

        Returns:
            :class:`WDMSignal` for ordinary signals, or a NumPy/CuPy array when
            ``is_psd is True``.
        """
        if settings is None:
            raise ValueError("Must provide WDMSettings for WDM transform.")
        assert isinstance(settings, WDMSettings)

        # phif = phitilde_vec_norm(settings.Nf, settings.Nt, 4.0)
        m = self.xp.repeat(self.xp.arange(0, settings.Nf)[:, None], settings.Nt, axis=-1)
        n = self.xp.tile(self.xp.arange(settings.Nt), (settings.Nf, 1))

        m_special = self.xp.repeat(self.xp.arange(0, settings.Nf + 1)[:, None], settings.Nt, axis=-1)
        
        # removed zero frequency and mirrored
        # TODO: WITH ROBBIE CHECK SECOND TO TOP INDEX START AND END
        k = settings.get_shift_map(m_special)
        neg_k = (k < 0)
        over_k = (k > int(settings.N / 2))
        k[neg_k] = np.abs(k[neg_k])
        k[over_k] = settings.N - k[over_k]
        base_window = (settings.window[:])

        arr_in = self.arr.copy()
        
        if self.ind_min != 0 or self.ind_max != self.N - 1:
            warnings.warn("Doing an ifft with a trimmed frequency domain array. Zero-padding.")
            arr_in = self.pad_array(arr_in)

        before_ifft = arr_in[:, k] / settings.data_dt

        if not is_psd:
            herm = neg_k | over_k
            if herm.any():
                before_ifft[:, herm] = self.xp.conj(before_ifft[:, herm])

        if is_psd:
            tmp_arr = before_ifft.copy()
            tmp_arr[:] *= (base_window[None, None, :]) ** 2 * np.pi * settings.data_dt
            psd_sum_tmp = tmp_arr.sum(axis=-1)
            psd_sum_tmp /= settings.Nf * settings.Nt   # = N

            wdmpsd = self.xp.zeros((self.nchannels, settings.Nf, settings.Nt), dtype=complex)

            wdmpsd[:, 1:] = psd_sum_tmp[:, 1:settings.Nf, None]          # regular layers
            wdmpsd[:, 0, 0::2] = psd_sum_tmp[:, 0, None]           # DC at even rows
            wdmpsd[:, 0, 1::2] = psd_sum_tmp[:, settings.Nf, None]

            wdmpsd_out = wdmpsd[:, settings.active_slice_f, settings.active_slice_t]
            return wdmpsd_out

        before_ifft[:] *= base_window[None, None, :]
        after_ifft = self.xp.fft.ifft(before_ifft, axis=-1)
        
        # TODO: fix this

        if self.backend.uses_cupy:
            # some issue with cupy and xp.real/imag
            cache = self.xp.fft.config.get_plan_cache()
            cache.clear()
        
        tmp_w_mn = self.xp.zeros((self.nchannels, settings.Nf + 1, settings.Nt), dtype=float)
        kappa = 2 * np.sqrt(np.pi * settings.data_dt) / settings.Nf
        m_here = np.concatenate([m, np.full((1, settings.Nt), settings.Nf)], axis=0)
        n_here = np.concatenate([n, np.array([np.arange(settings.Nt)])], axis=0)
        set_zero = ((m_here == settings.Nf) | (m_here == 0)) & ((m_here + n_here) % 2 != 0)
        tmp_w_mn[:, ~set_zero] = kappa * (-1) ** ((m_here + 1) * n_here)[~set_zero] * self.xp.real(self.xp.conj(settings.get_Cmn(m_here[~set_zero], n_here[~set_zero])) * after_ifft[:, ~set_zero])
        
        w_mn = self.xp.zeros((self.nchannels, settings.Nf, settings.Nt), dtype=float)
        w_mn[:, 1:] = tmp_w_mn[:, 1:-1]
        w_mn[:, 0, 0::2] = tmp_w_mn[:, 0, 0::2] / np.sqrt(2.)
        w_mn[:, 0, 1::2] = tmp_w_mn[:, settings.Nf, 0::2] / np.sqrt(2.)

        # if return_transpose_time_axis_first:
        #     output = w_mn[:, settings.active_slice_f, settings.active_slice_t].transpose(0, 2, 1).copy()
        # else:
        
        output = w_mn[:, settings.active_slice_f, settings.active_slice_t]

        return WDMSignal(output, settings=settings)

    def transform(self, new_domain: DomainSettingsBase, window: np.ndarray | cp.ndarray = None):
        """Dispatch to :meth:`ifft`, :meth:`wdmtransform`, etc. based on ``new_domain``."""
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
    """Short-time Fourier transform basis settings.

    Args:
        t0: Time of the first segment (seconds).
        dt: Segment cadence in seconds (spacing between successive STFT slices).
        df: Frequency bin spacing in Hz.
        NT: Number of time segments.
        NF: Number of frequency bins per segment.
        min_freq: Lower edge of the active band (Hz). Defaults to ``0.0``.
        max_freq: Upper edge of the active band (Hz). If ``None``, the full range
            is used.
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).
    """

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
        self._min_freq = value

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
        self._max_freq = value

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
            ind_max = self.NF - 1
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
        """Number of samples per segment given the underlying sample step ``small_dt``."""
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
    """STFT array wrapper paired with :class:`STFTSettings`.

    Args:
        arr: NumPy or CuPy array with trailing axes ``(NT, NF_active)``.
        settings: :class:`STFTSettings` describing the time-frequency grid.
    """

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
        """A fresh :class:`STFTSettings` matching this signal's grid."""
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
            ind_max = self.NF - 1
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
WAVELET_FILTER_CONSTANT = 4


class WDMSettings(DomainSettingsBase):
    """Wavelet (WDM) basis settings for time-frequency analysis.

    Args:
        Nf: Number of frequency layers (must be even).
        Nt: Number of time pixels per layer (must be even).
        dt: Underlying time-domain sample step in seconds.
        t0: Start time in seconds. Defaults to ``0.0``.
        oversample: Frequency oversampling factor used when building the WDM
            window via :meth:`setup_window`. Defaults to ``16``.
        window: Pre-computed WDM analysis window of length ``Nt``. If provided,
            ``omega`` must also be supplied; otherwise the window is built by
            :meth:`setup_window`.
        omega: Pre-computed angular-frequency grid that pairs with ``window``.
        min_freq: Lower edge of the active frequency band (Hz). ``None`` selects
            the full range.
        max_freq: Upper edge of the active frequency band (Hz).
        min_time: Lower edge of the active time band (seconds).
        max_time: Upper edge of the active time band (seconds).
        **kwargs: Forwarded to :class:`DomainSettingsBase` (e.g. ``force_backend``).
    """

    def __init__(
        self,
        Nf: float,
        Nt: float,
        dt: float,
        t0: float = 0.0,
        oversample: int = 16,
        window: Optional[np.ndarray] = None,
        # norm: Optional[float] = None,
        omega: Optional[np.ndarray] = None,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
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
        self.t0 = t0

        # these have to come after layer_df b/c setters
        # sets ind_min and ind_max
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_time = min_time
        self.max_time = max_time
        self.Nthalf = int(self.Nt / 2)
        self.oversample = oversample

        self.dOmega = 2 * np.pi * self.layer_df
        self.A = 0.0
        self.WAVELET_FILTER_CONSTANT = 4  # window roll-off parameter
        
        if window is None:
            self.setup_window()
        else:
            # assert norm is not None, "Must provide norm if providing window."
            assert omega is not None, "Must provide omega if providing window."
            assert len(window) == self.Nt
            self.window = window
            # self.norm = norm
            self.omega = omega
    
    @staticmethod
    def adjust_to_even_bins(t_min: float, t_max: float, dt: float, Tobs: float, num_linspace: Optional[int]=1000, verbose: Optional[bool] = False) -> Tuple[int, int, float]:
        """Pick a wavelet pixel duration in ``[t_min, t_max]`` that makes both ``Nf`` and ``Nt`` even.

        Args:
            t_min: Lower bound on the wavelet pixel duration in seconds.
            t_max: Upper bound on the wavelet pixel duration in seconds.
            dt: Underlying time-sample step (the duration is rounded to a multiple).
            Tobs: Total observation time (seconds).
            num_linspace: Number of candidate durations to scan between ``t_min`` and ``t_max``.
            verbose: If ``True``, print each attempted duration.

        Returns:
            ``(Nf, Nt, wavelet_duration)``.

        Raises:
            ValueError: If no candidate produces both ``Nf`` and ``Nt`` even.
        """
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
    def active_slice_f(
        self,
    ) -> slice:
        return slice(self.ind_min_f, self.ind_max_f + 1)
    
    @property
    def active_slice_t(
        self,
    ) -> slice:
        return slice(self.ind_min_t, self.ind_max_t + 1)

    @property
    def active_slice(
        self,
    ) -> slice:
        return (self.active_slice_f, self.active_slice_t)
    
    @property
    def Nf_active(self) -> int:
        sl = self.active_slice_f
        return sl.stop - sl.start

    @property
    def Nt_active(self) -> int:
        sl = self.active_slice_t
        return sl.stop - sl.start

    def __eq__(self, value):
        return (
            (value.Nt == self.Nt) and (value.Nf == self.Nf) 
            and (value.layer_dt == self.layer_dt) and (value.layer_df == self.layer_df) 
            and (value.data_dt == self.data_dt)
            and (value.ind_min_t == self.ind_min_t)
            and (value.ind_max_t == self.ind_max_t)
            and (value.ind_min_f == self.ind_min_f)
            and (value.ind_max_f == self.ind_max_f)
        )
    
    def eq_without_inds(self, value):
        return (
            (value.Nt == self.Nt) and (value.Nf == self.Nf) 
            and (value.layer_dt == self.layer_dt) and (value.layer_df == self.layer_df) 
            and (value.data_dt == self.data_dt)
        )

    @property
    def basis_shape(self) -> tuple:
        return (self.Nf, self.Nt)
    
    @property
    def basis_shape_active(self) -> tuple:
        return (self.Nf_active, self.Nt_active)
    
    @property
    def t_td_arr(self) -> np.ndarray:
        _tmp = self.xp.arange(self.N) * self.data_dt
        tmp = _tmp[(_tmp >= self.ind_min_t * self.layer_dt) & (_tmp <= self.ind_max_t * self.layer_dt)]
        return tmp
    
    @property
    def t_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nt)[self.active_slice_t] * self.layer_dt

    @property
    def f_arr(self) -> np.ndarray:
        return self.xp.arange(self.Nf)[self.active_slice_f] * self.layer_df
    
    @property
    def f_arr_edges(self) -> np.ndarray:
        return (self.xp.arange(self.Nf_active + 1) + self.ind_min_f) * self.layer_df
    @property
    def t_arr_edges(self) -> np.ndarray:

        return (self.xp.arange(self.Nt_active + 1) + self.ind_min_t) * self.layer_dt

    def phitilde(self, omega, dOmega):
        """Smooth WDM frequency-domain analysis function :math:`\\tilde{\\phi}(\\omega)`."""
        # TODO/DOCS: parameters ``A``, ``B`` and ``WAVELET_FILTER_CONSTANT``
        # control the window roll-off; the exact convention follows the WDM
        # transform of Cornish & Romano. Verify against the cited reference.
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
    
    def get_Cmn(self, m: np.array[int], n: np.array[int]) -> np.array[int]:
        """Return ``1`` for even ``(m + n)`` and ``1j`` for odd ``(m + n)``."""
        m_in = self.xp.atleast_1d(m)
        n_in = self.xp.atleast_1d(n)
        output = np.zeros(m_in.shape, dtype=complex)
        is_even = ((m_in + n_in) % 2) == 0
        output[is_even] = 1.0
        output[~is_even] = 1j
        return output

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
        """Return a 2D shift map ``m * Nt/2 + arange(-Nt/2, Nt/2)`` used by the WDM transform."""
        if m.ndim == 1:
            m_in = m[:, None]
        elif m.ndim == 2:
            m_in = m
        else:
            raise ValueError("m must be 1D or 2D array.")

        return m_in * int(self.Nt / 2) + self.xp.arange(-int(self.Nt / 2),  int(self.Nt / 2))[None, :]
        
    # def window_norm(self) -> float:
    #     dOmega_s = np.pi / self.Nf
    #     (2 * np.pi) / self.N 

    def setup_window(self):  # , forward: bool= True):
        """Build the default WDM analysis window and store it in :attr:`window` / :attr:`omega`."""
        # *DX = (double*)malloc(sizeof(double)*(2*wdm->N))
        # zero frequency
        # REAL(DX,0) =  wdm->inv_root_dOmega
        # IMAG(DX,0) =  0.0
        T = self.data_dt * self.N
        dOmega_s = np.pi / self.Nf

        self.A = dOmega_s / 4.0
        self.omega = omega =  2 * np.pi / self.N * (self.xp.arange(-int(self.Nt / 2),  int(self.Nt / 2)))
        phif = self.phitilde(omega, dOmega_s)
        self.window = phif

        # 2 * sqrt(pi) / Nf

        # we apply this outside the window setup so the window is the same for forward and backward
        # if forward:
        #     self.window *= 2. / self.Nf

        assert 0.0 in omega

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
            self.ind_min_f = int(np.ceil(value / self.layer_df))
            if self.ind_min_f < 0:
                self.ind_min_f = 0
        else:
            self.ind_min_f = 0
        self._min_freq = value

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
            self.ind_max_f = int(value / self.layer_df)
            if self.ind_max_f > (self.Nf - 1):
                self.ind_max_f = (self.Nf - 1)
        else:
            self.ind_max_f = (self.Nf - 1)
        self._max_freq = value

    @property
    def min_time(self) -> float:
        return self._min_time

    @min_time.setter
    def min_time(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_time must be non-negative.")

        # self._min_time = value
        # set it to the closest time bin
        self.min_time_input = value

        if value is not None:
            self.ind_min_t = int(np.ceil(value / self.layer_dt))
            if self.ind_min_t < 0:
                self.ind_min_t = 0
        else:
            self.ind_min_t = 0
        self._min_time = value

    @property
    def max_time(self) -> Optional[float]:
        return self._max_time

    @max_time.setter
    def max_time(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_time must be non-negative.")

        # self._max_time = value
        # set it to the closest time bin
        self.max_time_input = value
        if value is not None:
            self.ind_max_t = int(value / self.layer_dt)
            if self.ind_max_t > (self.Nt - 1):
                self.ind_max_t = (self.Nt - 1)
        else:
            self.ind_max_t = (self.Nt - 1)

        self._max_time = value

    @property
    def ind_min_f(self) -> int:
        return self._ind_min_f
    
    @ind_min_f.setter
    def ind_min_f(self, ind_min_f: int):
        if ind_min_f is None:
            ind_min_f = 0
        self._ind_min_f = ind_min_f

    @property
    def ind_max_f(self) -> int:
        return self._ind_max_f
    
    @ind_max_f.setter
    def ind_max_f(self, ind_max_f: int):
        if ind_max_f is None:
            ind_max_f = self.Nf - 1
        self._ind_max_f = ind_max_f

    @property
    def ind_min_t(self) -> int:
        return self._ind_min_t
    
    @ind_min_t.setter
    def ind_min_t(self, ind_min_t: int):
        if ind_min_t is None:
            ind_min_t = 0
        self._ind_min_t = ind_min_t

    @property
    def ind_max_t(self) -> int:
        return self._ind_max_t
    
    @ind_max_t.setter
    def ind_max_t(self, ind_max_t: int):
        if ind_max_t is None:
            ind_max_t = self.Nt - 1
        self._ind_max_t = ind_max_t

    @property
    def frequency_layer_mask(self) -> Optional[np.ndarray]:
        mask = self.xp.zeros(self.Nf, dtype=bool)
        mask[self.active_slice_f] = True
        return mask
    
    @property
    def time_layer_mask(self) -> Optional[np.ndarray]:
        mask = self.xp.zeros(self.Nt, dtype=bool)
        mask[self.active_slice_t] = True
        return mask
        
    @property
    def f_ind_array(self) -> np.ndarray:
        return self.xp.arange(self.ind_min_f, self.ind_max_f + 1)

    @property
    def t_ind_array(self) -> np.ndarray:
        return self.xp.arange(self.ind_min_t, self.ind_max_t + 1)

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
            # norm=self.norm, 
            omega=self.omega, 
            min_freq=self.min_freq, 
            max_freq=self.max_freq, 
            min_time=self.min_time, 
            max_time=self.max_time, 
            force_backend=self.backend
        )

    @property
    def args(self) -> tuple:
        return (self.Nf, self.Nt, self.data_dt)   
    
    @property
    def differential_component(self) -> float:
        return 0.25

    @property
    def total_terms(self) -> int:
        return self.Nt * self.Nf
    
    def apply_frequency_layer_mask(self, arr: np.ndarray) -> np.ndarray:
        """Apply :attr:`frequency_layer_mask` to ``arr`` along the WDM frequency axis (penultimate dim)."""
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
    """WDM wavelet array wrapper paired with :class:`WDMSettings`.

    Args:
        arr: NumPy or CuPy array with trailing axes ``(Nf_active, Nt_active)``.
        settings: :class:`WDMSettings` describing the wavelet grid.
    """

    # TEST back and forth
    # tmp_dat = np.zeros(wdm_set.N)
    # tmp_dat[0] = 1.0
    # _frq = np.fft.rfftfreq(wdm_set.N, wdm_set.data_dt)
    # tmp_dat_td = TDSignal(tmp_dat, TDSettings(wdm_set.N, wdm_set.data_dt))
    # tmp_dat_fd = tmp_dat_td.fft(FDSettings(_frq.shape[0], _frq[1] - _frq[0]))
    # check_1 = tmp_dat_td.fft().ifft()
    # tmp_dat_check_td_1 = tmp_dat_fd.wdmtransform(wdm_set).wdm_to_fd(tmp_dat_fd.settings).ifft()
    # tmp_dat_check_td = tmp_dat_td.wdmtransform(wdm_set).wdm_to_td()

    # assert np.allclose((_tmp := tmp_dat_check_td[:, 0]), np.ones_like(_tmp))
    # assert np.allclose((_tmp := tmp_dat_check_td[:, 1:]), np.zeros_like(_tmp))

    def __init__(self, arr, settings: WDMSettings):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        DomainBase.__init__(self, arr)

        # freq layers
        if self.arr.shape[-2] != self.Nf_active or self.arr.shape[-1] != self.Nt_active:
            if self.arr.shape[-2] != self.Nf_active:
                assert arr.shape[-2] == self.Nf
                f_slice = self.active_slice_f
            else:
                f_slice = slice(None)

            if self.arr.shape[-1] != self.Nt_active:
                assert arr.shape[-1] == self.Nt
                t_slice = self.active_slice_t
            else:
                t_slice = slice(None)

            _arr = self._arr.copy()
            del self._arr
            self.arr = _arr[:, f_slice, t_slice]

    @property
    def settings(self) -> WDMSettings:
        return WDMSettings(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"WDMSignal(Tobs={self.Tobs}, dt={self.data_dt}, t0={self.t0}, "
            f"Nt={self.Nt}, Nf={self.Nf}, oversample={self.oversample}, "
            f"backend={self.backend_name.split('_')[-1]})"
        )
    
    def wdm_to_td(self, settings=None, window=None):
        """Inverse-transform from WDM to time domain (via FD)."""
        return self.wdm_to_fd(settings=None).ifft(settings=settings, window=window)

    def wdm_to_fd(self, settings=None, window=None):
        """Inverse-transform from WDM to frequency domain.

        Args:
            settings: Optional target :class:`FDSettings`; if ``None``, one is
                derived from the underlying ``data_dt`` and ``N``.
            window: Currently unused (the WDM analysis window stored on the
                signal is used internally).

        Returns:
            :class:`FDSignal`.
        """
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

        m_special = self.xp.repeat(self.xp.arange(0, self.Nf + 1)[:, None], self.Nt, axis=-1)
        
        tmp_w_mn = self.xp.zeros((self.nchannels, self.Nf + 1, self.Nt))
        tmp_w_mn[:, 1:-1] = self.arr[:, 1:]
        
        # dc layer
        tmp_w_mn[:, 0, 0::2] = self.arr[:, 0, 0::2] * np.sqrt(2)
        # max freq layer
        tmp_w_mn[:, self.Nf, 0::2] = self.arr[:, 0, 1::2] * np.sqrt(2)

        lambda_coef = np.sqrt(np.pi / self.data_dt)

        m_here = self.xp.concatenate([m, self.xp.full((1, self.Nt), self.Nf)], axis=0)
        n_here = self.xp.concatenate([n, self.xp.array([self.xp.arange(self.Nt)])], axis=0)

        arr_fd = self.xp.zeros((self.nchannels, settings.N), dtype=complex)
        g_arr = tmp_w_mn * self.get_Cmn(m_here, n_here) * (-1) ** ((m_here + 1) * n_here)

        W_m = self.xp.fft.fft(g_arr, axis=-1)

        v = lambda_coef * base_window * W_m

        k = self.get_shift_map(m_special)
        
        k_even_m = k[0::2]
        k_odd_m = k[1::2]
        
        # for checking, but slow so commenting out
        # assert np.unique(k_even_m).shape[0] == np.prod(k_even_m.shape)
        # assert np.unique(k_odd_m).shape[0] == np.prod(k_odd_m.shape)
        keep_k_even = (k_even_m >= 0) & (k_even_m < settings.N)
        keep_k_odd = (k_odd_m >= 0) & (k_odd_m < settings.N)
        arr_fd[:, k_even_m[keep_k_even]] += v[:, 0::2][:, keep_k_even]
        arr_fd[:, k_odd_m[keep_k_odd]] += v[:, 1::2][:, keep_k_odd]

        # add dt factor for units
        arr_fd *= self.data_dt
        return FDSignal(arr_fd, settings)
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
        """Dispatch to the correct WDM-to-X conversion based on ``new_domain``."""
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
                return self.wdm_to_fd(settings=None, window=None).wdmtransform(
                    settings=new_domain, window=window
                )
        else:
            raise ValueError(f"new_domain type is not recognized {type(new_domain)}.")

    def heatmap(self, index: int = None, mag: bool = False, fig=None, ax=None, cax=None, add_cax=False, log: bool = False, **kwargs):
        """Produce a time-frequency heatmap of the WDM coefficients.

        Args:
            index: If given, plot only channel ``index`` on the supplied ``ax``.
                If ``None``, every channel is plotted on its own row.
            mag: If ``True``, plot ``|coeff|``; otherwise plot the signed value.
            fig: Existing :class:`matplotlib.figure.Figure` (optional).
            ax: Existing axes; required when ``index`` is provided.
            cax: Optional axes for the colourbar.
            add_cax: If ``True``, create a new colourbar axes.
            log: If ``True``, plot ``log10(|coeff|)``.
            **kwargs: Forwarded to :func:`matplotlib.pyplot.pcolormesh`.

        Returns:
            ``(fig, ax)``.
        """
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
                if mag:
                    z = np.abs(z)
                if log:
                    z = np.log10(np.abs(z))
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
            if mag:
                z = np.abs(z)
            if log:
                z = np.log10(np.abs(z))
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
    """Pre-computed sine/cosine WDM-pixel coefficient lookup table.

    Builds (or loads from disk) tables indexed by frequency offset, frequency
    derivative, time pixel, and frequency-layer offset that allow fast
    evaluation of monochromatic / chirping signal templates in the WDM basis.

    Args:
        settings: Underlying :class:`WDMSettings` defining the wavelet grid.
        nchannels: Number of channels stored in the table.
        m_ref: Reference frequency-layer index used as the centre of the
            table. Defaults to a sensible value chosen by the helper.
        norm_freq_single_layer: Sub-bin frequency offsets sampled within one
            wavelet layer.
        m_diffs: Integer offsets (in frequency layers) covered by the table.
        fdot_vals: Frequency-derivative grid (Hz/s) sampled in the table.
        store_path: HDF5 file path. If it exists, the table is loaded; otherwise
            it is built and saved here.
        batch_size_gen: Batch size used when generating the table on-the-fly
            (``-1`` means a single batch).
        td_window: Optional time-domain window applied while building the
            table.
    """

    def to_file(self, fp: str):
        """Persist the lookup table to an HDF5 file at ``fp``."""
        if os.path.exists(fp):
            raise ValueError("Trying to write to file that exists.")
        
        with h5py.File(fp, "w") as fp:
            g = fp.create_group("wdm")

            g.attrs["Nf"] = self.Nf
            g.attrs["Nt"] = self.Nt
            g.attrs["Nt_generate"] = self.sub_settings.Nt
            g.attrs["data_dt"] = self.data_dt

            if self.max_freq is not None:
                g.attrs["max_freq"] = self.max_freq
            if self.min_freq is not None:
                g.attrs["min_freq"] = self.min_freq
            if self.max_time is not None:
                g.attrs["max_time"] = self.max_time
            if self.min_time is not None:
                g.attrs["min_time"] = self.min_time

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
        """Construct a :class:`WDMLookupTable` from a previously-saved HDF5 file."""
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
            else:
                
                input_args = (
                    g.attrs["Nf"],
                    g.attrs["Nt"],
                    g.attrs["data_dt"] 
                )
                nchannels = g.attrs["nchannels"]

                min_freq = None
                if "min_freq" in g.attrs:
                    min_freq = g.attrs["min_freq"]

                max_freq = None
                if "max_freq" in g.attrs:
                    max_freq = g.attrs["max_freq"]

                min_time = None
                if "min_time" in g.attrs:
                    min_time = g.attrs["min_time"]

                max_time = None
                if "max_time" in g.attrs:
                    max_time = g.attrs["max_time"]

                input_kwargs = dict(
                    min_freq = min_freq,
                    max_freq = max_freq,
                    min_time = min_time,
                    max_time = max_time,
                    force_backend=force_backend,
                )
            settings = WDMSettings(*input_args, **input_kwargs)
            return WDMLookupTable(settings, nchannels, store_path=fp)

    def from_file_internal(self, fp: str):
        """Repopulate this instance's tables and metadata from ``fp``."""
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
        """Build a symmetric :math:`\\dot{f}` grid spaced by ``eps * df_layer / dt_layer``."""
        delta_fdot = eps * settings.layer_df / settings.layer_dt
        fdot_max_val = fdot_max_factor * settings.layer_df / settings.layer_dt
        _fdot = np.arange(0.0, fdot_max_val, delta_fdot)
        fdot_vals = np.concatenate([-_fdot[::-1][:-1], _fdot])
        return fdot_vals

    @staticmethod
    def apply_eps_frequency(eps: float, settings: WDMSettings, m_ref: Optional[int] = None, num_layers_diff: Optional[int] = 2) -> tuple:
        """Build sub-layer frequency offsets, layer-index offsets, and the reference layer."""
        delta_f = eps * settings.layer_df

        if m_ref is None:
            m_ref = int(settings.Nt / 2)

        norm_freq_single_layer = np.arange(0.0, settings.layer_df, delta_f)
        m_diffs = (_tmp := np.arange(2 * num_layers_diff + 2)) - _tmp[int(len(_tmp) / 2)]

        return norm_freq_single_layer, m_diffs, m_ref

    def __init__(self, settings: WDMSettings, nchannels: int, m_ref: int = None, norm_freq_single_layer: np.ndarray = None, m_diffs: np.ndarray = None, fdot_vals: np.ndarray = None, store_path: Optional[str] = None, batch_size_gen: Optional[int] = 20, td_window: Optional[np.ndarray] = None):
        WDMSettings.__init__(self, *settings.args, **settings.kwargs)
        # TODO: CHECK FIRST AND LAST TIME LAYERS DUE TO TIME WINDOWING?

        self.nchannels = nchannels
        
        self.store_path = store_path
        if os.path.exists(self.store_path):
            self.from_file_internal(self.store_path)
        else:
            self.build_lookup_table(m_ref, m_diffs, norm_freq_single_layer, fdot_vals, store_path, batch_size_gen, td_window)

    def build_lookup_table(self, m_ref: int, m_diffs: np.ndarray, norm_freq_single_layer: np.ndarray, fdot_vals: np.ndarray, store_path: str, batch_size_gen: int, td_window: Optional[np.ndarray] = None) -> None:
        """Generate the sin/cos coefficient tables and (optionally) save them to ``store_path``."""
        self.sub_settings = WDMSettings(self.Nf, self.Nt, self.data_dt, force_backend=self.force_backend)
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

        if td_window is None:
            td_window = self.xp.ones_like(t_diff)
        
        # TODO: put self.Nt at the end for memory coalescence on GPU?
        _table_sin = self.xp.zeros((self.Nt, len(m_diffs), total_f_fdot_vals,))
        _table_cos = self.xp.zeros((self.Nt, len(m_diffs), total_f_fdot_vals,))
        
        self.td_window = td_window
        for st_batch, end_batch in zip(batches[:-1], batches[1:]):
            inds = np.arange(st_batch, end_batch)
            
            # if not self.xp.allclose(_f_vals[inds] - (self.f_ref + 0 * self.layer_df), 0.0) or not _fdot_vals[inds][0] == 0.0:  # -3.257427471431767e-11:
            #     continue
            wave_sin = self.xp.sin(2 * np.pi * (_f_vals[inds, None] * t_diff[None, :] + 1. / 2. * _fdot_vals[inds, None] * t_diff[None, :] ** 2))
            wave_cos = self.xp.cos(2 * np.pi * (_f_vals[inds, None] * t_diff[None, :] + 1. / 2. * _fdot_vals[inds, None] * t_diff[None, :] ** 2))
            
            wave_sin_wdm = TDSignal(wave_sin, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)
            wave_cos_wdm = TDSignal(wave_cos, TDSettings(self.sub_settings.N, self.sub_settings.data_dt, force_backend=self.force_backend)).wdmtransform(settings=self.sub_settings, window=self.td_window)
        
            n_current = self.xp.arange(self.Nt)
            for m_i, m_diff in enumerate(m_diffs):
                m_current = self.m_ref + m_diff
                _sin_coeff = wave_sin_wdm[:, self.m_ref - m_diff, :].T
                _cos_coeff = wave_cos_wdm[:, self.m_ref - m_diff, :].T

                _f_norm = _f_vals[inds] - (self.m_ref - m_diff) * self.layer_df

                is_odd = ((m_current + n_current) % 2 == 1)[:, None]
                # switch odd numbered pixels
                sin_coeff = _sin_coeff * (~is_odd) + _cos_coeff * (is_odd)
                cos_coeff = _sin_coeff * (is_odd) + _cos_coeff * (~is_odd)
                try:
                    _table_sin[:, m_i, inds] = sin_coeff
                    _table_cos[:, m_i, inds] = cos_coeff
                except:
                    breakpoint()
            print(inds, total_f_fdot_vals)


        # TODO: verify if there is a minus sign needed here and below
        # freqs =  (_f_vals.reshape(-1, self.fdot_steps) +  self.xp.asarray(m_diffs)[:, None, None] * self.layer_df).transpose(1, 0, 2).reshape(self.fdot_steps, -1)
        _table_sin = _table_sin.reshape(self.Nt, len(m_diffs), self.fdot_steps, self.norm_f_steps).transpose(0, 2, 1, 3).reshape(self.Nt, self.fdot_steps, self.f_steps).copy()
        _table_cos = _table_cos.reshape(self.Nt, len(m_diffs), self.fdot_steps, self.norm_f_steps).transpose(0, 2, 1, 3).reshape(self.Nt, self.fdot_steps, self.f_steps).copy()

        assert _table_sin.shape == (self.Nt, self.fdot_vals.shape[0], self.f_vals.shape[0])
        assert _table_cos.shape == (self.Nt, self.fdot_vals.shape[0], self.f_vals.shape[0])
        
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
        """Construct a scipy/cupyx interpolator over ``table`` for fast template generation."""
        if self.backend.uses_cupy:
            interpolate = interpolate_gpu
        else:
            interpolate = interpolate_cpu

        # TODO: vectorized versions?
        if self.run_fdot:
            raise NotImplementedError
            return [interpolate.LinearNDInterpolator(self.norm_points, tmp.flatten(), rescale=True) for tmp in table]
        else:
            # TODO: is this an okay way to linear spline?
            self.factor_spacing = 1e-3
            assert (self.norm_points.max() - self.norm_points.min()) < self.factor_spacing
        
            norm_points_in = self.xp.tile(self.norm_points, (self.Nt, 1)).flatten()
            n_arr_in = self.xp.repeat(self.xp.arange(self.Nt)[:, None], self.norm_points.shape[0], axis=-1).flatten()
            x_points_in = self.get_x_points_no_fdot(norm_points_in, n_arr_in)
            y_points_in = table.flatten()
            return interpolate.interp1d(x_points_in, y_points_in)
        
    def get_x_points_no_fdot(self, f_norm: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
        """1D evaluation coordinate that encodes both ``f_norm`` and the time-pixel index."""
        return (f_norm + self.factor_spacing * n_arr)

    def get_table_coeffs(self, f_norm: np.ndarray, fdot_arr: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
        """Look up sin/cos coefficients at the requested ``(f_norm, fdot, n)`` points."""
        # ms = (f_arr // self.layer_df).astype(int)
        # TODO: vectorize?
        if self.run_fdot:
            raise NotImplementedError
            sin_coeffs = self.table_sin_interpolate(f_norm, fdot_arr)
            cos_coeffs = self.table_cos_interpolate(f_norm, fdot_arr)
            #breakpoint()
        else:
            x_points_in = self.get_x_points_no_fdot(f_norm, n_arr)
            sin_coeffs = self.table_sin_interpolate(x_points_in)
            cos_coeffs = self.table_cos_interpolate(x_points_in)

        sin_coeffs[np.isnan(sin_coeffs)] = 0.0
        cos_coeffs[np.isnan(cos_coeffs)] = 0.0
        return (sin_coeffs, cos_coeffs)

    def get_wdm_coeffs(self, amp_arr: np.ndarray, phi_arr: np.ndarray, f_arr: np.ndarray, fdot_arr: np.ndarray, n_arr: np.ndarray, num_m_layers: int = 1):
        """Compute amplitude / phase WDM coefficients for a batch of ``(f, fdot, n)`` queries.

        Returns:
            ``(wdm_coeffs_out, m_map)`` where ``wdm_coeffs_out`` has shape
            ``(len(amp_arr), 2 * num_m_layers + 1)`` and ``m_map`` records the
            integer frequency-layer index used for each output column (``-1``
            for entries that fall outside the table).
        """
        # TODO/DOCS: the parity logic that switches between sin/cos lookup
        # tables follows the WDM convention used elsewhere in this file;
        # verify the sign/swap rules against the WDM transform definition.
        ms = (f_arr / self.layer_df).astype(int)
        wdm_coeffs_out = self.xp.zeros((amp_arr.shape[0], num_m_layers * 2 + 1))
        m_map = -self.xp.ones((amp_arr.shape[0], num_m_layers * 2 + 1), dtype=int)
        is_m_ref_n_ref_even = (self.m_ref + self.n_ref) % 2 == 0
        for i, m_diff in enumerate(range(-num_m_layers, num_m_layers + 1)):

            ms_to_use = (ms + m_diff).astype(int)
            keep_now = self.xp.arange(ms_to_use.shape[0])[(ms_to_use >= 0) & (ms_to_use < self.Nf)]

            assert ms_to_use[keep_now].max() <= self.Nf + 1
            assert ms_to_use[keep_now].min() >= 0
            try:
                assert self.xp.all((f_arr[keep_now] >= 0.0) & (f_arr[keep_now] <= self.f_arr.max()))
            except AssertionError:
                breakpoint()
            assert self.xp.all((fdot_arr[keep_now] >= self.fdot_vals.min()) & (fdot_arr[keep_now] <= self.fdot_vals.max()))
            f_norm = (f_arr[keep_now] - ms_to_use[keep_now] * self.layer_df)
            
            _sin_coeffs, _cos_coeffs = self.get_table_coeffs(f_norm, fdot_arr[keep_now], n_arr[keep_now])
            is_m_plus_n_even = (((ms_to_use[keep_now] + n_arr[keep_now]) % 2 == 0)) 
            is_m_even = (ms_to_use[keep_now] % 2 == 0)

            sin_coeffs = self.xp.zeros_like(_sin_coeffs)
            cos_coeffs = self.xp.zeros_like(_cos_coeffs)
            
            sin_coeffs[~is_m_plus_n_even & is_m_even] = _sin_coeffs[~is_m_plus_n_even & is_m_even]
            cos_coeffs[~is_m_plus_n_even & is_m_even] = -_cos_coeffs[~is_m_plus_n_even & is_m_even]

            sin_coeffs[~is_m_plus_n_even & ~is_m_even] = _sin_coeffs[~is_m_plus_n_even & ~is_m_even]
            cos_coeffs[~is_m_plus_n_even & ~is_m_even] = _cos_coeffs[~is_m_plus_n_even & ~is_m_even]

            # if np.any(np.abs(_sin_coeffs) > 1e-3):
            #     breakpoint()
                
            sin_coeffs[is_m_plus_n_even & is_m_even] = _cos_coeffs[is_m_plus_n_even & is_m_even]
            cos_coeffs[is_m_plus_n_even & is_m_even] = _sin_coeffs[is_m_plus_n_even & is_m_even]

            sin_coeffs[is_m_plus_n_even & ~is_m_even] = -_cos_coeffs[is_m_plus_n_even & ~is_m_even]
            cos_coeffs[is_m_plus_n_even & ~is_m_even] = _sin_coeffs[is_m_plus_n_even & ~is_m_even]

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
    """Return the list of :class:`DomainSettingsBase` subclasses supported by the package."""
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
