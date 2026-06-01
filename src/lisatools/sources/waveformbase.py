# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments
# pyling: diable=too-many-keywords-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-instance-attributes, too-few-public-methods, too-many-locals
# pylint: disable=too-many-positional-arguments, too-many-statements, line-too-long
"""
WaveformBase module

This module defines the base wrappers for waveform generation, including the application of the LISA response.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

from fastlisaresponse import pyResponseTDI
from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse.tdionfly import TDTDIonTheFly

from ..domains import (
    DomainSettingsBase,
    DomainBase,
    DomainBaseArray,
    FDSettings,
    TDSettings,
    TDSignal,
    get_stft_settings,
)
from ..utils.parallelbase import LISAToolsParallelModule
from ..utils.typing import NDArrayLike, ArrayModule
from ..utils.utility import tukey

if TYPE_CHECKING:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        import numpy as cp

    from lisatools.detector import Orbits, EqualArmlengthOrbits

logger = logging.getLogger(__name__)

DEBUG_MODE = False

class AETTDIWaveform(ABC):
    """Base class for an AET TDI Waveform."""

    @property
    def dt(self) -> float:
        """Timestep in seconds."""
        return None

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return None

    @property
    def df(self) -> float:
        """Frequency bin size."""
        return None


class SNRWaveform(ABC):
    """Base class for a waveform built in a simpler fashion for SNR calculations."""

    @property
    def dt(self) -> float:
        """Timestep in seconds."""
        return None

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return None

    @property
    def df(self) -> float:
        """Frequency bin size."""
        return None


class TDWaveformBase(ABC, LISAToolsParallelModule):
    """
    Base class for a waveform built in the time domain.

    Args:
    waveform_t0: Initial time in seconds.
    dt: Time step in seconds.
    data_t0: Optional initial time for the data. If None, defaults to waveform_t0.
            If provided, the output time arrays will be shifted so that the first sample corresponds to a integer multiple of dt after data_t0.
            This allows for proper alignment of the waveform with an external time grid (e.g. from a loader) when data_t0 is set to the same reference time as the loader.
    response_kwargs: Keyword arguments for the TDI response.
    buffer_time: Time in seconds to add as buffer to the TDI response to ensure proper calculation at the beginning and end of the signal.
    tukey_alpha: Alpha parameter for the Tukey window applied to the output signal. Only applied if settings_class is not None.
    force_uniform_stft: If True, batched calls in STFT mode will force all signals onto a common STFT grid
        spanning the union of all source time ranges. If False (default), each source retains its natural
        STFT grid derived from its own time range. Only relevant for batched calls with output_domain='STFT'.
    """

    def __init__(
        self,
        waveform_t0: float,
        data_td_settings: TDSettings,
        tdi_generation: str = "2nd generation",
        tdi_channels: str = "XYZ",
        sampling_frequency: float = 0.4,
        orbits: Orbits = None,
        tukey_alpha: float = 0.01,
        stft_dt: float = None,
        freq_min: float = 1e-5,
        freq_max: float = 1.0,
        fft_batch_size: int = 1,
        force_backend: str = "cpu",
    ) -> None:

        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

        self.waveform_t0 = waveform_t0
        self.domain_settings = data_td_settings
        self.tukey_alpha = tukey_alpha

        self.tdi_config = TDIConfig(tdi=tdi_generation, force_backend=force_backend)
        self.tdi_generation = tdi_generation # store for repr
        self.tdi_channels = tdi_channels
        self.sampling_frequency = sampling_frequency
        self.orbits = orbits

        if stft_dt is None:
            logger.info(
                "No stft timestep provided. By default, the waveform will be transformed to the frequency domain"
            )
            self.transform_to_domain = self.fft
            self.nperseg = None

        else:
            assert self.dt <= stft_dt
            nperseg = round(stft_dt / self.dt)

            assert (
                abs(nperseg * self.dt - stft_dt) < 1e-10 * stft_dt
            ), f"stft_dt={stft_dt} must be an integer multiple of dt={self.dt}"

            logger.info(
                f"STFT timestep set to {stft_dt}. This corresponds to {nperseg} points per time segment"
            )
            self.nperseg = nperseg
            self.transform_to_domain = self.stft

        self.freq_min = freq_min
        self.freq_max = freq_max
        self.fft_batch_size = fft_batch_size

    @property
    def wrapper_kwargs(self) -> dict:
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""
        return {
            "waveform_t0": self.waveform_t0,
            "data_td_settings": self.domain_settings,
            "tdi_generation": self.tdi_generation,
            "tdi_channels": self.tdi_channels,
            "sampling_frequency": self.sampling_frequency,
            "orbits": self.orbits,
            "tukey_alpha": self.tukey_alpha,
            "stft_dt": self.nperseg * self.dt if self.nperseg else None,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "fft_batch_size": self.fft_batch_size,
            "force_backend": self.force_backend,
        }


    @property
    def force_backend(self) -> str:
        """Name of the backend to be passed to other classes."""
        return self.backend_name.split("_")[-1]

    @property
    def xp(self) -> ArrayModule:
        """Array module used for calculations."""
        return self.backend.xp

    @property
    def tdi_buffer_time(self) -> float:
        """Buffer time in seconds to ensure proper TDI response calculation at the boundaries."""
        return 600.0

    @property
    def dt(self):
        """Time step in seconds."""
        return self.domain_settings.dt

    @property
    def Tobs(self):
        """Observation time in seconds."""
        return self.domain_settings.N * self.domain_settings.dt

    @property
    def data_t0(self):
        """Start time of the data in seconds."""
        return self.domain_settings.t0

    @property
    def stft_t_arr(self):
        """Time array for the STFT segments, if applicable."""
        if self.nperseg:
            return self.domain_settings.t_arr[:: self.nperseg]
        else:
            return None

    @property
    def data_times_array(self):
        """Complete time array on which the data live."""
        return self.domain_settings.t_arr

    @property
    def orbits(self):
        """Orbits object for on-the-fly response generation."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits):
        """Set the Orbits object."""

        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=self.force_backend)
            logger.warning("No Orbits object provided. Using default EqualArmlengthOrbits.")

        if not orbits.configured:
            orbits.configure(linear_interp_setup=True)
        self._orbits = orbits

    @property
    def analysis_domain(self) -> str:
        """The domain in which the waveform is transformed for likelihood evaluation. Currently, either 'STFT' or 'FD'."""
        if self.nperseg:
            return "STFT"
        else:
            return "FD"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(waveform_t0={self.waveform_t0}, "
            f"data_td_settings={self.domain_settings}, "
            f"tdi_config={self.tdi_config}, "
            f"sampling_frequency={self.sampling_frequency}, "
            f"orbits={self.orbits}, "
            f"tukey_alpha={self.tukey_alpha}, "
            f"stft_dt={self.nperseg * self.dt if self.nperseg else None}, "
            f"freq_min={self.freq_min}, "
            f"freq_max={self.freq_max}, "
            f"force_backend='{self.force_backend}')"
        )

    @abstractmethod
    def compute_tdi_channels(
        self, *args, **kwargs
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Time domain TDI channels computation. The output must be a tuple of (times, channels), where `times` can be a 1D array of shape (num_times,) or a 2D array of shape (num_bin, num_times) for batched generation, and `channels` is the corresponding TDI response with shape (num_channels, num_times) or (num_bin, num_channels, num_times) respectively."""

        raise NotImplementedError("compute_tdi_channels method must be implemented in subclass.")

    def get_grid_time(self, times: NDArrayLike) -> NDArrayLike:
        """
        For a given array of times, compute the closest points on the grid defined by the data time step.

        Args:
            times (Array): Array of times to be projected on the grid. Shape: (num_times,).

        Returns:
            Times on the target grid.  Shape: (num_times,).
        """
        dt = self.dt
        t0 = self.data_t0
        return t0 + self.xp.round((times - t0) / dt) * dt

    def get_output_settings(self, times: NDArrayLike) -> DomainSettingsBase:
        """
        Get the settings for the output domain based on the evaluation times and the chosen analysis domain (STFT or FD).

        Args:
            times (Array): Array of evaluation times. Shape: (num_times,).

        Returns:
            DomainSettingsBase: The settings for the output domain.
        """
        if self.analysis_domain == "STFT":
            return get_stft_settings(
                times,
                big_dt=self.nperseg * self.dt,
                min_freq=self.freq_min,
                max_freq=self.freq_max,
                force_backend=self.force_backend,
            )
        elif self.analysis_domain == "FD":
            data_N = self.domain_settings.N
            return FDSettings(
                N=data_N // 2 + 1,
                df=1 / (data_N * self.dt),
                min_freq=self.freq_min,
                max_freq=self.freq_max,
                force_backend=self.force_backend,
            )

        raise NotImplementedError(f"Unsupported analysis domain: {self.analysis_domain}")

    def find_bin_edges(
        self, times: NDArrayLike
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """
        For a given array of times, compute the edges of the bins defined by the data time step that contain the times. This is used to determine the time segments for the STFT or the frequency bins for the FD transformation.

        Args:
            times (Array): Array of times. Shape: (num_times,) or (num_bin, num_times).

        Returns:
            Tuple of (left_edges, grid_length) where left_edges are the edges of the bins containing the times, and grid_length is the number of bins spanned by the times.
        """
        times = self.xp.atleast_2d(times)

        start_times = times[:, 0]
        end_times = times[:, -1]

        if self.analysis_domain == "STFT":
            # Use integer STFT-segment indices from stft_t_arr.  digitize caps
            # right_edges_i at len(stft_t_arr)=NT so each source's individual
            # grid_length is naturally bounded to the data end.
            left_edges_i = self.xp.digitize(start_times, self.stft_t_arr)
            right_edges_i = self.xp.digitize(end_times, self.stft_t_arr)
            left_edges = self.stft_t_arr[left_edges_i - 1]
            grid_length = (right_edges_i - left_edges_i + 1) * self.nperseg
            
        elif self.analysis_domain == "FD":
            left_edges = self.xp.full(shape=start_times.shape, fill_value=self.data_t0)
            grid_length = self.xp.full(shape=start_times.shape, fill_value=self.domain_settings.N)

        return left_edges, grid_length

    def build_common_grid(
        self, times: NDArrayLike, channels: NDArrayLike
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """
        For a given array of times and corresponding channels, build a common grid for all sources based on the analysis domain (STFT or FD).

        Args:
            times (Array): Array of times. Shape: (num_bin, num_times).
            channels (Array): Array of channels corresponding to the input times. Shape: (num_bin, num_channels, num_times).
        Returns:
            Tuple of (left_edges, padded_signals) where left_edges are the edges of the bins containing the times, and padded_signals is the array of channels padded to the common-size grid.
        """
        if len(channels.shape) == 2:
            channels = channels[None, :, :]  # add a batch dimension
            times = times[None, :]

        left_edges, grid_length = self.find_bin_edges(times)
        num_bin = left_edges.shape[0]
        max_grid_length = int(grid_length.max())

        # STFT per-source safety mask: #todo not really sure we need this, will come back later
        safe_samples_per_source = None
        if self.analysis_domain == "STFT":
            NT = self.domain_settings.N // self.nperseg
            segment_dt = self.dt * self.nperseg
            t_idx = self.xp.rint(
                (left_edges - self.data_t0) / segment_dt
            ).astype(int)
            safe_samples_per_source = (
                self.xp.maximum(NT - t_idx, 0) * self.nperseg
            )[:, None, None]  # (num_bin, 1, 1) for broadcasting with grid_time_indices

        # create a common grid
        padded_signals = self.xp.zeros(
            (channels.shape[:-1] + (max_grid_length,)), dtype=channels.dtype
        )  # shape (num_bin, num_channels, max_grid_length)

        # Use rint() to guard against floating-point truncation
        grid_time_indices = self.xp.rint(
            (times[:, None, :] - left_edges[:, None, None]) / self.dt
        ).astype(int)

        # Mask off indices that would push out of bounds (just in case)
        valid = (grid_time_indices >= 0) & (grid_time_indices < max_grid_length)
        if safe_samples_per_source is not None:
            valid = valid & (grid_time_indices < safe_samples_per_source)

        batch_indices = self.xp.arange(num_bin)[:, None, None]
        channel_indices = self.xp.arange(channels.shape[1])[None, :, None]
        
        # Zero out channel values whose source time falls outside max_grid_length
        # BEFORE scattering.  valid has shape (num_bin, 1, num_times) and channels
        # has shape (num_bin, num_channels, num_times) — broadcasting works along
        # the channel axis.  This avoids a shape mismatch from applying where()
        # post-scatter on the (num_bin, num_channels, max_grid_length) output array.
        channels_to_scatter = self.xp.where(valid, channels, 0)

        # Clip indices so the scatter never writes outside the allocated array;
        # the zeroed channels above ensure those positions carry no signal.
        safe_time_indices = self.xp.clip(
            grid_time_indices, 0, max(max_grid_length - 1, 0)
        )
        padded_signals[batch_indices, channel_indices, safe_time_indices] = channels_to_scatter

        return left_edges, padded_signals

    def _pad_td_signal(
        self,
        times: NDArrayLike,
        signals: NDArrayLike,
        align_samples: int,
        target_n: int = None,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Pad time-domain arrays so the start is aligned with data_t0 and reaches a target length.

        Accepts either a single source or a batch:

        - Single:  ``times (num_times,)``,        ``signals (3, num_times)``
        - Batched: ``times (num_bin, num_times)``, ``signals (num_bin, 3, num_times)``

        Left-pads with zeros so that the number of samples between the (new) t0 and
        data_t0 is an integer multiple of ``align_samples``.  For STFT this enforces
        segment-boundary alignment (align_samples = nperseg); for FD pass align_samples=target_n to align to the full FFT length.
        so that the signal starts exactly at data_t0.

        Then, if ``target_n`` is given, right-pads with zeros so that the total number
        of samples reaches ``target_n`` (ensuring the correct ``df`` after FFT).

        For batched inputs all sources must produce the same ``n_left``; this is
        guaranteed when callers have already snapped each source's time grid to integer
        multiples of ``dt`` relative to ``data_t0``.

        Args:
            times: Time array, shape ``(num_times,)`` or ``(num_bin, num_times)``.
            signals: Signal array, shape ``(3, num_times)`` or ``(num_bin, 3, num_times)``.
            align_samples: Left-padding granularity.  The signal is extended so that
                ``round((signal_t0 - data_t0) / dt)`` becomes divisible by this value.
            target_n: If provided, right-pad to at least this many total samples.

        Returns:
            ``(padded_times, padded_signals)`` with the same leading dimensions as the inputs.
        """
        dt = self.dt

        if times.ndim == 1:
            n_to_data_t0 = round((float(times[0]) - self.data_t0) / dt)
            n_left = n_to_data_t0 % align_samples
        else:
            n_to_data_t0 = self.xp.rint((times[:, 0] - self.data_t0) / dt).astype(int)
            n_left_per_bin = n_to_data_t0 % align_samples
            assert self.xp.all(n_left_per_bin == n_left_per_bin[0]), (
                "Batched _pad_td_signal: sources produce different n_left values — "
                "ensure time grids are snapped to the dt grid before padding."
            )
            n_left = int(n_left_per_bin[0])

        N = times.shape[-1]
        n_right = 0
        if target_n is not None:
            new_n = N + n_left
            if new_n < target_n:
                n_right = target_n - new_n

        if n_left == 0 and n_right == 0:
            return times, signals

        # Pad signals on the last (time) axis, preserving all leading dims.
        pad_width = [(0, 0)] * (signals.ndim - 1) + [(n_left, n_right)]
        padded_signals = self.xp.pad(signals, pad_width, mode="constant", constant_values=0)

        # Extend the time array.
        if times.ndim == 1:
            parts = []
            if n_left > 0:
                parts.append(times[0] - self.xp.arange(n_left, 0, -1) * dt)
            parts.append(times)
            if n_right > 0:
                parts.append(times[-1] + self.xp.arange(1, n_right + 1) * dt)
            padded_times = self.xp.concatenate(parts)
        else:
            parts = []
            if n_left > 0:
                parts.append(times[:, 0:1] - self.xp.arange(n_left, 0, -1)[None, :] * dt)
            parts.append(times)
            if n_right > 0:
                parts.append(times[:, -1:] + self.xp.arange(1, n_right + 1)[None, :] * dt)
            padded_times = self.xp.concatenate(parts, axis=-1)

        return padded_times, padded_signals

    def _td_to_output_domain(
        self,
        times_in: NDArrayLike,
        signal_in: NDArrayLike,
        output_domain: str = None,
        domain_kwargs: dict = None,
    ) -> DomainBase:
        """Transform a time-domain signal to the specified output domain. If output_domain is None, defaults to self.analysis_domain.

        Args:
            times_in: Time array of the input TDSignal.
            signal_in: Signal array of the input TDSignal.
            output_domain: Target domain ('TD', 'STFT', or 'FD'). If None, defaults to self.analysis_domain.
            domain_kwargs: Extra kwargs forwarded to the domain settings constructor. If None, it will be derived from the input times and self.analysis_domain.

        Returns:
            Signal in the requested output domain.
        """
        if len(times_in.shape) > 1:
            raise NotImplementedError(
                "Multi-dimensional time arrays are not supported yet. If going through the Domain interface, treat different sources separately."
            )
        if output_domain is not None:
            output_domain = output_domain.upper()  # allow case-insensitive domain names
            use_default_domain = False
        else:
            output_domain = self.analysis_domain.upper()
            use_default_domain = True

        t0_here = times_in[0]
        dt_here = self.dt #float(times_in[2] - times_in[0])
        

        if output_domain == "TD":
            return TDSignal(
                arr=signal_in,
                settings=TDSettings(
                    t0=t0_here, dt=dt_here, N=times_in.shape[-1], force_backend=self.force_backend
                ),
            )

        elif output_domain == "STFT":
            nperseg = (
                self.nperseg if use_default_domain else round(domain_kwargs["big_dt"] / dt_here)
            )

            # We must right-pad to a multiple of nperseg, otherwise the STFT
            # framing (NT = N // nperseg) will silently truncate the end of the
            # signal (which often contains the merger!)

            n_to_data_t0 = round(float((t0_here - self.data_t0) / dt_here))
            n_left = n_to_data_t0 % nperseg
            current_padded_len = times_in.shape[-1] + n_left
            target_n = current_padded_len + (nperseg - (current_padded_len % nperseg)) % nperseg

            padded_times, padded_signal = self._pad_td_signal(
                times_in, signal_in, align_samples=nperseg, target_n=target_n
            )

            if use_default_domain:
                out_settings = self.get_output_settings(padded_times)
            else:
                out_settings = get_stft_settings(
                    padded_times, **domain_kwargs, force_backend=self.force_backend
                )

            nperseg = out_settings.get_nperseg(dt_here)
            window = tukey(nperseg, alpha=self.tukey_alpha, xp=self.xp)

        elif output_domain == "FD":
            # Pad to full data length FIRST, then window at that length.
            # This matches the data-side processing (tukey(N_data) → FFT).
            if use_default_domain:
                df = 1 / self.Tobs
            else:
                df = domain_kwargs["df"]
            N_td_target = round(1 / (df * dt_here))
            padded_times, padded_signal = self._pad_td_signal(
                times_in, signal_in, align_samples=N_td_target, target_n=N_td_target
            )

            window = tukey(padded_times.shape[-1], alpha=self.tukey_alpha, xp=self.xp)
            if use_default_domain:
                out_settings = self.get_output_settings(padded_times)
            else:
                out_settings = FDSettings(**domain_kwargs, force_backend=self.force_backend)

        else:
            raise ValueError(
                f"output_domain must be either 'TD', 'STFT', or 'FD'. "
                f"'WDM' is not supported yet. Got: {output_domain}."
            )

        padded_td_signal = TDSignal(
            arr=padded_signal,
            settings=TDSettings(
                t0=padded_times[0],
                dt=dt_here,
                N=padded_times.shape[-1],
                force_backend=self.force_backend,
            ),
        )
        return padded_td_signal.transform(out_settings, window=window)

    def fft(
        self, start_times: NDArrayLike, signal_in: NDArrayLike
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """
        Transform pre-padded time domain data to the FD basis.

        Signals must be pre-padded (via build_common_grid) before calling this
        method — no inline padding is done here.

        Args:
            start_times (Array): Start time for each binary, shape `(num_binaries,)`.
                Not used in the FD transform but kept for signature consistency with `stft`.
            signal_in (Array): Time domain input signal with shape `(num_binaries, num_channels, num_times)`

        Returns:
            signal_out (Array): The transformed signal with shape `(num_binaries, num_channels, num_freqs)`
            start_freqs (Array): Starting frequencies for the likelihood calculation, shape `(num_binaries,)`
        """
        num_binaries = signal_in.shape[0]
        n = signal_in.shape[-1]
        
        outer_shape = signal_in.shape[:-1]

        window = tukey(n, alpha=self.tukey_alpha, xp=self.xp)
        # In-place windowing: avoids a (num_binaries, num_channels, N_data) copy.
        # signal_in is the padded array from build_common_grid and is not used after this call.
        signal_in *= window

        # Loop over sources instead of one batched rfft to prevent oom errors from large FFTs
        n_freqs_full = n // 2 + 1
        signal_fd = self.xp.empty(
            (*outer_shape, n_freqs_full), dtype=self.xp.complex128
        )

        fft_batch_size = getattr(self, "fft_batch_size", 1)  

        for i in range(0, num_binaries, fft_batch_size):
            start = i
            end = min(i + fft_batch_size, num_binaries)

            signal_fd[start:end] = self.xp.fft.rfft(signal_in[start:end], axis=-1) * self.dt

        freqs = self.xp.fft.rfftfreq(n, d=self.dt)

        #keep = (freqs >= self.freq_min) & (freqs <= self.freq_max)

        # Find the integer indices corresponding to the frequency bounds
        start_idx = self.xp.searchsorted(freqs, self.freq_min)
        end_idx = self.xp.searchsorted(freqs, self.freq_max, side='right')

        # Slicing creates a view, avoiding the copy
        signal_out = signal_fd[..., start_idx:end_idx]

        start_freqs = self.xp.full(shape=num_binaries, fill_value=freqs[start_idx])

        return signal_out, start_freqs
        # signal_out = signal_fd[..., keep] #try to avoid this copy

        # start_freqs = self.xp.full(shape=num_binaries, fill_value=self.xp.min(freqs[keep]))

        # return signal_out, start_freqs

    def stft(
        self,
        start_times: NDArrayLike,
        signal_in: NDArrayLike,
    ) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """
        Transform pre-padded time domain data to the STFT basis.

        Args:
            start_times (Array): Start time for each binary, shape `(num_binaries,)`.
            signal_in (Array): Time domain input signal with shape `(num_binaries, num_channels, num_times)`

        Returns:
            signal_out (Array): The transformed signal with shape `(num_binaries, num_channels, num_stft_times, num_freqs)`
            start_freqs (Array): Starting frequencies for the likelihood calculation, shape `(num_binaries,)`
            start_times (Array): Starting times for the likelihood calculation, shape `(num_binaries,)`
        """
        num_binaries, num_channels = signal_in.shape[:2]
        signal_in = signal_in.reshape(
            num_binaries, num_channels, -1, self.nperseg
        )  # (num_binaries, num_channels, num_segments, num_times_per_segment)

        signal_out, start_freqs = self.fft(start_times, signal_in)

        return signal_out, start_freqs, start_times

    def __call__(self, *args, **kwargs):
        """Generate the waveform and return the signal in the analysis domain.

        Sky/response parameters may be supplied either as explicit keyword
        arguments or as the last N positional arguments; ``compute_tdi_channels``
        on each subclass is responsible for peeling them off ``*args``.
        """
        times, channels = self.compute_tdi_channels(*args, **kwargs)
        left_edges, padded = self.build_common_grid(times, channels)
        return self.transform_to_domain(left_edges, padded)

    def get_signals_for_residuals(self, *args, **kwargs) -> DomainBaseArray:
        """Generate per-source domain-wrapped signals for residual operations.

        Sky/response parameters may be supplied either as explicit keyword
        arguments or as the last N positional arguments.

        Returns:
            :class:`DomainBaseArray` of per-source signals, or a single domain
            signal when ``times`` is 1-D (single-source path).
        """
        times, channels = self.compute_tdi_channels(*args, **kwargs)
        if times.ndim == 2:
            return DomainBaseArray([
                self._td_to_output_domain(times_in=times[i], signal_in=channels[i])
                for i in range(channels.shape[0])
            ])
        return self._td_to_output_domain(times_in=times, signal_in=channels)


class TDPyResponseWaveformBase(TDWaveformBase):
    """
    Base class for a time-domain waveform that uses the `pyResponseTDI` class for the TDI response calculation.
    """

    def __init__(
        self,
        waveform_t0: float,
        data_td_settings: TDSettings,
        tdi_generation: str = "2nd generation",
        tdi_channels: str = "XYZ",
        sampling_frequency: float = 0.4,
        orbits: Orbits = None,
        order: int = 35,
        tukey_alpha: float = 0.01,
        stft_dt: float = None,
        freq_min: float = 1e-5,
        freq_max: float = 1.0,
        fft_batch_size: int = 1,
        signal_duration: float = None,
        buffer_time: int = 5000,
        run_async: bool = False,
        force_backend: str = "cpu",
    ) -> None:

        super().__init__(
            waveform_t0=waveform_t0,
            data_td_settings=data_td_settings,
            tdi_generation=tdi_generation,
            tdi_channels=tdi_channels,
            sampling_frequency=sampling_frequency,
            orbits=orbits,
            tukey_alpha=tukey_alpha,
            stft_dt=stft_dt,
            freq_min=freq_min,
            freq_max=freq_max,
            fft_batch_size=fft_batch_size,
            force_backend=force_backend,
        )

        if signal_duration is None:
            signal_duration = self.Tobs

        num_points = int(signal_duration / self.dt)

        self.response = pyResponseTDI(
            sampling_frequency=sampling_frequency,
            num_pts=num_points,
            order=order,
            orbits=orbits,
            tdi=tdi_generation,
            tdi_chan=tdi_channels,
            force_backend=force_backend,
        )

        self.buffer_time = buffer_time
        self.run_async = run_async

    @property
    def wrapper_kwargs(self) -> dict:
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""
        base_kwargs = super().wrapper_kwargs
        base_kwargs.update(
            {
                "order": self.response.order,
                "signal_duration": self.response.num_pts * self.dt,
                "buffer_time": self.buffer_time,
                "run_async": self.run_async,
            }
        )
        return base_kwargs

    def wave_gen(
        self, *args, **kwargs
    ) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """Generate the waveform for a single source.

        Returns:
            Tuple of (t_arr, h_plus, h_cross).

        """
        raise NotImplementedError("wave_gen method must be implemented in subclass.")

    def wave_gen_batch(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate waveforms for a batch of sources.

        Subclasses that support batched waveform generation should override this
        method to return pre-masked, padded arrays. The batch loop in
        ``_call_batched`` will apply per-source masking and the TDI response.

        Returns:
            Tuple of (times_batch, h_plus_batch, h_cross_batch),
            each of shape (Nbatch, Ntimes).

        """
        raise NotImplementedError(
            "wave_gen_batch is not implemented for this waveform. "
            "Batched calls require implementing wave_gen_batch in the subclass."
        )

    # def _apply_response_single(
    #     self,
    #     t_arr: NDArrayLike,
    #     h_plus: NDArrayLike,
    #     h_cross: NDArrayLike,
    #     ra: float,
    #     dec: float,
    #     merger_time: float,
    # ) -> Tuple[NDArrayLike, NDArrayLike]:
    #     """Apply the TDI response to a single source and return a TDSignal.

    #     Args:
    #         t_arr: Time array relative to zero (output of wave_gen).
    #         h_plus: Plus polarization.
    #         h_cross: Cross polarization.
    #         ra: Right ascension in radians.
    #         dec: Declination in radians.
    #         merger_time: Time of merger in seconds (relative to waveform_t0).

    #     Returns:
    #         Tuple of (times, channels) where times is the time array after shifting and padding, and channels is the TDI response with shape (num_channels, num_times).
    #     """
    #     shifted_t_arr = t_arr + merger_time + self.waveform_t0
    #     # add 500 seconds to the end to prevent problems with the response

    #     # pad both sides with zeros by num_pad
    #     num_pad = int(self.buffer_time / self.dt)

    #     shifted_t_arr = self.xp.concatenate(
    #         [
    #             #shifted_t_arr[0] - self.dt * self.xp.arange(1, num_pad + 1),
    #             shifted_t_arr,
    #             shifted_t_arr[-1] + self.dt * self.xp.arange(1, num_pad + 1),
    #         ]
    #     )

    #     h_plus = self.xp.pad(h_plus, (0, num_pad), mode="edge")
    #     h_cross = self.xp.pad(h_cross, (0, num_pad), mode="edge")

    #     self.response.num_pts = shifted_t_arr.shape[-1]

    #     strain = h_plus + 1j * h_cross

    #     self.response.get_projections(
    #         strain, lam=ra, beta=dec, t0=float(shifted_t_arr[0]), t_buffer=self.buffer_time, run_async=self.run_async
    #     )
    #     tdis = self.xp.array(self.response.get_tdi_delays(run_async=self.run_async))

    #     # trim the invalid points
    #     shifted_t_arr = shifted_t_arr[:-num_pad]
    #     tdis[:, :num_pad] = 0.0  # zero out the corrupted points at the start
    #     tdis = tdis[:, :-num_pad]

    #     # now shift the time arrays so that the abs(t_arr[0] - data_t0) is an integer multiple of dt
    #     t_arr_shift = (self.data_t0 - shifted_t_arr[0]) % self.dt
    #     shifted_t_arr += t_arr_shift

    #     # now remove everything before the start of the data
    #     start_ind = int((self.data_t0 - shifted_t_arr[0]) / self.dt)
    #     if start_ind > 0:
    #         shifted_t_arr = shifted_t_arr[start_ind:]
    #         tdis = tdis[:, start_ind:]

    #     return shifted_t_arr, tdis


    def _apply_response(
        self,
        t_arr: NDArrayLike,
        h_plus: NDArrayLike,
        h_cross: NDArrayLike,
        ra: float | NDArrayLike,
        dec: float | NDArrayLike,
        merger_time: float | NDArrayLike,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Apply the TDI response to a batch of sources.

        Args:
            t_arr: Time array relative to zero, shape (Ntimes,) or (Nbatch, Ntimes).
            h_plus: Plus polarization, shape (Ntimes,) or (Nbatch, Ntimes).
            h_cross: Cross polarization, shape (Ntimes,) or (Nbatch, Ntimes).
            ra: Right ascension in radians, float or shape (Nbatch,).
            dec: Declination in radians, float or shape (Nbatch,).
            merger_time: Time of merger in seconds (relative to waveform_t0), float or shape (Nbatch,).

        Returns:
            Tuple of (times_batch, channels_batch) where times_batch is the time array after shifting and padding with shape (Nbatch, Ntimes), and channels_batch is the TDI response with shape (Nbatch, num_channels, num_times).
        """
        # If we know which device the response/orbits live on, ensure that device is
        # current for the entire computation.  This prevents an illegal-memory-access
        # when a previous caller (e.g. template_likelihood on a GPU-2 walker) left a
        # different device current and cudaMalloc inside get_response would then
        # allocate orbits_gpu on the wrong GPU while n_arr/ltt_arr/x_arr still
        # address GPU-0 memory.
        # _response_device_id = getattr(self, '_response_device_id', None)
        # if _response_device_id is not None:
        #     _saved_device = self.xp.cuda.runtime.getDevice()
        #     if _saved_device != _response_device_id:
        #         self.xp.cuda.runtime.setDevice(_response_device_id)
        # else:
        #     _saved_device = None

        single_source = isinstance(ra, float)

        ra = self.xp.atleast_1d(ra)
        dec = self.xp.atleast_1d(dec)
        merger_time = self.xp.atleast_1d(merger_time)

        t_arr = self.xp.atleast_2d(t_arr)
        h_plus = self.xp.atleast_2d(h_plus)
        h_cross = self.xp.atleast_2d(h_cross)

        shifted_t_arr = t_arr + self.xp.asarray(merger_time)[:, None] + self.waveform_t0
        # add 500 seconds to the end to prevent problems with the response

        # pad with zeros by num_pad
        num_pad = int(self.buffer_time / self.dt)

        pad_idx_right = self.xp.arange(1, num_pad + 1)[None, :]
        pad_idx_left = self.xp.arange(num_pad, 0, -1)[None, :]

        shifted_t_arr = self.xp.concatenate(
            [
                #shifted_t_arr[:, 0:1] - self.dt * pad_idx_left,
                shifted_t_arr,
                shifted_t_arr[:, -1:] + self.dt * pad_idx_right,
            ],
            axis=-1,
        )

        # condition the signal with a small taper at the start to mitigate edge effects in the response
        num_orig_pts = h_plus.shape[-1]
        taper_points = num_pad
        alpha = taper_points / num_orig_pts
        window_orig = tukey(num_orig_pts, alpha=alpha, xp=self.xp)
        window_orig[num_orig_pts//2:] = 1.0  # Only taper the start!

        h_plus = h_plus * window_orig[None, :]
        h_cross = h_cross * window_orig[None, :]

        # Pad zeros to both the front (buffer) and the back
        h_plus = self.xp.pad(h_plus, ((0, 0), (0, num_pad)), mode="constant", constant_values=0.0)
        h_cross = self.xp.pad(h_cross, ((0, 0), (0, num_pad)), mode="constant", constant_values=0.0)

        num_pts = shifted_t_arr.shape[-1]
        self.response.num_pts = num_pts

        strain = h_plus + 1j * h_cross

        # Diagnostic + safety guard: the response CUDA kernel reads
        # input_in[batch_ind * num_inputs + jj] where `jj` derives from
        # delays computed via orbits.get_pos(t, ...). If `shifted_t_arr[:, 0]`
        # is outside the orbit time range (or contains NaN/Inf), the
        # extrapolated orbit positions yield huge delays → out-of-bounds
        # read → `cudaErrorIllegalAddress` at LISAResponse.cu:758.
        # _data_time_check only guards the upper bound; we add a symmetric
        # check here and log per-source diagnostics so the offending source
        # is identified at the Python level before the kernel ever fires.
        if DEBUG_MODE:   
            try:
                _orbit_t_min = float(self.response.response_orbits.sc_t_base.min())
                _orbit_t_max = float(self.response.response_orbits.sc_t_base.max())
                _ltt_t_min = float(self.response.response_orbits.ltt_t.min())
                _ltt_t_max = float(self.response.response_orbits.ltt_t.max())
            except Exception:
                _orbit_t_min = _orbit_t_max = _ltt_t_min = _ltt_t_max = None

            _t0_arr = shifted_t_arr[:, 0]
            _t_last_arr = shifted_t_arr[:, -1]
            _t0_min = float(_t0_arr.min())
            _t0_max = float(_t0_arr.max())
            _t_last_min = float(_t_last_arr.min())
            _t_last_max = float(_t_last_arr.max())
            _has_nan = bool(
                self.xp.isnan(strain).any() or self.xp.isnan(shifted_t_arr).any()
            )
            _has_inf = bool(
                self.xp.isinf(strain).any() or self.xp.isinf(shifted_t_arr).any()
            )

            if _has_nan or _has_inf:
                logger.debug(
                "_apply_response: batch=%d num_pts=%d t0=[%.6e, %.6e] t_last=[%.6e, %.6e] "
                "orbit_sc_t=[%s, %s] orbit_ltt_t=[%s, %s] merger_time=%s nan=%s inf=%s",
                int(shifted_t_arr.shape[0]), num_pts, _t0_min, _t0_max, _t_last_min, _t_last_max,
                f"{_orbit_t_min:.6e}" if _orbit_t_min is not None else "?",
                f"{_orbit_t_max:.6e}" if _orbit_t_max is not None else "?",
                f"{_ltt_t_min:.6e}" if _ltt_t_min is not None else "?",
                f"{_ltt_t_max:.6e}" if _ltt_t_max is not None else "?",
                merger_time.tolist() if hasattr(merger_time, "tolist") else merger_time,
                _has_nan, _has_inf,
            )
            
                raise ValueError(
                    f"_apply_response: NaN/Inf detected before response kernel "
                    f"(nan={_has_nan}, inf={_has_inf}, merger_time={merger_time}, t0=[{_t0_min}, {_t0_max}])."
                )

            if _orbit_t_min is not None and (
                _t0_min < _orbit_t_min or _t_last_max > _orbit_t_max
            ):
                raise ValueError(
                    f"_apply_response: requested time window [{_t0_min:.6e}, {_t_last_max:.6e}] "
                    f"falls outside orbit sc_t range [{_orbit_t_min:.6e}, {_orbit_t_max:.6e}]. "
                    f"This would cause the response CUDA kernel to read out-of-bounds. "
                    f"merger_time={merger_time}, t_arr[0]={float(t_arr[:, 0].min()):.6e}."
                )

        self.response.get_projections(
            strain, lam=ra, beta=dec, t0=shifted_t_arr[:, 0], t_buffer=self.buffer_time, run_async=self.run_async
        )

        tdis = self.xp.array(self.response.get_tdi_delays(run_async=self.run_async)) # (Nbatch, num_channels, Ntimes) if batched else (num_channels, Ntimes)
        if len(tdis.shape) == 3:
            tdis = tdis.transpose(1, 0, 2)

        tdis = tdis[..., :-num_pad] # remove the padded points
        tdis[..., :num_pad] = 0.0  # zero out the corrupted points at the start
        shifted_t_arr = shifted_t_arr[:, :-num_pad]

        t_arr_shift = (self.data_t0 - shifted_t_arr[:, 0]) % self.dt
        shifted_t_arr += t_arr_shift[:, None]

        start_inds = self.xp.maximum(
            0, self.xp.rint((self.data_t0 - shifted_t_arr[:, 0]) / self.dt).astype(int)
        )
        start_ind = int(start_inds.max())

        if start_ind > 0:
            shifted_t_arr = shifted_t_arr[:, start_ind:]
            tdis = tdis[..., start_ind:]
        
        # now remove the extra time dimensions if we only had one source (to be consistent with the single-source path)
        if single_source:
            shifted_t_arr = shifted_t_arr[0]

        # if _saved_device is not None and _saved_device != _response_device_id:
        #     self.xp.cuda.runtime.setDevice(_saved_device)

        return shifted_t_arr, tdis

    def _call_single(
        self,
        *args,
        ra: float,
        dec: float,
        merger_time: float,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Handle single-source waveform generation and return a Tuple of times and channels."""

        t_arr, h_plus, h_cross = self.wave_gen(*args, ra, dec, merger_time, **kwargs)

        times, channels = self._apply_response(t_arr, h_plus, h_cross, ra, dec, merger_time)

        return times, channels

    def _call_batched(
        self,
        *args,
        ra: NDArrayLike,
        dec: NDArrayLike,
        merger_time: NDArrayLike,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Handle batched waveform generation and return a Tuple of times and channels.

        Loops over the batch dimension for the TDI response (which does not support
        batching natively), then optionally projects all signals onto a common STFT
        grid when ``self.force_uniform_stft`` is True.
        """
        times_batch, hplus_batch, hcross_batch = self.wave_gen_batch(*args, ra, dec, merger_time, **kwargs)

        return self._apply_response(times_batch, hplus_batch, hcross_batch, ra, dec, merger_time)

    def compute_tdi_channels(
        self,
        *args,
        ra: float | NDArrayLike = None,
        dec: float | NDArrayLike = None,
        merger_time: float | NDArrayLike = None,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Time domain TDI channels computation. In the case of multiple sources, the TDI response is applied sequentially to each source and the results are stacked together.

        Args:
            *args: Arguments for the wave_gen / wave_gen_batch method.
            ra: Right ascension in radians.  Scalar for single source, 1-D array for batch.
            dec: Declination in radians.  Same shape as ``ra``.
            merger_time: Time of merger in seconds.  Same shape as ``ra``.
            **kwargs: Keyword arguments for the wave_gen / wave_gen_batch method.

        Returns:
            Tuple of (times, channels) where `times` is the time array after shifting and padding, and `channels` are the TDI variables with shape (num_channels, num_times) for single source or (num_bin, num_channels, num_times) for batch.
        """
        if ra is None or dec is None or merger_time is None:
            *args, ra, dec, merger_time = args

        if np.ndim(ra) >= 1:
            return self._call_batched(*args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        return self._call_single(*args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)



class TDTDIOnFlyWaveformBase(TDWaveformBase):
    """
    Base class for a time domain waveform that computes the LISA response with the "tdi on the fly" method.

    Args:
        waveform_t0: Initial time for the waveform in seconds. It will be added to the sampled merger time to get the absolute time for the waveform generation.
        dt: float,
        Tobs: float,
        data_t0: float = None,
        tukey_alpha: float = 0.01,
        force_backend: str = "cpu",
    """

    def __init__(
        self,
        waveform_t0: float,
        data_td_settings: TDSettings,
        tdi_generation: str = "2nd generation",
        tdi_channels: str = "XYZ",
        sampling_frequency: float = 0.4,
        orbits: Orbits = None,
        tukey_alpha: float = 0.01,
        stft_dt: float = None,
        freq_min: float = 0.0,
        freq_max: float = 1.0,
        fft_batch_size: int = 1,
        zero_inclination: bool = False,
        force_backend: str = "cpu",
    ) -> None:

        if tdi_channels != "XYZ":
            raise NotImplementedError(
                "Only XYZ channels are supported for TDI on-the-fly waveforms for now."
            )

        super().__init__(
            waveform_t0=waveform_t0,
            data_td_settings=data_td_settings,
            tdi_generation=tdi_generation,
            tdi_channels=tdi_channels,
            sampling_frequency=sampling_frequency,
            orbits=orbits,
            tukey_alpha=tukey_alpha,
            stft_dt=stft_dt,
            freq_min=freq_min,
            freq_max=freq_max,
            fft_batch_size=fft_batch_size,
            force_backend=force_backend,
        )

        self.zero_inclination = zero_inclination

    @property
    def wrapper_kwargs(self):
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""
        base_kwargs = super().wrapper_kwargs
        base_kwargs.update(
            {
                "zero_inclination": self.zero_inclination,
            }
        )
        return base_kwargs

    @property
    def max_length(self) -> int:
        """
        maximum number of evaluation time points that can be stored in a gpu register.
        These may not be enough in some regions of the parameter space
        """
        return 2_000

    def get_amp_phase(
        self,
        *args,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """
        Generate amplitude and phase arrays for each mode of a batch of sources.
        Returns also the time array.

        Returns:
            Tuple of (t_arr, amp, phase).

        """
        raise NotImplementedError("amp_phase_gen method must be implemented in subclass.")

    def process_amp_phase(
        self, amp: NDArrayLike, phase: NDArrayLike
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """
        Process the amplitude and phase arrays to be fed to the TDI on-the-fly response generator.

        Parameters
        ----------
        amp: Amplitude array of shape (Nbatch, num_modes, num_times).
        phase: Phase array of shape (Nbatch, num_modes, num_times).

        Returns
        -------
        Tuple of (processed_amp, processed_phase).
        """

        raise NotImplementedError("process_amp_phase method must be implemented in subclass.")

    def stack_parameter(self, param: np.ndarray, num_modes: int) -> NDArrayLike:
        """
        Stack a parameter array for use in the TDI on-the-fly response generator.
        Given a parameter array of shape (Nbatch,), stack it to shape (Nbatch * num_modes,) by repeating each entry num_modes times. This is needed to match the expected input shape for the TDI on-the-fly response generator when using multiple modes per source.
        """
        param = self.xp.asarray(param)
        return self.xp.repeat(param, num_modes)

    def get_tdi_buffers(self, delta_t: np.ndarray) -> Tuple[int, int, float, float]:
        """
        Get the number of buffer points to add at the beginning and end of the time array based on the TDI buffer time and the maximum time step in the input time array.

        Args:
            delta_t (Array): Array of time steps between consecutive time points, shape (num_binaries * num_modes, num_times - 1).
        Returns:
            Tuple of (start_buffer, end_buffer, left_dt, right_dt) in number of points and time steps.
        """

        left_dt = self.xp.min(delta_t[:, 0])
        right_dt = self.xp.min(delta_t[:, -1])

        start_buffer = max(int(self.tdi_buffer_time / left_dt), 1)
        end_buffer = max(int(self.tdi_buffer_time / right_dt), 1)

        return start_buffer, end_buffer, left_dt, right_dt

    def pad(
        self,
        input_times: NDArrayLike,
        input_amplitudes: NDArrayLike,
        input_phases: NDArrayLike,
    ) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """
        Add a 500 s buffer at both sides to make sure that we can compute tdi on the times we are actually interested in.

        Args:
            input_times (Array): input time array, shape (num_binaries * num_modes, num_times)
            input_amplitudes (Array): input amplitude array, shape (num_binaries * num_modes, num_times)
            input_phases (Array): input phase array, shape (num_binaries * num_modes, num_times)

        Returns:
            Padded time, amplitude and phase arrays with shape (num_binaries * num_modes, num_times + 2 * buffer_length).
        """
        delta_t = self.xp.diff(input_times, axis=-1)

        pad_length_left, pad_length_right, left_dt, right_dt = self.get_tdi_buffers(delta_t)

        padded_times = self.xp.concatenate(
            [
                input_times[:, 0:1] - self.xp.arange(pad_length_left, 0, -1) * left_dt,
                input_times,
                input_times[:, -1:] + self.xp.arange(1, pad_length_right + 1) * right_dt,
            ],
            axis=-1,
        )

        padded_amplitudes = self.xp.pad(
            input_amplitudes, ((0, 0), (pad_length_left, pad_length_right)), mode="edge"
        )
        padded_phases = self.xp.pad(
            input_phases, ((0, 0), (pad_length_left, pad_length_right)), mode="edge"
        )

        return padded_times, padded_amplitudes, padded_phases

    def get_evaluation_times(self, input_times: NDArrayLike) -> NDArrayLike:
        """
        Get the time array on which to evaluate the TDI on-the-fly response. By default, this uses the same as the input time array from the amplitude and phase generation, but subclasses can override this method to define a different evaluation grid if needed (e.g. a regular grid).

        Args:
            input_times (Array): times used for amp/phase generation, shape (num_binaries  * num_modes, num_times)

        Returns:
            evaluation times of shape (num_binaries * num_modes, num_eval_times).
        """
        delta_t = self.xp.diff(input_times, axis=-1)

        start_buffer, end_buffer, _, _ = self.get_tdi_buffers(delta_t)

        in_times = input_times[:, start_buffer:-end_buffer]
        # evaluation_times = self.xp.concatenate([in_times[:, :1], in_times[:, -self.max_length :]], axis=-1)
        evaluation_times = in_times[:, -self.max_length :]

        return evaluation_times

    def get_dense_times(self, eval_times: NDArrayLike) -> NDArrayLike:
        """
        Get a dense time array on which to evaluate the TDI on-the-fly response. This can be used to ensure that the output response is sampled on a regular grid, even if the input amplitude and phase arrays are sampled irregularly.
        """

        t0 = self.data_t0
        dt = self.dt
        segment_dt = self.nperseg * dt if self.analysis_domain == "STFT" else dt

        # Use floor to ensure we don't truncate the start of the evaluation times,
        # which is crucial for not cutting off SNR in STFT block segmentations.
        start_times = t0 + self.xp.floor((eval_times[:, 0] - t0) / segment_dt) * segment_dt

        max_duration = self.xp.max(eval_times[:, -1] - start_times)
        num_times = int(self.xp.ceil(max_duration / self.dt)) + 1

        if self.analysis_domain == "STFT":
            # make sure num_times is a multiple of nperseg for the STFT case
            num_times = int(np.ceil(num_times / self.nperseg) * self.nperseg)

        return start_times[:, None] + self.xp.arange(num_times)[None, :] * self.dt

    def compute_tdi_channels(
        self,
        *args,
        inclination: NDArrayLike = None,
        psi: NDArrayLike = None,
        ra: NDArrayLike = None,
        dec: NDArrayLike = None,
        merger_time: NDArrayLike = None,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """
        Generate the on-the-fly response for a batch of sources, and return the computed TDI channels.

        Args:
            *args: positional arguments for the amplitude and phase generation method.
            inclination (Array): Inclination angles for the sources, shape (Nbatch,). If `self.zero_inclination`, this will be set to zero when passed to the response to avoid double-counting the spherical harmonic contribution.
            psi (Array): Polarization angles for the sources, shape (Nbatch,).
            ra (Array): Right ascension for the sources, shape (Nbatch,).
            dec (Array): Declination for the sources, shape (Nbatch,).
            merger_time (Array): merger time with respect to `waveform_t0`, shape (Nbatch,).
            **kwargs: keyword arguments for the amplitude and phase generation method.

        Returns:
            Tuple of (dense_times, tdi_channels) where `dense_times` has shape (Nbatch, num_times) and `tdi_channels` has shape (Nbatch, num_TDI_channels, num_times).
        """
        if inclination is None:
            *args, inclination, psi, ra, dec, merger_time = args

        # Ensure all inputs are at least 1-D for batch processing
        args = tuple(self.xp.atleast_1d(self.xp.asarray(a)) for a in args)
        inclination = self.xp.atleast_1d(self.xp.asarray(inclination))
        psi = self.xp.atleast_1d(self.xp.asarray(psi))
        ra = self.xp.atleast_1d(self.xp.asarray(ra))
        dec = self.xp.atleast_1d(self.xp.asarray(dec))
        merger_time = self.xp.atleast_1d(self.xp.asarray(merger_time))

        # step 1: generate the amplitude and phase arrays for each mode of each source:
        input_times, input_amplitudes, input_phases = self.get_amp_phase(
            *args, inclination, psi, ra, dec, merger_time, **kwargs
        )

        input_amplitudes, input_phases = self.process_amp_phase(input_amplitudes, input_phases)

        input_times = (
            input_times + merger_time[:, None] + self.waveform_t0
        )  # add the waveform t0 to get the absolute time for the TDI response generation

        num_binaries, num_modes = input_amplitudes.shape[:2]
        num_sub = num_binaries * num_modes

        if len(input_times.shape) == 2 and (input_times.shape[0] == num_binaries):
            input_times = self.xp.repeat(input_times, num_modes, axis=0)
        assert input_times.shape[0] == num_sub, "Time array mismatch"

        input_amplitudes = input_amplitudes.reshape(num_sub, -1)
        input_phases = input_phases.reshape(num_sub, -1)
        input_times = input_times.reshape(num_sub, -1)

        # step 2: Define the time array on which we are gonna evaluate the respone. it will be taken from the non padded one.
        evaluation_times = self.get_evaluation_times(input_times)

        # now padd with 500 seconds on each side
        padded_times, padded_amplitudes, padded_phases = self.pad(
            input_times, input_amplitudes, input_phases
        )

        tdi_generator = TDTDIonTheFly(
            evaluation_times,
            padded_amplitudes,
            padded_phases,
            sampling_frequency=self.sampling_frequency,
            num_sub=num_sub,
            t_input=padded_times,
            tdi_config=self.tdi_config,
            orbits=self.orbits,
            force_backend=self.force_backend,
        )

        # step 3: locate and repeat the sky parameters, polarization and inclination
        if self.zero_inclination:
            inclination_in = self.xp.zeros(num_sub)
        else:
            inclination_in = self.stack_parameter(inclination, num_modes)
        psi_in = self.stack_parameter(psi, num_modes)
        ra_in = self.stack_parameter(ra, num_modes)
        dec_in = self.stack_parameter(dec, num_modes)

        # step 4: generate the TDI on-the-fly response for each source and mode
        tdi_spline = tdi_generator(inclination_in, psi_in, ra_in, dec_in, return_spline=True)

        # step 5: get the desired dense time array on which to evaluate the response
        dense_times = self.get_dense_times(
            evaluation_times
        )  # shape nbinaries * num_modes, num_times

        # step 5.5: extend TDI spline domain to safely taper the signal to zero.
        # STFT boundary snapping in `get_dense_times` can expand the domain by thousands of
        # seconds. Anchoring the spline's immediate edges at zero ensures that out-of-bounds
        # evaluation flawlessly returns 0.0 without causing massive polynomial extrapolation ringing.
        # todo check this. I've stopped looking at TOF for the moment
        if self.analysis_domain == "STFT":
            anchor_time_end = tdi_spline.x[:, -1:] + self.dt  # (num_sub, 1)
            anchor_time_start = tdi_spline.x[:, 0:1] - self.dt

            tdi_spline.x = self.xp.concatenate(
                [anchor_time_start, tdi_spline.x, anchor_time_end], axis=-1
            )
            tdi_spline.tdi_amp = self.xp.concatenate(
                [
                    self.xp.zeros((num_sub, self.tdi_config.nchannels, 1)),
                    tdi_spline._tdi_amp,
                    self.xp.zeros((num_sub, self.tdi_config.nchannels, 1)),
                ],
                axis=-1,
            )
            tdi_spline.tdi_phase = self.xp.concatenate(
                [
                    tdi_spline._tdi_phase[:, :, 0:1],
                    tdi_spline._tdi_phase,
                    tdi_spline._tdi_phase[:, :, -1:],
                ],
                axis=-1,
            )
            tdi_spline.phase_ref = self.xp.concatenate(
                [
                    tdi_spline._phase_ref[:, 0:1],
                    tdi_spline._phase_ref,
                    tdi_spline._phase_ref[:, -1:],
                ],
                axis=-1,
            )

        # step 6: evaluate the TDI response on the dense time array
        tdi_out_raw = tdi_spline.eval_tdi(dense_times, error_out_of_bounds=False)

        tdi_out = tdi_out_raw.reshape(num_binaries, num_modes, self.tdi_config.nchannels, -1)

        tdi_channels = self.xp.sum(
            tdi_out, axis=1
        )  # sum over modes to get the total response for each binary
        dense_times = dense_times.reshape(num_binaries, num_modes, -1)[
            :, 0
        ]  # reshape and take the time array from the first mode (all modes share the same time array)

        return dense_times, tdi_channels
