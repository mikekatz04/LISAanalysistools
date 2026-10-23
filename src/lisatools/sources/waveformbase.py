# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments
# pyling: diable=too-many-keywords-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-instance-attributes, too-few-public-methods, too-many-locals
# pylint: disable=too-many-positional-arguments, too-many-statements, line-too-long
"""Base classes for source waveform generators, including the LISA response step.

Classes:

- :class:`AETTDIWaveform`, :class:`SNRWaveform`: minimal interfaces (``dt``, ``f_arr``,
  ``df``) for the simple generators the calculation controllers
  (:mod:`lisatools.sources.utils`) consume.
- :class:`TDWaveformBase`: a time-domain generator that returns TDI channels
  (:meth:`~TDWaveformBase.compute_tdi_channels`) and projects them onto the analysis domain
  (FD, STFT, or WDM through a :class:`~lisatools.domains.WDMSettings` target).
- :class:`TDPyResponseWaveformBase`: applies the LISA response with
  :class:`~lisatools.response.directresponse.pyResponseTDI` to ``(h+, hx)`` from
  ``wave_gen`` / ``wave_gen_batch``. Base of the MBH
  :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIWaveform` family, which the 9mo / 12mo
  runs use (``MBH_LIKELIHOOD=batched``: residual rebuilds through
  :meth:`~TDWaveformBase.get_signals_for_residuals`, scoring through the grid-aligned
  subclasses and :class:`~lisatools.sources.batching.MBHWindowedWDMSignalGen`).
- :class:`TDTDIOnFlyWaveformBase`: the TDI-on-the-fly variant
  (:class:`~lisatools.response.tdionfly.TDTDIonTheFly` on per-mode amplitude / phase). Base of
  :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIOnFlyWaveform`; the MBH
  ``use_tdionfly`` path is off in production.

``DEBUG_MODE`` (module constant, default ``False``) turns on the pre-response NaN / orbit-span
checks in :meth:`TDPyResponseWaveformBase._apply_response`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

# stft_tof merge: fastlisaresponse is retired -- these live in
# lisatools.response since Phase 3B/3E.
from ..response import pyResponseTDI
from ..response.tdiconfig import TDIConfig
from ..response.tdionfly import TDTDIonTheFly

from ..domains import (
    DomainSettingsBase,
    DomainBase,
    DomainBaseArray,
    FDSettings,
    TDSettings,
    TDSignal,
    WDMSettings,
    get_stft_settings,
    pad_td_signal,
    place_td_signal_on_grid,
)
from ..utils.parallelbase import LISAToolsParallelModule
from ..utils.typing import NDArrayLike, ArrayModule
from ..utils.utility import tukey
from ..utils.exceptions import BatchNotLaunchable

if TYPE_CHECKING:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        import numpy as cp

    from lisatools.detector import Orbits, EqualArmlengthOrbits

logger = logging.getLogger(__name__)

DEBUG_MODE = False

class AETTDIWaveform(ABC):
    """Base class for an AET TDI Waveform.

    Interface for the generators the calculation controllers
    (:class:`~lisatools.sources.utils.CalculationController`) call: a subclass is callable on
    its parameters, returns the TDI channels, and exposes :attr:`dt`, :attr:`f_arr` and
    :attr:`df` (all ``None`` here). Subclasses: :class:`~lisatools.sources.gb.GBAETWaveform`,
    :class:`~lisatools.sources.emri.EMRITDIWaveform`.
    """
    #: Whether this generator can evaluate a BATCH of independent parameter
    #: sets in one call, returning one result per set.
    #:
    #: Default ``False``, and deliberately so: batching is opt-in. The read
    #: site uses ``getattr(gen, "supports_batch", False)``, so a generator
    #: that never declares this -- including one with no common base class --
    #: is excluded automatically. Declaring it here is for discoverability,
    #: NOT as the safety mechanism.
    #:
    #: A subclass sets this True only if every row of a batch is guaranteed
    #: to share the sub-sample alignment ``pyResponseTDI`` requires; see
    #: :class:`~lisatools.sources.bbh.gridaligned.GridAlignedPhenomTHMTDIWaveform`.
    supports_batch: bool = False

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
    """Base class for a waveform built in a simpler fashion for SNR calculations.

    Same interface as :class:`AETTDIWaveform` (:attr:`dt`, :attr:`f_arr`, :attr:`df`, all
    ``None`` here). Subclass: :class:`~lisatools.sources.bbh.BBHSNRWaveform`.
    """

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
    """Base class for a waveform built in the time domain.

    A subclass implements :meth:`compute_tdi_channels` (absolute times and TDI channels). This
    base adds the projection onto the analysis domain:

    - :meth:`get_signals_for_residuals` (and :meth:`_td_to_output_domain`) return one
      :class:`~lisatools.domains.DomainBase` per source. With a
      :class:`~lisatools.domains.WDMSettings` ``output_domain_settings`` (the production
      case) the channels are placed on the full data grid, Tukey-windowed like the data and
      transformed with ``TDSignal.transform``.
    - :meth:`__call__` returns the raw ``(signal, start_freqs[, start_times])`` arrays of
      :meth:`fft` / :meth:`stft` on a common padded grid (FD and STFT targets only).

    The analysis domain is WDM when ``output_domain_settings`` is given, STFT when ``stft_dt``
    is given, else FD (see :attr:`analysis_domain`).

    Args:
        waveform_t0: Epoch [s] the sampled merger / reference times are measured from; the
            subclasses add it to the waveform's own time axis to get absolute times.
        data_td_settings: :class:`~lisatools.domains.TDSettings` of the data grid (``N``,
            ``dt``, ``t0`` = the absolute data start). Defines :attr:`dt`, :attr:`Tobs` and
            :attr:`data_t0`.
        tdi_generation: TDI generation string for :class:`~lisatools.response.tdiconfig.TDIConfig`
            (``"1st generation"`` or ``"2nd generation"``). Default ``"2nd generation"``.
        tdi_channels: TDI channel set. Default ``"XYZ"``.
        sampling_frequency: Sampling frequency [Hz] handed to the response. Default ``0.4``.
        orbits: :class:`~lisatools.detector.Orbits` for the response. ``None`` builds an
            :class:`~lisatools.detector.EqualArmlengthOrbits` on ``force_backend`` and logs a
            warning; production passes the run's unequal-arm orbits.
        tukey_alpha: Alpha of the Tukey window applied before every domain transform
            (``0`` skips the window on the WDM path). Default ``0.01``.
        stft_dt: STFT segment duration [s]; must be an integer multiple of ``dt`` and
            ``>= dt``. ``None`` (default) means no STFT target.
        freq_min: Lower frequency bound [Hz] of the FD / STFT output. Default ``1e-5``.
        freq_max: Upper frequency bound [Hz] of the FD / STFT output. Default ``1.0``.
        fft_batch_size: Sources per ``rfft`` call in :meth:`fft` (bounds memory). Default ``1``.
        output_domain_settings: Analysis-domain target as a settings object (domains are never
            communicated by string). Only :class:`~lisatools.domains.WDMSettings` is supported;
            FD and STFT targets are set by ``stft_dt`` and the frequency bounds.
        force_backend: Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda13x"``, ...). Default
            ``"cpu"``.

    Attributes:
        tdi_config (TDIConfig): Built from ``tdi_generation`` on ``force_backend``.
        domain_settings (TDSettings): ``data_td_settings``.
        nperseg (int | None): Samples per STFT segment (``None`` without ``stft_dt``).
        transform_to_domain (callable): :meth:`stft` when ``stft_dt`` is set, else :meth:`fft`.

    Raises:
        NotImplementedError: If ``output_domain_settings`` is not a
            :class:`~lisatools.domains.WDMSettings`.
        AssertionError: If both ``stft_dt`` and ``output_domain_settings`` are given, or
            ``stft_dt`` is smaller than or not a multiple of ``dt``.
    """
    #: Whether this generator can evaluate a BATCH of independent parameter
    #: sets in one call, returning one result per set.
    #:
    #: Default ``False``, and deliberately so: batching is opt-in. The read
    #: site uses ``getattr(gen, "supports_batch", False)``, so a generator
    #: that never declares this -- including one with no common base class --
    #: is excluded automatically. Declaring it here is for discoverability,
    #: NOT as the safety mechanism.
    #:
    #: A subclass sets this True only if every row of a batch is guaranteed
    #: to share the sub-sample alignment ``pyResponseTDI`` requires; see
    #: :class:`~lisatools.sources.bbh.gridaligned.GridAlignedPhenomTHMTDIWaveform`.
    supports_batch: bool = False

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
        output_domain_settings: DomainSettingsBase = None,
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

        if output_domain_settings is not None and not isinstance(
            output_domain_settings, WDMSettings
        ):
            raise NotImplementedError(
                "output_domain_settings currently supports WDMSettings only; "
                "FD / STFT targets are configured via stft_dt / the frequency "
                f"bounds. Got: {type(output_domain_settings).__name__}."
            )
        self.output_domain_settings = output_domain_settings

        if stft_dt is None:
            if output_domain_settings is None:
                logger.info(
                    "No stft timestep provided. By default, the waveform will be transformed to the frequency domain"
                )
            else:
                logger.info(
                    "Output domain set by settings object: "
                    f"{type(output_domain_settings).__name__}"
                )
            self.transform_to_domain = self.fft
            self.nperseg = None

        else:
            assert output_domain_settings is None, (
                "Pass either stft_dt (STFT target) or output_domain_settings "
                "(settings-object target), not both."
            )
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
        """Constructor keyword arguments that rebuild this generator (reproducibility, debugging).

        ``stft_dt`` is reconstructed as ``nperseg * dt``.
        """
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
            "output_domain_settings": self.output_domain_settings,
            "force_backend": self.force_backend,
        }


    @property
    def force_backend(self) -> str:
        """Short backend name (``"cpu"``, ``"cuda12x"``, ...) handed to the classes built here."""
        return self.backend_name.split("_")[-1]

    @property
    def xp(self) -> ArrayModule:
        """Array module used for calculations (numpy or cupy, following the backend)."""
        return self.backend.xp

    @property
    def tdi_buffer_time(self) -> float:
        """Margin [s] for the response's retarded reads at the window edges (fixed ``600.0``).

        Covers the SSB-to-spacecraft projection (``<~ 500 s``) plus the TDI arm delays; used for
        the pre-data crop in :meth:`TDPyResponseWaveformBase._apply_response` and the spline
        padding of :class:`TDTDIOnFlyWaveformBase`.
        """
        return 600.0

    @property
    def dt(self):
        """float: Time step in seconds (``data_td_settings.dt``)."""
        return self.domain_settings.dt

    @property
    def Tobs(self):
        """float: Observation time in seconds (``N * dt`` of the data grid)."""
        return self.domain_settings.N * self.domain_settings.dt

    @property
    def data_t0(self):
        """float: Absolute start time of the data in seconds (``data_td_settings.t0``)."""
        return self.domain_settings.t0

    @property
    def stft_t_arr(self):
        """Start times of the STFT segments (every ``nperseg``-th data time), or ``None``."""
        if self.nperseg:
            return self.domain_settings.t_arr[:: self.nperseg]
        else:
            return None

    @property
    def data_times_array(self):
        """Complete absolute time array of the data grid (``data_td_settings.t_arr``)."""
        return self.domain_settings.t_arr

    @property
    def orbits(self):
        """:class:`~lisatools.detector.Orbits` used by the response."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits):
        """Set the orbits; ``None`` builds ``EqualArmlengthOrbits`` and logs a warning."""

        if orbits is None:
            # Runtime import: detector is only imported under TYPE_CHECKING
            # at module level (keeps import-time cost down).
            from lisatools.detector import EqualArmlengthOrbits

            orbits = EqualArmlengthOrbits(force_backend=self.force_backend)
            logger.warning("No Orbits object provided. Using default EqualArmlengthOrbits.")

        self._orbits = orbits  # configures lazily on first use

    @property
    def analysis_domain(self) -> str:
        """str: Analysis domain of the output: ``"WDM"`` (``output_domain_settings`` given),
        ``"STFT"`` (``stft_dt`` given) or ``"FD"``."""
        if isinstance(self.output_domain_settings, WDMSettings):
            return "WDM"
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
        """Compute the time-domain TDI channels (implemented by subclasses).

        Returns:
            tuple: ``(times, channels)``. ``times`` holds absolute times [s], shape
            ``(num_times,)`` for one source or ``(num_bin, num_times)`` for a batch;
            ``channels`` is the TDI response, shape ``(num_channels, num_times)`` or
            ``(num_bin, num_channels, num_times)``.

        Raises:
            NotImplementedError: Always, in the base class.
        """

        raise NotImplementedError("compute_tdi_channels method must be implemented in subclass.")

    def get_grid_time(self, times: NDArrayLike) -> NDArrayLike:
        """Round times to the nearest point of the data lattice ``data_t0 + k * dt``.

        Args:
            times: Absolute times [s], shape ``(num_times,)``.

        Returns:
            xp.ndarray: The lattice times, same shape.
        """
        dt = self.dt
        t0 = self.data_t0
        return t0 + self.xp.round((times - t0) / dt) * dt

    def get_output_settings(self, times: NDArrayLike) -> DomainSettingsBase:
        """Output-domain settings for the FD or STFT target.

        STFT: :func:`~lisatools.domains.get_stft_settings` on ``times`` with segment length
        ``nperseg * dt`` and the frequency bounds. FD: :class:`~lisatools.domains.FDSettings`
        of the full data grid (``N // 2 + 1`` bins, ``df = 1 / (N dt)``) with the frequency
        bounds; ``times`` is unused.

        Args:
            times: Evaluation times [s], shape ``(num_times,)``.

        Returns:
            DomainSettingsBase: The settings for the output domain.

        Raises:
            NotImplementedError: For the WDM analysis domain (handled in
                :meth:`_td_to_output_domain`).
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
        """Start time and sample count of each source's output grid (the raw-array path).

        STFT: the start of the STFT segment holding each source's first time, and the number
        of samples (a multiple of ``nperseg``) up to the segment holding its last time. FD: the
        data start and the full data length for every source.

        Args:
            times: Absolute times [s], shape ``(num_times,)`` or ``(num_bin, num_times)``.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(left_edges, grid_length)``, each ``(num_bin,)``.

        Raises:
            NotImplementedError: For the WDM analysis domain (use
                :meth:`get_signals_for_residuals`).
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

        else:
            raise NotImplementedError(
                f"The raw-array path (__call__ / build_common_grid) supports "
                f"FD and STFT only; for {self.analysis_domain} targets use "
                f"get_signals_for_residuals()."
            )

        return left_edges, grid_length

    def build_common_grid(
        self, times: NDArrayLike, channels: NDArrayLike
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Scatter every source's channels onto one zero-padded array (FD / STFT targets).

        Each source is placed at ``rint((t - left_edge) / dt)`` on its own grid
        (:meth:`find_bin_edges`); all grids share the longest length. Samples that fall
        outside (and, for STFT, past the data end) are dropped.

        Args:
            times: Absolute times [s], shape ``(num_bin, num_times)`` (or ``(num_times,)``).
            channels: Channels at ``times``, shape ``(num_bin, num_channels, num_times)`` (or
                ``(num_channels, num_times)``; a batch axis is added).

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(left_edges, padded_signals)``; ``left_edges``
            ``(num_bin,)``, ``padded_signals`` ``(num_bin, num_channels, max_grid_length)``.
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

        Thin delegate to the stock :func:`lisatools.domains.pad_td_signal`
        utility (kept as a method for subclass / call-site compatibility);
        see its docstring for the full semantics.
        """
        return pad_td_signal(
            times,
            signals,
            data_t0=self.data_t0,
            dt=self.dt,
            align_samples=align_samples,
            target_n=target_n,
        )

    def _td_to_output_domain(
        self,
        times_in: NDArrayLike,
        signal_in: NDArrayLike,
        output_domain: str = None,
        domain_kwargs: dict = None,
    ) -> DomainBase:
        """Transform one source's time-domain channels to the output domain.

        With ``output_domain=None`` the target is :attr:`analysis_domain`. For a WDM target
        (``output_domain_settings``) the channels are placed on the full data grid
        (:func:`~lisatools.domains.place_td_signal_on_grid`), multiplied by a full-length Tukey
        window (``tukey_alpha``; none when 0) and transformed with ``TDSignal.transform``, the
        same chain as the data. STFT: right-padded to a multiple of the segment length and
        aligned to the segment lattice, Tukey per segment. FD: padded to the full data length
        and windowed at that length (as the data, ``tukey(N) -> FFT``). ``"TD"`` wraps the
        input as a :class:`~lisatools.domains.TDSignal`.

        Args:
            times_in: Absolute times [s] of the input, 1-D ``(num_times,)``.
            signal_in: Channels at ``times_in``, ``(num_channels, num_times)``.
            output_domain: Target domain (``"TD"``, ``"STFT"`` or ``"FD"``, case-insensitive).
                ``None`` (default) uses :attr:`analysis_domain`, the only way to reach WDM.
            domain_kwargs: Settings kwargs for an explicit ``output_domain``: ``big_dt`` plus
                :func:`~lisatools.domains.get_stft_settings` kwargs for STFT, ``df`` plus
                :class:`~lisatools.domains.FDSettings` kwargs for FD.

        Returns:
            DomainBase: The signal in the requested domain.

        Raises:
            NotImplementedError: If ``times_in`` is not 1-D.
            ValueError: For an unsupported explicit ``output_domain`` (including ``"WDM"``).
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

        if use_default_domain and isinstance(self.output_domain_settings, WDMSettings):
            # Settings-object target (sprint rule: no domain strings). Place
            # the signal on the full data grid, then run the canonical
            # lisatools transform chain (TDSignal -> FFT -> WDM).
            full_grid = TDSettings(
                N=self.domain_settings.N,
                dt=dt_here,
                t0=self.data_t0,
                force_backend=self.force_backend,
            )
            padded_td_signal = place_td_signal_on_grid(
                signal_in, full_grid, times=times_in
            )
            # Same treatment as the data side: full-length Tukey window
            # (alpha = 0 -> rectangular, skip the multiply).
            window = (
                tukey(full_grid.N, alpha=self.tukey_alpha, xp=self.xp)
                if self.tukey_alpha
                else None
            )
            return padded_td_signal.transform(self.output_domain_settings, window=window)

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
        """Transform pre-padded time-domain data to the FD basis.

        Signals must be pre-padded (via :meth:`build_common_grid`); no padding is done here.
        A Tukey window (``tukey_alpha``) of the last-axis length multiplies ``signal_in`` IN
        PLACE, then ``rfft(...) * dt`` runs over the last axis in batches of
        ``fft_batch_size`` sources, and the result is sliced to ``[freq_min, freq_max]``.

        Args:
            start_times: Start time of each source, shape ``(num_binaries,)``. Unused here;
                kept for signature consistency with :meth:`stft`.
            signal_in: Time-domain signal, shape ``(num_binaries, num_channels, num_times)``
                (or with an extra segment axis before the last, from :meth:`stft`). Modified in
                place by the window.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(signal_out, start_freqs)``. ``signal_out`` is a
            complex view ``(..., num_freqs)`` of the in-band bins; ``start_freqs``
            ``(num_binaries,)`` holds the first kept frequency [Hz] for every source.
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
        """Transform pre-padded time-domain data to the STFT basis.

        Reshapes the last axis into ``(num_segments, nperseg)`` and runs :meth:`fft` on the
        segments (per-segment Tukey window, applied in place).

        Args:
            start_times: Start time of each source, shape ``(num_binaries,)``; passed through.
            signal_in: Time-domain signal, shape ``(num_binaries, num_channels, num_times)``
                with ``num_times`` a multiple of ``nperseg``.

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(signal_out, start_freqs,
            start_times)``; ``signal_out`` has shape ``(num_binaries, num_channels,
            num_segments, num_freqs)``.
        """
        num_binaries, num_channels = signal_in.shape[:2]
        signal_in = signal_in.reshape(
            num_binaries, num_channels, -1, self.nperseg
        )  # (num_binaries, num_channels, num_segments, num_times_per_segment)

        signal_out, start_freqs = self.fft(start_times, signal_in)

        return signal_out, start_freqs, start_times

    def __call__(self, *args, **kwargs):
        """Generate the waveform and return the raw transform output (FD / STFT targets).

        Runs :meth:`compute_tdi_channels`, :meth:`build_common_grid` and
        :attr:`transform_to_domain`. Sky/response parameters may be supplied either as explicit
        keyword arguments or as the last N positional arguments; ``compute_tdi_channels`` on
        each subclass peels them off ``*args``. For a WDM target use
        :meth:`get_signals_for_residuals` (this path raises ``NotImplementedError``).

        Args:
            *args: Waveform parameters, forwarded to :meth:`compute_tdi_channels`.
            **kwargs: Forwarded to :meth:`compute_tdi_channels`.

        Returns:
            tuple: The output of :meth:`fft` ``(signal_out, start_freqs)`` or :meth:`stft`
            ``(signal_out, start_freqs, start_times)``, not a domain object.
        """
        times, channels = self.compute_tdi_channels(*args, **kwargs)
        left_edges, padded = self.build_common_grid(times, channels)
        return self.transform_to_domain(left_edges, padded)

    def get_signals_for_residuals(self, *args, **kwargs) -> DomainBaseArray:
        """Generate per-source domain-wrapped signals for residual operations.

        Runs :meth:`compute_tdi_channels` and :meth:`_td_to_output_domain` per source. The
        global-fit engine rebuilds MBH residuals through this method (``MBH_LIKELIHOOD=batched``
        keeps the stock :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIWaveform` here).
        Sky/response parameters may be supplied either as explicit keyword arguments or as the
        last N positional arguments.

        Args:
            *args: Waveform parameters, forwarded to :meth:`compute_tdi_channels`.
            **kwargs: Forwarded to :meth:`compute_tdi_channels`.

        Returns:
            DomainBaseArray | DomainBase: A :class:`~lisatools.domains.DomainBaseArray` of
            per-source signals when ``times`` is 2-D (batch), else a single domain signal.
        """
        times, channels = self.compute_tdi_channels(*args, **kwargs)
        if times.ndim == 2:
            return DomainBaseArray([
                self._td_to_output_domain(times_in=times[i], signal_in=channels[i])
                for i in range(channels.shape[0])
            ])
        return self._td_to_output_domain(times_in=times, signal_in=channels)


class TDPyResponseWaveformBase(TDWaveformBase):
    """Base class for a time-domain waveform whose TDI response comes from ``pyResponseTDI``.

    A subclass implements :meth:`wave_gen` (one source) and, for batches, :meth:`wave_gen_batch`;
    both return ``(t_arr, h_plus, h_cross)`` with ``t_arr`` relative to the merger.
    :meth:`compute_tdi_channels` shifts them by ``merger_time + waveform_t0`` and applies the
    response (:meth:`_apply_response`). The MBH
    :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIWaveform` is the production subclass.

    Args:
        waveform_t0: See :class:`TDWaveformBase`.
        data_td_settings: See :class:`TDWaveformBase`.
        tdi_generation: See :class:`TDWaveformBase`. Default ``"2nd generation"``.
        tdi_channels: See :class:`TDWaveformBase`. Default ``"XYZ"``.
        sampling_frequency: Sampling frequency [Hz] of the response. Default ``0.4``.
        orbits: See :class:`TDWaveformBase`.
        order: Lagrange interpolation order of ``pyResponseTDI``. Default ``35``.
        tukey_alpha: See :class:`TDWaveformBase`. Default ``0.01``.
        stft_dt: See :class:`TDWaveformBase`.
        freq_min: See :class:`TDWaveformBase`. Default ``1e-5``.
        freq_max: See :class:`TDWaveformBase`. Default ``1.0``.
        fft_batch_size: See :class:`TDWaveformBase`. Default ``1``.
        signal_duration: Initial response length [s] (``num_pts = int(signal_duration / dt)``);
            ``None`` uses :attr:`Tobs`. Each call resets ``response.num_pts`` to its own window.
        buffer_time: Response garbage buffer [s]: the strain is zero-padded by this much after
            its end, and the first ``buffer_time / dt`` output samples are zeroed. Default
            ``5000``.
        run_async: Forwarded to the batched ``pyResponseTDI`` API. Default ``False``.
        output_domain_settings: See :class:`TDWaveformBase`.
        force_backend: See :class:`TDWaveformBase`. Default ``"cpu"``.

    Attributes:
        response (pyResponseTDI): The response object, shared by every call.
        buffer_time (float): As above.
        run_async (bool): As above.
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
        output_domain_settings: DomainSettingsBase = None,
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
            output_domain_settings=output_domain_settings,
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
        """Constructor kwargs of :class:`TDWaveformBase` plus ``order``, ``signal_duration``,
        ``buffer_time`` and ``run_async`` (``signal_duration`` is the response's current
        ``num_pts * dt``)."""
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
        """Generate the polarizations of a single source (implemented by subclasses).

        Called as ``wave_gen(*args, ra, dec, merger_time, **kwargs)`` by :meth:`_call_single`.

        Returns:
            tuple: ``(t_arr, h_plus, h_cross)``, 1-D, with ``t_arr`` [s] relative to the merger.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("wave_gen method must be implemented in subclass.")

    def wave_gen_batch(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the polarizations of a batch of sources (implemented by subclasses).

        Subclasses that support batched generation override this to return rectangular,
        already-masked arrays; :meth:`_call_batched` passes them to :meth:`_apply_response`.

        Returns:
            tuple: ``(times_batch, h_plus_batch, h_cross_batch)``, each ``(Nbatch, Ntimes)``,
            times [s] relative to each source's merger.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError(
            "wave_gen_batch is not implemented for this waveform. "
            "Batched calls require implementing wave_gen_batch in the subclass."
        )

    def _apply_response(
        self,
        t_arr: NDArrayLike,
        h_plus: NDArrayLike,
        h_cross: NDArrayLike,
        ra: float | NDArrayLike,
        dec: float | NDArrayLike,
        merger_time: float | NDArrayLike,
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Apply the TDI response to one source or a batch, cropped to the data window.

        Steps: inputs are moved onto :attr:`xp`; absolute times are ``t_arr + merger_time +
        waveform_t0``; the samples before ``data_t0 - tdi_buffer_time`` are cropped (by the
        smallest safe amount across the batch, which stays rectangular); the strain
        ``h_plus + 1j h_cross`` is zero-padded by ``buffer_time``; the per-source sub-sample
        offset ``t0_shift_to_data`` (in ``(-dt/2, dt/2]``) puts the response on the data
        lattice; ``pyResponseTDI.get_projections`` / ``get_tdi_delays`` run (batched API, or a
        per-source loop on the legacy one); the padding is removed and the first
        ``buffer_time / dt`` samples zeroed; non-finite output is zeroed (a WARNING counts the
        ones inside the data span); everything before ``data_t0`` is cropped.

        ``ra`` / ``dec`` are passed to the response as its ``lam`` / ``beta`` and are read in the
        orbits' frame: ICRS (RA, Dec) for the run's ``frame="icrs"`` orbits.

        Args:
            t_arr: Times [s] relative to the merger, ``(Ntimes,)`` or ``(Nbatch, Ntimes)``.
            h_plus: Plus polarization, same shape.
            h_cross: Cross polarization, same shape.
            ra: Right ascension [rad]: a Python ``float`` (single source) or ``(Nbatch,)``.
            dec: Declination [rad], same shape as ``ra``.
            merger_time: Merger time [s] relative to ``waveform_t0``, same shape as ``ra``.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(times, channels)``: absolute times on the data
            lattice from ``data_t0`` on, ``(Nbatch, Ntimes)``, and the TDI channels
            ``(Nbatch, num_channels, Ntimes)``. A single source (``ra`` a ``float``) returns
            ``(Ntimes,)`` and ``(num_channels, Ntimes)``. A one-row batch on the batched
            response API can return ``(1, Ntimes)`` times with ``(num_channels, Ntimes)``
            channels (the response squeezes its batch axis; see
            :class:`~lisatools.sources.batching.BatchedDomainSignalGen`).

        Raises:
            LISAToolsException: Its batch-refusal subclass (see the shared-window guard in the
                code; a source-inspection test looks for the name there) if the batch's rows
                would start the data window at different samples (one shared crop would cut
                some rows at the wrong place); the caller falls back to per-row evaluation.
            ValueError: With ``DEBUG_MODE``, on NaN / Inf input or a window outside the orbits.
        """
        single_source = isinstance(ra, float)

        # wave_gen (e.g. phentax) can return host (numpy) arrays while the
        # response backend is cupy; xp.atleast_*() does not move host arrays
        # onto the device, so coerce every input onto self.xp before the device
        # arithmetic below. asarray is a no-op when the array is already on
        # backend, so the CPU path is unchanged.
        ra = self.xp.atleast_1d(self.xp.asarray(ra))
        dec = self.xp.atleast_1d(self.xp.asarray(dec))
        merger_time = self.xp.atleast_1d(self.xp.asarray(merger_time))

        t_arr = self.xp.atleast_2d(self.xp.asarray(t_arr))
        h_plus = self.xp.atleast_2d(self.xp.asarray(h_plus))
        h_cross = self.xp.atleast_2d(self.xp.asarray(h_cross))

        shifted_t_arr = t_arr + merger_time[:, None] + self.waveform_t0

        # Clip the pre-data window BEFORE paying for the response on it.
        #
        # Everything earlier than ``data_t0`` is cropped off again further down
        # (``start_ind``), so the response compute spent on it is pure waste --
        # and it is not a small tail: in synthetic mode ``waveform_t0`` sits at
        # the orbit start, so a month-long pre-merger inspiral sits at t < 0
        # where the orbits cannot even be evaluated. Those samples come back
        # non-finite, get zeroed, and are then thrown away. On all_sources_lite
        # that was ~1.5M samples per template against a ~10k-sample data grid,
        # which is what made a six-branch run unrunnable on a laptop.
        #
        # A sample can only reach a KEPT sample through the TDI delays, which
        # look back ~8 arm-delays; ``tdi_buffer_time`` is the margin allowed for
        # exactly that. So anything before ``data_t0 - tdi_buffer_time`` cannot
        # influence the output that survives, and dropping it changes nothing.
        # The batch must stay rectangular, so crop by the SMALLEST safe amount
        # across sources -- every source then keeps at least its own margin.
        _keep_from = self.data_t0 - self.tdi_buffer_time
        _lead = self.xp.floor((_keep_from - shifted_t_arr[:, 0]) / self.dt)
        _lead = int(self.xp.maximum(_lead, 0).min())
        # never clip the window out of existence (a wholly pre-data waveform
        # still has to fall out of the start_ind crop below, not of an
        # empty response call)
        _lead = min(_lead, int(shifted_t_arr.shape[-1]) - 1)
        if _lead > 0:
            shifted_t_arr = shifted_t_arr[:, _lead:]
            h_plus = h_plus[:, _lead:]
            h_cross = h_cross[:, _lead:]

        # pad with zeros by num_buffer_ponts
        num_buffer_ponts = int(self.buffer_time / self.dt)

        pad_idx_right = self.xp.arange(1, num_buffer_ponts + 1)[None, :]
        # pad_idx_left = self.xp.arange(num_buffer_ponts, 0, -1)[None, :]

        shifted_t_arr = self.xp.concatenate(
            [
                #shifted_t_arr[:, 0:1] - self.dt * pad_idx_left,
                shifted_t_arr,
                shifted_t_arr[:, -1:] + self.dt * pad_idx_right,
            ],
            axis=-1,
        )

        # condition the signal with a small taper at the start to mitigate edge effects in the response
        # num_orig_pts = h_plus.shape[-1]
        
        # alpha = (self.tdi_buffer_time / self.dt) / num_orig_pts #todo check if this works with the new polarization padding!! 
        # window_orig = tukey(num_orig_pts, alpha=alpha, xp=self.xp)
        # window_orig[num_orig_pts//2:] = 1.0  # Only taper the start!

        # h_plus = h_plus * window_orig[None, :]
        # h_cross = h_cross * window_orig[None, :]

        # Pad zeros after the original signal to provide a buffer. There the signal is zero anyways
        h_plus = self.xp.pad(h_plus, ((0, 0), (0, num_buffer_ponts)), mode="constant", constant_values=0.0)
        h_cross = self.xp.pad(h_cross, ((0, 0), (0, num_buffer_ponts)), mode="constant", constant_values=0.0)

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

        # Sub-dt grid alignment: the response builds an internal time array
        #     t_arr = arange(N) * dt + (t0 + t0_shift_to_data)
        # If shifted_t_arr[:, 0] is not on the data grid (data_t0 + k*dt),
        # h+/hx is sampled on a sub-dt-shifted grid and the resulting TDI
        # template is misaligned with the data — a O(1e-3) mismatch even when
        # all parameters are correct.  Compute t0_shift_to_data per source so
        # the response evaluates on the data grid (modulo integer offsets).
        delta = self.data_t0 - shifted_t_arr[:, 0]
        t0_shift_to_data = delta - self.xp.rint(delta / self.dt) * self.dt
        # |t0_shift_to_data| is now in (-dt/2, dt/2], satisfying
        # pyResponseTDI's assert(|t0_shift_to_data| < dt).

        if "run_async" in inspect.signature(self.response.get_projections).parameters:
            # stft_tof batched response API (array lam/beta/t0 + run_async).
            _kw = {}
            if "t0_shift_to_data" in inspect.signature(self.response.get_projections).parameters:
                _kw["t0_shift_to_data"] = t0_shift_to_data
            self.response.get_projections(
                strain, lam=ra, beta=dec, t0=shifted_t_arr[:, 0],
                t_buffer=self.buffer_time, run_async=self.run_async, **_kw,
            )

            tdis = self.xp.array(self.response.get_tdi_delays(run_async=self.run_async)) # (Nbatch, num_channels, Ntimes) if batched else (num_channels, Ntimes)
            if len(tdis.shape) == 3:
                tdis = tdis.transpose(1, 0, 2)
        else:
            # TODO(Phase B): lisatools' current ``pyResponseTDI`` is the
            # single-source legacy API (scalar lam/beta/t0, fixed num_pts,
            # no run_async). Loop the batch here until the tdi_on_fly
            # legacy-response updates (batched arrays + run_async) are
            # ported into ``lisatools.response.directresponse``.
            tdis_per_source = []
            for b in range(strain.shape[0]):
                # output length tracks this source's padded window
                self.response.num_pts = int(strain.shape[-1])
                self.response.get_projections(
                    strain[b],
                    lam=float(ra[b]),
                    beta=float(dec[b]),
                    t0=float(shifted_t_arr[b, 0]),
                    t0_shift_to_data=float(t0_shift_to_data[b]),
                    t_buffer=self.buffer_time,
                )
                tdis_per_source.append(
                    self.xp.asarray(self.response.get_tdi_delays())
                )
            tdis = self.xp.array(tdis_per_source)  # (Nbatch, num_channels, Ntimes)

        tdis = tdis[..., :-num_buffer_ponts] # remove the padded points
        tdis[..., :num_buffer_ponts] = 0.0  # zero out the corrupted points at the start
        shifted_t_arr = shifted_t_arr[:, :-num_buffer_ponts]

        # Apply the same sub-dt shift to the time labels so they describe what
        # the response actually evaluated.  After this the previous post-hoc
        # `% dt` correction below is a near-no-op (residual is < ULP).
        shifted_t_arr += t0_shift_to_data[:, None]

        t_arr_shift = (self.data_t0 - shifted_t_arr[:, 0]) % self.dt
        shifted_t_arr += t_arr_shift[:, None]

        start_inds = self.xp.maximum(
            0, self.xp.rint((self.data_t0 - shifted_t_arr[:, 0]) / self.dt).astype(int)
        )
        # THE BATCH MUST SHARE A WINDOW START, NOT JUST AN ALIGNMENT.
        #
        # Everything below applies ONE crop to every row. Rows whose valid
        # data begins at different samples are then cropped at the wrong
        # place, and the resulting likelihood is silently wrong -- finite,
        # plausible, and dependent on WHICH OTHER ROWS share the batch, which
        # is fatal for detailed balance in a sampler.
        #
        # The alignment check in pyResponseTDI does NOT cover this axis: it
        # compares ``t0_shift_to_data``, which grid-aligned generation drives
        # to exactly zero BY CONSTRUCTION, so it is inert for exactly the
        # generator that can batch. Measured on a 4-walker block: max
        # |dlogL| 7.6e5 nats against per-row evaluation, no warning, and the
        # error grows with the width of the walker cloud -- accurate once a
        # chain has converged and wrong through burn-in, which is the worst
        # possible failure shape.
        #
        # Refusing is the correct answer rather than a limitation: the caller
        # falls back to per-row evaluation, which is already verified. Making
        # the crop per-row is the real fix and is a separate change; this
        # guard is what makes the wrongness unreachable in the meantime.
        if start_inds.size > 1 and int(start_inds.max()) != int(start_inds.min()):
            raise BatchNotLaunchable(
                f"batched response needs every source to share a window "
                f"start: start_inds span "
                f"[{int(start_inds.min())}, {int(start_inds.max())}] samples "
                f"across {int(start_inds.size)} sources. One shared crop is "
                f"applied to the whole batch, so rows starting elsewhere "
                f"would be cropped at the wrong sample. Evaluate these "
                f"sources in separate calls."
            )
        start_ind = int(start_inds.max())

        # Non-finite response output is zeroed (ResponseWrapper
        # ``remove_garbage="zero"`` semantics) — one non-finite sample kept
        # in the data span would poison the whole domain transform
        # downstream (both the injected data and every template). Two
        # distinct populations, reported separately so the numbers read
        # correctly:
        #   (a) pre-data window samples, discarded by the ``start_ind`` crop
        #       below. EXPECTED and harmless whenever the waveform window
        #       opens before the orbit span (synthetic mode puts
        #       waveform_t0 at the orbit start, so the ~month-long
        #       pre-merger inspiral sits at t < 0 where orbits cannot be
        #       evaluated) — but it IS wasted response compute; clip the
        #       window if this dominates template timing.
        #   (b) samples INSIDE the kept data span: the TDI warm-up right at
        #       the orbit-span edge (output at t needs orbit data back to
        #       t - ~8 arm-delays). Small (~tens of samples) and zero is
        #       the correct value; a LARGE count here means something else
        #       is wrong (bad parameters / orbit coverage) — hence WARNING.
        _bad = ~self.xp.isfinite(tdis)
        if _bad.any():
            _n_kept = int(_bad[..., start_ind:].sum())
            _n_cropped = int(_bad.sum()) - _n_kept
            if _n_kept:
                logger.warning(
                    "_apply_response: zeroed %d non-finite TDI sample(s) "
                    "INSIDE the data span (TDI warm-up at the orbit-span "
                    "edge; expected ~O(10) at a t=0 orbit start — "
                    "investigate if large).",
                    _n_kept,
                )
            if _n_cropped:
                logger.debug(
                    "_apply_response: zeroed %d non-finite samples in the "
                    "pre-data portion of the waveform window (all discarded "
                    "by the data-span crop; expected when the window opens "
                    "before the orbit span, e.g. synthetic mode — wasted "
                    "response compute, not a correctness issue).",
                    _n_cropped,
                )
            tdis[_bad] = 0.0

        if start_ind > 0:
            shifted_t_arr = shifted_t_arr[:, start_ind:]
            tdis = tdis[..., start_ind:]
        
        # now remove the extra time dimensions if we only had one source (to be consistent with the single-source path)
        if single_source:
            shifted_t_arr = shifted_t_arr[0]
            if tdis.ndim == 3:
                # legacy-response fallback keeps a batch axis; squeeze to the
                # (num_channels, Ntimes) single-source contract.
                tdis = tdis[0]

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
        """One source: :meth:`wave_gen`, then :meth:`_apply_response`; ``(times, channels)``."""

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
        """A batch: :meth:`wave_gen_batch` then one :meth:`_apply_response` call for all rows.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(times, channels)`` as in :meth:`_apply_response`.
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
        """Compute the time-domain TDI channels of one source or a batch.

        If any of ``ra``, ``dec``, ``merger_time`` is ``None``, the last three positional
        arguments are taken as ``(ra, dec, merger_time)``. A scalar ``ra`` runs
        :meth:`_call_single`; a 1-D ``ra`` runs :meth:`_call_batched` (one response call for
        the whole batch).

        Args:
            *args: Intrinsic waveform parameters for :meth:`wave_gen` / :meth:`wave_gen_batch`
                (optionally followed by ``ra, dec, merger_time``).
            ra: Right ascension [rad] (ICRS with ICRS orbits). Scalar for one source, 1-D for a
                batch.
            dec: Declination [rad]. Same shape as ``ra``.
            merger_time: Merger time [s] relative to ``waveform_t0``. Same shape as ``ra``.
            **kwargs: Keyword arguments for :meth:`wave_gen` / :meth:`wave_gen_batch`.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(times, channels)``: absolute times on the data
            lattice and the TDI channels, ``(num_channels, num_times)`` for one source or
            ``(num_bin, num_channels, num_times)`` for a batch (see :meth:`_apply_response`).

        Raises:
            BatchNotLaunchable: From :meth:`_apply_response` (or the subclass batch
                generator) when a batch cannot share one evaluation.
        """
        if ra is None or dec is None or merger_time is None:
            *args, ra, dec, merger_time = args

        if np.ndim(ra) >= 1:
            return self._call_batched(*args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        return self._call_single(*args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)



class TDTDIOnFlyWaveformBase(TDWaveformBase):
    """Base class for a time-domain waveform with the TDI-on-the-fly response.

    A subclass implements :meth:`get_amp_phase` (per-mode amplitude and phase on a possibly
    sparse grid) and :meth:`process_amp_phase` (the response's sign / scale convention).
    :meth:`compute_tdi_channels` splines each ``(source, mode)`` pair, evaluates the response
    (:class:`~lisatools.response.tdionfly.TDTDIonTheFly`) on the input grid, and samples the
    summed TDI channels on a dense data-lattice grid. Base of
    :class:`~lisatools.sources.bbh.waveform.PhenomTHMTDIOnFlyWaveform` (the MBH
    ``use_tdionfly`` path, off in production).

    Args:
        waveform_t0: Epoch [s] added to the sampled merger time to get absolute times.
        data_td_settings: See :class:`TDWaveformBase`.
        tdi_generation: See :class:`TDWaveformBase`. Default ``"2nd generation"``.
        tdi_channels: Must be ``"XYZ"`` (the only supported set). Default ``"XYZ"``.
        sampling_frequency: Sampling frequency [Hz] handed to the response. Default ``0.4``.
        orbits: See :class:`TDWaveformBase`.
        tukey_alpha: See :class:`TDWaveformBase`. Default ``0.01``.
        stft_dt: See :class:`TDWaveformBase`.
        freq_min: See :class:`TDWaveformBase`. Default ``0.0``.
        freq_max: See :class:`TDWaveformBase`. Default ``1.0``.
        fft_batch_size: See :class:`TDWaveformBase`. Default ``1``.
        zero_inclination: If ``True``, the response is fed ``inclination = 0`` for every mode
            (for amplitudes that already carry the spherical-harmonic factors). Default
            ``False``.
        output_domain_settings: See :class:`TDWaveformBase`.
        force_backend: See :class:`TDWaveformBase`. Default ``"cpu"``.

    Raises:
        NotImplementedError: If ``tdi_channels != "XYZ"``.
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
        output_domain_settings: DomainSettingsBase = None,
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
            output_domain_settings=output_domain_settings,
            force_backend=force_backend,
        )

        self.zero_inclination = zero_inclination

    @property
    def wrapper_kwargs(self):
        """Constructor kwargs of :class:`TDWaveformBase` plus ``zero_inclination``."""
        base_kwargs = super().wrapper_kwargs
        base_kwargs.update(
            {
                "zero_inclination": self.zero_inclination,
            }
        )
        return base_kwargs

    @property
    def max_length(self) -> int:
        """int: Number of evaluation times kept by :meth:`get_evaluation_times` (``2000``).

        The base class keeps the LAST ``max_length`` nodes; this may not be enough in some
        regions of the parameter space (the MBH subclass overrides the selection).
        """
        return 2_000

    def get_amp_phase(
        self,
        *args,
        **kwargs,
    ) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
        """Generate per-mode amplitude and phase for a batch of sources (implemented by subclasses).

        Called by :meth:`compute_tdi_channels` as ``get_amp_phase(*args, inclination, psi, ra,
        dec, merger_time, **kwargs)`` with every argument at least 1-D.

        Returns:
            tuple: ``(t_arr, amp, phase)``; ``t_arr`` [s] relative to the merger,
            ``(Nbatch, num_times)`` (or one row per mode); ``amp`` and ``phase``
            ``(Nbatch, num_modes, num_times)``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("amp_phase_gen method must be implemented in subclass.")

    def process_amp_phase(
        self, amp: NDArrayLike, phase: NDArrayLike
    ) -> Tuple[NDArrayLike, NDArrayLike]:
        """Map amplitude and phase to the TDI-on-the-fly convention (implemented by subclasses).

        Args:
            amp: Amplitude, shape ``(Nbatch, num_modes, num_times)``.
            phase: Phase, shape ``(Nbatch, num_modes, num_times)``.

        Returns:
            tuple: ``(processed_amp, processed_phase)``, same shapes.

        Raises:
            NotImplementedError: Always, in the base class.
        """

        raise NotImplementedError("process_amp_phase method must be implemented in subclass.")

    def stack_parameter(self, param: np.ndarray, num_modes: int) -> NDArrayLike:
        """Repeat each per-source value once per mode for the per-``(source, mode)`` response.

        Args:
            param: Per-source values, shape ``(Nbatch,)``.
            num_modes: Modes per source.

        Returns:
            xp.ndarray: Shape ``(Nbatch * num_modes,)``, each entry repeated ``num_modes`` times.
        """
        param = self.xp.asarray(param)
        return self.xp.repeat(param, num_modes)

    def get_tdi_buffers(self, delta_t: np.ndarray) -> Tuple[int, int, float, float]:
        """Node counts that span :attr:`tdi_buffer_time` at each end of the input grid.

        ``left_dt`` / ``right_dt`` are the smallest first / last step across rows;
        ``start_buffer = max(int(tdi_buffer_time / left_dt), 1)`` and likewise
        ``end_buffer`` (floored; :meth:`pad` rounds up instead).

        Args:
            delta_t: Steps between consecutive input times, shape
                ``(num_binaries * num_modes, num_times - 1)``.

        Returns:
            tuple[int, int, float, float]: ``(start_buffer, end_buffer, left_dt, right_dt)``.
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
        """Extend the spline input by at least :attr:`tdi_buffer_time` (600 s) at both ends.

        Adds ``ceil(tdi_buffer_time / left_dt)`` nodes before and
        ``ceil(tdi_buffer_time / right_dt)`` after (steps from :meth:`get_tdi_buffers`), with
        edge-held amplitude and phase, so the response's retarded reads stay inside the
        amplitude / phase splines.

        Args:
            input_times: Input times, shape ``(num_binaries * num_modes, num_times)``.
            input_amplitudes: Input amplitudes, same shape.
            input_phases: Input phases, same shape.

        Returns:
            tuple: Padded ``(times, amplitudes, phases)``, each
            ``(num_binaries * num_modes, pad_left + num_times + pad_right)``.
        """
        delta_t = self.xp.diff(input_times, axis=-1)

        _, _, left_dt, right_dt = self.get_tdi_buffers(delta_t)

        # Padding is a minimum physical support requirement, so round up.  The
        # evaluation-grid trimming policy deliberately remains in
        # ``get_tdi_buffers``; changing its historical floor would discard an
        # additional adaptive node at each edge.  On sparse MBHB grids,
        # flooring the spline padding itself can provide substantially less
        # than ``tdi_buffer_time`` and leave valid retarded link evaluations
        # outside the amplitude/phase spline.
        pad_length_left = max(
            int(self.xp.ceil(self.tdi_buffer_time / left_dt)), 1
        )
        pad_length_right = max(
            int(self.xp.ceil(self.tdi_buffer_time / right_dt)), 1
        )

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
        """Times at which the TDI-on-the-fly response is evaluated (the response spline nodes).

        The base class drops :meth:`get_tdi_buffers`'s ``start_buffer`` / ``end_buffer`` nodes
        at the two ends and keeps the last :attr:`max_length` of the rest. Subclasses override
        this to choose a different grid.

        Args:
            input_times: Times of the amplitude / phase input, shape
                ``(num_binaries * num_modes, num_times)``.

        Returns:
            xp.ndarray: Evaluation times, shape ``(num_binaries * num_modes, num_eval_times)``.
        """
        delta_t = self.xp.diff(input_times, axis=-1)

        start_buffer, end_buffer, _, _ = self.get_tdi_buffers(delta_t)

        in_times = input_times[:, start_buffer:-end_buffer]
        # evaluation_times = self.xp.concatenate([in_times[:, :1], in_times[:, -self.max_length :]], axis=-1)
        evaluation_times = in_times[:, -self.max_length :]

        return evaluation_times

    def get_dense_times(self, eval_times: NDArrayLike) -> NDArrayLike:
        """Regular output grid on the data lattice covering each row's evaluation times.

        Each row starts at the lattice point (step ``dt``, or the STFT segment length) at or
        before its first evaluation time; all rows share ``ceil(max duration / dt) + 1``
        samples (rounded up to a multiple of ``nperseg`` for STFT).

        Args:
            eval_times: Evaluation times, shape ``(num_rows, num_eval_times)``.

        Returns:
            xp.ndarray: Dense absolute times, shape ``(num_rows, num_times)``, step ``dt``.
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
        """Generate the TDI-on-the-fly response of a batch of sources.

        If ``inclination`` is ``None``, the last five positional arguments are taken as
        ``(inclination, psi, ra, dec, merger_time)``. Steps: :meth:`get_amp_phase` and
        :meth:`process_amp_phase`; absolute times ``t + merger_time + waveform_t0``; one
        response row per ``(source, mode)`` on :meth:`get_evaluation_times` with the input
        padded by :meth:`pad`; evaluation on :meth:`get_dense_times` (zero outside each row's
        spline support); the modes summed per source.

        Args:
            *args: Intrinsic parameters for :meth:`get_amp_phase` (optionally followed by
                ``inclination, psi, ra, dec, merger_time``).
            inclination: Inclination [rad], shape ``(Nbatch,)``. Replaced by zeros for the
                response when ``zero_inclination`` is set.
            psi: Polarization angle [rad], shape ``(Nbatch,)``.
            ra: Right ascension [rad] (in the orbits' frame), shape ``(Nbatch,)``.
            dec: Declination [rad], shape ``(Nbatch,)``.
            merger_time: Merger time [s] relative to ``waveform_t0``, shape ``(Nbatch,)``.
            **kwargs: Keyword arguments for :meth:`get_amp_phase`.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(dense_times, tdi_channels)`` with shapes
            ``(Nbatch, num_times)`` and ``(Nbatch, 3, num_times)`` (XYZ).
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

        # now pad with at least tdi_buffer_time (600 s) on each side
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
