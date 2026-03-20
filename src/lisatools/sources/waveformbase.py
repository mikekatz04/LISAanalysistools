from __future__ import annotations

from abc import ABC
import logging
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        import numpy as cp

    from gpubackendtools.interpolate import CubicSplineInterpolant
    from lisatools.detector import Orbits

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
    STFTSettings,
    get_stft_settings,
)
from ..utils.constants import YRSID_SI
from ..utils.parallelbase import LISAToolsParallelModule
from ..utils.utility import tukey

logger = logging.getLogger(__name__)


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


class TDWaveformBase(LISAToolsParallelModule):
    """
    Base class for a waveform built in the time domain.

    Args:
    waveform_t0: Initial time in seconds.
    dt: Time step in seconds.
    Tobs: Observation time in years.
    data_t0: Optional initial time for the data. If None, defaults to waveform_t0. If provided, the output time arrays will be shifted so that the first sample corresponds to a integer multiple of dt after data_t0. This allows for proper alignment of the waveform with an external time grid (e.g. from a loader) when data_t0 is set to the same reference time as the loader.
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
        dt: float,
        Tobs: float,
        data_t0: float = None,
        response_kwargs: dict = None,
        buffer_time: int = 600,
        tukey_alpha: float = 0.01,
        force_backend: str = "cpu",
        force_uniform_stft: bool = False,
        ra_index: int = -3,
        dec_index: int = -2,
        merger_time_index: int = -1,
        num_remove: int = 3,
    ) -> None:

        super().__init__(force_backend=force_backend)

        self.waveform_t0 = waveform_t0
        self.data_t0 = data_t0 if data_t0 is not None else waveform_t0
        self.dt = dt
        self.Tobs = Tobs * YRSID_SI
        self.tukey_alpha = tukey_alpha
        self.force_uniform_stft = force_uniform_stft

        num_points = int(self.Tobs / self.dt)
        response_kwargs["num_pts"] = num_points

        self.response = pyResponseTDI(**response_kwargs, force_backend=force_backend)
        self.buffer_time = buffer_time

        self.ra_index = ra_index
        self.dec_index = dec_index
        self.merger_time_index = merger_time_index
        self.num_remove = num_remove

    @property
    def tdi_buffer_time(self) -> float:
        """Buffer time in seconds to ensure proper TDI response calculation at the boundaries."""
        return 600.0

    @property
    def xp(self):
        """Array module used for calculations."""
        return self.backend.xp

    def wave_gen(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """Generate the waveform for a single source.

        Returns:
            Tuple of (t_arr, h_plus, h_cross).

        """
        raise NotImplementedError("wave_gen method must be implemented in subclass.")

    def wave_gen_batch(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate waveforms for a batch of sources.

        Subclasses that support batched waveform generation should override this
        method to return pre-masked, padded arrays. The batch loop in
        ``_call_batched`` will apply per-source masking and the TDI response.

        Returns:
            Tuple of (times_batch, mask_batch, h_plus_batch, h_cross_batch),
            each of shape (Nbatch, Ntimes). ``mask_batch`` is a boolean array
            selecting the valid (non-padded) time samples for each source.

        """
        raise NotImplementedError(
            "wave_gen_batch is not implemented for this waveform. "
            "Batched calls require implementing wave_gen_batch in the subclass."
        )

    def _apply_response_single(
        self,
        t_arr: np.ndarray | cp.ndarray,
        h_plus: np.ndarray | cp.ndarray,
        h_cross: np.ndarray | cp.ndarray,
        ra: float,
        dec: float,
        merger_time: float,
    ) -> TDSignal:
        """Apply the TDI response to a single source and return a TDSignal.

        Args:
            t_arr: Time array relative to zero (output of wave_gen).
            h_plus: Plus polarization.
            h_cross: Cross polarization.
            ra: Right ascension in radians.
            dec: Declination in radians.
            merger_time: Time of merger in seconds (relative to waveform_t0).

        Returns:
            TDSignal with the TDI response applied.
        """
        shifted_t_arr = t_arr + merger_time + self.waveform_t0
        N_original = shifted_t_arr.shape[-1]
        # add 500 seconds to the end to prevent problems with the response
        
        #shifted_t_arr = self.xp.concatenate([shifted_t_arr, shifted_t_arr[-1] + self.dt * self.xp.arange(1, pad_length + 1)])
        # h_plus = self.xp.pad(h_plus, (0, pad_length), mode='edge')
        # h_cross = self.xp.pad(h_cross, (0, pad_length), mode='edge')

        # pad both sides with zeros by num_pad
        num_pad = int(self.buffer_time / self.dt)

        shifted_t_arr = self.xp.concatenate([
            shifted_t_arr[0] - self.dt * self.xp.arange(1, num_pad + 1),
            shifted_t_arr, 
            shifted_t_arr[-1] + self.dt * self.xp.arange(1, num_pad + 1)])

        h_plus = self.xp.pad(h_plus, (num_pad, num_pad), mode='edge')
        h_cross = self.xp.pad(h_cross, (num_pad, num_pad), mode='edge')

        self.response.num_pts = shifted_t_arr.shape[-1]
    
        strain = h_plus + 1j * h_cross

        self.response.get_projections(
            strain, lam=ra, beta=dec, t0=shifted_t_arr[0], t_buffer=self.buffer_time
        )
        tdis = self.xp.array(self.response.get_tdi_delays())

        # # Zero out the samples affected by the TDI boundary artefacts.
        # tdis[:, : num_pad] = 0.0
        # tdis[:, -num_pad :] = 0.0

        # trim the invalid points 
        tdis = tdis[:, num_pad:-num_pad]
        shifted_t_arr = shifted_t_arr[num_pad:-num_pad]

        # now shift the time arrays so that the abs(t_arr[0] - data_t0) is an integer multiple of dt

        t_arr_shift = (self.data_t0 - shifted_t_arr[0]) % self.dt
        shifted_t_arr += t_arr_shift

        # now remove everything before the start of the data
        start_ind = int((self.data_t0 - shifted_t_arr[0]) / self.dt)
        if start_ind > 0:
            shifted_t_arr = shifted_t_arr[start_ind:]
            tdis = tdis[:, start_ind:]

        td_settings = TDSettings(
            t0=float(shifted_t_arr[0]),
            dt=self.dt,
            N=int(shifted_t_arr.shape[-1]),
            force_backend=self.backend_name.split("_")[-1],
        )
        return TDSignal(arr=tdis, settings=td_settings)

    def _td_to_output_domain(
        self,
        td_signal: TDSignal,
        output_domain: str,
        domain_kwargs: dict,
    ) -> DomainBase:
        """Transform a TDSignal to the specified output domain.

        Args:
            td_signal: Input time-domain signal.
            output_domain: Target domain ('TD', 'STFT', or 'FD').
            domain_kwargs: Extra kwargs forwarded to the domain settings constructor.

        Returns:
            Signal in the requested output domain.
        """
        backend = self.backend_name.split("_")[-1]  # extract 'cpu' or 'cuda12x' from backend name
        output_domain = output_domain.upper()  # allow case-insensitive domain names

        if output_domain == "TD":
            return td_signal

        elif output_domain == "STFT":
            nperseg = round(domain_kwargs["big_dt"] / td_signal.settings.dt)
            td_signal = self._pad_td_signal(td_signal, align_samples=nperseg)
            out_settings = get_stft_settings(
                td_signal.settings.t_arr, **domain_kwargs, force_backend=backend
            )
            nperseg = out_settings.get_nperseg(td_signal.settings.dt)
            window = tukey(nperseg, alpha=self.tukey_alpha, xp=self.xp)

        elif output_domain == "FD":
            # Window at natural signal length FIRST (matches on-the-fly fft() path)
            window = tukey(td_signal.settings.N, alpha=self.tukey_alpha, xp=self.xp)
            windowed_signal = TDSignal(
                arr=td_signal.arr * window,
                settings=td_signal.settings,
            )

            # Then pad the windowed signal (extends with clean zeros)
            N_td_target = round(1 / (domain_kwargs["df"] * td_signal.settings.dt))
            windowed_signal = self._pad_td_signal(
                windowed_signal, align_samples=1, target_n=N_td_target
            )

            out_settings = FDSettings(**domain_kwargs, force_backend=backend)
            # Signal is pre-windowed — pass None so fft() uses ones
            return windowed_signal.transform(out_settings, window=None)

        else:
            raise ValueError(
                f"output_domain must be either 'TD', 'STFT', or 'FD'. "
                f"'WDM' is not supported yet. Got: {output_domain}."
            )
        return td_signal.transform(out_settings, window=window)

    def _pad_td_signal(
        self,
        td_signal: TDSignal,
        align_samples: int,
        target_n: int = None,
    ) -> TDSignal:
        """Pad a TDSignal so its start is aligned with data_t0 and it reaches a target length.

        Left-pads with zeros so that the number of samples between the (new) t0 and
        data_t0 is an integer multiple of ``align_samples``.  For STFT this enforces
        segment-boundary alignment (align_samples = nperseg); for FD pass the full
        time-domain target length as ``align_samples`` so that the signal starts
        exactly at data_t0.

        Then, if ``target_n`` is given, right-pads with zeros so that the total number
        of samples reaches ``target_n`` (ensuring the correct ``df`` after FFT).

        Args:
            td_signal: Input TDSignal (must already be on the dt grid of data_t0).
            align_samples: Left-padding granularity.  The signal is extended so that
                ``round((signal_t0 - data_t0) / dt)`` becomes divisible by this value.
            target_n: If provided, right-pad to at least this many total samples.
        """
        dt = td_signal.settings.dt
        n_to_data_t0 = round((td_signal.settings.t0 - self.data_t0) / dt)

        # Left-pad: absorb the remainder so that the offset from data_t0
        # becomes a multiple of align_samples.
        n_left = n_to_data_t0 % align_samples

        # Right-pad: reach target_n total samples.
        n_right = 0
        if target_n is not None:
            new_n = td_signal.settings.N + n_left
            if new_n < target_n:
                n_right = target_n - new_n

        if n_left == 0 and n_right == 0:
            return td_signal

        pad_width = [(0, 0)] * len(td_signal.outer_shape) + [(n_left, n_right)]
        padded_arr = self.xp.pad(td_signal.arr, pad_width, mode="constant", constant_values=0)
        padded_settings = TDSettings(
            t0=td_signal.settings.t0 - n_left * dt,
            dt=dt,
            N=td_signal.settings.N + n_left + n_right,
            force_backend=self.backend_name.split("_")[-1],
        )
        return TDSignal(arr=padded_arr, settings=padded_settings)

    def _call_batched(
        self,
        *args,
        ra: np.ndarray,
        dec: np.ndarray,
        merger_time: np.ndarray,
        output_domain: str,
        domain_kwargs: dict,
        **kwargs,
    ) -> DomainBaseArray:
        """Handle batched waveform generation and return a DomainBaseArray.

        Loops over the batch dimension for the TDI response (which does not support
        batching natively), then optionally projects all signals onto a common STFT
        grid when ``self.force_uniform_stft`` is True.
        """
        times_batch, mask_batch, hplus_batch, hcross_batch = self.wave_gen_batch(*args, **kwargs)

        Nbatch = times_batch.shape[0]
        td_signals: List[TDSignal] = []

        for i in range(Nbatch):
            mask_i = mask_batch[i]

            t_arr_i = times_batch[i][mask_i]
            hplus_i = hplus_batch[i][mask_i]
            hcross_i = hcross_batch[i][mask_i]

            td_signals.append(
                self._apply_response_single(
                    t_arr_i,
                    hplus_i,
                    hcross_i,
                    float(ra[i]),
                    float(dec[i]),
                    float(merger_time[i]),
                )
            )

        if output_domain == "TD":
            return DomainBaseArray(td_signals)

        if output_domain == "STFT" and self.force_uniform_stft:
            return self._to_uniform_stft(td_signals, domain_kwargs)

        # Natural (non-uniform) path: transform each signal with its own settings.
        return DomainBaseArray(
            [self._td_to_output_domain(s, output_domain, domain_kwargs) for s in td_signals]
        )

    def _to_uniform_stft(
        self,
        td_signals: List[TDSignal],
        domain_kwargs: dict,
    ) -> DomainBaseArray:
        """
        Project all TDSignals onto a common STFT grid spanning the union of their time ranges.

        Sources whose time range is shorter than the global span are zero-padded at
        the appropriate boundary so that all signals share exactly the same
        STFTSettings, yielding a uniform DomainBaseArray.
        """
        # Determine the global time span.
        waveform_t0_global = min(s.settings.waveform_t0 for s in td_signals)
        t_end_global = max(
            s.settings.waveform_t0 + s.settings.N * s.settings.dt for s in td_signals
        )
        N_global = int(round((t_end_global - waveform_t0_global) / self.dt))

        # Derive a common STFTSettings from the global time grid.
        ref_t_arr = self.xp.arange(N_global) * self.dt + waveform_t0_global
        common_settings = get_stft_settings(ref_t_arr, **domain_kwargs, force_backend=self.backend)
        nperseg = common_settings.get_nperseg(self.dt)
        window = tukey(nperseg, alpha=self.tukey_alpha, xp=self.xp)

        signals = []
        for td_sig in td_signals:
            left_pad = int(round((td_sig.settings.waveform_t0 - waveform_t0_global) / self.dt))
            right_pad = max(N_global - left_pad - td_sig.settings.N, 0)

            # pad_width: keep all outer dims (channels) intact, pad only the time axis.
            pad_width = [(0, 0)] * len(td_sig.outer_shape) + [(left_pad, right_pad)]
            padded_arr = self.xp.pad(td_sig.arr, pad_width, mode="constant", constant_values=0)
            padded_settings = TDSettings(
                waveform_t0=waveform_t0_global, dt=self.dt, N=N_global, force_backend=self.backend
            )
            padded_td = TDSignal(arr=padded_arr, settings=padded_settings)
            signals.append(padded_td.transform(common_settings, window=window))

        return DomainBaseArray(signals)

    def _extract_sky_params(
        self,
        args: tuple,
        ra: float | np.ndarray | None,
        dec: float | np.ndarray | None,
        merger_time: float | np.ndarray | None,
    ) -> tuple:
        """Split sky/response params from the positional argument tuple.

        If ``ra``, ``dec`` and ``merger_time`` are all ``None`` **and** the
        positional ``args`` contain at least ``n_sky_params`` extra entries
        beyond what ``wave_gen`` expects, the last ``n_sky_params`` values are
        peeled off and returned as ``(ra, dec, merger_time)``.

        This allows callers to pass the *full* parameter vector positionally
        (``wave_gen(*params)``) without needing to know which indices
        correspond to sky/response parameters.

        Returns:
            ``(waveform_args, ra, dec, merger_time)``
        """
        if ra is None and dec is None and merger_time is None:
            n = self.num_remove
            if len(args) < n:
                raise TypeError(
                    f"TDWaveformBase.__call__() requires 'ra', 'dec', and "
                    f"'merger_time' either as keyword arguments or as the "
                    f"last {n} positional arguments."
                )

            ra = args[self.ra_index]
            dec = args[self.dec_index]
            merger_time = args[self.merger_time_index]

            wf_args = args[:-n] if n > 0 else args

            return wf_args, ra, dec, merger_time
        return args, ra, dec, merger_time

    def __call__(
        self,
        *args,
        ra: float | np.ndarray = None,
        dec: float | np.ndarray = None,
        merger_time: float | np.ndarray = None,
        output_domain: str = "TD",
        domain_kwargs: dict = None,
        **kwargs,
    ) -> DomainBase | DomainBaseArray:
        """
        Generate the waveform and return the signal in the specified output domain.

        When ``ra`` is a 1-D array the call is treated as batched: ``wave_gen_batch``
        is invoked and a :class:`DomainBaseArray` is returned.  For scalar ``ra`` the
        single-source path is used and a :class:`DomainBase` is returned.

        Sky/response parameters (``ra``, ``dec``, ``merger_time``) can be
        passed either as explicit keyword arguments **or** as the last
        ``n_sky_params`` positional arguments.  The latter allows the common
        calling convention ``wave_gen(*full_param_vector, **kwargs)`` used
        throughout the global-fit pipeline.

        Args:
            *args: Arguments for the wave_gen / wave_gen_batch method.
                When ``ra``/``dec``/``merger_time`` are not given as keywords,
                the last 3 positional arguments are interpreted as
                ``(ra, dec, merger_time)`` and the remaining ones are
                forwarded to ``wave_gen``.
            ra: Right ascension in radians.  Scalar for single source, 1-D array for batch.
            dec: Declination in radians.  Same shape as ``ra``.
            merger_time: Time of merger in seconds.  Same shape as ``ra``.
            output_domain: Target output domain ('TD', 'STFT', or 'FD').
            domain_kwargs: Extra keyword arguments forwarded to the domain settings constructor.
            **kwargs: Keyword arguments for the wave_gen / wave_gen_batch method.

        Returns:
            Signal in the specified output domain.  A single :class:`DomainBase` for
            scalar ``ra``, a :class:`DomainBaseArray` for array ``ra``.
        """
        args, ra, dec, merger_time = self._extract_sky_params(args, ra, dec, merger_time)

        if np.ndim(ra) >= 1:
            return self._call_batched(
                *args,
                ra=ra,
                dec=dec,
                merger_time=merger_time,
                output_domain=output_domain,
                domain_kwargs=domain_kwargs,
                **kwargs,
            )

        else:
            # Single-source path.
            t_arr, h_plus, h_cross = self.wave_gen(*args, **kwargs)

            td_signal = self._apply_response_single(t_arr, h_plus, h_cross, ra, dec, merger_time)

            return self._td_to_output_domain(td_signal, output_domain, domain_kwargs)


class TDTDIOnFlyWaveformBase(LISAToolsParallelModule):
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
        tdi_config: str | TDIConfig = "2nd generation",
        sampling_frequency: float = 0.4,
        orbits: Orbits = None,
        zero_inclination: bool = False,
        tukey_alpha: float = 0.01,
        stft_dt: float = None,
        freq_min: float = 0.0,
        freq_max: float = 1.0,
        force_backend: str = "cpu",
    ) -> None:
        super().__init__(force_backend=force_backend)

        self.waveform_t0 = waveform_t0
        self.domain_settings = data_td_settings
        self.tukey_alpha = tukey_alpha
        if isinstance(tdi_config, str):
            self.tdi_config = TDIConfig(tdi=tdi_config, force_backend=force_backend)
        else:
            self.tdi_config = tdi_config

        self.sampling_frequency = sampling_frequency
        self.orbits = orbits
        self.zero_inclination = zero_inclination

        if stft_dt is None:
            logger.info(
                "No stft timestep provided. The waveform will be transformed in the frequency domain"
            )
            self.transform_to_domain = self.fft
            self.nperseg = None
            self.num_ll_args = 2  # fd_signal and starting frequencies

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
            self.num_ll_args = 3  # stft signal, start freqs and start times

        self.freq_min = freq_min
        self.freq_max = freq_max

    @property
    def max_length(self) -> int:
        """maximum number of evaluation time points that can be stored in a gpu register"""
        return 20_000

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
    def data_times_array(self):
        """Complete time array on which the data live."""
        return self.domain_settings.t_arr

    @property
    def xp(self):
        """Array module used for calculations."""
        return self.backend.xp

    @property
    def tdi_config(self):
        """TDI configuration for on-the-fly response generation."""
        return self._tdi_config

    @tdi_config.setter
    def tdi_config(self, config: TDIConfig):
        """Set the TDI configuration"""
        self._tdi_config = config

    @property
    def orbits(self):
        """Orbits object for on-the-fly response generation."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits):
        """Set the Orbits object."""
        assert (
            orbits is not None
        ), "Orbits object must be provided for on-the-fly response generation."

        if not orbits.configured:
            orbits.configure(linear_interp_setup=True)
        self._orbits = orbits

    @property
    def sampling_frequency(self):
        """Sampling frequency for the on-the-fly response generation."""
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, fs: float):
        """Set the sampling frequency for the on-the-fly response generation."""
        assert fs > 0, "Sampling frequency must be positive."
        self._sampling_frequency = fs

    @property
    def freq_min(self):
        return self._freq_min

    @freq_min.setter
    def freq_min(self, value: float):
        self._freq_min = value

    @property
    def freq_max(self):
        return self._freq_max

    @freq_max.setter
    def freq_max(self, value: float):
        self._freq_max = value

    @property
    def nperseg(self):
        return self._nperseg

    @nperseg.setter
    def nperseg(self, value: float):
        self._nperseg = value

    @property
    def analysis_domain(self) -> str:
        if self.nperseg:
            return "STFT"
        else:
            return "FD"

    def get_output_settings(self, eval_times: np.ndarray | cp.ndarray) -> DomainSettingsBase:
        """
        Get the settings for the output domain based on the evaluation times and the chosen analysis domain (STFT or FD).
        """
        if self.analysis_domain == "STFT":
            return get_stft_settings(
                eval_times,
                big_dt=self.nperseg * self.dt,
                min_freq=self.freq_min,
                max_freq=self.freq_max,
                force_backend=self.backend_name.split("_")[-1],
            )
        else:
            data_N = self.domain_settings.N
            return FDSettings(
                N=data_N // 2 + 1,
                df=1 / (data_N * self.dt),
                min_freq=self.freq_min,
                max_freq=self.freq_max,
                force_backend=self.backend_name.split("_")[-1],
            )

    def get_grid_time(self, times: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
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

    def get_amp_phase(
        self,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Generate amplitude and phase arrays for each mode of a batch of sources.
        Returns also the time array.

        Returns:
            Tuple of (t_arr, amp, phase).

        """
        raise NotImplementedError("amp_phase_gen method must be implemented in subclass.")

    def process_amp_phase(
        self, amp: np.ndarray | cp.ndarray, phase: np.ndarray | cp.ndarray
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
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

    def stack_parameter(self, param: np.ndarray, num_modes: int) -> np.ndarray | cp.ndarray:
        """
        Stack a parameter array for use in the TDI on-the-fly response generator.
        Given a parameter array of shape (Nbatch,), stack it to shape (Nbatch * num_modes,) by repeating each entry num_modes times. This is needed to match the expected input shape for the TDI on-the-fly response generator when using multiple modes per source.
        """
        param = self.xp.asarray(param)
        return self.xp.repeat(param, num_modes)

    def get_evaluation_times(self, input_times: np.ndarray | cp.ndarray, num_modes: int) -> np.ndarray | cp.ndarray:
        """
        Get the time array on which to evaluate the TDI on-the-fly response. By default, this uses the same as the input time array from the amplitude and phase generation, but subclasses can override this method to define a different evaluation grid if needed (e.g. a regular grid).

        Args:
            input_times (Array): times used for amp/phase generation, shape (num_binaries  * num_modes, num_times)
            num_modes (int): number of modes to repeat the times array by.

        Returns:
            evaluation times of shape (num_binaries * num_modes, num_eval_times).
        """
        delta_t = self.xp.diff(input_times, axis=-1)

        start_buffer = int(self.tdi_buffer_time / self.xp.min(delta_t[:, 0]))
        end_buffer = int(self.tdi_buffer_time / self.xp.min(delta_t[:, -1]))

        evaluation_times = input_times[:, start_buffer:-end_buffer][:, -self.max_length :]

        return evaluation_times

    def get_dense_times(self, eval_times: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """
        Get a dense time array on which to evaluate the TDI on-the-fly response. This can be used to ensure that the output response is sampled on a regular grid, even if the input amplitude and phase arrays are sampled irregularly.
        """
        # consider using powers of 2.
        # if self.analysis_domain == "FD":
        #     num_sub = eval_times.shape[0]
        #     return self.xp.repeat(self.data_times_array[None, :], num_sub, axis=0)

        # for the STFT case, we want to define a regular grid for each source that covers the time range of the input eval_times for that source, with a spacing of self.dt. This ensures that the output STFT is sampled on a regular grid, even if the input eval_times are not.
        # get the start and end times from the evaluation times
        start_times = self.get_grid_time(eval_times[:, 0])
        end_times = self.get_grid_time(eval_times[:, -1])

        num_times = int(self.xp.max((end_times - start_times) / self.dt) + 1)
        if self.analysis_domain == 'STFT':
            # make sure num_times is a multiple of nperseg for the STFT case
            num_times = int(np.ceil(num_times / self.nperseg) * self.nperseg)

        return start_times[:, None] + self.xp.arange(num_times)[None, :] * self.dt

    def fft(
        self, times_in: np.ndarray | cp.ndarray, signal_in: np.ndarray | cp.ndarray
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Transform the time domain data to the FT basis with the chosen settings.

        Args:
            times_in (Array): Time grid of the input signal with shape `(num_binaries, num_times)`. Ignored here but included in the signature for consistency with the `stft` method.
            signal_in (Array): Time domain input signal with shape `(num_binaries, num_times)`

        Returns:
            signal_out (Array): The transformed signal with shape `(num_binaries, num_freqs)`
            start_freqs (Array): Starting frequencies to be fed to the likelihood calculation Shape: `(num_binaries,)`
        """
        # for the moment we are gonna evaluate the likelihood on the same frequency support for every source
        num_binaries = signal_in.shape[0]
        n = signal_in.shape[-1]
        window = tukey(n, alpha=self.tukey_alpha, xp=self.xp)

        n_fft = self.domain_settings.N
        windowed = signal_in * window

        # Explicit zero-padding — window covers signal, zeros extend after
        if n < n_fft:
            pad_width = [(0, 0)] * (windowed.ndim - 1) + [(0, n_fft - n)]
            windowed = self.xp.pad(windowed, pad_width, mode="constant")

        signal_fd = self.xp.fft.rfft(windowed, axis=-1) * self.dt
        freqs = self.xp.fft.rfftfreq(n_fft, d=self.dt)

        keep = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        signal_out = signal_fd[..., keep]

        start_freqs = np.full(shape=num_binaries, fill_value=self.xp.min(freqs[keep]))

        return signal_out, start_freqs

    def stft(
        self,
        times_in: np.ndarray | cp.ndarray,
        signal_in: np.ndarray | cp.ndarray,
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Transform the time domain data to the STFT basis with the chosen settings.

        Args:
            times_in (Array): Time grid of the input signal with shape `(num_binaries, num_times)`
            signal_in (Array): Time domain input signal with shape `(num_binaries, num_times)`

        Returns:
            signal_out (Array): The transformed signal with shape `(num_binaries, num_stft_times, num_freqs)`
            start_freqs (Array): Starting frequencies to be fed to the likelihood calculation Shape: `(num_binaries,)`
            start_times (Array): Starting times to be fed to the likelihood calculation Shape: `(num_binaries,)`
        """

        num_binaries, num_channels = signal_in.shape[:2]
        signal_in = signal_in.reshape(
            num_binaries, num_channels, -1, self.nperseg
        )  # (num_binaries, num_channels, num_segments, num_times_per_segment)
        start_times = times_in[:, 0]
        dt = (
            times_in[0, 1] - times_in[0, 0]
        )  # here the time grid is dense for every source, so we can just take the first one to compute dt

        signal_out, start_freqs = self.fft(times_in, signal_in)

        return signal_out, start_freqs, start_times
    
    def pad(
            self, 
            input_times: np.ndarray | cp.ndarray,
            input_amplitudes: np.ndarray | cp.ndarray, 
            input_phases: np.ndarray | cp.ndarray
        ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
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
        left_dt = self.xp.max(delta_t[:, 0])
        right_dt = self.xp.max(delta_t[:, -1])

        pad_length_left = max(int(self.tdi_buffer_time / left_dt), 1)
        pad_length_right = max(int(self.tdi_buffer_time / right_dt), 1)

        padded_times = self.xp.concatenate(
            [
                input_times[:, 0:1] - self.xp.arange(pad_length_left, 0, -1) * left_dt,
                input_times,
                input_times[:, -1:] + self.xp.arange(1, pad_length_right + 1) * right_dt,
            ],
            axis=-1
        )

        padded_amplitudes = self.xp.pad(input_amplitudes, ((0, 0), (pad_length_left, pad_length_right)), mode="edge")
        padded_phases = self.xp.pad(input_phases, ((0, 0), (pad_length_left, pad_length_right)), mode="edge")

        return padded_times, padded_amplitudes, padded_phases

    def compute_tdi_channels(
        self,
        *args,
        inclination: np.ndarray | cp.ndarray,
        psi: np.ndarray | cp.ndarray,
        ra: np.ndarray | cp.ndarray,
        dec: np.ndarray | cp.ndarray,
        merger_time: np.ndarray | cp.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
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

        # Ensure all inputs are at least 1-D for batch processing
        args = tuple(self.xp.atleast_1d(self.xp.asarray(a)) for a in args)
        inclination = self.xp.atleast_1d(self.xp.asarray(inclination))
        psi = self.xp.atleast_1d(self.xp.asarray(psi))
        ra = self.xp.atleast_1d(self.xp.asarray(ra))
        dec = self.xp.atleast_1d(self.xp.asarray(dec))
        merger_time = self.xp.atleast_1d(self.xp.asarray(merger_time))

        # step 1: generate the amplitude and phase arrays for each mode of each source:
        input_times, input_amplitudes, input_phases = self.get_amp_phase(
            *args, inclination, psi, ra, dec, **kwargs
        )  # todo fixup?

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
        evaluation_times = self.get_evaluation_times(input_times, num_modes)

        # now padd with 500 seconds on each side
        padded_times, padded_amplitudes, padded_phases = self.pad(input_times, input_amplitudes, input_phases)

        tdi_generator = TDTDIonTheFly(
            evaluation_times,
            padded_amplitudes,
            padded_phases,
            sampling_frequency=self.sampling_frequency,
            num_sub=num_sub,
            t_input=padded_times,
            tdi_config=self.tdi_config,
            orbits=self.orbits,
            force_backend=self.backend_name.split("_")[-1],
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

        # step 6: evaluate the TDI response on the dense time array
        tdi_out = tdi_spline.eval_tdi(dense_times, error_out_of_bounds=False).reshape(
            num_binaries, num_modes, self.tdi_config.nchannels, -1
        )

        tdi_channels = self.xp.sum(
            tdi_out, axis=1
        )  # sum over modes to get the total response for each binary
        dense_times = dense_times.reshape(num_binaries, num_modes, -1)[
            :, 0
        ]  # reshape and take the time array from the first mode (all modes share the same time array)

        return dense_times, tdi_channels

    def __call__(
        self,
        *args,
        inclination: np.ndarray | cp.ndarray = None,
        psi: np.ndarray | cp.ndarray = None,
        ra: np.ndarray | cp.ndarray = None,
        dec: np.ndarray | cp.ndarray = None,
        merger_time: np.ndarray | cp.ndarray = None,
        **kwargs,
    ) -> (
        Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]
        | Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]
    ):
        """
        Generate the on-the-fly response for a batch of sources and transform it to the desired domain.

        Args:
            *args: Arguments for the amplitude and phase generation method.
                The last 5 positional arguments can be (inclination, psi, ra, dec, merger_time)
                if not provided as keyword arguments.
            inclination: Inclination angles for the sources, shape (Nbatch,).
            psi: Polarization angles for the sources, shape (Nbatch,).
            ra: Right ascension for the sources, shape (Nbatch,).
            dec: Declination for the sources, shape (Nbatch,).
            merger_time: Merger time with respect to `self.waveform_t0` in seconds, shape (Nbatch,).
            **kwargs: Keyword arguments for the amplitude and phase generation method.

        Returns:
            If self.transform_to_domain is self.fft: Tuple of (signal_out, start_freqs) where signal_out has shape (Nbatch, num_freqs) and start_freqs has shape (Nbatch,).
            If self.transform_to_domain is self.stft: Tuple of (signal_out, start_freqs, start_times) where signal_out has shape (Nbatch, num_stft_times, num_freqs), start_freqs has shape (Nbatch,), and start_times has shape (Nbatch,).
        """

        if inclination is None:
            *args, inclination, psi, ra, dec, merger_time = args

        dense_times, tdi_channels = self.compute_tdi_channels(
            *args,
            inclination=inclination,
            psi=psi,
            ra=ra,
            dec=dec,
            merger_time=merger_time,
            **kwargs,
        )

        return self.transform_to_domain(dense_times, tdi_channels)

    def get_signals_for_residuals(
        self,
        *args,
        inclination: np.ndarray | cp.ndarray = None,
        psi: np.ndarray | cp.ndarray = None,
        ra: np.ndarray | cp.ndarray = None,
        dec: np.ndarray | cp.ndarray = None,
        merger_time: np.ndarray | cp.ndarray = None,
        **kwargs,
    ) -> List[DomainBase]:
        """
        Generate the on-the-fly response for a batch of sources and return the domain wrapped signals for residual operations.

        Args:
            *args: Arguments for the amplitude and phase generation method.
                The last 5 positional arguments can be (inclination, psi, ra, dec, merger_time)
                if not provided as keyword arguments.
            inclination: Inclination angles for the sources, shape (Nbatch,).
            psi: Polarization angles for the sources, shape (Nbatch,).
            ra: Right ascension for the sources, shape (Nbatch,).
            dec: Declination for the sources, shape (Nbatch,).
            merger_time: Merger time with respect to `self.waveform_t0` in seconds, shape (Nbatch,).
            **kwargs: Keyword arguments for the amplitude and phase generation method.

        Returns:
            List of DomainBase objects containing the signals for each source, transformed to the desired output domain.
        """

        if inclination is None:
            *args, inclination, psi, ra, dec, merger_time = args

        dense_times, tdi_channels = self.compute_tdi_channels(
            *args,
            inclination=inclination,
            psi=psi,
            ra=ra,
            dec=dec,
            merger_time=merger_time,
            **kwargs,
        )

        signals_out = []
        for i in range(tdi_channels.shape[0]):
            td_signal = TDSignal(
                arr=tdi_channels[i],
                settings=TDSettings(
                    t0=dense_times[i, 0],
                    dt=self.dt,
                    N=dense_times.shape[-1],
                    force_backend=self.backend_name.split("_")[-1],
                ),
            )

            output_domain = self.get_output_settings(dense_times[i])

            if isinstance(output_domain, FDSettings):
                # Window at natural signal length FIRST (matches on-the-fly fft() path)
                natural_n = td_signal.settings.N
                window = tukey(natural_n, alpha=self.tukey_alpha, xp=self.xp)
                windowed_arr = td_signal.arr * window

                # Then pad the windowed signal to target FFT length
                n_fft = round(1 / (output_domain.df * td_signal.settings.dt))
                n_pad = n_fft - natural_n
                if n_pad > 0:
                    pad_width = [(0, 0)] * (windowed_arr.ndim - 1) + [(0, n_pad)]
                    windowed_arr = self.xp.pad(windowed_arr, pad_width, mode="constant")

                td_signal = TDSignal(
                    arr=windowed_arr,
                    settings=TDSettings(
                        t0=td_signal.settings.t0,
                        dt=self.dt,
                        N=n_fft,
                        force_backend=self.backend_name.split("_")[-1],
                    ),
                )
                # Signal is pre-windowed — pass None so fft() uses ones
                signals_out.append(td_signal.transform(new_domain=output_domain, window=None))
            else:
                # STFT case — window is per-segment, handled by stft() internally
                window_n = self.nperseg
                window = tukey(window_n, alpha=self.tukey_alpha, xp=self.xp)
                signals_out.append(td_signal.transform(new_domain=output_domain, window=window))

        return signals_out
