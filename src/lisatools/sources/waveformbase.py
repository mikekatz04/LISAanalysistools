from abc import ABC
from typing import List, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:
    import numpy as cp

from fastlisaresponse import pyResponseTDI
from lisaconstants import ASTRONOMICAL_YEAR as YRSID_SI

from ..domains import (DomainBase, DomainBaseArray, FDSettings, TDSettings,
                       TDSignal, get_stft_settings)
from ..utils.utility import tukey


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


class TDWaveformBase(ABC):
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
    ) -> None:

        self.waveform_t0 = waveform_t0
        self.data_t0 = data_t0 if data_t0 is not None else waveform_t0
        self.dt = dt
        self.Tobs = Tobs * YRSID_SI
        self.tukey_alpha = tukey_alpha
        self.force_uniform_stft = force_uniform_stft

        num_points = int(self.Tobs / self.dt)
        response_kwargs["num_pts"] = num_points
        response_kwargs["force_backend"] = force_backend
        self.backend = force_backend

        self.response = pyResponseTDI(**response_kwargs)
        self.buffer_time = buffer_time

    @property
    def xp(self):
        """Array module used for calculations."""
        return self.response.xp

    def wave_gen(
        self, *args, **kwargs
    ) -> Tuple[
        np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray
    ]:
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
        shifted_t_arr = (t_arr + merger_time + self.waveform_t0).copy()
        self.response.num_pts = shifted_t_arr.shape[-1]

        strain = h_plus + 1j * h_cross

        self.response.get_projections(
            strain, lam=ra, beta=dec, t0=shifted_t_arr[0], t_buffer=self.buffer_time
        )
        tdis = self.xp.array(self.response.get_tdi_delays())

        # Zero out the samples affected by the TDI boundary artefacts.
        tdis[:, : self.response.tdi_start_ind] = 0.0
        tdis[:, -self.response.tdi_start_ind :] = 0.0

        # now shift the time arrays so that the abs(t_arr[0] - data_t0) is an integer multiple of dt

        t_arr_shift = (self.data_t0 - shifted_t_arr[0]) % self.dt
        shifted_t_arr += t_arr_shift

        # now remove everything before the start of the data
        start_ind = int((self.data_t0 - shifted_t_arr[0]) / self.dt)
        if start_ind > 0:
            shifted_t_arr = shifted_t_arr[start_ind:]
            tdis = tdis[:, start_ind:]
            t_arr = t_arr[start_ind:]


        td_settings = TDSettings(
            t0=float(shifted_t_arr[0]),
            dt=self.dt,
            N=int(shifted_t_arr.shape[-1]),
            force_backend=self.backend,
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
        if output_domain == "TD":
            return td_signal

        elif output_domain == "STFT":
            # Derive STFT settings from the signal's own time grid.
            td_signal = self.pad_td_signal_for_stft(td_signal, domain_kwargs)
            out_settings = get_stft_settings(
                td_signal.settings.t_arr, **domain_kwargs, force_backend=self.backend
            )
            nperseg = out_settings.get_nperseg(td_signal.settings.dt)
            window = tukey(nperseg, alpha=self.tukey_alpha, xp=self.xp)

        elif output_domain == "FD":
            out_settings = FDSettings(**domain_kwargs, force_backend=self.backend)
            window = tukey(td_signal.settings.N, alpha=self.tukey_alpha, xp=self.xp)

        else:
            raise ValueError(
                f"output_domain must be either 'TD', 'STFT', or 'FD'. "
                f"'WDM' is not supported yet. Got: {output_domain}."
            )
        return td_signal.transform(out_settings, window=window)
    
    def pad_td_signal_for_stft(self, td_signal: TDSignal, domain_kwargs: dict) -> TDSignal:
        """
        Pad a TDSignal with zeros if the initial timepoint would not be aligned with the STFT grid used for the data.

        Args:
            td_signal: Input TDSignal to be padded if necessary.
            domain_kwargs: Keyword arguments for deriving the STFTSettings, used to determine the STFT grid.
        """
        nperseg = round( domain_kwargs['big_dt'] / td_signal.settings.dt)
        # now check if there is a integer number of nperseg samples between td_signal.settings.t0 and self.data_t0
        n_samples_to_data_t0 = round((self.data_t0 - td_signal.settings.t0) / td_signal.settings.dt)
        if n_samples_to_data_t0 % nperseg == 0:
            # already aligned, no padding needed
            return td_signal
        
        else:
            # need to pad with zeros at the beginning of the signal
            n_pad = nperseg - (n_samples_to_data_t0 % nperseg)
            # print(f"Padding TDSignal with {n_pad} zeros to align with STFT grid.")
            pad_width = [(0, 0)] * len(td_signal.outer_shape) + [(n_pad, 0)]
            padded_arr = self.xp.pad(
                td_signal.arr, pad_width, mode="constant", constant_values=0
            )
            padded_settings = TDSettings(
                t0=td_signal.settings.t0 - n_pad * td_signal.settings.dt,
                dt=td_signal.settings.dt,
                N=td_signal.settings.N + n_pad,
                force_backend=self.backend,
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
        times_batch, mask_batch, hplus_batch, hcross_batch = self.wave_gen_batch(
            *args, **kwargs
        )

        Nbatch = times_batch.shape[0]
        td_signals: List[TDSignal] = []

        for i in range(Nbatch):
            mask_i = mask_batch[i]

            t_arr_i = self.xp.asarray(times_batch[i][mask_i])
            hplus_i = self.xp.asarray(hplus_batch[i][mask_i])
            hcross_i = self.xp.asarray(hcross_batch[i][mask_i])

            td_signals.append(
                self._apply_response_single(
                    t_arr_i, hplus_i, hcross_i,
                    float(ra[i]), float(dec[i]), float(merger_time[i]),
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
        common_settings = get_stft_settings(
            ref_t_arr, **domain_kwargs, force_backend=self.backend
        )
        nperseg = common_settings.get_nperseg(self.dt)
        window = tukey(nperseg, alpha=self.tukey_alpha, xp=self.xp)

        signals = []
        for td_sig in td_signals:
            left_pad = int(round((td_sig.settings.waveform_t0 - waveform_t0_global) / self.dt))
            right_pad = max(N_global - left_pad - td_sig.settings.N, 0)

            # pad_width: keep all outer dims (channels) intact, pad only the time axis.
            pad_width = [(0, 0)] * len(td_sig.outer_shape) + [(left_pad, right_pad)]
            padded_arr = self.xp.pad(
                td_sig.arr, pad_width, mode="constant", constant_values=0
            )
            padded_settings = TDSettings(
                waveform_t0=waveform_t0_global, dt=self.dt, N=N_global, force_backend=self.backend
            )
            padded_td = TDSignal(arr=padded_arr, settings=padded_settings)
            signals.append(padded_td.transform(common_settings, window=window))

        return DomainBaseArray(signals)

    def __call__(
        self,
        *args,
        ra: float | np.ndarray,
        dec: float | np.ndarray,
        merger_time: float | np.ndarray,
        output_domain: str = "TD",
        domain_kwargs: dict = None,
        **kwargs,
    ) -> DomainBase | DomainBaseArray:
        """
        Generate the waveform and return the signal in the specified output domain.

        When ``ra`` is a 1-D array the call is treated as batched: ``wave_gen_batch``
        is invoked and a :class:`DomainBaseArray` is returned.  For scalar ``ra`` the
        single-source path is used and a :class:`DomainBase` is returned.

        Args:
            *args: Arguments for the wave_gen / wave_gen_batch method.
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

            td_signal = self._apply_response_single(
                t_arr, h_plus, h_cross, ra, dec, merger_time
            )
            
            return self._td_to_output_domain(td_signal, output_domain, domain_kwargs)
