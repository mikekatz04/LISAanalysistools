from abc import ABC
from typing import Tuple, Union

import numpy as np

try:
    import cupy as cp
except ImportError:
    import numpy as cp

from fastlisaresponse import pyResponseTDI
from lisaconstants import ASTRONOMICAL_YEAR as YRSID_SI

from ..domains import (DomainBase, DomainSettingsBase, FDSettings,
                       STFTSettings, TDSettings, TDSignal, get_stft_settings)
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
    t0: Initial time in seconds.
    dt: Time step in seconds.
    Tobs: Observation time in years.
    response_kwargs: Keyword arguments for the TDI response.
    buffer_time: Time in seconds to add as buffer to the TDI response to ensure proper calculation at the beginning and end of the signal.
    tukey_alpha: Alpha parameter for the Tukey window applied to the output signal. Only applied if settings_class is not None.

    """

    def __init__(
        self,
        t0: float,
        dt: float,
        Tobs: float,
        response_kwargs: dict = None,
        buffer_time: int = 600,
        tukey_alpha: float = 0.01,
        force_backend: str = "cpu",
    ) -> None:

        self.t0 = t0
        self.dt = dt
        self.Tobs = Tobs * YRSID_SI
        self.tukey_alpha = tukey_alpha

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
        """Generate the waveform.

        Returns:
            Tuple of (t_arr, h_plus, h_cross).

        """
        raise NotImplementedError("wave_gen method must be implemented in subclass.")

    def __call__(
        self,
        *args,
        ra: float | np.ndarray,
        dec: float | np.ndarray,
        merger_time: float | np.ndarray,
        output_domain: str = "TD",
        domain_kwargs: dict = None,
        **kwargs,
    ) -> DomainBase:
        """
        Generate the waveform and return the signal in the specified output domain.

        Args:
            *args: Arguments for the wave_gen method.
            ra: Right ascension in radians.
            dec: Declination in radians.
            merger_time: Time of merger in seconds.
            **kwargs: Keyword arguments for the wave_gen method.

        Returns:
            Signal in the specified output domain.
        """
        t_arr, h_plus, h_cross = self.wave_gen(*args, **kwargs)

        # add the buffer time to the time array to ensure that the TDI response is properly calculated at the beginning and end of the signal.

        # update the number of points in the output
        self.response.num_pts = t_arr.shape[-1]

        # shift the time array by the merger time and the initial offset
        merger_time = int((self.t0 + merger_time) / self.dt) * self.dt

        t_arr += merger_time

        strain = h_plus + 1j * h_cross

        self.response.get_projections(
            strain, lam=ra, beta=dec, t0=t_arr[0], t_buffer=self.buffer_time
        )
        tdis = self.xp.array(self.response.get_tdi_delays())

        # now set to zeros the points before the start and end of the TDI range
        tdis[:, : self.response.tdi_start_ind] = 0.0
        tdis[:, -self.response.tdi_start_ind :] = 0.0

        td_settings = TDSettings(t0=float(t_arr[0]), dt=self.dt, N=int(t_arr.shape[-1]), force_backend=self.backend)
        td_signal = TDSignal(arr=tdis, settings=td_settings)

        # now prepare the output. If the output domain is TD return the TDSignal
        if output_domain == "TD":
            return td_signal

        elif output_domain == "STFT":

            out_settings = get_stft_settings(t_arr, **domain_kwargs, force_backend=self.backend)
            nperseg = out_settings.get_nperseg(td_settings.dt)

            window = tukey(nperseg, alpha=self.tukey_alpha, xp=self.xp)

        elif output_domain == "FD":
            out_settings = FDSettings(**domain_kwargs, force_backend=self.backend)
            window = tukey(td_settings.N, alpha=self.tukey_alpha, xp=self.xp)

        else:
            raise ValueError(
                f"output_domain must be either 'TD', 'STFT', or 'FD'. 'WDM' is not supported yet. Got: {output_domain}."
            )

        return td_signal.transform(out_settings, window=window)
