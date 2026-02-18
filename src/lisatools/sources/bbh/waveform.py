from copy import deepcopy
from typing import Any, Optional, Tuple

import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
# imports
from fastlisaresponse import ResponseWrapper

from ...domains import DomainSettingsBase
from ...utils.constants import *
from ..waveformbase import SNRWaveform, TDWaveformBase

try:
    import phentax

    phentax_available = True
except (ImportError, ModuleNotFoundError):
    phentax_available = False


class BBHSNRWaveform(SNRWaveform):
    """Wrapper class for straightforward BBH SNR calculations.

    Calculates it for A and E channels in TDI2.

    Args:
        bbh_waveform_kwargs: ``amp_phase_kwargs`` for :class:`BBHWaveformFD`.
        response_kwargs: ``response_kwargs`` for :class:`BBHWaveformFD`.

    """

    def __init__(
        self,
        bbh_waveform_kwargs: Optional[dict] = {"run_phenomd": False},
        response_kwargs: Optional[dict] = {"TDItag": "AET", "tdi2": True},
    ) -> None:

        if "TDItag" not in response_kwargs:
            response_kwargs["TDItag"] = "AET"

        if "tdi2" not in response_kwargs:
            response_kwargs["tdi2"] = True

        # wave generating class
        self.wave_gen = BBHWaveformFD(
            amp_phase_kwargs=bbh_waveform_kwargs,
            response_kwargs=response_kwargs,
        )

    @property
    def f_arr(self) -> np.ndarray:
        """Frequency array."""
        return self._f_arr

    @f_arr.setter
    def f_arr(self, f_arr: np.ndarray) -> None:
        """Set frequency array."""
        self._f_arr = f_arr

    def __call__(
        self,
        *params: Any,
        return_array: Optional[bool] = False,
        mf_min: Optional[float] = 1e-4,
        mf_max: Optional[float] = 0.6,
        freqs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Generate waveforms for SNR calculations.

        Args:
            *params: Parameters for the ``__call__`` function
                for :class:`BBHWaveformFD`.
            return_array: If ``True``, return ``array([A, E, T]).
                If ``False``, return (A, E, T).
            mf_min: Minimum dimensionless frequency to evaluate.
            mf_max: Maximum dimensionless frequency to evaluate.
            freqs: If ``None``, then default will be ``np.logspace(mf_min / M, mf_max / M, 1024)``.
                Otherwise, it will calulate frequencies based on this exact array.
            **kwargs: ``kwargs`` for the ``__call__`` function
                for :class:`BBHWaveformFD`.

        Returns:
            Output waveform.

        """

        # determine frequency array (sparse, log-spaced)
        m1 = params[0]
        m2 = params[1]

        if freqs is None:
            min_f = mf_min / (MTSUN_SI * (m1 + m2))
            max_f = mf_max / (MTSUN_SI * (m1 + m2))
            self.f_arr = np.logspace(np.log10(min_f), np.log10(max_f), 1024)

        else:
            assert isinstance(freqs, np.ndarray)
            self.f_arr = freqs

        # generate waveform with proper settings
        AET = self.wave_gen(
            *params,
            direct=True,
            combine=True,
            freqs=self.f_arr,
            **kwargs,
        )[0]

        # prepare output
        if return_array:
            return AET
        else:
            return (AET[0], AET[1], AET[2])


class PhenomTHMTDIWaveform(TDWaveformBase):
    """
    Generate PhenomTHM waveforms with the TDI LISA Response.

    Args:
        waveform_kwargs: Keyword arguments forwarded to :class:`phentax.waveform.IMRPhenomTHM`.
        t0: Initial time in seconds.
        dt: Time step in seconds.
        Tobs: Observation time in years.
        response_kwargs: Keyword arguments for the TDI response.
        buffer_time: Buffer time in seconds added around the signal edges.
        tukey_alpha: Alpha parameter for the Tukey window.
        force_backend: Backend selector ('cpu' or 'cuda').
        force_uniform_stft: See :class:`TDWaveformBase`.

    """

    def __init__(
        self,
        waveform_kwargs: dict,
        t0: float = 0.0,
        dt: float = 10.0,
        Tobs: float = 1.0,
        response_kwargs: dict = None,
        buffer_time: int = 600,
        tukey_alpha: float = 0.01,
        force_backend: str = "cpu",
        force_uniform_stft: bool = False,
    ) -> None:

        if not phentax_available:
            raise ImportError(
                "PhenomTHM is not available. Please install phentax to use this waveform."
            )

        super().__init__(
            t0=t0,
            dt=dt,
            Tobs=Tobs,
            response_kwargs=response_kwargs,
            buffer_time=buffer_time,
            tukey_alpha=tukey_alpha,
            force_backend=force_backend,
            force_uniform_stft=force_uniform_stft,
        )

        self.waveform = phentax.waveform.IMRPhenomTHM(T=self.Tobs, **waveform_kwargs)

    def wave_gen(
        self,
        m1: float,
        m2: float,
        s1z: float,
        s2z: float,
        distance: float,
        phi_ref: float,
        ref_freq: float,
        start_freq: float,
        inclination: float,
        psi: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the waveform's polarizations for a single source.

        Returns:
            t_arr, h_plus, h_cross

        """

        times, mask, hplus, hcross = self.waveform.compute_polarizations_at_once(
            m1,
            m2,
            s1z,
            s2z,
            distance,
            phi_ref,
            ref_freq,
            start_freq,
            inclination,
            psi,
            delta_t=self.dt,
        )

        return (
            self.xp.asarray(times[mask]),
            self.xp.asarray(hplus[mask]),
            self.xp.asarray(hcross[mask]),
        )

    def wave_gen_batch(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        s1z: np.ndarray,
        s2z: np.ndarray,
        distance: np.ndarray,
        phi_ref: np.ndarray,
        ref_freq: np.ndarray,
        start_freq: np.ndarray,
        inclination: np.ndarray,
        psi: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate polarizations for a batch of sources using phentax's vectorised path.

        phentax uses JAX vmap internally, so all parameters must be broadcastable
        to the batch shape.  The returned arrays still carry the padded time axis;
        per-source masking is handled by the caller (:meth:`_call_batched`).

        Args:
            m1: Source-1 masses in solar masses, shape (Nbatch,).
            m2: Source-2 masses in solar masses, shape (Nbatch,).
            s1z: Dimensionless spin of source 1, shape (Nbatch,).
            s2z: Dimensionless spin of source 2, shape (Nbatch,).
            distance: Luminosity distance in Mpc, shape (Nbatch,).
            phi_ref: Reference phase in radians, shape (Nbatch,).
            ref_freq: Reference frequency in Hz, shape (Nbatch,).
            start_freq: Starting frequency in Hz, shape (Nbatch,).
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            **kwargs: Additional keyword arguments forwarded to
                ``compute_polarizations_at_once`` (e.g. ``T`` for observation time
                override, ``t_min``, ``t_ref``).

        Returns:
            Tuple of (times_batch, mask_batch, h_plus_batch, h_cross_batch),
            each of shape (Nbatch, Ntimes) as plain NumPy arrays.
        """
        times, mask, hplus, hcross = self.waveform.compute_polarizations_at_once(
            m1,
            m2,
            s1z,
            s2z,
            distance,
            phi_ref,
            ref_freq,
            start_freq,
            inclination,
            psi,
            delta_t=self.dt,
            **kwargs,
        )
        # Move to the target backend: zero-copy on GPU via __cuda_array_interface__,
        # host transfer on CPU. _call_batched will slice and re-wrap as needed.
        return (
            self.xp.asarray(times),
            self.xp.asarray(mask),
            self.xp.asarray(hplus),
            self.xp.asarray(hcross),
        )
