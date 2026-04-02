from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
from bbhx.waveformbuild import BBHWaveformFD

# imports
from fastlisaresponse import ResponseWrapper

from ...domains import DomainSettingsBase
from ...utils.constants import *
from ..waveformbase import SNRWaveform, TDWaveformBase, TDTDIOnFlyWaveformBase

try:
    import jax.numpy as jnp
    import phentax

    phentax_available = True
except (ImportError, ModuleNotFoundError):
    phentax_available = False
    jnp = np  # type: ignore

if TYPE_CHECKING:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        import numpy as cp  # type: ignore


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

class PhenomTHMWaveformBase:
    """
    Base class for PhenomTHM waveforms.

        This class is not meant to be used directly, but it contains the common code for both :class:`PhenomTHMTDIWaveform` and :class:`PhenomTHMTDIOnFlyWaveform`. In particular, it contains the common code for handling the waveform generation and the reference and starting frequencies.
    
        Args:
            waveform_kwargs: Keyword arguments forwarded to :class:`phentax.waveform.IMRPhenomTHM`.
            Tobs: Observation time in years.
            start_freq: Starting frequency in Hz for the waveform generation. If `None`, it has to be explicitly provided in the waveform generation calls.
            ref_freq: Reference frequency in Hz for the waveform generation. If `None` and `start_freq` is provided, it will default to `start_freq`. Otherwise, it has to be explicitly provided in the waveform generation calls.
    """
    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float,
        start_freq: float = None,
        ref_freq: float = None,
    ) -> None:

        if not phentax_available:
            raise ImportError(
                "PhenomTHM is not available. Please install phentax to use this waveform."
            )

        self.waveform = phentax.waveform.IMRPhenomTHM(T=Tobs, **waveform_kwargs)

        self.start_freq = start_freq
        self.ref_freq = ref_freq
    
    def trim_and_shift_times(
        self, 
        times: np.ndarray | cp.ndarray, 
        mask: np.ndarray | cp.ndarray,
    ) -> np.ndarray | cp.ndarray:
        """
        Shift and trim the time arrays for each source according to its mask, so that the resulting time arrays have the same shape (Nbatch, max_valid_times). the initial time points can be different across sources.

        Args:
            times (Array): Time arrays for each source, shape (Nbatch, Ntimes).
            mask (Array): Boolean mask indicating valid time samples for each source, shape (Nbatch, Ntimes).

        Returns:
            Shifted time arrays, shape (Nbatch, max_valid_times).
        """
        valid_points = mask.sum(axis=1)  # number of valid time samples for each source
        max_valid_points = int(
            valid_points.max()
        )  # maximum number of valid time samples across sources

        # Trim to max_valid_points: removes time points unused by ALL sources
        times_out = times[:, -max_valid_points:]

        # Per-source count of remaining invalid points in the trimmed array
        n_pad = max_valid_points - valid_points  # (Nbatch,)

        # Position index in trimmed array
        j = self.xp.arange(max_valid_points)  # (max_valid_points,)

        # Identify invalid positions: position j is invalid for source i if j < n_pad[i]
        is_invalid = j[None, :] < n_pad[:, None]  # (Nbatch, max_valid_points)

        # First valid time per source (at index n_pad[i] in trimmed array)
        batch_idx = self.xp.arange(times_out.shape[0])
        first_valid_time = times_out[batch_idx, n_pad]  # (Nbatch,)

        replacement_times = first_valid_time[:, None] - (n_pad[:, None] - j[None, :]) * self.dt

        # Apply only to invalid positions (valid positions keep their original times)
        times_out = self.xp.where(is_invalid, replacement_times, times_out)

        return times_out  # shape (Nbatch, max_valid_points)

class PhenomTHMTDIWaveform(TDWaveformBase, PhenomTHMWaveformBase):
    """
    Generate PhenomTHM waveforms with the TDI LISA Response.

    Args:
        waveform_kwargs: Keyword arguments forwarded to :class:`phentax.waveform.IMRPhenomTHM`.
        Tobs: Observation time in years.
        start_freq: Starting frequency in Hz for the waveform generation. If `None`, it has to be explicitly provided in the waveform generation calls.
        ref_freq: Reference frequency in Hz for the waveform generation. If `None` and `start_freq` is provided, it will default to `start_freq`. Otherwise, it has to be explicitly provided in the waveform generation calls.
        *args: Additional positional arguments forwarded to :class:`TDWaveformBase`.
        **kwargs: Additional keyword arguments forwarded to :class:`TDWaveformBase`.
    """

    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float = 1.0,
        start_freq: float = None,
        ref_freq: float = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        TDWaveformBase.__init__(
            self,
            *args,
            **kwargs,
            Tobs=Tobs,
        )
        PhenomTHMWaveformBase.__init__(
            self,
            waveform_kwargs=waveform_kwargs,
            Tobs=Tobs,
            start_freq=start_freq,
            ref_freq=ref_freq,
        )

        
    def wave_gen(
        self,
        m1: float,
        m2: float,
        s1z: float,
        s2z: float,
        distance: float,
        phi_ref: float,
        inclination: float,
        psi: float,
        ref_freq: float = None,
        start_freq: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the waveform's polarizations for a single source.

        Args:
            m1: Source-1 mass in solar masses.
            m2: Source-2 mass in solar masses.
            s1z: Dimensionless spin of source 1.
            s2z: Dimensionless spin of source 2.
            distance: Luminosity distance in Mpc.
            phi_ref: Reference phase in radians.
            inclination: Inclination angle in radians.
            psi: Polarisation angle in radians.
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.

        Returns:
            t_arr, h_plus, h_cross

        """

        start_freq = start_freq if start_freq is not None else self.start_freq
        ref_freq = ref_freq if ref_freq is not None else self.ref_freq

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

        hcross.block_until_ready()  # ensure all outputs are ready before moving to self.xp

        xp_mask = self.xp.asarray(mask)
        out_times = self.xp.asarray(times)[xp_mask]
        out_hplus = self.xp.asarray(hplus)[xp_mask]
        out_hcross = self.xp.asarray(hcross)[xp_mask]        

        return (
            out_times,
            out_hplus,
            out_hcross,
        )

    def wave_gen_batch(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        s1z: np.ndarray,
        s2z: np.ndarray,
        distance: np.ndarray,
        phi_ref: np.ndarray,
        inclination: np.ndarray,
        psi: np.ndarray,
        ref_freq: float = None,
        start_freq: float = None,
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
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            ref_freq: Reference frequency in Hz, float.
            start_freq: Starting frequency in Hz, float.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_polarizations_at_once`` (e.g. ``T`` for observation time
                override, ``t_min``, ``t_ref``).

        Returns:
            Tuple of (times_batch, h_plus_batch, h_cross_batch),
            each of shape (Nbatch, N_valid_times) as plain NumPy or Cupy arrays.
        """

        ref_freq = ref_freq if ref_freq is not None else self.ref_freq
        start_freq = start_freq if start_freq is not None else self.start_freq

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
        hcross.block_until_ready()  # ensure all outputs are ready before moving to self.xp
        
        times = self.xp.asarray(times).copy()
        mask = self.xp.asarray(mask).copy()
        hplus = self.xp.asarray(hplus).copy()
        hcross = self.xp.asarray(hcross).copy()

        # Move to the target backend: zero-copy on GPU via __cuda_array_interface__,
        # host transfer on CPU. _call_batched will slice and re-wrap as needed.
        times_out = self.trim_and_shift_times(times, mask)
        num_keep = times_out.shape[-1]

        return (
            times_out,
            hplus[:, -num_keep:],
            hcross[:, -num_keep:],
        )


class PhenomTHMTDIOnFlyWaveform(TDTDIOnFlyWaveformBase, PhenomTHMWaveformBase):
    """
    Generate PhenomTHM waveforms with the TDI LISA Response, on the fly.

    This class inherits from :class:`TDTDIOnFlyWaveformBase`, which means that it generates the waveform sparsely and evaluate the tdi delays on splined versions of each mode's amplitude and phase.

    Args:

    """

    def __init__(
        self,
        waveform_kwargs: dict,
        Tobs: float,
        start_freq: float = None,
        ref_freq: float = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        TDTDIOnFlyWaveformBase.__init__(
            self,
            *args,
            **kwargs,
        )

        PhenomTHMWaveformBase.__init__(
            self,
            waveform_kwargs=waveform_kwargs,
            Tobs=Tobs,
            start_freq=start_freq,
            ref_freq=ref_freq,
        )
        
    def get_amp_phase(
        self,
        m1: np.ndarray | cp.ndarray,
        m2: np.ndarray | cp.ndarray,
        s1z: np.ndarray | cp.ndarray,
        s2z: np.ndarray | cp.ndarray,
        distance: np.ndarray | cp.ndarray,
        phi_ref: np.ndarray | cp.ndarray,
        inclination: np.ndarray | cp.ndarray,
        psi: np.ndarray | cp.ndarray,
        ra: np.ndarray | cp.ndarray = None,
        dec: np.ndarray | cp.ndarray = None,
        ref_freq: np.ndarray | cp.ndarray = None,
        start_freq: np.ndarray | cp.ndarray = None,
        **kwargs,
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Get the waveform modes' amplitude and phase, already including the spherical harmonic contributions.

        Args:
            m1: Source-1 masses in solar masses, shape (Nbatch,).
            m2: Source-2 masses in solar masses, shape (Nbatch,).
            s1z: Dimensionless spin of source 1, shape (Nbatch,).
            s2z: Dimensionless spin of source 2, shape (Nbatch,).
            distance: Luminosity distance in Mpc, shape (Nbatch,).
            phi_ref: Reference phase in radians, shape (Nbatch,).
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            ra: Right ascension for the sources in radians, shape (Nbatch,).
            dec: Declination for the sources in radians, shape (Nbatch,).
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_strain_components_amp_phase``.

        Returns:
            times, mode amplitudes and mode phases
        """
        ref_freq = ref_freq if ref_freq is not None else self.ref_freq
        start_freq = start_freq if start_freq is not None else self.start_freq

        times, mask, amplitude, phase = self.waveform.compute_strain_components_amp_phase(
            jnp.asarray(m1),
            jnp.asarray(m2),
            jnp.asarray(s1z),
            jnp.asarray(s2z),
            jnp.asarray(distance),
            jnp.asarray(phi_ref),
            jnp.asarray(ref_freq),
            jnp.asarray(start_freq),
            jnp.asarray(inclination),
            jnp.asarray(psi),
            delta_t=jnp.asarray(self.dt),
        )
        amplitude.block_until_ready()  # ensure all outputs are ready before moving to self.xp

        times = self.xp.asarray(times).copy()
        mask = self.xp.asarray(mask).copy()
        amplitude = self.xp.asarray(amplitude).copy()
        phase = self.xp.asarray(phase).copy()

        times_out = self.trim_and_shift_times(times, mask)

        num_keep = times_out.shape[-1]

        return times_out, amplitude[..., -num_keep:], phase[..., -num_keep:]

    def process_amp_phase(
        self, amp: np.ndarray | cp.ndarray, phase: np.ndarray | cp.ndarray
    ) -> Tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """
        Process the waveform amplitude and phase to align to the tdi on fly conventions.
        For this specific waveform, we need :math:`A \\to A/2` and :math:`\\phi \\to \\pi-\\phi`.

        Args:
            amp (Array): Waveform modes amplitude, already including the spherical harmonic contributions. Shape (num_binaries, num_modes, num_times).
            phase (Array): Waveform modes phase, already including the spherical harmonic contributions. Shape (num_binaries, num_modes, num_times).

        Returns:
            Tuple of (processed_amplitude, processed_phase), each of shape (num_binaries, num_modes, num_times).
        """

        return (amp / 2.0, self.xp.pi - phase)

    def compute_tdi_channels(
        self,
        m1: np.ndarray | cp.ndarray,
        m2: np.ndarray | cp.ndarray,
        s1z: np.ndarray | cp.ndarray,
        s2z: np.ndarray | cp.ndarray,
        distance: np.ndarray | cp.ndarray,
        phi_ref: np.ndarray | cp.ndarray,
        inclination: np.ndarray | cp.ndarray,
        psi: np.ndarray | cp.ndarray,
        ra: np.ndarray | cp.ndarray,
        dec: np.ndarray | cp.ndarray,
        merger_time: np.ndarray | cp.ndarray,
        ref_freq: np.ndarray | cp.ndarray = None,
        start_freq: np.ndarray | cp.ndarray = None,
        **kwargs,
    ):
        """
        Call parent's :meth:`compute_tdi_channels` method with the correct signature for sampling.

        Args:
            m1: Source-1 masses in solar masses, shape (Nbatch,).
            m2: Source-2 masses in solar masses, shape (Nbatch,).
            s1z: Dimensionless spin of source 1, shape (Nbatch,).
            s2z: Dimensionless spin of source 2, shape (Nbatch,).
            distance: Luminosity distance in Mpc, shape (Nbatch,).
            phi_ref: Reference phase in radians, shape (Nbatch,).
            inclination: Inclination angle in radians, shape (Nbatch,).
            psi: Polarisation angle in radians, shape (Nbatch,).
            ra: Right ascension for the sources in radians, shape (Nbatch,).
            dec: Declination for the sources in radians, shape (Nbatch,).
            merger_time: Merger time with respect to `self.waveform_t0` in seconds, shape (Nbatch,).
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_strain_components_amp_phase``.
        """

        return super().compute_tdi_channels(
                        m1,
                        m2,
                        s1z,
                        s2z,
                        distance,
                        phi_ref,
                        inclination=inclination,
                        psi=psi,
                        ra=ra,
                        dec=dec,
                        merger_time=merger_time,
                        ref_freq=ref_freq,
                        start_freq=start_freq,
                    )
