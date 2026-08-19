"""Binary black hole waveform classes (frequency-domain SNR waveforms and PhenomTHM TDI waveforms)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
from bbhx.waveformbuild import BBHWaveformFD

# stft_tof merge: fastlisaresponse is retired -- ResponseWrapper lives in
# lisatools.response.directresponse since Phase 3B.
from lisatools.response.directresponse import ResponseWrapper

from ...domains import DomainSettingsBase
from ...utils.constants import *
from ...jax.jaxbase import JaxBase
from ..waveformbase import SNRWaveform, TDPyResponseWaveformBase, TDTDIOnFlyWaveformBase

try:
    import jax
    import jax.numpy as jnp
    import phentax

    phentax_available = True
except (ImportError, ModuleNotFoundError):
    phentax_available = False
    jnp = np  # type: ignore

if TYPE_CHECKING:
    try:
        import cupy as cp
        from jax import Device
    except (ImportError, ModuleNotFoundError):
        import numpy as cp  # type: ignore
        Device = Any  # type: ignore
        from ...utils.typing import NDArrayLike, ArrayModule


# ============================================================
# Stock MBH ``PhenomTHMTDIWaveform`` defaults (global-fit aligned)
# ============================================================
# The sources-side single source of truth for the LEGACY phentax MBH template
# path. These mirror ``SourceMBHSettings`` and ``get_mbh_phenom_wave_gen`` in
# ``lisatools.globalfit.stock.erebor`` — a bare ``PhenomTHMTDIWaveform`` builds
# the same generator configuration the stock ``all_sources`` fit uses for its
# default MBH branch. Kept equal to the erebor values (enforced by
# tests/test_stock_waveform_alignment.py).
#
# ``coarse_grain=False`` is mandatory: pyResponseTDI needs equispaced time
# arrays. ``include_negative_modes`` adds the negative-m modes by symmetry;
# ``t_low_fit`` seeds the ``t(f)`` root finder from the fit.
MBH_PHENOM_DEFAULT_WAVEFORM_KWARGS = dict(
    higher_modes=[21, 33, 44],
    include_negative_modes=True,
    t_low_fit=True,
    coarse_grain=False,
    atol=1e-12,
    rtol=1e-12,
)
MBH_PHENOM_DEFAULT_TOBS = YRSID_SI / 12.0  # phentax generation window (seconds)
MBH_PHENOM_DEFAULT_START_FREQ = 7e-5
MBH_PHENOM_DEFAULT_RESPONSE_ORDER = 30
MBH_PHENOM_DEFAULT_BUFFER_TIME = 15_000.0
MBH_PHENOM_DEFAULT_FREQ_MIN = 1e-4
MBH_PHENOM_DEFAULT_FREQ_MAX = 2.5e-2
MBH_PHENOM_DEFAULT_FFT_BATCH_SIZE = 2
MBH_PHENOM_DEFAULT_TDI_GENERATION = "2nd generation"
MBH_PHENOM_DEFAULT_TDI_CHANNELS = "XYZ"


class BBHSNRWaveform(SNRWaveform):
    """Wrapper class for straightforward BBH SNR calculations.

    Calculates A/E/T channels in **TDI 2nd generation** (``tdi2=True``, matching
    the stock ``all_sources`` MBH TDI generation). This is a quick bbhx
    frequency-domain SNR helper — it is *not* the global-fit MBH residual
    template (that is the legacy phentax :class:`PhenomTHMTDIWaveform`) and it
    reports the ``AET`` SNR channels rather than the ``XYZ`` residual channels.

    Args:
        bbh_waveform_kwargs: ``amp_phase_kwargs`` for :class:`BBHWaveformFD`
            (``None`` -> ``{"run_phenomd": False}``).
        response_kwargs: ``response_kwargs`` for :class:`BBHWaveformFD`
            (``None`` -> ``{"TDItag": "AET", "tdi2": True}``).
        force_backend: Backend selection at construction (``"cpu"`` /
            ``"cuda12x"`` / ...); forwarded to :class:`BBHWaveformFD`.

    """

    def __init__(
        self,
        bbh_waveform_kwargs: Optional[dict] = None,
        response_kwargs: Optional[dict] = None,
        force_backend: str = "cpu",
    ) -> None:

        # Resolve mutable defaults inside __init__ (never share a dict across
        # instances / mutate a default argument).
        bbh_waveform_kwargs = (
            dict(run_phenomd=False)
            if bbh_waveform_kwargs is None
            else dict(bbh_waveform_kwargs)
        )
        response_kwargs = (
            {} if response_kwargs is None else dict(response_kwargs)
        )
        response_kwargs.setdefault("TDItag", "AET")
        response_kwargs.setdefault("tdi2", True)

        # wave generating class
        self.wave_gen = BBHWaveformFD(
            amp_phase_kwargs=bbh_waveform_kwargs,
            response_kwargs=response_kwargs,
            force_backend=force_backend,
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

class PhenomTHMWaveformBase(JaxBase):
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
        use_reference_time: bool = True,
        use_coalescence_time: bool = True,
        time_bounded_start: bool = True,
    ) -> None:

        JaxBase.__init__(self)

        if not phentax_available:
            raise ImportError(
                "PhenomTHM is not available. Please install phentax to use this waveform."
            )

        self.waveform = phentax.waveform.IMRPhenomTHM(T=Tobs, **waveform_kwargs)

        self.start_freq = jnp.asarray(start_freq) if start_freq is not None else None
        self.ref_freq = jnp.asarray(ref_freq) if ref_freq is not None else None
        self.use_reference_time = use_reference_time
        self.ref_time = jnp.asarray(0.0)  if use_coalescence_time else None
        self.time_bounded_start = time_bounded_start

    @property
    def phenom_kwargs(self) -> dict:
        """Keyword arguments for the waveform generationDictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""

        waveform_kwargs = dict(
            higher_modes=self.waveform.higher_modes,
            include_negative_modes=self.waveform.include_negative_modes,
            coarse_grain=self.waveform.coarse_grain,
            t_low_fit=self.waveform.t_low == 0.0,
            atol=self.waveform.atol,
            rtol=self.waveform.rtol,
        )

        return {
            "waveform_kwargs": waveform_kwargs,
            "Tobs": self.waveform.T,
            "start_freq": self.start_freq,
            "ref_freq": self.ref_freq,
            "use_reference_time": self.use_reference_time,
            "use_coalescence_time": self.ref_time is not None,
            "time_bounded_start": self.time_bounded_start,
            }
        
    
    @staticmethod
    def trim_and_shift_times(
        times: np.ndarray | cp.ndarray,
        mask: np.ndarray | cp.ndarray,
        *,
        xp: ArrayModule,
        dt: float,
    ) -> np.ndarray | cp.ndarray:
        """
        Shift and trim the time arrays for each source according to its mask, so that the resulting time arrays have the same shape (Nbatch, max_valid_times). the initial time points can be different across sources.

        Args:
            times (Array): Time arrays for each source, shape (Nbatch, Ntimes).
            mask (Array): Boolean mask indicating valid time samples for each source, shape (Nbatch, Ntimes).
            xp: Array module (numpy or cupy).
            dt: Time step in seconds.

        Returns:
            Shifted time arrays, shape (Nbatch, max_valid_times).
        """
        valid_points = mask.sum(axis=1)
        max_valid_points = int(valid_points.max())

        times_out = times[:, -max_valid_points:]

        n_pad = max_valid_points - valid_points  # (Nbatch,)

        j = xp.arange(max_valid_points)  # (max_valid_points,)

        is_invalid = j[None, :] < n_pad[:, None]  # (Nbatch, max_valid_points)

        batch_idx = xp.arange(times_out.shape[0])
        first_valid_time = times_out[batch_idx, n_pad]  # (Nbatch,)

        replacement_times = first_valid_time[:, None] - (n_pad[:, None] - j[None, :]) * dt

        times_out = xp.where(is_invalid, replacement_times, times_out)

        return times_out  # shape (Nbatch, max_valid_points)
    
    @staticmethod
    def _leading_onset_ramp(num_points: int, num_pad: int, taper_length: int, xp: ArrayModule) -> NDArrayLike:
        """
        Get a leading onset ramp to smoothly turn on the waveform.
        This avoids the sharp jump from zero to the first valid point.

        Args:
            num_points: Total number of points in the output array.
            num_pad: Number of initial points that are invalid.
            taper_length: Length of the taper in number of points.
            xp: Array module (numpy or cupy).

        Returns:
            Ramp array of shape (num_points,).
        """
        rj = xp.arange(num_points)
        offset = rj[None, :] - xp.asarray(num_pad, dtype=int)[:, None]
        x = xp.clip(offset / float(taper_length), 0.0, 1.0)
        return 0.5 * (1.0 - xp.cos(xp.pi * x))

    def get_reference_quantities(self,
                                merger_time: float | np.ndarray | cp.ndarray, 
                                start_freq: float | np.ndarray | cp.ndarray = None,
                                ref_freq: float | np.ndarray | cp.ndarray = None
                                ) -> dict:
        """
        Get the reference quantities for the waveform generation, depending on the settings of the class.

        Args:
            merger_time: Merger time in seconds, shape (Nbatch,).
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.

        Returns:
            Dictionary of reference quantities for the waveform generation.
        """

        start_freq = self._to_jax(start_freq) if start_freq is not None else self.start_freq
        ref_freq = self._to_jax(ref_freq) if ref_freq is not None else self.ref_freq

        # Waveform start: phentax derives t_min from f_min ONLY when t_min is
        # NaN ("f_min ... Used if t_min is NaN to set the minimum time for
        # waveform generation").  Leaving t_min unset therefore starts the
        # template at the moment the (2,2) mode sweeps through ``start_freq``,
        # which for a high-mass MBH is only a few days before merger -- the
        # template is then identically zero over most of the analysis window
        # and the (3,3)/(4,4) modes are missing at the low-frequency band edge
        # (measured on mojito id19: legacy zero over 94.6% of a 48 d window;
        # dropping start_freq 7e-5 -> 2e-5 improved the in-band mismatch 6.6x).
        # ``time_bounded_start`` instead bounds the template in TIME at
        # ``t_min = -Tobs``, matching ``bbhx.mbhtdionfly.MBHTDIonFly`` (which
        # passes ``t_min=-waveform_duration, t_ref=-t_merge``), so the template
        # spans its full stated duration regardless of the source's mass.
        # ``start_freq`` is still passed and remains the fallback.
        extra = ({'t_min': self._to_jax(-float(self.waveform.T))}
                 if self.time_bounded_start else {})

        if self.use_reference_time:
            ref_time = self._to_jax(-merger_time) if self.ref_time is None else self.ref_time
            return {'t_ref': ref_time, 'f_min': start_freq, **extra}
        else:
            return {'f_min': start_freq, 'f_ref': ref_freq, **extra}

class PhenomTHMTDIWaveform(TDPyResponseWaveformBase, PhenomTHMWaveformBase):
    """
    Generate PhenomTHM waveforms with the TDI LISA Response (global-fit aligned).

    A bare instance reproduces the generator configuration the stock
    ``all_sources`` global fit uses for its default (legacy) MBH branch — see
    ``lisatools.globalfit.stock.erebor.wrappers.get_mbh_phenom_wave_gen`` and
    ``source_runtime.SourceMBHSettings``. Every waveform-config default below
    is drawn from the module-level ``MBH_PHENOM_DEFAULT_*`` constants (the
    sources-side single source of truth); only the *run-specific* arguments
    (``waveform_t0``, ``data_td_settings``, ``orbits``, ``sampling_frequency``,
    ``tukey_alpha``, ``output_domain_settings``, ``force_backend``) must be
    supplied per run.

    Args:
        waveform_kwargs: Keyword arguments forwarded to
            :class:`phentax.waveform.IMRPhenomTHM`. ``None`` (default) uses a
            copy of :data:`MBH_PHENOM_DEFAULT_WAVEFORM_KWARGS` (higher modes
            ``(21, 33, 44)``, negative modes on, ``t_low_fit=True``,
            ``coarse_grain=False``, ``atol=rtol=1e-12``).
        Tobs: phentax generation window in seconds (its ``T``); default
            :data:`MBH_PHENOM_DEFAULT_TOBS`.
        start_freq: Starting frequency in Hz; default
            :data:`MBH_PHENOM_DEFAULT_START_FREQ`.
        ref_freq: Reference frequency in Hz (``None`` -> ``start_freq``).
        order: pyResponseTDI Lagrange order; default
            :data:`MBH_PHENOM_DEFAULT_RESPONSE_ORDER`.
        buffer_time: TDI response buffer in seconds; default
            :data:`MBH_PHENOM_DEFAULT_BUFFER_TIME`.
        freq_min / freq_max: FD/STFT output frequency bounds; defaults
            :data:`MBH_PHENOM_DEFAULT_FREQ_MIN` / :data:`MBH_PHENOM_DEFAULT_FREQ_MAX`.
        fft_batch_size: FD transform batch size; default
            :data:`MBH_PHENOM_DEFAULT_FFT_BATCH_SIZE`.
        tdi_generation / tdi_channels: TDI configuration; defaults
            :data:`MBH_PHENOM_DEFAULT_TDI_GENERATION` (2nd gen) /
            :data:`MBH_PHENOM_DEFAULT_TDI_CHANNELS` (``XYZ``).
        *args: Additional positional arguments forwarded to
            :class:`TDPyResponseWaveformBase` (its first positional is
            ``waveform_t0``).
        **kwargs: Run-specific keyword arguments forwarded to
            :class:`TDPyResponseWaveformBase` (``waveform_t0``,
            ``data_td_settings``, ``orbits``, ``sampling_frequency``,
            ``tukey_alpha``, ``stft_dt``, ``output_domain_settings``,
            ``force_backend``).
    """

    def __init__(
        self,
        waveform_kwargs: Optional[dict] = None,
        Tobs: float = MBH_PHENOM_DEFAULT_TOBS,
        start_freq: float = MBH_PHENOM_DEFAULT_START_FREQ,
        ref_freq: float = None,
        use_reference_time: bool = True,
        use_coalescence_time: bool = False,
        time_bounded_start: bool = True,
        *args: Any,
        order: int = MBH_PHENOM_DEFAULT_RESPONSE_ORDER,
        buffer_time: int = MBH_PHENOM_DEFAULT_BUFFER_TIME,
        freq_min: float = MBH_PHENOM_DEFAULT_FREQ_MIN,
        freq_max: float = MBH_PHENOM_DEFAULT_FREQ_MAX,
        fft_batch_size: int = MBH_PHENOM_DEFAULT_FFT_BATCH_SIZE,
        tdi_generation: str = MBH_PHENOM_DEFAULT_TDI_GENERATION,
        tdi_channels: str = MBH_PHENOM_DEFAULT_TDI_CHANNELS,
        **kwargs: Any,
    ) -> None:

        if waveform_kwargs is None:
            # Copy so the shared module-level default is never mutated.
            waveform_kwargs = dict(MBH_PHENOM_DEFAULT_WAVEFORM_KWARGS)

        if "coarse_grain" in waveform_kwargs and waveform_kwargs["coarse_grain"] is True:
            raise ValueError("Applying the response through the `PyresponseTDI` class requires equispaced time arrays")

        TDPyResponseWaveformBase.__init__(
            self,
            *args,
            order=order,
            buffer_time=buffer_time,
            freq_min=freq_min,
            freq_max=freq_max,
            fft_batch_size=fft_batch_size,
            tdi_generation=tdi_generation,
            tdi_channels=tdi_channels,
            **kwargs,
        )

        PhenomTHMWaveformBase.__init__(
            self,
            waveform_kwargs=waveform_kwargs,
            Tobs=Tobs,
            start_freq=start_freq,
            ref_freq=ref_freq,
            use_reference_time=use_reference_time,
            use_coalescence_time=use_coalescence_time,
            time_bounded_start=time_bounded_start,
        )

    @property
    def kwargs(self) -> dict:
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""

        phenom_kwargs = self.phenom_kwargs  # get the PhenomTHM-specific kwargs
        wrapper_kwargs = self.wrapper_kwargs # get the TD TDI wrapper-specific kwargs

        return {
            **phenom_kwargs,
            **wrapper_kwargs,
        }
        
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
        ra: float = None,
        dec: float = None,
        merger_time: float = None,
        ref_freq: float = None,
        start_freq: float = None,
        synchronize: bool = False,
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
            ra: Right ascension for the source in radians.
            dec: Declination for the source in radians.
            merger_time: Time of merger in seconds.
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            synchronize: If `True`, it will call `block_until_ready()` on the waveform outputs. :meth:`self._from_jax` relies on `dlpack` to move data from JAX to Cupy, that should already handle CUDA streams correctly.

        Returns:
            t_arr, h_plus, h_cross

        """

        reference_kwargs = self.get_reference_quantities(merger_time=merger_time, start_freq=start_freq, ref_freq=ref_freq)

        times, mask, hplus, hcross = self.waveform.compute_polarizations_at_once(
            self._to_jax(m1),
            self._to_jax(m2),
            self._to_jax(s1z),
            self._to_jax(s2z),
            self._to_jax(distance),
            self._to_jax(phi_ref),
            self._to_jax(inclination),
            self._to_jax(psi),
            delta_t=self.dt,
            **reference_kwargs,
        )

        xp_mask = self._from_jax(mask, do_synchronize=synchronize)
        out_times = self._from_jax(times, do_synchronize=synchronize)[xp_mask]
        out_hplus = self._from_jax(hplus, do_synchronize=synchronize)[xp_mask]
        out_hcross = self._from_jax(hcross, do_synchronize=synchronize)[xp_mask]        

        return (
            out_times,
            out_hplus,
            out_hcross,
        )

    def wave_gen_batch(
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
        merger_time: np.ndarray | cp.ndarray = None,
        ref_freq: float | np.ndarray | cp.ndarray = None,
        start_freq: float | np.ndarray | cp.ndarray = None,
        synchronize: bool = False,
        onset_ramp: bool = True,
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
            ra: Right ascension in radians, shape (Nbatch,).
            dec: Declination in radians, shape (Nbatch,).
            merger_time: Time of merger in seconds, shape (Nbatch,).
            synchronize: If `True`, it will call `block_until_ready()` on the waveform outputs. :meth:`self._from_jax` relies on `dlpack` to move data from JAX to Cupy, that should already handle CUDA streams correctly.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_polarizations_at_once`` (e.g. ``T`` for observation time
                override, ``t_min``, ``t_ref``).

        Returns:
            Tuple of (times_batch, h_plus_batch, h_cross_batch),
            each of shape (Nbatch, N_valid_times) as plain NumPy or Cupy arrays.
        """

        reference_kwargs = self.get_reference_quantities(merger_time=merger_time, start_freq=start_freq, ref_freq=ref_freq)

        times, mask, hplus, hcross = self.waveform.compute_polarizations_at_once(
            self._to_jax(m1),
            self._to_jax(m2),
            self._to_jax(s1z),
            self._to_jax(s2z),
            self._to_jax(distance),
            self._to_jax(phi_ref),
            self._to_jax(inclination),
            self._to_jax(psi),
            delta_t=self.dt,
            **reference_kwargs,
            **kwargs,
        )
        
        times = self._from_jax(times, do_synchronize=synchronize)
        mask = self._from_jax(mask, do_synchronize=synchronize)
        hplus = self._from_jax(hplus, do_synchronize=synchronize)
        hcross = self._from_jax(hcross, do_synchronize=synchronize)

        times_out = self.trim_and_shift_times(times, mask, xp=self.xp, dt=self.dt)
        num_keep = times_out.shape[-1]

        num_pad = num_keep - mask.sum(axis=1).astype(int)  # (Nbatch,)

        if not onset_ramp:
            # Bit-identical-to-``wave_gen`` mode. The batch is rectangular, so
            # a source with fewer valid samples than the longest one carries
            # ``num_pad`` leading samples from OUTSIDE its own mask -- the ramp
            # is what suppresses those, so dropping it is only sound when no
            # source needs padding. Refuse rather than silently splice
            # unmasked waveform into the template.
            if int(self.xp.max(num_pad)) > 0:
                raise ValueError(
                    "onset_ramp=False requires every source in the batch to "
                    "produce the same number of valid samples (num_pad == 0 "
                    f"for all); got max num_pad = {int(self.xp.max(num_pad))}. "
                    "Sources of differing length need the onset ramp to mask "
                    "their leading pad -- either batch equal-length sources, "
                    "or leave onset_ramp=True and rebuild any injection with "
                    "the same setting so both sides of the likelihood match."
                )
            return (times_out, hplus[:, -num_keep:], hcross[:, -num_keep:])

        taper_length = int(self.tdi_buffer_time * 5 / self.dt)
        ramp = self._leading_onset_ramp(num_points=num_keep, num_pad=num_pad, taper_length=taper_length, xp=self.xp)  # (Nbatch, num_keep)

        return (
            times_out,
            hplus[:, -num_keep:] * ramp,  # apply ramp to h_plus to avoid sharp jump at the beginning of the waveform, which can cause issues with the TDI response
            hcross[:, -num_keep:] * ramp,  # apply ramp to h_cross to avoid sharp jump at the beginning of the waveform, which can cause issues with the TDI response
        )




class GridAlignedPhenomTHMTDIWaveform(PhenomTHMTDIWaveform):
    """:class:`PhenomTHMTDIWaveform` that evaluates on the DATA lattice.

    A drop-in replacement whose only difference from the stock class is the
    time grid it evaluates on. That difference is what makes a batch of
    independent parameter sets launchable at all.

    WHY THIS EXISTS
    ---------------
    ``pyResponseTDI`` shares ONE relative evaluation grid across a batch, so
    ``t0_shift_to_data`` -- the sub-sample offset between a source's own grid
    and the data grid -- must be identical for every row, and it refuses a
    batch whose offsets differ by more than 1e-12 s.

    Both parameters an MCMC walker actually moves break that:

    * ``t_merger`` is added straight onto the evaluation grid.
    * ``mT`` does too, less obviously: phentax builds its time grid BACKWARDS
      from ``tmax`` in geometric units, so the anchor moves by
      ``500 * MTSUN_SI`` ~ 2.5e-3 s per solar mass.

    So a walker batch was rejected outright. Evaluating every source on the
    data lattice makes each offset EXACTLY zero -- not merely inside the
    tolerance -- and the batch launches.

    The merger time is split into a lattice part and a sub-sample part. The
    grid carries the lattice part; the waveform is evaluated at
    ``t_arr - m_frac`` so the merger still lands at the requested time. The
    sub-sample part is spent inside the waveform rather than against the data
    grid, which is precisely what the response cannot absorb per-source.

    Set :attr:`grid_align` to False for stock behaviour in-process (that is
    how the A/B comparison is taken); the class is otherwise interchangeable.
    """

    #: Per-instance escape hatch; see the class docstring.
    grid_align: bool = True

    @property
    def supports_batch(self) -> bool:
        """True only while alignment is actually ON.

        ONE decision in ONE place. A class-level ``supports_batch = True``
        beside a separate ``grid_align`` flag lets the two disagree: with
        ``grid_align = False`` the generator would still advertise batching,
        the container would still try, and ``pyResponseTDI`` would refuse --
        a guaranteed failed launch per call, reported as a fallback warning.
        """
        return bool(self.grid_align)

    # -- preconditions -----------------------------------------------------
    def _check_alignable(self) -> None:
        """Refuse to claim an alignment we cannot actually deliver.

        Both of these are silent-wrongness risks rather than crashes, so they
        are checked rather than assumed.
        """
        if getattr(self.waveform, "coarse_grain", False):
            raise ValueError(
                "grid-aligned generation requires coarse_grain=False: the "
                "coarse-grained phentax grid is non-uniform, so 'the next "
                "sample is dt later' -- which the lattice construction "
                "assumes -- does not hold. The legacy response path already "
                "forces it off; set coarse_grain=False or use the stock "
                "PhenomTHMTDIWaveform."
            )
        dt = float(self.dt)
        for name in ("waveform_t0", "data_t0"):
            value = float(getattr(self, name))
            residual = value - np.rint(value / dt) * dt
            if abs(residual) > 1e-9:
                raise ValueError(
                    f"grid-aligned generation requires {name} to sit on the "
                    f"dt lattice; got {name} = {value!r} with dt = {dt!r}, "
                    f"residual {residual:.6e} s. The alignment is exact only "
                    f"because waveform_t0 and data_t0 cancel exactly; with a "
                    f"non-lattice value the per-source spread reappears at "
                    f"O(ulp(1e7)) ~ 2e-9 s, which EXCEEDS the 1e-12 tolerance "
                    f"in directresponse.py and would re-break the batch with "
                    f"a message pointing at the waveform rather than here."
                )

    # -- the grid ----------------------------------------------------------
    def _split_merger_time(self, merger_time):
        """``merger_time -> (m_int*dt, m_frac)``, ``m_frac`` in ``(-dt/2, dt/2]``."""
        mt = np.atleast_1d(np.asarray(merger_time, dtype=np.float64))
        m_grid = np.rint(mt / self.dt) * self.dt
        return m_grid, mt - m_grid

    def _aligned_polarizations(
        self, m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
        merger_time, start_freq=None, ref_freq=None, T=None,
        onset_ramp=True, synchronize=False,
    ):
        """Batched polarizations on a grid of exact multiples of ``dt``.

        Returns ``(times, h_plus, h_cross, merger_time_on_grid)``. The fourth
        item is what must reach :meth:`_apply_response` in place of the
        requested ``merger_time``: the sub-sample part has already been spent
        inside the waveform.
        """
        self._check_alignable()
        xp = self.xp
        dt = float(self.dt)
        wf = self.waveform

        ref_kw = self.get_reference_quantities(
            merger_time=merger_time, start_freq=start_freq, ref_freq=ref_freq)

        args = [self._to_jax(np.atleast_1d(np.asarray(v, dtype=np.float64)))
                for v in (m1, m2, s1z, s2z, distance, phi_ref,
                          inclination, psi)]

        # 1. Everything phentax needs, on ITS grid. This is the public entry
        #    point and exactly what compute_polarizations_at_once calls first.
        #
        #    ``**ref_kw`` IS FORWARDED WHOLE, BY KEYWORD, exactly as the stock
        #    ``wave_gen_batch`` does. Unpacking only the keys this method
        #    happens to name and filling ``initial_processing``'s positionals
        #    by hand silently dropped ``t_min``.
        #
        #    ``get_reference_quantities`` adds ``t_min = -T`` whenever
        #    ``time_bounded_start`` is set -- which is the DEFAULT -- and
        #    phentax derives the start from ``f_min`` only when ``t_min`` is
        #    NaN. Hardcoding NaN therefore un-bounded the template in time and
        #    shortened it badly at high total mass: measured 57,789 valid
        #    samples against the stock 525,970 at m1 = 1e7, m2 = 8e6 Msun, i.e.
        #    11% of the analysis window, for a reason that has nothing to do
        #    with grid alignment. A walker proposing high mT would have taken
        #    a likelihood hit attributable to this alone.
        wf_params, times_mass, mask, amp22, ph22 = wf.initial_processing(
            *args,
            delta_t=dt,
            T=T if T is not None else wf.T,
            **ref_kw,
        )

        M_sec = np.asarray(wf_params.total_mass) * MTSUN_SI          # (B,)
        n_times = int(times_mass.shape[-1])

        # 2. The lattice we want. ``times_mass`` is in geometric units and
        #    anchored at Mt_end, so its LAST sample sits t_last_sec after the
        #    peak -- that anchor, not t_arr[0], is what sets the alignment.
        t_last_sec = np.asarray(times_mass[:, -1]) * M_sec           # (B,)
        m_grid, m_frac = self._split_merger_time(merger_time)
        if m_grid.size == 1 and M_sec.size > 1:
            m_grid = np.repeat(m_grid, M_sec.size)
            m_frac = np.repeat(m_frac, M_sec.size)

        # t_arr[j] is an integer multiple of dt, so t_arr + m_grid +
        # waveform_t0 lands on the data lattice; the waveform is EVALUATED at
        # t_arr[j] - m_frac so the merger still sits at the requested time.
        # Built from integers rather than by adding a float offset, so
        # "integer multiple of dt" is exact and not merely close.
        n_last = np.rint((t_last_sec + m_frac) / dt)                 # (B,)
        j = np.arange(n_times, dtype=np.float64)
        n_grid = n_last[:, None] - (n_times - 1.0 - j[None, :])
        t_arr_sec = n_grid * dt
        eval_sec = t_arr_sec - m_frac[:, None]

        # 3. Exact re-evaluation of the model at those times.
        times_new = self._to_jax(eval_sec / M_sec[:, None])
        strain = jax.vmap(wf._compute_strain_single)(
            times_new, mask, wf_params, amp22, ph22,
            wf_params.inclination, wf_params.phi_ref,
        )
        h_plus = jnp.real(strain)
        h_cross = -jnp.imag(strain)
        h_plus, h_cross = wf.rotate_by_polarization_angle(
            h_plus, h_cross, wf_params.psi)

        h_plus = self._from_jax(h_plus, do_synchronize=synchronize)
        h_cross = self._from_jax(h_cross, do_synchronize=synchronize)
        mask_x = self._from_jax(mask, do_synchronize=synchronize)
        times_x = xp.asarray(t_arr_sec)

        # 4. The stock trim / onset-ramp tail, verbatim, so the two paths
        #    differ ONLY in the grid.
        times_out = self.trim_and_shift_times(times_x, mask_x, xp=xp, dt=dt)
        num_keep = times_out.shape[-1]
        num_pad = num_keep - mask_x.sum(axis=1).astype(int)

        if not onset_ramp:
            if int(xp.max(num_pad)) > 0:
                raise ValueError(
                    "onset_ramp=False requires every source in the batch to "
                    "produce the same number of valid samples (num_pad == 0 "
                    f"for all); got max num_pad = {int(xp.max(num_pad))}. "
                    "This is NOT a grid-alignment failure -- it is the same "
                    "equal-length requirement the stock batched path has, and "
                    "it depends on the f_min-determined inspiral length "
                    "versus the window."
                )
            return times_out, h_plus[:, -num_keep:], h_cross[:, -num_keep:], m_grid

        taper_length = int(self.tdi_buffer_time * 5 / dt)
        ramp = self._leading_onset_ramp(
            num_points=num_keep, num_pad=num_pad,
            taper_length=taper_length, xp=xp)
        return (times_out, h_plus[:, -num_keep:] * ramp,
                h_cross[:, -num_keep:] * ramp, m_grid)

    # -- dispatch ----------------------------------------------------------
    # These exist so the SPLIT merger time reaches ``_apply_response``.
    # Nothing in waveformbase changes.
    def _call_batched(self, *args, ra, dec, merger_time, **kwargs):
        if not self.grid_align:
            return super()._call_batched(
                *args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        kwargs.pop("ref_freq", None)
        t, hp, hc, m_grid = self._aligned_polarizations(
            *args, merger_time=merger_time, **kwargs)
        return self._apply_response(t, hp, hc, ra, dec, m_grid)

    def _call_single(self, *args, ra, dec, merger_time, **kwargs):
        if not self.grid_align:
            return super()._call_single(
                *args, ra=ra, dec=dec, merger_time=merger_time, **kwargs)
        kwargs.pop("ref_freq", None)
        # B == 1 => num_pad == 0 by construction, so this reproduces the stock
        # mask-and-drop path rather than the ramp.
        kwargs.setdefault("onset_ramp", False)
        t, hp, hc, m_grid = self._aligned_polarizations(
            *args, merger_time=merger_time, **kwargs)
        return self._apply_response(
            t[0], hp[0], hc[0], float(ra), float(dec), float(m_grid[0]))


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
        use_reference_time: bool = True,
        use_coalescence_time: bool = False,
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
            use_reference_time=use_reference_time,
            use_coalescence_time=use_coalescence_time,
        )

    @property
    def kwargs(self) -> dict:
        """Dictionary of waveform settings used to initialize the waveform, for reproducibility and debugging."""

        phenom_kwargs = self.phenom_kwargs  # get the PhenomTHM-specific kwargs
        wrapper_kwargs = self.wrapper_kwargs # get the TD TDI wrapper-specific kwargs

        return {
            **phenom_kwargs,
            **wrapper_kwargs,
        }

    def get_evaluation_times(
        self,
        input_times: NDArrayLike,
    ) -> NDArrayLike:
        """Return every adaptive Phentax node above the leading TDI buffer.

        The generic on-the-fly wrapper retains only its final ``max_length``
        nodes.  Long MBHB waveforms can contain more than 2,000 post-merger
        nodes, so that policy can discard the merger and the complete
        inspiral.  Phentax's coarse-grained grid is already the intended
        sparse response grid; retain it in full above the leading buffer.

        No TRAILING buffer is removed.  Its purpose is to keep the retarded
        reads the assembly makes above each evaluation time inside the
        amplitude/phase spline, and :meth:`pad` already provides exactly that
        -- it extends the spline ``ceil(tdi_buffer_time / right_dt)`` nodes
        past the last one, i.e. at least ``tdi_buffer_time`` = 600 s, while
        the largest such read is |k.x|/c <= 499 s.  Trimming here as well only
        discarded live ringdown: ``right_dt`` is the FINE post-merger spacing
        (0.39 s on mojito MBHB 0), so ``end_buffer`` ran to 1521 nodes and cut
        the grid 600 s below the last node, at merger+798.9 s where the
        ringdown is still 6.5e-5 of peak.  That truncation alone cost a
        full-band mismatch of 1.65e-05.  BBHx's MBHTDIonFly keeps its
        equivalent trim to a fixed 400 samples (~158 s at the same spacing)
        and hands the spline a zero tail instead.
        """

        delta_t = self.xp.diff(input_times, axis=-1)
        start_buffer, _, _, _ = self.get_tdi_buffers(delta_t)
        return input_times[:, start_buffer:]


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
        merger_time: np.ndarray | cp.ndarray = None,
        ref_freq: np.ndarray | cp.ndarray = None,
        start_freq: np.ndarray | cp.ndarray = None,
        synchronize: bool = False,
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
            merger_time: Time of merger in seconds, shape (Nbatch,).
            ref_freq: Reference frequency in Hz. If `None`, it will default to `self.ref_freq` if that is not `None`, otherwise it has to be explicitly provided.
            start_freq: Starting frequency in Hz. If `None`, it will default to `self.start_freq` if that is not `None`, otherwise it has to be explicitly provided.
            synchronize: If `True`, it will call `block_until_ready()` on the waveform outputs. :meth:`self._from_jax` relies on `dlpack` to move data from JAX to Cupy, that should already handle CUDA streams correctly.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_strain_components_amp_phase``.

        Returns:
            times, mode amplitudes and mode phases
        """
        
        reference_kwargs = self.get_reference_quantities(
            merger_time=merger_time,
            start_freq=start_freq,
            ref_freq=ref_freq,
        )
        waveform_delta_t = kwargs.pop("delta_t", self.dt)
        reference_kwargs.update(kwargs)

        times, mask, amplitude, phase = self.waveform.compute_strain_components_amp_phase(
            self._to_jax(m1),
            self._to_jax(m2),
            self._to_jax(s1z),
            self._to_jax(s2z),
            self._to_jax(distance),
            self._to_jax(phi_ref),
            self._to_jax(inclination),
            self._to_jax(psi),
            delta_t=waveform_delta_t,
            **reference_kwargs,
        )
        #amplitude.block_until_ready()  # ensure all outputs are ready before moving to self.xp

        times = self._from_jax(times, do_synchronize=synchronize)
        mask = self._from_jax(mask, do_synchronize=synchronize)
        amplitude = self._from_jax(amplitude, do_synchronize=synchronize)
        phase = self._from_jax(phase, do_synchronize=synchronize)

        times_out = self.trim_and_shift_times(
            times,
            mask,
            xp=self.xp,
            dt=waveform_delta_t,
        )

        num_keep = times_out.shape[-1]

        # Do not taper an adaptive grid by node index.  A fixed number of
        # sparse inspiral nodes can span days, and multiplying phase by an
        # onset ramp changes the waveform rather than merely windowing it.
        return (
            times_out,
            amplitude[..., -num_keep:],
            phase[..., -num_keep:],
        )

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
