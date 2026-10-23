"""Binary black hole waveform classes (bbhx FD SNR waveform and phentax PhenomTHM TDI waveforms).

Classes:

- :class:`PhenomTHMTDIWaveform`: phentax ``IMRPhenomTHM`` polarizations with LAT's
  :class:`~lisatools.response.directresponse.pyResponseTDI` response
  (:class:`~lisatools.sources.waveformbase.TDPyResponseWaveformBase`). The stock global-fit MBH
  generator: with ``MBH_LIKELIHOOD=batched`` (the 9mo / 12mo runs) it rebuilds the engine
  residuals and serves as the batched move's reference generator, while scoring runs through
  the grid-aligned subclasses in :mod:`lisatools.sources.bbh.gridaligned`.
- :class:`PhenomTHMTDIOnFlyWaveform`: the same model through the TDI-on-the-fly response
  (``use_tdionfly``; off in production).
- :class:`PhenomTHMWaveformBase`: the shared phentax wrapper (a
  :class:`~lisatools.jax.jaxbase.JaxBase` mixin; reference-time / start-frequency handling).
- :class:`BBHSNRWaveform`: a quick bbhx ``BBHWaveformFD`` AET SNR helper (not a global-fit
  template).

The ``MBH_PHENOM_DEFAULT_*`` constants are the stock :class:`PhenomTHMTDIWaveform` defaults,
kept equal to ``globalfit.stock.erebor`` (``SourceMBHSettings`` /
``wrappers.get_mbh_phenom_wave_gen``). jax and phentax are optional imports: without them the
module still imports (``BBHSNRWaveform`` works) but the PhenomTHM classes raise ``ImportError``.

Waveform basis of the PhenomTHM classes (the stock MBH transform's output, ICRS run frame):
``(m1 [Msun], m2 [Msun], s1z, s2z, distance [Mpc], phi_ref, inclination, psi, ra, dec,
merger_time [s from waveform_t0])``.
"""

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
from ...utils.exceptions import BatchNotLaunchable

try:
    import jax
    import jax.numpy as jnp
    import phentax

    phentax_available = True
except (ImportError, ModuleNotFoundError):
    phentax_available = False
    jnp = np  # type: ignore
    # Bind the name even when the import failed. ``gridaligned`` imports
    # ``jax`` from here, and ``sources/bbh/__init__.py`` imports that module
    # unconditionally, so leaving it unbound turned an OPTIONAL dependency
    # into a hard one: without jax, importing lisatools.sources.bbh at all
    # failed, locking out users who only want the pure-bbhx BBHSNRWaveform.
    jax = None  # type: ignore

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
# Response Lagrange order: 8 (user ruling 2026-09-29; PR #82 measured the MBH
# mismatch flat from order 30 down to 4, ~1.6x cheaper). Set 30 for the old value.
MBH_PHENOM_DEFAULT_RESPONSE_ORDER = 8
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
        bbh_waveform_kwargs: ``amp_phase_kwargs`` for :class:`bbhx.waveformbuild.BBHWaveformFD`
            (``None`` -> ``{"run_phenomd": False}``).
        response_kwargs: ``response_kwargs`` for :class:`bbhx.waveformbuild.BBHWaveformFD`
            (``None`` -> ``{}``; ``TDItag="AET"`` and ``tdi2=True`` are filled in when absent).
        force_backend: Backend selection at construction (``"cpu"`` /
            ``"cuda12x"`` / ...); forwarded to :class:`bbhx.waveformbuild.BBHWaveformFD`.

    Attributes:
        wave_gen (BBHWaveformFD): The bbhx generator.

    Example:
        .. code-block:: python

            from lisatools.sources.bbh import BBHSNRWaveform

            wave = BBHSNRWaveform(force_backend="cpu")
            # (m1, m2, chi1z, chi2z, distance [m], phi_ref, f_ref, inc, lam, beta, psi, t_ref [s])
            A, E, T = wave(*params)
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
        """Frequency array [Hz] of the last call (unset before the first call)."""
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

        Calls :class:`bbhx.waveformbuild.BBHWaveformFD` with ``direct=True``, ``combine=True``
        on a sparse frequency grid and stores the grid in :attr:`f_arr`.

        Args:
            *params: Parameters for :class:`bbhx.waveformbuild.BBHWaveformFD`: ``(m1, m2
                [Msun], chi1z, chi2z, distance [m], phi_ref, f_ref, inc, lam, beta, psi,
                t_ref [s])``.
            return_array: If ``True``, return ``array([A, E, T])``.
                If ``False``, return ``(A, E, T)``.
            mf_min: Minimum dimensionless frequency ``M f`` to evaluate.
            mf_max: Maximum dimensionless frequency ``M f`` to evaluate.
            freqs: If ``None``, 1024 points log-spaced between ``mf_min / M`` and
                ``mf_max / M`` (``M = (m1 + m2) * MTSUN_SI``, in s). Otherwise this exact
                ``np.ndarray`` [Hz].
            **kwargs: ``kwargs`` for the ``__call__`` function
                for :class:`bbhx.waveformbuild.BBHWaveformFD`.

        Returns:
            np.ndarray | tuple: The ``(3, len(f_arr))`` complex AET array, or the tuple
            ``(A, E, T)`` (backend array module).
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
    """Base class for PhenomTHM waveforms (phentax ``IMRPhenomTHM`` wrapper).

    Not meant to be used directly: the shared code of :class:`PhenomTHMTDIWaveform` and
    :class:`PhenomTHMTDIOnFlyWaveform` (the phentax model, the reference-time / start-frequency
    handling of :meth:`get_reference_quantities`, and the time-grid helpers). JAX inputs and
    outputs move through the :class:`~lisatools.jax.jaxbase.JaxBase` helpers ``_to_jax`` /
    ``_from_jax``.

    Args:
        waveform_kwargs: Keyword arguments forwarded to :class:`phentax.waveform.IMRPhenomTHM`.
        Tobs: phentax generation window ``T`` in **seconds** (the template spans ``[-Tobs, ...]``
            before the merger when ``time_bounded_start``).
        start_freq: Starting frequency in Hz (phentax ``f_min``). If ``None``, it has to be
            provided in the waveform generation calls.
        ref_freq: Reference frequency in Hz (phentax ``f_ref``), used only when
            ``use_reference_time`` is ``False``. ``None`` is passed through unless the call
            provides one.
        use_reference_time: If ``True`` (default), phases are referenced to a time ``t_ref``
            instead of to ``f_ref``.
        use_coalescence_time: With ``use_reference_time``: ``True`` (default here) sets
            ``t_ref = 0`` (the merger); ``False`` sets ``t_ref = -merger_time`` per call, i.e.
            ``phi_ref`` is the phase at ``waveform_t0``.
        time_bounded_start: If ``True`` (default), passes ``t_min = -Tobs`` so the template
            spans its full window regardless of mass (phentax otherwise starts it where the
            (2,2) mode reaches ``f_min``).

    Raises:
        ImportError: If jax / phentax are not installed.
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
        """dict: Constructor kwargs of :class:`PhenomTHMWaveformBase` read back from the phentax
        model (``waveform_kwargs``, ``Tobs``, ``start_freq``, ``ref_freq``, the three flags), for
        reproducibility and debugging."""

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
        """Trim every row to the longest valid suffix and rebuild the invalid leading times.

        Keeps the last ``max_valid = max(mask.sum(axis=1))`` columns. A row with fewer valid
        samples gets its ``n_pad`` leading entries replaced by times continuing its first valid
        time backwards in steps of ``dt``, so every row is a strictly increasing, ``dt``-spaced
        grid. Assumes each row's valid samples are a right-aligned suffix.

        Args:
            times: Time arrays for each source, shape ``(Nbatch, Ntimes)``.
            mask: Boolean mask of valid samples, shape ``(Nbatch, Ntimes)``.
            xp: Array module (numpy or cupy).
            dt: Time step in seconds.

        Returns:
            xp.ndarray: Times, shape ``(Nbatch, max_valid)``.
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
        """Leading onset ramp that turns each row's waveform on smoothly.

        ``0.5 (1 - cos(pi x))`` with ``x = clip((j - num_pad) / taper_length, 0, 1)``: zero on
        the invalid head, a raised-cosine over ``taper_length`` samples, then one. Avoids a
        sharp jump from zero to the first valid point.

        Args:
            num_points: Total number of points per row.
            num_pad: Invalid leading points per row, shape ``(Nbatch,)`` (int array-like).
            taper_length: Length of the taper in number of points.
            xp: Array module (numpy or cupy).

        Returns:
            xp.ndarray: Ramp, shape ``(Nbatch, num_points)``.
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
        """Reference kwargs for the phentax call, following the class flags.

        With ``use_reference_time``: ``{"t_ref", "f_min"}`` where ``t_ref = -merger_time`` (or
        ``0`` with ``use_coalescence_time``); otherwise ``{"f_min", "f_ref"}``. With
        ``time_bounded_start`` the dict also carries ``t_min = -Tobs``.

        Args:
            merger_time: Merger time [s] relative to ``waveform_t0``, scalar or ``(Nbatch,)``.
            start_freq: Starting frequency in Hz. ``None`` uses the constructor's
                ``start_freq``.
            ref_freq: Reference frequency in Hz. ``None`` uses the constructor's ``ref_freq``.

        Returns:
            dict: Keyword arguments for phentax (JAX arrays).
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
    """Generate PhenomTHM waveforms with the TDI LISA Response (global-fit aligned).

    A bare instance reproduces the generator configuration the stock ``all_sources`` global fit
    uses for its MBH branch (see
    ``lisatools.globalfit.stock.erebor.wrappers.get_mbh_phenom_wave_gen`` and
    ``source_runtime.SourceMBHSettings``). Every waveform-config default below is drawn from the
    module-level ``MBH_PHENOM_DEFAULT_*`` constants (the sources-side single source of truth);
    only the *run-specific* arguments (``waveform_t0``, ``data_td_settings``, ``orbits``,
    ``sampling_frequency``, ``tukey_alpha``, ``output_domain_settings``, ``force_backend``) must
    be supplied per run.

    In the 9mo / 12mo runs (``MBH_LIKELIHOOD=batched``) this class rebuilds the engine's MBH
    residuals (:meth:`~lisatools.sources.waveformbase.TDWaveformBase.get_signals_for_residuals`
    onto the WDM domain) and is the batched move's reference generator; the scoring itself
    uses :class:`~lisatools.sources.bbh.gridaligned.WindowedGridAlignedMBHWaveform`.

    Call / :meth:`compute_tdi_channels` parameters: ``(m1, m2, s1z, s2z, distance, phi_ref,
    inclination, psi, ra, dec, merger_time)`` -- masses in solar masses, distance in Mpc,
    angles in radians, ``(ra, dec, psi)`` in the orbits' frame (ICRS on the run),
    ``merger_time`` in seconds from ``waveform_t0``. Scalars give one source; 1-D arrays a
    batch.

    Args:
        waveform_kwargs: Keyword arguments forwarded to
            :class:`phentax.waveform.IMRPhenomTHM`. ``None`` (default) uses a
            copy of :data:`MBH_PHENOM_DEFAULT_WAVEFORM_KWARGS` (higher modes
            ``(21, 33, 44)``, negative modes on, ``t_low_fit=True``,
            ``coarse_grain=False``, ``atol=rtol=1e-12``).
        Tobs: phentax generation window in seconds (its ``T``); default
            :data:`MBH_PHENOM_DEFAULT_TOBS` (1/12 yr).
        start_freq: Starting frequency in Hz; default
            :data:`MBH_PHENOM_DEFAULT_START_FREQ`.
        ref_freq: Reference frequency in Hz, used only with ``use_reference_time=False``
            (``None`` is passed through).
        use_reference_time: Reference the phase to a time (default ``True``); see
            :class:`PhenomTHMWaveformBase`.
        use_coalescence_time: Default ``False`` here: ``phi_ref`` is the phase at
            ``waveform_t0`` (``t_ref = -merger_time``); ``True`` references it to the merger.
        time_bounded_start: Bound the template at ``t_min = -Tobs`` (default ``True``).
        *args: Additional positional arguments forwarded to
            :class:`~lisatools.sources.waveformbase.TDPyResponseWaveformBase` (its first
            positional is ``waveform_t0``). Pass the run arguments by keyword.
        order: pyResponseTDI Lagrange order; default
            :data:`MBH_PHENOM_DEFAULT_RESPONSE_ORDER` (8; the launchers set
            ``MBH_RESPONSE_ORDER=8``).
        buffer_time: TDI response buffer in seconds; default
            :data:`MBH_PHENOM_DEFAULT_BUFFER_TIME`.
        freq_min: FD / STFT output lower frequency bound [Hz]; default
            :data:`MBH_PHENOM_DEFAULT_FREQ_MIN`.
        freq_max: FD / STFT output upper frequency bound [Hz]; default
            :data:`MBH_PHENOM_DEFAULT_FREQ_MAX`.
        fft_batch_size: FD transform batch size; default
            :data:`MBH_PHENOM_DEFAULT_FFT_BATCH_SIZE`.
        tdi_generation: TDI generation; default :data:`MBH_PHENOM_DEFAULT_TDI_GENERATION`
            (2nd generation).
        tdi_channels: TDI channels; default :data:`MBH_PHENOM_DEFAULT_TDI_CHANNELS` (``XYZ``).
        **kwargs: Run-specific keyword arguments forwarded to
            :class:`~lisatools.sources.waveformbase.TDPyResponseWaveformBase`
            (``waveform_t0``, ``data_td_settings``, ``orbits``, ``sampling_frequency``,
            ``tukey_alpha``, ``stft_dt``, ``output_domain_settings``, ``force_backend``, ...).

    Raises:
        ValueError: If ``waveform_kwargs`` sets ``coarse_grain=True`` (``pyResponseTDI`` needs
            equispaced times).
        ImportError: If jax / phentax are not installed.

    Example:
        .. code-block:: python

            from lisatools.detector import EqualArmlengthOrbits
            from lisatools.domains import TDSettings
            from lisatools.sources.bbh import PhenomTHMTDIWaveform

            dt, Tobs_wf = 10.0, 7.0 * 86400.0
            mbh = PhenomTHMTDIWaveform(
                waveform_t0=0.0,
                data_td_settings=TDSettings(N=int(Tobs_wf / dt) + 4096, dt=dt, t0=0.0,
                                            force_backend="cpu"),
                Tobs=Tobs_wf,
                sampling_frequency=1.0 / dt,
                orbits=EqualArmlengthOrbits(force_backend="cpu"),
                force_backend="cpu",
            )
            # (m1, m2, s1z, s2z, dist [Mpc], phi_ref, iota, psi, ra, dec, merger_time [s])
            times, xyz = mbh.compute_tdi_channels(
                1.2e6, 0.8e6, 0.3, 0.2, 15000.0, 1.0, 1.0, 1.2, 2.0, 0.4, 0.9 * Tobs_wf)
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
        """dict: ``phenom_kwargs`` merged with ``wrapper_kwargs`` (reproducibility, debugging)."""

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
        """Generate the polarizations of a single source with phentax.

        ``compute_polarizations_at_once`` on the ``dt`` grid with the reference kwargs of
        :meth:`get_reference_quantities`; the outputs are masked to the valid samples.

        Args:
            m1: Source-1 mass in solar masses.
            m2: Source-2 mass in solar masses.
            s1z: Dimensionless aligned spin of source 1.
            s2z: Dimensionless aligned spin of source 2.
            distance: Luminosity distance in Mpc.
            phi_ref: Reference phase in radians.
            inclination: Inclination angle in radians.
            psi: Polarisation angle in radians.
            ra: Right ascension in radians; accepted for the call signature, unused here (the
                response consumes it).
            dec: Declination in radians; unused here, as ``ra``.
            merger_time: Merger time in seconds relative to ``waveform_t0`` (sets ``t_ref``).
            ref_freq: Reference frequency in Hz. ``None`` uses the constructor's value.
            start_freq: Starting frequency in Hz. ``None`` uses the constructor's value.
            synchronize: If ``True``, synchronize when moving the JAX outputs to :attr:`xp`.
                ``_from_jax`` relies on DLPack, which should already handle CUDA streams.

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(t_arr, h_plus, h_cross)``, 1-D on the
            valid samples, ``t_arr`` [s] relative to the merger.
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

        phentax uses JAX vmap internally, so all parameters must be broadcastable to the batch
        shape. The rows are trimmed to the longest valid suffix
        (:meth:`trim_and_shift_times`); rows with fewer valid samples are masked by the
        leading onset ramp (taper ``5 * tdi_buffer_time``) anchored at their own onset.

        Args:
            m1: Source-1 masses in solar masses, shape ``(Nbatch,)``.
            m2: Source-2 masses in solar masses, shape ``(Nbatch,)``.
            s1z: Dimensionless aligned spin of source 1, shape ``(Nbatch,)``.
            s2z: Dimensionless aligned spin of source 2, shape ``(Nbatch,)``.
            distance: Luminosity distance in Mpc, shape ``(Nbatch,)``.
            phi_ref: Reference phase in radians, shape ``(Nbatch,)``.
            inclination: Inclination angle in radians, shape ``(Nbatch,)``.
            psi: Polarisation angle in radians, shape ``(Nbatch,)``.
            ra: Right ascension in radians, shape ``(Nbatch,)``; unused here.
            dec: Declination in radians, shape ``(Nbatch,)``; unused here.
            merger_time: Merger time in seconds relative to ``waveform_t0``, ``(Nbatch,)``.
            ref_freq: Reference frequency in Hz (``None``: the constructor's).
            start_freq: Starting frequency in Hz (``None``: the constructor's).
            synchronize: If ``True``, synchronize when moving the JAX outputs to :attr:`xp`.
            onset_ramp: Apply the leading onset ramp (default ``True``). ``False`` returns the
                bit-identical-to-:meth:`wave_gen` samples and requires every row to have the
                same number of valid samples.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_polarizations_at_once`` (e.g. ``T`` for observation time
                override, ``t_min``, ``t_ref``).

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(times_batch, h_plus_batch,
            h_cross_batch)``, each ``(Nbatch, N_valid_times)`` on :attr:`xp`, times [s]
            relative to each merger.

        Raises:
            BatchNotLaunchable: With ``onset_ramp=False`` when the rows have different valid
                lengths (the container falls back to per-row evaluation).
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
                # A geometry refusal, not bad arguments: this batch's rows
                # produced unequal valid lengths. Typed so the container can
                # fall back to per-row evaluation instead of dying -- the
                # serial loop handles this case fine.
                raise BatchNotLaunchable(
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




class PhenomTHMTDIOnFlyWaveform(TDTDIOnFlyWaveformBase, PhenomTHMWaveformBase):
    """Generate PhenomTHM waveforms with the TDI LISA Response, on the fly.

    Inherits from :class:`~lisatools.sources.waveformbase.TDTDIOnFlyWaveformBase`: the
    waveform is generated per mode as amplitude and phase (already including the
    spherical-harmonic factors) and the TDI delays are evaluated on splines of each mode. No
    global-fit builder constructs it (``globalfit.moves.mbhspecialmove`` only names it in a
    type annotation); a general-use alternative to :class:`PhenomTHMTDIWaveform`.

    Args:
        waveform_kwargs: Keyword arguments for :class:`phentax.waveform.IMRPhenomTHM` (no
            default here).
        Tobs: phentax generation window ``T`` in seconds.
        start_freq: Starting frequency in Hz. Default ``None``.
        ref_freq: Reference frequency in Hz. Default ``None``.
        use_reference_time: See :class:`PhenomTHMWaveformBase`. Default ``True``.
        use_coalescence_time: See :class:`PhenomTHMWaveformBase`. Default ``False``.
        *args: Positional arguments for
            :class:`~lisatools.sources.waveformbase.TDTDIOnFlyWaveformBase` (first:
            ``waveform_t0``).
        **kwargs: Keyword arguments for
            :class:`~lisatools.sources.waveformbase.TDTDIOnFlyWaveformBase`
            (``data_td_settings``, ``orbits``, ``zero_inclination``, ...).
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
        """dict: ``phenom_kwargs`` merged with ``wrapper_kwargs`` (reproducibility, debugging)."""

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
        the largest such read is ``|k.x|/c <= 499`` s.  Trimming here as well only
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
            synchronize: If ``True``, synchronize when moving the JAX outputs to :attr:`xp`.
                ``_from_jax`` relies on DLPack, which should already handle CUDA streams.
            **kwargs: Additional keyword arguments forwarded to
                ``compute_strain_components_amp_phase`` (``delta_t`` is popped first and
                replaces ``dt`` as the generation step).

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(times, amplitude, phase)``: times
            ``(Nbatch, N)`` relative to the merger (:meth:`trim_and_shift_times`) and mode
            amplitudes / phases ``(Nbatch, num_modes, N)``. No onset taper is applied (it would
            change the waveform on an adaptive grid).
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
        r"""Map the phentax mode amplitude and phase to the TDI-on-the-fly convention.

        For this waveform :math:`A \to A/2` and :math:`\phi \to \pi - \phi`.

        Args:
            amp: Mode amplitudes, already including the spherical-harmonic contributions,
                shape ``(num_binaries, num_modes, num_times)``.
            phase: Mode phases, same shape.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(processed_amplitude, processed_phase)``, same
            shapes.
        """

        return (amp / 2.0, self.xp.pi - phase)
