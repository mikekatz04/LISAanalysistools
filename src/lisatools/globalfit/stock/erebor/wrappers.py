"""Waveform-generator and response-wrapper blocks for the Erebor family.

Cached template/injection generators (legacy ResponseWrapper paths and the
validated TDI-on-the-fly paths) plus the thin ``*WaveWrap`` adapters that
project raw TD channels onto the run's analysis domain. Everything here is
fully parameterized — no module-global run knobs — so stock variants and
settings files bind their own configuration explicitly.

Heavy third-party imports (``bbhx``, ``phentax``, ``few``) stay
function-local so importing this module never hard-fails for runs that
don't need them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.domains import TDSettings, TDSignal
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.sobbh import SOBBHWaveform
from lisatools.utils.constants import YRSID_SI

# Canonical EMRI response-wrapper machinery lives in the installed
# ``lisatools.sources.emri`` package (2026-07-01 carve-out); re-exported here
# so the EMRI and SOBBH/MBH wrapper families share one import home.
from lisatools.sources.emri import (  # noqa: F401
    EMRI_INSPIRAL_KWARGS,
    EMRI_MODE_SELECTOR_KWARGS,
    EMRI_SUM_KWARGS,
    EMRIWaveWrap,
    get_emri_response_wrapper,
)

_SOBBH_WAVE_GEN_CACHE = {}


def get_sobbh_response_wrapper(
    *,
    Tobs: float,
    dt: float,
    t_start: float,
    tdi_config: TDIConfig,
    tdi_chan: str = "XYZ",
    role: str = "template",
    order: int = 40,
    t_buffer: float = 3e4,
    orbits: Optional[Orbits] = None,
    force_backend: str = "cpu",
    reference_time: Optional[float] = None,
):
    """Build (and cache) a :class:`ResponseWrapper` around :class:`SOBBHWaveform`.

    Mirrors :func:`get_emri_response_wrapper`; one generator per
    ``(Tobs, dt, t_start, tdi_chan, order, force_backend, id(orbits))``
    cache key so the injection path and the template path share the same
    instance.
    """
    key = (
        Tobs, dt, t_start, tdi_chan, order, force_backend, id(orbits),
        reference_time,
    )
    if key in _SOBBH_WAVE_GEN_CACHE:
        return _SOBBH_WAVE_GEN_CACHE[key]

    sobbh_generator = SOBBHWaveform(
        Tobs=Tobs,
        dt=dt,
        t0=t_start,
        reference_time=reference_time,
        force_backend=force_backend,
    )

    response_kwargs = {
        "Tobs": Tobs / YRSID_SI,
        "dt": dt,
        "index_lambda": 7,
        "index_beta": 8,
        "flip_hx": True,
        "force_backend": force_backend,
        "tdi": tdi_config,
        "tdi_chan": tdi_chan,
        "order": order,
        "remove_garbage": "zero",
        "is_ecliptic_latitude": True,
        "t_buffer": t_buffer,
    }

    if orbits is None:
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
    wave_gen = ResponseWrapper(
        sobbh_generator,
        orbits=orbits,
        t0=t_start,
        **response_kwargs,
    )
    _SOBBH_WAVE_GEN_CACHE[key] = wave_gen
    return wave_gen


class SOBBHWaveWrap:
    """Run the cached SOBBH ResponseWrapper and project to the run's domain.

    Mirrors :class:`EMRIWaveWrap` — output is a :class:`DomainBase`
    subclass (FDSignal / WDMSignal / ...) so ACA dispatch and the SOBBH
    move's ``get_waveform_here`` land on the right kernels.
    """

    def __init__(
        self,
        wave_gen,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
    ):
        self.wave_gen = wave_gen
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels

    def __call__(self, *params, **kwargs):
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        arr = np.asarray(self.wave_gen(*params, **call_kwargs))
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )



# ============================================================
# *** TDI-on-the-fly template generators (MBH + SOBBH) ***
# ============================================================
#
# The validated (2026-06-15) time-domain TDI-on-the-fly path beats the
# legacy pyResponse / PhenomTHMTDIWaveform templates against the mojito
# L1 data by 1-2 orders of magnitude (MBH mm ~6.3e-7 >1mHz; SOBBH WDM
# mm5 ~3.5e-7).  These generators return raw TD channel arrays evaluated
# on the data time grid, so each is wrapped in a thin adapter that
# projects to the run's domain via ``TDSignal(...).transform(...)`` —
# mirroring :class:`EMRIWaveWrap` / :class:`SOBBHWaveWrap`.
#
# bbhx / phentax imports are deferred to the builders so importing this
# module never hard-fails for runs (e.g. EMRI-only) that don't need them.

# phentax IMRPhenomTHM configuration (validated in
# scripts/mbh/mbh_likelihood_compare.py).  coarse_graining_scale_factor=48
# resolves the merger/ringdown (~0.886 s spacing at merger) and drops the
# legacy<->on-the-fly mismatch to mm 6.3e-7 (>1mHz); the phentax default 12
# under-resolves it and leaves a spurious ~5e-22 low-frequency floor.
MBH_TDIONFLY_HIGHER_MODES = (21, 33, 44)
MBH_TDIONFLY_TOL = 1e-12
MBH_TDIONFLY_COARSE_SCALE = 48.0

_MBH_TDIONFLY_GEN_CACHE = {}
_SOBBH_TDIONFLY_GEN_CACHE = {}


def get_mbh_tdionfly_gen(
    *,
    dt: float,
    t_start: float,
    dur_s: float,
    tdi_config: TDIConfig,
    orbits: Optional[Orbits] = None,
    waveform_duration: Optional[float] = None,
    higher_modes=MBH_TDIONFLY_HIGHER_MODES,
    coarse_scale: float = MBH_TDIONFLY_COARSE_SCALE,
    tol: float = MBH_TDIONFLY_TOL,
    dt_min: float = 0.1,
    force_backend: str = "cpu",
):
    """Build (and cache) a :class:`bbhx.mbhtdionfly.MBHTDIonFly`.

    Mirrors the validated recipe in
    ``scripts/mbh/mbh_likelihood_compare.py``: a phentax ``IMRPhenomTHM``
    with coarse-grained amp/phase output feeding the LISA
    TDI-on-the-fly response.  ``t_start`` is the epoch the merger time is
    referenced to (``MOJITO_REFERENCE_TIME`` in mojito mode); the merger
    lands at ``t_start + t_merge``.  ``dur_s`` is both the phentax ``T``
    and the generator's observation window (pre-merger inspiral + margin).
    """
    higher_modes = tuple(higher_modes)
    key = (
        dt, t_start, dur_s, force_backend, waveform_duration, higher_modes,
        coarse_scale, tol, dt_min, id(orbits),
    )
    if key in _MBH_TDIONFLY_GEN_CACHE:
        return _MBH_TDIONFLY_GEN_CACHE[key]

    from bbhx.mbhtdionfly import MBHTDIonFly
    from phentax.waveform import IMRPhenomTHM

    wave_gen = IMRPhenomTHM(
        higher_modes=list(higher_modes),
        include_negative_modes=True,
        t_low_fit=True,
        coarse_grain=True,
        coarse_graining_scale_factor=coarse_scale,
        atol=tol,
        rtol=tol,
        T=dur_s,
    )
    gen = MBHTDIonFly(
        wave_gen,
        orbits,
        tdi_config,
        dt,
        dur_s,
        t0=t_start,
        dt_min=dt_min,
        waveform_duration=waveform_duration if waveform_duration is not None else dur_s,
        force_backend=force_backend,
    )
    _MBH_TDIONFLY_GEN_CACHE[key] = gen
    return gen


class MBHTDIonFlyWaveWrap:
    """Adapter: :class:`MBHTDIonFly` TD output -> run-domain signal.

    Consumes the MBH **waveform** basis produced by
    ``make_mbh_transform_container().both_transforms`` —
    ``(m1, m2, s1z, s2z, dist[Mpc], phi_ref, inc, psi, ra, dec, t_plunge)``
    (psi before ra/dec) — and reorders it to the ``MBHTDIonFly`` call
    order ``(..., inc, ra, dec, psi, t_merge)`` before evaluating on the
    data time grid (``t_arr``) with ``combine=True``.
    """

    def __init__(
        self,
        wave_gen,
        t_arr: np.ndarray,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
    ):
        self.wave_gen = wave_gen
        self.t_arr = t_arr
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels

    def raw_td(self, *params, **kwargs):
        """Combined TD TDI channels on the data grid (before domain projection)."""
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        # The TDI-on-the-fly generator reads (ra, dec) in the orbits frame
        # directly; drop the legacy ResponseWrapper kwarg if present.
        call_kwargs.pop("convert_to_ra_dec", None)
        m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = params
        arr = np.asarray(
            self.wave_gen(
                m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                upsample_t_arr=self.t_arr, combine=True, **call_kwargs,
            )
        )
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return arr

    def __call__(self, *params, **kwargs):
        arr = self.raw_td(*params, **kwargs)
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )


def get_sobbh_tdionfly_gen(
    *,
    Tobs: float,
    dt: float,
    t_start: float,
    tdi_config: TDIConfig,
    reference_time: Optional[float],
    orbits: Optional[Orbits] = None,
    n_grid: int = 2048,
    buffer_time: float = 5000.0,
    force_backend: str = "cpu",
):
    """Build (and cache) a :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly`.

    Mirrors ``scripts/sobbh/sobbh_likelihood_compare.py``: reuses the same
    :class:`SOBBHWaveform` amp/phase the legacy path uses, but projects via
    the analytic TDI-on-the-fly response.  ``reference_time`` is the epoch
    ``f_low`` is defined at (``MOJITO_REFERENCE_TIME`` in mojito mode).
    """
    key = (
        Tobs, dt, t_start, force_backend, reference_time, n_grid, buffer_time,
        id(orbits),
    )
    if key in _SOBBH_TDIONFLY_GEN_CACHE:
        return _SOBBH_TDIONFLY_GEN_CACHE[key]

    from bbhx.sobbhtdionfly import SOBBHTDIonFly

    wave_gen = SOBBHWaveform(
        Tobs=Tobs, dt=dt, t0=t_start, reference_time=reference_time,
        force_backend=force_backend,
    )
    gen = SOBBHTDIonFly(
        wave_gen,
        orbits,
        tdi_config,
        dt,
        Tobs,
        t0=t_start,
        n_grid=n_grid,
        buffer_time=buffer_time,
        force_backend=force_backend,
    )
    _SOBBH_TDIONFLY_GEN_CACHE[key] = gen
    return gen


class SOBBHTDIonFlyWaveWrap:
    """Adapter: :class:`SOBBHTDIonFly` TD output -> run-domain signal.

    Consumes the SOBBH **waveform** (full) basis
    ``(m1, m2, s1, s2, dist[Gpc], inc, f_low, ra, dec, psi, phi0)`` and
    reorders it to the ``SOBBHTDIonFly`` call order
    ``(m1, m2, s1, s2, dist, f_low, phi0, inc, ra, dec, psi)`` (no
    ``flip_hx`` arg — the on-the-fly response is the correct handedness;
    ``phi0`` is the catalogue ``TrueAnomaly``).
    """

    def __init__(
        self,
        wave_gen,
        t_arr: np.ndarray,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
    ):
        self.wave_gen = wave_gen
        self.t_arr = t_arr
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels

    def raw_td(self, *params, **kwargs):
        """Combined TD TDI channels on the data grid (before domain projection)."""
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs.pop("convert_to_ra_dec", None)
        m1, m2, s1, s2, dist, inc, f_low, ra, dec, psi, phi0 = params
        arr = np.asarray(
            self.wave_gen(
                m1, m2, s1, s2, dist, f_low, phi0, inc, ra, dec, psi,
                upsample_t_arr=self.t_arr, combine=True, **call_kwargs,
            )
        )
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return arr

    def __call__(self, *params, **kwargs):
        arr = self.raw_td(*params, **kwargs)
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )


# ---- Cached MBH PhenomTHMTDIWaveform (legacy MBH template path) ------------
#
# Shared between the engine-side residual rebuild (``signal_gen``), the
# synthetic injection stream, and the PE move so the (slow) phentax setup
# runs once per configuration. Fully parameterized version of the builder
# that grew up in full_year_combined_global_fit_settings.py; the defaults
# are that file's validated constants.

_MBH_PHENOM_GEN_CACHE = {}


def get_mbh_phenom_wave_gen(
    *,
    data_td_settings: TDSettings,
    waveform_t0: float,
    dt: float,
    orbits=None,
    output_domain_settings=None,
    tukey_alpha: float = 0.0,
    force_backend: str = "cpu",
    waveform_duration: Optional[float] = YRSID_SI / 12.0,
    data_span: Optional[float] = None,
    higher_modes: Sequence[int] = (21, 33, 44),
    phenom_tol: float = 1e-12,
    start_freq: float = 7e-5,
    response_order: int = 30,
    buffer_time: float = 15_000.0,
    tdi_gen_str: str = "2nd generation",
    tdi_chan: str = "XYZ",
    min_freq: float = 1e-4,
    max_freq: float = 2.5e-2,
    fft_batch_size: int = 2,
):
    """Build (and cache) the MBH ``PhenomTHMTDIWaveform`` (legacy path).

    ``waveform_duration`` is the phentax generation window (its ``T``), not
    the data span; ``None`` falls back to ``data_span`` (which must then be
    given). ``waveform_t0`` is the epoch merger times are referenced to
    (``MOJITO_REFERENCE_TIME`` in mojito mode, the data start in synthetic
    mode).
    """
    higher_modes = tuple(higher_modes)
    if waveform_duration is None:
        if data_span is None:
            raise ValueError("waveform_duration=None requires data_span.")
        waveform_duration = data_span
    key = (
        id(data_td_settings), waveform_t0, id(orbits), id(output_domain_settings),
        tukey_alpha, force_backend, waveform_duration, higher_modes, phenom_tol,
        start_freq, response_order, buffer_time, tdi_gen_str, tdi_chan, dt,
        min_freq, max_freq, fft_batch_size,
    )
    if key in _MBH_PHENOM_GEN_CACHE:
        return _MBH_PHENOM_GEN_CACHE[key]

    from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

    gen = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(
            higher_modes=list(higher_modes),
            include_negative_modes=True,  # negative m modes by symmetry
            t_low_fit=True,  # fit-seeded start time for the t(f) root finder
            coarse_grain=False,  # pyResponseTDI needs equispaced time arrays
            atol=phenom_tol,
            rtol=phenom_tol,
        ),
        # Waveform-generation window only (phentax ``T``), not the data span.
        Tobs=waveform_duration,
        start_freq=start_freq,
        use_reference_time=True,
        waveform_t0=waveform_t0,
        data_td_settings=data_td_settings,
        tdi_generation=tdi_gen_str,
        tdi_channels=tdi_chan,
        sampling_frequency=1.0 / dt,
        orbits=orbits,
        order=response_order,
        tukey_alpha=tukey_alpha,
        stft_dt=None,
        freq_min=min_freq,
        freq_max=max_freq,
        fft_batch_size=fft_batch_size,
        buffer_time=buffer_time,
        # WDM run target — communicated by settings object (sprint rule).
        # None on the injection path (TD output only).
        output_domain_settings=output_domain_settings,
        force_backend=force_backend,
    )
    _MBH_PHENOM_GEN_CACHE[key] = gen
    return gen
