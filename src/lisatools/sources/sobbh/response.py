"""Stock SOBBH response-wrapper builders + domain-projection adapters.

Carved out of the global-fit settings files (2026-07-01): the per-source
``get_sobbh_response_wrapper`` / ``SOBBHWaveWrap`` (legacy pyResponse path) and
``get_sobbh_tdionfly_gen`` / ``SOBBHTDIonFlyWaveWrap`` (validated on-the-fly
path) lived (duplicated) in ``sobbh_only`` and ``global_fit`` settings.

:func:`build_sobbh_stock_waveform` is the documented stand-alone entry (LATW tutorials). In
the 9mo / 12mo runs (``SOBBH_LIKELIHOOD=lookup``) scoring and template fills use
:mod:`lisatools.sources.sobbh.wdm_direct`; the stock code keeps its own TDI-on-the-fly
cross-check wrap (its own ``get_sobbh_tdionfly_gen`` in ``globalfit.stock.erebor.wrappers``),
and this module's :func:`get_sobbh_tdionfly_gen` is reached by the global fit only through the
synthetic injections. The ``SOBBH_STOCK_*`` constants are the defaults mirrored by
``SourceSOBBHSettings``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...detector import EqualArmlengthOrbits, Orbits
from ...domains import TDSettings, TDSignal
from ...response.directresponse import ResponseWrapper
from ...response.tdiconfig import TDIConfig
from ...utils.constants import YRSID_SI
from .waveform import SOBBHWaveform

# Process-wide caches (injection + template paths share one instance).
_SOBBH_WAVE_GEN_CACHE: dict = {}
_SOBBH_TDIONFLY_GEN_CACHE: dict = {}

# Stock SOBBH TDI-on-the-fly defaults — the sources-side single source of
# truth, mirrored by ``SourceSOBBHSettings`` (n_grid / buffer_time /
# response_order) and the ``get_sobbh_tdionfly_gen`` signature. Kept equal to
# the erebor values (enforced by tests/test_stock_waveform_alignment.py).
SOBBH_STOCK_N_GRID = 2048
SOBBH_STOCK_BUFFER_TIME = 5000.0
SOBBH_STOCK_RESPONSE_ORDER = 40  # legacy-ResponseWrapper path only (unused by TOF)


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

    One generator per
    ``(Tobs, dt, t_start, tdi_chan, order, force_backend, id(orbits), reference_time)``
    cache key so the injection path and template path share the same instance
    (``tdi_config`` and ``t_buffer`` are not in the key). The legacy pyResponse path
    (``use_tdionfly=False``): ``index_lambda=7``, ``index_beta=8`` (latitude,
    ``is_ecliptic_latitude=True``), ``flip_hx=True``, ``remove_garbage="zero"``.

    ``reference_time`` is the absolute epoch at which ``f_low`` is defined
    (``MOJITO_REFERENCE_TIME`` in mojito mode); it is decoupled from the
    data-window start ``t_start``. ``None`` -> ``f_low`` at the window start.

    Args:
        Tobs: Data span [s] (converted to years for the response).
        dt: Sample step [s].
        t_start: Data-window start [s] (the generator and response ``t0``).
        tdi_config: :class:`~lisatools.response.tdiconfig.TDIConfig`.
        tdi_chan: TDI channels. Default ``"XYZ"``.
        role: Informational only (not used, not in the key). Default ``"template"``.
        order: Response Lagrange order. Default ``40``.
        t_buffer: Response garbage buffer [s]. Default ``3e4``.
        orbits: :class:`~lisatools.detector.Orbits`; ``None`` builds
            :class:`~lisatools.detector.EqualArmlengthOrbits` on ``force_backend``.
        force_backend: Backend name. Default ``"cpu"``.
        reference_time: See above.

    Returns:
        ResponseWrapper: Called on the 11 (+ ``t_shift``) :class:`SOBBHWaveform` parameters;
        returns the per-channel TDI time series.
    """
    key = (
        Tobs,
        dt,
        t_start,
        tdi_chan,
        order,
        force_backend,
        id(orbits),
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

    # SOBBH output-basis positions of lam / beta.
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

    Output is a :class:`~lisatools.domains.DomainBase` subclass (FDSignal /
    WDMSignal / ...) so ACA dispatch and the SOBBH move's ``get_waveform_here``
    land on the right kernels. Calls default ``convert_to_ra_dec=False``.

    Args:
        wave_gen: Generator from :func:`get_sobbh_response_wrapper`.
        td_settings: :class:`~lisatools.domains.TDSettings` of the data grid.
        target_domain (DomainSettingsBase): Analysis-domain settings to transform onto.
        td_window (xp.ndarray, optional): Time-domain window for the transform.
        runtime_kwargs: Per-call kwargs; call kwargs override them.
        nchannels: Leading TDI channels to keep (``None``: all).
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
        """Template on ``target_domain`` from the :class:`SOBBHWaveform` parameters.

        Returns:
            DomainBase: The transformed signal.
        """
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        # SOBBHWaveform doesn't use convert_to_ra_dec; ResponseWrapper pops it.
        call_kwargs.setdefault("convert_to_ra_dec", False)
        arr = np.asarray(self.wave_gen(*params, **call_kwargs))
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return TDSignal(arr, self.td_settings).transform(self.target_domain, window=self.td_window)


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

    Reuses the same :class:`SOBBHWaveform` amp/phase the legacy path uses, but
    projects via the analytic TDI-on-the-fly response (validated in
    ``scripts/sobbh/sobbh_likelihood_compare.py``). ``reference_time`` is the
    epoch ``f_low`` is defined at (``MOJITO_REFERENCE_TIME`` in mojito mode).
    The cache key is ``(Tobs, dt, t_start, force_backend, reference_time, n_grid,
    buffer_time, orbits)`` (``id(orbits)``, or a fixed tag for ``None``); ``tdi_config`` is
    not in it.

    Args:
        Tobs: Data span [s].
        dt: Sample step [s].
        t_start: Data-window start [s] (the generator ``t0``).
        tdi_config: :class:`~lisatools.response.tdiconfig.TDIConfig`.
        reference_time: Epoch [s] of ``f_low`` / ``phi0``; ``None`` -> ``t_start``.
        orbits: Response orbits; ``None`` builds
            :class:`~lisatools.detector.EqualArmlengthOrbits` on ``force_backend`` (so the
            orbit tables live on the generator's backend).
        n_grid: Node grid size of the response. Default ``2048``.
        buffer_time: Node grid padding [s]. Default ``5000.0``.
        force_backend: Backend name. Default ``"cpu"``.

    Returns:
        SOBBHTDIonFly: Called as ``(m1, m2, s1, s2, dist, f_low, phi0, inc, ra, dec, psi)``
        plus its keyword options.
    """
    key = (
        Tobs,
        dt,
        t_start,
        force_backend,
        reference_time,
        n_grid,
        buffer_time,
        "default-orbits" if orbits is None else id(orbits),
    )
    if key in _SOBBH_TDIONFLY_GEN_CACHE:
        return _SOBBH_TDIONFLY_GEN_CACHE[key]

    if orbits is None:
        # Default the orbits on THIS generator's backend (mirrors
        # get_sobbh_response_wrapper). Leaving None lets the downstream
        # response default them with AUTO backend resolution, which picks
        # the best available LAT backend — on a GPU machine that hands
        # cupy orbit tables to a CPU-resolved generator (e.g. a CPU-only
        # bbhx build) and dies in OrbitsWrap.
        orbits = EqualArmlengthOrbits(force_backend=force_backend)

    from bbhx.sobbhtdionfly import SOBBHTDIonFly

    wave_gen = SOBBHWaveform(
        Tobs=Tobs,
        dt=dt,
        t0=t_start,
        reference_time=reference_time,
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
    ``(m1, m2, s1, s2, dist[Gpc], inc, f_low, ra, dec, psi, phi0)`` and reorders
    it to the ``SOBBHTDIonFly`` call order
    ``(m1, m2, s1, s2, dist, f_low, phi0, inc, ra, dec, psi)`` (no ``flip_hx``
    arg — the on-the-fly response is the correct handedness; ``phi0`` is the
    catalogue ``TrueAnomaly``). ``ra`` / ``dec`` / ``psi`` are read in the orbits' frame (ICRS
    with ``frame="icrs"`` orbits).

    Args:
        wave_gen (SOBBHTDIonFly): Generator from :func:`get_sobbh_tdionfly_gen`.
        t_arr: Absolute data times [s] the channels are evaluated on (``upsample_t_arr``).
        td_settings: :class:`~lisatools.domains.TDSettings` of the data grid.
        target_domain (DomainSettingsBase): Analysis-domain settings to transform onto.
        td_window (xp.ndarray, optional): Time-domain window for the transform.
        runtime_kwargs: Per-call kwargs; call kwargs override them.
        nchannels: Leading TDI channels to keep (``None``: all).
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
        """Combined TD TDI channels on the data grid (before domain projection).

        Args:
            *params: The 11 waveform-basis parameters
                ``(m1, m2, s1, s2, dist, inc, f_low, ra, dec, psi, phi0)``.
            **kwargs: Per-call kwargs (``convert_to_ra_dec`` is dropped).

        Returns:
            np.ndarray: ``(nch, len(t_arr))`` channels (summed over modes).
        """
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs.pop("convert_to_ra_dec", None)
        m1, m2, s1, s2, dist, inc, f_low, ra, dec, psi, phi0 = params
        arr = np.asarray(
            self.wave_gen(
                m1,
                m2,
                s1,
                s2,
                dist,
                f_low,
                phi0,
                inc,
                ra,
                dec,
                psi,
                upsample_t_arr=self.t_arr,
                combine=True,
                **call_kwargs,
            )
        )
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return arr

    def __call__(self, *params, **kwargs):
        """:meth:`raw_td` transformed onto ``target_domain`` (a ``DomainBase``)."""
        arr = self.raw_td(*params, **kwargs)
        return TDSignal(arr, self.td_settings).transform(self.target_domain, window=self.td_window)


def build_sobbh_stock_waveform(
    *,
    Tobs: float,
    dt: float,
    t_start: float,
    td_settings: TDSettings,
    target_domain,
    tdi_config: TDIConfig,
    reference_time: Optional[float] = None,
    orbits: Optional[Orbits] = None,
    n_grid: int = SOBBH_STOCK_N_GRID,
    buffer_time: float = SOBBH_STOCK_BUFFER_TIME,
    force_backend: str = "cpu",
    td_window=None,
    nchannels: Optional[int] = None,
):
    """Build THE stock-aligned SOBBH waveform (all_sources default: TDI-on-the-fly).

    This is the single documented entry point that reproduces the SOBBH waveform wrap
    of the stock ``all_sources`` global fit
    (``SourceSOBBHSettings.use_tdionfly=True``; see
    ``lisatools.globalfit.stock.erebor.source_runtime.get_sobbh_wave_wrap``; with
    ``SOBBH_LIKELIHOOD=lookup`` the run scores with the lookup scorer and keeps this wrap for
    the cross-check): a
    cached :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly` (built via
    :func:`get_sobbh_tdionfly_gen`) wrapped in :class:`SOBBHTDIonFlyWaveWrap`,
    which evaluates on the data grid and projects to ``target_domain``.

    The returned callable consumes the 11-parameter SOBBH **waveform** basis
    ``(m1, m2, s1, s2, dist[Gpc], inc, f_low, ra, dec, psi, phi0)`` and returns
    a :class:`~lisatools.domains.DomainBase` signal.

    :func:`get_sobbh_tdionfly_gen` caches on
    ``(Tobs, dt, t_start, force_backend, reference_time, n_grid, buffer_time,
    id(orbits))``. The stock run builds its wrap through its own copy of that getter
    (``globalfit.stock.erebor.wrappers.get_sobbh_tdionfly_gen``, a separate cache) with the same
    construction, so the same run essentials give the same output, but the generator object
    is shared only among callers of this module.

    Args:
        Tobs: Observation time in seconds.
        dt: Time-domain sample step in seconds.
        t_start: Data-window start (the generator ``t0``).
        td_settings: :class:`~lisatools.domains.TDSettings` of the data grid.
        target_domain: Analysis-domain settings to project onto (FD / WDM / ...).
        tdi_config: :class:`~lisatools.response.tdiconfig.TDIConfig` (the
            all_sources default is 2nd generation).
        reference_time: Epoch ``f_low`` is defined at (``None`` -> the window
            start; ``MOJITO_REFERENCE_TIME`` in mojito mode).
        orbits: Response orbits (``None`` -> analytic default inside the
            generator).
        n_grid: SOBBH TDI-on-the-fly interior grid size (default matches
            ``SourceSOBBHSettings.n_grid``).
        buffer_time: Generator garbage buffer in seconds (default matches
            ``SourceSOBBHSettings.buffer_time``).
        force_backend: Backend selection at construction.
        td_window: Optional time-domain window applied before the transform.
        nchannels: Number of leading TDI channels to keep.

    Returns:
        SOBBHTDIonFlyWaveWrap: Callable on the SOBBH waveform basis, returning a
        domain-projected signal.

    Example:
        .. code-block:: python

            from lisatools.domains import FDSettings, TDSettings
            from lisatools.response.tdiconfig import TDIConfig
            from lisatools.sources.sobbh.response import build_sobbh_stock_waveform

            dt, N = 10.0, 1 << 19
            wave = build_sobbh_stock_waveform(
                Tobs=N * dt, dt=dt, t_start=0.0,
                td_settings=TDSettings(N=N, dt=dt, t0=0.0, force_backend="cpu"),
                target_domain=FDSettings(N=N // 2 + 1, df=1.0 / (N * dt), min_freq=3e-3,
                                         max_freq=4e-2, force_backend="cpu"),
                tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
                force_backend="cpu", nchannels=3,
            )
            # (m1, m2, s1, s2, dist [Gpc], inc, f_low, ra, dec, psi, phi0)
            sig = wave(45.0, 38.0, 0.0, 0.0, 0.1, 0.9, 1.2e-2, 2.0, 0.4, 1.0, 1.0)
    """
    gen = get_sobbh_tdionfly_gen(
        Tobs=Tobs,
        dt=dt,
        t_start=t_start,
        tdi_config=tdi_config,
        reference_time=reference_time,
        orbits=orbits,
        n_grid=n_grid,
        buffer_time=buffer_time,
        force_backend=force_backend,
    )
    n = int(round(Tobs / dt))
    t_arr = np.arange(n) * dt + t_start
    return SOBBHTDIonFlyWaveWrap(
        gen,
        t_arr,
        td_settings,
        target_domain,
        td_window=td_window,
        nchannels=nchannels,
    )
