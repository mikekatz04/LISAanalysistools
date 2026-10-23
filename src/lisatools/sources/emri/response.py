"""Stock EMRI response-wrapper builders + domain-projection adapters (the legacy EMRI path).

Carved out of the global-fit settings files (2026-07-01): the per-source
``get_emri_response_wrapper`` / ``EMRIWaveWrap`` / ``_EMRISpecialFrameWrap``
lived (duplicated) in ``emri_only`` and ``global_fit`` settings. The canonical
version here supports both the simple CPU smoke path (``special_frame=False``)
and the validated SPECIAL-frame mojito path (``special_frame=True``).

Status in the 9mo / 12mo runs (``EMRI_LIKELIHOOD=direct``): standby. The stock getter
``globalfit.stock.erebor.source_runtime.get_emri_wave_wrap`` builds this wrap lazily per device;
it supplies the FEW generator the direct-to-WDM template reuses
(:mod:`lisatools.sources.emri.wdm_direct`), and it scores only the direct move's cross-check
(``EMRI_CHECK_LL``, off in production) and its fallback, plus synthetic injections. The
``EMRI_*_KWARGS`` constants are the FEW construction defaults.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from few.waveform.waveform import GenerateEMRIWaveform

from ...detector import EqualArmlengthOrbits, Orbits
from ...domains import TDSettings, TDSignal
from ...response.directresponse import ResponseWrapper
from ...response.tdiconfig import TDIConfig
from ...utils.constants import YRSID_SI
from ...utils.utility import get_array_module
from ..utils import icrs_to_ecliptic
from .domain import few_domain_guard

# Shared inspiral / sum / mode-selector defaults (mirrors the settings files).
EMRI_INSPIRAL_KWARGS = {
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    "force_backend": "cpu",
}
EMRI_SUM_KWARGS = {"pad_output": True}
# WARNING (traced 2026-09-14): this constructor value is INERT on the run
# path. FEW's ``_generate_waveform`` has its own call-time default
# (``mode_selection_threshold=1e-5``, few/waveform/base.py:143) which it hands
# to the selector unconditionally, and the selector consults the value below
# only when the call-time argument is None (few/utils/modeselector.py:243-244).
# So the effective threshold is whatever the CALL carries -- 1e-5 unless
# something supplies it. To actually change it, set the ``EMRI_EPS`` knob
# (EMRISettings.eps): it rides the per-call kwargs (branch
# ``waveform_kwargs``) and the wave wrap's ``runtime_kwargs``.
EMRI_MODE_SELECTOR_KWARGS = {"mode_selection_threshold": 1e-2}

# Process-wide cache so the injection and template paths share one generator
# (building two ``GenerateEMRIWaveform`` on CPU exhausts RAM).
_EMRI_WAVE_GEN_CACHE: dict = {}


class _EMRISpecialFrameWrap:
    """Force the SPECIAL EMRI frame on every call to the EMRI ResponseWrapper.

    The EMRI sky/spin params are FEW **ecliptic-polar** angles (qS, phiS, qK,
    phiK), so the FEW h+/hx (viewing + polarization) are built in the ecliptic
    frame. The response then runs against ``frame="icrs"`` orbits, so the sky
    must be converted ecliptic -> ICRS inside the wrapper: the underlying
    ResponseWrapper is built with ``is_ecliptic_latitude=False`` (beta = pi/2 - qS
    = ecliptic latitude) and every call passes ``convert_to_ra_dec=True`` (lam,
    beta -> ra, dec). This is the validated SPECIAL setup (ecl-polar sky + RAW
    FILE spin + ICRS orbits) that reproduces the mojito EMRI to mm ~ 4e-5 and
    removes the 1.49x amplitude (which came from ecliptic-converting the spin).
    The per-call deprecation warning the conversion emits is suppressed -- this
    is the intended EMRI path for now. Every other attribute is forwarded to the wrapped
    generator, except ``supports_batch`` (see ``_NO_FORWARD``).

    Args:
        wave_gen (ResponseWrapper): The EMRI response wrapper to call.
    """

    def __init__(self, wave_gen):
        self._wave_gen = wave_gen

    def __call__(self, *params, **kwargs):
        kwargs.setdefault("convert_to_ra_dec", True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return self._wave_gen(*params, **kwargs)

    #: Attributes this wrapper must NEVER forward from the wrapped generator.
    #: ``supports_batch`` is a capability whose entire contract is
    #: default-deny: the batched likelihood path reads it with
    #: ``getattr(gen, "supports_batch", False)``, so blanket forwarding would
    #: let a flag set anywhere below -- on ``ResponseWrapper``, say -- opt
    #: every EMRI generator in silently, past a gate that is supposed to
    #: require a deliberate declaration.
    _NO_FORWARD = frozenset({"supports_batch"})

    def __getattr__(self, name):
        if name in self._NO_FORWARD:
            raise AttributeError(
                f"{type(self).__name__} deliberately does not forward "
                f"{name!r} from the wrapped generator; see _NO_FORWARD."
            )
        return getattr(self._wave_gen, name)


def get_emri_response_wrapper(
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
    t0_shift_to_data: float = 0.0,
    special_frame: bool = True,
    inspiral_kwargs: Optional[dict] = None,
    sum_kwargs: Optional[dict] = None,
    mode_selector_kwargs: Optional[dict] = None,
):
    """Build (and cache) a :class:`ResponseWrapper` around ``GenerateEMRIWaveform``.

    Cached so the injection and template paths share one generator (building two
    on CPU exhausts RAM). ``role`` is intentionally excluded from the cache key.
    The key is ``(Tobs, dt, t_start, tdi_chan, order, force_backend, id(orbits),
    t0_shift_to_data, special_frame)``: ``tdi_config``, ``t_buffer`` and the FEW kwargs are
    not part of it, so a second call that differs only in those returns the first wrapper.

    The FEW model is ``FastKerrEccentricEquatorialFlux`` (``frame="detector"``,
    ``return_list=False``); the response uses ``index_lambda=8`` (``phiS``), ``index_beta=7``
    (``qS``, so ``is_ecliptic_latitude=False``), ``flip_hx=True`` and
    ``remove_garbage="zero"``.

    ``t_start`` is the epoch the intrinsic FEW phases (``Phi_phi0`` etc.) are
    referenced to — for mojito data this is the **catalogue reference time**, NOT
    the (trimmed) data-window start (``GenerateEMRIWaveform`` has no internal
    reference epoch; its ``t0`` IS the phase reference). ``t0_shift_to_data``
    carries the sub-sample (``|.| < dt``) remainder of ``data_t0 - t_start``; the
    integer part is handled by :class:`EMRIWaveWrap` slicing ``offset_int``
    samples off the front.

    Args:
        Tobs: Observation time in seconds (converted to years for the response).
        dt: Sampling cadence in seconds.
        t_start: Phase-reference epoch [s] (the response ``t0``), see above.
        tdi_config: :class:`~lisatools.response.tdiconfig.TDIConfig` (the response ``tdi``).
        tdi_chan: TDI channels. Default ``"XYZ"``.
        role: ``"template"`` or ``"injection"``; informational only (not used, not in the key).
        order: Lagrange interpolation order of the response. Default ``40``.
        t_buffer: Response garbage buffer [s]. Default ``3e4``.
        orbits: :class:`~lisatools.detector.Orbits`; ``None`` builds
            :class:`~lisatools.detector.EqualArmlengthOrbits` on ``force_backend`` (production
            passes the run's ``frame="icrs"`` unequal-arm orbits).
        force_backend: Backend name for FEW and the response. Default ``"cpu"``.
        t0_shift_to_data: Sub-sample alignment [s] onto the data grid. Default ``0.0``.
        special_frame: When ``True`` (default, validated mojito path) the
            returned generator is wrapped in :class:`_EMRISpecialFrameWrap` which
            forces ``convert_to_ra_dec=True`` (ecliptic->ICRS). When ``False``
            (CPU smoke path) the raw :class:`ResponseWrapper` is returned; the
            caller supplies ``convert_to_ra_dec`` per call.
        inspiral_kwargs: FEW ``inspiral_kwargs``; ``None`` uses :data:`EMRI_INSPIRAL_KWARGS`.
        sum_kwargs: FEW ``sum_kwargs``; ``None`` uses :data:`EMRI_SUM_KWARGS`.
        mode_selector_kwargs: FEW ``mode_selector_kwargs``; ``None`` uses
            :data:`EMRI_MODE_SELECTOR_KWARGS` (inert on the run path, see the comment on it).

    Returns:
        ResponseWrapper | _EMRISpecialFrameWrap: The cached generator, called on the
        14-parameter FEW basis and returning the per-channel TDI time series.
    """
    inspiral_kwargs = inspiral_kwargs if inspiral_kwargs is not None else EMRI_INSPIRAL_KWARGS
    sum_kwargs = sum_kwargs if sum_kwargs is not None else EMRI_SUM_KWARGS
    mode_selector_kwargs = (
        mode_selector_kwargs if mode_selector_kwargs is not None else EMRI_MODE_SELECTOR_KWARGS
    )

    key = (
        Tobs,
        dt,
        t_start,
        tdi_chan,
        order,
        force_backend,
        id(orbits),
        t0_shift_to_data,
        special_frame,
    )
    if key in _EMRI_WAVE_GEN_CACHE:
        return _EMRI_WAVE_GEN_CACHE[key]

    few_generator = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        return_list=False,
        inspiral_kwargs=inspiral_kwargs,
        sum_kwargs=sum_kwargs,
        frame="detector",
        mode_selector_kwargs=mode_selector_kwargs,
        force_backend=force_backend,
    )

    response_kwargs = {
        "Tobs": Tobs / YRSID_SI,
        "dt": dt,
        "index_lambda": 8,
        "index_beta": 7,
        "flip_hx": True,
        "force_backend": force_backend,
        "tdi": tdi_config,
        "tdi_chan": tdi_chan,
        "order": order,
        "remove_garbage": "zero",
        "t_buffer": t_buffer,
        # index_beta=7 carries the ecliptic POLAR angle qS, so
        # is_ecliptic_latitude=False maps it to the ecliptic latitude (pi/2 - qS).
        "is_ecliptic_latitude": False,
        # Sub-sample (|.| < dt) alignment of the REF-anchored waveform onto the
        # data grid; the integer-sample part is sliced by EMRIWaveWrap.
        "t0_shift_to_data": t0_shift_to_data,
    }

    if orbits is None:
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
    wave_gen = ResponseWrapper(
        few_generator,
        orbits=orbits,
        t0=t_start,
        **response_kwargs,
    )
    if special_frame:
        wave_gen = _EMRISpecialFrameWrap(wave_gen)

    _EMRI_WAVE_GEN_CACHE[key] = wave_gen
    return wave_gen


# NOTE: ``emri_catalogue_to_waveform_basis`` moved to
# ``lisatools.sources.emri.waveform`` (catalogue -> waveform-parameter glue;
# the intrinsic prograde/retrograde inclination flag is not response/viewing
# geometry). The ``lisatools.sources.emri`` re-export is unchanged.


class EMRIWaveWrap:
    """Run the cached ResponseWrapper and project to the run's domain.

    Output is a :class:`~lisatools.domains.DomainBase` subclass (FDSignal /
    WDMSignal / ...) so ACA dispatch and the EMRI move's ``get_waveform_here``
    land on the right kernels. In the 9mo / 12mo runs this is the standby production wrap
    (cross-check / fallback of the direct-to-WDM template; see the module docstring).

    Args:
        wave_gen: Generator from :func:`get_emri_response_wrapper`.
        td_settings: :class:`~lisatools.domains.TDSettings` of the data grid; the output is
            zero-padded or cut to its ``N``.
        target_domain (DomainSettingsBase): Analysis-domain settings to transform onto.
        td_window (xp.ndarray, optional): Time-domain window for the transform.
        runtime_kwargs: Per-call kwargs for ``wave_gen`` (e.g. the mode-selection
            threshold); call kwargs override them.
        nchannels: Leading TDI channels to keep (``None``: all).
        offset_int: Samples sliced off the front of the response output, the integer part of
            ``(data_t0 - t_start) / dt``. Default ``0``.
    """

    def __init__(
        self,
        wave_gen,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
        offset_int: int = 0,
    ):
        self.wave_gen = wave_gen
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels
        # Integer-sample offset between the REF-anchored response output and the
        # data grid: data_t0 = t_start(=REF) + offset_int*dt + t0_shift_to_data.
        # Slicing off the first ``offset_int`` samples lands the rest on the data
        # grid (t0 = data_t0).
        self.offset_int = int(offset_int)

    def __call__(self, *params, **kwargs):
        """Generate one template on the run domain.

        Args:
            *params: The 14-parameter FEW waveform basis.
            **kwargs: Per-call kwargs for ``wave_gen`` (merged over ``runtime_kwargs``).

        Returns:
            DomainBase: The template on ``target_domain``.

        Raises:
            WaveformDomainError: When FEW refuses the parameters.
        """
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        # Prior boxes leak outside the FEW interpolation grid; convert FEW's
        # bare out-of-domain ValueError/AssertionError into the typed
        # WaveformDomainError the sampling machinery catches (ll -> -1e300).
        with few_domain_guard():
            raw = self.wave_gen(*params, **call_kwargs)
        # ResponseWrapper returns a list of per-channel TDI arrays on the
        # response backend (cupy on GPU); stack on that backend rather than
        # forcing host with np.asarray.
        xp = get_array_module(raw[0] if isinstance(raw, (list, tuple)) else raw)
        arr = xp.atleast_2d(xp.asarray(raw))
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        if self.offset_int:
            arr = arr[:, self.offset_int :]
        N = self.td_settings.N
        if arr.shape[-1] < N:
            arr = xp.pad(arr, ((0, 0), (0, N - arr.shape[-1])))
        elif arr.shape[-1] > N:
            arr = arr[:, :N]
        return TDSignal(arr, self.td_settings).transform(self.target_domain, window=self.td_window)
