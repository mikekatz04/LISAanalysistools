"""Stock EMRI response-wrapper builders + domain-projection adapters.

Carved out of the global-fit settings files (2026-07-01): the per-source
``get_emri_response_wrapper`` / ``EMRIWaveWrap`` / ``_EMRISpecialFrameWrap``
lived (duplicated) in ``emri_only`` and ``global_fit`` settings. The canonical
version here supports both the simple CPU smoke path (``special_frame=False``)
and the validated SPECIAL-frame mojito path (``special_frame=True``).
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

# Shared inspiral / sum / mode-selector defaults (mirrors the settings files).
EMRI_INSPIRAL_KWARGS = {
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    "force_backend": "cpu",
}
EMRI_SUM_KWARGS = {"pad_output": True}
# 1e-2 threshold keeps things fast; tighten once on GPU.
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
    is the intended EMRI path for now.
    """

    def __init__(self, wave_gen):
        self._wave_gen = wave_gen

    def __call__(self, *params, **kwargs):
        kwargs.setdefault("convert_to_ra_dec", True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return self._wave_gen(*params, **kwargs)

    def __getattr__(self, name):
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

    ``t_start`` is the epoch the intrinsic FEW phases (``Phi_phi0`` etc.) are
    referenced to — for mojito data this is the **catalogue reference time**, NOT
    the (trimmed) data-window start (``GenerateEMRIWaveform`` has no internal
    reference epoch; its ``t0`` IS the phase reference). ``t0_shift_to_data``
    carries the sub-sample (``|.| < dt``) remainder of ``data_t0 - t_start``; the
    integer part is handled by :class:`EMRIWaveWrap` slicing ``offset_int``
    samples off the front.

    Args:
        special_frame: When ``True`` (default, validated mojito path) the
            returned generator is wrapped in :class:`_EMRISpecialFrameWrap` which
            forces ``convert_to_ra_dec=True`` (ecliptic->ICRS). When ``False``
            (CPU smoke path) the raw :class:`ResponseWrapper` is returned; the
            caller supplies ``convert_to_ra_dec`` per call.
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


def emri_catalogue_to_waveform_basis(entry: dict) -> np.ndarray:
    """Convert one mojito EMRI catalogue entry to the 14-param FEW waveform basis.

    SPECIAL EMRI frame (validated 2026-06-19,
    ``scripts/sobbh/emri_frame_convert_check.py``): ECLIPTIC-POLAR sky angles
    (``qS = pi/2 - ecliptic latitude``, ``phiS = ecliptic longitude``,
    converted from the catalogue ICRS RA/Dec) together with the RAW catalogue
    spin angles (``qK``/``phiK`` NOT converted — ecliptic-converting the spin
    produced the spurious 1.49x amplitude) and ``xI0 = +1``:
    FastKerrEccentricEquatorialFlux is an equatorial model, and the catalogue
    "InclinationAngle" is the VIEWING inclination, which FEW derives
    internally from the qS/qK sky+spin geometry rather than taking it as an
    intrinsic input.

    Pair with a ``get_emri_response_wrapper(special_frame=True)`` generator so
    the sky is converted back ecliptic -> ICRS for ``frame="icrs"`` orbits.

    The ICRS -> ecliptic conversion is the LISA-DDPC-SEG-TN-007 fixed-obliquity
    rotation (``lisatools.sources.utils.icrs_to_ecliptic``, the mojito
    convention) — the same one the validated full_year settings used — NOT the
    astropy ``barycentrictrueecliptic`` version in
    ``lisatools.response.directresponse`` (they differ by ~1e-4 rad).

    Args:
        entry: One source's catalogue dict (mojito L1 ``catalogue["EMRI"][i]``).

    Returns:
        ``(14,)`` array ``[M, mu, a, p0, e0, xI0, dist (Gpc), qS, phiS, qK,
        phiK, Phi_phi0, Phi_theta0, Phi_r0]``.
    """
    ra = float(entry["RightAscension"]) % (2 * np.pi)
    dec = float(entry["Declination"])
    lam_S, beta_S = icrs_to_ecliptic(ra, dec)
    return np.array(
        [
            entry["PrimaryMassSSBFrame"],  # M
            entry["SecondaryMassSSBFrame"],  # mu
            entry["PrimarySpinParameter"],  # a
            entry["SemiLatusRectum"],  # p0
            entry["Eccentricity"],  # e0
            1.0,  # xI0 (equatorial prograde)
            entry["LuminosityDistance"] / 1e3,  # dist (Mpc -> Gpc)
            float(np.pi / 2 - beta_S),  # qS (ecliptic polar)
            float(lam_S) % (2 * np.pi),  # phiS (ecliptic longitude)
            entry["PolarAnglePrimarySpin"],  # qK (RAW file spin)
            entry["AzimuthalAnglePrimarySpin"],  # phiK (RAW file spin)
            entry["AzimuthalPhase"],  # Phi_phi0
            entry["PolarPhase"],  # Phi_theta0
            entry["RadialPhase"],  # Phi_r0
        ]
    )


class EMRIWaveWrap:
    """Run the cached ResponseWrapper and project to the run's domain.

    Output is a :class:`~lisatools.domains.DomainBase` subclass (FDSignal /
    WDMSignal / ...) so ACA dispatch and the EMRI move's ``get_waveform_here``
    land on the right kernels.
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
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
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
