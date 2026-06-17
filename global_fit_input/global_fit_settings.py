"""Top-level global-fit settings (GB + foreground + PSD + MBH + EMRI + SOBBH).

Edit this file directly. The *backend* block below picks the array
module + GPU backend; ``DOMAIN_CHOICE`` (further down) picks the basis
grid as a :class:`DomainSettingsBase` / factory the engine consumes.

Built on the same pieces the smoke-test settings files use:

* ``recipe_steps.build_psd_moves`` / ``build_gb_moves`` drive the PSD
  and GB branches (mirrors ``gb_and_foreground_global_fit_settings.py``).
* The EMRI branch reuses the cached ``GenerateEMRIWaveform`` +
  ``ResponseWrapper`` pattern from ``emri_only_global_fit_settings.py``
  via :class:`EMRIWaveWrap`, with one waveform held in RAM at a time.
* The SOBBH branch mirrors EMRI with :class:`SOBBHWaveform` (3.5PN,
  aligned spins) + ``ResponseWrapper`` via :class:`SOBBHWaveWrap`.
* MBH is wired through :class:`MBHSpecialMove` with ``BBHWaveformFD``.
"""

import gc
import logging
import os
import shutil
from copy import deepcopy
from typing import Optional

import h5py
import numpy as np


# ============================================================
# *** Backend selection ***
# ============================================================
try:
    import cupy as cp

    GPU_BACKEND = "cuda13x"  # change to "cuda11x" / "cuda12x" if needed
    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as cp

    GPU_BACKEND = "cpu"
    gpu_available = False
# ============================================================

logger = logging.getLogger(__name__)


from eryn.moves import StretchMove
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from few.waveform import GenerateEMRIWaveform

from gbgpu.utils.utility import get_fdot

from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.domains import (
    DomainBaseArray,
    TDSettings,
    TDSignal,
    WDMSettings,
)
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
)
from lisatools.globalfit.moves import (
    GFCombineMove,
    MBHSpecialMove,
    ResidualAddOneRemoveOneMove,
)
from lisatools.globalfit.preprocessing import BaseProcessingStep, SangriaProcessingStep
from lisatools.globalfit.recipe import Recipe
from lisatools.globalfit.recipe_steps import (
    PERecipeStep,
    SearchRecipeStep,
    build_gb_moves,
    build_psd_moves,
)
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    EMRISettings,
    EMRISetup,
    GalForSettings,
    GalForSetup,
    GBSettings,
    GBSetup,
    MBHSettings,
    MBHSetup,
    PSDSettings,
    PSDSetup,
    SOBBHSettings,
    SOBBHSetup,
    make_emri_transform_container,
    make_sobbh_transform_container,
)
from lisatools.sources.sobbh import SOBBHWaveform
from lisatools.utils.constants import YRSID_SI


# ============================================================
# *** Domain selection ***
# ============================================================
#
# ``DOMAIN_CHOICE`` is the value the engine consumes via
# ``GeneralSettings.domain_settings`` — either a fully constructed
# :class:`DomainSettingsBase` or a factory
# ``(times, dt, force_backend) -> DomainSettingsBase``. Each Settings
# class exposes ``make_factory(...)`` for building the deferred form.
#
# Smoke-friendly default: 3-month Tobs on Sangria's dt = 5 s gives
# N = 1,555,200 samples. With ~1-hour wavelets (layer_dt = Nf*dt = 3600 s),
# Nf = 720 and Nt = 2160 (both even). Crop 20 wavelets (~20 h) from each
# time edge so boundary wavelets don't show up as NaN in the active band.
NF = 720
NT = 2160
DT = 5.0
TOBS = NF * NT * DT  # 90 d ≈ 3 mo
T_START = 0.0

DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=1e-4,
    max_freq=2.5e-2,
    min_time=20 * 3600.0,
    max_time=(NT - 20) * 3600.0,
)
# WDM lookup table removed sprint-wide -- the chunked-heterodyne
# template pipeline (gb_wdm_het.GBWDMHeterodyne) is now the only WDM
# backend. The build is wired in ``setup_recipe`` below; no lookup
# table is needed (or supported).
# Alternates (import FDSettings / STFTSettings from lisatools.domains first):
# DOMAIN_CHOICE = FDSettings.make_factory(min_freq=5e-5, max_freq=3e-2)
# DOMAIN_CHOICE = STFTSettings.make_factory(big_dt=24 * 3600.0, min_freq=5e-5, max_freq=3e-2)
# ============================================================


# ============================================================
# *** EMRI injection (synthetic, in-process) ***
# ============================================================
# FEW's ``FastKerrEccentricEquatorialFlux`` consumes 14 params in this order:
# M, mu, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0.
# EMRISetup's transform fills xI0 (idx 5) and Phi_theta0 (idx 12) so the
# sampling basis has 12 params with ``M -> logM, qS -> cos qS, qK -> cos qK``.
INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        1.0e6,      # M
        1.0e1,      # mu
        0.5,        # a
        10.0,       # p0
        0.3,        # e0
        1.0,        # xI0 — fill_value[0]
        1.0,        # dist (Gpc)
        np.pi / 3,  # qS
        1.0,        # phiS
        np.pi / 4,  # qK
        2.0,        # phiK
        0.0,        # Phi_phi0
        0.0,        # Phi_theta0 — fill_value[1]
        0.0,        # Phi_r0
    ]
)
SAMPLE_FILL_INDICES = [5, 12]


def emri_full_to_sampling(params_full):
    """Convert a 14-param waveform-basis vector to the 12-param sampling basis."""
    p = np.asarray(params_full, dtype=float)
    transform = make_emri_transform_container(p[SAMPLE_FILL_INDICES])
    return transform.both_inverse_transforms(p)


# Shared inspiral / sum / mode-selector kwargs (mirrors emri_only).
EMRI_INSPIRAL_KWARGS = {
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    "force_backend": "cpu",
}
EMRI_SUM_KWARGS = {"pad_output": True}
# 1e-2 threshold keeps things fast; tighten once on GPU.
EMRI_MODE_SELECTOR_KWARGS = {"mode_selection_threshold": 1e-2}

_EMRI_WAVE_GEN_CACHE = {}


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
):
    """Build (and cache) a :class:`ResponseWrapper` around ``GenerateEMRIWaveform``.

    Cached so the injection path and template path share one generator —
    building two on CPU exhausts RAM.
    """
    key = (Tobs, dt, t_start, tdi_chan, order, force_backend, id(orbits))
    if key in _EMRI_WAVE_GEN_CACHE:
        return _EMRI_WAVE_GEN_CACHE[key]

    few_generator = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        return_list=False,
        inspiral_kwargs=EMRI_INSPIRAL_KWARGS,
        sum_kwargs=EMRI_SUM_KWARGS,
        frame="detector",
        mode_selector_kwargs=EMRI_MODE_SELECTOR_KWARGS,
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
    }

    if orbits is None:
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
    wave_gen = ResponseWrapper(
        few_generator,
        orbits=orbits,
        t0=t_start,
        **response_kwargs,
    )
    _EMRI_WAVE_GEN_CACHE[key] = wave_gen
    return wave_gen


class EMRIWaveWrap:
    """Run the cached ResponseWrapper and project to the run's domain.

    Output is a :class:`DomainBase` subclass (FDSignal / WDMSignal / ...)
    so ACA dispatch and the EMRI move's ``get_waveform_here`` land on the
    right kernels.
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
# *** SOBBH injection (synthetic, in-process) ***
# ============================================================
# Output-basis order consumed by :class:`SOBBHWaveform.__call__` (11 params):
#   m1 (M_sun), m2 (M_sun), s1, s2, dist (Gpc), inc (rad), f_low (Hz),
#   lam (ecliptic longitude, rad), beta (ecliptic latitude, rad),
#   psi (polarization, rad), phi0 (coalescence-phase offset, rad).
#
# Sampling basis (consumed by ``SOBBHSetup.transform``, 11 params):
#   logm1, logm2, s1, s2, dist, cosinc, f_low, phiS, cosqS, psi, phi0
# with cosqS -> beta = pi/2 - arccos(cosqS) and phiS -> lam.
SOBBH_INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        40.0,        # m1 (M_sun)
        30.0,        # m2 (M_sun)
        0.3,         # s1
        -0.2,        # s2
        1.0,         # dist (Gpc)
        np.pi / 3,   # inc
        1.5e-2,      # f_low (Hz)
        1.0,         # lam   (phiS in sampling basis)
        np.pi / 4,   # beta  (cosqS = cos(pi/2 - beta) in sampling basis)
        0.3,         # psi
        0.0,         # phi0
    ]
)


def sobbh_full_to_sampling(params_full):
    """Convert an 11-param SOBBH waveform-basis vector to the 11-param sampling basis."""
    transform = make_sobbh_transform_container()
    return transform.both_inverse_transforms(np.asarray(params_full, dtype=float))


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


# ============================================================
# *** Synthetic in-process data processors ***
# ============================================================
#
# Each one matches the SangriaProcessingStep / BaseProcessingStep
# interface: it generates the source's signal in-process and hands
# ``(times, data, fs)`` to :class:`BaseProcessingStep`. The engine's
# downstream code (window / domain transform / sensitivity) doesn't
# know or care that the data is synthetic.
#
# To inject multiple branches simultaneously, see
# :class:`SyntheticCombinedProcessingStep` below — it forwards the call
# to a list of single-branch processors and adds their TD outputs.


class SyntheticEMRIProcessingStep(BaseProcessingStep):
    """Generate the EMRI injection in-process via the shared ResponseWrapper.

    Hands ``(times, data, fs)`` to :class:`BaseProcessingStep` — same
    interface as :class:`SangriaProcessingStep`. No external h5 path
    needed.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        injection_params_full_basis: np.ndarray,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
        wave_gen = get_emri_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            force_backend=force_backend,
        )

        td_signal = np.asarray(
            wave_gen(*injection_params_full_basis)
        )
        td_signal = np.atleast_2d(td_signal)[:nchannels]

        # ``ResponseWrapper`` can produce a couple fewer samples than
        # ``Tobs/dt`` (Lagrange buffer / remove_garbage). Pad/clip to
        # exactly ``Tobs/dt`` so downstream WDM ``N = Nf*Nt`` matches.
        target_N = int(round(Tobs / dt))
        if td_signal.shape[-1] < target_N:
            pad = target_N - td_signal.shape[-1]
            td_signal = np.pad(td_signal, ((0, 0), (0, pad)), mode="constant")
        elif td_signal.shape[-1] > target_N:
            td_signal = td_signal[:, :target_N]

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params_full_basis = injection_params_full_basis
        self.tdi_chan = tdi_chan


class SyntheticSOBBHProcessingStep(BaseProcessingStep):
    """Generate the SOBBH injection in-process via the shared ResponseWrapper.

    Mirrors :class:`SyntheticEMRIProcessingStep` — pads/clips to exactly
    ``Tobs/dt`` so downstream WDM ``N = Nf*Nt`` matches.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        injection_params_full_basis: np.ndarray,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
        wave_gen = get_sobbh_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            force_backend=force_backend,
        )

        td_signal = np.asarray(
            wave_gen(*injection_params_full_basis)
        )
        td_signal = np.atleast_2d(td_signal)[:nchannels]

        target_N = int(round(Tobs / dt))
        if td_signal.shape[-1] < target_N:
            pad = target_N - td_signal.shape[-1]
            td_signal = np.pad(td_signal, ((0, 0), (0, pad)), mode="constant")
        elif td_signal.shape[-1] > target_N:
            td_signal = td_signal[:, :target_N]

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params_full_basis = injection_params_full_basis
        self.tdi_chan = tdi_chan


# ----------------------------------------------------------------------
# Synthetic GB injection (matches the SyntheticEMRIProcessingStep pattern).
# ----------------------------------------------------------------------
#
# Param layout for ``GBGPU.generate_global_template``:
#   shape ``(num_sources, 9)``: [A, f0, fdot, fddot, phi0, iota, psi, lam, beta]
#
# Two example sources in band — mid-mHz, modest fdot. Edit / extend.
GB_INJECTION_PARAMS = np.array(
    [
        [1.0e-22, 2.0e-3,  1.0e-16, 0.0, 0.0, 0.6, 0.4, 1.5, 0.3],
        [3.0e-23, 8.5e-3,  5.0e-15, 0.0, 1.2, 0.9, 1.0, 4.0, -0.6],
    ]
)


class SyntheticGBProcessingStep(BaseProcessingStep):
    """In-process synthetic GB injection.

    Mirrors :class:`SyntheticEMRIProcessingStep`: fills a FD buffer with
    :meth:`GBGPU.generate_global_template`, IRFFTs back to TD, and hands
    ``(times, data, fs)`` to :class:`BaseProcessingStep`.

    Args:
        Tobs: Observation time in seconds.
        dt: Time-domain sample step in seconds.
        t_start: Initial time of the first sample.
        injection_params: ``(num_sources, 9)`` array of GB parameters in
            the order expected by ``GBGPU.generate_global_template``:
            ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``.
        tdi_chan: ``"XYZ"`` (3 channels) or ``"AE"`` (2 channels).
        nchannels: Number of leading channels to keep.
        use_tdi2: Use TDI 2nd generation kernel. Defaults to ``True``.
        oversample: GB-internal oversampling factor (1 is plenty for an
            injection).
        force_backend: ``"cpu"`` / ``"cuda12x"`` etc. — forwarded to GBGPU.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        injection_params: np.ndarray = GB_INJECTION_PARAMS,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        use_tdi2: bool = True,
        oversample: int = 1,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        from gbgpu.gbgpu import GBGPU

        params = np.atleast_2d(injection_params).astype(np.float64)
        assert params.shape[1] == 9, (
            f"injection_params must have shape (num_sources, 9); got {params.shape}"
        )
        num_sources = params.shape[0]

        nchannels_full = 3 if tdi_chan == "XYZ" else 2
        if nchannels > nchannels_full:
            raise ValueError(
                f"nchannels={nchannels} exceeds tdi_chan={tdi_chan}'s {nchannels_full} channels"
            )

        target_N = int(round(Tobs / dt))
        data_length = target_N // 2 + 1

        gb = GBGPU(force_backend=force_backend)
        # ``GBGPU.generate_global_template`` reads ``self.gpus`` directly,
        # which is only set elsewhere (e.g. in the move setup). For a
        # standalone injection we set it explicitly.
        gb.gpus = None if force_backend == "cpu" else [0]
        xp = gb.xp

        # NOTE: ``generate_global_template`` internally ``.flatten()``s any
        # 2D/3D buffer it receives, which returns a copy — the kernel
        # writes never propagate back to our caller. The ndim==1 path does
        # not copy, so we pass a 1D buffer of length
        # ``num_templates * nchannels_full * data_length`` and reshape after.
        template_flat = xp.zeros(
            nchannels_full * data_length, dtype=xp.complex128
        )
        group_index = xp.zeros(num_sources, dtype=xp.int32)

        gb.generate_global_template(
            params,
            group_index,
            template_flat,
            start_freq_ind=0,
            T=Tobs,
            dt=dt,
            tdi_channel_setup=tdi_chan,
            tdi2=use_tdi2,
            oversample=oversample,
            data_length=data_length,
        )

        # FD -> TD. GBGPU stores tilde-h(f) in the LDC convention
        # (FD = dt * rfft(td)), so the inverse is td = irfft(FD)/dt.
        fd_signal = template_flat.reshape(nchannels_full, data_length)
        if hasattr(fd_signal, "get"):
            fd_signal = fd_signal.get()
        td_signal = np.fft.irfft(fd_signal, n=target_N, axis=-1) / dt
        td_signal = td_signal[:nchannels]

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params = params
        self.tdi_chan = tdi_chan


class SyntheticCombinedProcessingStep(BaseProcessingStep):
    """Combine several synthetic processors into one residual.

    Each entry in ``processor_specs`` is a ``(class, init_kwargs)`` pair;
    each child generates its own TD signal at the same ``(Tobs, dt, t_start)``
    grid, and the combined output is the sum of all child TDs.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        processor_specs: list,  # list[(ProcessorClass, kwargs)]
        nchannels: int = 3,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        if not processor_specs:
            raise ValueError("processor_specs must contain at least one (class, kwargs) entry")

        target_N = int(round(Tobs / dt))
        combined = np.zeros((nchannels, target_N), dtype=np.float64)
        for cls, kwargs in processor_specs:
            child_kwargs = dict(kwargs)
            child_kwargs.setdefault("Tobs", Tobs)
            child_kwargs.setdefault("dt", dt)
            child_kwargs.setdefault("t_start", t_start)
            child_kwargs.setdefault("nchannels", nchannels)
            child = cls(**child_kwargs, verbose=False, do_plots=False)
            combined[: child.data.shape[0]] += child.data[:, :target_N]

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None


################
#  RECIPE
################


def setup_recipe(
    recipe: Recipe,
    engine_info,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    # Build the WDM-domain GB likelihood here (after the deepcopy in
    # ``CurrentInfoGlobalFit.__init__``) — the underlying C++ orbits wrap
    # is not picklable, so it must live outside the settings dataclass.
    # Chunked-heterodyne is the only WDM backend (legacy lookup-table
    # path removed sprint-wide); see ``gb_and_foreground_global_fit_settings.py``
    # for the same wiring rationale.
    gb_info = curr.source_info["gb"]
    if (
        isinstance(general_info.domain_settings, WDMSettings)
        and gb_info.gb_wdm_comp is None
    ):
        import sys
        if "/Users/mlkatz/Research/sprint_2026" not in sys.path:
            sys.path.insert(0, "/Users/mlkatz/Research/sprint_2026")
        from gb_wdm_het import GBWDMHeterodyne

        _wdm = general_info.domain_settings
        _t_obs_start = float(getattr(general_info, "t_obs_start", 0.0))
        gb_info.gb_wdm_comp = GBWDMHeterodyne(
            Nf=_wdm.Nf, Nt=_wdm.Nt, dt=general_info.dt,
            T_full=general_info.Tobs, t_ref_full=gb_info.t0,
            Nt_sub=int(os.environ.get("CHUNKED_NT_SUB", 256)),
            n_pad=int(os.environ.get("CHUNKED_N_PAD", 32)),
            N_sparse=int(os.environ.get("CHUNKED_N_SPARSE", 256)),
            nchannels=3,
            force_backend=general_info.force_backend,
            tdi_gen="2nd generation" if gb_info.use_tdi2 else "1st generation",
            orbits=general_info.gpu_orbits,
            t_obs_start=_t_obs_start,
            N_cp_sig=int(os.environ.get("CHUNKED_N_CP_SIG", 48)),
            N_cp_orbit=int(os.environ.get("CHUNKED_N_CP_ORBIT", 32)),
        )

    #* ================================= PSD =================================
    num_repeats_psd = 5 if not gpu_available else 60
    psd_search_move, psd_pe_move = build_psd_moves(
        engine_info, curr, acs, priors, num_repeats=num_repeats_psd,
    )

    #* ================================= GB =================================
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )
    # Smoke-friendly: keep only the prior-based RJ proposal in PE mode
    # until search/refit/fstat are validated for this config.
    gb_pe_moves = [m for m in gb_pe_moves if "prior" in m.name]

    #* ================================= MBH =================================
    mbh_pe_move = None
    if "mbh" in curr.source_info:
        mbh_pe_move = _build_mbh_move(curr, acs, priors, state)

    #* ================================= EMRI =================================
    emri_pe_move = None
    if "emri" in curr.source_info:
        emri_pe_move = _build_emri_move(curr, acs, priors, state)

    #* ================================= SOBBH =================================
    sobbh_pe_move = None
    if "sobbh" in curr.source_info:
        sobbh_pe_move = _build_sobbh_move(curr, acs, priors, state)

    #* ================================= COMBINE =================================
    pe_moves = [psd_pe_move]
    if mbh_pe_move is not None:
        pe_moves.append(mbh_pe_move)
    if emri_pe_move is not None:
        pe_moves.append(emri_pe_move)
    if sobbh_pe_move is not None:
        pe_moves.append(sobbh_pe_move)

    pe_moves +=  gb_pe_moves
    
    gf_pe_move = GFCombineMove(
        moves=pe_moves, verbose=True, share_temperature_control=False
    )
    gf_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(
        PERecipeStep(moves=[gf_pe_move]), name="full pe"
    )


def _build_mbh_move(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """MBH move using ``BBHWaveformFD`` + ``MBHSpecialMove``.

    Pre-subtracts injected starting waveforms from each cold-chain
    walker's residual, then returns the configured ``MBHSpecialMove``
    (PE only — no inner search loop).
    """
    from bbhx.waveformbuild import BBHWaveformFD

    general_info = curr.general_info
    mbh_info = curr.source_info["mbh"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    wave_gen = BBHWaveformFD(**mbh_info.initialize_kwargs)

    if np.any(mbh_inds := state.branches_inds["mbh"][0]):
        for leaf in range(mbh_inds.shape[-1]):
            if not mbh_inds[0, leaf]:
                continue
            assert np.all(mbh_inds[:, leaf])
            inj_coords = state.branches_coords["mbh"][0, :, leaf]
            inj_coords_in = mbh_info.transform.both_transforms(inj_coords)
            AET = wave_gen(
                *inj_coords_in.T,
                fill=True,
                freqs=cp.asarray(acs.f_arr),
                **mbh_info.waveform_kwargs,
            )
            acs.add_signal_to_residual(AET[:, :2])

    betas_all = np.tile(
        make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1)
    )
    state.sub_states["mbh"].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)
    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)

    mbh_pe_move = MBHSpecialMove(
        "mbh",
        coords_shape,
        wave_gen,
        tempering_kwargs,
        mbh_info.waveform_kwargs.copy(),
        mbh_info.waveform_kwargs.copy(),
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        mbh_info.inner_moves,
        acs.df,
        name="mbh_pe",
        run_search=False,
        force_backend=general_info.force_backend,
    )
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers))
    return mbh_pe_move


def _build_emri_move(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """EMRI move with cached generator + per-walker pre-injection.

    Mirrors the pattern from ``emri_only_global_fit_settings.py`` — one
    waveform held in RAM at a time during pre-injection so peak memory
    stays at a single template's worth.
    """
    general_info = curr.general_info
    emri_info = curr.source_info["emri"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    force_backend = "cpu" if not gpu_available else GPU_BACKEND
    tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
    template_wave_gen = get_emri_response_wrapper(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=T_START,
        tdi_config=tdi_config,
        tdi_chan="XYZ",
        role="template",
        force_backend=force_backend,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend=force_backend,
    )
    wave_gen = EMRIWaveWrap(
        template_wave_gen,
        td_settings,
        general_info.domain_settings,
        td_window=None,
        nchannels=acs.nchannels,
    )

    # Per-walker pre-injection — keeps peak RAM at one template's worth.
    if np.any(emri_inds := state.branches_inds["emri"][0]):
        for leaf in range(emri_inds.shape[-1]):
            if not emri_inds[0, leaf]:
                continue
            assert np.all(emri_inds[:, leaf])
            inj_coords = state.branches_coords["emri"][0, :, leaf]
            inj_coords_in = emri_info.transform.both_transforms(inj_coords)
            for i in range(inj_coords.shape[0]):
                sig = wave_gen(*inj_coords_in[i], **emri_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()

    betas_all = np.tile(
        make_ladder(emri_info.ndim, ntemps=ntemps), (emri_info.nleaves_max, 1)
    )
    state.sub_states["emri"].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, emri_info.nleaves_max, emri_info.ndim)
    emri_pe_move = ResidualAddOneRemoveOneMove(
        "emri",
        coords_shape,
        wave_gen,
        emri_info.waveform_kwargs.copy(),
        emri_info.waveform_kwargs.copy(),
        acs,
        emri_info.num_prop_repeats,
        emri_info.transform,
        priors,
        emri_info.inner_moves,
        Tmax=np.inf,
        betas_all=betas_all,
    )
    emri_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    return emri_pe_move


def _build_sobbh_move(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """SOBBH move with cached generator + per-walker pre-injection.

    Mirrors :func:`_build_emri_move`. One waveform held in RAM at a time
    during pre-injection so peak memory stays at a single template's worth.
    """
    general_info = curr.general_info
    sobbh_info = curr.source_info["sobbh"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    force_backend = "cpu" if not gpu_available else GPU_BACKEND
    tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
    template_wave_gen = get_sobbh_response_wrapper(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=T_START,
        tdi_config=tdi_config,
        tdi_chan="XYZ",
        role="template",
        force_backend=force_backend,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend=force_backend,
    )
    wave_gen = SOBBHWaveWrap(
        template_wave_gen,
        td_settings,
        general_info.domain_settings,
        td_window=None,
        nchannels=acs.nchannels,
    )

    # Per-walker pre-injection — keeps peak RAM at one template's worth.
    if np.any(sobbh_inds := state.branches_inds["sobbh"][0]):
        for leaf in range(sobbh_inds.shape[-1]):
            if not sobbh_inds[0, leaf]:
                continue
            assert np.all(sobbh_inds[:, leaf])
            inj_coords = state.branches_coords["sobbh"][0, :, leaf]
            inj_coords_in = sobbh_info.transform.both_transforms(inj_coords)
            for i in range(inj_coords.shape[0]):
                sig = wave_gen(*inj_coords_in[i], **sobbh_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()

    betas_all = np.tile(
        make_ladder(sobbh_info.ndim, ntemps=ntemps), (sobbh_info.nleaves_max, 1)
    )
    state.sub_states["sobbh"].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, sobbh_info.nleaves_max, sobbh_info.ndim)
    sobbh_pe_move = ResidualAddOneRemoveOneMove(
        "sobbh",
        coords_shape,
        wave_gen,
        sobbh_info.waveform_kwargs.copy(),
        sobbh_info.waveform_kwargs.copy(),
        acs,
        sobbh_info.num_prop_repeats,
        sobbh_info.transform,
        priors,
        sobbh_info.inner_moves,
        Tmax=np.inf,
        betas_all=betas_all,
    )
    sobbh_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    return sobbh_pe_move


#######################
##### SETTINGS ###########
###############


def get_gb_erebor_settings(general_set: GeneralSetup) -> GBSetup:
    delta_safe = 1e-5

    A_lims = [7e-26, 1e-19]
    f0_lims = [0.05e-3, 2.5e-2]

    m_chirp_lims = [0.001, 1.0]
    fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])

    fdot_lims = [-fdot_max_val, fdot_max_val]
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    lam_lims = [0.0, 2 * np.pi]
    beta_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    domain_settings = general_set.domain_settings
    if hasattr(domain_settings, "min_freq") and domain_settings.min_freq is not None:
        start_freq = float(domain_settings.min_freq)
    else:
        start_freq = 5e-5
    if hasattr(domain_settings, "max_freq") and domain_settings.max_freq is not None:
        end_freq = float(domain_settings.max_freq)
    else:
        end_freq = 3e-2

    oversample = 4
    extra_buffer = 5
    start_freq_ind = 0
    # TODO obtain ``t0_gbs`` properly from orbits.
    t0_gbs = 97729089.327664
    initialize_kwargs = dict(force_backend=general_set.force_backend)

    # Built lazily in setup_recipe — the underlying C++ orbits wrap is
    # not picklable, so it can't live on the settings dataclass.
    gb_wdm_comp = None

    ntemps_gb = general_set.ntemps
    gb_betas = 1.0 / 1.2 ** np.arange(ntemps_gb)
    gb_betas[-1] = 1e-4

    gb_settings = GBSettings(
        A_lims=A_lims,
        f0_lims=f0_lims,
        m_chirp_lims=m_chirp_lims,
        fdot_lims=fdot_lims,
        phi0_lims=phi0_lims,
        iota_lims=iota_lims,
        psi_lims=psi_lims,
        lam_lims=lam_lims,
        beta_lims=beta_lims,
        start_freq=start_freq,
        end_freq=end_freq,
        oversample=oversample,
        extra_buffer=extra_buffer,
        start_freq_ind=start_freq_ind,
        t0=t0_gbs,
        tdi_setup="XYZ",
        use_tdi2=True,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs,
        domain_settings=domain_settings,
        gb_wdm_comp=gb_wdm_comp,
        betas=gb_betas,
        nleaves_max=100,
        nleaves_min=0,
        ndim=8,
        log_dir=general_set.file_store_dir,
    )

    return GBSetup(gb_settings)


def get_mbh_erebor_settings(general_set: GeneralSetup) -> MBHSetup:
    gpu_orbits = EqualArmlengthOrbits(force_backend=general_set.force_backend)
    initialize_kwargs_mbh = dict(
        amp_phase_kwargs=dict(run_phenomd=True),
        response_kwargs=dict(TDItag="AET", orbits=gpu_orbits),
        force_backend=general_set.force_backend,
    )

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_mbh,
        nleaves_max=15,
        nleaves_min=0,
        ndim=11,
    )

    return MBHSetup(mbh_settings)


def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:
    initialize_kwargs_psd = dict()

    priors_psd = {
        r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),
        r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),
    }
    priors = {"psd": ProbDistContainer(priors_psd)}
    injection = np.array([15e-12, 3e-15])

    psd_settings = PSDSettings(
        ndim=2,
        injection=injection,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        priors=priors,
        log_dir=general_set.file_store_dir,
    )

    return PSDSetup(psd_settings)


def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:
    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        log_dir=general_set.file_store_dir,
    )

    return GalForSetup(galfor_settings)


def get_emri_erebor_settings(general_set: GeneralSetup) -> EMRISetup:
    """Build the EMRI :class:`EMRISetup`.

    ``initialize_kwargs`` is consumed by :class:`EMRISetup` only to track
    metadata for the run; the active waveform path is the cached
    generator in :func:`_build_emri_move`.
    """
    initialize_kwargs_emri = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs=dict(force_backend="cpu" if not gpu_available else GPU_BACKEND),
        response_kwargs=dict(
            t0=T_START,
            order=40,
            tdi="2nd generation",
            tdi_chan="XYZ",
            force_backend="cpu" if not gpu_available else GPU_BACKEND,
            remove_garbage="zero",
        ),
    )

    waveform_kwargs_pe = dict()

    delta_prior = 1e-2
    injection_sampling = emri_full_to_sampling(INJECTION_PARAMS_FULL_BASIS)

    logm1_lims = [
        (1 - delta_prior) * injection_sampling[0],
        (1 + delta_prior) * injection_sampling[0],
    ]
    m2_lims = [
        (1 - delta_prior) * injection_sampling[1],
        (1 + delta_prior) * injection_sampling[1],
    ]
    amax = min(0.999, (1 + delta_prior) * injection_sampling[2])
    a_lims = [(1 - delta_prior) * injection_sampling[2], amax]
    p0_lims = [
        (1 - delta_prior) * injection_sampling[3],
        (1 + delta_prior) * injection_sampling[3],
    ]
    e0_lims = [
        (1 - delta_prior) * injection_sampling[4],
        (1 + delta_prior) * injection_sampling[4],
    ]

    inner_moves = [(StretchMove(), 1.0)]
    fill_values = np.array(
        [
            INJECTION_PARAMS_FULL_BASIS[5],   # xI0
            INJECTION_PARAMS_FULL_BASIS[12],  # Phi_theta0
        ]
    )

    emri_settings = EMRISettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        fill_values=fill_values,
        logm1_lims=logm1_lims,
        m2_lims=m2_lims,
        a_lims=a_lims,
        p0_lims=p0_lims,
        e0_lims=e0_lims,
        injection=injection_sampling,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_emri,
        waveform_kwargs=waveform_kwargs_pe,
        info_matrix_gen=None,
        inner_moves=inner_moves,
        nleaves_max=1,
        nleaves_min=1,
        ndim=12,
    )

    return EMRISetup(emri_settings)


def get_sobbh_erebor_settings(general_set: GeneralSetup) -> SOBBHSetup:
    """Build the SOBBH :class:`SOBBHSetup`.

    ``initialize_kwargs`` is consumed by :class:`SOBBHSetup` only to track
    metadata for the run; the active waveform path is the cached
    generator in :func:`_build_sobbh_move`.
    """
    force_backend = "cpu" if not gpu_available else GPU_BACKEND
    initialize_kwargs_sobbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        sobbh_waveform_args=("SOBBHWaveform",),
        sobbh_waveform_kwargs=dict(force_backend=force_backend),
        response_kwargs=dict(
            t0=T_START,
            order=40,
            tdi="2nd generation",
            tdi_chan="XYZ",
            force_backend=force_backend,
            remove_garbage="zero",
        ),
    )

    waveform_kwargs_pe = dict()

    delta_prior = 1e-2
    injection_sampling = sobbh_full_to_sampling(SOBBH_INJECTION_PARAMS_FULL_BASIS)

    logm1_lims = [
        (1 - delta_prior) * injection_sampling[0],
        (1 + delta_prior) * injection_sampling[0],
    ]
    logm2_lims = [
        (1 - delta_prior) * injection_sampling[1],
        (1 + delta_prior) * injection_sampling[1],
    ]
    s1_inj = injection_sampling[2]
    s1_lims = [max(-0.99, s1_inj - delta_prior), min(0.99, s1_inj + delta_prior)]
    s2_inj = injection_sampling[3]
    s2_lims = [max(-0.99, s2_inj - delta_prior), min(0.99, s2_inj + delta_prior)]
    f_low_inj = injection_sampling[6]
    f_low_lims = [
        (1 - delta_prior) * f_low_inj,
        (1 + delta_prior) * f_low_inj,
    ]

    inner_moves = [(StretchMove(), 1.0)]
    fill_values = np.array([])  # no non-sampled SOBBH params at default

    sobbh_settings = SOBBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        fill_values=fill_values,
        logm1_lims=logm1_lims,
        logm2_lims=logm2_lims,
        s1_lims=s1_lims,
        s2_lims=s2_lims,
        f_low_lims=f_low_lims,
        injection=injection_sampling,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_sobbh,
        waveform_kwargs=waveform_kwargs_pe,
        info_matrix_gen=None,
        inner_moves=inner_moves,
        nleaves_max=1,
        nleaves_min=1,
        ndim=11,
    )

    return SOBBHSetup(sobbh_settings)


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    ldc_source_file = "/Users/mkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5"
    base_file_name = "global_fit_smoke_test"
    file_store_dir = "./gf_output/"

    gpus = [0] if gpu_available else None
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    nwalkers = 4
    ntemps = 2

    # No Tukey taper. ``window_taper_duration = 0`` gives alpha = 0
    # (rectangular) inside :func:`lisatools.utils.utility.windowfun`.
    window_taper_duration = 0.0

    domain_settings = DOMAIN_CHOICE

    processor_init_kwargs = dict(data_input_path=ldc_source_file)

    # The synthesised data already covers exactly Tobs = Nf*Nt*dt;
    # skip the engine's default highpass + edge-trim + Tobs trim
    # because the default Tobs trim uses ``T=(N-1)*dt`` which would
    # lose one sample and break the WDM ``Nf*Nt`` shape.
    preprocess_kwargs = dict(
        highpass_kwargs=None,
        trim_kwargs=None,
        Tobs=None,
        normalize=False,
    )

    # CompositeSensitivityBackend is the default; only ``tdi_generation`` is
    # consumed. The legacy XYZ-only kwargs (mask_percentage / use_splines /
    # spline_order) are filtered out at the engine.
    sensitivity_init_kwargs = dict(tdi_generation=2)

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        main_file_key="testing",
        domain_settings=domain_settings,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type="tukey",
        window_taper_duration=window_taper_duration,
        gpu_backend=GPU_BACKEND,
        gpus=gpus,
        data_processor=SangriaProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
    )

    return GeneralSetup(general_settings)


def get_global_fit_settings(copy_settings_file=False):
    general_setup = get_general_erebor_settings()

    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir
            + general_setup.base_file_name
            + "_"
            + __file__.split("/")[-1],
        )

    rank_info = RankInfo(head_rank=1, main_rank=0)

    gb_setup = get_gb_erebor_settings(general_setup)
    psd_setup = get_psd_erebor_settings(general_setup)
    galfor_setup = get_galfor_erebor_settings(general_setup)
    mbh_setup = get_mbh_erebor_settings(general_setup)
    emri_setup = get_emri_erebor_settings(general_setup)
    sobbh_setup = get_sobbh_erebor_settings(general_setup)

    gf_settings = GlobalFitSettings(
        source_info={
            "gb": gb_setup,
            "psd": psd_setup,
            "galfor": galfor_setup,
            "mbh": mbh_setup,
            "emri": emri_setup,
            "sobbh": sobbh_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )
    return CurrentInfoGlobalFit(gf_settings)


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print("Global-fit settings constructed OK")
