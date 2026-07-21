"""Shared EMRI / SOBBH / MBH source-branch runtime for stock global fits.

Extracted verbatim from ``full_year_combined`` so that variant *and*
``all_sources`` share a single source of truth for the phentax / TDI-on-the-fly
source branches:

- ``SourceMBHSettings`` / ``SourceEMRISettings`` / ``SourceSOBBHSettings`` —
  the per-branch knob blocks (phentax MBH default, TDI-on-the-fly SOBBH
  default, legacy-ResponseWrapper EMRI), each with a per-branch
  ``use_tdionfly`` toggle (``USE_TDIONFLY`` env flips both).
- ``prepare_{emri,sobbh,mbh}_branch`` — wide-prior branch resolution
  (catalogue vs synthetic injection, transform containers, initialize kwargs).
- ``SourceSignalGen`` + the cached ``get_*_wave_wrap`` template generators —
  the engine-side ``signal_gen`` adapters (built lazily post-deepcopy).
- ``build_{emri,sobbh,mbh}_move_runtime`` / ``build_source_moves`` — the
  runtime PE-move builders wrapping ``EMRIMoveBuilder`` / ``SOBBHMoveBuilder`` /
  ``MBHMoveBuilder`` / ``build_mbh_moves_phenom``.

The consuming variant supplies a ``general`` settings object exposing
``data_mode`` / ``n_injections`` / ``mbh_waveform_t0`` / ``sobbh_reference_time``
(duck-typed); everything here is behaviour-identical to the original
``full_year_combined`` implementation.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import typing

import numpy as np

from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.emri import emri_catalogue_to_waveform_basis
from lisatools.utils.constants import YRSID_SI

from ...engine import GeneralSetup
from ...preprocessing import normalize_source_ids
from ...recipe import (
    MOJITO_REFERENCE_TIME,
    EMRIMoveBuilder,
    MBHMoveBuilder,
    SOBBHMoveBuilder,
    build_mbh_moves_phenom,
    mbh_catalogue_to_sampling_basis,
)
from ..base import env_default
from .emri import EMRISettings
from .injections import (
    make_emri_injections,
    make_mbh_injections,
    make_sobbh_injections,
)
from .mbh import MBHSettings
from .sobbh import SOBBHSettings
from .transforms import (
    make_emri_transform_container,
    make_mbh_transform_container,
)
from .wrappers import (
    EMRIWaveWrap,
    MBHTDIonFlyWaveWrap,
    SOBBHTDIonFlyWaveWrap,
    SOBBHWaveWrap,
    get_emri_response_wrapper,
    get_mbh_phenom_wave_gen,
    get_mbh_tdionfly_gen,
    get_sobbh_response_wrapper,
    get_sobbh_tdionfly_gen,
)

logger = logging.getLogger(__name__)

# TDI-on-the-fly MBH waveform-window margin (seconds): the phentax window is
# sized from the DATA span + this margin so it is source-independent (the
# per-source merger time enters only as the call-time ``t_merge``).
MBH_TDIONFLY_MARGIN = 6.0 * 86400.0


def default_source_ids() -> dict:
    """Per-class source IDs; env MBHB_IDS / EMRI_IDS / SOBHB_IDS override."""
    ids = {"MBHB": [], "EMRI": [1], "SOBHB": []}
    for cls in ("MBHB", "EMRI", "SOBHB"):
        env_ids = os.environ.get(f"{cls}_IDS")
        if env_ids is not None:
            ids[cls] = [int(x) for x in env_ids.split(",") if x.strip() != ""]
    return normalize_source_ids(ids)


def _env_optional_duration(var: str, default: float):
    """Env float with ''/'none' -> None (MBH_WAVEFORM_DURATION semantics)."""

    def _factory():
        raw = os.environ.get(var)
        if raw is None:
            return default
        if raw.strip().lower() in ("", "none"):
            return None
        return float(raw)

    return _factory


# ============================================================
# Per-branch settings blocks (shared bases)
# ============================================================
@dataclasses.dataclass
class SourceMBHSettings(MBHSettings):
    """MBH branch block. Default path: LEGACY ``PhenomTHMTDIWaveform``."""

    num_prop_repeats: int = 2
    ndim: int = 11
    # Waveform path: legacy phentax (False, default) vs TDI-on-the-fly.
    use_tdionfly: bool = dataclasses.field(
        default_factory=env_default("USE_TDIONFLY", False, bool)
    )
    # phentax configuration (validated stft_tof choices).
    higher_modes: typing.Tuple[int, ...] = (21, 33, 44)
    phenom_tol: float = 1e-12
    start_freq: float = 7e-5
    response_order: int = 30
    buffer_time: float = 15_000.0
    # phentax generation window (None -> full data span).
    waveform_duration: typing.Optional[float] = dataclasses.field(
        default_factory=_env_optional_duration("MBH_WAVEFORM_DURATION", YRSID_SI / 12.0)
    )
    tdionfly_margin: float = MBH_TDIONFLY_MARGIN
    logM_prior: typing.Tuple[float, float] = (np.log(1e5), np.log(1e8))
    dist_prior: typing.Tuple[float, float] = (0.1, 150.0)  # Gpc
    t_plunge_pad: float = 3600.0


@dataclasses.dataclass
class SourceEMRISettings(EMRISettings):
    """EMRI branch block (always the legacy ResponseWrapper path)."""

    num_prop_repeats: int = 2
    ndim: int = 12
    response_order: int = 40


@dataclasses.dataclass
class SourceSOBBHSettings(SOBBHSettings):
    """SOBBH branch block. Default path: TDI-ON-THE-FLY (validated)."""

    num_prop_repeats: int = 2
    ndim: int = 11
    use_tdionfly: bool = dataclasses.field(
        default_factory=env_default("USE_TDIONFLY", True, bool)
    )
    n_grid: int = 2048
    buffer_time: float = 5000.0
    response_order: int = 40


# ============================================================
# Branch resolution (wide-prior catalogue/synthetic injection)
# ============================================================
def synthetic_injection_mode(gs) -> tuple:
    """``(mode, seed)`` for the ``make_*_injections`` calls, from the general block.

    Prior-mode draws apply only to synthetic data; anything else keeps the
    stock tables. The SAME values feed the data processor and the branch
    preparation so the injected data and the branch injection tables agree.
    """
    if getattr(gs, "data_mode", None) != "synthetic":
        return "stock", None
    mode = getattr(gs, "synthetic_injections", None) or "stock"
    return mode, getattr(gs, "synthetic_injection_seed", None)


def source_catalogue(general_setup: GeneralSetup, cls: str) -> typing.Optional[dict]:
    cat = getattr(general_setup, "catalogue", None) or getattr(
        general_setup.data_processor, "catalogue", {}
    )
    entry = cat.get(cls) if cat else None
    # Synthetic-mode catalogue rows are empty placeholders — treat as absent.
    if entry and all(len(v) > 0 for v in entry.values() if isinstance(v, dict)):
        return entry
    return None


def prepare_emri_branch(emri, general_setup: GeneralSetup, gs):
    from eryn.moves import StretchMove

    n = gs.n_injections["EMRI"]
    force_backend = general_setup.force_backend
    if emri.initialize_kwargs is None:
        emri.initialize_kwargs = dict(
            T=general_setup.Tobs / YRSID_SI,
            dt=general_setup.dt,
            emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
            emri_waveform_kwargs=dict(force_backend=force_backend),
            response_kwargs=dict(
                t0=general_setup.data_t0,
                order=emri.response_order,
                tdi=gs.tdi_gen_str,
                tdi_chan=gs.tdi_chan,
                force_backend=force_backend,
                remove_garbage="zero",
            ),
        )
    cat = source_catalogue(general_setup, "EMRI") if gs.data_mode == "mojito" else None
    if cat is not None:
        # SPECIAL EMRI frame (validated 2026-06-19): ecliptic-polar sky +
        # raw file spin angles; row construction lives in the stock
        # lisatools.sources.emri.emri_catalogue_to_waveform_basis.
        full_basis = np.asarray(
            [emri_catalogue_to_waveform_basis(cat[i]) for i in sorted(cat.keys())]
        )
    else:
        inj_mode, inj_seed = synthetic_injection_mode(gs)
        full_basis = make_emri_injections(n, mode=inj_mode, seed=inj_seed)
    # PER-LEAF transform fills: [xI0, Phi_theta0] per source (xI0 is the
    # intrinsic prograde/retrograde flag from the catalogue and can differ
    # per leaf). ``both_inverse_transforms`` (unfill) is value-independent,
    # so the injection inverse needs no leaf indices.
    leaf_fill_values = full_basis[:, [5, 12]]
    tc = make_emri_transform_container(leaf_fill_values)
    if emri.injection is None:
        emri.injection = tc.both_inverse_transforms(full_basis)
    if getattr(emri, "fill_values", None) is None or not np.asarray(
        emri.fill_values
    ).size:
        emri.fill_values = leaf_fill_values
    # Full-range priors: None keeps the wide defaults in EMRISetup.
    for lims in ("logm1_lims", "m2_lims", "a_lims", "p0_lims", "e0_lims"):
        if not getattr(emri, lims):
            setattr(emri, lims, None)
    if emri.waveform_kwargs is None:
        emri.waveform_kwargs = dict()
    if emri.inner_moves is None:
        emri.inner_moves = [(StretchMove(), 1.0)]
    emri.nleaves_max = n
    emri.nleaves_min = n
    return emri


def prepare_sobbh_branch(sobbh, general_setup: GeneralSetup, gs):
    from eryn.moves import StretchMove

    from .injections import SOBBH_INJECTION_PARAMS_FULL_BASIS  # noqa: F401
    from .transforms import make_sobbh_transform_container

    n = gs.n_injections["SOBHB"]
    force_backend = general_setup.force_backend
    if sobbh.initialize_kwargs is None:
        sobbh.initialize_kwargs = dict(
            T=general_setup.Tobs / YRSID_SI,
            dt=general_setup.dt,
            sobbh_waveform_args=("SOBBHWaveform",),
            sobbh_waveform_kwargs=dict(force_backend=force_backend),
            response_kwargs=dict(
                t0=general_setup.data_t0,
                order=sobbh.response_order,
                tdi=gs.tdi_gen_str,
                tdi_chan=gs.tdi_chan,
                force_backend=force_backend,
                remove_garbage="zero",
            ),
        )
    cat = source_catalogue(general_setup, "SOBHB") if gs.data_mode == "mojito" else None
    if cat is not None:
        from .injections import sobbh_catalogue_to_waveform_basis

        full_basis = np.asarray(
            [sobbh_catalogue_to_waveform_basis(cat[i]) for i in sorted(cat.keys())]
        )
    else:
        inj_mode, inj_seed = synthetic_injection_mode(gs)
        full_basis = make_sobbh_injections(n, mode=inj_mode, seed=inj_seed)
    if sobbh.injection is None:
        tc = make_sobbh_transform_container()
        sobbh.injection = np.stack(
            [tc.both_inverse_transforms(row) for row in full_basis], axis=0
        )
    if getattr(sobbh, "fill_values", None) is None:
        sobbh.fill_values = np.array([])
    for lims in ("logm1_lims", "logm2_lims", "s1_lims", "s2_lims", "f_low_lims"):
        if not getattr(sobbh, lims):
            setattr(sobbh, lims, None)
    if sobbh.waveform_kwargs is None:
        sobbh.waveform_kwargs = dict()
    if sobbh.inner_moves is None:
        sobbh.inner_moves = [(StretchMove(), 1.0)]
    sobbh.nleaves_max = n
    sobbh.nleaves_min = n
    return sobbh


def prepare_mbh_branch(mbh, general_setup: GeneralSetup, gs):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, log_uniform, uniform_dist

    n = gs.n_injections["MBHB"]
    if mbh.initialize_kwargs is None:
        mbh.initialize_kwargs = make_mbh_initialize_kwargs(mbh, general_setup, gs)
    cat = source_catalogue(general_setup, "MBHB") if gs.data_mode == "mojito" else None
    if cat is not None:
        injection = np.stack(
            [mbh_catalogue_to_sampling_basis(cat[i]) for i in sorted(cat.keys())],
            axis=0,
        )
    else:
        inj_mode, inj_seed = synthetic_injection_mode(gs)
        injection = make_mbh_injections(
            n, general_setup.Tobs, mode=inj_mode, seed=inj_seed
        )
    if mbh.injection is None:
        mbh.injection = injection
    if mbh.transform is None:
        mbh.transform = make_mbh_transform_container()
    if mbh.priors is None:
        # t_plunge is sampled relative to the waveform t0 epoch; the data
        # span starts at data_t0.
        t_rel_min = general_setup.data_t0 - gs.mbh_waveform_t0
        mbh.priors = {
            "mbh": ProbDistContainer(
                {
                    "logM": uniform_dist(*mbh.logM_prior),
                    "Q": log_uniform(1.0, 10.0),
                    "s1z": uniform_dist(-0.999999, 0.999999),
                    "s2z": uniform_dist(-0.999999, 0.999999),
                    "dist": uniform_dist(*mbh.dist_prior),
                    "phi_ref": uniform_dist(0.0, 2 * np.pi),
                    "cos_iota": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                    "psi": uniform_dist(0.0, np.pi),
                    "alpha": uniform_dist(0.0, 2 * np.pi),
                    "sin_delta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                    "t_plunge": uniform_dist(
                        t_rel_min,
                        t_rel_min + general_setup.Tobs + mbh.t_plunge_pad,
                    ),
                }
            )
        }
    if mbh.periodic is None:
        mbh.periodic = {
            "mbh": {"phi_ref": 2 * np.pi, "psi": np.pi, "alpha": 2 * np.pi}
        }
    if mbh.waveform_kwargs is None:
        mbh.waveform_kwargs = dict()
    if mbh.inner_moves is None:
        # TODO(post-merge): re-enable SkyMove hops once the move supports
        # the ICRS sampling basis.
        mbh.inner_moves = [(StretchMove(), 1.0)]
    mbh.nleaves_max = n
    mbh.nleaves_min = n
    return mbh


def make_mbh_initialize_kwargs(mbh, general_setup: GeneralSetup, gs) -> dict:
    """``PhenomTHMTDIWaveform`` construction kwargs for the legacy MBH path."""
    return dict(
        data_td_settings=general_setup.data_td_settings,
        waveform_t0=gs.mbh_waveform_t0,
        orbits=(
            general_setup.gpu_orbits
            if general_setup.gpus is not None
            else general_setup.orbits
        ),
        # WDM run target — communicated by settings object (sprint rule).
        output_domain_settings=general_setup.domain_settings,
        tukey_alpha=general_setup.window_alpha,
        force_backend=general_setup.force_backend,
    )


# ============================================================
# Runtime signal-gen config + cached wave-wrap generators
# ============================================================
def source_signal_cfg(gs, mbh, sobbh, emri) -> dict:
    """Plain-value config consumed by the wave-wrap getters below."""
    return dict(
        tdi_chan=gs.tdi_chan,
        tdi_gen_str=gs.tdi_gen_str,
        nchannels=gs.nchannels,
        data_mode=gs.data_mode,
        sobbh_reference_time=gs.sobbh_reference_time,
        mbh_waveform_t0=gs.mbh_waveform_t0,
        mbh_use_tdionfly=mbh.use_tdionfly,
        sobbh_use_tdionfly=sobbh.use_tdionfly,
        mbh_tdionfly_margin=mbh.tdionfly_margin,
        emri_response_order=emri.response_order,
        sobbh_response_order=sobbh.response_order,
        sobbh_n_grid=sobbh.n_grid,
        sobbh_buffer_time=sobbh.buffer_time,
        mbh_phenom_kwargs=dict(
            waveform_duration=mbh.waveform_duration,
            higher_modes=mbh.higher_modes,
            phenom_tol=mbh.phenom_tol,
            start_freq=mbh.start_freq,
            response_order=mbh.response_order,
            buffer_time=mbh.buffer_time,
            min_freq=gs.min_freq,
            max_freq=gs.max_freq,
        ),
    )


_WAVE_WRAP_CACHE = {}


def get_emri_wave_wrap(general_info, cfg):
    """EMRI legacy ResponseWrapper wrap (REF-anchored SPECIAL frame)."""
    key = ("emri", id(general_info), cfg["nchannels"])
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)

    dt = general_info.dt
    data_t0 = general_info.data_t0
    # Catalogue reference epoch (shared by all sources); falls back to the
    # data start only in synthetic mode (where REF == data_t0).
    ref = MOJITO_REFERENCE_TIME if cfg["data_mode"] == "mojito" else data_t0
    off = data_t0 - ref
    offset_int = int(round(off / dt))
    t0_shift = off - offset_int * dt  # sub-sample remainder, |t0_shift| < dt
    out_N = int(round(general_info.Tobs / dt))
    resp_Tobs = (out_N + offset_int) * dt

    template_wave_gen = get_emri_response_wrapper(
        Tobs=resp_Tobs,
        dt=dt,
        t_start=ref,
        t0_shift_to_data=t0_shift,
        tdi_config=tdi_config,
        tdi_chan=cfg["tdi_chan"],
        role="template",
        order=cfg["emri_response_order"],
        force_backend=force_backend,
        orbits=general_info.orbits,
    )
    wrap = EMRIWaveWrap(
        template_wave_gen,
        general_info.data_td_settings,
        general_info.domain_settings,
        td_window=None,
        nchannels=cfg["nchannels"],
        offset_int=offset_int,
    )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def get_sobbh_wave_wrap(general_info, cfg):
    """SOBBH wrap: TDI-on-the-fly (default) or legacy ResponseWrapper."""
    key = ("sobbh", id(general_info), cfg["nchannels"])
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)
    reference_time = cfg["sobbh_reference_time"]

    if cfg["sobbh_use_tdionfly"]:
        gen = get_sobbh_tdionfly_gen(
            Tobs=general_info.Tobs,
            dt=general_info.dt,
            t_start=general_info.data_t0,
            tdi_config=tdi_config,
            reference_time=reference_time,
            orbits=general_info.orbits,
            n_grid=cfg["sobbh_n_grid"],
            buffer_time=cfg["sobbh_buffer_time"],
            force_backend=force_backend,
        )
        n = int(round(general_info.Tobs / general_info.dt))
        t_arr = np.arange(n) * general_info.dt + general_info.data_t0
        wrap = SOBBHTDIonFlyWaveWrap(
            gen,
            t_arr,
            general_info.data_td_settings,
            general_info.domain_settings,
            td_window=None,
            nchannels=cfg["nchannels"],
        )
    else:
        template_wave_gen = get_sobbh_response_wrapper(
            Tobs=general_info.Tobs,
            dt=general_info.dt,
            t_start=general_info.data_t0,
            tdi_config=tdi_config,
            tdi_chan=cfg["tdi_chan"],
            role="template",
            order=cfg["sobbh_response_order"],
            force_backend=force_backend,
            orbits=general_info.orbits,
            reference_time=reference_time,
        )
        wrap = SOBBHWaveWrap(
            template_wave_gen,
            general_info.data_td_settings,
            general_info.domain_settings,
            td_window=None,
            nchannels=cfg["nchannels"],
        )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def get_mbh_tdionfly_wave_wrap(general_info, cfg):
    """MBH TDI-on-the-fly wrap (source-independent phentax window)."""
    key = ("mbh_tdionfly", id(general_info), cfg["nchannels"])
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)
    orbits = (
        general_info.gpu_orbits if general_info.gpus is not None else general_info.orbits
    )
    n = int(round(general_info.Tobs / general_info.dt))
    t_arr = np.arange(n) * general_info.dt + general_info.data_t0
    dur_s = general_info.Tobs + cfg["mbh_tdionfly_margin"]
    gen = get_mbh_tdionfly_gen(
        dt=general_info.dt,
        t_start=cfg["mbh_waveform_t0"],
        dur_s=dur_s,
        tdi_config=tdi_config,
        orbits=orbits,
        waveform_duration=dur_s,
        force_backend=force_backend,
    )
    wrap = MBHTDIonFlyWaveWrap(
        gen,
        t_arr,
        general_info.data_td_settings,
        general_info.domain_settings,
        nchannels=cfg["nchannels"],
    )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def get_mbh_phenom_gen(general_info, cfg):
    """MBH legacy ``PhenomTHMTDIWaveform`` (cached in ..wrappers)."""
    return get_mbh_phenom_wave_gen(
        data_td_settings=general_info.data_td_settings,
        waveform_t0=cfg["mbh_waveform_t0"],
        dt=general_info.dt,
        orbits=(
            general_info.gpu_orbits
            if general_info.gpus is not None
            else general_info.orbits
        ),
        output_domain_settings=general_info.domain_settings,
        tukey_alpha=general_info.window_alpha,
        force_backend=general_info.force_backend,
        data_span=general_info.Tobs,
        tdi_gen_str=cfg["tdi_gen_str"],
        tdi_chan=cfg["tdi_chan"],
        **cfg["mbh_phenom_kwargs"],
    )


class SourceSignalGen:
    """Named-class params-based engine ``signal_gen`` for one branch.

    Replaces the legacy ``_emri_signal_gen``-style closures. Holds the
    branch transform + the runtime ``general_info`` + a plain-value config;
    the (heavy, unpicklable) wave wraps are built lazily on first call and
    live in this module's cache — never on the instance. Attached to the
    runtime Setups post-deepcopy (``attach_runtime_objects``), so the
    pre-build fit config stays picklable.
    """

    def __init__(self, branch, transform, general_info, cfg):
        self.branch = branch
        self.transform = transform
        self.general_info = general_info
        self.cfg = cfg

    def __call__(self, *params, apply_transform=True, leaf_inds=None, **kwargs):
        """Build this branch's template from ``params``.

        Args:
            *params: Sampling-basis parameters by default. With
                ``apply_transform=False`` the caller has already applied the
                branch transform and ``params`` are waveform-basis — used by
                the add/remove move, whose choreography transforms once up
                front (the transform must be applied exactly once).
            leaf_inds: Per-row leaf indices, forwarded to the transform.
                Required (by Eryn) when the branch transform carries
                per-leaf fills (e.g. EMRI xI0/Phi_theta0) and
                ``apply_transform=True``; ignored otherwise.
            **kwargs: Forwarded to the wave wrap.
        """
        params_arr = np.asarray(params, dtype=float)
        params_in = (
            self.transform.both_transforms(params_arr, leaf_inds=leaf_inds)
            if apply_transform
            else params_arr
        )
        if self.branch == "emri":
            return get_emri_wave_wrap(self.general_info, self.cfg)(*params_in, **kwargs)
        if self.branch == "sobbh":
            return get_sobbh_wave_wrap(self.general_info, self.cfg)(*params_in, **kwargs)
        if self.branch == "mbh":
            if self.cfg["mbh_use_tdionfly"]:
                return get_mbh_tdionfly_wave_wrap(self.general_info, self.cfg)(
                    *params_in, **kwargs
                )
            gen = get_mbh_phenom_gen(self.general_info, self.cfg)
            return gen.get_signals_for_residuals(*params_in, **kwargs)
        raise ValueError(f"Unknown branch {self.branch!r} for SourceSignalGen.")


# ============================================================
# Runtime move builders
# ============================================================
def build_emri_move_runtime(curr, acs, priors, state, cfg):
    wave_gen = get_emri_wave_wrap(curr.general_info, cfg)
    _, moves = EMRIMoveBuilder(wave_gen=wave_gen).build(None, curr, acs, priors, state)
    return moves[0]


def build_sobbh_move_runtime(curr, acs, priors, state, cfg):
    wave_gen = get_sobbh_wave_wrap(curr.general_info, cfg)
    _, moves = SOBBHMoveBuilder(wave_gen=wave_gen).build(None, curr, acs, priors, state)
    return moves[0]


def build_mbh_move_runtime(curr, acs, priors, state, cfg):
    """MBH PE move: stretch RJ move on the tdionfly wrap, or the stock
    ``build_mbh_moves_phenom`` builder around the cached phentax generator."""
    mbh_info = curr.source_info["mbh"]
    if not cfg["mbh_use_tdionfly"]:
        wave_gen = get_mbh_phenom_gen(curr.general_info, cfg)
        _, move = build_mbh_moves_phenom(
            curr, acs, priors, state, wave_gen=wave_gen, subtract_initial=False
        )
        return move
    wave_gen = get_mbh_tdionfly_wave_wrap(curr.general_info, cfg)
    _, moves = MBHMoveBuilder(
        wave_gen=wave_gen, waveform_like_kwargs=mbh_info.waveform_kwargs
    ).build(None, curr, acs, priors, state)
    return moves[0]


def find_source_cfg(curr):
    """Recover the plain-value cfg carried by a branch's ``SourceSignalGen``."""
    for name in curr.engine_info.branch_names:
        sg = getattr(curr.source_info[name], "signal_gen", None)
        if isinstance(sg, SourceSignalGen):
            return sg.cfg
    return None


def build_source_moves(curr, acs, priors, state, cfg) -> dict:
    """Build the mbh/emri/sobbh PE moves present on ``curr`` into a name->move
    dict (matching the ``mbh_pe`` / ``emri_pe`` / ``sobbh_pe`` stock-move names)."""
    stock_moves = {}
    if "mbh" in curr.source_info:
        stock_moves["mbh_pe"] = build_mbh_move_runtime(curr, acs, priors, state, cfg)
    if "emri" in curr.source_info:
        stock_moves["emri_pe"] = build_emri_move_runtime(curr, acs, priors, state, cfg)
    if "sobbh" in curr.source_info:
        stock_moves["sobbh_pe"] = build_sobbh_move_runtime(curr, acs, priors, state, cfg)
    return stock_moves
