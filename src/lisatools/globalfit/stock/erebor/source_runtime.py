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
from lisatools.utils.device import (
    current_device,
    device_context,
    jax_device_context,
)

from ...engine import GeneralSetup
from ...preprocessing import normalize_source_ids
from ...recipe import (
    MOJITO_REFERENCE_TIME,
    EMRIMoveBuilder,
    MBHMoveBuilder,
    SOBBHChunkedMoveBuilder,
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


# ============================================================
# Per-device source-generator replicas (multi-GPU walker shards)
# ============================================================
# On a multi-GPU walker-shard run, each walker's residual/PSD views live on
# its owning device (``AnalysisContainerArray`` split). The source moves score
# each walker INSIDE that device context (addremovemove.compute_acs_like), so
# the template generator must build on the SAME device -- otherwise a walker on
# shard >= 1 generates through a generator whose orbit grid and (for MBH) JAX
# tables live on the primary device, tripping cupy/JAX peer access: slow, and a
# hard failure on nodes without P2P.
#
# Rather than a separate multi-GPU code path, generation stays UNIFIED: the
# device-keyed replica cache below has exactly one entry on CPU / single-GPU
# (the run's primary device reuses the shared orbits + shared generator cache,
# byte-identical to the pre-multi-GPU path) and one entry per device on
# multi-GPU. ``current_device(xp)`` -- set by the move's per-shard
# ``device_context`` -- selects the entry.
_DEVICE_ORBITS_REPLICAS: dict = {}


def _general_info_xp(general_info):
    """Array module (cupy/numpy) for a run's orbits (CPU path -> numpy)."""
    orbits = (
        general_info.gpu_orbits
        if general_info.gpus is not None
        else general_info.orbits
    )
    return orbits.xp


def _primary_device(general_info):
    """The run's main device (``gpus[0]``), or None on the CPU path."""
    gpus = general_info.gpus
    return int(gpus[0]) if gpus is not None else None


def _device_local_orbits(orbits, xp, primary_device):
    """An orbits replica resident on the cupy current device.

    CPU / single-GPU / the run's primary device reuse the shared ``orbits``
    (zero extra memory -- identical to the pre-multi-GPU path). A non-primary
    device gets a lazily-built, cached ``orbits.__class__(*args, **kwargs)``
    replica whose grid arrays land on the current device (the
    :class:`~lisatools.domaincomputation.DomainComputationGroupArray`
    ``build_cpp_objects`` pattern), so a generator built around it reads orbit
    data locally instead of via peer access off the primary device.

    Built once per (orbits, device) and kept for the whole run
    (memory-lifecycle rule: allocate-once, persist).
    """
    dev = current_device(xp)
    if dev is None or dev == primary_device:
        return orbits
    key = (id(orbits), dev)
    replica = _DEVICE_ORBITS_REPLICAS.get(key)
    if replica is None:
        with device_context(xp, dev):
            replica = orbits.__class__(*orbits.args, **orbits.kwargs)
        _DEVICE_ORBITS_REPLICAS[key] = replica
    return replica


# Per-device WDM/domain-settings replicas. The wave wraps project the raw TD
# channels onto ``general_info.domain_settings`` (a WDMSettings) via
# ``.transform()``, which multiplies by ``settings.window`` (the WDM analysis
# window). That window lives on the primary device, so a shard-1 walker's
# device-local template * primary-device window trips peer access on EVERY
# generation (domains.py ``before_ifft[:] *= base_window``). A per-device
# settings replica moves the window onto the owning device.
_DEVICE_DOMAIN_REPLICAS: dict = {}


def _device_local_domain_settings(settings, xp, primary_device):
    """A domain-settings replica whose device arrays live on the current device.

    CPU / single-GPU / the primary device reuse the shared ``settings``
    (byte-identical to the pre-multi-GPU path). A non-primary device rebuilds
    ``settings.__class__(*args, **kwargs)`` with the device-resident WDM
    ``window`` / ``omega`` DROPPED, so ``__init__`` regenerates them on THIS
    device via ``setup_window()`` -- deterministic in ``(Nf, Nt, dt,
    oversample)``, hence numerically identical to the primary-device window
    (lnL parity preserved), just with no peer access off the primary device.
    Domain types without those keys (FD / STFT / TD) rebuild harmlessly from
    their scalar args. Built once per (settings, device) and kept for the run.
    """
    dev = current_device(xp)
    if dev is None or dev == primary_device:
        return settings
    if not (hasattr(settings, "args") and hasattr(settings, "kwargs")):
        return settings  # unknown settings type -> leave shared
    key = (id(settings), dev)
    replica = _DEVICE_DOMAIN_REPLICAS.get(key)
    if replica is None:
        kw = dict(settings.kwargs)
        # Regenerate the WDM window/omega on the target device (no-op key pops
        # for non-WDM domain types).
        kw.pop("window", None)
        kw.pop("omega", None)
        with device_context(xp, dev):
            replica = settings.__class__(*settings.args, **kw)
        _DEVICE_DOMAIN_REPLICAS[key] = replica
    return replica


def _wrap_device_and_orbits(general_info):
    """``(xp, dev, orbits, domain_settings)`` for a per-device source wave wrap.

    ``dev`` is the run's cupy current device (None on CPU / single-GPU) -- set
    by the move's per-shard ``device_context`` -- used to key the wave-wrap
    cache and to build the FEW/bbhx generator tables locally. ``orbits`` is a
    per-device replica of ``general_info.orbits`` (the CPU/processor orbits the
    EMRI and SOBBH generators use): on a non-primary device it has a DISTINCT
    ``id(orbits)``, which fans the inner ``id(orbits)``-keyed generator caches
    (get_emri_response_wrapper / get_sobbh_tdionfly_gen / ...) out per device
    too, without touching those lower-level getters. ``domain_settings`` is a
    per-device replica of ``general_info.domain_settings`` so the wave wrap's
    WDM ``.transform()`` multiplies by a device-local window (no peer access).
    On CPU / single-GPU / the primary device the shared objects are reused
    unchanged.
    """
    xp = _general_info_xp(general_info)
    dev = current_device(xp)
    primary = _primary_device(general_info)
    orbits = _device_local_orbits(general_info.orbits, xp, primary)
    domain_settings = _device_local_domain_settings(
        general_info.domain_settings, xp, primary
    )
    return xp, dev, orbits, domain_settings


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

    # In-model stretch repeats per leaf visit: the expose/fold residual
    # round-trip and the prev_logl batch are paid once per visit, so more
    # repeats amortize the add/remove overhead across more proposals.
    num_prop_repeats: int = dataclasses.field(
        default_factory=env_default("MBH_NUM_PROP_REPEATS", 2, int)
    )
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
    # An MBH is only added to the sampler if its merger lands within the
    # observed data plus this buffer past the end (seconds): keep iff
    # ``t_merge < observation_end + mbh_merger_time_buffer``. Default ~2 days.
    # (Merger-window filtering is MBH-only for now.)
    mbh_merger_time_buffer: float = dataclasses.field(
        default_factory=env_default("MBH_MERGER_TIME_BUFFER", 2 * 86400.0, float)
    )


@dataclasses.dataclass
class SourceEMRISettings(EMRISettings):
    """EMRI branch block (always the legacy ResponseWrapper path)."""

    num_prop_repeats: int = dataclasses.field(
        default_factory=env_default("EMRI_NUM_PROP_REPEATS", 2, int)
    )
    ndim: int = 12
    response_order: int = 40


@dataclasses.dataclass
class SourceSOBBHSettings(SOBBHSettings):
    """SOBBH branch block. Default path: TDI-ON-THE-FLY (validated)."""

    num_prop_repeats: int = dataclasses.field(
        default_factory=env_default("SOBBH_NUM_PROP_REPEATS", 2, int)
    )
    ndim: int = 11
    use_tdionfly: bool = dataclasses.field(
        default_factory=env_default("USE_TDIONFLY", True, bool)
    )
    n_grid: int = 2048
    buffer_time: float = 5000.0
    response_order: int = 40
    # Which likelihood scores the add/remove proposals: "chunked" (DEFAULT,
    # A/B-gated 2026-07-30: identical cold-chain lnL, zero cross-check
    # warnings at tol 0.5, ~4x wall even with the check every leaf) =
    # SOBBHChunkedLikeMove over the chunked-heterodyne WDM kernel (one
    # vectorized call per batch); "full" = the exact full-TD container path
    # (the escape hatch — required for FD/STFT domains, multi-shard, DCGA).
    # The residual expose/fold stays on the exact generator either way, and
    # the built-in fast-vs-slow cross-check stays on (thin it with
    # SOBBH_CHECK_LL_EVERY=10 in production; SOBBH_CHECK_LL=0 disables).
    likelihood: str = dataclasses.field(
        default_factory=env_default("SOBBH_LIKELIHOOD", "chunked", str)
    )
    # chunked-path knobs (see lisatools.chunked_het.WDMComputationsBase).
    # Nt_sub errs SMALL (short chunks): the per-chunk heterodyne collapses
    # a source whose intra-chunk frequency sweep exceeds the band (measured
    # on full_year-lite: Nt_sub=256 -> 2 chunks -> template plateaus at the
    # chunk carriers). SOBBH scoring is cheap, so safety wins over the
    # per-chunk FFT amortization.
    nt_sub: int = dataclasses.field(
        default_factory=env_default("SOBBH_NT_SUB", 32, int)
    )
    n_sparse: int = dataclasses.field(
        default_factory=env_default("SOBBH_N_SPARSE", 256, int)
    )
    n_pad: int = dataclasses.field(
        default_factory=env_default("SOBBH_N_PAD", 4, int)
    )
    m_band_half_width: int = dataclasses.field(
        default_factory=env_default("SOBBH_M_BAND_HALF_WIDTH", 1, int)
    )
    # band half-width for the RESIDUAL/template fill (the engine-installed
    # chunked signal_gen). Wider than the scoring band on purpose: fill
    # truncation lands in the SHARED residual every other branch sees, so
    # err wide (fill mismatch ~3e-4 at 6; scoring stays narrow and fast).
    fill_m_band_half_width: int = dataclasses.field(
        default_factory=env_default("SOBBH_FILL_M_BAND_HALF_WIDTH", 8, int)
    )


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
        inj_ids = sorted(cat.keys())
        injection = np.stack(
            [mbh_catalogue_to_sampling_basis(cat[i]) for i in inj_ids],
            axis=0,
        )
    else:
        inj_ids = list(range(n))
        inj_mode, inj_seed = synthetic_injection_mode(gs)
        injection = make_mbh_injections(
            n, general_setup.Tobs, mode=inj_mode, seed=inj_seed
        )

    # Merger-window filter (MBH-only for now): only add an MBH to the sampler
    # if its merger falls within the observed data plus a small buffer past the
    # end, i.e. ``t_merge < observation_end + buffer``. The sampling-basis
    # ``t_plunge`` (injection[:, -1], == TimeCoalescencePhenomTPHMSSBFrame in
    # mojito mode) and the data window are both referenced to
    # ``gs.mbh_waveform_t0``, so the data ends at
    # ``(data_t0 - mbh_waveform_t0) + Tobs`` in that frame.
    buffer = getattr(mbh, "mbh_merger_time_buffer", 0.0)
    obs_end = (general_setup.data_t0 - gs.mbh_waveform_t0) + general_setup.Tobs
    t_merge = np.asarray(injection[:, -1], dtype=float)
    keep = t_merge < (obs_end + buffer)
    if not bool(keep.all()):
        dropped = [inj_ids[k] for k in range(len(inj_ids)) if not keep[k]]
        logger.info(
            "MBH merger-window filter: dropping %d/%d MBHB source(s) %s with "
            "t_merge >= observation_end (%.6e s) + buffer (%.6e s).",
            len(dropped), len(inj_ids), dropped, obs_end, buffer,
        )
        injection = injection[keep]
    if injection.shape[0] == 0:
        logger.warning(
            "MBH merger-window filter removed every requested MBHB source; the "
            "MBH branch will have zero leaves (nothing merges before "
            "observation_end + buffer = %.6e s).",
            obs_end + buffer,
        )
    if mbh.injection is None:
        mbh.injection = injection
    # nleaves follow the (possibly filtered) injection, not the requested count.
    n = int(np.asarray(mbh.injection).shape[0])
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
        sobbh_likelihood=sobbh.likelihood,
        sobbh_nt_sub=sobbh.nt_sub,
        sobbh_n_sparse=sobbh.n_sparse,
        sobbh_n_pad=sobbh.n_pad,
        sobbh_m_band_half_width=sobbh.m_band_half_width,
        sobbh_fill_m_band_half_width=sobbh.fill_m_band_half_width,
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
    """EMRI legacy ResponseWrapper wrap (REF-anchored SPECIAL frame).

    Per-device (multi-GPU walker shards): the wrap and its FEW
    ``GenerateEMRIWaveform`` are cached and BUILT per current device, so each
    walker shard scores through a generator whose interpolation tables
    (ampinterp2d / modeselector) live on its OWN device -- eliminating the
    peer-access flood off the primary device that gave EMRI zero 2-GPU
    speedup. On CPU / single-GPU this collapses to one entry, byte-identical
    to the old path.
    """
    xp, dev, orbits, domain_settings = _wrap_device_and_orbits(general_info)
    key = ("emri", id(general_info), cfg["nchannels"], dev)
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

    # Build on the owning device so the FEW cuda tables are device-local.
    with device_context(xp, dev):
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
            orbits=orbits,
        )
        wrap = EMRIWaveWrap(
            template_wave_gen,
            general_info.data_td_settings,
            domain_settings,
            td_window=None,
            nchannels=cfg["nchannels"],
            offset_int=offset_int,
        )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def get_sobbh_wave_wrap(general_info, cfg):
    """SOBBH wrap: TDI-on-the-fly (default) or legacy ResponseWrapper.

    Per-device (multi-GPU walker shards): the wrap and its bbhx generator
    (``SOBBHTDIonFly`` / ResponseWrapper around ``SOBBHWaveform``) are cached
    and BUILT per current device, so each walker shard generates on its OWN
    device -- no peer access off the primary device. On CPU / single-GPU this
    collapses to one entry, byte-identical to the old path.
    """
    xp, dev, orbits, domain_settings = _wrap_device_and_orbits(general_info)
    key = ("sobbh", id(general_info), cfg["nchannels"], dev)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)
    reference_time = cfg["sobbh_reference_time"]

    # Build on the owning device so the bbhx cuda tables are device-local.
    with device_context(xp, dev):
        if cfg["sobbh_use_tdionfly"]:
            gen = get_sobbh_tdionfly_gen(
                Tobs=general_info.Tobs,
                dt=general_info.dt,
                t_start=general_info.data_t0,
                tdi_config=tdi_config,
                reference_time=reference_time,
                orbits=orbits,
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
                domain_settings,
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
                orbits=orbits,
                reference_time=reference_time,
            )
            wrap = SOBBHWaveWrap(
                template_wave_gen,
                general_info.data_td_settings,
                domain_settings,
                td_window=None,
                nchannels=cfg["nchannels"],
            )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def get_mbh_tdionfly_wave_wrap(general_info, cfg):
    """MBH TDI-on-the-fly wrap (source-independent phentax window).

    Like the legacy phentax path this is a JAX (phentax) waveform + cupy/C++
    (bbhx) response on ``gpu_orbits``; per device (multi-GPU) the wrap + its
    ``MBHTDIonFly`` are cached and built with a device-local orbits replica so
    the ``id(orbits)``-keyed generator cache fans out per device. The caller
    (:meth:`SourceSignalGen.__call__`) enters device_context+jax_device_context
    so both the build and the call land on the owning device.
    """
    xp = _general_info_xp(general_info)
    dev = current_device(xp)
    primary = _primary_device(general_info)
    base_orbits = (
        general_info.gpu_orbits if general_info.gpus is not None else general_info.orbits
    )
    orbits = _device_local_orbits(base_orbits, xp, primary)
    domain_settings = _device_local_domain_settings(
        general_info.domain_settings, xp, primary
    )
    key = ("mbh_tdionfly", id(general_info), cfg["nchannels"], dev)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)
    n = int(round(general_info.Tobs / general_info.dt))
    t_arr = np.arange(n) * general_info.dt + general_info.data_t0
    dur_s = general_info.Tobs + cfg["mbh_tdionfly_margin"]
    with device_context(xp, dev):
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
            domain_settings,
            nchannels=cfg["nchannels"],
        )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def get_mbh_phenom_gen(general_info, cfg):
    """MBH legacy ``PhenomTHMTDIWaveform`` (cached in ..wrappers).

    On a multi-GPU shard run this is called from inside the owning walker's
    ``device_context`` (see :meth:`compute_acs_like`); it then resolves a
    device-local orbits replica so the ``id(orbits)``-keyed
    :func:`get_mbh_phenom_wave_gen` cache yields one phentax generator per
    device -- built on that device (cupy response tables) and, when wrapped in
    :func:`jax_device_context`, with its phentax JAX tables local too. On CPU /
    single-GPU the primary device reuses the shared orbits, so this collapses
    to the original single cached generator.
    """
    xp = _general_info_xp(general_info)
    primary = _primary_device(general_info)
    base_orbits = (
        general_info.gpu_orbits
        if general_info.gpus is not None
        else general_info.orbits
    )
    orbits = _device_local_orbits(base_orbits, xp, primary)
    output_domain_settings = _device_local_domain_settings(
        general_info.domain_settings, xp, primary
    )
    return get_mbh_phenom_wave_gen(
        data_td_settings=general_info.data_td_settings,
        waveform_t0=cfg["mbh_waveform_t0"],
        dt=general_info.dt,
        orbits=orbits,
        output_domain_settings=output_domain_settings,
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
        if self.branch not in ("emri", "sobbh", "mbh"):
            raise ValueError(f"Unknown branch {self.branch!r} for SourceSignalGen.")

        # Per-shard device discipline (multi-GPU walker shards): generate on
        # the walker's OWNING cupy device so the template, its residual/invC
        # views, and the (per-device) generator tables all live on one device
        # -- no peer access off the primary device. The move already entered
        # this context; re-entering the current device is a no-op, and it also
        # makes SourceSignalGen self-sufficient when called from other paths.
        # CPU / single-GPU: dev is None / the primary, so every context below
        # collapses to a no-op and behaviour is unchanged.
        xp = _general_info_xp(self.general_info)
        dev = current_device(xp)
        with device_context(xp, dev):
            if self.branch == "emri":
                return get_emri_wave_wrap(self.general_info, self.cfg)(
                    *params_in, **kwargs
                )
            if self.branch == "sobbh":
                # chunked mode: the ENGINE-side template/residual generator
                # is the chunked-het fill too (wide FILL band; same family
                # as the move's scoring kernel), so residual bookkeeping,
                # load-time rebuilds, and scoring are mutually consistent
                # and all fast. The move's cross-check stays independent —
                # it verifies against the slow tdionfly wrap explicitly.
                if self.cfg.get("sobbh_likelihood", "full") == "chunked":
                    return get_sobbh_chunked_signal_gen(
                        self.general_info, self.cfg
                    )(*params_in, **kwargs)
                return get_sobbh_wave_wrap(self.general_info, self.cfg)(
                    *params_in, **kwargs
                )
            if self.branch == "mbh":
                # BOTH MBH paths are a JAX (phentax) waveform + cupy/C++
                # (pyResponseTDI / bbhx) response. cupy's device_context does
                # NOT move JAX, so ALSO pin the matching JAX default device
                # (phentax scalar inputs / traced kernels land there). No-op on
                # CPU / single-GPU / when JAX can't see the device.
                with jax_device_context(dev):
                    if self.cfg["mbh_use_tdionfly"]:
                        return get_mbh_tdionfly_wave_wrap(
                            self.general_info, self.cfg
                        )(*params_in, **kwargs)
                    gen = get_mbh_phenom_gen(self.general_info, self.cfg)
                    return gen.get_signals_for_residuals(*params_in, **kwargs)
        # Unreachable: the branch is validated above.


# ============================================================
# Runtime move builders
# ============================================================
def build_emri_move_runtime(curr, acs, priors, state, cfg):
    wave_gen = get_emri_wave_wrap(curr.general_info, cfg)
    _, moves = EMRIMoveBuilder(wave_gen=wave_gen).build(None, curr, acs, priors, state)
    return moves[0]


def get_sobbh_chunked_comp(general_info, cfg):
    """Build (and cache) the ``SOBBHWDMComputations`` for the chunked SOBBH move.

    WDM-only: the chunked-heterodyne kernels score directly against the run's
    WDM residual buffers, so the run's ``domain_settings`` must be a
    :class:`~lisatools.domains.WDMSettings`. ``t_ref`` is the catalogue
    reference epoch — the SAME resolution as :func:`get_sobbh_wave_wrap`'s
    ``reference_time`` (f_low / phi_c are defined there; the bbhx C++ t_ref
    fix makes the kernel intrinsics honor it). ``d_d = 0``: the move folds
    the exposed-residual ``<r|r>`` in via its per-walker offset.
    """
    from lisatools.domains import WDMSettings

    xp, dev, orbits, domain_settings = _wrap_device_and_orbits(general_info)
    key = ("sobbh_chunked", id(general_info), cfg["nchannels"], dev)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    if not isinstance(domain_settings, WDMSettings):
        raise ValueError(
            "SOBBH_LIKELIHOOD=chunked needs a WDM run domain "
            f"(general.domain_settings is {type(domain_settings).__name__}); "
            "use SOBBH_LIKELIHOOD=full for FD/STFT runs."
        )
    from bbhx.sobbhcomps import SOBBHWDMComputations

    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)
    # None -> the data window start, mirroring SOBBHWaveform's
    # ``reference_time = t0 if reference_time is None`` resolution
    t_ref = cfg["sobbh_reference_time"]
    if t_ref is None:
        t_ref = general_info.data_t0
    with device_context(xp, dev):
        comp = SOBBHWDMComputations(
            domain_settings,
            t_ref=float(t_ref),
            Nt_sub=cfg["sobbh_nt_sub"],
            n_pad=cfg["sobbh_n_pad"],
            N_sparse=cfg["sobbh_n_sparse"],
            orbits=orbits,
            tdi_config=tdi_config,
            tdi_type=cfg["tdi_chan"],
            d_d=0.0,
            force_backend=force_backend,
            # the stock WDM domain settings carry an ARRAY-SPACE t0 (=0)
            # while the data physically starts at data_t0; anchor the
            # chunk times at the true absolute start so t_ref keeps its
            # physical meaning (source phase AND orbits evaluated at the
            # real epoch)
            t_obs_start=float(general_info.data_t0),
        )
    _WAVE_WRAP_CACHE[key] = comp
    return comp


def get_sobbh_chunked_signal_gen(general_info, cfg):
    """Engine-convention SOBBH template generator backed by the chunked fill.

    Returns a cached (per device) callable ``fn(*waveform_params, ...) ->
    WDMSignal`` that renders the template via ``fill_global_wdm`` on the
    active-band grid — the SAME family the move's scoring kernel uses, so
    residual bookkeeping, load-time rebuilds, and scoring stay mutually
    consistent (and all fast). Band width = ``SOBBH_FILL_M_BAND_HALF_WIDTH``
    (wide: the truncation lands in the shared residual). The move's
    fast-vs-slow cross-check remains independent — it explicitly verifies
    against the slow tdionfly wrap.
    """
    xp, dev, orbits, domain_settings = _wrap_device_and_orbits(general_info)
    key = ("sobbh_chunked_gen", id(general_info), cfg["nchannels"], dev)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]

    from lisatools.domains import WDMSignal

    from ...moves.sobbhspecialmove import SOBBHChunkedLikeMove

    comp = get_sobbh_chunked_comp(general_info, cfg)
    wdm = domain_settings
    nch = int(cfg["nchannels"])
    m_fill = int(cfg["sobbh_fill_m_band_half_width"])

    def sobbh_chunked_gen(*params, apply_transform=False, leaf_inds=None,
                          **kwargs):
        row = np.asarray(params[:11], dtype=float).reshape(1, 11)
        p = SOBBHChunkedLikeMove.to_chunked_basis(row)
        buf = comp.xp.zeros(
            (nch, int(wdm.Nf_active), int(wdm.Nt_active)), dtype=float
        )
        comp.fill_global_wdm(
            p, buf, convert_to_ra_dec=False, m_band_half_width=m_fill
        )
        return WDMSignal(buf, wdm)

    _WAVE_WRAP_CACHE[key] = sobbh_chunked_gen
    return sobbh_chunked_gen


def build_sobbh_move_runtime(curr, acs, priors, state, cfg):
    wave_gen = get_sobbh_wave_wrap(curr.general_info, cfg)
    if cfg.get("sobbh_likelihood", "full") == "chunked":
        comp = get_sobbh_chunked_comp(curr.general_info, cfg)
        _, moves = SOBBHChunkedMoveBuilder(
            wave_gen=wave_gen,
            chunked_comp=comp,
            m_band_half_width=cfg["sobbh_m_band_half_width"],
        ).build(None, curr, acs, priors, state)
    else:
        _, moves = SOBBHMoveBuilder(wave_gen=wave_gen).build(
            None, curr, acs, priors, state
        )
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
