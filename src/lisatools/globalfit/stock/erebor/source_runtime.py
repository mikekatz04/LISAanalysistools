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
from lisatools.utils.constants import MTSUN_SI, YRSID_SI
from lisatools.utils.device import (
    current_device,
    device_context,
    jax_device_context,
)

from ...engine import GeneralSetup
from ...preprocessing import normalize_source_ids
from ...recipe import (
    MOJITO_REFERENCE_TIME,
    EMRIDirectMoveBuilder,
    EMRIMoveBuilder,
    MBHBatchedMoveBuilder,
    MBHMoveBuilder,
    SOBBHChunkedMoveBuilder,
    SOBBHMoveBuilder,
    build_mbh_moves_phenom,
    mbh_catalogue_to_sampling_basis,
)
from ...warmstart.sources import parse_id_list
from ..base import env_default
from .common import resolve_inner_moves
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


# Per-device TDI-config replicas. ``TDIConfig.__init__`` uploads its six
# link/sign/channel tables (``pytdiconfig_args``,
# ``response/tdiconfig.py``) at construction, and a GB comp's
# ``cpp_tdi_config = backend.TDIConfigWrap(*tdi_config.pytdiconfig_args)``
# captures those pointers verbatim -- so a comp replica handed the SHARED
# TDIConfig would still dereference the primary device's tables. The
# rebuild is exact: ``tdi_combinations`` is the constructor's own input.
_DEVICE_TDI_CONFIG_REPLICAS: dict = {}


def _device_local_tdi_config(tdi_config, xp, device, primary_device):
    """A TDI-config replica whose link tables live on ``device``.

    CPU / the comp's own build device reuse the shared object (byte-identical
    to the single-GPU path, zero extra memory). Anything else gets one cached
    replica built from ``tdi_config.tdi_combinations`` -- the very list the
    constructor consumes -- inside the target device context, so the six
    ``pytdiconfig_args`` tables (and therefore the ``TDIConfigWrap`` pointer
    fields a comp builds from them) are device-local.
    """
    if tdi_config is None or device is None or primary_device is None:
        return tdi_config
    if int(device) == int(primary_device):
        return tdi_config
    if not hasattr(tdi_config, "tdi_combinations"):
        return tdi_config  # a string / unknown spec -> the comp rebuilds it
    key = (id(tdi_config), int(device))
    hit = _DEVICE_TDI_CONFIG_REPLICAS.get(key)
    if hit is not None:
        return hit[1]
    with device_context(xp, int(device)):
        replica = type(tdi_config)(
            tdi_config.tdi_combinations, force_backend=tdi_config.backend
        )
    # Keep a strong reference to the prototype in the VALUE: the key holds
    # only ``id(prototype)``, so a garbage-collected prototype could
    # otherwise let a recycled id alias a stale replica.
    _DEVICE_TDI_CONFIG_REPLICAS[key] = (tdi_config, replica)
    return replica


# Per-device GB computation-object replicas (the chunked-het / FD / sig-het
# comps the GB band moves score through). The comps are built ONCE at variant
# build time -- ``gb_info.gb_wdm_comp`` / ``gb_fd_comp`` -- and pin every
# buffer they allocate (chunk geometry, WDM window, ``OrbitsWrap`` /
# ``TDIConfigWrap`` pointer fields, and for sig-het the ``window_full`` /
# ``n_sparse_local`` / ``tdi_wrap`` / reference stash) to the device current
# at that moment. The GB shard router launches each walker shard inside its
# OWN device context (``_RoutedBandEngine``), so on a multi-GPU run every
# non-primary shard would dereference the primary device's pointers: a silent
# peer-access tax where P2P is enabled, an illegal access where it is not.
#
# One replica per (comp, device) closes it for every routed entry point at
# once -- ``get_ll`` / ``get_swap_ll`` / ``fill_template`` / the information
# matrix / the F-stat -- and, for sig-het, gives each shard its own reference
# stash, its own slot->reference map and its own ``_in_model`` flag, which is
# what makes multi-shard in-model scoring correct at all (see
# ``docs/superpowers/2026-08-03-multigpu-sighet-audit.md``).
_DEVICE_GB_COMP_REPLICAS: dict = {}


def _device_local_gb_comp(comp, xp, device, primary_device):
    """A GB comp replica resident on ``device``.

    ``primary_device`` is the device the prototype's own buffers live on
    (``comp._build_device`` where the router can read it, else ``gpus[0]``):
    that device -- and the CPU path (``device is None``) -- reuses the shared
    object, so single-GPU behaviour is byte-identical and allocates nothing.
    Any other device gets ONE lazily-built, cached replica.

    Three comp shapes are handled, all by duck-typing so this module keeps no
    import dependency on gbgpu:

    * the sig-het wrapper (``GBSignalHetComputations``) -- built through
      ``for_band_engine(chunked_comp, **knobs)``, never ``__class__(*args)``,
      so it is rebuilt the same way around a device-local delegate with the
      knobs recovered from the instance's own ``_g`` grid/knob dict;
    * ``WDMComputationsBase`` subclasses (``GBWDMComputations``) and
      ``GBFDComputations`` -- rebuilt from the recorded ``comp.args`` /
      ``comp.kwargs`` with the device-resident ``orbits`` / ``tdi_config`` /
      WDM-settings arguments swapped for their own device-local replicas;
    * anything else -- returned shared, unchanged (no silent half-fix).

    Built once per (comp, device) and kept for the whole run
    (memory-lifecycle rule: allocate-once, persist; module-level cache, never
    the settings tree, never the proposal-lifetime holder).
    """
    if comp is None or device is None or primary_device is None:
        return comp
    if int(device) == int(primary_device):
        return comp
    key = (id(comp), int(device))
    hit = _DEVICE_GB_COMP_REPLICAS.get(key)
    if hit is not None:
        return hit[1]
    with device_context(xp, int(device)):
        replica = _build_gb_comp_replica(comp, xp, int(device),
                                         int(primary_device))
    if replica is comp:
        return comp
    # Prototype kept alive in the VALUE (see _device_local_tdi_config).
    _DEVICE_GB_COMP_REPLICAS[key] = (comp, replica)
    return replica


#: ``GBSignalHetComputations.for_band_engine`` knob name -> the key it is
#: recorded under in the instance's ``_g`` dict. Deriving the replica's knobs
#: from ``_g`` (rather than a second stash) keeps the sig-het class free of
#: replica-only state; the recorded values are already RESOLVED (snapped
#: ``nt_layer``, resolved ``n_cp_build``), and both resolutions are
#: idempotent, so a rebuild reproduces the prototype exactly.
_SIGHET_REPLICA_KNOBS = {
    "nt_layer": "nt_layer",
    "n_sparse_fd": "n_sparse_fd",
    "m_active_half_width": "m_half",
    "max_r": "max_r",
    "n_cp_build": "n_cp_build",
    "v3_n_nodes": "v3_n_nodes",
    "v4_knots": "v4_knots",
    "v4_band": "v4_band",
    "v5": "v5",
    # tukey_alpha was MISSING here until 2026-08-19: the replica rebuild
    # then fell to for_band_engine's None default = legacy inheritance from
    # the chunked delegate's resolved per-chunk alpha (0.05), which the
    # EC>=taper guard clamps to 2*ind_min_t/Nt -- so the NON-PRIMARY
    # device's engine scored with a ~2x wider reference taper than the
    # primary's pinned SIGHET_TUKEY_ALPHA. A per-device likelihood config
    # divergence, and a direct violation of the Tukey equivalence policy
    # (equal alphas everywhere, never knob-resolved).
    "tukey_alpha": "tukey_alpha",
}


def _build_gb_comp_replica(comp, xp, device, primary_device):
    """Construct one GB comp replica (called inside ``device``'s context)."""
    # --- sig-het wrapper: rebuilt through its own factory -----------------
    chunked = getattr(comp, "chunked", None)
    if chunked is not None and hasattr(type(comp), "for_band_engine"):
        g = getattr(comp, "_g", None)
        if g is None:
            return comp
        knobs = {
            name: g[gkey]
            for name, gkey in _SIGHET_REPLICA_KNOBS.items()
            if gkey in g
        }
        return type(comp).for_band_engine(
            _device_local_gb_comp(chunked, xp, device, primary_device),
            **knobs,
        )

    # --- chunked-het / FD comps: rebuilt from recorded ctor arguments -----
    if not (hasattr(comp, "args") and hasattr(comp, "kwargs")):
        return comp
    args = list(comp.args)
    kw = dict(comp.kwargs)
    # The first positional is the domain settings (WDMSettings / FDSettings).
    # WDMSettings carries a device-resident analysis ``window``; the existing
    # domain replica helper regenerates it on THIS device from the same
    # deterministic (Nf, Nt, dt, oversample), so values are unchanged.
    if args and hasattr(args[0], "args") and hasattr(args[0], "kwargs"):
        args[0] = _device_local_domain_settings_on(
            args[0], xp, device, primary_device
        )
    # ``orbits=None`` is fine: the comp's setter builds EqualArmlengthOrbits
    # and configures it HERE, i.e. on this device.
    #
    # Orbits: hand the replica the PROTOTYPE'S CONFIGURED orbits object,
    # never a reconstructed one. Reconstruction
    # (``orbits.__class__(*args, **kwargs)``) loses post-construction
    # state -- for file-based ``L1Orbits`` the loader-installed base
    # tables / configured grid -- so the comp setter saw
    # ``configured=False`` and RE-configured the fresh object on a
    # different grid, and the replica produced NaN waveforms (2026-08-12
    # shard bug: every replica-scored cell's d_h/h_h NaN -> auto-reject ->
    # in-model acceptance uniformly HALVED on 2 GPUs). Sharing the
    # configured object is safe and bit-identical: the setter deepcopies
    # its input, the Python-side orbit tables are host numpy, and the
    # device-resident piece (``backend.OrbitsWrap``) is rebuilt inside
    # THIS device's context by the setter itself.
    _proto_orbits = getattr(comp, "_orbits", None)
    if _proto_orbits is not None:
        kw["orbits"] = _proto_orbits
    elif kw.get("orbits") is not None:
        kw["orbits"] = _device_local_orbits(kw["orbits"], xp, primary_device)
    if kw.get("tdi_config") is not None:
        kw["tdi_config"] = _device_local_tdi_config(
            kw["tdi_config"], xp, device, primary_device
        )
    return type(comp)(*args, **kw)


def _device_local_domain_settings_on(settings, xp, device, primary_device):
    """:func:`_device_local_domain_settings` with an EXPLICIT target device.

    The source-move callers already sit inside their shard's ``device_context``
    so the current-device form is enough for them; the GB replica builder
    resolves a device it was handed, so it needs the explicit spelling. Same
    cache, same "primary reuses the shared object" rule.
    """
    if device is None or primary_device is None or int(device) == int(primary_device):
        return settings
    with device_context(xp, int(device)):
        return _device_local_domain_settings(settings, xp, primary_device)


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
#: Hard default of ``SourceMBHSettings.waveform_duration`` (1/12 yr).
MBH_DEFAULT_WAVEFORM_DURATION = YRSID_SI / 12.0


@dataclasses.dataclass
class SourceMBHSettings(MBHSettings):
    """MBH branch block. Default waveform path: LEGACY phentax response
    (``use_tdionfly=False``); default scoring path ``likelihood="auto"`` (the
    batched windowed likelihood wherever the run can use it)."""

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
    # Response Lagrange order. User ruling 2026-09-29: default 8 (PR #82
    # evidence: mismatch flat from order 30 down to 4; ~1.6x cheaper).
    response_order: int = dataclasses.field(
        default_factory=env_default("MBH_RESPONSE_ORDER", 8, int)
    )
    # Scoring path: "batched" (the MBHBatchedLikeMove: one grid-aligned
    # response launch per chunk of rows on a per-leaf window, segment WDM
    # transform, per-walker residual+PSD), "full" (stock per-row container
    # path), or "auto" (DEFAULT since 2026-09-30, user "make this new path the
    # default"): "batched" whenever the run can use it, else "full" with one
    # INFO line naming why -- USE_TDIONFLY=1, a non-WDM run domain, or an
    # MBH_WAVEFORM_DURATION env value other than the window
    # (resolve_mbh_batched_cfg). An explicit "batched" still raises on those.
    likelihood: str = dataclasses.field(
        default_factory=env_default("MBH_LIKELIHOOD", "auto", str)
    )
    # Rows per batched generator launch. 8 (was 16; controller ruling
    # 2026-09-30 from the H100 benchmark): B=8 adds +8.5 GB of device memory
    # for 0.037 s/row, B=16 +18.5 GB for 0.034 s/row, and production cards
    # also carry the GB buffers.
    batch_max_size: int = dataclasses.field(
        default_factory=env_default("MBH_BATCH_MAX_SIZE", 8, int)
    )
    # Per-leaf window around the median cold-chain merger (days): kept box =
    # [t - before - margin, t + after + margin]; pad = discarded segment edge
    # (must be >= buffer_time: the response zeroes that much of the head).
    # A data span shorter than the window (3 months, the lite smokes) clamps
    # the kept box to the data's active box and the pads to the grid
    # (mbh_window_layers) instead of refusing the run.
    window_before_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_BEFORE_DAYS", 90.0, float)
    )
    window_after_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_AFTER_DAYS", 10.0, float)
    )
    window_pad_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_PAD_DAYS", 4.0, float)
    )
    window_margin_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_MARGIN_DAYS", 1.0, float)
    )
    # Batched window lattice DECIMATION FACTOR (2026-10-02; not the mass ratio):
    # the windowed template is generated, responded and WDM-transformed at
    # decimation*dt on Nf/decimation layers -- the same layer_dt, so the same
    # pixels -- for a ~decimation-fold cut of phentax + response + transform. The
    # epoch snaps onto the decimation*dt lattice (both generators). Laptop check
    # (2 days, layer 960 s, band 0.25-25 mHz; kept layers vs no decimation):
    # decimation 2 mismatch <= 1.5e-9 at 6e5 and 6e6 Msun; decimation 4 4.7e-9 at
    # 6e6 but 5.7e-5 with a 0.7 % norm loss at 6e5 Msun (its higher-mode ringdown
    # passes the 50 mHz coarse Nyquist). 1 = the data lattice.
    window_decimate: int = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_DECIMATE", 1, int)
    )
    buffer_time: float = 15_000.0
    # phentax generation window (None -> full data span).
    waveform_duration: typing.Optional[float] = dataclasses.field(
        default_factory=_env_optional_duration(
            "MBH_WAVEFORM_DURATION", MBH_DEFAULT_WAVEFORM_DURATION
        )
    )
    tdionfly_margin: float = MBH_TDIONFLY_MARGIN
    logM_prior: typing.Tuple[float, float] = (np.log(1e5), np.log(1e8))
    dist_prior: typing.Tuple[float, float] = (0.1, 150.0)  # Gpc
    t_plunge_pad: float = 3600.0
    # An MBH is only added to the sampler if its merger lands within the
    # observed data plus this buffer past the end (seconds): keep iff
    # ``observation_start <= t_merge < observation_end +
    # mbh_merger_time_buffer``. Default 7 days:
    # a source merging up to a week after the data end still has its late
    # inspiral in the data, so it is kept (user, 2026-09-30: drop an MBH only
    # where the data itself is dropped). The ``t_plunge`` prior's upper edge
    # follows this buffer so every admitted source sits inside its own prior.
    # (Merger-window filtering is MBH-only for now.)
    mbh_merger_time_buffer: float = dataclasses.field(
        default_factory=env_default("MBH_MERGER_TIME_BUFFER", 7 * 86400.0, float)
    )
    # ---- SOURCE WARM START (user 2026-10-08; lisatools.globalfit.warmstart.sources).
    # On a FRESH store, each leaf whose catalogue id is in the seed store with
    # seed-run optimal SNR (median over cold walkers of sqrt(h_h)) STRICTLY above
    # source_warm_start_snr_min starts at that store's last cold-chain positions
    # (walker w <- w; the whole ladder when the rung counts match, else the cold
    # point on every rung); every other leaf starts as before (MBH_START_FACTOR).
    # Unset store = off (today's start, byte-identical). Inert on a resume. The
    # store / SNR knobs are SHARED by mbh, emri and sobbh (like USE_TDIONFLY);
    # the id list is per branch because the store does not record it.
    source_warm_start_store: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_STORE", None, str)
    )
    source_warm_start_snr_min: float = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_SNR_MIN", 10.0, float)
    )
    # NaN seed record (the seed run had MBH_RECORD_DH off): "refuse" (default),
    # "truth" (injection start) or "warm" (seed start regardless of SNR).
    source_warm_start_snr_unknown: str = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_SNR_UNKNOWN", "refuse", str)
    )
    # The SEED run's catalogue ids for this branch (its MBHB_IDS after its own
    # merger-window filter), any order; "none" disables this branch.
    source_warm_start_ids: typing.Optional[tuple] = dataclasses.field(
        default_factory=env_default("MBH_SOURCE_WARM_START_IDS", None, parse_id_list)
    )
    # THIS run's catalogue ids in leaf order; COMPUTED by prepare_mbh_branch
    # (never an env knob).
    injection_ids: typing.Optional[tuple] = None


@dataclasses.dataclass
class SourceEMRISettings(EMRISettings):
    """EMRI branch block. ``likelihood`` picks the template family for BOTH the add/remove
    scoring and the engine's templates (residual rebuilds, fills): the direct-to-WDM lookup
    template by default (``auto`` on a WDM / XYZ run), the legacy ResponseWrapper path with
    ``full``; the move's cross-check always uses the latter."""

    num_prop_repeats: int = dataclasses.field(
        default_factory=env_default("EMRI_NUM_PROP_REPEATS", 2, int)
    )
    ndim: int = 12
    response_order: int = 40
    # Template path (2026-10-02; default auto since 10-03): "direct" (EMRIDirectLikeMove
    # scoring + the engine templates: the direct-to-WDM template -- n_ref lookup table +
    # dense-phase TDI-on-the-fly response + exact plunge chunk -- batched over
    # batch_max_size rows, scored against each walker's own residual and PSD), "full" (the
    # per-row production ResponseWrapper path for both), or "auto" (DEFAULT: direct
    # unless the run cannot serve it -- a non-WDM or unidentifiable run domain, channels
    # other than XYZ -- then full with one INFO line). resolve_emri_direct_cfg holds the
    # rules; the cross-check stays on the production generator either way.
    likelihood: str = dataclasses.field(
        default_factory=env_default("EMRI_LIKELIHOOD", "auto", str)
    )
    # Rows per direct generation: each row holds a full-grid (3, Nf, Nt) float64
    # accumulator while it is built (~150 MB at Nf 1440 x Nt 4320).
    batch_max_size: int = dataclasses.field(
        default_factory=env_default("EMRI_BATCH_MAX_SIZE", 8, int)
    )
    # The n_ref WDM lookup table on the run's Nf and dt. None (default): the canonical
    # file in the run's folder (general.file_store_dir; on the 6-month grid
    # wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5); a path points to a specific
    # table. Either way it is BUILT AND SAVED there when missing (once: restarts find it;
    # lisatools.wdm_lookup_store.ensure_lookup_table).
    direct_table: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("EMRI_DIRECT_TABLE", None, str)
    )
    # TDI-on-the-fly kernel of the direct template: "dense" (TDDenseTDIWaveform: exact
    # dense-output phases, geometry shared across harmonics, all rows in one launch;
    # needs a lisatools build with TDDenseTDIWaveformWrap) or "spline".
    direct_response: str = dataclasses.field(
        default_factory=env_default("EMRI_DIRECT_RESPONSE", "dense", str)
    )
    # Trajectory fan-out: worker PROCESSES integrating a direct batch's inspirals
    # (few.trajectory.pool); 0 integrates serially. Each worker is one CPU core:
    # the launcher's --cpus-per-task must cover them.
    traj_workers: int = dataclasses.field(
        default_factory=env_default("EMRI_TRAJ_WORKERS", 0, int)
    )
    # SOURCE WARM START: see SourceMBHSettings (shared store / SNR knobs).
    source_warm_start_store: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_STORE", None, str)
    )
    source_warm_start_snr_min: float = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_SNR_MIN", 10.0, float)
    )
    source_warm_start_snr_unknown: str = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_SNR_UNKNOWN", "refuse", str)
    )
    source_warm_start_ids: typing.Optional[tuple] = dataclasses.field(
        default_factory=env_default("EMRI_SOURCE_WARM_START_IDS", None, parse_id_list)
    )
    # computed by prepare_emri_branch
    injection_ids: typing.Optional[tuple] = None


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
    # Which likelihood scores the add/remove proposals: "lookup" (DEFAULT since
    # 2026-10-03; docs/sobbh-wdm-lookup.md) = SOBBHLookupComputations (per-device
    # replicas behind SOBBHLookupRouter) over a batched TDI-on-the-fly response
    # on a sparse 12-h grid + an n_ref WDM lookup table, scored by the fused
    # sobbh_lookup kernel (36-176x the chunked comp on one H100; lookup vs
    # production template <= 1.2e-6 on the mojito bricks; see the lookup_*
    # fields below); "chunked" = the chunked-heterodyne WDM kernel (the
    # 2026-07-30 default); "full" = the exact full-TD container path (the
    # escape hatch — required for FD/STFT domains, DCGA). All three drive the
    # same SOBBHChunkedLikeMove except "full".
    # The residual expose/fold stays on the exact generator either way, and
    # the built-in fast-vs-slow cross-check stays on (thin it with
    # SOBBH_CHECK_LL_EVERY=10 in production; SOBBH_CHECK_LL=0 disables).
    likelihood: str = dataclasses.field(
        default_factory=env_default("SOBBH_LIKELIHOOD", "lookup", str)
    )
    # chunked-path knobs (see lisatools.chunked_het.WDMComputationsBase).
    # Nt_sub errs SMALL (short chunks): the per-chunk heterodyne collapses
    # a source whose intra-chunk frequency sweep exceeds the band (measured
    # on full_year-lite: Nt_sub=256 -> 2 chunks -> template plateaus at the
    # chunk carriers). SOBBH scoring is cheap, so safety wins over the
    # per-chunk FFT amortization.
    # 0 = AUTO (2026-08-25): size from the branch's own injected sources --
    # the largest divisor-of-Nt chunk whose worst-case intra-chunk sweep
    # (leading-order fdot at WINDOW END, band-capped for sources chirping
    # out of band) stays below ``sweep_frac`` of a WDM layer. An EXPLICIT
    # value is honored but checked against the same criterion at comp
    # build, with a loud warning naming the safe value (the collapse is
    # otherwise SILENT -- the template just parks on per-chunk plateaus).
    # At 6 mo the worst catalogue chirper (id 0, 15.27 mHz, 1.6 yr from
    # merger) allows nt_sub <= ~49 -> the 32 default is safe; at 1 yr the
    # same source needs <= ~21 -> 32 is NOT, which is why auto exists.
    nt_sub: int = dataclasses.field(
        default_factory=env_default("SOBBH_NT_SUB", 32, int)
    )
    # AUTO/check criterion: target intra-chunk carrier sweep in WDM
    # LAYERS per chunk. MEASURED (6-mo removal-null sweep, 2026-08-25,
    # scripts/sobbh/sobbh_removal_null_6mo.py): the removal-residual
    # curve vs Nt_sub is U-SHAPED -- short chunks pay stitch/edge
    # overhead (13x worse at Nt_sub=8 than 16), long chunks shed chirped
    # power -- with its minimum at sweep ~3.5 layers/chunk (residual
    # fraction 5.6e-4), tolerable to ~7 with fill half-width >= 4
    # (<=2e-3), hard collapse by ~14 at any fill width. 3.5 targets the
    # measured optimum.
    sweep_max_layers: float = dataclasses.field(
        default_factory=env_default("SOBBH_SWEEP_MAX_LAYERS", 3.5, float)
    )
    # Worst-case leading-order fdot [Hz/s] at the end of the observation
    # window over the branch's injected sources; COMPUTED by
    # prepare_sobbh_branch (never an env knob). None -> auto/check skipped.
    chirp_fdot_max: typing.Optional[float] = None
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
    # ---- "lookup" likelihood (SOBBHLookupComputations: batched TDI-on-the-fly + n_ref WDM
    # lookup table; docs/sobbh-wdm-lookup.md). The table must have the run's layer duration
    # (``Nf * dt``, 3600 s in production); any sampling with that product works.
    lookup_table_path: str = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_TABLE_PATH", "", str)
    )
    # layers each side of the carrier layer per pixel (5 layers for 2; SOBBH |fdot| < 0.01
    # layer units everywhere realistic, so 2 is converged)
    lookup_num_m_layers: int = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_NUM_M_LAYERS", 2, int)
    )
    # response evaluation step [s] for the per-channel amplitude/phase splines: a SPARSE grid
    # (12 h; one point per 12 pixels), ending exactly at the window and kept inside the orbit
    # tables. 1800 .. 86400 s reproduce the 600-s gate to every printed digit at 6 months; the
    # spline error grows as the step to the fourth (24 h = 16x the 12-h error, still 5e-5 rad
    # for the chirpiest in-band case), so 12 h keeps a margin. tests/test_sobbh_sparse_response.py
    lookup_eval_dt: float = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_EVAL_DT", 43200.0, float)
    )
    # table interpolation: "spline" (the table's uniform cubic B-spline; the fused kernels'
    # semantics), "cubic" (Keys) or "linear"
    lookup_interp: str = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_INTERP", "spline", str)
    )
    # rows per tracer / lookup block (bounds their temporaries, ~17 MB per row at 6 months);
    # the response is built once per call for all rows
    lookup_row_batch: int = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_ROW_BATCH", 32, int)
    )
    # the fused C++/CUDA lookup (sobbh_lookup_kernel.cu: response splines -> <d|h>, <h|h> or the
    # fill in one launch): "auto" = when the backend module has it, "kernel" = required,
    # "python" = the Python lookup (tracer / lookup / inner in row blocks)
    lookup_kernel: str = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_KERNEL", "auto", str)
    )
    # SOURCE WARM START: see SourceMBHSettings (shared store / SNR knobs).
    source_warm_start_store: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_STORE", None, str)
    )
    source_warm_start_snr_min: float = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_SNR_MIN", 10.0, float)
    )
    source_warm_start_snr_unknown: str = dataclasses.field(
        default_factory=env_default("SOURCE_WARM_START_SNR_UNKNOWN", "refuse", str)
    )
    source_warm_start_ids: typing.Optional[tuple] = dataclasses.field(
        default_factory=env_default("SOBBH_SOURCE_WARM_START_IDS", None, parse_id_list)
    )
    # computed by prepare_sobbh_branch
    injection_ids: typing.Optional[tuple] = None


# ============================================================
# SOBBH chunk sizing (leading-order chirp vs the WDM layer)
# ============================================================
def sobbh_leading_order_fdot(m1_msun, m2_msun, f_hz):
    """Leading-order (0PN) GW-frequency derivative [Hz/s].

    Sizing/diagnostic helper ONLY -- the likelihood's frequency evolution
    stays with the waveform classes. fdot = (96/5) pi^{8/3} (G Mc/c^3)^{5/3}
    f^{11/3}; conservative-enough for a chunk-size bound (spin corrections
    are percent-level at these frequencies and clamped anyway).
    """
    m1 = np.asarray(m1_msun, dtype=float)
    m2 = np.asarray(m2_msun, dtype=float)
    f = np.asarray(f_hz, dtype=float)
    mc_s = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2 * MTSUN_SI  # seconds
    return (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * mc_s ** (5.0 / 3.0) \
        * f ** (11.0 / 3.0)


def sobbh_fdot_max_at_window_end(full_basis, tobs, f_cap=None):
    """Worst leading-order fdot [Hz/s] over sources at the END of the window.

    ``full_basis`` rows are the 11-param SOBBH INJECTION basis
    (m1, m2, s1, s2, dist, inc, f_low, ra, dec, psi, phi0 -- the order
    ``SOBBHTDIonFlyWaveWrap.raw_td`` unpacks; f_low is COLUMN 6, in Hz,
    quoted at the window start). NB the signal-het comp vector uses a
    DIFFERENT order with f0 at index 5 -- reading column 5 here returns
    the inclination and produced a 6.5x-overstated chirp bound (false
    kappa=23 collapse warning, caught in production 2026-08-27). Each source is evolved to t = tobs via the
    0PN time-to-coalescence relation; a source that would merge in-window
    or chirp past ``f_cap`` (the run's max_freq) is capped at ``f_cap`` --
    the sweep bound then reflects the fastest chirp the band can hold.
    Returns 0.0 for an empty basis (auto/check become no-ops).
    """
    rows = np.atleast_2d(np.asarray(full_basis, dtype=float))
    if rows.size == 0:
        return 0.0
    m1, m2, f0 = rows[:, 0], rows[:, 1], rows[:, 6]
    mc_s = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2 * MTSUN_SI
    # 0PN: tau(f) = (5/256) pi^{-8/3} Mc^{-5/3} f^{-8/3}
    tau0 = (5.0 / 256.0) * np.pi ** (-8.0 / 3.0) * mc_s ** (-5.0 / 3.0) \
        * f0 ** (-8.0 / 3.0)
    tau_end = tau0 - float(tobs)
    f_end = np.where(tau_end > 0.0,
                     f0 * (tau0 / np.maximum(tau_end, 1e-30)) ** 0.375,
                     np.inf)
    if f_cap is not None:
        f_end = np.minimum(f_end, float(f_cap))
    return float(np.max(sobbh_leading_order_fdot(m1, m2, f_end)))


# Measured regime boundaries (6-mo removal-null sweep, 2026-08-25):
# residual fraction <= 2e-3 holds to ~7 layers/chunk with fill >= 4;
# hard collapse (residual ~ 1) by ~14. Small chunks pay stitch/edge
# overhead, so auto never goes below 16 (Nt_sub=8 measured 13x worse
# than 16).
SOBBH_SWEEP_WARN_LAYERS = 7.0
SOBBH_NT_SUB_AUTO_MIN = 16


def resolve_sobbh_nt_sub(nt_sub, fdot_max, Nt, layer_dt, layer_df,
                         sweep_max_layers, log_name="sobbh"):
    """Resolve/validate the chunked-het chunk length against the chirp.

    The intra-chunk carrier sweep ``fdot_max * nt_sub * layer_dt`` (in
    units of a WDM layer ``layer_df``) controls how much chirped power
    the per-chunk heterodyne sheds -- SILENTLY (template parks on
    per-chunk carrier plateaus). MEASURED at 6 mo (removal-null sweep):
    optimum ~3.5 layers/chunk, tolerable to ~7 with a fill half-width
    >= 4, collapse by ~14; chunks shorter than 16 layers lose to
    stitch/edge overhead instead.

    ``nt_sub == 0`` -> AUTO: the largest divisor of ``Nt`` with sweep
    <= ``sweep_max_layers``, clamped to [16, 256]; when even 16 exceeds
    the collapse threshold a WARNING says no chirp-safe chunk exists on
    this grid (widen bands / reconsider the grid -- quadratic in
    layer_dt, so this is a coarse-layer-grid disease). An explicit
    value is returned unchanged but warned about above the collapse
    threshold. ``fdot_max`` None/0 -> nothing to size against: auto
    falls back to 32, explicit values pass silently.
    """
    nt_sub = int(nt_sub)
    if not fdot_max:
        if nt_sub == 0:
            logger.info("%s: SOBBH_NT_SUB=0 (auto) with no chirp bound "
                        "available; falling back to 32.", log_name)
            return 32
        return nt_sub
    sweep_per_layer_chunk = fdot_max * layer_dt / layer_df  # layers per
    #                                            one-layer-long chunk
    if nt_sub > 0:
        kappa = sweep_per_layer_chunk * nt_sub
        if kappa > SOBBH_SWEEP_WARN_LAYERS:
            bound = sweep_max_layers / sweep_per_layer_chunk
            safe = max((d for d in range(SOBBH_NT_SUB_AUTO_MIN,
                                         min(int(bound), 256) + 1)
                        if Nt % d == 0), default=SOBBH_NT_SUB_AUTO_MIN)
            logger.warning(
                "%s: SOBBH_NT_SUB=%d puts the intra-chunk sweep at %.1f "
                "layers/chunk (worst fdot %.3e Hz/s) -- beyond the "
                "measured %.0f-layer tolerance band; the per-chunk "
                "heterodyne sheds chirped power SILENTLY. Use "
                "SOBBH_NT_SUB=0 (auto) or ~%d.", log_name, nt_sub, kappa,
                fdot_max, SOBBH_SWEEP_WARN_LAYERS, safe)
        return nt_sub
    bound = sweep_max_layers / sweep_per_layer_chunk
    auto = max((d for d in range(SOBBH_NT_SUB_AUTO_MIN,
                                 min(int(bound), 256) + 1)
                if Nt % d == 0), default=SOBBH_NT_SUB_AUTO_MIN)
    kappa = sweep_per_layer_chunk * auto
    if kappa > SOBBH_SWEEP_WARN_LAYERS:
        logger.warning(
            "%s: no chirp-safe chunk exists on this grid -- even "
            "Nt_sub=%d sweeps %.1f layers/chunk (> %.0f). This is the "
            "coarse-layer-grid regime (sweep is QUADRATIC in layer_dt); "
            "widen SOBBH_FILL_M_BAND_HALF_WIDTH/SOBBH_M_BAND_HALF_WIDTH "
            "or use a finer-layer grid.", log_name, auto, kappa,
            SOBBH_SWEEP_WARN_LAYERS)
    else:
        logger.info(
            "%s: SOBBH_NT_SUB auto -> %d (Nt=%d, worst fdot %.3e Hz/s, "
            "sweep/chunk %.2f layers <= target %.2f).", log_name, auto,
            Nt, fdot_max, kappa, sweep_max_layers)
    return auto


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


EMRI_MODE_SELECTION_THRESHOLD_KEY = "mode_selection_threshold"


def apply_emri_mode_selection_threshold(emri) -> typing.Optional[float]:
    """Resolve the effective EMRI mode-selection threshold (the ``eps`` knob).

    FEW 2.x spells the mode-selection threshold ``mode_selection_threshold``
    (the old ``eps`` name is now swallowed by ``**kwargs`` with no effect),
    and ``_generate_waveform`` hands its OWN call-time default (1e-5) to the
    selector unconditionally -- the selector falls back to its constructor
    value only when the call-time argument is ``None``. So the constructor
    pin in :mod:`lisatools.sources.emri.response` never applies, and the knob
    has to ride the per-call kwargs to have any effect at all.

    RESOLUTION ONLY -- the branch ``waveform_kwargs`` are NEVER stamped
    (2026-09-15, 6mo production crash). For EMRI that dict doubles as the
    move's likelihood kwargs (``EMRIMoveBuilder.like_kwargs_from_waveform_kwargs``),
    and ``inner_product`` has no ``mode_selection_threshold`` parameter, so
    stamping it killed the run's first likelihood call. Delivery is the wave
    wrap alone: :func:`source_signal_cfg` carries the value returned here into
    ``cfg["emri_mode_selection_threshold"]``, and
    :func:`get_emri_wave_wrap` installs it as the wrap's per-call
    ``runtime_kwargs`` -- the ONE wrap both the engine-side generator
    (:class:`SourceSignalGen`, template + residual builds) and the move-side
    :class:`DeviceLocalWaveGen` resolve through, so it is on every EMRI
    waveform this run builds.

    An explicit ``waveform_kwargs[mode_selection_threshold]`` wins over the
    ``eps`` field; ``eps=None`` means FEW's own default stands. Pure and
    idempotent; returns ``None`` for "FEW default".
    """
    kwargs = emri.waveform_kwargs if emri.waveform_kwargs is not None else {}
    if EMRI_MODE_SELECTION_THRESHOLD_KEY in kwargs:
        return kwargs[EMRI_MODE_SELECTION_THRESHOLD_KEY]
    eps = getattr(emri, "eps", None)
    return None if eps is None else float(eps)


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
        inj_ids = sorted(cat.keys())
        full_basis = np.asarray(
            [emri_catalogue_to_waveform_basis(cat[i]) for i in inj_ids]
        )
    else:
        inj_ids = list(range(n))
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
        # leaf j carries catalogue id inj_ids[j] (source warm start maps by id)
        emri.injection_ids = tuple(int(i) for i in inj_ids)
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
    # NOTE: the mode-selection threshold is deliberately NOT stamped here --
    # see apply_emri_mode_selection_threshold; source_signal_cfg resolves it
    # and the wave wrap delivers it.
    if emri.inner_moves is None:
        resolve_inner_moves(emri)
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

        inj_ids = sorted(cat.keys())
        full_basis = np.asarray(
            [sobbh_catalogue_to_waveform_basis(cat[i]) for i in inj_ids]
        )
    else:
        inj_ids = list(range(n))
        inj_mode, inj_seed = synthetic_injection_mode(gs)
        full_basis = make_sobbh_injections(n, mode=inj_mode, seed=inj_seed)
    if sobbh.injection is None:
        tc = make_sobbh_transform_container()
        sobbh.injection = np.stack(
            [tc.both_inverse_transforms(row) for row in full_basis], axis=0
        )
        # leaf j carries catalogue id inj_ids[j] (source warm start maps by id)
        sobbh.injection_ids = tuple(int(i) for i in inj_ids)
    # Worst end-of-window chirp over THIS branch's sources -- feeds the
    # chunked-het Nt_sub auto/criterion at comp build (band-capped so a
    # source chirping out of band bounds at the run's max_freq).
    if sobbh.chirp_fdot_max is None:
        sobbh.chirp_fdot_max = sobbh_fdot_max_at_window_end(
            full_basis, general_setup.Tobs,
            f_cap=getattr(gs, "max_freq", None),
        )
    if getattr(sobbh, "fill_values", None) is None:
        sobbh.fill_values = np.array([])
    for lims in ("logm1_lims", "logm2_lims", "s1_lims", "s2_lims", "f_low_lims"):
        if not getattr(sobbh, lims):
            setattr(sobbh, lims, None)
    if sobbh.waveform_kwargs is None:
        sobbh.waveform_kwargs = dict()
    if sobbh.inner_moves is None:
        resolve_inner_moves(sobbh)
    sobbh.nleaves_max = n
    sobbh.nleaves_min = n
    return sobbh


def prepare_mbh_branch(mbh, general_setup: GeneralSetup, gs):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from ....sampling.prior import LogUniformLinear

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
    # if its merger falls within the observed data plus a buffer past the
    # end, i.e. ``observation_start <= t_merge < observation_end + buffer``.
    # A merger BEFORE the data start leaves no inspiral in the data (only a
    # minutes-long ringdown), and the ``t_plunge`` prior below starts at the
    # data start, so such a source is dropped rather than admitted outside
    # its own prior. The sampling-basis ``t_plunge`` (injection[:, -1], ==
    # TimeCoalescencePhenomTPHMSSBFrame in mojito mode) and the data window
    # are both referenced to ``gs.mbh_waveform_t0``, so the data spans
    # ``[data_t0 - mbh_waveform_t0, data_t0 - mbh_waveform_t0 + Tobs]`` in
    # that frame.
    buffer = getattr(mbh, "mbh_merger_time_buffer", 0.0)
    obs_start = general_setup.data_t0 - gs.mbh_waveform_t0
    obs_end = obs_start + general_setup.Tobs
    t_merge = np.asarray(injection[:, -1], dtype=float)
    keep = (t_merge >= obs_start) & (t_merge < (obs_end + buffer))
    if not bool(keep.all()):
        dropped = [inj_ids[k] for k in range(len(inj_ids)) if not keep[k]]
        logger.info(
            "MBH merger-window filter: dropping %d/%d MBHB source(s) %s with "
            "t_merge outside [observation_start (%.6e s), observation_end "
            "(%.6e s) + buffer (%.6e s)).",
            len(dropped), len(inj_ids), dropped, obs_start, obs_end, buffer,
        )
        injection = injection[keep]
        inj_ids = [inj_ids[k] for k in range(len(inj_ids)) if keep[k]]
    if injection.shape[0] == 0:
        logger.warning(
            "MBH merger-window filter removed every requested MBHB source; the "
            "MBH branch will have zero leaves (nothing merges inside "
            "[%.6e s, observation_end + buffer = %.6e s)).",
            obs_start, obs_end + buffer,
        )
    if mbh.injection is None:
        mbh.injection = injection
        # leaf j carries catalogue id inj_ids[j] AFTER the merger-window filter
        # (source warm start maps by id, never by leaf index)
        mbh.injection_ids = tuple(int(i) for i in inj_ids)
    # nleaves follow the (possibly filtered) injection, not the requested count.
    n = int(np.asarray(mbh.injection).shape[0])
    if mbh.transform is None:
        mbh.transform = make_mbh_transform_container()
    if mbh.priors is None:
        # t_plunge is sampled relative to the waveform t0 epoch; the data
        # span starts at data_t0 (the filter's lower edge). The upper edge
        # reaches past the data end by the merger-window buffer PLUS
        # t_plunge_pad: a source the filter above admits is inside its own
        # prior with t_plunge_pad of headroom, so the start-walker scatter
        # around a merger just under end + buffer cannot land at -inf.
        t_rel_min = obs_start
        t_plunge_hi = obs_end + float(buffer) + float(mbh.t_plunge_pad)
        mbh.priors = {
            "mbh": ProbDistContainer(
                {
                    "logM": uniform_dist(*mbh.logM_prior),
                    # LINEAR-column log-uniform (Q = m1/m2 >= 1); eryn's
                    # ``log_uniform`` is ln-space -- see LogUniformLinear.
                    "Q": LogUniformLinear(1.0, 10.0),
                    "s1z": uniform_dist(-0.999999, 0.999999),
                    "s2z": uniform_dist(-0.999999, 0.999999),
                    "dist": uniform_dist(*mbh.dist_prior),
                    "phi_ref": uniform_dist(0.0, 2 * np.pi),
                    "cos_iota": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                    "psi": uniform_dist(0.0, np.pi),
                    "alpha": uniform_dist(0.0, 2 * np.pi),
                    "sin_delta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                    "t_plunge": uniform_dist(t_rel_min, t_plunge_hi),
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
        mbh.inner_moves = make_mbh_inner_moves(mbh, general_setup, gs)
    mbh.nleaves_max = n
    mbh.nleaves_min = n
    return mbh


#: Column indices of the sky / orientation / time parameters in the stock MBH
#: sampling basis (``make_mbh_transform_container().input_basis``:
#: ``cos_iota=6, psi=7, alpha=8, sin_delta=9, t_plunge=10``), in the
#: :class:`~lisatools.sampling.moves.skymodehop.SkyMove` key names.
MBH_SKY_MOVE_IND_MAP = dict(cosinc=6, psi=7, lam=8, sinbeta=9, t_ref=10)


def make_mbh_inner_moves(mbh, general_setup, gs) -> list:
    """The MBH add/remove move's inner-move stack: the ``inner_move_kind``
    proposal plus the LISA-frame sky-mode hops.

    The hops are :class:`~lisatools.sampling.moves.skymodehop.SkyMove`
    instances with ``coord_frame="icrs"`` (the stock basis samples
    ``alpha / sin_delta / psi`` in ICRS against ICRS orbits) whose
    constellation phase is read off the run's orbits
    (:class:`~lisatools.sampling.moves.skymodehop.ConstellationPhase`,
    ``t_offset = mbh_waveform_t0``: the sampled ``t_plunge`` is referenced
    to that epoch while the orbit file carries absolute times). Draw
    fractions come from ``mbh.sky_move_{both,lat,long}_weight``; the
    ``inner_move_kind`` stack takes the remainder. Every weight at 0 gives
    the bare stack.
    """
    from lisatools.sampling.moves.skymodehop import ConstellationPhase, SkyMove

    from .common import make_default_inner_moves

    weights = {
        "both": float(getattr(mbh, "sky_move_both_weight", 0.0)),
        "lat": float(getattr(mbh, "sky_move_lat_weight", 0.0)),
        "long": float(getattr(mbh, "sky_move_long_weight", 0.0)),
    }
    for kind, w in weights.items():
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"mbh.sky_move_{kind}_weight must be in [0, 1], got {w}.")
    total = sum(weights.values())
    if total >= 1.0:
        raise ValueError(
            "the MBH sky-move weights must leave room for the in-model stack "
            f"(sum {total} >= 1)."
        )
    base = make_default_inner_moves(getattr(mbh, "inner_move_kind", "eigen"))
    base_total = float(sum(w for _, w in base))
    moves = [(move, (1.0 - total) * w / base_total) for move, w in base]
    if total == 0.0:
        logger.info(
            "MBH sky-mode hops DISABLED (sky_move_both/lat/long_weight all 0); "
            "inner-move stack = the bare %s move.",
            getattr(mbh, "inner_move_kind", "eigen"),
        )
        return moves

    transform = getattr(mbh, "transform", None)
    basis = list(getattr(transform, "input_basis", []) or [])
    if basis:
        expect = {"cosinc": "cos_iota", "psi": "psi", "lam": "alpha", "sinbeta": "sin_delta", "t_ref": "t_plunge"}
        got = {k: basis[i] if i < len(basis) else None for k, i in MBH_SKY_MOVE_IND_MAP.items()}
        if got != expect:
            raise ValueError(
                "the MBH sky-mode hops assume the stock ICRS sampling basis "
                f"{expect} at columns {MBH_SKY_MOVE_IND_MAP}; this transform has {got}."
            )

    orbits = getattr(general_setup, "orbits", None)
    t_offset = float(getattr(gs, "mbh_waveform_t0", 0.0))
    if orbits is not None:
        phase = ConstellationPhase.from_orbits(orbits, t_offset=t_offset)
        logger.info(
            "MBH sky-mode hops: constellation phase table from %s (%s frame) over "
            "[%.6e, %.6e] s, t_offset = %.6e s; weights both %.3f / lat %.3f / long %.3f, "
            "%s stack %.3f.",
            type(orbits).__name__, getattr(orbits, "frame", "?"), phase.t_grid[0],
            phase.t_grid[-1], t_offset, weights["both"], weights["lat"], weights["long"],
            getattr(mbh, "inner_move_kind", "eigen"), 1.0 - total,
        )
    else:
        phase = None
        logger.warning(
            "MBH sky-mode hops: the general setup carries no orbits; falling back to "
            "the analytic constellation phase alpha = Omega * t (the data-start phase "
            "is NOT known, so the hops land off the symmetric modes by the orbit's "
            "phase at the epoch)."
        )
    for which in ("both", "lat", "long"):
        if weights[which] > 0.0:
            moves.append(
                (
                    SkyMove(
                        ind_map=dict(MBH_SKY_MOVE_IND_MAP),
                        which=which,
                        coord_frame="icrs",
                        constellation_phase=phase,
                    ),
                    weights[which],
                )
            )
    return moves


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
def run_domain_settings_class(domain_settings):
    """The :class:`~lisatools.domains.DomainSettingsBase` subclass a run-domain
    SPEC resolves to, or ``None`` when it cannot be known before the build.

    ``domain_settings`` is what ``general.domain_settings`` holds: a settings
    instance (its class), a ``make_factory`` factory (the class it declares as
    ``domain_settings_class``), a hand-written factory (``None``: unknown), or
    ``None``."""
    from lisatools.domains import DomainSettingsBase

    if isinstance(domain_settings, DomainSettingsBase):
        return type(domain_settings)
    cls = getattr(domain_settings, "domain_settings_class", None)
    return cls if isinstance(cls, type) else None


def run_domain_spec(general_info):
    """The run-domain SPEC a built ``general_info`` was configured with.

    The injection sites resolve ``MBH_LIKELIHOOD=auto`` from the resolved
    general settings' ``domain_settings`` (a factory before the build);
    :class:`~lisatools.globalfit.engine.GeneralSetup` keeps exactly that
    object as ``.settings`` and replaces its own ``domain_settings`` with the
    built instance. Returning the SPEC makes the moves' resolution see what
    the injection saw (identical even for a hand-written factory, which is
    ``unknown`` at both sites); the built instance is the fallback."""
    spec = getattr(getattr(general_info, "settings", None), "domain_settings", None)
    return spec if spec is not None else getattr(general_info, "domain_settings", None)


#: ``MBH_LIKELIHOOD`` values; ``auto`` resolves to ``batched`` or ``full``.
MBH_LIKELIHOOD_MODES = ("auto", "batched", "full")


def _mbh_batched_blockers(mbh, domain_settings, *, explicit_batched) -> list:
    """Why the batched path cannot serve this run (empty: it can).

    Shared by ``auto`` (falls back to ``full`` naming the reasons) and an
    explicit ``batched`` (raises with them). A run domain that cannot be
    identified before the build (a hand-written factory) blocks ``auto`` only:
    an explicit ``batched`` is taken at its word and the windowed generator
    checks the BUILT domain itself (:func:`get_mbh_windowed_gen`)."""
    from lisatools.domains import WDMSettings

    before = float(mbh.window_before_days) * 86400.0
    out = []
    if bool(mbh.use_tdionfly):
        out.append(
            "USE_TDIONFLY=1 (the batched path is the legacy-response grid-aligned "
            "generator)"
        )
    cls = run_domain_settings_class(domain_settings)
    if cls is None:
        if not explicit_batched:
            out.append(
                "the run domain cannot be identified before the build "
                f"(general.domain_settings is {type(domain_settings).__name__}; a "
                "factory must declare domain_settings_class, as the make_factory "
                "ones do)"
            )
    elif not issubclass(cls, WDMSettings):
        out.append(f"the run domain is {cls.__name__}, not WDM")
    explicit = os.environ.get("MBH_WAVEFORM_DURATION")
    duration = mbh.waveform_duration
    if explicit is not None and (duration is None or abs(float(duration) - before) > 1.0):
        out.append(
            f"MBH_WAVEFORM_DURATION={explicit} disagrees with the window: the batched "
            f"path generates {mbh.window_before_days} days before the merger "
            "(MBH_WINDOW_BEFORE_DAYS) for BOTH the stock and the windowed generator"
        )
    return out


def resolve_mbh_batched_cfg(mbh, *, domain_settings) -> dict:
    """Plain-value MBH scoring-path config, with the batched-mode consistency rules.

    ``domain_settings`` is the run-domain SPEC (``general.domain_settings`` of
    the RESOLVED general settings: an instance or a ``make_factory`` factory;
    :func:`run_domain_spec` recovers it from a built ``general_info``). It is
    REQUIRED so every site -- the injections (:func:`mbh_injection_duration`)
    and the moves (:func:`source_signal_cfg`) -- resolves ``auto`` the same way.

    ``auto`` (the default) becomes ``batched`` unless :func:`_mbh_batched_blockers`
    names a reason -- ``USE_TDIONFLY=1``, a non-WDM (or unidentifiable) run
    domain, an ``MBH_WAVEFORM_DURATION`` env value other than the window --
    in which case it becomes ``full`` with ONE INFO line naming the reason(s).
    An explicit ``batched`` raises on the same reasons; ``full`` passes the
    stock values through untouched. The returned ``mbh_likelihood`` is always
    the RESOLVED mode (``batched`` / ``full``), never ``auto``.

    ``batched`` pins BOTH generators (the stock one the engine installs for
    residual rebuilds and the cross-check, and the windowed one) to
    ``waveform_duration = window_before`` so the two agree on the inspiral
    length. The discarded segment pad must also cover ``buffer_time``: the
    response ZEROES the first ``buffer_time`` of the lattice head, so a
    shorter pad would let that dead stretch reach the kept box.
    """
    mode = str(mbh.likelihood)
    if mode not in MBH_LIKELIHOOD_MODES:
        raise ValueError(
            f"MBH_LIKELIHOOD must be one of {MBH_LIKELIHOOD_MODES}; got {mode!r}"
        )
    before = float(mbh.window_before_days) * 86400.0
    after = float(mbh.window_after_days) * 86400.0
    pad = float(mbh.window_pad_days) * 86400.0
    margin = float(mbh.window_margin_days) * 86400.0
    duration = mbh.waveform_duration
    decimate = int(getattr(mbh, "window_decimate", 1))
    if decimate < 1:
        raise ValueError(f"MBH_WINDOW_DECIMATE must be >= 1; got {decimate}")
    if mode != "full":
        blockers = _mbh_batched_blockers(
            mbh, domain_settings, explicit_batched=(mode == "batched")
        )
        if blockers and mode == "batched":
            raise ValueError(
                "MBH_LIKELIHOOD=batched cannot serve this run: " + "; ".join(blockers)
                + ". Fix the conflict or use MBH_LIKELIHOOD=auto/full."
            )
        if blockers:
            logger.info(
                "MBH_LIKELIHOOD=auto -> full: %s.", "; ".join(blockers)
            )
            mode = "full"
        else:
            mode = "batched"
    if mode == "batched":
        explicit = os.environ.get("MBH_WAVEFORM_DURATION")
        if explicit is None and (
            duration is None
            or (
                abs(float(duration) - MBH_DEFAULT_WAVEFORM_DURATION) > 1.0
                and abs(float(duration) - before) > 1.0
            )
        ):
            # Set programmatically (not the default, not the window): the
            # window wins, but say so -- the caller asked for something else.
            logger.warning(
                "MBH_LIKELIHOOD=batched: mbh.waveform_duration=%r is overridden by "
                "the window (MBH_WINDOW_BEFORE_DAYS=%s -> %.1f s) for both the stock "
                "and the windowed generator.",
                duration, mbh.window_before_days, before,
            )
        buffer_time = float(mbh.buffer_time or 0.0)
        if pad < buffer_time:
            raise ValueError(
                f"MBH_WINDOW_PAD_DAYS={mbh.window_pad_days} ({pad:.0f} s) is shorter "
                f"than the MBH response buffer_time ({buffer_time:.0f} s), which the "
                "response zeroes at the lattice head; the discarded pad must cover it."
            )
        duration = before
    return dict(
        mbh_likelihood=mode,
        mbh_batch_max_size=int(mbh.batch_max_size),
        mbh_window_before=before,
        mbh_window_after=after,
        mbh_window_pad=pad,
        mbh_window_margin=margin,
        mbh_waveform_duration=duration,
        # the batched path only; 1 on the full path (nothing decimates there)
        mbh_window_decimate=decimate if mode == "batched" else 1,
    )


def mbh_injection_duration(mbh, *, domain_settings):
    """``waveform_duration`` an MBH INJECTION must be generated with.

    The resolved value from :func:`resolve_mbh_batched_cfg` (same required
    run-domain spec, so ``auto`` resolves here exactly as it does for the
    moves): the raw ``mbh.waveform_duration`` on the ``full`` path, the
    window length (``window_before``) on the ``batched`` path, so the
    injected signal and the templates agree on the inspiral length.
    """
    return resolve_mbh_batched_cfg(mbh, domain_settings=domain_settings)[
        "mbh_waveform_duration"
    ]


def snap_waveform_t0_to_lattice(waveform_t0: float, data_t0: float, dt: float):
    """``(waveform_t0_snapped, snap)`` with the snapped epoch on ``data_t0 + k*dt``.

    ``snap = waveform_t0_snapped - waveform_t0``; callers subtract it from
    every ``t_plunge`` so absolute merger times are unchanged (PR #82 remedy
    for ``GridAlignedPhenomTHMTDIWaveform._check_alignable``).
    """
    offset = float(waveform_t0) - float(data_t0)
    k = int(np.rint(offset / float(dt)))
    snapped = float(data_t0) + k * float(dt)
    return snapped, snapped - float(waveform_t0)


#: ``EMRI_LIKELIHOOD`` values.
EMRI_LIKELIHOOD_MODES = ("full", "direct", "auto")
#: ``EMRI_DIRECT_RESPONSE`` values (``EMRIDirectWDM(response=...)``).
EMRI_DIRECT_RESPONSES = ("dense", "spline")


def resolve_emri_direct_cfg(emri, *, domain_settings, tdi_chan) -> dict:
    """Plain-value EMRI scoring-path config, with the direct-mode consistency rules.

    ``domain_settings`` is the run-domain SPEC (as for :func:`resolve_mbh_batched_cfg`)
    and ``tdi_chan`` the run's channel set. ``full`` passes through untouched.
    ``direct`` RAISES when the run cannot serve it: a run domain that is known not to
    be WDM (an unidentifiable factory is taken at its word; the getter checks the BUILT
    domain), channels other than XYZ (the direct template is the X, Y, Z response), an
    unknown ``EMRI_DIRECT_RESPONSE``, ``EMRI_BATCH_MAX_SIZE < 1`` or
    ``EMRI_TRAJ_WORKERS < 0``. ``auto`` (the default) becomes ``direct`` unless one of
    those reasons -- or an unidentifiable run domain -- applies, then ``full`` with ONE
    INFO line naming them (as ``MBH_LIKELIHOOD=auto``). The returned ``emri_likelihood``
    is the RESOLVED mode, never ``auto``. The lookup table is not required to exist: the
    getter finds it (``EMRI_DIRECT_TABLE`` or the run folder's canonical file) or builds
    and saves it.
    """
    from lisatools.domains import WDMSettings

    # getattr defaults: a plain EMRISettings block (no scoring-path fields) is "full"
    mode = str(getattr(emri, "likelihood", "full"))
    if mode not in EMRI_LIKELIHOOD_MODES:
        raise ValueError(
            f"EMRI_LIKELIHOOD must be one of {EMRI_LIKELIHOOD_MODES}; got {mode!r}"
        )
    table = getattr(emri, "direct_table", None)
    out = dict(
        emri_likelihood=mode,
        emri_batch_max_size=int(getattr(emri, "batch_max_size", 8)),
        emri_direct_table=table,
        emri_direct_response=str(getattr(emri, "direct_response", "dense")),
        emri_traj_workers=int(getattr(emri, "traj_workers", 0)),
    )
    if mode == "full":
        return out
    blockers = []
    cls = run_domain_settings_class(domain_settings)
    if cls is not None and not issubclass(cls, WDMSettings):
        blockers.append(f"the run domain is {cls.__name__}, not WDM")
    if cls is None and mode == "auto":
        blockers.append("the run domain is not identifiable as WDM")
    if str(tdi_chan).upper() != "XYZ":
        blockers.append(f"the run's channels are {tdi_chan!r}; the direct template is XYZ")
    if out["emri_direct_response"] not in EMRI_DIRECT_RESPONSES:
        blockers.append(
            f"EMRI_DIRECT_RESPONSE={out['emri_direct_response']!r} is not one of "
            f"{EMRI_DIRECT_RESPONSES}"
        )
    if out["emri_batch_max_size"] < 1:
        blockers.append(f"EMRI_BATCH_MAX_SIZE={out['emri_batch_max_size']} < 1")
    if out["emri_traj_workers"] < 0:
        blockers.append(f"EMRI_TRAJ_WORKERS={out['emri_traj_workers']} < 0")
    if blockers and mode == "auto":
        logger.info("EMRI_LIKELIHOOD=auto -> full: %s.", "; ".join(blockers))
        out["emri_likelihood"] = "full"
        return out
    if blockers:
        raise ValueError(
            "EMRI_LIKELIHOOD=direct cannot serve this run: " + "; ".join(blockers)
            + ". Fix it or use EMRI_LIKELIHOOD=full."
        )
    out["emri_likelihood"] = "direct"
    return out


def source_signal_cfg(gs, mbh, sobbh, emri, *, domain_settings) -> dict:
    """Plain-value config consumed by the wave-wrap getters below.

    ``domain_settings``: the run-domain SPEC, for the MBH and EMRI scoring-path
    resolutions (:func:`resolve_mbh_batched_cfg`, :func:`resolve_emri_direct_cfg`);
    variants pass ``run_domain_spec(self.general_info)`` so it matches the injection
    sites' (``general.domain_settings`` of the resolved general settings)."""
    _mbh_batched = resolve_mbh_batched_cfg(mbh, domain_settings=domain_settings)
    _emri_direct = resolve_emri_direct_cfg(
        emri, domain_settings=domain_settings, tdi_chan=gs.tdi_chan
    )
    return dict(
        **_mbh_batched,
        **_emri_direct,
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
        # Effective FEW mode-selection threshold (the ``eps`` knob). Carried
        # here so EVERY call through the EMRI wave wrap gets it -- including
        # the engine-side template/residual generation, which does not pass
        # the branch ``waveform_kwargs``.
        emri_mode_selection_threshold=apply_emri_mode_selection_threshold(emri),
        sobbh_response_order=sobbh.response_order,
        sobbh_n_grid=sobbh.n_grid,
        sobbh_buffer_time=sobbh.buffer_time,
        sobbh_likelihood=sobbh.likelihood,
        sobbh_nt_sub=sobbh.nt_sub,
        sobbh_sweep_max_layers=sobbh.sweep_max_layers,
        # NB: chirp_fdot_max is COMPUTED by prepare_sobbh_branch on the
        # PREPARED settings copy -- callers must pass that copy
        # (source_info["sobbh"].settings), not the fit-level block, or
        # this reads None and the criterion silently disengages (found
        # by the 6-mo removal-null run).
        sobbh_chirp_fdot_max=sobbh.chirp_fdot_max,
        sobbh_n_sparse=sobbh.n_sparse,
        sobbh_n_pad=sobbh.n_pad,
        sobbh_m_band_half_width=sobbh.m_band_half_width,
        sobbh_fill_m_band_half_width=sobbh.fill_m_band_half_width,
        sobbh_lookup_table_path=sobbh.lookup_table_path,
        sobbh_lookup_num_m_layers=sobbh.lookup_num_m_layers,
        sobbh_lookup_eval_dt=sobbh.lookup_eval_dt,
        sobbh_lookup_interp=sobbh.lookup_interp,
        sobbh_lookup_row_batch=sobbh.lookup_row_batch,
        sobbh_lookup_kernel=str(getattr(sobbh, "lookup_kernel", "auto")),
        mbh_phenom_kwargs=dict(
            waveform_duration=_mbh_batched["mbh_waveform_duration"],
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
    # The mode-selection threshold is part of the key: it changes the template
    # (kept harmonics), so a wrap built at one value must never be handed back
    # for another.
    mode_threshold = cfg.get("emri_mode_selection_threshold")
    key = ("emri", id(general_info), cfg["nchannels"], dev, mode_threshold)
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
            # Per-call FLOOR for the mode-selection threshold: FEW ignores
            # its selector's constructor value whenever the call supplies one
            # (and its own call default, 1e-5, always does), so the only way
            # the knob bites is through the call kwargs. Unset -> empty dict,
            # i.e. byte-identical to the pre-knob path. An explicit per-call
            # entry still wins (``EMRIWaveWrap.__call__`` updates over this).
            runtime_kwargs=(
                {EMRI_MODE_SELECTION_THRESHOLD_KEY: mode_threshold}
                if mode_threshold is not None
                else None
            ),
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
    waveform_t0, snap = cfg["mbh_waveform_t0"], 0.0
    if cfg.get("mbh_likelihood", "full") == "batched":
        # The batched move's windowed generator runs on the lattice-snapped
        # epoch; give the stock generator (engine residual rebuilds AND the
        # move's cross-check) the SAME epoch, t_plunge shifted by the snap.
        # Measured (mojito id 17, SNR 1420, 2026-09-30): against the
        # UNSNAPPED stock the near-truth rows differ by up to 0.59 nats; the
        # snapped stock agrees to 1.5e-3. A decimated window snaps onto the
        # decimation*dt lattice (a subset of the data lattice); the same
        # decimation factor here keeps the two epochs identical.
        waveform_t0, snap = snap_waveform_t0_to_lattice(
            waveform_t0, general_info.data_t0,
            general_info.dt * int(cfg.get("mbh_window_decimate", 1)),
        )
    gen = get_mbh_phenom_wave_gen(
        data_td_settings=general_info.data_td_settings,
        waveform_t0=waveform_t0,
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
    if snap == 0.0:
        return gen
    key = ("mbh_snapped_stock", id(gen), snap)
    if key not in _WAVE_WRAP_CACHE:
        _WAVE_WRAP_CACHE[key] = SnappedEpochMBHGen(gen, snap)
    return _WAVE_WRAP_CACHE[key]


class SnappedEpochMBHGen:
    """Stock MBH generator built on the lattice-SNAPPED epoch, fed stock rows.

    ``gen`` was built with ``waveform_t0 = stock epoch + snap``; the rows'
    ``t_plunge`` (waveform-basis column 10 -- the LAST positional, which
    ``compute_tdi_channels`` peels as ``merger_time`` -- or the
    ``merger_time`` keyword) is relative to the STOCK epoch, so it is shifted
    by ``-snap`` here: the same absolute merger, evaluated on the data
    lattice exactly like the batched windowed generator (MBH_LIKELIHOOD=
    batched only). ``waveform_t0`` reports the STOCK epoch the rows are
    relative to; every other attribute forwards to ``gen``.
    """

    def __init__(self, gen, snap):
        self.gen = gen
        self.t_plunge_snap = float(snap)

    @property
    def waveform_t0(self):
        return float(self.gen.waveform_t0) - self.t_plunge_snap

    def _shift(self, args, kwargs):
        if kwargs.get("merger_time") is not None:
            kwargs = dict(kwargs)
            kwargs["merger_time"] = kwargs["merger_time"] - self.t_plunge_snap
            return args, kwargs
        if not args:
            raise TypeError("SnappedEpochMBHGen: no t_plunge (positional or merger_time=)")
        args = list(args)
        args[-1] = args[-1] - self.t_plunge_snap
        return tuple(args), kwargs

    def get_signals_for_residuals(self, *args, **kwargs):
        args, kwargs = self._shift(args, kwargs)
        return self.gen.get_signals_for_residuals(*args, **kwargs)

    def compute_tdi_channels(self, *args, **kwargs):
        args, kwargs = self._shift(args, kwargs)
        return self.gen.compute_tdi_channels(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        args, kwargs = self._shift(args, kwargs)
        return self.gen(*args, **kwargs)

    def __getattr__(self, name):
        # dunder/underscore guard (deepcopy/pickle probing, pre-__init__)
        if name.startswith("_") or name == "gen":
            raise AttributeError(name)
        return getattr(self.gen, name)


def get_mbh_windowed_gen(general_info, cfg):
    """Per-device (cached) windowed grid-aligned MBH generator in its sub-transform adapter.

    Built like :func:`get_mbh_phenom_gen` (device-local orbits and domain
    settings; same generator kwargs as
    :func:`~.wrappers.get_mbh_phenom_wave_gen`) but from
    :class:`WindowedGridAlignedMBHWaveform`, with ``waveform_t0`` snapped onto
    the data lattice; the adapter carries ``waveform_t0`` (snapped),
    ``t_plunge_snap`` and ``t0_abs = general_info.data_t0`` (the absolute time
    of WDM layer 0) for the move.
    """
    from lisatools.domains import TDSettings, WDMSettings
    from lisatools.sources.batching import MBHWindowedWDMSignalGen
    from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

    xp = _general_info_xp(general_info)
    primary = _primary_device(general_info)
    base_orbits = (
        general_info.gpu_orbits if general_info.gpus is not None else general_info.orbits
    )
    orbits = _device_local_orbits(base_orbits, xp, primary)
    wdm = _device_local_domain_settings(general_info.domain_settings, xp, primary)
    pk = cfg["mbh_phenom_kwargs"]
    decimation = int(cfg.get("mbh_window_decimate", 1))   # MBH_WINDOW_DECIMATE
    # Device-local orbits/settings key the device; the cfg values that shape
    # the generator key the configuration (a cfg change must not be served a
    # stale generator).
    key = (
        "mbh_windowed", id(general_info), id(orbits), id(wdm), cfg["nchannels"],
        cfg["mbh_waveform_t0"], decimation,
        tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                     for k, v in pk.items())),
    )
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    if not isinstance(wdm, WDMSettings):
        raise ValueError(
            "MBH_LIKELIHOOD=batched needs a WDM run domain "
            f"(general.domain_settings is {type(wdm).__name__})."
        )
    # MBH_WINDOW_DECIMATE: the windowed generator samples at decimation*dt (its
    # own TD settings), its epoch snapped onto that lattice -- the stock
    # generator (get_mbh_phenom_gen) snaps with the same factor, so both share it.
    lattice_dt = float(general_info.dt) * decimation
    t0_snapped, snap = snap_waveform_t0_to_lattice(
        cfg["mbh_waveform_t0"], general_info.data_t0, lattice_dt
    )
    if snap != 0.0:
        logger.info(
            "[MBH_BATCH] waveform_t0 snapped onto the data lattice by %+.6f s "
            "(t_plunge rows are shifted by the same amount inside the move)", snap,
        )
    if decimation == 1:
        gen_td = general_info.data_td_settings
    else:
        td = general_info.data_td_settings
        if int(td.N) % decimation:
            raise ValueError(
                f"MBH_WINDOW_DECIMATE={decimation} does not divide the data length N={td.N}")
        gen_td = TDSettings(int(td.N) // decimation, lattice_dt, t0=float(td.t0),
                            force_backend=general_info.force_backend)
        logger.info(
            "[MBH_BATCH] window lattice decimated %dx: generator, response and segment "
            "transform at %.3g s on Nf/%d layers (MBH_WINDOW_DECIMATE)", decimation,
            lattice_dt, decimation,
        )
    gen = WindowedGridAlignedMBHWaveform(
        waveform_kwargs=dict(
            higher_modes=list(pk["higher_modes"]), include_negative_modes=True,
            t_low_fit=True, coarse_grain=False, atol=pk["phenom_tol"], rtol=pk["phenom_tol"],
        ),
        Tobs=float(pk["waveform_duration"]),
        start_freq=pk["start_freq"],
        use_reference_time=True,
        waveform_t0=t0_snapped,
        data_td_settings=gen_td,
        tdi_generation=cfg["tdi_gen_str"],
        tdi_channels=cfg["tdi_chan"],
        sampling_frequency=1.0 / lattice_dt,
        orbits=orbits,
        order=pk["response_order"],
        tukey_alpha=general_info.window_alpha,
        stft_dt=None,
        freq_min=pk["min_freq"],
        freq_max=pk["max_freq"],
        # Only chunks the FULL-grid rfft loop, which the windowed adapter never
        # uses (it calls compute_tdi_channels + its own segment transform).
        fft_batch_size=1,
        buffer_time=pk["buffer_time"],
        output_domain_settings=wdm,
        force_backend=general_info.force_backend,
    )
    # t0_abs: the data start is the ABSOLUTE time of WDM layer 0. NOT
    # ``wdm.t0``: ``WDMSettings.make_factory`` builds t0 = 0 (it ignores
    # times[0]) and a GB comp build later sets ``domain_settings.t0 =
    # data_t0`` in place (recipe.py GBFillGlobalSignalGen._comp, gb_no_fg /
    # vgb setup), so that t0 depends on build order.
    adapter = MBHWindowedWDMSignalGen(
        gen, wdm, nchannels=cfg["nchannels"], tukey_alpha=general_info.window_alpha,
        t0_abs=float(general_info.data_t0), decimate=decimation,
    )
    adapter.waveform_t0 = t0_snapped
    adapter.t_plunge_snap = snap
    _WAVE_WRAP_CACHE[key] = adapter
    return adapter


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
                # direct mode: the ENGINE-side template generator is the direct-to-WDM
                # template too, so residual rebuilds, the move's expose/fold and its
                # scoring all use one template family (as SOBBH's chunked mode does); the
                # move's cross-check and its fallback stay on the production wrap, which
                # it carries as its own ``waveform_gen``.
                if self.cfg.get("emri_likelihood", "full") == "direct":
                    return get_emri_direct_gen(self.general_info, self.cfg)(
                        *params_in, **kwargs
                    )
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
                if self.cfg.get("sobbh_likelihood", "full") in SOBBH_FAST_LIKELIHOODS:
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
class _LateBoundWaveGenMethod:
    """A :class:`DeviceLocalWaveGen` method resolved at CALL time.

    ``DeviceLocalWaveGen.__getattr__`` used to hand back
    ``getattr(self._resolve(), name)`` — a bound method of whatever device
    replica existed AT ATTRIBUTE-ACCESS TIME. That is per-call dispatch only
    for a consumer that re-accesses the attribute on every call; a consumer
    that captures it ONCE gets the primary device's generator forever.

    ``recipe.py::build_mbh_moves_phenom`` is exactly that consumer:
    ``MBHMoveBuilder(wave_gen=wave_gen.get_signals_for_residuals)`` runs at
    recipe-build time on the main thread, hence on ``gpus[0]``, and the move
    keeps the handle for the life of the run
    (``addremovemove.py`` ``self.waveform_gen``). Under walker sharding a
    worker thread on ``gpus[1]``
    (``AnalysisContainerArray._vectorized_dispatch``) then generated its
    template through the gpu-0 generator, whose cached device arrays — the
    WDM analysis ``window`` and the ``fold_shift_map`` gather indices read in
    ``domains.py::FDSignal.wdmtransform`` — met a gpu-1 residual.
    ``_coerce_transform_backend`` compares array MODULES, never devices, so
    it waves that through: a peer-access tax where P2P is enabled, and an
    asynchronous ``cudaErrorIllegalAddress`` where it is not, surfacing at
    the next allocation (the 2026-09-16 nogb-null ``mbh_pe`` crash).

    EMRI and SOBBH pass the ``DeviceLocalWaveGen`` OBJECT itself (see the
    build_* functions below), so their ``__call__`` re-resolved every time —
    which is why only the MBH block crashed, after EMRI and SOBBH had run
    for ~40 minutes on the same two-GPU sharding.

    ``__self__`` / ``__name__`` are preserved because
    ``recipe.py::MoveBuilder.build`` duck-types the handle as a bound method
    to recover a ``.kwargs``-bearing generator for the per-device DCGA
    replica path; that lookup resolves on the current device, which is what
    that path wants.
    """

    __slots__ = ("_owner", "_method_name")

    def __init__(self, owner: "DeviceLocalWaveGen", name: str):
        self._owner = owner
        self._method_name = name

    @property
    def __self__(self):
        """The generator replica for the CURRENT device (DCGA duck-type)."""
        return self._owner._resolve()

    @property
    def __name__(self):
        return self._method_name

    def __call__(self, *args, **kwargs):
        return getattr(self._owner._resolve(), self._method_name)(*args, **kwargs)

    def __repr__(self):
        return (
            f"<late-bound {self._method_name} of "
            f"{type(self._owner).__name__}>"
        )


class DeviceLocalWaveGen:
    """Per-call, per-device dispatch for MOVE-side waveform generators.

    The move builders used to CAPTURE one wrap instance at build time.
    Under multi-GPU threaded dispatch (``_run_per_split``: one thread per
    device split, each inside its own ``cuda.Device`` context) both
    threads then drove that single generator: cross-device peer-access
    traffic, and a DATA RACE on stateful generators — FEW's trajectory
    integrator lost its spline mid-eval and the 2026-08-27 sources probe
    died with dopr853 "between t_min (0.0) and t_max (0.0)".

    Resolving through the getter on EVERY call (and attribute access)
    keys into the device-local ``_WAVE_WRAP_CACHE`` — cupy's current
    device is thread-local — so each split lazily builds and uses its own
    replica, exactly like the engine ``signal_gen`` path already does.
    """

    def __init__(self, getter, general_info, cfg):
        self._getter = getter
        self._general_info = general_info
        self._cfg = cfg

    def _resolve(self):
        return self._getter(self._general_info, self._cfg)

    def __call__(self, *params, **kwargs):
        return self._resolve()(*params, **kwargs)

    def __getattr__(self, name):
        # dunder/underscore guard (deepcopy/pickle probing rule)
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._resolve(), name)
        if callable(attr):
            # Late-bind: a captured handle must follow the CALLER's device,
            # not the device that happened to be current at capture time.
            # See :class:`_LateBoundWaveGenMethod`.
            return _LateBoundWaveGenMethod(self, name)
        return attr


def get_emri_direct_gen(general_info, cfg):
    """Direct-to-WDM EMRI template adapter (``EMRI_LIKELIHOOD=direct``), per device.

    Reuses the FEW generator inside THIS device's production wrap
    (:func:`get_emri_wave_wrap`: one FEW construction per device, its amplitude tables
    are GBs), the same orbits and the same REF epoch / data start, and the same
    mode-selection threshold (``runtime_kwargs``). The template is assembled on the
    run's FULL wavelet grid and cropped to the run domain's active box
    (:class:`~lisatools.sources.emri.direct_signal_gen.EMRIDirectWDMSignalGen`).

    The lookup table is ``EMRI_DIRECT_TABLE`` or, unset, the canonical file in the run's
    folder (``general_info.file_store_dir``); when missing it is built and saved there
    first (one builder under a lock file, the other ranks wait), so a restart finds it.
    """
    from lisatools import wdm_lookup_store
    from lisatools.domains import WDMLookupTable, WDMSettings
    from lisatools.sources.emri.direct_signal_gen import EMRIDirectWDMSignalGen
    from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

    xp, dev, orbits, domain_settings = _wrap_device_and_orbits(general_info)
    mode_threshold = cfg.get("emri_mode_selection_threshold")
    key = (
        "emri_direct", id(general_info), cfg["nchannels"], dev, mode_threshold,
        cfg["emri_direct_table"], cfg["emri_direct_response"],
    )
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    if not isinstance(domain_settings, WDMSettings):
        raise ValueError(
            "EMRI_LIKELIHOOD=direct needs a WDM run domain "
            f"(general.domain_settings is {type(domain_settings).__name__})."
        )
    wrap = get_emri_wave_wrap(general_info, cfg)
    few_gen = wrap.wave_gen.waveform_gen.waveform_generator
    force_backend = general_info.force_backend
    data_t0 = general_info.data_t0
    # the production wrap's phase epoch: catalogue REF (mojito), else the data start
    ref = MOJITO_REFERENCE_TIME if cfg["data_mode"] == "mojito" else data_t0
    Nf, dt = int(domain_settings.Nf), float(domain_settings.data_dt)
    table_path = wdm_lookup_store.lookup_table_path(
        cfg["emri_direct_table"], getattr(general_info, "file_store_dir", None), Nf, dt)
    with device_context(xp, dev):
        status = wdm_lookup_store.ensure_lookup_table(
            table_path, Nf=Nf, dt=dt, force_backend=force_backend)
        logger.info("[EMRI_DIRECT] lookup table %s (%s).", table_path, status)
        table = WDMLookupTable.from_file(table_path, force_backend=force_backend)
        if int(table.Nf) != Nf or abs(float(table.data_dt) - dt) > 1e-9:
            raise ValueError(
                f"EMRI lookup table {table_path} is built for Nf={table.Nf}, "
                f"dt={table.data_dt}; the run grid is Nf={Nf}, dt={dt}."
            )
        full = WDMSettings(
            int(domain_settings.Nf), int(domain_settings.Nt), float(domain_settings.data_dt),
            force_backend=force_backend,
        )
        direct = EMRIDirectWDM(
            few_gen, table, full, orbits=orbits,
            tdi_config=TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend),
            t_start=ref, data_t0=data_t0, force_backend=force_backend,
            response=cfg["emri_direct_response"],
        )
        gen = EMRIDirectWDMSignalGen(
            direct, domain_settings, nchannels=cfg["nchannels"],
            runtime_kwargs=(
                {EMRI_MODE_SELECTION_THRESHOLD_KEY: mode_threshold}
                if mode_threshold is not None else None
            ),
            traj_workers=cfg.get("emri_traj_workers", 0),
        )
    _WAVE_WRAP_CACHE[key] = gen
    return gen


def build_emri_move_runtime(curr, acs, priors, state, cfg):
    """EMRI PE move: :class:`~lisatools.globalfit.moves.EMRIDirectLikeMove` when
    ``cfg["emri_likelihood"] == "direct"`` (scoring through the direct-to-WDM template; the
    residual expose/fold through the containers' installed generator, which in that mode is
    the direct template as well -- :class:`SourceSignalGen`; the cross-check and the fallback
    on the production wrap), else the stock per-row move."""
    wave_gen = DeviceLocalWaveGen(get_emri_wave_wrap, curr.general_info, cfg)
    if cfg.get("emri_likelihood", "full") == "direct":
        direct = DeviceLocalWaveGen(get_emri_direct_gen, curr.general_info, cfg)
        _, moves = EMRIDirectMoveBuilder(
            wave_gen=wave_gen,
            direct_gen=direct,
            batch_max_size=cfg["emri_batch_max_size"],
        ).build(None, curr, acs, priors, state)
        return moves[0]
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
    # Chirp-safe chunk length (2026-08-25): auto-size (SOBBH_NT_SUB=0) or
    # criterion-check an explicit value against the branch's worst
    # end-of-window fdot -- the collapse beyond the bound is silent.
    # WDMSettings carries layer_dt/layer_df directly (the sample step is
    # ``data_dt``, there is no ``dt`` attribute).
    nt_sub = resolve_sobbh_nt_sub(
        cfg["sobbh_nt_sub"], cfg.get("sobbh_chirp_fdot_max"),
        int(domain_settings.Nt), float(domain_settings.layer_dt),
        float(domain_settings.layer_df),
        float(cfg.get("sobbh_sweep_max_layers", 3.5)),
        log_name="sobbh_chunked",
    )
    # Band-width advisories against the resolved sweep (measured 6-mo
    # removal-null): fill half-width < 4 collapses one Nt_sub step early
    # (fill=2 at ~7 layers/chunk -> residual 0.21); the m=1 SCORING band
    # sheds ~15% of <h|h> already at ~3.5 layers/chunk (bias
    # ~0.076*SNR^2) while m=2 recovers 99.9%.
    fd = cfg.get("sobbh_chirp_fdot_max")
    if fd:
        kappa = fd * nt_sub * float(domain_settings.layer_dt) \
            / float(domain_settings.layer_df)
        if kappa > 3.5 and int(cfg.get("sobbh_fill_m_band_half_width", 8)) < 4:
            logger.warning(
                "sobbh_chunked: fill band half-width %d < 4 with a %.1f-"
                "layer intra-chunk sweep -- the removal fill will leave "
                "chirped power in the SHARED residual; raise "
                "SOBBH_FILL_M_BAND_HALF_WIDTH to >= 4.",
                int(cfg.get("sobbh_fill_m_band_half_width", 8)), kappa)
        if kappa > 1.0 and int(cfg.get("sobbh_m_band_half_width", 1)) < 2:
            logger.warning(
                "sobbh_chunked: scoring band half-width %d with a %.1f-"
                "layer intra-chunk sweep biases lnL by ~0.08*SNR^2 "
                "(measured); set SOBBH_M_BAND_HALF_WIDTH=2.",
                int(cfg.get("sobbh_m_band_half_width", 1)), kappa)
    with device_context(xp, dev):
        comp = SOBBHWDMComputations(
            domain_settings,
            t_ref=float(t_ref),
            Nt_sub=nt_sub,
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


#: The SOBBH likelihoods served by a vectorized comp with the ``get_ll_wdm`` /
#: ``fill_global_wdm`` surface (``SOBBHChunkedLikeMove`` + the engine signal gen).
SOBBH_FAST_LIKELIHOODS = ("chunked", "lookup")


def resolve_sobbh_lookup_table(general_info, cfg, *, force_backend=None):
    """``(path, status)`` of the SOBBH lookup table for this run, the EMRI way.

    ``SOBBH_LOOKUP_TABLE_PATH`` set: that file (``status = "explicit"``; it must exist, and
    the caller checks its LAYER duration against the run's -- any ``(Nf, dt)`` with the same
    ``Nf * dt`` serves, the table depends on the layer duration only). Unset: the canonical
    n_ref table of the run folder (``general_info.file_store_dir``; the EMRI recipe, so it is
    the SAME file ``EMRI_LIKELIHOOD=direct`` uses), found there (``"found"``) or built and
    saved there first (``"built"``, one builder under a lock file while other ranks wait,
    ``"waited"``; :mod:`lisatools.wdm_lookup_store`), so a restart never rebuilds it.
    """
    from lisatools import wdm_lookup_store

    domain_settings = general_info.domain_settings
    explicit = str(cfg.get("sobbh_lookup_table_path") or "")
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.exists(path):
            raise ValueError(
                f"SOBBH_LOOKUP_TABLE_PATH={explicit!r} does not exist. Point it at an n_ref "
                f"lookup table with the run's layer duration ({float(domain_settings.layer_dt):g}"
                " s), or unset it to find-or-build the run folder's own table."
            )
        return path, "explicit"
    table_dir = getattr(general_info, "file_store_dir", None)
    if not table_dir:
        raise ValueError(
            "SOBBH_LIKELIHOOD=lookup needs a lookup table: set SOBBH_LOOKUP_TABLE_PATH, or run "
            "with a run folder (general_info.file_store_dir) where the canonical n_ref table is "
            "found or built and saved (lisatools.wdm_lookup_store)."
        )
    Nf, dt = int(domain_settings.Nf), float(domain_settings.data_dt)
    path = wdm_lookup_store.lookup_table_path(None, table_dir, Nf, dt)
    status = wdm_lookup_store.ensure_lookup_table(
        path, Nf=Nf, dt=dt, force_backend=force_backend or general_info.force_backend
    )
    return path, status


def get_sobbh_lookup_comp(general_info, cfg):
    """Build (and cache per device) the ``SOBBHLookupComputations`` for
    ``SOBBH_LIKELIHOOD=lookup``.

    Same ``t_ref`` / ``t_obs_start`` resolution as :func:`get_sobbh_chunked_comp`; the table
    comes from :func:`resolve_sobbh_lookup_table` (``SOBBH_LOOKUP_TABLE_PATH`` or the run
    folder's find-or-build canonical file), loads on the run's backend and must carry the run
    domain's layer duration.
    """
    from lisatools.domains import WDMLookupTable, WDMSettings
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

    # one comp per device (the cache key below carries the CURRENT device and
    # _wrap_device_and_orbits hands device-local orbits / domain settings);
    # multi-GPU walker shards reach the right one through SOBBHLookupRouter
    xp, dev, orbits, domain_settings = _wrap_device_and_orbits(general_info)
    key = ("sobbh_lookup", id(general_info), cfg["nchannels"], dev)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    if not isinstance(domain_settings, WDMSettings):
        raise ValueError(
            "SOBBH_LIKELIHOOD=lookup needs a WDM run domain "
            f"(general.domain_settings is {type(domain_settings).__name__}); "
            "use SOBBH_LIKELIHOOD=full for FD/STFT runs."
        )
    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)
    t_ref = cfg["sobbh_reference_time"]
    if t_ref is None:
        t_ref = general_info.data_t0
    with device_context(xp, dev):
        path, status = resolve_sobbh_lookup_table(general_info, cfg, force_backend=force_backend)
        logger.info("[SOBBH_LOOKUP] lookup table %s (%s).", path, status)
        table = WDMLookupTable.from_file(path, force_backend=force_backend)
        if abs(float(table.layer_dt) - float(domain_settings.layer_dt)) > 1e-6:
            raise ValueError(
                f"SOBBH lookup table {path} has layer duration {float(table.layer_dt):g} s "
                f"(Nf={table.Nf}, dt={table.data_dt}); the run's layer duration is "
                f"{float(domain_settings.layer_dt):g} s (Nf={domain_settings.Nf}, "
                f"dt={domain_settings.data_dt}). Point SOBBH_LOOKUP_TABLE_PATH at a table "
                "with the run's layer duration, or unset it to use the run folder's own."
            )
        comp = SOBBHLookupComputations(
            domain_settings, float(t_ref), table,
            orbits=orbits, tdi_config=tdi_config, tdi_type=cfg["tdi_chan"],
            t_obs_start=float(general_info.data_t0),
            n_grid=cfg["sobbh_n_grid"], buffer_time=cfg["sobbh_buffer_time"],
            eval_dt=float(cfg["sobbh_lookup_eval_dt"]),
            num_m_layers=int(cfg["sobbh_lookup_num_m_layers"]),
            interp=cfg["sobbh_lookup_interp"], row_batch=int(cfg["sobbh_lookup_row_batch"]),
            kernel=str(cfg.get("sobbh_lookup_kernel", "auto")),
            force_backend=force_backend, d_d=0.0,
        )
    _WAVE_WRAP_CACHE[key] = comp
    return comp


def get_sobbh_lookup_router(general_info, cfg):
    """The (cached) :class:`~lisatools.sources.sobbh.wdm_direct.SOBBHLookupRouter` over
    :func:`get_sobbh_lookup_comp`: one lookup comp per device, each built on first use under
    that device's context, every call sent to the current device's (multi-GPU walker shards)."""
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupRouter

    key = ("sobbh_lookup_router", id(general_info), cfg["nchannels"])
    if key not in _WAVE_WRAP_CACHE:
        _WAVE_WRAP_CACHE[key] = SOBBHLookupRouter(
            lambda _dev: get_sobbh_lookup_comp(general_info, cfg),
            general_info.force_backend,
        )
    return _WAVE_WRAP_CACHE[key]


def get_sobbh_fast_comp(general_info, cfg):
    """The vectorized SOBBH comp ``cfg["sobbh_likelihood"]`` selects
    (``"lookup"``, the default: the per-device router; ``"chunked"``)."""
    kind = cfg.get("sobbh_likelihood", "full")
    if kind == "chunked":
        return get_sobbh_chunked_comp(general_info, cfg)
    if kind == "lookup":
        return get_sobbh_lookup_router(general_info, cfg)
    raise ValueError(f"no vectorized SOBBH comp for sobbh_likelihood={kind!r} "
                     f"(expected one of {SOBBH_FAST_LIKELIHOODS})")


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
    key = (
        "sobbh_chunked_gen",
        cfg.get("sobbh_likelihood"),
        id(general_info),
        cfg["nchannels"],
        dev,
    )
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]

    from lisatools.domains import WDMSignal

    from ...moves.sobbhspecialmove import SOBBHChunkedLikeMove

    comp = get_sobbh_fast_comp(general_info, cfg)
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
    wave_gen = DeviceLocalWaveGen(get_sobbh_wave_wrap, curr.general_info, cfg)
    if cfg.get("sobbh_likelihood", "full") in SOBBH_FAST_LIKELIHOODS:
        comp = get_sobbh_fast_comp(curr.general_info, cfg)
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
    """MBH PE move: batched windowed (``cfg["mbh_likelihood"] == "batched"``,
    the resolved ``auto`` default on a WDM legacy-response run), stretch RJ
    move on the tdionfly wrap, or the stock ``build_mbh_moves_phenom`` builder
    around the cached phentax generator."""
    mbh_info = curr.source_info["mbh"]
    if cfg.get("mbh_likelihood", "full") == "batched":
        slow = DeviceLocalWaveGen(get_mbh_phenom_gen, curr.general_info, cfg)
        batched = DeviceLocalWaveGen(get_mbh_windowed_gen, curr.general_info, cfg)
        _, moves = MBHBatchedMoveBuilder(
            wave_gen=slow.get_signals_for_residuals,
            batched_gen=batched,
            batch_max_size=cfg["mbh_batch_max_size"],
            window_before=cfg["mbh_window_before"],
            window_after=cfg["mbh_window_after"],
            window_pad=cfg["mbh_window_pad"],
            window_margin=cfg["mbh_window_margin"],
        ).build(None, curr, acs, priors, state)
        return moves[0]
    if not cfg["mbh_use_tdionfly"]:
        wave_gen = DeviceLocalWaveGen(get_mbh_phenom_gen, curr.general_info, cfg)
        _, move = build_mbh_moves_phenom(
            curr, acs, priors, state, wave_gen=wave_gen, subtract_initial=False
        )
        return move
    wave_gen = DeviceLocalWaveGen(
        get_mbh_tdionfly_wave_wrap, curr.general_info, cfg)
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
    dict (matching the ``mbh_pe`` / ``emri_pe`` / ``sobbh_pe`` stock-move names).

    MBH scoring path (``cfg["mbh_likelihood"]``, ALREADY resolved from the
    ``MBH_LIKELIHOOD=auto`` default by :func:`resolve_mbh_batched_cfg`):
    ``batched`` -- the default wherever the run allows it (WDM domain, legacy
    response, no conflicting ``MBH_WAVEFORM_DURATION``) -- builds
    :class:`~lisatools.globalfit.moves.MBHBatchedLikeMove`: batched
    grid-aligned phentax generation on a per-leaf window with a segment WDM
    transform (``MBHBatchedMoveBuilder``, spec
    ``docs/superpowers/specs/2026-09-29-mbh-batched-windowed-likelihood-design.md``);
    ``full`` scores ONE ROW AT A TIME through
    ``AnalysisContainer.build_template`` (the base
    ``ResidualAddOneRemoveOneMove`` path).
    ``sobbh_pe`` uses the batched chunked-heterodyne kernel.

    ``emri_pe`` (``cfg["emri_likelihood"]``, the RESOLVED mode: ``auto`` -- the default --
    is ``direct`` on a WDM / XYZ run): ``direct`` builds
    :class:`~lisatools.globalfit.moves.EMRIDirectLikeMove`, which scores chunks of rows
    through the direct-to-WDM template (one FEW call per row, one response launch per
    chunk; the expose/fold through the installed generator, the direct template too);
    ``full`` scores per row through the production container path (~1040 ms/row on the
    6-mo probe, job 373).
    """
    stock_moves = {}
    if "mbh" in curr.source_info:
        stock_moves["mbh_pe"] = build_mbh_move_runtime(curr, acs, priors, state, cfg)
    if "emri" in curr.source_info:
        stock_moves["emri_pe"] = build_emri_move_runtime(curr, acs, priors, state, cfg)
    if "sobbh" in curr.source_info:
        stock_moves["sobbh_pe"] = build_sobbh_move_runtime(curr, acs, priors, state, cfg)
    return stock_moves
