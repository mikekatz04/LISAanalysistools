"""Stock variant ``gb_no_fg``: GB-only global fit, no foreground fitting.

The installed version of ``global_fit_input/gb_no_foreground_global_fit_settings.py``:

* **No galactic-foreground branch** — the foreground is not fit.
* **Fixed PSD** — no ``psd`` branch; the sensitivity is a fixed
  :class:`InstrumentNoise` from ``general.fixed_psd_params`` via the
  engine's no-psd-branch path.
* **Frequency band restricted to f > 6 mHz** so the unresolved galactic
  confusion is out of band — no foreground model needed anywhere.
* Data: the mojito L1 GB galaxy (``catalogue["GB"]`` populated so leaves
  can start at true catalogue points, SNR > 3 subset).

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.gb_no_fg(nwalkers=4, file_base_name="my_run")
    fit.gb.center_freq = 8.0e-3        # move the narrow GB band
    fit.recipe.add_move(MoveSpec("rj_fstat_mcmc", branch="gb"), stage="gb_pe")
    fit.build()                        # heavy: loads + pours the data
    fit.run()                          # or hand it to run_global.py / GlobalFit

Every env var the legacy file honored (``TOBS_TARGET``, ``NWALKERS``,
``GB_MODE``, ``GB_DEBUG``, ``CHUNKED_*``, ...) still works: it feeds the
matching knob's default at construction (explicit kwargs win).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import typing
from copy import deepcopy

import numpy as np

from gbgpu.utils.utility import get_fdot

from lisatools.domains import FDSettings, WDMSettings

from ....engine import GeneralSetup, Settings
from ....recipe import (
    build_gb_moves,
    select_gb_injection_subset_by_snr,
    setup_state_for_injection,
    subtract_gb_neighbors_from_data,
)
from ...base import (
    MoveBuildContext,
    MoveSpec,
    RecipeSpec,
    StageSpec,
    env_default,
    env_resolve,
    materialize_recipe,
)
from ..common import tdi_generation_info
from ..fit import EreborFit, EreborGeneralSettings
from ..gb import GBSettings, GBSetup

logger = logging.getLogger(__name__)

_DELTA_SAFE = 1e-5

# GB phase/frequency reference epoch = the mojito catalogue epoch
# (``MOJITO_REFERENCE_TIME`` = TimeReferenceSSBFrame). Validated convention
# from scripts/gb/gb_mojito_match.py (params at REF, phys phi0 = +TrueAnomaly).
GB_MOJITO_T_REF = 97729089.327664

# GB_DEBUG=1 preset: the fast smoke configuration. Seeded through
# ``os.environ.setdefault`` so every knob keeps its usual explicit env
# override — identical semantics to the legacy settings file.
_DEBUG_ENV_PRESET = {
    "TOBS_TARGET": str(3 * 86400.0),  # 3 days
    "NWALKERS": "3",
    "NTEMPS": "2",
    "CHUNKED_NT_SUB": "64",
    "CHUNKED_N_PAD": "8",
    "CHUNKED_N_SPARSE": "64",
    "CHUNKED_N_CP_SIG": "16",
    "CHUNKED_N_CP_ORBIT": "16",
    "GF_NUM_ITER": "4",
}


def _default_phi0_lims():
    return [0.0, 2 * np.pi]


def _default_iota_lims():
    return [0.0 + _DELTA_SAFE, np.pi - _DELTA_SAFE]


def _default_psi_lims():
    return [0.0, np.pi]


def _default_alpha_lims():
    return [0.0, 2 * np.pi]


def _default_delta_lims():
    return [-np.pi / 2.0 + _DELTA_SAFE, np.pi / 2.0 - _DELTA_SAFE]


def _default_gb_periodic():
    # Sampling-basis periodic parameters, keyed by plain names (run.py
    # translates names -> indices through the branch transform).
    return {"gb": {"phi0": 2 * np.pi, "psi": np.pi, "alpha": 2 * np.pi}}


@dataclasses.dataclass
class GBNoFgGBSettings(GBSettings):
    """GB branch block for ``gb_no_fg``: stock GBSettings + this variant's knobs.

    Band-derivation fields (``center_freq``/``n_layers``) and chunked-het
    kernel sizes are resolved into the underlying :class:`GBSettings`
    fields / the runtime likelihood object at build time. Env-backed
    defaults mirror the legacy settings file's env vars.
    """

    # -- parameter limits (prior ranges) --
    A_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [7e-26, 1e-19]
    )
    m_chirp_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.001, 1.0]
    )
    phi0_lims: typing.List[float] = dataclasses.field(default_factory=_default_phi0_lims)
    iota_lims: typing.List[float] = dataclasses.field(default_factory=_default_iota_lims)
    psi_lims: typing.List[float] = dataclasses.field(default_factory=_default_psi_lims)
    alpha_lims: typing.List[float] = dataclasses.field(default_factory=_default_alpha_lims)
    delta_lims: typing.List[float] = dataclasses.field(default_factory=_default_delta_lims)
    # f0_lims / fdot_lims: derived at build from the data band + m_chirp_lims
    # unless set explicitly.

    # -- shape / sampling --
    ndim: int = 8
    nleaves_min: int = 0
    nleaves_max: int = dataclasses.field(default_factory=env_default("GB_NLEAVES_MAX", 100, int))
    num_repeat_proposals: int = dataclasses.field(
        default_factory=env_default("GB_NUM_REPEAT_PROPOSALS", 100, int)
    )
    periodic: dict = dataclasses.field(default_factory=_default_gb_periodic)
    # Phase/frequency reference epoch. None -> resolved at build from the
    # data mode: the mojito catalogue epoch in mojito mode, the synthetic
    # stream start in synthetic mode.
    t0: typing.Optional[float] = None
    extra_buffer: int = 5
    start_freq_ind: int = 0
    # ``oversample=None`` -> auto at build: 2 on FD (halves per-band N at
    # short Tobs), 4 on WDM.
    oversample: typing.Optional[int] = None

    # -- narrow GB band, decoupled from the (wide) data band --
    # Expressed in integer WDM-layer units around ``center_freq`` (the
    # samplable interior layer CONTAINS center_freq; floor, not round).
    center_freq: float = dataclasses.field(
        default_factory=env_default("GB_CENTER_FREQ", 7.5e-3, float)
    )
    n_layers: int = dataclasses.field(default_factory=env_default("GB_N_LAYERS", 3, int))
    # DATA_BAND_LAYERS (memory knob): clip the DATA band — and every
    # per-walker ACA slab — to +-N layers around the GB band. None = full band.
    data_band_layers: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("DATA_BAND_LAYERS", None, int)
    )
    # FD basis only: GB band width in Hz centered on center_freq (None ->
    # the full data band so several bands survive the band-walker trim).
    fd_bandwidth: typing.Optional[float] = dataclasses.field(
        default_factory=env_default("GB_FD_BANDWIDTH", None, float)
    )

    # -- run-mode knobs --
    # mode="pe": leaves start AT the true catalogue points (SNR > 3 subset).
    # mode="search": zero-knowledge start; per-band progressive leaf caps.
    mode: str = dataclasses.field(default_factory=env_default("GB_MODE", "pe", str))
    # start="prior": PE dimensionality without truth seeding.
    start: str = dataclasses.field(default_factory=env_default("GB_START", "truth", str))
    snr_threshold: float = 3.0
    # Focused-central-band mode: subtract neighboring-layer catalogue
    # sources from the data as KNOWN signals.
    subtract_neighbors: bool = dataclasses.field(
        default_factory=env_default("GB_SUBTRACT_NEIGHBORS", False, bool)
    )
    neighbor_window_layers: int = 4

    # -- chunked-het kernel sizes (WDM likelihood; validated defaults) --
    nt_sub: int = dataclasses.field(default_factory=env_default("CHUNKED_NT_SUB", 256, int))
    n_pad: int = dataclasses.field(default_factory=env_default("CHUNKED_N_PAD", 32, int))
    n_sparse: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_SPARSE", 256, int)
    )
    n_cp_sig: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_CP_SIG", 48, int)
    )
    n_cp_orbit: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_CP_ORBIT", 32, int)
    )
    # Sig-het in-model scoring (chunked-het delegate for RJ/fills/swaps).
    sighet_inmodel: bool = dataclasses.field(
        default_factory=env_default("GB_SIGHET_INMODEL", False, bool)
    )
    sighet_nt_layer: int = dataclasses.field(
        default_factory=env_default("SIGHET_NT_LAYER", 64, int)
    )
    sighet_n_sparse_fd: int = dataclasses.field(
        default_factory=env_default("SIGHET_N_SPARSE_FD", 1024, int)
    )
    # NOTE: no __post_init__ here — Setup.__init__ re-runs this dataclass's
    # __init__ on the (non-dataclass) GBSetup instance, which cannot resolve
    # dataclass hooks. Value validation happens in prepare_branch_settings.


@dataclasses.dataclass
class GBNoFgGeneralSettings(EreborGeneralSettings):
    """General block for ``gb_no_fg`` (defaults mirror the legacy file)."""

    min_freq: float = 6e-3  # foreground-free band restriction
    max_freq: float = 2.5e-2
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output_gb_no_fg/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "gb_no_fg_test_2")
    )
    source_types: typing.Tuple[str, ...] = ("GB",)
    fixed_psd_params: typing.Optional[typing.List[float]] = dataclasses.field(
        default_factory=lambda: [15e-12, 3e-15]
    )
    # data_mode="synthetic" injects these GB rows in-process instead of
    # loading the mojito galaxy: ``(num_sources, 9)`` in the GBGPU basis
    # ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``. None -> the
    # stock two-source table (injections.GB_INJECTION_PARAMS).
    gb_injection_params: typing.Optional[typing.Any] = None


class GBNoForegroundGlobalFit(EreborFit):
    """GB-only global fit without foreground fitting (fixed PSD, f > 6 mHz)."""

    option_name = "gb_no_fg"
    description = (
        "GB-only fit on the mojito L1 GB galaxy: fixed PSD, no foreground "
        "branch, f in [6, 25] mHz, WDM chunked-het likelihood."
    )
    general_settings_class = GBNoFgGeneralSettings
    setup_classes = {"gb": GBSetup}

    def __init__(self, **knobs):
        # GB_DEBUG preset FIRST (before any env-backed default resolves), so
        # the debug values seed the environment exactly like the legacy file.
        debug = knobs.pop("debug", None)
        if debug is None:
            debug = env_resolve("GB_DEBUG", False, bool)
        if debug:
            for _knob, _debug_val in _DEBUG_ENV_PRESET.items():
                os.environ.setdefault(_knob, _debug_val)
            # Arm the GB special-move debug instrumentation too (band
            # residual round-trip checks + begin/middle/end band plots under
            # GB_DEBUG_DIR) — consumed inside build_gb_moves from the env.
            os.environ.setdefault("GB_DEBUG", "1")
        super().__init__(**knobs)
        self.debug = bool(debug)

    def apply_debug_preset(self) -> None:
        """Apply the GB_DEBUG smoke preset to the current fields.

        Field-level version of the env preset used at construction (clones
        never re-run ``__init__``, so ``fit(debug=True)`` routes here).
        Explicit env values still win, mirroring the setdefault semantics.
        """
        os.environ.setdefault("GB_DEBUG", "1")
        self.general.tobs_target = env_resolve("TOBS_TARGET", 3 * 86400.0, float)
        self.general.nwalkers = env_resolve("NWALKERS", 3, int)
        self.general.ntemps = env_resolve("NTEMPS", 2, int)
        self.general.num_iterations = env_resolve("GF_NUM_ITER", 4, int)
        self.gb.nt_sub = env_resolve("CHUNKED_NT_SUB", 64, int)
        self.gb.n_pad = env_resolve("CHUNKED_N_PAD", 8, int)
        self.gb.n_sparse = env_resolve("CHUNKED_N_SPARSE", 64, int)
        self.gb.n_cp_sig = env_resolve("CHUNKED_N_CP_SIG", 16, int)
        self.gb.n_cp_orbit = env_resolve("CHUNKED_N_CP_ORBIT", 16, int)
        self.debug = True

    def apply_overrides(self, overrides: dict) -> None:
        debug = overrides.pop("debug", None)
        if debug:
            self.apply_debug_preset()
        elif debug is not None:
            self.debug = False
        super().apply_overrides(overrides)

    # -- default blocks -------------------------------------------------------

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {"gb": GBNoFgGBSettings()}

    def default_recipe(self) -> RecipeSpec:
        # Smoke-validated default: ONLY the prior-based RJ proposal (which
        # carries the in-model repeat blocks internally). Add
        # MoveSpec("rj_fstat_mcmc"/"rj_refit", branch="gb") for the full PE
        # stack, or *_search names in a kind="search" stage.
        return RecipeSpec(
            [
                StageSpec(
                    name="gb_pe",
                    kind="pe",
                    moves=[MoveSpec("rj_prior", branch="gb")],
                    combine_kwargs=dict(verbose=True, share_temperature_control=False),
                )
            ]
        )

    # -- general resolution -----------------------------------------------------

    def _gb_band(self, layer_df: float) -> typing.Tuple[float, float, int]:
        """(GB start_freq, end_freq, k_center) in integer WDM-layer units."""
        gb = self.gb
        k_center = int(math.floor(gb.center_freq / layer_df))
        k_lo = k_center - gb.n_layers // 2
        return k_lo * layer_df, (k_lo + gb.n_layers) * layer_df, k_center

    def adjust_general(self, gs: GBNoFgGeneralSettings) -> None:
        # ACA frequency clipping (memory knob): narrow the DATA band to
        # +-data_band_layers WDM layers around the GB band.
        gb = self.gb
        if gb.data_band_layers is not None:
            layer_df = self.layer_df
            L = int(gb.data_band_layers)
            min_needed = gb.n_layers // 2 + 5
            assert L >= min_needed, (
                f"data_band_layers={L} too narrow: need >= n_layers//2 + 5 = "
                f"{min_needed} to cover the GB band + the 5-layer chunked-het "
                "gating."
            )
            _, _, k_center = self._gb_band(layer_df)
            gs.min_freq = max(gs.min_freq, (k_center - L) * layer_df)
            gs.max_freq = min(gs.max_freq, (k_center + 1 + L) * layer_df)
            logger.info(
                "ACA data band CLIPPED to [%.6e, %.6e] Hz (+-%d layers around "
                "the GB band).", gs.min_freq, gs.max_freq, L,
            )

    def set_default_processor(self, gs: GBNoFgGeneralSettings) -> None:
        if gs.data_mode == "mojito":
            # Mojito L1 GB galaxy (populates catalogue["GB"] for the
            # true-point start).
            super().set_default_processor(gs)
            return
        if gs.data_mode == "synthetic":
            # In-process GB injection — no external data needed. With no
            # catalogue the true-point seeding in setup_recipe is a no-op
            # and the sampler starts from prior draws / RJ births.
            from ..injections import GB_INJECTION_PARAMS, SyntheticGBProcessingStep

            params = (
                gs.gb_injection_params
                if gs.gb_injection_params is not None
                else GB_INJECTION_PARAMS
            )
            gs.data_processor_class = SyntheticGBProcessingStep
            gs.processor_init_kwargs = dict(
                Tobs=self.wdm_grid[3],
                dt=gs.dt,
                t_start=gs.synthetic_t_start,
                injection_params=np.atleast_2d(np.asarray(params, dtype=float)),
                tdi_chan=gs.tdi_chan,
                nchannels=gs.nchannels,
                use_tdi2=tdi_generation_info(gs.tdi_chan)[0] == 2,
                force_backend="cpu",
            )
            return
        raise ValueError(
            f"gb_no_fg data_mode={gs.data_mode!r} not recognised; use "
            "'mojito' or 'synthetic' (or swap data_processor_class wholesale)."
        )

    def default_preprocess_kwargs(self) -> dict:
        if self.general.data_mode == "synthetic":
            # The synthetic stream covers exactly Tobs = Nf*Nt*dt; skip the
            # engine's default highpass + edge-trim + Tobs trim so the WDM
            # shape stays exact.
            return dict(
                highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False
            )
        return super().default_preprocess_kwargs()

    def make_domain_settings(self, gs, Nf, Nt, wavelet_duration, edge_crop):
        # GB_DOMAIN=fd switches the run basis to the frequency domain (same
        # band restriction); the object-level swap (fit.general.domain_settings
        # = FDSettings.make_factory(...)) is the primary interface.
        if env_resolve("GB_DOMAIN", "wdm", str).lower() == "fd":
            return FDSettings.make_factory(min_freq=gs.min_freq, max_freq=gs.max_freq)
        return super().make_domain_settings(gs, Nf, Nt, wavelet_duration, edge_crop)

    # -- branch resolution --------------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        if name != "gb":
            return settings
        gb: GBNoFgGBSettings = settings
        if gb.t0 is None:
            # Mojito: catalogue params are referenced at the catalogue epoch
            # (validated convention); synthetic: phases reference the stream
            # start.
            gb.t0 = (
                GB_MOJITO_T_REF
                if self.general.data_mode == "mojito"
                else float(self.general.synthetic_t_start)
            )
        if gb.mode not in ("pe", "search"):
            raise ValueError(f"GB mode must be 'pe' or 'search', got {gb.mode!r}.")
        if gb.start not in ("truth", "prior"):
            raise ValueError(f"GB start must be 'truth' or 'prior', got {gb.start!r}.")
        domain_settings = general_setup.domain_settings  # resolved instance

        # f0 prior clamped to the (possibly clipped) data band.
        if not gb.f0_lims:
            gb.f0_lims = [general_setup.min_freq, general_setup.max_freq]
        if not gb.fdot_lims:
            fdot_max_val = get_fdot(gb.f0_lims[-1], Mc=gb.m_chirp_lims[-1])
            gb.fdot_lims = [-fdot_max_val, fdot_max_val]

        # Narrow GB band (WDM) / widened band (FD band walker).
        is_fd = isinstance(domain_settings, FDSettings)
        layer_df = 1.0 / (2 * general_setup.domain_settings.Nf * gb.dt) if not is_fd else None
        if is_fd:
            if gb.fd_bandwidth is None:
                gb.start_freq = general_setup.min_freq
                gb.end_freq = general_setup.max_freq
            else:
                half_bw = 0.5 * float(gb.fd_bandwidth)
                gb.start_freq = max(general_setup.min_freq, gb.center_freq - half_bw)
                gb.end_freq = min(general_setup.max_freq, gb.center_freq + half_bw)
            logger.info(
                "FD basis: GB band widened to [%.6e, %.6e] Hz for the FD band "
                "walker.", gb.start_freq, gb.end_freq,
            )
        else:
            gb.start_freq, gb.end_freq, _ = self._gb_band(layer_df)
            logger.info(
                "GB band: [%.6e, %.6e] Hz (%d WDM layers @ ~%.3f mHz, "
                "layer_df=%.4e Hz)", gb.start_freq, gb.end_freq, gb.n_layers,
                gb.center_freq * 1e3, layer_df,
            )

        # FD basis at short Tobs: oversample=2 halves the per-band N.
        if gb.oversample is None:
            gb.oversample = 2 if is_fd else 4

        gb.tdi_setup = general_setup.tdi_chan
        gb.use_tdi2 = tdi_generation_info(general_setup.tdi_chan)[0] == 2
        gb.initialize_kwargs = dict(force_backend=general_setup.force_backend)

        # Match the GB tempering ladder length to ntemps (band_temps shape).
        if gb.betas is None:
            betas = 1.0 / 1.2 ** np.arange(general_setup.ntemps)
            betas[-1] = 1e-4
            gb.betas = betas

        # WDM-domain GB likelihood object: built lazily in setup_recipe (the
        # underlying C++ orbits wrap is not picklable).
        gb.gb_wdm_comp = None
        return gb


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``gb_no_fg`` (the run's ``setup_function``).

    Variant-specific pre-work (chunked-het likelihood build, true-point /
    prior seeding, neighbor subtraction) followed by the generic
    materialization of the fit's :class:`RecipeSpec`.
    """
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        import cupy as cp

        cp.cuda.runtime.setDevice(gpus[0])

    gb_info = curr.source_info["gb"]
    tdi_gen = 2 if getattr(gb_info, "use_tdi2", True) else 1
    tdi_gen_str = f"{tdi_gen}{'nd' if tdi_gen == 2 else 'st'} generation"

    # Search-mode arming + debug-plot cell selection: consumed inside
    # ``build_gb_moves`` (and the GB moves) from the environment; seeded via
    # setdefault so explicit env overrides still win.
    if getattr(gb_info, "mode", "pe") == "search":
        os.environ.setdefault("GB_LEAF_CAP_START", "1")
        os.environ.setdefault("GB_RJ_PHASE_MAXIMIZE", "1")
    os.environ.setdefault("GB_DEBUG_PLOT_WALKER", "0")
    os.environ.setdefault("GB_DEBUG_PLOT_BAND", str(getattr(gb_info, "n_layers", 3) // 2))

    # Build the WDM-domain GB likelihood here (after the deepcopy in
    # ``CurrentInfoGlobalFit.__init__``) — the underlying C++ orbits wrap
    # is not picklable, so it must live outside the settings dataclass.
    # Chunked-heterodyne ``GBWDMComputations`` is the only WDM backend.
    if (
        isinstance(general_info.domain_settings, WDMSettings)
        and gb_info.gb_wdm_comp is None
    ):
        from gbgpu.gbcomps import GBWDMComputations

        _wdm = general_info.domain_settings
        # Anchor the WDM domain t0 on the absolute data start so the het
        # samples the orbits inside their span (WDM coefficient addressing
        # is t0-independent; only absolute orbit-evaluation times depend on
        # this).
        _t_obs_start = float(getattr(general_info, "data_t0", 0.0))
        _orig_wdm_t0 = float(_wdm.t0)
        _wdm.t0 = _t_obs_start
        gb_info.gb_wdm_comp = GBWDMComputations(
            _wdm,
            t_ref=gb_info.t0,
            Nt_sub=int(gb_info.nt_sub),
            n_pad=int(gb_info.n_pad),
            N_sparse=int(gb_info.n_sparse),
            N_cp_sig=int(gb_info.n_cp_sig),
            N_cp_orbit=int(gb_info.n_cp_orbit),
            orbits=general_info.gpu_orbits,
            tdi_config=tdi_gen_str,
            force_backend=general_info.force_backend,
            tdi_type="XYZ",
        )
        logger.info(
            "Chunked-het GB likelihood: Nf=%d Nt=%d Nt_sub=%d N_sparse=%d "
            "N_cp_sig=%d N_cp_orbit=%d (domain t0 %.6e -> het t_obs_start=%.6e, "
            "t_ref=%.6e, chunk_t_starts=[%.6e, %.6e])",
            _wdm.Nf, _wdm.Nt,
            gb_info.gb_wdm_comp.Nt_sub,
            gb_info.gb_wdm_comp.N_sparse,
            gb_info.gb_wdm_comp.N_cp_sig,
            gb_info.gb_wdm_comp.N_cp_orbit,
            _orig_wdm_t0, gb_info.gb_wdm_comp.t_obs_start, gb_info.t0,
            float(gb_info.gb_wdm_comp.chunk_t_starts.min()),
            float(gb_info.gb_wdm_comp.chunk_t_starts.max()),
        )

        # Sig-het in-model scoring: the move receives a
        # GBSignalHetComputations wrapping the chunked comp; RJ / fills /
        # swaps keep the chunked-het path (pure type dispatch).
        if getattr(gb_info, "sighet_inmodel", False):
            from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

            gb_info.gb_wdm_comp = GBSignalHetComputations.for_band_engine(
                gb_info.gb_wdm_comp,
                nt_layer=int(gb_info.sighet_nt_layer),
                n_sparse_fd=int(gb_info.sighet_n_sparse_fd),
            )
            logger.info(
                "GB in-model likelihood: SIGNAL-HET "
                "(chunked-het delegate for RJ / fills / swaps)."
            )

    # FD-domain mirror: the GB move auto-builds a config-only
    # ``GBFDComputations`` from the ACA's FDSettings; it only needs the
    # orbits + TDI-config handles (wired here after the settings deepcopy,
    # exactly like ``gb_wdm_comp`` on the WDM path).
    if isinstance(general_info.domain_settings, FDSettings):
        if getattr(gb_info, "orbits", None) is None:
            gb_info.orbits = general_info.gpu_orbits
        if getattr(gb_info, "tdi_config", None) is None:
            gb_info.tdi_config = tdi_gen_str

    # ============== START GBs AT TRUE CATALOG POINTS (SNR cut) ==============
    if gb_info.gb_wdm_comp is not None:
        layer_df = 1.0 / (2 * general_info.domain_settings.Nf * gb_info.dt)
        k_center = int(math.floor(getattr(gb_info, "center_freq", 7.5e-3) / layer_df))
        _injection_f0_lims = None
        if getattr(gb_info, "subtract_neighbors", False):
            _central_lims = (k_center * layer_df, (k_center + 1) * layer_df)
            _injection_f0_lims = _central_lims
            subtract_gb_neighbors_from_data(
                curr, acs, gb_info, gb_info.gb_wdm_comp,
                exclude_f0_lims=_central_lims,
                window_hz=getattr(gb_info, "neighbor_window_layers", 4) * layer_df,
            )
        # mode="search": NO true-point seeding — the gb branch keeps the
        # engine's zero-leaf state and the sampler must FIND the sources
        # through RJ under the per-band progressive leaf cap.
        if getattr(gb_info, "mode", "pe") != "search":
            gb_snr_subset_inds = select_gb_injection_subset_by_snr(
                curr, acs, gb_info, gb_info.gb_wdm_comp,
                snr_threshold=getattr(gb_info, "snr_threshold", 3.0),
                f0_lims=_injection_f0_lims,
            )
            if getattr(gb_info, "start", "truth") == "prior":
                # PE dimensionality WITHOUT truth seeding: same leaf count as
                # the SNR-selected subset, coordinates drawn from the PRIOR.
                _n_true = int(len(gb_snr_subset_inds))
                _gb_inds = state.branches["gb"].inds
                _gb_coords = state.branches["gb"].coords
                _nt, _nw, _nl, _ndim = _gb_coords.shape
                _draws = priors["gb"].rvs(size=(_nt, _nw, _n_true))
                _draws = (
                    _draws.get() if hasattr(_draws, "get") else np.asarray(_draws)
                )
                _gb_inds[:] = False
                _gb_inds[:, :, :_n_true] = True
                _gb_coords[:, :, :_n_true, :] = _draws
                logger.info(
                    "GB start: %d leaves per walker at PRIOR draws "
                    "(start='prior'; truth seeding skipped).", _n_true,
                )
            else:
                # Per-dimension scatter for the true-point start, sized to a
                # small fraction of each prior dimension's width (the stock
                # scalar default is ~10 orders too wide for GB fdot).
                _gb_draws = priors["gb"].rvs(size=20000)
                _gb_draws = (
                    _gb_draws.get() if hasattr(_gb_draws, "get") else np.asarray(_gb_draws)
                )
                _gb_spread = 1e-4 * (_gb_draws.max(axis=0) - _gb_draws.min(axis=0))
                setup_state_for_injection(
                    curr, state, source_type="GB", branch_name="gb",
                    subset_inds=gb_snr_subset_inds, priors=priors,
                    spread=_gb_spread,
                )

    # ========================= MATERIALIZE THE RECIPE =========================
    # The fit's RecipeSpec (carried through the settings deepcopy) names the
    # GB moves to install; ``build_gb_moves`` owns the GB reference recipe
    # and is steered by exactly those names.
    recipe_spec: RecipeSpec = curr.source_metadata["recipe_spec"]
    requested = [
        mv.name
        for stage in recipe_spec.stages
        for mv in stage.moves
        if mv.instance is None and mv.target is None
    ]
    include_search = any(name.endswith("_search") for name in requested)
    include_refit = any(name.startswith("rj_refit") for name in requested)
    pe_names = [name for name in requested if not name.endswith("_search")]
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state,
        include_search=include_search,
        include_refit=include_refit,
        pe_move_names=pe_names or None,
    )
    stock_moves = {m.name: m for m in list(gb_search_moves) + list(gb_pe_moves)}

    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state,
    )
    materialize_recipe(recipe, recipe_spec, ctx, stock_moves, ntemps, nwalkers)


GBNoForegroundGlobalFit.default_setup_function = staticmethod(setup_recipe)
