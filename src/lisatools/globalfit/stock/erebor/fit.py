"""Erebor-family base fit: shared general knobs + build resolution.

:class:`EreborGeneralSettings` extends the engine's
:class:`~lisatools.globalfit.engine.GeneralSettings` dataclass with the
Erebor-family knobs every variant shares (WDM-grid targets, analysis band,
TDI channel, mojito data path, fixed-PSD values, compute-backend choice).
Env-var-backed defaults resolve *explicit kwarg > env var > hard default*
at construction (see :func:`lisatools.globalfit.stock.base.env_default`).

:class:`EreborFit` extends :class:`StockGlobalFit` with the shared
build-time resolution: the deferred general block is turned into a final
:class:`GeneralSettings` (grid derived, window taper realized, domain
factory built, data processor chosen) only inside
:meth:`EreborFit.make_general_settings` — never at construction.
"""

from __future__ import annotations

import dataclasses
import typing
from copy import deepcopy

from lisatools.domains import WDMSettings

from ...engine import GeneralSettings, GeneralSetup, Settings
from ..base import StockGlobalFit, env_default
from .common import (
    default_edge_crop_wavelets,
    derive_wdm_grid,
    resolve_compute,
    tdi_generation_info,
)

__all__ = ["EreborGeneralSettings", "EreborFit"]


def _parse_gpu_list(raw: str) -> typing.List[int]:
    """GPUS env parser: comma-separated device indices."""
    return [int(x) for x in raw.split(",") if x.strip() != ""]


@dataclasses.dataclass
class EreborGeneralSettings(GeneralSettings):
    """Erebor-family general block: engine fields + shared Erebor knobs.

    Fields left ``None`` here are *derived at build time* by
    :meth:`EreborFit.make_general_settings` (e.g. ``Tobs`` from the WDM
    grid, ``domain_settings`` from the band knobs, ``window_taper_duration``
    from ``window_tukey_alpha``); setting them explicitly overrides the
    derivation — swap whole objects, don't fight the resolver.
    """

    # --- run shape (env-backed) ---
    num_iterations: int = dataclasses.field(
        default_factory=env_default("GF_NUM_ITER", 500, int)
    )
    nwalkers: int = dataclasses.field(default_factory=env_default("NWALKERS", 4, int))
    ntemps: int = dataclasses.field(default_factory=env_default("NTEMPS", 2, int))
    random_seed: int = 103209
    backup_iter: int = 5
    main_file_key: str = "testing"
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "erebor_run")
    )

    # --- grid / domain targets (env-backed) ---
    # ``Tobs`` (inherited) stays None; the build derives it as Nf*Nt*dt.
    dt: float = 2.5
    tobs_target: float = dataclasses.field(
        default_factory=env_default("TOBS_TARGET", 90 * 86400.0, float)
    )
    # Fixed-grid override: when BOTH are set, the WDM grid is (nf, nt)
    # directly (Tobs = nf*nt*dt) instead of adjust_to_even_bins on
    # tobs_target / the wavelet-duration bounds.
    nf: typing.Optional[int] = None
    nt: typing.Optional[int] = None
    wavelet_duration_min: float = dataclasses.field(
        default_factory=env_default("WAVELET_DUR_MIN", 3600.0, float)
    )
    wavelet_duration_max: float = dataclasses.field(
        default_factory=env_default("WAVELET_DUR_MAX", 4400.0, float)
    )
    min_freq: float = 1e-4
    max_freq: float = 2.5e-2
    window_type: str = "tukey"
    window_tukey_alpha: float = dataclasses.field(
        default_factory=env_default("WINDOW_TUKEY_ALPHA", 0.05, float)
    )
    edge_crop_wavelets: typing.Optional[int] = None  # None -> auto from alpha

    # --- TDI / channels ---
    tdi_chan: str = "XYZ"
    nchannels: int = 3

    # --- data source ---
    # Which stock data processor the variant builds when the user has not
    # swapped one in wholesale. Allowed values are variant-defined (every
    # variant supports "synthetic"; gb_no_fg/full_year default "mojito",
    # all_sources defaults "sangria"). One assignment (or
    # DATA_PROCESSOR=<mode>) swaps the whole data pipeline; an explicit
    # ``data_processor_class`` always wins over this knob.
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_PROCESSOR", "mojito", str)
    )
    # Start time of the synthetic data stream (mojito/sangria modes pull the
    # start from the data files).
    synthetic_t_start: float = 0.0
    mojito_data_path: str = dataclasses.field(
        default_factory=env_default(
            "MOJITO_DATA_PATH",
            "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",
        )
    )
    ldc_source_file: str = dataclasses.field(
        default_factory=env_default(
            "LDC_SOURCE_FILE",
            "/Users/mkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5",
        )
    )
    source_types: typing.Tuple[str, ...] = ("GB",)
    orbits_frame: str = "icrs"

    # --- fixed sensitivity (used when no psd branch is present) ---
    fixed_psd_params: typing.Optional[typing.List[float]] = None

    # --- compute backend (construction-time choice; sprint rule) ---
    # USE_GPU=0 forces CPU even when cupy is present; unset -> auto-detect.
    use_gpu: typing.Optional[bool] = dataclasses.field(
        default_factory=env_default("USE_GPU", None, bool)
    )
    # GPU_BACKEND selects the lisatools/gbgpu backend wheel flavor
    # (cuda11x / cuda12x / cuda13x); only consulted when the GPU is active.
    gpu_backend: str = dataclasses.field(
        default_factory=env_default("GPU_BACKEND", "cuda13x", str)
    )
    # GPUS: comma-separated device indices (e.g. "0" or "2,3"); unset ->
    # device 0 when the GPU is active.
    gpus: typing.Optional[typing.List[int]] = dataclasses.field(
        default_factory=env_default("GPUS", None, _parse_gpu_list)
    )

    @property
    def tdi_gen(self) -> int:
        return tdi_generation_info(self.tdi_chan)[0]

    @property
    def tdi_gen_str(self) -> str:
        return tdi_generation_info(self.tdi_chan)[1]

    @property
    def wavelet_duration_bounds(self) -> typing.Tuple[float, float]:
        return (self.wavelet_duration_min, self.wavelet_duration_max)


class EreborFit(StockGlobalFit):
    """Deferred-build base for Erebor stock variants.

    Headline knobs (top-level attribute/constructor access, delegated to
    ``fit.general``): everything in :attr:`_HEADLINE_KNOBS`. All other
    general fields: ``fit.general.<field>``; per-branch fields:
    ``fit.<branch>.<field>``; whole-object swaps everywhere.
    """

    general_settings_class: typing.ClassVar[type] = EreborGeneralSettings

    _HEADLINE_KNOBS = StockGlobalFit._HEADLINE_KNOBS + (
        "tobs_target",
        "min_freq",
        "max_freq",
        "tdi_chan",
        "window_tukey_alpha",
        "mojito_data_path",
        "use_gpu",
    )

    # -- defaults ------------------------------------------------------------

    def default_general(self) -> EreborGeneralSettings:
        return type(self).general_settings_class()

    # -- derived views (recompute from current knobs; mutation-safe) -----------

    @property
    def wdm_grid(self) -> typing.Tuple[int, int, float, float]:
        """``(Nf, Nt, wavelet_duration, Tobs)`` from the current grid knobs."""
        gs = self.general
        if gs.nf is not None and gs.nt is not None:
            return gs.nf, gs.nt, gs.nf * gs.dt, gs.nf * gs.nt * gs.dt
        return derive_wdm_grid(gs.tobs_target, gs.dt, gs.wavelet_duration_bounds)

    @property
    def layer_df(self) -> float:
        Nf, _, _, _ = self.wdm_grid
        return 1.0 / (2 * Nf * self.general.dt)

    # -- build-time resolution --------------------------------------------------

    def adjust_general(self, gs: EreborGeneralSettings) -> None:
        """Variant hook: mutate the copied general block before resolution."""

    def make_general_settings(self) -> EreborGeneralSettings:
        gs = deepcopy(self.general)
        self.adjust_general(gs)

        # Compute backend (gpus=None -> engine resolves force_backend="cpu").
        gs.gpus, gs.gpu_backend = resolve_compute(gs.use_gpu, gs.gpu_backend, gs.gpus)

        # WDM grid + exact Tobs (fixed nf/nt override wins over derivation).
        if gs.nf is not None and gs.nt is not None:
            Nf, Nt = gs.nf, gs.nt
            wavelet_duration, tobs = Nf * gs.dt, Nf * Nt * gs.dt
        else:
            Nf, Nt, wavelet_duration, tobs = derive_wdm_grid(
                gs.tobs_target, gs.dt, gs.wavelet_duration_bounds
            )
        if gs.Tobs is None:
            gs.Tobs = tobs

        # Window taper realized from the alpha knob on the resolved grid.
        if gs.window_taper_duration is None:
            gs.window_taper_duration = gs.window_tukey_alpha * gs.Tobs

        # WDM time-edge crop: cover boundary wavelets AND the Tukey taper.
        edge_crop = (
            gs.edge_crop_wavelets
            if gs.edge_crop_wavelets is not None
            else default_edge_crop_wavelets(gs.window_tukey_alpha, Nt)
        )
        gs.edge_crop_wavelets = edge_crop

        # Domain: WDM factory from the grid knobs unless the user swapped in
        # their own DomainSettings instance/factory.
        if gs.domain_settings is None:
            gs.domain_settings = self.make_domain_settings(
                gs, Nf, Nt, wavelet_duration, edge_crop
            )

        # Data processor (whole-object swap wins over the variant default).
        if gs.data_processor_class is None:
            self.set_default_processor(gs)
        if gs.preprocess_kwargs is None:
            gs.preprocess_kwargs = self.default_preprocess_kwargs()
        if gs.sensitivity_init_kwargs is None:
            gs.sensitivity_init_kwargs = dict(tdi_generation=gs.tdi_gen)
        if gs.fixed_psd_kwargs is None and gs.fixed_psd_params is not None:
            gs.fixed_psd_kwargs = dict(
                psd_params=list(gs.fixed_psd_params), galfor_params=None
            )
        self.finalize_general(gs)
        return gs

    def finalize_general(self, gs: EreborGeneralSettings) -> None:
        """Variant hook: last-touch mutation after Tobs/domain are resolved."""

    def make_domain_settings(
        self,
        gs: EreborGeneralSettings,
        Nf: int,
        Nt: int,
        wavelet_duration: float,
        edge_crop: int,
    ):
        """Default domain block: WDM factory on the derived grid."""
        return WDMSettings.make_factory(
            Nf=Nf,
            Nt=Nt,
            min_freq=gs.min_freq,
            max_freq=gs.max_freq,
            min_time=edge_crop * wavelet_duration,
            max_time=(Nt - edge_crop) * wavelet_duration,
        )

    def default_preprocess_kwargs(self) -> dict:
        """Variant hook: kwargs merged over the engine's preprocess defaults."""
        return dict(normalize=False)

    def set_default_processor(self, gs: EreborGeneralSettings) -> None:
        """Default data source: mojito L1 loader on ``gs.source_types``."""
        from lisatools.detector import L1Orbits
        from lisatools.globalfit.preprocessing import L1ProcessingStep

        force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
        gs.data_processor_class = L1ProcessingStep
        gs.processor_init_kwargs = dict(
            L1_folder=gs.mojito_data_path,
            source_types=list(gs.source_types),
            source_ids=None,
            orbits_class=L1Orbits,
            orbits_kwargs=dict(force_backend=force_backend, frame=gs.orbits_frame),
            # NOTE: do NOT pass Tobs here. The engine's preprocess trims the
            # ends and then keeps the first ``Tobs`` seconds, so the final
            # data must be exactly Nf*Nt samples; pre-trimming at load would
            # leave less than Tobs and mismatch the fixed WDM grid.
        )

    # -- branch preparation ---------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        # Common Erebor bindings: every branch runs on the resolved domain
        # and logs beside the run output.
        if getattr(settings, "domain_settings", None) is None and hasattr(
            settings, "domain_settings"
        ):
            settings.domain_settings = general_setup.domain_settings
        if settings.log_dir is None:
            settings.log_dir = general_setup.file_store_dir
        return settings
