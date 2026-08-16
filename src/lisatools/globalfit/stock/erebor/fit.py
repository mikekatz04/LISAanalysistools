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
import logging
import os
import typing
from copy import deepcopy

from lisatools.domains import WDMSettings

from ...engine import GeneralSettings, GeneralSetup, Settings
from ..base import StockGlobalFit, engine_ntemps_default, env_default
from .common import (
    default_edge_crop_wavelets,
    derive_wdm_grid,
    resolve_compute,
    tdi_generation_info,
)

__all__ = ["EreborGeneralSettings", "EreborFit"]

logger = logging.getLogger(__name__)


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
        default_factory=env_default("NUM_ITERATIONS", 500, int)
    )
    nwalkers: int = dataclasses.field(default_factory=env_default("NWALKERS", 4, int))
    # RETIRED as a tempering knob: the engine runs cold-chain only (each
    # branch tempers internally; see the per-branch <BRANCH>_NTEMPS knobs).
    # A set NTEMPS env var raises. erebor.blank overrides this field to keep
    # a real engine ladder for simple-API branches.
    ntemps: int = dataclasses.field(default_factory=engine_ntemps_default())
    random_seed: int = 103209
    backup_iter: int = 1
    main_file_key: str = "testing"
    # Diagnostic plotting during the run. MAKE_DIAGNOSTIC_PLOTS=0 disables the eryn
    # diagnostic plots entirely (fastest, and dodges plot-only crashes);
    # PLOT_ITERATIONS sets how many sampler iterations between plot refreshes.
    make_diagnostic_plots: bool = dataclasses.field(
        default_factory=env_default("MAKE_DIAGNOSTIC_PLOTS", True, bool)
    )
    # Console verbosity (headline knob; env VERBOSE). Default: quiet — logs
    # go to the run's log files only, no progress bars.
    verbose: bool = dataclasses.field(default_factory=env_default("VERBOSE", False, bool))
    # None -> follow verbose (see GeneralSettings.progress). Env: PROGRESS.
    progress: typing.Optional[bool] = dataclasses.field(
        default_factory=env_default("PROGRESS", None, bool)
    )
    plot_iterations: int = dataclasses.field(
        default_factory=env_default("PLOT_ITERATIONS", 100, int)
    )
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
        default_factory=env_default("WAVELET_DURATION_MIN", 3600.0, float)
    )
    wavelet_duration_max: float = dataclasses.field(
        default_factory=env_default("WAVELET_DURATION_MAX", 4400.0, float)
    )
    # Env-backed per the stock convention (knob = capitalized field name):
    # these were plain defaults, so MIN_FREQ/MAX_FREQ in the environment
    # were SILENTLY ignored — the exact failure mode rule 0 warns about.
    min_freq: float = dataclasses.field(
        default_factory=env_default("MIN_FREQ", 1e-4, float)
    )
    max_freq: float = dataclasses.field(
        default_factory=env_default("MAX_FREQ", 2.5e-2, float)
    )
    window_type: str = "tukey"
    window_tukey_alpha: float = dataclasses.field(
        default_factory=env_default("WINDOW_TUKEY_ALPHA", 0.05, float)
    )
    edge_crop_wavelets: typing.Optional[int] = None  # None -> auto from alpha
    # Deterministic taper width in WDM wavelets (each Nf*dt seconds). When
    # > 0 this OVERRIDES window_tukey_alpha with 2*K/Nt: the taper is a
    # fixed ABSOLUTE duration set by the wavelet grid, not a fraction of
    # Tobs (a fixed alpha burns alpha/2 * Tobs of data per side -- ~18 d
    # at 2 yr). DERIVATION of the default (2 wavelets): a Tukey ramp of
    # duration tau leaks over bandwidth ~1/tau; confining edge leakage to
    # ~one WDM layer requires tau >= 1/layer_df = 2*Nf*dt = 2 wavelet
    # widths -- shorter smears across layers, longer only discards data.
    # The DIRTY region (taper + the wavelet's own ~2-pixel support +
    # boundary-wavelet contamination) is removed from the ANALYZED region
    # by the edge crop (min_time/max_time, auto = max(20, taper+4), floor
    # dominating), so the analysis always sees window == 1 on clean
    # wavelets. 0 = legacy fraction-of-Tobs via window_tukey_alpha.
    window_taper_wavelets: int = dataclasses.field(
        default_factory=env_default("WINDOW_TAPER_WAVELETS", 2, int)
    )

    # --- TDI / channels ---
    tdi_chan: str = "XYZ"
    nchannels: int = 3

    # --- data source ---
    # Which stock data processor the variant builds when the user has not
    # swapped one in wholesale. Allowed values are variant-defined (every
    # variant supports "synthetic"; gb_no_fg/full_year default "mojito",
    # all_sources defaults "sangria"). One assignment (or
    # DATA_MODE=<mode>) swaps the whole data pipeline; an explicit
    # ``data_processor_class`` always wins over this knob.
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_MODE", "mojito", str)
    )
    # Start time of the synthetic data stream (mojito/sangria modes pull the
    # start from the data files). Default 10,000 s — NOT 0 — so the TDI2
    # warm-up look-back (output at t needs orbit data back to t - ~8
    # arm-delays ~ 85 s) stays inside the orbit span, which starts at t=0;
    # a t=0 data start makes the first ~85 s of every response unevaluable
    # (they would be NaN-scrubbed to zero).
    synthetic_t_start: float = 10_000.0
    mojito_data_path: str = dataclasses.field(
        default_factory=env_default(
            "MOJITO_DATA_PATH",
            # Home-relative so the same default resolves on any machine
            # (laptop /Users/<u>, cluster /home/<u>); override with
            # MOJITO_DATA_PATH when the cache lives elsewhere.
            os.path.expanduser(
                "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
            ),
        )
    )
    # --- mojito NOISE brick (proper noise) ---
    # Explicit path to a mojito NOISE L1 file. None -> auto-discover
    # ``<mojito_data_path>/data/INSTRUMENT/L1/NOISE_*``.
    noise_file: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("NOISE_FILE", None, str)
    )
    # Fixed-PSD variants (gb_no_fg, full_year_combined): read the
    # instrument-noise parameters off the NOISE brick's tabulated estimates
    # (linear least-squares of the analytic model, see
    # ``lisatools.sensitivity.estimate_noise_params_from_file``) instead of
    # the hardcoded stock levels. None -> auto: True when the run uses mojito
    # data and the brick is found; explicit True errors if the brick is
    # missing; False keeps the stock analytic levels.
    psd_from_noise_file: typing.Optional[bool] = dataclasses.field(
        default_factory=env_default("PSD_FROM_NOISE_FILE", None, bool)
    )
    ldc_source_file: str = dataclasses.field(
        default_factory=env_default(
            "LDC_SOURCE_FILE",
            "/Users/mkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5",
        )
    )
    source_types: typing.Tuple[str, ...] = ("GB",)
    orbits_frame: str = "icrs"

    # --- synthetic-injection coordinates ---
    # How the synthetic data processors pick injection coordinates when the
    # user has not supplied them (``gb_injection_params`` / per-branch
    # ``.injection``): "stock" -> the fixed stock tables; "prior" -> seeded
    # draws from (an interior region of) each source class's sampling
    # priors. ``None`` -> auto: "stock" normally, "prior" when a missing
    # mojito folder forced the synthetic fallback (see
    # :meth:`EreborFit.resolve_data_source`). Explicit injection tables
    # always win over either mode.
    synthetic_injections: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("SYNTHETIC_INJECTIONS", None, str)
    )
    synthetic_injection_seed: int = dataclasses.field(
        default_factory=env_default("SYNTHETIC_INJECTION_SEED", 1234, int)
    )

    # --- fixed sensitivity (used when no psd branch is present) ---
    fixed_psd_params: typing.Optional[typing.List[float]] = None

    # --- compute backend (construction-time choice; sprint rule) ---
    # USE_GPU=0 forces CPU even when cupy is present; unset -> auto-detect.
    use_gpu: typing.Optional[bool] = dataclasses.field(
        default_factory=env_default("USE_GPU", None, bool)
    )
    # GPU_BACKEND selects the lisatools/gbgpu backend wheel flavor
    # (cuda11x / cuda12x / cuda13x); only consulted when the GPU is active.
    # Default "auto" -> resolve_compute detects the wheel matching this
    # machine's cupy runtime at build time, so a non-propagated GPU_BACKEND
    # env var can't silently pick the wrong flavor. An explicit value
    # (env or kwarg) is always honored verbatim.
    gpu_backend: str = dataclasses.field(
        default_factory=env_default("GPU_BACKEND", "auto", str)
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

    def resolve_data_source(self) -> None:
        """Resolve the data source on the FIT-LEVEL general block (idempotent).

        When ``data_mode == "mojito"`` but the mojito L1 folder does not
        exist on this machine, fall back to ``data_mode = "synthetic"`` so
        the bare quickstart (``fit = erebor.all_sources(); fit.build()``)
        works with no external data. Fallback injection coordinates are
        drawn from the source priors (``synthetic_injections = "prior"``)
        unless the user chose a mode or supplied explicit tables
        (``gb_injection_params`` / per-branch ``.injection``), which always
        win.

        Deliberately mutates ``self.general`` (not just the build-time
        copy): branch preparation reads the fit-level block, so the data
        processor and the branch injections must resolve identically.
        """
        gs = self.general
        if gs.data_mode == "mojito" and not os.path.isdir(gs.mojito_data_path):
            logger.warning(
                "data_mode='mojito' but the mojito data folder %r does not "
                "exist — falling back to data_mode='synthetic' (in-process "
                "injections, coordinates drawn from the source priors). Set "
                "MOJITO_DATA_PATH / general.mojito_data_path to use mojito "
                "data.",
                gs.mojito_data_path,
            )
            gs.data_mode = "synthetic"
            if gs.synthetic_injections is None:
                gs.synthetic_injections = "prior"
        if gs.synthetic_injections is None:
            gs.synthetic_injections = "stock"
        elif gs.synthetic_injections not in ("stock", "prior"):
            raise ValueError(
                f"synthetic_injections={gs.synthetic_injections!r} not "
                "recognised; use 'stock', 'prior', or None (auto)."
            )
        # Prior-mode GB rows ride the existing ``gb_injection_params`` knob
        # (the synthetic GB processor and the GB branch prep both read it),
        # so variants need no extra wiring. An explicit user table wins.
        if (
            gs.data_mode == "synthetic"
            and gs.synthetic_injections == "prior"
            and "gb" in self._branch_names
            and getattr(gs, "gb_injection_params", "absent") is None
        ):
            from .injections import make_gb_injections

            gs.gb_injection_params = make_gb_injections(
                self._gb_injection_count(),
                mode="prior",
                seed=gs.synthetic_injection_seed,
                band=self._gb_injection_band(),
            )

    def resolve_noise_file_psd_params(
        self, gs: EreborGeneralSettings
    ) -> typing.Optional[typing.List[float]]:
        """``[Soms_d, Sa_a]`` read off the mojito NOISE brick, or ``None``.

        Honors ``gs.psd_from_noise_file``: ``False`` -> ``None``; ``None``
        (auto) -> only for mojito-data runs with a resolvable brick;
        ``True`` -> required (raises when the brick is missing or the fit
        fails). The scalar levels come from a linear least-squares fit of the
        analytic instrument model to the brick's tabulated
        ``noise_estimates`` (:func:`lisatools.sensitivity.estimate_noise_params_from_file`).
        Results are cached per file path on the fit instance.
        """
        from .noise import noise_params_from_file, resolve_noise_file

        want = gs.psd_from_noise_file
        if want is False:
            return None
        if want is None and gs.data_mode != "mojito":
            return None
        noise_file = resolve_noise_file(gs.mojito_data_path, gs.noise_file)
        if noise_file is None:
            if want:
                raise FileNotFoundError(
                    "psd_from_noise_file=True but no mojito NOISE brick was "
                    "found: set general.noise_file / NOISE_FILE or add "
                    "data/INSTRUMENT/L1/NOISE_* under "
                    f"{gs.mojito_data_path!r}."
                )
            return None
        cache = getattr(self, "_noise_file_params_cache", None)
        if cache is None:
            cache = self._noise_file_params_cache = {}
        if noise_file not in cache:
            cache[noise_file] = noise_params_from_file(
                noise_file, tdi_generation=gs.tdi_gen
            )
        params = cache[noise_file]
        if params is None and want:
            raise ValueError(
                f"psd_from_noise_file=True but the noise-parameter fit failed "
                f"for {noise_file!r} (see the warning above)."
            )
        return list(params) if params is not None else None

    def resolve_noise_source(
        self, gs: EreborGeneralSettings
    ) -> typing.Union[bool, str]:
        """Concrete data-noise source: ``False``, ``"synthetic"``, or ``"mojito"``.

        Resolves a variant's ``add_instrument_noise`` knob: ``True`` (auto)
        becomes ``"mojito"`` — the real NOISE brick summed into the data by
        the L1 loader — when the run uses mojito data and the brick is found,
        else ``"synthetic"`` (the FD-correlated draw). Explicit strings are
        validated and passed through; ``"mojito"`` requires mojito data.
        """
        from .noise import resolve_noise_file

        value = gs.add_instrument_noise
        if value is True:
            has_brick = (
                gs.data_mode == "mojito"
                and resolve_noise_file(gs.mojito_data_path, gs.noise_file) is not None
            )
            value = "mojito" if has_brick else "synthetic"
            logger.info("add_instrument_noise=True resolved to %r.", value)
        elif value is not False and value not in ("synthetic", "mojito"):
            raise ValueError(
                f"add_instrument_noise={gs.add_instrument_noise!r} not "
                "recognised; use False, True (auto), 'synthetic', or 'mojito'."
            )
        if value == "mojito" and gs.data_mode != "mojito":
            raise ValueError(
                "add_instrument_noise='mojito' (the real NOISE brick) requires "
                f"data_mode='mojito'; got data_mode={gs.data_mode!r}."
            )
        return value

    def _gb_injection_count(self) -> int:
        """Number of prior-drawn synthetic GB rows (mirrors the class counts)."""
        ids = getattr(self.general, "mojito_source_ids", None) or {}
        return len(ids.get("GB", [])) or 2  # 2 matches the stock table

    def _gb_injection_band(self) -> typing.Tuple[float, float]:
        """Frequency band for prior-drawn GB f0s: the GB branch band."""
        gb = self.gb if "gb" in self._branch_names else None
        lo = hi = None
        if gb is not None:
            center = getattr(gb, "center_freq", None)
            if center is not None:
                n_layers = int(getattr(gb, "n_layers", None) or 3)
                half = 0.5 * n_layers * self.layer_df
                lo, hi = center - half, center + half
            else:
                lo = getattr(gb, "min_freq", None)
                hi = getattr(gb, "max_freq", None)
        if lo is None:
            lo = self.general.min_freq
        if hi is None:
            hi = self.general.max_freq
        lo, hi = max(float(lo), 3.0e-4), min(float(hi), 2.3e-2)
        return (lo, hi) if lo < hi else (1.0e-3, 1.0e-2)

    def make_general_settings(self) -> EreborGeneralSettings:
        self.resolve_data_source()
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
        # window_taper_wavelets > 0 pins the taper to a fixed wavelet count
        # (deterministic per WDM settings) instead of a Tobs fraction.
        if int(getattr(gs, "window_taper_wavelets", 0) or 0) > 0:
            gs.window_tukey_alpha = min(
                1.0, 2.0 * int(gs.window_taper_wavelets) / float(Nt))
        if gs.window_taper_duration is None:
            gs.window_taper_duration = gs.window_tukey_alpha * gs.Tobs
        import logging as _logging
        _logging.getLogger(__name__).info(
            "window taper: %.1f s = %.2f wavelets per side "
            "(alpha=%.5f of Tobs=%.3e s)",
            0.5 * gs.window_taper_duration,
            0.5 * gs.window_taper_duration / (gs.Tobs / Nt),
            gs.window_tukey_alpha, gs.Tobs)

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
            verbose=gs.verbose,
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
