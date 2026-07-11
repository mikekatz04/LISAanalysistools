"""Stock variant ``full_year_combined``: multi-leaf MBH+EMRI+SOBBH, full year.

The installed version of
``global_fit_input/full_year_combined_global_fit_settings.py``: catalogue-
driven multi-leaf EMRI / SOBBH / MBH branches (no GB, no PSD, no galfor) on
either the mojito L1 data (default) or an all-synthetic stream, with fixed
instrument noise + an annually-modulated galactic foreground baked into the
sensitivity model.

Waveform-path defaults (2026-07-09): **SOBBH = TDI-on-the-fly** (validated,
WDM mm5 ~3.5e-7), **MBH = legacy** ``PhenomTHMTDIWaveform``, **EMRI =
legacy** ResponseWrapper. Each is a per-branch knob
(``fit.mbh.use_tdionfly`` / ``fit.sobbh.use_tdionfly``), still
env-overridable via ``USE_TDIONFLY`` (which flips both).

The per-branch settings blocks + branch resolution + engine-side
``signal_gen`` adapters + runtime move builders now live in the shared
``..source_runtime`` module so ``all_sources`` composes the identical
source setup; this variant is a thin wiring layer over them.

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.full_year_combined()
    fit.general.mojito_source_ids = {"MBHB": [0], "EMRI": [1], "SOBHB": []}
    fit.mbh.use_tdionfly = True            # swap the MBH waveform path
    fit.build(); fit.run()
"""

from __future__ import annotations

import dataclasses
import logging
import os
import typing

import numpy as np

from lisatools.utils.constants import YRSID_SI

from ....engine import GeneralSetup, Settings
from ....recipe import MOJITO_REFERENCE_TIME
from ...base import (
    MoveBuildContext,
    MoveSpec,
    RecipeSpec,
    StageSpec,
    env_default,
    env_resolve,
    materialize_recipe,
)
from ..emri import EMRISetup
from ..fit import EreborFit, EreborGeneralSettings
from ..injections import (
    AnnualCovarianceModulation,
    L1ProcessingStepWithSyntheticNoise,
    SyntheticDataProcessor,
    make_emri_injections,
    make_mbh_injections,
    make_sobbh_injections,
)
from ..mbh import MBHSetup
from ..sobbh import SOBBHSetup
from ..source_runtime import (
    SourceEMRISettings,
    SourceMBHSettings,
    SourceSOBBHSettings,
    SourceSignalGen,
    build_source_moves,
    default_source_ids,
    find_source_cfg,
    prepare_emri_branch,
    prepare_mbh_branch,
    prepare_sobbh_branch,
    source_signal_cfg,
)

logger = logging.getLogger(__name__)

# Per-branch settings blocks are the shared source_runtime bases; aliased here
# so the variant's public knob classes keep their historical names.
FullYearMBHSettings = SourceMBHSettings
FullYearEMRISettings = SourceEMRISettings
FullYearSOBBHSettings = SourceSOBBHSettings


@dataclasses.dataclass
class FullYearGeneralSettings(EreborGeneralSettings):
    """General block for ``full_year_combined``."""

    dt: float = 2.5
    # tobs_target=None -> resolved from chop_window / active source at
    # construction (full span = 1 yr by default); env TOBS_TARGET wins.
    tobs_target: typing.Optional[float] = None
    wavelet_duration_min: float = dataclasses.field(
        default_factory=env_default("WAVELET_DUR_MIN", 40000.0, float)
    )
    wavelet_duration_max: float = dataclasses.field(
        default_factory=env_default("WAVELET_DUR_MAX", 48000.0, float)
    )
    min_freq: float = 1e-4
    max_freq: float = 2.5e-2
    window_tukey_alpha: float = 0.0  # rectangular window
    edge_crop_wavelets: typing.Optional[int] = 20
    nwalkers: int = dataclasses.field(default_factory=env_default("NWALKERS", 6, int))
    ntemps: int = dataclasses.field(default_factory=env_default("NTEMPS", 3, int))
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "full_year_combined_run")
    )

    # --- data source ---
    # "mojito": L1 loader + synthetic noise/foreground; "synthetic": build
    # every stream in-process (no mojito folder needed).
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_PROCESSOR", "mojito", str)
    )
    mojito_source_ids: dict = dataclasses.field(default_factory=default_source_ids)
    # Mojito data window chopping (test/validation path): a merger-centered
    # snippet for MBH, ~6 months for SOBBH/EMRI.
    chop_window: bool = dataclasses.field(
        default_factory=env_default("CHOP_WINDOW", False, bool)
    )
    merger_frac: float = 0.72
    # synthetic_t_start inherits the Erebor default (10,000 s: keeps the
    # TDI2 warm-up look-back inside the orbit span).

    # --- fixed sensitivity components (no psd / galfor branches) ---
    # Fixed PSD (no psd branch) -> report source-only log L = -1/2 <r|r>
    # (drop the constant -sum(log|detC|) noise normalization term).
    likelihood_source_only: bool = True
    add_instrument_noise: bool = False
    noise_soms_d: float = 15e-12
    noise_sa_a: float = 3e-15
    noise_seed: int = 12345
    add_galactic_foreground: bool = False
    foreground_params: typing.Optional[typing.Sequence[float]] = None
    foreground_seed: int = 67890
    annual_amp: float = 0.10
    annual_phase0: float = 0.0

    @property
    def active_source(self) -> typing.Optional[str]:
        active = [k for k, v in self.mojito_source_ids.items() if v]
        return active[0] if len(active) == 1 else None

    @property
    def n_injections(self) -> typing.Dict[str, int]:
        return {k: len(v) for k, v in self.mojito_source_ids.items()}

    @property
    def mbh_waveform_t0(self) -> float:
        """Epoch merger times (t_plunge) are referenced to."""
        return (
            MOJITO_REFERENCE_TIME if self.data_mode == "mojito" else self.synthetic_t_start
        )

    @property
    def sobbh_reference_time(self) -> typing.Optional[float]:
        """Epoch ``f_low`` is defined at (None -> the window start)."""
        return MOJITO_REFERENCE_TIME if self.data_mode == "mojito" else None


class FullYearCombinedGlobalFit(EreborFit):
    """Multi-leaf MBH+EMRI+SOBBH fit on the full-year mojito (or synthetic) data."""

    option_name = "full_year_combined"
    description = (
        "Catalogue-driven multi-leaf MBH+EMRI+SOBBH fit over the full year "
        "(mojito L1 or synthetic data); fixed noise + annually-modulated "
        "foreground; SOBBH TDI-on-the-fly, MBH/EMRI legacy paths."
    )
    general_settings_class = FullYearGeneralSettings
    setup_classes = {"mbh": MBHSetup, "emri": EMRISetup, "sobbh": SOBBHSetup}

    def __init__(self, **knobs):
        super().__init__(**knobs)
        gs: FullYearGeneralSettings = self.general
        if gs.data_mode not in ("mojito", "synthetic"):
            raise ValueError(
                f"data_mode={gs.data_mode!r} not recognised; use 'mojito' or "
                "'synthetic'."
            )
        if sum(gs.n_injections.values()) < 1:
            raise ValueError(
                "mojito_source_ids must inject at least 1 source total across "
                "MBHB / EMRI / SOBHB (all three are currently empty)."
            )
        if gs.chop_window and gs.active_source is None:
            raise ValueError(
                "chop_window is a single-source-at-a-time snippet run: set "
                "exactly one of mojito_source_ids non-empty "
                f"(got {[k for k, v in gs.mojito_source_ids.items() if v]})."
            )
        if gs.tobs_target is None:
            # Full span (production) by default; chopped windows are shorter.
            if not gs.chop_window:
                default_tobs = YRSID_SI
            elif gs.active_source == "MBHB":
                default_tobs = 48 * 86400.0  # merger-centered snippet
            else:
                default_tobs = 0.5 * YRSID_SI  # SOBBH / EMRI: ~6 months
            gs.tobs_target = env_resolve("TOBS_TARGET", default_tobs, float)
        # Drop branches with zero injected leaves.
        for cls, branch in (("MBHB", "mbh"), ("EMRI", "emri"), ("SOBHB", "sobbh")):
            if gs.n_injections[cls] == 0 and branch in self._branch_names:
                self.remove_branch(branch)
                for stage in self.recipe.stages:
                    stage.moves = [m for m in stage.moves if m.branch != branch]

    # -- default blocks -------------------------------------------------------

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {
            "mbh": FullYearMBHSettings(),
            "emri": FullYearEMRISettings(),
            "sobbh": FullYearSOBBHSettings(),
        }

    def default_recipe(self) -> RecipeSpec:
        return RecipeSpec(
            [
                StageSpec(
                    name="full_pe",
                    kind="pe",
                    moves=[
                        MoveSpec("mbh_pe", branch="mbh"),
                        MoveSpec("emri_pe", branch="emri"),
                        MoveSpec("sobbh_pe", branch="sobbh"),
                    ],
                    combine_kwargs=dict(verbose=True, share_temperature_control=False),
                )
            ]
        )

    # -- general resolution -----------------------------------------------------

    def _mbh_chop_window_offset(self) -> float:
        """Seconds from the L1 file start at which to begin a chopped MBH window."""
        gs: FullYearGeneralSettings = self.general
        if not (gs.chop_window and gs.active_source == "MBHB"):
            return 0.0
        import glob

        import h5py

        tobs = self.wdm_grid[3]
        mbh_id = int(gs.mojito_source_ids["MBHB"][0])
        cat_files = sorted(
            glob.glob(os.path.join(gs.mojito_data_path, "catalogues", "mbhb_cat_*.hdf5"))
        )
        if not cat_files:
            logger.warning(
                "chop_window: no mbhb_cat_*.hdf5 under %s/catalogues; using "
                "offset 0.", gs.mojito_data_path,
            )
            return 0.0
        with h5py.File(cat_files[0], "r") as f:
            # MBHB catalogue row == id (0-based IDs coincide with rows).
            t_plunge = float(f["Binaries"]["TimeCoalescencePhenomTPHMSSBFrame"][mbh_id])
        offset = max(0.0, t_plunge - gs.merger_frac * tobs)
        logger.info(
            "chop_window MBH id=%d: t_plunge=%.1f s -> window_start_offset=%.1f s",
            mbh_id, t_plunge, offset,
        )
        return offset

    def set_default_processor(self, gs: FullYearGeneralSettings) -> None:
        force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
        tobs = self.wdm_grid[3]
        common = dict(
            add_instrument_noise=gs.add_instrument_noise,
            noise_soms_d=gs.noise_soms_d,
            noise_sa_a=gs.noise_sa_a,
            noise_seed=gs.noise_seed,
            add_galactic_foreground=gs.add_galactic_foreground,
            foreground_params=gs.foreground_params,
            foreground_seed=gs.foreground_seed,
            annual_amp=gs.annual_amp,
            annual_phase0=gs.annual_phase0,
            tdi_generation=gs.tdi_gen,
        )
        if gs.data_mode == "mojito":
            from lisatools.detector import L1Orbits

            gs.data_processor_class = L1ProcessingStepWithSyntheticNoise
            gs.processor_init_kwargs = dict(
                L1_folder=gs.mojito_data_path,
                source_ids={k: list(v) for k, v in gs.mojito_source_ids.items()},
                orbits_class=L1Orbits,
                orbits_kwargs=dict(force_backend=force_backend, frame=gs.orbits_frame),
                verbose=True,
                do_plots=False,
                Tobs=tobs,
                window_start_offset=self._mbh_chop_window_offset(),
                **common,
            )
        else:
            mbh = self.mbh if "mbh" in self._branch_names else FullYearMBHSettings()
            gs.data_processor_class = SyntheticDataProcessor
            gs.processor_init_kwargs = dict(
                Tobs=tobs,
                dt=gs.dt,
                t_start=gs.synthetic_t_start,
                emri_injection_params_full_basis=make_emri_injections(
                    gs.n_injections["EMRI"]
                ),
                sobbh_injection_params_full_basis=make_sobbh_injections(
                    gs.n_injections["SOBHB"]
                ),
                mbh_injection_params_sampling_basis=make_mbh_injections(
                    gs.n_injections["MBHB"], tobs
                ),
                source_ids={k: list(v) for k, v in gs.mojito_source_ids.items()},
                nchannels=gs.nchannels,
                force_backend="cpu",
                verbose=True,
                do_plots=False,
                tdi_chan=gs.tdi_chan,
                tdi_gen_str=gs.tdi_gen_str,
                sobbh_reference_time=gs.sobbh_reference_time,
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
                **common,
            )

    def default_preprocess_kwargs(self) -> dict:
        # Both processors emit exactly Tobs = Nf*Nt*dt samples (the mojito
        # loader trims at load via its Tobs kwarg); skip the engine's
        # default highpass + edge-trim + Tobs trim so the WDM shape stays
        # exact.
        return dict(highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False)

    def finalize_general(self, gs: FullYearGeneralSettings) -> None:
        # Fixed instrument noise + annually-modulated foreground baked into
        # the sensitivity model (no psd / galfor branches).
        if gs.sensitivity_init_kwargs is None or (
            isinstance(gs.sensitivity_init_kwargs, dict)
            and "extra_components" not in gs.sensitivity_init_kwargs
        ):
            from lisatools.detector import DefaultOrbits, LISAModel
            from lisatools.sensitivity import GalacticForeground, InstrumentNoise
            from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground

            base = dict(gs.sensitivity_init_kwargs or {})
            base.setdefault("tdi_generation", gs.tdi_gen)
            # Annually-modulated galactic foreground via the general modulation
            # framework: GalacticForeground(modulation=...) with the analytic
            # annual covariance modulation (a picklable callable). The fitted
            # spectral model takes (Tobs,) as its params.
            base["extra_components"] = [
                InstrumentNoise(
                    tdi_generation=gs.tdi_gen,
                    model=LISAModel(
                        gs.noise_soms_d ** 2,
                        gs.noise_sa_a ** 2,
                        DefaultOrbits(),
                        "full_year_fixed_noise",
                    ),
                    fill_nans=0.0,
                ),
                GalacticForeground(
                    foreground_params=(
                        gs.foreground_params
                        if gs.foreground_params is not None
                        else (gs.Tobs,)
                    ),
                    modulation=AnnualCovarianceModulation(gs.annual_amp, gs.annual_phase0),
                    tdi_generation=gs.tdi_gen,
                    stochastic_fn=FittedHyperbolicTangentGalacticForeground,
                ),
            ]
            gs.sensitivity_init_kwargs = base

    # -- branch resolution --------------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        gs: FullYearGeneralSettings = self.general
        if name == "emri":
            return prepare_emri_branch(settings, general_setup, gs)
        if name == "sobbh":
            return prepare_sobbh_branch(settings, general_setup, gs)
        if name == "mbh":
            return prepare_mbh_branch(settings, general_setup, gs)
        return settings

    # -- runtime objects (post-deepcopy) -------------------------------------------

    def attach_runtime_objects(self) -> None:
        """Register per-branch ``signal_gen`` adapters on the runtime Setups."""
        gs: FullYearGeneralSettings = self.general
        mbh = self.mbh if "mbh" in self.branches else FullYearMBHSettings()
        sobbh = self.sobbh if "sobbh" in self.branches else FullYearSOBBHSettings()
        emri = self.emri if "emri" in self.branches else FullYearEMRISettings()
        cfg = source_signal_cfg(gs, mbh, sobbh, emri)
        for branch in self.branch_names:
            info = self.source_info[branch]
            info.signal_gen = SourceSignalGen(branch, info.transform, self.general_info, cfg)


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``full_year_combined``.

    Moves only: residual generation/subtraction already ran under the hood
    in the engine (``setup_acs(rebuild_residuals=True)`` consuming each
    branch's registered ``signal_gen``).
    """
    general_info = curr.general_info
    nwalkers, ntemps = general_info.nwalkers, general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        import cupy as cp

        cp.cuda.runtime.setDevice(gpus[0])

    # The signal-gen adapters carry the plain-value config; reuse it for the
    # move-side wave wraps so both share the cached generators.
    cfg = find_source_cfg(curr)
    if cfg is None:
        raise RuntimeError(
            "full_year_combined setup_recipe found no SourceSignalGen on any "
            "branch — build through FullYearCombinedGlobalFit.build()."
        )

    stock_moves = build_source_moves(curr, acs, priors, state, cfg)

    recipe_spec: RecipeSpec = curr.source_metadata["recipe_spec"]
    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state,
    )
    materialize_recipe(recipe, recipe_spec, ctx, stock_moves, ntemps, nwalkers)


FullYearCombinedGlobalFit.default_setup_function = staticmethod(setup_recipe)
