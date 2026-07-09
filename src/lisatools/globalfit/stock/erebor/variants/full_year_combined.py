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
import math
import os
import typing

import numpy as np

from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.emri import emri_catalogue_to_waveform_basis
from lisatools.utils.constants import YRSID_SI

from ....engine import GeneralSetup, Settings
from ....preprocessing import normalize_source_ids
from ....recipe import (
    MOJITO_REFERENCE_TIME,
    EMRIMoveBuilder,
    MBHMoveBuilder,
    SOBBHMoveBuilder,
    build_mbh_moves_phenom,
    mbh_catalogue_to_sampling_basis,
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
from ..emri import EMRISettings, EMRISetup
from ..fit import EreborFit, EreborGeneralSettings
from ..injections import (
    AnnualModulatedGalacticForeground,
    SyntheticDataProcessor,
    make_emri_injections,
    make_mbh_injections,
    make_sobbh_injections,
)
from ..mbh import MBHSettings, MBHSetup
from ..sobbh import SOBBHSettings, SOBBHSetup
from ..transforms import (
    make_emri_transform_container,
    make_mbh_transform_container,
)
from ..wrappers import (
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
from ..injections import (
    L1ProcessingStepWithSyntheticNoise,
)

logger = logging.getLogger(__name__)

# TDI-on-the-fly MBH waveform-window margin (seconds): the phentax window is
# sized from the DATA span + this margin so it is source-independent (the
# per-source merger time enters only as the call-time ``t_merge``).
MBH_TDIONFLY_MARGIN = 6.0 * 86400.0


def _default_source_ids() -> dict:
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


@dataclasses.dataclass
class FullYearMBHSettings(MBHSettings):
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
class FullYearEMRISettings(EMRISettings):
    """EMRI branch block (always the legacy ResponseWrapper path)."""

    num_prop_repeats: int = 2
    ndim: int = 12
    response_order: int = 40


@dataclasses.dataclass
class FullYearSOBBHSettings(SOBBHSettings):
    """SOBBH branch block. Default path: TDI-ON-THE-FLY (validated)."""

    num_prop_repeats: int = 2
    ndim: int = 11
    use_tdionfly: bool = dataclasses.field(
        default_factory=env_default("USE_TDIONFLY", True, bool)
    )
    n_grid: int = 2048
    buffer_time: float = 5000.0
    response_order: int = 40


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
    mojito_source_ids: dict = dataclasses.field(default_factory=_default_source_ids)
    # Mojito data window chopping (test/validation path): a merger-centered
    # snippet for MBH, ~6 months for SOBBH/EMRI.
    chop_window: bool = dataclasses.field(
        default_factory=env_default("CHOP_WINDOW", False, bool)
    )
    merger_frac: float = 0.72
    synthetic_t_start: float = 0.0

    # --- fixed sensitivity components (no psd / galfor branches) ---
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
            from lisatools.sensitivity import InstrumentNoise

            base = dict(gs.sensitivity_init_kwargs or {})
            base.setdefault("tdi_generation", gs.tdi_gen)
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
                AnnualModulatedGalacticForeground(
                    foreground_params=gs.foreground_params,
                    tdi_generation=gs.tdi_gen,
                    tobs=gs.Tobs,
                    annual_amp=gs.annual_amp,
                    annual_phase0=gs.annual_phase0,
                ),
            ]
            gs.sensitivity_init_kwargs = base

    # -- branch resolution --------------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        prep = getattr(self, f"_prepare_{name}", None)
        return prep(settings, general_setup) if prep is not None else settings

    def _catalogue(self, general_setup: GeneralSetup, cls: str) -> typing.Optional[dict]:
        cat = getattr(general_setup, "catalogue", None) or getattr(
            general_setup.data_processor, "catalogue", {}
        )
        entry = cat.get(cls) if cat else None
        # Synthetic-mode catalogue rows are empty placeholders — treat as absent.
        if entry and all(len(v) > 0 for v in entry.values() if isinstance(v, dict)):
            return entry
        return None

    def _prepare_emri(self, emri: FullYearEMRISettings, general_setup: GeneralSetup):
        from eryn.moves import StretchMove

        gs: FullYearGeneralSettings = self.general
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
        cat = self._catalogue(general_setup, "EMRI") if gs.data_mode == "mojito" else None
        if cat is not None:
            # SPECIAL EMRI frame (validated 2026-06-19): ecliptic-polar sky +
            # raw file spin angles; row construction lives in the stock
            # lisatools.sources.emri.emri_catalogue_to_waveform_basis.
            full_basis = np.asarray(
                [emri_catalogue_to_waveform_basis(cat[i]) for i in sorted(cat.keys())]
            )
        else:
            full_basis = make_emri_injections(n)
        tc = make_emri_transform_container([full_basis[0, 5], full_basis[0, -2]])
        if emri.injection is None:
            emri.injection = tc.both_inverse_transforms(full_basis)
        if getattr(emri, "fill_values", None) is None or not np.asarray(
            emri.fill_values
        ).size:
            emri.fill_values = np.array([full_basis[0, 5], full_basis[0, 12]])
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

    def _prepare_sobbh(self, sobbh: FullYearSOBBHSettings, general_setup: GeneralSetup):
        from eryn.moves import StretchMove

        from ..injections import SOBBH_INJECTION_PARAMS_FULL_BASIS  # noqa: F401
        from ..transforms import make_sobbh_transform_container

        gs: FullYearGeneralSettings = self.general
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
        cat = self._catalogue(general_setup, "SOBHB") if gs.data_mode == "mojito" else None
        if cat is not None:
            # ICRS run frame: sky + polarization read raw (RA in the lam
            # slot, Dec in the beta slot, psi ICRS); orbits loaded icrs.
            full_basis = np.asarray(
                [
                    [
                        cat[i]["PrimaryMassSSBFrame"],
                        cat[i]["SecondaryMassSSBFrame"],
                        cat[i]["PrimarySpinCompZ"],
                        cat[i]["SecondarySpinCompZ"],
                        cat[i]["LuminosityDistance"] / 1e3,  # Mpc -> Gpc
                        cat[i]["InclinationAngle"],
                        cat[i]["GW22FrequencySSBFrame"],
                        cat[i]["RightAscension"] % (2 * np.pi),
                        cat[i]["Declination"],
                        cat[i]["PolarisationAngle"] % np.pi,
                        cat[i]["TrueAnomaly"],
                    ]
                    for i in sorted(cat.keys())
                ]
            )
        else:
            full_basis = make_sobbh_injections(n)
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

    def _prepare_mbh(self, mbh: FullYearMBHSettings, general_setup: GeneralSetup):
        from eryn.moves import StretchMove
        from eryn.prior import ProbDistContainer, log_uniform, uniform_dist

        gs: FullYearGeneralSettings = self.general
        n = gs.n_injections["MBHB"]
        if mbh.initialize_kwargs is None:
            mbh.initialize_kwargs = _make_mbh_initialize_kwargs(
                mbh, general_setup, gs
            )
        cat = self._catalogue(general_setup, "MBHB") if gs.data_mode == "mojito" else None
        if cat is not None:
            injection = np.stack(
                [mbh_catalogue_to_sampling_basis(cat[i]) for i in sorted(cat.keys())],
                axis=0,
            )
        else:
            injection = make_mbh_injections(n, general_setup.Tobs)
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

    # -- runtime objects (post-deepcopy) -------------------------------------------

    def attach_runtime_objects(self) -> None:
        """Register per-branch ``signal_gen`` adapters on the runtime Setups."""
        gs: FullYearGeneralSettings = self.general
        cfg = _signal_cfg(gs, self)
        for branch in self.branch_names:
            info = self.source_info[branch]
            info.signal_gen = FullYearSignalGen(branch, info.transform, self.general_info, cfg)


def _signal_cfg(gs: FullYearGeneralSettings, fit: FullYearCombinedGlobalFit) -> dict:
    """Plain-value config consumed by the wave-wrap getters below."""
    mbh = fit.mbh if "mbh" in fit.branches else FullYearMBHSettings()
    sobbh = fit.sobbh if "sobbh" in fit.branches else FullYearSOBBHSettings()
    emri = fit.emri if "emri" in fit.branches else FullYearEMRISettings()
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


def _make_mbh_initialize_kwargs(
    mbh: FullYearMBHSettings, general_setup: GeneralSetup, gs: FullYearGeneralSettings
) -> dict:
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
# Cached domain-wrapped template generators (runtime side)
# ============================================================
_WAVE_WRAP_CACHE = {}


def _get_emri_wave_wrap(general_info, cfg):
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


def _get_sobbh_wave_wrap(general_info, cfg):
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


def _get_mbh_tdionfly_wave_wrap(general_info, cfg):
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


def _get_mbh_phenom_gen(general_info, cfg):
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


class FullYearSignalGen:
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

    def __call__(self, *params, **kwargs):
        params_in = self.transform.both_transforms(np.asarray(params, dtype=float))
        if self.branch == "emri":
            return _get_emri_wave_wrap(self.general_info, self.cfg)(*params_in, **kwargs)
        if self.branch == "sobbh":
            return _get_sobbh_wave_wrap(self.general_info, self.cfg)(*params_in, **kwargs)
        if self.branch == "mbh":
            if self.cfg["mbh_use_tdionfly"]:
                return _get_mbh_tdionfly_wave_wrap(self.general_info, self.cfg)(
                    *params_in, **kwargs
                )
            gen = _get_mbh_phenom_gen(self.general_info, self.cfg)
            return gen.get_signals_for_residuals(*params_in, **kwargs)
        raise ValueError(f"Unknown branch {self.branch!r} for FullYearSignalGen.")


# ============================================================
# Runtime move builders + setup_recipe
# ============================================================


def _build_emri_move_runtime(curr, acs, priors, state, cfg):
    wave_gen = _get_emri_wave_wrap(curr.general_info, cfg)
    _, moves = EMRIMoveBuilder(wave_gen=wave_gen).build(None, curr, acs, priors, state)
    return moves[0]


def _build_sobbh_move_runtime(curr, acs, priors, state, cfg):
    wave_gen = _get_sobbh_wave_wrap(curr.general_info, cfg)
    _, moves = SOBBHMoveBuilder(wave_gen=wave_gen).build(None, curr, acs, priors, state)
    return moves[0]


def _build_mbh_move_runtime(curr, acs, priors, state, cfg):
    """MBH PE move: stretch RJ move on the tdionfly wrap, or the stock
    ``build_mbh_moves_phenom`` builder around the cached phentax generator."""
    mbh_info = curr.source_info["mbh"]
    if not cfg["mbh_use_tdionfly"]:
        wave_gen = _get_mbh_phenom_gen(curr.general_info, cfg)
        _, move = build_mbh_moves_phenom(
            curr, acs, priors, state, wave_gen=wave_gen, subtract_initial=False
        )
        return move
    wave_gen = _get_mbh_tdionfly_wave_wrap(curr.general_info, cfg)
    _, moves = MBHMoveBuilder(
        wave_gen=wave_gen, waveform_like_kwargs=mbh_info.waveform_kwargs
    ).build(None, curr, acs, priors, state)
    return moves[0]


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
    cfg = None
    for name in curr.engine_info.branch_names:
        sg = getattr(curr.source_info[name], "signal_gen", None)
        if isinstance(sg, FullYearSignalGen):
            cfg = sg.cfg
            break
    if cfg is None:
        raise RuntimeError(
            "full_year_combined setup_recipe found no FullYearSignalGen on any "
            "branch — build through FullYearCombinedGlobalFit.build()."
        )

    stock_moves = {}
    if "mbh" in curr.source_info:
        stock_moves["mbh_pe"] = _build_mbh_move_runtime(curr, acs, priors, state, cfg)
    if "emri" in curr.source_info:
        stock_moves["emri_pe"] = _build_emri_move_runtime(curr, acs, priors, state, cfg)
    if "sobbh" in curr.source_info:
        stock_moves["sobbh_pe"] = _build_sobbh_move_runtime(curr, acs, priors, state, cfg)

    recipe_spec: RecipeSpec = curr.source_metadata["recipe_spec"]
    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state,
    )
    materialize_recipe(recipe, recipe_spec, ctx, stock_moves, ntemps, nwalkers)


FullYearCombinedGlobalFit.default_setup_function = staticmethod(setup_recipe)
