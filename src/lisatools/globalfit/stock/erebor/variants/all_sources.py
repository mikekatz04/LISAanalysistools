"""Stock variant ``all_sources``: every branch, mojito or synthetic data.

Six branches (``gb``, ``psd``, ``galfor``, ``mbh``, ``emri``, ``sobbh``) on a
fixed smoke-friendly WDM grid (Nf=720, Nt=2160, dt=5 s -> 90 d; the mojito
default doubles Nf and halves dt to match the 2.5 s L1 sampling).
``general.data_mode`` selects the data pipeline: ``"mojito"`` (default —
GB galaxy + selected MBHB/EMRI/SOBHB from the L1 folder, plus synthetic
instrument noise for the psd branch), ``"synthetic"`` (all streams built
in-process, no external data), or ``"sangria"`` (the legacy LDC file).

This variant is a **composition of the canonical per-branch setups** — each
branch is the same shared code the specialized variant runs:

* **GB** = the ``gb_no_fg`` setup (``prepare_gb_branch`` + ``setup_gb_moves``:
  chunked-het WDM likelihood, catalogue true-point seeding, dynamic
  nleaves_max), with the GB band widened to the full data band so the f0
  prior extends down to ~0.1 mHz (the only intended difference).
* **PSD / galfor** = the ``noise`` setup (``prepare_psd_branch`` /
  ``prepare_galfor_branch`` + ``noise_sensitivity_init_kwargs`` threading the
  per-branch noise-model choice onto the CompositeSensitivityBackend); both
  ride the single PSDMove.
* **MBH / EMRI / SOBBH** = the ``full_year_combined`` setup
  (``..source_runtime``: phentax MBH via ``build_mbh_moves_phenom``,
  TDI-on-the-fly SOBBH, legacy-ResponseWrapper EMRI, wide priors,
  engine-side ``SourceSignalGen`` residual subtraction).

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.all_sources(nwalkers=4)
    fit.remove_branch("galfor")            # sub whole branches in/out
    fit.pop_move("sobbh_pe")               # sub moves in/out
    fit.mbh.use_tdionfly = True            # swap the MBH waveform path
    fit.build(); fit.run()
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np

from ....engine import GeneralSetup, Settings
from ....recipe import MOJITO_REFERENCE_TIME, build_psd_moves
from ...base import (
    MoveBuildContext,
    MoveSpec,
    RecipeSpec,
    StageSpec,
    env_default,
    materialize_recipe,
)
from ..common import tdi_generation_info
from ..emri import EMRISetup
from ..fit import EreborFit, EreborGeneralSettings
from ..gb import GBSetup
from ..mbh import MBHSetup
from ..noise import (
    GalForSettings,
    GalForSetup,
    PSDSetup,
    noise_sensitivity_init_kwargs,
    prepare_galfor_branch,
    prepare_psd_branch,
)
from ..sobbh import SOBBHSetup
from ..source_runtime import (
    SourceEMRISettings,
    SourceMBHSettings,
    SourceSOBBHSettings,
    SourceSignalGen,
    build_source_moves,
    find_source_cfg,
    make_emri_injections,
    make_mbh_injections,
    make_sobbh_injections,
    prepare_emri_branch,
    prepare_mbh_branch,
    prepare_sobbh_branch,
    source_signal_cfg,
)
from ..stochastic import SGWBSetup
from .gb_no_fg import GB_MOJITO_T_REF, GBNoFgGBSettings, prepare_gb_branch, setup_gb_moves
from .noise import (
    GALFOR_INJECTION,
    PSD_INJECTION,
    NoisePSDSettings,
    NoiseSGWBSettings,
)

logger = logging.getLogger(__name__)

# Kept as the historical public name (the mojito catalogue epoch; also the
# legacy constant used for the sangria stream's GB reference).
GB_T_REF = GB_MOJITO_T_REF

# Default synthetic-data start (matches EreborGeneralSettings.synthetic_t_start):
# 10,000 s keeps the TDI2 warm-up look-back inside the orbit span.
_SYNTH_T0_DEFAULT = 10_000.0

_SOURCE_BRANCH_TO_CLASS = {"mbh": "MBHB", "emri": "EMRI", "sobbh": "SOBHB"}


@dataclasses.dataclass
class AllSourcesGBSettings(GBNoFgGBSettings):
    """GB branch block: the ``gb_no_fg`` setup on the FULL data band.

    Identical to :class:`GBNoFgGBSettings` except the GB band spans the whole
    data band, extending the f0 prior floor down to ~0.1 mHz (snapped inward
    to WDM layer boundaries like every gb_no_fg band).
    """

    min_freq: float = dataclasses.field(
        default_factory=env_default("GB_MIN_FREQ", 1e-4, float)
    )
    max_freq: float = dataclasses.field(
        default_factory=env_default("GB_MAX_FREQ", 2.5e-2, float)
    )


# Per-branch source blocks are the shared source_runtime bases (identical to
# full_year_combined); aliased so the variant's knob classes keep their names.
AllSourcesMBHSettings = SourceMBHSettings
AllSourcesEMRISettings = SourceEMRISettings
AllSourcesSOBBHSettings = SourceSOBBHSettings
# PSD / SGWB blocks are the noise-variant blocks (identical to noise_only /
# noise_sgwb: 2-param instrument PSD, 2-param power-law SGWB).
AllSourcesPSDSettings = NoisePSDSettings
AllSourcesSGWBSettings = NoiseSGWBSettings


@dataclasses.dataclass
class AllSourcesGeneralSettings(EreborGeneralSettings):
    """General block for ``all_sources`` (defaults mirror the legacy file)."""

    dt: float = 5.0
    nf: typing.Optional[int] = 720
    nt: typing.Optional[int] = 2160
    min_freq: float = 1e-4
    max_freq: float = 2.5e-2
    window_tukey_alpha: float = 0.0  # rectangular window
    edge_crop_wavelets: typing.Optional[int] = 20
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "global_fit_smoke_test")
    )
    # The psd branch is FIT here, so keep the noise-normalization
    # ``-logdet_factor * sum log det C`` term in the likelihood.
    likelihood_source_only: bool = False
    # Data source: "mojito" (default) loads the GB galaxy + selected
    # MBHB/EMRI/SOBHB sources from a mojito L1 folder (+ synthetic
    # instrument noise for the psd branch); "synthetic" builds every
    # present branch's stream in-process (no external data needed);
    # "sangria" (legacy) loads the LDC training file.
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_PROCESSOR", "mojito", str)
    )
    # Mojito-mode source selection per class (GB loads the whole galaxy
    # file; the id list just gates inclusion). Filtered by the branches
    # actually present on the fit. In synthetic mode the per-class COUNTS
    # set how many stock injections are built.
    mojito_source_ids: dict = dataclasses.field(
        default_factory=lambda: {"GB": [0], "MBHB": [0], "EMRI": [1], "SOBHB": [0]}
    )
    # Mojito L1 signals are noiseless: add synthetic instrument noise so the
    # psd branch has something to fit (the loaded GB galaxy itself plays the
    # role of the foreground for the galfor branch). The same knob feeds the
    # synthetic-mode noise stream.
    add_instrument_noise: bool = True
    noise_soms_d: float = 15e-12
    noise_sa_a: float = 3e-15
    noise_seed: int = 12345
    # Synthetic-data start time. Default 10,000 s so the TDI2 warm-up
    # look-back stays inside the orbit span (which starts at t=0); the
    # legacy Sangria stream itself starts at 0 (see ``mbh_waveform_t0``).
    t_start: float = _SYNTH_T0_DEFAULT
    # data_mode="synthetic": GB rows in the GBGPU basis
    # ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``; None -> the
    # stock two-source table.
    gb_injection_params: typing.Optional[typing.Any] = None

    # --- noise-branch knobs (identical to the noise variants) ---
    # PSD injection ``[Soms_d, Sa_a]`` consumed by prepare_psd_branch.
    psd_injection: typing.Sequence[float] = dataclasses.field(
        default_factory=lambda: list(PSD_INJECTION)
    )
    # Galactic-foreground truth reference (informational; the foreground in
    # the DATA is the GB galaxy itself in mojito mode).
    galfor_injection: typing.Sequence[float] = dataclasses.field(
        default_factory=lambda: list(GALFOR_INJECTION)
    )
    # Foreground time modulation: None (default) => stationary; set a path
    # (or GALFOR_MODULATION_PATH) to load a tabulated GalForTimeModulation.
    galfor_modulation_path: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("GALFOR_MODULATION_PATH", None, str)
    )

    # --- opt-in SGWB branch (env ALL_SOURCES_SGWB=1) ---
    # Default off keeps the six-branch behaviour unchanged; when on, a
    # power-law SGWB is fit jointly with psd+galfor through the same PSDMove.
    # Note: the stock data processors do not yet inject an SGWB, so enable
    # this only with data that contains one (or extend the processor).
    fit_sgwb: bool = dataclasses.field(
        default_factory=env_default("ALL_SOURCES_SGWB", False, bool)
    )
    sgwb_injection: typing.Sequence[float] = dataclasses.field(
        default_factory=lambda: [-9.5, 2.0 / 3.0]
    )
    sgwb_stochastic_fn: str = "PowerLawSGWB"

    # --- properties read by the shared source_runtime glue (duck-typed) ---
    @property
    def active_source(self) -> typing.Optional[str]:
        active = [
            k for k, v in self.mojito_source_ids.items() if v and k != "GB"
        ]
        return active[0] if len(active) == 1 else None

    @property
    def n_injections(self) -> typing.Dict[str, int]:
        return {k: len(v) for k, v in self.mojito_source_ids.items()}

    @property
    def mbh_waveform_t0(self) -> float:
        """Epoch merger times (t_plunge) are referenced to."""
        if self.data_mode == "mojito":
            return MOJITO_REFERENCE_TIME
        if self.data_mode == "sangria":
            return 0.0  # the legacy Sangria stream starts at 0
        return float(self.t_start)

    @property
    def sobbh_reference_time(self) -> typing.Optional[float]:
        """Epoch ``f_low`` is defined at (None -> the window start)."""
        return MOJITO_REFERENCE_TIME if self.data_mode == "mojito" else None


class AllSourcesGlobalFit(EreborFit):
    """Six-branch global fit (gb+psd+galfor+mbh+emri+sobbh), mojito default."""

    option_name = "all_sources"
    description = (
        "All six branches (gb, psd, galfor, mbh, emri, sobbh) composed from "
        "the canonical per-branch setups (gb_no_fg GB down to 0.1 mHz, noise "
        "PSD/galfor, full_year phentax MBH + EMRI + SOBBH) on mojito L1 "
        "(default), synthetic, or legacy Sangria data."
    )
    general_settings_class = AllSourcesGeneralSettings
    setup_classes = {
        "gb": GBSetup,
        "psd": PSDSetup,
        "galfor": GalForSetup,
        "mbh": MBHSetup,
        "emri": EMRISetup,
        "sobbh": SOBBHSetup,
        "sgwb": SGWBSetup,
    }

    def adjust_general(self, gs: AllSourcesGeneralSettings) -> None:
        # Mojito L1 data is sampled at dt = 2.5 s (the class defaults are the
        # Sangria-legacy dt = 5 grid). When the user left the grid at its
        # defaults, keep the 90-d span and ~1-h wavelets by doubling Nf.
        if gs.data_mode == "mojito" and gs.dt == 5.0:
            gs.dt = 2.5
            if gs.nf == 720 and gs.nt == 2160:
                gs.nf = 1440
        # Keep the legacy ``t_start`` knob and the inherited
        # ``synthetic_t_start`` in sync (either may be set; a value that
        # differs from the shared default is the user's).
        if gs.t_start != _SYNTH_T0_DEFAULT:
            gs.synthetic_t_start = gs.t_start
        elif gs.synthetic_t_start != _SYNTH_T0_DEFAULT:
            gs.t_start = gs.synthetic_t_start
        # gb_no_fg's ACA data-band clipping is a single-band memory knob; in
        # this multi-source variant the data band must cover every branch.
        gb = getattr(self, "gb", None) if "gb" in self._branch_names else None
        if gb is not None and getattr(gb, "data_band_layers", None) is not None:
            logger.warning(
                "all_sources ignores gb.data_band_layers (the data band must "
                "cover the MBH/EMRI/SOBBH branches); narrowing the GB band "
                "itself is gb.min_freq/max_freq."
            )

    def default_branches(self) -> typing.Dict[str, Settings]:
        branches = {
            "gb": AllSourcesGBSettings(),
            "psd": AllSourcesPSDSettings(),
            "galfor": GalForSettings(),
            "mbh": AllSourcesMBHSettings(),
            "emri": AllSourcesEMRISettings(),
            "sobbh": AllSourcesSOBBHSettings(),
        }
        if self.general.fit_sgwb:
            branches["sgwb"] = AllSourcesSGWBSettings()
        return branches

    def default_recipe(self) -> RecipeSpec:
        # One combined PE stage, legacy order: psd, mbh, emri, sobbh, gb.
        return RecipeSpec(
            [
                StageSpec(
                    name="full_pe",
                    kind="pe",
                    moves=[
                        MoveSpec("psd_pe", branch="psd"),
                        MoveSpec("mbh_pe", branch="mbh"),
                        MoveSpec("emri_pe", branch="emri"),
                        MoveSpec("sobbh_pe", branch="sobbh"),
                        MoveSpec("rj_prior", branch="gb"),
                    ],
                    combine_kwargs=dict(verbose=True, share_temperature_control=False),
                )
            ]
        )

    # -- data processors ------------------------------------------------------

    def set_default_processor(self, gs: AllSourcesGeneralSettings) -> None:
        if gs.data_mode == "mojito":
            from ..injections import L1ProcessingStepWithSyntheticNoise

            force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
            branch_to_class = {
                "gb": "GB", "mbh": "MBHB", "emri": "EMRI", "sobbh": "SOBHB"
            }
            source_ids = {
                cls: list(gs.mojito_source_ids.get(cls, []))
                for branch, cls in branch_to_class.items()
                if branch in self._branch_names and gs.mojito_source_ids.get(cls)
            }
            if not source_ids:
                raise ValueError(
                    "all_sources data_mode='mojito' has no loadable classes "
                    "(check mojito_source_ids vs the present branches)."
                )
            gs.data_processor_class = L1ProcessingStepWithSyntheticNoise
            gs.processor_init_kwargs = dict(
                L1_folder=gs.mojito_data_path,
                source_ids=source_ids,
                orbits_kwargs=dict(
                    force_backend=force_backend, frame=gs.orbits_frame
                ),
                verbose=True,
                do_plots=False,
                Tobs=gs.Tobs,
                add_instrument_noise=gs.add_instrument_noise,
                noise_soms_d=gs.noise_soms_d,
                noise_sa_a=gs.noise_sa_a,
                noise_seed=gs.noise_seed,
                tdi_generation=tdi_generation_info(gs.tdi_chan)[0],
            )
            return
        if gs.data_mode == "sangria":
            from lisatools.globalfit.preprocessing import SangriaProcessingStep

            gs.data_processor_class = SangriaProcessingStep
            gs.processor_init_kwargs = dict(data_input_path=gs.ldc_source_file)
            return
        if gs.data_mode == "synthetic":
            # In-process streams for the PRESENT branches, summed by
            # SyntheticCombinedProcessingStep: GB via the same generator as
            # gb_no_fg; MBH/EMRI/SOBBH (+ instrument noise for the psd
            # branch) via full_year's SyntheticDataProcessor, which uses the
            # SAME stock injections/generators as the branch templates so
            # the residual nulls at the injected start.
            from ..injections import (
                GB_INJECTION_PARAMS,
                SyntheticCombinedProcessingStep,
                SyntheticDataProcessor,
                SyntheticGBProcessingStep,
            )

            use_tdi2 = tdi_generation_info(gs.tdi_chan)[0] == 2
            specs = []
            if "gb" in self._branch_names:
                params = (
                    gs.gb_injection_params
                    if gs.gb_injection_params is not None
                    else GB_INJECTION_PARAMS
                )
                specs.append(
                    (
                        SyntheticGBProcessingStep,
                        dict(
                            injection_params=np.atleast_2d(
                                np.asarray(params, dtype=float)
                            ),
                            tdi_chan=gs.tdi_chan,
                            use_tdi2=use_tdi2,
                            force_backend="cpu",
                        ),
                    )
                )
            source_branches = [
                b for b in _SOURCE_BRANCH_TO_CLASS if b in self._branch_names
            ]
            if source_branches:
                # Injection counts gated by branch presence (a removed branch
                # injects nothing).
                n_inj = {
                    cls: (
                        gs.n_injections.get(cls, 0)
                        if branch in self._branch_names
                        else 0
                    )
                    for branch, cls in _SOURCE_BRANCH_TO_CLASS.items()
                }
                source_ids = {
                    cls: (
                        list(gs.mojito_source_ids.get(cls, []))
                        if branch in self._branch_names
                        else []
                    )
                    for branch, cls in _SOURCE_BRANCH_TO_CLASS.items()
                }
                mbh = (
                    self.mbh
                    if "mbh" in self._branch_names
                    else AllSourcesMBHSettings()
                )
                # Same (mode, seed) as the branch prep -> identical tables.
                inj_mode = gs.synthetic_injections or "stock"
                inj_seed = gs.synthetic_injection_seed
                specs.append(
                    (
                        SyntheticDataProcessor,
                        dict(
                            emri_injection_params_full_basis=make_emri_injections(
                                n_inj["EMRI"], mode=inj_mode, seed=inj_seed
                            ),
                            sobbh_injection_params_full_basis=make_sobbh_injections(
                                n_inj["SOBHB"], mode=inj_mode, seed=inj_seed
                            ),
                            mbh_injection_params_sampling_basis=make_mbh_injections(
                                n_inj["MBHB"], gs.Tobs, mode=inj_mode, seed=inj_seed
                            ),
                            source_ids=source_ids,
                            force_backend="cpu",
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
                            add_instrument_noise=gs.add_instrument_noise,
                            noise_soms_d=gs.noise_soms_d,
                            noise_sa_a=gs.noise_sa_a,
                            noise_seed=gs.noise_seed,
                            tdi_generation=gs.tdi_gen,
                        ),
                    )
                )
            if not specs:
                raise ValueError(
                    "all_sources data_mode='synthetic' has no injectable "
                    "branches (gb/mbh/emri/sobbh all removed)."
                )
            gs.data_processor_class = SyntheticCombinedProcessingStep
            gs.processor_init_kwargs = dict(
                Tobs=gs.Tobs,
                dt=gs.dt,
                t_start=gs.t_start,
                processor_specs=specs,
                nchannels=gs.nchannels,
            )
            return
        raise ValueError(
            f"all_sources data_mode={gs.data_mode!r} not recognised; use "
            "'mojito', 'synthetic', or 'sangria' (or swap "
            "data_processor_class wholesale)."
        )

    def default_preprocess_kwargs(self) -> dict:
        if self.general.data_mode in ("mojito", "synthetic"):
            # Both emit exactly Tobs = Nf*Nt*dt samples (the mojito loader
            # trims at load via its Tobs kwarg); skip the engine's default
            # highpass + edge-trim + Tobs trim.
            return dict(
                highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False
            )
        # Sangria is a year-long stream: keep the engine's default
        # highpass + edge-trim + Tobs trim so the final data is exactly
        # Nf*Nt samples (the window is built on the processed grid).
        return dict(normalize=False)

    # -- sensitivity backend: thread the per-branch noise-MODEL choice, same
    #    as the noise variants (fit.galfor.stochastic_fn / .modulation /
    #    fit.psd.instrument_model_cls / fit.sgwb.stochastic_fn) --

    def finalize_general(self, gs: AllSourcesGeneralSettings) -> None:
        gs.sensitivity_init_kwargs = noise_sensitivity_init_kwargs(
            gs.sensitivity_init_kwargs,
            tdi_generation=gs.tdi_gen,
            galfor=getattr(self, "galfor", None) if "galfor" in self._branch_names else None,
            psd=getattr(self, "psd", None) if "psd" in self._branch_names else None,
            galfor_modulation_path=gs.galfor_modulation_path,
            extra=self._sgwb_sens_kwargs(gs),
        )

    def _sgwb_sens_kwargs(self, gs: AllSourcesGeneralSettings) -> dict:
        if "sgwb" not in self._branch_names:
            return {}
        sgwb = getattr(self, "sgwb", None)
        fn = getattr(sgwb, "stochastic_fn", None) if sgwb is not None else None
        return dict(sgwb_stochastic_fn=fn or gs.sgwb_stochastic_fn)

    # -- branch resolution --------------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        prep = getattr(self, f"_prepare_{name}", None)
        return prep(settings, general_setup) if prep is not None else settings

    def _prepare_gb(self, gb: AllSourcesGBSettings, general_setup: GeneralSetup):
        if gb.t0 is None and self.general.data_mode == "sangria":
            # Legacy constant (the Sangria stream carries no catalogue epoch).
            gb.t0 = GB_T_REF
        return prepare_gb_branch(
            gb,
            general_setup,
            data_mode=self.general.data_mode,
            synthetic_t_start=float(self.general.t_start),
        )

    def _prepare_psd(self, psd: AllSourcesPSDSettings, general_setup: GeneralSetup):
        return prepare_psd_branch(psd, self.general.psd_injection)

    def _prepare_galfor(self, galfor: GalForSettings, general_setup: GeneralSetup):
        return prepare_galfor_branch(galfor)

    def _prepare_sgwb(self, sgwb: AllSourcesSGWBSettings, general_setup: GeneralSetup):
        from eryn.prior import ProbDistContainer, uniform_dist

        if sgwb.initialize_kwargs is None:
            sgwb.initialize_kwargs = {}
        if sgwb.priors is None:
            # Wide enough to contain a detectable log10_A injection (the stock
            # (-22, -18) default would exclude it).
            sgwb.priors = {
                "sgwb": ProbDistContainer(
                    {0: uniform_dist(-16.0, -9.0), 1: uniform_dist(-1.0, 2.0)}
                )
            }
        if sgwb.injection is None:
            sgwb.injection = np.asarray(self.general.sgwb_injection, dtype=float)
        return sgwb

    def _prepare_mbh(self, mbh: AllSourcesMBHSettings, general_setup: GeneralSetup):
        return prepare_mbh_branch(mbh, general_setup, self.general)

    def _prepare_emri(self, emri: AllSourcesEMRISettings, general_setup: GeneralSetup):
        return prepare_emri_branch(emri, general_setup, self.general)

    def _prepare_sobbh(self, sobbh: AllSourcesSOBBHSettings, general_setup: GeneralSetup):
        return prepare_sobbh_branch(sobbh, general_setup, self.general)

    # -- runtime objects (post-deepcopy) -------------------------------------------

    def attach_runtime_objects(self) -> None:
        """Register engine-side ``signal_gen`` adapters on the source Setups.

        Only mbh/emri/sobbh get one (the engine builds/subtracts their
        templates in ``setup_acs``); gb/psd/galfor keep the recipe-side path.
        """
        source_branches = [
            b for b in _SOURCE_BRANCH_TO_CLASS if b in self.branch_names
        ]
        if not source_branches:
            return
        gs: AllSourcesGeneralSettings = self.general
        mbh = self.mbh if "mbh" in self.branches else AllSourcesMBHSettings()
        sobbh = self.sobbh if "sobbh" in self.branches else AllSourcesSOBBHSettings()
        emri = self.emri if "emri" in self.branches else AllSourcesEMRISettings()
        cfg = source_signal_cfg(gs, mbh, sobbh, emri)
        for branch in source_branches:
            info = self.source_info[branch]
            info.signal_gen = SourceSignalGen(
                branch, info.transform, self.general_info, cfg
            )


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``all_sources``: the canonical per-branch move stacks.

    GB rides ``setup_gb_moves`` (the gb_no_fg stack), PSD/galfor/sgwb the
    single PSDMove, and MBH/EMRI/SOBBH the shared source_runtime builders
    (their residual subtraction already ran engine-side via ``signal_gen``).
    """
    general_info = curr.general_info
    nwalkers, ntemps = general_info.nwalkers, general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        import cupy as cp

        cp.cuda.runtime.setDevice(gpus[0])

    recipe_spec: RecipeSpec = curr.source_metadata["recipe_spec"]
    requested = [
        mv.name
        for stage in recipe_spec.stages
        for mv in stage.moves
        if mv.instance is None and mv.target is None
    ]
    stock_moves = {}

    # --- GB: the shared gb_no_fg stack (chunked-het likelihood, seeding, moves) ---
    if "gb" in curr.source_info and any(
        n.startswith("rj_") or n.startswith("gb_") for n in requested
    ):
        stock_moves.update(setup_gb_moves(engine_info, curr, acs, priors, state))

    # --- PSD (galfor + sgwb ride the same PSDMove) ---
    if "psd" in curr.source_info and any(n.startswith("psd") for n in requested):
        psd_info = curr.source_info["psd"]
        num_repeats_psd = getattr(psd_info, "num_repeats", None)
        if num_repeats_psd is None:
            num_repeats_psd = 5 if gpus is None else 60
        psd_search_move, psd_pe_move = build_psd_moves(
            engine_info, curr, acs, priors, num_repeats=num_repeats_psd
        )
        stock_moves["psd_pe"] = psd_pe_move
        stock_moves["psd_search"] = psd_search_move

    # --- MBH / EMRI / SOBBH: the shared full_year runtime builders ---
    source_present = [b for b in _SOURCE_BRANCH_TO_CLASS if b in curr.source_info]
    if source_present and any(f"{b}_pe" in requested for b in source_present):
        cfg = find_source_cfg(curr)
        if cfg is None:
            raise RuntimeError(
                "all_sources setup_recipe found no SourceSignalGen on any "
                "source branch — build through AllSourcesGlobalFit.build()."
            )
        stock_moves.update(build_source_moves(curr, acs, priors, state, cfg))

    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state,
    )
    materialize_recipe(recipe, recipe_spec, ctx, stock_moves, ntemps, nwalkers)


AllSourcesGlobalFit.default_setup_function = staticmethod(setup_recipe)
