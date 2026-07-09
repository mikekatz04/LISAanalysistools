"""Stock variant ``all_sources``: every branch, Sangria data, smoke-scale grid.

The installed version of ``global_fit_input/global_fit_settings.py``: six
branches (``gb``, ``psd``, ``galfor``, ``mbh``, ``emri``, ``sobbh``) on the
Sangria training data with a fixed smoke-friendly WDM grid (Nf=720, Nt=2160,
dt=5 s -> 90 d). Waveform paths are the file's legacy ones: MBH via
``BBHWaveformFD`` + ``MBHSpecialMove``, EMRI/SOBBH via the cached legacy
``ResponseWrapper`` wave wraps (the TDI-on-the-fly builders live in
``..wrappers`` when a run wants to swap them in).

Two latent bugs in the legacy file are fixed by construction: the
``data_processor=`` (vs ``data_processor_class=``) kwarg, and the
pre-ICRS ``lam_lims``/``beta_lims`` GB kwargs (now ``alpha``/``delta``).

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.all_sources(nwalkers=4)
    fit.remove_branch("galfor")            # sub whole branches in/out
    fit.pop_move("sobbh_pe")               # sub moves in/out
    fit.build(); fit.run()
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import typing

import numpy as np

from gbgpu.utils.utility import get_fdot

from lisatools.domains import TDSettings, WDMSettings
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI

from ....engine import GeneralSetup, Settings
from ....recipe import build_gb_moves, build_psd_moves
from ...base import (
    MoveBuildContext,
    MoveSpec,
    RecipeSpec,
    StageSpec,
    env_default,
    materialize_recipe,
)
from ..emri import EMRISettings, EMRISetup
from ..fit import EreborFit, EreborGeneralSettings
from ..gb import GBSettings, GBSetup
from ..injections import (
    INJECTION_PARAMS_FULL_BASIS,
    SOBBH_INJECTION_PARAMS_FULL_BASIS,
    emri_full_to_sampling,
    sobbh_full_to_sampling,
)
from ..mbh import MBHSettings, MBHSetup
from ..noise import GalForSettings, GalForSetup, PSDSettings, PSDSetup
from ..sobbh import SOBBHSettings, SOBBHSetup
from ..wrappers import (
    EMRIWaveWrap,
    SOBBHWaveWrap,
    get_emri_response_wrapper,
    get_sobbh_response_wrapper,
)

logger = logging.getLogger(__name__)

_DELTA_SAFE = 1e-5

# GB phase/frequency reference epoch (mojito catalogue epoch; kept for
# consistency with the other variants — TODO in the legacy file: derive
# from orbits).
GB_T_REF = 97729089.327664


@dataclasses.dataclass
class AllSourcesGBSettings(GBSettings):
    """GB branch block for ``all_sources`` (full-band, ICRS sky)."""

    A_lims: typing.List[float] = dataclasses.field(default_factory=lambda: [7e-26, 1e-19])
    f0_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.05e-3, 2.5e-2]
    )
    m_chirp_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.001, 1.0]
    )
    phi0_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.0, 2 * np.pi]
    )
    iota_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.0 + _DELTA_SAFE, np.pi - _DELTA_SAFE]
    )
    psi_lims: typing.List[float] = dataclasses.field(default_factory=lambda: [0.0, np.pi])
    alpha_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.0, 2 * np.pi]
    )
    delta_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [-np.pi / 2.0 + _DELTA_SAFE, np.pi / 2.0 - _DELTA_SAFE]
    )
    ndim: int = 8
    nleaves_min: int = 0
    nleaves_max: int = dataclasses.field(default_factory=env_default("GB_NLEAVES_MAX", 100, int))
    t0: float = GB_T_REF
    oversample: int = 4
    extra_buffer: int = 5
    start_freq_ind: int = 0
    # chunked-het kernel sizes (WDM likelihood)
    nt_sub: int = dataclasses.field(default_factory=env_default("CHUNKED_NT_SUB", 256, int))
    n_pad: int = dataclasses.field(default_factory=env_default("CHUNKED_N_PAD", 32, int))
    n_sparse: int = dataclasses.field(default_factory=env_default("CHUNKED_N_SPARSE", 256, int))
    n_cp_sig: int = dataclasses.field(default_factory=env_default("CHUNKED_N_CP_SIG", 48, int))
    n_cp_orbit: int = dataclasses.field(
        default_factory=env_default("CHUNKED_N_CP_ORBIT", 32, int)
    )


@dataclasses.dataclass
class AllSourcesMBHSettings(MBHSettings):
    """MBH branch block: legacy ``BBHWaveformFD`` path (PhenomD, AET)."""

    nleaves_max: int = 15
    nleaves_min: int = 0
    ndim: int = 11


@dataclasses.dataclass
class AllSourcesPSDSettings(PSDSettings):
    ndim: int = 2
    # PSD proposal repeats per iteration (5 on CPU / 60 on GPU when None).
    num_repeats: typing.Optional[int] = None


@dataclasses.dataclass
class AllSourcesEMRISettings(EMRISettings):
    """EMRI branch: legacy ResponseWrapper path, tight prior around injection."""

    num_prop_repeats: int = 2
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 12
    delta_prior: float = 1e-2
    response_order: int = 40


@dataclasses.dataclass
class AllSourcesSOBBHSettings(SOBBHSettings):
    """SOBBH branch: legacy ResponseWrapper path, tight prior around injection."""

    num_prop_repeats: int = 2
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 11
    delta_prior: float = 1e-2
    response_order: int = 40


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
    # Synthetic-data start time (Sangria stream starts at 0).
    t_start: float = 0.0


class AllSourcesGlobalFit(EreborFit):
    """Six-branch global fit (gb+psd+galfor+mbh+emri+sobbh) on Sangria data."""

    option_name = "all_sources"
    description = (
        "All six branches (gb, psd, galfor, mbh, emri, sobbh) on the Sangria "
        "training data; fixed 90-d WDM grid; legacy MBH/EMRI/SOBBH waveform "
        "paths."
    )
    general_settings_class = AllSourcesGeneralSettings
    setup_classes = {
        "gb": GBSetup,
        "psd": PSDSetup,
        "galfor": GalForSetup,
        "mbh": MBHSetup,
        "emri": EMRISetup,
        "sobbh": SOBBHSetup,
    }

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {
            "gb": AllSourcesGBSettings(),
            "psd": AllSourcesPSDSettings(),
            "galfor": GalForSettings(),
            "mbh": AllSourcesMBHSettings(),
            "emri": AllSourcesEMRISettings(),
            "sobbh": AllSourcesSOBBHSettings(),
        }

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

    def set_default_processor(self, gs: AllSourcesGeneralSettings) -> None:
        from lisatools.globalfit.preprocessing import SangriaProcessingStep

        gs.data_processor_class = SangriaProcessingStep
        gs.processor_init_kwargs = dict(data_input_path=gs.ldc_source_file)

    def default_preprocess_kwargs(self) -> dict:
        # Sangria is a year-long stream: keep the engine's default
        # highpass + edge-trim + Tobs trim so the final data is exactly
        # Nf*Nt samples (the window is built on the processed grid). A
        # synthetic processor that already emits exactly Nf*Nt samples
        # should instead set fit.general.preprocess_kwargs =
        # dict(highpass_kwargs=None, trim_kwargs=None, Tobs=None,
        # normalize=False).
        return dict(normalize=False)

    # -- branch resolution --------------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        prep = getattr(self, f"_prepare_{name}", None)
        return prep(settings, general_setup) if prep is not None else settings

    def _prepare_gb(self, gb: AllSourcesGBSettings, general_setup: GeneralSetup):
        if not gb.fdot_lims:
            fdot_max_val = get_fdot(gb.f0_lims[-1], Mc=gb.m_chirp_lims[-1])
            gb.fdot_lims = [-fdot_max_val, fdot_max_val]
        domain_settings = general_setup.domain_settings
        gb.start_freq = float(getattr(domain_settings, "min_freq", None) or 5e-5)
        gb.end_freq = float(getattr(domain_settings, "max_freq", None) or 3e-2)
        gb.tdi_setup = general_setup.tdi_chan
        gb.use_tdi2 = True
        gb.initialize_kwargs = dict(force_backend=general_setup.force_backend)
        if gb.betas is None:
            betas = 1.0 / 1.2 ** np.arange(general_setup.ntemps)
            betas[-1] = 1e-4
            gb.betas = betas
        gb.gb_wdm_comp = None
        return gb

    def _prepare_mbh(self, mbh: AllSourcesMBHSettings, general_setup: GeneralSetup):
        if mbh.initialize_kwargs is None:
            from lisatools.detector import EqualArmlengthOrbits

            gpu_orbits = EqualArmlengthOrbits(force_backend=general_setup.force_backend)
            mbh.initialize_kwargs = dict(
                amp_phase_kwargs=dict(run_phenomd=True),
                response_kwargs=dict(TDItag="AET", orbits=gpu_orbits),
                force_backend=general_setup.force_backend,
            )
        return mbh

    def _prepare_psd(self, psd: AllSourcesPSDSettings, general_setup: GeneralSetup):
        from eryn.prior import ProbDistContainer, uniform_dist

        if psd.initialize_kwargs is None:
            psd.initialize_kwargs = dict()
        if psd.priors is None:
            psd.priors = {
                "psd": ProbDistContainer(
                    {
                        r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),
                        r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),
                    }
                )
            }
        if psd.injection is None:
            psd.injection = np.array([15e-12, 3e-15])
        return psd

    def _prepare_galfor(self, galfor: GalForSettings, general_setup: GeneralSetup):
        if galfor.initialize_kwargs is None:
            galfor.initialize_kwargs = {}
        return galfor

    def _prepare_emri(self, emri: AllSourcesEMRISettings, general_setup: GeneralSetup):
        from eryn.moves import StretchMove

        force_backend = general_setup.force_backend
        t_start = float(self.general.t_start)
        if emri.initialize_kwargs is None:
            emri.initialize_kwargs = dict(
                T=general_setup.Tobs / YRSID_SI,
                dt=general_setup.dt,
                emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
                emri_waveform_kwargs=dict(force_backend=force_backend),
                response_kwargs=dict(
                    t0=t_start,
                    order=emri.response_order,
                    tdi="2nd generation",
                    tdi_chan=general_setup.tdi_chan,
                    force_backend=force_backend,
                    remove_garbage="zero",
                ),
            )
        if emri.waveform_kwargs is None:
            emri.waveform_kwargs = dict()
        if emri.injection is None:
            emri.injection = emri_full_to_sampling(INJECTION_PARAMS_FULL_BASIS)
        if getattr(emri, "fill_values", None) is None:
            emri.fill_values = np.array(
                [INJECTION_PARAMS_FULL_BASIS[5], INJECTION_PARAMS_FULL_BASIS[12]]
            )
        d = emri.delta_prior
        inj = np.asarray(emri.injection, dtype=float)
        if not emri.logm1_lims:
            emri.logm1_lims = [(1 - d) * inj[0], (1 + d) * inj[0]]
        if not emri.m2_lims:
            emri.m2_lims = [(1 - d) * inj[1], (1 + d) * inj[1]]
        if not emri.a_lims:
            emri.a_lims = [(1 - d) * inj[2], min(0.999, (1 + d) * inj[2])]
        if not emri.p0_lims:
            emri.p0_lims = [(1 - d) * inj[3], (1 + d) * inj[3]]
        if not emri.e0_lims:
            emri.e0_lims = [(1 - d) * inj[4], (1 + d) * inj[4]]
        if emri.inner_moves is None:
            emri.inner_moves = [(StretchMove(), 1.0)]
        return emri

    def _prepare_sobbh(self, sobbh: AllSourcesSOBBHSettings, general_setup: GeneralSetup):
        from eryn.moves import StretchMove

        force_backend = general_setup.force_backend
        t_start = float(self.general.t_start)
        if sobbh.initialize_kwargs is None:
            sobbh.initialize_kwargs = dict(
                T=general_setup.Tobs / YRSID_SI,
                dt=general_setup.dt,
                sobbh_waveform_args=("SOBBHWaveform",),
                sobbh_waveform_kwargs=dict(force_backend=force_backend),
                response_kwargs=dict(
                    t0=t_start,
                    order=sobbh.response_order,
                    tdi="2nd generation",
                    tdi_chan=general_setup.tdi_chan,
                    force_backend=force_backend,
                    remove_garbage="zero",
                ),
            )
        if sobbh.waveform_kwargs is None:
            sobbh.waveform_kwargs = dict()
        if sobbh.injection is None:
            sobbh.injection = sobbh_full_to_sampling(SOBBH_INJECTION_PARAMS_FULL_BASIS)
        if getattr(sobbh, "fill_values", None) is None:
            sobbh.fill_values = np.array([])
        d = sobbh.delta_prior
        inj = np.asarray(sobbh.injection, dtype=float)
        if not sobbh.logm1_lims:
            sobbh.logm1_lims = [(1 - d) * inj[0], (1 + d) * inj[0]]
        if not sobbh.logm2_lims:
            sobbh.logm2_lims = [(1 - d) * inj[1], (1 + d) * inj[1]]
        if not sobbh.s1_lims:
            sobbh.s1_lims = [max(-0.99, inj[2] - d), min(0.99, inj[2] + d)]
        if not sobbh.s2_lims:
            sobbh.s2_lims = [max(-0.99, inj[3] - d), min(0.99, inj[3] + d)]
        if not sobbh.f_low_lims:
            sobbh.f_low_lims = [(1 - d) * inj[6], (1 + d) * inj[6]]
        if sobbh.inner_moves is None:
            sobbh.inner_moves = [(StretchMove(), 1.0)]
        return sobbh


# ---------------------------------------------------------------------------
# Runtime move builders (legacy paths, ported from global_fit_settings.py)
# ---------------------------------------------------------------------------


def _build_mbh_move(curr, acs, priors, state):
    """MBH move using ``BBHWaveformFD`` + ``MBHSpecialMove`` (legacy path)."""
    from eryn.moves.tempering import make_ladder

    from bbhx.waveformbuild import BBHWaveformFD

    from ....moves import MBHSpecialMove

    general_info = curr.general_info
    mbh_info = curr.source_info["mbh"]
    nwalkers, ntemps = general_info.nwalkers, general_info.ntemps
    xp = np if general_info.gpus is None else __import__("cupy")

    wave_gen = BBHWaveformFD(**mbh_info.initialize_kwargs)

    # Pre-subtract injected starting waveforms from each cold-chain walker.
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
                freqs=xp.asarray(acs.f_arr),
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


def _build_residual_pe_move(branch, curr, acs, priors, state, wave_gen):
    """Shared EMRI/SOBBH ``ResidualAddOneRemoveOneMove`` construction.

    Per-walker pre-injection keeps peak RAM at a single template's worth.
    """
    from eryn.moves.tempering import make_ladder

    from ....moves import ResidualAddOneRemoveOneMove

    general_info = curr.general_info
    info = curr.source_info[branch]
    nwalkers, ntemps = general_info.nwalkers, general_info.ntemps

    if np.any(inds := state.branches_inds[branch][0]):
        for leaf in range(inds.shape[-1]):
            if not inds[0, leaf]:
                continue
            assert np.all(inds[:, leaf])
            inj_coords = state.branches_coords[branch][0, :, leaf]
            inj_coords_in = info.transform.both_transforms(inj_coords)
            for i in range(inj_coords.shape[0]):
                sig = wave_gen(*inj_coords_in[i], **info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()

    betas_all = np.tile(make_ladder(info.ndim, ntemps=ntemps), (info.nleaves_max, 1))
    state.sub_states[branch].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, info.nleaves_max, info.ndim)
    move = ResidualAddOneRemoveOneMove(
        branch,
        coords_shape,
        wave_gen,
        info.waveform_kwargs.copy(),
        info.waveform_kwargs.copy(),
        acs,
        info.num_prop_repeats,
        info.transform,
        priors,
        info.inner_moves,
        Tmax=np.inf,
        betas_all=betas_all,
    )
    move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    return move


def _legacy_wave_wrap(branch, curr, acs):
    """Cached legacy ResponseWrapper wave wrap for EMRI / SOBBH."""
    general_info = curr.general_info
    info = curr.source_info[branch]
    force_backend = general_info.force_backend
    t_start = float(info.initialize_kwargs["response_kwargs"]["t0"])
    order = int(info.initialize_kwargs["response_kwargs"]["order"])
    tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
    getter = get_emri_response_wrapper if branch == "emri" else get_sobbh_response_wrapper
    template_wave_gen = getter(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=t_start,
        tdi_config=tdi_config,
        tdi_chan=general_info.tdi_chan,
        role="template",
        order=order,
        force_backend=force_backend,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend=force_backend,
    )
    wrap_cls = EMRIWaveWrap if branch == "emri" else SOBBHWaveWrap
    return wrap_cls(
        template_wave_gen,
        td_settings,
        general_info.domain_settings,
        td_window=None,
        nchannels=acs.nchannels,
    )


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``all_sources``: per-branch pre-work + materialization."""
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

    # --- GB: chunked-het WDM likelihood + moves ---
    if "gb" in curr.source_info:
        gb_info = curr.source_info["gb"]
        if (
            isinstance(general_info.domain_settings, WDMSettings)
            and gb_info.gb_wdm_comp is None
        ):
            from gbgpu.gbcomps import GBWDMComputations

            _wdm = general_info.domain_settings
            gb_info.gb_wdm_comp = GBWDMComputations(
                _wdm,
                t_ref=gb_info.t0,
                Nt_sub=int(gb_info.nt_sub),
                n_pad=int(gb_info.n_pad),
                N_sparse=int(gb_info.n_sparse),
                N_cp_sig=int(gb_info.n_cp_sig),
                N_cp_orbit=int(gb_info.n_cp_orbit),
                orbits=general_info.gpu_orbits,
                tdi_config="2nd generation" if gb_info.use_tdi2 else "1st generation",
                force_backend=general_info.force_backend,
                tdi_type="XYZ",
            )
        gb_names = [
            n for n in requested if n.startswith("rj_") or n.startswith("gb_")
        ]
        if gb_names:
            include_search = any(n.endswith("_search") for n in gb_names)
            include_refit = any(n.startswith("rj_refit") for n in gb_names)
            pe_names = [n for n in gb_names if not n.endswith("_search")]
            gb_search_moves, gb_pe_moves = build_gb_moves(
                engine_info, curr, acs, priors, state,
                include_search=include_search,
                include_refit=include_refit,
                pe_move_names=pe_names or None,
            )
            stock_moves.update(
                {m.name: m for m in list(gb_search_moves) + list(gb_pe_moves)}
            )

    # --- PSD ---
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

    # --- MBH / EMRI / SOBBH (legacy waveform paths) ---
    if "mbh" in curr.source_info and "mbh_pe" in requested:
        stock_moves["mbh_pe"] = _build_mbh_move(curr, acs, priors, state)
    if "emri" in curr.source_info and "emri_pe" in requested:
        wave_gen = _legacy_wave_wrap("emri", curr, acs)
        stock_moves["emri_pe"] = _build_residual_pe_move(
            "emri", curr, acs, priors, state, wave_gen
        )
    if "sobbh" in curr.source_info and "sobbh_pe" in requested:
        wave_gen = _legacy_wave_wrap("sobbh", curr, acs)
        stock_moves["sobbh_pe"] = _build_residual_pe_move(
            "sobbh", curr, acs, priors, state, wave_gen
        )

    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state,
    )
    materialize_recipe(recipe, recipe_spec, ctx, stock_moves, ntemps, nwalkers)


AllSourcesGlobalFit.default_setup_function = staticmethod(setup_recipe)
