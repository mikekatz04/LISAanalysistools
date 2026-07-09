"""Global-fit settings: EMRI-only smoke test (CPU + WDM + self-generated injection).

Layout matches ``emri_test_script_td_wave.py``: the EMRI waveform is built
from ``few.waveform.GenerateEMRIWaveform`` + ``fastlisaresponse.ResponseWrapper``
directly (no ``EMRITDIonFly``), wrapped in a small ``EMRIWaveWrap`` that
forwards the call to :class:`TDSignal.transform` so the global-fit
machinery sees a domain-aware signal.

Smoke-test choices:

* CPU-only (no cupy / no GPU paths).
* WDM grid ``Nf=720, Nt=2160, dt=5`` so ``Tobs = Nf*Nt*dt = 90 d ≈ 3 mo``.
* ``nwalkers=4`` and ``ntemps=2`` for fast turnaround.
* EMRI injection generated in-process — no external h5 paths.
* Single cached ``ResponseWrapper`` reused between the synthetic-injection
  data loader and the template move so the slow ``GenerateEMRIWaveform``
  setup runs once.
"""

import gc
import logging
import shutil

import numpy as np

# CPU-only smoke test. Map ``cp`` to numpy so the existing call sites
# (``cp.cuda.runtime.setDevice`` etc.) can still be guarded by
# ``gpu_available``.
import numpy as cp

GPU_BACKEND = "cpu"
gpu_available = False

logger = logging.getLogger(__name__)

from eryn.moves import StretchMove
from eryn.moves.tempering import make_ladder

from lisatools.response.tdiconfig import TDIConfig

from lisatools.domains import (
    TDSettings,
    WDMSettings,
)
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
)
from lisatools.globalfit.recipe import EMRIMoveBuilder, PERecipeStep
from lisatools.globalfit.preprocessing import SyntheticSourceProcessingStep
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    EMRISettings,
    EMRISetup,
    make_emri_transform_container,
)
# EMRI response-wrapper + domain-projection adapter now live in the stock
# ``lisatools.sources.emri`` package (carved out of the settings files).
from lisatools.sources.emri import EMRIWaveWrap, get_emri_response_wrapper
from lisatools.utils.constants import YRSID_SI


# ============================================================
# *** WDM grid ***
# ============================================================
NF = 720
NT = 2160
DT = 5.0
TOBS = NF * NT * DT  # 7,776,000 s ≈ 90 d ≈ 3 mo
T_START = 0.0


DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=1e-4,
    max_freq=2.5e-2,
    min_time=20 * 3600.0,
    max_time=(NT - 20) * 3600.0,
)


# ============================================================
# *** Self-generated EMRI injection ***
# ============================================================
# ``few``'s ``FastKerrEccentricEquatorialFlux`` consumes 14 parameters in
# this order: M, mu, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0,
# Phi_theta0, Phi_r0. EMRISetup's transform fills xI0 (idx 5) and
# Phi_theta0 (idx 12) from ``fill_values`` so the sampling basis has 12
# parameters (with ``M -> logM``, ``qS -> cos qS``, ``qK -> cos qK``).
INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        1.0e6,      # M
        1.0e1,      # mu
        0.5,        # a
        10.0,       # p0
        0.3,        # e0
        1.0,        # xI0 — fill_value[0] (prograde equatorial)
        1.0,        # dist (Gpc)
        np.pi / 3,  # qS
        1.0,        # phiS
        np.pi / 4,  # qK
        2.0,        # phiK
        0.0,        # Phi_phi0
        0.0,        # Phi_theta0 — fill_value[1]
        0.0,        # Phi_r0
    ]
)
SAMPLE_FILL_INDICES = [5, 12]


def emri_full_to_sampling(params_full):
    """Convert a 14-param waveform-basis vector to the 12-param sampling basis."""
    p = np.asarray(params_full, dtype=float)
    transform = make_emri_transform_container(p[SAMPLE_FILL_INDICES])
    return transform.both_inverse_transforms(p)


# ---- EMRI response wrapper + domain adapter: imported from the stock
#      ``lisatools.sources.emri`` package (was inlined + duplicated here).
#      ``special_frame=False`` reproduces this smoke test's raw-wrapper path
#      (sampler basis stays [qS, phiS, qK, phiK]; convert_to_ra_dec=False). ----


def _make_emri_injection_wave_gen(*, Tobs, dt, t_start, tdi_chan):
    """Module-level factory so the (unpicklable) wrapper is built lazily inside
    the stock :class:`SyntheticSourceProcessingStep` (its ``processor_init_kwargs``
    are deep-copied, so only this picklable function reference is stored)."""
    return get_emri_response_wrapper(
        Tobs=Tobs,
        dt=dt,
        t_start=t_start,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan=tdi_chan,
        role="injection",
        special_frame=False,
    )


# Cached domain-wrapped template generator, shared between the engine-side
# ``signal_gen`` (residual rebuild) and the PE move so the (slow) response build
# happens once. ``get_emri_response_wrapper`` caches the underlying
# ``ResponseWrapper`` keyed by (Tobs, dt, t_start, ..., special_frame) with
# ``role`` excluded, so this template wrapper shares the SAME orbit + LTT object
# as the ``role="injection"`` data path — the residual therefore cancels at the
# true injection point. Orbits default to ``EqualArmlengthOrbits`` (constant,
# non-sampled light travel times).
_WAVE_WRAP_CACHE = {}


def _get_emri_wave_wrap(general_info, nchannels: int = 3):
    """Build (and cache) the EMRI domain-wrapped template generator.

    Reproduces exactly the wrapper the recipe used to build inline, so the
    engine's ``setup_acs(rebuild_residuals=True)`` subtracts the identical
    template the injection added.
    """
    key = ("emri", id(general_info), nchannels)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    template_wave_gen = get_emri_response_wrapper(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=T_START,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan="XYZ",
        role="template",
        special_frame=False,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend="cpu",
    )
    wrap = EMRIWaveWrap(
        template_wave_gen,
        td_settings,
        general_info.domain_settings,
        td_window=None,
        # Smoke path: keep the sampler basis [qS, phiS, qK, phiK]; the stock
        # EMRIWaveWrap does not force convert_to_ra_dec (that is the SPECIAL
        # frame's job), so pass it explicitly here.
        runtime_kwargs={"convert_to_ra_dec": False},
        nchannels=nchannels,
    )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


################
#  RECIPE STEPS
################


################
#  RECIPE
################


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    print("[setup_recipe] entered", flush=True)
    emri_info = curr.source_info["emri"]
    general_info = curr.general_info
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    domain_settings = general_info.domain_settings
    print(
        f"[setup_recipe] nwalkers={nwalkers} ntemps={ntemps} "
        f"domain={type(domain_settings).__name__} nchannels={acs.nchannels}",
        flush=True,
    )

    # Template generator, shared with the engine-side ``signal_gen`` through the
    # ``_WAVE_WRAP_CACHE`` (same cached ResponseWrapper as the injection).
    wave_gen = _get_emri_wave_wrap(general_info, nchannels=acs.nchannels)

    # No recipe-side residual pre-injection: the engine's
    # ``setup_acs(rebuild_residuals=True)`` already subtracted the state's EMRI
    # templates via the registered ``signal_gen`` (see get_emri_erebor_settings).
    # Doing it here as well would double-subtract.
    print("[setup_recipe] building EMRI PE move", flush=True)

    # Stock single-source PE-move builder (make_ladder -> betas_all ->
    # ResidualAddOneRemoveOneMove); the machinery lives in
    # ``lisatools.globalfit.recipe``.
    _, emri_pe_moves = EMRIMoveBuilder(wave_gen=wave_gen).build(
        engine_info, curr, acs, priors, state
    )
    recipe.add_recipe_component(
        PERecipeStep(moves=emri_pe_moves), name="emri pe"
    )


##########################
#  SETTINGS
##########################


def get_emri_erebor_settings(general_set: GeneralSetup) -> EMRISetup:
    """Build the EMRI :class:`EMRISetup` for the CPU smoke test."""
    # ``initialize_kwargs`` is consumed by ``EMRISetup`` only to track
    # metadata for the run; we don't construct another generator from it
    # (the cached one in ``setup_recipe`` is the active path).
    initialize_kwargs_emri = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs=dict(force_backend="cpu"),
        response_kwargs=dict(
            t0=T_START,
            order=40,
            tdi="2nd generation",
            tdi_chan="XYZ",
            force_backend="cpu",
            remove_garbage="zero",
        ),
    )

    waveform_kwargs_pe = dict()

    delta_prior = 1e-2
    injection_sampling = emri_full_to_sampling(INJECTION_PARAMS_FULL_BASIS)

    logm1_lims = [
        (1 - delta_prior) * injection_sampling[0],
        (1 + delta_prior) * injection_sampling[0],
    ]
    m2_lims = [
        (1 - delta_prior) * injection_sampling[1],
        (1 + delta_prior) * injection_sampling[1],
    ]
    amax = min(0.999, (1 + delta_prior) * injection_sampling[2])
    a_lims = [(1 - delta_prior) * injection_sampling[2], amax]
    p0_lims = [
        (1 - delta_prior) * injection_sampling[3],
        (1 + delta_prior) * injection_sampling[3],
    ]
    e0_lims = [
        (1 - delta_prior) * injection_sampling[4],
        (1 + delta_prior) * injection_sampling[4],
    ]

    inner_moves = [(StretchMove(), 1.0)]
    fill_values = np.array(
        [
            INJECTION_PARAMS_FULL_BASIS[5],   # xI0
            INJECTION_PARAMS_FULL_BASIS[12],  # Phi_theta0
        ]
    )

    emri_settings = EMRISettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        fill_values=fill_values,
        logm1_lims=logm1_lims,
        m2_lims=m2_lims,
        a_lims=a_lims,
        p0_lims=p0_lims,
        e0_lims=e0_lims,
        injection=injection_sampling,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_emri,
        waveform_kwargs=waveform_kwargs_pe,
        info_matrix_gen=None,
        inner_moves=inner_moves,
        nleaves_max=1,
        nleaves_min=1,
        ndim=12,
    )

    emri_setup = EMRISetup(emri_settings)

    # Engine-side template generation: one sampling-basis row in -> one
    # run-domain template out. ``setup_acs(rebuild_residuals=True)`` calls this
    # to subtract the state's EMRI templates from the residuals (replacing the
    # old recipe-side pre-injection loop). The wave-wrap build is deferred to
    # first call (cached) so settings construction stays cheap.
    def _emri_signal_gen(*params, **kwargs):
        wave_wrap = _get_emri_wave_wrap(general_set)
        params_in = emri_setup.transform.both_transforms(
            np.asarray(params, dtype=float)
        )
        return wave_wrap(*params_in, **kwargs)

    emri_setup.signal_gen = _emri_signal_gen
    return emri_setup


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    base_file_name = "emri_only_smoke_test"
    file_store_dir = "./gf_output/"

    nwalkers = 4
    ntemps = 2

    domain_settings = DOMAIN_CHOICE

    # The stock ``SyntheticSourceProcessingStep`` builds the (cached) injection
    # response wrapper via the module-level factory and injects it onto the data
    # grid. The template path in ``setup_recipe`` re-requests the wrapper and
    # hits the same cache entry.
    processor_init_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        t_start=T_START,
        wave_gen_factory=_make_emri_injection_wave_gen,
        injections=INJECTION_PARAMS_FULL_BASIS,
        call_kwargs={"convert_to_ra_dec": False},
        tdi_chan="XYZ",
        nchannels=3,
    )

    sensitivity_init_kwargs = dict(
        tdi_generation=2, mask_percentage=0.02, use_splines=False
    )

    # Smoke test: no Tukey taper.
    window_taper_duration = 0.0

    # Skip the engine's default highpass + 200 h edge-trim + Tobs trim.
    # The synthesised signal already covers exactly ``Tobs = Nf*Nt*dt``;
    # the default ``Tobs`` trim uses ``T=(N-1)*dt`` which would lose one
    # sample and break the WDM ``Nf*Nt`` shape.
    preprocess_kwargs = dict(
        highpass_kwargs=None,
        trim_kwargs=None,
        Tobs=None,
    )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        main_file_key="testing",
        domain_settings=domain_settings,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type="tukey",
        window_taper_duration=window_taper_duration,
        gpu_backend=GPU_BACKEND,
        gpus=None,
        data_processor_class=SyntheticSourceProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
    )

    return GeneralSetup(general_settings)


def get_global_fit_settings(copy_settings_file=False):
    general_setup = get_general_erebor_settings()
    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir
            + general_setup.base_file_name
            + "_"
            + __file__.split("/")[-1],
        )

    rank_info = RankInfo(head_rank=1, main_rank=0)

    emri_setup = get_emri_erebor_settings(general_setup)

    global_settings = GlobalFitSettings(
        source_info={"emri": emri_setup},
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    return CurrentInfoGlobalFit(global_settings)


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print("EMRI smoke-test settings constructed OK")
