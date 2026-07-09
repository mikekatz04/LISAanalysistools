"""Global-fit settings: SOBBH-only smoke test (CPU + WDM + self-generated injection).

Direct mirror of ``emri_only_global_fit_settings.py``. The SOBBH waveform
is built from :class:`lisatools.sources.sobbh.SOBBHWaveform` (TaylorT3,
3.5PN, aligned spins) and projected onto LISA via
``fastlisaresponse.ResponseWrapper`` — same wrapping pattern as the EMRI
smoke test. ``SOBBHWaveWrap`` then forwards the call to
:class:`TDSignal.transform` so the global-fit machinery sees a
domain-aware signal.

Smoke-test choices:

* CPU-only (no cupy / no GPU paths).
* WDM grid ``Nf=720, Nt=2160, dt=5`` so ``Tobs = Nf*Nt*dt = 90 d ≈ 3 mo``.
* ``nwalkers=4`` and ``ntemps=2`` for fast turnaround.
* SOBBH injection generated in-process — no external h5 paths.
* Single cached :class:`SOBBHWaveform` reused between the injection
  data loader and the template move.
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
from lisatools.globalfit.preprocessing import SyntheticSourceProcessingStep
from lisatools.globalfit.recipe import PERecipeStep, SOBBHMoveBuilder
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    SOBBHSettings,
    SOBBHSetup,
    make_sobbh_transform_container,
)
# SOBBH response-wrapper + domain-projection adapter now live in the stock
# ``lisatools.sources.sobbh`` package (carved out of the settings files).
from lisatools.sources.sobbh import SOBBHWaveWrap, get_sobbh_response_wrapper
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
# *** Self-generated SOBBH injection ***
# ============================================================
# Output-basis order consumed by :class:`SOBBHWaveform.__call__`:
#   0  m1    (M_sun)
#   1  m2    (M_sun)
#   2  s1    (dimensionless aligned spin)
#   3  s2    (dimensionless aligned spin)
#   4  dist  (Gpc)
#   5  inc   (rad)
#   6  f_low (Hz)
#   7  lam   (ecliptic longitude, rad)
#   8  beta  (ecliptic latitude, rad)
#   9  psi   (polarization, rad)
#  10  phi0  (coalescence-phase offset, rad)
#
# Sampling basis (consumed by ``SOBBHSetup.transform``) is 11 params:
#   logm1, logm2, s1, s2, dist, cosinc, f_low, phiS, cosqS, psi, phi0
# with ``cosqS -> beta = pi/2 - arccos(cosqS)`` and ``phiS -> lam``.
INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        40.0,           # m1 (M_sun)
        30.0,           # m2 (M_sun)
        0.3,            # s1
        -0.2,           # s2
        1.0,            # dist (Gpc)
        np.pi / 3,      # inc
        1.5e-2,         # f_low (Hz) — well above LISA's low-freq wall
        1.0,            # lam (phiS in sampling basis)
        np.pi / 4,      # beta  (qS = pi/2 - beta in sampling basis)
        0.3,            # psi
        0.0,            # phi0
    ]
)


def sobbh_full_to_sampling(params_full):
    """Convert an 11-param waveform-basis vector to the 11-param sampling basis.

    The waveform basis is ``(m1, m2, s1, s2, dist, inc, f_low, lam, beta, psi, phi0)``;
    the sampling basis is ``(logm1, logm2, s1, s2, dist, cosinc, f_low, phiS, cosqS, psi, phi0)``.
    """
    transform = make_sobbh_transform_container()
    return transform.both_inverse_transforms(np.asarray(params_full, dtype=float))


# ---- SOBBH response wrapper + domain adapter: imported from the stock
#      ``lisatools.sources.sobbh`` package (was inlined + duplicated here). ----


def _make_sobbh_injection_wave_gen(*, Tobs, dt, t_start, tdi_chan):
    """Module-level factory so the (unpicklable) wrapper is built lazily inside
    the stock :class:`SyntheticSourceProcessingStep` (its ``processor_init_kwargs``
    are deep-copied, so only this picklable function reference is stored)."""
    return get_sobbh_response_wrapper(
        Tobs=Tobs,
        dt=dt,
        t_start=t_start,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan=tdi_chan,
        role="injection",
    )


# Cached domain-wrapped template generator, shared between the engine-side
# ``signal_gen`` (residual rebuild) and the PE move. ``get_sobbh_response_wrapper``
# caches the underlying ``ResponseWrapper`` with ``role`` excluded from the key,
# so this template shares the SAME orbit + LTT object as the ``role="injection"``
# data path — the residual cancels at the true injection point. Orbits default to
# ``EqualArmlengthOrbits`` (constant, non-sampled light travel times).
_WAVE_WRAP_CACHE = {}


def _get_sobbh_wave_wrap(general_info, nchannels: int = 3):
    """Build (and cache) the SOBBH domain-wrapped template generator.

    Reproduces exactly the wrapper the recipe used to build inline, so the
    engine's ``setup_acs(rebuild_residuals=True)`` subtracts the identical
    template the injection added.
    """
    key = ("sobbh", id(general_info), nchannels)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    template_wave_gen = get_sobbh_response_wrapper(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=T_START,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan="XYZ",
        role="template",
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend="cpu",
    )
    wrap = SOBBHWaveWrap(
        template_wave_gen,
        td_settings,
        general_info.domain_settings,
        td_window=None,
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
    sobbh_info = curr.source_info["sobbh"]
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
    wave_gen = _get_sobbh_wave_wrap(general_info, nchannels=acs.nchannels)

    # No recipe-side residual pre-injection: the engine's
    # ``setup_acs(rebuild_residuals=True)`` already subtracted the state's SOBBH
    # templates via the registered ``signal_gen`` (see get_sobbh_erebor_settings).
    # Doing it here as well would double-subtract.
    print("[setup_recipe] building SOBBH PE move", flush=True)

    # Stock single-source PE-move builder (make_ladder -> betas_all ->
    # ResidualAddOneRemoveOneMove); the machinery lives in
    # ``lisatools.globalfit.recipe``.
    _, sobbh_pe_moves = SOBBHMoveBuilder(wave_gen=wave_gen).build(
        engine_info, curr, acs, priors, state
    )
    recipe.add_recipe_component(
        PERecipeStep(moves=sobbh_pe_moves), name="sobbh pe"
    )


##########################
#  SETTINGS
##########################


def get_sobbh_erebor_settings(general_set: GeneralSetup) -> SOBBHSetup:
    """Build the SOBBH :class:`SOBBHSetup` for the CPU smoke test."""
    # ``initialize_kwargs`` is metadata only; the active waveform path is
    # the cached generator in :func:`setup_recipe`.
    initialize_kwargs_sobbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        sobbh_waveform_args=("SOBBHWaveform",),
        sobbh_waveform_kwargs=dict(force_backend="cpu"),
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
    injection_sampling = sobbh_full_to_sampling(INJECTION_PARAMS_FULL_BASIS)

    logm1_lims = [
        (1 - delta_prior) * injection_sampling[0],
        (1 + delta_prior) * injection_sampling[0],
    ]
    logm2_lims = [
        (1 - delta_prior) * injection_sampling[1],
        (1 + delta_prior) * injection_sampling[1],
    ]
    s1_inj = injection_sampling[2]
    s1_lims = [max(-0.99, s1_inj - delta_prior), min(0.99, s1_inj + delta_prior)]
    s2_inj = injection_sampling[3]
    s2_lims = [max(-0.99, s2_inj - delta_prior), min(0.99, s2_inj + delta_prior)]
    f_low_inj = injection_sampling[6]
    f_low_lims = [
        (1 - delta_prior) * f_low_inj,
        (1 + delta_prior) * f_low_inj,
    ]

    inner_moves = [(StretchMove(), 1.0)]
    # SOBBH has no non-sampled params at default — keep the field empty.
    fill_values = np.array([])

    sobbh_settings = SOBBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        fill_values=fill_values,
        logm1_lims=logm1_lims,
        logm2_lims=logm2_lims,
        s1_lims=s1_lims,
        s2_lims=s2_lims,
        f_low_lims=f_low_lims,
        injection=injection_sampling,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_sobbh,
        waveform_kwargs=waveform_kwargs_pe,
        info_matrix_gen=None,
        inner_moves=inner_moves,
        nleaves_max=1,
        nleaves_min=1,
        ndim=11,
    )

    sobbh_setup = SOBBHSetup(sobbh_settings)

    # Engine-side template generation: one sampling-basis row in -> one
    # run-domain template out. ``setup_acs(rebuild_residuals=True)`` calls this
    # to subtract the state's SOBBH templates from the residuals (replacing the
    # old recipe-side pre-injection loop). Build deferred to first call (cached).
    def _sobbh_signal_gen(*params, **kwargs):
        wave_wrap = _get_sobbh_wave_wrap(general_set)
        params_in = sobbh_setup.transform.both_transforms(
            np.asarray(params, dtype=float)
        )
        return wave_wrap(*params_in, **kwargs)

    sobbh_setup.signal_gen = _sobbh_signal_gen
    return sobbh_setup


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    base_file_name = "sobbh_only_smoke_test"
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
        wave_gen_factory=_make_sobbh_injection_wave_gen,
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

    sobbh_setup = get_sobbh_erebor_settings(general_setup)

    global_settings = GlobalFitSettings(
        source_info={"sobbh": sobbh_setup},
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    return CurrentInfoGlobalFit(global_settings)


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print("SOBBH smoke-test settings constructed OK")
