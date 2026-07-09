"""Global-fit settings: MBH-only smoke test (CPU + WDM + self-generated
injection) wired around the ``phentax`` IMRPhenomTHM JAX waveform model.

Structurally a 1:1 mirror of ``emri_only_global_fit_settings.py``:

* ``IMRPhenomTHMWaveform`` plays the role of the EMRI waveform-with-
  ``ResponseWrapper``-compatible-``__call__`` adapter (analogous to
  :class:`lisatools.sources.sobbh.SOBBHWaveform`).
* ``get_mbh_phentax_response_wrapper`` caches one
  :class:`ResponseWrapper` so the synthetic-injection data loader and
  the template move share a single waveform instance.
* ``MBHWaveWrap`` projects the TD output to the run's domain
  (FD / STFT / WDM).
* ``SyntheticMBHProcessingStep`` generates the injection in-process.
* The recipe wires a :class:`ResidualAddOneRemoveOneMove` for MBH PE.

Smoke-test choices:

* CPU-only (no cupy / no GPU paths).
* WDM grid ``Nf=720, Nt=2160, dt=5`` so ``Tobs = Nf*Nt*dt = 90 d ≈ 3 mo``.
* ``nwalkers=4`` and ``ntemps=2`` for fast turnaround.
* MBH injection generated in-process — no external h5 paths.
* Single cached ``ResponseWrapper`` reused between the synthetic-
  injection data loader and the template move so the (slow) phentax
  generator setup runs once.
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
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.response.tdiconfig import TDIConfig

from lisatools.domains import TDSettings, WDMSettings
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
)
from lisatools.globalfit.preprocessing import SyntheticSourceProcessingStep
from lisatools.globalfit.recipe import MBHMoveBuilder, PERecipeStep
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import MBHSettings, MBHSetup
# MBH phentax response-wrapper + domain adapter + transform live in BBHx (the
# sprint's MBH-physics owner, home of MBHTDIonFly + the phentax extra).
from bbhx.mbhphentax import (
    MBHWaveWrap,
    get_mbh_phentax_response_wrapper,
    make_mbh_phentax_transform_container,
)
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
# *** Self-generated MBH injection ***
# ============================================================
# phentax's :class:`IMRPhenomTHM` consumes 8 intrinsic + 3 extrinsic
# parameters in the order used below by :class:`IMRPhenomTHMWaveform`.
# We choose an 11-parameter waveform basis to match the existing
# ``MBHSetup`` sampling philosophy (intrinsic masses + spins + sky
# direction + polarization + merger time within the observation):
#
#   waveform basis (call-signature order — what the waveform sees):
#     0 m1       (M_sun)
#     1 m2       (M_sun)
#     2 s1z      (dimensionless aligned spin)
#     3 s2z      (dimensionless aligned spin)
#     4 dist     (Gpc — converted to Mpc inside)
#     5 phi_ref  (rad)
#     6 inc      (rad)
#     7 psi      (rad)
#     8 lam      (ecliptic longitude, rad) — consumed by ResponseWrapper
#     9 beta     (ecliptic latitude, rad)  — consumed by ResponseWrapper
#    10 t_plunge (s, merger time in the observation window)
#
INJECTION_PARAMS_FULL_BASIS = np.array(
    [
        1.0e6,          # m1 (M_sun)
        5.0e5,          # m2 (M_sun)
        0.5,            # s1z
        0.3,            # s2z
        10.0,           # dist (Gpc)
        1.0,            # phi_ref
        np.pi / 3.0,    # inclination
        0.5,            # psi
        1.5,            # lam (ecliptic longitude)
        0.3,            # beta (ecliptic latitude)
        0.5 * TOBS,     # t_plunge — merger placed mid-observation
    ]
)


def mbh_full_to_sampling(params_full: np.ndarray) -> np.ndarray:
    """Convert an 11-param waveform-basis vector to the 11-param sampling basis.

    Sampling basis re-parameterises the intrinsic + extrinsic ranges to be
    flat-prior friendly:

      ``logM = log(m1 + m2)`` (total mass)
      ``q = m2 / m1`` (mass ratio, 0 < q < 1 when m1 >= m2 — we don't
        enforce ordering here; pass the same convention as the prior
        bounds use)
      ``cos_iota = cos(inclination)``
      ``sin_beta = sin(beta)``
      All other params pass through unchanged.
    """
    transform = make_mbh_phentax_transform_container()
    return transform.both_inverse_transforms(np.asarray(params_full, dtype=float))


# ---- MBH phentax response wrapper + domain adapter: imported from the stock
#      ``lisatools.sources.bbh`` package (was inlined + duplicated here). ----


def _make_mbh_injection_wave_gen(*, Tobs, dt, t_start, tdi_chan):
    """Module-level factory so the (unpicklable) wrapper is built lazily inside
    the stock :class:`SyntheticSourceProcessingStep` (its ``processor_init_kwargs``
    are deep-copied, so only this picklable function reference is stored)."""
    return get_mbh_phentax_response_wrapper(
        Tobs=Tobs,
        dt=dt,
        t_start=t_start,
        tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
        tdi_chan=tdi_chan,
        role="injection",
        higher_modes="all",
    )


# Cached domain-wrapped template generator, shared between the engine-side
# ``signal_gen`` (residual rebuild) and the PE move. ``get_mbh_phentax_response_wrapper``
# caches the underlying ``ResponseWrapper`` with ``role`` excluded from the key,
# so this template shares the SAME orbit + LTT object as the ``role="injection"``
# data path. Orbits default to ``EqualArmlengthOrbits`` (constant, non-sampled
# light travel times). This reproduces the wrapper the recipe built inline.
_WAVE_WRAP_CACHE = {}


def _get_mbh_wave_wrap(general_info, nchannels: int = 3):
    """Build (and cache) the MBH-phentax domain-wrapped template generator.

    Reproduces exactly the wrapper the recipe used to build inline, so the
    engine's ``setup_acs(rebuild_residuals=True)`` subtracts the identical
    template the recipe pre-injection used to.
    """
    key = ("mbh", id(general_info), nchannels)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    template_wave_gen = get_mbh_phentax_response_wrapper(
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
    wrap = MBHWaveWrap(
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
    mbh_info = curr.source_info["mbh"]
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
    wave_gen = _get_mbh_wave_wrap(general_info, nchannels=acs.nchannels)

    # No recipe-side residual pre-injection: the engine's
    # ``setup_acs(rebuild_residuals=True)`` already subtracted the state's MBH
    # templates via the registered ``signal_gen`` (see
    # get_mbh_phentax_erebor_settings). Doing it here as well would double-subtract.
    print("[setup_recipe] building MBH PE move", flush=True)

    # Stock single-source PE-move builder. This phentax path passes
    # ``waveform_kwargs`` as the likelihood kwargs (matching the inline move it
    # replaces); ``MBHMoveBuilder`` defaults the like-kwargs to ``{}`` (the
    # phenom ``build_mbh_moves_phenom`` convention), so pass it explicitly here.
    _, mbh_pe_moves = MBHMoveBuilder(
        wave_gen=wave_gen, waveform_like_kwargs=mbh_info.waveform_kwargs
    ).build(engine_info, curr, acs, priors, state)
    recipe.add_recipe_component(
        PERecipeStep(moves=mbh_pe_moves), name="mbh pe"
    )


##########################
#  SETTINGS
##########################


def get_mbh_phentax_erebor_settings(general_set: GeneralSetup) -> MBHSetup:
    """Build the MBH :class:`MBHSetup` for the CPU phentax smoke test."""
    # ``initialize_kwargs`` is consumed by ``MBHSetup`` only to track
    # metadata for the run; the active waveform path is the cached
    # generator in ``setup_recipe``.
    initialize_kwargs_mbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        mbh_waveform="phentax.IMRPhenomTHM",
        mbh_waveform_kwargs=dict(higher_modes="all", force_backend="cpu"),
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
    # TODO: adjust this to regular transform container indexing
    injection_sampling = mbh_full_to_sampling(INJECTION_PARAMS_FULL_BASIS)

    # Tight box around the injection in the sampling basis. Angles
    # (phi_ref, psi, lam) use additive deltas to avoid issues around
    # 0; cos_iota / sin_beta use absolute deltas clamped to [-1, 1].
    logM_inj = injection_sampling[0]
    q_inj = injection_sampling[1]
    s1z_inj = injection_sampling[2]
    s2z_inj = injection_sampling[3]
    dist_inj = injection_sampling[4]
    phi_ref_inj = injection_sampling[5]
    cos_iota_inj = injection_sampling[6]
    psi_inj = injection_sampling[7]
    lam_inj = injection_sampling[8]
    sin_beta_inj = injection_sampling[9]
    t_plunge_inj = injection_sampling[10]

    priors_mbh = {
        "logM":     uniform_dist((1 - delta_prior) * logM_inj,
                                 (1 + delta_prior) * logM_inj),
        "q":        uniform_dist(max(0.01, (1 - delta_prior) * q_inj),
                                 min(0.999, (1 + delta_prior) * q_inj)),
        "s1z":      uniform_dist(max(-0.99, s1z_inj - delta_prior),
                                 min(0.99, s1z_inj + delta_prior)),
        "s2z":      uniform_dist(max(-0.99, s2z_inj - delta_prior),
                                 min(0.99, s2z_inj + delta_prior)),
        "dist":     uniform_dist((1 - delta_prior) * dist_inj,
                                 (1 + delta_prior) * dist_inj),
        "phi_ref":  uniform_dist(phi_ref_inj - delta_prior * np.pi,
                                 phi_ref_inj + delta_prior * np.pi),
        "cos_iota": uniform_dist(max(-1.0 + 1e-6, cos_iota_inj - delta_prior),
                                 min(1.0 - 1e-6, cos_iota_inj + delta_prior)),
        "psi":      uniform_dist(psi_inj - delta_prior * np.pi,
                                 psi_inj + delta_prior * np.pi),
        "lam":      uniform_dist(lam_inj - delta_prior * np.pi,
                                 lam_inj + delta_prior * np.pi),
        "sin_beta": uniform_dist(max(-1.0 + 1e-6, sin_beta_inj - delta_prior),
                                 min(1.0 - 1e-6, sin_beta_inj + delta_prior)),
        "t_plunge": uniform_dist(t_plunge_inj - 60.0,    # +/- 1 min around injection
                                 t_plunge_inj + 60.0),
    }
    priors = {"mbh": ProbDistContainer(priors_mbh)}

    inner_moves = [(StretchMove(), 1.0)]

    mbh_transform = make_mbh_phentax_transform_container()

    # Engine-side template generation: one sampling-basis row in -> one
    # run-domain template out. ``setup_acs(rebuild_residuals=True)`` calls this
    # to subtract the state's MBH templates from the residuals (replacing the
    # old recipe-side pre-injection loop). Build deferred to first call (cached).
    def _mbh_signal_gen(*params, **kwargs):
        wave_wrap = _get_mbh_wave_wrap(general_set)
        params_in = mbh_transform.both_transforms(np.asarray(params, dtype=float))
        return wave_wrap(*params_in, **kwargs)

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        injection=injection_sampling,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_mbh,
        waveform_kwargs=waveform_kwargs_pe,
        inner_moves=inner_moves,
        nleaves_max=1,
        nleaves_min=1,
        ndim=11,
        transform=mbh_transform,
        priors=priors,
        # Angles where periodicity matters in the sampling basis:
        periodic={"mbh": {"phi_ref": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}},
        log_dir=general_set.file_store_dir,
        signal_gen=_mbh_signal_gen,
    )

    return MBHSetup(mbh_settings)


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    base_file_name = "mbh_phentax_only_smoke_test"
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
        wave_gen_factory=_make_mbh_injection_wave_gen,
        injections=INJECTION_PARAMS_FULL_BASIS,
        call_kwargs={"convert_to_ra_dec": False},
        tdi_chan="XYZ",
        nchannels=3,
    )

    # CompositeSensitivityBackend is the default; only ``tdi_generation``
    # is consumed.
    sensitivity_init_kwargs = dict(tdi_generation=2)

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

    mbh_setup = get_mbh_phentax_erebor_settings(general_setup)

    global_settings = GlobalFitSettings(
        source_info={"mbh": mbh_setup},
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    return CurrentInfoGlobalFit(global_settings)


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print("MBH (phentax) smoke-test settings constructed OK")
