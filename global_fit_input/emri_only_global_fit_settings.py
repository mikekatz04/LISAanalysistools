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
from typing import Optional

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

from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from few.waveform import GenerateEMRIWaveform

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import (
    DomainBaseArray,
    FDSettings,
    TDSettings,
    TDSignal,
    WDMSettings,
)
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
)
from lisatools.globalfit.moves import ResidualAddOneRemoveOneMove
from lisatools.globalfit.preprocessing import BaseProcessingStep
from lisatools.globalfit.recipe import RecipeStep
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    EMRISettings,
    EMRISetup,
    make_emri_transform_container,
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


# ---- Cached EMRI generator + wrappers (shared across data path + moves) ----

# Shared inspiral / sum / mode-selector kwargs (mirrors emri_test_script_td_wave.py).
INSPIRAL_KWARGS = {
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e4),
    "force_backend": "cpu",
}
SUM_KWARGS = {"pad_output": True}
# Single mode-selection threshold for both injection and template paths.
# 1e-2 keeps the smoke test fast; revisit once GPU is available.
MODE_SELECTOR_KWARGS = {"mode_selection_threshold": 1e-2}

_WAVE_GEN_CACHE = {}


def get_emri_response_wrapper(
    *,
    Tobs: float,
    dt: float,
    t_start: float,
    tdi_config: TDIConfig,
    tdi_chan: str = "XYZ",
    role: str = "template",
    order: int = 40,
    t_buffer: float = 3e4,
    force_backend: str = "cpu",
):
    """Build (and cache) a :class:`ResponseWrapper` around ``GenerateEMRIWaveform``.

    Note: this smoke test uses *one* generator for both the synthetic
    injection and the template path. Building two ``GenerateEMRIWaveform``
    instances (as the original test script does for tight vs loose mode
    selection) crashes the CPU process here, so we just share the looser
    template threshold and pay the small accuracy cost — the smoke test
    only exercises pipeline plumbing, not source recovery.
    """
    # ``role`` is intentionally ignored in the cache key so injection and
    # template both reuse the first generator built.
    key = (Tobs, dt, t_start, tdi_chan, order, force_backend)
    if key in _WAVE_GEN_CACHE:
        return _WAVE_GEN_CACHE[key]

    few_generator = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        return_list=False,
        inspiral_kwargs=INSPIRAL_KWARGS,
        sum_kwargs=SUM_KWARGS,
        frame="detector",
        mode_selector_kwargs=MODE_SELECTOR_KWARGS,
        force_backend=force_backend,
    )

    response_kwargs = {
        "Tobs": Tobs / YRSID_SI,
        "dt": dt,
        "index_lambda": 8,
        "index_beta": 7,
        "flip_hx": True,
        "force_backend": force_backend,
        "tdi": tdi_config,
        "tdi_chan": tdi_chan,
        "order": order,
        "remove_garbage": "zero",
        "is_ecliptic_latitude": False,
        "t_buffer": t_buffer,
    }

    orbits = EqualArmlengthOrbits(force_backend=force_backend)
    wave_gen = ResponseWrapper(
        few_generator,
        orbits=orbits,
        t0=t_start,
        **response_kwargs,
    )
    _WAVE_GEN_CACHE[key] = wave_gen
    return wave_gen


class EMRIWaveWrap:
    """Adapter that runs the cached ResponseWrapper and projects to the run's domain.

    Mirrors ``EMRIWaveWrap`` in ``emri_test_script_td_wave.py``: the call
    output is a :class:`DomainBase` subclass (FDSignal / WDMSignal / ...)
    so the global-fit move and ACA dispatch land on the right kernels.
    """

    def __init__(
        self,
        wave_gen,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
    ):
        self.wave_gen = wave_gen
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels

    def __call__(self, *params, **kwargs):
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        # ``convert_to_ra_dec=False`` keeps the sampler basis aligned with
        # ``[qS, phiS, qK, phiK]`` rather than (ra, dec).
        call_kwargs.setdefault("convert_to_ra_dec", False)
        arr = np.asarray(self.wave_gen(*params, **call_kwargs))
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )


class SyntheticEMRIProcessingStep(BaseProcessingStep):
    """Generate the EMRI injection in-process via the shared ResponseWrapper.

    Hands ``(times, data, fs)`` to :class:`BaseProcessingStep` — same
    interface as :class:`SangriaProcessingStep`. No external h5 path
    needed.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        injection_params_full_basis: np.ndarray,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        tdi_config = TDIConfig("2nd generation", force_backend="cpu")
        wave_gen = get_emri_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
        )

        td_signal = np.asarray(
            wave_gen(*injection_params_full_basis, convert_to_ra_dec=False)
        )
        td_signal = np.atleast_2d(td_signal)[:nchannels]

        # ``ResponseWrapper`` can produce a couple fewer samples than
        # ``Tobs/dt`` (Lagrange buffer / remove_garbage). Pad/clip to
        # exactly ``Tobs/dt`` so downstream WDM ``N = Nf*Nt`` matches.
        target_N = int(round(Tobs / dt))
        if td_signal.shape[-1] < target_N:
            pad = target_N - td_signal.shape[-1]
            td_signal = np.pad(td_signal, ((0, 0), (0, pad)), mode="constant")
        elif td_signal.shape[-1] > target_N:
            td_signal = td_signal[:, :target_N]

        N = td_signal.shape[-1]
        times = np.arange(N) * dt + t_start
        fs = 1.0 / dt

        BaseProcessingStep.__init__(
            self, times, td_signal, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.injection_params_full_basis = injection_params_full_basis
        self.tdi_chan = tdi_chan


################
#  RECIPE STEPS
################


class EMRIPERecipeStep(RecipeStep):
    """PE-only step (runs indefinitely; outer ``run_mcmc`` count caps it)."""

    def setup_run(self, iteration, last_sample, sampler):
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        return False


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

    # Pull the cached ResponseWrapper (built once at data-load time) and
    # wrap it for the template path. ``EMRIWaveWrap`` returns a
    # ``DomainBase`` so ACA add/remove_signal_to_residual and the EMRI
    # move's get_waveform_here use the right kernels.
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    template_wave_gen = get_emri_response_wrapper(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=T_START,
        tdi_config=tdi_config,
        tdi_chan="XYZ",
        role="template",
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend="cpu",
    )
    wave_gen = EMRIWaveWrap(
        template_wave_gen,
        td_settings,
        domain_settings,
        td_window=None,
        nchannels=acs.nchannels,
    )

    # Pre-inject starting templates into each walker's residual.
    if np.any(emri_inds := state.branches_inds["emri"][0]):
        print(
            f"[setup_recipe] pre-injecting EMRI starts; nleaves_max={emri_inds.shape[-1]}",
            flush=True,
        )
        for leaf in range(emri_inds.shape[-1]):
            if not emri_inds[0, leaf]:
                continue
            assert np.all(emri_inds[:, leaf])
            inj_coords = state.branches_coords["emri"][0, :, leaf]
            inj_coords_in = emri_info.transform.both_transforms(inj_coords)
            n_starts = inj_coords.shape[0]
            print(
                f"[setup_recipe] generating + applying {n_starts} starting waveforms (leaf {leaf}) one at a time...",
                flush=True,
            )
            # One EMRI waveform at a time: generate, add to that walker's
            # residual, drop the reference, force a gc cycle. Keeps peak RAM
            # at one waveform worth instead of ``n_starts`` simultaneously.
            for i in range(n_starts):
                print(
                    f"[setup_recipe]   walker {i}/{n_starts}: generating...",
                    flush=True,
                )
                sig = wave_gen(*inj_coords_in[i], **emri_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()
    print("[setup_recipe] pre-injection done; building EMRI PE move", flush=True)

    betas_all = np.tile(
        make_ladder(emri_info.ndim, ntemps=ntemps), (emri_info.nleaves_max, 1)
    )
    state.sub_states["emri"].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, emri_info.nleaves_max, emri_info.ndim)

    emri_pe_move = ResidualAddOneRemoveOneMove(
        "emri",
        coords_shape,
        wave_gen,
        emri_info.waveform_kwargs.copy(),
        emri_info.waveform_kwargs.copy(),
        acs,
        emri_info.num_prop_repeats,
        emri_info.transform,
        priors,
        emri_info.inner_moves,
        Tmax=np.inf,
        betas_all=betas_all,
    )
    emri_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(
        EMRIPERecipeStep(moves=[emri_pe_move]), name="emri pe"
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

    return EMRISetup(emri_settings)


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    base_file_name = "emri_only_smoke_test"
    file_store_dir = "./gf_output/"

    nwalkers = 4
    ntemps = 2

    domain_settings = DOMAIN_CHOICE

    processor_init_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        t_start=T_START,
        injection_params_full_basis=INJECTION_PARAMS_FULL_BASIS,
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
        data_processor=SyntheticEMRIProcessingStep,
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
