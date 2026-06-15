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

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import (
    DomainBaseArray,
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
    SOBBHSettings,
    SOBBHSetup,
    make_sobbh_transform_container,
)
from lisatools.sources.sobbh import SOBBHWaveform
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


# ---- Cached SOBBH generator + wrappers (shared across data path + moves) ----

_WAVE_GEN_CACHE = {}


def get_sobbh_response_wrapper(
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
    reference_time: Optional[float] = None,
):
    """Build (and cache) a :class:`ResponseWrapper` around :class:`SOBBHWaveform`.

    One generator is built per
    ``(Tobs, dt, t_start, tdi_chan, order, force_backend, reference_time)``
    cache key — injection + template paths share the same instance, mirroring
    the EMRI smoke setup.

    ``reference_time`` is the absolute epoch at which ``f_low`` is defined; it
    is decoupled from the data-window start ``t_start`` (the PN inspiral
    measures time from it). This run is synthetic-only (``t_start == 0``) so it
    is left ``None`` -> f_low at the window start; the kwarg exists so this
    duplicate stays signature-consistent with the canonical
    ``global_fit_settings.get_sobbh_response_wrapper``.
    """
    key = (Tobs, dt, t_start, tdi_chan, order, force_backend, reference_time)
    if key in _WAVE_GEN_CACHE:
        return _WAVE_GEN_CACHE[key]

    sobbh_generator = SOBBHWaveform(
        Tobs=Tobs,
        dt=dt,
        t0=t_start,
        reference_time=reference_time,
        force_backend=force_backend,
    )

    # SOBBH output-basis positions of lam / beta (see header).
    response_kwargs = {
        "Tobs": Tobs / YRSID_SI,
        "dt": dt,
        "index_lambda": 7,
        "index_beta": 8,
        "flip_hx": True,
        "force_backend": force_backend,
        "tdi": tdi_config,
        "tdi_chan": tdi_chan,
        "order": order,
        "remove_garbage": "zero",
        "is_ecliptic_latitude": True,
        "t_buffer": t_buffer,
    }

    orbits = EqualArmlengthOrbits(force_backend=force_backend)
    wave_gen = ResponseWrapper(
        sobbh_generator,
        orbits=orbits,
        t0=t_start,
        **response_kwargs,
    )
    _WAVE_GEN_CACHE[key] = wave_gen
    return wave_gen


class SOBBHWaveWrap:
    """Adapter that runs the cached ResponseWrapper and projects to the run's domain.

    Mirrors ``EMRIWaveWrap`` from ``emri_only_global_fit_settings.py``: the
    output is a :class:`DomainBase` subclass (FDSignal / WDMSignal / …) so
    the global-fit move and ACA dispatch land on the right kernels.
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
        # SOBBHWaveform doesn't use convert_to_ra_dec; ResponseWrapper
        # will pop the flag from kwargs.
        call_kwargs.setdefault("convert_to_ra_dec", False)
        arr = np.asarray(self.wave_gen(*params, **call_kwargs))
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )


class SyntheticSOBBHProcessingStep(BaseProcessingStep):
    """Generate the SOBBH injection in-process via the shared ResponseWrapper.

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
        wave_gen = get_sobbh_response_wrapper(
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

        # Pad/clip to exactly ``Tobs/dt`` to keep WDM ``N = Nf*Nt`` consistent.
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


class SOBBHPERecipeStep(RecipeStep):
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

    # Pull the cached ResponseWrapper (built once at data-load time) and
    # wrap it for the template path.
    tdi_config = TDIConfig("2nd generation", force_backend="cpu")
    template_wave_gen = get_sobbh_response_wrapper(
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
    wave_gen = SOBBHWaveWrap(
        template_wave_gen,
        td_settings,
        domain_settings,
        td_window=None,
        nchannels=acs.nchannels,
    )

    # Pre-inject starting templates into each walker's residual.
    if np.any(sobbh_inds := state.branches_inds["sobbh"][0]):
        print(
            f"[setup_recipe] pre-injecting SOBBH starts; nleaves_max={sobbh_inds.shape[-1]}",
            flush=True,
        )
        for leaf in range(sobbh_inds.shape[-1]):
            if not sobbh_inds[0, leaf]:
                continue
            assert np.all(sobbh_inds[:, leaf])
            inj_coords = state.branches_coords["sobbh"][0, :, leaf]
            inj_coords_in = sobbh_info.transform.both_transforms(inj_coords)
            n_starts = inj_coords.shape[0]
            print(
                f"[setup_recipe] generating + applying {n_starts} starting waveforms (leaf {leaf}) one at a time...",
                flush=True,
            )
            # One SOBBH waveform at a time: generate, add to that walker's
            # residual, drop the reference, force a gc cycle.
            for i in range(n_starts):
                print(
                    f"[setup_recipe]   walker {i}/{n_starts}: generating...",
                    flush=True,
                )
                sig = wave_gen(*inj_coords_in[i], **sobbh_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()
    print("[setup_recipe] pre-injection done; building SOBBH PE move", flush=True)

    betas_all = np.tile(
        make_ladder(sobbh_info.ndim, ntemps=ntemps), (sobbh_info.nleaves_max, 1)
    )
    state.sub_states["sobbh"].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, sobbh_info.nleaves_max, sobbh_info.ndim)

    sobbh_pe_move = ResidualAddOneRemoveOneMove(
        "sobbh",
        coords_shape,
        wave_gen,
        sobbh_info.waveform_kwargs.copy(),
        sobbh_info.waveform_kwargs.copy(),
        acs,
        sobbh_info.num_prop_repeats,
        sobbh_info.transform,
        priors,
        sobbh_info.inner_moves,
        Tmax=np.inf,
        betas_all=betas_all,
    )
    sobbh_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(
        SOBBHPERecipeStep(moves=[sobbh_pe_move]), name="sobbh pe"
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

    return SOBBHSetup(sobbh_settings)


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    base_file_name = "sobbh_only_smoke_test"
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
        data_processor=SyntheticSOBBHProcessingStep,
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
