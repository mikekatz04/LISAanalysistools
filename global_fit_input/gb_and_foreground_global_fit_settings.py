"""Global-fit settings for the GB + foreground run.

This file is meant to be edited directly. Two high-level config blocks
live near the top:

* The *backend* block (right below this docstring) picks the array
  module + GPU backend name. Edit the import lines themselves — comment
  the cupy branch and uncomment the numpy fallback to run on CPU, or
  change ``GPU_BACKEND`` to match a different CUDA toolkit.
* The *top-of-file knobs* section (after the imports) holds the run
  surface — ``TOBS_TARGET`` / ``NWALKERS`` / ``NTEMPS`` are
  env-overridable, the WDM grid is derived from ``TOBS_TARGET`` via
  :meth:`WDMSettings.adjust_to_even_bins`, and ``TDI_CHAN`` pins the
  TDI generation (mirrors ``full_year_combined_global_fit_settings.py``).
* ``DOMAIN_CHOICE`` (further down) is the :class:`DomainSettingsBase`
  instance or factory the engine consumes. Different time-frequency
  grids are different ``DOMAIN_CHOICE`` values — see the factories
  defined just above the assignment.
"""

import logging
import os
import shutil
from copy import deepcopy

import h5py
import numpy as np


# ============================================================
# *** Backend selection ***
# ============================================================
#
# Edit these lines directly. ``GPU_BACKEND`` is the string the rest of
# the lisatools / gbgpu / fastlisaresponse stack expects as a
# ``force_backend`` argument; ``cp`` is the array module (cupy on GPU,
# numpy on CPU) used by this settings file when building objects.
try:
    import cupy as cp

    GPU_BACKEND = "cuda13x"  # change to "cuda11x" / "cuda12x" if needed
    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as cp

    GPU_BACKEND = "cpu"
    gpu_available = False
# ============================================================


logger = logging.getLogger(__name__)
level = logging.INFO
logger.setLevel(level)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.detector import EqualArmlengthOrbits, L1Orbits
from eryn.moves import TemperatureControl
from lisatools.utils.constants import YRSID_SI
from gbgpu.utils.utility import get_fdot
from lisatools.globalfit.hdfbackend import (
    EMRIHDFBackend,
    GBHDFBackend,
    GFHDFBackend,
    MBHHDFBackend,
)
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.state import MBHState, EMRIState, GBState

from lisatools.globalfit.generatefuncs import *
from lisatools.utils.utility import AET
from lisatools.sampling.prior import (
    AmplitudeFromSNR,
    AmplitudeFrequencySNRPrior,
    GBPriorWrap,
    SNRPrior,
)

from lisatools.globalfit.stock.erebor import (
    GalForSetup,
    GalForSettings,
    PSDSetup,
    PSDSettings,
    MBHSetup,
    MBHSettings,
    GBSetup,
    GBSettings,
)

from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove, CombineMove
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.globalfit.moves import (
    GBSpecialStretchMove,
    GBSpecialRJRefitMove,
    GBSpecialRJSearchMove,
    GBSpecialRJPriorMove,
    PSDMove,
    MBHSpecialMove,
    ResidualAddOneRemoveOneMove,
    GBSpecialRJSerialSearchMCMC,
    GFCombineMove,
)
from lisatools.globalfit.galaxyglobal import make_gmm
from lisatools.globalfit.moves import GlobalFitMove
from lisatools.utils.utility import tukey
from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.domains import WDMSettings

# basic transform functions for pickling
def f_ms_to_s(x):
    return x * 1e-3

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep, SangriaProcessingStep
from lisatools.globalfit.recipe import Recipe, RecipeStep
import time

from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe_steps import (
    SearchRecipeStep,
    PERecipeStep,
    RJRecipeStep,
    build_psd_moves,
    build_gb_moves,
)
from lisatools.globalfit.priors.gbpriors import get_fdot_mojito


# ============================================================
# *** Top-of-file knobs (the "surface" the user touches) ***
# ============================================================
# (mirrors full_year_combined_global_fit_settings.py)

# Target observation length and sample step. TOBS_TARGET is
# env-overridable so tests can run on a shorter/longer stretch (e.g.
# TOBS_TARGET=2.6e6 for ~1 month) without editing this file. Sangria
# training data is sampled at dt = 5 s.
TOBS_TARGET = float(os.environ.get("TOBS_TARGET", 90 * 86400.0))
DT = 5.0

# ~1-hour wavelet-duration search window for adjust_to_even_bins. The
# lower bound is the 3600 s the original hand-computed grid used; at the
# default TOBS_TARGET (90 d) this reproduces Nf=720 / Nt=2160 exactly.
WAVELET_DUR_BOUNDS = (3600.0, 4400.0)

# Sangria input file (env-overridable).
LDC_SOURCE_FILE = os.environ.get(
    "LDC_SOURCE_FILE",
    "/Users/mkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5",
)

# TDI channel — TDI generation is derived from it.
TDI_CHAN = "XYZ"
_CHAN_TO_GEN = {
    "XYZ": 2, "AET": 2, "XYZ2": 2, "AET2": 2,
    "XYZ1": 1, "AET1": 1,
}
if TDI_CHAN not in _CHAN_TO_GEN:
    raise ValueError(
        f"TDI_CHAN={TDI_CHAN!r} not recognised. "
        f"Add it to _CHAN_TO_GEN to pin the TDI generation."
    )
TDI_GEN = _CHAN_TO_GEN[TDI_CHAN]
TDI_GEN_STR = f"{TDI_GEN}{'nd' if TDI_GEN == 2 else 'st'} generation"
NCHANNELS = 3

# Engine-level (NWALKERS / NTEMPS env-overridable for smoke tests).
RANDOM_SEED = 103209
NWALKERS = int(os.environ.get("NWALKERS", 4))
NTEMPS = int(os.environ.get("NTEMPS", 2))

# No Tukey taper for this smoke test — the chunked-het templates are
# built without windowing, and we want the data path to match.
# ``window_taper_duration = 0`` gives ``alpha = 0`` which is a
# rectangular window inside :func:`lisatools.utils.utility.windowfun`.
WINDOW_TAPER_DURATION = 0.0  # rectangular window

# Output
FILE_STORE_DIR = "./gf_output/"
BASE_FILE_NAME = "gb_fg_smoke_test"


# ============================================================
# *** Derived: WDM grid via adjust_to_even_bins ***
# ============================================================
NF, NT, WAVELET_DURATION = WDMSettings.adjust_to_even_bins(
    t_min=WAVELET_DUR_BOUNDS[0],
    t_max=WAVELET_DUR_BOUNDS[1],
    dt=DT,
    Tobs=TOBS_TARGET,
)
TOBS = NF * NT * DT
TARGET_N = NF * NT  # total TD sample count

logger.info(
    "WDM grid: Nf=%d, Nt=%d, wavelet_duration=%.1f s, Tobs=%.6e s (target %.6e s)",
    NF, NT, WAVELET_DURATION, TOBS, TOBS_TARGET,
)


################

### DEFINE RECIPE

#############


def setup_recipe(
    recipe: Recipe,
    engine_info,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict[str, ProbDistContainer],
    state,
):
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    # Build the WDM-domain GB likelihood here (after the deepcopy in
    # ``CurrentInfoGlobalFit.__init__``) — the underlying C++ orbits wrap
    # is not picklable, so it must live outside the settings dataclass.
    #
    # The chunked-heterodyne ``GBWDMHeterodyne`` is the only WDM backend
    # supported here -- the lookup-table path has been removed sprint-wide.
    # The instance exposes the same ``get_ll_wdm`` / ``get_swap_ll_wdm`` /
    # ``get_ll_grad_wdm`` / ``hessian_wdm`` / ``fill_global_wdm`` surface
    # that :class:`WDMBandLikelihoodEngine` consumes; ``hessian_wdm`` is
    # what lets the chunked-het NUTS / gradient move replace the
    # info-matrix Cholesky proposal (see
    # ``LISAanalysistools/src/lisatools/globalfit/moves/gbspecialstretch.py``).
    gb_info = curr.source_info["gb"]
    if (
        isinstance(general_info.domain_settings, WDMSettings)
        and gb_info.gb_wdm_comp is None
    ):
        # Chunked-heterodyne backend. Reads grid params off the resolved
        # WDM domain settings; injection / orbit start time comes from
        # the data processor (defaults to 0 -- adjust if the data has
        # been offset). N_cp_sig=48 / N_cp_orbit=32 are the validated
        # defaults (median mm5 ~1e-9; see CLAUDE.md sprint root).
        #
        # ``force_backend`` is the single point at which the chunked-het
        # compute backend (cpu / cuda12x / ... / jax) is selected for
        # ``gb_wdm_comp``. Per the sprint-wide "no runtime backend
        # kwarg" rule, the Buffer / WDMBandLikelihoodEngine no longer
        # accept ``backend=`` at call time — get_ll, swap_ll, get_ll_grad
        # and hessian all dispatch through whichever backend this
        # instance was built with. To run the JAX-autograd grad/hessian
        # path, build a separate ``GBWDMHeterodyne(force_backend="jax")``
        # and hand it to the Buffer.
        import sys
        _gb_wdm_het_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "scripts", "gb_chunked_het",
            )
        )
        if _gb_wdm_het_dir not in sys.path:
            sys.path.insert(0, _gb_wdm_het_dir)
        from gb_wdm_het import GBWDMHeterodyne

        _wdm = general_info.domain_settings
        # ``data_t0`` is the runtime data start time set by the engine
        # (``times[0]`` after the preprocess trim) — the chunked-het WDM
        # grid must be anchored there, not at 0.
        _t_obs_start = float(getattr(general_info, "data_t0", 0.0))
        # CHUNKED_JAX_CHUNK env knob lets a JAX-backed instance split
        # the leaf axis for memory; unused on the C++ backends. None
        # falls through to GBWDMHeterodyne's _resolve_jax_chunk which
        # honours GBHET_JAX_CHUNK / JAX_GRAD_CHUNK at call time.
        _jax_chunk_env = os.environ.get("CHUNKED_JAX_CHUNK")
        _jax_chunk = int(_jax_chunk_env) if _jax_chunk_env else None
        gb_info.gb_wdm_comp = GBWDMHeterodyne(
            Nf=_wdm.Nf, Nt=_wdm.Nt, dt=general_info.dt,
            T_full=general_info.Tobs, t_ref_full=gb_info.t0,
            Nt_sub=int(os.environ.get("CHUNKED_NT_SUB", 256)),
            n_pad=int(os.environ.get("CHUNKED_N_PAD", 32)),
            N_sparse=int(os.environ.get("CHUNKED_N_SPARSE", 256)),
            nchannels=3,
            force_backend=general_info.force_backend,
            tdi_gen=TDI_GEN_STR,
            orbits=general_info.gpu_orbits,
            t_obs_start=_t_obs_start,
            N_cp_sig=int(os.environ.get("CHUNKED_N_CP_SIG", 48)),
            N_cp_orbit=int(os.environ.get("CHUNKED_N_CP_ORBIT", 32)),
            jax_chunk=_jax_chunk,
        )
        logger.info(
            "Chunked-het GB likelihood: Nf=%d Nt=%d Nt_sub=%d N_sparse=%d "
            "N_cp_sig=%d N_cp_orbit=%d (t_obs_start=%.3e t_ref=%.3e)",
            _wdm.Nf, _wdm.Nt,
            gb_info.gb_wdm_comp.Nt_sub,
            gb_info.gb_wdm_comp.N_sparse,
            gb_info.gb_wdm_comp.N_cp_sig,
            gb_info.gb_wdm_comp.N_cp_orbit,
            _t_obs_start, gb_info.t0,
        )

    #* ================================= BUILD MOVES =================================
    # Smoke test: pretend search is already done — run PE moves only so the
    # PSD does one standard MCMC pass and yields to the GB move, instead of
    # the search PSD move's max-loglikelihood convergence loop.
    num_repeats_psd = 5  # standard = 60 (smoke test: keep PSD inner loop short so GB runs sooner)
    permute_every_psd = 50  # standard = 50
    psd_search_move, psd_pe_move = build_psd_moves(
        engine_info,
        curr,
        acs,
        priors,
        num_repeats=num_repeats_psd,
        permute_every=permute_every_psd,
    )
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    # Smoke test: use ONLY the prior-based RJ proposal for GBs (100%) —
    # drops the fstat / refit moves. ``build_gb_moves`` returns the PE list
    # as ``[prior, refit, fstat_mcmc]``; keep just the prior one.
    gb_pe_moves = [m for m in gb_pe_moves if "prior" in m.name]

    #* ================================= SETUP PE (no search) =================================
    all_pe_moves = [psd_pe_move] + gb_pe_moves
    gf_pe_move = GFCombineMove(
        moves=all_pe_moves, verbose=True, share_temperature_control=False
    )
    gf_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(
        PERecipeStep(moves=[gf_pe_move]), name="gb + psd pe"
    )


#######################
##### SETTINGS ###########
###############


# ----------------------------------------------------------------------
# Domain selection helpers
#
# The engine accepts either a fully constructed DomainSettingsBase or a
# factory ``(times, dt, force_backend) -> DomainSettingsBase``. We
# define one factory per domain so the choice is a single-line swap at
# the call site (no string flag).
# ----------------------------------------------------------------------


# ============================================================
# *** High-level user choice: domain ***
# ============================================================
#
# Pick the basis grid directly here. ``DOMAIN_CHOICE`` is the value the
# engine consumes via ``GeneralSettings.domain_settings`` — either a
# fully constructed :class:`DomainSettingsBase` or a factory
# ``(times, dt, force_backend) -> DomainSettingsBase``. Each Settings
# class exposes ``make_factory(...)`` for building these. Different
# time-frequency grids are different values of ``DOMAIN_CHOICE`` (not
# different strings):
#
#   FD : DOMAIN_CHOICE = FDSettings.make_factory(min_freq=5e-5, max_freq=3e-2)
#   STFT: DOMAIN_CHOICE = STFTSettings.make_factory(big_dt=24*3600.0, ...)
#   WDM : DOMAIN_CHOICE = WDMSettings.make_factory(Nf=2048, Nt=8192, ...)
#
# The WDM grid (NF / NT / WAVELET_DURATION) is derived from TOBS_TARGET
# at the top of this file via ``WDMSettings.adjust_to_even_bins``. At
# the default 90-day TOBS_TARGET on Sangria's dt = 5 s this gives
# ~1-hour wavelets with Nf = 720 and Nt = 2160 (both even);
# layer_df = 1/(2*Nf*dt) ≈ 1.389e-4 Hz.
#
# Crop 20 wavelets from each time edge so the boundary wavelets (which
# extend past the data) don't show up as NaN in the active WDM band.
# Matches the pattern the original NF365 lookup table used.
DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=1e-4,
    max_freq=2.5e-2,
    min_time=20 * WAVELET_DURATION,
    max_time=(NT - 20) * WAVELET_DURATION,
)
# WDM lookup table removed sprint-wide -- the chunked-heterodyne template
# pipeline (gb_wdm_het.GBWDMHeterodyne) is now the only WDM backend. The
# build is wired in ``setup_recipe`` above when the domain is WDM.
# Example alternates:
# DOMAIN_CHOICE = FDSettings.make_factory(min_freq=5e-5, max_freq=3e-2)
# DOMAIN_CHOICE = STFTSettings.make_factory(big_dt=24 * 3600.0, min_freq=5e-5, max_freq=3e-2)
# ============================================================


def get_gb_erebor_settings(general_set: GeneralSetup) -> GBSetup:
    delta_safe = 1e-5

    A_lims = [7e-26, 1e-19]
    f0_lims = [0.05e-3, 2.5e-2]  #! TODO: check validity for mojito

    m_chirp_lims = [0.001, 1.0]
    # fdot bounds from the mojito-catalog relation (stft_tof), evaluated at
    # the band's upper edge -- matches the erebor GBSetup default.
    fdot_lims = [get_fdot_mojito(f0_lims[-1], sign="-"), get_fdot_mojito(f0_lims[-1], sign="+")]
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    lam_lims = [0.0, 2 * np.pi]
    beta_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    # GB band runs across the full active band of the parent domain
    # settings (resolved by GeneralSetup at construction time). The
    # GBSetup.init_band_structure clamps WDM runs to [ind_min_f,
    # ind_max_f] automatically; FD runs follow ``min_freq``/``max_freq``
    # already baked into ``FDSettings``.
    domain_settings = general_set.domain_settings
    if hasattr(domain_settings, "min_freq") and domain_settings.min_freq is not None:
        start_freq = float(domain_settings.min_freq)
    else:
        start_freq = 5e-5
    if hasattr(domain_settings, "max_freq") and domain_settings.max_freq is not None:
        end_freq = float(domain_settings.max_freq)
    else:
        end_freq = 3e-2

    oversample = 4
    extra_buffer = 5
    start_freq_ind = 0
    # TODO obtain ``t0_gbs`` properly from orbits; copied from federico's
    # gbgpu validation for the time being.
    t0_gbs = 97729089.327664
    initialize_kwargs = dict(force_backend=general_set.force_backend)

    # WDM-domain GB likelihood object: built lazily in ``setup_recipe``
    # below. We can't hold it on the settings dataclass because
    # ``CurrentInfoGlobalFit.__init__`` deepcopies the whole settings tree,
    # and the underlying fastlisaresponse C++ wrap (``OrbitsWrapCPU``)
    # is not picklable.
    gb_wdm_comp = None

    # Match GB tempering ladder length to ``general_set.ntemps`` so the
    # ``band_temps`` array initialized inside ``state.initialize_band_information``
    # has the shape the assertion there checks for.
    ntemps_gb = general_set.ntemps
    gb_betas = 1.0 / 1.2 ** np.arange(ntemps_gb)
    gb_betas[-1] = 1e-4
    
    assert start_freq and end_freq and general_set.Tobs and general_set.preprocess_kwargs
    start_freq_ind = int(start_freq * general_set.Tobs)
    
    try:
        data_start_time = getattr(general_set.orbits, 'sc_t0')
    except AttributeError:
        data_start_time = 97729089.327664 + 850.5
    t0_gbs = data_start_time + general_set.preprocess_kwargs["trim_kwargs"]["trim_duration"]
    
    initialize_kwargs = dict(force_backend=general_set.gpu_backend)
    
    gb_settings = GBSettings(
        A_lims=A_lims,
        f0_lims=f0_lims,
        m_chirp_lims=m_chirp_lims,
        fdot_lims=fdot_lims,
        phi0_lims=phi0_lims,
        iota_lims=iota_lims,
        psi_lims=psi_lims,
        lam_lims=lam_lims,
        beta_lims=beta_lims,
        start_freq=start_freq,
        end_freq=end_freq,
        oversample=oversample,
        extra_buffer=extra_buffer,
        start_freq_ind=start_freq_ind,
        t0=t0_gbs,
        tdi_setup=TDI_CHAN,
        use_tdi2=(TDI_GEN == 2),
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs,
        domain_settings=domain_settings,
        gb_wdm_comp=gb_wdm_comp,
        betas=gb_betas,
        nleaves_max=100,  # smoke test: keep state arrays small
        nleaves_min=0,
        ndim=8,
        log_dir=general_set.file_store_dir,
    )

    return GBSetup(gb_settings)


def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:
    initialize_kwargs_psd = dict()

    priors_psd = {
        r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),
        r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),
    }
    priors = {"psd": ProbDistContainer(priors_psd)}
    injection = np.array([15e-12, 3e-15])

    psd_settings = PSDSettings(
        ndim=2,
        injection=injection,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        priors=priors,
        log_dir=general_set.file_store_dir,
    )

    return PSDSetup(psd_settings)


def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:
    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        log_dir=general_set.file_store_dir,
    )

    return GalForSetup(galfor_settings)


def get_general_erebor_settings() -> GeneralSetup:
    gpus = [0] if gpu_available else None
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    # The basis grid is configured at the top of this file
    # (``DOMAIN_CHOICE``); we just forward it here. The WDM lookup
    # table has been removed -- chunked-het is built lazily in
    # ``setup_recipe``.
    domain_settings = DOMAIN_CHOICE

    processor_init_kwargs = dict(
        data_input_path=LDC_SOURCE_FILE,
        remove_from_data=["mbhb"],  # keep GB foreground / noise; drop MBHs
    )

    preprocess_kwargs = dict(normalize=False)

    # CompositeSensitivityBackend is the default; only ``tdi_generation`` is
    # consumed. The legacy XYZ-only kwargs (mask_percentage / use_splines /
    # spline_order) are filtered out at the engine.
    sensitivity_init_kwargs = dict(tdi_generation=TDI_GEN)

    general_settings = GeneralSettings(
        Tobs=TOBS,
        dt=DT,
        file_store_dir=FILE_STORE_DIR,
        base_file_name=BASE_FILE_NAME,
        main_file_key="testing",
        domain_settings=domain_settings,
        random_seed=RANDOM_SEED,
        backup_iter=5,
        nwalkers=NWALKERS,
        ntemps=NTEMPS,
        window_type="tukey",
        window_taper_duration=WINDOW_TAPER_DURATION,
        gpu_backend=GPU_BACKEND,
        gpus=gpus,
        data_processor_class=SangriaProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=normalize_window,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


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

    ###############################
    ###############################
    ######    Rank/GPU setup  #####
    ###############################
    ###############################

    head_rank = 1
    main_rank = 0

    rank_info = RankInfo(head_rank=head_rank, main_rank=main_rank)

    gb_setup = get_gb_erebor_settings(general_setup)
    psd_setup = get_psd_erebor_settings(general_setup)
    galfor_setup = get_galfor_erebor_settings(general_setup)

    global_settings = GlobalFitSettings(
        source_info={
            "gb": gb_setup,
            "psd": psd_setup,
            "galfor": galfor_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print(
        f"GB + foreground settings constructed OK\n"
        f"  Tobs = {TOBS:.6e} s  (target {TOBS_TARGET:.6e} s)\n"
        f"  WDM grid Nf={NF}, Nt={NT}, wavelet_duration={WAVELET_DURATION:.1f} s\n"
        f"  Backend: {GPU_BACKEND} (GPU available={gpu_available})\n"
        f"  Data: {LDC_SOURCE_FILE}"
    )
