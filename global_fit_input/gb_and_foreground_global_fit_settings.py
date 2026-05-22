"""Global-fit settings for the GB + foreground run.

This file is meant to be edited directly. Two high-level config blocks
live near the top:

* The *backend* block (right below this docstring) picks the array
  module + GPU backend name. Edit the import lines themselves — comment
  the cupy branch and uncomment the numpy fallback to run on CPU, or
  change ``GPU_BACKEND`` to match a different CUDA toolkit.
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
from lisatools.domains import WDMLookupTable, WDMSettings

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
    gb_info = curr.source_info["gb"]
    if (
        isinstance(general_info.domain_settings, WDMSettings)
        and general_info.wdm_lookup_table is not None
        and gb_info.gb_wdm_comp is None
    ):
        from fastlisaresponse.gbcomps import GBWDMComputations

        gb_info.gb_wdm_comp = GBWDMComputations(
            wdm_lookup_table=general_info.wdm_lookup_table,
            T=general_info.Tobs,
            t_ref=gb_info.t0,
            orbits=general_info.gpu_orbits,
            tdi_config="2nd generation" if gb_info.use_tdi2 else "1st generation",
            force_backend=general_info.force_backend,
            tdi_type=gb_info.tdi_setup,
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
# When running WDM, also set ``WDM_LOOKUP_TABLE`` (instance or
# ``(WDMSettings) -> WDMLookupTable`` factory) so the table is built
# once the WDM grid is known.
# Smoke test: 3-month Tobs (= 90 d) on Sangria's dt = 5 s gives a total
# of 1,555,200 samples. With ~1-hour wavelets (layer_dt = Nf*dt = 3600 s),
# Nf = 720 and Nt = 2160 (both even). layer_df = 1/(2*Nf*dt) ≈ 1.389e-4 Hz.
#
# Crop 20 wavelets (~20 hours) from each time edge so the boundary
# wavelets (which extend past the data) don't show up as NaN in the
# active WDM band. Matches the pattern the original NF365 lookup table
# used (73,000 s ≈ 20 layers off each end).
DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=720,
    Nt=2160,
    min_freq=1e-4,
    max_freq=2.5e-2,
    min_time=20 * 3600.0,
    max_time=(2160 - 20) * 3600.0,
)
WDM_LOOKUP_TABLE = lambda wdm_settings: WDMLookupTable.from_file(
    "wdm_lookup_n_ref_NF720_NT2160_3mo.h5",
    force_backend=GPU_BACKEND,
)
# Example alternates:
# DOMAIN_CHOICE = FDSettings.make_factory(min_freq=5e-5, max_freq=3e-2)
# WDM_LOOKUP_TABLE = None
# DOMAIN_CHOICE = STFTSettings.make_factory(big_dt=24 * 3600.0, min_freq=5e-5, max_freq=3e-2)
# ============================================================


def get_gb_erebor_settings(general_set: GeneralSetup) -> GBSetup:
    delta_safe = 1e-5

    A_lims = [7e-26, 1e-19]
    f0_lims = [0.05e-3, 2.5e-2]  #! TODO: check validity for mojito

    m_chirp_lims = [0.001, 1.0]
    fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])

    fdot_lims = [-fdot_max_val, fdot_max_val]
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
    # and the underlying fastlisaresponse C++ wrap (``OrbitsWrapCPU_responselisa``)
    # is not picklable.
    gb_wdm_comp = None

    # Match GB tempering ladder length to ``general_set.ntemps`` so the
    # ``band_temps`` array initialized inside ``state.initialize_band_information``
    # has the shape the assertion there checks for.
    ntemps_gb = general_set.ntemps
    gb_betas = 1.0 / 1.2 ** np.arange(ntemps_gb)
    gb_betas[-1] = 1e-4

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
        tdi_setup="XYZ",
        use_tdi2=True,
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
    # Smoke test: 3 months on Sangria training data (dt = 5 s).
    Tobs = 90 * 86400.0
    dt = 5.0

    ldc_source_file = "/Users/mkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5"
    base_file_name = "gb_fg_smoke_test"
    file_store_dir = "./gf_output/"

    gpus = [0] if gpu_available else None
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])
    # Lightweight smoke-test config.
    nwalkers = 4
    ntemps = 2

    # No Tukey taper for this smoke test — the WDM lookup table is built
    # without windowing, and we want the data path to match.
    # ``window_taper_duration = 0`` gives ``alpha = 0`` which is a
    # rectangular window inside :func:`lisatools.utils.utility.windowfun`.
    window_taper_duration = 0.0

    # The basis grid and (optional) WDM lookup table are configured at
    # the top of this file (``DOMAIN_CHOICE`` / ``WDM_LOOKUP_TABLE``);
    # we just forward them here.
    domain_settings = DOMAIN_CHOICE
    wdm_lookup_table = WDM_LOOKUP_TABLE

    processor_init_kwargs = dict(
        data_input_path=ldc_source_file,
        remove_from_data=["mbhb"],  # keep GB foreground / noise; drop MBHs
    )

    preprocess_kwargs = dict(normalize=False)

    sensitivity_init_kwargs = dict(
        tdi_generation=2, mask_percentage=0.02, use_splines=False
    )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        main_file_key="testing",
        domain_settings=domain_settings,
        wdm_lookup_table=wdm_lookup_table,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_taper_duration=window_taper_duration,
        gpu_backend=GPU_BACKEND,
        gpus=gpus,
        data_processor=SangriaProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
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
    breakpoint()
