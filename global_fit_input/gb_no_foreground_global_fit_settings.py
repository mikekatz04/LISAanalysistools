"""Global-fit settings for a GB-only run WITHOUT foreground fitting.

Copy of ``gb_and_foreground_global_fit_settings.py`` with:

* **No galactic-foreground branch** (``galfor`` removed from
  ``source_info``) — the foreground is not fit.
* **Fixed PSD** — no ``psd`` sampling branch either. The sensitivity is
  a fixed :class:`InstrumentNoise` built from the Sangria injection
  values via ``GeneralSettings.fixed_psd_kwargs``
  (``psd_params=[15e-12, 3e-15]``, ``galfor_params=None``), which the
  engine uses whenever no ``psd`` branch is present.
* **Frequency band restricted to f > 6 mHz** (``MIN_FREQ`` knob): both
  the WDM domain's active band and the GB ``f0`` prior start at 6 mHz.
  Above 6 mHz the unresolved galactic confusion in the Sangria data is
  negligible, so GB tests can run without any foreground noise model.

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

# No PSDSetup / GalForSetup here — the PSD is fixed and the foreground
# is not fit in this configuration.
from lisatools.globalfit.stock.erebor import (
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
    build_gb_moves,
    select_gb_injection_subset_by_snr,
    setup_state_for_injection,
)


# ============================================================
# *** Top-of-file knobs (the "surface" the user touches) ***
# ============================================================
# (mirrors full_year_combined_global_fit_settings.py)

# Target observation length and sample step. TOBS_TARGET is
# env-overridable so tests can run on a shorter/longer stretch (e.g.
# TOBS_TARGET=2.6e6 for ~1 month) without editing this file. Sangria
# training data is sampled at dt = 5 s.
TOBS_TARGET = float(os.environ.get("TOBS_TARGET", 90 * 86400.0))
# dt = 2.5 s matches the mojito L1 GB data (``GB_*_2.5s_L1_*.h5``), so the
# loaded data grid and the WDM grid stay consistent without a downsample
# step (mirrors full_year_combined_global_fit_settings.py's DT=2.5).
DT = 2.5

# ~1-hour wavelet-duration search window for adjust_to_even_bins. The
# lower bound is the 3600 s the original hand-computed grid used; at the
# default TOBS_TARGET (90 d) this reproduces Nf=720 / Nt=2160 exactly.
WAVELET_DUR_BOUNDS = (3600.0, 4400.0)

# Mojito L1 data folder (env-overridable). The GB galaxy is loaded as the
# data ("full data"), and ``catalogue["GB"]`` is populated so the GBs can be
# started at their true catalog points (see setup_recipe). This is the
# DEFAULT data source for this file.
MOJITO_DATA_PATH = os.environ.get(
    "MOJITO_DATA_PATH",
    "/Users/mkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",
)

# Sangria input file (env-overridable) — kept as an available alternative
# data source (see the commented Sangria block in get_general_erebor_settings).
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

# Frequency band: restrict to f > 6 mHz so the unresolved galactic
# confusion (which dominates well below this) is out of band — no
# foreground model needed anywhere in the fit.
MIN_FREQ = 6e-3
MAX_FREQ = 2.5e-2

# Fixed PSD (no psd sampling branch): Sangria injection values
# ``[Soms_d, Sa_a]`` in linear (square-root) units. Used by the engine's
# no-psd-branch path via ``GeneralSettings.fixed_psd_kwargs``;
# ``galfor_params=None`` means no galactic-foreground component is added
# to the sensitivity matrix.
FIXED_PSD_PARAMS = [15e-12, 3e-15]

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
BASE_FILE_NAME = "gb_no_fg_test"


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


# ============================================================
# *** GB band: narrow, decoupled from the (full) data band ***
# ============================================================
# The WDM domain active band above ([MIN_FREQ, MAX_FREQ]) is the DATA band
# and stays wide ("keep the data as the full data"). The GB f0 band /
# band_edges are restricted to a few WDM layers around ~7.5 mHz so a single
# bright GB is easy to visualize while the sampler runs. Because
# ``GBSetup.init_band_structure`` builds ONE band per WDM layer and raises if
# the GB range spans < 1 layer, the band is expressed in integer layer units
# (layer_df = 1/(2*Nf*dt)). GB_N_LAYERS=3 gives 3 sub-bands (1 samplable
# interior layer + 2 guard layers); bump to n+2 for n samplable layers.
LAYER_DF = 1.0 / (2 * NF * DT)
GB_CENTER_FREQ = float(os.environ.get("GB_CENTER_FREQ", 7.5e-3))
GB_N_LAYERS = int(os.environ.get("GB_N_LAYERS", 3))
_gb_k_center = int(round(GB_CENTER_FREQ / LAYER_DF))
_gb_k_lo = _gb_k_center - GB_N_LAYERS // 2
GB_MIN_FREQ = _gb_k_lo * LAYER_DF
GB_MAX_FREQ = (_gb_k_lo + GB_N_LAYERS) * LAYER_DF
logger.info(
    "GB band: [%.6e, %.6e] Hz (%d WDM layers @ ~%.3f mHz, layer_df=%.4e Hz)",
    GB_MIN_FREQ, GB_MAX_FREQ, GB_N_LAYERS, GB_CENTER_FREQ * 1e3, LAYER_DF,
)

# GB special-move debug instrumentation (band residual round-trip ll checks +
# begin/middle/end band plots). Off by default; set GB_DEBUG=1 to enable. The
# flag is consumed inside build_gb_moves via the same env var.
GB_DEBUG = bool(int(os.environ.get("GB_DEBUG", "0")))


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
        # Chunked-het GB likelihood: the INSTALLED GBGPU class (no sys.path hack,
        # no dev gb_wdm_het). GBWDMComputations takes the band WDMSettings domain +
        # t_ref directly -- the Nf/Nt/ind_*/T_full/t_obs_start args GBWDMHeterodyne
        # needed are all carried by the domain (Nf/Nt/dt/band/t0).
        from gbgpu.gbcomps import GBWDMComputations

        _wdm = general_info.domain_settings
        # ``data_t0`` is the runtime data start time set by the engine
        # (``times[0]`` after the preprocess trim) — the chunked-het WDM
        # grid must be anchored there, not at 0.
        _t_obs_start = float(getattr(general_info, "data_t0", 0.0))
        # The band WDMSettings domain (_wdm) carries Nf/Nt/dt + the active band
        # (ind_min_f/ind_max_f/ind_min_t/ind_max_t) + t0 (anchored at data_t0 by the
        # engine), so GBWDMComputations reads all of that from it directly. t_ref is
        # the GB phase reference (was t_ref_full). tdi_config replaces tdi_gen.
        gb_info.gb_wdm_comp = GBWDMComputations(
            _wdm,
            t_ref=gb_info.t0,
            Nt_sub=int(os.environ.get("CHUNKED_NT_SUB", 256)),
            n_pad=int(os.environ.get("CHUNKED_N_PAD", 32)),
            N_sparse=int(os.environ.get("CHUNKED_N_SPARSE", 256)),
            N_cp_sig=int(os.environ.get("CHUNKED_N_CP_SIG", 48)),
            N_cp_orbit=int(os.environ.get("CHUNKED_N_CP_ORBIT", 32)),
            orbits=general_info.gpu_orbits,
            tdi_config=TDI_GEN_STR,
            force_backend=general_info.force_backend,
            tdi_type="XYZ",
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

    #* ============== START GBs AT TRUE CATALOG POINTS (SNR cut) ==============
    # Read the GB catalogue, compute optimal SNR (sqrt<h|h>) via the WDM
    # likelihood for sources inside the (narrow) GB band, cut at optimal
    # SNR > 3, and inject those as the starting leaves of the "gb" branch —
    # mirrors how MBH/EMRI/SOBBH start at truth in the full-year settings.
    # Runs before build_gb_moves so the moves' band setup sees true-point
    # coords. On a data source without a GB catalogue (e.g. Sangria) this is
    # a no-op and the state falls back to prior draws.
    if gb_info.gb_wdm_comp is not None:
        gb_snr_subset_inds = select_gb_injection_subset_by_snr(
            curr, acs, gb_info, gb_info.gb_wdm_comp, snr_threshold=3.0
        )
        setup_state_for_injection(
            curr, state, source_type="GB", branch_name="gb",
            subset_inds=gb_snr_subset_inds, priors=priors,
        )

    #* ================================= BUILD MOVES =================================
    # No PSD / foreground moves: the PSD is fixed (no ``psd`` branch) and
    # the foreground is not fit, so the recipe holds GB moves only.
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    # Smoke test: use ONLY the prior-based RJ proposal for GBs (100%) —
    # drops the fstat / refit moves. ``build_gb_moves`` returns the PE list
    # as ``[prior, refit, fstat_mcmc]``; keep just the prior one.
    gb_pe_moves = [m for m in gb_pe_moves if "prior" in m.name]

    #* ================================= SETUP PE (no search) =================================
    all_pe_moves = gb_pe_moves
    gf_pe_move = GFCombineMove(
        moves=all_pe_moves, verbose=True, share_temperature_control=False
    )
    gf_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(
        PERecipeStep(moves=[gf_pe_move]), name="gb pe"
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
# NOTE: ``min_freq=MIN_FREQ`` (6 mHz) is the foreground-free band
# restriction — everything below it is excluded from the likelihood.
DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=MIN_FREQ,
    max_freq=MAX_FREQ,
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
    # f0 prior clamped to the foreground-free band (> 6 mHz) to match
    # the restricted WDM domain.
    f0_lims = [MIN_FREQ, MAX_FREQ]

    m_chirp_lims = [0.001, 1.0]
    fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])

    fdot_lims = [-fdot_max_val, fdot_max_val]
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    # ICRS run frame (post-merge): sky params are alpha/RA + delta/dec, not
    # ecliptic lam/beta. GBSettings consumes ``alpha_lims``/``delta_lims``.
    alpha_lims = [0.0, 2 * np.pi]
    delta_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    # GB band: narrow (few WDM layers @ ~7.5 mHz), DECOUPLED from the wide
    # WDM *data* band so a single bright source is visualizable. These are
    # the ``start_freq``/``end_freq`` overrides (module-level GB_MIN_FREQ /
    # GB_MAX_FREQ knobs derived from the WDM layer grid). GBSetup.
    # init_band_structure clamps the GB band to the WDM active band
    # automatically; here the GB band is already a strict subset of it.
    domain_settings = general_set.domain_settings
    start_freq = GB_MIN_FREQ
    end_freq = GB_MAX_FREQ

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

    gb_settings = GBSettings(
        A_lims=A_lims,
        f0_lims=f0_lims,
        m_chirp_lims=m_chirp_lims,
        fdot_lims=fdot_lims,
        phi0_lims=phi0_lims,
        iota_lims=iota_lims,
        psi_lims=psi_lims,
        alpha_lims=alpha_lims,
        delta_lims=delta_lims,
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


# No ``get_psd_erebor_settings`` / ``get_galfor_erebor_settings`` here:
# the PSD is fixed via ``fixed_psd_kwargs`` (engine no-psd-branch path)
# and the galactic foreground is not fit.


def get_general_erebor_settings() -> GeneralSetup:
    gpus = [0] if gpu_available else None
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    # The basis grid is configured at the top of this file
    # (``DOMAIN_CHOICE``); we just forward it here. The WDM lookup
    # table has been removed -- chunked-het is built lazily in
    # ``setup_recipe``.
    domain_settings = DOMAIN_CHOICE

    # DEFAULT data source: mojito L1. ``source_types=["GB"]`` loads the whole
    # GB galaxy TD (one file, ids=[0]) as the data AND populates
    # ``catalogue["GB"]`` so the GBs can be started at their true catalog
    # points via setup_state_for_injection. ``frame="icrs"`` matches the GB
    # sampling basis (alpha/RA, sin_delta, psi ICRS).
    data_processor_class = L1ProcessingStep
    processor_init_kwargs = dict(
        L1_folder=MOJITO_DATA_PATH,
        source_types=["GB"],  # add "VGB" to also load verification binaries
        source_ids=None,      # unused for GB/VGB (single-file galaxy)
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=GPU_BACKEND, frame="icrs"),
        # NOTE: do NOT pass Tobs here. The engine's preprocess trims
        # 200 h from each end and then keeps the first ``Tobs`` seconds, so
        # the final data must be exactly Nf*Nt samples. Pre-trimming to Tobs
        # at load would leave Tobs-1.44e6 s < Tobs and mismatch the fixed WDM
        # grid; loading the full file lets preprocess land on exactly Nf*Nt.
    )

    # --- Sangria alternative (kept available; not the default) --------------
    # To run on the Sangria training data instead, swap the two lines above
    # for:
    #     data_processor_class = SangriaProcessingStep
    #     processor_init_kwargs = dict(
    #         data_input_path=LDC_SOURCE_FILE,
    #         remove_from_data=["mbhb"],  # keep GB foreground / noise; drop MBHs
    #     )
    # NOTE: SangriaDataLoader does not populate ``catalogue`` — the GB truth
    # must be read from the HDF5 directly (fp["sky"]["dgb"/"igb"/"vgb"]["cat"],
    # see gathergalaxy.py:1181-1245) via sangria_gb_catalogue_to_sampling_basis.
    # -----------------------------------------------------------------------

    preprocess_kwargs = dict(normalize=False)

    # CompositeSensitivityBackend is the default; only ``tdi_generation`` is
    # consumed. The legacy XYZ-only kwargs (mask_percentage / use_splines /
    # spline_order) are filtered out at the engine.
    sensitivity_init_kwargs = dict(tdi_generation=TDI_GEN)

    # With no ``psd`` branch in ``source_info`` the engine builds every
    # walker's sensitivity from these fixed values (InstrumentNoise only;
    # ``galfor_params=None`` -> no foreground component).
    fixed_psd_kwargs = dict(
        psd_params=list(FIXED_PSD_PARAMS),
        galfor_params=None,
    )

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
        fixed_psd_kwargs=fixed_psd_kwargs,
        data_processor_class=data_processor_class,
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

    # GB only: no ``psd`` branch (fixed PSD via ``fixed_psd_kwargs``) and
    # no ``galfor`` branch (foreground not fit).
    global_settings = GlobalFitSettings(
        source_info={
            "gb": gb_setup,
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
        f"GB no-foreground settings constructed OK\n"
        f"  Tobs = {TOBS:.6e} s  (target {TOBS_TARGET:.6e} s), dt = {DT} s\n"
        f"  WDM grid Nf={NF}, Nt={NT}, wavelet_duration={WAVELET_DURATION:.1f} s\n"
        f"  Data band (WDM domain): [{MIN_FREQ:.3e}, {MAX_FREQ:.3e}] Hz (full data)\n"
        f"  GB band: [{GB_MIN_FREQ:.6e}, {GB_MAX_FREQ:.6e}] Hz "
        f"({GB_N_LAYERS} WDM layers @ ~{GB_CENTER_FREQ*1e3:.3f} mHz)\n"
        f"  Walkers/Temps: NWALKERS={NWALKERS}, NTEMPS={NTEMPS}, GB_DEBUG={GB_DEBUG}\n"
        f"  PSD: FIXED at {FIXED_PSD_PARAMS} (no psd branch, no galfor branch)\n"
        f"  Backend: {GPU_BACKEND} (GPU available={gpu_available})\n"
        f"  Data: mojito L1 GB galaxy @ {MOJITO_DATA_PATH}"
    )
