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
from lisatools.domains import WDMSettings, FDSettings, STFTSettings
from lisatools.sensitivity import tdi_generation_from_channel

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep, SangriaProcessingStep
from lisatools.globalfit.recipe import Recipe, RecipeStep
import time

from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe import (
    SearchRecipeStep,
    PERecipeStep,
    RJRecipeStep,
    build_gb_moves,
    select_gb_injection_subset_by_snr,
    setup_state_for_injection,
    subtract_gb_neighbors_from_data,
)


# ============================================================
# *** DEBUG MODE (single switch) ***
# ============================================================
# GB_DEBUG=1 turns this settings file into the fast smoke configuration.
# The debug values below are seeded through ``os.environ.setdefault``, so
# every knob keeps its usual explicit env override and the ONLY difference
# from a true run is this flag: short Tobs (3 d), 3 walkers / 2 temps,
# small chunked-het kernel sizes, a handful of iterations, and the GB
# special-move debug instrumentation (band residual round-trip ll checks +
# begin/middle/end band plots; consumed inside ``build_gb_moves`` from the
# same env var, plots under GB_DEBUG_DIR, default ./gf_output/gb_debug/).
#
#   GB_DEBUG=1 python scripts/run_global.py \
#       -sfp global_fit_input/gb_no_foreground_global_fit_settings.py
#
GB_DEBUG = bool(int(os.environ.get("GB_DEBUG", "0")))
if GB_DEBUG:
    for _knob, _debug_val in {
        "TOBS_TARGET": str(3 * 86400.0),  # 3 days
        "NWALKERS": "3",
        "NTEMPS": "2",
        "CHUNKED_NT_SUB": "64",
        "CHUNKED_N_PAD": "8",
        "CHUNKED_N_SPARSE": "64",
        "CHUNKED_N_CP_SIG": "16",
        "CHUNKED_N_CP_ORBIT": "16",
        "GF_NUM_ITER": "4",
    }.items():
        os.environ.setdefault(_knob, _debug_val)


# ============================================================
# *** RUN MODE: parameter estimation vs search ***
# ============================================================
# GB_MODE=pe (default): today's behavior -- leaves start AT the true
# catalogue points (SNR > 3 subset, scattered), no leaf caps.
#
# GB_MODE=search: the realistic zero-knowledge configuration. The gb
# branch starts with ZERO leaves (the engine's default prior-draw state,
# all inds False) and the per-band progressive leaf cap arms: every band
# allows at most GB_LEAF_CAP_START (default 1) leaves per (temp, walker)
# cell -- enforced at every temperature via a -inf prior on over-cap RJ
# births -- and each band's cap increments INDEPENDENTLY once it has
# spent GB_LEAF_CAP_MIN_ITERS iterations at the current cap and every
# cold walker's band residual ll has converged to within
# GB_LEAF_CAP_LL_NSIGMA * sqrt(N_dof/2) of the running best (plus, by
# default, an occupancy check: some cold walker must actually hold cap
# leaves in the band). RJ stays on throughout; the caps only bound it
# from above. See GBSpecialBase._update_band_leaf_caps for details.
#
#   GB_MODE=search GB_DEBUG=1 python scripts/run_global.py \
#       -sfp global_fit_input/gb_no_foreground_global_fit_settings.py
#
GB_MODE = os.environ.get("GB_MODE", "pe").lower()
if GB_MODE not in ("pe", "search"):
    raise ValueError(f"GB_MODE must be 'pe' or 'search', got {GB_MODE!r}.")
if GB_MODE == "search":
    # Arm the leaf-cap machinery (consumed by build_gb_moves via env);
    # explicit env overrides still win via setdefault.
    os.environ.setdefault("GB_LEAF_CAP_START", "1")
    # Annealing default: phase-maximised RJ births (two-quadrature
    # analytic maximisation in the band engines; accepted phi0 rotated
    # to the maximum). GB_RJ_PHASE_MAXIMIZE=0 turns it back off.
    os.environ.setdefault("GB_RJ_PHASE_MAXIMIZE", "1")


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

# Wavelet-duration search window for adjust_to_even_bins
# (env-overridable). Default ~1 h reproduces the original grid; the
# validated mojito match scripts (gb_mojito_match / three_ways) run on
# ~4-h wavelets (Nf*dt = 14600 s) -- set WAVELET_DUR_MIN/MAX = 14000 /
# 15000 to reproduce their grid.
WAVELET_DUR_BOUNDS = (
    float(os.environ.get("WAVELET_DUR_MIN", 3600.0)),
    float(os.environ.get("WAVELET_DUR_MAX", 4400.0)),
)

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

# TDI channel — TDI generation is derived from it via the stock helper.
TDI_CHAN = "XYZ"
TDI_GEN = tdi_generation_from_channel(TDI_CHAN)
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

# Data->WDM Tukey window, expressed as the scipy ``alpha``
# (env-overridable). The engine derives ``window_alpha =
# window_taper_duration / Tobs``, so the taper DURATION is computed
# from this alpha after the grid is known (below). alpha = 0.05 matches
# the validated mojito match scripts; the WDM edge crop below is sized
# to cover the tapered region (edge-cut >= alpha*Nt/2, the chunked-het
# Tukey/edge-leak rule).
WINDOW_TUKEY_ALPHA = float(os.environ.get("WINDOW_TUKEY_ALPHA", "0.05"))

# Output
FILE_STORE_DIR = str(os.environ.get("FILE_STORE_DIR", "./gf_output_gb_no_fg/"))
BASE_FILE_NAME = str(os.environ.get("BASE_FILE_NAME", "gb_no_fg_test_2"))


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

# Derived data-window taper duration (seconds): the engine computes
# ``window_alpha = window_taper_duration / Tobs``, so this realises the
# WINDOW_TUKEY_ALPHA knob above on the resolved grid.
WINDOW_TAPER_DURATION = WINDOW_TUKEY_ALPHA * TOBS

# WDM time-edge crop (wavelets per end): must cover both the boundary
# wavelets AND the Tukey taper region (edge-cut >= alpha*Nt/2 -- the
# chunked-het Tukey/edge-leak rule).
EDGE_CROP_WAVELETS = max(20, int(np.ceil(WINDOW_TUKEY_ALPHA * NT / 2)) + 4)

logger.info(
    "WDM grid: Nf=%d, Nt=%d, wavelet_duration=%.1f s, Tobs=%.6e s (target %.6e s), "
    "window alpha=%.3f (taper %.0f s), edge crop=%d wavelets",
    NF, NT, WAVELET_DURATION, TOBS, TOBS_TARGET,
    WINDOW_TUKEY_ALPHA, WINDOW_TAPER_DURATION, EDGE_CROP_WAVELETS,
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
# floor (not round) so the SAMPLABLE interior layer is the one that
# CONTAINS GB_CENTER_FREQ -- rounding up pushed a source in the upper
# half of its layer into the lower guard band, where the SNR-cut
# injection never looks.
_gb_k_center = int(np.floor(GB_CENTER_FREQ / LAYER_DF))
_gb_k_lo = _gb_k_center - GB_N_LAYERS // 2
GB_MIN_FREQ = _gb_k_lo * LAYER_DF
GB_MAX_FREQ = (_gb_k_lo + GB_N_LAYERS) * LAYER_DF
logger.info(
    "GB band: [%.6e, %.6e] Hz (%d WDM layers @ ~%.3f mHz, layer_df=%.4e Hz)",
    GB_MIN_FREQ, GB_MAX_FREQ, GB_N_LAYERS, GB_CENTER_FREQ * 1e3, LAYER_DF,
)

# Debug-plot cell selection: with GB_DEBUG=1 the GB move saves ONE figure
# per plotted step for a single (walker, band) cell -- a panel per
# temperature -- instead of a PNG for every cell the proposal touches.
# Defaults: walker 0 and the CENTRAL GB band (the samplable layer that
# contains GB_CENTER_FREQ). Env-overridable like every other knob.
os.environ.setdefault("GB_DEBUG_PLOT_WALKER", "0")
os.environ.setdefault("GB_DEBUG_PLOT_BAND", str(GB_N_LAYERS // 2))

# ============================================================
# *** ACA frequency clipping (memory knob) ***
# ============================================================
# DATA_BAND_LAYERS (env, int) narrows the DATA band -- and with it every
# per-walker ACA slab (data + XYZ-cross invC scale with Nf_active) -- to
# +-DATA_BAND_LAYERS WDM layers around the GB band, instead of the full
# [6, 25] mHz. This is what makes a full-scale (many walkers x many
# temps) run on a single source's sub-bands fit in laptop memory. The
# clipped band must comfortably contain the GB band plus the chunked-het
# 5-layer inner-product gating; >= GB_N_LAYERS/2 + 5 is safe. Unset =
# full data band (default, unchanged).
_data_band_layers = os.environ.get("DATA_BAND_LAYERS")
if _data_band_layers is not None:
    _L = int(_data_band_layers)
    assert _L >= GB_N_LAYERS // 2 + 5, (
        f"DATA_BAND_LAYERS={_L} too narrow: need >= GB_N_LAYERS//2 + 5 = "
        f"{GB_N_LAYERS // 2 + 5} to cover the GB band + the 5-layer "
        "chunked-het gating."
    )
    MIN_FREQ = max(MIN_FREQ, (_gb_k_center - _L) * LAYER_DF)
    MAX_FREQ = min(MAX_FREQ, (_gb_k_center + 1 + _L) * LAYER_DF)
    logger.info(
        "ACA data band CLIPPED to [%.6e, %.6e] Hz (+-%d layers around the "
        "GB band).", MIN_FREQ, MAX_FREQ, _L,
    )

# (GB_DEBUG is defined in the DEBUG MODE block at the top of the file.)


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
        # (``times[0]`` after the preprocess trim). The chunked-het evaluates
        # the orbits at chunk times anchored on ``wdm_settings.t0`` (-> the
        # het's ``self.t_obs_start``); the Mojito orbits + data live at the
        # absolute epoch ``data_t0`` (~9.8e7 s). Nothing in the engine anchors
        # the WDM domain t0 to data_t0, so it stays 0 -> the het would sample
        # the orbits ~1e8 s BEFORE their span and every WDM template comes back
        # identically zero (=> NaN <h|h>). Anchor it here so ``t_obs_start`` is
        # the absolute data start. (WDM coefficient addressing is t0-independent
        # -- only the absolute orbit-evaluation times depend on this.)
        _t_obs_start = float(getattr(general_info, "data_t0", 0.0))
        _orig_wdm_t0 = float(_wdm.t0)
        _wdm.t0 = _t_obs_start
        # The band WDMSettings domain (_wdm) carries Nf/Nt/dt + the active band
        # (ind_min_f/ind_max_f/ind_min_t/ind_max_t) + t0; t_ref is the GB phase
        # reference (was t_ref_full). tdi_config replaces tdi_gen.
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
            "N_cp_sig=%d N_cp_orbit=%d (domain t0 %.6e -> het t_obs_start=%.6e, "
            "t_ref=%.6e, chunk_t_starts=[%.6e, %.6e])",
            _wdm.Nf, _wdm.Nt,
            gb_info.gb_wdm_comp.Nt_sub,
            gb_info.gb_wdm_comp.N_sparse,
            gb_info.gb_wdm_comp.N_cp_sig,
            gb_info.gb_wdm_comp.N_cp_orbit,
            _orig_wdm_t0, gb_info.gb_wdm_comp.t_obs_start, gb_info.t0,
            float(gb_info.gb_wdm_comp.chunk_t_starts.min()),
            float(gb_info.gb_wdm_comp.chunk_t_starts.max()),
        )

        # GB_SIGHET_INMODEL=1: score the IN-MODEL repeat blocks through the
        # signal-heterodyne likelihood (per-source reference rebuilt each
        # repeat block from the source-free cell residual) while RJ /
        # fills / swaps keep the chunked-het path. Pure type dispatch: the
        # move receives a GBSignalHetComputations wrapping the chunked
        # comp above; its setup_in_model/clear_in_model hooks (called
        # inside _run_in_model_repeats at the friends/info-matrix stage)
        # do the switching -- chunked comps inherit no-op hooks. CPU-only
        # for now (the sig-het kernels have no CUDA build yet).
        if bool(int(os.environ.get("GB_SIGHET_INMODEL", "0"))):
            from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
            gb_info.gb_wdm_comp = GBSignalHetComputations.for_band_engine(
                gb_info.gb_wdm_comp,
                nt_layer=int(os.environ.get("SIGHET_NT_LAYER", 64)),
                n_sparse_fd=int(os.environ.get("SIGHET_N_SPARSE_FD", 1024)),
            )
            logger.info(
                "GB in-model likelihood: SIGNAL-HET "
                "(chunked-het delegate for RJ / fills / swaps)."
            )

    # FD-domain mirror of the block above: the GB move auto-builds a
    # config-only ``GBFDComputations(fd_settings, t_ref, ...)`` from the
    # ACA's FDSettings; it only needs the orbits + TDI-config handles
    # (C++ orbit wraps are not picklable, so they are wired here after the
    # settings deepcopy, exactly like ``gb_wdm_comp`` on the WDM path).
    if isinstance(general_info.domain_settings, FDSettings):
        if getattr(gb_info, "orbits", None) is None:
            gb_info.orbits = general_info.gpu_orbits
        if getattr(gb_info, "tdi_config", None) is None:
            gb_info.tdi_config = TDI_GEN_STR

    # STFT-domain mirror of the WDM block: the STFT band engine drives an
    # ``STFTGBComputations`` whose ``stft_comps`` is the PARENT
    # ``STFTComputationGroup`` (the engine rebinds it to the band ACA's
    # per-split groups per call; the recipe's initial-subtraction arm and
    # the move-level parent fills use the parent binding built here). Like
    # ``gb_wdm_comp`` it holds C++ orbit wraps, so it cannot live on the
    # settings dataclass (deepcopy) and is built here instead.
    if (
        isinstance(general_info.domain_settings, STFTSettings)
        and getattr(gb_info, "gb_stft_comp", None) is None
    ):
        from gbgpu.gbcomps import STFTGBComputations

        # The ACA's per-split STFTComputationGroups must carry the Fresnel
        # configuration the engine expects; assigning domain_group_kwargs
        # invalidates any cached groups, so ``cpp_splits`` below rebuilds
        # with these kwargs. NOTE (pending box-run validation):
        # ``window_alpha`` here is the PER-SEGMENT window of the STFT
        # evaluator -- it must match how the engine STFT'd the data
        # stream; keep both on the same knob when wiring the real run.
        acs.domain_group_kwargs = dict(
            tdi_type=TDI_CHAN,
            window_alpha=float(os.environ.get("STFT_WINDOW_ALPHA", "0.0")),
            use_midpoint=bool(int(os.environ.get("STFT_USE_MIDPOINT", "0"))),
        )
        _stft = acs.settings
        gb_info.gb_stft_comp = STFTGBComputations(
            stft_comps=acs.cpp_splits[0],
            T=float(_stft.NT * _stft.dt),
            t_ref=gb_info.t0,
            orbits=general_info.gpu_orbits,
            tdi_config=TDI_GEN_STR,
            force_backend=general_info.force_backend,
            # n_side_bins=2 -> ~5 columns per source; the accuracy floor is
            # the shared band-truncation leakage (see the STFT agreement
            # studies), env-overridable for wider stencils.
            n_side_bins=int(os.environ.get("STFT_N_SIDE_BINS", "2")),
            window_factor=1.0,
            # Default ON: per-bin Fresnel chirp frequency/rate from the TDI
            # phase (includes the LISA orbital Doppler, whose rate typically
            # exceeds the astrophysical fdot).
            freq_from_tdi_phase=bool(
                int(os.environ.get("STFT_FREQ_FROM_TDI_PHASE", "1"))
            ),
        )
        logger.info(
            "STFT GB likelihood: NT=%d NF_active=%d big_dt=%.1f s t0=%.6e "
            "n_side_bins=%d freq_from_tdi_phase=%s window_alpha=%.3f t_ref=%.6e",
            _stft.NT, _stft.NF_active, float(_stft.dt), float(_stft.t0),
            gb_info.gb_stft_comp.n_side_bins,
            gb_info.gb_stft_comp.freq_from_tdi_phase,
            acs.domain_group_kwargs["window_alpha"],
            gb_info.t0,
        )

    #* ============== START GBs AT TRUE CATALOG POINTS (SNR cut) ==============
    # Read the GB catalogue, compute optimal SNR (sqrt<h|h>) via the WDM
    # likelihood for sources inside the (narrow) GB band, cut at optimal
    # SNR > 3, and inject those as the starting leaves of the "gb" branch —
    # mirrors how MBH/EMRI/SOBBH start at truth in the full-year settings.
    # Runs before build_gb_moves so the moves' band setup sees true-point
    # coords. On a data source without a GB catalogue (e.g. Sangria) this is
    # a no-op and the state falls back to prior draws.
    # Domain-appropriate GB comp for the injection-seeding helpers below:
    # WDM runs use ``gb_wdm_comp``; STFT runs the ``gb_stft_comp`` built
    # above (``select_gb_injection_subset_by_snr`` dispatches on which
    # ``get_ll_*`` surface the comp exposes). FD runs have neither -> the
    # block is skipped and the state stays at the engine default.
    _gb_seed_comp = (
        gb_info.gb_wdm_comp
        if gb_info.gb_wdm_comp is not None
        else getattr(gb_info, "gb_stft_comp", None)
    )
    if _gb_seed_comp is not None:
        # GB_SUBTRACT_NEIGHBORS=1 (NOT the default): focused-central-band
        # mode. Injected (sampled) leaves are restricted to the CENTRAL GB
        # band layer, and every catalogue source in the neighboring layers
        # (window = 4 layers each side) is subtracted from the data as a
        # KNOWN signal so its spectral spread does not bias the in-band fit.
        # The two selections are disjoint by construction.
        _subtract_neighbors = bool(int(os.environ.get("GB_SUBTRACT_NEIGHBORS", "0")))
        _injection_f0_lims = None
        if _subtract_neighbors:
            assert gb_info.gb_wdm_comp is not None, (
                "GB_SUBTRACT_NEIGHBORS is WDM-only for now "
                "(subtract_gb_neighbors_from_data drives fill_global_wdm)."
            )
            _central_lims = (_gb_k_center * LAYER_DF,
                             (_gb_k_center + 1) * LAYER_DF)
            _injection_f0_lims = _central_lims
            subtract_gb_neighbors_from_data(
                curr, acs, gb_info, gb_info.gb_wdm_comp,
                exclude_f0_lims=_central_lims,
                window_hz=4 * LAYER_DF,
            )
        # GB_MODE=search: NO true-point seeding. The gb branch keeps the
        # engine's default zero-leaf state (prior-draw coords, all inds
        # False) and the sampler must FIND the sources through RJ under the
        # per-band progressive leaf cap (armed at the top of this file).
        # Neighbor subtraction above still applies -- known out-of-band
        # signals are removed from the data in either mode.
        if GB_MODE != "search":
            gb_snr_subset_inds = select_gb_injection_subset_by_snr(
                curr, acs, gb_info, _gb_seed_comp, snr_threshold=3.0,
                f0_lims=_injection_f0_lims,
            )
            # GB_START=prior: PE dimensionality WITHOUT truth seeding --
            # the same number of leaves as the SNR-selected injection
            # subset, but every leaf's coordinates drawn from the PRIOR.
            # The sampler must assemble the fit itself (the debug seq
            # figures then show the total template converging toward the
            # data instead of starting on it). Default GB_START=truth is
            # the existing injection-seeded start.
            _gb_start = os.environ.get("GB_START", "truth").lower()
            if _gb_start not in ("truth", "prior"):
                raise ValueError(
                    f"GB_START must be 'truth' or 'prior', got {_gb_start!r}."
                )
            if _gb_start == "prior":
                _n_true = int(len(gb_snr_subset_inds))
                _gb_inds = state.branches["gb"].inds
                _gb_coords = state.branches["gb"].coords
                _nt, _nw, _nl, _ndim = _gb_coords.shape
                _draws = priors["gb"].rvs(size=(_nt, _nw, _n_true))
                _draws = (_draws.get() if hasattr(_draws, "get")
                          else np.asarray(_draws))
                _gb_inds[:] = False
                _gb_inds[:, :, :_n_true] = True
                _gb_coords[:, :, :_n_true, :] = _draws
                logger.info(
                    "GB start: %d leaves per walker at PRIOR draws "
                    "(GB_START=prior; truth seeding skipped).", _n_true,
                )
            else:
                # Per-dimension scatter for the true-point start. The stock scalar
                # default (1e-5) is ~10 orders too wide for the GB ``fdot`` dimension
                # (~1e-15 scale, prior half-width ~1e-12), so every scattered walker
                # lands outside the fdot prior. Use the existing per-dim ``spread``
                # array path, sized to a small fraction of each prior dimension's width
                # so it auto-scales across the decade-spanning GB parameters (logA,
                # f0[mHz], fdot, angles) and stays inside the prior even after the
                # betas widening applied in scatter_around_injection. Widths come from
                # the prior's own draws (correct sampling-basis order; the prior is
                # keyed by parameter labels, not dimension indices).
                _gb_draws = priors["gb"].rvs(size=20000)
                _gb_draws = _gb_draws.get() if hasattr(_gb_draws, "get") else np.asarray(_gb_draws)
                _gb_spread = 1e-4 * (_gb_draws.max(axis=0) - _gb_draws.min(axis=0))
                setup_state_for_injection(
                    curr, state, source_type="GB", branch_name="gb",
                    subset_inds=gb_snr_subset_inds, priors=priors, spread=_gb_spread,
                )

    #* ================================= BUILD MOVES =================================
    # No PSD / foreground moves: the PSD is fixed (no ``psd`` branch) and
    # the foreground is not fit, so the recipe holds GB moves only.
    # Smoke test: use ONLY the prior-based RJ proposal for GBs (100%) — drops
    # the fstat / refit moves and the search list. The design knobs on
    # ``build_gb_moves`` express this directly (replacing the old post-hoc
    # ``[m for m in gb_pe_moves if "prior" in m.name]`` filter).
    _, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state,
        include_search=False, pe_move_names=["rj_prior"],
    )

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
    min_time=EDGE_CROP_WAVELETS * WAVELET_DURATION,
    max_time=(NT - EDGE_CROP_WAVELETS) * WAVELET_DURATION,
)
# WDM lookup table removed sprint-wide -- the chunked-heterodyne template
# pipeline (gb_wdm_het.GBWDMHeterodyne) is now the only WDM backend. The
# build is wired in ``setup_recipe`` above when the domain is WDM.
# ``GB_DOMAIN=fd`` switches the run basis to the frequency domain,
# ``GB_DOMAIN=stft`` to the STFT/Fresnel basis (same band restriction);
# default (unset / "wdm") keeps the WDM grid above.
_GB_DOMAIN = os.environ.get("GB_DOMAIN", "wdm").lower()
if _GB_DOMAIN == "fd":
    DOMAIN_CHOICE = FDSettings.make_factory(
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
    )
elif _GB_DOMAIN == "stft":
    # STFT basis: ``STFT_BIG_DT`` (s) is the segment length (df = 1/big_dt);
    # 21600 s (6 h) is the segment scale the STFT GB kernel suite is
    # validated on. The factory anchors t0 = times[0] (absolute data
    # epoch), so no WDM-style t0 re-anchor is needed in setup_recipe.
    DOMAIN_CHOICE = STFTSettings.make_factory(
        big_dt=float(os.environ.get("STFT_BIG_DT", 21600.0)),
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
    )
elif _GB_DOMAIN != "wdm":
    raise ValueError(f"GB_DOMAIN must be 'wdm', 'fd', or 'stft', got {_GB_DOMAIN!r}.")
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
    # FD basis: GBSetup's FD band walker builds bands (2*N + extra_buffer)
    # rfft bins wide and trims the outer THREE edges, so the few-WDM-layer
    # band the WDM smoke uses is far too narrow (it yields ZERO bands) --
    # at the 3-day debug Tobs a single band is already ~2 mHz wide
    # (N=256 at 7.5 mHz). Default to the FULL data band (env-overridable
    # via GB_FD_BANDWIDTH, Hz, centered on GB_CENTER_FREQ) so several
    # bands survive the trim; a true 90-day run has 30x finer df and can
    # narrow this again. The STFT basis shares this branch: GBSetup's
    # band structure falls into the same FD-style walker (the STFT band
    # engine is whole-grid, so bands are bookkeeping windows there), and
    # the same zero-band trim hazard applies.
    if isinstance(domain_settings, (FDSettings, STFTSettings)):
        _bw = os.environ.get("GB_FD_BANDWIDTH")
        if _bw is None:
            start_freq, end_freq = MIN_FREQ, MAX_FREQ
        else:
            _half_bw = 0.5 * float(_bw)
            start_freq = max(MIN_FREQ, GB_CENTER_FREQ - _half_bw)
            end_freq = min(MAX_FREQ, GB_CENTER_FREQ + _half_bw)
        logger.info(
            "%s basis: GB band widened to [%.6e, %.6e] Hz for the FD-style band walker.",
            type(domain_settings).__name__, start_freq, end_freq,
        )

    # FD/STFT basis at the short debug Tobs: oversample=2 halves the
    # per-band N (128 at 7.5 mHz -- still ample for a near-monochromatic
    # GB) so the band walker yields several bands across the data band.
    oversample = 2 if isinstance(domain_settings, (FDSettings, STFTSettings)) else 4
    extra_buffer = 5
    start_freq_ind = 0
    # GB phase/frequency reference epoch = the mojito catalogue epoch
    # (``MOJITO_REFERENCE_TIME`` = TimeReferenceSSBFrame). This is the
    # VALIDATED convention from scripts/gb/gb_mojito_match.py +
    # scripts/gb_chunked_het/gb_mojito_mcmc_three_ways.py (mm ~ 1e-8
    # against the band-passed data): catalogue params are used AT this
    # epoch (no trim evolution) and the chunked-het comp gets
    # ``t_ref=REF`` while the WDM domain t0 stays the trimmed data start.
    # The conversion in ``gb_catalogue_to_sampling_basis`` must use the
    # SAME anchor (params at REF, phys phi0 = +TrueAnomaly).
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
        # Max GB leaves per chain (env-overridable). RJ cost scales with
        # this (the BandSorter pre-draws EVERY inds=False leaf and scores
        # one RJ step per leaf per iteration), so focused few-source runs
        # should set it near the expected source count (e.g. 6), not 100.
        nleaves_max=int(os.environ.get("GB_NLEAVES_MAX", 100)),
        nleaves_min=0,
        ndim=8,
        # In-model repeat count per picked source (env-overridable).
        # High-fidelity runs want enough repeats that the friends table /
        # Fisher factor stay valid between rebuilds (user guidance: ~300
        # for the focused single-source runs).
        num_repeat_proposals=int(os.environ.get("GB_NUM_REPEAT_PROPOSALS", 100)),
        # Sampling-basis periodic parameters, keyed by the same plain
        # names as the priors/transform (run.py translates names ->
        # indices through the branch transform's ``input_basis`` before
        # building the sampler's PeriodicContainer, which the engine
        # wires onto every move -- without it the GB move's in-model
        # wrap crashes on ``periodic=None``).
        periodic={"gb": {"phi0": 2 * np.pi, "psi": np.pi,
                         "alpha": 2 * np.pi}},
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
        # Env-overridable iteration cap (default 500). Set GF_NUM_ITER=2 for a
        # quick GB_DEBUG smoke run that exercises a few proposals.
        num_iterations=int(os.environ.get("GF_NUM_ITER", 500)),
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
