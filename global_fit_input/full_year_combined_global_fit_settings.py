"""Full-year combined-source MCMC settings — EMRI + SOBBH + MBH (phentax).

Source classes
--------------
- MBH (phentax IMRPhenomTHM -> ``PhenomTHMTDIWaveform``, stft_tof
  structure): StretchMove inside ResidualAddOneRemoveOneMove. The TD
  waveform goes through the legacy ``pyResponseTDI`` response and is
  placed on the full data grid, then transformed to the run's WDM
  domain via the canonical ``TDSignal.transform`` chain
  (``output_domain_settings``). Waveform basis: ``(m1, m2, s1z, s2z,
  dist [Mpc], phi_ref, inc, psi, ra, dec, t_plunge)`` — sky in ICRS
  (the run frame; ``bbhx.mbhtdionfly.MBHTDIonFly`` remains available in
  BBHx as the spline TDI-on-the-fly alternative but is not wired here).
- EMRI                     : StretchMove inside ResidualAddOneRemoveOneMove.
- SOBBH                    : StretchMove inside ResidualAddOneRemoveOneMove.

Template generation runs **under the hood in the engine**: each branch
registers a params-based ``signal_gen`` on its Setup; ``run.py``'s
``setup_acs(rebuild_residuals=True)`` subtracts the state's current
templates from the residuals (the converted ``get_templates`` process).
The recipe only builds moves — no ``subtract_initial_signal`` /
pre-injection loops here.

Run frame: ICRS. Catalogue sky/polarization parameters are sampled raw
(``alpha``/RA, ``sin_delta``, ``psi`` ICRS) and the mojito orbits are
loaded with ``frame='icrs'`` so the response consumes them directly —
no per-injection frame conversion.

No GB branch, no PSD branch, no galactic-foreground (galfor) sampling
branch. Instrument noise and galactic foreground are fixed components of
the sensitivity model AND of the synthesized data realization — same
constants drive both, so the model is consistent with the data.

Data sources
------------
- DATA_PROCESSOR = "mojito"   : :class:`L1ProcessingStepWithSyntheticNoise`
  subclasses :class:`L1ProcessingStep` (the mojito loader) and, after
  the L1 source TD signals are loaded, sums synthetic FD-correlated
  instrument noise + annually-modulated galactic-foreground TD realization.
- DATA_PROCESSOR = "synthetic": :class:`SyntheticDataProcessor` builds
  per-class injection signals via the cached response wrappers (no
  mojito file needed) and sums the same noise + foreground TD streams.
  Source counts derive from MOJITO_SOURCE_IDS so the two paths produce
  the same number of leaves per class.

Annual foreground modulation
----------------------------
- TD data realization: foreground TD samples multiplied by
  ``sqrt(annual_modulation_envelope(t))`` before summation.
- WDM noise covariance: a :class:`AnnualModulatedGalacticForeground`
  (subclass of GalacticForeground) is passed via
  ``sensitivity_init_kwargs['extra_components']`` so its per-(t, f) pixel
  modulation is folded into the likelihood automatically.
- Same (ANNUAL_AMP, ANNUAL_PHASE0) constants drive both.

GPU
---
Install cupy (cuda12x / cuda13x) and set GPUS = [<dev_id>] in
:func:`get_general_erebor_settings`. All response wrappers (EMRI, SOBBH,
MBH) and the synthetic noise generator pick up ``GPU_BACKEND`` when
``gpu_available``. When cupy isn't available the configuration falls
back to a pure-CPU run automatically.

WDM grid
--------
Sized via :meth:`WDMSettings.adjust_to_even_bins` with ``dt = 2.5 s`` and
a half-day wavelet-duration search window so both Nf and Nt are even.
Actual achieved Tobs may differ from ``TOBS_TARGET`` by a few wavelet
pixels — that's the helper's job.
"""

import gc
import logging
import os
import shutil
import sys
from copy import deepcopy
from typing import Optional, Sequence

import h5py
import numpy as np


# ============================================================
# *** Backend selection ***
# ============================================================
try:
    import cupy as cp

    GPU_BACKEND = "cuda13x"
    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as cp

    GPU_BACKEND = "cpu"
    gpu_available = False
# ============================================================

logger = logging.getLogger(__name__)


from lisatools.response.tdiconfig import TDIConfig

from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.detector import DefaultOrbits, LISAModel
from lisatools.sources.emri import emri_catalogue_to_waveform_basis
from lisatools.domains import (
    FDSettings,
    TDSettings,
    WDMSettings,
    place_td_signal_on_grid,
)
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
    Setup,
)
from lisatools.globalfit.moves import GFCombineMove, ResidualAddOneRemoveOneMove
from lisatools.globalfit.preprocessing import (
    BaseProcessingStep,
    L1ProcessingStep,
    normalize_source_ids,
)
from lisatools.globalfit.recipe import Recipe
from lisatools.globalfit.recipe import (
    MOJITO_REFERENCE_TIME,
    PERecipeStep,
    build_mbh_moves_phenom,
    mbh_catalogue_to_sampling_basis,
    EMRIMoveBuilder,
    MBHMoveBuilder,
    SOBBHMoveBuilder,
)
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    EMRISettings,
    EMRISetup,
    MBHSettings,
    MBHSetup,
    SOBBHSettings,
    SOBBHSetup,
    make_mbh_transform_container,
    make_emri_transform_container,
)
from lisatools.sensitivity import (
    CompositeSensitivityMatrix,
    GalacticForeground,
    InstrumentNoise,
    generate_correlated_instrument_noise_td,
    tdi_generation_from_channel,
)
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground
from lisatools.utils.constants import YRSID_SI

from eryn.moves import StretchMove
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, log_uniform, uniform_dist
from eryn.utils import TransformContainer


# ============================================================
# *** Sibling-script imports (reuse waveform wrappers + injections) ***
# ============================================================
# run_global.py loads this module via spec_from_file_location so its
# sibling directory isn't on sys.path; add it here so the relative
# import resolves both ways (CLI and ``python -m`` / direct import).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from global_fit_settings import (
    EMRIWaveWrap,
    INJECTION_PARAMS_FULL_BASIS as _SINGLE_EMRI_INJECTION,
    MBHTDIonFlyWaveWrap,
    SOBBHTDIonFlyWaveWrap,
    SOBBHWaveWrap,
    SOBBH_INJECTION_PARAMS_FULL_BASIS as _SINGLE_SOBBH_INJECTION,
    emri_full_to_sampling,
    get_emri_response_wrapper,
    get_mbh_tdionfly_gen,
    get_sobbh_response_wrapper,
    get_sobbh_tdionfly_gen,
    sobbh_full_to_sampling,
)

from mbh_phentax_only_global_fit_settings import (
    INJECTION_PARAMS_FULL_BASIS as _SINGLE_MBH_PHENTAX_INJECTION,
)

# ============================================================
# *** Top-of-file knobs (the "surface" the user touches) ***
# ============================================================

# Template path: TDI-on-the-fly (default; validated 2026-06-15 to beat
# the legacy pyResponse path against the mojito data by 1-2 orders of
# magnitude) vs the legacy path. Set USE_TDIONFLY=0 to revert
# MBH -> PhenomTHMTDIWaveform and SOBBH -> ResponseWrapper(SOBBHWaveform).
# (EMRI is always legacy here.)
USE_TDIONFLY = os.environ.get("USE_TDIONFLY", "1") == "1"

# Mojito data window. Default = the FULL data span (production global
# fit). Set CHOP_WINDOW=1 to opt in to a short per-source snippet (the
# test / validation path): a merger-centered window for MBH, ~6 months
# for SOBBH / EMRI. TOBS_TARGET is computed below once the active source
# is known (it differs between the full-window and chopped cases); both
# knobs are env-overridable. MERGER_FRAC is where the MBH merger sits in
# the chopped window (matches scripts/mbh/mbh_mojito_match_debug.py).
CHOP_WINDOW = os.environ.get("CHOP_WINDOW", "0") == "1"
MERGER_FRAC = 0.72

DT = 2.5

# Half-day-ish wavelet-duration search window for adjust_to_even_bins.
# Env-overridable (WAVELET_DUR_MIN / WAVELET_DUR_MAX) so smoke tests can
# pick a tiny wavelet duration that yields a small WDM grid at a short
# TOBS_TARGET without editing this file.
WAVELET_DUR_BOUNDS = (
    float(os.environ.get("WAVELET_DUR_MIN", 40000.0)),
    float(os.environ.get("WAVELET_DUR_MAX", 48000.0)),
)

# Data source selection.
#   "mojito"    — load source TD signals from a mojito L1 folder.
#   "synthetic" — build source TD signals locally for testing.
DATA_PROCESSOR = os.environ.get("DATA_PROCESSOR", "mojito")
MOJITO_DATA_PATH = os.environ.get(
    "MOJITO_DATA_PATH",
    "/Users/mlkatz/.mojito_cache/brickmarket/mojito_light_v1_0_0/",  # USER: set this to the actual L1 folder.
)

# Per-class source IDs (used by both modes — defines the leaf counts).
# Edit to match the mojito catalog you intend to load. Each class is
# capped at len(<list>) leaves per branch.
MOJITO_SOURCE_IDS = {
    "MBHB": [], # range(2)
    "EMRI": [1],  # 0, 1, 2], # range(2)
    "SOBHB": [],  # [0, 1, 2], # range(2)
}


# ``normalize_source_ids`` (per-class id -> list coercion) now lives in the
# stock ``lisatools.globalfit.preprocessing`` module.


# Per-class env override (used by the test / validation harness to select a
# single source without editing this file): MBHB_IDS / EMRI_IDS / SOBHB_IDS,
# comma-separated ints; empty string -> []. Unset -> keep the literal above.
for _cls in ("MBHB", "EMRI", "SOBHB"):
    _env_ids = os.environ.get(f"{_cls}_IDS")
    if _env_ids is not None:
        MOJITO_SOURCE_IDS[_cls] = [
            int(x) for x in _env_ids.split(",") if x.strip() != ""
        ]

MOJITO_SOURCE_IDS = normalize_source_ids(MOJITO_SOURCE_IDS)

# Active source = the single non-empty class. Single-source-at-a-time
# snippet runs (CHOP_WINDOW=1) require exactly one class; the full-window
# default supports the combined multi-source fit. The UX is: edit
# MOJITO_SOURCE_IDS above ([1] for the active class, [] for the rest) and
# set CHOP_WINDOW=1 for a snippet run — TOBS, the window center, and the
# on-the-fly-vs-legacy routing all follow from that.
_ACTIVE = [k for k, v in MOJITO_SOURCE_IDS.items() if v]
ACTIVE_SOURCE = _ACTIVE[0] if len(_ACTIVE) == 1 else None

if not CHOP_WINDOW:
    # Production default: full data span (one year).
    TOBS_TARGET = float(os.environ.get("TOBS_TARGET", YRSID_SI))
elif ACTIVE_SOURCE == "MBHB":
    # Short merger-centered snippet (~7 weeks) — covers the in-band
    # inspiral + merger for one MBH.
    TOBS_TARGET = float(os.environ.get("TOBS_TARGET", 48 * 86400.0))
else:
    # SOBBH / EMRI: ~6 months.
    TOBS_TARGET = float(os.environ.get("TOBS_TARGET", 0.5 * YRSID_SI))

if CHOP_WINDOW and ACTIVE_SOURCE is None:
    raise ValueError(
        "CHOP_WINDOW=1 is a single-source-at-a-time snippet run: set exactly "
        "one of MOJITO_SOURCE_IDS non-empty (got "
        f"{[k for k, v in MOJITO_SOURCE_IDS.items() if v]}). Leave CHOP_WINDOW "
        "unset/0 for the full-window combined fit."
    )

# Synthetic instrument noise (fixed, no PSD branch).
ADD_INSTRUMENT_NOISE = False
NOISE_SOMS_D = 15e-12
NOISE_SA_A = 3e-15
NOISE_SEED = 12345

# Galactic foreground (fixed, no galfor branch). Pass `None` to use the
# FittedHyperbolicTangentGalacticForeground default tabulated values.
ADD_GALACTIC_FOREGROUND = False
FOREGROUND_PARAMS = None  # or e.g. (3.27e-44, 1e-2, 1.183, 941.0, 103.0)
FOREGROUND_SEED = 67890

# Annual modulation envelope: noise foreground TD amplitude (and WDM
# covariance modulation) gets a (1 + A*cos(2*pi*t/yr + phi0)) factor.
ANNUAL_AMP = 0.10
ANNUAL_PHASE0 = 0.0

# TDI channel — TDI generation is derived from it via the stock helper.
TDI_CHAN = "XYZ"
TDI_GEN = tdi_generation_from_channel(TDI_CHAN)
TDI_GEN_STR = f"{TDI_GEN}{'nd' if TDI_GEN == 2 else 'st'} generation"
NCHANNELS = 3

# Synthetic-mode start time (mojito mode pulls t_start from the L1 file).
SYNTHETIC_T_START = 0.0

# MBH phentax waveform window before merger (seconds; the phentax ``T``).
# ``None`` (env value "none") falls back to the full data span. The
# 1-month default keeps per-call cost down while covering the in-band
# signal. NOTE: must stay <= the earliest sampled t_plunge, otherwise the
# waveform grid extends before the data start.
_mwd_env = os.environ.get("MBH_WAVEFORM_DURATION", str(YRSID_SI / 12.0))
MBH_WAVEFORM_DURATION = (
    None if _mwd_env.strip().lower() in ("", "none") else float(_mwd_env)
)

# MBH phentax configuration (stft_tof choices: 21/33/44 higher modes,
# fit-seeded t(f) root finder, dense equispaced grid for pyResponseTDI).
MBH_HIGHER_MODES = (21, 33, 44)
MBH_PHENOM_TOL = 1e-12     # phentax root-finding tolerance
MBH_START_FREQ = 7e-5      # Hz; waveform-generation start frequency
MBH_RESPONSE_ORDER = 30    # Lagrange interpolation order (pyResponseTDI)
MBH_BUFFER_TIME = 15_000.0  # s; response edge buffer

# TDI-on-the-fly MBH waveform-window margin (seconds). The phentax window
# (dur_s) is sized from the DATA span (Tobs) + this margin so it is
# source-INDEPENDENT and built ONCE: a merger may sit anywhere in the data
# window, and the per-source merger time enters ONLY as the call-time
# ``t_merge`` argument (the on-the-fly response places the merger at
# ``MBH_WAVEFORM_T0 + t_plunge``). The extra margin gives the same spline
# headroom the validated debug recipe used.
MBH_TDIONFLY_MARGIN = 6.0 * 86400.0

# Epoch the merger times (t_plunge) are referenced to. Mojito catalogue
# coalescence times are relative to the mojito reference epoch; the
# synthetic injections are relative to the synthetic data start.
MBH_WAVEFORM_T0 = (
    MOJITO_REFERENCE_TIME if DATA_PROCESSOR == "mojito" else SYNTHETIC_T_START
)

# Engine-level (NWALKERS / NTEMPS env-overridable for smoke tests).
RANDOM_SEED = 103209
NWALKERS = int(os.environ.get("NWALKERS", 6))
NTEMPS = int(os.environ.get("NTEMPS", 3))
WINDOW_TAPER_DURATION = 0.0  # rectangular window

# Output
FILE_STORE_DIR = "./gf_output/"
BASE_FILE_NAME = "full_year_combined_run"


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

# Leaf counts for each branch (drives nleaves_max).
N_MBH_INJECTIONS = len(MOJITO_SOURCE_IDS["MBHB"])
N_EMRI_INJECTIONS = len(MOJITO_SOURCE_IDS["EMRI"])
N_SOBBH_INJECTIONS = len(MOJITO_SOURCE_IDS["SOBHB"])

if N_MBH_INJECTIONS + N_EMRI_INJECTIONS + N_SOBBH_INJECTIONS < 1:
    raise ValueError(
        "MOJITO_SOURCE_IDS must inject at least 1 source total across "
        "MBHB / EMRI / SOBHB (all three are currently empty)."
    )


# ============================================================
# *** Domain ***
# ============================================================
MIN_FREQ = 1e-4   # Hz; analysis band lower edge
MAX_FREQ = 2.5e-2  # Hz; analysis band upper edge

DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=MIN_FREQ,
    max_freq=MAX_FREQ,
    min_time=20 * WAVELET_DURATION,
    max_time=(NT - 20) * WAVELET_DURATION,
)


# ============================================================
# *** Annual modulation for the galactic-foreground noise ***
# ============================================================
def annual_amplitude_envelope(t_arr: np.ndarray) -> np.ndarray:
    """Per-sample amplitude envelope ``1 + A*cos(2*pi*t/yr + phi0)``."""
    return 1.0 + ANNUAL_AMP * np.cos(
        2.0 * np.pi * np.asarray(t_arr) / YRSID_SI + ANNUAL_PHASE0
    )


def annual_modulation_callable(t_arr):
    """``(nch, nch, Nt)`` modulation matrix for GalacticForeground.

    Uses the standard isotropic-foreground per-element pattern (diag = 1,
    off-diag = -1/2) with an overall annual amplitude envelope applied
    uniformly across elements. The envelope is the square of the TD
    amplitude envelope because the modulation acts on the covariance
    (power), not the realization amplitude.
    """
    t_arr = np.asarray(t_arr)
    Nt = t_arr.shape[0]
    base = np.array([
        [ 1.0, -0.5, -0.5],
        [-0.5,  1.0, -0.5],
        [-0.5, -0.5,  1.0],
    ])
    env = annual_amplitude_envelope(t_arr) ** 2  # power-domain envelope
    return base[:, :, None] * env[None, None, :]


class AnnualModulatedGalacticForeground(GalacticForeground):
    """:class:`GalacticForeground` with the annual modulation pre-bound."""

    def __init__(
        self,
        foreground_params: Optional[Sequence[float]] = None,
        tdi_generation: int = TDI_GEN,
    ):
        # FOREGROUND_PARAMS=None -> use the FittedHyperbolicTangent default
        # (no per-parameter tuple required; the class reads Tobs at call
        # time). Pass a placeholder param vector that the fitted class
        # ignores so the SeparableComponent contract is satisfied.
        if foreground_params is None:
            foreground_params = (TOBS,)
            stochastic_fn = FittedHyperbolicTangentGalacticForeground
        else:
            stochastic_fn = FittedHyperbolicTangentGalacticForeground
        super().__init__(
            foreground_params=foreground_params,
            modulation=annual_modulation_callable,
            tdi_generation=tdi_generation,
            stochastic_fn=stochastic_fn,
        )


# ============================================================
# *** Synthetic noise + foreground TD generators ***
# ============================================================
# ``_generate_correlated_fd_noise`` -> stock
# ``lisatools.sensitivity.generate_correlated_instrument_noise_td`` (bit-identical
# fixed-seed realization; was duplicated verbatim across settings files).


def _generate_foreground_fd_only_covariance(
    Nf_rfft: int,
    df: float,
    Tobs: float,
    foreground_params: Optional[Sequence[float]],
    tdi_generation: int,
) -> np.ndarray:
    """Build a 3x3xNf_rfft FD covariance for the galactic foreground only
    (no instrument), using the same path as the model-side
    :class:`GalacticForeground` so the synthetic realization matches the
    likelihood model.
    """
    fd_settings = FDSettings(N=Nf_rfft, df=df, force_backend="cpu")
    fg_params = (
        foreground_params
        if foreground_params is not None
        else (Tobs,)
    )
    fg_component = GalacticForeground(
        foreground_params=fg_params,
        modulation=None,  # stationary base — we apply the annual envelope in TD
        tdi_generation=tdi_generation,
        stochastic_fn=FittedHyperbolicTangentGalacticForeground,
    )
    sens = CompositeSensitivityMatrix(fd_settings, [fg_component])
    return np.asarray(sens.sens_mat)  # (3, 3, Nf_rfft)


def _generate_modulated_foreground_td(
    N: int,
    dt: float,
    Tobs: float,
    foreground_params: Optional[Sequence[float]],
    tdi_generation: int,
    seed: int,
) -> np.ndarray:
    """Sample a 3-channel TD foreground realization and apply the annual
    amplitude envelope per-sample.

    Workflow: build the stationary 3x3 FD covariance, per-frequency
    Cholesky, draw complex Gaussian, IRFFT to TD, then multiply each
    channel by ``sqrt(annual_amplitude_envelope(t))`` so the realization
    is non-stationary in the same way the model covariance is.
    """
    Nf_rfft = N // 2 + 1
    df = 1.0 / (N * dt)
    cov = _generate_foreground_fd_only_covariance(
        Nf_rfft, df, Tobs, foreground_params, tdi_generation
    )

    rng = np.random.default_rng(seed)
    norm = 0.5 * (1.0 / df) ** 0.5
    z = rng.normal(0, norm, (3, Nf_rfft)) + 1j * rng.normal(0, norm, (3, Nf_rfft))

    n_fd = np.zeros_like(z)
    eye = np.eye(3) * 1e-60
    for k in range(Nf_rfft):
        C_k = cov[..., k] + eye
        try:
            L = np.linalg.cholesky(C_k)
        except np.linalg.LinAlgError:
            diag = np.maximum(np.diag(cov[..., k]), 0.0)
            n_fd[:, k] = np.sqrt(diag) * z[:, k]
            continue
        n_fd[:, k] = L @ z[:, k]

    n_td = (np.fft.irfft(n_fd, n=N, axis=-1) / dt).astype(np.float64)

    # Apply per-sample annual envelope (amplitude domain).
    t_arr = np.arange(N) * dt  # mojito processor adds its own t0 separately
    env = np.sqrt(annual_amplitude_envelope(t_arr))
    n_td *= env[None, :]
    return n_td


# ============================================================
# *** Per-class synthetic injection arrays (10 sources each) ***
# ============================================================
def _make_emri_injections(n: int) -> np.ndarray:
    """Return ``(n, 14)`` EMRI waveform-basis injection vectors.

    Intrinsic params shared (so a tight per-leaf prior is feasible);
    sky / phase / distance vary per source.
    """
    base = _SINGLE_EMRI_INJECTION.copy()
    if n == 0:
        return np.zeros((0, base.size), dtype=base.dtype)
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        row = base.copy()
        row[6]  = 1.0 + 2.0 * rng.uniform()           # dist (Gpc)
        row[7]  = rng.uniform(0.05, np.pi - 0.05)     # qS
        row[8]  = rng.uniform(0.0, 2.0 * np.pi)       # phiS
        row[9]  = rng.uniform(0.05, np.pi - 0.05)     # qK
        row[10] = rng.uniform(0.0, 2.0 * np.pi)       # phiK
        row[11] = rng.uniform(0.0, 2.0 * np.pi)       # Phi_phi0
        rows.append(row)
    return np.stack(rows, axis=0)


def _make_sobbh_injections(n: int) -> np.ndarray:
    """Return ``(n, 11)`` SOBBH waveform-basis injection vectors."""
    base = _SINGLE_SOBBH_INJECTION.copy()
    if n == 0:
        return np.zeros((0, base.size), dtype=base.dtype)
    rng = np.random.default_rng(22)
    rows = []
    for i in range(n):
        row = base.copy()
        row[4]  = 1.0 + 1.0 * rng.uniform()           # dist
        row[5]  = rng.uniform(0.05, np.pi - 0.05)     # inc
        row[7]  = rng.uniform(0.0, 2.0 * np.pi)       # lam
        row[8]  = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)  # beta
        row[9]  = rng.uniform(0.0, np.pi)             # psi
        row[10] = rng.uniform(0.0, 2.0 * np.pi)       # phi0
        rows.append(row)
    return np.stack(rows, axis=0)


# MBH sampling-basis -> waveform-basis transform: the stock forward +
# inverse container (direct ICRS) from erebor. Sampling basis matches
# ``mbh_catalogue_to_sampling_basis``: ``(logM, Q, s1z, s2z, dist [Gpc],
# phi_ref, cos_iota, psi, alpha, sin_delta, t_plunge)`` with
# ``Q = m1/m2 >= 1``; the positional output is the
# ``PhenomTHMTDIWaveform`` call order ``(m1, m2, s1z, s2z, dist [Mpc],
# phi_ref, inc, psi, ra, dec, merger_time)`` — sky / polarization pass
# through unchanged in ICRS (the run frame; orbits loaded with
# ``frame='icrs'``).
MBH_TRANSFORM = make_mbh_transform_container()


def _make_mbh_injections(n: int, tobs: float) -> np.ndarray:
    """Return ``(n, 11)`` MBH sampling-basis injection vectors.

    Basis: ``(logM, Q, s1z, s2z, dist [Gpc], phi_ref, cos_iota, psi,
    alpha, sin_delta, t_plunge)`` — the MBH sampling basis (transformed
    to the ``PhenomTHMTDIWaveform`` call order via ``MBH_TRANSFORM``).
    Masses and spins come from the shared phentax baseline; sky / phase /
    distance / merger time vary per source. Synthetic mode uses the
    stock analytic (ecliptic-frame) orbits, so the "ICRS" sky slots are
    simply interpreted in the orbits' native frame — self-consistent
    between injection and template.
    """
    base = _SINGLE_MBH_PHENTAX_INJECTION.copy()  # (m1, m2, s1z, s2z, ...)
    if n == 0:
        return np.zeros((0, 11), dtype=float)
    m1, m2 = float(base[0]), float(base[1])
    rng = np.random.default_rng(33)
    rows = []
    for i in range(n):
        rows.append(np.array([
            np.log(m1 + m2),                               # logM
            m1 / m2,                                       # Q (>= 1)
            base[2],                                       # s1z
            base[3],                                       # s2z
            10.0 + 15.0 * rng.uniform(),                   # dist (Gpc)
            rng.uniform(0.0, 2.0 * np.pi),                 # phi_ref
            np.cos(rng.uniform(0.1, np.pi - 0.1)),         # cos_iota
            rng.uniform(0.0, np.pi),                       # psi
            rng.uniform(0.0, 2.0 * np.pi),                 # alpha
            np.sin(rng.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1)),  # sin_delta
            # Spread merger times across the interior of the observation
            # (relative to MBH_WAVEFORM_T0 = the data start in synthetic mode).
            (0.2 + 0.6 * (i + 1) / (n + 1)) * tobs,        # t_plunge
        ]))
    return np.stack(rows, axis=0)


# ---- Cached MBH PhenomTHMTDIWaveform (shared: signal_gen registration,
# ----   synthetic injection stream, and the recipe's PE move) ----
_MBH_PHENOM_GEN_CACHE = {}


def _get_mbh_phenom_wave_gen(
    *,
    data_td_settings: TDSettings,
    waveform_t0: float,
    orbits=None,
    output_domain_settings=None,
    tukey_alpha: float = 0.0,
    force_backend: str = "cpu",
):
    """Build (and cache) the MBH ``PhenomTHMTDIWaveform``.

    One instance is shared between the engine-side residual rebuild
    (``signal_gen``), the synthetic injection stream, and the PE move so
    the (slow) phentax setup runs once per configuration.
    """
    key = (
        id(data_td_settings), waveform_t0, id(orbits),
        id(output_domain_settings), tukey_alpha, force_backend,
    )
    if key in _MBH_PHENOM_GEN_CACHE:
        return _MBH_PHENOM_GEN_CACHE[key]

    from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

    gen = PhenomTHMTDIWaveform(
        waveform_kwargs=dict(
            higher_modes=list(MBH_HIGHER_MODES),
            include_negative_modes=True,  # negative m modes by symmetry
            t_low_fit=True,  # fit-seeded start time for the t(f) root finder
            coarse_grain=False,  # pyResponseTDI needs equispaced time arrays
            atol=MBH_PHENOM_TOL,
            rtol=MBH_PHENOM_TOL,
        ),
        # Waveform-generation window only (phentax ``T``), not the data span.
        Tobs=(MBH_WAVEFORM_DURATION if MBH_WAVEFORM_DURATION is not None else TOBS),
        start_freq=MBH_START_FREQ,
        use_reference_time=True,
        waveform_t0=waveform_t0,
        data_td_settings=data_td_settings,
        tdi_generation=TDI_GEN_STR,
        tdi_channels=TDI_CHAN,
        sampling_frequency=1.0 / DT,
        orbits=orbits,
        order=MBH_RESPONSE_ORDER,
        tukey_alpha=tukey_alpha,
        stft_dt=None,
        freq_min=MIN_FREQ,
        freq_max=MAX_FREQ,
        fft_batch_size=2,
        buffer_time=MBH_BUFFER_TIME,
        # WDM run target — communicated by settings object (sprint rule).
        # None on the injection path (TD output only).
        output_domain_settings=output_domain_settings,
        force_backend=force_backend,
    )
    _MBH_PHENOM_GEN_CACHE[key] = gen
    return gen


EMRI_INJECTIONS_FULL_BASIS = _make_emri_injections(N_EMRI_INJECTIONS)
SOBBH_INJECTIONS_FULL_BASIS = _make_sobbh_injections(N_SOBBH_INJECTIONS)
# Sampling basis (logM, Q, ..., alpha, sin_delta, t_plunge); transformed to
# the waveform basis via MBH_TRANSFORM where needed.
MBH_INJECTIONS_SAMPLING_BASIS = _make_mbh_injections(N_MBH_INJECTIONS, TOBS)


# ============================================================
# *** Data processors ***
# ============================================================
def _build_synthetic_source_streams(
    Tobs: float,
    dt: float,
    t_start: float,
    target_N: int,
    nchannels: int,
    force_backend: str,
    emri_injections: np.ndarray,
    sobbh_injections: np.ndarray,
    mbh_injections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-class TD signal sums via the cached response wrappers."""
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=force_backend)
    zero = np.zeros((nchannels, target_N), dtype=np.float64)
    # Stock full-grid placement target (right-pad / clip onto the data grid).
    grid = TDSettings(N=target_N, dt=dt, t0=t_start, force_backend="cpu")

    emri_td = zero.copy()
    if emri_injections.shape[0] > 0:
        emri_wave_gen = get_emri_response_wrapper(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config, tdi_chan=TDI_CHAN,
            role="injection", force_backend=force_backend,
        )
        for ii, params in enumerate(emri_injections):
            print(f"EMRI inject signal {ii + 1} of {len(emri_injections)} [start]")

            sig = np.asarray(emri_wave_gen(*params))
            emri_td += np.asarray(
                place_td_signal_on_grid(np.atleast_2d(sig)[:nchannels], grid).arr
            )
            print(f"EMRI inject signal {ii + 1} of {len(emri_injections)} [end]")


    sobbh_td = zero.copy()
    if sobbh_injections.shape[0] > 0:
        sobbh_wave_gen = get_sobbh_response_wrapper(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config, tdi_chan=TDI_CHAN,
            role="injection", force_backend=force_backend,
            # Same f_low-epoch convention as the template path: in mojito mode
            # f_low is defined at the fixed catalogue epoch (decoupled from the
            # data-window start); synthetic mode (this builder's only caller)
            # has no separate epoch, so ``None`` -> f_low at the window start.
            reference_time=(
                MOJITO_REFERENCE_TIME if DATA_PROCESSOR == "mojito" else None
            ),
        )
        for ii, params in enumerate(sobbh_injections):
            print(f"SOBBH inject signal {ii + 1} of {len(sobbh_injections)} [start]")
            sig = np.asarray(sobbh_wave_gen(*params))
            sobbh_td += np.asarray(
                place_td_signal_on_grid(np.atleast_2d(sig)[:nchannels], grid).arr
            )
            print(f"SOBBH inject signal {ii + 1} of {len(sobbh_injections)} [end]")

    mbh_td = zero.copy()
    if mbh_injections.shape[0] > 0:
        # PhenomTHMTDIWaveform path (stft_tof structure): legacy
        # pyResponseTDI response on the dense phentax grid, then stock
        # full-grid placement. Same generator class as the PE templates so
        # the engine-side residual rebuild nulls the injections exactly.
        mbh_wave_gen = _get_mbh_phenom_wave_gen(
            data_td_settings=grid,
            waveform_t0=t_start,
            orbits=None,  # stock analytic orbits (synthetic mode)
            output_domain_settings=None,  # TD output only here
            force_backend=force_backend,
        )
        for ii, params in enumerate(mbh_injections):
            print(f"MBHB inject signal {ii + 1} of {len(mbh_injections)} [start]")
            params_in = MBH_TRANSFORM.both_transforms(
                np.asarray(params, dtype=float)
            )
            times, channels = mbh_wave_gen.compute_tdi_channels(*params_in)
            mbh_td += np.asarray(
                place_td_signal_on_grid(channels, grid, times=times).arr
            )[:nchannels]
            print(f"MBHB inject signal {ii + 1} of {len(mbh_injections)} [end]")
    return emri_td, sobbh_td, mbh_td


class L1ProcessingStepWithSyntheticNoise(L1ProcessingStep):
    """Mojito L1 source loader + synthetic FD noise + modulated foreground.

    Loads MBHB / EMRI / SOBHB source TD signals from a mojito L1 folder
    (no NOISE, no GB / VGB), then adds:
      1. Synthetic FD-correlated instrument noise drawn from the
         ``(NOISE_SOMS_D, NOISE_SA_A)`` covariance.
      2. A galactic-foreground TD realization with the annual amplitude
         envelope applied per-sample.

    The mojito catalog populated by the base loader on ``self.catalogue``
    is preserved for downstream factories.
    """

    def __init__(
        self,
        L1_folder: str,
        source_ids: dict,
        orbits_class=None,
        orbits_kwargs: Optional[dict] = None,
        verbose: bool = True,
        do_plots: bool = False,
        Tobs: float = None,
        window_start_offset: float = 0.0,
    ):
        # Source types are MBHB / EMRI / SOBHB; drop any class whose
        # source_ids list is empty (mojito's L1DataLoader raises on
        # missing IDs). Instrument noise + galactic foreground come from
        # the synthetic generators below, not from mojito.
        source_types = [
            t for t in ["MBHB", "EMRI", "SOBHB"] if source_ids.get(t)
        ]
        if orbits_class is None:
            from lisatools.detector import L1Orbits
            orbits_class = L1Orbits

        super().__init__(
            L1_folder=L1_folder,
            source_types=source_types,
            source_ids=source_ids,
            orbits_class=orbits_class,
            orbits_kwargs=orbits_kwargs,
            verbose=verbose,
            do_plots=do_plots,
            Tobs=Tobs,
            window_start_offset=window_start_offset,
        )
        # super().__init__ already called load_data() and stored .data,
        # .times, .fs, .orbits, .catalogue.

        # Add synthetic FD instrument noise + modulated foreground.
        # Stock full-grid placement (right-pad / clip onto the data grid).
        N = int(round(self.T / self.dt))
        nch = self.data.shape[0]
        grid = TDSettings(
            N=N, dt=self.dt, t0=float(self.times[0]), force_backend="cpu"
        )
        combined = np.asarray(
            place_td_signal_on_grid(self.data[:nch], grid).arr
        )
        if ADD_INSTRUMENT_NOISE:
            noise_td = generate_correlated_instrument_noise_td(
                N=N, dt=self.dt,
                Soms_d=NOISE_SOMS_D, Sa_a=NOISE_SA_A,
                tdi_generation=TDI_GEN, seed=NOISE_SEED,
                model_name="full_year_noise_model",
            )
            combined = combined + np.asarray(
                place_td_signal_on_grid(noise_td[:nch], grid).arr
            )
        if ADD_GALACTIC_FOREGROUND:
            fg_td = _generate_modulated_foreground_td(
                N=N, dt=self.dt, Tobs=self.T,
                foreground_params=FOREGROUND_PARAMS,
                tdi_generation=TDI_GEN, seed=FOREGROUND_SEED,
            )
            combined = combined + np.asarray(
                place_td_signal_on_grid(fg_td[:nch], grid).arr
            )
        self.data = combined


class SyntheticDataProcessor(BaseProcessingStep):
    """All-synthetic processor used for testing without a mojito L1 folder.

    Builds per-class source TD signals via the cached response wrappers
    and sums the same instrument-noise + modulated-foreground TD streams
    as :class:`L1ProcessingStepWithSyntheticNoise`. Exposes a
    ``catalogue`` dict in the same shape so downstream factories can
    read leaf counts uniformly.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        emri_injection_params_full_basis: np.ndarray,
        sobbh_injection_params_full_basis: np.ndarray,
        mbh_injection_params_sampling_basis: np.ndarray,
        nchannels: int = NCHANNELS,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        target_N = int(round(Tobs / dt))
        emri = np.atleast_2d(emri_injection_params_full_basis)
        sobbh = np.atleast_2d(sobbh_injection_params_full_basis)
        # MBH rows arrive in the sampling basis; _build_synthetic_source_streams
        # applies MBH_TRANSFORM per row.
        mbh = np.atleast_2d(mbh_injection_params_sampling_basis)

        emri_td, sobbh_td, mbh_td = _build_synthetic_source_streams(
            Tobs=Tobs, dt=dt, t_start=t_start, target_N=target_N,
            nchannels=nchannels, force_backend=force_backend,
            emri_injections=emri,
            sobbh_injections=sobbh,
            mbh_injections=mbh,
        )

        combined = emri_td + sobbh_td + mbh_td
        if ADD_INSTRUMENT_NOISE:
            noise_td = generate_correlated_instrument_noise_td(
                N=target_N, dt=dt,
                Soms_d=NOISE_SOMS_D, Sa_a=NOISE_SA_A,
                tdi_generation=TDI_GEN, seed=NOISE_SEED,
                model_name="full_year_noise_model",
            )[:nchannels]
            combined = combined + noise_td
        if ADD_GALACTIC_FOREGROUND:
            fg_td = _generate_modulated_foreground_td(
                N=target_N, dt=dt, Tobs=Tobs,
                foreground_params=FOREGROUND_PARAMS,
                tdi_generation=TDI_GEN, seed=FOREGROUND_SEED,
            )[:nchannels]
            combined = combined + fg_td
        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots,
        )
        self.orbits = None
        self.tdi_chan = TDI_CHAN
        # Synthetic-mode catalogue mirrors the mojito shape so downstream
        # factories can use a uniform `len(general_info.catalogue[<CLASS>])`.
        self.catalogue = {
            "MBHB": {sid: {} for sid in MOJITO_SOURCE_IDS["MBHB"]},
            "EMRI": {sid: {} for sid in MOJITO_SOURCE_IDS["EMRI"]},
            "SOBHB": {sid: {} for sid in MOJITO_SOURCE_IDS["SOBHB"]},
        }
        # Stash truths so plot/diagnostic code can find them.
        self.emri_injection_params_full_basis = emri
        self.sobbh_injection_params_full_basis = sobbh
        self.mbh_injection_params_sampling_basis = mbh


# ============================================================
# *** Per-branch erebor setups (priors + injection arrays) ***
# ============================================================
def _force_backend_for_branch() -> str:
    return GPU_BACKEND if gpu_available else "cpu"


def get_emri_multi_erebor_settings(general_set: GeneralSetup) -> Optional[EMRISetup]:
    """EMRI setup with ``nleaves_max = N_EMRI_INJECTIONS``.

    Stretch inner moves. Full-range priors: every parameter uses the
    default wide priors inside :class:`EMRISetup` (all ``*_lims`` are
    ``None``). Injections come from the mojito catalogue when
    ``DATA_PROCESSOR == "mojito"``. Returns ``None`` when no EMRI
    leaves are injected so the caller can skip the branch.
    """
    if N_EMRI_INJECTIONS == 0:
        return None
    force_backend = _force_backend_for_branch()
    initialize_kwargs_emri = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs=dict(force_backend=force_backend),
        response_kwargs=dict(
            t0=general_set.data_t0,
            order=40,
            tdi=TDI_GEN_STR,
            tdi_chan=TDI_CHAN,
            force_backend=force_backend,
            remove_garbage="zero",
        ),
    )

    if DATA_PROCESSOR == "mojito":
        # SPECIAL EMRI frame (validated 2026-06-19, scripts/sobbh/
        # emri_frame_convert_check.py): the FEW intrinsic h+/hx are built
        # from ECLIPTIC-POLAR sky angles (qS = pi/2 - ecliptic-latitude,
        # phiS = ecliptic-longitude — converted from the catalogue ICRS
        # RA/Dec) together with the RAW FILE spin angles (qK, phiK read
        # straight from the catalogue, NOT ecliptic-converted — converting
        # the spin is what produced the spurious 1.49x amplitude). The
        # ResponseWrapper is built is_ecliptic_latitude=False and called
        # convert_to_ra_dec=True (see _EMRISpecialFrameWrap in
        # lisatools.sources.emri.response) so the sky goes ecliptic -> ICRS
        # for the frame="icrs" orbits, while the spin stays in the FEW frame.
        # This reproduces the mojito EMRI to mm ~ 4e-5. Row construction lives
        # in the stock lisatools.sources.emri.emri_catalogue_to_waveform_basis.
        _emri_cat = general_set.data_processor.catalogue["EMRI"]
        emri_injections_full_basis = np.asarray(
            [
                emri_catalogue_to_waveform_basis(_emri_cat[i])
                for i in sorted(_emri_cat.keys())
            ]
        )
    else:
        emri_injections_full_basis = EMRI_INJECTIONS_FULL_BASIS

    tc = make_emri_transform_container([emri_injections_full_basis[0, 5], emri_injections_full_basis[0, -2]])
    injection_sampling_per_leaf = tc.both_inverse_transforms(emri_injections_full_basis)

    # Full-range priors: passing None for every *_lims keeps the wide
    # defaults built inside EMRISetup.setup_priors (dist in Gpc).
    logm1_lims = None
    m2_lims = None
    a_lims = None
    p0_lims = None
    e0_lims = None

    fill_values = np.array([
        emri_injections_full_basis[0, 5],   # xI0
        emri_injections_full_basis[0, 12],  # Phi_theta0
    ])

    emri_settings = EMRISettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        fill_values=fill_values,
        logm1_lims=logm1_lims,
        m2_lims=m2_lims,
        a_lims=a_lims,
        p0_lims=p0_lims,
        e0_lims=e0_lims,
        injection=injection_sampling_per_leaf,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_emri,
        waveform_kwargs=dict(),
        info_matrix_gen=None,
        inner_moves=[(StretchMove(), 1.0)],
        nleaves_max=N_EMRI_INJECTIONS,
        nleaves_min=N_EMRI_INJECTIONS,
        ndim=12,
    )
    emri_setup = EMRISetup(emri_settings)

    # Engine-side template generation (the converted ``get_templates``
    # process): one sampling-basis row in -> one run-domain template out.
    # The (slow) response build is deferred to first call via the cache.
    def _emri_signal_gen(*params, **kwargs):
        wave_wrap = _get_emri_wave_wrap(general_set)
        params_in = emri_setup.transform.both_transforms(
            np.asarray(params, dtype=float)
        )
        return wave_wrap(*params_in, **kwargs)

    emri_setup.signal_gen = _emri_signal_gen
    return emri_setup


def get_sobbh_multi_erebor_settings(general_set: GeneralSetup) -> Optional[SOBBHSetup]:
    """SOBBH setup with ``nleaves_max = N_SOBBH_INJECTIONS``.

    Uses ``StretchMove`` as the inner move (mirrors the EMRI / MBH
    branches). Full-range priors (all ``*_lims`` are ``None`` so the
    :class:`SOBBHSetup` defaults apply); injections come from the mojito
    catalogue when ``DATA_PROCESSOR == "mojito"``. Returns ``None`` when
    no SOBBH leaves are injected.
    """
    if N_SOBBH_INJECTIONS == 0:
        return None
    force_backend = _force_backend_for_branch()
    initialize_kwargs_sobbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        sobbh_waveform_args=("SOBBHWaveform",),
        sobbh_waveform_kwargs=dict(force_backend=force_backend),
        response_kwargs=dict(
            t0=general_set.data_t0,
            order=40,
            tdi=TDI_GEN_STR,
            tdi_chan=TDI_CHAN,
            force_backend=force_backend,
            remove_garbage="zero",
        ),
    )

    if DATA_PROCESSOR == "mojito":
        # NOTE: catalogue field names follow the MBHB schema (same MT
        # processing); GW22FrequencySSBFrame -> f_low is a best guess —
        # correct against the actual sobhb_cat_mojito_lite_processed_MT.hdf5
        # keys. ICRS run frame (2026-06 reversion): sky + polarization are
        # read raw in ICRS (RA in the lam slot, Dec in the beta slot,
        # psi ICRS) and the orbits are loaded with frame='icrs' to match —
        # no rotation here.
        _sobbh_cat = general_set.data_processor.catalogue["SOBHB"]
        sobbh_injections_full_basis = np.asarray([
            [
                _sobbh_cat[i]["PrimaryMassSSBFrame"],       # m1
                _sobbh_cat[i]["SecondaryMassSSBFrame"],     # m2
                _sobbh_cat[i]["PrimarySpinCompZ"],          # s1
                _sobbh_cat[i]["SecondarySpinCompZ"],        # s2
                _sobbh_cat[i]["LuminosityDistance"] / 1e3,  # dist (Mpc -> Gpc)
                _sobbh_cat[i]["InclinationAngle"],          # inc
                _sobbh_cat[i]["GW22FrequencySSBFrame"],     # f_low
                _sobbh_cat[i]["RightAscension"] % (2 * np.pi),  # RA (lam slot)
                _sobbh_cat[i]["Declination"],               # Dec (beta slot)
                _sobbh_cat[i]["PolarisationAngle"] % np.pi, # psi (ICRS)
                _sobbh_cat[i]["TrueAnomaly"], # phi0
            ]
            for i in sorted(_sobbh_cat.keys())
        ])
    else:
        sobbh_injections_full_basis = SOBBH_INJECTIONS_FULL_BASIS

    injection_sampling_per_leaf = np.stack(
        [sobbh_full_to_sampling(row) for row in sobbh_injections_full_basis],
        axis=0,
    )

    # Full-range priors: passing None for every *_lims keeps the wide
    # defaults built inside SOBBHSetup.setup_priors (dist in Gpc).
    logm1_lims = None
    logm2_lims = None
    s1_lims = None
    s2_lims = None
    f_low_lims = None

    sobbh_settings = SOBBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        fill_values=np.array([]),
        logm1_lims=logm1_lims,
        logm2_lims=logm2_lims,
        s1_lims=s1_lims,
        s2_lims=s2_lims,
        f_low_lims=f_low_lims,
        injection=injection_sampling_per_leaf,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_sobbh,
        waveform_kwargs=dict(),
        info_matrix_gen=None,
        inner_moves=[(StretchMove(), 1.0)],
        nleaves_max=N_SOBBH_INJECTIONS,
        nleaves_min=N_SOBBH_INJECTIONS,
        ndim=11,
    )
    sobbh_setup = SOBBHSetup(sobbh_settings)

    # Engine-side template generation (the converted ``get_templates``
    # process): one sampling-basis row in -> one run-domain template out.
    def _sobbh_signal_gen(*params, **kwargs):
        wave_wrap = _get_sobbh_wave_wrap(general_set)
        params_in = sobbh_setup.transform.both_transforms(
            np.asarray(params, dtype=float)
        )
        return wave_wrap(*params_in, **kwargs)

    sobbh_setup.signal_gen = _sobbh_signal_gen
    return sobbh_setup


def _make_mbh_initialize_kwargs(general_set: GeneralSetup) -> dict:
    """``PhenomTHMTDIWaveform`` construction kwargs for the template path.

    Consumed by ``_get_mbh_phenom_wave_gen`` (keyword-for-keyword) so the
    engine-side ``signal_gen``, the recipe's PE move, and any restart all
    share one cached generator instance.
    """
    force_backend = _force_backend_for_branch()
    return dict(
        data_td_settings=general_set.data_td_settings,
        waveform_t0=MBH_WAVEFORM_T0,
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits,
        # WDM run target — communicated by settings object (sprint rule).
        output_domain_settings=general_set.domain_settings,
        # Same window treatment as the data side (rectangular when the
        # engine's taper duration is 0).
        tukey_alpha=general_set.window_alpha,
        force_backend=force_backend,
    )


def get_mbh_phenom_multi_erebor_settings(general_set: GeneralSetup) -> Optional[MBHSetup]:
    """MBH ``PhenomTHMTDIWaveform`` setup (stft_tof structure, WDM-adjusted).

    Waveform basis (11): ``(m1, m2, s1z, s2z, dist [Mpc], phi_ref, inc,
    psi, ra, dec, t_plunge)`` — the ``PhenomTHMTDIWaveform`` call order
    (sky in ICRS; the orbits are loaded with ``frame='icrs'`` in mojito
    mode). Sampling basis: ``(logM, Q, s1z, s2z, dist [Gpc], phi_ref,
    cos_iota, psi, alpha, sin_delta, t_plunge)`` — matches
    ``mbh_catalogue_to_sampling_basis``. The per-source output is the
    run's WDM domain via ``output_domain_settings``.

    Registers ``signal_gen`` on the returned setup so the engine's
    ``setup_acs(rebuild_residuals=True)`` builds/subtracts the MBH
    templates under the hood. Returns ``None`` when no MBH leaves are
    injected.
    """
    if N_MBH_INJECTIONS == 0:
        return None

    initialize_kwargs_mbh = _make_mbh_initialize_kwargs(general_set)

    if DATA_PROCESSOR == "mojito":
        # Direct-ICRS sampling rows (logM, Q, ..., psi_icrs, ra, sin_dec,
        # t_ssb) — no frame conversion (ICRS run frame; orbits loaded with
        # frame='icrs').
        _mbh_cat = general_set.data_processor.catalogue["MBHB"]
        injection_sampling_per_leaf = np.stack(
            [
                mbh_catalogue_to_sampling_basis(_mbh_cat[i])
                for i in sorted(_mbh_cat.keys())
            ],
            axis=0,
        )
    else:
        injection_sampling_per_leaf = MBH_INJECTIONS_SAMPLING_BASIS

    # t_plunge is sampled relative to MBH_WAVEFORM_T0 (the epoch the
    # catalogue/injection merger times are referenced to); the data span
    # starts at data_t0 = MBH_WAVEFORM_T0 + trim in mojito mode and at
    # MBH_WAVEFORM_T0 exactly in synthetic mode.
    _t_rel_min = general_set.data_t0 - MBH_WAVEFORM_T0
    priors_mbh = {
        "logM":      uniform_dist(np.log(1e5), np.log(1e8)),
        "Q":         log_uniform(1.0, 10.0),
        "s1z":       uniform_dist(-0.999999, 0.999999),
        "s2z":       uniform_dist(-0.999999, 0.999999),
        "dist":      uniform_dist(0.1, 150.0),  # Gpc
        "phi_ref":   uniform_dist(0.0, 2 * np.pi),
        "cos_iota":  uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        "psi":       uniform_dist(0.0, np.pi),
        "alpha":     uniform_dist(0.0, 2 * np.pi),
        "sin_delta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        "t_plunge":  uniform_dist(_t_rel_min, _t_rel_min + general_set.Tobs + 3600.0),
    }
    priors = {"mbh": ProbDistContainer(priors_mbh)}

    # TODO(post-merge): re-enable SkyMove hops once the move supports the
    # ICRS sampling basis (the existing implementation assumes an
    # SSB-ecliptic basis; sky-frame checker + ICRS SkyMove pending).
    inner_moves_mbh = [(StretchMove(), 1.0)]

    # Engine-side template generation (the converted ``get_templates``
    # process): one sampling-basis row in -> one WDM-domain template out.
    # The wave-gen build is deferred to first call (cached) so settings
    # construction stays cheap.
    def _mbh_signal_gen(*params, **kwargs):
        params_in = MBH_TRANSFORM.both_transforms(
            np.asarray(params, dtype=float)
        )
        if USE_TDIONFLY:
            return _get_mbh_tdionfly_wave_wrap(general_set)(*params_in, **kwargs)
        wave_gen = _get_mbh_phenom_wave_gen(**initialize_kwargs_mbh)
        return wave_gen.get_signals_for_residuals(*params_in, **kwargs)

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        injection=injection_sampling_per_leaf,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_mbh,
        waveform_kwargs=dict(),
        inner_moves=inner_moves_mbh,
        nleaves_max=N_MBH_INJECTIONS,
        nleaves_min=N_MBH_INJECTIONS,
        ndim=11,
        transform=MBH_TRANSFORM,
        priors=priors,
        periodic={"mbh": {"phi_ref": 2 * np.pi, "psi": np.pi, "alpha": 2 * np.pi}},
        log_dir=general_set.file_store_dir,
        signal_gen=_mbh_signal_gen,
    )
    mbh_setup = MBHSetup(mbh_settings)

    return mbh_setup


# ============================================================
# *** Per-branch move builders (runtime t_start) ***
# ============================================================
# Cached domain-wrapped template generators, shared between the
# engine-side ``signal_gen`` registrations and the PE moves so each
# branch builds its (slow) response machinery once.
_WAVE_WRAP_CACHE = {}


def _get_emri_wave_wrap(general_info, nchannels: int = NCHANNELS):
    """Build (and cache) the EMRI domain-wrapped template generator.

    The intrinsic FEW phases are referenced to the **catalogue reference
    epoch** (``MOJITO_REFERENCE_TIME``, the same for every source), NOT the
    trimmed data-window start ``data_t0`` — ``GenerateEMRIWaveform`` has no
    internal reference time, so its ``t0`` IS the phase reference. The
    response is therefore generated from REF over ``(N + offset_int)``
    samples; ``t0_shift_to_data`` carries the sub-sample remainder and
    :class:`EMRIWaveWrap` slices the integer ``offset_int`` off the front so
    the rest lands on the data grid (``t0 = data_t0``). Mirrors the validated
    SPECIAL recipe in scripts/sobbh/emri_frame_convert_check.py.
    """
    key = ("emri", id(general_info), nchannels)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = _force_backend_for_branch()
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=force_backend)

    dt = general_info.dt
    data_t0 = general_info.data_t0
    # Catalogue reference epoch (shared by all sources); falls back to the
    # data start only in synthetic mode (where REF == data_t0).
    ref = MOJITO_REFERENCE_TIME if DATA_PROCESSOR == "mojito" else data_t0
    off = data_t0 - ref
    offset_int = int(round(off / dt))
    t0_shift = off - offset_int * dt  # sub-sample remainder, |t0_shift| < dt
    out_N = int(round(general_info.Tobs / dt))
    # Generate offset_int extra samples so that, after the front slice, the
    # surviving window still spans the full data grid.
    resp_Tobs = (out_N + offset_int) * dt

    template_wave_gen = get_emri_response_wrapper(
        Tobs=resp_Tobs, dt=dt,
        t_start=ref,
        t0_shift_to_data=t0_shift,
        tdi_config=tdi_config, tdi_chan=TDI_CHAN,
        role="template", force_backend=force_backend,
        orbits=general_info.orbits,
    )
    # Engine-provided TD settings (carries the loader's data_t0 anchor).
    wrap = EMRIWaveWrap(
        template_wave_gen, general_info.data_td_settings,
        general_info.domain_settings,
        td_window=None, nchannels=nchannels, offset_int=offset_int,
    )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def _get_sobbh_wave_wrap(general_info, nchannels: int = NCHANNELS):
    """Build (and cache) the SOBBH domain-wrapped template generator.

    With ``USE_TDIONFLY`` (default) this is the validated
    :class:`SOBBHTDIonFly` path (WDM mm5 ~3.5e-7 vs the legacy 2.4e-5);
    otherwise it falls back to the legacy ``ResponseWrapper(SOBBHWaveform)``
    (which hardcodes ``flip_hx=True`` — the wrong handedness for mojito).
    Used by both the engine ``signal_gen`` and the PE move, so toggling
    here routes both.
    """
    key = ("sobbh", id(general_info), nchannels)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = _force_backend_for_branch()
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=force_backend)
    # f_low is defined at the fixed catalogue epoch, NOT the (trimmed)
    # data-window start ``data_t0``. Pass the epoch explicitly in mojito
    # mode so the PN inspiral evolves ``f_low`` forward to the window start
    # (mirrors the GB ``evolve_galactic_binary`` convention). In synthetic
    # mode there is no separate epoch, so leave it ``None`` -> f_low at the
    # window start.
    reference_time = (
        MOJITO_REFERENCE_TIME if DATA_PROCESSOR == "mojito" else None
    )

    if USE_TDIONFLY:
        gen = get_sobbh_tdionfly_gen(
            Tobs=general_info.Tobs, dt=general_info.dt,
            t_start=general_info.data_t0, tdi_config=tdi_config,
            reference_time=reference_time, orbits=general_info.orbits,
            force_backend=force_backend,
        )
        n = int(round(general_info.Tobs / general_info.dt))
        t_arr = np.arange(n) * general_info.dt + general_info.data_t0
        wrap = SOBBHTDIonFlyWaveWrap(
            gen, t_arr, general_info.data_td_settings,
            general_info.domain_settings, td_window=None, nchannels=nchannels,
        )
        _WAVE_WRAP_CACHE[key] = wrap
        return wrap

    template_wave_gen = get_sobbh_response_wrapper(
        Tobs=general_info.Tobs, dt=general_info.dt,
        t_start=general_info.data_t0,
        tdi_config=tdi_config, tdi_chan=TDI_CHAN,
        role="template", force_backend=force_backend,
        orbits=general_info.orbits,
        reference_time=reference_time,
    )
    # Engine-provided TD settings (carries the loader's data_t0 anchor).
    wrap = SOBBHWaveWrap(
        template_wave_gen, general_info.data_td_settings,
        general_info.domain_settings,
        td_window=None, nchannels=nchannels,
    )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def _get_mbh_tdionfly_wave_wrap(general_info, nchannels: int = NCHANNELS):
    """Build (and cache) the MBH TDI-on-the-fly domain-wrapped generator.

    The :class:`bbhx.mbhtdionfly.MBHTDIonFly` generator is built ONCE and is
    **source-independent**: its phentax waveform window (``dur_s``) is sized
    from the DATA span (``Tobs`` + :data:`MBH_TDIONFLY_MARGIN`) so a merger
    may sit anywhere in the window. The per-source merger time
    (``t_plunge``, the last waveform-basis slot) enters ONLY as the
    call-time ``t_merge`` argument — the on-the-fly response places the
    merger absolutely at ``MBH_WAVEFORM_T0 + t_plunge`` (``t0 =
    MBH_WAVEFORM_T0``), while the output grid / domain use the data-window
    ``data_t0``. Used by both the engine ``signal_gen`` and the PE move, so
    one instance serves every leaf and every sampler proposal.
    """
    key = ("mbh_tdionfly", id(general_info), nchannels)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    force_backend = _force_backend_for_branch()
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=force_backend)
    orbits = general_info.gpu_orbits if gpu_available else general_info.orbits
    n = int(round(general_info.Tobs / general_info.dt))
    t_arr = np.arange(n) * general_info.dt + general_info.data_t0
    # Source-independent waveform window: cover the whole data span (a
    # merger may sit anywhere in it) + margin, so the same generator serves
    # every t_plunge the sampler proposes (no per-call rebuild).
    dur_s = general_info.Tobs + MBH_TDIONFLY_MARGIN
    gen = get_mbh_tdionfly_gen(
        dt=general_info.dt, t_start=MBH_WAVEFORM_T0, dur_s=dur_s,
        tdi_config=tdi_config, orbits=orbits,
        waveform_duration=dur_s, force_backend=force_backend,
    )
    wrap = MBHTDIonFlyWaveWrap(
        gen, t_arr, general_info.data_td_settings,
        general_info.domain_settings, nchannels=nchannels,
    )
    _WAVE_WRAP_CACHE[key] = wrap
    return wrap


def _build_emri_move_runtime(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """EMRI move using the runtime ``general_info.data_t0`` for t_start.

    Builds the move only — the engine's ``setup_acs(rebuild_residuals=
    True)`` already subtracted the state's templates from the residuals
    via the registered ``signal_gen`` (no residual writes here).
    """
    general_info = curr.general_info
    emri_info = curr.source_info["emri"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    wave_gen = _get_emri_wave_wrap(general_info)

    # Stock single-source PE-move builder (betas_all / coords / move machinery).
    _, moves = EMRIMoveBuilder(wave_gen=wave_gen).build(None, curr, acs, priors, state)
    return moves[0]


def _build_mbh_move_runtime(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """MBH PE move.

    With ``USE_TDIONFLY`` (default) the MBH branch uses the same stretch
    ``ResidualAddOneRemoveOneMove`` pattern as EMRI / SOBBH, driven by the
    per-leaf :class:`_MBHTDIonFlyWaveWrap` (the TDI-on-the-fly generator
    has no special-move API). Otherwise it falls back to the stock
    ``build_mbh_moves_phenom`` builder around the cached
    ``PhenomTHMTDIWaveform``. Either way move construction only — the
    engine's ``setup_acs(rebuild_residuals=True)`` already subtracted the
    state's MBH templates via the registered ``signal_gen``.
    """
    mbh_info = curr.source_info["mbh"]

    if not USE_TDIONFLY:
        wave_gen = _get_mbh_phenom_wave_gen(**mbh_info.initialize_kwargs)
        _, move = build_mbh_moves_phenom(
            curr, acs, priors, state,
            wave_gen=wave_gen,
            subtract_initial=False,
        )
        return move

    general_info = curr.general_info
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    wave_gen = _get_mbh_tdionfly_wave_wrap(general_info)

    # Stock single-source PE-move builder. The tdionfly MBH path passes
    # ``waveform_kwargs`` as the likelihood kwargs (unlike the phenom
    # ``build_mbh_moves_phenom`` default of ``{}``), so pass it explicitly.
    _, moves = MBHMoveBuilder(
        wave_gen=wave_gen, waveform_like_kwargs=mbh_info.waveform_kwargs
    ).build(None, curr, acs, priors, state)
    return moves[0]


def _build_sobbh_move_runtime(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """SOBBH move using the runtime ``general_info.data_t0`` for t_start.

    Mirrors the EMRI / MBH stretch builders — move construction only; the
    engine's ``setup_acs(rebuild_residuals=True)`` already subtracted the
    state's templates via the registered ``signal_gen`` (no residual
    writes here).
    """
    general_info = curr.general_info
    sobbh_info = curr.source_info["sobbh"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    wave_gen = _get_sobbh_wave_wrap(general_info)

    # Stock single-source PE-move builder (betas_all / coords / move machinery).
    _, moves = SOBBHMoveBuilder(wave_gen=wave_gen).build(None, curr, acs, priors, state)
    return moves[0]


# ============================================================
# *** setup_recipe ***
# ============================================================
def setup_recipe(
    recipe: Recipe,
    engine_info,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """Three source branches — MBH (stretch), EMRI (stretch), SOBBH (stretch).

    Moves only: residual generation/subtraction already ran under the
    hood in the engine (``setup_acs(rebuild_residuals=True)`` consuming
    each branch's registered ``signal_gen``).
    """
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    # Only build moves for branches that were populated in source_info
    # (i.e. classes with >= 1 injected leaf).
    pe_moves = []
    if "mbh" in curr.source_info:
        pe_moves.append(_build_mbh_move_runtime(curr, acs, priors, state))
    if "emri" in curr.source_info:
        pe_moves.append(_build_emri_move_runtime(curr, acs, priors, state))
    if "sobbh" in curr.source_info:
        pe_moves.append(_build_sobbh_move_runtime(curr, acs, priors, state))
    gf_pe_move = GFCombineMove(
        moves=pe_moves, verbose=True, share_temperature_control=False,
    )
    gf_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(
        PERecipeStep(moves=[gf_pe_move]), name="full pe"
    )


# ============================================================
# *** General setup ***
# ============================================================
def _mbh_chop_window_offset() -> float:
    """Seconds from the L1 file start at which to begin a chopped MBH window.

    Only nonzero when ``CHOP_WINDOW`` and the active source is MBH: the MBH
    merger falls mid-mission, so the snippet must begin before it. The
    merger is placed at ``MERGER_FRAC`` of the window. The merger epoch
    (``TimeCoalescencePhenomTPHMSSBFrame``) is read directly from the
    mojito MBHB catalogue HDF5 for the active id; it is measured relative
    to ``MOJITO_REFERENCE_TIME``, which is ~the L1 file start, so the
    offset is ``max(0, t_plunge - MERGER_FRAC*TOBS)``. Default 0.0 keeps the
    full window (production).
    """
    if not (CHOP_WINDOW and ACTIVE_SOURCE == "MBHB"):
        return 0.0
    import glob

    mbh_id = int(MOJITO_SOURCE_IDS["MBHB"][0])
    cat_files = sorted(
        glob.glob(os.path.join(MOJITO_DATA_PATH, "catalogues", "mbhb_cat_*.hdf5"))
    )
    if not cat_files:
        logger.warning(
            "CHOP_WINDOW: no mbhb_cat_*.hdf5 under %s/catalogues; using offset 0.",
            MOJITO_DATA_PATH,
        )
        return 0.0
    with h5py.File(cat_files[0], "r") as f:
        # MBHB catalogue row == id (0-based IDs coincide with rows).
        t_plunge = float(f["Binaries"]["TimeCoalescencePhenomTPHMSSBFrame"][mbh_id])
    offset = max(0.0, t_plunge - MERGER_FRAC * TOBS)
    logger.info(
        "CHOP_WINDOW MBH id=%d: t_plunge=%.1f s -> window_start_offset=%.1f s "
        "(merger at ~%.0f%% of the %.1f-day window).",
        mbh_id, t_plunge, offset, 100 * MERGER_FRAC, TOBS / 86400.0,
    )
    return offset


def _select_data_processor():
    """Return ``(processor_class, processor_init_kwargs)`` per DATA_PROCESSOR."""
    if DATA_PROCESSOR == "mojito":
        from lisatools.detector import L1Orbits
        kwargs = dict(
            L1_folder=MOJITO_DATA_PATH,
            source_ids={k: list(v) for k, v in MOJITO_SOURCE_IDS.items()},
            orbits_class=L1Orbits,
            # icrs: the run frame is ICRS (2026-06 reversion) — the
            # catalogue sky/polarization parameters are sampled raw
            # (alpha/RA, sin_delta, psi ICRS) with no per-injection
            # conversion, so the orbits must be ICRS too for the response
            # to see consistent sky coordinates.
            orbits_kwargs=dict(
                force_backend=_force_backend_for_branch(),
                frame="icrs",
            ),
            verbose=True,
            do_plots=False,
            Tobs=TOBS,
            # Default 0.0 (full window from file start). Nonzero only for a
            # chopped single-MBH snippet (CHOP_WINDOW=1).
            window_start_offset=_mbh_chop_window_offset(),
        )
        return L1ProcessingStepWithSyntheticNoise, kwargs

    if DATA_PROCESSOR == "synthetic":
        kwargs = dict(
            Tobs=TOBS,
            dt=DT,
            t_start=SYNTHETIC_T_START,
            emri_injection_params_full_basis=EMRI_INJECTIONS_FULL_BASIS,
            sobbh_injection_params_full_basis=SOBBH_INJECTIONS_FULL_BASIS,
            mbh_injection_params_sampling_basis=MBH_INJECTIONS_SAMPLING_BASIS,
            nchannels=NCHANNELS,
            force_backend="cpu",
            verbose=True,
            do_plots=False,
        )
        return SyntheticDataProcessor, kwargs

    raise ValueError(
        f"DATA_PROCESSOR={DATA_PROCESSOR!r} not recognised. "
        f"Use 'mojito' or 'synthetic'."
    )


def get_general_erebor_settings() -> GeneralSetup:
    gpus = [0] if gpu_available else None
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    processor_class, processor_init_kwargs = _select_data_processor()

    # The synthesised data already spans exactly Tobs = Nf*Nt*dt; skip
    # the engine's default highpass + edge-trim + Tobs trim so the WDM
    # Nf*Nt shape stays exact. (Same as combined_*.py.)
    preprocess_kwargs = dict(
        highpass_kwargs=None,
        trim_kwargs=None,
        Tobs=None,
        normalize=False,
    )

    # Fixed instrument noise + annually-modulated foreground baked into
    # the sensitivity model. No PSD branch / no galfor branch.
    sensitivity_init_kwargs = dict(
        tdi_generation=TDI_GEN,
        extra_components=[
            InstrumentNoise(
                tdi_generation=TDI_GEN,
                model=LISAModel(
                    NOISE_SOMS_D ** 2, NOISE_SA_A ** 2,
                    DefaultOrbits(), "full_year_fixed_noise",
                ),
                fill_nans=0.0,
            ),
            AnnualModulatedGalacticForeground(
                foreground_params=FOREGROUND_PARAMS,
                tdi_generation=TDI_GEN,
            ),
        ],
    )

    general_settings = GeneralSettings(
        Tobs=TOBS,
        dt=DT,
        file_store_dir=FILE_STORE_DIR,
        base_file_name=BASE_FILE_NAME,
        main_file_key="testing",
        domain_settings=DOMAIN_CHOICE,
        random_seed=RANDOM_SEED,
        backup_iter=5,
        nwalkers=NWALKERS,
        ntemps=NTEMPS,
        window_type="tukey",
        window_taper_duration=WINDOW_TAPER_DURATION,
        gpu_backend=GPU_BACKEND,
        gpus=gpus,
        data_processor_class=processor_class,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
    )
    return GeneralSetup(general_settings)


def get_global_fit_settings(copy_settings_file: bool = False):
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

    # Skip branches with zero injected leaves; the per-branch setup
    # functions return ``None`` in that case.
    source_info = {}
    for key, setup in (
        ("mbh", get_mbh_phenom_multi_erebor_settings(general_setup)),
        ("emri", get_emri_multi_erebor_settings(general_setup)),
        ("sobbh", get_sobbh_multi_erebor_settings(general_setup)),
    ):
        if setup is not None:
            source_info[key] = setup

    gf_settings = GlobalFitSettings(
        source_info=source_info,
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )
    return CurrentInfoGlobalFit(gf_settings)


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print(
        f"Full-year combined settings constructed OK\n"
        f"  Tobs = {TOBS:.6e} s  (target {TOBS_TARGET:.6e} s)\n"
        f"  WDM grid Nf={NF}, Nt={NT}, wavelet_duration={WAVELET_DURATION:.1f} s\n"
        f"  Leaves: MBH={N_MBH_INJECTIONS}, EMRI={N_EMRI_INJECTIONS}, "
        f"SOBBH={N_SOBBH_INJECTIONS}\n"
        f"  Backend: {GPU_BACKEND} (GPU available={gpu_available})\n"
        f"  Data processor: {DATA_PROCESSOR}"
    )
