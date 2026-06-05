"""Full-year combined-source MCMC settings — EMRI + SOBBH + MBH (phentax).

Source classes
--------------
- MBH (phentax IMRPhenomTHM): StretchMove inside ResidualAddOneRemoveOneMove.
- EMRI                     : StretchMove inside ResidualAddOneRemoveOneMove.
- SOBBH                    : NUTSMove with JAX gradient via
                             SOBBHWDMHeterodyne.get_ll_grad_jax, inside
                             ResidualAddOneRemoveOneMove.

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
``gpu_available``. SOBBH NUTS uses ``force_backend="jax"`` for the
gradient path regardless of GPU availability (JAX picks the best device
at runtime).

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
from lisatools.domains import FDSettings, TDSettings, WDMSettings
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
    Setup,
)
from lisatools.globalfit.moves import GFCombineMove, ResidualAddOneRemoveOneMove
from lisatools.globalfit.preprocessing import BaseProcessingStep, L1ProcessingStep
from lisatools.globalfit.recipe import Recipe
from lisatools.globalfit.recipe_steps import PERecipeStep
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    EMRISettings,
    EMRISetup,
    MBHSettings,
    MBHSetup,
    SOBBHSettings,
    SOBBHSetup,
)
from lisatools.sensitivity import (
    CompositeSensitivityMatrix,
    GalacticForeground,
    InstrumentNoise,
)
from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground
from lisatools.utils.constants import YRSID_SI

from eryn.moves import NUTSMove, StretchMove
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist


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
    SOBBHWaveWrap,
    SOBBH_INJECTION_PARAMS_FULL_BASIS as _SINGLE_SOBBH_INJECTION,
    emri_full_to_sampling,
    get_emri_response_wrapper,
    get_sobbh_response_wrapper,
    sobbh_full_to_sampling,
)

from mbh_phentax_only_global_fit_settings import (
    INJECTION_PARAMS_FULL_BASIS as _SINGLE_MBH_PHENTAX_INJECTION,
    IMRPhenomTHMWaveform,
    MBHWaveWrap,
    _build_mbh_phentax_transform,
    get_mbh_phentax_response_wrapper,
    mbh_full_to_sampling,
)

# SOBBHWDMHeterodyne (chunked-het with JAX gradient path) was moved at
# Phase 2 (2026-06-02) from the sprint root to
# LISAanalysistools/scripts/gb_chunked_het/. It re-imports utilities from
# scripts/diagnostics/check_shortened_wdm.py via a bare import, so both
# subdirectories need to be on sys.path. Computed relative to this file
# so the script is portable across machines.
import os as _os_for_path
_THIS_DIR = _os_for_path.dirname(_os_for_path.abspath(__file__))
_LAT_SCRIPTS_DIR = _os_for_path.normpath(_os_for_path.join(_THIS_DIR, "..", "scripts"))
_GB_CHUNKED_HET_DIR = _os_for_path.join(_LAT_SCRIPTS_DIR, "gb_chunked_het")
_DIAGNOSTICS_DIR = _os_for_path.join(_LAT_SCRIPTS_DIR, "diagnostics")
for _p in (_DIAGNOSTICS_DIR, _GB_CHUNKED_HET_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gb_wdm_het import SOBBHWDMHeterodyne  # noqa: E402


# ============================================================
# *** Top-of-file knobs (the "surface" the user touches) ***
# ============================================================

# Target observation length and sample step.
TOBS_TARGET = YRSID_SI
DT = 2.5

# Half-day-ish wavelet-duration search window for adjust_to_even_bins.
WAVELET_DUR_BOUNDS = (40000.0, 48000.0)

# Data source selection.
#   "mojito"    — load source TD signals from a mojito L1 folder.
#   "synthetic" — build source TD signals locally for testing.
DATA_PROCESSOR = os.environ.get("DATA_PROCESSOR", "synthetic")
MOJITO_DATA_PATH = os.environ.get(
    "MOJITO_DATA_PATH",
    "/data/mojito_L1/",  # USER: set this to the actual L1 folder.
)

# Per-class source IDs (used by both modes — defines the leaf counts).
# Edit to match the mojito catalog you intend to load. Each class is
# capped at len(<list>) leaves per branch.
MOJITO_SOURCE_IDS = {
    "MBHB": list(range(10)),
    "EMRI": list(range(10)),
    "SOBHB": list(range(10)),
}

# Synthetic instrument noise (fixed, no PSD branch).
NOISE_SOMS_D = 15e-12
NOISE_SA_A = 3e-15
NOISE_SEED = 12345

# Galactic foreground (fixed, no galfor branch). Pass `None` to use the
# FittedHyperbolicTangentGalacticForeground default tabulated values.
FOREGROUND_PARAMS = None  # or e.g. (3.27e-44, 1e-2, 1.183, 941.0, 103.0)
FOREGROUND_SEED = 67890

# Annual modulation envelope: noise foreground TD amplitude (and WDM
# covariance modulation) gets a (1 + A*cos(2*pi*t/yr + phi0)) factor.
ANNUAL_AMP = 0.10
ANNUAL_PHASE0 = 0.0

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

# Synthetic-mode start time (mojito mode pulls t_start from the L1 file).
SYNTHETIC_T_START = 0.0

# Engine-level
RANDOM_SEED = 103209
NWALKERS = 6
NTEMPS = 3
WINDOW_TAPER_DURATION = 0.0  # rectangular window

# SOBBH NUTS knobs
NUTS_STEP_SIZE = 0.6
NUTS_MAX_TREE_DEPTH = 2

# SOBBHWDMHeterodyne chunking knobs (env-overridable to match the rest
# of the sprint).
CHUNKED_NT_SUB = int(os.environ.get("CHUNKED_NT_SUB", 256))
CHUNKED_N_PAD = int(os.environ.get("CHUNKED_N_PAD", 32))
CHUNKED_N_SPARSE = int(os.environ.get("CHUNKED_N_SPARSE", 256))
CHUNKED_N_CP_SIG = int(os.environ.get("CHUNKED_N_CP_SIG", 48))
CHUNKED_N_CP_ORBIT = int(os.environ.get("CHUNKED_N_CP_ORBIT", 32))
CHUNKED_JAX_CHUNK_ENV = os.environ.get("CHUNKED_JAX_CHUNK")
CHUNKED_JAX_CHUNK = int(CHUNKED_JAX_CHUNK_ENV) if CHUNKED_JAX_CHUNK_ENV else None

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


# ============================================================
# *** Domain ***
# ============================================================
DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=1e-4,
    max_freq=2.5e-2,
    min_time=20 * 3600.0,
    max_time=(NT - 20) * 3600.0,
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
def _generate_correlated_fd_noise(
    N: int,
    dt: float,
    Soms_d: float,
    Sa_a: float,
    tdi_generation: int,
    seed: int,
) -> np.ndarray:
    """Sample a 3-channel TD instrument-noise realization from the FD
    instrument-noise covariance built by :class:`InstrumentNoise`.

    Returns shape ``(3, N)`` float64.
    """
    Nf_rfft = N // 2 + 1
    df = 1.0 / (N * dt)
    fd_settings = FDSettings(N=Nf_rfft, df=df, force_backend="cpu")
    model = LISAModel(
        Soms_d ** 2, Sa_a ** 2, DefaultOrbits(), "full_year_noise_model",
    )
    instrument = InstrumentNoise(
        tdi_generation=tdi_generation, model=model, fill_nans=0.0
    )
    sens = CompositeSensitivityMatrix(fd_settings, [instrument])
    cov = np.asarray(sens.sens_mat)

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

    n_td = np.fft.irfft(n_fd, n=N, axis=-1) / dt
    return n_td.astype(np.float64)


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


def _make_mbh_injections(n: int, tobs: float) -> np.ndarray:
    """Return ``(n, 11)`` MBH-phentax waveform-basis injection vectors."""
    base = _SINGLE_MBH_PHENTAX_INJECTION.copy()
    rng = np.random.default_rng(33)
    rows = []
    for i in range(n):
        row = base.copy()
        row[4]  = 10.0 + 15.0 * rng.uniform()         # dist (Gpc)
        row[5]  = rng.uniform(0.0, 2.0 * np.pi)       # phi_ref
        row[6]  = rng.uniform(0.1, np.pi - 0.1)       # inc
        row[7]  = rng.uniform(0.0, 2.0 * np.pi)       # psi
        row[8]  = rng.uniform(0.0, 2.0 * np.pi)       # lam
        row[9]  = rng.uniform(-np.pi / 2 + 0.1, np.pi / 2 - 0.1)  # beta
        # Spread merger times across the interior of the observation.
        row[10] = (0.2 + 0.6 * (i + 1) / (n + 1)) * tobs
        rows.append(row)
    return np.stack(rows, axis=0)


EMRI_INJECTIONS_FULL_BASIS = _make_emri_injections(N_EMRI_INJECTIONS)
SOBBH_INJECTIONS_FULL_BASIS = _make_sobbh_injections(N_SOBBH_INJECTIONS)
MBH_INJECTIONS_FULL_BASIS = _make_mbh_injections(N_MBH_INJECTIONS, TOBS)


# ============================================================
# *** Data processors ***
# ============================================================
def _pad_or_clip(arr: np.ndarray, target_N: int) -> np.ndarray:
    if arr.shape[-1] < target_N:
        pad = target_N - arr.shape[-1]
        return np.pad(arr, ((0, 0), (0, pad)), mode="constant")
    if arr.shape[-1] > target_N:
        return arr[:, :target_N]
    return arr


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

    emri_td = zero.copy()
    if emri_injections.shape[0] > 0:
        emri_wave_gen = get_emri_response_wrapper(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config, tdi_chan=TDI_CHAN,
            role="injection", force_backend=force_backend,
        )
        for params in emri_injections:
            sig = np.asarray(emri_wave_gen(*params, convert_to_ra_dec=False))
            emri_td += _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)

    sobbh_td = zero.copy()
    if sobbh_injections.shape[0] > 0:
        sobbh_wave_gen = get_sobbh_response_wrapper(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config, tdi_chan=TDI_CHAN,
            role="injection", force_backend=force_backend,
        )
        for params in sobbh_injections:
            sig = np.asarray(sobbh_wave_gen(*params, convert_to_ra_dec=False))
            sobbh_td += _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)

    mbh_td = zero.copy()
    if mbh_injections.shape[0] > 0:
        mbh_wave_gen = get_mbh_phentax_response_wrapper(
            Tobs=Tobs, dt=dt, t_start=t_start,
            tdi_config=tdi_config, tdi_chan=TDI_CHAN,
            role="injection", force_backend=force_backend,
        )
        for params in mbh_injections:
            sig = np.asarray(mbh_wave_gen(*params, convert_to_ra_dec=False))
            mbh_td += _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)

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
    ):
        # We hardcode source_types to MBHB / EMRI / SOBHB only — instrument
        # noise + galactic foreground come from the synthetic generators
        # below, not from mojito.
        source_types = ["MBHB", "EMRI", "SOBHB"]
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
        )
        # super().__init__ already called load_data() and stored .data,
        # .times, .fs, .orbits, .catalogue.

        # Add synthetic FD instrument noise.
        N = int(round(self.T / self.dt))
        noise_td = _generate_correlated_fd_noise(
            N=N, dt=self.dt,
            Soms_d=NOISE_SOMS_D, Sa_a=NOISE_SA_A,
            tdi_generation=TDI_GEN, seed=NOISE_SEED,
        )
        fg_td = _generate_modulated_foreground_td(
            N=N, dt=self.dt, Tobs=self.T,
            foreground_params=FOREGROUND_PARAMS,
            tdi_generation=TDI_GEN, seed=FOREGROUND_SEED,
        )
        # mojito gives (nch, n_times); align lengths defensively.
        nch = self.data.shape[0]
        self.data = (
            self.data
            + _pad_or_clip(noise_td[:nch], N)
            + _pad_or_clip(fg_td[:nch], N)
        )


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
        mbh_injection_params_full_basis: np.ndarray,
        nchannels: int = NCHANNELS,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        target_N = int(round(Tobs / dt))
        emri = np.atleast_2d(emri_injection_params_full_basis)
        sobbh = np.atleast_2d(sobbh_injection_params_full_basis)
        mbh = np.atleast_2d(mbh_injection_params_full_basis)

        emri_td, sobbh_td, mbh_td = _build_synthetic_source_streams(
            Tobs=Tobs, dt=dt, t_start=t_start, target_N=target_N,
            nchannels=nchannels, force_backend=force_backend,
            emri_injections=emri,
            sobbh_injections=sobbh,
            mbh_injections=mbh,
        )

        noise_td = _generate_correlated_fd_noise(
            N=target_N, dt=dt,
            Soms_d=NOISE_SOMS_D, Sa_a=NOISE_SA_A,
            tdi_generation=TDI_GEN, seed=NOISE_SEED,
        )[:nchannels]
        fg_td = _generate_modulated_foreground_td(
            N=target_N, dt=dt, Tobs=Tobs,
            foreground_params=FOREGROUND_PARAMS,
            tdi_generation=TDI_GEN, seed=FOREGROUND_SEED,
        )[:nchannels]

        combined = emri_td + sobbh_td + mbh_td + noise_td + fg_td
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
        self.mbh_injection_params_full_basis = mbh


# ============================================================
# *** Per-branch erebor setups (priors + injection arrays) ***
# ============================================================
def _force_backend_for_branch() -> str:
    return GPU_BACKEND if gpu_available else "cpu"


def get_emri_multi_erebor_settings(general_set: GeneralSetup) -> EMRISetup:
    """EMRI setup with ``nleaves_max = N_EMRI_INJECTIONS``.

    Stretch inner moves. Tight per-leaf intrinsic priors derived from the
    shared intrinsic baseline; sky/phase/distance use the default wide
    priors inside :class:`EMRISetup`.
    """
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

    delta_prior = 1e-2
    injection_sampling_per_leaf = np.stack(
        [emri_full_to_sampling(row) for row in EMRI_INJECTIONS_FULL_BASIS],
        axis=0,
    )
    base = injection_sampling_per_leaf[0]
    logm1_lims = [(1 - delta_prior) * base[0], (1 + delta_prior) * base[0]]
    m2_lims = [(1 - delta_prior) * base[1], (1 + delta_prior) * base[1]]
    a_lims = [(1 - delta_prior) * base[2], min(0.999, (1 + delta_prior) * base[2])]
    p0_lims = [(1 - delta_prior) * base[3], (1 + delta_prior) * base[3]]
    e0_lims = [(1 - delta_prior) * base[4], (1 + delta_prior) * base[4]]

    fill_values = np.array([
        EMRI_INJECTIONS_FULL_BASIS[0, 5],   # xI0
        EMRI_INJECTIONS_FULL_BASIS[0, 12],  # Phi_theta0
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
    return EMRISetup(emri_settings)


def get_sobbh_multi_erebor_settings(general_set: GeneralSetup) -> SOBBHSetup:
    """SOBBH setup with ``nleaves_max = N_SOBBH_INJECTIONS``.

    ``inner_moves`` left as the default StretchMove here — the
    NUTS-with-JAX-gradient inner move is wired in :func:`setup_recipe`
    after the heterodyne object is built against the runtime orbits.
    """
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

    delta_prior = 1e-2
    injection_sampling_per_leaf = np.stack(
        [sobbh_full_to_sampling(row) for row in SOBBH_INJECTIONS_FULL_BASIS],
        axis=0,
    )
    base = injection_sampling_per_leaf[0]
    logm1_lims = [(1 - delta_prior) * base[0], (1 + delta_prior) * base[0]]
    logm2_lims = [(1 - delta_prior) * base[1], (1 + delta_prior) * base[1]]
    s1_lims = [
        max(-0.99, base[2] - delta_prior), min(0.99, base[2] + delta_prior)
    ]
    s2_lims = [
        max(-0.99, base[3] - delta_prior), min(0.99, base[3] + delta_prior)
    ]
    f_low_inj = base[6]
    f_low_lims = [(1 - delta_prior) * f_low_inj, (1 + delta_prior) * f_low_inj]

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
        # Placeholder — overridden in setup_recipe with the NUTS move.
        inner_moves=[(StretchMove(), 1.0)],
        nleaves_max=N_SOBBH_INJECTIONS,
        nleaves_min=N_SOBBH_INJECTIONS,
        ndim=11,
    )
    return SOBBHSetup(sobbh_settings)


def get_mbh_phentax_multi_erebor_settings(general_set: GeneralSetup) -> MBHSetup:
    """MBH-phentax setup with ``nleaves_max = N_MBH_INJECTIONS``.

    GPU-ready: ``force_backend`` flips with ``gpu_available`` (mirrors
    the EMRI / SOBBH pattern), so a cupy install + visible GPU drives
    the phentax response wrapper onto GPU automatically. If phentax
    itself isn't GPU-ready it will surface at first call.
    """
    force_backend = _force_backend_for_branch()
    initialize_kwargs_mbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        mbh_waveform="phentax.IMRPhenomTHM",
        mbh_waveform_kwargs=dict(higher_modes="all", force_backend=force_backend),
        response_kwargs=dict(
            t0=general_set.data_t0,
            order=40,
            tdi=TDI_GEN_STR,
            tdi_chan=TDI_CHAN,
            force_backend=force_backend,
            remove_garbage="zero",
        ),
    )

    delta_prior = 1e-2
    injection_sampling_per_leaf = np.stack(
        [mbh_full_to_sampling(row) for row in MBH_INJECTIONS_FULL_BASIS],
        axis=0,
    )
    base = injection_sampling_per_leaf[0]
    logM_inj = float(base[0])
    q_inj = float(base[1])
    s1z_inj = float(base[2])
    s2z_inj = float(base[3])

    dist_min = float(injection_sampling_per_leaf[:, 4].min())
    dist_max = float(injection_sampling_per_leaf[:, 4].max())
    t_plunge_min = float(injection_sampling_per_leaf[:, 10].min())
    t_plunge_max = float(injection_sampling_per_leaf[:, 10].max())

    priors_mbh = {
        "logM":     uniform_dist((1 - delta_prior) * logM_inj,
                                 (1 + delta_prior) * logM_inj),
        "q":        uniform_dist(max(0.01, (1 - delta_prior) * q_inj),
                                 min(0.999, (1 + delta_prior) * q_inj)),
        "s1z":      uniform_dist(max(-0.99, s1z_inj - delta_prior),
                                 min(0.99, s1z_inj + delta_prior)),
        "s2z":      uniform_dist(max(-0.99, s2z_inj - delta_prior),
                                 min(0.99, s2z_inj + delta_prior)),
        "dist":     uniform_dist((1 - delta_prior) * dist_min,
                                 (1 + delta_prior) * dist_max),
        "phi_ref":  uniform_dist(0.0, 2 * np.pi),
        "cos_iota": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        "psi":      uniform_dist(0.0, 2 * np.pi),
        "lam":      uniform_dist(0.0, 2 * np.pi),
        "sin_beta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        "t_plunge": uniform_dist(max(0.0, t_plunge_min - 60.0),
                                 t_plunge_max + 60.0),
    }
    priors = {"mbh": ProbDistContainer(priors_mbh)}

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        injection=injection_sampling_per_leaf,
        num_prop_repeats=2,
        initialize_kwargs=initialize_kwargs_mbh,
        waveform_kwargs=dict(),
        inner_moves=[(StretchMove(), 1.0)],
        nleaves_max=N_MBH_INJECTIONS,
        nleaves_min=N_MBH_INJECTIONS,
        ndim=11,
        transform=_build_mbh_phentax_transform(),
        priors=priors,
        periodic={"mbh": {"phi_ref": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}},
        log_dir=general_set.file_store_dir,
    )
    return MBHSetup(mbh_settings)


# ============================================================
# *** Per-branch move builders (runtime t_start) ***
# ============================================================
def _build_emri_move_runtime(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """EMRI move using the runtime ``general_info.data_t0`` for t_start."""
    general_info = curr.general_info
    emri_info = curr.source_info["emri"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    force_backend = _force_backend_for_branch()
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=force_backend)
    template_wave_gen = get_emri_response_wrapper(
        Tobs=general_info.Tobs, dt=general_info.dt,
        t_start=general_info.data_t0,
        tdi_config=tdi_config, tdi_chan=TDI_CHAN,
        role="template", force_backend=force_backend,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend=force_backend,
    )
    wave_gen = EMRIWaveWrap(
        template_wave_gen, td_settings, general_info.domain_settings,
        td_window=None, nchannels=acs.nchannels,
    )

    if np.any(emri_inds := state.branches_inds["emri"][0]):
        for leaf in range(emri_inds.shape[-1]):
            if not emri_inds[0, leaf]:
                continue
            assert np.all(emri_inds[:, leaf])
            inj_coords = state.branches_coords["emri"][0, :, leaf]
            inj_coords_in = emri_info.transform.both_transforms(inj_coords)
            for i in range(inj_coords.shape[0]):
                sig = wave_gen(*inj_coords_in[i], **emri_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()

    betas_all = np.tile(
        make_ladder(emri_info.ndim, ntemps=ntemps), (emri_info.nleaves_max, 1)
    )
    state.sub_states["emri"].betas_all = betas_all
    coords_shape = (ntemps, nwalkers, emri_info.nleaves_max, emri_info.ndim)
    move = ResidualAddOneRemoveOneMove(
        "emri", coords_shape, wave_gen,
        emri_info.waveform_kwargs.copy(),
        emri_info.waveform_kwargs.copy(),
        acs, emri_info.num_prop_repeats,
        emri_info.transform, priors,
        emri_info.inner_moves,
        Tmax=np.inf, betas_all=betas_all,
    )
    move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    return move


def _build_mbh_phentax_move_runtime(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """MBH-phentax move using runtime ``general_info.data_t0`` and the
    branch's force_backend (GPU when available)."""
    general_info = curr.general_info
    mbh_info = curr.source_info["mbh"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    force_backend = _force_backend_for_branch()
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=force_backend)
    template_wave_gen = get_mbh_phentax_response_wrapper(
        Tobs=general_info.Tobs, dt=general_info.dt,
        t_start=general_info.data_t0,
        tdi_config=tdi_config, tdi_chan=TDI_CHAN,
        role="template", force_backend=force_backend,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend=force_backend,
    )
    wave_gen = MBHWaveWrap(
        template_wave_gen, td_settings, general_info.domain_settings,
        td_window=None, nchannels=acs.nchannels,
    )

    if np.any(mbh_inds := state.branches_inds["mbh"][0]):
        for leaf in range(mbh_inds.shape[-1]):
            if not mbh_inds[0, leaf]:
                continue
            assert np.all(mbh_inds[:, leaf])
            inj_coords = state.branches_coords["mbh"][0, :, leaf]
            inj_coords_in = mbh_info.transform.both_transforms(inj_coords)
            for i in range(inj_coords.shape[0]):
                sig = wave_gen(*inj_coords_in[i], **mbh_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()

    betas_all = np.tile(
        make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1)
    )
    state.sub_states["mbh"].betas_all = betas_all
    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)
    move = ResidualAddOneRemoveOneMove(
        "mbh", coords_shape, wave_gen,
        mbh_info.waveform_kwargs.copy(),
        mbh_info.waveform_kwargs.copy(),
        acs, mbh_info.num_prop_repeats,
        mbh_info.transform, priors,
        mbh_info.inner_moves,
        Tmax=np.inf, betas_all=betas_all,
    )
    move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    return move


def _prior_half_widths_sobbh(sobbh_info) -> np.ndarray:
    """Per-parameter prior half-widths for the SOBBH sampling basis.

    Used as the ``scale`` argument for :class:`NUTSMove` so each
    coordinate's natural step is ``~scale``. For uniform priors this is
    half the interval width; for the sampling-basis ``cos_inc`` etc.
    coordinates the priors live on ``[-1, 1]`` so the width is 1.
    """
    prior_container = sobbh_info.priors["sobbh"]
    half = np.zeros(sobbh_info.ndim, dtype=float)
    for i, dist in enumerate(prior_container.priors_in.values()):
        lo = getattr(dist, "min_val", None)
        hi = getattr(dist, "max_val", None)
        if lo is not None and hi is not None:
            half[i] = 0.5 * (float(hi) - float(lo))
        else:
            half[i] = 1.0
    return half


def _build_sobbh_nuts_move(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """SOBBH move with NUTSMove driven by :meth:`SOBBHWDMHeterodyne.get_ll_grad_jax`.

    Mirrors the GB pattern in ``LISAanalysistools/examples/gb_jax_likelihood_grad.py``:

    1. Build a :class:`SOBBHWDMHeterodyne` instance with
       ``force_backend="jax"`` so ``get_ll_grad_jax`` is available.
    2. Wrap the JAX-side gradient as ``grad_log_like_eryn(theta_batch)``
       returning a NumPy array shaped ``(num_walkers, ndim)`` for NUTS.
    3. Instantiate :class:`NUTSMove` with the gradient callable, prior
       half-widths as the per-parameter step scale, and a small
       ``max_tree_depth`` (2 — 4 leapfrog steps per proposal) for cost
       control.
    4. Hand the NUTSMove to :class:`ResidualAddOneRemoveOneMove` as the
       per-leaf inner move so the global-fit residual swap logic
       composes with the NUTS step.

    KNOWN RISK: NUTSMove's ``grad_log_like_fn`` must see the per-leaf
    residual that ``ResidualAddOneRemoveOneMove`` maintains. The
    SOBBHWDMHeterodyne reads the current residual from ``acs`` at call
    time (via its bound ``data_d`` / ``invC`` view), so the closure stays
    correct as the outer move iterates leaves — *provided* acs is
    updated in place by the outer move. If this assumption breaks,
    surface it with a wrapper that re-reads ``acs.data_residual`` per
    propose() call.
    """
    import jax
    import jax.numpy as jnp

    general_info = curr.general_info
    sobbh_info = curr.source_info["sobbh"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps
    domain = general_info.domain_settings
    if not isinstance(domain, WDMSettings):
        raise NotImplementedError(
            "SOBBH NUTSMove requires the WDM basis; "
            f"got {type(domain).__name__}."
        )

    # Cached template generator for the residual pre-injection (same as
    # the stretch path — the NUTS move proposes new theta, the residual
    # update path still rebuilds h(theta) via this wrapper).
    force_backend = _force_backend_for_branch()
    tdi_config = TDIConfig(TDI_GEN_STR, force_backend=force_backend)
    template_wave_gen = get_sobbh_response_wrapper(
        Tobs=general_info.Tobs, dt=general_info.dt,
        t_start=general_info.data_t0,
        tdi_config=tdi_config, tdi_chan=TDI_CHAN,
        role="template", force_backend=force_backend,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend=force_backend,
    )
    wave_gen = SOBBHWaveWrap(
        template_wave_gen, td_settings, general_info.domain_settings,
        td_window=None, nchannels=acs.nchannels,
    )

    # JAX-backed chunked heterodyne for the gradient path.
    sobbh_het_jax = SOBBHWDMHeterodyne(
        Nf=domain.Nf, Nt=domain.Nt, dt=general_info.dt,
        T_full=general_info.Tobs, t_ref_full=general_info.data_t0,
        Nt_sub=CHUNKED_NT_SUB,
        n_pad=CHUNKED_N_PAD,
        N_sparse=CHUNKED_N_SPARSE,
        nchannels=acs.nchannels,
        force_backend="jax",
        tdi_gen=TDI_GEN_STR,
        orbits=general_info.gpu_orbits,
        t_obs_start=float(general_info.data_t0),
        N_cp_sig=CHUNKED_N_CP_SIG,
        N_cp_orbit=CHUNKED_N_CP_ORBIT,
        jax_chunk=CHUNKED_JAX_CHUNK,
    )

    def _residual_views():
        """Snapshot of (data_d, invC) for the current per-leaf residual.

        Pulled from acs at NUTS call time so the closure tracks the
        outer ResidualAddOneRemoveOneMove's residual updates. The
        SOBBHWDMHeterodyne API takes these as JAX arrays.
        """
        data_d = jnp.asarray(np.asarray(acs.data_residual))
        invC = jnp.asarray(np.asarray(acs.invC))
        return data_d, invC

    def log_like_eryn(theta_batch):
        theta_jax = jnp.asarray(theta_batch, dtype=jnp.float64)
        data_d, invC = _residual_views()
        d_h, h_h = sobbh_het_jax.get_ll_jax(
            theta_jax, data_d, invC, chunk=CHUNKED_JAX_CHUNK,
        )
        # GW likelihood: log L = <d|h> - 0.5 <h|h>
        return np.asarray(d_h - 0.5 * h_h)

    def grad_log_like_eryn(theta_batch):
        theta_jax = jnp.asarray(theta_batch, dtype=jnp.float64)
        data_d, invC = _residual_views()
        grad = sobbh_het_jax.get_ll_grad_jax(
            theta_jax, data_d, invC, chunk=CHUNKED_JAX_CHUNK,
        )
        return np.asarray(grad)

    nuts_scale = _prior_half_widths_sobbh(sobbh_info)
    nuts = NUTSMove(
        grad_log_like_fn=grad_log_like_eryn,
        ndim=sobbh_info.ndim,
        scale=nuts_scale,
        step_size=NUTS_STEP_SIZE,
        max_tree_depth=NUTS_MAX_TREE_DEPTH,
    )
    sobbh_info.inner_moves = [(nuts, 1.0)]

    # Per-walker pre-injection (same residual setup as the stretch path).
    if np.any(sobbh_inds := state.branches_inds["sobbh"][0]):
        for leaf in range(sobbh_inds.shape[-1]):
            if not sobbh_inds[0, leaf]:
                continue
            assert np.all(sobbh_inds[:, leaf])
            inj_coords = state.branches_coords["sobbh"][0, :, leaf]
            inj_coords_in = sobbh_info.transform.both_transforms(inj_coords)
            for i in range(inj_coords.shape[0]):
                sig = wave_gen(*inj_coords_in[i], **sobbh_info.waveform_kwargs)
                acs.add_signal_to_residual([sig], data_index=np.array([i]))
                del sig
                gc.collect()

    betas_all = np.tile(
        make_ladder(sobbh_info.ndim, ntemps=ntemps), (sobbh_info.nleaves_max, 1)
    )
    state.sub_states["sobbh"].betas_all = betas_all
    coords_shape = (ntemps, nwalkers, sobbh_info.nleaves_max, sobbh_info.ndim)
    move = ResidualAddOneRemoveOneMove(
        "sobbh", coords_shape, wave_gen,
        sobbh_info.waveform_kwargs.copy(),
        sobbh_info.waveform_kwargs.copy(),
        acs, sobbh_info.num_prop_repeats,
        sobbh_info.transform, priors,
        sobbh_info.inner_moves,
        Tmax=np.inf, betas_all=betas_all,
    )
    move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    return move


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
    """Three source branches — MBH (stretch), EMRI (stretch), SOBBH (NUTS+JAX)."""
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    mbh_pe_move = _build_mbh_phentax_move_runtime(curr, acs, priors, state)
    emri_pe_move = _build_emri_move_runtime(curr, acs, priors, state)
    sobbh_pe_move = _build_sobbh_nuts_move(curr, acs, priors, state)

    pe_moves = [mbh_pe_move, emri_pe_move, sobbh_pe_move]
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
def _select_data_processor():
    """Return ``(processor_class, processor_init_kwargs)`` per DATA_PROCESSOR."""
    if DATA_PROCESSOR == "mojito":
        from lisatools.detector import L1Orbits
        kwargs = dict(
            L1_folder=MOJITO_DATA_PATH,
            source_ids={k: list(v) for k, v in MOJITO_SOURCE_IDS.items()},
            orbits_class=L1Orbits,
            orbits_kwargs=dict(
                force_backend=_force_backend_for_branch(),
                frame="ecliptic",
            ),
            verbose=True,
            do_plots=False,
        )
        return L1ProcessingStepWithSyntheticNoise, kwargs

    if DATA_PROCESSOR == "synthetic":
        kwargs = dict(
            Tobs=TOBS,
            dt=DT,
            t_start=SYNTHETIC_T_START,
            emri_injection_params_full_basis=EMRI_INJECTIONS_FULL_BASIS,
            sobbh_injection_params_full_basis=SOBBH_INJECTIONS_FULL_BASIS,
            mbh_injection_params_full_basis=MBH_INJECTIONS_FULL_BASIS,
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
        data_processor=processor_class,
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

    emri_setup = get_emri_multi_erebor_settings(general_setup)
    sobbh_setup = get_sobbh_multi_erebor_settings(general_setup)
    mbh_setup = get_mbh_phentax_multi_erebor_settings(general_setup)

    gf_settings = GlobalFitSettings(
        source_info={
            "mbh": mbh_setup,
            "emri": emri_setup,
            "sobbh": sobbh_setup,
        },
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
