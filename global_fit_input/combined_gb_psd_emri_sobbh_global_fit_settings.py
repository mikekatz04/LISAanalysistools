"""Combined smoke-test settings: GB + galfor + PSD + EMRI + SOBBH + MBH (phentax).

Reuses the pieces defined in ``global_fit_settings.py`` (the per-branch
``get_*_erebor_settings`` helpers, the shared EMRI / SOBBH waveform
wrappers, the EMRI/SOBBH per-walker pre-injection helpers, injection
constants) and the MBH-phentax wiring from
``mbh_phentax_only_global_fit_settings.py``. The unique bits here are:

* A custom :class:`SangriaPlusInjectionsProcessingStep` that loads Sangria
  data (mbhb removed, keeping the GB / galactic-foreground content) and
  sums synthetic FD instrument noise plus EMRI, SOBBH **and phentax MBH**
  TD waveforms on top.
* A local :func:`setup_recipe` that wires the GB + PSD + galfor + EMRI +
  SOBBH branches via the imports from ``global_fit_settings.py`` and
  swaps the MBH branch onto the phentax + :class:`ResidualAddOneRemoveOneMove`
  path (rather than the legacy ``BBHWaveformFD`` + ``MBHSpecialMove`` path
  in ``global_fit_settings._build_mbh_move``).

PSD is fit against the synthetic instrument noise injected here (Sangria
noise removed via ``remove_from_data=["noise", "mbhb"]``).
"""

import gc
import logging
import os
import shutil
from copy import deepcopy
from typing import Optional

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

from lisatools.detector import DefaultOrbits, EqualArmlengthOrbits, LISAModel
from lisatools.domains import FDSettings, WDMSettings
from lisatools.globalfit.engine import (
    GeneralSettings,
    GeneralSetup,
    GlobalFitSettings,
    RankInfo,
)
from lisatools.globalfit.preprocessing import (
    BaseProcessingStep,
    SangriaDataLoader,
)
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.sensitivity import CompositeSensitivityMatrix, InstrumentNoise
from lisatools.utils.constants import YRSID_SI

# Share the heavy lifting (per-branch settings + setup_recipe + waveform
# wrappers + injection constants) with the full global fit settings file.
# ``run_global.py`` loads this module via ``spec_from_file_location`` so its
# sibling directory isn't on ``sys.path``; add it here so the relative
# import resolves both ways (CLI and ``python -m`` / direct import).
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from global_fit_settings import (
    DT,
    INJECTION_PARAMS_FULL_BASIS as _SINGLE_EMRI_INJECTION,
    NF,
    NT,
    SOBBH_INJECTION_PARAMS_FULL_BASIS as _SINGLE_SOBBH_INJECTION,
    T_START,
    TOBS,
    _build_emri_move,
    _build_sobbh_move,
    emri_full_to_sampling,
    get_emri_response_wrapper,
    get_galfor_erebor_settings,
    get_gb_erebor_settings,
    get_psd_erebor_settings,
    get_sobbh_response_wrapper,
    sobbh_full_to_sampling,
)

# MBH-phentax wiring lives in the standalone smoke-test file; reuse the
# waveform adapter, response-wrapper builder, transform builder, and
# injection constants directly so this combined file stays in lock-step
# with the single-branch smoke test.
from mbh_phentax_only_global_fit_settings import (
    INJECTION_PARAMS_FULL_BASIS as _SINGLE_MBH_PHENTAX_INJECTION,
    IMRPhenomTHMWaveform,
    MBHWaveWrap,
    _build_mbh_phentax_transform,
    get_mbh_phentax_response_wrapper,
    mbh_full_to_sampling,
)

from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.domains import TDSettings
from lisatools.globalfit.engine import Setup
from lisatools.globalfit.moves import GFCombineMove, ResidualAddOneRemoveOneMove
from lisatools.globalfit.recipe import Recipe
from lisatools.globalfit.recipe import (
    PERecipeStep,
    build_gb_moves,
    build_psd_moves,
)
from lisatools.globalfit.stock.erebor import (
    EMRISettings,
    EMRISetup,
    MBHSettings,
    MBHSetup,
    SOBBHSettings,
    SOBBHSetup,
)
from eryn.moves import StretchMove
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist


# ============================================================
# *** Domain selection ***
# ============================================================
# Reuse the (Nf, Nt, dt) grid from ``global_fit_settings.py`` so this file
# is a drop-in slim variant: 3-month Tobs on Sangria's dt = 5 s.
DOMAIN_CHOICE = WDMSettings.make_factory(
    Nf=NF,
    Nt=NT,
    min_freq=1e-4,
    max_freq=2.5e-2,
    min_time=20 * 3600.0,
    max_time=(NT - 20) * 3600.0,
)


# ============================================================
# *** Multi-source EMRI / SOBBH injections ***
# ============================================================
# Three sources of each type. Same intrinsics as
# ``global_fit_settings.INJECTION_PARAMS_FULL_BASIS`` (so the EMRI / SOBBH
# *_lims priors still bracket every source), but with distinct sky angles
# / coalescence phases / distances. That lets ``ResidualAddOneRemoveOneMove``
# iterate ``nleaves_max = 3`` leaves per outer pass and verifies the
# multi-source residual update path end-to-end.

# EMRI: 14-param waveform basis. ``SAMPLE_FILL_INDICES = [5, 12]``
# (``xI0``, ``Phi_theta0``) are filled by the transform and stripped during
# sampling -- keep them identical across sources so the 12-D sampling
# coordinate has consistent fills.
def _make_emri_injections() -> np.ndarray:
    base = _SINGLE_EMRI_INJECTION.copy()
    rows = []
    # Sky position / orbit-plane orientation / coalescence phase shuffled
    # per source; distance varied to give different SNRs.
    sky_phase_specs = [
        # (qS,           phiS,   qK,            phiK,    Phi_phi0, dist_gpc)
        (np.pi / 3.0,    1.0,    np.pi / 4.0,   2.0,     0.0,      1.0),
        (np.pi / 5.0,    3.7,    np.pi / 3.2,   0.6,     1.7,      1.5),
        (2.0 * np.pi / 5.0, 5.5, 1.1,           4.3,     2.9,      2.0),
    ]
    for qS, phiS, qK, phiK, Phi_phi0, dist in sky_phase_specs:
        row = base.copy()
        row[6] = dist       # dist (Gpc)
        row[7] = qS         # qS
        row[8] = phiS       # phiS
        row[9] = qK         # qK
        row[10] = phiK      # phiK
        row[11] = Phi_phi0  # Phi_phi0
        rows.append(row)
    return np.stack(rows, axis=0)


# SOBBH: 11-param waveform basis. Vary sky, polarisation, coalescence
# phase and distance; keep intrinsic masses / spins / f_low fixed so a
# single tight ``logm1/logm2/s1/s2/f_low`` prior covers all three sources.
def _make_sobbh_injections() -> np.ndarray:
    base = _SINGLE_SOBBH_INJECTION.copy()
    rows = []
    sky_phase_specs = [
        # (dist, inc,           lam, beta,            psi,  phi0)
        (1.0,  np.pi / 3.0,     1.0, np.pi / 4.0,     0.3,  0.0),
        (1.4,  0.5 * np.pi / 2, 3.2, -np.pi / 5.0,    1.1,  1.6),
        (1.8,  0.8 * np.pi / 2, 5.6, np.pi / 6.0,     2.4,  3.1),
    ]
    for dist, inc, lam, beta, psi, phi0 in sky_phase_specs:
        row = base.copy()
        row[4] = dist
        row[5] = inc
        row[7] = lam
        row[8] = beta
        row[9] = psi
        row[10] = phi0
        rows.append(row)
    return np.stack(rows, axis=0)


EMRI_INJECTIONS_FULL_BASIS = _make_emri_injections()    # shape (3, 14)
SOBBH_INJECTIONS_FULL_BASIS = _make_sobbh_injections()  # shape (3, 11)
N_EMRI_INJECTIONS = EMRI_INJECTIONS_FULL_BASIS.shape[0]
N_SOBBH_INJECTIONS = SOBBH_INJECTIONS_FULL_BASIS.shape[0]


# MBH (phentax): 11-param waveform basis matching
# :meth:`IMRPhenomTHMWaveform.__call__`:
#   [m1, m2, s1z, s2z, dist, phi_ref, inc, psi, lam, beta, t_plunge]
# Three sources placed at distinct sky positions and merger times within
# the observation. Same intrinsic masses / spins so a single tight
# ``logM / q / s1z / s2z`` prior covers all three.
def _make_mbh_injections() -> np.ndarray:
    base = _SINGLE_MBH_PHENTAX_INJECTION.copy()
    rows = []
    sky_phase_specs = [
        # (dist, phi_ref, inc,           psi,  lam, beta,            t_plunge_frac)
        (10.0,  1.0,      np.pi / 3.0,   0.5,  1.5,  0.3,             0.35),
        (15.0,  2.5,      np.pi / 4.0,   1.2, -1.2,  0.6,             0.55),
        (20.0,  4.8,      0.7 * np.pi/2, 2.4,  3.0, -0.4,             0.75),
    ]
    for dist, phi_ref, inc, psi, lam, beta, t_frac in sky_phase_specs:
        row = base.copy()
        row[4] = dist
        row[5] = phi_ref
        row[6] = inc
        row[7] = psi
        row[8] = lam
        row[9] = beta
        row[10] = t_frac * TOBS
        rows.append(row)
    return np.stack(rows, axis=0)


MBH_INJECTIONS_FULL_BASIS = _make_mbh_injections()  # shape (3, 11)
N_MBH_INJECTIONS = MBH_INJECTIONS_FULL_BASIS.shape[0]


# ============================================================
# *** Combined data processor: Sangria GB + synthetic FD noise +
#     synthetic EMRI + synthetic SOBBH ***
# ============================================================
def _pad_or_clip(arr: np.ndarray, target_N: int) -> np.ndarray:
    """Pad with zeros on the right or clip to exactly ``target_N`` samples."""
    if arr.shape[-1] < target_N:
        pad = target_N - arr.shape[-1]
        return np.pad(arr, ((0, 0), (0, pad)), mode="constant")
    if arr.shape[-1] > target_N:
        return arr[:, :target_N]
    return arr


def _generate_correlated_fd_noise(
    N: int,
    dt: float,
    Soms_d: float,
    Sa_a: float,
    tdi_generation: int,
    seed: int,
) -> np.ndarray:
    """Sample a 3-channel TD noise realisation from the LISA instrument-noise
    (3, 3, Nf) frequency-domain covariance.

    Workflow: build the FD instrument-noise covariance for the requested
    ``(Soms_d, Sa_a)`` model via :class:`CompositeSensitivityMatrix`, do a
    per-frequency Cholesky factor, draw a complex Gaussian ``z`` and form
    ``L @ z``. Inverse-rFFT (LDC convention ``td = irfft(fd) / dt``) gives
    correlated TD noise that the downstream WDM transform consumes
    transparently.

    Args:
        N: Number of TD samples (must equal ``Nf * Nt`` downstream).
        dt: Sample spacing in seconds.
        Soms_d / Sa_a: Linear instrument-noise levels (matches the
            CompositeSensitivityBackend convention; the covariance uses the
            squared values internally).
        tdi_generation: 1 (TDI 1.5) or 2 (TDI 2.0).
        seed: RNG seed for reproducibility.

    Returns:
        ``(3, N)`` float array of TD noise samples.
    """
    Nf_rfft = N // 2 + 1
    df = 1.0 / (N * dt)
    fd_settings = FDSettings(N=Nf_rfft, df=df, force_backend="cpu")
    model = LISAModel(
        Soms_d ** 2, Sa_a ** 2, DefaultOrbits(), "smoke_test_noise_model"
    )
    sens = CompositeSensitivityMatrix(
        fd_settings,
        [InstrumentNoise(
            tdi_generation=tdi_generation, model=model, fill_nans=0.0
        )],
    )
    cov = np.asarray(sens.sens_mat)  # shape (3, 3, Nf_rfft)

    rng = np.random.default_rng(seed)
    # Scaling so that ``<|n(f)|^2>`` of the IFFT output reproduces the input
    # PSD. ``generate_noise_fd`` uses ``norm = 0.5 / sqrt(df)`` for the real /
    # imaginary parts; the LDC convention ``FD = dt * rfft(TD)`` then maps to
    # ``TD = irfft(FD) / dt``.
    norm = 0.5 * (1.0 / df) ** 0.5
    z = rng.normal(0, norm, (3, Nf_rfft)) + 1j * rng.normal(0, norm, (3, Nf_rfft))

    n_fd = np.zeros_like(z)
    eye = np.eye(3) * 1e-60  # tiny regulariser for the DC bin where cov ~ 0
    for k in range(Nf_rfft):
        C_k = cov[..., k] + eye
        try:
            L = np.linalg.cholesky(C_k)
        except np.linalg.LinAlgError:
            # f=0 / negative eigenvalues: fall back to diagonal sqrt.
            diag = np.maximum(np.diag(cov[..., k]), 0.0)
            n_fd[:, k] = np.sqrt(diag) * z[:, k]
            continue
        n_fd[:, k] = L @ z[:, k]

    n_td = np.fft.irfft(n_fd, n=N, axis=-1) / dt
    return n_td.astype(np.float64)


class SangriaPlusInjectionsProcessingStep(BaseProcessingStep):
    """Build smoke-test data from Sangria GBs + synthetic FD noise +
    synthetic EMRI + synthetic SOBBH + synthetic phentax MBH.

    Data composition:

    1. ``Sangria`` sky-only slice (``remove_from_data=["noise", "mbhb"]``)
       gives the GB / galactic-foreground content with no instrument-noise
       contribution.
    2. :func:`_generate_correlated_fd_noise` synthesises correlated XYZ
       noise from the chosen LISA instrument model in the FD domain and
       inverse-rFFTs it to TD.
    3. Synthetic EMRI, SOBBH and MBH (phentax) waveforms are produced by
       the shared response wrappers used elsewhere in the move
       pre-injection paths.

    All four streams are padded/clipped to exactly ``N = Tobs/dt`` samples
    so the downstream WDM transform's ``N = Nf*Nt`` shape is exact.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t_start: float,
        data_input_path: str,
        emri_injection_params_full_basis: np.ndarray,
        sobbh_injection_params_full_basis: np.ndarray,
        mbh_injection_params_full_basis: Optional[np.ndarray] = None,
        noise_Soms_d: float = 15e-12,
        noise_Sa_a: float = 3e-15,
        noise_seed: int = 12345,
        tdi_generation: int = 2,
        remove_from_data=("noise", "mbhb"),
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        force_backend: str = "cpu",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        target_N = int(round(Tobs / dt))
        emri_injections = np.atleast_2d(emri_injection_params_full_basis)
        sobbh_injections = np.atleast_2d(sobbh_injection_params_full_basis)
        mbh_injections = (
            np.atleast_2d(mbh_injection_params_full_basis)
            if mbh_injection_params_full_basis is not None
            else np.zeros((0, 11))
        )

        # --- Sangria GB sky-only slice ---------------------------------------
        sangria = SangriaDataLoader(
            data_input_path=data_input_path,
            remove_from_data=list(remove_from_data),
        )
        _, fs, sangria_data, _ = sangria.load_data()
        sangria_data = np.atleast_2d(sangria_data)[:nchannels]
        sangria_data = _pad_or_clip(sangria_data, target_N)

        # --- Synthetic FD instrument noise -----------------------------------
        noise_td = _generate_correlated_fd_noise(
            N=target_N,
            dt=dt,
            Soms_d=noise_Soms_d,
            Sa_a=noise_Sa_a,
            tdi_generation=tdi_generation,
            seed=noise_seed,
        )[:nchannels]

        tdi_config = TDIConfig("2nd generation", force_backend=force_backend)

        # --- EMRI injections (sum over multiple sources) ---------------------
        logger.info(f"Generating {len(emri_injections)} EMRI injection(s)...")
        emri_wave_gen = get_emri_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            force_backend=force_backend,
        )
        emri_td = np.zeros_like(sangria_data)
        for i, params in enumerate(emri_injections):
            logger.info(f"  EMRI {i+1}/{len(emri_injections)}: generating waveform...")
            # SPECIAL EMRI frame: let get_emri_response_wrapper's _EMRISpecialFrameWrap
            # force convert_to_ra_dec=True (ecl sky -> ICRS for frame="icrs" orbits) so the
            # synthetic injection matches the (also-converting) template. Do NOT pass
            # convert_to_ra_dec=False here -- that would desync injection vs template.
            sig = np.asarray(emri_wave_gen(*params))
            sig = _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)
            emri_td = emri_td + sig
            logger.info(f"  EMRI {i+1}/{len(emri_injections)}: done")

        # --- SOBBH injections (sum over multiple sources) --------------------
        logger.info(f"Generating {len(sobbh_injections)} SOBBH injection(s)...")
        sobbh_wave_gen = get_sobbh_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            force_backend=force_backend,
            # ``reference_time`` is the absolute epoch where ``f_low`` is
            # defined (decoupled from the data-window start). This is a
            # synthetic (Sangria) run with no separate catalogue epoch, so
            # ``None`` -> f_low at the window start. (mojito runs pass
            # MOJITO_REFERENCE_TIME here; see full_year_combined.)
            reference_time=None,
        )
        sobbh_td = np.zeros_like(sangria_data)
        for i, params in enumerate(sobbh_injections):
            logger.info(f"  SOBBH {i+1}/{len(sobbh_injections)}: generating waveform...")
            sig = np.asarray(sobbh_wave_gen(*params, convert_to_ra_dec=False))
            sig = _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)
            sobbh_td = sobbh_td + sig
            logger.info(f"  SOBBH {i+1}/{len(sobbh_injections)}: done")

        # --- MBH (phentax) injections (sum over multiple sources) -----------
        mbh_td = np.zeros_like(sangria_data)
        if mbh_injections.shape[0] > 0:
            mbh_wave_gen = get_mbh_phentax_response_wrapper(
                Tobs=Tobs,
                dt=dt,
                t_start=t_start,
                tdi_config=tdi_config,
                tdi_chan=tdi_chan,
                role="injection",
                force_backend=force_backend,
            )
            for i, params in enumerate(mbh_injections):
                sig = np.asarray(mbh_wave_gen(*params, convert_to_ra_dec=False))
                sig = _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)
                mbh_td = mbh_td + sig

        combined = sangria_data + noise_td + emri_td + sobbh_td + mbh_td

        times = np.arange(target_N) * dt + t_start
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.tdi_chan = tdi_chan
        self.emri_injection_params_full_basis = emri_injections
        self.sobbh_injection_params_full_basis = sobbh_injections
        self.mbh_injection_params_full_basis = mbh_injections


def get_general_erebor_settings() -> GeneralSetup:
    Tobs = TOBS
    dt = DT

    ldc_source_file = "/Users/mlkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5"
    base_file_name = "combined_smoke_test"
    file_store_dir = "./gf_output/"

    gpus = [0] if gpu_available else None
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    nwalkers = 6
    ntemps = 3

    # Rectangular window (alpha = 0) — matches the rest of the WDM smoke
    # configs.
    window_taper_duration = 0.0

    domain_settings = DOMAIN_CHOICE

    processor_init_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        t_start=T_START,
        data_input_path=ldc_source_file,
        emri_injection_params_full_basis=EMRI_INJECTIONS_FULL_BASIS,
        sobbh_injection_params_full_basis=SOBBH_INJECTIONS_FULL_BASIS,
        mbh_injection_params_full_basis=MBH_INJECTIONS_FULL_BASIS,
        remove_from_data=("noise", "mbhb"),
        tdi_chan="XYZ",
        nchannels=3,
        force_backend="cpu",
    )

    # The synthesised data already covers exactly Tobs = Nf*Nt*dt; skip
    # the engine's default highpass + edge-trim + Tobs trim so the WDM
    # ``Nf*Nt`` shape stays exact.
    preprocess_kwargs = dict(
        highpass_kwargs=None,
        trim_kwargs=None,
        Tobs=None,
        normalize=False,
    )

    # CompositeSensitivityBackend is the default; only ``tdi_generation`` is
    # consumed by it.
    sensitivity_init_kwargs = dict(tdi_generation=2)

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
        gpus=gpus,
        data_processor=SangriaPlusInjectionsProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
    )

    return GeneralSetup(general_settings)


def get_emri_multi_erebor_settings(general_set: GeneralSetup) -> EMRISetup:
    """Multi-leaf (3 source) EMRI :class:`EMRISetup`.

    Mirrors ``global_fit_settings.get_emri_erebor_settings`` but stacks
    the 3 sampling-basis injection vectors and bumps
    ``nleaves_max = nleaves_min = N_EMRI_INJECTIONS`` so
    :class:`ResidualAddOneRemoveOneMove` iterates one leaf per source.
    The tight ``logm1 / m2 / a / p0 / e0`` priors stay tied to the
    shared intrinsic params (the injections only differ in
    sky / phase / distance, all of which use the default wide priors).
    """
    initialize_kwargs_emri = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs=dict(force_backend="cpu" if not gpu_available else GPU_BACKEND),
        response_kwargs=dict(
            t0=T_START,
            order=40,
            tdi="2nd generation",
            tdi_chan="XYZ",
            force_backend="cpu" if not gpu_available else GPU_BACKEND,
            remove_garbage="zero",
        ),
    )

    delta_prior = 1e-2
    # All injections share the intrinsic params from the first row; the
    # tight ``*_lims`` priors bracket every source.
    injection_sampling_per_leaf = np.stack(
        [emri_full_to_sampling(row) for row in EMRI_INJECTIONS_FULL_BASIS],
        axis=0,
    )
    base_sampling = injection_sampling_per_leaf[0]

    logm1_lims = [(1 - delta_prior) * base_sampling[0], (1 + delta_prior) * base_sampling[0]]
    m2_lims = [(1 - delta_prior) * base_sampling[1], (1 + delta_prior) * base_sampling[1]]
    amax = min(0.999, (1 + delta_prior) * base_sampling[2])
    a_lims = [(1 - delta_prior) * base_sampling[2], amax]
    p0_lims = [(1 - delta_prior) * base_sampling[3], (1 + delta_prior) * base_sampling[3]]
    e0_lims = [(1 - delta_prior) * base_sampling[4], (1 + delta_prior) * base_sampling[4]]

    fill_values = np.array(
        [
            EMRI_INJECTIONS_FULL_BASIS[0, 5],   # xI0 — shared
            EMRI_INJECTIONS_FULL_BASIS[0, 12],  # Phi_theta0 — shared
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
    """Multi-leaf (3 source) SOBBH :class:`SOBBHSetup`.

    Same intrinsic params + per-leaf sky/phase/distance variations as the
    EMRI builder above. ``nleaves_max = nleaves_min = N_SOBBH_INJECTIONS``.
    """
    force_backend = "cpu" if not gpu_available else GPU_BACKEND
    initialize_kwargs_sobbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        sobbh_waveform_args=("SOBBHWaveform",),
        sobbh_waveform_kwargs=dict(force_backend=force_backend),
        response_kwargs=dict(
            t0=T_START,
            order=40,
            tdi="2nd generation",
            tdi_chan="XYZ",
            force_backend=force_backend,
            remove_garbage="zero",
        ),
    )

    delta_prior = 1e-2
    injection_sampling_per_leaf = np.stack(
        [sobbh_full_to_sampling(row) for row in SOBBH_INJECTIONS_FULL_BASIS],
        axis=0,
    )
    base_sampling = injection_sampling_per_leaf[0]

    logm1_lims = [(1 - delta_prior) * base_sampling[0], (1 + delta_prior) * base_sampling[0]]
    logm2_lims = [(1 - delta_prior) * base_sampling[1], (1 + delta_prior) * base_sampling[1]]
    s1_lims = [max(-0.99, base_sampling[2] - delta_prior), min(0.99, base_sampling[2] + delta_prior)]
    s2_lims = [max(-0.99, base_sampling[3] - delta_prior), min(0.99, base_sampling[3] + delta_prior)]
    f_low_inj = base_sampling[6]
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
        inner_moves=[(StretchMove(), 1.0)],
        nleaves_max=N_SOBBH_INJECTIONS,
        nleaves_min=N_SOBBH_INJECTIONS,
        ndim=11,
    )
    return SOBBHSetup(sobbh_settings)


def get_mbh_phentax_multi_erebor_settings(general_set: GeneralSetup) -> MBHSetup:
    """Multi-leaf (3 source) MBH :class:`MBHSetup` for the phentax wiring.

    Mirrors :func:`mbh_phentax_only_global_fit_settings.get_mbh_phentax_erebor_settings`
    but stacks the per-leaf injection vectors and bumps
    ``nleaves_max = nleaves_min = N_MBH_INJECTIONS`` so
    :class:`ResidualAddOneRemoveOneMove` iterates one leaf per source.
    Intrinsic params (logM, q, s1z, s2z) stay tied across leaves so a
    tight shared prior covers all three; only the per-leaf
    ``t_plunge`` prior is widened to bracket every injection.
    """
    force_backend = "cpu" if not gpu_available else GPU_BACKEND
    initialize_kwargs_mbh = dict(
        T=general_set.Tobs / YRSID_SI,
        dt=general_set.dt,
        mbh_waveform="phentax.IMRPhenomTHM",
        mbh_waveform_kwargs=dict(higher_modes="all", force_backend=force_backend),
        response_kwargs=dict(
            t0=T_START,
            order=40,
            tdi="2nd generation",
            tdi_chan="XYZ",
            force_backend=force_backend,
            remove_garbage="zero",
        ),
    )

    delta_prior = 1e-2
    injection_sampling_per_leaf = np.stack(
        [mbh_full_to_sampling(row) for row in MBH_INJECTIONS_FULL_BASIS],
        axis=0,
    )
    # All intrinsic params (positions 0..3) are identical across leaves;
    # tighten priors around the first leaf for those, and broaden the
    # extrinsic ones to span the union of leaves with a small pad.
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
        # Angles use the full prior on (0, 2pi) / (-1, 1): the 3 leaves
        # land at very different sky positions / phases so a tight per-
        # leaf prior would exclude two of them.
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


def _build_mbh_phentax_move(
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """MBH move using phentax + :class:`ResidualAddOneRemoveOneMove`.

    Mirrors :func:`global_fit_settings._build_emri_move` and
    :func:`global_fit_settings._build_sobbh_move` — the EMRI/SOBBH pattern
    rather than the legacy ``MBHSpecialMove`` + ``BBHWaveformFD`` path:

    1. Pull the cached :class:`ResponseWrapper` (built once at data-load
       time by the synthetic injection processor) and wrap it for the
       template path via :class:`MBHWaveWrap` so the output is a
       :class:`DomainBase` subclass that the ACA and the per-walker move
       both understand.
    2. Pre-subtract each walker's starting MBH waveform from its residual,
       one template at a time, so peak RAM stays at one waveform.
    3. Build and return the :class:`ResidualAddOneRemoveOneMove` (PE only,
       no inner search loop).
    """
    general_info = curr.general_info
    mbh_info = curr.source_info["mbh"]
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    force_backend = "cpu" if not gpu_available else GPU_BACKEND
    tdi_config = TDIConfig("2nd generation", force_backend=force_backend)
    template_wave_gen = get_mbh_phentax_response_wrapper(
        Tobs=general_info.Tobs,
        dt=general_info.dt,
        t_start=T_START,
        tdi_config=tdi_config,
        tdi_chan="XYZ",
        role="template",
        force_backend=force_backend,
    )
    td_settings = TDSettings(
        int(round(general_info.Tobs / general_info.dt)),
        general_info.dt,
        force_backend=force_backend,
    )
    wave_gen = MBHWaveWrap(
        template_wave_gen,
        td_settings,
        general_info.domain_settings,
        td_window=None,
        nchannels=acs.nchannels,
    )

    # Per-walker pre-injection — keeps peak RAM at one template's worth.
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
    mbh_pe_move = ResidualAddOneRemoveOneMove(
        "mbh",
        coords_shape,
        wave_gen,
        mbh_info.waveform_kwargs.copy(),
        mbh_info.waveform_kwargs.copy(),
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        mbh_info.inner_moves,
        Tmax=np.inf,
        betas_all=betas_all,
    )
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    return mbh_pe_move


def setup_recipe(
    recipe: Recipe,
    engine_info,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state,
):
    """Combined-smoke setup_recipe with phentax MBH wiring.

    Mirrors :func:`global_fit_settings.setup_recipe` for PSD / GB / EMRI /
    SOBBH but swaps the MBH branch onto the phentax +
    :class:`ResidualAddOneRemoveOneMove` path (:func:`_build_mbh_phentax_move`)
    instead of the legacy ``BBHWaveformFD`` + ``MBHSpecialMove`` builder.
    """
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])

    # Build the WDM-domain GB likelihood here (after the deepcopy in
    # ``CurrentInfoGlobalFit.__init__``) — the underlying C++ orbits wrap
    # is not picklable, so it must live outside the settings dataclass.
    # Logic copied verbatim from ``global_fit_settings.setup_recipe``.
    #
    # ``force_backend`` is the single point at which the chunked-het
    # compute backend (cpu / cuda12x / ... / jax) is selected for
    # ``gb_wdm_comp``. Per the sprint-wide "no runtime backend kwarg"
    # rule, the Buffer / WDMBandLikelihoodEngine no longer accept a
    # ``backend=`` kwarg at call time — get_ll, swap_ll, get_ll_grad
    # and hessian all dispatch through whichever backend this instance
    # was built with. To run the JAX-autograd grad/hessian path,
    # build a separate ``GBWDMHeterodyne(force_backend="jax")`` and
    # hand it to the Buffer.
    gb_info = curr.source_info["gb"]
    if (
        isinstance(general_info.domain_settings, WDMSettings)
        and gb_info.gb_wdm_comp is None
    ):
        # Chunked-het GB likelihood: the INSTALLED GBGPU class (removes a stale
        # hardcoded sys.path). GBWDMComputations reads Nf/Nt/dt/band/t0 from the domain.
        from gbgpu.gbcomps import GBWDMComputations

        _wdm = general_info.domain_settings
        _t_obs_start = float(getattr(general_info, "t_obs_start", 0.0))
        # The band WDMSettings domain (_wdm) carries Nf/Nt/dt + the active band +
        # t0; t_ref is the GB phase reference; tdi_config replaces tdi_gen.
        gb_info.gb_wdm_comp = GBWDMComputations(
            _wdm,
            t_ref=gb_info.t0,
            Nt_sub=int(os.environ.get("CHUNKED_NT_SUB", 256)),
            n_pad=int(os.environ.get("CHUNKED_N_PAD", 32)),
            N_sparse=int(os.environ.get("CHUNKED_N_SPARSE", 256)),
            N_cp_sig=int(os.environ.get("CHUNKED_N_CP_SIG", 48)),
            N_cp_orbit=int(os.environ.get("CHUNKED_N_CP_ORBIT", 32)),
            orbits=general_info.gpu_orbits,
            tdi_config="2nd generation" if gb_info.use_tdi2 else "1st generation",
            force_backend=general_info.force_backend,
            tdi_type="XYZ",
        )

    #* ============================== PSD ===============================
    num_repeats_psd = 5 if not gpu_available else 60
    _psd_search_move, psd_pe_move = build_psd_moves(
        engine_info, curr, acs, priors, num_repeats=num_repeats_psd,
    )

    #* ============================== GB ================================
    # Smoke-friendly: keep only the prior-based RJ proposal in PE mode (no
    # search list), expressed via the ``build_gb_moves`` design knobs.
    _, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state,
        include_search=False, pe_move_names=["rj_prior"],
    )

    #* ============================== MBH (phentax) =====================
    mbh_pe_move = None
    if "mbh" in curr.source_info:
        mbh_pe_move = _build_mbh_phentax_move(curr, acs, priors, state)

    #* ============================== EMRI ==============================
    emri_pe_move = None
    if "emri" in curr.source_info:
        emri_pe_move = _build_emri_move(curr, acs, priors, state)

    #* ============================== SOBBH =============================
    sobbh_pe_move = None
    if "sobbh" in curr.source_info:
        sobbh_pe_move = _build_sobbh_move(curr, acs, priors, state)

    #* ============================== COMBINE ===========================
    pe_moves = [psd_pe_move] + gb_pe_moves
    if mbh_pe_move is not None:
        pe_moves.append(mbh_pe_move)
    if emri_pe_move is not None:
        pe_moves.append(emri_pe_move)
    if sobbh_pe_move is not None:
        pe_moves.append(sobbh_pe_move)

    gf_pe_move = GFCombineMove(
        moves=pe_moves, verbose=True, share_temperature_control=False
    )
    gf_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(
        PERecipeStep(moves=[gf_pe_move]), name="full pe"
    )


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

    gb_setup = get_gb_erebor_settings(general_setup)
    psd_setup = get_psd_erebor_settings(general_setup)
    galfor_setup = get_galfor_erebor_settings(general_setup)
    emri_setup = get_emri_multi_erebor_settings(general_setup)
    sobbh_setup = get_sobbh_multi_erebor_settings(general_setup)
    mbh_setup = get_mbh_phentax_multi_erebor_settings(general_setup)

    # GB goes last for debugging: if it errors there, we know it made it
    # through all other proposals.
    gf_settings = GlobalFitSettings(
        source_info={
            "psd": psd_setup,
            "galfor": galfor_setup,
            "mbh": mbh_setup,
            "emri": emri_setup,
            "sobbh": sobbh_setup,
            "gb": gb_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )
    return CurrentInfoGlobalFit(gf_settings)


if __name__ == "__main__":
    settings = get_global_fit_settings()
    print("Combined GB+galfor+PSD+MBH+EMRI+SOBBH smoke settings constructed OK")
