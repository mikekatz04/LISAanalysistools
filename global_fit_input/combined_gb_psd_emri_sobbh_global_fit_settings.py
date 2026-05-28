"""Combined smoke-test settings: GB + galfor + PSD + EMRI + SOBBH (no MBH).

Reuses the pieces defined in ``global_fit_settings.py`` (the per-branch
``get_*_erebor_settings`` helpers, ``setup_recipe``, the shared waveform
wrappers, injection-parameter constants). The two unique bits here are:

* A custom :class:`SangriaPlusInjectionsProcessingStep` that loads Sangria
  data (mbhb removed, keeping the GB / galactic-foreground content) and
  adds the synthetic EMRI + SOBBH time-domain waveforms on top.
* A ``get_global_fit_settings`` that drops MBH from ``source_info`` so
  the shared ``setup_recipe`` skips the MBH branch.

PSD is fit against the Sangria instrument noise that's already baked into
the data.
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


from fastlisaresponse.tdiconfig import TDIConfig

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
    emri_full_to_sampling,
    get_emri_response_wrapper,
    get_galfor_erebor_settings,
    get_gb_erebor_settings,
    get_psd_erebor_settings,
    get_sobbh_response_wrapper,
    setup_recipe,
    sobbh_full_to_sampling,
)
from lisatools.globalfit.stock.erebor import EMRISettings, EMRISetup, SOBBHSettings, SOBBHSetup
from eryn.moves import StretchMove


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
    synthetic EMRI + synthetic SOBBH.

    Data composition:

    1. ``Sangria`` sky-only slice (``remove_from_data=["noise", "mbhb"]``)
       gives the GB / galactic-foreground content with no instrument-noise
       contribution.
    2. :func:`_generate_correlated_fd_noise` synthesises correlated XYZ
       noise from the chosen LISA instrument model in the FD domain and
       inverse-rFFTs it to TD.
    3. Synthetic EMRI and SOBBH waveforms are produced by the shared
       response wrappers used elsewhere in the move pre-injection paths.

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
            sig = np.asarray(emri_wave_gen(*params, convert_to_ra_dec=False))
            sig = _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)
            emri_td = emri_td + sig

        # --- SOBBH injections (sum over multiple sources) --------------------
        sobbh_wave_gen = get_sobbh_response_wrapper(
            Tobs=Tobs,
            dt=dt,
            t_start=t_start,
            tdi_config=tdi_config,
            tdi_chan=tdi_chan,
            role="injection",
            force_backend=force_backend,
        )
        sobbh_td = np.zeros_like(sangria_data)
        for i, params in enumerate(sobbh_injections):
            sig = np.asarray(sobbh_wave_gen(*params, convert_to_ra_dec=False))
            sig = _pad_or_clip(np.atleast_2d(sig)[:nchannels], target_N)
            sobbh_td = sobbh_td + sig

        combined = sangria_data + noise_td + emri_td + sobbh_td

        times = np.arange(target_N) * dt + t_start
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.tdi_chan = tdi_chan
        self.emri_injection_params_full_basis = emri_injections
        self.sobbh_injection_params_full_basis = sobbh_injections


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

    # No MBH — ``setup_recipe`` already gates on ``"mbh" in source_info``.
    # GB goes last for debugging: if it errors there, we know it made it through all other proposals.
    gf_settings = GlobalFitSettings(
        source_info={
            "psd": psd_setup,
            "galfor": galfor_setup,
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
    print("Combined GB+galfor+PSD+EMRI+SOBBH smoke settings constructed OK")
