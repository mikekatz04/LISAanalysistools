"""PSD-parameter recovery MCMC demo (FD then WDM).

End-to-end exercise that uses lisatools' stock noise machinery and Eryn's
sampler to recover two PSD parameters from a single noise realisation:

1. Build a 3x3 XYZ TDI 2 composite covariance from
   :class:`InstrumentNoise` + :class:`GalacticForeground` + :class:`SGWB`.
2. Draw an FD noise realisation correlated by that covariance, using
   :func:`generate_noise_fd` for the per-channel unit-PSD white noise and
   a per-frequency Cholesky of the truth covariance to colour it.
3. Run an Eryn ``EnsembleSampler`` over the foreground amplitude
   ``log10_A_fg`` and the SGWB amplitude ``log10_A_sgwb`` against the
   lisatools likelihood (``inner_product`` + ``noise_likelihood_term`` via
   :class:`AnalysisContainer`).
4. Transform the same FD noise realisation into the WDM basis with
   :meth:`FDSignal.wdmtransform` and rerun the same MCMC against the WDM
   composite covariance, demonstrating that the recovery survives the
   basis change.
5. Drop one corner plot per basis, centred on the truth parameters.

Run from the LISAanalysistools repo root:

    python plot_tests/mcmc_psd_recovery.py
"""

import os
import warnings

import corner
import matplotlib.pyplot as plt
import numpy as np

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import DefaultOrbits, LISAModel
from lisatools.diagnostic import (
    inner_product,
    noise_likelihood_term,
    residual_source_likelihood_term,
)
from lisatools.domains import FDSettings, FDSignal, WDMSettings
from lisatools.sensitivity import (
    CompositeSensitivityMatrix,
    GalacticForeground,
    InstrumentNoise,
    SGWB,
)
from lisatools.stochastic import PowerLawSGWB
from lisatools.utils.utility import generate_noise_fd


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Truth model — full 9-parameter noise + SGWB vector.
# ---------------------------------------------------------------------------
# The MCMC samples (log10) of every dimensional noise parameter plus the SGWB
# spectral index. Order:
#   0: log10_Soms_d         OMS displacement noise (sangria)
#   1: log10_Sa_a           Test-mass acceleration noise (sangria)
#   2: log10_A_fg           Galactic-foreground amplitude
#   3: log10_fk_fg          Foreground knee frequency [Hz]
#   4: alpha_fg             Foreground low-f power-law exponent
#   5: log10_s1_fg          Foreground slope-below-knee parameter
#   6: log10_s2_fg          Foreground slope-above-knee parameter
#   7: log10_A_SGWB         Power-law SGWB amplitude at f_ref = 25 Hz
#   8: alpha_SGWB           SGWB spectral index

# sangria-tuned truth values:
SOMS_D_TRUE = (7.9e-12) ** 2  # m^2 / Hz
SA_A_TRUE = (2.4e-15) ** 2    # m^2 s^-4 / Hz
FG_AMP_TRUE = 3.26651613e-44
FG_KNEE_TRUE = 2.09278117e-03
FG_ALPHA_TRUE = 1.18300266e00
FG_S1_TRUE = 3.01430978e03
FG_S2_TRUE = 2.95774596e03
SGWB_LOG10_A_TRUE = -8.0
SGWB_ALPHA_TRUE = 2.0 / 3.0

TRUTHS = np.array([
    float(np.log10(SOMS_D_TRUE)),
    float(np.log10(SA_A_TRUE)),
    float(np.log10(FG_AMP_TRUE)),
    float(np.log10(FG_KNEE_TRUE)),
    FG_ALPHA_TRUE,
    float(np.log10(FG_S1_TRUE)),
    float(np.log10(FG_S2_TRUE)),
    SGWB_LOG10_A_TRUE,
    SGWB_ALPHA_TRUE,
])
NDIM = TRUTHS.size

PARAM_LABELS = [
    r"$\log_{10} S_\mathrm{oms}$",
    r"$\log_{10} S_\mathrm{acc}$",
    r"$\log_{10} A_\mathrm{fg}$",
    r"$\log_{10} f_k$",
    r"$\alpha_\mathrm{fg}$",
    r"$\log_{10} s_1$",
    r"$\log_{10} s_2$",
    r"$\log_{10} A_\mathrm{SGWB}$",
    r"$\alpha_\mathrm{SGWB}$",
]

# Sampling. With 9 free dims the corner needs ~10^5 post-burn samples for
# smooth contours; 32 walkers x 2500 steps after a 500-step burn gives 80K.
NWALKERS = 32
NSTEPS = 2500
BURN = 500
SEED = 1234

# Prior bounds — wide-ish, but still bounding the recoverable region.
PRIOR_BOUNDS = np.array([
    [-24.0, -20.0],   # log10 Soms_d
    [-31.0, -27.0],   # log10 Sa_a
    [-46.0, -41.0],   # log10 A_fg
    [-4.0,  -1.5],    # log10 fk_fg
    [ 0.0,   3.0],    # alpha_fg
    [ 2.0,   4.5],    # log10 s1_fg
    [ 2.0,   4.5],    # log10 s2_fg
    [-12.0, -5.0],    # log10 A_SGWB
    [-1.0,   2.0],    # alpha_SGWB
])


# ---------------------------------------------------------------------------
# Composite-PSD builders
# ---------------------------------------------------------------------------


def _build_lisa_model(log10_Soms_d: float, log10_Sa_a: float) -> LISAModel:
    """Build a sangria-flavoured :class:`LISAModel` with custom noise levels."""
    return LISAModel(
        Soms_d=10.0**log10_Soms_d,
        Sa_a=10.0**log10_Sa_a,
        orbits=DefaultOrbits(),
        name="sangria_custom",
    )


def make_composite(settings, theta: np.ndarray):
    """Return a fresh ``CompositeSensitivityMatrix`` at the given parameter vector.

    ``fill_nans=0.0`` on the instrument component keeps the f=0 row finite,
    which matters for the WDM ``fold`` PSD integral (the boundary layer
    integrand otherwise carries a NaN through to the composite covariance and
    trips the assertion in :func:`noise_likelihood_term`).
    """
    (
        log10_Soms_d,
        log10_Sa_a,
        log10_A_fg,
        log10_fk_fg,
        alpha_fg,
        log10_s1_fg,
        log10_s2_fg,
        log10_A_sgwb,
        alpha_sgwb,
    ) = theta

    model = _build_lisa_model(log10_Soms_d, log10_Sa_a)
    components = [
        InstrumentNoise(tdi_generation=2, model=model, fill_nans=0.0),
        GalacticForeground(
            foreground_params=(
                10.0**log10_A_fg,
                10.0**log10_fk_fg,
                alpha_fg,
                10.0**log10_s1_fg,
                10.0**log10_s2_fg,
            ),
            tdi_generation=2,
            modulation=None,
        ),
        SGWB(
            sgwb_params=(log10_A_sgwb, alpha_sgwb),
            stochastic_fn=PowerLawSGWB,
            tdi_generation=2,
            modulation=None,
        ),
    ]
    return CompositeSensitivityMatrix(settings, components)


def set_composite_params(composite, theta: np.ndarray):
    """Mutate every component on an existing composite and re-sum.

    Updates instrument noise levels (via a swapped-in :class:`LISAModel`),
    foreground parameters, and SGWB parameters in place, then triggers a
    per-component cache refresh. The lazy ``invC`` / ``detC`` on
    :class:`SensitivityMatrixBase` defer the matrix inverse to the next
    likelihood read.
    """
    (
        log10_Soms_d,
        log10_Sa_a,
        log10_A_fg,
        log10_fk_fg,
        alpha_fg,
        log10_s1_fg,
        log10_s2_fg,
        log10_A_sgwb,
        alpha_sgwb,
    ) = theta

    composite.components[0].model = _build_lisa_model(log10_Soms_d, log10_Sa_a)
    composite.components[1].foreground_params = (
        10.0**log10_A_fg,
        10.0**log10_fk_fg,
        alpha_fg,
        10.0**log10_s1_fg,
        10.0**log10_s2_fg,
    )
    composite.components[2].sgwb_params = (log10_A_sgwb, alpha_sgwb)
    composite.update_component(0)
    composite.update_component(1)
    composite.update_component(2)


# ---------------------------------------------------------------------------
# Correlated FD noise draw
# ---------------------------------------------------------------------------


def draw_fd_noise(settings: FDSettings, sigma_truth: np.ndarray, seed: int) -> np.ndarray:
    """Draw an XYZ FD noise realisation with covariance ``sigma_truth``.

    ``sigma_truth`` is the full ``(3, 3, N_active)`` truth covariance from
    :class:`CompositeSensitivityMatrix.sens_mat`. The function generates three
    unit-PSD white-noise channels with :func:`generate_noise_fd` (the FD noise
    utility the user asked us to lean on) and colours them per-frequency by
    the Cholesky factor of the truth covariance.
    """
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        # Three independent unit-PSD white-noise channels covering the full
        # f >= 0 grid; we slice to the active band below.
        z_full = np.stack(
            [generate_noise_fd(settings.N, settings.df, func=lambda f: np.ones_like(f))
             for _ in range(3)]
        )
    finally:
        np.random.set_state(rng_state)

    z_active = z_full[:, settings.active_slice]
    # Cholesky per frequency: (Nf, 3, 3). Cholesky requires SPD inputs, so
    # bail out if a frequency bin has non-finite covariance (e.g. f=0 left in).
    sigma_per_f = np.transpose(sigma_truth, (2, 0, 1))
    bad = ~np.all(np.isfinite(sigma_per_f), axis=(1, 2))
    if np.any(bad):
        raise ValueError(
            f"Truth covariance has {bad.sum()} non-finite frequency bins in the "
            "active band; tighten min_freq above 0 or fill NaNs before calling."
        )
    L_per_f = np.linalg.cholesky(sigma_per_f)  # (Nf, 3, 3)
    # d_i(f) = sum_j L_ij(f) * z_j(f). The Cholesky picks up the cross-channel
    # correlation that ``generate_noise_fd`` itself can't (it's 1-D).
    d_active = np.einsum("kij,jk->ik", L_per_f, z_active)
    return d_active


# ---------------------------------------------------------------------------
# Likelihood + sampler
# ---------------------------------------------------------------------------


def make_log_like_fn(ac: AnalysisContainer):
    """Build an Eryn-compatible log-likelihood that mutates ``ac.sens_mat`` in place.

    The full lisatools likelihood (``inner_product`` source term +
    ``noise_likelihood_term``) is correct out-of-the-box in the FD: each
    complex bin carries two real degrees of freedom, so ``-log det Σ`` per
    bin acts like ``-(1/2) log det Σ`` *per real dof*, which matches the
    standard Whittle form. In the WDM basis the coefficients are real (a
    single dof per pixel), so the noise term in lisatools is a factor of 2
    too large; we halve it here to bring the MLE back to truth.
    """
    composite = ac.sens_mat
    bounds_lo = PRIOR_BOUNDS[:, 0]
    bounds_hi = PRIOR_BOUNDS[:, 1]
    # Real-coefficient bases (WDM) need a 0.5 factor on the noise term to
    # match the source-term normalisation.
    noise_term_factor = 0.5 if isinstance(composite.basis_settings, WDMSettings) else 1.0
    data_res_arr = ac.data_res_arr

    def log_like_fn(x, *args):
        theta = np.atleast_1d(np.asarray(x, dtype=float))
        if np.any(theta < bounds_lo) or np.any(theta > bounds_hi):
            return -1e30
        set_composite_params(composite, theta)
        try:
            nlt = float(np.real(noise_likelihood_term(composite)))
            src = float(np.real(
                residual_source_likelihood_term(data_res_arr, psd=composite)
            ))
            return noise_term_factor * nlt + src
        except (np.linalg.LinAlgError, FloatingPointError):
            return -1e30

    return log_like_fn


def run_sampler(ac: AnalysisContainer, label: str):
    priors = ProbDistContainer(
        {i: uniform_dist(PRIOR_BOUNDS[i, 0], PRIOR_BOUNDS[i, 1]) for i in range(NDIM)}
    )

    np.random.seed(SEED)
    # Initialise walkers in a tight ball around truth (a small fraction of the
    # prior width per parameter -- enough for the stretch move to find the
    # mode within a few hundred steps without spending samples on burn-in).
    prior_width = PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0]
    spread = 0.02 * prior_width
    # Eryn expects (ntemps, nwalkers, nleaves_max, ndim) initial coords.
    coords = TRUTHS + spread * np.random.randn(1, NWALKERS, 1, NDIM)
    # Clamp to prior bounds to avoid an initial-state rejection cascade.
    coords = np.clip(coords, PRIOR_BOUNDS[:, 0] + 1e-6, PRIOR_BOUNDS[:, 1] - 1e-6)

    log_like_fn = make_log_like_fn(ac)

    sampler = EnsembleSampler(
        NWALKERS,
        NDIM,
        log_like_fn,
        priors,
        vectorize=False,
    )
    print(f"[{label}] running {NSTEPS} steps x {NWALKERS} walkers x {NDIM} dims...")
    sampler.run_mcmc(coords, NSTEPS, burn=BURN, progress=True)
    return sampler


def corner_plot(samples: np.ndarray, label: str, out_path: str):
    fig = corner.corner(
        samples,
        labels=PARAM_LABELS,
        truths=list(TRUTHS),
        truth_color="C3",
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
        hist_kwargs={"density": True},
        label_kwargs={"fontsize": 10},
        title_kwargs={"fontsize": 9},
    )
    fig.suptitle(f"PSD recovery ({NDIM} params) — {label}", y=1.005, fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    np.random.seed(SEED)

    # --- FD setup -----------------------------------------------------------
    # Pick the underlying time-domain length once: N_time = Nf_wdm * Nt_wdm so
    # the FD->WDM transform lines up exactly (N_fd = N_time/2 + 1). The grid is
    # deliberately small (N_time = 2048) so the per-walker WDM PSD build stays
    # under ~15 ms -- enough to finish the WDM MCMC in a few minutes on CPU.
    Nf_wdm = 32
    Nt_wdm = 64
    N_time = Nf_wdm * Nt_wdm  # = 2048
    data_dt = 10.0
    df = 1.0 / (N_time * data_dt)
    N_fd = N_time // 2 + 1  # rFFT length

    fd_settings = FDSettings(
        N=N_fd,
        df=df,
        min_freq=3e-4,
        max_freq=3e-2,
        force_backend="cpu",
    )

    truth_composite_fd = make_composite(fd_settings, TRUTHS)
    sigma_truth_fd = np.asarray(truth_composite_fd.sens_mat)
    print(
        f"FD: N_active={fd_settings.N_active}, "
        f"sigma shape={sigma_truth_fd.shape}, df={fd_settings.df}"
    )

    fd_noise_active = draw_fd_noise(fd_settings, sigma_truth_fd, seed=SEED)
    data_arr_fd = DataResidualArray(fd_noise_active, input_signal_domain=fd_settings)

    sampling_composite = make_composite(fd_settings, TRUTHS)
    ac_fd = AnalysisContainer(data_arr_fd, sampling_composite)
    truth_ll_fd = float(np.real(ac_fd.likelihood()))
    print(f"FD log-L at truth: {truth_ll_fd:.3e}")

    sampler_fd = run_sampler(ac_fd, label="FD")
    chain_fd = sampler_fd.get_chain(discard=BURN)["model_0"]  # (nsteps, nwalkers, 1, NDIM)
    samples_fd = chain_fd.reshape(-1, NDIM)
    corner_plot(samples_fd, "FD", os.path.join(OUT_DIR, "mcmc_psd_recovery_fd.png"))

    # --- WDM setup ----------------------------------------------------------
    # Same total time-domain length as the FD setup (Nf*Nt = N_time), so the
    # FD->WDM transform's index math lines up bin-for-bin.
    wdm_settings = WDMSettings(
        Nf=Nf_wdm,
        Nt=Nt_wdm,
        dt=data_dt,
        min_freq=fd_settings.min_freq,
        max_freq=fd_settings.max_freq,
        force_backend="cpu",
    )
    print(
        f"WDM: Nf={wdm_settings.Nf}, Nt={wdm_settings.Nt}, "
        f"data_dt={wdm_settings.data_dt}, layer_df={wdm_settings.layer_df:.3e}"
    )

    # Wrap the FD noise as an FDSignal on the full f >= 0 grid and transform
    # to WDM. We deliberately rebuild a *full-range* FDSettings here (no
    # min/max masking) because ``FDSignal.wdmtransform`` needs the underlying
    # rFFT-length array, and the masked FDSettings used for the FD MCMC would
    # otherwise be re-trimmed inside ``FDSignal.__init__``.
    full_fd_settings = FDSettings(
        N=fd_settings.N,
        df=fd_settings.df,
        force_backend="cpu",
    )
    full_fd_noise = np.zeros((3, fd_settings.N), dtype=complex)
    full_fd_noise[:, fd_settings.active_slice] = fd_noise_active
    fd_signal = FDSignal(full_fd_noise, full_fd_settings)
    wdm_signal = fd_signal.wdmtransform(settings=wdm_settings)

    data_arr_wdm = DataResidualArray(wdm_signal, input_signal_domain=wdm_settings)

    sampling_composite_wdm = make_composite(wdm_settings, TRUTHS)
    ac_wdm = AnalysisContainer(data_arr_wdm, sampling_composite_wdm)
    truth_ll_wdm = float(np.real(ac_wdm.likelihood()))
    print(f"WDM log-L at truth: {truth_ll_wdm:.3e}")

    sampler_wdm = run_sampler(ac_wdm, label="WDM")
    chain_wdm = sampler_wdm.get_chain(discard=BURN)["model_0"]
    samples_wdm = chain_wdm.reshape(-1, NDIM)
    corner_plot(samples_wdm, "WDM", os.path.join(OUT_DIR, "mcmc_psd_recovery_wdm.png"))


if __name__ == "__main__":
    main()
