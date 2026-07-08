"""EMRI test MCMC against CD1 (mojito L1) data.

Loads one EMRI source from a CD1 L1 dataset, builds the WDM-domain
injection from the (noiseless) L1 data stream and a matching template
via the validated SPECIAL EMRI frame recipe, sanity-checks the
injection likelihood (should be ~0: the template nulls the data), then
runs an Eryn ensemble MCMC over the 12-parameter stock sampling basis
(``xI0`` and ``Phi_theta0`` held fixed).

SPECIAL EMRI frame (validated 2026-06-19, see
``global_fit_input/full_year_combined_global_fit_settings.py`` and
``scripts/sobbh/run_mojito_null_checks.sh``):

- FEW gets ECLIPTIC-POLAR sky angles (``qS = pi/2 - ecliptic latitude``,
  ``phiS = ecliptic longitude``, converted from the catalogue ICRS
  RA/Dec) together with the RAW catalogue spin angles (``qK``/``phiK``
  NOT converted) and ``xI0 = +1`` (equatorial prograde — the catalogue
  "InclinationAngle" is the viewing inclination, which FEW derives
  internally).
- The response runs against ``frame="icrs"`` orbits, so the sky is
  converted ecliptic -> ICRS per call (``convert_to_ra_dec=True`` via
  ``_EMRISpecialFrameWrap`` inside ``get_emri_response_wrapper``).
- The FEW phases are referenced to the catalogue reference epoch
  (``TimeReferenceSSBFrame`` == ``lisatools.globalfit.recipe.
  MOJITO_REFERENCE_TIME``): the response ``t0`` is REF, the integer-
  sample part of ``data_t0 - REF`` is sliced off the response output by
  ``EMRIWaveWrap`` and the sub-sample remainder goes through
  ``t0_shift_to_data``.

credit Michael Katz and Alessandro Santini
"""

import os
import sys

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import PeriodicContainer

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import L1Orbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.stock.erebor import make_emri_transform_container
from lisatools.response import TDIConfig
from lisatools.response.directresponse import icrs_to_ecliptic
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

# The validated EMRI response/frame helpers live in the repo-level
# global_fit_input directory (not the installed package).
_LAT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_LAT_ROOT, "global_fit_input"))
from global_fit_settings import EMRIWaveWrap, get_emri_response_wrapper

# --- configuration -----------------------------------------------------------

L1_PATH = "/scratch-jpl/335-lisa/mlkatz/cd1L_data"
SOURCE_ID = 0
BACKEND = "cuda12x"  # "cpu" or "cuda12x"

# Analysis window from the data start (the EMRI null-check default is
# 3-6 months; the WDM grid is fit inside this target).
TOBS_TARGET = 0.25 * YRSID_SI
WAVELET_DUR_BOUNDS = (40000.0, 48000.0)  # adjust_to_even_bins search window [s]
MIN_FREQ = 1e-4  # Hz
MAX_FREQ = 2.5e-2  # Hz

NTEMPS = 4
NWALKERS = 32
NSTEPS = 2000
THIN_BY = 5

PRIOR_HALF_WIDTH = 1e-2  # fractional prior half-width on the intrinsic parameters
START_HALF_WIDTH_INTRINSIC = 1e-7  # start-ball half-widths around the injection
START_HALF_WIDTH_EXTRINSIC = 1e-3

xp = np if BACKEND == "cpu" else cp

# Stock EMRI sampling basis (make_emri_transform_container order).
SAMPLED_BASIS = [
    "logm1", "m2", "a", "p0", "e0", "dist",
    "cosqS", "phiS", "cosqK", "phiK",
    "Phi_phi0", "Phi_r0",
]
INTRINSIC = ["logm1", "m2", "a", "p0", "e0"]


def fractional_ball(center, half_width):
    # sorted() keeps the bounds ordered when the center is negative (cos angles)
    lo, hi = sorted((center * (1.0 - half_width), center * (1.0 + half_width)))
    return uniform_dist(lo, hi)


if __name__ == "__main__":
    # --- load the L1 data + catalogue entry -----------------------------------
    loader = L1ProcessingStep(
        L1_folder=L1_PATH,
        source_types=["emri"],
        source_ids=dict(emri=[SOURCE_ID]),
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),
        verbose=True,
        do_plots=False,
    )
    binary_params = loader.catalogue["EMRI"][SOURCE_ID]
    dt = loader.dt
    data_t0 = float(loader.times[0])

    print(f"Source ID {SOURCE_ID} catalogue parameters:")
    for key, value in binary_params.items():
        print(f"  {key}: {value}")

    # --- analysis grids --------------------------------------------------------
    Nf, Nt, wavelet_duration = WDMSettings.adjust_to_even_bins(
        t_min=WAVELET_DUR_BOUNDS[0],
        t_max=WAVELET_DUR_BOUNDS[1],
        dt=dt,
        Tobs=TOBS_TARGET,
    )
    N = Nf * Nt
    assert loader.data.shape[1] >= N, "data span is shorter than the WDM grid"

    td_set = TDSettings(N, dt, t0=data_t0, force_backend=BACKEND)
    wdm_set = WDMSettings(
        Nf,
        Nt,
        dt,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        min_time=20 * wavelet_duration,
        max_time=(Nt - 20) * wavelet_duration,
        force_backend=BACKEND,
    )

    # injection (data) in the WDM domain — rectangular window, matching the
    # full_year pipeline (WINDOW_TAPER_DURATION = 0, td_window=None)
    injection_wave = xp.asarray(loader.data[:, :N])
    injection = DataResidualArray(TDSignal(injection_wave, td_set).transform(wdm_set))

    # --- SPECIAL-frame injection parameters in FEW order ------------------------
    ra = float(binary_params["RightAscension"]) % (2 * np.pi)
    dec = float(binary_params["Declination"])
    lambda_ecl, beta_ecl = icrs_to_ecliptic(ra, dec)
    wf_params = np.array(
        [
            binary_params["PrimaryMassSSBFrame"],  # M
            binary_params["SecondaryMassSSBFrame"],  # mu
            binary_params["PrimarySpinParameter"],  # a
            binary_params["SemiLatusRectum"],  # p0
            binary_params["Eccentricity"],  # e0
            1.0,  # xI0: equatorial prograde (NOT cos(InclinationAngle))
            binary_params["LuminosityDistance"] / 1e3,  # dist [Gpc]
            float(np.pi / 2 - beta_ecl),  # qS (ecliptic polar)
            float(lambda_ecl) % (2 * np.pi),  # phiS (ecliptic longitude)
            binary_params["PolarAnglePrimarySpin"],  # qK (RAW file spin)
            binary_params["AzimuthalAnglePrimarySpin"],  # phiK (RAW file spin)
            binary_params["AzimuthalPhase"],  # Phi_phi0
            binary_params["PolarPhase"],  # Phi_theta0
            binary_params["RadialPhase"],  # Phi_r0
        ]
    )

    # --- REF-anchored response, aligned onto the data grid -----------------------
    ref = float(binary_params["TimeReferenceSSBFrame"])
    off = data_t0 - ref
    offset_int = int(round(off / dt))
    t0_shift = off - offset_int * dt  # sub-sample remainder, |.| < dt
    resp_Tobs = (N + offset_int) * dt

    wave_gen = get_emri_response_wrapper(
        Tobs=resp_Tobs,
        dt=dt,
        t_start=ref,
        t0_shift_to_data=t0_shift,
        tdi_config=TDIConfig("2nd generation", force_backend=BACKEND),
        tdi_chan="XYZ",
        force_backend=BACKEND,
        orbits=loader.orbits,
    )
    signal_gen = EMRIWaveWrap(
        wave_gen,
        td_set,
        wdm_set,
        td_window=None,
        nchannels=3,
        offset_int=offset_int,
    )

    # --- analysis container + injection null check -------------------------------
    sens_mat = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")
    analysis = AnalysisContainer(injection, sens_mat, signal_gen=signal_gen)

    template = signal_gen(*wf_params)
    check_ll = analysis.template_likelihood(template)  # -0.5 <d-h|d-h>: ~0 when h nulls d
    check_snr = analysis.template_snr(template)
    overlap = analysis.template_inner_product(template, normalize=True)
    print(
        f"EMRI {SOURCE_ID}: source-only log-likelihood {check_ll} (expect ~0), "
        f"mismatch {1.0 - overlap}, overlap {overlap}, "
        f"SNR (observed, optimal) {check_snr}"
    )

    # --- MCMC setup ----------------------------------------------------------------
    # stock transform: sampling basis -> FEW basis, filling xI0 and Phi_theta0
    transform_fn = make_emri_transform_container([wf_params[5], wf_params[12]])
    injection_sampled = dict(
        zip(SAMPLED_BASIS, transform_fn.both_inverse_transforms(wf_params))
    )

    priors = {
        "emri": ProbDistContainer(
            {
                "logm1": fractional_ball(injection_sampled["logm1"], PRIOR_HALF_WIDTH),
                "m2": fractional_ball(injection_sampled["m2"], PRIOR_HALF_WIDTH),
                "a": fractional_ball(injection_sampled["a"], PRIOR_HALF_WIDTH),
                "p0": fractional_ball(injection_sampled["p0"], PRIOR_HALF_WIDTH),
                "e0": fractional_ball(injection_sampled["e0"], PRIOR_HALF_WIDTH),
                "dist": uniform_dist(0.1, 10.0),  # Gpc
                "cosqS": uniform_dist(-1.0, 1.0),
                "phiS": uniform_dist(0.0, 2 * np.pi),
                "cosqK": uniform_dist(-1.0, 1.0),
                "phiK": uniform_dist(0.0, 2 * np.pi),
                "Phi_phi0": uniform_dist(0.0, 2 * np.pi),
                "Phi_r0": uniform_dist(0.0, 2 * np.pi),
            }
        )
    }

    # tight start ball around the injection
    start_dist = ProbDistContainer(
        {
            name: fractional_ball(
                injection_sampled[name],
                START_HALF_WIDTH_INTRINSIC if name in INTRINSIC else START_HALF_WIDTH_EXTRINSIC,
            )
            for name in SAMPLED_BASIS
        }
    )

    periodic = PeriodicContainer(
        {"emri": {name: 2 * np.pi for name in ["phiS", "phiK", "Phi_phi0", "Phi_r0"]}},
        key_order={"emri": SAMPLED_BASIS},
    )

    fp = f"emri_cd1_mcmc_id_{SOURCE_ID}.h5"
    if os.path.exists(fp):
        start_state = HDFBackend(fp).get_last_sample()
    else:
        start_state = State({"emri": start_dist.rvs(size=(NTEMPS, NWALKERS, 1))})

    sampler = EnsembleSampler(
        NWALKERS,
        {"emri": len(SAMPLED_BASIS)},
        analysis.eryn_likelihood_function,
        priors,
        tempering_kwargs=dict(ntemps=NTEMPS),
        kwargs=dict(transform_fn=transform_fn, source_only=True),
        moves=StretchMove(live_dangerously=True),
        branch_names=["emri"],
        periodic=periodic,
        backend=fp,
    )

    if start_state.log_like is None:
        start_state.log_prior = sampler.compute_log_prior(start_state.branches_coords)
        start_state.log_like = sampler.compute_log_like(
            start_state.branches_coords, logp=start_state.log_prior
        )[0]
    print("start log_like:", start_state.log_like)

    injection_coords = np.array([injection_sampled[name] for name in SAMPLED_BASIS])
    injection_state = State({"emri": np.tile(injection_coords, (NTEMPS, NWALKERS, 1, 1))})
    injection_like = sampler.compute_log_like(injection_state.branches_coords)[0]
    print("injection log_like:", injection_like[0, 0], "(expect ~0)")

    sampler.run_mcmc(start_state, nsteps=NSTEPS, burn=0, thin_by=THIN_BY, progress=True)
