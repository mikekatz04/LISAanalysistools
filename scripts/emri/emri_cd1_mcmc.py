"""EMRI test MCMC against CD1 (mojito L1) data.

Loads one EMRI source from a CD1 L1 dataset, builds the WDM-domain
injection from the L1 data stream and a matching legacy-response
(``ResponseWrapper``) template, sanity-checks the injection
likelihood / overlap / SNR, then runs an Eryn ensemble MCMC over 12
sampled parameters (``x0`` and ``Phi_theta0`` held fixed).

credit Michael Katz and Alessandro Santini
"""

import os

import numpy as np
from scipy.signal.windows import tukey

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import PeriodicContainer, TransformContainer
from few.waveform import GenerateEMRIWaveform
from lisaconstants import ASTRONOMICAL_YEAR

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.detector import L1Orbits
from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.response import ResponseWrapper, TDIConfig
from lisatools.response.directresponse import icrs_to_ecliptic
from lisatools.sensitivity import XYZ2SensitivityMatrix

# --- configuration -----------------------------------------------------------

L1_PATH = "/scratch-jpl/335-lisa/mlkatz/cd1L_data"
SOURCE_ID = 0
BACKEND = "cuda12x"  # "cpu" or "cuda12x"

NTEMPS = 4
NWALKERS = 32
NSTEPS = 2000
THIN_BY = 5

PRIOR_HALF_WIDTH = 1e-2  # fractional prior half-width on the intrinsic parameters
START_HALF_WIDTH_INTRINSIC = 1e-7  # start-ball half-widths around the injection
START_HALF_WIDTH_EXTRINSIC = 1e-3
CHOP_ENDS = 10000  # samples trimmed from each end of both data and template

xp = np if BACKEND == "cpu" else cp

# full = FEW waveform parameter order (must never change); sampled = MCMC order
FULL_BASIS = [
    "M", "mu", "a", "p0", "e0", "x0", "dist",
    "qS", "phiS", "qK", "phiK",
    "Phi_phi0", "Phi_theta0", "Phi_r0",
]
SAMPLED_BASIS = [
    "M", "mu", "a", "p0", "e0", "dist",
    "cosqS", "phiS", "cosqK", "phiK",
    "Phi_phi0", "Phi_r0",
]
INTRINSIC = ["M", "mu", "a", "p0", "e0"]


class WaveWrap:
    """Generate legacy-response TDI channels aligned to the data grid, in the WDM domain."""

    def __init__(self, wave_gen, temp_start, chop_ends, td_set, output_set, window):
        self.wave_gen = wave_gen
        self.temp_start, self.chop_ends = temp_start, chop_ends
        self.td_set, self.output_set, self.window = td_set, output_set, window

    def __call__(self, *params, **kwargs):
        tdi_channels = self.wave_gen(*params, **kwargs)
        wave = xp.asarray([chan[self.temp_start:] for chan in tdi_channels])
        wave = wave[:, self.chop_ends:-self.chop_ends][:, : self.td_set.N]
        wdm = TDSignal(wave, self.td_set).transform(self.output_set, window=self.window)
        return DataResidualArray(wdm)


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
        orbits_kwargs=dict(force_backend=BACKEND, frame="icrs"),  # equatorial coords (ra/dec)
        verbose=True,
        do_plots=False,
    )
    binary_params = loader.catalogue["EMRI"][SOURCE_ID]
    orbits = loader.orbits
    dt = loader.dt

    print(f"Source ID {SOURCE_ID} catalogue parameters:")
    for key, value in binary_params.items():
        print(f"  {key}: {value}")

    # --- analysis grids --------------------------------------------------------
    # 1 or 2 years of data depending on when the EMRI plunges
    Tobs_wave = np.ceil(binary_params["TimeCoalescenceSSBFrame"] / ASTRONOMICAL_YEAR)
    Tobs_sec = Tobs_wave * ASTRONOMICAL_YEAR

    # wavelet duration between half and 0.6 days
    Nf, Nt, wavelet_duration = WDMSettings.adjust_to_even_bins(
        0.5 * 24 * 3600.0, 0.6 * 24 * 3600.0, dt, Tobs_sec
    )
    N = Nf * Nt
    td_set = TDSettings(N, dt, force_backend=BACKEND)
    wdm_set = WDMSettings(
        Nf,
        Nt,
        dt,
        min_freq=1e-4,
        max_freq=0.030,
        min_time=wavelet_duration * 20,
        max_time=wavelet_duration * (Nt - 20),
        force_backend=BACKEND,
    )
    window = xp.asarray(tukey(N, alpha=0.05))

    # --- EMRI waveform + legacy response ---------------------------------------
    few_generator = GenerateEMRIWaveform(
        "FastKerrEccentricEquatorialFlux",
        return_list=False,  # hp - i*hx as a single complex array
        inspiral_kwargs={
            "DENSE_STEPPING": 0,  # sparsely sampled trajectory
            "max_init_len": int(1e4),
            "force_backend": "cpu",
        },
        sum_kwargs={"pad_output": True},
        mode_selector_kwargs={"mode_selection_threshold": 1e-2},
        frame="detector",
        force_backend=BACKEND,
    )

    # snap the waveform reference time onto the data time grid
    t0 = binary_params["TimeReferenceSSBFrame"]
    t0_shift_to_data = loader.times[0] + round((t0 - loader.times[0]) / dt) * dt - t0

    legacy_tdi_generator = ResponseWrapper(
        few_generator,
        orbits=orbits,
        t0=t0,
        t0_shift_to_data=t0_shift_to_data,
        Tobs=Tobs_wave,
        dt=dt,
        index_lambda=8,
        index_beta=7,
        flip_hx=True,
        force_backend=BACKEND,
        tdi=TDIConfig("2nd generation"),
        tdi_chan="XYZ",
        order=40,
        remove_garbage="zero",
        is_ecliptic_latitude=False,
        t_buffer=30000.0,
    )

    # --- injection parameters in FEW order --------------------------------------
    lambda_ecl, beta_ecl = icrs_to_ecliptic(
        binary_params["RightAscension"], binary_params["Declination"]
    )
    wf_params = np.array(
        [
            binary_params["PrimaryMassSSBFrame"],  # M
            binary_params["SecondaryMassSSBFrame"],  # mu
            binary_params["PrimarySpinParameter"],  # a
            binary_params["SemiLatusRectum"],  # p0
            binary_params["Eccentricity"],  # e0
            np.cos(binary_params["InclinationAngle"]),  # x0
            binary_params["LuminosityDistance"] * 1e-3,  # dist [Gpc]
            np.pi / 2 - beta_ecl,  # qS
            lambda_ecl,  # phiS
            binary_params["PolarAnglePrimarySpin"],  # qK
            binary_params["AzimuthalAnglePrimarySpin"],  # phiK
            binary_params["AzimuthalPhase"],  # Phi_phi0
            binary_params["PolarPhase"],  # Phi_theta0
            binary_params["RadialPhase"],  # Phi_r0
        ]
    )
    injection_full = dict(zip(FULL_BASIS, wf_params))

    # --- align template + injection on a common grid -----------------------------
    tdi_channels = legacy_tdi_generator(*wf_params)
    times_waveform = legacy_tdi_generator.response_model.t_arr_proj

    assert times_waveform[0] < loader.times[0]
    assert times_waveform[-1] < loader.times[-1]

    # template starts before the data: drop its head; the data outlives the
    # template: trim the data tail (inj_end is negative)
    temp_start = int((loader.times[0] - times_waveform[0]) / dt)
    inj_end = int((times_waveform[-1] - loader.times[-1]) / dt)

    template_time = times_waveform[temp_start:][CHOP_ENDS:-CHOP_ENDS]
    injection_time = loader.times[:inj_end][CHOP_ENDS:-CHOP_ENDS]
    assert xp.allclose(xp.asarray(injection_time), xp.asarray(template_time))

    injection_wave = xp.asarray(loader.data[:, :inj_end])[:, CHOP_ENDS:-CHOP_ENDS][:, :N]
    injection = DataResidualArray(
        TDSignal(injection_wave, td_set).transform(wdm_set, window=window)
    )

    # --- analysis container + injection sanity check ------------------------------
    signal_gen = WaveWrap(legacy_tdi_generator, temp_start, CHOP_ENDS, td_set, wdm_set, window)
    sens_mat = XYZ2SensitivityMatrix(injection.data_res_arr.settings, model="scirdv1")
    analysis = AnalysisContainer(injection, sens_mat, signal_gen=signal_gen)

    template = signal_gen(*wf_params)
    check_ll = analysis.template_likelihood(template)
    check_snr = analysis.template_snr(template)
    overlap = analysis.template_inner_product(template, normalize=True)
    print(
        f"EMRI {SOURCE_ID}: log-likelihood {check_ll}, mismatch {1.0 - overlap}, "
        f"overlap {overlap}, SNR (observed, optimal) {check_snr}"
    )

    # --- MCMC setup ----------------------------------------------------------------
    transform_fn = TransformContainer(
        input_basis=SAMPLED_BASIS,
        output_basis=FULL_BASIS,
        parameter_transforms={"cosqS": np.arccos, "cosqK": np.arccos},
        fill_dict={
            "x0": injection_full["x0"],
            "Phi_theta0": injection_full["Phi_theta0"],
        },
        key_map={"cosqS": "qS", "cosqK": "qK"},
    )

    # injection values in the sampled coordinates
    injection_sampled = {
        name: np.cos(injection_full[name[3:]]) if name.startswith("cos") else injection_full[name]
        for name in SAMPLED_BASIS
    }

    priors = {
        "emri": ProbDistContainer(
            {
                "M": fractional_ball(injection_sampled["M"], PRIOR_HALF_WIDTH),
                "mu": fractional_ball(injection_sampled["mu"], PRIOR_HALF_WIDTH),
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
    print("injection log_like:", injection_like[0, 0])

    sampler.run_mcmc(start_state, nsteps=NSTEPS, burn=0, thin_by=THIN_BY, progress=True)
