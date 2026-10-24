"""Global-fit settings of the high-frequency GB run with each STFT response, DFFR and TDI on the fly.

Derived from mojito_input/gb_high_freq_global_fit_settings.py, with these changes: all 713
one-day segments after the 200 h trims, an STFT grid of 9.514-23.0 mHz (f0 sampled over
9.578-22.898 mHz, so every catalogue source above 9.75 mHz is sampled), a 1e4 s taper, 16 side
bins, the catalogue sources of the sampled f0 range as the start state, 64 leaves, 32 walkers,
24 temperatures, one GPU, and a stop after 1.45 days of wall clock. The two runs differ only in
``gb_stft_kwargs["response"]``; run_global.py picks one through the settings function:

    GLOBAL_FIT_RUNS_DIR=<folder> python scripts/run_global.py \\
        -sfp scripts/dffr/gb_high_freq_globalfit_settings.py -sff get_dffr_settings

The ``*_smoke_settings`` functions stop after a few iterations. Each run writes into
``GLOBAL_FIT_RUNS_DIR/<run name>/`` and copies this file there. Environment: GLOBAL_FIT_RUNS_DIR
(required), MOJITO_FOLDER (default /project/lisa_nlddpc/Data/mojito_light).
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import time

import numpy as np

try:
    import cupy as cp

    gpu_available = bool(cp.cuda.is_available())
except (ModuleNotFoundError, ImportError):
    gpu_available = False

from gbgpu.gbcomps import STFTGBComputations

from lisatools.detector import L1Orbits
from lisatools.domains import STFTSettings
from lisatools.globalfit.engine import GeneralSettings, GeneralSetup, GlobalFitSettings, RankInfo
from lisatools.globalfit.moves.gbdebug import GBDebugSettings
from lisatools.globalfit.postprocessing import SourceMetadata
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe import PERecipeStep, build_gb_moves, setup_state_for_injection
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import GBSettings, GBSetup, get_fdot_mojito
from lisatools.sensitivity import XYZSensitivityBackend

# ? Under "lisatools", whose level and run-log handler the global fit sets; __name__ is not.
logger = logging.getLogger("lisatools.globalfit.dffr_globalfit_settings")

# * Taken when this file is imported, so the wall-clock budget includes the data loading.
IMPORT_TIME_S = time.time()

STFT_SEGMENT_S = 86400.0
# * 713.8 days remain after the 200 h trims, so 713 whole segments are the most the data hold.
N_SEGMENTS = 713
# * Grid start as a bin of the 1/86400 Hz grid (bin 822 = 9.514 mHz). The band walk from there
# * samples f0 over 9.578-22.898 mHz; one bin higher moves the top of that range to 19.92 mHz.
START_BIN = 822
END_FREQ_HZ = 23.0e-3
# ! Setup stops if a catalogue source above this frequency lies outside the sampled f0 range.
SAMPLED_FROM_HZ = 9.75e-3
WINDOW_TAPER_S = 1.0e4
N_SIDE_BINS = 16
NLEAVES_MAX = 64
NWALKERS = 32
NTEMPS = 24
# ! Cells per GPU batch. A cell holds a 555-bin window of data and of the 3x3 CSD (76 MB) plus
# ! copies; 512 cells ran out of the A100's 40 GB next to the 15.5 GB of 32 walkers.
NUM_BAND_PRELOAD = 64
# * Tempering batches hold (band, walker) units of NTEMPS cells each.
NUM_BANDS_PRELOAD_TEMP = 2
WALL_CLOCK_BUDGET_S = 1.45 * 86400.0
SMOKE_ITERATIONS = 3
# ? An upper bound only: the wall-clock budget ends a full run first.
MAX_ITERATIONS = 1_000_000
RESPONSES = ("tdi_on_the_fly", "dffr")


class BudgetPERecipeStep(PERecipeStep):
    """Parameter estimation until a wall-clock budget, or an iteration count, is reached."""

    def __init__(self, moves, weights=None, budget_s=WALL_CLOCK_BUDGET_S, stop_iterations=None):
        super().__init__(moves=moves, weights=weights)
        self.budget_s = float(budget_s)
        self.stop_iterations = stop_iterations

    def stopping_function(self, iteration, last_sample, sampler):
        # * Eryn calls this after it saved iteration ``iteration`` (counted from 0).
        elapsed_s = time.time() - IMPORT_TIME_S
        stop = elapsed_s >= self.budget_s or (
            self.stop_iterations is not None and iteration + 1 >= self.stop_iterations
        )
        if stop:
            logger.info(
                f"Run stops after iteration {iteration} and {elapsed_s:.0f} s of wall clock."
            )
        return stop


def run_name(response: str, smoke_test: bool) -> str:
    return f"gb_high_freq_{response}" + ("_smoke" if smoke_test else "")


def setup_recipe(recipe, engine_info, curr, acs, priors, state, stop_iterations=None):
    general_info = curr.general_info
    if gpu_available:
        cp.cuda.runtime.setDevice(general_info.gpus[0])

    # * Start state: every catalogue source whose f0 lies in the sampled range, at its true values.
    gb_info = curr.source_info["gb"]
    f0_lo_hz, f0_hi_hz = (float(limit) for limit in gb_info.f0_lims)
    catalogue = general_info.catalogue["GB"]
    # ? Same row order as setup_state_for_injection, which stacks the entries by sorted source id.
    frequencies_hz = np.concatenate(
        [
            np.atleast_1d(catalogue[source_id]["GW22FrequencySSBFrame"])
            for source_id in sorted(catalogue)
        ]
    )
    in_range = (frequencies_hz >= f0_lo_hz) & (frequencies_hz < f0_hi_hz)
    missing = (frequencies_hz > SAMPLED_FROM_HZ) & ~in_range
    if np.any(missing):
        raise ValueError(
            f"catalogue sources at {np.sort(frequencies_hz[missing]) * 1e3} mHz lie outside the sampled "
            f"f0 range [{f0_lo_hz * 1e3:.4f}, {f0_hi_hz * 1e3:.4f}] mHz; move START_BIN or END_FREQ_HZ."
        )
    subset_inds = np.flatnonzero(in_range)
    if subset_inds.size > gb_info.nleaves_max:
        raise ValueError(
            f"{subset_inds.size} injected sources exceed nleaves_max {gb_info.nleaves_max}."
        )
    logger.info(
        f"Injecting {subset_inds.size} catalogue sources with f0 in [{f0_lo_hz * 1e3:.4f}, "
        f"{f0_hi_hz * 1e3:.4f}] mHz ({int(np.sum(frequencies_hz[in_range] >= 1e-2))} above 10 mHz)."
    )
    setup_state_for_injection(
        curr, state, "GB", "gb", spread=0.0, subset_inds=subset_inds, priors=priors
    )

    _, gb_pe_moves = build_gb_moves(engine_info, curr, acs, priors, state)
    recipe.add_recipe_component(
        BudgetPERecipeStep(moves=gb_pe_moves, stop_iterations=stop_iterations), name="gb_pe"
    )


def get_gb_settings(general_set: GeneralSetup, response: str) -> tuple[GBSetup, SourceMetadata]:
    waveform_model = "GBGPU"
    waveform_model_code_link = "https://github.com/Erebor-L2D/GBGPU/tree/cd1l-run0"
    prior_model_code_link = "https://priors-database-f0027f.gitlab.io/mojito_light_1a.html#massive-black-hole-binaries-mbhb"

    delta_safe = 1e-9
    A_lims = [10 ** (-23.2), 1e-20]
    # * f0 and fdot limits are reset from the band edges by GBSetup.
    f0_lims = [1e-4, 0.023]
    m_chirp_lims = [0.03, 1.34]
    fdot_lims = [get_fdot_mojito(f0_lims[-1], sign="-"), get_fdot_mojito(f0_lims[-1], sign="+")]
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    alpha_lims = [0.0, 2 * np.pi]
    delta_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    domain_settings = general_set.domain_settings
    start_freq = float(domain_settings.min_freq)
    end_freq = float(domain_settings.max_freq)
    Tobs = general_set.Tobs

    oversample = 4
    extra_buffer = 5
    start_freq_ind = int(np.round(start_freq * Tobs))

    initialize_kwargs = dict(
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits,
        t0=general_set.data_t0,
        force_backend=general_set.gpu_backend,
    )

    # * Geometric ladder of the source file; the hottest chain samples nearly the prior.
    betas = 1 / 1.2 ** np.arange(general_set.ntemps)
    betas[-1] = 0.0001

    search_kwargs = dict(
        nwalkers=32,
        ntemps=24,
        shutoff_band_iteration=2,
        shutoff_frequency_threshold=None,
        burn_1=700,
        nsteps_1=300,
        snr_threshold=8.0,
        burn_2=500,
        nsteps_2=500,
        refit_start_iteration=5,
    )
    stretch_probability = 0.0

    waveform_kwargs = dict(
        dt=general_set.dt,
        T=Tobs,
        use_c_implementation=True,
        start_freq_ind=start_freq_ind,
        tdi_channel_setup="XYZ",
        tdi2=True,
        oversample=oversample,
        window=general_set.window_type,
        window_alpha=general_set.window_alpha,
    )

    # * The two runs differ only in "response"; DFFR ignores the TDI-on-the-fly knobs and ran with
    # * first-order delays, which is no longer its default.
    gb_stft_kwargs = dict(
        n_side_bins=N_SIDE_BINS,
        freq_from_tdi_phase=True,
        use_midpoint=True,
        linear_envelope=True,
        response=response,
        first_order_delays=True,
    )

    debug_settings = GBDebugSettings(
        enabled=False,
        plot_walker=0,
        plot_band=None,
        seq_pick="first",
        rj_trace=False,
        prop_timing_sync=False,
    )

    gb_settings = GBSettings(
        Tobs=Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs,
        nleaves_max=NLEAVES_MAX,
        nleaves_min=0,
        ndim=8,
        betas=betas,
        log_dir=general_set.file_store_dir,
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
        search_kwargs=search_kwargs,
        stretch_probability=stretch_probability,
        start_freq_ind=start_freq_ind,
        num_repeat_proposals=50,
        num_band_preload=NUM_BAND_PRELOAD,
        num_bands_preload_temp=NUM_BANDS_PRELOAD_TEMP,
        t0=general_set.data_t0,
        waveform_kwargs=waveform_kwargs,
        domain_settings=domain_settings,
        gb_wdm_comp=None,
        gb_stft_comp=STFTGBComputations,
        gb_stft_kwargs=gb_stft_kwargs,
        stft_band_alpha=general_set.window_alpha,
        debug_settings=debug_settings,
        leaf_cap_settings=None,
        sighet_settings=None,
    )
    gb_setup = GBSetup(gb_settings)

    band_edges = gb_setup.band_edges.copy()
    frequency_ranges = list(zip(band_edges[:-1], band_edges[1:]))
    metadata_initialize_kwargs = initialize_kwargs.copy()
    metadata_initialize_kwargs.pop("orbits", None)
    # ? CPU backend in the metadata, which is serialised.
    metadata_initialize_kwargs["force_backend"] = "cpu"
    gb_metadata = SourceMetadata(
        source_type="gb",
        frequency_ranges=frequency_ranges,
        waveform_model=waveform_model,
        waveform_model_code_link=waveform_model_code_link,
        prior_model_code_link=prior_model_code_link,
        waveform_model_config=dict(
            init_kwargs=metadata_initialize_kwargs, runtime_kwargs=waveform_kwargs
        ),
    )
    return gb_setup, gb_metadata


def get_general_settings(name: str, num_iterations: int) -> GeneralSetup:
    runs_dir = os.environ.get("GLOBAL_FIT_RUNS_DIR")
    if not runs_dir:
        raise ValueError("set GLOBAL_FIT_RUNS_DIR to the folder that receives one folder per run.")
    data_input_path = os.environ.get("MOJITO_FOLDER", "/project/lisa_nlddpc/Data/mojito_light")
    # ? The setup classes join paths by string concatenation, so the folder ends in a slash.
    file_store_dir = os.path.join(os.path.abspath(runs_dir), name) + "/"

    dt = 5.0
    Tobs = N_SEGMENTS * STFT_SEGMENT_S
    # * Half a bin under START_BIN: STFTSettings rounds min_freq up to the grid, onto START_BIN.
    start_freq = (START_BIN - 0.5) / STFT_SEGMENT_S
    end_freq = END_FREQ_HZ

    gpus = [0] if gpu_available else None
    if gpu_available:
        cp.cuda.runtime.setDevice(gpus[0])
        # ! Restrict JAX to the same GPU before its backend starts.
        import jax

        jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))
    # ! The shared environment builds CUDA 13 backends only.
    backend = "cuda13x" if gpus is not None else "cpu"

    domain_settings = STFTSettings.make_factory(
        big_dt=STFT_SEGMENT_S, min_freq=start_freq, max_freq=end_freq
    )

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["gb", "noise"],
        source_ids=dict(),
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=backend, frame="icrs", armlength=2493162305.42235),
    )
    downsample_kwargs = {"target_fs": 1 / dt, "window": ("kaiser", 31.0)}
    highpass_kwargs = {"cutoff": 1e-5, "order": 2, "zero_phase": True}
    lowpass_kwargs = {"cutoff": 1e-1, "order": 2, "zero_phase": True}
    trim_kwargs = {"duration": 200 * 3600, "is_percent": False, "trimming_type": "from_each_end"}
    # * After the trims, keep the first Tobs: the whole segments, so no partial day is left over.
    preprocess_kwargs = dict(
        highpass_kwargs=highpass_kwargs,
        lowpass_kwargs=lowpass_kwargs,
        trim_kwargs=trim_kwargs,
        downsample_kwargs=downsample_kwargs,
        Tobs=Tobs,
    )
    sensitivity_init_kwargs = dict(
        tdi_generation=2, mask_percentage=0.02, average_transfer_functions=True
    )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=name,
        domain_settings=domain_settings,
        random_seed=5701424,
        backup_iter=5,
        nwalkers=NWALKERS,
        ntemps=NTEMPS,
        num_iterations=num_iterations,
        window_type="tukey",
        window_taper_duration=WINDOW_TAPER_S,
        gpus=gpus,
        gpu_backend=backend,
        data_processor_class=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=True,
        sensitivity_backend_class=XYZSensitivityBackend,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        global_fit_codename="erebor",
        global_fit_version=name,
        global_fit_contact="ereborl2d@googlegroups.com",
        global_fit_code_link="https://github.com/Erebor-L2D",
        input_data_link=data_input_path,
        input_reference="mojito light",
        noise_model="parametric",
        noise_model_code_link="https://github.com/Erebor-L2D",
        submission_parent_folder=None,
        comment="High-frequency GB global fit, one run per STFT response (DFFR, TDI on the fly)",
    )
    return GeneralSetup(general_settings)


def build_settings(response: str, smoke_test: bool) -> CurrentInfoGlobalFit:
    if response not in RESPONSES:
        raise ValueError(f"response must be one of {RESPONSES}, got {response!r}.")
    name = run_name(response, smoke_test)
    general_setup = get_general_settings(name, SMOKE_ITERATIONS if smoke_test else MAX_ITERATIONS)
    shutil.copy(__file__, general_setup.file_store_dir + name + "_" + os.path.basename(__file__))

    gb_setup, gb_metadata = get_gb_settings(general_setup, response)
    global_settings = GlobalFitSettings(
        source_info={"gb": gb_setup},
        general_info=general_setup,
        rank_info=RankInfo(head_rank=1, main_rank=0),
        setup_function=functools.partial(
            setup_recipe, stop_iterations=SMOKE_ITERATIONS if smoke_test else None
        ),
        source_metadata={"gb": gb_metadata},
    )
    return CurrentInfoGlobalFit(global_settings)


def get_tdi_on_the_fly_settings():
    return build_settings("tdi_on_the_fly", smoke_test=False)


def get_dffr_settings():
    return build_settings("dffr", smoke_test=False)


def get_tdi_on_the_fly_smoke_settings():
    return build_settings("tdi_on_the_fly", smoke_test=True)


def get_dffr_smoke_settings():
    return build_settings("dffr", smoke_test=True)
