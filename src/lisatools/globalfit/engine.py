from __future__ import annotations

import dataclasses
import logging
import os
from collections import namedtuple
from typing import Optional

import h5py
import numpy as np
from eryn.ensemble import EnsembleSampler

__all__ = ["GlobalFitInfo", "GlobalFitEngine"]


import typing

from eryn.backends import Backend as eryn_Backend
from eryn.prior import ProbDistContainer
from eryn.state import State as eryn_State
from eryn.utils.transform import TransformContainer

from lisatools.detector import EqualArmlengthOrbits, Orbits

from ..analysiscontainer import AnalysisContainerArray
from ..detector import LISAModel, sangria
from ..domains import TDSettings
from ..sensitivity import (
    AE1SensitivityMatrix,
    AE2SensitivityMatrix,
    AET2SensitivityMatrix,
    XYZ1SensitivityMatrix,
    XYZ2SensitivityMatrix,
    XYZSensitivityBackend,
)
from ..utils.utility import AET, detrend, windowfun
from .preprocessing import BaseProcessingStep


@dataclasses.dataclass
class RankInfo:
    head_rank: int = -1
    main_rank: int = -1


class Setup:

    def __init__(self, settings_holder: Settings):
        self._settings_class = type(settings_holder)
        self.settings = settings_holder

        self._settings_names = [field.name for field in dataclasses.fields(self._settings_class)]
        self._settings_args_names = [
            field.name
            for field in dataclasses.fields(self._settings_class)
            if (
                field.default == dataclasses.MISSING
                and field.default_factory == dataclasses.MISSING
            )
        ]
        self._settings_kwargs_names = [
            field.name
            for field in dataclasses.fields(self._settings_class)
            if (
                field.default != dataclasses.MISSING or field.default_factory != dataclasses.MISSING
            )
        ]
        _args = tuple([getattr(settings_holder, key) for key in self._settings_args_names])
        _kwargs = {key: getattr(settings_holder, key) for key in self._settings_kwargs_names}
        self._settings_class.__init__(self, *_args, **_kwargs)
        self.init_df()

    @property
    def settings(self) -> Settings:
        return self._settings

    @settings.setter
    def settings(self, settings: Settings):
        assert isinstance(settings, Settings)
        self._settings = settings

    def init_df(self):
        self.Tobs = int(self.Tobs / self.dt) * self.dt
        self.df = 1.0 / self.Tobs


@dataclasses.dataclass
class Settings:
    Tobs: float | None = None
    dt: float | None = None
    initialize_kwargs: dict | None = None
    transform: Optional[TransformContainer] = None
    priors: Optional[typing.Dict[str, ProbDistContainer]] = None
    periodic: Optional[dict] = None
    nleaves_max: Optional[int] = None
    nleaves_min: Optional[int] = None
    ndim: Optional[int] = None
    betas: Optional[np.ndarray] = None
    other_tempering_kwargs: Optional[dict] = None
    branch_state: Optional[eryn_State] = None
    branch_backend: Optional[eryn_Backend] = None
    log_dir: Optional[str] = None


@dataclasses.dataclass
class GeneralSettings(Settings):
    Tobs: float | None = None
    dt: float | None = None
    file_store_dir: str | None = None
    base_file_name: str | None = None
    main_file_key: Optional[str] = "parameter_estimation_main"
    past_file_for_start: Optional[str] = None
    orbits: Orbits | None = None
    gpu_orbits: Orbits | None = None
    basis_domain: str = "stft"
    start_freq: float | None = None
    end_freq: float | None = None
    stft_dt: float | None = None
    random_seed: int | None = None
    backup_iter: int | None = None
    nwalkers: int | None = None
    ntemps: int | None = None
    window_type: str = "tukey"
    window_taper_duration: float | None = None
    gpu_backend: str = "cuda12x"
    gpus: typing.List[int] | None = None
    fixed_psd_kwargs: typing.Dict[str, typing.Any] | None = None
    # channels: typing.List[str] = dataclasses.field(default_factory=lambda: ["A", "E"])
    # noise_model: Optional[LISAModel] = None
    data_processor: Optional[BaseProcessingStep] = None
    processor_init_kwargs: Optional[dict] = None
    preprocess_kwargs: Optional[dict] = None
    sensitivity_init_kwargs: Optional[dict] = None
    normalize_window: bool = False
    catalogue: typing.Optional[dict] = None
    # ---- Galactic foreground geometry (fixed, not inferred) ----
    # If None, the galactic foreground is disabled in the likelihood.
    # Keys: R_d [kpc], z_d [kpc], alpha0 [rad], beta0 [rad],
    #       N_lambda (int, default 90), N_beta (int, default 60)
    galactic_grid_kwargs: Optional[typing.Dict[str, typing.Any]] = None

    # --- run metadata (propagated to RunMetadata.from_curr) ---
    global_fit_codename: Optional[str] = None
    global_fit_version: Optional[str] = None
    global_fit_contact: Optional[str] = None
    global_fit_code_link: Optional[str] = None
    submission_parent_folder: Optional[str] = None
    input_data_link: Optional[str] = None
    input_reference: Optional[str] = None
    noise_model: Optional[str] = None
    noise_model_code_link: Optional[str] = None
    run_waveform_model: Optional[str] = None
    run_waveform_model_code_link: Optional[str] = None
    comment: Optional[str] = None


from .loginfo import init_logger


class GeneralSetup(Setup, GeneralSettings):
    def __init__(self, general_settings: GeneralSettings):

        Setup.__init__(self, general_settings)

        level = logging.DEBUG
        name = "GeneralSetup"
        if not os.path.exists(self.artifacts_file_dir):
            os.makedirs(self.artifacts_file_dir)
        self.logger = init_logger(
            filename="general_setup.log", level=level, name=name, log_dir=self.artifacts_file_dir
        )

        self.init_setup()

    @property
    def main_file_path(self) -> str:
        return self.file_store_dir + self.base_file_name + "_" + self.main_file_key + ".h5"

    @property
    def artifacts_file_dir(self) -> str:
        return self.file_store_dir + self.base_file_name + "_artifacts/"

    @property
    def data_t0(self) -> float:
        return self.data_td_settings.t0

    @property
    def data_dt(self) -> float:
        return self.data_td_settings.dt

    def init_setup(self):
        if self.file_store_dir is None:
            raise ValueError("Must provide file_store_dir settings for GeneralSetup.")
        if self.base_file_name is None:
            raise ValueError("Must provide base_file_name settings for GeneralSetup.")

        self.force_backend = self.gpu_backend if self.gpus is not None else "cpu"
        self.logger.debug(f"Saving h5 backend to {self.main_file_path}")
        self.logger.debug(f"Saving artifacts to {self.artifacts_file_dir}")
        if not os.path.exists(self.artifacts_file_dir):
            os.makedirs(self.artifacts_file_dir)
            self.logger.debug(f"Created artifacts directory")

        self.init_data_information()

    def init_orbit_information(self):
        if self.orbits is None:
            self.orbits = EqualArmlengthOrbits()
            self.gpu_orbits = EqualArmlengthOrbits(force_backend=self.gpu_backend)
        else:
            if self.gpu_orbits is None and self.force_backend == self.gpu_backend:
                # TODO: make better
                raise ValueError("If adding orbits, make sure to duplicate into GPU orbits.")

    def init_data_information(self):

        if self.data_processor is None:
            raise ValueError("Must provide data_processor for GeneralSetup.")

        data_processor = self.data_processor(**(self.processor_init_kwargs or {}))

        if self.fixed_psd_kwargs is None:
            self.fixed_psd_kwargs = dict(
                psd_params=[15e-12, 3e-15],
                galfor_params=None,
            )

        self.logger.info(f"Using fixed PSD kwargs: {self.fixed_psd_kwargs}")

        default_preprocess_kwargs = dict(
            plot_folder=self.artifacts_file_dir,
            highpass_kwargs=dict(cutoff=2e-5, order=2, zero_phase=True),
            trim_kwargs=dict(duration=200 * 3600, is_percent=False, trimming_type="from_each_end"),
            Tobs=self.Tobs,
        )

        if self.preprocess_kwargs is None:
            self.preprocess_kwargs = default_preprocess_kwargs
        else:
            self.preprocess_kwargs = {**default_preprocess_kwargs, **self.preprocess_kwargs}

        for key, value in self.preprocess_kwargs.items():
            self.logger.debug(f"Preprocess setting: {key} = {value}")

        times, _ = data_processor.process(**self.preprocess_kwargs)
        dt = data_processor.td_signal.settings.dt
        Nt = len(times)
        self.data_td_settings = TDSettings(
            *data_processor.td_signal.settings.args, force_backend=self.force_backend
        )
        self.Tobs = Nt * dt
        self.catalogue = getattr(data_processor, "catalogue", {})

        if self.basis_domain == "stft":
            from ..domains import get_stft_settings

            if self.stft_dt is None:
                raise ValueError("Must provide `stft_dt` for stft basis domain.")
            self.basis_kwargs = dict(
                big_dt=self.stft_dt, min_freq=self.start_freq, max_freq=self.end_freq
            )
            domain_settings = get_stft_settings(
                times=times,
                **self.basis_kwargs,
                force_backend=self.force_backend,
            )
            nperseg = domain_settings.get_nperseg(dt)

            self.window_alpha = self.window_taper_duration / (nperseg * dt)
            window, _ = windowfun(self.window_type, nperseg, alpha=self.window_alpha)

            plot_kwargs_list = [
                dict(
                    channel=0, plot_type="stft", filename=self.artifacts_file_dir + "stft_data.png"
                ),
                dict(
                    channel=0,
                    plot_type="fd",
                    time_bin=0,
                    filename=self.artifacts_file_dir + "fd_data.png",
                ),
                dict(
                    channel=0,
                    plot_type="td",
                    freq_bin=0,
                    filename=self.artifacts_file_dir + "td_data.png",
                ),
            ]

        elif self.basis_domain == "fd":
            from ..domains import FDSettings

            df = 1.0 / (Nt * dt)
            Nf = Nt // 2 + 1

            self.window_alpha = self.window_taper_duration / (Nt * dt)
            self.basis_kwargs = dict(N=Nf, df=df, min_freq=self.start_freq, max_freq=self.end_freq)
            domain_settings = FDSettings(**self.basis_kwargs, force_backend=self.force_backend)

            self.window_alpha = self.window_taper_duration / (Nt * dt)
            window, _ = windowfun(self.window_type, Nt, alpha=self.window_alpha)
            plot_kwargs_list = [dict(channel=0, filename=self.artifacts_file_dir + "fd_data.png")]

        else:
            raise NotImplementedError(f"Basis domain {self.basis_domain} not implemented.")

        # window_factor = np.sqrt(np.sum(window**2) / len(window)) if normalize_window else 1.0
        # self.logger.debug(f"Window factor for normalization: {window_factor}")

        self.logger.debug(f"Applying window {self.window_type} with alpha: {self.window_alpha}")
        self.input_data_residual_array, orbits = data_processor.pour(
            settings=domain_settings, window=window, return_orbits=True
        )

        if self.basis_domain == "fd":  # TODO check if this is also necessary for STFT or TD
            self.input_data_residual_array.data_length = len(
                domain_settings.f_arr
            )  #! use acs.data_shape[0]
            self.input_data_residual_array._store_time_and_frequency_information(
                df=domain_settings.df, f_arr=domain_settings.f_arr
            )

        for plot_kwargs_here in plot_kwargs_list:
            _ = self.input_data_residual_array.data_res_arr.plot(**plot_kwargs_here)

        for key, value in domain_settings.__dict__.items():
            self.logger.info(f"Domain setting: {key} = {value}")

        if orbits is not None:
            self.orbits = orbits
            orbits_kwargs = orbits.kwargs

            if self.force_backend == self.gpu_backend:
                orbits_kwargs["force_backend"] = self.gpu_backend
                self.logger.debug(f"Initializing GPU orbits with kwargs: {orbits_kwargs}")

                self.gpu_orbits = data_processor.orbits_class(*orbits.args, **orbits_kwargs)

        self.init_orbit_information()

        # ---- Sensitivity backend ----
        self.sensitivity_backend = XYZSensitivityBackend(
            orbits=self.gpu_orbits,
            settings=domain_settings,
            force_backend=self.force_backend,
            window_values=window if self.normalize_window else None,
            **self.sensitivity_init_kwargs,
        )

        # ---- Galactic grid — initialized once, never recomputed during inference ----
        # Only the spectral envelope params (Amp, alpha, f_1, f_knee, f_2)
        # are inferred; the sky geometry (R_d, z_d, alpha0, beta0) is fixed here.
        if self.galactic_grid_kwargs is not None:
            self._init_galactic_grid(domain_settings)

    def _init_galactic_grid(self, domain_settings):
        """
        Compute the fixed galactic sky geometry and attach it to the sensitivity backend.

        Called once during setup.  After this call, sensitivity_backend.pycpp_sensitivity_matrix
        has gal_R_avg wired in and will include the galactic foreground in every likelihood
        evaluation automatically, scaled by the per-walker spectral parameters passed via
        Amp_all, alpha_all, f_1_all, f_knee_all, f_2_all.

        Args:
            domain_settings: Domain settings object (STFTSettings or FDSettings),
                             used to extract the segment centre times array.
        """
        gkw = self.galactic_grid_kwargs

        # Validate required keys
        for key in ("R_d", "z_d", "alpha0", "beta0"):
            if key not in gkw:
                raise ValueError(
                    f"galactic_grid_kwargs must contain '{key}'. " f"Got keys: {list(gkw.keys())}"
                )

        self.logger.info(
            f"Initializing galactic grid: R_d={gkw['R_d']} kpc, z_d={gkw['z_d']} kpc, "
            f"alpha0={gkw['alpha0']:.4f} rad, beta0={gkw['beta0']:.4f} rad"
        )

        # Build host-side quadrature geometry
        setup = self.sensitivity_backend.backend.GalacticGridSetup()
        setup.compute(
            N_lambda=gkw.get("N_lambda", 90),
            N_beta=gkw.get("N_beta", 60),
        )

        self.logger.info(f"Galactic sky grid: N_sky={setup.N_sky}, N_quad={setup.N_quad}")

        # Get segment centre times from domain settings.
        # For STFT: domain_settings.t_arr contains the segment centres.
        # For FD: there is only one time bin; pass a length-1 array at t=0.
        if hasattr(domain_settings, "t_arr"):
            t_arr = domain_settings.t_arr
        else:
            t_arr = np.array([0.0])
            self.logger.warning(
                "FD domain detected — using t=0 for galactic sky average. "
                "This is correct only for stationary (non-cyclostationary) analyses."
            )

        # xp is cupy on GPU, numpy on CPU
        xp = self.sensitivity_backend.xp

        # Initialize the galactic grid on the sensitivity backend.
        # This calls GalacticGrid::initialize(R_d, z_d, times, N_times) on the C++ side,
        # which computes and stores weights and R_avg — then wires R_avg into the
        # C++ XYZSensitivityMatrix so get_noise_covariance can access it.
        self.sensitivity_backend.initialize_galactic_grid(
            times=xp.asarray(t_arr),
            R_d=float(gkw["R_d"]),
            z_d=float(gkw["z_d"]),
            R_vals_quad=xp.asarray(setup.R_vals_quad),
            z_vals_quad=xp.asarray(setup.z_vals_quad),
            quad_weights=xp.asarray(setup.quad_weights),
            cos_beta_ecl=xp.asarray(setup.cos_beta_ecl),
            lam_ecl=xp.asarray(setup.lam_ecl),
            beta_ecl=xp.asarray(setup.beta_ecl),
            N_quad=setup.N_quad,
            N_sky=setup.N_sky,
            alpha0=float(gkw["alpha0"]),
            beta0=float(gkw["beta0"]),
        )

        self.logger.info("Galactic grid initialized and attached to sensitivity backend.")


@dataclasses.dataclass
class GlobalFitSettings:
    source_info: typing.Dict[str, Setup]
    general_info: GeneralSetup
    rank_info: RankInfo
    setup_function: typing.Callable[(...), None]
    source_metadata: typing.Dict[str, dataclasses.dataclass] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class EngineInfo:
    branch_names: typing.List[str]
    ndims: typing.Dict[str, int]
    nleaves_max: typing.Dict[str, int]
    nleaves_min: typing.Dict[str, int]
    branch_states: typing.Dict[str, eryn_State] = None
    branch_backends: typing.Dict[str, eryn_Backend] = None


GlobalFitInfo = namedtuple(
    "GlobalFitInfo",
    (
        "analysis_container_arr",
        "map_fn",
        "random",
    ),
)


class GlobalFitEngine(EnsembleSampler):

    def __init__(self, analysis_container_arr: AnalysisContainerArray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_container_arr = analysis_container_arr

    def get_model(self):
        """Get ``Model`` object from sampler.

        Returns:
            :class:`GlobalFitInfo`: model object used by the sampler.
        """
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        model = GlobalFitInfo(
            self.analysis_container_arr,
            map_fn,
            self._random,
        )
        return model
